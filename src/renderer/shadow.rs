// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Cascaded Shadow Maps (CSM) + PCSS                                          ║
// ║  4 kaskady, każda 2048×2048 Depth32Float                                   ║
// ║                                                                              ║
// ║  Zmiany vs poprzednia wersja:                                               ║
// ║  [PCSS-1] CsmPbrUniform — dodane pole shadow_ext (light_size, max_radius,  ║
// ║           quality, _pad). shadow_params.w nadal = cascade_bias_multiplier. ║
// ║  [PCSS-2] ShadowSettings — publiczna struct do sterowania z UI.            ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub const NUM_CASCADES:    usize = 4;
pub const SHADOW_MAP_SIZE: u32   = 2048;
pub const SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const CASCADE_LAMBDA: f32 = 0.75;

// ─────────────────────────────────────────────────────────────────────────────
// [PCSS-2] Ustawienia cieni — używane przez UI i przekazywane do update()
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy)]
pub struct ShadowSettings {
    /// Kątowy rozmiar źródła światła w przestrzeni world (przybliżenie).
    /// Słońce ≈ 1.5, duże okno ≈ 5.0, mała latarnia ≈ 0.3.
    /// Większa wartość → szerszy penumbra (miększe cienie).
    pub light_size: f32,
    /// Maksymalny promień kernela PCF w texelach.
    /// Zapobiega eksplodowaniu cieni przy bardzo dużym light_size.
    pub max_radius_texels: f32,
    /// Jakość cieni:
    ///   0.0 = PCF 3×3 (stały kernel, najszybszy — fallback)
    ///   1.0 = PCSS 16 próbek (dobry do real-time)
    ///   2.0 = PCSS 32 próbki (screenshot quality)
    pub quality: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            light_size:        1.5,
            max_radius_texels: 8.0,
            quality:           1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU structs
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CsmUniform {
    pub light_view_proj: [[[f32; 4]; 4]; 4],
    pub cascade_index:   u32,
    pub _pad:            [u32; 3],
}

/// Dane wysyłane do PBR shadera (group 3 binding 1).
///
/// Układ pamięci (każde pole = vec4 = 16 B, łącznie 288 B):
///   offset   0..256 — light_view_proj [4× mat4]
///   offset 256..272 — cascade_splits
///   offset 272..288 — shadow_params  (x=min_bias, y=slope_bias, z=texel_size, w=cascade_bias_mult)
///   offset 288..304 — shadow_ext     (x=light_size, y=max_radius_texels, z=quality, w=_pad)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CsmPbrUniform {
    pub light_view_proj: [[[f32; 4]; 4]; 4],
    /// Granice kaskad w view-space Z (addytywne — każda to delta od poprzedniej).
    pub cascade_splits:  [f32; 4],
    /// x=min_bias  y=slope_bias  z=texel_size  w=cascade_bias_multiplier
    /// UWAGA: shadow_params.w = cascade_bias_multiplier — NIE usuwać!
    pub shadow_params:   [f32; 4],
    // [PCSS-1] Nowe pole — rozszerzenie PCSS.
    /// x=light_size_world  y=max_radius_texels  z=quality (0/1/2)  w=_pad
    pub shadow_ext:      [f32; 4],
}

// ─────────────────────────────────────────────────────────────────────────────
// CsmRenderer
// ─────────────────────────────────────────────────────────────────────────────

pub struct CsmRenderer {
    pub shadow_texture:    wgpu::Texture,
    pub shadow_array_view: wgpu::TextureView,
    cascade_views:         [wgpu::TextureView; NUM_CASCADES],
    pub sampler:           wgpu::Sampler,
    pub pipeline:          wgpu::RenderPipeline,
    pub csm_uniform_buf:   wgpu::Buffer,
    pub csm_uniform_bgl:   wgpu::BindGroupLayout,
    pub csm_uniform_bg:    wgpu::BindGroup,
    pub pbr_uniform_buf:   wgpu::Buffer,
    pub object_bgl:        wgpu::BindGroupLayout,
    pub last_pbr_uniform:  CsmPbrUniform,
    /// Aktualne ustawienia cieni — modyfikuj i wywołaj update() żeby zastosować.
    pub settings:          ShadowSettings,
}

impl CsmRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("csm_shadow_array"),
            size: wgpu::Extent3d {
                width:                 SHADOW_MAP_SIZE,
                height:                SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          SHADOW_DEPTH_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                           | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shadow_array_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label:             Some("csm_array_view"),
            dimension:         Some(wgpu::TextureViewDimension::D2Array),
            aspect:            wgpu::TextureAspect::DepthOnly,
            array_layer_count: Some(NUM_CASCADES as u32),
            ..Default::default()
        });

        let cascade_views = std::array::from_fn(|i| {
            shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label:             Some(&format!("csm_cascade_{i}")),
                dimension:         Some(wgpu::TextureViewDimension::D2),
                aspect:            wgpu::TextureAspect::DepthOnly,
                base_array_layer:  i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("csm_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::MipmapFilterMode::Nearest,
            compare:        Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let csm_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("csm_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("csm_object_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let zero_csm = CsmUniform {
            light_view_proj: [Mat4::IDENTITY.to_cols_array_2d(); 4],
            cascade_index: 0,
            _pad: [0; 3],
        };
        let csm_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("csm_uniform_buf"),
            contents: bytemuck::bytes_of(&zero_csm),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let settings = ShadowSettings::default();
        let zero_pbr = CsmPbrUniform {
            light_view_proj: [Mat4::IDENTITY.to_cols_array_2d(); 4],
            cascade_splits:  [10.0, 50.0, 150.0, 500.0],
            shadow_params:   [0.001, 0.003, 1.0 / SHADOW_MAP_SIZE as f32, 1.5],
            // [PCSS-1] Inicjalizacja shadow_ext z wartościami domyślnymi ShadowSettings.
            shadow_ext:      [settings.light_size, settings.max_radius_texels, settings.quality, 0.0],
        };
        let pbr_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("csm_pbr_uniform_buf"),
            contents: bytemuck::bytes_of(&zero_pbr),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let csm_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("csm_uniform_bg"),
            layout:  &csm_uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: csm_uniform_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("csm_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("csm_pipeline_layout"),
            bind_group_layouts: &[&csm_uniform_bgl, &object_bgl],
            immediate_size:     0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("csm_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::assets::Vertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &[wgpu::VertexAttribute {
                        offset: 0, shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              SHADOW_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment:    None,
            multiview_mask: None,
            cache: None,
        });

        Self {
            shadow_texture, shadow_array_view, cascade_views, sampler,
            pipeline, csm_uniform_buf, csm_uniform_bgl, csm_uniform_bg,
            pbr_uniform_buf, object_bgl,
            last_pbr_uniform: zero_pbr,
            settings,
        }
    }

    /// Lambda-blended cascade splits (Engel 2007)
    pub fn compute_cascade_splits(near: f32, far: f32) -> [f32; NUM_CASCADES] {
        let range = far - near;
        let ratio = far / near;
        std::array::from_fn(|i| {
            let p       = (i + 1) as f32 / NUM_CASCADES as f32;
            let log     = near * ratio.powf(p);
            let uniform = near + range * p;
            CASCADE_LAMBDA * (log - uniform) + uniform - near
        })
    }

    /// Oblicz macierze ortograficzne okalające każdą kaskadę frustum kamery.
    pub fn compute_cascade_matrices(
        light_dir: Vec3,
        cam_view:  Mat4,
        cam_proj:  Mat4,
        near:      f32,
        splits:    &[f32; NUM_CASCADES],
    ) -> [Mat4; NUM_CASCADES] {
        let inv_vp = (cam_proj * cam_view).inverse();
        let up = if light_dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let light_view = Mat4::look_at_rh(Vec3::ZERO, light_dir, up);

        let mut slice_near = near;
        std::array::from_fn(|i| {
            let slice_far = slice_near + splits[i];
            let corners   = frustum_slice_world(inv_vp, slice_near, slice_far, cam_proj);
            slice_near    = slice_far;

            let ls: Vec<Vec3> = corners.iter()
                .map(|c| (light_view * Vec4::from((*c, 1.0))).truncate())
                .collect();

            let mn = ls.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
            let mx = ls.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
            let z_slack = (mx.z - mn.z) * 0.2;

            let proj = Mat4::orthographic_rh(mn.x, mx.x, mn.y, mx.y,
                                             mn.z - z_slack, mx.z + z_slack);
            proj * light_view
        })
    }

    /// Aktualizuj bufor GPU danymi kaskad i aktualnymi `self.settings`.
    /// Wywołaj każdą klatkę (lub gdy zmienią się settings).
    pub fn update(&mut self, queue: &wgpu::Queue,
                  matrices: &[Mat4; NUM_CASCADES], splits: &[f32; NUM_CASCADES]) {
        let pbr = CsmPbrUniform {
            light_view_proj: std::array::from_fn(|i| matrices[i].to_cols_array_2d()),
            cascade_splits:  *splits,
            // shadow_params.w = 1.5 = cascade_bias_multiplier — NIE zmieniaj!
            shadow_params:   [0.001, 0.003, 1.0 / SHADOW_MAP_SIZE as f32, 1.5],
            // [PCSS-1] Przekaż aktualne ustawienia PCSS.
            shadow_ext: [
                self.settings.light_size,
                self.settings.max_radius_texels,
                self.settings.quality,
                0.0,
            ],
        };
        queue.write_buffer(&self.pbr_uniform_buf, 0, bytemuck::bytes_of(&pbr));
        self.last_pbr_uniform = pbr;

        let csm_u = CsmUniform {
            light_view_proj: std::array::from_fn(|i| matrices[i].to_cols_array_2d()),
            cascade_index: 0,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.csm_uniform_buf, 0, bytemuck::bytes_of(&csm_u));
    }

    pub fn set_cascade(&self, queue: &wgpu::Queue, index: usize) -> &wgpu::TextureView {
        let offset = std::mem::size_of::<[[[f32; 4]; 4]; 4]>() as u64;
        let idx = index as u32;
        queue.write_buffer(&self.csm_uniform_buf, offset, bytemuck::bytes_of(&idx));
        &self.cascade_views[index]
    }
}

fn frustum_slice_world(inv_vp: Mat4, near: f32, far: f32, proj: Mat4) -> [Vec3; 8] {
    let proj_near = -proj.w_axis.z / proj.z_axis.z;
    let proj_far  =  proj.w_axis.z / (1.0 - proj.z_axis.z);
    let total_range = proj_far - proj_near;

    let ndc_near = if total_range.abs() > 0.0001 {
        (near - proj_near) / total_range
    } else { 0.0 };
    let ndc_far = if total_range.abs() > 0.0001 {
        (far  - proj_near) / total_range
    } else { 1.0 };

    let ndc_pts = [
        Vec4::new(-1.0, -1.0, ndc_near, 1.0), Vec4::new(1.0, -1.0, ndc_near, 1.0),
        Vec4::new(-1.0,  1.0, ndc_near, 1.0), Vec4::new(1.0,  1.0, ndc_near, 1.0),
        Vec4::new(-1.0, -1.0, ndc_far,  1.0), Vec4::new(1.0, -1.0, ndc_far,  1.0),
        Vec4::new(-1.0,  1.0, ndc_far,  1.0), Vec4::new(1.0,  1.0, ndc_far,  1.0),
    ];
    std::array::from_fn(|i| {
        let w = inv_vp * ndc_pts[i];
        w.truncate() / w.w
    })
}