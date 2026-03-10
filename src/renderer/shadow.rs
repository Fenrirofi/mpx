use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// GPU uniform for shadow pass
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowUniform {
    pub light_view_proj: [[f32; 4]; 4],
    /// x=bias, y=normal_bias, z=pcf_radius, w=enabled
    pub shadow_params: [f32; 4],
}

impl ShadowUniform {
    pub fn new(light_view_proj: Mat4, enabled: bool) -> Self {
        Self {
            light_view_proj: light_view_proj.to_cols_array_2d(),
            shadow_params: [0.0005, 0.002, 2.0, if enabled { 1.0 } else { 0.0 }],
        }
    }
}

pub struct ShadowRenderer {
    pub shadow_texture: wgpu::Texture,
    pub shadow_view:    wgpu::TextureView,
    pub sampler:        wgpu::Sampler,
    pub pipeline:       wgpu::RenderPipeline,
    pub uniform_buf:    wgpu::Buffer,
    pub uniform_bgl:    wgpu::BindGroupLayout,
    pub uniform_bg:     wgpu::BindGroup,
    pub object_bgl:     wgpu::BindGroupLayout,
}

impl ShadowRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Shadow map texture ────────────────────────────────────────────────
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("shadow_map"),
            size:            wgpu::Extent3d {
                width:                SHADOW_MAP_SIZE,
                height:               SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          SHADOW_DEPTH_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                           | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });

        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::MipmapFilterMode::Nearest,
            compare:        Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_uniform_bgl"),
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
            label: Some("shadow_object_bgl"),
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

        // ── Uniform buffer ────────────────────────────────────────────────────
        let identity = ShadowUniform::new(Mat4::IDENTITY, false);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("shadow_uniform_buf"),
            contents: bytemuck::bytes_of(&identity),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("shadow_uniform_bg"),
            layout:  &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("shadow_pipeline_layout"),
            bind_group_layouts: &[&uniform_bgl, &object_bgl],
            immediate_size:     0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("shadow_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_shadow"),
                // Only position needed
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::assets::Vertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &[wgpu::VertexAttribute {
                        offset:           0,
                        shader_location:  0,
                        format:           wgpu::VertexFormat::Float32x3,
                    }],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::TriangleList,
                cull_mode:          Some(wgpu::Face::Front), // Peter-panning fix
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              SHADOW_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        Self {
            shadow_texture,
            shadow_view,
            sampler,
            pipeline,
            uniform_buf,
            uniform_bgl,
            uniform_bg,
            object_bgl,
        }
    }

    /// Compute orthographic light-space matrix for a directional light.
    pub fn compute_light_matrix(light_dir: glam::Vec3, scene_center: glam::Vec3, scene_radius: f32) -> Mat4 {
        let light_pos = scene_center - light_dir * scene_radius * 2.0;
        let view = Mat4::look_at_rh(light_pos, scene_center, glam::Vec3::Y);
        let proj = Mat4::orthographic_rh(
            -scene_radius, scene_radius,
            -scene_radius, scene_radius,
            0.1, scene_radius * 4.0,
        );
        proj * view
    }

    /// Update the shadow uniform with the current light matrix.
    pub fn update(&self, queue: &wgpu::Queue, light_view_proj: Mat4) {
        let uniform = ShadowUniform::new(light_view_proj, true);
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
    }
}