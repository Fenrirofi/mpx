use std::sync::Arc;
use anyhow::Result;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::scene::Scene;
use crate::scene::light::LightKind;
use super::{
    gpu_mesh::GpuMesh,
    ibl::{bake_ibl, IblMaps},
    pipeline,
    post::{PostParams, PostProcessor},
    shadow::{CsmRenderer, NUM_CASCADES},
    skybox::{SkyboxRenderer, SkyUniform},
    ssao::SsaoRenderer,
    ssr::SsrRenderer,
    taa::TaaRenderer,
    texture,
    uniforms::{CameraUniform, EnvUniform, LightArrayUniform, MaterialUniform, ObjectUniform},
};

struct GpuObject {
    mesh:                  GpuMesh,
    object_uniform_buf:    wgpu::Buffer,
    object_bind_group:     wgpu::BindGroup,
    _material_uniform_buf: wgpu::Buffer,
    material_bind_group:   wgpu::BindGroup,
    shadow_object_bg:      wgpu::BindGroup,
}

pub struct RenderContext {
    surface:    wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    config:     wgpu::SurfaceConfiguration,
    size:       PhysicalSize<u32>,

    _camera_bgl:       wgpu::BindGroupLayout,
    _object_bgl:       wgpu::BindGroupLayout,
    _material_bgl:     wgpu::BindGroupLayout,
    lights_shadow_bgl: wgpu::BindGroupLayout,

    pbr_pipeline:     wgpu::RenderPipeline,
    gbuffer_pipeline: wgpu::RenderPipeline,

    camera_uniform_buf: wgpu::Buffer,
    camera_bind_group:  wgpu::BindGroup,
    lights_uniform_buf: wgpu::Buffer,
    lights_shadow_bg:   wgpu::BindGroup,

    depth_tex:    wgpu::Texture,
    depth_view:   wgpu::TextureView,
    gbuffer_normals_tex:  wgpu::Texture,
    gbuffer_normals_view: wgpu::TextureView,
    gbuffer_mr_tex:  wgpu::Texture,
    gbuffer_mr_view: wgpu::TextureView,

    white_view:       wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    default_sampler:  wgpu::Sampler,

    fallback_cube_view: wgpu::TextureView,
    fallback_lut_view:  wgpu::TextureView,
    fallback_sampler:   wgpu::Sampler,

    pub shadow: CsmRenderer,
    pub skybox: SkyboxRenderer,
    pub post:   PostProcessor,
    pub ssao:   SsaoRenderer,
    pub taa:    TaaRenderer,
    pub ssr:    SsrRenderer,
    pub ibl:    Option<IblMaps>,

    gpu_objects: Vec<GpuObject>,

    pub sky_uniform:      SkyUniform,
    pub post_params:      PostParams,
    frame_time:           f32,
    pub env_rotation_yaw: f32,
    env_uniform_buf:      wgpu::Buffer,
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

fn create_gbuffer_tex(device: &wgpu::Device, w: u32, h: u32, label: &str)
    -> (wgpu::Texture, wgpu::TextureView)
{
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size:  wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format:    wgpu::TextureFormat::Rgba16Float,
        usage:     wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_gbuffer_pipeline(
    device:  &wgpu::Device,
    cam:     &wgpu::BindGroupLayout,
    obj:     &wgpu::BindGroupLayout,
    mat_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gbuffer"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gbuffer.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gbuffer_layout"),
        bind_group_layouts: &[cam, obj, mat_bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gbuffer_pipeline"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[crate::assets::Vertex::buffer_layout()],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back), ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(), bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fs_main"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        multiview_mask: None, cache: None,
    })
}

/// Proceduralny fallback cubemap — neutralne niebo zamiast czarnego.
/// Górna hemisfera (+Y): chłodny błękit overcast, dolna (-Y): ciemna ziemia.
/// Boczne ściany: gradient pionowy. Wartości w liniowym HDR (nie sRGB).
fn create_fallback_cubemap(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    const SZ: u32 = 4;

    fn f16(v: f32) -> u16 { half::f16::from_f32(v).to_bits() }
    fn px(r: f32, g: f32, b: f32) -> [u8; 8] {
        let h = [f16(r), f16(g), f16(b), f16(1.0)];
        let mut out = [0u8; 8];
        for (i, v) in h.iter().enumerate() { let b = v.to_le_bytes(); out[i*2]=b[0]; out[i*2+1]=b[1]; }
        out
    }
    fn lerp_px(a: (f32,f32,f32), b: (f32,f32,f32), t: f32) -> [u8; 8] {
        px(a.0*(1.0-t)+b.0*t, a.1*(1.0-t)+b.1*t, a.2*(1.0-t)+b.2*t)
    }

    let sky    = (0.15_f32, 0.22_f32, 0.40_f32);
    let ground = (0.04_f32, 0.04_f32, 0.04_f32);
    let horiz  = (0.10_f32, 0.13_f32, 0.20_f32);

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fallback_cube"),
        size:  wgpu::Extent3d { width: SZ, height: SZ, depth_or_array_layers: 6 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format:    wgpu::TextureFormat::Rgba16Float,
        usage:     wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for face in 0..6u32 {
        let mut data = Vec::with_capacity((SZ * SZ * 8) as usize);
        for row in 0..SZ {
            let t = row as f32 / (SZ - 1) as f32;
            let color = match face {
                2 => px(sky.0, sky.1, sky.2),
                3 => px(ground.0, ground.1, ground.2),
                _ => {
                    if t < 0.45 { lerp_px(ground, horiz, t / 0.45) }
                    else        { lerp_px(horiz,  sky,  (t - 0.45) / 0.55) }
                }
            };
            for _ in 0..SZ { data.extend_from_slice(&color); }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SZ * 8), rows_per_image: Some(SZ) },
            wgpu::Extent3d { width: SZ, height: SZ, depth_or_array_layers: 1 },
        );
    }

    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6), ..Default::default()
    });
    (tex, view)
}

fn create_fallback_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fallback_lut"),
        size:  wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format:    wgpu::TextureFormat::Rgba16Float,
        usage:     wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data: [u16; 4] = [0x3c00, 0, 0, 0x3c00];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    queue.write_texture(tex.as_image_copy(), &bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

// ─────────────────────────────────────────────────────────────────────────────
// impl RenderContext
// ─────────────────────────────────────────────────────────────────────────────

impl RenderContext {
    fn build_lights_shadow_bg(
        device:          &wgpu::Device,
        bgl:             &wgpu::BindGroupLayout,
        lights_buf:      &wgpu::Buffer,
        shadow:          &CsmRenderer,
        irradiance_view: &wgpu::TextureView,
        prefilter_view:  &wgpu::TextureView,
        lut_view:        &wgpu::TextureView,
        ibl_sampler:     &wgpu::Sampler,
        lut_sampler:     &wgpu::Sampler,
        env_uniform_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights_shadow_ibl_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shadow.pbr_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&shadow.shadow_array_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&shadow.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(irradiance_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(prefilter_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(ibl_sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(lut_sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: env_uniform_buf.as_entire_binding() },
            ],
        })
    }

    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false, compatible_surface: Some(&surface),
        }).await.map_err(|e| anyhow::anyhow!("Adapter: {e:?}"))?;

        log::info!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pbr_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }).await?;

        let caps = surface.get_capabilities(&adapter);
        // Wybierz non-sRGB — gamma korekcja jest robiona ręcznie w post.wgsl
        let format = caps.formats.iter().copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0], view_formats: vec![],
        };
        surface.configure(&device, &config);

        let camera_bgl        = pipeline::camera_bind_group_layout(&device);
        let object_bgl        = pipeline::object_bind_group_layout(&device);
        let material_bgl      = pipeline::material_bind_group_layout(&device);
        let lights_shadow_bgl = pipeline::lights_shadow_bind_group_layout(&device);

        let shadow = CsmRenderer::new(&device);
        let skybox = SkyboxRenderer::new(&device, wgpu::TextureFormat::Rgba16Float, &camera_bgl);
        let post   = PostProcessor::new(&device, &queue, config.width, config.height, format);
        let ssao   = SsaoRenderer::new(&device, &queue, config.width, config.height);
        let taa    = TaaRenderer::new(&device, config.width, config.height);
        let ssr    = SsrRenderer::new(&device, config.width, config.height);

        let pbr_pipeline     = pipeline::create_pbr_pipeline(&device, &camera_bgl, &object_bgl, &material_bgl, &lights_shadow_bgl);
        let gbuffer_pipeline = create_gbuffer_pipeline(&device, &camera_bgl, &object_bgl, &material_bgl);

        let camera_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buf"), size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"), layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buf.as_entire_binding() }],
        });
        let lights_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights_buf"), size: std::mem::size_of::<LightArrayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let (depth_tex, depth_view) = texture::create_depth_texture(&device, config.width, config.height);
        let (gn_tex, gn_view) = create_gbuffer_tex(&device, config.width, config.height, "gbuf_normals");
        let (gm_tex, gm_view) = create_gbuffer_tex(&device, config.width, config.height, "gbuf_mr");

        let (_, white_view, default_sampler) = texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white",       true);
        let (_, flat_normal_view, _)         = texture::create_fallback_texture(&device, &queue, [128,128,255,255], "flat_normal", false);

        let (_fb_cube_tex, fallback_cube_view) = create_fallback_cubemap(&device, &queue);
        let (_fb_lut_tex,  fallback_lut_view)  = create_fallback_lut(&device, &queue);
        let fallback_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fallback_ibl_sampler"),
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear, ..Default::default()
        });

        let env_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_uniform_buf"),
            contents: bytemuck::bytes_of(&EnvUniform::identity()),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lights_shadow_bg = Self::build_lights_shadow_bg(
            &device, &lights_shadow_bgl, &lights_uniform_buf, &shadow,
            &fallback_cube_view, &fallback_cube_view, &fallback_lut_view,
            &fallback_sampler, &fallback_sampler, &env_uniform_buf,
        );

        let mut ctx = Self {
            surface, device, queue, config, size,
            _camera_bgl: camera_bgl, _object_bgl: object_bgl,
            _material_bgl: material_bgl, lights_shadow_bgl,
            pbr_pipeline, gbuffer_pipeline,
            camera_uniform_buf, camera_bind_group,
            lights_uniform_buf, lights_shadow_bg,
            depth_tex, depth_view,
            gbuffer_normals_tex: gn_tex, gbuffer_normals_view: gn_view,
            gbuffer_mr_tex: gm_tex, gbuffer_mr_view: gm_view,
            white_view, flat_normal_view, default_sampler,
            fallback_cube_view, fallback_lut_view, fallback_sampler,
            shadow, skybox, post, ssao, taa, ssr, ibl: None,
            gpu_objects: Vec::new(),
            sky_uniform: SkyUniform::day(),
            post_params: PostParams::default(),
            frame_time: 0.0,
            env_rotation_yaw: 0.0,
            env_uniform_buf,
        };
        ctx.upload_scene_placeholder();
        Ok(ctx)
    }

    pub fn load_hdr(&mut self, path: &str) -> Result<()> {
        let maps = bake_ibl(&self.device, &self.queue, path)?;
        self.lights_shadow_bg = Self::build_lights_shadow_bg(
            &self.device, &self.lights_shadow_bgl, &self.lights_uniform_buf, &self.shadow,
            &maps.irradiance_view, &maps.prefilter_view, &maps.brdf_lut_view,
            &maps.sampler, &maps.lut_sampler, &self.env_uniform_buf,
        );
        self.ibl = Some(maps);
        log::info!("IBL loaded — PBR ambient/specular aktywny.");
        Ok(())
    }

    fn create_gpu_object(&self, mesh: &crate::assets::Mesh, material: &crate::assets::Material) -> GpuObject {
        let gpu_mesh = GpuMesh::upload(&self.device, mesh);

        let object_uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("obj_buf"), size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let object_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("obj_bg"), layout: &self._object_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: object_uniform_buf.as_entire_binding() }],
        });
        let shadow_object_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_obj_bg"), layout: &self.shadow.object_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: object_uniform_buf.as_entire_binding() }],
        });

        let mat_u = MaterialUniform {
            base_color:            material.base_color.to_array(),
            metallic_roughness_ao: [material.metallic, material.roughness, material.ao_strength, material.normal_scale],
            emissive:              [material.emissive[0], material.emissive[1], material.emissive[2], 0.0],
        };
        let material_uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mat_buf"), contents: bytemuck::bytes_of(&mat_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mat_bg"), layout: &self._material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: material_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.white_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.default_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.flat_normal_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.white_view) },
            ],
        });

        GpuObject { mesh: gpu_mesh, object_uniform_buf, object_bind_group,
                    _material_uniform_buf: material_uniform_buf, material_bind_group, shadow_object_bg }
    }

    fn upload_scene_placeholder(&mut self) {
        let scene = crate::scene::Scene::default_scene();
        self.gpu_objects = scene.objects.iter().map(|o| self.create_gpu_object(&o.mesh, &o.material)).collect();
    }

    pub fn size(&self) -> PhysicalSize<u32> { self.size }

    /// Format powierzchni — potrzebny do inicjalizacji EguiRenderer.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size          = new_size;
        self.config.width  = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        let (dt, dv) = texture::create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_tex  = dt;
        self.depth_view = dv;
        let (gnt, gnv) = create_gbuffer_tex(&self.device, self.config.width, self.config.height, "gbuf_normals");
        let (gmt, gmv) = create_gbuffer_tex(&self.device, self.config.width, self.config.height, "gbuf_mr");
        self.gbuffer_normals_tex  = gnt; self.gbuffer_normals_view = gnv;
        self.gbuffer_mr_tex  = gmt;      self.gbuffer_mr_view      = gmv;
        self.post.resize(&self.device, &self.queue, self.config.width, self.config.height, self.config.format);
        self.ssao.resize(&self.device, &self.queue, self.config.width, self.config.height);
        self.taa.resize(&self.device, self.config.width, self.config.height);
        self.ssr.resize(&self.device, self.config.width, self.config.height);
    }

    /// Odbuduj GpuObject dla jednego obiektu sceny z nowym materiałem.
    /// Wywoływane gdy layer stack zmienia parametry materiału.
    pub fn rebuild_object(
        &mut self,
        index:    usize,
        mesh:    &crate::assets::Mesh,
        material: &crate::assets::Material,
    ) {
        if index >= self.gpu_objects.len() { return; }
        self.gpu_objects[index] = self.create_gpu_object(mesh, material);
    }

    // ── Współdzielona logika przygotowania klatki ─────────────────────────────

    fn prepare_frame(&mut self, scene: &Scene) -> (glam::Mat4, glam::Mat4) {
        self.frame_time += 1.0 / 60.0;

        let (jx, jy)        = self.taa.jitter(self.config.width, self.config.height);
        let unjittered_vp   = scene.camera.view_proj();
        let mut jitter_proj = scene.camera.proj();
        jitter_proj.w_axis.x += jx;
        jitter_proj.w_axis.y += jy;
        let jittered_vp = jitter_proj * scene.camera.view();

        let cam_u = CameraUniform {
            view_proj:  jittered_vp.to_cols_array_2d(),
            view:       scene.camera.view().to_cols_array_2d(),
            proj:       jitter_proj.to_cols_array_2d(),
            camera_pos: [scene.camera.position.x, scene.camera.position.y,
                         scene.camera.position.z, 1.0],
        };
        self.queue.write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&cam_u));
        self.queue.write_buffer(&self.lights_uniform_buf, 0,
            bytemuck::bytes_of(&LightArrayUniform::from_scene(scene)));
        self.queue.write_buffer(&self.env_uniform_buf, 0,
            bytemuck::bytes_of(&EnvUniform::from_yaw(self.env_rotation_yaw)));

        let sun_dir = scene.lights.iter()
            .find(|l| l.enabled && matches!(l.kind, LightKind::Directional))
            .map(|l| l.position_or_direction.normalize())
            .unwrap_or(Vec3::new(0.4, 0.8, 0.3).normalize());

        let (s, c) = self.env_rotation_yaw.sin_cos();
        let rotated_sun = Vec3::new(
            sun_dir.x * c + sun_dir.z * s,
            sun_dir.y,
            -sun_dir.x * s + sun_dir.z * c,
        );

        let near    = scene.camera.near;
        let far     = scene.camera.far;
        let splits  = CsmRenderer::compute_cascade_splits(near, far);
        let matrices= CsmRenderer::compute_cascade_matrices(
            rotated_sun, scene.camera.view(), scene.camera.proj(), near, &splits,
        );
        self.shadow.update(&self.queue, &matrices, &splits);

        self.sky_uniform.sun_direction = [rotated_sun.x, rotated_sun.y, rotated_sun.z, 0.0];
        self.skybox.update(&self.queue, &self.sky_uniform);

        for (i, obj) in scene.objects.iter().enumerate() {
            if i >= self.gpu_objects.len() { break; }
            let mut t = obj.transform.clone();
            let u = ObjectUniform {
                model:         t.matrix().to_cols_array_2d(),
                normal_matrix: t.normal_matrix().to_cols_array_2d(),
            };
            self.queue.write_buffer(&self.gpu_objects[i].object_uniform_buf, 0, bytemuck::bytes_of(&u));
        }

        self.ssao.update_params(&self.queue,
            scene.camera.proj(), scene.camera.proj().inverse(), scene.camera.view(),
            self.config.width, self.config.height);
        self.ssr.update_params(&self.queue, scene.camera.proj(), scene.camera.view());
        self.taa.update_params(&self.queue, unjittered_vp);

        let mut pp = self.post_params; pp.time = self.frame_time;
        self.post.update_params(&self.queue, &pp);

        (unjittered_vp, jittered_vp)
    }

    // ── Współdzielone passy renderowania (1–8) ────────────────────────────────

    fn encode_pbr_passes(
        &self,
        scene:        &Scene,
        encoder:      &mut wgpu::CommandEncoder,
        unjittered_vp: glam::Mat4,
    ) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView), wgpu::SurfaceError> {
        // Pass 1: CSM Shadow
        for cascade in 0..NUM_CASCADES {
            let cascade_view = self.shadow.set_cascade(&self.queue, cascade);
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_{cascade}")),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: cascade_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }), ..Default::default()
            });
            rp.set_pipeline(&self.shadow.pipeline);
            rp.set_bind_group(0, &self.shadow.csm_uniform_bg, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let g = &self.gpu_objects[i];
                rp.set_bind_group(1, &g.shadow_object_bg, &[]);
                rp.set_vertex_buffer(0, g.mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(g.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.mesh.index_count, 0, 0..1);
            }
        }

        // Pass 2: Skybox → HDR
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post.hdr_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }), ..Default::default()
            });
            rp.set_pipeline(&self.skybox.pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            rp.set_bind_group(1, &self.skybox.sky_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // Pass 3: PBR geometry → HDR
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pbr"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post.hdr_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }), ..Default::default()
            });
            rp.set_pipeline(&self.pbr_pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            rp.set_bind_group(3, &self.lights_shadow_bg, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let g = &self.gpu_objects[i];
                rp.set_bind_group(1, &g.object_bind_group, &[]);
                rp.set_bind_group(2, &g.material_bind_group, &[]);
                rp.set_vertex_buffer(0, g.mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(g.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.mesh.index_count, 0, 0..1);
            }
        }

        // Pass 4: GBuffer
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer_normals_view, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer_mr_view, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }), ..Default::default()
            });
            rp.set_pipeline(&self.gbuffer_pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let g = &self.gpu_objects[i];
                rp.set_bind_group(1, &g.object_bind_group, &[]);
                rp.set_bind_group(2, &g.material_bind_group, &[]);
                rp.set_vertex_buffer(0, g.mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(g.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.mesh.index_count, 0, 0..1);
            }
        }

        // Pass 5: SSAO
        self.ssao.render(&self.device, encoder, &self.depth_view, &self.gbuffer_normals_view);

        // Pass 6: SSR
        self.ssr.render(&self.device, encoder,
            &self.post.hdr_view, &self.depth_view,
            &self.gbuffer_normals_view, &self.gbuffer_mr_view);

        // Pass 7: TAA
        self.taa.render(&self.device, encoder, &self.post.hdr_view, &self.depth_view, unjittered_vp);

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.taa.output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.post.hdr_texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
        );

        // Pass 8: Post-processing → swapchain
        let output     = self.surface.get_current_texture()?;
        let final_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.post.render(&self.device, encoder, &final_view, &self.ssao.ao_blur_view);

        Ok((output, final_view))
    }

    // ── render() — oryginalna metoda bez egui (zachowana dla kompatybilności) ─

    pub fn render(&mut self, scene: &Scene) -> Result<(), wgpu::SurfaceError> {
        let (unjittered_vp, _) = self.prepare_frame(scene);

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame") });

        let (output, _final_view) = self.encode_pbr_passes(scene, &mut encoder, unjittered_vp)?;

        self.taa.advance_frame(unjittered_vp);
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    // ── render_with_egui() — render + egui pass na końcu ─────────────────────
    /// Używaj tej metody zamiast render() gdy UI jest aktywne.
    /// egui_renderer.begin_frame() musi być wywołane PRZED tą metodą.
    pub fn render_with_egui(
        &mut self,
        scene:         &Scene,
        egui_renderer: &mut crate::ui::egui_renderer::EguiRenderer,
        window:        &Window,
    ) -> Result<(), wgpu::SurfaceError> {
        let (unjittered_vp, _) = self.prepare_frame(scene);

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame_egui") });

        let (output, final_view) = self.encode_pbr_passes(scene, &mut encoder, unjittered_vp)?;

        // Pass 9: egui → swapchain (LoadOp::Load zachowuje wynik PBR)
        egui_renderer.end_frame_and_render(
            &self.device,
            &self.queue,
            &mut encoder,
            window,
            &final_view,
        );

        self.taa.advance_frame(unjittered_vp);
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}