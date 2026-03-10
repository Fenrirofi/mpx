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
    shadow::ShadowRenderer,
    skybox::{SkyboxRenderer, SkyUniform},
    ssao::SsaoRenderer,
    texture,
    uniforms::{CameraUniform, LightArrayUniform, MaterialUniform, ObjectUniform},
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
    surface:   wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    config:     wgpu::SurfaceConfiguration,
    size:       PhysicalSize<u32>,

    // Layouts (kept alive)
    _camera_bgl:       wgpu::BindGroupLayout,
    _object_bgl:       wgpu::BindGroupLayout,
    _material_bgl:     wgpu::BindGroupLayout,
    lights_shadow_bgl: wgpu::BindGroupLayout,

    // Pipelines
    pbr_pipeline:     wgpu::RenderPipeline,
    gbuffer_pipeline: wgpu::RenderPipeline,

    // Per-frame bind groups
    camera_uniform_buf: wgpu::Buffer,
    camera_bind_group:  wgpu::BindGroup,
    lights_uniform_buf: wgpu::Buffer,
    lights_shadow_bg:   wgpu::BindGroup,

    // Render targets
    depth_view:    wgpu::TextureView,
    gbuffer_tex:   wgpu::Texture,
    gbuffer_view:  wgpu::TextureView,

    // Fallback textures
    white_view:       wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    white_ao_view:    wgpu::TextureView,
    default_sampler:  wgpu::Sampler,

    // Sub-renderers
    pub shadow:  ShadowRenderer,
    pub skybox:  SkyboxRenderer,
    pub post:    PostProcessor,
    pub ssao:    SsaoRenderer,
    pub ibl:     Option<IblMaps>,

    // Scene GPU objects
    gpu_objects: Vec<GpuObject>,

    // State
    pub sky_uniform: SkyUniform,
    pub post_params: PostParams,
    frame_time:      f32,
}

// ── Helper: create gbuffer normal texture ─────────────────────────────────────
fn create_gbuffer(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gbuffer_normals"),
        size:  wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::Rgba16Float,
        usage:        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_gbuffer_pipeline(
    device:     &wgpu::Device,
    camera_bgl: &wgpu::BindGroupLayout,
    object_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("gbuffer"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gbuffer.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gbuffer_layout"), bind_group_layouts: &[camera_bgl, object_bgl], immediate_size: 0,
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
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None, cache: None,
    })
}

impl RenderContext {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface:     Some(&surface),
        }).await.map_err(|e| anyhow::anyhow!("Adapter: {e:?}"))?;

        log::info!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label:                 Some("pbr_device"),
            required_features:     wgpu::Features::empty(),
            required_limits:       wgpu::Limits::default(),
            memory_hints:          wgpu::MemoryHints::Performance,
            trace:                 wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }).await?;

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0], view_formats: vec![],
        };
        surface.configure(&device, &config);

        // ── Bind group layouts ────────────────────────────────────────────────
        let camera_bgl        = pipeline::camera_bind_group_layout(&device);
        let object_bgl        = pipeline::object_bind_group_layout(&device);
        let material_bgl      = pipeline::material_bind_group_layout(&device);
        let lights_shadow_bgl = pipeline::lights_shadow_bind_group_layout(&device);

        // ── Sub-renderers ─────────────────────────────────────────────────────
        let shadow = ShadowRenderer::new(&device);
        let skybox = SkyboxRenderer::new(&device, wgpu::TextureFormat::Rgba16Float, &camera_bgl);
        let post   = PostProcessor::new(&device, &queue, config.width, config.height, format);
        let ssao   = SsaoRenderer::new(&device, &queue, config.width, config.height);

        // ── Pipelines ─────────────────────────────────────────────────────────
        let pbr_pipeline     = pipeline::create_pbr_pipeline(&device, &camera_bgl, &object_bgl, &material_bgl, &lights_shadow_bgl);
        let gbuffer_pipeline = create_gbuffer_pipeline(&device, &camera_bgl, &object_bgl);

        // ── Uniforms ──────────────────────────────────────────────────────────
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

        let lights_shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights_shadow_bg"), layout: &lights_shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shadow.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&shadow.shadow_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&shadow.sampler) },
            ],
        });

        // ── Render targets ────────────────────────────────────────────────────
        let (_, depth_view) = texture::create_depth_texture(&device, config.width, config.height);
        let (gbuffer_tex, gbuffer_view) = create_gbuffer(&device, config.width, config.height);

        // ── Fallback textures ─────────────────────────────────────────────────
        let (_, white_view, default_sampler) = texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white");
        let (_, flat_normal_view, _)         = texture::create_fallback_texture(&device, &queue, [128,128,255,255], "flat_normal");
        let (_, white_ao_view, _)            = texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white_ao");

        let mut ctx = Self {
            surface, device, queue, config, size,
            _camera_bgl: camera_bgl, _object_bgl: object_bgl,
            _material_bgl: material_bgl, lights_shadow_bgl,
            pbr_pipeline, gbuffer_pipeline,
            camera_uniform_buf, camera_bind_group,
            lights_uniform_buf, lights_shadow_bg,
            depth_view, gbuffer_tex, gbuffer_view,
            white_view, flat_normal_view, white_ao_view, default_sampler,
            shadow, skybox, post, ssao, ibl: None,
            gpu_objects: Vec::new(),
            sky_uniform: SkyUniform::day(),
            post_params: PostParams::default(),
            frame_time: 0.0,
        };

        ctx.upload_scene_placeholder();
        Ok(ctx)
    }

    // ── Public: load an HDR environment ──────────────────────────────────────
    pub fn load_hdr(&mut self, path: &str) -> Result<()> {
        let maps = bake_ibl(&self.device, &self.queue, path)?;
        self.ibl = Some(maps);
        Ok(())
    }

    // ── Internal ──────────────────────────────────────────────────────────────
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

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size          = new_size;
        self.config.width  = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        let (_, dv) = texture::create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_view = dv;
        let (gt, gv) = create_gbuffer(&self.device, self.config.width, self.config.height);
        self.gbuffer_tex  = gt;
        self.gbuffer_view = gv;
        self.post.resize(&self.device, &self.queue, self.config.width, self.config.height, self.config.format);
        self.ssao.resize(&self.device, &self.queue, self.config.width, self.config.height);
    }

    pub fn render(&mut self, scene: &Scene) -> Result<(), wgpu::SurfaceError> {
        self.frame_time += 1.0 / 60.0;

        // ── Upload uniforms ────────────────────────────────────────────────────
        self.queue.write_buffer(&self.camera_uniform_buf, 0,
            bytemuck::bytes_of(&CameraUniform::from_camera(&scene.camera)));
        self.queue.write_buffer(&self.lights_uniform_buf, 0,
            bytemuck::bytes_of(&LightArrayUniform::from_scene(scene)));

        let sun_dir = scene.lights.iter()
            .find(|l| l.enabled && matches!(l.kind, LightKind::Directional))
            .map(|l| l.position_or_direction.normalize())
            .unwrap_or(Vec3::new(0.4, 0.8, 0.3).normalize());

        let light_vp = ShadowRenderer::compute_light_matrix(sun_dir, Vec3::ZERO, 8.0);
        self.shadow.update(&self.queue, light_vp);
        self.sky_uniform.sun_direction = [sun_dir.x, sun_dir.y, sun_dir.z, 0.0];
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

        // SSAO camera matrices
        self.ssao.update_params(&self.queue,
            scene.camera.proj(), scene.camera.proj().inverse(), scene.camera.view(),
            self.config.width, self.config.height);

        let mut pp = self.post_params; pp.time = self.frame_time;
        self.post.update_params(&self.queue, &pp);

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 1: Shadow map
        // ═══════════════════════════════════════════════════════════════════
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.shadow.pipeline);
            rp.set_bind_group(0, &self.shadow.uniform_bg, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let g = &self.gpu_objects[i];
                rp.set_bind_group(1, &g.shadow_object_bg, &[]);
                rp.set_vertex_buffer(0, g.mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(g.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.mesh.index_count, 0, 0..1);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 2: Skybox → HDR
        // ═══════════════════════════════════════════════════════════════════
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
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.skybox.pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            rp.set_bind_group(1, &self.skybox.sky_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 3: PBR geometry → HDR
        // ═══════════════════════════════════════════════════════════════════
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
                }),
                ..Default::default()
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

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 4: GBuffer (world normals, depth-read only)
        // ═══════════════════════════════════════════════════════════════════
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.gbuffer_pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let g = &self.gpu_objects[i];
                rp.set_bind_group(1, &g.object_bind_group, &[]);
                rp.set_vertex_buffer(0, g.mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(g.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.mesh.index_count, 0, 0..1);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 5: SSAO compute + blur
        // ═══════════════════════════════════════════════════════════════════
        // Need depth view for SSAO — create a depth-only view
        let depth_ssao_view = self.depth_view.clone();

        self.ssao.render(&self.device, &mut encoder, &depth_ssao_view, &self.gbuffer_view);

        // ═══════════════════════════════════════════════════════════════════
        //  Pass 6-9: Post-processing → swapchain (includes SSAO composite)
        // ═══════════════════════════════════════════════════════════════════
        let output     = self.surface.get_current_texture()?;
        let final_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.post.render(&self.device, &mut encoder, &final_view, &self.ssao.ao_blur_view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}