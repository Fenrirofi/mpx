use std::sync::Arc;
use anyhow::Result;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::scene::Scene;
use crate::scene::light::LightKind;
use super::{
    gpu_mesh::GpuMesh,
    pipeline,
    post::{PostParams, PostProcessor},
    shadow::ShadowRenderer,
    skybox::{SkyboxRenderer, SkyUniform},
    texture,
    uniforms::{CameraUniform, LightArrayUniform, MaterialUniform, ObjectUniform},
};

struct GpuObject {
    mesh:                 GpuMesh,
    object_uniform_buf:   wgpu::Buffer,
    object_bind_group:    wgpu::BindGroup,
    _material_uniform_buf: wgpu::Buffer,
    material_bind_group:  wgpu::BindGroup,
    shadow_object_bg:     wgpu::BindGroup,
}

pub struct RenderContext {
    surface:  wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    config:     wgpu::SurfaceConfiguration,
    size:       PhysicalSize<u32>,

    _camera_bgl:       wgpu::BindGroupLayout,
    _object_bgl:       wgpu::BindGroupLayout,
    _material_bgl:     wgpu::BindGroupLayout,
    lights_shadow_bgl: wgpu::BindGroupLayout,

    pbr_pipeline: wgpu::RenderPipeline,

    camera_uniform_buf: wgpu::Buffer,
    camera_bind_group:  wgpu::BindGroup,
    lights_uniform_buf: wgpu::Buffer,
    lights_shadow_bg:   wgpu::BindGroup,

    depth_view:       wgpu::TextureView,
    white_view:       wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    default_sampler:  wgpu::Sampler,

    pub shadow:  ShadowRenderer,
    pub skybox:  SkyboxRenderer,
    pub post:    PostProcessor,

    gpu_objects: Vec<GpuObject>,

    pub sky_uniform: SkyUniform,
    pub post_params: PostParams,
    frame_time:      f32,
}

impl RenderContext {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface:     Some(&surface),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Adapter error: {e:?}"))?;

        log::info!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label:                 Some("pbr_device"),
            required_features:     wgpu::Features::empty(),
            required_limits:       wgpu::Limits::default(),
            memory_hints:          wgpu::MemoryHints::Performance,
            trace:                 wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }).await?;

        let surface_caps   = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied()
            .find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width:  size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let camera_bgl        = pipeline::camera_bind_group_layout(&device);
        let object_bgl        = pipeline::object_bind_group_layout(&device);
        let material_bgl      = pipeline::material_bind_group_layout(&device);
        let lights_shadow_bgl = pipeline::lights_shadow_bind_group_layout(&device);

        let shadow = ShadowRenderer::new(&device);
        let skybox = SkyboxRenderer::new(&device, wgpu::TextureFormat::Rgba16Float, &camera_bgl);
        let post   = PostProcessor::new(&device, &queue, config.width, config.height, surface_format);

        let pbr_pipeline = pipeline::create_pbr_pipeline(
            &device, &camera_bgl, &object_bgl, &material_bgl, &lights_shadow_bgl,
        );

        let camera_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buf"),
            size:  std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"), layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buf.as_entire_binding() }],
        });

        let lights_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights_buf"),
            size:  std::mem::size_of::<LightArrayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        let (_, depth_view)        = texture::create_depth_texture(&device, config.width, config.height);
        let (_, white_view, default_sampler) = texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white");
        let (_, flat_normal_view, _)         = texture::create_fallback_texture(&device, &queue, [128,128,255,255], "flat_normal");

        let sky_uniform = SkyUniform::day();
        let post_params = PostParams::default();

        let mut ctx = Self {
            surface, device, queue, config, size,
            _camera_bgl: camera_bgl, _object_bgl: object_bgl,
            _material_bgl: material_bgl, lights_shadow_bgl,
            pbr_pipeline,
            camera_uniform_buf, camera_bind_group,
            lights_uniform_buf, lights_shadow_bg,
            depth_view, white_view, flat_normal_view, default_sampler,
            shadow, skybox, post,
            gpu_objects: Vec::new(),
            sky_uniform, post_params, frame_time: 0.0,
        };

        ctx.upload_scene_objects_placeholder();
        Ok(ctx)
    }

    fn create_gpu_object(&self, mesh: &crate::assets::Mesh, material: &crate::assets::Material) -> GpuObject {
        let gpu_mesh = GpuMesh::upload(&self.device, mesh);

        let object_uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("obj_buf"), size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("obj_bg"), layout: &self._object_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: object_uniform_buf.as_entire_binding() }],
        });
        let shadow_object_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_obj_bg"), layout: &self.shadow.object_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: object_uniform_buf.as_entire_binding() }],
        });

        let mat_uniform = MaterialUniform {
            base_color:            material.base_color.to_array(),
            metallic_roughness_ao: [material.metallic, material.roughness, material.ao_strength, material.normal_scale],
            emissive:              [material.emissive[0], material.emissive[1], material.emissive[2], 0.0],
        };
        let material_uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mat_buf"), contents: bytemuck::bytes_of(&mat_uniform),
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

    fn upload_scene_objects_placeholder(&mut self) {
        let scene = Scene::default_scene();
        self.gpu_objects = scene.objects.iter()
            .map(|o| self.create_gpu_object(&o.mesh, &o.material))
            .collect();
    }

    pub fn size(&self) -> PhysicalSize<u32> { self.size }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size          = new_size;
        self.config.width  = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        let (_, dv) = texture::create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_view = dv;
        self.post.resize(&self.device, &self.queue, self.config.width, self.config.height, self.config.format);
    }

    pub fn render(&mut self, scene: &Scene) -> Result<(), wgpu::SurfaceError> {
        self.frame_time += 1.0 / 60.0;

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
            let mut t  = obj.transform.clone();
            let obj_u  = ObjectUniform {
                model:         t.matrix().to_cols_array_2d(),
                normal_matrix: t.normal_matrix().to_cols_array_2d(),
            };
            self.queue.write_buffer(&self.gpu_objects[i].object_uniform_buf, 0, bytemuck::bytes_of(&obj_u));
        }

        let mut pp  = self.post_params;
        pp.time     = self.frame_time;
        self.post.update_params(&self.queue, &pp);

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ── Pass 1: Shadow ────────────────────────────────────────────────────
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rpass.set_pipeline(&self.shadow.pipeline);
            rpass.set_bind_group(0, &self.shadow.uniform_bg, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let gpu = &self.gpu_objects[i];
                rpass.set_bind_group(1, &gpu.shadow_object_bg, &[]);
                rpass.set_vertex_buffer(0, gpu.mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(gpu.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..gpu.mesh.index_count, 0, 0..1);
            }
        }

        // ── Pass 2: Skybox → HDR ──────────────────────────────────────────────
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post.hdr_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rpass.set_pipeline(&self.skybox.pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_bind_group(1, &self.skybox.sky_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        // ── Pass 3: PBR geometry → HDR ────────────────────────────────────────
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pbr_pass"),
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
            rpass.set_pipeline(&self.pbr_pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_bind_group(3, &self.lights_shadow_bg, &[]);
            for (i, _) in scene.objects.iter().enumerate() {
                if i >= self.gpu_objects.len() { break; }
                let gpu = &self.gpu_objects[i];
                rpass.set_bind_group(1, &gpu.object_bind_group, &[]);
                rpass.set_bind_group(2, &gpu.material_bind_group, &[]);
                rpass.set_vertex_buffer(0, gpu.mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(gpu.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..gpu.mesh.index_count, 0, 0..1);
            }
        }

        // ── Pass 4-7: Post → swapchain ────────────────────────────────────────
        let output     = self.surface.get_current_texture()?;
        let final_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.post.render(&self.device, &mut encoder, &final_view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}