// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  SSR — Screen Space Reflections Renderer                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SsrParams {
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub texel_size: [f32; 2],
    pub max_steps: u32,
    pub max_distance: f32,
    pub thickness: f32,
    pub stride: f32,
    pub _pad: [f32; 2],
}

pub struct SsrRenderer {
    pub output_tex: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    pub width: u32,
    pub height: u32,
}

impl SsrRenderer {
    fn make_output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr_output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    pub fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let (ot, ov) = Self::make_output(device, w, h);

        let zero_params = SsrParams {
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            inv_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            texel_size: [1.0 / w as f32, 1.0 / h as f32],
            max_steps: 32,
            max_distance: 50.0,
            thickness: 0.5,
            stride: 0.02,
            _pad: [0.0; 2],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssr_params"),
            contents: bytemuck::bytes_of(&zero_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_bgl"),
            entries: &[
                bgl_e(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_e(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_e(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_e(
                    3,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_e(
                    4,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_e(
                    5,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                bgl_e(
                    6,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                ),
                bgl_e(
                    7,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssr.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssr_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_ssr"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            output_tex: ot,
            output_view: ov,
            pipeline,
            params_buf,
            bgl,
            width: w,
            height: h,
        }
    }

    pub fn update_params(&self, queue: &wgpu::Queue, proj: Mat4, view: Mat4) {
        let p = SsrParams {
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
            max_steps: 24,
            max_distance: 50.0,
            thickness: 0.3,
            stride: 0.04,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        metallic_roughness: &wgpu::TextureView,
    ) {
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normals),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(metallic_roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssr_pass"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&self.pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let (t, v) = Self::make_output(device, w, h);
        self.output_tex = t;
        self.output_view = v;
        self.width = w;
        self.height = h;
    }
}

fn bgl_e(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}
