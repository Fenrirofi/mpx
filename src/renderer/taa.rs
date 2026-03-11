// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  TAA — Temporal Anti-Aliasing Renderer                                      ║
// ║  Ping-pong bufor historii + jitter macierzy projekcji                       ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

/// Halton sequence dla jittera (mniejszy aliasing niż MSAA)
const HALTON_2: [f32; 16] = [
    0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875,
    0.4375, 0.9375, 0.03125,
];
const HALTON_3: [f32; 16] = [
    0.333, 0.667, 0.111, 0.444, 0.778, 0.222, 0.556, 0.889, 0.037, 0.370, 0.704, 0.148, 0.481,
    0.815, 0.259, 0.593,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TaaParams {
    pub curr_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub texel_size: [f32; 2],
    pub blend_factor: f32,
    pub _pad: f32,
}

pub struct TaaRenderer {
    // Ping-pong tekstury historii
    history_textures: [wgpu::Texture; 2],
    pub history_views: [wgpu::TextureView; 2],

    // Wyjście TAA (czyste, bez jazu)
    pub output_tex: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,

    pub width: u32,
    pub height: u32,

    frame_index: usize,
    pub prev_view_proj: Mat4,
    jitter_index: usize,
}

impl TaaRenderer {
    fn make_tex(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    pub fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let (t0, v0) = Self::make_tex(device, w, h, "taa_hist_0");
        let (t1, v1) = Self::make_tex(device, w, h, "taa_hist_1");
        let (to, vo) = Self::make_tex(device, w, h, "taa_output");

        let zero_params = TaaParams {
            curr_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            prev_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            texel_size: [1.0 / w as f32, 1.0 / h as f32],
            blend_factor: 0.1,
            _pad: 0.0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taa_params"),
            contents: bytemuck::bytes_of(&zero_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa_bgl"),
            entries: &[
                // params
                bgl_entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // t_current
                bgl_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                // t_history
                bgl_entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                // t_depth
                bgl_entry(
                    3,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                // s_linear
                bgl_entry(
                    4,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // s_nearest
                bgl_entry(
                    5,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                ),
                // out_taa (storage)
                bgl_entry(
                    6,
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
            label: Some("taa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/taa.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa_layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_taa"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            history_textures: [t0, t1],
            history_views: [v0, v1],
            output_tex: to,
            output_view: vo,
            pipeline,
            params_buf,
            bgl,
            width: w,
            height: h,
            frame_index: 0,
            prev_view_proj: Mat4::IDENTITY,
            jitter_index: 0,
        }
    }

    /// Zwraca offset jittera dla bieżącej klatki w pikselach NDC.
    pub fn jitter(&self, width: u32, height: u32) -> (f32, f32) {
        let i = self.jitter_index % 16;
        let jx = (HALTON_2[i] - 0.5) * 2.0 / width as f32;
        let jy = (HALTON_3[i] - 0.5) * 2.0 / height as f32;
        (jx, jy)
    }

    /// Przesuń o 1 klatkę: zapisz poprzednią macierz, przesuń jitter
    pub fn advance_frame(&mut self, curr_view_proj: Mat4) {
        self.prev_view_proj = curr_view_proj;
        self.jitter_index = (self.jitter_index + 1) % 16;
        self.frame_index = (self.frame_index + 1) % 2;
    }

    /// Uruchom pass TAA.
    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        curr_hdr: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        curr_vp: Mat4,
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

        // Ping-pong: bieżąca historia = poprzednia klatka
        let hist_idx = (self.frame_index + 1) % 2;
        let out_idx = self.frame_index;

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(curr_hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[hist_idx]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        let params = TaaParams {
            curr_view_proj: curr_vp.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            inv_view_proj: curr_vp.inverse().to_cols_array_2d(),
            texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
            blend_factor: 0.1,
            _pad: 0.0,
        };
        // Nie możemy pisać tu do bufora (borrow), używamy push via encoder copy
        // W praktyce: zaktualizuj przed render()
        let _ = params; // już zaktualizowane w update_params()

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("taa_pass"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&self.pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        drop(cp);

        // Skopiuj wynik do output_view (history[out_idx] → output_tex)
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.history_textures[out_idx],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn update_params(&self, queue: &wgpu::Queue, curr_vp: Mat4) {
        let params = TaaParams {
            curr_view_proj: curr_vp.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            inv_view_proj: curr_vp.inverse().to_cols_array_2d(),
            texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
            blend_factor: 0.05,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let (t0, v0) = Self::make_tex(device, w, h, "taa_hist_0");
        let (t1, v1) = Self::make_tex(device, w, h, "taa_hist_1");
        let (to, vo) = Self::make_tex(device, w, h, "taa_output");
        self.history_textures = [t0, t1];
        self.history_views = [v0, v1];
        self.output_tex = to;
        self.output_view = vo;
        self.width = w;
        self.height = h;
    }
}

fn bgl_entry(
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
