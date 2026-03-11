// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  SSAO Renderer                                                              ║
// ║  Inputs:  depth texture + world-normal GBuffer                             ║
// ║  Output:  blurred AO texture (R8Unorm), read by composite pass             ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use wgpu::util::DeviceExt;

pub const KERNEL_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SsaoParams {
    pub proj:         [[f32; 4]; 4],
    pub inv_proj:     [[f32; 4]; 4],
    pub view:         [[f32; 4]; 4],
    pub radius:       f32,
    pub bias:         f32,
    pub strength:     f32,
    pub sample_count: u32,
    pub noise_scale:  [f32; 2],
    pub _pad:         [f32; 2],
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }

/// Generate a random hemisphere kernel (view space).
/// Używamy prawdziwego RNG zamiast złotej proporcji — ta dawała widoczne powtarzające się wzorce.
pub fn generate_ssao_kernel() -> Vec<[f32; 4]> {
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_1234_5678);
    let mut kernel = Vec::with_capacity(KERNEL_SIZE);
    for i in 0..KERNEL_SIZE {
        let scale = i as f32 / KERNEL_SIZE as f32;
        let scale = lerp(0.1, 1.0, scale * scale);
        let v = Vec3::new(
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>(),           // z ∈ [0, 1) — hemisfera, nie sfera
        ).normalize() * scale;
        kernel.push([v.x, v.y, v.z, 0.0]);
    }
    kernel
}

/// Generate 4×4 random rotation noise texture.
pub fn generate_noise_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE_9876_5432);
    let mut noise = vec![0f32; 16 * 3];
    for i in 0..16 {
        noise[i*3]   = rng.gen::<f32>() * 2.0 - 1.0;
        noise[i*3+1] = rng.gen::<f32>() * 2.0 - 1.0;
        noise[i*3+2] = 0.0; // z=0 — rotacja tylko w płaszczyźnie XY
    }
    // Pad RGB → RGBA (GPU wymaga wyrównanego formatu)
    let rgba: Vec<f32> = noise.chunks(3).flat_map(|c| [c[0], c[1], c[2], 1.0]).collect();

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_noise"),
        size:  wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::Rgba32Float,
        usage:        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(), bytemuck::cast_slice(&rgba),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4*16), rows_per_image: Some(4) },
        wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub struct SsaoRenderer {
    // Pipelines
    ssao_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,

    // Textures
    pub ao_raw_tex:    wgpu::Texture,
    pub ao_raw_view:   wgpu::TextureView,
    pub ao_blur_tex:   wgpu::Texture,
    pub ao_blur_view:  wgpu::TextureView,
    _noise_tex:        wgpu::Texture,
    noise_view:        wgpu::TextureView,

    // Buffers
    params_buf:  wgpu::Buffer,
    kernel_buf:  wgpu::Buffer,
    blur_p_buf:  wgpu::Buffer,

    // Layouts
    ssao_bgl:   wgpu::BindGroupLayout,
    blur_bgl:   wgpu::BindGroupLayout,

    // Sampler
    nearest_sampler: wgpu::Sampler,

    pub width:  u32,
    pub height: u32,
}

fn ao_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size:  wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::Rgba8Unorm,
        usage:        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl SsaoRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:         Some("ssao_nearest"),
            mag_filter:    wgpu::FilterMode::Nearest,
            min_filter:    wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // ── Kernel + noise ────────────────────────────────────────────────────
        let kernel_data  = generate_ssao_kernel();
        let kernel_buf   = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_kernel"), contents: bytemuck::cast_slice(&kernel_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (_noise_tex, noise_view) = generate_noise_texture(device, queue);

        // ── AO textures ───────────────────────────────────────────────────────
        let (ao_raw_tex, ao_raw_view)   = ao_texture(device, width, height, "ao_raw");
        let (ao_blur_tex, ao_blur_view) = ao_texture(device, width, height, "ao_blur");

        // ── Params buffer ─────────────────────────────────────────────────────
        let default_params = SsaoParams {
            proj:         Mat4::IDENTITY.to_cols_array_2d(),
            inv_proj:     Mat4::IDENTITY.to_cols_array_2d(),
            view:         Mat4::IDENTITY.to_cols_array_2d(),
            radius:       0.3,
            bias:         0.025,
            strength:     0.8,
            sample_count: KERNEL_SIZE as u32,
            noise_scale:  [width as f32 / 4.0, height as f32 / 4.0],
            _pad:         [0.0; 2],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_params"), contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── SSAO BGL ──────────────────────────────────────────────────────────
        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                // params
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }, count: None },
                // depth
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // world normal
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // kernel storage
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None }, count: None },
                // noise
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // nearest sampler
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
                // output AO
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });

        // ── Blur BGL ──────────────────────────────────────────────────────────
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:  Some("ssao_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&ssao_bgl], immediate_size: 0,
            })),
            module:  &shader, entry_point: Some("cs_ssao"),
            compilation_options: Default::default(), cache: None,
        });

        #[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
        struct BlurP { texel: [f32;2], _pad: [f32;2] }
        let bp = BlurP { texel: [1.0/width as f32, 1.0/height as f32], _pad: [0.0;2] };
        let blur_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_p"), contents: bytemuck::bytes_of(&bp),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:  Some("ssao_blur_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&blur_bgl], immediate_size: 0,
            })),
            module:  &shader, entry_point: Some("cs_ssao_blur"),
            compilation_options: Default::default(), cache: None,
        });

        Self {
            ssao_pipeline, blur_pipeline,
            ao_raw_tex, ao_raw_view, ao_blur_tex, ao_blur_view,
            _noise_tex, noise_view,
            params_buf, kernel_buf, blur_p_buf,
            ssao_bgl, blur_bgl,
            nearest_sampler,
            width, height,
        }
    }

    /// Call every frame with current camera matrices and GBuffer textures.
    pub fn update_params(&self, queue: &wgpu::Queue, proj: Mat4, inv_proj: Mat4, view: Mat4, width: u32, height: u32) {
        let p = SsaoParams {
            proj:         proj.to_cols_array_2d(),
            inv_proj:     inv_proj.to_cols_array_2d(),
            view:         view.to_cols_array_2d(),
            radius:       0.5,
            bias:         0.025,
            strength:     1.5,
            sample_count: KERNEL_SIZE as u32,
            noise_scale:  [width as f32 / 4.0, height as f32 / 4.0],
            _pad:         [0.0; 2],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    /// Dispatch SSAO compute + blur.
    pub fn render(
        &self,
        device:      &wgpu::Device,
        encoder:     &mut wgpu::CommandEncoder,
        depth_view:  &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
    ) {
        // SSAO pass
        let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bg"), layout: &self.ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: self.kernel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.noise_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.nearest_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.ao_raw_view) },
            ],
        });

        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_bg"), layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.ao_raw_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.nearest_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ao_blur_view) },
                wgpu::BindGroupEntry { binding: 4, resource: self.blur_p_buf.as_entire_binding() },
            ],
        });

        let wx = (self.width  + 7) / 8;
        let wy = (self.height + 7) / 8;

        // SSAO pass
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ssao"), timestamp_writes: None });
        pass.set_pipeline(&self.ssao_pipeline);
        pass.set_bind_group(0, &ssao_bg, &[]);
        pass.dispatch_workgroups(wx, wy, 1);
        drop(pass);
        
        // Blur pass
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ssao_blur"), timestamp_writes: None });
        pass.set_pipeline(&self.blur_pipeline);
        pass.set_bind_group(0, &blur_bg, &[]);
        pass.dispatch_workgroups(wx, wy, 1);
        drop(pass);
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) {
        *self = Self::new(device, queue, w, h);
    }
}