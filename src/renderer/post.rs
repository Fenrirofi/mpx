use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Parameters sent to the composite shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PostParams {
    pub bloom_strength:      f32,
    pub exposure:            f32,
    pub ca_strength:         f32,
    pub vignette_strength:   f32,
    pub vignette_radius:     f32,
    pub vignette_smoothness: f32,
    pub grain_strength:      f32,
    pub time:                f32,
    pub ssao_strength:       f32,
    pub _pad0:               f32,
    pub _pad1:               f32,
    pub _pad2:               f32,
}

impl Default for PostParams {
    fn default() -> Self {
        Self {
            bloom_strength:      0.015,
            exposure:            0.0,
            ca_strength:         0.04,
            vignette_strength:   0.45,
            vignette_radius:     0.75,
            vignette_smoothness: 0.4,
            grain_strength:      0.02,
            time:                0.0,
            ssao_strength:       0.0,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    threshold: f32,
    knee:      f32,
    scatter:   f32,
    _pad:      f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    texel_size: [f32; 2],
    horizontal: u32,
    _pad:       u32,
}

/// HDR render target + full post-processing stack.
pub struct PostProcessor {
    // HDR offscreen target
    pub hdr_texture: wgpu::Texture,
    pub hdr_view:    wgpu::TextureView,

    // Bloom ping-pong buffers
    bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,

    // Pipelines
    threshold_pipeline: wgpu::RenderPipeline,
    blur_pipeline:      wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    threshold_bgl: wgpu::BindGroupLayout,
    blur_bgl:      wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    // Buffers
    bloom_params_buf: wgpu::Buffer,
    blur_h_buf:       wgpu::Buffer,
    blur_v_buf:       wgpu::Buffer,
    post_params_buf:  wgpu::Buffer,

    // Samplers
    nearest_sampler: wgpu::Sampler,
    linear_sampler:  wgpu::Sampler,

    pub width:  u32,
    pub height: u32,
}

fn bloom_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some(label),
        size:            wgpu::Extent3d { width: w / 2, height: h / 2, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rgba16Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn hdr_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("hdr_texture"),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rgba16Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn fullscreen_bgl(device: &wgpu::Device, label: &str, extra_entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled:   false,
            }, count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    entries.extend_from_slice(extra_entries);
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label), entries: &entries,
    })
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry:  &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some(entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader, entry_point: Some("vs_fullscreen"),
            buffers: &[], compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample:   wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader, entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl PostProcessor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/post.wgsl").into()),
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest"), mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest, mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"), mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (hdr_texture, hdr_view)   = hdr_texture(device, width, height);
        let (bloom_a, bloom_a_view)   = bloom_texture(device, width, height, "bloom_a");
        let (bloom_b, bloom_b_view)   = bloom_texture(device, width, height, "bloom_b");

        // Bloom threshold BGL
        let threshold_bgl = fullscreen_bgl(device, "threshold_bgl", &[
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ]);

        // Blur BGL
        let blur_bgl = fullscreen_bgl(device, "blur_bgl", &[
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ]);

        // Composite BGL: hdr + bloom + sampler + params
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }, count: None },
                // binding 4: SSAO texture
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        });

        // Pipelines
        let thresh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thresh_layout"), bind_group_layouts: &[&threshold_bgl], immediate_size: 0,
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur_layout"), bind_group_layouts: &[&blur_bgl], immediate_size: 0,
        });
        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("comp_layout"), bind_group_layouts: &[&composite_bgl], immediate_size: 0,
        });

        let bloom_fmt = wgpu::TextureFormat::Rgba16Float;
        let threshold_pipeline = fullscreen_pipeline(device, &thresh_layout, &shader, "fs_bloom_threshold", bloom_fmt);
        let blur_pipeline       = fullscreen_pipeline(device, &blur_layout,  &shader, "fs_bloom_blur",      bloom_fmt);
        let composite_pipeline  = fullscreen_pipeline(device, &comp_layout,  &shader, "fs_composite",       surface_format);

        // Buffers
        let bloom_p = BloomParams { threshold: 1.0, knee: 0.1, scatter: 0.7, _pad: 0.0 };
        let bloom_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom_params"), contents: bytemuck::bytes_of(&bloom_p),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let w2 = (width / 2) as f32; let h2 = (height / 2) as f32;
        let blur_h = BlurParams { texel_size: [1.0/w2, 1.0/h2], horizontal: 1, _pad: 0 };
        let blur_v = BlurParams { texel_size: [1.0/w2, 1.0/h2], horizontal: 0, _pad: 0 };
        let blur_h_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_h"), contents: bytemuck::bytes_of(&blur_h),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_v"), contents: bytemuck::bytes_of(&blur_v),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let post_p = PostParams::default();
        let post_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post_params"), contents: bytemuck::bytes_of(&post_p),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            hdr_texture, hdr_view,
            bloom_a, bloom_a_view, bloom_b, bloom_b_view,
            threshold_pipeline, blur_pipeline, composite_pipeline,
            threshold_bgl, blur_bgl, composite_bgl,
            bloom_params_buf, blur_h_buf, blur_v_buf, post_params_buf,
            nearest_sampler, linear_sampler,
            width, height,
        }
    }

    /// Update post-processing params (call each frame or when changed).
    pub fn update_params(&self, queue: &wgpu::Queue, params: &PostParams) {
        queue.write_buffer(&self.post_params_buf, 0, bytemuck::bytes_of(params));
    }

    /// Run the full post stack: threshold → blur×N → composite → swapchain.
    pub fn render(
        &self,
        device:      &wgpu::Device,
        encoder:     &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        ssao_view:   &wgpu::TextureView,
    ) {
        // ── Pass 1: bloom threshold (HDR → bloom_a) ───────────────────────────
        let thresh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thresh_bg"), layout: &self.threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.nearest_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.bloom_params_buf.as_entire_binding() },
            ],
        });
        self.run_pass(encoder, &self.threshold_pipeline, &thresh_bg, &self.bloom_a_view);

        // ── Passes 2-5: ping-pong blur (4 iterations) ────────────────────────
        for i in 0..4u32 {
            let (src, dst, buf) = if i % 2 == 0 {
                (&self.bloom_a_view, &self.bloom_b_view, &self.blur_h_buf)
            } else {
                (&self.bloom_b_view, &self.bloom_a_view, &self.blur_v_buf)
            };
            let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur_bg"), layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                ],
            });
            self.run_pass(encoder, &self.blur_pipeline, &blur_bg, dst);
        }

        // ── Pass 6: composite (HDR + bloom_a → swapchain) ────────────────────
        let comp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("comp_bg"), layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.bloom_a_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.post_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ssao_view) },
            ],
        });
        self.run_pass(encoder, &self.composite_pipeline, &comp_bg, output_view);
    }

    fn run_pass(
        &self,
        encoder:  &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bg:       &wgpu::BindGroup,
        target:   &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Recreate size-dependent resources after window resize.
    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, surface_format: wgpu::TextureFormat) {
        *self = Self::new(device, queue, width, height, surface_format);
    }
}