use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// GPU uniform for sky parameters
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SkyUniform {
    pub sun_direction:   [f32; 4],
    pub sun_color:       [f32; 4], // rgb + intensity
    pub sky_color_top:   [f32; 4],
    pub sky_color_horiz: [f32; 4],
    pub ground_color:    [f32; 4],
    /// x=sun_size(cos angle), y=halo_size, z=exposure, w=unused
    pub params:          [f32; 4],
}

impl SkyUniform {
    pub fn day() -> Self {
        Self {
            sun_direction:   [ 0.4,  0.8,  0.3, 0.0],
            sun_color:       [ 1.0,  0.95, 0.8, 12.0],
            sky_color_top:   [ 0.1,  0.25, 0.7, 1.0],
            sky_color_horiz: [ 0.5,  0.7,  0.9, 1.0],
            ground_color:    [ 0.15, 0.12, 0.08, 1.0],
            params:          [0.9998, 0.995, 1.0, 0.0],
        }
    }

    pub fn sunset() -> Self {
        Self {
            sun_direction:   [ 0.9,  0.15,  0.2, 0.0],
            sun_color:       [ 1.0,  0.45, 0.1, 10.0],
            sky_color_top:   [ 0.1,  0.12, 0.35, 1.0],
            sky_color_horiz: [ 0.85, 0.35, 0.1,  1.0],
            ground_color:    [ 0.1,  0.07, 0.05, 1.0],
            params:          [0.9997, 0.993, 1.0, 0.0],
        }
    }
}

pub struct SkyboxRenderer {
    pub pipeline:   wgpu::RenderPipeline,
    pub uniform_buf: wgpu::Buffer,
    pub sky_bgl:    wgpu::BindGroupLayout,
    pub sky_bg:     wgpu::BindGroup,
}

impl SkyboxRenderer {
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl:     &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("skybox_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/skybox.wgsl").into()),
        });

        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let sky_data    = SkyUniform::day();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("sky_uniform_buf"),
            contents: bytemuck::bytes_of(&sky_data),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sky_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("sky_bg"),
            layout:  &sky_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("sky_pipeline_layout"),
            bind_group_layouts: &[camera_bgl, &sky_bgl],
            immediate_size:     0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("skybox_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[], // fullscreen triangle, no VBO
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // sky never writes depth
                depth_compare:       wgpu::CompareFunction::LessEqual,
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache:          None,
        });

        Self { pipeline, uniform_buf, sky_bgl, sky_bg }
    }

    pub fn update(&self, queue: &wgpu::Queue, sky: &SkyUniform) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(sky));
    }

    /// Sun direction (normalized, pointing TOWARD sun) for light matching.
    pub fn sun_direction_from_uniform(sky: &SkyUniform) -> Vec3 {
        Vec3::from([sky.sun_direction[0], sky.sun_direction[1], sky.sun_direction[2]]).normalize()
    }
}