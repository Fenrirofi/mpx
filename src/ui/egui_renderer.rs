// ══════════════════════════════════════════════════════════════════════════════
//  src/ui/egui_renderer.rs — Własny renderer egui dla wgpu 28
//
//  Zastępuje egui-wgpu (która powoduje konflikt naga 28 vs naga 25).
//  egui + egui-winit nie zależą od wgpu — tylko egui-wgpu ma ten problem.
//
//  Implementacja:
//    - Własny shader WGSL (vertex transform + texture sample)
//    - Własny upload vertex/index bufferów (tworzonych co draw call — ok dla UI)
//    - Własne zarządzanie atlasem fontów jako Rgba8UnormSrgb
//    - Scissor rect per ClippedPrimitive
//    - Premultiplied alpha blending (standard egui)
// ══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use egui::{ClippedPrimitive, TextureId, epaint::Primitive};
use wgpu::util::DeviceExt;
use winit::window::Window;

// ─────────────────────────────────────────────────────────────────────────────
// Vertex — musi pasować do @location() w shaderze poniżej
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EguiVertex {
    pos:   [f32; 2],  // logical pixels
    uv:    [f32; 2],
    color: [u8; 4],   // sRGB + alpha (egui::Color32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen uniform — rozmiar viewportu w logical pixels
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// Tekstura GPU
// ─────────────────────────────────────────────────────────────────────────────

struct GpuTexture {
    texture:    wgpu::Texture,
    _view:      wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

// ─────────────────────────────────────────────────────────────────────────────
// EguiRenderer
// ─────────────────────────────────────────────────────────────────────────────

pub struct EguiRenderer {
    pub context:    egui::Context,
    state:          egui_winit::State,
    pipeline:       wgpu::RenderPipeline,
    screen_buf:     wgpu::Buffer,
    screen_bg:      wgpu::BindGroup,
    texture_bgl:    wgpu::BindGroupLayout,
    sampler:        wgpu::Sampler,
    textures:       HashMap<TextureId, GpuTexture>,
}

impl EguiRenderer {
    pub fn new(
        device:         &wgpu::Device,
        surface_format:  wgpu::TextureFormat,
        window:         &Window,
    ) -> Self {
        let context = egui::Context::default();
        super::apply_dark_theme(&context);

        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("egui_shader"),
            source: wgpu::ShaderSource::Wgsl(EGUI_SHADER.into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        let screen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("egui_screen_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("egui_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("egui_layout"),
            bind_group_layouts:  &[&screen_bgl, &texture_bgl],
            immediate_size:      0,
        });

        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<EguiVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8,  shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Unorm8x4  },
            ],
        };

        // Premultiplied alpha — standard egui
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("egui_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[vbl],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── Screen buf + BG ───────────────────────────────────────────────────
        let screen_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("egui_screen"),
            contents: bytemuck::bytes_of(&ScreenUniform { size: [1280.0, 720.0] }),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let screen_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("egui_screen_bg"),
            layout:  &screen_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: screen_buf.as_entire_binding() }],
        });

        // ── Sampler ───────────────────────────────────────────────────────────
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("egui_sampler"),
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            context, state,
            pipeline, screen_buf, screen_bg,
            texture_bgl, sampler,
            textures: HashMap::new(),
        }
    }

    // ── Input ─────────────────────────────────────────────────────────────────

    pub fn handle_input(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    pub fn wants_pointer_input(&self)  -> bool { self.context.wants_pointer_input() }
    pub fn wants_keyboard_input(&self) -> bool { self.context.wants_keyboard_input() }

    // ── Frame ─────────────────────────────────────────────────────────────────

    pub fn begin_frame(&mut self, window: &Window) -> &egui::Context {
        self.context.begin_pass(self.state.take_egui_input(window));
        &self.context
    }

    pub fn end_frame_and_render(
        &mut self,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window:  &Window,
        view:    &wgpu::TextureView,
    ) {
        let full_output = self.context.end_pass();
        self.state.handle_platform_output(window, full_output.platform_output);

        let ppp    = full_output.pixels_per_point;
        let size   = window.inner_size();
        let log_w  = size.width  as f32 / ppp;
        let log_h  = size.height as f32 / ppp;

        queue.write_buffer(&self.screen_buf, 0,
            bytemuck::bytes_of(&ScreenUniform { size: [log_w, log_h] }));

        // Aktualizuj tekstury egui (font atlas, user images)
        for (id, delta) in &full_output.textures_delta.set {
            self.update_texture(device, queue, *id, delta);
        }

        let prims = self.context.tessellate(full_output.shapes, ppp);

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load,  // zachowaj PBR pod spodem
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.screen_bg, &[]);

            for ClippedPrimitive { clip_rect, primitive } in &prims {
                // Scissor rect → fizyczne piksele, clamp do okna
                let x0 = (clip_rect.min.x * ppp).round() as u32;
                let y0 = (clip_rect.min.y * ppp).round() as u32;
                let x1 = (clip_rect.max.x * ppp).round() as u32;
                let y1 = (clip_rect.max.y * ppp).round() as u32;
                let x0 = x0.min(size.width);
                let y0 = y0.min(size.height);
                let sw = (x1.min(size.width)).saturating_sub(x0);
                let sh = (y1.min(size.height)).saturating_sub(y0);
                if sw == 0 || sh == 0 { continue; }
                rp.set_scissor_rect(x0, y0, sw, sh);

                let Primitive::Mesh(mesh) = primitive else { continue; };
                if mesh.vertices.is_empty() || mesh.indices.is_empty() { continue; }

                let Some(gpu_tex) = self.textures.get(&mesh.texture_id) else { continue; };
                rp.set_bind_group(1, &gpu_tex.bind_group, &[]);

                let verts: Vec<EguiVertex> = mesh.vertices.iter().map(|v| EguiVertex {
                    pos:   [v.pos.x, v.pos.y],
                    uv:    [v.uv.x, v.uv.y],
                    color: v.color.to_array(),
                }).collect();

                let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:    Some("egui_v"),
                    contents: bytemuck::cast_slice(&verts),
                    usage:    wgpu::BufferUsages::VERTEX,
                });
                let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:    Some("egui_i"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage:    wgpu::BufferUsages::INDEX,
                });

                rp.set_vertex_buffer(0, vbuf.slice(..));
                rp.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }
        }

        for id in &full_output.textures_delta.free {
            self.textures.remove(id);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Aktualizacja tekstury egui (font atlas + user images)
    // ─────────────────────────────────────────────────────────────────────────

    fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        id:     TextureId,
        delta:  &egui::epaint::ImageDelta,
    ) {
        let [w, h] = delta.image.size();
        let (w, h) = (w as u32, h as u32);

        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(img) => {
                img.pixels.iter().flat_map(|c| c.to_array()).collect()
            }
        };

        // Częściowa aktualizacja istniejącej tekstury
        if let Some(pos) = delta.pos {
            if let Some(gpu) = self.textures.get(&id) {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture:   &gpu.texture,
                        mip_level: 0,
                        origin:    wgpu::Origin3d { x: pos[0] as u32, y: pos[1] as u32, z: 0 },
                        aspect:    wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        offset:         0,
                        bytes_per_row:  Some(w * 4),
                        rows_per_image: Some(h),
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                return;
            }
        }

        // Nowa tekstura
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("egui_tex"),
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        queue.write_texture(
            texture.as_image_copy(),
            &data,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("egui_tex_bg"),
            layout:  &self.texture_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        self.textures.insert(id, GpuTexture { texture, _view: view, bind_group });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — minimalistyczny renderer egui
// ─────────────────────────────────────────────────────────────────────────────

const EGUI_SHADER: &str = r#"
struct Screen { size: vec2<f32> }
@group(0) @binding(0) var<uniform> screen: Screen;
@group(1) @binding(0) var t_tex: texture_2d<f32>;
@group(1) @binding(1) var s_tex: sampler;

struct VIn {
    @location(0) pos:   vec2<f32>,
    @location(1) uv:    vec2<f32>,
    @location(2) color: vec4<f32>,  // Unorm8x4 → automatycznie [0,1]
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:         vec2<f32>,
    @location(1) color:      vec4<f32>,
}

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.clip  = vec4<f32>(
         2.0 * in.pos.x / screen.size.x - 1.0,
        -2.0 * in.pos.y / screen.size.y + 1.0,
        0.0, 1.0
    );
    out.uv    = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // egui color: sRGB z premultiplied alpha
    // Tekstura fontów: Rgba8UnormSrgb z GPU-side dekodowaniem
    return in.color * textureSample(t_tex, s_tex, in.uv);
}
"#;