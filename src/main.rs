// ══════════════════════════════════════════════════════════════════════════════
//  src/main.rs — Malachite PBR Engine z egui UI i systemem warstw
// ══════════════════════════════════════════════════════════════════════════════

mod renderer;
mod scene;
mod assets;
mod math;
mod utils;
mod layers;  // ← NOWY: system warstw
mod ui;      // ← NOWY: egui panels

use anyhow::Result;
use log::info;
use std::time::Instant;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use renderer::RenderContext;
use scene::Scene;
use scene::camera::CameraController;
use layers::LayerStack;
use ui::{UiState, egui_renderer::EguiRenderer};

// ─────────────────────────────────────────────────────────────────────────────
// AppState
// ─────────────────────────────────────────────────────────────────────────────

struct AppState {
    window:      Arc<Window>,
    render_ctx:  RenderContext,
    scene:       Scene,
    controller:  CameraController,

    // UI
    egui_renderer: EguiRenderer,
    ui_state:      UiState,

    // Layer system
    /// Jeden LayerStack per obiekt sceny
    layer_stacks: Vec<LayerStack>,
    /// Indeks aktualnie wybranego obiektu sceny (dla layer stack)
    selected_object: usize,

    // FPS
    fps_timer:   Instant,
    fps_frames:  u32,
    fps_last:    f32,
}

struct App { state: Option<AppState> }

impl App { fn new() -> Self { Self { state: None } } }

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Malachite — PBR Engine")
                    .with_inner_size(winit::dpi::PhysicalSize::new(1600, 900))
            ).expect("window")
        );

        let mut render_ctx = pollster::block_on(RenderContext::new(Arc::clone(&window)))
            .expect("renderer");

        match render_ctx.load_hdr("assets/ticknock_01_4k.hdr") {
            Ok(_)  => log::info!("HDR loaded — pełne IBL aktywne."),
            Err(e) => log::warn!("Brak pliku HDR ({e}). Używam fallback sky."),
        }

        let scene = Scene::default_scene();

        // Inicjalizuj egui renderer — surface_format musi być zgodny z swapchain
        let surface_format = render_ctx.surface_format();
        let egui_renderer = EguiRenderer::new(
            &render_ctx.device,
            surface_format,
            &window,
        );

        // Jeden LayerStack na obiekt sceny
        let layer_stacks: Vec<LayerStack> = scene.objects.iter()
            .map(|_| LayerStack::default())
            .collect();

        let controller = CameraController::new(6.0, -0.6, 0.4);

        self.state = Some(AppState {
            window, render_ctx, scene, controller,
            egui_renderer,
            ui_state: UiState::default(),
            layer_stacks,
            selected_object: 0,
            fps_timer:  Instant::now(),
            fps_frames: 0,
            fps_last:   0.0,
        });

        info!("Malachite initialized");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _wid: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        // ── egui input ────────────────────────────────────────────────────────
        // egui_renderer.handle_input zwraca true jeśli egui pochłonął event.
        // W takim przypadku NIE przekazujemy eventu do CameraController.
        let egui_consumed = state.egui_renderer.handle_input(&state.window, &event);

        match &event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::Resized(sz) => {
                if sz.width > 0 && sz.height > 0 {
                    state.render_ctx.resize(*sz);
                    state.scene.camera.set_aspect(sz.width as f32 / sz.height as f32);
                }
            }

            WindowEvent::RedrawRequested => {
                // Aktualizacja kamery — tylko jeśli egui nie ma myszy
                if !state.egui_renderer.wants_pointer_input() {
                    state.controller.update_camera(&mut state.scene.camera);
                }
                state.render_ctx.env_rotation_yaw = state.controller.env_rotation_yaw;

                // FPS counter
                state.fps_frames += 1;
                if state.fps_timer.elapsed().as_secs_f32() >= 0.5 {
                    state.fps_last   = state.fps_frames as f32 / state.fps_timer.elapsed().as_secs_f32();
                    state.fps_frames = 0;
                    state.fps_timer  = Instant::now();
                    state.window.set_title(&format!(
                        "Malachite  |  {:.0} fps  |  layers: {}",
                        state.fps_last,
                        state.layer_stacks.get(state.selected_object)
                            .map(|s| s.layers.len()).unwrap_or(0)
                    ));
                }

                // ── egui frame ────────────────────────────────────────────────
                // begin_frame zwraca &Context; używamy go do rysowania UI
                {
                    let ctx = state.egui_renderer.begin_frame(&state.window).clone();

                    // Pobierz aktywny layer stack dla wybranego obiektu
                    let layer_stack = state.layer_stacks
                        .get_mut(state.selected_object)
                        .expect("layer stack out of bounds");

                    let output = ui::draw(
                        &ctx,
                        &mut state.ui_state,
                        layer_stack,
                        &mut state.render_ctx.post_params,
                        &mut state.scene,
                        &mut state.render_ctx.shadow.settings,
                    );

                    // ── Aplikuj warstwy → materiał obiektu ────────────────────
                    if output.layers_dirty {
                        let obj_idx = state.selected_object;
                        if let Some(stack) = state.layer_stacks.get(obj_idx) {
                            let flat = stack.flatten(obj_idx);
                            if let Some(obj) = state.scene.objects.get_mut(obj_idx) {
                                obj.material = flat.to_material();
                                // Odbuduj GPU object z nowym materiałem
                                state.render_ctx.rebuild_object(obj_idx, &obj.mesh, &obj.material);
                            }
                        }
                    }
                }

                // ── Render PBR ────────────────────────────────────────────────
                match state.render_ctx.render_with_egui(
                    &state.scene,
                    &mut state.egui_renderer,
                    &state.window,
                ) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let sz = state.render_ctx.size();
                        state.render_ctx.resize(sz);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => { event_loop.exit(); }
                    Err(e) => log::warn!("Surface error: {e:?}"),
                }
                state.window.request_redraw();
            }

            // Eventy klawiatury/myszy — tylko jeśli egui nie pochłonął
            WindowEvent::KeyboardInput { event, .. } if !egui_consumed => {
                state.controller.process_keyboard(event, &mut state.scene.camera);
            }
            WindowEvent::MouseInput { button, state: btn_state, .. } if !egui_consumed => {
                state.controller.process_mouse_button(button, btn_state);
            }
            WindowEvent::CursorMoved { position, .. } if !egui_consumed => {
                state.controller.process_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                state.controller.process_scroll(delta);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _did: DeviceId, _event: DeviceEvent) {}
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting Malachite PBR Engine...");
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::new())?;
    Ok(())
}