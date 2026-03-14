mod renderer;
mod scene;
mod assets;
mod math;
mod utils;

use anyhow::Result;
use log::info;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;

use renderer::RenderContext;
use scene::Scene;
use scene::camera::CameraController;

struct App { state: Option<AppState> }

struct AppState {
    window:      Arc<Window>,
    render_ctx:  RenderContext,
    scene:       Scene,
    controller:  CameraController,
    // FPS tracking
    fps_timer:   Instant,
    fps_frames:  u32,
    fps_last:    f32,
}

impl App { fn new() -> Self { Self { state: None } } }

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("PBR Engine")
                    .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
            ).expect("window")
        );

        let mut render_ctx = pollster::block_on(RenderContext::new(Arc::clone(&window)))
            .expect("renderer");

        // Próbuj wczytać plik HDR z katalogu roboczego.
        // Jeśli go nie ma, renderer używa proceduralnego fallback sky (neutral overcast).
        // Aby aktywować pełne IBL: umieść plik .hdr w assets/ lub podaj inną ścieżkę.
        match render_ctx.load_hdr("assets/ticknock_01_4k.hdr") {
            Ok(_)  => log::info!("HDR loaded — pełne IBL aktywne."),
            Err(e) => log::warn!(
                "Brak pliku HDR ({e}). \
                 Używam fallback sky cubemap (neutral overcast). \
                 Aby uzyskać pełne IBL, umieść plik .hdr w assets/ i zaktualizuj ścieżkę."
            ),
        }

        let scene      = Scene::default_scene();
        let controller = CameraController::new(6.0, -0.6, 0.4);

        self.state = Some(AppState {
            window, render_ctx, scene, controller,
            fps_timer:  Instant::now(),
            fps_frames: 0,
            fps_last:   0.0,
        });
        info!("PBR Engine initialized");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _wid: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::Resized(sz) => {
                if sz.width > 0 && sz.height > 0 {
                    state.render_ctx.resize(sz);
                    state.scene.camera.set_aspect(sz.width as f32 / sz.height as f32);
                }
            }

            WindowEvent::RedrawRequested => {
                state.controller.update_camera(&mut state.scene.camera);
                // Sync environment rotation from controller to renderer
                state.render_ctx.env_rotation_yaw = state.controller.env_rotation_yaw;

                // FPS counter
                state.fps_frames += 1;
                let elapsed = state.fps_timer.elapsed().as_secs_f32();
                if elapsed >= 0.5 {
                    state.fps_last   = state.fps_frames as f32 / elapsed;
                    state.fps_frames = 0;
                    state.fps_timer  = Instant::now();

                    let cam  = &state.scene.camera;
                    let ctrl = &state.controller;
                    let title = format!(
                        "PBR Engine  |  {:.0} fps  |  dist:{:.2}  yaw:{:.1}°  pitch:{:.1}°  env:{:.1}°  |  LMB:orbit  Shift+LMB:env  MMB:pan  Scroll:zoom  F:reset  R:reset env",
                        state.fps_last,
                        ctrl.radius,
                        ctrl.yaw.to_degrees(),
                        ctrl.pitch.to_degrees(),
                        ctrl.env_rotation_yaw.to_degrees(),
                    );
                    state.window.set_title(&title);
                }

                match state.render_ctx.render(&state.scene) {
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

            WindowEvent::KeyboardInput { event, .. } => {
                state.controller.process_keyboard(&event, &mut state.scene.camera);
            }
            WindowEvent::MouseInput { button, state: btn_state, .. } => {
                state.controller.process_mouse_button(&button, &btn_state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.controller.process_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                state.controller.process_scroll(&delta);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _did: DeviceId, _event: DeviceEvent) {}
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting PBR Engine...");
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::new())?;
    Ok(())
}