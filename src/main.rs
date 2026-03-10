mod assets;
mod math;
mod renderer;
mod scene;
mod utils;

use anyhow::Result;
use log::info;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use renderer::RenderContext;
use scene::Scene;

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    render_ctx: RenderContext,
    scene: Scene,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("PBR Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        let mut render_ctx = pollster::block_on(RenderContext::new(Arc::clone(&window)))
            .expect("Failed to initialize renderer");

        let scene = Scene::default_scene();

        render_ctx.load_hdr("assets/ticknock_01_4k.hdr").unwrap();

        self.state = Some(AppState {
            window,
            render_ctx,
            scene,
        });

        info!("PBR Engine initialized successfully");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                info!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.render_ctx.resize(new_size);
                }
            }
            WindowEvent::RedrawRequested => {
                state.scene.update();

                match state.render_ctx.render(&state.scene) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.render_ctx.size();
                        state.render_ctx.resize(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory!");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Surface error: {:?}", e),
                }

                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                state.scene.camera_controller.process_keyboard(&event);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        if let DeviceEvent::MouseMotion { delta } = event {
            state
                .scene
                .camera_controller
                .process_mouse(delta.0, delta.1);
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting PBR Engine...");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
