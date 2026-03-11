use glam::{Mat4, Vec3};
use winit::{
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Camera {
    pub position:      Vec3,
    pub target:        Vec3,
    pub up:            Vec3,
    pub fov_y_radians: f32,
    pub aspect:        f32,
    pub near:          f32,
    pub far:           f32,
    view:              Mat4,
    proj:              Mat4,
    view_proj:         Mat4,
    pub dirty:         bool,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        let fov  = std::f32::consts::FRAC_PI_4;
        let near = 0.05;
        let far  = 1000.0;
        let view = Mat4::look_at_rh(position, target, Vec3::Y);
        let proj = Mat4::perspective_rh(fov, aspect, near, far);
        Self {
            position, target, up: Vec3::Y,
            fov_y_radians: fov, aspect, near, far,
            view, proj, view_proj: proj * view, dirty: false,
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) { self.aspect = aspect; self.dirty = true; }

    pub fn rebuild_if_dirty(&mut self) {
        if self.dirty {
            self.view      = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
            self.proj      = Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far);
            self.view_proj = self.proj * self.view;
            self.dirty     = false;
        }
    }

    pub fn view(&self)      -> Mat4 { self.view }
    pub fn proj(&self)      -> Mat4 { self.proj }
    pub fn view_proj(&self) -> Mat4 { self.view_proj }
}

#[derive(PartialEq)]
enum DragMode { None, Orbit, Pan, EnvRotate }

/// Orbital camera — Blender/Maya style:
///   LMB drag          → orbit around target
///   Shift + LMB drag  → rotate environment (like Substance Painter)
///   MMB drag          → pan
///   Scroll            → dolly zoom
///   Numpad 1/3/7      → front/right/top view
///   F                 → reset to origin
pub struct CameraController {
    pub radius: f32,
    pub yaw:    f32,
    pub pitch:  f32,
    pub env_rotation_yaw: f32,
    drag_mode:    DragMode,
    shift_held:   bool,
    last_x:       f64,
    last_y:       f64,
    pending_dx:   f64,
    pending_dy:   f64,
    pending_zoom: f32,
    orbit_speed:  f32,
    pan_speed:    f32,
    zoom_speed:   f32,
    env_rot_speed: f32,
}

impl CameraController {
    pub fn new(distance: f32, yaw: f32, pitch: f32) -> Self {
        Self {
            radius: distance, yaw, pitch,
            env_rotation_yaw: 0.0,
            drag_mode: DragMode::None,
            shift_held: false,
            last_x: 0.0, last_y: 0.0,
            pending_dx: 0.0, pending_dy: 0.0, pending_zoom: 0.0,
            orbit_speed: 0.005, pan_speed: 0.001, zoom_speed: 0.15,
            env_rot_speed: 0.005,
        }
    }

    pub fn process_mouse_button(&mut self, button: &MouseButton, state: &ElementState) {
        let pressed = *state == ElementState::Pressed;
        match button {
            MouseButton::Left => {
                if pressed {
                    // Shift+LMB → env rotate (Substance Painter style)
                    if self.shift_held {
                        self.drag_mode = DragMode::EnvRotate;
                    } else {
                        self.drag_mode = DragMode::Orbit;
                    }
                } else if matches!(self.drag_mode, DragMode::Orbit | DragMode::EnvRotate) {
                    self.drag_mode = DragMode::None;
                }
            }
            MouseButton::Middle => {
                if pressed { self.drag_mode = DragMode::Pan; }
                else if self.drag_mode == DragMode::Pan { self.drag_mode = DragMode::None; }
            }
            _ => {}
        }
    }

    pub fn process_cursor_moved(&mut self, x: f64, y: f64) {
        let dx = x - self.last_x;
        let dy = y - self.last_y;
        self.last_x = x;
        self.last_y = y;
        match self.drag_mode {
            DragMode::Orbit | DragMode::Pan | DragMode::EnvRotate => {
                self.pending_dx += dx;
                self.pending_dy += dy;
            }
            DragMode::None => {}
        }
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.pending_zoom += match delta {
            MouseScrollDelta::LineDelta(_, y)  => *y,
            MouseScrollDelta::PixelDelta(pos)  => pos.y as f32 * 0.05,
        };
    }

    pub fn process_keyboard(&mut self, event: &KeyEvent, camera: &mut Camera) {
        let pressed = event.state == ElementState::Pressed;
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                    self.shift_held = pressed;
                    // If mid-orbit and shift pressed, switch to env rotate
                    if pressed && self.drag_mode == DragMode::Orbit {
                        self.drag_mode = DragMode::EnvRotate;
                    } else if !pressed && self.drag_mode == DragMode::EnvRotate {
                        self.drag_mode = DragMode::None;
                    }
                }
                _ if !pressed => {}
                KeyCode::Numpad1 => { self.yaw = 0.0;                         self.pitch = 0.0; camera.dirty = true; }
                KeyCode::Numpad3 => { self.yaw = std::f32::consts::FRAC_PI_2; self.pitch = 0.0; camera.dirty = true; }
                KeyCode::Numpad7 => { self.pitch = std::f32::consts::FRAC_PI_2 - 0.01; camera.dirty = true; }
                KeyCode::Numpad9 => { self.pitch = -(std::f32::consts::FRAC_PI_2 - 0.01); camera.dirty = true; }
                KeyCode::KeyF    => { camera.target = Vec3::ZERO; self.radius = 5.0; camera.dirty = true; }
                KeyCode::KeyR    => { self.env_rotation_yaw = 0.0; } // reset env rotation
                KeyCode::NumpadAdd      => { self.pending_zoom += 2.0; }
                KeyCode::NumpadSubtract => { self.pending_zoom -= 2.0; }
                _ => {}
            }
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let dx = self.pending_dx as f32;
        let dy = self.pending_dy as f32;

        match self.drag_mode {
            DragMode::Orbit => {
                self.yaw   += dx * self.orbit_speed;
                self.pitch += dy * self.orbit_speed;
                self.pitch  = self.pitch.clamp(
                    -(std::f32::consts::FRAC_PI_2 - 0.02),
                     std::f32::consts::FRAC_PI_2 - 0.02,
                );
                camera.dirty = true;
            }
            DragMode::EnvRotate => {
                // Only horizontal drag affects env yaw (same as Substance Painter)
                self.env_rotation_yaw += dx * self.env_rot_speed;
            }
            DragMode::Pan => {
                let forward   = (camera.target - camera.position).normalize();
                let right     = forward.cross(Vec3::Y).normalize();
                let up        = right.cross(forward).normalize();
                let pan_scale = self.radius * self.pan_speed;
                camera.target -= right * dx * pan_scale;
                camera.target += up    * dy * pan_scale;
                camera.dirty   = true;
            }
            DragMode::None => {}
        }

        if self.pending_zoom.abs() > 0.001 {
            self.radius  *= (1.0 - self.pending_zoom * self.zoom_speed).max(0.01);
            self.radius   = self.radius.clamp(0.05, 500.0);
            camera.dirty  = true;
        }

        self.pending_dx   = 0.0;
        self.pending_dy   = 0.0;
        self.pending_zoom = 0.0;

        if camera.dirty {
            let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
            let (sy, cy) = (self.yaw.sin(),   self.yaw.cos());
            camera.position = camera.target + Vec3::new(cp * cy, sp, cp * sy) * self.radius;
            camera.rebuild_if_dirty();
        }
    }
}