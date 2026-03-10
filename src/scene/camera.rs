use glam::{Mat4, Vec3};
use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

/// Perspective camera with cached view/projection matrices.
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,
    dirty: bool,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        let fov = std::f32::consts::FRAC_PI_4;
        let near = 0.1;
        let far = 500.0;
        let view = Mat4::look_at_rh(position, target, Vec3::Y);
        let proj = Mat4::perspective_rh(fov, aspect, near, far);
        Self {
            position,
            target,
            up: Vec3::Y,
            fov_y_radians: fov,
            aspect,
            near,
            far,
            view,
            proj,
            view_proj: proj * view,
            dirty: false,
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.dirty = true;
    }

    /// Rebuild matrices only when dirty.
    pub fn rebuild_if_dirty(&mut self) {
        if self.dirty {
            self.view = Mat4::look_at_rh(self.position, self.target, self.up);
            self.proj = Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far);
            self.view_proj = self.proj * self.view;
            self.dirty = false;
        }
    }

    pub fn view(&self) -> Mat4 { self.view }
    pub fn proj(&self) -> Mat4 { self.proj }
    pub fn view_proj(&self) -> Mat4 { self.view_proj }
}

/// Simple FPS-style orbit camera controller.
pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    yaw: f32,   // radians
    pitch: f32, // radians
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    delta_x: f64,
    delta_y: f64,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.3,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            delta_x: 0.0,
            delta_y: 0.0,
        }
    }

    pub fn process_keyboard(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::KeyW | KeyCode::ArrowUp => self.forward = pressed,
                KeyCode::KeyS | KeyCode::ArrowDown => self.backward = pressed,
                KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
                KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
                KeyCode::KeyE | KeyCode::Space => self.up = pressed,
                KeyCode::KeyQ => self.down = pressed,
                _ => {}
            }
        }
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        self.delta_x += dx;
        self.delta_y += dy;
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let dt = 1.0 / 60.0_f32; // fixed step; replace with actual delta

        // Apply mouse look
        self.yaw += self.delta_x as f32 * self.sensitivity * dt;
        self.pitch -= self.delta_y as f32 * self.sensitivity * dt;
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self.delta_x = 0.0;
        self.delta_y = 0.0;

        // Compute forward/right/up vectors from yaw+pitch
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();
        let right = forward.cross(Vec3::Y).normalize();

        // Move camera
        let mut velocity = Vec3::ZERO;
        if self.forward  { velocity += forward; }
        if self.backward { velocity -= forward; }
        if self.right    { velocity += right; }
        if self.left     { velocity -= right; }
        if self.up       { velocity += Vec3::Y; }
        if self.down     { velocity -= Vec3::Y; }

        if velocity.length_squared() > 0.0 {
            camera.position += velocity.normalize() * self.speed * dt;
            camera.dirty = true;
        }

        // Always update target from yaw/pitch
        camera.target = camera.position + forward;
        camera.dirty = true;

        camera.rebuild_if_dirty();
    }
}
