pub mod camera;
pub mod light;
pub mod object;

pub use camera::{Camera, CameraController};
pub use light::Light;
pub use object::SceneObject;

use glam::Vec3;

pub struct Scene {
    pub camera:            Camera,
    pub camera_controller: CameraController,
    pub objects:           Vec<SceneObject>,
    pub lights:            Vec<Light>,
    pub ambient_color:     Vec3,
    pub ambient_intensity: f32,
}

impl Scene {
    pub fn default_scene() -> Self {
        use crate::assets::{Material, Mesh};
        use crate::math::Transform;
        use glam::{Vec4, vec3};

        let camera = Camera::new(vec3(0.0, 2.0, 6.0), vec3(0.0, 0.0, 0.0), 1280.0 / 720.0);

        let objects = vec![
            // Gold metallic sphere
            SceneObject {
                mesh:      Mesh::uv_sphere(32, 32),
                material:  Material::metallic(Vec4::new(1.0, 0.78, 0.34, 1.0), 0.1),
                transform: Transform::identity(),
            },
            // Rough dielectric sphere
            SceneObject {
                mesh:      Mesh::uv_sphere(32, 32),
                material:  Material::dielectric(Vec4::new(0.2, 0.4, 0.9, 1.0), 0.6),
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(2.5, 0.0, 0.0));
                    t
                },
            },
            // Smooth plastic sphere
            SceneObject {
                mesh:      Mesh::uv_sphere(32, 32),
                material:  Material::dielectric(Vec4::new(0.9, 0.15, 0.1, 1.0), 0.2),
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(-2.5, 0.0, 0.0));
                    t
                },
            },
            // Floor
            SceneObject {
                mesh:     Mesh::cube(),
                material: Material::dielectric(Vec4::new(0.18, 0.18, 0.2, 1.0), 0.85),
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(0.0, -1.2, 0.0));
                    t.set_scale(vec3(8.0, 0.2, 8.0));
                    t
                },
            },
        ];

        let lights = vec![
            // Sun – warm directional matched to skybox
            Light::directional(vec3(0.4, 0.8, 0.3).normalize(), Vec3::new(1.0, 0.92, 0.76), 8.0),
            // Cool sky fill
            Light::point(vec3(-6.0, 5.0, 3.0), Vec3::new(0.3, 0.5, 1.0), 80.0),
            // Warm rim
            Light::point(vec3(4.0, 3.0, -5.0), Vec3::new(1.0, 0.7, 0.4), 60.0),
        ];

        Self {
            camera,
            camera_controller: CameraController::new(4.0, 0.3),
            objects,
            lights,
            ambient_color:     Vec3::new(0.08, 0.12, 0.22),
            ambient_intensity: 1.4,
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
    }
}