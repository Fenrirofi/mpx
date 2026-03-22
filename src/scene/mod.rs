pub mod camera;
pub mod light;
pub mod object;

pub use camera::{Camera, CameraController};
pub use light::Light;
pub use object::SceneObject;

use glam::Vec3;

pub struct Scene {
    pub camera:            Camera,
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
            // Złoto — anizotropia + height (efekt szczotkowanego metalu)
            SceneObject {
                mesh: Mesh::uv_sphere(64, 64),
                material: {
                    let mut m = Material::metallic(Vec4::new(0.96, 0.78, 0.22, 1.0), 0.18);
                    m.detail_maps = true;
                    m.normal_scale = 1.2;
                    m.height_scale = 0.11;
                    m.anisotropy = 0.9;
                    m.anisotropy_rotation = 1.2;
                    m.ao_strength = 0.95;
                    m
                },
                transform: Transform::identity(),
            },
            // Niebieski — clear coat (lakier jak w SP)
            SceneObject {
                mesh: Mesh::uv_sphere(64, 64),
                material: {
                    let mut m = Material::dielectric(Vec4::new(0.08, 0.32, 0.92, 1.0), 0.42);
                    m.detail_maps = true;
                    m.normal_scale = 1.05;
                    m.height_scale = 0.07;
                    m.clearcoat = 0.55;
                    m.clearcoat_roughness = 0.06;
                    m.ao_strength = 0.9;
                    m
                },
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(2.5, 0.0, 0.0));
                    t
                },
            },
            // Czerwony — subsurface + lekki sheen (kolor „miękki”)
            SceneObject {
                mesh: Mesh::uv_sphere(64, 64),
                material: {
                    let mut m = Material::dielectric(Vec4::new(0.88, 0.1, 0.07, 1.0), 0.22);
                    m.detail_maps = true;
                    m.normal_scale = 1.1;
                    m.height_scale = 0.06;
                    m.subsurface = 0.52;
                    m.sheen_color = [0.35, 0.12, 0.08];
                    m.sheen_roughness = 0.55;
                    m.ao_strength = 0.92;
                    m.iridescence = 0.42;
                    m.iridescence_ior = 1.38;
                    m.iridescence_thickness_nm = 420.0;
                    m
                },
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(-2.5, 0.0, 0.0));
                    t
                },
            },
            // Szkło — transmission / IOR (przybliżenie kubemapą)
            SceneObject {
                mesh: Mesh::uv_sphere(56, 56),
                material: {
                    let mut m = Material::dielectric(Vec4::new(0.72, 0.86, 0.94, 1.0), 0.028);
                    m.detail_maps = true;
                    m.ior = 1.52;
                    m.transmission = 0.91;
                    m.thickness = 0.35;
                    m.specular_factor = 1.0;
                    m.attenuation_color = [0.88, 0.96, 1.0];
                    m.attenuation_strength = 1.5;
                    m.normal_scale = 0.4;
                    m.height_scale = 0.02;
                    m.ao_strength = 0.45;
                    m
                },
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(0.0, 0.22, 2.35));
                    t.set_scale(vec3(0.42, 0.42, 0.42));
                    t
                },
            },
            // Podłoga — subtelny beton / mikro-roughness
            SceneObject {
                mesh: Mesh::cube(),
                material: {
                    let mut m = Material::dielectric(Vec4::new(0.16, 0.17, 0.2, 1.0), 0.88);
                    m.detail_maps = true;
                    m.normal_scale = 0.5;
                    m.height_scale = 0.038;
                    m.ao_strength = 0.85;
                    m
                },
                transform: {
                    let mut t = Transform::identity();
                    t.set_translation(vec3(0.0, -1.2, 0.0));
                    t.set_scale(vec3(8.0, 0.2, 8.0));
                    t
                },
            },
        ];

        let spot_pos = vec3(6.2, 6.5, 2.2);
        let spot_dir = (Vec3::ZERO - spot_pos).normalize();
        let lights = vec![
            Light::directional(vec3(0.4, 0.8, 0.3).normalize(), Vec3::new(1.0, 0.92, 0.76), 4.0),
            Light::point(vec3(-6.0, 5.0, 3.0), Vec3::new(0.3, 0.5, 1.0), 40.0),
            Light::point(vec3(4.0, 3.0, -5.0), Vec3::new(1.0, 0.7, 0.4), 20.0),
            Light::spot(
                spot_pos,
                spot_dir,
                0.35,
                0.52,
                Vec3::new(1.0, 0.94, 0.86),
                125.0,
                26.0,
            ),
        ];

        Self {
            camera,
            objects,
            lights,
            ambient_color:     Vec3::new(0.08, 0.12, 0.22),
            ambient_intensity: 1.4,
        }
    }
}