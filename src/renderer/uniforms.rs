use bytemuck::{Pod, Zeroable};
use crate::scene::{Camera, Scene};
use crate::scene::light::LightKind;

/// Environment rotation uniform — passed to PBR + skybox shaders.
/// Contains a 3×3 rotation matrix (padded to mat4x4 for WGSL alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EnvUniform {
    /// Column-major mat3x3 rotation, padded as mat4x4 (rows 0-2 used, w=0).
    pub rotation: [[f32; 4]; 4],
}

impl EnvUniform {
    pub fn identity() -> Self {
        Self {
            rotation: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Build from a yaw rotation around Y axis (radians).
    pub fn from_yaw(yaw: f32) -> Self {
        let (s, c) = yaw.sin_cos();
        Self {
            rotation: [
                [ c,  0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0,  c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj:  [[f32; 4]; 4],
    pub view:       [[f32; 4]; 4],
    pub proj:       [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj:  camera.view_proj().to_cols_array_2d(),
            view:       camera.view().to_cols_array_2d(),
            proj:       camera.proj().to_cols_array_2d(),
            camera_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ObjectUniform {
    pub model:         [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniform {
    pub base_color:            [f32; 4],  // offset  0
    pub metallic_roughness_ao: [f32; 4],  // offset 16  [metallic, roughness, ao_strength, normal_scale]
    pub emissive:              [f32; 4],  // offset 32  [r, g, b, strength]
    // Clearcoat — [clearcoat, clearcoat_roughness, _pad, _pad]  offset 48
    pub clearcoat_params:      [f32; 4],
    // Sheen    — [sheen_r, sheen_g, sheen_b, sheen_roughness]   offset 64
    pub sheen_params:          [f32; 4],
    /// [anisotropy -1..1, aniso_rotation_rad, subsurface 0..1, parallax_height_scale 0=off]
    pub advanced:              [f32; 4],
    /// [ior, transmission 0..1, thickness (Beer), specular_factor KHR]
    pub ext_ior_trans:         [f32; 4],
    /// [specular_color RGB, _pad]
    pub ext_specular_color:    [f32; 4],
    /// [attenuation_color RGB (extinction), strength]
    pub ext_attenuation:       [f32; 4],
    /// [iridescence, film_ior, thickness_nm, _pad]
    pub ext_iridescence:       [f32; 4],
}

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    pub position_or_dir: [f32; 4],
    pub spot_dir:        [f32; 4],
    pub color_intensity: [f32; 4],
    pub params:          [f32; 4],
}

impl GpuLight {
    pub fn from_light(light: &crate::scene::Light) -> Self {
        let kind_id = match light.kind {
            LightKind::Directional => 0.0,
            LightKind::Point       => 1.0,
            LightKind::Spot { .. } => 2.0,
        };
        let (inner, outer) = match light.kind {
            LightKind::Spot { inner_angle, outer_angle } => (inner_angle, outer_angle),
            _ => (0.0, 0.0),
        };
        let v = light.spot_direction;
        let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        let (sx, sy, sz) = if len > 1e-6 {
            (v.x / len, v.y / len, v.z / len)
        } else {
            (0.0_f32, -1.0_f32, 0.0_f32)
        };
        Self {
            position_or_dir: [
                light.position_or_direction.x,
                light.position_or_direction.y,
                light.position_or_direction.z,
                0.0,
            ],
            spot_dir: [sx, sy, sz, 0.0],
            color_intensity: [light.color.x, light.color.y, light.color.z, light.intensity],
            params:          [kind_id, light.range, inner, outer],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightArrayUniform {
    pub lights:        [GpuLight; MAX_LIGHTS],
    pub count_and_pad: [u32; 4],
    pub ambient:       [f32; 4],
}

impl LightArrayUniform {
    pub fn from_scene(scene: &Scene) -> Self {
        let mut lights = [GpuLight::zeroed(); MAX_LIGHTS];
        let count = scene.lights.len().min(MAX_LIGHTS);
        for (i, light) in scene.lights.iter().take(MAX_LIGHTS).enumerate() {
            if light.enabled { lights[i] = GpuLight::from_light(light); }
        }
        Self {
            lights,
            count_and_pad: [count as u32, 0, 0, 0],
            ambient: [
                scene.ambient_color.x * scene.ambient_intensity,
                scene.ambient_color.y * scene.ambient_intensity,
                scene.ambient_color.z * scene.ambient_intensity,
                1.0,
            ],
        }
    }
}