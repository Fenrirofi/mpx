use glam::Vec4;

#[derive(Debug, Clone)]
pub struct Material {
    pub base_color:  Vec4,
    pub metallic:    f32,
    pub roughness:   f32,
    pub emissive:    [f32; 3],
    pub normal_scale: f32,
    pub ao_strength: f32,
    pub base_color_texture:        Option<String>,
    pub normal_texture:            Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub emissive_texture:          Option<String>,
    pub ao_texture:                Option<String>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color:  Vec4::new(0.8, 0.8, 0.8, 1.0),
            metallic:    0.0,
            roughness:   0.5,
            emissive:    [0.0; 3],
            normal_scale: 1.0,
            ao_strength: 1.0,
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            ao_texture: None,
        }
    }
}

impl Material {
    pub fn metallic(base_color: Vec4, roughness: f32) -> Self {
        Self { base_color, metallic: 1.0, roughness, ..Default::default() }
    }
    pub fn dielectric(base_color: Vec4, roughness: f32) -> Self {
        Self { base_color, metallic: 0.0, roughness, ..Default::default() }
    }
}