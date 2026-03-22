use glam::Vec4;

#[derive(Debug, Clone)]
pub struct Material {
    pub base_color:  Vec4,
    pub metallic:    f32,
    pub roughness:   f32,
    pub emissive:    [f32; 3],
    pub normal_scale: f32,
    pub ao_strength: f32,

    // Clearcoat: second GGX lobe with fixed IOR=1.5 (f0=0.04).
    // clearcoat=0.0 → feature disabled (zero cost in shader).
    pub clearcoat:           f32,
    pub clearcoat_roughness: f32,

    // Sheen: Charlie NDF for fabric retro-reflection.
    // sheen_color=[0,0,0] → feature disabled.
    pub sheen_color:     [f32; 3],
    pub sheen_roughness: f32,

    /// Kierunkowa anizotropia odbić (jak w SP / glTF anisotropy). 0 = izotropowo.
    pub anisotropy: f32,
    /// Obrót kierunku rys (rad), w przestrzeni stycznej.
    pub anisotropy_rotation: f32,
    /// „Miękki” dyfuz przy krawędziach (wrap + lekki backlight), 0 = wyłączone.
    pub subsurface: f32,
    /// Skala parallax / height (0 = wyłączone). Mapa wysokości: linear, 0.5 = neutral.
    pub height_scale: f32,
    /// Mnożnik HDR emisji (oprócz koloru `emissive`).
    pub emissive_strength: f32,

    /// Proceduralne mapy szczegółu (normal / MR / height) zamiast płaskich fallbacków.
    pub detail_maps: bool,

    /// IOR dielektryka (powietrze 1.0 → materiał). F0 z Fresnela Schlicka.
    pub ior: f32,
    /// Przenikanie światła (0–1), styl glTF KHR_materials_transmission — przybliżenie kubemapą.
    pub transmission: f32,
    /// Grubość dla absorpcji Beer–Lamberta (skala dowolna × `attenuation_strength`).
    pub thickness: f32,
    /// Mnożnik KHR_materials_specular (0–1).
    pub specular_factor: f32,
    /// Barwa odbić specularnych (dielektryk).
    pub specular_color: [f32; 3],
    /// Kolor absorpcji przy transmission.
    pub attenuation_color: [f32; 3],
    pub attenuation_strength: f32,
    /// Siła irydescencji (0–1), cienka warstwa — przybliżenie fazowe.
    pub iridescence: f32,
    pub iridescence_ior: f32,
    /// Reprezentatywna grubość filmu (nm) dla barwy tęczy.
    pub iridescence_thickness_nm: f32,

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
            clearcoat:           0.0,
            clearcoat_roughness: 0.0,
            sheen_color:         [0.0; 3],
            sheen_roughness:     0.0,
            anisotropy:          0.0,
            anisotropy_rotation: 0.0,
            subsurface:          0.0,
            height_scale:        0.0,
            emissive_strength:   1.0,
            detail_maps:         false,
            ior:                 1.5,
            transmission:        0.0,
            thickness:           0.0,
            specular_factor:     1.0,
            specular_color:      [1.0, 1.0, 1.0],
            attenuation_color:   [1.0, 1.0, 1.0],
            attenuation_strength: 1.0,
            iridescence:         0.0,
            iridescence_ior:     1.3,
            iridescence_thickness_nm: 400.0,
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