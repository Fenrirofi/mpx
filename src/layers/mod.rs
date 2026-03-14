// ══════════════════════════════════════════════════════════════════════════════
//  src/layers/mod.rs — System warstw materiałów Malachite
//
//  Model danych:
//    LayerStack  → lista warstw (Vec<Layer>), indeks aktywnej warstwy
//    Layer       → nazwa, widoczność, opacity, blend mode, lista kanałów
//    Channel     → typ (Color/Roughness/Metallic/Normal/AO/Emissive), wartość
//
//  Warstwa nie jest bezpośrednio shaderem — to zestaw parametrów PBR który
//  jest "kompilowany" do MaterialUniform i wysyłany do GPU przez context.rs.
//  Blend mode działa na CPU (prosty lerp/mul między warstwami).
// ══════════════════════════════════════════════════════════════════════════════

use glam::{Vec3, Vec4};
use glam::FloatExt;

// ─────────────────────────────────────────────────────────────────────────────
// Channel — jeden kanał PBR w warstwie
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelKind {
    BaseColor,   // RGB kolor + alpha
    Roughness,   // f32 [0,1]
    Metallic,    // f32 [0,1]
    Normal,      // skalowanie mapy normalnych [0,2]
    AO,          // ambient occlusion strength [0,1]
    Emissive,    // kolor emisji (HDR, może być > 1)
}

impl ChannelKind {
    pub fn label(self) -> &'static str {
        match self {
            ChannelKind::BaseColor => "Base Color",
            ChannelKind::Roughness => "Roughness",
            ChannelKind::Metallic  => "Metallic",
            ChannelKind::Normal    => "Normal Scale",
            ChannelKind::AO        => "AO Strength",
            ChannelKind::Emissive  => "Emissive",
        }
    }

    pub fn all() -> &'static [ChannelKind] {
        &[
            ChannelKind::BaseColor,
            ChannelKind::Roughness,
            ChannelKind::Metallic,
            ChannelKind::Normal,
            ChannelKind::AO,
            ChannelKind::Emissive,
        ]
    }
}

/// Wartość kanału — w zależności od typu może to być kolor lub skalary.
#[derive(Debug, Clone)]
pub enum ChannelValue {
    Color(Vec4),   // base_color lub emissive (xyz=rgb, w=alpha/intensity)
    Scalar(f32),   // roughness, metallic, normal_scale, ao
}

impl ChannelValue {
    pub fn default_for(kind: ChannelKind) -> Self {
        match kind {
            ChannelKind::BaseColor => ChannelValue::Color(Vec4::new(0.8, 0.8, 0.8, 1.0)),
            ChannelKind::Roughness => ChannelValue::Scalar(0.5),
            ChannelKind::Metallic  => ChannelValue::Scalar(0.0),
            ChannelKind::Normal    => ChannelValue::Scalar(1.0),
            ChannelKind::AO        => ChannelValue::Scalar(1.0),
            ChannelKind::Emissive  => ChannelValue::Color(Vec4::new(0.0, 0.0, 0.0, 0.0)),
        }
    }

    pub fn as_color(&self) -> Vec4 {
        match self { ChannelValue::Color(c) => *c, ChannelValue::Scalar(s) => Vec4::splat(*s) }
    }

    pub fn as_scalar(&self) -> f32 {
        match self { ChannelValue::Scalar(s) => *s, ChannelValue::Color(c) => c.x }
    }
}

#[derive(Debug, Clone)]
pub struct Channel {
    pub kind:    ChannelKind,
    pub value:   ChannelValue,
    pub enabled: bool,
}

impl Channel {
    pub fn new(kind: ChannelKind) -> Self {
        Self { kind, value: ChannelValue::default_for(kind), enabled: true }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlendMode — jak warstwa miesza się z niższymi
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,     // lerp(base, layer, opacity)
    Multiply,   // base * layer
    Add,        // base + layer * opacity
    Overlay,    // soft light
}

impl BlendMode {
    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal   => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Add      => "Add",
            BlendMode::Overlay  => "Overlay",
        }
    }

    pub fn all() -> &'static [BlendMode] {
        &[BlendMode::Normal, BlendMode::Multiply, BlendMode::Add, BlendMode::Overlay]
    }

    /// Blenduj dwa skalary: base (dolna warstwa) i top (ta warstwa), t=opacity
    pub fn blend_scalar(self, base: f32, top: f32, t: f32) -> f32 {
        let blended = match self {
            BlendMode::Normal   => top,
            BlendMode::Multiply => base * top,
            BlendMode::Add      => base + top,
            BlendMode::Overlay  => {
                if base < 0.5 { 2.0 * base * top }
                else          { 1.0 - 2.0 * (1.0 - base) * (1.0 - top) }
            }
        };
        base + (blended - base) * t
    }

    /// Blenduj dwa kolory Vec3
    pub fn blend_color(self, base: Vec3, top: Vec3, t: f32) -> Vec3 {
        let b = match self {
            BlendMode::Normal   => top,
            BlendMode::Multiply => base * top,
            BlendMode::Add      => base + top,
            BlendMode::Overlay  => {
                Vec3::new(
                    if base.x < 0.5 { 2.0*base.x*top.x } else { 1.0-2.0*(1.0-base.x)*(1.0-top.x) },
                    if base.y < 0.5 { 2.0*base.y*top.y } else { 1.0-2.0*(1.0-base.y)*(1.0-top.y) },
                    if base.z < 0.5 { 2.0*base.z*top.z } else { 1.0-2.0*(1.0-base.z)*(1.0-top.z) },
                )
            }
        };
        base.lerp(b, t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Layer {
    pub name:       String,
    pub visible:    bool,
    pub opacity:    f32,            // [0, 1]
    pub blend_mode: BlendMode,
    pub channels:   Vec<Channel>,
    /// Indeks obiektu sceny (0..N) na który ta warstwa oddziałuje.
    /// None = oddziałuje na wszystkie obiekty.
    pub target_object: Option<usize>,
}

impl Layer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name:          name.into(),
            visible:       true,
            opacity:       1.0,
            blend_mode:    BlendMode::Normal,
            channels:      ChannelKind::all().iter().map(|&k| Channel::new(k)).collect(),
            target_object: None,
        }
    }

    pub fn channel(&self, kind: ChannelKind) -> Option<&Channel> {
        self.channels.iter().find(|c| c.kind == kind)
    }

    pub fn channel_mut(&mut self, kind: ChannelKind) -> Option<&mut Channel> {
        self.channels.iter_mut().find(|c| c.kind == kind)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerStack — cały stos warstw dla jednego obiektu lub sceny
// ─────────────────────────────────────────────────────────────────────────────

pub struct LayerStack {
    pub layers:       Vec<Layer>,
    pub active_index: Option<usize>,
}

impl Default for LayerStack {
    fn default() -> Self {
        let mut stack = Self { layers: Vec::new(), active_index: None };
        // Domyślna warstwa bazowa
        stack.push(Layer::new("Base Material"));
        stack
    }
}

impl LayerStack {
    pub fn push(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.active_index = Some(self.layers.len() - 1);
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.layers.len() {
            self.layers.remove(index);
            self.active_index = if self.layers.is_empty() {
                None
            } else {
                Some(self.active_index.unwrap_or(0).min(self.layers.len() - 1))
            };
        }
    }

    pub fn move_up(&mut self, index: usize) {
        if index + 1 < self.layers.len() {
            self.layers.swap(index, index + 1);
            if self.active_index == Some(index) { self.active_index = Some(index + 1); }
        }
    }

    pub fn move_down(&mut self, index: usize) {
        if index > 0 {
            self.layers.swap(index, index - 1);
            if self.active_index == Some(index) { self.active_index = Some(index - 1); }
        }
    }

    pub fn active(&self) -> Option<&Layer> {
        self.active_index.and_then(|i| self.layers.get(i))
    }

    pub fn active_mut(&mut self) -> Option<&mut Layer> {
        self.active_index.and_then(|i| self.layers.get_mut(i))
    }

    // ── Kompilacja stosu → FlatMaterial ──────────────────────────────────────
    /// Łączy wszystkie widoczne warstwy od dołu do góry w jeden zestaw
    /// parametrów PBR gotowych do wysłania jako MaterialUniform.
    pub fn flatten(&self, object_index: usize) -> FlatMaterial {
        let mut result = FlatMaterial::default();

        for layer in &self.layers {
            if !layer.visible { continue; }
            // Filtruj warstwy które nie dotyczą tego obiektu
            if let Some(target) = layer.target_object {
                if target != object_index { continue; }
            }
            let t = layer.opacity;

            for ch in &layer.channels {
                if !ch.enabled { continue; }
                match ch.kind {
                    ChannelKind::BaseColor => {
                        let top = ch.value.as_color();
                        result.base_color = Vec4::new(
                            layer.blend_mode.blend_color(
                                result.base_color.truncate(),
                                top.truncate(), t
                            ).x,
                            layer.blend_mode.blend_color(
                                result.base_color.truncate(),
                                top.truncate(), t
                            ).y,
                            layer.blend_mode.blend_color(
                                result.base_color.truncate(),
                                top.truncate(), t
                            ).z,
                            result.base_color.w,
                        );
                        // Prostsze podejście:
                        let blended = layer.blend_mode.blend_color(
                            result.base_color.truncate(), top.truncate(), t
                        );
                        result.base_color = Vec4::new(blended.x, blended.y, blended.z,
                            result.base_color.w.lerp(top.w, t));
                    }
                    ChannelKind::Roughness => {
                        result.roughness = layer.blend_mode.blend_scalar(
                            result.roughness, ch.value.as_scalar(), t
                        ).clamp(0.0, 1.0);
                    }
                    ChannelKind::Metallic => {
                        result.metallic = layer.blend_mode.blend_scalar(
                            result.metallic, ch.value.as_scalar(), t
                        ).clamp(0.0, 1.0);
                    }
                    ChannelKind::Normal => {
                        result.normal_scale = layer.blend_mode.blend_scalar(
                            result.normal_scale, ch.value.as_scalar(), t
                        ).clamp(0.0, 2.0);
                    }
                    ChannelKind::AO => {
                        result.ao_strength = layer.blend_mode.blend_scalar(
                            result.ao_strength, ch.value.as_scalar(), t
                        ).clamp(0.0, 1.0);
                    }
                    ChannelKind::Emissive => {
                        let top = ch.value.as_color();
                        let blended = layer.blend_mode.blend_color(
                            result.emissive, top.truncate(), t
                        );
                        result.emissive = blended;
                    }
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlatMaterial — wynik spłaszczenia stosu, gotowy do GPU
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FlatMaterial {
    pub base_color:   Vec4,
    pub roughness:    f32,
    pub metallic:     f32,
    pub normal_scale: f32,
    pub ao_strength:  f32,
    pub emissive:     Vec3,
}

impl Default for FlatMaterial {
    fn default() -> Self {
        Self {
            base_color:   Vec4::new(0.8, 0.8, 0.8, 1.0),
            roughness:    0.5,
            metallic:     0.0,
            normal_scale: 1.0,
            ao_strength:  1.0,
            emissive:     Vec3::ZERO,
        }
    }
}

impl FlatMaterial {
    /// Konwertuj do crate::assets::Material do przekazania do renderera
    pub fn to_material(&self) -> crate::assets::Material {
        crate::assets::Material {
            base_color:   self.base_color,
            metallic:     self.metallic,
            roughness:    self.roughness,
            emissive:     self.emissive.to_array(),
            normal_scale: self.normal_scale,
            ao_strength:  self.ao_strength,
            ..Default::default()
        }
    }
}