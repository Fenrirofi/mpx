use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightKind {
    Directional,
    Point,
    Spot { inner_angle: f32, outer_angle: f32 },
}

#[derive(Debug, Clone)]
pub struct Light {
    pub position_or_direction: Vec3,
    pub color:     Vec3,
    pub intensity: f32,
    pub kind:      LightKind,
    pub range:     f32,
    pub enabled:   bool,
    /// Dla `Spot`: oś stożka (światło leci w tym kierunku), znormalizowany.
    pub spot_direction: Vec3,
}

impl Light {
    pub fn point(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position_or_direction: position,
            color,
            intensity,
            kind: LightKind::Point,
            range: 20.0,
            enabled: true,
            spot_direction: Vec3::new(0.0, -1.0, 0.0),
        }
    }
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position_or_direction: direction,
            color,
            intensity,
            kind: LightKind::Directional,
            range: f32::INFINITY,
            enabled: true,
            spot_direction: Vec3::new(0.0, -1.0, 0.0),
        }
    }

    /// Reflektor — kąty w radianach (wewnętrzny / zewnętrzny stożek).
    pub fn spot(
        position: Vec3,
        spot_direction: Vec3,
        inner_angle: f32,
        outer_angle: f32,
        color: Vec3,
        intensity: f32,
        range: f32,
    ) -> Self {
        Self {
            position_or_direction: position,
            color,
            intensity,
            kind: LightKind::Spot {
                inner_angle,
                outer_angle,
            },
            range,
            enabled: true,
            spot_direction,
        }
    }
}