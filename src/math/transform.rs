use glam::{Mat4, Quat, Vec3};

/// Composable TRS (Translation-Rotation-Scale) transform.
/// Stores components separately to avoid precision loss from repeated matrix decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Cached world matrix – rebuilt lazily when dirty
    cached_matrix: Mat4,
    dirty: bool,
}

impl Transform {
    pub fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        let matrix = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        Self {
            translation,
            rotation,
            scale,
            cached_matrix: matrix,
            dirty: false,
        }
    }

    pub fn identity() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            cached_matrix: Mat4::IDENTITY,
            dirty: false,
        }
    }

    pub fn set_translation(&mut self, t: Vec3) {
        self.translation = t;
        self.dirty = true;
    }

    pub fn set_rotation(&mut self, r: Quat) {
        self.rotation = r;
        self.dirty = true;
    }

    pub fn set_scale(&mut self, s: Vec3) {
        self.scale = s;
        self.dirty = true;
    }

    /// Returns the world matrix, rebuilding it only if dirty.
    pub fn matrix(&mut self) -> Mat4 {
        if self.dirty {
            self.cached_matrix = Mat4::from_scale_rotation_translation(
                self.scale,
                self.rotation,
                self.translation,
            );
            self.dirty = false;
        }
        self.cached_matrix
    }

    /// Normal matrix = transpose(inverse(model)) – used in shaders.
    pub fn normal_matrix(&mut self) -> Mat4 {
        self.matrix().inverse().transpose()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}
