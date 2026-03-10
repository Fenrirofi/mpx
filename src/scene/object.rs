use crate::assets::{Material, Mesh};
use crate::math::Transform;

pub struct SceneObject {
    pub mesh:      Mesh,
    pub material:  Material,
    pub transform: Transform,
}