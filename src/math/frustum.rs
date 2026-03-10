use glam::{Mat4, Vec3, Vec4};

/// Six-plane view frustum for CPU-side culling.
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from the combined view-projection matrix.
    /// Uses Gribb-Hartmann method (row-major sign convention via glam).
    pub fn from_view_proj(vp: Mat4) -> Self {
        let m = vp.transpose();
        let cols = m.to_cols_array_2d();

        let row = |i: usize| Vec4::new(cols[0][i], cols[1][i], cols[2][i], cols[3][i]);

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let planes = [
            normalize_plane(r3 + r0), // left
            normalize_plane(r3 - r0), // right
            normalize_plane(r3 + r1), // bottom
            normalize_plane(r3 - r1), // top
            normalize_plane(r3 + r2), // near
            normalize_plane(r3 - r2), // far
        ];

        Self { planes }
    }

    /// AABB intersection test. Returns `false` if the box is fully outside any plane.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let positive = Vec3::new(
                if plane.x > 0.0 { max.x } else { min.x },
                if plane.y > 0.0 { max.y } else { min.y },
                if plane.z > 0.0 { max.z } else { min.z },
            );
            if plane.x * positive.x + plane.y * positive.y + plane.z * positive.z + plane.w < 0.0 {
                return false;
            }
        }
        true
    }

    /// Sphere intersection test.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist =
                plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false;
            }
        }
        true
    }
}

#[inline]
fn normalize_plane(p: Vec4) -> Vec4 {
    let inv_len = 1.0 / Vec3::new(p.x, p.y, p.z).length();
    p * inv_len
}
