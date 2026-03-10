use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

/// Interleaved vertex layout – matches the WGSL struct in pbr.wgsl.
/// `repr(C)` + bytemuck traits guarantee safe GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4], // xyz = tangent, w = bitangent sign
    pub uv: [f32; 2],
}

impl Vertex {
    /// wgpu vertex buffer layout descriptor – keep in sync with WGSL.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // tangent
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }

    pub fn new(
        position: Vec3,
        normal: Vec3,
        tangent: Vec4,
        uv: Vec2,
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            uv: uv.to_array(),
        }
    }
}

/// CPU-side mesh representation.
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Bounding sphere (center xyz, radius w) for culling.
    pub bounding_sphere: Vec4,
}

impl Mesh {
    /// Build a simple UV sphere with mikktspace-style tangents placeholder.
    pub fn uv_sphere(stacks: u32, slices: u32) -> Self {
        use std::f32::consts::PI;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for stack in 0..=stacks {
            let phi = PI * stack as f32 / stacks as f32;
            for slice in 0..=slices {
                let theta = 2.0 * PI * slice as f32 / slices as f32;
                let sin_phi = phi.sin();
                let x = sin_phi * theta.cos();
                let y = phi.cos();
                let z = sin_phi * theta.sin();
                let normal = Vec3::new(x, y, z);
                let tangent = Vec4::new(-theta.sin(), 0.0, theta.cos(), 1.0);
                let uv = Vec2::new(slice as f32 / slices as f32, stack as f32 / stacks as f32);
                vertices.push(Vertex::new(normal, normal, tangent, uv));
            }
        }

        for stack in 0..stacks {
            for slice in 0..slices {
                let first = stack * (slices + 1) + slice;
                let second = first + slices + 1;
                indices.extend_from_slice(&[first, second, first + 1, second, second + 1, first + 1]);
            }
        }

        Self {
            vertices,
            indices,
            bounding_sphere: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Axis-aligned box mesh.
    pub fn cube() -> Self {
        let positions: [[f32; 3]; 24] = [
            // +Z face
            [-1., -1.,  1.], [ 1., -1.,  1.], [ 1.,  1.,  1.], [-1.,  1.,  1.],
            // -Z face
            [ 1., -1., -1.], [-1., -1., -1.], [-1.,  1., -1.], [ 1.,  1., -1.],
            // +X face
            [ 1., -1.,  1.], [ 1., -1., -1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],
            // -X face
            [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1.,  1.], [-1.,  1., -1.],
            // +Y face
            [-1.,  1.,  1.], [ 1.,  1.,  1.], [ 1.,  1., -1.], [-1.,  1., -1.],
            // -Y face
            [-1., -1., -1.], [ 1., -1., -1.], [ 1., -1.,  1.], [-1., -1.,  1.],
        ];
        let normals: [[f32; 3]; 6] = [
            [0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
        ];
        let uvs: [[f32; 2]; 4] = [[0., 1.], [1., 1.], [1., 0.], [0., 0.]];

        let mut vertices = Vec::with_capacity(24);
        for face in 0..6usize {
            for vert in 0..4usize {
                let pos = Vec3::from(positions[face * 4 + vert]);
                let nor = Vec3::from(normals[face]);
                let uv = Vec2::from(uvs[vert]);
                let tan = Vec4::new(1., 0., 0., 1.);
                vertices.push(Vertex::new(pos, nor, tan, uv));
            }
        }

        let mut indices = Vec::with_capacity(36);
        for face in 0..6u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self {
            vertices,
            indices,
            bounding_sphere: Vec4::new(0., 0., 0., 1.732),
        }
    }
}
