// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Shadow Map Shader – depth-only pass for directional light                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
}

struct ObjectUniform {
    model:         mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: ShadowUniform;
@group(1) @binding(0) var<uniform> object: ObjectUniform;

@vertex
fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow.light_view_proj * object.model * vec4<f32>(position, 1.0);
}
