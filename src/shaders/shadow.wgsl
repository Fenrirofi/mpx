// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Cascaded Shadow Maps (CSM) — depth-only pass                              ║
// ║  4 kaskady — każda renderowana osobno z tym samym shaderem                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

struct CsmUniform {
    light_view_proj: array<mat4x4<f32>, 4>,
    cascade_index:   u32,
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
}

struct ObjectUniform {
    model:         mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> csm:    CsmUniform;
@group(1) @binding(0) var<uniform> object: ObjectUniform;

@vertex
fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return csm.light_view_proj[csm.cascade_index] * object.model * vec4<f32>(position, 1.0);
}