// GBuffer — world normals + metallic/roughness dla SSAO i SSR
// Używa tego samego material_bgl co PBR (binding 0=uniform, 4=metallic_roughness, 2=sampler)

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ObjectUniform {
    model:         mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> object: ObjectUniform;

// Dopasowane do material_bgl:
// binding 0 = MaterialUniform (nie używamy, ale musi być w layoucie)
struct MaterialUniform { base_color: vec4<f32>, mr_ao: vec4<f32>, emissive: vec4<f32>, }
@group(2) @binding(0) var<uniform> _mat:              MaterialUniform;
@group(2) @binding(2) var s_main:                     sampler;
@group(2) @binding(4) var t_metallic_roughness:       texture_2d<f32>;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec2<f32>,
}
struct VertOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       uv:           vec2<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    let w4 = object.model * vec4<f32>(in.position, 1.0);
    out.clip_pos     = camera.view_proj * w4;
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv           = in.uv;
    return out;
}

struct GbufferOut {
    @location(0) normals:            vec4<f32>,
    @location(1) metallic_roughness: vec4<f32>,
}

@fragment
fn fs_main(in: VertOut) -> GbufferOut {
    let n  = normalize(in.world_normal) * 0.5 + 0.5;
    let mr = textureSample(t_metallic_roughness, s_main, in.uv);
    var out: GbufferOut;
    out.normals            = vec4<f32>(n, 1.0);
    out.metallic_roughness = vec4<f32>(mr.b, mr.g, 0.0, 1.0);
    return out;
}
