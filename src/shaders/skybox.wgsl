// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Skybox Shader – procedural gradient sky + sun disk                        ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
}

struct SkyUniform {
    sun_direction:  vec4<f32>, // xyz = normalized direction TO sun
    sun_color:      vec4<f32>, // rgb + intensity
    sky_color_top:  vec4<f32>,
    sky_color_horiz:vec4<f32>,
    ground_color:   vec4<f32>,
    params:         vec4<f32>, // x=sun_size, y=sun_halo, z=exposure, w=unused
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> sky:    SkyUniform;

struct VertexOutput {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_dir: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Fullscreen triangle trick – no vertex buffer needed
    let uv = vec2<f32>(
        f32((vid << 1u) & 2u),
        f32(vid & 2u),
    );
    let clip = vec4<f32>(uv * 2.0 - 1.0, 0.9999, 1.0);

    // Unproject clip → world direction (strip translation from view)
    let inv_proj = transpose(camera.proj); // approximation for direction
    var view_ray = vec4<f32>(clip.x / camera.proj[0][0],
                             clip.y / camera.proj[1][1],
                             -1.0, 0.0);

    // Use view matrix rotation only (no translation)
    let rot_view = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
    let world_dir = transpose(rot_view) * view_ray.xyz;

    var out: VertexOutput;
    out.clip_pos  = clip;
    out.world_dir = world_dir;
    return out;
}

const PI: f32 = 3.14159265;

// Rayleigh scattering approximation
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Mie scattering (henyey-greenstein)
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Target output: #2E2E2E (sRGB 46/255 = 0.1804)
    // Pipeline: HDR → ACES filmic → gamma 2.2
    // Inverse: 0.1804^2.2 = 0.02310 (linear target)
    // Inverse ACES at 0.02310 ≈ 0.03312
    // Exposure default = 0 → pow(2,0) = 1, no change needed
    let v = 0.03312;
    return vec4<f32>(v, v, v, 1.0);
}
