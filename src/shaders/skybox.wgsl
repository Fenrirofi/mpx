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
    let dir        = normalize(in.world_dir);
    let sun_dir    = normalize(sky.sun_direction.xyz);
    let cos_theta  = dot(dir, sun_dir);
    let up         = vec3<f32>(0.0, 1.0, 0.0);
    let horizon_t  = clamp(dir.y * 4.0 + 0.5, 0.0, 1.0);
    let ground_t   = clamp(-dir.y * 8.0, 0.0, 1.0);

    // Sky gradient: ground → horizon → zenith
    var sky_color  = mix(sky.sky_color_horiz.rgb, sky.sky_color_top.rgb, pow(max(dir.y, 0.0), 0.5));
    sky_color      = mix(sky_color, sky.ground_color.rgb, ground_t);

    // Rayleigh: blue scatter near horizon
    let rayleigh   = rayleigh_phase(cos_theta);
    let scatter    = vec3<f32>(0.12, 0.24, 0.6) * rayleigh * 0.3 * max(dir.y + 0.2, 0.0);
    sky_color     += scatter;

    // Mie: halo around sun
    let sun_halo   = sky.params.y;
    let halo       = mie_phase(cos_theta, 0.85) * sky.sun_color.rgb * sky.sun_color.a * 0.002;
    sky_color     += halo;

    // Sun disk
    let sun_size   = sky.params.x;
    let sun_edge   = smoothstep(sun_size * 1.02, sun_size, cos_theta);
    let sun_limb   = smoothstep(sun_size * 0.98, sun_size * 1.0, cos_theta); // limb darkening
    let sun_bright = sky.sun_color.rgb * sky.sun_color.a * sun_edge * mix(0.7, 1.0, sun_limb);
    sky_color     += sun_bright;

    // Exposure + output (skybox feeds into HDR buffer, tone map in post)
    sky_color *= sky.params.z;

    return vec4<f32>(sky_color, 1.0);
}
