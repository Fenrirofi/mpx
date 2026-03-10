// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  SSAO – Screen Space Ambient Occlusion                                     ║
// ║  Pass 1 (cs_ssao):   compute raw AO into R8 texture                       ║
// ║  Pass 2 (cs_blur):   4-tap bilateral blur (depth-aware)                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

struct SsaoParams {
    proj:          mat4x4<f32>,
    inv_proj:      mat4x4<f32>,
    view:          mat4x4<f32>,
    radius:        f32,
    bias:          f32,
    strength:      f32,
    sample_count:  u32,
    noise_scale:   vec2<f32>,
    _pad:          vec2<f32>,
}

@group(0) @binding(0) var<uniform>          ssao_params:  SsaoParams;
@group(0) @binding(1) var t_depth:       texture_depth_2d;
@group(0) @binding(2) var t_normal:      texture_2d<f32>;
@group(0) @binding(3) var<storage, read> kernel: array<vec4<f32>>;
@group(0) @binding(4) var t_noise:       texture_2d<f32>;
@group(0) @binding(5) var s_nearest:     sampler;
@group(0) @binding(6) var out_ao:        texture_storage_2d<rgba8unorm, write>;

// Load depth using integer coords (depth textures can't use textureSampleLevel in compute)
fn load_depth(coords: vec2<i32>) -> f32 {
    return textureLoad(t_depth, coords, 0);
}
fn load_depth_uv(uv: vec2<f32>, dims: vec2<f32>) -> f32 {
    let c = vec2<i32>(uv * dims);
    let clamped = clamp(c, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    return textureLoad(t_depth, clamped, 0);
}

fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc  = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = ssao_params.inv_proj * ndc;
    return view.xyz / view.w;
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_ao);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let fdims = vec2<f32>(dims);
    let uv    = (vec2<f32>(gid.xy) + 0.5) / fdims;
    let depth = load_depth(vec2<i32>(gid.xy));
    if depth >= 0.9999 {
        textureStore(out_ao, vec2<i32>(gid.xy), vec4<f32>(1.0));
        return;
    }

    let pos = view_pos_from_depth(uv, depth);

    // View-space normal
    let world_n = textureSampleLevel(t_normal, s_nearest, uv, 0.0).xyz * 2.0 - 1.0;
    let normal  = normalize((ssao_params.view * vec4<f32>(world_n, 0.0)).xyz);

    // Random rotation — textureLoad on 4×4 tiled noise
    let noise_coord = vec2<i32>(gid.xy) % vec2<i32>(4);
    let rnd  = textureLoad(t_noise, noise_coord, 0).rgb;
    let rvec = normalize(rnd * 2.0 - 1.0);

    let tangent   = normalize(rvec - normal * dot(rvec, normal));
    let bitangent = cross(normal, tangent);
    let tbn       = mat3x3<f32>(tangent, bitangent, normal);

    var occlusion = 0.0;
    for (var i = 0u; i < ssao_params.sample_count; i++) {
        let s      = (tbn * kernel[i].xyz) * ssao_params.radius;
        let sample = pos + s;

        let offset = ssao_params.proj * vec4<f32>(sample, 1.0);
        let suv    = offset.xy / offset.w * 0.5 + 0.5;
        if any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0)) { continue; }

        let sd  = load_depth_uv(suv, fdims);
        let sp  = view_pos_from_depth(suv, sd);

        let range_check = smoothstep(0.0, 1.0, ssao_params.radius / abs(pos.z - sp.z));
        occlusion += select(0.0, 1.0, sp.z >= sample.z + ssao_params.bias) * range_check;
    }

    let ao = 1.0 - (occlusion / f32(ssao_params.sample_count)) * ssao_params.strength;
    textureStore(out_ao, vec2<i32>(gid.xy), vec4<f32>(clamp(ao, 0.0, 1.0)));
}

// ── Bilateral blur ─────────────────────────────────────────────────────────────
@group(0) @binding(0) var t_ao_in:    texture_2d<f32>;
@group(0) @binding(1) var t_dep_blur: texture_depth_2d;
@group(0) @binding(2) var s_blur:     sampler;
@group(0) @binding(3) var out_blur:   texture_storage_2d<rgba8unorm, write>;

struct BlurParams {
    texel_size: vec2<f32>,
    _pad:       vec2<f32>,
}
@group(0) @binding(4) var<uniform> blur_params: BlurParams;

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_blur);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let center_depth = textureLoad(t_dep_blur, vec2<i32>(gid.xy), 0);

    var result = 0.0; var weight = 0.0;
    let OFFSETS = array<vec2<i32>, 9>(
        vec2<i32>(-1,-1), vec2<i32>(0,-1), vec2<i32>(1,-1),
        vec2<i32>(-1, 0), vec2<i32>(0, 0), vec2<i32>(1, 0),
        vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1),
    );
    for (var i = 0; i < 9; i++) {
        let sc = clamp(vec2<i32>(gid.xy) + OFFSETS[i], vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        let d  = textureLoad(t_dep_blur, sc, 0);
        let w  = exp(-abs(d - center_depth) * 200.0);
        result += textureLoad(t_ao_in, sc, 0).r * w;
        weight += w;
    }
    textureStore(out_blur, vec2<i32>(gid.xy), vec4<f32>(result / max(weight, 0.0001)));
}
