// ══════════════════════════════════════════════════════════════════════════════
//  TAA — Temporal Anti-Aliasing
//  Pass 1: Reprojekcja + akumulacja historii
// ══════════════════════════════════════════════════════════════════════════════

struct TaaParams {
    curr_view_proj:  mat4x4<f32>,
    prev_view_proj:  mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    texel_size:      vec2<f32>,
    blend_factor:    f32,
    _pad:            f32,
}

@group(0) @binding(0) var<uniform> taa:       TaaParams;
@group(0) @binding(1) var t_current:           texture_2d<f32>;
@group(0) @binding(2) var t_history:           texture_2d<f32>;
@group(0) @binding(3) var t_depth:             texture_depth_2d;
@group(0) @binding(4) var s_linear:            sampler;
@group(0) @binding(5) var s_nearest:           sampler;
@group(0) @binding(6) var out_taa:             texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// YCoCg bez offsetu — poprawne dla float HDR
fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
         0.5  * c.r             - 0.5  * c.b,
    );
}
fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    let tmp = c.x - c.y;
    return vec3<f32>(
        tmp + c.z, // R
        c.x + c.y, // G
        tmp - c.z  // B
    );
}

fn clip_history(history: vec3<f32>, curr_uv: vec2<f32>) -> vec3<f32> {
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let off = vec2<f32>(f32(x), f32(y)) * taa.texel_size;
            let c   = to_ycocg(textureSampleLevel(t_current, s_nearest, curr_uv + off, 0.0).rgb);
            m1 += c;
            m2 += c * c;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    let sigma = sqrt(max(m2 - m1 * m1, vec3<f32>(0.0)));
    let gamma = 1.25;
    let mn = m1 - sigma * gamma;
    let mx = m1 + sigma * gamma;
    return from_ycocg(clamp(to_ycocg(history), mn, mx));
}

@compute @workgroup_size(8, 8, 1)
fn cs_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_taa);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let fdims = vec2<f32>(dims);
    let uv    = (vec2<f32>(gid.xy) + 0.5) / fdims;

    let curr = textureSampleLevel(t_current, s_nearest, uv, 0.0).rgb;

    let depth    = textureLoad(t_depth, vec2<i32>(gid.xy), 0);
    let ndc      = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let ndc_flip = vec4<f32>(ndc.x, -ndc.y, ndc.z, ndc.w);
    let world_h  = taa.inv_view_proj * ndc_flip;
    let world    = world_h.xyz / world_h.w;

    let prev_clip = taa.prev_view_proj * vec4<f32>(world, 1.0);
    let prev_ndc  = prev_clip.xyz / prev_clip.w;
    let prev_uv   = vec2<f32>(prev_ndc.x * 0.5 + 0.5, -prev_ndc.y * 0.5 + 0.5);

    let hist_raw = textureSampleLevel(t_history, s_linear, prev_uv, 0.0).rgb;
    let hist     = clip_history(hist_raw, uv);

    let in_screen = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));
    var result: vec3<f32>;
    if in_screen {
        let w_curr = 1.0 / (1.0 + luma(curr));
        let w_hist = 1.0 / (1.0 + luma(hist));
        let alpha  = taa.blend_factor * w_curr / (w_curr + w_hist);
        result = mix(hist, curr, alpha);
    } else {
        result = curr;
    }

    textureStore(out_taa, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}