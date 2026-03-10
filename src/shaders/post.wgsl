// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Post-Processing Shader                                                     ║
// ║  Pass 1 (bloom_threshold): extract bright pixels                           ║
// ║  Pass 2 (bloom_blur):      gaussian blur                                   ║
// ║  Pass 3 (composite):       combine HDR + bloom, tonemap, CA, vignette      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Shared fullscreen quad ─────────────────────────────────────────────────────

struct PostVOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> PostVOut {
    let uvs = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let p  = uvs[vid];
    var out: PostVOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv  = p * vec2<f32>(0.5, -0.5) + 0.5;
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pass 1 – Bloom threshold / bright extraction
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var t_hdr:     texture_2d<f32>;
@group(0) @binding(1) var s_nearest: sampler;

struct BloomParams {
    threshold:    f32,
    knee:         f32,
    scatter:      f32,
    _pad:         f32,
}
@group(0) @binding(2) var<uniform> bloom: BloomParams;

// Quadratic threshold curve (Jimenez 2014)
fn quadratic_threshold(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let luma   = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let rq     = clamp(luma - threshold + knee, 0.0, 2.0 * knee);
    let weight = (rq * rq) / (4.0 * knee + 0.00001);
    return color * max(weight, luma - threshold) / max(luma, 0.00001);
}

@fragment
fn fs_bloom_threshold(in: PostVOut) -> @location(0) vec4<f32> {
    let color = textureSample(t_hdr, s_nearest, in.uv).rgb;
    let bright = quadratic_threshold(color, bloom.threshold, bloom.knee);
    return vec4<f32>(bright, 1.0);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pass 2 – Bloom blur (separable 13-tap Gaussian, run twice: H + V)
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var t_bloom_src: texture_2d<f32>;
@group(0) @binding(1) var s_linear:    sampler;

struct BlurParams {
    texel_size: vec2<f32>,
    horizontal: u32,
    _pad:       u32,
}
@group(0) @binding(2) var<uniform> blur: BlurParams;

// Kawase dual-filter weights (approximate Gaussian)
const WEIGHTS: array<f32, 7> = array<f32, 7>(
    0.0625, 0.125, 0.25, 0.375, 0.25, 0.125, 0.0625
);

@fragment
fn fs_bloom_blur(in: PostVOut) -> @location(0) vec4<f32> {
    var result = vec3<f32>(0.0);
    let dir = select(vec2<f32>(0.0, blur.texel_size.y),
                     vec2<f32>(blur.texel_size.x, 0.0),
                     blur.horizontal == 1u);

    for (var i = 0; i < 7; i++) {
        let offset = dir * f32(i - 3);
        result += textureSample(t_bloom_src, s_linear, in.uv + offset).rgb * WEIGHTS[i];
    }
    return vec4<f32>(result, 1.0);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pass 3 – Composite: HDR + bloom + tone map + CA + vignette + grain
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var t_hdr_in:   texture_2d<f32>;
@group(0) @binding(1) var t_bloom_in: texture_2d<f32>;
@group(0) @binding(2) var s_comp:     sampler;

struct PostParams {
    bloom_strength:       f32,
    exposure:             f32,   // EV stops
    ca_strength:          f32,   // chromatic aberration magnitude
    vignette_strength:    f32,
    vignette_radius:      f32,
    vignette_smoothness:  f32,
    grain_strength:       f32,
    time:                 f32,   // for animated grain
}
@group(0) @binding(3) var<uniform> post: PostParams;

// ── ACES fitted (Hill/Narkowicz) ──────────────────────────────────────────────
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Chromatic aberration (radial barrel) ──────────────────────────────────────
fn chromatic_aberration(uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let center  = uv - 0.5;
    let dist    = length(center);
    let dir     = normalize(center + vec2<f32>(0.0001));
    let r = textureSample(t_hdr_in, s_comp, uv + dir * strength * dist * 1.0).r;
    let g = textureSample(t_hdr_in, s_comp, uv                              ).g;
    let b = textureSample(t_hdr_in, s_comp, uv - dir * strength * dist * 0.7).b;
    return vec3<f32>(r, g, b);
}

// ── Vignette ──────────────────────────────────────────────────────────────────
fn vignette(uv: vec2<f32>, radius: f32, smoothness: f32) -> f32 {
    let d = length(uv - 0.5) * 2.0;
    return 1.0 - smoothstep(radius - smoothness, radius + smoothness, d);
}

// ── Interleaved gradient noise (Jimenez 2014) – fast film grain ───────────────
fn igr_noise(pixel: vec2<f32>, t: f32) -> f32 {
    let p = pixel + t * 5.588238;
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_composite(in: PostVOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_hdr_in));
    let pixel    = in.uv * tex_size;

    // HDR color with chromatic aberration
    var hdr = chromatic_aberration(in.uv, post.ca_strength * 0.005);

    // Add bloom
    let bloom_color = textureSample(t_bloom_in, s_comp, in.uv).rgb;
    hdr += bloom_color * post.bloom_strength;

    // Exposure
    hdr *= pow(2.0, post.exposure);

    // ACES tone mapping
    var ldr = aces_filmic(hdr);

    // Vignette
    let vig  = vignette(in.uv, post.vignette_radius, post.vignette_smoothness);
    ldr     *= mix(1.0 - post.vignette_strength, 1.0, vig);

    // Film grain
    let noise = igr_noise(pixel, post.time);
    ldr      += (noise - 0.5) * post.grain_strength * 0.04;

    // Gamma correction
    ldr = pow(max(ldr, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(ldr, 1.0);
}
