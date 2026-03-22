// ══════════════════════════════════════════════════════════════════════════════
//  pbr.wgsl
//
//  Zmiany vs poprzednia wersja:
//  [1] Specular AA  — specular_aa_roughness() via dFdx/dFdy normalnych
//  [2] Shadow bias  — adaptacyjny per-cascade (shadow_params.w = mult)
//  [3] V_Smith fix  — Height-Correlated Smith GGX
//  [4] Dithering    — triangular PDF dither w post.wgsl
//  [PCSS] PCSS      — Percentage-Closer Soft Shadows (Fernando 2005)
//                     shadow_ext.z = 0 → PCF 3×3 (fallback)
//                     shadow_ext.z = 1 → PCSS 16 próbek
//                     shadow_ext.z = 2 → PCSS 32 próbek
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<uniform> camera: CameraUniform;
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
}

@group(1) @binding(0) var<uniform> object: ObjectUniform;
struct ObjectUniform {
    model:         mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct MaterialUniform {
    base_color:            vec4<f32>,
    metallic_roughness_ao: vec4<f32>,
    emissive:              vec4<f32>,
    clearcoat_params:      vec4<f32>,
    sheen_params:          vec4<f32>,
    advanced:              vec4<f32>,
    ext_ior_trans:         vec4<f32>,
    ext_specular_color:    vec4<f32>,
    ext_attenuation:       vec4<f32>,
    ext_iridescence:       vec4<f32>,
}
@group(2) @binding(0) var<uniform> material:         MaterialUniform;
@group(2) @binding(1) var t_base_color:               texture_2d<f32>;
@group(2) @binding(2) var s_main:                     sampler;
@group(2) @binding(3) var t_normal:                   texture_2d<f32>;
@group(2) @binding(4) var t_metallic_roughness:       texture_2d<f32>;
@group(2) @binding(5) var t_emissive:                 texture_2d<f32>;
@group(2) @binding(6) var t_occlusion:                texture_2d<f32>;
@group(2) @binding(7) var t_height:                   texture_2d<f32>;

struct GpuLight {
    position_or_dir: vec4<f32>,
    spot_dir:        vec4<f32>,
    color_intensity: vec4<f32>,
    params:          vec4<f32>,
}
struct LightArrayUniform {
    lights:        array<GpuLight, 16>,
    count_and_pad: vec4<u32>,
    ambient:       vec4<f32>,
}

struct CsmPbrUniform {
    light_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits:  vec4<f32>,
    // x=min_bias  y=slope_bias  z=texel_size  w=cascade_bias_multiplier (ZACHOWAĆ!)
    shadow_params:   vec4<f32>,
    // [PCSS] x=light_size  y=max_radius_texels  z=quality(0/1/2)  w=_pad
    shadow_ext:      vec4<f32>,
}

@group(3) @binding(0) var<uniform> lights:     LightArrayUniform;
@group(3) @binding(1) var<uniform> csm:        CsmPbrUniform;
@group(3) @binding(2) var t_shadow_map:         texture_depth_2d_array;
@group(3) @binding(3) var s_shadow:             sampler_comparison;
@group(3) @binding(4) var t_irradiance:         texture_cube<f32>;
@group(3) @binding(5) var t_prefilter:          texture_cube<f32>;
@group(3) @binding(6) var t_brdf_lut:           texture_2d<f32>;
@group(3) @binding(7) var s_ibl:                sampler;
@group(3) @binding(8) var s_lut:                sampler;

struct EnvUniform { rotation: mat4x4<f32>, }
@group(3) @binding(9) var<uniform> env: EnvUniform;

const PI:                f32 = 3.14159265358979;
const INV_PI:            f32 = 0.31830988618;
const EPSILON:           f32 = 0.00001;
const IBL_SCALE:         f32 = 1.0;
const MAX_PREFILTER_LOD: f32 = 4.0;

// ─────────────────────────────────────────────────────────────────────────────
// Vertex
// ─────────────────────────────────────────────────────────────────────────────
struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec2<f32>,
}
struct VertOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)        world_pos:    vec3<f32>,
    @location(1)        world_normal: vec3<f32>,
    @location(2)        world_tan:    vec3<f32>,
    @location(3)        world_bitan:  vec3<f32>,
    @location(4)        uv:           vec2<f32>,
    @location(5)        view_pos:     vec3<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    let w4        = object.model * vec4<f32>(in.position, 1.0);
    out.world_pos = w4.xyz;
    out.clip_pos  = camera.view_proj * w4;
    out.view_pos  = (camera.view * w4).xyz;
    let n = normalize((object.normal_matrix * vec4<f32>(in.normal,      0.0)).xyz);
    let t = normalize((object.normal_matrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_normal = n;
    out.world_tan    = t;
    out.world_bitan  = cross(n, t) * in.tangent.w;
    out.uv           = in.uv;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// [1] SPECULAR ALIASING — Toksvig-style roughness adjustment
// ─────────────────────────────────────────────────────────────────────────────
fn specular_aa_roughness(n: vec3<f32>, roughness: f32) -> f32 {
    let variance = length(dpdx(n)) + length(dpdy(n));
    let kappa    = 0.25;
    return clamp(sqrt(roughness * roughness + kappa * variance), roughness, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] HEIGHT-CORRELATED SMITH GGX
// ─────────────────────────────────────────────────────────────────────────────
fn V_SmithGGX_Correlated(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let a2       = roughness * roughness * roughness * roughness;
    let lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, EPSILON);
}

fn D_GGX(ndh: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness * roughness * roughness;
    let d  = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, EPSILON);
}

fn F_Schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn F_SchlickR(cos_t: f32, f0: vec3<f32>, r: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - r), f0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn Fd_Burley(ndv: f32, ndl: f32, ldh: f32, r: f32) -> f32 {
    let f90 = 0.5 + 2.0 * r * ldh * ldh;
    return (1.0 + (f90 - 1.0) * pow(1.0 - ndl, 5.0))
         * (1.0 + (f90 - 1.0) * pow(1.0 - ndv, 5.0))
         * INV_PI;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLEARCOAT & SHEEN BRDF
// ─────────────────────────────────────────────────────────────────────────────
fn F_Schlick_f0(cos_t: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn V_Kelemen(ldh: f32) -> f32 {
    return 0.25 / max(ldh * ldh, EPSILON);
}

fn D_Charlie(ndh: f32, roughness: f32) -> f32 {
    let inv_alpha = 1.0 / max(roughness, EPSILON);
    let sin2h     = max(1.0 - ndh * ndh, 0.0078125);
    return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) * 0.5 * INV_PI;
}

fn V_Ashikhmin(ndv: f32, ndl: f32) -> f32 {
    return 1.0 / max(4.0 * (ndl + ndv - ndl * ndv), EPSILON);
}

// ─────────────────────────────────────────────────────────────────────────────
// IBL
// ─────────────────────────────────────────────────────────────────────────────
fn specular_occlusion(ndv: f32, ao: f32) -> f32 {
    let a = clamp(ao, 0.0, 1.0);
    return clamp(pow(ndv + a, 5.0) - 1.0 + a, 0.0, 1.0);
}

fn f0_from_ior(ior: f32) -> f32 {
    let t = (ior - 1.0) / max(ior + 1.0, EPSILON);
    return t * t;
}

fn iridescence_tint(ndv: f32, strength: f32, film_ior: f32, d_nm: f32) -> vec3<f32> {
    if strength < 0.001 { return vec3<f32>(1.0); }
    let opl = ndv * d_nm * 0.014 * max(film_ior, 1.0);
    let ph = opl * 0.03141592653;
    let r = sin(ph) * 0.5 + 0.5;
    let g = sin(ph + 2.094395) * 0.5 + 0.5;
    let b = sin(ph + 4.18879) * 0.5 + 0.5;
    return mix(vec3<f32>(1.0), vec3<f32>(r, g, b), strength);
}

fn sample_transmission(n: vec3<f32>, v: vec3<f32>, ior: f32, rough: f32) -> vec3<f32> {
    let eta = 1.0 / max(ior, 1.001);
    let cos_i = dot(n, v);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    var dir = reflect(-v, n);
    if k > 0.0 {
        dir = normalize(eta * (-v) - (eta * cos_i + sqrt(k)) * n);
    }
    let d_rot = (env.rotation * vec4<f32>(dir, 0.0)).xyz;
    let lod = min(rough * MAX_PREFILTER_LOD + 2.8, MAX_PREFILTER_LOD);
    return textureSampleLevel(t_prefilter, s_ibl, d_rot, lod).rgb * IBL_SCALE;
}

fn ibl(n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>, albedo: vec3<f32>,
       metallic: f32, perc_rough: f32, ao: f32, trans_m: f32, irid: vec3<f32>) -> vec3<f32> {
    let ndv   = max(dot(n, v), EPSILON);
    let r     = reflect(-v, n);
    let n_rot = (env.rotation * vec4<f32>(n, 0.0)).xyz;
    let r_rot = (env.rotation * vec4<f32>(r, 0.0)).xyz;

    let irrad    = textureSample(t_irradiance, s_ibl, n_rot).rgb * IBL_SCALE;
    let F        = F_SchlickR(ndv, f0, perc_rough);
    let kd       = (1.0 - F) * (1.0 - metallic);
    let en_trans = max(1.0 - trans_m, 0.0);
    let diffuse  = kd * albedo * irrad * en_trans;

    let lod       = perc_rough * MAX_PREFILTER_LOD;
    let env_color = textureSampleLevel(t_prefilter, s_ibl, r_rot, lod).rgb * IBL_SCALE;
    let brdf      = textureSample(t_brdf_lut, s_lut, vec2<f32>(ndv, perc_rough)).rg;
    let spec_occ  = specular_occlusion(ndv, ao);
    let specular  = env_color * (f0 * irid * brdf.x + brdf.y) * spec_occ;

    return (diffuse + specular) * ao;
}

// ─────────────────────────────────────────────────────────────────────────────
// Anizotropowy NDF
// ─────────────────────────────────────────────────────────────────────────────
fn D_GGX_aniso(ndh: f32, toh: f32, boh: f32, at: f32, ab: f32) -> f32 {
    let a2 = at * ab;
    let x  = (toh * toh) / max(at * at, EPSILON);
    let y  = (boh * boh) / max(ab * ab, EPSILON);
    let z  = ndh * ndh;
    let v  = x + y + z;
    return a2 / max(PI * v * v, EPSILON);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallax mapping
// ─────────────────────────────────────────────────────────────────────────────
fn parallax_uv(uv: vec2<f32>, V: vec3<f32>, T: vec3<f32>, B: vec3<f32>, N: vec3<f32>, scale: f32) -> vec2<f32> {
    if scale <= 0.00001 { return uv; }
    let v_ts = vec3<f32>(dot(V, T), dot(V, B), dot(V, N));
    let vz   = max(v_ts.z, 0.08);
    let num  = 16.0;
    let layer = 1.0 / num;
    let delta_uv = vec2<f32>(v_ts.x, v_ts.y) / vz * scale * 0.12 / num;
    var cur_uv = uv;
    var cur_d  = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let h = textureSample(t_height, s_main, cur_uv).r;
        if cur_d >= h { break; }
        cur_d += layer;
        cur_uv -= delta_uv;
    }
    return cur_uv;
}

// ═════════════════════════════════════════════════════════════════════════════
// [PCSS] PERCENTAGE-CLOSER SOFT SHADOWS
//
// Algorytm (Fernando 2005, GPU Gems 1 Ch.11):
//   Krok 1 — Blocker search:
//     Próbkuj shadow mapę Poissonem w promieniu searchRadius.
//     Oblicz avg_blocker_depth ze wszystkich próbek BLIŻEJ niż receiver.
//
//   Krok 2 — Penumbra estimation:
//     penumbra = (d_receiver - d_blocker) * light_size / d_blocker
//     pcf_radius = clamp(penumbra * texels_per_unit, 0.5, max_radius)
//
//   Krok 3 — Variable-radius PCF:
//     Próbkuj shadow mapę Poissonem w promieniu pcf_radius.
//     Zwróć średnią widoczność.
//
// Rotacja dysku Poissona per-piksel via IGN — eliminuje wzory banding.
// IGN (Roberts 2016): f(x,y) = fract(52.9829189 * fract(0.06711056*x + 0.00583715*y))
//
// Koszt:
//   quality=0 (PCF 3×3):   9 tapów — najszybszy, twarde cienie
//   quality=1 (PCSS 16):   16+16=32 tapy — dobry balans real-time
//   quality=2 (PCSS 32):   16+32=48 tapów — screenshot quality
// ═════════════════════════════════════════════════════════════════════════════

// ── Poisson disk 16 próbek (van der Corput) ───────────────────────────────
const POISSON_16 = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
);

// ── Poisson disk 32 próbek (blue-noise distributed) ───────────────────────
const POISSON_32 = array<vec2<f32>, 32>(
    vec2<f32>(-0.61339800,  0.52506000),
    vec2<f32>( 0.25734900, -0.75584000),
    vec2<f32>(-0.97510000,  0.05667000),
    vec2<f32>( 0.71023000,  0.31830000),
    vec2<f32>(-0.14040000,  0.93820000),
    vec2<f32>( 0.56140000, -0.36950000),
    vec2<f32>(-0.38360000, -0.80970000),
    vec2<f32>( 0.98220000, -0.13640000),
    vec2<f32>(-0.74060000,  0.28960000),
    vec2<f32>( 0.07220000,  0.48470000),
    vec2<f32>(-0.53440000, -0.23450000),
    vec2<f32>( 0.46280000,  0.81840000),
    vec2<f32>(-0.28420000,  0.18530000),
    vec2<f32>( 0.84490000, -0.52760000),
    vec2<f32>(-0.66780000, -0.53900000),
    vec2<f32>( 0.17290000, -0.16980000),
    vec2<f32>(-0.08510000, -0.61230000),
    vec2<f32>( 0.63620000,  0.06620000),
    vec2<f32>(-0.91370000,  0.40600000),
    vec2<f32>( 0.38940000, -0.91530000),
    vec2<f32>(-0.49280000,  0.74940000),
    vec2<f32>( 0.29560000,  0.57680000),
    vec2<f32>(-0.15300000, -0.96540000),
    vec2<f32>( 0.76510000,  0.61730000),
    vec2<f32>(-0.88540000, -0.09260000),
    vec2<f32>( 0.05060000,  0.99720000),
    vec2<f32>(-0.69810000,  0.71560000),
    vec2<f32>( 0.92070000,  0.38210000),
    vec2<f32>(-0.32470000,  0.51030000),
    vec2<f32>( 0.48150000, -0.62080000),
    vec2<f32>(-0.83060000, -0.32870000),
    vec2<f32>( 0.11590000,  0.21630000),
);

// ── IGN — Interleaved Gradient Noise (Roberts 2016) ───────────────────────
// Dobry szum screenspace — brak powtarzającego się wzoru, tani.
fn ign(px: f32, py: f32) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * px + 0.00583715 * py));
}

// ── Obrót 2D wektora o kąt angle (radiany) ────────────────────────────────
// WGSL mat2x2 jest column-major: mat2x2(col0, col1).
// col0 = (cos, sin), col1 = (-sin, cos) — obrót CCW.
fn rotate2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(c, s, -s, c) * v;
}

// ── Wspólna logika wyboru kaskady i obliczania biasu ──────────────────────
struct CascadeInfo {
    cascade:    u32,
    uv:         vec2<f32>,
    depth:      f32,    // p.z (NDC depth w shadow space)
    bias:       f32,
    valid:      bool,   // false = poza shadow mapą → fully lit
}

fn cascade_info(world_pos: vec3<f32>, view_z: f32, ndl: f32) -> CascadeInfo {
    var ci: CascadeInfo;
    ci.valid   = true;
    ci.cascade = 3u;

    let near = 0.05;
    let d    = -view_z - near;

    if      d < csm.cascade_splits.x { ci.cascade = 0u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y { ci.cascade = 1u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z { ci.cascade = 2u; }

    let lp    = csm.light_view_proj[ci.cascade] * vec4<f32>(world_pos, 1.0);
    let p     = lp.xyz / lp.w;
    ci.uv     = vec2<f32>(p.x * 0.5 + 0.5, p.y * -0.5 + 0.5);
    ci.depth  = p.z;

    if any(ci.uv < vec2<f32>(0.005)) || any(ci.uv > vec2<f32>(0.995)) {
        ci.valid = false;
        return ci;
    }

    // Adaptacyjny bias: rośnie wykładniczo z indeksem kaskady
    let bias_mult  = max(csm.shadow_params.w, 1.0);
    let bias_scale = pow(bias_mult, f32(ci.cascade));
    ci.bias = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x) * bias_scale;

    return ci;
}

// ─────────────────────────────────────────────────────────────────────────────
// [2] CSM Shadow — PCF 3×3 (fallback / quality=0)
//
// Zachowany bez zmian — używany gdy shadow_ext.z < 0.5.
// ─────────────────────────────────────────────────────────────────────────────
fn csm_shadow(world_pos: vec3<f32>, view_z: f32, ndl: f32) -> f32 {
    var cascade = 3u;
    let near    = 0.05;
    let d       = -view_z - near;

    if      d < csm.cascade_splits.x { cascade = 0u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y { cascade = 1u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z { cascade = 2u; }

    let lp  = csm.light_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let p   = lp.xyz / lp.w;
    let uv  = vec2<f32>(p.x * 0.5 + 0.5, p.y * -0.5 + 0.5);

    if any(uv < vec2<f32>(0.005)) || any(uv > vec2<f32>(0.995)) { return 1.0; }

    let bias_mult  = max(csm.shadow_params.w, 1.0);
    let bias_scale = pow(bias_mult, f32(cascade));
    let bias       = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x) * bias_scale;
    let texel      = csm.shadow_params.z;

    var pcf = 0.0;
    for (var sx = -1; sx <= 1; sx++) {
        for (var sy = -1; sy <= 1; sy++) {
            let off = vec2<f32>(f32(sx), f32(sy)) * texel;
            pcf += textureSampleCompareLevel(t_shadow_map, s_shadow,
                                             uv + off, i32(cascade), p.z - bias);
        }
    }

    let raw = pcf / 9.0;
    if cascade < 3u {
        let split = select(
            csm.cascade_splits.x,
            select(csm.cascade_splits.x + csm.cascade_splits.y,
                   csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z,
                   cascade == 2u),
            cascade == 1u,
        );
        let blend_t = clamp((d - (split - split * 0.1)) / (split * 0.1), 0.0, 1.0);
        if blend_t > 0.0 {
            let next       = cascade + 1u;
            let next_scale = pow(bias_mult, f32(next));
            let next_bias  = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x) * next_scale;
            let lp2  = csm.light_view_proj[next] * vec4<f32>(world_pos, 1.0);
            let p2   = lp2.xyz / lp2.w;
            let uv2  = vec2<f32>(p2.x * 0.5 + 0.5, p2.y * -0.5 + 0.5);
            var pcf2 = 0.0;
            if all(uv2 >= vec2<f32>(0.005)) && all(uv2 <= vec2<f32>(0.995)) {
                for (var sx = -1; sx <= 1; sx++) {
                    for (var sy = -1; sy <= 1; sy++) {
                        let off = vec2<f32>(f32(sx), f32(sy)) * texel;
                        pcf2 += textureSampleCompareLevel(t_shadow_map, s_shadow,
                                                          uv2 + off, i32(next), p2.z - next_bias);
                    }
                }
            } else { pcf2 = 9.0; }
            return mix(raw, pcf2 / 9.0, blend_t);
        }
    }
    return raw;
}

// ─────────────────────────────────────────────────────────────────────────────
// [PCSS] Krok 1 — Blocker search
//
// Szuka fragmentów shadow mapy bliższych niż receiver_depth w promieniu
// search_r (w texelach). Zwraca średnią głębię blokujących fragmentów
// lub -1.0 gdy żaden bloker nie znaleziony (= fully lit).
// Zawsze używa POISSON_16 niezależnie od quality (16 tapów = wystarczy).
// ─────────────────────────────────────────────────────────────────────────────
fn blocker_search(
    uv:             vec2<f32>,
    cascade:        u32,
    receiver_depth: f32,
    bias:           f32,
    search_r:       f32,    // promień w texelach
    rotation_angle: f32,
) -> f32 {
    let texel = csm.shadow_params.z;    // 1 / SHADOW_MAP_SIZE
    var blocker_sum   = 0.0;
    var blocker_count = 0;

    for (var i = 0; i < 16; i++) {
        let s   = rotate2d(POISSON_16[i], rotation_angle);
        let off = s * search_r * texel;
        // textureSampleLevel (nie comparison) — pobieramy surową głębię blokera.
        // Używamy s_ibl (zwykły sampler linear) bo s_shadow jest comparison-only.
        // t_shadow_map jest texture_depth_2d_array — próbkowanie przez textureLoad.
        let sample_uv  = uv + off;
        let su_clamped = clamp(sample_uv, vec2<f32>(0.001), vec2<f32>(0.999));
        let dims       = vec2<f32>(textureDimensions(t_shadow_map));
        let coord      = vec2<i32>(su_clamped * dims);
        let depth_s    = textureLoad(t_shadow_map, coord, i32(cascade), 0);

        // depth_s < receiver_depth - bias → ten fragment jest blokerem (cień pada na receiver)
        if depth_s < (receiver_depth - bias) {
            blocker_sum   += depth_s;
            blocker_count += 1;
        }
    }

    if blocker_count == 0  { return -1.0; }   // brak blokerów → fully lit
    if blocker_count == 16 { return -2.0; }   // wszystko zablokowane → fully shadowed
    return blocker_sum / f32(blocker_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// [PCSS] Krok 3 — Variable-radius PCF
//
// Krok z textureSampleCompareLevel (hardware PCF) w promieniu pcf_r texeli.
// use_32 = true → 32 próbki (quality=2), false → 16 próbek (quality=1).
// ─────────────────────────────────────────────────────────────────────────────
fn variable_pcf(
    uv:             vec2<f32>,
    cascade:        u32,
    receiver_depth: f32,
    bias:           f32,
    pcf_r:          f32,    // promień w texelach
    rotation_angle: f32,
    use_32:         bool,
) -> f32 {
    let texel = csm.shadow_params.z;
    var sum   = 0.0;

    if use_32 {
        for (var i = 0; i < 32; i++) {
            let s   = rotate2d(POISSON_32[i], rotation_angle);
            let off = s * pcf_r * texel;
            sum += textureSampleCompareLevel(t_shadow_map, s_shadow,
                                             uv + off, i32(cascade),
                                             receiver_depth - bias);
        }
        return sum / 32.0;
    } else {
        for (var i = 0; i < 16; i++) {
            let s   = rotate2d(POISSON_16[i], rotation_angle);
            let off = s * pcf_r * texel;
            sum += textureSampleCompareLevel(t_shadow_map, s_shadow,
                                             uv + off, i32(cascade),
                                             receiver_depth - bias);
        }
        return sum / 16.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// [PCSS] Główna funkcja — dispatch + cascade blend
//
// shadow_ext: x=light_size  y=max_radius_texels  z=quality  w=_pad
//
// Gdy quality < 0.5 → deleguje do csm_shadow() (PCF 3×3).
// Gdy quality >= 0.5 → PCSS z 16 lub 32 próbkami.
//
// Cascade blend działa tak samo jak w csm_shadow() — płynne przejście
// między kaskadami na ostatnich 10% zakresu każdej kaskady.
// ─────────────────────────────────────────────────────────────────────────────
fn pcss_shadow(
    world_pos:  vec3<f32>,
    view_z:     f32,
    ndl:        f32,
    pixel_xy:   vec2<f32>,  // @builtin(position).xy — do IGN
) -> f32 {
    let quality    = csm.shadow_ext.z;
    let light_size = csm.shadow_ext.x;
    let max_r      = csm.shadow_ext.y;

    // Fallback do oryginalnego PCF 3×3
    if quality < 0.5 {
        return csm_shadow(world_pos, view_z, ndl);
    }

    let use_32 = quality > 1.5;

    // Per-piksel rotacja dysku Poissona — eliminuje wzory banding
    let rot_angle = ign(pixel_xy.x, pixel_xy.y) * (2.0 * PI);

    // Wybierz kaskadę
    var cascade = 3u;
    let near    = 0.05;
    let d       = -view_z - near;
    if      d < csm.cascade_splits.x { cascade = 0u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y { cascade = 1u; }
    else if d < csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z { cascade = 2u; }

    // Oblicz shadow-space pozycję
    let lp  = csm.light_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let p   = lp.xyz / lp.w;
    let uv  = vec2<f32>(p.x * 0.5 + 0.5, p.y * -0.5 + 0.5);

    if any(uv < vec2<f32>(0.005)) || any(uv > vec2<f32>(0.995)) { return 1.0; }

    // Bias (identyczny z csm_shadow)
    let bias_mult  = max(csm.shadow_params.w, 1.0);
    let bias_scale = pow(bias_mult, f32(cascade));
    let bias       = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x) * bias_scale;

    let vis = pcss_single_cascade(uv, cascade, p.z, bias, light_size, max_r,
                                   rot_angle, use_32);

    // Cascade blend (ostatnie 10% zakresu kaskady)
    if cascade < 3u {
        let split = select(
            csm.cascade_splits.x,
            select(csm.cascade_splits.x + csm.cascade_splits.y,
                   csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z,
                   cascade == 2u),
            cascade == 1u,
        );
        let blend_t = clamp((d - (split - split * 0.1)) / (split * 0.1), 0.0, 1.0);
        if blend_t > 0.0 {
            let next       = cascade + 1u;
            let next_scale = pow(bias_mult, f32(next));
            let next_bias  = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x) * next_scale;
            let lp2  = csm.light_view_proj[next] * vec4<f32>(world_pos, 1.0);
            let p2   = lp2.xyz / lp2.w;
            let uv2  = vec2<f32>(p2.x * 0.5 + 0.5, p2.y * -0.5 + 0.5);
            if all(uv2 >= vec2<f32>(0.005)) && all(uv2 <= vec2<f32>(0.995)) {
                let vis2 = pcss_single_cascade(uv2, next, p2.z, next_bias,
                                               light_size, max_r, rot_angle, use_32);
                return mix(vis, vis2, blend_t);
            }
        }
    }

    return vis;
}

// ─────────────────────────────────────────────────────────────────────────────
// [PCSS] Jeden przebieg PCSS dla jednej kaskady
// ─────────────────────────────────────────────────────────────────────────────
fn pcss_single_cascade(
    uv:         vec2<f32>,
    cascade:    u32,
    recv_depth: f32,
    bias:       f32,
    light_size: f32,
    max_r:      f32,
    rot_angle:  f32,
    use_32:     bool,
) -> f32 {
    // Krok 1: Blocker search
    // search_radius w texelach: większy light_size → szersze przeszukiwanie
    let search_r = clamp(light_size * 4.0, 2.0, max_r * 2.0);
    let avg_blocker = blocker_search(uv, cascade, recv_depth, bias, search_r, rot_angle);

    if avg_blocker < -1.5 { return 0.0; }   // fully shadowed
    if avg_blocker < 0.0  { return 1.0; }   // fully lit (brak blokerów)

    // Krok 2: Penumbra — szerokość penumbry w texelach
    // penumbra = (d_receiver - d_blocker) * light_size / d_blocker
    // * (1/texel_size) konwertuje do texeli (texel_size = 1/2048)
    let texel     = csm.shadow_params.z;
    let penumbra  = (recv_depth - avg_blocker) * light_size / max(avg_blocker, 0.001);
    let pcf_r     = clamp(penumbra / texel, 0.5, max_r);

    // Krok 3: Variable-radius PCF
    return variable_pcf(uv, cascade, recv_depth, bias, pcf_r, rot_angle, use_32);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragment
// ─────────────────────────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let N_geom = normalize(in.world_normal);
    let T0     = normalize(in.world_tan - N_geom * dot(N_geom, in.world_tan));
    let B0     = cross(N_geom, T0) * sign(dot(in.world_bitan, cross(N_geom, normalize(in.world_tan))));

    let v  = normalize(camera.camera_pos.xyz - in.world_pos);
    let uv = parallax_uv(in.uv, v, T0, B0, N_geom, material.advanced.w);

    let albedo_s = textureSample(t_base_color,         s_main, uv);
    let mr_s     = textureSample(t_metallic_roughness, s_main, uv);
    let norm_s   = textureSample(t_normal,             s_main, uv);
    let occ_s    = textureSample(t_occlusion,          s_main, uv);

    let albedo   = material.base_color.rgb * albedo_s.rgb;
    let alpha    = material.base_color.a   * albedo_s.a;
    let metallic = clamp(material.metallic_roughness_ao.x * mr_s.b, 0.0, 1.0);
    let ao_mix   = material.metallic_roughness_ao.z;
    let ao_comb  = clamp(mr_s.r * occ_s.r, 0.0, 1.0);
    let ao       = mix(1.0, ao_comb, ao_mix);

    let raw_roughness = clamp(material.metallic_roughness_ao.y * mr_s.g, 0.045, 1.0);

    let n_ts = norm_s.rgb * 2.0 - 1.0;
    let n    = normalize(
        T0 * n_ts.x * material.metallic_roughness_ao.w +
        B0 * n_ts.y * material.metallic_roughness_ao.w +
        N_geom * n_ts.z
    );

    let perc_rough = specular_aa_roughness(n, raw_roughness);

    let ndv = max(dot(n, v), EPSILON);
    let ior = max(material.ext_ior_trans.x, 1.001);
    let f0d = vec3<f32>(f0_from_ior(ior)) * material.ext_specular_color.xyz * material.ext_ior_trans.w;
    let f0  = mix(f0d, albedo, metallic);
    let dc  = albedo * (1.0 - metallic);
    let trans_m = clamp(material.ext_ior_trans.y * (1.0 - metallic), 0.0, 1.0);
    let diff_en = max(1.0 - trans_m, 0.0);
    let irid = iridescence_tint(ndv, material.ext_iridescence.x,
                                material.ext_iridescence.y, material.ext_iridescence.z);

    let ani = clamp(material.advanced.x, -1.0, 1.0);
    let rot = material.advanced.y;
    let cr  = cos(rot);
    let sr  = sin(rot);
    var T_a = normalize(T0 * cr + B0 * sr);
    T_a = normalize(T_a - n * dot(T_a, n));
    let B_a = normalize(cross(n, T_a));
    let at  = max(perc_rough * (1.0 + ani), 0.045);
    let ab  = max(perc_rough * (1.0 - ani), 0.045);
    let r_vis = sqrt(at * ab);

    let clearcoat           = material.clearcoat_params.x;
    let clearcoat_roughness = clamp(material.clearcoat_params.y, 0.045, 1.0);
    let sheen_color         = material.sheen_params.xyz;
    let sheen_roughness     = material.sheen_params.w;
    let cc_perc_rough       = specular_aa_roughness(N_geom, clearcoat_roughness);
    let subs                = clamp(material.advanced.z, 0.0, 1.0);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < lights.count_and_pad.x; i++) {
        let l_data = lights.lights[i];
        var l_dir:    vec3<f32>;
        var radiance: vec3<f32>;
        var is_sun   = false;

        if l_data.params.x < 0.5 {
            l_dir    = normalize(-l_data.position_or_dir.xyz);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a;
            is_sun   = true;
        } else {
            let Lp   = l_data.position_or_dir.xyz;
            let to_l = Lp - in.world_pos;
            let dist = length(to_l);
            l_dir    = to_l / max(dist, EPSILON);
            var base = l_data.color_intensity.rgb * l_data.color_intensity.a
                     / max(dist * dist, 0.01);
            let rng = l_data.params.y;
            if rng < 900.0 && rng > 0.05 {
                base *= pow(saturate(1.0 - dist / rng), 2.0);
            }
            if l_data.params.x < 1.5 {
                radiance = base;
            } else {
                let spot_axis = normalize(l_data.spot_dir.xyz);
                let cos_theta = dot(l_dir, spot_axis);
                let cos_in    = cos(l_data.params.z);
                let cos_out   = cos(l_data.params.w);
                let spot_f    = smoothstep(cos_out, cos_in, cos_theta);
                radiance      = base * spot_f;
            }
        }

        let ndl_raw = dot(n, l_dir);
        let ndl     = max(ndl_raw, 0.0);

        if subs > 0.001 {
            let back = max(dot(n, -l_dir), 0.0);
            lo += subs * (1.0 - metallic) * dc * radiance * back * 0.22 * diff_en;
        }

        if ndl <= 0.0 { continue; }

        let h   = normalize(v + l_dir);
        let ndh = max(dot(n, h), EPSILON);
        let ldh = max(dot(l_dir, h), EPSILON);

        var shd = 1.0;
        if is_sun {
            // [PCSS] Użyj pcss_shadow zamiast csm_shadow.
            // Dispatch do PCF 3×3 lub PCSS w zależności od shadow_ext.z.
            shd = pcss_shadow(in.world_pos, in.view_pos.z, ndl, in.clip_pos.xy);
        }

        let wrap_amt = subs * 0.45;
        let diff_wrap = mix(1.0,
            saturate((ndl + wrap_amt) / (1.0 + wrap_amt)) / max(ndl, EPSILON),
            subs);
        let diff_wrap_clamped = min(diff_wrap, 2.0);

        var D = 0.0;
        if abs(ani) < 0.002 {
            D = D_GGX(ndh, perc_rough);
        } else {
            let toh = dot(T_a, h);
            let boh = dot(B_a, h);
            D = D_GGX_aniso(ndh, toh, boh, at, ab);
        }

        let Vis = V_SmithGGX_Correlated(ndv, ndl, r_vis);
        let F   = F_Schlick(ldh, f0) * irid;
        let kd  = (1.0 - F) * (1.0 - metallic);

        var cc_contrib = 0.0;
        var Fcc        = 0.0;
        if clearcoat > 0.001 {
            let D_cc = D_GGX(ndh, cc_perc_rough);
            let V_cc = V_Kelemen(ldh);
            Fcc      = F_Schlick_f0(ldh, 0.04) * clearcoat;
            cc_contrib = D_cc * V_cc * Fcc;
        }

        var sheen_contrib = vec3<f32>(0.0);
        if sheen_roughness > 0.001 {
            sheen_contrib = D_Charlie(ndh, max(sheen_roughness, 0.045))
                          * V_Ashikhmin(ndv, ndl)
                          * sheen_color;
        }

        let base_pbr = kd * dc * Fd_Burley(ndv, ndl, ldh, perc_rough) * diff_wrap_clamped * diff_en
                     + D * Vis * F;
        lo += (base_pbr * (1.0 - Fcc) + vec3<f32>(cc_contrib) + sheen_contrib) * radiance * ndl * shd;
    }

    lo += lights.ambient.rgb * dc * (1.0 - metallic) * ao * INV_PI * diff_en;

    var ambient = ibl(n, v, f0, dc, metallic, perc_rough, ao, trans_m, irid);

    if clearcoat > 0.001 {
        let cc_ndv   = max(dot(N_geom, v), EPSILON);
        let Fc_ibl   = F_Schlick_f0(cc_ndv, 0.04) * clearcoat;
        ambient      = ambient * (1.0 - Fc_ibl);
        let r_cc_rot = (env.rotation * vec4<f32>(reflect(-v, N_geom), 0.0)).xyz;
        let lod_cc   = cc_perc_rough * MAX_PREFILTER_LOD;
        let env_cc   = textureSampleLevel(t_prefilter, s_ibl, r_cc_rot, lod_cc).rgb * IBL_SCALE;
        let brdf_cc  = textureSample(t_brdf_lut, s_lut, vec2<f32>(cc_ndv, cc_perc_rough)).rg;
        let s_occ_cc = specular_occlusion(cc_ndv, ao);
        ambient     += env_cc * (vec3<f32>(0.04) * brdf_cc.x + brdf_cc.y) * clearcoat * s_occ_cc * irid;
    }

    let em_tex = textureSample(t_emissive, s_main, uv).rgb;
    let em_hdr = em_tex * material.emissive.rgb * material.emissive.w;

    var out_rgb = lo + ambient + em_hdr;
    if trans_m > 0.02 {
        var tcol = sample_transmission(n, v, ior, perc_rough);
        let att  = material.ext_attenuation.xyz * material.ext_attenuation.w
                 * max(material.ext_ior_trans.z, 0.0001);
        tcol *= vec3<f32>(exp(-att.x), exp(-att.y), exp(-att.z));
        out_rgb = mix(out_rgb, tcol, trans_m);
    }

    return vec4<f32>(out_rgb, alpha);
}
