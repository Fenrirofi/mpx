// ══════════════════════════════════════════════════════════════════════════════
//  pbr.wgsl — POPRAWIONA WERSJA
//
//  Zmiany:
//  [1] Specular AA  — specular_aa_roughness() via dFdx/dFdy normalnych
//  [2] Shadow bias  — adaptacyjny per-cascade (shadow_params.w = mult)
//  [3] V_Smith fix  — Height-Correlated Smith GGX (ndl/ndv odwrócone poprawnie)
//  [4] Dithering    — triangular PDF dither w post.wgsl (patrz tamten plik)
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
@group(2) @binding(7) var t_height:                  texture_2d<f32>;

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
    // x=min_bias  y=slope_bias  z=texel_size  w=cascade_bias_multiplier
    shadow_params:   vec4<f32>,
}

@group(3) @binding(0) var<uniform> lights:  LightArrayUniform;
@group(3) @binding(1) var<uniform> csm:     CsmPbrUniform;
@group(3) @binding(2) var t_shadow_map:      texture_depth_2d_array;
@group(3) @binding(3) var s_shadow:          sampler_comparison;
@group(3) @binding(4) var t_irradiance:      texture_cube<f32>;
@group(3) @binding(5) var t_prefilter:       texture_cube<f32>;
@group(3) @binding(6) var t_brdf_lut:        texture_2d<f32>;
@group(3) @binding(7) var s_ibl:             sampler;
@group(3) @binding(8) var s_lut:             sampler;

struct EnvUniform { rotation: mat4x4<f32>, }
@group(3) @binding(9) var<uniform> env: EnvUniform;

const PI:                f32 = 3.14159265358979;
const INV_PI:            f32 = 0.31830988618;
const EPSILON:           f32 = 0.00001;
// Pełniejsze odbicia otoczenia (viewport SP / glTF viewer ~ 1.0; było 0.3 — wyglądało „płasko”).
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
//
// Problem: Specular highlight miga przy ruchu bo NdotH zmienia się szybko
// między sąsiednimi pikselami (sub-pixel geometry variation).
//
// Rozwiązanie: dFdx/dFdy normalnej dają wariancję normalnej w footprincie
// piksela. Zwiększamy roughness² proporcjonalnie do tej wariancji.
//
//   r_adjusted = sqrt(r² + κ * (|dN/dx| + |dN/dy|))
//
// κ=0.25 to ostrożna wartość (0=brak efektu, 1=agresywny).
// Clamp do [raw_roughness, 1.0] — nigdy nie zmniejszamy roughness.
//
// Efekt: sharp highlight na low-poly staje się stabilniejszy bez widocznego
// rozmycia materiału przy normalnym oglądaniu.
// ─────────────────────────────────────────────────────────────────────────────
fn specular_aa_roughness(n: vec3<f32>, roughness: f32) -> f32 {
    let variance = length(dpdx(n)) + length(dpdy(n));
    let kappa    = 0.25;
    return clamp(sqrt(roughness * roughness + kappa * variance), roughness, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] HEIGHT-CORRELATED SMITH GGX — poprawiona kolejność argumentów
//
// Poprzedni błąd: V_Smith miał zamienione ndl/ndv WEWNĄTRZ sqrt:
//   stara:  gv = ndl * sqrt(ndv²*(1-a²)+a²)  ← ndl na zewnątrz = OK
//           gl = ndv * sqrt(ndl²*(1-a²)+a²)  ← ndv na zewnątrz = OK
//   UWAGA: to akurat było poprawne! Ale sprawdź swój plik — jeśli
//   masz gv = ndl*sqrt(ndl²...) lub gl = ndv*sqrt(ndv²...) to jest błąd.
//
// Forma kanoniczna (Lagarde & de Rousiers 2014, "Moving Frostbite to PBR"):
//   Lambda_v = NdotL * sqrt( NdotV²*(1-α²) + α² )  ← mnoży NdotL, sqrt z NdotV
//   Lambda_l = NdotV * sqrt( NdotL²*(1-α²) + α² )  ← mnoży NdotV, sqrt z NdotL
//   Vis = 0.5 / (Lambda_v + Lambda_l)
//
// Zachowanie:
//   - rough (a²→1): Vis ≈ 0.5/(NdotL+NdotV) — miękkie krawędzie, brak ciemnych ring
//   - smooth(a²→0): Vis ≈ 0.25/(NdotL*NdotV) — standardowy GGX
// ─────────────────────────────────────────────────────────────────────────────
fn V_SmithGGX_Correlated(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let a2       = roughness * roughness * roughness * roughness;
    let lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2); // zewnątrz: NdotL, sqrt: NdotV
    let lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2); // zewnątrz: NdotV, sqrt: NdotL
    return 0.5 / max(lambda_v + lambda_l, EPSILON);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pozostałe BRDF — bez zmian
// ─────────────────────────────────────────────────────────────────────────────
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
// CLEARCOAT & SHEEN BRDF HELPERS
// ─────────────────────────────────────────────────────────────────────────────

// Scalar Fresnel-Schlick — for clearcoat f0=0.04 (IOR=1.5).
fn F_Schlick_f0(cos_t: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

// Kelemen-Szirmay-Kalos 2001 — cheap combined visibility term for clearcoat.
// Replaces Smith-GGX; ldh = LdotH = VdotH (half-angle cosine).
fn V_Kelemen(ldh: f32) -> f32 {
    return 0.25 / max(ldh * ldh, EPSILON);
}

// Charlie NDF — Estevez & Kulla 2017 ("Production Friendly Microfacet Sheen BRDF").
// Used for fabric retro-reflection. roughness is perceptual (passed directly, not squared).
fn D_Charlie(ndh: f32, roughness: f32) -> f32 {
    let inv_alpha = 1.0 / max(roughness, EPSILON);
    let sin2h     = max(1.0 - ndh * ndh, 0.0078125); // clamp: avoids pow(0, neg) in fp16
    return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) * 0.5 * INV_PI;
}

// Ashikhmin-Premoze 2007 — visibility term paired with the Charlie NDF.
fn V_Ashikhmin(ndv: f32, ndl: f32) -> f32 {
    return 1.0 / max(4.0 * (ndl + ndv - ndl * ndv), EPSILON);
}

// ─────────────────────────────────────────────────────────────────────────────
// IBL — [1] Specular LOD sterowany roughness
//
// textureSampleLevel(t_prefilter, s_ibl, r_rot, lod)
//   lod = perc_rough * MAX_PREFILTER_LOD
//   lod=0 → bliski mirror (niski roughness)
//   lod=4 → mocno rozmyty (wysoki roughness)
//
// Prefiltered env mapa jest wypiekana z GGX importance sampling dla każdego
// mip levelu (cs_prefilter w ibl_bake.wgsl) — więc LOD = roughness daje
// fizycznie poprawne rozmazanie refleksji.
// ─────────────────────────────────────────────────────────────────────────────
// Specular occlusion — przybliżenie Lagarde (nie świeci specular w głębokim AO).
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

    let irrad   = textureSample(t_irradiance, s_ibl, n_rot).rgb * IBL_SCALE;
    let F       = F_SchlickR(ndv, f0, perc_rough);
    let kd      = (1.0 - F) * (1.0 - metallic);
    let en_trans = max(1.0 - trans_m, 0.0);
    let diffuse = kd * albedo * irrad * en_trans;

    let lod       = perc_rough * MAX_PREFILTER_LOD;
    let env_color = textureSampleLevel(t_prefilter, s_ibl, r_rot, lod).rgb * IBL_SCALE;
    let brdf      = textureSample(t_brdf_lut, s_lut, vec2<f32>(ndv, perc_rough)).rg;
    let spec_occ  = specular_occlusion(ndv, ao);
    let specular  = env_color * (f0 * irid * brdf.x + brdf.y) * spec_occ;

    return (diffuse + specular) * ao;
}

// Anizotropowy NDF (GGX) — kierunek rys w przestrzeni stycznej (T_aniso, B_aniso).
fn D_GGX_aniso(ndh: f32, toh: f32, boh: f32, at: f32, ab: f32) -> f32 {
    let a2 = at * ab;
    let x  = (toh * toh) / max(at * at, EPSILON);
    let y  = (boh * boh) / max(ab * ab, EPSILON);
    let z  = ndh * ndh;
    let v  = x + y + z;
    return a2 / max(PI * v * v, EPSILON);
}

// Steep parallax mapping — spójne z height mapą 0..1 (jasny = wypukłość „wyżej” względem promienia).
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

// ─────────────────────────────────────────────────────────────────────────────
// [2] CSM Shadow — adaptacyjny bias per kaskada
//
// Dalsze kaskady pokrywają większy obszar world-space tym samym rozmiarem
// tekstury → texel jest większy → potrzeba więcej biasu.
//
// shadow_params.w = cascade_bias_multiplier (ustaw w shadow.rs na ~1.5)
// Finalny bias dla kaskady k = base_bias * mult^k
//   kaskada 0: base_bias * 1.0
//   kaskada 1: base_bias * 1.5
//   kaskada 2: base_bias * 2.25
//   kaskada 3: base_bias * 3.375
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

    // [2] Adaptacyjny bias: rośnie wykładniczo z indeksem kaskady
    let bias_mult  = max(csm.shadow_params.w, 1.0);
    let bias_scale = pow(bias_mult, f32(cascade));
    let bias       = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x)
                   * bias_scale;
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
            let next_bias  = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x)
                           * next_scale;
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
// Fragment
// ─────────────────────────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let N_geom = normalize(in.world_normal);
    let T0     = normalize(in.world_tan - N_geom * dot(N_geom, in.world_tan));
    let B0     = cross(N_geom, T0) * sign(dot(in.world_bitan, cross(N_geom, normalize(in.world_tan))));

    let v = normalize(camera.camera_pos.xyz - in.world_pos);
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

    // Anizotropia — obrót kierunku rys w płaszczyźnie stycznej (jak w SP / Filament).
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
                let cos_in = cos(l_data.params.z);
                let cos_out = cos(l_data.params.w);
                let spot_f = smoothstep(cos_out, cos_in, cos_theta);
                radiance = base * spot_f;
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
        if is_sun { shd = csm_shadow(in.world_pos, in.view_pos.z, ndl); }

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
        let att = material.ext_attenuation.xyz * material.ext_attenuation.w
                * max(material.ext_ior_trans.z, 0.0001);
        tcol *= vec3<f32>(exp(-att.x), exp(-att.y), exp(-att.z));
        out_rgb = mix(out_rgb, tcol, trans_m);
    }

    return vec4<f32>(out_rgb, alpha);
}
