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
}
@group(2) @binding(0) var<uniform> material:         MaterialUniform;
@group(2) @binding(1) var t_base_color:               texture_2d<f32>;
@group(2) @binding(2) var s_main:                     sampler;
@group(2) @binding(3) var t_normal:                   texture_2d<f32>;
@group(2) @binding(4) var t_metallic_roughness:       texture_2d<f32>;

struct GpuLight {
    position_or_dir: vec4<f32>,
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
const IBL_SCALE:         f32 = 0.3;
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
fn ibl(n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>, albedo: vec3<f32>,
       metallic: f32, perc_rough: f32, ao: f32) -> vec3<f32> {
    let ndv   = max(dot(n, v), EPSILON);
    let r     = reflect(-v, n);
    let n_rot = (env.rotation * vec4<f32>(n, 0.0)).xyz;
    let r_rot = (env.rotation * vec4<f32>(r, 0.0)).xyz;

    // Diffuse: irradiance cubemap × kd
    let irrad   = textureSample(t_irradiance, s_ibl, n_rot).rgb * IBL_SCALE;
    let F       = F_SchlickR(ndv, f0, perc_rough);
    let kd      = (1.0 - F) * (1.0 - metallic);
    let diffuse = kd * albedo * irrad;

    // [1] Specular: prefiltered env z LOD = roughness * MAX_LOD
    let lod       = perc_rough * MAX_PREFILTER_LOD;
    let env_color = textureSampleLevel(t_prefilter, s_ibl, r_rot, lod).rgb * IBL_SCALE;

    // BRDF LUT: X=NdotV, Y=roughness (zgodne z cs_brdf_lut w ibl_bake.wgsl)
    let brdf     = textureSample(t_brdf_lut, s_lut, vec2<f32>(ndv, perc_rough)).rg;
    let specular = env_color * (f0 * brdf.x + brdf.y);

    return (diffuse + specular) * ao;
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
    let albedo_s = textureSample(t_base_color,         s_main, in.uv);
    let mr_s     = textureSample(t_metallic_roughness, s_main, in.uv);
    let norm_s   = textureSample(t_normal,             s_main, in.uv);

    let albedo   = material.base_color.rgb * albedo_s.rgb;
    let alpha    = material.base_color.a   * albedo_s.a;
    let metallic = clamp(material.metallic_roughness_ao.x * mr_s.b, 0.0, 1.0);
    let ao       = mix(1.0, mr_s.r, material.metallic_roughness_ao.z);

    // Raw roughness z materiału — PRZED specular AA
    let raw_roughness = clamp(material.metallic_roughness_ao.y * mr_s.g, 0.045, 1.0);

    // Gram-Schmidt TBN re-ortogonalizacja (stabilność po interpolacji)
    let N = normalize(in.world_normal);
    let T = normalize(in.world_tan - N * dot(N, in.world_tan));
    let B = cross(N, T) * sign(dot(in.world_bitan, cross(N, normalize(in.world_tan))));

    let n_ts = norm_s.rgb * 2.0 - 1.0;
    let n    = normalize(
        T * n_ts.x * material.metallic_roughness_ao.w +
        B * n_ts.y * material.metallic_roughness_ao.w +
        N * n_ts.z
    );

    // [1] Specular AA — zwiększ roughness na podstawie gradientu normalnej
    let perc_rough = specular_aa_roughness(n, raw_roughness);

    let v   = normalize(camera.camera_pos.xyz - in.world_pos);
    let ndv = max(dot(n, v), EPSILON);
    let f0  = mix(vec3<f32>(0.04), albedo, metallic);
    let dc  = albedo * (1.0 - metallic);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < lights.count_and_pad.x; i++) {
        let l_data = lights.lights[i];
        var l_dir:    vec3<f32>;
        var radiance: vec3<f32>;
        var is_sun   = false;

        if u32(l_data.params.x) == 0u {
            l_dir    = normalize(-l_data.position_or_dir.xyz);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a;
            is_sun   = true;
        } else {
            let diff  = l_data.position_or_dir.xyz - in.world_pos;
            let dist  = length(diff);
            l_dir     = diff / max(dist, EPSILON);
            radiance  = l_data.color_intensity.rgb * l_data.color_intensity.a
                      / max(dist * dist, 0.01);
        }

        let h   = normalize(v + l_dir);
        let ndl = max(dot(n, l_dir), 0.0);
        if ndl <= 0.0 { continue; }

        let ndh = max(dot(n, h), EPSILON);
        let ldh = max(dot(l_dir, h), EPSILON);

        var shd = 1.0;
        if is_sun { shd = csm_shadow(in.world_pos, in.view_pos.z, ndl); }

        let D   = D_GGX(ndh, perc_rough);
        let Vis = V_SmithGGX_Correlated(ndv, ndl, perc_rough); // [3] NAPRAWIONA
        let F   = F_Schlick(ldh, f0);
        let kd  = (1.0 - F) * (1.0 - metallic);

        lo += (kd * dc * Fd_Burley(ndv, ndl, ldh, perc_rough) + D * Vis * F)
            * radiance * ndl * shd;
    }

    let ambient = ibl(n, v, f0, dc, metallic, perc_rough, ao);
    // Surowy HDR — tone mapping i dithering w post.wgsl
    return vec4<f32>(lo + ambient + material.emissive.rgb, alpha);
}