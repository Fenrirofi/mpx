// ══════════════════════════════════════════════════════════════════════════════
//  PBR Shader — CSM + IBL + PCF  —  POPRAWIONA WERSJA
//
//  ZMIANA #3 (Bug #3): Naprawiona funkcja V_Smith (height-correlated GGX).
//
//  Przyczyna błędu: poprzednia implementacja miała zamienione argumenty ndl/ndv
//  w wewnętrznych wyrażeniach sqrt. Poprawna forma (Lagarde & de Rousiers 2014,
//  "Moving Frostbite to PBR", eq. 72) to:
//
//    Vis = 0.5 / (NdotL * sqrt(NdotV² * (1-α²) + α²)
//                + NdotV * sqrt(NdotL² * (1-α²) + α²))
//
//  W praktyce: term "dla view" (gv) mnoży NdotL na zewnątrz i ma NdotV w sqrt.
//              term "dla light" (gl) mnoży NdotV na zewnątrz i ma NdotL w sqrt.
//
//  Stara wersja miała je odwrotnie, co powodowało błędne przyciemnianie/
//  rozjaśnianie specular pod kątem glancing (ndl lub ndv → 0).
//
//  ZMIANA #4 (dodatkowa): Zabezpieczenie TBN przed degeneracją w fragment shaderze.
//  normalize() na wejściu do TBN jest konieczny, ale przy interpolacji GPU
//  wektory mogą mieć length_squared bliski 0 — dodano bezpieczniejszą
//  re-ortogonalizację Gram-Schmidt w fragment shaderze.
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
    shadow_params:   vec4<f32>,
}

@group(3) @binding(0) var<uniform> lights:      LightArrayUniform;
@group(3) @binding(1) var<uniform> csm:         CsmPbrUniform;
@group(3) @binding(2) var t_shadow_map:          texture_depth_2d_array;
@group(3) @binding(3) var s_shadow:              sampler_comparison;
@group(3) @binding(4) var t_irradiance:          texture_cube<f32>;
@group(3) @binding(5) var t_prefilter:           texture_cube<f32>;
@group(3) @binding(6) var t_brdf_lut:            texture_2d<f32>;
@group(3) @binding(7) var s_ibl:                 sampler;
@group(3) @binding(8) var s_lut:                 sampler;

struct EnvUniform { rotation: mat4x4<f32>, }
@group(3) @binding(9) var<uniform> env: EnvUniform;

const PI:              f32 = 3.14159265358979;
const INV_PI:          f32 = 0.31830988618;
const EPSILON:         f32 = 0.00001;
const IBL_SCALE:       f32 = 0.3;
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
    let w4           = object.model * vec4<f32>(in.position, 1.0);
    out.world_pos    = w4.xyz;
    out.clip_pos     = camera.view_proj * w4;
    let v4           = camera.view * w4;
    out.view_pos     = v4.xyz;
    let n = normalize((object.normal_matrix * vec4<f32>(in.normal,      0.0)).xyz);
    let t = normalize((object.normal_matrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_normal = n;
    out.world_tan    = t;
    out.world_bitan  = cross(n, t) * in.tangent.w;
    out.uv           = in.uv;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// PBR Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn D_GGX(ndh: f32, r: f32) -> f32 {
    let a2 = r * r * r * r;
    let d  = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, EPSILON);
}

// ─────────────────────────────────────────────────────────────────────────────
// ZMIANA #3: NAPRAWIONA V_Smith (height-correlated GGX)
//
// Poprawna forma (Lagarde 2014):
//   Vis = 0.5 / (NdotL * sqrt(NdotV²*(1-a²) + a²)
//              + NdotV * sqrt(NdotL²*(1-a²) + a²))
//
// Kluczowe: gv mnoży NdotL na zewnątrz, a NdotV jest WEWNĄTRZ sqrt.
//           gl mnoży NdotV na zewnątrz, a NdotL jest WEWNĄTRZ sqrt.
// Poprzednia wersja miała te dwa terminy odwrócone miejscami.
// ─────────────────────────────────────────────────────────────────────────────
fn V_SmithGGX(ndv: f32, ndl: f32, r: f32) -> f32 {
    let a2 = r * r * r * r;
    //          ↓ zewnętrzny czynnik   ↓ wewnętrzny sqrt z DRUGIM kątem
    let gv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);   // term view
    let gl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);   // term light
    return 0.5 / max(gv + gl, EPSILON);
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
// IBL
// ─────────────────────────────────────────────────────────────────────────────
fn ibl(
    n:          vec3<f32>,
    v:          vec3<f32>,
    f0:         vec3<f32>,
    albedo:     vec3<f32>,
    metallic:   f32,
    perc_rough: f32,
    ao:         f32,
) -> vec3<f32> {
    let ndv   = max(dot(n, v), EPSILON);
    let r     = reflect(-v, n);
    let n_rot = (env.rotation * vec4<f32>(n, 0.0)).xyz;
    let r_rot = (env.rotation * vec4<f32>(r, 0.0)).xyz;

    let irrad   = textureSample(t_irradiance, s_ibl, n_rot).rgb * IBL_SCALE;
    let F       = F_SchlickR(ndv, f0, perc_rough);
    let kd      = (1.0 - F) * (1.0 - metallic);
    let diffuse = kd * albedo * irrad;

    let lod       = perc_rough * MAX_PREFILTER_LOD;
    let env_color = textureSampleLevel(t_prefilter, s_ibl, r_rot, lod).rgb * IBL_SCALE;
    let brdf_uv   = vec2<f32>(ndv, perc_rough);  // X=NdotV, Y=roughness — zgodne z ibl_bake.wgsl
    let brdf      = textureSample(t_brdf_lut, s_lut, brdf_uv).rg;
    let specular  = env_color * (f0 * brdf.x + brdf.y);

    return (diffuse + specular) * ao;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSM Shadow
// ─────────────────────────────────────────────────────────────────────────────
fn csm_shadow(world_pos: vec3<f32>, view_z: f32, ndl: f32) -> f32 {
    var cascade = 3u;
    let near = 0.05;
    let d    = -view_z - near;
    if d < csm.cascade_splits.x {
        cascade = 0u;
    } else if d < csm.cascade_splits.x + csm.cascade_splits.y {
        cascade = 1u;
    } else if d < csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z {
        cascade = 2u;
    }

    let lp  = csm.light_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let p   = lp.xyz / lp.w;
    let uv  = vec2<f32>(p.x * 0.5 + 0.5, p.y * -0.5 + 0.5);

    if any(uv < vec2<f32>(0.01)) || any(uv > vec2<f32>(0.99)) { return 1.0; }

    let bias  = max(csm.shadow_params.y * (1.0 - ndl), csm.shadow_params.x);
    let texel = csm.shadow_params.z;

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
            select(
                csm.cascade_splits.x + csm.cascade_splits.y,
                csm.cascade_splits.x + csm.cascade_splits.y + csm.cascade_splits.z,
                cascade == 2u,
            ),
            cascade == 1u,
        );
        let blend_zone = split * 0.1;
        let blend_t    = clamp((d - (split - blend_zone)) / blend_zone, 0.0, 1.0);
        if blend_t > 0.0 {
            let next = cascade + 1u;
            let lp2  = csm.light_view_proj[next] * vec4<f32>(world_pos, 1.0);
            let p2   = lp2.xyz / lp2.w;
            let uv2  = vec2<f32>(p2.x * 0.5 + 0.5, p2.y * -0.5 + 0.5);
            var pcf2 = 0.0;
            if all(uv2 >= vec2<f32>(0.01)) && all(uv2 <= vec2<f32>(0.99)) {
                for (var sx = -1; sx <= 1; sx++) {
                    for (var sy = -1; sy <= 1; sy++) {
                        let off = vec2<f32>(f32(sx), f32(sy)) * texel;
                        pcf2 += textureSampleCompareLevel(t_shadow_map, s_shadow,
                                                          uv2 + off, i32(next), p2.z - bias);
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

    let albedo     = material.base_color.rgb * albedo_s.rgb;
    let alpha      = material.base_color.a   * albedo_s.a;
    let metallic   = clamp(material.metallic_roughness_ao.x * mr_s.b, 0.0, 1.0);
    let perc_rough = clamp(material.metallic_roughness_ao.y * mr_s.g, 0.045, 1.0);
    let ao         = mix(1.0, mr_s.r, material.metallic_roughness_ao.z);

    // ─────────────────────────────────────────────────────────────────────────
    // ZMIANA #4: Gram-Schmidt re-ortogonalizacja TBN w fragment shaderze
    //
    // Po interpolacji rasteryzera wektory T/B/N mogą stracić ortogonalność
    // (szczególnie widoczne na biegunach sfery przy niskiej rozdzielczości siatki).
    // Gram-Schmidt w FS kosztuje 2 dot + 2 multiply, ale eliminuje artefakty.
    //
    // Kolejność: najpierw re-ortogonalizujemy T względem N, potem odbudowujemy B.
    // ─────────────────────────────────────────────────────────────────────────
    let N_raw = normalize(in.world_normal);
    let T_raw = normalize(in.world_tan);
    // Gram-Schmidt: T' = normalize(T - N * dot(N, T))
    let T_gs  = normalize(T_raw - N_raw * dot(N_raw, T_raw));
    // Bitangent odbudowany z N i T (nie interpolowany), z zachowaniem znaku
    let B_gs  = cross(N_raw, T_gs) * sign(dot(in.world_bitan, cross(N_raw, T_raw)));

    let n_ts = norm_s.rgb * 2.0 - 1.0;
    let n    = normalize(
        T_gs  * n_ts.x * material.metallic_roughness_ao.w +
        B_gs  * n_ts.y * material.metallic_roughness_ao.w +
        N_raw * n_ts.z
    );

    let v   = normalize(camera.camera_pos.xyz - in.world_pos);
    let ndv = max(dot(n, v), EPSILON);
    let f0  = mix(vec3<f32>(0.04), albedo, metallic);
    let dc  = albedo * (1.0 - metallic);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < lights.count_and_pad.x; i++) {
        let l_data = lights.lights[i];
        var l_dir:    vec3<f32>;
        var radiance: vec3<f32>;
        var is_sun = false;

        if u32(l_data.params.x) == 0u {
            l_dir    = normalize(-l_data.position_or_dir.xyz);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a;
            is_sun   = true;
        } else {
            let diff = l_data.position_or_dir.xyz - in.world_pos;
            let dist = length(diff);
            l_dir    = diff / max(dist, EPSILON);
            let atten = 1.0 / max(dist * dist, 0.01);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a * atten;
        }

        let h   = normalize(v + l_dir);
        let ndl = max(dot(n, l_dir), 0.0);
        if ndl <= 0.0 { continue; }

        let ndh = max(dot(n, h), EPSILON);
        let ldh = max(dot(l_dir, h), EPSILON);

        var shd = 1.0;
        if is_sun {
            shd = csm_shadow(in.world_pos, in.view_pos.z, ndl);
        }

        let D   = D_GGX(ndh, perc_rough);
        let Vis = V_SmithGGX(ndv, ndl, perc_rough);   // ← ZMIANA: poprawna nazwa
        let F   = F_Schlick(ldh, f0);
        let kd  = (1.0 - F) * (1.0 - metallic);

        lo += (kd * dc * Fd_Burley(ndv, ndl, ldh, perc_rough) + D * Vis * F)
            * radiance * ndl * shd;
    }

    let ambient = ibl(n, v, f0, dc, metallic, perc_rough, ao);
    let color   = lo + ambient + material.emissive.rgb;

    // Surowy HDR — tone mapping i gamma w post.wgsl
    return vec4<f32>(color, alpha);
}