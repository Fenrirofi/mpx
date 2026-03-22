// ══════════════════════════════════════════════════════════════════════════════
//  ibl_bake.wgsl — Kulla-Conty edition
//
//  Zmiany vs poprzednia wersja:
//  [KC-1] cs_brdf_lut — kanał B teraz zawiera Ess(NdotV, roughness)
//         potrzebny do multiscatter IBL (Kulla-Conty 2017).
//         Kanał A i B bez zmian (split-sum scale/bias).
//  [KC-2] cs_eavg_lut — nowy compute pass generujący 1D LUT Eavg(roughness)
//         32×1 tekstura R16Float, całkuje Ess po hemisferze.
// ══════════════════════════════════════════════════════════════════════════════

const PI:      f32 = 3.14159265358979;
const TWO_PI:  f32 = 6.28318530717959;

struct BakeParams {
    face:      u32,
    roughness: f32,
    width:     u32,
    height:    u32,
}
@group(0) @binding(0) var<uniform> params: BakeParams;

// --- Utility ---
fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0,  -v,  -u)); }
        case 1u: { return normalize(vec3<f32>(-1.0,  -v,   u)); }
        case 2u: { return normalize(vec3<f32>(  u,   1.0,   v)); }
        case 3u: { return normalize(vec3<f32>(  u,  -1.0,  -v)); }
        case 4u: { return normalize(vec3<f32>(  u,   -v,  1.0)); }
        default: { return normalize(vec3<f32>( -u,   -v, -1.0)); }
    }
}

fn dir_to_equirect(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        atan2(d.x, d.z) / TWO_PI + 0.5,
        asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5,
    );
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = (bits_in << 16u) | (bits_in >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a   = roughness * roughness;
    let phi = TWO_PI * xi.x;
    let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let h_local = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    let up      = select(vec3<f32>(0., 1., 0.), vec3<f32>(1., 0., 0.), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitan   = cross(n, tangent);
    return normalize(tangent * h_local.x + bitan * h_local.y + n * h_local.z);
}

fn distribution_ggx(nh: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = (nh * nh * (a2 - 1.0) + 1.0);
    return a2 / (PI * d * d);
}

// Height-correlated Smith G2 (używany w Ess)
fn V_SmithGGX_Correlated_lut(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let a2       = roughness * roughness * roughness * roughness;
    let lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 0.00001);
}

// ══════════════════════════════════════════════════════════════════════════════
//  PASS A — Equirectangular → Cubemap
// ══════════════════════════════════════════════════════════════════════════════
@group(0) @binding(1) var t_equirect: texture_2d<f32>;
@group(0) @binding(2) var s_linear:   sampler;
@group(0) @binding(3) var out_face:   texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_equirect_to_face(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv  = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let dir = face_uv_to_dir(params.face, uv);
    let euv = dir_to_equirect(dir);
    let col = textureSampleLevel(t_equirect, s_linear, euv, 0.0);
    textureStore(out_face, vec2<i32>(gid.xy), col);
}

// ══════════════════════════════════════════════════════════════════════════════
//  PASS B — Irradiance
// ══════════════════════════════════════════════════════════════════════════════
@group(0) @binding(1) var t_env_cube_irrad: texture_cube<f32>;
@group(0) @binding(3) var out_irrad: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let n  = face_uv_to_dir(params.face, uv);

    var irradiance = vec3<f32>(0.0);
    let SAMPLES    = 512u;
    for (var i = 0u; i < SAMPLES; i++) {
        let xi    = hammersley(i, SAMPLES);
        let phi   = TWO_PI * xi.x;
        let cos_t = sqrt(1.0 - xi.y);
        let sin_t = sqrt(xi.y);
        let h_local = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
        let up      = select(vec3<f32>(0., 1., 0.), vec3<f32>(1., 0., 0.), abs(n.y) > 0.999);
        let tangent = normalize(cross(up, n));
        let bitan   = cross(n, tangent);
        let world   = tangent * h_local.x + bitan * h_local.y + n * h_local.z;
        irradiance += textureSampleLevel(t_env_cube_irrad, s_linear, world, 0.0).rgb;
    }
    irradiance /= f32(SAMPLES);
    textureStore(out_irrad, vec2<i32>(gid.xy), vec4<f32>(irradiance, 1.0));
}

// ══════════════════════════════════════════════════════════════════════════════
//  PASS C — Specular Prefilter
// ══════════════════════════════════════════════════════════════════════════════
@group(0) @binding(1) var t_env_prefilter: texture_cube<f32>;
@group(0) @binding(3) var out_prefilter: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let n  = face_uv_to_dir(params.face, uv);
    let r  = params.roughness;
    let v  = n;

    let SAMPLES      = 1024u;
    var total_weight = 0.0;
    var color        = vec3<f32>(0.0);

    for (var i = 0u; i < SAMPLES; i++) {
        let xi  = hammersley(i, SAMPLES);
        let h   = importance_sample_ggx(xi, n, r);
        let l   = normalize(2.0 * dot(v, h) * h - v);
        let nl  = max(dot(n, l), 0.0);

        if (nl > 0.0) {
            let nh  = max(dot(n, h), 0.0);
            let vh  = max(dot(v, h), 0.0);
            let pdf = (distribution_ggx(nh, r) * nh) / (4.0 * vh) + 0.0001;
            let sa_sample = 1.0 / (f32(SAMPLES) * pdf + 0.0001);
            let sa_texel  = 4.0 * PI / (6.0 * f32(params.width) * f32(params.width));
            let lod       = select(0.5 * log2(sa_sample / sa_texel), 0.0, r == 0.0);
            color        += textureSampleLevel(t_env_prefilter, s_linear, l, lod).rgb * nl;
            total_weight += nl;
        }
    }
    textureStore(out_prefilter, vec2<i32>(gid.xy), vec4<f32>(color / max(total_weight, 0.001), 1.0));
}

// ══════════════════════════════════════════════════════════════════════════════
//  PASS D — BRDF integration LUT
//
//  [KC-1] Kanał B (brdf.b) = Ess(NdotV, roughness)
//         Ess = całka (1-F0) * DGF / (4 NdotV NdotL) po hemisferze
//         przy F0=1 (white furnace), wynik to "ile energii specular wraca"
//         dla danego NdotV i roughness.
//         Używane w pbr.wgsl do kompensacji multiscatter.
// ══════════════════════════════════════════════════════════════════════════════
fn geometry_schlick_ggx_ibl(ndv: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    return ndv / (ndv * (1.0 - k) + k);
}

@group(0) @binding(1) var out_lut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_brdf_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    // x-axis = NdotV, y-axis = roughness (identycznie jak przed)
    let ndv       = (f32(gid.x) + 0.5) / f32(params.width);
    let roughness = (f32(gid.y) + 0.5) / f32(params.height);
    let SAMPLES   = 1024u;

    var A   = 0.0;  // split-sum scale (F0 term)
    var B   = 0.0;  // split-sum bias  (1-F0 term)
    var Ess = 0.0;  // [KC-1] directional albedo przy F0=1

    let n = vec3<f32>(0., 0., 1.);
    let v = vec3<f32>(sqrt(1.0 - ndv * ndv), 0., ndv);

    for (var i = 0u; i < SAMPLES; i++) {
        let xi  = hammersley(i, SAMPLES);
        let h   = importance_sample_ggx(xi, n, roughness);
        let l   = normalize(2.0 * dot(v, h) * h - v);
        let nl  = max(l.z, 0.0);
        let nh  = max(h.z, 0.0);
        let vh  = max(dot(v, h), 0.0);

        if (nl > 0.0) {
            // --- split-sum (A, B) ---
            let G   = geometry_schlick_ggx_ibl(ndv, roughness)
                    * geometry_schlick_ggx_ibl(nl,  roughness);
            let G_v = G * vh / (nh * ndv);
            let Fc  = pow(1.0 - vh, 5.0);
            A      += G_v * (1.0 - Fc);
            B      += G_v * Fc;

            // --- [KC-1] Ess przy F0=1 (white furnace) ---
            // Używamy height-correlated Smith G2 — spójne z pbr.wgsl
            let Vis  = V_SmithGGX_Correlated_lut(ndv, nl, roughness);
            // DGF / (4 NdotV NdotL) * NdotL = Vis * D (już całkujemy po próbkach GGX)
            // przy importance sampling GGX pdf eliminuje D, zostaje: G2 * vh / (nh * ndv)
            // identycznie jak A+B, ale Fc=1 przez cały czas (F0=1)
            Ess    += G_v;  // G_v * (Fc + (1-Fc)) = G_v * 1 = G_v
        }
    }

    let inv_n = 1.0 / f32(SAMPLES);
    textureStore(out_lut, vec2<i32>(gid.xy), vec4<f32>(
        A   * inv_n,
        B   * inv_n,
        Ess * inv_n,   // [KC-1] kanał B
        1.0,
    ));
}

// ══════════════════════════════════════════════════════════════════════════════
//  PASS E — Eavg 1D LUT  [KC-2]
//
//  Eavg(roughness) = (2 / 1) * integral_0^1 Ess(NdotV, roughness) * NdotV dNdotV
//
//  Tekstura: 32×1 R16Float
//  Oś X: roughness ∈ [0, 1]
//  Wynik: średnia energia specularu po wszystkich kątach widzenia.
//
//  W pbr.wgsl: ms_weight = (1 - Ess) / (1 - Eavg)
//              ms_color  = lerp(1, f0, metallic)
//              specular  = specular_ss + env_color * ms_weight * ms_color * (1 - Ess)
// ══════════════════════════════════════════════════════════════════════════════
@group(0) @binding(1) var out_eavg: texture_storage_2d<r16float, write>;

@compute @workgroup_size(32, 1, 1)
fn cs_eavg_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width) { return; }

    let roughness = (f32(gid.x) + 0.5) / f32(params.width);
    let SAMPLES   = 32u;   // zewnętrzna całka po NdotV — 32 próbki wystarczą

    var eavg = 0.0;

    // Numerycznie całkujemy: Eavg = 2 * integral_0^1 Ess(mu, r) * mu d_mu
    // Ess(mu, r) liczymy wewnętrzną pętlą Hammersley (identycznie jak w cs_brdf_lut)
    for (var outer = 0u; outer < SAMPLES; outer++) {
        let ndv     = (f32(outer) + 0.5) / f32(SAMPLES);
        let n       = vec3<f32>(0., 0., 1.);
        let v       = vec3<f32>(sqrt(1.0 - ndv * ndv), 0., ndv);
        let INNER   = 256u;
        var ess_sum = 0.0;

        for (var i = 0u; i < INNER; i++) {
            let xi  = hammersley(i, INNER);
            let h   = importance_sample_ggx(xi, n, roughness);
            let l   = normalize(2.0 * dot(v, h) * h - v);
            let nl  = max(l.z, 0.0);
            let nh  = max(h.z, 0.0);
            let vh  = max(dot(v, h), 0.0);

            if (nl > 0.0) {
                let G   = geometry_schlick_ggx_ibl(ndv, roughness)
                        * geometry_schlick_ggx_ibl(nl,  roughness);
                let G_v = G * vh / (nh * ndv);
                ess_sum += G_v;
            }
        }
        let ess  = ess_sum / f32(INNER);
        eavg    += 2.0 * ess * ndv;   // cosine-weighted sum
    }
    eavg /= f32(SAMPLES);

    textureStore(out_eavg, vec2<i32>(i32(gid.x), 0), vec4<f32>(eavg, 0., 0., 1.));
}
