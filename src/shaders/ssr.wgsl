// ══════════════════════════════════════════════════════════════════════════════
//  SSR — Screen Space Reflections
//  Compute shader: raymarch w przestrzeni ekranu, zwraca kolor refleksji
// ══════════════════════════════════════════════════════════════════════════════

struct SsrParams {
    proj:           mat4x4<f32>,
    inv_proj:       mat4x4<f32>,
    view:           mat4x4<f32>,
    texel_size:     vec2<f32>,
    max_steps:      u32,
    max_distance:   f32,
    thickness:      f32,   // grubość powierzchni (tolerancja głębii)
    stride:         f32,   // krok raymarchu w pikselach
    _pad:           vec2<f32>,
}

@group(0) @binding(0) var<uniform>  ssr:        SsrParams;
@group(0) @binding(1) var t_hdr:                texture_2d<f32>;    // bieżąca klatka HDR
@group(0) @binding(2) var t_depth:              texture_depth_2d;
@group(0) @binding(3) var t_normal:             texture_2d<f32>;    // world-space normals (gbuffer)
@group(0) @binding(4) var t_metallic_roughness: texture_2d<f32>;    // roughness w G (metallic w B)
@group(0) @binding(5) var s_linear:             sampler;
@group(0) @binding(6) var s_nearest:            sampler;
@group(0) @binding(7) var out_ssr:              texture_storage_2d<rgba16float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc  = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = ssr.inv_proj * ndc;
    return view.xyz / view.w;
}

fn view_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = ssr.proj * vec4<f32>(view_pos, 1.0);
    let ndc  = clip.xyz / clip.w;
    return ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
}

fn load_depth(uv: vec2<f32>) -> f32 {
    let dims = textureDimensions(t_depth);
    let c    = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - 1);
    return textureLoad(t_depth, c, 0);
}

// ── Raymarch ──────────────────────────────────────────────────────────────────

struct RayHit {
    hit:    bool,
    hit_uv: vec2<f32>,
    fade:   f32,   // zanik na krawędziach ekranu i przy dużej chropowatości
}

fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RayHit {
    var result: RayHit;
    result.hit  = false;
    result.fade = 1.0;

    // Krok w pikselach (skaluj do view-space)
    var step_size = ssr.stride;
    var pos = origin;

    for (var i = 0u; i < ssr.max_steps; i++) {
        pos += dir * step_size;
        step_size *= 1.05; // rosnący krok = mniej iteracji na dalekich

        // Projekcja na ekran
        let uv = view_to_uv(pos);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) { break; }

        let scene_depth = load_depth(uv);
        let scene_pos   = view_pos_from_depth(uv, scene_depth);

        let ray_depth   = pos.z;
        let diff        = scene_pos.z - ray_depth;

        // Trafienie: ray jest za powierzchnią, ale w granicach thickness
        if diff > 0.0 && diff < ssr.thickness {
            result.hit    = true;
            result.hit_uv = uv;
            // Zanik przy krawędziach ekranu
            let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            result.fade = smoothstep(0.0, 0.1, edge);
            break;
        }
    }
    return result;
}

// ── Compute ───────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn cs_ssr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_ssr);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let fdims = vec2<f32>(dims);
    let uv    = (vec2<f32>(gid.xy) + 0.5) / fdims;
    let coord = vec2<i32>(gid.xy);

    // Roughness — SSR niewidoczne na bardzo szorstkich powierzchniach
    let mr     = textureSampleLevel(t_metallic_roughness, s_nearest, uv, 0.0);
    let rough  = mr.g;
    let metal  = mr.b;

    // Refleksje widoczne tylko na gładkich (rough < 0.5) lub metalicznych powierzchniach
    if rough > 0.5 && metal < 0.1 {
        textureStore(out_ssr, coord, vec4<f32>(0.0));
        return;
    }

    let depth = textureLoad(t_depth, coord, 0);
    if depth >= 0.9999 {
        textureStore(out_ssr, coord, vec4<f32>(0.0));
        return;
    }

    // View-space pozycja i normalna
    let view_pos = view_pos_from_depth(uv, depth);
    let world_n  = textureSampleLevel(t_normal, s_nearest, uv, 0.0).xyz * 2.0 - 1.0;
    let view_n   = normalize((ssr.view * vec4<f32>(world_n, 0.0)).xyz);

    // Kierunek refleksji w view-space
    let view_dir = normalize(view_pos);
    let refl_dir = normalize(reflect(view_dir, view_n));

    // Refleksje w górę (za kamerą) można pominąć
    if refl_dir.z > -0.01 {
        textureStore(out_ssr, coord, vec4<f32>(0.0));
        return;
    }

    let hit = raymarch(view_pos, refl_dir);

    var out_color = vec4<f32>(0.0);
    if hit.hit {
        let refl_color = textureSampleLevel(t_hdr, s_linear, hit.hit_uv, 0.0).rgb;
        // Zanik przez roughness (bardziej szorstkie = słabsze refleksje)
        let rough_fade  = 1.0 - smoothstep(0.0, 0.5, rough);
        // Zanik przez kąt grazing (Fresnel-like)
        let fresnel     = pow(1.0 - max(dot(-view_dir, view_n), 0.0), 5.0);
        let strength    = rough_fade * hit.fade * mix(0.04, 1.0, fresnel);
        out_color       = vec4<f32>(refl_color * strength, strength);
    }

    textureStore(out_ssr, coord, out_color);
}
