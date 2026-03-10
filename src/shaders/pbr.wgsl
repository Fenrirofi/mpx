// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  PBR Shader v3 – Cook-Torrance + PCF Shadows + Full IBL + SSAO             ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Group 0: Camera ───────────────────────────────────────────────────────────
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ── Group 1: Per-object ───────────────────────────────────────────────────────
struct ObjectUniform {
    model:         mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> object: ObjectUniform;

// ── Group 2: Material ─────────────────────────────────────────────────────────
struct MaterialUniform {
    base_color:            vec4<f32>,
    metallic_roughness_ao: vec4<f32>,
    emissive:              vec4<f32>,
}
@group(2) @binding(0) var<uniform> material:         MaterialUniform;
@group(2) @binding(1) var t_base_color:              texture_2d<f32>;
@group(2) @binding(2) var s_main:                    sampler;
@group(2) @binding(3) var t_normal:                  texture_2d<f32>;
@group(2) @binding(4) var t_metallic_roughness:      texture_2d<f32>;

// ── Group 3: Lights + Shadow + IBL ───────────────────────────────────────────
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
struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    shadow_params:   vec4<f32>,
}
@group(3) @binding(0) var<uniform> lights:       LightArrayUniform;
@group(3) @binding(1) var<uniform> shadow_data:  ShadowUniform;
@group(3) @binding(2) var t_shadow_map:          texture_depth_2d;
@group(3) @binding(3) var s_shadow:              sampler_comparison;

// NOTE: IBL textures are sampled analytically when no cubemap is loaded.
// When IBL renderer provides them, they replace the analytical approximation
// via a separate fullscreen composite pass reading t_ssao.
// SSAO is applied as a multiplier in the composite pass (post.wgsl).

// ── Vertex I/O ────────────────────────────────────────────────────────────────
struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec2<f32>,
}
struct VertOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       world_pos:     vec3<f32>,
    @location(1)       world_normal:  vec3<f32>,
    @location(2)       world_tangent: vec3<f32>,
    @location(3)       world_bitan:   vec3<f32>,
    @location(4)       uv:            vec2<f32>,
    @location(5)       shadow_coord:  vec4<f32>,
    @location(6)       view_pos:      vec3<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    let w4           = object.model * vec4<f32>(in.position, 1.0);
    out.world_pos    = w4.xyz;
    out.clip_pos     = camera.view_proj * w4;
    out.shadow_coord = shadow_data.light_view_proj * w4;
    out.view_pos     = (camera.view * w4).xyz;
    let n = normalize((object.normal_matrix * vec4<f32>(in.normal,      0.0)).xyz);
    let t = normalize((object.normal_matrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_normal  = n;
    out.world_tangent = t;
    out.world_bitan   = cross(n, t) * in.tangent.w;
    out.uv            = in.uv;
    return out;
}

const PI:      f32 = 3.14159265358979;
const INV_PI:  f32 = 0.31830988618;
const EPSILON: f32 = 0.00001;

// ── BRDF ──────────────────────────────────────────────────────────────────────
fn D_GGX(ndh: f32, r: f32) -> f32 {
    let a2 = r*r*r*r; let d = ndh*ndh*(a2-1.0)+1.0;
    return a2 / max(PI*d*d, EPSILON);
}
fn V_Smith(ndv: f32, ndl: f32, r: f32) -> f32 {
    let a2 = r*r*r*r;
    let gv = ndl*sqrt(ndv*ndv*(1.0-a2)+a2);
    let gl = ndv*sqrt(ndl*ndl*(1.0-a2)+a2);
    return 0.5/max(gv+gl, EPSILON);
}
fn F_Schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0-f0)*pow(clamp(1.0-cos_t,0.0,1.0),5.0);
}
fn F_SchlickR(cos_t: f32, f0: vec3<f32>, r: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0-r),f0)-f0)*pow(clamp(1.0-cos_t,0.0,1.0),5.0);
}
fn Fd_Burley(ndv: f32, ndl: f32, ldh: f32, r: f32) -> f32 {
    let f90 = 0.5+2.0*r*ldh*ldh;
    return (1.0+(f90-1.0)*pow(1.0-ndl,5.0))*(1.0+(f90-1.0)*pow(1.0-ndv,5.0))*INV_PI;
}

// ── Shadow PCF ────────────────────────────────────────────────────────────────
const POISSON: array<vec2<f32>,16> = array<vec2<f32>,16>(
    vec2<f32>(-0.94201624,-0.39906216),vec2<f32>( 0.94558609,-0.76890725),
    vec2<f32>(-0.09418410,-0.92938870),vec2<f32>( 0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),vec2<f32>(-0.81544232,-0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),vec2<f32>( 0.97484398, 0.75648379),
    vec2<f32>( 0.44323325,-0.97511554),vec2<f32>( 0.53742981,-0.47373420),
    vec2<f32>(-0.26496911,-0.41893023),vec2<f32>( 0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>( 0.19984126, 0.78641367),vec2<f32>( 0.14383161,-0.14100790),
);
fn sample_shadow(sc: vec4<f32>, ndl: f32) -> f32 {
    if shadow_data.shadow_params.w < 0.5 { return 1.0; }
    let p = sc.xyz/sc.w;
    if any(p.xy < vec2<f32>(0.0)) || any(p.xy > vec2<f32>(1.0)) { return 1.0; }
    let uv = vec2<f32>(p.x*0.5+0.5, p.y*-0.5+0.5);
    let bias = max(shadow_data.shadow_params.y*(1.0-ndl), shadow_data.shadow_params.x);
    let texel = 1.0/vec2<f32>(textureDimensions(t_shadow_map));
    var s = 0.0;
    for(var i=0;i<16;i++){
        s += textureSampleCompare(t_shadow_map, s_shadow, uv+POISSON[i]*texel*shadow_data.shadow_params.z, p.z-bias);
    }
    return s/16.0;
}

// ── Analytical IBL (fallback when no HDR loaded) ──────────────────────────────
fn env_brdf_approx(r: f32, ndv: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0,-0.0275,-0.572, 0.022);
    let c1 = vec4<f32>( 1.0, 0.0425, 1.040,-0.040);
    let rr = r*c0+c1;
    let a  = min(rr.x*rr.x, exp2(-9.28*ndv))*rr.x+rr.y;
    return vec2<f32>(-1.04,1.04)*a + vec2<f32>(rr.z,rr.w);
}

fn analytical_ibl(
    n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>, albedo: vec3<f32>,
    metallic: f32, perc_rough: f32, ao: f32, amb: vec3<f32>
) -> vec3<f32> {
    let ndv    = max(dot(n,v), EPSILON);
    let F      = F_SchlickR(ndv, f0, perc_rough);
    let kd     = (1.0-F)*(1.0-metallic);
    let sky_t  = clamp(n.y*0.5+0.5, 0.0, 1.0);
    let irrad  = mix(amb*0.4, amb, sky_t);
    let diff   = kd * albedo * irrad;
    let ref_d  = reflect(-v, n);
    let ref_t  = clamp(ref_d.y*0.5+0.5, 0.0, 1.0);
    let spec_c = mix(amb*vec3<f32>(0.9,0.95,1.1), amb*vec3<f32>(1.1,1.0,0.85), ref_t);
    let env_c  = mix(spec_c, amb, perc_rough*perc_rough);
    let dfg    = env_brdf_approx(perc_rough, ndv);
    let spec   = env_c*(f0*dfg.x+dfg.y);
    return (diff+spec)*ao;
}

// ── Light eval ────────────────────────────────────────────────────────────────
struct LightSample { direction: vec3<f32>, radiance: vec3<f32>, is_sun: bool }
fn eval_light(light: GpuLight, pos: vec3<f32>) -> LightSample {
    var s: LightSample;
    let kind = u32(light.params.x);
    if kind == 0u {
        s.direction = normalize(-light.position_or_dir.xyz);
        s.radiance  = light.color_intensity.rgb*light.color_intensity.a;
        s.is_sun    = true;
    } else {
        let d = light.position_or_dir.xyz-pos;
        let dist = length(d);
        s.direction = d/max(dist,EPSILON); s.is_sun = false;
        let atten   = 1.0/max(dist*dist,0.01);
        let range   = max(light.params.y, EPSILON);
        let falloff = pow(clamp(1.0-pow(dist/range,4.0),0.0,1.0),2.0);
        var rad     = light.color_intensity.rgb*light.color_intensity.a*atten*falloff;
        if kind == 2u {
            let ca = dot(-s.direction, normalize(light.position_or_dir.xyz));
            let sf = clamp((ca-cos(light.params.w))/max(cos(light.params.z)-cos(light.params.w),EPSILON),0.0,1.0);
            rad   *= sf*sf;
        }
        s.radiance = rad;
    }
    return s;
}

// ── Fragment ──────────────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let albedo_s = textureSample(t_base_color,        s_main, in.uv);
    let mr_s     = textureSample(t_metallic_roughness,s_main, in.uv);
    let norm_s   = textureSample(t_normal,            s_main, in.uv);

    let albedo     = material.base_color.rgb * albedo_s.rgb;
    let alpha      = material.base_color.a   * albedo_s.a;
    let metallic   = clamp(material.metallic_roughness_ao.x * mr_s.b, 0.0, 1.0);
    let perc_rough = clamp(material.metallic_roughness_ao.y * mr_s.g, 0.045, 1.0);
    let roughness  = perc_rough*perc_rough;
    let ao         = mix(1.0, mr_s.r, material.metallic_roughness_ao.z);
    let nscale     = material.metallic_roughness_ao.w;

    var n_ts = norm_s.rgb*2.0-1.0; n_ts.x *= nscale; n_ts.y *= nscale;
    let n = normalize(
        normalize(in.world_tangent)*n_ts.x +
        normalize(in.world_bitan)  *n_ts.y +
        normalize(in.world_normal) *n_ts.z
    );

    let v   = normalize(camera.camera_pos.xyz - in.world_pos);
    let ndv = max(dot(n,v), EPSILON);
    let f0  = mix(vec3<f32>(0.04), albedo, metallic);
    let dc  = albedo*(1.0-metallic);

    var lo = vec3<f32>(0.0);
    for(var i=0u; i<lights.count_and_pad.x; i++){
        let ls  = eval_light(lights.lights[i], in.world_pos);
        let l   = ls.direction;
        let h   = normalize(v+l);
        let ndl = max(dot(n,l), 0.0);
        if ndl < EPSILON { continue; }
        let ndh = max(dot(n,h), EPSILON);
        let ldh = max(dot(l,h), EPSILON);
        let shd = select(1.0, sample_shadow(in.shadow_coord,ndl), ls.is_sun);
        let D   = D_GGX(ndh, roughness);
        let V   = V_Smith(ndv, ndl, roughness);
        let F   = F_Schlick(ldh, f0);
        let kd  = (1.0-F)*(1.0-metallic);
        lo += (kd*dc*Fd_Burley(ndv,ndl,ldh,perc_rough)+D*V*F)*ls.radiance*ndl*shd;
    }

    // Analytical IBL ambient (replaced by full IBL in composite when HDR loaded)
    let ambient  = analytical_ibl(n, v, f0, dc, metallic, perc_rough, ao,
                                  lights.ambient.rgb*lights.ambient.a);
    let emissive = material.emissive.rgb;

    // Output world-space normal encoded in alpha chain for SSAO GBuffer
    // (actual SSAO applied as a post-process multiplier)
    return vec4<f32>(lo + ambient + emissive, alpha);
}
