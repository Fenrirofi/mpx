// ══════════════════════════════════════════════════════════════════════════════
//  PBR Shader — Final Balanced Version
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
@group(2) @binding(0) var<uniform> material:          MaterialUniform;
@group(2) @binding(1) var t_base_color:                texture_2d<f32>;
@group(2) @binding(2) var s_main:                      sampler;
@group(2) @binding(3) var t_normal:                    texture_2d<f32>;
@group(2) @binding(4) var t_metallic_roughness:        texture_2d<f32>;

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
@group(3) @binding(0) var<uniform> lights:      LightArrayUniform;
@group(3) @binding(1) var<uniform> shadow_data: ShadowUniform;
@group(3) @binding(2) var t_shadow_map:          texture_depth_2d;
@group(3) @binding(3) var s_shadow:              sampler_comparison;
@group(3) @binding(4) var t_irradiance:          texture_cube<f32>;
@group(3) @binding(5) var t_prefilter:           texture_cube<f32>;
@group(3) @binding(6) var t_brdf_lut:            texture_2d<f32>;
@group(3) @binding(7) var s_ibl:                 sampler;
@group(3) @binding(8) var s_lut:                 sampler;

const PI:       f32 = 3.14159265358979;
const INV_PI:   f32 = 0.31830988618;
const EPSILON:  f32 = 0.00001;
const IBL_SCALE: f32 = 0.1; // Skala dla Twojego HDR-a
const MAX_PREFILTER_LOD: f32 = 4.0;

// --- Vertex Shader ---
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
    @location(5)        shadow_coord: vec4<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    let w4           = object.model * vec4<f32>(in.position, 1.0);
    out.world_pos    = w4.xyz;
    out.clip_pos     = camera.view_proj * w4;
    out.shadow_coord = shadow_data.light_view_proj * w4;
    let n = normalize((object.normal_matrix * vec4<f32>(in.normal,      0.0)).xyz);
    let t = normalize((object.normal_matrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_normal = n;
    out.world_tan    = t;
    out.world_bitan  = cross(n, t) * in.tangent.w;
    out.uv           = in.uv;
    return out;
}

// --- PBR Helpers ---
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

// --- Post-Process ---
fn tone_map_aces(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3(0.0), vec3(1.0));
}

fn interleaved_gradient_noise(uv: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// --- IBL ---
fn ibl(n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>, albedo: vec3<f32>,
        metallic: f32, perc_rough: f32, ao: f32) -> vec3<f32> {
    let ndv = max(dot(n, v), EPSILON);
    let r   = reflect(-v, n);

    let irrad   = textureSample(t_irradiance, s_ibl, n).rgb * IBL_SCALE;
    let F       = F_SchlickR(ndv, f0, perc_rough);
    let kd      = (1.0 - F) * (1.0 - metallic);
    let diffuse = kd * albedo * irrad;

    let lod       = perc_rough * MAX_PREFILTER_LOD;
    let env_color = textureSampleLevel(t_prefilter, s_ibl, r, lod).rgb * IBL_SCALE;
    let brdf_uv   = vec2<f32>(ndv, perc_rough);
    let brdf      = textureSample(t_brdf_lut, s_lut, brdf_uv).rg;
    let specular  = env_color * (f0 * brdf.x + brdf.y);

    return (diffuse + specular) * ao;
}

// --- Fragment Shader ---
@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let albedo_s   = textureSample(t_base_color,           s_main, in.uv);
    let mr_s       = textureSample(t_metallic_roughness, s_main, in.uv);
    let norm_s     = textureSample(t_normal,             s_main, in.uv);

    let albedo     = material.base_color.rgb * albedo_s.rgb;
    let alpha      = material.base_color.a   * albedo_s.a;
    let metallic   = clamp(material.metallic_roughness_ao.x * mr_s.b, 0.0, 1.0);
    let perc_rough = clamp(material.metallic_roughness_ao.y * mr_s.g, 0.045, 1.0);
    let roughness  = perc_rough * perc_rough;
    let ao         = mix(1.0, mr_s.r, material.metallic_roughness_ao.z);
    
    let n_ts = norm_s.rgb*2.0-1.0;
    let n = normalize(
        normalize(in.world_tan)    * n_ts.x * material.metallic_roughness_ao.w +
        normalize(in.world_bitan)  * n_ts.y * material.metallic_roughness_ao.w +
        normalize(in.world_normal) * n_ts.z
    );

    let v   = normalize(camera.camera_pos.xyz - in.world_pos);
    let ndv = max(dot(n, v), EPSILON);
    let f0  = mix(vec3<f32>(0.04), albedo, metallic);
    let dc  = albedo * (1.0 - metallic);

    var lo = vec3<f32>(0.0);
    for(var i=0u; i<lights.count_and_pad.x; i++){
        let l_data = lights.lights[i];
        var l_dir: vec3<f32>;
        var radiance: vec3<f32>;
        var is_sun = false;

        if u32(l_data.params.x) == 0u {
            l_dir = normalize(-l_data.position_or_dir.xyz);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a;
            is_sun = true;
        } else {
            let diff = l_data.position_or_dir.xyz - in.world_pos;
            let dist = length(diff);
            l_dir = diff / max(dist, EPSILON);
            let atten = 1.0 / max(dist*dist, 0.01);
            radiance = l_data.color_intensity.rgb * l_data.color_intensity.a * atten;
        }

        let h   = normalize(v + l_dir);
        let ndl = max(dot(n, l_dir), 0.0);
        if ndl <= 0.0 { continue; }
        
        let ndh = max(dot(n, h), EPSILON);
        let ldh = max(dot(l_dir, h), EPSILON);
        
        var shd = 1.0;
        if is_sun {
             let p = in.shadow_coord.xyz/in.shadow_coord.w;
             let uv_sh = vec2<f32>(p.x*0.5+0.5, p.y*-0.5+0.5);
             let bias = max(shadow_data.shadow_params.y*(1.0-ndl), shadow_data.shadow_params.x);
             shd = textureSampleCompare(t_shadow_map, s_shadow, uv_sh, p.z-bias);
        }
        
        let D   = D_GGX(ndh, roughness);
        let Vis = V_Smith(ndv, ndl, roughness);
        let F   = F_Schlick(ldh, f0);
        let kd  = (1.0-F)*(1.0-metallic);
        lo += (kd*dc*Fd_Burley(ndv,ndl,ldh,perc_rough) + D*Vis*F) * radiance * ndl * shd;
    }

    let ambient  = ibl(n, v, f0, dc, metallic, perc_rough, ao);
    var color = lo + ambient + material.emissive.rgb;

    // --- FINAL COLOR (Balanced) ---
    // Tone mapping jest obowiązkowy dla HDR
    color = tone_map_aces(color);

    // Dithering usuwa schodkowanie
    let noise = interleaved_gradient_noise(in.clip_pos.xy);
    color += (noise - 0.5) / 255.0;

    // UWAGA: Nie robimy pow(color, 1.0/2.2) bo wgpu/sRGB zrobi to za nas.
    return vec4<f32>(color, alpha);
}