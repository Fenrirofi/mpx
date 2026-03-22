// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  post.wgsl — POPRAWIONA WERSJA                                              ║
// ║                                                                             ║
// ║  [4] DITHERING — triangular PDF zamiast grain                               ║
// ║                                                                             ║
// ║  Poprzedni grain: (noise - 0.5) * strength * 0.04                          ║
// ║  Problem: to jest uniform dither — dodaje szum o amplitudzie strength*0.04 ║
// ║  ale nie eliminuje color banding bo rozkład jest płaski, nie trójkątny.     ║
// ║                                                                             ║
// ║  Rozwiązanie: triangular PDF dither (Roberts 2016, Wolfe 2017):             ║
// ║    noise1 = IGN(pixel, t)      — jeden hash                                 ║
// ║    noise2 = IGN(pixel + 23.0)  — drugi hash (inny offset)                  ║
// ║    tpdf   = noise1 - noise2    ∈ (-1, 1), rozkład trójkątny               ║
// ║    output += tpdf * (1/255)    — amplituda = 1 LSB w 8-bit                 ║
// ║                                                                             ║
// ║  Rozkład trójkątny jest optymalny dla dithera kwantyzacyjnego:              ║
// ║  eliminuje pattern quantization bez widocznego wzoru szumu.                 ║
// ║  Amplituda 1/255 ≈ 0.00392 to dokładnie 1 krok kwantyzacji 8-bit.         ║
// ║  Zbyt mała żeby być widoczna, wystarczająca żeby złamać banding.           ║
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
    let color  = textureSample(t_hdr, s_nearest, in.uv).rgb;
    let bright = quadratic_threshold(color, bloom.threshold, bloom.knee);
    return vec4<f32>(bright, 1.0);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pass 2 – Bloom blur (separable 7-tap Gaussian, run twice: H + V)
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var t_bloom_src: texture_2d<f32>;
@group(0) @binding(1) var s_linear:    sampler;

struct BlurParams {
    texel_size: vec2<f32>,
    horizontal: u32,
    _pad:       u32,
}
@group(0) @binding(2) var<uniform> blur: BlurParams;

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
        result += textureSample(t_bloom_src, s_linear, in.uv + dir * f32(i - 3)).rgb * WEIGHTS[i];
    }
    return vec4<f32>(result, 1.0);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pass 3 – Composite: HDR + bloom + tone map + CA + vignette + dither
// ══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0) var t_hdr_in:   texture_2d<f32>;
@group(0) @binding(1) var t_bloom_in: texture_2d<f32>;
@group(0) @binding(2) var s_comp:     sampler;

struct PostParams {
    bloom_strength:      f32,
    exposure:            f32,
    ca_strength:         f32,
    vignette_strength:   f32,
    vignette_radius:     f32,
    vignette_smoothness: f32,
    grain_strength:      f32,
    time:                f32,
    ssao_strength:       f32,
    contrast:            f32,
    saturation:          f32,
    lift:                f32,
    _pad_post:           f32,
}
@group(0) @binding(3) var<uniform> post: PostParams;
@group(0) @binding(4) var t_ssao:        texture_2d<f32>;

// ── ACES fitted (Narkowicz 2015, Hill) ────────────────────────────────────────
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Chromatic aberration (radial barrel) ──────────────────────────────────────
fn chromatic_aberration(uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let center = uv - 0.5;
    let dist   = length(center);
    let dir    = normalize(center + vec2<f32>(0.0001));
    let r = textureSample(t_hdr_in, s_comp, uv + dir * strength * dist * 1.0).r;
    let g = textureSample(t_hdr_in, s_comp, uv                              ).g;
    let b = textureSample(t_hdr_in, s_comp, uv - dir * strength * dist * 0.7).b;
    return vec3<f32>(r, g, b);
}

// ── Vignette ──────────────────────────────────────────────────────────────────
fn vignette(uv: vec2<f32>, radius: f32, smoothness: f32) -> f32 {
    return 1.0 - smoothstep(radius - smoothness, radius + smoothness,
                             length(uv - 0.5) * 2.0);
}

// ── Interleaved Gradient Noise (Jimenez 2014) ─────────────────────────────────
// Szybka, dobrze rozmieszczona funkcja hash dla dithera.
// Używamy dwóch oddzielnych hashy z różnymi offsetami do konstruowania TPDF.
fn ign(pixel: vec2<f32>, seed: f32) -> f32 {
    let p = pixel + seed;
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ── [4] Triangular PDF Dither ─────────────────────────────────────────────────
//
// Cel: złamanie color banding w ciemnych obszarach przy kwantyzacji do 8-bit.
//
// Metoda: triangular probability density function (TPDF)
//   Dwie niezależne próbki uniform[0,1] → ich różnica ma rozkład trójkątny [-1,1]
//   Amplituda = 1/255 (jeden krok 8-bit) → kwantyzacja bez widocznego szumu
//
// Dlaczego TPDF a nie uniform?
//   Uniform dither ma "DC offset" — średnia wartość szumu może być ≠ 0,
//   co powoduje lekkie jaśnienie/ciemnienie w jednolitych gradientach.
//   TPDF ma średnią = 0 i wariancję 1/6 (zamiast 1/12 dla uniform).
//   Wyższa wariancja = lepsze "złamanie" progów kwantyzacji, zerowa DC = brak DC bias.
//
// Implementacja:
//   n1 = IGN(pixel, t)       — hash oparty na czasie (animowany)
//   n2 = IGN(pixel, t + 23.) — drugi hash z innym seedem (niezależny)
//   tpdf = n1 - n2           ∈ (-1, 1), rozkład trójkątny
//   dither = tpdf / 255.0    — amplituda = 1 LSB w 8-bit
//
// Aplikacja: po tone mappingu i gamma, PRZED zwrotem — dodajemy do LDR [0,1].
// Nie wpływa na HDR values > 1.0, tylko na finalną kwantyzację.
// ─────────────────────────────────────────────────────────────────────────────
fn tpdf_dither(pixel: vec2<f32>, t: f32) -> vec3<f32> {
    // Dwie niezależne próbki IGN z różnymi seedami
    let n1 = ign(pixel, t);
    let n2 = ign(pixel, t + 23.0); // 23.0 = arytmetycznie niezależny seed
    // Różnica = TPDF ∈ (-1, 1)
    let tpdf = n1 - n2;
    // Amplituda = 1 krok 8-bit (0.00392...)
    // Stosujemy do wszystkich kanałów RGB równo (chroma-neutral dither)
    return vec3<f32>(tpdf) * (1.0 / 255.0);
}

// ─────────────────────────────────────────────────────────────────────────────

@fragment
fn fs_composite(in: PostVOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_hdr_in));
    let pixel    = in.uv * tex_size;

    // HDR + chromatic aberration
    var hdr = chromatic_aberration(in.uv, post.ca_strength * 0.005);

    // SSAO — ściemnia ambient, nie bezpośrednie światła
    let ao   = textureSample(t_ssao, s_comp, in.uv).r;
    hdr     *= clamp(mix(1.0, ao, post.ssao_strength), 0.0, 1.0);

    // Bloom
    hdr += textureSample(t_bloom_in, s_comp, in.uv).rgb * post.bloom_strength;

    // Exposure
    hdr *= pow(2.0, post.exposure);

    // ACES tone mapping (HDR → LDR)
    var ldr = aces_filmic(hdr);

    // Grading: kontrast wokół 0.5, saturacja, lift
    ldr = (ldr - vec3<f32>(0.5)) * post.contrast + vec3<f32>(0.5);
    let luma = dot(ldr, vec3<f32>(0.2126, 0.7152, 0.0722));
    ldr = mix(vec3<f32>(luma), ldr, post.saturation);
    ldr = ldr + vec3<f32>(post.lift);

    // Vignette
    ldr *= mix(1.0 - post.vignette_strength, 1.0,
               vignette(in.uv, post.vignette_radius, post.vignette_smoothness));

    // Gamma correction (linear → sRGB display)
    ldr = pow(max(ldr, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    // [4] TPDF Dither — po gamma, przed kwantyzacją do 8-bit
    //
    // Kluczowe: dither stosujemy w przestrzeni gamma (sRGB), NIE liniowej.
    // W przestrzeni liniowej ciemne wartości są bardzo małe i szum byłby
    // zbyt duży (gamma ≈ 2.2 amplifikuje różnice w cieniach).
    // W przestrzeni gamma 1 LSB = 1/255 jest równomiernie rozmieszczony
    // przez cały zakres tonalny → optymalny dla eliminacji bandingu.
    ldr += tpdf_dither(pixel, post.time);

    // Clamp do [0,1] po ditherze (TPDF może wyjść poza zakres przy ±1/255)
    ldr = clamp(ldr, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(ldr, 1.0);
}