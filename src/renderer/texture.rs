// ─────────────────────────────────────────────────────────────────────────────
//  renderer/texture.rs  —  POPRAWIONA WERSJA
//
//  ZMIANA #1 (Bug #1):
//    Dodano parametr `is_srgb: bool` do create_fallback_texture.
//
//    Przyczyna błędu: poprzednia wersja tworzyła WSZYSTKIE fallback tekstury
//    jako Rgba8UnormSrgb. GPU przy próbkowaniu automatycznie dekoduje sRGB→linear,
//    więc wartość [128,128,255] (flat normal) zamiast 0.502 stawała się ~0.216.
//    To powodowało że "płaskie" normalne nie były płaskie — TBN matrix był
//    pochylony, oświetlenie wyglądało jak gdyby powierzchnia była wklęsła/wypukła
//    nawet bez mapy normalnych.
//
//    Reguła:
//      is_srgb = true  → Rgba8UnormSrgb   — dla albedo/base_color (dane artysty)
//      is_srgb = false → Rgba8Unorm       — dla normalnych, metallic, roughness,
//                                            AO (dane liniowe / dane wektorowe)
// ─────────────────────────────────────────────────────────────────────────────

/// Tworzy 1×1 teksturę fallback z podanym kolorem RGBA.
///
/// # Parametry
/// - `rgba`    — wartości [0..255] w sRGB (albedo) lub liniowe (normal/MR/AO)
/// - `is_srgb` — czy GPU ma dekodować sRGB przy próbkowaniu
///               true  → Rgba8UnormSrgb  (albedo, emissive)
///               false → Rgba8Unorm      (normal, metallic_roughness, ao)
pub fn create_fallback_texture(
    device:  &wgpu::Device,
    queue:   &wgpu::Queue,
    rgba:    [u8; 4],
    label:   &str,
    is_srgb: bool,                         // ← NOWY parametr
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {

    // ZMIANA: wybieramy format w zależności od flagi
    let format = if is_srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some(label),
        size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format,                            // ← używamy zmiennej zamiast hardcoded Rgba8UnormSrgb
        usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
    });

    queue.write_texture(
        texture.as_image_copy(),
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label:            Some("default_sampler"),
        address_mode_u:   wgpu::AddressMode::Repeat,
        address_mode_v:   wgpu::AddressMode::Repeat,
        address_mode_w:   wgpu::AddressMode::Repeat,
        mag_filter:       wgpu::FilterMode::Linear,
        min_filter:       wgpu::FilterMode::Linear,
        mipmap_filter:    wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 16,
        ..Default::default()
    });

    (texture, view, sampler)
}

// ─────────────────────────────────────────────────────────────────────────────
//  WYMAGANE ZMIANY W context.rs:
//
//  Znajdź linie gdzie tworzysz fallback tekstury i dodaj parametr is_srgb:
//
//  PRZED:
//    let (_, white_view, default_sampler) =
//        texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white");
//    let (_, flat_normal_view, _) =
//        texture::create_fallback_texture(&device, &queue, [128,128,255,255], "flat_normal");
//
//  PO:
//    let (_, white_view, default_sampler) =
//        texture::create_fallback_texture(&device, &queue, [255,255,255,255], "white", true);
//    let (_, flat_normal_view, _) =
//        texture::create_fallback_texture(&device, &queue, [128,128,255,255], "flat_normal", false);
//
//  Jeśli w przyszłości dodasz fallback dla metallic_roughness:
//    let (_, mr_fallback_view, _) =
//        texture::create_fallback_texture(&device, &queue, [0, 128, 0, 255], "mr_fallback", false);
//    //  R=0 (AO=0), G=128 (roughness=0.5 liniowe), B=0 (metallic=0) — liniowe!
// ─────────────────────────────────────────────────────────────────────────────

pub fn create_depth_texture(
    device: &wgpu::Device,
    width:  u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("depth_texture"),
        size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}