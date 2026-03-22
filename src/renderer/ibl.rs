// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  IBL Renderer — Kulla-Conty edition                                         ║
// ║                                                                              ║
// ║  Zmiany vs poprzednia wersja:                                               ║
// ║  [KC-1] IRRAD_SIZE 32→64, PREFILTER_SIZE 128→256, PREFILTER_MIPS 5→9      ║
// ║  [KC-2] brdf_lut format Rgba16Float — kanał B = Ess(NdotV, roughness)     ║
// ║  [KC-3] Nowa tekstura eavg_lut (32×1 R16Float) — Eavg(roughness)          ║
// ║  [KC-4] IblMaps::eavg_lut + eavg_lut_view                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const ENV_SIZE:       u32 = 256;
const IRRAD_SIZE:     u32 = 64;    // [KC-1] było 32
const PREFILTER_SIZE: u32 = 256;   // [KC-1] było 128
const PREFILTER_MIPS: u32 = 9;     // [KC-1] było 5  (log2(256)+1 = 9)
const LUT_SIZE:       u32 = 512;
const EAVG_SIZE:      u32 = 32;    // [KC-3] Eavg 1D LUT — 32 próbki roughness

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BakeParams {
    face:      u32,
    roughness: f32,
    width:     u32,
    height:    u32,
}

pub struct IblMaps {
    pub env_cubemap:      wgpu::Texture,
    pub env_view:         wgpu::TextureView,
    pub irradiance:       wgpu::Texture,
    pub irradiance_view:  wgpu::TextureView,
    pub prefilter:        wgpu::Texture,
    pub prefilter_view:   wgpu::TextureView,
    pub brdf_lut:         wgpu::Texture,
    pub brdf_lut_view:    wgpu::TextureView,
    pub eavg_lut:         wgpu::Texture,         // [KC-3]
    pub eavg_lut_view:    wgpu::TextureView,     // [KC-3]
    pub sampler:          wgpu::Sampler,
    pub lut_sampler:      wgpu::Sampler,
}

fn cube_texture(device: &wgpu::Device, size: u32, mips: u32, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size:  wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        mip_level_count: mips,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rgba16Float,
        usage:           wgpu::TextureUsages::TEXTURE_BINDING
                       | wgpu::TextureUsages::STORAGE_BINDING
                       | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
    })
}

fn cube_view(tex: &wgpu::Texture, _mip: Option<u32>) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension:         Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6),
        ..Default::default()
    })
}

fn face_storage_view(tex: &wgpu::Texture, face: u32, mip: u32) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension:         Some(wgpu::TextureViewDimension::D2),
        base_array_layer:  face,
        array_layer_count: Some(1),
        base_mip_level:    mip,
        mip_level_count:   Some(1),
        ..Default::default()
    })
}

fn params_buf(device: &wgpu::Device, p: BakeParams) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("bake_params"),
        contents: bytemuck::bytes_of(&p),
        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn dispatch_face(
    device:        &wgpu::Device,
    queue:         &wgpu::Queue,
    pipeline:      &wgpu::ComputePipeline,
    bgl:           &wgpu::BindGroupLayout,
    extra_entries: &[wgpu::BindGroupEntry],
    face:          u32,
    roughness:     f32,
    width:         u32,
    height:        u32,
) {
    let p     = BakeParams { face, roughness, width, height };
    let p_buf = params_buf(device, p);

    let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
    ];
    entries.extend_from_slice(extra_entries);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake_bg"), layout: bgl, entries: &entries,
    });

    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("bake_face") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None, timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

pub fn bake_ibl(
    device:   &wgpu::Device,
    queue:    &wgpu::Queue,
    hdr_path: &str,
) -> Result<IblMaps> {
    // ── Load equirectangular HDR ──────────────────────────────────────────────
    log::info!("Loading HDR: {hdr_path}");
    let file    = std::fs::File::open(hdr_path)
        .map_err(|e| anyhow::anyhow!("Cannot open HDR file '{hdr_path}': {e}"))?;
    let reader  = std::io::BufReader::new(file);
    let decoder = image::codecs::hdr::HdrDecoder::new(reader)
        .map_err(|e| anyhow::anyhow!("HDR decode error: {e}"))?;
    let meta = decoder.metadata();
    let w    = meta.width;
    let h    = meta.height;

    use image::ImageDecoder;
    let color_type  = decoder.color_type();
    let total_bytes = decoder.total_bytes() as usize;
    let mut buf     = vec![0u8; total_bytes];
    decoder.read_image(&mut buf)
        .map_err(|e| anyhow::anyhow!("HDR read error: {e}"))?;

    let rgba: Vec<f32> = match color_type {
        image::ColorType::Rgb32F => {
            let floats: &[f32] = bytemuck::cast_slice(&buf);
            floats.chunks(3).flat_map(|p| [p[0], p[1], p[2], 1.0f32]).collect()
        }
        image::ColorType::Rgba32F => bytemuck::cast_slice::<u8, f32>(&buf).to_vec(),
        _ => {
            buf.chunks(3).flat_map(|p| {
                [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, 1.0f32]
            }).collect()
        }
    };

    let equirect_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("equirect"),
        size:  wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::Rgba16Float,
        usage:        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let rgba_f16: Vec<u16> = rgba.iter().map(|&f| half::f16::from_f32(f).to_bits()).collect();
    queue.write_texture(
        equirect_tex.as_image_copy(),
        bytemuck::cast_slice(&rgba_f16),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let equirect_view = equirect_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label:          Some("ibl_linear"),
        mag_filter:     wgpu::FilterMode::Linear,
        min_filter:     wgpu::FilterMode::Linear,
        mipmap_filter:  wgpu::MipmapFilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        lod_min_clamp:  0.0,
        lod_max_clamp:  9.0,   // [KC-1] było 4.0, teraz 9 mipów
        ..Default::default()
    });

    let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label:          Some("lut_sampler"),
        mag_filter:     wgpu::FilterMode::Linear,
        min_filter:     wgpu::FilterMode::Linear,
        mipmap_filter:  wgpu::MipmapFilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    // ── Create output textures ────────────────────────────────────────────────
    let env_tex = cube_texture(device, ENV_SIZE,       1,              "env_cubemap");
    let irr_tex = cube_texture(device, IRRAD_SIZE,     1,              "irradiance");    // [KC-1] 64³
    let pre_tex = cube_texture(device, PREFILTER_SIZE, PREFILTER_MIPS, "prefilter");    // [KC-1] 256³×9

    // [KC-2] brdf_lut — Rgba16Float (kanał R=A, G=B split-sum, B=Ess)
    let lut_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf_lut"),
        size:  wgpu::Extent3d { width: LUT_SIZE, height: LUT_SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::Rgba16Float,  // [KC-2] teraz 4 kanały
        usage:        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // [KC-3] Eavg 1D LUT: 32×1 R16Float — COPY_DST (bez STORAGE_BINDING, CPU upload)
    let eavg_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("eavg_lut"),
        size:  wgpu::Extent3d { width: EAVG_SIZE, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension:    wgpu::TextureDimension::D2,
        format:       wgpu::TextureFormat::R16Float,
        usage:        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // ── Shader ────────────────────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("ibl_bake"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl_bake.wgsl").into()),
    });

    // Shared BGL dla passów używających cube tekstury
    let cube_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cube_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube, multisampled: false },
                count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });

    // ══════════════════════════════════════════════════════════════════════════
    //  PASS A — Equirect → Cubemap
    // ══════════════════════════════════════════════════════════════════════════
    let equirect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("equirect_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });

    let equirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:        Some("equirect_pipeline"),
        layout:       Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&equirect_bgl], immediate_size: 0,
        })),
        module:       &shader,
        entry_point:  Some("cs_equirect_to_face"),
        compilation_options: Default::default(),
        cache:        None,
    });

    log::info!("Baking env cubemap ({ENV_SIZE}³)...");
    for face in 0..6u32 {
        let out_view = face_storage_view(&env_tex, face, 0);
        dispatch_face(device, queue, &equirect_pipeline, &equirect_bgl, &[
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&equirect_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&out_view) },
        ], face, 0.0, ENV_SIZE, ENV_SIZE);
    }

    let env_cube_view = env_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6),
        ..Default::default()
    });

    // ══════════════════════════════════════════════════════════════════════════
    //  PASS B — Irradiance  [KC-1] 64³
    // ══════════════════════════════════════════════════════════════════════════
    let irrad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:        Some("irrad_pipeline"),
        layout:       Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&cube_bgl], immediate_size: 0,
        })),
        module:       &shader,
        entry_point:  Some("cs_irradiance"),
        compilation_options: Default::default(),
        cache:        None,
    });

    log::info!("Baking irradiance ({IRRAD_SIZE}³)...");
    for face in 0..6u32 {
        let out_view = face_storage_view(&irr_tex, face, 0);
        dispatch_face(device, queue, &irrad_pipeline, &cube_bgl, &[
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&env_cube_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&out_view) },
        ], face, 0.0, IRRAD_SIZE, IRRAD_SIZE);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  PASS C — Specular prefilter  [KC-1] 256³×9
    // ══════════════════════════════════════════════════════════════════════════
    let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:        Some("prefilter_pipeline"),
        layout:       Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&cube_bgl], immediate_size: 0,
        })),
        module:       &shader,
        entry_point:  Some("cs_prefilter"),
        compilation_options: Default::default(),
        cache:        None,
    });

    log::info!("Baking specular prefilter ({PREFILTER_SIZE}³ × {PREFILTER_MIPS} mips)...");
    for mip in 0..PREFILTER_MIPS {
        let mip_size  = (PREFILTER_SIZE >> mip).max(1);
        let roughness = mip as f32 / (PREFILTER_MIPS - 1) as f32;
        for face in 0..6u32 {
            let out_view = face_storage_view(&pre_tex, face, mip);
            dispatch_face(device, queue, &pre_pipeline, &cube_bgl, &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&env_cube_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&out_view) },
            ], face, roughness, mip_size, mip_size);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  PASS D — BRDF LUT  [KC-2] Rgba16Float z kanałem Ess
    // ══════════════════════════════════════════════════════════════════════════
    let lut_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lut_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,   // [KC-2]
                    view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });

    let lut_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:        Some("lut_pipeline"),
        layout:       Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&lut_bgl], immediate_size: 0,
        })),
        module:       &shader,
        entry_point:  Some("cs_brdf_lut"),
        compilation_options: Default::default(),
        cache:        None,
    });

    log::info!("Baking BRDF LUT ({LUT_SIZE}×{LUT_SIZE}) + Ess channel...");
    let lut_params = BakeParams { face: 0, roughness: 0.0, width: LUT_SIZE, height: LUT_SIZE };
    let lut_pbuf   = params_buf(device, lut_params);
    let lut_view   = lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let lut_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lut_bg"), layout: &lut_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: lut_pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_view) },
        ],
    });
    {
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("lut") });
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&lut_pipeline);
            pass.set_bind_group(0, &lut_bg, &[]);
            pass.dispatch_workgroups((LUT_SIZE + 7) / 8, (LUT_SIZE + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  PASS E — Eavg 1D LUT  [KC-3] — generowany na CPU, upload przez COPY_DST
    //
    //  R16Float STORAGE_BINDING nie jest gwarantowane przez WebGPU / wgpu limits
    //  (Intel HD, Mali, starsze GPU nie obsługują). Generujemy Eavg na CPU:
    //
    //  Eavg(r) = 2 * integral_0^1 Ess(mu, r) * mu d_mu
    //
    //  Ess(ndv, r) liczymy próbkując tę samą formułę co cs_brdf_lut (G_v sum).
    //  32 pikseli roughness × 256 próbek wewnętrznych — ~8ms na CPU, jednorazowo.
    // ══════════════════════════════════════════════════════════════════════════
    log::info!("Baking Eavg LUT ({EAVG_SIZE}×1) on CPU for Kulla-Conty multiscatter...");

    // Odbicie Van der Corputa (Hammersley bit-reversal)
    fn radical_inverse(mut bits: u32) -> f32 {
        bits = (bits << 16) | (bits >> 16);
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
        bits as f32 * 2.328_306_4e-10
    }

    // GGX importance sample — zwraca H w przestrzeni stycznej (z=up)
    fn ggx_sample(xi: (f32, f32), roughness: f32) -> (f32, f32, f32) {
        let a     = roughness * roughness;
        let phi   = std::f32::consts::TAU * xi.0;
        let cos_t = ((1.0 - xi.1) / (1.0 + (a * a - 1.0) * xi.1)).sqrt();
        let sin_t = (1.0_f32 - cos_t * cos_t).max(0.0).sqrt();
        (phi.cos() * sin_t, phi.sin() * sin_t, cos_t)
    }

    // Smith G1 GGX (IBL wersja)
    fn g_schlick(ndv: f32, roughness: f32) -> f32 {
        let k = roughness * roughness / 2.0;
        ndv / (ndv * (1.0 - k) + k)
    }

    // Ess(ndv, roughness) — directional albedo przy F0=1 (white furnace integral)
    fn ess(ndv: f32, roughness: f32, samples: u32) -> f32 {
        let vx = (1.0 - ndv * ndv).max(0.0).sqrt();
        let vy = 0.0_f32;
        let vz = ndv;
        let mut sum = 0.0_f32;
        for i in 0..samples {
            let xi = (i as f32 / samples as f32, radical_inverse(i));
            let (hx, hy, hz) = ggx_sample(xi, roughness);
            // reflect V around H
            let vdoth = vx * hx + vy * hy + vz * hz;
            let lz    = 2.0 * vdoth * hz - vz;
            let nl    = lz.max(0.0);
            let nh    = hz.max(0.0);
            let vh    = vdoth.max(0.0);
            if nl > 0.0 && nh > 0.0 {
                let g   = g_schlick(ndv, roughness) * g_schlick(nl, roughness);
                let g_v = g * vh / (nh * ndv + 1e-5);
                sum    += g_v;   // F0=1 → Fc항 무시, G_v 직접 합산
            }
        }
        sum / samples as f32
    }

    // Eavg(roughness) = 2 * integral_0^1 Ess(mu, r) * mu d_mu  (32 outer, 256 inner)
    let mut eavg_data = Vec::with_capacity(EAVG_SIZE as usize);
    for slot in 0..EAVG_SIZE {
        let roughness = (slot as f32 + 0.5) / EAVG_SIZE as f32;
        let outer = 32u32;
        let inner = 256u32;
        let mut acc = 0.0_f32;
        for o in 0..outer {
            let ndv = (o as f32 + 0.5) / outer as f32;
            let e   = ess(ndv, roughness, inner);
            acc    += 2.0 * e * ndv;
        }
        eavg_data.push(acc / outer as f32);
    }

    // Konwertuj f32 → f16 i wgraj przez write_texture
    let eavg_f16: Vec<u16> = eavg_data.iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let eavg_view = eavg_tex.create_view(&wgpu::TextureViewDescriptor::default());
    queue.write_texture(
        eavg_tex.as_image_copy(),
        bytemuck::cast_slice(&eavg_f16),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(EAVG_SIZE * 2),   // 2 bajty na piksel R16Float
            rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: EAVG_SIZE, height: 1, depth_or_array_layers: 1 },
    );

    // ── Build final views ──────────────────────────────────────────────────────
    let env_view = env_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6), ..Default::default()
    });
    let irradiance_view = irr_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6), ..Default::default()
    });
    let prefilter_view = pre_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension:         Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6),
        mip_level_count:   Some(PREFILTER_MIPS),
        ..Default::default()
    });

    log::info!("IBL bake complete (Kulla-Conty ready).");

    Ok(IblMaps {
        env_cubemap: env_tex,   env_view,
        irradiance:  irr_tex,   irradiance_view,
        prefilter:   pre_tex,   prefilter_view,
        brdf_lut:    lut_tex,   brdf_lut_view: lut_view,
        eavg_lut:    eavg_tex,  eavg_lut_view: eavg_view,   // [KC-3]
        sampler:     linear_sampler,
        lut_sampler,
    })
}