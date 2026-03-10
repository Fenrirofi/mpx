# PBR Engine

A modern, physically-based rendering engine written in Rust using **wgpu 28** and **winit 0.30**.

## Features

| Category | Implementation |
|---|---|
| **Shading** | Cook-Torrance BRDF — GGX NDF, Smith-GGX visibility, Fresnel-Schlick |
| **Lights** | Directional, Point (inverse-square + range falloff), Spot (cone) — up to 16 |
| **Normal mapping** | TBN matrix, mikktspace-compatible tangent space |
| **Materials** | Metallic-roughness PBR, emissive, ambient occlusion |
| **Camera** | Perspective, FPS-style orbit controller (WASD + mouse) |
| **Culling** | Frustum planes (Gribb-Hartmann), AABB & sphere tests |
| **Tone mapping** | Reinhard + gamma correction |
| **Architecture** | Clean ECS-style separation: Scene / Renderer / Assets / Math |

---

## Building

```bash
# Requires Rust 1.80+
cargo build --release
cargo run --release
```

### Dependencies

```toml
wgpu    = "28.0.0"
winit   = "0.30.13"
glam    = "0.29"       # SIMD math
bytemuck = "1.16"      # Safe transmute for GPU uploads
slotmap  = "1.0"       # Generational indices for asset handles
```

---

## Architecture

```
src/
├── main.rs              ← winit ApplicationHandler, event loop
│
├── math/
│   ├── transform.rs     ← TRS transform with dirty-flag caching
│   └── frustum.rs       ← Six-plane frustum culling (Gribb-Hartmann)
│
├── assets/
│   ├── mesh.rs          ← Vertex layout, CPU mesh, UV sphere + cube builders
│   └── material.rs      ← PBR material parameters (CPU side)
│
├── scene/
│   ├── camera.rs        ← Perspective camera + FPS controller
│   ├── light.rs         ← Light types (directional / point / spot)
│   ├── object.rs        ← SceneObject = mesh + material + transform
│   └── mod.rs           ← Scene container + update loop
│
├── renderer/
│   ├── context.rs       ← RenderContext: wgpu init, resize, frame render
│   ├── pipeline.rs      ← PBR pipeline + all BindGroupLayout descriptors
│   ├── uniforms.rs      ← GPU uniform structs (Pod + Zeroable)
│   ├── gpu_mesh.rs      ← Vertex/index buffer upload
│   └── texture.rs       ← Depth texture + fallback texture helpers
│
├── utils/
│   └── mod.rs           ← align_up, align_uniform utilities
│
└── shaders/
    └── pbr.wgsl         ← Full PBR WGSL shader
```

---

## Controls

| Input | Action |
|---|---|
| `W / A / S / D` | Move forward / left / back / right |
| `E / Space` | Move up |
| `Q` | Move down |
| Mouse movement | Look around |

---

## Extending the Engine

### Add a new mesh type
Implement a function in `src/assets/mesh.rs` returning a `Mesh` struct with `vertices`, `indices`, and `bounding_sphere`.

### Add a new light type
1. Add a variant to `LightKind` in `src/scene/light.rs`
2. Handle it in `GpuLight::from_light` in `src/renderer/uniforms.rs`
3. Add evaluation logic in the WGSL `evaluate_light` function

### Add texture loading
Use the `image` crate to decode image bytes, then call `queue.write_texture(...)`. Store the `TextureView` alongside each material's GPU resources in `context.rs`.

### Add instancing
- Replace per-object `ObjectUniform` with a `STORAGE` buffer of transforms
- Change `draw_indexed` to `draw_indexed_indirect` with an instance count
- Use `@builtin(instance_index)` in the vertex shader

### Add shadows
- Render a depth-only pass from the light's perspective
- Sample the shadow map in `fs_main` with a PCF filter
- Add a `ShadowUniform` bind group with the light's view-projection matrix

---

## PBR Shader Overview

The fragment shader implements the **Cook-Torrance reflectance model**:

```
Lo(ω_o) = Σ (kd · (albedo/π) + ks · DGF/(4·NdotV·NdotL)) · Li · NdotL
```

Where:
- **D** — GGX Normal Distribution Function
- **G** — Smith-GGX Height-Correlated Visibility
- **F** — Fresnel-Schlick Approximation
- **kd** — Diffuse factor (0 for metals, scaled by (1-F) for dielectrics)
- **ks = F** — Specular factor

Output is Reinhard tone-mapped and gamma-corrected to sRGB.
