//! Proceduralne mapy PBR do podglądu bez zewnętrznych plików (mikro-szczegół, AO, height).

pub const DEMO_MAP_SIZE: u32 = 512;

fn hash2(x: f32, y: f32) -> f32 {
    let p = x * 127.1 + y * 311.7;
    (p.sin() * 43758.5453).fract()
}

fn noise(u: f32, v: f32) -> f32 {
    let i = u.floor();
    let j = v.floor();
    let f = u - i;
    let g = v - j;
    let a = hash2(i, j);
    let b = hash2(i + 1.0, j);
    let c = hash2(i, j + 1.0);
    let d = hash2(i + 1.0, j + 1.0);
    let uu = f * f * (3.0 - 2.0 * f);
    let vv = g * g * (3.0 - 2.0 * g);
    a * (1.0 - uu) * (1.0 - vv) + b * uu * (1.0 - vv) + c * (1.0 - uu) * vv + d * uu * vv
}

fn height_fn(u: f32, v: f32) -> f32 {
    let su = u * std::f32::consts::TAU * 2.0;
    let sv = v * std::f32::consts::TAU * 2.0;
    let macro_h = 0.11 * (su.sin() * sv.cos() * 2.0
        + (su * 2.07 + 0.9).sin() * (sv * 1.93).cos() * 0.5);
    let fine = noise(u * 16.0, v * 16.0) * 0.085 + noise(u * 31.0, v * 29.0) * 0.045;
    (0.5 + macro_h + fine).clamp(0.0, 1.0)
}

/// Zwraca: (normal RGBA linear, ORM RGBA linear, height RGBA linear — R = height 0..1).
pub fn generate_demo_detail_maps() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let s = DEMO_MAP_SIZE as usize;
    let mut hmap = vec![0f32; s * s];
    for j in 0..s {
        for i in 0..s {
            let u = i as f32 / s as f32;
            let v = j as f32 / s as f32;
            hmap[j * s + i] = height_fn(u, v);
        }
    }

    let mut nrm = vec![0u8; s * s * 4];
    let mut mr = vec![0u8; s * s * 4];
    let mut htex = vec![0u8; s * s * 4];

    let du = 1.0 / DEMO_MAP_SIZE as f32;
    let dv = 1.0 / DEMO_MAP_SIZE as f32;
    let bump = 5.0f32;

    for j in 0..s {
        for i in 0..s {
            let idx = j * s + i;
            let u = i as f32 / s as f32;
            let v = j as f32 / s as f32;
            let ip = (i + 1).min(s - 1);
            let im = i.saturating_sub(1);
            let jp = (j + 1).min(s - 1);
            let jm = j.saturating_sub(1);

            let hx = (hmap[j * s + ip] - hmap[j * s + im]) / (2.0 * du);
            let hy = (hmap[jp * s + i] - hmap[jm * s + i]) / (2.0 * dv);
            let mut nx = -hx * bump;
            let mut ny = -hy * bump;
            let nz = 1.0f32;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(0.0001);
            nx /= len;
            ny /= len;
            let nz = nz / len;

            nrm[idx * 4] = ((nx * 0.5 + 0.5) * 255.0) as u8;
            nrm[idx * 4 + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            nrm[idx * 4 + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
            nrm[idx * 4 + 3] = 255;

            let h = hmap[idx];
            let ao = (0.32 + 0.68 * h).clamp(0.0, 1.0);
            let rough_var = (0.82 + 0.16 * hash2(u * 113.0, v * 97.0)).clamp(0.0, 1.0);
            mr[idx * 4] = (ao * 255.0) as u8;
            mr[idx * 4 + 1] = (rough_var * 255.0) as u8;
            mr[idx * 4 + 2] = 255;
            mr[idx * 4 + 3] = 255;

            let hb = (h * 255.0) as u8;
            htex[idx * 4] = hb;
            htex[idx * 4 + 1] = hb;
            htex[idx * 4 + 2] = hb;
            htex[idx * 4 + 3] = 255;
        }
    }

    (nrm, mr, htex)
}
