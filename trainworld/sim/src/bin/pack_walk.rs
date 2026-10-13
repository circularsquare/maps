//! T-275: the walk's detour factor between nearby zones, for `pipeline/routes.py`.
//!
//!   cargo run --release --features demand --bin pack_walk -- PACK.json OUT.u8 [--window 2]
//!       [--probe lon,lat,lon,lat ...]   (only print the walk between pairs of places, a check)
//!
//! PACK is a finished pack that already has the walking barrier raster (`walk_row`, `walk_x`) and
//! the `routes` block. For each zone, the walk is searched from its representative cell (the
//! cell nearest its weighted centre) with the game's own search (`demand::routes`), and the
//! factor to each zone of its window is the cells' walking metres over their straight metres,
//! weighted by their commuters; the two directions are averaged. Written in the canonical window
//! order (docs/demand.md 4.3), one byte each, `round(64 ln f)`.

use trainworld_sim::demand::{clock::now_ms, kernel, pack, routes};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = vec![];
    let mut win = 2i32;
    let mut probes: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--probe" => {
                probes.push(args[i + 1].clone());
                i += 1;
            }
            "--window" => {
                win = args[i + 1].parse().unwrap();
                i += 1;
            }
            a if a.starts_with("--") => panic!("unknown argument {a}"),
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let [inp, out] = <[String; 2]>::try_from(pos).expect("usage: pack_walk PACK.json OUT.u8 [--window W]");
    let (header, bin) = pack::load_raw(&inp).unwrap_or_else(|e| panic!("{e}"));
    let city = pack::parse(&header, &bin).unwrap_or_else(|e| panic!("{e}"));
    let pg = city.gravity.as_ref().expect("the pack has no zones");
    let z = kernel::zones_from_pack(&city, pg);
    let r = routes::Routes::from_pack(&header, &bin, &z).unwrap_or_else(|e| panic!("{e}")).expect("the pack has no routes block");
    let grid = r.walk.as_ref().expect("the pack has no walking barrier raster");
    if !probes.is_empty() {
        // `--probe lon,lat,lon,lat`: the walk between two places, as the game finds a station's
        let v: serde_json::Value = serde_json::from_str(&header).unwrap();
        let (lon0, lat0) = (v["origin"]["lon"].as_f64().unwrap(), v["origin"]["lat"].as_f64().unwrap());
        let k = lat0.to_radians().cos();
        let xy = |lon: f64, lat: f64| ((6371008.8 * (lon - lon0).to_radians() * k) as f32, (6371008.8 * (lat - lat0).to_radians()) as f32);
        for p in &probes {
            let c: Vec<f64> = p.split(',').map(|s| s.parse().unwrap()).collect();
            let ((ax, ay), (bx, by)) = (xy(c[0], c[1]), xy(c[2], c[3]));
            let s = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
            let walk = grid.field(ax, ay, 4000.0).map_or(Some(s), |f| f.eff(bx, by, s));
            println!("probe {p}: straight {s:.0} m, walk {}", walk.map_or("cut off within 4 km".into(), |w| format!("{w:.0} m")));
        }
        return;
    }
    let t0 = now_ms();
    let n = z.n;
    let nreg = z.region.iter().copied().max().map_or(1, |v| v as usize + 1);
    // cells by zone, weights, and each zone's representative cell
    let w: Vec<f64> = (0..city.len()).map(|c| city.pop[c] as f64 + city.jobs[c] as f64 + 1e-3).collect();
    let mut members: Vec<Vec<u32>> = vec![vec![]; n];
    let (mut sx, mut sy, mut sw) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for c in 0..city.len() {
        let zi = z.cell_zone[c] as usize;
        members[zi].push(c as u32);
        sx[zi] += w[c] * city.x[c] as f64;
        sy[zi] += w[c] * city.y[c] as f64;
        sw[zi] += w[c];
    }
    let rep: Vec<usize> = (0..n)
        .map(|zi| {
            let (mx, my) = (sx[zi] / sw[zi], sy[zi] / sw[zi]);
            // the cell nearest the weighted centre, off the barrier if any cell is
            let mut best = (true, f64::INFINITY, 0usize);
            for &c in &members[zi] {
                let wet = grid.is_barrier(city.x[c as usize], city.y[c as usize]);
                let d = ((city.x[c as usize] as f64 - mx).powi(2) + (city.y[c as usize] as f64 - my).powi(2)).sqrt();
                if (wet, d) < (best.0, best.1) {
                    best = (wet, d, c as usize);
                }
            }
            best.2
        })
        .collect();
    let mut grid_zone = vec![u32::MAX; (z.w * z.h) as usize * nreg];
    for zi in 0..n {
        grid_zone[((z.gx[zi] * z.h + z.gy[zi]) as usize) * nreg + z.region[zi] as usize] = zi as u32;
    }
    let window = |zi: usize| {
        let mut v = vec![];
        for dx in -win..=win {
            let gx = z.gx[zi] + dx;
            if gx < 0 || gx >= z.w {
                continue;
            }
            for dy in -win..=win {
                let gy = z.gy[zi] + dy;
                if gy < 0 || gy >= z.h {
                    continue;
                }
                for reg in 0..nreg {
                    let j = grid_zone[((gx * z.h + gy) as usize) * nreg + reg];
                    if j != u32::MAX {
                        v.push(j as usize);
                    }
                }
            }
        }
        v
    };
    // the search reaches the window's farthest cells: (W + 1) squares on each axis
    let reach = z.zone_m * (win as f32 + 1.0) * std::f32::consts::SQRT_2;
    let fmax = (255.0 / routes::Q_SCALE).exp();
    let mut f: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut searched = 0usize;
    for zi in 0..n {
        let js = window(zi);
        let (rx, ry) = (city.x[rep[zi]], city.y[rep[zi]]);
        let field = grid.field(rx, ry, reach);
        if std::env::var("PACK_WALK_DEBUG").ok().and_then(|v| v.parse::<usize>().ok()) == Some(zi) {
            println!("zone {zi}: rep cell {} at ({rx:.0}, {ry:.0}), on a barrier {}, field {}", rep[zi], grid.is_barrier(rx, ry), field.is_some());
            if let Some(fl) = &field {
                for &c in members[zi].iter().chain(js.iter().flat_map(|&j| members[j].iter().take(2))).take(30) {
                    let c = c as usize;
                    let s = ((city.x[c] - rx).powi(2) + (city.y[c] - ry).powi(2)).sqrt();
                    println!("  cell {c} ({:.0}, {:.0}) barrier {} straight {s:.0} walk {:?}", city.x[c], city.y[c], grid.is_barrier(city.x[c], city.y[c]), fl.eff(city.x[c], city.y[c], s));
                }
            }
        }
        let mut row = vec![1.0f32; js.len()];
        if let Some(fl) = &field {
            searched += 1;
            for (k, &j) in js.iter().enumerate() {
                let (mut num, mut den) = (0f64, 0f64);
                for &c in &members[j] {
                    let c = c as usize;
                    let s = ((city.x[c] - rx).powi(2) + (city.y[c] - ry).powi(2)).sqrt();
                    if s < 1.0 {
                        continue;
                    }
                    let e = fl.eff(city.x[c], city.y[c], s).unwrap_or(s * fmax);
                    num += w[c] * e as f64;
                    den += w[c] * s as f64;
                }
                if den > 0.0 {
                    row[k] = (num / den) as f32;
                }
            }
        }
        if std::env::var("PACK_WALK_DEBUG").ok().and_then(|v| v.parse::<usize>().ok()) == Some(zi) {
            println!("  window {:?}", js.iter().zip(&row).map(|(j, v)| (*j, (v * 100.0).round() / 100.0)).collect::<Vec<_>>());
        }
        f.push(row);
    }
    // both directions averaged
    let wins: Vec<Vec<usize>> = (0..n).map(window).collect();
    let mut bytes = Vec::new();
    let (mut over, mut cut, mut total) = (0usize, 0usize, 0usize);
    for zi in 0..n {
        for (k, &j) in wins[zi].iter().enumerate() {
            let back = wins[j].iter().position(|&x| x == zi).map_or(f[zi][k], |b| f[j][b]);
            let v = 0.5 * (f[zi][k] + back);
            total += 1;
            over += (v > 1.2) as usize;
            cut += (v > 5.0) as usize;
            bytes.push((routes::Q_SCALE * v.max(1.0).ln()).round().clamp(0.0, 255.0) as u8);
        }
    }
    std::fs::write(&out, &bytes).unwrap();
    println!(
        "walk factors: {n} zones ({searched} with water within reach), {total} window pairs, {over} with a detour over 1.2, {cut} over 5 (cut off); {:.0} ms",
        now_ms() - t0
    );
}
