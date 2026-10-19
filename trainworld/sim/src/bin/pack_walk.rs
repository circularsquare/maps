//! T-275: the walk's detour factor between nearby zones, for `pipeline/routes.py`.
//!
//!   cargo run --release --features demand --bin pack_walk -- PACK.json OUT.u8 [--window 2]
//!       [--parts 4] [--probe lon,lat,lon,lat ...]   (--probe: only print the walk between pairs
//!       of places, a check)
//!
//! PACK is a finished pack that already has the walking barrier raster (`walk_row`, `walk_x`) and
//! the `routes` block. Each zone's cells split by bank into at most `--parts` parts (T-700), and
//! the factor between parts is searched with the game's own walk search (`routes::walk_tables`,
//! weighted by commuters). OUT gets the factors in the canonical window order (docs/demand.md 4.3),
//! one byte each, `round(64 ln f)`; OUT.part each cell's part, one byte.

use trainworld_sim::demand::{clock::now_ms, kernel, pack, routes};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = vec![];
    let mut win = 2i32;
    let mut parts = routes::WALK_PARTS;
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
            "--parts" => {
                parts = args[i + 1].parse().unwrap();
                i += 1;
            }
            a if a.starts_with("--") => panic!("unknown argument {a}"),
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let [inp, out] = <[String; 2]>::try_from(pos).expect("usage: pack_walk PACK.json OUT.u8 [--window W] [--parts P]");
    assert!((1..=255).contains(&parts), "--parts is 1 to 255");
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
    let w: Vec<f64> = (0..city.len()).map(|c| city.pop[c] as f64 + city.jobs[c] as f64 + 1e-3).collect();
    let t = routes::walk_tables(grid, &z, &city.x, &city.y, &w, win, parts);
    std::fs::write(&out, &t.bytes).unwrap();
    std::fs::write(format!("{out}.part"), &t.part).unwrap();
    println!(
        "walk factors: {} zones, {} split by water into {} parts in all ({} searched), {} window pairs, {} with a detour over 1.2, {} over 5 (cut off); {:.0} ms",
        z.n,
        t.split_zones,
        t.units,
        t.searched,
        t.pairs,
        t.over,
        t.cut,
        now_ms() - t0
    );
}
