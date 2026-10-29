//! T-1779: how long the drawing tool's preview takes for branches off a real save's track, natively.
//!
//!   cargo run --release --bin branch_bench -- <track bytes> [edges] [seed]
//!
//! Loads the network as `edit_bench` does, then from the middle of `edges` sampled edges (default
//! 200) previews a route to open ground at several angles and distances, as the track tool asks (through, design 160, auto ramp). Prints the slowest previews
//! and the time per preview. `TW_ADD=1` also places each accepted one and undoes it, timing both;
//! `TW_NORAMP=1` previews without the auto ramp; `TW_WHY=1` with `TW_ADD` prints each placement's
//! demand-dirty lines (T-1805: none, unless it changes a train's run). A save's track bytes are those after its gzip,
//! magic and JSON header (`saveFile.ts`).

use std::time::Instant;
use trainworld_sim::track::wasm::TrackApi;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&args[0]).expect("track bytes");
    let want: usize = args.get(1).map_or(200, |s| s.parse().unwrap());
    let mut seed: u64 = args.get(2).map_or(7, |s| s.parse().unwrap());
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert!(a.load(&bytes), "not a save");
    a.set_unlimited(true);
    let info = a.edges_info();
    let edges: Vec<(u32, f64)> = info.chunks(10).filter(|c| c[5] > 300.0).map(|c| (c[0] as u32, c[8])).collect();
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut times = vec![];
    let mut slow: Vec<(f64, String)> = vec![];
    let (mut ok, mut bad) = (0, 0);
    let mut adds: Vec<(f64, f64, u32)> = vec![];
    for _ in 0..want {
        let (e, _) = edges[(rnd() % edges.len() as u64) as usize];
        let pl = a.edge_polyline(e, 0.0);
        let m = pl.len() / 4 * 2;
        if m + 3 >= pl.len() {
            continue;
        }
        let (x, y) = (pl[m] as f64, pl[m + 1] as f64);
        let h = (pl[m + 3] as f64 - pl[m + 1] as f64).atan2(pl[m + 2] as f64 - pl[m] as f64);
        for side in [1.0, -1.0] {
            for deg in [8.0f64, 20.0, 40.0] {
                for d in [150.0, 400.0, 1000.0, 2500.0] {
                    let t = h + side * deg.to_radians();
                    let (tx, ty) = (x + d * t.cos(), y + d * t.sin());
                    a.set_route_tangent(true);
                    a.set_design_speed(160.0);
                    a.set_route_hsr(false);
                    a.set_route_ramp(std::env::var("TW_NORAMP").is_err());
                    a.set_route_beside(u32::MAX, u32::MAX);
                    let ends = [2.0, e as f64, x, y, 0.0, 0.0, 0.0, tx, ty, 0.0];
                    let t0 = Instant::now();
                    let out = a.preview_route(&ends, &[], 2, false);
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    let good = out[0] == 1.0;
                    if good && std::env::var("TW_ADD").is_ok() {
                        // a blueprint placed as the click does, then undone
                        let t1 = Instant::now();
                        a.set_route_tangent(true);
                        a.set_route_ramp(true);
                        a.set_route_beside(u32::MAX, u32::MAX);
                        let r = a.add_route(&ends, &[], 2, false);
                        let add = t1.elapsed().as_secs_f64() * 1e3;
                        if std::env::var("TW_WHY").is_ok() && r == 0 {
                            // T-1805: what a blueprint branch marks for demand and for drawing
                            eprintln!("add {add:.1} ms: demand lines {:?}, recomputed {}, charge {:.3}", a.dirty_demand_lines(), a.dirty_lines().len(), a.last_charge());
                        }
                        let t2 = Instant::now();
                        a.undo();
                        adds.push((add, t2.elapsed().as_secs_f64() * 1e3, r));
                    }
                    if good { ok += 1 } else { bad += 1 }
                    times.push(ms);
                    let why = if good { String::new() } else { a.issue_kinds() };
                    slow.push((ms, format!("edge {e} at ({x:.0},{y:.0}) side {side} {deg} deg {d} m -> ok {good} {why}")));
                }
            }
        }
    }
    slow.sort_by(|p, q| q.0.total_cmp(&p.0));
    for (ms, s) in slow.iter().take(15) {
        eprintln!("{ms:>8.1} ms  {s}");
    }
    if !adds.is_empty() {
        let mut ad: Vec<f64> = adds.iter().map(|a| a.0).collect();
        ad.sort_by(f64::total_cmp);
        let un = adds.iter().map(|a| a.1).sum::<f64>() / adds.len() as f64;
        let failed = adds.iter().filter(|a| a.2 != 0).count();
        eprintln!("{} adds ({failed} failed): median {:.1} ms, p95 {:.1}, max {:.1}; undo mean {un:.1} ms", ad.len(), ad[ad.len() / 2], ad[ad.len() * 95 / 100], ad[ad.len() - 1]);
    }
    times.sort_by(f64::total_cmp);
    let n = times.len();
    let mean = times.iter().sum::<f64>() / n as f64;
    eprintln!("{n} previews ({ok} ok, {bad} refused): mean {mean:.1} ms, median {:.1}, p95 {:.1}, max {:.1}", times[n / 2], times[n * 95 / 100], times[n - 1]);
}
