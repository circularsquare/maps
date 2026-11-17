//! T-3388: edit times on a real save, natively, by edit kind (scratch probe; a copy is kept in
//! notes/T-3388/).
//!
//!   cargo run --release --bin t3388_probe -- <track bytes> <out prefix> [steps] [line]
//!
//! Loads as edit_bench does, then on `line` (default the busiest Medium line): a train-count edit
//! (+1 medium) and its undo, every level +1 and its undo, a spur of track off one of its plain
//! nodes and its undo, lone track and its undo, a station on one of its edges and its undo.
//! `steps`: comma-separated step names to run (default all, "all"). Writes per line: id, then
//! tph (high, medium, low), round trip s, trains, pace s, after the load and after each step to
//! `<out prefix>.<step>.txt`, and prints the paced line-levels whose pace is under their headway.

use std::time::Instant;
use trainworld_sim::track::wasm::TrackApi;

fn dump(a: &TrackApi, path: &str) -> usize {
    let w = a.world();
    let mut s = String::new();
    let mut bad = 0;
    for l in 0..w.net.line_count() {
        if !w.net.line_ok(l as u32) || l >= w.svc.lines.len() { continue; }
        let ls = &w.svc.lines[l];
        let p = w.net.line_pace_of(l as u32);
        let c = w.net.line_trains[l].unwrap_or([0; 3]);
        for lev in 0..3 {
            if p[lev] > 0.0 && c[lev] > 0 && ls.round_trip[lev] / c[lev] as f64 > p[lev] as f64 * 1.000001 {
                bad += 1;
            }
        }
        s += &format!("{l}\t{:?}\t{:?}\t{:?}\t{:?}\n", w.net.line_tph[l], ls.round_trip, c, p);
    }
    std::fs::write(path, s).unwrap();
    bad
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&args[0]).expect("track bytes");
    let out = args.get(1).cloned().unwrap_or_else(|| "probe".into());
    let only: Vec<String> = args.get(2).filter(|s| *s != "all").map_or(vec![], |s| s.split(',').map(String::from).collect());
    let line: Option<u32> = args.get(3).map(|s| s.parse().unwrap());
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    let mut a = TrackApi::new(-73.985, 40.758, true);
    let t = Instant::now();
    assert!(a.load(&bytes), "not a save");
    eprintln!("load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    a.set_unlimited(true);
    let mut now = 111600.0;
    a.set_now(now);
    let bad = dump(&a, &format!("{out}.load.txt"));
    let info = a.lines_info();
    let best = line.unwrap_or_else(|| info.chunks(22).max_by_key(|c| c[14] as u32).map(|c| c[0] as u32).unwrap());
    let tr = |a: &TrackApi| a.world().net.line_trains[best as usize].unwrap_or([0; 3]);
    let t0 = tr(&a);
    eprintln!("{} lines; line {best} {:?}, trains {:?}; paced under headway after load {bad}", info.len() / 22, a.line_name(best), t0);
    // a plain node on the line (two ports, no platform), and an edge of it
    let (node, edge, nx, ny, ex, ey) = {
        let net = &a.world().net;
        let path: Vec<u32> = net.line_path(best).iter().map(|&p| trainworld_sim::track::net::path_edge(p)).collect();
        let e = path[path.len() / 2];
        let mut node = u32::MAX;
        for pass in 0..2 {
            for &pe in path.iter().skip(path.len() / 3).chain(path.iter()) {
                for n in [net.edge_a[pe as usize], net.edge_b[pe as usize]] {
                    if net.node_ports[n as usize].n == 2 && (pass == 1 || net.node_platform[n as usize] == 0) { node = n; break; }
                }
                if node != u32::MAX { break; }
            }
            if node != u32::MAX { break; }
        }
        let (ea, eb) = (net.edge_a[e as usize] as usize, net.edge_b[e as usize] as usize);
        (node, e, net.node_x[node as usize], net.node_y[node as usize], 0.5 * (net.node_x[ea] + net.node_x[eb]), 0.5 * (net.node_y[ea] + net.node_y[eb]))
    };
    eprintln!("spur node {node} at ({nx:.0}, {ny:.0}); station edge {edge} at ({ex:.0}, {ey:.0})");
    let mut step = |a: &mut TrackApi, name: &str, f: &dyn Fn(&mut TrackApi) -> u32| {
        if !only.is_empty() && !only.iter().any(|o| o == name) { return; }
        now += 30.0;
        a.set_now(now);
        eprintln!("{name}:");
        let t = Instant::now();
        let r = f(a);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let bad = dump(a, &format!("{out}.{name}.txt"));
        eprintln!("{name:<16} {ms:>8.1} ms  r={r}  trains {:?}  paced under headway {bad}", tr(a));
    };
    let [h, m, l] = t0;
    step(&mut a, "count", &|a| a.set_train_counts(best, h, m + 1, l));
    step(&mut a, "count_undo", &|a| a.undo());
    step(&mut a, "count2", &|a| a.set_train_counts(best, h + 1, m + 1, l + 1));
    step(&mut a, "count2_undo", &|a| a.undo());
    // a spur off the plain node
    step(&mut a, "spur", &|a| a.add_route(&[1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, nx + 300.0, ny + 260.0, 0.0], &[], 2, false));
    step(&mut a, "spur_undo", &|a| a.undo());
    // track nowhere near anything
    step(&mut a, "lone", &|a| a.add_route(&[0.0, 0.0, nx + 20000.0, ny + 20000.0, 0.0, 0.0, 0.0, nx + 20400.0, ny + 20300.0, 0.0], &[], 2, false));
    step(&mut a, "lone_undo", &|a| a.undo());
    step(&mut a, "station", &|a| a.add_station(2, edge, ex, ey, 0, "Probe".into()));
    step(&mut a, "station_undo", &|a| a.undo());
}
