//! T-727: can the track model take one long route edit with thousands of clicked corners?
//! Reads lon,lat lines (start, PIs, end), places them as one `add_route` in a Mercator world.
//!
//!     t727_edit_test <pis.txt> [tracks]
use std::time::Instant;
use trainworld_sim::track::wasm::TrackApi;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let tracks: u8 = args.get(2).map_or(2, |s| s.parse().unwrap());
    let pts: Vec<(f64, f64)> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut it = l.split(',').map(|v| v.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    let mut a = TrackApi::new(pts[0].0, pts[0].1, true);
    assert!(a.set_empty_frame(2));
    a.set_unlimited(true);
    let xy: Vec<(f64, f64)> = pts.iter().map(|&(lon, lat)| {
        let p = a.from_lonlat(lon, lat);
        (p[0], p[1])
    }).collect();
    let n = xy.len();
    let ends = [0.0, 0.0, xy[0].0, xy[0].1, 0.0, 0.0, 0.0, xy[n - 1].0, xy[n - 1].1, 0.0];
    let mut pis = Vec::with_capacity((n - 2) * 4);
    for p in &xy[1..n - 1] {
        pis.extend_from_slice(&[p.0, p.1, 0.0, 0.0]);
    }
    let t = Instant::now();
    let pv = a.preview_route(&ends, &pis, tracks, false);
    println!("preview: ok {} cost {:.0} $M length {:.1} km vertices {} in {:.0} ms; issues: {}",
        pv[0], pv[1], pv[2] / 1000.0, pv[3], t.elapsed().as_secs_f64() * 1000.0, a.issue_kinds());
    let t = Instant::now();
    let r = a.add_route(&ends, &pis, tracks, false);
    println!("add_route: {} problems in {:.0} ms; issues: {}", r, t.elapsed().as_secs_f64() * 1000.0, a.issue_kinds());
    let s = a.save();
    println!("save: {} bytes", s.len());
    let t = Instant::now();
    let u = a.undo();
    println!("undo: {} in {:.0} ms", u, t.elapsed().as_secs_f64() * 1000.0);
}
