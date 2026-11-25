use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}

/// T-244: the corners of a stroke, and the same stroke as one edge (set directly).
const CLICKS: [(f64, f64); 4] = [(1500.0, 300.0), (2600.0, -400.0), (4800.0, 1200.0), (6200.0, 900.0)];
fn stroke(level_of: &dyn Fn(usize) -> i8) -> (Vec<f64>, Vec<f64>) {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(7600.0, 2400.0).iter()).copied().collect();
    let pis = CLICKS.iter().enumerate().flat_map(|(k, &(x, y))| [x, y, 0.0, level_of(k) as f64]).collect();
    (ends, pis)
}
fn one_edge(pis: &[f64]) -> (TrackApi, u32) {
    let mut b = api();
    let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
    assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(b.set_node(n1, 7600.0, 2400.0, 0, false), 0);
    assert_eq!(b.set_edge(e, n0, n1, 2, pis), 0, "{}", b.issues());
    (b, e)
}
fn live_edges(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// Every drawn point of `a`'s track lies on `b`'s edge `e`, and their lengths agree.
fn same_track(a: &TrackApi, b: &TrackApi, e: u32, tol: f64) {
    let mut len = 0.0;
    for p in live_edges(a) {
        len += a.w.net.edge_len[p as usize];
        for q in a.edge_render(p).chunks(3) {
            let off = b.w.net.project(e, q[0], q[1]).1;
            assert!(off < tol, "piece {p}: {off} m off the stroke at {q:?}");
        }
    }
    assert!((len - b.w.net.edge_len[e as usize]).abs() < tol, "{len} against {}", b.w.net.edge_len[e as usize]);
}

fn from_node(n: u32, x: f64, y: f64) -> Vec<f64> {
    vec![1.0, n as f64, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, 0.0]
}

fn free_at(x: f64, y: f64, level: i8) -> Vec<f64> {
    vec![0.0, 0.0, x, y, level as f64]
}
/// A straight piece from a free point or an earlier piece's end, `pis` (x, y, level) between;
/// returns its far node.
fn piece(a: &mut TrackApi, from: Result<u32, (f64, f64, i8)>, pis: &[(f64, f64, i8)], to: (f64, f64, i8)) -> u32 {
    let mut ends = match from {
        Ok(n) => vec![1.0, n as f64, 0.0, 0.0, 0.0],
        Err((x, y, l)) => free_at(x, y, l),
    };
    ends.extend(free_at(to.0, to.1, to.2));
    let pis: Vec<f64> = pis.iter().flat_map(|p| [p.0, p.1, 0.0, p.2 as f64]).collect();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    a.route_end()
}
/// The edge a click at (x, y) lands on: the one it is nearest, with the click inside it.
fn edge_under(a: &TrackApi, x: f64, y: f64) -> u32 {
    let net = &a.w.net;
    live_edges(a)
        .into_iter()
        .map(|e| {
            let (s, off) = net.project(e, x, y);
            (e, if off > 1.0 { 1e6 } else { off + if s < 0.01 || s > net.edge_len[e as usize] - 0.01 { 1e3 } else { 0.0 } })
        })
        .min_by(|p, q| p.1.total_cmp(&q.1))
        .unwrap()
        .0
}
fn never(res: &[(f64, String)]) {
    let bad: Vec<_> = res.iter().filter(|r| !r.1.is_empty()).collect();
    assert!(bad.is_empty(), "{bad:?}");
}

mod ground_tests;
mod line_tests;
mod fleet_tests;
mod blueprint_tests;
mod drag_tests;
mod route_tests;
mod stroke_tests;
mod platform_tests;
mod station_tests;
mod answer_tests;
mod quad_tests;
mod group_tests;
mod reach_tests;
