use super::*;
use crate::track::geom::height_at;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn base(a: &TrackApi) -> (f64, f64) {
    let p = a.from_lonlat(-73.985, 40.758);
    (p[0], p[1])
}
fn ends(a: &TrackApi, from: (f64, f64), to: (f64, f64), level: i8) -> [f64; 10] {
    let (bx, by) = base(a);
    let l = level as f64;
    [0.0, 0.0, bx + from.0, by + from.1, l, 0.0, 0.0, bx + to.0, by + to.1, l]
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn flats(a: &TrackApi) -> usize {
    a.w.net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))).count()
}
fn cost(a: &TrackApi) -> f64 {
    alive(a).iter().map(|&e| a.w.net.edge_cost[e as usize]).sum()
}
/// The lowest and highest heights along the edges added after `before`.
fn span(a: &TrackApi, before: &[u32]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for e in alive(a).into_iter().filter(|e| !before.contains(e)) {
        let (v, len) = (a.w.net.edge_vert(e), a.w.net.edge_len[e as usize]);
        let mut s = 0.0;
        while s <= len {
            let h = height_at(v, s);
            (lo, hi) = (lo.min(h), hi.max(h));
            s += 5.0;
        }
    }
    (lo, hi)
}
/// A line across x = 1500 at `level`, then a route along y = 0 at the same level, drawn with the
/// switch `on`: the heights the route reaches, the flat crossings, and the preview's price against
/// the built one.
fn cross(level: i8, on: bool, from: f64, tracks: u8) -> ((f64, f64), usize, f64, f64) {
    let mut a = api();
    assert_eq!(a.add_route(&ends(&a, (1500.0, -1500.0), (1500.0, 1500.0), level), &[], 2, false), 0, "{}", a.issue_kinds());
    let before = alive(&a);
    let c0 = cost(&a);
    a.set_route_ramp(on);
    let e = ends(&a, (from, 0.0), (3000.0, 0.0), level);
    let pv = a.preview_route(&e, &[], tracks, false);
    assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
    assert_eq!(a.add_route(&e, &[], tracks, false), 0, "{}", a.issue_kinds());
    (span(&a, &before), flats(&a), pv[1], cost(&a) - c0)
}

#[test]
fn at_grade_the_route_ramps_over_and_off_it_crosses_flat() {
    let ((lo, hi), n, pv, built) = cross(0, true, 0.0, 2);
    assert_eq!((lo, hi, n), (0.0, LEVEL_H, 0));
    assert!((pv - built).abs() < 1e-6, "preview {pv} built {built}");
    let ((lo, hi), n, _, _) = cross(0, false, 0.0, 2);
    assert_eq!((lo, hi, n), (0.0, 0.0, 1));
    // quad: both pairs go over
    let ((lo, hi), n, pv, built) = cross(0, true, 0.0, 4);
    assert_eq!((lo, hi, n), (0.0, LEVEL_H, 0));
    assert!((pv - built).abs() < 1e-6, "preview {pv} built {built}");
}

#[test]
fn below_ground_the_route_ramps_under() {
    let ((lo, hi), n, pv, built) = cross(-1, true, 0.0, 2);
    assert_eq!((lo, hi, n), (-2.0 * LEVEL_H, -LEVEL_H, 0));
    assert!((pv - built).abs() < 1e-6);
    let ((lo, hi), n, _, _) = cross(1, true, 0.0, 2);
    assert_eq!((lo, hi, n), (LEVEL_H, 2.0 * LEVEL_H, 0));
    // at the top level there is no way up, so it goes under
    let ((lo, hi), n, _, _) = cross(3, true, 0.0, 2);
    assert_eq!((lo, hi, n), (2.0 * LEVEL_H, 3.0 * LEVEL_H, 0));
}

#[test]
fn without_room_for_a_ramp() {
    // T-2414: a new start with no room for the ramp up stays up and goes over
    let ((lo, hi), n, _, _) = cross(0, true, 1350.0, 2);
    assert_eq!((lo, hi, n), (0.0, LEVEL_H, 0));
    // from a track end there, it crosses flat
    let mut a = api();
    assert_eq!(a.add_route(&ends(&a, (1500.0, -1500.0), (1500.0, 1500.0), 0), &[], 2, false), 0);
    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (1350.0, 0.0), 0), &[], 2, false), 0);
    let before = alive(&a);
    a.set_route_ramp(true);
    let (bx, by) = base(&a);
    let n = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - bx - 1350.0).abs() < 1.0).unwrap();
    assert_eq!(a.add_route(&[1.0, n as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 3000.0, by, 0.0], &[], 2, false), 0, "{}", a.issue_kinds());
    assert_eq!((span(&a, &before), flats(&a)), ((0.0, 0.0), 1));
}

#[test]
fn a_flat_crossing_on_the_way_up_sends_it_under() {
    let mut a = api();
    assert_eq!(a.add_route(&ends(&a, (1500.0, -1500.0), (1500.0, 1500.0), 0), &[], 2, false), 0);
    assert_eq!(a.add_route(&ends(&a, (1620.0, -1500.0), (1620.0, 1500.0), 1), &[], 2, false), 0, "{}", a.issue_kinds());
    let before = alive(&a);
    a.set_route_ramp(true);
    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (3000.0, 0.0), 0), &[], 2, false), 0, "{}", a.issue_kinds());
    let (lo, hi) = span(&a, &before);
    assert_eq!((lo, hi, flats(&a)), (-LEVEL_H, 0.0, 0));
}
