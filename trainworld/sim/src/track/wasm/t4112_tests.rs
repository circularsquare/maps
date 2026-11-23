//! T-4112: a station placed on a blueprint piece that ramps over water, away from the ramp, leaves the ramp where
//! it was (a cut inside a wide curve gave the curve's PI, which held the ramp, a new middle, and the ramp slid onto the
//! water: "track at ground level can't cross water" for a station kilometres away).

use super::*;
use crate::track::geom::height_at;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.w = TrackWorld::new(Network::new_in(FrameKind::Merc, -73.985, 40.758, true));
    a.set_cash(1e9);
    a
}
fn base(a: &TrackApi) -> (f64, f64) {
    let p = a.from_lonlat(-73.985, 40.758);
    (p[0], p[1])
}
/// Water where x (from `base`) lies between `x0` and `x1`.
fn water(a: &mut TrackApi, x0: f64, x1: f64) {
    let (bx, _) = base(a);
    a.w.net.set_water_mask(Box::new(move |x: f64, _y: f64| x > bx + x0 && x < bx + x1));
    a.w.reload();
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// The height at each place along the old alignment, read off the piece nearest there now: the largest difference.
fn moved(a: &TrackApi, pieces: &[crate::track::geom::Piece], vert: &[[f64; 2]], len: f64) -> f64 {
    let mut worst = 0.0f64;
    let mut t = 0.0;
    while t <= len {
        let (x, y, _) = crate::track::geom::pos_at(pieces, t);
        let (q, (qs, _)) = alive(a).into_iter().map(|q| (q, a.w.net.project(q, x, y))).min_by(|p, r| p.1 .1.total_cmp(&r.1 .1)).unwrap();
        worst = worst.max((height_at(a.w.net.edge_vert(q), qs) - height_at(vert, t)).abs());
        t += 10.0;
    }
    worst
}

/// Her case, rebuilt: high speed track drawn at 320 km/h through a wide curve onto a bridge, the ramp held by the
/// curve's PI. A station anywhere on the curve, kilometres from the ramp, is placed and the ramp stays; on the ramp
/// it is refused for the platform. A tram stop standing on water at ground level is still refused for that.
#[test]
fn t4112_a_station_far_from_a_ramp_leaves_it_where_it_was() {
    let mut a = api();
    water(&mut a, 7600.0, 10600.0);
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    let (bx, by) = base(&a);
    let e = [0.0, 0.0, bx, by, 0.0, 0.0, 0.0, bx + 10000.0, by + 6000.0, 0.0];
    assert_eq!(a.add_route(&e, &[bx + 5000.0, by, 9000.0, 0.0], 2, false), 0, "{}", a.issues());
    let p = alive(&a)[0];
    let (pieces, vert, len) = (a.w.net.edge_pieces(p).to_vec(), a.w.net.edge_vert(p).to_vec(), a.w.net.edge_len[p as usize]);
    let ramps = crate::track::geom::profile_ramps(&vert);
    assert_eq!(ramps.len(), 1, "{ramps:?}");
    for s in [900.0, 3000.0, 5000.0, 6900.0] {
        let (x, y, _) = crate::track::geom::pos_at(&pieces, s);
        assert_eq!(a.add_station(2, p, x, y, 400, "S".into()), 0, "at {s}: {}", a.issues());
        let dh = moved(&a, &pieces, &vert, len);
        assert!(dh < 0.05, "at {s}: the ramp moved ({dh} m of height)");
        assert_eq!(a.undo(), 0);
    }
    let mid = (ramps[0][0] + ramps[0][1]) / 2.0;
    let (x, y, _) = crate::track::geom::pos_at(&pieces, mid);
    assert_ne!(a.add_station(2, p, x, y, 400, "S".into()), 0);
    assert_eq!(a.issue_kinds(), "PlatformNotLevel");

    let mut t = TrackApi::new(-73.985, 40.758, true);
    t.set_cash(1e9);
    water(&mut t, 1000.0, 1500.0);
    t.set_route_kind(2);
    let (bx, by) = base(&t);
    assert_eq!(t.add_route(&[0.0, 0.0, bx, by, 0.0, 0.0, 0.0, bx + 3000.0, by, 0.0], &[], 2, false), 0, "{}", t.issues());
    let q = alive(&t)[0];
    let (x, y, _) = crate::track::geom::pos_at(t.w.net.edge_pieces(q), 1250.0);
    assert_ne!(t.add_station(2, q, x, y, 60, "S".into()), 0);
    assert_eq!(t.issue_kinds(), "StationGroundOverWater");
}
