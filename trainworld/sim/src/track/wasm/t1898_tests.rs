//! T-1898: a drag keeps the speed track was drawn for, and a selection's track type.
use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}

/// A 1 km straight from the origin, then at `kmh` a tangent arc from its east end through a click
/// 5 km off at 25 degrees (radius about 5,915 m, set since it is over 1,796). The east joint and
/// the arc's edge.
fn drawn(kmh: f64) -> (TrackApi, u32, u32) {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let east = a.route_end();
    a.set_route_tangent(true);
    a.set_design_speed(kmh);
    let off = 25f64.to_radians();
    let ends = vec![1.0, east as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0 + 5000.0 * off.cos(), 5000.0 * off.sin(), 0.0];
    let before: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let arc = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && !before.contains(&e)).unwrap();
    (a, east, arc)
}

fn tightest(a: &TrackApi, e: u32) -> f64 {
    a.w.net.edge_pieces(e).iter().filter(|p| p.k != 0.0).map(|p| p.radius()).fold(f64::INFINITY, f64::min)
}

/// The arc's one PI moved `out` metres away from the chord (negative: towards it).
fn bent(a: &TrackApi, e: u32, out: f64) -> Vec<f64> {
    let d = a.edge_pis(e);
    assert_eq!(d.len(), 9, "one PI");
    let (x, y, r, l) = (d[1], d[2], d[3], d[4]);
    // the chord runs from the east joint (1000, 0) to the click at 25 degrees; the PI is on its right
    let th = 25f64.to_radians();
    let (nx, ny) = (th.sin(), -th.cos());
    vec![x + nx * out, y + ny * out, r, l]
}

/// Anita (T-1898): a 320 km/h curve bent a little keeps about its speed instead of dropping to
/// 160; bent gentler it keeps its radius; bent hard it narrows only as far as its legs need.
#[test]
fn a_bent_curve_keeps_its_design_speed() {
    let (mut a, _, arc) = drawn(320.0);
    let r0 = tightest(&a, arc);
    assert!(r0 > 5800.0 && r0 < 6000.0, "drawn at {r0}");
    let set = a.edge_pis(arc)[3];
    assert!(set > R_CAP, "set radius {set}");
    for (out, at_least) in [(-20.0, r0 - 1.0), (5.0, 0.95 * r0), (20.0, 0.9 * r0), (400.0, R_CAP + 1.0)] {
        let mut b = drawn(320.0).0;
        let pis = bent(&b, arc, out);
        assert_eq!(b.set_edge_pis(arc, &pis), 0, "{out}: {}", b.issues());
        let r = tightest(&b, arc);
        assert!(r >= at_least && r <= set + 1e-6, "bent {out} m: radius {r}, want at least {at_least}");
        assert_eq!(b.edge_pis(arc)[3], set, "{out}: the set radius is kept as the most");
    }
    // bent out and back in two edits: the curve is as wide as before
    let pis = bent(&a, arc, 30.0);
    assert_eq!(a.set_edge_pis(arc, &pis), 0);
    assert!(tightest(&a, arc) < r0 - 1.0);
    let pis = bent(&a, arc, -30.0);
    assert_eq!(a.set_edge_pis(arc, &pis), 0);
    assert!((tightest(&a, arc) - r0).abs() < 1.0, "back to {}", tightest(&a, arc));
}

/// A curve with an auto radius (under 1,796 m when drawn) bent wider takes the design speed's
/// radius where it has room, as drawing would; at 160 it stays auto.
#[test]
fn a_bent_auto_corner_widens_to_the_design_speed() {
    for kmh in [160.0, 320.0] {
        let mut a = api();
        a.set_route_tangent(true);
        a.set_design_speed(kmh);
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
        // a straight edge with an auto corner made at 160, then that corner dragged towards the chord
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let e = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e)).unwrap();
        a.set_design_speed(160.0);
        assert_eq!(a.set_edge_pis(e, &[2000.0, 900.0, 0.0, 0.0]), 0);
        assert!(tightest(&a, e) <= R_CAP + 1e-6);
        a.set_design_speed(kmh);
        assert_eq!(a.set_edge_pis(e, &[2000.0, 300.0, 0.0, 0.0]), 0);
        let r = tightest(&a, e);
        if kmh == 160.0 {
            assert!((r - R_CAP).abs() < 1e-6, "160: {r}");
            assert_eq!(a.edge_pis(e)[3], 0.0);
        } else {
            assert!(r > 3 as f64 * R_CAP, "320: {r}");
        }
    }
}

/// A joint dragged sideways: the straight's new lead-in curve is laid out for the design speed
/// (wider than 1,796 m at 320), and the arc beyond keeps its set radius.
#[test]
fn a_dragged_joint_lays_its_new_curve_for_the_design_speed() {
    for kmh in [160.0, 320.0] {
        let (mut a, east, arc) = drawn(kmh);
        let straight = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && e != arc).unwrap();
        let set = a.edge_pis(arc)[3];
        let (x, y) = (a.w.net.node_x[east as usize], a.w.net.node_y[east as usize]);
        assert_eq!(a.move_node(east, x, y + 30.0), 0, "{kmh}: {}", a.issues());
        let r = tightest(&a, straight);
        if kmh == 160.0 {
            assert!(r <= R_CAP + 1e-6, "160: {r}");
        } else {
            assert!(r > R_CAP + 1.0, "320: lead-in {r}");
            assert_eq!(a.edge_pis(arc)[3], set, "the arc's radius kept");
        }
        let (mut b, east, _) = drawn(kmh);
        assert_eq!(b.move_nodes(&[east], 0.0, 30.0), 0);
        assert!((tightest(&b, straight) - r).abs() < 1e-6, "move_nodes as move_node");
    }
}

/// The selection's track type: double blueprint track changes, single track and constructed
/// track are counted as skipped; one undo puts it back.
#[test]
fn set_edges_hsr_changes_double_blueprint_only() {
    let mut a = api();
    a.set_cash(1e6);
    let mut ids = vec![];
    for (y, tracks) in [(0.0, 2u8), (500.0, 1), (1000.0, 2)] {
        let ends: Vec<f64> = free(0.0, y).iter().chain(free(2000.0, y).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], tracks, false), 0);
        ids.push((0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).max().unwrap());
    }
    assert_eq!(a.construct(&[ids[2]], &[]), 0, "{}", a.issues());
    assert_eq!(a.set_edges_hsr(&ids, true), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 2);
    assert!(a.w.net.edge_hsr[ids[0] as usize] && !a.w.net.edge_hsr[ids[1] as usize] && !a.w.net.edge_hsr[ids[2] as usize]);
    // already high speed: nothing to do, nothing skipped among the double blueprint
    assert_eq!(a.set_edges_hsr(&ids[..1], true), 1);
    assert_eq!(a.last_skipped(), 0);
    a.undo();
    assert!(!a.w.net.edge_hsr[ids[0] as usize]);
    assert_eq!(a.set_edges_hsr(&ids[..1], true), 0);
    assert_eq!(a.set_edges_hsr(&ids[..1], false), 0);
    assert!(!a.w.net.edge_hsr[ids[0] as usize]);
}
