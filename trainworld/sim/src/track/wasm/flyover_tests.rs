//! T-3103: the flat-junction badge only where moves can cross; T-3092: real flyovers.

use super::*;
use crate::track::capacity::junction_crosses;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// The node nearest (x, y) with 3+ edge ends.
fn junction_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n >= 3)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}
/// The flags column of `nodes_info` for node `k`.
fn flags(a: &TrackApi, k: u32) -> u8 {
    a.nodes_info().chunks_exact(10).find(|c| c[0] == k as f64).unwrap()[6] as u8
}

/// A main from (0, y) to (3000, y) of `main` tracks, and a branch of `branch` tracks leaving it at
/// x = 1000 for (2000, y - 600) (to the right looking along the main); the junction.
fn y_junction(a: &mut TrackApi, y: f64, main: u8, branch: u8) -> u32 {
    assert_eq!(a.add_route(&ends(free(0.0, y), free(3000.0, y)), &[], main, false), 0, "{}", a.issues());
    let e = *alive(a).iter().find(|&&e| a.w.net.project(e, 1000.0, y).1 < 1.0).unwrap();
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1000.0, y, 0.0], free(2000.0, y - 600.0)), &[], branch, false), 0, "{}", a.issues());
    junction_near(a, 1000.0, y)
}

#[test]
fn t3103_badge_only_where_moves_cross() {
    let mut a = api();
    // double track splitting: one direction's branch move crosses the other's main track
    let j = y_junction(&mut a, 0.0, 2, 2);
    assert!(junction_crosses(&a.w.net, j));
    assert_eq!(flags(&a, j), 2);
    // a two-way single branch off double track crosses too
    let j = y_junction(&mut a, 3000.0, 2, 1);
    assert!(junction_crosses(&a.w.net, j));
    // single track throughout: every move shares the one track; merges only
    let j = y_junction(&mut a, 6000.0, 1, 1);
    assert!(!junction_crosses(&a.w.net, j));
    assert_eq!(flags(&a, j), 0);
    // one-way single branch off double track: one way merges, the other crosses
    let j = y_junction(&mut a, 9000.0, 2, 1);
    let leg = a.w.net.node_ports[j as usize].ends().iter().map(|&c| c >> 1).find(|&e| a.w.net.edge_tracks[e as usize] == 1).unwrap();
    let mut crossing = vec![];
    for dir in [1, 2] {
        assert_eq!(a.set_edge_oneway(leg, dir), 0, "{}", a.issues());
        crossing.push(junction_crosses(&a.w.net, j));
    }
    assert_eq!(crossing.iter().filter(|&&c| c).count(), 1, "{crossing:?}");
    // a flying junction has no badge
    let j = y_junction(&mut a, 12000.0, 2, 2);
    assert_eq!(a.set_flying(j, true), 0, "{}", a.issues());
    assert_eq!(flags(&a, j), 1);
}

#[test]
fn t3103_preview_marks_only_crossing_junctions() {
    let mut a = api();
    for (y, tracks) in [(0.0, 2u8), (3000.0, 1)] {
        assert_eq!(a.add_route(&ends(free(0.0, y), free(3000.0, y)), &[], tracks, false), 0, "{}", a.issues());
        let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, y).1 < 1.0).unwrap();
        assert_eq!(a.preview_route(&ends([2.0, e as f64, 1000.0, y, 0.0], free(2000.0, y - 600.0)), &[], tracks, false)[0], 1.0, "{}", a.issues());
        let m = a.preview_marks();
        let marked = m.chunks(3).any(|r| r[2] == 1.0 && (r[0] - 1000.0).abs() < 1.0);
        assert_eq!(marked, tracks == 2, "tracks {tracks}: {m:?}");
    }
}

/// Every node with 3+ edge ends.
fn junctions(a: &TrackApi) -> Vec<u32> {
    let n = &a.w.net;
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n >= 3).collect()
}
fn dead_ends(a: &TrackApi) -> Vec<u32> {
    let n = &a.w.net;
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n == 1).collect()
}

/// A double main from (-3000, y) to (5000, y), and a double branch drawn with ramps on and real
/// flyovers from x = 1000 for (3500, y + side 1500).
fn flyover_y(a: &mut TrackApi, y: f64, side: f64) -> u32 {
    assert_eq!(a.add_route(&ends(free(-3000.0, y), free(5000.0, y)), &[], 2, false), 0, "{}", a.issues());
    let e = *alive(a).iter().find(|&&e| a.w.net.project(e, 1000.0, y).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    let r = a.add_route(&ends([2.0, e as f64, 1000.0, y, 0.0], free(3500.0, y + side * 1500.0)), &[], 2, true);
    a.set_route_ramp(false);
    r
}

#[test]
fn t3092_flyover_lays_two_one_way_legs() {
    for (right_hand, side) in [(true, 1.0), (true, -1.0), (false, 1.0)] {
        let mut a = TrackApi::new(-73.985, 40.758, right_hand);
        a.set_cash(1e9);
        assert_eq!(flyover_y(&mut a, 0.0, side), 0, "{}", a.issues());
        assert_eq!(a.route_flyovers_flat(), 0, "right hand {right_hand}, side {side}");
        let net = &a.w.net;
        let js = junctions(&a);
        assert_eq!(js.len(), 3, "J1, J2 and J3");
        for &j in &js {
            assert!(!junction_crosses(net, j), "right hand {right_hand}, side {side}: moves cross at {j}");
            assert!(!net.node_flying[j as usize]);
        }
        let singles: Vec<u32> = alive(&a).into_iter().filter(|&e| net.edge_tracks[e as usize] == 1).collect();
        assert!(singles.len() >= 2 && singles.iter().all(|&e| net.edge_oneway[e as usize] != 0), "{singles:?}");
        // the leg goes a level up over the main
        let up = singles.iter().any(|&e| net.edge_vert(e).iter().any(|v| v[1] >= LEVEL_H - 0.01));
        assert!(up, "the crossing leg is ramped");
        assert_eq!(net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_) | CrossKind::Bad)).count(), 0);
        // a line from the main's west end to the branch runs out and back on different legs, and a
        // line along the main meets it at no crossing move
        let ends_ = dead_ends(&a);
        let at = |x: f64, y: f64| *ends_.iter().min_by(|&&p, &&q| {
            let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap();
        let (w, e, b) = (at(-3000.0, 0.0), at(5000.0, 0.0), at(3500.0, side * 1500.0));
        for (n, name) in [(w, "W"), (e, "E"), (b, "B")] {
            let (x, y) = (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
            assert_eq!(a.add_station(1, n, x, y, 200, name.into()), 0, "{}", a.issues());
        }
        let branch = a.new_line_id();
        assert_eq!(a.set_line_with_trains(branch, &[w, b], 4, 2, 0, 30.0, 180.0, 0, 0, "Branch".into()), 0, "{}", a.issues());
        let main = a.new_line_id();
        assert_eq!(a.set_line_with_trains(main, &[w, e], 4, 2, 0, 30.0, 180.0, 0, 0, "Main".into()), 0, "{}", a.issues());
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        let ls = &a.w.svc.lines[branch as usize];
        let used = |r: usize| -> Vec<u32> { ls.runs[r].segs.iter().map(|&p| p >> 1).filter(|e| a.w.net.edge_tracks[*e as usize] == 1).collect() };
        let (out, back) = (used(0), used(1));
        assert!(!out.is_empty() && !back.is_empty());
        assert!(out.iter().all(|e| !back.contains(e)), "each direction its own leg: {out:?} {back:?}");
        assert!(crate::track::capacity::move_conflicts(&a.w.net, &a.w.svc.lines).is_empty());
    }
}

#[test]
fn t3092_no_room_falls_back_flat() {
    let mut a = api();
    // too short a route for J3
    assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 2, false), 0);
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    let to = ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(1250.0, 150.0));
    assert_eq!(a.preview_route(&to, &[], 2, true)[0], 1.0, "{}", a.issues());
    assert_eq!(a.route_flyovers_flat(), 1);
    assert!(a.preview_marks().chunks(3).any(|r| r[2] == 1.0), "the flat badge shows");
    assert_eq!(a.add_route(&to, &[], 2, true), 0, "{}", a.issues());
    assert_eq!(a.route_flyovers_flat(), 1);
    let js = junctions(&a);
    assert_eq!(js.len(), 1);
    assert!(junction_crosses(&a.w.net, js[0]));
    // single track: nothing to lay, no note
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 1, false), 0);
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(3500.0, 1500.0)), &[], 1, true), 0, "{}", a.issues());
    assert_eq!(a.route_flyovers_flat(), 0);
    assert_eq!(alive(&a).len(), 3);
}

#[test]
fn t3092_preview_draws_the_leg() {
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 2, false), 0);
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    let to = ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(3500.0, 1500.0));
    let flat = a.preview_route(&to, &[], 2, false);
    let fly = a.preview_route(&to, &[], 2, true);
    assert_eq!(fly[0], 1.0, "{}", a.issues());
    assert!(fly.len() > flat.len());
    assert!(a.preview_render_speeds().contains(&-1.0));
    assert!(fly[1] > flat[1], "the flyover costs more: {} vs {}", fly[1], flat[1]);
    assert!(!a.preview_marks().chunks(3).any(|r| r[2] == 1.0), "no flat badge");
    let before = alive(&a).len();
    assert_eq!(a.add_route(&to, &[], 2, true), 0);
    assert!(alive(&a).len() > before + 3);
    // undo takes the whole route
    a.undo();
    assert_eq!(alive(&a).len(), before);
}
