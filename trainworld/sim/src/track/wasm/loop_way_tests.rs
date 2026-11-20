//! T-3835: a loop set both ways whose track takes it round only one way runs one way: its return run
//! never turns back at a station partway or goes round twice.

use super::*;
use crate::track::capacity::ResKind;
use crate::track::net::LoopKind;
use crate::track::tests::{edge, line, node, station, world};

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a
}

/// Single track round a triangle A, B, C (edges AB, BC, CA, a to b as named), a station at each corner.
fn triangle() -> (TrackApi, [u32; 3], [u32; 3]) {
    let mut w = world();
    let n = [node(&mut w, 0.0, 0.0, 0), node(&mut w, 3000.0, 0.0, 0), node(&mut w, 1500.0, 3000.0, 0)];
    let e = [
        edge(&mut w, n[0], n[1], &[], 0).unwrap(),
        edge(&mut w, n[1], n[2], &[(5000.0, 0.0), (5000.0, 3000.0)], 0).unwrap(),
        edge(&mut w, n[2], n[0], &[(-2000.0, 3000.0), (-2000.0, 0.0)], 0).unwrap(),
    ];
    for &x in &e {
        let mut d = w.net.edge_data(x).unwrap();
        d.tracks = 1;
        w.apply(Op::Edge { id: x, data: Some(d) }).unwrap();
    }
    for &x in &n {
        station(&mut w, x, 200);
    }
    let mut a = api();
    a.w = w;
    (a, n, e)
}

fn termini(a: &TrackApi) -> Vec<u32> {
    let cap = &a.w.svc.cap;
    (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Terminus).map(|r| cap.res[r].node).collect()
}

/// Nodes run 1 of line `l` passes, first stop to first stop.
fn back_nodes(a: &TrackApi, l: u32) -> Vec<u32> {
    a.w.svc.lines[l as usize].runs[1].nodes.clone()
}

#[test]
fn a_loop_its_track_takes_round_one_way_runs_one_way() {
    let (mut a, n, e) = triangle();
    let l = line(&mut a.w, &[n[0], n[1], n[2], n[0]], [8.0, 4.0, 2.0]);
    assert!(!a.line_one_way(l) && back_nodes(&a, l) == vec![n[0], n[2], n[1], n[0]], "two-way track: both ways round");
    // its count set first, as a save loads: kept across the change
    assert_eq!(a.set_train_counts(l, 4, 2, 2), 0, "{}", a.issues());
    // AB one way: run 1 used to go A, C, B and turn back at B to reach A by C, a turn at a station on the loop
    assert_eq!(a.set_edge_oneway(e[0], 1), 0, "{}", a.issues());
    assert_eq!(a.w.svc.lines[l as usize].kind, LoopKind::Circle);
    assert!(a.line_one_way(l) && a.line_one_way_by_track(l), "it runs one way, as its track goes round");
    assert!(termini(&a).is_empty(), "no turn anywhere: {:?}", termini(&a));
    assert!(a.w.svc.lines[l as usize].running && !a.w.net.line_broken[l as usize]);
    // the round trip is one lap: run 0 and the dwell at the first stop
    let one_lap = |a: &TrackApi| {
        let ls = &a.w.svc.lines[l as usize];
        let lev = ROW_SLOT[0];
        (ls.round_trip[lev] - (ls.runs[0].prof[lev].duration + a.w.net.line_dwell[l as usize] as f64)).abs() < 1e-6
    };
    assert!(one_lap(&a));
    assert_eq!(a.w.net.line_trains[l as usize], Some([4, 2, 2]), "the count stays");
    assert!(ROW_SLOT.iter().all(|&lev| a.w.net.line_tph[l as usize][lev] > 0.0));
    // one way, an odd count is kept
    assert_eq!(a.set_train_counts(l, 3, 2, 1), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_trains[l as usize], Some([3, 2, 1]));
    // every piece one way: run 1 used to go round twice to stop the other way round
    for &x in &e[1..] {
        assert_eq!(a.set_edge_oneway(x, 1), 0, "{}", a.issues());
        assert!(a.line_one_way(l) && termini(&a).is_empty());
    }
    assert!(one_lap(&a));
    // set one way by the player, it is no longer the track's doing
    assert_eq!(a.set_line_one_way(l, true), 0, "{}", a.issues());
    assert!(a.line_one_way(l) && !a.line_one_way_by_track(l));
    assert_eq!(a.set_line_one_way(l, false), 0, "{}", a.issues());
    // the track two-way again: both ways round again
    for &x in &e {
        assert_eq!(a.set_edge_oneway(x, 0), 0, "{}", a.issues());
    }
    assert!(!a.line_one_way(l) && back_nodes(&a, l) == vec![n[0], n[2], n[1], n[0]]);
    assert_eq!(a.w.net.line_trains[l as usize], Some([4, 2, 2]), "both ways again, the odd counts round up");
}

/// A ring of double track whose bottom side splits into two singles 150 m apart and joins again (a pair
/// of one-way streets), the legs one way each, opposite: a loop over it runs both ways, each way on
/// its own leg. With both legs set east it runs one way.
#[test]
fn a_loop_with_a_way_back_of_its_own_runs_both_ways() {
    let mut a = api();
    let free = |x: f64, y: f64| [0.0, 0.0, x, y, 0.0];
    let ends = |p: [f64; 5], q: [f64; 5]| p.iter().chain(q.iter()).copied().collect::<Vec<f64>>();
    let near = |a: &TrackApi, x: f64, y: f64| {
        let n = &a.w.net;
        let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
        (0..n.node_count() as u32).filter(|&k| n.node_ok(k)).min_by(|&p, &q| d(p).total_cmp(&d(q))).unwrap()
    };
    let at = |a: &TrackApi, n: u32| [1.0, n as f64, a.w.net.node_x[n as usize], a.w.net.node_y[n as usize], 0.0];
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(2000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(free(4000.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let (j0, j1) = (near(&a, 2000.0, 0.0), near(&a, 4000.0, 0.0));
    for y in [75.0, -75.0] {
        assert_eq!(a.add_route(&ends(at(&a, j0), at(&a, j1)), &[3000.0, y, 0.0, 0.0], 1, false), 0, "{}", a.issues());
    }
    let (w0, e0) = (near(&a, 0.0, 0.0), near(&a, 6000.0, 0.0));
    let round = [7500.0, 0.0, 0.0, 0.0, 7500.0, 3000.0, 0.0, 0.0, -1500.0, 3000.0, 0.0, 0.0, -1500.0, 0.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends(at(&a, e0), at(&a, w0)), &round, 2, false), 0, "{}", a.issues());
    let net = &a.w.net;
    let legs: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_tracks[e as usize] == 1).collect();
    assert_eq!(legs.len(), 2);
    // each leg one way, east along one and west along the other (set, as a player sets an inferred way)
    let east = |net: &crate::track::net::Network, e: u32, w: u8| (net.node_x[net.edge_a[e as usize] as usize] < net.node_x[net.edge_b[e as usize] as usize]) == (w == 1);
    let way = |net: &crate::track::net::Network, e: u32, want_east: bool| if east(net, e, 1) == want_east { 1u8 } else { 2 };
    let (w_a, w_b) = (way(net, legs[0], true), way(net, legs[1], false));
    assert_eq!(a.set_edge_oneway(legs[0], w_a), 0, "{}", a.issues());
    assert_eq!(a.set_edge_oneway(legs[1], w_b), 0, "{}", a.issues());
    let mut st = vec![];
    for (x, y) in [(1000.0, 0.0), (5000.0, 0.0), (3000.0, 3000.0), (3000.0, -75.0)] {
        let n = &a.w.net;
        let e = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).min_by(|&p, &q| n.project(p, x, y).1.total_cmp(&n.project(q, x, y).1)).unwrap();
        let (px, py, _) = crate::track::geom::pos_at(n.edge_pieces(e), n.project(e, x, y).0);
        assert_eq!(a.add_station(2, e, px, py, 120, "S".into()), 0, "{}", a.issues());
        st.push(near(&a, px, py));
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[st[0], st[1], st[2], st[0]], 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issue_kinds());
    assert_eq!(a.w.svc.lines[l as usize].kind, LoopKind::Circle);
    assert!(!a.line_one_way(l), "run 1 goes round by its own leg");
    assert!(termini(&a).is_empty(), "{:?}", termini(&a));
    // a stop on the westbound leg too (a tram stop on one of a pair of one-way streets)
    let k = a.new_line_id();
    assert_eq!(a.set_line_with_trains(k, &[st[0], st[1], st[2], st[0]], 4, 2, 0, 30.0, 180.0, 0, 0, "K".into()), 0, "{}", a.issue_kinds());
    // (the click makes its stops three laps long, a separate bug; still no turn)
    assert_eq!(a.add_stop(k, st[3]), 0, "{}", a.issues());
    assert!(termini(&a).is_empty(), "{:?}", termini(&a));
    assert_eq!(a.remove_line(k), 0, "{}", a.issues());
    let w_b2 = way(&a.w.net, legs[1], true);
    assert_eq!(a.set_edge_oneway(legs[1], w_b2), 0, "{}", a.issues());
    assert!(a.line_one_way_by_track(l) && termini(&a).is_empty(), "both legs east: one way round");
}

/// A balloon (out along a stem, round a loop, back) set both ways keeps its turn at its first stop, the end of
/// its stem.
#[test]
fn a_balloon_still_turns_at_its_end() {
    let (mut w, [s, m, _, k]) = crate::track::tests::balloon_fixture();
    let l = line(&mut w, &[s, m, k, m, s], [12.0, 6.0, 3.0]);
    let mut a = api();
    a.w = w;
    assert!(!a.line_one_way(l));
    assert_eq!(termini(&a), vec![s]);
}
