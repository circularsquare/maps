//! T-3094: one-way single track.

use super::*;
use crate::track::capacity::ResKind;
use crate::track::net::LoopKind;
use crate::track::tests::{edge, line, node, station, world};

/// Single track round a ring with a station at each of A, B and C: AB is a straight (the short way
/// from A to B), BC and CA go round the long way, over the top; edges AB, BC, CA, a to b as named.
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

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a
}

fn runs(a: &TrackApi, l: u32) -> [Vec<u32>; 2] {
    let ls = &a.w.svc.lines[l as usize];
    [ls.runs[0].segs.clone(), ls.runs[1].segs.clone()]
}

#[test]
fn routing_never_runs_against_one_way_track() {
    let (mut a, n, e) = triangle();
    assert_eq!(a.set_edge_oneway(e[0], 1), 0, "{}", a.issues());
    assert_eq!(a.edge_oneway(e[0]), 1);
    assert_eq!(a.oneway_edges(), vec![e[0], 1]);
    let net = &a.w.net;
    assert_eq!(net.find_path(n[0], n[1], None).unwrap().0, vec![e[0] << 1], "A to B goes with it");
    let back = net.find_path(n[1], n[0], None).unwrap().0;
    assert!(back.iter().all(|&p| net.dir_ok(p)) && !back.contains(&(e[0] << 1 | 1)), "B to A goes round by C: {back:?}");
    assert_eq!(back.len(), 2);
    // flipped, the other way round
    assert_eq!(a.set_edge_oneway(e[0], 2), 0, "{}", a.issues());
    let net = &a.w.net;
    assert_eq!(net.find_path(n[1], n[0], None).unwrap().0, vec![e[0] << 1 | 1]);
    assert_eq!(net.find_path(n[0], n[1], None).unwrap().0.len(), 2);
    // double track has no flag
    let mut d = a.w.net.edge_data(e[0]).unwrap();
    d.tracks = 2;
    a.w.apply(Op::Edge { id: e[0], data: Some(d) }).unwrap();
    assert_eq!(a.edge_oneway(e[0]), 0);
    assert_eq!(a.set_edge_oneway(e[0], 1), 1, "double track refuses");
}

#[test]
fn a_line_takes_another_way_in_the_same_edit_and_undo_puts_it_back() {
    let (mut a, n, e) = triangle();
    let l = line(&mut a.w, &[n[0], n[1]], [8.0, 4.0, 2.0]);
    assert_eq!(a.w.net.line_path(l), &[e[0] << 1]);
    assert_eq!(runs(&a, l)[1], vec![e[0] << 1 | 1]);
    assert_eq!(a.oneway_effects(e[0], 1), vec![l as f64, 1.0], "its way back would change");
    assert_eq!(a.set_edge_oneway(e[0], 1), 0, "{}", a.issues());
    // out over AB, back round by C
    assert_eq!(a.w.net.line_path(l), &[e[0] << 1]);
    let back = runs(&a, l)[1].clone();
    assert_eq!(back.len(), 2, "{back:?}");
    assert!(!back.contains(&(e[0] << 1 | 1)));
    assert!(!a.w.net.line_broken[l as usize]);
    assert!(a.w.svc.lines[l as usize].running);
    // reversing sends the way out round by C and the way back over AB
    assert_eq!(a.oneway_effects(e[0], 2), vec![l as f64, 1.0]);
    assert_eq!(a.set_edge_oneway(e[0], 2), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_path(l).len(), 2);
    assert_eq!(runs(&a, l)[1].len(), 2, "back the same way round");
    assert!(!a.w.net.line_broken[l as usize]);
    // one undo step each
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edge_oneway(e[0]), 1);
    assert_eq!(a.w.net.line_path(l), &[e[0] << 1]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edge_oneway(e[0]), 0);
    assert_eq!(runs(&a, l)[1], vec![e[0] << 1 | 1]);
    assert!(a.oneway_effects(e[0], 0).is_empty());
}

#[test]
fn a_line_with_no_other_way_refuses_it_naming_the_line() {
    let (mut a, n, e) = triangle();
    let l = line(&mut a.w, &[n[0], n[1]], [8.0, 4.0, 2.0]);
    // take the way round away
    let mut cut = a.w.net.edge_data(e[1]).unwrap();
    cut.built = false;
    a.w.apply(Op::DeleteEdge { id: e[1] }).unwrap();
    let _ = &mut cut;
    assert_eq!(a.oneway_effects(e[0], 1), vec![l as f64, 0.0]);
    assert_eq!(a.set_edge_oneway(e[0], 1), 1);
    assert_eq!(a.issue_kinds(), format!("OneWayNoWay:{l}"));
    assert_eq!(a.edge_oneway(e[0]), 0, "nothing changed");
}

#[test]
fn deleting_the_track_a_return_run_uses_breaks_the_line() {
    let (mut a, n, e) = triangle();
    let l = line(&mut a.w, &[n[0], n[1]], [8.0, 4.0, 2.0]);
    assert_eq!(a.set_edge_oneway(e[0], 1), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[l as usize]);
    a.w.apply(Op::DeleteEdge { id: e[2] }).unwrap();
    assert!(a.w.net.line_broken[l as usize], "its way back is gone");
}

#[test]
fn one_way_single_track_has_double_tracks_capacity_per_direction() {
    let (mut a, n, e) = triangle();
    let l = line(&mut a.w, &[n[0], n[1], n[2], n[0]], [8.0, 4.0, 2.0]);
    let mut d = a.w.net.line_data(l).unwrap();
    d.one_way = true;
    a.w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    assert_eq!(a.w.net.line_loop(l), LoopKind::Circle);
    let singles = |a: &TrackApi| a.w.svc.cap.res.iter().filter(|r| r.kind == ResKind::SingleTrack).count();
    assert!(singles(&a) > 0, "shared single track");
    // the loop goes A to B to C to A: each edge as drawn
    for &x in &e {
        assert_eq!(a.set_edge_oneway(x, 1), 0, "{}", a.issues());
    }
    assert_eq!(singles(&a), 0, "{:?}", a.w.svc.cap.res.iter().map(|r| r.kind).collect::<Vec<_>>());
    assert!(a.w.svc.cap.res.iter().any(|r| r.kind == ResKind::Section));
    assert!(a.w.svc.lines[l as usize].running);
    // against the loop's way it has no other way
    assert_eq!(a.set_edge_oneway(e[0], 2), 1);
    assert_eq!(a.issue_kinds(), format!("OneWayNoWay:{l}"));
    assert_eq!(a.edge_oneway(e[0]), 1);
}

#[test]
fn the_flag_survives_a_save_and_a_split() {
    let (mut a, _, e) = triangle();
    assert_eq!(a.set_edge_oneway(e[1], 2), 0, "{}", a.issues());
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.oneway_edges(), vec![e[1], 2]);
    // a station in the middle of it: both halves keep the way
    let (x, y) = (5000.0, 1500.0);
    assert_eq!(b.add_station(2, e[1], x, y, 200, "Mid".into()), 0, "{}", b.issues());
    let flagged: Vec<u32> = b.oneway_edges();
    assert_eq!(flagged.len(), 4, "{flagged:?}");
    assert!(flagged.chunks(2).all(|c| c[1] == 2));
}
