//! T-3122: a loop line stops at every pass of a station it stops at (`every_pass`), and which way each
//! stop is served (`line_stop_ways`). T-3140: a click never turns a line round a loop; the right click
//! (`turn_end`) does. T-3141: a one-way circle with one branch ends there (`Network::branch_end`).

use super::*;
use crate::track::tests::{edge, node, ring_fixture, station, terminals_at_dead_ends, world};

fn api(w: TrackWorld) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.w = w;
    a
}

/// A stem A - B - M - J along x and a loop at J through K, as `turn_tests::teardrop`; with `west`, a
/// loop at Jw past A too, through X, a station when `x_stop`. `[a, b, m, k, x]` (`x` the loop's node).
fn barbell(west: bool, x_stop: bool) -> (TrackWorld, [u32; 5]) {
    let mut w = world();
    let a = node(&mut w, -4000.0, 0.0, 0);
    let b = node(&mut w, -2000.0, 0.0, 0);
    let m = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 1000.0, 0.0, 0);
    let k = node(&mut w, 2000.0, 750.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    edge(&mut w, b, m, &[], 0).unwrap();
    edge(&mut w, m, j, &[], 0).unwrap();
    edge(&mut w, j, k, &[(1200.0, 0.0)], 0).unwrap();
    edge(&mut w, k, j, &[(2800.0, 1500.0), (3500.0, -750.0), (1400.0, 0.0)], 0).unwrap();
    let mut x = u32::MAX;
    if west {
        let jw = node(&mut w, -5000.0, 0.0, 0);
        x = node(&mut w, -6000.0, 750.0, 0);
        edge(&mut w, jw, a, &[], 0).unwrap();
        edge(&mut w, jw, x, &[(-5200.0, 0.0)], 0).unwrap();
        edge(&mut w, x, jw, &[(-6800.0, 1500.0), (-7500.0, -750.0), (-5400.0, 0.0)], 0).unwrap();
        if x_stop {
            station(&mut w, x, 200);
        }
    }
    for n in [a, b, m, k] {
        station(&mut w, n, 200);
    }
    terminals_at_dead_ends(&mut w);
    (w, [a, b, m, k, x])
}

/// A new line from clicks on `order`, each as the line tool sends it.
fn clicked(a: &mut TrackApi, order: &[u32]) -> u32 {
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &order[..1], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    for &s in &order[1..] {
        assert_eq!(a.add_stop(id, s), 0, "{}", a.issues());
    }
    id
}

/// How many times the line stops at `s` in a lap or a run.
fn visits(a: &TrackApi, id: u32, s: u32) -> usize {
    let st = a.w.net.line_stops(id);
    let c = if closed(st) { &st[..st.len() - 1] } else { st };
    c.iter().filter(|&&x| x == s).count()
}

fn one_way_at(a: &TrackApi, id: u32) -> Vec<u32> {
    let mut v: Vec<u32> = a.line_stop_ways(id).chunks_exact(2).map(|c| c[0] as u32).collect();
    v.sort_unstable();
    v
}

/// Right clicks on every end the line offers to turn round a loop (`turn_end`), first offered first.
fn turn_all(a: &mut TrackApi, id: u32) {
    for _ in 0..2 {
        let Some(t) = turns(&a.w.net, id, &mut HopMemo::new()).into_iter().next() else { return };
        assert_eq!(a.turn_end(id, t.branch), 0, "{}", a.issues());
    }
}

#[test]
fn a_click_never_turns_a_line_round_a_loop() {
    // T-3140: stations clicked once each, ends on loops included, make an open line run both ways
    let (w, [a0, b, m, k, x]) = barbell(true, true);
    let mut a = api(w);
    for order in [[x, a0, b, m, k], [k, m, b, a0, x], [k, x, a0, b, m], [b, m, k, a0, x]] {
        let id = clicked(&mut a, &order);
        let st = a.w.net.line_stops(id);
        assert_eq!((a.line_loop(id), a.line_one_way(id)), (0, false), "{order:?}: {st:?}");
        assert_eq!(st.len(), 5, "{order:?}: every station once, {st:?}");
        assert!(one_way_at(&a, id).is_empty());
        assert_eq!(a.remove_line(id), 0);
    }
}

#[test]
fn a_barbell_turned_at_both_ends_runs_round_both_loops_stopping_at_every_pass() {
    let (w, [a0, b, m, k, x]) = barbell(true, true);
    let mut a = api(w);
    let mut want = vec![k, x];
    want.sort_unstable();
    let check = |a: &TrackApi, id: u32, what: &str| {
        assert_eq!((a.line_loop(id), a.line_one_way(id)), (1, true), "{what}: {:?}", a.w.net.line_stops(id));
        assert!(a.line_reversals(id).is_empty());
        for s in [a0, b, m] {
            assert_eq!(visits(a, id, s), 2, "{what}: the stem both ways, {:?}", a.w.net.line_stops(id));
        }
        assert_eq!((visits(a, id, k), visits(a, id, x)), (1, 1));
        assert_eq!(one_way_at(a, id), want, "{what}: the loops' stops one way, the stem's both");
    };
    // clicked in any order, then both ends right clicked
    for order in [[x, a0, b, m, k], [k, m, b, a0, x], [k, x, a0, b, m], [b, m, k, a0, x]] {
        let id = clicked(&mut a, &order);
        turn_all(&mut a, id);
        check(&a, id, &format!("{order:?}"));
        assert_eq!(a.remove_line(id), 0);
    }
    // a loop turned while the stem is short, the stem then clicked on, the other loop turned last
    let id = clicked(&mut a, &[k, m]);
    turn_all(&mut a, id);
    assert_eq!(a.line_loop(id), 2, "{:?}", a.w.net.line_stops(id));
    for s in [b, a0, x] {
        assert_eq!(a.add_stop(id, s), 0, "{}", a.issues());
    }
    turn_all(&mut a, id);
    check(&a, id, "east loop first");
}

#[test]
fn a_stem_and_one_loop_turned_is_a_one_way_balloon() {
    let (w, [a0, b, m, k, _]) = barbell(false, false);
    let mut a = api(w);
    for order in [[a0, b, m, k], [k, m, b, a0], [m, k, a0, b]] {
        let id = clicked(&mut a, &order);
        assert_eq!(a.line_loop(id), 0);
        turn_all(&mut a, id);
        let st = a.w.net.line_stops(id);
        assert_eq!((a.line_loop(id), a.line_one_way(id)), (2, true), "{order:?}: {st:?}");
        assert_eq!(st.len(), 7, "{order:?}: {st:?}");
        assert_eq!(visits(&a, id, k), 1);
        for s in [b, m] {
            assert_eq!(visits(&a, id, s), 2, "{order:?}: {st:?}");
        }
        assert_eq!(one_way_at(&a, id), vec![k]);
        assert_eq!(a.remove_line(id), 0);
    }
}

#[test]
fn a_loop_with_no_stop_of_its_own_stops_at_its_station_once() {
    // the loop past A has no station: A's pass before it and after it would be two stops in a row
    let (w, [a0, b, m, k, _]) = barbell(true, false);
    let mut a = api(w);
    let id = clicked(&mut a, &[a0, b, m, k]);
    turn_all(&mut a, id);
    assert_eq!(a.line_one_way(id), true);
    assert_eq!(visits(&a, id, a0), 1, "{:?}", a.w.net.line_stops(id));
    assert_eq!((visits(&a, id, b), visits(&a, id, m)), (2, 2));
    // stopped at once, but its trains pass both ways, round the loop or turning back there: a terminal
    assert!(!one_way_at(&a, id).contains(&a0), "{:?}", a.w.net.line_stops(id));
    assert_eq!(one_way_at(&a, id), vec![k]);
}

#[test]
fn a_ring_clicked_once_each_stays_open_until_closed_and_then_stops_once_each() {
    let (w, ns) = ring_fixture();
    let mut a = api(w);
    let id = clicked(&mut a, &ns);
    assert_eq!(a.w.net.line_stops(id), &ns[..], "no end stands on a loop that comes back onto the line");
    assert_eq!(a.line_loop(id), 0);
    let mut closed_stops = ns.clone();
    closed_stops.push(ns[0]);
    assert_eq!(a.set_stops(id, &closed_stops), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &closed_stops[..], "every station once");
    assert_eq!(a.line_loop(id), 1);
    assert!(one_way_at(&a, id).is_empty(), "a circle run both ways serves every stop both ways");
    assert_eq!(a.set_line_one_way(id, true), 0);
    let mut all = ns.clone();
    all.sort_unstable();
    assert_eq!(one_way_at(&a, id), all, "run one way, every stop one way");
}

#[test]
fn a_loop_whose_way_back_runs_past_its_stops_stops_there_too() {
    // a circle closing from K straight back to A passes B and M on its way: they become stops there
    let (w, [a0, b, m, k, _]) = barbell(false, false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.set_stops(id, &[a0, b, m, k, a0]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
}

#[test]
fn a_save_whose_loop_passes_its_stops_loads_stopping_there() {
    let (w, [a0, b, m, k, _]) = barbell(false, false);
    let mut a = api(w);
    let id = clicked(&mut a, &[a0, b, m, k]);
    turn_all(&mut a, id);
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
    // as a save from before the rule could hold it: the same path, the way back past M and B
    let mut d = a.w.net.line_data(id).unwrap();
    d.stops = vec![a0, b, m, k, a0];
    a.w.net.set_line(id, Some(d)).unwrap();
    let bytes = a.save();
    let mut c = api(world());
    assert!(c.load(&bytes));
    assert_eq!(c.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
    assert!(c.w.svc.lines[id as usize].ok);
}

#[test]
fn a_heading_points_the_way_the_trains_go() {
    let (w, [a0, b, m, k, _]) = barbell(false, false);
    let mut a = api(w);
    let id = clicked(&mut a, &[a0, b, m, k]);
    turn_all(&mut a, id);
    let ways = a.line_stop_ways(id);
    assert_eq!(ways.len(), 2);
    let (n, h) = (ways[0] as u32, ways[1]);
    assert_eq!(n, k);
    // the way the path leaves K: toward the leaving edge's first bend
    let net = &a.w.net;
    let path = net.line_path(id);
    let i = nodes_of(net, path).iter().position(|&x| x == k).unwrap();
    let p = path[i];
    let pts = net.edge_vertices(p >> 1);
    let next = if p & 1 == 0 { pts[1] } else { pts[pts.len() - 2] };
    let (px, py) = (next.x - net.node_x[k as usize], next.y - net.node_y[k as usize]);
    assert!(h.cos() * px + h.sin() * py > 0.0, "heading {h} against the way on {px},{py}");
    // T-3139: along the track's tangent there, which points at the edge's first bend
    assert!((h.cos() * py - h.sin() * px).abs() / px.hypot(py) < 0.05, "heading {h} off the tangent {px},{py}");
}

/// `barbell(false, false)`'s track with a circle round the loop that runs out along the stem to B and
/// back (stops K, B, K), applied as a save could hold it, `one_way` as given.
fn branch_circle(one_way: bool) -> (TrackApi, u32, [u32; 4]) {
    let mut w = world();
    let a0 = node(&mut w, -4000.0, 0.0, 0);
    let b = node(&mut w, -2000.0, 0.0, 0);
    let m = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 1000.0, 0.0, 0);
    let k = node(&mut w, 2000.0, 750.0, 0);
    edge(&mut w, a0, b, &[], 0).unwrap();
    let bm = edge(&mut w, b, m, &[], 0).unwrap();
    let mj = edge(&mut w, m, j, &[], 0).unwrap();
    let jk = edge(&mut w, j, k, &[(1200.0, 0.0)], 0).unwrap();
    let kj = edge(&mut w, k, j, &[(2800.0, 1500.0), (3500.0, -750.0), (1400.0, 0.0)], 0).unwrap();
    for n in [a0, b, m, k] {
        station(&mut w, n, 200);
    }
    terminals_at_dead_ends(&mut w);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[k, b], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let mut d = a.w.net.line_data(id).unwrap();
    d.stops = vec![k, b, k];
    d.path = vec![kj << 1, mj << 1 | 1, bm << 1 | 1, bm << 1, mj << 1, jk << 1];
    d.one_way = one_way;
    a.w.apply(Op::Line { id, data: Some(d) }).unwrap();
    (a, id, [b, k, bm, mj])
}

#[test]
fn a_one_way_circle_with_one_branch_ends_at_the_branch() {
    // T-3141: the circle starts at B, a balloon turning at B as at an end, no reversal partway
    let (a, id, [b, k, bm, mj]) = branch_circle(true);
    assert_eq!(a.w.net.line_stops(id), &[b, k, b]);
    assert_eq!(a.w.net.line_path(id)[..2], [bm << 1, mj << 1]);
    assert_eq!(a.line_loop(id), 2);
    assert!(a.line_reversals(id).is_empty(), "{:?}", a.line_reversals(id));
    assert!(a.w.svc.lines[id as usize].ok);
}

#[test]
fn a_circle_with_one_branch_run_both_ways_keeps_its_reversal_until_set_one_way() {
    let (mut a, id, [b, k, ..]) = branch_circle(false);
    assert_eq!(a.w.net.line_stops(id), &[k, b, k]);
    assert_eq!(a.line_loop(id), 1);
    assert_eq!(a.line_reversals(id).len(), 1);
    assert_eq!(a.set_line_one_way(id, true), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[b, k, b]);
    assert_eq!(a.line_loop(id), 2);
    assert!(a.line_reversals(id).is_empty());
}
