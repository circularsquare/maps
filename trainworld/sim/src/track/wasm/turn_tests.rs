//! T-3021: turning a line's end round a loop of track (`stop_tiers` tiers 7 and 8, `turn_end`,
//! `open_loop`).

use super::*;
use crate::track::tests::{edge, node, station, terminals_at_dead_ends, world};

fn api(w: TrackWorld) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.w = w;
    a
}

/// A stem A - B - M - J along x and a loop at J through K (`balloon_fixture`'s at half size, about
/// 7 km round), leaving J on its east side both ways. With `west`, the same loop mirrored at Jw
/// past A, with no station on it.
pub(super) fn teardrop(west: bool) -> (TrackWorld, [u32; 5]) {
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
    if west {
        let jw = node(&mut w, -5000.0, 0.0, 0);
        let x = node(&mut w, -6000.0, 750.0, 0);
        edge(&mut w, jw, a, &[], 0).unwrap();
        edge(&mut w, jw, x, &[(-5200.0, 0.0)], 0).unwrap();
        edge(&mut w, x, jw, &[(-6800.0, 1500.0), (-7500.0, -750.0), (-5400.0, 0.0)], 0).unwrap();
    }
    for n in [a, b, m, k] {
        station(&mut w, n, 200);
    }
    terminals_at_dead_ends(&mut w);
    (w, [a, b, m, j, k])
}

pub(super) fn tier_of(a: &TrackApi, line: u32, n: u32) -> u8 {
    a.stop_tiers(line).chunks_exact(3).find(|c| c[0] == n as f64).map_or(99, |c| c[1] as u8)
}

#[test]
fn a_loop_at_the_end_turns_the_line_one_way_round_it() {
    let (w, [a0, b, m, _, k]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    // the stop before the branch offers it; the first stop no longer offers a balloon back past B
    // and M (a closing that turns back at A)
    assert_eq!(tier_of(&a, id, m), 7);
    assert_eq!(tier_of(&a, id, a0), 0);
    assert_eq!((tier_of(&a, id, b), tier_of(&a, id, k)), (0, 0));
    let before = a.w.svc.lines[id as usize].round_trip[0];
    assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
    assert_eq!((a.line_loop(id), a.line_one_way(id)), (2, true));
    assert!(a.line_reversals(id).is_empty());
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.round_trip[0] <= before + 120.0, "{} against {before}", ls.round_trip[0]);
    // only run 0 runs: the stem both ways, the loop one way
    let cap = &a.w.svc.cap;
    let runs: Vec<u8> = (0..cap.res.len()).flat_map(|r| cap.users_of(r)).filter(|u| u.line == id).map(|u| u.run).collect();
    assert!(!runs.is_empty() && runs.iter().all(|&r| r == 0), "{runs:?}");
    // the same stop now opens it again; the first stop's marker opens it the same way
    assert_eq!(tier_of(&a, id, m), 8);
    assert_eq!(tier_of(&a, id, a0), 6);
    assert_ne!(a.turn_end(id, b), 0, "no turn at B");
    // undo, redo, and opening it
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!((a.w.net.line_stops(id), a.line_one_way(id)), (&[a0, b, m, k][..], false));
    assert_eq!(a.redo(), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
    assert_eq!(a.open_loop(id), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k]);
    assert_eq!(a.line_loop(id), 0);
}

#[test]
fn a_new_stop_on_a_turned_stem_is_served_both_ways() {
    let (w, [a0, b, m, _, k]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, m, k, m, a0]);
    // B, on the stem, goes in on both passes
    let s: Vec<u32> = a.insert_stop(id, b).iter().map(|&x| x as u32).collect();
    assert_eq!(s, vec![a0, b, m, k, m, b, a0]);
    assert_eq!(a.set_stops(id, &s), 0, "{}", a.issues());
    assert!(a.insert_stop(id, b).is_empty(), "B is a stop already");
    // K, on the loop, is passed once: taken out and put back, it is one visit
    assert_eq!(a.set_stops(id, &[a0, b, m, b, a0]), 0, "{}", a.issues());
    let s: Vec<u32> = a.insert_stop(id, k).iter().map(|&x| x as u32).collect();
    assert_eq!(s, vec![a0, b, m, k, b, a0]);
}

#[test]
fn a_loop_past_the_end_stop_turns_at_the_end_stop() {
    // the line ends at M; the loop past it has K on it, but the line does not stop there
    let (w, [a0, b, m, _, _]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(tier_of(&a, id, m), 7);
    assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, b, a0]);
    assert_eq!(a.line_loop(id), 2);
    assert!(a.line_reversals(id).is_empty(), "round the loop, not back at M");
}

#[test]
fn a_long_loop_is_not_offered() {
    // `long_balloon`'s loop is about 40 km: reversing at K is far quicker
    let mut w = world();
    let a0 = node(&mut w, -3000.0, 0.0, 0);
    let m = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let k = node(&mut w, 5000.0, 1500.0, 0);
    edge(&mut w, a0, m, &[], 0).unwrap();
    edge(&mut w, m, j, &[], 0).unwrap();
    edge(&mut w, j, k, &[(3400.0, 0.0)], 0).unwrap();
    edge(&mut w, k, j, &[(13000.0, 9000.0), (16000.0, -6000.0), (3800.0, 0.0)], 0).unwrap();
    for n in [a0, m, k] {
        station(&mut w, n, 200);
    }
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(tier_of(&a, id, m), 0);
    assert_ne!(a.turn_end(id, m), 0);
}

#[test]
fn both_ends_turn() {
    let (w, [a0, b, m, _, k]) = teardrop(true);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    // the west loop is past A, so A is the stop before it; the east one's is M
    assert_eq!((tier_of(&a, id, m), tier_of(&a, id, a0)), (7, 7));
    assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
    // the closing hop carries on round the west loop: both ends at once, a circle
    let stops = a.w.net.line_stops(id).to_vec();
    assert_eq!(a.line_loop(id), 1, "{stops:?}");
    assert!(a.line_reversals(id).is_empty());
    assert_eq!(stops.iter().filter(|&&s| s == b).count(), 2, "{stops:?}");
    assert_eq!(a.undo(), 0);
    // turned at the west end first, then the east
    assert_eq!(a.turn_end(id, a0), 0, "{}", a.issues());
    let one = a.line_loop(id);
    assert!(one == 1 || tier_of(&a, id, m) == 7, "east still to turn: loop {one}");
    if one == 2 {
        assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
        assert_eq!(a.line_loop(id), 1);
    }
    assert!(a.line_reversals(id).is_empty());
}

#[test]
fn a_balloon_turns_at_its_first_stop_too() {
    // a balloon made before the west loop was built: the west end is offered on its own
    let (w, [a0, b, m, _, k]) = teardrop(true);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let mut d = a.w.net.line_data(id).unwrap();
    d.stops = vec![a0, b, m, k, m, b, a0];
    d.path = route_plain_close(&a.w.net, &d.stops, &mut HopMemo::new()).unwrap();
    d.one_way = true;
    a.w.apply(Op::Line { id, data: Some(d) }).unwrap();
    assert_eq!(a.line_loop(id), 2);
    let t = turns(&a.w.net, id, &mut HopMemo::new());
    assert_eq!(t.len(), 1, "{t:?}");
    assert_eq!(tier_of(&a, id, t[0].branch), 7);
    assert_eq!(a.turn_end(id, t[0].branch), 0, "{}", a.issues());
    assert_eq!(a.line_loop(id), 1);
    assert!(a.line_reversals(id).is_empty());
}
