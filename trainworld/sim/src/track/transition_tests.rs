//! T-1071: a change to a line's service moves no train. Every train drawn is placed on the ground
//! every couple of seconds around an edit; each must be where the one before it could have got to,
//! and a train may come into or out of service only standing at a terminal.

use super::params::*;
use super::tests::{balloon_fixture, edge, line, node, reversal_fixture, ring_fixture, station, world};
use super::world::{Op, TrackWorld};

/// A drawn train: line, ground position, and whether it is standing at a terminal (at its trip's
/// start or past its run's end), where it may come into or leave service.
#[derive(Clone, Copy, Debug)]
struct Place {
    line: u32,
    x: f64,
    y: f64,
    terminal: bool,
    /// The trip drawn (for messages): run, departure, drawn from, until, seconds into it.
    trip: (u8, f64, f64, f64, f64),
}

/// Every train drawn at `t`.
fn places(w: &TrackWorld, t: f64) -> Vec<Place> {
    let mut out = vec![];
    for x in w.svc.trips(&w.net, t, t + 1e-3) {
        if x.draw > t || x.end < t {
            continue;
        }
        let p = &w.svc.lines[x.line as usize].runs[x.run as usize].prof[x.level as usize];
        let tau = (t - x.draw).min(p.duration);
        let (s, v) = super::profile::state_at(&p.phases, tau);
        let (px, py) = w.svc.position(&w.net, x.line, x.run as usize, s);
        // a train that left its terminal within the last step counts as there
        let terminal = tau <= STEP_S + 1e-6 || (v < 0.01 && tau >= p.duration - 1e-6);
        out.push(Place { line: x.line, x: px, y: py, terminal, trip: (x.run, x.dep, x.draw, x.end, t - x.draw) });
    }
    out
}

/// The trains of `a` that no matching of `a` to `b` (same line, within `reach` metres) can pair
/// when it must pair every train of `a` not at a terminal (Kuhn's augmenting paths).
fn unpaired(a: &[Place], b: &[Place], reach: f64) -> Vec<Place> {
    let adj: Vec<Vec<usize>> = a
        .iter()
        .map(|p| (0..b.len()).filter(|&j| b[j].line == p.line && (p.x - b[j].x).hypot(p.y - b[j].y) <= reach).collect())
        .collect();
    fn augment(i: usize, adj: &[Vec<usize>], seen: &mut [bool], to: &mut [Option<usize>]) -> bool {
        for &j in &adj[i] {
            if !seen[j] {
                seen[j] = true;
                if to[j].map_or(true, |k| augment(k, adj, seen, to)) {
                    to[j] = Some(i);
                    return true;
                }
            }
        }
        false
    }
    let mut to = vec![None; b.len()];
    let mut out = vec![];
    for i in (0..a.len()).filter(|&i| !a[i].terminal) {
        if !augment(i, &adj, &mut vec![false; b.len()], &mut to) {
            out.push(a[i]);
        }
    }
    out
}

/// Trains of `a` and of `b` (not at a terminal) that cannot be paired with one of the other within
/// `reach`: gone and new. A matching that pairs every such train on both sides exists exactly when
/// each side's can be paired on its own (Mendelsohn-Dulmage).
fn pair(a: &[Place], b: &[Place], reach: f64) -> (Vec<Place>, Vec<Place>) {
    (unpaired(a, b, reach), unpaired(b, a, reach))
}

const STEP_S: f64 = 2.0;

/// Trains from `t0` to `t1` in steps: each step's trains where the step before's could have got to
/// (top speed times the step, plus a metre), any other only standing at a terminal. Old patterns
/// are pruned from `edit` on, as the clock worker does.
fn assert_smooth(w: &mut TrackWorld, t0: f64, t1: f64, edit: f64, what: &str) {
    let reach = 160.0 / 3.6 * STEP_S + 1.0;
    let mut prev = places(w, t0);
    let mut t = t0 + STEP_S;
    while t <= t1 {
        if t >= edit {
            w.svc.prune_eras(&w.net, t);
        }
        let now = places(w, t);
        let (gone, new) = pair(&prev, &now, reach);
        for g in gone.iter().filter(|g| !g.terminal) {
            panic!("{what}: a train vanished mid-line at +{:.0} s: {g:?}", t - t0);
        }
        for n in new.iter().filter(|n| !n.terminal) {
            if std::env::var("T1071_DEBUG").is_ok() {
                for (k, v) in [(t - STEP_S, &prev), (t, &now)] {
                    eprintln!("at {k}:");
                    for p in v.iter().filter(|p| p.line == n.line && (p.x - n.x).hypot(p.y - n.y) < 3000.0) {
                        eprintln!("  {p:?}");
                    }
                }
                for x in w.svc.trips(&w.net, t - 4000.0, t + 1.0).iter().filter(|x| x.line == n.line && x.run == n.trip.0) {
                    eprintln!("  trip {x:?}");
                }
                eprintln!("eras {:?}", w.svc.eras_of(n.line as usize));
            }
            panic!("{what}: a train appeared mid-line at +{:.0} s: {n:?}", t - t0);
        }
        prev = now;
        t += STEP_S;
    }
}

/// Apply `edit` at game time `now` as the clock worker does, and check that no train moves: the
/// same trains at the same places a moment either side of it, then smooth for three round trips.
fn edit_at(w: &mut TrackWorld, now: f64, what: &str, edit: impl FnOnce(&mut TrackWorld)) {
    let a = places(w, now);
    let before = w.svc.before_edit(&w.net, now);
    edit(w);
    w.svc.note_changes(&w.net, &before, now);
    // the edit changed some line's departures (else this tests nothing)
    assert!(w.svc.eras.iter().flatten().any(|e| e.until == now), "{what}: no line's departures changed: {:?}", (0..w.svc.lines.len()).map(|l| w.svc.lattice(&w.net, l)).collect::<Vec<_>>());
    let b = places(w, now);
    // at the instant itself not even a train standing at a terminal comes or goes
    let all = |v: Vec<Place>| v.into_iter().map(|p| Place { terminal: false, ..p }).collect::<Vec<_>>();
    let (gone, new) = pair(&all(a), &all(b), 0.5);
    assert!(gone.is_empty() && new.is_empty(), "{what}: at the edit, gone {gone:?}, new {new:?}");
    let rt = w.svc.lines.iter().filter(|l| l.running).map(|l| l.round_trip[MEDIUM].max(l.round_trip[HIGH])).fold(0.0, f64::max);
    assert_smooth(w, now - 60.0, now + 4.0 * rt + 7200.0, now, what);
}

fn counts(w: &mut TrackWorld, id: u32, trains: [u32; 3]) {
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some(trains), tph }).unwrap();
}

/// Stations 2 km apart along a straight line.
fn corridor(n: usize) -> (TrackWorld, Vec<u32>) {
    let mut w = world();
    let ns: Vec<u32> = (0..n).map(|i| node(&mut w, i as f64 * 2000.0, 0.0, 0)).collect();
    for i in 0..n - 1 {
        edge(&mut w, ns[i], ns[i + 1], &[], 0).unwrap();
    }
    for &n in &ns {
        station(&mut w, n, 200);
    }
    (w, ns)
}

/// Midday on a weekday, at an odd second.
const MIDDAY: f64 = 2.0 * 86400.0 + 11.0 * 3600.0 + 417.0;

#[test]
fn a_count_change_moves_no_train() {
    for (from, to) in [(6u32, 9u32), (9, 6), (6, 7), (7, 6), (2, 1), (1, 3), (8, 0), (0, 5)] {
        let (mut w, ns) = corridor(7);
        let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
        counts(&mut w, id, [from, from, from]);
        edit_at(&mut w, MIDDAY, &format!("{from} to {to}"), |w| counts(w, id, [to, to, to]));
    }
}

#[test]
fn a_line_on_another_local_time_changes_smoothly() {
    // Tokyo, Paris: the line's periods on its own hours (T-309)
    for shift in [14.0 * 3600.0, 6.0 * 3600.0, -3.0 * 3600.0, 5.5 * 3600.0] {
        let (mut w, ns) = corridor(7);
        let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
        w.svc.lines[id as usize].shift = shift;
        counts(&mut w, id, [6, 6, 6]);
        for (k, n) in [9u32, 4, 7].into_iter().enumerate() {
            edit_at(&mut w, MIDDAY + k as f64 * 5000.0, &format!("shift {shift}, to {n}"), |w| counts(w, id, [n, n, n]));
        }
    }
}

#[test]
fn edits_in_quick_succession_move_no_train() {
    // the stepper clicked again and again: each edit lands while the last one's change is going on
    let (mut w, ns) = corridor(7);
    let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
    counts(&mut w, id, [6, 6, 6]);
    let mut now = MIDDAY;
    for n in [7, 8, 9, 10, 4, 3, 8] {
        edit_at(&mut w, now, &format!("to {n}"), |w| counts(w, id, [n, n, n]));
        now += 47.0;
    }
}

#[test]
fn a_change_on_shared_track_moves_no_other_train() {
    // the other line's holds move with this one's count
    let (mut w, ns) = corridor(6);
    let long = line(&mut w, &ns, [8.0, 6.0, 3.0]);
    let short = line(&mut w, &ns[1..5], [8.0, 6.0, 3.0]);
    counts(&mut w, long, [2, 4, 4]);
    counts(&mut w, short, [3, 3, 3]);
    edit_at(&mut w, MIDDAY, "shared, more", |w| counts(w, long, [4, 4, 4]));
    edit_at(&mut w, MIDDAY + 3.0 * 3600.0, "shared, fewer", |w| counts(w, long, [1, 4, 4]));
}

#[test]
fn a_change_on_full_track_moves_no_train() {
    // two lines filling their shared track: one's fleet makes or takes the other's room (T-517)
    let (mut w, ns) = corridor(5);
    let long = line(&mut w, &ns, [8.0, 3.0, 1.0]);
    let short = line(&mut w, &ns[1..4], [8.0, 3.0, 1.0]);
    counts(&mut w, long, [40, 40, 2]);
    counts(&mut w, short, [40, 40, 2]);
    edit_at(&mut w, MIDDAY, "full, fewer", |w| counts(w, long, [3, 40, 2]));
    edit_at(&mut w, MIDDAY + 4.0 * 3600.0, "full, fewer again", |w| counts(w, long, [2, 40, 2]));
}

#[test]
fn a_headway_change_moves_no_train() {
    let (mut w, ns) = corridor(7);
    let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
    counts(&mut w, id, [6, 6, 6]);
    w.apply(Op::AutoTrains { id, auto: Some([300, 600, 900]), trains: Some([6, 6, 6]) }).unwrap();
    edit_at(&mut w, MIDDAY + 4.0 * 3600.0, "shorter headways", |w| w.apply(Op::AutoTrains { id, auto: Some([200, 300, 900]), trains: w.net.line_trains[id as usize] }).map(|_| ()).unwrap());
}

#[test]
fn loops_and_reversals_change_without_moving_a_train() {
    let (mut w, ns) = ring_fixture();
    let mut stops = ns.clone();
    stops.push(ns[0]);
    let c = line(&mut w, &stops, [12.0, 6.0, 3.0]);
    counts(&mut w, c, [4, 8, 2]);
    edit_at(&mut w, MIDDAY, "circle, more", |w| counts(w, c, [8, 8, 2]));
    edit_at(&mut w, MIDDAY + 3.0 * 3600.0, "circle, fewer", |w| counts(w, c, [2, 8, 2]));

    let (mut w, [a, m, _, k]) = balloon_fixture();
    let b = line(&mut w, &[a, m, k, m, a], [12.0, 6.0, 3.0]);
    counts(&mut w, b, [4, 8, 2]);
    edit_at(&mut w, MIDDAY, "balloon, more", |w| counts(w, b, [8, 8, 2]));
    edit_at(&mut w, MIDDAY + 3.0 * 3600.0, "balloon, fewer", |w| counts(w, b, [2, 8, 2]));

    let (mut w, [_, t, _, m, br]) = reversal_fixture();
    let r = line(&mut w, &[br, t, m], [6.0, 3.0, 0.0]);
    counts(&mut w, r, [2, 6, 2]);
    edit_at(&mut w, MIDDAY, "reversing, more", |w| counts(w, r, [3, 6, 2]));
    edit_at(&mut w, MIDDAY + 3.0 * 3600.0, "reversing, fewer", |w| counts(w, r, [1, 6, 2]));
}

#[test]
fn a_game_saved_while_trains_change_over_loads_the_same() {
    // a line sharing track, so profiles shift as well as departures, saved mid-transition
    let (mut w, ns) = corridor(6);
    let long = line(&mut w, &ns, [8.0, 6.0, 3.0]);
    let short = line(&mut w, &ns[1..5], [8.0, 6.0, 3.0]);
    counts(&mut w, long, [2, 4, 4]);
    counts(&mut w, short, [3, 3, 3]);
    let before = w.svc.before_edit(&w.net, MIDDAY);
    counts(&mut w, long, [4, 4, 4]);
    w.svc.note_changes(&w.net, &before, MIDDAY);
    let t = MIDDAY + 600.0;
    w.svc.prune_eras(&w.net, t);
    assert!(!w.svc.eras_of(long as usize).is_empty(), "{:?} {:?}", w.svc.eras, w.svc.lattice(&w.net, long as usize));
    let eras = w.svc.encode_eras(&w.net);
    let mut back = TrackWorld::new(super::save::decode(&super::save::encode(&w.net)).unwrap());
    back.svc.decode_eras(&eras).unwrap();
    assert_eq!(back.svc.eras, w.svc.eras);
    assert_eq!(back.svc.shifts, w.svc.shifts);
    // the same trips, to well under a second: the rates a load settles again agree to about 1e-3 s
    // a departure, summed over those since the level's stretch began (6:00, here five hours back)
    let (a, b) = (back.svc.trips(&back.net, t, t + 3600.0), w.svc.trips(&w.net, t, t + 3600.0));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((x.line, x.run) == (y.line, y.run) && (x.draw - y.draw).abs() < 0.5 && (x.end - y.end).abs() < 0.5, "{x:?} against {y:?}");
    }
    assert_smooth(&mut back, t, t + 3.0 * 3600.0, t, "after loading");
}

#[test]
fn a_change_just_before_a_new_level_moves_no_train() {
    let (mut w, ns) = corridor(7);
    let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
    counts(&mut w, id, [9, 6, 3]);
    // the evening starts at 20:00
    edit_at(&mut w, 2.0 * 86400.0 + 19.9 * 3600.0, "before the evening", |w| counts(w, id, [12, 4, 3]));
}

