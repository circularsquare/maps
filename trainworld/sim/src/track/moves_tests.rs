//! T-4544 (T-4511 stage 2): a line's moved departures, applied wherever its timetable is read.

use super::hours_tests::corridor;
use super::moves::{Move, MoveKind};
use super::params::*;
use super::service::{lattice_deps, lattice_deps_moved, Trip};
use super::world::{Op, TrackWorld};

const DAY: f64 = 86400.0;

fn trips(w: &TrackWorld, l: usize, r: usize, a: f64, b: f64) -> Vec<Trip> {
    let mut v = vec![];
    w.svc.run_deps(&w.net, l, r, a, b, &mut v);
    v.sort_by(|x, y| x.dep.total_cmp(&y.dep));
    v
}

fn own(run: u8, at: f64, secs: f32) -> Move {
    Move { run, at, secs, kind: MoveKind::Own }
}

/// One departure moves, as far as it is drawn and ends; its neighbours, the other direction and every other field
/// of it stay; any window agrees; with no moves, or moves no departure is near, every trip is as before.
#[test]
fn t4544_one_departure_moves_and_nothing_else() {
    let (mut w, _, c) = corridor(3);
    let (a, b) = (2.0 * DAY, 3.0 * DAY);
    let base: Vec<Vec<Trip>> = (0..2).map(|r| trips(&w, c, r, a, b)).collect();
    assert!(base[0].len() > 20 && base[1].len() > 20);
    let k = base[0].len() / 2;
    let target = base[0][k].dep;

    // a move for a day outside the window moves no trip in it, to the bit
    w.svc.set_moves(c, vec![own(0, a - 5.0 * DAY, 60.0)]);
    for r in 0..2 {
        assert_eq!(format!("{:?}", trips(&w, c, r, a, b)), format!("{:?}", base[r]), "a waiting move moves no one");
    }

    // (two moves for one departure leave the nearer)
    w.svc.set_moves(c, vec![own(0, target + 10.0, 500.0), own(0, target, 120.0)]);
    let got = trips(&w, c, 0, a, b);
    assert_eq!(got.len(), base[0].len());
    for (i, (g, o)) in got.iter().zip(&base[0]).enumerate() {
        if i == k {
            assert_eq!((g.dep, g.level, g.var), (o.dep, o.level, o.var), "the pattern's departure and its timetable stay");
            assert_eq!(g.draw, o.draw + 120.0);
            // (T-4545: moved later it stands less at the far end, down to a minute: its turnaround here is 3 min, so it is ready when it was)
            assert_eq!((g.moved, g.ready), (120.0, o.end));
            assert!((g.end - o.end).abs() < 1e-6, "{} {}", g.end, o.end);
        } else {
            assert_eq!(format!("{g:?}"), format!("{o:?}"), "departure {i}");
        }
    }
    assert_eq!(format!("{:?}", trips(&w, c, 1, a, b)), format!("{:?}", base[1]), "the other direction");

    // any window agrees
    let all = trips(&w, c, 0, a, b);
    for (lo, hi) in [(a, target + 1.0), (target, b), (target - 3.0 * 3600.0, target + 3.0 * 3600.0), (target + 1.0, b)] {
        let want: Vec<String> = all.iter().filter(|t| t.dep >= lo && t.dep < hi).map(|t| format!("{t:?}")).collect();
        let have: Vec<String> = trips(&w, c, 0, lo, hi).iter().map(|t| format!("{t:?}")).collect();
        assert_eq!(have, want, "window {lo}..{hi}");
    }

    // a time moved out of or into a window leaves or enters it
    let lat = w.svc.lattice(&w.net, c).unwrap();
    let mv = w.svc.moves_of(c);
    let mut plain = vec![];
    lattice_deps(&lat, 0, a, b, &mut plain);
    let d = plain.iter().map(|x| x.0).find(|&d| (d - target).abs() < 1.0).unwrap();
    let (mut before, mut after) = (vec![], vec![]);
    lattice_deps_moved(&lat, &mv, 0, d - 600.0, d + 60.0, &mut before);
    lattice_deps_moved(&lat, &mv, 0, d + 60.0, d + 600.0, &mut after);
    assert!(before.iter().all(|x| (x.0 - (d + 120.0)).abs() > 1e-6), "moved out of the window");
    assert!(after.iter().any(|x| (x.0 - (d + 120.0)).abs() < 1e-6), "moved into the window");

    // no moves again: to the bit
    w.svc.set_moves(c, vec![]);
    for r in 0..2 {
        assert_eq!(format!("{:?}", trips(&w, c, r, a, b)), format!("{:?}", base[r]));
    }
}

/// An edit that changes the pattern leaves the era the trains running before it carry on in with the moves they ran.
#[test]
fn t4544_eras_keep_their_moves() {
    let (mut w, _, c) = corridor(3);
    w.svc.set_moves(c, vec![own(0, 2.0 * DAY + 9.0 * 3600.0, 90.0)]);
    let now = 2.0 * DAY + 8.0 * 3600.0;
    let before = w.svc.before_edit(&w.net, now);
    let tph = slots_to_rows(w.net.line_tph[c]);
    let t = w.net.trains_of(c).unwrap();
    w.apply(Op::TrainCounts { id: c as u32, trains: Some([t[0] + 1, t[1], t[2]]), tph }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    let eras = w.svc.eras_of(c);
    assert_eq!(eras.len(), 1);
    assert_eq!(&*eras[0].moves, &*w.svc.moves_of(c));
    assert_eq!(eras[0].moves.len(), 1);
}
