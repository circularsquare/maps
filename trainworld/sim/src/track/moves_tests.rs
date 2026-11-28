//! T-4544 (T-4511 stage 2): a line's moved departures, applied wherever its timetable is read.

use super::hours_tests::corridor;
use super::moves::{self, Move, MoveKind};
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
    Move { run, at, secs, kind: MoveKind::Own, row: 0 }
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

    // the key of a departure: its time of day plus the line's lead (T-4547), so the move holds on every day
    let lat0 = w.svc.lattice(&w.net, c).unwrap();
    let key = |d: f64| moves::key_of(d, lat0.shift + lat0.lead);
    // a move whose key no departure is within half a headway of waits: it moves no trip, to the bit
    w.svc.set_moves(c, vec![own(0, key(target + 0.5 * base[0][k].hw), 60.0)]);
    for r in 0..2 {
        assert_eq!(format!("{:?}", trips(&w, c, r, a, b)), format!("{:?}", base[r]), "a waiting move moves no one");
    }

    // (two moves for one departure leave the nearer)
    w.svc.set_moves(c, vec![own(0, key(target + 10.0), 500.0), own(0, key(target), 120.0)]);
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
    w.svc.set_moves(c, vec![own(0, 9.0 * 3600.0, 90.0)]);
    let now = 2.0 * DAY + 8.0 * 3600.0;
    let before = w.svc.before_edit(&w.net, now);
    let tph = slots_to_rows(w.net.line_tph[c]);
    let t = w.net.trains_of(c).unwrap();
    w.apply(Op::TrainCounts { id: c as u32, trains: Some([t[0] + 1, t[1], t[2]]), tph }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    let eras = w.svc.eras_of(c);
    assert_eq!(eras.len(), 1);
    // (the era holds what ran before the edit; the line's own are clamped to the room it leaves)
    assert_eq!((eras[0].moves.len(), eras[0].moves[0].secs), (1, 90.0));
    assert_eq!(&*w.svc.stored_of(c), &*eras[0].moves);
}

/// T-4549: a move belongs to one kind of day: a weekday move is absent on a weekend day, a weekend move is absent on weekdays.
#[test]
fn t4549_a_move_runs_on_its_kind_of_day_only() {
    let (mut w, _, c) = corridor(3);
    let (wk, we) = (2.0 * DAY, 6.0 * DAY);
    assert_eq!((row_of_day(2), row_of_day(6)), (WEEKDAYS, WEEKEND));
    let base_wk = trips(&w, c, 0, wk, wk + DAY);
    let base_we = trips(&w, c, 0, we, we + DAY);
    let target = base_wk[base_wk.len() / 2].dep;
    let lat = w.svc.lattice(&w.net, c).unwrap();
    let key = moves::key_of(target, lat.shift + lat.lead);
    let at_day = |v: &[Trip], day: f64| v.iter().find(|t| (t.dep - (target + (day - 2.0) * DAY)).abs() < 1.0).map(|t| t.moved);

    w.svc.set_moves(c, vec![own(0, key, 120.0)]);
    let (a, b) = (trips(&w, c, 0, wk, wk + DAY), trips(&w, c, 0, we, we + DAY));
    assert_eq!((at_day(&a, 2.0), at_day(&b, 6.0)), (Some(120.0), Some(0.0)), "a weekday move");
    assert_eq!(format!("{:?}", b), format!("{:?}", base_we), "the weekend as it was");

    w.svc.set_moves(c, vec![Move { row: WEEKEND as u8, ..own(0, key, 60.0) }]);
    let (a, b) = (trips(&w, c, 0, wk, wk + DAY), trips(&w, c, 0, we, we + DAY));
    assert_eq!((at_day(&a, 2.0), at_day(&b, 6.0)), (Some(0.0), Some(60.0)), "a weekend move");
    assert_eq!(format!("{:?}", a), format!("{:?}", base_wk), "the weekdays as they were");

    // the boarding account's reference day of a kind agrees
    let mut v = vec![];
    w.svc.run_deps_as(&w.net, c, 0, we, we + DAY, Some(WEEKDAYS), &mut v);
    assert!(v.iter().all(|t| t.moved == 0.0), "a weekday row has no weekend move");
}
