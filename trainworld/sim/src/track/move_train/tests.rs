use super::super::hours_tests::corridor;
use super::super::params::HIGH;
use super::super::moves::{MoveKind, Train};
use super::super::overtake::tests::{set_passing, shinkansen};
use super::super::service::Trip;
use super::super::tests::{edge, headway_line, node, set_headway, station, terminals_at_dead_ends, world};
use super::super::moves::{self, Move};
use super::super::params::slots_to_rows;
use super::super::world::{Op, TrackWorld};
use super::Why;

const DAY: f64 = 86400.0;

fn trips(w: &TrackWorld, l: usize, a: f64, b: f64) -> Vec<Trip> {
    w.svc.line_trips(&w.net, l, a, b)
}

/// A mid-morning departure of run 0 of day 2 with neighbours on both sides.
fn pick(w: &TrackWorld, l: usize) -> Trip {
    let v: Vec<Trip> = trips(w, l, 2.0 * DAY + 8.0 * 3600.0, 2.0 * DAY + 11.0 * 3600.0).into_iter().filter(|t| t.run == 0).collect();
    assert!(v.len() >= 3, "{} departures", v.len());
    v[v.len() / 2]
}

/// What the day's trains are as far as their identity goes: each departure, whether its train goes on, and whether a
/// drawn train stands between: the shape a move must leave alone.
fn shape(w: &TrackWorld, l: usize) -> Vec<(u8, u64, bool)> {
    let mut v: Vec<(u8, u64, bool)> = trips(w, l, 2.0 * DAY, 3.0 * DAY).iter().map(|t| (t.run, t.dep.to_bits(), t.next.is_some())).collect();
    v.sort();
    v
}

/// Step a train by `by` until the step is refused; the total reached and the reason.
fn walk(w: &mut TrackWorld, l: u32, t: &Trip, by: f64, steps: &mut usize) -> (f64, Why) {
    for _ in 0..200 {
        match w.move_op(l, t.run, t.dep, by) {
            Ok(Some(op)) => {
                w.apply(op).unwrap();
                *steps += 1;
            }
            Ok(None) => panic!("no step"),
            Err(e) => return (w.move_range(l, t.run, t.dep).unwrap().now, e.why),
        }
    }
    panic!("never refused")
}


fn dump(w: &TrackWorld, l: usize) -> String {
    format!("{:?}", trips(w, l, 2.0 * DAY, 3.0 * DAY))
}

/// A count line's train moves as far as it still makes its next departure (its 3 min turnaround less the 1 min it
/// must stand), each end refusing with its reason; no train enters or leaves service on the way; undone to the bit.
#[test]
fn a_train_on_a_count_line_moves_two_minutes_either_way() {
    let (mut w, _, c) = corridor(3);
    let t = pick(&w, c);
    let r = w.move_range(c as u32, t.run, t.dep).unwrap();
    assert!((r.hi - 120.0).abs() < 1.0 && (r.lo + 120.0).abs() < 1.0, "{r:?}");
    let stops = w.net.line_stops(c as u32).to_vec();
    assert!(matches!(r.hi_why, Why::Leaves { node, .. } if node == *stops.last().unwrap()), "{:?}", r.hi_why);
    assert!(matches!(r.lo_why, Why::Arrives { node, .. } if node == stops[0]), "{:?}", r.lo_why);

    let (base, shape0) = (dump(&w, c), shape(&w, c));
    let mut steps = 0;
    let (up, why) = walk(&mut w, c as u32, &t, 60.0, &mut steps);
    assert!((up - r.hi).abs() < 1e-3 && matches!(why, Why::Leaves { .. }), "{up} {why:?}");
    assert_eq!(shape(&w, c), shape0, "no train enters or leaves service, every train takes its next departure");
    let moved = trips(&w, c, t.draw - 10.0, t.draw + 300.0).into_iter().find(|x| x.run == t.run && x.dep == t.dep).unwrap();
    assert_eq!((moved.draw - t.draw) as f32, up as f32, "drawn that much later");
    assert!(moved.end <= moved.next.unwrap() + 1e-6, "it makes its next departure");
    let (down, why) = walk(&mut w, c as u32, &t, -60.0, &mut steps);
    assert!((down - r.lo).abs() < 1e-3 && matches!(why, Why::Arrives { .. }), "{down} {why:?}");
    assert_eq!(shape(&w, c), shape0);
    let moved = trips(&w, c, t.draw - 400.0, t.draw + 300.0).into_iter().find(|x| x.run == t.run && x.dep == t.dep).unwrap();
    let prev = trips(&w, c, t.draw - 4000.0, t.draw).into_iter().find(|x| x.next == Some(moved.draw)).unwrap();
    assert!(moved.draw - (prev.draw + w.svc.lines[c].prof_of(prev.run as usize, prev.level as usize, 0).duration) >= 59.0, "its train stood a minute at the first stop");
    // a step past the end says why and by how much
    let e = w.move_op(c as u32, t.run, t.dep, -60.0).unwrap_err();
    assert!((e.limit - r.lo).abs() < 1e-3 && e.why == r.lo_why);
    // undone step by step to the bit
    for _ in 0..steps { w.undo().unwrap().unwrap(); }
    assert_eq!(dump(&w, c), base);
    assert!(w.svc.moves_of(c).is_empty());
    assert_eq!(w.move_op(c as u32, t.run, t.dep, 0.0), Ok(None));
    assert_eq!(w.move_op(c as u32, t.run, t.dep + 77.0, 60.0).unwrap_err().why, Why::NoTrain);
}

/// A train keeps clear of its own line's trains each side, and the line's order never changes.
#[test]
fn a_train_keeps_clear_of_its_lines_neighbours() {
    let (mut w, h, _) = corridor(3);
    let t = pick(&w, h);
    let r = w.move_range(h as u32, t.run, t.dep).unwrap();
    assert!(matches!(r.hi_why, Why::Close { .. }) && matches!(r.lo_why, Why::Close { .. }), "{r:?}");
    let (mut steps, shape0) = (0, shape(&w, h));
    let (up, why) = walk(&mut w, h as u32, &t, 45.0, &mut steps);
    assert_eq!(shape(&w, h), shape0, "no train enters or leaves service on a headway line either");
    assert!(matches!(why, Why::Close { .. }) && (up - r.hi).abs() < 1e-3 && up > 100.0 && up < t.hw - 60.0, "{up} {why:?}");
    let day: Vec<Trip> = trips(&w, h, 2.0 * DAY, 3.0 * DAY).into_iter().filter(|x| x.run == 0).collect();
    let mut by_dep = day.clone();
    by_dep.sort_by(|a, b| a.dep.total_cmp(&b.dep));
    assert!(by_dep.windows(2).all(|p| p[1].draw - p[0].draw >= 60.0), "the order and a minute between hold");
    let (down, why) = walk(&mut w, h as u32, &t, -45.0, &mut steps);
    assert!(matches!(why, Why::Close { .. }) && (down - r.lo).abs() < 1e-3, "{down} {why:?}");
    assert_eq!(shape(&w, h), shape0);
}

/// A moved train holds its row: the line's next edit leaves the row's times where they were.
#[test]
fn a_moved_train_holds_its_row() {
    let build = || {
        let mut w = world();
        let n: Vec<u32> = (0..=4).map(|k| node(&mut w, k as f64 * 1500.0, 0.0, 0)).collect();
        for k in 0..4 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
        for &x in &n { station(&mut w, x, 200); }
        terminals_at_dead_ends(&mut w);
        let a = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
        let b = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
        (w, a, b)
    };
    let (mut w, _, b) = build();
    let t = trips(&w, b as usize, DAY + 8.0 * 3600.0, DAY + 9.0 * 3600.0).into_iter().find(|x| x.run == 0).unwrap();
    assert!(!w.svc.move_holds(&w.net, b as usize)[HIGH]);
    let op = w.move_op(b, 0, t.dep, 60.0).unwrap().unwrap();
    w.apply(op).unwrap();
    assert!(w.svc.move_holds(&w.net, b as usize)[HIGH], "the row of the train moved is held");
    let held = w.net.line_phase_of(b)[HIGH];
    set_headway(&mut w, b, [6.0, 3.0, 2.0]);
    assert_eq!(w.net.line_phase_of(b)[HIGH], held, "not placed again");
}

/// A move is a service edit with the edit lead: a train that left before the edit runs as it did, one after it moves.
#[test]
fn a_move_leaves_an_era_at_the_edit() {
    let (mut w, _, c) = corridor(3);
    let now = 2.0 * DAY + 9.0 * 3600.0;
    let all = trips(&w, c, now - 3.0 * 3600.0, now + 3.0 * 3600.0);
    let early = all.iter().find(|t| t.run == 0 && t.dep < now - 3600.0).copied().unwrap();
    let late = all.iter().find(|t| t.run == 0 && t.dep > now + 3600.0).copied().unwrap();
    let before = w.svc.before_edit(&w.net, now);
    for t in [early, late] {
        let op = w.move_op(c as u32, 0, t.dep, 60.0).unwrap().unwrap();
        w.apply(op).unwrap();
    }
    w.svc.note_changes(&w.net, &before, now);
    assert_eq!(w.svc.eras_of(c).len(), 1, "an era at the edit");
    assert!(w.svc.eras_of(c)[0].moves.is_empty(), "the trains before it ran no move");
    let after = trips(&w, c, now - 3.0 * 3600.0, now + 3.0 * 3600.0);
    let get = |t: &Trip| *after.iter().find(|x| x.run == 0 && x.dep == t.dep).unwrap();
    assert_eq!(get(&early).draw, early.draw, "a train that has left stays");
    assert_eq!(get(&late).draw, late.draw + 60.0, "a later one moves");
    assert!(w.svc.moves_of(c).iter().all(|m| m.kind == MoveKind::Own) && w.svc.moves_of(c).len() == 2);
}


/// The day's trains of run 0 that run moved, as (dep, seconds).
fn moved_trains(w: &TrackWorld, l: usize, run: u8) -> Vec<(f64, f32)> {
    let mut v: Vec<(f64, f32)> = trips(w, l, 2.0 * DAY, 3.0 * DAY).iter().filter(|t| t.run == run && t.moved != 0.0).map(|t| (t.dep, t.moved)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v
}

fn key_at(w: &TrackWorld, l: usize, dep: f64) -> f64 {
    let lat = w.svc.lattice(&w.net, l).unwrap();
    moves::key_of(dep, lat.shift + lat.lead)
}

/// A move is the train leaving at a time of day: it runs on every day, and a rate step (a train more on a count line) or a
/// stop added at the first end leaves it on the departure nearest that time; undone to the bit.
#[test]
fn a_moved_train_stays_the_one_at_its_time_through_a_count_step_and_a_new_first_stop() {
    let (mut w, h, c) = corridor(3);
    let t = pick(&w, c);
    let op = w.move_op(c as u32, t.run, t.dep, 60.0).unwrap().unwrap();
    w.apply(op).unwrap();
    let key = key_at(&w, c, t.dep);
    let stored = w.svc.stored_of(c).to_vec();
    assert_eq!(moved_trains(&w, c, t.run), vec![(t.dep, 60.0)]);
    // the same time of day on another day
    let other = trips(&w, c, 4.0 * DAY, 5.0 * DAY).into_iter().filter(|x| x.run == t.run && x.moved != 0.0).count();
    assert_eq!(other, 1, "the move runs every day");
    assert_eq!(w.svc.moves_waiting(&w.net, c), 0);

    // a count step
    let before = dump(&w, c);
    let n = w.net.trains_of(c).unwrap();
    let tph = slots_to_rows(w.net.line_tph[c]);
    w.apply(Op::TrainCounts { id: c as u32, trains: Some([n[0] + 1, n[1], n[2]]), tph }).unwrap();
    let got = moved_trains(&w, c, t.run);
    assert_eq!(got.len(), 1, "{got:?}");
    let hw = trips(&w, c, t.dep - 100.0, t.dep + 100.0)[0].hw;
    assert!(moves::apart(key_at(&w, c, got[0].0), key) < 0.5 * hw && got[0].1 > 0.0 && got[0].1 <= 60.0, "{got:?} {key} {hw}");
    w.undo().unwrap().unwrap();
    assert_eq!(dump(&w, c), before, "undone to the bit");
    assert_eq!(&*w.svc.stored_of(c), &stored[..]);

    // a stop added at the first end: the line starts a station earlier
    let stops = w.net.line_stops(c as u32).to_vec();
    let first = w.net.line_path(h as u32)[0];
    let mut d = w.net.line_data(c as u32).unwrap();
    d.stops.insert(0, w.net.line_stops(h as u32)[0]);
    d.path.insert(0, first);
    assert_ne!(d.stops, stops);
    w.apply(Op::Line { id: c as u32, data: Some(d) }).unwrap();
    let got = moved_trains(&w, c, t.run);
    assert_eq!(got.len(), 1, "{got:?}");
    assert!(got[0].1 > 0.0 && got[0].1 <= 60.0, "{got:?}");
    w.undo().unwrap().unwrap();
    assert_eq!(dump(&w, c), before);
    assert_eq!(&*w.svc.stored_of(c), &stored[..]);
}

/// An Offset step on the row moves the train's key with the row: the train that was moved is the same one, a half minute
/// later, with its seconds; undone to the bit.
#[test]
fn an_offset_step_keeps_the_moved_train() {
    let (mut w, _, c) = corridor(3);
    let t = pick(&w, c);
    let op = w.move_op(c as u32, t.run, t.dep, 60.0).unwrap().unwrap();
    w.apply(op).unwrap();
    let (before, stored) = (dump(&w, c), w.svc.stored_of(c).to_vec());
    let row = w.net.row_slots(c).iter().position(|&(k, _)| k == w.net.place(c, t.level as usize)).unwrap();
    let op = w.nudge_op(c as u32, row, 30.0, false).unwrap().unwrap();
    w.apply(op).unwrap();
    let got = moved_trains(&w, c, t.run);
    assert_eq!(got.len(), 1, "{got:?}");
    assert!((got[0].0 - (t.dep + 30.0)).abs() < 1e-3 && got[0].1 == 60.0, "{got:?} against {}", t.dep + 30.0);
    w.undo().unwrap().unwrap();
    assert_eq!(dump(&w, c), before, "undone to the bit");
    assert_eq!(&*w.svc.stored_of(c), &stored[..]);
}

/// A line turned round (its first and last stops swapped) keeps each moved train in the other direction; a line removed
/// takes them, and its undo gives them back.
#[test]
fn a_turned_round_or_removed_line_takes_its_moves_along() {
    let (mut w, _, c) = corridor(3);
    let t = pick(&w, c);
    let op = w.move_op(c as u32, t.run, t.dep, 60.0).unwrap().unwrap();
    w.apply(op).unwrap();
    let stored = w.svc.stored_of(c).to_vec();
    let mut d = w.net.line_data(c as u32).unwrap();
    d.stops.reverse();
    d.path = d.path.iter().rev().map(|p| p ^ 1).collect();
    w.apply(Op::Line { id: c as u32, data: Some(d) }).unwrap();
    let now = w.svc.stored_of(c);
    assert_eq!((now.len(), now[0].run, now[0].at, now[0].secs), (1, 1 - stored[0].run, stored[0].at, 60.0));
    w.undo().unwrap().unwrap();
    assert_eq!(&*w.svc.stored_of(c), &stored[..]);
    w.apply(Op::Line { id: c as u32, data: None }).unwrap();
    assert!(w.svc.stored_of(c).is_empty() && w.svc.moves_of(c).is_empty());
    w.undo().unwrap().unwrap();
    assert_eq!(&*w.svc.stored_of(c), &stored[..]);
}

/// Seconds stored beyond the room a train has run clamped to it, and the stored value stays; a move no departure is near
/// waits, counted, and runs nothing.
#[test]
fn stored_seconds_run_clamped_and_a_move_with_no_train_waits() {
    let (mut w, _, c) = corridor(3);
    let t = pick(&w, c);
    let r = w.move_range(c as u32, t.run, t.dep).unwrap();
    let key = key_at(&w, c, t.dep);
    w.svc.set_moves(c, vec![Move { run: t.run, at: key, secs: 500.0, kind: moves::MoveKind::Own, row: 0 }]);
    w.retime();
    assert_eq!(w.svc.stored_of(c)[0].secs, 500.0);
    assert!((w.svc.moves_of(c)[0].secs as f64 - r.hi).abs() < 1e-3, "{} against {}", w.svc.moves_of(c)[0].secs, r.hi);
    assert!((moved_trains(&w, c, t.run)[0].1 as f64 - r.hi).abs() < 1e-3);
    // a line with no trains has no departure for it: it waits, and runs again when the trains come back
    w.svc.set_moves(c, vec![Move { run: t.run, at: key, secs: 60.0, kind: moves::MoveKind::Own, row: 0 }]);
    let tph = slots_to_rows(w.net.line_tph[c]);
    w.apply(Op::TrainCounts { id: c as u32, trains: Some([0, 0, 0]), tph }).unwrap();
    assert!(moved_trains(&w, c, t.run).is_empty());
    assert_eq!(w.svc.moves_waiting(&w.net, c), 1);
    assert_eq!(w.svc.stored_of(c).len(), 1);
    w.undo().unwrap().unwrap();
    assert_eq!(w.svc.moves_waiting(&w.net, c), 0);
    assert_eq!(moved_trains(&w, c, t.run), vec![(t.dep, 60.0)]);}

/// Local and express headways (minutes) on the test line: far enough apart that a train has room to move.
const HL: f32 = 30.0;
const HX: f32 = 20.0;

/// Every mid-morning departure of run 0 of `line` with a train timed with it that the group can move 40 s either way, and its
/// partners (T-4548).
fn partnered(w: &TrackWorld, line: u32) -> Vec<(Trip, Vec<Train>)> {
    let (a, b) = (2.0 * DAY + 6.0 * 3600.0, 2.0 * DAY + 14.0 * 3600.0);
    trips(w, line as usize, a, b)
        .into_iter()
        .filter(|t| t.run == 0)
        .filter_map(|t| {
            let lat = w.svc.lattice(&w.net, line as usize)?;
            let me = w.svc.train_at(&w.net, line as usize, 0, moves::key_of(t.dep, lat.shift + lat.lead))?;
            let ps = w.svc.partners(me);
            (!ps.is_empty() && w.move_op(line, 0, t.dep, 40.0).is_ok() && w.move_op(line, 0, t.dep, -40.0).is_ok()).then_some((t, ps))
        })
        .collect()
}

/// A train's own timetable's length, by its place in the day.
fn run_secs(w: &TrackWorld, t: Train) -> f64 {
    let ls = &w.svc.lines[t.line as usize];
    ls.prof_of(t.run as usize, t.lev as usize, ls.runs[t.run as usize].var_at(t.lev as usize, t.place)).duration
}

/// An express with nowhere to pass follows its local: a step of the local moves its express by the same seconds, the express
/// keeps the waits it had behind it, the whole group stops at the tightest range, and a step back to 0 takes the express back.
#[test]
fn a_local_moves_the_express_timed_behind_it() {
    let (mut w, _, local, express) = shinkansen();
    for (id, h) in [(local, HL), (express, HX)] {
        set_headway(&mut w, id, [60.0 / h, 60.0 / h, 0.0]);
    }
    let all = partnered(&w, local);
    assert!(all.len() >= 5, "{} locals with a train timed behind them", all.len());
    let base = dump_both(&w, local, express);
    // a step of 30 s on each: the express goes the same seconds, and keeps its waits unless it has no room left for them (a
    // train moved later has that much less stand at its far end, and its waits are fitted into what is left)
    let (mut kept, mut cut) = (0, 0);
    let mut shown = None;
    for (t, ps) in &all {
        assert!(ps.iter().all(|p| p.line == express), "{ps:?}");
        let before: Vec<f64> = ps.iter().map(|p| run_secs(&w, *p)).collect();
        w.apply(w.move_op(local, t.run, t.dep, 30.0).unwrap().unwrap()).unwrap();
        let with = w.svc.moves_of(express as usize);
        assert!(with.len() == ps.len() && with.iter().all(|m| m.kind == MoveKind::With && (m.secs - 30.0).abs() < 1e-3), "the same seconds: {with:?}");
        let own = w.svc.moves_of(local as usize);
        assert!(own.len() == 1 && own[0].kind == MoveKind::Own && (own[0].secs - 30.0).abs() < 1e-3, "{own:?}");
        let after: Vec<f64> = ps.iter().map(|p| run_secs(&w, *p)).collect();
        if before.iter().zip(&after).all(|(b, a)| (a - b).abs() < 1.0) {
            kept += 1;
            shown.get_or_insert((*t, ps.clone()));
        } else {
            cut += 1;
        }
        w.undo().unwrap().unwrap();
        assert_eq!(dump_both(&w, local, express), base, "undone to the bit");
    }
    eprintln!("{kept} expresses keep their waits, {cut} have less room for them after the move");
    assert!(kept == all.len(), "{cut} expresses lose waits behind the local");
    let (t, _) = shown.unwrap();
    w.apply(w.move_op(local, t.run, t.dep, 30.0).unwrap().unwrap()).unwrap();
    assert!(w.svc.move_holds(&w.net, express as usize)[0], "the express's row is held with it");
    // walk to the end: the step stops at the tightest range of the group, the express always the local's seconds
    let mut steps = 1;
    let (up, _) = walk(&mut w, local, &t, 30.0, &mut steps);
    let top = w.svc.moves_of(express as usize).iter().map(|m| m.secs as f64).fold(f64::NEG_INFINITY, f64::max);
    assert!((top - up).abs() < 1e-3 && up > 30.0 && up <= 120.0 + 1e-3, "{up} {top}");
    // back to 0: the express is where it was
    let now = w.move_range(local, t.run, t.dep).unwrap().now;
    w.apply(w.move_op(local, t.run, t.dep, -now).unwrap().unwrap()).unwrap();
    assert!(w.svc.moves_of(local as usize).is_empty() && w.svc.moves_of(express as usize).is_empty());
    assert_eq!(dump_both(&w, local, express), base);
}

fn dump_both(w: &TrackWorld, a: u32, b: u32) -> String {
    format!("{}{}", dump(w, a as usize), dump(w, b as usize))
}

/// An express every local waits for at a passing station moves with the one local moved, not with the other locals it passes:
/// one step, never through a partner.
#[test]
fn an_express_passing_two_locals_moves_only_with_the_one_moved() {
    let (mut w, n, local, express) = shinkansen();
    for (id, h) in [(local, 10.0f32), (express, 10.0f32)] {
        set_headway(&mut w, id, [60.0 / h, 60.0 / h, 0.0]);
    }
    set_passing(&mut w, &n[1..6], true);
    // the express's partners are the locals that wait for it: find one with several
    let express_train = trips(&w, express as usize, 2.0 * DAY + 8.0 * 3600.0, 2.0 * DAY + 12.0 * 3600.0).into_iter().filter(|t| t.run == 0).find_map(|t| {
        let lat = w.svc.lattice(&w.net, express as usize)?;
        let me = w.svc.train_at(&w.net, express as usize, 0, moves::key_of(t.dep, lat.shift + lat.lead))?;
        let ps = w.svc.partners(me);
        (ps.len() >= 2).then_some((t, ps))
    });
    let (_, ps) = express_train.expect("an express two locals wait for");
    assert!(ps.iter().all(|p| p.line == local), "{ps:?}");
    // move the first of the locals: its express moves, and so does no other local
    let first = ps[0];
    let (lat, k) = (w.svc.lattice(&w.net, local as usize).unwrap(), w.svc.place_key(&w.net, first).unwrap());
    let t = trips(&w, local as usize, 2.0 * DAY + 6.0 * 3600.0, 2.0 * DAY + 14.0 * 3600.0).into_iter().find(|x| x.run == first.run && moves::apart(moves::key_of(x.dep, lat.shift + lat.lead), k) < 1.0).unwrap();
    let op = w.move_op(local, t.run, t.dep, 30.0).unwrap().unwrap();
    w.apply(op).unwrap();
    let ours = w.svc.moves_of(local as usize);
    assert!(ours.len() == 1 && ours[0].kind == MoveKind::Own, "no other local moves: {ours:?}");
    assert!(ps[1..].iter().all(|p| w.svc.partners(*p).iter().all(|x| x.line != local)), "(a local's partners are expresses)");
    let theirs = w.svc.moves_of(express as usize);
    assert!(!theirs.is_empty() && theirs.iter().all(|m| m.kind == MoveKind::With && (m.secs - ours[0].secs).abs() < 1e-3), "{theirs:?}");
}


