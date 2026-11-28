use super::super::hours_tests::corridor;
use super::super::params::HIGH;
use super::super::moves::MoveKind;
use super::super::service::Trip;
use super::super::tests::{edge, headway_line, node, set_headway, station, terminals_at_dead_ends, world};
use super::super::world::TrackWorld;
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
