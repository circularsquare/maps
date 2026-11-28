use super::super::params::{HIGH, LOW, SLOTS};
use super::super::service::Trip;
use super::super::tests::{edge, headway_line, node, set_headway, station, terminals_at_dead_ends, world};
use super::super::world::{Op, TrackWorld};

/// A line's departures on run `run` from 07:40 to 09:20 of day 1.
fn deps(w: &TrackWorld, l: u32, run: usize) -> Vec<f64> {
    let mut out: Vec<Trip> = vec![];
    let t0 = 86400.0 + 8.0 * 3600.0;
    w.svc.run_deps(&w.net, l as usize, run, t0 - 1200.0, t0 + 4800.0, &mut out);
    out.iter().map(|t| t.dep).collect()
}

/// Whether every departure of `after` from 08:00 to 09:00 is one of `before`'s `by` seconds later.
fn moved_by(before: &[f64], after: &[f64], by: f64) -> bool {
    let t0 = 86400.0 + 8.0 * 3600.0;
    let inside: Vec<f64> = after.iter().copied().filter(|&d| d >= t0 && d < t0 + 3600.0).collect();
    !inside.is_empty() && inside.iter().all(|a| before.iter().any(|b| (a - by - b).abs() < 1e-2))
}

/// Two headway lines on one track and a third on its own.
fn two_on_one_track() -> (TrackWorld, Vec<u32>, u32, u32, u32) {
    let mut w = world();
    let n: Vec<u32> = (0..=4).map(|k| node(&mut w, k as f64 * 1500.0, 0.0, 0)).collect();
    for k in 0..4 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
    for &x in &n { station(&mut w, x, 200); }
    let q = [node(&mut w, 0.0, 600.0, 0), node(&mut w, 6000.0, 600.0, 0)];
    edge(&mut w, q[0], q[1], &[], 0).unwrap();
    for &x in &q { station(&mut w, x, 200); }
    terminals_at_dead_ends(&mut w);
    let a = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
    let b = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
    let alone = headway_line(&mut w, &q, [6.0, 4.0, 2.0]);
    (w, n, a, b, alone)
}

fn nudge(w: &mut TrackWorld, l: u32, row: usize, by: f64) -> bool {
    match w.nudge_op(l, row, by).unwrap() {
        Some(op) => { w.apply(op).unwrap(); true }
        None => false,
    }
}

/// A nudge moves that row's departures both ways and nothing else; the row is held through the line's next edit,
/// a line made later is placed around its times, and an undo puts every phase back to the bit.
#[test]
fn a_nudge_moves_its_row_alone_and_holds_it() {
    let (mut w, n, a, b, _) = two_on_one_track();
    let (b0, b1, a0, night) = (deps(&w, b, 0), deps(&w, b, 1), deps(&w, a, 0), w.net.line_phase_of(b)[LOW]);
    let phases: Vec<[f32; SLOTS]> = (0..w.net.line_count() as u32).map(|l| w.net.line_phase_of(l)).collect();
    assert!(nudge(&mut w, b, 0, 120.0));
    assert!(moved_by(&b0, &deps(&w, b, 0), 120.0) && moved_by(&b1, &deps(&w, b, 1), 120.0), "both directions, 2 min later");
    assert_eq!(deps(&w, a, 0), a0, "the other line keeps its times");
    assert_eq!(w.net.line_phase_of(b)[LOW], night, "another row keeps its times");
    assert_eq!(w.net.offset_at(b, HIGH), Some(120.0));
    assert_eq!(w.row_nudge(b, 0), Some((120.0, 300.0, true, vec![])));
    // undone to the bit, and done again
    w.undo().unwrap().unwrap();
    assert_eq!((0..w.net.line_count() as u32).map(|l| w.net.line_phase_of(l)).collect::<Vec<_>>(), phases);
    assert_eq!(w.net.offset_at(b, HIGH), None);
    w.redo().unwrap().unwrap();
    assert_eq!(w.net.offset_at(b, HIGH), Some(120.0));
    // held through the line's own edit
    let held = w.net.line_phase_of(b)[HIGH];
    set_headway(&mut w, b, [6.0, 3.0, 2.0]);
    assert_eq!(w.net.line_phase_of(b)[HIGH], held, "the line's next edit leaves the nudged row");
    // a line made later is placed around the nudged times
    let c = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
    let (da, db, dc) = (deps(&w, a, 0), deps(&w, b, 0), deps(&w, c, 0));
    let closest = |x: &[f64], y: &[f64]| x.iter().flat_map(|p| y.iter().map(move |q| (p - q).abs())).fold(f64::INFINITY, f64::min);
    assert!(closest(&dc, &db) > 60.0 && closest(&dc, &da) > 60.0, "{da:?} {db:?} {dc:?}");
}

/// The range is half a headway either way; back at 0 the row is placed by the spacing again.
#[test]
fn a_nudge_stops_at_half_a_headway_and_zero_releases_it() {
    let (mut w, _, _, b, _) = two_on_one_track();
    let base = w.net.line_phase_of(b)[HIGH];
    assert!(nudge(&mut w, b, 0, 10_000.0));
    assert_eq!(w.row_nudge(b, 0).unwrap().0, 300.0, "half the 10 min headway");
    assert!(!nudge(&mut w, b, 0, 60.0), "nothing past the end of the range");
    assert!(nudge(&mut w, b, 0, -360.0));
    assert_eq!(w.row_nudge(b, 0).unwrap().0, -60.0);
    assert!(nudge(&mut w, b, 0, 60.0));
    assert_eq!(w.net.offset_at(b, HIGH), None, "back at 0 the row is released");
    assert!((w.net.line_phase_of(b)[HIGH] - base).abs() < 1e-3, "at its times before the first nudge");
    assert!(w.nudge_op(b, 0, 0.0).unwrap().is_none());
}

/// Held rows come back from a save with their times; a removed line holds none, and its undo holds them again.
#[test]
fn held_rows_last_through_a_save_and_a_removal_undone() {
    let (mut w, _, a, b, _) = two_on_one_track();
    nudge(&mut w, b, 0, -75.0);
    nudge(&mut w, a, 2, 600.0);
    let bytes = super::super::save::encode(&w.net);
    assert!(bytes.windows(4).any(|x| x == b"LOF1"));
    let back = TrackWorld::new(super::super::save::decode(&bytes).unwrap());
    for l in [a, b] {
        assert_eq!(back.net.line_phase_of(l), w.net.line_phase_of(l));
        assert_eq!(back.net.line_offset.get(l as usize), w.net.line_offset.get(l as usize));
        assert_eq!(deps(&back, l, 0), deps(&w, l, 0));
    }
    assert_eq!(super::super::save::encode(&back.net), bytes);
    let d = w.net.line_data(b).unwrap();
    w.apply(Op::Line { id: b, data: None }).unwrap();
    assert!(!w.net.any_offset(b));
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.offset_at(b, HIGH), Some(-75.0));
    assert_eq!(w.net.line_data(b).unwrap().path, d.path);
    // and a network with none writes no block
    let (fresh, ..) = two_on_one_track();
    assert!(!super::super::save::encode(&fresh.net).windows(4).any(|x| x == b"LOF1"));
}

/// A line running other rates on weekends has its weekend row moved and held with the weekday one.
#[test]
fn weekends_follow_the_weekday_nudge() {
    let (mut w, _, _, b, _) = two_on_one_track();
    w.apply(Op::Weekend { id: b, weekend: super::super::net::Weekend { share: 50, counts: None, headways: [0; 3] } }).unwrap();
    assert!(w.net.rest_solved && w.net.runs_own_weekend(b as usize));
    let rest = w.net.line_phase_rest_of(b)[HIGH];
    nudge(&mut w, b, 0, 90.0);
    assert!((w.net.line_phase_rest_of(b)[HIGH] - rest - 90.0).abs() < 1e-3);
    w.apply(Op::Weekend { id: b, weekend: super::super::net::Weekend { share: 60, counts: None, headways: [0; 3] } }).unwrap();
    assert!((w.net.line_phase_rest_of(b)[HIGH] - rest - 90.0).abs() < 1e-3, "a weekend edit leaves it held");
}

/// The anchor rule: an express timed through passing stations behind its local holds no offset of its own. A nudge
/// of either moves the local's, and both lines by the same seconds, so the express keeps its place and its waits.
#[test]
fn a_nudge_moves_a_timed_pair_together() {
    let (mut w, n, local, express) = super::super::overtake::tests::shinkansen();
    for id in [local, express] { set_headway(&mut w, id, [6.0, 6.0, 0.0]); }
    super::super::overtake::tests::set_passing(&mut w, &n[1..6], true);
    let pairs = w.svc.cap.follow_pairs(&w.net, HIGH);
    assert!(pairs.contains(&(express, local)), "{pairs:?}");
    let gap = |w: &TrackWorld| w.net.line_phase_of(express)[HIGH] - w.net.line_phase_of(local)[HIGH];
    let (g0, holds) = (gap(&w), w.svc.lines[local as usize].runs[0].vars[HIGH].iter().map(|v| v.stop_holds.clone()).collect::<Vec<_>>());
    let (l0, x0) = (deps(&w, local, 0), deps(&w, express, 0));
    assert!(nudge(&mut w, express, 0, 60.0));
    assert_eq!(w.net.offset_at(local, HIGH), Some(60.0), "the slower line holds it");
    assert_eq!(w.net.offset_at(express, HIGH), None);
    assert!((gap(&w) - g0).abs() < 1e-3, "the express keeps its place behind the local");
    assert!(moved_by(&l0, &deps(&w, local, 0), 60.0) && moved_by(&x0, &deps(&w, express, 0), 60.0));
    for (l, other) in [(express, local), (local, express)] {
        let (o, half, held, others) = w.row_nudge(l, 0).unwrap();
        assert!(o == 60.0 && (half - 300.0).abs() < 0.1 && held && others == vec![other], "{l}: {o} {half} {held} {others:?}");
    }
    let after: Vec<_> = w.svc.lines[local as usize].runs[0].vars[HIGH].iter().map(|v| v.stop_holds.clone()).collect();
    assert_eq!(after.len(), holds.len(), "as many timetables");
    for (x, y) in after.iter().zip(&holds) {
        assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1.0), "the local's waits for its express stay: {x:?} {y:?}");
    }
}
