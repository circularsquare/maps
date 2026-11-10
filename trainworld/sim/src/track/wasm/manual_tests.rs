//! T-3023: manual mode (`set_manual`): every stop in order, a reversal at a stop a terminus.

use super::*;
use crate::track::tests::{edge, node, station, terminals_at_dead_ends, world};

fn api(w: TrackWorld) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.w = w;
    a
}

/// A - B - C - D along x, 2 km apart, all stations.
fn straight() -> (TrackWorld, [u32; 4]) {
    let mut w = world();
    let n: Vec<u32> = (0..4).map(|i| node(&mut w, 2000.0 * i as f64, 0.0, 0)).collect();
    for p in n.windows(2) {
        edge(&mut w, p[0], p[1], &[], 0).unwrap();
    }
    for &s in &n {
        station(&mut w, s, 200);
    }
    terminals_at_dead_ends(&mut w);
    (w, [n[0], n[1], n[2], n[3]])
}

/// `stop_order_tests::detour`: a line `[X1, A, B]` reverses at A.
fn detour() -> (TrackWorld, [u32; 4]) {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let ja = node(&mut w, 1000.0, 0.0, 0);
    let jb = node(&mut w, 11000.0, 0.0, 0);
    let b = node(&mut w, 12000.0, 0.0, 0);
    let x1 = node(&mut w, 4000.0, 15000.0, 0);
    let x2 = node(&mut w, 8000.0, 15000.0, 0);
    edge(&mut w, a, ja, &[], 0).unwrap();
    edge(&mut w, ja, jb, &[], 0).unwrap();
    edge(&mut w, jb, b, &[], 0).unwrap();
    edge(&mut w, ja, x1, &[(2500.0, 0.0), (2500.0, 15000.0)], 0).unwrap();
    edge(&mut w, x1, x2, &[], 0).unwrap();
    edge(&mut w, x2, jb, &[(9500.0, 15000.0), (9500.0, 0.0)], 0).unwrap();
    for n in [a, b, x1, x2] {
        station(&mut w, n, 200);
    }
    (w, [a, x1, x2, b])
}

fn limiter_at(a: &TrackApi, n: u32) -> bool {
    let cap = &a.w.svc.cap;
    (0..cap.res.len()).any(|r| cap.res[r].node == n && cap.wire_kind(r) == 9)
}

#[test]
fn a_manual_line_writes_the_whole_cycle_and_folds_back() {
    let (w, [s0, s1, s2, s3]) = straight();
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[s0, s1, s2, s3], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert!(!a.line_manual(id) && !a.manual_folds(id));
    let two_way = a.w.svc.lines[id as usize].round_trip[0];
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s3, s2, s1, s0]);
    assert!(a.line_manual(id) && a.line_hand(id) && a.line_one_way(id) && a.manual_folds(id));
    assert_eq!(a.line_loop(id), 2);
    assert!(a.w.svc.lines[id as usize].ok);
    let lap = a.w.svc.lines[id as usize].round_trip[0];
    assert!((lap - two_way).abs() < 1.0, "{lap} against {two_way}");
    // a visit out: C on the way back is skipped; the list no longer folds
    assert_eq!(a.drop_stop(id, 4), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s3, s1, s0]);
    assert!(a.line_manual(id) && !a.manual_folds(id));
    // T-3089: C, a stop on the way out, is passed on the way back (the leg from D to B), which the map draws as a circle
    assert_eq!(a.passed_stations(id), vec![3.0, s2 as f64]);
    // switching off a list that does not fold keeps its stops as a normal line (T-3079)
    assert_eq!(a.set_manual(id, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s3, s1, s0]);
    assert!(!a.line_manual(id));
    assert_eq!(a.undo(), 0);
    // T-3089: the turning visit goes too: the visits either side of it become one
    assert_eq!(a.drop_stop(id, 3), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s1, s0]);
    assert_eq!(a.undo(), 0);
    // the first stop (the cycle's turning point) goes: the line now turns at the next one
    assert_eq!(a.drop_stop(id, 0), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s1, s2, s3, s1]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.drop_stop(id, 5), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s1, s2, s3, s1]);
    assert_eq!(a.undo(), 0);
    // a cycle of two stations does not shrink to one
    assert_eq!(a.order_stops(id, &[s0, s1, s0]), 0, "{}", a.issues());
    assert_ne!(a.drop_stop(id, 1), 0);
    assert_eq!(a.undo(), 0);
    // a visit in after row 3 through order_stops keeps the manual list as given
    assert_eq!(a.order_stops(id, &[s0, s1, s2, s3, s2, s1, s0]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s3, s2, s1, s0], "order_stops does not fold an out-and-back");
    assert!(a.line_manual(id));
    // a save carries the flag; a save with the block removed loads as a plain line
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LMN1").expect("LMN1 written");
    let mut b = api(world());
    assert!(b.load(&bytes) && b.line_manual(id));
    assert!(save::decode(&bytes[..at]).map(|n| n.line_manual.iter().all(|&m| !m)).unwrap());
    // back to the open line
    assert_eq!(a.set_manual(id, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, s2, s3]);
    assert!(!a.line_manual(id) && !a.line_one_way(id));
    assert!(!a.save().windows(4).any(|w| w == b"LMN1"), "no manual line: no block");
    assert_eq!(a.undo(), 0);
    assert!(a.line_manual(id) && a.w.net.line_stops(id).len() == 7);
}

#[test]
fn a_reversal_on_a_manual_line_turns_as_a_line_end_does() {
    let (w, [sa, x1, _, sb]) = detour();
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[sa, sb], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.order_stops(id, &[x1, sa, sb]), 0, "{}", a.issues());
    assert_eq!(a.line_reversals(id), [1.0]);
    assert!(!limiter_at(&a, sa), "no reversal limiter any more (T-3107): the turn is the station's pool");
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert!(a.line_reversals(id).contains(&1.0), "the train still turns at A");
    assert!(!limiter_at(&a, sa));
    // the flag survives a save and load
    let bytes = a.save();
    let mut b = api(world());
    assert!(b.load(&bytes));
    assert!(b.line_manual(id));
}

/// The loop marker stays a shortcut, but a manual list has no "stop looping": its rows are the
/// player's to remove.
#[test]
fn a_manual_loop_has_no_stop_looping_marker() {
    use super::turn_tests::{teardrop, tier_of};
    let (w, [a0, b, m, _, k]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(tier_of(&a, id, m), 7);
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k, m, b, a0]);
    assert_ne!(tier_of(&a, id, m), 8);
    assert_ne!(a.open_loop(id), 0, "a manual line opens through set_manual");
    assert_eq!(a.set_manual(id, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[a0, b, m, k]);
}

/// A 30-station line is 59 rows in manual mode and stays editable (an edit routes the whole list).
#[test]
fn a_long_manual_line_edits_in_a_blink() {
    let mut w = world();
    let n: Vec<u32> = (0..30).map(|i| node(&mut w, 1500.0 * i as f64, 0.0, 0)).collect();
    for p in n.windows(2) {
        edge(&mut w, p[0], p[1], &[], 0).unwrap();
    }
    for &s in &n {
        station(&mut w, s, 200);
    }
    terminals_at_dead_ends(&mut w);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &n, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let t = std::time::Instant::now();
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id).len(), 59);
    let mut stops = a.w.net.line_stops(id).to_vec();
    stops.remove(40);
    let t2 = std::time::Instant::now();
    assert_eq!(a.order_stops(id, &stops), 0, "{}", a.issues());
    eprintln!("set_manual {:?}, one order_stops {:?}", t2 - t, t2.elapsed());
    assert_eq!(a.w.net.line_stops(id).len(), 58);
}

/// T-3079: a leg that ends where the next begins at a listed stop turns there for nothing, so a loop
/// of track past the stop is not taken to turn round.
#[test]
fn a_listed_stop_turns_the_train_with_no_detour_round_a_loop() {
    use super::turn_tests::teardrop;
    let (w, [a0, b, m, _, k]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert_eq!(a.line_reversals(id), [3.0], "the train turns at K, the listed stop");
    assert_eq!(a.order_fault_line(id, &[a0, b, m, k, m, b, a0]), Vec::<f64>::new());
}

/// T-3099: Exact edits on and off gives back the same line: a loop line keeps the route it runs (a stop on
/// its loop is not turned at), and a turned line whose loop has one station, whose stops read the same
/// both ways, stays turned rather than folding into its open line.
#[test]
fn exact_edits_on_and_off_keeps_a_turned_line() {
    use super::turn_tests::teardrop;
    let (w, [a0, b, m, _, k]) = teardrop(false);
    let mut a = api(w);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[a0, b, m, k], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.turn_end(id, m), 0, "{}", a.issues());
    let (stops, path) = (a.w.net.line_stops(id).to_vec(), a.w.net.line_path(id).to_vec());
    assert_eq!(stops, [a0, b, m, k, m, b, a0]);
    assert!(a.line_reversals(id).is_empty());
    assert_eq!(a.set_manual(id, true), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_path(id), &path[..], "Exact edits on keeps the route round the loop");
    assert!(!a.manual_folds(id), "a list that goes round a loop does not fold");
    assert_eq!(a.set_manual(id, false), 0, "{}", a.issues());
    assert_eq!((a.w.net.line_stops(id), a.w.net.line_path(id)), (&stops[..], &path[..]), "Exact edits off gives the turned line back");
    assert_eq!((a.line_loop(id), a.line_one_way(id)), (2, true));
    // an open line still goes out and back and folds to the open line
    let id2 = a.new_line_id();
    assert_eq!(a.set_line(id2, &[a0, b, m], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "M".into()), 0, "{}", a.issues());
    assert_eq!(a.set_manual(id2, true), 0, "{}", a.issues());
    assert!(a.manual_folds(id2));
    assert_eq!(a.set_manual(id2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id2), &[a0, b, m]);
}
