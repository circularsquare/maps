use super::*;

/// A built line between two stations 6 km apart with two trains at every level, its id.
fn line_api() -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, e) = (node(&a, 0.0), node(&a, 6000.0));
    for (n, x) in [(w, 0.0), (e, 6000.0)] {
        assert_eq!(a.add_station(1, n, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let edges: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&k| a.w.net.edge_ok(k)).collect();
    assert_eq!(a.construct(&edges, &[w, e]), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[w, e], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 2, 2, 2), 0, "{}", a.issues());
    (a, id)
}

/// T-1071: an edit with a lead takes effect that much later: the trips drawn before then are
/// the old service's to the bit, and a save made before the change is over loads with it.
#[test]
fn a_count_change_with_a_lead_takes_effect_then_and_saves_with_its_eras() {
    let (mut a, id) = line_api();
    let now: f64 = 2.0 * 86400.0 + 11.0 * 3600.0 + 417.0;
    let epoch = (now / 3600.0).floor() * 3600.0;
    a.set_now(now);
    let old = a.trips(epoch);
    a.set_edit_lead(300.0);
    assert_eq!(a.set_train_counts(id, 3, 2, 2), 0, "{}", a.issues());
    assert_eq!(a.edit_time(), now + 300.0);
    // the lead is used once
    a.set_now(now + 1.0);
    assert_eq!(a.set_train_counts(id, 3, 2, 2), 0);
    assert_eq!(a.edit_time(), now + 1.0);
    let new = a.trips(epoch);
    // trips that left before the change are the same ones (draw, as the fourth number may
    // name a later departure), and none of the new pattern leaves before it
    let left = |v: &[f32]| v.chunks(4).filter(|t| (t[1] as f64) + epoch < now + 300.0 - 1.0).map(|t| (t[0], t[1])).collect::<Vec<_>>();
    assert_eq!(left(&old), left(&new));
    assert_ne!(old, new, "the change shows after the lead");
    // a save made now loads with the change still going on
    let bytes = a.save();
    assert_eq!(&bytes[bytes.len() - 4..], ERAS_TAG);
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    b.set_now(now + 1.0);
    assert_eq!(b.w.svc.eras, a.w.svc.eras);
    let (x, y) = (a.trips(epoch), b.trips(epoch));
    assert_eq!(x.len(), y.len());
    assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 0.01 || (p.is_nan() && q.is_nan())));
    // and a save with no change going on ends as before
    b.set_now(now + 6.0 * 3600.0);
    assert!(b.w.svc.eras.iter().all(|e| e.is_empty()));
    assert_eq!(b.save(), save::encode(&b.w.net));
}

/// T-1073: an undo or redo of a train count says it is a service edit, and one of a track edit does not.
#[test]
fn undo_and_redo_of_a_count_change_are_service_edits() {
    let (mut a, id) = line_api();
    assert!(!a.step_is_service(true));
    assert_eq!(a.set_train_counts(id, 3, 2, 2), 0, "{}", a.issues());
    assert!(a.step_is_service(false));
    assert!(!a.step_is_service(true));
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert!(a.step_is_service(true));
    assert_eq!(a.redo(), 0, "{}", a.issues());
    assert!(a.step_is_service(false));
}

/// T-1072: a stop added past one end moves no train on the ground, the other direction's included.
#[test]
fn a_stop_added_at_an_end_moves_no_train() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    assert_eq!(a.add_station(2, 0, 6000.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    for x in [0.0, 12000.0] {
        let n = node(&a, x);
        assert_eq!(a.add_station(1, n, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let ns: Vec<u32> = [0.0, 6000.0, 12000.0].iter().map(|&x| node(&a, x)).collect();
    let edges: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&k| a.w.net.edge_ok(k)).collect();
    assert_eq!(a.construct(&edges, &ns), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &ns[..2], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 4, 4, 4), 0, "{}", a.issues());
    let now = 2.0 * 86400.0 + 11.0 * 3600.0 + 417.0;
    a.set_now(now);
    let at_now = |a: &TrackApi| {
        let mut v: Vec<(u8, f64, f64)> = a.w.svc.trips(&a.w.net, now, now + 1e-3).iter().filter_map(|t| {
            let p = &a.w.svc.lines[t.line as usize].runs[t.run as usize].prof[t.level as usize];
            let tau = now - t.draw;
            (tau >= 0.0 && tau < p.duration).then(|| {
                let (s, _) = crate::track::profile::state_at(&p.phases, tau);
                let (x, _) = a.w.svc.position(&a.w.net, t.line, t.run as usize, s);
                (t.run, t.dep, x)
            })
        }).collect();
        v.sort_by(|p, q| p.1.total_cmp(&q.1).then(p.0.cmp(&q.0)));
        v
    };
    let before = at_now(&a);
    assert!(before.iter().any(|t| t.0 == 1), "a train on the way back: {before:?}");
    assert_eq!(a.set_line(id, &ns, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let after = at_now(&a);
    assert_eq!(before.len(), after.len(), "{before:?} {after:?}");
    for (p, q) in before.iter().zip(&after) {
        assert!((p.2 - q.2).abs() < 30.0, "{p:?} moved to {q:?}");
    }
}

/// T-3521: a line's weekend whole trains (`line_weekend`), high, medium, low.
fn weekend_of(a: &TrackApi, id: u32) -> Vec<f64> {
    a.line_weekend(id)[2..5].to_vec()
}

/// T-3521: a line follows its weekdays at its percent, so a weekday change moves the weekend
/// (counts rounded up); counts set by hand stay as set, keeping the percent for the panel.
#[test]
fn t3521_a_weekday_change_moves_a_weekend_at_its_percent() {
    let (mut a, id) = line_api();
    assert_eq!(a.line_weekend(id), vec![100.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], "a new line runs its weekday trains every day");
    assert_eq!(a.set_line_weekend_share(id, 50), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 5, 3, 2), 0, "{}", a.issues());
    assert_eq!(weekend_of(&a, id), vec![3.0, 2.0, 1.0], "half, rounded up");
    assert_eq!(a.set_line_weekend_counts(id, 1, 1, 0), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 6, 3, 2), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id), vec![50.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "counts of its own are left as set");
    assert_eq!(a.set_line_weekend_share(id, 50), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id), vec![50.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0], "the percent again");
}

/// T-3521: a headway line's weekend headway is its weekday headway over the percent, rounded down
/// to a whole minute: 5 min at 70% is 7 min; 2 min at 50% is 4 min, and under 100% never more
/// trains than the weekdays run.
#[test]
fn t3521_a_headway_weekend_rounds_down_to_a_minute() {
    assert_eq!(crate::track::net::weekend_headway(300, 70), 420);
    assert_eq!(crate::track::net::weekend_headway(300, 100), 300);
    assert_eq!(crate::track::net::weekend_headway(600, 150), 360);
    assert_eq!(crate::track::net::weekend_headway(0, 50), 0);
    let (mut a, id) = line_api();
    assert_eq!(a.set_auto_headway(id, 120, 120, 120), 0, "{}", a.issues());
    let week = a.w.net.line_trains[id as usize].unwrap();
    assert_eq!(a.set_line_weekend_share(id, 50), 0, "{}", a.issues());
    let trip = a.w.svc.lines[id as usize].round_trip;
    let most = a.w.svc.cap.line_limits(&a.w.net);
    let want: Vec<f64> = (0..3).map(|row| { let lev = ROW_SLOT[row]; crate::track::world::clamp_auto(auto_trains(trip[lev], 240), trip[lev], 240, most[id as usize][lev].0).min(week[row]) as f64 }).collect();
    assert_eq!(weekend_of(&a, id), want);
    assert!(want.iter().zip(week).all(|(w, k)| *w <= k as f64), "{want:?} on weekends, {week:?} on weekdays: never more under 100%");
}

/// T-3521: a headway line's weekends run the weekend headway, its trains standing at the ends for
/// the rest, as its weekdays keep theirs; the weekdays are left to the bit. Counts set by hand run
/// as a count line's.
#[test]
fn t3521_a_headway_weekend_runs_its_headway() {
    let (mut a, id) = line_api();
    let runs_every = |a: &TrackApi, what: &str, h: [f64; 3]| {
        let rest = a.line_tph_rest(id);
        for lev in 0..3 {
            let run = 3600.0 / rest[lev] as f64;
            assert!(run >= h[lev] - 0.5 && run <= h[lev] * 1.005 + 0.5, "{what}: every {run:.1} s at level {lev}, set {} s", h[lev]);
        }
    };
    // the first keeps the line's trains (T-752), the second takes the headway's
    assert_eq!(a.set_auto_headway(id, 480, 480, 480), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(id, 480, 480, 480), 0, "{}", a.issues());
    let week = a.w.net.line_tph[id as usize].map(f32::to_bits);
    assert!(slots_to_rows(a.w.net.line_tph[id as usize]).iter().all(|&t| (3600.0 / t as f64 - 480.0).abs() < 2.0), "the weekdays every 8 minutes: {:?}", a.w.net.line_tph[id as usize]);
    let trip = a.w.svc.lines[id as usize].round_trip;
    assert!(slots_to_rows(trip).iter().all(|&t| t > 600.0), "a round trip of a few headways: {trip:?}");
    // 8 minutes at 70% is 11:26, down to 11 minutes; at 150%, 5:20, down to 5
    for (share, h) in [(70, 660.0), (50, 960.0), (150, 300.0)] {
        assert_eq!(a.set_line_weekend_share(id, share), 0, "{}", a.issues());
        runs_every(&a, &format!("{share}%"), [h; 3]);
        assert_eq!(a.w.net.line_tph[id as usize].map(f32::to_bits), week, "{share}%: the weekdays as they were");
    }
    // a weekday edit keeps the weekend service on the weekend headways (`mirror_weekend`)
    assert_eq!(a.set_auto_headway(id, 480, 480, 600), 0, "{}", a.issues());
    runs_every(&a, "150% after a weekday edit", [300.0, 300.0, 360.0]);
    let more = a.w.net.line_trains[id as usize].unwrap().map(|x| x + 2);
    assert_eq!(a.set_line_weekend_counts(id, more[0], more[1], more[2]), 0, "{}", a.issues());
    let rest = a.line_tph_rest(id);
    assert!((0..3).all(|lev| rest[lev] > a.w.net.line_tph[id as usize][ROW_SLOT[lev]] * 1.2), "two trains more run more often: {rest:?}");
}

/// T-3521: a headway line whose weekend count is its weekday count (one train) still runs the
/// weekend headway: 15 minutes at 70% is 21 minutes.
#[test]
fn t3521_a_headway_weekend_on_the_weekday_count() {
    let (mut a, id) = line_api();
    // the first keeps the line's trains (T-752), the second takes the headway's
    assert_eq!(a.set_auto_headway(id, 900, 900, 900), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(id, 900, 900, 900), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_trains[id as usize], Some([1, 1, 1]));
    let week = a.w.net.line_tph[id as usize].map(f32::to_bits);
    assert_eq!(a.set_line_weekend_share(id, 70), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id), vec![70.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!(a.w.net.rest_solved);
    let rest = a.line_tph_rest(id);
    assert!(rest.iter().all(|&t| (3600.0 / t as f64 - 1260.0).abs() < 1.0), "{rest:?}");
    assert_eq!(a.w.net.line_tph[id as usize].map(f32::to_bits), week);
}

/// T-3557: a headway line's weekends stepped to a headway by hand run that headway, not the round
/// trip over the trains set (shorter than it by up to a train's share), and a count edit at another
/// level keeps it; a percent drops it; it saves and loads.
#[test]
fn t3557_a_stepped_weekend_headway_is_run() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_auto_headway(id, 480, 480, 480), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(id, 480, 480, 480), 0, "{}", a.issues());
    let trip = a.w.svc.lines[id as usize].round_trip[0];
    // a headway the round trip does not divide: whole trains for it run it faster than it
    let h = (trip / 1.7).round() as u32;
    assert!(h > 300, "a headway the track allows: {h} s over a round trip of {trip} s");
    let n = (trip / h as f64).ceil() as u32;
    assert!(trip / n as f64 + 1.0 < h as f64, "the count alone runs faster than {h} s: {} s", trip / n as f64);
    // stepping the High level alone leaves the other two at the headway they ran (their trains
    // would otherwise run their own faster rate)
    let rest0 = a.line_tph_rest(id);
    let m = a.w.net.line_trains[id as usize].unwrap();
    assert_eq!(a.set_line_weekend_headways(id, n, m[1], m[2], h, 0, 0), 0, "{}", a.issues());
    let rest1 = a.line_tph_rest(id);
    assert!((3600.0 / rest1[0] as f64 - h as f64).abs() <= 0.005 * h as f64 + 0.5, "High every {h} s: {}", 3600.0 / rest1[0] as f64);
    assert!((1..3).all(|lev| (rest1[lev] - rest0[lev]).abs() <= 0.005 * rest0[lev]), "the other levels as they ran: {rest0:?} then {rest1:?}");
    assert_eq!(a.set_line_weekend_headways(id, n, n, n, h, h, h), 0, "{}", a.issues());
    let run = |a: &TrackApi, lev: usize| 3600.0 / a.line_tph_rest(id)[lev] as f64;
    assert!((0..3).all(|lev| (run(&a, lev) - h as f64).abs() <= 0.005 * h as f64 + 0.5), "every {h} s: {} {} {}", run(&a, 0), run(&a, 1), run(&a, 2));
    assert_eq!(a.line_weekend(id)[5..], [h as f64; 3]);
    // a count edit at one level drops that level's headway only
    assert_eq!(a.set_line_weekend_counts(id, n + 1, n, n), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id)[5..], [0.0, h as f64, h as f64]);
    assert!(run(&a, 0) < h as f64 - 1.0, "that level runs its trains' own rate");
    // the headways save and load
    assert_eq!(a.set_line_weekend_headways(id, n, n, n, h, h, h), 0, "{}", a.issues());
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&a.save()));
    assert_eq!(b.line_weekend(id), a.line_weekend(id));
    // a percent drops them
    assert_eq!(a.set_line_weekend_share(id, 70), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id)[5..], [0.0; 3]);
}

/// The fleet is the most trains a line runs on either kind of day at any level, so more weekend
/// trains are bought.
#[test]
fn more_weekend_trains_are_bought() {
    let (mut a, id) = line_api();
    let cars = a.cars_needed();
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    assert_eq!(a.cars_needed(), cars * 3, "six trains on a weekend, two on weekdays");
    assert_eq!(a.set_line_weekend_share(id, 200), 0, "{}", a.issues());
    assert_eq!(a.cars_needed(), cars * 2, "twice the weekdays'");
    assert_eq!(a.set_line_weekend_counts(id, 1, 1, 0), 0, "{}", a.issues());
    assert_eq!(a.cars_needed(), cars, "fewer at the weekend: the weekdays' fleet");
}

/// Trips over six of local day `day`'s busiest hours, as `trips` gives them, as bits (a trip that
/// leaves service has NaN for its next departure; the line's local time is New York's, a few hours
/// behind game time).
fn day_trips(a: &TrackApi, day: f64) -> Vec<u32> {
    (0..6).flat_map(|h| a.trips(day * 86400.0 + (14 + h) as f64 * 3600.0)).map(f32::to_bits).collect()
}

fn saturday_trips(a: &TrackApi) -> Vec<u32> {
    day_trips(a, 6.0)
}

fn tuesday_trips(a: &TrackApi) -> Vec<u32> {
    day_trips(a, 2.0)
}

/// Trips in a list from `trips` (four numbers each).
fn count(trips: &[u32]) -> usize {
    trips.len() / 4
}

/// Undo and redo of weekend counts move the weekend's rates and trips back and forth to the bit.
#[test]
fn weekend_counts_undo_and_redo_to_the_bit() {
    let (mut a, id) = line_api();
    let (w0, sat0, tue0) = (a.line_weekend(id), saturday_trips(&a), tuesday_trips(&a));
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    assert!(a.step_is_service(false), "a weekend edit is a service edit");
    let (rates, sat) = (a.line_tph_rest(id), saturday_trips(&a));
    assert!(count(&sat) > count(&sat0), "{} trips with six weekend trains, {} with two", count(&sat), count(&sat0));
    assert_eq!(tuesday_trips(&a), tue0, "Tuesday as it was");
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_weekend(id), w0);
    assert!(!a.w.net.rest_solved, "every day back on the weekdays'");
    assert_eq!(saturday_trips(&a), sat0);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.line_weekend(id)[2..5], [6.0, 2.0, 2.0]);
    assert_eq!(a.line_tph_rest(id), rates);
    assert_eq!(saturday_trips(&a), sat);
}

/// A save keeps the weekend counts and their settled rates, so a reload runs the same trips on a
/// Saturday and on a public holiday from the line's calendar (which is not saved: the clock worker
/// sends it again).
#[test]
fn weekend_and_holiday_trips_load_from_a_save_as_they_ran() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 0), 0, "{}", a.issues());
    // local Tuesday (day 2) a public holiday
    let rows = [id as f64, 3.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0];
    a.set_line_rows(rows.to_vec());
    let (sat, hol) = (saturday_trips(&a), tuesday_trips(&a));
    assert_eq!(sat, hol, "the holiday runs the weekend's trains");
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&a.save()));
    b.set_line_rows(rows.to_vec());
    assert_eq!(b.w.net.rest_solved, a.w.net.rest_solved);
    assert_eq!(b.line_tph_rest(id), a.line_tph_rest(id));
    assert_eq!(b.line_weekend(id), a.line_weekend(id));
    assert_eq!(saturday_trips(&b), sat, "Saturday");
    assert_eq!(tuesday_trips(&b), hol, "the holiday");
}

/// A line's calendar (`set_line_rows`) makes a weekday a public holiday: that day runs the weekend
/// trains, and taking the holiday off puts the weekday's back.
#[test]
fn a_holiday_on_the_lines_calendar_runs_the_weekend_trains() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_line_weekend_counts(id, 8, 2, 2), 0, "{}", a.issues());
    let week = tuesday_trips(&a);
    a.set_line_rows(vec![id as f64, 3.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0]);
    let hol = tuesday_trips(&a);
    assert!(count(&hol) > count(&week), "{} trips on the holiday, {} on a Tuesday", count(&hol), count(&week));
    a.set_line_rows(vec![]);
    assert_eq!(tuesday_trips(&a), week);
}

/// A weekend-only line (no weekday trains) runs on Saturday alone, a weekday-only line (no weekend
/// trains) on Tuesday alone.
#[test]
fn weekend_only_and_weekday_only_lines_run_on_their_days() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_line_weekend_counts(id, 2, 2, 2), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 0, 0, 0), 0, "{}", a.issues());
    assert_eq!(count(&tuesday_trips(&a)), 0, "no trains on a Tuesday");
    assert!(count(&saturday_trips(&a)) > 0, "trains on a Saturday");
    let (mut b, id) = line_api();
    assert_eq!(b.set_line_weekend_counts(id, 0, 0, 0), 0, "{}", b.issues());
    assert!(count(&tuesday_trips(&b)) > 0, "trains on a Tuesday");
    assert_eq!(count(&saturday_trips(&b)), 0, "no trains on a Saturday");
}

/// A weekday count lowered below the weekend's frees no trains: the weekend still runs them.
#[test]
fn a_weekday_cut_below_the_weekend_keeps_the_fleet() {
    let (mut a, id) = line_api();
    let two = a.cars_needed();
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    let six = a.cars_needed();
    assert_eq!(six, two * 3);
    assert_eq!(a.set_train_counts(id, 1, 1, 1), 0, "{}", a.issues());
    assert_eq!(a.cars_needed(), six, "the weekend's six trains");
    assert_eq!(a.set_line_weekend_counts(id, 1, 1, 1), 0, "{}", a.issues());
    assert_eq!(a.cars_needed(), two / 2, "one train every day");
}

/// Weekend counts equal to the weekdays' or a share of 100% cost no settle of their own, and give
/// every day the weekday pattern (`rest_solved` off).
#[test]
fn weekend_counts_equal_to_the_weekdays_settle_nothing_more() {
    let (mut a, id) = line_api();
    let sat = saturday_trips(&a);
    assert_eq!(a.set_line_weekend_counts(id, 2, 2, 2), 0, "{}", a.issues());
    assert!(!a.w.net.rest_solved);
    assert!(!a.w.svc.lattice(&a.w.net, id as usize).unwrap().rest_on);
    assert_eq!(saturday_trips(&a), sat);
    assert_eq!(a.set_line_weekend_share(id, 100), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_weekend[id as usize], Default::default());
    assert_eq!(a.save(), save::encode(&a.w.net));
    assert!(!a.save().windows(4).any(|x| x == b"LWE1"), "the default writes no block");
}

/// Removing a line takes its weekend counts with it, and undo brings them back.
#[test]
fn a_removed_line_takes_its_weekend_counts() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    assert_eq!(a.remove_line(id), 0, "{}", a.issues());
    assert!(a.line_weekend(id).is_empty());
    assert!(!a.w.net.rest_solved, "no line left with weekends of its own");
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_weekend(id)[2..5], [6.0, 2.0, 2.0]);
    assert!(a.w.net.rest_solved);
}

/// One edit sets every line's weekends to a percent, counts set by hand dropped; an undo puts each
/// line back as it was.
#[test]
fn a_percent_for_every_line_is_one_edit() {
    let (mut a, id) = line_api();
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    assert_eq!(a.set_weekend_share_all(50), 0, "{}", a.issues());
    assert_eq!(a.line_weekend(id), vec![50.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!(a.step_is_service(false));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_weekend(id), vec![100.0, 1.0, 6.0, 2.0, 2.0, 0.0, 0.0, 0.0]);
}

/// T-3521: each kind of day's boarding account runs its own trains: a Saturday's and a Sunday's (a
/// public holiday's too) the weekend's, a weekday's and a superholiday's the weekday's.
#[test]
fn t3521_accounts_run_their_kinds_trains() {
    use crate::track::boarding::Boarding;
    assert_eq!([0, 1, 2, AWAY, 4, BACK].map(Boarding::kind_row), [WEEKDAYS, WEEKEND, WEEKEND, WEEKDAYS, WEEKDAYS, WEEKDAYS]);
    let (mut a, id) = line_api();
    let deps = |a: &TrackApi, row: usize| {
        let mut v = vec![];
        a.w.svc.run_deps_as(&a.w.net, id as usize, 0, 2.0 * 86400.0, 3.0 * 86400.0, Some(row), &mut v);
        v.len()
    };
    let (wk0, we0) = (deps(&a, WEEKDAYS), deps(&a, WEEKEND));
    assert_eq!(wk0, we0, "the same trains every day at 100%");
    assert_eq!(a.set_line_weekend_counts(id, 6, 2, 2), 0, "{}", a.issues());
    assert_eq!(deps(&a, WEEKDAYS), wk0, "the weekday account's trains as they were");
    assert!(deps(&a, WEEKEND) > we0, "{} weekend departures with six trains, {we0} with two", deps(&a, WEEKEND));
}

/// T-3521: a weekend's running cost is its own trains': the weekdays' to the bit at 100%, about half
/// at 50%, the weekdays' left as they were.
#[test]
fn t3521_running_cost_per_kind_of_day() {
    // two trains at every level, well within what the track takes (more are held, at the same rate)
    let (mut a, id) = line_api();
    let week: Vec<f64> = (0..3).map(|lev| a.running_cost(lev)).collect();
    for lev in 0..3 {
        assert_eq!(a.running_cost(3 + lev).to_bits(), week[lev].to_bits(), "level {lev} at 100%");
    }
    assert_eq!(a.set_line_weekend_share(id, 50), 0, "{}", a.issues());
    for lev in 0..3 {
        assert_eq!(a.running_cost(lev).to_bits(), week[lev].to_bits(), "level {lev}'s weekdays");
        let r = a.running_cost(3 + lev) / week[lev];
        assert!(r > 0.4 && r < 0.65, "level {lev}: a weekend costs {r} of a weekday at 50%");
    }
}

/// T-3528: a count set without history (the cap lowering on load) changes the counts but leaves the
/// undo and redo history as it was, so an undo takes back the player's edit before it.
#[test]
fn an_unrecorded_count_change_stays_off_the_undo_history() {
    let depth = |mut a: TrackApi| { let mut n = 0; while a.can_undo() && a.undo() == 0 { n += 1; } n };
    let (mut a, id) = line_api();
    let (mut b, idb) = line_api();
    for (t, i) in [(&mut a, id), (&mut b, idb)] {
        assert_eq!(t.set_train_counts(i, 3, 3, 3), 0, "{}", t.issues());
    }
    assert_eq!(b.set_train_counts_unrecorded(idb, 4, 4, 4), 0, "{}", b.issues());
    assert_eq!(depth(a), depth(b), "the lowering is not an undo step");
    let (mut c, idc) = line_api();
    assert_eq!(c.set_train_counts(idc, 3, 3, 3), 0, "{}", c.issues());
    assert_eq!(c.undo(), 0, "{}", c.issues());
    assert_eq!(c.set_train_counts_unrecorded(idc, 4, 4, 4), 0, "{}", c.issues());
    assert!(c.can_redo(), "the lowering must not clear the redo history");
}
