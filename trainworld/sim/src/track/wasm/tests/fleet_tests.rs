use super::*;

#[test]
fn auto_trains_is_the_fewest_that_hold_the_headway() {
    assert_eq!(auto_trains(900.0, 300), 3);
    assert_eq!(auto_trains(901.0, 300), 3, "half a percent over is the same headway");
    assert_eq!(auto_trains(905.0, 300), 4);
    assert_eq!(auto_trains(900.0, 0), 0);
    assert_eq!(auto_trains(0.0, 300), 0);
}

#[test]
fn clamp_auto_binds_only_at_the_limit() {
    use crate::track::world::clamp_auto;
    // a 4 minute limit (15 trains an hour) on a 2,000 s round trip: 8 trains fit, 9 do not
    assert_eq!(auto_trains(2000.0, 240), 9);
    assert_eq!(clamp_auto(9, 2000.0, 240, 15.0), 8, "at the limit, the count that fits");
    assert_eq!(clamp_auto(auto_trains(2000.0, 239), 2000.0, 239, 15.0), 8, "half a percent under is still the limit");
    assert_eq!(clamp_auto(auto_trains(2000.0, 238), 2000.0, 238, 15.0), 9, "past the slack the count is the player's");
    assert_eq!(clamp_auto(17, 2000.0, 120, 15.0), 17, "well under: unclamped, so the panel warns");
    assert_eq!(clamp_auto(auto_trains(2000.0, 300), 2000.0, 300, 15.0), 7, "a slower headway is untouched");
    assert_eq!(clamp_auto(auto_trains(2400.0, 240), 2400.0, 240, 15.0), 10, "an exact fit keeps its last train");
    // lines with one or two trains
    assert_eq!(clamp_auto(auto_trains(300.0, 240), 300.0, 240, 15.0), 1, "two would cross the limit");
    assert_eq!(clamp_auto(1, 200.0, 240, 15.0), 1, "at least one train runs");
    assert_eq!(clamp_auto(1, 900.0, 1800, 15.0), 1);
    assert_eq!(clamp_auto(2, 900.0, 450, 15.0), 2);
    // nothing limits the line, or no room is left: no clamp
    assert_eq!(clamp_auto(9, 2000.0, 240, f64::INFINITY), 9);
    assert_eq!(clamp_auto(9, 2000.0, 240, 0.0), 9);
    assert_eq!(clamp_auto(0, 2000.0, 0, 15.0), 0);
}

/// A 6 km double-track line between two dead-end termini with terminals, turning in 420 s: each terminus
/// pools its two platform tracks, (420 + 60) / 2 = 240 s a train, so the line's limit is 4:00.
fn four_minute_line() -> (TrackApi, u32) {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    crate::track::tests::terminals_at_dead_ends(&mut a.w); // both platforms at each end (T-2986)
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 2, 2, 2, 30.0, 420.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (a, id)
}

#[test]
fn full_track_lists_the_edges_of_full_sections() {
    // T-819: with a quick turnaround the open track is what fills first
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    crate::track::tests::terminals_at_dead_ends(&mut a.w); // both platforms at each end (T-2986)
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 60, 2, 2, 30.0, 0.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let full = a.full_track(0);
    assert!(!full.is_empty() && full.chunks(3).all(|c| c[1] >= FULL_RHO && c[2] > 20.0), "{full:?}");
    assert!(a.full_track(1).is_empty(), "two trains leave room: {:?}", a.full_track(1));
}

#[test]
fn a_headway_at_the_track_limit_holds_no_train_out() {
    let (mut a, id) = four_minute_line();
    let limit = &a.line_held(id)[7 * ROWS..7 * ROWS + 5];
    assert!((limit[0] - 240.0).abs() < 0.01, "the terminus limits the line to 4:00: {limit:?}");
    assert_eq!(limit[1], 3.0, "named as a terminus");
    let trip = a.w.svc.lines[id as usize].round_trip[0];
    let up = auto_trains(trip, 240);
    assert!(trip / (up as f64) < 240.0 && trip / (up - 1) as f64 >= 240.0, "the fixture rounds up across the limit: {trip}");

    // the cause: the rounded up count by hand holds part of a train out at the terminus
    assert_eq!(a.set_train_counts(id, up, 1, 1), 0, "{}", a.issues());
    assert!(a.line_held(id)[0] > 0.0, "rounding up crosses the limit");

    // a headway typed at the limit takes the trains that fit, and none waits (the switch to
    // headways first, which keeps the trains, T-752)
    assert_eq!(a.set_auto_headway(id, 300, 480, 0), 0, "{}", a.issues());
    assert_eq!(a.w.svc.lines[id as usize].trains[0], up, "the switch keeps the trains");
    assert_eq!(a.set_auto_headway(id, 240, 480, 0), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert_eq!(s.trains[0], up - 1, "clamped at the limit");
    assert!(s.round_trip[0] / s.trains[0] as f64 >= 240.0 - 1e-6, "the real headway is just over the typed one");
    assert_eq!(s.trains[1], auto_trains(s.round_trip[1], 480), "a headway over the limit is untouched");
    assert!(a.line_held(id)[..7 * ROWS].iter().all(|&v| v == 0.0), "no train held out: {:?}", a.line_held(id));
    assert_eq!(a.line_auto(id), vec![240.0, 480.0, 0.0], "the typed headway stays");

    // well under the limit the count is the player's, and the panel warns
    assert_eq!(a.set_auto_headway(id, 120, 480, 0), 0, "{}", a.issues());
    assert!(a.w.svc.lines[id as usize].trains[0] >= up, "the whole trains that wait are dropped, the last part train is not");
    assert!(a.line_held(id)[0] > 0.0, "trains held out at 2:00 against a 4:00 limit");
}

/// T-2935: the dot marks trains held back only when a whole train is out of service.
#[test]
fn a_part_train_is_not_marked_as_held() {
    let (mut a, id) = four_minute_line();
    let trip = a.w.svc.lines[id as usize].round_trip[0];
    let up = auto_trains(trip, 240);
    let marked = |a: &TrackApi| a.markers(0).chunks(7).any(|m| m[6] > 0.0);
    assert_eq!(a.set_train_counts(id, up, 1, 1), 0, "{}", a.issues());
    assert!(a.line_held(id)[0] > 0.0 && a.line_held(id)[0] < 0.99, "part of a train past the limit: {}", a.line_held(id)[0]);
    assert!(!marked(&a));
    assert_eq!(a.set_train_counts(id, up + 2, 1, 1), 0, "{}", a.issues());
    assert!(a.line_held(id)[0] >= 1.0);
    assert!(marked(&a));
}

#[test]
fn headway_lines_follow_their_round_trip() {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    let e = a.edges_info()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    crate::track::tests::terminals_at_dead_ends(&mut a.w); // both platforms at each end (T-2986)
    assert_eq!(a.add_station(2, e, 3000.0, 30.0, 200, "Middle".into()), 0, "{}", a.issues());
    let mid = a.nodes_info().chunks(10).find(|c| (c[1] - 3000.0).abs() < 1.0).unwrap()[0] as u32;
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 2, 2, 2, 30.0, 180.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.line_auto(id).is_empty());
    let manual = slots_to_rows(a.w.svc.lines[id as usize].trains);

    // a headway that takes three trains over the round trip as it is now
    let rt0 = a.w.svc.lines[id as usize].round_trip[0];
    let h = (rt0 / 3.0).ceil() as u32;
    assert_eq!(a.set_auto_headway(id, 2 * h, 2 * h, 2 * h), 0, "{}", a.issues());
    assert_eq!(slots_to_rows(a.w.svc.lines[id as usize].trains), manual, "the switch to headways keeps the trains (T-752)");
    assert_eq!(a.set_auto_headway(id, h, 0, 2 * h), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert_eq!(s.trains[0], 3);
    assert_eq!(s.trains[1], 0, "0 seconds is no service");
    assert!(s.round_trip[0] / s.trains[0] as f64 <= h as f64 * 1.005 + 1e-6, "headway held");
    assert!(s.round_trip[0] / (s.trains[0] - 1) as f64 > h as f64 * 1.005, "and with the fewest trains");
    assert_eq!(a.line_auto(id), vec![h as f64, 0.0, 2.0 * h as f64]);
    assert_eq!(a.fleet(), a.cars_needed(), "growing the fleet bought the cars");

    // a stop in the middle lengthens the round trip, and the count follows it
    let cash = a.cash();
    let cars = a.w.svc.lines[id as usize].cars;
    assert_eq!(a.set_stops(id, &[n0, mid, n1]), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert!(s.round_trip[0] > rt0);
    assert_eq!(s.trains[0], 4);
    assert!(s.round_trip[0] / s.trains[0] as f64 <= h as f64 * 1.005 + 1e-6);
    assert_eq!(a.line_auto(id).len(), 3, "an edit to the stops keeps the headways");
    assert!((cash - a.cash() - cars as f64 * CAR_PRICE).abs() < 1e-6, "one more train, bought with the edit");

    // short of money the edit is refused and nothing moves
    let (fleet, trains) = (a.fleet(), a.w.svc.lines[id as usize].trains);
    a.set_cash(1.0);
    assert_eq!(a.set_auto_headway(id, h / 2, 0, 2 * h), 1, "{}", a.issues());
    assert!(a.issue_kinds().starts_with("Money:"), "{}", a.issue_kinds());
    assert_eq!((a.fleet(), a.w.svc.lines[id as usize].trains), (fleet, trains));
    assert_eq!(a.line_auto(id)[0], h as f64, "the old headways stay");
    // a longer headway needs fewer trains; the cars it frees stay owned
    assert_eq!(a.set_auto_headway(id, 2 * h, 0, 2 * h), 0, "{}", a.issues());
    assert!(a.w.svc.lines[id as usize].trains[0] < trains[0]);
    assert_eq!(a.fleet(), fleet, "cars already owned are kept");

    // the headways are saved; a manual line is saved as before
    a.set_cash(6000.0);
    let snapshot = a.save();
    let mut b = api();
    assert!(b.load(&snapshot));
    assert_eq!(b.line_auto(0), vec![2.0 * h as f64, 0.0, 2.0 * h as f64]);
    assert_eq!(b.w.svc.lines[0].trains, a.w.svc.lines[id as usize].trains);

    // by hand again: the counts stay as they were, and undo goes back to headways
    assert_eq!(a.clear_auto_headway(id), 0);
    assert!(a.line_auto(id).is_empty());
    assert!(!a.save().windows(4).any(|w| w == b"LAU1"));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_auto(id).len(), 3);
    assert_eq!(a.set_train_counts(id, 5, 5, 5), 0);
    assert!(a.line_auto(id).is_empty(), "a hand-set count ends the headways");
    assert_eq!(slots_to_rows(a.w.svc.lines[id as usize].trains), [5, 5, 5]);
    assert_ne!(manual, [5, 5, 5]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_auto(id).len(), 3, "undo brings the headways back");
}

/// T-817: high-speed trains per line, their own fleet, prices, running costs and saves.
#[test]
fn high_speed_trains_are_a_fleet_of_their_own() {
    let mut a = api();
    a.set_cash(1e6);
    a.set_route_hsr(true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(30000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Line 1".into()), 0, "{}", a.issues());
    assert!(!a.line_fast(id));
    // a planned line's train quote follows its kind
    let ordinary_quote = a.line_train_cost(id);
    assert_eq!(a.set_line_fast(id, true), 0, "{}", a.issues());
    assert!((a.line_train_cost(id) - ordinary_quote * 1.5).abs() < 1e-9);
    assert_eq!(a.set_line_fast(id, false), 0);
    assert_eq!(a.construct_line(id), 0, "{}", a.issues());
    let need = a.cars_needed();
    assert!(need > 0);
    assert_eq!((a.fleet(), a.fleet_fast(), a.cars_needed_fast()), (need, 0, 0));
    let slow = a.stop_times(id, 0, 0)[1];
    let speeds = a.line_stretch_speeds(id);
    assert!((speeds[0] - 160.0).abs() < 1e-6 && (speeds[1] - 160.0).abs() < 1e-6 && speeds[2] == 0.0, "{speeds:?}");
    // choosing high-speed trains buys the cars at 3.75 each; the ordinary ones stay as spares
    let cash = a.cash();
    assert_eq!(a.set_line_fast(id, true), 0, "{}", a.issues());
    assert!(a.line_fast(id));
    let hs = a.cars_needed_fast();
    assert_eq!((a.cars_needed(), a.fleet(), a.fleet_fast()), (0, need, hs));
    assert!((cash - a.cash() - hs as f64 * HS_CAR_PRICE).abs() < 1e-6, "paid {}", cash - a.cash());
    assert!(a.stop_times(id, 0, 0)[1] < slow * 0.75, "{} against {slow}", a.stop_times(id, 0, 0)[1]);
    let speeds = a.line_stretch_speeds(id);
    assert!((speeds[0] - 320.0).abs() < 1e-6 && (speeds[1] - 320.0).abs() < 1e-6 && speeds[2] == 0.0, "{speeds:?}");
    // running costs 5 a train-km plus 3 a car-km
    let info = a.lines_info();
    let (len_km, tph, cars) = (info[9] / 1000.0, info[5], info[21]);
    let tk = tph * 2.0 * len_km;
    assert!((a.running_cost(0) - (5.0 * tk + 3.0 * tk * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // back to ordinary trains and forward again: both kinds are owned, nothing is paid
    let cash = a.cash();
    assert_eq!(a.set_line_fast(id, false), 0);
    assert_eq!((a.fleet(), a.fleet_fast(), a.cash()), (need, hs, cash));
    let tk = a.lines_info()[5] * 2.0 * len_km;
    assert!((a.running_cost(0) - (4.0 * tk + 2.0 * tk * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // the spare high-speed cars never serve an ordinary line that grows
    assert_eq!(a.set_train_counts(id, 6, 4, 2), 0, "{}", a.issues());
    assert!((cash - a.cash() - (a.cars_needed() - need) as f64 * CAR_PRICE).abs() < 1e-6);
    assert_eq!(a.fleet_fast(), hs);
    assert_eq!(a.set_line_fast(id, true), 0);
    // saves keep the kind; a save without the block loads as ordinary
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LHS1").expect("LHS1 written");
    let mut b = api();
    assert!(b.load(&bytes));
    assert!(b.line_fast(0));
    assert_eq!(b.stop_times(0, 0, 0), a.stop_times(id, 0, 0));
    assert!(save::decode(&bytes[..at]).map(|n| !n.line_fast[0]).unwrap());
    assert_eq!(a.set_line_fast(id, false), 0);
    assert!(!a.save().windows(4).any(|w| w == b"LHS1"));
}

/// T-3655: on London's shared save (`data/work/realnet/lon_real.twt2`; skipped where it is not
/// there), its busiest line, held on full track, takes its share at each new headway or count, up
/// and down: the same share at the same fleet whichever way it came, no line beside it swinging,
/// in at most 20 settle rounds an edit (a stepped level runs to hundreds); undo and redo put back the exact shares.
#[test]
fn a_new_count_or_headway_on_full_track_takes_its_share() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/work/realnet/lon_real.twt2");
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("skipped: no {path}");
        return;
    };
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert!(a.load(&bytes));
    a.set_unlimited(true);
    let medium_all = |a: &TrackApi| -> Vec<(u32, f64)> { a.lines_info().chunks(22).map(|c| (c[0] as u32, c[6])).collect() };
    let line = a.lines_info().chunks(22).max_by_key(|c| c[14] as u32).map(|c| c[0] as u32).unwrap();
    let medium = |a: &TrackApi| medium_all(a).iter().find(|r| r.0 == line).unwrap().1;
    let state = |a: &TrackApi| crate::track::save::encode(&a.world().net);
    let mut now = 111600.0;
    // the settle rounds an edit runs (`SETTLE_UPDATES`), each a service update of every line
    let mut edit = |a: &mut TrackApi, f: &dyn Fn(&mut TrackApi) -> u32| -> u32 {
        now += 30.0;
        a.set_now(now);
        crate::track::world::SETTLE_UPDATES.with(|c| c.set(0));
        assert_eq!(f(a), 0);
        crate::track::world::SETTLE_UPDATES.with(|c| c.get())
    };
    // `ups` then `downs` edits, each giving (the line's medium rate, every line's, rounds)
    let sweep = |a: &mut TrackApi, edit: &mut dyn FnMut(&mut TrackApi, &dyn Fn(&mut TrackApi) -> u32) -> u32, set: &dyn Fn(&mut TrackApi, u32) -> u32, steps: &[u32]| {
        steps.iter().map(|&s| {
            let r = edit(a, &|a| set(a, s));
            (medium(a), medium_all(a), r)
        }).collect::<Vec<_>>()
    };
    // the same fleet up and down: every line within 1% (the pacing's own noise is up to 3%, T-3388)
    let same = |x: &[(u32, f64)], y: &[(u32, f64)], what: &str| {
        for (p, q) in x.iter().zip(y) {
            assert!((p.1 - q.1).abs() <= 0.01 * p.1.max(q.1) + 1e-3, "{what}: line {} {:.4} going up, {:.4} coming down", p.0, p.1, q.1);
        }
    };
    // no line beside it turns back as the line's fleet keeps growing (or shrinking)
    let no_swing = |runs: &[(f64, Vec<(u32, f64)>, u32)], what: &str| {
        for (k, &(l, _)) in runs[0].1.iter().enumerate().filter(|(_, x)| x.0 != line) {
            let v: Vec<f64> = runs.iter().map(|r| r.1[k].1).collect();
            let d: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).filter(|d| d.abs() > 0.005 * v[0].max(0.1)).collect();
            assert!(d.iter().all(|&x| x > 0.0) || d.iter().all(|&x| x < 0.0), "{what}: line {l} swings: {v:?}");
        }
    };
    let headway = |a: &mut TrackApi, m: u32| a.set_auto_headway(line, 300, m, 1200);
    edit(&mut a, &|a| headway(a, 600));
    let up = sweep(&mut a, &mut edit, &headway, &[540, 480, 420, 360]);
    let down = sweep(&mut a, &mut edit, &headway, &[420, 480, 540]);
    let rates: Vec<f64> = up.iter().chain(&down).map(|r| r.0).collect();
    assert!(up.windows(2).all(|w| w[1].0 > w[0].0 + 0.05), "a shorter headway runs more often: {rates:?}");
    assert!(down.windows(2).all(|w| w[1].0 < w[0].0 - 0.05), "a longer one less often: {rates:?}");
    assert!(up[3].0 > 1.3 * up[0].0, "360 s runs well over 540 s: {rates:?}");
    for (k, d) in down.iter().enumerate() {
        same(&up[2 - k].1, &d.1, &format!("headway {}", [420, 480, 540][k]));
    }
    no_swing(&up, "headways going up");
    no_swing(&down, "headways coming down");
    assert!(up.iter().chain(&down).all(|r| r.2 <= 20), "settle rounds an edit: {:?}", up.iter().chain(&down).map(|r| r.2).collect::<Vec<_>>());
    // counts, after the switch back
    edit(&mut a, &|a| a.clear_auto_headway(line));
    let counts = |a: &mut TrackApi, n: u32| a.set_train_counts(line, 20, n, 4);
    edit(&mut a, &|a| counts(a, 10));
    let up = sweep(&mut a, &mut edit, &counts, &[11, 12, 13, 14]);
    let down = sweep(&mut a, &mut edit, &counts, &[13, 12, 11]);
    let rates: Vec<f64> = up.iter().chain(&down).map(|r| r.0).collect();
    assert!(up.windows(2).all(|w| w[1].0 > w[0].0 + 0.05), "more trains run more often: {rates:?}");
    assert!(down.windows(2).all(|w| w[1].0 < w[0].0 - 0.05), "fewer less often: {rates:?}");
    for (k, d) in down.iter().enumerate() {
        same(&up[2 - k].1, &d.1, &format!("{} trains", [13, 12, 11][k]));
    }
    no_swing(&up, "counts going up");
    no_swing(&down, "counts coming down");
    assert!(up.iter().chain(&down).all(|r| r.2 <= 20), "settle rounds an edit: {:?}", up.iter().chain(&down).map(|r| r.2).collect::<Vec<_>>());
    // undo and redo put back the shares exactly, a headway's and a count's
    let changes: [&dyn Fn(&mut TrackApi) -> u32; 2] = [&|a| headway(a, 420), &|a| counts(a, 14)];
    for f in changes {
        let before = state(&a);
        edit(&mut a, f);
        let after = state(&a);
        assert!(after != before);
        assert_eq!(a.undo(), 0);
        assert!(state(&a) == before, "undo lands elsewhere");
        assert_eq!(a.redo(), 0);
        assert!(state(&a) == after, "redo lands elsewhere");
    }
}
