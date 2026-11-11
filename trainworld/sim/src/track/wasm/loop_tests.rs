//! T-2899: bus and ferry loop lines through the clock worker's API.

use super::*;
use crate::track::bus::{self, KIND_END, KIND_MIDDLE, KIND_STATION};
use crate::track::ferry::{K_PORT, K_ROUTE};
use crate::track::net::LoopKind;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    a
}

fn stops(a: &TrackApi, l: u32) -> Vec<Vec<f64>> {
    a.bus_stops(l).chunks_exact(8).map(|c| c.to_vec()).collect()
}

/// A square of `side` m from (x, y), as corners with the first point repeated at the end.
fn square(x: f64, y: f64, side: f64, closed: bool) -> Vec<f64> {
    let mut p = vec![x, y, 1.0, x + side, y, 1.0, x + side, y + side, 1.0, x, y + side, 1.0];
    if closed {
        p.extend([x, y, 1.0]);
    }
    p
}

#[test]
fn a_bus_loop_starts_and_ends_at_one_stop_and_runs_as_a_circle() {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &square(0.0, 0.0, 3000.0, true), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    assert!(s.len() >= 3, "{s:?}");
    assert_eq!(s[0][0], s.last().unwrap()[0], "the first stop is the last: {s:?}");
    assert_eq!((s[0][4], s.last().unwrap()[4]), (KIND_END as f64, KIND_END as f64));
    assert_eq!(s[0][3], 0.0);
    let road = 12000.0 * bus::ROAD_FACTOR;
    assert!((s.last().unwrap()[3] - road).abs() < 1e-6, "{s:?}");
    assert_eq!(a.line_loop(b), 1);
    assert_eq!(a.w.svc.lines[b as usize].kind, LoopKind::Circle);
    // the lap is one run and the dwell; a new bus loop runs one way (T-2981), its round trip the lap
    let t = a.stop_times(b, 0, 0);
    let lap = t[t.len() - 1] + 30.0;
    assert!(a.line_one_way(b));
    let ls = &a.w.svc.lines[b as usize];
    assert!((ls.round_trip[0] - lap).abs() < 1e-6, "{} {lap}", ls.round_trip[0]);
    // both ways: twice that, a bus each way
    assert_eq!(a.set_line_one_way(b, false), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[b as usize];
    assert!((ls.round_trip[0] - 2.0 * lap).abs() < 1e-6, "{} {lap}", ls.round_trip[0]);
    assert_eq!(a.stop_times(b, 1, 0).last(), t.last());
    assert!(ls.trains[0] % 2 == 0 && ls.trains[0] > 0, "{}", ls.trains[0]);
    // the same route open is an out-and-back line with two ends
    let o = a.new_line_id();
    assert_eq!(a.set_bus_line(o, &square(0.0, 0.0, 3000.0, false), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B2".into()), 0, "{}", a.issues());
    assert_eq!(a.line_loop(o), 0);
    let so = stops(&a, o);
    assert_ne!(so[0][0], so.last().unwrap()[0]);
    // a loop's cover reaches round its first stop from both sides
    assert!(s[0][6] == 0.0 && (s.last().unwrap()[7] - road).abs() < 1e-6, "{s:?}");
    // construct, save and load: the loop comes back
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    let saved = a.save();
    let before: Vec<u64> = a.bus_stops(b).into_iter().map(f64::to_bits).collect();
    assert!(a.load(&saved));
    assert_eq!(a.line_loop(b), 1);
    assert!(!a.line_one_way(b), "both ways is saved");
    assert_eq!(a.bus_stops(b).into_iter().map(f64::to_bits).collect::<Vec<_>>(), before);
}

#[test]
fn a_bus_loop_too_short_for_a_middle_stop_gets_one() {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &square(0.0, 0.0, 150.0, true), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    assert!(s.len() >= 3 && s.iter().any(|c| c[4] == KIND_MIDDLE as f64), "{s:?}");
}

#[test]
fn a_bus_loop_begins_at_a_station_near_its_closing_point() {
    let mut a = api();
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 500.0, 0.0, -1.0], &[], 2, false), 0, "{}", a.issues());
    let st = a.route_end();
    assert_eq!(a.add_station(1, st, 0.0, 0.0, 120, "S".into()), 0, "{}", a.issues());
    let b = a.new_line_id();
    // 80 m from the station
    assert_eq!(a.set_bus_line(b, &square(500.0, 80.0, 2000.0, true), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    assert_eq!((s[0][4], s[0][5]), (KIND_STATION as f64, st as f64), "{s:?}");
    assert_eq!(s[0][0], s.last().unwrap()[0]);
    assert_eq!(s.iter().filter(|c| c[5] == st as f64).count(), 2, "the station once at each end: {s:?}");
}

/// Each stop of a bus line as (x, y), run 0's order, to the metre.
fn places(a: &TrackApi, l: u32) -> Vec<(i64, i64)> {
    stops(a, l).iter().map(|c| (c[1].round() as i64, c[2].round() as i64)).collect()
}

/// T-3123: stops by hand on a ring: the first placed one is a stop with the stop the ring had at half its length
/// kept, a stop taken out goes, and the last stop between stays. Every stop is served each way the bus runs.
#[test]
fn stops_by_hand_on_a_ring_add_and_go() {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &square(0.0, 0.0, 3000.0, true), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 3000), (0, 0)], "half way round, as before");
    assert_eq!(a.set_bus_stop(b, 3010.0, 1000.0, true), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 1000), (3000, 3000), (0, 0)]);
    assert_eq!(a.set_bus_stop(b, 1000.0, 2990.0, true), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 1000), (3000, 3000), (1000, 3000), (0, 0)]);
    assert_eq!(a.stop_times(b, 0, 0).len(), 5, "one way: each stop once");
    assert_eq!(a.set_line_one_way(b, false), 0, "{}", a.issues());
    assert_eq!(a.stop_times(b, 1, 0).len(), 5, "both ways: each stop the other way too");
    // near the closing stop, or on a stop: nothing
    assert_eq!(a.set_bus_stop(b, 20.0, 0.0, true), 0);
    assert_eq!(a.set_bus_stop(b, 3000.0, 1020.0, true), 0);
    assert_eq!(places(&a, b).len(), 5);
    // out, out, and the last one stays
    assert_eq!(a.set_bus_stop(b, 3000.0, 1000.0, false), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 1000.0, 3000.0, false), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 3000), (0, 0)]);
    assert_eq!(a.set_bus_stop(b, 3000.0, 3000.0, false), 0);
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 3000), (0, 0)]);
    assert_eq!(a.undo(), 0);
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 3000), (1000, 3000), (0, 0)], "one undo step per edit");
    // automatic and back to by hand on the loop
    assert_eq!(a.set_bus_spacing(b, 0.0), 0, "{}", a.issues());
    assert!(!a.bus_hand(b));
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 3000), (1000, 3000), (0, 0)]);
}

/// T-3123: stops by hand on a route that loops back: one placed on the loop is a stop once, on the way out a stop
/// on the way out and on the way back; taking either out takes both.
#[test]
fn stops_by_hand_on_a_route_that_loops_back_serve_the_way_out_both_ways() {
    let mut a = api();
    let b = a.new_line_id();
    // out 2 km, round a 1 km square and back
    let route = [0.0, 0.0, 1.0, 2000.0, 0.0, 1.0, 3000.0, 0.0, 1.0, 3000.0, 1000.0, 1.0, 2000.0, 1000.0, 1.0, 2000.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 1000), (0, 0)]);
    assert_eq!(a.set_bus_stop(b, 3000.0, 400.0, true), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 400), (3000, 1000), (0, 0)], "on the loop: once");
    assert_eq!(a.set_bus_stop(b, 1000.0, 10.0, true), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (1000, 0), (3000, 400), (3000, 1000), (1000, 0), (0, 0)], "on the way out: both passes");
    assert_eq!(a.stop_times(b, 0, 0).len(), 6);
    assert_eq!(a.set_bus_stop(b, 1000.0, 0.0, false), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 400), (3000, 1000), (0, 0)], "both passes go");
    assert_eq!(a.set_bus_stop(b, 3000.0, 1000.0, false), 0, "{}", a.issues());
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 400), (0, 0)]);
    assert_eq!(a.set_bus_stop(b, 3000.0, 400.0, false), 0);
    assert_eq!(places(&a, b), vec![(0, 0), (3000, 400), (0, 0)], "the last stop between stays");
}

fn port(a: &mut TrackApi, x: f64, y: f64, name: &str) -> u32 {
    assert_eq!(a.add_port(x, y - 100.0, x, y, name.into()), 0, "{}", a.issues());
    a.last_port()
}

#[test]
fn a_ferry_loop_names_its_first_port_twice_and_pays_for_it_once() {
    let mut a = api();
    let (p, q, r) = (port(&mut a, 0.0, 0.0, "A"), port(&mut a, 4000.0, 0.0, "B"), port(&mut a, 2000.0, 3000.0, "C"));
    let route = [0.0, 0.0, K_PORT as f64, 8.0, 4000.0, 0.0, K_PORT as f64, 8.0, 2000.0, 3000.0, K_PORT as f64, 8.0, 1000.0, 1500.0, K_ROUTE as f64, 8.0, 0.0, 0.0, K_PORT as f64, 8.0];
    let l = a.new_line_id();
    assert_eq!(a.set_ferry_line(l, &[p, q, r, p], &route, 900, 1200, 1800, 60.0, 300.0, 0x1f77b4, "Ferry 1".into()), 0, "{}", a.issues());
    assert_eq!(a.line_loop(l), 1);
    assert_eq!(a.line_stops(l), vec![p, q, r, p]);
    let ls = &a.w.svc.lines[l as usize];
    assert_eq!(ls.kind, LoopKind::Circle);
    let t = a.stop_times(l, 0, 0);
    assert_eq!(t.len(), 4);
    assert!(ls.one_way, "a new ferry loop runs one way (T-2981)");
    assert!((ls.round_trip[0] - (t[3] + 60.0)).abs() < 1e-6, "{} {t:?}", ls.round_trip[0]);
    assert_eq!(a.set_line_one_way(l, false), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[l as usize];
    assert!((ls.round_trip[0] - 2.0 * (t[3] + 60.0)).abs() < 1e-6, "{} {t:?}", ls.round_trip[0]);
    // three ports to pay for, not four
    assert!((a.line_build_cost(l) - 3.0 * PORT_COST).abs() < 1e-9, "{}", a.line_build_cost(l));
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    assert!(a.w.svc.lines[l as usize].running);
    let saved = a.save();
    assert!(a.load(&saved));
    assert_eq!(a.line_loop(l), 1);
    assert_eq!(a.line_stops(l), vec![p, q, r, p]);
    // the same ports in a row without the closing one are an out-and-back line
    let o = a.new_line_id();
    let open = &route[..12];
    assert_eq!(a.set_ferry_line(o, &[p, q, r], open, 900, 1200, 1800, 60.0, 300.0, 0x1f77b4, "Ferry 2".into()), 0, "{}", a.issues());
    assert_eq!(a.line_loop(o), 0);
}

/// T-3142: `[id, heading]` per stop a line serves one way.
fn ways(a: &TrackApi, l: u32) -> Vec<(u32, f64)> {
    a.line_stop_ways(l).chunks_exact(2).map(|c| (c[0] as u32, c[1])).collect()
}

/// The ids of a bus line's stops at (x, y), to the metre.
fn ids_at(a: &TrackApi, l: u32, x: i64, y: i64) -> Vec<u32> {
    stops(a, l).iter().filter(|c| (c[1].round() as i64, c[2].round() as i64) == (x, y)).map(|c| c[0] as u32).collect()
}

#[test]
fn a_one_way_bus_ring_serves_every_stop_one_way_and_a_two_way_one_none() {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &square(0.0, 0.0, 3000.0, true), 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 3010.0, 1000.0, true), 0, "{}", a.issues());
    let w = ways(&a, b);
    let s = ids_at(&a, b, 3000, 1000)[0];
    let h = w.iter().find(|x| x.0 == s).expect("the stop on the east side is one way").1;
    assert!((h - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "heading +y: {h}");
    assert_eq!(w.len(), 3, "the closing stop once, the half way one and the placed one: {w:?}");
    assert_eq!(a.set_line_one_way(b, false), 0, "{}", a.issues());
    assert!(ways(&a, b).is_empty(), "run both ways, every stop is both");
}

#[test]
fn a_bus_route_that_loops_back_serves_its_loop_one_way_and_its_way_out_both() {
    let mut a = api();
    let b = a.new_line_id();
    let route = [0.0, 0.0, 1.0, 2000.0, 0.0, 1.0, 3000.0, 0.0, 1.0, 3000.0, 1000.0, 1.0, 2000.0, 1000.0, 1.0, 2000.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 3000.0, 400.0, true), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 1000.0, 10.0, true), 0, "{}", a.issues());
    let w: Vec<u32> = ways(&a, b).into_iter().map(|x| x.0).collect();
    assert!(ids_at(&a, b, 3000, 400).iter().all(|i| w.contains(i)), "the loop's stop is one way: {w:?}");
    assert!(ids_at(&a, b, 1000, 0).iter().all(|i| !w.contains(i)), "the way out is both ways: {w:?}");
    assert!(ids_at(&a, b, 0, 0).iter().all(|i| !w.contains(i)), "the end the bus turns at is both ways: {w:?}");
}

#[test]
fn a_one_way_ferry_loop_serves_every_port_one_way() {
    let mut a = api();
    let (p, q, r) = (port(&mut a, 0.0, 0.0, "A"), port(&mut a, 4000.0, 0.0, "B"), port(&mut a, 2000.0, 3000.0, "C"));
    let route = [0.0, 0.0, K_PORT as f64, 8.0, 4000.0, 0.0, K_PORT as f64, 8.0, 2000.0, 3000.0, K_PORT as f64, 8.0, 1000.0, 1500.0, K_ROUTE as f64, 8.0, 0.0, 0.0, K_PORT as f64, 8.0];
    let l = a.new_line_id();
    assert_eq!(a.set_ferry_line(l, &[p, q, r, p], &route, 900, 1200, 1800, 60.0, 300.0, 0x1f77b4, "Ferry 1".into()), 0, "{}", a.issues());
    let w = ways(&a, l);
    assert_eq!(w.iter().map(|x| x.0).collect::<Vec<_>>(), vec![p, q, r]);
    assert!(w[1].1.abs() > 1.0 && (w[1].1 - (3000.0f64).atan2(-2000.0)).abs() < 1e-6, "B heads on to C: {}", w[1].1);
    assert_eq!(a.set_line_one_way(l, false), 0, "{}", a.issues());
    assert!(ways(&a, l).is_empty());
}

/// T-3131: a route drawn out and back by hand along one road (the passes 15 m apart, no loop-back) gets a hand stop
/// on both passes; a stop on the lone far end stays one; taking either pass out takes both.
#[test]
fn stops_by_hand_on_a_route_drawn_out_and_back_serve_both_passes() {
    let mut a = api();
    let b = a.new_line_id();
    let route = [0.0, 0.0, 1.0, 3000.0, 0.0, 1.0, 3000.0, 15.0, 1.0, 0.0, 15.0, 1.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 1000.0, 5.0, true), 0, "{}", a.issues());
    let at = places(&a, b);
    assert_eq!(at.iter().filter(|p| p.0 == 1000).count(), 2, "{at:?}");
    assert_eq!(a.set_bus_stop(b, 1000.0, 5.0, false), 0, "{}", a.issues());
    assert!(places(&a, b).iter().all(|p| p.0 != 1000), "both passes go");
}
