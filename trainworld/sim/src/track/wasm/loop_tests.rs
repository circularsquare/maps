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
    // the lap is one run and the dwell; the round trip twice that, both ways with a bus each
    let t = a.stop_times(b, 0, 0);
    let ls = &a.w.svc.lines[b as usize];
    let lap = t[t.len() - 1] + 30.0;
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
