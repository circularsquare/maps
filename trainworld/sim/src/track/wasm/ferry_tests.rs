//! T-2193: ports and ferry lines through the clock worker's API.

use super::*;
use crate::track::boarding::{PERIODS, QUEUE_STRIDE};
use crate::track::ferry::{BOAT_A, K_PORT, K_ROUTE, K_WAYPOINT};
use crate::track::upgrades;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    a
}

/// A port with its landing 100 m south of its berth at (x, 0).
fn port(a: &mut TrackApi, x: f64, name: &str) -> u32 {
    assert_eq!(a.add_port(x, -100.0, x, 0.0, name.into()), 0, "{}", a.issues());
    a.last_port()
}

/// A straight route along y = 0 through `xs`, the given ones ports, every point at `kmh`.
fn straight(xs: &[f64], ports: &[usize], kmh: f64) -> Vec<f64> {
    xs.iter().enumerate().flat_map(|(i, &x)| [x, 0.0, if ports.contains(&i) { K_PORT } else { K_ROUTE } as f64, kmh / 3.6]).collect()
}

fn ferry_line(a: &mut TrackApi, ports: &[u32], route: &[f64]) -> u32 {
    let l = a.new_line_id();
    assert_eq!(a.set_ferry_line(l, ports, route, 900, 1200, 1800, 60.0, 300.0, 0x1f77b4, "Ferry 1".into()), 0, "{}", a.issues());
    l
}

/// Rows of `blueprint_cost_items` of a kind: (count, US$M).
fn row(a: &mut TrackApi, kind: f64) -> Option<(f64, f64)> {
    a.blueprint_cost_items().chunks_exact(7).find(|r| r[0] == kind).map(|r| (r[5], r[6]))
}

#[test]
fn a_ferry_line_runs_its_boats_at_its_speeds_and_pays_ports_and_boats() {
    let mut a = api();
    let (p, q) = (port(&mut a, 0.0, "Whitehall"), port(&mut a, 8000.0, "St George"));
    assert!(a.is_port(p) && a.ports() == vec![p, q] && a.port_berth(q) == vec![8000.0, 0.0]);
    let cash = a.cash();
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 8000.0], &[0, 1], 25.0));
    assert!(a.is_ferry(l) && !a.is_bus(l) && a.ferry_lines() == vec![l]);
    assert_eq!(a.line_stops(l), vec![p, q]);
    // planned: nothing paid, nothing running; the quote has two ports, the free route and the boats
    assert!(!a.w.svc.lines[l as usize].running && a.cash() == cash);
    assert_eq!(row(&mut a, 12.0), Some((2.0, 2.0 * PORT_COST)));
    assert_eq!(row(&mut a, 14.0), Some((8000.0, 0.0)));
    // the run: 8 km at 25 km/h, one acceleration and one braking at 0.15 m/s²
    let v = crate::track::ferry::quant_v(25.0 / 3.6) as f64;
    let run = 8000.0 / v + v / BOAT_A;
    let t = a.stop_times(l, 0, 0);
    assert!((t[1] - run).abs() < 1e-6, "{t:?} {run}");
    assert!((a.stop_times(l, 1, 0)[1] - run).abs() < 1e-6);
    let q8 = a.ferry_quote(u32::MAX, &[p, q], &straight(&[0.0, 8000.0], &[0, 1], 25.0), 60.0);
    assert!((q8[0] - 8000.0).abs() < 1e-9 && (q8[1] - run).abs() < 1e-6 && q8[2] == 2.0 * PORT_COST && q8[3] == 1.0, "{q8:?}");
    // boats: the round trip (two runs and two turnarounds of 300 s) over the 15 minute headway
    let boats = auto_trains(2.0 * (run + 300.0), 900);
    assert_eq!(a.w.svc.lines[l as usize].trains[0], boats);
    assert_eq!(row(&mut a, 13.0), Some((boats as f64, boats as f64 * BOAT_PRICE)));
    let quote = a.line_build_cost(l);
    assert!((quote - 2.0 * PORT_COST).abs() < 1e-9, "{quote}");
    // construct: the ports at $10M and the boats at $8M, out of the boat fleet only
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[l as usize];
    assert!(ls.running && ls.ferry && ls.crush() == BOAT_CRUSH);
    assert!((cash - a.cash() - (2.0 * PORT_COST + boats as f64 * BOAT_PRICE)).abs() < 1e-6, "{}", cash - a.cash());
    assert_eq!((a.fleet_ferry(), a.boats_needed(), a.fleet(), a.fleet_bus()), (boats, boats, 0, 0));
    // running: a Medium boat's rate a boat-km at the economy scale
    let hour = a.w.net.line_tph[l as usize][0] as f64 * 2.0 * 8.0 * RUN_COST_BOAT_KM * ECONOMY / 1e6;
    assert!((a.running_cost(0) - hour).abs() < 1e-12, "{} {hour}", a.running_cost(0));
    // upkeep: $10k a port a game day, no station among them
    let up = a.upkeep();
    assert!((up[7] - 0.02).abs() < 1e-12 && up[8] == 2.0 && up[2] == 0.0 && up[4] == 0.0 && (up[0] - 0.02).abs() < 1e-12, "{up:?}");
    let shares = a.line_upkeep(&[4.0, 6.0, 4.0, 4.0, 6.0]);
    assert!(shares.len() == 4 && shares[0] == l as f64 && (shares[1] - 0.02).abs() < 1e-12, "{shares:?}");
    // a boat is drawn along the route, the way back from the far end
    let (x, y) = a.w.svc.position(&a.w.net, l, 0, 4000.0);
    assert!((x - 4000.0).abs() < 1e-9 && y.abs() < 1e-9);
    let (x, _) = a.w.svc.position(&a.w.net, l, 1, 1000.0);
    assert!((x - 7000.0).abs() < 1e-9);
    // redrawn with the same ports: free, undoable, still running
    let c0 = a.cash();
    assert!(!a.can_undo());
    assert_eq!(a.set_ferry_route(l, &[p, q], &straight(&[0.0, 4000.0, 8000.0], &[0, 2], 25.0)), 0, "{}", a.issues());
    assert!(a.cash() == c0 && a.can_undo() && a.w.svc.lines[l as usize].running);
    // removing the line: its ports stay, its boats become spares, and it can be undone
    assert_eq!(a.remove_line(l), 0, "{}", a.issues());
    assert!(a.can_undo() && a.is_port(p) && a.fleet_ferry() == boats && a.boats_needed() == 0);
    // a second line sharing a constructed port and a new one takes the spare boats: only its new
    // port is paid
    let r = port(&mut a, -4000.0, "Battery");
    let m = ferry_line(&mut a, &[r, p], &straight(&[-4000.0, 0.0], &[0, 1], 25.0));
    let before = a.cash();
    assert_eq!(a.construct_line(m), 0, "{}", a.issues());
    let need = a.boats_needed();
    assert!(need > 0 && need <= boats && a.fleet_ferry() == boats);
    assert!((before - a.cash() - PORT_COST).abs() < 1e-6, "{}", before - a.cash());
    // that line redrawn to a third port, planned: the port is constructed with it and paid now, with
    // any boats the longer line needs past the spares
    let s = port(&mut a, -8000.0, "Pier A");
    let (c1, f1) = (a.cash(), a.fleet_ferry());
    assert_eq!(a.set_ferry_route(m, &[s, r, p], &straight(&[-8000.0, -4000.0, 0.0], &[0, 1, 2], 25.0)), 0, "{}", a.issues());
    let bought = a.fleet_ferry() - f1;
    assert!(a.w.net.node_built[s as usize] && a.w.svc.lines[m as usize].running && a.fleet_ferry() == a.boats_needed().max(f1));
    assert!((c1 - a.cash() - PORT_COST - bought as f64 * BOAT_PRICE).abs() < 1e-6, "{} {bought}", c1 - a.cash());
}

#[test]
fn a_port_is_a_station_off_the_track() {
    let mut a = api();
    let p = port(&mut a, 0.0, "Pier");
    // no track to it
    let ends = [1.0, p as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, -100.0, 0.0];
    assert_ne!(a.add_route(&ends, &[], 2, false), 0);
    assert!(a.issue_kinds().contains("NotAtPort"), "{}", a.issue_kinds());
    // no platform upgrades, the waiting room apart (T-3419)
    assert_ne!(a.set_station_upgrade(&[p], upgrades::TERMINAL, true), 0);
    assert_ne!(a.set_station_upgrade(&[p], upgrades::PARK_RIDE, true), 0);
    assert_eq!(a.set_station_upgrade(&[p], upgrades::WAITING_ROOM, true), 0, "{}", a.issues());
    assert_eq!(a.station_upgrades(), vec![p, upgrades::WAITING_ROOM as u32]);
    assert_eq!(row(&mut a, 8.0), Some((1.0, upgrades::WAITING_ROOM_COST)));
    assert_eq!(a.set_station_upgrade(&[p], upgrades::WAITING_ROOM, false), 0, "{}", a.issues());
    // a berth out of reach
    assert_ne!(a.add_port(5000.0, -100.0, 5000.0, 900.0, "Far".into()), 0);
    assert!(a.issue_kinds().contains("PortBerth") && a.last_port() == u32::MAX, "{}", a.issue_kinds());
    // a bus along the shore stops there as at a train station
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[-3000.0, -250.0, 1.0, 3000.0, -250.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    assert!(a.bus_stops(b).chunks_exact(8).any(|c| c[5] == p as f64), "{:?}", a.bus_stops(b));
    // a passage to a station near it
    let q = port(&mut a, 250.0, "Pier 2");
    assert_eq!(a.link_stations(p, q), 0, "{}", a.issues());
    // a ferry line holds its ports: neither goes while it stops there (taking it out of the line
    // leaves a line its route no longer fits)
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 250.0], &[0, 1], 10.0));
    assert_ne!(a.remove_station(p), 0);
    assert!(a.is_port(p) && a.line_stops(l) == vec![p, q], "{}", a.issue_kinds());
    // its route must start, stop and end at its ports' berths
    let refused = |a: &mut TrackApi, ports: &[u32], route: &[f64]| a.set_ferry_route(l, ports, route) != 0;
    assert!(refused(&mut a, &[p, q], &[0.0, 0.0, K_PORT as f64, 2.0, 250.0, 80.0, K_PORT as f64, 2.0]), "80 m off the berth");
    assert!(refused(&mut a, &[p, q], &straight(&[0.0, 250.0], &[0], 10.0)), "a port with no point");
    assert!(refused(&mut a, &[p], &straight(&[0.0, 250.0], &[0, 1], 10.0)), "one port");
    assert!(refused(&mut a, &[p, p], &straight(&[0.0, 0.0], &[0, 1], 10.0)), "no length");
    // a waypoint is passed, not stopped at
    assert_eq!(a.set_ferry_route(l, &[p, q], &[0.0, 0.0, K_PORT as f64, 2.0, 125.0, 40.0, K_WAYPOINT as f64, 2.0, 250.0, 0.0, K_PORT as f64, 2.0]), 0, "{}", a.issues());
    assert_eq!(a.stop_times(l, 0, 0).len(), 2);
    assert!(a.ferry_m(l) > 250.0);
}

#[test]
fn ferries_save_and_load_and_saves_without_them_are_unchanged() {
    let mut a = api();
    // track and a bus line, saved before any ferry
    let ends = [0.0, 0.0, 0.0, 2000.0, 0.0, 0.0, 0.0, 6000.0, 2000.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 1000.0, 1.0, 3000.0, 1000.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    let plain = a.save();
    assert!(!plain.windows(4).any(|w| w == b"LFR3") && plain.windows(4).any(|w| w == b"LBS2"));
    assert!(a.load(&plain));
    assert_eq!(a.save(), plain, "a save with no ferry loads and saves to the bit");
    // ports and a constructed line with a slow end and a waypoint
    let (p, q) = (port(&mut a, 0.0, "A"), port(&mut a, 5000.0, "B"));
    let route = [0.0, 0.0, K_PORT as f64, 2.778, 300.0, 0.0, K_ROUTE as f64, 5.139, 2500.0, 150.0, K_WAYPOINT as f64, 6.944, 5000.0, 0.0, K_PORT as f64, 3.417];
    let l = ferry_line(&mut a, &[p, q], &route);
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    let saved = a.save();
    assert!(saved.windows(4).any(|w| w == b"LFR3"));
    assert_eq!(TrackApi::save_unread(&saved), 0);
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let (times, back, way) = (bits(a.stop_times(l, 0, 0)), bits(a.stop_times(l, 1, 1)), a.ferry_route(l));
    assert_eq!(way, [vec![1.0], route.to_vec()].concat());
    let mut b = api();
    assert!(b.load(&saved));
    assert!(b.is_ferry(l) && b.is_port(p) && b.port_berth(q) == vec![5000.0, 0.0]);
    assert_eq!(b.ferry_route(l), way);
    assert_eq!((bits(b.stop_times(l, 0, 0)), bits(b.stop_times(l, 1, 1))), (times, back));
    assert!(b.w.svc.lines[l as usize].running);
    assert_eq!(b.save(), saved, "a ferry save round trips to the bit");
    // into another frame and back: ports and routes move with the ground
    let merc = TrackApi::convert_save_frame(&saved, 2);
    let mut c = api();
    assert!(c.load(&merc));
    let berth = c.port_berth(q);
    let landing = c.nodes_info().chunks_exact(10).find(|n| n[0] == q as f64).map(|n| [n[1], n[2]]).unwrap();
    assert!(((berth[0] - landing[0]).hypot(berth[1] - landing[1]) / c.w.net.k_at(landing[1]) - 100.0).abs() < 0.5);
    assert!(c.is_ferry(l) && c.w.svc.lines[l as usize].running && (c.ferry_m(l) - b.ferry_m(l)).abs() < 1.0);
    // T-2450: the same block as an LFR1 save wrote it loads with its speeds on today's curve: the
    // channel speed as it was, 25 km/h (6.944) the top then, 12.5 (45 km/h) now, between them in proportion
    let at = saved.windows(4).position(|w| w == b"LFR3").unwrap();
    let mut old = saved.clone();
    old[at + 3] = b'1';
    let mut d = api();
    assert!(d.load(&old));
    let speeds: Vec<f64> = d.ferry_route(l)[1..].chunks_exact(4).map(|c| c[3]).collect();
    assert_eq!(speeds, vec![2.778, 8.287, 12.499, 4.269]);
    // T-3435: an LFR2 save, searched at 32 km/h (8.889), loads on the 45 km/h curve the same way
    old[at + 3] = b'2';
    let mut e = api();
    assert!(e.load(&old));
    let speeds: Vec<f64> = e.ferry_route(l)[1..].chunks_exact(4).map(|c| c[3]).collect();
    assert_eq!(speeds, vec![2.778, 6.534, 9.406, 3.795]);
    assert!(d.stop_times(l, 0, 0)[1] < b.stop_times(l, 0, 0)[1]);
}

#[test]
fn riders_wait_at_ports() {
    let mut a = api();
    let (p, q) = (port(&mut a, 0.0, "A"), port(&mut a, 3000.0, "B"));
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 3000.0], &[0, 1], 25.0));
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    // 200 an hour each way all day: boarding at the first port of each run, alighting at the last
    let mut v = vec![1.0f32; PERIODS];
    v.extend([l as f32, 2.0]);
    for &(h0, h1) in PERIOD_HOURS.iter() {
        let x = 200.0 * (h1 - h0) as f32;
        v.extend([x, 0.0, x, 0.0, 0.0, x, 0.0, x]);
    }
    a.set_riders(v, 0);
    while a.boarding_step(8) {}
    let h = a.boarding_hour(86400.0 + 8.0 * 3600.0);
    let (trips, loads, queues) = (h[0] as usize, h[1] as usize, h[2] as usize);
    assert!(trips > 0 && queues > 0, "{:?}", &h[..5]);
    let q0 = 5 + 2 * trips + loads;
    let nodes: Vec<u32> = (0..queues).map(|k| h[q0 + k * QUEUE_STRIDE] as u32).collect();
    assert!(nodes.contains(&p) && nodes.contains(&q), "{nodes:?}");
    // a boat takes two cars' riders, rated at two thirds of its crush as a train car is (T-2919)
    assert_eq!(a.w.svc.lines[l as usize].crush(), 480.0);
    assert_eq!(a.w.svc.lines[l as usize].rated(), 320.0);
}

/// T-3892: a line's boat sets its riders, speed, price, running cost and time in port; each size is a
/// fleet of its own, the old boats stay as spares, and undo puts it all back.
#[test]
fn boat_sizes_carry_buy_run_and_undo() {
    use crate::track::ferry::Boat;
    let mut a = api();
    let (p, q) = (port(&mut a, 0.0, "A"), port(&mut a, 20000.0, "B"));
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 20000.0], &[0, 1], 45.0));
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    assert_eq!(a.ferry_boat(l), Boat::Medium.code());
    let ls = &a.w.svc.lines[l as usize];
    assert_eq!((ls.crush(), ls.rated()), (480.0, 320.0));
    let (medium, run_m) = (a.boats_needed(), a.stop_times(l, 0, 0)[1]);
    assert_eq!(a.fleet_ferry(), medium);
    // Large: 1,280 riders, five minutes in port and twenty at the ends, bought at US$50M each
    let cash = a.cash();
    assert_eq!(a.set_ferry_boat(l, Boat::Large.code()), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[l as usize];
    assert_eq!((ls.crush(), ls.rated(), ls.cars), (1920.0, 1280.0, 1));
    assert_eq!((a.w.net.line_dwell[l as usize], a.w.net.line_turn[l as usize]), (300.0, 1200.0));
    let large = a.boats_needed_large();
    assert!(large > 0 && a.boats_needed() == 0 && a.fleet_ferry() == medium && a.fleet_ferry_large() == large);
    assert!((cash - a.cash() - large as f64 * LARGE_BOAT_PRICE).abs() < 1e-6, "{}", cash - a.cash());
    let hour = a.w.net.line_tph[l as usize][0] as f64 * 2.0 * 20.0 * LARGE_RUN_COST_BOAT_KM * ECONOMY / 1e6;
    assert!((a.running_cost(0) - hour).abs() < 1e-12, "{} {hour}", a.running_cost(0));
    assert_eq!(a.stop_times(l, 0, 0)[1], run_m, "Large runs at Medium's speed");
    // undone: Medium again with its old dwell, the Large boats kept as spares
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!((a.ferry_boat(l), a.w.net.line_dwell[l as usize], a.w.net.line_turn[l as usize]), (Boat::Medium.code(), 60.0, 300.0));
    assert_eq!(a.boats_needed_large(), 0);
    // Small: 160 riders, 65 km/h in open water, so the run is shorter, at US$15M a boat
    let cash = a.cash();
    assert_eq!(a.set_ferry_boat(l, Boat::Small.code()), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[l as usize];
    assert_eq!((ls.crush(), ls.rated()), (240.0, 160.0));
    let run_s = a.stop_times(l, 0, 0)[1];
    let v = crate::track::ferry::SMALL_TOP;
    assert!((run_s - (20000.0 / v + v / BOAT_A)).abs() < 0.5, "{run_s} against {run_m}");
    let small = a.boats_needed_small();
    assert!(small > 0 && (cash - a.cash() - small as f64 * SMALL_BOAT_PRICE).abs() < 1e-6);
    // the drawing tool quotes the line's own boat
    let quote = a.ferry_quote(l, &[p, q], &straight(&[0.0, 20000.0], &[0, 1], 45.0), 60.0);
    assert!((quote[1] - run_s).abs() < 1e-6, "{quote:?}");
    // a boat code past Large is refused
    assert_ne!(a.set_ferry_boat(l, 3), 0);
    // redrawing the route keeps the boat
    assert_eq!(a.set_ferry_route(l, &[p, q], &straight(&[0.0, 10000.0, 20000.0], &[0, 2], 45.0)), 0, "{}", a.issues());
    assert_eq!(a.ferry_boat(l), Boat::Small.code());
}

/// T-3892: the boats save in `LFB1`; a save without it (an older one) loads every ferry as Medium, and
/// a game with only Medium boats writes none.
#[test]
fn boat_sizes_save_and_older_saves_load_as_medium() {
    use crate::track::ferry::Boat;
    let mut a = api();
    let (p, q, s) = (port(&mut a, 0.0, "A"), port(&mut a, 8000.0, "B"), port(&mut a, -8000.0, "C"));
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 8000.0], &[0, 1], 45.0));
    let m = ferry_line(&mut a, &[s, p], &straight(&[-8000.0, 0.0], &[0, 1], 45.0));
    let plain = a.save();
    assert!(!plain.windows(4).any(|w| w == b"LFB1"));
    assert_eq!(a.set_ferry_boat(l, Boat::Large.code()), 0, "{}", a.issues());
    let saved = a.save();
    let at = saved.windows(4).position(|w| w == b"LFB1").expect("LFB1 written");
    assert_eq!(TrackApi::save_unread(&saved), 0);
    let mut b = api();
    assert!(b.load(&saved));
    assert_eq!((b.ferry_boat(l), b.ferry_boat(m)), (Boat::Large.code(), Boat::Medium.code()));
    assert_eq!(b.w.svc.lines[l as usize].crush(), 1920.0);
    assert_eq!(b.save(), saved, "a save with boat sizes round trips to the bit");
    // the same save as an older build wrote it, with no LFB1: every boat Medium
    let mut c = api();
    assert!(c.load(&saved[..at]));
    assert_eq!((c.ferry_boat(l), c.ferry_boat(m)), (Boat::Medium.code(), Boat::Medium.code()));
}

/// T-3767: a ferry line sets five rows as a rail line does; a midday edit moves the midday's rate at once.
#[test]
fn a_ferry_line_runs_five_periods() {
    let mut a = api();
    let (p, q) = (port(&mut a, 0.0, "Whitehall"), port(&mut a, 8000.0, "St George"));
    let l = ferry_line(&mut a, &[p, q], &straight(&[0.0, 8000.0], &[0, 1], 25.0));
    assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    assert_eq!(a.set_game_periods(5), 0, "{}", a.issues());
    assert_eq!(a.line_period_count(l), 5);
    assert_eq!(a.line_row_headways(l), vec![900, 900, 900, 1200, 1800]);
    assert_eq!(a.set_line_row_headways(l, vec![900, 3600, 900, 1200, 1800]), 0, "{}", a.issues());
    let r = a.w.net.line_tph[l as usize];
    assert!((r[1] - 1.0).abs() < 0.01 && (r[0] - 4.0).abs() < 0.01 && (r[2] - 4.0).abs() < 0.01, "{r:?}");
}
