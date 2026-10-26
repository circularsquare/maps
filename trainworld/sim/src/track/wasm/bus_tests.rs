//! T-1464: bus lines through the clock worker's API.

use super::*;
use crate::track::bus::{self, BUS_V, KIND_CROSSING, KIND_END, KIND_MIDDLE, KIND_STATION, ROAD_FACTOR};

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    a
}

/// A constructed rail line along y = 0 with stations every `step` m up to `km` km; its stations.
fn rail(a: &mut TrackApi, km: usize, step: f64) -> (u32, Vec<u32>) {
    let n = (km as f64 * 1000.0 / step).round() as usize;
    let mut st = vec![];
    for i in 0..n {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * step, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &s) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 120, format!("S{i}")), 0, "{}", a.issues());
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Rail".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (l, st)
}

fn stops(a: &TrackApi, l: u32) -> Vec<Vec<f64>> {
    a.bus_stops(l).chunks_exact(8).map(|c| c.to_vec()).collect()
}

#[test]
fn a_feeder_bus_stops_at_the_station_it_reaches_and_runs_at_its_speed() {
    let mut a = api();
    let (_, st) = rail(&mut a, 10, 5000.0);
    let cash = a.cash();
    let b = a.new_line_id();
    // 6 km north of the middle station, ending 100 m short of it
    assert_eq!(a.set_bus_line(b, &[5000.0, 6100.0, 5000.0, 100.0], 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert!(a.is_bus(b) && a.bus_lines() == vec![b]);
    let s = stops(&a, b);
    assert_eq!(s[0][4], KIND_END as f64);
    assert_eq!(s.last().unwrap()[4], KIND_STATION as f64, "the end merges into the station: {s:?}");
    assert_eq!(s.last().unwrap()[5], st[1] as f64);
    // 7.2 km of road: ends, and stops at about every 2 km on the 400 m grid
    let road = 6000.0 * ROAD_FACTOR;
    assert!((a.lines_info()[22 + 9] - road).abs() < 1.0, "{:?}", a.lines_info());
    let mid: Vec<&Vec<f64>> = s.iter().filter(|c| c[4] == KIND_MIDDLE as f64).collect();
    assert_eq!(mid.len(), 3, "{s:?}");
    for c in &mid {
        let k = c[3] / (road / (road / 400.0).round());
        assert!((k - k.round()).abs() < 1e-6, "a middle stop on the street grid: {c:?}");
    }
    assert_eq!(a.bus_street(b).len() / 2, (road / 400.0).round() as usize - 1 - 3);
    // covers: every ground metre of the route belongs to one stop that is not a station
    let covers: Vec<[f64; 2]> = s.iter().filter(|c| c[6].is_finite()).map(|c| [c[6], c[7]]).collect();
    assert_eq!(covers.len(), s.len() - 1);
    assert_eq!(covers[0][0], 0.0);
    assert!((covers.last().unwrap()[1] - 6000.0).abs() < 1e-6);
    assert!(covers.windows(2).all(|w| (w[0][1] - w[1][0]).abs() < 1e-9));
    // planned: no service yet, no cost, nothing charged
    assert!(!a.w.svc.lines[b as usize].running && a.cash() == cash);
    // run time: road at 18 km/h, no dwell between (no station between)
    let t = a.stop_times(b, 0, 0);
    assert!((t.last().unwrap() - road / BUS_V).abs() < 1e-6, "{t:?}");
    // construct: the road at $1M a km and the buses at $0.6M
    let quote = a.line_build_cost(b);
    assert!((quote - road / 1000.0).abs() < 1e-6, "{quote}");
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[b as usize];
    assert!(ls.running);
    let buses = ls.trains[0];
    assert!(buses > 0 && a.fleet_bus() == buses && a.buses_needed() == buses);
    assert!((cash - a.cash() - (road / 1000.0 + buses as f64 * BUS_PRICE)).abs() < 1e-6, "{} {}", cash - a.cash(), buses);
    // running cost: $3 a bus-km at the economy scale, on top of the rail line's
    let rail_only = {
        let mut r = api();
        rail(&mut r, 10, 5000.0);
        r.running_cost(0)
    };
    let bus_hour = a.w.net.line_tph[b as usize][0] as f64 * 2.0 * road / 1000.0 * RUN_COST_BUS_KM * ECONOMY / 1e6;
    assert!((a.running_cost(0) - rail_only - bus_hour).abs() < 1e-9);
    // upkeep: $3k a road-km a game day
    let up = a.upkeep();
    assert!((up[5] - road / 1000.0 * 0.003).abs() < 1e-9 && (up[6] - road / 1000.0).abs() < 1e-9, "{up:?}");
    // a bus is drawn along the route
    let (x, y) = a.w.svc.position(&a.w.net, b, 0, road / 2.0);
    assert!((x - 5000.0).abs() < 1e-6 && (y - 3100.0).abs() < 1e-6, "{x} {y}");
    let (x, y) = a.w.svc.position(&a.w.net, b, 1, 0.0);
    assert!((x - 5000.0).abs() < 1e-6 && (y - 100.0).abs() < 1e-6);
    // save and load: the route, its state and its stops come back
    let saved = a.save();
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let before = bits(a.bus_stops(b));
    assert!(a.load(&saved));
    assert_eq!(bits(a.bus_stops(b)), before);
    assert_eq!(a.bus_route(b)[0], 1.0);
    // removing a constructed route is final, as for track
    assert_eq!(a.remove_line(b), 0, "{}", a.issues());
    assert!(!a.can_undo() && !a.is_bus(b));
}

#[test]
fn stops_follow_stations_built_near_the_route_and_crossings() {
    let mut a = api();
    let (_, st) = rail(&mut a, 4, 2000.0);
    let b = a.new_line_id();
    // along the rail line 150 m north of it: every station is a stop
    assert_eq!(a.set_bus_line(b, &[-1000.0, 150.0, 5000.0, 150.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    let at: Vec<f64> = s.iter().filter(|c| c[4] == KIND_STATION as f64).map(|c| c[5]).collect();
    assert_eq!(at, st.iter().map(|&n| n as f64).collect::<Vec<_>>(), "{s:?}");
    // the dwell stands only at stations
    let t = a.stop_times(b, 0, 0);
    let d = a.stop_departures(b, 0, 0);
    for (k, c) in s.iter().enumerate().skip(1).take(s.len() - 2) {
        let want = if c[4] == KIND_STATION as f64 { 30.0 } else { 0.0 };
        assert!((d[k] - t[k] - want).abs() < 1e-9);
    }
    // a second bus crossing the first: both stop where they cross, at the same point
    let c = a.new_line_id();
    assert_eq!(a.set_bus_line(c, &[3000.0, -2000.0, 3000.0, 3000.0], 480, 600, 1200, 30.0, 120.0, 0, "C".into()), 0, "{}", a.issues());
    let cross = |a: &TrackApi, l: u32| stops(a, l).into_iter().find(|c| c[4] == KIND_CROSSING as f64);
    let (p, q) = (cross(&a, b).expect("b stops at the crossing"), cross(&a, c).expect("c stops at the crossing"));
    assert!((p[1] - q[1]).abs() < 1e-6 && (p[2] - q[2]).abs() < 1e-6 && (p[1] - 3000.0).abs() < 1e-6);
    // undoing the second line takes the first's crossing stop away
    assert!(a.undo() == 0);
    assert!(cross(&a, b).is_none());
    // a station built near the route becomes a stop (the line keeps its stops otherwise)
    let ends: Vec<f64> = [0.0, 0.0, 0.0, 600.0, -1.0, 0.0, 0.0, 2000.0, 600.0, -1.0].to_vec();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let n0 = a.nodes_info().chunks_exact(10).find(|c| c[1] == 0.0 && c[2] == 600.0).unwrap()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 120, "North".into()), 0, "{}", a.issues());
    assert!(!stops(&a, b).iter().any(|c| c[5] == n0 as f64), "600 m away is out of reach");
    let nodes_before = a.nodes_info().len();
    let ends: Vec<f64> = [0.0, 0.0, 500.0, 300.0, -1.0, 0.0, 0.0, 1500.0, 300.0, -1.0].to_vec();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert!(a.nodes_info().len() > nodes_before);
    let n1 = a.route_end();
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 120, "Near".into()), 0, "{}", a.issues());
    assert!(stops(&a, b).iter().any(|c| c[5] == n1 as f64), "a planned bus previews planned stations");
}

#[test]
fn a_bus_route_needs_length_and_lines_without_buses_save_as_before() {
    let mut a = api();
    let (l, _) = rail(&mut a, 2, 1000.0);
    let plain = a.save();
    assert!(!plain.windows(4).any(|w| w == b"LBS1"));
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0, 50.0, 500.0], 480, 600, 1200, 30.0, 120.0, 0, "Short".into()), 1);
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0], 480, 600, 1200, 30.0, 120.0, 0, "One".into()), 1);
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0, 3000.0, 500.0], 480, 600, 1200, 30.0, 120.0, 0, "Ok".into()), 0, "{}", a.issues());
    assert!(a.save().windows(4).any(|w| w == b"LBS1"));
    // the quote agrees with the line
    let q = a.bus_quote(u32::MAX, &[0.0, 500.0, 3000.0, 500.0]);
    assert!((q[0] - 3000.0 * ROAD_FACTOR).abs() < 1e-6 && q[2] == 1.0 && q[3] as usize == stops(&a, b).len(), "{q:?}");
    // removing the planned bus line leaves the rail line's service as it was
    let rail_times = a.stop_times(l, 0, 0);
    assert_eq!(a.remove_line(b), 0);
    assert_eq!(a.stop_times(l, 0, 0), rail_times);
    assert_eq!(a.save(), plain);
    assert!(bus::is_stop_id(bus::stop_id(5, 3)) && !bus::is_stop_id(5));
}
