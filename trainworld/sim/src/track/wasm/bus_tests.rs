//! T-1464: bus lines through the clock worker's API.

use super::*;
use crate::track::bus::{self, KIND_CROSSING, KIND_END, KIND_MIDDLE, KIND_STATION, ROAD_FACTOR};

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
    assert_eq!(a.set_bus_line(b, &[5000.0, 6100.0, 1.0, 5000.0, 100.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert!(a.is_bus(b) && a.bus_lines() == vec![b]);
    let s = stops(&a, b);
    assert_eq!(s[0][4], KIND_END as f64);
    assert_eq!(s.last().unwrap()[4], KIND_STATION as f64, "the end merges into the station: {s:?}");
    assert_eq!(s.last().unwrap()[5], st[1] as f64);
    // T-2982: the stop is on the route, the point nearest the station, not on the station's centre
    let (sx, sy) = (s.last().unwrap()[1], s.last().unwrap()[2]);
    assert!((sx - 5000.0).abs() < 1e-6 && (sy - 100.0).abs() < 1e-6, "{sx} {sy}");
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
    assert!((covers.last().unwrap()[1] - road).abs() < 1e-6);
    assert!(covers.windows(2).all(|w| (w[0][1] - w[1][0]).abs() < 1e-9));
    // planned: no service yet, no cost, nothing charged
    assert!(!a.w.svc.lines[b as usize].running && a.cash() == cash);
    // run time: open-country road with a stop every 400 m, no dwell between (no station between)
    let t = a.stop_times(b, 0, 0);
    assert!((t.last().unwrap() - bus::Pace::default().auto_time(0.0, road)).abs() < 1e-6, "{t:?}");
    // construct: the road at `BUS_ROUTE_COST_PER_KM` and the buses at $0.6M
    let quote = a.line_build_cost(b);
    assert!((quote - road / 1000.0 * BUS_ROUTE_COST_PER_KM).abs() < 1e-6, "{quote}");
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[b as usize];
    assert!(ls.running);
    // T-2919: a bus is a third of a train car, 80 at crush and rated at two thirds of it
    assert!(ls.crush() == 80.0 && (ls.rated() - 160.0 / 3.0).abs() < 1e-9, "{} {}", ls.crush(), ls.rated());
    let buses = ls.trains[0];
    assert!(buses > 0 && a.fleet_bus() == buses && a.buses_needed() == buses);
    assert!((cash - a.cash() - (road / 1000.0 * BUS_ROUTE_COST_PER_KM + buses as f64 * BUS_PRICE)).abs() < 1e-6, "{} {}", cash - a.cash(), buses);
    // running cost: `RUN_COST_BUS_KM` a bus-km at the economy scale, on top of the rail line's
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
    assert_eq!(a.set_bus_line(b, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
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
    assert_eq!(a.set_bus_line(c, &[3000.0, -2000.0, 1.0, 3000.0, 3000.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "C".into()), 0, "{}", a.issues());
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

/// T-2952 (found in T-2949): two bus lines crossing near a station only one of them reaches.
#[test]
fn crossing_bus_lines_share_one_stop_or_a_station() {
    let mut a = api();
    let (_, st) = rail(&mut a, 4, 2000.0);
    let s1 = st[1] as f64;
    // A runs 150 m north of the rail line: its stop for the station at x = 2000 is 150 m along from where B crosses
    let ln = a.new_line_id();
    assert_eq!(a.set_bus_line(ln, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "A".into()), 0, "{}", a.issues());
    // B crosses A at (2150, 150) square to the line from the station, so it passes no nearer than 212 m: out of reach
    let (cx, cy, d) = (2150.0, 150.0, std::f64::consts::FRAC_1_SQRT_2);
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[cx - 2000.0 * d, cy + 2000.0 * d, 1.0, cx + 2000.0 * d, cy - 2000.0 * d, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    let cross = |a: &TrackApi, l: u32| stops(a, l).into_iter().filter(|c| c[4] == KIND_CROSSING as f64).collect::<Vec<_>>();
    let (p, q) = (cross(&a, ln), cross(&a, b));
    assert_eq!((p.len(), q.len()), (1, 1), "both stop at the crossing: {p:?} {q:?}");
    assert!((p[0][1] - cx).abs() < 1e-6 && (p[0][2] - cy).abs() < 1e-6 && (q[0][1] - cx).abs() < 1e-6 && (q[0][2] - cy).abs() < 1e-6, "{p:?} {q:?}");
    assert!(stops(&a, ln).iter().any(|c| c[5] == s1), "A still stops at the station");
    assert!(!stops(&a, b).iter().any(|c| c[5] == s1), "B is out of its reach");
    // C crosses A 100 m east of the station and reaches it too: the two change there, and neither stops at the crossing
    let c = a.new_line_id();
    assert_eq!(a.set_bus_line(c, &[2100.0, 2000.0, 1.0, 2100.0, -2000.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "C".into()), 0, "{}", a.issues());
    assert!(stops(&a, c).iter().any(|x| x[5] == s1), "C stops at the station");
    let at_c = |a: &TrackApi, l: u32| cross(a, l).into_iter().filter(|x| (x[1] - 2100.0).abs() < 1.0 && (x[2] - 150.0).abs() < 1.0).count();
    assert_eq!((at_c(&a, ln), at_c(&a, c)), (0, 0), "{:?} {:?}", cross(&a, ln), cross(&a, c));
    // B and C cross far from any station: both stop there
    assert_eq!((cross(&a, b).len(), cross(&a, c).len()), (2, 1), "{:?} {:?}", cross(&a, b), cross(&a, c));
    // by hand, B keeps only its own stops; A still stops where B crosses
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert!(cross(&a, b).is_empty());
    assert_eq!(cross(&a, ln).len(), 1);
}

#[test]
fn a_bus_route_needs_length_and_lines_without_buses_save_as_before() {
    let mut a = api();
    let (l, _) = rail(&mut a, 2, 1000.0);
    let plain = a.save();
    assert!(!plain.windows(4).any(|w| w == b"LBS2"));
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0, 1.0, 50.0, 500.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "Short".into()), 1);
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "One".into()), 1);
    assert_eq!(a.set_bus_line(b, &[0.0, 500.0, 1.0, 3000.0, 500.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "Ok".into()), 0, "{}", a.issues());
    assert!(a.save().windows(4).any(|w| w == b"LBS2"));
    // the quote agrees with the line
    let q = a.bus_quote(u32::MAX, &[0.0, 500.0, 1.0, 3000.0, 500.0, 1.0]);
    assert!((q[0] - 3000.0 * ROAD_FACTOR).abs() < 1e-6 && q[2] == 1.0 && q[3] as usize == stops(&a, b).len(), "{q:?}");
    // removing the planned bus line leaves the rail line's service as it was
    let rail_times = a.stop_times(l, 0, 0);
    assert_eq!(a.remove_line(b), 0);
    assert_eq!(a.stop_times(l, 0, 0), rail_times);
    assert_eq!(a.save(), plain);
    assert!(bus::is_stop_id(bus::stop_id(5, 3)) && !bus::is_stop_id(5));
}

#[test]
fn stop_spacing_is_one_undoable_edit_kept_through_a_redraw_and_a_save() {
    let bits = |a: &TrackApi, b: u32| stops(a, b).iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>();
    let mut a = api();
    let (_, st) = rail(&mut a, 20, 2000.0);
    let b = a.new_line_id();
    // along the rail line, 100 m from its 11 stations
    let route = [-500.0, 100.0, 1.0, 20_500.0, 100.0, 3.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0, "Far".into()), 0, "{}", a.issues());
    let auto = stops(&a, b);
    let auto_bits = bits(&a, b);
    assert_eq!(auto.iter().filter(|c| c[4] == KIND_STATION as f64).count(), st.len());
    assert!(!a.save().windows(4).any(|w| w == b"LBP1"), "no spacing set: no block");
    assert_eq!(a.bus_spacing(b), 0.0);
    let auto_trip = a.stop_times(b, 0, 0);

    assert_eq!(a.set_bus_spacing(b, 5000.0), 0, "{}", a.issues());
    assert_eq!(a.bus_spacing(b), 5000.0);
    let far = stops(&a, b);
    let far_bits = bits(&a, b);
    assert!(far.len() < auto.len() && far.iter().filter(|c| c[4] == KIND_STATION as f64).count() >= 3, "{far:?}");
    assert!(a.bus_street(b).is_empty());
    assert!(far.windows(2).all(|w| w[1][3] - w[0][3] >= 2500.0 - 1e-6), "{far:?}");
    assert!(a.stop_times(b, 0, 0).last() < auto_trip.last(), "fewer stops, a shorter trip");

    assert_eq!(a.undo(), 0);
    assert_eq!((a.bus_spacing(b), bits(&a, b)), (0.0, auto_bits.clone()));
    assert_eq!(a.redo(), 0);
    assert_eq!((a.bus_spacing(b), bits(&a, b)), (5000.0, far_bits.clone()));

    // a redraw keeps it, and so does a save
    assert_eq!(a.set_bus_route(b, &route), 0, "{}", a.issues());
    assert_eq!(a.bus_spacing(b), 5000.0);
    let bytes = a.save();
    assert!(bytes.windows(4).any(|w| w == b"LBP1"));
    let mut c = api();
    assert!(c.load(&bytes));
    assert_eq!((c.bus_spacing(b), bits(&c, b)), (5000.0, far_bits));

    // back to automatic
    assert_eq!(a.set_bus_spacing(b, 0.0), 0, "{}", a.issues());
    assert_eq!(bits(&a, b), auto_bits);
    assert!(!a.save().windows(4).any(|w| w == b"LBP1"));
    // limits
    assert_eq!(a.set_bus_spacing(b, 1.0), 0);
    assert_eq!(a.bus_spacing(b), 0.0);
    assert_eq!(a.set_bus_spacing(b, bus::SPACING_MIN_M as f64), 0);
    assert_eq!(a.bus_spacing(b), 0.0);
    assert_eq!(a.set_bus_spacing(b, 1e9), 0);
    assert_eq!(a.bus_spacing(b), bus::SPACING_MAX_M as f64);
}

#[test]
fn stops_by_hand_clear_the_automatic_ones_and_keep_through_undo_a_redraw_and_a_save() {
    let bits = |a: &TrackApi, b: u32| stops(a, b).iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>();
    let kinds = |a: &TrackApi, b: u32| stops(a, b).iter().map(|c| c[4] as u8).collect::<Vec<_>>();
    let mut a = api();
    let (_, st) = rail(&mut a, 20, 2000.0);
    let b = a.new_line_id();
    let route = [-500.0, 100.0, 1.0, 20_500.0, 100.0, 3.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0, "Hand".into()), 0, "{}", a.issues());
    let auto_bits = bits(&a, b);
    let auto_trip = a.stop_times(b, 0, 0);
    let plain_save = a.save();
    assert!(!plain_save.windows(4).any(|w| w == b"LBH1"), "nobody placed stops: no block");

    // the ends alone, one long hop, running fast
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert!(a.bus_hand(b));
    assert_eq!(kinds(&a, b), vec![KIND_END, KIND_END]);
    assert!(a.bus_street(b).is_empty());
    let one_hop = *a.stop_times(b, 0, 0).last().unwrap();
    assert!(one_hop < 0.5 * auto_trip.last().unwrap(), "{one_hop} against {auto_trip:?}");
    assert_eq!(a.bus_reach(b).len() / 3, st.len(), "every station stays in reach");

    // near a station is the station; elsewhere it is a stop where the click was, 400 m of route each way
    assert_eq!(a.set_bus_stop(b, 6120.0, 90.0, true), 0, "{}", a.issues());
    assert_eq!(a.set_bus_stop(b, 9000.0, 50.0, true), 0, "{}", a.issues());
    let s = stops(&a, b);
    assert_eq!(s.iter().map(|c| c[4] as u8).collect::<Vec<_>>(), vec![KIND_END, KIND_STATION, KIND_MIDDLE, KIND_END], "{s:?}");
    assert_eq!(s[1][5], st[3] as f64);
    assert!((s[2][1] - 9000.0).abs() < 1e-6 && (s[2][2] - 100.0).abs() < 1e-6, "on the route: {s:?}");
    assert!((s[2][6] - 9100.0).abs() < 1e-6 && (s[2][7] - 9900.0).abs() < 1e-6, "{s:?}");
    assert!(s[0][6] == 0.0 && (s[0][7] - 400.0).abs() < 1e-6, "{s:?}");
    let two = bits(&a, b);

    // on top of a stop or an end, nothing changes and no undo step is made
    assert_eq!(a.set_bus_stop(b, 9020.0, 100.0, true), 0);
    assert_eq!(a.set_bus_stop(b, -480.0, 100.0, true), 0);
    assert_eq!(a.set_bus_stop(b, -500.0, 100.0, false), 0);
    assert_eq!(bits(&a, b), two);
    assert_eq!(a.undo(), 0);
    assert_eq!(stops(&a, b).len(), 3, "the last real edit went");
    assert_eq!(a.redo(), 0);
    assert_eq!(bits(&a, b), two);

    // a click on a stop takes it out, and all of a station's points with it
    let at = s[1].clone();
    assert_eq!(a.set_bus_stop(b, at[1], at[2], false), 0, "{}", a.issues());
    assert_eq!(kinds(&a, b), vec![KIND_END, KIND_MIDDLE, KIND_END]);
    assert_eq!(a.undo(), 0);
    assert_eq!(bits(&a, b), two);

    // a copy has them; a redraw keeps those it still passes, on the new route
    let copy = a.new_line_id();
    assert_eq!(a.copy_line(b, copy, "Copy".into()), 0, "{}", a.issues());
    assert_eq!(bits(&a, copy).len(), bits(&a, b).len());
    assert!(a.bus_hand(copy));
    assert_eq!(a.remove_line(copy), 0);
    assert_eq!(a.set_bus_route(b, &[-500.0, 150.0, 1.0, 10_500.0, 150.0, 3.0]), 0, "{}", a.issues());
    assert_eq!(kinds(&a, b), vec![KIND_END, KIND_STATION, KIND_MIDDLE, KIND_END]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.set_bus_route(b, &[-500.0, 100.0, 1.0, 7000.0, 100.0, 3.0]), 0, "{}", a.issues());
    assert_eq!(kinds(&a, b), vec![KIND_END, KIND_STATION, KIND_END], "the stop at 9 km is off the new road");
    assert_eq!(a.undo(), 0);
    assert_eq!(bits(&a, b), two);

    // a save keeps them
    let bytes = a.save();
    assert!(bytes.windows(4).any(|w| w == b"LBH1"));
    let mut c = api();
    assert!(c.load(&bytes));
    assert!(c.bus_hand(b));
    assert_eq!(bits(&c, b), two);
    assert_eq!(c.save(), bytes, "bit for bit");

    // a save from before has none, and loads as it did
    assert!(c.load(&plain_save));
    assert!(!c.bus_hand(b));
    assert_eq!(bits(&c, b), auto_bits);
    assert_eq!(c.save(), plain_save);

    // leaving goes back to the automatic stops; a spacing is another way out
    assert_eq!(a.set_bus_spacing(b, 0.0), 0, "{}", a.issues());
    assert!(!a.bus_hand(b));
    assert_eq!(bits(&a, b), auto_bits);
    // T-2939: choosing By hand again gives the stops back; a redraw in between keeps those still on the route
    assert_eq!(a.set_bus_hand(b), 0);
    assert_eq!(bits(&a, b), two, "the placed stops came back");
    assert_eq!(a.set_bus_spacing(b, 0.0), 0);
    assert_eq!(a.set_bus_route(b, &[-500.0, 100.0, 1.0, 7000.0, 100.0, 3.0]), 0, "{}", a.issues());
    assert_eq!(a.set_bus_hand(b), 0);
    assert_eq!(kinds(&a, b), vec![KIND_END, KIND_STATION, KIND_END], "the stop at 9 km is off the new road");
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.set_bus_spacing(b, 5000.0), 0);
    assert!(!a.bus_hand(b) && a.bus_spacing(b) == 5000.0);
    // only a by-hand line takes a placed stop
    assert_eq!(a.set_bus_stop(b, 9000.0, 100.0, true), 1);
}

#[test]
fn a_route_laid_along_roads_counts_its_drawn_length_and_keeps_its_corners() {
    let mut a = api();
    let b = a.new_line_id();
    // corner, road, road+corner, then a straight stretch to the last corner: 1000 m of road at 1.0
    // and 500 m straight at 1.2
    let route = [0.0, 0.0, 1.0, 500.0, 0.0, 2.0, 1000.0, 0.0, 3.0, 1000.0, 500.0, 1.0];
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0, "R".into()), 0, "{}", a.issues());
    let road = 1000.0 + 500.0 * ROAD_FACTOR;
    assert!((a.bus_road_m(b) - road).abs() < 1e-6);
    assert!((a.bus_quote(u32::MAX, &route)[0] - road).abs() < 1e-6);
    assert_eq!(a.bus_route(b), [vec![0.0], route.to_vec()].concat());
    // a bus is drawn along it, at road metres
    let (x, y) = a.w.svc.position(&a.w.net, b, 0, 1000.0 + 120.0);
    assert!((x - 1000.0).abs() < 1e-6 && (y - 100.0).abs() < 1e-6, "{x} {y}");
    let saved = a.save();
    assert!(a.load(&saved));
    assert_eq!(a.bus_route(b), [vec![0.0], route.to_vec()].concat());
    // drawn again straight: the same corners cost more
    let straight = [0.0, 0.0, 1.0, 1000.0, 0.0, 1.0, 1000.0, 500.0, 1.0];
    assert_eq!(a.set_bus_route(b, &straight), 0, "{}", a.issues());
    assert!((a.bus_road_m(b) - 1500.0 * ROAD_FACTOR).abs() < 1e-6);
}

#[test]
fn a_long_road_route_is_thinned_for_demand_and_still_covers_its_ends() {
    let mut a = api();
    let b = a.new_line_id();
    // 8 km along x with a point every 20 m, 0.5 m of wobble: 400 points
    let mut route = vec![];
    for i in 0..=400 {
        route.extend([i as f64 * 20.0, if i % 2 == 0 { 0.0 } else { 0.5 }, if i == 0 || i == 400 { 3.0 } else { 2.0 }]);
    }
    assert_eq!(a.set_bus_line(b, &route, 480, 600, 1200, 30.0, 120.0, 0, "L".into()), 0, "{}", a.issues());
    let c = a.bus_catches(b);
    let mut k = 0;
    while k < c.len() {
        let n = c[k] as usize;
        assert!(n <= 4, "a stretch of 2 km straight within 15 m keeps few points, got {n}");
        k += 1 + 2 * n;
    }
    assert!((a.bus_road_m(b) - 8000.0).abs() < 5.0);
}

/// T-1577: a pack's street speeds slow a bus where its zones are dense, time its trip again and
/// need more buses at the same headway; they outlive a load.
#[test]
fn a_bus_in_dense_streets_runs_slower_and_needs_more_buses() {
    let mut a = api();
    let b = a.new_line_id();
    // 6 km of straight route east from the origin: 7.2 km of road, timed as open country
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    let road = 6000.0 * ROAD_FACTOR;
    let flat = *a.stop_times(b, 0, 0).last().unwrap();
    assert!((flat - bus::Pace::default().auto_time(0.0, road)).abs() < 1e-6);
    let buses = a.w.svc.lines[b as usize].trains[0];
    // a pack at the same origin: one zone of 2 km squares from x = -1 km, three squares, the first
    // two dense (about 100,000 homes and jobs a km2), the third open
    let (pop, jobs) = ([30_000.0f32, 30_000.0, 50.0], [20_000.0f32, 20_000.0, 5.0]);
    assert!(!a.add_pack_streets(2000.0, -1000.0, -1000.0, 3, 1, &pop, &jobs, &[0, 1], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758));
    assert!(a.add_pack_streets(2000.0, -1000.0, -1000.0, 3, 1, &pop, &jobs, &[0, 1, 2], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758));
    let slow = *a.stop_times(b, 0, 0).last().unwrap();
    // the first 3 km of ground (3.6 km of road) in the bus lane, the rest at open country's bus speed, and 18 stops
    let want = 3600.0 / (crate::track::streets::BUS_LANE_KMH / 3.6) + 3600.0 / (crate::track::streets::bus_kmh(44.0) / 3.6) + 18.0 * 28.0;
    assert!(slow > flat && (slow - want).abs() < 0.04 * want, "{flat} {slow} {want}");
    assert!(a.w.svc.lines[b as usize].trains[0] >= buses);
    // the way back is as long
    let back = *a.stop_times(b, 1, 0).last().unwrap();
    assert!((back - slow).abs() < 1e-6, "{back} {slow}");
    // kept across a load
    let saved = a.save();
    assert!(a.load(&saved));
    assert!((*a.stop_times(b, 0, 0).last().unwrap() - slow).abs() < 1e-6);
}

/// T-3704: a bus line uses no track, so no capacity resource names it; a change to its buses or its
/// headway still settles into the buses an hour it runs, on every edit.
fn a_built_bus() -> (TrackApi, u32) {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 900, 1800, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    (a, b)
}

#[test]
fn a_bus_lines_rate_follows_its_buses() {
    let (mut a, b) = a_built_bus();
    let trip = a.w.svc.lines[b as usize].round_trip[0];
    for n in [1u32, 2, 4, 8, 12, 20, 4, 2] {
        assert_eq!(a.set_train_counts(b, n, 3, 2), 0, "{}", a.issues());
        let tph = a.w.net.line_tph[b as usize][0] as f64;
        assert!((tph - n as f64 * 3600.0 / trip).abs() < 0.01, "{n} buses: {tph} an hour over a {trip} s round trip");
    }
}

#[test]
fn a_bus_lines_rate_follows_its_headway() {
    let (mut a, b) = a_built_bus();
    for h in [600u32, 300, 1200, 240, 120] {
        assert_eq!(a.set_auto_headway(b, h, 900, 1800), 0, "{}", a.issues());
        let tph = a.w.net.line_tph[b as usize][0] as f64;
        assert!((tph - 3600.0 / h as f64).abs() < 0.01, "a {h} s headway runs {tph} an hour");
    }
}

/// T-3767: a bus line sets five rows as a rail line does: made in a game of five, its midday and
/// afternoon start at its high headway; a midday headway or count edit moves the midday's rate at once
/// (T-3704), and only the midday's running cost.
#[test]
fn a_bus_line_runs_five_periods() {
    let mut a = api();
    assert_eq!(a.set_game_periods(5), 0, "{}", a.issues());
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 900, 1800, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    assert_eq!(a.line_period_count(b), 5);
    assert_eq!(a.line_row_headways(b), vec![480, 480, 480, 900, 1800]);
    let tph = |a: &TrackApi, s: usize| a.w.net.line_tph[b as usize][s] as f64;
    assert!((tph(&a, 1) - 7.5).abs() < 0.01, "midday at high's 8 minutes: {}", tph(&a, 1));
    let cost = |a: &TrackApi| (0..5).map(|k| a.running_cost_period(k)).collect::<Vec<_>>();
    let before = cost(&a);
    assert_eq!(a.set_line_row_headways(b, vec![480, 1200, 480, 900, 1800]), 0, "{}", a.issues());
    assert!((tph(&a, 1) - 3.0).abs() < 0.01 && (tph(&a, 0) - 7.5).abs() < 0.01 && (tph(&a, 2) - 7.5).abs() < 0.01, "a 20 minute midday: {:?}", a.w.net.line_tph[b as usize]);
    let after = cost(&a);
    assert!(after[1] < before[1] && [0, 2, 3, 4].iter().all(|&k| after[k] == before[k]), "only midday's cost moves: {before:?} {after:?}");
    // its buses by count, then a midday count of its own
    assert_eq!(a.clear_auto_headway(b), 0, "{}", a.issues());
    let trip = a.w.svc.lines[b as usize].round_trip[1];
    let mut t = a.line_row_trains(b);
    t[1] = 1;
    assert_eq!(a.set_line_row_trains(b, t.clone()), 0, "{}", a.issues());
    assert!((tph(&a, 1) - 3600.0 / trip).abs() < 0.01, "one midday bus over a {trip} s round trip: {}", tph(&a, 1));
    t[1] = 3;
    assert_eq!(a.set_line_row_trains(b, t), 0, "{}", a.issues());
    assert!((tph(&a, 1) - 3.0 * 3600.0 / trip).abs() < 0.01, "three midday buses: {}", tph(&a, 1));
}

/// T-3767: a bus line follows the game's hours: the morning from 5 puts an hour of its 5 minute
/// morning into the night's slot, (5 x 3 + 12) / 6 = 4.5 an hour (hours_tests `t3542_morning_from_five`).
#[test]
fn a_bus_line_follows_the_games_hours() {
    let mut a = api();
    assert_eq!(a.set_game_periods(5), 0, "{}", a.issues());
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 300, 900, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    assert!((a.w.net.line_tph[b as usize][4] - 3.0).abs() < 1e-3);
    assert_eq!(a.set_game_hours(vec![5, 10, 16, 20, 24]), 0, "{}", a.issues());
    assert_eq!(a.line_hours(b), vec![5, 10, 16, 20, 24]);
    let night = a.w.net.line_tph[b as usize][4];
    assert!((night - 4.5).abs() < 1e-3, "night slot {night} against 4.5");
    // its own periods and hours, as a rail line's
    assert_eq!(a.set_line_periods(b, 4), 0, "{}", a.issues());
    assert_eq!((a.line_period_count(b), a.line_periods_own(b)), (4, 4));
    assert_eq!(a.set_line_hours(b, vec![6, 10, 16, 20, 24]), 0, "{}", a.issues());
    assert_eq!(a.line_hours_own(b), vec![6, 10, 16, 20, 24]);
    assert!((a.w.net.line_tph[b as usize][4] - 3.0).abs() < 1e-3);
}

/// T-3767: a save from before bus lines followed the game's rows (level 28) loads with its bus lines
/// on three rows of their own at the default hours, as it ran; one of this level follows the game.
#[test]
fn an_older_saves_bus_lines_keep_three_rows() {
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 900, 1800, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    assert_eq!(a.set_game_periods(5), 0, "{}", a.issues());
    assert_eq!(a.set_game_hours(vec![5, 10, 16, 20, 24]), 0, "{}", a.issues());
    let saved = a.save();
    assert!(a.load(&saved));
    assert_eq!((a.line_period_count(b), a.line_periods_own(b), a.line_hours(b)), (5, 0, vec![5, 10, 16, 20, 24]));
    a.set_load_level(crate::track::save::OLD_BUS_ROWS);
    assert!(a.load(&saved));
    assert_eq!((a.line_period_count(b), a.line_periods_own(b), a.line_hours(b)), (3, 3, CUTS_DEFAULT.iter().map(|&c| c as u32).collect::<Vec<_>>()));
    assert_eq!(a.line_row_headways(b), vec![480, 900, 1800]);
    // a game of three rows at the default hours is left as it was
    let mut c = api();
    let b = c.new_line_id();
    assert_eq!(c.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 900, 1800, 30.0, 120.0, 0, "B".into()), 0, "{}", c.issues());
    let saved = c.save();
    c.set_load_level(crate::track::save::OLD_BUS_ROWS);
    assert!(c.load(&saved));
    assert_eq!(c.line_periods_own(b), 0);
    assert_eq!(c.save(), saved);
}

/// T-3907: a BRT line runs at no less than its floor along its whole route, costs `BRT_COST_PER_KM` a road-km on a
/// constructed line (nothing back when switched off or undone), upkeeps at `UPKEEP_BRT_KM`, is kept by a save and a
/// redraw, and a planned line's BRT is in the blueprint.
#[test]
fn a_brt_line_runs_at_its_floor_is_paid_for_saved_and_undone() {
    use crate::track::streets::{bus_kmh, lane_kmh};
    let mut a = api();
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    let road = 6000.0 * ROAD_FACTOR;
    // the first 3.6 km of road dense, the rest open country (as `a_bus_in_dense_streets_runs_slower_and_needs_more_buses`)
    let (pop, jobs) = ([30_000.0f32, 30_000.0, 50.0], [20_000.0f32, 20_000.0, 5.0]);
    assert!(a.add_pack_streets(2000.0, -1000.0, -1000.0, 3, 1, &pop, &jobs, &[0, 1, 2], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758));
    let time = |a: &TrackApi| *a.stop_times(b, 0, 0).last().unwrap();
    let plain = time(&a);
    let want_plain = 3600.0 / (crate::track::streets::BUS_LANE_KMH / 3.6) + 3600.0 / (bus_kmh(44.0) / 3.6) + 18.0 * 28.0;
    assert!((plain - want_plain).abs() < 0.04 * want_plain, "{plain} {want_plain}");
    // planned: switching on is free now and in the blueprint as its own row (kind 16)
    let cash = a.cash();
    assert_eq!(a.set_bus_brt(b, true), 0, "{}", a.issues());
    assert!(a.bus_brt(b) && a.cash() == cash);
    // in town: the dense 3.6 km of road, where a plain bus is under BRT's floor; open country's 37.4 is not
    let town = a.bus_town_m(b);
    assert!((town - 3600.0).abs() <= 100.0, "{town}");
    let rows = a.w.net.blueprint_cost_items();
    let brt_row = rows.chunks_exact(7).find(|r| r[0] == 16.0).expect("a BRT row");
    assert!((brt_row[5] - town).abs() < 1e-6 && (brt_row[6] - town / 1000.0 * BRT_COST_PER_KM).abs() < 1e-6, "{brt_row:?}");
    assert!((a.w.net.total_cost() - a.w.net.built_cost() - rows.chunks_exact(7).map(|r| r[6]).sum::<f64>()).abs() < 1e-8);
    // the dense streets rise from 25 to BRT's floor; open country's 37.4 is over it and stays
    let brt = time(&a);
    let want = 3600.0 / (lane_kmh(true) / 3.6) + 3600.0 / (bus_kmh(44.0).max(lane_kmh(true)) / 3.6) + 18.0 * 28.0;
    assert!((brt - want).abs() < 0.04 * want && brt < plain, "{brt} {want} {plain}");
    // off again, constructed plain, then on: paid at once, the rise in built value
    assert_eq!(a.set_bus_brt(b, false), 0);
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    let cash = a.cash();
    assert!((time(&a) - plain).abs() < 1e-6);
    assert_eq!(a.set_bus_brt(b, true), 0, "{}", a.issues());
    assert!((cash - a.cash() - town / 1000.0 * BRT_COST_PER_KM).abs() < 1e-6, "{}", cash - a.cash());
    assert!((time(&a) - brt).abs() < 1e-6);
    // the upkeep: street tram track's a road-km in town, a plain route's out of town
    let up = a.upkeep();
    assert_eq!(UPKEEP_BRT_KM, UPKEEP_GROUND_KM * TrackKind::StreetTram.spec().upkeep);
    assert!((up[5] - ((road - town) * UPKEEP_BUS_KM + town * UPKEEP_BRT_KM) / 1000.0 * ECONOMY / 1e6).abs() < 1e-9, "{up:?}");
    // saved and loaded (`LBR1`)
    let saved = a.save();
    assert!(saved.windows(4).any(|w| w == b"LBR1"));
    assert!(a.load(&saved));
    assert!(a.bus_brt(b) && (time(&a) - brt).abs() < 1e-6);
    // one undo step: off, nothing back; redo: on, nothing charged again
    let cash = a.cash();
    assert_eq!(a.set_bus_brt(b, false), 0);
    assert!(!a.bus_brt(b) && a.cash() == cash && (time(&a) - plain).abs() < 1e-6);
    assert!(a.can_undo());
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert!(a.bus_brt(b) && a.cash() == cash && (time(&a) - brt).abs() < 1e-6);
    assert_eq!(a.redo(), 0, "{}", a.issues());
    assert!(!a.bus_brt(b) && a.cash() == cash);
    assert_eq!(a.undo(), 0);
    // a redraw keeps it
    assert_eq!(a.set_bus_route(b, &[0.0, 0.0, 1.0, 6500.0, 0.0, 1.0]), 0, "{}", a.issues());
    assert!(a.bus_brt(b));
    // a save with no BRT line writes no block and loads with none
    assert_eq!(a.set_bus_brt(b, false), 0);
    let plain_save = a.save();
    assert!(!plain_save.windows(4).any(|w| w == b"LBR1"));
    assert!(a.load(&plain_save) && !a.bus_brt(b));
}

/// T-3378: two BRT lines on one road share its lanes: the second pays only its own road in town, run either way and
/// a carriageway apart; the lanes' upkeep is paid once, split between them; switching one off or deleting it costs
/// nothing; a planned line on constructed lanes adds nothing to the blueprint; and a save keeps it all.
#[test]
fn brt_lines_on_one_road_share_its_lanes() {
    let mut a = api();
    let (pop, jobs) = ([30_000.0f32, 30_000.0, 50.0], [20_000.0f32, 20_000.0, 5.0]);
    assert!(a.add_pack_streets(2000.0, -1000.0, -1000.0, 3, 1, &pop, &jobs, &[0, 1, 2], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758));
    let line = |a: &mut TrackApi, pts: &[f64]| {
        let l = a.new_line_id();
        assert_eq!(a.set_bus_line(l, pts, 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
        assert_eq!(a.construct_line(l), 0, "{}", a.issues());
        l
    };
    let price = |m: f64| m / 1000.0 * BRT_COST_PER_KM;
    let l1 = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    // the other way, 10 m over, and on 500 m further west, in the dense streets too
    let l2 = line(&mut a, &[6000.0, 10.0, 1.0, -500.0, 10.0, 1.0]);
    let (t1, t2) = (a.bus_town_m(l1), a.bus_town_m(l2));
    assert!((t2 - t1 - 500.0 * ROAD_FACTOR).abs() <= 100.0, "{t1} {t2}");
    let cash = a.cash();
    assert_eq!(a.set_bus_brt(l1, true), 0, "{}", a.issues());
    assert!((cash - a.cash() - price(t1)).abs() < 1e-6, "{}", cash - a.cash());
    // the second pays its 600 road-m west of the first, and its hover says so
    let lanes = a.bus_lanes(l2);
    assert!((lanes[0] - t2).abs() < 1e-6 && (lanes[1] - 500.0 * ROAD_FACTOR).abs() <= 100.0 && lanes[2] == 0.0, "{lanes:?}");
    let cash = a.cash();
    assert_eq!(a.set_bus_brt(l2, true), 0, "{}", a.issues());
    let paid = cash - a.cash();
    assert!((paid - price(lanes[1])).abs() <= price(100.0) + 1e-6, "{paid} {lanes:?}");
    let union = a.w.net.brt_lane_m(true);
    assert!((union - t1 - lanes[1]).abs() <= 100.0, "{union}");
    // upkeep: the shared road once, half on each line's share; each line's own road at a plain route's rate
    let parts = a.w.net.brt_upkeep_m();
    assert!((parts[l1 as usize] + parts[l2 as usize] - union).abs() <= 100.0, "{parts:?} {union}");
    assert!((parts[l1 as usize] - 0.5 * t1).abs() <= 100.0, "{parts:?}");
    let road = a.bus_road_m(l1) + a.bus_road_m(l2);
    let want = ((road - parts[l1 as usize] - parts[l2 as usize]) * UPKEEP_BUS_KM + (parts[l1 as usize] + parts[l2 as usize]) * UPKEEP_BRT_KM) / 1000.0 * ECONOMY / 1e6;
    assert!((a.upkeep()[5] - want).abs() < 1e-9, "{} {want}", a.upkeep()[5]);
    // a planned third line on the first's road: no blueprint row for its lanes
    let l3 = a.new_line_id();
    assert_eq!(a.set_bus_line(l3, &[100.0, 0.0, 1.0, 2900.0, 0.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0, "C".into()), 0, "{}", a.issues());
    assert_eq!(a.set_bus_brt(l3, true), 0, "{}", a.issues());
    let rows = a.w.net.blueprint_cost_items();
    assert!(rows.chunks_exact(7).filter(|r| r[0] == 16.0).all(|r| r[5] <= 100.0), "{rows:?}");
    assert!((a.w.net.total_cost() - a.w.net.built_cost() - rows.chunks_exact(7).map(|r| r[6]).sum::<f64>()).abs() < 1e-8);
    // a save keeps the sharing
    let built = a.w.net.built_cost();
    let saved = a.save();
    assert!(a.load(&saved));
    assert!((a.w.net.built_cost() - built).abs() < 1e-9);
    // off and deleted: nothing charged
    let cash = a.cash();
    assert_eq!(a.set_bus_brt(l1, false), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash);
    assert_eq!(a.set_bus_brt(l1, true), 0, "{}", a.issues());
    assert!(cash - a.cash() <= price(100.0) + 1e-6, "{}", cash - a.cash());
    let cash = a.cash();
    assert_eq!(a.remove_line(l1), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash);
}

/// T-3378 stage 2: a BRT lane takes `LANE_BUSES_H` buses an hour each way; past it the lines on it run fewer, each
/// cut by the same share, plain lines on the road are not, and the panel's reads say so.
#[test]
fn a_full_brt_lane_cuts_its_lines_buses() {
    use crate::track::bus::lanes::LANE_BUSES_H;
    let mut a = api();
    let (pop, jobs) = ([30_000.0f32, 30_000.0, 50.0], [20_000.0f32, 20_000.0, 5.0]);
    assert!(a.add_pack_streets(2000.0, -1000.0, -1000.0, 3, 1, &pop, &jobs, &[0, 1, 2], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758));
    // a bus every 90 s each: 40 an hour each way, 80 on the shared road
    let line = |a: &mut TrackApi, pts: &[f64]| {
        let l = a.new_line_id();
        assert_eq!(a.set_bus_line(l, pts, 90, 90, 90, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
        assert_eq!(a.construct_line(l), 0, "{}", a.issues());
        l
    };
    let l1 = line(&mut a, &[0.0, 0.0, 1.0, 2800.0, 0.0, 1.0]);
    let l2 = line(&mut a, &[-400.0, 10.0, 1.0, 2800.0, 10.0, 1.0]);
    let lev = a.w.net.period_slot()[0];
    let tph = |a: &TrackApi, l: u32| a.w.net.line_tph[l as usize][lev] as f64;
    let (p1, p2) = (tph(&a, l1), tph(&a, l2));
    assert!(p1 > 38.0 && p2 > 38.0, "{p1} {p2}");
    // on plain lines nothing is cut, and the lane read is empty
    assert_eq!(a.bus_lane_load(l1)[0], 0.0);
    assert_eq!(a.set_bus_brt(l1, true), 0, "{}", a.issues());
    assert!((tph(&a, l1) - p1).abs() < 0.5, "{} {p1}", tph(&a, l1));
    // both on BRT: cut to fit, each by about the same share
    assert_eq!(a.set_bus_brt(l2, true), 0, "{}", a.issues());
    let (b1, b2) = (tph(&a, l1), tph(&a, l2));
    assert!(b1 + b2 <= LANE_BUSES_H + 0.5 && b1 + b2 >= LANE_BUSES_H - 2.0, "{b1} {b2}");
    assert!((b1 / p1 - b2 / p2).abs() < 0.05, "{b1} {b2} {p1} {p2}");
    let load = a.bus_lane_load(l2);
    assert!((load[0] - b1 - b2).abs() < 1e-3 && load[5] == LANE_BUSES_H, "{load:?}");
    // a headway the lane can't give shows as limited by it (`LANE_KIND`)
    let held = a.line_held(l1);
    let limits = &held[7 * ROWS..12 * ROWS];
    assert!(limits.chunks_exact(5).any(|r| r[1] == crate::track::capacity::LANE_KIND as f64 && r[0] > 90.0), "{held:?}");
    // a plain line on the same road neither counts nor is cut
    let l3 = line(&mut a, &[0.0, 5.0, 1.0, 2800.0, 5.0, 1.0]);
    assert!(tph(&a, l3) > 38.0 && (tph(&a, l1) - b1).abs() < 0.5, "{} {}", tph(&a, l3), tph(&a, l1));
    // one off BRT: the other runs all its buses again
    assert_eq!(a.set_bus_brt(l2, false), 0, "{}", a.issues());
    assert!(tph(&a, l1) > 38.0 && tph(&a, l2) > 38.0, "{} {}", tph(&a, l1), tph(&a, l2));
}
