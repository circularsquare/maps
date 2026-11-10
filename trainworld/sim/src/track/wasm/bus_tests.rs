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
    // construct: the road at $1M a km and the buses at $0.6M
    let quote = a.line_build_cost(b);
    assert!((quote - road / 1000.0).abs() < 1e-6, "{quote}");
    assert_eq!(a.construct_line(b), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[b as usize];
    assert!(ls.running);
    // T-2919: a bus is a third of a train car, 80 at crush and rated at two thirds of it
    assert!(ls.crush() == 80.0 && (ls.rated() - 160.0 / 3.0).abs() < 1e-9, "{} {}", ls.crush(), ls.rated());
    let buses = ls.trains[0];
    assert!(buses > 0 && a.fleet_bus() == buses && a.buses_needed() == buses);
    assert!((cash - a.cash() - (road / 1000.0 + buses as f64 * BUS_PRICE)).abs() < 1e-6, "{} {}", cash - a.cash(), buses);
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
    // the first 3 km of ground (3.6 km of road) in the 30 km/h lane, the rest at open country's 39.6, and 18 stops
    let want = 3600.0 / (30.0 / 3.6) + 3600.0 / (39.6 / 3.6) + 18.0 * 28.0;
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
