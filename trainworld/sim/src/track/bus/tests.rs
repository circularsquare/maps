use super::*;
use crate::track::profile::state_at;

#[test]
fn profile_runs_flat_and_stands_between() {
    let p = profile(&[0.0, 1000.0, 3000.0], &[0.0, 30.0, 0.0], &[], &Pace::default());
    let (pace, hop) = (Pace::default(), Pace::default().auto_time(0.0, 1000.0));
    assert_eq!(p.arr, vec![0.0, hop, hop + 30.0 + pace.auto_time(1000.0, 3000.0)]);
    assert_eq!(p.dep[1] - p.arr[1], 30.0);
    assert!((p.duration - p.arr[2]).abs() < 1e-9);
    assert!((state_at(&p.phases, hop / 2.0).0 - 500.0).abs() < 1e-6);
    assert_eq!(state_at(&p.phases, hop + 10.0), (1000.0, 0.0));
    assert_eq!(state_at(&p.phases, p.duration + 100.0), (3000.0, 0.0));
}

/// T-3109: a route out 2 km, round a 1 km square and back: a station and a stop placed by hand on the way out are
/// stops on the way back too; one on the loop is a stop once.
#[test]
fn a_route_that_loops_back_stops_both_ways_on_its_way_out() {
    use crate::track::net::{NodeData, StationData};
    let mut net = Network::new(-73.985, 40.758, true);
    let n = net.alloc_node();
    net.set_node(n, Some(NodeData { x: 500.0, y: 40.0, level: 0, flying: false })).unwrap();
    net.set_station(n, Some(StationData { platform: 2, name: "S".into(), named: false, built: true, off: false, upgrades: 0, berth: None, day: 0, one: 0, fare: 0, own: String::new(), road: None })).unwrap();
    let pts = [[0.0, 0.0], [2000.0, 0.0], [3000.0, 0.0], [3000.0, 1000.0], [2000.0, 1000.0], [2000.0, 0.0], [0.0, 0.0]];
    assert_eq!(loop_back_at(&pts), Some(1));
    assert_eq!(loop_back_at(&[[0.0, 0.0], [2000.0, 0.0], [3000.0, 0.0], [3000.0, 1000.0], [0.0, 0.0]]), None, "a plain loop");
    let p = plan(&net, 0, &pts, &[], false);
    let at: Vec<f64> = p.stops.iter().filter(|s| s.station == Some(n)).map(|s| s.s).collect();
    assert_eq!(at.len(), 2, "the station both ways: {at:?}");
    assert!((at[0] - 500.0 * ROAD_FACTOR).abs() < 1.0 && (p.len - at[1] - 500.0 * ROAD_FACTOR).abs() < 1.0);
    let hand = [[1500.0, 0.0], [3000.0, 500.0]];
    let p = plan_by(&net, 0, &pts, &[], false, StopRule::Hand(&hand));
    let middles = p.stops.iter().filter(|s| s.kind == KIND_MIDDLE).count();
    assert_eq!(middles, 3, "{:?}", p.stops.iter().map(|s| s.s).collect::<Vec<_>>());
}

#[test]
fn plan_on_an_empty_network() {
    let net = Network::new(-73.985, 40.758, true);
    // 1 km drawn, 1.2 km of road: two ends and a stop every 400 m between, each one demand sees (T-3924)
    let p = plan(&net, 0, &[[0.0, 0.0], [1000.0, 0.0]], &[], false);
    assert_eq!(p.stops.iter().map(|s| (s.kind, s.s.round() as i64)).collect::<Vec<_>>(), vec![(KIND_END, 0), (KIND_MIDDLE, 400), (KIND_MIDDLE, 800), (KIND_END, 1200)]);
    assert!(p.street.is_empty());
    assert_eq!(p.stops[0].cover, Some([0.0, 200.0]));
    assert_eq!(p.stops[1].cover, Some([200.0, 600.0]));
    assert_eq!(p.stops[3].cover, Some([1000.0, 1200.0]));
    // 10 km of drawn route, 12 km of road: 29 middles
    let p = plan(&net, 3, &[[0.0, 0.0], [5000.0, 0.0], [5000.0, 5000.0]], &[], false);
    assert_eq!(p.stops.iter().filter(|s| s.kind == KIND_MIDDLE).count(), 29);
    assert!(p.stops.iter().enumerate().all(|(k, s)| s.id == stop_id(3, k)));
    assert!((p.len - 12_000.0).abs() < 1e-6);
    assert_eq!(plan(&net, 0, &[[0.0, 0.0], [50.0, 0.0]], &[], false), Plan::default());
    // 1,000 km of road: the grid widens so the stop ids still fit
    let p = plan(&net, 1, &[[0.0, 0.0], [1_000_000.0 / ROAD_FACTOR, 0.0]], &[], false);
    assert!(p.stops.len() <= SPACED_MAX_STOPS as usize + 1 && p.stops.len() > 900, "{}", p.stops.len());
}

#[test]
fn a_spaced_plan_has_no_street_stops_and_stops_serve_only_near_them() {
    let net = Network::new(-73.985, 40.758, true);
    let pts = [[0.0, 0.0], [10_000.0, 0.0]];
    let kinds = [K_CORNER, K_ROAD | K_CORNER];
    let p = plan_spaced(&net, 2, &pts, &kinds, false, 2000);
    assert_eq!(p.stops.iter().map(|s| s.s.round() as i64).collect::<Vec<_>>(), vec![0, 2000, 4000, 6000, 8000, 10_000]);
    assert!(p.street.is_empty());
    assert_eq!(p.stops[1].kind, KIND_MIDDLE);
    assert_eq!(p.stops[0].cover, Some([0.0, 400.0]));
    assert_eq!(p.stops[1].cover, Some([1600.0, 2400.0]));
    assert_eq!(p.stops[5].cover, Some([9600.0, 10_000.0]));
    // the last hop is never under half the spacing
    let p = plan_spaced(&net, 2, &pts, &kinds, false, 4500);
    assert_eq!(p.stops.iter().map(|s| s.s.round() as i64).collect::<Vec<_>>(), vec![0, 4500, 10_000]);
    // 0 is the automatic plan exactly
    assert_eq!(plan_spaced(&net, 2, &pts, &kinds, false, 0), plan(&net, 2, &pts, &kinds, false));
}

#[test]
fn spaced_stops_prefer_stations_near_their_place_and_pass_the_rest() {
    let st = |s: f64, n: u32| (s, KIND_STATION, Some(n));
    let ends = |mid: Vec<(f64, u8, Option<u32>)>| {
        let mut v = vec![(0.0, KIND_END, None)];
        v.extend(mid);
        v.push((10_000.0, KIND_END, None));
        v
    };
    // every 5 km: the station at 4,000 is nearer 5,000 than the one at 3,000, and a crossing nearer still loses to it
    let got = spaced_stops(&ends(vec![st(3000.0, 1), st(4000.0, 2), (4900.0, KIND_CROSSING, None)]), 5000.0);
    assert_eq!(got.iter().map(|c| (c.0.round() as i64, c.2)).collect::<Vec<_>>(), vec![(0, None), (4000, Some(2)), (10_000, None)]);
    // no station in the window: a crossing there, else the place itself
    let got = spaced_stops(&ends(vec![(4000.0, KIND_CROSSING, None)]), 5000.0);
    assert_eq!(got[1], (4000.0, KIND_CROSSING, None));
    let got = spaced_stops(&ends(vec![st(1000.0, 1)]), 5000.0);
    assert_eq!(got[1], (5000.0, KIND_MIDDLE, None));
}

#[test]
fn a_spaced_trip_takes_each_hops_time_and_stands_inside_it() {
    let pace = Pace::default();
    let hop = pace.hop_time(0.0, 2000.0);
    let p = spaced_profile(&[0.0, 2000.0, 4000.0], &[0.0, 0.0, 0.0], &pace);
    assert!((p.arr[1] - (hop - 20.0)).abs() < 1e-9 && (p.dep[1] - hop).abs() < 1e-9 && (p.duration - 2.0 * hop).abs() < 1e-9, "{p:?}");
    // a station keeps its own dwell and no extra stand
    let p = spaced_profile(&[0.0, 2000.0, 4000.0], &[0.0, 30.0, 0.0], &pace);
    assert!((p.arr[1] - hop).abs() < 1e-9 && (p.dep[1] - (hop + 30.0)).abs() < 1e-9, "{p:?}");
    assert!((spaced_profile(&[0.0, 400.0], &[0.0, 0.0], &pace).duration - pace.hop_time(0.0, 400.0)).abs() < 1e-9);
}

/// A route with the car's local streets at `car` km/h all the way, of `km` km.
fn flat(car: f64, km: f64) -> Pace {
    Pace { step: 100.0, car: vec![car; (km * 10.0).ceil() as usize], trunk: vec![], lane: 0.0, busway: vec![] }
}

fn hop_kmh(p: &Pace, m: f64) -> f64 {
    m / p.hop_time(0.0, m) * 3.6
}

#[test]
fn a_hop_is_the_cars_time_at_its_share_plus_the_stop() {
    use super::super::streets::{car_street_kmh, BUS_SHARE};
    let dense = flat(16.5, 30.0);
    let mid = flat(car_street_kmh(5000.0), 30.0);
    let open = Pace::default();
    // the dense street at the lane's floor, mid density at its share of the car or the floor; open country its car's
    // 44 times the share (T-3907: a 25 floor, 16.8 km/h over 400 m)
    let at = |kmh: f64, m: f64| m / (m / (kmh / 3.6) + 28.0) * 3.6;
    for m in [400.0, 800.0, 1000.0, 2000.0] {
        assert!((hop_kmh(&dense, m) - at(super::super::streets::BUS_LANE_KMH, m)).abs() < 1e-9, "{m}: {}", hop_kmh(&dense, m));
    }
    assert!((at(25.0, 400.0) - 16.82).abs() < 0.01);
    assert!((hop_kmh(&mid, 400.0) - at(super::super::streets::bus_kmh(car_street_kmh(5000.0)), 400.0)).abs() < 1e-9);
    // m/s on open country's local streets, main roads and highway
    let (local, main, open_hw) = (44.0 * BUS_SHARE / 3.6, 88.0 * BUS_SHARE / 3.6, 110.0 * BUS_SHARE / 3.6);
    let want = 400.0 / (400.0 / local + 28.0) * 3.6;
    assert!((hop_kmh(&open, 400.0) - want).abs() < 1e-6 && (want - 21.65).abs() < 0.01, "{} {want}", hop_kmh(&open, 400.0));
    // 20 km of open country: 2 km of local at each end, 8 km of main road each end
    let want = 20_000.0 / (4000.0 / local + 16_000.0 / main + 28.0) * 3.6;
    assert!((hop_kmh(&open, 20_000.0) - want).abs() < 1e-6 && (want - 60.9).abs() < 0.1, "{want}");
    // 40 km: 20 km of it open highway
    let want = 40_000.0 / (4000.0 / local + 16_000.0 / main + 20_000.0 / open_hw + 28.0) * 3.6;
    assert!((hop_kmh(&open, 40_000.0) - want).abs() < 1e-6);
}

#[test]
fn spacing_rises_smoothly_and_400_m_matches_automatic_stops() {
    for pace in [flat(16.5, 30.0), flat(30.25, 30.0), flat(44.0, 30.0), Pace::default()] {
        let mut last = 0.0;
        for m in [400.0, 600.0, 800.0, 1000.0, 2000.0, 4000.0, 5000.0, 8000.0, 10_000.0, 20_000.0] {
            let v = hop_kmh(&pace, m);
            assert!(v > last, "{m}: {v} after {last}");
            last = v;
        }
        // five 400 m hops, as a line spaced at 400 m and an automatic line run them, each standing at its stops
        let stops: Vec<f64> = (0..6).map(|i| i as f64 * 400.0).collect();
        let spaced = spaced_profile(&stops, &[0.0; 6], &pace);
        let auto = profile(&stops, &[0.0; 6], &[], &pace);
        assert!((spaced.duration - auto.duration).abs() < 1e-6, "{} {}", spaced.duration, auto.duration);
        assert!((spaced.dwell - auto.dwell).abs() < 1e-6 && auto.dwell > 0.0, "{} {}", spaced.dwell, auto.dwell);
        assert!((pace.auto_time(0.0, 400.0) - pace.hop_time(0.0, 400.0)).abs() < 1e-9);
    }
}

#[test]
fn a_trunk_road_runs_at_highway_speed_even_on_a_short_hop() {
    let mut p = flat(16.5, 2.0);
    p.trunk = vec![true; p.car.len()];
    // T-3907: in a dense centre 400 m at `BUS_TRUNK_MAX_X` times the bus's street floor (under the share of 88) and the stop
    let want = 400.0 / (400.0 / (super::super::streets::BUS_TRUNK_MAX_X * super::super::streets::BUS_LANE_KMH / 3.6) + 28.0) * 3.6;
    assert!((hop_kmh(&p, 400.0) - want).abs() < 1e-6, "{}", hop_kmh(&p, 400.0));
    assert!(hop_kmh(&p, 400.0) > hop_kmh(&flat(16.5, 2.0), 400.0) + 3.0);
    // automatic stops too
    assert!((p.auto_time(0.0, 400.0) - p.hop_time(0.0, 400.0)).abs() < 1e-9);
    // a route's kinds mark the stretches: the first 1 km local, the second on a trunk road
    let net = Network::new(-73.985, 40.758, true);
    let (pts, kinds) = ([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]], [K_CORNER, K_ROAD, K_ROAD | K_TRUNK]);
    let pace = Pace::along(&net, &pts, &[0.0, 1000.0, 2000.0], &kinds, 2000.0);
    assert!(!pace.trunk[4] && pace.trunk[15], "{pace:?}");
    assert!(pace.reversed().trunk[4] && !pace.reversed().trunk[15]);
    assert_eq!(Pace::along(&net, &pts, &[0.0, 1000.0, 2000.0], &[K_CORNER, K_ROAD, K_ROAD], 2000.0), Pace::default());
}

#[test]
fn a_road_stretch_counts_at_its_drawn_length() {
    let net = Network::new(-73.985, 40.758, true);
    let pts = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0]];
    // the first stretch straight (1.2), the second on road (1.0)
    let kinds = [K_CORNER, K_CORNER, K_ROAD | K_CORNER];
    let poly = Poly::new(&net, &pts, &kinds);
    assert!((poly.road_len() - 2200.0).abs() < 1e-6 && (poly.len() - 2000.0).abs() < 1e-6);
    assert_eq!(poly.at_road(600.0), (500.0, 0.0));
    let (x, y) = poly.at_road(1700.0);
    assert!((x - 1000.0).abs() < 1e-6 && (y - 500.0).abs() < 1e-6);
    let (s, d) = poly.project(&net, 1010.0, 400.0);
    assert!((s - 1600.0).abs() < 1e-6 && (d - 10.0).abs() < 1e-6);
    let p = plan(&net, 0, &pts, &kinds, false);
    assert!((p.len - 2200.0).abs() < 1e-6);
    assert_eq!(p.cum, poly.road);
    assert_eq!(at_road(&pts, &p.cum, 1200.0), (1000.0, 0.0));
}

#[test]
fn crossings_of_long_routes_are_found_in_road_metres() {
    let net = Network::new(-73.985, 40.758, true);
    // a zigzag of 100 points along x and a straight route across it at x = 3275
    let a: Vec<[f64; 2]> = (0..100).map(|i| [i as f64 * 50.0, if i % 2 == 0 { 0.0 } else { 20.0 }]).collect();
    let b = [[3275.0, -500.0], [3275.0, 500.0]];
    let kinds = vec![K_ROAD; 100];
    let (pa, pb) = (Poly::new(&net, &a, &kinds), Poly::new(&net, &b, &[]));
    let x = crossings(&pa, &pb);
    assert_eq!(x.len(), 1);
    let (cx, _) = pa.at_road(x[0].0);
    assert!((cx - 3275.0).abs() < 1e-6);
    // and along b: 510 m of straight route to y = 10, in road metres
    assert!((x[0].1 - 510.0 * ROAD_FACTOR).abs() < 1e-6, "{x:?}");
    assert!(crossings(&pb, &pa).len() == 1);
}

#[test]
fn simplify_keeps_the_shape_within_the_tolerance() {
    let line: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64, if i % 10 == 0 { 0.0 } else { 0.4 })).collect();
    let s = simplify(&line, 1.0);
    assert_eq!(s.first(), line.first());
    assert_eq!(s.last(), line.last());
    assert_eq!(s.len(), 2);
    assert!(simplify(&line, 0.1).len() > 30);
}

#[test]
fn a_hop_takes_the_time_of_the_streets_it_crosses() {
    // 100 m steps: the car at 100 km/h for the first 300 m, 40 for the next 200, then 100 again; the bus at its share
    let pace = Pace { step: 100.0, car: vec![100.0, 100.0, 100.0, 40.0, 40.0, 100.0, 100.0, 100.0, 100.0, 100.0], trunk: vec![], lane: 0.0, busway: vec![] };
    let (fast, slow) = (100.0 * super::super::streets::BUS_SHARE / 3.6, 40.0 * super::super::streets::BUS_SHARE / 3.6);
    assert!(slow * 3.6 > 30.0, "the slow stretch is above the bus lane's floor");
    let t = |p: &Pace, a: f64, b: f64| p.seconds(a, b, street_bus_kmh_at);
    let all = 800.0 / fast + 200.0 / slow;
    assert!((t(&pace, 0.0, 1000.0) - all).abs() < 1e-9);
    // inside one step, across two
    assert!((t(&pace, 120.0, 180.0) - 60.0 / fast).abs() < 1e-9);
    assert!((t(&pace, 250.0, 350.0) - (50.0 / fast + 50.0 / slow)).abs() < 1e-9);
    // past the end counts at the last step
    assert!((t(&pace, 900.0, 1100.0) - 200.0 / fast).abs() < 1e-9);
    // the same route from its other end
    assert!((t(&pace.reversed(), 0.0, 1000.0) - all).abs() < 1e-9);
    assert!((t(&pace.reversed(), 0.0, 500.0) - 500.0 / fast).abs() < 1e-9);
    // the profile speeds up from each stop and brakes into the next, in the hop's time
    // (standing at the stop between inside the first hop's time)
    let p = profile(&[0.0, 500.0, 1000.0], &[0.0, 0.0, 0.0], &[], &pace);
    assert!((p.dep[1] - pace.auto_time(0.0, 500.0)).abs() < 1e-9 && p.arr[1] < p.dep[1] && (p.arr[2] - pace.auto_time(0.0, 1000.0)).abs() < 1e-9);
    assert_eq!((p.phases[0].v, p.phases[0].a), (0.0, BUS_ACCEL));
    let (hop, mid) = (p.arr[1], 0.5 * p.arr[1]);
    let (s, v) = state_at(&p.phases, mid);
    assert!((s - 250.0).abs() < 1e-6 && v > 500.0 / hop, "fastest in the middle: {v}");
    assert!(state_at(&p.phases, 1.0).1 < 1.2);
}

/// T-3908: every stretch between standstills takes the time it took at constant speed, whatever its length,
/// and the speed never needs more than `BUS_ACCEL` unless the stretch is too short to cruise.
#[test]
fn a_ramp_takes_exactly_the_stretch_time() {
    for (d, dur) in [(400.0, 58.0), (2000.0, 200.0), (100.0, 14.0), (30.0, 3.0), (5000.0, 300.0)] {
        let mut ph = vec![];
        ramp(&mut ph, 10.0, 100.0, d, dur);
        ph.push(crate::track::profile::Phase { t: 10.0 + dur, s: 100.0 + d, v: 0.0, a: 0.0 });
        let (s, v) = crate::track::profile::state_at(&ph, 10.0 + dur - 1e-9);
        assert!((s - (100.0 + d)).abs() < 1e-3 && v < 1e-3, "{d} m in {dur} s: {s} {v}");
        let (s, _) = crate::track::profile::state_at(&ph, 10.0 + 0.5 * dur);
        assert!((s - (100.0 + 0.5 * d)).abs() < 1e-6);
        assert!(ph.iter().all(|p| p.a.abs() <= BUS_ACCEL + 1e-9) || d / dur > 0.2 * BUS_ACCEL * dur, "{d} {dur}");
    }
}

#[test]
fn a_bus_stands_at_its_street_stops_within_the_hops_time() {
    let stops = [0.0, 2000.0];
    let street = [400.0, 800.0, 1200.0, 1600.0];
    let flat = profile(&stops, &[0.0, 0.0], &[], &Pace::default());
    let p = profile(&stops, &[0.0, 0.0], &street, &Pace::default());
    // the trip takes the same time, so demand's run times do not move
    assert!((p.duration - flat.duration).abs() < 1e-9 && (p.arr[1] - flat.arr[1]).abs() < 1e-9);
    // standing STREET_DWELL_S at each street stop, then away
    for &s in &street {
        let t = crate::track::profile::time_at(&p.phases, s);
        assert_eq!(crate::track::profile::state_at(&p.phases, t - STREET_DWELL_S / 2.0), (s, 0.0), "at {s}");
        assert!(crate::track::profile::state_at(&p.phases, t + 1.0).1 > 0.0);
    }
    // never behind the flat run by more than the stands, never ahead of the end
    assert_eq!(crate::track::profile::state_at(&p.phases, p.duration + 50.0), (2000.0, 0.0));
    // a short hop gives up at most 40% of its time to standing
    let short = profile(&[0.0, 100.0], &[0.0, 0.0], &[50.0], &Pace::default());
    let flat = profile(&[0.0, 100.0], &[0.0, 0.0], &[], &Pace::default());
    assert!((short.duration - flat.duration).abs() < 1e-9);
    assert!(short.phases.iter().all(|p| p.v < 30.0));
}

#[test]
fn a_route_through_dense_streets_runs_slower() {
    use super::super::streets::{StreetGrid, CAR_TOWN_KMH};
    let mut net = Network::new(-73.985, 40.758, true);
    let pts = [[0.0, 0.0], [6000.0, 0.0]];
    let poly = Poly::new(&net, &pts, &[]);
    let len = poly.road_len();
    assert_eq!(Pace::along(&net, &pts, &poly.road, &[], len), Pace::default());
    // the pack's frame at the network's origin; the zone square on x 0..2 km is dense, the rest open
    let (pop, jobs) = ([40_000.0f32, 10.0, 10.0], [40_000.0f32, 10.0, 10.0]);
    let g = StreetGrid::from_zones(2000.0, -1000.0, -1000.0, 4, 1, &pop, &jobs, &[0, 1, 2], &[0, 1, 2], &[0, 0, 0], &[], &[1, 1, 1], -73.985, 40.758).unwrap();
    net.streets.grids.push(std::sync::Arc::new(g));
    let pace = Pace::along(&net, &pts, &poly.road, &[], len);
    assert!((pace.car[2] - 16.5).abs() < 1.0 && pace.car[30] > 40.0, "{} {}", pace.car[2], pace.car[30]);
    assert!(pace.auto_time(0.0, len) > Pace::default().auto_time(0.0, len));
    // where no grid covers, open country
    assert_eq!(net.car_kmh_at(30_000.0, 0.0), CAR_TOWN_KMH);
}

/// T-3907: a trunk road lifts a bus to at most `BUS_TRUNK_MAX_X` times its line's street speed at the place (35 in a dense
/// centre, 52 on open country's streets), never under the road's speed without it; a BRT line takes the same cap and
/// never less than its floor.
#[test]
fn a_trunk_road_is_capped_at_its_share_over_the_street_speed_and_brt_keeps_its_floor() {
    use super::super::streets::{bus_kmh, lane_kmh, BUS_LANE_KMH, BUS_SHARE, BUS_TRUNK_MAX_X};
    let trunk = |car: f64| {
        let mut p = flat(car, 30.0);
        p.trunk = vec![true; p.car.len()];
        p
    };
    // the moving speed over 2 km of local street on a trunk road, no stop
    let moving = |p: &Pace| 2000.0 / p.seconds(0.0, 2000.0, street_bus_kmh_at) * 3.6;
    let (dense, open) = (trunk(16.5), trunk(44.0));
    assert!((moving(&dense) - BUS_TRUNK_MAX_X * BUS_LANE_KMH).abs() < 1e-9, "{}", moving(&dense));
    assert!((moving(&open) - BUS_TRUNK_MAX_X * bus_kmh(44.0)).abs() < 1e-9 && moving(&open) < 88.0 * BUS_SHARE, "{}", moving(&open));
    // a long hop on a dense trunk road: its main roads are capped too, so the whole 20 km runs at the cap
    assert!((20_000.0 / (dense.hop_time(0.0, 20_000.0) - 28.0) * 3.6 - BUS_TRUNK_MAX_X * BUS_LANE_KMH).abs() < 1e-9);
    // open country's main roads are no slower for the trunk: 2 km of streets each end at 52, 16 km at the share of 88
    let want = 20_000.0 / (4000.0 / (BUS_TRUNK_MAX_X * bus_kmh(44.0) / 3.6) + 16_000.0 / (88.0 * BUS_SHARE / 3.6) + 28.0) * 3.6;
    assert!((hop_kmh(&open, 20_000.0) - want).abs() < 1e-9 && hop_kmh(&open, 20_000.0) > hop_kmh(&flat(44.0, 30.0), 20_000.0), "{want}");
    // off the trunk road nothing changes
    assert!((moving(&flat(16.5, 30.0)) - BUS_LANE_KMH).abs() < 1e-9);
    // BRT: the dense street at its floor, the dense trunk road at the plain cap above it
    assert_eq!(lane_kmh(true), BUS_LANE_KMH + crate::track::params::BRT_LANE_BONUS_KMH);
    assert!((moving(&flat(16.5, 30.0).brt(true)) - lane_kmh(true)).abs() < 1e-9);
    // the BRT line's own street speed (40) times the share: 60
    assert!((moving(&dense.clone().brt(true)) - BUS_TRUNK_MAX_X * lane_kmh(true)).abs() < 1e-9);
    // open country streets (37.4) are over the floor: out of town, BRT changes nothing there
    assert!((moving(&flat(44.0, 30.0).brt(true)) - bus_kmh(44.0).max(lane_kmh(true))).abs() < 1e-9);
    assert_eq!(flat(44.0, 30.0).town_m(3000.0), 0.0);
    assert_eq!(flat(16.5, 30.0).town_m(3000.0), 3000.0);
    assert_eq!(Pace::default().town_m(3000.0), 0.0);
    assert!((moving(&flat(100.0, 30.0).brt(true)) - 100.0 * BUS_SHARE).abs() < 1e-9);
    // and both ways
    assert!((moving(&flat(16.5, 30.0).brt(true).reversed()) - lane_kmh(true)).abs() < 1e-9);
    assert_eq!(Pace::default().brt(true).auto_time(0.0, 1000.0), Pace::default().auto_time(0.0, 1000.0));
}

/// T-3378: a stop that costs less (a BRT line at a BRT station) comes that much sooner, and with no dwell of its
/// own the bus stands what the stop costs, on either rule.
#[test]
fn a_cheaper_stop_comes_sooner_and_stands_its_cost() {
    let pace = Pace::default();
    let (s, dw) = ([0.0, 2000.0, 4000.0], [0.0, 0.0, 0.0]);
    for spaced in [false, true] {
        let plain = profile_cut(&s, &dw, &[], &[], &pace, spaced);
        let cut = profile_cut(&s, &dw, &[0.0, 10.0, 0.0], &[], &pace, spaced);
        assert!((plain.arr[1] - cut.arr[1] - 10.0 - (cut.dep[1] - cut.arr[1]) + (plain.dep[1] - plain.arr[1])).abs() < 1e-9, "{spaced}");
        assert!((cut.dep[1] - cut.arr[1] - 18.0).abs() < 1e-9, "{spaced} {:?}", cut);
        assert!((plain.duration - cut.duration - 10.0).abs() < 1e-9, "{spaced}");
    }
}
