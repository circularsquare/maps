use super::*;
use crate::track::profile::state_at;

#[test]
fn profile_runs_flat_and_stands_between() {
    let p = profile(&[0.0, 1000.0, 3000.0], &[0.0, 30.0, 0.0], &[], &Pace::default());
    let (pace, hop) = (Pace::default(), Pace::default().auto_time(0.0, 1000.0));
    assert_eq!(p.arr, vec![0.0, hop, hop + 30.0 + pace.auto_time(1000.0, 3000.0)]);
    assert_eq!(p.dep[1] - p.arr[1], 30.0);
    assert!((p.duration - p.arr[2]).abs() < 1e-9);
    assert_eq!(state_at(&p.phases, hop / 2.0).0, 500.0);
    assert_eq!(state_at(&p.phases, hop + 10.0), (1000.0, 0.0));
    assert_eq!(state_at(&p.phases, p.duration + 100.0), (3000.0, 0.0));
}

#[test]
fn plan_on_an_empty_network() {
    let net = Network::new(-73.985, 40.758, true);
    // 1 km: two ends, no middle stop, street stops every 400 m between
    let p = plan(&net, 0, &[[0.0, 0.0], [1000.0, 0.0]], &[], false);
    assert_eq!(p.stops.len(), 2);
    assert_eq!((p.stops[0].kind, p.stops[1].kind), (KIND_END, KIND_END));
    assert_eq!(p.street.len(), 2);
    assert_eq!(p.stops[0].cover, Some([0.0, 600.0]));
    assert_eq!(p.stops[1].cover, Some([600.0, 1200.0]));
    // 10 km of drawn route, 12 km of road: five middles
    let p = plan(&net, 3, &[[0.0, 0.0], [5000.0, 0.0], [5000.0, 5000.0]], &[], false);
    assert_eq!(p.stops.iter().filter(|s| s.kind == KIND_MIDDLE).count(), 5);
    assert!(p.stops.iter().enumerate().all(|(k, s)| s.id == stop_id(3, k)));
    assert!((p.len - 12_000.0).abs() < 1e-6);
    assert_eq!(plan(&net, 0, &[[0.0, 0.0], [50.0, 0.0]], &[], false), Plan::default());
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
    Pace { step: 100.0, car: vec![car; (km * 10.0).ceil() as usize], trunk: vec![] }
}

fn hop_kmh(p: &Pace, m: f64) -> f64 {
    m / p.hop_time(0.0, m) * 3.6
}

#[test]
fn a_hop_is_the_cars_time_at_nine_tenths_plus_the_stop() {
    use super::super::streets::car_street_kmh;
    let dense = flat(16.5, 30.0);
    let mid = flat(car_street_kmh(5000.0), 30.0);
    let open = Pace::default();
    // the dense street's lane and mid density's are the same 30 km/h; open country its car's 39.6
    for (m, want) in [(400.0, 18.95), (800.0, 23.23), (1000.0, 24.32), (2000.0, 26.87)] {
        assert!((hop_kmh(&dense, m) - want).abs() < 0.01, "{m}: {}", hop_kmh(&dense, m));
    }
    assert!((hop_kmh(&mid, 400.0) - 18.95).abs() < 0.01);
    assert!((hop_kmh(&open, 400.0) - 22.37).abs() < 0.01, "{}", hop_kmh(&open, 400.0));
    // 20 km of open country: 2 km of local at each end at 39.6, 8 km of main road each end at 79.2
    let want = 20_000.0 / (4000.0 / 11.0 + 16_000.0 / 22.0 + 28.0) * 3.6;
    assert!((hop_kmh(&open, 20_000.0) - want).abs() < 1e-6 && (want - 64.3).abs() < 0.1, "{want}");
    // 40 km: 20 km of it open highway at 99
    let want = 40_000.0 / (4000.0 / 11.0 + 16_000.0 / 22.0 + 20_000.0 / 27.5 + 28.0) * 3.6;
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
        // five 400 m hops, as a line spaced at 400 m runs them, and one automatic 2 km hop with its street stops
        let stops: Vec<f64> = (0..6).map(|i| i as f64 * 400.0).collect();
        let spaced = spaced_profile(&stops, &[0.0; 6], &pace);
        let auto = profile(&[0.0, 2000.0], &[0.0, 0.0], &[400.0, 800.0, 1200.0, 1600.0], &pace);
        assert!((spaced.duration - auto.duration).abs() < 1e-6, "{} {}", spaced.duration, auto.duration);
        assert!((pace.auto_time(0.0, 400.0) - pace.hop_time(0.0, 400.0)).abs() < 1e-9);
    }
}

#[test]
fn a_trunk_road_runs_at_highway_speed_even_on_a_short_hop() {
    let mut p = flat(16.5, 2.0);
    p.trunk = vec![true; p.car.len()];
    // 400 m at 79.2 km/h and the stop
    let want = 400.0 / (400.0 / 22.0 + 28.0) * 3.6;
    assert!((hop_kmh(&p, 400.0) - want).abs() < 1e-6, "{}", hop_kmh(&p, 400.0));
    assert!(hop_kmh(&p, 400.0) > hop_kmh(&flat(16.5, 2.0), 400.0) + 10.0);
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
    // 100 m steps: 90 km/h moving for the first 300 m, 36 for the next 200, then 90 again
    let pace = Pace { step: 100.0, car: vec![100.0, 100.0, 100.0, 40.0, 40.0, 100.0, 100.0, 100.0, 100.0, 100.0], trunk: vec![] };
    let t = |p: &Pace, a: f64, b: f64| p.seconds(a, b, street_bus_kmh);
    assert!((t(&pace, 0.0, 1000.0) - (12.0 + 20.0 + 20.0)).abs() < 1e-9);
    // inside one step, across two
    assert!((t(&pace, 120.0, 180.0) - 2.4).abs() < 1e-9);
    assert!((t(&pace, 250.0, 350.0) - (2.0 + 5.0)).abs() < 1e-9);
    // past the end counts at the last step
    assert!((t(&pace, 900.0, 1100.0) - 8.0).abs() < 1e-9);
    // the same route from its other end
    assert!((t(&pace.reversed(), 0.0, 1000.0) - 52.0).abs() < 1e-9);
    assert!((t(&pace.reversed(), 0.0, 500.0) - 20.0).abs() < 1e-9);
    // the profile runs each hop at its mean
    let p = profile(&[0.0, 500.0, 1000.0], &[0.0, 0.0, 0.0], &[], &pace);
    assert!((p.arr[1] - pace.auto_time(0.0, 500.0)).abs() < 1e-9 && (p.arr[2] - pace.auto_time(0.0, 1000.0)).abs() < 1e-9);
    assert!((p.phases[0].v - 500.0 / pace.auto_time(0.0, 500.0)).abs() < 1e-9);
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
