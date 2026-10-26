use super::*;
use crate::track::profile::state_at;

#[test]
fn profile_runs_flat_and_stands_between() {
    let p = profile(&[0.0, 1000.0, 3000.0], &[0.0, 30.0, 0.0]);
    let hop = 1000.0 / BUS_V;
    assert_eq!(p.arr, vec![0.0, hop, hop + 30.0 + 2000.0 / BUS_V]);
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
    let (cx, _) = pa.at_road(x[0]);
    assert!((cx - 3275.0).abs() < 1e-6);
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
