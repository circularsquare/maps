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
    let p = plan(&net, 0, &[[0.0, 0.0], [1000.0, 0.0]], false);
    assert_eq!(p.stops.len(), 2);
    assert_eq!((p.stops[0].kind, p.stops[1].kind), (KIND_END, KIND_END));
    assert_eq!(p.street.len(), 2);
    assert_eq!(p.stops[0].cover, Some([0.0, 500.0]));
    assert_eq!(p.stops[1].cover, Some([500.0, 1000.0]));
    // 10 km of drawn route, 12 km of road: five middles
    let p = plan(&net, 3, &[[0.0, 0.0], [5000.0, 0.0], [5000.0, 5000.0]], false);
    assert_eq!(p.stops.iter().filter(|s| s.kind == KIND_MIDDLE).count(), 5);
    assert!(p.stops.iter().enumerate().all(|(k, s)| s.id == stop_id(3, k)));
    assert!((p.len - 12_000.0).abs() < 1e-6);
    assert_eq!(plan(&net, 0, &[[0.0, 0.0], [50.0, 0.0]], false), Plan::default());
}
