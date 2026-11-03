use super::*;

/// Seconds to run `d` m from standstill to standstill at top speed `v` (m/s) under acceleration
/// `BOAT_A` both ways, long enough to reach it.
fn flat_run(d: f64, v: f64) -> f64 {
    d / v + v / BOAT_A
}

/// Seconds to change between speeds `u` and `v` at `BOAT_A`, and the metres it takes.
fn change(u: f64, v: f64) -> (f64, f64) {
    ((v - u).abs() / BOAT_A, (v * v - u * u).abs() / (2.0 * BOAT_A))
}

/// A route along y = 0: (x, km/h) per point, the given ones ports.
fn route(points: &[(f64, f64)], ports: &[usize]) -> FerryRoute {
    let pts = points.iter().map(|p| [p.0, 0.0]).collect();
    let kinds = (0..points.len()).map(|i| if ports.contains(&i) { K_PORT } else { K_ROUTE }).collect();
    FerryRoute::new(pts, kinds, points.iter().map(|p| p.1 / 3.6).collect(), false)
}

fn cum_x(r: &FerryRoute) -> Vec<f64> {
    r.points.iter().map(|p| p[0]).collect()
}

/// A speed as a route keeps it, m/s.
fn kept(kmh: f64) -> f64 {
    quant_v(kmh / 3.6) as f64
}

#[test]
fn a_straight_run_is_its_length_at_top_speed_plus_one_acceleration() {
    let r = route(&[(0.0, 25.0), (8000.0, 25.0)], &[0, 1]);
    let cum = cum_x(&r);
    let p = profile(&limits(&r, &cum), &[0.0, 8000.0], &[0.0, 0.0]);
    let v = kept(25.0);
    assert!((p.duration - flat_run(8000.0, v)).abs() < 1e-6, "{} {}", p.duration, flat_run(8000.0, v));
    // 46 s and 160 m to reach 25 km/h
    let (t, d) = change(0.0, v);
    assert!((t - 46.3).abs() < 0.05 && (d - 160.7).abs() < 0.1, "{t} {d}");
    // the boat is where its phases say, at rest at both ends
    let (s, sp) = super::super::profile::state_at(&p.phases, p.duration);
    assert!((s - 8000.0).abs() < 1e-6 && sp.abs() < 1e-9);
    let (s, sp) = super::super::profile::state_at(&p.phases, t);
    assert!((s - d).abs() < 1e-6 && (sp - v).abs() < 1e-9);
}

#[test]
fn a_slow_channel_and_a_port_between_cost_their_braking_and_dwell() {
    // 25 km/h, a channel at 10 km/h from 2999 to 3601 m (the 1 m either side runs at the lower of its
    // ends), 25 again, a port at 4500 m with a 60 s dwell, the last port at 6000 m
    let r = route(&[(0.0, 25.0), (2999.0, 25.0), (3000.0, 10.0), (3600.0, 10.0), (3601.0, 25.0), (4500.0, 25.0), (6000.0, 25.0)], &[0, 5, 6]);
    let cum = cum_x(&r);
    let iv = limits(&r, &cum);
    let p = profile(&iv, &[0.0, 4500.0, 6000.0], &[0.0, 60.0, 0.0]);
    let (v, w) = (kept(25.0), kept(10.0));
    let (t0, d0) = change(0.0, v);
    let (t1, d1) = change(w, v);
    let slow = 3601.0 - 2999.0;
    let hop1 = t0 + (2999.0 - d0 - d1) / v + t1 + slow / w + t1 + (4500.0 - 3601.0 - d1 - d0) / v + t0;
    let hop2 = flat_run(1500.0, v);
    assert!((p.arr[1] - hop1).abs() < 1e-6, "{} {hop1}", p.arr[1]);
    assert!((p.dep[1] - p.arr[1] - 60.0).abs() < 1e-9);
    assert!((p.duration - (hop1 + 60.0 + hop2)).abs() < 1e-6, "{} {}", p.duration, hop1 + 60.0 + hop2);
    // the other way: the same hops in reverse, as long
    let back = profile(&reversed(&iv, 6000.0), &[0.0, 1500.0, 6000.0], &[0.0, 60.0, 0.0]);
    assert!((back.duration - p.duration).abs() < 1e-6);
}

#[test]
fn a_stretch_runs_at_the_lower_of_its_two_points() {
    let r = route(&[(0.0, 10.0), (1000.0, 25.0), (1500.0, 25.0), (2000.0, 18.0)], &[0, 3]);
    assert_eq!(limits(&r, &cum_x(&r)), vec![(0.0, 1000.0, kept(10.0)), (1000.0, 1500.0, kept(25.0)), (1500.0, 2000.0, kept(18.0))]);
    // a stretch at one speed is one interval
    let r = route(&[(0.0, 25.0), (500.0, 25.0), (2000.0, 25.0)], &[0, 2]);
    assert_eq!(limits(&r, &cum_x(&r)), vec![(0.0, 2000.0, kept(25.0))]);
}

#[test]
fn speeds_are_kept_on_a_mm_grid_and_within_the_boats_range() {
    let top = quant_v(BOAT_TOP);
    assert_eq!(top, 6.944);
    assert_eq!((quant_v(0.0), quant_v(f64::NAN), quant_v(-3.0), quant_v(40.0)), (top, top, top, top));
    assert_eq!(quant_v(0.01), 0.25);
    assert_eq!(quant_v(2.77777), 2.778);
    assert_eq!(v_mm(quant_v(2.77777)), 2778);
    let r = FerryRoute::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], vec![K_PORT], vec![2.5], false);
    assert_eq!((r.kinds, r.v), (vec![K_PORT, K_ROUTE, K_ROUTE], vec![2.5, top, top]));
}
