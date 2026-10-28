use super::*;

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn nearest(a: &TrackApi, edges: &[u32], x: f64, y: f64) -> u32 {
    *edges.iter().min_by(|&&p, &&q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap()
}

/// T-1685: a route drawn beside a curved run of single track (two pieces) runs `QUAD_SPACING`
/// from it all along, curves included, on the side its end is; without `set_route_beside` the
/// same ends make a plain route that does not.
#[test]
fn route_beside_track_runs_parallel() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0], &[], 1, false), 0, "{}", a.issues());
    let j = a.route_end();
    assert_eq!(a.add_route(&[1.0, j as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 2600.0, 1400.0, 0.0], &[1800.0, 0.0, 600.0, 0.0], 1, false), 0, "{}", a.issues());
    let local = live(&a);
    let (s, e) = ((300.0, 20.0), (2350.0, 1150.0));
    let ends = [0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, e.0, e.1, 0.0];
    a.set_route_beside(nearest(&a, &local, s.0, s.1), nearest(&a, &local, e.0, e.1));
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let new: Vec<u32> = live(&a).into_iter().filter(|e| !local.contains(e)).collect();
    let mut worst: f64 = 0.0;
    let mut n = 0;
    for &p in &new {
        for q in a.edge_render(p).chunks(3) {
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            worst = worst.max((off - QUAD_SPACING).abs());
            n += 1;
            assert!(q[1] > 0.0, "on the end's side");
        }
    }
    assert!(n > 20 && worst < 0.05, "{n} points, worst {worst} m off the spacing");
    // the same ends drawn plain: a straight across the curve
    a.set_route_beside(u32::MAX, u32::MAX);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert!(pv[3] <= 3.0, "no corners: {}", pv[3]);
    // the end to the right of the track: the start goes to the right too
    let r = (2500.0, 1000.0);
    a.set_route_beside(nearest(&a, &local, s.0, s.1), nearest(&a, &local, r.0, r.1));
    let pv = a.preview_route(&[0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, r.0, r.1, 0.0], &[], 2, false);
    assert!((pv[6] + QUAD_SPACING).abs() < 0.01, "start at {}", pv[6]);
}
