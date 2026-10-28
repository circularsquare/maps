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
    // T-1721: from a track end 300 m off the run, coming in and then running along it
    let far = (900.0, -300.0);
    assert_eq!(a.add_route(&[0.0, 0.0, 300.0, -300.0, 0.0, 0.0, 0.0, far.0, far.1, 0.0], &[], 2, false), 0, "{}", a.issues());
    let tip = a.route_end();
    let near = (2500.0, 1000.0); // right of the curve's second leg (the left has the first route)
    let before = live(&a);
    a.set_route_beside(u32::MAX, nearest(&a, &local, near.0, near.1));
    assert_eq!(a.add_route(&[1.0, tip as f64, 0.0, 0.0, 0.0, 0.0, 0.0, near.0, near.1, 0.0], &[], 2, false), 0, "{}", a.issues());
    let came: Vec<u32> = live(&a).into_iter().filter(|e| !before.contains(e)).collect();
    let pts: Vec<f64> = came.iter().flat_map(|&p| a.edge_render(p)).collect();
    let offs: Vec<f64> = pts.chunks(3).map(|q| local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min)).collect();
    let on = offs.iter().filter(|o| (*o - QUAD_SPACING).abs() < 0.05).count();
    assert!(on * 3 > offs.len(), "{on} of {} points along the line", offs.len());
    let end = a.route_end() as usize;
    let off = local.iter().map(|&l| a.w.net.project(l, a.w.net.node_x[end], a.w.net.node_y[end]).1).fold(f64::INFINITY, f64::min);
    assert!((off - QUAD_SPACING).abs() < 0.05, "end {off} m off");
    // the plain route again, for the rest of this test
    a.set_route_beside(u32::MAX, u32::MAX);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert!(pv[3] <= 3.0, "no corners: {}", pv[3]);
    // the end to the right of the track: the start goes to the right too
    let r = (2500.0, 1000.0);
    a.set_route_beside(nearest(&a, &local, s.0, s.1), nearest(&a, &local, r.0, r.1));
    let pv = a.preview_route(&[0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, r.0, r.1, 0.0], &[], 2, false);
    assert!((pv[6] + QUAD_SPACING).abs() < 0.01, "start at {}", pv[6]);
}

/// T-1708: a route ramping over a line with a station near the crossing keeps a level stretch on
/// the bridge at least that station's platform long; with no station there it is shorter.
#[test]
fn bridge_beside_a_station_has_room_for_one() {
    let level_run = |station: bool| -> f64 {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        a.set_route_ramp(true);
        assert_eq!(a.add_route(&[0.0, 0.0, 1500.0, -800.0, 0.0, 0.0, 0.0, 1500.0, 800.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        if station {
            assert_eq!(a.add_station(2, live(&a)[0], 1500.0, 160.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let before = live(&a);
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        let xs: Vec<f64> = live(&a).into_iter().filter(|e| !before.contains(e)).flat_map(|e| a.edge_render(e)).collect::<Vec<f64>>().chunks(3).filter(|q| (q[2] - LEVEL_H).abs() < 0.01).map(|q| q[0]).collect();
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let (plain, room) = (level_run(false), level_run(true));
    assert!(room >= 200.0 && room > plain, "level stretch {room} m with a station, {plain} m without");
}

/// T-1722: beside track that ramps over a line on a slight curve, the route beside it ramps over
/// that line too.
#[test]
fn route_beside_bridges_on_a_slight_curve() {
    for turn in [0.0f64, 1.0, 3.0, 5.0, 8.0, 12.0, -3.0, -8.0] {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        a.set_route_ramp(true);
        assert_eq!(a.add_route(&[0.0, 0.0, 1500.0, -800.0, 0.0, 0.0, 0.0, 1500.0, 800.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        let t = turn.to_radians();
        let end = (1500.0 + 1500.0 * t.cos(), 1500.0 * t.sin());
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, end.0, end.1, 0.0], &[1500.0, 0.0, 0.0, 0.0], 2, false), 0, "turn {turn}: {}", a.issues());
        let local = live(&a);
        let (s, e) = ((200.0, 14.0), (end.0 - 200.0 * t.cos() - 14.0 * t.sin(), end.1 - 200.0 * t.sin() + 14.0 * t.cos()));
        a.set_route_beside(u32::MAX, nearest(&a, &local, e.0, e.1));
        assert_eq!(a.add_route(&[0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, e.0, e.1, 0.0], &[], 2, false), 0, "turn {turn}: {}", a.issues());
    }
}
