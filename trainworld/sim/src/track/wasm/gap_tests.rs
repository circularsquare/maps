//! T-1896: the gap shown when a stop cannot be added for want of track.
use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

fn route(a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64) {
    let ends = [0.0, 0.0, x0, y0, 0.0, 0.0, 0.0, x1, y1, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
}

fn station(a: &mut TrackApi, x: f64, y: f64, name: &str) -> u32 {
    let net = &a.w.net;
    let e = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e)).min_by(|&p, &q| net.project(p, x, y).1.total_cmp(&net.project(q, x, y).1)).unwrap();
    assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{name}: {}", a.issues());
    let net = &a.w.net;
    (0..net.node_count() as u32).filter(|&k| net.is_station(k)).min_by(|&p, &q| {
        let d = |k: u32| (net.node_x[k as usize] - x).hypot(net.node_y[k as usize] - y);
        d(p).total_cmp(&d(q))
    }).unwrap()
}

fn near(got: [f64; 4], want: [f64; 4]) -> bool {
    (got[0] - want[0]).hypot(got[1] - want[1]) < 1.0 && (got[2] - want[2]).hypot(got[3] - want[3]) < 1.0
}

/// End to end: a main line that stops 500 m short of the next stretch; the gap is between the two
/// track ends, not the far ends or the stub elsewhere. A branch built up to the main line but not
/// joined: the gap runs from its end straight to the main line. A station on the line's own
/// track has none.
#[test]
fn the_gap_is_between_the_nearest_loose_ends() {
    let mut a = api();
    route(&mut a, 0.0, 0.0, 9000.0, 0.0);
    route(&mut a, 9500.0, 0.0, 15000.0, 0.0);
    route(&mut a, 30000.0, 8000.0, 32000.0, 8000.0);
    route(&mut a, 5000.0, 400.0, 5000.0, 3000.0);
    let s1 = station(&mut a, 1000.0, 0.0, "A");
    let s2 = station(&mut a, 3000.0, 0.0, "B");
    let far = station(&mut a, 12000.0, 0.0, "C");
    let branch = station(&mut a, 5000.0, 2000.0, "D");
    let stub = station(&mut a, 31000.0, 8000.0, "E");
    let gap = |a: &TrackApi, n: u32| gap::gap_between(&a.w.net, &[s1, s2], n);
    assert!(near(gap(&a, far).unwrap(), [9000.0, 0.0, 9500.0, 0.0]), "{:?}", gap(&a, far));
    assert!(near(gap(&a, branch).unwrap(), [5000.0, 0.0, 5000.0, 400.0]), "{:?}", gap(&a, branch));
    assert!(near(gap(&a, stub).unwrap(), [9000.0, 0.0, 30000.0, 8000.0]), "{:?}", gap(&a, stub));
    assert_eq!(gap(&a, s2), None);
    // and through the API, as longitude and latitude
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[s1, s2], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let ll = a.stop_gap(id, far);
    assert_eq!(ll.len(), 4);
    let f = a.w.net.frame();
    let (x, y) = f.from_lonlat(ll[0], ll[1]);
    assert!((x - 9000.0).abs() < 1.0 && y.abs() < 1.0, "{x} {y}");
    assert!(a.stop_gap(id, s2).is_empty());
}
