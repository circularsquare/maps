//! T-4249: an edit rebuilds the bus lines and ways back it reaches, and leaves every line as rebuilding them all would
//! (`TrackWorld::rebuild_all`, the twin each test compares with).

use super::*;
use crate::track::bus::KIND_CROSSING;

fn api(full: bool) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a.set_rebuild_all(full);
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

/// A planned bus line over `pts` (x, y, kind each).
fn bus(a: &mut TrackApi, pts: &[f64]) -> u32 {
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, pts, 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    b
}

fn stops(a: &TrackApi, l: u32) -> Vec<Vec<f64>> {
    a.bus_stops(l).chunks_exact(8).map(|c| c.to_vec()).collect()
}

/// Blueprint track from (x0, y0) to (x1, y1) no line uses, then a station at its far end: the station's node.
fn new_station(a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64) -> u32 {
    assert_eq!(a.add_route(&[0.0, 0.0, x0, y0, -1.0, 0.0, 0.0, x1, y1, -1.0], &[], 2, false), 0, "{}", a.issues());
    let n = a.route_end();
    a.take_service_moved();
    assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, "New".into()), 0, "{}", a.issues());
    n
}

fn rebuilt(a: &TrackApi) -> Vec<u32> {
    a.last.as_ref().map_or(vec![], |r| r.dirty.service.rebuilt.clone())
}

/// What the game shows and demand reads of every line: its service, rates, departures, counts and an hour's trips.
fn figures(a: &TrackApi) -> String {
    format!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}", a.w.svc.lines, a.w.net.line_tph, a.w.net.line_phase, a.w.net.line_trains, a.trips(8.0 * 3600.0))
}

/// After an edit: the lines it rebuilt, whether any line's service moved, the figures.
fn step(a: &mut TrackApi) -> (Vec<u32>, bool, String) {
    (rebuilt(a), a.take_service_moved(), figures(a))
}

#[test]
fn a_station_rebuilds_the_bus_lines_near_it_and_no_others() {
    let mut runs = vec![];
    for full in [false, true] {
        let mut a = api(full);
        let (rl, _) = rail(&mut a, 4, 2000.0);
        let near = bus(&mut a, &[-1000.0, 1500.0, 1.0, 5000.0, 1500.0, 1.0]);
        let far = bus(&mut a, &[-1000.0, 8000.0, 1.0, 5000.0, 8000.0, 1.0]);
        let mut steps = vec![];
        // 100 m from the near line
        let n1 = new_station(&mut a, 500.0, 1400.0, 1500.0, 1400.0);
        assert!(stops(&a, near).iter().any(|c| c[5] == n1 as f64), "the near line stops there");
        steps.push(step(&mut a));
        // 3 km from the near line, 3.5 km from the far one
        new_station(&mut a, 500.0, 4500.0, 1500.0, 4500.0);
        steps.push(step(&mut a));
        assert_eq!(a.remove_station(n1), 0, "{}", a.issues());
        assert!(!stops(&a, near).iter().any(|c| c[5] == n1 as f64));
        steps.push(step(&mut a));
        // a rail line's own edit: no bus line reads it
        assert_eq!(a.set_line_cars(rl, 4), 0, "{}", a.issues());
        steps.push(step(&mut a));
        runs.push((rl, near, far, steps));
    }
    let (rl, near, far, scoped) = &runs[0];
    let all = &runs[1].3;
    assert_eq!((scoped[0].0.clone(), scoped[0].1), (vec![*near], true), "the near line only");
    assert_eq!((scoped[1].0.clone(), scoped[1].1), (vec![], false), "none, and no train runs otherwise");
    assert_eq!((scoped[2].0.clone(), scoped[2].1), (vec![*near], true));
    assert_eq!(scoped[3].0, vec![*rl]);
    assert!(all.iter().all(|s| s.0.contains(near) && s.0.contains(far)), "as before: every bus line");
    for (k, (s, f)) in scoped.iter().zip(all).enumerate() {
        assert!(s.2 == f.2, "step {k}: the lines as rebuilding every one leaves them");
    }
}

#[test]
fn a_constructed_bus_line_waits_for_the_station_to_be_constructed() {
    let mut runs = vec![];
    for full in [false, true] {
        let mut a = api(full);
        rail(&mut a, 4, 2000.0);
        let b = bus(&mut a, &[-1000.0, 1500.0, 1.0, 5000.0, 1500.0, 1.0]);
        assert_eq!(a.construct_line(b), 0, "{}", a.issues());
        let mut steps = vec![];
        let n = new_station(&mut a, 500.0, 1400.0, 1500.0, 1400.0);
        assert!(!stops(&a, b).iter().any(|c| c[5] == n as f64), "a constructed route stops at constructed stations only");
        steps.push(step(&mut a));
        a.take_service_moved();
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        assert!(stops(&a, b).iter().any(|c| c[5] == n as f64));
        steps.push(step(&mut a));
        runs.push((b, steps));
    }
    let (b, scoped) = &runs[0];
    assert!(!scoped[0].0.contains(b) && !scoped[0].1, "placed: not rebuilt, and no train runs otherwise");
    assert!(scoped[1].0.contains(b) && scoped[1].1, "constructed: rebuilt");
    for (k, (s, f)) in scoped.iter().zip(&runs[1].1).enumerate() {
        assert!(s.2 == f.2, "step {k}: the lines as rebuilding every one leaves them");
    }
}

/// Where two bus routes cross near a station both reach, they change there; a station near one route only can take
/// that route's stop from the shared station (of two within `STATION_GAP_M` along a route, the nearer), so the other
/// route stops at the crossing: it is rebuilt though the station is out of its reach.
#[test]
fn a_station_near_a_crossing_route_moves_the_other_routes_crossing_stop() {
    let mut figs = vec![];
    for full in [false, true] {
        let mut a = api(full);
        let (_, st) = rail(&mut a, 4, 2000.0);
        // both along the roads (road metres are ground metres): A 150 m north of the rail line, B 100 m east of its
        // station at x = 2000, crossing A at (2100, 150)
        let la = bus(&mut a, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 3.0]);
        let lb = bus(&mut a, &[2100.0, 2000.0, 1.0, 2100.0, -2000.0, 3.0]);
        let crossing = |a: &TrackApi, l: u32| stops(a, l).iter().any(|c| c[4] == KIND_CROSSING as f64 && (c[1] - 2100.0).abs() < 1.0 && (c[2] - 150.0).abs() < 1.0);
        assert!(!crossing(&a, la) && !crossing(&a, lb), "both reach station {} and change there", st[1]);
        // 40 m from B, 210 m from A, 60 m along B from the station
        let s = new_station(&mut a, 1660.0, -60.0, 2060.0, -60.0);
        assert!(stops(&a, lb).iter().any(|c| c[5] == s as f64) && !stops(&a, lb).iter().any(|c| c[5] == st[1] as f64), "B stops at the nearer station");
        assert!(!stops(&a, la).iter().any(|c| c[5] == s as f64), "out of A's reach");
        assert!(crossing(&a, la), "A stops where B crosses");
        if !full {
            assert!(rebuilt(&a).contains(&la) && rebuilt(&a).contains(&lb), "{:?}", rebuilt(&a));
        }
        figs.push(figures(&a));
    }
    assert!(figs[0] == figs[1]);
}

/// A ring of double track whose bottom side is a pair of one-way singles, east and west (as
/// `loop_way_tests::a_loop_with_a_way_back_of_its_own_runs_both_ways`), and a loop line round it whose way back
/// takes the westbound leg: the line, the legs (eastbound first) and a way to set a leg one way.
fn ring(a: &mut TrackApi) -> (u32, [u32; 2]) {
    let free = |x: f64, y: f64| [0.0, 0.0, x, y, 0.0];
    let ends = |p: [f64; 5], q: [f64; 5]| p.iter().chain(q.iter()).copied().collect::<Vec<f64>>();
    let near = |a: &TrackApi, x: f64, y: f64| {
        let n = &a.w.net;
        let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
        (0..n.node_count() as u32).filter(|&k| n.node_ok(k)).min_by(|&p, &q| d(p).total_cmp(&d(q))).unwrap()
    };
    let at = |a: &TrackApi, n: u32| [1.0, n as f64, a.w.net.node_x[n as usize], a.w.net.node_y[n as usize], 0.0];
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(2000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(free(4000.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let (j0, j1) = (near(a, 2000.0, 0.0), near(a, 4000.0, 0.0));
    for y in [75.0, -75.0] {
        assert_eq!(a.add_route(&ends(at(a, j0), at(a, j1)), &[3000.0, y, 0.0, 0.0], 1, false), 0, "{}", a.issues());
    }
    let (w0, e0) = (near(a, 0.0, 0.0), near(a, 6000.0, 0.0));
    let round = [7500.0, 0.0, 0.0, 0.0, 7500.0, 3000.0, 0.0, 0.0, -1500.0, 3000.0, 0.0, 0.0, -1500.0, 0.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends(at(a, e0), at(a, w0)), &round, 2, false), 0, "{}", a.issues());
    let net = &a.w.net;
    let legs: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_tracks[e as usize] == 1).collect();
    assert_eq!(legs.len(), 2);
    let east = |net: &crate::track::net::Network, e: u32| if net.node_x[net.edge_a[e as usize] as usize] < net.node_x[net.edge_b[e as usize] as usize] { 1u8 } else { 2 };
    let west = |net: &crate::track::net::Network, e: u32| 3 - east(net, e);
    let (wa, wb) = (east(net, legs[0]), west(net, legs[1]));
    assert_eq!(a.set_edge_oneway(legs[0], wa), 0, "{}", a.issues());
    assert_eq!(a.set_edge_oneway(legs[1], wb), 0, "{}", a.issues());
    let mut st = vec![];
    for (x, y) in [(1000.0, 0.0), (5000.0, 0.0), (3000.0, 3000.0)] {
        let n = &a.w.net;
        let e = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).min_by(|&p, &q| n.project(p, x, y).1.total_cmp(&n.project(q, x, y).1)).unwrap();
        let (px, py, _) = crate::track::geom::pos_at(n.edge_pieces(e), n.project(e, x, y).0);
        assert_eq!(a.add_station(2, e, px, py, 120, "S".into()), 0, "{}", a.issues());
        st.push(near(a, px, py));
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[st[0], st[1], st[2], st[0]], 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issue_kinds());
    assert!(!a.line_one_way(l), "run 1 goes round by its own leg");
    assert!(a.w.net.back_lines().contains(&l));
    (l, [legs[0], legs[1]])
}

#[test]
fn a_way_back_is_rebuilt_where_the_edit_reaches_it() {
    let mut runs = vec![];
    for full in [false, true] {
        let mut a = api(full);
        let (l, legs) = ring(&mut a);
        let on = |a: &TrackApi, r: usize, e: u32| a.w.svc.lines[l as usize].runs[r].segs.iter().any(|&p| p >> 1 == e);
        assert!(on(&a, 0, legs[0]) && !on(&a, 0, legs[1]) && on(&a, 1, legs[1]) && !on(&a, 1, legs[0]), "out on the eastbound leg, back on the westbound one");
        let mut steps = vec![];
        // far off, on track no line uses
        new_station(&mut a, 0.0, -6000.0, 1000.0, -6000.0);
        steps.push(step(&mut a));
        // on the westbound leg, which only the way back runs over
        let n = &a.w.net;
        let (px, py, _) = crate::track::geom::pos_at(n.edge_pieces(legs[1]), n.project(legs[1], 2500.0, -75.0).0);
        assert_eq!(a.add_station(2, legs[1], px, py, 120, "W".into()), 0, "{}", a.issues());
        steps.push(step(&mut a));
        // the service left behind the track by hand (no edit does this): the next edit anywhere finds the way back moved
        let back: Vec<u32> = {
            let n = &a.w.net;
            (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e) && n.edge_tracks[e as usize] == 1 && !on(&a, 0, e)).collect()
        };
        assert!(!back.is_empty());
        let w = &mut a.w.net;
        for e in back {
            let mut d = w.edge_data(e).unwrap();
            d.oneway = 3 - d.oneway;
            w.set_edge(e, Some(d)).unwrap();
        }
        let t = w.take_touch();
        w.derive(&t);
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, -7000.0, -1.0, 0.0, 0.0, 1000.0, -7000.0, -1.0], &[], 2, false), 0, "{}", a.issues());
        assert!(a.line_one_way_by_track(l), "both legs east: round one way");
        steps.push(step(&mut a));
        runs.push((l, steps));
    }
    let (l, scoped) = &runs[0];
    let all = &runs[1].1;
    assert!(!scoped[0].0.contains(l) && all[0].0.contains(l), "a station far off: not in scope");
    assert!(scoped[1].0.contains(l), "a station on its way back's track");
    assert!(scoped[2].0.contains(l), "a way back a rebuild would find otherwise");
    for (k, (s, f)) in scoped.iter().zip(all).enumerate() {
        assert!(s.2 == f.2, "step {k}: the lines as rebuilding every one leaves them");
    }
}
