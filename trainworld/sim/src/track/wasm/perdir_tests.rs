//! T-3348 stage 1: per-direction stops.

use super::*;

fn api(right_hand: bool) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, right_hand);
    a.set_cash(1e9);
    a
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn node_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k)).min_by(|&p, &q| d(p).total_cmp(&d(q))).unwrap()
}
fn at_node(a: &TrackApi, n: u32) -> [f64; 5] {
    [1.0, n as f64, a.w.net.node_x[n as usize], a.w.net.node_y[n as usize], 0.0]
}
/// A station splitting the track nearest (x, y), at its point nearest.
fn station_near(a: &mut TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    let e = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).min_by(|&p, &q| n.project(p, x, y).1.total_cmp(&n.project(q, x, y).1)).unwrap();
    let (px, py, _) = crate::track::geom::pos_at(n.edge_pieces(e), n.project(e, x, y).0);
    assert_eq!(a.add_station(2, e, px, py, 120, "S".into()), 0, "{}", a.issues());
    node_near(a, px, py)
}
fn end_station(a: &mut TrackApi, x: f64, y: f64) -> u32 {
    let n = node_near(a, x, y);
    assert_eq!(a.add_station(1, n, x, y, 200, "T".into()), 0, "{}", a.issues());
    n
}
fn svc_stop(a: &TrackApi, l: u32, run: usize, k: usize) -> u32 {
    a.w.svc.lines[l as usize].stop_node(run, k).unwrap()
}

/// Her case (Anita 2026-11-13, reports/t3348-split-legs.png): a double splits into two one-way
/// singles about 150 m apart, a station on each leg, and they rejoin as double. A two-way line
/// listing one of them stops at it one way and at the other leg's station the other way, whichever
/// leg the click was on, on both running sides; nothing in between turns back.
#[test]
fn t3348_split_legs_stop_each_way_on_its_own_leg() {
    for right_hand in [true, false] {
        let mut a = api(right_hand);
        assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(2000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.add_route(&ends(free(4000.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
        let (j0, j1) = (node_near(&a, 2000.0, 0.0), node_near(&a, 4000.0, 0.0));
        for y in [75.0, -75.0] {
            assert_eq!(a.add_route(&ends(at_node(&a, j0), at_node(&a, j1)), &[3000.0, y, 0.0, 0.0], 1, false), 0, "{}", a.issues());
        }
        // the player sets the leg the tool left two-way to run the other way from its pair
        let net = &a.w.net;
        let legs: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_tracks[e as usize] == 1).collect();
        assert_eq!(legs.len(), 2);
        let east = |net: &crate::track::net::Network, e: u32, w: u8| (net.node_x[net.edge_a[e as usize] as usize] < net.node_x[net.edge_b[e as usize] as usize]) == (w == 1);
        // and sets both, since a way only inferred stops at a station (`derive_ways`)
        let k = (0..2).find(|&k| net.edge_way(legs[k]) == 0).unwrap_or(0);
        let o = legs[1 - k];
        let ow = net.edge_way(o);
        let want_east = !east(net, o, ow);
        let w = if east(net, legs[k], 1) == want_east { 1 } else { 2 };
        assert_eq!(a.set_edge_oneway(legs[k], w), 0, "{}", a.issues());
        assert_eq!(a.set_edge_oneway(o, ow), 0, "{}", a.issues());
        let net = &a.w.net;
        let ways: Vec<u8> = legs.iter().map(|&e| net.edge_way(e)).collect();
        assert!(ways.iter().all(|&w| w != 0) && east(net, legs[0], ways[0]) != east(net, legs[1], ways[1]), "right hand {right_hand}: legs run one way each, {ways:?}");
        let (p, q) = (end_station(&mut a, 0.0, 0.0), end_station(&mut a, 6000.0, 0.0));
        let (n, s) = (station_near(&mut a, 3000.0, 75.0), station_near(&mut a, 3000.0, -75.0));
        let wy = |k: u32| a.w.net.node_ports[k as usize].ends().iter().map(|&c| (a.w.net.edge_way(c >> 1), a.w.net.edge_oneway[(c >> 1) as usize])).collect::<Vec<_>>();
        assert!(a.w.net.way_limited(n) && a.w.net.way_limited(s), "right hand {right_hand}: {:?} {:?}", wy(n), wy(s));
        for click in [n, s] {
            let l = a.new_line_id();
            assert_eq!(a.set_line_with_trains(l, &[p, q], 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
            assert_eq!(a.add_stop(l, click), 0, "right hand {right_hand}: {}", a.issues());
            let net = &a.w.net;
            assert!(!net.line_broken[l as usize]);
            let stops = net.line_stops(l).to_vec();
            assert_eq!(stops.len(), 3, "one stop at the pair: {stops:?}");
            let mid = stops[1];
            let other = if mid == n { s } else { n };
            // the forward run runs along the stop's leg, the return along the other
            assert!(net.line_path(l).iter().all(|&e| net.dir_ok(e)));
            assert!(net.path_reversals(net.line_path(l)).is_empty(), "no turn back on the way out");
            let (back_stops, back) = net.back_run(l).unwrap();
            assert_eq!(back_stops, vec![q, other, p], "right hand {right_hand}");
            assert!(net.path_reversals(&back).is_empty(), "no turn back on the way back");
            let ls = &a.w.svc.lines[l as usize];
            assert!(ls.ok, "the service builds");
            assert_eq!((svc_stop(&a, l, 0, 1), svc_stop(&a, l, 1, 1)), (mid, other));
            assert!(ls.runs.iter().all(|r| !r.pass.contains(&true)));
            assert!(a.w.net.check_line(l).is_ok());
            // the way back's station goes only with the line's stop, as a listed stop does
            assert_ne!(a.remove_station(other), 0, "right hand {right_hand}: the way back's station is in use");
            assert!(a.w.net.is_station(other) && !a.w.net.line_broken[l as usize]);
            assert_eq!(a.remove_line(l), 0, "{}", a.issues());
        }
    }
}

/// Two one-track stations a few metres apart along double track (one complex), one on each track:
/// a two-way line listing either stops at it its way and at the other the other way; a lone
/// one-track station is still passed one way (T-3332).
#[test]
fn t3348_one_track_pair_on_double_stops_both_ways() {
    for right_hand in [true, false] {
        let mut a = api(right_hand);
        assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(4000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
        let (p, q) = (end_station(&mut a, 0.0, 0.0), end_station(&mut a, 4000.0, 0.0));
        a.set_station_one(&[1500.0, -5.0]);
        let e = station_near(&mut a, 1500.0, 0.0);
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, &[p, e, q], 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        let pass: Vec<bool> = a.w.svc.lines[l as usize].runs.iter().map(|r| r.passes(1)).collect();
        assert_eq!(pass.iter().filter(|&&x| x).count(), 1, "right hand {right_hand}: alone, it is passed one way, {pass:?}");
        a.set_station_one(&[1650.0, 5.0]);
        let w = station_near(&mut a, 1650.0, 0.0);
        a.set_station_one(&[]);
        let net = &a.w.net;
        assert!(net.one_side(e).is_some() && net.one_side(w).is_some() && net.one_side(e) != net.one_side(w) || net.node_one[e as usize] != net.node_one[w as usize]);
        assert!(net.complex_of(e).contains(&w));
        assert!(!net.line_broken[l as usize]);
        let ls = &a.w.svc.lines[l as usize];
        assert!(ls.ok);
        let at: Vec<u32> = (0..2).map(|r| svc_stop(&a, l, r, 1)).collect();
        assert!(at.contains(&e) && at.contains(&w), "right hand {right_hand}: each run its own platform, {at:?}");
        assert!(ls.runs.iter().all(|r| !r.pass.contains(&true)), "nobody passes");
        assert!(a.line_stop_ways(l).is_empty(), "served both ways");
    }
}
