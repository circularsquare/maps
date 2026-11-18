//! T-3565: stations on one-way single track (a leg of a one-way pair), each served once a round
//! trip, by the run along its leg, wherever it was clicked in.

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
fn new_line(a: &mut TrackApi, stops: &[u32]) -> u32 {
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, stops, 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    l
}

/// Double track (0, 0) to (2000, 0), two one-way single legs to (4000, 0) through (3000, `gap`)
/// and (3000, -`gap`), one each way, then double to (6000, 0).
fn pair_track(right_hand: bool, gap: f64) -> TrackApi {
    let mut a = api(right_hand);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(2000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(free(4000.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let (j0, j1) = (node_near(&a, 2000.0, 0.0), node_near(&a, 4000.0, 0.0));
    for y in [gap, -gap] {
        assert_eq!(a.add_route(&ends(at_node(&a, j0), at_node(&a, j1)), &[3000.0, y, 0.0, 0.0], 1, false), 0, "{}", a.issues());
    }
    let net = &a.w.net;
    let legs: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_tracks[e as usize] == 1).collect();
    assert_eq!(legs.len(), 2);
    let east = |net: &crate::track::net::Network, e: u32, w: u8| (net.node_x[net.edge_a[e as usize] as usize] < net.node_x[net.edge_b[e as usize] as usize]) == (w == 1);
    for (k, &e) in legs.iter().enumerate() {
        let w = if east(&a.w.net, e, 1) == (k == 0) { 1 } else { 2 };
        assert_eq!(a.set_edge_oneway(e, w), 0, "{}", a.issues());
    }
    let net = &a.w.net;
    assert!(east(net, legs[0], net.edge_way(legs[0])) != east(net, legs[1], net.edge_way(legs[1])), "the legs run opposite ways");
    a
}

/// `pair_track` with end stations P (0, 0) and Q (6000, 0) and a station on each leg's middle.
/// Returns (P, Q, the north leg's station, the south leg's).
fn pair(right_hand: bool, gap: f64) -> (TrackApi, [u32; 4]) {
    let mut a = pair_track(right_hand, gap);
    let (p, q) = (end_station(&mut a, 0.0, 0.0), end_station(&mut a, 6000.0, 0.0));
    let (n, s) = (station_near(&mut a, 3000.0, gap), station_near(&mut a, 3000.0, -gap));
    assert!(a.w.net.leg_heading(n).is_some() && a.w.net.leg_heading(s).is_some());
    (a, [p, q, n, s])
}

/// The checker's case (notes/T-3565-check.md): `pair_track` with stations on the doubles at 300,
/// 1200, 4800 and 5700 m (AR, LO, E12, UN; bare track ends past the outer two, as past her
/// Arismendi), on the north leg at x offsets `xn` from its middle and on the south leg at `xs`.
fn pair_wide(right_hand: bool, gap: f64, xn: &[f64], xs: &[f64]) -> (TrackApi, [u32; 4], Vec<u32>, Vec<u32>) {
    let mut a = pair_track(right_hand, gap);
    let ends = [300.0, 1200.0, 4800.0, 5700.0].map(|x| station_near(&mut a, x, 0.0));
    let ns: Vec<u32> = xn.iter().map(|&dx| station_near(&mut a, 3000.0 + dx, gap)).collect();
    let ss: Vec<u32> = xs.iter().map(|&dx| station_near(&mut a, 3000.0 + dx, -gap)).collect();
    assert!(ns.iter().chain(&ss).all(|&s| a.w.net.leg_heading(s).is_some()));
    (a, ends, ns, ss)
}

/// Line `l`'s two runs: no turn back on either, each one-way station in `legs` stopped at by exactly
/// one run, at its own platform, every other stop by both, the way back about as long as the way
/// out, each one-way station drawn and listed one way. Returns both runs' times.
fn assert_served_once(a: &TrackApi, l: u32, legs: &[u32], what: &str) -> (f64, f64) {
    let net = &a.w.net;
    assert!(!net.line_broken[l as usize] && net.check_line(l).is_ok(), "{what}: the line runs, {:?}", net.check_line(l));
    let ls = &a.w.svc.lines[l as usize];
    assert!(ls.ok, "{what}: the service builds");
    let n = ls.stops.len();
    for (r, run) in ls.runs.iter().enumerate() {
        assert!(net.path_reversals(&run.segs).is_empty(), "{what}: run {r} turns back, stops {:?}", ls.stops);
    }
    for (k, &st) in ls.stops.iter().enumerate() {
        let stop = |r: usize| !ls.runs[r].passes(if r == 0 { k } else { n - 1 - k });
        let both = [stop(0), stop(1)];
        if legs.contains(&st) {
            assert!(both[0] != both[1], "{what}: one-way stop {k} stopped at by exactly one run, {both:?}");
            let r = if both[0] { 0 } else { 1 };
            let j = if r == 0 { k } else { n - 1 - k };
            assert_eq!(ls.runs[r].nodes[ls.runs[r].stop_idx[j] as usize], st, "{what}: at its own platform");
        } else {
            assert_eq!(both, [true, true], "{what}: stop {k} both ways");
        }
    }
    let (t0, t1) = (ls.runs[0].free.duration, ls.runs[1].free.duration);
    let (l0, l1) = (ls.runs[0].len, ls.runs[1].len);
    assert!((l0 - l1).abs() < 0.1 * l0.max(l1), "{what}: way out {l0:.0} m, {t0:.0} s, way back {l1:.0} m, {t1:.0} s");
    let ways = a.line_stop_ways(l);
    for &st in legs {
        assert!(ways.chunks(2).any(|c| c[0] as u32 == st), "{what}: one-way stop {st} drawn one way, {ways:?}");
        let deps = a.station_departures(l, st, 0.0, 2.0 * 86400.0);
        let runs: std::collections::BTreeSet<u32> = deps.chunks(3).map(|c| c[0] as u32).collect();
        assert!(runs.len() <= 1, "{what}: one-way stop {st} departs one way only, runs {runs:?}");
    }
    (t0, t1)
}

/// Every stop of line `l` stopped at by at least one of its runs.
fn assert_all_served(a: &TrackApi, l: u32, what: &str) {
    let ls = &a.w.svc.lines[l as usize];
    if ls.stops.len() < 2 {
        return;
    }
    assert!(ls.ok, "{what}: the service builds");
    let n = ls.stops.len();
    for k in 0..n {
        assert!(!(ls.runs[0].passes(k) && ls.runs[1].passes(n - 1 - k)), "{what}: stop {k} of {:?} served by neither run", ls.stops);
    }
}

/// A line as stored: its stops, path and way back.
type Made = (Vec<u32>, Vec<u32>, Option<Vec<u32>>);
fn made(a: &TrackApi, l: u32) -> Made {
    (a.w.net.line_stops(l).to_vec(), a.w.net.line_path(l).to_vec(), a.w.net.back_path(l))
}
/// The same line run the other way round: its stops reversed, its way back its path.
fn reversed(m: &Made) -> Made {
    (m.0.iter().rev().copied().collect(), m.2.clone().unwrap_or_default(), Some(m.1.clone()))
}

/// Her case (Anita 2026-11-17, Caracas Line 2): legs 500 m apart, a station on each, both on the
/// line, given or clicked in either order, on both running sides: each run stops at its own leg's
/// and passes the other's, as quick as a line between the ends; an unrelated edit and a stop taken
/// out and clicked in again leave it as it was.
#[test]
fn t3565_far_legs_each_served_one_way() {
    for right_hand in [true, false] {
        for order in 0..2 {
            let (mut a, [p, q, n, s]) = pair(right_hand, 250.0);
            let what = format!("right hand {right_hand}, order {order}");
            let mid = if order == 0 { [n, s] } else { [s, n] };
            let l = new_line(&mut a, &[p, mid[0], mid[1], q]);
            let (t0, t1) = assert_served_once(&a, l, &[n, s], &format!("{what}, set"));
            let o = new_line(&mut a, &[p, q]);
            let plain = &a.w.svc.lines[o as usize];
            assert!((plain.runs[0].free.duration - t0).abs() < 120.0 && (plain.runs[1].free.duration - t1).abs() < 120.0, "{what}: {t0:.0}/{t1:.0} against P-Q's {:.0}/{:.0}", plain.runs[0].free.duration, plain.runs[1].free.duration);
            let c = new_line(&mut a, &[p, q]);
            assert_eq!(a.add_stop(c, mid[0]), 0, "{what}: {}", a.issues());
            assert_eq!(a.add_stop(c, mid[1]), 0, "{what}: {}", a.issues());
            assert_eq!(a.w.net.line_stops(c).len(), 4, "{what}: both one-way stations are stops");
            assert_served_once(&a, c, &[n, s], &format!("{what}, clicked"));
            let before = made(&a, l);
            assert_eq!(a.add_route(&ends(free(0.0, 3000.0), free(1000.0, 3000.0)), &[], 2, false), 0, "{}", a.issues());
            assert_eq!(made(&a, l), before, "{what}: unrelated track");
            let i = a.w.net.line_stops(l).iter().position(|&x| x == mid[1]).unwrap();
            assert_eq!(a.drop_stop(l, i as u32), 0, "{what}: {}", a.issues());
            assert_eq!(a.add_stop(l, mid[1]), 0, "{what}: {}", a.issues());
            assert_eq!(made(&a, l), before, "{what}: a stop out and in again");
        }
    }
}

/// Anita (2026-11-17): a one-way station pairs with the other leg's only when their platforms are
/// within about 50 m. Paired (40 m apart), one of them alone is one stop served each way at its
/// own platform (T-3348); both are each served once. 70 m apart, one alone is served once.
#[test]
fn t3565_pair_only_side_by_side() {
    for right_hand in [true, false] {
        let (mut a, [p, q, n, s]) = pair(right_hand, 20.0);
        let l = new_line(&mut a, &[p, q]);
        assert_eq!(a.add_stop(l, n), 0, "{}", a.issues());
        assert_eq!(a.w.net.line_stops(l).len(), 3);
        let ls = &a.w.svc.lines[l as usize];
        let at = [ls.stop_node(0, 1).unwrap(), ls.stop_node(1, 1).unwrap()];
        assert!(at.contains(&n) && at.contains(&s) && ls.runs.iter().all(|r| !r.pass.contains(&true)), "right hand {right_hand}: paired, each way at its own, {at:?}");
        // the other way's station goes only with the line's stop, as a listed stop does (T-3348)
        let other = if a.w.net.line_stops(l)[1] == n { s } else { n };
        assert_ne!(a.remove_station(other), 0, "right hand {right_hand}: the way back's station is in use");
        let first = a.w.net.line_stops(l)[1];
        assert_eq!(a.add_stop(l, if first == n { s } else { n }), 0, "{}", a.issues());
        assert_eq!(a.w.net.line_stops(l).len(), 4);
        assert_served_once(&a, l, &[n, s], &format!("right hand {right_hand}, both of a pair"));

        let (mut a, [p, q, n, s]) = pair(right_hand, 35.0);
        for one in [n, s] {
            let l = new_line(&mut a, &[p, q]);
            assert_eq!(a.add_stop(l, one), 0, "{}", a.issues());
            assert_eq!(a.w.net.line_stops(l), &[p, one, q][..], "right hand {right_hand}: the station clicked");
            assert_served_once(&a, l, &[one], &format!("right hand {right_hand}, 70 m apart, one listed"));
            assert_eq!(a.remove_line(l), 0, "{}", a.issues());
        }
    }
}

/// Anita (2026-11-17): any number of stations on a leg, none on the other allowed. Three on the
/// north leg and one on the south, then three and none: every click order the same line.
#[test]
fn t3565_three_and_one_three_and_none() {
    for right_hand in [true, false] {
        let rh = format!("right hand {right_hand}");
        let (mut a, [ar, lo, e12, un], ns, ss) = pair_wide(right_hand, 250.0, &[-500.0, 0.0, 500.0], &[0.0]);
        let set: Vec<u32> = [ar, lo, e12, un].into_iter().chain(ns.iter().copied()).chain(ss.iter().copied()).collect();
        let l = same_line_every_order(&mut a, &set, 60, &format!("{rh}, three and one"));
        let legs: Vec<u32> = ns.iter().chain(&ss).copied().collect();
        assert_served_once(&a, l, &legs, &format!("{rh}, three and one"));

        let (mut a, [ar, lo, e12, un], ns, _) = pair_wide(right_hand, 250.0, &[-500.0, 0.0, 500.0], &[]);
        let set: Vec<u32> = [ar, lo, e12, un].into_iter().chain(ns.iter().copied()).collect();
        let l = same_line_every_order(&mut a, &set, 60, &format!("{rh}, three and none"));
        assert_served_once(&a, l, &ns, &format!("{rh}, three and none"));
    }
}

/// Orders of `set`: every permutation when there are at most 7, else `sample` drawn with a fixed
/// seed, plus the set as given and reversed.
fn orders(set: &[u32], sample: usize) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![];
    if set.len() <= 7 {
        // Heap's algorithm
        let mut a = set.to_vec();
        let n = a.len();
        let mut c = vec![0usize; n];
        out.push(a.clone());
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 { a.swap(0, i) } else { a.swap(c[i], i) }
                out.push(a.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        return out;
    }
    out.push(set.to_vec());
    out.push(set.iter().rev().copied().collect());
    let mut seed: u64 = 3565;
    for _ in 0..sample {
        let mut a = set.to_vec();
        for i in (1..a.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            a.swap(i, (seed % (i as u64 + 1)) as usize);
        }
        out.push(a);
    }
    out
}

/// A line made by clicking the stations of `set` one at a time onto a new line, in every order
/// `orders` gives: each click adds the station clicked, and every order makes the same line, up to
/// the way round (stops, path and way back, or the stops reversed with the two swapped); ordering
/// it again changes nothing. Returns the last line, left in place.
fn same_line_every_order(a: &mut TrackApi, set: &[u32], sample: usize, what: &str) -> u32 {
    let mut first: Option<Made> = None;
    let mut last = u32::MAX;
    for (k, order) in orders(set, sample).iter().enumerate() {
        if last != u32::MAX {
            assert_eq!(a.remove_line(last), 0, "{}", a.issues());
        }
        let l = new_line(a, &order[..1]);
        for &st in &order[1..] {
            assert_eq!(a.add_stop(l, st), 0, "{what}, order {k} {order:?}: {}", a.issues());
            assert!(a.w.net.line_stops(l).contains(&st), "{what}, order {k} {order:?}: the click adds the station clicked");
            assert_all_served(a, l, &format!("{what}, order {k} {order:?}, after {st}"));
        }
        let got = made(a, l);
        assert!(!a.w.net.line_broken[l as usize], "{what}, order {k} {order:?}: runs");
        let again = legs::ordered(&a.w.net, &got.0, &mut HopMemo::new());
        assert!(again.as_ref().map_or(true, |x| x.0 == got.0 && x.1 == got.1), "{what}, order {k}: ordered again {:?} from {:?}", again.map(|x| x.0), got.0);
        match &first {
            None => first = Some(got),
            Some(f) => assert!(&got == f || &reversed(&got) == f, "{what}: order {k} {order:?} makes {:?}, order 0 {:?} made {:?}", got.0, orders(set, sample)[0], f.0),
        }
        last = l;
    }
    last
}

/// Anita's acceptance (2026-11-17): for a fixed set of stations, every click order makes the same
/// line. One station on each leg; three on each (the checker's case); two on one leg and one on
/// the other; and a set whose one-way station has to be an end (no stop past the pair).
#[test]
fn t3565_click_order_never_matters() {
    for right_hand in [true, false] {
        let rh = format!("right hand {right_hand}");
        let (mut a, [ar, lo, e12, un], ns, ss) = pair_wide(right_hand, 250.0, &[0.0], &[0.0]);
        let l = same_line_every_order(&mut a, &[ar, lo, ns[0], ss[0], e12, un], 0, &format!("{rh}, one a leg"));
        assert_served_once(&a, l, &[ns[0], ss[0]], &format!("{rh}, one a leg"));
        same_line_every_order(&mut a, &[ar, lo, ns[0]], 0, &format!("{rh}, an end on a leg"));
        same_line_every_order(&mut a, &[ar, lo, ns[0], ss[0]], 0, &format!("{rh}, both legs past the end"));

        let (mut a, [ar, lo, e12, un], ns, ss) = pair_wide(right_hand, 1200.0, &[-400.0, 0.0, 400.0], &[-200.0, 200.0, 600.0]);
        let set: Vec<u32> = [ar, lo, e12, un].into_iter().chain(ns.iter().copied()).chain(ss.iter().copied()).collect();
        let l = same_line_every_order(&mut a, &set, 60, &format!("{rh}, three a leg"));
        let legs: Vec<u32> = ns.iter().chain(&ss).copied().collect();
        assert_served_once(&a, l, &legs, &format!("{rh}, three a leg"));

        let (mut a, [ar, lo, e12, un], ns, ss) = pair_wide(right_hand, 250.0, &[-500.0, 500.0], &[0.0]);
        let set: Vec<u32> = [ar, lo, e12, un, ns[0], ns[1], ss[0]].to_vec();
        let l = same_line_every_order(&mut a, &set, 80, &format!("{rh}, unequal legs"));
        assert_served_once(&a, l, &[ns[0], ns[1], ss[0]], &format!("{rh}, unequal legs"));
    }
}

/// Stops given with a one-way station first, as her save had them (Este 18, Arismendi, Loyola,
/// Calzadilla I, ...): the stop edit puts it between the ends, served once, nothing turning back.
#[test]
fn t3565_leg_end_straightened() {
    for right_hand in [true, false] {
        let (mut a, [p, q, n, s]) = pair(right_hand, 250.0);
        let l = new_line(&mut a, &[p, q]);
        let west = |a: &TrackApi, k: u32| a.w.net.leg_heading(k).unwrap().cos() < 0.0;
        let (w, e) = if west(&a, n) { (n, s) } else { (s, n) };
        assert_eq!(a.set_stops(l, &[w, p, e, q]), 0, "{}", a.issues());
        let stops = a.w.net.line_stops(l).to_vec();
        assert!(stops.first() == Some(&p) && stops.last() == Some(&q) || stops.first() == Some(&q) && stops.last() == Some(&p), "right hand {right_hand}: ends P and Q, {stops:?}");
        assert_served_once(&a, l, &[n, s], &format!("right hand {right_hand}"));
    }
}

/// Stops ordered by hand (T-2895) keep the player's order, one-way stations against the track
/// included, and still each served once with nothing turning back.
#[test]
fn t3565_hand_order_kept() {
    for right_hand in [true, false] {
        let (mut a, [ar, lo, e12, un], ns, _) = pair_wide(right_hand, 250.0, &[-500.0, 500.0], &[]);
        let l = new_line(&mut a, &[ar, lo, e12, un]);
        // the north leg's two against the way its trains run
        let along = a.w.net.leg_heading(ns[0]).unwrap().cos() > 0.0;
        let mid = if along { [ns[1], ns[0]] } else { ns.clone().try_into().unwrap() };
        let given = [ar, lo, mid[0], mid[1], e12, un];
        assert_eq!(a.order_stops(l, &given), 0, "{}", a.issues());
        assert!(a.line_hand(l));
        assert_eq!(a.w.net.line_stops(l), &given[..], "right hand {right_hand}: the player's order");
        assert!(!a.w.net.line_broken[l as usize], "right hand {right_hand}: runs");
        assert_all_served(&a, l, &format!("right hand {right_hand}, by hand"));
        assert_eq!(a.sort_stops(l), 0, "{}", a.issues());
        assert_ne!(a.w.net.line_stops(l), &given[..], "right hand {right_hand}: the game's own order again");
        assert_served_once(&a, l, &ns, &format!("right hand {right_hand}, sorted"));
    }
}

/// Taking stops off: the end next to a one-way station, and the line ended at one; neither is
/// refused, and what is left runs.
#[test]
fn t3565_drop_and_end_beside_a_leg() {
    for right_hand in [true, false] {
        let (mut a, [ar, lo, e12, un], ns, ss) = pair_wide(right_hand, 250.0, &[0.0], &[0.0]);
        let l = new_line(&mut a, &[ar, lo, ns[0], ss[0], e12, un]);
        assert_served_once(&a, l, &[ns[0], ss[0]], "before");
        let n = a.w.net.line_stops(l).len() as u32;
        assert_eq!(a.drop_stop(l, n - 1), 0, "right hand {right_hand}: {}", a.issues());
        assert_served_once(&a, l, &[ns[0], ss[0]], &format!("right hand {right_hand}, an end off"));
        let stops = a.w.net.line_stops(l).to_vec();
        let i = stops.iter().position(|&s| s == ns[0] || s == ss[0]).unwrap();
        assert_eq!(a.end_line_at(l, i as u32), 0, "right hand {right_hand}: {}", a.issues());
        assert!(!a.w.net.line_broken[l as usize], "right hand {right_hand}: ended at a one-way station, runs");
    }
}

/// A leg's way taken off and put back: the line holding a station on each leg comes back as it was.
#[test]
fn t3565_way_off_and_on() {
    for right_hand in [true, false] {
        let (mut a, [p, q, n, s]) = pair(right_hand, 250.0);
        let l = new_line(&mut a, &[p, n, s, q]);
        let before = made(&a, l);
        let net = &a.w.net;
        let e = net.node_ports[n as usize].ends()[0] >> 1;
        let w = net.edge_way(e);
        let legs: Vec<u32> = (0..net.edge_count() as u32).filter(|&x| net.edge_ok(x) && net.edge_tracks[x as usize] == 1 && net.edge_way(x) == w && net.node_ports[n as usize].ends().iter().any(|&c| c >> 1 == x)).collect();
        for &x in &legs {
            assert_eq!(a.set_edge_oneway(x, 0), 0, "{}", a.issues());
        }
        for &x in &legs {
            assert_eq!(a.set_edge_oneway(x, w), 0, "{}", a.issues());
        }
        assert!(!a.w.net.line_broken[l as usize], "right hand {right_hand}: runs");
        assert_eq!(made(&a, l), before, "right hand {right_hand}: as it was");
    }
}

/// Probe: a real save's rail lines (`T3565_NET`, from `python trainworld/checks/save_net.py <save>
/// <out>`), one line each: id, name, broken, stops, both runs' lengths and reversals, round trip.
///   cargo test --release oneleg_probe -- --ignored --nocapture
#[test]
#[ignore]
fn oneleg_probe() {
    let raw = std::fs::read(std::env::var("T3565_NET").expect("T3565_NET")).unwrap();
    let mut a = TrackApi::new(-66.9, 10.5, true);
    assert!(a.load(&super::save::convert_frame(&raw, FrameKind::Merc).unwrap()));
    let net = &a.w.net;
    let name = |n: u32| if net.node_platform[n as usize] > 0 { net.node_name[n as usize].clone() } else { format!("#{n}") };
    let verbose = std::env::var("T3565_PATHS").is_ok();
    for l in 0..net.line_count() as u32 {
        if !net.line_ok(l) || net.line_bus[l as usize].is_some() || net.line_ferry[l as usize].is_some() {
            continue;
        }
        let ls = &a.w.svc.lines[l as usize];
        let runs: Vec<String> = ls.runs.iter().map(|r| format!("{:.0} m {:.0} s rev {}", r.len, r.free.duration, net.path_reversals(&r.segs).len())).collect();
        println!(
            "PROBE line {l} {:?} broken {} hand {} stops {:?} runs {:?} round {:.0}",
            net.line_name[l as usize],
            net.line_broken[l as usize],
            net.line_hand[l as usize],
            net.line_stops(l).iter().map(|&s| name(s)).collect::<Vec<_>>(),
            runs,
            ls.round_trip[0]
        );
        if verbose {
            for (r, run) in ls.runs.iter().enumerate() {
                let nodes: Vec<String> = run.nodes.iter().map(|&n| name(n)).filter(|s| !s.starts_with('#')).collect();
                let away: Vec<String> = run.away.iter().map(|x| name(x.1)).collect();
                println!("  run {r} {:?} pass {:?} away {:?}", nodes, run.pass, away);
            }
        }
    }
}
