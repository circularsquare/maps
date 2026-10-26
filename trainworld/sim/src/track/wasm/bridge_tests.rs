use super::*;
use crate::track::geom::{height_at, pos_at};

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
/// A world frame (Mercator) network at the same place: plane metres are not ground metres.
fn merc() -> TrackApi {
    let mut a = api();
    a.w = TrackWorld::new(Network::new_in(FrameKind::Merc, -73.985, 40.758, true));
    a
}
fn base(a: &TrackApi) -> (f64, f64) {
    let p = a.from_lonlat(-73.985, 40.758);
    (p[0], p[1])
}
/// Water where x (from `base`) lies in any of the bands.
fn water(a: &mut TrackApi, bands: &[(f64, f64)]) {
    let (bx, _) = base(a);
    let bands = bands.to_vec();
    a.w.net.set_water_mask(Box::new(move |x: f64, _y: f64| bands.iter().any(|b| x > bx + b.0 && x < bx + b.1)));
    a.w.reload();
}
fn ends(a: &TrackApi, from: (f64, f64), to: (f64, f64)) -> [f64; 10] {
    let (bx, by) = base(a);
    [0.0, 0.0, bx + from.0, by + from.1, 0.0, 0.0, 0.0, bx + to.0, by + to.1, 0.0]
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn height(a: &TrackApi, e: u32, s: f64) -> f64 {
    height_at(a.w.net.edge_vert(e), s)
}
/// Every sample over water is on the bridge and none of a ramp is over water.
fn bridge_ok(a: &TrackApi, e: u32) {
    let net = &a.w.net;
    let (pieces, len) = (net.edge_pieces(e), net.edge_len[e as usize]);
    let (mut s, mut wet, mut ramp) = (0.0, 0, 0);
    while s <= len {
        let (x, y, _) = pos_at(pieces, s);
        let h = height(a, e, s);
        if net.water_mask.is_water(x, y) {
            wet += 1;
            assert!(h > 7.99, "edge {e} s {s}: {h} m over water");
        }
        if h > 0.01 && h < 7.99 {
            ramp += 1;
            assert!(!net.water_mask.is_water(x, y), "edge {e} s {s}: a ramp at {h} m over water");
        }
        s += 2.0;
    }
    assert!(wet > 0 && ramp > 0, "water {wet}, ramp {ramp}");
}
fn problems(a: &TrackApi) -> Vec<Issue> {
    let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).collect();
    a.w.net.validate(&alive(a), &nodes, &[])
}
fn first_edge_cost(a: &TrackApi) -> f64 {
    alive(a).iter().map(|&e| a.w.net.edge_cost[e as usize]).sum()
}

#[test]
fn a_river_gets_a_bridge_and_the_preview_agrees() {
    for world in [false, true] {
        let mut a = if world { merc() } else { api() };
        water(&mut a, &[(1000.0, 1500.0)]);
        let e = ends(&a, (0.0, 0.0), (3000.0, 0.0));
        let pv = a.preview_route(&e, &[], 2, false);
        assert_eq!(pv[0], 1.0, "world {world}: {}", a.issue_kinds());
        assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
        let edges = alive(&a);
        assert_eq!(edges.len(), 1, "one piece");
        bridge_ok(&a, edges[0]);
        assert!((pv[1] - first_edge_cost(&a)).abs() < 1e-6, "{} against {}", pv[1], first_edge_cost(&a));
        // the ends stay on the ground; the same segment without water costs less
        let len = a.w.net.edge_len[edges[0] as usize];
        assert_eq!((height(&a, edges[0], 0.0), height(&a, edges[0], len)), (0.0, 0.0));
        let mut dry = if world { merc() } else { api() };
        assert_eq!(dry.add_route(&e, &[], 2, false), 0);
        assert!(pv[1] > 1.5 * first_edge_cost(&dry), "bridge and water cost more: {} against {}", pv[1], first_edge_cost(&dry));
        // one undo takes it all back
        assert_eq!(a.undo(), 0);
        assert!(alive(&a).is_empty());
        assert_eq!(a.redo(), 0);
        assert_eq!(alive(&a).len(), 1);
    }
}

/// T-667, T-727: a route through clicked corners at ground level is bridged over water too, the
/// bridge cut into pieces with the corners, all high over the water and one undo.
#[test]
fn clicked_corners_over_a_river_get_a_bridge() {
    for world in [false, true] {
        let mut a = if world { merc() } else { api() };
        water(&mut a, &[(1000.0, 1500.0)]);
        let (bx, by) = base(&a);
        let e = ends(&a, (0.0, 0.0), (6000.0, 0.0));
        let pis = [bx + 1250.0, by + 150.0, 0.0, 0.0, bx + 3000.0, by - 300.0, 0.0, 0.0, bx + 4500.0, by + 200.0, 0.0, 0.0];
        let pv = a.preview_route(&e, &pis, 2, false);
        assert_eq!(pv[0], 1.0, "world {world}: {}", a.issue_kinds());
        assert_eq!(a.add_route(&e, &pis, 2, false), 0, "{}", a.issue_kinds());
        let edges = alive(&a);
        // a leg that ramps, or that a bridge's curve fills, is not cut
        assert!(edges.len() >= 2, "pieces {}", edges.len());
        let mut over = 0;
        for &ed in &edges {
            let len = a.w.net.edge_len[ed as usize];
            let mut s = 0.0;
            while s <= len {
                let (x, _, _) = pos_at(a.w.net.edge_pieces(ed), s);
                if x > bx + 1000.0 && x < bx + 1500.0 {
                    over += 1;
                    assert!(height(&a, ed, s) >= LEVEL_H - 1e-6, "world {world}: low over water at x {}", x - bx);
                }
                s += 10.0;
            }
        }
        assert!(over > 25, "{over} samples over the water");
        assert!((pv[1] - first_edge_cost(&a)).abs() < 1e-6, "{} against {}", pv[1], first_edge_cost(&a));
        assert_eq!(a.undo(), 0);
        assert!(alive(&a).is_empty());
    }
}

/// T-727: a real line traced into corners goes in as one route, double or quad.
#[test]
fn a_traced_line_goes_in_as_one_route() {
    for tracks in [2u8, 4] {
        let mut a = merc();
        let (bx, by) = base(&a);
        // a straight, a 400 m curve through 120 degrees, a straight, sampled every 5 m
        let mut xy: Vec<(f64, f64)> = (0..=300).map(|i| (bx + i as f64 * 5.0, by)).collect();
        let r = 400.0;
        for i in 1..=167 {
            let t = i as f64 * 5.0 / r;
            xy.push((bx + 1500.0 + r * t.sin(), by + r - r * t.cos()));
        }
        let (ex, ey) = *xy.last().unwrap();
        let h = 2.0 * PI / 3.0;
        for i in 1..=300 {
            xy.push((ex + i as f64 * 5.0 * h.cos(), ey + i as f64 * 5.0 * h.sin()));
        }
        let lonlat: Vec<f64> = xy.iter().flat_map(|&(x, y)| a.to_lonlat(x, y)).collect();
        let t = a.trace_route(&lonlat, 25.0, tracks, 0);
        assert!(!t.is_empty());
        assert!(t[0] <= 25.0, "worst {}", t[0]);
        let n = t[2] as usize;
        assert!(n >= 1 && n <= 8, "{n} corners");
        let (sx, sy) = xy[0];
        let (fx, fy) = *xy.last().unwrap();
        let e = [0.0, 0.0, sx, sy, 0.0, 0.0, 0.0, fx, fy, 0.0];
        assert_eq!(a.add_route(&e, &t[3..], tracks, false), 0, "{}", a.issue_kinds());
    }
}

#[test]
fn a_long_route_bridges_each_river_on_its_own() {
    // a gentle zigzag with a corner every km; then corners 10 km apart, a river mid-leg
    for (step, rivers) in [(1000.0, [(10_000.0, 10_200.0), (40_000.0, 40_300.0)]), (10_000.0, [(55_000.0, 55_300.0), (125_000.0, 125_200.0)])] {
        let mut a = merc();
        water(&mut a, &rivers);
        let (bx, by) = base(&a);
        let e = ends(&a, (0.0, 0.0), (step * 60.0, 0.0));
        let pis: Vec<f64> = (1..60).flat_map(|i| [bx + i as f64 * step, by + if i % 2 == 0 { 30.0 } else { -30.0 }, 0.0, 0.0]).collect();
        let t = std::time::Instant::now();
        let pv = a.preview_route(&e, &pis, 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
        assert!(t.elapsed().as_secs_f64() < 2.0, "preview took {:?}", t.elapsed());
        assert_eq!(a.add_route(&e, &pis, 2, false), 0, "{}", a.issue_kinds());
        let edges = alive(&a);
        // every sample over either river is up on a bridge
        let mut wet = [0, 0];
        for &e in &edges {
            let (pieces, len) = (a.w.net.edge_pieces(e), a.w.net.edge_len[e as usize]);
            let mut s = 0.0;
            while s <= len {
                let (x, y, _) = pos_at(pieces, s);
                if a.w.net.water_mask.is_water(x, y) {
                    wet[usize::from(x > bx + rivers[0].1 + 1000.0)] += 1;
                    assert!(height(&a, e, s) > 7.99, "edge {e} s {s} over water");
                }
                s += 5.0;
            }
        }
        assert!(wet[0] > 0 && wet[1] > 0, "both rivers crossed: {wet:?}");
    }
}

/// A zigzag with a corner every 600 m (plane), each auto radius held by half its legs.
fn zigzag(a: &TrackApi, n: usize) -> ([f64; 10], Vec<f64>) {
    let (bx, by) = base(a);
    let pis = (1..n).flat_map(|i| [bx + i as f64 * 600.0, by + if i % 2 == 0 { 60.0 } else { -60.0 }, 0.0, 0.0]).collect();
    (ends(a, (0.0, 0.0), (n as f64 * 600.0, 0.0)), pis)
}

#[test]
fn a_long_route_keeps_its_radii_beside_each_bridge() {
    // T-1187: alone, a river's window let the corner next to each of its ends take the whole end
    // leg, and the bridge set that wider radius, which did not fit in the route
    let mut a = merc();
    // legs 300 and 900 m (plane) in turn, so a corner's curve is held by its short leg; rivers at
    // four phases of the pattern, so windows start and end on short legs and on long ones
    water(&mut a, &[(15_000.0, 15_200.0), (31_000.0, 31_150.0), (47_300.0, 47_450.0), (63_900.0, 64_050.0)]);
    let (bx, by) = base(&a);
    let pis: Vec<f64> = (1..120).flat_map(|i| [bx + (i / 2) as f64 * 1200.0 + if i % 2 == 1 { 300.0 } else { 0.0 }, by + if i % 2 == 0 { 60.0 } else { -60.0 }, 0.0, 0.0]).collect();
    let e = ends(&a, (0.0, 0.0), (72_000.0, 0.0));
    let pv = a.preview_route(&e, &pis, 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
    assert_eq!(a.add_route(&e, &pis, 2, false), 0, "{}", a.issue_kinds());
    assert!(problems(&a).is_empty());
}

#[test]
fn a_long_route_from_a_river_bank_says_there_is_no_room() {
    // T-1187: a river 110 m from the first click leaves no room for the ramp up, even raising the
    // window whole; the player is told so rather than "cannot cross water"
    let mut a = merc();
    water(&mut a, &[(150.0, 350.0)]);
    let (e, pis) = zigzag(&a, 100);
    let pv = a.preview_route(&e, &pis, 2, false);
    assert_eq!(pv[0], 0.0);
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
}

#[test]
fn a_long_route_bridges_slivers_beside_a_river() {
    // T-1187: a strip of water 4.5 m wide just past the river, which the plain sampling (every
    // 12.5 m) can step over but the route's pieces each read at their own phase; at five phases
    for k in 0..5 {
        let mut a = merc();
        let at = 15_414.4 + 7.3 * k as f64;
        water(&mut a, &[(15_000.0, 15_200.0), (at, at + 6.0)]);
        let (e, pis) = zigzag(&a, 100);
        let ends: Vec<Pi> = vec![Pi::new(e[2], e[3], 0.0, 0), Pi::new(e[7], e[8], 0.0, 0)];
        let verts: Vec<Pi> = std::iter::once(ends[0]).chain(pis.chunks(4).map(|p| Pi::new(p[0], p[1], 0.0, 0))).chain(std::iter::once(ends[1])).collect();
        let scaled = a.w.net.frame().scaled();
        let b = bridge::bridge_long(&verts, scaled, a.w.net.water_mask.as_ref(), [true, true]).unwrap().unwrap();
        // bridged with ramps beside the water, not the window raised whole between its corners
        assert!(b.len() > verts.len(), "phase {k}: {} vertices, {} before", b.len(), verts.len());
        let g = geom::fit_in(&b, scaled);
        let mut s = 0.0;
        while s <= g.len {
            let (x, y, _) = pos_at(&g.pieces, s);
            if a.w.net.water_mask.is_water(x, y) {
                assert!(height_at(&g.vert, s) > LEVEL_H - 0.01, "phase {k}, s {s}: {} m over water", height_at(&g.vert, s));
            }
            s += 0.5;
        }
    }
}

#[test]
fn each_crossing_gets_its_own_bridge_and_close_ones_share() {
    let mut a = api();
    water(&mut a, &[(1000.0, 1100.0), (3000.0, 3200.0)]);
    let e = ends(&a, (0.0, 0.0), (5000.0, 0.0));
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let edges = alive(&a);
    bridge_ok(&a, edges[0]);
    assert_eq!(height(&a, edges[0], 2000.0), 0.0, "ground between the rivers");
    assert_eq!(height(&a, edges[0], 1050.0), 8.0);
    assert_eq!(height(&a, edges[0], 3100.0), 8.0);
    // two rivers 200 m apart have no room for a ramp down and up: one bridge
    let mut b = api();
    water(&mut b, &[(1000.0, 1100.0), (1300.0, 1400.0)]);
    let e = ends(&b, (0.0, 0.0), (3000.0, 0.0));
    assert_eq!(b.add_route(&e, &[], 2, false), 0, "{}", b.issue_kinds());
    let edge = alive(&b)[0];
    bridge_ok(&b, edge);
    assert_eq!(height(&b, edge, 1200.0), 8.0, "level between the two");
}

#[test]
fn no_room_on_land_is_refused_and_says_so() {
    let mut a = api();
    water(&mut a, &[(100.0, 300.0)]);
    // water 100 m from the start
    let e = ends(&a, (0.0, 0.0), (3000.0, 0.0));
    let pv = a.preview_route(&e, &[], 2, false);
    assert_eq!(pv[0], 0.0);
    assert!(pv.len() > 20, "the ghost still has its shape");
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
    assert_ne!(a.add_route(&e, &[], 2, false), 0);
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
    assert!(alive(&a).is_empty());
    // water 100 m before the end
    let mut b = api();
    water(&mut b, &[(2600.0, 2900.0)]);
    let e = ends(&b, (0.0, 0.0), (3000.0, 0.0));
    assert_ne!(b.add_route(&e, &[], 2, false), 0);
    assert_eq!(b.issue_kinds(), "BridgeNoRoom");
    // just enough: 200 m of land each side
    let mut c = api();
    water(&mut c, &[(300.0, 400.0)]);
    let e = ends(&c, (0.0, 0.0), (700.0, 0.0));
    assert_eq!(c.add_route(&e, &[], 2, false), 0, "{}", c.issue_kinds());
    bridge_ok(&c, alive(&c)[0]);
}

/// The level of the node at x from `base`.
fn level_at_x(a: &TrackApi, x: f64) -> i8 {
    let n = (0..a.w.net.node_count()).find(|&n| a.w.net.node_ok(n as u32) && (a.w.net.node_x[n] - base(a).0 - x).abs() < 1.0).unwrap();
    a.w.net.node_level[n]
}

/// T-1269: a new end over water stays up on the bridge, with the one ramp on the land side.
#[test]
fn an_end_over_water_stays_up() {
    for world in [false, true] {
        for tracks in [2u8, 4] {
            for (from, to) in [(0.0, 2700.0), (2700.0, 0.0)] {
                let mut a = if world { merc() } else { api() };
                water(&mut a, &[(2600.0, 2900.0)]);
                let e = ends(&a, (from, 0.0), (to, 0.0));
                let name = format!("world {world}, {tracks} tracks, {from} to {to}");
                let pv = a.preview_route(&e, &[], tracks, false);
                assert_eq!(pv[0], 1.0, "{name}: {}", a.issue_kinds());
                assert_eq!(a.add_route(&e, &[], tracks, false), 0, "{name}: {}", a.issue_kinds());
                assert!((pv[1] - first_edge_cost(&a)).abs() < 1e-6, "{name}: {} against {}", pv[1], first_edge_cost(&a));
                assert_eq!((level_at_x(&a, 0.0), level_at_x(&a, 2700.0)), (0, 1), "{name}");
                for e in alive(&a) {
                    bridge_ok(&a, e);
                }
                assert!(problems(&a).is_empty(), "{name}: {:?}", problems(&a));
            }
        }
    }
    // both ends over the water: all of it up, no ramp
    let mut a = api();
    water(&mut a, &[(1000.0, 1500.0)]);
    assert_eq!(a.add_route(&ends(&a, (1100.0, 0.0), (1400.0, 0.0)), &[], 2, false), 0, "{}", a.issue_kinds());
    let edge = alive(&a)[0];
    assert!(a.w.net.edge_vert(edge).iter().all(|p| p[1] == LEVEL_H));
    // a long route (the real-network snap's) ending in a lake
    let mut l = merc();
    water(&mut l, &[(59_500.0, 61_000.0)]);
    let (e, pis) = zigzag(&l, 100);
    assert_eq!(l.add_route(&e, &pis, 2, false), 0, "{}", l.issue_kinds());
    assert_eq!(level_at_x(&l, 60_000.0), 1);
    assert!(problems(&l).is_empty(), "{:?}", problems(&l));
}

/// T-1269: drawing on at ground level from an end left up over the water stays up across it, and
/// drawing back into it from the far bank ramps up on the land side only; from an end up on land
/// with room before the water, the track comes down and goes up again.
#[test]
fn drawing_on_from_an_end_left_up() {
    for world in [false, true] {
        for back in [false, true] {
            let name = if back { "back" } else { "on" };
            let mut b = if world { merc() } else { api() };
            water(&mut b, &[(2600.0, 2900.0)]);
            assert_eq!(b.add_route(&ends(&b, (0.0, 0.0), (2700.0, 0.0)), &[], 2, false), 0, "{}", b.issue_kinds());
            let mid = (0..b.w.net.node_count() as u32).find(|&n| b.w.net.node_ok(n) && (b.w.net.node_x[n as usize] - base(&b).0 - 2700.0).abs() < 1.0).unwrap();
            let (bx, by) = base(&b);
            let on = if back { [0.0, 0.0, bx + 3500.0, by, 0.0, 1.0, mid as f64, 0.0, 0.0, 0.0] } else { [1.0, mid as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 3500.0, by, 0.0] };
            let pv = b.preview_route(&on, &[], 2, false);
            assert_eq!(pv[0], 1.0, "world {world} {name}: {}", b.issue_kinds());
            assert_eq!(b.add_route(&on, &[], 2, false), 0, "world {world} {name}: {}", b.issue_kinds());
            assert_eq!(level_at_x(&b, 3500.0), 0);
            for e in alive(&b) {
                bridge_ok(&b, e);
            }
            assert!(problems(&b).is_empty(), "world {world} {name}: {:?}", problems(&b));
        }
    }
    // an end up on land, 1.5 km short of the water
    let mut a = api();
    let mut e = ends(&a, (-1000.0, 0.0), (0.0, 0.0));
    (e[4], e[9]) = (1.0, 1.0);
    assert_eq!(a.add_route(&e, &[], 2, false), 0);
    water(&mut a, &[(1500.0, 1800.0)]);
    let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0).abs() < 1.0).unwrap();
    let (bx, by) = base(&a);
    assert_eq!(a.add_route(&[1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 3000.0, by, 0.0], &[], 2, false), 0, "{}", a.issue_kinds());
    let new = *alive(&a).iter().max_by(|&&p, &&q| a.w.net.edge_len[p as usize].total_cmp(&a.w.net.edge_len[q as usize])).unwrap();
    assert_eq!((height(&a, new, 0.0), height(&a, new, 700.0), height(&a, new, 1650.0)), (8.0, 0.0, 8.0));
    assert!(problems(&a).is_empty(), "{:?}", problems(&a));
}

/// T-1269: track already there keeps its level, so water at it still leaves no room.
#[test]
fn an_end_on_track_over_water_is_still_refused() {
    let mut a = api();
    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (1000.0, 0.0)), &[], 2, false), 0);
    // a newer water mask puts the track's end in a river
    water(&mut a, &[(900.0, 1500.0)]);
    let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 1000.0).abs() < 1.0).unwrap();
    let (bx, by) = base(&a);
    let e = [1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 3000.0, by, 0.0];
    assert_eq!(a.preview_route(&e, &[], 2, false)[0], 0.0);
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
    assert_ne!(a.add_route(&e, &[], 2, false), 0);
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
}

#[test]
fn water_near_a_station_leaves_no_room() {
    let mut a = api();
    water(&mut a, &[(2100.0, 2300.0)]);
    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (2000.0, 0.0)), &[], 2, false), 0);
    let end = a.w.net.node_count() as u32;
    let end = (0..end).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 2000.0).abs() < 1.0).unwrap();
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 100, "River".into()), 0, "{}", a.issue_kinds());
    let e = [1.0, end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, base(&a).0 + 4000.0, base(&a).1, 0.0];
    assert_eq!(a.preview_route(&e, &[], 2, false)[0], 0.0);
    assert_eq!(a.issue_kinds(), "BridgeNoRoom");
    assert_ne!(a.add_route(&e, &[], 2, false), 0);
    // water further out leaves the room, and the platform stays off the ramp
    water(&mut a, &[(2400.0, 2600.0)]);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    assert!(problems(&a).is_empty(), "{:?}", problems(&a));
}

#[test]
fn other_levels_are_left_alone() {
    // a viaduct is drawn as it is, a tunnel too, and neither gets ramps
    for level in [1.0, -1.0] {
        let mut a = api();
        water(&mut a, &[(1000.0, 1500.0)]);
        let mut e = ends(&a, (0.0, 0.0), (3000.0, 0.0));
        (e[4], e[9]) = (level, level);
        assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
        let edges = alive(&a);
        assert_eq!(edges.len(), 1);
        assert_eq!(height(&a, edges[0], 1250.0), level * 8.0);
        assert!(a.w.net.edge_vert(edges[0]).iter().all(|p| p[1] == level * 8.0), "flat at the level drawn");
    }
}

#[test]
fn a_curve_keeps_its_shape_with_a_bridge_on_it() {
    for world in [false, true] {
        // track going east, then a tangent arc turning north-east across the river
        let draw = |water_on: bool| {
            let mut a = if world { merc() } else { api() };
            if water_on {
                water(&mut a, &[(1300.0, 1450.0)]);
            }
            assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (600.0, 0.0)), &[], 2, false), 0);
            let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 600.0).abs() < 1.0).unwrap();
            a.set_route_tangent(true);
            let (bx, by) = base(&a);
            let e = [1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 2100.0, by + 1000.0, 0.0];
            let pv = a.preview_route(&e, &[], 2, false);
            assert_eq!(pv[0], 1.0, "world {world} water {water_on}: {}", a.issue_kinds());
            assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
            (a, pv)
        };
        let (dry, _) = draw(false);
        let (wet, pv) = draw(true);
        let edge = |a: &TrackApi| *alive(a).iter().max_by(|&&p, &&q| a.w.net.edge_len[p as usize].total_cmp(&a.w.net.edge_len[q as usize])).unwrap();
        let (d, w) = (edge(&dry), edge(&wet));
        bridge_ok(&wet, w);
        let (len_d, len_w) = (dry.w.net.edge_len[d as usize], wet.w.net.edge_len[w as usize]);
        assert!((len_d - len_w).abs() < 0.05, "same length: {len_d} {len_w}");
        let mut s = 0.0;
        while s < len_d {
            let (p, q) = (pos_at(dry.w.net.edge_pieces(d), s), pos_at(wet.w.net.edge_pieces(w), s));
            assert!((p.0 - q.0).hypot(p.1 - q.1) < 0.3, "world {world}: shape differs at {s}");
            s += 20.0;
        }
        // the ghost showed the same thing: the same length, and a price above the dry one
        assert!((pv[2] - len_w).abs() < 0.05);
        assert!(pv[1] > first_edge_cost(&dry) * 0.99);
        assert!(problems(&wet).is_empty());
    }
}

/// T-817: a wide curve drawn at 320 km/h bridges a river like any other.
#[test]
fn a_design_speed_curve_bridges_a_river() {
    for world in [false, true] {
        for (tx, ty, from) in [(4000.0, 1000.0, 2000.0), (6000.0, 2500.0, 3000.0), (3000.0, 1500.0, 1500.0)] {
            for width in [40.0, 120.0] {
                let mut a = if world { merc() } else { api() };
                water(&mut a, &[(from, from + width)]);
                assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (600.0, 0.0)), &[], 2, false), 0);
                let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 600.0).abs() < 1.0).unwrap();
                a.set_route_tangent(true);
                a.set_design_speed(320.0);
                let (bx, by) = base(&a);
                let e = [1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + tx, by + ty, 0.0];
                let r = a.add_route(&e, &[], 2, false);
                assert_eq!(r, 0, "world {world} to ({tx}, {ty}), water {from} + {width}: {}", a.issue_kinds());
                let w = *alive(&a).iter().max_by(|&&p, &&q| a.w.net.edge_len[p as usize].total_cmp(&a.w.net.edge_len[q as usize])).unwrap();
                assert!(!a.w.net.edge_water(w).is_empty(), "the curve crosses the water");
                bridge_ok(&a, w);
            }
        }
    }
}

/// Curves in both frames against many rivers: a bridge keeps the shape and the ramps stay on
/// land, or the segment is refused for want of room on land and nothing else.
#[test]
fn bridges_on_curves_hold_the_shape_wherever_the_water_is() {
    let targets = [(2100.0, 1000.0), (1500.0, -1200.0), (2600.0, 300.0), (1100.0, 2400.0), (-300.0, 1800.0)];
    let (mut built, mut refused) = (0, vec![]);
    for world in [false, true] {
        for &(tx, ty) in &targets {
            let dry = {
                let mut a = if world { merc() } else { api() };
                assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (600.0, 0.0)), &[], 2, false), 0);
                let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 600.0).abs() < 1.0).unwrap();
                a.set_route_tangent(true);
                let (bx, by) = base(&a);
                assert_eq!(a.add_route(&[1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + tx, by + ty, 0.0], &[], 2, false), 0, "{}", a.issue_kinds());
                a
            };
            let longest = |a: &TrackApi| *alive(a).iter().max_by(|&&p, &&q| a.w.net.edge_len[p as usize].total_cmp(&a.w.net.edge_len[q as usize])).unwrap();
            let d = longest(&dry);
            for from in (700..2400).step_by(130) {
                for width in [60.0, 200.0, 420.0] {
                    let mut a = if world { merc() } else { api() };
                    water(&mut a, &[(from as f64, from as f64 + width)]);
                    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (600.0, 0.0)), &[], 2, false), 0);
                    let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 600.0).abs() < 1.0).unwrap();
                    a.set_route_tangent(true);
                    let (bx, by) = base(&a);
                    let e = [1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + tx, by + ty, 0.0];
                    let pv = a.preview_route(&e, &[], 2, false);
                    let r = a.add_route(&e, &[], 2, false);
                    assert_eq!(pv[0] == 1.0, r == 0, "preview and edit agree");
                    let name = format!("world {world} to ({tx}, {ty}), water {from} + {width}");
                    if r != 0 {
                        assert_eq!(a.issue_kinds(), "BridgeNoRoom", "{name}");
                        refused.push(name);
                        continue;
                    }
                    let w = longest(&a);
                    if a.w.net.edge_water(w).is_empty() {
                        continue; // the curve never reaches this water
                    }
                    built += 1;
                    bridge_ok(&a, w);
                    assert!(problems(&a).is_empty(), "{name}: {:?}", problems(&a));
                    let len = dry.w.net.edge_len[d as usize];
                    assert!((len - a.w.net.edge_len[w as usize]).abs() < 0.05, "{name}: length");
                    let mut s = 0.0;
                    while s < len {
                        let (p, q) = (pos_at(dry.w.net.edge_pieces(d), s), pos_at(a.w.net.edge_pieces(w), s));
                        assert!((p.0 - q.0).hypot(p.1 - q.1) < 0.3, "{name}: shape differs at {s}");
                        s += 20.0;
                    }
                }
            }
        }
    }
    // a curve cut short of room for its ramps (a short arc, a tight turn) is refused; most are built
    assert!(built > 120 && built > refused.len(), "built {built}, refused {}", refused.len());
}

#[test]
fn a_bridge_off_constructed_track_and_quad_track() {
    let mut a = api();
    water(&mut a, &[(1200.0, 1500.0)]);
    assert_eq!(a.add_route(&ends(&a, (0.0, 0.0), (800.0, 0.0)), &[], 2, false), 0);
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    let built = alive(&a)[0];
    assert!(a.w.net.edge_built[built as usize]);
    let node = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - base(&a).0 - 800.0).abs() < 1.0).unwrap();
    let (bx, by) = base(&a);
    let e = [1.0, node as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx + 3000.0, by, 0.0];
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let new = *alive(&a).iter().find(|&&x| x != built).unwrap();
    assert!(!a.w.net.edge_built[new as usize], "the bridge is blueprint");
    assert!(a.w.net.edge_built[built as usize]);
    bridge_ok(&a, new);
    // quad track: both pairs cross on a bridge
    let mut q = api();
    water(&mut q, &[(1000.0, 1300.0)]);
    let e = ends(&q, (0.0, 0.0), (3000.0, 0.0));
    let pv = q.preview_route(&e, &[], 4, false);
    assert_eq!(pv[0], 1.0, "{}", q.issue_kinds());
    assert_eq!(q.add_route(&e, &[], 4, false), 0, "{}", q.issue_kinds());
    let edges = alive(&q);
    assert!(edges.len() >= 2);
    for &x in &edges {
        if q.w.net.edge_water(x).is_empty() {
            continue;
        }
        bridge_ok(&q, x);
    }
    assert!(problems(&q).is_empty());
}
