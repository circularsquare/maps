//! T-3137: lanes inside a piece and the move table at nodes.

use super::*;
use crate::track::capacity::junction_crosses;
use crate::track::lanes::{ends_in, WAY_AB, WAY_BA};
use crate::track::net::{path_dir, path_edge};

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
fn edge_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, y).1 < 1.0).unwrap()
}
fn junctions(a: &TrackApi) -> Vec<u32> {
    let n = &a.w.net;
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n >= 3).collect()
}

/// Every single-track piece's joins cleared, as a save from before joins has them (T-3137 C2).
fn unjoin(a: &mut TrackApi) {
    let singles: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e) && a.w.net.edge_tracks[e as usize] == 1).collect();
    for e in singles {
        let d = a.w.net.edge_data(e).unwrap();
        a.w.apply(Op::Edge { id: e, data: Some(EdgeData { join: [0; 2], ..d }) }).unwrap();
    }
}

/// A double main with a double Y to one side and a single branch to the other, the single branch
/// one way if `oneway` is set; no joins (C1's tests: the move table before joins).
fn network(right_hand: bool, oneway: u8) -> TrackApi {
    let mut a = api(right_hand);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0);
    let e = edge_at(&a, 1500.0, 0.0);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1500.0, 0.0, 0.0], free(3500.0, -1500.0)), &[], 2, false), 0);
    let e = edge_at(&a, 3500.0, 0.0);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 3500.0, 0.0, 0.0], free(5000.0, 1500.0)), &[], 1, false), 0);
    unjoin(&mut a);
    if oneway != 0 {
        let s = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.edge_tracks[e as usize] == 1).unwrap();
        assert_eq!(a.set_edge_oneway(s, oneway), 0, "{}", a.issues());
    }
    a
}

#[test]
fn t3137_lanes_by_running_side() {
    for right_hand in [true, false] {
        let a = network(right_hand, 0);
        let net = &a.w.net;
        for e in (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e)) {
            let lanes = net.lanes(e);
            if net.edge_tracks[e as usize] == 1 {
                assert_eq!(lanes.all().len(), 1);
                assert_eq!((lanes.all()[0].off, lanes.all()[0].ways), (0.0, WAY_AB | WAY_BA));
                continue;
            }
            let [r, l] = [lanes.all()[0], lanes.all()[1]];
            assert_eq!((r.off, l.off), (-2.0, 2.0));
            // keep right: the right lane looking from a to b carries a to b
            assert_eq!((r.ways, l.ways), if right_hand { (WAY_AB, WAY_BA) } else { (WAY_BA, WAY_AB) });
            assert_eq!(net.lane_of(e << 1).unwrap(), if right_hand { r } else { l });
        }
    }
    for (oneway, ways) in [(1, WAY_AB), (2, WAY_BA)] {
        let a = network(true, oneway);
        let s = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.edge_tracks[e as usize] == 1).unwrap();
        assert_eq!(a.w.net.lanes(s).all()[0].ways, ways);
        assert_eq!(a.w.net.dir_ok(s << 1), oneway == 1);
        assert_eq!(a.w.net.dir_ok(s << 1 | 1), oneway == 2);
    }
}

/// The rank arithmetic the move table replaced: ports right to left, a double port's two tracks by
/// running side, a single port in between, in units of one track.
fn old_track_pos(net: &Network, n: u32, k: usize, along: bool) -> f64 {
    let p = &net.node_ports[n as usize];
    let rank = p.rank[k] as f64 * 2.0;
    if net.edge_tracks[(p.end[k] >> 1) as usize] == 1 {
        rank + 0.5
    } else if along == net.right_hand {
        rank
    } else {
        rank + 1.0
    }
}

/// Every lane position is the old track position in metres (4 x - 2 for 8 m slots and lanes 2 m
/// off the centre), so every crossing test comes out as before; and every move the old rule
/// allowed (opposite sides, both ways allowed) is in the table, and no other.
#[test]
fn t3137_table_matches_the_rank_rule() {
    for right_hand in [true, false] {
        for oneway in [0, 1, 2] {
            let a = network(right_hand, oneway);
            let net = &a.w.net;
            assert_eq!(junctions(&a).len(), 2);
            for n in (0..net.node_count() as u32).filter(|&n| net.node_ok(n)) {
                let p = &net.node_ports[n as usize];
                let m = &net.node_moves[n as usize];
                for i in 0..p.n as usize {
                    // arriving through an end on side 0, or leaving through one on side 1, runs along the heading
                    let (arr_along, leave_along) = (p.side[i] == 0, p.side[i] == 1);
                    assert_eq!(m.arrive_at[i] as f64, 4.0 * old_track_pos(net, n, i, arr_along) - 2.0);
                    assert_eq!(m.leave_at[i] as f64, 4.0 * old_track_pos(net, n, i, leave_along) - 2.0);
                    for j in 0..p.n as usize {
                        let old = p.side[i] != p.side[j] && net.dir_ok(p.end[i] ^ 1) && net.dir_ok(p.end[j]);
                        assert_eq!(m.to[i] >> j & 1 != 0, old, "right hand {right_hand}, one way {oneway}, node {n}, {i} to {j}");
                        assert_eq!(m.link[i] >> j & 1 != 0, p.side[i] != p.side[j]);
                    }
                }
            }
        }
    }
}

/// Routing, line checks and the badge read the table: a move taken out of it is no longer routed,
/// a line through it is broken, and a junction whose only crossing move is gone loses its badge.
#[test]
fn t3137_routing_reads_the_table() {
    let mut a = network(true, 0);
    let y = *junctions(&a).iter().find(|&&j| (a.w.net.node_x[j as usize] - 1500.0).abs() < 1.0).unwrap();
    let ends_ = |a: &TrackApi| -> Vec<u32> { (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n) && a.w.net.node_ports[n as usize].n == 1).collect() };
    for n in ends_(&a) {
        let (x, yy) = (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
        assert_eq!(a.add_station(1, n, x, yy, 200, "S".into()), 0, "{}", a.issues());
    }
    let near = |a: &TrackApi, x: f64, y: f64| *ends_(a).iter().min_by(|&&p, &&q| {
        let d = |n: u32| (a.w.net.node_x[n as usize] - x).hypot(a.w.net.node_y[n as usize] - y);
        d(p).total_cmp(&d(q))
    }).unwrap();
    let (w, b) = (near(&a, 0.0, 0.0), near(&a, 3500.0, -1500.0));
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[w, b], 4, 2, 0, 30.0, 180.0, 0, 0, "Y".into()), 0, "{}", a.issues());
    let path = a.w.net.line_path(l).to_vec();
    let at = path.iter().position(|&p| a.w.net.path_nodes(p).1 == y).unwrap();
    let (p, q) = (path[at], path[at + 1]);
    assert!(a.w.net.move_ok(p, q));
    assert!(junction_crosses(&a.w.net, y));
    // the branch move out of the table, both ways: no route, the line broken, no crossing left
    let ports = a.w.net.node_ports[y as usize];
    let (i, j) = (ports.find(path_edge(p), 1 - path_dir(p)).unwrap(), ports.find(path_edge(q), path_dir(q)).unwrap());
    let m = &mut a.w.net.node_moves[y as usize];
    m.to[i] &= !(1 << j);
    m.to[j] &= !(1 << i);
    assert!(!a.w.net.move_ok(p, q));
    assert_eq!(a.w.net.route(&[w, b]), None);
    assert!(a.w.net.check_line(l).is_err());
    let left: Vec<usize> = (0..ports.n as usize).flat_map(|i| ends_in(a.w.net.node_moves[y as usize].to[i])).collect();
    assert_eq!(left.len(), 2, "only the main's two moves are left");
    assert!(!junction_crosses(&a.w.net, y));
}

// ---- C2: single track joined to one track of double track, and the way it runs

fn on_edge(a: &TrackApi, x: f64, y: f64) -> [f64; 5] {
    [2.0, edge_at(a, x, y) as f64, x, y, 0.0]
}
/// An end on the track nearest (x, y), at the point of it nearest.
fn near_track(a: &TrackApi, x: f64, y: f64) -> [f64; 5] {
    let n = &a.w.net;
    let e = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).min_by(|&p, &q| n.project(p, x, y).1.total_cmp(&n.project(q, x, y).1)).unwrap();
    let (px, py, _) = crate::track::geom::pos_at(n.edge_pieces(e), n.project(e, x, y).0);
    [2.0, e as f64, px, py, 0.0]
}
fn node_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k)).min_by(|&p, &q| d(p).total_cmp(&d(q))).unwrap()
}
fn at_node(a: &TrackApi, n: u32) -> [f64; 5] {
    [1.0, n as f64, a.w.net.node_x[n as usize], a.w.net.node_y[n as usize], 0.0]
}
fn singles(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e) && a.w.net.edge_tracks[e as usize] == 1).collect()
}
fn bare_ends(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n) && a.w.net.node_ports[n as usize].n == 1).collect()
}
fn stations_at_ends(a: &mut TrackApi) {
    for n in bare_ends(a) {
        let (x, y) = (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
        assert_eq!(a.add_station(1, n, x, y, 200, "S".into()), 0, "{}", a.issues());
    }
}

/// A flyover laid by hand: a double main west to east, a double branch from J3 north-east, and two
/// single legs from one point J on the main, one off each side: the north one to the branch's end,
/// the south one round over the main to it too. Drawn with the tool's own joins, each runs one
/// way, the way trains come into it, and no move crosses at J or J3.
#[test]
fn t3137_hand_made_flyover_runs_one_way_with_no_badge() {
    let mut a = api(true);
    a.set_design_speed(80.0);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(8000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(free(3500.0, 1500.0), free(6000.0, 3500.0)), &[], 2, false), 0, "{}", a.issues());
    let j3 = node_near(&a, 3500.0, 1500.0);
    assert_eq!(a.add_route(&ends(on_edge(&a, 1000.0, 0.0), at_node(&a, j3)), &[], 1, false), 0, "{}", a.issues());
    assert_eq!(a.route_joins()[..2], [1, 0], "the north track, looking out along it; none at the plain joint");
    let north: Vec<u32> = singles(&a);
    let j = node_near(&a, 1000.0, 0.0);
    a.set_route_ramp(true);
    let r = a.add_route(&ends(at_node(&a, j), at_node(&a, j3)), &[2000.0, -300.0, 0.0, 0.0, 2800.0, 300.0, 0.0, 0.0], 1, false);
    a.set_route_ramp(false);
    assert_eq!(r, 0, "{}", a.issues());
    assert_eq!(a.route_joins()[..2], [2, 1], "the south track at J; the left of the two singles at J3");
    let net = &a.w.net;
    for e in singles(&a) {
        // the north leg was drawn from J, so runs b to a, into the main; the south one a to b
        let want = if north.contains(&e) { 2 } else { 1 };
        assert_eq!(net.edge_way(e), want, "piece {e}");
    }
    for n in [j, j3] {
        assert!(!junction_crosses(net, n), "no move crosses at {n}");
    }
    // a line from the main's west end to the branch goes out by the south leg and back by the north
    stations_at_ends(&mut a);
    let (w, b) = (node_near(&a, 0.0, 0.0), node_near(&a, 6000.0, 3500.0));
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[w, b], 4, 2, 0, 30.0, 180.0, 0, 0, "F".into()), 0, "{}", a.issues());
    let net = &a.w.net;
    assert!(!net.line_broken[l as usize]);
    let back = net.back_path(l).unwrap();
    assert!(net.line_path(l).iter().chain(&back).all(|&p| net.dir_ok(p)));
    assert!(back.iter().any(|&p| north.contains(&path_edge(p))) && !net.line_path(l).iter().any(|&p| north.contains(&path_edge(p))));
}

/// A single-track branch to a station (a terminus) runs both ways though joined to one track, and
/// its trains back cross the other track: the badge stays. A stub to open ground runs one way.
#[test]
fn t3137_terminus_branch_stays_two_way_with_its_badge() {
    for right_hand in [true, false] {
        let mut a = api(right_hand);
        assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0);
        assert_eq!(a.add_route(&ends(on_edge(&a, 2000.0, 0.0), free(3500.0, 1500.0)), &[], 1, false), 0, "{}", a.issues());
        let j = junctions(&a)[0];
        let stub = singles(&a);
        // north of an eastward main: with right-hand running its track runs west, into the main
        let want = if right_hand { 2 } else { 1 };
        assert!(stub.iter().all(|&e| a.w.net.edge_way(e) == want));
        assert!(!junction_crosses(&a.w.net, j));
        stations_at_ends(&mut a);
        assert!(singles(&a).iter().all(|&e| a.w.net.edge_way(e) == 0));
        assert!(singles(&a).iter().any(|&e| a.w.net.edge_join[e as usize] != [0; 2]), "the join is kept");
        assert!(junction_crosses(&a.w.net, j));
        assert_eq!(a.w.net.edge_oneway[stub[0] as usize], 0, "nothing stored but the join");
    }
}

/// A single-track link whose joins send trains out of it at both ends runs both ways (the setup
/// works badly, as Anita expects), its junctions flat. The player's way wins over the joins.
#[test]
fn t3137_inconsistent_joins_run_both_ways() {
    let mut a = api(true);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0);
    assert_eq!(a.add_route(&ends(free(0.0, 3000.0), free(6000.0, 3000.0)), &[], 2, false), 0);
    assert_eq!(a.add_route(&ends(on_edge(&a, 1500.0, 0.0), on_edge(&a, 4500.0, 3000.0)), &[], 1, false), 0, "{}", a.issues());
    let joins = a.route_joins();
    assert!(joins[0] != 0 && joins[1] != 0, "{joins:?}");
    assert!(singles(&a).iter().all(|&e| a.w.net.edge_way(e) == 0));
    for j in junctions(&a) {
        assert!(junction_crosses(&a.w.net, j));
    }
    let s = singles(&a)[0];
    assert_eq!(a.set_edge_oneway(s, 1), 0, "{}", a.issues());
    assert_eq!(a.edge_way(s), 1);
    assert_eq!(a.set_edge_oneway(s, 3), 0, "{}", a.issues());
    assert_eq!((a.edge_oneway(s), a.edge_way(s)), (3, 0));
}

/// An edit that changes a run's way moves a line that ran against it onto another way through its
/// same stops, in the same edit.
#[test]
fn t3137_a_way_change_reroutes_a_line() {
    let mut a = api(true);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(8000.0, 0.0)), &[], 2, false), 0);
    // L: single from the main north to N, then double on east from N (a plain joint: both ways)
    assert_eq!(a.add_route(&ends(on_edge(&a, 1000.0, 0.0), free(2000.0, 1500.0)), &[], 1, false), 0, "{}", a.issues());
    let n = node_near(&a, 2000.0, 1500.0);
    assert_eq!(a.add_route(&ends(at_node(&a, n), free(7000.0, 1500.0)), &[], 2, false), 0, "{}", a.issues());
    let leg = singles(&a);
    assert!(leg.iter().all(|&e| a.w.net.edge_way(e) == 0));
    stations_at_ends(&mut a);
    let (w, e) = (node_near(&a, 0.0, 0.0), node_near(&a, 7000.0, 1500.0));
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[w, e], 4, 2, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let was = a.w.net.line_path(l).to_vec();
    assert!(was.iter().any(|&p| leg.contains(&path_edge(p))), "the line runs over L");
    // K: a double link further east, another way between the stations
    assert_eq!(a.add_route(&ends(on_edge(&a, 4000.0, 0.0), near_track(&a, 6000.0, 1500.0)), &[], 2, false), 0, "{}", a.issues());
    let was = a.w.net.line_path(l).to_vec();
    assert!(was.iter().any(|&p| leg.contains(&path_edge(p))), "no way changed: the line stays on L");
    // a stub from N on L's side makes N a junction: L joins the track beside it there, which runs
    // into L, so L runs one way, from N to the main, against the line
    assert_eq!(a.add_route(&ends(at_node(&a, n), free(2500.0, 300.0)), &[], 1, false), 0, "{}", a.issues());
    let net = &a.w.net;
    assert!(leg.iter().all(|&e| net.edge_way(e) == 2), "{:?}", leg.iter().map(|&e| net.edge_way(e)).collect::<Vec<_>>());
    assert!(!net.line_broken[l as usize]);
    let now = net.line_path(l).to_vec();
    assert!(now.iter().all(|&p| !leg.contains(&path_edge(p))), "rerouted by K: {now:?}");
    // one edit: undo puts the line back
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_path(l), &was[..]);
    assert!(leg.iter().all(|&e| a.w.net.edge_way(e) == 0));
}

/// Joins and a way set to 3 survive a save; a save with none reads every way as both.
#[test]
fn t3137_save_keeps_joins() {
    let mut a = api(true);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0);
    assert_eq!(a.add_route(&ends(on_edge(&a, 2000.0, 0.0), free(3500.0, 1500.0)), &[], 1, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(on_edge(&a, 4000.0, 0.0), free(5500.0, -1500.0)), &[], 1, false), 0, "{}", a.issues());
    let s = *singles(&a).iter().find(|&&e| a.w.net.project(e, 5500.0, -1500.0).1 < 1.0).unwrap();
    assert_eq!(a.set_edge_oneway(s, 3), 0, "{}", a.issues());
    let key = |n: &Network| -> Vec<([u8; 2], u8, u8)> {
        let mut v: Vec<_> = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).map(|e| (n.edge_join[e as usize], n.edge_oneway[e as usize], n.edge_way(e))).collect();
        v.sort();
        v
    };
    let net = &a.w.net;
    assert_eq!(singles(&a).iter().filter(|&&e| net.edge_join[e as usize] != [0; 2]).count(), 2);
    let back = crate::track::world::TrackWorld::new(crate::track::save::decode(&crate::track::save::encode(net)).unwrap()).net;
    assert_eq!(key(&back), key(net));
    assert!(key(net).iter().any(|k| k.1 == 3) && key(net).iter().any(|k| k.2 != 0));
    unjoin(&mut a);
    let old = crate::track::world::TrackWorld::new(crate::track::save::decode(&crate::track::save::encode(&a.w.net)).unwrap()).net;
    assert!(key(&old).iter().all(|k| k.0 == [0; 2] && k.2 == 0));
}

/// T-3092's flyovers set their legs one way and join them at J and J3: the way the joins give by
/// themselves is the way the legs are set.
#[test]
fn t3137_flyover_legs_way_matches_their_flags() {
    for (right_hand, side) in [(true, 1.0), (true, -1.0), (false, 1.0), (false, -1.0)] {
        let mut a = api(right_hand);
        assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 2, false), 0);
        a.set_route_ramp(true);
        let r = a.add_route(&ends(on_edge(&a, 1000.0, 0.0), free(3500.0, side * 1500.0)), &[], 2, true);
        a.set_route_ramp(false);
        assert_eq!(r, 0, "{}", a.issues());
        assert_eq!(a.route_flyovers_flat(), 0);
        let net = &a.w.net;
        let legs = singles(&a);
        assert!(legs.len() >= 2);
        for e in legs {
            assert_eq!(net.edge_infer[e as usize], net.edge_oneway[e as usize], "right hand {right_hand}, side {side}, piece {e}");
        }
        let joined: usize = a.join_edges().chunks(2).map(|c| (c[1] & 3 != 0) as usize + (c[1] >> 2 != 0) as usize).sum();
        assert_eq!(joined, 4, "both legs joined at J and J3: {:?}", a.join_edges());
    }
}
