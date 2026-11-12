//! T-3059: rebuilding constructed track in place at another level or track type.

use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// A built 8 km run with a station 1 km from each end and a line over it, 3 pieces of track.
fn run() -> (TrackApi, [u32; 2], u32) {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let mut st = [0; 2];
    for (i, x) in [1000.0, 7000.0].into_iter().enumerate() {
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{i}")), 0, "{}", a.issues());
        st[i] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.w.svc.lines[line as usize].running);
    (a, st, line)
}

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}

fn levels(a: &TrackApi, e: u32) -> Vec<i8> {
    let d = a.w.net.edge_data(e).unwrap();
    d.pis.iter().chain(d.thru.iter()).map(|p| p.level).collect()
}

/// The whole run and both stations as a viaduct: the same ids and names, the line still running, the
/// full new price taken, no undo, and the quote said the same beforehand.
#[test]
fn the_whole_run_rebuilds_as_a_viaduct_for_its_full_price() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let names: Vec<String> = st.iter().map(|&n| a.w.net.station_data(n).unwrap().name).collect();
    let q = a.rebuild_quote(&edges, &st, 1, 0);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0 && q[3] == 0.0, "{q:?} {}", a.issues());
    assert_eq!(q.len(), 5 + edges.len(), "the changed edges follow the (no) refused ones");
    let (cash, built) = (a.cash(), a.built_cost());
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 0);
    assert!((cash - a.cash() - q[0]).abs() < 1e-9, "charged {} quoted {}", cash - a.cash(), q[0]);
    assert!((a.last_charge() - q[0]).abs() < 1e-9);
    let full: f64 = edges.iter().map(|&e| a.w.net.edge_cost[e as usize]).sum::<f64>() + st.iter().map(|&n| a.w.net.station_build_price(n)).sum::<f64>();
    assert!((q[0] - full).abs() < 1e-9, "the full price of all of it");
    assert!(a.built_cost() != built, "the built value moved with the price");
    assert_eq!(live(&a), edges, "no edge made or lost");
    for &e in &edges {
        assert!(a.w.net.edge_built[e as usize]);
        assert!(levels(&a, e).iter().all(|&l| l == 1), "{e}: {:?}", levels(&a, e));
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.w.net.station_data(n).unwrap().name, names[i]);
        assert!(a.w.net.node_built[n as usize] && a.w.net.node_level[n as usize] == 1);
    }
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
    assert!(!a.can_undo(), "final");
    // rebuilt again as it is: nothing to do, nothing charged
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 1);
    assert_eq!(a.last_skipped(), 0);
}

/// Cheaper track is charged in full too, and a viaduct that was already there costs nothing to leave.
#[test]
fn a_cheaper_rebuild_still_pays_the_full_new_price() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 0, "{}", a.issues());
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &st, 0, 0);
    assert!(q[0] > 0.0, "{q:?}");
    assert_eq!(a.rebuild_pieces(&edges, &st, 0, 0), 0, "{}", a.issues());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
}

#[test]
fn the_whole_run_rebuilds_as_high_speed() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &st, 99, 2);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0, "{q:?}");
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 2), 0, "{}", a.issues());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
    for &e in &edges {
        assert!(a.w.net.edge_kind[e as usize].is_hsr() && a.w.net.edge_built[e as usize]);
        assert!(levels(&a, e).iter().all(|&l| l == 0), "level kept");
    }
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running);
    // and back
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 1), 0, "{}", a.issues());
    assert!(edges.iter().all(|&e| !a.w.net.edge_kind[e as usize].is_hsr()));
}

/// Level and type in one rebuild.
#[test]
fn level_and_type_change_together() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &st, -1, 2), 0, "{}", a.issues());
    for &e in &edges {
        assert!(a.w.net.edge_kind[e as usize].is_hsr());
        assert!(levels(&a, e).iter().all(|&l| l == -1));
    }
}

/// T-3241: a pick of track that leaves its stations out still takes them to the new level, and charges them.
#[test]
fn stations_on_the_picked_track_move_with_it_unpicked() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &[], 1, 0);
    assert_eq!((q[3], q[2]), (0.0, 2.0), "nothing refused, both stations charged: {q:?} {}", a.issue_kinds());
    assert_eq!(a.cash(), cash, "a quote spends nothing");
    assert_eq!(a.rebuild_pieces(&edges, &[], 1, 0), 0, "{}", a.issue_kinds());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
    assert!(edges.iter().all(|&e| levels(&a, e).iter().all(|&l| l == 1)));
    assert!(st.iter().all(|&n| a.w.net.node_level[n as usize] == 1), "stations moved");
    assert_eq!(a.w.net.station_data(st[0]).unwrap().name, "S0");
    assert!(a.w.svc.lines[line as usize].running);
}

/// Heights along `e` from node `from` every 10 m, in levels: (metres from `from`, level).
fn profile_from(a: &TrackApi, e: u32, from: u32) -> Vec<(f64, f64)> {
    let (vert, len) = (a.w.net.edge_vert(e), a.w.net.edge_len[e as usize]);
    let flip = a.w.net.edge_a[e as usize] != from;
    let mut out = vec![];
    let mut s = 0.0;
    while s <= len + 1e-6 {
        let at = if flip { len - s } else { s };
        out.push((s, crate::track::geom::height_at(vert, at.clamp(0.0, len)) / LEVEL_H));
        s += 10.0;
    }
    out
}

/// The quote's stretches beyond the pick: (edge, the node it starts at, metres).
fn beyond_of(q: &[f64]) -> Vec<(u32, u32, f64)> {
    q[5 + q[3] as usize + q[1] as usize..].chunks(3).map(|c| (c[0] as u32, c[1] as u32, c[2])).collect()
}

fn ends_of(a: &TrackApi, e: u32) -> [u32; 2] {
    [a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]]
}

/// No track rule broken anywhere.
fn all_valid(a: &TrackApi) {
    let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).collect();
    assert_eq!(a.w.net.validate(&live(a), &nodes, &[]), vec![]);
}

/// T-3241, the players' case (a 4.3 km piece between Roissy-en-Brie and Ozoir-la-Ferriere to -2): the one
/// piece between two stations rebuilt to -2, and to +1, the stations not picked. Both stations go to the level
/// with it, platforms flat, and each ramps on the piece beyond it, starting just past its platform; nothing is
/// refused, the line runs, and the price is the piece, the two stations and the stretches beyond as built new.
#[test]
fn a_piece_between_two_stations_takes_them_and_ramps_beyond() {
    for level in [-2i8, 1] {
        let (mut a, st, line) = run();
        let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).iter().all(|n| st.contains(n))).unwrap();
        let outer: Vec<u32> = live(&a).into_iter().filter(|&e| e != mid).collect();
        assert!(a.w.net.edge_data(mid).unwrap().pis.is_empty(), "a piece with no corners, as drawn");
        let q = a.rebuild_quote(&[mid], &[], level as i32, 0);
        assert_eq!((q[1], q[2], q[3]), (1.0, 2.0, 0.0), "{level}: the piece and two stations, nothing refused: {q:?} {}", a.issue_kinds());
        let bey = beyond_of(&q);
        let ramp = 100.0 + 2.0 + 200.0 * (level as f64).abs();
        assert_eq!(bey.len(), 2, "{bey:?}");
        for &(e, from, m) in &bey {
            assert!(outer.contains(&e) && st.contains(&from) && ends_of(&a, e).contains(&from), "{bey:?}");
            assert!((m - ramp).abs() < 0.5, "{level}: the ramp ends {m} m past the station, wanted about {ramp}");
        }
        let cash = a.cash();
        assert_eq!(a.rebuild_pieces(&[mid], &[], level as i32, 0), 0, "{}", a.issue_kinds());
        assert_eq!(a.last_skipped(), 0);
        assert!((cash - a.cash() - q[0]).abs() < 1e-9, "charged {} quoted {}", cash - a.cash(), q[0]);
        let net = &a.w.net;
        // the stretches beyond: the platform's end and the ramp, at the new level's rate (a ramp is charged
        // at its dearer end), not the rest of those pieces at ground
        let part = q[0] - net.edge_cost[mid as usize] - st.iter().map(|&n| net.station_build_price(n)).sum::<f64>();
        let want = 2.0 * ramp / 1000.0 * BASE_COST_PER_KM * LEVEL_MULT[(level + 3) as usize];
        assert!((part - want).abs() < 0.1, "{level}: the stretches beyond cost {part}, wanted {want}");
        assert!(st.iter().all(|&n| net.node_level[n as usize] == level), "{level}: both stations moved");
        flat_at(&a, mid, level);
        for &(e, from, _) in &bey {
            let far = ends_of(&a, e).into_iter().find(|&n| n != from).unwrap();
            assert_eq!(net.node_level[far as usize], 0, "the far end stays");
            for (s, h) in profile_from(&a, e, from) {
                if s <= 102.0 {
                    assert!((h - level as f64).abs() < 1e-9, "{level}: flat at the platform, {s} m: {h}");
                }
                if s >= ramp + 0.5 {
                    assert!(h.abs() < 1e-9, "{level}: back at ground by {s} m: {h}");
                }
            }
        }
        all_valid(&a);
        assert_eq!(a.line_stops(line), st);
        assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
    }
}

/// The same pick as a blueprint: the level button does the same, free and undoable.
#[test]
fn the_level_button_takes_blueprint_stations_and_ramps_beyond() {
    let mut a = api();
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let mut st = vec![];
    for x in [1000.0, 7000.0] {
        let e = live(&a).into_iter().find(|&e| a.w.net.node_x[a.w.net.edge_a[e as usize] as usize] < x && a.w.net.node_x[a.w.net.edge_b[e as usize] as usize] > x).unwrap();
        assert_eq!(a.add_station(2, e, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
        st.push((0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap());
    }
    let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).iter().all(|n| st.contains(n))).unwrap();
    let cash = a.cash();
    assert_eq!(a.set_edges_level(&[mid], -2), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped(), 0);
    assert_eq!(a.cash(), cash);
    assert!(st.iter().all(|&n| a.w.net.node_level[n as usize] == -2));
    flat_at(&a, mid, -2);
    all_valid(&a);
    assert_eq!(a.undo(), 0);
    assert!(st.iter().all(|&n| a.w.net.node_level[n as usize] == 0));
    assert!(live(&a).iter().all(|&e| a.w.net.edge_data(e).unwrap().pis.is_empty()), "the ramps' corners gone with the undo");
}

/// The piece beyond a station is too short for its ramp and ends at a junction: the station stays, so the
/// picked piece (no corners to ramp on) cannot change, and the refusal says there is no room to change level.
#[test]
fn no_room_beyond_a_station_before_a_junction() {
    let (mut a, st, _) = run();
    let tail = *live(&a).iter().find(|&&e| ends_of(&a, e).contains(&st[1]) && !ends_of(&a, e).contains(&st[0])).unwrap();
    let ends = [2.0, tail as f64, 7300.0, 0.0, 0.0, 0.0, 0.0, 9500.0, 1500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issue_kinds());
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).iter().all(|n| st.contains(n))).unwrap();
    let q = a.rebuild_quote(&[mid], &[], -2, 0);
    assert_eq!((q[1], q[3]), (0.0, 1.0), "{q:?}");
    assert!(a.issue_kinds().contains("RampTooShort"), "{}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&[mid], &[], -2, 0), 1);
    assert!(st.iter().all(|&n| a.w.net.node_level[n as usize] == 0));
}

/// A short piece beyond the station runs on through a plain joint: it and the joint go to the level whole,
/// and the ramp is on the piece after, from the joint.
#[test]
fn a_short_piece_beyond_carries_the_ramp_on_through_a_joint() {
    let mut a = api();
    a.set_cash(1e6);
    let free = |x: f64| [0.0, 0.0, x, 0.0, 0.0];
    let node_at = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 0.5).unwrap();
    let route = |a: &mut TrackApi, from: [f64; 5], x: f64| {
        let ends: Vec<f64> = from.iter().copied().chain(free(x)).collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issue_kinds());
    };
    route(&mut a, free(0.0), 3000.0);
    let n3000 = node_at(&a, 3000.0);
    route(&mut a, [1.0, n3000 as f64, 0.0, 0.0, 0.0], 3150.0);
    let joint = node_at(&a, 3150.0);
    route(&mut a, [1.0, joint as f64, 0.0, 0.0, 0.0], 6000.0);
    assert_eq!(a.add_station(1, n3000, 0.0, 0.0, 200, "S0".into()), 0, "{}", a.issue_kinds());
    let e0 = live(&a).into_iter().find(|&e| ends_of(&a, e).contains(&n3000) && !ends_of(&a, e).contains(&joint)).unwrap();
    assert_eq!(a.add_station(2, e0, 500.0, 0.0, 200, "S1".into()), 0, "{}", a.issue_kinds());
    let s1 = node_at(&a, 500.0);
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &[s1, n3000], 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issue_kinds());
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    assert_eq!(a.w.net.node_ports[joint as usize].n, 2, "a plain joint");
    let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).contains(&s1) && ends_of(&a, e).contains(&n3000)).unwrap();
    let short = *live(&a).iter().find(|&&e| ends_of(&a, e).contains(&n3000) && ends_of(&a, e).contains(&joint)).unwrap();
    let long = *live(&a).iter().find(|&&e| ends_of(&a, e).contains(&joint) && e != short).unwrap();
    let q = a.rebuild_quote(&[mid], &[], -2, 0);
    assert_eq!(q[3], 0.0, "{q:?} {}", a.issue_kinds());
    let bey = beyond_of(&q);
    assert!(bey.iter().any(|&(e, f, m)| e == short && f == n3000 && (m - 150.0).abs() < 0.5), "{bey:?}");
    assert!(bey.iter().any(|&(e, f, m)| e == long && f == joint && (m - 402.0).abs() < 0.5), "{bey:?}");
    assert_eq!(a.rebuild_pieces(&[mid], &[], -2, 0), 0, "{}", a.issue_kinds());
    assert_eq!((a.w.net.node_level[n3000 as usize], a.w.net.node_level[joint as usize], a.w.net.node_level[s1 as usize]), (-2, -2, -2));
    flat_at(&a, short, -2);
    all_valid(&a);
    assert!(a.w.svc.lines[line as usize].running);
}

/// Track beyond the station that curves: the ramp goes on the first straight it fits on (before the curve
/// for one level, after it for two), the shape and the speed limit of that piece stay as they were, on
/// ordinary and on high-speed track.
#[test]
fn a_ramp_beyond_finds_a_straight_and_keeps_the_shape() {
    for hsr in [false, true] {
        for level in [-1i8, -2] {
            let mut a = api();
            a.set_cash(1e6);
            a.set_route_hsr(hsr);
            // west to east, then a 300 m curve at x = 1500 up to the north-east
            let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 1777.0, 0.0];
            assert_eq!(a.add_route(&ends, &[1500.0, 0.0, 300.0, 0.0], 2, false), 0, "{}", a.issues());
            let mut st = vec![];
            for x in [400.0, 1000.0] {
                let e = live(&a).into_iter().find(|&e| a.w.net.node_x[a.w.net.edge_a[e as usize] as usize] < x && a.w.net.node_x[a.w.net.edge_b[e as usize] as usize] > x).unwrap();
                assert_eq!(a.add_station(2, e, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
                st.push((0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && (a.w.net.node_x[n as usize] - x).abs() < 0.5).unwrap());
            }
            let line = a.new_line_id();
            assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issue_kinds());
            assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
            let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).iter().all(|n| st.contains(n))).unwrap();
            let curve = *live(&a).iter().find(|&&e| !a.w.net.edge_data(e).unwrap().pis.is_empty()).unwrap();
            assert!(ends_of(&a, curve).contains(&st[1]));
            let info = |a: &TrackApi, e: u32| a.edges_info().chunks(10).find(|c| c[0] as u32 == e).unwrap().to_vec();
            let (before, shape) = (info(&a, curve), a.w.net.edge_pieces(curve).to_vec());
            let q = a.rebuild_quote(&[mid], &[], level as i32, 0);
            assert_eq!(q[3], 0.0, "{hsr} {level}: {q:?} {}", a.issue_kinds());
            let bey = beyond_of(&q);
            let on_curve = bey.iter().find(|x| x.0 == curve).unwrap().2;
            assert_eq!(a.w.net.edge_kind[curve as usize].is_hsr(), hsr);
            let top = if hsr { V_HSR } else { V_TOP };
            let slow = |p: &[crate::track::geom::Piece]| p.iter().filter(|p| p.v_curve() < top - 0.01).count();
            if level == -1 {
                assert!((on_curve - 302.0).abs() < 0.5, "{hsr} {level}: on the straight before the curve, {on_curve}");
            } else {
                assert!(on_curve > 650.0 && on_curve < 1100.0, "{hsr} {level}: on the straight after the curve, {on_curve}");
            }
            assert_eq!(a.rebuild_pieces(&[mid], &[], level as i32, 0), 0, "{}", a.issue_kinds());
            let after = info(&a, curve);
            assert!((after[5] - before[5]).abs() < 0.05, "{hsr} {level}: length {} to {}", before[5], after[5]);
            assert_eq!(after[7], before[7], "{hsr} {level}: the speed limit stays");
            assert_eq!(slow(a.w.net.edge_pieces(curve)), slow(&shape), "{hsr} {level}: no new curve under the track's top speed");
            let mut s = 0.0;
            while s < before[5] {
                let (p, n) = (crate::track::geom::pos_at(&shape, s), crate::track::geom::pos_at(a.w.net.edge_pieces(curve), s));
                assert!((p.0 - n.0).hypot(p.1 - n.1) < 0.25, "{hsr} {level}: the shape moved at {s}");
                s += 10.0;
            }
            for (s, h) in profile_from(&a, curve, st[1]) {
                if s <= 102.0 {
                    assert!((h - level as f64).abs() < 1e-9, "{hsr} {level}: flat at the platform, {s} m: {h}");
                }
                if s >= on_curve + 0.5 {
                    assert!(h.abs() < 1e-9, "{hsr} {level}: back at ground by {s} m: {h}");
                }
            }
            all_valid(&a);
            assert!(a.w.svc.lines[line as usize].running);
        }
    }
}

/// A station at the end of the track just moves; a short piece from a station to a loose end moves whole.
#[test]
fn a_terminus_moves_and_a_short_stub_beyond_moves_whole() {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 0.5).unwrap();
    let t = node_at(&a, 0.0);
    assert_eq!(a.add_station(1, t, 0.0, 0.0, 200, "T".into()), 0, "{}", a.issues());
    let e = live(&a)[0];
    assert_eq!(a.add_station(2, e, 3850.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    let s = node_at(&a, 3850.0);
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &[t, s], 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issue_kinds());
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    let mid = *live(&a).iter().find(|&&e| ends_of(&a, e).contains(&t)).unwrap();
    let stub = *live(&a).iter().find(|&&e| e != mid).unwrap();
    let end = ends_of(&a, stub).into_iter().find(|&n| n != s).unwrap();
    let q = a.rebuild_quote(&[mid], &[], -1, 0);
    assert_eq!((q[2], q[3]), (2.0, 0.0), "{q:?} {}", a.issue_kinds());
    assert_eq!(beyond_of(&q).len(), 1, "only the stub; nothing beyond the terminus");
    assert_eq!(a.rebuild_pieces(&[mid], &[], -1, 0), 0, "{}", a.issue_kinds());
    assert_eq!((a.w.net.node_level[t as usize], a.w.net.node_level[s as usize], a.w.net.node_level[end as usize]), (-1, -1, -1));
    flat_at(&a, stub, -1);
    all_valid(&a);
    assert!(a.w.svc.lines[line as usize].running);
}

/// Water where x (from the frame's origin at the api's place) lies in any of the bands.
fn water(a: &mut TrackApi, bands: &[(f64, f64)]) {
    let bx = a.from_lonlat(-73.985, 40.758)[0];
    let bands = bands.to_vec();
    a.w.net.set_water_mask(Box::new(move |x: f64, _y: f64| bands.iter().any(|b| x > bx + b.0 && x < bx + b.1)));
    a.w.reload();
}

/// An 8 km run drawn at ground over three stretches of water (so bridges at +1), stations 1 km from
/// each end, a line, constructed.
fn wet_run(cross: bool) -> (TrackApi, Vec<u32>, u32) {
    let mut a = api();
    a.set_cash(1e6);
    water(&mut a, &[(1400.0, 1600.0), (3700.0, 4300.0), (5400.0, 5600.0), (6400.0, 6550.0)]);
    let o = a.from_lonlat(-73.985, 40.758);
    let (bx, by) = (o[0], o[1]);
    if cross {
        // constructed tunnel crossing the run's 2-3 km piece
        let ends = [0.0, 0.0, bx + 2500.0, by - 1500.0, -1.0, 0.0, 0.0, bx + 2500.0, by + 1500.0, -1.0];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issue_kinds());
        assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    }
    // drawn a km at a time, as a player does, so the run has joints, one of them on a bridge
    let mut from = [0.0, 0.0, bx, by, 0.0];
    for k in 1..=8 {
        let x = bx + 1000.0 * k as f64;
        let ends: Vec<f64> = from.iter().copied().chain([0.0, 0.0, x, by, 0.0]).collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "piece {k}: {}", a.issue_kinds());
        let n = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 1.0 && (a.w.net.node_y[n as usize] - by).abs() < 1.0).unwrap();
        from = [1.0, n as f64, 0.0, 0.0, 0.0];
    }
    let mut st = vec![];
    for (i, x) in [500.0, 7500.0].into_iter().enumerate() {
        let x = bx + x;
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        assert_eq!(a.add_station(2, edge, x, by, 200, format!("S{i}")), 0, "{}", a.issue_kinds());
        st.push((0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && (a.w.net.node_x[n as usize] - x).abs() < 1.0).unwrap());
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issue_kinds());
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    let all: Vec<i8> = live(&a).iter().flat_map(|&e| levels(&a, e)).collect();
    assert!(all.contains(&0) && all.contains(&1), "drawn at 0 with bridges at +1: {all:?}");
    (a, st, line)
}

/// The run's pieces (the crossing tunnel left out).
fn run_edges(a: &TrackApi) -> Vec<u32> {
    live(a).into_iter().filter(|&e| a.w.net.node_y[a.w.net.edge_a[e as usize] as usize] == a.w.net.node_y[a.w.net.edge_b[e as usize] as usize]).collect()
}

/// After a rebuild to -1 that refused `refused`: every other piece is at -1, and flat at -1 where
/// neither of its nodes belongs to a refused piece or is in `kept` (nodes left out of the pick);
/// nothing anywhere breaks the track rules.
fn all_down_but(a: &TrackApi, edges: &[u32], refused: &[u32], kept: &[u32], before: &[Vec<i8>]) {
    let near_refused = |n: u32| kept.contains(&n) || refused.iter().any(|&r| a.w.net.edge_a[r as usize] == n || a.w.net.edge_b[r as usize] == n);
    for (i, &e) in edges.iter().enumerate() {
        if refused.contains(&e) {
            assert_eq!(levels(a, e), before[i], "{e} left as it was");
            continue;
        }
        assert!(levels(a, e).iter().all(|&l| l == -1), "{e}: {:?}", levels(a, e));
        let (na, nb) = (a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]);
        if !near_refused(na) && !near_refused(nb) {
            assert_eq!((a.w.net.node_level[na as usize], a.w.net.node_level[nb as usize]), (-1, -1), "{e}: no ramp inside the pick");
            flat_at(a, e, -1);
        }
    }
    let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).collect();
    assert_eq!(a.w.net.validate(&live(a), &nodes, &[]), vec![], "nothing at ground over water");
}

/// Every height along `e` is level `l`'s.
fn flat_at(a: &TrackApi, e: u32, l: i8) {
    let (vert, len) = (a.w.net.edge_vert(e), a.w.net.edge_len[e as usize]);
    let mut s = 0.0;
    while s <= len {
        let h = crate::track::geom::height_at(vert, s);
        assert!((h - l as f64 * LEVEL_H).abs() < 1e-6, "{e} at {s}: {h}");
        s += 10.0;
    }
}

/// T-3087: a run at ground and on bridges (pieces at 0 and +1) rebuilt to -1 with its stations: every
/// piece and node goes to -1 (a tunnel under the water), nothing refused.
#[test]
fn a_run_at_ground_and_on_bridges_goes_down_whole() {
    let (mut a, st, line) = wet_run(false);
    let edges = run_edges(&a);
    let q = a.rebuild_quote(&edges, &st, -1, 0);
    assert_eq!(q[3], 0.0, "nothing refused: {q:?} {}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&edges, &st, -1, 0), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped(), 0);
    let before: Vec<Vec<i8>> = edges.iter().map(|&e| levels(&a, e)).collect();
    all_down_but(&a, &edges, &[], &[], &before);
    for &e in &edges {
        for n in [a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]] {
            assert_eq!(a.w.net.node_level[n as usize], -1, "node {n}");
        }
    }
    assert!(a.w.svc.lines[line as usize].running);
}

/// T-3087, Anita's case: one piece of the run that cannot go down (here it would cross a tunnel flat;
/// for her, a ramp with no room) is refused alone. Every other piece and the nodes between them go to
/// -1 whole, ramping only to the refused piece, and nothing is left at ground over water.
#[test]
fn one_refused_piece_does_not_leave_the_rest_ramping_at_every_node() {
    let (mut a, st, line) = wet_run(true);
    let edges = run_edges(&a);
    let q = a.rebuild_quote(&edges, &st, -1, 0);
    let refused: Vec<u32> = q[5..5 + q[3] as usize].iter().map(|&e| e as u32).collect();
    let x = |n: u32| a.w.net.node_x[n as usize] - a.from_lonlat(-73.985, 40.758)[0];
    let crossing = |e: u32| x(a.w.net.edge_a[e as usize]) < 2500.0 && x(a.w.net.edge_b[e as usize]) > 2500.0;
    assert!(refused.len() == 1 && crossing(refused[0]), "only the piece over the tunnel: {refused:?} {}", a.issue_kinds());
    let before: Vec<Vec<i8>> = edges.iter().map(|&e| levels(&a, e)).collect();
    assert_eq!(a.rebuild_pieces(&edges, &st, -1, 0), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped() as usize, refused.len());
    all_down_but(&a, &edges, &refused, &[], &before);
    assert!(a.w.svc.lines[line as usize].running);
}

/// The same run with one station left out of the pick: it goes down with its track all the same (T-3241).
#[test]
fn a_station_left_out_goes_down_with_its_track() {
    for left_out in [0, 1] {
        let (mut a, st, line) = wet_run(false);
        let edges = run_edges(&a);
        let picked = [st[1 - left_out]];
        let q = a.rebuild_quote(&edges, &picked, -1, 0);
        assert_eq!((q[2], q[3]), (2.0, 0.0), "both stations charged, nothing refused: {q:?} {}", a.issue_kinds());
        let before: Vec<Vec<i8>> = edges.iter().map(|&e| levels(&a, e)).collect();
        assert_eq!(a.rebuild_pieces(&edges, &picked, -1, 0), 0, "{}", a.issue_kinds());
        assert_eq!(a.last_skipped(), 0);
        all_down_but(&a, &edges, &[], &[], &before);
        assert_eq!(a.w.net.node_level[st[left_out] as usize], -1);
        assert!(a.w.svc.lines[line as usize].running);
    }
}

/// A short purse: nothing changes.
#[test]
fn a_short_purse_refuses_the_rebuild() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    a.set_cash(0.001);
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 1);
    assert!(a.money_short.is_some());
    assert!(edges.iter().all(|&e| levels(&a, e).iter().all(|&l| l == 0)));
}

/// Blueprint pieces in the pick are not rebuilt (Construct builds them).
#[test]
fn blueprint_pieces_are_not_rebuilt() {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &[], 1, 0), 1);
    assert_eq!(a.rebuild_quote(&edges, &[], 1, 0)[..5], [0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// T-3088: the run and its stations rebuilt as tram track. The stations become tram stops cut to the
/// longest tram platform, keeping their names; the line becomes a tram line, takes the tram dwell and
/// turnaround it had rail's of, needs trams and runs. Then back to rail.
#[test]
fn a_rail_run_rebuilds_as_tram_and_back() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    assert!(!a.line_tram(line) && a.cars_needed() > 0);
    let q = a.rebuild_quote(&edges, &st, 99, 3);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0 && q[3] == 0.0, "{q:?} {}", a.issue_kinds());
    let cash = a.cash();
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 3), 0, "{}", a.issue_kinds());
    assert!(cash - a.cash() > q[0] + 1.0, "the quote, and the trams the line needs: {} {}", cash - a.cash(), q[0]);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram && a.w.net.edge_built[e as usize]));
    assert_eq!(live(&a), edges);
    for (i, &n) in st.iter().enumerate() {
        let s = a.w.net.station_data(n).unwrap();
        assert_eq!((s.platform, s.name.as_str(), s.built), (80, ["S0", "S1"][i], true));
    }
    assert!(a.line_tram(line));
    assert_eq!(a.line_stops(line), st);
    let l = line as usize;
    assert_eq!((a.w.net.line_dwell[l], a.w.net.line_turn[l]), (15.0, 120.0));
    assert!(a.w.svc.lines[l].running && !a.w.net.line_broken[l]);
    assert!(a.trams_needed() > 0 && a.cars_needed() == 0 && a.fleet_tram() >= a.trams_needed());
    // and back to rail: a tram stop's 80 m is long enough for a train
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 1), 0, "{}", a.issue_kinds());
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Regular));
    assert!(!a.line_tram(line) && a.w.svc.lines[l].running);
    assert_eq!((a.w.net.line_dwell[l], a.w.net.line_turn[l]), (30.0, 180.0));
    assert!(st.iter().all(|&n| a.w.net.node_platform[n as usize] == 80));
}

/// Part of a run cannot become tram track: it would join the rail left beside it. The piece that
/// meets that rail is refused for its own sake, as a tram-train join; the one beyond it only with it.
#[test]
fn part_of_a_run_cannot_change_family() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    let part: Vec<u32> = edges.iter().copied().filter(|&e| a.w.net.edge_a[e as usize] == st[0] || a.w.net.edge_b[e as usize] == st[0]).collect();
    assert_eq!(part.len(), 2);
    let meets = *part.iter().find(|&&e| [a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]].iter().all(|&n| a.w.net.node_ports[n as usize].n == 2)).unwrap();
    let q = a.rebuild_quote(&part, &st[..1], 99, 3);
    assert_eq!((q[1], q[3], q[4], q[5]), (0.0, 2.0, 1.0, meets as f64), "{q:?}: the piece that meets the rail is the red one");
    assert!(a.issue_kinds().contains("TrackFamily"), "{}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&part, &st[..1], 99, 3), 1);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Regular));
}

/// Tram track with a curve too tight for trains, rebuilt as rail: the curve is refused for its own
/// sake, the straight beside it only because it cannot become rail alone, and nothing changes.
#[test]
fn a_tram_curve_too_tight_for_trains_stays_tram() {
    let mut a = api();
    a.set_cash(1e6);
    a.set_route_kind(3);
    assert_eq!(a.add_route(&[0.0, 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
    let end = a.route_end();
    assert_eq!(a.add_route(&[1.0, end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 300.0, 0.0], &[300.0, 0.0, 20.0, 0.0], 2, false), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(edges.len(), 2);
    let curve = *edges.iter().find(|&&e| !a.w.net.edge_data(e).unwrap().pis.is_empty()).unwrap();
    let q = a.rebuild_quote(&edges, &[], 99, 1);
    assert_eq!((q[1], q[3], q[4], q[5]), (0.0, 2.0, 1.0, curve as f64), "{q:?} {}", a.issue_kinds());
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 1), 1);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram));
}

/// T-3108: blueprint track changes between regular, high speed and tram by the rebuild rules, free and
/// undoable: whole stretches, stations cut to a tram stop and back, the line following.
#[test]
fn blueprint_switches_between_regular_high_speed_and_tram() {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let mut st = [0; 2];
    for (i, x) in [1000.0, 7000.0].into_iter().enumerate() {
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{i}")), 0, "{}", a.issues());
        st[i] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issues());
    let edges = live(&a);
    let kinds = |a: &TrackApi, k: TrackKind| edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == k);
    let cash = a.cash();

    // part of the run alone would meet the rail beside it: nothing changes
    let part: Vec<u32> = edges.iter().copied().filter(|&e| a.w.net.edge_a[e as usize] == st[0] || a.w.net.edge_b[e as usize] == st[0]).collect();
    assert_eq!(a.set_edges_kind(&part, 3), 1);
    assert_eq!(a.last_skipped(), 2);
    assert!(kinds(&a, TrackKind::Regular));

    // rail to tram: stations cut to 80 m, the line a tram line with the tram dwell, nothing paid
    assert_eq!(a.set_edges_kind(&edges, 3), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped(), 0);
    assert!(kinds(&a, TrackKind::Tram));
    assert!(st.iter().all(|&n| a.w.net.node_platform[n as usize] == 80 && !a.w.net.node_built[n as usize]));
    assert!(a.line_tram(line));
    assert_eq!((a.w.net.line_dwell[line as usize], a.w.net.line_turn[line as usize]), (15.0, 120.0));
    assert_eq!(a.cash(), cash);
    // already tram: nothing to do
    assert_eq!(a.set_edges_kind(&edges, 3), 1);

    // tram to high speed and back to tram
    assert_eq!(a.set_edges_kind(&edges, 2), 0, "{}", a.issue_kinds());
    assert!(kinds(&a, TrackKind::HighSpeed) && !a.line_tram(line));
    assert_eq!(a.set_edges_kind(&edges, 3), 0, "{}", a.issue_kinds());
    assert!(kinds(&a, TrackKind::Tram) && a.line_tram(line));

    // a mixed pick (one piece high speed, the rest tram) goes to regular in one go, and one undo reverts it
    assert_eq!(a.set_edges_kind(&edges[..1], 2), 1, "a lone piece cannot leave its family");
    assert_eq!(a.set_edges_kind(&edges, 1), 0, "{}", a.issue_kinds());
    assert!(kinds(&a, TrackKind::Regular) && !a.line_tram(line));
    a.undo();
    assert!(kinds(&a, TrackKind::Tram) && a.line_tram(line));
}
