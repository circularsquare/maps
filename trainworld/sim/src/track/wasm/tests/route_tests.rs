use super::*;

#[test]
fn degenerate_routes_do_not_panic() {
    let mut a = api();
    let same: Vec<f64> = free(10.0, 10.0).iter().chain(free(10.0, 10.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&same, &[], 2, false)[0], 0.0);
    assert_ne!(a.add_route(&same, &[], 2, false), 0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&ends, &[0.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
    assert_eq!(a.preview_route(&ends, &[1000.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let n = a.nodes_info()[0];
    let e = a.edges_info()[0];
    // From a node to the same node, and onto the edge at its own end.
    let r = [1.0, n, 0.0, 0.0, 0.0, 1.0, n, 0.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    let r = [1.0, n, 0.0, 0.0, 0.0, 2.0, e, 1.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    let r = [2.0, e, 500.0, 0.0, 0.0, 2.0, e, 500.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    assert_eq!(a.edges_info().len(), 10);
}

#[test]
fn branch_leaves_along_the_heading() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    // Branch from the middle of the edge, off to the north-east.
    let ends = [2.0, e, 2000.0, 5.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.edges_info().len() / 10, 3);
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).map(|c| c[0]);
    assert!(j.is_some(), "a junction");
    // Extending a track end, turning a little.
    let end_node = a.nodes_info().chunks(10).find(|c| (c[1] - 4000.0).abs() < 1e-3).unwrap()[0];
    let ends = [1.0, end_node, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, -600.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    // Undo all three, redo one.
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edges_info().len() / 10, 1);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.edges_info().len() / 10, 3);
    // A segment that ends 100 m past water has no land for a ramp down (T-656): its new end stays up
    // (T-2414).
    let mut w = api();
    w.w.net.set_water_mask(Box::new(|x: f64, _y: f64| x > 1000.0 && x < 1500.0));
    w.w.reload();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1600.0, 0.0).iter()).copied().collect();
    assert_eq!(w.add_route(&ends, &[], 2, false), 0, "{}", w.issue_kinds());
    assert!(w.nodes_info().chunks(10).any(|c| (c[1] - 1600.0).abs() < 1e-3 && c[3] == 1.0), "{:?}", w.nodes_info());
}

/// T-237: the stop markers' tiers come from the line tool's own search: a station past either
/// end over track the line can carry on over is reachable as it is (the path only grows), a
/// branch past an end too, a station on a bypass changes the path, and a branch the line would
/// have to turn back to reach or a station on other track cannot be added.
#[test]
fn stop_tiers_follow_the_line_tool() {
    let mut a = api();
    let route = |a: &mut TrackApi, ends: [f64; 10], pis: &[f64]| assert_eq!(a.add_route(&ends, pis, 2, false), 0, "{}", a.issues());
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9000.0, 0.0, 0.0], &[]);
    let on_main = |a: &TrackApi, x: f64| (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, 0.0).1 < 0.01 && a.w.net.node_y[a.w.net.edge_a[e as usize] as usize] == 0.0 && a.w.net.node_y[a.w.net.edge_b[e as usize] as usize] == 0.0).unwrap() as f64;
    // a bypass from 2 km to 4 km, a branch from 6 km past the end, one from 4.5 km facing back
    let e1 = on_main(&a, 2000.0);
    let ends = [2.0, e1, 2000.0, 0.0, 0.0, 2.0, e1, 4000.0, 0.0, 0.0];
    route(&mut a, ends, &[3000.0, 800.0, 0.0, 0.0]);
    let e = on_main(&a, 6000.0);
    route(&mut a, [2.0, e, 6000.0, 0.0, 0.0, 0.0, 0.0, 7500.0, 1500.0, 0.0], &[]);
    let e = on_main(&a, 4500.0);
    route(&mut a, [2.0, e, 4500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, -1500.0, 0.0], &[]);
    // other track, not joined
    route(&mut a, [0.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 7000.0, 5000.0, 0.0], &[]);
    let nearest = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
    let station = |a: &mut TrackApi, x: f64, y: f64, name: &str| -> u32 {
        let e = nearest(a, x, y);
        assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{name}: {}", a.issues());
        let n = a.w.net.node_count() as u32;
        (0..n).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).min_by(|&p, &q| {
            let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap()
    };
    let first = station(&mut a, 1000.0, 0.0, "A");
    let last = station(&mut a, 5000.0, 0.0, "B");
    let between = station(&mut a, 3000.0, 0.0, "M");
    let past_end = station(&mut a, 8000.0, 0.0, "C");
    let before_start = station(&mut a, 300.0, 0.0, "D");
    let bypass = station(&mut a, 3000.0, 650.0, "S");
    let branch = station(&mut a, 7200.0, 1300.0, "Y");
    let facing_back = station(&mut a, 3700.0, -1300.0, "Z");
    let other = station(&mut a, 6000.0, 5000.0, "E");
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[first, last], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let raw = a.stop_tiers(id);
    let tiers: std::collections::HashMap<u32, f64> = raw.chunks(3).map(|c| (c[0] as u32, c[1])).collect();
    assert_eq!(tiers.len(), 9);
    // T-797: the branch facing back is reached by reversing at B
    let want = [(first, 0.0), (last, 0.0), (between, 1.0), (past_end, 1.0), (before_start, 1.0), (bypass, 2.0), (branch, 1.0), (facing_back, 4.0), (other, 3.0)];
    for (n, t) in want {
        assert_eq!(tiers[&n], t, "station {}", a.w.net.node_name[n as usize]);
    }
    for c in raw.chunks(3) {
        assert_eq!(c[2], if c[0] as u32 == facing_back { last as f64 } else { -1.0 });
    }
    // and a click does what the marker says
    let path = a.line_path(id);
    for (n, t) in want {
        let stops: Vec<u32> = a.insert_stop(id, n).iter().map(|&s| s as u32).collect();
        assert_eq!(stops.is_empty(), t == 0.0 || t == 3.0, "station {}", a.w.net.node_name[n as usize]);
        if stops.is_empty() {
            continue;
        }
        let guess: std::collections::BTreeSet<u32> = a.click_path(id, n).iter().map(|&e| e as u32).collect();
        assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
        let grown = a.line_path(id);
        // T-4147: the path read at the hover is the one the click's edit takes
        assert_eq!(guess, grown.iter().chain(&a.line_back_path(id)).map(|&p| p >> 1).collect(), "station {}", a.w.net.node_name[n as usize]);
        if t == 4.0 {
            assert_eq!(a.line_reversals(id), vec![1.0], "trains reverse at B, stop 1 of A, B, Z");
        } else {
            assert_eq!(grown.windows(path.len()).any(|w| w == path), t == 1.0, "station {}", a.w.net.node_name[n as usize]);
        }
        assert_eq!(a.undo(), 0);
    }
    // T-4097: the path read over a stop is the one taking it out takes, round the bypass too
    for via in [between, bypass] {
        assert_eq!(a.set_stops(id, &[first, via, last]), 0, "{}", a.issues());
        let guess: std::collections::BTreeSet<u32> = a.click_path(id, via).iter().map(|&e| e as u32).collect();
        let was = a.line_path(id);
        assert_eq!(a.take_stop(id, via), 0, "{}", a.issues());
        assert_eq!(a.line_stops(id), vec![first, last]);
        assert_eq!(guess, a.line_path(id).iter().chain(&a.line_back_path(id)).map(|&p| p >> 1).collect(), "station {}", a.w.net.node_name[via as usize]);
        assert_eq!(was == a.line_path(id), via == between, "only the bypass moves the path");
        assert!(a.take_path(id, via).is_empty(), "no stop now");
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);
    }
    // with no route yet, every station joined to the one stop counts as on it
    let lone = a.new_line_id();
    assert_eq!(a.set_line_with_trains(lone, &[first], 0, 0, 0, 30.0, 180.0, 0, 0, "One".into()), 0, "{}", a.issues());
    let tiers: std::collections::HashMap<u32, f64> = a.stop_tiers(lone).chunks(3).map(|c| (c[0] as u32, c[1])).collect();
    assert_eq!((tiers[&past_end], tiers[&bypass], tiers[&other]), (1.0, 1.0, 3.0));
}

/// T-797 B3: from M, Br is reached only by running out past T to the bare end X and back.
#[test]
fn a_station_past_a_headshunt_reverses_at_the_track_end() {
    use super::super::super::tests::{line, reversal_fixture_with};
    let mut a = api();
    let (w, [x, t, _, m, br]) = reversal_fixture_with(false);
    a.w = w;
    let id = line(&mut a.w, &[m], [6.0, 3.0, 0.0]);
    let raw = a.stop_tiers(id);
    let tier = |n: u32| raw.chunks(3).find(|c| c[0] as u32 == n).map(|c| (c[1], c[2])).unwrap();
    assert_eq!(tier(br), (4.0, x as f64));
    assert_eq!(tier(t), (1.0, -1.0));
    let stops: Vec<u32> = a.insert_stop(id, br).iter().map(|&s| s as u32).collect();
    assert_eq!(stops, vec![m, br]);
    assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    assert_eq!(a.line_end_turns(id), vec![x as f64]);
    assert!(a.line_reversals(id).is_empty());
    let split = a.line_dwell_split(id);
    assert!((split[1] - 4.0 * 180.0).abs() < 1e-9, "two ends and the track end both ways: {split:?}");
}

/// T-3541: blueprint drawn on from the bare end a running line turns at leaves it turning there and
/// running; once that track is built the line turns at its new end, still running.
#[test]
fn blueprint_past_a_headshunt_leaves_the_line_running() {
    use super::super::super::tests::{line, reversal_fixture_with};
    let mut a = api();
    let (w, [x, _, _, m, br]) = reversal_fixture_with(false);
    a.w = w;
    let id = line(&mut a.w, &[m, br], [6.0, 3.0, 0.0]);
    assert_eq!(a.line_end_turns(id), vec![x as f64]);
    let ends = [1.0, x as f64, 0.0, 0.0, 0.0, 0.0, 0.0, -4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.ok && ls.running, "a blueprint past X stops the line");
    assert_eq!(a.line_end_turns(id), vec![x as f64]);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.ok && ls.running, "built past X, the line no longer runs");
    assert_ne!(a.line_end_turns(id), vec![x as f64], "X is no track end once built past");
}

/// T-3595: as above, but the new track ends at a station the line does not stop at. Built, X is no
/// track end; the line turns at the station's end instead, running in and out without stopping.
#[test]
fn built_past_a_headshunt_to_a_station_leaves_the_line_running() {
    use super::super::super::tests::{line, reversal_fixture_with, station};
    let mut a = api();
    let (w, [x, _, _, m, br]) = reversal_fixture_with(false);
    a.w = w;
    let id = line(&mut a.w, &[m, br], [6.0, 3.0, 0.0]);
    let ends = [1.0, x as f64, 0.0, 0.0, 0.0, 0.0, 0.0, -4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let net = &a.w.net;
    let far = (0..net.node_count() as u32).filter(|&n| net.node_ok(n)).min_by(|&p, &q| {
        let d = |n: u32| (net.node_x[n as usize] + 4000.0).hypot(net.node_y[n as usize]);
        d(p).total_cmp(&d(q))
    }).unwrap();
    station(&mut a.w, far, 200);
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.ok && ls.running, "a station on blueprint past X stops the line");
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.ok && ls.running, "built past X to a station, the line no longer runs");
    assert_eq!(a.line_end_turns(id), vec![far as f64], "it turns at the new end, the station");
    assert!(!ls.runs.iter().any(|r| r.stop_idx.iter().any(|&k| r.nodes[k as usize] == far)), "and does not stop there");
    // built on again past that station: the line turns at the newer end, still running
    let fx = a.w.net.node_x[far as usize];
    let on = [1.0, far as f64, fx, 0.0, 0.0, 0.0, 0.0, fx - 1500.0, 0.0, 0.0];
    assert_eq!(a.add_route(&on, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let ls = &a.w.svc.lines[id as usize];
    assert!(ls.ok && ls.running, "built past the station too, the line no longer runs");
    let turns = a.line_end_turns(id);
    assert!(turns.len() == 1 && turns[0] as u32 != far, "it turns at the newest end, {turns:?}");
}

/// T-798: on a ring the first stop's marker closes the line into a circle, then opens it; a
/// closed line takes new stops inside the loop only.
#[test]
fn the_first_stop_closes_and_opens_a_loop() {
    let mut a = api();
    let (w, ns) = super::super::super::tests::ring_fixture();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &ns[..4], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let tier = |a: &TrackApi, n: u32| a.stop_tiers(id).chunks(3).find(|c| c[0] as u32 == n).unwrap()[1];
    assert_eq!(tier(&a, ns[0]), 5.0, "closes");
    assert_eq!(tier(&a, ns[3]), 5.0, "T-3203: the last stop closes it too");
    let mut stops = ns[..4].to_vec();
    stops.push(ns[0]);
    assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    assert_eq!(a.line_loop(id), 1, "a circle");
    assert_eq!(tier(&a, ns[0]), 6.0, "opens");
    // S4 and S5 lie on the closing stretch: added between S3 and the closing S0
    let grown: Vec<u32> = a.insert_stop(id, ns[4]).iter().map(|&s| s as u32).collect();
    assert_eq!(grown, vec![ns[0], ns[1], ns[2], ns[3], ns[4], ns[0]]);
    assert_eq!(tier(&a, ns[5]), 1.0, "on the loop's path");
    assert_eq!(a.set_stops(id, &ns[..4]), 0);
    assert_eq!(a.line_loop(id), 0);
    // two stops are enough when the track leads on round (on straight track, going back the
    // way it came makes no loop: the T-797 test above keeps tier 0 there)
    let mut b = api();
    let (w, ns) = super::super::super::tests::ring_fixture();
    b.w = w;
    let id = b.new_line_id();
    assert_eq!(b.set_line_with_trains(id, &ns[..2], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0);
    assert_eq!(b.stop_tiers(id).chunks(3).find(|c| c[0] as u32 == ns[0]).unwrap()[1], 5.0, "the ring leads round");
}

/// T-4374: a loop whose first station is removed closes again on its new first stop (T-3050), as
/// taking the stop out of the line does, instead of turning into an open line.
#[test]
fn removing_a_loops_first_station_keeps_the_loop() {
    let mut a = api();
    let (w, ns) = super::super::super::tests::ring_fixture();
    a.w = w;
    let id = a.new_line_id();
    let mut stops = ns[..4].to_vec();
    stops.push(ns[0]);
    assert_eq!(a.set_line_with_trains(id, &stops, 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_loop(id), crate::track::net::LoopKind::Circle);
    assert_eq!(a.remove_station(ns[0]), 0, "{}", a.issues());
    a.settle_owed();
    assert_eq!(a.w.net.line_stops(id), &[ns[1], ns[2], ns[3], ns[1]]);
    assert_eq!(a.w.net.line_loop(id), crate::track::net::LoopKind::Circle, "still a circle");
    assert!(a.w.net.check_line(id).is_ok());
}

/// T-4374: each problem with a line's stops is its own issue; LineStops is only a line with
/// fewer than two stops and a path to run.
#[test]
fn each_problem_with_a_lines_stops_is_named() {
    use crate::track::net::Issue;
    let mut a = api();
    let (w, ns) = super::super::super::tests::ring_fixture();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &ns[..3], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let net = &a.w.net;
    let path = net.line_path(id).to_vec();
    let check = |stops: &[u32]| net.check_route(id, stops, &path, false).map(|_| ());
    assert_eq!(check(&ns[..3]), Ok(()));
    assert_eq!(check(&[ns[0]]), Err(Issue::LineStops { line: id }));
    assert_eq!(check(&[ns[0], ns[1], ns[1], ns[2]]), Err(Issue::StopTwice { line: id, node: ns[1] }));
    assert_eq!(check(&[ns[0], 99_999, ns[2]]), Err(Issue::StopNotStation { line: id, node: 99_999 }));
    assert_eq!(check(&[ns[2], ns[1], ns[0]]), Err(Issue::StopsOffPath { line: id }), "the path starts at another stop");
    assert_eq!(check(&[ns[0], ns[2], ns[1], ns[2]]), Err(Issue::StopsOffPath { line: id }), "not along the path in order");
}

/// T-4223: a line round part of a ring closes into a circle however long the stretch from its last
/// stop round to its first, stop-less: going back over its own track is no way round.
#[test]
fn a_long_stretch_without_stops_still_closes() {
    use super::super::super::tests::{edge, node, station, world};
    for r in [3000.0, 30000.0] {
        let mut w = world();
        let ns: Vec<u32> = (0..6).map(|i| {
            let a = i as f64 * std::f64::consts::PI / 3.0;
            node(&mut w, r * a.cos(), r * a.sin(), 0)
        }).collect();
        let pi = r / (std::f64::consts::PI / 6.0).cos();
        for i in 0..6 {
            let a = (i as f64 + 0.5) * std::f64::consts::PI / 3.0;
            edge(&mut w, ns[i], ns[(i + 1) % 6], &[(pi * a.cos(), pi * a.sin())], 0).unwrap();
        }
        for &n in &ns[..3] {
            station(&mut w, n, 200);
        }
        let mut a = api();
        a.w = w;
        let id = a.new_line_id();
        assert_eq!(a.set_line_with_trains(id, &ns[..3], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        let tier = |a: &TrackApi, n: u32| a.stop_tiers(id).chunks(3).find(|c| c[0] as u32 == n).unwrap()[1];
        assert_eq!(tier(&a, ns[0]), 5.0, "radius {r}: the first stop closes");
        assert_eq!(tier(&a, ns[2]), 5.0, "radius {r}: the last stop closes");
        let mut stops = ns[..3].to_vec();
        stops.push(ns[0]);
        assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
        assert_eq!(a.line_loop(id), 1, "radius {r}: a circle");
        assert!(a.w.net.path_reversals(a.w.net.line_path(id)).is_empty(), "radius {r}: never turning back");
    }
}

/// T-3042: a junction a looped route passes twice is one mark, at its first pass.
#[test]
fn a_junction_passed_twice_is_one_mark() {
    let mut a = api();
    let (w, [s0, m, _, k]) = super::super::super::tests::balloon_fixture();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[s0, m, k, m, s0], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let marks = a.line_marks(id);
    assert_eq!(marks.len(), 6, "{marks:?}");
    assert_eq!((marks[0], marks[1]), (4.0, 1.0), "the junction, after the second stop");
}

/// T-4255: cached path-tree stop-marker scan matches the uncached scan on open, branching,
/// reversing, and ring networks, both under the default budget and when capped to one line tree.
#[test]
fn stop_tiers_cached_matches_uncached() {
    let mut a = api();
    let route = |a: &mut TrackApi, ends: [f64; 10], pis: &[f64]| assert_eq!(a.add_route(&ends, pis, 2, false), 0, "{}", a.issues());
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9000.0, 0.0, 0.0], &[]);
    let on_main = |a: &TrackApi, x: f64| (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, 0.0).1 < 0.01 && a.w.net.node_y[a.w.net.edge_a[e as usize] as usize] == 0.0 && a.w.net.node_y[a.w.net.edge_b[e as usize] as usize] == 0.0).unwrap() as f64;
    let e1 = on_main(&a, 2000.0);
    route(&mut a, [2.0, e1, 2000.0, 0.0, 0.0, 2.0, e1, 4000.0, 0.0, 0.0], &[3000.0, 800.0, 0.0, 0.0]);
    let e = on_main(&a, 6000.0);
    route(&mut a, [2.0, e, 6000.0, 0.0, 0.0, 0.0, 0.0, 7500.0, 1500.0, 0.0], &[]);
    let e = on_main(&a, 4500.0);
    route(&mut a, [2.0, e, 4500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, -1500.0, 0.0], &[]);
    route(&mut a, [0.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 7000.0, 5000.0, 0.0], &[]);
    let nearest = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
    let station = |a: &mut TrackApi, x: f64, y: f64, name: &str| -> u32 {
        let e = nearest(a, x, y);
        assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{name}: {}", a.issues());
        let n = a.w.net.node_count() as u32;
        (0..n).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).min_by(|&p, &q| {
            let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap()
    };
    let s1 = station(&mut a, 1000.0, 0.0, "A");
    let s2 = station(&mut a, 3000.0, 0.0, "M");
    let s3 = station(&mut a, 5000.0, 0.0, "B");
    let _c = station(&mut a, 8000.0, 0.0, "C");
    let _d = station(&mut a, 300.0, 0.0, "D");
    let _s = station(&mut a, 3000.0, 650.0, "S");
    let _y = station(&mut a, 7200.0, 1300.0, "Y");
    let _z = station(&mut a, 3700.0, -1300.0, "Z");
    let _e = station(&mut a, 6000.0, 5000.0, "E");
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[s1, s2, s3], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let cached = a.stop_tiers(id);
    let uncached = a.stop_tiers_with(id, &mut HopMemo::new());
    assert_eq!(cached, uncached, "cached scan matches uncached on open branching line");
    let bytes_per_tree = (2 * a.w.net.edge_count() + 3 * a.w.net.node_count()) * std::mem::size_of::<u32>();
    let mut capped_memo = HopMemo::for_stop_scan_budget(&a.w.net, a.w.net.line_stops(id), bytes_per_tree * 7);
    let capped = a.stop_tiers_with(id, &mut capped_memo);
    assert_eq!(capped, uncached, "capped tree scan matches uncached when falling back");
    // Ring with 3 stops (tests closing hop + carry_worth with PathTree)
    let mut b = api();
    let (w, ns) = super::super::super::tests::ring_fixture();
    b.w = w;
    let rid = b.new_line_id();
    assert_eq!(b.set_line_with_trains(rid, &ns[..3], 6, 4, 2, 30.0, 180.0, 0, 0, "R".into()), 0, "{}", b.issues());
    assert_eq!(b.stop_tiers(rid), b.stop_tiers_with(rid, &mut HopMemo::new()), "cached scan matches uncached on ring");
}
