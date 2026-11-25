use super::*;

/// What `preview_station` says at (x, y) on the track along y = 0: its issue kinds, and whether
/// `add_station` agrees (undone again).
fn platform_at(a: &mut TrackApi, x: f64, plat: u16) -> String {
    let e = edge_under(a, x, 0.0);
    platform_on(a, e, x, 0.0, plat)
}
/// The same on edge `e` at (x, y): "" when it works, else the issue kinds. The real edit must
/// agree with the preview, and is undone.
fn platform_on(a: &mut TrackApi, e: u32, x: f64, y: f64, plat: u16) -> String {
    let dry = a.preview_station(2, e, x, y, plat);
    let kinds = a.issue_kinds();
    let nodes = live_edges(a).len();
    let real = a.add_station(2, e, x, y, plat, "P".into());
    assert_eq!(dry == 0, real == 0, "at {x}: preview {dry} ({kinds}) real {real} ({})", a.issues());
    if real == 0 {
        assert_eq!(a.undo(), 0);
        assert_eq!(live_edges(a).len(), nodes, "undone");
        return String::new();
    }
    assert_eq!(kinds, a.issue_kinds(), "at {x}: preview and edit give the same reasons");
    kinds
}

/// A line along y = 0 from x = 0 to 3000 on `level`, in `cuts` pieces (joints at those x), and
/// a cross track at x = 1500 on `cross`; a 200 m platform every `step` m across `from..to`.
/// The platform (reach 100, 1 m slack) alerts for the crossing exactly when it is within
/// 101 m of it, and never for being too long.
fn crossing_scan(level: i8, cross: i8, cuts: &[f64], from: f64, to: f64, step: f64) -> Vec<(f64, String)> {
    let mut a = api();
    let mut at = Err((0.0, 0.0, level));
    for &c in cuts {
        at = Ok(piece(&mut a, at, &[], (c, 0.0, level)));
    }
    piece(&mut a, at, &[], (3000.0, 0.0, level));
    piece(&mut a, Err((1500.0, -1000.0, cross)), &[], (1500.0, 1000.0, cross));
    let mut out = vec![];
    let mut x = from;
    while x <= to {
        out.push((x, platform_at(&mut a, x, 200)));
        x += step;
    }
    out
}
fn expect_crossing(res: &[(f64, String)], flat: bool) {
    for (x, k) in res {
        let near = (x - 1500.0).abs() <= 101.0;
        if flat && near && !((x - 1500.0).abs() > 99.0) {
            assert_eq!(k, "PlatformCrossing", "at {x}");
        } else if !flat || (x - 1500.0).abs() >= 103.0 {
            assert_eq!(k, "", "at {x}");
        }
    }
}

#[test]
fn a_platform_alerts_for_a_flat_crossing_within_its_span_and_not_beyond() {
    expect_crossing(&crossing_scan(-1, -1, &[], 1300.0, 1700.0, 2.0), true);
    expect_crossing(&crossing_scan(0, 0, &[], 1300.0, 1700.0, 2.0), true);
}

#[test]
fn a_platform_over_a_grade_separated_crossing_never_alerts() {
    expect_crossing(&crossing_scan(-1, 0, &[], 1300.0, 1700.0, 2.0), false);
    expect_crossing(&crossing_scan(-1, -2, &[], 1300.0, 1700.0, 2.0), false);
}

/// A route of curved segments drawn through their ends (tangent arcs) in Tokyo's frame, a track
/// crossing the fifth segment at 70 degrees on the same level: a 200 m platform every 3 m along
/// the whole route alerts for the crossing exactly when it is within 101 m of it.
#[test]
fn a_platform_along_curved_joined_segments_alerts_only_near_the_crossing() {
    let mut a = TrackApi::new(139.7, 35.7, true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(400.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let mut end = a.route_end();
    a.set_route_tangent(true);
    let mut chain = vec![live_edges(&a)[0]];
    let mut heading = 0.0f64;
    for (k, turn) in [4.0f64, -3.0, 6.0, -5.0, 3.0, -4.0, 5.0, -2.0, 4.0, -3.0].into_iter().enumerate() {
        let off = turn.to_radians();
        let len = 140.0 + 37.0 * (k % 4) as f64;
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + len * (heading + off).cos(), y0 + len * (heading + off).sin());
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 2, false), 0, "{}", a.issues());
        heading += 2.0 * off;
        end = a.route_end();
        chain.push(*live_edges(&a).last().unwrap());
    }
    let lens: Vec<f64> = chain.iter().map(|&e| a.w.net.edge_len[e as usize]).collect();
    let total: f64 = lens.iter().sum();
    // the crossing, a little before the middle of the fifth segment's length
    let start5: f64 = lens[..5].iter().sum();
    let at_cross = start5 + lens[5] * 0.4;
    let (cx, cy, th) = geom::pos_at(a.w.net.edge_pieces(chain[5]), lens[5] * 0.4);
    let (ca, sa) = ((th + 70f64.to_radians()).cos(), (th + 70f64.to_radians()).sin());
    piece(&mut a, Err((cx - 600.0 * ca, cy - 600.0 * sa, 0)), &[], (cx + 600.0 * ca, cy + 600.0 * sa, 0));
    assert_eq!(a.w.net.crossings.len(), 1, "one flat crossing");
    let mut c = 110.0;
    let step = 1.0;
    while c < total - 110.0 {
        let k = (0..chain.len()).rev().find(|&k| lens[..k].iter().sum::<f64>() <= c).unwrap();
        let s = c - lens[..k].iter().sum::<f64>();
        let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(chain[k]), s);
        let d = (c - at_cross).abs();
        if !(99.0..=103.0).contains(&d) {
            let got = platform_on(&mut a, chain[k], x, y, 200);
            assert_eq!(got, if d < 99.0 { "PlatformCrossing" } else { "" }, "at {c} of {total} (edge {k}, s {s}), crossing at {at_cross}");
        }
        c += step;
    }
}

/// Two pieces along y = 0 joined at x = 1000, then a 200 m platform every 10 m from 850 to
/// 1150. `setup` turns the pieces into the case (built, ramped, ...) and returns what to
/// prepare; the answer at each x.
fn joint_scan(setup: &dyn Fn(&mut TrackApi, u32, u32), levels: [i8; 3], tracks: [u8; 2]) -> Vec<(f64, String)> {
    let mut a = api();
    a.set_cash(1e9);
    let mut ends = free_at(0.0, 0.0, levels[0]);
    ends.extend(free_at(1000.0, 0.0, levels[1]));
    assert_eq!(a.add_route(&ends, &[], tracks[0], false), 0, "{}", a.issues());
    let n = a.route_end();
    let e1 = live_edges(&a)[0];
    let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
    ends.extend(free_at(2000.0, 0.0, levels[2]));
    assert_eq!(a.add_route(&ends, &[], tracks[1], false), 0, "{}", a.issues());
    let e2 = *live_edges(&a).last().unwrap();
    setup(&mut a, e1, e2);
    let mut out = vec![];
    let mut x = 850.0;
    while x <= 1150.0 {
        out.push((x, platform_at(&mut a, x, 200)));
        x += 10.0;
    }
    out
}
#[test]
fn a_joint_with_a_ramp_beside_it_is_merged_under_a_platform() {
    // level 0 to level -1 over one piece: the ramp is mid piece, 300 m from the joint at its
    // nearest, and the platform stays clear of it
    never(&joint_scan(&|_, _, _| {}, [0, 0, -1], [2, 2]));
    never(&joint_scan(&|_, _, _| {}, [-1, 0, 0], [2, 2]));
}

/// Joining pieces for a platform leaves the heights as they were: the ramp stays where it was
/// drawn, whichever piece it is on.
#[test]
fn joining_pieces_for_a_platform_keeps_the_ramps_where_they_were() {
    for levels in [[0, 0, -1], [-1, 0, 0], [-1, -1, 0], [0, -1, -1]] {
        let mut a = api();
        let mut ends = free_at(0.0, 0.0, levels[0]);
        ends.extend(free_at(1000.0, 0.0, levels[1]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n = a.route_end();
        let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, levels[2]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let before = heights(&a);
        let e = edge_under(&a, 960.0, 0.0);
        assert_eq!(a.add_station(2, e, 960.0, 0.0, 200, "P".into()), 0, "{levels:?}: {}", a.issues());
        assert!(live_edges(&a).len() == 2, "the joint went, the station split once: {}", live_edges(&a).len());
        same_heights(&before, &heights(&a), &format!("{levels:?}"));
    }
}

/// The heights along y = 0 every 49 m from x = 25.
fn heights(a: &TrackApi) -> Vec<f64> {
    (0..=40)
        .map(|k| {
            let x = 25.0 + 49.0 * k as f64;
            let e = edge_under(a, x, 0.0);
            geom::height_at(a.w.net.edge_vert(e), a.w.net.project(e, x, 0.0).0)
        })
        .collect()
}
fn same_heights(before: &[f64], after: &[f64], what: &str) {
    for (k, (p, q)) in before.iter().zip(after).enumerate() {
        assert!((p - q).abs() < 1e-6, "{what} at x {}: {p} before, {q} after", 25 + 49 * k);
    }
}

/// A station splits a stretch with a ramp in it: the ramp stays where it was drawn, whether
/// the station is before or after it (it used to follow the station, so nudging the station
/// moved the ramp under other track).
#[test]
fn a_station_on_a_stretch_with_a_ramp_leaves_the_ramp_where_it_was() {
    let mut checked = 0;
    for (l0, l1) in [(-1, 0), (0, -1), (-2, 0), (1, -1)] {
        for x in [150.0, 330.0, 700.0, 900.0, 1100.0, 1300.0, 1700.0, 1850.0] {
            let mut a = api();
            let mut ends = free_at(0.0, 0.0, l0);
            ends.extend(free_at(2000.0, 0.0, l1));
            assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
            let before = heights(&a);
            let e = edge_under(&a, x, 0.0);
            let got = a.add_station(2, e, x, 0.0, 60, "P".into());
            if got != 0 {
                // on the ramp (refused), or too near an end
                assert!(a.issue_kinds().contains("SplitPoint") || a.issue_kinds().contains("PlatformNotLevel"), "{l0}/{l1} at {x}: {}", a.issue_kinds());
                continue;
            }
            same_heights(&before, &heights(&a), &format!("{l0}/{l1} station at {x}"));
            checked += 1;
        }
    }
    assert!(checked >= 16, "only {checked} placements were checked");
}

#[test]
fn a_joint_that_cannot_be_merged_says_so() {
    // built beside blueprint, single beside double track (blueprint or built): the platform
    // reaches the joint, which stays, and the answer names it (not "a junction or another station");
    // within 5 m of it the station goes on the joint (T-1964)
    let cases: [(&str, Box<dyn Fn(&mut TrackApi, u32, u32)>, [u8; 2]); 4] = [
        ("built | blueprint", Box::new(|a, e1, _| assert_eq!(a.construct(&[e1], &[]), 0, "{}", a.issues())), [2, 2]),
        ("blueprint | built", Box::new(|a, _, e2| assert_eq!(a.construct(&[e2], &[]), 0, "{}", a.issues())), [2, 2]),
        ("single | double", Box::new(|_, _, _| {}), [1, 2]),
        ("built single | built double", Box::new(|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues())), [1, 2]),
    ];
    for (what, setup, tracks) in cases {
        for (x, k) in joint_scan(&*setup, [0, 0, 0], tracks) {
            let near = (x - 1000.0).abs() < 105.0;
            let on = (x - 1000.0).abs() < 5.0;
            match (near && !on, k.as_str()) {
                (true, "PlatformJoint") | (false, "") => {}
                _ => panic!("{what} at {x}: {k:?}"),
            }
        }
    }
}

/// T-624: two constructed pieces of one kind join under a platform as blueprint ones do.
#[test]
fn constructed_pieces_of_one_kind_are_merged_under_a_platform() {
    for tracks in [[1, 1], [2, 2]] {
        never(&joint_scan(&|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues()), [0, 0, 0], tracks));
    }
    // a ramp in either piece stays where it was
    for levels in [[0, 0, -1], [-1, 0, 0], [-1, -1, 0], [0, -1, -1]] {
        never(&joint_scan(&|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues()), levels, [2, 2]));
    }
}

/// T-624: built double track in two pieces with a running line over both, a station placed
/// across the joint. The pieces become one edge (then split at the station); the line's
/// path, its trains, the money and the track's cost are as they were, and undo restores
/// the two pieces.
#[test]
fn a_station_across_a_built_joint_keeps_the_line_its_trains_and_the_money() {
    for levels in [[0, 0, 0], [0, 0, -1], [-1, 0, 0]] {
        let mut a = api();
        a.set_cash(1e9);
        let mut ends = free_at(0.0, 0.0, levels[0]);
        ends.extend(free_at(1000.0, 0.0, levels[1]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let joint = a.route_end();
        let mut ends = vec![1.0, joint as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, levels[2]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, east) = (end(&a, 0.0), end(&a, 2000.0));
        for (n, x) in [(w, 0.0), (east, 2000.0)] {
            assert_eq!(a.add_station(1, n, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let pieces = live_edges(&a);
        assert_eq!(pieces.len(), 2);
        assert_eq!(a.construct(&pieces, &[w, east]), 0, "{}", a.issues());
        let id = a.new_line_id();
        assert_eq!(a.set_line_with_trains(id, &[w, east], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        assert!(a.w.svc.lines[id as usize].running, "{levels:?}: the line runs");
        a.set_now(40_000.0);
        let snap = |t: &TrackApi| {
            let trips = t.trips(40_000.0 - 1800.0);
            let at: Vec<Vec<f64>> = (0..8).map(|k| t.train_at(id, 0, 0, 40_000.0 - 600.0, 40_000.0 + 90.0 * k as f64)).collect();
            (t.cash, t.built_cost(), t.total_cost(), t.stop_times(id, 0, 0), t.stop_times(id, 1, 0), trips, at, t.line_summary(id))
        };
        let before = snap(&a);
        let before_heights = heights(&a);
        assert!(!before.5.is_empty(), "trains are running");
        let len: f64 = pieces.iter().map(|&e| a.w.net.edge_len[e as usize]).sum();

        let e = edge_under(&a, 960.0, 0.0);
        assert_eq!(a.preview_station(2, e, 960.0, 0.0, 200), 0, "{levels:?}: {}", a.issues());
        assert_eq!(a.add_station(2, e, 960.0, 0.0, 200, "Middle".into()), 0, "{levels:?}: {}", a.issues());
        // (its services worked out after its answer, as the clock worker does before its state: T-4272)
        a.settle_owed();
        assert!(!a.w.net.node_ok(joint), "{levels:?}: the joint went");
        let now = live_edges(&a);
        assert_eq!(now.len(), 2, "joined, then split at the station");
        assert!(now.iter().all(|&e| a.w.net.edge_built[e as usize]), "still constructed");
        assert!((now.iter().map(|&e| a.w.net.edge_len[e as usize]).sum::<f64>() - len).abs() < 1e-6);
        let l = a.w.net.line_path(id).to_vec();
        assert!(l.iter().all(|&p| a.w.net.edge_ok(crate::track::net::path_edge(p))), "the path uses live edges");
        assert!(!a.w.net.line_broken[id as usize] && a.w.svc.lines[id as usize].running);
        assert_eq!(a.w.net.line_stops(id), &[w, east], "stops unchanged");
        let after = snap(&a);
        assert_eq!(before.0, after.0, "{levels:?}: cash");
        assert!((before.1 - after.1).abs() < 1e-9, "{levels:?}: built cost {} {}", before.1, after.1);
        assert_eq!((&before.3, &before.4, &before.5, &before.7), (&after.3, &after.4, &after.5, &after.7), "{levels:?}: timetable");
        for (p, q) in before.6.iter().zip(&after.6) {
            assert!((p[0] - q[0]).hypot(p[1] - q[1]) < 1e-3, "{levels:?}: a train moved: {p:?} {q:?}");
        }
        assert!(a.w.svc.eras_of(id as usize).is_empty() && a.w.svc.shifts[id as usize].is_empty(), "no train was moved or re-timed");
        same_heights(&before_heights, &heights(&a), &format!("{levels:?}"));

        assert_eq!(a.undo(), 0);
        assert_eq!(live_edges(&a), pieces, "{levels:?}: the two pieces are back");
        assert!(a.w.net.node_ok(joint) && pieces.iter().all(|&e| a.w.net.edge_built[e as usize]));
        let undone = snap(&a);
        assert_eq!(before.0, undone.0);
        assert!((before.1 - undone.1).abs() < 1e-9);
        assert_eq!((&before.3, &before.5), (&undone.3, &undone.5));
        same_heights(&before_heights, &heights(&a), &format!("{levels:?} undone"));
    }
}

#[test]
fn a_plain_joint_under_a_platform_is_merged_and_the_crossing_still_answers() {
    // joints at 1000, 1450 and 2000: the crossing at 1500 sits just past the middle one
    let res = crossing_scan(-1, -1, &[1000.0, 1450.0, 2000.0], 800.0, 2200.0, 5.0);
    for (x, k) in &res {
        assert!(!k.contains("TooLong"), "at {x}: {k}");
    }
    expect_crossing(&res, true);
    let res = crossing_scan(-1, 0, &[1000.0, 1450.0, 2000.0], 800.0, 2200.0, 5.0);
    assert!(res.iter().all(|(_, k)| k.is_empty()), "{res:?}");
    // the joint right under the crossing: it is two crossings until the pieces are joined
    let res = crossing_scan(-1, -1, &[1500.0], 1200.0, 1800.0, 5.0);
    assert!(res.iter().all(|(x, k)| !k.contains("TooLong") && !k.contains("Joint") && (k.is_empty() == ((x - 1500.0).abs() > 101.0))), "{res:?}");
}
