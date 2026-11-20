use super::*;

/// A station refused for room names what its platform would reach (T-1870): a loose end, a
/// junction or another station; a station on a junction node is refused as such.
#[test]
fn platform_room_refusals_name_what_is_reached() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    let ends = [2.0, e, 3000.0, 5.0, 0.0, 0.0, 0.0, 5000.0, 2500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    let e = edge_under(&a, 60.0, 0.0);
    assert_ne!(a.add_station(2, e, 60.0, 0.0, 200, "End".into()), 0);
    assert_eq!(a.issue_kinds(), "PlatformTooLong");
    let e = edge_under(&a, 3050.0, 0.0);
    assert_ne!(a.add_station(2, e, 3050.0, 0.0, 200, "Junction".into()), 0);
    assert_eq!(a.issue_kinds(), "PlatformJunction");
    assert_ne!(a.add_station(1, j, 0.0, 0.0, 200, "On it".into()), 0);
    assert_eq!(a.issue_kinds(), "StationJunction");
    let e = edge_under(&a, 1500.0, 0.0);
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 200, "First".into()), 0, "{}", a.issues());
    let e = edge_under(&a, 1650.0, 0.0);
    assert_ne!(a.add_station(2, e, 1650.0, 0.0, 200, "Second".into()), 0);
    assert_eq!(a.issue_kinds(), "PlatformStation");
}

/// T-2978: a point added halfway round a 320 km/h curve and nudged takes the heading that bisects
/// the directions to its neighbours, so the curve keeps most of its speed: 260 of 288 km/h for a
/// 110 m nudge (15 px at zoom 14). A curve is limited by geometry too: the ends stay put, so the
/// bent arc through the nudged point cannot be as wide as the arc was.
#[test]
fn a_nudged_point_in_a_fast_curve_keeps_its_speed() {
    let mut a = api();
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    // a 3 km straight, then a tangent curve from its end 15 degrees off, as the drawing tool lays it
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let joint = a.route_end();
    let off = 15.0f64.to_radians();
    let to = [0.0, 0.0, 3000.0 + 5000.0 * off.cos(), 5000.0 * off.sin(), 0.0];
    let ends: Vec<f64> = [1.0, joint as f64, 0.0, 0.0, 0.0].iter().chain(to.iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let curve = a.w.net.edge_count() as u32 - 1;
    assert_eq!(a.add_point_middle(curve), 0, "{}", a.issues());
    let at = a.nodes_info().chunks(10).find(|c| c[4] == 2.0 && c[1] > 3100.0).expect("the point").to_vec();
    let (n, x0, y0) = (at[0] as u32, at[1], at[2]);
    // the slowest curve in a preview, km/h
    let slowest = |pv: &[f64]| {
        let (mut i, mut v) = (3, f64::INFINITY);
        for _ in 0..pv[2] as usize {
            let k = pv[i] as usize;
            for r in 0..k {
                let row = &pv[i + 1 + 4 * r..i + 5 + 4 * r];
                if row[2] > 0.0 {
                    v = v.min(row[3]);
                }
            }
            i += 1 + 4 * k;
            i += 1 + 3 * pv[i] as usize;
        }
        v
    };
    let before = slowest(&a.preview_move_node(n, x0, y0 + 0.01));
    assert!(before > 285.0, "{before}");
    for (dx, dy) in [(0.0, 110.0), (0.0, -110.0), (110.0, 0.0), (-80.0, 80.0), (0.0, 40.0), (40.0, 40.0)] {
        let pv = a.preview_move_node(n, x0 + dx, y0 + dy);
        assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
        let v = slowest(&pv);
        assert!(v > 0.88 * before, "nudge ({dx}, {dy}): {v:.0} of {before:.0} km/h");
        assert_eq!(a.move_node(n, x0 + dx, y0 + dy), 0, "{}", a.issues());
        assert_eq!(a.undo(), 0);
    }
}

/// Sign changes of curvature along the stretch's edges, in order of x.
fn curvature_flips(a: &TrackApi) -> usize {
    let net = &a.w.net;
    let mut es: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e)).collect();
    es.sort_by(|&p, &q| net.node_x[net.edge_a[p as usize] as usize].total_cmp(&net.node_x[net.edge_a[q as usize] as usize]));
    let mut last = 0.0f64;
    let mut flips = 0;
    for e in es {
        for p in net.edge_pieces(e) {
            if p.k.abs() > 1e-7 {
                if last != 0.0 && p.k.signum() != last {
                    flips += 1;
                }
                last = p.k.signum();
            }
        }
    }
    flips
}

/// Turn-sign changes of the polygon through the network's nodes in order of x: the S the points
/// themselves ask for.
fn polygon_flips(a: &TrackApi) -> usize {
    let mut pts: Vec<(f64, f64)> = a.nodes_info().chunks(10).map(|c| (c[1], c[2])).collect();
    pts.sort_by(|p, q| p.0.total_cmp(&q.0));
    let mut last = 0.0f64;
    let mut flips = 0;
    for w in pts.windows(3) {
        let cross = (w[1].0 - w[0].0) * (w[2].1 - w[1].1) - (w[1].1 - w[0].1) * (w[2].0 - w[1].0);
        if cross.abs() > 1.0 {
            if last != 0.0 && cross.signum() != last {
                flips += 1;
            }
            last = cross.signum();
        }
    }
    flips
}

/// T-2978: points on a straight stretch between two free ends, one dragged sideways by any
/// distance: the track through them bends one way each side, with no S (curvature keeps one
/// sign). The plain neighbours of the dragged point re-aim with it.
#[test]
fn a_dragged_point_among_plain_points_makes_no_s() {
    let mut worst = vec![];
    for (label, fracs, drag) in [("one point", vec![0.5], 0), ("two in a row", vec![0.4, 0.6], 1), ("three in a row", vec![0.3, 0.5, 0.7], 1)] {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        for f in &fracs {
            let at = f * 6000.0;
            let e = (0..a.w.net.edge_count() as u32)
                .find(|&e| {
                    let p = a.w.net.project(e, at, 0.0);
                    a.w.net.edge_ok(e) && p.1.abs() < 1.0 && p.0 > 5.0 && a.w.net.edge_len[e as usize] - p.0 > 5.0
                })
                .unwrap();
            assert_eq!(a.add_point(e, at, 0.0), 0, "{}", a.issues());
        }
        let mid = fracs[drag] * 6000.0;
        let n = a.nodes_info().chunks(10).find(|c| (c[1] - mid).abs() < 1.0).unwrap()[0] as u32;
        for dy in [30.0, 80.0, 150.0, 300.0, -150.0] {
            assert_eq!(a.move_node(n, mid, dy), 0, "{}", a.issues());
            let (flips, asked) = (curvature_flips(&a), polygon_flips(&a));
                        if flips > asked {
                worst.push(format!("{label} drag {dy}: {flips}"));
            }
            assert_eq!(a.undo(), 0);
        }
    }
    assert!(worst.is_empty(), "{worst:?}");
}

/// T-2978: a nudge in the middle of a 320 km/h curve with a plain point either side. The curve
/// is 288 km/h; a 110 m nudge leaves 186 to 223 km/h and a 40 m one 242 (the old rule: the same
/// to within 3 km/h), since the arc through a moved point and fixed ends is tighter.
#[test]
fn a_nudged_point_with_neighbours_in_a_fast_curve_keeps_its_speed() {
    let mut a = api();
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let joint = a.route_end();
    let off = 15.0f64.to_radians();
    let to = [0.0, 0.0, 3000.0 + 5000.0 * off.cos(), 5000.0 * off.sin(), 0.0];
    let ends: Vec<f64> = [1.0, joint as f64, 0.0, 0.0, 0.0].iter().chain(to.iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let curve = a.w.net.edge_count() as u32 - 1;
    // halfway, then halfway along each half: points at a quarter, half and three quarters
    assert_eq!(a.add_point_middle(curve), 0, "{}", a.issues());
    let second = a.w.net.edge_count() as u32 - 1;
    assert_eq!(a.add_point_middle(curve), 0, "{}", a.issues());
    assert_eq!(a.add_point_middle(second), 0, "{}", a.issues());
    let mut pts: Vec<(u32, f64, f64)> = a.nodes_info().chunks(10).filter(|c| c[4] == 2.0 && c[1] > 3100.0).map(|c| (c[0] as u32, c[1], c[2])).collect();
    pts.sort_by(|p, q| p.1.total_cmp(&q.1));
    assert_eq!(pts.len(), 3);
    let (n, x0, y0) = pts[1];
    let slowest = |pv: &[f64]| {
        let (mut i, mut v) = (3, f64::INFINITY);
        for _ in 0..pv[2] as usize {
            let k = pv[i] as usize;
            for r in 0..k {
                let row = &pv[i + 1 + 4 * r..i + 5 + 4 * r];
                if row[2] > 0.0 {
                    v = v.min(row[3]);
                }
            }
            i += 1 + 4 * k;
            i += 1 + 3 * pv[i] as usize;
        }
        v
    };
    let before = slowest(&a.preview_move_node(n, x0, y0 + 0.01));
    let mut worst = f64::INFINITY;
    for (dx, dy) in [(0.0, 110.0), (0.0, -110.0), (110.0, 0.0), (0.0, 40.0)] {
        let pv = a.preview_move_node(n, x0 + dx, y0 + dy);
        assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
        let v = slowest(&pv);
        worst = worst.min(v / before);
    }
    assert!(worst > 0.6, "the slowest nudge kept {:.0}%", worst * 100.0);
}

/// T-3286: a platform lengthens through a node that is only where the track was clicked, joining the
/// pieces under it as a new station's does; a loose end still stops it.
#[test]
fn a_platform_lengthens_past_a_clicked_node() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    for x in [2000.0, 3000.0] {
        let joint = a.route_end();
        let to = [0.0, 0.0, x, 0.0, 0.0];
        let ends: Vec<f64> = [1.0, joint as f64, 0.0, 0.0, 0.0].iter().chain(to.iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    }
    let count = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).count();
    let e = edge_under(&a, 1100.0, 0.0);
    assert_eq!(a.add_station(2, e, 1100.0, 0.0, 60, "S".into()), 0, "{}", a.issues());
    let st = |a: &TrackApi| a.nodes_info().chunks(10).find(|c| c[7] > 0.0).unwrap()[0] as u32;
    let n = st(&a);
    let before = count(&a);
    assert_eq!(a.set_station_props(n, 400, "S".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_platform[n as usize], 400);
    assert!(count(&a) < before, "the pieces under the platform were joined");
    assert_eq!(a.undo(), 0);
    assert_eq!((a.w.net.node_platform[n as usize], count(&a)), (60, before));
    // a loose end stops it: 100 m from the start, 400 m needs 200 each way
    let e = edge_under(&a, 100.0, 0.0);
    assert_eq!(a.add_station(2, e, 100.0, 0.0, 60, "E".into()), 0, "{}", a.issues());
    let m = a.nodes_info().chunks(10).find(|c| c[7] > 0.0 && c[1] < 500.0).unwrap()[0] as u32;
    assert_ne!(a.set_station_props(m, 400, "E".into()), 0);
    assert_eq!(a.issue_kinds(), "PlatformTooLong");
}

/// T-3288: the complex's rename sets a name and leaves the flag as it was; a typed rename flags it.
#[test]
fn a_complex_rename_does_not_flag_the_station() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let e = edge_under(&a, 1500.0, 0.0);
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 60, "Old".into()), 0, "{}", a.issues());
    let n = a.nodes_info().chunks(10).find(|c| c[7] > 0.0).unwrap()[0] as u32;
    assert_eq!(a.set_station_name(n, "New".into()), 0, "{}", a.issues());
    assert_eq!((a.node_name(n), a.named_stations()), ("New".to_string(), vec![]));
    // T-3750: its own name typed over the complex's name it shows flags it, the name as it was
    assert_eq!(a.type_station_name(n, "New".into()), 0);
    assert_eq!((a.node_name(n), a.named_stations()), ("New".to_string(), vec![n]));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.named_stations(), Vec::<u32>::new());
    assert_eq!(a.set_station_props(n, 60, "Typed".into()), 0);
    assert_eq!(a.named_stations(), vec![n]);
    assert_eq!(a.set_station_name(n, "Again".into()), 0);
    assert_eq!(a.named_stations(), vec![n]);
}

/// T-3412: a station placed within a split's 5 m of a bare end of track goes on the end; at a junction it is still refused.
#[test]
fn a_station_near_a_bare_end_goes_on_the_end_and_a_junction_still_refuses() {
    for (x, platform) in [(2.0, 200u16), (2998.0, 200), (3.0, 60), (2997.0, 100)] {
        let mut a = api();
        let mut ends = free_at(0.0, 0.0, 0);
        ends.extend(free_at(3000.0, 0.0, 0));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let e = edge_under(&a, 1500.0, 0.0);
        let before = a.w.net.edge_count();
        assert_eq!(a.add_station(2, e, x, 0.0, platform, "End".into()), 0, "at {x}: {}", a.issues());
        assert_eq!(a.w.net.edge_count(), before, "at {x}: no split, the end node is the station");
        let n = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && a.w.net.node_platform[n as usize] > 0).expect("a station");
        assert_eq!(a.w.net.node_ports[n as usize].n, 1, "at {x}: on the bare end");
        assert_eq!(a.undo(), 0);
        assert!(!(0..a.w.net.node_count() as u32).any(|n| a.w.net.node_ok(n) && a.w.net.node_platform[n as usize] > 0));
    }
    // a junction: three pieces meet, a station within 5 m of it is refused as before
    let mut a = api();
    let mut ends = free_at(0.0, 0.0, 0);
    ends.extend(free_at(1000.0, 0.0, 0));
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let j = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - 1000.0).abs() < 1.0 && a.w.net.node_y[n as usize].abs() < 1.0).unwrap();
    for to in [(2000.0, 0.0), (1000.0, 1000.0)] {
        let mut e2 = vec![1.0, j as f64, 0.0, 0.0, 0.0];
        e2.extend(free_at(to.0, to.1, 0));
        assert_eq!(a.add_route(&e2, &[], 2, false), 0, "{}", a.issues());
    }
    assert_eq!(a.w.net.node_ports[j as usize].n, 3);
    let e = edge_under(&a, 500.0, 0.0);
    assert_ne!(a.add_station(2, e, 998.0, 0.0, 100, "J".into()), 0);
    assert!(a.issue_kinds().contains("SplitPoint"), "{}", a.issue_kinds());
}

/// T-3690: a route ending on track within a split's 5 m of a piece's end goes where a click could put it: on a
/// plain joint, on a bare end it carries on past, else split 5 m in (a junction, a bare end it heads back from).
#[test]
fn a_route_end_near_a_piece_end_goes_on_the_joint_or_slides_in() {
    let node_at = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 0.5 && (a.w.net.node_y[n as usize] - y).abs() < 0.5);
    let on = |a: &TrackApi, x: f64| vec![2.0, edge_under(a, x, 0.0) as f64, x, 0.0, 0.0];
    let route = |a: &mut TrackApi, from: Vec<f64>, to: Vec<f64>| {
        let ends: Vec<f64> = from.into_iter().chain(to).collect();
        a.add_route(&ends, &[], 2, false)
    };
    let mut a = api();
    let end = piece(&mut a, Err((0.0, 0.0, 0)), &[], (1000.0, 0.0, 0));
    piece(&mut a, Ok(end), &[], (2000.0, 0.0, 0));
    let joint = node_at(&a, 1000.0, 0.0).unwrap();
    assert_eq!(a.w.net.node_ports[joint as usize].n, 2);
    // a junction at 500
    assert_eq!({ let p = on(&a, 500.0); route(&mut a, p, free_at(900.0, -300.0, 0)) }, 0, "{}", a.issues());
    let junction = node_at(&a, 500.0, 0.0).unwrap();
    let live = |a: &TrackApi| (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).count();
    let nodes = live(&a);

    // 3 m from the joint, from either piece and as either end: the joint is the junction
    for (x, rev) in [(997.0, false), (1003.0, false), (997.0, true)] {
        let (p, q) = (on(&a, x), free_at(x + 300.0, 200.0, 0));
        assert_eq!(if rev { route(&mut a, q, p) } else { route(&mut a, p, q) }, 0, "at {x}: {}", a.issues());
        assert_eq!(a.w.net.node_ports[joint as usize].n, 3, "at {x}");
        assert!(node_at(&a, x, 0.0).is_none());
        assert_eq!(a.undo(), 0);
    }
    // 3 m from the junction: a new junction 5 m from it, the old one as it was
    assert_eq!({ let p = on(&a, 503.0); route(&mut a, p, free_at(800.0, 250.0, 0)) }, 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[junction as usize].n, 3);
    assert!(node_at(&a, 505.1, 0.0).is_some());
    assert_eq!(a.undo(), 0);
    // 3 m from the bare end: heading back along the track, a junction 5 m in; on past the end, from the end
    assert_eq!({ let p = on(&a, 3.0); route(&mut a, p, free_at(300.0, 150.0, 0)) }, 0, "{}", a.issues());
    assert!(node_at(&a, 5.1, 0.0).is_some());
    assert_eq!(a.undo(), 0);
    assert_eq!({ let p = on(&a, 3.0); route(&mut a, p, free_at(-300.0, 50.0, 0)) }, 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[node_at(&a, 0.0, 0.0).unwrap() as usize].n, 2);
    assert_eq!(a.undo(), 0);
    assert_eq!(live(&a), nodes);
}

#[test]
fn a_branch_off_a_ramp_says_ramp() {
    let mut a = api();
    piece(&mut a, Err((0.0, 0.0, 0)), &[], (1000.0, 0.0, 1));
    let e = edge_under(&a, 500.0, 0.0);
    assert!(crate::track::geom::on_ramp(a.w.net.edge_vert(e), 499.0, 501.0));
    let ends: Vec<f64> = vec![2.0, e as f64, 500.0, 0.0, 0.0].into_iter().chain(free_at(700.0, 250.0, 0)).collect();
    assert_ne!(a.add_route(&ends, &[], 2, false), 0);
    assert!(a.issue_kinds().contains("SplitRamp"), "{}", a.issue_kinds());
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 0.0);
    assert!(a.issue_kinds().contains("SplitRamp"), "{}", a.issue_kinds());
}

#[test]
fn a_route_ending_beside_a_bare_end_from_the_tracks_side_branches_in() {
    let node_at = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 0.5 && (a.w.net.node_y[n as usize] - y).abs() < 0.5);
    let mut a = api();
    piece(&mut a, Err((0.0, 0.0, 0)), &[], (1000.0, 0.0, 0));
    let end = node_at(&a, 0.0, 0.0).unwrap();
    // from beside the track, 100 m along it: the end would have to be turned back to
    let ends: Vec<f64> = free_at(100.0, 70.0, 0).into_iter().chain([1.0, end as f64, 0.0, 0.0, 0.0]).collect();
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 1.0, "{}", a.issue_kinds());
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issue_kinds());
    assert!(node_at(&a, 5.1, 0.0).is_some());
    assert_eq!(a.w.net.node_ports[end as usize].n, 1);
    // from beyond the end it joins the end itself, as before
    assert_eq!(a.undo(), 0);
    let ends: Vec<f64> = free_at(-100.0, 70.0, 0).into_iter().chain([1.0, end as f64, 0.0, 0.0, 0.0]).collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issue_kinds());
    assert_eq!(a.w.net.node_ports[end as usize].n, 2);
}
