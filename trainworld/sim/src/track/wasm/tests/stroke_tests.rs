use super::*;

#[test]
fn a_stroke_is_one_piece_per_corner() {
    let mut a = api();
    let (ends, pis) = stroke(&|_| 0);
    let pv = a.preview_route(&ends, &pis, 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert!(live_edges(&a).is_empty(), "a preview builds nothing");
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let (b, e) = one_edge(&pis);
    // four corners, four pieces, joined at three plain nodes, each corner in a piece of its own
    let edges = live_edges(&a);
    assert_eq!(edges.len(), 4);
    let joints: Vec<_> = a.nodes_info().chunks(10).filter(|c| c[4] == 2.0).map(|c| c[0] as u32).collect();
    assert_eq!(joints.len(), 3);
    for &p in &edges {
        let corners = a.edge_pis(p);
        assert_eq!(corners.len(), 9, "one corner");
        assert_eq!(corners[3], 0.0, "auto radius");
    }
    let whole = b.edge_pis(e);
    let used: Vec<f64> = edges.iter().map(|&p| a.edge_pis(p)[5]).collect();
    for k in 0..4 {
        assert!((used[k] - whole[5 + 8 * k]).abs() < 0.01, "corner {k}: {} against {}", used[k], whole[5 + 8 * k]);
    }
    // the very same track, price and preview
    same_track(&a, &b, e, 0.005);
    assert!((a.total_cost() - b.total_cost()).abs() < 1e-6, "{} against {}", a.total_cost(), b.total_cost());
    assert!((pv[1] - b.total_cost()).abs() < 1e-6);
    // one undo takes the whole stroke away, one redo brings it back
    assert_eq!(a.undo(), 0);
    assert!(live_edges(&a).is_empty() && a.nodes_info().is_empty());
    assert_eq!(a.redo(), 0);
    assert_eq!(live_edges(&a), edges);
    // saves keep the pieces
    let mut c = api();
    assert!(c.load(&a.save()));
    assert_eq!(c.edges_info(), a.edges_info());
    // removing a piece in the middle leaves its neighbours, ending where it was
    assert_eq!(a.delete_edge(edges[1]), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 3);
    for &j in &joints[..2] {
        assert_eq!(a.nodes_info().chunks(10).find(|c| c[0] as u32 == j).unwrap()[4], 1.0, "a track end now");
    }
    // a leg that changes level is not cut: its ramp is centred between its corners
    let mut d = api();
    let (ends, pis) = stroke(&|k| if k >= 2 { 1 } else { 0 });
    assert_eq!(d.add_route(&ends, &pis, 2, false), 0, "{}", d.issues());
    assert_eq!(live_edges(&d).len(), 3);
    let (_, pis_flat) = stroke(&|k| if k >= 2 { 1 } else { 0 });
    let mut e2 = api();
    let (n0, n1, x) = (e2.new_node_id(), e2.new_node_id(), e2.new_edge_id());
    assert_eq!(e2.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(e2.set_node(n1, 7600.0, 2400.0, 0, false), 0);
    assert_eq!(e2.set_edge(x, n0, n1, 2, &pis_flat), 0, "{}", e2.issues());
    same_track(&d, &e2, x, 0.005);
    assert!((d.total_cost() - e2.total_cost()).abs() < 1e-6);
}

/// T-244: a line over a stroke in pieces runs exactly as over the stroke in one edge: the
/// joints add no speed limit, no delay and no cost.
#[test]
fn pieces_run_like_the_whole_stroke() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let (mut b, _) = one_edge(&pis);
    let mut times = vec![];
    for t in [&mut a, &mut b] {
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, e) = (end(t, 0.0), end(t, 7600.0));
        assert_eq!(t.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
        assert_eq!(t.add_station(1, e, 0.0, 0.0, 200, "East".into()), 0);
        let id = t.new_line_id();
        assert_eq!(t.set_line_with_trains(id, &[w, e], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", t.issues());
        let quote = t.blueprint_cost();
        assert_eq!(t.construct_all(), 0, "{}", t.issues());
        times.push((t.stop_times(id, 0, 0), t.stop_times(id, 1, 0), t.line_summary(id), t.lines_info()[9], quote, t.markers(0).len()));
    }
    let (p, q) = (&times[0], &times[1]);
    for k in 0..2 {
        let (x, y) = if k == 0 { (&p.0, &q.0) } else { (&p.1, &q.1) };
        assert_eq!(x.len(), y.len());
        for (s, t) in x.iter().zip(y) {
            assert!((s - t).abs() < 1e-3, "run {k}: {s} s against {t} s");
        }
    }
    for (s, t) in p.2.iter().zip(&q.2) {
        assert!((s - t).abs() < 1e-3, "summary {:?} against {:?}", p.2, q.2);
    }
    assert!((p.3 - q.3).abs() < 0.01, "length {} against {}", p.3, q.3);
    assert!((p.4 - q.4).abs() < 1e-6, "price {} against {}", p.4, q.4);
    assert_eq!(p.5, q.5, "capacity markers");
    // one track section for the line, as over one edge
    let sections = |t: &TrackApi| t.w.svc.cap.res.iter().filter(|r| r.kind == super::super::super::capacity::ResKind::Section).count();
    assert_eq!(sections(&a), sections(&b));
}

/// T-258: a station can go anywhere along a curve (partway round, at its ends), on track drawn
/// from an existing track end (a lead-in curve, then the corner's curve) and on plain track,
/// and the split leaves the track, its run time and its price as they were.
#[test]
fn stations_go_anywhere_on_a_curve() {
    for (from_track, corner, radius) in [(true, (1500.0, 300.0), 0.0), (false, (1500.0, 300.0), 0.0), (true, (900.0, 700.0), 0.0), (false, (700.0, -900.0), 0.0), (true, (1500.0, 300.0), 150.0), (false, (1500.0, 1200.0), 400.0), (true, (1200.0, -500.0), 120.0)] {
        let mut a = api();
        let mut start = [0.0, 0.0, 0.0, 0.0, 0.0];
        if from_track {
            let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
            assert_eq!(a.add_route(&lead, &[], 2, false), 0);
            let n = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
            start = [1.0, n, 0.0, 0.0, 0.0];
        }
        let ends: Vec<f64> = start.iter().chain(free(3000.0, -600.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[corner.0, corner.1, radius, 0.0], 2, false), 0, "{}", a.issues());
        let e = *live_edges(&a).last().unwrap();
        let len = a.w.net.edge_len[e as usize];
        let pieces: Vec<_> = a.w.net.edge_pieces(e).to_vec();
        let (cost, before) = (a.total_cost(), a.edge_render(e));
        let mut tried = 0;
        let mut failed = vec![];
        for p in pieces.iter().filter(|p| p.k != 0.0) {
            for f in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
                let s = (p.s0 + f * p.len).clamp(110.0, len - 110.0);
                let (x, y, _) = geom::pos_at(&pieces, s);
                tried += 1;
                if a.add_station(2, e, x, y, 200, "S".into()) != 0 {
                    failed.push((s, a.issues(), format!("{:?}", a.w.net.split_plan(e, a.w.net.project(e, x, y).0)), a.w.net.edge_data(e)));
                    continue;
                }
                // the same track and price, and an undo puts it back exactly
                for q in before.chunks(3) {
                    let off = live_edges(&a).iter().map(|&k| a.w.net.project(k, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
                    assert!(off < 0.01, "split at {s}: {off} m off");
                }
                let station = 75.0 * (0.4 + 0.6 * 200.0 / 200.0) * 1.0;
                assert!((a.total_cost() - cost - station).abs() < 0.05, "split at {s}: {} against {}", a.total_cost() - station, cost);
                assert_eq!(a.undo(), 0);
                assert_eq!(a.edge_render(e), before);
            }
        }
        assert!(tried > 8);
        assert!(failed.is_empty(), "{from_track} {corner:?} {radius}: {failed:?}");
    }
}

/// T-244: a station placed next to a joint joins the two pieces again first, so the platform
/// has room; the track and the lines over it are as they were.
#[test]
fn a_station_next_to_a_joint_joins_the_pieces() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[w, east], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let t0 = a.stop_times(id, 0, 0);
    let edges = live_edges(&a);
    // 30 m before the second joint, on the second piece
    let j = a.w.net.edge_b[edges[1] as usize];
    let len = a.w.net.edge_len[edges[1] as usize];
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edges[1]), len - 30.0);
    assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
    assert!(!a.w.net.node_ok(j), "the joint went");
    assert_eq!(live_edges(&a).len(), 4, "two pieces joined, then split at the station");
    let (b, e) = one_edge(&pis);
    same_track(&a, &b, e, 0.005);
    assert!(!a.w.net.line_broken[id as usize]);
    let t1 = a.stop_times(id, 0, 0);
    assert!((t0.last().unwrap() - t1.last().unwrap()).abs() < 1e-3, "{t0:?} {t1:?}");
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a), edges);
    assert!(a.w.net.node_ok(j));
    // after reversing the line's path: still fine
    let id2 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id2, &[east, w], 6, 4, 2, 30.0, 180.0, 0, 0, "R".into()), 0);
    assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[id2 as usize] && !a.w.net.line_broken[id as usize]);
}

/// T-418: a straight from (0, 0) east to (1000, 0), tangent drawing on, and its east end.
fn tangent_start() -> (TrackApi, u32) {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let east = a.route_end();
    assert!((a.w.net.node_x[east as usize] - 1000.0).abs() < 1e-6);
    a.set_route_tangent(true);
    (a, east)
}
/// T-442: the same straight in quad track, and its drawn pair's east end.
fn quad_start() -> (TrackApi, u32) {
    let mut a = api();
    a.set_route_tangent(true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0);
    let end = a.route_end();
    (a, end)
}
/// T-471: a click past the cap: the capped curve from the start, its circle's, then one
/// straight to the click.
fn capped_then_straight(f: &geom::Fit, arc: &[f64], start: (f64, f64), what: &str) {
    let curves: Vec<_> = f.pieces.iter().filter(|p| p.k != 0.0).collect();
    assert!(curves.iter().all(|p| (p.radius() - R_CAP).abs() < 1.0), "{what}: every curve capped");
    assert!(f.pieces[0].k != 0.0 || f.pieces[0].len < 0.05, "{what}: the curve starts at the start, not after {}", f.pieces[0].len);
    let straights: Vec<_> = f.pieces.iter().filter(|p| p.k == 0.0 && p.len > 1.0).collect();
    assert_eq!(straights.len(), 1, "{what}: one straight");
    assert_eq!(f.pieces.last().unwrap().k, 0.0, "{what}: the straight last");
    assert!((arc[2] - R_CAP).abs() < 1e-6 && ((arc[0] - start.0).hypot(arc[1] - start.1) - R_CAP).abs() < 0.5, "{what}: circle {arc:?}");
}

/// T-418: from a track end, a click d away at a off the heading is one arc of radius
/// d / (2 sin a) ending at the click on heading 2a; one wider than the cap is the capped curve
/// then a straight to the click (T-471). The circle is the curve's.
#[test]
fn a_tangent_segment_is_one_arc_through_the_click() {
    let (mut a, east) = tangent_start();
    for (d, deg) in [(500.0, 30.0), (800.0, -50.0), (300.0, 12.0), (2000.0, 70.0), (500.0, 2.0)] {
        let off: f64 = (deg as f64).to_radians();
        let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
        let pv = a.preview_route(&from_node(east, x, y), &[], 2, false);
        assert_eq!(pv[0], 1.0, "{d} m at {deg}: {}", a.issues());
        let f = a.w.net.fit(&a.route_vertices);
        let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
        assert!((ex - x).hypot(ey - y) < 0.01, "{d} m at {deg}: ends at {ex}, {ey}");
        let want = d / (2.0 * off.abs().sin());
        let arc = a.route_arc();
        let (cx, cy) = (arc[0], arc[1]);
        if want <= R_CAP {
            assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{d} m at {deg}: leaves on {eh}");
            assert!((f.radius[1] - want).abs() < 0.01 * want, "{d} m at {deg}: radius {} against {want}", f.radius[1]);
            assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{d} m at {deg}: one arc, no straight");
            assert!(((cx - 1000.0).hypot(cy) - want).abs() < 0.5 && ((cx - x).hypot(cy - y) - want).abs() < 0.5, "{d} m at {deg}: circle {arc:?}");
        } else {
            capped_then_straight(&f, &arc, (1000.0, 0.0), &format!("{d} m at {deg}"));
        }
    }
    // placed, the end node comes back, and the next segment leaves along this one
    let off: f64 = 30f64.to_radians();
    let (x, y) = (1000.0 + 500.0 * off.cos(), 500.0 * off.sin());
    assert_eq!(a.add_route(&from_node(east, x, y), &[], 2, false), 0, "{}", a.issues());
    let joint = a.route_end();
    assert!((a.w.net.node_x[joint as usize] - x).abs() < 0.01 && (a.w.net.node_y[joint as usize] - y).abs() < 0.01);
    let (x2, y2) = (x + 400.0 * (2.0 * off).cos(), y + 400.0 * (2.0 * off).sin());
    assert_eq!(a.add_route(&from_node(joint, x2, y2), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[joint as usize].n, 2, "a plain joint");
    assert!(a.route_arc().is_empty(), "straight on: no curve, no circle");
    // the first segment's corner dragged sideways: the joint after it follows
    let e = live_edges(&a).into_iter().find(|&e| a.w.net.edge_b[e as usize] == joint).unwrap();
    let pis = a.edge_pis(e);
    assert_eq!(pis.len(), 9, "one corner");
    assert_eq!(a.set_edge_pis(e, &[pis[1] - 5.0, pis[2] + 10.0, 0.0, pis[4]]), 0, "{}", a.issues());
    assert!((a.w.net.node_x[joint as usize] - x).hypot(a.w.net.node_y[joint as usize] - y) > 1.0, "the joint moved");
    // with tangent drawing off, the same click gives the old curve then straight
    a.set_route_tangent(false);
    let pv = a.preview_route(&from_node(east, x, -y), &[], 2, false);
    assert_eq!(pv[0], 1.0);
    let f = a.w.net.fit(&a.route_vertices);
    assert!(f.radius[1] < 450.0, "old lead-in: {}", f.radius[1]);
}

/// T-817: the radius each design speed lays curves out for, and 160 is the old cap.
#[test]
fn design_radius_per_speed() {
    assert_eq!(design_radius(160.0), R_CAP);
    for (kmh, want) in [(160.0, 1796.0), (200.0, 2806.0), (250.0, 4385.0), (320.0, 7183.0)] {
        let r = design_radius(kmh);
        assert_eq!(r, want, "{kmh} km/h");
        assert!((A_LAT * r).sqrt() >= kmh / 3.6 && (A_LAT * (r - 1.0)).sqrt() < kmh / 3.6, "{kmh} km/h: {r} is the smallest");
    }
    assert_eq!(DESIGN_SPEEDS[0], 160.0);
}

/// T-817: at 160 the drawing tool sets no radius and lays every shape as before; at a higher
/// design speed a click past 1,796 m is one wider arc through it (radius set, so it keeps its
/// shape), and past the design radius the capped curve then a straight.
#[test]
fn design_speed_lays_wider_curves() {
    let clicks = [(500.0, 30.0), (5000.0, 40.0), (9000.0, -100.0), (20000.0, 30.0), (3000.0, 8.0)];
    for tracks in [2u8, 4] {
        // 160, named or by default, draws exactly what the default does
        let (mut plain, east) = if tracks == 4 { quad_start() } else { tangent_start() };
        let (mut at160, _) = if tracks == 4 { quad_start() } else { tangent_start() };
        at160.set_design_speed(160.0);
        for (d, deg) in clicks {
            let off: f64 = (deg as f64).to_radians();
            let ends = from_node(east, 1000.0 + d * off.cos(), d * off.sin());
            assert_eq!(plain.preview_route(&ends, &[], tracks, false), at160.preview_route(&ends, &[], tracks, false), "{tracks} tracks, {d} m at {deg}");
            assert_eq!(plain.route_vertices, at160.route_vertices);
            assert!(plain.route_vertices.iter().all(|p| p.radius == 0.0), "{tracks} tracks, {d} m at {deg}: no radius set at 160");
        }
        for kmh in [200.0, 250.0, 320.0] {
            let cap = design_radius(kmh);
            for (d, deg) in clicks {
                let (mut a, east) = if tracks == 4 { quad_start() } else { tangent_start() };
                a.set_design_speed(kmh);
                let what = format!("{tracks} tracks at {kmh}, {d} m at {deg}");
                let off: f64 = (deg as f64).to_radians();
                let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
                let ends = from_node(east, x, y);
                assert_eq!(a.preview_route(&ends, &[], tracks, false)[0], 1.0, "{what}: {}", a.issue_kinds());
                let f = a.w.net.fit(&a.route_vertices);
                let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
                assert!((ex - x).hypot(ey - y) < 0.01, "{what}: ends at {ex}, {ey}");
                let want = d / (2.0 * off.abs().sin());
                let arc = a.route_arc();
                let curves: Vec<_> = f.pieces.iter().filter(|p| p.k != 0.0).collect();
                if want <= cap {
                    assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{what}: leaves on {eh}");
                    assert!(curves.iter().all(|p| (p.radius() - want).abs() < 0.01), "{what}: one radius, {want}");
                    assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{what}: one arc, no straight");
                    assert!((arc[2] - want).abs() < 0.01, "{what}: circle {arc:?}");
                } else {
                    assert!(curves.iter().all(|p| (p.radius() - cap).abs() < 0.01), "{what}: every curve at the design radius");
                    assert_eq!(f.pieces.last().unwrap().k, 0.0, "{what}: then a straight");
                    assert!((arc[2] - cap).abs() < 1e-6, "{what}: circle {arc:?}");
                }
                // placed, every piece keeps the shape (the radius set where it is over 1,796 m)
                let before = live_edges(&a);
                assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{what}: {}", a.issues());
                if tracks == 2 && want.min(cap) > R_CAP {
                    for e in live_edges(&a).into_iter().filter(|e| !before.contains(e)) {
                        for p in a.w.net.edge_pieces(e).iter().filter(|p| p.k != 0.0) {
                            assert!((p.radius() - want.min(cap)).abs() < 0.01, "{what}: built at {}", p.radius());
                        }
                    }
                }
            }
        }
    }
}

/// T-817: a segment ending on track at a higher design speed: its lead-in curves may be wider
/// than 1,796 m, still starting at the node.
#[test]
fn design_speed_widens_the_lead_in() {
    let (mut a, east) = tangent_start();
    a.set_design_speed(320.0);
    // from far off to the east end, arriving well off its heading
    let ends: Vec<f64> = free(20000.0, 6000.0).iter().copied().chain([1.0, east as f64, 0.0, 0.0, 0.0]).collect();
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 1.0, "{}", a.issue_kinds());
    let f = a.w.net.fit(&a.route_vertices);
    let r = f.radius.iter().copied().fold(0.0, f64::max);
    assert!(r > R_CAP && r <= 7183.0 + 1e-6, "lead-in radius {r}");
    let last = f.pieces.last().unwrap();
    assert!(last.k != 0.0 || last.len < 0.05, "the curve reaches the node");
    a.set_design_speed(160.0);
    a.preview_route(&ends, &[], 2, false);
    assert!(a.w.net.fit(&a.route_vertices).radius.iter().all(|&r| r <= R_CAP + 1e-6));
}

/// T-817 b1: high-speed track is drawn as double or quad track, costs 1.5x double track as
/// previewed and quoted (its own quote rows), and survives a save.
#[test]
fn high_speed_track_costs_half_again_and_is_saved() {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    let mut plain = api();
    assert_eq!(plain.add_route(&ends, &[], 2, false), 0, "{}", plain.issues());
    let mut a = api();
    a.set_route_hsr(true);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let hsr = a.hsr_edges();
    assert!(!hsr.is_empty() && hsr == live_edges(&a));
    let rows_ok = |a: &mut TrackApi| a.blueprint_cost_items().chunks_exact(7).filter(|r| r[0] == 0.0).all(|r| r[2] == HSR_ROW as f64 || r[2] == (HSR_ROW + SINGLE_ROW_OFFSET) as f64);
    assert!((a.blueprint_cost() - HSR_COST * plain.blueprint_cost()).abs() < 1e-6, "{} against {}", a.blueprint_cost(), plain.blueprint_cost());
    assert!((pv[1] - a.blueprint_cost()).abs() < 1e-6, "previewed {} built {}", pv[1], a.blueprint_cost());
    // the preview's multiplier is level and water only, as for single track
    assert!((pv[4] - plain.preview_route(&ends, &[], 2, false)[4]).abs() < 1e-9);
    let rows = a.blueprint_cost_items();
    assert!(rows.chunks_exact(7).filter(|r| r[0] == 0.0).all(|r| r[2] == HSR_ROW as f64));
    assert!((rows.chunks_exact(7).map(|r| r[6]).sum::<f64>() - a.blueprint_cost()).abs() < 1e-6);
    // T-3352: single track drawn with the setting on is high-speed too; quad is two high-speed doubles (T-3126)
    let e: Vec<f64> = free(0.0, 2000.0).iter().chain(free(3000.0, 2000.0).iter()).copied().collect();
    assert_eq!(a.add_route(&e, &[], 1, false), 0, "{}", a.issues());
    assert!(a.hsr_edges().len() > hsr.len());
    assert!(rows_ok(&mut a));
    let e: Vec<f64> = free(0.0, 4000.0).iter().chain(free(3000.0, 4000.0).iter()).copied().collect();
    assert_eq!(a.add_route(&e, &[], 4, false), 0, "{}", a.issues());
    assert!(a.hsr_edges().len() >= hsr.len() + 3);
    // saved with flag 64 and loaded as it was; a save without it loads with none
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.hsr_edges().len(), a.hsr_edges().len());
    assert_eq!(b.save(), bytes);
    assert!((b.blueprint_cost() - a.blueprint_cost()).abs() < 1e-6);
    let mut c = api();
    assert!(c.load(&plain.save()));
    assert!(c.hsr_edges().is_empty());
    // high-speed track doubles up into high-speed quad (T-3166), priced as a second high-speed double
    let before = (a.hsr_edges().len(), a.blueprint_cost());
    let price = a.preview_double_up(hsr[0]);
    assert!(price > 0.0, "{}", a.issue_kinds());
    assert_eq!(a.double_up(hsr[0]), 0, "{}", a.issues());
    assert!(a.hsr_edges().len() > before.0, "the pair is high speed too");
    assert!((a.blueprint_cost() - before.1 - price).abs() < 1e-6);
    assert!(a.hsr_edges().iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::HighSpeed));
    assert_ne!(a.double_up(hsr[0]), 0);
    assert!(a.issue_kinds().contains("Alongside"), "{}", a.issue_kinds());
}

/// T-817 b1: a split keeps high-speed track high speed; two high-speed pieces join under a
/// platform as double track does, but never with ordinary track, which still meets it at a
/// plain node.
#[test]
fn high_speed_track_survives_split_and_join() {
    for second in [true, false] {
        let mut a = api();
        a.set_route_hsr(true);
        let mut ends = free_at(0.0, 0.0, 0);
        ends.extend(free_at(1000.0, 0.0, 0));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n = a.route_end();
        a.set_route_hsr(second);
        let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, 0));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "ordinary track joins at a plain node: {}", a.issues());
        assert_eq!(a.hsr_edges().len(), if second { 2 } else { 1 });
        let e = edge_under(&a, 960.0, 0.0);
        let r = a.add_station(2, e, 960.0, 0.0, 200, "P".into());
        if second {
            assert_eq!(r, 0, "{}", a.issues());
            assert_eq!(live_edges(&a).len(), 2, "joined, then split at the station");
            assert_eq!(a.hsr_edges(), live_edges(&a), "both halves high speed");
        } else {
            assert_ne!(r, 0, "high-speed and ordinary track are not joined");
            assert!(a.issue_kinds().contains("PlatformJoint"), "{}", a.issue_kinds());
            // a station clear of the joint splits the high-speed piece into two high-speed halves
            let e = edge_under(&a, 400.0, 0.0);
            assert_eq!(a.add_station(2, e, 400.0, 0.0, 200, "Q".into()), 0, "{}", a.issues());
            assert_eq!(a.hsr_edges().len(), 2);
            assert_eq!(live_edges(&a).len(), 3);
        }
    }
}

/// T-442: a click beyond a quarter turn is still one arc of radius d / (2 sin a) through it,
/// up to 350 degrees, split into corners of at most 90 degrees; on quad track the express pair
/// is concentric all the way round; a level change finds room on a hairpin.
#[test]
fn a_tangent_segment_turns_past_a_half_turn() {
    for tracks in [2u8, 4] {
        for (d, deg) in [(400.0, 84.0), (400.0, -100.0), (300.0, 150.0), (200.0, -170.0), (60.0, 174.0), (4000.0, 120.0), (3600.0, 170.0), (9000.0, -100.0), (5000.0, 40.0)] {
            let (mut a, east) = if tracks == 4 { quad_start() } else { tangent_start() };
            let off: f64 = (deg as f64).to_radians();
            let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
            let ends = from_node(east, x, y);
            let pv = a.preview_route(&ends, &[], tracks, false);
            assert_eq!(pv[0], 1.0, "{tracks} tracks, {d} m at {deg}: {}", a.issue_kinds());
            let f = a.w.net.fit(&a.route_vertices);
            let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
            assert!((ex - x).hypot(ey - y) < 0.01, "{d} m at {deg}: ends at {ex}, {ey}");
            let want = d / (2.0 * off.abs().sin());
            if want <= R_CAP {
                assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{d} m at {deg}: leaves on {eh}");
                for i in 1..a.route_vertices.len() - 1 {
                    assert!((f.radius[i] - want).abs() < 0.01 * want, "{d} m at {deg}: corner {i} radius {} against {want}", f.radius[i]);
                }
                assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{d} m at {deg}: one arc, no straight");
                assert!((a.route_arc()[2] - want).abs() < 0.5);
            } else {
                capped_then_straight(&f, &a.route_arc(), (1000.0, 0.0), &format!("{tracks} tracks, {d} m at {deg}"));
            }
            if tracks == 4 {
                let ex = a.w.net.fit(&a.route_express);
                let (qx, qy, qh) = geom::pos_at(&ex.pieces, ex.len);
                assert!(((qx - x).hypot(qy - y) - QUAD_SPACING).abs() < 0.05, "{d} m at {deg}: express ends 8 m beside");
                assert!(geom::wrap(qh - eh).abs() < 1e-3, "{d} m at {deg}: express parallel at the end");
            }
            assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{tracks} tracks, {d} m at {deg}: {}", a.issues());
        }
    }
    // a hairpin of 300 degrees going up a level ramps on its middle leg
    let (mut a, east) = tangent_start();
    let off: f64 = 150f64.to_radians();
    let (x, y) = (1000.0 + 400.0 * off.cos(), 400.0 * off.sin());
    let mut ends = from_node(east, x, y);
    ends[9] = 1.0;
    ends[5] = 0.0;
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_level[a.route_end() as usize], 1);
}

/// T-418: a click too close and too far off to the side is refused, the ghost and circle
/// still drawn; one behind the end is refused with its circle.
#[test]
fn a_too_tight_or_backward_tangent_click_is_refused() {
    let (mut a, east) = tangent_start();
    let off: f64 = 60f64.to_radians();
    let ends = from_node(east, 1000.0 + 40.0 * off.cos(), 40.0 * off.sin());
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 0.0);
    assert!(pv.len() > 5, "the red ghost is drawn");
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    let r = a.route_arc()[2];
    assert!((r - 40.0 / (2.0 * off.sin())).abs() < 0.5, "{r}");
    let fr = a.w.net.fit(&a.route_vertices).radius[1];
    assert!((fr - r).abs() < 0.5, "the ghost's curve is the circle's: {fr} against {r}");
    assert_ne!(a.add_route(&ends, &[], 2, false), 0);
    let back = from_node(east, 600.0, 5.0);
    assert_eq!(a.preview_route(&back, &[], 2, false)[0], 0.0);
    assert!(a.issue_kinds().contains("Reversal"), "{}", a.issue_kinds());
    assert_eq!(a.route_arc().len(), 3);
    assert_ne!(a.add_route(&back, &[], 2, false), 0);
    assert_eq!(live_edges(&a).len(), 1, "nothing added");
}

/// T-418: a 400 m platform across three 120 m segments joins all their joints first; dragging
/// a segment's corner afterwards moves the joint with it.
#[test]
fn a_station_spans_several_short_segments() {
    let (mut a, east) = tangent_start();
    let mut end = east;
    let mut heading = 0.0f64;
    let mut segs = vec![];
    for turn in [3.0f64, -3.0, 2.0, -2.0, 3.0] {
        let off = turn.to_radians();
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + 120.0 * (heading + off).cos(), y0 + 120.0 * (heading + off).sin());
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 2, false), 0, "{}", a.issues());
        heading += 2.0 * off;
        end = a.route_end();
        segs.push(*live_edges(&a).last().unwrap());
    }
    let mid = segs[2];
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(mid), a.w.net.edge_len[mid as usize] / 2.0);
    assert_eq!(a.add_station(2, mid, x, y, 400, "Long".into()), 0, "{}", a.issues());
    assert!(a.w.net.node_platform.contains(&400));
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a).len(), 6);
}

/// T-859: removing stations a line stops at takes them out of its stops and routes it again;
/// where the track between its stops is gone too it is left broken. One undo restores it all.
#[test]
fn removing_a_stop_reroutes_or_breaks_the_line() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 1.0 && a.w.net.node_y[n as usize].abs() < 1.0).unwrap();
    for x in [1000.0, 2000.0] {
        let e = edge_under(&a, x, 0.0);
        assert_eq!(a.add_station(2, e, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    for x in [0.0, 3000.0] {
        let n = node_at(&a, x);
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    let st: Vec<u32> = [0.0, 1000.0, 2000.0, 3000.0].iter().map(|&x| node_at(&a, x)).collect();
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 0, 0, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[l as usize]);
    // a middle stop, its track kept: the line runs on past it
    assert_eq!(a.removal_lines(&[], &[st[1]]), vec![l as f64, 1.0, 0.0]);
    assert_eq!(a.remove_station(st[1]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(l), &[st[0], st[2], st[3]]);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.w.net.line_stops(l), &st[..]);
    // the end stop and its track: the line ends a stop sooner
    let last = edge_under(&a, 2500.0, 0.0);
    assert_eq!(a.delete_pieces(&[last], &[st[3]]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(l), &st[..3]);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    // a middle stop and the track on to the next: broken, until undone
    let mid = edge_under(&a, 1500.0, 0.0);
    assert_eq!(a.removal_lines(&[mid], &[st[1]]), vec![l as f64, 1.0, 1.0]);
    assert_eq!(a.delete_pieces(&[mid], &[st[1]]), 0, "{}", a.issues());
    assert!(a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.w.net.line_stops(l), &st[..]);
}

/// T-860: a station on a loose end whose last segment is shorter than the platform joins the
/// segments behind it first, as a split does; one undo puts the joint back.
#[test]
fn a_station_on_a_short_last_segment_end() {
    let (mut a, east) = tangent_start();
    assert_eq!(a.add_route(&from_node(east, 1080.0, 0.0), &[], 2, false), 0, "{}", a.issues());
    let end = a.route_end();
    assert_eq!(live_edges(&a).len(), 2);
    assert_eq!(a.preview_station(1, end, 0.0, 0.0, 200), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 200, "End".into()), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 1, "joined under the platform");
    assert_eq!(a.w.net.node_platform[end as usize], 200);
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a).len(), 2);
}

// ---- T-622: a platform beside a crossing or a joint
