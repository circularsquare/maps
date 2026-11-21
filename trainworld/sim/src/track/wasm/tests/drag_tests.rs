use super::*;

/// T-092: a click leaves the track drawn before it alone. The clicks so far with the cursor on
/// the next point, then that point clicked and the cursor moved on: nothing changes up to
/// where the curve at the previous click starts, that curve keeps its corner and can only get
/// tighter (when the leg to the cursor was what limited it: it shares that leg with the new
/// click's curve now), and the new click becomes a curve.
#[test]
fn a_click_keeps_the_track_drawn_before() {
    let mut a = api();
    // start on an existing track end, so the lead-in is in it too
    let ends: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
    let clicks = [(1500.0, 300.0), (2600.0, -400.0), (2850.0, -250.0), (4800.0, 1200.0), (5200.0, 2600.0), (7000.0, 2500.0), (7600.0, 1400.0)];
    let fit_of = |a: &mut TrackApi, k: usize, cursor: (f64, f64)| -> (geom::Fit, Vec<Pi>) {
        let pis: Vec<f64> = clicks[..k].iter().flat_map(|&(x, y)| [x, y, 0.0, 0.0]).collect();
        let ends = [1.0, start, 0.0, 0.0, 0.0, 0.0, 0.0, cursor.0, cursor.1, 0.0];
        let pv = a.preview_route(&ends, &pis, 2, false);
        assert_eq!(pv[0], 1.0, "{} clicks: {}", k, a.issues());
        let v = a.route_vertices.clone();
        (geom::fit(&v), v)
    };
    // where the curve at vertex i starts and ends (its middle where there is none)
    let arc = |f: &geom::Fit, i: usize| -> (f64, f64) {
        match f.pieces.iter().zip(&f.piece_at_vertex).find(|&(p, &v)| v as usize == i && p.k != 0.0) {
            Some((p, _)) => (p.s0, p.s0 + p.len),
            None => (f.vert_s[i], f.vert_s[i]),
        }
    };
    let (mut tighter, mut same) = (0, 0);
    for k in 1..clicks.len() - 1 {
        let (before, vb) = fit_of(&mut a, k, clicks[k]);
        let (after, va) = fit_of(&mut a, k + 1, clicks[k + 1]);
        let prev = vb.len() - 2; // the previous click's vertex in both
        assert_eq!(&vb[..=prev], &va[..=prev], "the vertices up to the previous click, lead-in included");
        // up to where the previous click's curve starts, the very same track
        let upto = arc(&before, prev).0.min(arc(&after, prev).0);
        let mut s = 0.0;
        while s <= upto {
            let (p, q) = (geom::pos_at(&before.pieces, s), geom::pos_at(&after.pieces, s));
            assert!((p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9 && (p.2 - q.2).abs() < 1e-12, "click {k}: moved at {s} m");
            s += 1.0;
        }
        for i in 1..prev {
            assert_eq!(before.radius[i], after.radius[i], "click {k}: an earlier curve changed");
        }
        // the previous click's curve: same corner, never wider
        let (rb, ra) = (before.radius[prev], after.radius[prev]);
        assert!(ra <= rb + 1e-9, "click {k}: the previous curve got wider, {rb} to {ra}");
        let leg = (vb[prev + 1].x - vb[prev].x).hypot(vb[prev + 1].y - vb[prev].y);
        if before.tangent[prev] <= leg / 2.0 + 1e-9 {
            // not limited by the leg to the cursor: the whole curve is untouched
            assert_eq!(rb, ra, "click {k}");
            let end = arc(&before, prev).1;
            assert!((end - arc(&after, prev).1).abs() < 1e-9);
            same += 1;
        } else if ra < rb - 1e-6 {
            tighter += 1;
        }
        // and the new click is a corner with its own curve
        assert!(after.radius[prev + 1] > 0.0, "click {k}");
    }
    assert!(tighter >= 1 && same >= 3, "both cases covered: {tighter} tighter, {same} unchanged");
}

/// T-475: several blueprint nodes dragged together: track between two of them shifts whole,
/// track to a node that stays bends; one undo puts all back.
#[test]
fn drag_several_blueprint_nodes() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    assert_eq!(a.add_station(2, e as u32, 4500.0, 0.0, 200, "Middle".into()), 0, "{}", a.issues());
    let ends = [2.0, e, 1500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
    let (w, s, east, b) = (node_at(&a, 0.0, 0.0), node_at(&a, 4500.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 3500.0, 1500.0));
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    let l1 = a.new_line_id();
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "W".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "E".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_with_trains(l1, &[w, s, east], 12, 8, 4, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
    let at = |a: &TrackApi, n: u32| (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
    let edge_between = |a: &TrackApi, p: u32, q: u32| live_edges(a).into_iter().find(|&e| { let (ea, eb) = (a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]); (ea == p && eb == q) || (ea == q && eb == p) }).unwrap();
    let (js, jb) = (edge_between(&a, j, s), edge_between(&a, j, b));
    let len = |a: &TrackApi, e: u32| a.w.net.edge_len[e as usize];
    let (ljs, ljb, lwj) = (len(&a, js), len(&a, jb), len(&a, edge_between(&a, w, j)));
    let pis_jb = a.edge_pis(jb);
    let before: Vec<(f64, f64)> = [w, j, s, east, b].iter().map(|&n| at(&a, n)).collect();
    let (dx, dy) = (120.0, 250.0);
    let pv = a.preview_move_nodes(&[j, s, b], dx, dy);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(pv[2], 4.0, "four edges change");
    assert_eq!(at(&a, j), before[1], "a preview changes nothing");
    assert_eq!(a.move_nodes(&[j, s, b], dx, dy), 0, "{}", a.issues());
    for (k, &n) in [j, s, b].iter().enumerate() {
        let (x0, y0) = before[[1, 2, 4][k]];
        assert!((at(&a, n).0 - x0 - dx).abs() < 0.01 && (at(&a, n).1 - y0 - dy).abs() < 0.01, "node {n} moved by the offset");
    }
    assert_eq!(at(&a, w), before[0], "unselected nodes stay");
    assert!((len(&a, js) - ljs).abs() < 0.01 && (len(&a, jb) - ljb).abs() < 0.01, "track between moved nodes keeps its shape");
    let shifted = a.edge_pis(jb);
    assert_eq!(shifted.len(), pis_jb.len());
    for c in 0..(pis_jb.len() - 1) / 8 {
        let (i, k) = (1 + c * 8, 2 + c * 8);
        assert!((shifted[i] - pis_jb[i] - dx).abs() < 0.01 && (shifted[k] - pis_jb[k] - dy).abs() < 0.01, "corners shift too");
    }
    assert!((len(&a, edge_between(&a, w, j)) - lwj).abs() > 1.0, "track to a node that stays bends");
    assert_eq!(a.lines_info()[2], 1.0, "the line still works");
    assert_eq!(a.undo(), 0);
    let after: Vec<(f64, f64)> = [w, j, s, east, b].iter().map(|&n| at(&a, n)).collect();
    assert_eq!(after, before, "one undo puts all back");
}

/// T-093: dragging blueprint nodes.
#[test]
fn drag_blueprint_nodes() {
    let mut a = api();
    // a 6 km line east, a station splitting it at 4.5 km, a branch from a junction at 1.5 km
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    assert_eq!(a.add_station(2, e as u32, 4500.0, 0.0, 200, "Middle".into()), 0, "{}", a.issues());
    let ends = [2.0, e, 1500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
    let (w, s, east, b) = (node_at(&a, 0.0, 0.0), node_at(&a, 4500.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 3500.0, 1500.0));
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    for n in [w, east, b] {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let l1 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l1, &[w, s, east], 12, 8, 4, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
    let l2 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l2, &[w, b], 12, 8, 4, 30.0, 180.0, 0, 0x0000ff, "Branch".into()), 0, "{}", a.issues());
    let heading = |a: &TrackApi, n: u32| a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap()[9];
    let same_axis = |h1: f64, h2: f64| geom::wrap(h1 - h2).abs() < 1e-6 || (geom::wrap(h1 - h2).abs() - PI).abs() < 1e-6;
    let at = |a: &TrackApi, n: u32| { let c = a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap().to_vec(); (c[1], c[2]) };
    let paths = |a: &TrackApi| (a.line_path(l1), a.line_path(l2), a.line_stops(l1), a.line_stops(l2));
    let before = paths(&a);
    let (hs, hj) = (heading(&a, s), heading(&a, j));
    let t_end = a.stop_times(l1, 0, 0)[2];

    // the end of the line (a terminus station on a track end): its heading follows its edge;
    // the station before it keeps its heading, its track bends out of it
    let pv = a.preview_move_node(east, 6200.0, 400.0);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(pv[2], 1.0, "one edge follows");
    assert_eq!(at(&a, east), (6000.0, 0.0), "a preview changes nothing");
    assert_eq!(a.move_node(east, 6200.0, 400.0), 0, "{}", a.issues());
    assert_eq!(at(&a, east), (6200.0, 400.0));
    assert!(same_axis(heading(&a, s), hs), "the station keeps its heading");
    assert!(a.stop_times(l1, 0, 0)[2] > t_end, "a longer run to the end");
    assert_eq!(paths(&a), before, "lines keep their stops and paths");
    assert_eq!(a.lines_info()[2], 1.0, "and still work");
    assert_eq!(a.undo(), 0);
    assert_eq!(at(&a, east), (6000.0, 0.0));
    assert_eq!(a.stop_times(l1, 0, 0)[2], t_end);

    // the station in the middle: it keeps its heading, both sides bend to meet it
    assert_eq!(a.move_node(s, 4500.0, 250.0), 0, "{}", a.issues());
    assert!(same_axis(heading(&a, s), hs) && same_axis(heading(&a, j), hj), "headings kept");
    assert_eq!(paths(&a), before);
    assert!(a.lines_info()[2] == 1.0 && a.lines_info()[4] == 0.0, "the line works, not broken");
    // and again: the lead-in from the last drag moves with it, nothing piles up
    let n_pis = a.w.net.pis.data.len();
    let pis_s = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).map(|e| a.w.net.edge_data(e).unwrap().pis.len()).sum::<usize>();
    let k0 = pis_s(&a);
    assert_eq!(a.move_node(s, 4600.0, 300.0), 0, "{}", a.issues());
    assert_eq!(pis_s(&a), k0, "{} PIs before, {} after ({n_pis})", k0, pis_s(&a));
    assert!(same_axis(heading(&a, s), hs));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);

    // the junction: its branch's lead-in moves with it, the main line bends
    assert_eq!(a.move_node(j, 1700.0, -150.0), 0, "{}", a.issues());
    assert!(same_axis(heading(&a, j), hj));
    assert_eq!(paths(&a), before);
    assert_eq!(a.undo(), 0);

    // too far: the track would turn back on itself; refused, nothing changes
    let pv = a.preview_move_node(east, 4400.0, 60.0);
    assert_eq!(pv[0], 0.0);
    assert_ne!(a.move_node(east, 4400.0, 60.0), 0);
    assert!(!a.issue_kinds().is_empty());
    assert_eq!(at(&a, east), (6000.0, 0.0));

    // constructed track meeting there: it stays put
    let e_wj = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && { let d = a.w.net.edge_data(e).unwrap(); (d.a == w && d.b == j) || (d.a == j && d.b == w) }).unwrap();
    assert_eq!(a.construct(&[e_wj], &[]), 0, "{}", a.issues());
    assert_ne!(a.move_node(j, 1700.0, -150.0), 0);
    assert!(a.issue_kinds().contains("NodeInUse"));
    assert_eq!(a.move_node(b, 3600.0, 1600.0), 0, "a blueprint end elsewhere still moves: {}", a.issues());
    // a constructed station stays put
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert_ne!(a.move_node(s, 4500.0, 250.0), 0);
    assert_eq!(a.preview_move_node(s, 4500.0, 250.0)[0], 0.0);
}

/// T-2951: a point added to blueprint track, then dragged.
#[test]
fn add_point_to_blueprint() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0] as u32;
    let count = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).count();
    let nodes = |a: &TrackApi| a.nodes_info().len() / 10;
    let (e0, n0) = (count(&a), nodes(&a));

    // a click 4 m beside the track: the point goes on it
    assert_eq!(a.add_point(e, 2500.0, 4.0), 0, "{}", a.issues());
    assert_eq!((count(&a), nodes(&a)), (e0 + 1, n0 + 1));
    let p = a.nodes_info().chunks(10).find(|c| (c[1] - 2500.0).abs() < 0.01 && c[2].abs() < 0.01).expect("a node at the point")[0] as u32;
    // it drags like any node, one undo each
    assert_eq!(a.move_node(p, 2500.0, 300.0), 0, "{}", a.issues());
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!((count(&a), nodes(&a)), (e0, n0), "one undo takes the point away");

    // the inspector's button: halfway
    assert_eq!(a.add_point_middle(e), 0, "{}", a.issues());
    assert!(a.nodes_info().chunks(10).any(|c| (c[1] - 3000.0).abs() < 0.01 && c[2].abs() < 0.01));
    assert_eq!(a.undo(), 0);

    // too near an end: refused, nothing changes
    assert_ne!(a.add_point(e, 2.0, 0.0), 0);
    assert!(a.issue_kinds().contains("SplitPoint"));
    assert_eq!((count(&a), nodes(&a)), (e0, n0));

    // inside a station's platform: refused
    assert_eq!(a.add_station(2, e, 4500.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    let (e1, n1) = (count(&a), nodes(&a));
    let on = (0..a.w.net.edge_count() as u32).find(|&x| a.w.net.edge_ok(x) && a.w.net.edge_data(x).unwrap().b != a.w.net.edge_data(x).unwrap().a && a.w.net.project(x, 4550.0, 0.0).1.abs() < 1.0 && a.w.net.project(x, 4550.0, 0.0).0 > 1.0).unwrap();
    assert_ne!(a.add_point(on, 4550.0, 0.0), 0, "a point inside a platform is refused");
    assert!(a.issue_kinds().contains("PlatformJoint"));
    assert_eq!((count(&a), nodes(&a)), (e1, n1));

    // constructed: refused
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert_ne!(a.add_point(e, 1000.0, 0.0), 0);
    assert!(a.issue_kinds().contains("NodeInUse"));
    assert_eq!((count(&a), nodes(&a)), (e1, n1));
}

/// Track drawn while the track ran through the clicked points (T-079, saves TWT3 with flag 8)
/// loads and looks as it did; its first reshape cuts the corners at its clicks.
#[test]
fn t079_track_keeps_its_shape_until_reshaped() {
    let mut a = api();
    let (n0, n1, e) = (a.new_node_id(), a.new_node_id(), a.new_edge_id());
    assert_eq!(a.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(a.set_node(n1, 4000.0, 0.0, 0, false), 0);
    // as T-079 stored it: fitted PIs with set radii, and the point clicked
    let fitted = vec![Pi::new(1200.0, 380.0, 1500.0, 0), Pi::new(2800.0, 380.0, 1500.0, 0)];
    let clicks = vec![Pi::new(2000.0, 300.0, 0.0, 0)];
    let op = Op::Edge { id: e, data: Some(EdgeData { a: n0, b: n1, tracks: 2, pis: fitted.clone(), built: false, off: false, thru: clicks.clone(), crossover: false, kind: TrackKind::Regular, sep: false, day: 0, oneway: 0, join: [0; 2], limit: 0, ramp: 0 }) };
    assert!(a.w.apply(op).is_ok());
    let bytes = a.save();
    assert_eq!(&bytes[..4], b"TWT4");
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.edges_info(), a.edges_info(), "loads as it was");
    assert_eq!(b.edge_render(0), a.edge_render(e), "and looks the same");
    assert_eq!(b.save(), bytes, "and saves the same");
    // its squares are the clicks, as auto PIs
    let p = b.edge_pis(0);
    assert_eq!(&p[..5], &[1.0, 2000.0, 300.0, 0.0, 0.0]);
    // reshaping it: the click becomes a corner; no clicks kept after that
    assert_eq!(b.set_edge_pis(0, &[2000.0, 400.0, 0.0, 0.0]), 0, "{}", b.issues());
    assert_eq!(b.edge_pis(0)[0], 0.0);
    assert!(b.w.net.edge_thru[0].is_empty());
    assert_ne!(b.edges_info()[5], a.edges_info()[5]);
    assert_eq!(b.undo(), 0);
    assert_eq!(b.edge_render(0), a.edge_render(e), "undo brings the T-079 shape back");
    assert_eq!(b.edge_pis(0)[0], 1.0);
}
