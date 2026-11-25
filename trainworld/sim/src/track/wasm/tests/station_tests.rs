use super::*;

/// T-311: `preview_station` answers as `add_station` would, in the middle of a stretch and
/// near its end, and changes nothing (no edit, no undo step, no cash, no ids).
#[test]
fn preview_station_agrees_with_add_station() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let edge = live_edges(&a)[0];
    let len = a.w.net.edge_len[edge as usize];
    let (cash, nodes, edges_n) = (a.cash, a.nodes_info(), live_edges(&a));
    let mut seen = [0, 0];
    for s in [len * 0.5, 20.0, len - 20.0, len * 0.3] {
        let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edge), s);
        let dry = a.preview_station(2, edge, x, y, 200);
        assert_eq!((a.cash, a.nodes_info(), live_edges(&a)), (cash, nodes.clone(), edges_n.clone()), "dry run at {s}");
        let why = a.issues();
        let real = a.add_station(2, edge, x, y, 200, "S".into());
        assert_eq!(dry == 0, real == 0, "at {s}: preview {dry} ({why}) real {real} ({})", a.issues());
        seen[(real == 0) as usize] += 1;
        if real == 0 {
            assert_eq!(a.undo(), 0);
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "both answers should occur: {seen:?}");
}

/// T-4214: a station placed with no name (its name sent after the click) takes the name that comes, once,
/// outside the undo history: an undo takes the station away and its redo brings it back named.
#[test]
fn a_station_placed_before_its_name_takes_it_after() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let edge = live_edges(&a)[0];
    let len = a.w.net.edge_len[edge as usize];
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edge), len * 0.5);
    assert_eq!(a.add_station(2, edge, x, y, 200, String::new()), 0, "{}", a.issues());
    let st = a.last_stations();
    assert_eq!(st.len(), 1);
    let node = st[0] as usize;
    assert_eq!(a.w.net.node_name[node], "");
    assert_eq!(a.name_new_stations(&st, "Late".into()), 1);
    assert_eq!(a.w.net.node_name[node], "Late");
    assert_eq!(a.name_new_stations(&st, "Other".into()), 0, "a named station keeps its name");
    assert_eq!(a.undo(), 0);
    assert!(a.w.net.station_data(node as u32).is_none());
    assert_eq!(a.redo(), 0);
    assert_eq!(a.w.net.node_name[node], "Late", "its redo brings it back named");
}

/// T-2797: the short piece past a station's platform cannot be removed (a removal would lay it
/// back), and says so; removing the station frees it.
#[test]
fn a_stations_end_piece_is_refused_until_the_station_goes() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(240.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_free_station(&ends, 2, 200, "Open".into()), 0, "{}", a.issues());
    let station = a.nodes_info().chunks(10).find(|c| c[7] > 0.0).unwrap()[0] as u32;
    // 120 m each side of a 200 m platform: the part past the trim goes
    for e in live_edges(&a) {
        assert_eq!(a.delete_edge(e), 0, "{}", a.issues());
    }
    let left = live_edges(&a);
    assert_eq!(left.len(), 2);
    for &e in &left {
        assert_eq!(a.delete_edge(e), 1, "only the station's own track is left");
        assert_eq!(a.issue_kinds(), "StationTrack");
    }
    assert_eq!(a.delete_pieces(&left, &[]), 1);
    assert_eq!(a.issue_kinds(), "StationTrack");
    assert_eq!(live_edges(&a), left, "refused, nothing moved");
    assert_eq!(a.remove_station(station), 0, "{}", a.issues());
    for e in live_edges(&a) {
        assert_eq!(a.delete_edge(e), 0, "{}", a.issues());
    }
    assert!(live_edges(&a).is_empty());
}

/// T-794: a station on open ground lays a straight piece of track a little longer than the
/// platform and puts the station at its middle, one undoable edit; the trial changes nothing
/// and prices both; the piece's ends are loose ends that later track carries on from; on quad
/// the express pair gets the twin.
#[test]
fn a_free_station_lays_its_own_track() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(240.0, 0.0).iter()).copied().collect();
    let (cash, nodes) = (a.cash, a.nodes_info());
    assert_eq!(a.preview_free_station(&ends, 2, 200), 0, "{}", a.issues());
    assert_eq!((a.cash, a.nodes_info(), live_edges(&a).len()), (cash, nodes, 0), "the trial leaves nothing");
    let quoted = a.station_cost();
    assert!(quoted > 0.0);
    assert_eq!(a.add_free_station(&ends, 2, 200, "Open".into()), 0, "{}", a.issues());
    assert!((a.total_cost() - quoted).abs() < 1e-6, "{} against {quoted}", a.total_cost());
    let info = a.nodes_info();
    let rows: Vec<&[f64]> = info.chunks(10).collect();
    assert_eq!(rows.len(), 3);
    let st: Vec<&&[f64]> = rows.iter().filter(|c| c[7] > 0.0).collect();
    assert_eq!(st.len(), 1);
    assert!((st[0][1] - 120.0).abs() < 1.0, "the station at the middle: {:?}", st[0]);
    let loose: Vec<f64> = rows.iter().filter(|c| c[4] == 1.0).map(|c| c[0]).collect();
    assert_eq!(loose.len(), 2, "both ends loose");
    // carrying on from an end: an ordinary route, the station keeps its platform
    let end = rows.iter().find(|c| c[4] == 1.0 && c[1] > 200.0).unwrap()[0];
    let on = [1.0, end, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&on, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.undo(), 0);
    // one undo takes the station and its track
    assert_eq!(a.undo(), 0);
    assert!(live_edges(&a).is_empty() && a.nodes_info().is_empty());
    // too short for the platform: refused, nothing left
    let short: Vec<f64> = free(0.0, 0.0).iter().chain(free(150.0, 0.0).iter()).copied().collect();
    assert_ne!(a.add_free_station(&short, 2, 200, "Short".into()), 0);
    assert!(a.issue_kinds().contains("PlatformTooLong"), "{}", a.issues());
    assert!(live_edges(&a).is_empty());
    // quad: on the drawn pair only (T-1687)
    assert_eq!(a.add_free_station(&ends, 4, 200, "Quad".into()), 0, "{}", a.issues());
    let named = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == "Quad").count();
    assert_eq!(named, 1, "the station alone");
    // across other track on the same level: refused, though a flat crossing is allowed elsewhere
    let mut b = api();
    let across: Vec<f64> = free(230.0, -300.0).iter().chain(free(230.0, 300.0).iter()).copied().collect();
    assert_eq!(b.add_route(&across, &[], 2, false), 0, "{}", b.issues());
    assert_ne!(b.preview_free_station(&ends, 2, 200), 0);
    assert!(b.issue_kinds().contains("PlatformCrossing"), "{}", b.issue_kinds());
    assert_eq!(live_edges(&b).len(), 1);
}

/// T-1299: removing a station on its own piece of track (laid on open ground, or splitting a
/// piece) leaves one piece where the two either side simply join, built or not, with the lines
/// over it still running; one undo brings back the station and both pieces. Pieces unlike each
/// other (built beside blueprint) stay two.
#[test]
fn removing_a_station_joins_the_pieces_either_side() {
    let station = |t: &TrackApi| (0..t.w.net.node_count() as u32).find(|&k| t.w.net.node_ok(k) && t.w.net.node_platform[k as usize] > 0).unwrap();
    // on open ground
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(240.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_free_station(&ends, 2, 200, "Open".into()), 0, "{}", a.issues());
    let (st, before) = (station(&a), live_edges(&a));
    assert_eq!(before.len(), 2);
    assert_eq!(a.remove_station(st), 0, "{}", a.issues());
    let left = live_edges(&a);
    assert_eq!(left.len(), 1, "one piece left");
    assert!(!a.w.net.node_ok(st), "the joint went too");
    assert!((a.w.net.edge_len[left[0] as usize] - 240.0).abs() < 0.01);
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a), before);
    assert_eq!(a.w.net.node_platform[st as usize], 200);
    // splitting a constructed piece, a line stopping there and running through
    let mut b = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(b.add_route(&ends, &[], 2, false), 0);
    let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, east) = (end(&b, 0.0), end(&b, 3000.0));
    assert_eq!(b.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(b.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
    assert_eq!(b.add_station(2, live_edges(&b)[0], 1500.0, 0.0, 200, "Middle".into()), 0, "{}", b.issues());
    let mid = end(&b, 1500.0);
    let id = b.new_line_id();
    assert_eq!(b.set_line_with_trains(id, &[w, mid, east], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", b.issues());
    assert_eq!(b.construct_line(id), 0, "{}", b.issues());
    assert_eq!(live_edges(&b).len(), 2);
    assert_eq!(b.remove_station(mid), 0, "{}", b.issues());
    let left = live_edges(&b);
    assert_eq!(left.len(), 1, "one piece left");
    assert!(b.w.net.edge_built[left[0] as usize]);
    assert!((b.w.net.edge_len[left[0] as usize] - 3000.0).abs() < 0.01);
    assert!(!b.w.net.line_broken[id as usize]);
    assert_eq!(b.w.net.line_stops(id), &[w, east]);
    // built track beside blueprint: two pieces stay
    let mut c = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(c.add_route(&ends, &[], 2, false), 0);
    let j = end(&c, 1000.0);
    assert_eq!(c.add_station(1, j, 0.0, 0.0, 200, "Joint".into()), 0, "{}", c.issues());
    assert_eq!(c.construct(&live_edges(&c), &[j]), 0, "{}", c.issues());
    let on = [1.0, j as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0, 0.0, 0.0];
    assert_eq!(c.add_route(&on, &[], 2, false), 0, "{}", c.issues());
    assert_eq!(c.remove_station(j), 0, "{}", c.issues());
    assert_eq!(live_edges(&c).len(), 2, "built and blueprint stay apart");
}
