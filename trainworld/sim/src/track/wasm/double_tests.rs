//! T-3296: rebuilding single track as double track; T-3321: double track as single.

use super::*;
use crate::track::capacity::ResKind;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// A built 8 km run of `tracks`, a station 1 km from each end and a line over it.
fn run(tracks: u8) -> (TrackApi, [u32; 2], u32) {
    run_of(tracks, 0)
}

/// `run` of track kind `kind` (`TrackKind` number: 0 regular, 1 high speed, 3 tram).
fn run_of(tracks: u8, kind: u8) -> (TrackApi, [u32; 2], u32) {
    let mut a = api();
    a.set_cash(1e6);
    a.set_route_hsr(kind == 1);
    a.set_route_kind(kind);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{}", a.issues());
    let platform = if kind == 3 { 60 } else { 200 };
    let mut st = [0; 2];
    for (i, x) in [1000.0, 7000.0].into_iter().enumerate() {
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, platform, format!("S{i}")), 0, "{}", a.issues());
        st[i] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issues());
    (a, st, line)
}

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}

fn singles(a: &TrackApi) -> usize {
    a.w.svc.cap.res.iter().filter(|r| r.kind == ResKind::SingleTrack).count()
}

/// Built single track rebuilt as double: the same pieces, stations and line, now with two tracks and no shared
/// single-track section, the full double price taken as quoted, and no undo (money was spent).
#[test]
fn built_single_track_rebuilds_as_double_for_its_full_price() {
    let (mut a, st, line) = run(1);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1));
    assert!(singles(&a) > 0, "one shared track before");
    let station_before = a.w.net.station_build_price(st[0]);
    let q = a.rebuild_quote(&edges, &[], 99, 0, 2);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0 && q[3] == 0.0, "{q:?} {}", a.issue_kinds());
    let cash = a.cash();
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 2), 0, "{}", a.issue_kinds());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9, "charged {} quoted {}", cash - a.cash(), q[0]);
    let full: f64 = edges.iter().map(|&e| a.w.net.edge_cost[e as usize]).sum::<f64>() + st.iter().map(|&n| a.w.net.station_build_price(n)).sum::<f64>();
    assert!((q[0] - full).abs() < 1e-9, "full double price of track and stations");
    assert_eq!(live(&a), edges, "no edge made or lost");
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 2 && a.w.net.edge_built[e as usize]));
    assert!(st.iter().all(|&n| a.w.net.node_tracks(n) == 2), "double-track stations");
    assert!(a.w.net.station_build_price(st[0]) > station_before, "priced as a double station");
    assert_eq!(singles(&a), 0, "the single's margin is gone");
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
    assert!(!a.can_undo(), "paid, so final");
    // doubled already: nothing to do
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 2), 1);
    assert_eq!(a.last_skipped(), 0);
}

/// Blueprint single track becomes double free, in one undoable edit.
#[test]
fn blueprint_single_track_becomes_double_and_undoes() {
    let (mut a, st, _) = run(1);
    let edges = live(&a);
    assert_eq!(a.set_edges_tracks(&edges, 2), 0, "{}", a.issue_kinds());
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 2));
    assert!(st.iter().all(|&n| a.w.net.node_tracks(n) == 2));
    assert_eq!(a.set_edges_tracks(&edges, 2), 1, "nothing left to change");
    assert_eq!(a.undo(), 0);
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1));
}

/// Where doubled track meets single, the single end stays as it was (a plain double-to-single joint), and
/// a one-way single piece stays single and is refused.
#[test]
fn the_stretch_ends_in_a_plain_join_and_one_way_pieces_stay() {
    let (mut a, _, _) = run(1);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(edges.len(), 3);
    let part = &edges[..1];
    assert_eq!(a.rebuild_pieces(part, &[], 99, 0, 2), 0, "{}", a.issue_kinds());
    assert_eq!(a.w.net.edge_tracks[edges[0] as usize], 2);
    assert!(edges[1..].iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1));
    assert!(a.w.svc.lines.iter().any(|l| l.running));
    // pinned one way: refused, and counted
    assert_eq!(a.set_edge_oneway(edges[2], 1), 0, "{}", a.issues());
    let q = a.rebuild_quote(&edges[2..], &[], 99, 0, 2);
    assert_eq!((q[0], q[1], q[3]), (0.0, 0.0, 1.0), "{q:?}");
    assert_eq!(a.rebuild_pieces(&edges[2..], &[], 99, 0, 2), 1);
    assert_eq!(a.w.net.edge_tracks[edges[2] as usize], 1);
}

// ---- T-3321: double track rebuilt as single

fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn edge_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, y).1 < 1.0).unwrap()
}
fn on_edge(a: &TrackApi, x: f64, y: f64) -> [f64; 5] {
    [2.0, edge_at(a, x, y) as f64, x, y, 0.0]
}
fn node_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k)).min_by(|&p, &q| d(p).total_cmp(&d(q))).unwrap()
}
/// The live pieces of double track whose two ends both lie on y = `y` (within a metre).
fn on_line(a: &TrackApi, y: f64) -> Vec<u32> {
    let n = &a.w.net;
    live(a).into_iter().filter(|&e| [n.edge_a[e as usize], n.edge_b[e as usize]].iter().all(|&k| (n.node_y[k as usize] - y).abs() < 1.0)).collect()
}
fn tracks_of(a: &TrackApi, edges: &[u32]) -> Vec<u8> {
    edges.iter().map(|&e| a.w.net.edge_tracks[e as usize]).collect()
}
/// A double main west to east, (0, 0) to (6000, 0), built, with a single branch from (2000, 0) to (3500, 1500).
fn main_and_branch(a: &mut TrackApi) {
    a.set_cash(1e6);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(on_edge(a, 2000.0, 0.0), free(3500.0, 1500.0)), &[], 1, false), 0, "{}", a.issues());
}

/// Built double track rebuilt as single: free, nothing back, the same pieces, stations and line; the stations
/// become single-track stations and the stretch a shared single-track section. Final, as every built rebuild.
#[test]
fn built_double_track_rebuilds_as_single_for_free() {
    let (mut a, st, line) = run(2);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(singles(&a), 0);
    let q = a.rebuild_quote(&edges, &[], 99, 0, 1);
    assert_eq!((q[0], q[1] as usize, q[2], q[3]), (0.0, edges.len(), 0.0, 0.0), "free, every piece, no station charged: {q:?} {}", a.issue_kinds());
    let cash = a.cash();
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.cash(), cash, "nothing charged, nothing back");
    assert_eq!(a.last_charge(), 0.0);
    assert_eq!(live(&a), edges, "no edge made or lost");
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1 && a.w.net.edge_built[e as usize]));
    assert!(edges.iter().all(|&e| a.w.net.edge_way(e) == 0), "runs both ways");
    assert!(st.iter().all(|&n| a.w.net.node_tracks(n) == 1), "single-track stations");
    assert!(singles(&a) > 0, "one shared track now");
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
    assert!(!a.can_undo(), "built: final");
    // single already: nothing to do
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 1);
    assert_eq!(a.last_skipped(), 0);
    // and back to double, at the full double price
    let q = a.rebuild_quote(&edges, &[], 99, 0, 2);
    assert!(q[0] > 0.0 && q[2] == 2.0, "{q:?}");
}

/// Part of a stretch made single: a station between a single and a double piece stays a double-track
/// station (its platform on the double side), and the line still runs.
#[test]
fn part_of_a_stretch_goes_single_and_its_station_keeps_the_double_side() {
    let (mut a, st, line) = run(2);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(edges.len(), 3);
    let first = *edges.iter().find(|&&e| { let n = &a.w.net; let ends = [n.edge_a[e as usize], n.edge_b[e as usize]]; ends.contains(&st[0]) && ends.iter().any(|&k| n.node_ports[k as usize].n == 1) }).unwrap();
    assert_eq!(a.rebuild_pieces(&[first], &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.w.net.edge_tracks[first as usize], 1);
    assert_eq!(edges.iter().filter(|&&e| a.w.net.edge_tracks[e as usize] == 2).count(), 2);
    assert_eq!(a.w.net.node_tracks(st[0]), 2);
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
}

/// Blueprint double track becomes single at once, free, in one undoable edit.
#[test]
fn blueprint_double_track_becomes_single_and_undoes() {
    let (mut a, st, _) = run(2);
    let edges = live(&a);
    let cash = a.cash();
    assert_eq!(a.set_edges_tracks(&edges, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped(), 0);
    assert_eq!(a.cash(), cash);
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1));
    assert!(st.iter().all(|&n| a.w.net.node_tracks(n) == 1));
    assert_eq!(a.set_edges_tracks(&edges, 1), 1, "nothing left to change");
    assert_eq!(a.undo(), 0);
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 2));
}

/// A level change with it pays as any level change does, for the single track it becomes.
#[test]
fn single_with_a_level_pays_the_single_price_at_the_level() {
    let (mut a, st, _) = run(2);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    let q = a.rebuild_quote(&edges, &st, 1, 0, 1);
    assert!(q[0] > 0.0 && q[3] == 0.0, "{q:?} {}", a.issue_kinds());
    let cash = a.cash();
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0, 1), 0, "{}", a.issue_kinds());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
    let full: f64 = edges.iter().map(|&e| a.w.net.edge_cost[e as usize]).sum::<f64>() + st.iter().map(|&n| a.w.net.station_build_price(n)).sum::<f64>();
    assert!((q[0] - full).abs() < 1e-6, "the full price of single track at level +1: {} {full}", q[0]);
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1));
}

/// High-speed track is refused (and named) unless it is made regular too; tram track is refused.
#[test]
fn high_speed_and_tram_track_stay_double() {
    let (mut a, _, _) = run_of(2, 1);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let edges = live(&a);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize].is_hsr()));
    let q = a.rebuild_quote(&edges, &[], 99, 0, 1);
    assert_eq!((q[1], q[3]), (0.0, edges.len() as f64), "{q:?}");
    assert!(a.issue_kinds().contains("NoSingle:HighSpeed"), "{}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 1);
    assert!(a.issue_kinds().contains("NoSingle:HighSpeed"), "{}", a.issue_kinds());
    // with Regular picked too it goes
    let q = a.rebuild_quote(&edges, &[], 99, 1, 1);
    assert!(q[0] > 0.0 && q[3] == 0.0, "the type change is paid: {q:?} {}", a.issue_kinds());
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 1, 1), 0, "{}", a.issue_kinds());
    assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1 && a.w.net.edge_kind[e as usize] == TrackKind::Regular));

    let (mut a, _, _) = run_of(2, 3);
    let edges = live(&a);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram));
    assert_eq!(a.set_edges_tracks(&edges, 1), 1);
    assert_eq!(a.last_skipped() as usize, edges.len());
    assert!(a.issue_kinds().contains("NoSingle:Tram"), "{}", a.issue_kinds());
}

//// The y of edge `e`'s middle.
fn mid_y(a: &TrackApi, e: u32) -> f64 {
    crate::track::geom::pos_at(a.w.net.edge_pieces(e), a.w.net.edge_len[e as usize] / 2.0).1
}

/// A built quad run, (0, 0) to (6000, 0) doubled up, with a station 1 km from each end on the drawn pair and a line
/// between them.
fn quad_run() -> (TrackApi, [u32; 2], u32) {
    let mut a = api();
    a.set_cash(1e6);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.double_up(live(&a)[0]), 0, "{}", a.issue_kinds());
    let mut st = [0; 2];
    for (i, x) in [1000.0, 5000.0].into_iter().enumerate() {
        assert_eq!(a.add_station(2, edge_at(&a, x, 0.0), x, 0.0, 200, format!("Q{i}")), 0, "{}", a.issues());
        st[i] = node_near(&a, x, 0.0);
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Quad".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (a, st, line)
}

/// T-3339: quad track is two doubles. Picked whole it becomes two singles side by side, free, its stations
/// single-track stations and its line still running; one pair picked leaves single beside double.
#[test]
fn quad_track_goes_single_pair_by_pair() {
    let (mut a, st, line) = quad_run();
    let edges = live(&a);
    let (express, drawn): (Vec<u32>, Vec<u32>) = edges.iter().partition(|&&e| mid_y(&a, e) > 4.0);
    assert!(!express.is_empty() && drawn.len() >= 3);
    let q = a.rebuild_quote(&edges, &[], 99, 0, 1);
    assert_eq!((q[0], q[1] as usize, q[3]), (0.0, edges.len(), 0.0), "{q:?} {}", a.issue_kinds());
    let cash = a.cash();
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.cash(), cash);
    assert!(tracks_of(&a, &edges).iter().all(|&t| t == 1));
    assert!(st.iter().all(|&n| a.w.net.node_tracks(n) == 1));
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);

    let (mut a, _, line) = quad_run();
    let express: Vec<u32> = live(&a).into_iter().filter(|&e| mid_y(&a, e) > 4.0).collect();
    assert_eq!(a.rebuild_pieces(&express, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert!(tracks_of(&a, &express).iter().all(|&t| t == 1));
    assert!(live(&a).iter().filter(|&&e| mid_y(&a, e) < 4.0).all(|&e| a.w.net.edge_tracks[e as usize] == 2), "the drawn pair stays double");
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
}

/// A crossover at a piece's end keeps it double; the pieces with none go single.
#[test]
fn a_crossover_keeps_its_track_double() {
    let mut a = api();
    a.set_cash(1e6);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let n = node_near(&a, 6000.0, 0.0);
    assert_eq!(a.add_route(&ends([1.0, n as f64, 6000.0, 0.0, 0.0], free(9000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let gap = 12.0 * a.w.net.k_at(0.0);
    assert_eq!(a.add_route(&ends(free(0.0, gap), free(6000.0, gap)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_crossover(edge_at(&a, 3000.0, 0.0), 3000.0, 0.0), 0, "{}", a.issue_kinds());
    let main = on_line(&a, 0.0);
    assert!(main.len() >= 3, "{main:?}");
    let at_x = |a: &TrackApi, e: u32| [a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]].iter().any(|&n| a.w.net.node_ports[n as usize].ends().iter().any(|&c| a.w.net.edge_crossover[(c >> 1) as usize]));
    let (kept, gone): (Vec<u32>, Vec<u32>) = main.iter().partition(|&&e| at_x(&a, e));
    assert!(!kept.is_empty() && !gone.is_empty());
    assert_eq!(a.set_edges_tracks(&main, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.last_skipped() as usize, kept.len());
    assert!(a.issue_kinds().contains("NoSingle:Crossover"), "{}", a.issue_kinds());
    assert!(tracks_of(&a, &kept).iter().all(|&t| t == 2) && tracks_of(&a, &gone).iter().all(|&t| t == 1));
}

/// A one-way single branch joined to one track of the main (a stub runs one way, as a flyover's leg does) keeps
/// the main's piece it joins double. With a station at its end it runs both ways: the main goes single,
/// its join kept, counting again once the main is double.
#[test]
fn a_one_way_join_keeps_the_main_double_a_two_way_one_does_not() {
    let mut a = api();
    main_and_branch(&mut a);
    let j = node_near(&a, 2000.0, 0.0);
    let branch: Vec<u32> = live(&a).into_iter().filter(|&e| a.w.net.edge_tracks[e as usize] == 1).collect();
    assert!(branch.iter().all(|&e| a.w.net.edge_way(e) != 0), "the stub runs one way");
    let main = on_line(&a, 0.0);
    let at_j: Vec<u32> = main.iter().copied().filter(|&e| a.w.net.edge_a[e as usize] == j || a.w.net.edge_b[e as usize] == j).collect();
    assert_eq!(at_j.len(), 2);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    // the branch leaves eastward, so it is joined to a track of the west piece: that one stays, the east one goes
    let west = *at_j.iter().find(|&&e| a.w.net.node_x[a.w.net.edge_a[e as usize] as usize].min(a.w.net.node_x[a.w.net.edge_b[e as usize] as usize]) < 1.0).unwrap();
    let q = a.rebuild_quote(&main, &[], 99, 0, 1);
    assert_eq!((q[0], q[1], q[3], q[5]), (0.0, 1.0, 1.0, west as f64), "{q:?}");
    assert!(a.issue_kinds().contains("NoSingle:OneWay"), "{}", a.issue_kinds());
    // a station at the branch's end: both ways
    let end = node_near(&a, 3500.0, 1500.0);
    let (x, y) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
    assert_eq!(a.add_station(1, end, x, y, 200, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(branch.iter().all(|&e| a.w.net.edge_way(e) == 0));
    let joined = |a: &TrackApi| branch.iter().any(|&e| (0..2u32).any(|k| a.w.net.join_away(e << 1 | k).is_some()));
    assert!(joined(&a));
    assert_eq!(a.rebuild_pieces(&main, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert!(tracks_of(&a, &main).iter().all(|&t| t == 1));
    assert!(!joined(&a) && branch.iter().any(|&e| a.w.net.edge_join[e as usize] != [0; 2]), "the join is kept, not counted");
    assert_eq!(a.rebuild_pieces(&main, &[], 99, 0, 2), 0, "{}", a.issue_kinds());
    assert!(joined(&a), "and counts again");
}

/// A flying junction keeps the track at it double.
#[test]
fn a_flying_junction_keeps_its_track_double() {
    let mut a = api();
    a.set_cash(1e6);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(on_edge(&a, 2000.0, 0.0), free(3500.0, 1500.0)), &[], 2, false), 0, "{}", a.issues());
    let j = node_near(&a, 2000.0, 0.0);
    assert!(a.w.net.node_ports[j as usize].n >= 3);
    assert_eq!(a.set_flying(j, true), 0, "{}", a.issue_kinds());
    let at_j: Vec<u32> = live(&a).into_iter().filter(|&e| a.w.net.edge_a[e as usize] == j || a.w.net.edge_b[e as usize] == j).collect();
    assert_eq!(a.set_edges_tracks(&at_j, 1), 1);
    assert!(a.issue_kinds().contains("NoSingle:Flyover"), "{}", a.issue_kinds());
    assert!(tracks_of(&a, &at_j).iter().all(|&t| t == 2));
}

/// T-3339: a station with passing tracks keeps its double track along its platform and `PASSING_RUN_OUT` past each
/// end; the pieces either side are split there and go single beyond. Free, nothing refused, the line runs, and on
/// blueprint one undo puts the pieces back whole.
#[test]
fn a_station_with_passing_tracks_stays_the_double_passing_place() {
    for built in [false, true] {
        let (mut a, st, line) = run(2);
        assert_eq!(a.set_station_upgrade(&st[..1], crate::track::upgrades::PASSING, true), 0, "{}", a.issue_kinds());
        if built {
            assert_eq!(a.construct_all(), 0, "{}", a.issues());
        }
        let edges = live(&a);
        assert_eq!(edges.len(), 3);
        let cash = a.cash();
        if built {
            let q = a.rebuild_quote(&edges, &[], 99, 0, 1);
            assert_eq!((q[0], q[1] as usize, q[2], q[3]), (0.0, 3, 0.0, 0.0), "{q:?} {}", a.issue_kinds());
            assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
        } else {
            assert_eq!(a.set_edges_tracks(&edges, 1), 0, "{}", a.issue_kinds());
        }
        assert_eq!(a.last_skipped(), 0, "{built} {}", a.issue_kinds());
        assert_eq!(a.cash(), cash, "free");
        let net = &a.w.net;
        let now = live(&a);
        assert_eq!(now.len(), 5, "the two pieces at the station split");
        let at_st: Vec<u32> = now.iter().copied().filter(|&e| net.edge_a[e as usize] == st[0] || net.edge_b[e as usize] == st[0]).collect();
        assert_eq!(at_st.len(), 2);
        for &e in &at_st {
            assert_eq!(net.edge_tracks[e as usize], 2);
            assert!((net.edge_len[e as usize] - (100.0 + super::rebuild::PASSING_RUN_OUT)).abs() < 1.0, "platform half and run-out: {}", net.edge_len[e as usize]);
            assert_eq!(net.edge_built[e as usize], built);
        }
        assert!(now.iter().filter(|e| !at_st.contains(e)).all(|&e| net.edge_tracks[e as usize] == 1));
        assert_eq!(net.node_tracks(st[0]), 2, "the passing place");
        assert_eq!(net.node_tracks(st[1]), 1);
        assert_eq!(net.station_data(st[0]).unwrap().upgrades & crate::track::upgrades::PASSING, crate::track::upgrades::PASSING);
        assert!((!built || a.w.svc.lines[line as usize].running) && !net.line_broken[line as usize]);
        if built {
            assert!(!a.can_undo());
        } else {
            assert_eq!(a.undo(), 0);
            assert_eq!(live(&a), edges);
            assert!(tracks_of(&a, &edges).iter().all(|&t| t == 2));
        }
        // made single again: nothing left to do
        if built {
            assert_eq!(a.rebuild_pieces(&live(&a), &[], 99, 0, 1), 1);
        }
    }
}
