//! T-3405: a speed limit on a piece of track.

use super::*;
use crate::track::tests::{edge, line, node, station, world};

/// Two built 40 km straights A-B and B-C of ordinary double track, stations at A and C, a line A-C
/// with a fixed fleet (so no edit buys trains); the edges in order.
fn two_pieces() -> (TrackApi, u32, [u32; 2]) {
    let mut w = world();
    let n = [node(&mut w, 0.0, 0.0, 0), node(&mut w, 40000.0, 0.0, 0), node(&mut w, 80000.0, 0.0, 0)];
    let e = [edge(&mut w, n[0], n[1], &[], 0).unwrap(), edge(&mut w, n[1], n[2], &[], 0).unwrap()];
    station(&mut w, n[0], 200);
    station(&mut w, n[2], 200);
    let l = line(&mut w, &[n[0], n[2]], [4.0, 2.0, 1.0]);
    let mut d = w.net.line_data(l).unwrap();
    d.trains = Some([4, 4, 4]);
    w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a.w = w;
    // an edit that changes nothing buys the line's fleet, so later cash is the edits' own
    assert_eq!(a.set_line_limit(l, 0), 0);
    (a, l, e)
}

fn top_kmh(a: &TrackApi, l: u32, s0: f64, s1: f64) -> f64 {
    crate::track::profile::v_max(&a.w.svc.lines[l as usize].runs[0].free.phases, s0, s1) * 3.6
}

#[test]
fn a_piece_limit_slows_trains_there_only_is_free_and_clears_exactly() {
    let (mut a, l, e) = two_pieces();
    let free = a.w.svc.lines[l as usize].clone();
    let cash = a.cash();
    assert_eq!(a.set_edges_limit(&[e[1]], 60), 0, "{}", a.issues());
    assert_eq!(a.edge_limits(), vec![e[1], 60]);
    assert_eq!(a.cash(), cash, "free");
    assert!((top_kmh(&a, l, 0.0, 38000.0) - 160.0).abs() < 1e-6);
    assert!((top_kmh(&a, l, 41000.0, 79000.0) - 60.0).abs() < 1e-6);
    // the run's limits are 160 then 60 from the joint (less half a train, which must be under the
    // limit as it enters), and the time added is what the train profile gives for exactly that:
    // about 40000 (1/16.7 - 1/44.4) = 1500 s, a little less for braking from 60 at C, not 160
    let run = &a.w.svc.lines[l as usize].runs[0];
    let (s0, s1) = (free.runs[0].clone(), run.clone());
    let half = (40000.0 - s1.limits[1][0]).max(0.0);
    assert_eq!(s1.limits.len(), 2, "{:?}", s1.limits);
    assert!((s1.limits[1][1] * 3.6 - 60.0).abs() < 1e-9 && half > 0.0 && half < 300.0, "{:?}", s1.limits);
    let prof = |lim: &[[f64; 2]], r: &crate::track::service::Run| {
        crate::track::profile::run(lim, &r.stop_s, &r.jump, &vec![0.0; r.stop_s.len()], &[]).duration
    };
    let want = prof(&s1.limits, &s1) - prof(&s0.limits, &s0);
    let more = s1.free.duration - s0.free.duration;
    assert!((more - want).abs() < 1e-6 && (more - 1500.0).abs() < 30.0, "{more} s more, the profile says {want}");
    // the stop-time hover names the piece's limit
    let sp = a.line_stretch_speeds(l);
    assert_eq!((sp[1].round(), sp[2]), (160.0, 0.0), "the fastest stretch is the unlimited piece");
    // a second set to the same value changes nothing; clearing restores the run exactly
    assert_eq!(a.set_edges_limit(&[e[1]], 60), 1);
    assert_eq!(a.set_edges_limit(&[e[1]], 0), 0, "{}", a.issues());
    assert!(a.edge_limits().is_empty());
    assert_eq!(a.w.svc.lines[l as usize].round_trip, free.round_trip);
    assert_eq!(a.w.svc.lines[l as usize].runs[0].free.duration, free.runs[0].free.duration);
    assert_eq!(a.cash(), cash);
    // undo puts it back
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edge_limits(), vec![e[1], 60]);
}

#[test]
fn a_piece_limit_is_kept_within_its_kind_and_never_on_a_crossover() {
    let (mut a, l, e) = two_pieces();
    assert_eq!(a.set_edges_limit(&[e[0], e[1]], 10), 0);
    assert_eq!(a.edge_limits(), vec![e[0], 40, e[1], 40], "40 at least");
    // T-3525: at or over ordinary track's 160 is no limit
    assert_eq!(a.set_edges_limit(&[e[0]], 500), 0);
    assert_eq!(a.edge_limits(), vec![e[1], 40], "over the top is none");
    let sp = a.line_stretch_speeds(l);
    assert_eq!((sp[1].round(), sp[2]), (160.0, 0.0));
    assert_eq!(a.set_edges_limit(&[e[0]], 100), 0);
    let sp = a.line_stretch_speeds(l);
    assert_eq!((sp[1].round(), sp[2]), (100.0, 4.0));
    // a line limit below the piece's wins, and above it the piece's does
    assert_eq!(a.set_line_limit(l, 80), 0);
    assert!((top_kmh(&a, l, 1000.0, 38000.0) - 80.0).abs() < 1e-6);
    assert!((top_kmh(&a, l, 41000.0, 79000.0) - 40.0).abs() < 1e-6);
    // a crossover is refused
    let mut d = a.w.net.edge_data(e[0]).unwrap();
    d.crossover = true;
    d.limit = 100;
    a.w.apply(Op::Edge { id: e[0], data: Some(d) }).unwrap();
    assert_eq!(a.w.net.edge_limit[e[0] as usize], 0, "set_edge drops it on a crossover");
    assert_ne!(a.set_edges_limit(&[e[0]], 60), 0);
}

#[test]
fn a_split_keeps_the_limit_on_both_halves_and_unlike_pieces_never_join() {
    let (mut a, _l, e) = two_pieces();
    assert_eq!(a.set_edges_limit(&[e[1]], 60), 0);
    let node = a.w.net.alloc_node();
    let new_edge = a.w.net.alloc_edge();
    a.w.apply(Op::Split { edge: e[1], s: 20000.0, node, new_edge }).unwrap();
    assert_eq!((a.w.net.edge_limit[e[1] as usize], a.w.net.edge_limit[new_edge as usize]), (60, 60));
    // the joint between the two halves joins (same limit); the joint between e0 and e1 does not
    let (ops, _) = a.unjoin_near(e[1], 1000.0, 30000.0).expect("the halves join");
    let joined: Vec<u32> = ops.iter().filter_map(|o| match o { Op::Edge { id, data: None } => Some(*id), _ => None }).collect();
    assert!(!joined.contains(&e[0]), "the unlimited piece stays apart: {joined:?}");
    for o in &ops {
        if let Op::Edge { data: Some(d), .. } = o {
            assert_eq!(d.limit, 60);
        }
    }
}

/// T-3525: a limit at the kind's top speed is none, set on many pieces at once (one undo), so the piece
/// joins its unlimited neighbour; an old save's limit at the top loads as none.
#[test]
fn a_limit_at_the_top_is_none_and_joins_unlimited_track() {
    let (mut a, _l, e) = two_pieces();
    assert_eq!(a.set_edges_limit(&[e[0], e[1]], 100), 0);
    assert_eq!(a.edge_limits(), vec![e[0], 100, e[1], 100]);
    assert_eq!(a.set_edges_limit(&[e[0], e[1]], 160), 0);
    assert!(a.edge_limits().is_empty(), "160 on ordinary track is none: {:?}", a.edge_limits());
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edge_limits(), vec![e[0], 100, e[1], 100], "one undo puts both back");
    assert_eq!(a.set_edges_limit(&[e[1]], 160), 0);
    assert_eq!(a.set_edges_limit(&[e[0]], 0), 0);
    assert!(a.edge_limits().is_empty());
    assert_eq!(a.set_edges_limit(&[e[1]], 160), 1, "the top again changes nothing");
    let (ops, _) = a.unjoin_near(e[1], 1000.0, 30000.0).expect("the pieces join");
    let gone: Vec<u32> = ops.iter().filter_map(|o| match o { Op::Edge { id, data: None } => Some(*id), _ => None }).collect();
    assert!(gone.contains(&e[0]) || gone.contains(&e[1]), "a piece set to its top joins the unlimited one: {gone:?}");
    // a save written with a limit at the top (before T-3525) loads with none: `LPS1` is read through
    // `set_edge`, as here
    let d = a.w.net.edge_data(e[1]).unwrap();
    a.w.net.set_edge(e[1], Some(EdgeData { limit: 160, ..d.clone() })).unwrap();
    assert_eq!(a.w.net.edge_limit[e[1] as usize], 0);
    a.w.net.set_edge(e[1], Some(EdgeData { limit: 150, ..d })).unwrap();
    assert_eq!(a.w.net.edge_limit[e[1] as usize], 150);
}

#[test]
fn piece_limits_save_and_load_and_a_network_without_them_saves_as_before() {
    let (mut a, l, e) = two_pieces();
    let before = crate::track::save::encode(&a.w.net);
    assert!(!before.windows(4).any(|x| x == b"LPS1"));
    assert_eq!(a.set_edges_limit(&[e[1]], 60), 0);
    let bytes = crate::track::save::encode(&a.w.net);
    assert!(bytes.windows(4).any(|x| x == b"LPS1"));
    let back = crate::track::world::TrackWorld::new(crate::track::save::decode(&bytes).unwrap());
    assert_eq!(back.net.edge_limit[e[1] as usize], 60);
    assert_eq!(back.net.edge_limit[e[0] as usize], 0);
    assert_eq!(crate::track::save::encode(&back.net), bytes);
    assert_eq!(back.svc.lines[l as usize].round_trip, a.w.svc.lines[l as usize].round_trip);
    assert_eq!(a.set_edges_limit(&[e[1]], 0), 0);
    assert_eq!(crate::track::save::encode(&a.w.net), before);
}
