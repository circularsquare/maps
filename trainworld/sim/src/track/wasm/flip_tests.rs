//! T-3297: a game's running side, set on an empty network and turned over on one with track.

use super::flip::*;
use super::*;
use crate::track::capacity::junction_crosses;

fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn crossing(a: &TrackApi) -> Vec<u32> {
    let n = &a.w.net;
    (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && junction_crosses(n, k)).collect()
}
fn singles(a: &TrackApi) -> Vec<u32> {
    alive(a).into_iter().filter(|&e| a.w.net.edge_tracks[e as usize] == 1).collect()
}
fn sites(out: &[f64], kind: f64) -> Vec<u32> {
    out.chunks_exact(2).filter(|c| c[0] == kind).map(|c| c[1] as u32).collect()
}
/// The dead end or station nearest (x, y).
fn end_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n == 1)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}

/// A double main from (-3000, 0) to (5000, 0), a double branch from x = 1000 for (3500, 1500) laid with
/// a flyover, stations at the three ends, and a line from the main's west end to the branch's.
fn flyover_game(right_hand: bool) -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, right_hand);
    a.set_cash(1e9);
    assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(3500.0, 1500.0)), &[], 2, true), 0, "{}", a.issues());
    a.set_route_ramp(false);
    let mut st = vec![];
    for (k, (x, y)) in [(-3000.0, 0.0), (5000.0, 0.0), (3500.0, 1500.0)].into_iter().enumerate() {
        let n = end_near(&a, x, y);
        let (nx, ny) = (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
        assert_eq!(a.add_station(1, n, nx, ny, 200, format!("S{k}")), 0, "{}", a.issues());
        st.push(n);
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &[st[0], st[2]], 6, 4, 2, 30.0, 180.0, 0, 0, "Branch".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (a, l)
}

#[test]
fn t3297_side_is_set_only_on_an_empty_network() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    assert!(a.set_right_hand(false));
    assert!(!a.right_hand());
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    assert!(!a.set_right_hand(true), "refused once track is drawn");
    assert!(!a.right_hand());
    // and it is saved
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&a.save()));
    assert!(!b.right_hand());
}

#[test]
fn t3297_flip_keeps_a_flyover_grade_separated() {
    for right_hand in [true, false] {
        let (mut a, line) = flyover_game(right_hand);
        let legs = singles(&a);
        let ways: Vec<u8> = legs.iter().map(|&e| a.w.net.edge_oneway[e as usize]).collect();
        assert!(legs.len() >= 2 && ways.iter().all(|&w| w == 1 || w == 2), "{ways:?}");
        assert!(crossing(&a).is_empty(), "laid grade-separated");
        let out = a.flip_right_hand();
        assert_eq!(a.right_hand(), !right_hand);
        // every leg turned round, nothing left to fix, the line still running on the legs
        assert_eq!(sites(&out, FLIP_REVERSED).len(), legs.len(), "{out:?}");
        for (&e, &w) in legs.iter().zip(&ways) {
            assert_eq!(a.w.net.edge_oneway[e as usize], 3 - w, "leg {e}");
            assert_eq!(a.w.net.edge_way(e), 3 - w, "leg {e} runs its set way");
        }
        assert!(sites(&out, FLIP_BROKEN).is_empty() && sites(&out, FLIP_FLAT).is_empty() && sites(&out, FLIP_UNUSABLE).is_empty(), "{out:?}");
        assert!(crossing(&a).is_empty(), "still grade-separated, right hand {right_hand}");
        assert!(!a.w.net.line_broken[line as usize]);
        // its way out is now the other leg (the one that was its way back), so it is moved there
        assert_eq!(sites(&out, FLIP_MOVED), vec![line], "{out:?}");
        assert!(!a.can_undo(), "the history is dropped");
        // and back
        a.flip_right_hand();
        assert_eq!(a.right_hand(), right_hand);
        for (&e, &w) in legs.iter().zip(&ways) {
            assert_eq!(a.w.net.edge_oneway[e as usize], w);
        }
        assert!(crossing(&a).is_empty());
    }
}

#[test]
fn t3297_flip_turns_a_set_stub_and_keeps_a_flat_y() {
    for right_hand in [true, false] {
        let mut a = TrackApi::new(-73.985, 40.758, right_hand);
        a.set_cash(1e9);
        assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
        // a flat double Y: its badge stays (another move crosses), so it is not listed
        let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
        assert_eq!(a.add_route(&ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(2500.0, -1500.0)), &[], 2, false), 0, "{}", a.issues());
        // a single stub off the main, set one way by hand the other way from its join: it crosses
        let e = *alive(&a).iter().find(|&&e| a.w.net.edge_tracks[e as usize] == 2 && a.w.net.project(e, 4000.0, 0.0).1 < 1.0).unwrap();
        assert_eq!(a.add_route(&ends([2.0, e as f64, 4000.0, 0.0, 0.0], free(5000.0, 1200.0)), &[], 1, false), 0, "{}", a.issues());
        let stub = *singles(&a).first().unwrap();
        let auto = a.w.net.edge_way(stub);
        assert_ne!(auto, 0);
        assert_eq!(a.set_edge_oneway(stub, auto), 0, "{}", a.issues());
        let before = crossing(&a);
        let out = a.flip_right_hand();
        assert_eq!(sites(&out, FLIP_REVERSED), vec![stub], "{out:?}");
        assert_eq!(a.w.net.edge_way(stub), 3 - auto);
        assert_eq!(crossing(&a), before, "the Y keeps its badge, the stub gets none");
        assert!(sites(&out, FLIP_FLAT).is_empty(), "{out:?}");
    }
}

/// A one-way stub whose way the player set was not joined (pieces laid before T-3137 have none): its
/// way still turns round, since the side alone decides which track it meets.
#[test]
fn t3297_flip_turns_an_unjoined_set_stub() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(6000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 3000.0, 0.0).1 < 1.0).unwrap();
    assert_eq!(a.add_route(&ends([2.0, e as f64, 3000.0, 0.0, 0.0], free(4000.0, 1200.0)), &[], 1, false), 0, "{}", a.issues());
    let stub = *singles(&a).first().unwrap();
    assert_eq!(a.set_edge_oneway(stub, 1), 0, "{}", a.issues());
    // as an old save holds it
    a.w.net.edge_join[stub as usize] = [0, 0];
    a.w.reload();
    let out = a.flip_right_hand();
    assert_eq!(sites(&out, FLIP_REVERSED), vec![stub], "{out:?}");
    assert_eq!(a.w.net.edge_oneway[stub as usize], 2);
}
