//! T-3134: a preview's answer says each segment's track count, so the ghost draws single track as single.

use super::*;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn aligned(a: &TrackApi) {
    assert_eq!(a.preview_render_lanes().len(), a.preview_render_speeds().len());
}

#[test]
fn route_preview_lanes_follow_the_track_count() {
    let mut a = api();
    for (tracks, want) in [(1u8, 0u8), (2, 255), (4, 255)] {
        let out = a.preview_route(&ends(free(0.0, 0.0), free(2000.0, 300.0)), &[], tracks, false);
        assert_eq!(out[0], 1.0, "{}", a.issues());
        aligned(&a);
        let lanes = a.preview_render_lanes();
        assert!(!lanes.is_empty() && lanes.iter().zip(a.preview_render_speeds()).all(|(&l, s)| s < 0.0 || l == want), "{tracks}: {lanes:?}");
    }
}

#[test]
fn flyover_preview_has_single_legs_on_a_double_route() {
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(-3000.0, 0.0), free(5000.0, 0.0)), &[], 2, false), 0);
    let e = *alive(&a).iter().find(|&&e| a.w.net.project(e, 1000.0, 0.0).1 < 1.0).unwrap();
    a.set_route_ramp(true);
    let to = ends([2.0, e as f64, 1000.0, 0.0, 0.0], free(3500.0, 1500.0));
    assert_eq!(a.preview_route(&to, &[], 2, true)[0], 1.0, "{}", a.issues());
    aligned(&a);
    let (lanes, speeds) = (a.preview_render_lanes(), a.preview_render_speeds());
    let gap = speeds.iter().position(|&s| s < 0.0).expect("the crossing leg follows a step");
    assert!(lanes[gap + 1..].iter().all(|&l| l == 0), "the crossing leg is single");
    let main = &lanes[..gap];
    assert!(main.contains(&0) && main.contains(&255), "the route is single from the junction to J3, then double: {main:?}");
    // the single stretch is one run, at the route's junction end
    let ones = main.iter().filter(|&&l| l == 0).count();
    assert!(main[..ones].iter().all(|&l| l == 0) || main[main.len() - ones..].iter().all(|&l| l == 0));
    // flat: the whole route is double
    a.preview_route(&to, &[], 2, false);
    assert!(a.preview_render_lanes().iter().all(|&l| l == 255));
}

#[test]
fn edge_and_node_previews_carry_their_edge_lanes() {
    for (tracks, want) in [(1u8, 0u8), (2, 255)] {
        let mut a = api();
        assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(2000.0, 0.0)), &[], tracks, false), 0);
        let e = alive(&a)[0];
        assert_eq!(a.preview_edge(e, &[1000.0, 300.0, 0.0, 0.0])[0], 1.0, "{}", a.issues());
        aligned(&a);
        assert!(a.preview_render_lanes().iter().all(|&l| l == want));
        let node = a.w.net.end_node(e, 0);
        assert_eq!(a.preview_move_node(node, 0.0, 200.0)[0], 1.0, "{}", a.issues());
        assert_eq!(a.preview_part_lanes(), vec![want]);
    }
}
