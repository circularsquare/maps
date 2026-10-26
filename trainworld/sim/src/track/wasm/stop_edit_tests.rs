use super::*;
use crate::track::tests::{edge, node, station, world};

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// `balloon_fixture` with a long loop (about 40 km): from M, back to A is quicker by reversing at M
/// than round the loop.
fn long_balloon() -> (TrackWorld, [u32; 4]) {
    let mut w = world();
    let a = node(&mut w, -3000.0, 0.0, 0);
    let m = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let k = node(&mut w, 5000.0, 1500.0, 0);
    edge(&mut w, a, m, &[], 0).unwrap();
    edge(&mut w, m, j, &[], 0).unwrap();
    edge(&mut w, j, k, &[(3400.0, 0.0)], 0).unwrap();
    edge(&mut w, k, j, &[(13000.0, 9000.0), (16000.0, -6000.0), (3800.0, 0.0)], 0).unwrap();
    for n in [a, m, k] {
        station(&mut w, n, 200);
    }
    (w, [a, m, j, k])
}

/// No reversing limiter (capacity kind 9) on any of the line's nodes.
fn no_limiter(a: &TrackApi, l: u32) -> bool {
    let cap = &a.w.svc.cap;
    (0..cap.res.len()).all(|r| cap.wire_kind(r) != 9 || !cap.users_of(r).iter().any(|u| u.line == l))
}

/// T-1273: taking the loop's stops out of a balloon left [A, M, A], a "loop" that runs out to M and
/// back the same way, reversing partway at M with the limiter; M is the line's end, a terminus.
#[test]
fn a_loop_that_only_goes_out_and_back_ends_where_it_turns() {
    let (w, [a, m, _, k]) = long_balloon();
    let mut api = api();
    api.w = w;
    let id = api.new_line_id();
    assert_eq!(api.set_line(id, &[a, m, k, m, a], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", api.issues());
    assert_eq!(api.line_loop(id), 2, "a balloon");
    // the line panel's cross on K: M twice in a row counts once
    assert_eq!(api.set_stops(id, &[a, m, m, a]), 0, "{}", api.issues());
    assert_eq!(api.w.net.line_stops(id), &[a, m]);
    assert_eq!(api.line_loop(id), 0);
    assert!(api.line_reversals(id).is_empty());
    assert!(no_limiter(&api, id));
    // closing it again runs out to K and back, quicker than round the loop: K is the end
    assert_eq!(api.set_stops(id, &[a, m, k, m, a]), 0, "{}", api.issues());
    assert_eq!(api.w.net.line_stops(id), &[a, m, k]);
    assert!(api.line_reversals(id).is_empty());
    assert!(no_limiter(&api, id));
    // a balloon that goes round its loop stays one, and so does one through a single stop
    let (w, [a, m, _, k]) = crate::track::tests::balloon_fixture();
    api.w = w;
    let id = api.new_line_id();
    assert_eq!(api.set_line(id, &[a, m, k, m, a], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", api.issues());
    assert_eq!(api.set_stops(id, &[a, m, m, a]), 0, "{}", api.issues());
    assert_eq!((api.w.net.line_stops(id), api.line_loop(id)), (&[a, m, a][..], 2));
    assert_eq!(api.set_stops(id, &[a, m, k, m, a]), 0, "{}", api.issues());
    assert_eq!((api.w.net.line_stops(id), api.line_loop(id)), (&[a, m, k, m, a][..], 2));
}
