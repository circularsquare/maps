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

/// T-1674: S0 - S1 - T - J - M along x, and a branch J - B1 - B2 - B3 leaving J towards M (from T
/// a train may take either; from M to the branch it has to turn back).
fn y_fixture() -> (TrackWorld, [u32; 8]) {
    let mut w = world();
    let s0 = node(&mut w, -6000.0, 0.0, 0);
    let s1 = node(&mut w, -3000.0, 0.0, 0);
    let t = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let m = node(&mut w, 6000.0, 0.0, 0);
    let b1 = node(&mut w, 5000.0, 1500.0, 0);
    let b2 = node(&mut w, 6600.0, 3000.0, 0);
    let b3 = node(&mut w, 8200.0, 4500.0, 0);
    edge(&mut w, s0, s1, &[], 0).unwrap();
    edge(&mut w, s1, t, &[], 0).unwrap();
    edge(&mut w, t, j, &[], 0).unwrap();
    edge(&mut w, j, m, &[], 0).unwrap();
    edge(&mut w, j, b1, &[(3400.0, 0.0)], 0).unwrap();
    edge(&mut w, b1, b2, &[], 0).unwrap();
    edge(&mut w, b2, b3, &[], 0).unwrap();
    for n in [s0, s1, t, m, b1, b2, b3] {
        station(&mut w, n, 200);
    }
    (w, [s0, s1, t, j, m, b1, b2, b3])
}

/// The stations a line's path runs through in order, stops or not, and where it turns back.
fn path_stations(a: &TrackApi, l: u32) -> (Vec<u32>, Vec<u32>) {
    let net = &a.w.net;
    let path = net.line_path(l);
    let nodes = std::iter::once(net.path_nodes(path[0]).0).chain(path.iter().map(|&p| net.path_nodes(p).1));
    (nodes.filter(|&n| net.is_station(n)).collect(), turn_nodes(net, path))
}

/// T-1674: stops added along a branch go in route order. Out along the trunk and up the branch
/// the line runs straight through; on a copy that also keeps a stop on the Y's other arm (Anita's
/// "Line 1 copy"), the branch stops are served on the pass that leads on to the rest of the line,
/// and the train turns back only at the far one, which the stops require.
#[test]
fn branch_stops_go_in_route_order() {
    let (w, [s0, s1, t, _, m, b1, b2, b3]) = y_fixture();
    let mut a = api();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[s0, t], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    for s in [b3, b1, b2, s1] {
        let stops: Vec<u32> = a.insert_stop(id, s).iter().map(|&x| x as u32).collect();
        assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    }
    assert_eq!(a.w.net.line_stops(id), &[s0, s1, t, b1, b2, b3]);
    assert_eq!(path_stations(&a, id), (vec![s0, s1, t, b1, b2, b3], vec![]));
    // a copy with M (on the other arm) at one end and the trunk's end taken off, then the trunk
    // stops added back from the branch point outwards
    let copy = a.new_line_id();
    assert_eq!(a.copy_line(id, copy, "L copy".into()), 0, "{}", a.issues());
    assert_eq!(a.set_stops(copy, &[m, t, b1, b2, b3]), 0, "{}", a.issues());
    for s in [s1, s0] {
        let stops: Vec<u32> = a.insert_stop(copy, s).iter().map(|&x| x as u32).collect();
        assert_eq!(a.set_stops(copy, &stops), 0, "{}", a.issues());
    }
    assert_eq!(a.w.net.line_stops(copy), &[m, s0, s1, t, b1, b2, b3]);
    assert_eq!(path_stations(&a, copy), (vec![m, t, s1, s0, s1, t, b1, b2, b3], vec![s0]));
    assert_eq!(a.line_reversals(copy), [1.0]);
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
