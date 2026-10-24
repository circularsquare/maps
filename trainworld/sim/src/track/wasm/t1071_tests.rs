use super::*;

/// A built line between two stations 6 km apart with two trains at every level, its id.
fn line_api() -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, e) = (node(&a, 0.0), node(&a, 6000.0));
    for (n, x) in [(w, 0.0), (e, 6000.0)] {
        assert_eq!(a.add_station(1, n, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let edges: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&k| a.w.net.edge_ok(k)).collect();
    assert_eq!(a.construct(&edges, &[w, e]), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[w, e], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 2, 2, 2), 0, "{}", a.issues());
    (a, id)
}

/// T-1071: an edit with a lead takes effect that much later: the trips drawn before then are
/// the old service's to the bit, and a save made before the change is over loads with it.
#[test]
fn a_count_change_with_a_lead_takes_effect_then_and_saves_with_its_eras() {
    let (mut a, id) = line_api();
    let now: f64 = 2.0 * 86400.0 + 11.0 * 3600.0 + 417.0;
    let epoch = (now / 3600.0).floor() * 3600.0;
    a.set_now(now);
    let old = a.trips(epoch);
    a.set_edit_lead(300.0);
    assert_eq!(a.set_train_counts(id, 2, 3, 2), 0, "{}", a.issues());
    assert_eq!(a.edit_time(), now + 300.0);
    // the lead is used once
    a.set_now(now + 1.0);
    assert_eq!(a.set_train_counts(id, 2, 3, 2), 0);
    assert_eq!(a.edit_time(), now + 1.0);
    let new = a.trips(epoch);
    // trips that left before the change are the same ones (draw, as the fourth number may
    // name a later departure), and none of the new pattern leaves before it
    let left = |v: &[f32]| v.chunks(4).filter(|t| (t[1] as f64) + epoch < now + 300.0 - 1.0).map(|t| (t[0], t[1])).collect::<Vec<_>>();
    assert_eq!(left(&old), left(&new));
    assert_ne!(old, new, "the change shows after the lead");
    // a save made now loads with the change still going on
    let bytes = a.save();
    assert_eq!(&bytes[bytes.len() - 4..], ERAS_TAG);
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    b.set_now(now + 1.0);
    assert_eq!(b.w.svc.eras, a.w.svc.eras);
    let (x, y) = (a.trips(epoch), b.trips(epoch));
    assert_eq!(x.len(), y.len());
    assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 0.01 || (p.is_nan() && q.is_nan())));
    // and a save with no change going on ends as before
    b.set_now(now + 6.0 * 3600.0);
    assert!(b.w.svc.eras.iter().all(|e| e.is_empty()));
    assert_eq!(b.save(), save::encode(&b.w.net));
}
