use super::*;

#[test]
fn a_copied_express_runs_beside_its_local() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && c[2].abs() < 1.0).unwrap()[0] as u32;
    let (w, e) = (node_at(&a, 0.0), node_at(&a, 6000.0));
    for s in [w, e] {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let edge = a.edges_info()[0] as u32;
    assert_eq!(a.add_station(2, edge, 3000.0, 0.0, 200, "M".into()), 0, "{}", a.issues());
    let m = node_at(&a, 3000.0);
    let local = a.new_line_id();
    assert_eq!(a.set_line(local, &[w, m, e], 12.0, 6.0, 3.0, 30.0, 180.0, 0, 0xff0000, "Local".into()), 0, "{}", a.issues());
    let express = a.new_line_id();
    assert_eq!(a.set_line(express, &[w, e], 6.0, 3.0, 2.0, 30.0, 180.0, 0, 0xff0000, "Express".into()), 0, "{}", a.issues());
    let ni = a.nodes_info();
    let max = ni.chunks(10).map(|c| c[0] as usize).max().unwrap();
    let mut cx = vec![u32::MAX; max + 1];
    for c in ni.chunks(10).filter(|c| c[7] > 0.0) {
        cx[c[0] as usize] = c[0] as u32;
    }
    let v = a.line_related(local, &cx);
    // one group: the express, beside, on the same track, over the whole line, three meetings
    assert_eq!((v[0], v[1], v[2], v[3]), (express as f64, 1.0, 1.0, 1.0), "{v:?}");
    assert!(v[4].abs() < 1.0 && (v[5] - 6000.0).abs() < 5.0, "{v:?}");
    assert_eq!((v[13], v[14], v[15], v[16]), (3.0, 1.0, 0.0, 0.0), "{v:?}");
    assert_eq!(v.len(), 17 + 3 * 5 + 2, "{v:?}"); // no own stops
    // the middle station: the local stops, the express passes
    assert_eq!((v[17 + 5 + 3], v[17 + 5 + 4]), (1.0, -1.0));
    let back = a.line_related(express, &cx);
    assert_eq!((back[0], back[1]), (local as f64, 1.0));
    assert!(a.line_related(99, &cx).is_empty());
}
