//! T-1447: a pick copied out of one game and pasted into another as blueprint (`wasm/copy.rs`).

use super::*;

/// A line along y = `y` with a station every `step` m up to `km` km, constructed or not; the line and its stations.
fn rail(a: &mut TrackApi, km: usize, step: f64, y: f64, built: bool) -> (u32, Vec<u32>) {
    let n = (km as f64 * 1000.0 / step).round() as usize;
    let mut st: Vec<u32> = vec![];
    for i in 0..n {
        let from = if i == 0 { [0.0, 0.0, 0.0, y, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * step, y, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            let ni = a.nodes_info();
            let first = ni.chunks_exact(10).find(|c| c[1].abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
            st.push(first);
        }
        st.push(a.route_end());
    }
    for (i, &s) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 120, format!("S{y}-{i}")), 0, "{}", a.issues());
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, format!("L{y}")), 0, "{}", a.issues());
    if built {
        assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    }
    (l, st)
}

fn edges(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn stations(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.is_station(n)).collect()
}
fn lines(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.line_count() as u32).filter(|&l| a.w.net.line_ok(l)).collect()
}

/// Built track and stations copy as blueprint, the line wholly on them with them, the line leaving them not; the
/// paste is one undo step.
#[test]
fn a_pick_pastes_into_another_game_as_blueprint() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (la, sa) = rail(&mut a, 8, 2000.0, 0.0, true);
    let picked_edges: Vec<u32> = edges(&a);
    // a second line on the same track, then on to track outside the pick
    let (_, sb) = rail(&mut a, 4, 2000.0, 5000.0, false);
    let lb = a.new_line_id();
    assert_eq!(a.set_line_with_trains(lb, &[sa[0], sa[1]], 2, 2, 2, 30.0, 180.0, 0, 0x2d7dd2, "B".into()), 0, "{}", a.issues());
    let lc = a.new_line_id();
    assert_eq!(a.set_line_with_trains(lc, &[sb[0], sb[1]], 2, 2, 2, 30.0, 180.0, 0, 0x2d7dd2, "C".into()), 0, "{}", a.issues());
    let clip = a.clip(&picked_edges, &sa);
    assert!(!clip.is_empty());
    assert_eq!(a.clip_left_out(), 0);
    assert_eq!(TrackApi::clip_check(&clip, save::LEVEL), 0);

    let mut b = TrackApi::new(-73.985, 40.758, true);
    b.set_cash(1e6);
    let (_, other) = rail(&mut b, 4, 2000.0, -8000.0, false);
    let (e0, s0, l0) = (edges(&b).len(), stations(&b).len(), lines(&b).len());
    assert_eq!(b.paste(&clip, save::LEVEL), 0, "{}", b.issues());
    let p = b.last_paste();
    let ne = p[0] as usize;
    let ns = p[1 + ne] as usize;
    let nl = p[2 + ne + ns] as usize;
    assert_eq!((ne, ns, nl, *p.last().unwrap()), (picked_edges.len(), sa.len(), 2, 0), "the track, its stations and lines A and B");
    assert_eq!(edges(&b).len(), e0 + ne);
    assert_eq!(stations(&b).len(), s0 + ns);
    assert_eq!(lines(&b).len(), l0 + 2);
    for &e in &p[1..1 + ne] {
        assert!(!b.w.net.edge_built[e as usize], "pasted track is blueprint");
    }
    for &n in &p[2 + ne..2 + ne + ns] {
        let s = b.w.net.station_data(n).unwrap();
        assert!(!s.built && s.name.starts_with("S0-"), "pasted stations are blueprint, with their names");
    }
    let names: Vec<String> = p[3 + ne + ns..3 + ne + ns + nl].iter().map(|&l| b.w.net.line_data(l).unwrap().name).collect();
    assert!(names.contains(&"L0".to_string()) && names.contains(&"B".to_string()), "{names:?}");
    let la_b = p[3 + ne + ns..3 + ne + ns + nl].iter().copied().find(|&l| b.w.net.line_data(l).unwrap().name == "L0").unwrap();
    assert_eq!(b.w.net.line_data(la_b).unwrap().trains, a.w.net.line_data(la).unwrap().trains, "the line keeps its trains");
    assert_eq!(b.w.net.line_stops(la_b).len(), sa.len());
    // one undo takes it all back, the game's own line kept
    assert_eq!(b.undo(), 0, "{}", b.issues());
    assert_eq!((edges(&b).len(), stations(&b).len(), lines(&b).len()), (e0, s0, l0));
    assert!(b.w.net.is_station(other[0]));
}

/// Pasted onto the same track (into another version of the game), nothing doubles.
#[test]
fn a_clip_over_the_same_track_adds_nothing() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (_, sa) = rail(&mut a, 6, 2000.0, 0.0, false);
    let clip = a.clip(&edges(&a), &sa);
    let (e0, s0, l0) = (edges(&a).len(), stations(&a).len(), lines(&a).len());
    assert_ne!(a.paste(&clip, save::LEVEL), 0);
    assert_eq!((edges(&a).len(), stations(&a).len(), lines(&a).len()), (e0, s0, l0));
    assert_eq!(*a.last_paste().last().unwrap() as usize, e0 + s0);
}

/// A picked station whose track is not picked is left out of the clip, and said so.
#[test]
fn a_station_without_its_track_is_left_out() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (_, sa) = rail(&mut a, 6, 2000.0, 0.0, false);
    let clip = a.clip(&[], &sa[1..2]);
    assert!(clip.is_empty());
    assert_eq!(a.clip_left_out(), 1);
}

/// A clip from a newer build waits: its level is above this build's.
#[test]
fn a_newer_clip_is_not_pasted() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (_, sa) = rail(&mut a, 4, 2000.0, 0.0, false);
    let clip = a.clip(&edges(&a), &sa);
    assert_eq!(TrackApi::clip_check(&clip, save::LEVEL + 1), 1);
    assert_eq!(TrackApi::clip_check(&clip[..clip.len() / 2], save::LEVEL), 2);
}
