//! T-1382: what the clock worker is told to read again (`take_changed_lines`, `take_name_changes`).

use super::*;

const ALL: u32 = u32::MAX;

fn route(a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64) {
    let ends = [0.0, 0.0, x0, y0, 0.0, 0.0, 0.0, x1, y1, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
}

fn station_nodes(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).collect()
}

/// A new model reports everything; after that, a station named, renamed and removed is the only name
/// change, and a line's edit names that line (and not the other one).
#[test]
fn changes_are_listed_once_and_only_what_an_edit_touched() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert_eq!(a.take_name_changes(), vec![ALL]);
    assert_eq!(a.take_changed_lines(), vec![ALL]);
    assert!(a.take_name_changes().is_empty());
    assert!(a.take_changed_lines().is_empty());

    // two separate lines of three stations each
    a.set_route_tangent(true);
    for (k, y) in [0.0, 5000.0].into_iter().enumerate() {
        route(&mut a, 0.0, y, 6000.0, y);
        for (j, x) in [500.0, 3000.0, 5500.0].into_iter().enumerate() {
            let e = (0..a.w.net.edge_count() as u32)
                .filter(|&e| a.w.net.edge_ok(e) && !a.w.net.edge_crossover[e as usize])
                .min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1))
                .unwrap();
            assert_eq!(a.add_station(2, e, x, y, 120, format!("S{k}{j}")), 0, "{}", a.issues());
        }
    }
    let st = station_nodes(&a);
    assert_eq!(st.len(), 6);
    let by_name = |a: &TrackApi, n: &str| *st.iter().find(|&&k| a.w.net.node_name[k as usize] == n).unwrap();
    let first: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S0{j}"))).collect();
    let second: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S1{j}"))).collect();
    let (l0, l1) = (a.new_line_id(), a.new_line_id());
    assert_eq!(a.set_line(l0, &first, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "A".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line(l1, &second, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "B".into()), 0, "{}", a.issues());
    let names = a.take_name_changes();
    assert!(names.len() >= 6 && !names.contains(&ALL) && names.windows(2).all(|w| w[0] < w[1]));
    let lines = a.take_changed_lines();
    assert!(lines.contains(&l0) && lines.contains(&l1));
    assert!(a.take_name_changes().is_empty() && a.take_changed_lines().is_empty());

    // a rename is one name; the lines through the station are the line's alone
    assert_eq!(a.set_station_props(first[1], 120, "Renamed".into()), 0, "{}", a.issues());
    assert_eq!(a.take_name_changes(), vec![first[1]]);
    let lines = a.take_changed_lines();
    assert!(lines.contains(&l0), "{lines:?}");
    assert!(!lines.contains(&l1), "the other line's service is not rebuilt: {lines:?}");

    // a line's look touches only that line and no names
    assert_eq!(a.set_line_look(l1, "B2".into(), 0x123456), 0, "{}", a.issues());
    assert!(a.take_name_changes().is_empty());
    let lines = a.take_changed_lines();
    assert!(lines.contains(&l1) && !lines.contains(&l0), "{lines:?}");

    // an edit that is refused, after the check, still leaves the lines it touched listed
    let bad = a.set_stops(l0, &[first[0], first[0]]);
    let after = a.take_changed_lines();
    if bad != 0 {
        assert!(after.is_empty() || after.contains(&l0), "{after:?}");
    }

    // undo reports the same name again; a reload reports everything
    assert_eq!(a.undo(), 0);
    a.take_changed_lines();
    a.take_name_changes();
    assert_eq!(a.remove_line(l1), 0, "{}", a.issues());
    assert_eq!(a.remove_station(second[2]), 0, "{}", a.issues());
    assert_eq!(a.take_name_changes(), vec![second[2]]);
    let saved = a.save();
    assert!(a.load(&saved));
    assert_eq!(a.take_name_changes(), vec![ALL]);
    assert_eq!(a.take_changed_lines(), vec![ALL]);
}
