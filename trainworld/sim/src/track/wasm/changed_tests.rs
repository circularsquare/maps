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
    assert_eq!(a.set_line_with_trains(l0, &first, 6, 4, 2, 30.0, 180.0, 0, 0, "A".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_with_trains(l1, &second, 6, 4, 2, 30.0, 180.0, 0, 0, "B".into()), 0, "{}", a.issues());
    let names = a.take_name_changes();
    assert!(names.len() >= 6 && !names.contains(&ALL) && names.windows(2).all(|w| w[0] < w[1]));
    let lines = a.take_changed_lines();
    assert!(lines.contains(&l0) && lines.contains(&l1));
    assert!(a.take_name_changes().is_empty() && a.take_changed_lines().is_empty());

    // a rename is one name; T-4113: no line's rows hold a station's name, so no line is listed (the line through it is
    // noted, and left out as its stamp is unchanged)
    assert_eq!(a.set_station_props(first[1], 120, "Renamed".into()), 0, "{}", a.issues());
    assert_eq!(a.take_name_changes(), vec![first[1]]);
    let lines = a.take_changed_lines();
    assert!(!lines.contains(&l0) && !lines.contains(&l1), "{lines:?}");
    // a stop dropped moves the line's rows: it is listed, the other line not
    assert_eq!(a.set_stops(l0, &first[..2]), 0, "{}", a.issues());
    let lines = a.take_changed_lines();
    assert!(lines.contains(&l0) && !lines.contains(&l1), "{lines:?}");
    assert_eq!(a.undo(), 0);
    a.take_changed_lines();

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

/// T-4231: a weekend edit lists every line whose weekend setting changed, even where its weekend
/// counts round up to the same numbers as weekdays (so its weekend rates and phases do not move).
#[test]
fn weekend_edits_list_lines_whose_rates_do_not_move() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.take_changed_lines();
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
    let by_name = |a: &TrackApi, n: &str| *st.iter().find(|&&k| a.w.net.node_name[k as usize] == n).unwrap();
    let first: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S0{j}"))).collect();
    let second: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S1{j}"))).collect();
    let (big, small) = (a.new_line_id(), a.new_line_id());
    assert_eq!(a.set_line_with_trains(big, &first, 6, 4, 2, 30.0, 180.0, 0, 0, "Big".into()), 0, "{}", a.issues());
    // 70% of 3/2/1 rounds up to 3/2/1: weekend rates and phases stay as on weekdays
    assert_eq!(a.set_line_with_trains(small, &second, 3, 2, 1, 30.0, 180.0, 0, 0, "Small".into()), 0, "{}", a.issues());
    a.take_changed_lines();

    // Apply to all lines at 70%: both lines must be listed, and again on undo
    assert_eq!(a.set_weekend_share_all(70), 0, "{}", a.issues());
    let lines = a.take_changed_lines();
    assert!(lines.contains(&big) && lines.contains(&small), "apply all missed a line: {lines:?}");
    assert_eq!(a.undo(), 0);
    let lines = a.take_changed_lines();
    assert!(lines.contains(&big) && lines.contains(&small), "undo missed a line: {lines:?}");

    // Single-line share and manual counts matching weekdays still list the line
    assert_eq!(a.set_line_weekend_share(small, 70), 0, "{}", a.issues());
    assert_eq!(a.take_changed_lines(), vec![small]);
    assert_eq!(a.set_line_weekend_counts(small, 3, 2, 1), 0, "{}", a.issues());
    assert_eq!(a.take_changed_lines(), vec![small]);
}

/// T-4149: `take_moved_lines` lists every line whose runs lie anywhere new on the track (a new model: every line), and
/// a line it leaves out is sampled where it was, bit for bit, whatever was rebuilt or worked out again: a rename at a
/// stop, a train count, a stop dropped (that line listed, the other not), a station on the other line's track, undo; and
/// none after the construct-all quote, whose trial (`trial_with`) and rollback leave every line where it ran.
#[test]
fn lines_left_off_the_moved_list_run_where_they_ran() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
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
    let by_name = |a: &TrackApi, n: &str| *st.iter().find(|&&k| a.w.net.node_name[k as usize] == n).unwrap();
    let first: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S0{j}"))).collect();
    let second: Vec<u32> = (0..3).map(|j| by_name(&a, &format!("S1{j}"))).collect();
    let (l0, l1) = (a.new_line_id(), a.new_line_id());
    assert_eq!(a.set_line_with_trains(l0, &first, 6, 4, 2, 30.0, 180.0, 0, 0, "A".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_with_trains(l1, &second, 6, 4, 2, 30.0, 180.0, 0, 0, "B".into()), 0, "{}", a.issues());
    assert_eq!(a.take_moved_lines(), vec![ALL], "a new model: every line");
    let drawn = |a: &TrackApi| [l0, l1].map(|l| (0..2).map(|r| a.run_polyline(l, r, 50.0).iter().chain(&a.run_bends(l, r, 50.0, 0.25)).map(|x| x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>());
    let mut was = drawn(&a);
    let mut step = |a: &mut TrackApi, what: &str, r: u32| -> Vec<u32> {
        assert_eq!(r, 0, "{what}: {}", a.issues());
        a.settle_owed(); // T-4272: a station or delete answered first has its services after
        let moved = a.take_moved_lines();
        let now = drawn(a);
        for (k, l) in [l0, l1].into_iter().enumerate() {
            if !moved.contains(&l) && !moved.contains(&ALL) { assert_eq!(now[k], was[k], "{what}: line {l} left off the list ({moved:?}) but drawn elsewhere"); }
        }
        was = now;
        moved
    };
    let r = a.set_station_props(first[1], 120, "Renamed".into());
    step(&mut a, "a rename", r);
    let r = a.set_train_counts(l1, 3, 2, 1);
    step(&mut a, "a train count", r);
    let r = a.set_stops(l0, &first[..2]);
    let moved = step(&mut a, "a stop dropped", r);
    assert!(moved.contains(&l0) && !moved.contains(&l1), "{moved:?}");
    let e = (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e) && !a.w.net.edge_crossover[e as usize]).min_by(|&p, &q| a.w.net.project(p, 4200.0, 5000.0).1.total_cmp(&a.w.net.project(q, 4200.0, 5000.0).1)).unwrap();
    let r = a.add_station(2, e, 4200.0, 5000.0, 120, "Mid".into());
    step(&mut a, "a station on the other line's track", r);
    let r = a.undo();
    step(&mut a, "undo", r);
    let r = a.undo();
    let moved = step(&mut a, "undo of the stop dropped", r);
    assert!(moved.contains(&l0), "{moved:?}");
    assert!(a.w.svc.lines.iter().any(|s| s.ok && !s.running), "the lines are planned, so the quote tries constructing them");
    assert!(!a.blueprint_cost_items().is_empty());
    let moved = step(&mut a, "a construct quote", 0);
    assert!(moved.is_empty(), "{moved:?}");
}

/// T-4281: a station on a line's track moves that line's runs only around it (`take_moved_spans`): outside the span its
/// samples stay within the span's drift of a whole resample, the ranged calls give the whole run's numbers bit for bit,
/// and two stations before one take give one span over both.
#[test]
fn a_station_on_a_lines_track_moves_its_runs_only_around_it() {
    const STEP: f64 = 50.0;
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    // a straight, a 600 m curve, a straight
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 4000.0, 0.0];
    assert_eq!(a.add_route(&ends, &[4000.0, 0.0, 600.0, 0.0], 2, false), 0, "{}", a.issues());
    let near = |a: &TrackApi, x: f64, y: f64| {
        (0..a.w.net.edge_count() as u32)
            .filter(|&e| a.w.net.edge_ok(e) && !a.w.net.edge_crossover[e as usize])
            .min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1))
            .unwrap()
    };
    let station = |a: &mut TrackApi, x: f64, y: f64, name: &str| {
        let e = near(a, x, y);
        assert_eq!(a.add_station(2, e, x, y, 120, name.into()), 0, "{}", a.issues());
        a.settle_owed();
    };
    station(&mut a, 300.0, 0.0, "A");
    station(&mut a, 4000.0, 3700.0, "B");
    let st = station_nodes(&a);
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    a.settle_owed();
    // stations the line passes, so its runs have several entries (a run of one entry split is all new)
    for (x, y, n) in [(1000.0, 0.0, "P"), (2000.0, 0.0, "Q"), (3000.0, 0.0, "R"), (4000.0, 2000.0, "S")] {
        station(&mut a, x, y, n);
    }
    a.take_moved_lines();
    a.take_moved_spans();
    let whole = |a: &TrackApi, r: usize| (a.run_polyline(l, r, STEP), a.run_bends(l, r, STEP, 0.25));
    let entries = |b: &[f64], keep: &dyn Fn(usize) -> bool| {
        let (mut out, mut i) = (vec![], 0);
        while i < b.len() {
            let m = b[i + 1] as usize;
            if keep(b[i] as usize) {
                out.extend_from_slice(&b[i..i + 2 + 2 * m]);
            }
            i += 2 + 2 * m;
        }
        out
    };
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for r in 0..2 {
        let (p, b) = whole(&a, r);
        let n = p.len() / 2 - 1;
        assert!(!b.is_empty(), "run {r}: the curve bends between samples");
        for (k0, k1) in [(0, n), (3, 40), (n - 5, n), (40, 90)] {
            assert_eq!(bits(&a.run_polyline_range(l, r, STEP, k0, k1)), bits(&p[2 * k0..2 * (k1 + 1)]), "run {r}: samples {k0}..={k1}");
            assert_eq!(bits(&a.run_bends_range(l, r, STEP, 0.25, k0, k1)), bits(&entries(&b, &|k| k >= k0 && k < k1)), "run {r}: bends {k0}..{k1}");
        }
    }
    let check = |a: &mut TrackApi, before: &[(Vec<f64>, Vec<f64>)], at: &[f64], what: &str| {
        assert_eq!(a.take_moved_lines(), vec![l], "{what}");
        let spans = a.take_moved_spans();
        assert_eq!(spans.len(), 12, "{what}: a span for each run: {spans:?}");
        for s in spans.chunks(6) {
            let (r, lo, hi, drift) = (s[1] as usize, s[2], s[3], s[4]);
            let (p, b) = whole(a, r);
            let n = p.len() / 2 - 1;
            assert_eq!(p.len(), before[r].0.len(), "{what}");
            let (k0, k1) = (((lo / STEP).floor() as usize).saturating_sub(1), ((hi / STEP).ceil() as usize + 1).min(n));
            // (the entries beside a split edge count as touched too)
            assert!(n - (k1 - k0) >= 30, "{what}: run {r} moved only near the stations, samples {k0}..={k1} of {n}");
            assert!(drift < 0.01, "{what}: {drift}");
            // the stations' places, in this run's offsets, inside the span
            let len = a.w.svc.lines[l as usize].runs[r].len;
            assert_eq!(s[5].to_bits(), len.to_bits(), "{what}: the run's length");
            for &x in at {
                let s = if r == 0 { x - 300.0 } else { len - (x - 300.0) };
                assert!(s >= lo - STEP && s <= hi + STEP, "{what}: run {r}: a station at {s} m, the span {lo}..{hi}");
            }
            for k in (0..=n).filter(|&k| k < k0 || k > k1) {
                let d = (p[2 * k] - before[r].0[2 * k]).hypot(p[2 * k + 1] - before[r].0[2 * k + 1]);
                assert!(d <= drift + 1e-9, "{what}: run {r} sample {k} moved {d} m, its span's drift {drift}");
            }
            let outside = |k: usize| k + 1 < k0 || k > k1;
            let (x, y) = (entries(&b, &outside), entries(&before[r].1, &outside));
            assert_eq!(x.len(), y.len(), "{what}: run {r}: the bends outside the span");
            for (u, v) in x.iter().zip(&y) {
                assert!((u - v).abs() <= drift + 1e-9, "{what}: run {r}: a bend outside the span moved {}", (u - v).abs());
            }
        }
    };
    let before: Vec<_> = (0..2).map(|r| whole(&a, r)).collect();
    station(&mut a, 1500.0, 0.0, "Mid");
    check(&mut a, &before, &[1500.0], "a station on the line's track");
    let before: Vec<_> = (0..2).map(|r| whole(&a, r)).collect();
    for (x, n) in [(1200.0, "C"), (2200.0, "D")] {
        let e = near(&a, x, 0.0);
        assert_eq!(a.add_station(2, e, x, 0.0, 120, n.into()), 0, "{}", a.issues());
        a.settle_owed();
    }
    check(&mut a, &before, &[1200.0, 2200.0], "two stations before one take");
    // a whole pass knows no span
    a.w.svc.moved = Some(vec![]);
    let r = a.undo();
    assert_eq!(r, 0, "{}", a.issues());
    a.settle_owed();
    let moved = a.take_moved_lines();
    let spans = a.take_moved_spans();
    assert!(spans.chunks(6).all(|s| moved.contains(&(s[0] as u32))), "{spans:?} of {moved:?}");
}
