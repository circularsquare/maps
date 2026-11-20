use super::*;

#[test]
fn an_express_added_over_constructed_track_runs() {
    // T-849: five stations 10 km apart, an all-stops line constructed, then an express
    // calling at every other station over the same track
    let mut a = api();
    a.set_cash(1e6);
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    let mut st = vec![];
    for i in 0..4 {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * 10_000.0, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 400, format!("S{i}")), 0, "{}", a.issues());
    }
    let all = a.new_line_id();
    assert_eq!(a.set_line_with_trains(all, &st, 4, 3, 1, 30.0, 180.0, 0, 0, "All".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_fast(all, true), 0, "{}", a.issues());
    assert_eq!(a.set_line_cars(all, 10), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(all, 900, 1200, 3600), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.w.svc.lines[all as usize].running);
    let ex = a.new_line_id();
    // as the app adds a line: no trains, then straight to headways (the edit API's setHeadway)
    assert_eq!(a.set_line_with_trains(ex, &[st[0], st[2], st[4]], 0, 0, 0, 30.0, 180.0, 0, 0, "Ex".into()), 0, "{}", a.issues());
    assert!(a.w.svc.lines[ex as usize].running, "over constructed track the line runs at once");
    assert_eq!(a.set_line_fast(ex, true), 0, "{}", a.issues());
    assert_eq!(a.set_line_cars(ex, 10), 0, "{}", a.issues());
    // (it turns where the all-stops line does, so takes its time, T-1965: a medium headway well
    // clear of a whole number of trains)
    assert_eq!(a.set_auto_headway(ex, 900, 1500, 0), 0, "{}", a.issues());
    let (trip, tph) = (slots_to_rows(a.w.svc.lines[ex as usize].round_trip), slots_to_rows(a.w.net.line_tph[ex as usize]));
    assert_eq!(slots_to_rows(a.w.svc.lines[ex as usize].trains), [auto_trains(trip[0], 900), auto_trains(trip[1], 1500), 0]);
    assert!(tph[0] > 3.9 && tph[1] > 2.3 && tph[2] == 0.0, "{tph:?}");
    // nothing to construct: refused, and the issue says so
    assert_eq!(a.construct_line(ex), 1);
    assert_eq!(a.issue_kinds(), "NothingToConstruct");
    assert_eq!(a.construct_all(), 1);
    assert_eq!(a.issue_kinds(), "NothingToConstruct");
    assert_eq!(slots_to_rows(a.w.net.line_tph[ex as usize]), tph);
}

#[test]
fn draft_line_copy_symbol_save_and_undo() {
    let mut a = api();
    let id = a.new_line_id();
    let name = a.next_line_name();
    assert_eq!(name, "Line 1");
    assert_eq!(a.set_line_with_trains(id, &[], 0, 0, 0, 30.0, 180.0, 0, 0xd7263d, name), 0);
    assert!(!a.w.net.line_broken[id as usize]);
    assert!(!a.w.svc.lines[id as usize].running);
    assert_eq!(a.lines_info()[5..8], [0.0; 3]);
    assert_eq!(a.set_line_badge(id, "Blue".into(), 0x123456, "AB".into()), 0);
    let copied = a.new_line_id();
    assert_eq!(a.copy_line(id, copied, "Blue copy".into()), 0);
    assert_eq!(a.line_symbol(copied), "AB");
    assert_eq!(a.set_line_look(copied, "Branch".into(), 0xffffff), 0);
    assert_eq!(a.line_name(id), "Blue");
    assert_eq!(a.line_symbol(copied), "AB");
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_name(copied), "Blue copy");
    assert_eq!(a.undo(), 0);
    assert!(!a.w.net.line_ok(copied));
    assert_eq!(a.redo(), 0);
    assert!(a.w.net.line_ok(copied));
    let bytes = a.save();
    let saved = save::decode(&bytes).unwrap();
    assert_eq!(saved.line_symbol, ["AB", "AB"]);
    assert_eq!(saved.next_line_number, 3);
    assert!(saved.line_stops(0).is_empty());
    // Core TWT4 without extension is still accepted, custom symbols default to initials.
    let extension = bytes.windows(4).rposition(|w| w == b"LUI1").unwrap();
    let old = save::decode(&bytes[..extension]).unwrap();
    assert_eq!(old.line_symbol, ["", ""]);
    assert_eq!(old.next_line_number, 3);
    assert!(save::decode(&bytes[..bytes.len()-1]).is_err(), "truncated known extension refused");
    assert_eq!(a.remove_line(id), 0);
    let reused = a.new_line_id();
    assert_eq!(a.next_line_name(), "Line 3", "creation numbering survives id reuse");
    assert_eq!(a.set_line_with_trains(reused, &[], 0, 0, 0, 30.0, 180.0, 0, 0, "Line 4".into()), 0);
    assert_eq!(a.next_line_name(), "Line 5", "taken seeded names skipped");
}

#[test]
fn insert_stop_infill_ends_and_reduce_to_empty() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    for x in [1000.0, 2000.0, 4000.0, 5000.0] {
        let edge = (0..a.w.net.edge_count() as u32).find(|&e| {
            let n = &a.w.net;
            n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
        }).unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    let station = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    let (left, first, second, right) = (station(&a, 1000.0), station(&a, 2000.0), station(&a, 4000.0), station(&a, 5000.0));
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[first, right], 0, 0, 0, 30.0, 180.0, 0, 0, "Test".into()), 0);
    assert_eq!(a.insert_stop(id, second), [first as f64, second as f64, right as f64]);
    assert_eq!(a.insert_stop(id, left), [left as f64, first as f64, right as f64]);
    assert_eq!(a.set_stops(id, &[first, second]), 0);
    assert_eq!(a.insert_stop(id, right), [first as f64, second as f64, right as f64]);
    assert!(a.insert_stop(id, first).is_empty());
    assert!(a.insert_stop(id, u32::MAX).is_empty());
    assert_eq!(a.construct_all(), 0);
    assert_eq!(a.set_train_counts(id, 2, 1, 0), 0);
    let copy = a.new_line_id();
    assert_eq!(a.copy_line(id, copy, "Copy".into()), 0);
    assert_eq!(a.w.net.line_data(copy).unwrap().trains, Some([0, 0, 0]), "a copy starts with no trains (T-1161)");
    assert_eq!(slots_to_rows(a.w.svc.lines[copy as usize].trains), [0, 0, 0]);
    assert_eq!(slots_to_rows(a.w.svc.lines[id as usize].trains), [2, 1, 0], "the source keeps its trains");
    assert_eq!(a.set_auto_headway(id, 600, 900, 0), 0);
    let copy2 = a.new_line_id();
    assert_eq!(a.copy_line(id, copy2, "Copy 2".into()), 0);
    assert_eq!(a.line_auto(copy2), [0.0; 3], "a headway line's copy keeps headways, at none");
    assert_eq!(slots_to_rows(a.w.svc.lines[copy2 as usize].trains), [0, 0, 0]);
    assert_eq!(a.remove_line(copy2), 0);
    assert_eq!(a.clear_auto_headway(id), 0);
    assert_eq!(a.set_stops(id, &[first]), 0);
    assert_eq!(a.line_stops(copy), [first, second], "copy stops independent");
    assert!(!a.w.net.line_broken[id as usize]);
    assert!(!a.w.svc.lines[id as usize].running);
    assert_eq!(a.set_stops(id, &[]), 0);
    assert_eq!(a.insert_stop(id, left), [left as f64]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_stops(id), [first]);
    assert_eq!(a.redo(), 0);
    assert!(a.line_stops(id).is_empty());
    let loaded = super::super::super::world::TrackWorld::new(save::decode(&a.save()).unwrap());
    assert!(!loaded.svc.lines[0].running);
    assert!(loaded.net.line_stops(0).is_empty());
}

/// T-1242: stops left at one station (a loop's middle stop taken out, say) make a one-stop line,
/// inactive like a line with none, never a refusal.
#[test]
fn stops_at_one_station_make_a_one_stop_line() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    for x in [1000.0, 5000.0] {
        let edge = (0..a.w.net.edge_count() as u32).find(|&e| {
            let n = &a.w.net;
            n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
        }).unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    let station = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    let (first, last) = (station(&a, 1000.0), station(&a, 5000.0));
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[first, last], 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0);
    assert!(a.w.svc.lines[id as usize].running);
    assert_eq!(a.set_stops(id, &[first, first]), 0, "{}", a.issues());
    assert_eq!(a.line_stops(id), [first]);
    assert!(!a.w.net.line_broken[id as usize]);
    assert!(!a.w.svc.lines[id as usize].running);
    assert_eq!(a.set_stops(id, &[first, last, last]), 0, "{}", a.issues());
    assert_eq!(a.line_stops(id), [first, last]);
    let lone = a.new_line_id();
    assert_eq!(a.set_line_with_trains(lone, &[last, last], 2, 1, 0, 30.0, 180.0, 0, 0, "Lone".into()), 0, "{}", a.issues());
    assert_eq!(a.line_stops(lone), [last]);
    assert!(!a.w.svc.lines[lone as usize].running);
}

#[test]
fn crossing_and_junction_marks() {
    let mut a = api();
    let line = |x0: f64, y0: f64, x1: f64, y1: f64| -> Vec<f64> { free(x0, y0).iter().chain(free(x1, y1).iter()).copied().collect() };
    assert_eq!(a.add_route(&line(0.0, 0.0, 1000.0, 0.0), &[], 2, false), 0);
    // a route across it: the preview names the crossing, nothing is made
    assert_eq!(a.preview_route(&line(500.0, -300.0, 500.0, 300.0), &[], 2, false)[0], 1.0);
    let m = a.preview_marks();
    assert_eq!(m.len(), 4, "{m:?}");
    assert!((m[0] - 500.0).abs() < 1.0 && m[1].abs() < 1.0 && m[2] == 0.0 && m[3] == 0.0, "rail over rail: no trams {m:?}");
    assert!(a.crossings_info().is_empty());
    assert_eq!(a.add_route(&line(500.0, -300.0, 500.0, 300.0), &[], 2, false), 0);
    let c = a.crossings_info();
    assert_eq!(c.len(), 6);
    assert_eq!(&c[2..], &[0.0, 0.0, 0.0, 0.0], "a flat crossing at level 0, not built, no trams");
    // T-3535: tram track across rail: the preview's crossing counts the tram as the placed one does
    a.set_route_kind(TrackKind::Tram as u8);
    assert_eq!(a.preview_route(&line(800.0, -300.0, 800.0, 300.0), &[], 2, false)[0], 1.0, "{}", a.issues());
    let m = a.preview_marks();
    assert!(m.chunks(4).any(|r| (r[0] - 800.0).abs() < 1.0 && r[2] == 0.0 && r[3] == 1.0), "tram over rail: one tram {m:?}");
    assert_eq!(a.add_route(&line(800.0, -300.0, 800.0, 300.0), &[], 2, false), 0, "{}", a.issues());
    assert!(a.crossings_info().chunks(6).any(|r| (r[0] - 800.0).abs() < 1.0 && r[5] == 1.0), "placed: one tram");
    // tram across tram: two
    assert_eq!(a.preview_route(&line(700.0, 150.0, 900.0, 150.0), &[], 2, false)[0], 1.0, "{}", a.issues());
    assert!(a.preview_marks().chunks(4).any(|r| (r[0] - 800.0).abs() < 1.0 && r[3] == 2.0), "tram over tram: two");
    a.set_route_kind(TrackKind::Regular as u8);
    // rail track across tram: one
    assert_eq!(a.preview_route(&line(700.0, 200.0, 900.0, 200.0), &[], 2, false)[0], 1.0, "{}", a.issues());
    assert!(a.preview_marks().chunks(4).any(|r| (r[0] - 800.0).abs() < 1.0 && r[3] == 1.0), "rail over tram: one");
    // a route leaving the first track: a flat junction at its start, flying none
    let e = a.edges_info()[0];
    let from_edge = [2.0, e, 200.0, 0.0, 0.0];
    let ends: Vec<f64> = from_edge.iter().chain(free(200.0, -600.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 1.0);
    let m = a.preview_marks();
    assert!(m.chunks(4).any(|r| r[2] == 1.0 && (r[0] - 200.0).abs() < 1.0), "{m:?}");
}

#[test]
fn preview_render_speeds_match_committed_geometry() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 500.0).iter()).copied().collect();
    let pis = [500.0, 0.0, 0.0, 0.0];
    let preview = a.preview_route(&ends, &pis, 2, false);
    let speed = a.preview_render_speeds();
    let p0 = 5 + preview[3] as usize * 4;
    assert_eq!(speed.len(), (preview.len() - p0) / 3 - 1);
    assert!(speed.iter().any(|&s| s < V_TOP as f32 * 3.6 - 1.0));
    assert!(speed.iter().any(|&s| (s - V_TOP as f32 * 3.6).abs() < 0.01));
    assert!(a.edges_info().is_empty(), "preview cannot create track");
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0);
    let edge = a.edges_info()[0] as u32;
    assert_eq!(speed, a.edge_render_speeds(edge));
    assert_eq!(&preview[p0..], a.edge_render(edge).as_slice());
    let edge_preview = a.preview_edge(edge, &pis);
    let ep0 = 5 + edge_preview[3] as usize * 4;
    assert_eq!(a.preview_render_speeds().len(), (edge_preview.len()-ep0)/3-1);
    a.preview_edge(u32::MAX, &[]);
    assert!(a.preview_render_speeds().is_empty(), "refusal cannot reuse old preview speeds");
}

#[test]
fn render_segment_speeds_and_whole_line_fleet_charge() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 500.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[500.0, 0.0, 0.0, 0.0], 2, false), 0, "{}", a.issues());
    let edge = a.edges_info()[0] as u32;
    let pts = a.edge_render(edge);
    let speeds = a.edge_render_speeds(edge);
    assert_eq!(speeds.len(), pts.len() / 3 - 1);
    assert!(speeds.iter().any(|&s| s < V_TOP as f32 * 3.6 - 1.0), "curve speed present");
    assert!(speeds.iter().any(|&s| (s - V_TOP as f32 * 3.6).abs() < 0.01), "straight speed present");
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 8, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Line".into()), 0);
    assert_eq!(slots_to_rows(a.w.svc.lines[id as usize].trains), [8, 4, 2]);
    let before = a.save();
    let (cash_before, undo_before) = (a.cash(), a.can_undo());
    let quoted: f64 = a.blueprint_cost_items().chunks(7).map(|r| r[6]).sum();
    let repeated: f64 = a.blueprint_cost_items().chunks(7).map(|r| r[6]).sum();
    assert_eq!(quoted, repeated);
    assert_eq!(a.save(), before, "quoting leaves the count-scheduled save unchanged");
    assert_eq!((a.cash(), a.can_undo()), (cash_before, undo_before));
    assert_eq!(a.construct_all(), 0);
    assert!((a.last_charge() - quoted).abs() < 0.000001);
    let cars = a.w.svc.lines[id as usize].cars;
    assert_eq!(a.fleet(), 8 * cars);
    let cash = a.cash();
    assert_eq!(a.set_train_counts(id, 9, 4, 0), 0);
    assert!((cash - a.cash() - cars as f64 * CAR_PRICE).abs() < 0.000001);
    assert_eq!(slots_to_rows(a.w.svc.lines[id as usize].trains), [9, 4, 0]);
    let snapshot = a.save();
    let mut b = api();
    assert!(b.load(&snapshot));
    assert_eq!(slots_to_rows(b.w.svc.lines[0].trains), [9, 4, 0]);
}
