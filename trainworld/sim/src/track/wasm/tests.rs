use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}

#[test]
fn upkeep_of_a_ten_km_line_and_its_shares() {
    // T-867: 10 km of double track with a station every km: $30k a km and $50k a station a day
    let mut a = api();
    a.set_cash(1e6);
    let mut st = vec![];
    for i in 0..10 {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * 1000.0, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, format!("S{i}")), 0, "{}", a.issues());
    }
    assert_eq!(a.upkeep(), vec![0.0, 0.0, 0.0, 0.0, 0.0], "blueprint costs nothing");
    let l1 = a.new_line_id();
    assert_eq!(a.set_line(l1, &st, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0xd7263d, "L1".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_cars(l1, 6), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let up = a.upkeep();
    assert!((up[0] - 0.85).abs() < 1e-3 && (up[3] - 10.0).abs() < 1e-3 && up[4] == 11.0, "{up:?}");
    let hours = [8.0, 10.0, 6.0];
    let day: f64 = (0..3).map(|l| hours[l] * a.running_cost(l)).sum();
    let len = a.lines_info()[9] / 1000.0;
    assert!((day - 2000.0 * len / 10.0 * (4.0 + 2.0 * 6.0) * ECONOMY / 1e6).abs() < 1e-6, "{day} for {len} km");
    let s = a.line_upkeep(&hours);
    assert!(s.len() == 4 && s[0] == l1 as f64 && (s[1] - 0.85).abs() < 1e-3 && s[3].abs() < 1e-9, "{s:?}");
    // a second line as busy over the first half shares that half's track and stations
    let l2 = a.new_line_id();
    assert_eq!(a.set_line(l2, &st[..6], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0x2d7dd2, "L2".into()), 0, "{}", a.issues());
    let s = a.line_upkeep(&hours);
    let share = |l: u32| s.chunks(2).find(|c| c[0] == l as f64).map_or(0.0, |c| c[1]);
    assert!((share(l1) - 0.625).abs() < 1e-3 && (share(l2) - 0.225).abs() < 1e-3, "{s:?}");
    // without trains the first line carries nothing; its own half goes to the rest
    assert_eq!(a.set_schedule(l1, 0.0, 0.0, 0.0), 0, "{}", a.issues());
    let s = a.line_upkeep(&hours);
    assert!(!s.chunks(2).any(|c| c[0] == l1 as f64), "{s:?}");
    assert!((s[s.len() - 1] - 0.4).abs() < 1e-3 && (s.chunks(2).map(|c| c[1]).sum::<f64>() - 0.85).abs() < 1e-3, "{s:?}");
}

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
    assert_eq!(a.set_auto_headway(ex, 900, 1200, 0), 0, "{}", a.issues());
    let (trip, tph) = (a.w.svc.lines[ex as usize].round_trip, a.w.net.line_tph[ex as usize]);
    assert_eq!(a.w.svc.lines[ex as usize].trains, [auto_trains(trip[0], 900), auto_trains(trip[1], 1200), 0]);
    assert!(tph[0] > 3.9 && tph[1] > 2.9 && tph[2] == 0.0, "{tph:?}");
    // nothing to construct: refused, and the issue says so
    assert_eq!(a.construct_line(ex), 1);
    assert_eq!(a.issue_kinds(), "NothingToConstruct");
    assert_eq!(a.construct_all(), 1);
    assert_eq!(a.issue_kinds(), "NothingToConstruct");
    assert_eq!(a.w.net.line_tph[ex as usize], tph);
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
    assert_eq!(a.w.svc.lines[copy as usize].trains, [0, 0, 0]);
    assert_eq!(a.w.svc.lines[id as usize].trains, [2, 1, 0], "the source keeps its trains");
    assert_eq!(a.set_auto_headway(id, 600, 900, 0), 0);
    let copy2 = a.new_line_id();
    assert_eq!(a.copy_line(id, copy2, "Copy 2".into()), 0);
    assert_eq!(a.line_auto(copy2), [0.0; 3], "a headway line's copy keeps headways, at none");
    assert_eq!(a.w.svc.lines[copy2 as usize].trains, [0, 0, 0]);
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
    let loaded = super::super::world::TrackWorld::new(save::decode(&a.save()).unwrap());
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
    assert_eq!(m.len(), 3, "{m:?}");
    assert!((m[0] - 500.0).abs() < 1.0 && m[1].abs() < 1.0 && m[2] == 0.0);
    assert!(a.crossings_info().is_empty());
    assert_eq!(a.add_route(&line(500.0, -300.0, 500.0, 300.0), &[], 2, false), 0);
    let c = a.crossings_info();
    assert_eq!(c.len(), 5);
    assert_eq!(&c[2..], &[0.0, 0.0, 0.0], "a flat crossing at level 0, not built");
    // a route leaving the first track: a flat junction at its start, flying none
    let e = a.edges_info()[0];
    let from_edge = [2.0, e, 200.0, 0.0, 0.0];
    let ends: Vec<f64> = from_edge.iter().chain(free(200.0, -600.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 1.0);
    let m = a.preview_marks();
    assert!(m.chunks(3).any(|r| r[2] == 1.0 && (r[0] - 200.0).abs() < 1.0), "{m:?}");
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
    assert_eq!(a.w.svc.lines[id as usize].trains, [8, 4, 2]);
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
    assert_eq!(a.w.svc.lines[id as usize].trains, [9, 4, 0]);
    let snapshot = a.save();
    let mut b = api();
    assert!(b.load(&snapshot));
    assert_eq!(b.w.svc.lines[0].trains, [9, 4, 0]);
}

#[test]
fn auto_trains_is_the_fewest_that_hold_the_headway() {
    assert_eq!(auto_trains(900.0, 300), 3);
    assert_eq!(auto_trains(901.0, 300), 3, "half a percent over is the same headway");
    assert_eq!(auto_trains(905.0, 300), 4);
    assert_eq!(auto_trains(900.0, 0), 0);
    assert_eq!(auto_trains(0.0, 300), 0);
}

#[test]
fn clamp_auto_binds_only_at_the_limit() {
    use crate::track::world::clamp_auto;
    // a 4 minute limit (15 trains an hour) on a 2,000 s round trip: 8 trains fit, 9 do not
    assert_eq!(auto_trains(2000.0, 240), 9);
    assert_eq!(clamp_auto(9, 2000.0, 240, 15.0), 8, "at the limit, the count that fits");
    assert_eq!(clamp_auto(auto_trains(2000.0, 239), 2000.0, 239, 15.0), 8, "half a percent under is still the limit");
    assert_eq!(clamp_auto(auto_trains(2000.0, 238), 2000.0, 238, 15.0), 9, "past the slack the count is the player's");
    assert_eq!(clamp_auto(17, 2000.0, 120, 15.0), 17, "well under: unclamped, so the panel warns");
    assert_eq!(clamp_auto(auto_trains(2000.0, 300), 2000.0, 300, 15.0), 7, "a slower headway is untouched");
    assert_eq!(clamp_auto(auto_trains(2400.0, 240), 2400.0, 240, 15.0), 10, "an exact fit keeps its last train");
    // lines with one or two trains
    assert_eq!(clamp_auto(auto_trains(300.0, 240), 300.0, 240, 15.0), 1, "two would cross the limit");
    assert_eq!(clamp_auto(1, 200.0, 240, 15.0), 1, "at least one train runs");
    assert_eq!(clamp_auto(1, 900.0, 1800, 15.0), 1);
    assert_eq!(clamp_auto(2, 900.0, 450, 15.0), 2);
    // nothing limits the line, or no room is left: no clamp
    assert_eq!(clamp_auto(9, 2000.0, 240, f64::INFINITY), 9);
    assert_eq!(clamp_auto(9, 2000.0, 240, 0.0), 9);
    assert_eq!(clamp_auto(0, 2000.0, 0, 15.0), 0);
}

/// A 6 km double-track line between two dead-end termini, turning in 420 s: each terminus
/// pools its two platform tracks, (420 + 60) / 2 = 240 s a train, so the line's limit is 4:00.
fn four_minute_line() -> (TrackApi, u32) {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 2, 2, 2, 30.0, 420.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (a, id)
}

#[test]
fn full_track_lists_the_edges_of_full_sections() {
    // T-819: with a quick turnaround the open track is what fills first
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 60, 2, 2, 30.0, 0.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let full = a.full_track(0);
    assert!(!full.is_empty() && full.chunks(3).all(|c| c[1] >= FULL_RHO && c[2] > 20.0), "{full:?}");
    assert!(a.full_track(1).is_empty(), "two trains leave room: {:?}", a.full_track(1));
}

#[test]
fn a_headway_at_the_track_limit_holds_no_train_out() {
    let (mut a, id) = four_minute_line();
    let limit = &a.line_held(id)[7 * LEVELS..7 * LEVELS + 5];
    assert!((limit[0] - 240.0).abs() < 0.01, "the terminus limits the line to 4:00: {limit:?}");
    assert_eq!(limit[1], 3.0, "named as a terminus");
    let trip = a.w.svc.lines[id as usize].round_trip[0];
    let up = auto_trains(trip, 240);
    assert!(trip / (up as f64) < 240.0 && trip / (up - 1) as f64 >= 240.0, "the fixture rounds up across the limit: {trip}");

    // the cause: the rounded up count by hand holds part of a train out at the terminus
    assert_eq!(a.set_train_counts(id, up, 1, 1), 0, "{}", a.issues());
    assert!(a.line_held(id)[0] > 0.0, "rounding up crosses the limit");

    // a headway typed at the limit takes the trains that fit, and none waits (the switch to
    // headways first, which keeps the trains, T-752)
    assert_eq!(a.set_auto_headway(id, 300, 480, 0), 0, "{}", a.issues());
    assert_eq!(a.w.svc.lines[id as usize].trains[0], up, "the switch keeps the trains");
    assert_eq!(a.set_auto_headway(id, 240, 480, 0), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert_eq!(s.trains[0], up - 1, "clamped at the limit");
    assert!(s.round_trip[0] / s.trains[0] as f64 >= 240.0 - 1e-6, "the real headway is just over the typed one");
    assert_eq!(s.trains[1], auto_trains(s.round_trip[1], 480), "a headway over the limit is untouched");
    assert!(a.line_held(id)[..7 * LEVELS].iter().all(|&v| v == 0.0), "no train held out: {:?}", a.line_held(id));
    assert_eq!(a.line_auto(id), vec![240.0, 480.0, 0.0], "the typed headway stays");

    // well under the limit the count is the player's, and the panel warns
    assert_eq!(a.set_auto_headway(id, 120, 480, 0), 0, "{}", a.issues());
    assert!(a.w.svc.lines[id as usize].trains[0] >= up, "the whole trains that wait are dropped, the last part train is not");
    assert!(a.line_held(id)[0] > 0.0, "trains held out at 2:00 against a 4:00 limit");
}

/// T-752: on track another line's fixed rate already overfills, switching between counts and
/// headways buys and sells no train; the trains that cannot run are held out of service.
#[test]
fn switching_to_headways_on_overfull_track_keeps_every_train() {
    let (mut a, id) = four_minute_line();
    let stops = a.line_stops(id).to_vec();
    let other = a.new_line_id();
    // 20 an hour by rate against the termini's 15: over-full before the counted line runs at all
    assert_eq!(a.set_line(other, &stops, 20.0, 4.0, 2.0, 30.0, 420.0, 0, 0x2a7de1, "Rate".into()), 0, "{}", a.issues());
    assert_eq!(a.set_train_counts(id, 6, 2, 2), 0, "{}", a.issues());
    let (trains, fleet) = (a.w.svc.lines[id as usize].trains, a.fleet());
    assert_eq!(trains, [6, 2, 2]);
    assert!(a.line_held(id)[0] > 0.0, "the high level's trains wait at the ends");
    let trip = a.w.svc.lines[id as usize].round_trip;
    let h: [u32; 3] = std::array::from_fn(|lev| (trip[lev] / trains[lev] as f64).ceil() as u32);
    assert_eq!(a.set_auto_headway(id, h[0], h[1], h[2]), 0, "{}", a.issues());
    assert_eq!(a.line_auto(id).len(), 3);
    assert_eq!(a.w.svc.lines[id as usize].trains, trains, "no train sold by the switch");
    assert_eq!(a.w.net.line_trains[id as usize], Some(trains));
    assert!(a.line_held(id)[0] > 0.0, "held out of service instead");
    assert_eq!(a.clear_auto_headway(id), 0, "{}", a.issues());
    assert_eq!(a.w.svc.lines[id as usize].trains, trains, "and none by the switch back");
    assert_eq!(a.fleet(), fleet, "no train bought");
    // another edit on the headway line keeps the held trains too: the track is over-full
    assert_eq!(a.set_auto_headway(id, h[0], h[1], h[2]), 0, "{}", a.issues());
    assert_eq!(a.set_schedule(other, 21.0, 4.0, 2.0), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert_eq!(s.trains[0], auto_trains(s.round_trip[0], h[0]), "the headway's count, with no train trimmed on over-full track");
    assert!(a.line_held(id)[0] > 0.0);
}

#[test]
fn headway_lines_follow_their_round_trip() {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    let e = a.edges_info()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    assert_eq!(a.add_station(2, e, 3000.0, 30.0, 200, "Middle".into()), 0, "{}", a.issues());
    let mid = a.nodes_info().chunks(10).find(|c| (c[1] - 3000.0).abs() < 1.0).unwrap()[0] as u32;
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 2, 2, 2, 30.0, 180.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.line_auto(id).is_empty());
    let manual = a.w.svc.lines[id as usize].trains;

    // a headway that takes three trains over the round trip as it is now
    let rt0 = a.w.svc.lines[id as usize].round_trip[0];
    let h = (rt0 / 3.0).ceil() as u32;
    assert_eq!(a.set_auto_headway(id, 2 * h, 2 * h, 2 * h), 0, "{}", a.issues());
    assert_eq!(a.w.svc.lines[id as usize].trains, manual, "the switch to headways keeps the trains (T-752)");
    assert_eq!(a.set_auto_headway(id, h, 0, 2 * h), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert_eq!(s.trains[0], 3);
    assert_eq!(s.trains[1], 0, "0 seconds is no service");
    assert!(s.round_trip[0] / s.trains[0] as f64 <= h as f64 * 1.005 + 1e-6, "headway held");
    assert!(s.round_trip[0] / (s.trains[0] - 1) as f64 > h as f64 * 1.005, "and with the fewest trains");
    assert_eq!(a.line_auto(id), vec![h as f64, 0.0, 2.0 * h as f64]);
    assert_eq!(a.fleet(), a.cars_needed(), "growing the fleet bought the cars");

    // a stop in the middle lengthens the round trip, and the count follows it
    let cash = a.cash();
    let cars = a.w.svc.lines[id as usize].cars;
    assert_eq!(a.set_stops(id, &[n0, mid, n1]), 0, "{}", a.issues());
    let s = &a.w.svc.lines[id as usize];
    assert!(s.round_trip[0] > rt0);
    assert_eq!(s.trains[0], 4);
    assert!(s.round_trip[0] / s.trains[0] as f64 <= h as f64 * 1.005 + 1e-6);
    assert_eq!(a.line_auto(id).len(), 3, "an edit to the stops keeps the headways");
    assert!((cash - a.cash() - cars as f64 * CAR_PRICE).abs() < 1e-6, "one more train, bought with the edit");

    // short of money the edit is refused and nothing moves
    let (fleet, trains) = (a.fleet(), a.w.svc.lines[id as usize].trains);
    a.set_cash(1.0);
    assert_eq!(a.set_auto_headway(id, h / 2, 0, 2 * h), 1, "{}", a.issues());
    assert!(a.issue_kinds().starts_with("Money:"), "{}", a.issue_kinds());
    assert_eq!((a.fleet(), a.w.svc.lines[id as usize].trains), (fleet, trains));
    assert_eq!(a.line_auto(id)[0], h as f64, "the old headways stay");
    // a longer headway needs fewer trains; the cars it frees stay owned
    assert_eq!(a.set_auto_headway(id, 2 * h, 0, 2 * h), 0, "{}", a.issues());
    assert!(a.w.svc.lines[id as usize].trains[0] < trains[0]);
    assert_eq!(a.fleet(), fleet, "cars already owned are kept");

    // the headways are saved; a manual line is saved as before
    a.set_cash(6000.0);
    let snapshot = a.save();
    let mut b = api();
    assert!(b.load(&snapshot));
    assert_eq!(b.line_auto(0), vec![2.0 * h as f64, 0.0, 2.0 * h as f64]);
    assert_eq!(b.w.svc.lines[0].trains, a.w.svc.lines[id as usize].trains);

    // by hand again: the counts stay as they were, and undo goes back to headways
    assert_eq!(a.clear_auto_headway(id), 0);
    assert!(a.line_auto(id).is_empty());
    assert!(!a.save().windows(4).any(|w| w == b"LAU1"));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_auto(id).len(), 3);
    assert_eq!(a.set_train_counts(id, 5, 5, 5), 0);
    assert!(a.line_auto(id).is_empty(), "a hand-set count ends the headways");
    assert_eq!(a.w.svc.lines[id as usize].trains, [5, 5, 5]);
    assert_ne!(manual, [5, 5, 5]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_auto(id).len(), 3, "undo brings the headways back");
}

/// T-817: high-speed trains per line, their own fleet, prices, running costs and saves.
#[test]
fn high_speed_trains_are_a_fleet_of_their_own() {
    let mut a = api();
    a.set_cash(1e6);
    a.set_route_hsr(true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(30000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Line 1".into()), 0, "{}", a.issues());
    assert!(!a.line_fast(id));
    // a planned line's train quote follows its kind
    let ordinary_quote = a.line_train_cost(id);
    assert_eq!(a.set_line_fast(id, true), 0, "{}", a.issues());
    assert!((a.line_train_cost(id) - ordinary_quote * 1.5).abs() < 1e-9);
    assert_eq!(a.set_line_fast(id, false), 0);
    assert_eq!(a.construct_line(id), 0, "{}", a.issues());
    let need = a.cars_needed();
    assert!(need > 0);
    assert_eq!((a.fleet(), a.fleet_fast(), a.cars_needed_fast()), (need, 0, 0));
    let slow = a.stop_times(id, 0, 0)[1];
    let speeds = a.line_stretch_speeds(id);
    assert!((speeds[0] - 160.0).abs() < 1e-6 && (speeds[1] - 160.0).abs() < 1e-6 && speeds[2] == 0.0, "{speeds:?}");
    // choosing high-speed trains buys the cars at 3.75 each; the ordinary ones stay as spares
    let cash = a.cash();
    assert_eq!(a.set_line_fast(id, true), 0, "{}", a.issues());
    assert!(a.line_fast(id));
    let hs = a.cars_needed_fast();
    assert_eq!((a.cars_needed(), a.fleet(), a.fleet_fast()), (0, need, hs));
    assert!((cash - a.cash() - hs as f64 * HS_CAR_PRICE).abs() < 1e-6, "paid {}", cash - a.cash());
    assert!(a.stop_times(id, 0, 0)[1] < slow * 0.75, "{} against {slow}", a.stop_times(id, 0, 0)[1]);
    let speeds = a.line_stretch_speeds(id);
    assert!((speeds[0] - 320.0).abs() < 1e-6 && (speeds[1] - 320.0).abs() < 1e-6 && speeds[2] == 0.0, "{speeds:?}");
    // running costs 5 a train-km plus 3 a car-km
    let info = a.lines_info();
    let (len_km, tph, cars) = (info[9] / 1000.0, info[5], info[21]);
    let tk = tph * 2.0 * len_km;
    assert!((a.running_cost(0) - (5.0 * tk + 3.0 * tk * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // back to ordinary trains and forward again: both kinds are owned, nothing is paid
    let cash = a.cash();
    assert_eq!(a.set_line_fast(id, false), 0);
    assert_eq!((a.fleet(), a.fleet_fast(), a.cash()), (need, hs, cash));
    let tk = a.lines_info()[5] * 2.0 * len_km;
    assert!((a.running_cost(0) - (4.0 * tk + 2.0 * tk * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // the spare high-speed cars never serve an ordinary line that grows
    assert_eq!(a.set_train_counts(id, 6, 4, 2), 0, "{}", a.issues());
    assert!((cash - a.cash() - (a.cars_needed() - need) as f64 * CAR_PRICE).abs() < 1e-6);
    assert_eq!(a.fleet_fast(), hs);
    assert_eq!(a.set_line_fast(id, true), 0);
    // saves keep the kind; a save without the block loads as ordinary
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LHS1").expect("LHS1 written");
    let mut b = api();
    assert!(b.load(&bytes));
    assert!(b.line_fast(0));
    assert_eq!(b.stop_times(0, 0, 0), a.stop_times(id, 0, 0));
    assert!(save::decode(&bytes[..at]).map(|n| !n.line_fast[0]).unwrap());
    assert_eq!(a.set_line_fast(id, false), 0);
    assert!(!a.save().windows(4).any(|w| w == b"LHS1"));
}

#[test]
fn blueprint_construct_run() {
    let mut a = api();
    // A straight 3 km route, then a station at each end and one in the middle.
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert!((pv[1] - 3.0 * 45.0).abs() < 0.5, "cost {}", pv[1]);
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.built_cost(), 0.0);
    assert!((a.blueprint_cost() - 135.0).abs() < 0.5);
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    let e = a.edges_info()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(2, e, 1500.0, 30.0, 200, "Middle".into()), 0, "{}", a.issues());
    let mid = a.nodes_info().chunks(10).find(|c| (c[1] - 1500.0).abs() < 1.0).unwrap()[0] as u32;
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[n0, mid, n1], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0xd7263d, "Line 1".into()), 0, "{}", a.issues());
    // Planned over blueprint: times shown, no trips.
    let info = a.lines_info();
    assert_eq!((info[2], info[3]), (1.0, 0.0));
    assert!(a.stop_times(id, 0, 0)[2] > 100.0);
    assert!(a.trips(8.0 * 3600.0).is_empty());
    assert!(a.can_undo());
    // Not enough money.
    a.set_cash(10.0);
    assert_eq!(a.construct_line(id), 1);
    assert!(a.issue_kinds().starts_with("Money:"));
    assert!(a.can_undo(), "a refused construct keeps the history");
    a.set_cash(6000.0);
    let trains = a.line_train_cost(id);
    assert!(trains > 0.0);
    let quote = a.blueprint_cost_items();
    let total: f64 = quote.chunks_exact(7).map(|r| r[6]).sum();
    assert!((total - (a.blueprint_cost() + trains)).abs() < 1e-8);
    assert_eq!(quote.chunks_exact(7).filter(|r| r[0] == 1.0).map(|r| r[5]).sum::<f64>(), 3.0);
    assert_eq!(a.construct_line(id), 0, "{}", a.issues());
    assert!((total - a.last_charge()).abs() < 1e-8, "quote must include the actual fleet charge");
    assert!(a.blueprint_cost_items().is_empty());
    // Construction plus the trains its busiest schedule needs (T-028).
    assert_eq!(a.fleet(), a.cars_needed());
    assert!((a.fleet() as f64 * CAR_PRICE - trains).abs() < 1e-9);
    // three stations at grade, 75 each (T-1267)
    assert!((6000.0 - a.cash() - (135.0 + 225.0 + trains)).abs() < 0.5, "paid {}", 6000.0 - a.cash());
    assert!((a.last_charge() - (135.0 + 225.0 + trains)).abs() < 0.5);
    assert!(!a.can_undo(), "construct is final");
    // More trains an hour buys more trains; short of money it is refused and nothing changes.
    let (fleet, cash) = (a.fleet(), a.cash());
    a.set_cash(1.0);
    assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 1);
    assert!(a.issue_kinds().starts_with("Money:"), "{}", a.issue_kinds());
    assert_eq!((a.fleet(), a.lines_info()[5]), (fleet, 12.0));
    // A sandbox (T-728) refuses nothing for money; the cash still pays, into the red.
    a.set_unlimited(true);
    assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 0, "{}", a.issues());
    assert!(a.fleet() > fleet && a.last_train_charge() > 0.0);
    assert!((a.cash() - (1.0 - a.last_train_charge())).abs() < 1e-9 && a.cash() < 0.0);
    a.set_unlimited(false);
    a.set_cash(cash - a.last_train_charge());
    // Fewer trains keeps the cars; going back up again is free, and so is undo / redo.
    let (fleet, cash) = (a.fleet(), a.cash());
    assert_eq!(a.set_schedule(id, 12.0, 8.0, 4.0), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 0);
    assert_eq!((a.fleet(), a.cash()), (fleet, cash));
    // Running cost (T-807, T-867): 4 a train-km plus 2 a car-km, trains an hour both ways times length.
    let len_km = a.lines_info()[9] / 1000.0;
    let cars = a.lines_info()[21];
    let train_km = 30.0 * 2.0 * len_km;
    assert_eq!((RUN_COST_TRAIN_KM, RUN_COST_CAR_KM), (4.0, 2.0));
    assert_eq!(a.money_params(), vec![CAR_PRICE, 2.0, ECONOMY, 4.0, 3.75, 3.0, 5.0, 300.0, 500.0, 0.6, 1.5]);
    assert!((a.running_cost(0) - (4.0 * train_km + 2.0 * train_km * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // Upkeep (T-867): $30k a km of double track and $50k a station a game day, all on the one
    // line; built track no line uses goes to the rest.
    let up = a.upkeep();
    assert!((up[3] - 3.0).abs() < 0.01 && up[4] == 3.0, "{up:?}");
    assert!((up[1] - 0.03 * up[3]).abs() < 1e-9 && (up[2] - 0.15).abs() < 1e-9);
    let shares = a.line_upkeep(&[8.0, 10.0, 6.0]);
    assert_eq!(shares.len(), 4, "{shares:?}");
    assert_eq!(shares[0], id as f64);
    assert!((shares[1] - up[0]).abs() < 1e-9 && shares[3].abs() < 1e-9, "{shares:?}");
    // In debt, edits that cost nothing still go through.
    a.set_cash(-50.0);
    assert_eq!(a.set_schedule(id, 20.0, 8.0, 4.0), 0, "{}", a.issues());
    assert_eq!(a.blueprint_cost(), 0.0);
    let info = a.lines_info();
    assert_eq!(info[3], 1.0, "running");
    assert!(!a.trips(8.0 * 3600.0).is_empty());
    // Removing constructed track refunds nothing and breaks the line.
    let cash = a.cash();
    let e0 = a.edges_info()[0] as u32;
    assert_eq!(a.delete_edge(e0), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash);
    assert!(a.trips(8.0 * 3600.0).is_empty());
}

#[test]
fn blueprint_switched_off_is_skipped_saved_and_undone() {
    let mut a = api();
    a.set_cash(6000.0);
    for y in [0.0, 5000.0] {
        let ends: Vec<f64> = free(0.0, y).iter().chain(free(2000.0, y).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    }
    let nodes = a.nodes_info();
    let n0 = nodes[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "A".into()), 0, "{}", a.issues());
    let edges: Vec<u32> = a.edges_info().chunks(10).map(|c| c[0] as u32).collect();
    let all = a.blueprint_cost();
    let first = a.blueprint_cost_items().chunks_exact(7).map(|r| r[6]).sum::<f64>();
    assert!((all - first).abs() < 1e-8);
    // switch the second route off: the quote drops by its track, the station stays
    assert_eq!(a.set_blueprint_off(&edges[1..], &[], true), 0, "{}", a.issues());
    assert_eq!(a.off_edges(), edges[1..].to_vec());
    let quote: f64 = a.blueprint_cost_items().chunks_exact(7).map(|r| r[6]).sum();
    assert!(quote < first - 25.0, "{quote} vs {first}");
    // the same again changes nothing; switching the station off drops its price too
    assert_eq!(a.set_blueprint_off(&edges[1..], &[], true), 1);
    assert_eq!(a.set_blueprint_off(&[], &[n0], true), 0);
    assert_eq!(a.off_stations(), vec![n0]);
    assert!(a.blueprint_cost_items().chunks_exact(7).map(|r| r[6]).sum::<f64>() < quote - 1.0);
    // saved and loaded
    let snapshot = a.save();
    let mut b = api();
    assert!(b.load(&snapshot));
    assert_eq!(b.off_edges(), edges[1..].to_vec());
    assert_eq!(b.off_stations(), vec![n0]);
    // undo puts the station back
    assert_eq!(a.undo(), 0);
    assert!(a.off_stations().is_empty());
    // Construct builds only what is on, and clears nothing else
    let quote2: f64 = a.blueprint_cost_items().chunks_exact(7).map(|r| r[6]).sum();
    assert!((quote2 - quote).abs() < 1e-8, "the first switch-off left the station on");
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!((a.last_charge() - quote2).abs() < 1e-6, "charge {} quote {}", a.last_charge(), quote2);
    assert!(a.blueprint_cost_items().is_empty());
    assert!(a.built_cost() > 0.0 && a.blueprint_cost() > 0.0, "the switched off track is still blueprint");
    assert_eq!(a.off_edges(), edges[1..].to_vec());
    // building a switched off piece by hand clears the switch
    assert_eq!(a.construct(&edges[1..], &[]), 0, "{}", a.issues());
    assert!(a.off_edges().is_empty());
}

#[test]
fn several_pieces_change_level_at_once_but_not_into_a_flat_crossing() {
    let mut a = api();
    let route = |a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64, level: f64| {
        let ends = [0.0, 0.0, x0, y0, level, 0.0, 0.0, x1, y1, level];
        let mid = [(x0 + x1) / 2.0 + 300.0, (y0 + y1) / 2.0 + 300.0, 0.0, level];
        assert_eq!(a.add_route(&ends, &mid, 2, false), 0, "{}", a.issues());
    };
    // two east-west pieces at ground, and one north-south at +1 crossing both
    route(&mut a, 0.0, 0.0, 4000.0, 0.0, 0.0);
    route(&mut a, 0.0, 3000.0, 4000.0, 3000.0, 0.0);
    route(&mut a, 2000.0, -1500.0, 2000.0, 4500.0, 1.0);
    let ids: Vec<u32> = a.edges_info().chunks(10).map(|c| c[0] as u32).collect();
    assert_eq!(ids.len(), 3);
    let crossings = a.w.net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))).count();
    assert_eq!(crossings, 0, "grade separated");
    // both ground pieces to -1: one edit, the +1 piece still clear of them
    assert_eq!(a.set_edges_level(&ids[..2], -1), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 0);
    let lv = |a: &TrackApi, e: u32| a.edges_info().chunks(10).find(|c| c[0] as u32 == e).unwrap()[8];
    assert_eq!((lv(&a, ids[0]), lv(&a, ids[1])), (-1.0, -1.0));
    // all three to -1: the +1 piece would cross both flat, so it is left out and counted
    assert_eq!(a.set_edges_level(&ids, -1), 1, "nothing could change");
    assert_eq!(a.last_skipped(), 1);
    // all three to 0: the ground pieces go, the +1 piece would cross them flat
    assert_eq!(a.set_edges_level(&ids, 0), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 1);
    assert_eq!((lv(&a, ids[0]), lv(&a, ids[1])), (0.0, 0.0));
    // one undo for the batch
    assert_eq!(a.undo(), 0);
    assert_eq!((lv(&a, ids[0]), lv(&a, ids[1])), (-1.0, -1.0));
    assert_eq!(a.undo(), 0);
    assert_eq!((lv(&a, ids[0]), lv(&a, ids[1])), (0.0, 0.0));
}

#[test]
fn a_picked_run_changes_level_with_its_joints() {
    let mut a = api();
    // a straight run of two pieces with no corners, and a third piece off its far end
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(2000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let e0 = a.edges_info()[0] as u32;
    assert!(a.w.net.edge_data(e0).unwrap().pis.is_empty(), "drawn with no corners");
    let mid = a.w.net.edge_b[e0 as usize];
    let next = [1.0, mid as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&next, &[], 2, false), 0, "{}", a.issues());
    let far = a.w.net.edge_b[1];
    let off = [1.0, far as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&off, &[], 2, false), 0, "{}", a.issues());
    let lv = |a: &TrackApi, e: u32| { let c = a.edges_info(); let r = c.chunks(10).find(|c| c[0] as u32 == e).unwrap(); (r[8], r[9]) };
    // the first two: the loose end and the joint go down with them; the far node has the third
    // piece, so the second piece, with no corners to ramp at, cannot change; nor then the joint,
    // so the first cannot either
    assert_eq!(a.set_edges_level(&[0, 1], -2), 1);
    assert_eq!(a.last_skipped(), 2);
    // all three: every node goes, the whole run in a tunnel, one undo
    assert_eq!(a.set_edges_level(&[0, 1, 2], -2), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 0);
    for e in 0..3 {
        assert_eq!(lv(&a, e), (-2.0, -2.0));
    }
    assert_eq!(a.undo(), 0);
    for e in 0..3 {
        assert_eq!(lv(&a, e), (0.0, 0.0));
    }
}

#[test]
fn several_pieces_removed_at_once() {
    let mut a = api();
    a.set_cash(6000.0);
    for y in [0.0, 3000.0] {
        let ends: Vec<f64> = free(0.0, y).iter().chain(free(2000.0, y).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    }
    let n0 = a.nodes_info()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "A".into()), 0, "{}", a.issues());
    let ids: Vec<u32> = a.edges_info().chunks(10).map(|c| c[0] as u32).collect();
    let nodes = a.nodes_info().len() / 10;
    // blueprint: one undoable edit takes both pieces, the station and the nodes left bare
    assert_eq!(a.delete_pieces(&ids, &[n0]), 0, "{}", a.issues());
    assert!(a.edges_info().is_empty() && a.nodes_info().is_empty());
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!(a.edges_info().len() / 10, 2);
    assert_eq!(a.nodes_info().len() / 10, nodes);
    assert!(a.w.net.station_data(n0).is_some());
    // without the station, its node stays
    assert_eq!(a.delete_pieces(&ids, &[]), 0, "{}", a.issues());
    assert_eq!(a.nodes_info().len() / 10, 1);
    assert_eq!(a.undo(), 0);
    // constructed: final, nothing refunded
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let cash = a.cash();
    assert_eq!(a.delete_pieces(&ids, &[]), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash);
    assert!(a.edges_info().is_empty());
    assert!(!a.can_undo());
}

#[test]
fn blueprint_quote_shared_lines_matches_charge_without_changing_game() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let n0 = a.w.net.edge_a[0];
    let n1 = a.w.net.edge_b[0];
    for n in [n0, n1] { assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, format!("S{n}")), 0); }
    for i in 0..2 {
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[n0, n1], 20.0, 10.0, 4.0, 30.0, 180.0, 0, 0, format!("L{i}")), 0);
    }
    let saved = save::encode(&a.w.net);
    let cash = a.cash();
    let quote = a.blueprint_cost_items();
    assert_eq!(quote, a.blueprint_cost_items());
    assert_eq!(saved, save::encode(&a.w.net));
    assert_eq!(cash, a.cash());
    assert!(a.can_undo());
    let total: f64 = quote.chunks_exact(7).map(|r| r[6]).sum();
    assert_eq!(a.construct_all(), 0);
    assert!((total - a.last_charge()).abs() < 1e-8, "quote {total}, charge {}", a.last_charge());
    assert!(a.blueprint_cost_items().is_empty());
}

#[test]
fn junction_inspector_and_delay_split() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    let ends = [2.0, e, 3000.0, 5.0, 0.0, 0.0, 0.0, 5000.0, 2500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
    let (w, east, branch) = (node_at(&a, 0.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 5000.0, 2500.0));
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    for s in [w, east, branch] {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let l1 = a.new_line_id();
    assert_eq!(a.set_line(l1, &[w, east], 15.0, 8.0, 4.0, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
    let l2 = a.new_line_id();
    assert_eq!(a.set_line(l2, &[w, branch], 18.0, 8.0, 4.0, 30.0, 180.0, 0, 0x0000ff, "Branch".into()), 0, "{}", a.issues());
    let flyover = a.flyover_cost(j);
    assert!(flyover > 0.0);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!((a.flyover_cost(j) - flyover).abs() < 1e-9, "the same price built or not");
    let v = a.junction_info(j);
    let np = v[0] as usize;
    assert_eq!(np, 3);
    let mut i = 1 + np * 4;
    let nm = v[i] as usize;
    i += 1;
    assert_eq!(nm, 4, "each line both ways");
    let mut crossing = 0;
    let mut delayed = 0;
    for _ in 0..nm {
        let tph = v[i + 2];
        assert!(tph == 15.0 || tph == 18.0);
        let delay = v[i + 5];
        let nl = v[i + 11] as usize;
        let nc = v[i + 12 + nl] as usize;
        crossing += nc;
        if delay > 0.0 {
            delayed += 1;
        }
        i += 13 + nl + nc;
    }
    assert_eq!(i, v.len());
    // one crossing pair (SPEC 6.2: a double-track Y has one), seen from both moves
    assert_eq!(crossing, 2);
    assert!(delayed >= 1);
    // the round trip splits into running, dwell, turnaround and delay
    let s = a.line_summary(l1);
    let dd = a.line_dwell_split(l1);
    let delay = s[2];
    assert!(delay > 0.0 || a.line_summary(l2)[2] > 0.0);
    assert!(s[0] - dd[0] - dd[1] - delay > 0.0);
    assert!((dd[1] - 360.0).abs() < 1e-9);
    let worst = a.line_delays(l2, 0);
    assert!(worst.is_empty() || worst[4] > 0.0);
    // the main line's route shows the flat junction between its two stops, with the branch line there
    let m = a.line_marks(l1);
    assert_eq!(m.len(), 7, "{m:?}");
    assert_eq!((m[0], m[1], m[5], m[6]), (4.0, 0.0, 1.0, l2 as f64));
    assert!(m[2] > 0.0 && m[2] < 1.0);
}

#[test]
fn reshape_blueprint_track() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
    // Drawn with one PI (T-092): the track cuts the corner there, auto radius.
    let one_pi = [2000.0, 500.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &one_pi, 2, false), 0, "{}", a.issues());
    let e = a.edges_info()[0] as u32;
    let p = a.edge_pis(e);
    assert_eq!(p.len(), 9);
    assert_eq!(&p[..5], &[0.0, 2000.0, 500.0, 0.0, 0.0], "not a T-079 edge; the PI as clicked, auto radius");
    assert!(p[5] > 100.0, "auto radius used {}", p[5]);
    assert!(a.w.net.project(e, 2000.0, 500.0).1 > 1.0, "the curve cuts the corner");
    let len0 = a.edges_info()[5];
    // drag the PI further out: preview, then commit; longer, undoable
    let pv = a.preview_edge(e, &[2000.0, 1000.0, 0.0, 0.0]);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.edges_info()[5], len0, "a preview changes nothing");
    assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 0.0, 0.0]), 0, "{}", a.issues());
    assert!(a.edges_info()[5] > len0);
    // a set radius, then too small to be built
    assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 400.0, 0.0]), 0, "{}", a.issues());
    assert!((a.edge_pis(e)[5] - 400.0).abs() < 1e-6);
    assert_ne!(a.set_edge_pis(e, &[2000.0, 1000.0, 50.0, 0.0]), 0);
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    // the whole stretch one level down, shape kept
    assert_eq!(a.set_edge_level(e, -1), 0, "{}", a.issues());
    assert_eq!(a.edges_info()[8], -1.0);
    assert!((a.edge_pis(e)[5] - 400.0).abs() < 1e-6);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edges_info()[5], len0);
    // a set radius the edit leaves alone goes back to auto when its corner moves where it no
    // longer fits (radii a split froze); one the edit sets is refused instead
    assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 1500.0, 0.0]), 0, "{}", a.issues());
    assert_eq!(a.set_edge_pis(e, &[300.0, 1500.0, 1500.0, 0.0]), 0, "{}", a.issues());
    assert_eq!(a.edge_pis(e)[3], 0.0, "back to auto");
    assert_ne!(a.set_edge_pis(e, &[300.0, 1500.0, 1700.0, 0.0]), 0);
    assert!(a.issue_kinds().contains("ArcDoesNotFit"), "{}", a.issue_kinds());
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edges_info()[5], len0);
    // the flyover's price comes from a trial, nothing changes
    assert_eq!(a.flyover_cost(a.nodes_info()[0] as u32), 0.0, "an end is no junction; nothing to fly over");
    // a save round trip keeps the PIs, and a TWT2 save no longer loads (T-1146)
    let mut b = api();
    assert!(b.load(&a.save()));
    assert_eq!(b.edges_info(), a.edges_info());
    assert_eq!(b.edge_pis(0), a.edge_pis(e));
    let mut old = api();
    let mut bytes = old.save();
    bytes[..4].copy_from_slice(b"TWT2");
    assert!(!old.load(&bytes));
    // constructed track stays as it is
    assert_eq!(a.construct_all(), 0);
    assert_ne!(a.set_edge_pis(e, &[2000.0, 800.0, 0.0, 0.0]), 0);
    assert!(a.issue_kinds().contains("NodeInUse"));
}

/// T-092: a click leaves the track drawn before it alone. The clicks so far with the cursor on
/// the next point, then that point clicked and the cursor moved on: nothing changes up to
/// where the curve at the previous click starts, that curve keeps its corner and can only get
/// tighter (when the leg to the cursor was what limited it: it shares that leg with the new
/// click's curve now), and the new click becomes a curve.
#[test]
fn a_click_keeps_the_track_drawn_before() {
    let mut a = api();
    // start on an existing track end, so the lead-in is in it too
    let ends: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
    let clicks = [(1500.0, 300.0), (2600.0, -400.0), (2850.0, -250.0), (4800.0, 1200.0), (5200.0, 2600.0), (7000.0, 2500.0), (7600.0, 1400.0)];
    let fit_of = |a: &mut TrackApi, k: usize, cursor: (f64, f64)| -> (geom::Fit, Vec<Pi>) {
        let pis: Vec<f64> = clicks[..k].iter().flat_map(|&(x, y)| [x, y, 0.0, 0.0]).collect();
        let ends = [1.0, start, 0.0, 0.0, 0.0, 0.0, 0.0, cursor.0, cursor.1, 0.0];
        let pv = a.preview_route(&ends, &pis, 2, false);
        assert_eq!(pv[0], 1.0, "{} clicks: {}", k, a.issues());
        let v = a.route_vertices.clone();
        (geom::fit(&v), v)
    };
    // where the curve at vertex i starts and ends (its middle where there is none)
    let arc = |f: &geom::Fit, i: usize| -> (f64, f64) {
        match f.pieces.iter().zip(&f.piece_at_vertex).find(|&(p, &v)| v as usize == i && p.k != 0.0) {
            Some((p, _)) => (p.s0, p.s0 + p.len),
            None => (f.vert_s[i], f.vert_s[i]),
        }
    };
    let (mut tighter, mut same) = (0, 0);
    for k in 1..clicks.len() - 1 {
        let (before, vb) = fit_of(&mut a, k, clicks[k]);
        let (after, va) = fit_of(&mut a, k + 1, clicks[k + 1]);
        let prev = vb.len() - 2; // the previous click's vertex in both
        assert_eq!(&vb[..=prev], &va[..=prev], "the vertices up to the previous click, lead-in included");
        // up to where the previous click's curve starts, the very same track
        let upto = arc(&before, prev).0.min(arc(&after, prev).0);
        let mut s = 0.0;
        while s <= upto {
            let (p, q) = (geom::pos_at(&before.pieces, s), geom::pos_at(&after.pieces, s));
            assert!((p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9 && (p.2 - q.2).abs() < 1e-12, "click {k}: moved at {s} m");
            s += 1.0;
        }
        for i in 1..prev {
            assert_eq!(before.radius[i], after.radius[i], "click {k}: an earlier curve changed");
        }
        // the previous click's curve: same corner, never wider
        let (rb, ra) = (before.radius[prev], after.radius[prev]);
        assert!(ra <= rb + 1e-9, "click {k}: the previous curve got wider, {rb} to {ra}");
        let leg = (vb[prev + 1].x - vb[prev].x).hypot(vb[prev + 1].y - vb[prev].y);
        if before.tangent[prev] <= leg / 2.0 + 1e-9 {
            // not limited by the leg to the cursor: the whole curve is untouched
            assert_eq!(rb, ra, "click {k}");
            let end = arc(&before, prev).1;
            assert!((end - arc(&after, prev).1).abs() < 1e-9);
            same += 1;
        } else if ra < rb - 1e-6 {
            tighter += 1;
        }
        // and the new click is a corner with its own curve
        assert!(after.radius[prev + 1] > 0.0, "click {k}");
    }
    assert!(tighter >= 1 && same >= 3, "both cases covered: {tighter} tighter, {same} unchanged");
}

/// T-475: several blueprint nodes dragged together: track between two of them shifts whole,
/// track to a node that stays bends; one undo puts all back.
#[test]
fn drag_several_blueprint_nodes() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    assert_eq!(a.add_station(2, e as u32, 4500.0, 0.0, 200, "Middle".into()), 0, "{}", a.issues());
    let ends = [2.0, e, 1500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
    let (w, s, east, b) = (node_at(&a, 0.0, 0.0), node_at(&a, 4500.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 3500.0, 1500.0));
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    let l1 = a.new_line_id();
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "W".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "E".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line(l1, &[w, s, east], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
    let at = |a: &TrackApi, n: u32| (a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]);
    let edge_between = |a: &TrackApi, p: u32, q: u32| live_edges(a).into_iter().find(|&e| { let (ea, eb) = (a.w.net.edge_a[e as usize], a.w.net.edge_b[e as usize]); (ea == p && eb == q) || (ea == q && eb == p) }).unwrap();
    let (js, jb) = (edge_between(&a, j, s), edge_between(&a, j, b));
    let len = |a: &TrackApi, e: u32| a.w.net.edge_len[e as usize];
    let (ljs, ljb, lwj) = (len(&a, js), len(&a, jb), len(&a, edge_between(&a, w, j)));
    let pis_jb = a.edge_pis(jb);
    let before: Vec<(f64, f64)> = [w, j, s, east, b].iter().map(|&n| at(&a, n)).collect();
    let (dx, dy) = (120.0, 250.0);
    let pv = a.preview_move_nodes(&[j, s, b], dx, dy);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(pv[2], 4.0, "four edges change");
    assert_eq!(at(&a, j), before[1], "a preview changes nothing");
    assert_eq!(a.move_nodes(&[j, s, b], dx, dy), 0, "{}", a.issues());
    for (k, &n) in [j, s, b].iter().enumerate() {
        let (x0, y0) = before[[1, 2, 4][k]];
        assert!((at(&a, n).0 - x0 - dx).abs() < 0.01 && (at(&a, n).1 - y0 - dy).abs() < 0.01, "node {n} moved by the offset");
    }
    assert_eq!(at(&a, w), before[0], "unselected nodes stay");
    assert!((len(&a, js) - ljs).abs() < 0.01 && (len(&a, jb) - ljb).abs() < 0.01, "track between moved nodes keeps its shape");
    let shifted = a.edge_pis(jb);
    assert_eq!(shifted.len(), pis_jb.len());
    for c in 0..(pis_jb.len() - 1) / 8 {
        let (i, k) = (1 + c * 8, 2 + c * 8);
        assert!((shifted[i] - pis_jb[i] - dx).abs() < 0.01 && (shifted[k] - pis_jb[k] - dy).abs() < 0.01, "corners shift too");
    }
    assert!((len(&a, edge_between(&a, w, j)) - lwj).abs() > 1.0, "track to a node that stays bends");
    assert_eq!(a.lines_info()[2], 1.0, "the line still works");
    assert_eq!(a.undo(), 0);
    let after: Vec<(f64, f64)> = [w, j, s, east, b].iter().map(|&n| at(&a, n)).collect();
    assert_eq!(after, before, "one undo puts all back");
}

/// T-093: dragging blueprint nodes.
#[test]
fn drag_blueprint_nodes() {
    let mut a = api();
    // a 6 km line east, a station splitting it at 4.5 km, a branch from a junction at 1.5 km
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    assert_eq!(a.add_station(2, e as u32, 4500.0, 0.0, 200, "Middle".into()), 0, "{}", a.issues());
    let ends = [2.0, e, 1500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
    let (w, s, east, b) = (node_at(&a, 0.0, 0.0), node_at(&a, 4500.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 3500.0, 1500.0));
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
    for n in [w, east, b] {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
    }
    let l1 = a.new_line_id();
    assert_eq!(a.set_line(l1, &[w, s, east], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
    let l2 = a.new_line_id();
    assert_eq!(a.set_line(l2, &[w, b], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0x0000ff, "Branch".into()), 0, "{}", a.issues());
    let heading = |a: &TrackApi, n: u32| a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap()[9];
    let same_axis = |h1: f64, h2: f64| geom::wrap(h1 - h2).abs() < 1e-6 || (geom::wrap(h1 - h2).abs() - PI).abs() < 1e-6;
    let at = |a: &TrackApi, n: u32| { let c = a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap().to_vec(); (c[1], c[2]) };
    let paths = |a: &TrackApi| (a.line_path(l1), a.line_path(l2), a.line_stops(l1), a.line_stops(l2));
    let before = paths(&a);
    let (hs, hj) = (heading(&a, s), heading(&a, j));
    let t_end = a.stop_times(l1, 0, 0)[2];

    // the end of the line (a terminus station on a track end): its heading follows its edge;
    // the station before it keeps its heading, its track bends out of it
    let pv = a.preview_move_node(east, 6200.0, 400.0);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(pv[2], 1.0, "one edge follows");
    assert_eq!(at(&a, east), (6000.0, 0.0), "a preview changes nothing");
    assert_eq!(a.move_node(east, 6200.0, 400.0), 0, "{}", a.issues());
    assert_eq!(at(&a, east), (6200.0, 400.0));
    assert!(same_axis(heading(&a, s), hs), "the station keeps its heading");
    assert!(a.stop_times(l1, 0, 0)[2] > t_end, "a longer run to the end");
    assert_eq!(paths(&a), before, "lines keep their stops and paths");
    assert_eq!(a.lines_info()[2], 1.0, "and still work");
    assert_eq!(a.undo(), 0);
    assert_eq!(at(&a, east), (6000.0, 0.0));
    assert_eq!(a.stop_times(l1, 0, 0)[2], t_end);

    // the station in the middle: it keeps its heading, both sides bend to meet it
    assert_eq!(a.move_node(s, 4500.0, 250.0), 0, "{}", a.issues());
    assert!(same_axis(heading(&a, s), hs) && same_axis(heading(&a, j), hj), "headings kept");
    assert_eq!(paths(&a), before);
    assert!(a.lines_info()[2] == 1.0 && a.lines_info()[4] == 0.0, "the line works, not broken");
    // and again: the lead-in from the last drag moves with it, nothing piles up
    let n_pis = a.w.net.pis.data.len();
    let pis_s = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).map(|e| a.w.net.edge_data(e).unwrap().pis.len()).sum::<usize>();
    let k0 = pis_s(&a);
    assert_eq!(a.move_node(s, 4600.0, 300.0), 0, "{}", a.issues());
    assert_eq!(pis_s(&a), k0, "{} PIs before, {} after ({n_pis})", k0, pis_s(&a));
    assert!(same_axis(heading(&a, s), hs));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);

    // the junction: its branch's lead-in moves with it, the main line bends
    assert_eq!(a.move_node(j, 1700.0, -150.0), 0, "{}", a.issues());
    assert!(same_axis(heading(&a, j), hj));
    assert_eq!(paths(&a), before);
    assert_eq!(a.undo(), 0);

    // too far: the track would turn back on itself; refused, nothing changes
    let pv = a.preview_move_node(east, 4400.0, 60.0);
    assert_eq!(pv[0], 0.0);
    assert_ne!(a.move_node(east, 4400.0, 60.0), 0);
    assert!(!a.issue_kinds().is_empty());
    assert_eq!(at(&a, east), (6000.0, 0.0));

    // constructed track meeting there: it stays put
    let e_wj = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && { let d = a.w.net.edge_data(e).unwrap(); (d.a == w && d.b == j) || (d.a == j && d.b == w) }).unwrap();
    assert_eq!(a.construct(&[e_wj], &[]), 0, "{}", a.issues());
    assert_ne!(a.move_node(j, 1700.0, -150.0), 0);
    assert!(a.issue_kinds().contains("NodeInUse"));
    assert_eq!(a.move_node(b, 3600.0, 1600.0), 0, "a blueprint end elsewhere still moves: {}", a.issues());
    // a constructed station stays put
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert_ne!(a.move_node(s, 4500.0, 250.0), 0);
    assert_eq!(a.preview_move_node(s, 4500.0, 250.0)[0], 0.0);
}

/// Track drawn while the track ran through the clicked points (T-079, saves TWT3 with flag 8)
/// loads and looks as it did; its first reshape cuts the corners at its clicks.
#[test]
fn t079_track_keeps_its_shape_until_reshaped() {
    let mut a = api();
    let (n0, n1, e) = (a.new_node_id(), a.new_node_id(), a.new_edge_id());
    assert_eq!(a.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(a.set_node(n1, 4000.0, 0.0, 0, false), 0);
    // as T-079 stored it: fitted PIs with set radii, and the point clicked
    let fitted = vec![Pi::new(1200.0, 380.0, 1500.0, 0), Pi::new(2800.0, 380.0, 1500.0, 0)];
    let clicks = vec![Pi::new(2000.0, 300.0, 0.0, 0)];
    let op = Op::Edge { id: e, data: Some(EdgeData { a: n0, b: n1, tracks: 2, pis: fitted.clone(), built: false, off: false, thru: clicks.clone(), crossover: false, hsr: false }) };
    assert!(a.w.apply(op).is_ok());
    let bytes = a.save();
    assert_eq!(&bytes[..4], b"TWT4");
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.edges_info(), a.edges_info(), "loads as it was");
    assert_eq!(b.edge_render(0), a.edge_render(e), "and looks the same");
    assert_eq!(b.save(), bytes, "and saves the same");
    // its squares are the clicks, as auto PIs
    let p = b.edge_pis(0);
    assert_eq!(&p[..5], &[1.0, 2000.0, 300.0, 0.0, 0.0]);
    // reshaping it: the click becomes a corner; no clicks kept after that
    assert_eq!(b.set_edge_pis(0, &[2000.0, 400.0, 0.0, 0.0]), 0, "{}", b.issues());
    assert_eq!(b.edge_pis(0)[0], 0.0);
    assert!(b.w.net.edge_thru[0].is_empty());
    assert_ne!(b.edges_info()[5], a.edges_info()[5]);
    assert_eq!(b.undo(), 0);
    assert_eq!(b.edge_render(0), a.edge_render(e), "undo brings the T-079 shape back");
    assert_eq!(b.edge_pis(0)[0], 1.0);
}

#[test]
fn degenerate_routes_do_not_panic() {
    let mut a = api();
    let same: Vec<f64> = free(10.0, 10.0).iter().chain(free(10.0, 10.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&same, &[], 2, false)[0], 0.0);
    assert_ne!(a.add_route(&same, &[], 2, false), 0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.preview_route(&ends, &[0.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
    assert_eq!(a.preview_route(&ends, &[1000.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let n = a.nodes_info()[0];
    let e = a.edges_info()[0];
    // From a node to the same node, and onto the edge at its own end.
    let r = [1.0, n, 0.0, 0.0, 0.0, 1.0, n, 0.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    let r = [1.0, n, 0.0, 0.0, 0.0, 2.0, e, 1.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    let r = [2.0, e, 500.0, 0.0, 0.0, 2.0, e, 500.0, 0.0, 0.0];
    assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
    assert_eq!(a.edges_info().len(), 10);
}

#[test]
fn branch_leaves_along_the_heading() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let e = a.edges_info()[0];
    // Branch from the middle of the edge, off to the north-east.
    let ends = [2.0, e, 2000.0, 5.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.edges_info().len() / 10, 3);
    let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).map(|c| c[0]);
    assert!(j.is_some(), "a junction");
    // Extending a track end, turning a little.
    let end_node = a.nodes_info().chunks(10).find(|c| (c[1] - 4000.0).abs() < 1e-3).unwrap()[0];
    let ends = [1.0, end_node, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, -600.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    // Undo all three, redo one.
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edges_info().len() / 10, 1);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.edges_info().len() / 10, 3);
    // A segment that ends 100 m short of water has no land for a bridge (T-656): refused with a kind
    // the UI can word.
    let mut w = api();
    w.w.net.set_water_mask(Box::new(|x: f64, _y: f64| x > 1000.0 && x < 1500.0));
    w.w.reload();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1600.0, 0.0).iter()).copied().collect();
    assert_ne!(w.add_route(&ends, &[], 2, false), 0);
    assert!(w.issue_kinds().contains("BridgeNoRoom"), "{}", w.issue_kinds());
}

/// T-237: the stop markers' tiers come from the line tool's own search: a station past either
/// end over track the line can carry on over is reachable as it is (the path only grows), a
/// branch past an end too, a station on a bypass changes the path, and a branch the line would
/// have to turn back to reach or a station on other track cannot be added.
#[test]
fn stop_tiers_follow_the_line_tool() {
    let mut a = api();
    let route = |a: &mut TrackApi, ends: [f64; 10], pis: &[f64]| assert_eq!(a.add_route(&ends, pis, 2, false), 0, "{}", a.issues());
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9000.0, 0.0, 0.0], &[]);
    let on_main = |a: &TrackApi, x: f64| (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, 0.0).1 < 0.01 && a.w.net.node_y[a.w.net.edge_a[e as usize] as usize] == 0.0 && a.w.net.node_y[a.w.net.edge_b[e as usize] as usize] == 0.0).unwrap() as f64;
    // a bypass from 2 km to 4 km, a branch from 6 km past the end, one from 4.5 km facing back
    let e1 = on_main(&a, 2000.0);
    let ends = [2.0, e1, 2000.0, 0.0, 0.0, 2.0, e1, 4000.0, 0.0, 0.0];
    route(&mut a, ends, &[3000.0, 800.0, 0.0, 0.0]);
    let e = on_main(&a, 6000.0);
    route(&mut a, [2.0, e, 6000.0, 0.0, 0.0, 0.0, 0.0, 7500.0, 1500.0, 0.0], &[]);
    let e = on_main(&a, 4500.0);
    route(&mut a, [2.0, e, 4500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, -1500.0, 0.0], &[]);
    // other track, not joined
    route(&mut a, [0.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 7000.0, 5000.0, 0.0], &[]);
    let nearest = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
    let station = |a: &mut TrackApi, x: f64, y: f64, name: &str| -> u32 {
        let e = nearest(a, x, y);
        assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{name}: {}", a.issues());
        let n = a.w.net.node_count() as u32;
        (0..n).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).min_by(|&p, &q| {
            let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap()
    };
    let first = station(&mut a, 1000.0, 0.0, "A");
    let last = station(&mut a, 5000.0, 0.0, "B");
    let between = station(&mut a, 3000.0, 0.0, "M");
    let past_end = station(&mut a, 8000.0, 0.0, "C");
    let before_start = station(&mut a, 300.0, 0.0, "D");
    let bypass = station(&mut a, 3000.0, 650.0, "S");
    let branch = station(&mut a, 7200.0, 1300.0, "Y");
    let facing_back = station(&mut a, 3700.0, -1300.0, "Z");
    let other = station(&mut a, 6000.0, 5000.0, "E");
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[first, last], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let raw = a.stop_tiers(id);
    let tiers: std::collections::HashMap<u32, f64> = raw.chunks(3).map(|c| (c[0] as u32, c[1])).collect();
    assert_eq!(tiers.len(), 9);
    // T-797: the branch facing back is reached by reversing at B
    let want = [(first, 0.0), (last, 0.0), (between, 1.0), (past_end, 1.0), (before_start, 1.0), (bypass, 2.0), (branch, 1.0), (facing_back, 4.0), (other, 3.0)];
    for (n, t) in want {
        assert_eq!(tiers[&n], t, "station {}", a.w.net.node_name[n as usize]);
    }
    for c in raw.chunks(3) {
        assert_eq!(c[2], if c[0] as u32 == facing_back { last as f64 } else { -1.0 });
    }
    // and a click does what the marker says
    let path = a.line_path(id);
    for (n, t) in want {
        let stops: Vec<u32> = a.insert_stop(id, n).iter().map(|&s| s as u32).collect();
        assert_eq!(stops.is_empty(), t == 0.0 || t == 3.0, "station {}", a.w.net.node_name[n as usize]);
        if stops.is_empty() {
            continue;
        }
        assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
        let grown = a.line_path(id);
        if t == 4.0 {
            assert_eq!(a.line_reversals(id), vec![1.0], "trains reverse at B, stop 1 of A, B, Z");
        } else {
            assert_eq!(grown.windows(path.len()).any(|w| w == path), t == 1.0, "station {}", a.w.net.node_name[n as usize]);
        }
        assert_eq!(a.undo(), 0);
    }
    // with no route yet, every station joined to the one stop counts as on it
    let lone = a.new_line_id();
    assert_eq!(a.set_line(lone, &[first], 0.0, 0.0, 0.0, 30.0, 180.0, 0, 0, "One".into()), 0, "{}", a.issues());
    let tiers: std::collections::HashMap<u32, f64> = a.stop_tiers(lone).chunks(3).map(|c| (c[0] as u32, c[1])).collect();
    assert_eq!((tiers[&past_end], tiers[&bypass], tiers[&other]), (1.0, 1.0, 3.0));
}

/// T-797 B3: from M, Br is reached only by running out past T to the bare end X and back.
#[test]
fn a_station_past_a_headshunt_reverses_at_the_track_end() {
    use super::super::tests::{line, reversal_fixture_with};
    let mut a = api();
    let (w, [x, t, _, m, br]) = reversal_fixture_with(false);
    a.w = w;
    let id = line(&mut a.w, &[m], [6.0, 3.0, 0.0]);
    let raw = a.stop_tiers(id);
    let tier = |n: u32| raw.chunks(3).find(|c| c[0] as u32 == n).map(|c| (c[1], c[2])).unwrap();
    assert_eq!(tier(br), (4.0, x as f64));
    assert_eq!(tier(t), (1.0, -1.0));
    let stops: Vec<u32> = a.insert_stop(id, br).iter().map(|&s| s as u32).collect();
    assert_eq!(stops, vec![m, br]);
    assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    assert_eq!(a.line_end_turns(id), vec![x as f64]);
    assert!(a.line_reversals(id).is_empty());
    let split = a.line_dwell_split(id);
    assert!((split[1] - 4.0 * 180.0).abs() < 1e-9, "two ends and the track end both ways: {split:?}");
}

/// T-798: on a ring the first stop's marker closes the line into a circle, then opens it; a
/// closed line takes new stops inside the loop only.
#[test]
fn the_first_stop_closes_and_opens_a_loop() {
    let mut a = api();
    let (w, ns) = super::super::tests::ring_fixture();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &ns[..4], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let tier = |a: &TrackApi, n: u32| a.stop_tiers(id).chunks(3).find(|c| c[0] as u32 == n).unwrap()[1];
    assert_eq!(tier(&a, ns[0]), 5.0, "closes");
    assert_eq!(tier(&a, ns[3]), 0.0);
    let mut stops = ns[..4].to_vec();
    stops.push(ns[0]);
    assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    assert_eq!(a.line_loop(id), 1, "a circle");
    assert_eq!(tier(&a, ns[0]), 6.0, "opens");
    // S4 and S5 lie on the closing stretch: added between S3 and the closing S0
    let grown: Vec<u32> = a.insert_stop(id, ns[4]).iter().map(|&s| s as u32).collect();
    assert_eq!(grown, vec![ns[0], ns[1], ns[2], ns[3], ns[4], ns[0]]);
    assert_eq!(tier(&a, ns[5]), 1.0, "on the loop's path");
    assert_eq!(a.set_stops(id, &ns[..4]), 0);
    assert_eq!(a.line_loop(id), 0);
    // two stops are enough when the track leads on round (on straight track, going back the
    // way it came makes no loop: the T-797 test above keeps tier 0 there)
    let mut b = api();
    let (w, ns) = super::super::tests::ring_fixture();
    b.w = w;
    let id = b.new_line_id();
    assert_eq!(b.set_line(id, &ns[..2], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0);
    assert_eq!(b.stop_tiers(id).chunks(3).find(|c| c[0] as u32 == ns[0]).unwrap()[1], 5.0, "the ring leads round");
}

/// T-244: the corners of a stroke, and the same stroke as one edge (set directly).
const CLICKS: [(f64, f64); 4] = [(1500.0, 300.0), (2600.0, -400.0), (4800.0, 1200.0), (6200.0, 900.0)];
fn stroke(level_of: &dyn Fn(usize) -> i8) -> (Vec<f64>, Vec<f64>) {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(7600.0, 2400.0).iter()).copied().collect();
    let pis = CLICKS.iter().enumerate().flat_map(|(k, &(x, y))| [x, y, 0.0, level_of(k) as f64]).collect();
    (ends, pis)
}
fn one_edge(pis: &[f64]) -> (TrackApi, u32) {
    let mut b = api();
    let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
    assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(b.set_node(n1, 7600.0, 2400.0, 0, false), 0);
    assert_eq!(b.set_edge(e, n0, n1, 2, pis), 0, "{}", b.issues());
    (b, e)
}
fn live_edges(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// Every drawn point of `a`'s track lies on `b`'s edge `e`, and their lengths agree.
fn same_track(a: &TrackApi, b: &TrackApi, e: u32, tol: f64) {
    let mut len = 0.0;
    for p in live_edges(a) {
        len += a.w.net.edge_len[p as usize];
        for q in a.edge_render(p).chunks(3) {
            let off = b.w.net.project(e, q[0], q[1]).1;
            assert!(off < tol, "piece {p}: {off} m off the stroke at {q:?}");
        }
    }
    assert!((len - b.w.net.edge_len[e as usize]).abs() < tol, "{len} against {}", b.w.net.edge_len[e as usize]);
}

#[test]
fn a_stroke_is_one_piece_per_corner() {
    let mut a = api();
    let (ends, pis) = stroke(&|_| 0);
    let pv = a.preview_route(&ends, &pis, 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert!(live_edges(&a).is_empty(), "a preview builds nothing");
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let (b, e) = one_edge(&pis);
    // four corners, four pieces, joined at three plain nodes, each corner in a piece of its own
    let edges = live_edges(&a);
    assert_eq!(edges.len(), 4);
    let joints: Vec<_> = a.nodes_info().chunks(10).filter(|c| c[4] == 2.0).map(|c| c[0] as u32).collect();
    assert_eq!(joints.len(), 3);
    for &p in &edges {
        let corners = a.edge_pis(p);
        assert_eq!(corners.len(), 9, "one corner");
        assert_eq!(corners[3], 0.0, "auto radius");
    }
    let whole = b.edge_pis(e);
    let used: Vec<f64> = edges.iter().map(|&p| a.edge_pis(p)[5]).collect();
    for k in 0..4 {
        assert!((used[k] - whole[5 + 8 * k]).abs() < 0.01, "corner {k}: {} against {}", used[k], whole[5 + 8 * k]);
    }
    // the very same track, price and preview
    same_track(&a, &b, e, 0.005);
    assert!((a.total_cost() - b.total_cost()).abs() < 1e-6, "{} against {}", a.total_cost(), b.total_cost());
    assert!((pv[1] - b.total_cost()).abs() < 1e-6);
    // one undo takes the whole stroke away, one redo brings it back
    assert_eq!(a.undo(), 0);
    assert!(live_edges(&a).is_empty() && a.nodes_info().is_empty());
    assert_eq!(a.redo(), 0);
    assert_eq!(live_edges(&a), edges);
    // saves keep the pieces
    let mut c = api();
    assert!(c.load(&a.save()));
    assert_eq!(c.edges_info(), a.edges_info());
    // removing a piece in the middle leaves its neighbours, ending where it was
    assert_eq!(a.delete_edge(edges[1]), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 3);
    for &j in &joints[..2] {
        assert_eq!(a.nodes_info().chunks(10).find(|c| c[0] as u32 == j).unwrap()[4], 1.0, "a track end now");
    }
    // a leg that changes level is not cut: its ramp is centred between its corners
    let mut d = api();
    let (ends, pis) = stroke(&|k| if k >= 2 { 1 } else { 0 });
    assert_eq!(d.add_route(&ends, &pis, 2, false), 0, "{}", d.issues());
    assert_eq!(live_edges(&d).len(), 3);
    let (_, pis_flat) = stroke(&|k| if k >= 2 { 1 } else { 0 });
    let mut e2 = api();
    let (n0, n1, x) = (e2.new_node_id(), e2.new_node_id(), e2.new_edge_id());
    assert_eq!(e2.set_node(n0, 0.0, 0.0, 0, false), 0);
    assert_eq!(e2.set_node(n1, 7600.0, 2400.0, 0, false), 0);
    assert_eq!(e2.set_edge(x, n0, n1, 2, &pis_flat), 0, "{}", e2.issues());
    same_track(&d, &e2, x, 0.005);
    assert!((d.total_cost() - e2.total_cost()).abs() < 1e-6);
}

/// T-244: a line over a stroke in pieces runs exactly as over the stroke in one edge: the
/// joints add no speed limit, no delay and no cost.
#[test]
fn pieces_run_like_the_whole_stroke() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let (mut b, _) = one_edge(&pis);
    let mut times = vec![];
    for t in [&mut a, &mut b] {
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, e) = (end(t, 0.0), end(t, 7600.0));
        assert_eq!(t.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
        assert_eq!(t.add_station(1, e, 0.0, 0.0, 200, "East".into()), 0);
        let id = t.new_line_id();
        assert_eq!(t.set_line_with_trains(id, &[w, e], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", t.issues());
        let quote = t.blueprint_cost();
        assert_eq!(t.construct_all(), 0, "{}", t.issues());
        times.push((t.stop_times(id, 0, 0), t.stop_times(id, 1, 0), t.line_summary(id), t.lines_info()[9], quote, t.markers(0).len()));
    }
    let (p, q) = (&times[0], &times[1]);
    for k in 0..2 {
        let (x, y) = if k == 0 { (&p.0, &q.0) } else { (&p.1, &q.1) };
        assert_eq!(x.len(), y.len());
        for (s, t) in x.iter().zip(y) {
            assert!((s - t).abs() < 1e-3, "run {k}: {s} s against {t} s");
        }
    }
    for (s, t) in p.2.iter().zip(&q.2) {
        assert!((s - t).abs() < 1e-3, "summary {:?} against {:?}", p.2, q.2);
    }
    assert!((p.3 - q.3).abs() < 0.01, "length {} against {}", p.3, q.3);
    assert!((p.4 - q.4).abs() < 1e-6, "price {} against {}", p.4, q.4);
    assert_eq!(p.5, q.5, "capacity markers");
    // one track section for the line, as over one edge
    let sections = |t: &TrackApi| t.w.svc.cap.res.iter().filter(|r| r.kind == super::super::capacity::ResKind::Section).count();
    assert_eq!(sections(&a), sections(&b));
}

/// T-258: a station can go anywhere along a curve (partway round, at its ends), on track drawn
/// from an existing track end (a lead-in curve, then the corner's curve) and on plain track,
/// and the split leaves the track, its run time and its price as they were.
#[test]
fn stations_go_anywhere_on_a_curve() {
    for (from_track, corner, radius) in [(true, (1500.0, 300.0), 0.0), (false, (1500.0, 300.0), 0.0), (true, (900.0, 700.0), 0.0), (false, (700.0, -900.0), 0.0), (true, (1500.0, 300.0), 150.0), (false, (1500.0, 1200.0), 400.0), (true, (1200.0, -500.0), 120.0)] {
        let mut a = api();
        let mut start = [0.0, 0.0, 0.0, 0.0, 0.0];
        if from_track {
            let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
            assert_eq!(a.add_route(&lead, &[], 2, false), 0);
            let n = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
            start = [1.0, n, 0.0, 0.0, 0.0];
        }
        let ends: Vec<f64> = start.iter().chain(free(3000.0, -600.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[corner.0, corner.1, radius, 0.0], 2, false), 0, "{}", a.issues());
        let e = *live_edges(&a).last().unwrap();
        let len = a.w.net.edge_len[e as usize];
        let pieces: Vec<_> = a.w.net.edge_pieces(e).to_vec();
        let (cost, before) = (a.total_cost(), a.edge_render(e));
        let mut tried = 0;
        let mut failed = vec![];
        for p in pieces.iter().filter(|p| p.k != 0.0) {
            for f in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
                let s = (p.s0 + f * p.len).clamp(110.0, len - 110.0);
                let (x, y, _) = geom::pos_at(&pieces, s);
                tried += 1;
                if a.add_station(2, e, x, y, 200, "S".into()) != 0 {
                    failed.push((s, a.issues(), format!("{:?}", a.w.net.split_plan(e, a.w.net.project(e, x, y).0)), a.w.net.edge_data(e)));
                    continue;
                }
                // the same track and price, and an undo puts it back exactly
                for q in before.chunks(3) {
                    let off = live_edges(&a).iter().map(|&k| a.w.net.project(k, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
                    assert!(off < 0.01, "split at {s}: {off} m off");
                }
                let station = 75.0 * (0.4 + 0.6 * 200.0 / 200.0) * 1.0;
                assert!((a.total_cost() - cost - station).abs() < 0.05, "split at {s}: {} against {}", a.total_cost() - station, cost);
                assert_eq!(a.undo(), 0);
                assert_eq!(a.edge_render(e), before);
            }
        }
        assert!(tried > 8);
        assert!(failed.is_empty(), "{from_track} {corner:?} {radius}: {failed:?}");
    }
}

/// T-244: a station placed next to a joint joins the two pieces again first, so the platform
/// has room; the track and the lines over it are as they were.
#[test]
fn a_station_next_to_a_joint_joins_the_pieces() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[w, east], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let t0 = a.stop_times(id, 0, 0);
    let edges = live_edges(&a);
    // 30 m before the second joint, on the second piece
    let j = a.w.net.edge_b[edges[1] as usize];
    let len = a.w.net.edge_len[edges[1] as usize];
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edges[1]), len - 30.0);
    assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
    assert!(!a.w.net.node_ok(j), "the joint went");
    assert_eq!(live_edges(&a).len(), 4, "two pieces joined, then split at the station");
    let (b, e) = one_edge(&pis);
    same_track(&a, &b, e, 0.005);
    assert!(!a.w.net.line_broken[id as usize]);
    let t1 = a.stop_times(id, 0, 0);
    assert!((t0.last().unwrap() - t1.last().unwrap()).abs() < 1e-3, "{t0:?} {t1:?}");
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a), edges);
    assert!(a.w.net.node_ok(j));
    // after reversing the line's path: still fine
    let id2 = a.new_line_id();
    assert_eq!(a.set_line(id2, &[east, w], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "R".into()), 0);
    assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[id2 as usize] && !a.w.net.line_broken[id as usize]);
}

/// T-418: a straight from (0, 0) east to (1000, 0), tangent drawing on, and its east end.
fn tangent_start() -> (TrackApi, u32) {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let east = a.route_end();
    assert!((a.w.net.node_x[east as usize] - 1000.0).abs() < 1e-6);
    a.set_route_tangent(true);
    (a, east)
}
/// T-442: the same straight in quad track, and its drawn pair's east end.
fn quad_start() -> (TrackApi, u32) {
    let mut a = api();
    a.set_route_tangent(true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0);
    let end = a.route_end();
    (a, end)
}
fn from_node(n: u32, x: f64, y: f64) -> Vec<f64> {
    vec![1.0, n as f64, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, 0.0]
}

/// T-471: a click past the cap: the capped curve from the start, its circle's, then one
/// straight to the click.
fn capped_then_straight(f: &geom::Fit, arc: &[f64], start: (f64, f64), what: &str) {
    let curves: Vec<_> = f.pieces.iter().filter(|p| p.k != 0.0).collect();
    assert!(curves.iter().all(|p| (p.radius() - R_CAP).abs() < 1.0), "{what}: every curve capped");
    assert!(f.pieces[0].k != 0.0 || f.pieces[0].len < 0.05, "{what}: the curve starts at the start, not after {}", f.pieces[0].len);
    let straights: Vec<_> = f.pieces.iter().filter(|p| p.k == 0.0 && p.len > 1.0).collect();
    assert_eq!(straights.len(), 1, "{what}: one straight");
    assert_eq!(f.pieces.last().unwrap().k, 0.0, "{what}: the straight last");
    assert!((arc[2] - R_CAP).abs() < 1e-6 && ((arc[0] - start.0).hypot(arc[1] - start.1) - R_CAP).abs() < 0.5, "{what}: circle {arc:?}");
}

/// T-418: from a track end, a click d away at a off the heading is one arc of radius
/// d / (2 sin a) ending at the click on heading 2a; one wider than the cap is the capped curve
/// then a straight to the click (T-471). The circle is the curve's.
#[test]
fn a_tangent_segment_is_one_arc_through_the_click() {
    let (mut a, east) = tangent_start();
    for (d, deg) in [(500.0, 30.0), (800.0, -50.0), (300.0, 12.0), (2000.0, 70.0), (500.0, 2.0)] {
        let off: f64 = (deg as f64).to_radians();
        let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
        let pv = a.preview_route(&from_node(east, x, y), &[], 2, false);
        assert_eq!(pv[0], 1.0, "{d} m at {deg}: {}", a.issues());
        let f = a.w.net.fit(&a.route_vertices);
        let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
        assert!((ex - x).hypot(ey - y) < 0.01, "{d} m at {deg}: ends at {ex}, {ey}");
        let want = d / (2.0 * off.abs().sin());
        let arc = a.route_arc();
        let (cx, cy) = (arc[0], arc[1]);
        if want <= R_CAP {
            assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{d} m at {deg}: leaves on {eh}");
            assert!((f.radius[1] - want).abs() < 0.01 * want, "{d} m at {deg}: radius {} against {want}", f.radius[1]);
            assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{d} m at {deg}: one arc, no straight");
            assert!(((cx - 1000.0).hypot(cy) - want).abs() < 0.5 && ((cx - x).hypot(cy - y) - want).abs() < 0.5, "{d} m at {deg}: circle {arc:?}");
        } else {
            capped_then_straight(&f, &arc, (1000.0, 0.0), &format!("{d} m at {deg}"));
        }
    }
    // placed, the end node comes back, and the next segment leaves along this one
    let off: f64 = 30f64.to_radians();
    let (x, y) = (1000.0 + 500.0 * off.cos(), 500.0 * off.sin());
    assert_eq!(a.add_route(&from_node(east, x, y), &[], 2, false), 0, "{}", a.issues());
    let joint = a.route_end();
    assert!((a.w.net.node_x[joint as usize] - x).abs() < 0.01 && (a.w.net.node_y[joint as usize] - y).abs() < 0.01);
    let (x2, y2) = (x + 400.0 * (2.0 * off).cos(), y + 400.0 * (2.0 * off).sin());
    assert_eq!(a.add_route(&from_node(joint, x2, y2), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[joint as usize].n, 2, "a plain joint");
    assert!(a.route_arc().is_empty(), "straight on: no curve, no circle");
    // the first segment's corner dragged sideways: the joint after it follows
    let e = live_edges(&a).into_iter().find(|&e| a.w.net.edge_b[e as usize] == joint).unwrap();
    let pis = a.edge_pis(e);
    assert_eq!(pis.len(), 9, "one corner");
    assert_eq!(a.set_edge_pis(e, &[pis[1] - 5.0, pis[2] + 10.0, 0.0, pis[4]]), 0, "{}", a.issues());
    assert!((a.w.net.node_x[joint as usize] - x).hypot(a.w.net.node_y[joint as usize] - y) > 1.0, "the joint moved");
    // with tangent drawing off, the same click gives the old curve then straight
    a.set_route_tangent(false);
    let pv = a.preview_route(&from_node(east, x, -y), &[], 2, false);
    assert_eq!(pv[0], 1.0);
    let f = a.w.net.fit(&a.route_vertices);
    assert!(f.radius[1] < 450.0, "old lead-in: {}", f.radius[1]);
}

/// T-817: the radius each design speed lays curves out for, and 160 is the old cap.
#[test]
fn design_radius_per_speed() {
    assert_eq!(design_radius(160.0), R_CAP);
    for (kmh, want) in [(160.0, 1796.0), (200.0, 2806.0), (250.0, 4385.0), (320.0, 7183.0)] {
        let r = design_radius(kmh);
        assert_eq!(r, want, "{kmh} km/h");
        assert!((A_LAT * r).sqrt() >= kmh / 3.6 && (A_LAT * (r - 1.0)).sqrt() < kmh / 3.6, "{kmh} km/h: {r} is the smallest");
    }
    assert_eq!(DESIGN_SPEEDS[0], 160.0);
}

/// T-817: at 160 the drawing tool sets no radius and lays every shape as before; at a higher
/// design speed a click past 1,796 m is one wider arc through it (radius set, so it keeps its
/// shape), and past the design radius the capped curve then a straight.
#[test]
fn design_speed_lays_wider_curves() {
    let clicks = [(500.0, 30.0), (5000.0, 40.0), (9000.0, -100.0), (20000.0, 30.0), (3000.0, 8.0)];
    for tracks in [2u8, 4] {
        // 160, named or by default, draws exactly what the default does
        let (mut plain, east) = if tracks == 4 { quad_start() } else { tangent_start() };
        let (mut at160, _) = if tracks == 4 { quad_start() } else { tangent_start() };
        at160.set_design_speed(160.0);
        for (d, deg) in clicks {
            let off: f64 = (deg as f64).to_radians();
            let ends = from_node(east, 1000.0 + d * off.cos(), d * off.sin());
            assert_eq!(plain.preview_route(&ends, &[], tracks, false), at160.preview_route(&ends, &[], tracks, false), "{tracks} tracks, {d} m at {deg}");
            assert_eq!(plain.route_vertices, at160.route_vertices);
            assert!(plain.route_vertices.iter().all(|p| p.radius == 0.0), "{tracks} tracks, {d} m at {deg}: no radius set at 160");
        }
        for kmh in [200.0, 250.0, 320.0] {
            let cap = design_radius(kmh);
            for (d, deg) in clicks {
                let (mut a, east) = if tracks == 4 { quad_start() } else { tangent_start() };
                a.set_design_speed(kmh);
                let what = format!("{tracks} tracks at {kmh}, {d} m at {deg}");
                let off: f64 = (deg as f64).to_radians();
                let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
                let ends = from_node(east, x, y);
                assert_eq!(a.preview_route(&ends, &[], tracks, false)[0], 1.0, "{what}: {}", a.issue_kinds());
                let f = a.w.net.fit(&a.route_vertices);
                let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
                assert!((ex - x).hypot(ey - y) < 0.01, "{what}: ends at {ex}, {ey}");
                let want = d / (2.0 * off.abs().sin());
                let arc = a.route_arc();
                let curves: Vec<_> = f.pieces.iter().filter(|p| p.k != 0.0).collect();
                if want <= cap {
                    assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{what}: leaves on {eh}");
                    assert!(curves.iter().all(|p| (p.radius() - want).abs() < 0.01), "{what}: one radius, {want}");
                    assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{what}: one arc, no straight");
                    assert!((arc[2] - want).abs() < 0.01, "{what}: circle {arc:?}");
                } else {
                    assert!(curves.iter().all(|p| (p.radius() - cap).abs() < 0.01), "{what}: every curve at the design radius");
                    assert_eq!(f.pieces.last().unwrap().k, 0.0, "{what}: then a straight");
                    assert!((arc[2] - cap).abs() < 1e-6, "{what}: circle {arc:?}");
                }
                // placed, every piece keeps the shape (the radius set where it is over 1,796 m)
                let before = live_edges(&a);
                assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{what}: {}", a.issues());
                if tracks == 2 && want.min(cap) > R_CAP {
                    for e in live_edges(&a).into_iter().filter(|e| !before.contains(e)) {
                        for p in a.w.net.edge_pieces(e).iter().filter(|p| p.k != 0.0) {
                            assert!((p.radius() - want.min(cap)).abs() < 0.01, "{what}: built at {}", p.radius());
                        }
                    }
                }
            }
        }
    }
}

/// T-817: a segment ending on track at a higher design speed: its lead-in curves may be wider
/// than 1,796 m, still starting at the node.
#[test]
fn design_speed_widens_the_lead_in() {
    let (mut a, east) = tangent_start();
    a.set_design_speed(320.0);
    // from far off to the east end, arriving well off its heading
    let ends: Vec<f64> = free(20000.0, 6000.0).iter().copied().chain([1.0, east as f64, 0.0, 0.0, 0.0]).collect();
    assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 1.0, "{}", a.issue_kinds());
    let f = a.w.net.fit(&a.route_vertices);
    let r = f.radius.iter().copied().fold(0.0, f64::max);
    assert!(r > R_CAP && r <= 7183.0 + 1e-6, "lead-in radius {r}");
    let last = f.pieces.last().unwrap();
    assert!(last.k != 0.0 || last.len < 0.05, "the curve reaches the node");
    a.set_design_speed(160.0);
    a.preview_route(&ends, &[], 2, false);
    assert!(a.w.net.fit(&a.route_vertices).radius.iter().all(|&r| r <= R_CAP + 1e-6));
}

/// T-817 b1: high-speed track is drawn as double track only, costs 1.5x double track as
/// previewed and quoted (its own quote rows), and survives a save.
#[test]
fn high_speed_track_costs_half_again_and_is_saved() {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    let mut plain = api();
    assert_eq!(plain.add_route(&ends, &[], 2, false), 0, "{}", plain.issues());
    let mut a = api();
    a.set_route_hsr(true);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issue_kinds());
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let hsr = a.hsr_edges();
    assert!(!hsr.is_empty() && hsr == live_edges(&a));
    assert!((a.blueprint_cost() - HSR_COST * plain.blueprint_cost()).abs() < 1e-6, "{} against {}", a.blueprint_cost(), plain.blueprint_cost());
    assert!((pv[1] - a.blueprint_cost()).abs() < 1e-6, "previewed {} built {}", pv[1], a.blueprint_cost());
    // the preview's multiplier is level and water only, as for single track
    assert!((pv[4] - plain.preview_route(&ends, &[], 2, false)[4]).abs() < 1e-9);
    let rows = a.blueprint_cost_items();
    assert!(rows.chunks_exact(7).filter(|r| r[0] == 0.0).all(|r| r[2] == HSR_ROW as f64));
    assert!((rows.chunks_exact(7).map(|r| r[6]).sum::<f64>() - a.blueprint_cost()).abs() < 1e-6);
    // single and quad track drawn with the setting on are ordinary track
    for (y, tracks) in [(2000.0, 1u8), (4000.0, 4)] {
        let e: Vec<f64> = free(0.0, y).iter().chain(free(3000.0, y).iter()).copied().collect();
        assert_eq!(a.add_route(&e, &[], tracks, false), 0, "{}", a.issues());
    }
    assert_eq!(a.hsr_edges(), hsr);
    // saved with flag 64 and loaded as it was; a save without it loads with none
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.hsr_edges().len(), hsr.len());
    assert_eq!(b.save(), bytes);
    assert!((b.blueprint_cost() - a.blueprint_cost()).abs() < 1e-6);
    let mut c = api();
    assert!(c.load(&plain.save()));
    assert!(c.hsr_edges().is_empty());
    // high-speed track has no quad form
    assert_ne!(a.double_up(hsr[0]), 0);
    assert!(a.issue_kinds().contains("HighSpeed"), "{}", a.issue_kinds());
}

/// T-817 b1: a split keeps high-speed track high speed; two high-speed pieces join under a
/// platform as double track does, but never with ordinary track, which still meets it at a
/// plain node.
#[test]
fn high_speed_track_survives_split_and_join() {
    for second in [true, false] {
        let mut a = api();
        a.set_route_hsr(true);
        let mut ends = free_at(0.0, 0.0, 0);
        ends.extend(free_at(1000.0, 0.0, 0));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n = a.route_end();
        a.set_route_hsr(second);
        let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, 0));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "ordinary track joins at a plain node: {}", a.issues());
        assert_eq!(a.hsr_edges().len(), if second { 2 } else { 1 });
        let e = edge_under(&a, 960.0, 0.0);
        let r = a.add_station(2, e, 960.0, 0.0, 200, "P".into());
        if second {
            assert_eq!(r, 0, "{}", a.issues());
            assert_eq!(live_edges(&a).len(), 2, "joined, then split at the station");
            assert_eq!(a.hsr_edges(), live_edges(&a), "both halves high speed");
        } else {
            assert_ne!(r, 0, "high-speed and ordinary track are not joined");
            assert!(a.issue_kinds().contains("PlatformJoint"), "{}", a.issue_kinds());
            // a station clear of the joint splits the high-speed piece into two high-speed halves
            let e = edge_under(&a, 400.0, 0.0);
            assert_eq!(a.add_station(2, e, 400.0, 0.0, 200, "Q".into()), 0, "{}", a.issues());
            assert_eq!(a.hsr_edges().len(), 2);
            assert_eq!(live_edges(&a).len(), 3);
        }
    }
}

/// T-442: a click beyond a quarter turn is still one arc of radius d / (2 sin a) through it,
/// up to 350 degrees, split into corners of at most 90 degrees; on quad track the express pair
/// is concentric all the way round; a level change finds room on a hairpin.
#[test]
fn a_tangent_segment_turns_past_a_half_turn() {
    for tracks in [2u8, 4] {
        for (d, deg) in [(400.0, 84.0), (400.0, -100.0), (300.0, 150.0), (200.0, -170.0), (60.0, 174.0), (4000.0, 120.0), (3600.0, 170.0), (9000.0, -100.0), (5000.0, 40.0)] {
            let (mut a, east) = if tracks == 4 { quad_start() } else { tangent_start() };
            let off: f64 = (deg as f64).to_radians();
            let (x, y) = (1000.0 + d * off.cos(), d * off.sin());
            let ends = from_node(east, x, y);
            let pv = a.preview_route(&ends, &[], tracks, false);
            assert_eq!(pv[0], 1.0, "{tracks} tracks, {d} m at {deg}: {}", a.issue_kinds());
            let f = a.w.net.fit(&a.route_vertices);
            let (ex, ey, eh) = geom::pos_at(&f.pieces, f.len);
            assert!((ex - x).hypot(ey - y) < 0.01, "{d} m at {deg}: ends at {ex}, {ey}");
            let want = d / (2.0 * off.abs().sin());
            if want <= R_CAP {
                assert!(geom::wrap(eh - 2.0 * off).abs() < 1e-4, "{d} m at {deg}: leaves on {eh}");
                for i in 1..a.route_vertices.len() - 1 {
                    assert!((f.radius[i] - want).abs() < 0.01 * want, "{d} m at {deg}: corner {i} radius {} against {want}", f.radius[i]);
                }
                assert!(f.pieces.iter().all(|p| p.k != 0.0 || p.len < 0.05), "{d} m at {deg}: one arc, no straight");
                assert!((a.route_arc()[2] - want).abs() < 0.5);
            } else {
                capped_then_straight(&f, &a.route_arc(), (1000.0, 0.0), &format!("{tracks} tracks, {d} m at {deg}"));
            }
            if tracks == 4 {
                let ex = a.w.net.fit(&a.route_express);
                let (qx, qy, qh) = geom::pos_at(&ex.pieces, ex.len);
                assert!(((qx - x).hypot(qy - y) - QUAD_SPACING).abs() < 0.05, "{d} m at {deg}: express ends 8 m beside");
                assert!(geom::wrap(qh - eh).abs() < 1e-3, "{d} m at {deg}: express parallel at the end");
            }
            assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{tracks} tracks, {d} m at {deg}: {}", a.issues());
        }
    }
    // a hairpin of 300 degrees going up a level ramps on its middle leg
    let (mut a, east) = tangent_start();
    let off: f64 = 150f64.to_radians();
    let (x, y) = (1000.0 + 400.0 * off.cos(), 400.0 * off.sin());
    let mut ends = from_node(east, x, y);
    ends[9] = 1.0;
    ends[5] = 0.0;
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_level[a.route_end() as usize], 1);
}

/// T-418: a click too close and too far off to the side is refused, the ghost and circle
/// still drawn; one behind the end is refused with its circle.
#[test]
fn a_too_tight_or_backward_tangent_click_is_refused() {
    let (mut a, east) = tangent_start();
    let off: f64 = 60f64.to_radians();
    let ends = from_node(east, 1000.0 + 150.0 * off.cos(), 150.0 * off.sin());
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 0.0);
    assert!(pv.len() > 5, "the red ghost is drawn");
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    let r = a.route_arc()[2];
    assert!((r - 150.0 / (2.0 * off.sin())).abs() < 0.5, "{r}");
    let fr = a.w.net.fit(&a.route_vertices).radius[1];
    assert!((fr - r).abs() < 0.5, "the ghost's curve is the circle's: {fr} against {r}");
    assert_ne!(a.add_route(&ends, &[], 2, false), 0);
    let back = from_node(east, 600.0, 5.0);
    assert_eq!(a.preview_route(&back, &[], 2, false)[0], 0.0);
    assert!(a.issue_kinds().contains("Reversal"), "{}", a.issue_kinds());
    assert_eq!(a.route_arc().len(), 3);
    assert_ne!(a.add_route(&back, &[], 2, false), 0);
    assert_eq!(live_edges(&a).len(), 1, "nothing added");
}

/// T-418: a 400 m platform across three 120 m segments joins all their joints first; dragging
/// a segment's corner afterwards moves the joint with it.
#[test]
fn a_station_spans_several_short_segments() {
    let (mut a, east) = tangent_start();
    let mut end = east;
    let mut heading = 0.0f64;
    let mut segs = vec![];
    for turn in [3.0f64, -3.0, 2.0, -2.0, 3.0] {
        let off = turn.to_radians();
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + 120.0 * (heading + off).cos(), y0 + 120.0 * (heading + off).sin());
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 2, false), 0, "{}", a.issues());
        heading += 2.0 * off;
        end = a.route_end();
        segs.push(*live_edges(&a).last().unwrap());
    }
    let mid = segs[2];
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(mid), a.w.net.edge_len[mid as usize] / 2.0);
    assert_eq!(a.add_station(2, mid, x, y, 400, "Long".into()), 0, "{}", a.issues());
    assert!(a.w.net.node_platform.iter().any(|&p| p == 400));
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a).len(), 6);
}

/// T-859: removing stations a line stops at takes them out of its stops and routes it again;
/// where the track between its stops is gone too it is left broken. One undo restores it all.
#[test]
fn removing_a_stop_reroutes_or_breaks_the_line() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let node_at = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && (a.w.net.node_x[n as usize] - x).abs() < 1.0 && a.w.net.node_y[n as usize].abs() < 1.0).unwrap();
    for x in [1000.0, 2000.0] {
        let e = edge_under(&a, x, 0.0);
        assert_eq!(a.add_station(2, e, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    for x in [0.0, 3000.0] {
        let n = node_at(&a, x);
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
    }
    let st: Vec<u32> = [0.0, 1000.0, 2000.0, 3000.0].iter().map(|&x| node_at(&a, x)).collect();
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 0, 0, 0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[l as usize]);
    // a middle stop, its track kept: the line runs on past it
    assert_eq!(a.removal_lines(&[], &[st[1]]), vec![l as f64, 1.0, 0.0]);
    assert_eq!(a.remove_station(st[1]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(l), &[st[0], st[2], st[3]]);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.w.net.line_stops(l), &st[..]);
    // the end stop and its track: the line ends a stop sooner
    let last = edge_under(&a, 2500.0, 0.0);
    assert_eq!(a.delete_pieces(&[last], &[st[3]]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(l), &st[..3]);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    // a middle stop and the track on to the next: broken, until undone
    let mid = edge_under(&a, 1500.0, 0.0);
    assert_eq!(a.removal_lines(&[mid], &[st[1]]), vec![l as f64, 1.0, 1.0]);
    assert_eq!(a.delete_pieces(&[mid], &[st[1]]), 0, "{}", a.issues());
    assert!(a.w.net.line_broken[l as usize]);
    assert_eq!(a.undo(), 0);
    assert!(!a.w.net.line_broken[l as usize]);
    assert_eq!(a.w.net.line_stops(l), &st[..]);
}

/// T-860: a station on a loose end whose last segment is shorter than the platform joins the
/// segments behind it first, as a split does; one undo puts the joint back.
#[test]
fn a_station_on_a_short_last_segment_end() {
    let (mut a, east) = tangent_start();
    assert_eq!(a.add_route(&from_node(east, 1080.0, 0.0), &[], 2, false), 0, "{}", a.issues());
    let end = a.route_end();
    assert_eq!(live_edges(&a).len(), 2);
    assert_eq!(a.preview_station(1, end, 0.0, 0.0, 200), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 200, "End".into()), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 1, "joined under the platform");
    assert_eq!(a.w.net.node_platform[end as usize], 200);
    assert_eq!(a.undo(), 0);
    assert_eq!(live_edges(&a).len(), 2);
}

// ---- T-622: a platform beside a crossing or a joint

fn free_at(x: f64, y: f64, level: i8) -> Vec<f64> {
    vec![0.0, 0.0, x, y, level as f64]
}
/// A straight piece from a free point or an earlier piece's end, `pis` (x, y, level) between;
/// returns its far node.
fn piece(a: &mut TrackApi, from: Result<u32, (f64, f64, i8)>, pis: &[(f64, f64, i8)], to: (f64, f64, i8)) -> u32 {
    let mut ends = match from {
        Ok(n) => vec![1.0, n as f64, 0.0, 0.0, 0.0],
        Err((x, y, l)) => free_at(x, y, l),
    };
    ends.extend(free_at(to.0, to.1, to.2));
    let pis: Vec<f64> = pis.iter().flat_map(|p| [p.0, p.1, 0.0, p.2 as f64]).collect();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    a.route_end()
}
/// The edge a click at (x, y) lands on: the one it is nearest, with the click inside it.
fn edge_under(a: &TrackApi, x: f64, y: f64) -> u32 {
    let net = &a.w.net;
    live_edges(a)
        .into_iter()
        .map(|e| {
            let (s, off) = net.project(e, x, y);
            (e, if off > 1.0 { 1e6 } else { off + if s < 0.01 || s > net.edge_len[e as usize] - 0.01 { 1e3 } else { 0.0 } })
        })
        .min_by(|p, q| p.1.total_cmp(&q.1))
        .unwrap()
        .0
}
/// What `preview_station` says at (x, y) on the track along y = 0: its issue kinds, and whether
/// `add_station` agrees (undone again).
fn platform_at(a: &mut TrackApi, x: f64, plat: u16) -> String {
    let e = edge_under(a, x, 0.0);
    platform_on(a, e, x, 0.0, plat)
}
/// The same on edge `e` at (x, y): "" when it works, else the issue kinds. The real edit must
/// agree with the preview, and is undone.
fn platform_on(a: &mut TrackApi, e: u32, x: f64, y: f64, plat: u16) -> String {
    let dry = a.preview_station(2, e, x, y, plat);
    let kinds = a.issue_kinds();
    let nodes = live_edges(a).len();
    let real = a.add_station(2, e, x, y, plat, "P".into());
    assert_eq!(dry == 0, real == 0, "at {x}: preview {dry} ({kinds}) real {real} ({})", a.issues());
    if real == 0 {
        assert_eq!(a.undo(), 0);
        assert_eq!(live_edges(a).len(), nodes, "undone");
        return String::new();
    }
    assert_eq!(kinds, a.issue_kinds(), "at {x}: preview and edit give the same reasons");
    kinds
}

/// A line along y = 0 from x = 0 to 3000 on `level`, in `cuts` pieces (joints at those x), and
/// a cross track at x = 1500 on `cross`; a 200 m platform every `step` m across `from..to`.
/// The platform (reach 100, 1 m slack) alerts for the crossing exactly when it is within
/// 101 m of it, and never for being too long.
fn crossing_scan(level: i8, cross: i8, cuts: &[f64], from: f64, to: f64, step: f64) -> Vec<(f64, String)> {
    let mut a = api();
    let mut at = Err((0.0, 0.0, level));
    for &c in cuts {
        at = Ok(piece(&mut a, at, &[], (c, 0.0, level)));
    }
    piece(&mut a, at, &[], (3000.0, 0.0, level));
    piece(&mut a, Err((1500.0, -1000.0, cross)), &[], (1500.0, 1000.0, cross));
    let mut out = vec![];
    let mut x = from;
    while x <= to {
        out.push((x, platform_at(&mut a, x, 200)));
        x += step;
    }
    out
}
fn expect_crossing(res: &[(f64, String)], flat: bool) {
    for (x, k) in res {
        let near = (x - 1500.0).abs() <= 101.0;
        if flat && near && !((x - 1500.0).abs() > 99.0) {
            assert_eq!(k, "PlatformCrossing", "at {x}");
        } else if !flat || (x - 1500.0).abs() >= 103.0 {
            assert_eq!(k, "", "at {x}");
        }
    }
}

#[test]
fn a_platform_alerts_for_a_flat_crossing_within_its_span_and_not_beyond() {
    expect_crossing(&crossing_scan(-1, -1, &[], 1300.0, 1700.0, 2.0), true);
    expect_crossing(&crossing_scan(0, 0, &[], 1300.0, 1700.0, 2.0), true);
}

#[test]
fn a_platform_over_a_grade_separated_crossing_never_alerts() {
    expect_crossing(&crossing_scan(-1, 0, &[], 1300.0, 1700.0, 2.0), false);
    expect_crossing(&crossing_scan(-1, -2, &[], 1300.0, 1700.0, 2.0), false);
}

/// A route of curved segments drawn through their ends (tangent arcs) in Tokyo's frame, a track
/// crossing the fifth segment at 70 degrees on the same level: a 200 m platform every 3 m along
/// the whole route alerts for the crossing exactly when it is within 101 m of it.
#[test]
fn a_platform_along_curved_joined_segments_alerts_only_near_the_crossing() {
    let mut a = TrackApi::new(139.7, 35.7, true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(400.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let mut end = a.route_end();
    a.set_route_tangent(true);
    let mut chain = vec![live_edges(&a)[0]];
    let mut heading = 0.0f64;
    for (k, turn) in [4.0f64, -3.0, 6.0, -5.0, 3.0, -4.0, 5.0, -2.0, 4.0, -3.0].into_iter().enumerate() {
        let off = turn.to_radians();
        let len = 140.0 + 37.0 * (k % 4) as f64;
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + len * (heading + off).cos(), y0 + len * (heading + off).sin());
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 2, false), 0, "{}", a.issues());
        heading += 2.0 * off;
        end = a.route_end();
        chain.push(*live_edges(&a).last().unwrap());
    }
    let lens: Vec<f64> = chain.iter().map(|&e| a.w.net.edge_len[e as usize]).collect();
    let total: f64 = lens.iter().sum();
    // the crossing, a little before the middle of the fifth segment's length
    let start5: f64 = lens[..5].iter().sum();
    let at_cross = start5 + lens[5] * 0.4;
    let (cx, cy, th) = geom::pos_at(a.w.net.edge_pieces(chain[5]), lens[5] * 0.4);
    let (ca, sa) = ((th + 70f64.to_radians()).cos(), (th + 70f64.to_radians()).sin());
    piece(&mut a, Err((cx - 600.0 * ca, cy - 600.0 * sa, 0)), &[], (cx + 600.0 * ca, cy + 600.0 * sa, 0));
    assert_eq!(a.w.net.crossings.len(), 1, "one flat crossing");
    let mut c = 110.0;
    let step = 1.0;
    while c < total - 110.0 {
        let k = (0..chain.len()).rev().find(|&k| lens[..k].iter().sum::<f64>() <= c).unwrap();
        let s = c - lens[..k].iter().sum::<f64>();
        let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(chain[k]), s);
        let d = (c - at_cross).abs();
        if !(99.0..=103.0).contains(&d) {
            let got = platform_on(&mut a, chain[k], x, y, 200);
            assert_eq!(got, if d < 99.0 { "PlatformCrossing" } else { "" }, "at {c} of {total} (edge {k}, s {s}), crossing at {at_cross}");
        }
        c += step;
    }
}

/// Two pieces along y = 0 joined at x = 1000, then a 200 m platform every 10 m from 850 to
/// 1150. `setup` turns the pieces into the case (built, ramped, ...) and returns what to
/// prepare; the answer at each x.
fn joint_scan(setup: &dyn Fn(&mut TrackApi, u32, u32), levels: [i8; 3], tracks: [u8; 2]) -> Vec<(f64, String)> {
    let mut a = api();
    a.set_cash(1e9);
    let mut ends = free_at(0.0, 0.0, levels[0]);
    ends.extend(free_at(1000.0, 0.0, levels[1]));
    assert_eq!(a.add_route(&ends, &[], tracks[0], false), 0, "{}", a.issues());
    let n = a.route_end();
    let e1 = live_edges(&a)[0];
    let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
    ends.extend(free_at(2000.0, 0.0, levels[2]));
    assert_eq!(a.add_route(&ends, &[], tracks[1], false), 0, "{}", a.issues());
    let e2 = *live_edges(&a).last().unwrap();
    setup(&mut a, e1, e2);
    let mut out = vec![];
    let mut x = 850.0;
    while x <= 1150.0 {
        out.push((x, platform_at(&mut a, x, 200)));
        x += 10.0;
    }
    out
}
fn never(res: &[(f64, String)]) {
    let bad: Vec<_> = res.iter().filter(|r| !r.1.is_empty()).collect();
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn a_joint_with_a_ramp_beside_it_is_merged_under_a_platform() {
    // level 0 to level -1 over one piece: the ramp is mid piece, 300 m from the joint at its
    // nearest, and the platform stays clear of it
    never(&joint_scan(&|_, _, _| {}, [0, 0, -1], [2, 2]));
    never(&joint_scan(&|_, _, _| {}, [-1, 0, 0], [2, 2]));
}

/// Joining pieces for a platform leaves the heights as they were: the ramp stays where it was
/// drawn, whichever piece it is on.
#[test]
fn joining_pieces_for_a_platform_keeps_the_ramps_where_they_were() {
    for levels in [[0, 0, -1], [-1, 0, 0], [-1, -1, 0], [0, -1, -1]] {
        let mut a = api();
        let mut ends = free_at(0.0, 0.0, levels[0]);
        ends.extend(free_at(1000.0, 0.0, levels[1]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n = a.route_end();
        let mut ends = vec![1.0, n as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, levels[2]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let before = heights(&a);
        let e = edge_under(&a, 960.0, 0.0);
        assert_eq!(a.add_station(2, e, 960.0, 0.0, 200, "P".into()), 0, "{levels:?}: {}", a.issues());
        assert!(live_edges(&a).len() == 2, "the joint went, the station split once: {}", live_edges(&a).len());
        same_heights(&before, &heights(&a), &format!("{levels:?}"));
    }
}

/// The heights along y = 0 every 49 m from x = 25.
fn heights(a: &TrackApi) -> Vec<f64> {
    (0..=40)
        .map(|k| {
            let x = 25.0 + 49.0 * k as f64;
            let e = edge_under(a, x, 0.0);
            geom::height_at(a.w.net.edge_vert(e), a.w.net.project(e, x, 0.0).0)
        })
        .collect()
}
fn same_heights(before: &[f64], after: &[f64], what: &str) {
    for (k, (p, q)) in before.iter().zip(after).enumerate() {
        assert!((p - q).abs() < 1e-6, "{what} at x {}: {p} before, {q} after", 25 + 49 * k);
    }
}

/// A station splits a stretch with a ramp in it: the ramp stays where it was drawn, whether
/// the station is before or after it (it used to follow the station, so nudging the station
/// moved the ramp under other track).
#[test]
fn a_station_on_a_stretch_with_a_ramp_leaves_the_ramp_where_it_was() {
    let mut checked = 0;
    for (l0, l1) in [(-1, 0), (0, -1), (-2, 0), (1, -1)] {
        for x in [150.0, 330.0, 700.0, 900.0, 1100.0, 1300.0, 1700.0, 1850.0] {
            let mut a = api();
            let mut ends = free_at(0.0, 0.0, l0);
            ends.extend(free_at(2000.0, 0.0, l1));
            assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
            let before = heights(&a);
            let e = edge_under(&a, x, 0.0);
            let got = a.add_station(2, e, x, 0.0, 60, "P".into());
            if got != 0 {
                // on the ramp (refused), or too near an end
                assert!(a.issue_kinds().contains("SplitPoint") || a.issue_kinds().contains("PlatformNotLevel"), "{l0}/{l1} at {x}: {}", a.issue_kinds());
                continue;
            }
            same_heights(&before, &heights(&a), &format!("{l0}/{l1} station at {x}"));
            checked += 1;
        }
    }
    assert!(checked >= 16, "only {checked} placements were checked");
}

#[test]
fn a_joint_that_cannot_be_merged_says_so() {
    // built beside blueprint, single beside double track (blueprint or built): the platform
    // reaches the joint, which stays, and the answer names it (not "a junction or another station")
    let cases: [(&str, Box<dyn Fn(&mut TrackApi, u32, u32)>, [u8; 2]); 4] = [
        ("built | blueprint", Box::new(|a, e1, _| assert_eq!(a.construct(&[e1], &[]), 0, "{}", a.issues())), [2, 2]),
        ("blueprint | built", Box::new(|a, _, e2| assert_eq!(a.construct(&[e2], &[]), 0, "{}", a.issues())), [2, 2]),
        ("single | double", Box::new(|_, _, _| {}), [1, 2]),
        ("built single | built double", Box::new(|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues())), [1, 2]),
    ];
    for (what, setup, tracks) in cases {
        for (x, k) in joint_scan(&*setup, [0, 0, 0], tracks) {
            let near = (x - 1000.0).abs() < 105.0;
            match (near, k.as_str()) {
                (true, "PlatformJoint") | (true, "SplitPoint") | (false, "") => {}
                _ => panic!("{what} at {x}: {k:?}"),
            }
        }
    }
}

/// T-624: two constructed pieces of one kind join under a platform as blueprint ones do.
#[test]
fn constructed_pieces_of_one_kind_are_merged_under_a_platform() {
    for tracks in [[1, 1], [2, 2]] {
        never(&joint_scan(&|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues()), [0, 0, 0], tracks));
    }
    // a ramp in either piece stays where it was
    for levels in [[0, 0, -1], [-1, 0, 0], [-1, -1, 0], [0, -1, -1]] {
        never(&joint_scan(&|a, e1, e2| assert_eq!(a.construct(&[e1, e2], &[]), 0, "{}", a.issues()), levels, [2, 2]));
    }
}

/// T-624: built double track in two pieces with a running line over both, a station placed
/// across the joint. The pieces become one edge (then split at the station); the line's
/// path, its trains, the money and the track's cost are as they were, and undo restores
/// the two pieces.
#[test]
fn a_station_across_a_built_joint_keeps_the_line_its_trains_and_the_money() {
    for levels in [[0, 0, 0], [0, 0, -1], [-1, 0, 0]] {
        let mut a = api();
        a.set_cash(1e9);
        let mut ends = free_at(0.0, 0.0, levels[0]);
        ends.extend(free_at(1000.0, 0.0, levels[1]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let joint = a.route_end();
        let mut ends = vec![1.0, joint as f64, 0.0, 0.0, 0.0];
        ends.extend(free_at(2000.0, 0.0, levels[2]));
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, east) = (end(&a, 0.0), end(&a, 2000.0));
        for (n, x) in [(w, 0.0), (east, 2000.0)] {
            assert_eq!(a.add_station(1, n, x, 0.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let pieces = live_edges(&a);
        assert_eq!(pieces.len(), 2);
        assert_eq!(a.construct(&pieces, &[w, east]), 0, "{}", a.issues());
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[w, east], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        assert!(a.w.svc.lines[id as usize].running, "{levels:?}: the line runs");
        a.set_now(40_000.0);
        let snap = |t: &TrackApi| {
            let trips = t.trips(40_000.0 - 1800.0);
            let at: Vec<Vec<f64>> = (0..8).map(|k| t.train_at(id, 0, 0, 40_000.0 - 600.0, 40_000.0 + 90.0 * k as f64)).collect();
            (t.cash, t.built_cost(), t.total_cost(), t.stop_times(id, 0, 0), t.stop_times(id, 1, 0), trips, at, t.line_summary(id))
        };
        let before = snap(&a);
        let before_heights = heights(&a);
        assert!(!before.5.is_empty(), "trains are running");
        let len: f64 = pieces.iter().map(|&e| a.w.net.edge_len[e as usize]).sum();

        let e = edge_under(&a, 960.0, 0.0);
        assert_eq!(a.preview_station(2, e, 960.0, 0.0, 200), 0, "{levels:?}: {}", a.issues());
        assert_eq!(a.add_station(2, e, 960.0, 0.0, 200, "Middle".into()), 0, "{levels:?}: {}", a.issues());
        assert!(!a.w.net.node_ok(joint), "{levels:?}: the joint went");
        let now = live_edges(&a);
        assert_eq!(now.len(), 2, "joined, then split at the station");
        assert!(now.iter().all(|&e| a.w.net.edge_built[e as usize]), "still constructed");
        assert!((now.iter().map(|&e| a.w.net.edge_len[e as usize]).sum::<f64>() - len).abs() < 1e-6);
        let l = a.w.net.line_path(id).to_vec();
        assert!(l.iter().all(|&p| a.w.net.edge_ok(crate::track::net::path_edge(p))), "the path uses live edges");
        assert!(!a.w.net.line_broken[id as usize] && a.w.svc.lines[id as usize].running);
        assert_eq!(a.w.net.line_stops(id), &[w, east], "stops unchanged");
        let after = snap(&a);
        assert_eq!(before.0, after.0, "{levels:?}: cash");
        assert!((before.1 - after.1).abs() < 1e-9, "{levels:?}: built cost {} {}", before.1, after.1);
        assert_eq!((&before.3, &before.4, &before.5, &before.7), (&after.3, &after.4, &after.5, &after.7), "{levels:?}: timetable");
        for (p, q) in before.6.iter().zip(&after.6) {
            assert!((p[0] - q[0]).hypot(p[1] - q[1]) < 1e-3, "{levels:?}: a train moved: {p:?} {q:?}");
        }
        assert!(a.w.svc.eras_of(id as usize).is_empty() && a.w.svc.shifts[id as usize].is_empty(), "no train was moved or re-timed");
        same_heights(&before_heights, &heights(&a), &format!("{levels:?}"));

        assert_eq!(a.undo(), 0);
        assert_eq!(live_edges(&a), pieces, "{levels:?}: the two pieces are back");
        assert!(a.w.net.node_ok(joint) && pieces.iter().all(|&e| a.w.net.edge_built[e as usize]));
        let undone = snap(&a);
        assert_eq!(before.0, undone.0);
        assert!((before.1 - undone.1).abs() < 1e-9);
        assert_eq!((&before.3, &before.5), (&undone.3, &undone.5));
        same_heights(&before_heights, &heights(&a), &format!("{levels:?} undone"));
    }
}

#[test]
fn a_plain_joint_under_a_platform_is_merged_and_the_crossing_still_answers() {
    // joints at 1000, 1450 and 2000: the crossing at 1500 sits just past the middle one
    let res = crossing_scan(-1, -1, &[1000.0, 1450.0, 2000.0], 800.0, 2200.0, 5.0);
    for (x, k) in &res {
        assert!(!k.contains("TooLong"), "at {x}: {k}");
    }
    expect_crossing(&res, true);
    let res = crossing_scan(-1, 0, &[1000.0, 1450.0, 2000.0], 800.0, 2200.0, 5.0);
    assert!(res.iter().all(|(_, k)| k.is_empty()), "{res:?}");
    // the joint right under the crossing: it is two crossings until the pieces are joined
    let res = crossing_scan(-1, -1, &[1500.0], 1200.0, 1800.0, 5.0);
    assert!(res.iter().all(|(x, k)| !k.contains("TooLong") && !k.contains("Joint") && (k.is_empty() == ((x - 1500.0).abs() > 101.0))), "{res:?}");
}

/// T-311: `preview_station` answers as `add_station` would, in the middle of a stretch and
/// near its end, and changes nothing (no edit, no undo step, no cash, no ids).
#[test]
fn preview_station_agrees_with_add_station() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let edge = live_edges(&a)[0];
    let len = a.w.net.edge_len[edge as usize];
    let (cash, nodes, edges_n) = (a.cash, a.nodes_info(), live_edges(&a));
    let mut seen = [0, 0];
    for s in [len * 0.5, 20.0, len - 20.0, len * 0.3] {
        let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edge), s);
        let dry = a.preview_station(2, edge, x, y, 200);
        assert_eq!((a.cash, a.nodes_info(), live_edges(&a)), (cash, nodes.clone(), edges_n.clone()), "dry run at {s}");
        let why = a.issues();
        let real = a.add_station(2, edge, x, y, 200, "S".into());
        assert_eq!(dry == 0, real == 0, "at {s}: preview {dry} ({why}) real {real} ({})", a.issues());
        seen[(real == 0) as usize] += 1;
        if real == 0 {
            assert_eq!(a.undo(), 0);
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "both answers should occur: {seen:?}");
}

/// T-794: a station on open ground lays a straight piece of track a little longer than the
/// platform and puts the station at its middle, one undoable edit; the trial changes nothing
/// and prices both; the piece's ends are loose ends that later track carries on from; on quad
/// the express pair gets the twin.
#[test]
fn a_free_station_lays_its_own_track() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(240.0, 0.0).iter()).copied().collect();
    let (cash, nodes) = (a.cash, a.nodes_info());
    assert_eq!(a.preview_free_station(&ends, 2, 200), 0, "{}", a.issues());
    assert_eq!((a.cash, a.nodes_info(), live_edges(&a).len()), (cash, nodes, 0), "the trial leaves nothing");
    let quoted = a.station_cost();
    assert!(quoted > 0.0);
    assert_eq!(a.add_free_station(&ends, 2, 200, "Open".into()), 0, "{}", a.issues());
    assert!((a.total_cost() - quoted).abs() < 1e-6, "{} against {quoted}", a.total_cost());
    let info = a.nodes_info();
    let rows: Vec<&[f64]> = info.chunks(10).collect();
    assert_eq!(rows.len(), 3);
    let st: Vec<&&[f64]> = rows.iter().filter(|c| c[7] > 0.0).collect();
    assert_eq!(st.len(), 1);
    assert!((st[0][1] - 120.0).abs() < 1.0, "the station at the middle: {:?}", st[0]);
    let loose: Vec<f64> = rows.iter().filter(|c| c[4] == 1.0).map(|c| c[0]).collect();
    assert_eq!(loose.len(), 2, "both ends loose");
    // carrying on from an end: an ordinary route, the station keeps its platform
    let end = rows.iter().find(|c| c[4] == 1.0 && c[1] > 200.0).unwrap()[0];
    let on = [1.0, end, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&on, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.undo(), 0);
    // one undo takes the station and its track
    assert_eq!(a.undo(), 0);
    assert!(live_edges(&a).is_empty() && a.nodes_info().is_empty());
    // too short for the platform: refused, nothing left
    let short: Vec<f64> = free(0.0, 0.0).iter().chain(free(150.0, 0.0).iter()).copied().collect();
    assert_ne!(a.add_free_station(&short, 2, 200, "Short".into()), 0);
    assert!(a.issue_kinds().contains("PlatformTooLong"), "{}", a.issues());
    assert!(live_edges(&a).is_empty());
    // quad: the twin on the express pair
    assert_eq!(a.add_free_station(&ends, 4, 200, "Quad".into()), 0, "{}", a.issues());
    let named = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == "Quad").count();
    assert_eq!(named, 2, "the station and its twin");
    // across other track on the same level: refused, though a flat crossing is allowed elsewhere
    let mut b = api();
    let across: Vec<f64> = free(230.0, -300.0).iter().chain(free(230.0, 300.0).iter()).copied().collect();
    assert_eq!(b.add_route(&across, &[], 2, false), 0, "{}", b.issues());
    assert_ne!(b.preview_free_station(&ends, 2, 200), 0);
    assert!(b.issue_kinds().contains("PlatformCrossing"), "{}", b.issue_kinds());
    assert_eq!(live_edges(&b).len(), 1);
}

/// T-255: quad track is the drawn double track and an express double track beside it, end to
/// end; a station on one pair gets its twin on the other.
#[test]
fn quad_track_and_twin_stations() {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(12000.0, 0.0).iter()).copied().collect();
    let pis = vec![3000.0, 400.0, 0.0, 0.0, 6000.0, 0.0, 0.0, 0.0, 9000.0, 400.0, 0.0, 0.0];
    let mut a = api();
    // continuing an existing track end, so the start becomes a junction
    let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&lead, &[], 2, false), 0);
    let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
    let mut quad_ends = ends.clone();
    quad_ends[..5].copy_from_slice(&[1.0, start, 0.0, 0.0, 0.0]);
    let pv = a.preview_route(&quad_ends, &pis, 4, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    let before = a.total_cost();
    assert_eq!(a.add_route(&quad_ends, &pis, 4, false), 0, "{}", a.issues());
    let mut b = api();
    let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
    assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false) + b.set_node(n1, 12000.0, 0.0, 0, false) + b.set_edge(e, n0, n1, 2, &pis), 0);
    let on_local = |a: &TrackApi, p: u32| { let r = a.edge_render(p); let k = r.len() / 6 * 3; b.w.net.project(e, r[k], r[k + 1]).1 < 2.0 };
    let new: Vec<u32> = live_edges(&a).into_iter().filter(|&p| p != 0).collect();
    let (local, express) = (new[..3].to_vec(), new[3..].to_vec());
    assert!(local[1..].iter().all(|&p| on_local(&a, p)) && express.iter().all(|&p| !on_local(&a, p)));
    assert!(express.len() >= 3, "{express:?}");
    // beside it, 8 m to the left, away from the tapers at the ends
    let len = b.w.net.edge_len[e as usize];
    for &p in &express {
        for q in a.edge_render(p).chunks(3) {
            let (s, _) = b.w.net.project(e, q[0], q[1]);
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            if s > QUAD_TAPER + 300.0 && s < len - QUAD_TAPER - 50.0 {
                assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off at {s}");
            }
        }
    }
    // two double tracks' price, and a junction where it leaves the old track
    let added = a.total_cost() - before;
    assert!((added / b.total_cost() - 2.0).abs() < 0.1, "{added} against {}", b.total_cost());
    let j = a.nodes_info().chunks(10).find(|c| c[0] == start).unwrap()[4];
    assert_eq!(j, 3.0, "the old track end is a junction now");
    // stations: each gets its twin on the other pair, same name
    let mut stops = vec![];
    for (piece, at, name) in [(local[1], 60.0, "First"), (local[2], 400.0, "Last")] {
        let p = piece; let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(p), at);
        assert_eq!(a.add_station(2, p, x, y, 200, name.into()), 0, "{} on {p} at {:?}: {:?} {:?}", a.issues(), a.w.net.project(p, x, y), a.w.net.edge_data(p), a.edge_pis(p));
        let named: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).collect();
        assert_eq!(named.len(), 2, "{name} and its twin");
        let local_one = *named.iter().min_by(|&&p, &&q| { let d = |k: u32| b.w.net.project(e, a.w.net.node_x[k as usize], a.w.net.node_y[k as usize]).1; d(p).total_cmp(&d(q)) }).unwrap();
        stops.push(local_one);
    }
    // a station on a joint node of the drawn pair gets its twin too
    let joint = (0..a.w.net.node_count() as u32).find(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 2 && a.w.net.node_platform[k as usize] == 0 && b.w.net.project(e, a.w.net.node_x[k as usize], a.w.net.node_y[k as usize]).1 < 0.5 && (1000.0..11000.0).contains(&a.w.net.node_x[k as usize])).unwrap();
    assert_eq!(a.add_station(1, joint, 0.0, 0.0, 200, "Joint".into()), 0, "{}", a.issues());
    assert_eq!((0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == "Joint").count(), 2);
    assert_eq!(a.undo(), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &stops, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "Local".into()), 0, "{}", a.issues());
    assert!(a.line_path(id).iter().all(|&p| on_local(&a, p >> 1)));
    // deleting an express piece leaves the drawn pair
    assert_eq!(a.delete_edge(express[express.len() / 2]), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[id as usize]);
}

/// T-423: quad drawn through points. A chain of four segments carries both pairs on at every
/// joint (no swing, no junction: two loose ends at each end of the chain, plain joints
/// between, the express pair 8 m beside all the way, each local segment one arc with the
/// express arc concentric); a segment from the express pair's loose end carries both on the
/// other way; one ending on other track joins it as a junction; twin stations go across a
/// joint and on the loose ends; a crossover goes on it.
#[test]
fn quad_drawn_through_points_carries_both_pairs_on() {
    let mut a = api();
    a.set_route_tangent(true);
    let net_nodes = |a: &TrackApi| (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k)).collect::<Vec<u32>>();
    let nearest = |a: &TrackApi, x: f64, y: f64| -> u32 { net_nodes(a).into_iter().min_by(|&p, &q| { let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y); d(p).total_cmp(&d(q)) }).unwrap() };
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(600.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 2);
    let (mut end, mut heading) = (a.route_end(), 0.0f64);
    let mut local = vec![live_edges(&a)[0]];
    let mut express = vec![live_edges(&a)[1]];
    for turn in [20.0f64, -15.0, 10.0] {
        let off = turn.to_radians();
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + 500.0 * (heading + off).cos(), y0 + 500.0 * (heading + off).sin());
        let before = live_edges(&a);
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 4, false), 0, "{turn}: {}", a.issues());
        let new: Vec<u32> = live_edges(&a).into_iter().filter(|e| !before.contains(e)).collect();
        assert_eq!(new.len(), 2, "{turn}: one edge each pair");
        let (l, e) = if a.w.net.edge_b[new[0] as usize] == a.route_end() { (new[0], new[1]) } else { (new[1], new[0]) };
        // the local arc is the tangent arc, the express one concentric with it
        let want = 500.0 / (2.0 * off.abs().sin());
        let rl = a.edge_pis(l)[5];
        assert!((rl - want).abs() < 0.01 * want, "{turn}: local {rl} against {want}");
        let pe = a.edge_pis(e);
        assert_eq!(pe.len(), 9, "{turn}: one corner on the express pair too");
        let inside = turn > 0.0;
        assert!((pe[5] - if inside { rl - QUAD_SPACING } else { rl + QUAD_SPACING }).abs() < 0.5, "{turn}: express radius {} beside {rl}", pe[5]);
        local.push(l);
        express.push(e);
        heading += 2.0 * off;
        end = a.route_end();
    }
    // plain joints, no junctions; four loose ends
    let ports: Vec<u8> = net_nodes(&a).iter().map(|&k| a.w.net.node_ports[k as usize].n as u8).collect();
    assert!(ports.iter().all(|&p| p <= 2), "{ports:?}");
    assert_eq!(ports.iter().filter(|&&p| p == 1).count(), 4, "{ports:?}");
    for &p in &express {
        for q in a.edge_render(p).chunks(3) {
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off");
        }
    }
    // the express pair is to the left: its last loose end 8 m left of the local's
    let (ex, ey) = (a.w.net.node_x[end as usize] - QUAD_SPACING * heading.sin(), a.w.net.node_y[end as usize] + QUAD_SPACING * heading.cos());
    let ex_end = nearest(&a, ex, ey);
    assert!((a.w.net.node_x[ex_end as usize] - ex).hypot(a.w.net.node_y[ex_end as usize] - ey) < 0.05);
    // twin stations: across the first joint, and on the far loose ends
    let (sx, sy, _) = geom::pos_at(a.w.net.edge_pieces(local[0]), a.w.net.edge_len[local[0] as usize] - 40.0);
    assert_eq!(a.add_station(2, local[0], sx, sy, 200, "Joint".into()), 0, "{}", a.issues());
    let named = |a: &TrackApi, name: &str| net_nodes(a).into_iter().filter(|&k| a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).count();
    assert_eq!(named(&a, "Joint"), 2, "the station and its twin across the joint");
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 120, "End".into()), 0, "{}", a.issues());
    assert_eq!(named(&a, "End"), 2, "the loose ends both");
    assert!(a.w.net.node_platform[ex_end as usize] > 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    // a crossover on the new quad track, mid-segment
    let mid = local[2];
    let (cx, cy, _) = geom::pos_at(a.w.net.edge_pieces(mid), a.w.net.edge_len[mid as usize] / 2.0);
    assert_eq!(a.add_crossover(mid, cx, cy), 0, "{}", a.issues());
    assert_eq!(a.undo(), 0);
    // from the express pair's loose end at the start, the other way: both pairs carry on
    let ex0 = nearest(&a, 0.0, QUAD_SPACING);
    let before = live_edges(&a);
    assert_eq!(a.add_route(&from_node(ex0, -500.0, -150.0), &[], 4, false), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), before.len() + 2);
    assert!(net_nodes(&a).iter().all(|&k| a.w.net.node_ports[k as usize].n != 1 || a.w.net.node_x[k as usize] < -100.0 || a.w.net.node_x[k as usize] > 100.0), "no loose end left at the start");
    // ending on a plain double track ahead: both pairs meet it at one new junction
    let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
    let (ux, uy) = (heading.cos(), heading.sin());
    // the ghost of a segment from the loose end: both pairs, no step drawn between their ends
    let pv = a.preview_route(&from_node(end, x0 + 400.0 * (heading + 0.1).cos(), y0 + 400.0 * (heading + 0.1).sin()), &[], 4, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    let sp = a.preview_render_speeds();
    assert_eq!(sp.len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    assert_eq!(sp.iter().filter(|&&v| v < 0.0).count(), 1);
    assert_eq!(a.route_arc().len(), 3, "the local arc's circle");
    let (px, py) = (x0 + 700.0 * ux, y0 + 700.0 * uy);
    let cross: Vec<f64> = free(px + 1500.0 * uy, py - 1500.0 * ux).iter().chain(free(px - 1500.0 * uy, py + 1500.0 * ux).iter()).copied().collect();
    assert_eq!(a.add_route(&cross, &[], 2, false), 0, "{}", a.issues());
    let other = *live_edges(&a).last().unwrap();
    let (tx, ty) = (px - 200.0 * uy, py + 200.0 * ux);
    let to = vec![1.0, end as f64, 0.0, 0.0, 0.0, 2.0, other as f64, tx, ty, 0.0];
    assert_eq!(a.add_route(&to, &[], 4, false), 0, "{}", a.issues());
    let meet = a.route_end();
    assert_eq!(a.w.net.node_ports[meet as usize].n, 4, "the crossed track, the local and the express pair");
    assert_eq!(net_nodes(&a).iter().filter(|&&k| a.w.net.node_ports[k as usize].n == 1).count(), 4, "the new start's two and the crossed track's two");
}

/// T-338: a crossover on quad track joins the two pairs mid-stretch, so a line can change from
/// the local pair to the express pair there; it runs at 50 km/h at most over it, is priced as
/// two junctions, has a fixed shape, saves, deletes as a whole and undoes in one step. Track
/// with nothing beside it is refused.
#[test]
fn crossover_joins_the_pairs() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    let edge_at = |t: &TrackApi, x: f64, y: f64| -> u32 {
        live_edges(t).into_iter().filter(|&e| !t.w.net.edge_crossover[e as usize]).min_by(|&p, &q| t.w.net.project(p, x, y).1.total_cmp(&t.w.net.project(q, x, y).1)).unwrap()
    };
    let station_at = |t: &TrackApi, x: f64, y: f64| -> u32 {
        t.nodes_info().chunks(10).filter(|c| c[7] > 0.0).min_by(|p, q| (p[1] - x).hypot(p[2] - y).total_cmp(&(q[1] - x).hypot(q[2] - y))).unwrap()[0] as u32
    };
    for x in [500.0, 3500.0] {
        assert_eq!(a.add_station(2, edge_at(&a, x, 0.0), x, 0.0, 120, format!("S{x}")), 0, "{}", a.issues());
    }
    // the crossovers placed past them (T-733) out of the way, to place one by hand
    while let Some(&x) = a.crossover_edges().first() {
        assert_eq!(a.delete_edge(x), 0, "{}", a.issues());
    }
    let (w_local, e_express) = (station_at(&a, 500.0, 0.0), station_at(&a, 3500.0, QUAD_SPACING));
    assert!((a.w.net.node_y[e_express as usize] - QUAD_SPACING).abs() < 0.1);
    let id = a.new_line_id();
    assert_ne!(a.set_line(id, &[w_local, e_express], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "X".into()), 0, "no way across yet");
    // nothing beside plain double track
    let mut plain = api();
    assert_eq!(plain.add_route(&ends, &[], 2, false), 0);
    let only = live_edges(&plain)[0];
    assert_eq!(plain.preview_crossover(only, 2000.0, 0.0)[0], 0.0);
    assert_eq!(plain.issue_kinds(), "NoParallel");
    assert_ne!(plain.add_crossover(only, 2000.0, 0.0), 0);
    // on the local pair, in the middle
    let local = edge_at(&a, 2000.0, 0.0);
    let (edges0, cost0, cash0) = (live_edges(&a).len(), a.total_cost(), a.cash);
    let pv = a.preview_crossover(local, 2000.0, 0.0);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    let two_junctions = 2.0 * super::super::cost::junction_cost(0, false, false);
    assert!((pv[1] - two_junctions).abs() < 1e-6, "priced {} not {two_junctions}", pv[1]);
    assert_eq!(live_edges(&a).len(), edges0, "a preview changes nothing");
    assert_eq!(a.add_crossover(local, 2000.0, 0.0), 0, "{}", a.issues());
    assert!((a.total_cost() - cost0 - two_junctions).abs() < 1e-6);
    assert_eq!(a.cash, cash0, "a blueprint is free until constructed");
    let xs = a.crossover_edges();
    assert_eq!(xs.len(), 2);
    assert_eq!(live_edges(&a).len(), edges0 + 6, "four splits and two diagonals");
    for &x in &xs {
        let len = a.w.net.edge_len[x as usize];
        assert!(len > 110.0 && len < 130.0, "diagonal {len} m");
        assert!(a.w.net.edge_time[x as usize] >= len / V_CROSSOVER - 1e-6);
        assert_eq!(crossover_partner(&a.w.net, x), xs.iter().copied().find(|&o| o != x));
    }
    // a line from the local pair to the express pair now runs across it, at 50 km/h at most
    assert_eq!(a.set_line(id, &[w_local, e_express], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "X".into()), 0, "{}", a.issues());
    let path = a.line_path(id);
    let over: Vec<u32> = path.iter().map(|p| p >> 1).filter(|e| xs.contains(e)).collect();
    assert_eq!(over.len(), 1, "one diagonal: {path:?}");
    let run = &a.w.svc.lines[id as usize].runs[0];
    let k = run.segs.iter().position(|&p| p >> 1 == over[0]).unwrap();
    let (s0, s1) = (run.seg_off[k], run.seg_off[k] + a.w.net.edge_len[over[0] as usize]);
    let v = profile::v_max(&run.free.phases, s0, s1);
    assert!(v <= V_CROSSOVER + 1e-6, "{} km/h over the crossover", v * 3.6);
    // fixed shape
    assert_ne!(a.set_edge_pis(xs[0], &[]), 0);
    assert_eq!(a.issue_kinds(), "Crossover");
    let n = a.w.net.edge_a[xs[0] as usize];
    assert_ne!(a.move_node(n, a.w.net.node_x[n as usize] + 5.0, a.w.net.node_y[n as usize]), 0);
    // saved and loaded with the flag
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.crossover_edges().len(), 2);
    assert!((b.total_cost() - a.total_cost()).abs() < 1e-6);
    // deleting one diagonal takes the whole crossover; the line goes back to no route
    assert_eq!(a.remove_line(id), 0);
    assert_eq!(a.delete_edge(xs[0]), 0, "{}", a.issues());
    assert!(a.crossover_edges().is_empty());
    assert_eq!(a.undo(), 0);
    assert_eq!(a.crossover_edges().len(), 2);
    // constructing one diagonal builds both (T-391)
    assert_eq!(b.crossover_partners(), vec![xs[1], xs[0]]);
    b.cash = 1e6;
    assert_eq!(b.construct(&[xs[0]], &[]), 0, "{}", b.issues());
    assert!(xs.iter().all(|&x| b.w.net.edge_built[x as usize]));
    // undoing the placement takes it all away
    assert_eq!(a.undo(), 0); // the line's removal
    assert_eq!(a.undo(), 0); // the line
    assert_eq!(a.undo(), 0); // the crossover
    assert!(a.crossover_edges().is_empty());
    assert_eq!(live_edges(&a).len(), edges0);
    assert!((a.total_cost() - cost0).abs() < 1e-6);
}

/// T-244: dragging a corner of a piece moves the joints next to it along, so the stroke
/// reshapes as if it were still one edge.
#[test]
fn reshaping_a_piece_moves_its_joints() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let edges = live_edges(&a);
    let mid = edges[1];
    let moved = [2700.0, -900.0, 0.0, 0.0];
    let pv = a.preview_edge(mid, &moved);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    assert_eq!(a.set_edge_pis(mid, &moved), 0, "{}", a.issues());
    let mut whole = pis.clone();
    whole[4..8].copy_from_slice(&moved);
    let (b, e) = one_edge(&whole);
    same_track(&a, &b, e, 0.01);
    assert_eq!(live_edges(&a), edges, "the same pieces");
    assert_eq!(a.undo(), 0);
    let (b, e) = one_edge(&pis);
    same_track(&a, &b, e, 0.005);
    // a joint drags like any blueprint node, keeping its heading
    let j = a.w.net.edge_b[edges[0] as usize];
    let (x, y) = (a.w.net.node_x[j as usize], a.w.net.node_y[j as usize]);
    assert_eq!(a.move_node(j, x + 40.0, y + 60.0), 0, "{}", a.issues());
    // a line still routes over the pieces through the joints
    let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[w, east], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.line_path(id).len(), 4);
}

/// T-1161: removing the piece past a station on the track's last stretch stops at the platform's
/// far end, so the station stays where it was; between two stations both keep theirs.
#[test]
fn removing_past_a_station_stops_at_its_platform_end() {
    for built in [false, true] {
        let mut a = api();
        a.set_cash(1e6);
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        for x in [1000.0, 3000.0] {
            let e = (0..a.w.net.edge_count() as u32).find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            }).unwrap();
            assert_eq!(a.add_station(2, e, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
        }
        if built {
            assert_eq!(a.construct_all(), 0, "{}", a.issues());
        }
        let station = |t: &TrackApi, x: f64| (0..t.w.net.node_count() as u32).find(|&n| t.w.net.node_ok(n) && t.w.net.node_platform[n as usize] > 0 && (t.w.net.node_x[n as usize] - x).abs() < 1e-6).unwrap();
        let edge_between = |t: &TrackApi, p: u32, q: u32| live_edges(t).into_iter().find(|&e| {
            let (ea, eb) = (t.w.net.edge_a[e as usize], t.w.net.edge_b[e as usize]);
            (ea == p && eb == q) || (ea == q && eb == p)
        }).unwrap();
        let loose = |t: &TrackApi| -> Vec<f64> {
            let mut v: Vec<f64> = (0..t.w.net.node_count() as u32).filter(|&n| t.w.net.node_ok(n) && t.w.net.node_ports[n as usize].n == 1).map(|n| t.w.net.node_x[n as usize]).collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let (s1, s3) = (station(&a, 1000.0), station(&a, 3000.0));
        let far = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && a.w.net.node_x[n as usize] == 4000.0).unwrap();
        // the last stretch, past the station at 3 km
        assert_eq!(a.delete_edge(edge_between(&a, s3, far)), 0, "{}", a.issues());
        assert_eq!(a.w.net.node_x[s3 as usize], 3000.0, "the station keeps its place");
        assert_eq!(a.w.net.node_ports[s3 as usize].n, 2);
        let ends = loose(&a);
        assert_eq!(ends.len(), 2, "{ends:?}");
        let past = ends[1] - 3000.0 - 100.0;
        assert!(past > 1.0 && past <= super::links::STATION_TRIM_MARGIN_M + 0.01, "track ends {past} m past the platform");
        // between the two stations: both stay, with the platform's length of track past each
        let mid = edge_between(&a, s1, s3);
        assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
        assert_eq!((a.w.net.node_x[s1 as usize], a.w.net.node_x[s3 as usize]), (1000.0, 3000.0));
        assert_eq!((a.w.net.node_ports[s1 as usize].n, a.w.net.node_ports[s3 as usize].n), (2, 2));
        let ends = loose(&a);
        assert_eq!(ends.len(), 4, "{ends:?}");
        assert!(ends[1] > 1101.0 && ends[1] <= 1104.01 && ends[2] < 2899.0 && ends[2] >= 2895.99, "{ends:?}");
        if !built {
            // undone in two steps, back to the whole track
            assert_eq!(a.undo(), 0);
            assert_eq!(a.undo(), 0);
            assert_eq!(loose(&a), [0.0, 4000.0]);
        }
    }
}
