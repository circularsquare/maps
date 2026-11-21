use super::*;

#[test]
fn blueprint_construct_run() {
    let mut a = api();
    // A straight 3 km route, then a station at each end and one in the middle.
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    let pv = a.preview_route(&ends, &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert!((pv[1] - 3.0 * 37.5).abs() < 0.5, "cost {}", pv[1]);
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.built_cost(), 0.0);
    assert!((a.blueprint_cost() - 112.5).abs() < 0.5);
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
    assert!((6000.0 - a.cash() - (112.5 + 225.0 + trains)).abs() < 0.5, "paid {}", 6000.0 - a.cash());
    assert!((a.last_charge() - (112.5 + 225.0 + trains)).abs() < 0.5);
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
    assert_eq!(a.money_params(), vec![CAR_PRICE, 2.0, ECONOMY, 4.0, 3.75, 3.0, 5.0, 350.0, 500.0, 0.6, 1.5, 0.6, 6.0, 30.0, BUS_ROUTE_COST_PER_KM, 500.0, 1000.0, 1300.0, 8.0, 30.0, 100.0, 10.0, 2000.0, 0.6, 15.0, 30.0, 50.0, 75.0, BRT_COST_PER_KM, UPKEEP_BRT_KM, crate::track::streets::lane_kmh(true)]);
    assert!((a.running_cost(0) - (4.0 * train_km + 2.0 * train_km * cars) * ECONOMY / 1e6).abs() < 1e-9);
    // T-2669: a line in a country at 0.39x its prices runs at 0.39x; the list is replaced, and a
    // load clears it
    let world = a.running_cost(0);
    a.set_line_run_factors(&[id as f64, 0.39, 99.0, 0.5]);
    assert!((a.running_cost(0) - 0.39 * world).abs() < 1e-9 && a.line_run_factor(id) == 0.39);
    a.set_line_run_factors(&[]);
    assert!((a.running_cost(0) - world).abs() < 1e-12 && a.line_run_factor(id) == 1.0);
    // Upkeep (T-867, T-1931, T-2619): $35k a km of double track on the ground and $50k a station at
    // grade a game day, all on the one line; built track no line uses goes to the rest.
    let up = a.upkeep();
    assert!((up[3] - 3.0).abs() < 0.01 && up[4] == 3.0, "{up:?}");
    assert!((up[1] - 0.035 * up[3]).abs() < 1e-9 && (up[2] - 0.15).abs() < 1e-9);
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
    // T-2603: one line's trips over a window are the drawn trips of each hour in it
    let t0 = 8.0 * 3600.0;
    let one = a.line_trips(id, t0, t0 + 3600.0);
    let all = a.trips(t0);
    assert_eq!(one.len(), all.len(), "{one:?} {all:?}");
    for (x, y) in one.chunks(4).zip(all.chunks(4)) {
        assert_eq!(x[0] as f32, y[0]);
        assert!((x[1] - t0 - y[1] as f64).abs() < 0.01 && (x[2] - t0 - y[2] as f64).abs() < 0.01, "{x:?} {y:?}");
    }
    assert!(a.line_trips(id, t0, t0 + 4.0 * 3600.0).len() > one.len());
    assert!(a.line_trips(id + 1, t0, t0 + 3600.0).is_empty());
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
fn blueprint_pieces_add_up_to_the_quote() {
    // T-2269: the pieces counted add to the quote, all of them to the blueprint's whole cost
    let mut a = api();
    let line = |x0: f64, y0: f64, x1: f64, y1: f64| -> Vec<f64> { free(x0, y0).iter().chain(free(x1, y1).iter()).copied().collect() };
    assert_eq!(a.add_route(&line(0.0, 0.0, 1000.0, 0.0), &[], 2, false), 0);
    let n0 = a.nodes_info()[0] as u32;
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "A".into()), 0, "{}", a.issues());
    assert_eq!(a.add_route(&line(500.0, -300.0, 500.0, 300.0), &[], 2, false), 0);
    let e = a.edges_info()[0];
    let ends: Vec<f64> = [2.0, e, 400.0, 0.0, 0.0].iter().chain(free(400.0, -600.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let sums = |a: &mut TrackApi| {
        let rows = a.blueprint_piece_costs();
        let on: f64 = rows.chunks_exact(5).filter(|r| r[4] == 1.0).map(|r| r[3]).sum();
        let all: f64 = rows.chunks_exact(5).map(|r| r[3]).sum();
        let kinds: Vec<u8> = rows.chunks_exact(5).map(|r| r[0] as u8).collect();
        (on, all, kinds, a.blueprint_cost_items().chunks_exact(7).map(|r| r[6]).sum::<f64>())
    };
    let (on, all, kinds, quote) = sums(&mut a);
    assert!([0, 1, 2, 3].iter().all(|k| kinds.contains(k)), "track, a station, a junction and a flat crossing: {kinds:?}");
    assert!((on - quote).abs() < 1e-8 && (all - a.blueprint_cost()).abs() < 1e-8, "{on} {all} {quote}");
    // one route switched off: what is counted follows the quote, the whole stays
    let off: Vec<u32> = a.edges_info().chunks(10).map(|c| c[0] as u32).skip(1).take(1).collect();
    assert_eq!(a.set_blueprint_off(&off, &[], true), 0, "{}", a.issues());
    let (on2, all2, _, quote2) = sums(&mut a);
    assert!(on2 < on - 1.0 && (on2 - quote2).abs() < 1e-8 && (all2 - all).abs() < 1e-8, "{on2} {quote2} {all2}");
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
    // without the station, it keeps a piece under its platform (T-1699)
    assert_eq!(a.delete_pieces(&ids, &[]), 0, "{}", a.issues());
    assert_eq!((a.nodes_info().len() / 10, a.edges_info().len() / 10), (2, 1));
    assert_eq!(a.undo(), 0);
    // constructed (T-1649): nothing refunded, and one undo brings it back, free; the kept piece
    // stays constructed
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let cash = a.cash();
    assert_eq!(a.delete_pieces(&ids, &[]), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash);
    let kept: Vec<u32> = a.edges_info().chunks(10).map(|c| c[0] as u32).collect();
    assert!(kept.len() == 1 && a.w.net.edge_built[kept[0] as usize]);
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!(a.edges_info().len() / 10, 2);
    assert!(a.w.net.edge_built[ids[0] as usize]);
    assert_eq!(a.cash(), cash);
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
    assert_ne!(a.set_edge_pis(e, &[2000.0, 1000.0, 20.0, 0.0]), 0);
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    // the whole stretch one level down, shape kept
    assert_eq!(a.set_edge_level(e, -1), 0, "{}", a.issues());
    assert_eq!(a.edges_info()[8], -1.0);
    assert!((a.edge_pis(e)[5] - 400.0).abs() < 1e-6);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.edges_info()[5], len0);
    // a set radius the edit leaves alone stays set when its corner moves where it no longer fits,
    // narrowed by the fit (T-1898); one the edit sets becomes the widest that fits (T-1895)
    assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 1500.0, 0.0]), 0, "{}", a.issues());
    assert_eq!(a.set_edge_pis(e, &[300.0, 1500.0, 1500.0, 0.0]), 0, "{}", a.issues());
    assert_eq!(a.edge_pis(e)[3], 1500.0, "kept as the most");
    assert!(a.edge_pis(e)[5] < 1500.0, "narrowed to {}", a.edge_pis(e)[5]);
    assert_eq!(a.set_edge_pis(e, &[300.0, 1500.0, 1700.0, 0.0]), 0, "{}", a.issues());
    let r = a.edge_pis(e)[3];
    assert!(r > 100.0 && r < 1700.0, "narrowed to {r}");
    assert_eq!(a.undo(), 0);
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
