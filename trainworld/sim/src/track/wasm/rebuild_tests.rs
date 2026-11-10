//! T-3059: rebuilding constructed track in place at another level or track type.

use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// A built 8 km run with a station 1 km from each end and a line over it, 3 pieces of track.
fn run() -> (TrackApi, [u32; 2], u32) {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let mut st = [0; 2];
    for (i, x) in [1000.0, 7000.0].into_iter().enumerate() {
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{i}")), 0, "{}", a.issues());
        st[i] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    }
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.w.svc.lines[line as usize].running);
    (a, st, line)
}

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}

fn levels(a: &TrackApi, e: u32) -> Vec<i8> {
    let d = a.w.net.edge_data(e).unwrap();
    d.pis.iter().chain(d.thru.iter()).map(|p| p.level).collect()
}

/// The whole run and both stations as a viaduct: the same ids and names, the line still running, the
/// full new price taken, no undo, and the quote said the same beforehand.
#[test]
fn the_whole_run_rebuilds_as_a_viaduct_for_its_full_price() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let names: Vec<String> = st.iter().map(|&n| a.w.net.station_data(n).unwrap().name).collect();
    let q = a.rebuild_quote(&edges, &st, 1, 0);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0 && q[3] == 0.0, "{q:?} {}", a.issues());
    assert_eq!(q.len(), 4 + edges.len(), "the changed edges follow the (no) refused ones");
    let (cash, built) = (a.cash(), a.built_cost());
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 0, "{}", a.issues());
    assert_eq!(a.last_skipped(), 0);
    assert!((cash - a.cash() - q[0]).abs() < 1e-9, "charged {} quoted {}", cash - a.cash(), q[0]);
    assert!((a.last_charge() - q[0]).abs() < 1e-9);
    let full: f64 = edges.iter().map(|&e| a.w.net.edge_cost[e as usize]).sum::<f64>() + st.iter().map(|&n| a.w.net.station_build_price(n)).sum::<f64>();
    assert!((q[0] - full).abs() < 1e-9, "the full price of all of it");
    assert!(a.built_cost() != built, "the built value moved with the price");
    assert_eq!(live(&a), edges, "no edge made or lost");
    for &e in &edges {
        assert!(a.w.net.edge_built[e as usize]);
        assert!(levels(&a, e).iter().all(|&l| l == 1), "{e}: {:?}", levels(&a, e));
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.w.net.station_data(n).unwrap().name, names[i]);
        assert!(a.w.net.node_built[n as usize] && a.w.net.node_level[n as usize] == 1);
    }
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running && !a.w.net.line_broken[line as usize]);
    assert!(!a.can_undo(), "final");
    // rebuilt again as it is: nothing to do, nothing charged
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 1);
    assert_eq!(a.last_skipped(), 0);
}

/// Cheaper track is charged in full too, and a viaduct that was already there costs nothing to leave.
#[test]
fn a_cheaper_rebuild_still_pays_the_full_new_price() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 0, "{}", a.issues());
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &st, 0, 0);
    assert!(q[0] > 0.0, "{q:?}");
    assert_eq!(a.rebuild_pieces(&edges, &st, 0, 0), 0, "{}", a.issues());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
}

#[test]
fn the_whole_run_rebuilds_as_high_speed() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &st, 99, 2);
    assert!(q[0] > 0.0 && q[1] as usize == edges.len() && q[2] == 2.0, "{q:?}");
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 2), 0, "{}", a.issues());
    assert!((cash - a.cash() - q[0]).abs() < 1e-9);
    for &e in &edges {
        assert!(a.w.net.edge_kind[e as usize].is_hsr() && a.w.net.edge_built[e as usize]);
        assert!(levels(&a, e).iter().all(|&l| l == 0), "level kept");
    }
    assert_eq!(a.line_stops(line), st);
    assert!(a.w.svc.lines[line as usize].running);
    // and back
    assert_eq!(a.rebuild_pieces(&edges, &st, 99, 1), 0, "{}", a.issues());
    assert!(edges.iter().all(|&e| !a.w.net.edge_kind[e as usize].is_hsr()));
}

/// Level and type in one rebuild.
#[test]
fn level_and_type_change_together() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &st, -1, 2), 0, "{}", a.issues());
    for &e in &edges {
        assert!(a.w.net.edge_kind[e as usize].is_hsr());
        assert!(levels(&a, e).iter().all(|&l| l == -1));
    }
}

/// A pick that leaves a station out cannot move the track beside it to a level the platform is not on:
/// those pieces are named by the quote, drawn by the app, and left as they are by the rebuild.
#[test]
fn pieces_that_cannot_change_are_named_and_left_alone() {
    let (mut a, st, line) = run();
    let edges = live(&a);
    let cash = a.cash();
    let q = a.rebuild_quote(&edges, &[], 1, 0);
    let refused: Vec<u32> = q[4..4 + q[3] as usize].iter().map(|&e| e as u32).collect();
    assert!(q[3] > 0.0 && q[3] as usize == refused.len(), "{q:?}");
    assert!(refused.iter().all(|e| edges.contains(e)));
    assert_eq!(a.cash(), cash, "a quote spends nothing");
    assert_eq!(a.w.net.station_data(st[0]).unwrap().name, "S0");
    let before: Vec<Vec<i8>> = edges.iter().map(|&e| levels(&a, e)).collect();
    let r = a.rebuild_pieces(&edges, &[], 1, 0);
    assert_eq!(a.last_skipped() as usize, refused.len(), "the same pieces");
    for (i, &e) in edges.iter().enumerate() {
        if refused.contains(&e) {
            assert_eq!(levels(&a, e), before[i], "{e} left as it was");
        }
    }
    assert!(a.w.net.node_level[st[0] as usize] == 0 && a.w.net.node_level[st[1] as usize] == 0, "stations stay");
    if r == 0 {
        assert!(cash > a.cash());
    }
    assert!(a.w.svc.lines[line as usize].running);
}

/// A short purse: nothing changes.
#[test]
fn a_short_purse_refuses_the_rebuild() {
    let (mut a, st, _) = run();
    let edges = live(&a);
    a.set_cash(0.001);
    assert_eq!(a.rebuild_pieces(&edges, &st, 1, 0), 1);
    assert!(a.money_short.is_some());
    assert!(edges.iter().all(|&e| levels(&a, e).iter().all(|&l| l == 0)));
}

/// Blueprint pieces in the pick are not rebuilt (Construct builds them).
#[test]
fn blueprint_pieces_are_not_rebuilt() {
    let mut a = api();
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let edges = live(&a);
    assert_eq!(a.rebuild_pieces(&edges, &[], 1, 0), 1);
    assert_eq!(a.rebuild_quote(&edges, &[], 1, 0)[..4], [0.0, 0.0, 0.0, 0.0]);
}
