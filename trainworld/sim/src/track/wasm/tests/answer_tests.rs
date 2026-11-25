use super::*;

/// A running line West to East on constructed track, and blueprint track 3 km north of it that no line uses.
fn line_and_blueprint() -> (TrackApi, u32) {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    crate::track::tests::terminals_at_dead_ends(&mut a.w);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 4, 2, 2, 30.0, 420.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let ends: Vec<f64> = free(0.0, 3000.0).iter().chain(free(6000.0, 3000.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let blue = a.route_pieces[0];
    (a, blue)
}

/// Everything a state reads of the model, and the money.
fn seen(a: &TrackApi) -> (Vec<f64>, Vec<f64>, Vec<f64>, u64, [u32; 7], Vec<(bool, Vec<u32>)>) {
    (a.nodes_info(), a.edges_info(), a.lines_info(), a.cash.to_bits(), a.fleet, a.w.svc.lines.iter().map(|l| (l.running, l.trains.to_vec())).collect())
}

fn station_at(a: &mut TrackApi, x: f64, y: f64) -> u32 {
    let edge = edge_under(a, x, y);
    a.add_station(2, edge, x, y, 200, String::new())
}

/// T-4272: stations placed and track deleted where no running line goes are answered before their services are
/// worked out, and once those are (`settle_owed`) the model is what the whole path leaves, edit for edit, and so
/// after undoing them all.
#[test]
fn answered_edits_end_as_the_whole_path_does() {
    let (mut a, blue) = line_and_blueprint();
    let (mut b, _) = line_and_blueprint();
    b.set_answer_first(false);
    let start = seen(&b);
    assert_eq!(seen(&a), start);
    assert!(a.w.net.edge_ok(blue));
    for (k, x) in [1500.0, 4000.0].into_iter().enumerate() {
        assert_eq!(station_at(&mut a, x, 3000.0), 0, "{}", a.issues());
        assert!(a.owes(), "station {k} answered before its services");
        assert_eq!(station_at(&mut b, x, 3000.0), 0, "{}", b.issues());
        assert!(!b.owes());
    }
    // the track between the two stations, past their platforms
    let mid = live_edges(&a).into_iter().find(|&e| a.w.net.edge_a[e as usize] != a.w.net.edge_b[e as usize] && a.w.net.lines_on_edge(e).is_empty() && !a.w.net.edge_built[e as usize] && a.w.net.edge_len[e as usize] > 2000.0).expect("the middle piece");
    let ra = a.delete_edge(mid);
    let rb = b.delete_edge(mid);
    assert_eq!((ra, rb), (0, 0), "{} / {}", a.issues(), b.issues());
    assert!(a.owes());
    assert!(a.settle_owed());
    assert!(!a.owes());
    assert_eq!(seen(&a), seen(&b), "after the services were worked out");
    for k in 0..3 {
        assert_eq!(a.undo(), 0, "undo {k}: {}", a.issues());
        assert_eq!(b.undo(), 0, "undo {k}: {}", b.issues());
        assert_eq!(seen(&a), seen(&b), "after undo {k}");
    }
    assert_eq!(seen(&a), start, "all undone");
    for k in 0..3 {
        assert_eq!(a.redo(), 0, "redo {k}");
        assert_eq!(b.redo(), 0, "redo {k}");
        assert_eq!(seen(&a), seen(&b), "after redo {k}");
    }
}

/// T-4272: an answered edit's services are worked out by the next edit that needs them, before it reads them.
#[test]
fn the_next_edit_works_out_what_is_owed_first() {
    let (mut a, _) = line_and_blueprint();
    let (mut b, _) = line_and_blueprint();
    b.set_answer_first(false);
    assert_eq!(station_at(&mut a, 2000.0, 3000.0), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, 2000.0, 3000.0), 0, "{}", b.issues());
    assert!(a.owes());
    // a train count on the running line: a whole edit
    let line = a.lines_info()[0] as u32;
    assert_eq!(a.set_train_counts(line, 3, 3, 2), 0, "{}", a.issues());
    assert_eq!(b.set_train_counts(line, 3, 3, 2), 0, "{}", b.issues());
    assert!(!a.owes());
    assert_eq!(seen(&a), seen(&b));
}

/// T-4272: a station on a running line's track can move that line's trains (its platforms change the track's
/// capacity, which the settle shares out), and the trains it needs are paid at the click: it goes the whole way.
#[test]
fn a_station_on_a_running_lines_track_goes_the_whole_way() {
    let (mut a, _) = line_and_blueprint();
    let (mut b, _) = line_and_blueprint();
    b.set_answer_first(false);
    let cash = a.cash;
    assert_eq!(station_at(&mut a, 3000.0, 0.0), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, 3000.0, 0.0), 0, "{}", b.issues());
    assert!(!a.owes(), "worked out before its answer");
    assert_eq!(seen(&a), seen(&b));
    assert_eq!(a.cash, cash, "a blueprint station buys nothing (a count line needs no train more)");
}

/// T-4272: a station on constructed track constructs nothing: the re-pricing of the two pieces its split leaves is no
/// purchase (it was charged before, so the click counted as paid and waited for its rows).
#[test]
fn a_station_on_constructed_track_charges_nothing() {
    let (mut a, _) = line_and_blueprint();
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let cash = a.cash;
    for x in [700.0, 2300.0, 5100.0] {
        assert_eq!(station_at(&mut a, x, 3000.0), 0, "{}", a.issues());
        assert!(a.owes(), "no line runs there");
        assert_eq!(a.last_charge(), 0.0);
    }
    a.settle_owed();
    assert_eq!(a.cash, cash);
}

/// The platform node nearest (x, y).
fn platform_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_platform[k as usize] > 0)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}

/// `line_and_blueprint` with a planned line on the blueprint track, between stations at its ends.
fn planned_line() -> (TrackApi, u32) {
    let (mut a, _) = line_and_blueprint();
    for x in [300.0, 5700.0] {
        assert_eq!(station_at(&mut a, x, 3000.0), 0, "{}", a.issues());
        a.settle_owed();
    }
    let ends = [platform_near(&a, 300.0, 3000.0), platform_near(&a, 5700.0, 3000.0)];
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &ends, 4, 2, 2, 30.0, 420.0, 0, 0x1f77b4, "Planned".into()), 0, "{}", a.issues());
    assert!(!a.w.svc.lines[id as usize].running);
    (a, id)
}

/// T-4272: an answered station on a planned line's track owes that line's services. Whatever works them out first (an
/// edit refused for money, the Build tab's quote, which trials Construct all) leaves what they touched for
/// `settle_owed`, the worker's step after the answer, which reports it as the whole path's edit would have: the line's
/// profiles and its demand, so its drawn line, rows and trips follow.
#[test]
fn what_a_refused_edit_or_a_trial_worked_out_still_reaches_settle_owed() {
    for first in ["refused", "trial"] {
        let (mut a, planned) = planned_line();
        let (mut b, _) = planned_line();
        b.set_answer_first(false);
        assert_eq!(station_at(&mut a, 3000.0, 3000.0), 0, "{}", a.issues());
        assert_eq!(station_at(&mut b, 3000.0, 3000.0), 0, "{}", b.issues());
        assert!(a.owes(), "{first}: no running line there");
        let (lines, demand) = (b.dirty_lines(), b.dirty_demand_lines());
        assert!(lines.contains(&planned), "{first}: the whole path worked the planned line out again: {lines:?}");
        if first == "refused" {
            for x in [&mut a, &mut b] {
                x.set_cash(0.0);
                assert_ne!(x.construct_all(), 0, "{first}: refused for money");
            }
        } else {
            for x in [&mut a, &mut b] {
                x.blueprint_cost_items();
            }
        }
        assert!(!a.owes(), "{first}: worked out first");
        assert!(a.settle_owed(), "{first}: left for the worker's step");
        let got = a.dirty_lines();
        assert!(lines.iter().all(|l| got.contains(l)), "{first}: {got:?} against {lines:?}");
        assert_eq!(a.take_settled_demand(), demand, "{first}");
        assert!(!a.settle_owed(), "{first}: once");
        assert_eq!(seen(&a), seen(&b), "{first}");
    }
}

/// A station clicked on quad track at (x, y): on the piece nearest, never a crossover.
fn quad_station(a: &mut TrackApi, x: f64, y: f64) -> u32 {
    let e = live_edges(a).into_iter().filter(|&e| !a.w.net.edge_crossover[e as usize]).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
    a.add_station(2, e, x, y, 200, String::new())
}

/// A 4 km quad stretch with a planned line on its drawn pair between West, Mid and East, on both pairs (T-3333) once
/// every stop has a station beside it.
fn quad_planned(headway: bool) -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    let stops: Vec<u32> = [500.0, 2000.0, 3500.0].iter().map(|&x| {
        assert_eq!(quad_station(&mut a, x, 0.0), 0, "{}", a.issues());
        a.settle_owed();
        platform_near(&a, x, 0.0)
    }).collect();
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &stops, 6, 4, 2, 30.0, 180.0, 0, 0xff0000, "Q".into()), 0, "{}", a.issues());
    if headway {
        assert_eq!(a.set_auto_headway(id, 300, 600, 0), 0, "{}", a.issues());
    }
    assert_eq!(a.set_line_pairs(id, true), 0, "{}", a.issues());
    (a, id)
}

/// T-4272: a station that gives a planned line its twin (T-3333) halves the line's trains in its own edit, which the
/// services and the undo must read from before it: it goes the whole way, and its undo gives the line its trains back
/// as the whole path's does.
#[test]
fn a_station_making_a_twin_goes_the_whole_way() {
    for headway in [false, true] {
        let (mut a, l) = quad_planned(headway);
        let (mut b, _) = quad_planned(headway);
        b.set_answer_first(false);
        let rates = |x: &TrackApi| (x.w.net.line_trains.clone(), x.w.net.line_auto.clone(), x.w.net.line_tph.clone(), x.w.net.line_phase.clone(), x.w.net.settle_kept.clone());
        let whole = a.w.net.line_trains[l as usize];
        for x in [500.0, 2000.0] {
            assert_eq!(quad_station(&mut a, x, QUAD_SPACING), 0, "{}", a.issues());
            assert_eq!(quad_station(&mut b, x, QUAD_SPACING), 0, "{}", b.issues());
            assert!(a.owes(), "no twin yet: nothing set");
            a.settle_owed();
        }
        assert_eq!(a.line_twin(l), -1);
        assert_eq!(quad_station(&mut a, 3500.0, QUAD_SPACING), 0, "{}", a.issues());
        assert_eq!(quad_station(&mut b, 3500.0, QUAD_SPACING), 0, "{}", b.issues());
        assert!(a.line_twin(l) >= 0, "the station's own edit made the twin");
        assert!(!a.owes(), "worked out before its answer");
        assert_eq!((seen(&a), rates(&a)), (seen(&b), rates(&b)), "headway {headway}");
        for k in 0..3 {
            assert_eq!(a.undo(), 0, "undo {k}: {}", a.issues());
            assert_eq!(b.undo(), 0, "undo {k}: {}", b.issues());
            assert_eq!((seen(&a), rates(&a)), (seen(&b), rates(&b)), "headway {headway}, after undo {k}");
        }
        assert_eq!(a.w.net.line_trains[l as usize], whole, "the whole line's trains again");
    }
}
