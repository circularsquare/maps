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

type Seen = (Vec<f64>, Vec<f64>, Vec<f64>, u64, [u32; 7], Vec<(bool, Vec<u32>)>, Vec<Vec<crate::track::service::Era>>, Vec<Vec<(u8, f64, f64)>>);

/// Everything a state reads of the model, the money, and where the trains running through an edit are drawn from.
fn seen(a: &TrackApi) -> Seen {
    let s = &a.w.svc;
    (a.nodes_info(), a.edges_info(), a.lines_info(), a.cash.to_bits(), a.fleet, s.lines.iter().map(|l| (l.running, l.trains.to_vec())).collect(), s.eras.clone(), s.shifts.clone())
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

/// T-4272: a station on a running line's track can move that line's trains (its platforms change the track's capacity,
/// which the settle shares out): answered all the same, its services and the trains they need worked out after, as the
/// whole path's, the trains running through it noted where they were, and so through its undo and redo.
#[test]
fn a_station_on_a_running_lines_track_is_answered_and_ends_as_the_whole_path_does() {
    let (mut a, _) = line_and_blueprint();
    let (mut b, _) = line_and_blueprint();
    b.set_answer_first(false);
    for x in [&mut a, &mut b] {
        x.set_now(8.0 * 3600.0);
    }
    let start = seen(&b);
    assert_eq!(station_at(&mut a, 3000.0, 0.0), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, 3000.0, 0.0), 0, "{}", b.issues());
    assert!(a.owes(), "answered before its services");
    assert!(a.settle_owed());
    assert_eq!(seen(&a), seen(&b));
    let moved = seen(&b);
    assert!((&moved.5, &moved.6, &moved.7) != (&start.5, &start.6, &start.7), "the station moved the line's trains or where they run from");
    for k in 0..2 {
        assert_eq!(a.undo(), 0, "undo {k}: {}", a.issues());
        assert_eq!(b.undo(), 0, "undo {k}: {}", b.issues());
        assert_eq!(seen(&a), seen(&b), "after undo {k}");
        assert_eq!(a.redo(), 0, "redo {k}: {}", a.issues());
        assert_eq!(b.redo(), 0, "redo {k}: {}", b.issues());
        assert_eq!(seen(&a), seen(&b), "after redo {k}");
        assert_eq!(a.undo(), 0);
        assert_eq!(b.undo(), 0);
    }
    assert_eq!(seen(&a).2, start.2, "undone");
}

/// A point halfway along the detour of `headway_and_detour` (its piece reaching furthest north).
fn detour_mid(a: &TrackApi) -> (f64, f64) {
    let e = live_edges(a).into_iter().max_by(|&p, &q| {
        let top = |e: u32| a.edge_render(e).chunks(3).map(|c| c[1]).fold(f64::MIN, f64::max);
        top(p).total_cmp(&top(q))
    }).unwrap();
    let pts = a.edge_render(e);
    let k = pts.len() / 3 / 2;
    (pts[3 * k], pts[3 * k + 1])
}

/// A headway line West to East on constructed track, with a constructed detour north of its middle (x 1000 to 5000 by
/// way of (3000, 4000)) that it does not run on.
fn headway_and_detour(cash: f64) -> (TrackApi, u32) {
    let mut a = api();
    a.set_cash(6000.0);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    let joint = |a: &mut TrackApi, x: f64| {
        assert_eq!(a.add_point(edge_under(a, x, 0.0), x, 0.0), 0, "{}", a.issues());
        let n = &a.w.net;
        (0..n.node_count() as u32).find(|&k| n.node_ok(k) && (n.node_x[k as usize] - x).abs() < 1.0 && n.node_y[k as usize].abs() < 1.0).unwrap()
    };
    let (j1, j2) = (joint(&mut a, 1000.0), joint(&mut a, 5000.0));
    let ends = [1.0, j1 as f64, 0.0, 0.0, 0.0, 1.0, j2 as f64, 0.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[3000.0, 4000.0, 0.0, 0.0], 2, false), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    crate::track::tests::terminals_at_dead_ends(&mut a.w);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 4, 2, 2, 30.0, 420.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(id, 240, 480, 900), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    a.set_cash(cash);
    a.set_now(8.0 * 3600.0);
    (a, id)
}

/// T-4272 (her 11-25: "yes can buy trains after confirm, thats fine to make money go negative"): deleting track a
/// running headway line uses sends it the long way round; the delete is answered at once and the trains the longer round
/// trip needs are bought once its services are worked out, as the whole path buys them at the click, through its undo
/// (which gives no money back, as today) and redo. An edit that can move running trains is owed alone: a station owed
/// before it (on the detour, no line there yet) is worked out first, and a station clicked while the delete's services
/// are owed (on the detour, the line there now) works those out first.
#[test]
fn trains_a_delete_needs_are_bought_after_its_answer() {
    let (mut a, line) = headway_and_detour(6000.0);
    let (mut b, _) = headway_and_detour(6000.0);
    b.set_answer_first(false);
    assert_eq!(seen(&a), seen(&b));
    let (cash, trains) = (a.cash, a.w.svc.lines[line as usize].trains);
    let (x, y) = detour_mid(&a);
    assert_eq!(station_at(&mut a, x, y), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, x, y), 0, "{}", b.issues());
    assert!(a.owes(), "no line on the detour yet");
    let mid = edge_under(&a, 3000.0, 0.0);
    assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
    assert_eq!(b.delete_edge(mid), 0, "{}", b.issues());
    assert!(a.owes(), "answered before its services");
    assert_eq!(a.cash, cash, "nothing bought at the click");
    let now = b.w.svc.lines[line as usize].trains;
    assert!(b.cash < cash && now.iter().max() > trains.iter().max(), "the long way round needs trains: {trains:?} to {now:?}, cash {cash} to {}", b.cash);
    let (x, y) = detour_mid(&a);
    assert_eq!(station_at(&mut a, x, y), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, x, y), 0, "{}", b.issues());
    assert!(a.owes(), "the station's own services owed");
    assert!(a.settle_owed());
    assert_eq!(seen(&a), seen(&b), "after all three");
    assert!(!a.take_trains_bought().is_empty(), "the trains bought, for the ledger");
    for k in 0..3 {
        assert_eq!(a.undo(), 0, "undo {k}: {}", a.issues());
        assert_eq!(b.undo(), 0, "undo {k}: {}", b.issues());
        assert_eq!(seen(&a), seen(&b), "after undo {k}");
    }
    assert_eq!(a.cash, b.cash, "the undo gives no money back, as on the whole path");
    for k in 0..3 {
        assert_eq!(a.redo(), 0, "redo {k}: {}", a.issues());
        assert_eq!(b.redo(), 0, "redo {k}: {}", b.issues());
        assert_eq!(seen(&a), seen(&b), "after redo {k}");
    }
}

/// T-4272: an undo right after an answered delete, before its services are worked out, ends as the whole path's: the
/// trains the delete needed are bought first, and kept.
#[test]
fn an_undo_before_the_services_buys_what_the_whole_path_bought() {
    let (mut a, _) = headway_and_detour(6000.0);
    let (mut b, _) = headway_and_detour(6000.0);
    b.set_answer_first(false);
    let mid = edge_under(&a, 3000.0, 0.0);
    assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
    assert_eq!(b.delete_edge(mid), 0, "{}", b.issues());
    assert!(a.owes());
    assert_eq!(a.undo(), 0, "{}", a.issues());
    assert_eq!(b.undo(), 0, "{}", b.issues());
    assert!(!a.owes());
    assert_eq!(seen(&a), seen(&b));
    assert!(!a.settle_owed(), "what the services touched went with the undo's own result");
}

/// The running lines short of the cars they need, as a save running more than its fleet, or lines a reload re-derived,
/// can leave them before a click.
fn short_fleet(a: &mut TrackApi) {
    let need = cars_needed(&a.w);
    a.fleet = std::array::from_fn(|k| need[k].saturating_sub(4));
    assert!(trains_to_buy(&a.w, a.fleet) > 1.0, "short");
}

/// T-4272: a delete is never refused for the trains it needs, nor a station's ghost for money: with the cash short and
/// the running lines already short of cars, the whole path refuses both (the old rule, which the ghost followed), and an
/// answered delete buys them all the same, the cash going below zero by exactly what the whole path with the cash to
/// spare spends.
#[test]
fn trains_bought_after_an_answer_may_take_the_cash_below_zero() {
    let spare = 5000.0;
    let (mut a, _) = headway_and_detour(1.0);
    let (mut b, _) = headway_and_detour(1.0 + spare);
    let (mut c, _) = headway_and_detour(1.0);
    b.set_answer_first(false);
    c.set_answer_first(false);
    for x in [&mut a, &mut b, &mut c] {
        short_fleet(x);
    }
    // on the headway line's own track, beside the piece deleted
    let (x, y) = (2000.0, 0.0);
    assert_ne!(station_at(&mut c, x, y), 0, "the whole path with the cash short refused the station");
    assert!(c.money_short.is_some(), "for money: {}", c.issues());
    let mid = edge_under(&a, 3000.0, 0.0);
    assert_ne!(c.delete_edge(mid), 0, "and the delete, as before");
    assert_eq!(a.preview_station(2, edge_under(&a, x, y), x, y, 200), 0, "the ghost is not refused for money: {}", a.issues());
    assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
    assert_eq!(b.delete_edge(mid), 0, "{}", b.issues());
    assert!(a.settle_owed());
    assert!(a.cash < 0.0, "below zero: {}", a.cash);
    assert!((b.cash - a.cash - spare).abs() < 1e-9, "{} against {}", a.cash, b.cash);
    assert_eq!(a.fleet, b.fleet);
}

/// T-4272 (review30): a station clicked while the running lines already lack cars is charged that shortfall once its
/// services are worked out, as the whole path charges it at the click (its answer never gives the cars away), and books
/// it to no line, as none of their trains rose.
#[test]
fn a_shortfall_from_before_a_click_is_charged_as_on_the_whole_path() {
    let (mut a, _) = line_and_blueprint();
    let (mut b, _) = line_and_blueprint();
    b.set_answer_first(false);
    for x in [&mut a, &mut b] {
        short_fleet(x);
    }
    let cash = a.cash;
    assert_eq!(station_at(&mut a, 2000.0, 3000.0), 0, "{}", a.issues());
    assert_eq!(station_at(&mut b, 2000.0, 3000.0), 0, "{}", b.issues());
    assert!(b.cash < cash, "the whole path bought the cars at the click");
    assert!(a.owes(), "no running line there");
    assert!(a.settle_owed());
    assert_eq!(seen(&a), seen(&b));
    let got = a.take_trains_bought();
    assert!(got.len() == 2 && got[0] == -1.0 && (got[1] - (cash - a.cash)).abs() < 1e-9, "all to no line: {got:?}, spent {}", cash - a.cash);
}

/// A count line on its own constructed track 3 km south, running.
fn line_south(a: &mut TrackApi) -> u32 {
    let ends: Vec<f64> = free(0.0, -3000.0).iter().chain(free(6000.0, -3000.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let at = |a: &TrackApi, x: f64| {
        let n = &a.w.net;
        (0..n.node_count() as u32).find(|&k| n.node_ok(k) && (n.node_x[k as usize] - x).abs() < 1.0 && (n.node_y[k as usize] + 3000.0).abs() < 1.0).unwrap()
    };
    let (n0, n1) = (at(a, 0.0), at(a, 6000.0));
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "South West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "South East".into()), 0, "{}", a.issues());
    crate::track::tests::terminals_at_dead_ends(&mut a.w);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[n0, n1], 4, 2, 2, 30.0, 420.0, 0, 0x1f77b4, "South".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.w.svc.lines[id as usize].running);
    id
}

/// T-4272: the trains a delete's services buy are booked, for the ledger's regions, to the line whose trains rose: all of
/// what was spent, and none to a running line the delete left as it was.
#[test]
fn trains_bought_are_booked_to_the_line_that_needed_them() {
    let (mut a, line) = headway_and_detour(6000.0);
    let other = line_south(&mut a);
    a.set_cash(6000.0);
    let mid = edge_under(&a, 3000.0, 0.0);
    assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
    assert!(a.settle_owed());
    let spent = 6000.0 - a.cash;
    assert!(spent > 0.0, "the long way round needs trains");
    let got = a.take_trains_bought();
    assert!(got.len() == 2 && got[0] == line as f64 && (got[1] - spent).abs() < 1e-9, "all to line {line}, none to {other}: {got:?}, spent {spent}");
    assert_eq!(a.take_bought_vehicle(), 1, "trains");
    assert_eq!(a.take_bought_vehicle(), 0, "once");
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
