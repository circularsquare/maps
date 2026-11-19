use super::*;
use crate::track::upgrades;

/// A straight double track 0..3000 (plane m) with a 200 m blueprint station at 1500.
fn station() -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e4);
    a.set_route_tangent(true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
    let n = &a.w.net;
    let e = (0..n.edge_count() as u32).find(|&e| n.edge_ok(e)).unwrap();
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 200, "Mid".into()), 0, "{}", a.issues());
    let n = &a.w.net;
    let st = (0..n.node_count() as u32).find(|&m| n.is_station(m)).unwrap();
    (a, st)
}

/// The live edge whose ends lie either side of x on the line y = 0.
fn edge_over(a: &TrackApi, x: f64) -> u32 {
    let n = &a.w.net;
    (0..n.edge_count() as u32)
        .find(|&e| n.edge_ok(e) && {
            let (xa, xb) = (n.node_x[n.edge_a[e as usize] as usize], n.node_x[n.edge_b[e as usize] as usize]);
            xa.min(xb) < x && xa.max(xb) > x
        })
        .unwrap()
}

fn station_at(a: &TrackApi, x: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32).find(|&m| n.is_station(m) && (n.node_x[m as usize] - x).abs() < 1.0).unwrap()
}

/// T-1281: a passage costs US$10M plus US$90M a km of the straight line, paid at once between
/// constructed stations, else with the second of them; at most 1 km.
#[test]
fn a_passage_is_priced_by_its_length_and_built_with_its_stations() {
    let (mut a, st) = station();
    let e = edge_over(&a, 2400.0);
    assert_eq!(a.add_station(2, e, 2400.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let east = station_at(&a, 2400.0);
    let price = 10.0 + 90.0 * 0.9;
    assert!((a.passage_cost(st, east) - price).abs() < 1e-6, "{}", a.passage_cost(st, east));
    let quote = a.blueprint_cost();
    let cash = a.cash();
    assert_eq!(a.link_stations(st, east), 0, "{}", a.issues());
    assert_eq!(a.cash(), cash, "nothing paid between blueprint stations");
    assert!((a.blueprint_cost() - quote - price).abs() < 1e-6, "in the blueprint's quote");
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!((cash - a.cash() - quote - price).abs() < 1e-6, "built with the stations");
    assert_eq!(a.unlink_stations(st, east), 0);
    assert!((cash - a.cash() - quote - price).abs() < 1e-6, "nothing refunded");
    let cash = a.cash();
    assert_eq!(a.link_stations(st, east), 0, "{}", a.issues());
    assert!((cash - a.cash() - price).abs() < 1e-6 && (a.link_charge() - price).abs() < 1e-6, "paid at once between built stations");
    assert_eq!(a.link_stations(st, east), 0, "already joined: nothing more paid");
    assert!((cash - a.cash() - price).abs() < 1e-6);
    // too far
    let e = edge_over(&a, 200.0);
    assert_eq!(a.add_station(2, e, 200.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    let west = station_at(&a, 200.0);
    assert_eq!(a.link_stations(west, st), 1, "1.3 km is past a passage's reach");
}

/// T-3740: a new station takes the fare zone of the nearest station with one within `NEIGHBOUR_FARE_M`, in the
/// placing edit itself (one undo takes both); farther out it has none. Taking out a removed area's codes is on
/// neither history, and tidies a code an undo or redo brings back.
#[test]
fn new_stations_take_their_neighbours_fare_zone_and_a_removed_areas_codes_go() {
    let (mut a, mid) = station();
    let code = 3 << 8 | 2;
    assert_eq!(a.set_station_fares(&[mid], &[code]), 0, "{}", a.issues());
    // 900 m away: its zone
    let e = edge_over(&a, 2400.0);
    assert_eq!(a.add_station(2, e, 2400.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let east = station_at(&a, 2400.0);
    assert_eq!(a.w.net.node_fare[east as usize], code);
    assert_eq!(a.undo(), 0);
    assert!(!a.w.net.is_station(east) || a.w.net.node_fare[east as usize] == 0, "one undo takes the station and its zone");
    assert_eq!(a.station_fares(), vec![mid, code]);
    assert_eq!(a.redo(), 0);
    let east = station_at(&a, 2400.0);
    assert_eq!(a.w.net.node_fare[east as usize], code, "a redo puts it back zoned");
    // the nearest zoned one wins: 300 m from a zone 5 station, 1.2 km from zone 2
    let z5 = 3 << 8 | 5;
    assert_eq!(a.set_station_fares(&[east], &[z5]), 0);
    let e = edge_over(&a, 2700.0);
    assert_eq!(a.add_station(2, e, 2700.0, 0.0, 100, "Far East".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_fare[station_at(&a, 2700.0) as usize], z5);
    // on a piece of track farther out, over 2 km from any
    assert_eq!(a.add_route(&[0.0, 0.0, 6000.0, 0.0, 0.0, 0.0, 0.0, 9000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
    let e = edge_over(&a, 7000.0);
    assert_eq!(a.add_station(2, e, 7000.0, 0.0, 200, "Out".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_fare[station_at(&a, 7000.0) as usize], 0, "4.3 km from the nearest zoned station: none");
    // area 3 kept: nothing to tidy; area 3 gone: its three stations, on no history
    assert_eq!(a.tidy_station_fares(&[3]), 0);
    let undo_before = a.can_undo();
    assert_eq!(a.tidy_station_fares(&[1, 7]), 3);
    assert!(a.station_fares().is_empty());
    assert_eq!(a.can_undo(), undo_before);
    // an undo past the zone 5 paint brings back the code East had before it, which the tidy takes out again
    assert_eq!(a.undo(), 0); // Out
    assert_eq!(a.undo(), 0); // the route
    assert_eq!(a.undo(), 0); // Far East
    assert_eq!(a.undo(), 0); // the zone 5 paint
    assert_eq!(a.station_fares(), vec![east, code], "an undo brings back a removed area's code");
    assert_eq!(a.tidy_station_fares(&[]), 1);
    // and a redo of a paint undone before its area went
    assert_eq!(a.set_station_fares(&[east], &[z5]), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.tidy_station_fares(&[]), 0);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.station_fares(), vec![east, z5], "a redo brings one back too");
    assert_eq!(a.tidy_station_fares(&[]), 1);
    assert!(a.station_fares().is_empty());
}

/// T-3740: a station on open ground and a port take their neighbours' fare zone too, and a tidy keeps the edit before
/// it as the last one (an undo it follows still says which edges to redraw).
#[test]
fn free_stations_and_ports_take_their_neighbours_fare_zone_and_a_tidy_keeps_the_last_edit() {
    let (mut a, mid) = station();
    let code = 3 << 8 | 2;
    assert_eq!(a.set_station_fares(&[mid], &[code]), 0, "{}", a.issues());
    let free = |x: f64, y: f64| [0.0, 0.0, x, y, 0.0];
    let ends: Vec<f64> = free(1500.0, 400.0).iter().chain(free(1720.0, 400.0).iter()).copied().collect();
    assert_eq!(a.add_free_station(&ends, 2, 200, "Free".into()), 0, "{}", a.issues());
    let n = &a.w.net;
    let st = (0..n.node_count() as u32).find(|&m| n.is_station(m) && n.node_y[m as usize] > 300.0).unwrap();
    assert_eq!(n.node_fare[st as usize], code, "a station on open ground 400 m away");
    assert_eq!(a.add_port(1500.0, -300.0, 1500.0, -200.0, "Port".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_fare[a.last_port() as usize], code, "a port 300 m away");
    assert_eq!(a.add_port(1500.0, -3000.0, 1500.0, -2900.0, "Far port".into()), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_fare[a.last_port() as usize], 0, "a port 3 km away");
    // an edit that splits track, then a tidy: the split's edges are still the last edit's
    let e = edge_over(&a, 2400.0);
    assert_eq!(a.add_station(2, e, 2400.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let dirty = a.dirty_edges();
    assert!(!dirty.is_empty());
    assert_eq!(a.tidy_station_fares(&[]), 4);
    assert_eq!(a.dirty_edges(), dirty);
}

#[test]
fn upgrades_go_into_the_blueprint_and_are_paid_at_once_on_a_built_station() {
    let (mut a, st) = station();
    assert!((a.station_upgrade_cost(st, upgrades::TERMINAL) - 37.5).abs() < 1e-9, "1 km of track at ground level");
    assert_eq!(a.station_upgrade_cost(st, upgrades::PARK_RIDE), 20.0);
    assert_eq!(a.station_upgrade_cost(st, upgrades::WAITING_ROOM), 5.0);
    let quote = a.blueprint_cost();
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 0, "{}", a.issues());
    assert!((a.blueprint_cost() - quote - 37.5).abs() < 1e-6, "in the blueprint's quote");
    assert_eq!(a.station_upgrades(), vec![st, upgrades::TERMINAL as u32]);
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 1, "already there");
    assert_eq!(a.undo(), 0);
    assert!(a.station_upgrades().is_empty(), "undoable while a blueprint");
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 0);
    let cash = a.cash();
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!((cash - a.cash() - quote - 37.5).abs() < 1e-6, "constructed with the station");
    // on the built station: paid at once, final
    let cash = a.cash();
    assert_eq!(a.set_station_upgrade(&[st], upgrades::WAITING_ROOM, true), 0, "{}", a.issues());
    assert!((cash - a.cash() - 5.0).abs() < 1e-9);
    assert!(!a.can_undo());
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, false), 0);
    assert!((cash - a.cash() - 5.0).abs() < 1e-9, "removing refunds nothing");
    a.set_cash(1.0);
    assert_eq!(a.set_station_upgrade(&[st], upgrades::PARK_RIDE, true), 1, "short of cash");
    assert_eq!(a.station_upgrades(), vec![st, upgrades::WAITING_ROOM as u32]);
    // renaming or a longer platform keeps them, and so does a save
    assert_eq!(a.set_station_props(st, 200, "Middle".into()), 0);
    let bytes = a.save();
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    let n = &b.w.net;
    let st2 = (0..n.node_count() as u32).find(|&m| n.is_station(m)).unwrap();
    assert_eq!(b.station_upgrades(), vec![st2, upgrades::WAITING_ROOM as u32]);
    assert_eq!(a.set_station_upgrade(&[st], 3, true), 1, "one upgrade at a time");
}
