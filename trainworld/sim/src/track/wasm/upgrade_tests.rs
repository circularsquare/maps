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

#[test]
fn upgrades_go_into_the_blueprint_and_are_paid_at_once_on_a_built_station() {
    let (mut a, st) = station();
    assert!((a.station_upgrade_cost(st, upgrades::TERMINAL) - 45.0).abs() < 1e-9, "1 km of track at ground level");
    assert_eq!(a.station_upgrade_cost(st, upgrades::PARK_RIDE), 20.0);
    assert_eq!(a.station_upgrade_cost(st, upgrades::WAITING_ROOM), 5.0);
    let quote = a.blueprint_cost();
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 0, "{}", a.issues());
    assert!((a.blueprint_cost() - quote - 45.0).abs() < 1e-6, "in the blueprint's quote");
    assert_eq!(a.station_upgrades(), vec![st, upgrades::TERMINAL as u32]);
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 1, "already there");
    assert_eq!(a.undo(), 0);
    assert!(a.station_upgrades().is_empty(), "undoable while a blueprint");
    assert_eq!(a.set_station_upgrade(&[st], upgrades::TERMINAL, true), 0);
    let cash = a.cash();
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!((cash - a.cash() - quote - 45.0).abs() < 1e-6, "constructed with the station");
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
