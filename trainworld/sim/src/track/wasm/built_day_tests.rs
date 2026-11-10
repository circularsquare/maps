//! T-2871: each constructed element's build day, across edits, undo, save and load.

use super::*;
use crate::track::bus::{BusRoute, K_CORNER};
use crate::track::net::DAY_UNKNOWN;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

fn joint(a: &mut TrackApi, x: f64) -> u32 {
    let id = a.w.net.alloc_node();
    a.w.apply(Op::Node { id, data: Some(NodeData { x, y: 0.0, level: 0, flying: false }) }).unwrap();
    id
}

fn piece(a: u32, b: u32, built: bool) -> EdgeData {
    EdgeData { a, b, tracks: 2, pis: vec![], built, off: false, thru: vec![], crossover: false, kind: TrackKind::Regular, sep: false, day: 0 }
}

fn built_edge(a: &mut TrackApi, from: u32, to: u32) -> u32 {
    let id = a.w.net.alloc_edge();
    a.w.apply(Op::Edge { id, data: Some(piece(from, to, true)) }).unwrap();
    id
}

fn station(name: &str, built: bool) -> StationData {
    StationData { platform: 200, name: name.into(), named: false, built, off: false, upgrades: 0, berth: None, day: 0 }
}

fn plain_line(stops: Vec<u32>, path: Vec<u32>) -> LineData {
    LineData { name: "L".into(), symbol: String::new(), colour: 0, stops, path, tph: [0.0; 3], trains: Some([0; 3]), auto: None, dwell_s: 30.0, turnaround_s: 120.0, cars: 0, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0 }
}

/// Two built stations, an edge between them and a line over it, all built on day 5.
fn dated() -> (TrackApi, u32, u32) {
    let mut a = api();
    a.set_game_day(5);
    let (p, q) = (joint(&mut a, 0.0), joint(&mut a, 2000.0));
    let e = built_edge(&mut a, p, q);
    a.w.apply(Op::Station { node: p, data: Some(station("P", true)) }).unwrap();
    a.w.apply(Op::Station { node: q, data: Some(station("Q", true)) }).unwrap();
    let l = a.w.net.alloc_line();
    a.w.apply(Op::Line { id: l, data: Some(plain_line(vec![p, q], vec![e << 1])) }).unwrap();
    (a, e, l)
}

#[test]
fn blueprint_is_dated_the_day_it_is_built_not_the_day_it_was_drawn() {
    let mut a = api();
    a.set_game_day(3);
    let (p, q) = (joint(&mut a, 0.0), joint(&mut a, 1000.0));
    let id = a.w.net.alloc_edge();
    a.w.apply(Op::Edge { id, data: Some(piece(p, q, false)) }).unwrap();
    assert_eq!(a.w.net.edge_day[id as usize], 0);
    assert_eq!(a.edge_built_days()[id as usize], -2, "not constructed yet");
    a.set_game_day(7);
    a.w.apply(Op::Edge { id, data: Some(EdgeData { built: true, ..a.w.net.edge_data(id).unwrap() }) }).unwrap();
    assert_eq!(a.edge_built_days()[id as usize], 7);
}

#[test]
fn a_redrawn_or_upgraded_edge_keeps_its_day() {
    let (mut a, e, _) = dated();
    a.set_game_day(20);
    let mut d = a.w.net.edge_data(e).unwrap();
    d.tracks = 1;
    d.day = 0; // a fresh description of track that already exists
    a.w.apply(Op::Edge { id: e, data: Some(d) }).unwrap();
    assert_eq!(a.edge_built_days()[e as usize], 5);
}

#[test]
fn splitting_keeps_the_earlier_day_and_undo_and_redo_restore_it() {
    let (mut a, e, _) = dated();
    a.set_game_day(9);
    let (node, new_edge) = (a.w.net.alloc_node(), a.w.net.alloc_edge());
    a.w.apply(Op::Split { edge: e, s: 800.0, node, new_edge }).unwrap();
    let days = a.edge_built_days();
    assert_eq!((days[e as usize], days[new_edge as usize]), (5, 5), "both halves are old track");
    a.w.undo().unwrap().unwrap();
    assert_eq!(a.edge_built_days()[e as usize], 5);
    assert!(!a.w.net.edge_ok(new_edge), "the far half is gone again");
    // redone on a later day, the halves are still the old track
    a.set_game_day(30);
    a.w.redo().unwrap().unwrap();
    let days = a.edge_built_days();
    assert_eq!((days[e as usize], days[new_edge as usize]), (5, 5));
}

#[test]
fn undo_of_a_construct_clears_the_day_and_redo_restores_it() {
    let mut a = api();
    a.set_game_day(3);
    let (p, q) = (joint(&mut a, 0.0), joint(&mut a, 1000.0));
    let id = a.w.net.alloc_edge();
    a.w.apply(Op::Edge { id, data: Some(piece(p, q, false)) }).unwrap();
    a.set_game_day(7);
    a.w.apply(Op::Edge { id, data: Some(piece(p, q, true)) }).unwrap();
    assert_eq!(a.edge_built_days()[id as usize], 7);
    a.w.undo().unwrap().unwrap();
    assert_eq!(a.edge_built_days()[id as usize], -2, "a blueprint again");
    a.set_game_day(12);
    a.w.redo().unwrap().unwrap();
    assert_eq!(a.edge_built_days()[id as usize], 7, "redo restores the day, not today");
}

#[test]
fn joining_pieces_keeps_the_earliest_day() {
    let mut a = api();
    a.set_game_day(9);
    let (p, m, q) = (joint(&mut a, 0.0), joint(&mut a, 1000.0), joint(&mut a, 2000.0));
    let late = built_edge(&mut a, p, m);
    a.set_game_day(4);
    let early = built_edge(&mut a, m, q);
    let (ops, _) = a.unjoin_near(late, 500.0, 2000.0).expect("a plain joint to join across");
    a.set_game_day(50);
    a.w.apply(Op::Batch(ops)).unwrap();
    assert_eq!(a.edge_built_days()[late as usize], 4, "the joined piece is as old as its oldest part");
    assert!(!a.w.net.edge_ok(early));
}

#[test]
fn a_copied_line_is_a_new_line() {
    let (mut a, _, l) = dated();
    a.set_game_day(11);
    let id = a.new_line_id();
    assert_eq!(a.copy_line(l, id, "copy".into()), 0, "{}", a.issues());
    let days = a.line_built_days();
    assert_eq!((days[l as usize], days[id as usize]), (5, 11));
}

#[test]
fn a_planned_bus_route_has_no_day_until_constructed() {
    let mut a = api();
    a.set_game_day(2);
    let id = a.w.net.alloc_line();
    let route = BusRoute::new(vec![[0.0, 0.0], [100.0, 0.0]], vec![K_CORNER, K_CORNER], false);
    let mut d = LineData { bus: Some(route), ..plain_line(vec![], vec![]) };
    a.w.net.set_line(id, Some(d.clone())).unwrap();
    assert_eq!(a.line_built_days()[id as usize], -2);
    a.set_game_day(6);
    d.bus.as_mut().unwrap().built = true;
    a.w.net.set_line(id, Some(d)).unwrap();
    assert_eq!(a.line_built_days()[id as usize], 6);
}

#[test]
fn a_game_with_no_recorded_day_saves_as_an_older_game_did() {
    let mut a = api();
    let (p, q) = (joint(&mut a, 0.0), joint(&mut a, 1000.0));
    built_edge(&mut a, p, q);
    let bytes = save::encode(&a.w.net);
    assert!(!bytes.windows(4).any(|w| w == b"LBD1"), "no day known: no block");
}

#[test]
fn an_older_save_reads_as_built_before_recording() {
    let (mut a, e, l) = dated();
    a.w.net.edge_day[e as usize] = DAY_UNKNOWN;
    let id = a.w.net.alloc_edge();
    let far = joint(&mut a, 3000.0);
    a.w.apply(Op::Edge { id, data: Some(piece(a.w.net.edge_b[e as usize], far, false)) }).unwrap();
    // stamped nothing: every built element unknown, as a save from before this build
    a.w.net.node_day.iter_mut().for_each(|d| *d = if *d > 0 { DAY_UNKNOWN } else { 0 });
    a.w.net.line_day[l as usize] = DAY_UNKNOWN;
    let bytes = save::encode(&a.w.net);
    assert!(!bytes.windows(4).any(|w| w == b"LBD1"));
    let (net, unread) = save::decode_read(&bytes).unwrap();
    assert_eq!(unread, 0);
    assert_eq!(net.edge_day[e as usize], DAY_UNKNOWN);
    assert_eq!(net.edge_day[id as usize], 0, "a blueprint has no day");
    assert_eq!(net.line_day[l as usize], DAY_UNKNOWN);
    assert!(net.node_day.iter().any(|&d| d == DAY_UNKNOWN));
}

#[test]
fn days_survive_a_round_trip() {
    let (a, e, l) = dated();
    let bytes = save::encode(&a.w.net);
    assert!(bytes.windows(4).any(|w| w == b"LBD1"));
    let (net, unread) = save::decode_read(&bytes).unwrap();
    assert_eq!(unread, 0);
    assert_eq!(net.edge_day[e as usize], a.w.net.edge_day[e as usize]);
    assert_eq!(net.line_day[l as usize], a.w.net.line_day[l as usize]);
    assert_eq!(net.node_day, a.w.net.node_day);
    assert_eq!(save::encode(&net), bytes, "and writes back the same bytes");
}

#[test]
fn a_damaged_day_block_loses_only_the_days() {
    let (a, e, _) = dated();
    let bytes = save::encode(&a.w.net);
    let at = bytes.windows(4).position(|w| w == b"LBD1").unwrap();
    for cut in at + 4..bytes.len() {
        let (net, unread) = save::decode_read(&bytes[..cut]).unwrap_or_else(|m| panic!("cut {cut}: {m}"));
        assert_eq!(unread, 0, "cut {cut}");
        assert_eq!(net.edge_day[e as usize], DAY_UNKNOWN, "cut {cut}");
    }
    // any single byte changed in the block loads, whatever the days then say
    for i in at + 4..bytes.len() {
        for x in [1u8, 0x7f, 0x80, 0xff] {
            let mut b = bytes.clone();
            b[i] ^= x;
            let _ = save::decode_read(&b).unwrap_or_else(|m| panic!("byte {i}: {m}"));
        }
    }
    // a count that does not match the network, and an absurd one
    let mut wrong = bytes[..at].to_vec();
    wrong.extend_from_slice(b"LBD1\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01");
    let (net, _) = save::decode_read(&wrong).unwrap();
    assert_eq!(net.edge_day[e as usize], DAY_UNKNOWN);
}

#[test]
fn the_day_block_is_read_only_after_every_known_block() {
    let (a, ..) = dated();
    let bytes = save::encode(&a.w.net);
    // a later, unknown block after LBD1 is reported, not read past
    let mut newer = bytes.clone();
    newer.extend_from_slice(b"ZZZ1\x00");
    assert_eq!(save::decode_read(&newer).unwrap().1, 5);
}
