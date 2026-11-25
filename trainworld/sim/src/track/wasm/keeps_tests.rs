//! T-4080: kept bus stops through the clock worker's API: naming, connecting, undo, saves, and stops that stay put.

use super::*;
use crate::track::bus::{KIND_CROSSING, KIND_MIDDLE, KIND_STATION};

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    a
}

/// A constructed rail line along y = 0 with stations every `step` m up to `km` km; its stations.
fn rail(a: &mut TrackApi, km: usize, step: f64) -> Vec<u32> {
    let n = (km as f64 * 1000.0 / step).round() as usize;
    let mut st = vec![];
    for i in 0..n {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * step, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &s) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 120, format!("S{i}")), 0, "{}", a.issues());
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Rail".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    st
}

/// A planned bus line over straight stretches between `pts` (x, y, kind each).
fn line(a: &mut TrackApi, pts: &[f64]) -> u32 {
    let b = a.new_line_id();
    assert_eq!(a.set_bus_line(b, pts, 480, 600, 1200, 30.0, 120.0, 0, "B".into()), 0, "{}", a.issues());
    b
}

fn stops(a: &TrackApi, l: u32) -> Vec<Vec<f64>> {
    a.bus_stops(l).chunks_exact(8).map(|c| c.to_vec()).collect()
}

fn bits(a: &TrackApi, l: u32) -> Vec<u64> {
    a.bus_stops(l).into_iter().map(f64::to_bits).collect()
}

/// The stop of line `l` within 1 mm of (x, y).
fn stop_at(a: &TrackApi, l: u32, x: f64, y: f64) -> Option<Vec<f64>> {
    stops(a, l).into_iter().find(|c| (c[1] - x).abs() < 1e-3 && (c[2] - y).abs() < 1e-3)
}

/// The kept stop of line `l`'s stop within 1 mm of (x, y), `u32::MAX` for none or no stop.
fn keep_of(a: &TrackApi, l: u32, x: f64, y: f64) -> u32 {
    stop_at(a, l, x, y).map_or(u32::MAX, |c| a.bus_stop_keep(c[0] as u32))
}

#[test]
fn naming_a_bus_stop_moves_no_stop_and_undoes() {
    let mut a = api();
    let b = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    let before = stops(&a, b);
    let before_bits = bits(&a, b);
    let m = 5;
    assert_eq!(before[m][4], KIND_MIDDLE as f64);
    let (id, x, y) = (before[m][0] as u32, before[m][1], before[m][2]);
    assert!(a.keeps().is_empty() && a.bus_stop_keep(id) == u32::MAX);

    assert_eq!(a.name_keep(u32::MAX, x, y, "Elm St".into()), 0, "{}", a.issues());
    let k = a.last_keep();
    assert_ne!(k, u32::MAX);
    assert_eq!(a.keep_name(k), "Elm St");
    // the line's stops are where they were, ids and kinds included; the named one within a millimetre
    let after = stops(&a, b);
    assert_eq!(after.len(), before.len());
    for (p, q) in before.iter().zip(&after) {
        assert!(p[0] == q[0] && p[4] == q[4] && (p[3] - q[3]).abs() < 1e-3, "{p:?} {q:?}");
    }
    assert_eq!(a.bus_stop_keep(id), k);
    let ks = a.bus_stop_keeps(b);
    assert!(ks.iter().enumerate().all(|(j, &v)| v == if j == m { k as f64 } else { -1.0 }), "{ks:?}");
    assert_eq!(a.keep_at(x, y + 30.0), k);
    assert_eq!(a.keep_at(x, y + 200.0), u32::MAX);

    // naming the stop again renames that kept stop; undo and redo
    assert_eq!(a.name_keep(u32::MAX, x, y, "Oak St".into()), 0, "{}", a.issues());
    assert_eq!((a.last_keep(), a.keeps().len(), a.keep_name(k)), (k, 3, "Oak St".to_string()));
    let epoch = a.keep_epoch();
    assert_eq!(a.undo(), 0);
    assert_eq!(a.keep_name(k), "Elm St");
    assert_ne!(a.keep_epoch(), epoch, "a rename's undo is told");
    assert_eq!(a.redo(), 0);
    assert_eq!(a.keep_name(k), "Oak St");

    // an empty name clears it: the kept stop goes and the stops are exactly as before
    assert_eq!(a.name_keep(k, x, y, "  ".into()), 0, "{}", a.issues());
    assert!(a.keeps().is_empty() && a.last_keep() == u32::MAX);
    assert_eq!(bits(&a, b), before_bits);
    assert_eq!(a.undo(), 0);
    assert_eq!((a.keep_name(k), a.bus_stop_keep(id)), ("Oak St".to_string(), k));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert!(a.keeps().is_empty());
    assert_eq!(bits(&a, b), before_bits);
    assert_eq!(a.redo(), 0);
    assert_eq!((a.keep_name(k), a.bus_stop_keep(id)), ("Elm St".to_string(), k));
    // a name for a place with no kept stop and nothing typed is no edit
    assert_eq!(a.name_keep(u32::MAX, x + 1000.0, y, "".into()), 0);
    assert_eq!(a.keeps().len(), 3);
}

#[test]
fn a_named_stop_stays_put_when_its_line_is_extended_a_station_is_built_or_a_line_crosses() {
    let mut a = api();
    let mut fresh = api();
    let st = rail(&mut a, 4, 2000.0);
    rail(&mut fresh, 4, 2000.0);
    // 150 m north of the rail line: its stations at x = 0, 2000, 4000 are stops; name the first stop past 4000
    let route = [-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0];
    let b = line(&mut a, &route);
    assert_eq!(line(&mut fresh, &route), b);
    let s0 = stops(&a, b);
    assert_eq!(s0.iter().filter(|c| c[4] == KIND_STATION as f64).count(), st.len(), "{s0:?}");
    let m = s0.iter().position(|c| c[4] == KIND_MIDDLE as f64 && c[1] > 4000.0).unwrap();
    let (x, y) = (s0[m][1], s0[m][2]);
    assert_eq!(a.name_keep(u32::MAX, x, y, "Elm St".into()), 0, "{}", a.issues());
    let k = a.last_keep();
    assert_eq!(keep_of(&a, b, x, y), k);
    // the line without the kept stop: each edit below moves that stop, so it is held there
    let moved = |f: &TrackApi| stop_at(f, b, x, y).is_none();

    // an end extended
    let longer = [-1000.0, 150.0, 1.0, 6300.0, 150.0, 1.0];
    assert_eq!(a.set_bus_route(b, &longer), 0, "{}", a.issues());
    assert_eq!(fresh.set_bus_route(b, &longer), 0, "{}", fresh.issues());
    assert!(moved(&fresh), "{:?}", stops(&fresh, b));
    assert_eq!(keep_of(&a, b, x, y), k, "{:?}", stops(&a, b));
    // undo and redo of it
    assert_eq!(a.undo(), 0);
    assert_eq!(keep_of(&a, b, x, y), k);
    assert_eq!(a.redo(), 0);
    assert_eq!(keep_of(&a, b, x, y), k);

    // a station built within 200 m of the route, in the named stop's gap
    for t in [&mut a, &mut fresh] {
        let ends = [0.0, 0.0, 5400.0, 0.0, -1.0, 0.0, 0.0, 5800.0, 0.0, -1.0];
        assert_eq!(t.add_route(&ends, &[], 2, false), 0, "{}", t.issues());
        let n = t.route_end();
        assert_eq!(t.add_station(1, n, 0.0, 0.0, 120, "East".into()), 0, "{}", t.issues());
        t.settle_owed(); // T-4272: its services after its answer
        assert!(stops(t, b).iter().any(|c| c[5] == n as f64), "the new station is a stop");
    }
    assert!(moved(&fresh), "{:?}", stops(&fresh, b));
    assert_eq!(keep_of(&a, b, x, y), k, "{:?}", stops(&a, b));

    // a line crossing 30 m from it: the crossing folds into the held stop, and the new line stops at the kept stop too
    let c = line(&mut a, &[x + 30.0, -2000.0, 1.0, x + 30.0, 2000.0, 1.0]);
    assert_eq!(keep_of(&a, b, x, y), k);
    assert!(!stops(&a, b).iter().any(|s| s[4] == KIND_CROSSING as f64), "{:?}", stops(&a, b));
    let theirs: Vec<Vec<f64>> = stops(&a, c).into_iter().filter(|s| a.bus_stop_keep(s[0] as u32) == k).collect();
    assert_eq!(theirs.len(), 1, "{:?}", stops(&a, c));
    assert!((theirs[0][1] - (x + 30.0)).abs() < 1e-3 && (theirs[0][2] - y).abs() < 1e-3, "{theirs:?}");
    assert!(!stops(&a, c).iter().any(|s| s[4] == KIND_CROSSING as f64), "{:?}", stops(&a, c));
    // undoing the crossing line leaves the named stop as it was
    assert_eq!(a.undo(), 0);
    assert_eq!(keep_of(&a, b, x, y), k);
}

#[test]
fn a_spaced_line_stops_at_a_kept_stop_and_a_stop_by_hand_takes_one() {
    let mut a = api();
    let b = line(&mut a, &[0.0, 0.0, 1.0, 20000.0, 0.0, 1.0]);
    assert_eq!(a.set_bus_spacing(b, 5000.0), 0, "{}", a.issues());
    let spaced = stops(&a, b);
    // a kept stop 600 m short of the second stop's place: within half the spacing, so the line stops there instead
    let (x, y) = (spaced[1][1] - 600.0, 0.0);
    assert_eq!(a.name_keep(u32::MAX, x, y, "Elm St".into()), 0, "{}", a.issues());
    let k = a.last_keep();
    assert_eq!(keep_of(&a, b, x, y), k, "{:?}", stops(&a, b));
    assert!(stop_at(&a, b, spaced[1][1], spaced[1][2]).is_none());
    // by hand: the kept stop holds nothing, and a stop placed 20 m from it is its
    assert_eq!(a.set_bus_hand(b), 0, "{}", a.issues());
    assert!(stop_at(&a, b, x, y).is_none());
    assert_eq!(a.set_bus_stop(b, x + 20.0, 0.0, true), 0, "{}", a.issues());
    assert_eq!(keep_of(&a, b, x + 20.0, 0.0), k, "{:?}", stops(&a, b));
}

#[test]
fn kept_stops_round_trip_through_a_save_and_older_saves_have_none() {
    let mut a = api();
    let st = rail(&mut a, 4, 2000.0);
    let b = line(&mut a, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0]);
    let plain = a.save();
    assert!(!plain.windows(4).any(|w| w == b"LBK1"), "no kept stops: no block");
    let s = stops(&a, b);
    let mid: Vec<&Vec<f64>> = s.iter().filter(|c| c[4] == KIND_MIDDLE as f64 && c[1] > 2000.0 && c[1] < 4000.0).collect();
    let (p, q) = (mid[0], mid[1]);
    assert_eq!(a.name_keep(u32::MAX, p[1], p[2], "Elm St".into()), 0, "{}", a.issues());
    let k1 = a.last_keep();
    assert_eq!(a.connect_keep(u32::MAX, q[1], q[2], 0, st[1], 0.0, 0.0), 0, "{}", a.issues());
    let k2 = a.last_keep();
    assert_eq!(a.connect_keep(k1, 0.0, 0.0, 1, k2, 0.0, 0.0), 0, "{}", a.issues());
    // a kept stop no bus stop takes: forgotten at the next load
    assert_eq!(a.name_keep(u32::MAX, 2500.0, 3000.0, "Nowhere".into()), 0, "{}", a.issues());
    assert_eq!(a.keeps().len(), 9);
    let bytes = a.save();
    assert!(bytes.windows(4).any(|w| w == b"LBK1"));

    let mut c = api();
    assert!(c.load(&bytes));
    assert_eq!(bits(&c, b), bits(&a, b));
    let (c1, c2) = (keep_of(&c, b, p[1], p[2]), keep_of(&c, b, q[1], q[2]));
    assert!(c1 != u32::MAX && c2 != u32::MAX && c1 != c2);
    assert_eq!(c.keeps().len(), 6, "the one no stop took is gone: {:?}", c.keeps());
    assert_eq!((c.keep_name(c1), c.keep_name(c2)), ("Elm St".to_string(), String::new()));
    assert_eq!(c.keep_links(c1), vec![1, c2]);
    assert_eq!(c.keep_links(c2), vec![0, st[1], 1, c1]);
    assert_eq!(c.station_keeps(st[1]), vec![c2]);
    // and again: the same bytes
    assert_eq!(c.save(), {
        let mut d = api();
        assert!(d.load(&c.save()));
        d.save()
    });
    // a save from before kept stops loads with none
    assert!(c.load(&plain));
    assert!(c.keeps().is_empty() && c.bus_stop_keeps(b).iter().all(|&v| v == -1.0));
    // undone, the network saves as it did before
    for _ in 0..4 {
        assert_eq!(a.undo(), 0);
    }
    assert!(a.keeps().is_empty());
    assert_eq!(a.save(), plain);
}

#[test]
fn connections_are_made_from_a_bus_stop_removed_from_either_side_and_undone() {
    let mut a = api();
    let st = rail(&mut a, 4, 2000.0);
    let b = line(&mut a, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0]);
    let s = stops(&a, b);
    let mid: Vec<&Vec<f64>> = s.iter().filter(|c| c[4] == KIND_MIDDLE as f64 && c[1] > 2000.0 && c[1] < 4000.0).collect();
    let (p, q) = (mid[0], mid[1]);
    let bus_bits = bits(&a, b);

    // to a station: free, and no stop moves
    let cash = a.cash();
    assert_eq!(a.connect_keep(u32::MAX, p[1], p[2], 0, st[1], 0.0, 0.0), 0, "{}", a.issues());
    let k = a.last_keep();
    assert_eq!((a.keep_links(k), a.station_keeps(st[1]), a.keep_name(k)), (vec![0, st[1]], vec![k], String::new()));
    assert_eq!(a.cash(), cash);
    assert_eq!(bits(&a, b).len(), bus_bits.len());
    // refused: again, too far, no such station, itself
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 0, st[1], 0.0, 0.0), 1);
    assert_eq!(a.connect_keep(u32::MAX, p[1], p[2], 0, st[1], 0.0, 0.0), 1, "the bus stop's kept stop is used");
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 0, st[0], 0.0, 0.0), 1, "over 1 km");
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 0, 9999, 0.0, 0.0), 1);
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 1, k, 0.0, 0.0), 1);
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 1, u32::MAX, p[1] + 20.0, p[2]), 1, "a bus stop at the same place");
    assert_eq!(a.keeps().len(), 3);

    // to another bus stop with no kept stop: one is made there
    assert_eq!(a.connect_keep(k, 0.0, 0.0, 1, u32::MAX, q[1], q[2]), 0, "{}", a.issues());
    let k2 = keep_of(&a, b, q[1], q[2]);
    assert!(k2 != u32::MAX && k2 != k);
    assert_eq!((a.keep_links(k), a.keep_links(k2)), (vec![0, st[1], 1, k2], vec![1, k]));
    assert_eq!(a.connect_keep(k2, 0.0, 0.0, 1, k, 0.0, 0.0), 1, "already connected, from the other side");

    // removed from the other side: k2, with no name and no connection left, goes; k keeps its station
    assert_eq!(a.disconnect_keep(k2, 1, k), 0, "{}", a.issues());
    assert_eq!((a.keeps().len(), a.keep_links(k)), (3, vec![0, st[1]]));
    assert_eq!(keep_of(&a, b, q[1], q[2]), u32::MAX);
    assert_eq!(a.undo(), 0);
    assert_eq!((a.keep_links(k2), a.keep_links(k)), (vec![1, k], vec![0, st[1], 1, k2]));
    assert_eq!(a.disconnect_keep(k2, 1, k), 0);
    assert_eq!(a.disconnect_keep(k2, 1, k), 1, "not connected now");
    // the last connection of a stop with no name: it goes, and the stops are as before
    assert_eq!(a.disconnect_keep(k, 0, st[1]), 0, "{}", a.issues());
    assert!(a.keeps().is_empty() && a.station_keeps(st[1]).is_empty());
    assert_eq!(bits(&a, b), bus_bits);
    // a named one stays
    assert_eq!(a.undo(), 0);
    assert_eq!(a.name_keep(k, 0.0, 0.0, "Elm St".into()), 0);
    assert_eq!(a.disconnect_keep(k, 0, st[1]), 0);
    assert_eq!((a.keep_name(k), a.keep_links(k)), ("Elm St".to_string(), vec![]));
}

#[test]
fn a_kept_stop_far_from_every_route_changes_no_stop() {
    let mut a = api();
    rail(&mut a, 4, 2000.0);
    let b = line(&mut a, &[-1000.0, 150.0, 1.0, 5000.0, 150.0, 1.0]);
    let c = line(&mut a, &[3000.0, -2000.0, 1.0, 3000.0, 3000.0, 1.0]);
    let d = line(&mut a, &[0.0, 1000.0, 1.0, 20000.0, 1000.0, 1.0]);
    assert_eq!(a.set_bus_spacing(d, 3000.0), 0, "{}", a.issues());
    let before: Vec<Vec<u64>> = [b, c, d].iter().map(|&l| bits(&a, l)).collect();
    assert_eq!(a.name_keep(u32::MAX, 10000.0, -3000.0, "Nowhere".into()), 0, "{}", a.issues());
    assert_eq!([b, c, d].iter().map(|&l| bits(&a, l)).collect::<Vec<_>>(), before);
    assert!([b, c, d].iter().all(|&l| a.bus_stop_keeps(l).iter().all(|&v| v == -1.0)));
}

#[test]
fn a_connection_to_a_removed_station_ends_and_its_unnamed_stop_goes_with_it() {
    let mut a = api();
    let b = line(&mut a, &[0.0, 150.0, 1.0, 6000.0, 150.0, 1.0]);
    // a station on no line, 400 m from the route
    let ends = [0.0, 0.0, 2800.0, 550.0, -1.0, 0.0, 0.0, 3200.0, 550.0, -1.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let n = a.route_end();
    assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, "North".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    let m = s.iter().min_by(|p, q| (p[1] - 3000.0).abs().total_cmp(&(q[1] - 3000.0).abs())).unwrap().clone();
    assert_eq!(a.connect_keep(u32::MAX, m[1], m[2], 0, n, 0.0, 0.0), 0, "{}", a.issues());
    let k = a.last_keep();
    assert_eq!(a.remove_station(n), 0, "{}", a.issues());
    // (its services worked out after its answer, as the clock worker does before its state: T-4272)
    a.settle_owed();
    assert!(a.keep_links(k).is_empty() && a.station_keeps(n).is_empty());
    // the bare stop goes in the same edit; an undo brings it and the connection back
    assert_eq!(keep_of(&a, b, m[1], m[2]), u32::MAX);
    assert_eq!(a.undo(), 0);
    assert_eq!(keep_of(&a, b, m[1], m[2]), k);
    assert_eq!(a.keep_links(k), vec![0, n]);
    assert_eq!(a.redo(), 0);
    let mut c = api();
    assert!(c.load(&a.save()));
    assert!(c.keeps().is_empty());
}

#[test]
fn an_out_and_back_bus_route_holds_a_kept_stop_on_both_passes() {
    let mut a = api();
    // loop-back route: stem (0,0)-(2000,0), loop to (3000,0) and back along the stem to (0,0)
    let route = [
        0.0, 0.0, 1.0,
        2000.0, 0.0, 1.0,
        2500.0, 500.0, 1.0,
        3000.0, 0.0, 1.0,
        2500.0, -500.0, 1.0,
        2000.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
    ];
    let b = line(&mut a, &route);
    assert_eq!(a.name_keep(u32::MAX, 1100.0, 0.0, "Stem Stop".into()), 0, "{}", a.issues());
    let k = a.last_keep();
    let held: Vec<Vec<f64>> = stops(&a, b).into_iter().filter(|s| a.bus_stop_keep(s[0] as u32) == k).collect();
    assert_eq!(held.len(), 2, "kept stop held on both outbound and return passes: {held:?}");
    for s in &held {
        assert!((s[1] - 1100.0).abs() < 1e-3 && s[2].abs() < 1e-3, "{s:?}");
    }
    assert!((held[1][3] - held[0][3]).abs() > 1000.0, "one on each pass: {held:?}");
}

#[test]
fn a_kept_stop_left_bare_at_load_is_dropped_and_its_line_replans() {
    let mut a = api();
    let b1 = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    let b2 = line(&mut a, &[0.0, 500.0, 1.0, 6000.0, 500.0, 1.0]);
    let s1 = stops(&a, b1);
    let s2 = stops(&a, b2);
    let p1 = s1.iter().find(|s| s[4] == KIND_MIDDLE as f64 && s[1] > 1500.0).unwrap().clone();
    let q = s2.iter().find(|s| s[4] == KIND_MIDDLE as f64 && s[1] > 2300.0).unwrap().clone();
    // unnamed p1 (on b1) connected only to unnamed q (on b2)
    assert_eq!(a.connect_keep(u32::MAX, p1[1], p1[2], 1, u32::MAX, q[1], q[2]), 0, "{}", a.issues());
    // extend b1 so p1 holds its stop, then delete b2 (q is taken by no stop at the next load, so p1 is left bare)
    let longer = [0.0, 0.0, 1.0, 7300.0, 0.0, 1.0];
    assert_eq!(a.set_bus_route(b1, &longer), 0, "{}", a.issues());
    assert_eq!(a.remove_line(b2), 0, "{}", a.issues());

    let mut fresh = api();
    let fb = line(&mut fresh, &longer);
    assert_ne!(bits(&a, b1), bits(&fresh, fb), "the held stop moved b1 from its unheld layout");

    let mut c = api();
    assert!(c.load(&a.save()));
    assert!(c.keeps().is_empty(), "bare kept stop dropped at load: {:?}", c.keeps());
    assert!(c.bus_stop_keeps(b1).iter().all(|&v| v == -1.0));
    assert_eq!(bits(&c, b1), bits(&fresh, fb), "b1 re-planned without the dropped bare kept stop");
}

#[test]
fn of_two_kept_stops_within_50m_a_route_holds_the_nearer_one() {
    let mut a = api();
    let la = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    let lb = line(&mut a, &[0.0, 45.0, 1.0, 6000.0, 45.0, 1.0]);
    // K1 on la at (2000, 0), K2 on lb at (2025, 45): hypot(25, 45) = 51.48 m > 50 m (distinct keeps),
    // but K1 projects onto lb at (2000, 45) (45 m off lb, 25 m earlier than K2 which is 0 m off lb)
    assert_eq!(a.name_keep(u32::MAX, 2000.0, 0.0, "Alpha".into()), 0, "{}", a.issues());
    let k1 = a.last_keep();
    assert_eq!(a.name_keep(u32::MAX, 2025.0, 45.0, "Beta".into()), 0, "{}", a.issues());
    let k2 = a.last_keep();
    assert_ne!(k1, k2);
    assert_eq!(keep_of(&a, la, 2000.0, 0.0), k1, "la holds K1 (0 m off la)");
    assert_eq!(keep_of(&a, lb, 2025.0, 45.0), k2, "lb holds K2 (0 m off lb), not K1 (45 m off lb)");
}

#[test]
fn keep_epoch_moves_when_a_connected_station_is_removed_or_restored() {
    let mut a = api();
    let b = line(&mut a, &[0.0, 150.0, 1.0, 6000.0, 150.0, 1.0]);
    let ends = [0.0, 0.0, 2800.0, 550.0, -1.0, 0.0, 0.0, 3200.0, 550.0, -1.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let n = a.route_end();
    assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, "North".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    let m = s.iter().min_by(|p, q| (p[1] - 3000.0).abs().total_cmp(&(q[1] - 3000.0).abs())).unwrap().clone();
    assert_eq!(a.connect_keep(u32::MAX, m[1], m[2], 0, n, 0.0, 0.0), 0, "{}", a.issues());

    let e0 = a.keep_epoch();
    assert_eq!(a.remove_station(n), 0, "{}", a.issues());
    let e1 = a.keep_epoch();
    assert_ne!(e1, e0, "removing a connected station bumps keep_epoch");
    assert_eq!(a.undo(), 0);
    let e2 = a.keep_epoch();
    assert_ne!(e2, e1, "undoing removal of a connected station bumps keep_epoch");
}

#[test]
fn an_lbk1_entry_outside_the_world_is_skipped_on_load() {
    let mut a = api();
    a.w = TrackWorld::new(Network::new_in(FrameKind::Merc, 0.0, 0.0, true));
    let b = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    assert_eq!(a.name_keep(u32::MAX, 2000.0, 0.0, "Elm St".into()), 0, "{}", a.issues());
    let k0 = a.last_keep();
    // inject an out-of-world entry between two valid entries (as a frame conversion can produce)
    let k1 = a.w.net.alloc_keep();
    a.w.net.keeps[k1 as usize] = Some(crate::track::bus::keep::KeptStop {
        x: 1e9,
        y: 1e9,
        name: "Outside".into(),
        links: vec![],
    });
    assert_eq!(a.name_keep(u32::MAX, 2600.0, 0.0, "Oak St".into()), 0, "{}", a.issues());
    let k2 = a.last_keep();
    assert_eq!(a.connect_keep(k2, 0.0, 0.0, 1, k0, 0.0, 0.0), 0, "{}", a.issues());
    let bytes = a.save();

    let mut c = api();
    assert!(c.load(&bytes), "out-of-world LBK1 entry skipped instead of aborting load");
    let c0 = keep_of(&c, b, 2000.0, 0.0);
    let c2 = keep_of(&c, b, 2600.0, 0.0);
    assert!(c0 != u32::MAX && c2 != u32::MAX && c0 != c2);
    assert_eq!((c.keep_name(c0), c.keep_name(c2)), ("Elm St".to_string(), "Oak St".to_string()));
    assert_eq!(c.keep_links(c2), vec![1, c0], "link indices past the skipped entry still resolve");
    assert_eq!(c.keeps().len(), 6, "only the two valid entries loaded");
}

#[test]
fn a_link_to_a_skipped_lbk1_entry_is_dropped_so_its_slot_can_be_reused() {
    let mut a = api();
    a.w = TrackWorld::new(Network::new_in(FrameKind::Merc, 0.0, 0.0, true));
    let b = line(&mut a, &[0.0, 0.0, 1.0, 6000.0, 0.0, 1.0]);
    assert_eq!(a.name_keep(u32::MAX, 2000.0, 0.0, "Elm St".into()), 0, "{}", a.issues());
    let k1 = a.w.net.alloc_keep();
    a.w.net.keeps[k1 as usize] = Some(crate::track::bus::keep::KeptStop { x: 1e9, y: 1e9, name: "Outside".into(), links: vec![] });
    assert_eq!(a.name_keep(u32::MAX, 2600.0, 0.0, "Oak St".into()), 0, "{}", a.issues());
    let k2 = a.last_keep();
    a.w.net.keeps[k2 as usize].as_mut().unwrap().links.push(crate::track::bus::keep::KeepLink::Keep(k1));
    let bytes = a.save();

    let mut c = api();
    assert!(c.load(&bytes));
    let c2 = keep_of(&c, b, 2600.0, 0.0);
    assert_eq!(c.keep_name(c2), "Oak St");
    assert!(c.keep_links(c2).is_empty());
    // a new kept stop takes the free slot; it must not pick up the old link
    assert_eq!(c.name_keep(u32::MAX, 4000.0, 0.0, "Pine St".into()), 0, "{}", c.issues());
    assert!(c.keep_links(c2).is_empty(), "the skipped entry's link did not come back");
    assert!(c.keep_links(c.last_keep()).is_empty());
}

#[test]
fn a_kept_stop_its_station_left_bare_goes_in_the_same_edit_and_undo_brings_it_back() {
    let mut a = api();
    let b = line(&mut a, &[0.0, 150.0, 1.0, 6000.0, 150.0, 1.0]);
    let ends = [0.0, 0.0, 2800.0, 550.0, -1.0, 0.0, 0.0, 3200.0, 550.0, -1.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let n = a.route_end();
    assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, "North".into()), 0, "{}", a.issues());
    let s = stops(&a, b);
    let m = s.iter().min_by(|p, q| (p[1] - 3000.0).abs().total_cmp(&(q[1] - 3000.0).abs())).unwrap().clone();
    assert_eq!(a.connect_keep(u32::MAX, m[1], m[2], 0, n, 0.0, 0.0), 0, "{}", a.issues());
    let k = a.last_keep();
    let before = stops(&a, b);

    assert_eq!(a.remove_station(n), 0, "{}", a.issues());
    assert!(a.keeps().is_empty(), "the bare kept stop went with the station");
    // a reload agrees with the session
    let mut c = api();
    assert!(c.load(&a.save()));
    assert_eq!(stops(&c, b), stops(&a, b), "session and reload place the line alike");

    assert_eq!(a.undo(), 0);
    assert_eq!(a.keeps().len(), 3, "undo brings the kept stop back");
    assert_eq!(a.keep_links(k), vec![0, n]);
    assert_eq!(stops(&a, b), before);
}
