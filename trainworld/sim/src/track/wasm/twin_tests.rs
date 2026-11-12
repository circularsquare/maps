//! T-3333: a line run on both pairs of quad track through a hidden twin.

use super::*;
use crate::track::twin::Twin;

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn track_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    live(a).into_iter().filter(|&e| !a.w.net.edge_crossover[e as usize]).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap()
}
fn station(a: &mut TrackApi, x: f64, y: f64, name: &str) -> u32 {
    assert_eq!(a.add_station(2, track_at(a, x, y), x, y, 200, name.into()), 0, "{}", a.issues());
    (0..a.w.net.node_count() as u32)
        .filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0)
        .min_by(|&p, &q| {
            let d = |n: u32| (a.w.net.node_x[n as usize] - x).hypot(a.w.net.node_y[n as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}
fn trains(a: &TrackApi, l: u32) -> [u32; 3] {
    a.w.net.line_trains[l as usize].unwrap()
}

/// A 4 km quad stretch with a line on its drawn pair between West, Mid and East.
fn quad_line() -> (TrackApi, u32, Vec<u32>) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    let stops: Vec<u32> = [(500.0, "West"), (2000.0, "Mid"), (3500.0, "East")].iter().map(|&(x, n)| station(&mut a, x, 0.0, n)).collect();
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &stops, 6, 4, 2, 30.0, 180.0, 0, 0xff0000, "Q".into()), 0, "{}", a.issues());
    (a, id, stops)
}

#[test]
fn t3333_a_line_runs_on_both_pairs_once_every_stop_has_a_twin() {
    let (mut a, l, stops) = quad_line();
    assert!(!a.line_pairs_ok(l), "no stations on the other pair yet");
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    assert_eq!(a.line_pairs(l), 2, "on, one pair for now");
    assert_eq!(a.line_pairs_why(l), vec![1.0, 0.0], "West has nothing beside it");
    assert_eq!(trains(&a, l), [6, 4, 2], "the whole line's trains");
    let west = station(&mut a, 500.0, QUAD_SPACING, "x");
    let mid = station(&mut a, 2000.0, QUAD_SPACING, "x");
    assert_eq!(a.line_pairs_why(l), vec![1.0, 2.0], "East next");
    assert_eq!(a.line_twin(l), -1);
    let east = station(&mut a, 3500.0, QUAD_SPACING, "x");
    // the station's own edit split the line and made its twin
    assert_eq!(a.line_pairs(l), 1, "{:?}", a.line_pairs_why(l));
    let t = a.line_twin(l) as u32;
    assert_eq!(a.line_twin_of(t), l as i32);
    assert_eq!(a.w.net.line_stops(t), &[west, mid, east]);
    assert_eq!(a.w.net.line_stops(l), &stops[..], "the line keeps its own stops");
    assert_eq!(trains(&a, l), [3, 2, 1], "one pair's trains");
    assert_eq!(trains(&a, t), [3, 2, 1]);
    let (lp, tp) = (a.w.net.line_path(l).to_vec(), a.w.net.line_path(t).to_vec());
    assert!(lp.iter().all(|p| !tp.contains(p)), "each pair its own track");
    assert!(a.w.svc.lines[t as usize].ok && a.w.svc.lines[l as usize].ok);
    assert_eq!(a.w.net.line_colour[t as usize], 0xff0000);
    // departures half a headway apart
    let (pp, tq, tph) = (a.w.net.line_phase_of(l), a.w.net.line_phase_of(t), a.w.net.line_tph[l as usize]);
    for lev in 0..3 {
        if tph[lev] > 0.0 {
            let h = 3600.0 / tph[lev] as f64;
            let d = (tq[lev] as f64 - pp[lev] as f64).rem_euclid(h);
            assert!((d - h / 2.0).abs() < 1.0, "level {lev}: {d} of {h}");
        }
    }
    // the whole line's counts, half for each pair
    assert_eq!(a.set_train_counts(l, 9, 4, 2), 0, "{}", a.issues());
    assert_eq!((trains(&a, l), trains(&a, t)), ([5, 2, 1], [5, 2, 1]));
    assert_eq!(a.undo(), 0);
    assert_eq!((trains(&a, l), trains(&a, t)), ([3, 2, 1], [3, 2, 1]));
    // a stop's twin taken away: one pair again, with the whole line's trains, and a reason
    assert_eq!(a.remove_station(mid), 0, "{}", a.issues());
    assert_eq!(a.line_pairs(l), 2);
    assert_eq!(a.line_pairs_why(l), vec![1.0, 1.0], "Mid has nothing beside it");
    assert_eq!(a.line_twin(l), -1, "the twin goes");
    assert!(!(0..a.w.net.line_count() as u32).any(|x| a.w.net.line_ok(x) && a.w.net.is_twin(x)));
    assert_eq!(trains(&a, l), [6, 4, 2]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_pairs(l), 1);
    assert_eq!(a.line_twin(l), t as i32, "undo brings the same twin back");
    assert_eq!(trains(&a, l), [3, 2, 1]);
    // off: the twin goes and the line runs the whole line's trains
    assert_eq!(a.set_line_pairs(l, false), 0, "{}", a.issues());
    assert_eq!((a.line_pairs(l), a.line_twin(l)), (0, -1));
    assert_eq!(trains(&a, l), [6, 4, 2]);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_twin(l), t as i32);
    // removing the line takes its twin with it
    assert_eq!(a.remove_line(l), 0, "{}", a.issues());
    assert!(!a.w.net.line_ok(t));
    assert_eq!(a.undo(), 0);
    assert!(a.w.net.line_ok(t) && a.line_twin(l) == t as i32);
}

#[test]
fn t3333_headways_and_saves() {
    let (mut a, l, _) = quad_line();
    for x in [500.0, 2000.0, 3500.0] {
        station(&mut a, x, QUAD_SPACING, "x");
    }
    assert_eq!(a.set_auto_headway(l, 300, 600, 0), 0, "{}", a.issues());
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    let t = a.line_twin(l) as u32;
    assert_eq!(a.w.net.line_auto[l as usize], Some([600, 1200, 0]), "each pair every 10 minutes");
    assert_eq!(a.w.net.line_auto[t as usize], Some([600, 1200, 0]));
    assert_eq!(a.set_auto_headway(l, 240, 600, 0), 0);
    assert_eq!(a.w.net.line_auto[t as usize], Some([480, 1200, 0]));
    let bytes = save::encode(&a.w.net);
    let (net, unread) = save::decode_read(&bytes).unwrap();
    assert_eq!(unread, 0);
    assert_eq!(net.line_twin_of(l), Twin::Parent { split: true });
    assert_eq!(net.line_twin_of(t), Twin::Of(l));
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    assert_eq!((b.line_pairs(l), b.line_twin(l)), (1, t as i32));
    // a save without the block has no line on two pairs
    let mut c = TrackApi::new(-73.985, 40.758, true);
    let (mut a2, l2, _) = quad_line();
    assert!(!save::encode(&a2.w.net).windows(4).any(|w| w == b"LTW1"));
    assert_eq!(a2.set_line_pairs(l2, true), 0);
    assert!(save::encode(&a2.w.net).windows(4).any(|w| w == b"LTW1"), "on alone is kept too");
    assert!(c.load(&save::encode(&a2.w.net)));
    assert_eq!(c.line_pairs(l2), 2);
}

/// Built track: the twin runs only once the other pair is built too.
#[test]
fn t3333_built_line_waits_for_the_other_pair() {
    let (mut a, l, _) = quad_line();
    let drawn: Vec<u32> = live(&a).into_iter().filter(|&e| a.w.net.node_y[a.w.net.edge_a[e as usize] as usize].abs() < 1.0 && a.w.net.node_y[a.w.net.edge_b[e as usize] as usize].abs() < 1.0).collect();
    let stops = a.w.net.line_stops(l).to_vec();
    a.cash = 1e9;
    assert_eq!(a.construct(&drawn, &stops), 0, "{}", a.issues());
    assert!(a.w.net.line_constructed(l));
    let twins: Vec<u32> = [500.0, 2000.0, 3500.0].iter().map(|&x| station(&mut a, x, QUAD_SPACING, "x")).collect();
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    assert_eq!(a.line_pairs_why(l), vec![4.0, 0.0]);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert_eq!(a.line_pairs(l), 1, "{:?}", a.line_pairs_why(l));
    assert_eq!(a.w.net.line_stops(a.line_twin(l) as u32), &twins[..]);
}

/// T-3297: a game turned over to the other running side keeps its line on both pairs.
#[test]
fn t3333_a_flip_keeps_both_pairs() {
    let (mut a, l, _) = quad_line();
    for x in [500.0, 2000.0, 3500.0] {
        station(&mut a, x, QUAD_SPACING, "x");
    }
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    let t = a.line_twin(l) as u32;
    a.flip_right_hand();
    assert_eq!((a.line_pairs(l), a.line_twin(l)), (1, t as i32), "{:?}", a.line_pairs_why(l));
    assert!(a.w.svc.lines[l as usize].ok && a.w.svc.lines[t as usize].ok);
}

/// T-3356: a station beside a stop is any station of its complex on the pair beside, as a player
/// places them (clicked a few tens of metres off level); one past the complex counts once linked.
#[test]
fn t3356_stations_beside_are_the_complex_on_the_other_pair() {
    let (mut a, l, stops) = quad_line();
    let west = station(&mut a, 500.0 + 35.0, QUAD_SPACING, "x");
    let mid = station(&mut a, 2000.0 - 140.0, QUAD_SPACING, "x");
    // 400 m on from East with 200 m platforms: 200 m between their ends, not one complex
    let far = station(&mut a, 3500.0 - 400.0, QUAD_SPACING, "x");
    assert!(a.w.net.complex_of(mid).contains(&stops[1]), "Mid's platforms overlap");
    assert_eq!(a.w.net.station_beside(stops[0]), Some(west));
    assert_eq!(a.w.net.station_beside(stops[1]), Some(mid));
    assert_eq!(a.w.net.station_beside(stops[2]), None);
    assert!(!a.line_pairs_ok(l));
    assert_eq!(a.link_stations(stops[2], far), 0, "{}", a.issues());
    assert_eq!(a.w.net.station_beside(stops[2]), Some(far), "linked into East's complex");
    assert!(a.line_pairs_ok(l));
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    let t = a.line_twin(l) as u32;
    assert_eq!(a.w.net.line_stops(t), &[west, mid, far]);
    // a station of the complex on the same pair is never beside
    let (mut b, _, s2) = quad_line();
    let on = station(&mut b, 2000.0 + 230.0, 0.0, "x");
    assert!(b.w.net.complex_of(s2[1]).contains(&on));
    assert_eq!(b.w.net.station_beside(s2[1]), None);
}
