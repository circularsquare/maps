//! T-3332: one-track stations: a platform on one track of double track (a stop one way), and on single
//! track a stop with no passing loop.

use super::*;
use crate::track::capacity::ResKind;
use std::f64::consts::PI;

/// An 8 km straight run of `tracks` west to east, stations S0 and S1 1 km from its ends, and Mid at 4
/// km: one-track with its platform on the track nearer (4000, `side_y`) when `side_y` is set. Built
/// when `built`.
fn run(right_hand: bool, tracks: u8, side_y: Option<f64>, built: bool) -> (TrackApi, [u32; 3]) {
    let mut a = TrackApi::new(-73.985, 40.758, right_hand);
    a.set_cash(1e6);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{}", a.issues());
    let mut st = [0; 3];
    for (i, x) in [1000.0, 7000.0, 4000.0].into_iter().enumerate() {
        let edge = (0..a.w.net.edge_count() as u32)
            .find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            })
            .unwrap();
        match (i, side_y) {
            (2, Some(y)) => a.set_station_one(&[x, y]),
            _ => a.set_station_one(&[]),
        }
        assert_eq!(a.add_station(2, edge, x, 0.0, 200, ["S0", "S1", "Mid"][i].into()), 0, "{}", a.issues());
        st[i] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
    }
    a.set_station_one(&[]);
    if built {
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
    }
    (a, [st[0], st[2], st[1]])
}

fn line(a: &mut TrackApi, stops: &[u32]) -> u32 {
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, stops, 4, 2, 0, 30.0, 180.0, 0, 0, "Line".into()), 0, "{}", a.issue_kinds());
    id
}

fn try_line(a: &mut TrackApi, stops: &[u32]) -> u32 {
    let id = a.new_line_id();
    a.set_line_with_trains(id, stops, 4, 2, 0, 30.0, 180.0, 0, 0, "Line".into())
}

/// Users of the platform resources at `n`: (line, run).
fn platform_users(a: &TrackApi, n: u32) -> Vec<(u32, u8)> {
    let cap = &a.w.svc.cap;
    (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Platform && cap.res[r].node == n).flat_map(|r| cap.users_of(r).iter().map(|u| (u.line, u.run)).collect::<Vec<_>>()).collect()
}

fn singles(a: &TrackApi) -> usize {
    a.w.svc.cap.res.iter().filter(|r| r.kind == ResKind::SingleTrack).count()
}

/// On right-hand double track the platform south of an eastbound line's centre is on the eastbound
/// track; a line both ways stops there eastbound and runs past westbound, never standing; the station
/// is priced as a single-track one; its platform is the eastbound trains' alone.
#[test]
fn a_one_track_station_on_double_track_stops_one_way() {
    let (mut a, [s0, mid, s1]) = run(true, 2, Some(-10.0), true);
    assert_eq!(a.station_one(mid), 1);
    assert!(a.station_one_heading(mid).cos() > 0.99, "eastbound: {}", a.station_one_heading(mid));
    assert!(a.w.net.station_build_price(mid) < a.w.net.station_build_price(s0), "priced as single track");
    let l = line(&mut a, &[s0, mid, s1]);
    let ls = &a.w.svc.lines[l as usize];
    assert!(ls.running);
    assert_eq!(ls.runs[0].pass, Vec::<bool>::new(), "eastbound stops");
    assert_eq!(ls.runs[1].pass, vec![false, true, false], "westbound passes");
    for lev in 0..2 {
        let (p0, p1) = (&ls.runs[0].prof[lev], &ls.runs[1].prof[lev]);
        assert!(p0.dep[1] - p0.arr[1] >= 29.0, "a dwell eastbound");
        assert_eq!(p1.arr[1], p1.dep[1], "no stand westbound");
        assert!(p1.arr[1] > p1.arr[0] && p1.arr[1] < p1.arr[2]);
        // westbound passes at speed: well under the eastbound time with its stop
        assert!(p1.duration + 40.0 < p0.duration, "{} {}", p1.duration, p0.duration);
        assert!(crate::track::profile::speed_at(&ls.runs[1].free.phases, ls.runs[1].stop_s[1]) > 10.0, "running through");
    }
    assert!(platform_users(&a, mid).iter().all(|&(_, r)| r == 0), "{:?}", platform_users(&a, mid));
    assert!(!platform_users(&a, mid).is_empty());
    let ways = a.line_stop_ways(l);
    assert_eq!(ways.len(), 2);
    assert_eq!(ways[0] as u32, mid);
    assert!(ways[1].cos() > 0.99, "half dot eastbound");
    assert_eq!(a.line_dwell_split(l)[0], 30.0, "Mid, eastbound only");
}

/// Left-hand track carries eastbound trains on its north track: the same point south serves westbound.
#[test]
fn the_running_side_picks_the_way() {
    let (a, [_, mid, _]) = run(false, 2, Some(-10.0), false);
    assert!(a.station_one_heading(mid).cos() < -0.99);
    let (a, [_, mid, _]) = run(true, 2, Some(10.0), false);
    assert!(a.station_one_heading(mid).cos() < -0.99, "north of the centre on the right: westbound");
}

/// A line ends at a one-track station only arriving on its platform's track; one stopping only where it
/// passes is refused.
#[test]
fn a_line_ends_there_only_on_the_platforms_track() {
    let (mut a, [s0, mid, s1]) = run(true, 2, Some(-10.0), true);
    assert_eq!(try_line(&mut a, &[s1, mid]), 1, "arrives westbound");
    assert!(a.issue_kinds().contains("OneTrackStop"), "{}", a.issue_kinds());
    let l = line(&mut a, &[s0, mid]);
    assert!(a.w.svc.lines[l as usize].running, "arrives eastbound and turns");
    // turned round under that line: refused
    assert_ne!(a.reverse_station(mid), 0);
    assert!(a.issue_kinds().contains("OneTrackStop"));
    assert_eq!(a.w.net.node_one[mid as usize], 1);
    // a line through it both ways turns round with it
    a.remove_line(l);
    let l = line(&mut a, &[s1, mid, s0]);
    assert_eq!(a.reverse_station(mid), 0, "{}", a.issue_kinds());
    assert!(a.station_one_heading(mid).cos() < -0.99);
    let ls = &a.w.svc.lines[l as usize];
    assert_eq!((ls.runs[0].pass.clone(), ls.runs[1].pass.clone()), (vec![], vec![false, true, false]), "now stops westbound (run 0)");
}

/// The station panel's switch: to both tracks pays the difference on a constructed station, back to one
/// is free with nothing back; on blueprint both ways are free and undo. A terminal goes with the second
/// track and cannot be added back.
#[test]
fn switching_between_one_track_and_both() {
    let (mut a, [s0, mid, s1]) = run(true, 2, Some(-10.0), true);
    let l = line(&mut a, &[s0, mid, s1]);
    let cash = a.cash();
    assert_eq!(a.set_station_tracks(mid, false), 0, "{}", a.issue_kinds());
    assert!(cash - a.cash() > 1.0, "paid the difference: {}", cash - a.cash());
    assert!(a.w.svc.lines[l as usize].runs[1].pass.is_empty(), "both ways stop");
    assert_eq!(a.set_station_upgrade(&[mid], crate::track::upgrades::TERMINAL, true), 0);
    let cash = a.cash();
    assert_eq!(a.set_station_tracks(mid, true), 0, "{}", a.issue_kinds());
    assert_eq!(a.cash(), cash, "free, nothing back");
    assert_eq!(a.w.net.node_upgrades[mid as usize] & crate::track::upgrades::TERMINAL, 0, "the terminal goes");
    assert_ne!(a.set_station_upgrade(&[mid], crate::track::upgrades::TERMINAL, true), 0);
    assert_eq!(a.station_one(mid), 1);
    // blueprint: free and undone
    let (mut b, [_, mid, _]) = run(true, 2, None, false);
    assert_eq!(b.set_station_tracks(mid, true), 0, "{}", b.issue_kinds());
    assert_eq!(b.station_one(mid), 1);
    assert_eq!(b.undo(), 0);
    assert_eq!(b.station_one(mid), 0);
}

/// A line ending at a both-tracks station eastbound: switched to one track it keeps the way the line
/// needs (eastbound), whichever is tried first.
#[test]
fn switching_to_one_track_keeps_the_way_its_lines_need() {
    let (mut a, [s0, mid, s1]) = run(true, 2, None, true);
    line(&mut a, &[s1, mid]);
    assert_eq!(a.set_station_tracks(mid, true), 0, "{}", a.issue_kinds());
    assert!(a.station_one_heading(mid).cos() < -0.99, "westbound, the line's way in");
    assert_ne!(try_line(&mut a, &[s0, mid]), 0, "eastbound in is the other way");
}

/// Saved and loaded: the way and the station's price stay.
#[test]
fn one_track_stations_save_and_load() {
    let (mut a, [s0, mid, s1]) = run(true, 2, Some(10.0), true);
    line(&mut a, &[s1, mid, s0]);
    let bytes = a.save();
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    assert_eq!(b.w.net.node_one[mid as usize], a.w.net.node_one[mid as usize]);
    assert!(b.station_one_heading(mid).cos() < -0.99);
    assert!((b.w.net.station_build_price(mid) - a.w.net.station_build_price(mid)).abs() < 1e-9);
    assert_eq!(b.w.svc.lines[0].runs[1].pass, vec![false, true, false]);
}

/// On single track a one-track station is a stop with no passing loop: the single-track section runs
/// on through it, where a passing station cuts it.
#[test]
fn on_single_track_it_is_a_stop_with_no_passing() {
    let (mut a, [s0, mid, s1]) = run(true, 1, None, true);
    line(&mut a, &[s0, mid, s1]);
    let passing = singles(&a);
    let (mut b, [s0, mid, s1]) = run(true, 1, Some(-10.0), true);
    assert_eq!(b.station_one(mid), 2, "no passing");
    line(&mut b, &[s0, mid, s1]);
    assert!(b.w.svc.lines[0].runs.iter().all(|r| r.pass.is_empty()), "both ways stop");
    assert_eq!(singles(&b), passing - 1, "one section the less");
    assert_eq!(platform_users(&b, mid).len(), 2);
}

/// One-way single track takes no one-track station; neither does a junction.
#[test]
fn not_on_one_way_single_track() {
    let (mut a, [_, mid, _]) = run(true, 1, None, false);
    let e = a.w.net.node_ports[mid as usize].ends()[0] >> 1;
    assert_eq!(a.set_edge_oneway(e, 1), 0, "{}", a.issue_kinds());
    assert_ne!(a.set_station_tracks(mid, true), 0);
    assert!(a.issue_kinds().contains("OneTrackHere"));
}

/// Rebuilt: double to single keeps it one-track (no passing); single to double makes it an ordinary
/// station.
#[test]
fn rebuilding_the_track() {
    let (mut a, [_, mid, _]) = run(true, 2, Some(-10.0), true);
    let edges: Vec<u32> = (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect();
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 1), 0, "{}", a.issue_kinds());
    assert_eq!(a.station_one(mid), 2, "a stop with no passing");
    assert_eq!(a.rebuild_pieces(&edges, &[], 99, 0, 2), 0, "{}", a.issue_kinds());
    assert_eq!(a.station_one(mid), 0, "both tracks");
}

/// A running side switched with track laid: the platform stays on its track, which now carries the
/// other way, and the station is listed.
#[test]
fn a_flip_keeps_the_platform_on_its_track() {
    let (mut a, [s0, mid, s1]) = run(true, 2, Some(-10.0), true);
    line(&mut a, &[s0, mid, s1]);
    let sites = a.flip_right_hand();
    assert!(sites.chunks(2).any(|c| c[0] == 6.0 && c[1] as u32 == mid), "{sites:?}");
    assert!(a.station_one_heading(mid).cos() < -0.99, "the south track is westbound on the left");
    assert_eq!(a.w.svc.lines[0].runs[1].pass, Vec::<bool>::new());
    assert_eq!(a.w.svc.lines[0].runs[0].pass, vec![false, true, false]);
    let _ = PI;
}

/// T-3333: on quad track a line runs on both pairs only where each one-track stop has a one-track station
/// beside it serving the same way.
#[test]
fn a_twin_serves_the_same_way() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    let ends = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    let at = |a: &TrackApi, x: f64, y: f64| {
        (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e) && !a.w.net.edge_crossover[e as usize]).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap()
    };
    let place = |a: &mut TrackApi, x: f64, y: f64, one: Option<f64>| {
        match one {
            Some(dy) => a.set_station_one(&[x, y + dy]),
            None => a.set_station_one(&[]),
        }
        assert_eq!(a.add_station(2, at(a, x, y), x, y, 200, "S".into()), 0, "{}", a.issues());
        a.set_station_one(&[]);
        (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).min_by(|&p, &q| {
            let d = |n: u32| (a.w.net.node_x[n as usize] - x).hypot(a.w.net.node_y[n as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap()
    };
    let stops = [place(&mut a, 500.0, 0.0, None), place(&mut a, 2000.0, 0.0, Some(-1.0)), place(&mut a, 3500.0, 0.0, None)];
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &stops, 6, 4, 2, 30.0, 180.0, 0, 0, "Q".into()), 0, "{}", a.issues());
    assert_eq!(a.set_line_pairs(l, true), 0, "{}", a.issues());
    place(&mut a, 500.0, QUAD_SPACING, None);
    place(&mut a, 3500.0, QUAD_SPACING, None);
    // the other way on the other pair: no twin
    let mid = place(&mut a, 2000.0, QUAD_SPACING, Some(1.0));
    assert_eq!(a.station_one(mid), 1);
    assert!(a.station_one_heading(mid).cos() < -0.99);
    assert_eq!(a.line_pairs_why(l), vec![1.0, 1.0], "Mid's station beside serves the other way");
    assert_eq!(a.line_twin(l), -1);
    assert_eq!(a.reverse_station(mid), 0, "{}", a.issue_kinds());
    assert_eq!(a.line_pairs(l), 1, "the same way now: {:?}", a.line_pairs_why(l));
    assert!(a.line_twin(l) >= 0);
}

/// Anita 2026-11-13: a station with one platform track (a one way station, or a stop on single track with no
/// passing loop) pays 0.6x the upkeep of a two way station, at grade and underground.
#[test]
fn a_one_platform_station_pays_six_tenths_upkeep() {
    use crate::track::params::{UPKEEP_ONE_PLATFORM_SHARE as SHARE, UPKEEP_STATION as ST, UPKEEP_STATION_UNDER as UG};
    let (two, [_, mid2, _]) = run(true, 2, None, true);
    let (mut one, [_, mid1, _]) = run(true, 2, Some(-10.0), true);
    let (mut single, [_, mid_s, _]) = run(true, 1, None, true);
    let up = |a: &TrackApi, n: u32| station_upkeep(&a.w.net, n as usize);
    assert!((up(&two, mid2) - ST * ECONOMY / 1e6).abs() < 1e-9);
    assert!((up(&one, mid1) - SHARE * up(&two, mid2)).abs() < 1e-9);
    assert!((up(&single, mid_s) - SHARE * up(&two, mid2)).abs() < 1e-9, "single track, no loop");
    // a passing loop is two platforms again
    assert_eq!(single.set_station_upgrade(&[mid_s], crate::track::upgrades::PASSING, true), 0, "{}", single.issue_kinds());
    assert!((up(&single, mid_s) - up(&two, mid2)).abs() < 1e-9, "single track with a passing loop");
    // underground
    let mut two = two;
    two.w.net.node_level[mid2 as usize] = -1;
    one.w.net.node_level[mid1 as usize] = -1;
    assert!((up(&two, mid2) - UG * ECONOMY / 1e6).abs() < 1e-9);
    assert!((up(&one, mid1) - SHARE * up(&two, mid2)).abs() < 1e-9);
}
