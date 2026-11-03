use super::super::capacity::ResKind;
use super::super::net::EdgeData;
use super::super::params::PLATFORM_MARGIN_S;
use super::super::service::Service;
use super::super::tests::{line, node, station, world};
use super::super::upgrades::{self, PASSING};
use super::super::world::{Op, TrackWorld};
use super::*;

/// The Shinkansen-style line of notes/T-2228.md 2: 180 km of high-speed track, stations A to G every
/// 30 km with 400 m platforms; high-speed locals at every station, high-speed expresses at A, D and
/// G. High demand runs both every 10 minutes, medium locals every 30 and expresses every 10.
fn shinkansen() -> (TrackWorld, Vec<u32>, u32, u32) {
    let mut w = world();
    let n: Vec<u32> = (0..7).map(|k| node(&mut w, k as f64 * 30_000.0, 0.0, 0)).collect();
    for k in 0..6 {
        let id = w.net.alloc_edge();
        w.apply(Op::Edge { id, data: Some(EdgeData { a: n[k], b: n[k + 1], tracks: 2, pis: vec![], built: true, off: false, thru: vec![], crossover: false, hsr: true, sep: false }) }).unwrap();
    }
    for &x in &n {
        station(&mut w, x, 400);
    }
    let fast = |w: &mut TrackWorld, l: u32| {
        let mut d = w.net.line_data(l).unwrap();
        d.fast = true;
        w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    };
    let local = line(&mut w, &n, [6.0, 2.0, 0.0]);
    fast(&mut w, local);
    let express = line(&mut w, &[n[0], n[3], n[6]], [6.0, 6.0, 0.0]);
    fast(&mut w, express);
    (w, n, local, express)
}

fn set_passing(w: &mut TrackWorld, nodes: &[u32], on: bool) {
    let ops = nodes.iter().map(|&x| {
        let mut d = w.net.station_data(x).unwrap();
        d.upgrades = if on { d.upgrades | PASSING } else { d.upgrades & !PASSING };
        Op::Station { node: x, data: Some(d) }
    }).collect();
    w.apply(Op::Batch(ops)).unwrap();
}

/// The incremental capacity pass and the lines' figures against everything worked out afresh.
fn same_as_global(w: &TrackWorld, what: &str) {
    let mut g = Service::default();
    g.update_all(&w.net);
    assert_eq!(w.svc.cap.canonical(), g.cap.canonical(), "{what}: resources");
    for (l, (x, y)) in w.svc.lines.iter().zip(&g.lines).enumerate() {
        for r in 0..2 {
            assert_eq!((&x.runs[r].holds, &x.runs[r].stop_holds), (&y.runs[r].holds, &y.runs[r].stop_holds), "{what}: line {l} run {r}");
        }
        assert_eq!((x.round_trip, x.delay), (y.round_trip, y.delay), "{what}: line {l}");
    }
}

/// Minutes over its free time each run of `l` takes at `lev`.
fn over(w: &TrackWorld, l: u32, lev: usize) -> [f64; 2] {
    std::array::from_fn(|r| {
        let run = &w.svc.lines[l as usize].runs[r];
        (run.prof[lev].duration - run.free.duration) / 60.0
    })
}

/// T-2231's done-when (notes/T-2228.md 6): with passing tracks at B to F the express runs within a
/// minute of its free time at 10/10 and 30/10 and the local is 5-12 minutes slower one way; at 10/10
/// every local waits at the same stations, and its waits stand at those stops.
#[test]
fn passing_tracks_let_the_express_by() {
    let (mut w, n, local, express) = shinkansen();
    let free = |l: u32| w.svc.lines[l as usize].runs[0].free.duration / 60.0;
    eprintln!("free: local {:.1} min, express {:.1} min", free(local), free(express));
    for lev in 0..2 {
        eprintln!("level {lev} without: local +{:.1?} express +{:.1?}", over(&w, local, lev), over(&w, express, lev));
        assert!(over(&w, express, lev).iter().all(|&m| m > 8.0), "without passing tracks the express trails the local");
    }
    set_passing(&mut w, &n[1..6], true);
    same_as_global(&w, "passing B to F");
    for lev in 0..2 {
        let (lo, ex) = (over(&w, local, lev), over(&w, express, lev));
        eprintln!("level {lev} with: local +{lo:.1?} express +{ex:.1?}");
        // the overtakes' part of it (kind 11); the rest is the queue on track two lines share, which
        // it pays with or without passing tracks (T-517)
        let cap = &w.svc.cap;
        let behind: Vec<f64> = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) == 11 && cap.users_of(r).iter().any(|u| u.owner && u.line == express)).map(|r| cap.delay[r][lev]).collect();
        let queue: Vec<f64> = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) != 11 && cap.users_of(r).iter().any(|u| u.owner && u.line == express && u.run == 0)).map(|r| cap.delay[r][lev]).collect();
        eprintln!("  express behind slower trains {behind:.0?} s, queues on run 0 {queue:.0?} s");
        assert!(behind.len() == 2 && behind.iter().all(|&d| d < 60.0), "level {lev}: the express loses under a minute to overtakes: {behind:?}");
        assert!(ex.iter().all(|&m| m < 1.0 + queue.iter().sum::<f64>() / 60.0), "level {lev}: the express within a minute of its free time and its queues: {ex:?}");
        assert!(lo.iter().all(|&m| (5.0..=12.0).contains(&m)), "level {lev}: the local 5-12 minutes slower: {lo:?}");
    }
    // the express's run 0 behind the local: its following loss reads the plan
    let cap = &w.svc.cap;
    let r = (0..cap.res.len()).find(|&r| cap.res[r].kind == ResKind::Follow && cap.users_of(r).iter().any(|u| u.owner && u.line == express && u.run == 0)).unwrap();
    let plan = &cap.plans_of(r)[0];
    assert_eq!(plan.slow.0, local);
    let (o, each) = plan.trains_at(600.0, 600.0);
    let waited = |t: &Vec<f64>| t.iter().map(|&x| x > 1.0).collect::<Vec<bool>>();
    eprintln!("10/10: offset {:.0} s, loss {:.0} s, waits {:?}", o.offset, o.loss, o.waits.iter().map(|x| x.round()).collect::<Vec<_>>());
    assert_eq!(each.len(), 6, "an hour of locals");
    assert!(each.iter().all(|t| waited(t) == waited(&each[0])), "every local waits at the same stations: {each:?}");
    assert!(waited(&each[0]).iter().any(|&x| x), "it waits somewhere");
    // the waits stand at the local's stops (kind 13), only at passing stations
    let stops = &w.svc.lines[local as usize].runs[0];
    for (j, &h) in stops.stop_holds[0].iter().enumerate() {
        let st = stops.nodes[stops.stop_idx[j] as usize];
        assert!(h == 0.0 || n[1..6].contains(&st), "a wait at {st}, not a passing station");
    }
    assert!(stops.stop_holds[0].iter().sum::<f64>() > 120.0);
    let waits: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) == 13).collect();
    assert!(!waits.is_empty());
    for &r in &waits {
        assert_eq!(cap.rho[r], [0.0; 3], "no utilisation, no dot");
        assert!(!cap.limits(r));
        assert_eq!(cap.wait_for(r), Some(express));
        assert!(n[1..6].contains(&cap.res[r].node));
    }
    // the express's phase is placed at the plan's offset behind the local
    assert!(cap.follow_targets(&w.net, 0).iter().any(|x| (x.0 .0, x.1 .0) == (express, local)));
    // removing them puts it back as it was
    let with = over(&w, express, 0);
    set_passing(&mut w, &n[1..6], false);
    same_as_global(&w, "passing removed");
    assert!(over(&w, express, 0).iter().all(|&m| m > 8.0));
    assert!(!(0..w.svc.cap.res.len()).any(|r| w.svc.cap.wire_kind(r) == 13));
    w.undo().unwrap().unwrap();
    same_as_global(&w, "passing undone");
    assert_eq!(over(&w, express, 0), with);
}

/// T-2231: a station with passing tracks gives each direction two platform tracks: a stop is pooled
/// over them, and a train that does not stop holds none.
#[test]
fn passing_tracks_free_the_platform() {
    let (mut w, n, local, express) = shinkansen();
    set_passing(&mut w, &[n[1]], true);
    let cap = &w.svc.cap;
    let plats: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Platform && cap.res[r].node == n[1]).collect();
    assert_eq!(plats.len(), 2, "one a direction");
    for r in plats {
        let us = cap.users_of(r);
        assert!(us.iter().all(|u| u.line == local), "the express passes on its own track: {us:?}");
        assert_eq!(us[0].occ, (30.0 + PLATFORM_MARGIN_S) / 2.0);
    }
    let _ = express;
}

/// T-2231: the bit is saved and loaded, and the loaded game runs the same.
#[test]
fn passing_tracks_save_and_load() {
    let (mut w, n, local, express) = shinkansen();
    set_passing(&mut w, &n[1..6], true);
    let back = TrackWorld::new(super::super::save::decode(&super::super::save::encode(&w.net)).unwrap());
    for &x in &n[1..6] {
        assert_eq!(back.net.station_data(x).unwrap().upgrades, PASSING);
    }
    for l in [local, express] {
        assert_eq!(back.svc.lines[l as usize].round_trip, w.svc.lines[l as usize].round_trip);
    }
}

/// T-2231: the price is the platform plus 400 m of the station's track at its level: US$22.5M for a
/// 200 m ground station, US$45M for a 400 m one on high-speed track, twice over water.
#[test]
fn passing_tracks_price() {
    assert!((upgrades::price(PASSING, 0, false, 200, false) - 22.5).abs() < 1e-9);
    assert!((upgrades::price(PASSING, 0, false, 400, true) - 45.0).abs() < 1e-9);
    assert!((upgrades::price(PASSING, 0, true, 200, false) - 45.0).abs() < 1e-9);
    let (w, n, _, _) = shinkansen();
    assert!((w.net.station_upgrades_price(n[2], PASSING, false) - 45.0).abs() < 1e-9);
    // the other upgrades keep their prices
    assert!((upgrades::price(upgrades::TERMINAL, 0, false, 400, true) - 37.5).abs() < 1e-9);
}

/// T-2231: the line button's stations: the local's stops strictly inside the shared stretch, from
/// either line's side, less those with passing tracks.
#[test]
fn passing_spots_are_the_slower_lines_stops_inside() {
    let (mut w, n, local, express) = shinkansen();
    assert_eq!(passing_spots(&w.net, &w.svc.lines, express), n[1..6].to_vec());
    assert_eq!(passing_spots(&w.net, &w.svc.lines, local), n[1..6].to_vec());
    set_passing(&mut w, &[n[2], n[4]], true);
    assert_eq!(passing_spots(&w.net, &w.svc.lines, express), vec![n[1], n[3], n[5]]);
    set_passing(&mut w, &[n[1], n[3], n[5]], true);
    assert!(passing_spots(&w.net, &w.svc.lines, express).is_empty());
    // a line on its own track has none
    let q = [node(&mut w, 0.0, 5000.0, 0), node(&mut w, 30_000.0, 5000.0, 0)];
    super::super::tests::edge(&mut w, q[0], q[1], &[], 0).unwrap();
    for &x in &q {
        station(&mut w, x, 200);
    }
    let alone = line(&mut w, &q, [6.0, 0.0, 0.0]);
    assert!(passing_spots(&w.net, &w.svc.lines, alone).is_empty());
}
