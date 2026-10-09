//! End-to-end tests of the track model through edit operations.

use super::capacity::ResKind;
use super::cost::{RunMask, WaterMask};
use super::geom::{self, GeomIssue, Pi};
use super::net::{CrossKind, EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::world::{Op, TrackWorld};

pub fn world() -> TrackWorld {
    TrackWorld::new(Network::new(-73.985, 40.758, true))
}

pub fn node(w: &mut TrackWorld, x: f64, y: f64, level: i8) -> u32 {
    let id = w.net.alloc_node();
    w.apply(Op::Node { id, data: Some(NodeData { x, y, level, flying: false }) }).unwrap();
    id
}

pub fn edge(w: &mut TrackWorld, a: u32, b: u32, pis: &[(f64, f64)], level: i8) -> Result<u32, Vec<Issue>> {
    let id = w.net.alloc_edge();
    let pis = pis.iter().map(|&(x, y)| Pi::new(x, y, 0.0, level)).collect();
    w.apply(Op::Edge { id, data: Some(EdgeData { a, b, tracks: 2, pis, built: true, thru: vec![] }) })?;
    Ok(id)
}

pub fn station(w: &mut TrackWorld, n: u32, platform: u16) {
    w.apply(Op::Station { node: n, data: Some(StationData { platform, name: format!("S{n}"), built: true }) }).unwrap();
}

pub fn line(w: &mut TrackWorld, stops: &[u32], tph: [f32; 3]) -> u32 {
    let path = w.net.route(stops).expect("no route");
    let id = w.net.alloc_line();
    let d = LineData { name: format!("L{id}"), colour: 0, stops: stops.to_vec(), path, tph, dwell_s: 30.0, turnaround_s: 180.0, cars: 0 };
    w.apply(Op::Line { id, data: Some(d) }).unwrap();
    id
}

/// A - B - C, stations at all three; B-C bends at a PI.
fn simple() -> (TrackWorld, [u32; 3], [u32; 2], u32) {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 3000.0, 0.0, 0);
    let c = node(&mut w, 6000.0, 800.0, 0);
    let e0 = edge(&mut w, a, b, &[], 0).unwrap();
    let e1 = edge(&mut w, b, c, &[(4500.0, 0.0)], 0).unwrap();
    for n in [a, b, c] {
        station(&mut w, n, 200);
    }
    let l = line(&mut w, &[a, b, c], [12.0, 6.0, 3.0]);
    (w, [a, b, c], [e0, e1], l)
}

#[test]
fn a_line_runs_with_times_and_fleet() {
    let (w, _, _, l) = simple();
    let ls = &w.svc.lines[l as usize];
    assert!(ls.ok);
    assert_eq!(ls.cars, 10);
    let cum = ls.cumulative(0, HIGH);
    assert_eq!(cum.len(), 3);
    assert_eq!(cum[0], 0.0);
    assert!(cum[1] > 60.0 && cum[2] > cum[1] + 30.0, "{cum:?}");
    let s2s = ls.stop_to_stop(0, HIGH);
    assert!((s2s[2] - (cum[2] - cum[1] - 30.0)).abs() < 1e-9);
    // Both directions take the same time on a symmetric free run.
    let rt = ls.round_trip[HIGH];
    assert!((rt - (ls.runs[0].prof[HIGH].duration + ls.runs[1].prof[HIGH].duration + 360.0)).abs() < 1e-9);
    assert_eq!(ls.trains[HIGH], (rt * 12.0 / 3600.0).ceil() as u32);
    // Trips in the 8:00 hour: 12 per direction start, plus those still running from before.
    let trips = w.svc.trips(&w.net, 8.0 * 3600.0, 9.0 * 3600.0);
    let starting = trips.iter().filter(|t| t.dep >= 8.0 * 3600.0).count();
    assert_eq!(starting, 24);
    assert!(trips.len() > 24);
    // The train's position follows the track.
    let ph = &ls.runs[0].prof[HIGH].phases;
    let (s, _) = super::profile::state_at(ph, cum[1] + 30.0 + 20.0);
    let (x, y) = w.svc.position(&w.net, l, 0, s);
    assert!(x > 3000.0 && x < 4500.0 && y.abs() < 1e-6, "{x} {y}");
}

#[test]
fn split_keeps_geometry_and_times_and_undoes_exactly() {
    let (mut w, _, [_, e1], l) = simple();
    let before = w.net.edge_data(e1).unwrap();
    let len = w.net.edge_len[e1 as usize];
    let pts: Vec<(f64, f64)> = (0..=60).map(|j| {
        let (x, y, _) = geom::pos_at(w.net.edge_pieces(e1), len * j as f64 / 60.0);
        (x, y)
    }).collect();
    let times = w.svc.lines[l as usize].cumulative(0, HIGH).to_vec();
    let arc = w.net.edge_pieces(e1).iter().find(|p| p.k != 0.0).copied().unwrap();
    let s = arc.s0 + arc.len * 0.4;
    let n = w.net.alloc_node();
    let e2 = w.net.alloc_edge();
    let r = w.apply(Op::Split { edge: e1, s, node: n, new_edge: e2 }).unwrap();
    assert!(r.dirty.service.demand.is_empty(), "a split must not wake demand");
    assert_eq!(w.net.node_ports[n as usize].n, 2);
    let (l1, l2) = (w.net.edge_len[e1 as usize], w.net.edge_len[e2 as usize]);
    assert!((l1 + l2 - len).abs() < 0.01, "{l1} + {l2} vs {len}");
    for (j, &(x, y)) in pts.iter().enumerate() {
        let c = len * j as f64 / 60.0;
        let (e, d) = if c <= l1 { (e1, c) } else { (e2, c - l1) };
        let (px, py, _) = geom::pos_at(w.net.edge_pieces(e), d);
        assert!((px - x).hypot(py - y) < 0.01, "at {c}: moved {}", (px - x).hypot(py - y));
    }
    let after = w.svc.lines[l as usize].cumulative(0, HIGH).to_vec();
    for (a, b) in times.iter().zip(&after) {
        assert!((a - b).abs() < 0.05, "{times:?} vs {after:?}");
    }
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.edge_data(e1).unwrap(), before);
    assert!(!w.net.node_ok(n) && !w.net.edge_ok(e2));
    assert!(w.svc.lines[l as usize].ok);
    w.redo().unwrap().unwrap();
    assert!(w.net.node_ok(n) && w.net.edge_ok(e2));
}

#[test]
fn bad_edits_are_refused_and_change_nothing() {
    let (mut w, [_, b, c], [_, e1], _) = simple();
    let before = w.net.edge_data(e1).unwrap();
    let mut d = before.clone();
    d.pis[0].radius = 50.0;
    let err = w.apply(Op::Edge { id: e1, data: Some(d) }).unwrap_err();
    assert!(err.contains(&Issue::Geom { edge: e1, issue: GeomIssue::RadiusTooSmall(1) }), "{err:?}");
    assert_eq!(w.net.edge_data(e1).unwrap(), before);
    // A second edge leaving B at an angle is a kink, not a switch.
    let x = node(&mut w, 3000.0, 2000.0, 0);
    assert!(edge(&mut w, b, x, &[], 0).is_err());
    // Removing a station a line stops at.
    assert!(w.apply(Op::Station { node: c, data: None }).is_err());
    // Platform longer than the room on the edge.
    let mut s = w.net.station_data(b).unwrap();
    s.platform = 400;
    assert!(w.apply(Op::Station { node: b, data: Some(s.clone()) }).is_ok());
    s.platform = 410;
    assert!(w.apply(Op::Station { node: b, data: Some(s) }).is_err());
}

#[test]
fn crossings_flat_separated_or_invalid() {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 4000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    // Same level: a flat crossing, charged.
    let c = node(&mut w, 2000.0, -2000.0, 0);
    let d = node(&mut w, 2000.0, 2000.0, 0);
    let before = w.net.total_cost();
    let e = edge(&mut w, c, d, &[], 0).unwrap();
    assert_eq!(w.net.crossings.len(), 1);
    assert_eq!(w.net.crossings[0].kind, CrossKind::Flat(0));
    let want = 4.0 * 27.0 + super::cost::crossing_cost(0, false);
    assert!((w.net.total_cost() - before - want).abs() < 1e-6);
    // Moved down a level (ramps on long legs): separated.
    let mut ed = w.net.edge_data(e).unwrap();
    ed.pis = vec![Pi::new(2000.0, -1000.0, 0.0, -1), Pi::new(2000.0, 1000.0, 0.0, -1)];
    w.apply(Op::Edge { id: e, data: Some(ed.clone()) }).unwrap();
    assert!(w.net.crossings.is_empty());
    // A ramp over the other track (from 0 at c to -1 at d, centred on y = 0): refused.
    ed.pis = vec![];
    let ops = vec![Op::Edge { id: e, data: Some(ed) }, Op::Node { id: d, data: Some(NodeData { x: 2000.0, y: 2000.0, level: -1, flying: false }) }];
    let err = w.apply(Op::Batch(ops)).unwrap_err();
    assert!(err.iter().any(|i| matches!(i, Issue::Crossing { .. })), "{err:?}");
}

/// Y junction: trunk T-J, main J-M, branch J-Br. Line A runs T-Br, line B T-M.
fn y_junction(tph: [f32; 3], flying: bool) -> (TrackWorld, u32, u32, u32) {
    let mut w = world();
    let t = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let m = node(&mut w, 6000.0, 0.0, 0);
    let br = node(&mut w, 5000.0, 1500.0, 0);
    edge(&mut w, t, j, &[], 0).unwrap();
    edge(&mut w, j, m, &[], 0).unwrap();
    edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
    for n in [t, m, br] {
        station(&mut w, n, 200);
    }
    if flying {
        w.apply(Op::Node { id: j, data: Some(NodeData { x: 3000.0, y: 0.0, level: 0, flying: true }) }).unwrap();
    }
    let a = line(&mut w, &[t, br], tph);
    let b = line(&mut w, &[t, m], tph);
    (w, a, b, j)
}

#[test]
fn junction_delay_matches_the_spec_example() {
    // Each move of the crossing pair sees 15 + 15 trains at 90 s: 75% and 27 s (high), and
    // 18 + 18: 90% and 81 s (medium).
    let (w, a, b, j) = y_junction([15.0, 18.0, 0.0], false);
    let cap = &w.svc.cap;
    let jr: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Junction).collect();
    assert_eq!(jr.len(), 2, "one crossing pair, one resource per move");
    for &r in &jr {
        assert_eq!(cap.res[r].node, j);
        assert!((cap.rho[r][HIGH] - 0.75).abs() < 1e-9);
        assert_eq!(cap.delay[r][HIGH], 27.0);
        assert!((cap.rho[r][MEDIUM] - 0.9).abs() < 1e-9);
        assert_eq!(cap.delay[r][MEDIUM], 81.0);
        assert_eq!(cap.delay[r][LOW], 0.0);
    }
    // The crossing pair is A towards the branch and B in from the main line (right-hand running).
    let owners: Vec<(u32, u8)> = jr.iter().map(|&r| cap.users_of(r).iter().find(|u| u.owner).map(|u| (u.line, u.run)).unwrap()).collect();
    assert!(owners.contains(&(a, 0)) && owners.contains(&(b, 1)), "{owners:?}");
    // Baked into the profile as a hold.
    let run = &w.svc.lines[a as usize].runs[0];
    assert!(run.holds[HIGH].iter().any(|h| h.delay == 27.0));
    assert!(run.prof[HIGH].hold >= 27.0);
    // A flying junction removes it.
    let (w, _, _, _) = y_junction([15.0, 18.0, 0.0], true);
    assert!(!w.svc.cap.res.iter().any(|r| r.kind == ResKind::Junction));
}

#[test]
fn schedule_change_dirties_shared_lines_and_demand() {
    let (mut w, a, b, _) = y_junction([10.0, 6.0, 3.0], false);
    let r = w.apply(Op::Schedule { id: a, tph: [16.0, 6.0, 3.0] }).unwrap();
    assert!(r.dirty.service.shared.contains(&b));
    assert!(r.dirty.service.demand.contains(&a));
    assert!(r.dirty.tiles.is_empty());
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.line_tph[a as usize], [10.0, 6.0, 3.0]);
}

#[test]
fn delete_edge_breaks_or_reroutes_and_undo_restores() {
    let (mut w, a, b, _) = y_junction([10.0, 6.0, 3.0], false);
    let jm = 1; // J-M
    let path_b = w.net.line_path(b).to_vec();
    w.apply(Op::DeleteEdge { id: jm }).unwrap();
    assert!(w.net.line_broken[b as usize] && !w.svc.lines[b as usize].ok);
    assert!(w.svc.lines[a as usize].ok);
    w.undo().unwrap().unwrap();
    assert!(!w.net.line_broken[b as usize] && w.svc.lines[b as usize].ok);
    assert_eq!(w.net.line_path(b), &path_b[..]);
}

/// Read the New York pack's water mask, if the pack is there (it is generated, not checked in).
pub fn nyc_water() -> Option<RunMask> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/packs/");
    let head = std::fs::read_to_string(format!("{dir}nyc.json")).ok()?;
    let bin = std::fs::read(format!("{dir}nyc.bin")).ok()?;
    // The first `n` numbers after `key`, searching from byte `from`.
    let nums = |from: usize, key: &str, n: usize| -> Vec<f64> {
        let mut rest = &head[head[from..].find(key).unwrap() + from + key.len()..];
        let mut out = vec![];
        while out.len() < n {
            let i = rest.find(|c: char| c.is_ascii_digit() || c == '-').unwrap();
            rest = &rest[i..];
            let e = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e')).unwrap();
            out.push(rest[..e].parse().unwrap());
            rest = &rest[e..];
        }
        out
    };
    let arr = |name: &str| {
        let i = head.find(&format!("\"{name}\"")).unwrap();
        (nums(i, "\"offset\"", 1)[0] as usize, nums(i, "\"count\"", 1)[0] as usize)
    };
    let w = head.find("\"water\": {")?;
    let cell = nums(w, "\"cell_m\"", 1)[0];
    let o = nums(w, "\"origin_m\"", 2);
    let sz = nums(w, "\"size\"", 2);
    let (ro, rc) = arr("water_row");
    let (xo, xc) = arr("water_x");
    let (ox, oy, sw, sh) = (o[0], o[1], sz[0], sz[1]);
    Some(RunMask::from_pack(cell, [ox, oy], [sw as u32, sh as u32], &bin[ro..ro + 4 * rc], &bin[xo..xo + 2 * xc]))
}

#[test]
fn hudson_crossing_needs_a_bridge_or_tunnel() {
    let Some(mask) = nyc_water() else {
        eprintln!("data/packs/nyc.* not found; skipping");
        return;
    };
    // Straight west from the origin (Times Square) to Weehawken: the Hudson is in between.
    let wet: usize = (0..400).filter(|&j| mask.is_water(-(j as f64) * 10.0, 0.0)).count();
    assert!(wet > 60, "expected ~1.2 km of river, got {} samples", wet);
    assert!(!mask.is_water(0.0, 0.0));
    let mut w = world();
    w.net.set_water_mask(Box::new(mask));
    w.reload();
    let a = node(&mut w, -500.0, 0.0, 0);
    let b = node(&mut w, -4000.0, 0.0, 0);
    let err = edge(&mut w, a, b, &[], 0).unwrap_err();
    assert!(err.iter().any(|i| matches!(i, Issue::GroundOverWater { .. })), "{err:?}");
    // Under the river at -1 (ramps on land at both ends): fine, and the wet part costs double.
    let e = edge(&mut w, a, b, &[(-1000.0, 0.0), (-3500.0, 0.0)], -1).unwrap();
    let wet_len: f64 = w.net.edge_water(e).iter().map(|s| s[1] - s[0]).sum();
    assert!(wet_len > 600.0);
    let len = w.net.edge_len[e as usize];
    let c = w.net.edge_cost[e as usize];
    assert!(c > len / 1000.0 * 90.0 * 0.9, "water should raise the price: {c}");
    println!("Hudson at -1: {len:.0} m, {wet_len:.0} m wet, US${c:.0}M");
}

#[test]
fn save_round_trip_is_exact() {
    let (w, _, _, l) = simple();
    let (y, _, _, _) = y_junction([10.0, 6.0, 3.0], true);
    for w in [&w, &y] {
        let bytes = super::save::encode(&w.net);
        let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
        assert_eq!(super::save::encode(&back.net), bytes);
        assert_eq!(back.net.total_cost(), w.net.total_cost());
        let a: Vec<f64> = w.svc.lines.iter().filter(|x| x.ok).map(|x| x.round_trip[HIGH]).collect();
        let b: Vec<f64> = back.svc.lines.iter().filter(|x| x.ok).map(|x| x.round_trip[HIGH]).collect();
        assert_eq!(a, b);
    }
    let _ = l;
}

#[test]
fn synthetic_network_builds() {
    let w = TrackWorld::new(super::bench::synth(200.0, 50, 9, 3));
    let ok = w.svc.lines.iter().filter(|l| l.ok).count();
    assert!(ok >= 6, "{ok} lines running");
}

/// T-063 measurements: the synthetic network as a track save, for loading into the app.
/// `TW_BENCH_SAVE=<file> TW_BENCH_KM=2000 cargo test --release --lib write_bench_save -- --ignored`
#[test]
#[ignore]
fn write_bench_save() {
    let path = std::env::var("TW_BENCH_SAVE").expect("TW_BENCH_SAVE");
    let km: f64 = std::env::var("TW_BENCH_KM").ok().and_then(|v| v.parse().ok()).unwrap_or(2000.0);
    let (stations, lines) = ((km * 0.24) as usize, (km * 0.03).max(3.0) as usize);
    let w = TrackWorld::new(super::bench::synth(km, stations, lines, 7));
    std::fs::write(&path, super::save::encode(&w.net)).unwrap();
}
