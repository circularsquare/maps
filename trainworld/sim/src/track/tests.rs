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
    let d = LineData { symbol: String::new(), trains: None, name: format!("L{id}"), colour: 0, stops: stops.to_vec(), path, tph, dwell_s: 30.0, turnaround_s: 180.0, cars: 0 };
    w.apply(Op::Line { id, data: Some(d) }).unwrap();
    id
}

#[test]
fn unused_blueprint_extension_preserves_running_service_but_planned_lines_preview_it() {
    for tracks in [1, 2] {
        let mut w = world();
        let a = node(&mut w, 0.0, 0.0, 0);
        let b = node(&mut w, 3000.0, 0.0, 0);
        let c = node(&mut w, 6000.0, 0.0, 0);
        let built = edge(&mut w, a, b, &[], 0).unwrap();
        let mut data = w.net.edge_data(built).unwrap();
        data.tracks = tracks;
        w.apply(Op::Edge { id: built, data: Some(data) }).unwrap();
        for n in [a, b, c] { station(&mut w, n, 200); }
        let l = line(&mut w, &[a, b], [8.0, 4.0, 2.0]);
        let before = w.svc.lines[l as usize].clone();
        let capacity = w.svc.cap.canonical();
        let trips = w.svc.trips(&w.net, 28800.0, 32400.0);
        let blueprint = w.net.alloc_edge();
        w.apply(Op::Edge { id: blueprint, data: Some(EdgeData {
            a: b, b: c, tracks: 2, pis: vec![], built: false, thru: vec![],
        }) }).unwrap();
        let check = |w: &TrackWorld| {
            let after = &w.svc.lines[l as usize];
            assert!(after.running);
            assert_eq!(before.round_trip, after.round_trip);
            assert_eq!(before.trains, after.trains);
            assert_eq!(before.delay, after.delay);
            for run in 0..2 {
                assert_eq!(before.runs[run].stop_s, after.runs[run].stop_s);
                assert_eq!(before.runs[run].free, after.runs[run].free);
                assert_eq!(before.runs[run].prof, after.runs[run].prof);
            }
            assert_eq!(capacity, w.svc.cap.canonical());
            assert_eq!(trips, w.svc.trips(&w.net, 28800.0, 32400.0));
        };
        check(&w);
        w.undo().unwrap().unwrap(); check(&w);
        w.redo().unwrap().unwrap(); check(&w);
        let planned = line(&mut w, &[b, c], [8.0, 4.0, 2.0]);
        let preview = &w.svc.lines[planned as usize];
        assert!(preview.ok && !preview.running);
        assert_eq!(preview.runs[0].stop_s[0], 0.0, "planned station B has two planned ports");
        assert_eq!(preview.runs[1].stop_s[1], preview.runs[1].len);
        check(&w);
    }
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
fn blueprint_quote_accounts_for_partial_junctions_and_flyovers() {
    let (mut w, _, _, j) = y_junction([12.0, 6.0, 3.0], true);
    let e = w.net.node_ports[j as usize].ends()[2] >> 1;
    let mut data = w.net.edge_data(e).unwrap();
    data.built = false;
    w.apply(Op::Edge { id: e, data: Some(data) }).unwrap();
    let rows = w.net.blueprint_cost_items();
    assert!(rows.chunks_exact(7).any(|r| r[0] == 2.0));
    assert!(rows.chunks_exact(7).any(|r| r[0] == 3.0));
    let quoted: f64 = rows.chunks_exact(7).map(|r| r[6]).sum();
    assert!((quoted - (w.net.total_cost() - w.net.built_cost())).abs() < 1e-8);
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

/// T-197: the player's station links survive a save, by position (ids are renumbered); a network
/// with none encodes as before, and a pair that is not two stations is neither made nor saved.
#[test]
fn station_links_save_and_load() {
    let (mut w, nodes, _, _) = simple();
    let plain = super::save::encode(&w.net);
    assert!(!plain.windows(4).any(|x| x == b"LNK1"));
    let (a, b) = (nodes[0], nodes[2]);
    assert!(w.net.is_station(a) && w.net.is_station(b));
    assert!(!w.net.link(a, a));
    assert!(w.net.link(b, a));
    assert!(w.net.link(a, b), "linking again is fine");
    assert_eq!(w.net.links, vec![[a.min(b), a.max(b)]]);
    let bytes = super::save::encode(&w.net);
    assert_eq!(&bytes[..plain.len()], &plain[..], "the block is appended at the end");
    let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
    let at = |net: &super::net::Network, n: u32| (net.node_x[n as usize], net.node_y[n as usize]);
    let got: Vec<_> = back.net.links_live().map(|[p, q]| (at(&back.net, p), at(&back.net, q))).collect();
    let want: Vec<_> = w.net.links_live().map(|[p, q]| (at(&w.net, p), at(&w.net, q))).collect();
    assert_eq!(got, want);
    assert_eq!(super::save::encode(&back.net), bytes);
    // without the block the save still loads (older saves), with no links
    let cut = bytes.windows(4).position(|x| x == b"LNK1").unwrap();
    assert!(super::save::decode(&bytes[..cut]).unwrap().links.is_empty());
    // a link to a node that is no longer a station is not saved
    w.net.node_platform[b as usize] = 0;
    assert_eq!(w.net.links_live().count(), 0);
    assert!(!super::save::encode(&w.net).windows(4).any(|x| x == b"LNK1"));
    assert!(w.net.unlink(a, b));
    assert!(!w.net.unlink(a, b));
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


#[test]
fn whole_line_counts_survive_route_changes_save_and_undo() {
    let (mut w, nodes, _, id) = simple();
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some([9, 5, 0]), tph }).unwrap();
    for lev in 0..LEVELS {
        assert_eq!(w.svc.lines[id as usize].trains[lev], [9, 5, 0][lev]);
        let implied = w.net.line_tph[id as usize][lev] as f64 * w.svc.lines[id as usize].round_trip[lev] / 3600.0;
        assert!((implied - [9, 5, 0][lev] as f64).abs() < 0.001, "{implied}");
    }
    let mut d = w.net.line_data(id).unwrap();
    d.stops = vec![nodes[0], nodes[2]];
    w.apply(Op::Line { id, data: Some(d) }).unwrap();
    assert_eq!(w.svc.lines[id as usize].trains, [9, 5, 0]);
    let bytes = super::save::encode(&w.net);
    let reloaded = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(reloaded.net.line_trains[0], Some([9, 5, 0]));
    assert_eq!(reloaded.svc.lines[0].trains, [9, 5, 0]);
    assert_eq!(super::save::encode(&reloaded.net), bytes);
    w.undo().unwrap().unwrap();
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.line_trains[id as usize], None);
    assert_eq!(w.net.line_tph[id as usize], tph);
    w.redo().unwrap().unwrap();
    assert_eq!(w.svc.lines[id as usize].trains, [9, 5, 0]);
}

#[test]
fn whole_line_counts_include_congestion_and_odd_fleets() {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 1000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    station(&mut w, a, 200); station(&mut w, b, 200);
    let first = line(&mut w, &[a, b], [8.0, 3.0, 1.0]);
    w.apply(Op::TrainCounts { id: first, trains: Some([9, 3, 0]), tph: [8.0, 3.0, 1.0] }).unwrap();
    let free_hw = w.net.line_tph[first as usize][0];
    let second = line(&mut w, &[a, b], [8.0, 3.0, 1.0]);
    w.apply(Op::TrainCounts { id: second, trains: Some([9, 3, 0]), tph: [8.0, 3.0, 1.0] }).unwrap();
    assert!(w.net.line_tph[first as usize][0] < free_hw);
    for id in [first, second] {
        let ls = &w.svc.lines[id as usize];
        assert_eq!(ls.trains, [9, 3, 0]);
        let implied = w.net.line_tph[id as usize][0] as f64 * ls.round_trip[0] / 3600.0;
        assert!((implied - 9.0).abs() < 0.001, "{implied}");
        let t = 8.0 * 3600.0 + 37.0;
        let trips = w.svc.trips(&w.net, t, t + 0.001);
        assert_eq!(trips.iter().filter(|tr| tr.line == id).count(), 9, "odd whole-line fleet at steady demand");
    }
}


#[test]
#[ignore = "native schedule edit timing"]
fn whole_line_count_edit_timing() {
    use std::time::Instant;
    for (km, stations, lines) in [(2000.0, 800, 120), (10000.0, 3500, 500)] {
        let mut w = TrackWorld::new(super::bench::synth(km, stations, lines, 7));
        let id = w.svc.lines.iter().position(|s| s.ok).unwrap() as u32;
        let mut counts = w.svc.lines[id as usize].trains;
        let tph = w.net.line_tph[id as usize];
        let start = Instant::now();
        w.apply(Op::TrainCounts { id, trains: Some(counts), tph }).unwrap();
        let conversion = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        counts[0] += 1;
        w.apply(Op::TrainCounts { id, trains: Some(counts), tph }).unwrap();
        let edit = start.elapsed().as_secs_f64() * 1000.0;
        let ls = &w.svc.lines[id as usize];
        let residual = (w.net.line_tph[id as usize][0] as f64 * ls.round_trip[0] / 3600.0 - counts[0] as f64).abs();
        assert!(residual < 0.001, "{residual}");
        println!("{km} km: legacy conversion {conversion:.2} ms; add one train {edit:.2} ms; residual {residual:.6}");
    }
}

// ---- T-206: trains join and leave at the ends of a line

/// A - B - C with a whole-line fleet of `n` at every level.
fn fleet_line(n: u32) -> (TrackWorld, u32) {
    let (mut w, _, _, id) = simple();
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some([n, n, n]), tph }).unwrap();
    (w, id)
}

/// Change the line's fleet at game time `now`, as the clock worker does.
fn set_fleet(w: &mut TrackWorld, id: u32, n: u32, now: f64) {
    let before = w.svc.before_edit(&w.net, now);
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some([n, n, n]), tph }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
}

/// Trains drawn at `t`: (run, departure).
fn drawn_at(w: &TrackWorld, t: f64) -> Vec<(u8, f64)> {
    let mut v: Vec<(u8, f64)> = w.svc.trips(&w.net, t, t + 1e-3).iter().filter(|x| x.draw <= t && x.end >= t).map(|x| (x.run, x.dep)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn a_steady_fleet_is_handed_over_without_standing() {
    let (w, id) = fleet_line(7);
    let ls = &w.svc.lines[id as usize];
    let t0 = 3.0 * 86400.0 + 7.0 * 3600.0;
    for t in w.svc.trips(&w.net, t0, t0 + 3600.0) {
        let ready = t.draw + ls.trip_len(&w.net, id, t.run as usize, t.level as usize);
        assert!(t.end - ready < 1.5, "a train stood {} s longer at a terminal mid-period", t.end - ready);
    }
    for k in 0..20 {
        assert_eq!(drawn_at(&w, t0 + 600.0 + k as f64 * 131.0).len(), 7);
    }
}

#[test]
fn changing_the_fleet_moves_no_train() {
    for (from, to) in [(6u32, 9u32), (9, 6), (6, 7), (7, 6)] {
        let (mut w, id) = fleet_line(from);
        let now = 2.0 * 86400.0 + 11.0 * 3600.0 + 417.0;
        let before = drawn_at(&w, now);
        // where each train is drawn at `t`, by (run, departure)
        let place = |w: &TrackWorld, t: f64| -> Vec<((u8, f64), (f64, f64))> {
            let ls = &w.svc.lines[id as usize];
            w.svc.trips(&w.net, t, t + 1e-3).iter().filter(|x| x.draw <= t && x.end >= t).map(|x| {
                let (s, _) = super::profile::state_at(&ls.runs[x.run as usize].prof[x.level as usize].phases, t - x.draw);
                ((x.run, x.dep), w.svc.position(&w.net, id, x.run as usize, s))
            }).collect()
        };
        let positions = place(&w, now);
        set_fleet(&mut w, id, to, now);
        // every train running a moment before the change is still there, where it was
        let after = drawn_at(&w, now + 1e-3);
        let moved = place(&w, now + 1e-3);
        for (b, p0) in &positions {
            assert!(after.contains(b), "{from}->{to}: train {b:?} vanished at the change");
            let p = moved.iter().find(|m| m.0 == *b).unwrap().1;
            assert!((p.0 - p0.0).hypot(p.1 - p0.1) < 0.5, "{from}->{to}: a train jumped {} m: {b:?} {p0:?} {p:?}", (p.0 - p0.0).hypot(p.1 - p0.1));
        }
        // and nothing appears mid-line: every trip drawn after the change but not before left a
        // terminal after it
        let rt = w.svc.lines[id as usize].round_trip[MEDIUM];
        for t in w.svc.trips(&w.net, now, now + 2.0 * rt) {
            assert!(t.dep >= now - 1e-6 || before.contains(&(t.run, t.dep)), "{from}->{to}: {t:?} appeared mid-line");
        }
        // trains drawn: never far from either fleet while it changes, the new fleet once settled
        let (lo, hi) = (from.min(to) as usize, from.max(to) as usize);
        let mut t = now;
        while t < now + 1.5 * rt {
            let n = drawn_at(&w, t).len();
            assert!(n + 1 >= lo && n <= hi + 1, "{from}->{to}: {n} trains at +{:.0} s", t - now);
            t += 37.0;
        }
        let settled = now + 3.0 * rt + 300.0;
        assert_eq!(drawn_at(&w, settled).len(), to as usize, "{from}->{to} settled");
        // and the change is forgotten once its trains are gone, leaving the plain pattern
        let trips = w.svc.trips(&w.net, settled, settled + 3600.0);
        w.svc.prune_eras(&w.net, settled + 4.0 * 3600.0);
        assert!(w.svc.eras_of(id as usize).is_empty());
        let plain = w.svc.trips(&w.net, settled, settled + 3600.0);
        let late = |v: &[super::service::Trip]| v.iter().filter(|x| x.dep > settled + rt).map(|x| (x.run, x.dep, x.end)).collect::<Vec<_>>();
        assert_eq!(late(&trips), late(&plain));
    }
}

#[test]
fn a_new_pattern_never_sends_trains_out_in_pairs() {
    // the period changes and an edit: departures from each terminal at least half a headway apart
    let (mut w, id) = fleet_line(6);
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some([9, 5, 2]), tph }).unwrap();
    let now = 86400.0 + 8.0 * 3600.0 + 1000.0;
    let before = w.svc.before_edit(&w.net, now);
    w.apply(Op::TrainCounts { id, trains: Some([12, 5, 2]), tph }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    for r in 0..2 {
        let mut d = vec![];
        w.svc.run_deps(&w.net, id as usize, r, 86400.0, 2.0 * 86400.0, &mut d);
        d.sort_by(|a, b| a.dep.total_cmp(&b.dep));
        for p in d.windows(2) {
            assert!(p[1].dep - p[0].dep >= 0.5 * p[1].hw - 1e-6, "run {r}: {:?} then {:?}", p[0], p[1]);
        }
    }
}

#[test]
fn each_train_names_the_trip_it_takes_next() {
    // T-245: a handed-over trip names its next departure (the other run's trip from its terminal),
    // no departure is taken by two trains, and a train leaving service names none
    let (mut w, id) = fleet_line(7);
    let t0 = 3.0 * 86400.0 + 7.0 * 3600.0;
    let ls = &w.svc.lines[id as usize];
    let steady = w.svc.trips(&w.net, t0, t0 + 3600.0);
    for t in &steady {
        let ready = t.draw + ls.trip_len(&w.net, id, t.run as usize, t.level as usize);
        if ready < t0 + 3000.0 {
            assert!(t.next.is_some(), "a steady fleet's train left service: {t:?}");
        }
    }
    let now = 86400.0 + 9.0 * 3600.0 + 1234.0;
    set_fleet(&mut w, id, 4, now);
    let all = w.svc.trips(&w.net, now - 3600.0, now + 3.0 * 3600.0);
    let mut taken: Vec<(u8, f64)> = vec![];
    let mut left = 0;
    for t in &all {
        match t.next {
            Some(n) => {
                let u = all.iter().find(|u| u.run != t.run && (u.draw - n).abs() < 1e-6);
                assert!(u.is_some() || n > now + 2.5 * 3600.0, "{t:?} names a departure not drawn");
                assert!((t.end - n).abs() <= 1.0 + 1e-6, "{t:?} stands until {} but leaves at {n}", t.end);
                assert!(!taken.contains(&(1 - t.run, n)), "two trains take the departure at {n}");
                taken.push((1 - t.run, n));
            }
            None => left += 1,
        }
    }
    assert!(left >= 3, "three of seven trains leave service after the cut, {left} did");
}

#[test]
fn every_window_hands_trains_over_alike() {
    let (mut w, id) = fleet_line(7);
    let now = 86400.0 + 9.0 * 3600.0 + 1234.0; // a change late in the morning peak
    set_fleet(&mut w, id, 4, now);
    let a = w.svc.trips(&w.net, now - 1800.0, now + 1800.0);
    let b = w.svc.trips(&w.net, now, now + 3600.0);
    for t in &a {
        if let Some(u) = b.iter().find(|u| u.run == t.run && u.dep == t.dep) {
            assert_eq!(t.end, u.end, "{t:?} against {u:?}");
            assert_eq!(t.next, u.next, "{t:?} against {u:?}");
        }
    }
    // a standing train always leaves on a departure from that terminal
    let ls = &w.svc.lines[id as usize];
    for t in &b {
        let ready = t.draw + ls.trip_len(&w.net, id, t.run as usize, t.level as usize);
        if t.end > ready + 1.5 {
            assert!(b.iter().chain(&a).any(|u| u.run != t.run && (u.draw - t.end).abs() < 1e-6), "{t:?} stood for nothing");
        }
    }
}
