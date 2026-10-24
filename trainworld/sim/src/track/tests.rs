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
    w.apply(Op::Edge { id, data: Some(EdgeData { a, b, tracks: 2, pis, built: true, off: false, thru: vec![], crossover: false, hsr: false }) })?;
    Ok(id)
}

pub fn station(w: &mut TrackWorld, n: u32, platform: u16) {
    w.apply(Op::Station { node: n, data: Some(StationData { platform, name: format!("S{n}"), built: true, off: false }) }).unwrap();
}

pub fn line(w: &mut TrackWorld, stops: &[u32], tph: [f32; 3]) -> u32 {
    let path = w.net.route(stops).expect("no route");
    let id = w.net.alloc_line();
    let d = LineData { symbol: String::new(), trains: None, auto: None, name: format!("L{id}"), colour: 0, stops: stops.to_vec(), path, tph, dwell_s: 30.0, turnaround_s: 180.0, cars: 0, fast: false };
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
            a: b, b: c, tracks: 2, pis: vec![], built: false, off: false, thru: vec![], crossover: false, hsr: false,
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

/// T-1146: a line saved at a frequency is given whole trains when the save loads, its rate follows
/// them, and a save in a format older than TWT4 is refused.
#[test]
fn a_frequency_line_is_given_trains_on_load() {
    let (w, _, _, l) = simple();
    let i = l as usize;
    assert!(w.net.line_trains[i].is_none());
    let want = w.svc.lines[i].trains;
    let bytes = super::save::encode(&w.net);
    let mut loaded = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert!(loaded.net.line_trains[i].is_none());
    assert!(loaded.allocate_unallocated());
    assert_eq!(loaded.net.line_trains[i], Some(want));
    assert!(!loaded.allocate_unallocated(), "only once");
    // the whole trains run at least the frequency asked, and less than one more train's worth
    for (lev, asked) in [12.0f32, 6.0, 3.0].into_iter().enumerate() {
        let (rate, trip) = (loaded.net.line_tph[i][lev], loaded.svc.lines[i].round_trip[lev]);
        assert!(rate >= asked - 1e-3 && (rate as f64) < asked as f64 + 3600.0 / trip, "level {lev}: {rate} for {asked}");
    }
    let again = super::save::encode(&loaded.net);
    assert!(super::save::decode(&again).unwrap().line_trains[i].is_some());
    let mut old = again.clone();
    old[..4].copy_from_slice(b"TWT2");
    assert!(super::save::decode(&old).is_err());
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
    let want = 4.0 * 45.0 + super::cost::crossing_cost(0, false);
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

/// T-760: two dead-end stops closer than their platforms stop halfway at most, never past each other.
#[test]
fn stops_closer_than_their_platforms_keep_their_order() {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 400.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    for n in [a, b] { station(&mut w, n, 100); }
    let l = line(&mut w, &[a, b], [8.0, 4.0, 2.0]);
    // the editor refuses such platforms; a save made elsewhere (Japan's N02 network) has them
    for n in [a, b] { w.net.node_platform[n as usize] = 600; }
    let w = TrackWorld::new(super::save::decode(&super::save::encode(&w.net)).unwrap());
    let svc = &w.svc.lines[l as usize];
    assert!(svc.ok);
    for run in 0..2 {
        let s = &svc.runs[run].stop_s;
        assert!(s[1] >= s[0], "run {run}: {s:?}");
        assert!((s[0] - svc.runs[run].len / 2.0).abs() < 1e-9 && (s[1] - s[0]).abs() < 1e-9, "both at the middle: {s:?}");
    }
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
    w.apply(Op::TrainCounts { id, trains: Some([5, 3, 0]), tph }).unwrap();
    for lev in 0..LEVELS {
        assert_eq!(w.svc.lines[id as usize].trains[lev], [5, 3, 0][lev]);
        let implied = w.net.line_tph[id as usize][lev] as f64 * w.svc.lines[id as usize].round_trip[lev] / 3600.0;
        assert!((implied - [5, 3, 0][lev] as f64).abs() < 0.001, "{implied}");
    }
    let mut d = w.net.line_data(id).unwrap();
    d.stops = vec![nodes[0], nodes[2]];
    w.apply(Op::Line { id, data: Some(d) }).unwrap();
    assert_eq!(w.svc.lines[id as usize].trains, [5, 3, 0]);
    let bytes = super::save::encode(&w.net);
    let reloaded = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(reloaded.net.line_trains[0], Some([5, 3, 0]));
    assert_eq!(reloaded.svc.lines[0].trains, [5, 3, 0]);
    assert_eq!(super::save::encode(&reloaded.net), bytes);
    w.undo().unwrap().unwrap();
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.line_trains[id as usize], None);
    assert_eq!(w.net.line_tph[id as usize], tph);
    w.redo().unwrap().unwrap();
    assert_eq!(w.svc.lines[id as usize].trains, [5, 3, 0]);
}

#[test]
fn whole_line_counts_include_congestion_and_odd_fleets() {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 10000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    station(&mut w, a, 200); station(&mut w, b, 200);
    let first = line(&mut w, &[a, b], [8.0, 3.0, 1.0]);
    w.apply(Op::TrainCounts { id: first, trains: Some([3, 1, 0]), tph: [8.0, 3.0, 1.0] }).unwrap();
    let free_hw = w.net.line_tph[first as usize][0];
    let second = line(&mut w, &[a, b], [8.0, 3.0, 1.0]);
    w.apply(Op::TrainCounts { id: second, trains: Some([3, 1, 0]), tph: [8.0, 3.0, 1.0] }).unwrap();
    assert!(w.net.line_tph[first as usize][0] < free_hw);
    for id in [first, second] {
        let ls = &w.svc.lines[id as usize];
        assert_eq!(ls.trains, [3, 1, 0]);
        let implied = w.net.line_tph[id as usize][0] as f64 * ls.round_trip[0] / 3600.0;
        assert!((implied - 3.0).abs() < 0.001, "{implied}");
        let t = 8.0 * 3600.0 + 37.0;
        let trips = w.svc.trips(&w.net, t, t + 0.001);
        assert_eq!(trips.iter().filter(|tr| tr.line == id).count(), 3, "odd whole-line fleet at steady demand");
    }
}

#[test]
fn a_count_edit_at_one_level_leaves_the_others_to_the_bit() {
    // T-415: two fleets sharing track; editing the high level must not move medium or low
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 4000.0, 0.0, 0);
    let c = node(&mut w, 10000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    edge(&mut w, b, c, &[], 0).unwrap();
    station(&mut w, a, 200); station(&mut w, b, 200); station(&mut w, c, 200);
    let first = line(&mut w, &[a, b, c], [8.0, 5.0, 2.0]);
    let second = line(&mut w, &[a, b], [8.0, 5.0, 2.0]);
    for (id, trains) in [(first, [3, 2, 1]), (second, [1, 1, 1])] {
        let tph = w.net.line_tph[id as usize];
        w.apply(Op::TrainCounts { id, trains: Some(trains), tph }).unwrap();
    }
    let figures = |w: &TrackWorld| -> Vec<(u32, u32, Vec<f64>)> {
        let mut v = vec![];
        for id in [first, second] {
            for lev in 1..LEVELS {
                for r in 0..2 {
                    let mut f = w.svc.lines[id as usize].stop_to_stop(r, lev);
                    f.push(w.svc.lines[id as usize].round_trip[lev]);
                    v.push((w.net.line_tph[id as usize][lev].to_bits(), lev as u32, f));
                }
            }
        }
        v
    };
    for n in [4, 3, 5, 2] {
        let before = figures(&w);
        let high = w.net.line_tph[first as usize][0];
        let tph = w.net.line_tph[first as usize];
        w.apply(Op::TrainCounts { id: first, trains: Some([n, 2, 1]), tph }).unwrap();
        assert_ne!(w.net.line_tph[first as usize][0], high);
        let after = figures(&w);
        for (x, y) in before.iter().zip(&after) {
            assert_eq!(x.0, y.0, "level {} trains an hour moved", x.1);
            assert!(x.2.iter().zip(&y.2).all(|(p, q)| p.to_bits() == q.to_bits()), "level {} times moved", x.1);
        }
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

/// Seven stations 2 km apart (room for 14 trains, T-517) with a whole-line fleet of `n` at every level.
fn fleet_line(n: u32) -> (TrackWorld, u32) {
    let (mut w, ns) = corridor(7);
    let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
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

// ---- T-517: no track carries more trains than its capacity

/// `n` stations 2 km apart on double track.
fn corridor(n: usize) -> (TrackWorld, Vec<u32>) {
    let mut w = world();
    let ns: Vec<u32> = (0..n).map(|i| node(&mut w, i as f64 * 2000.0, 0.0, 0)).collect();
    for i in 0..n - 1 {
        edge(&mut w, ns[i], ns[i + 1], &[], 0).unwrap();
    }
    for &n in &ns {
        station(&mut w, n, 200);
    }
    (w, ns)
}

fn set_counts(w: &mut TrackWorld, id: u32, trains: [u32; 3]) {
    let tph = w.net.line_tph[id as usize];
    w.apply(Op::TrainCounts { id, trains: Some(trains), tph }).unwrap();
}

/// The fullest limiting resource at a level.
fn fullest(w: &TrackWorld, lev: usize) -> f64 {
    (0..w.svc.cap.res.len()).filter(|&r| w.svc.cap.limits(r)).map(|r| w.svc.cap.rho[r][lev]).fold(0.0, f64::max)
}

#[test]
fn a_fleet_beyond_capacity_waits_at_the_ends() {
    let (mut w, ns) = corridor(5);
    let id = line(&mut w, &ns, [8.0, 3.0, 1.0]);
    set_counts(&mut w, id, [20, 3, 0]);
    let (rate, trip) = (w.net.line_tph[id as usize][HIGH], w.svc.lines[id as usize].round_trip[HIGH]);
    assert!((fullest(&w, HIGH) - 1.0).abs() < 1e-4, "{}", fullest(&w, HIGH));
    assert!(rate as f64 * trip / 3600.0 < 19.0, "the fleet is held by the track");
    for n in [30, 300, 3000] {
        set_counts(&mut w, id, [n, 3, 0]);
        assert_eq!(w.net.line_tph[id as usize][HIGH].to_bits(), rate.to_bits(), "{n} trains");
        assert_eq!(w.svc.lines[id as usize].round_trip[HIGH], trip, "surplus trains add no delay");
        assert_eq!(w.svc.lines[id as usize].trains[HIGH], n);
    }
    // a fleet that fits runs at its own rate again
    set_counts(&mut w, id, [4, 3, 0]);
    let ls = &w.svc.lines[id as usize];
    assert!((w.net.line_tph[id as usize][HIGH] as f64 * ls.round_trip[HIGH] / 3600.0 - 4.0).abs() < 1e-3);
    assert!(fullest(&w, HIGH) < 0.9);
}

#[test]
fn shared_track_splits_its_capacity() {
    let (mut w, ns) = corridor(5);
    let long = line(&mut w, &ns, [8.0, 3.0, 1.0]);
    let short = line(&mut w, &ns[1..4], [8.0, 3.0, 1.0]);
    set_counts(&mut w, long, [40, 4, 2]);
    set_counts(&mut w, short, [40, 4, 2]);
    assert!((fullest(&w, HIGH) - 1.0).abs() < 1e-4, "{}", fullest(&w, HIGH));
    let rates = |w: &TrackWorld| [long, short].map(|id| w.net.line_tph[id as usize]);
    let high = rates(&w);
    assert!(high[0][HIGH] > 0.0 && high[1][HIGH] > 0.0, "{high:?}");
    // editing another level leaves the held one to the bit
    set_counts(&mut w, short, [40, 5, 2]);
    assert_eq!(rates(&w).map(|r| r[HIGH].to_bits()), high.map(|r| r[HIGH].to_bits()));
    // fewer trains on one line leave room for the other
    set_counts(&mut w, long, [3, 4, 2]);
    assert!(w.net.line_tph[short as usize][HIGH] > high[1][HIGH]);
    assert!(fullest(&w, HIGH) <= 1.0 + 1e-4);
}

/// T-797: X - T - J - M, and a branch J - Br leaving towards M. From T, Br is reached only by
/// turning back (at T, a through station, or at X, a dead end).
pub fn reversal_fixture() -> (TrackWorld, [u32; 5]) {
    reversal_fixture_with(true)
}

/// `reversal_fixture`, with X a station or a bare track end (T-797 B3: a headshunt).
pub fn reversal_fixture_with(x_station: bool) -> (TrackWorld, [u32; 5]) {
    let mut w = world();
    let x = node(&mut w, -3000.0, 0.0, 0);
    let t = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let m = node(&mut w, 6000.0, 0.0, 0);
    let br = node(&mut w, 5000.0, 1500.0, 0);
    edge(&mut w, x, t, &[], 0).unwrap();
    edge(&mut w, t, j, &[], 0).unwrap();
    edge(&mut w, j, m, &[], 0).unwrap();
    edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
    for n in [x, t, m, br] {
        if n != x || x_station {
            station(&mut w, n, 200);
        }
    }
    (w, [x, t, j, m, br])
}

#[test]
fn a_line_reversing_at_a_mid_stop_runs() {
    let (mut w, [_, t, _, m, br]) = reversal_fixture();
    let l = line(&mut w, &[br, t, m], [6.0, 3.0, 0.0]);
    assert!(w.net.check_line(l).is_ok());
    assert_eq!(w.net.path_reversals(w.net.line_path(l)).len(), 1);
    let ls = &w.svc.lines[l as usize];
    assert!(ls.ok && ls.running);
    for r in 0..2 {
        let run = &ls.runs[r];
        assert_eq!(run.turn, vec![false, true, false]);
        assert!((run.jump[1] - ls.train_len).abs() < 1e-9);
        // It stands the turnaround, not the dwell, half a train short of T on the side it came
        // in on, and leaves from the same place.
        let f = &run.free;
        assert!((f.dep[1] - f.arr[1] - 180.0).abs() < 1e-9, "{:?} {:?}", f.arr, f.dep);
        let (s, v) = super::profile::state_at(&f.phases, f.arr[1] + 10.0);
        assert!((s - (run.stop_s[1] + run.jump[1])).abs() < 1e-6 && v == 0.0);
        let a = w.svc.position(&w.net, l, r, run.stop_s[1]);
        let b = w.svc.position(&w.net, l, r, run.stop_s[1] + run.jump[1]);
        assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.5, "{a:?} {b:?}");
        assert!((a.0 - ls.train_len / 2.0).abs() < 0.5 && a.1.abs() < 1e-6, "{a:?}");
        // Offsets only move forward.
        assert!(f.phases.windows(2).all(|p| p[1].s >= p[0].s && p[1].t >= p[0].t));
        assert!(f.arr[2] > f.dep[1]);
    }
    let rt = ls.round_trip[HIGH];
    assert!((rt - (ls.runs[0].prof[HIGH].duration + ls.runs[1].prof[HIGH].duration + 360.0)).abs() < 1e-9);
    assert!(ls.runs[0].prof[HIGH].duration > ls.runs[0].free.arr[1] + 180.0);
    // Trains are drawn through the hour.
    let trips = w.svc.trips(&w.net, 8.0 * 3600.0, 9.0 * 3600.0);
    assert_eq!(trips.iter().filter(|x| x.line == l && x.dep >= 8.0 * 3600.0).count(), 12);
    // A path that turns back at a station it does not stop at is refused.
    let mut d = w.net.line_data(l).unwrap();
    d.stops = vec![br, m];
    assert!(w.apply(Op::Line { id: l, data: Some(d) }).is_err());
}

#[test]
fn a_station_stopped_at_twice_around_a_dead_end_reversal() {
    let (mut w, [x, t, _, m, br]) = reversal_fixture();
    let l = line(&mut w, &[br, t, x, t, m], [6.0, 3.0, 0.0]);
    assert!(w.net.check_line(l).is_ok());
    let ls = &w.svc.lines[l as usize];
    assert!(ls.ok);
    for r in 0..2 {
        let run = &ls.runs[r];
        assert_eq!(run.stop_idx, vec![0, 2, 3, 4, 6]);
        assert_eq!(run.turn, vec![false, false, true, false, false]);
        // At a dead end the train stands at the platform's middle.
        assert!((run.jump[2] - 200.0).abs() < 1e-9);
        let a = w.svc.position(&w.net, l, r, run.stop_s[2]);
        assert!((a.0 + 2900.0).abs() < 0.5, "{a:?}");
        let f = &run.free;
        assert!(f.arr.windows(2).all(|p| p[1] > p[0]));
        assert!((f.dep[2] - f.arr[2] - 180.0).abs() < 1e-9);
        assert!((f.dep[1] - f.arr[1] - 30.0).abs() < 1e-9 && (f.dep[3] - f.arr[3] - 30.0).abs() < 1e-9);
    }
    // T is a platform stop twice a run, at the dwell; X is a pooled terminus with the limiter.
    let cap = &w.svc.cap;
    let at_t: usize = (0..cap.res.len()).filter(|&r| cap.res[r].node == t && cap.res[r].kind == ResKind::Platform)
        .map(|r| cap.users_of(r).iter().filter(|u| u.line == l && u.stop >= 0).count()).sum();
    assert_eq!(at_t, 4);
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == x && cap.wire_kind(r) == 9));
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == x && cap.res[r].kind == ResKind::Terminus && cap.wire_kind(r) == ResKind::Terminus as u8));
    // matching skips an earlier pass of a stop when a reversal needs it later
    let nodes = [1, 2, 1, 3, 1, 4];
    assert_eq!(super::net::match_stops(&nodes, &[1, 3, 1, 4], &[]), Some(vec![0, 3, 4, 5]));
    assert_eq!(super::net::match_stops(&nodes, &[1, 1, 4], &[2]), Some(vec![0, 2, 5]));
    assert_eq!(super::net::match_stops(&nodes, &[1, 1, 4], &[3]), None);
    assert_eq!(super::net::match_stops(&[5, 1, 6, 1, 7], &[5, 1, 7], &[3]), Some(vec![0, 3, 4]));
}

/// T-797: P - S, and from each a long way round to K and on to the dead end Q. From S, Q is
/// reached forward over the east loop, or a little shorter by turning back at S over P.
#[test]
fn a_route_without_a_reversal_wins_when_similar() {
    let mut w = world();
    let p = node(&mut w, -3000.0, 0.0, 0);
    let s = node(&mut w, 0.0, 0.0, 0);
    let k = node(&mut w, 0.0, 6000.0, 0);
    let q = node(&mut w, 0.0, 9000.0, 0);
    edge(&mut w, p, s, &[], 0).unwrap();
    edge(&mut w, p, k, &[(-6000.0, 0.0), (-6000.0, 3000.0), (0.0, 3000.0)], 0).unwrap();
    edge(&mut w, s, k, &[(7000.0, 0.0), (7000.0, 4500.0), (0.0, 4500.0)], 0).unwrap();
    edge(&mut w, k, q, &[], 0).unwrap();
    for n in [p, s, q] {
        station(&mut w, n, 200);
    }
    let len = |w: &TrackWorld, path: &[u32]| path.iter().map(|&x| w.net.edge_len[(x >> 1) as usize]).sum::<f64>();
    let path = w.net.route(&[p, s, q]).unwrap();
    assert!(w.net.path_reversals(&path).is_empty());
    let back = 6000.0 + len(&w, &w.net.route(&[p, q]).unwrap());
    let fwd = len(&w, &path);
    assert!(fwd > back && fwd < back + 3000.0, "{fwd} vs {back}");
    // A stop on the far side of the loop only by turning back is still found.
    let l = line(&mut w, &[p, s, q], [4.0, 2.0, 0.0]);
    assert!(w.svc.lines[l as usize].ok);
}

#[test]
fn reversing_lines_save_round_trip() {
    let (mut w, [x, t, j, m, br]) = reversal_fixture();
    let a = line(&mut w, &[br, t, m], [6.0, 3.0, 0.0]);
    let b = line(&mut w, &[br, t, x, t, m], [4.0, 2.0, 0.0]);
    let bytes = super::save::encode(&w.net);
    let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(super::save::encode(&back.net), bytes);
    for l in [a, b] {
        assert_eq!(back.net.line_path(l), w.net.line_path(l));
        assert_eq!(back.svc.lines[l as usize].round_trip, w.svc.lines[l as usize].round_trip);
        assert_eq!(back.svc.lines[l as usize].runs[0].turn, w.svc.lines[l as usize].runs[0].turn);
    }
    let _ = (j, &mut w);
}

/// T-797 B3: with X a bare track end (a headshunt), Br to M runs out to X and back, passing T
/// twice without stopping; it stands the turnaround there, as at a stop.
#[test]
fn a_line_reversing_at_a_bare_track_end_runs() {
    let (mut w, [x, t, _, m, br]) = reversal_fixture_with(false);
    let l = line(&mut w, &[br, m], [3.0, 6.0, 8.0]);
    assert!(w.net.check_line(l).is_ok());
    let path = w.net.line_path(l).to_vec();
    assert_eq!(w.net.path_reversals(&path).len(), 1);
    assert!(w.net.stop_reversals(&path).is_empty());
    assert_eq!(w.net.turn_count(w.net.line_stops(l), &path), 1, "a reroute may not add one");
    let ls = &w.svc.lines[l as usize];
    assert!(ls.ok && ls.running);
    for r in 0..2 {
        let run = &ls.runs[r];
        assert_eq!(run.turn, vec![false, false]);
        assert_eq!(run.nodes.iter().filter(|&&n| n == t).count(), 2);
        assert_eq!(run.ends.len(), 1);
        let (k, s, jump) = run.ends[0];
        assert_eq!(run.nodes[k as usize], x);
        assert!((jump - (ls.train_len + 2.0 * BARE_END_CLEAR_M)).abs() < 1e-9);
        let f = &run.free;
        assert_eq!((f.arr.len(), f.dep.len()), (2, 2), "the track end is no stop");
        // a stand of the turnaround, at the far side of the jump
        let ph = &f.phases;
        let i = ph.iter().position(|p| (p.s - (s + jump)).abs() < 1e-6 && p.v == 0.0 && p.a == 0.0).expect("a stand at the end");
        assert!((ph[i + 1].t - ph[i].t - 180.0).abs() < 1e-9, "{:?}", &ph[i..i + 2]);
        assert!(ph.windows(2).all(|p| p[1].s >= p[0].s && p[1].t >= p[0].t));
        // the same place on the ground both ways, the train's tail 20 m short of the end
        let a = w.svc.position(&w.net, l, r, s);
        let b = w.svc.position(&w.net, l, r, s + jump);
        assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.5, "{a:?} {b:?}");
        assert!((a.0 - (-3000.0 + ls.train_len / 2.0 + BARE_END_CLEAR_M)).abs() < 0.5 && a.1.abs() < 1e-6, "{a:?}");
    }
    let rt = ls.round_trip[HIGH];
    assert!((rt - (ls.runs[0].prof[HIGH].duration + ls.runs[1].prof[HIGH].duration + 360.0)).abs() < 1e-9);
    // X is a one-track terminus with the limiter: 12 reversals an hour, both directions from one side
    let cap = &w.svc.cap;
    let lim: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) == 9).collect();
    assert_eq!(lim.len(), 1);
    let r = lim[0];
    assert_eq!((cap.res[r].node, cap.users_of(r).len()), (x, 2));
    assert!((cap.rho[r][MEDIUM] - 1.0).abs() < 1e-12, "12 an hour fill it");
    assert!((cap.fill(&w.net, &[(l, 10.0)], HIGH)[0] - 6.0).abs() < 1e-9);
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == x && cap.res[r].kind == ResKind::Terminus && cap.wire_kind(r) == ResKind::Terminus as u8));
    assert!(!(0..cap.res.len()).any(|r| cap.res[r].node == t && cap.res[r].kind == ResKind::Platform && cap.users_of(r).iter().any(|u| u.stop >= 0)));
    // more trains than the end takes are held at the ends of the line
    set_counts(&mut w, l, [40, 3, 0]);
    assert!(w.net.line_tph[l as usize][HIGH] <= 6.0 + 1e-6, "{:?}", w.net.line_tph[l as usize]);
    // saves round trip
    let bytes = super::save::encode(&w.net);
    let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(super::save::encode(&back.net), bytes);
    assert_eq!(back.net.line_path(l), w.net.line_path(l));
    assert_eq!(back.svc.lines[l as usize].round_trip, w.svc.lines[l as usize].round_trip);
    assert_eq!(back.svc.lines[l as usize].runs[0].ends, w.svc.lines[l as usize].runs[0].ends);
    // a station on the stub is a stop like any other: X a station, the same path needs it as one
    let mut d = w.net.line_data(l).unwrap();
    station(&mut w, x, 200);
    assert!(w.net.check_line(l).is_err());
    d.stops = vec![br, x, m];
    w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    assert!(w.svc.lines[l as usize].runs[0].ends.is_empty());
}

// ---- T-835: a departure belongs to the period in force when it reaches the line's middle

/// Departures of one run in `[a, b)`, sorted.
fn deps_of(w: &TrackWorld, id: u32, r: usize, a: f64, b: f64) -> Vec<super::service::Trip> {
    let mut d = vec![];
    w.svc.run_deps(&w.net, id as usize, r, a, b, &mut d);
    d.sort_by(|x, y| x.dep.total_cmp(&y.dep));
    d
}

/// Departures a day the periods ask for: trains an hour times each period's hours, per run.
fn day_count(tph: [f32; 3]) -> f64 {
    PERIOD_HOURS.iter().zip(PERIOD_LEVEL).map(|(&(a, b), lev)| tph[lev] as f64 * (b - a)).sum()
}

#[test]
fn the_middle_of_a_long_line_changes_period_on_time() {
    let (mut w, ns) = corridor(31);
    let id = line(&mut w, &ns, [12.0, 6.0, 3.0]);
    let ls = &w.svc.lines[id as usize];
    let lat = w.svc.lattice(&w.net, id as usize).unwrap();
    let run = 0.5 * (ls.runs[0].prof[MEDIUM].duration + ls.runs[1].prof[MEDIUM].duration);
    assert!(run > 1800.0, "a long line: {run} s");
    assert!((lat.lead - run / 2.0).abs() < 1e-9);
    let day = 2.0 * 86400.0;
    let k = ns.len() / 2;
    for r in 0..2 {
        let deps = deps_of(&w, id, r, day - 86400.0, day + 2.0 * 86400.0);
        // the busier pattern leaves each end half a run before 06:00
        let first = deps.iter().find(|t| t.dep >= day + 3.0 * 3600.0 && t.level as usize == HIGH).unwrap();
        assert!((first.dep - (day + 6.0 * 3600.0 - lat.lead)).abs() < 300.0 + 1e-6, "{first:?}");
        // trains passing the middle stop: the period's headway from its start, at 06:00 and 16:00
        let pass: Vec<(f64, usize)> = deps.iter().map(|t| (t.dep + ls.runs[r].prof[t.level as usize].dep[k], t.level as usize)).collect();
        for (h0, h1) in [(6.0, 10.0), (16.0, 20.0)] {
            let (s, e) = (day + h0 * 3600.0, day + h1 * 3600.0);
            let hw = 3600.0 / 12.0;
            let mut times: Vec<f64> = pass.iter().filter(|p| p.0 >= s - hw && p.0 < e).map(|p| p.0).collect();
            times.sort_by(f64::total_cmp);
            assert!(times.iter().any(|&t| t >= s && t <= s + hw + 60.0), "run {r}: no train at the middle within a headway of {h0}:00");
            for g in times.windows(2).filter(|g| g[1] >= s) {
                assert!(g[1] - g[0] <= hw + 60.0, "run {r}: a gap of {} s at the middle after {h0}:00", g[1] - g[0]);
            }
            // and the trains passing then are the peak's
            assert!(pass.iter().filter(|p| p.0 >= s + 60.0 && p.0 < e - 60.0).all(|p| p.1 == HIGH), "run {r}: a quieter train mid-peak at the middle");
        }
        // a day's departures are the periods' counts, as before: the same car-km
        let n = deps.iter().filter(|t| t.dep >= day && t.dep < day + 86400.0).count() as f64;
        assert!((n - day_count(w.net.line_tph[id as usize])).abs() <= 1.0, "run {r}: {n} departures a day");
    }
}

#[test]
fn sparse_and_reversing_lines_lead_too() {
    // two trains a period on a long line with no night service, and a line reversing at a mid stop
    let (mut w, ns) = corridor(31);
    let sparse = line(&mut w, &ns, [0.5, 0.25, 0.0]);
    let (mut v, [_, t, _, m, br]) = reversal_fixture();
    let rev = line(&mut v, &[br, t, m], [6.0, 3.0, 0.0]);
    let day = 2.0 * 86400.0;
    for (w, id) in [(&w, sparse), (&v, rev)] {
        let lat = w.svc.lattice(&w.net, id as usize).unwrap();
        assert!(lat.lead > 0.0);
        let tph = w.net.line_tph[id as usize];
        for r in 0..2 {
            let deps = deps_of(w, id, r, day - 86400.0, day + 2.0 * 86400.0);
            let today: Vec<_> = deps.iter().filter(|t| t.dep >= day && t.dep < day + 86400.0).collect();
            // a whole number of departures a period, from its first; a period followed by no service
            // runs to its own end, so it gains the lead
            let want: f64 = PERIOD_HOURS.iter().zip(PERIOD_LEVEL).map(|(&(a, b), lev)| {
                let q = PERIOD_HOURS.iter().position(|&(s, _)| s == b % 24.0).unwrap();
                let len = (b - a) * 3600.0 + if tph[PERIOD_LEVEL[q]] <= 0.0 { lat.lead } else { 0.0 };
                (len * tph[lev] as f64 / 3600.0 - 1e-9).ceil().max(0.0)
            }).sum();
            assert!((today.len() as f64 - want).abs() <= 1.0, "line {id} run {r}: {} departures, want {want}", today.len());
            // the day's first train leaves half a run before 06:00; the evening runs to midnight
            assert!((today[0].dep - (day + 6.0 * 3600.0 - lat.lead)).abs() < 1.0, "line {id} run {r}: {:?}", today[0]);
            let last = today.last().unwrap();
            let hw = 3600.0 / tph[MEDIUM] as f64;
            assert!(last.dep > day + 86400.0 - hw - 1.0 && last.dep < day + 86400.0, "line {id} run {r}: {last:?}");
            for p in deps.windows(2) {
                assert!(p[1].dep - p[0].dep >= 0.5 * p[1].hw - 1e-6, "line {id} run {r}: {:?} then {:?}", p[0], p[1]);
            }
        }
        // every window draws the same trips
        let t0 = day + 6.0 * 3600.0 - lat.lead;
        let a = w.svc.trips(&w.net, t0 - 1800.0, t0 + 1800.0);
        let b = w.svc.trips(&w.net, t0, t0 + 3600.0);
        for x in a.iter().filter(|x| x.line == id) {
            if let Some(y) = b.iter().find(|y| y.line == id && y.run == x.run && y.dep == x.dep) {
                assert_eq!((x.end, x.next), (y.end, y.next));
            }
        }
    }
}

// ---- T-817: high-speed track and trains

/// A - B - C in a straight line, 40 km each, stations at A and C only; `hsr` per edge, then a line
/// A to C with high-speed trains or not.
fn hsr_world(hsr: [bool; 2], fast: bool) -> (TrackWorld, u32) {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 40000.0, 0.0, 0);
    let c = node(&mut w, 80000.0, 0.0, 0);
    for (k, (p, q)) in [(a, b), (b, c)].into_iter().enumerate() {
        let id = w.net.alloc_edge();
        w.apply(Op::Edge { id, data: Some(EdgeData { a: p, b: q, tracks: 2, pis: vec![], built: true, off: false, thru: vec![], crossover: false, hsr: hsr[k] }) }).unwrap();
    }
    station(&mut w, a, 200);
    station(&mut w, c, 200);
    let l = line(&mut w, &[a, c], [4.0, 2.0, 1.0]);
    let mut d = w.net.line_data(l).unwrap();
    d.fast = fast;
    w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    (w, l)
}

fn top_kmh(w: &TrackWorld, l: u32, s0: f64, s1: f64) -> f64 {
    super::profile::v_max(&w.svc.lines[l as usize].runs[0].free.phases, s0, s1) * 3.6
}

#[test]
fn high_speed_trains_reach_320_only_on_high_speed_track() {
    let (w, l) = hsr_world([true, true], true);
    assert!((top_kmh(&w, l, 0.0, 80000.0) - 320.0).abs() < 1e-6);
    let fast = w.svc.lines[l as usize].runs[0].free.duration;
    // ordinary trains on high-speed track: 160
    let (w, l) = hsr_world([true, true], false);
    assert!((top_kmh(&w, l, 0.0, 80000.0) - 160.0).abs() < 1e-6);
    let slow = w.svc.lines[l as usize].runs[0].free.duration;
    assert!(fast < slow * 0.6, "{fast} against {slow}");
    // high-speed trains on ordinary track: 160, and the same run as ordinary trains
    let (w, l) = hsr_world([false, false], true);
    assert!((top_kmh(&w, l, 0.0, 80000.0) - 160.0).abs() < 1e-6);
    assert_eq!(w.svc.lines[l as usize].runs[0].free.duration, slow);
}

#[test]
fn a_mixed_line_runs_each_stretch_at_its_own_limit() {
    let (w, l) = hsr_world([true, false], true);
    assert!((top_kmh(&w, l, 1000.0, 30000.0) - 320.0).abs() < 1e-6);
    // past the joint at 40 km, with the train's half length, it is down to 160
    assert!((top_kmh(&w, l, 40200.0, 79000.0) - 160.0).abs() < 1e-6);
    let lim = &w.svc.lines[l as usize].runs[0].limits;
    assert!(lim.iter().any(|x| (x[1] - V_HSR).abs() < 1e-9) && lim.iter().any(|x| (x[1] - V_TOP).abs() < 1e-9));
    // the other way round too
    let (w, l) = hsr_world([false, true], true);
    assert!((top_kmh(&w, l, 1000.0, 39000.0) - 160.0).abs() < 1e-6);
    assert!((top_kmh(&w, l, 50000.0, 70000.0) - 320.0).abs() < 1e-6);
}

#[test]
fn curves_cap_high_speed_trains_below_the_track_cap() {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let c = node(&mut w, 80000.0, 8000.0, 0);
    let id = w.net.alloc_edge();
    // one bend of 3,000 m radius in the middle: sqrt(1.1 x 3000) = 57.4 m/s, 207 km/h
    let pis = vec![Pi::new(40000.0, 0.0, 3000.0, 0)];
    w.apply(Op::Edge { id, data: Some(EdgeData { a, b: c, tracks: 2, pis, built: true, off: false, thru: vec![], crossover: false, hsr: true }) }).unwrap();
    station(&mut w, a, 200);
    station(&mut w, c, 200);
    let l = line(&mut w, &[a, c], [4.0, 2.0, 1.0]);
    let mut d = w.net.line_data(l).unwrap();
    d.fast = true;
    w.apply(Op::Line { id: l, data: Some(d) }).unwrap();
    let run = &w.svc.lines[l as usize].runs[0];
    let curve = (A_LAT * 3000.0f64).sqrt();
    assert!(run.limits.iter().any(|x| (x[1] - curve).abs() < 1e-6), "{:?}", run.limits);
    assert!((top_kmh(&w, l, 0.0, 30000.0) - 320.0).abs() < 1e-6);
    let mid = run.len / 2.0;
    assert!(super::profile::speed_at(&run.free.phases, mid) <= curve + 1e-6);
}

#[test]
fn high_speed_acceleration_takes_the_third_band() {
    use super::profile::{run, time_at, speed_at};
    let p = run(&[[0.0, V_HSR]], &[0.0, 60000.0], &[], &[0.0, 0.0], &[]);
    // 0-60 at 1.0, 60-200 at 0.5, 200-320 at 0.3 (notes/T-817.md): 206 s and 10.97 km
    let (v1, v2, v3) = (ACCEL_SWITCH_V, ACCEL_TOP_V, V_HSR);
    let d = v1 * v1 / 2.0 + (v2 * v2 - v1 * v1) / 1.0 + (v3 * v3 - v2 * v2) / 0.6;
    let t = v1 + (v2 - v1) / 0.5 + (v3 - v2) / 0.3;
    assert!((d - 10972.0).abs() < 1.0 && (t - 205.6).abs() < 0.1, "{d} {t}");
    assert!((speed_at(&p.phases, d) - V_HSR).abs() < 1e-6);
    assert!((time_at(&p.phases, d) - t).abs() < 1e-6);
    assert!(speed_at(&p.phases, d - 100.0) < V_HSR - 0.1);
    // the time is exact: d / v at 320 for the rest, braking at 0.9
    let brake = V_HSR * V_HSR / (2.0 * BRAKE);
    let want = t + (60000.0 - d - brake) / V_HSR + V_HSR / BRAKE;
    assert!((p.duration - want).abs() < 1e-6, "{} {want}", p.duration);
    // an ordinary train never leaves the second band: 0-160 unchanged
    let q = run(&[[0.0, V_TOP]], &[0.0, 60000.0], &[], &[0.0, 0.0], &[]);
    assert!(q.phases.iter().all(|ph| ph.a != ACCEL_TOP));
}

#[test]
fn a_320_section_takes_16_trains_an_hour() {
    // T-517's headway: 70 s plus v / 0.6
    let hw = SECTION_BASE_S + V_HSR / SIGNAL_BRAKE;
    assert_eq!((3600.0 / hw).floor(), 16.0);
    assert_eq!((3600.0 / (SECTION_BASE_S + V_TOP / SIGNAL_BRAKE)).floor(), 24.0);
}

// ---- T-798: loop lines

/// Six stations round a circle of 3 km, double track between them, each edge bent at the meeting
/// of the circle's tangents, so a train carries straight on through every station.
pub fn ring_fixture() -> (TrackWorld, Vec<u32>) {
    let mut w = world();
    let r = 3000.0;
    let ns: Vec<u32> = (0..6).map(|i| {
        let a = i as f64 * std::f64::consts::PI / 3.0;
        node(&mut w, r * a.cos(), r * a.sin(), 0)
    }).collect();
    let pi = r / (std::f64::consts::PI / 6.0).cos();
    for i in 0..6 {
        let a = (i as f64 + 0.5) * std::f64::consts::PI / 3.0;
        edge(&mut w, ns[i], ns[(i + 1) % 6], &[(pi * a.cos(), pi * a.sin())], 0).unwrap();
    }
    for &n in &ns {
        station(&mut w, n, 200);
    }
    (w, ns)
}

fn closed(ns: &[u32]) -> Vec<u32> {
    let mut v = ns.to_vec();
    v.push(ns[0]);
    v
}

#[test]
fn a_circle_runs_both_ways_with_its_own_trains() {
    use super::net::LoopKind;
    let (mut w, ns) = ring_fixture();
    let id = line(&mut w, &closed(&ns), [12.0, 6.0, 3.0]);
    assert!(w.net.check_line(id).is_ok());
    assert_eq!(w.net.line_loop(id), LoopKind::Circle);
    assert!(w.net.path_reversals(w.net.line_path(id)).is_empty());
    let ls = &w.svc.lines[id as usize];
    assert!(ls.ok && ls.running && ls.kind == LoopKind::Circle);
    for r in 0..2 {
        let run = &ls.runs[r];
        assert_eq!(run.stop_idx, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(run.nodes[0], run.nodes[6]);
        assert!(run.turn.iter().all(|&t| !t));
    }
    // run 1 goes the other way round
    assert_eq!(ls.runs[1].nodes[1], ns[5]);
    // a lap is a run and the dwell at the first stop; each way has its own trains
    let dwell = w.net.line_dwell[id as usize] as f64;
    for lev in 0..LEVELS {
        let lap = |r: usize| ls.runs[r].prof[lev].duration + dwell;
        assert!((ls.round_trip[lev] - 2.0 * lap(0).max(lap(1))).abs() < 1e-9);
        assert!((ls.trip_len(&w.net, id, 0, lev) - lap(0)).abs() < 1e-9);
        let each = (lap(0).max(lap(1)) * w.net.line_tph[id as usize][lev] as f64 / 3600.0 - 1e-9).ceil() as u32;
        assert_eq!(ls.trains[lev], 2 * each, "level {lev}");
    }
    // no turnaround anywhere: the first stop is a platform stop for both runs
    let cap = &w.svc.cap;
    assert!(cap.res.iter().all(|r| r.kind != ResKind::Terminus), "no terminus on a circle");
    let at_a: usize = (0..cap.res.len()).filter(|&r| cap.res[r].node == ns[0] && cap.res[r].kind == ResKind::Platform).map(|r| cap.users_of(r).len()).sum();
    assert_eq!(at_a, 2, "each run stops at the first stop once");
    // the middle is half a lap from the first stop
    let lat = w.svc.lattice(&w.net, id as usize).unwrap();
    assert!((lat.lead - 0.25 * (ls.runs[0].prof[MEDIUM].duration + ls.runs[1].prof[MEDIUM].duration)).abs() < 1e-9);
    assert_eq!(lat.off, [0.0; 3]);
}

#[test]
fn a_circle_fleet_is_even_and_each_train_carries_on_round() {
    let (mut w, ns) = ring_fixture();
    let id = line(&mut w, &closed(&ns), [12.0, 6.0, 3.0]);
    set_counts(&mut w, id, [7, 4, 2]);
    assert_eq!(w.net.line_trains[id as usize], Some([8, 4, 2]), "an odd count rounds up");
    let ls = &w.svc.lines[id as usize];
    assert_eq!(ls.trains, [8, 4, 2]);
    // four trains each way: the headway is the longer lap over four
    let lap = ls.round_trip[HIGH] / 2.0;
    assert!((3600.0 / w.net.line_tph[id as usize][HIGH] as f64 - lap / 4.0).abs() < 1.0);
    // in steady service a train finishing a lap takes its own run's next departure, standing at
    // the first stop under a headway
    let day = 2.0 * 86400.0;
    let trips = w.svc.trips(&w.net, day + 7.0 * 3600.0, day + 9.0 * 3600.0);
    let mid: Vec<_> = trips.iter().filter(|t| t.line == id && t.dep > day + 7.0 * 3600.0 && t.dep < day + 8.0 * 3600.0).collect();
    assert!(mid.len() >= 8);
    for t in mid {
        let n = t.next.expect("every train carries on");
        assert!(trips.iter().any(|u| u.line == id && u.draw == n && u.run == t.run), "{t:?} goes on round the same way");
        let ready = t.draw + ls.trip_len(&w.net, id, t.run as usize, t.level as usize);
        assert!(n >= ready - 1.0 && n - ready < t.hw, "{t:?}: stands {} s", n - ready);
    }
    // headway mode keeps the count even
    w.apply(Op::AutoTrains { id, auto: Some([300, 600, 1200]), trains: Some([8, 4, 2]) }).unwrap();
    let ls = &w.svc.lines[id as usize];
    for lev in 0..LEVELS {
        let c = w.net.line_trains[id as usize].unwrap()[lev];
        let h = [300.0, 600.0, 1200.0][lev];
        assert_eq!(c % 2, 0, "level {lev}");
        assert_eq!(c, 2 * super::world::auto_trains(ls.round_trip[lev] / 2.0, h as u32), "level {lev}");
    }
}

/// A dead-end station A, a stop M on the stem, and a loop at the junction J through K.
pub fn balloon_fixture() -> (TrackWorld, [u32; 4]) {
    let mut w = world();
    let a = node(&mut w, -3000.0, 0.0, 0);
    let m = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let k = node(&mut w, 5000.0, 1500.0, 0);
    edge(&mut w, a, m, &[], 0).unwrap();
    edge(&mut w, m, j, &[], 0).unwrap();
    edge(&mut w, j, k, &[(3400.0, 0.0)], 0).unwrap();
    edge(&mut w, k, j, &[(6600.0, 3000.0), (8000.0, -1500.0), (3800.0, 0.0)], 0).unwrap();
    for n in [a, m, k] {
        station(&mut w, n, 200);
    }
    (w, [a, m, j, k])
}

#[test]
fn a_balloon_turns_back_at_its_first_stop() {
    use super::net::LoopKind;
    let (mut w, [a, m, _, k]) = balloon_fixture();
    let id = line(&mut w, &[a, m, k, m, a], [12.0, 6.0, 3.0]);
    assert!(w.net.check_line(id).is_ok());
    assert_eq!(w.net.line_loop(id), LoopKind::Balloon);
    assert_eq!(w.net.turn_count(w.net.line_stops(id), w.net.line_path(id)), 1);
    let ls = &w.svc.lines[id as usize];
    assert!(ls.ok && ls.running && ls.kind == LoopKind::Balloon);
    // each run passes M twice, on the way out and back
    for r in 0..2 {
        assert_eq!(ls.runs[r].stop_idx.len(), 5);
        assert!(ls.runs[r].turn.iter().all(|&t| !t));
    }
    // the closing stand is the turnaround; A is a dead-end terminus
    let turn = w.net.line_turn[id as usize] as f64;
    for lev in 0..LEVELS {
        let lap = |r: usize| ls.runs[r].prof[lev].duration + turn;
        assert!((ls.round_trip[lev] - 2.0 * lap(0).max(lap(1))).abs() < 1e-9);
        assert_eq!(ls.trains[lev] % 2, 0);
    }
    let cap = &w.svc.cap;
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == a && cap.res[r].kind == ResKind::Terminus));
    assert!((0..cap.res.len()).all(|r| cap.res[r].node != a || cap.wire_kind(r) != 9), "no reversing limiter at a loop's own end");
    // the loop itself is plain track: no terminus or limiter at K
    assert!((0..cap.res.len()).all(|r| cap.res[r].node != k || cap.res[r].kind == ResKind::Platform));
    // trains take their own run's next departure from A
    let day = 2.0 * 86400.0;
    let trips = w.svc.trips(&w.net, day + 7.0 * 3600.0, day + 9.0 * 3600.0);
    for t in trips.iter().filter(|t| t.dep > day + 7.0 * 3600.0 && t.dep < day + 8.0 * 3600.0) {
        let n = t.next.unwrap();
        assert!(trips.iter().any(|u| u.draw == n && u.run == t.run), "{t:?}");
    }
}

#[test]
fn loops_save_round_trip_and_open_again() {
    use super::net::LoopKind;
    let (mut w, ns) = ring_fixture();
    let c = line(&mut w, &closed(&ns), [12.0, 6.0, 3.0]);
    set_counts(&mut w, c, [8, 4, 2]);
    let (mut v, [a, m, _, k]) = balloon_fixture();
    let b = line(&mut v, &[a, m, k, m, a], [6.0, 3.0, 0.0]);
    for (w, l, kind) in [(&w, c, LoopKind::Circle), (&v, b, LoopKind::Balloon)] {
        let bytes = super::save::encode(&w.net);
        let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
        assert_eq!(super::save::encode(&back.net), bytes);
        // positions are saved to a millimetre or so: the times agree to well under a second
        assert_eq!(back.svc.lines[l as usize].kind, kind);
        assert!((back.svc.lines[l as usize].round_trip[HIGH] - w.svc.lines[l as usize].round_trip[HIGH]).abs() < 1e-3);
        assert_eq!(back.svc.lines[l as usize].trains, w.svc.lines[l as usize].trains);
        assert_eq!(back.net.line_path(l), w.net.line_path(l));
    }
    // an odd count in a save rounds up on load
    let mut net = super::save::decode(&super::save::encode(&w.net)).unwrap();
    net.line_trains[c as usize] = Some([7, 3, 1]);
    let back = TrackWorld::new(net);
    assert_eq!(back.net.line_trains[c as usize], Some([8, 4, 2]));
    // opening the loop again (the closing stop removed) makes a plain line
    let mut d = w.net.line_data(c).unwrap();
    d.stops.pop();
    d.path = w.net.route(&d.stops).unwrap();
    w.apply(Op::Line { id: c, data: Some(d) }).unwrap();
    let ls = &w.svc.lines[c as usize];
    assert!(ls.ok && ls.kind == LoopKind::None);
    assert!((ls.round_trip[HIGH] - (ls.runs[0].prof[HIGH].duration + ls.runs[1].prof[HIGH].duration + 2.0 * w.net.line_turn[c as usize] as f64)).abs() < 1e-9);
}

/// The real save has no line whose last stop is its first (T-798 checks before loading them as
/// loops). `T798_TRACK=<track bytes> cargo test --release t798_real -- --ignored --nocapture`.
#[test]
#[ignore]
fn t798_real_save_has_no_loops() {
    let bytes = std::fs::read(std::env::var("T798_TRACK").expect("T798_TRACK: the save's track bytes")).unwrap();
    let net = super::save::decode(&bytes).unwrap();
    let loops: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l) && net.line_stops(l).len() >= 2 && net.line_stops(l).first() == net.line_stops(l).last()).collect();
    println!("{} lines, {} whose last stop is their first: {loops:?}", (0..net.line_count() as u32).filter(|&l| net.line_ok(l)).count(), loops.len());
    assert!(loops.is_empty());
}
