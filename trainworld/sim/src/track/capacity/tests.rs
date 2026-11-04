use super::*;

#[test]
fn spec_delay_table() {
    // SPEC 6.2: a 90 s junction shared by two equal lines (mix 0.5).
    let w = |rho| kingman(rho, 0.5, 90.0);
    assert!((w(0.75) - 27.0).abs() < 1e-9);
    assert!((w(0.9) - 81.0).abs() < 1e-9);
    assert!((w(1.0) - 171.0).abs() < 1e-9);
    assert!((w(1.1) - 425.0).abs() < 1.0);
    // One line alone: nothing until over capacity.
    assert_eq!(kingman(0.95, 0.0, 90.0), 0.0);
    assert!(kingman(1.2, 0.0, 90.0) > 0.0);
    // T-008's 80 km/h section, two lines.
    let s = 90.0 + (80.0 / 3.6) / 0.6;
    assert!((kingman(0.75, 0.5, s) - 38.0).abs() < 1.0);
}

/// T-2231: every line's figures on the real save (trainworld/T-007-real-nyc.save), on synthetic
/// networks and on the following fixtures, written to `T2231_OUT` so two builds can be compared to
/// the bit. On request, on the save's track bytes (gunzipped, after its JSON header):
/// `T2231_TRACK=<file> T2231_OUT=<file> cargo test --manifest-path sim/Cargo.toml real_save_figures -- --ignored`.
#[test]
#[ignore]
fn real_save_figures() {
    use std::fmt::Write;
    use super::super::tests::line;
    let mut worlds = vec![TrackWorld::new(super::super::save::decode(&std::fs::read(std::env::var("T2231_TRACK").unwrap()).unwrap()).unwrap())];
    for seed in 1..=3u64 {
        worlds.push(TrackWorld::new(synth(250.0, 70, 12, seed)));
    }
    let (mut w, n, _) = twenty_km();
    line(&mut w, &n, [10.0, 4.0, 0.0]);
    line(&mut w, &[n[1], n[20]], [6.0, 2.0, 2.0]);
    line(&mut w, &[n[0], n[20]], [2.0, 1.0, 1.0]);
    worlds.push(w);
    let mut s = String::new();
    for w in &worlds {
        writeln!(s, "world").unwrap();
        for (l, ls) in w.svc.lines.iter().enumerate() {
            let bits = |v: &[f64]| v.iter().map(|x| format!("{:x}", x.to_bits())).collect::<Vec<_>>().join(",");
            writeln!(s, "line {l} {} {} rt {} trains {:?} delay {} tph {:?} phase {:?}", ls.ok, ls.running, bits(&ls.round_trip), ls.trains, bits(&ls.delay), w.net.line_tph.get(l), w.net.line_phase_of(l as u32)).unwrap();
            for (r, run) in ls.runs.iter().enumerate() {
                for lev in 0..LEVELS {
                    let h: Vec<f64> = run.holds[lev].iter().flat_map(|h| [h.s, h.delay, h.span]).collect();
                    writeln!(s, "  {r} {lev} dur {} holds {} stops {}", bits(&[run.prof[lev].duration]), bits(&h), bits(&run.stop_holds[lev])).unwrap();
                }
            }
        }
        for x in w.svc.cap.canonical() {
            writeln!(s, "{x:?}").unwrap();
        }
    }
    std::fs::write(std::env::var("T2231_OUT").unwrap(), s).unwrap();
}

use super::super::bench::synth;
use super::super::net::StationData;
use super::super::service::Service;
use super::super::upgrades;
use super::super::world::{Op, TrackWorld};

fn turnback_fixture(single: bool, pass: bool) -> (TrackWorld, u32, u32, u32) {
    use super::super::tests::{edge, line, node, station, world};
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 3000.0, 0.0, 0);
    let c = node(&mut w, 6000.0, 0.0, 0);
    for (x, y) in [(a, b), (b, c)] {
        let e = edge(&mut w, x, y, &[], 0).unwrap();
        if single {
            let mut d = w.net.edge_data(e).unwrap(); d.tracks = 1;
            w.apply(Op::Edge { id: e, data: Some(d) }).unwrap();
        }
    }
    for n in [a, b, c] { station(&mut w, n, 200); }
    let stops = if pass { vec![a, c] } else { vec![a, b, c] };
    let through = line(&mut w, &stops, [12.0, 8.0, 4.0]);
    let short = line(&mut w, &[a, b], [8.0, 0.0, 0.0]);
    (w, b, through, short)
}

#[test]
fn short_turns_occupy_platform_and_conflict_with_both_through_directions() {
    let (mut w, b, through, short) = turnback_fixture(false, false);
    let cap = &w.svc.cap;
    let crossing = cap.res.iter().enumerate().find(|(r, x)| x.node == b && x.kind == ResKind::Terminus
        && cap.users_of(*r).iter().any(|u| u.line == short && u.owner)).unwrap().0;
    // 8 reversing moves cross 12 opposite-direction through moves, 90 seconds each.
    assert!((cap.rho[crossing][0] - 20.0 * CONFLICT_S / 3600.0).abs() < 1e-12);
    assert!(cap.delay[crossing][0] > 0.0);
    assert_eq!(cap.rho[crossing][1], 0.0, "no crossover conflict when the reversing line is off");
    assert_eq!(cap.delay[crossing][1], 0.0);
    let users = cap.users_of(crossing);
    assert_eq!(users.iter().filter(|u| u.line == through && u.owner).count(), 0);
    assert_eq!(users.iter().filter(|u| u.line == through && !u.owner).count(), 1);
    assert_eq!(users.iter().filter(|u| u.line == short && u.owner).count(), 1, "turnaround counted once per train");
    let reciprocal = cap.res.iter().enumerate().find(|(r, x)| x.node == b && x.kind == ResKind::Terminus
        && cap.users_of(*r).iter().any(|u| u.line == through && u.owner)).unwrap().0;
    assert_eq!(cap.users_of(reciprocal).iter().filter(|u| u.line == through && u.owner).count(), 1);
    assert_eq!(cap.rho[reciprocal][0], cap.rho[crossing][0]);
    for run in 0..2 {
        assert!(w.svc.lines[through as usize].runs[run].stop_holds[0][1] > 0.0, "through direction {run} waits at the short turn");
        assert_eq!(w.svc.lines[through as usize].runs[run].stop_holds[1][1], 0.0);
    }
    let platforms: Vec<_> = cap.res.iter().enumerate().filter(|(_, r)| r.node == b && r.kind == ResKind::Platform).map(|(i, _)| i).collect();
    assert_eq!(platforms.len(), 2);
    let arrival = *platforms.iter().find(|&&r| cap.users_of(r).iter().any(|u| u.line == short)).unwrap();
    assert!((cap.rho[arrival][0] - (12.0 * (30.0 + PLATFORM_MARGIN_S) + 8.0 * (180.0 + TERMINUS_MARGIN_S)) / 3600.0).abs() < 1e-12);
    assert_eq!(cap.wire_kind(arrival), 6, "a reversal on a through platform is its own kind");
    assert!(platforms.iter().filter(|&&r| r != arrival).all(|&r| cap.wire_kind(r) == 2), "stopping trains alone are a plain platform");
    assert_eq!(cap.wire_kind(crossing), 8);
    assert!((0..cap.res.len()).any(|r| cap.res[r].kind == ResKind::Section && cap.wire_kind(r) == 7), "the stretch both lines run over");
    assert!((0..cap.res.len()).any(|r| cap.res[r].kind == ResKind::Section && cap.wire_kind(r) == 0), "the stretch only one runs over");
    same_as_global(&w, "short turn start");
    w.apply(Op::Schedule { id: short, tph: [0.0; 3] }).unwrap();
    same_as_global(&w, "short turn off");
    for run in 0..2 { assert_eq!(w.svc.lines[through as usize].runs[run].stop_holds[0][1], 0.0); }
    w.undo().unwrap().unwrap(); same_as_global(&w, "short turn undo");
    w.redo().unwrap().unwrap(); same_as_global(&w, "short turn redo");
}

#[test]
fn single_track_platform_shares_stops_and_short_turns_without_extra_tracks() {
    let (mut w, b, through, short) = turnback_fixture(true, false);
    assert_eq!(w.net.node_tracks(b), 1);
    let platforms: Vec<_> = w.svc.cap.res.iter().enumerate().filter(|(_, r)| r.node == b && r.kind == ResKind::Platform).map(|(i, _)| i).collect();
    assert_eq!(platforms.len(), 1);
    let platform = platforms[0];
    let cap = &w.svc.cap;
    assert_eq!(cap.users_of(platform).len(), 3);
    assert_eq!(cap.users_of(platform).iter().filter(|u| u.line == through).count(), 2);
    assert!((cap.rho[platform][0] - (24.0 * (30.0 + PLATFORM_MARGIN_S) + 8.0 * (180.0 + TERMINUS_MARGIN_S)) / 3600.0).abs() < 1e-12);
    assert!(cap.res.iter().all(|r| r.node != b || r.kind != ResKind::Terminus), "one shared platform; no fictional crossover or isolated turnaround pool");
    same_as_global(&w, "single short turn");
    w.apply(Op::Line { id: short, data: None }).unwrap();
    same_as_global(&w, "single remove short turn");
    let platform = w.svc.cap.res.iter().position(|r| r.node == b && r.kind == ResKind::Platform).unwrap();
    assert!((w.svc.cap.rho[platform][0] - 24.0 * (30.0 + PLATFORM_MARGIN_S) / 3600.0).abs() < 1e-12);
}

#[test]
fn short_turn_conflicts_include_trains_passing_the_station() {
    for single in [false, true] {
        let (w, b, through, _) = turnback_fixture(single, true);
        let cap = &w.svc.cap;
        for run in 0..2 {
            let resource = cap.res.iter().enumerate().find(|(r, x)| x.node == b && cap.delay[*r][0] > 0.0
                && cap.users_of(*r).iter().any(|u| u.line == through && u.run == run && u.owner)).unwrap().0;
            let u = cap.users_of(resource).iter().find(|u| u.line == through && u.run == run && u.owner).unwrap();
            assert_eq!(u.stop, -1);
            assert!(w.svc.lines[through as usize].runs[run as usize].holds[0].iter().any(|h| h.s == u.at && h.delay == cap.delay[resource][0]), "passing trains wait before the station");
        }
        same_as_global(&w, "passing short turn");
    }
}

#[test]
fn true_dead_end_keeps_pooled_platform_turnaround_capacity() {
    let (w, b, _, _) = turnback_fixture(false, false);
    let cap = &w.svc.cap;
    for (r, resource) in cap.res.iter().enumerate().filter(|(_, x)| x.kind == ResKind::Terminus && x.node != b) {
        assert!(cap.users_of(r).iter().all(|u| u.occ == (180.0 + TERMINUS_MARGIN_S) / 2.0));
        assert_eq!(w.net.node_ports[resource.node as usize].n, 1);
    }
}

#[test]
fn blueprint_extension_does_not_change_dead_end_turnaround_capacity() {
    use super::super::tests::{edge, line, node, station, world};
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 3000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    station(&mut w, a, 200); station(&mut w, b, 200);
    line(&mut w, &[a, b], [12.0, 8.0, 4.0]);
    let c = node(&mut w, 6000.0, 0.0, 0);
    let id = w.net.alloc_edge();
    w.apply(Op::Edge { id, data: Some(super::super::net::EdgeData { a: b, b: c, tracks: 2, pis: vec![], built: false, off: false, thru: vec![], crossover: false, hsr: false, sep: false }) }).unwrap();
    assert_eq!(w.net.built_ports(b), 1);
    let cap = &w.svc.cap;
    let term = cap.res.iter().position(|r| r.node == b && r.kind == ResKind::Terminus).unwrap();
    assert_eq!(cap.users_of(term).len(), 1);
    assert_eq!(cap.users_of(term)[0].occ, (180.0 + TERMINUS_MARGIN_S) / 2.0);
    assert!(cap.res.iter().all(|r| r.node != b || r.kind != ResKind::Platform));
    same_as_global(&w, "blueprint extension of terminus");
}

#[test]
fn short_turn_counts_settle_against_shared_platform_and_crossover_delays() {
    let (mut w, _, through, short) = turnback_fixture(false, false);
    for (id, counts) in [(through, [4, 4, 2]), (short, [1, 0, 0])] {
        let tph = w.net.line_tph[id as usize];
        w.apply(Op::TrainCounts { id, trains: Some(counts), tph }).unwrap();
    }
    for id in [through, short] {
        let ls = &w.svc.lines[id as usize];
        let counts = w.net.line_trains[id as usize].unwrap();
        assert_eq!(ls.trains, counts);
        for lev in 0..LEVELS {
            let implied = w.net.line_tph[id as usize][lev] as f64 * ls.round_trip[lev] / 3600.0;
            assert!((implied - counts[lev] as f64).abs() < 0.001, "count residual {implied}, target {}", counts[lev]);
        }
    }
    assert!(w.svc.lines[through as usize].delay[0] > 0.0);
    assert!(w.svc.lines[short as usize].delay[0] > 0.0);
}

#[test]
fn tiny_reversal_does_not_couple_independent_through_directions() {
    use super::super::tests::line;
    let (mut w, b, through, short) = turnback_fixture(false, true);
    w.apply(Op::Schedule { id: through, tph: [10.0, 8.0, 4.0] }).unwrap();
    let stops = w.net.line_stops(through).to_vec();
    let second = line(&mut w, &stops, [10.0, 8.0, 4.0]);
    w.apply(Op::Schedule { id: short, tph: [0.0; 3] }).unwrap();
    let baseline: Vec<_> = [through, second].iter().map(|&id| w.svc.lines[id as usize].runs.iter().map(|r| r.prof[0].hold).collect::<Vec<_>>()).collect();
    w.apply(Op::Schedule { id: short, tph: [0.001, 0.0, 0.0] }).unwrap();
    let cap = &w.svc.cap;
    for (r, resource) in cap.res.iter().enumerate().filter(|(_, x)| x.node == b && x.kind == ResKind::Terminus) {
        assert!(cap.rho[r][0] < 1.0, "two independent 20/h through directions must not be pooled: {}", cap.rho[r][0]);
        if cap.users_of(r).iter().any(|u| u.owner && (u.line == through || u.line == second)) {
            assert_eq!(cap.delay[r][0], 0.0, "rare reversing service adds no material trunk queue");
            assert_eq!(cap.users_of(r).iter().filter(|u| u.owner).map(|u| u.run).collect::<std::collections::BTreeSet<_>>().len(), 1);
        }
        assert_eq!(resource.kind, ResKind::Terminus);
    }
    for (j, &id) in [through, second].iter().enumerate() {
        for run in 0..2 { assert!((w.svc.lines[id as usize].runs[run].prof[0].hold - baseline[j][run]).abs() < 2.0); }
    }
    same_as_global(&w, "tiny reversal");
}

#[test]
fn opposite_short_turns_cross_without_through_trains() {
    use super::super::tests::line;
    let (mut w, b, through, short) = turnback_fixture(false, false);
    let far = *w.net.line_stops(through).last().unwrap();
    w.apply(Op::Line { id: through, data: None }).unwrap();
    let opposite = line(&mut w, &[far, b], [8.0, 0.0, 0.0]);
    let cap = &w.svc.cap;
    for id in [short, opposite] {
        let r = cap.res.iter().enumerate().find(|(r, x)| x.node == b && x.kind == ResKind::Terminus
            && cap.users_of(*r).iter().any(|u| u.line == id && u.owner)).unwrap().0;
        assert!(cap.delay[r][0] > 0.0, "opposite reversals must not be subtracted as one owners-only stream");
        assert!(cap.users_of(r).iter().any(|u| u.line != id && !u.owner));
    }
    same_as_global(&w, "opposite reversals");
}

#[test]
fn double_track_blueprint_cannot_double_single_track_terminal_capacity() {
    use super::super::tests::{edge, line, node, station, world};
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0); let b = node(&mut w, 3000.0, 0.0, 0);
    let e = edge(&mut w, a, b, &[], 0).unwrap();
    let mut d = w.net.edge_data(e).unwrap(); d.tracks = 1;
    w.apply(Op::Edge { id: e, data: Some(d) }).unwrap();
    station(&mut w, a, 200); station(&mut w, b, 200); line(&mut w, &[a,b], [8.0,4.0,2.0]);
    let c = node(&mut w, 6000.0, 0.0, 0); let extension = w.net.alloc_edge();
    w.apply(Op::Edge { id: extension, data: Some(super::super::net::EdgeData { a:b,b:c,tracks:2,pis:vec![],built:false,off:false,thru:vec![],crossover:false, hsr:false, sep: false }) }).unwrap();
    assert_eq!(w.net.node_tracks(b), 2, "planning geometry retains proposed double track");
    assert_eq!(active_node_tracks(&w.net,b),1, "active capacity sees only constructed track");
    let r = w.svc.cap.res.iter().position(|r|r.node==b&&r.kind==ResKind::Terminus).unwrap();
    assert_eq!(w.svc.cap.users_of(r)[0].occ,180.0+TERMINUS_MARGIN_S);
    same_as_global(&w,"mixed-track blueprint terminal");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random edit of the kinds that move resources: PIs, schedules, splits, deletes, flying
/// junctions, single or double track, platforms, stops, removing a line.
fn random_op(w: &mut TrackWorld, rng: &mut Rng) -> Option<Op> {
    let n = &w.net;
    let edges: Vec<u32> = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).collect();
    let lines: Vec<u32> = (0..n.line_count() as u32).filter(|&l| n.line_ok(l)).collect();
    let nodes: Vec<u32> = (0..n.node_count() as u32).filter(|&x| n.node_ok(x)).collect();
    // Busy edges are where the interesting changes are: prefer edges with lines.
    let used: Vec<u32> = edges.iter().copied().filter(|&e| !n.lines_on_edge(e).is_empty()).collect();
    let pick_edge = |rng: &mut Rng| if !used.is_empty() && rng.below(4) > 0 { used[rng.below(used.len())] } else { edges[rng.below(edges.len())] };
    Some(match rng.below(9) {
        0 => {
            let e = pick_edge(rng);
            let mut d = n.edge_data(e)?;
            if d.pis.is_empty() {
                return None;
            }
            let k = rng.below(d.pis.len());
            d.pis[k].x = quant(d.pis[k].x + rng.unit() * 40.0 - 20.0);
            Op::Edge { id: e, data: Some(d) }
        }
        1 | 2 => {
            let l = *lines.get(rng.below(lines.len()))?;
            let t = n.line_tph[l as usize];
            let tph = [(t[0] + rng.below(9) as f32 - 4.0).clamp(0.0, 30.0), (t[1] + rng.below(5) as f32 - 2.0).clamp(0.0, 30.0), t[2]];
            Op::Schedule { id: l, tph }
        }
        3 => {
            let e = pick_edge(rng);
            let s = n.edge_len[e as usize] * (0.2 + 0.6 * rng.unit());
            let node = w.net.alloc_node();
            let new_edge = w.net.alloc_edge();
            Op::Split { edge: e, s, node, new_edge }
        }
        4 => Op::DeleteEdge { id: pick_edge(rng) },
        5 => {
            let j: Vec<u32> = nodes.iter().copied().filter(|&x| n.node_ports[x as usize].n >= 3).collect();
            let x = *j.get(rng.below(j.len()))?;
            let mut d = n.node_data(x)?;
            d.flying = !d.flying;
            Op::Node { id: x, data: Some(d) }
        }
        6 => {
            let e = pick_edge(rng);
            let mut d = n.edge_data(e)?;
            d.tracks = 3 - d.tracks;
            Op::Edge { id: e, data: Some(d) }
        }
        7 => {
            let st: Vec<u32> = nodes.iter().copied().filter(|&x| n.node_platform[x as usize] > 0).collect();
            let x = *st.get(rng.below(st.len()))?;
            let d = n.station_data(x)?;
            Op::Station { node: x, data: Some(StationData { platform: if d.platform > 120 { d.platform - 40 } else { d.platform + 40 }, upgrades: d.upgrades ^ upgrades::TERMINAL, ..d }) }
        }
        _ => {
            let l = *lines.get(rng.below(lines.len()))?;
            let mut d = n.line_data(l)?;
            if rng.below(3) == 0 || d.stops.len() <= 3 {
                Op::Line { id: l, data: None }
            } else {
                // Drop a middle stop; the path stays (it still passes the station).
                d.stops.remove(1 + rng.below(d.stops.len() - 2));
                Op::Line { id: l, data: Some(d) }
            }
        }
    })
}

/// The incremental state against everything recomputed from the network.
fn same_as_global(w: &TrackWorld, what: &str) {
    let mut g = Service::default();
    g.update_all(&w.net);
    let (a, b) = (w.svc.cap.canonical(), g.cap.canonical());
    assert_eq!(a.len(), b.len(), "{what}: resource count");
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x, y, "{what}: resource");
    }
    for (l, (x, y)) in w.svc.lines.iter().zip(&g.lines).enumerate() {
        assert_eq!((x.ok, x.running), (y.ok, y.running), "{what}: line {l} state");
        if !x.ok {
            continue;
        }
        for r in 0..2 {
            assert_eq!(x.runs[r].holds, y.runs[r].holds, "{what}: line {l} run {r} holds");
            assert_eq!(x.runs[r].stop_holds, y.runs[r].stop_holds, "{what}: line {l} run {r} stop holds");
            assert_eq!(x.runs[r].prof, y.runs[r].prof, "{what}: line {l} run {r} profiles");
        }
        assert_eq!((x.round_trip, x.trains, x.delay), (y.round_trip, y.trains, y.delay), "{what}: line {l} round trip");
    }
}

/// T-050: random edit sequences (with undo and redo) on synthetic networks; after every
/// step the incremental capacity pass must equal the global one.
#[test]
fn incremental_equals_global() {
    let mut applied = 0;
    for seed in 1..=6u64 {
        let mut w = TrackWorld::new(synth(250.0, 70, 12, seed));
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        same_as_global(&w, "start");
        for step in 0..80 {
            let what = format!("seed {seed} step {step}");
            match rng.below(10) {
                0 => {
                    let _ = w.undo();
                }
                1 => {
                    let _ = w.redo();
                }
                _ => {
                    let Some(op) = random_op(&mut w, &mut rng) else { continue };
                    if w.apply(op).is_ok() {
                        applied += 1;
                    }
                }
            }
            same_as_global(&w, &what);
        }
    }
    assert!(applied > 200, "too few edits applied: {applied}");
}

/// T-797: a line reversing at T has T's short-turn platform and crossover, and the limiter:
/// 300 s a train for both directions arriving on the same side, so at most 6 an hour each way.
#[test]
fn reversing_partway_is_capped_at_twelve_an_hour_a_side() {
    use super::super::tests::{line, reversal_fixture};
    let (mut w, [_, t, _, m, br]) = reversal_fixture();
    let l = line(&mut w, &[br, t, m], [3.0, 6.0, 8.0]);
    let cap = &w.svc.cap;
    let lim: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) == 9).collect();
    assert_eq!(lim.len(), 1, "both directions arrive at T on the same side");
    let r = lim[0];
    assert_eq!(cap.res[r].node, t);
    assert_eq!(cap.users_of(r).len(), 2);
    assert!((cap.rho[r][HIGH] - 6.0 * REVERSAL_LIMIT_S / 3600.0).abs() < 1e-12);
    assert!((cap.rho[r][MEDIUM] - 1.0).abs() < 1e-12, "12 an hour fill it");
    assert_eq!(cap.delay[r][..2], [0.0; 2], "one line's two directions are one timetable: no queue until over");
    assert!(cap.delay[r][LOW] > 0.0, "16 an hour are over");
    let most = cap.line_limits(&w.net)[l as usize];
    assert!((most[HIGH].0 - 6.0).abs() < 1e-9 && most[HIGH].1 as usize == r, "{most:?}");
    assert!((cap.fill(&w.net, &[(l, 10.0)], HIGH)[0] - 6.0).abs() < 1e-9);
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == t && cap.wire_kind(r) == 6), "the arrival platform held for the turnaround");
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == t && cap.wire_kind(r) == 8), "the crossover");
    let j = w.net.path_nodes(w.net.line_path(l)[0]).1;
    assert!(!(0..cap.res.len()).any(|r| cap.res[r].node == t && cap.res[r].kind == ResKind::Junction));
    // No section joins across the turn: each direction of T - J is its own.
    let te = w.net.line_path(l)[1] >> 1;
    for r in (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Section) {
        if let Tag::Section { first, count } = cap.tag[r] {
            let ks = &cap.keys[first as usize..(first + count) as usize];
            assert!(!(ks.contains(&(te * 2)) && ks.contains(&(te * 2 + 1))), "{ks:?}");
        }
    }
    let _ = j;
    same_as_global(&w, "reversal");
    w.apply(Op::Schedule { id: l, tph: [0.0; 3] }).unwrap();
    same_as_global(&w, "reversal off");
    w.undo().unwrap().unwrap();
    same_as_global(&w, "reversal undo");
    // A second line reversing on the same side shares the limiter.
    let other = line(&mut w, &[br, t, m], [3.0, 0.0, 0.0]);
    same_as_global(&w, "two reversing");
    let cap = &w.svc.cap;
    let r = (0..cap.res.len()).find(|&r| cap.wire_kind(r) == 9).unwrap();
    assert_eq!(cap.users_of(r).len(), 4);
    assert!((cap.line_limits(&w.net)[other as usize][HIGH].0 - 3.0).abs() < 1e-9);
}

/// Give station `n` a terminal (T-1281), or take it away.
fn set_terminal(w: &mut TrackWorld, n: u32, on: bool) {
    let mut d = w.net.station_data(n).unwrap();
    d.upgrades = if on { d.upgrades | upgrades::TERMINAL } else { d.upgrades & !upgrades::TERMINAL };
    w.apply(Op::Station { node: n, data: Some(d) }).unwrap();
}

/// T-1281: a line ending at a through station with a terminal holds its two platforms for the
/// dwell and turns in the sidings, pooled over two: 30 an hour at a 3-minute turnaround.
#[test]
fn a_terminal_turns_in_sidings_and_frees_the_platform() {
    let (mut w, b, through, short) = turnback_fixture(false, false);
    set_terminal(&mut w, b, true);
    same_as_global(&w, "terminal");
    let cap = &w.svc.cap;
    assert!(!(0..cap.res.len()).any(|r| cap.res[r].node == b && cap.wire_kind(r) == 6), "no platform held for the turnaround");
    let sidings: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].node == b && cap.wire_kind(r) == 10).collect();
    assert_eq!(sidings.len(), 1);
    let u = cap.users_of(sidings[0]);
    assert_eq!(u.len(), 1);
    assert_eq!((u[0].line, u[0].occ), (short, (180.0 + TERMINUS_MARGIN_S) / 2.0));
    assert!((cap.rho[sidings[0]][HIGH] - 8.0 / 30.0).abs() < 1e-12, "30 an hour fill them");
    let platforms: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].node == b && cap.res[r].kind == ResKind::Platform).collect();
    assert_eq!(platforms.len(), 2);
    for &p in &platforms {
        assert_eq!(cap.wire_kind(p), 2);
        assert!(cap.users_of(p).iter().any(|u| u.line == short), "arrival and departure platforms");
        assert!((cap.rho[p][HIGH] - (12.0 + 8.0) * (30.0 + PLATFORM_MARGIN_S) / 3600.0).abs() < 1e-12);
    }
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == b && cap.wire_kind(r) == 8), "the turn still crosses the other track");
    let _ = through;
    set_terminal(&mut w, b, false);
    same_as_global(&w, "terminal removed");
    assert!((0..w.svc.cap.res.len()).any(|r| w.svc.cap.res[r].node == b && w.svc.cap.wire_kind(r) == 6));
    w.undo().unwrap().unwrap();
    same_as_global(&w, "terminal undo");
}

/// T-1281: a short turnaround at a dead end turns no more than a real terminus does (36 an hour
/// on two platform tracks), and reversing partway at a terminal is held to 30 an hour a side.
#[test]
fn pooled_turns_have_a_floor_and_a_terminal_lifts_the_reversing_limit() {
    let (w, b, _, _) = turnback_fixture(false, false);
    let mut w = w;
    let short = (0..w.net.line_count() as u32).filter(|&l| w.net.line_ok(l)).last().unwrap();
    let mut d = w.net.line_data(short).unwrap();
    d.turnaround_s = 20.0;
    w.apply(Op::Line { id: short, data: Some(d) }).unwrap();
    let cap = &w.svc.cap;
    let dead = (0..cap.res.len()).find(|&r| cap.res[r].node != b && cap.res[r].kind == ResKind::Terminus && cap.users_of(r).iter().any(|u| u.line == short)).unwrap();
    assert_eq!(cap.users_of(dead).iter().find(|u| u.line == short).unwrap().occ, upgrades::POOLED_TURN_MIN_S / 2.0);

    use super::super::tests::{line, reversal_fixture};
    let (mut w, [_, t, _, m, br]) = reversal_fixture();
    line(&mut w, &[br, t, m], [3.0, 6.0, 8.0]);
    set_terminal(&mut w, t, true);
    same_as_global(&w, "reversing at a terminal");
    let cap = &w.svc.cap;
    let r = (0..cap.res.len()).find(|&r| cap.wire_kind(r) == 9).unwrap();
    assert!((cap.rho[r][HIGH] - 6.0 * upgrades::TERMINAL_REVERSAL_S / 3600.0).abs() < 1e-12);
    assert!((0..cap.res.len()).any(|r| cap.res[r].node == t && cap.wire_kind(r) == 10), "it turns in the sidings");
}

/// T-798: a circle's sections join across its first stop, and the incremental pass agrees
/// with the global one around loops.
#[test]
fn loops_join_sections_and_update_incrementally() {
    use super::super::tests::{balloon_fixture, line, ring_fixture};
    let (mut w, ns) = ring_fixture();
    let mut stops = ns.clone();
    stops.push(ns[0]);
    let c = line(&mut w, &stops, [12.0, 6.0, 3.0]);
    same_as_global(&w, "circle");
    let partial = line(&mut w, &ns[1..4], [6.0, 3.0, 0.0]);
    same_as_global(&w, "circle and a line beside it");
    // the circle alone on S4 - S5 - S0 - S1: one section a direction runs through S0
    let cap = &w.svc.cap;
    let e_in = w.net.line_path(c)[5] >> 1;
    let e_out = w.net.line_path(c)[0] >> 1;
    let joined = (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Section).any(|r| match cap.tag[r] {
        Tag::Section { first, count } => {
            let ks = &cap.keys[first as usize..(first + count) as usize];
            ks.iter().any(|k| k / 2 == e_in) && ks.iter().any(|k| k / 2 == e_out)
        }
        _ => false,
    });
    assert!(joined);
    w.apply(Op::Line { id: partial, data: None }).unwrap();
    same_as_global(&w, "beside removed");
    w.undo().unwrap().unwrap();
    same_as_global(&w, "beside undone");
    let (mut v, [a, m, _, k]) = balloon_fixture();
    line(&mut v, &[a, m, k, m, a], [12.0, 6.0, 3.0]);
    same_as_global(&v, "balloon");
    line(&mut v, &[m, k], [6.0, 3.0, 0.0]);
    same_as_global(&v, "balloon and a line onto its loop");
}

/// T-1924: the formula of notes/T-1900.md's worked example: 16.5 minutes longer, a local every 6
/// minutes, 1m47s spacing: the express loses 14.4 minutes on average; behind a local once an hour,
/// only some trips catch it up.
#[test]
fn following_loss_formula() {
    use super::super::tests::world;
    let mut w = world();
    let (f, s) = (w.net.alloc_line(), w.net.alloc_line());
    w.net.line_tph[f as usize] = [6.0, 6.0, 0.0];
    w.net.line_tph[s as usize] = [10.0, 1.0, 10.0];
    let us = [
        User { line: f, run: 0, owner: true, occ: 0.0, at: 100.0, stop: -1 },
        User { line: s, run: 0, owner: false, occ: 990.0, at: 107.0, stop: -1 },
    ];
    let (high, who) = follow_loss(&us, &w.net, 0, &[]);
    assert!((high - (990.0 - (360.0 - 107.0) / 2.0)).abs() < 1e-9, "{high}");
    assert_eq!(who, s);
    let (rare, _) = follow_loss(&us, &w.net, 1, &[]);
    assert!((rare - 990.0 * 990.0 / (2.0 * (3600.0 - 107.0))).abs() < 1e-9, "{rare}");
    assert_eq!(follow_loss(&us, &w.net, 2, &[]).0, 0.0, "no express trains, no loss");
}

/// T-2419: the timetabled loss. At one headway each, `D + 2s - H`, or nothing where the gap holds the
/// gain; fewer faster trains the same; more of them, each a faster headway more; smooth in the rates.
#[test]
fn timetabled_following_loss() {
    let (d, s) = (900.0, 100.0);
    assert_eq!(follow_best(d, s, 600.0, 600.0), (500.0, 500.0));
    assert_eq!(follow_best(d, s, 1800.0, 1800.0).0, 0.0);
    assert_eq!(follow_best(d, s, 600.0, 1200.0).0, 500.0);
    // two faster trains a slower headway: 500 and 800 s
    assert!((follow_best(d, s, 600.0, 300.0).0 - 650.0).abs() < 1e-9);
    let at = |k: f64| follow_best(d, s, 600.0, 600.0 / k).0;
    for k in [1.0, 2.0, 3.0] {
        assert!((at(k - 1e-7) - at(k)).abs() < 1e-3, "smooth at {k}: {} {}", at(k - 1e-7), at(k));
    }
}

/// A straight 20 km double track with a station every km (stops `n[0..=20]`), and a second
/// double track beside it with stations only at its ends (`q`), as an express pair of quad track
/// is: separate edges.
fn twenty_km() -> (TrackWorld, Vec<u32>, [u32; 2]) {
    use super::super::tests::{edge, node, station, world};
    let mut w = world();
    let n: Vec<u32> = (0..=20).map(|k| node(&mut w, k as f64 * 1000.0, 0.0, 0)).collect();
    for k in 0..20 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
    for &x in &n { station(&mut w, x, 200); }
    let q = [node(&mut w, 0.0, 400.0, 0), node(&mut w, 20000.0, 400.0, 0)];
    edge(&mut w, q[0], q[1], &[], 0).unwrap();
    for &x in &q { station(&mut w, x, 200); }
    (w, n, q)
}

/// T-1924: an express behind a local on 20 km of shared double track, a local every 6 minutes,
/// takes about the local's time; on its own track (quad track's other pair) it keeps its speed. Its
/// loss is slower running over the stretch, never a stand between stops (T-2018).
#[test]
fn express_behind_local_takes_about_its_time() {
    use super::super::tests::line;
    let (mut w, n, q) = twenty_km();
    let local = line(&mut w, &n, [10.0, 4.0, 0.0]);
    let express = line(&mut w, &[n[1], n[20]], [6.0, 2.0, 2.0]);
    let own = line(&mut w, &q, [6.0, 2.0, 2.0]);
    same_as_global(&w, "express and local");
    let dur = |w: &TrackWorld, l: u32, lev: usize| w.svc.lines[l as usize].runs[0].prof[lev].duration;
    let (lo, ex, free) = (dur(&w, local, 0), dur(&w, express, 0), dur(&w, own, 0));
    eprintln!("high: local {:.0} s, express {:.0} s (free {:.0} s), own track {:.0} s", lo, ex, w.svc.lines[express as usize].runs[0].free.duration, free);
    assert!(ex > lo - 300.0 && ex < lo, "the express follows the local: {ex} against {lo}");
    assert!((free - w.svc.lines[own as usize].runs[0].free.duration).abs() < 30.0, "own track: no loss");
    assert!(!w.svc.cap.res.iter().zip(&w.svc.cap.tag).any(|(_, t)| matches!(t, Tag::Follow { line } if *line == own)));
    assert!(w.svc.cap.res.iter().zip(&w.svc.cap.tag).all(|(_, t)| !matches!(t, Tag::Follow { line } if *line == local)), "the local is never behind the express");
    // T-2018: the loss is slower running over the whole stretch: no stand between its stops, none
    // of it at the first stop
    let run = &w.svc.lines[express as usize].runs[0];
    eprintln!("holds {:?}", run.holds[0].iter().map(|h| (h.s.round(), h.delay.round(), h.span.round())).collect::<Vec<_>>());
    assert_eq!(run.holds[0].iter().filter(|h| h.span > 0.0).count(), 1, "one spread hold");
    assert_eq!(run.stop_holds[0][0], 0.0);
    never_stands_between_stops(run, 0);
    // at medium the locals run every 15 minutes: some trips get through first, so it loses less
    let (lo1, ex1) = (dur(&w, local, 1), dur(&w, express, 1));
    assert!(ex1 < ex - 60.0 && ex1 < lo1, "medium: express {ex1}, high {ex}");
    // at low no local runs: no loss
    assert!((dur(&w, express, 2) - w.svc.lines[express as usize].runs[0].free.duration).abs() < 1.0);
    // the panel's split names the local
    let r = (0..w.svc.cap.res.len()).find(|&r| matches!(w.svc.cap.tag[r], Tag::Follow { line } if line == express)).unwrap();
    assert_eq!(w.svc.cap.follow_leader(&w.net, r, 0), Some(local));
    assert_eq!(w.svc.cap.wire_kind(r), 11);
    // the local's schedule moves the express; removing the local frees it
    w.apply(Op::Schedule { id: local, tph: [4.0, 4.0, 0.0] }).unwrap();
    same_as_global(&w, "fewer locals");
    assert!(dur(&w, express, 0) < ex - 60.0);
    w.apply(Op::Line { id: local, data: None }).unwrap();
    same_as_global(&w, "no local");
    assert!((dur(&w, express, 0) - w.svc.lines[express as usize].runs[0].free.duration).abs() < 1.0);
    w.undo().unwrap().unwrap();
    same_as_global(&w, "local back");
}

/// T-2018: a run's profile at `lev` stands still only at its stops, and keeps moving everywhere
/// between them (sampled every 5 s).
fn never_stands_between_stops(run: &super::super::service::Run, lev: usize) {
    use super::super::profile::state_at;
    let p = &run.prof[lev];
    let mut t = 0.0;
    while t < p.duration {
        let (s, v) = state_at(&p.phases, t);
        let at_stop = run.stop_s.iter().any(|&x| (x - s).abs() < 5.0);
        assert!(v > 0.01 || at_stop, "stands at {s:.0} m, t {t:.0} s, between stops {:?}", run.stop_s);
        t += 5.0;
    }
}

/// T-2419: an express that turns where the local turns is timetabled just ahead of a local: with a
/// local every 15 minutes it gains more than the gap and catches the local ahead, losing its gain
/// less the gap; with one an hour it gets through. It leaves a little before a local.
#[test]
fn express_on_the_locals_path_leaves_just_ahead_of_it() {
    use super::super::tests::line;
    let (mut w, n, _) = twenty_km();
    let local = line(&mut w, &n, [4.0, 1.0, 0.0]);
    let express = line(&mut w, &[n[0], n[20]], [2.0, 1.0, 1.0]);
    same_as_global(&w, "same path");
    let dur = |w: &TrackWorld, l: u32, lev: usize| w.svc.lines[l as usize].runs[0].prof[lev].duration;
    let free = |w: &TrackWorld, l: u32| w.svc.lines[l as usize].runs[0].free.duration;
    let gain = free(&w, local) - free(&w, express);
    for lev in 0..2 {
        eprintln!("level {lev}: local {:.0} s, express {:.0} s (free {:.0} s, gain {gain:.0} s)", dur(&w, local, lev), dur(&w, express, lev), free(&w, express));
    }
    let lost = dur(&w, express, 0) - free(&w, express);
    assert!(lost > gain - 900.0 && lost < gain - 600.0, "a local every 15 minutes: the express loses its gain less the gap: {lost:.0} of {gain:.0} s");
    assert!(dur(&w, express, 1) - free(&w, express) < 60.0, "a local an hour: the express gets through");
    assert!((dur(&w, express, 2) - free(&w, express)).abs() < 1.0, "no local at low");
    // T-2018: it runs the whole way at a lowered speed, never standing between its two stops
    for r in 0..2 {
        never_stands_between_stops(&w.svc.lines[express as usize].runs[r], 0);
    }
    // its departures a little before a local's
    let deps = |l: u32| {
        let mut out = vec![];
        let t0 = 86400.0 + 8.0 * 3600.0;
        w.svc.run_deps(&w.net, l as usize, 0, t0, t0 + 3600.0, &mut out);
        out.iter().map(|t| t.dep).collect::<Vec<f64>>()
    };
    let (dl, dx) = (deps(local), deps(express));
    for x in &dx {
        let before = dl.iter().map(|l| l - x).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
        assert!(before < 300.0, "an express leaves {before:.0} s before a local: {dl:?} {dx:?}");
    }
}

/// T-2419 (Anita, 2026-11-04: "you should not helpfully be able to put the 7 express and local on the
/// double same track in nyc"): an express and a local on one double track, 16 km with a station every
/// 800 m, the express at every fourth, both every 3 minutes (high) and every 4 (medium), no passing
/// tracks: the express loses most of what it would gain, and the track is as full as two lines
/// sharing it make it.
#[test]
fn seven_express_and_local_on_one_double_track() {
    use super::super::tests::{edge, line, node, station, world};
    let mut w = world();
    let n: Vec<u32> = (0..=20).map(|k| node(&mut w, k as f64 * 800.0, 0.0, 0)).collect();
    for k in 0..20 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
    for &x in &n { station(&mut w, x, 200); }
    let local = line(&mut w, &n, [20.0, 15.0, 0.0]);
    let stops: Vec<u32> = n.iter().step_by(4).copied().collect();
    let express = line(&mut w, &stops, [20.0, 15.0, 0.0]);
    same_as_global(&w, "the 7");
    let run = |l: u32| &w.svc.lines[l as usize].runs[0];
    let gain = run(local).free.duration - run(express).free.duration;
    let rho = |lev: usize| (0..w.svc.cap.res.len()).filter(|&r| w.svc.cap.res[r].kind == ResKind::Section).map(|r| w.svc.cap.rho[r][lev]).fold(0.0, f64::max);
    for lev in 0..2 {
        let lost = run(express).prof[lev].duration - run(express).free.duration;
        let cap = &w.svc.cap;
        let behind = (0..cap.res.len()).filter(|&r| cap.wire_kind(r) == 11 && cap.users_of(r).iter().any(|u| u.owner && u.line == express && u.run == 0)).map(|r| cap.delay[r][lev]).fold(0.0, f64::max);
        eprintln!(
            "level {lev}: local {:.0} s ({:.0} free), express {:.0} s ({:.0} free), gain {gain:.0} s, lost {lost:.0} s of which behind the local {behind:.0} s; trains an hour {:?} {:?}; fullest track {:.0}%",
            run(local).prof[lev].duration, run(local).free.duration, run(express).prof[lev].duration, run(express).free.duration,
            w.net.line_tph[local as usize][lev], w.net.line_tph[express as usize][lev], 100.0 * rho(lev)
        );
        assert!(behind > 0.75 * gain, "level {lev}: the express keeps under a quarter of its gain: behind the local {behind:.0} of {gain:.0} s");
        assert!(rho(lev) > 0.8, "level {lev}: the shared track is busy: {:.2}", rho(lev));
    }
}

/// T-1924: a high-speed train behind ordinary trains on high-speed track loses time too.
#[test]
fn high_speed_behind_ordinary_trains() {
    use super::super::tests::{line, node, station, world};
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 60000.0, 0.0, 0);
    let id = w.net.alloc_edge();
    w.apply(Op::Edge { id, data: Some(super::super::net::EdgeData { a, b, tracks: 2, pis: vec![], built: true, off: false, thru: vec![], crossover: false, hsr: true, sep: false }) }).unwrap();
    station(&mut w, a, 200);
    station(&mut w, b, 200);
    let slow = line(&mut w, &[a, b], [8.0, 0.0, 0.0]);
    let fast = line(&mut w, &[a, b], [4.0, 0.0, 0.0]);
    let mut d = w.net.line_data(fast).unwrap();
    d.fast = true;
    w.apply(Op::Line { id: fast, data: Some(d) }).unwrap();
    same_as_global(&w, "high speed");
    let run = &w.svc.lines[fast as usize].runs[0];
    let slow_run = &w.svc.lines[slow as usize].runs[0];
    eprintln!("high speed {:.0} s (free {:.0} s), ordinary {:.0} s", run.prof[0].duration, run.free.duration, slow_run.prof[0].duration);
    assert!(run.free.duration < slow_run.free.duration - 500.0);
    assert!(run.prof[0].duration > run.free.duration + 300.0, "it is held behind the ordinary trains");
    // timetabled just ahead of an ordinary train every 7.5 minutes (T-2419), it catches the one ahead
    // and keeps only the gap of what it would gain
    assert!(run.prof[0].duration < slow_run.prof[0].duration - 60.0, "it keeps some of its gain");
}
