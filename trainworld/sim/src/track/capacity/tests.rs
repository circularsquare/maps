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

use super::super::bench::synth;
use super::super::net::StationData;
use super::super::service::Service;
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
    w.apply(Op::Edge { id, data: Some(super::super::net::EdgeData { a: b, b: c, tracks: 2, pis: vec![], built: false, off: false, thru: vec![], crossover: false, hsr: false }) }).unwrap();
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
    w.apply(Op::Edge { id: extension, data: Some(super::super::net::EdgeData { a:b,b:c,tracks:2,pis:vec![],built:false,off:false,thru:vec![],crossover:false, hsr:false }) }).unwrap();
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
            Op::Station { node: x, data: Some(StationData { platform: if d.platform > 120 { d.platform - 40 } else { d.platform + 40 }, ..d }) }
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
