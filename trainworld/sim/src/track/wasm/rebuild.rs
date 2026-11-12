//! Changing picked track's level and track type in place (T-815, T-1898), and rebuilding built track that
//! way (T-3059): the same edges, nodes and stations, so lines, names and everything keyed by id stay.

use super::*;

/// What a pick of edges can change: the ops to apply, and the picked edges the track model would refuse.
pub(super) struct Plan {
    pub take: Vec<Op>,
    /// Every picked edge left as it is.
    pub refused: Vec<u32>,
    /// Of those, the ones refused for their own sake (T-3088): the rest only join one of these, or
    /// track outside the pick, and cannot change family alone.
    pub root: Vec<u32>,
    pub issues: Vec<Issue>,
    /// T-3241: track outside the pick that changes, taking the ramps of the stations at the pick's
    /// ends: each edge, the node its change starts at, and how far along it the change reaches, m.
    pub beyond: Vec<(u32, u32, f64)>,
    /// Stations that move with the pick (picked or not).
    pub moved: Vec<u32>,
}

/// The settings a plan puts on its pieces.
#[derive(Clone, Copy)]
struct Want<'a> {
    level: Option<i8>,
    kind: Option<TrackKind>,
    stations: &'a [u32],
    /// T-3241: nodes that move although some of their track is outside the pick (sorted).
    ext: &'a [u32],
}

/// T-3241: a ramp beyond a station starts this far past the platform's end, m.
const PAST_PLATFORM: f64 = 2.0;
/// T-3241: the most pieces a station's ramp runs on through, joint after joint.
const BEYOND_PIECES: usize = 8;

/// T-3241: a station at an end of a level change's pick goes to the new level with it, and the ramp goes on
/// the track beyond it, outside the pick.
#[derive(Default)]
struct Beyond {
    /// Every station that moves with the pick (sorted): at its ends, with its track or at a dead end, and
    /// inside it.
    stations: Vec<u32>,
    /// The nodes that move with track outside the pick (sorted): the stations with a ramp beyond them,
    /// and the joints and loose ends that track runs through.
    nodes: Vec<u32>,
    runs: Vec<Run>,
    /// Stations whose ramp has no room beyond them, with the piece it has none on.
    no_room: Vec<(u32, u32)>,
}

/// One station's ramp beyond the pick.
struct Run {
    station: u32,
    /// The picked piece at the station.
    anchor: u32,
    ops: Vec<Op>,
    /// The joints and loose ends that move.
    nodes: Vec<u32>,
    /// The edges changed: each, the node its change starts at, and how far along it that reaches, m.
    edges: Vec<(u32, u32, f64)>,
}

impl Beyond {
    fn anchor_of_edge(&self, e: u32) -> Option<u32> {
        self.runs.iter().find(|r| r.edges.iter().any(|x| x.0 == e)).map(|r| r.anchor)
    }
    fn anchor_of_node(&self, n: u32) -> Option<u32> {
        self.runs.iter().find(|r| r.station == n || r.nodes.contains(&n)).map(|r| r.anchor)
    }
}

fn floor_mm(v: f64) -> f64 {
    ((v / QUANT) + 1e-6).floor() * QUANT
}

/// How far from the start two height profiles of one alignment differ: the chainage past which they agree.
fn changed_to(old: &[[f64; 2]], new: &[[f64; 2]], len: f64) -> f64 {
    let mut at: Vec<f64> = old.iter().chain(new.iter()).map(|p| p[0]).chain([0.0, len]).collect();
    at.sort_by(f64::total_cmp);
    at.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let differ = |s: f64| (geom::height_at(old, s) - geom::height_at(new, s)).abs() > 1e-6;
    match (0..at.len()).rev().find(|&i| differ(at[i])) {
        Some(i) => at.get(i + 1).copied().unwrap_or(len).min(len),
        None => 0.0,
    }
}

/// `e` leaving node `p`, which goes to level `l` (its platform reaching `reach` along `e`), ramping back to
/// the edge's own levels as soon past `reach` as its shape allows, the far end kept (its platform reaching
/// `reach_m`): the edge's new data and how far along it the change reaches from `p`, m. A ramp is centred
/// between two vertices of different levels, so it is held either by the edge's corners on level `l` up to
/// some corner, or by two new vertices on a straight with no turn (as a bridge's are, track/bridge.rs), the
/// one that changes the least track. The shape stays as it is. None where no ramp fits.
fn ramp_in(net: &Network, e: u32, p: u32, l: i8, reach: f64, reach_m: f64) -> Option<(EdgeData, f64)> {
    let mut d = net.edge_data(e)?;
    let flip = net.edge_a[e as usize] != p;
    let mut old = net.edge_vertices(e);
    if flip {
        old.reverse();
    }
    let f0 = net.fit(&old);
    if f0.pieces.is_empty() {
        return None;
    }
    let n = old.len();
    let bad = |g: &geom::Fit| g.issues.len() > f0.issues.len() || g.issues.iter().any(|i| matches!(i, GeomIssue::RampTooShort(_) | GeomIssue::LevelRange(_)));
    let judge = |v: &[Pi], moved: bool| -> Option<f64> {
        let g = net.fit(v);
        if bad(&g) || (g.len - f0.len).abs() > 0.05 {
            return None;
        }
        if moved {
            let mut s = 0.0;
            while s < f0.len {
                let (a, b) = (geom::pos_at(&f0.pieces, s), geom::pos_at(&g.pieces, s));
                if (a.0 - b.0).hypot(a.1 - b.1) > 0.25 {
                    return None;
                }
                s += 25.0;
            }
        }
        let flat = (geom::height_at(&g.vert, 0.0) - l as f64 * LEVEL_H).abs() < 1e-6 && !geom::on_ramp(&g.vert, 0.0, reach + 1.0);
        if !flat || (reach_m > 0.0 && geom::on_ramp(&g.vert, g.len - reach_m - 1.0, g.len)) {
            return None;
        }
        Some(changed_to(&f0.vert, &g.vert, g.len))
    };
    let mut base = old.clone();
    base[0].level = l;
    let mut best: Option<(f64, Vec<Pi>)> = None;
    let mut take = |v: Vec<Pi>, moved: bool| {
        if let Some(c) = judge(&v, moved) {
            if best.as_ref().map_or(true, |b| c < b.0 - 1e-6) {
                best = Some((c, v));
            }
        }
    };
    // the corners up to the j-th on the new level, the ramp centred on the leg after
    for j in 0..n - 1 {
        let mut v = base.clone();
        for p in &mut v[1..=j] {
            p.level = l;
        }
        take(v, false);
    }
    // two new vertices holding the ramp on a straight, from just past the platform or a curve's end
    if d.thru.is_empty() {
        let radius = if d.kind.is_hsr() { quant(design_radius(V_HSR * 3.6)) } else { 0.0 };
        for (j, q) in f0.pieces.iter().enumerate() {
            if q.k != 0.0 {
                continue;
            }
            let i = f0.piece_at_vertex[j] as usize;
            if old[i].level != old[i + 1].level {
                continue;
            }
            let u = (q.s0 + 1.0).max(reach + PAST_PLATFORM);
            let w = u + RAMP_PER_LEVEL * (l as f64 - old[i].level as f64).abs() + 0.01;
            if w > q.s0 + q.len - 1.0 {
                continue;
            }
            let at = |s: f64, level: i8| {
                let (x, y, _) = q.at(s - q.s0);
                Pi::new(quant(x), quant(y), radius, level)
            };
            let mut v: Vec<Pi> = base[..=i].to_vec();
            for p in &mut v[1..] {
                p.level = l;
            }
            v.push(at(u, l));
            v.push(at(w, old[i].level));
            v.extend_from_slice(&old[i + 1..]);
            // the corners either side keep the curve their legs gave them
            for k in [i, i + 3] {
                if k > 0 && k < v.len() - 1 && v[k].radius == 0.0 {
                    let src = if k == i { i } else { i + 1 };
                    if f0.radius[src] > 0.0 {
                        v[k].radius = floor_mm(f0.radius[src]);
                    }
                }
            }
            take(v, true);
        }
    }
    let (c, mut v) = best?;
    if flip {
        v.reverse();
    }
    d.pis = v[1..v.len() - 1].to_vec();
    Some((d, c))
}

/// `e` whole on level `l`.
fn whole_at(net: &Network, e: u32, l: i8) -> Option<EdgeData> {
    let mut d = net.edge_data(e)?;
    for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
        p.level = l;
    }
    Some(d)
}

/// Where the level of `cand` changes to `l` (not across families), the stations at its ends and the
/// track beyond them that takes their ramps (T-3241). A station whose track is all picked, or that ends
/// the track, just moves. Else the track beyond it ramps back to its own level as soon past the platform
/// as it can, on the next piece or, where that is too short, on through plain joints (`BEYOND_PIECES`),
/// those pieces and joints moving whole. Track running on to another station that moves, or back into the
/// pick, moves whole; a loose end with no room for the ramp moves with it. A station with no room for its
/// ramp (a junction or another station too close, constructed track in a blueprint pick) stays.
fn beyond(net: &Network, cand: &[u32], level: Option<i8>, built_ok: bool, families: bool) -> Beyond {
    let mut out = Beyond::default();
    let Some(l) = level else { return out };
    if families {
        return out;
    }
    let picked = |e: u32| cand.binary_search(&e).is_ok();
    let mut ends: Vec<u32> = cand
        .iter()
        .flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]])
        .filter(|&n| {
            let i = n as usize;
            net.node_platform[i] > 0 && net.node_berth[i].is_none() && net.node_level[i] != l && (built_ok || !net.node_built[i])
        })
        .collect();
    ends.sort_unstable();
    ends.dedup();
    let usable = |e: u32| net.edge_ok(e) && !net.edge_crossover[e as usize] && (built_ok || !net.edge_built[e as usize]) && !picked(e);
    let plain = |n: u32| {
        let i = n as usize;
        net.node_ports[i].n == 2 && net.node_platform[i] == 0 && net.node_berth[i].is_none()
    };
    let other = |e: u32, n: u32| if net.edge_a[e as usize] == n { net.edge_b[e as usize] } else { net.edge_a[e as usize] };
    let node_op = |n: u32| net.node_data(n).filter(|d| d.level != l).map(|d| Op::Node { id: n, data: Some(NodeData { level: l, ..d }) });
    for &s in &ends {
        let at: Vec<u32> = net.node_ports[s as usize].ends().iter().map(|&c| c >> 1).collect();
        let Some(anchor) = at.iter().copied().find(|&e| picked(e)) else { continue };
        let rest: Vec<u32> = at.iter().copied().filter(|&e| !picked(e)).collect();
        let e0 = match rest.as_slice() {
            [] => {
                out.stations.push(s);
                continue;
            }
            [e] => *e,
            _ => continue,
        };
        // the pieces beyond, through plain joints, and what the last one ends at
        let mut chain: Vec<(u32, u32)> = vec![];
        let (mut p, mut e) = (s, e0);
        let (mut joins, mut loose) = (false, false);
        while usable(e) && chain.len() < BEYOND_PIECES {
            chain.push((e, p));
            let m = other(e, p);
            if ends.binary_search(&m).is_ok() {
                joins = true;
                break;
            }
            if !plain(m) {
                loose = net.node_ports[m as usize].n == 1 && net.node_platform[m as usize] == 0;
                break;
            }
            let next = net.node_ports[m as usize].ends().iter().map(|&c| c >> 1).find(|&x| x != e).unwrap_or(e);
            if picked(next) {
                joins = true;
                break;
            }
            p = m;
            e = next;
        }
        let Some(&(first, _)) = chain.first() else {
            out.no_room.push((s, e0));
            continue;
        };
        let mut run = Run { station: s, anchor, ops: vec![], nodes: vec![], edges: vec![] };
        // the first `upto` pieces whole on the new level, with the joints between them and, where
        // `last_node`, the node after the last
        let all_whole = |run: &mut Run, upto: usize, last_node: bool| {
            for (k, &(e, p)) in chain[..upto].iter().enumerate() {
                run.ops.extend(whole_at(net, e, l).map(|d| Op::Edge { id: e, data: Some(d) }));
                run.edges.push((e, p, net.edge_len[e as usize]));
                let m = other(e, p);
                if k + 1 < upto || last_node {
                    run.ops.extend(node_op(m));
                    run.nodes.push(m);
                }
            }
        };
        if joins {
            // on to another station that moves, or back into the pick (whose node there moves with the
            // pick's own ops): no ramp, all of it moves
            all_whole(&mut run, chain.len(), false);
            let (e, p) = chain[chain.len() - 1];
            let m = other(e, p);
            if ends.binary_search(&m).is_err() {
                run.nodes.push(m);
            }
            out.runs.push(run);
            out.stations.push(s);
            continue;
        }
        // the first piece of the chain the ramp fits on, those before it moving whole
        let mut found = false;
        for k in 0..chain.len() {
            let (e, p) = chain[k];
            let m = other(e, p);
            let reach = if k == 0 { net.platform_reach(s) } else { 0.0 };
            let reach_m = if net.node_platform[m as usize] > 0 { net.platform_reach(m) } else { 0.0 };
            if let Some((d, c)) = ramp_in(net, e, p, l, reach, reach_m) {
                all_whole(&mut run, k, true);
                run.ops.push(Op::Edge { id: e, data: Some(d) });
                run.edges.push((e, p, c));
                found = true;
                break;
            }
        }
        if !found && loose {
            all_whole(&mut run, chain.len(), true);
            found = true;
        }
        if found {
            out.runs.push(run);
            out.stations.push(s);
        } else {
            out.no_room.push((s, first));
        }
    }
    // two stations whose track beyond runs into each other share it: one op for each piece and joint
    let (mut seen_e, mut seen_n): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
    for r in &mut out.runs {
        r.ops.retain(|op| match op {
            Op::Edge { id, .. } => !seen_e.contains(id),
            Op::Node { id, .. } => !seen_n.contains(id),
            _ => true,
        });
        r.edges.retain(|x| !seen_e.contains(&x.0));
        seen_e.extend(r.edges.iter().map(|x| x.0));
        seen_n.extend(r.ops.iter().filter_map(|op| if let Op::Node { id, .. } = op { Some(*id) } else { None }));
    }
    out.stations.sort_unstable();
    out.nodes = out.runs.iter().flat_map(|r| std::iter::once(r.station).chain(r.nodes.iter().copied())).collect();
    out.nodes.sort_unstable();
    out.nodes.dedup();
    out
}

/// A piece whose track type can change: plain double track (single track and crossovers have no high-speed or
/// tram form).
fn kind_ok(d: &EdgeData) -> bool {
    !d.crossover && d.tracks == 2
}

/// The family (rail or tram) `e` has once the plan's pieces `cand` have changed.
fn family_after(net: &Network, cand: &[u32], w: Want, e: u32) -> u8 {
    let k = net.edge_kind[e as usize];
    match w.kind {
        Some(to) if cand.binary_search(&e).is_ok() && net.edge_data(e).is_some_and(|d| kind_ok(&d)) => to.spec().family,
        _ => k.spec().family,
    }
}

/// Whether node `n` moves to the new level with `cand`: all its track is among them, and a constructed
/// station on it is in the pick; or it moves with track beyond the pick (T-3241).
fn movable(net: &Network, cand: &[u32], w: Want, n: u32) -> bool {
    if w.ext.binary_search(&n).is_ok() {
        return true;
    }
    net.node_ports[n as usize].ends().iter().all(|&c| cand.binary_search(&(c >> 1)).is_ok())
        && (!(net.node_platform[n as usize] > 0 && net.node_built[n as usize]) || w.stations.binary_search(&n).is_ok())
}

fn cornerless(net: &Network, e: u32) -> bool {
    net.edge_data(e).is_some_and(|d| d.pis.is_empty() && d.thru.is_empty())
}

/// The op that puts `e` on the plan's settings, None when it has them already.
fn edge_op(net: &Network, w: Want, e: u32) -> Option<Op> {
    let mut d = net.edge_data(e)?;
    let mut changed = false;
    if let Some(level) = w.level {
        if !d.pis.iter().chain(d.thru.iter()).all(|p| p.level == level) {
            for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
                p.level = level;
            }
            changed = true;
        }
    }
    if let Some(k) = w.kind {
        if kind_ok(&d) && d.kind != k {
            d.kind = k;
            changed = true;
        }
    }
    changed.then_some(Op::Edge { id: e, data: Some(d) })
}

/// Drop from `cand` the pieces that cannot change with the rest as it stands: a piece with no corners
/// whose ends cannot both move (it has no level of its own to ramp from), and a piece whose new family
/// would meet the other family at a node. Dropped pieces go to `refused`, and to `root` unless all
/// they meet is a piece refused already. `clash_at`: the first node where the families meet.
fn prune(net: &Network, cand: &mut Vec<u32>, w: Want, refused: &mut Vec<u32>, root: &mut Vec<u32>, clash_at: &mut Option<u32>) {
    let families = w.kind.is_some_and(|k| cand.iter().any(|&e| net.edge_kind[e as usize].spec().family != k.spec().family));
    loop {
        let was = cand.clone();
        let mut gone: Vec<(u32, bool)> = vec![];
        for &e in &was {
            if w.level.is_some() && cornerless(net, e) && !(movable(net, &was, w, net.edge_a[e as usize]) && movable(net, &was, w, net.edge_b[e as usize])) {
                gone.push((e, true));
                continue;
            }
            if !families {
                continue;
            }
            let f = family_after(net, &was, w, e);
            let mut own = false;
            let mut clash = false;
            for n in [net.edge_a[e as usize], net.edge_b[e as usize]] {
                for &c in net.node_ports[n as usize].ends() {
                    let o = c >> 1;
                    if family_after(net, &was, w, o) != f {
                        clash = true;
                        clash_at.get_or_insert(n);
                        own |= refused.binary_search(&o).is_err();
                    }
                }
            }
            if clash {
                gone.push((e, own));
            }
        }
        if gone.is_empty() {
            return;
        }
        for (e, own) in gone {
            cand.retain(|&c| c != e);
            refused.push(e);
            if own {
                root.push(e);
            }
        }
        refused.sort_unstable();
        root.sort_unstable();
    }
}

/// The picked edges an edit's problems put the blame on: the piece named, the pieces of the pick at
/// the node named, or for a piece outside the pick, the picked pieces at its ends. Of two picked
/// pieces that cross, the later one. A problem on track beyond the pick that a station's ramp takes is the
/// picked piece's at that station (T-3241).
fn culprits(net: &Network, cand: &[u32], issues: &[Issue], flats: &[(u32, u32)], bey: &Beyond) -> Vec<u32> {
    let picked = |e: u32| cand.binary_search(&e).is_ok();
    let mut out = vec![];
    let at_node = |n: u32, out: &mut Vec<u32>| {
        if net.node_ok(n) {
            out.extend(net.node_ports[n as usize].ends().iter().map(|&c| c >> 1).filter(|&e| picked(e)));
            out.extend(bey.anchor_of_node(n));
        }
    };
    let edge = |e: u32, out: &mut Vec<u32>| {
        if picked(e) {
            out.push(e);
        } else if let Some(a) = bey.anchor_of_edge(e) {
            out.push(a);
        } else if net.edge_ok(e) {
            at_node(net.edge_a[e as usize], out);
            at_node(net.edge_b[e as usize], out);
        }
    };
    let pair = |e1: u32, e2: u32, out: &mut Vec<u32>| match (picked(e1), picked(e2)) {
        (true, true) => out.push(e1.max(e2)),
        (true, false) => out.push(e1),
        (false, true) => out.push(e2),
        _ => {}
    };
    for i in issues {
        match *i {
            Issue::Crossing { e1, e2, .. } => pair(e1, e2, &mut out),
            Issue::Geom { edge: e, .. }
            | Issue::GroundOverWater { edge: e, .. }
            | Issue::WaterTooLong { edge: e }
            | Issue::NoSuchEdge { edge: e }
            | Issue::StationTrack { edge: e }
            | Issue::EdgeLoop { edge: e }
            | Issue::Tracks { edge: e }
            | Issue::NodeHeading { edge: e, .. }
            | Issue::SplitPoint { edge: e }
            | Issue::Crossover { edge: e }
            | Issue::NoParallel { edge: e }
            | Issue::Alongside { edge: e }
            | Issue::HighSpeed { edge: e } => edge(e, &mut out),
            Issue::NoSuchNode { node }
            | Issue::TooManyPorts { node }
            | Issue::NodeOneSided { node }
            | Issue::TrackFamily { node }
            | Issue::NodeLevel { node }
            | Issue::NodeInUse { node }
            | Issue::StationPorts { node }
            | Issue::StationJunction { node }
            | Issue::PlatformLength { node }
            | Issue::PlatformTooLong { node }
            | Issue::PlatformStation { node }
            | Issue::PlatformJunction { node }
            | Issue::PlatformJoint { node }
            | Issue::PlatformNotLevel { node }
            | Issue::PlatformCrossing { node }
            | Issue::StationGroundOverWater { node }
            | Issue::StationInUse { node, .. }
            | Issue::BranchNoRoom { node }
            | Issue::NotAtPort { node }
            | Issue::PortBerth { node } => at_node(node, &mut out),
            _ => {}
        }
    }
    for &(e1, e2) in flats {
        pair(e1, e2, &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// What changes with a piece's family (T-3088): each station on track that changes family gets a
/// platform its new kind allows (cut to a tram stop's longest, or out to a rail station's shortest),
/// and each line on it loses high-speed trains (rail only) and takes the new kind's dwell and
/// turnaround where it had the old kind's.
fn family_ops(net: &Network, cand: &[u32], w: Want) -> Vec<Op> {
    let Some(to) = w.kind else { return vec![] };
    let spec = to.spec();
    let moved = |e: u32| cand.binary_search(&e).is_ok() && net.edge_data(e).is_some_and(|d| kind_ok(&d)) && net.edge_kind[e as usize].spec().family != spec.family;
    let mut ops = vec![];
    let mut seen = vec![];
    for &e in cand {
        if !moved(e) {
            continue;
        }
        for n in [net.edge_a[e as usize], net.edge_b[e as usize]] {
            if seen.contains(&n) || net.node_platform[n as usize] == 0 {
                continue;
            }
            seen.push(n);
            if let Some(mut s) = net.station_data(n) {
                let p = s.platform.clamp(spec.platform.0, spec.platform.1);
                if p != s.platform {
                    s.platform = p;
                    ops.push(Op::Station { node: n, data: Some(s) });
                }
            }
        }
    }
    for l in 0..net.line_count() as u32 {
        if !net.line_ok(l) || net.line_bus[l as usize].is_some() || net.line_ferry[l as usize].is_some() || !net.line_path(l).iter().any(|&p| moved(p >> 1)) {
            continue;
        }
        let Some(mut d) = net.line_data(l) else { continue };
        let from = d.stops.first().map_or(TrackKind::Regular, |&n| net.node_kind(n)).spec();
        let mut changed = false;
        if d.fast && spec.family != 0 {
            d.fast = false;
            changed = true;
        }
        if d.dwell_s == from.dwell_s && d.turnaround_s == from.turnaround_s && (d.dwell_s, d.turnaround_s) != (spec.dwell_s, spec.turnaround_s) {
            (d.dwell_s, d.turnaround_s) = (spec.dwell_s, spec.turnaround_s);
            changed = true;
        }
        if changed {
            ops.push(Op::Line { id: l, data: Some(d) });
        }
    }
    ops
}

/// The pieces of `cand` joined through moving nodes, each with its ops (the nodes' included, and those of
/// the track beyond a station at its ends, T-3241).
fn groups(net: &Network, cand: &[u32], w: Want, bey: &Beyond) -> Vec<(Vec<u32>, Vec<Op>)> {
    let mut up: Vec<usize> = (0..cand.len()).collect();
    fn root(up: &mut [usize], mut i: usize) -> usize {
        while up[i] != i {
            up[i] = up[up[i]];
            i = up[i];
        }
        i
    }
    let mut nodes: Vec<u32> = vec![];
    if w.level.is_some() {
        nodes = cand.iter().flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]]).filter(|&n| movable(net, cand, w, n)).collect();
        nodes.sort_unstable();
        nodes.dedup();
    }
    for &n in &nodes {
        let at: Vec<usize> = net.node_ports[n as usize].ends().iter().filter_map(|&c| cand.binary_search(&(c >> 1)).ok()).collect();
        for p in at.windows(2) {
            let (a, b) = (root(&mut up, p[0]), root(&mut up, p[1]));
            up[a] = b;
        }
    }
    let node_op = |n: u32| {
        let level = w.level?;
        net.node_data(n).filter(|d| d.level != level).map(|d| Op::Node { id: n, data: Some(NodeData { level, ..d }) })
    };
    let mut out: Vec<(Vec<u32>, Vec<Op>)> = vec![];
    let mut of_root = std::collections::HashMap::new();
    for i in 0..cand.len() {
        let r = root(&mut up, i);
        let g = *of_root.entry(r).or_insert_with(|| {
            out.push((vec![], vec![]));
            out.len() - 1
        });
        out[g].0.push(cand[i]);
        out[g].1.extend(edge_op(net, w, cand[i]));
    }
    for &n in &nodes {
        // a node moving with track beyond the pick has track of the pick's too: it is an end of a piece
        let Some(i) = net.node_ports[n as usize].ends().iter().find_map(|&c| cand.binary_search(&(c >> 1)).ok()) else { continue };
        let g = of_root[&root(&mut up, i)];
        out[g].1.extend(node_op(n));
    }
    for r in &bey.runs {
        if let Ok(i) = cand.binary_search(&r.anchor) {
            let g = of_root[&root(&mut up, i)];
            out[g].1.extend(r.ops.iter().cloned());
        }
    }
    out.retain(|g| !g.1.is_empty());
    out
}

impl TrackApi {
    /// Put `ids` on `level` and/or make them track `kind`, as `set_edges_level` describes. `built_ok`:
    /// constructed track may change too, and a constructed station among `picked_stations` moves with
    /// its track. Nothing is applied.
    ///
    /// The whole pick is tried at once; a refusal blames the pieces its problems name (T-3087), which
    /// leave the pick, so the rest still change whole and ramp only to them. Pieces with corners
    /// refused that way are tried last on their own, ends kept. Problems that name no piece fall back
    /// to trying the groups of pieces joined through moving nodes one by one.
    ///
    /// T-3241: a level change takes the stations on the picked track with it, picked or not, and a station
    /// at an end of the pick ramps on the track beyond it (`beyond`); one whose ramp has no room there stays,
    /// and the picked piece ramps to it as before.
    pub(super) fn plan_edges(&mut self, ids: &[u32], level: Option<i8>, kind: Option<TrackKind>, built_ok: bool, picked_stations: &[u32]) -> Plan {
        let mut stations = picked_stations.to_vec();
        stations.sort_unstable();
        let w = Want { level, kind, stations: &stations, ext: &[] };
        let net = &self.w.net;
        let mut want: Vec<u32> = ids.iter().copied().filter(|&e| net.edge_ok(e)).collect();
        want.sort_unstable();
        want.dedup();
        let mut cand = want.clone();
        cand.retain(|&e| {
            (built_ok || !net.edge_built[e as usize]) && !net.edge_crossover[e as usize] && (level.is_some() || net.edge_data(e).is_some_and(|d| kind_ok(&d)))
        });
        let mut refused: Vec<u32> = want.iter().copied().filter(|e| cand.binary_search(e).is_err()).collect();
        let mut root = refused.clone();
        let families = kind.is_some_and(|k| cand.iter().any(|&e| net.edge_kind[e as usize].spec().family != k.spec().family));
        let flat_key = |c: &crate::track::net::Crossing| (c.e1.min(c.e2), c.e1.max(c.e2), c.x.round() as i64, c.y.round() as i64);
        let flat = |net: &Network| -> Vec<(u32, u32, i64, i64)> {
            let mut v: Vec<_> = net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))).map(flat_key).collect();
            v.sort_unstable();
            v
        };
        let before = flat(net);
        // the track model accepts `ops` with no new flat crossing; else its problems and the new crossings
        let try_ops = |tw: &mut TrackWorld, ops: Vec<Op>| -> Result<(), (Vec<Issue>, Vec<(u32, u32)>)> {
            match tw.trial_light(Op::Batch(ops), |tw| flat(&tw.net)) {
                Ok((_, after)) => {
                    let new: Vec<(u32, u32)> = after.iter().filter(|k| before.binary_search(k).is_err()).map(|k| (k.0, k.1)).collect();
                    if new.is_empty() { Ok(()) } else { Err((vec![], new)) }
                }
                Err(e) => Err((e, vec![])),
            }
        };
        let mut last_issues = vec![];
        let mut take = vec![];
        // pieces refused by the track model, to try alone with their ends kept
        let mut todo: Vec<u32> = vec![];
        let mut clash_at = None;
        // T-3241: the stations moving with the pick and the track beyond them, worked out again each round
        let mut bey = Beyond::default();
        let mut both: Vec<u32> = stations.clone();
        // stations with no room for their ramp beyond, and the problems a ramp beyond was refused for
        let mut no_room: Vec<(u32, u32)> = vec![];
        let mut why: Vec<(u32, Vec<Issue>)> = vec![];
        for round in 0.. {
            loop {
                bey = beyond(&self.w.net, &cand, level, built_ok, families);
                for x in &bey.no_room {
                    if !no_room.contains(x) {
                        no_room.push(*x);
                    }
                }
                both = stations.iter().chain(bey.stations.iter()).copied().collect();
                both.sort_unstable();
                both.dedup();
                let n0 = cand.len();
                let wb = Want { level, kind, stations: &both, ext: &bey.nodes };
                prune(&self.w.net, &mut cand, wb, &mut refused, &mut root, &mut clash_at);
                if cand.len() == n0 {
                    break;
                }
            }
            let wb = Want { level, kind, stations: &both, ext: &bey.nodes };
            let gs = groups(&self.w.net, &cand, wb, &bey);
            if gs.is_empty() {
                break;
            }
            let mut all: Vec<Op> = gs.iter().flat_map(|g| g.1.iter().cloned()).collect();
            let extra = family_ops(&self.w.net, &cand, wb);
            all.extend(extra.iter().cloned());
            let (issues, flats) = match try_ops(&mut self.w, all.clone()) {
                Ok(()) => {
                    take = all;
                    break;
                }
                Err(x) => x,
            };
            if !issues.is_empty() {
                last_issues = issues.clone();
            }
            let blamed = culprits(&self.w.net, &cand, &issues, &flats, &bey);
            if !bey.runs.is_empty() && !issues.is_empty() {
                let own = culprits(&self.w.net, &cand, &issues, &flats, &Beyond::default());
                for r in &bey.runs {
                    if blamed.binary_search(&r.anchor).is_ok() && own.binary_search(&r.anchor).is_err() {
                        why.push((r.station, issues.clone()));
                    }
                }
            }
            if !blamed.is_empty() && round < 200 {
                cand.retain(|e| blamed.binary_search(e).is_err());
                if families {
                    refused.extend(&blamed);
                    root.extend(&blamed);
                    refused.sort_unstable();
                    root.sort_unstable();
                } else {
                    todo.extend(&blamed);
                }
                continue;
            }
            // nothing to blame: the groups one by one, as far as they fit
            if families {
                todo.clear();
                refused.extend(&cand);
                root.extend(&cand);
                break;
            }
            for (edges, ops) in gs {
                let mut more = take.clone();
                more.extend(ops.iter().cloned());
                match try_ops(&mut self.w, more.clone()) {
                    Ok(()) => take = more,
                    Err((issues, _)) => {
                        if !issues.is_empty() {
                            last_issues = issues;
                        }
                        todo.extend(edges);
                    }
                }
            }
            break;
        }
        // alone with the ends kept, twice, since a ramp may fit only once its neighbour has changed;
        // not across families, where a piece cannot change without its neighbours
        let net = &self.w.net;
        let solo: Vec<(u32, Op)> = todo
            .iter()
            .filter_map(|&e| match (w.level.is_some() && !cornerless(net, e) && !families, edge_op(net, w, e)) {
                (true, Some(op)) => Some((e, op)),
                _ => {
                    refused.push(e);
                    root.push(e);
                    None
                }
            })
            .collect();
        let mut todo = solo;
        for _ in 0..2 {
            if todo.is_empty() {
                break;
            }
            let mut rest = vec![];
            for (e, op) in todo {
                let mut more = take.clone();
                more.push(op.clone());
                match try_ops(&mut self.w, more.clone()) {
                    Ok(()) => take = more,
                    Err((issues, _)) => {
                        if !issues.is_empty() {
                            last_issues = issues;
                        }
                        rest.push((e, op));
                    }
                }
            }
            todo = rest;
        }
        for (e, _) in todo {
            refused.push(e);
            root.push(e);
        }
        refused.sort_unstable();
        refused.dedup();
        root.sort_unstable();
        root.dedup();
        if last_issues.is_empty() {
            if let Some(node) = clash_at {
                last_issues = vec![Issue::TrackFamily { node }];
            }
        }
        let moved_node = |n: u32| take.iter().any(|op| matches!(op, Op::Node { id, data: Some(_) } if *id == n));
        // a station that stayed for want of room for its ramp beyond says so, rather than that its
        // platform would be on the ramp the picked piece then needs
        let stayed = last_issues.iter().find_map(|i| match *i {
            Issue::PlatformNotLevel { node } if !moved_node(node) => Some(node),
            _ => None,
        });
        if let Some(n) = stayed {
            if let Some((_, is)) = why.iter().rev().find(|x| x.0 == n) {
                last_issues = is.clone();
            } else if let Some(&(_, e)) = no_room.iter().find(|x| x.0 == n) {
                last_issues = vec![Issue::Geom { edge: e, issue: GeomIssue::RampTooShort(0) }];
            }
        } else if last_issues.is_empty() && !refused.is_empty() {
            // a piece with no corners refused at a station that could not move has no problem of its own
            if let Some(&(_, e)) = no_room.iter().find(|x| !moved_node(x.0)) {
                last_issues = vec![Issue::Geom { edge: e, issue: GeomIssue::RampTooShort(0) }];
            }
        }
        let beyond: Vec<(u32, u32, f64)> = bey.runs.iter().filter(|r| moved_node(r.station)).flat_map(|r| r.edges.iter().copied()).collect();
        let moved: Vec<u32> = both.iter().copied().filter(|&n| moved_node(n)).collect();
        Plan { take, refused, root, issues: last_issues, beyond, moved }
    }

    /// The edges and stations of a rebuild pick that are constructed (a blueprint piece is not rebuilt
    /// but built: Construct), the level (None: kept) and the track type (None: kept) the numbers name.
    fn rebuild_pick(&self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> (Vec<u32>, Vec<u32>, Option<i8>, Option<TrackKind>) {
        let net = &self.w.net;
        let edges = edges.iter().copied().filter(|&e| net.edge_ok(e) && net.edge_built[e as usize]).collect();
        let stations = stations.iter().copied().filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && net.node_built[n as usize]).collect();
        let kind = match kind {
            1 => Some(TrackKind::Regular),
            2 => Some(TrackKind::HighSpeed),
            3 => Some(TrackKind::Tram),
            _ => None,
        };
        (edges, stations, (-3..=3).contains(&level).then_some(level as i8), kind)
    }

    /// What a rebuild changes: its plan, the edges it changes, and the stations it charges for (those in
    /// the pick or moving with it whose level or track type changes, and any whose track changes family).
    fn rebuild_plan(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> (Plan, Vec<u32>, Vec<u32>) {
        let (edges, stations, level, kind) = self.rebuild_pick(edges, stations, level, kind);
        let plan = self.plan_edges(&edges, level, kind, true, &stations);
        let net = &self.w.net;
        // an edge with no corners has no level of its own: its nodes' levels are its profile; the track
        // beyond the pick that a station's ramp takes is charged for the stretch that changes
        let outside = |e: u32| plan.beyond.iter().any(|x| x.0 == e);
        let mut changed: Vec<u32> = vec![];
        for op in &plan.take {
            match op {
                Op::Edge { id, .. } => changed.push(*id),
                Op::Node { id, data: Some(_) } => changed.extend(net.node_ports[*id as usize].ends().iter().map(|&c| c >> 1)),
                _ => {}
            }
        }
        changed.retain(|&e| !outside(e));
        changed.sort_unstable();
        changed.dedup();
        let mut stations = stations;
        stations.extend(plan.moved.iter().copied().filter(|&n| net.node_built[n as usize]));
        stations.sort_unstable();
        stations.dedup();
        let retyped: Vec<u32> = plan
            .take
            .iter()
            .filter_map(|op| match op {
                Op::Edge { id, data: Some(d) } if d.kind != net.edge_kind[*id as usize] => Some(*id),
                _ => None,
            })
            .collect();
        let refamily = |n: u32| net.node_ports[n as usize].ends().iter().any(|&c| retyped.contains(&(c >> 1)) && kind.is_some_and(|k| k.spec().family != net.edge_kind[(c >> 1) as usize].spec().family));
        let mut charged: Vec<u32> = stations
            .into_iter()
            .filter(|&n| {
                plan.take.iter().any(|op| matches!(op, Op::Node { id, data: Some(_) } if *id == n))
                    || net.node_ports[n as usize].ends().iter().any(|&c| retyped.contains(&(c >> 1)))
            })
            .collect();
        let mut ends: Vec<u32> = retyped.iter().flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]]).collect();
        ends.sort_unstable();
        ends.dedup();
        charged.extend(ends.into_iter().filter(|&n| net.node_platform[n as usize] > 0 && net.node_built[n as usize] && refamily(n)));
        charged.sort_unstable();
        charged.dedup();
        (plan, changed, charged)
    }
}

/// The full price of building `edges` and `stations` new as they stand, and of the constructed stretches
/// `beyond` (edge, the node it starts at, metres), US$M.
fn full_price(w: &TrackWorld, edges: &[u32], stations: &[u32], beyond: &[(u32, u32, f64)]) -> f64 {
    edges.iter().map(|&e| w.net.edge_cost[e as usize]).sum::<f64>()
        + stations.iter().map(|&n| w.net.station_build_price(n)).sum::<f64>()
        + beyond.iter().map(|&(e, from, m)| stretch_price(&w.net, e, from, m)).sum::<f64>()
}

/// T-3241: what the first `m` metres of constructed edge `e` from node `from` cost built new as they stand,
/// US$M: its track over that stretch (as `track_cost_t` prices the whole edge) and the road crossings on it.
fn stretch_price(net: &Network, e: u32, from: u32, m: f64) -> f64 {
    let i = e as usize;
    if !net.edge_ok(e) || !net.edge_built[i] {
        return 0.0;
    }
    let len = net.edge_len[i];
    if m >= len - 1e-6 {
        return net.edge_cost[i];
    }
    // the stretch as [0, m] of a copy shifted so that it starts at 0
    let off = if net.edge_a[i] == from { 0.0 } else { len - m };
    let vert: Vec<[f64; 2]> = net.edge_vert(e).iter().map(|p| [p[0] - off, p[1]]).collect();
    let water: Vec<[f64; 2]> = net.edge_water(e).iter().map(|p| [p[0] - off, p[1] - off]).collect();
    let terrain: Vec<[f64; 3]> = net.edge_terrain(e).iter().map(|p| [p[0] - off, p[1] - off, p[2]]).collect();
    let track = crate::track::cost::track_cost_t(m, &vert, &water, &terrain, net.edge_tracks[i], net.edge_kind[i]);
    let roads: Vec<[f64; 2]> = net.edge_roads(e).iter().copied().filter(|c| c[0] >= off && c[0] <= off + m).collect();
    track + crate::track::roads::charge(net.edge_vert(e), &roads).1
}

#[wasm_bindgen]
impl TrackApi {
    /// What rebuilding constructed track in place would do (T-3059), nothing applied: the picked
    /// constructed `edges` and `stations` on `level` (outside -3 to 3: kept) and as track type `kind`
    /// (1 regular, 2 high speed, 3 tram, else kept). Returns `[price US$M, edges changed, stations charged,
    /// edges refused, of those refused for their own sake, ...the refused ids (those first), ...the changed
    /// ids]`; the price is the full price of the changed track and stations built new, nothing back for the
    /// old (`issue_kinds()` says why the last refusal was). A refused piece not refused for its own sake
    /// would change family alone (T-3088). T-3241: then, for the track outside the pick that the ramps of
    /// the stations at its ends take, `edge, the node its change starts at, metres changed` each (in the
    /// price, not among the changed ids).
    pub fn rebuild_quote(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> Vec<f64> {
        let (plan, changed, charged) = self.rebuild_plan(edges, stations, level, kind);
        let price = if plan.take.is_empty() {
            0.0
        } else {
            self.w.trial_light(Op::Batch(plan.take.clone()), |w| full_price(w, &changed, &charged, &plan.beyond)).map_or(0.0, |r| r.1)
        };
        self.money_short = None;
        self.issues = plan.issues;
        let mut out = vec![price, changed.len() as f64, charged.len() as f64, plan.refused.len() as f64, plan.root.len() as f64];
        let rest = plan.refused.iter().filter(|e| plan.root.binary_search(e).is_err());
        out.extend(plan.root.iter().chain(rest).chain(changed.iter()).map(|&e| e as f64));
        for &(e, from, m) in &plan.beyond {
            out.extend([e as f64, from as f64, m]);
        }
        out
    }

    /// Rebuild constructed track in place (T-3059): `rebuild_quote`'s pieces changed in one final edit
    /// (no undo), paid at the full price of what changed. The edges and stations keep their ids, so
    /// lines, names and upgrades stay. Pieces the track model refuses are left as they are and counted in
    /// `last_skipped`. 1 when nothing changed or the cash is short (`money_short`).
    pub fn rebuild_pieces(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> u32 {
        let (plan, changed, charged) = self.rebuild_plan(edges, stations, level, kind);
        self.skipped = plan.refused.len() as u32;
        if plan.take.is_empty() {
            self.money_short = None;
            self.issues = plan.issues;
            return 1;
        }
        let fleet = self.fleet;
        let at = self.edit_start();
        let before = self.w.svc.before_edit(&self.w.net, at);
        let cash = self.spendable();
        let beyond = plan.beyond;
        let r = self.w.apply_priced(Op::Batch(plan.take), cash, true, &|w| trains_to_buy(w, fleet), Some(&|w: &TrackWorld| full_price(w, &changed, &charged, &beyond)));
        self.settle(r, &before)
    }
}
