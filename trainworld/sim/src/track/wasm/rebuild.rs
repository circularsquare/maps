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
}

/// The settings a plan puts on its pieces.
#[derive(Clone, Copy)]
struct Want<'a> {
    level: Option<i8>,
    kind: Option<TrackKind>,
    stations: &'a [u32],
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
/// station on it is in the pick.
fn movable(net: &Network, cand: &[u32], w: Want, n: u32) -> bool {
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
/// pieces that cross, the later one.
fn culprits(net: &Network, cand: &[u32], issues: &[Issue], flats: &[(u32, u32)]) -> Vec<u32> {
    let picked = |e: u32| cand.binary_search(&e).is_ok();
    let mut out = vec![];
    let at_node = |n: u32, out: &mut Vec<u32>| {
        if net.node_ok(n) {
            out.extend(net.node_ports[n as usize].ends().iter().map(|&c| c >> 1).filter(|&e| picked(e)));
        }
    };
    let edge = |e: u32, out: &mut Vec<u32>| {
        if picked(e) {
            out.push(e);
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

/// The pieces of `cand` joined through moving nodes, each with its ops (the nodes' included).
fn groups(net: &Network, cand: &[u32], w: Want) -> Vec<(Vec<u32>, Vec<Op>)> {
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
        let first = net.node_ports[n as usize].ends()[0] >> 1;
        let g = of_root[&root(&mut up, cand.binary_search(&first).unwrap())];
        out[g].1.extend(node_op(n));
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
    pub(super) fn plan_edges(&mut self, ids: &[u32], level: Option<i8>, kind: Option<TrackKind>, built_ok: bool, picked_stations: &[u32]) -> Plan {
        let mut stations = picked_stations.to_vec();
        stations.sort_unstable();
        let w = Want { level, kind, stations: &stations };
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
        for round in 0.. {
            prune(&self.w.net, &mut cand, w, &mut refused, &mut root, &mut clash_at);
            let gs = groups(&self.w.net, &cand, w);
            if gs.is_empty() {
                break;
            }
            let mut all: Vec<Op> = gs.iter().flat_map(|g| g.1.iter().cloned()).collect();
            let extra = family_ops(&self.w.net, &cand, w);
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
            let blamed = culprits(&self.w.net, &cand, &issues, &flats);
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
        Plan { take, refused, root, issues: last_issues }
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
    /// the pick whose level or track type changes, and any whose track changes family).
    fn rebuild_plan(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> (Plan, Vec<u32>, Vec<u32>) {
        let (edges, stations, level, kind) = self.rebuild_pick(edges, stations, level, kind);
        let plan = self.plan_edges(&edges, level, kind, true, &stations);
        let net = &self.w.net;
        // an edge with no corners has no level of its own: its nodes' levels are its profile
        let mut changed: Vec<u32> = vec![];
        for op in &plan.take {
            match op {
                Op::Edge { id, .. } => changed.push(*id),
                Op::Node { id, data: Some(_) } => changed.extend(net.node_ports[*id as usize].ends().iter().map(|&c| c >> 1)),
                _ => {}
            }
        }
        changed.sort_unstable();
        changed.dedup();
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

/// The full price of building `edges` and `stations` new as they stand, US$M.
fn full_price(w: &TrackWorld, edges: &[u32], stations: &[u32]) -> f64 {
    edges.iter().map(|&e| w.net.edge_cost[e as usize]).sum::<f64>() + stations.iter().map(|&n| w.net.station_build_price(n)).sum::<f64>()
}

#[wasm_bindgen]
impl TrackApi {
    /// What rebuilding constructed track in place would do (T-3059), nothing applied: the picked
    /// constructed `edges` and `stations` on `level` (outside -3 to 3: kept) and as track type `kind`
    /// (1 regular, 2 high speed, 3 tram, else kept). Returns `[price US$M, edges changed, stations charged,
    /// edges refused, of those refused for their own sake, ...the refused ids (those first), ...the changed
    /// ids]`; the price is the full price of the changed track and stations built new, nothing back for the
    /// old (`issue_kinds()` says why the last refusal was). A refused piece not refused for its own sake
    /// would change family alone (T-3088).
    pub fn rebuild_quote(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> Vec<f64> {
        let (plan, changed, charged) = self.rebuild_plan(edges, stations, level, kind);
        let price = if plan.take.is_empty() {
            0.0
        } else {
            self.w.trial_light(Op::Batch(plan.take.clone()), |w| full_price(w, &changed, &charged)).map_or(0.0, |r| r.1)
        };
        self.money_short = None;
        self.issues = plan.issues;
        let mut out = vec![price, changed.len() as f64, charged.len() as f64, plan.refused.len() as f64, plan.root.len() as f64];
        let rest = plan.refused.iter().filter(|e| plan.root.binary_search(e).is_err());
        out.extend(plan.root.iter().chain(rest).chain(changed.iter()).map(|&e| e as f64));
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
        let r = self.w.apply_priced(Op::Batch(plan.take), cash, true, &|w| trains_to_buy(w, fleet), Some(&|w: &TrackWorld| full_price(w, &changed, &charged)));
        self.settle(r, &before)
    }
}
