//! Changing picked track's level and track type in place (T-815, T-1898), and rebuilding built track that
//! way (T-3059): the same edges, nodes and stations, so lines, names and everything keyed by id stay.

use super::*;

/// What a pick of edges can change: the ops to apply, and the picked edges the track model would refuse.
pub(super) struct Plan {
    pub take: Vec<Op>,
    pub refused: Vec<u32>,
    pub issues: Vec<Issue>,
}

impl TrackApi {
    /// Put `ids` on `level` and/or make them high-speed or regular, as `set_edges_level` and
    /// `set_edges_hsr` do (see the former for the rules). `built_ok`: constructed track may change too,
    /// and a constructed station among `picked_stations` moves with its track. Nothing is applied.
    pub(super) fn plan_edges(&mut self, ids: &[u32], level: Option<i8>, hsr: Option<bool>, built_ok: bool, picked_stations: &[u32]) -> Plan {
        let net = &self.w.net;
        let mut want: Vec<u32> = ids.iter().copied().filter(|&e| net.edge_ok(e)).collect();
        want.sort_unstable();
        want.dedup();
        let kind_ok = |d: &EdgeData| !d.crossover && d.tracks == 2 && d.kind.spec().family == 0;
        let mut cand = want.clone();
        cand.retain(|&e| {
            (built_ok || !net.edge_built[e as usize]) && !net.edge_crossover[e as usize] && (level.is_some() || net.edge_data(e).is_some_and(|d| kind_ok(&d)))
        });
        let mut stations = picked_stations.to_vec();
        stations.sort_unstable();
        let movable = |cand: &[u32], n: u32| {
            net.node_ports[n as usize].ends().iter().all(|&c| cand.binary_search(&(c >> 1)).is_ok())
                && (!(net.node_platform[n as usize] > 0 && net.node_built[n as usize]) || stations.binary_search(&n).is_ok())
        };
        let cornerless = |e: u32| net.edge_data(e).is_some_and(|d| d.pis.is_empty() && d.thru.is_empty());
        if level.is_some() {
            loop {
                let was = cand.clone();
                cand.retain(|&e| !cornerless(e) || (movable(&was, net.edge_a[e as usize]) && movable(&was, net.edge_b[e as usize])));
                if cand.len() == was.len() {
                    break;
                }
            }
        }
        let mut refused: Vec<u32> = want.iter().copied().filter(|e| cand.binary_search(e).is_err()).collect();
        // groups joined through moving nodes (union-find over positions in `cand`)
        let mut up: Vec<usize> = (0..cand.len()).collect();
        fn root(up: &mut [usize], mut i: usize) -> usize {
            while up[i] != i {
                up[i] = up[up[i]];
                i = up[i];
            }
            i
        }
        let mut nodes: Vec<u32> = vec![];
        if level.is_some() {
            nodes = cand.iter().flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]]).filter(|&n| movable(&cand, n)).collect();
            nodes.sort_unstable();
            nodes.dedup();
        }
        for &n in &nodes {
            let at: Vec<usize> = net.node_ports[n as usize].ends().iter().filter_map(|&c| cand.binary_search(&(c >> 1)).ok()).collect();
            for w in at.windows(2) {
                let (a, b) = (root(&mut up, w[0]), root(&mut up, w[1]));
                up[a] = b;
            }
        }
        let edge_op = |e: u32| {
            let mut d = net.edge_data(e)?;
            let mut changed = false;
            if let Some(level) = level {
                if !d.pis.iter().chain(d.thru.iter()).all(|p| p.level == level) {
                    for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
                        p.level = level;
                    }
                    changed = true;
                }
            }
            if let Some(h) = hsr {
                if kind_ok(&d) && d.kind.is_hsr() != h {
                    d.kind = TrackKind::from_hsr(h);
                    changed = true;
                }
            }
            changed.then_some(Op::Edge { id: e, data: Some(d) })
        };
        let node_op = |n: u32| {
            let level = level?;
            net.node_data(n).filter(|d| d.level != level).map(|d| Op::Node { id: n, data: Some(NodeData { level, ..d }) })
        };
        let mut groups: Vec<(Vec<u32>, Vec<Op>)> = vec![];
        let mut of_root = std::collections::HashMap::new();
        for i in 0..cand.len() {
            let r = root(&mut up, i);
            let g = *of_root.entry(r).or_insert_with(|| {
                groups.push((vec![], vec![]));
                groups.len() - 1
            });
            groups[g].0.push(cand[i]);
            groups[g].1.extend(edge_op(cand[i]));
        }
        for &n in &nodes {
            let first = net.node_ports[n as usize].ends()[0] >> 1;
            let g = of_root[&root(&mut up, cand.binary_search(&first).unwrap())];
            groups[g].1.extend(node_op(n));
        }
        groups.retain(|g| !g.1.is_empty());
        // each piece alone with its ends kept, for a refused group's pieces that have corners to ramp at
        let solo: std::collections::HashMap<u32, Op> = if level.is_some() {
            groups
                .iter()
                .flat_map(|g| g.0.iter().copied())
                .filter(|&e| !cornerless(e))
                .filter_map(|e| edge_op(e).map(|op| (e, op)))
                .collect()
        } else {
            Default::default()
        };
        if groups.is_empty() {
            return Plan { take: vec![], refused, issues: vec![] };
        }
        let flat_key = |c: &crate::track::net::Crossing| (c.e1.min(c.e2), c.e1.max(c.e2), c.x.round() as i64, c.y.round() as i64);
        let flat = |net: &Network| -> Vec<(u32, u32, i64, i64)> {
            let mut v: Vec<_> = net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))).map(flat_key).collect();
            v.sort_unstable();
            v
        };
        let before = flat(net);
        let mut last_issues = vec![];
        // the track model accepts `ops` with no new flat crossing
        let fits = |w: &mut TrackWorld, ops: Vec<Op>, last_issues: &mut Vec<Issue>| match w.trial_light(Op::Batch(ops), |w| flat(&w.net)) {
            Ok((_, after)) => after.iter().all(|k| before.binary_search(k).is_ok()),
            Err(e) => {
                *last_issues = e;
                false
            }
        };
        let all: Vec<Op> = groups.iter().flat_map(|g| g.1.iter().cloned()).collect();
        let mut take = vec![];
        if fits(&mut self.w, all.clone(), &mut last_issues) {
            take = all;
        } else {
            let mut gone = vec![];
            for (edges, ops) in groups {
                let mut more = take.clone();
                more.extend(ops.iter().cloned());
                if fits(&mut self.w, more.clone(), &mut last_issues) {
                    take = more;
                } else {
                    gone.extend(edges);
                }
            }
            // twice, since a ramp may fit only once its neighbour has changed
            let mut todo: Vec<(u32, Op)> = vec![];
            for e in gone {
                match solo.get(&e) {
                    Some(op) => todo.push((e, op.clone())),
                    None => refused.push(e),
                }
            }
            for _ in 0..2 {
                let mut rest = vec![];
                for (e, op) in todo {
                    let mut more = take.clone();
                    more.push(op.clone());
                    if fits(&mut self.w, more.clone(), &mut last_issues) {
                        take = more;
                    } else {
                        rest.push((e, op));
                    }
                }
                todo = rest;
                if todo.is_empty() {
                    break;
                }
            }
            refused.extend(todo.into_iter().map(|t| t.0));
        }
        refused.sort_unstable();
        refused.dedup();
        Plan { take, refused, issues: last_issues }
    }

    /// The edges and stations of a rebuild pick that are constructed (a blueprint piece is not rebuilt
    /// but built: Construct), the level (None: kept) and the track type (None: kept) the numbers name.
    fn rebuild_pick(&self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> (Vec<u32>, Vec<u32>, Option<i8>, Option<bool>) {
        let net = &self.w.net;
        let edges = edges.iter().copied().filter(|&e| net.edge_ok(e) && net.edge_built[e as usize]).collect();
        let stations = stations.iter().copied().filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && net.node_built[n as usize]).collect();
        ((edges), stations, (-3..=3).contains(&level).then_some(level as i8), match kind {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        })
    }

    /// What a rebuild changes: its ops, the edges they change, the stations it charges for (those in
    /// the pick whose level or track type changes), and the planner's refusals.
    fn rebuild_plan(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> (Plan, Vec<u32>, Vec<u32>) {
        let (edges, stations, level, hsr) = self.rebuild_pick(edges, stations, level, kind);
        let plan = self.plan_edges(&edges, level, hsr, true, &stations);
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
        let charged: Vec<u32> = stations
            .into_iter()
            .filter(|&n| {
                plan.take.iter().any(|op| matches!(op, Op::Node { id, data: Some(_) } if *id == n))
                    || net.node_ports[n as usize].ends().iter().any(|&c| retyped.contains(&(c >> 1)))
            })
            .collect();
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
    /// (1 regular, 2 high speed, else kept). Returns `[price US$M, edges changed, stations charged, edges
    /// refused, ...the refused edge ids, ...the changed edge ids]`; the price is the full price of the changed track and stations
    /// built new, nothing back for the old (`issue_kinds()` says why the last refusal was).
    pub fn rebuild_quote(&mut self, edges: &[u32], stations: &[u32], level: i32, kind: i32) -> Vec<f64> {
        let (plan, changed, charged) = self.rebuild_plan(edges, stations, level, kind);
        let price = if plan.take.is_empty() {
            0.0
        } else {
            self.w.trial_light(Op::Batch(plan.take.clone()), |w| full_price(w, &changed, &charged)).map_or(0.0, |r| r.1)
        };
        self.money_short = None;
        self.issues = plan.issues;
        let mut out = vec![price, changed.len() as f64, charged.len() as f64, plan.refused.len() as f64];
        out.extend(plan.refused.iter().chain(changed.iter()).map(|&e| e as f64));
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
