//! Station links, deleting, constructing and what removing track costs.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- station links (T-197): stations the player joins into one complex
    /// Link two stations into one complex. 0 done (or already linked); otherwise one of them is
    /// not a station, or they are the same. Not undoable (a setting, like fares).
    pub fn link_stations(&mut self, a: u32, b: u32) -> u32 {
        if self.w.net.link(a, b) { 0 } else { self.refuse(Issue::NoSuchNode { node: if self.w.net.is_station(a) { b } else { a } }) }
    }
    /// Undo a link between two stations. 0 done; 1 if they were not linked.
    pub fn unlink_stations(&mut self, a: u32, b: u32) -> u32 {
        if self.w.net.unlink(a, b) { 0 } else { 1 }
    }
    /// The links whose two ends are stations now: node pairs, flat.
    pub fn station_links(&self) -> Vec<u32> {
        self.w.net.links_live().flatten().collect()
    }

    /// Remove an edge, and nodes left with no track and no station. Lines using it reroute or
    /// break. Constructed: final, no refund (SPEC 6.4).
    pub fn delete_edge(&mut self, id: u32) -> u32 {
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        // a station the removal would leave at a bare end keeps its place (T-1161); where the
        // trimmed removal is refused, the whole edge goes as before
        if let Some((ops, nodes, edges)) = self.station_trim_ops(id) {
            let r = if self.w.net.edge_built[id as usize] { self.run_charged(Op::Batch(ops)) } else { self.run(Op::Batch(ops)) };
            if r == 0 {
                return 0;
            }
            nodes.into_iter().for_each(|n| self.w.net.release_node(n));
            edges.into_iter().for_each(|e| self.w.net.release_edge(e));
        }
        let net = &self.w.net;
        let mut ops = vec![Op::DeleteEdge { id }];
        let (a, b) = (net.edge_a[id as usize], net.edge_b[id as usize]);
        for n in [a, b] {
            if net.node_ports[n as usize].n == 1 && net.node_platform[n as usize] == 0 {
                ops.push(Op::Node { id: n, data: None });
            }
        }
        // a crossover goes as a whole: both diagonals (T-338)
        let partner = crossover_partner(net, id);
        if let Some(p) = partner {
            ops.push(Op::DeleteEdge { id: p });
        }
        if net.edge_built[id as usize] || partner.is_some_and(|p| net.edge_built[p as usize]) {
            self.run_charged(Op::Batch(ops))
        } else {
            self.run(Op::Batch(ops))
        }
    }

    /// Flat or flying junction. On a constructed junction the flyover is paid at once (final).
    pub fn set_flying(&mut self, node: u32, flying: bool) -> u32 {
        let Some(mut d) = self.w.net.node_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
        let net = &self.w.net;
        if let Some(&c) = net.node_ports[node as usize].ends().iter().find(|&&c| net.edge_crossover[(c >> 1) as usize]) {
            return self.refuse(Issue::Crossover { edge: c >> 1 });
        }
        d.flying = flying;
        let op = Op::Node { id: node, data: Some(d) };
        if self.w.net.built_ports(node) >= 3 {
            self.run_charged(op)
        } else {
            self.run(op)
        }
    }

    /// Construct blueprint edges and stations: pay and build (final). Already constructed ones
    /// are skipped. Refused if there is nothing to build or not enough cash.
    pub fn construct(&mut self, edges: &[u32], stations: &[u32]) -> u32 {
        let ops = self.construct_ops(edges, stations);
        if ops.is_empty() {
            return self.refuse(Issue::NothingToConstruct);
        }
        self.run_charged(Op::Batch(ops))
    }


    /// What a line still needs constructed costs, US$M.
    pub fn line_build_cost(&mut self, line: u32) -> f64 {
        if !self.w.net.line_ok(line) {
            return 0.0;
        }
        let edges: Vec<u32> = self.w.net.line_path(line).iter().map(|&p| p >> 1).collect();
        let stops = self.w.net.line_stops(line).to_vec();
        self.construct_cost(&edges, &stops)
    }

    pub(super) fn construct_ops(&self, edges: &[u32], stations: &[u32]) -> Vec<Op> {
        let net = &self.w.net;
        let mut ops = vec![];
        // a crossover is built as a whole: both diagonals (T-391)
        let mut all = edges.to_vec();
        for &e in edges {
            if let Some(p) = crossover_partner(net, e) {
                if !all.contains(&p) {
                    all.push(p);
                }
            }
        }
        for &e in &all {
            if let Some(mut d) = net.edge_data(e) {
                if !d.built {
                    d.built = true;
                    ops.push(Op::Edge { id: e, data: Some(d) });
                }
            }
        }
        for &n in stations {
            if let Some(mut d) = net.station_data(n) {
                if !d.built {
                    d.built = true;
                    ops.push(Op::Station { node: n, data: Some(d) });
                }
            }
        }
        ops
    }

    /// The blueprint edges and stations switched off (T-333), as two lists of ids.
    pub fn off_edges(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_off[e as usize]).collect()
    }
    pub fn off_stations(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && net.node_off[n as usize]).collect()
    }

    /// Construct the whole blueprint except what is switched off (T-333).
    pub fn construct_all(&mut self) -> u32 {
        let net = &self.w.net;
        let edges: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && !net.edge_built[e as usize] && !net.edge_off[e as usize]).collect();
        let stations: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && !net.node_built[n as usize] && !net.node_off[n as usize]).collect();
        self.construct(&edges, &stations)
    }

    /// Switch blueprint edges and stations off or on (T-333): Construct and the blueprint quote
    /// skip what is off. One undoable edit; constructed pieces and ones already in that state are
    /// left alone. Returns 1 when nothing changed.
    pub fn set_blueprint_off(&mut self, edges: &[u32], stations: &[u32], off: bool) -> u32 {
        let net = &self.w.net;
        let mut ops = vec![];
        for &e in edges {
            if let Some(mut d) = net.edge_data(e) {
                if !d.built && d.off != off {
                    d.off = off;
                    ops.push(Op::Edge { id: e, data: Some(d) });
                }
            }
        }
        for &n in stations {
            if let Some(mut d) = net.station_data(n) {
                if !d.built && d.off != off {
                    d.off = off;
                    ops.push(Op::Station { node: n, data: Some(d) });
                }
            }
        }
        if ops.is_empty() {
            self.money_short = None;
            self.issues = vec![];
            return 1;
        }
        self.run(Op::Batch(ops))
    }

    /// Put several blueprint edges on one level (T-815), one undoable edit. Each piece's corners go
    /// to the level as `set_edge_level` does for one, and so does every node all of whose track is
    /// among them (a joint, a loose end, a junction, a blueprint station), so a picked run has no
    /// ramp at its joints; a node with other track keeps its level and the piece ramps to it.
    /// Pieces that cannot change are left out and counted in `last_skipped`: constructed track, a
    /// crossover's diagonal, a piece with no corners whose ends cannot both move, and pieces the
    /// track model refuses on that level or that would make a flat crossing not there now. Those
    /// are tried as groups (pieces joined through moving nodes), then a refused group's pieces with
    /// corners one by one with their ends kept. Pieces already on the level are not counted. 1 when
    /// nothing changed (the issues of the last refusal, if any).
    pub fn set_edges_level(&mut self, ids: &[u32], level: i8) -> u32 {
        let net = &self.w.net;
        let mut cand: Vec<u32> = ids.iter().copied().filter(|&e| net.edge_ok(e)).collect();
        cand.sort_unstable();
        cand.dedup();
        let total = cand.len();
        cand.retain(|&e| !net.edge_built[e as usize] && !net.edge_crossover[e as usize]);
        let movable = |cand: &[u32], n: u32| {
            net.node_ports[n as usize].ends().iter().all(|&c| cand.binary_search(&(c >> 1)).is_ok()) && !(net.node_platform[n as usize] > 0 && net.node_built[n as usize])
        };
        let cornerless = |e: u32| net.edge_data(e).is_some_and(|d| d.pis.is_empty() && d.thru.is_empty());
        loop {
            let was = cand.clone();
            cand.retain(|&e| !cornerless(e) || (movable(&was, net.edge_a[e as usize]) && movable(&was, net.edge_b[e as usize])));
            if cand.len() == was.len() {
                break;
            }
        }
        let mut skipped = (total - cand.len()) as u32;
        // groups joined through moving nodes (union-find over positions in `cand`)
        let mut up: Vec<usize> = (0..cand.len()).collect();
        fn root(up: &mut [usize], mut i: usize) -> usize {
            while up[i] != i {
                up[i] = up[up[i]];
                i = up[i];
            }
            i
        }
        let mut nodes: Vec<u32> = cand.iter().flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]]).filter(|&n| movable(&cand, n)).collect();
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            let at: Vec<usize> = net.node_ports[n as usize].ends().iter().filter_map(|&c| cand.binary_search(&(c >> 1)).ok()).collect();
            for w in at.windows(2) {
                let (a, b) = (root(&mut up, w[0]), root(&mut up, w[1]));
                up[a] = b;
            }
        }
        let edge_op = |e: u32| {
            let mut d = net.edge_data(e)?;
            if d.pis.iter().chain(d.thru.iter()).all(|p| p.level == level) {
                return None;
            }
            for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
                p.level = level;
            }
            Some(Op::Edge { id: e, data: Some(d) })
        };
        let node_op = |n: u32| net.node_data(n).filter(|d| d.level != level).map(|d| Op::Node { id: n, data: Some(NodeData { level, ..d }) });
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
        if groups.is_empty() {
            self.skipped = skipped;
            self.money_short = None;
            self.issues.clear();
            return 1;
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
        let fits = |w: &mut TrackWorld, ops: Vec<Op>, last_issues: &mut Vec<Issue>| match w.trial_with(Op::Batch(ops), |w| flat(&w.net)) {
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
            let mut refused = vec![];
            for (edges, ops) in groups {
                let mut more = take.clone();
                more.extend(ops.iter().cloned());
                if fits(&mut self.w, more.clone(), &mut last_issues) {
                    take = more;
                } else {
                    refused.extend(edges);
                }
            }
            // a refused group's pieces with corners one by one, their ends kept, twice since a
            // ramp may fit only once its neighbour has changed
            let mut todo: Vec<(u32, Op)> = vec![];
            for e in refused {
                match self.w.net.edge_data(e).filter(|d| !(d.pis.is_empty() && d.thru.is_empty())) {
                    Some(mut d) if d.pis.iter().chain(d.thru.iter()).any(|p| p.level != level) => {
                        for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
                            p.level = level;
                        }
                        todo.push((e, Op::Edge { id: e, data: Some(d) }));
                    }
                    _ => skipped += 1,
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
            skipped += todo.len() as u32;
        }
        if take.is_empty() {
            self.skipped = skipped;
            self.money_short = None;
            self.issues = last_issues;
            return 1;
        }
        let r = self.run(Op::Batch(take));
        self.skipped = skipped;
        r
    }

    /// How many pieces the last `set_edges_level` left out (T-815).
    pub fn last_skipped(&self) -> u32 {
        self.skipped
    }

    /// Remove several edges and stations at once (T-829): one edit, undoable when all of it is
    /// blueprint, else final with no refund, as `delete_edge` and `remove_station` are for one. A
    /// crossover goes whole; nodes left with no track and no station go too. A line that stops at
    /// a removed station loses that stop and is routed again through the rest (T-859), else left
    /// broken, as removed track leaves it.
    pub fn delete_pieces(&mut self, edges: &[u32], stations: &[u32]) -> u32 {
        let Some((ops, built, st)) = self.pieces_ops(edges, stations) else {
            return self.refuse(Issue::NoSuchEdge { edge: edges.first().copied().unwrap_or(u32::MAX) });
        };
        let r = if built { self.run_charged(Op::Batch(ops)) } else { self.run(Op::Batch(ops)) };
        if r == 0 {
            self.w.net.links.retain(|&[a, b]| st.binary_search(&a).is_err() && st.binary_search(&b).is_err());
        }
        r
    }

    /// What `delete_pieces` would do to lines (T-859), tried and rolled back: per line that stops
    /// at a removed station or would be left broken (and was not), three numbers: the line, 1 if
    /// it loses a stop, 1 if it would be broken.
    pub fn removal_lines(&mut self, edges: &[u32], stations: &[u32]) -> Vec<f64> {
        let Some((ops, _, st)) = self.pieces_ops(edges, stations) else { return vec![] };
        let net = &self.w.net;
        let lines: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l)).collect();
        let was: Vec<bool> = lines.iter().map(|&l| net.line_broken[l as usize]).collect();
        let stopping: Vec<bool> = lines.iter().map(|&l| net.line_stops(l).iter().any(|s| st.binary_search(s).is_ok())).collect();
        let Ok((_, after)) = self.w.trial_with(Op::Batch(ops), |w| lines.iter().map(|&l| w.net.line_broken[l as usize]).collect::<Vec<_>>()) else { return vec![] };
        let mut out = vec![];
        for (i, &l) in lines.iter().enumerate() {
            if stopping[i] || (after[i] && !was[i]) {
                out.extend([l as f64, stopping[i] as u8 as f64, after[i] as u8 as f64]);
            }
        }
        out
    }

    /// The ops of `delete_pieces`, whether any of it is constructed, and the stations removed
    /// (sorted); None when nothing named is there.
    pub(super) fn pieces_ops(&mut self, edges: &[u32], stations: &[u32]) -> Option<(Vec<Op>, bool, Vec<u32>)> {
        let net = &self.w.net;
        let mut gone: Vec<u32> = edges.iter().copied().filter(|&e| net.edge_ok(e)).collect();
        for &e in edges {
            if let Some(p) = crossover_partner(net, e).filter(|&p| net.edge_ok(p)) {
                gone.push(p);
            }
        }
        gone.sort_unstable();
        gone.dedup();
        let mut st: Vec<u32> = stations.iter().copied().filter(|&n| net.station_data(n).is_some()).collect();
        st.sort_unstable();
        st.dedup();
        if gone.is_empty() && st.is_empty() {
            return None;
        }
        let built = gone.iter().any(|&e| net.edge_built[e as usize]) || st.iter().any(|&n| net.node_built[n as usize]);
        let mut ops: Vec<Op> = gone.iter().map(|&id| Op::DeleteEdge { id }).collect();
        let mut tail = vec![];
        let mut nodes: Vec<u32> = gone.iter().flat_map(|&e| [net.edge_a[e as usize], net.edge_b[e as usize]]).chain(st.iter().copied()).collect();
        nodes.sort_unstable();
        nodes.dedup();
        for n in nodes {
            let left = net.node_ports[n as usize].ends().iter().any(|&c| gone.binary_search(&(c >> 1)).is_err());
            let station = net.node_platform[n as usize] > 0 && st.binary_search(&n).is_err();
            if !left && !station {
                tail.push(Op::Node { id: n, data: None });
            }
        }
        // lines stopping at a removed station: routed on the track as it will be, the stop left out,
        // never with a reversal the old path did not make (as DeleteEdge reroutes, T-797)
        let users: Vec<_> = (0..net.line_count() as u32)
            .filter(|&l| net.line_ok(l) && net.line_stops(l).iter().any(|s| st.binary_search(s).is_ok()))
            .filter_map(|l| net.line_data(l).map(|d| (l, net.turn_count(&d.stops, &d.path), d)))
            .collect();
        if !users.is_empty() {
            let inv = self.w.sketch(Op::Batch(ops.clone()), false).ok();
            for (l, turns, mut d) in users {
                let mut stops: Vec<u32> = d.stops.iter().copied().filter(|s| st.binary_search(s).is_err()).collect();
                stops.dedup();
                if let Some(p) = self.w.net.route(&stops).filter(|p| self.w.net.turn_count(&stops, p) <= turns) {
                    d.path = p;
                }
                d.stops = stops;
                ops.push(Op::Line { id: l, data: Some(d) });
            }
            if let Some(inv) = inv {
                self.w.unsketch(inv);
            }
        }
        ops.extend(st.iter().map(|&node| Op::Station { node, data: None }));
        ops.extend(tail);
        let joins = self.station_joins(&ops, &st);
        ops.extend(joins);
        Some((ops, built, st))
    }

    /// T-1299: the ops joining the two pieces left at each removed station's node into one where
    /// they simply join (`unjoin_near`'s rules: on one straight, alike in kind and state), planned
    /// on the track as `ops` leave it. A piece already joined at another station is left as it is.
    fn station_joins(&mut self, ops: &[Op], st: &[u32]) -> Vec<Op> {
        let Ok(inv) = self.w.sketch(Op::Batch(ops.to_vec()), false) else { return vec![] };
        let mut out = vec![];
        let mut used: Vec<u32> = vec![];
        for &n in st {
            let net = &self.w.net;
            if !net.node_ok(n) || net.node_ports[n as usize].n != 2 || net.node_platform[n as usize] > 0 {
                continue;
            }
            let ends: Vec<u32> = net.node_ports[n as usize].ends().iter().map(|&c| c >> 1).collect();
            let e = ends[0];
            if ends.iter().any(|x| used.contains(x)) || net.edge_a[e as usize] == net.edge_b[e as usize] {
                continue;
            }
            let s = if net.edge_a[e as usize] == n { 0.0 } else { net.edge_len[e as usize] };
            if let Some((j, _)) = self.unjoin_near(e, s, 1e-6) {
                used.extend(ends);
                out.extend(j);
            }
        }
        self.w.unsketch(inv);
        out
    }

    /// Construct what a line needs to run: its path's edges and its stops.
    pub fn construct_line(&mut self, line: u32) -> u32 {
        if !self.w.net.line_ok(line) {
            return self.refuse(Issue::NoSuchLine { line });
        }
        let edges: Vec<u32> = self.w.net.line_path(line).iter().map(|&p| p >> 1).collect();
        let stops = self.w.net.line_stops(line).to_vec();
        self.construct(&edges, &stops)
    }

}

/// Track kept past a trimmed station's platform end (T-1161): `split_plan` may snap a split up to
/// 2 m toward a piece boundary, and `check_station` wants 1 m clear of the platform.
pub(super) const STATION_TRIM_MARGIN_M: f64 = 4.0;

impl TrackApi {
    /// `delete_edge` for an edge with a station at an end that has track on both sides (T-1161):
    /// removing the edge would leave that station at a bare end, its platform running back from the
    /// node, so it would move half a platform. The edge is split just past the platform instead and
    /// only the rest removed. The ops with the node and edge ids they claim; None when no end is
    /// such a station or the edge is too short to split.
    fn station_trim_ops(&mut self, id: u32) -> Option<(Vec<Op>, Vec<u32>, Vec<u32>)> {
        let net = &self.w.net;
        let i = id as usize;
        let (a, b, len) = (net.edge_a[i], net.edge_b[i], net.edge_len[i]);
        if a == b || net.edge_crossover[i] {
            return None;
        }
        let keep = |n: u32| (net.node_platform[n as usize] > 0 && net.node_ports[n as usize].n == 2).then(|| net.platform_reach(n) + STATION_TRIM_MARGIN_M);
        let (ka, kb) = (keep(a), keep(b));
        // split_plan keeps 5 m from either end of what it splits, plus a snap of 2 m
        if (ka.is_none() && kb.is_none()) || ka.unwrap_or(0.0) + kb.unwrap_or(0.0) + 7.0 > len {
            return None;
        }
        let lone = |n: u32| net.node_ports[n as usize].n == 1 && net.node_platform[n as usize] == 0;
        let (lone_a, lone_b) = (lone(a), lone(b));
        let (n1, e1) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
        Some(match (ka, kb) {
            (Some(sa), None) => {
                let mut ops = vec![Op::Split { edge: id, s: sa, node: n1, new_edge: e1 }, Op::DeleteEdge { id: e1 }];
                if lone_b { ops.push(Op::Node { id: b, data: None }); }
                (ops, vec![n1], vec![e1])
            }
            (None, Some(sb)) => {
                let mut ops = vec![Op::Split { edge: id, s: len - sb, node: n1, new_edge: e1 }, Op::DeleteEdge { id }];
                if lone_a { ops.push(Op::Node { id: a, data: None }); }
                (ops, vec![n1], vec![e1])
            }
            (Some(sa), Some(sb)) => {
                // the b end first, so the a end's chainage is the same on what is left
                let (n2, e2) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
                let ops = vec![
                    Op::Split { edge: id, s: len - sb, node: n2, new_edge: e2 },
                    Op::Split { edge: id, s: sa, node: n1, new_edge: e1 },
                    Op::DeleteEdge { id: e1 },
                ];
                (ops, vec![n1, n2], vec![e1, e2])
            }
            (None, None) => unreachable!(),
        })
    }
}
