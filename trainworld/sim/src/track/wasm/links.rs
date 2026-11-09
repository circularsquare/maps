//! Station links, deleting, constructing and what removing track costs.

use super::*;
use crate::track::upgrades;

#[wasm_bindgen]
impl TrackApi {
    // ---- station links (T-197): stations the player joins into one complex by a passage (T-1281)
    /// Join two stations by a passage, into one complex. Between two constructed stations it is
    /// paid at once, else it is built with the second of them (`Network::passages`). Not undoable
    /// (a setting, like fares). 0 done (or already linked); otherwise one of them is not a
    /// station, they are the same or further apart than a passage reaches, or cash is short.
    pub fn link_stations(&mut self, a: u32, b: u32) -> u32 {
        self.link_charge = 0.0;
        let net = &self.w.net;
        if a == b || !net.is_station(a) || !net.is_station(b) || net.passage_m(a, b) > upgrades::PASSAGE_MAX_M + 1e-6 {
            return self.refuse(Issue::NoSuchNode { node: if net.is_station(a) { b } else { a } });
        }
        if net.links.contains(&[a.min(b), a.max(b)]) {
            return 0;
        }
        let price = if net.node_built[a as usize] && net.node_built[b as usize] { upgrades::passage_price(net.passage_m(a, b)) } else { 0.0 };
        let cash = self.spendable();
        if price > cash + 1e-6 {
            self.issues.clear();
            self.money_short = Some((price, cash));
            return 1;
        }
        self.w.net.link(a, b);
        self.cash -= price;
        self.link_charge = price;
        self.money_short = None;
        self.issues.clear();
        0
    }
    /// What the last `link_stations` paid, US$M.
    pub fn link_charge(&self) -> f64 {
        self.link_charge
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
    /// break. Constructed: no refund, and undoable, the undo free (T-1649).
    pub fn delete_edge(&mut self, id: u32) -> u32 {
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        if self.holds_station(id) {
            return self.refuse(Issue::StationTrack { edge: id });
        }
        // a station the removal would leave at a bare end keeps its place (T-1161); where the
        // trimmed removal is refused, the whole edge goes as before
        if let Some((ops, nodes, edges)) = self.station_trim_ops(id) {
            let r = self.run(Op::Batch(ops));
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
        let gone: Vec<u32> = [Some(id), partner].into_iter().flatten().collect();
        let (keep, nn, ne) = self.keep_under_stations(&gone, &[]);
        let kept = !keep.is_empty();
        let plain = ops.clone();
        ops.extend(keep);
        let r = self.run(Op::Batch(ops));
        if r != 0 {
            self.release(&nn, &ne);
            // T-1895: never refused for the track kept under a station; it goes without
            if kept {
                return self.run(Op::Batch(plain));
            }
        }
        r
    }

    /// T-1699: a station that removing `gone` takes an end from keeps a straight piece under it for
    /// that end, along its old direction, the platform's reach plus 10 m long (as a station on open
    /// ground has; less where the removed piece was shorter), with the removed piece's kind and
    /// state. So the station stays on track in its old orientation, its platform where it was.
    /// `removed`: stations going too (sorted).
    fn keep_under_stations(&mut self, gone: &[u32], removed: &[u32]) -> (Vec<Op>, Vec<u32>, Vec<u32>) {
        let (mut ops, mut nn, mut ne) = (vec![], vec![], vec![]);
        let mut nodes: Vec<u32> = gone.iter().flat_map(|&e| [self.w.net.edge_a[e as usize], self.w.net.edge_b[e as usize]]).collect();
        nodes.sort_unstable();
        nodes.dedup();
        for n in nodes {
            let net = &self.w.net;
            let i = n as usize;
            let p = net.node_ports[i];
            if net.node_platform[i] == 0 || removed.binary_search(&n).is_ok() {
                continue;
            }
            let reach = net.platform_reach(n);
            let (x, y, h, level) = (net.node_x[i], net.node_y[i], net.node_heading[i], net.node_level[i]);
            let ends: Vec<(u32, f64)> = p.ends().iter().enumerate().filter(|&(_, &c)| gone.contains(&(c >> 1))).map(|(k, &c)| (c >> 1, if p.side[k] == 1 { h } else { h + std::f64::consts::PI })).collect();
            for (e, th) in ends {
                let net = &self.w.net;
                let ei = e as usize;
                let far = if net.edge_a[ei] == n { net.edge_b[ei] } else { net.edge_a[ei] };
                // short of a far end that stays, so the kept piece never meets it
                let fi = far as usize;
                let room = if net.node_platform[fi] > 0 {
                    net.edge_len[ei] / 2.0 - 2.0 // that station may keep a piece too
                } else if net.node_ports[fi].n > 1 {
                    net.edge_len[ei] - 5.0
                } else {
                    net.edge_len[ei]
                };
                let len = (reach + 10.0).min(room).max(reach + 1.0);
                let d = EdgeData { a: n, b: 0, tracks: net.edge_tracks[ei], pis: vec![], built: net.edge_built[ei], off: net.edge_off[ei], thru: vec![], crossover: false, hsr: net.edge_hsr[ei], sep: net.edge_sep[ei], day: net.edge_day[ei] };
                let k = len * kk(net.frame().scaled(), y);
                let (m, id) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
                nn.push(m);
                ne.push(id);
                ops.push(Op::Node { id: m, data: Some(NodeData { x: quant(x + k * th.cos()), y: quant(y + k * th.sin()), level, flying: false }) });
                ops.push(Op::Edge { id, data: Some(EdgeData { b: m, ..d }) });
            }
        }
        (ops, nn, ne)
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
        self.construct_with_buses(edges, stations, &[])
    }

    /// `construct`, with bus routes (T-1464) and ferry routes (T-2193) too: `buses` are the lines.
    pub fn construct_with_buses(&mut self, edges: &[u32], stations: &[u32], buses: &[u32]) -> u32 {
        let ops = self.construct_ops(edges, stations, buses);
        if ops.is_empty() {
            return self.refuse(Issue::NothingToConstruct);
        }
        self.run_charged(Op::Batch(ops))
    }

    /// What a line still needs constructed costs, US$M.
    pub fn line_build_cost(&mut self, line: u32) -> f64 {
        let Some((edges, stops, buses)) = self.line_parts(line) else { return 0.0 };
        self.construct_cost(&edges, &stops, &buses)
    }

    /// What a line needs constructed to run: its path's edges, its stations, and itself when it is a
    /// bus line (its route; its stations are those its planned route stops at).
    fn line_parts(&self, line: u32) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return None;
        }
        if net.is_bus(line) {
            let r = net.line_bus[line as usize].as_ref().unwrap();
            let stops = crate::track::bus::plan(net, line, &r.points, &r.kinds, false).stops.iter().filter_map(|s| s.station).collect();
            return Some((vec![], stops, vec![line]));
        }
        // a ferry line: its route (free) and its ports (T-2193)
        if net.is_ferry(line) {
            return Some((vec![], net.line_stops(line).to_vec(), vec![line]));
        }
        Some((net.line_path(line).iter().map(|&p| p >> 1).collect(), net.line_stops(line).to_vec(), vec![]))
    }

    pub(super) fn construct_ops(&self, edges: &[u32], stations: &[u32], buses: &[u32]) -> Vec<Op> {
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
        for &l in buses {
            if let Some(mut d) = net.line_data(l) {
                if let Some(r) = d.bus.as_mut().filter(|r| !r.built) {
                    r.built = true;
                    ops.push(Op::Line { id: l, data: Some(d) });
                } else if let Some(r) = d.ferry.as_mut().filter(|r| !r.built) {
                    r.built = true;
                    ops.push(Op::Line { id: l, data: Some(d) });
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
        let mut buses = net.bus_lines();
        buses.extend(net.ferry_lines());
        self.construct_with_buses(&edges, &stations, &buses)
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

    /// Make blueprint edges high-speed or regular track (T-1898), shape and level unchanged. One
    /// undoable edit. Constructed track, single track and crossovers cannot change and are counted
    /// in `last_skipped`; pieces already that kind are not. Returns 1 when nothing changed.
    pub fn set_edges_hsr(&mut self, ids: &[u32], hsr: bool) -> u32 {
        let net = &self.w.net;
        let mut ops = vec![];
        let mut skipped = 0;
        for &e in ids {
            let Some(mut d) = net.edge_data(e) else { continue };
            if d.built || d.crossover || d.tracks != 2 {
                skipped += 1;
            } else if d.hsr != hsr {
                d.hsr = hsr;
                ops.push(Op::Edge { id: e, data: Some(d) });
            }
        }
        self.skipped = skipped;
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
        if stations.is_empty() && !edges.is_empty() && edges.iter().all(|&e| self.w.net.edge_ok(e) && self.holds_station(e)) {
            return self.refuse(Issue::StationTrack { edge: edges[0] });
        }
        // T-1895: a removal is never refused for what the track it keeps or joins would be (a
        // curve, a crossing): refused that way, it goes without them
        for plain in [false, true] {
            let Some((ops, (nn, ne), st)) = self.pieces_ops(edges, stations, plain) else {
                return self.refuse(Issue::NoSuchEdge { edge: edges.first().copied().unwrap_or(u32::MAX) });
            };
            let r = self.run(Op::Batch(ops));
            if r == 0 {
                self.w.net.links.retain(|&[a, b]| st.binary_search(&a).is_err() && st.binary_search(&b).is_err());
                return 0;
            }
            self.release(&nn, &ne);
            if plain {
                return r;
            }
        }
        unreachable!()
    }

    /// What `delete_pieces` would do to lines (T-859), tried and rolled back: per line that stops
    /// at a removed station or would be left broken (and was not), three numbers: the line, 1 if
    /// it loses a stop, 1 if it would be broken.
    pub fn removal_lines(&mut self, edges: &[u32], stations: &[u32]) -> Vec<f64> {
        let Some((ops, (nn, ne), st)) = self.pieces_ops(edges, stations, false) else { return vec![] };
        let net = &self.w.net;
        let lines: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l)).collect();
        let was: Vec<bool> = lines.iter().map(|&l| net.line_broken[l as usize]).collect();
        let stopping: Vec<bool> = lines.iter().map(|&l| net.line_stops(l).iter().any(|s| st.binary_search(s).is_ok())).collect();
        let tried = self.w.trial_with(Op::Batch(ops), |w| lines.iter().map(|&l| w.net.line_broken[l as usize]).collect::<Vec<_>>());
        self.release(&nn, &ne);
        let Ok((_, after)) = tried else { return vec![] };
        let mut out = vec![];
        for (i, &l) in lines.iter().enumerate() {
            if stopping[i] || (after[i] && !was[i]) {
                out.extend([l as f64, stopping[i] as u8 as f64, after[i] as u8 as f64]);
            }
        }
        out
    }

    /// The ops of `delete_pieces`, the node and edge ids they claim (track kept under stations,
    /// T-1699), and the stations removed (sorted); None when nothing named is there. `plain`: no
    /// track kept or joined.
    pub(super) fn pieces_ops(&mut self, edges: &[u32], stations: &[u32], plain: bool) -> Option<(Vec<Op>, (Vec<u32>, Vec<u32>), Vec<u32>)> {
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
        if plain {
            return Some((ops, (vec![], vec![]), st));
        }
        let (keep, nn, ne) = self.keep_under_stations(&gone, &st);
        ops.extend(keep);
        let joins = self.station_joins(&ops, &st);
        ops.extend(joins);
        Some((ops, (nn, ne), st))
    }

    /// T-1299: the ops joining the two pieces left at each removed station's node into one where
    /// they simply join (`unjoin_near`'s rules: a plain joint, alike in kind and state), planned
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
        let Some((edges, stops, buses)) = self.line_parts(line) else { return self.refuse(Issue::NoSuchLine { line }) };
        self.construct_with_buses(&edges, &stops, &buses)
    }

}

/// Track kept past a trimmed station's platform end (T-1161): `split_plan` may snap a split up to
/// 2 m toward a piece boundary, and `check_station` wants 1 m clear of the platform.
pub(super) const STATION_TRIM_MARGIN_M: f64 = 4.0;
/// The longest piece past a platform's end that a removal only lays back: the trim's margin plus
/// the 7 m `station_trim_ops` needs to split.
const STATION_STUB_M: f64 = STATION_TRIM_MARGIN_M + 7.0;

impl TrackApi {
    /// A passage's price between two stations, US$M (T-1281), paid at once when both are
    /// constructed; 0 when they cannot be joined.
    pub fn passage_cost(&self, a: u32, b: u32) -> f64 {
        let net = &self.w.net;
        if a == b || !net.is_station(a) || !net.is_station(b) { 0.0 } else { upgrades::passage_price(net.passage_m(a, b)) }
    }

    /// T-2797: the edge runs from a station to a bare end and is no longer than what removing it
    /// would keep there (`STATION_STUB_M` past the platform), so a removal would change nothing.
    /// The app's `stationStub` is the same rule.
    fn holds_station(&self, id: u32) -> bool {
        let net = &self.w.net;
        let i = id as usize;
        if net.edge_crossover[i] {
            return false;
        }
        let bare = |n: u32| net.node_ports[n as usize].n == 1 && net.node_platform[n as usize] == 0;
        let (a, b) = (net.edge_a[i], net.edge_b[i]);
        let st = if bare(b) { a } else if bare(a) { b } else { return false };
        net.node_platform[st as usize] > 0 && net.edge_len[i] <= net.platform_reach(st) + STATION_STUB_M
    }

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
