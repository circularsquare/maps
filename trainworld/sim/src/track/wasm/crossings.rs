//! Doubling up (T-708) and the crossover tool (T-338).

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- doubling up (T-708)

    /// Double up: an express pair beside double track, as blueprint, making it quad. One undoable
    /// edit. The rules are in docs/track-and-service.md 6.1 ("Doubling up").
    /// A crossover goes in past each station on the run where it fits (T-733); without them when
    /// the cash covers only the doubling.
    pub fn double_up(&mut self, edge: u32) -> u32 {
        let (op, nn, ne, stations) = match self.double_up_op(edge) {
            Ok(v) => v,
            Err(i) => return self.refuse(i),
        };
        let mut r = 1;
        let (full, n2, e2) = self.with_crossovers(op.clone(), &stations);
        if !e2.is_empty() {
            r = self.run(full);
            if r != 0 {
                self.release(&n2, &e2);
            }
        }
        if r != 0 {
            r = self.run(op);
        }
        if r != 0 {
            self.release(&nn, &ne);
        }
        r
    }

    /// What `double_up` would add to the blueprint, crossovers included, US$M, or -1 when it would
    /// be refused (`issue_kinds()` says why).
    pub fn preview_double_up(&mut self, edge: u32) -> f64 {
        let (op, nn, ne, stations) = match self.double_up_op(edge) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return -1.0;
            }
        };
        let (full, n2, e2) = self.with_crossovers(op.clone(), &stations);
        let with = if e2.is_empty() { None } else { self.w.trial(full).ok() };
        self.release(&n2, &e2);
        let r = match with {
            Some(r) => Ok(r),
            None => self.w.trial(op),
        };
        self.release(&nn, &ne);
        let cost = r.as_ref().map_or(-1.0, |r| r.cost_after - r.cost_before);
        self.done(r.map(|mut r| {
            r.dirty = Default::default();
            r
        }));
        self.last = None;
        cost
    }

    /// The double-up edit on the left of the piece's own direction, else on its right where the
    /// left is refused; the left's problems when both are. Also the run's stations.
    #[allow(clippy::type_complexity)]
    pub(super) fn double_up_op(&mut self, edge: u32) -> Result<(Op, Vec<u32>, Vec<u32>, Vec<u32>), Issue> {
        let (run, nodes) = self.double_run(edge)?;
        let mut stations: Vec<u32> = nodes.iter().copied().filter(|&n| self.w.net.is_station(n)).collect();
        if stations.len() > 1 && stations.first() == stations.last() {
            stations.pop(); // a ring ends where it starts
        }
        let mut first_err = None;
        for side in [1.0, -1.0] {
            let (op, nn, ne) = self.double_side(&run, &nodes, side)?;
            let r = self.w.trial(op.clone());
            match r {
                Ok(_) => return Ok((op, nn, ne, stations)),
                Err(e) => {
                    self.release(&nn, &ne);
                    first_err.get_or_insert(e);
                }
            }
        }
        // the left's own issues, through `refuse`'s single slot: the first stands for them all
        let mut e = first_err.unwrap_or_default();
        Err(if e.is_empty() { Issue::NoSuchEdge { edge } } else { e.swap_remove(0) })
    }

    /// The run a double-up covers: the pieces in order (edge, and whether it runs b to a) and the
    /// nodes from end to end, starting from `edge`'s a end. It carries on through plain joints and
    /// stations into double track, and stops at a junction, a track end, other track counts, a
    /// crossover or track that has a pair beside it already.
    pub(super) fn double_run(&self, edge: u32) -> Result<(Vec<(u32, bool)>, Vec<u32>), Issue> {
        let net = &self.w.net;
        if !net.edge_ok(edge) {
            return Err(Issue::NoSuchEdge { edge });
        }
        if net.edge_crossover[edge as usize] {
            return Err(Issue::Crossover { edge });
        }
        if net.edge_tracks[edge as usize] != 2 {
            return Err(Issue::Tracks { edge });
        }
        if net.edge_hsr[edge as usize] {
            return Err(Issue::HighSpeed { edge });
        }
        let doubled = |e: u32| self.twin_at(e, net.edge_len[e as usize] / 2.0).is_some();
        if doubled(edge) {
            return Err(Issue::Alongside { edge });
        }
        let mut run = vec![(edge, false)];
        let mut nodes = vec![net.edge_a[edge as usize], net.edge_b[edge as usize]];
        // `back`: walking out of the a end, so pieces go in at the front
        for back in [false, true] {
            loop {
                let (node, from) = if back { (nodes[0], run[0].0) } else { (*nodes.last().unwrap(), run.last().unwrap().0) };
                let p = &net.node_ports[node as usize];
                if p.n != 2 || p.side[0] == p.side[1] {
                    break;
                }
                let Some(&c) = p.ends().iter().find(|&&c| c >> 1 != from) else { break };
                let e = c >> 1;
                if run.iter().any(|r| r.0 == e) || net.edge_crossover[e as usize] || net.edge_tracks[e as usize] != 2 || net.edge_hsr[e as usize] || doubled(e) {
                    break;
                }
                // c & 1 is the end of e at `node`: walking forward, e runs a to b when that is its a
                let far = net.end_node(e, 1 - (c & 1));
                if back {
                    run.insert(0, (e, c & 1 == 0));
                    nodes.insert(0, far);
                } else {
                    run.push((e, c & 1 == 1));
                    nodes.push(far);
                }
            }
        }
        Ok((run, nodes))
    }

    /// `double_up_op` on one side: 1 left of the run's direction, -1 right.
    pub(super) fn double_side(&mut self, run: &[(u32, bool)], nodes: &[u32], side: f64) -> Result<(Op, Vec<u32>, Vec<u32>), Issue> {
        let scaled = self.w.net.frame().scaled();
        let shapes: Vec<Vec<Pi>> = run
            .iter()
            .map(|&(e, rev)| {
                let mut d = self.editable(e).expect("a live edge");
                if rev {
                    d.pis.reverse();
                    std::mem::swap(&mut d.a, &mut d.b);
                }
                vertices_of(&self.w.net, &d, &[])
            })
            .collect();
        let last = nodes.len() - 1;
        let (mut ops, mut nn, mut ne) = (vec![], vec![], vec![]);
        // each node's partner on the express pair: None at a junction end, where it swings in
        let mut beside: Vec<Option<(u32, f64, f64)>> = vec![];
        for (i, &n) in nodes.iter().enumerate() {
            let net = &self.w.net;
            let ni = n as usize;
            if i == last && n == nodes[0] {
                beside.push(beside[0]); // a ring: back where it started
                continue;
            }
            let end = i == 0 || i == last;
            let ports = net.node_ports[ni].n;
            if end && ports >= 3 && !net.node_ports[ni].ends().iter().any(|&c| net.edge_crossover[(c >> 1) as usize]) {
                beside.push(None);
                continue;
            }
            let (x, y) = if i < last { express_end(&shapes[i], 0, side, scaled) } else { express_end(&shapes[i - 1], 1, side, scaled) };
            if end {
                if let Some(m) = loose_near(net, n, x, y) {
                    beside.push(Some((m, net.node_x[m as usize], net.node_y[m as usize])));
                    continue;
                }
            }
            let m = self.w.net.alloc_node();
            nn.push(m);
            ops.push(Op::Node { id: m, data: Some(NodeData { x, y, level: self.w.net.node_level[ni], flying: false }) });
            if let Some(mut st) = self.w.net.station_data(n) {
                st.built = false;
                st.off = false;
                ops.push(Op::Station { node: m, data: Some(st) });
            }
            beside.push(Some((m, x, y)));
        }
        for (i, verts) in shapes.iter().enumerate() {
            let (a, b) = (beside[i], beside[i + 1]);
            let Some((ex, _, _)) = express_verts(verts, 1, verts.len() - 2, scaled, side, a.map(|o| (o.1, o.2)), b.map(|o| (o.1, o.2))) else {
                self.release(&nn, &ne);
                return Err(Issue::Geom { edge: run[i].0, issue: GeomIssue::ZeroLeg(0) });
            };
            let id = self.w.net.alloc_edge();
            ne.push(id);
            let (na, nb) = (a.map_or(nodes[i], |o| o.0), b.map_or(nodes[i + 1], |o| o.0));
            ops.push(Op::Edge { id, data: Some(EdgeData { a: na, b: nb, tracks: 2, pis: ex[1..ex.len() - 1].to_vec(), built: false, off: false, thru: vec![], crossover: false, hsr: false }) });
        }
        Ok((Op::Batch(ops), nn, ne))
    }

    // ---- crossovers (T-338)

    /// Place a crossover on edge `edge` near (x, y): both diagonals between it and the track
    /// running parallel beside it, as blueprint. One undoable edit.
    pub fn add_crossover(&mut self, edge: u32, x: f64, y: f64) -> u32 {
        let (op, nn, ne, _) = match self.crossover_op(edge, x, y) {
            Ok(v) => v,
            Err(i) => return self.refuse(i),
        };
        let r = self.run(op);
        if r != 0 {
            self.release(&nn, &ne);
        }
        r
    }

    /// What `add_crossover` would do, without doing it: `[ok (1/0), cost US$M, then per diagonal
    /// its point count and its centre line as x, y, height triples]`. `issue_kinds()` says why not.
    pub fn preview_crossover(&mut self, edge: u32, x: f64, y: f64) -> Vec<f64> {
        let (op, nn, ne, diagonals) = match self.crossover_op(edge, x, y) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0, 0.0];
            }
        };
        let r = self.w.trial(op);
        self.release(&nn, &ne);
        let ok = r.is_ok();
        let cost = r.as_ref().map_or(0.0, |r| r.cost_after - r.cost_before);
        self.done(r.map(|mut r| {
            r.dirty = Default::default();
            r
        }));
        self.last = None;
        let mut out = vec![ok as u8 as f64, cost];
        for verts in &diagonals {
            let f = self.w.net.fit(verts);
            let mut pts = vec![];
            render_samples(&f.pieces, &f.vert, &mut pts);
            out.push((pts.len() / 3) as f64);
            out.extend(pts);
        }
        out
    }

    /// Edges of high-speed track (T-817).
    pub fn hsr_edges(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_hsr[e as usize]).collect()
    }

    /// Edges that are a crossover's diagonals.
    pub fn crossover_edges(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_crossover[e as usize]).collect()
    }

    /// Each of `crossover_edges`' other diagonal, in the same order (`u32::MAX` if it has none).
    pub fn crossover_partners(&self) -> Vec<u32> {
        let net = &self.w.net;
        self.crossover_edges().into_iter().map(|e| crossover_partner(net, e).unwrap_or(u32::MAX)).collect()
    }

    /// The crossover edit (`add_crossover`): splits at the four ends (or plain joints there), and
    /// the two diagonals, each leaving its ends along the track like a turnout and swinging
    /// across in an S over `CROSSOVER_LEN`. Also the new ids (to release if it is not applied)
    /// and each diagonal's vertices.
    #[allow(clippy::type_complexity)]
    pub(super) fn crossover_op(&mut self, edge: u32, x: f64, y: f64) -> Result<(Op, Vec<u32>, Vec<u32>, Vec<Vec<Pi>>), Issue> {
        let net = &self.w.net;
        if !net.edge_ok(edge) {
            return Err(Issue::NoSuchEdge { edge });
        }
        if net.edge_crossover[edge as usize] {
            return Err(Issue::Crossover { edge });
        }
        let (s, _) = net.project(edge, x, y);
        let (cx, cy, th) = geom::pos_at(net.edge_pieces(edge), s);
        let k = net.k_at(cy);
        // the nearest track beside, running the same way or the other
        let mut best: Option<(u32, f64, f64)> = None;
        for f in 0..net.edge_count() as u32 {
            if f == edge || !net.edge_ok(f) || net.edge_crossover[f as usize] {
                continue;
            }
            let (sf, dist) = net.project(f, cx, cy);
            if !(CROSSOVER_MIN_GAP * k..=CROSSOVER_REACH * k).contains(&dist) || best.is_some_and(|b| b.2 <= dist) {
                continue;
            }
            if (th - geom::pos_at(net.edge_pieces(f), sf).2).sin().abs() > 0.05 {
                continue;
            }
            best = Some((f, sf, dist));
        }
        let Some((f, sf, _)) = best else { return Err(Issue::NoParallel { edge }) };
        let same = (th - geom::pos_at(net.edge_pieces(f), sf).2).cos() > 0.0;
        let half = CROSSOVER_LEN / 2.0;
        let none = || Issue::NoParallel { edge };
        // back and forward along the drawn edge, then level with them beside
        let a0 = locate(net, edge, s - half).ok_or_else(none)?;
        let a1 = locate(net, edge, s + half).ok_or_else(none)?;
        let (b0, b1) = if same { (sf - half, sf + half) } else { (sf + half, sf - half) };
        let b0 = locate(net, f, b0).ok_or_else(none)?;
        let b1 = locate(net, f, b1).ok_or_else(none)?;
        let locs = [a0, a1, b0, b1];
        // the ends: where each node will be and its heading in the drawn edge's direction
        let mut ends: Vec<(f64, f64, i8, f64)> = vec![];
        for l in &locs {
            let (x, y, level, h) = match *l {
                Loc::Node(n) => {
                    let i = n as usize;
                    (net.node_x[i], net.node_y[i], net.node_level[i], net.node_heading[i])
                }
                Loc::Split(e, s) => {
                    let (nd, _, _) = net.split_plan(e, s)?;
                    let h = geom::pos_at(net.edge_pieces(e), net.project(e, nd.x, nd.y).0).2;
                    (nd.x, nd.y, nd.level, h)
                }
            };
            let h = if (h - th).cos() < 0.0 { h + PI } else { h };
            ends.push((x, y, level, h));
        }
        // no two ends on one node, and the span far enough from where it was clicked to be real
        for i in 0..4 {
            for j in i + 1..4 {
                if (ends[i].0 - ends[j].0).hypot(ends[i].1 - ends[j].1) < CROSSOVER_MIN_GAP * 0.5 * k {
                    return Err(none());
                }
            }
        }
        let tracks = net.edge_tracks[edge as usize].min(net.edge_tracks[f as usize]);
        // splits, several on one edge in order along it: each after the first falls in the new half
        let mut nodes = vec![0u32; 4];
        let mut new_nodes = vec![];
        let mut new_edges = vec![];
        let mut ops = vec![];
        let mut splits: Vec<(u32, f64, usize)> = vec![];
        for (j, l) in locs.iter().enumerate() {
            match *l {
                Loc::Node(n) => nodes[j] = n,
                Loc::Split(e, s) => splits.push((e, s, j)),
            }
        }
        splits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut prev: Option<(u32, f64, u32)> = None; // original edge, chainage split at, its new half
        for &(e, s, j) in &splits {
            let (target, at) = match prev {
                Some((pe, ps, half)) if pe == e => (half, s - ps),
                _ => (e, s),
            };
            let (nd, _, _) = self.w.net.split_plan(e, s)?; // checked above: never fails here
            let node = self.w.net.alloc_node();
            let new_edge = self.w.net.alloc_edge();
            new_nodes.push(node);
            new_edges.push(new_edge);
            nodes[j] = node;
            ops.push(Op::Split { edge: target, s: at, node, new_edge });
            prev = Some((e, self.w.net.project(e, nd.x, nd.y).0, new_edge));
        }
        let mut diagonals = vec![];
        for (p, q) in [(0usize, 3usize), (2, 1)] {
            let (px, py, pl, ph) = ends[p];
            let (qx, qy, ql, qh) = ends[q];
            let t = (qx - px).hypot(qy - py) / 4.0;
            let pis = vec![
                Pi::new(px + ph.cos() * t, py + ph.sin() * t, 0.0, pl).quantised(),
                Pi::new(qx - qh.cos() * t, qy - qh.sin() * t, 0.0, ql).quantised(),
            ];
            let id = self.w.net.alloc_edge();
            new_edges.push(id);
            ops.push(Op::Edge { id, data: Some(EdgeData { a: nodes[p], b: nodes[q], tracks, pis: pis.clone(), built: false, off: false, thru: vec![], crossover: true, hsr: false }) });
            let mut v = vec![Pi::new(px, py, 0.0, pl)];
            v.extend(pis);
            v.push(Pi::new(qx, qy, 0.0, ql));
            diagonals.push(v);
        }
        Ok((Op::Batch(ops), new_nodes, new_edges, diagonals))
    }

    /// Rename a station or change its platform. A longer platform on a constructed station is
    /// paid for at once (final).
    pub fn set_station_props(&mut self, node: u32, platform: u16, name: String) -> u32 {
        let Some(old) = self.w.net.station_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
        let op = Op::Station { node, data: Some(StationData { platform, name, built: old.built, off: old.off }) };
        if old.built && old.platform != platform {
            self.run_charged(op)
        } else {
            self.run(op)
        }
    }

    /// Remove a station (and its node if no track is left there), as `delete_pieces` does.
    /// Constructed: final, no refund. Its links (T-197) go with it, and undoing the removal does
    /// not bring them back.
    pub fn remove_station(&mut self, node: u32) -> u32 {
        if self.w.net.station_data(node).is_none() {
            return self.refuse(Issue::NoSuchNode { node });
        }
        self.delete_pieces(&[], &[node])
    }

}
