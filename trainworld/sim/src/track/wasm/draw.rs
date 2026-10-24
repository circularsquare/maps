//! The drawing tool: routes, previews and reshaping a blueprint edge.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- the drawing tool (T-023, T-092)

    /// `ends`: start then end, 5 numbers each: kind (0 free, 1 node, 2 edge), id, x, y, level
    /// (level used for a free end). `pis`: the player's PIs between, x, y, radius (0 = auto),
    /// level: the track cuts the corner at each with a circular curve (SPEC 6.1). A start or end on
    /// existing track gets a lead-in PI on that track's heading, so it leaves like a turnout.
    pub(super) fn route_op(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Result<(Op, Vec<Pi>, Vec<u32>, Vec<u32>), Issue> {
        let pis = pis_from(pis);
        let scaled = self.w.net.frame().scaled();
        let (a, b) = (end_from(&ends[0..5]), end_from(&ends[5..10]));
        let mut new_nodes = vec![];
        let mut new_edges = vec![];
        let net = &self.w.net;
        let b_at = match b {
            End::Free { x, y, .. } | End::Edge { x, y, .. } => (x, y),
            End::Node(n) if net.node_ok(n) => (net.node_x[n as usize], net.node_y[n as usize]),
            End::Node(_) => (0.0, 0.0),
        };
        let (a, from_a) = branch_end(net, a, pis.first().map_or(b_at, |p| (p.x, p.y)))?;
        // Both ends on one edge: the second split must name the half it falls in.
        let first_split = if let End::Edge { e, x, y } = a { Some((e, self.w.net.project(e, x, y).0)) } else { None };
        let mut ra = self.resolve(a, None, &mut new_nodes, &mut new_edges)?;
        leave_away(&mut ra, from_a);
        let (b, from_b) = match branch_end(&self.w.net, b, pis.last().map_or((ra.x, ra.y), |p| (p.x, p.y))) {
            Ok(v) => v,
            Err(i) => {
                self.release(&new_nodes, &new_edges);
                return Err(i);
            }
        };
        let mut rb = self.resolve(b, first_split, &mut new_nodes, &mut new_edges)?;
        leave_away(&mut rb, from_b);
        if ra.node == rb.node {
            self.release(&new_nodes, &new_edges);
            return Err(Issue::EdgeLoop { edge: u32::MAX });
        }
        self.route_end = rb.node;
        self.route_arc.clear();
        self.route_bridged.clear();
        self.bridge_refused = false;
        let mut verts: Vec<Pi> = vec![Pi::new(ra.x, ra.y, 0.0, ra.level)];
        let next_a = pis.first().map_or((rb.x, rb.y), |p| (p.x, p.y));
        let prev_b = pis.last().map_or((ra.x, ra.y), |p| (p.x, p.y));
        // T-418: ending on open ground, the segment is one arc through the end; ending on track it
        // must join that like a turnout, so it keeps the usual shape
        let quad = tracks == 4;
        let through = self.route_tangent && pis.is_empty();
        // T-423: quad drawn through points carries both pairs on from a loose quad end, the other
        // pair's end beside the one clicked (on either side); from other track it swings out as in
        // Corners, which needs a straight, so that segment is not one arc
        let beside_a = if quad && through { quad_beside(&self.w.net, ra.node) } else { None };
        let side = beside_a.map_or(1.0, |b| b.1);
        let beside_b = if quad && through { quad_beside(&self.w.net, rb.node).filter(|b| b.1 == -side) } else { None };
        let tangent = through && rb.dirs.is_empty() && (!quad || beside_a.is_some());
        if let (true, Some(dir)) = (tangent, nearer_dir(&ra, next_a)) {
            let (pi, circle) = tangent_arc(ra.x, ra.y, ra.level, rb.level, dir, next_a, scaled, self.route_cap);
            if let Some(c) = circle {
                self.route_arc = c.to_vec();
            }
            match pi {
                Ok(p) => verts.extend(p),
                Err(()) => {
                    self.release(&new_nodes, &new_edges);
                    return Err(Issue::Geom { edge: u32::MAX, issue: GeomIssue::Reversal(1) });
                }
            }
        } else if let Some(p) = lead_in(&ra, next_a, scaled, self.route_cap) {
            verts.push(p);
        }
        let first_click = verts.len();
        verts.extend_from_slice(&pis);
        let lead_b = lead_in(&rb, prev_b, scaled, self.route_cap);
        if let Some(p) = lead_b {
            verts.push(p);
        }
        verts.push(Pi::new(rb.x, rb.y, 0.0, rb.level));
        // T-817: the corners made here, not the clicked ones (a real line's, fitted at R_CAP)
        let made: Vec<usize> = (1..first_click).chain(lead_b.map(|_| verts.len() - 2)).collect();
        design_radii(&mut verts, &made, self.route_cap, scaled);
        let mut ops = ra.ops.clone();
        ops.extend(rb.ops.clone());
        // Quad track (T-255): the drawn pair, and an express pair beside it from end to end.
        // High-speed track is double only (T-817).
        let hsr = self.route_hsr && tracks == 2;
        let tracks = if quad { 2 } else { tracks };
        // One piece of track per corner clicked (T-244), joined at plain nodes, the shape unchanged.
        // T-656: ground level over water is a bridge, raised over the water with ramps on land.
        // Clicked corners at ground level too (T-667, the real-network snap's routes, T-727): the
        // corners the bridge adds are cut into pieces like the clicked ones.
        let bridged = self.bridged(&verts);
        let cut = |b: &Option<Vec<Pi>>, n: usize, clicks: usize| match b {
            Some(b) if clicks > 0 => (clicks + b.len()).saturating_sub(n),
            _ => clicks,
        };
        let pieces = stroke_pieces(bridged.as_deref().unwrap_or(&verts), first_click, cut(&bridged, verts.len(), pis.len()), scaled);
        self.push_pieces(&pieces, ra.node, rb.node, tracks, hsr, &mut ops, &mut new_nodes, &mut new_edges);
        self.route_bridged = bridged.unwrap_or_default();
        let mut express = vec![];
        if quad {
            // T-423: the express pair's own ends, at a loose quad end or (through points) open ground
            let mut own = |at: usize, beside: Option<(u32, f64)>, r: &Resolved, ops: &mut Vec<Op>, nodes: &mut Vec<u32>| -> Option<(u32, (f64, f64))> {
                if let Some((m, _)) = beside {
                    let net = &self.w.net;
                    return Some((m, (net.node_x[m as usize], net.node_y[m as usize])));
                }
                if !(through && r.dirs.is_empty()) {
                    return None;
                }
                let (x, y) = express_end(&verts, at, side, scaled);
                let n = self.w.net.alloc_node();
                nodes.push(n);
                ops.push(Op::Node { id: n, data: Some(NodeData { x, y, level: r.level, flying: false }) });
                Some((n, (x, y)))
            };
            let own_a = own(0, beside_a, &ra, &mut ops, &mut new_nodes);
            let own_b = own(1, beside_b, &rb, &mut ops, &mut new_nodes);
            let (ex, first, clicks) = match express_verts(&verts, first_click, pis.len(), scaled, side, own_a.map(|o| o.1), own_b.map(|o| o.1)) {
                Some(v) => v,
                None => {
                    self.release(&new_nodes, &new_edges);
                    return Err(Issue::Geom { edge: u32::MAX, issue: GeomIssue::ZeroLeg(0) });
                }
            };
            // the express pair crosses the water on its own bridge, in one piece
            let ex_bridged = self.bridged(&ex);
            let pieces = match &ex_bridged {
                Some(b) if pis.is_empty() => stroke_pieces(b, first, 0, scaled),
                Some(b) => stroke_pieces(b, first, cut(&ex_bridged, ex.len(), clicks), scaled),
                None => stroke_pieces(&ex, first, clicks, scaled),
            };
            self.push_pieces(&pieces, own_a.map_or(ra.node, |o| o.0), own_b.map_or(rb.node, |o| o.0), tracks, false, &mut ops, &mut new_nodes, &mut new_edges);
            express = ex_bridged.unwrap_or(ex);
        }
        // A junction this route makes (it had two edge ends, or it splits track) is flying if
        // the build setting says so; never one that already stands.
        self.route_junctions.clear();
        for r in [&ra, &rb] {
            if flying && r.new_junction {
                ops.push(Op::Node { id: r.node, data: Some(NodeData { x: r.x, y: r.y, level: r.level, flying: true }) });
            } else if r.new_junction {
                self.route_junctions.push([r.x, r.y]);
            }
        }
        self.route_express = express;
        Ok((Op::Batch(ops), verts, new_nodes, new_edges))
    }

    pub(super) fn resolve(&mut self, end: End, other_split: Option<(u32, f64)>, nodes: &mut Vec<u32>, edges: &mut Vec<u32>) -> Result<Resolved, Issue> {
        let net = &self.w.net;
        match end {
            End::Free { x, y, level } => {
                let node = self.w.net.alloc_node();
                nodes.push(node);
                Ok(Resolved { node, x, y, level, dirs: vec![], new_junction: false, ops: vec![Op::Node { id: node, data: Some(NodeData { x, y, level, flying: false }) }] })
            }
            End::Node(n) => {
                if !net.node_ok(n) {
                    return Err(Issue::NoSuchNode { node: n });
                }
                let i = n as usize;
                let p = &net.node_ports[i];
                let h = net.node_heading[i];
                let dirs = match p.n {
                    0 => vec![],
                    // A track end: carry on away from the track that is there.
                    1 => vec![if p.side[0] == 1 { h + PI } else { h }],
                    _ => vec![h, h + PI],
                };
                Ok(Resolved { node: n, x: net.node_x[i], y: net.node_y[i], level: net.node_level[i], dirs, new_junction: p.n == 2 && net.node_platform[i] == 0, ops: vec![] })
            }
            End::Edge { e, x, y } => {
                if !net.edge_ok(e) {
                    return Err(Issue::NoSuchEdge { edge: e });
                }
                let (mut s, _) = net.project(e, x, y);
                let mut target = e;
                let (nd, _, _) = net.split_plan(e, s)?;
                let heading = geom::pos_at(net.edge_pieces(e), net.project(e, nd.x, nd.y).0).2;
                if let Some((oe, os)) = other_split {
                    if oe == e {
                        // The first split keeps `e` for 0..os; anything beyond is in its new half.
                        let (snd, _, _) = net.split_plan(e, os)?;
                        let os = net.project(e, snd.x, snd.y).0;
                        if s > os {
                            target = u32::MAX; // patched below, once the first split's edge id is known
                            s -= os;
                        }
                    }
                }
                let node = self.w.net.alloc_node();
                let new_edge = self.w.net.alloc_edge();
                nodes.push(node);
                edges.push(new_edge);
                if target == u32::MAX {
                    // The first end's split created edges[0] (its new half).
                    target = edges[0];
                }
                Ok(Resolved { node, x: nd.x, y: nd.y, level: nd.level, dirs: vec![heading, heading + PI], new_junction: true, ops: vec![Op::Split { edge: target, s, node, new_edge }] })
            }
        }
    }

    /// T-656: `shape` as a bridge over any water it crosses at ground level, or None where it needs
    /// none. With no room on land for the ramps it is left as drawn and `bridge_refused` set, so the
    /// edit fails and says why (`bridge_refusal`).
    pub(super) fn bridged(&mut self, shape: &[Pi]) -> Option<Vec<Pi>> {
        let scaled = self.w.net.frame().scaled();
        // a snapped real route (T-727d) bridges each river on its own
        let b = if shape.len() > LONG_ROUTE_VERTS { bridge::bridge_long(shape, scaled, self.w.net.water_mask.as_ref()) } else { bridge::bridge(shape, scaled, self.w.net.water_mask.as_ref()) };
        match b {
            Ok(b) => b,
            Err(bridge::NoRoom) => {
                self.bridge_refused = true;
                None
            }
        }
    }

    pub(super) fn release(&mut self, nodes: &[u32], edges: &[u32]) {
        for &n in nodes {
            self.w.net.release_node(n);
        }
        for &e in edges {
            self.w.net.release_edge(e);
        }
    }

    /// The drawing preview: what `add_route` with these arguments would do, without doing it.
    /// Returns [ok (1/0), cost US$M, length m, vertex count, then per vertex x, y, radius used,
    /// speed limit km/h, then the polyline as x, y, height (m) triples]. `issue_kinds()` holds
    /// the problems when not ok.
    pub fn preview_route(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Vec<f64> {
        self.preview_speeds.clear();
        self.preview_marks.clear();
        let (op, verts, nn, ne) = match self.route_op(ends, pis, tracks, flying) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let r = self.w.trial(op);
        self.release(&nn, &ne);
        let r = if self.bridge_refused { r.map_err(bridge_refusal) } else { r };
        let shape = std::mem::take(&mut self.route_bridged);
        let mut out = self.preview_out(r, &verts, &shape, tracks.min(2), self.route_hsr && tracks == 2);
        for j in &self.route_junctions {
            self.preview_marks.extend([j[0], j[1], 1.0]);
        }
        if !self.route_express.is_empty() && out.len() > 5 {
            // T-711: the multiplier over both pairs' track, as the price is
            let (fl, tl, _) = self.w.preview(if shape.is_empty() { &verts } else { &shape }, 2, false);
            let (fe, te, _) = self.w.preview(&self.route_express, 2, false);
            let base = BASE_COST_PER_KM * (fl.len + fe.len) / 1000.0;
            out[4] = if base > 0.0 { (tl + te) / base } else { 0.0 };
            // a quad route's express pair, drawn on from the end back to the start (T-255)
            let mut back = self.route_express.clone();
            back.reverse();
            let f = self.w.net.fit(&back);
            let end = verts[verts.len() - 1];
            // -1: no track between the pairs' own ends (T-423), so the ghost draws no step there
            let joint = if (back[0].x - end.x).hypot(back[0].y - end.y) > 0.5 { -1.0 } else { self.preview_speeds.last().copied().unwrap_or((V_TOP * 3.6) as f32) };
            self.preview_speeds.push(joint);
            render_samples(&f.pieces, &f.vert, &mut out);
            self.preview_speeds.extend(render_speeds(&f.pieces, &f.vert, V_TOP));
        }
        self.route_vertices = verts;
        out
    }

    /// T-418: whether routes after this are drawn through their end (`tangent_arc`): a route with
    /// no PIs from an end with a heading to open ground becomes one tangent arc (quad only from a
    /// loose quad end, T-423).
    pub fn set_route_tangent(&mut self, on: bool) {
        self.route_tangent = on;
    }

    /// T-817: the design speed of routes after this, km/h: their curves are laid out for it, up to
    /// `design_radius(kmh)`. Clamped to 160..320; 160 is the drawing tool as before.
    pub fn set_design_speed(&mut self, kmh: f64) {
        let kmh = if kmh.is_finite() { kmh.clamp(DESIGN_SPEEDS[0], DESIGN_SPEEDS[DESIGN_SPEEDS.len() - 1]) } else { DESIGN_SPEEDS[0] };
        self.route_cap = design_radius(kmh);
    }

    /// T-817: whether double track drawn after this (routes and stations on open ground) is
    /// high-speed track. Single and quad track are never.
    pub fn set_route_hsr(&mut self, on: bool) {
        self.route_hsr = on;
    }

    /// The circle of the last route preview's tangent arc (T-418): centre x, y and ground radius,
    /// even when the route was refused; empty when it had none.
    pub fn route_arc(&self) -> Vec<f64> {
        self.route_arc.clone()
    }

    /// A real railway's line as corners for `add_route` (T-727, `trace.rs`): `lonlat` the line as
    /// lon, lat pairs from one end to the other, `tol` how far the track may lie from it (ground
    /// m), quad track (`tracks` 4) keeping its inner pair at 100 m. Returns [worst, mean distance
    /// (m), corner count, then each corner's x, y, radius (0 auto), level], or empty if it cannot
    /// be fitted.
    pub fn trace_route(&self, lonlat: &[f64], tol: f64, tracks: u8, level: i8) -> Vec<f64> {
        let mut t = self.trace_start(lonlat, tol, tracks, level);
        while !t.step() {}
        t.result()
    }

    /// `trace_route` in steps (T-754): `step` until it says done, then `result`.
    pub fn trace_start(&self, lonlat: &[f64], tol: f64, tracks: u8, level: i8) -> RouteTrace {
        let f = self.w.net.frame();
        let line: Vec<(f64, f64)> = lonlat.chunks_exact(2).map(|p| f.from_lonlat(p[0], p[1])).collect();
        let r_min = if tracks == 4 { MIN_RADIUS + QUAD_SPACING } else { MIN_RADIUS };
        RouteTrace { t: crate::track::trace::Tracer::new(&line, f.scaled(), tol, r_min, level) }
    }

    /// The node the last `add_route` ended at, so the next segment can start from it (T-418).
    pub fn route_end(&self) -> u32 {
        self.route_end
    }

    /// Edges for the pieces of a stroke (`stroke_pieces`) from node `a` to node `b`, with new
    /// joints between them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_pieces(&mut self, pieces: &[Vec<Pi>], a: u32, b: u32, tracks: u8, hsr: bool, ops: &mut Vec<Op>, nodes: &mut Vec<u32>, edges: &mut Vec<u32>) {
        let mut from = a;
        for (k, p) in pieces.iter().enumerate() {
            let to = if k + 1 == pieces.len() {
                b
            } else {
                let joint = *p.last().unwrap();
                let n = self.w.net.alloc_node();
                nodes.push(n);
                ops.push(Op::Node { id: n, data: Some(NodeData { x: joint.x, y: joint.y, level: joint.level, flying: false }) });
                n
            };
            let edge = self.w.net.alloc_edge();
            edges.push(edge);
            ops.push(Op::Edge { id: edge, data: Some(EdgeData { a: from, b: to, tracks, pis: p[1..p.len() - 1].to_vec(), built: false, off: false, thru: vec![], crossover: false, hsr }) });
            from = to;
        }
    }

    /// Draw a route as blueprint (one undoable edit). Same arguments as `preview_route`.
    pub fn add_route(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> u32 {
        let (op, _, nn, ne) = match self.route_op(ends, pis, tracks, flying) {
            Ok(v) => v,
            Err(i) => return self.refuse(i),
        };
        let r = self.run(op);
        if r != 0 {
            self.release(&nn, &ne);
            if self.bridge_refused && self.money_short.is_none() {
                self.issues = bridge_refusal(std::mem::take(&mut self.issues));
            }
        }
        r
    }

    /// A blueprint station: on an existing node (kind 1, `id`) or splitting an edge near (x, y)
    /// (kind 2, `id` = edge).
    pub fn add_station(&mut self, kind: u32, id: u32, x: f64, y: f64, platform: u16, name: String) -> u32 {
        self.station_at(kind, id, x, y, platform, name, false)
    }

    /// Whether `add_station` with these arguments would succeed, without doing it (T-311): the
    /// placement ghost's colour. The same code as `add_station` with the edit tried and rolled
    /// back, so the ghost and the click cannot disagree (cash included). 0 when it would work;
    /// otherwise `issue_kinds()` says why.
    pub fn preview_station(&mut self, kind: u32, id: u32, x: f64, y: f64, platform: u16) -> u32 {
        self.station_at(kind, id, x, y, platform, String::new(), true)
    }

    /// An edit, or with `dry` a trial of it: checked as a real edit is, then rolled back.
    pub(super) fn attempt(&mut self, op: Op, dry: bool) -> u32 {
        if !dry {
            return self.run(op);
        }
        let fleet = self.fleet;
        match self.w.trial_with(op, |w| trains_to_buy(w, fleet)) {
            Ok((r, trains)) => {
                self.station_cost = r.cost_after - r.cost_before;
                let need = r.charge + trains;
                self.issues.clear();
                self.money_short = None;
                if need > 1e-9 && need > self.spendable() + 1e-6 {
                    self.money_short = Some((need, self.cash));
                    return 1;
                }
                0
            }
            Err(e) => self.done(Err(e)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn station_at(&mut self, kind: u32, id: u32, x: f64, y: f64, platform: u16, name: String, dry: bool) -> u32 {
        let st = StationData { platform, name, built: false, off: false };
        if kind == 1 {
            if !self.w.net.node_ok(id) {
                return self.refuse(Issue::NoSuchNode { node: id });
            }
            if self.w.net.node_platform[id as usize] > 0 {
                return self.refuse(Issue::StationPorts { node: id });
            }
            // a joint or plain node on quad track: its twin beside it too (T-255)
            let ends = self.w.net.node_ports[id as usize].ends().to_vec();
            if let Some(&c) = ends.first() {
                let (e, which) = (c >> 1, c & 1);
                let s = if which == 0 { 0.0 } else { self.w.net.edge_len[e as usize] };
                if let Some((twin, ts)) = self.twin_at(e, s) {
                    let r = self.station_ops(None, &[(u32::MAX, id as f64), (twin, ts)], &st, dry);
                    if r == 0 {
                        return 0;
                    }
                }
            }
            return self.station_ops(None, &[(u32::MAX, id as f64)], &st, dry);
        }
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        let s = self.w.net.project(id, x, y).0;
        // On quad track (T-255) the pair beside gets a station of the same name level with it:
        // the local and express platforms of one station, a complex. If that cannot be built,
        // the one station alone.
        if let Some((twin, ts)) = self.twin_at(id, s) {
            let r = self.station_ops(None, &[(id, s), (twin, ts)], &st, dry);
            if r == 0 {
                return 0;
            }
        }
        self.station_ops(None, &[(id, s)], &st, dry)
    }

    /// A station on open ground (T-794): a straight piece of new track between two free `ends` (as
    /// `add_route`'s, both kind 0) and a station at its middle, one undoable edit. The piece's ends
    /// are ordinary loose ends. On quad (`tracks` 4) the express pair gets the twin, as on drawn
    /// quad track. 0 when placed; otherwise `issue_kinds()` says why.
    pub fn add_free_station(&mut self, ends: &[f64], tracks: u8, platform: u16, name: String) -> u32 {
        self.free_station(ends, tracks, platform, name, false)
    }

    /// Whether `add_free_station` would succeed, tried and rolled back (the ghost); `station_cost`
    /// then holds what it would add to the blueprint, US$M.
    pub fn preview_free_station(&mut self, ends: &[f64], tracks: u8, platform: u16) -> u32 {
        self.free_station(ends, tracks, platform, String::new(), true)
    }

    /// What the last station tried (`preview_*station`) or placed adds to the blueprint, US$M.
    pub fn station_cost(&self) -> f64 {
        self.station_cost
    }

    pub(super) fn free_station(&mut self, ends: &[f64], tracks: u8, platform: u16, name: String, dry: bool) -> u32 {
        if ends.len() < 10 || ends[0] != 0.0 || ends[5] != 0.0 {
            return self.refuse(Issue::Geom { edge: u32::MAX, issue: GeomIssue::ZeroLeg(0) });
        }
        // through: a quad piece's express pair gets its own ends beside the drawn pair's
        let keep = self.route_tangent;
        self.route_tangent = true;
        let r = self.route_op(ends, &[], tracks, false);
        self.route_tangent = keep;
        let (route, _, nn, ne) = match r {
            Ok(v) => v,
            Err(i) => return self.refuse(i),
        };
        let inv = match self.w.sketch(route.clone(), false) {
            Ok(inv) => inv,
            Err(e) => {
                self.release(&nn, &ne);
                return self.done(Err(e));
            }
        };
        let (mx, my) = ((ends[2] + ends[7]) / 2.0, (ends[3] + ends[8]) / 2.0);
        let mid = ne.iter().filter(|&&e| self.w.net.edge_ok(e)).map(|&e| (e, self.w.net.project(e, mx, my))).min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map(|(e, (s, _))| (e, s));
        let twin = mid.and_then(|(e, s)| self.twin_at(e, s));
        // a piece this short crossing other track flat would put the crossing at the platform's end
        let flat = self.w.net.crossings.iter().any(|c| matches!(c.kind, CrossKind::Flat(_)) && (ne.contains(&c.e1) || ne.contains(&c.e2)));
        self.w.unsketch(inv);
        if flat {
            self.release(&nn, &ne);
            return self.refuse(Issue::PlatformCrossing { node: u32::MAX });
        }
        let Some((e, s)) = mid else {
            self.release(&nn, &ne);
            return self.refuse(Issue::Geom { edge: u32::MAX, issue: GeomIssue::ZeroLeg(0) });
        };
        let st = StationData { platform, name, built: false, off: false };
        let mut r = 1;
        if let Some((tw, ts)) = twin {
            r = self.station_ops(Some(&route), &[(e, s), (tw, ts)], &st, dry);
        }
        if r != 0 {
            r = self.station_ops(Some(&route), &[(e, s)], &st, dry);
        }
        if r != 0 || dry {
            self.release(&nn, &ne);
        }
        r
    }

    /// Stations splitting edges at chainages, one undoable edit (`add_station`). The joints of a
    /// drawn route (T-244, T-418) within a platform's reach go first: the pieces become one edge
    /// again, so the platform has room. `(u32::MAX, node)` puts one on an existing node instead.
    /// Two stations are a quad complex: a crossover goes in past the first where it fits (T-733).
    /// `route`: new track laid first in the same edit (`add_free_station`, T-794), sketched while
    /// the stations are planned on it.
    pub(super) fn station_ops(&mut self, route: Option<&Op>, at: &[(u32, f64)], st: &StationData, dry: bool) -> u32 {
        let mut ops = vec![];
        let sketched = match route {
            Some(r) => match self.w.sketch(r.clone(), false) {
                Ok(inv) => {
                    ops.push(r.clone());
                    Some(inv)
                }
                Err(e) => return self.done(Err(e)),
            },
            None => None,
        };
        let (mut nodes, mut edges) = (vec![], vec![]);
        let mut first = None;
        for &(e, s) in at {
            if e == u32::MAX {
                // an existing node, `s` its id; at a loose end the platform runs its whole length
                // back along the track, so the joints within it go first as they do for a split (T-860)
                let n = s as u32;
                let p = self.w.net.node_ports[n as usize];
                if p.n == 1 {
                    let (pe, which) = (p.ends()[0] >> 1, p.ends()[0] & 1);
                    let ps = if which == 0 { 0.0 } else { self.w.net.edge_len[pe as usize] };
                    if let Some((more, _)) = self.unjoin_near(pe, ps, st.platform as f64 + 2.0) {
                        ops.extend(more);
                    }
                }
                ops.push(Op::Station { node: n, data: Some(st.clone()) });
                first.get_or_insert(n);
                continue;
            }
            let (more, s) =self.unjoin_near(e, s, st.platform as f64 / 2.0 + 2.0).unwrap_or((vec![], s));
            ops.extend(more);
            let (node, new_edge) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
            nodes.push(node);
            edges.push(new_edge);
            first.get_or_insert(node);
            ops.extend([Op::Split { edge: e, s, node, new_edge }, Op::Station { node, data: Some(st.clone()) }]);
        }
        if let Some(inv) = sketched {
            self.w.unsketch(inv);
        }
        let base = Op::Batch(ops);
        let mut r = 1;
        if at.len() == 2 {
            let (op, nn, ne) = self.with_crossovers(base.clone(), &[first.unwrap()]);
            if !ne.is_empty() {
                r = self.attempt(op, dry);
                if r != 0 || dry {
                    self.release(&nn, &ne);
                }
            }
        }
        if r != 0 {
            // without the crossover: one that will not fit, or the cash short for it, is skipped
            r = self.attempt(base, dry);
        }
        if r != 0 || dry {
            self.release(&nodes, &edges);
        }
        r
    }

    /// `base` and one crossover just past each of `stations` (nodes `base` leaves on quad track)
    /// where it fits, as one edit (T-733); also the ids the crossovers claim (none: `base` alone).
    pub(super) fn with_crossovers(&mut self, base: Op, stations: &[u32]) -> (Op, Vec<u32>, Vec<u32>) {
        if stations.is_empty() {
            return (base, vec![], vec![]);
        }
        let Ok(inv) = self.w.sketch(base.clone(), false) else { return (base, vec![], vec![]) };
        let (mut ops, mut invs, mut nn, mut ne) = (vec![base], vec![], vec![], vec![]);
        for &n in stations {
            if let Some((op, back, n2, e2)) = self.crossover_past(n) {
                ops.push(op);
                invs.push(back);
                nn.extend(n2);
                ne.extend(e2);
            }
        }
        for back in invs.into_iter().rev() {
            self.w.unsketch(back);
        }
        self.w.unsketch(inv);
        (Op::Batch(ops), nn, ne)
    }

    /// A crossover past station `n` on the sketched track, left sketched (with its inverse): off
    /// the station's first end, else its second; None where neither has room.
    #[allow(clippy::type_complexity)]
    pub(super) fn crossover_past(&mut self, n: u32) -> Option<(Op, Op, Vec<u32>, Vec<u32>)> {
        let d = self.w.net.platform_reach(n) + QUAD_CROSSOVER_CLEAR + CROSSOVER_LEN / 2.0;
        let ends = self.w.net.node_ports[n as usize].ends().to_vec();
        for c in ends {
            let net = &self.w.net;
            let (e, which) = (c >> 1, c & 1);
            let s = if which == 0 { d } else { net.edge_len[e as usize] - d };
            let (edge, x, y) = match locate(net, e, s) {
                Some(Loc::Split(e, s)) => {
                    let (x, y, _) = geom::pos_at(net.edge_pieces(e), s);
                    (e, x, y)
                }
                Some(Loc::Node(m)) => (net.node_ports[m as usize].ends()[0] >> 1, net.node_x[m as usize], net.node_y[m as usize]),
                None => continue,
            };
            let Ok((op, nn, ne, _)) = self.crossover_op(edge, x, y) else { continue };
            match self.w.sketch(op.clone(), true) {
                Ok(back) => return Some((op, back, nn, ne)),
                Err(_) => self.release(&nn, &ne),
            }
        }
        None
    }

    /// The track beside edge `e` at chainage `s` on quad track (T-255): another edge running
    /// parallel at about `QUAD_SPACING`, and the chainage on it level with `s`.
    pub(super) fn twin_at(&self, e: u32, s: f64) -> Option<(u32, f64)> {
        let net = &self.w.net;
        let (x, y, th) = geom::pos_at(net.edge_pieces(e), s);
        let mut best: Option<(u32, f64, f64)> = None;
        for f in 0..net.edge_count() as u32 {
            if f == e || !net.edge_ok(f) {
                continue;
            }
            let (sf, dist) = net.project(f, x, y);
            if !(QUAD_SPACING * 0.5..=QUAD_SPACING * 1.5).contains(&dist) {
                continue;
            }
            let thf = geom::pos_at(net.edge_pieces(f), sf).2;
            if (th - thf).sin().abs() > 0.03 || best.is_some_and(|b| b.2 <= dist) {
                continue;
            }
            best = Some((f, sf, dist));
        }
        let (f, sf, _) = best?;
        // level with a loose end or a joint of the pair beside (quad drawn through points, T-423):
        // the twin goes on that node, as a split needs room from it
        let len = net.edge_len[f as usize];
        for (which, at) in [(0, 0.0), (1, len)] {
            let n = net.end_node(f, which);
            let ports = net.node_ports[n as usize].n;
            let near = (sf - at).abs();
            if net.node_platform[n as usize] == 0 && ((ports == 1 && near < JOINT_SNAP) || (ports == 2 && near < 1.0)) {
                return Some((u32::MAX, n as f64));
            }
        }
        Some((f, sf))
    }

}

/// How far past a platform's end track drawn from the station branches off, m (T-1155): beyond
/// the `JOINT_SNAP` a joint snaps within, and short of a quad station's crossover 50 m out.
pub(super) const BRANCH_CLEAR: f64 = 20.0;

/// T-1155: a route end on a station's node or on track within its platform, as the end the route
/// takes instead, with the station's position when the route branches off (so it leaves away
/// from the station). A station node takes no junction, so the route leaves the track
/// `BRANCH_CLEAR` past the platform's end on the side `toward` lies (or a plain joint within
/// `JOINT_SNAP` of that), or carries on from a track end short of it. A station at the end of its
/// track, clicked on its node or with `toward` out past that end, is extended as it is. Any other
/// end is unchanged.
fn branch_end(net: &Network, end: End, toward: (f64, f64)) -> Result<(End, Option<(f64, f64)>), Issue> {
    let station = |n: u32| net.node_ok(n) && net.node_platform[n as usize] > 0 && net.node_ports[n as usize].n > 0;
    let n = match end {
        End::Node(n) if station(n) && net.node_ports[n as usize].n >= 2 => n,
        End::Edge { e, x, y } if net.edge_ok(e) => {
            let (s, _) = net.project(e, x, y);
            let len = net.edge_len[e as usize];
            let within = |(w, d): &(u32, f64)| {
                let m = net.end_node(e, *w);
                station(m) && *d < net.platform_reach(m) + 1.0
            };
            match [(0, s), (1, len - s)].into_iter().filter(within).min_by(|a, b| a.1.total_cmp(&b.1)) {
                Some((w, _)) => net.end_node(e, w),
                None => return Ok((end, None)),
            }
        }
        _ => return Ok((end, None)),
    };
    let i = n as usize;
    let p = net.node_ports[i];
    let (sx, sy, h) = (net.node_x[i], net.node_y[i], net.node_heading[i]);
    // the port on the side `toward` lies (side 1 leaves along the heading)
    let want = if (toward.0 - sx) * h.cos() + (toward.1 - sy) * h.sin() >= 0.0 { 1 } else { 0 };
    let Some(k) = (0..p.n as usize).find(|&k| p.side[k] == want) else {
        return Ok((End::Node(n), None));
    };
    let (e, which) = (p.end[k] >> 1, p.end[k] & 1);
    let d = net.platform_reach(n) + BRANCH_CLEAR;
    let s = if which == 0 { d } else { net.edge_len[e as usize] - d };
    let room = Issue::BranchNoRoom { node: n };
    match locate_or_stop(net, e, s) {
        Ok(Loc::Split(e2, s2)) => {
            // clear of another station's platform reaching in along that piece
            let len = net.edge_len[e2 as usize];
            for (w, dist) in [(0, s2), (1, len - s2)] {
                let m = net.end_node(e2, w);
                if station(m) && dist < net.platform_reach(m) + 1.0 {
                    return Err(room);
                }
            }
            let (x, y, _) = geom::pos_at(net.edge_pieces(e2), s2);
            Ok((End::Edge { e: e2, x, y }, Some((sx, sy))))
        }
        Ok(Loc::Node(m)) => Ok((End::Node(m), Some((sx, sy)))),
        Err(m) if m != u32::MAX && net.node_ports[m as usize].n == 1 && net.node_platform[m as usize] == 0 => Ok((End::Node(m), None)),
        Err(_) => Err(room),
    }
}

/// A branch past a platform (`branch_end`) leaves its junction away from the station at `from`.
fn leave_away(r: &mut Resolved, from: Option<(f64, f64)>) {
    if let Some((sx, sy)) = from {
        r.dirs.retain(|d| d.cos() * (r.x - sx) + d.sin() * (r.y - sy) > 0.0);
    }
}
