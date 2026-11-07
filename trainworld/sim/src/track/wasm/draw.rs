//! The drawing tool: routes, previews and reshaping a blueprint edge.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- the drawing tool (T-023, T-092)

    /// `ends`: start then end, 5 numbers each: kind (0 free, 1 node, 2 edge), id, x, y, level
    /// (level used for a free end). `pis`: the player's PIs between, x, y, radius (0 = auto),
    /// level: the track cuts the corner at each with a circular curve (SPEC 6.1). A start or end on
    /// existing track gets a lead-in PI on that track's heading, so it leaves like a turnout.
    /// T-1615: with `route_ramp`, a route crossing track at its own level ramps over or under it,
    /// the preferred way, else the other, else crosses flat: the first whose edit passes the checks.
    pub(super) fn route_op(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Result<(Op, Vec<Pi>, Vec<u32>, Vec<u32>), Issue> {
        if let Some(eb) = self.route_beside.take() {
            if let Some((b_ends, b_pis, on_line, run)) = pis.is_empty().then(|| self.beside_ends(ends, eb)).flatten() {
                // a start on the line leaves along it; one coming in keeps its lead-in
                self.beside_on = on_line;
                let r = self.route_op_ramped(&b_ends, &b_pis, tracks, flying);
                self.beside_on = false;
                // T-1895: never across the track followed, lead-ins included; the plain route
                // instead (a start on that track meets it there)
                let scaled = self.w.net.frame().scaled();
                let crossed = |verts: &[Pi]| {
                    let skip = 0.5 * beside::BESIDE_SPACING * kk(scaled, verts[0].y);
                    let on_run = run.iter().any(|&e| self.w.net.edge_ok(e) && self.w.net.project(e, verts[0].x, verts[0].y).1 < skip);
                    run.iter().any(|&e| self.w.net.edge_ok(e) && beside::route_crosses(verts, self.w.net.edge_pieces(e), scaled, if on_run { 2.0 * skip } else { 0.0 }))
                };
                match &r {
                    Ok((_, verts, nn, ne)) if crossed(verts) => {
                        let (nn, ne) = (nn.clone(), ne.clone());
                        self.release(&nn, &ne);
                    }
                    _ => return r,
                }
            }
        }
        self.route_op_ramped(ends, pis, tracks, flying)
    }

    fn route_op_ramped(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Result<(Op, Vec<Pi>, Vec<u32>, Vec<u32>), Issue> {
        if self.route_ramp {
            for t in [Ramp::First, Ramp::Second] {
                self.ramp_try = t;
                (self.ramp_found, self.ramp_failed) = (false, false);
                let r = self.route_op_as(ends, pis, tracks, flying);
                if !self.ramp_found {
                    self.ramp_try = Ramp::Flat;
                    return r;
                }
                if let Ok((op, _, nn, ne)) = &r {
                    if !self.ramp_failed {
                        if let Ok(inv) = self.w.sketch(op.clone(), true) {
                            self.w.unsketch(inv);
                            self.ramp_try = Ramp::Flat;
                            return r;
                        }
                    }
                    let (nn, ne) = (nn.clone(), ne.clone());
                    self.release(&nn, &ne);
                }
            }
        }
        self.ramp_try = Ramp::Flat;
        self.route_op_as(ends, pis, tracks, flying)
    }

    fn route_op_as(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Result<(Op, Vec<Pi>, Vec<u32>, Vec<u32>), Issue> {
        let pis = pis_from(pis);
        let scaled = self.w.net.frame().scaled();
        let (a, b) = (end_from(&ends[0..5]), end_from(&ends[5..10]));
        let lift = [matches!(a, End::Free { .. }), matches!(b, End::Free { .. })];
        let mut new_nodes = vec![];
        let mut new_edges = vec![];
        let net = &self.w.net;
        let station = |end: End| matches!(end, End::Node(n) if net.node_ok(n) && net.node_platform[n as usize] > 0);
        let mut at_station = [station(a), station(b)];
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
        let existing = [matches!(a, End::Node(_)), matches!(b, End::Node(_))];
        at_station = [at_station[0] || from_a.is_some(), at_station[1] || from_b.is_some()];
        let mut rb = self.resolve(b, first_split, &mut new_nodes, &mut new_edges)?;
        leave_away(&mut rb, from_b);
        if ra.node == rb.node {
            self.release(&new_nodes, &new_edges);
            return Err(Issue::EdgeLoop { edge: u32::MAX });
        }
        self.route_arc.clear();
        self.route_bridged.clear();
        self.bridge_refused = false;
        self.bridge_short = Issue::BridgeNoRoom;
        // T-418: ending on open ground, the segment is one arc through the end; ending on track it
        // must join that like a turnout, so it keeps the usual shape
        let quad = tracks == 4;
        let through = self.route_tangent && pis.is_empty();
        // T-423: quad drawn through points carries both pairs on from a loose quad end, the other
        // pair's end beside the one clicked (on either side); from other track it swings out as in
        // Corners, which needs a straight, so that segment is not one arc
        let beside_a = if quad && through { quad_beside(&self.w.net, ra.node) } else { None };
        let side = beside_a.map_or(1.0, |b| b.1);
        let mut beside_b = if quad && through { quad_beside(&self.w.net, rb.node).filter(|b| b.1 == -side) } else { None };
        let mut beside_a = beside_a;
        // a dead-end station keeps its platform where it is (T-1667); on quad, its twin too. `own`
        // reads a moved twin end's position from `was`, its node not being in the network yet.
        let mut was = vec![];
        for (r, beside) in [(&mut ra, &mut beside_a), (&mut rb, &mut beside_b)] {
            if let Some((ops, m, x, y)) = self.end_station_shift(r.node, &mut new_nodes, &mut new_edges) {
                r.ops.extend(ops);
                (r.node, r.x, r.y) = (m, x, y);
            }
            if let Some(b) = beside.as_mut() {
                if let Some((ops, m, x, y)) = self.end_station_shift(b.0, &mut new_nodes, &mut new_edges) {
                    r.ops.extend(ops);
                    was.push((m, x, y));
                    b.0 = m;
                }
            }
        }
        self.route_end = rb.node;
        let mut verts: Vec<Pi> = vec![Pi::new(ra.x, ra.y, 0.0, ra.level)];
        let next_a = pis.first().map_or((rb.x, rb.y), |p| (p.x, p.y));
        let prev_b = pis.last().map_or((ra.x, ra.y), |p| (p.x, p.y));
        let tangent = through && rb.dirs.is_empty() && (!quad || beside_a.is_some()) && !self.beside_on;
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
        } else if let Some(p) = lead_in(&ra, next_a, scaled, self.route_cap).filter(|_| !self.beside_on) {
            verts.push(p);
        }
        let first_click = verts.len();
        verts.extend_from_slice(&pis);
        let lead_b = lead_in(&rb, prev_b, scaled, self.route_cap).filter(|_| !self.beside_on);
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
        let mut bridged = self.shaped(&verts, lift);
        // T-2414: no room for a ramp between a dead end of blueprint track and the water: that end
        // goes a level up, its track ramping up to it, and the bridge starts there. One end, then
        // the other, then both.
        if self.bridge_refused && !quad {
            let raise = [existing[0].then(|| self.raise_end(ra.node)).flatten(), existing[1].then(|| self.raise_end(rb.node)).flatten()];
            for which in [[true, false], [false, true], [true, true]] {
                if (0..2).any(|k| which[k] && raise[k].is_none()) {
                    continue;
                }
                let mut up = verts.clone();
                let last = up.len() - 1;
                for (k, i) in [(0, 0), (1, last)] {
                    if which[k] {
                        up[i].level += 1;
                    }
                }
                self.bridge_refused = false;
                let b = self.shaped(&up, lift);
                if !self.bridge_refused && b.is_some() {
                    for (k, r) in [(0, &mut ra), (1, &mut rb)] {
                        if let (true, Some(o)) = (which[k], &raise[k]) {
                            ops.extend(o.iter().cloned());
                            r.level += 1;
                        }
                    }
                    (verts, bridged) = (up, b);
                    break;
                }
                self.bridge_refused = true;
            }
        }
        if self.bridge_refused {
            self.bridge_short = self.short_end(&verts, [&ra, &rb], lift, at_station);
        }
        if let Some(b) = &bridged {
            lift_end(&mut ops, &mut ra, b[0].level);
            lift_end(&mut ops, &mut rb, b[b.len() - 1].level);
        }
        let cut =|b: &Option<Vec<Pi>>, n: usize, clicks: usize| match b {
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
                    let at = was.iter().find(|w| w.0 == m).map_or_else(|| (net.node_x[m as usize], net.node_y[m as usize]), |w| (w.1, w.2));
                    return Some((m, at));
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
            // the express pair crosses the water on its own bridge, in one piece; an end it shares
            // with the drawn pair stays up only where that one does
            let ex_lift = [lift[0] && beside_a.is_none(), lift[1] && beside_b.is_none()];
            let ex_bridged = self.shaped(&ex, ex_lift);
            let got = ex_bridged.as_deref().unwrap_or(&ex);
            let ex_ends = [got[0].level, got[got.len() - 1].level];
            for (k, own, r) in [(0, own_a, &ra), (1, own_b, &rb)] {
                match own {
                    Some((m, _)) if ex_lift[k] => set_node_level(&mut ops, m, ex_ends[k]),
                    None if ex_ends[k] != r.level => {
                        self.release(&new_nodes, &new_edges);
                        return Err(Issue::BridgeNoRoom);
                    }
                    _ => {}
                }
            }
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

    /// Track drawn on from a station at a dead end (T-1667): that platform runs back from the node,
    /// a through one is centred on it, so the station would move half a platform. Instead the
    /// station's node moves back to the platform's middle, keeping its id (stops, names), and a new
    /// plain node just past the old end takes the route. The ops, that node and where it is; None
    /// for any other node, or where the edge cannot be split there.
    pub(super) fn end_station_shift(&mut self, n: u32, nodes: &mut Vec<u32>, edges: &mut Vec<u32>) -> Option<(Vec<Op>, u32, f64, f64)> {
        let net = &self.w.net;
        let i = n as usize;
        if !net.node_ok(n) || net.node_platform[i] == 0 || net.node_ports[i].n != 1 {
            return None;
        }
        let c = net.node_ports[i].ends()[0];
        let (e, at_b) = (c >> 1, c & 1 == 1);
        if net.edge_a[e as usize] == net.edge_b[e as usize] {
            return None;
        }
        let (half, len) = (net.node_platform[i] as f64 / 2.0, net.edge_len[e as usize]);
        let (mut nd, mut first, mut second) = net.split_plan(e, if at_b { len - half } else { half }).ok()?;
        let mut old = net.node_data(n)?;
        nd.level = old.level;
        // the stub runs on straight past the old end until the platform clears its far node by
        // the 1 m `check_station` wants (a split snapped toward the end leaves it shorter)
        let stub_len = if at_b { len - net.project(e, nd.x, nd.y).0 } else { net.project(e, nd.x, nd.y).0 };
        let on = (half + 1.5 - stub_len).max(0.0) + 0.5;
        let inner = if at_b { second.pis.last() } else { first.pis.first() }.map_or((nd.x, nd.y), |p| (p.x, p.y));
        let (dx, dy) = (old.x - inner.0, old.y - inner.1);
        let d = dx.hypot(dy);
        if d < 1e-6 {
            return None;
        }
        (old.x, old.y) = (old.x + dx / d * on, old.y + dy / d * on);
        let (m, e2) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
        nodes.push(m);
        edges.push(e2);
        // `e` keeps the part the station's lines run on, ending at `n`; `e2` is the stub past it
        let (keep, stub) = if at_b {
            (first.b, second.a, second.b) = (n, n, m);
            (first, second)
        } else {
            (first.a, first.b, second.a) = (m, n, n);
            (second, first)
        };
        let ops = vec![Op::Node { id: m, data: Some(old.clone()) }, Op::Node { id: n, data: Some(nd) }, Op::Edge { id: e, data: Some(keep) }, Op::Edge { id: e2, data: Some(stub) }];
        Some((ops, m, old.x, old.y))
    }

    /// T-656: `shape` as a bridge over any water it crosses at ground level, or None where it needs
    /// none. With no room on land for the ramps it is left as drawn and `bridge_refused` set, so the
    /// edit fails and says why (`bridge_refusal`). `lift`: which ends are new nodes, free to stay
    /// up on the bridge where they are over water (T-1269).
    pub(super) fn bridged(&mut self, shape: &[Pi], lift: [bool; 2]) -> Option<Vec<Pi>> {
        let scaled = self.w.net.frame().scaled();
        let mask = self.w.net.water_mask.as_ref();
        // a snapped real route (T-727d) bridges each river on its own
        let b = if shape.len() > LONG_ROUTE_VERTS { bridge::bridge_long(shape, scaled, mask, lift) } else { bridge::bridge(shape, scaled, mask, lift) };
        match b {
            Ok(b) => b,
            Err(bridge::NoRoom) => {
                self.bridge_refused = true;
                None
            }
        }
    }

    /// T-2414: what a bridge refusal names: the end of `shape` (start first) with water within a
    /// ramp and its shore, where the end cannot move up (a station, a junction, built track); else
    /// `BridgeNoRoom`. `free`: which ends are new nodes; `station`: which are at a station.
    fn short_end(&self, shape: &[Pi], ends: [&Resolved; 2], free: [bool; 2], station: [bool; 2]) -> Issue {
        let net = &self.w.net;
        let f = geom::fit_in(shape, net.frame().scaled());
        let wet = bridge::water(net.water_mask.as_ref())(&f, false);
        let (Some(first), Some(last)) = (wet.first(), wet.last()) else { return Issue::BridgeNoRoom };
        let reach = RAMP_PER_LEVEL + 2.0 * WATER_STEP;
        for k in 0..2 {
            let near = if k == 0 { first[0] < reach } else { f.len - last[1] < reach };
            if !near || free[k] {
                continue;
            }
            let r = ends[k];
            let i = r.node as usize;
            let on_track = net.node_ok(r.node);
            if station[k] {
                return Issue::BridgeNoRoomStation;
            }
            if r.new_junction || on_track && net.node_ports[i].n >= 2 {
                return Issue::BridgeNoRoomJunction;
            }
            if on_track && net.node_ports[i].ends().iter().any(|&c| net.edge_built[(c >> 1) as usize]) {
                return Issue::BridgeNoRoomBuilt;
            }
        }
        Issue::BridgeNoRoom
    }

    /// T-2414: the ops that put `n`, a dead end of blueprint track at ground level (no station), a
    /// level up, its piece of track ramping up to it as near it as the bridge's ramps fit. The
    /// piece's corners are at ground level or one up (a bridge, or a ramp over track); what is up
    /// now stays up, its ramps laid again, so a bridge just before `n` carries on to it. None for
    /// any other node, or where the ramp has no room on that piece.
    pub(super) fn raise_end(&self, n: u32) -> Option<Vec<Op>> {
        if !self.dead_end_on_ground(n) {
            return None;
        }
        let net = &self.w.net;
        let e = net.node_ports[n as usize].ends()[0] >> 1;
        let mut d = net.edge_data(e)?;
        if d.built || d.crossover || !d.thru.is_empty() || d.a == d.b || d.pis.iter().any(|p| !(0..=1).contains(&p.level)) {
            return None;
        }
        let at_b = d.b == n;
        let end = |m: u32| net.node_data(m).map(|p| Pi::new(p.x, p.y, 0.0, p.level));
        let mut now: Vec<Pi> = std::iter::once(end(d.a)?).chain(d.pis.iter().copied()).chain(std::iter::once(end(d.b)?)).collect();
        if !at_b {
            now.reverse();
        }
        let scaled = net.frame().scaled();
        // the stretches up now, along the piece from its far end
        let g = geom::fit_in(&now, scaled);
        let mut held: Vec<[f64; 2]> = vec![];
        let mut s = 0.0;
        while s <= g.len {
            if geom::height_at(&g.vert, s) >= LEVEL_H - 0.01 {
                match held.last_mut() {
                    Some(h) if s - h[1] < 2.5 => h[1] = s,
                    _ => held.push([s, s]),
                }
            }
            s += 2.0;
        }
        let last = now.len() - 1;
        let shape: Vec<Pi> = now.iter().enumerate().map(|(i, p)| if i == 0 || i == last { *p } else { Pi { level: 0, ..*p } }).collect();
        let water = bridge::water(net.water_mask.as_ref());
        let near_end = |f: &geom::Fit, fine: bool| {
            let mut s = water(f, fine);
            s.extend(held.iter().copied());
            s.push([f.len - 1e-3, f.len]);
            s.sort_by(|a, b| a[0].total_cmp(&b[0]));
            s
        };
        let up_node = |m: u32| net.node_data(m).map(|p| Op::Node { id: m, data: Some(NodeData { level: 1, ..p }) });
        let far = if at_b { d.a } else { d.b };
        let mut ops = vec![up_node(n)?];
        match bridge::ramp_over(&shape, scaled, &near_end, 0, true, [false, true], shape.len() > LONG_ROUTE_VERTS) {
            Ok(Some(mut up)) => {
                if !at_b {
                    up.reverse();
                }
                d.pis = up[1..up.len() - 1].to_vec();
            }
            // a piece too short for the ramp from another dead end goes up whole, as a new end
            // with no room would have
            _ if self.dead_end_on_ground(far) && now[0].level == 0 => {
                for p in &mut d.pis {
                    p.level = 1;
                }
                ops.extend(up_node(far));
            }
            _ => return None,
        }
        ops.push(Op::Edge { id: e, data: Some(d) });
        Some(ops)
    }

    /// A plain node at ground level with one track end, no station (T-2414).
    fn dead_end_on_ground(&self, n: u32) -> bool {
        let net = &self.w.net;
        let i = n as usize;
        net.node_ok(n) && net.node_platform[i] == 0 && net.node_ports[i].n == 1 && net.node_level[i] == 0
    }

    /// `shape` as `bridged` makes it; with a ramp to try (T-1615) and track crossed at the shape's
    /// own level, ramped over or under that track instead: up first from level 0 or above, down
    /// first from -1 or below. Going up from level 0 clears the water in the same pass. Sets
    /// `ramp_found` where there is such track, and `ramp_failed` where the ramps have no room.
    pub(super) fn shaped(&mut self, shape: &[Pi], lift: [bool; 2]) -> Option<Vec<Pi>> {
        let n = shape.len();
        if self.ramp_try == Ramp::Flat || n < 2 {
            return self.bridged(shape, lift);
        }
        let base = if n > 2 { shape[n / 2].level } else if shape[0].level == shape[1].level { shape[0].level } else { return self.bridged(shape, lift) };
        let net = &self.w.net;
        let scaled = net.frame().scaled();
        let f = geom::fit_in(shape, scaled);
        if !f.issues.is_empty() || flat_spans(net, &f, base, base as f64 * LEVEL_H).is_empty() {
            return self.bridged(shape, lift);
        }
        self.ramp_found = true;
        let up = (base >= 0) == (self.ramp_try == Ramp::First);
        let wet = up && base == 0;
        let water = bridge::water(net.water_mask.as_ref());
        // seen from `ramp_over`, the route runs at level 0. T-1708: first with room for a station
        // beside each crossed track's station, then without.
        let mut r = Err(bridge::NoRoom);
        for room in [true, false] {
            let spans = |f: &geom::Fit, fine: bool| {
                let mut s = flat_spans(net, f, base, 0.0);
                if room {
                    for w in &mut s {
                        let (x, y, _) = geom::pos_at(&f.pieces, w[0]);
                        let h = station_room(net, x, y);
                        (w[0], w[1]) = ((w[0] - h).max(0.0), (w[1] + h).min(f.len));
                    }
                }
                if wet {
                    s.extend(water(f, fine));
                    s.sort_by(|a, b| a[0].total_cmp(&b[0]));
                }
                s
            };
            r = bridge::ramp_over(shape, scaled, &spans, base, up, if wet { lift } else { [false; 2] }, n > LONG_ROUTE_VERTS);
            if r.is_ok() {
                break;
            }
        }
        // a ramp onto a flat crossing at the next level does not fit either
        let to = base + if up { 1 } else { -1 };
        match r {
            Ok(Some(v)) if flat_spans(net, &geom::fit_in(&v, scaled), to, to as f64 * LEVEL_H).is_empty() => Some(v),
            _ => {
                self.ramp_failed = true;
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
        let r = if self.bridge_refused { r.map_err(|i| bridge_refusal(i, &self.bridge_short)) } else { r };
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
    /// `design_radius(kmh)`, and so are corners a reshape or node drag moves or adds (T-1898).
    /// Clamped to 160..320; 160 is the drawing tool as before.
    pub fn set_design_speed(&mut self, kmh: f64) {
        let kmh = if kmh.is_finite() { kmh.clamp(DESIGN_SPEEDS[0], DESIGN_SPEEDS[DESIGN_SPEEDS.len() - 1]) } else { DESIGN_SPEEDS[0] };
        self.route_cap = design_radius(kmh);
    }

    /// T-817: whether double track drawn after this (routes and stations on open ground) is
    /// high-speed track. Single and quad track are never.
    pub fn set_route_hsr(&mut self, on: bool) {
        self.route_hsr = on;
    }

    /// T-1615: whether routes after this ramp over or under track they would cross flat.
    pub fn set_route_ramp(&mut self, on: bool) {
        self.route_ramp = on;
    }

    /// T-1685: the next route (preview or add) runs beside track: `to_edge` the piece its end is
    /// beside (u32::MAX for none; `beside_ends`). `from_edge` is unused since T-1721, where a start
    /// anywhere may come in to the line. Used once.
    pub fn set_route_beside(&mut self, _from_edge: u32, to_edge: u32) {
        self.route_beside = (to_edge != u32::MAX).then_some(to_edge);
    }

    /// The circle of the last route preview's tangent arc (T-418): centre x, y and ground radius,
    /// even when the route was refused; empty when it had none.
    pub fn route_arc(&self) -> Vec<f64> {
        self.route_arc.clone()
    }

    /// A real railway's line as corners for `add_route` (T-727, `trace.rs`): `lonlat` the line as
    /// lon, lat pairs from one end to the other, `tol` how far the track may lie from it (ground
    /// m), quad track (`tracks` 4) keeping its inner pair at 100 m, curves laid out for the design
    /// speed (`set_design_speed`, T-1290). Returns [worst, mean distance
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
        RouteTrace { t: crate::track::trace::Tracer::new(&line, f.scaled(), tol, r_min, self.route_cap, level) }
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
            ops.push(Op::Edge { id: edge, data: Some(EdgeData { a: from, b: to, tracks, pis: p[1..p.len() - 1].to_vec(), built: false, off: false, thru: vec![], crossover: false, hsr, sep: false }) });
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
                self.issues = bridge_refusal(std::mem::take(&mut self.issues), &self.bridge_short);
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
        let mut st = StationData { platform, name, named: false, built: false, off: false, upgrades: 0, berth: None };
        // On quad track only the pair clicked gets the station (T-1687); one level with it on the
        // pair beside lends its name, so the two read as one station.
        if kind == 1 {
            if !self.w.net.node_ok(id) {
                return self.refuse(Issue::NoSuchNode { node: id });
            }
            if self.w.net.node_platform[id as usize] > 0 {
                return self.refuse(Issue::StationPorts { node: id });
            }
            if let Some(&c) = self.w.net.node_ports[id as usize].ends().first() {
                let (e, which) = (c >> 1, c & 1);
                let s = if which == 0 { 0.0 } else { self.w.net.edge_len[e as usize] };
                if let Some(n) = self.station_beside(e, s) {
                    st.name = self.w.net.node_name[n as usize].clone();
                    st.named = self.w.net.node_named[n as usize];
                }
            }
            return self.station_ops(None, &[(u32::MAX, id as f64)], &st, dry);
        }
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        let s = self.w.net.project(id, x, y).0;
        if let Some(n) = self.station_beside(id, s) {
            st.name = self.w.net.node_name[n as usize].clone();
            st.named = self.w.net.node_named[n as usize];
        }
        self.station_ops(None, &[(id, s)], &st, dry)
    }

    /// The station on the pair beside edge `e` (quad track) whose platform is level with chainage `s`.
    fn station_beside(&self, e: u32, s: f64) -> Option<u32> {
        let (f, fs) = self.twin_at(e, s)?;
        let net = &self.w.net;
        if f == u32::MAX {
            return net.is_station(fs as u32).then_some(fs as u32);
        }
        let (x, y, _) = geom::pos_at(net.edge_pieces(f), fs);
        [net.edge_a[f as usize], net.edge_b[f as usize]]
            .into_iter()
            .filter(|&n| net.is_station(n))
            .map(|n| (n, (net.node_x[n as usize] - x).hypot(net.node_y[n as usize] - y)))
            .filter(|&(n, d)| d <= net.node_platform[n as usize] as f64 / 2.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(n, _)| n)
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
        // a platform is level, so the piece never ramps over track it crosses
        let keep = (self.route_tangent, self.route_ramp);
        (self.route_tangent, self.route_ramp) = (true, false);
        let r = self.route_op(ends, &[], tracks, false);
        (self.route_tangent, self.route_ramp) = keep;
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
        let st = StationData { platform, name, named: false, built: false, off: false, upgrades: 0, berth: None };
        // on quad, the drawn pair's only (T-1687)
        let r = self.station_ops(Some(&route), &[(e, s)], &st, dry);
        if r != 0 || dry {
            self.release(&nn, &ne);
        }
        r
    }

    /// Stations splitting edges at chainages, one undoable edit (`add_station`). The joints of a
    /// drawn route (T-244, T-418) within a platform's reach go first: the pieces become one edge
    /// again, so the platform has room. `(u32::MAX, node)` puts one on an existing node instead.
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
                continue;
            }
            let (more, s) = self.unjoin_near(e, s, st.platform as f64 / 2.0 + 2.0).unwrap_or((vec![], s));
            // within a split's reach of a joint that stays (unlike pieces, T-1964), on the joint
            if let Some(n) = self.joint_at(e, &more, s) {
                ops.extend(more);
                ops.push(Op::Station { node: n, data: Some(st.clone()) });
                continue;
            }
            ops.extend(more);
            let (node, new_edge) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
            nodes.push(node);
            edges.push(new_edge);
            ops.extend([Op::Split { edge: e, s, node, new_edge }, Op::Station { node, data: Some(st.clone()) }]);
        }
        if let Some(inv) = sketched {
            self.w.unsketch(inv);
        }
        let r = self.attempt(Op::Batch(ops), dry);
        if r != 0 || dry {
            self.release(&nodes, &edges);
        }
        // a split refused on a ramp is the platform's ramp, said as such (T-1964)
        if r != 0 && self.issues.iter().any(|i| matches!(i, Issue::SplitPoint { .. })) {
            let net = &self.w.net;
            let ramp = at.iter().any(|&(e, s)| e != u32::MAX && net.edge_ok(e) && geom::on_ramp(net.edge_vert(e), s - 3.0, s + 3.0));
            if ramp {
                self.issues.retain(|i| !matches!(i, Issue::SplitPoint { .. } | Issue::PlatformNotLevel { .. }));
                self.issues.push(Issue::PlatformNotLevel { node: u32::MAX });
            }
        }
        // a platform on a ramp is the reason; the shape a join under it gives that ramp is not
        // something the player drew (T-1917)
        if r != 0 && self.issues.iter().any(|i| matches!(i, Issue::PlatformNotLevel { .. })) {
            self.issues.retain(|i| !matches!(i, Issue::Geom { .. } | Issue::NodeHeading { .. }));
        }
        r
    }

    /// The joint a station at chainage `s` of edge `e` goes on (T-1964): one within a split's 5 m
    /// of the edge's end, after `joins` (`unjoin_near`'s ops, which may have made `e` longer), where
    /// two pieces meet that could not be joined.
    fn joint_at(&self, e: u32, joins: &[Op], s: f64) -> Option<u32> {
        let net = &self.w.net;
        let (mut a, mut b, mut len) = (net.edge_a[e as usize], net.edge_b[e as usize], net.edge_len[e as usize]);
        for op in joins {
            match op {
                Op::Edge { id, data: Some(d) } if *id == e => (a, b) = (d.a, d.b),
                Op::Edge { id, data: None } => len += net.edge_len[*id as usize],
                _ => {}
            }
        }
        let n = if s < 5.0 { a } else if s > len - 5.0 { b } else { return None };
        let p = &net.node_ports[n as usize];
        (p.n == 2 && net.node_platform[n as usize] == 0 && p.side[0] != p.side[1]).then_some(n)
    }

    /// The track beside edge `e` at chainage `s` on quad track (T-255): another edge running
    /// parallel at about `QUAD_SPACING` (or the wider `BESIDE_SPACING` of track drawn beside, T-1895),
    /// and the chainage on it level with `s`.
    pub(super) fn twin_at(&self, e: u32, s: f64) -> Option<(u32, f64)> {
        let net = &self.w.net;
        let (x, y, th) = geom::pos_at(net.edge_pieces(e), s);
        let k = net.k_at(y);
        let mut best: Option<(u32, f64, f64)> = None;
        for f in 0..net.edge_count() as u32 {
            if f == e || !net.edge_ok(f) {
                continue;
            }
            let (sf, dist) = net.project(f, x, y);
            if !(QUAD_SPACING * 0.5 * k..=beside::BESIDE_SPACING * 1.25 * k).contains(&dist) {
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

/// Which way `route_op` tries a route that crosses track at its own level (T-1615).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Ramp {
    Flat,
    /// up from level 0 or above, down from -1 or below
    First,
    Second,
}

/// Where alignment `f`, at height `own` there, crosses other track flat at level `base`, as spans
/// of no length; none within `SHARED_NODE_CLEAR` of its ends, where it meets the track it leaves.
fn flat_spans(net: &Network, f: &geom::Fit, base: i8, own: f64) -> Vec<[f64; 2]> {
    let z = base as f64 * LEVEL_H;
    let (mut out, mut hits) = (vec![], vec![]);
    for e in net.grid.near(&f.pieces) {
        if !net.edge_ok(e) {
            continue;
        }
        hits.clear();
        crate::track::cross::intersect_alignments(&f.pieces, net.edge_pieces(e), false, &mut hits);
        let v = net.edge_vert(e);
        for &(s, t, overlap) in &hits {
            if overlap || s < SHARED_NODE_CLEAR || s > f.len - SHARED_NODE_CLEAR {
                continue;
            }
            if (geom::height_at(&f.vert, s) - own).abs() < 1e-6 && (geom::height_at(v, t) - z).abs() < 1e-6 && !geom::on_ramp(v, t - 1.0, t + 1.0) {
                out.push([s, s]);
            }
        }
    }
    out.sort_by(|a, b| a[0].total_cmp(&b[0]));
    out
}

/// How far each way a crossing at (x, y) is padded (T-1708): with a station within
/// `STATION_ROOM_REACH`, half its platform and 10 m, so the bridge over the crossed track keeps a
/// level stretch for a station of the new track beside it; else 0.
fn station_room(net: &Network, x: f64, y: f64) -> f64 {
    let k = net.k_at(y);
    let reach = STATION_ROOM_REACH * k;
    let mut h: f64 = 0.0;
    for m in 0..net.node_count() {
        let p = net.node_platform[m];
        if p == 0 || !net.node_ok(m as u32) || (net.node_x[m] - x).abs() > reach || (net.node_y[m] - y).abs() > reach {
            continue;
        }
        if (net.node_x[m] - x).hypot(net.node_y[m] - y) <= reach {
            h = h.max((p as f64 / 2.0 + 10.0) * k);
        }
    }
    h
}

/// A station this near a crossing makes the bridge over it room for one (T-1708), ground m.
const STATION_ROOM_REACH: f64 = 400.0;

/// T-1269: a new node the bridge leaves up, at the level its track reaches.
fn lift_end(ops: &mut [Op], r: &mut Resolved, level: i8) {
    if r.level != level {
        r.level = level;
        set_node_level(ops, r.node, level);
    }
}

/// The level of the node `node` that `ops` create.
fn set_node_level(ops: &mut [Op], node: u32, level: i8) {
    for op in ops {
        if let Op::Node { id, data: Some(d) } = op {
            if *id == node {
                d.level = level;
            }
        }
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
