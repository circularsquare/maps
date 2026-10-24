//! New ids, the low-level edits and dragging a blueprint node.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- ids for new objects (claimed when the edit using them applies)
    pub fn new_line_id(&mut self) -> u32 {
        self.w.net.alloc_line()
    }

    pub(super) fn done(&mut self, r: Result<EditResult, Vec<Issue>>) -> u32 {
        self.money_short = None;
        match r {
            Ok(r) => {
                self.last = Some(r);
                self.issues.clear();
                0
            }
            Err(e) => {
                let n = e.len() as u32;
                self.issues = e;
                n.max(1)
            }
        }
    }

    /// An undoable edit. It pays for any trains it makes the running lines need (T-028), and is
    /// refused if the cash is short.
    pub(super) fn run(&mut self, op: Op) -> u32 {
        let fleet = self.fleet;
        let at = self.edit_start();
        let before = self.w.svc.before_edit(&self.w.net, at);
        let cash = self.spendable();
        let r = self.w.apply_paid(op, cash, false, &|w| trains_to_buy(w, fleet));
        self.settle(r, &before)
    }

    /// A final, charged edit (see the module comment), trains included.
    pub(super) fn run_charged(&mut self, op: Op) -> u32 {
        let fleet = self.fleet;
        let at = self.edit_start();
        let before = self.w.svc.before_edit(&self.w.net, at);
        let cash = self.spendable();
        let r = self.w.apply_paid(op, cash, true, &|w| trains_to_buy(w, fleet));
        self.settle(r, &before)
    }

    /// The game time now, from the clock worker (T-206): an edit that changes a line's departures
    /// takes effect from here, trains already running keep their places, and patterns from before
    /// are forgotten once their trains are no longer drawn.
    pub fn set_now(&mut self, t: f64) {
        self.now = t;
        self.w.svc.prune_eras(&self.w.net, t);
    }

    /// T-1071: the next edit takes effect this many game seconds after `now`, so its new trips
    /// reach the screen before anything they change is drawn (the clock worker's estimate of its
    /// own delay, at the clock's rate; 0 when paused). Used once, then back to 0.
    pub fn set_edit_lead(&mut self, s: f64) {
        self.lead = if s.is_finite() { s.max(0.0) } else { 0.0 };
    }

    /// The game time the last edit took effect (T-1071): the renderer swaps in its trips then.
    pub fn edit_time(&self) -> f64 {
        self.edit_at
    }

    pub(super) fn edit_start(&mut self) -> f64 {
        let lead = std::mem::take(&mut self.lead);
        // never before an edit still to take effect, so they reach the screen in order
        let at = if lead > 0.0 && self.edit_at > self.now + lead { self.edit_at } else { self.now + lead };
        self.edit_at = at;
        at
    }

    pub(super) fn settle(&mut self, r: Result<(EditResult, f64), ChargeError>, before: &crate::track::service::Before) -> u32 {
        self.last_trains = 0.0;
        if r.is_ok() {
            self.w.svc.note_changes(&self.w.net, before, self.edit_at);
        }
        match r {
            Ok((r, trains)) => {
                self.cash -= r.charge + trains;
                let need = cars_needed(&self.w);
                self.fleet = [self.fleet[0].max(need[0]), self.fleet[1].max(need[1])];
                let out = self.done(Ok(r));
                self.last_trains = trains;
                out
            }
            Err(ChargeError::Issues(e)) => self.done(Err(e)),
            Err(ChargeError::Money { need, have }) => {
                self.issues.clear();
                self.money_short = Some((need, have));
                1
            }
        }
    }

    pub(super) fn refuse(&mut self, i: Issue) -> u32 {
        self.money_short = None;
        self.issues = vec![i];
        1
    }

    // ---- low-level edits (tests and tools; the drawing tool uses `add_route`)
    pub fn set_node(&mut self, id: u32, x: f64, y: f64, level: i8, flying: bool) -> u32 {
        if self.w.net.node_ok(id) && self.w.net.built_ports(id) > 0 {
            return self.refuse(Issue::NodeInUse { node: id });
        }
        self.run(Op::Node { id, data: Some(NodeData { x: quant(x), y: quant(y), level, flying }) })
    }
    /// Reshape a blueprint edge (T-064): new PIs (x, y, radius 0 = auto, level each), same ends
    /// and tracks. One undoable edit. Constructed track cannot be reshaped.
    pub fn set_edge_pis(&mut self, id: u32, pis: &[f64]) -> u32 {
        match self.reshaped(id, pis) {
            Ok(r) => self.run(r.op),
            Err(i) => self.refuse(i),
        }
    }

    /// What `set_edge_pis` would do, without doing it: the layout of `preview_route`. Its
    /// polyline includes the next pieces of the stroke where a joint moves with it (T-244).
    pub fn preview_edge(&mut self, id: u32, pis: &[f64]) -> Vec<f64> {
        self.preview_speeds.clear();
        self.preview_marks.clear();
        let r = match self.reshaped(id, pis) {
            Ok(r) => r,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let t = self.w.trial(r.op);
        let mut out = self.preview_out(t, &r.verts, &[], r.tracks, r.hsr);
        if r.before.is_empty() && r.after.is_empty() {
            return out;
        }
        let main = out.split_off(5 + 4 * r.verts.len());
        let main_speeds = std::mem::take(&mut self.preview_speeds);
        let joint_speed = main_speeds.first().copied().unwrap_or((V_TOP * 3.6) as f32);
        let mut speeds = vec![];
        if !r.before.is_empty() {
            let f = self.w.net.fit(&r.before);
            render_samples(&f.pieces, &f.vert, &mut out);
            speeds.extend(render_speeds(&f.pieces, &f.vert, if r.hsr { V_HSR } else { V_TOP }));
            speeds.push(joint_speed);
        }
        out.extend(main);
        speeds.extend(main_speeds);
        if !r.after.is_empty() {
            let f = self.w.net.fit(&r.after);
            speeds.push(joint_speed);
            render_samples(&f.pieces, &f.vert, &mut out);
            speeds.extend(render_speeds(&f.pieces, &f.vert, if r.hsr { V_HSR } else { V_TOP }));
        }
        self.preview_speeds = speeds;
        out
    }

    /// A blueprint edge with new PIs, its ends and tracks kept. A set radius the edit did not
    /// change that no longer fits its legs (a neighbour moved; radii frozen by a split) goes back
    /// to auto; one the edit set is kept, and refused if it does not fit.
    ///
    /// A joint between two pieces of one stroke (T-244: a plain node where this edge and one
    /// other blueprint edge meet, lying on the straight between the corners either side of it)
    /// moves with a corner next to it, to the same fraction of the way between the corners, so
    /// the corner drags as it did before the stroke was cut into pieces. The other piece keeps its
    /// corners; its leg to the joint turns with it.
    pub(super) fn reshaped(&self, id: u32, pis: &[f64]) -> Result<Reshape, Issue> {
        let Some(mut d) = self.editable(id) else { return Err(Issue::NoSuchEdge { edge: id }) };
        if d.built {
            return Err(Issue::NodeInUse { node: d.a });
        }
        if d.crossover {
            return Err(Issue::Crossover { edge: id });
        }
        let old = std::mem::replace(&mut d.pis, pis_from(pis));
        d.thru.clear();
        let same_count = old.len() == d.pis.len();
        let net = &self.w.net;
        let mut moved: Vec<(u32, f64, f64)> = vec![];
        let mut others: Vec<(u32, u32, EdgeData)> = vec![];
        if !old.is_empty() && !d.pis.is_empty() {
            let ends = [(d.a, old[0], d.pis[0]), (d.b, old[old.len() - 1], d.pis[d.pis.len() - 1])];
            for (node, near_old, near_new) in ends {
                if (near_old.x, near_old.y) == (near_new.x, near_new.y) || moved.iter().any(|m| m.0 == node) {
                    continue;
                }
                let Some((o, which, od, far, t)) = self.free_joint(node, id, near_old, false) else { continue };
                moved.push((node, quant(near_new.x + t * (far.x - near_new.x)), quant(near_new.y + t * (far.y - near_new.y))));
                others.push((o, which, od));
            }
        }
        relax_radii(net, &mut d, &moved, &|k, p| same_count && old[k].radius == p.radius);
        let verts = vertices_of(net, &d, &moved);
        let (tracks, hsr) = (d.tracks, d.hsr);
        if moved.is_empty() {
            return Ok(Reshape { op: Op::Edge { id, data: Some(d) }, verts, tracks, hsr, before: vec![], after: vec![] });
        }
        let mut ops = vec![];
        for &(n, x, y) in &moved {
            let mut nd = net.node_data(n).ok_or(Issue::NoSuchNode { node: n })?;
            nd.x = x;
            nd.y = y;
            ops.push(Op::Node { id: n, data: Some(nd) });
        }
        let (a, b) = (d.a, d.b);
        ops.push(Op::Edge { id, data: Some(d) });
        let (mut before, mut after) = (vec![], vec![]);
        for (o, which, mut od) in others {
            relax_radii(net, &mut od, &moved, &|_, _| true);
            let mut v = vertices_of(net, &od, &moved);
            let joint = net.end_node(o, which);
            // drawn in the stroke's order: up to the joint at a, on from the joint at b
            if (joint == a) == (which == 0) {
                v.reverse();
            }
            if joint == a {
                before = v;
            } else if joint == b {
                after = v;
            }
            ops.push(Op::Edge { id: o, data: Some(od) });
        }
        Ok(Reshape { op: Op::Batch(ops), verts, tracks, hsr, before, after })
    }

    /// The other piece at a joint of a stroke (T-244) that may follow a corner of `e` next to it
    /// (`near`, before the edit): a node with no station where only `e` and one other edge meet
    /// (neither constructed unless `built_ok`), on the straight between `near` and the other
    /// piece's corner next to it. Returns
    /// that edge, which of its ends is at the joint, its data, that corner (or its far node, if it
    /// has none) and how far along from `near` to it the joint lies (0..1).
    pub(super) fn free_joint(&self, node: u32, e: u32, near: Pi, built_ok: bool) -> Option<(u32, u32, EdgeData, Pi, f64)> {
        let net = &self.w.net;
        let i = node as usize;
        let p = &net.node_ports[i];
        if p.n != 2 || net.node_platform[i] > 0 || (!built_ok && net.built_ports(node) > 0) {
            return None;
        }
        let (o, which) = p.ends().iter().map(|&c| (c >> 1, c & 1)).find(|&(o, _)| o != e)?;
        if !net.edge_thru[o as usize].is_empty() || net.end_node(o, 1 - which) == node {
            return None;
        }
        let od = net.edge_data(o)?;
        let far = match od.pis.len() {
            0 => {
                let f = net.end_node(o, 1 - which) as usize;
                Pi::new(net.node_x[f], net.node_y[f], 0.0, net.node_level[f])
            }
            n => od.pis[if which == 0 { 0 } else { n - 1 }],
        };
        let (dx, dy) = (far.x - near.x, far.y - near.y);
        let l = dx.hypot(dy);
        if l < 1.0 {
            return None;
        }
        let (jx, jy) = (net.node_x[i] - near.x, net.node_y[i] - near.y);
        let t = (jx * dx + jy * dy) / (l * l);
        let off = (jx * dy - jy * dx).abs() / l;
        (t > 0.0 && t < 1.0 && off < 0.05).then_some((o, which, od, far, t))
    }

    /// Ops joining edge `e` with the pieces at its joints (T-244, `free_joint`) within `reach` of
    /// chainage `s`, so something placed at `s` has room; and where `s` is on the joined edge,
    /// which keeps `e`'s id and direction. The shape is unchanged (a radius the joined fit would
    /// change is set to the one it had), lines through it follow. Pieces join when both are
    /// constructed or both blueprint (switched alike), with the same track count (T-624) and type
    /// (high speed or not, T-817); a built piece's cost is derived from its shape, so the joined
    /// edge costs what the two did. None if no joint is that near.
    pub(super) fn unjoin_near(&self, e: u32, s: f64, reach: f64) -> Option<(Vec<Op>, f64)> {
        let net = &self.w.net;
        let d = net.edge_data(e)?;
        if !d.thru.is_empty() {
            return None;
        }
        let len = net.edge_len[e as usize];
        let own = vertices_of(net, &d, &[]);
        // T-418: joint after joint on each side, nearest first, until the reach is clear (a
        // platform over several short segments): (edge, its end at the joint nearer `e`, data)
        let mut sides: [Vec<(u32, u32, EdgeData)>; 2] = [vec![], vec![]];
        let mut seen = vec![e];
        for (k, side) in sides.iter_mut().enumerate() {
            let (mut node, mut gap, mut corner) = if k == 0 { (d.a, s, own[1]) } else { (d.b, len - s, own[own.len() - 2]) };
            let mut cur = e;
            while gap < reach {
                let Some((o, which, od, _, _)) = self.free_joint(node, cur, corner, true) else { break };
                if seen.contains(&o) || od.built != d.built || od.off != d.off || od.tracks != d.tracks || od.hsr != d.hsr || od.crossover || d.crossover {
                    break;
                }
                seen.push(o);
                let v = vertices_of(net, &od, &[]);
                node = net.end_node(o, 1 - which);
                corner = if which == 0 { v[v.len() - 2] } else { v[1] };
                gap += net.edge_len[o as usize];
                cur = o;
                side.push((o, which, od));
            }
        }
        if sides.iter().all(|m| m.is_empty()) {
            return None;
        }
        // the vertices of each part in the joined edge's direction, with each corner's radius
        let radii = |v: &[Pi]| net.fit(v).radius;
        let oriented = |od: &EdgeData, flip: bool| {
            let mut v = vertices_of(net, od, &[]);
            let mut r = radii(&v);
            if flip {
                v.reverse();
                r.reverse();
            }
            (v, r)
        };
        let mut parts: Vec<(Vec<Pi>, Vec<f64>)> = sides[0].iter().rev().map(|(_, which, od)| oriented(od, *which == 0)).collect();
        parts.push((own.clone(), radii(&own)));
        parts.extend(sides[1].iter().map(|(_, which, od)| oriented(od, *which == 1)));
        // A joint is dropped as a vertex unless its level differs from a neighbour's: a ramp is
        // centred between two vertices of different levels, so dropping it would move the ramp.
        let mut verts: Vec<(Pi, f64)> = vec![];
        let last = parts.len() - 1;
        let mut before: Option<Pi> = None;
        for (p, (v, r)) in parts.into_iter().enumerate() {
            if let Some(prev) = before {
                if !(prev.level == v[0].level && v[0].level == v[1].level) {
                    verts.push((v[0], 0.0));
                }
            }
            let keep = if p == last { v.len() } else { v.len() - 1 };
            before = v.len().checked_sub(2).map(|k| v[k]);
            verts.extend(v.into_iter().zip(r).take(keep).skip(1.min(p)));
        }
        let s = s + sides[0].iter().map(|m| net.edge_len[m.0 as usize]).sum::<f64>();
        let a = sides[0].last().map_or(d.a, |m| net.end_node(m.0, 1 - m.1));
        let b = sides[1].last().map_or(d.b, |m| net.end_node(m.0, 1 - m.1));
        let mut pis: Vec<Pi> = verts.iter().map(|v| v.0).collect();
        let f = net.fit(&pis);
        for j in 1..pis.len() - 1 {
            let want = verts[j].1;
            if want > 0.0 && (f.radius[j] - want).abs() > 0.01 {
                pis[j].radius = quant(want);
            }
        }
        let joined = EdgeData { a, b, tracks: d.tracks, pis: pis[1..pis.len() - 1].to_vec(), built: d.built, off: d.off, thru: vec![], crossover: false, hsr: d.hsr };
        // each part's direction that runs the joined edge's way
        let mut forward: Vec<(u32, u32)> = vec![(e, 0)];
        forward.extend(sides[0].iter().map(|&(o, which, _)| (o, if which == 1 { 0 } else { 1 })));
        forward.extend(sides[1].iter().map(|&(o, which, _)| (o, if which == 0 { 0 } else { 1 })));
        let mut ops = vec![Op::Edge { id: e, data: Some(joined) }];
        for &(o, which, _) in sides.iter().flatten() {
            ops.push(Op::Edge { id: o, data: None });
            ops.push(Op::Node { id: net.end_node(o, which), data: None });
        }
        for l in 0..net.line_count() as u32 {
            if !net.line_ok(l) || !net.line_path(l).iter().any(|&p| forward.iter().any(|f| f.0 == crate::track::net::path_edge(p))) {
                continue;
            }
            let mut ld = net.line_data(l)?;
            let mut path: Vec<u32> = vec![];
            for &p in &ld.path {
                let pe = crate::track::net::path_edge(p);
                match forward.iter().find(|f| f.0 == pe) {
                    Some(&(_, fwd)) => {
                        let step = e << 1 | (if (p & 1) == fwd { 0 } else { 1 });
                        if path.last() != Some(&step) {
                            path.push(step);
                        }
                    }
                    None => path.push(p),
                }
            }
            ld.path = path;
            ops.push(Op::Line { id: l, data: Some(ld) });
        }
        Some((ops, s))
    }

    /// An edge as an edit starts from it. One drawn while the track ran through the clicked
    /// points (T-079) has those points as its PIs, auto radius, instead of the PIs that were
    /// fitted through them: its first reshape cuts the corners at its clicks, like any track.
    pub(super) fn editable(&self, id: u32) -> Option<EdgeData> {
        let mut d = self.w.net.edge_data(id)?;
        if !d.thru.is_empty() {
            d.pis = std::mem::take(&mut d.thru);
        }
        Some(d)
    }

    /// An edge's PIs for editing (T-064): `[drawn through its points (T-079: 1, else 0), then per
    /// PI x, y, set radius (0 = auto), level, the radius the fit uses (0 = no curve), the curve's
    /// speed limit km/h, x, y of the middle of its curve]`. A T-079 edge lists the points it was
    /// drawn through, with the curves they will make once it is reshaped (`editable`).
    pub fn edge_pis(&self, id: u32) -> Vec<f64> {
        let Some(d) = self.editable(id) else { return vec![] };
        let net = &self.w.net;
        let f = net.fit(&vertices_of(net, &d, &[]));
        let mut out = vec![!net.edge_thru[id as usize].is_empty() as u8 as f64];
        for (i, p) in d.pis.iter().enumerate() {
            let rad = f.radius.get(i + 1).copied().unwrap_or(0.0);
            let (mx, my) = if rad > 0.0 && !f.pieces.is_empty() { let (x, y, _) = geom::pos_at(&f.pieces, f.vert_s[i + 1]); (x, y) } else { (p.x, p.y) };
            out.extend([p.x, p.y, p.radius, p.level as f64, rad, if rad > 0.0 { geom::curve_speed(rad) * 3.6 } else { V_TOP * 3.6 }, mx, my]);
        }
        out
    }


    /// Put a whole blueprint edge on one level (its ends keep theirs, with ramps), shape
    /// unchanged. One undoable edit.
    pub fn set_edge_level(&mut self, id: u32, level: i8) -> u32 {
        let Some(mut d) = self.w.net.edge_data(id) else { return self.refuse(Issue::NoSuchEdge { edge: id }) };
        if d.built {
            return self.refuse(Issue::NodeInUse { node: d.a });
        }
        if d.crossover {
            return self.refuse(Issue::Crossover { edge: id });
        }
        for p in d.pis.iter_mut().chain(d.thru.iter_mut()) {
            p.level = level;
        }
        self.run(Op::Edge { id, data: Some(d) })
    }

    // ---- dragging a blueprint node (T-093)

    /// Move a node (a track end, a junction, a station) to (x, y), with the blueprint track that
    /// meets there. Only where every edge there is blueprint and its station, if any, is not
    /// constructed. One undoable edit; lines keep their stops and paths.
    ///
    /// The edges' PIs stay where they are, except at a node with two or more edge ends, which
    /// keeps its heading (SPEC 6.2: every end leaves along it): there each edge's first PI, the
    /// one that holds the heading, moves with the node; an edge with no PI of its own to move
    /// (none at all, or only one that also holds the far node's heading) gets a lead-in PI on the
    /// heading, as drawing does. A track end's heading follows its edge. A set radius that no
    /// longer fits its legs goes back to auto.
    pub fn move_node(&mut self, node: u32, x: f64, y: f64) -> u32 {
        match self.moved(node, x, y) {
            Ok((op, _)) => self.run(op),
            Err(i) => self.refuse(i),
        }
    }

    /// What `move_node` would do, without doing it: `[ok (1/0), cost change US$M, edges, then
    /// per edge: vertex count, per vertex x, y (the middle of its curve), radius used, km/h, then
    /// sample count and the polyline as x, y, height triples]`. `issue_kinds()` holds the
    /// problems when not ok.
    pub fn preview_move_node(&mut self, node: u32, x: f64, y: f64) -> Vec<f64> {
        let (op, shapes) = match self.moved(node, x, y) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 3];
            }
        };
        self.preview_moved(op, shapes)
    }

    pub(super) fn preview_moved(&mut self, op: Op, shapes: Vec<Vec<Pi>>) -> Vec<f64> {
        let r = self.w.trial(op);
        self.preview_marks = trial_crossings(&r);
        let (ok, cost) = match &r {
            Ok(r) => (1.0, r.cost_after - r.cost_before),
            Err(_) => (0.0, 0.0),
        };
        self.done(r.map(|mut r| {
            r.dirty = Default::default();
            r
        }));
        self.last = None;
        let mut out = vec![ok, cost, shapes.len() as f64];
        for verts in &shapes {
            let f = self.w.net.fit(verts);
            out.push(verts.len() as f64);
            vertex_rows(&f, verts, &mut out, V_TOP);
            let mut pts = vec![];
            render_samples(&f.pieces, &f.vert, &mut pts);
            out.push((pts.len() / 3) as f64);
            out.extend(pts);
        }
        out
    }

    /// The edit moving `node` to (x, y) (`move_node`), and each changed edge's vertices.
    pub(super) fn moved(&self, node: u32, x: f64, y: f64) -> Result<(Op, Vec<Vec<Pi>>), Issue> {
        let Some(nd) = self.w.net.node_data(node) else { return Err(Issue::NoSuchNode { node }) };
        self.moved_by(&[node], quant(x) - nd.x, quant(y) - nd.y)
    }

    /// Move several blueprint nodes together by (dx, dy) (T-475), as one undoable edit: track
    /// between two of them keeps its shape, shifted whole; track from one of them to a node that
    /// stays reshapes as `move_node` does. Every node must be movable (`move_node`'s rule).
    pub fn move_nodes(&mut self, nodes: &[u32], dx: f64, dy: f64) -> u32 {
        match self.moved_by(nodes, dx, dy) {
            Ok((op, _)) => self.run(op),
            Err(i) => self.refuse(i),
        }
    }

    /// What `move_nodes` would do, laid out as `preview_move_node`'s answer.
    pub fn preview_move_nodes(&mut self, nodes: &[u32], dx: f64, dy: f64) -> Vec<f64> {
        let (op, shapes) = match self.moved_by(nodes, dx, dy) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 3];
            }
        };
        self.preview_moved(op, shapes)
    }

    /// The edit moving `nodes` by (dx, dy), and each changed edge's vertices.
    pub(super) fn moved_by(&self, nodes: &[u32], dx: f64, dy: f64) -> Result<(Op, Vec<Vec<Pi>>), Issue> {
        let net = &self.w.net;
        let mut ops = vec![];
        let mut moves: Vec<(u32, f64, f64)> = vec![];
        for &node in nodes {
            let Some(mut nd) = net.node_data(node) else { return Err(Issue::NoSuchNode { node }) };
            let i = node as usize;
            if net.built_ports(node) > 0 || (net.node_platform[i] > 0 && net.node_built[i]) {
                return Err(Issue::NodeInUse { node });
            }
            if let Some(&c) = net.node_ports[i].ends().iter().find(|&&c| net.edge_crossover[(c >> 1) as usize]) {
                return Err(Issue::Crossover { edge: c >> 1 });
            }
            if moves.iter().any(|m| m.0 == node) {
                continue;
            }
            nd.x = quant(nd.x + dx);
            nd.y = quant(nd.y + dy);
            moves.push((node, nd.x, nd.y));
            ops.push(Op::Node { id: node, data: Some(nd) });
        }
        let moving = |n: u32| moves.iter().any(|m| m.0 == n);
        let mut shapes = vec![];
        let mut done: Vec<u32> = vec![];
        for &(node, x, y) in &moves {
            let i = node as usize;
            let ports = net.node_ports[i];
            let h = net.node_heading[i];
            // A node where two or more edge ends meet keeps its heading; a track end's follows.
            let keep = ports.n >= 2;
            for (k, &c) in ports.ends().iter().enumerate() {
                let (e, which) = (c >> 1, c & 1);
                if done.contains(&e) {
                    continue;
                }
                done.push(e);
                let Some(mut d) = self.editable(e) else { continue };
                if moving(net.end_node(e, 1 - which)) {
                    // both ends move: the whole stretch shifts
                    for p in &mut d.pis {
                        p.x = quant(p.x + dx);
                        p.y = quant(p.y + dy);
                    }
                    d.thru.clear();
                    shapes.push(vertices_of(net, &d, &moves));
                    ops.push(Op::Edge { id: e, data: Some(d) });
                    continue;
                }
                let far = net.end_node(e, 1 - which);
                let fi = far as usize;
                let (fx, fy) = (net.node_x[fi], net.node_y[fi]);
                let far_keep = net.node_ports[fi].n >= 2;
                // PIs from the moved node outwards
                let mut v = std::mem::take(&mut d.pis);
                if which == 1 {
                    v.reverse();
                }
                let leave = if ports.side[k] == 1 { h } else { h + PI };
                if keep && (v.len() >= 2 || (v.len() == 1 && !far_keep)) {
                    // its first PI holds the heading: it moves with the node
                    v[0].x = quant(v[0].x + dx);
                    v[0].y = quant(v[0].y + dy);
                } else {
                    if v.is_empty() && far_keep {
                        let fp = &net.node_ports[fi];
                        let side = fp.find(e, 1 - which).map_or(1, |q| fp.side[q]);
                        let fdir = if side == 1 { net.node_heading[fi] } else { net.node_heading[fi] + PI };
                        if let Some(p) = lead_in_along(fx, fy, net.node_level[fi], fdir, (x, y), net.frame().scaled()) {
                            v.push(p);
                        }
                    }
                    if keep {
                        let toward = v.first().map_or((fx, fy), |p| (p.x, p.y));
                        if let Some(p) = lead_in_along(x, y, net.node_level[i], leave, toward, net.frame().scaled()) {
                            v.insert(0, p);
                        }
                    }
                }
                if which == 1 {
                    v.reverse();
                }
                d.pis = v;
                d.thru.clear();
                relax_radii(net, &mut d, &moves, &|_, _| true);
                shapes.push(vertices_of(net, &d, &moves));
                ops.push(Op::Edge { id: e, data: Some(d) });
            }
        }
        Ok((Op::Batch(ops), shapes))
    }

    /// What making a junction flying would cost, US$M (T-080): the flyover's price, paid at once
    /// on a constructed junction, added to the blueprint otherwise. 0 if it is flying already.
    pub fn flyover_cost(&mut self, node: u32) -> f64 {
        let Some(mut d) = self.w.net.node_data(node) else { return 0.0 };
        if d.flying {
            return 0.0;
        }
        d.flying = true;
        self.w.trial(Op::Node { id: node, data: Some(d) }).map_or(0.0, |r| r.cost_after - r.cost_before)
    }

    /// The preview layout shared by `preview_route` and `preview_edge`:
    /// [ok, cost, length, vertex count, cost multiplier (the track's price over its length at
    /// the base price: level and water averaged, T-085), per vertex x, y (the middle of its
    /// curve, or the vertex where it has none), radius used, km/h, then the polyline].
    ///
    /// `shape` (T-656) is the alignment actually built where it differs from `verts`, the one the
    /// player drew (a bridge over water): it gives the polyline, length and price; `verts` gives
    /// the rows.
    pub(super) fn preview_out(&mut self, r: Result<EditResult, Vec<Issue>>, verts: &[Pi], shape: &[Pi], tracks: u8, hsr: bool) -> Vec<f64> {
        let (f, track, _) = self.w.preview(if shape.is_empty() { verts } else { shape }, tracks, hsr);
        self.preview_marks = trial_crossings(&r);
        let (ok, cost) = match &r {
            Ok(r) => (1.0, r.cost_after - r.cost_before),
            Err(_) => (0.0, track),
        };
        let base = BASE_COST_PER_KM * f.len / 1000.0 * crate::track::cost::track_factor(tracks, hsr);
        let mult = if base > 0.0 { track / base } else { 0.0 };
        self.done(r.map(|mut r| {
            r.dirty = Default::default();
            r
        }));
        self.last = None;
        let mut out = vec![ok, cost, f.len, verts.len() as f64, mult];
        if shape.is_empty() {
            vertex_rows(&f, verts, &mut out, if hsr { V_HSR } else { V_TOP });
        } else {
            vertex_rows(&self.w.net.fit(verts), verts, &mut out, if hsr { V_HSR } else { V_TOP });
        }
        render_samples(&f.pieces, &f.vert, &mut out);
        self.preview_speeds = render_speeds(&f.pieces, &f.vert, if hsr { V_HSR } else { V_TOP });
        out
    }

    /// Curve-limited km/h per consecutive segment in the last route/edge preview polyline.
    /// Separate from its stable vertex/point layout; empty after an unrenderable refusal.
    pub fn preview_render_speeds(&self) -> Vec<f32> {
        self.preview_speeds.clone()
    }

}
