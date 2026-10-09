//! The clock worker's API (T-025; notes/T-040.md has the buffer formats, notes/T-025.md the rest).
//!
//! The worker owns one `TrackApi`. Edits go in as calls that each build one op; every edit returns
//! 0 (applied) or the number of problems (nothing changed; `issue_kinds()` says what). After an
//! edit, `dirty_*` say what to refresh. Buffers come back as typed arrays (copies the worker can
//! transfer to the main thread).
//!
//! Blueprint and construct (T-055, SPEC 6.4): everything the player draws is a blueprint, free
//! and undoable. `construct` pays for blueprint edges and stations out of `cash` and makes them
//! real; it, removing constructed track or stations, and changing a constructed junction or
//! platform are final (they clear the undo history). Money is US$M.

use super::cost::RunMask;
use super::geom::{self, GeomIssue, Pi};
use super::net::{EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::profile;
use super::save;
use super::world::{ChargeError, EditResult, Op, TrackWorld};
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TrackApi {
    w: TrackWorld,
    last: Option<EditResult>,
    issues: Vec<Issue>,
    /// Set when the last refusal was for money: (need, have).
    money_short: Option<(f64, f64)>,
    /// Cash, US$M (SPEC 6.4: $6B to start).
    cash: f64,
    /// Cars the network owns (T-028). Running lines need `cars_needed`; an edit that needs more
    /// buys them, spare cars stay for later.
    fleet: u32,
    /// What the last edit paid for trains, US$M.
    last_trains: f64,
    /// The last route the drawing tool asked about: (cost, length, issues), and its vertices.
    route_vertices: Vec<Pi>,
}

/// Cars the running lines need: each line's busiest schedule's trains times its cars (T-028).
fn cars_needed(w: &TrackWorld) -> u32 {
    w.svc.lines.iter().filter(|l| l.running).map(|l| l.trains.iter().copied().max().unwrap_or(0) * l.cars).sum()
}

fn pis_from(flat: &[f64]) -> Vec<Pi> {
    flat.chunks_exact(4).map(|c| Pi::new(c[0], c[1], c[2], c[3] as i8).quantised()).collect()
}

/// Where a drawn route starts or ends.
#[derive(Clone, Copy, Debug)]
enum End {
    /// A new node here, at this level.
    Free { x: f64, y: f64, level: i8 },
    /// An existing node (a track end, a junction, a station).
    Node(u32),
    /// A new node splitting this edge near (x, y).
    Edge { e: u32, x: f64, y: f64 },
}

fn end_from(c: &[f64]) -> End {
    match c[0] as i32 {
        1 => End::Node(c[1] as u32),
        2 => End::Edge { e: c[1] as u32, x: c[2], y: c[3] },
        _ => End::Free { x: quant(c[2]), y: quant(c[3]), level: c[4] as i8 },
    }
}

/// A resolved end: the node (new or existing), where it is, its level, and the directions a new
/// edge may leave it in (empty = any).
struct Resolved {
    node: u32,
    x: f64,
    y: f64,
    level: i8,
    dirs: Vec<f64>,
    /// The route turns it into a junction (it had two edge ends, or it splits track).
    new_junction: bool,
    /// Ops creating it (a split or a new node), applied before the edge.
    ops: Vec<Op>,
}

#[wasm_bindgen]
impl TrackApi {
    /// An empty network for a city: its origin and running side (right in the US).
    #[wasm_bindgen(constructor)]
    pub fn new(origin_lon: f64, origin_lat: f64, right_hand: bool) -> TrackApi {
        TrackApi {
            w: TrackWorld::new(Network::new(origin_lon, origin_lat, right_hand)),
            last: None,
            issues: vec![],
            money_short: None,
            cash: 6000.0,
            fleet: 0,
            last_trains: 0.0,
            route_vertices: vec![],
        }
    }

    /// The pack's water mask (notes/T-004.md): header `cell_m`, `origin_m`, `size`, and the
    /// `water_row` (u32) and `water_x` (u16) arrays.
    pub fn set_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16]) {
        let m = RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() };
        self.w.net.set_water_mask(Box::new(m));
        self.w.reload();
    }

    /// Replace the network with a saved one. False if the bytes are not a save.
    pub fn load(&mut self, bytes: &[u8]) -> bool {
        match save::decode(bytes) {
            Ok(mut net) => {
                std::mem::swap(&mut net.water_mask, &mut self.w.net.water_mask);
                self.w = TrackWorld::new(net);
                true
            }
            Err(_) => false,
        }
    }
    pub fn save(&self) -> Vec<u8> {
        save::encode(&self.w.net)
    }

    // ---- money
    pub fn cash(&self) -> f64 {
        self.cash
    }
    pub fn set_cash(&mut self, v: f64) {
        self.cash = v;
    }
    /// Cars owned (T-028).
    pub fn fleet(&self) -> u32 {
        self.fleet
    }
    pub fn set_fleet(&mut self, cars: u32) {
        self.fleet = cars;
    }
    /// Cars the running lines need now.
    pub fn cars_needed(&self) -> u32 {
        cars_needed(&self.w)
    }
    /// [price of a car US$M, running cost per car-km US$, economy scale] (params.rs). The real
    /// running cost per car-km; the game counts it, and fares, `ECONOMY` times (T-081).
    pub fn money_params(&self) -> Vec<f64> {
        vec![CAR_PRICE, RUN_COST_CAR_KM, ECONOMY]
    }
    /// Running cost of the running lines per hour at a demand level, US$M: trains an hour in each
    /// direction times the line's length and cars, at the game's economy scale (T-081).
    pub fn running_cost(&self, level: usize) -> f64 {
        let n = &self.w.net;
        let mut car_km = 0.0;
        for (l, s) in self.w.svc.lines.iter().enumerate() {
            if s.running {
                car_km += n.line_tph[l][level.min(LEVELS - 1)] as f64 * 2.0 * s.runs[0].len / 1000.0 * s.cars as f64;
            }
        }
        car_km * RUN_COST_CAR_KM * ECONOMY / 1e6
    }
    /// What the trains a planned line needs would cost once it runs, US$M, after the spare cars.
    pub fn line_train_cost(&self, line: u32) -> f64 {
        let Some(s) = self.w.svc.lines.get(line as usize).filter(|s| s.ok && !s.running) else { return 0.0 };
        let want = s.trains.iter().copied().max().unwrap_or(0) * s.cars;
        let spare = self.fleet.saturating_sub(cars_needed(&self.w));
        want.saturating_sub(spare) as f64 * CAR_PRICE
    }
    /// What the last edit paid for trains, US$M (also in `last_charge`).
    pub fn last_train_charge(&self) -> f64 {
        self.last_trains
    }
    /// Cost of everything drawn, blueprint included, US$M.
    pub fn total_cost(&self) -> f64 {
        self.w.net.total_cost()
    }
    /// Cost of what is constructed.
    pub fn built_cost(&self) -> f64 {
        self.w.net.built_cost()
    }
    /// What constructing the whole blueprint would cost now.
    pub fn blueprint_cost(&self) -> f64 {
        (self.w.net.total_cost() - self.w.net.built_cost()).max(0.0)
    }

    // ---- ids for new objects (claimed when the edit using them applies)
    pub fn new_node_id(&mut self) -> u32 {
        self.w.net.alloc_node()
    }
    pub fn new_edge_id(&mut self) -> u32 {
        self.w.net.alloc_edge()
    }
    pub fn new_line_id(&mut self) -> u32 {
        self.w.net.alloc_line()
    }

    fn done(&mut self, r: Result<EditResult, Vec<Issue>>) -> u32 {
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
    fn run(&mut self, op: Op) -> u32 {
        let fleet = self.fleet;
        let r = self.w.apply_paid(op, self.cash, false, &|w| trains_to_buy(w, fleet));
        self.settle(r)
    }

    /// A final, charged edit (see the module comment), trains included.
    fn run_charged(&mut self, op: Op) -> u32 {
        let fleet = self.fleet;
        let r = self.w.apply_paid(op, self.cash, true, &|w| trains_to_buy(w, fleet));
        self.settle(r)
    }

    fn settle(&mut self, r: Result<(EditResult, f64), ChargeError>) -> u32 {
        self.last_trains = 0.0;
        match r {
            Ok((r, trains)) => {
                self.cash -= r.charge + trains;
                self.fleet = self.fleet.max(cars_needed(&self.w));
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

    fn refuse(&mut self, i: Issue) -> u32 {
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
    pub fn remove_node(&mut self, id: u32) -> u32 {
        self.run(Op::Node { id, data: None })
    }
    /// Add a blueprint edge, or replace one. Constructed track cannot be reshaped (remove it and
    /// draw again). `pis`: x, y, radius (0 = auto), level per PI.
    pub fn set_edge(&mut self, id: u32, a: u32, b: u32, tracks: u8, pis: &[f64]) -> u32 {
        if self.w.net.edge_ok(id) && self.w.net.edge_built[id as usize] {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        self.run(Op::Edge { id, data: Some(EdgeData { a, b, tracks, pis: pis_from(pis), built: false, thru: vec![] }) })
    }
    /// Reshape a blueprint edge (T-064): new PIs (x, y, radius 0 = auto, level each), same ends
    /// and tracks. One undoable edit. Constructed track cannot be reshaped.
    pub fn set_edge_pis(&mut self, id: u32, pis: &[f64]) -> u32 {
        match self.reshaped(id, pis) {
            Ok(d) => self.run(Op::Edge { id, data: Some(d) }),
            Err(i) => self.refuse(i),
        }
    }

    /// What `set_edge_pis` would do, without doing it: the layout of `preview_route`.
    pub fn preview_edge(&mut self, id: u32, pis: &[f64]) -> Vec<f64> {
        let d = match self.reshaped(id, pis) {
            Ok(d) => d,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let verts = vertices_of(&self.w.net, &d, None);
        let tracks = d.tracks;
        let r = self.w.trial(Op::Edge { id, data: Some(d) });
        self.preview_out(r, &verts, tracks)
    }

    /// A blueprint edge with new PIs, its ends and tracks kept. A set radius the edit did not
    /// change that no longer fits its legs (a neighbour moved; radii frozen by a split) goes back
    /// to auto; one the edit set is kept, and refused if it does not fit.
    fn reshaped(&self, id: u32, pis: &[f64]) -> Result<EdgeData, Issue> {
        let Some(mut d) = self.editable(id) else { return Err(Issue::NoSuchEdge { edge: id }) };
        if d.built {
            return Err(Issue::NodeInUse { node: d.a });
        }
        let old = std::mem::replace(&mut d.pis, pis_from(pis));
        d.thru.clear();
        let same_count = old.len() == d.pis.len();
        relax_radii(&self.w.net, &mut d, None, &|k, p| same_count && old[k].radius == p.radius);
        Ok(d)
    }

    /// An edge as an edit starts from it. One drawn while the track ran through the clicked
    /// points (T-079) has those points as its PIs, auto radius, instead of the PIs that were
    /// fitted through them: its first reshape cuts the corners at its clicks, like any track.
    fn editable(&self, id: u32) -> Option<EdgeData> {
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
        let f = geom::fit(&vertices_of(net, &d, None));
        let mut out = vec![!net.edge_thru[id as usize].is_empty() as u8 as f64];
        for (i, p) in d.pis.iter().enumerate() {
            let rad = f.radius.get(i + 1).copied().unwrap_or(0.0);
            let (mx, my) = if rad > 0.0 && !f.pieces.is_empty() { let (x, y, _) = geom::pos_at(&f.pieces, f.vert_s[i + 1]); (x, y) } else { (p.x, p.y) };
            out.extend([p.x, p.y, p.radius, p.level as f64, rad, if rad > 0.0 { geom::curve_speed(rad) * 3.6 } else { V_TOP * 3.6 }, mx, my]);
        }
        out
    }

    pub fn split_edge(&mut self, edge: u32, s: f64, node: u32, new_edge: u32) -> u32 {
        self.run(Op::Split { edge, s, node, new_edge })
    }
    pub fn set_station(&mut self, node: u32, platform: u16, name: String) -> u32 {
        self.run(Op::Station { node, data: Some(StationData { platform, name, built: false }) })
    }

    /// Put a whole blueprint edge on one level (its ends keep theirs, with ramps), shape
    /// unchanged. One undoable edit.
    pub fn set_edge_level(&mut self, id: u32, level: i8) -> u32 {
        let Some(mut d) = self.w.net.edge_data(id) else { return self.refuse(Issue::NoSuchEdge { edge: id }) };
        if d.built {
            return self.refuse(Issue::NodeInUse { node: d.a });
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
        let r = self.w.trial(op);
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
            let f = geom::fit(verts);
            out.push(verts.len() as f64);
            vertex_rows(&f, verts, &mut out);
            let mut pts = vec![];
            render_samples(&f.pieces, &f.vert, &mut pts);
            out.push((pts.len() / 3) as f64);
            out.extend(pts);
        }
        out
    }

    /// The edit moving `node` to (x, y) (`move_node`), and each changed edge's vertices.
    fn moved(&self, node: u32, x: f64, y: f64) -> Result<(Op, Vec<Vec<Pi>>), Issue> {
        let net = &self.w.net;
        let Some(mut nd) = net.node_data(node) else { return Err(Issue::NoSuchNode { node }) };
        let i = node as usize;
        if net.built_ports(node) > 0 || (net.node_platform[i] > 0 && net.node_built[i]) {
            return Err(Issue::NodeInUse { node });
        }
        let (x, y) = (quant(x), quant(y));
        let (dx, dy) = (x - nd.x, y - nd.y);
        let ports = net.node_ports[i];
        let h = net.node_heading[i];
        // A node where two or more edge ends meet keeps its heading; a track end's follows.
        let keep = ports.n >= 2;
        nd.x = x;
        nd.y = y;
        let mut ops = vec![Op::Node { id: node, data: Some(nd.clone()) }];
        let mut shapes = vec![];
        for (k, &c) in ports.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let Some(mut d) = self.editable(e) else { continue };
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
                    if let Some(p) = lead_in_along(fx, fy, net.node_level[fi], fdir, (x, y)) {
                        v.push(p);
                    }
                }
                if keep {
                    let toward = v.first().map_or((fx, fy), |p| (p.x, p.y));
                    if let Some(p) = lead_in_along(x, y, nd.level, leave, toward) {
                        v.insert(0, p);
                    }
                }
            }
            if which == 1 {
                v.reverse();
            }
            d.pis = v;
            d.thru.clear();
            relax_radii(net, &mut d, Some((node, x, y)), &|_, _| true);
            shapes.push(vertices_of(net, &d, Some((node, x, y))));
            ops.push(Op::Edge { id: e, data: Some(d) });
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
    fn preview_out(&mut self, r: Result<EditResult, Vec<Issue>>, verts: &[Pi], tracks: u8) -> Vec<f64> {
        let (f, track, _) = self.w.preview(verts, tracks);
        let (ok, cost) = match &r {
            Ok(r) => (1.0, r.cost_after - r.cost_before),
            Err(_) => (0.0, track),
        };
        let base = BASE_COST_PER_KM * f.len / 1000.0 * if tracks == 1 { SINGLE_TRACK_COST } else { 1.0 };
        let mult = if base > 0.0 { track / base } else { 0.0 };
        self.done(r.map(|mut r| {
            r.dirty = Default::default();
            r
        }));
        self.last = None;
        let mut out = vec![ok, cost, f.len, verts.len() as f64, mult];
        vertex_rows(&f, verts, &mut out);
        render_samples(&f.pieces, &f.vert, &mut out);
        out
    }

    // ---- the drawing tool (T-023, T-092)

    /// `ends`: start then end, 5 numbers each: kind (0 free, 1 node, 2 edge), id, x, y, level
    /// (level used for a free end). `pis`: the player's PIs between, x, y, radius (0 = auto),
    /// level: the track cuts the corner at each with a circular curve (SPEC 6.1). A start or end on
    /// existing track gets a lead-in PI on that track's heading, so it leaves like a turnout.
    fn route_op(&mut self, ends: &[f64], pis: &[f64], tracks: u8, flying: bool) -> Result<(Op, Vec<Pi>, Vec<u32>, Vec<u32>), Issue> {
        let pis = pis_from(pis);
        let (a, b) = (end_from(&ends[0..5]), end_from(&ends[5..10]));
        let mut new_nodes = vec![];
        let mut new_edges = vec![];
        // Both ends on one edge: the second split must name the half it falls in.
        let first_split = if let End::Edge { e, x, y } = a { Some((e, self.w.net.project(e, x, y).0)) } else { None };
        let ra = self.resolve(a, None, &mut new_nodes, &mut new_edges)?;
        let rb = self.resolve(b, first_split, &mut new_nodes, &mut new_edges)?;
        if ra.node == rb.node {
            self.release(&new_nodes, &new_edges);
            return Err(Issue::EdgeLoop { edge: u32::MAX });
        }
        let mut verts: Vec<Pi> = vec![Pi::new(ra.x, ra.y, 0.0, ra.level)];
        let next_a = pis.first().map_or((rb.x, rb.y), |p| (p.x, p.y));
        let prev_b = pis.last().map_or((ra.x, ra.y), |p| (p.x, p.y));
        if let Some(p) = lead_in(&ra, next_a) {
            verts.push(p);
        }
        verts.extend_from_slice(&pis);
        if let Some(p) = lead_in(&rb, prev_b) {
            verts.push(p);
        }
        verts.push(Pi::new(rb.x, rb.y, 0.0, rb.level));
        let edge = self.w.net.alloc_edge();
        new_edges.push(edge);
        let mut ops = ra.ops.clone();
        ops.extend(rb.ops.clone());
        ops.push(Op::Edge { id: edge, data: Some(EdgeData { a: ra.node, b: rb.node, tracks, pis: verts[1..verts.len() - 1].to_vec(), built: false, thru: vec![] }) });
        // A junction this route makes (it had two edge ends, or it splits track) is flying if
        // the build setting says so; never one that already stands.
        for r in [&ra, &rb] {
            if flying && r.new_junction {
                ops.push(Op::Node { id: r.node, data: Some(NodeData { x: r.x, y: r.y, level: r.level, flying: true }) });
            }
        }
        Ok((Op::Batch(ops), verts, new_nodes, new_edges))
    }

    fn resolve(&mut self, end: End, other_split: Option<(u32, f64)>, nodes: &mut Vec<u32>, edges: &mut Vec<u32>) -> Result<Resolved, Issue> {
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

    fn release(&mut self, nodes: &[u32], edges: &[u32]) {
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
        let (op, verts, nn, ne) = match self.route_op(ends, pis, tracks, flying) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let r = self.w.trial(op);
        self.release(&nn, &ne);
        let out = self.preview_out(r, &verts, tracks);
        self.route_vertices = verts;
        out
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
        }
        r
    }

    /// A blueprint station: on an existing node (kind 1, `id`) or splitting an edge near (x, y)
    /// (kind 2, `id` = edge).
    pub fn add_station(&mut self, kind: u32, id: u32, x: f64, y: f64, platform: u16, name: String) -> u32 {
        let st = StationData { platform, name, built: false };
        if kind == 1 {
            if !self.w.net.node_ok(id) {
                return self.refuse(Issue::NoSuchNode { node: id });
            }
            if self.w.net.node_platform[id as usize] > 0 {
                return self.refuse(Issue::StationPorts { node: id });
            }
            return self.run(Op::Station { node: id, data: Some(st) });
        }
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        let s = self.w.net.project(id, x, y).0;
        let (node, new_edge) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
        let r = self.run(Op::Batch(vec![Op::Split { edge: id, s, node, new_edge }, Op::Station { node, data: Some(st) }]));
        if r != 0 {
            self.release(&[node], &[new_edge]);
        }
        r
    }

    /// Rename a station or change its platform. A longer platform on a constructed station is
    /// paid for at once (final).
    pub fn set_station_props(&mut self, node: u32, platform: u16, name: String) -> u32 {
        let Some(old) = self.w.net.station_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
        let op = Op::Station { node, data: Some(StationData { platform, name, built: old.built }) };
        if old.built && old.platform != platform {
            self.run_charged(op)
        } else {
            self.run(op)
        }
    }

    /// Remove a station (and its node if no track is left there). Constructed: final, no refund.
    pub fn remove_station(&mut self, node: u32) -> u32 {
        let Some(old) = self.w.net.station_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
        let mut ops = vec![Op::Station { node, data: None }];
        if self.w.net.node_ports[node as usize].n == 0 {
            ops.push(Op::Node { id: node, data: None });
        }
        if old.built {
            self.run_charged(Op::Batch(ops))
        } else {
            self.run(Op::Batch(ops))
        }
    }

    /// Remove an edge, and nodes left with no track and no station. Lines using it reroute or
    /// break. Constructed: final, no refund (SPEC 6.4).
    pub fn delete_edge(&mut self, id: u32) -> u32 {
        let net = &self.w.net;
        if !net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        let mut ops = vec![Op::DeleteEdge { id }];
        let (a, b) = (net.edge_a[id as usize], net.edge_b[id as usize]);
        for n in [a, b] {
            if net.node_ports[n as usize].n == 1 && net.node_platform[n as usize] == 0 {
                ops.push(Op::Node { id: n, data: None });
            }
        }
        if net.edge_built[id as usize] {
            self.run_charged(Op::Batch(ops))
        } else {
            self.run(Op::Batch(ops))
        }
    }

    /// Flat or flying junction. On a constructed junction the flyover is paid at once (final).
    pub fn set_flying(&mut self, node: u32, flying: bool) -> u32 {
        let Some(mut d) = self.w.net.node_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
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
            self.money_short = None;
            self.issues = vec![];
            return 1;
        }
        self.run_charged(Op::Batch(ops))
    }

    /// What constructing these would cost now, US$M (junctions and crossings they complete
    /// included).
    pub fn construct_cost(&mut self, edges: &[u32], stations: &[u32]) -> f64 {
        let ops = self.construct_ops(edges, stations);
        if ops.is_empty() {
            return 0.0;
        }
        self.w.trial(Op::Batch(ops)).map_or(0.0, |r| r.charge)
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

    fn construct_ops(&self, edges: &[u32], stations: &[u32]) -> Vec<Op> {
        let net = &self.w.net;
        let mut ops = vec![];
        for &e in edges {
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

    /// Construct the whole blueprint.
    pub fn construct_all(&mut self) -> u32 {
        let net = &self.w.net;
        let edges: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && !net.edge_built[e as usize]).collect();
        let stations: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && !net.node_built[n as usize]).collect();
        self.construct(&edges, &stations)
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

    // ---- lines
    /// Add or replace a line; its path is the fastest route through the stops.
    #[allow(clippy::too_many_arguments)]
    pub fn set_line(&mut self, id: u32, stops: &[u32], high: f32, medium: f32, low: f32, dwell_s: f32, turnaround_s: f32, cars: u8, colour: u32, name: String) -> u32 {
        if stops.len() < 2 {
            return self.refuse(Issue::LineStops { line: id });
        }
        let Some(path) = self.w.net.route(stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { name, colour, stops: stops.to_vec(), path, tph: [high, medium, low], dwell_s, turnaround_s, cars }) })
    }
    /// Change a line's stops, keeping everything else; the path is found again.
    pub fn set_stops(&mut self, id: u32, stops: &[u32]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        if stops.len() < 2 {
            return self.refuse(Issue::LineStops { line: id });
        }
        let Some(path) = self.w.net.route(stops) else { return self.refuse(Issue::LinePath { line: id }) };
        d.stops = stops.to_vec();
        d.path = path;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Rename or recolour a line; its path stays.
    pub fn set_line_look(&mut self, id: u32, name: String, colour: u32) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.name = name;
        d.colour = colour;
        self.run(Op::Line { id, data: Some(d) })
    }
    pub fn remove_line(&mut self, id: u32) -> u32 {
        self.run(Op::Line { id, data: None })
    }
    pub fn set_schedule(&mut self, id: u32, high: f32, medium: f32, low: f32) -> u32 {
        self.run(Op::Schedule { id, tph: [high, medium, low] })
    }

    // ---- undo
    pub fn can_undo(&self) -> bool {
        self.w.can_undo()
    }
    pub fn can_redo(&self) -> bool {
        self.w.can_redo()
    }
    pub fn undo(&mut self) -> u32 {
        self.step(false)
    }
    pub fn redo(&mut self) -> u32 {
        self.step(true)
    }
    fn step(&mut self, redo: bool) -> u32 {
        let fleet = self.fleet;
        match self.w.step_paid(redo, self.cash, &|w| trains_to_buy(w, fleet)) {
            Some(r) => self.settle(r),
            None => {
                self.money_short = None;
                self.issues.clear();
                1
            }
        }
    }

    /// The last refusal's problems, one per line, debug text.
    pub fn issues(&self) -> String {
        self.issues.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n")
    }
    /// The last refusal's problems as kinds the UI turns into plain words, one per line:
    /// the `Issue` variant (`GroundOverWater`, `Crossing`, `CrossingOverlap`, ...), or
    /// `Geom:<GeomIssue variant>`, or `Money:<need>:<have>` (US$M).
    pub fn issue_kinds(&self) -> String {
        if let Some((need, have)) = self.money_short {
            return format!("Money:{need:.1}:{have:.1}");
        }
        let mut v: Vec<String> = self.issues.iter().map(issue_kind).collect();
        v.dedup();
        v.join("\n")
    }

    // ---- what the last edit changed
    /// What the last edit paid, US$M: construction plus trains (`last_train_charge`).
    pub fn last_charge(&self) -> f64 {
        self.last.as_ref().map_or(0.0, |r| r.charge + self.last_trains)
    }
    /// Zoom-12 tiles to rebuild, as x, y pairs.
    pub fn dirty_tiles(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.tiles.iter().flat_map(|t| [t.0, t.1]).collect())
    }
    pub fn dirty_edges(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.edges.clone())
    }
    /// Lines whose profiles changed (re-upload their phase tables).
    pub fn dirty_lines(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.service.recomputed.clone())
    }
    /// Lines whose service changed for demand (stops, frequency, running, a time moving 5 s).
    pub fn dirty_demand_lines(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.service.demand.clone())
    }

    // ---- the network, for the main thread's snapshot
    /// Per live edge, 10 numbers: id, a, b, tracks, constructed (1/0), length m, cost US$M,
    /// lowest speed limit km/h, lowest and highest level.
    pub fn edges_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for e in 0..n.edge_count() as u32 {
            if !n.edge_ok(e) {
                continue;
            }
            let i = e as usize;
            let vmin = n.edge_pieces(e).iter().map(|p| p.v_limit()).fold(V_TOP, f64::min);
            let zs = n.edge_vert(e).iter().map(|v| v[1]);
            let (lo, hi) = zs.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), z| (a.min(z), b.max(z)));
            out.extend([e as f64, n.edge_a[i] as f64, n.edge_b[i] as f64, n.edge_tracks[i] as f64, n.edge_built[i] as u8 as f64, n.edge_len[i], n.edge_cost[i], vmin * 3.6, (lo / LEVEL_H).round(), (hi / LEVEL_H).round()]);
        }
        out
    }
    /// An edge's centre line for drawing: x, y, height (m) triples.
    pub fn edge_render(&self, e: u32) -> Vec<f64> {
        let mut out = vec![];
        if self.w.net.edge_ok(e) {
            render_samples(self.w.net.edge_pieces(e), self.w.net.edge_vert(e), &mut out);
        }
        out
    }
    /// An edge as a polyline (x, y pairs, local metres), `lateral` metres to the left.
    pub fn edge_polyline(&self, e: u32, lateral: f64) -> Vec<f32> {
        let mut pts = vec![];
        geom::sample(self.w.net.edge_pieces(e), 50.0, 0.05, lateral, &mut pts);
        pts.iter().flat_map(|p| [p[0] as f32, p[1] as f32]).collect()
    }
    /// Per live node, 10 numbers: id, x, y, level, edge ends, constructed edge ends, flying,
    /// platform length (0 = no station), station constructed, heading (rad).
    pub fn nodes_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for k in 0..n.node_count() as u32 {
            if !n.node_ok(k) {
                continue;
            }
            let i = k as usize;
            out.extend([k as f64, n.node_x[i], n.node_y[i], n.node_level[i] as f64, n.node_ports[i].n as f64, n.built_ports(k) as f64, n.node_flying[i] as u8 as f64, n.node_platform[i] as f64, n.node_built[i] as u8 as f64, n.node_heading[i]]);
        }
        out
    }
    pub fn node_name(&self, id: u32) -> String {
        self.w.net.node_name.get(id as usize).cloned().unwrap_or_default()
    }
    /// Per live line, 22 numbers: id, colour (0xRRGGBB), ok, running, broken, tph high, medium,
    /// low, cars, path length m, round trip s x3, trains needed x3, delay s x3, dwell s,
    /// turnaround s, cars that fit.
    pub fn lines_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for l in 0..n.line_count() as u32 {
            if !n.line_ok(l) {
                continue;
            }
            let i = l as usize;
            let svc = self.w.svc.lines.get(i);
            let ok = svc.map_or(false, |s| s.ok);
            let running = svc.map_or(false, |s| s.running);
            out.extend([l as f64, n.line_colour[i] as f64, ok as u8 as f64, running as u8 as f64, n.line_broken[i] as u8 as f64]);
            out.extend(n.line_tph[i].iter().map(|&t| t as f64));
            out.push(n.line_cars[i] as f64);
            match svc.filter(|s| s.ok) {
                Some(s) => {
                    out.push(s.runs[0].len);
                    out.extend(s.round_trip);
                    out.extend(s.trains.iter().map(|&t| t as f64));
                    out.extend(s.delay);
                    out.extend([n.line_dwell[i] as f64, n.line_turn[i] as f64, s.cars as f64]);
                }
                None => {
                    out.extend([0.0; 10]);
                    out.extend([n.line_dwell[i] as f64, n.line_turn[i] as f64, 0.0]);
                }
            }
        }
        out
    }
    pub fn line_name(&self, id: u32) -> String {
        self.w.net.line_name.get(id as usize).cloned().unwrap_or_default()
    }
    pub fn line_stops(&self, id: u32) -> Vec<u32> {
        if self.w.net.line_ok(id) {
            self.w.net.line_stops(id).to_vec()
        } else {
            vec![]
        }
    }
    /// The line's path, `edge << 1 | direction`, first stop to last.
    pub fn line_path(&self, id: u32) -> Vec<u32> {
        if self.w.net.line_ok(id) {
            self.w.net.line_path(id).to_vec()
        } else {
            vec![]
        }
    }
    /// Where each stop is along a run's path, m (the train's middle when standing).
    pub fn line_stop_s(&self, line: u32, run: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].stop_s.clone())
    }

    // ---- service
    /// Time of each stop from the start of the line (SPEC 6.3), seconds: arrival times.
    pub fn stop_times(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.cumulative(run, level).to_vec())
    }
    /// Departure time from each stop, seconds from the start of the trip.
    pub fn stop_departures(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].prof[level].dep.clone())
    }
    /// Per demand level (high, medium, low): round trip s, trains needed, delay s.
    pub fn line_summary(&self, line: u32) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        (0..LEVELS).flat_map(|k| [l.round_trip[k], l.trains[k] as f64, l.delay[k]]).collect()
    }
    /// All phase tables: 4 floats per phase (t, s, v, a). notes/T-040.md "Keyframes".
    pub fn phases(&self) -> Vec<f32> {
        let mut out = vec![];
        for l in &self.w.svc.lines {
            for r in 0..2 {
                for k in 0..LEVELS {
                    if l.ok {
                        out.extend(l.runs[r].prof[k].phases.iter().flat_map(|p| [p.t as f32, p.s as f32, p.v as f32, p.a as f32]));
                    }
                }
            }
        }
        out
    }
    /// Per profile `(line * 2 + run) * 3 + level`: first phase, phase count, trip seconds
    /// (run plus turnaround), path length m. Absent lines have count 0.
    pub fn profile_meta(&self) -> Vec<f32> {
        let mut out = vec![];
        let mut first = 0usize;
        for (li, l) in self.w.svc.lines.iter().enumerate() {
            for r in 0..2 {
                for k in 0..LEVELS {
                    if l.ok {
                        let n = l.runs[r].prof[k].phases.len();
                        out.extend([first as f32, n as f32, l.trip_len(&self.w.net, li as u32, r, k) as f32, l.runs[r].len as f32]);
                        first += n;
                    } else {
                        out.extend([first as f32, 0.0, 0.0, 0.0]);
                    }
                }
            }
        }
        out
    }
    /// Trips drawn in `[epoch, epoch + 3600)`: per trip, profile index and departure relative to
    /// `epoch` (may be negative: still running from before). Running lines only.
    pub fn trips(&self, epoch: f64) -> Vec<f32> {
        self.w
            .svc
            .trips(&self.w.net, epoch, epoch + 3600.0)
            .iter()
            .flat_map(|t| [((t.line * 2 + t.run as u32) * 3 + t.level as u32) as f32, (t.dep - epoch) as f32])
            .collect()
    }
    /// A run's path resampled every `step` m (x, y pairs), for the renderer's line texture.
    pub fn run_polyline(&self, line: u32, run: usize, step: f64) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let len = l.runs[run].len;
        let n = (len / step).ceil() as usize;
        (0..=n)
            .flat_map(|i| {
                let (x, y) = self.w.svc.position(&self.w.net, line, run, (i as f64 * step).min(len));
                [x, y]
            })
            .collect()
    }
    /// Where a line's train is at game time `t` on a trip that left at `dep`: x, y (checks).
    pub fn train_at(&self, line: u32, run: usize, level: usize, dep: f64, t: f64) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let (s, _) = profile::state_at(&l.runs[run].prof[level].phases, t - dep);
        let (x, y) = self.w.svc.position(&self.w.net, line, run, s);
        vec![x, y]
    }
    /// The junction inspector (T-031): a junction's edge ends and the moves running lines make
    /// through it. Flat: `[ports, then per port: edge, end (0 a, 1 b), dx, dy (unit vector from the
    /// node to the track 1 km out, or its far end), moves, then per move: in port, out port, trains an hour x3
    /// (high, medium, low), delay per train s x3, utilisation x3, line count, lines..., crossing
    /// count, indices of the moves it crosses...]`. Delay and utilisation are the conflict
    /// resource's (0 for a move that crosses nothing, or at a flying junction).
    pub fn junction_info(&self, node: u32) -> Vec<f64> {
        let n = &self.w.net;
        let svc = &self.w.svc;
        if !n.node_ok(node) {
            return vec![0.0, 0.0];
        }
        let (nx, ny) = (n.node_x[node as usize], n.node_y[node as usize]);
        let mut ports: Vec<(u32, u32)> = vec![];
        let mut out = vec![0.0];
        for e in 0..n.edge_count() as u32 {
            if !n.edge_ok(e) {
                continue;
            }
            for (end, at) in [(0u32, n.edge_a[e as usize]), (1u32, n.edge_b[e as usize])] {
                if at != node {
                    continue;
                }
                let len = n.edge_len[e as usize];
                let s = if end == 0 { len.min(1000.0) } else { (len - 1000.0).max(0.0) };
                let (x, y, _) = geom::pos_at(n.edge_pieces(e), s);
                let d = (x - nx).hypot(y - ny).max(1e-9);
                ports.push((e, end));
                out.extend([e as f64, end as f64, (x - nx) / d, (y - ny) / d]);
            }
        }
        out[0] = ports.len() as f64;
        // moves: (in port, out port) -> line directions through it
        let mut moves: Vec<((usize, usize), Vec<(u32, u8)>)> = vec![];
        for (l, ls) in svc.lines.iter().enumerate() {
            if !ls.running {
                continue;
            }
            for (r, run) in ls.runs.iter().enumerate() {
                for k in 1..run.nodes.len().saturating_sub(1) {
                    if run.nodes[k] != node {
                        continue;
                    }
                    let (p, q) = (run.segs[k - 1], run.segs[k]);
                    let pin = (super::net::path_edge(p), if super::net::path_dir(p) == 0 { 1 } else { 0 });
                    let pout = (super::net::path_edge(q), if super::net::path_dir(q) == 0 { 0 } else { 1 });
                    let (Some(i), Some(o)) = (ports.iter().position(|&x| x == pin), ports.iter().position(|&x| x == pout)) else { continue };
                    match moves.iter_mut().find(|m| m.0 == (i, o)) {
                        Some(m) => m.1.push((l as u32, r as u8)),
                        None => moves.push(((i, o), vec![(l as u32, r as u8)])),
                    }
                }
            }
        }
        // each move's conflict resource: the junction resource it owns
        let cap = &svc.cap;
        let res_of = |m: &Vec<(u32, u8)>| {
            (0..cap.res.len()).find(|&r| {
                cap.res[r].kind == super::capacity::ResKind::Junction && cap.res[r].node == node && cap.users_of(r).iter().any(|u| u.owner && m.contains(&(u.line, u.run)))
            })
        };
        out.push(moves.len() as f64);
        for (k, ((i, o), users)) in moves.iter().enumerate() {
            let mut tph = [0.0f64; 3];
            let mut lines: Vec<u32> = vec![];
            for &(l, _) in users {
                for (lev, t) in tph.iter_mut().enumerate() {
                    *t += n.line_tph[l as usize][lev] as f64;
                }
                if !lines.contains(&l) {
                    lines.push(l);
                }
            }
            let r = res_of(users);
            out.extend([*i as f64, *o as f64]);
            out.extend(tph);
            out.extend(r.map_or([0.0; 3], |r| cap.delay[r]));
            out.extend(r.map_or([0.0; 3], |r| cap.rho[r]));
            out.push(lines.len() as f64);
            out.extend(lines.iter().map(|&l| l as f64));
            let crossing: Vec<usize> = match r {
                Some(r) => (0..moves.len())
                    .filter(|&j| j != k && cap.users_of(r).iter().any(|u| !u.owner && moves[j].1.contains(&(u.line, u.run))))
                    .collect(),
                None => vec![],
            };
            out.push(crossing.len() as f64);
            out.extend(crossing.iter().map(|&j| j as f64));
        }
        out
    }

    /// Where a line loses time to capacity at a demand level (T-031): per resource it waits at,
    /// largest first, `[kind (ResKind as u8), node (or -1), x, y, delay per round trip s]`; delay
    /// counts each of its directions that waits there.
    pub fn line_delays(&self, line: u32, level: usize) -> Vec<f64> {
        let cap = &self.w.svc.cap;
        let lev = level.min(LEVELS - 1);
        let mut v: Vec<(usize, f64)> = vec![];
        for r in 0..cap.res.len() {
            let d = cap.delay[r][lev];
            if d <= 0.0 {
                continue;
            }
            let k = cap.users_of(r).iter().filter(|u| u.owner && u.line == line).count();
            if k > 0 {
                v.push((r, d * k as f64));
            }
        }
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.iter()
            .flat_map(|&(r, d)| {
                let x = &cap.res[r];
                [x.kind as u8 as f64, if x.node == u32::MAX { -1.0 } else { x.node as f64 }, x.x, x.y, d]
            })
            .collect()
    }

    /// A line's dwell over a round trip at a level, s: the dwell at every intermediate stop both
    /// ways (capacity holds at platforms are delay, not dwell), and the two turnarounds.
    pub fn line_dwell_split(&self, line: u32) -> Vec<f64> {
        let n = &self.w.net;
        let Some(ls) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![0.0, 0.0] };
        let stops = ls.runs[0].stop_s.len().saturating_sub(2) as f64;
        vec![2.0 * stops * n.line_dwell[line as usize] as f64, 2.0 * n.line_turn[line as usize] as f64]
    }

    /// Capacity markers for a level (SPEC 6.2 shows 75% and up): x, y, utilisation, delay s, kind.
    pub fn markers(&self, level: usize) -> Vec<f32> {
        self.w
            .busy(level)
            .iter()
            .flat_map(|&(r, kind, rho, d)| {
                let res = &self.w.svc.cap.res[r];
                [res.x as f32, res.y as f32, rho as f32, d as f32, kind as u8 as f32]
            })
            .collect()
    }
}

/// What buying the cars the running lines need beyond `fleet` costs, US$M.
fn trains_to_buy(w: &TrackWorld, fleet: u32) -> f64 {
    cars_needed(w).saturating_sub(fleet) as f64 * CAR_PRICE
}

/// An edge's vertices: its a node, its PIs, its b node; `moved` puts one node somewhere else.
fn vertices_of(net: &Network, d: &EdgeData, moved: Option<(u32, f64, f64)>) -> Vec<Pi> {
    let at = |n: u32| -> Pi {
        let i = n as usize;
        match moved {
            Some((m, x, y)) if m == n => Pi::new(x, y, 0.0, net.node_level[i]),
            _ => Pi::new(net.node_x[i], net.node_y[i], 0.0, net.node_level[i]),
        }
    };
    let mut v = vec![at(d.a)];
    v.extend_from_slice(&d.pis);
    v.push(at(d.b));
    v
}

/// Set radii that no longer fit their legs go back to auto, where `may(pi index, pi)` allows it.
fn relax_radii(net: &Network, d: &mut EdgeData, moved: Option<(u32, f64, f64)>, may: &dyn Fn(usize, &Pi) -> bool) {
    let verts = vertices_of(net, d, moved);
    for issue in geom::fit(&verts).issues {
        if let GeomIssue::ArcDoesNotFit(j) = issue {
            if j > 0 && j < verts.len() - 1 && may(j - 1, &d.pis[j - 1]) {
                d.pis[j - 1].radius = 0.0;
            }
        }
    }
}

/// Per vertex of a fitted alignment: x, y (the middle of its curve, or the vertex where it has
/// none), radius used (0 = no curve), speed limit km/h.
fn vertex_rows(f: &geom::Fit, verts: &[Pi], out: &mut Vec<f64>) {
    for (i, v) in verts.iter().enumerate() {
        let rad = f.radius.get(i).copied().unwrap_or(0.0);
        let (x, y) = if rad > 0.0 && !f.pieces.is_empty() { let (x, y, _) = geom::pos_at(&f.pieces, f.vert_s[i]); (x, y) } else { (v.x, v.y) };
        out.extend([x, y, rad, if rad > 0.0 { geom::curve_speed(rad) * 3.6 } else { V_TOP * 3.6 }]);
    }
}

/// A PI along a node's heading, so a new edge leaves the node the way its track runs (like a
/// turnout) and its first arc starts right at the node: on the main line's heading, a straight
/// would run on top of the main line (an overlap). `toward` is the next vertex of the route; of
/// the directions the node allows, the one nearer to it. None where the node has no track yet
/// or the route already leaves along the heading.
fn lead_in(r: &Resolved, toward: (f64, f64)) -> Option<Pi> {
    if r.dirs.is_empty() {
        return None;
    }
    let want = (toward.1 - r.y).atan2(toward.0 - r.x);
    let dir = r.dirs.iter().copied().min_by(|a, b| geom::wrap(a - want).abs().total_cmp(&geom::wrap(b - want).abs())).unwrap();
    lead_in_along(r.x, r.y, r.level, dir, toward)
}

/// A lead-in PI leaving (x, y) on heading `dir` towards `toward` (`lead_in`), or None when the
/// way to `toward` is already within the node's tolerance of the heading.
fn lead_in_along(x: f64, y: f64, level: i8, dir: f64, toward: (f64, f64)) -> Option<Pi> {
    let (dx, dy) = (toward.0 - x, toward.1 - y);
    let dist = dx.hypot(dy);
    if dist < 1.0 {
        return None;
    }
    let delta = geom::wrap(dy.atan2(dx) - dir).abs();
    if delta < 0.002 {
        return None;
    }
    // The arc at the lead-in uses the whole first leg (auto radius, notes/T-008.md 1.2) as long
    // as its radius stays under the cap; 0.3 of the way keeps it inside half the next leg.
    let d = (R_CAP * (delta / 2.0).tan()).min(0.3 * dist).max(0.5);
    Some(Pi::new(x + dir.cos() * d, y + dir.sin() * d, 0.0, level).quantised())
}

/// Points along an alignment for drawing: x, y, height triples; arcs within 5 cm, straights
/// every 100 m (so ramps show), ramp ends included.
fn render_samples(pieces: &[geom::Piece], vert: &[[f64; 2]], out: &mut Vec<f64>) {
    let mut ss: Vec<f64> = vec![];
    for p in pieces {
        let step = if p.k == 0.0 { 100.0 } else { 50.0f64.min((8.0 * p.radius() * 0.05).sqrt()) };
        let n = (p.len / step).ceil().max(1.0) as usize;
        for j in 0..n {
            ss.push(p.s0 + p.len * j as f64 / n as f64);
        }
    }
    if let Some(p) = pieces.last() {
        ss.push(p.s0 + p.len);
    }
    for v in vert {
        ss.push(v[0]);
    }
    ss.sort_by(|a, b| a.total_cmp(b));
    ss.dedup_by(|a, b| (*a - *b).abs() < 0.5);
    for s in ss {
        let (x, y, _) = geom::pos_at(pieces, s);
        out.extend([x, y, geom::height_at(vert, s)]);
    }
}

fn issue_kind(i: &Issue) -> String {
    match i {
        Issue::Geom { issue, .. } => format!(
            "Geom:{}",
            match issue {
                GeomIssue::ZeroLeg(_) => "ZeroLeg",
                GeomIssue::Reversal(_) => "Reversal",
                GeomIssue::RadiusTooSmall(_) => "RadiusTooSmall",
                GeomIssue::ArcDoesNotFit(_) => "ArcDoesNotFit",
                GeomIssue::RampTooShort(_) => "RampTooShort",
                GeomIssue::LevelRange(_) => "LevelRange",
            }
        ),
        Issue::Crossing { overlap: true, .. } => "CrossingOverlap".into(),
        other => {
            let d = format!("{other:?}");
            d.split(|c: char| c == ' ' || c == '{' || c == '(').next().unwrap_or("").to_string()
        }
    }
}

/// The T-040 benchmark in WASM (feature `track-bench`): same report as the native binary.
#[cfg(feature = "track-bench")]
#[wasm_bindgen]
pub fn track_bench(km: f64, stations: u32, lines: u32, reps: u32) -> String {
    super::bench::run(km, stations as usize, lines as usize, reps as usize).0
}

/// The benchmark network's save bytes (to check the compressed size).
#[cfg(feature = "track-bench")]
#[wasm_bindgen]
pub fn track_bench_save(km: f64, stations: u32, lines: u32) -> Vec<u8> {
    let w = TrackWorld::new(super::bench::synth(km, stations as usize, lines as usize, 7));
    save::encode(&w.net)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api() -> TrackApi {
        TrackApi::new(-73.985, 40.758, true)
    }
    fn free(x: f64, y: f64) -> [f64; 5] {
        [0.0, 0.0, x, y, 0.0]
    }

    #[test]
    fn blueprint_construct_run() {
        let mut a = api();
        // A straight 3 km route, then a station at each end and one in the middle.
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
        let pv = a.preview_route(&ends, &[], 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert!((pv[1] - 3.0 * 27.0).abs() < 0.5, "cost {}", pv[1]);
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.built_cost(), 0.0);
        assert!((a.blueprint_cost() - 81.0).abs() < 0.5);
        let nodes = a.nodes_info();
        let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
        let e = a.edges_info()[0] as u32;
        assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
        assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
        assert_eq!(a.add_station(2, e, 1500.0, 30.0, 200, "Middle".into()), 0, "{}", a.issues());
        let mid = a.nodes_info().chunks(10).find(|c| (c[1] - 1500.0).abs() < 1.0).unwrap()[0] as u32;
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[n0, mid, n1], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0xd7263d, "Line 1".into()), 0, "{}", a.issues());
        // Planned over blueprint: times shown, no trips.
        let info = a.lines_info();
        assert_eq!((info[2], info[3]), (1.0, 0.0));
        assert!(a.stop_times(id, 0, 0)[2] > 100.0);
        assert!(a.trips(8.0 * 3600.0).is_empty());
        assert!(a.can_undo());
        // Not enough money.
        a.set_cash(10.0);
        assert_eq!(a.construct_line(id), 1);
        assert!(a.issue_kinds().starts_with("Money:"));
        assert!(a.can_undo(), "a refused construct keeps the history");
        a.set_cash(6000.0);
        let trains = a.line_train_cost(id);
        assert!(trains > 0.0);
        assert_eq!(a.construct_line(id), 0, "{}", a.issues());
        // Construction plus the trains its busiest schedule needs (T-028).
        assert_eq!(a.fleet(), a.cars_needed());
        assert!((a.fleet() as f64 * CAR_PRICE - trains).abs() < 1e-9);
        assert!((6000.0 - a.cash() - (81.0 + 30.0 + trains)).abs() < 0.5, "paid {}", 6000.0 - a.cash());
        assert!((a.last_charge() - (81.0 + 30.0 + trains)).abs() < 0.5);
        assert!(!a.can_undo(), "construct is final");
        // More trains an hour buys more trains; short of money it is refused and nothing changes.
        let (fleet, cash) = (a.fleet(), a.cash());
        a.set_cash(1.0);
        assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 1);
        assert!(a.issue_kinds().starts_with("Money:"), "{}", a.issue_kinds());
        assert_eq!((a.fleet(), a.lines_info()[5]), (fleet, 12.0));
        a.set_cash(cash);
        assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 0, "{}", a.issues());
        assert!(a.fleet() > fleet && a.last_train_charge() > 0.0);
        // Fewer trains keeps the cars; going back up again is free, and so is undo / redo.
        let (fleet, cash) = (a.fleet(), a.cash());
        assert_eq!(a.set_schedule(id, 12.0, 8.0, 4.0), 0);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.redo(), 0);
        assert_eq!(a.set_schedule(id, 30.0, 8.0, 4.0), 0);
        assert_eq!((a.fleet(), a.cash()), (fleet, cash));
        // Running cost: trains an hour both ways times length and cars.
        let len_km = a.lines_info()[9] / 1000.0;
        let cars = a.lines_info()[21];
        assert!((a.running_cost(0) - 30.0 * 2.0 * len_km * cars * RUN_COST_CAR_KM * ECONOMY / 1e6).abs() < 1e-9);
        // In debt, edits that cost nothing still go through.
        a.set_cash(-50.0);
        assert_eq!(a.set_schedule(id, 20.0, 8.0, 4.0), 0, "{}", a.issues());
        assert_eq!(a.blueprint_cost(), 0.0);
        let info = a.lines_info();
        assert_eq!(info[3], 1.0, "running");
        assert!(!a.trips(8.0 * 3600.0).is_empty());
        // Removing constructed track refunds nothing and breaks the line.
        let cash = a.cash();
        let e0 = a.edges_info()[0] as u32;
        assert_eq!(a.delete_edge(e0), 0, "{}", a.issues());
        assert_eq!(a.cash(), cash);
        assert!(a.trips(8.0 * 3600.0).is_empty());
    }

    #[test]
    fn junction_inspector_and_delay_split() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let e = a.edges_info()[0];
        let ends = [2.0, e, 3000.0, 5.0, 0.0, 0.0, 0.0, 5000.0, 2500.0, 0.0];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
        let (w, east, branch) = (node_at(&a, 0.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 5000.0, 2500.0));
        let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
        for s in [w, east, branch] {
            assert_eq!(a.add_station(1, s, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let l1 = a.new_line_id();
        assert_eq!(a.set_line(l1, &[w, east], 15.0, 8.0, 4.0, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
        let l2 = a.new_line_id();
        assert_eq!(a.set_line(l2, &[w, branch], 18.0, 8.0, 4.0, 30.0, 180.0, 0, 0x0000ff, "Branch".into()), 0, "{}", a.issues());
        let flyover = a.flyover_cost(j);
        assert!(flyover > 0.0);
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        assert!((a.flyover_cost(j) - flyover).abs() < 1e-9, "the same price built or not");
        let v = a.junction_info(j);
        let np = v[0] as usize;
        assert_eq!(np, 3);
        let mut i = 1 + np * 4;
        let nm = v[i] as usize;
        i += 1;
        assert_eq!(nm, 4, "each line both ways");
        let mut crossing = 0;
        let mut delayed = 0;
        for _ in 0..nm {
            let tph = v[i + 2];
            assert!(tph == 15.0 || tph == 18.0);
            let delay = v[i + 5];
            let nl = v[i + 11] as usize;
            let nc = v[i + 12 + nl] as usize;
            crossing += nc;
            if delay > 0.0 {
                delayed += 1;
            }
            i += 13 + nl + nc;
        }
        assert_eq!(i, v.len());
        // one crossing pair (SPEC 6.2: a double-track Y has one), seen from both moves
        assert_eq!(crossing, 2);
        assert!(delayed >= 1);
        // the round trip splits into running, dwell, turnaround and delay
        let s = a.line_summary(l1);
        let dd = a.line_dwell_split(l1);
        let delay = s[2];
        assert!(delay > 0.0 || a.line_summary(l2)[2] > 0.0);
        assert!(s[0] - dd[0] - dd[1] - delay > 0.0);
        assert!((dd[1] - 360.0).abs() < 1e-9);
        let worst = a.line_delays(l2, 0);
        assert!(worst.is_empty() || worst[4] > 0.0);
    }

    #[test]
    fn reshape_blueprint_track() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
        // Drawn with one PI (T-092): the track cuts the corner there, auto radius.
        let one_pi = [2000.0, 500.0, 0.0, 0.0];
        assert_eq!(a.add_route(&ends, &one_pi, 2, false), 0, "{}", a.issues());
        let e = a.edges_info()[0] as u32;
        let p = a.edge_pis(e);
        assert_eq!(p.len(), 9);
        assert_eq!(&p[..5], &[0.0, 2000.0, 500.0, 0.0, 0.0], "not a T-079 edge; the PI as clicked, auto radius");
        assert!(p[5] > 100.0, "auto radius used {}", p[5]);
        assert!(a.w.net.project(e, 2000.0, 500.0).1 > 1.0, "the curve cuts the corner");
        let len0 = a.edges_info()[5];
        // drag the PI further out: preview, then commit; longer, undoable
        let pv = a.preview_edge(e, &[2000.0, 1000.0, 0.0, 0.0]);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(a.edges_info()[5], len0, "a preview changes nothing");
        assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 0.0, 0.0]), 0, "{}", a.issues());
        assert!(a.edges_info()[5] > len0);
        // a set radius, then too small to be built
        assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 400.0, 0.0]), 0, "{}", a.issues());
        assert!((a.edge_pis(e)[5] - 400.0).abs() < 1e-6);
        assert_ne!(a.set_edge_pis(e, &[2000.0, 1000.0, 50.0, 0.0]), 0);
        assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
        // the whole stretch one level down, shape kept
        assert_eq!(a.set_edge_level(e, -1), 0, "{}", a.issues());
        assert_eq!(a.edges_info()[8], -1.0);
        assert!((a.edge_pis(e)[5] - 400.0).abs() < 1e-6);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.edges_info()[5], len0);
        // a set radius the edit leaves alone goes back to auto when its corner moves where it no
        // longer fits (radii a split froze); one the edit sets is refused instead
        assert_eq!(a.set_edge_pis(e, &[2000.0, 1000.0, 1500.0, 0.0]), 0, "{}", a.issues());
        assert_eq!(a.set_edge_pis(e, &[300.0, 1500.0, 1500.0, 0.0]), 0, "{}", a.issues());
        assert_eq!(a.edge_pis(e)[3], 0.0, "back to auto");
        assert_ne!(a.set_edge_pis(e, &[300.0, 1500.0, 1700.0, 0.0]), 0);
        assert!(a.issue_kinds().contains("ArcDoesNotFit"), "{}", a.issue_kinds());
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.edges_info()[5], len0);
        // the flyover's price comes from a trial, nothing changes
        assert_eq!(a.flyover_cost(a.nodes_info()[0] as u32), 0.0, "an end is no junction; nothing to fly over");
        // a save round trip keeps the PIs, and a TWT2 save still loads
        let mut b = api();
        assert!(b.load(&a.save()));
        assert_eq!(b.edges_info(), a.edges_info());
        assert_eq!(b.edge_pis(0), a.edge_pis(e));
        let mut old = api();
        let mut bytes = old.save();
        bytes[..4].copy_from_slice(b"TWT2");
        assert!(old.load(&bytes));
        // constructed track stays as it is
        assert_eq!(a.construct_all(), 0);
        assert_ne!(a.set_edge_pis(e, &[2000.0, 800.0, 0.0, 0.0]), 0);
        assert!(a.issue_kinds().contains("NodeInUse"));
    }

    /// T-092: a click leaves the track drawn before it alone. The clicks so far with the cursor on
    /// the next point, then that point clicked and the cursor moved on: nothing changes up to
    /// where the curve at the previous click starts, that curve keeps its corner and can only get
    /// tighter (when the leg to the cursor was what limited it: it shares that leg with the new
    /// click's curve now), and the new click becomes a curve.
    #[test]
    fn a_click_keeps_the_track_drawn_before() {
        let mut a = api();
        // start on an existing track end, so the lead-in is in it too
        let ends: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
        let clicks = [(1500.0, 300.0), (2600.0, -400.0), (2850.0, -250.0), (4800.0, 1200.0), (5200.0, 2600.0), (7000.0, 2500.0), (7600.0, 1400.0)];
        let fit_of = |a: &mut TrackApi, k: usize, cursor: (f64, f64)| -> (geom::Fit, Vec<Pi>) {
            let pis: Vec<f64> = clicks[..k].iter().flat_map(|&(x, y)| [x, y, 0.0, 0.0]).collect();
            let ends = [1.0, start, 0.0, 0.0, 0.0, 0.0, 0.0, cursor.0, cursor.1, 0.0];
            let pv = a.preview_route(&ends, &pis, 2, false);
            assert_eq!(pv[0], 1.0, "{} clicks: {}", k, a.issues());
            let v = a.route_vertices.clone();
            (geom::fit(&v), v)
        };
        // where the curve at vertex i starts and ends (its middle where there is none)
        let arc = |f: &geom::Fit, i: usize| -> (f64, f64) {
            match f.pieces.iter().zip(&f.piece_at_vertex).find(|&(p, &v)| v as usize == i && p.k != 0.0) {
                Some((p, _)) => (p.s0, p.s0 + p.len),
                None => (f.vert_s[i], f.vert_s[i]),
            }
        };
        let (mut tighter, mut same) = (0, 0);
        for k in 1..clicks.len() - 1 {
            let (before, vb) = fit_of(&mut a, k, clicks[k]);
            let (after, va) = fit_of(&mut a, k + 1, clicks[k + 1]);
            let prev = vb.len() - 2; // the previous click's vertex in both
            assert_eq!(&vb[..=prev], &va[..=prev], "the vertices up to the previous click, lead-in included");
            // up to where the previous click's curve starts, the very same track
            let upto = arc(&before, prev).0.min(arc(&after, prev).0);
            let mut s = 0.0;
            while s <= upto {
                let (p, q) = (geom::pos_at(&before.pieces, s), geom::pos_at(&after.pieces, s));
                assert!((p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9 && (p.2 - q.2).abs() < 1e-12, "click {k}: moved at {s} m");
                s += 1.0;
            }
            for i in 1..prev {
                assert_eq!(before.radius[i], after.radius[i], "click {k}: an earlier curve changed");
            }
            // the previous click's curve: same corner, never wider
            let (rb, ra) = (before.radius[prev], after.radius[prev]);
            assert!(ra <= rb + 1e-9, "click {k}: the previous curve got wider, {rb} to {ra}");
            let leg = (vb[prev + 1].x - vb[prev].x).hypot(vb[prev + 1].y - vb[prev].y);
            if before.tangent[prev] <= leg / 2.0 + 1e-9 {
                // not limited by the leg to the cursor: the whole curve is untouched
                assert_eq!(rb, ra, "click {k}");
                let end = arc(&before, prev).1;
                assert!((end - arc(&after, prev).1).abs() < 1e-9);
                same += 1;
            } else if ra < rb - 1e-6 {
                tighter += 1;
            }
            // and the new click is a corner with its own curve
            assert!(after.radius[prev + 1] > 0.0, "click {k}");
        }
        assert!(tighter >= 1 && same >= 3, "both cases covered: {tighter} tighter, {same} unchanged");
    }

    /// T-093: dragging blueprint nodes.
    #[test]
    fn drag_blueprint_nodes() {
        let mut a = api();
        // a 6 km line east, a station splitting it at 4.5 km, a branch from a junction at 1.5 km
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let e = a.edges_info()[0];
        assert_eq!(a.add_station(2, e as u32, 4500.0, 0.0, 200, "Middle".into()), 0, "{}", a.issues());
        let ends = [2.0, e, 1500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let node_at = |a: &TrackApi, x: f64, y: f64| a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
        let (w, s, east, b) = (node_at(&a, 0.0, 0.0), node_at(&a, 4500.0, 0.0), node_at(&a, 6000.0, 0.0), node_at(&a, 3500.0, 1500.0));
        let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).unwrap()[0] as u32;
        for n in [w, east, b] {
            assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let l1 = a.new_line_id();
        assert_eq!(a.set_line(l1, &[w, s, east], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0xff0000, "Main".into()), 0, "{}", a.issues());
        let l2 = a.new_line_id();
        assert_eq!(a.set_line(l2, &[w, b], 12.0, 8.0, 4.0, 30.0, 180.0, 0, 0x0000ff, "Branch".into()), 0, "{}", a.issues());
        let heading = |a: &TrackApi, n: u32| a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap()[9];
        let same_axis = |h1: f64, h2: f64| geom::wrap(h1 - h2).abs() < 1e-6 || (geom::wrap(h1 - h2).abs() - PI).abs() < 1e-6;
        let at = |a: &TrackApi, n: u32| { let c = a.nodes_info().chunks(10).find(|c| c[0] as u32 == n).unwrap().to_vec(); (c[1], c[2]) };
        let paths = |a: &TrackApi| (a.line_path(l1), a.line_path(l2), a.line_stops(l1), a.line_stops(l2));
        let before = paths(&a);
        let (hs, hj) = (heading(&a, s), heading(&a, j));
        let t_end = a.stop_times(l1, 0, 0)[2];

        // the end of the line (a terminus station on a track end): its heading follows its edge;
        // the station before it keeps its heading, its track bends out of it
        let pv = a.preview_move_node(east, 6200.0, 400.0);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(pv[2], 1.0, "one edge follows");
        assert_eq!(at(&a, east), (6000.0, 0.0), "a preview changes nothing");
        assert_eq!(a.move_node(east, 6200.0, 400.0), 0, "{}", a.issues());
        assert_eq!(at(&a, east), (6200.0, 400.0));
        assert!(same_axis(heading(&a, s), hs), "the station keeps its heading");
        assert!(a.stop_times(l1, 0, 0)[2] > t_end, "a longer run to the end");
        assert_eq!(paths(&a), before, "lines keep their stops and paths");
        assert_eq!(a.lines_info()[2], 1.0, "and still work");
        assert_eq!(a.undo(), 0);
        assert_eq!(at(&a, east), (6000.0, 0.0));
        assert_eq!(a.stop_times(l1, 0, 0)[2], t_end);

        // the station in the middle: it keeps its heading, both sides bend to meet it
        assert_eq!(a.move_node(s, 4500.0, 250.0), 0, "{}", a.issues());
        assert!(same_axis(heading(&a, s), hs) && same_axis(heading(&a, j), hj), "headings kept");
        assert_eq!(paths(&a), before);
        assert!(a.lines_info()[2] == 1.0 && a.lines_info()[4] == 0.0, "the line works, not broken");
        // and again: the lead-in from the last drag moves with it, nothing piles up
        let n_pis = a.w.net.pis.data.len();
        let pis_s = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).map(|e| a.w.net.edge_data(e).unwrap().pis.len()).sum::<usize>();
        let k0 = pis_s(&a);
        assert_eq!(a.move_node(s, 4600.0, 300.0), 0, "{}", a.issues());
        assert_eq!(pis_s(&a), k0, "{} PIs before, {} after ({n_pis})", k0, pis_s(&a));
        assert!(same_axis(heading(&a, s), hs));
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);

        // the junction: its branch's lead-in moves with it, the main line bends
        assert_eq!(a.move_node(j, 1700.0, -150.0), 0, "{}", a.issues());
        assert!(same_axis(heading(&a, j), hj));
        assert_eq!(paths(&a), before);
        assert_eq!(a.undo(), 0);

        // too far: the track would turn back on itself; refused, nothing changes
        let pv = a.preview_move_node(east, 4400.0, 60.0);
        assert_eq!(pv[0], 0.0);
        assert_ne!(a.move_node(east, 4400.0, 60.0), 0);
        assert!(!a.issue_kinds().is_empty());
        assert_eq!(at(&a, east), (6000.0, 0.0));

        // constructed track meeting there: it stays put
        let e_wj = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && { let d = a.w.net.edge_data(e).unwrap(); (d.a == w && d.b == j) || (d.a == j && d.b == w) }).unwrap();
        assert_eq!(a.construct(&[e_wj], &[]), 0, "{}", a.issues());
        assert_ne!(a.move_node(j, 1700.0, -150.0), 0);
        assert!(a.issue_kinds().contains("NodeInUse"));
        assert_eq!(a.move_node(b, 3600.0, 1600.0), 0, "a blueprint end elsewhere still moves: {}", a.issues());
        // a constructed station stays put
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        assert_ne!(a.move_node(s, 4500.0, 250.0), 0);
        assert_eq!(a.preview_move_node(s, 4500.0, 250.0)[0], 0.0);
    }

    /// Track drawn while the track ran through the clicked points (T-079, saves TWT3 with flag 8)
    /// loads and looks as it did; its first reshape cuts the corners at its clicks.
    #[test]
    fn t079_track_keeps_its_shape_until_reshaped() {
        let mut a = api();
        let (n0, n1, e) = (a.new_node_id(), a.new_node_id(), a.new_edge_id());
        assert_eq!(a.set_node(n0, 0.0, 0.0, 0, false), 0);
        assert_eq!(a.set_node(n1, 4000.0, 0.0, 0, false), 0);
        // as T-079 stored it: fitted PIs with set radii, and the point clicked
        let fitted = vec![Pi::new(1200.0, 380.0, 1500.0, 0), Pi::new(2800.0, 380.0, 1500.0, 0)];
        let clicks = vec![Pi::new(2000.0, 300.0, 0.0, 0)];
        let op = Op::Edge { id: e, data: Some(EdgeData { a: n0, b: n1, tracks: 2, pis: fitted.clone(), built: false, thru: clicks.clone() }) };
        assert!(a.w.apply(op).is_ok());
        let bytes = a.save();
        assert_eq!(&bytes[..4], b"TWT3");
        let mut b = api();
        assert!(b.load(&bytes));
        assert_eq!(b.edges_info(), a.edges_info(), "loads as it was");
        assert_eq!(b.edge_render(0), a.edge_render(e), "and looks the same");
        assert_eq!(b.save(), bytes, "and saves the same");
        // its squares are the clicks, as auto PIs
        let p = b.edge_pis(0);
        assert_eq!(&p[..5], &[1.0, 2000.0, 300.0, 0.0, 0.0]);
        // reshaping it: the click becomes a corner; no clicks kept after that
        assert_eq!(b.set_edge_pis(0, &[2000.0, 400.0, 0.0, 0.0]), 0, "{}", b.issues());
        assert_eq!(b.edge_pis(0)[0], 0.0);
        assert!(b.w.net.edge_thru[0].is_empty());
        assert_ne!(b.edges_info()[5], a.edges_info()[5]);
        assert_eq!(b.undo(), 0);
        assert_eq!(b.edge_render(0), a.edge_render(e), "undo brings the T-079 shape back");
        assert_eq!(b.edge_pis(0)[0], 1.0);
    }

    #[test]
    fn degenerate_routes_do_not_panic() {
        let mut a = api();
        let same: Vec<f64> = free(10.0, 10.0).iter().chain(free(10.0, 10.0).iter()).copied().collect();
        assert_eq!(a.preview_route(&same, &[], 2, false)[0], 0.0);
        assert_ne!(a.add_route(&same, &[], 2, false), 0);
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.preview_route(&ends, &[0.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
        assert_eq!(a.preview_route(&ends, &[1000.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0], 2, false)[0], 0.0);
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n = a.nodes_info()[0];
        let e = a.edges_info()[0];
        // From a node to the same node, and onto the edge at its own end.
        let r = [1.0, n, 0.0, 0.0, 0.0, 1.0, n, 0.0, 0.0, 0.0];
        assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
        let r = [1.0, n, 0.0, 0.0, 0.0, 2.0, e, 1.0, 0.0, 0.0];
        assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
        let r = [2.0, e, 500.0, 0.0, 0.0, 2.0, e, 500.0, 0.0, 0.0];
        assert_eq!(a.preview_route(&r, &[], 2, false)[0], 0.0);
        assert_eq!(a.edges_info().len(), 10);
    }

    #[test]
    fn branch_leaves_along_the_heading() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let e = a.edges_info()[0];
        // Branch from the middle of the edge, off to the north-east.
        let ends = [2.0, e, 2000.0, 5.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 0.0];
        let pv = a.preview_route(&ends, &[], 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.edges_info().len() / 10, 3);
        let j = a.nodes_info().chunks(10).find(|c| c[4] == 3.0).map(|c| c[0]);
        assert!(j.is_some(), "a junction");
        // Extending a track end, turning a little.
        let end_node = a.nodes_info().chunks(10).find(|c| (c[1] - 4000.0).abs() < 1e-3).unwrap()[0];
        let ends = [1.0, end_node, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, -600.0, 0.0];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        // Undo all three, redo one.
        assert_eq!(a.undo(), 0);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.edges_info().len() / 10, 1);
        assert_eq!(a.redo(), 0);
        assert_eq!(a.edges_info().len() / 10, 3);
        // Ground level over water is refused with a kind the UI can word.
        let mut w = api();
        w.w.net.set_water_mask(Box::new(|x: f64, _y: f64| x > 1000.0 && x < 1500.0));
        w.w.reload();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
        assert_ne!(w.add_route(&ends, &[], 2, false), 0);
        assert!(w.issue_kinds().contains("GroundOverWater"), "{}", w.issue_kinds());
    }
}
