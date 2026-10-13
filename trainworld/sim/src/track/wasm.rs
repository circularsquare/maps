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

use super::boarding::Boarding;
use super::cost::{GeoMask, NoWater, RunMask, TransformedMask, UnionMask, WaterMask};
use super::frame::FrameKind;
use super::geom::{self, GeomIssue, Pi};
use super::net::{EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::profile;
use super::save;
use super::world::{ChargeError, EditResult, Op, TrackWorld};
use std::f64::consts::PI;
use std::sync::Arc;
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
    /// The express pair of the last quad route asked about (T-255), else empty.
    route_express: Vec<Pi>,
    /// Per rendered segment of the last route/edge preview, authoritative geometry speed km/h.
    preview_speeds: Vec<f32>,
    /// The pack rasters installed with `add_pack_water`, kept so the water mask can be rebuilt
    /// when a loaded save is in another frame (T-125).
    rasters: Vec<PackWater>,
    /// The boarding account (T-164): demand's riders per period, and each train's load and each
    /// platform's crowd worked out from them. Never saved.
    boarding: Boarding,
    /// Game seconds now (`set_now`; NaN until the worker sets it: edits then re-space at once).
    now: f64,
}

/// A pack's water raster and the origin of the pack's own frame.
struct PackWater {
    mask: Arc<RunMask>,
    lon: f64,
    lat: f64,
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

/// A reshape of a blueprint edge (`TrackApi::reshaped`): the edit, the edge's vertices after it,
/// and the neighbouring pieces whose joint moves with it (T-244), in the stroke's order: `before`
/// ends at the edge's a node, `after` starts at its b node (empty where none moves).
struct Reshape {
    op: Op,
    verts: Vec<Pi>,
    tracks: u8,
    before: Vec<Pi>,
    after: Vec<Pi>,
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
            route_express: vec![],
            preview_speeds: vec![],
            rasters: vec![],
            boarding: Boarding::default(),
            now: f64::NAN,
        }
    }

    /// Demand's riders for the boarding account (T-164), as `Boarding::set_riders` reads them:
    /// 5 busiest-hour factors, then per line its id, stop count `n`, and per period boardings
    /// then alightings at each of its `2n` route nodes. One kind of day's (T-260: 0 weekday,
    /// 1 weekend; each hour reads its own day's, `params::day_kind_at`).
    pub fn set_riders(&mut self, packed: Vec<f32>, day: u32) {
        self.boarding.set_riders_day(&packed, day as usize);
    }
    /// A line's day in the boarding account for its panel (T-230, `Boarding::line_summary`), on
    /// a kind of day (T-260); empty before riders.
    pub fn line_account(&self, line: u32, day: u32) -> Vec<f64> {
        self.boarding.line_summary_day(&self.w.svc, line as usize, day as usize)
    }
    /// Forget the riders (a game was loaded: its demand comes later).
    pub fn clear_riders(&mut self) {
        self.boarding.clear();
    }
    pub fn has_riders(&self) -> bool {
        self.boarding.has_riders()
    }
    /// The hour `[epoch, epoch + 3600)` of the boarding account, lined up with `trips(epoch)`
    /// (`Boarding::hour` has the layout). Brings the account up to date first.
    pub fn boarding_hour(&mut self, epoch: f64) -> Vec<f32> {
        self.boarding.hour(&self.w.net, &self.w.svc, epoch)
    }
    /// Work out at most `max_lines` out-of-date lines of the account; true if more remain. The
    /// clock worker steps through new riders this way between messages (T-164).
    pub fn boarding_step(&mut self, max_lines: u32) -> bool {
        self.boarding.ensure_some(&self.w.net, &self.w.svc, max_lines as usize)
    }
    /// Lines whose account was worked out again since the last call.
    pub fn boarding_redone(&mut self) -> u32 {
        self.boarding.take_redone() as u32
    }

    /// Make the network empty and put it in the world frame (Lambert conformal, `frame.rs`) or the
    /// legacy one. False, changing nothing, if anything is drawn: a network's metres cannot be
    /// reinterpreted, only converted (`convert_save_frame`).
    pub fn set_empty_frame(&mut self, world: bool) -> bool {
        let n = &self.w.net;
        if (0..n.node_count() as u32).any(|i| n.node_ok(i)) || (0..n.line_count() as u32).any(|l| n.line_ok(l)) {
            return false;
        }
        let kind = if world { FrameKind::Lcc } else { FrameKind::Equirect };
        let mut net = Network::new_in(kind, n.origin_lon, n.origin_lat, n.right_hand);
        std::mem::swap(&mut net.water_mask, &mut self.w.net.water_mask);
        self.w = TrackWorld::new(net);
        self.rebuild_water();
        true
    }

    /// The frame of the loaded network: 0 legacy equirectangular, 1 world (Lambert conformal).
    pub fn frame_kind(&self) -> u8 {
        self.w.net.frame_kind as u8
    }
    /// The frame's origin: [lon, lat] in degrees.
    pub fn frame_origin(&self) -> Vec<f64> {
        vec![self.w.net.origin_lon, self.w.net.origin_lat]
    }
    /// Plane metres to [lon, lat] in the loaded network's frame.
    pub fn to_lonlat(&self, x: f64, y: f64) -> Vec<f64> {
        let (lon, lat) = self.w.net.frame().to_lonlat(x, y);
        vec![lon, lat]
    }
    /// [lon, lat] to plane metres in the loaded network's frame.
    pub fn from_lonlat(&self, lon: f64, lat: f64) -> Vec<f64> {
        let (x, y) = self.w.net.frame().from_lonlat(lon, lat);
        vec![x, y]
    }
    /// Many points at once: x, y pairs in, lon, lat pairs out.
    pub fn many_to_lonlat(&self, xy: &[f64]) -> Vec<f64> {
        let f = self.w.net.frame();
        let mut out = Vec::with_capacity(xy.len());
        for p in xy.chunks_exact(2) {
            let (lon, lat) = f.to_lonlat(p[0], p[1]);
            out.push(lon);
            out.push(lat);
        }
        out
    }

    /// The pack's water mask (notes/T-004.md): header `cell_m`, `origin_m`, `size`, and the
    /// `water_row` (u32) and `water_x` (u16) arrays.
    pub fn set_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16]) {
        let m = RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() };
        self.rasters.clear();
        self.w.net.set_water_mask(Box::new(m));
        self.w.reload();
    }

    /// Reset coverage before installing the region's pack rasters.
    pub fn clear_water(&mut self) {
        self.rasters.clear();
        self.w.net.set_water_mask(Box::new(NoWater));
        self.w.reload();
    }

    /// Append a pack raster transformed into the fixed network frame by an offset and scale
    /// (the legacy frame only; `add_pack_water` serves every frame).
    pub fn append_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16], offset_x: f64, offset_y: f64, scale_x: f64, scale_y: f64) {
        assert!(scale_x.is_finite() && scale_x > 0.0 && scale_y.is_finite() && scale_y > 0.0);
        let m = TransformedMask {
            mask: Arc::new(RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() }),
            offset: [offset_x, offset_y], scale: [scale_x, scale_y],
        };
        let previous = std::mem::replace(&mut self.w.net.water_mask, Box::new(NoWater));
        self.w.net.set_water_mask(Box::new(UnionMask(previous, Box::new(m))));
        self.w.reload();
    }

    /// Add a pack's water raster (T-125). `pack_lon`, `pack_lat` are the origin of the pack's own
    /// equirectangular frame, in which the raster's grid is laid out. The mask follows the loaded
    /// network's frame and is rebuilt when a save in another frame is loaded.
    pub fn add_pack_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16], pack_lon: f64, pack_lat: f64) {
        let mask = Arc::new(RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() });
        self.rasters.push(PackWater { mask, lon: pack_lon, lat: pack_lat });
        self.rebuild_water();
    }

    /// Replace the network with a saved one. False if the bytes are not a save.
    pub fn load(&mut self, bytes: &[u8]) -> bool {
        match save::decode(bytes) {
            Ok(mut net) => {
                std::mem::swap(&mut net.water_mask, &mut self.w.net.water_mask);
                self.w = TrackWorld::new(net);
                if !self.rasters.is_empty() {
                    self.rebuild_water();
                }
                true
            }
            Err(_) => false,
        }
    }

    /// The save's bytes in another frame (0 legacy, 1 world), see `save::convert_frame`. Empty if
    /// the bytes are not a save.
    pub fn convert_save_frame(bytes: &[u8], world: bool) -> Vec<u8> {
        save::convert_frame(bytes, if world { FrameKind::Lcc } else { FrameKind::Equirect }).unwrap_or_default()
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

    /// Cost rows for Construct all, including new cars after using the spare fleet.
    /// Infrastructure rows come from the same pricing intervals as track_cost.
    pub fn blueprint_cost_items(&mut self) -> Vec<f64> {
        let mut rows = self.w.net.blueprint_cost_items();
        if !rows.is_empty() && self.w.svc.lines.iter().any(|s| s.ok && !s.running) {
            let net = &self.w.net;
            let edges: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && !net.edge_built[e as usize]).collect();
            let stations: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && !net.node_built[n as usize]).collect();
            let ops = self.construct_ops(&edges, &stations);
            let fleet = self.fleet;
            // Starting several lines together can change junction delays and their fleet needs.
            // Quote the derived constructed service, not a sum of the planned lines' estimates.
            let cost = self.w.trial_with(Op::Batch(ops), |w| trains_to_buy(w, fleet)).map_or(0.0, |(_, c)| c);
            if cost > 0.0 {
                rows.extend([5.0, 0.0, 0.0, 0.0, 0.0, cost / CAR_PRICE, cost]);
            }
        }
        rows
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
        let before = self.w.svc.before_edit(&self.w.net, self.now);
        let r = self.w.apply_paid(op, self.cash, false, &|w| trains_to_buy(w, fleet));
        self.settle(r, &before)
    }

    /// A final, charged edit (see the module comment), trains included.
    fn run_charged(&mut self, op: Op) -> u32 {
        let fleet = self.fleet;
        let before = self.w.svc.before_edit(&self.w.net, self.now);
        let r = self.w.apply_paid(op, self.cash, true, &|w| trains_to_buy(w, fleet));
        self.settle(r, &before)
    }

    /// The game time now, from the clock worker (T-206): an edit that changes a line's departures
    /// takes effect from here, trains already running keep their places, and patterns from before
    /// are forgotten once their trains are no longer drawn.
    pub fn set_now(&mut self, t: f64) {
        self.now = t;
        self.w.svc.prune_eras(&self.w.net, t);
    }

    fn settle(&mut self, r: Result<(EditResult, f64), ChargeError>, before: &super::service::Before) -> u32 {
        self.last_trains = 0.0;
        if r.is_ok() {
            self.w.svc.note_changes(&self.w.net, before, self.now);
        }
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
            Ok(r) => self.run(r.op),
            Err(i) => self.refuse(i),
        }
    }

    /// What `set_edge_pis` would do, without doing it: the layout of `preview_route`. Its
    /// polyline includes the next pieces of the stroke where a joint moves with it (T-244).
    pub fn preview_edge(&mut self, id: u32, pis: &[f64]) -> Vec<f64> {
        self.preview_speeds.clear();
        let r = match self.reshaped(id, pis) {
            Ok(r) => r,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let t = self.w.trial(r.op);
        let mut out = self.preview_out(t, &r.verts, r.tracks);
        if r.before.is_empty() && r.after.is_empty() {
            return out;
        }
        let main = out.split_off(5 + 4 * r.verts.len());
        let main_speeds = std::mem::take(&mut self.preview_speeds);
        let joint_speed = main_speeds.first().copied().unwrap_or((V_TOP * 3.6) as f32);
        let mut speeds = vec![];
        if !r.before.is_empty() {
            let f = geom::fit(&r.before);
            render_samples(&f.pieces, &f.vert, &mut out);
            speeds.extend(render_speeds(&f.pieces, &f.vert));
            speeds.push(joint_speed);
        }
        out.extend(main);
        speeds.extend(main_speeds);
        if !r.after.is_empty() {
            let f = geom::fit(&r.after);
            speeds.push(joint_speed);
            render_samples(&f.pieces, &f.vert, &mut out);
            speeds.extend(render_speeds(&f.pieces, &f.vert));
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
    fn reshaped(&self, id: u32, pis: &[f64]) -> Result<Reshape, Issue> {
        let Some(mut d) = self.editable(id) else { return Err(Issue::NoSuchEdge { edge: id }) };
        if d.built {
            return Err(Issue::NodeInUse { node: d.a });
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
                let Some((o, which, od, far, t)) = self.free_joint(node, id, near_old) else { continue };
                moved.push((node, quant(near_new.x + t * (far.x - near_new.x)), quant(near_new.y + t * (far.y - near_new.y))));
                others.push((o, which, od));
            }
        }
        relax_radii(net, &mut d, &moved, &|k, p| same_count && old[k].radius == p.radius);
        let verts = vertices_of(net, &d, &moved);
        let tracks = d.tracks;
        if moved.is_empty() {
            return Ok(Reshape { op: Op::Edge { id, data: Some(d) }, verts, tracks, before: vec![], after: vec![] });
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
        Ok(Reshape { op: Op::Batch(ops), verts, tracks, before, after })
    }

    /// The other piece at a joint of a stroke (T-244) that may follow a corner of `e` next to it
    /// (`near`, before the edit): a node with no station where only `e` and one other blueprint
    /// edge meet, on the straight between `near` and the other piece's corner next to it. Returns
    /// that edge, which of its ends is at the joint, its data, that corner (or its far node, if it
    /// has none) and how far along from `near` to it the joint lies (0..1).
    fn free_joint(&self, node: u32, e: u32, near: Pi) -> Option<(u32, u32, EdgeData, Pi, f64)> {
        let net = &self.w.net;
        let i = node as usize;
        let p = &net.node_ports[i];
        if p.n != 2 || net.node_platform[i] > 0 || net.built_ports(node) > 0 {
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
    /// change is set to the one it had), lines through it follow. None if no joint is that near.
    fn unjoin_near(&self, e: u32, s: f64, reach: f64) -> Option<(Vec<Op>, f64)> {
        let net = &self.w.net;
        let d = net.edge_data(e)?;
        if !d.thru.is_empty() {
            return None;
        }
        let len = net.edge_len[e as usize];
        let own = vertices_of(net, &d, &[]);
        let ends = [(d.a, s < reach, own[1]), (d.b, len - s < reach, own[own.len() - 2])];
        let mut merges: [Option<(u32, u32, EdgeData)>; 2] = [None, None];
        for (k, &(node, near, corner)) in ends.iter().enumerate() {
            if !near {
                continue;
            }
            if let Some((o, which, od, _, _)) = self.free_joint(node, e, corner) {
                if od.built == d.built && od.tracks == d.tracks && o != e && merges[0].as_ref().map_or(true, |m| m.0 != o) {
                    merges[k] = Some((o, which, od));
                }
            }
        }
        if merges.iter().all(|m| m.is_none()) {
            return None;
        }
        // the vertices of each part in the joined edge's direction, with each corner's radius
        let radii = |v: &[Pi]| geom::fit(v).radius;
        let mut verts: Vec<(Pi, f64)> = vec![];
        let mut s = s;
        let mut a = d.a;
        if let Some((o, which, od)) = &merges[0] {
            let mut v = vertices_of(net, od, &[]);
            let mut r = radii(&v);
            if *which == 0 {
                v.reverse();
                r.reverse();
            }
            a = net.end_node(*o, 1 - which);
            verts.extend(v.iter().copied().zip(r).take(v.len() - 1));
            s += net.edge_len[*o as usize];
        }
        let r = radii(&own);
        let skip = if merges[0].is_some() { 1 } else { 0 };
        let keep = if merges[1].is_some() { own.len() - 1 } else { own.len() };
        verts.extend(own.iter().copied().zip(r).take(keep).skip(skip));
        let mut b = d.b;
        if let Some((o, which, od)) = &merges[1] {
            let mut v = vertices_of(net, od, &[]);
            let mut r = radii(&v);
            if *which == 1 {
                v.reverse();
                r.reverse();
            }
            b = net.end_node(*o, 1 - which);
            verts.extend(v.iter().copied().zip(r).skip(1));
        }
        let mut pis: Vec<Pi> = verts.iter().map(|v| v.0).collect();
        let f = geom::fit(&pis);
        for j in 1..pis.len() - 1 {
            let want = verts[j].1;
            if want > 0.0 && (f.radius[j] - want).abs() > 0.01 {
                pis[j].radius = quant(want);
            }
        }
        let joined = EdgeData { a, b, tracks: d.tracks, pis: pis[1..pis.len() - 1].to_vec(), built: d.built, thru: vec![] };
        // each part's direction that runs the joined edge's way
        let mut forward: Vec<(u32, u32)> = vec![(e, 0)];
        if let Some((o, which, _)) = &merges[0] {
            forward.push((*o, if *which == 1 { 0 } else { 1 }));
        }
        if let Some((o, which, _)) = &merges[1] {
            forward.push((*o, if *which == 0 { 0 } else { 1 }));
        }
        let mut ops = vec![Op::Edge { id: e, data: Some(joined) }];
        for (k, m) in merges.iter().enumerate() {
            if let Some((o, _, _)) = m {
                ops.push(Op::Edge { id: *o, data: None });
                ops.push(Op::Node { id: if k == 0 { d.a } else { d.b }, data: None });
            }
        }
        for l in 0..net.line_count() as u32 {
            if !net.line_ok(l) || !net.line_path(l).iter().any(|&p| forward.iter().any(|f| f.0 == super::net::path_edge(p))) {
                continue;
            }
            let mut ld = net.line_data(l)?;
            let mut path: Vec<u32> = vec![];
            for &p in &ld.path {
                let pe = super::net::path_edge(p);
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
        let f = geom::fit(&vertices_of(net, &d, &[]));
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
            relax_radii(net, &mut d, &[(node, x, y)], &|_, _| true);
            shapes.push(vertices_of(net, &d, &[(node, x, y)]));
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
        self.preview_speeds = render_speeds(&f.pieces, &f.vert);
        out
    }

    /// Curve-limited km/h per consecutive segment in the last route/edge preview polyline.
    /// Separate from its stable vertex/point layout; empty after an unrenderable refusal.
    pub fn preview_render_speeds(&self) -> Vec<f32> {
        self.preview_speeds.clone()
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
        let first_click = verts.len();
        verts.extend_from_slice(&pis);
        if let Some(p) = lead_in(&rb, prev_b) {
            verts.push(p);
        }
        verts.push(Pi::new(rb.x, rb.y, 0.0, rb.level));
        let mut ops = ra.ops.clone();
        ops.extend(rb.ops.clone());
        // Quad track (T-255): the drawn pair, and an express pair beside it from end to end.
        let quad = tracks == 4;
        let tracks = if quad { 2 } else { tracks };
        // One piece of track per corner clicked (T-244), joined at plain nodes, the shape unchanged.
        let pieces = stroke_pieces(&verts, first_click, pis.len());
        self.push_pieces(&pieces, ra.node, rb.node, tracks, &mut ops, &mut new_nodes, &mut new_edges);
        let mut express = vec![];
        if quad {
            let (ex, first, clicks) = match express_verts(&verts, first_click, pis.len()) {
                Some(v) => v,
                None => {
                    self.release(&new_nodes, &new_edges);
                    return Err(Issue::Geom { edge: u32::MAX, issue: GeomIssue::ZeroLeg(0) });
                }
            };
            let pieces = stroke_pieces(&ex, first, clicks);
            self.push_pieces(&pieces, ra.node, rb.node, tracks, &mut ops, &mut new_nodes, &mut new_edges);
            express = ex;
        }
        // A junction this route makes (it had two edge ends, or it splits track) is flying if
        // the build setting says so; never one that already stands.
        for r in [&ra, &rb] {
            if flying && r.new_junction {
                ops.push(Op::Node { id: r.node, data: Some(NodeData { x: r.x, y: r.y, level: r.level, flying: true }) });
            }
        }
        self.route_express = express;
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
        self.preview_speeds.clear();
        let (op, verts, nn, ne) = match self.route_op(ends, pis, tracks, flying) {
            Ok(v) => v,
            Err(i) => {
                self.refuse(i);
                return vec![0.0; 5];
            }
        };
        let r = self.w.trial(op);
        self.release(&nn, &ne);
        let mut out = self.preview_out(r, &verts, tracks.min(2));
        if !self.route_express.is_empty() && out.len() > 5 {
            // a quad route's express pair, drawn on from the end back to the start (T-255)
            let mut back = self.route_express.clone();
            back.reverse();
            let f = geom::fit(&back);
            let joint = self.preview_speeds.last().copied().unwrap_or((V_TOP * 3.6) as f32);
            self.preview_speeds.push(joint);
            render_samples(&f.pieces, &f.vert, &mut out);
            self.preview_speeds.extend(render_speeds(&f.pieces, &f.vert));
        }
        self.route_vertices = verts;
        out
    }

    /// Edges for the pieces of a stroke (`stroke_pieces`) from node `a` to node `b`, with new
    /// joints between them.
    #[allow(clippy::too_many_arguments)]
    fn push_pieces(&mut self, pieces: &[Vec<Pi>], a: u32, b: u32, tracks: u8, ops: &mut Vec<Op>, nodes: &mut Vec<u32>, edges: &mut Vec<u32>) {
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
            ops.push(Op::Edge { id: edge, data: Some(EdgeData { a: from, b: to, tracks, pis: p[1..p.len() - 1].to_vec(), built: false, thru: vec![] }) });
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
            // a joint or plain node on quad track: its twin beside it too (T-255)
            let ends = self.w.net.node_ports[id as usize].ends().to_vec();
            if let Some(&c) = ends.first() {
                let (e, which) = (c >> 1, c & 1);
                let s = if which == 0 { 0.0 } else { self.w.net.edge_len[e as usize] };
                if let Some((twin, ts)) = self.twin_at(e, s) {
                    let r = self.station_ops(&[(u32::MAX, id as f64), (twin, ts)], &st);
                    if r == 0 {
                        return 0;
                    }
                }
            }
            return self.run(Op::Station { node: id, data: Some(st) });
        }
        if !self.w.net.edge_ok(id) {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        let s = self.w.net.project(id, x, y).0;
        // On quad track (T-255) the pair beside gets a station of the same name level with it:
        // the local and express platforms of one station, a complex. If that cannot be built,
        // the one station alone.
        if let Some((twin, ts)) = self.twin_at(id, s) {
            let r = self.station_ops(&[(id, s), (twin, ts)], &st);
            if r == 0 {
                return 0;
            }
        }
        self.station_ops(&[(id, s)], &st)
    }

    /// Stations splitting edges at chainages, one undoable edit (`add_station`). A joint of a drawn
    /// route (T-244) within a platform's reach goes first: the pieces either side become one edge
    /// again, so the platform has room. `(u32::MAX, node)` puts one on an existing node instead.
    fn station_ops(&mut self, at: &[(u32, f64)], st: &StationData) -> u32 {
        let mut ops = vec![];
        let (mut nodes, mut edges) = (vec![], vec![]);
        for &(e, s) in at {
            if e == u32::MAX {
                // an existing node, `s` its id
                ops.push(Op::Station { node: s as u32, data: Some(st.clone()) });
                continue;
            }
            let (more, s) =self.unjoin_near(e, s, st.platform as f64 / 2.0 + 2.0).unwrap_or((vec![], s));
            ops.extend(more);
            let (node, new_edge) = (self.w.net.alloc_node(), self.w.net.alloc_edge());
            nodes.push(node);
            edges.push(new_edge);
            ops.extend([Op::Split { edge: e, s, node, new_edge }, Op::Station { node, data: Some(st.clone()) }]);
        }
        let r = self.run(Op::Batch(ops));
        if r != 0 {
            self.release(&nodes, &edges);
        }
        r
    }

    /// The track beside edge `e` at chainage `s` on quad track (T-255): another edge running
    /// parallel at about `QUAD_SPACING`, and the chainage on it level with `s`.
    fn twin_at(&self, e: u32, s: f64) -> Option<(u32, f64)> {
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
        best.map(|b| (b.0, b.1))
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
    /// Its links (T-197) go with it, and undoing the removal does not bring them back.
    pub fn remove_station(&mut self, node: u32) -> u32 {
        let Some(old) = self.w.net.station_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
        let mut ops = vec![Op::Station { node, data: None }];
        if self.w.net.node_ports[node as usize].n == 0 {
            ops.push(Op::Node { id: node, data: None });
        }
        let r = if old.built {
            self.run_charged(Op::Batch(ops))
        } else {
            self.run(Op::Batch(ops))
        };
        if r == 0 {
            self.w.net.links.retain(|&[a, b]| a != node && b != node);
        }
        r
    }

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
        let Some(path) = self.w.net.route(stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { symbol: String::new(), trains: None, name, colour, stops: stops.to_vec(), path, tph: [high, medium, low], dwell_s, turnaround_s, cars }) })
    }
    /// Add a line with a whole-line fleet at each demand level.
    #[allow(clippy::too_many_arguments)]
    pub fn set_line_with_trains(&mut self, id: u32, stops: &[u32], high: u32, medium: u32, low: u32, dwell_s: f32, turnaround_s: f32, cars: u8, colour: u32, name: String) -> u32 {
        let Some(path) = self.w.net.route(stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { symbol: String::new(), trains: Some([high, medium, low]), name, colour, stops: stops.to_vec(), path, tph: if stops.len() < 2 { [0.0; 3] } else { [6.0, 3.0, 1.0] }, dwell_s, turnaround_s, cars }) })
    }
    /// Change a line's stops, keeping everything else; the path is found again.
    pub fn set_stops(&mut self, id: u32, stops: &[u32]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
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
    /// Custom badge symbol; omitted look edits retain it.
    pub fn set_line_badge(&mut self, id: u32, name: String, colour: u32, symbol: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.name = name; d.colour = colour; d.symbol = symbol;
        self.run(Op::Line { id, data: Some(d) })
    }
    pub fn copy_line(&mut self, source: u32, id: u32, name: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(source) else { return self.refuse(Issue::NoSuchLine { line: source }) };
        d.name = name;
        let r = self.run(Op::Line { id, data: Some(d) });
        if r == 0 { self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1); }
        r
    }
    /// A copy of a line for an express service (T-255): every stop with a station on the pair
    /// beside it (quad track, `twin_at`'s stations of one name) moves there, and the path is found
    /// again over them, so the copy runs on the other pair wherever there is one. Where that has
    /// no route, a plain copy. The player then takes out the stops it skips.
    pub fn express_copy(&mut self, source: u32, id: u32, name: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(source) else { return self.refuse(Issue::NoSuchLine { line: source }) };
        let net = &self.w.net;
        let stops: Vec<u32> = d.stops.iter().map(|&s| twin_station(net, s).unwrap_or(s)).collect();
        if stops != d.stops {
            if let Some(path) = net.route(&stops) {
                d.stops = stops;
                d.path = path;
            }
        }
        d.name = name;
        let r = self.run(Op::Line { id, data: Some(d) });
        if r == 0 { self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1); }
        r
    }
    pub fn next_line_name(&mut self) -> String {
        loop {
            let name = format!("Line {}", self.w.net.next_line_number);
            self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1);
            if !(0..self.w.net.line_count() as u32).any(|l| self.w.net.line_ok(l) && self.w.net.line_name[l as usize] == name) { return name; }
        }
    }
    /// Try every insertion slot using the same turn-valid route as set_stops.
    pub fn insert_stop(&self, line: u32, station: u32) -> Vec<f64> {
        best_insert(&self.w.net, line, station, &mut HopMemo::new()).map_or(vec![], |(stops, _)| stops.iter().map(|&s| s as f64).collect())
    }

    /// How every station stands to a line while its stops are edited (T-233, T-237): per station
    /// node, `[node, tier]`, tier 0 a stop of the line, 1 adding it keeps the line's path (a
    /// station on it, or past an end on track the path can carry on over: the path is extended),
    /// 2 adding it changes the path, 3 it cannot be added. Worked out with `insert_stop`'s own
    /// search, so the markers say what a click does. Empty if the line is not there.
    pub fn stop_tiers(&self, line: u32) -> Vec<f64> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return vec![];
        }
        let stops = net.line_stops(line);
        let old = net.line_path(line);
        let reach = reachable_from(net, stops);
        let mut memo = HopMemo::new();
        let mut out = vec![];
        for n in 0..net.node_count() as u32 {
            if !net.node_ok(n) || net.node_platform[n as usize] == 0 {
                continue;
            }
            let tier = if stops.contains(&n) {
                0.0
            } else if !stops.is_empty() && !reach[n as usize] {
                3.0
            } else {
                match best_insert(net, line, n, &mut memo) {
                    None => 3.0,
                    Some((_, path)) if old.is_empty() || path.windows(old.len()).any(|w| w == old) => 1.0,
                    Some(_) => 2.0,
                }
            };
            out.extend([n as f64, tier]);
        }
        out
    }
    pub fn remove_line(&mut self, id: u32) -> u32 {
        self.run(Op::Line { id, data: None })
    }
    pub fn set_schedule(&mut self, id: u32, high: f32, medium: f32, low: f32) -> u32 {
        self.run(Op::Schedule { id, tph: [high, medium, low] })
    }

    /// Allocate whole trains to the entire line, across both directions.
    pub fn set_train_counts(&mut self, id: u32, high: u32, medium: u32, low: u32) -> u32 {
        let Some(d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        self.run(Op::TrainCounts { id, trains: Some([high, medium, low]), tph: d.tph })
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
        let before = self.w.svc.before_edit(&self.w.net, self.now);
        match self.w.step_paid(redo, self.cash, &|w| trains_to_buy(w, fleet)) {
            Some(r) => self.settle(r, &before),
            None => {
                self.money_short = None;
                self.issues.clear();
                1
            }
        }
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
    /// Curve-limited km/h for every consecutive segment in edge_render's point triples.
    pub fn edge_render_speeds(&self, e: u32) -> Vec<f32> {
        let n = &self.w.net;
        if !n.edge_ok(e) { return vec![]; }
        render_speeds(n.edge_pieces(e), n.edge_vert(e))
    }

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
                    out.extend([0.0; 4]);
                    out.extend(n.line_trains[i].unwrap_or([0; 3]).iter().map(|&t| t as f64));
                    out.extend([0.0; 3]);
                    out.extend([n.line_dwell[i] as f64, n.line_turn[i] as f64, 0.0]);
                }
            }
        }
        out
    }
    pub fn line_name(&self, id: u32) -> String {
        self.w.net.line_name.get(id as usize).cloned().unwrap_or_default()
    }
    pub fn line_symbol(&self, id: u32) -> String {
        self.w.net.line_symbol.get(id as usize).cloned().unwrap_or_default()
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
    /// Trips drawn in `[epoch, epoch + 3600)`: per trip, profile index, departure relative to
    /// `epoch` (may be negative: still running from before) and when it stops being drawn,
    /// relative to `epoch` (its train leaves service or takes its next departure, T-206), and the
    /// departure its train takes next from that terminal, relative to `epoch` (the other run's
    /// trip drawn from then; NaN when it leaves service there, T-245). Running lines only.
    pub fn trips(&self, epoch: f64) -> Vec<f32> {
        self.w
            .svc
            .trips(&self.w.net, epoch, epoch + 3600.0)
            .iter()
            .flat_map(|t| {
                let next = t.next.map_or(f32::NAN, |n| (n - epoch) as f32);
                [((t.line * 2 + t.run as u32) * 3 + t.level as u32) as f32, (t.draw - epoch) as f32, (t.end - epoch) as f32, next]
            })
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
fn vertices_of(net: &Network, d: &EdgeData, moved: &[(u32, f64, f64)]) -> Vec<Pi> {
    let at = |n: u32| -> Pi {
        let i = n as usize;
        match moved.iter().find(|m| m.0 == n) {
            Some(&(_, x, y)) => Pi::new(x, y, 0.0, net.node_level[i]),
            None => Pi::new(net.node_x[i], net.node_y[i], 0.0, net.node_level[i]),
        }
    };
    let mut v = vec![at(d.a)];
    v.extend_from_slice(&d.pis);
    v.push(at(d.b));
    v
}

/// Set radii that no longer fit their legs go back to auto, where `may(pi index, pi)` allows it.
fn relax_radii(net: &Network, d: &mut EdgeData, moved: &[(u32, f64, f64)], may: &dyn Fn(usize, &Pi) -> bool) {
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

/// Hops already searched while trying stops (T-237): `(from, to, side arrived on at from, 2 for
/// none)` to `Network::find_path`'s answer. Valid for one state of the network.
type HopMemo = std::collections::HashMap<(u32, u32, u8), Option<(Vec<u32>, u8)>>;

/// `Network::route`, with each hop's search kept in `memo`: the same answer, but trying a stop in
/// every slot searches each hop once.
fn route_memo(net: &Network, stops: &[u32], memo: &mut HopMemo) -> Option<Vec<u32>> {
    if stops.iter().any(|&s| !net.node_ok(s) || net.node_platform[s as usize] == 0) {
        return None;
    }
    let mut path = vec![];
    let mut side = None;
    for w in stops.windows(2) {
        let hop = memo.entry((w[0], w[1], side.unwrap_or(2))).or_insert_with(|| net.find_path(w[0], w[1], side));
        let (p, s) = hop.as_ref()?;
        path.extend_from_slice(p);
        side = Some(*s);
    }
    Some(path)
}

/// The line's stops with `station` inserted where its path is shortest, and that path
/// (`TrackApi::insert_stop`): every slot tried with the route search `set_stops` uses, the end
/// winning ties. None if no slot has a route, or it is a stop already.
fn best_insert(net: &Network, line: u32, station: u32, memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    if !net.line_ok(line) {
        return None;
    }
    let old = net.line_stops(line);
    if old.contains(&station) {
        return None;
    }
    let mut best = None;
    let mut best_len = f64::INFINITY;
    for slot in (0..=old.len()).rev() {
        let mut stops = old.to_vec();
        stops.insert(slot, station);
        if let Some(path) = route_memo(net, &stops, memo) {
            let len: f64 = path.iter().map(|p| net.edge_len[(p >> 1) as usize]).sum();
            if len < best_len - 0.001 {
                best_len = len;
                best = Some((stops, path));
            }
        }
    }
    best
}

/// Nodes a train can reach from any of `stops`, leaving it either way, over valid moves. Moves
/// work both ways, so a node outside it has no route to or from a stop either.
fn reachable_from(net: &Network, stops: &[u32]) -> Vec<bool> {
    let mut seen_node = vec![false; net.node_count()];
    let mut seen = vec![false; 2 * net.edge_count()];
    let mut queue: Vec<u32> = vec![];
    for &s in stops {
        if !net.node_ok(s) {
            continue;
        }
        seen_node[s as usize] = true;
        for &c in net.node_ports[s as usize].ends() {
            if !seen[c as usize] {
                seen[c as usize] = true;
                queue.push(c);
            }
        }
    }
    while let Some(st) = queue.pop() {
        let (_, node) = net.path_nodes(st);
        seen_node[node as usize] = true;
        let arr = net.port_side(super::net::path_edge(st), 1 - super::net::path_dir(st)).unwrap_or(0);
        let p = &net.node_ports[node as usize];
        for (k, &c) in p.ends().iter().enumerate() {
            if p.side[k] != arr && !seen[c as usize] {
                seen[c as usize] = true;
                queue.push(c);
            }
        }
    }
    seen_node
}

/// A drawn stroke cut into pieces, one per corner clicked (T-244), so removing one removes only
/// that stretch. `verts` is the whole stroke (end, lead-in, the `clicks` corners from index
/// `first`, lead-in, end); each piece is a vertex list from end or joint to joint or end.
///
/// The cut is at the middle of the leg between two consecutive corners. That point is on the
/// straight (an auto curve takes at most half of a leg between corners), so each corner stays a
/// PI of its own piece with its auto radius, and the radius is the same as in the whole stroke: a
/// corner next to an end may use the whole leg, which is now the half leg it had anyway. A leg
/// that changes level is not cut, since a ramp is centred between its corners and the joint, a
/// node, must be at a whole level. Any radius that would still come out different (rounding to the
/// millimetre grid) is set to the whole stroke's, so the track is where it was drawn.
fn stroke_pieces(verts: &[Pi], first: usize, clicks: usize) -> Vec<Vec<Pi>> {
    let f = geom::fit(verts);
    let mut cuts: Vec<(usize, Pi)> = vec![];
    if f.issues.is_empty() {
        for i in first..(first + clicks).saturating_sub(1) {
            let (p, q) = (verts[i], verts[i + 1]);
            let leg = (q.x - p.x).hypot(q.y - p.y);
            if p.level != q.level || leg < 2.0 * MIN_JOINT_LEG || f.tangent[i] > leg / 2.0 + 1e-6 || f.tangent[i + 1] > leg / 2.0 + 1e-6 {
                continue;
            }
            cuts.push((i, Pi::new(quant((p.x + q.x) / 2.0), quant((p.y + q.y) / 2.0), 0.0, p.level)));
        }
    }
    let mut out = vec![];
    let mut start = 0;
    let mut head = verts[0];
    let end = (verts.len() - 1, verts[verts.len() - 1]);
    for &(i, joint) in cuts.iter().chain(std::iter::once(&end)) {
        let mut p = vec![head];
        p.extend_from_slice(&verts[start + 1..=i.min(verts.len() - 2)]);
        p.push(joint);
        // what the whole stroke uses at each corner of the piece
        let pf = geom::fit(&p);
        for j in 1..p.len() - 1 {
            let g = start + j;
            if f.radius[g] > 0.0 && (pf.radius[j] - f.radius[g]).abs() > 0.01 {
                p[j].radius = quant(f.radius[g]);
            }
        }
        out.push(p);
        start = i;
        head = joint;
    }
    out
}

/// Centre line to centre line between the two double tracks of quad track, m (T-255): 4 m between
/// every pair of neighbouring tracks, as on the double track itself.
pub const QUAD_SPACING: f64 = 8.0;
/// How long the express pair takes to swing out to its place beside the drawn pair, m.
const QUAD_TAPER: f64 = 150.0;

/// The express pair of a quad route (T-255): from the drawn route's `verts` (end, lead-in, the
/// `clicks` corners from index `first`, lead-in, end), a double track starting and ending at the
/// same nodes, leaving each end along its heading, swinging out to `QUAD_SPACING` to the left of
/// the drawn direction over `QUAD_TAPER` (less on a short leg), and running parallel between,
/// each corner concentric with the drawn one (radius smaller by the spacing on the inside of the
/// turn, larger outside). Returns its vertices, the index of its first corner-like vertex and how
/// many there are (the taper ends and the corners), for `stroke_pieces`. None if degenerate.
fn express_verts(verts: &[Pi], first: usize, clicks: usize) -> Option<(Vec<Pi>, usize, usize)> {
    let n = verts.len();
    let f = geom::fit(verts);
    let leg = |i: usize| -> Option<(f64, f64, f64)> {
        let (dx, dy) = (verts[i + 1].x - verts[i].x, verts[i + 1].y - verts[i].y);
        let l = dx.hypot(dy);
        (l > 1e-6).then(|| (dx / l, dy / l, l))
    };
    let d = QUAD_SPACING;
    // the legs the tapers lie on: from the vertex before the first corner, into the one after the last
    let (s, e) = (first - 1, first + clicks);
    let (ux, uy, la) = leg(s)?;
    let (vx, vy, lb) = leg(e - 1)?;
    let ta = QUAD_TAPER.min(if clicks == 0 { 0.3 } else { 0.4 } * la);
    let tb = QUAD_TAPER.min(if clicks == 0 { 0.3 } else { 0.4 } * lb);
    let qa = Pi::new(quant(verts[s].x + ux * ta - uy * d), quant(verts[s].y + uy * ta + ux * d), 0.0, verts[s].level);
    let qb = Pi::new(quant(verts[e].x - vx * tb - vy * d), quant(verts[e].y - vy * tb + vx * d), 0.0, verts[e].level);
    let mut corners = vec![];
    for i in first..first + clicks {
        let (ax, ay, _) = leg(i - 1)?;
        let (bx, by, _) = leg(i)?;
        let (n1, n2) = ((-ay, ax), (-by, bx));
        let dot = n1.0 * n2.0 + n1.1 * n2.1;
        if f.radius[i] <= 0.0 || dot < -0.9 {
            continue; // no curve here (straight on), or turning back
        }
        let k = d / (1.0 + dot);
        let left = ax * by - ay * bx > 0.0;
        let r = if left { f.radius[i] - d } else { f.radius[i] + d };
        corners.push(Pi::new(quant(verts[i].x + k * (n1.0 + n2.0)), quant(verts[i].y + k * (n1.1 + n2.1)), quant(r.max(1.0)), verts[i].level));
    }
    let (a, b) = (verts[0], verts[n - 1]);
    let mut out = vec![Pi::new(a.x, a.y, 0.0, a.level)];
    let (hx, hy, _) = leg(0)?;
    if let Some(p) = lead_in_along(a.x, a.y, a.level, hy.atan2(hx), (qa.x, qa.y)) {
        out.push(p);
    }
    let first_out = out.len();
    out.push(qa);
    out.extend(corners.iter().copied());
    out.push(qb);
    let count = out.len() - first_out;
    let (gx, gy, _) = leg(n - 2)?;
    if let Some(p) = lead_in_along(b.x, b.y, b.level, (-gy).atan2(-gx), (qb.x, qb.y)) {
        out.push(p);
    }
    out.push(Pi::new(b.x, b.y, 0.0, b.level));
    Some((out, first_out, count))
}

/// The station beside station `s` on quad track (T-255): the nearest other station within one
/// and a half times `QUAD_SPACING`, the other platform of the same stop.
fn twin_station(net: &Network, s: u32) -> Option<u32> {
    if !net.node_ok(s) {
        return None;
    }
    let (x, y) = (net.node_x[s as usize], net.node_y[s as usize]);
    (0..net.node_count() as u32)
        .filter(|&n| n != s && net.node_ok(n) && net.node_platform[n as usize] > 0)
        .map(|n| (n, (net.node_x[n as usize] - x).hypot(net.node_y[n as usize] - y)))
        .filter(|&(_, d)| d <= QUAD_SPACING * 1.5)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(n, _)| n)
}

/// The shortest leg either side of a joint between pieces of a stroke, m (T-244).
const MIN_JOINT_LEG: f64 = 20.0;

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
fn render_speeds(pieces: &[geom::Piece], vert: &[[f64; 2]]) -> Vec<f32> {
    if pieces.is_empty() { return vec![]; }
    render_chainages(pieces, vert).windows(2).map(|pair| {
        let mid = (pair[0] + pair[1]) * 0.5;
        let i = pieces.partition_point(|p| p.s0 <= mid).saturating_sub(1);
        (pieces[i].v_limit() * 3.6) as f32
    }).collect()
}

fn render_chainages(pieces: &[geom::Piece], vert: &[[f64; 2]]) -> Vec<f64> {
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
    ss
}

fn render_samples(pieces: &[geom::Piece], vert: &[[f64; 2]], out: &mut Vec<f64>) {
    for s in render_chainages(pieces, vert) {
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

impl TrackApi {
    /// Compose the water mask from the pack rasters for the loaded network's frame. In the legacy
    /// frame each raster is the affine `TransformedMask` earlier builds used (same arithmetic as
    /// the clock worker's, so the costs of legacy saves are unchanged); in the world frame it is
    /// a `GeoMask` through longitude and latitude.
    fn rebuild_water(&mut self) {
        let net = &self.w.net;
        let (frame, rad) = (net.frame(), std::f64::consts::PI / 180.0);
        let mut mask: Box<dyn WaterMask> = Box::new(NoWater);
        for r in &self.rasters {
            let m: Box<dyn WaterMask> = if frame.kind == FrameKind::Equirect {
                Box::new(TransformedMask {
                    mask: r.mask.clone(),
                    offset: [EARTH_R * (r.lon - frame.lon0) * rad * (frame.lat0 * rad).cos(), EARTH_R * (r.lat - frame.lat0) * rad],
                    scale: [(frame.lat0 * rad).cos() / (r.lat * rad).cos(), 1.0],
                })
            } else {
                Box::new(GeoMask::new(r.mask.clone(), frame, r.lon, r.lat))
            };
            mask = Box::new(UnionMask(mask, m));
        }
        self.w.net.set_water_mask(mask);
        self.w.reload();
    }
}

#[cfg(test)]
impl TrackApi {
    /// Detailed refusal text for native test failures; absent from the production WASM API.
    fn issues(&self) -> String {
        self.issues.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n")
    }
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
    fn draft_line_copy_symbol_save_and_undo() {
        let mut a = api();
        let id = a.new_line_id();
        let name = a.next_line_name();
        assert_eq!(name, "Line 1");
        assert_eq!(a.set_line_with_trains(id, &[], 0, 0, 0, 30.0, 180.0, 0, 0xd7263d, name), 0);
        assert!(!a.w.net.line_broken[id as usize]);
        assert!(!a.w.svc.lines[id as usize].running);
        assert_eq!(a.lines_info()[5..8], [0.0; 3]);
        assert_eq!(a.set_line_badge(id, "Blue".into(), 0x123456, "AB".into()), 0);
        let copied = a.new_line_id();
        assert_eq!(a.copy_line(id, copied, "Blue copy".into()), 0);
        assert_eq!(a.line_symbol(copied), "AB");
        assert_eq!(a.set_line_look(copied, "Branch".into(), 0xffffff), 0);
        assert_eq!(a.line_name(id), "Blue");
        assert_eq!(a.line_symbol(copied), "AB");
        assert_eq!(a.undo(), 0);
        assert_eq!(a.line_name(copied), "Blue copy");
        assert_eq!(a.undo(), 0);
        assert!(!a.w.net.line_ok(copied));
        assert_eq!(a.redo(), 0);
        assert!(a.w.net.line_ok(copied));
        let bytes = a.save();
        let saved = save::decode(&bytes).unwrap();
        assert_eq!(saved.line_symbol, ["AB", "AB"]);
        assert_eq!(saved.next_line_number, 3);
        assert!(saved.line_stops(0).is_empty());
        // Core TWT4 without extension is still accepted, custom symbols default to initials.
        let extension = bytes.windows(4).rposition(|w| w == b"LUI1").unwrap();
        let old = save::decode(&bytes[..extension]).unwrap();
        assert_eq!(old.line_symbol, ["", ""]);
        assert_eq!(old.next_line_number, 3);
        assert!(save::decode(&bytes[..bytes.len()-1]).is_err(), "truncated known extension refused");
        assert_eq!(a.remove_line(id), 0);
        let reused = a.new_line_id();
        assert_eq!(a.next_line_name(), "Line 3", "creation numbering survives id reuse");
        assert_eq!(a.set_line_with_trains(reused, &[], 0, 0, 0, 30.0, 180.0, 0, 0, "Line 4".into()), 0);
        assert_eq!(a.next_line_name(), "Line 5", "taken seeded names skipped");
    }

    #[test]
    fn insert_stop_infill_ends_and_reduce_to_empty() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        for x in [1000.0, 2000.0, 4000.0, 5000.0] {
            let edge = (0..a.w.net.edge_count() as u32).find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            }).unwrap();
            assert_eq!(a.add_station(2, edge, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
        }
        let station = |a: &TrackApi, x: f64| (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_platform[n as usize] > 0 && a.w.net.node_x[n as usize] == x).unwrap();
        let (left, first, second, right) = (station(&a, 1000.0), station(&a, 2000.0), station(&a, 4000.0), station(&a, 5000.0));
        let id = a.new_line_id();
        assert_eq!(a.set_line_with_trains(id, &[first, right], 0, 0, 0, 30.0, 180.0, 0, 0, "Test".into()), 0);
        assert_eq!(a.insert_stop(id, second), [first as f64, second as f64, right as f64]);
        assert_eq!(a.insert_stop(id, left), [left as f64, first as f64, right as f64]);
        assert_eq!(a.set_stops(id, &[first, second]), 0);
        assert_eq!(a.insert_stop(id, right), [first as f64, second as f64, right as f64]);
        assert!(a.insert_stop(id, first).is_empty());
        assert!(a.insert_stop(id, u32::MAX).is_empty());
        assert_eq!(a.construct_all(), 0);
        assert_eq!(a.set_train_counts(id, 2, 1, 0), 0);
        let copy = a.new_line_id();
        assert_eq!(a.copy_line(id, copy, "Copy".into()), 0);
        assert_eq!(a.w.net.line_data(copy).unwrap().trains, Some([2, 1, 0]));
        assert_eq!(a.set_stops(id, &[first]), 0);
        assert_eq!(a.line_stops(copy), [first, second], "copy stops independent");
        assert!(!a.w.net.line_broken[id as usize]);
        assert!(!a.w.svc.lines[id as usize].running);
        assert_eq!(a.set_stops(id, &[]), 0);
        assert_eq!(a.insert_stop(id, left), [left as f64]);
        assert_eq!(a.undo(), 0);
        assert_eq!(a.line_stops(id), [first]);
        assert_eq!(a.redo(), 0);
        assert!(a.line_stops(id).is_empty());
        let loaded = super::super::world::TrackWorld::new(save::decode(&a.save()).unwrap());
        assert!(!loaded.svc.lines[0].running);
        assert!(loaded.net.line_stops(0).is_empty());
    }

    #[test]
    fn preview_render_speeds_match_committed_geometry() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 500.0).iter()).copied().collect();
        let pis = [500.0, 0.0, 0.0, 0.0];
        let preview = a.preview_route(&ends, &pis, 2, false);
        let speed = a.preview_render_speeds();
        let p0 = 5 + preview[3] as usize * 4;
        assert_eq!(speed.len(), (preview.len() - p0) / 3 - 1);
        assert!(speed.iter().any(|&s| s < V_TOP as f32 * 3.6 - 1.0));
        assert!(speed.iter().any(|&s| (s - V_TOP as f32 * 3.6).abs() < 0.01));
        assert!(a.edges_info().is_empty(), "preview cannot create track");
        assert_eq!(a.add_route(&ends, &pis, 2, false), 0);
        let edge = a.edges_info()[0] as u32;
        assert_eq!(speed, a.edge_render_speeds(edge));
        assert_eq!(&preview[p0..], a.edge_render(edge).as_slice());
        let edge_preview = a.preview_edge(edge, &pis);
        let ep0 = 5 + edge_preview[3] as usize * 4;
        assert_eq!(a.preview_render_speeds().len(), (edge_preview.len()-ep0)/3-1);
        a.preview_edge(u32::MAX, &[]);
        assert!(a.preview_render_speeds().is_empty(), "refusal cannot reuse old preview speeds");
    }

    #[test]
    fn render_segment_speeds_and_whole_line_fleet_charge() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 500.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[500.0, 0.0, 0.0, 0.0], 2, false), 0, "{}", a.issues());
        let edge = a.edges_info()[0] as u32;
        let pts = a.edge_render(edge);
        let speeds = a.edge_render_speeds(edge);
        assert_eq!(speeds.len(), pts.len() / 3 - 1);
        assert!(speeds.iter().any(|&s| s < V_TOP as f32 * 3.6 - 1.0), "curve speed present");
        assert!(speeds.iter().any(|&s| (s - V_TOP as f32 * 3.6).abs() < 0.01), "straight speed present");
        let nodes = a.nodes_info();
        let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
        assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
        assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
        let id = a.new_line_id();
        assert_eq!(a.set_line_with_trains(id, &[n0, n1], 8, 4, 2, 30.0, 180.0, 0, 0xd7263d, "Line".into()), 0);
        assert_eq!(a.w.svc.lines[id as usize].trains, [8, 4, 2]);
        let before = a.save();
        let (cash_before, undo_before) = (a.cash(), a.can_undo());
        let quoted: f64 = a.blueprint_cost_items().chunks(7).map(|r| r[6]).sum();
        let repeated: f64 = a.blueprint_cost_items().chunks(7).map(|r| r[6]).sum();
        assert_eq!(quoted, repeated);
        assert_eq!(a.save(), before, "quoting leaves the count-scheduled save unchanged");
        assert_eq!((a.cash(), a.can_undo()), (cash_before, undo_before));
        assert_eq!(a.construct_all(), 0);
        assert!((a.last_charge() - quoted).abs() < 0.000001);
        let cars = a.w.svc.lines[id as usize].cars;
        assert_eq!(a.fleet(), 8 * cars);
        let cash = a.cash();
        assert_eq!(a.set_train_counts(id, 9, 4, 0), 0);
        assert!((cash - a.cash() - cars as f64 * CAR_PRICE).abs() < 0.000001);
        assert_eq!(a.w.svc.lines[id as usize].trains, [9, 4, 0]);
        let snapshot = a.save();
        let mut b = api();
        assert!(b.load(&snapshot));
        assert_eq!(b.w.svc.lines[0].trains, [9, 4, 0]);
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
        let quote = a.blueprint_cost_items();
        let total: f64 = quote.chunks_exact(7).map(|r| r[6]).sum();
        assert!((total - (a.blueprint_cost() + trains)).abs() < 1e-8);
        assert_eq!(quote.chunks_exact(7).filter(|r| r[0] == 1.0).map(|r| r[5]).sum::<f64>(), 3.0);
        assert_eq!(a.construct_line(id), 0, "{}", a.issues());
        assert!((total - a.last_charge()).abs() < 1e-8, "quote must include the actual fleet charge");
        assert!(a.blueprint_cost_items().is_empty());
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
    fn blueprint_quote_shared_lines_matches_charge_without_changing_game() {
        let mut a = api();
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0);
        let n0 = a.w.net.edge_a[0];
        let n1 = a.w.net.edge_b[0];
        for n in [n0, n1] { assert_eq!(a.add_station(1, n, 0.0, 0.0, 200, format!("S{n}")), 0); }
        for i in 0..2 {
            let id = a.new_line_id();
            assert_eq!(a.set_line(id, &[n0, n1], 20.0, 10.0, 4.0, 30.0, 180.0, 0, 0, format!("L{i}")), 0);
        }
        let saved = save::encode(&a.w.net);
        let cash = a.cash();
        let quote = a.blueprint_cost_items();
        assert_eq!(quote, a.blueprint_cost_items());
        assert_eq!(saved, save::encode(&a.w.net));
        assert_eq!(cash, a.cash());
        assert!(a.can_undo());
        let total: f64 = quote.chunks_exact(7).map(|r| r[6]).sum();
        assert_eq!(a.construct_all(), 0);
        assert!((total - a.last_charge()).abs() < 1e-8, "quote {total}, charge {}", a.last_charge());
        assert!(a.blueprint_cost_items().is_empty());
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
        assert_eq!(&bytes[..4], b"TWT4");
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

    /// T-237: the stop markers' tiers come from the line tool's own search: a station past either
    /// end over track the line can carry on over is reachable as it is (the path only grows), a
    /// branch past an end too, a station on a bypass changes the path, and a branch the line would
    /// have to turn back to reach or a station on other track cannot be added.
    #[test]
    fn stop_tiers_follow_the_line_tool() {
        let mut a = api();
        let route = |a: &mut TrackApi, ends: [f64; 10], pis: &[f64]| assert_eq!(a.add_route(&ends, pis, 2, false), 0, "{}", a.issues());
        route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9000.0, 0.0, 0.0], &[]);
        let on_main = |a: &TrackApi, x: f64| (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.project(e, x, 0.0).1 < 0.01 && a.w.net.node_y[a.w.net.edge_a[e as usize] as usize] == 0.0 && a.w.net.node_y[a.w.net.edge_b[e as usize] as usize] == 0.0).unwrap() as f64;
        // a bypass from 2 km to 4 km, a branch from 6 km past the end, one from 4.5 km facing back
        let e1 = on_main(&a, 2000.0);
        let ends = [2.0, e1, 2000.0, 0.0, 0.0, 2.0, e1, 4000.0, 0.0, 0.0];
        route(&mut a, ends, &[3000.0, 800.0, 0.0, 0.0]);
        let e = on_main(&a, 6000.0);
        route(&mut a, [2.0, e, 6000.0, 0.0, 0.0, 0.0, 0.0, 7500.0, 1500.0, 0.0], &[]);
        let e = on_main(&a, 4500.0);
        route(&mut a, [2.0, e, 4500.0, 0.0, 0.0, 0.0, 0.0, 3500.0, -1500.0, 0.0], &[]);
        // other track, not joined
        route(&mut a, [0.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 7000.0, 5000.0, 0.0], &[]);
        let nearest = |a: &TrackApi, x: f64, y: f64| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
        let station = |a: &mut TrackApi, x: f64, y: f64, name: &str| -> u32 {
            let e = nearest(a, x, y);
            assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{name}: {}", a.issues());
            let n = a.w.net.node_count() as u32;
            (0..n).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0).min_by(|&p, &q| {
                let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y);
                d(p).total_cmp(&d(q))
            }).unwrap()
        };
        let first = station(&mut a, 1000.0, 0.0, "A");
        let last = station(&mut a, 5000.0, 0.0, "B");
        let between = station(&mut a, 3000.0, 0.0, "M");
        let past_end = station(&mut a, 8000.0, 0.0, "C");
        let before_start = station(&mut a, 300.0, 0.0, "D");
        let bypass = station(&mut a, 3000.0, 650.0, "S");
        let branch = station(&mut a, 7200.0, 1300.0, "Y");
        let facing_back = station(&mut a, 3700.0, -1300.0, "Z");
        let other = station(&mut a, 6000.0, 5000.0, "E");
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[first, last], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        let tiers: std::collections::HashMap<u32, f64> = a.stop_tiers(id).chunks(2).map(|c| (c[0] as u32, c[1])).collect();
        assert_eq!(tiers.len(), 9);
        let want = [(first, 0.0), (last, 0.0), (between, 1.0), (past_end, 1.0), (before_start, 1.0), (bypass, 2.0), (branch, 1.0), (facing_back, 3.0), (other, 3.0)];
        for (n, t) in want {
            assert_eq!(tiers[&n], t, "station {}", a.w.net.node_name[n as usize]);
        }
        // and a click does what the marker says
        let path = a.line_path(id);
        for (n, t) in want {
            let stops: Vec<u32> = a.insert_stop(id, n).iter().map(|&s| s as u32).collect();
            assert_eq!(stops.is_empty(), t == 0.0 || t == 3.0, "station {}", a.w.net.node_name[n as usize]);
            if stops.is_empty() {
                continue;
            }
            assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
            let grown = a.line_path(id);
            assert_eq!(grown.windows(path.len()).any(|w| w == path), t == 1.0, "station {}", a.w.net.node_name[n as usize]);
            assert_eq!(a.undo(), 0);
        }
        // with no route yet, every station joined to the one stop counts as on it
        let lone = a.new_line_id();
        assert_eq!(a.set_line(lone, &[first], 0.0, 0.0, 0.0, 30.0, 180.0, 0, 0, "One".into()), 0, "{}", a.issues());
        let tiers: std::collections::HashMap<u32, f64> = a.stop_tiers(lone).chunks(2).map(|c| (c[0] as u32, c[1])).collect();
        assert_eq!((tiers[&past_end], tiers[&bypass], tiers[&other]), (1.0, 1.0, 3.0));
    }

    /// T-244: the corners of a stroke, and the same stroke as one edge (set directly).
    const CLICKS: [(f64, f64); 4] = [(1500.0, 300.0), (2600.0, -400.0), (4800.0, 1200.0), (6200.0, 900.0)];
    fn stroke(level_of: &dyn Fn(usize) -> i8) -> (Vec<f64>, Vec<f64>) {
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(7600.0, 2400.0).iter()).copied().collect();
        let pis = CLICKS.iter().enumerate().flat_map(|(k, &(x, y))| [x, y, 0.0, level_of(k) as f64]).collect();
        (ends, pis)
    }
    fn one_edge(pis: &[f64]) -> (TrackApi, u32) {
        let mut b = api();
        let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
        assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false), 0);
        assert_eq!(b.set_node(n1, 7600.0, 2400.0, 0, false), 0);
        assert_eq!(b.set_edge(e, n0, n1, 2, pis), 0, "{}", b.issues());
        (b, e)
    }
    fn live_edges(a: &TrackApi) -> Vec<u32> {
        (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
    }
    /// Every drawn point of `a`'s track lies on `b`'s edge `e`, and their lengths agree.
    fn same_track(a: &TrackApi, b: &TrackApi, e: u32, tol: f64) {
        let mut len = 0.0;
        for p in live_edges(a) {
            len += a.w.net.edge_len[p as usize];
            for q in a.edge_render(p).chunks(3) {
                let off = b.w.net.project(e, q[0], q[1]).1;
                assert!(off < tol, "piece {p}: {off} m off the stroke at {q:?}");
            }
        }
        assert!((len - b.w.net.edge_len[e as usize]).abs() < tol, "{len} against {}", b.w.net.edge_len[e as usize]);
    }

    #[test]
    fn a_stroke_is_one_piece_per_corner() {
        let mut a = api();
        let (ends, pis) = stroke(&|_| 0);
        let pv = a.preview_route(&ends, &pis, 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert!(live_edges(&a).is_empty(), "a preview builds nothing");
        assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
        let (b, e) = one_edge(&pis);
        // four corners, four pieces, joined at three plain nodes, each corner in a piece of its own
        let edges = live_edges(&a);
        assert_eq!(edges.len(), 4);
        let joints: Vec<_> = a.nodes_info().chunks(10).filter(|c| c[4] == 2.0).map(|c| c[0] as u32).collect();
        assert_eq!(joints.len(), 3);
        for &p in &edges {
            let corners = a.edge_pis(p);
            assert_eq!(corners.len(), 9, "one corner");
            assert_eq!(corners[3], 0.0, "auto radius");
        }
        let whole = b.edge_pis(e);
        let used: Vec<f64> = edges.iter().map(|&p| a.edge_pis(p)[5]).collect();
        for k in 0..4 {
            assert!((used[k] - whole[5 + 8 * k]).abs() < 0.01, "corner {k}: {} against {}", used[k], whole[5 + 8 * k]);
        }
        // the very same track, price and preview
        same_track(&a, &b, e, 0.005);
        assert!((a.total_cost() - b.total_cost()).abs() < 1e-6, "{} against {}", a.total_cost(), b.total_cost());
        assert!((pv[1] - b.total_cost()).abs() < 1e-6);
        // one undo takes the whole stroke away, one redo brings it back
        assert_eq!(a.undo(), 0);
        assert!(live_edges(&a).is_empty() && a.nodes_info().is_empty());
        assert_eq!(a.redo(), 0);
        assert_eq!(live_edges(&a), edges);
        // saves keep the pieces
        let mut c = api();
        assert!(c.load(&a.save()));
        assert_eq!(c.edges_info(), a.edges_info());
        // removing a piece in the middle leaves its neighbours, ending where it was
        assert_eq!(a.delete_edge(edges[1]), 0, "{}", a.issues());
        assert_eq!(live_edges(&a).len(), 3);
        for &j in &joints[..2] {
            assert_eq!(a.nodes_info().chunks(10).find(|c| c[0] as u32 == j).unwrap()[4], 1.0, "a track end now");
        }
        // a leg that changes level is not cut: its ramp is centred between its corners
        let mut d = api();
        let (ends, pis) = stroke(&|k| if k >= 2 { 1 } else { 0 });
        assert_eq!(d.add_route(&ends, &pis, 2, false), 0, "{}", d.issues());
        assert_eq!(live_edges(&d).len(), 3);
        let (_, pis_flat) = stroke(&|k| if k >= 2 { 1 } else { 0 });
        let mut e2 = api();
        let (n0, n1, x) = (e2.new_node_id(), e2.new_node_id(), e2.new_edge_id());
        assert_eq!(e2.set_node(n0, 0.0, 0.0, 0, false), 0);
        assert_eq!(e2.set_node(n1, 7600.0, 2400.0, 0, false), 0);
        assert_eq!(e2.set_edge(x, n0, n1, 2, &pis_flat), 0, "{}", e2.issues());
        same_track(&d, &e2, x, 0.005);
        assert!((d.total_cost() - e2.total_cost()).abs() < 1e-6);
    }

    /// T-244: a line over a stroke in pieces runs exactly as over the stroke in one edge: the
    /// joints add no speed limit, no delay and no cost.
    #[test]
    fn pieces_run_like_the_whole_stroke() {
        let (ends, pis) = stroke(&|_| 0);
        let mut a = api();
        assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
        let (mut b, _) = one_edge(&pis);
        let mut times = vec![];
        for t in [&mut a, &mut b] {
            let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
            let (w, e) = (end(t, 0.0), end(t, 7600.0));
            assert_eq!(t.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
            assert_eq!(t.add_station(1, e, 0.0, 0.0, 200, "East".into()), 0);
            let id = t.new_line_id();
            assert_eq!(t.set_line_with_trains(id, &[w, e], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", t.issues());
            let quote = t.blueprint_cost();
            assert_eq!(t.construct_all(), 0, "{}", t.issues());
            times.push((t.stop_times(id, 0, 0), t.stop_times(id, 1, 0), t.line_summary(id), t.lines_info()[9], quote, t.markers(0).len()));
        }
        let (p, q) = (&times[0], &times[1]);
        for k in 0..2 {
            let (x, y) = if k == 0 { (&p.0, &q.0) } else { (&p.1, &q.1) };
            assert_eq!(x.len(), y.len());
            for (s, t) in x.iter().zip(y) {
                assert!((s - t).abs() < 1e-3, "run {k}: {s} s against {t} s");
            }
        }
        for (s, t) in p.2.iter().zip(&q.2) {
            assert!((s - t).abs() < 1e-3, "summary {:?} against {:?}", p.2, q.2);
        }
        assert!((p.3 - q.3).abs() < 0.01, "length {} against {}", p.3, q.3);
        assert!((p.4 - q.4).abs() < 1e-6, "price {} against {}", p.4, q.4);
        assert_eq!(p.5, q.5, "capacity markers");
        // one track section for the line, as over one edge
        let sections = |t: &TrackApi| t.w.svc.cap.res.iter().filter(|r| r.kind == super::super::capacity::ResKind::Section).count();
        assert_eq!(sections(&a), sections(&b));
    }

    /// T-258: a station can go anywhere along a curve (partway round, at its ends), on track drawn
    /// from an existing track end (a lead-in curve, then the corner's curve) and on plain track,
    /// and the split leaves the track, its run time and its price as they were.
    #[test]
    fn stations_go_anywhere_on_a_curve() {
        for (from_track, corner, radius) in [(true, (1500.0, 300.0), 0.0), (false, (1500.0, 300.0), 0.0), (true, (900.0, 700.0), 0.0), (false, (700.0, -900.0), 0.0), (true, (1500.0, 300.0), 150.0), (false, (1500.0, 1200.0), 400.0), (true, (1200.0, -500.0), 120.0)] {
            let mut a = api();
            let mut start = [0.0, 0.0, 0.0, 0.0, 0.0];
            if from_track {
                let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
                assert_eq!(a.add_route(&lead, &[], 2, false), 0);
                let n = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
                start = [1.0, n, 0.0, 0.0, 0.0];
            }
            let ends: Vec<f64> = start.iter().chain(free(3000.0, -600.0).iter()).copied().collect();
            assert_eq!(a.add_route(&ends, &[corner.0, corner.1, radius, 0.0], 2, false), 0, "{}", a.issues());
            let e = *live_edges(&a).last().unwrap();
            let len = a.w.net.edge_len[e as usize];
            let pieces: Vec<_> = a.w.net.edge_pieces(e).to_vec();
            let (cost, before) = (a.total_cost(), a.edge_render(e));
            let mut tried = 0;
            let mut failed = vec![];
            for p in pieces.iter().filter(|p| p.k != 0.0) {
                for f in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
                    let s = (p.s0 + f * p.len).clamp(110.0, len - 110.0);
                    let (x, y, _) = geom::pos_at(&pieces, s);
                    tried += 1;
                    if a.add_station(2, e, x, y, 200, "S".into()) != 0 {
                        failed.push((s, a.issues(), format!("{:?}", a.w.net.split_plan(e, a.w.net.project(e, x, y).0)), a.w.net.edge_data(e)));
                        continue;
                    }
                    // the same track and price, and an undo puts it back exactly
                    for q in before.chunks(3) {
                        let off = live_edges(&a).iter().map(|&k| a.w.net.project(k, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
                        assert!(off < 0.01, "split at {s}: {off} m off");
                    }
                    let station = 10.0 * (0.4 + 0.6 * 200.0 / 200.0) * 1.0;
                    assert!((a.total_cost() - cost - station).abs() < 0.05, "split at {s}: {} against {}", a.total_cost() - station, cost);
                    assert_eq!(a.undo(), 0);
                    assert_eq!(a.edge_render(e), before);
                }
            }
            assert!(tried > 8);
            assert!(failed.is_empty(), "{from_track} {corner:?} {radius}: {failed:?}");
        }
    }

    /// T-244: a station placed next to a joint joins the two pieces again first, so the platform
    /// has room; the track and the lines over it are as they were.
    #[test]
    fn a_station_next_to_a_joint_joins_the_pieces() {
        let (ends, pis) = stroke(&|_| 0);
        let mut a = api();
        assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
        assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
        assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[w, east], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        let t0 = a.stop_times(id, 0, 0);
        let edges = live_edges(&a);
        // 30 m before the second joint, on the second piece
        let j = a.w.net.edge_b[edges[1] as usize];
        let len = a.w.net.edge_len[edges[1] as usize];
        let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(edges[1]), len - 30.0);
        assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
        assert!(!a.w.net.node_ok(j), "the joint went");
        assert_eq!(live_edges(&a).len(), 4, "two pieces joined, then split at the station");
        let (b, e) = one_edge(&pis);
        same_track(&a, &b, e, 0.005);
        assert!(!a.w.net.line_broken[id as usize]);
        let t1 = a.stop_times(id, 0, 0);
        assert!((t0.last().unwrap() - t1.last().unwrap()).abs() < 1e-3, "{t0:?} {t1:?}");
        assert_eq!(a.undo(), 0);
        assert_eq!(live_edges(&a), edges);
        assert!(a.w.net.node_ok(j));
        // after reversing the line's path: still fine
        let id2 = a.new_line_id();
        assert_eq!(a.set_line(id2, &[east, w], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "R".into()), 0);
        assert_eq!(a.add_station(2, edges[1], x, y, 200, "Middle".into()), 0, "{}", a.issues());
        assert!(!a.w.net.line_broken[id2 as usize] && !a.w.net.line_broken[id as usize]);
    }

    /// T-255: quad track is the drawn double track and an express double track beside it, end to
    /// end; a station on one pair gets its twin on the other; an express copy of a line runs on
    /// the express pair.
    #[test]
    fn quad_track_and_express_copy() {
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(12000.0, 0.0).iter()).copied().collect();
        let pis = vec![3000.0, 400.0, 0.0, 0.0, 6000.0, 0.0, 0.0, 0.0, 9000.0, 400.0, 0.0, 0.0];
        let mut a = api();
        // continuing an existing track end, so the start becomes a junction
        let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&lead, &[], 2, false), 0);
        let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
        let mut quad_ends = ends.clone();
        quad_ends[..5].copy_from_slice(&[1.0, start, 0.0, 0.0, 0.0]);
        let pv = a.preview_route(&quad_ends, &pis, 4, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
        let before = a.total_cost();
        assert_eq!(a.add_route(&quad_ends, &pis, 4, false), 0, "{}", a.issues());
        let mut b = api();
        let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
        assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false) + b.set_node(n1, 12000.0, 0.0, 0, false) + b.set_edge(e, n0, n1, 2, &pis), 0);
        let on_local = |a: &TrackApi, p: u32| { let r = a.edge_render(p); let k = r.len() / 6 * 3; b.w.net.project(e, r[k], r[k + 1]).1 < 2.0 };
        let new: Vec<u32> = live_edges(&a).into_iter().filter(|&p| p != 0).collect();
        let (local, express) = (new[..3].to_vec(), new[3..].to_vec());
        assert!(local[1..].iter().all(|&p| on_local(&a, p)) && express.iter().all(|&p| !on_local(&a, p)));
        assert!(express.len() >= 3, "{express:?}");
        // beside it, 8 m to the left, away from the tapers at the ends
        let len = b.w.net.edge_len[e as usize];
        for &p in &express {
            for q in a.edge_render(p).chunks(3) {
                let (s, _) = b.w.net.project(e, q[0], q[1]);
                let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
                if s > QUAD_TAPER + 300.0 && s < len - QUAD_TAPER - 50.0 {
                    assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off at {s}");
                }
            }
        }
        // two double tracks' price, and a junction where it leaves the old track
        let added = a.total_cost() - before;
        assert!((added / b.total_cost() - 2.0).abs() < 0.1, "{added} against {}", b.total_cost());
        let j = a.nodes_info().chunks(10).find(|c| c[0] == start).unwrap()[4];
        assert_eq!(j, 3.0, "the old track end is a junction now");
        // stations: each gets its twin on the other pair, same name
        let mut stops = vec![];
        for (piece, at, name) in [(local[1], 60.0, "First"), (local[2], 400.0, "Last")] {
            let p = piece; let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(p), at);
            assert_eq!(a.add_station(2, p, x, y, 200, name.into()), 0, "{} on {p} at {:?}: {:?} {:?}", a.issues(), a.w.net.project(p, x, y), a.w.net.edge_data(p), a.edge_pis(p));
            let named: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).collect();
            assert_eq!(named.len(), 2, "{name} and its twin");
            let local_one = *named.iter().min_by(|&&p, &&q| { let d = |k: u32| b.w.net.project(e, a.w.net.node_x[k as usize], a.w.net.node_y[k as usize]).1; d(p).total_cmp(&d(q)) }).unwrap();
            stops.push(local_one);
        }
        // a station on a joint node of the drawn pair gets its twin too
        let joint = (0..a.w.net.node_count() as u32).find(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 2 && a.w.net.node_platform[k as usize] == 0 && b.w.net.project(e, a.w.net.node_x[k as usize], a.w.net.node_y[k as usize]).1 < 0.5 && (1000.0..11000.0).contains(&a.w.net.node_x[k as usize])).unwrap();
        assert_eq!(a.add_station(1, joint, 0.0, 0.0, 200, "Joint".into()), 0, "{}", a.issues());
        assert_eq!((0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == "Joint").count(), 2);
        assert_eq!(a.undo(), 0);
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &stops, 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "Local".into()), 0, "{}", a.issues());
        let ex = a.new_line_id();
        assert_eq!(a.express_copy(id, ex, "Local express".into()), 0, "{}", a.issues());
        let es = a.line_stops(ex);
        assert_eq!(es.len(), 2);
        assert!(es.iter().zip(&stops).all(|(p, q)| p != q && a.w.net.node_name[*p as usize] == a.w.net.node_name[*q as usize]), "the twins");
        let on_express = a.line_path(ex).iter().filter(|&&p| !on_local(&a, p >> 1)).count();
        assert!(on_express == a.line_path(ex).len() && on_express >= 2, "{:?}", a.line_path(ex));
        assert!(a.line_path(id).iter().all(|&p| on_local(&a, p >> 1)));
        assert!(!a.w.net.line_broken[ex as usize]);
        assert!(a.stop_times(ex, 0, 0).last().unwrap() > &60.0);
        assert_eq!(a.undo(), 0);
        assert!(!a.w.net.line_ok(ex));
        // deleting an express piece leaves the drawn pair
        assert_eq!(a.delete_edge(express[express.len() / 2]), 0, "{}", a.issues());
        assert!(!a.w.net.line_broken[id as usize]);
    }

    /// T-244: dragging a corner of a piece moves the joints next to it along, so the stroke
    /// reshapes as if it were still one edge.
    #[test]
    fn reshaping_a_piece_moves_its_joints() {
        let (ends, pis) = stroke(&|_| 0);
        let mut a = api();
        assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
        let edges = live_edges(&a);
        let mid = edges[1];
        let moved = [2700.0, -900.0, 0.0, 0.0];
        let pv = a.preview_edge(mid, &moved);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
        assert_eq!(a.set_edge_pis(mid, &moved), 0, "{}", a.issues());
        let mut whole = pis.clone();
        whole[4..8].copy_from_slice(&moved);
        let (b, e) = one_edge(&whole);
        same_track(&a, &b, e, 0.01);
        assert_eq!(live_edges(&a), edges, "the same pieces");
        assert_eq!(a.undo(), 0);
        let (b, e) = one_edge(&pis);
        same_track(&a, &b, e, 0.005);
        // a joint drags like any blueprint node, keeping its heading
        let j = a.w.net.edge_b[edges[0] as usize];
        let (x, y) = (a.w.net.node_x[j as usize], a.w.net.node_y[j as usize]);
        assert_eq!(a.move_node(j, x + 40.0, y + 60.0), 0, "{}", a.issues());
        // a line still routes over the pieces through the joints
        let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
        let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
        assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
        assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[w, east], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        assert_eq!(a.line_path(id).len(), 4);
    }
}
#[cfg(test)]
mod regional_water_tests {
    use super::*;
    fn api() -> TrackApi { TrackApi::new(-73.985, 40.758, true) }
    #[cfg(feature = "demand")]
    fn free(x: f64, y: f64) -> [f64; 5] { [0.0, 0.0, x, y, 0.0] }
    #[test]
    fn pack_water_follows_the_frame_of_the_loaded_save() {
        // A pack at Philadelphia with one water cell of 100 m at its origin, in the Northeast.
        let mut a = api();
        let legacy_empty = a.save();
        a.clear_water();
        a.add_pack_water(100.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1], -75.1636, 39.9526);
        let at = |a: &TrackApi, lon: f64, lat: f64| {
            let p = a.from_lonlat(lon, lat);
            a.w.net.water_mask.is_water(p[0], p[1])
        };
        // legacy frame: the Philadelphia cell is water, a point 1 km east is not
        assert_eq!(a.frame_kind(), 0);
        assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
        assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
        // a world-frame network: the same ground is water, through the other chart
        assert!(a.set_empty_frame(true));
        assert_eq!(a.frame_kind(), 1);
        assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
        assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
        let world_empty = a.save();
        assert_eq!(&world_empty[..4], b"TWT5");
        assert_eq!(&legacy_empty[..4], b"TWT4");
        // loading each kind of save rebuilds the mask for its frame
        assert!(a.load(&legacy_empty));
        assert_eq!(a.frame_kind(), 0);
        assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
        assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
        assert!(a.load(&world_empty));
        assert_eq!(a.frame_kind(), 1);
        assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
        assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
        // a frame cannot be switched under drawn track
        assert!(a.set_empty_frame(false));
        let n = a.new_node_id();
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 2, false), 0);
        let _ = n;
        assert!(!a.set_empty_frame(true));
        assert_eq!(TrackApi::convert_save_frame(&a.save(), true)[..4], *b"TWT5");
        assert!(TrackApi::convert_save_frame(b"junk", true).is_empty());
    }

    #[test]
    fn regional_water_union_survives_legacy_network_load() {
        let mut a = api();
        let legacy = a.save();
        a.set_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1]);
        a.append_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1], -100000.0, -90000.0, 0.98, 1.0);
        assert!(a.w.net.water_mask.is_water(12.0, 12.0));
        assert!(a.w.net.water_mask.is_water(-99988.0, -89988.0));
        assert!(!a.w.net.water_mask.is_water(-50000.0, -45000.0));
        assert!(a.load(&legacy));
        assert_eq!(a.save(), legacy, "adding region coverage changes no saved inputs");
        assert_eq!(a.w.net.origin_lon, -73.985);
        assert_eq!(a.w.net.origin_lat, 40.758);
        assert!(a.w.net.water_mask.is_water(-99988.0, -89988.0));
        a.set_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1]);
        assert!(!a.w.net.water_mask.is_water(-99988.0, -89988.0), "legacy setter replaces coverage");
        a.clear_water();
        assert!(a.w.net.water_mask.is_empty());
    }

    #[test]
    #[cfg(feature = "demand")]
    #[ignore = "requires the generated NYC and Philadelphia city packs"]
    fn northeast_pack_water_crossings() {
        // The same crossings, in the legacy Northeast frame and in the world frame (T-125).
        for world_frame in [false, true] {
            let mut a = api();
            assert!(a.set_empty_frame(world_frame));
            for city in ["nyc", "phl"] {
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/packs");
                let head: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{city}.json"))).unwrap()).unwrap();
                let bin = std::fs::read(dir.join(format!("{city}.bin"))).unwrap();
                let array = |name: &str, width: usize| {
                    let item = head["arrays"].as_array().unwrap().iter().find(|v| v["name"] == name).unwrap();
                    let start = item["offset"].as_u64().unwrap() as usize;
                    &bin[start..start + item["count"].as_u64().unwrap() as usize * width]
                };
                let water = &head["water"];
                let m = RunMask::from_pack(water["cell_m"].as_f64().unwrap(),
                    [water["origin_m"][0].as_f64().unwrap(), water["origin_m"][1].as_f64().unwrap()],
                    [water["size"][0].as_u64().unwrap() as u32, water["size"][1].as_u64().unwrap() as u32],
                    array("water_row", 4), array("water_x", 2));
                let lon = head["origin"]["lon"].as_f64().unwrap();
                let lat = head["origin"]["lat"].as_f64().unwrap();
                a.add_pack_water(m.cell, m.x0, m.y0, m.w, m.h, &m.row, &m.xs, lon, lat);
            }
            let world = |a: &TrackApi, lon: f64, lat: f64| { let p = a.from_lonlat(lon, lat); [p[0], p[1]] };
            for (name, lon, lat, water) in [
                ("Hudson", -74.018, 40.742, true), ("Times Square", -73.985, 40.758, false),
                ("Delaware at Philadelphia", -75.135, 39.950, true), ("City Hall", -75.1636, 39.9526, false),
                ("Delaware at Trenton", -74.7755, 40.2169, true), ("Trenton station", -74.754, 40.219, false),
            ] {
                let [x, y] = world(&a, lon, lat);
                assert_eq!(a.w.net.water_mask.is_water(x, y), water, "{name} (world frame {world_frame})");
            }
            let [x, y] = world(&a, -74.7755, 40.2169);
            let ends: Vec<f64> = free(x - 200.0, y).into_iter().chain(free(x + 200.0, y)).collect();
            assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 0.0, "ground-level Delaware crossing refused");
            assert!(a.issue_kinds().contains("GroundOverWater"));
        }
    }
}
