//! The player's network (SPEC 6.1-6.3, notes/T-008.md 4): nodes, edges, stations and lines as
//! flat per-field arrays indexed by stable u32 ids, plus everything derived from them (pieces,
//! heights, water, cost, node ports, crossings, the edge-to-lines index).
//!
//! Inputs change only through the raw setters (`set_node`, `set_edge`, `set_station`,
//! `set_line`, `set_schedule`), which record what they touched; `derive` then rebuilds just those
//! parts. `edit.rs` builds undoable operations on top.

use super::cost::{self, NoWater, WaterMask};
use super::cross::{self, Grid};
use super::geom::{self, GeomIssue, Pi, Piece};
use super::params::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

/// Most edge ends one node can have.
pub const MAX_PORTS: usize = 8;

/// A run of items in a pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub off: u32,
    pub len: u32,
}

/// Variable-length per-object data in one flat vector. Replaced runs become garbage until the
/// owner compacts.
#[derive(Clone, Debug)]
pub struct Pool<T> {
    pub data: Vec<T>,
    pub garbage: usize,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool { data: Vec::new(), garbage: 0 }
    }
}

impl<T: Clone> Pool<T> {
    pub fn get(&self, s: Span) -> &[T] {
        &self.data[s.off as usize..(s.off + s.len) as usize]
    }
    pub fn put(&mut self, items: &[T]) -> Span {
        let off = self.data.len() as u32;
        self.data.extend_from_slice(items);
        Span { off, len: items.len() as u32 }
    }
    pub fn replace(&mut self, old: Span, items: &[T]) -> Span {
        if items.len() <= old.len as usize {
            let o = old.off as usize;
            self.data[o..o + items.len()].clone_from_slice(items);
            self.garbage += old.len as usize - items.len();
            Span { off: old.off, len: items.len() as u32 }
        } else {
            self.garbage += old.len as usize;
            self.put(items)
        }
    }
    pub fn free(&mut self, s: Span) {
        self.garbage += s.len as usize;
    }
    fn needs_compact(&self) -> bool {
        self.garbage > 4096 && self.garbage * 2 > self.data.len()
    }
    /// Copy every live run into a fresh vector, in owner order.
    fn compact(&mut self, spans: &mut [Span]) {
        let mut d = Vec::with_capacity(self.data.len() - self.garbage);
        for s in spans.iter_mut() {
            let off = d.len() as u32;
            d.extend_from_slice(&self.data[s.off as usize..(s.off + s.len) as usize]);
            s.off = off;
        }
        self.data = d;
        self.garbage = 0;
    }
}

/// The edge ends at a node. `side` 1 (B) leaves along the node's heading, 0 (A) against it;
/// `rank` orders ports on one side from right to left looking along the heading.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ports {
    pub n: u8,
    /// `edge << 1 | end` (end 0 = the edge's a, 1 = its b), ascending.
    pub end: [u32; MAX_PORTS],
    pub side: [u8; MAX_PORTS],
    pub rank: [u8; MAX_PORTS],
}

impl Ports {
    pub fn ends(&self) -> &[u32] {
        &self.end[..self.n as usize]
    }
    pub fn find(&self, edge: u32, which: u32) -> Option<usize> {
        self.ends().iter().position(|&e| e == edge << 1 | which)
    }
    fn add(&mut self, code: u32) -> bool {
        if self.n as usize >= MAX_PORTS {
            return false;
        }
        let i = self.ends().partition_point(|&e| e < code);
        let n = self.n as usize;
        self.end.copy_within(i..n, i + 1);
        self.end[i] = code;
        self.n += 1;
        true
    }
    fn remove(&mut self, code: u32) {
        if let Some(i) = self.ends().iter().position(|&e| e == code) {
            let n = self.n as usize;
            self.end.copy_within(i + 1..n, i);
            self.n -= 1;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeData {
    pub x: f64,
    pub y: f64,
    pub level: i8,
    pub flying: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeData {
    pub a: u32,
    pub b: u32,
    /// 2 = double track (default), 1 = single.
    pub tracks: u8,
    /// Interior PIs from a to b.
    pub pis: Vec<Pi>,
    /// Constructed (paid for, can carry trains); false = blueprint (SPEC 6.4).
    pub built: bool,
    /// Kept only for track drawn while the track ran through the clicked points (T-079, saves
    /// from then until T-092): the points clicked, which its PIs were fitted through. Geometry
    /// never reads them. Such an edge keeps its shape until it is reshaped; the first reshape
    /// (a PI drag, a radius, a node drag) makes these points its PIs with auto radius and clears
    /// them (`TrackApi::editable`). Empty for every edge drawn before T-079 or after T-092.
    pub thru: Vec<Pi>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StationData {
    pub platform: u16,
    pub name: String,
    /// Constructed; false = blueprint (SPEC 6.4).
    pub built: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineData {
    pub name: String,
    pub colour: u32,
    /// Station nodes in order.
    pub stops: Vec<u32>,
    /// The exact path from the first stop to the last, `edge << 1 | dir` (dir 0 = a to b).
    pub path: Vec<u32>,
    /// Trains per hour for high, medium and low demand (SPEC 6.3).
    pub tph: [f32; 3],
    pub dwell_s: f32,
    pub turnaround_s: f32,
    /// Cars per train; 0 = as many as the shortest platform allows.
    pub cars: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CrossKind {
    /// Same whole level: a flat crossing, with its delay and cost.
    Flat(i8),
    /// Heights differ by less than a level, or one is on a ramp.
    Bad,
    /// The two alignments run on top of each other.
    Overlap,
}

/// Where two edges cross without being separated by a level. Derived from geometry, never saved.
#[derive(Clone, Copy, Debug)]
pub struct Crossing {
    pub e1: u32,
    pub s1: f64,
    pub e2: u32,
    pub s2: f64,
    pub x: f64,
    pub y: f64,
    pub kind: CrossKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Issue {
    Geom { edge: u32, issue: GeomIssue },
    GroundOverWater { edge: u32, s: f64 },
    Crossing { e1: u32, e2: u32, x: f64, y: f64, overlap: bool },
    NoSuchNode { node: u32 },
    NoSuchEdge { edge: u32 },
    NoSuchLine { line: u32 },
    EdgeLoop { edge: u32 },
    Tracks { edge: u32 },
    TooManyPorts { node: u32 },
    /// An edge end does not leave along the node's heading.
    NodeHeading { node: u32, edge: u32 },
    /// A station's two ends leave on one side (its track does not run through).
    NodeOneSided { node: u32 },
    NodeLevel { node: u32 },
    /// Removing a node that still has edges or a station.
    NodeInUse { node: u32 },
    StationPorts { node: u32 },
    PlatformLength { node: u32 },
    /// The platform reaches another node or another platform.
    PlatformTooLong { node: u32 },
    PlatformNotLevel { node: u32 },
    PlatformCrossing { node: u32 },
    StationGroundOverWater { node: u32 },
    StationInUse { node: u32, line: u32 },
    LineStops { line: u32 },
    LinePath { line: u32 },
    LineSchedule { line: u32 },
    /// A split point on a ramp, too near a node, or outside the edge.
    SplitPoint { edge: u32 },
}

/// What the raw setters touched since the last `take_touch`.
#[derive(Clone, Debug, Default)]
pub struct Touch {
    pub nodes: Vec<u32>,
    pub edges: Vec<u32>,
    pub stations: Vec<u32>,
    pub lines: Vec<u32>,
    pub schedules: Vec<u32>,
    pub paths_changed: bool,
}

/// What `derive` rebuilt.
#[derive(Clone, Debug, Default)]
pub struct Derived {
    pub edges: Vec<u32>,
    pub nodes: Vec<u32>,
    /// Render tiles (zoom 12) of the old and new extent.
    pub tiles: Vec<(u32, u32)>,
    /// Positions of crossings removed or added.
    pub crossings: Vec<[f64; 2]>,
}

pub struct Network {
    /// City origin; local metres are equirectangular from here (as in the city pack).
    pub origin_lon: f64,
    pub origin_lat: f64,
    /// Running side from the city pack: right in the US, left in Japan and the UK.
    pub right_hand: bool,

    pub node_alive: Vec<bool>,
    pub node_x: Vec<f64>,
    pub node_y: Vec<f64>,
    pub node_level: Vec<i8>,
    pub node_flying: Vec<bool>,
    /// Platform length, 0 = not a station.
    pub node_platform: Vec<u16>,
    pub node_name: Vec<String>,
    /// The station on the node is constructed (meaningless without a platform).
    pub node_built: Vec<bool>,
    pub node_ports: Vec<Ports>,
    /// Derived: the track heading through the node.
    pub node_heading: Vec<f64>,
    node_free: Vec<u32>,

    pub edge_alive: Vec<bool>,
    pub edge_a: Vec<u32>,
    pub edge_b: Vec<u32>,
    pub edge_tracks: Vec<u8>,
    /// Constructed; false = blueprint.
    pub edge_built: Vec<bool>,
    pub edge_pis: Vec<Span>,
    /// Per edge: the points a T-079 edge was drawn through (`EdgeData::thru`), else empty.
    pub edge_thru: Vec<Vec<Pi>>,
    pub edge_len: Vec<f64>,
    pub edge_pieces: Vec<Span>,
    pub edge_vert: Vec<Span>,
    pub edge_water: Vec<Span>,
    pub edge_cost: Vec<f64>,
    /// Time to run the edge at its speed limits, no acceleration (for routing).
    pub edge_time: Vec<f64>,
    edge_free: Vec<u32>,
    pub pis: Pool<Pi>,
    pub pieces: Pool<Piece>,
    pub verts: Pool<[f64; 2]>,
    pub water: Pool<[f64; 2]>,

    pub line_alive: Vec<bool>,
    pub line_name: Vec<String>,
    pub line_colour: Vec<u32>,
    pub line_stops: Vec<Span>,
    pub line_path: Vec<Span>,
    pub line_tph: Vec<[f32; 3]>,
    pub line_dwell: Vec<f32>,
    pub line_turn: Vec<f32>,
    pub line_cars: Vec<u8>,
    /// Derived: the path uses a deleted edge or no longer fits the stops.
    pub line_broken: Vec<bool>,
    line_free: Vec<u32>,
    pub stops: Pool<u32>,
    pub paths: Pool<u32>,

    pub crossings: Vec<Crossing>,
    pub grid: Grid,
    /// CSR by edge id: lines whose path uses the edge, ascending.
    pub edge_lines_off: Vec<u32>,
    pub edge_lines: Vec<u32>,
    pub water_mask: Box<dyn WaterMask>,
    pub touch: Touch,
}

/// Slack in a Dijkstra key.
#[derive(PartialEq)]
struct Key(f64, u32);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

pub fn path_edge(p: u32) -> u32 {
    p >> 1
}
pub fn path_dir(p: u32) -> u32 {
    p & 1
}

impl Network {
    pub fn new(origin_lon: f64, origin_lat: f64, right_hand: bool) -> Network {
        Network {
            origin_lon,
            origin_lat,
            right_hand,
            node_alive: vec![],
            node_x: vec![],
            node_y: vec![],
            node_level: vec![],
            node_flying: vec![],
            node_platform: vec![],
            node_name: vec![],
            node_built: vec![],
            node_ports: vec![],
            node_heading: vec![],
            node_free: vec![],
            edge_alive: vec![],
            edge_a: vec![],
            edge_b: vec![],
            edge_tracks: vec![],
            edge_built: vec![],
            edge_pis: vec![],
            edge_thru: vec![],
            edge_len: vec![],
            edge_pieces: vec![],
            edge_vert: vec![],
            edge_water: vec![],
            edge_cost: vec![],
            edge_time: vec![],
            edge_free: vec![],
            pis: Pool::default(),
            pieces: Pool::default(),
            verts: Pool::default(),
            water: Pool::default(),
            line_alive: vec![],
            line_name: vec![],
            line_colour: vec![],
            line_stops: vec![],
            line_path: vec![],
            line_tph: vec![],
            line_dwell: vec![],
            line_turn: vec![],
            line_cars: vec![],
            line_broken: vec![],
            line_free: vec![],
            stops: Pool::default(),
            paths: Pool::default(),
            crossings: vec![],
            grid: Grid::default(),
            edge_lines_off: vec![0],
            edge_lines: vec![],
            water_mask: Box::new(NoWater),
            touch: Touch::default(),
        }
    }

    // ---------------------------------------------------------------- ids

    pub fn node_count(&self) -> usize {
        self.node_alive.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edge_alive.len()
    }
    pub fn line_count(&self) -> usize {
        self.line_alive.len()
    }
    pub fn node_ok(&self, n: u32) -> bool {
        self.node_alive.get(n as usize).copied().unwrap_or(false)
    }
    pub fn edge_ok(&self, e: u32) -> bool {
        self.edge_alive.get(e as usize).copied().unwrap_or(false)
    }
    pub fn line_ok(&self, l: u32) -> bool {
        self.line_alive.get(l as usize).copied().unwrap_or(false)
    }

    /// A free node id for a new node (claimed when the op adding it is applied).
    pub fn alloc_node(&mut self) -> u32 {
        self.node_free.pop().unwrap_or_else(|| {
            self.grow_nodes();
            self.node_alive.len() as u32 - 1
        })
    }
    pub fn alloc_edge(&mut self) -> u32 {
        self.edge_free.pop().unwrap_or_else(|| {
            self.grow_edges();
            self.edge_alive.len() as u32 - 1
        })
    }
    pub fn alloc_line(&mut self) -> u32 {
        self.line_free.pop().unwrap_or_else(|| {
            self.grow_lines();
            self.line_alive.len() as u32 - 1
        })
    }

    /// Give back ids handed out by `alloc_*` that no op claimed (a refused or trial edit).
    pub fn release_node(&mut self, id: u32) {
        if !self.node_ok(id) && (id as usize) < self.node_count() && !self.node_free.contains(&id) {
            self.node_free.push(id);
        }
    }
    pub fn release_edge(&mut self, id: u32) {
        if !self.edge_ok(id) && (id as usize) < self.edge_count() && !self.edge_free.contains(&id) {
            self.edge_free.push(id);
        }
    }
    pub fn release_line(&mut self, id: u32) {
        if !self.line_ok(id) && (id as usize) < self.line_count() && !self.line_free.contains(&id) {
            self.line_free.push(id);
        }
    }

    fn grow_nodes(&mut self) {
        self.node_alive.push(false);
        self.node_x.push(0.0);
        self.node_y.push(0.0);
        self.node_level.push(0);
        self.node_flying.push(false);
        self.node_platform.push(0);
        self.node_name.push(String::new());
        self.node_built.push(false);
        self.node_ports.push(Ports::default());
        self.node_heading.push(0.0);
    }
    fn grow_edges(&mut self) {
        self.edge_alive.push(false);
        self.edge_a.push(0);
        self.edge_b.push(0);
        self.edge_tracks.push(2);
        self.edge_built.push(false);
        self.edge_pis.push(Span::default());
        self.edge_thru.push(vec![]);
        self.edge_len.push(0.0);
        self.edge_pieces.push(Span::default());
        self.edge_vert.push(Span::default());
        self.edge_water.push(Span::default());
        self.edge_cost.push(0.0);
        self.edge_time.push(0.0);
        self.edge_lines_off.push(*self.edge_lines_off.last().unwrap());
    }
    fn grow_lines(&mut self) {
        self.line_alive.push(false);
        self.line_name.push(String::new());
        self.line_colour.push(0);
        self.line_stops.push(Span::default());
        self.line_path.push(Span::default());
        self.line_tph.push([0.0; 3]);
        self.line_dwell.push(DEFAULT_DWELL_S);
        self.line_turn.push(DEFAULT_TURNAROUND_S);
        self.line_cars.push(0);
        self.line_broken.push(false);
    }
    fn claim_node(&mut self, id: u32) {
        while self.node_alive.len() <= id as usize {
            self.grow_nodes();
            let n = self.node_alive.len() as u32 - 1;
            if n != id {
                self.node_free.push(n);
            }
        }
        self.node_free.retain(|&x| x != id);
    }
    fn claim_edge(&mut self, id: u32) {
        while self.edge_alive.len() <= id as usize {
            self.grow_edges();
            let n = self.edge_alive.len() as u32 - 1;
            if n != id {
                self.edge_free.push(n);
            }
        }
        self.edge_free.retain(|&x| x != id);
    }
    fn claim_line(&mut self, id: u32) {
        while self.line_alive.len() <= id as usize {
            self.grow_lines();
            let n = self.line_alive.len() as u32 - 1;
            if n != id {
                self.line_free.push(n);
            }
        }
        self.line_free.retain(|&x| x != id);
    }

    // ---------------------------------------------------------------- reading inputs

    pub fn node_data(&self, n: u32) -> Option<NodeData> {
        self.node_ok(n).then(|| {
            let i = n as usize;
            NodeData { x: self.node_x[i], y: self.node_y[i], level: self.node_level[i], flying: self.node_flying[i] }
        })
    }
    pub fn edge_data(&self, e: u32) -> Option<EdgeData> {
        self.edge_ok(e).then(|| {
            let i = e as usize;
            EdgeData { a: self.edge_a[i], b: self.edge_b[i], tracks: self.edge_tracks[i], pis: self.pis.get(self.edge_pis[i]).to_vec(), built: self.edge_built[i], thru: self.edge_thru[i].clone() }
        })
    }
    pub fn station_data(&self, n: u32) -> Option<StationData> {
        (self.node_ok(n) && self.node_platform[n as usize] > 0)
            .then(|| StationData { platform: self.node_platform[n as usize], name: self.node_name[n as usize].clone(), built: self.node_built[n as usize] })
    }
    pub fn line_data(&self, l: u32) -> Option<LineData> {
        self.line_ok(l).then(|| {
            let i = l as usize;
            LineData {
                name: self.line_name[i].clone(),
                colour: self.line_colour[i],
                stops: self.stops.get(self.line_stops[i]).to_vec(),
                path: self.paths.get(self.line_path[i]).to_vec(),
                tph: self.line_tph[i],
                dwell_s: self.line_dwell[i],
                turnaround_s: self.line_turn[i],
                cars: self.line_cars[i],
            }
        })
    }
    pub fn edge_pieces(&self, e: u32) -> &[Piece] {
        self.pieces.get(self.edge_pieces[e as usize])
    }
    pub fn edge_vert(&self, e: u32) -> &[[f64; 2]] {
        self.verts.get(self.edge_vert[e as usize])
    }
    pub fn edge_water(&self, e: u32) -> &[[f64; 2]] {
        self.water.get(self.edge_water[e as usize])
    }
    pub fn line_path(&self, l: u32) -> &[u32] {
        self.paths.get(self.line_path[l as usize])
    }
    pub fn line_stops(&self, l: u32) -> &[u32] {
        self.stops.get(self.line_stops[l as usize])
    }
    pub fn lines_on_edge(&self, e: u32) -> &[u32] {
        let i = e as usize;
        if i + 1 >= self.edge_lines_off.len() {
            return &[];
        }
        &self.edge_lines[self.edge_lines_off[i] as usize..self.edge_lines_off[i + 1] as usize]
    }
    /// The edge's vertices: its a node, its PIs, its b node.
    pub fn edge_vertices(&self, e: u32) -> Vec<Pi> {
        let i = e as usize;
        let (a, b) = (self.edge_a[i] as usize, self.edge_b[i] as usize);
        let mut v = Vec::with_capacity(self.edge_pis[i].len as usize + 2);
        v.push(Pi::new(self.node_x[a], self.node_y[a], 0.0, self.node_level[a]));
        v.extend_from_slice(self.pis.get(self.edge_pis[i]));
        v.push(Pi::new(self.node_x[b], self.node_y[b], 0.0, self.node_level[b]));
        v
    }
    /// The node an edge end is at.
    pub fn end_node(&self, e: u32, which: u32) -> u32 {
        if which == 0 {
            self.edge_a[e as usize]
        } else {
            self.edge_b[e as usize]
        }
    }
    /// Entry and exit node of a path entry.
    pub fn path_nodes(&self, p: u32) -> (u32, u32) {
        let e = path_edge(p);
        if path_dir(p) == 0 {
            (self.edge_a[e as usize], self.edge_b[e as usize])
        } else {
            (self.edge_b[e as usize], self.edge_a[e as usize])
        }
    }
    /// Side (0 = A, 1 = B) of an edge end at its node.
    pub fn port_side(&self, e: u32, which: u32) -> Option<u8> {
        let n = self.end_node(e, which);
        let p = &self.node_ports[n as usize];
        p.find(e, which).map(|i| p.side[i])
    }
    /// Tracks at a station (2 if any of its edges is double).
    pub fn node_tracks(&self, n: u32) -> u8 {
        let p = &self.node_ports[n as usize];
        p.ends().iter().map(|&c| self.edge_tracks[(c >> 1) as usize]).max().unwrap_or(2)
    }

    // ---------------------------------------------------------------- raw setters

    fn touch_node(&mut self, n: u32) {
        self.touch.nodes.push(n);
    }

    pub fn set_node(&mut self, id: u32, data: Option<NodeData>) -> Result<Option<NodeData>, Issue> {
        let old = self.node_data(id);
        match data {
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                if self.node_ports[i].n > 0 || self.node_platform[i] > 0 {
                    return Err(Issue::NodeInUse { node: id });
                }
                self.node_alive[i] = false;
                self.node_free.push(id);
            }
            Some(d) => {
                if !(MIN_LEVEL..=MAX_LEVEL).contains(&d.level) {
                    return Err(Issue::NodeLevel { node: id });
                }
                self.claim_node(id);
                let i = id as usize;
                self.node_alive[i] = true;
                self.node_x[i] = d.x;
                self.node_y[i] = d.y;
                self.node_level[i] = d.level;
                self.node_flying[i] = d.flying;
                let ends: Vec<u32> = self.node_ports[i].ends().to_vec();
                for c in ends {
                    self.touch.edges.push(c >> 1);
                }
            }
        }
        self.touch_node(id);
        Ok(old)
    }

    pub fn set_edge(&mut self, id: u32, data: Option<EdgeData>) -> Result<Option<EdgeData>, Issue> {
        let old = self.edge_data(id);
        if let Some(d) = &data {
            if !self.node_ok(d.a) {
                return Err(Issue::NoSuchNode { node: d.a });
            }
            if !self.node_ok(d.b) {
                return Err(Issue::NoSuchNode { node: d.b });
            }
            if d.a == d.b {
                return Err(Issue::EdgeLoop { edge: id });
            }
            if d.tracks != 1 && d.tracks != 2 {
                return Err(Issue::Tracks { edge: id });
            }
            for (node, which) in [(d.a, 0u32), (d.b, 1u32)] {
                let p = &self.node_ports[node as usize];
                let mine = old.as_ref().map_or(false, |o| (if which == 0 { o.a } else { o.b }) == node);
                if p.n as usize >= MAX_PORTS && !mine {
                    return Err(Issue::TooManyPorts { node });
                }
            }
        }
        if let Some(o) = &old {
            self.node_ports[o.a as usize].remove(id << 1);
            self.node_ports[o.b as usize].remove(id << 1 | 1);
            self.touch_node(o.a);
            self.touch_node(o.b);
        }
        match data {
            Some(d) => {
                self.claim_edge(id);
                let i = id as usize;
                self.edge_alive[i] = true;
                self.edge_a[i] = d.a;
                self.edge_b[i] = d.b;
                self.edge_tracks[i] = d.tracks;
                self.edge_built[i] = d.built;
                self.edge_pis[i] = self.pis.replace(self.edge_pis[i], &d.pis);
                self.edge_thru[i] = d.thru;
                self.node_ports[d.a as usize].add(id << 1);
                self.node_ports[d.b as usize].add(id << 1 | 1);
                self.touch_node(d.a);
                self.touch_node(d.b);
            }
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                self.edge_alive[i] = false;
                self.edge_built[i] = false;
                self.pis.free(self.edge_pis[i]);
                self.edge_pis[i] = Span::default();
                self.edge_thru[i] = vec![];
                self.edge_free.push(id);
            }
        }
        self.touch.edges.push(id);
        Ok(old)
    }

    pub fn set_station(&mut self, node: u32, data: Option<StationData>) -> Result<Option<StationData>, Issue> {
        if !self.node_ok(node) {
            return Err(Issue::NoSuchNode { node });
        }
        let old = self.station_data(node);
        let i = node as usize;
        match data {
            None => {
                for l in 0..self.line_count() as u32 {
                    if self.line_ok(l) && self.line_stops(l).contains(&node) {
                        return Err(Issue::StationInUse { node, line: l });
                    }
                }
                self.node_platform[i] = 0;
                self.node_name[i].clear();
                self.node_built[i] = false;
            }
            Some(d) => {
                self.node_platform[i] = d.platform;
                self.node_name[i] = d.name;
                self.node_built[i] = d.built;
            }
        }
        self.touch.stations.push(node);
        self.touch_node(node);
        Ok(old)
    }

    pub fn set_line(&mut self, id: u32, data: Option<LineData>) -> Result<Option<LineData>, Issue> {
        let old = self.line_data(id);
        match data {
            Some(d) => {
                if d.tph.iter().any(|t| !t.is_finite() || *t < 0.0) || d.dwell_s < 0.0 || d.turnaround_s < 0.0 {
                    return Err(Issue::LineSchedule { line: id });
                }
                self.claim_line(id);
                let i = id as usize;
                self.line_alive[i] = true;
                self.line_name[i] = d.name;
                self.line_colour[i] = d.colour;
                self.line_stops[i] = self.stops.replace(self.line_stops[i], &d.stops);
                self.line_path[i] = self.paths.replace(self.line_path[i], &d.path);
                self.line_tph[i] = d.tph;
                self.line_dwell[i] = d.dwell_s;
                self.line_turn[i] = d.turnaround_s;
                self.line_cars[i] = d.cars;
            }
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                self.line_alive[i] = false;
                self.stops.free(self.line_stops[i]);
                self.paths.free(self.line_path[i]);
                self.line_stops[i] = Span::default();
                self.line_path[i] = Span::default();
                self.line_free.push(id);
            }
        }
        self.touch.lines.push(id);
        self.touch.paths_changed = true;
        Ok(old)
    }

    pub fn set_schedule(&mut self, id: u32, tph: [f32; 3]) -> Result<[f32; 3], Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        if tph.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(Issue::LineSchedule { line: id });
        }
        let old = self.line_tph[id as usize];
        self.line_tph[id as usize] = tph;
        self.touch.schedules.push(id);
        Ok(old)
    }

    pub fn set_water_mask(&mut self, mask: Box<dyn WaterMask>) {
        self.water_mask = mask;
        for e in 0..self.edge_count() as u32 {
            if self.edge_ok(e) {
                self.touch.edges.push(e);
            }
        }
    }

    pub fn take_touch(&mut self) -> Touch {
        let mut t = std::mem::take(&mut self.touch);
        for v in [&mut t.nodes, &mut t.edges, &mut t.stations, &mut t.lines, &mut t.schedules] {
            v.sort_unstable();
            v.dedup();
        }
        t
    }

    /// Mark everything touched (after a bulk load).
    pub fn touch_all(&mut self) {
        self.touch.nodes = (0..self.node_count() as u32).filter(|&n| self.node_ok(n)).collect();
        self.touch.edges = (0..self.edge_count() as u32).filter(|&e| self.edge_ok(e)).collect();
        self.touch.lines = (0..self.line_count() as u32).filter(|&l| self.line_ok(l)).collect();
        self.touch.stations = self.touch.nodes.iter().copied().filter(|&n| self.node_platform[n as usize] > 0).collect();
        self.touch.paths_changed = true;
    }

    // ---------------------------------------------------------------- derive

    /// Rebuild what the touched inputs feed: edge geometry and grid, node ports, crossings, the
    /// edge-to-lines index and broken flags. `t` is the touch record (from `take_touch`).
    pub fn derive(&mut self, t: &Touch) -> Derived {
        let mut d = Derived::default();
        let mut edges = t.edges.clone();
        for &n in &t.nodes {
            if (n as usize) < self.node_count() {
                for &c in self.node_ports[n as usize].ends() {
                    edges.push(c >> 1);
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let mut fresh = vec![];
        for &e in &edges {
            let i = e as usize;
            let old = self.edge_pieces[i];
            if old.len > 0 {
                let p = self.pieces.get(old).to_vec();
                self.grid.remove(e, &p);
                self.tiles_of(&p, &mut d.tiles);
            }
            if !self.edge_alive[i] {
                self.pieces.free(old);
                self.verts.free(self.edge_vert[i]);
                self.water.free(self.edge_water[i]);
                self.edge_pieces[i] = Span::default();
                self.edge_vert[i] = Span::default();
                self.edge_water[i] = Span::default();
                self.edge_len[i] = 0.0;
                self.edge_cost[i] = 0.0;
                self.edge_time[i] = 0.0;
                continue;
            }
            let f = geom::fit(&self.edge_vertices(e));
            let mut wet = vec![];
            cost::water_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); (x, y) }, self.water_mask.as_ref(), &mut wet);
            self.edge_len[i] = f.len;
            self.edge_cost[i] = cost::track_cost(f.len, &f.vert, &wet, self.edge_tracks[i]);
            self.edge_time[i] = f.pieces.iter().map(|p| p.len / p.v_limit()).sum::<f64>().max(1e-3);
            self.edge_pieces[i] = self.pieces.replace(old, &f.pieces);
            self.edge_vert[i] = self.verts.replace(self.edge_vert[i], &f.vert);
            self.edge_water[i] = self.water.replace(self.edge_water[i], &wet);
            self.grid.insert(e, &f.pieces);
            self.tiles_of(&f.pieces, &mut d.tiles);
            fresh.push(e);
        }
        // Nodes: the touched ones and every node at a re-derived edge's ends.
        let mut nodes = t.nodes.clone();
        for &e in &fresh {
            nodes.push(self.edge_a[e as usize]);
            nodes.push(self.edge_b[e as usize]);
        }
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            if self.node_ok(n) {
                self.derive_node(n);
                if self.node_ports[n as usize].n == 0 {
                    // An edgeless node still marks its tile.
                    d.tiles.push(self.tile_of(self.node_x[n as usize], self.node_y[n as usize]));
                }
            }
        }
        // Crossings of touched edges.
        let before = self.crossings.len();
        let is_touched = |e: u32| edges.binary_search(&e).is_ok();
        let mut kept = Vec::with_capacity(before);
        for c in self.crossings.drain(..) {
            if is_touched(c.e1) || is_touched(c.e2) {
                d.crossings.push([c.x, c.y]);
            } else {
                kept.push(c);
            }
        }
        self.crossings = kept;
        for &e in &fresh {
            let near = self.grid.near(self.edge_pieces(e));
            for f in near {
                // Pairs of two touched edges are tested once, from the smaller id.
                if f < e && fresh.binary_search(&f).is_ok() {
                    continue;
                }
                self.find_crossings(e, f, &mut d.crossings);
            }
        }
        self.crossings.sort_by(|a, b| (a.e1, a.e2).cmp(&(b.e1, b.e2)).then(a.s1.total_cmp(&b.s1)));
        if t.paths_changed || edges.iter().any(|&e| !self.edge_alive[e as usize]) {
            self.rebuild_edge_lines();
        }
        // Broken flags of lines through touched edges and of touched lines.
        let mut lines: Vec<u32> = t.lines.clone();
        for &e in &edges {
            lines.extend_from_slice(self.lines_on_edge(e));
        }
        for &n in &t.stations {
            for l in 0..self.line_count() as u32 {
                if self.line_ok(l) && self.line_stops(l).contains(&n) {
                    lines.push(l);
                }
            }
        }
        lines.sort_unstable();
        lines.dedup();
        for l in lines {
            if self.line_ok(l) {
                self.line_broken[l as usize] = self.check_line(l).is_err();
            }
        }
        self.maybe_compact();
        d.tiles.sort_unstable();
        d.tiles.dedup();
        d.edges = edges;
        d.nodes = nodes;
        d
    }

    fn derive_node(&mut self, n: u32) {
        let i = n as usize;
        let ports = self.node_ports[i];
        if ports.n == 0 {
            self.node_heading[i] = 0.0;
            return;
        }
        let (nx, ny) = (self.node_x[i], self.node_y[i]);
        // Outward heading of each end; the lowest end defines the node's heading (its side B).
        let mut out_th = [0.0f64; MAX_PORTS];
        let mut lat = [0.0f64; MAX_PORTS];
        for (k, &c) in ports.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            if pcs.is_empty() {
                continue;
            }
            out_th[k] = if which == 0 { geom::pos_at(pcs, 0.0).2 } else { geom::pos_at(pcs, self.edge_len[e as usize]).2 + PI };
        }
        let h = out_th[0];
        self.node_heading[i] = h;
        let (hc, hs) = (h.cos(), h.sin());
        let mut p = ports;
        for (k, &c) in ports.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            let len = self.edge_len[e as usize];
            p.side[k] = if (out_th[k] - h).cos() > 0.0 { 1 } else { 0 };
            if pcs.is_empty() {
                continue;
            }
            let d = len.min(200.0);
            let (px, py, _) = if which == 0 { geom::pos_at(pcs, d) } else { geom::pos_at(pcs, len - d) };
            lat[k] = hc * (py - ny) - hs * (px - nx);
        }
        for side in 0..2u8 {
            let mut idx: Vec<usize> = (0..ports.n as usize).filter(|&k| p.side[k] == side).collect();
            idx.sort_by(|&a, &b| lat[a].total_cmp(&lat[b]).then(ports.end[a].cmp(&ports.end[b])));
            for (r, &k) in idx.iter().enumerate() {
                p.rank[k] = r as u8;
            }
        }
        self.node_ports[i] = p;
    }

    fn find_crossings(&mut self, e: u32, f: u32, changed: &mut Vec<[f64; 2]>) {
        let (ie, i_f) = (e as usize, f as usize);
        let mut hits = vec![];
        cross::intersect_alignments(self.edge_pieces(e), self.edge_pieces(f), e == f, &mut hits);
        if hits.is_empty() {
            return;
        }
        let shared: Vec<u32> = [self.edge_a[ie], self.edge_b[ie]]
            .into_iter()
            .filter(|&n| e != f && (n == self.edge_a[i_f] || n == self.edge_b[i_f]))
            .collect();
        for (s1, s2, overlap) in hits {
            let (x, y, _) = geom::pos_at(self.edge_pieces(e), s1);
            if shared.iter().any(|&n| (self.node_x[n as usize] - x).hypot(self.node_y[n as usize] - y) < SHARED_NODE_CLEAR) {
                continue;
            }
            let (v1, v2) = (self.edge_vert(e), self.edge_vert(f));
            let (z1, z2) = (geom::height_at(v1, s1), geom::height_at(v2, s2));
            let dz = (z1 - z2).abs();
            if dz >= LEVEL_H - 1e-6 {
                continue;
            }
            let kind = if overlap {
                CrossKind::Overlap
            } else if dz < 1e-6 && !geom::on_ramp(v1, s1 - 1.0, s1 + 1.0) && !geom::on_ramp(v2, s2 - 1.0, s2 + 1.0) {
                CrossKind::Flat(cost::level_of(z1))
            } else {
                CrossKind::Bad
            };
            let (e1, s1, e2, s2) = if e <= f { (e, s1, f, s2) } else { (f, s2, e, s1) };
            self.crossings.push(Crossing { e1, s1, e2, s2, x, y, kind });
            changed.push([x, y]);
        }
    }

    fn rebuild_edge_lines(&mut self) {
        let mut pairs: Vec<(u32, u32)> = vec![];
        for l in 0..self.line_count() as u32 {
            if !self.line_ok(l) {
                continue;
            }
            for &p in self.line_path(l) {
                pairs.push((path_edge(p), l));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let ne = self.edge_count();
        let mut off = vec![0u32; ne + 1];
        for &(e, _) in &pairs {
            if (e as usize) < ne {
                off[e as usize + 1] += 1;
            }
        }
        for i in 0..ne {
            off[i + 1] += off[i];
        }
        self.edge_lines = pairs.iter().filter(|p| (p.0 as usize) < ne).map(|p| p.1).collect();
        self.edge_lines_off = off;
    }

    fn maybe_compact(&mut self) {
        if self.pis.needs_compact() {
            self.pis.compact(&mut self.edge_pis);
        }
        if self.pieces.needs_compact() {
            self.pieces.compact(&mut self.edge_pieces);
        }
        if self.verts.needs_compact() {
            self.verts.compact(&mut self.edge_vert);
        }
        if self.water.needs_compact() {
            self.water.compact(&mut self.edge_water);
        }
        if self.stops.needs_compact() {
            self.stops.compact(&mut self.line_stops);
        }
        if self.paths.needs_compact() {
            self.paths.compact(&mut self.line_path);
        }
    }

    // ---------------------------------------------------------------- tiles

    /// Web mercator tile (zoom 12) of a local point.
    pub fn tile_of(&self, x: f64, y: f64) -> (u32, u32) {
        let lat0 = self.origin_lat.to_radians();
        let lon = self.origin_lon + (x / (EARTH_R * lat0.cos())).to_degrees();
        let lat = (self.origin_lat + (y / EARTH_R).to_degrees()).to_radians();
        let z = (1u32 << TILE_ZOOM) as f64;
        let tx = ((lon + 180.0) / 360.0 * z).floor();
        let ty = ((1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / PI) / 2.0 * z).floor();
        (tx.clamp(0.0, z - 1.0) as u32, ty.clamp(0.0, z - 1.0) as u32)
    }

    fn tiles_of(&self, pieces: &[Piece], out: &mut Vec<(u32, u32)>) {
        let mut pts = vec![];
        geom::sample(pieces, 200.0, 10.0, 0.0, &mut pts);
        for p in pts {
            out.push(self.tile_of(p[0], p[1]));
        }
    }

    pub fn edge_tiles(&self, e: u32) -> Vec<(u32, u32)> {
        let mut t = vec![];
        self.tiles_of(self.edge_pieces(e), &mut t);
        t.sort_unstable();
        t.dedup();
        t
    }

    // ---------------------------------------------------------------- validity

    /// Problems with the given edges, nodes (with their stations) and lines, and the crossings of
    /// those edges. Empty = valid.
    pub fn validate(&self, edges: &[u32], nodes: &[u32], lines: &[u32]) -> Vec<Issue> {
        let mut out = vec![];
        for &e in edges {
            if self.edge_ok(e) {
                self.check_edge(e, &mut out);
            }
        }
        for c in &self.crossings {
            if !matches!(c.kind, CrossKind::Flat(_)) && (edges.binary_search(&c.e1).is_ok() || edges.binary_search(&c.e2).is_ok()) {
                out.push(Issue::Crossing { e1: c.e1, e2: c.e2, x: c.x, y: c.y, overlap: c.kind == CrossKind::Overlap });
            }
        }
        for &n in nodes {
            if self.node_ok(n) {
                self.check_node(n, &mut out);
            }
        }
        for &l in lines {
            if self.line_ok(l) && !self.line_broken[l as usize] {
                if let Err(i) = self.check_line(l) {
                    out.push(i);
                }
            }
        }
        out
    }

    fn check_edge(&self, e: u32, out: &mut Vec<Issue>) {
        let f = geom::fit(&self.edge_vertices(e));
        for issue in f.issues {
            out.push(Issue::Geom { edge: e, issue });
        }
        let vert = self.edge_vert(e);
        for w in self.edge_water(e) {
            // Ground level over water: anything nearer the ground than half a level.
            let mut s = w[0];
            while s <= w[1] {
                if geom::height_at(vert, s).abs() < LEVEL_H / 2.0 {
                    out.push(Issue::GroundOverWater { edge: e, s });
                    break;
                }
                s += WATER_STEP / 2.0;
            }
        }
    }

    fn check_node(&self, n: u32, out: &mut Vec<Issue>) {
        let i = n as usize;
        let p = &self.node_ports[i];
        let h = self.node_heading[i];
        for (k, &c) in p.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            if pcs.is_empty() {
                continue;
            }
            let th = if which == 0 { geom::pos_at(pcs, 0.0).2 } else { geom::pos_at(pcs, self.edge_len[e as usize]).2 + PI };
            let d = geom::wrap(th - h);
            let off = if p.side[k] == 1 { d.abs() } else { PI - d.abs() };
            if off > HEADING_TOL {
                out.push(Issue::NodeHeading { node: n, edge: e });
            }
        }
        // Two ends leaving the same way with nothing behind is a harmless fork (e.g. a junction
        // whose stem was removed); a station must have its platform track run through.
        if p.n >= 2 && self.node_platform[i] > 0 {
            let b = p.side[..p.n as usize].iter().filter(|&&s| s == 1).count();
            if b == 0 || b == p.n as usize {
                out.push(Issue::NodeOneSided { node: n });
            }
        }
        if self.node_platform[i] > 0 {
            self.check_station(n, out);
        }
    }

    /// How far a station's platform reaches along each of its edges.
    pub fn platform_reach(&self, n: u32) -> f64 {
        let i = n as usize;
        let plat = self.node_platform[i] as f64;
        if self.node_ports[i].n <= 1 {
            plat
        } else {
            plat / 2.0
        }
    }

    fn check_station(&self, n: u32, out: &mut Vec<Issue>) {
        let i = n as usize;
        let plat = self.node_platform[i];
        if !(PLATFORM_MIN..=PLATFORM_MAX).contains(&plat) || plat % PLATFORM_STEP != 0 {
            out.push(Issue::PlatformLength { node: n });
        }
        let p = self.node_ports[i];
        // A station whose track was removed waits for new track (its lines are broken).
        if p.n > 2 {
            out.push(Issue::StationPorts { node: n });
            return;
        }
        let reach = self.platform_reach(n);
        for &c in p.ends() {
            let (e, which) = (c >> 1, c & 1);
            let len = self.edge_len[e as usize];
            let other = self.end_node(e, 1 - which);
            let other_reach = if self.node_platform[other as usize] > 0 { self.platform_reach(other) } else { 0.0 };
            if reach + other_reach > len - 1.0 {
                out.push(Issue::PlatformTooLong { node: n });
            }
            let (a, b) = if which == 0 { (0.0, reach) } else { (len - reach, len) };
            if geom::on_ramp(self.edge_vert(e), a, b) {
                out.push(Issue::PlatformNotLevel { node: n });
            }
            for x in &self.crossings {
                let s = if x.e1 == e { x.s1 } else if x.e2 == e { x.s2 } else { continue };
                if s >= a - 1.0 && s <= b + 1.0 {
                    out.push(Issue::PlatformCrossing { node: n });
                }
            }
        }
        if self.node_level[i] == 0 && self.water_mask.is_water(self.node_x[i], self.node_y[i]) {
            out.push(Issue::StationGroundOverWater { node: n });
        }
    }

    /// The node sequence of a line's path, or the first problem with it.
    pub fn check_line(&self, l: u32) -> Result<Vec<u32>, Issue> {
        let path = self.line_path(l);
        let stops = self.line_stops(l);
        if path.is_empty() {
            return Err(Issue::LinePath { line: l });
        }
        if stops.len() < 2 || stops.windows(2).any(|w| w[0] == w[1]) {
            return Err(Issue::LineStops { line: l });
        }
        for &p in path {
            if !self.edge_ok(path_edge(p)) {
                return Err(Issue::LinePath { line: l });
            }
        }
        let mut nodes = Vec::with_capacity(path.len() + 1);
        nodes.push(self.path_nodes(path[0]).0);
        for k in 0..path.len() {
            let (from, to) = self.path_nodes(path[k]);
            if k > 0 {
                let prev = path[k - 1];
                if self.path_nodes(prev).1 != from {
                    return Err(Issue::LinePath { line: l });
                }
                // Through a node: arrive on one side, leave on the other.
                let arr = self.port_side(path_edge(prev), 1 - path_dir(prev));
                let dep = self.port_side(path_edge(path[k]), path_dir(path[k]));
                if arr.is_none() || arr == dep {
                    return Err(Issue::LinePath { line: l });
                }
            }
            nodes.push(to);
        }
        if stops[0] != nodes[0] || *stops.last().unwrap() != *nodes.last().unwrap() {
            return Err(Issue::LineStops { line: l });
        }
        let mut k = 0;
        for &s in stops {
            if !self.node_ok(s) || self.node_platform[s as usize] == 0 {
                return Err(Issue::LineStops { line: l });
            }
            while k < nodes.len() && nodes[k] != s {
                k += 1;
            }
            if k == nodes.len() {
                return Err(Issue::LineStops { line: l });
            }
            k += 1;
        }
        Ok(nodes)
    }

    // ---------------------------------------------------------------- cost

    /// Build cost of everything standing, US$M.
    pub fn total_cost(&self) -> f64 {
        let mut c = 0.0;
        for e in 0..self.edge_count() {
            if self.edge_alive[e] {
                c += self.edge_cost[e];
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.water_mask.is_water(self.node_x[n], self.node_y[n]);
            if self.node_platform[n] > 0 {
                c += cost::station_cost(self.node_level[n], self.node_platform[n], self.node_tracks(n as u32), wet());
            }
            if self.node_ports[n].n >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
            }
        }
        c
    }

    /// Build cost of what is constructed, US$M (SPEC 6.4): constructed edges and stations, a
    /// junction once three of its edge ends are constructed, a flat crossing once both its edges
    /// are. `total_cost() - built_cost()` is what the blueprint would cost to construct.
    pub fn built_cost(&self) -> f64 {
        let mut c = 0.0;
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && self.edge_built[e] {
                c += self.edge_cost[e];
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.water_mask.is_water(self.node_x[n], self.node_y[n]);
            if self.node_platform[n] > 0 && self.node_built[n] {
                c += cost::station_cost(self.node_level[n], self.node_platform[n], self.node_tracks(n as u32), wet());
            }
            if self.built_ports(n as u32) >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.edge_built[x.e1 as usize] && self.edge_built[x.e2 as usize] {
                    c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                }
            }
        }
        c
    }

    /// Edge ends at a node whose edge is constructed.
    pub fn built_ports(&self, n: u32) -> usize {
        self.node_ports[n as usize].ends().iter().filter(|&&c| self.edge_built[(c >> 1) as usize]).count()
    }

    /// The constructed station or edge set is complete for a line: it can carry trains.
    pub fn line_constructed(&self, l: u32) -> bool {
        self.line_path(l).iter().all(|&p| self.edge_ok(path_edge(p)) && self.edge_built[path_edge(p) as usize])
            && self.line_stops(l).iter().all(|&s| self.node_ok(s) && self.node_built[s as usize])
    }

    /// The point of an edge nearest to (x, y): chainage and distance, m.
    pub fn project(&self, e: u32, x: f64, y: f64) -> (f64, f64) {
        let mut best = (0.0, f64::INFINITY);
        for p in self.edge_pieces(e) {
            let cands: [f64; 2] = if p.k == 0.0 {
                let d = ((x - p.x0) * p.th0.cos() + (y - p.y0) * p.th0.sin()).clamp(0.0, p.len);
                [d, d]
            } else {
                // Angle of the point around the centre, as a distance along the arc from its
                // start; the wrap can land either side of the start, so try both turns.
                let (cx, cy) = p.center();
                let a0 = (p.y0 - cy).atan2(p.x0 - cx);
                let a = (y - cy).atan2(x - cx);
                let t = geom::wrap((a - a0) * p.k.signum());
                [(t / p.k.abs()).clamp(0.0, p.len), ((t + 2.0 * PI) / p.k.abs()).clamp(0.0, p.len)]
            };
            for ds in cands {
                let (px, py, _) = p.at(ds);
                let dist = (px - x).hypot(py - y);
                if dist < best.1 {
                    best = (p.s0 + ds, dist);
                }
            }
        }
        best
    }

    // ---------------------------------------------------------------- routing

    /// Fastest path from `from` to `to` over valid moves, by speed-limit running time. `arrived`
    /// is the side the train came in on at `from` (it must leave on the other), or None at a
    /// line's first stop. Returns the path and the side it arrives on at `to`.
    pub fn find_path(&self, from: u32, to: u32, arrived: Option<u8>) -> Option<(Vec<u32>, u8)> {
        let ne = self.edge_count();
        let mut dist = vec![f64::INFINITY; 2 * ne];
        let mut prev = vec![u32::MAX; 2 * ne];
        let mut heap = BinaryHeap::new();
        let p = &self.node_ports[from as usize];
        for (k, &c) in p.ends().iter().enumerate() {
            if arrived == Some(p.side[k]) {
                continue;
            }
            let (e, which) = (c >> 1, c & 1);
            let st = e << 1 | which; // leaving from end a = direction 0
            dist[st as usize] = self.edge_time[e as usize];
            heap.push(Key(dist[st as usize], st));
        }
        while let Some(Key(d, st)) = heap.pop() {
            if d > dist[st as usize] {
                continue;
            }
            let (_, node) = self.path_nodes(st);
            let arr_side = self.port_side(path_edge(st), 1 - path_dir(st)).unwrap_or(0);
            if node == to {
                let mut path = vec![st];
                let mut cur = st;
                while prev[cur as usize] != u32::MAX {
                    cur = prev[cur as usize];
                    path.push(cur);
                }
                path.reverse();
                return Some((path, arr_side));
            }
            let p = &self.node_ports[node as usize];
            for (k, &c) in p.ends().iter().enumerate() {
                if p.side[k] == arr_side {
                    continue;
                }
                let (e, which) = (c >> 1, c & 1);
                let nst = e << 1 | which;
                let nd = d + self.edge_time[e as usize];
                if nd < dist[nst as usize] {
                    dist[nst as usize] = nd;
                    prev[nst as usize] = st;
                    heap.push(Key(nd, nst));
                }
            }
        }
        None
    }

    /// Fastest path through a list of stops, hop by hop.
    pub fn route(&self, stops: &[u32]) -> Option<Vec<u32>> {
        let mut path = vec![];
        let mut side = None;
        for w in stops.windows(2) {
            let (p, s) = self.find_path(w[0], w[1], side)?;
            path.extend(p);
            side = Some(s);
        }
        Some(path)
    }

    /// Where to split an edge at chainage `s` without changing its geometry (notes/T-008.md 4.1):
    /// the new node, the first part (keeps the edge's id) and the second part. A split inside an
    /// arc gets a new PI on each side with the arc's radius; the neighbouring PIs whose legs
    /// change keep their current radius as a set one.
    pub fn split_plan(&self, e: u32, s: f64) -> Result<(NodeData, EdgeData, EdgeData), Issue> {
        if !self.edge_ok(e) {
            return Err(Issue::NoSuchEdge { edge: e });
        }
        let v = self.edge_vertices(e);
        let f = geom::fit(&v);
        let mut s = s;
        if s < 5.0 || s > f.len - 5.0 {
            return Err(Issue::SplitPoint { edge: e });
        }
        // Snap to a piece boundary within 2 m, so new PIs are never a hair from the node.
        for p in &f.pieces {
            for b in [p.s0, p.s0 + p.len] {
                if (s - b).abs() < 2.0 {
                    s = b;
                }
            }
        }
        if geom::on_ramp(&f.vert, s - 1.0, s + 1.0) {
            return Err(Issue::SplitPoint { edge: e });
        }
        let z = geom::height_at(&f.vert, s);
        let level = cost::level_of(z);
        let j = geom::piece_at(&f.pieces, s);
        let piece = f.pieces[j];
        let (x, y, th) = piece.at(s - piece.s0);
        let node = NodeData { x: quant(x), y: quant(y), level, flying: false };
        let freeze = |vv: &mut Vec<Pi>, k: usize, src: usize| {
            if k > 0 && k < vv.len() - 1 && vv[k].radius == 0.0 && f.radius[src] > 0.0 {
                vv[k].radius = quant(f.radius[src]);
            }
        };
        let marker = Pi::new(node.x, node.y, 0.0, level);
        let (mut p1, mut p2): (Vec<Pi>, Vec<Pi>);
        let inside_arc = piece.k != 0.0 && s > piece.s0 + 1e-9 && s < piece.s0 + piece.len - 1e-9;
        if inside_arc {
            let m = f.piece_at_vertex[j] as usize;
            let r = quant(1.0 / piece.k.abs());
            let t1 = r * ((s - piece.s0) * piece.k.abs() / 2.0).tan();
            let t2 = r * ((piece.s0 + piece.len - s) * piece.k.abs() / 2.0).tan();
            let pi1 = Pi::new(quant(x - t1 * th.cos()), quant(y - t1 * th.sin()), r, v[m].level);
            let pi2 = Pi::new(quant(x + t2 * th.cos()), quant(y + t2 * th.sin()), r, v[m].level);
            p1 = v[..m].to_vec();
            p1.push(pi1);
            p1.push(marker);
            p2 = vec![marker, pi2];
            p2.extend_from_slice(&v[m + 1..]);
            freeze(&mut p1, m - 1, m - 1);
            freeze(&mut p2, 2, m + 1);
        } else {
            // On the straight of leg i (a boundary snap lands on a straight's leg too).
            let i = if piece.k == 0.0 {
                f.piece_at_vertex[j] as usize
            } else if (s - piece.s0).abs() < 1e-9 {
                f.piece_at_vertex[j] as usize - 1
            } else {
                f.piece_at_vertex[j] as usize
            };
            p1 = v[..=i].to_vec();
            p1.push(marker);
            p2 = vec![marker];
            p2.extend_from_slice(&v[i + 1..]);
            freeze(&mut p1, i, i);
            freeze(&mut p2, 1, i + 1);
        }
        let i = e as usize;
        let (tracks, built) = (self.edge_tracks[i], self.edge_built[i]);
        // A T-079 edge's clicked points go with the half they lie on; one at the split is the
        // node now.
        let (mut t1, mut t2) = (vec![], vec![]);
        for p in &self.edge_thru[i] {
            let sp = self.project(e, p.x, p.y).0;
            if sp < s - 5.0 {
                t1.push(*p);
            } else if sp > s + 5.0 {
                t2.push(*p);
            }
        }
        let a = EdgeData { a: self.edge_a[i], b: u32::MAX, tracks, pis: p1[1..p1.len() - 1].to_vec(), built, thru: t1 };
        let b = EdgeData { a: u32::MAX, b: self.edge_b[i], tracks, pis: p2[1..p2.len() - 1].to_vec(), built, thru: t2 };
        Ok((node, a, b))
    }
}
