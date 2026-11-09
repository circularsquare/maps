//! The player's network (SPEC 6.1-6.3, notes/T-008.md 4): nodes, edges, stations and lines as
//! flat per-field arrays indexed by stable u32 ids, plus everything derived from them (pieces,
//! heights, water, cost, node ports, crossings, the edge-to-lines index).
//!
//! Inputs change only through the raw setters (`set_node`, `set_edge`, `set_station`,
//! `set_line`, `set_tph`), which record what they touched; `derive` then rebuilds just those
//! parts. `world.rs` builds undoable operations on top.

pub use super::bus::BusRoute;
pub use super::ferry::FerryRoute;
use super::cost::{self, NoWater, Terrain, WaterMask};
use super::cross::{self, Grid};
use super::frame::{Frame, FrameKind};
use super::geom::{self, GeomIssue, Pi, Piece};
use super::params::*;
use super::roads::{self, Roads};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

/// Most edge ends one node can have.
pub const MAX_PORTS: usize = 8;

/// What a newly installed file changes in the edges it reaches (`Network::rework_ground`).
#[derive(Clone, Copy)]
enum Ground {
    Water,
    Terrain,
}

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
    /// A blueprint the player switched off (T-333): Construct and the blueprint quote skip it. It
    /// changes nothing else, and building it by hand (a piece's or a line's Construct) clears it.
    pub off: bool,
    /// Kept only for track drawn while the track ran through the clicked points (T-079, saves
    /// from then until T-092): the points clicked, which its PIs were fitted through. Geometry
    /// never reads them. Such an edge keeps its shape until it is reshaped; the first reshape
    /// (a PI drag, a radius, a node drag) makes these points its PIs with auto radius and clears
    /// them (`TrackApi::editable`). Empty for every edge drawn before T-079 or after T-092.
    pub thru: Vec<Pi>,
    /// One diagonal of a crossover (T-338): fixed shape, `V_CROSSOVER`, priced as a junction.
    pub crossover: bool,
    /// High-speed track (T-817): always double, priced at `HSR_COST`.
    pub hsr: bool,
    /// Grade-separated from every edge with this set, itself included (T-1751): where two meet there is no
    /// crossing at all, flat or bad. Set on an imported real network's edges, whose lines were
    /// each given one of six levels and cannot all avoid one another.
    pub sep: bool,
    /// The day this was first constructed (T-2871), as `DAY_UNKNOWN` documents; 0 on a blueprint.
    pub day: u32,
}

/// T-2871: a track piece's, station's or line's build day: 0 until constructed, this for anything
/// built before recording began (an older save), else the game day + 2 (`Network::today`).
pub const DAY_UNKNOWN: u32 = 1;

/// The earlier of two build days, a day still unset (0) losing to any other.
pub fn earliest_day(a: u32, b: u32) -> u32 {
    match (a, b) {
        (0, b) => b,
        (a, 0) => a,
        _ => a.min(b),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StationData {
    pub platform: u16,
    pub name: String,
    /// The player typed the name, so the game keeps it (T-2683).
    pub named: bool,
    /// Constructed; false = blueprint (SPEC 6.4).
    pub built: bool,
    /// Switched off while a blueprint (T-333, see `EdgeData::off`).
    pub off: bool,
    /// Station upgrades, `upgrades::TERMINAL` and the like (T-1281); built when the station is.
    pub upgrades: u8,
    /// A port (T-2193, `ferry.rs`): its berth, where boats stop, in the water near the node (the
    /// landing). None for a station on track.
    pub berth: Option<[f64; 2]>,
    /// T-2871: when first constructed, as `EdgeData::day`.
    pub day: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineData {
    pub name: String,
    pub symbol: String,
    pub colour: u32,
    /// Station nodes in order.
    pub stops: Vec<u32>,
    /// The exact path from the first stop to the last, `edge << 1 | dir` (dir 0 = a to b).
    pub path: Vec<u32>,
    /// Trains per hour for high, medium and low demand (SPEC 6.3).
    pub tph: [f32; 3],
    /// Whole trains allocated across both directions; None only until the next settle, which gives a line its trains from `tph` (`TrackWorld::allocate_unallocated`, T-1146).
    pub trains: Option<[u32; 3]>,
    /// T-625: headway in seconds for high, medium and low demand (0 = no service). When set the
    /// whole-line counts follow it: the fewest trains that keep it over the round trip.
    pub auto: Option<[u32; 3]>,
    pub dwell_s: f32,
    pub turnaround_s: f32,
    /// Cars per train; 0 = as many as the shortest platform allows.
    pub cars: u8,
    /// High-speed trains (T-817): top speed 320 km/h on high-speed track, dearer cars and running.
    pub fast: bool,
    /// A bus line (T-1464): its route; its stops and path stay empty (`bus::plan` places the stops).
    pub bus: Option<BusRoute>,
    /// A ferry line (T-2193): its route on water; its stops are its ports, its path stays empty.
    pub ferry: Option<FerryRoute>,
    /// The player's line group (T-1791), 0 for none: a grouping for the Lines tab only, which
    /// the service never reads. Kept on the line so a copy, an undo and a delete carry it.
    pub group: u32,
    /// The group's name, the same on every line of it; empty for the default (its oldest line's).
    pub group_name: String,
    /// The group's bullet as the player set it (T-1813), the same on every line of it.
    pub group_look: GroupLook,
    /// T-2871: the day this line was added, or for a bus or ferry line its route constructed, as
    /// `EdgeData::day` (0 for a planned route).
    pub day: u32,
    /// T-2895: the player set the stop order by hand, so no stop edit puts the stops in a
    /// straighter order (`straighten`).
    pub hand: bool,
}

/// A line group's bullet set by hand (T-1813): its text (empty for the name's own) and its colour
/// (None for the colour most of its lines share).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GroupLook {
    pub symbol: String,
    pub colour: Option<u32>,
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
    /// A segment drawn at ground level over water that has no room on land for a bridge's
    /// ramps (T-656); the drawing tool words it in place of `GroundOverWater`.
    BridgeNoRoom,
    /// `BridgeNoRoom` where the water is too near a station the segment starts or ends at (T-2414).
    BridgeNoRoomStation,
    /// `BridgeNoRoom` where the water is too near a junction, or track the segment branches off.
    BridgeNoRoomJunction,
    /// `BridgeNoRoom` where the water is too near the end of built track.
    BridgeNoRoomBuilt,
    /// A viaduct or cut and cover tunnel over more than `WATER_SPAN_MAX` of unbroken water (T-1930).
    WaterTooLong { edge: u32 },
    Crossing { e1: u32, e2: u32, x: f64, y: f64, overlap: bool },
    NoSuchNode { node: u32 },
    NoSuchEdge { edge: u32 },
    /// T-2797: the piece is only the track a station stands on; removing it would lay it back.
    StationTrack { edge: u32 },
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
    /// A station on a node that has one already.
    StationPorts { node: u32 },
    /// A station on a node where more than two pieces of track meet.
    StationJunction { node: u32 },
    PlatformLength { node: u32 },
    /// The platform reaches a loose end.
    PlatformTooLong { node: u32 },
    /// The platform reaches another station's platform.
    PlatformStation { node: u32 },
    /// The platform reaches a junction.
    PlatformJunction { node: u32 },
    /// The platform reaches the end of its piece of track at a plain joint that could not be
    /// merged away (T-622: constructed track, a different track count or built state beside).
    PlatformJoint { node: u32 },
    PlatformNotLevel { node: u32 },
    PlatformCrossing { node: u32 },
    StationGroundOverWater { node: u32 },
    StationInUse { node: u32, line: u32 },
    LineStops { line: u32 },
    LinePath { line: u32 },
    LineSchedule { line: u32 },
    /// A split point on a ramp, too near a node, or outside the edge.
    SplitPoint { edge: u32 },
    /// A node or PI nearer the poles than 85 degrees or past the antimeridian (Mercator frame,
    /// docs/world.md).
    OutsideWorld { x: f64, y: f64 },
    /// A crossover's shape is fixed (T-338): it cannot be reshaped, split, raised or moved.
    Crossover { edge: u32 },
    /// No track runs parallel within `CROSSOVER_REACH` for a crossover, or not for its length.
    NoParallel { edge: u32 },
    /// Doubling up track that has track beside it already (T-708).
    Alongside { edge: u32 },
    /// Doubling up high-speed track, which has no quad form (T-817).
    HighSpeed { edge: u32 },
    /// Construct with nothing left to construct (T-849).
    NothingToConstruct,
    /// A station building (T-2217, `building.rs`) with fewer than 3 or more than 24 corners, two
    /// corners in one place, or one outside the world.
    BuildingCorners,
    /// Two of a building's edges cross.
    BuildingEdgesCross,
    /// Two of a building's corners further apart than `building::REACH_M`.
    BuildingTooWide,
    /// Less floor than `building::MIN_M2`.
    BuildingTooSmall,
    /// A building over another one.
    BuildingOverlap { building: u32 },
    /// No constructed station's platform touches the building.
    BuildingNoStation,
    NoSuchBuilding { building: u32 },
    /// Track drawn from a station has no room past the platform for its junction (T-1155): a
    /// junction, a station or a crossover is in the way.
    BranchNoRoom { node: u32 },
    /// Track or a station upgrade at a port (T-2193): a port has neither.
    NotAtPort { node: u32 },
    /// A port's berth missing, outside the world or too far from its landing (T-2193).
    PortBerth { node: u32 },
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
    /// Positions of crossings removed or added.
    pub crossings: Vec<[f64; 2]>,
}

pub struct Network {
    /// The frame's origin (T-125, `frame.rs`).
    pub origin_lon: f64,
    pub origin_lat: f64,
    /// Which frame the metres are in: `Equirect` for every network made before T-125.
    pub frame_kind: FrameKind,
    frame: Frame,
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
    /// The player typed the station's name (T-2683); no code renames it afterwards.
    pub node_named: Vec<bool>,
    /// The station on the node is constructed (meaningless without a platform).
    pub node_built: Vec<bool>,
    pub node_day: Vec<u32>,
    /// The blueprint station is switched off (T-333).
    pub node_off: Vec<bool>,
    /// The station's upgrades (`StationData::upgrades`).
    pub node_upgrades: Vec<u8>,
    /// A port's berth (`StationData::berth`, T-2193); None for every other node.
    pub node_berth: Vec<Option<[f64; 2]>>,
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
    pub edge_day: Vec<u32>,
    /// The blueprint edge is switched off (T-333).
    pub edge_off: Vec<bool>,
    pub edge_pis: Vec<Span>,
    /// Per edge: the points a T-079 edge was drawn through (`EdgeData::thru`), else empty.
    pub edge_thru: Vec<Vec<Pi>>,
    pub edge_crossover: Vec<bool>,
    pub edge_hsr: Vec<bool>,
    pub edge_sep: Vec<bool>,
    pub edge_len: Vec<f64>,
    pub edge_pieces: Vec<Span>,
    pub edge_vert: Vec<Span>,
    pub edge_water: Vec<Span>,
    /// Derived: chainage spans of ground that is not flat, `[s0, s1, class]` (T-1270).
    pub edge_terrain: Vec<Span>,
    /// Derived: chainages of the roads the alignment crosses, `[s, weight]` (T-1929).
    pub edge_roads: Vec<Span>,
    pub edge_cost: Vec<f64>,
    /// Time to run the edge at its speed limits, no acceleration (for routing).
    pub edge_time: Vec<f64>,
    edge_free: Vec<u32>,
    pub pis: Pool<Pi>,
    pub pieces: Pool<Piece>,
    pub verts: Pool<[f64; 2]>,
    pub water: Pool<[f64; 2]>,
    pub terrain_spans: Pool<[f64; 3]>,
    pub road_spans: Pool<[f64; 2]>,

    pub line_alive: Vec<bool>,
    pub line_day: Vec<u32>,
    pub line_hand: Vec<bool>,
    pub line_name: Vec<String>,
    pub line_symbol: Vec<String>,
    pub next_line_number: u32,
    /// What is stamped on an element constructed now (T-2871): `DAY_UNKNOWN` until the clock
    /// worker sets it (`TrackApi::set_game_day`).
    pub today: u32,
    pub line_colour: Vec<u32>,
    pub line_stops: Vec<Span>,
    pub line_path: Vec<Span>,
    pub line_tph: Vec<[f32; 3]>,
    pub line_trains: Vec<Option<[u32; 3]>>,
    pub line_auto: Vec<Option<[u32; 3]>>,
    /// T-1787: `line_tph` holds settled rates (a settle has run, or a save carried them), so a save
    /// writes them to the bit in `LRT1`
    pub rates_settled: bool,
    /// T-1787: the rates came from a save's `LRT1`, so loading settles from them with an edit's
    /// quick settle, a no-op when nothing changed; without it a load settles fully, as before
    pub rates_from_save: bool,
    /// The lines and levels the last settle left circling (`TrackWorld::settle_counts`): line, level,
    /// rate and round trip bits, trains. Saved with the rates, so a load keeps them settled.
    pub settle_kept: Vec<(u32, usize, u32, u64, u32)>,
    pub line_dwell: Vec<f32>,
    pub line_turn: Vec<f32>,
    pub line_cars: Vec<u8>,
    pub line_fast: Vec<bool>,
    /// Per line: its bus route (T-1464), None for a rail line.
    pub line_bus: Vec<Option<BusRoute>>,
    /// Per line: its ferry route (T-2193), None for any other line.
    pub line_ferry: Vec<Option<FerryRoute>>,
    pub line_group: Vec<u32>,
    pub line_group_name: Vec<String>,
    pub line_group_look: Vec<GroupLook>,
    /// T-1813: the Lines and Stations tabs' hand order, by line and by node id: 1 first, 0 for none.
    /// Outside the undo history, like the sort it replaces.
    pub line_rank: Vec<u32>,
    pub node_rank: Vec<u32>,
    /// T-1925: by line id, how many seconds after its pattern's own times each level's departures
    /// leave, so lines sharing track or a flat junction do not leave together (`spacing::place`);
    /// shorter than a missing entry is 0. Outside the undo history; saved in `LPH1`.
    pub line_phase: Vec<[f32; 3]>,
    /// T-1925: `line_phase` is placed (a save carried it, or a load placed every line), so a
    /// reload keeps it rather than placing every line
    pub phases_placed: bool,
    /// Derived: the path uses a deleted edge or no longer fits the stops.
    pub line_broken: Vec<bool>,
    line_free: Vec<u32>,
    pub stops: Pool<u32>,
    pub paths: Pool<u32>,

    /// Stations the player linked into one complex by hand (T-197): node pairs, smaller id first.
    /// Not part of the undo history (a setting, like fares); a pair whose node is no longer a
    /// station is ignored (`links_live`) and not saved.
    pub links: Vec<[u32; 2]>,
    /// The station buildings (T-2217, `building.rs`) by id, None where one was removed. Outside the
    /// undo history, like the links; saved in `BLD1`, renumbered densely.
    pub buildings: Vec<Option<super::building::Building>>,

    pub crossings: Vec<Crossing>,
    pub grid: Grid,
    /// CSR by edge id: lines whose path uses the edge, ascending.
    pub edge_lines_off: Vec<u32>,
    pub edge_lines: Vec<u32>,
    pub water_mask: Box<dyn WaterMask>,
    /// The terrain grids installed (T-1270): from the clock worker, not saved, and carried over to a
    /// loaded network like the water mask.
    pub terrain: Terrain,
    /// The road tiles installed (T-1929): from the clock worker, not saved, and carried over to a
    /// loaded network like the terrain.
    pub roads: Roads,
    /// The street speeds installed (T-1577, `streets.rs`): from the clock worker, not saved, and carried
    /// over to a loaded network like the terrain.
    pub streets: super::streets::Streets,
    /// Each region's local time (T-309, `localtime.rs`): from the clock worker, not saved, and
    /// carried over to a loaded network like the water mask.
    pub time_zones: super::localtime::TimeZones,
    pub touch: Touch,
    /// T-1382: the nodes whose station name was set since `take_name_changes`, for the clock
    /// worker's cache of names; `names_all` while nothing has been taken from this network yet.
    names_changed: Vec<u32>,
    names_all: bool,
    /// Edge prices worked out again outside an edit (water or terrain installed), since `take_repriced`.
    repriced: bool,
    /// T-1540: whether each node lies on water, kept as (x, y, mask epoch, answer) and used only
    /// while the node is where it was and the mask the one it was asked of (`node_is_wet`).
    node_wet: std::cell::RefCell<Vec<(f64, f64, u32, bool)>>,
    /// Counts the masks installed, so a kept answer from an earlier mask is not used.
    mask_epoch: u32,
    /// Off: every node is asked of the mask each time (the rows check compares the two).
    pub wet_cache: bool,
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

/// T-798: a line whose last stop is its first is a loop. A circle carries straight on through
/// the first stop; a balloon turns back there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopKind {
    #[default]
    None,
    Circle,
    Balloon,
}

pub fn path_edge(p: u32) -> u32 {
    p >> 1
}
pub fn path_dir(p: u32) -> u32 {
    p & 1
}

/// Where each stop falls in a path's node list: the earliest matching that stops at every index
/// in `must` (the nodes where the path turns back, T-797: a train reverses only at a stop). None
/// if there is none. Without `must`, the plain greedy match.
pub fn match_stops(nodes: &[u32], stops: &[u32], must: &[usize]) -> Option<Vec<usize>> {
    let (n, m) = (nodes.len(), stops.len());
    if must.is_empty() {
        let mut out = Vec::with_capacity(m);
        let mut k = 0;
        for &s in stops {
            while k < n && nodes[k] != s {
                k += 1;
            }
            if k == n {
                return None;
            }
            out.push(k);
            k += 1;
        }
        return Some(out);
    }
    if m == 0 || n == 0 {
        return None;
    }
    let mut is_must = vec![false; n];
    for &k in must {
        if k < n {
            is_must[k] = true;
        }
    }
    // suf[j * n + k]: stop j at node k, and stops j.. can all be matched after it, no must skipped
    let mut suf = vec![false; m * n];
    suf[(m - 1) * n + n - 1] = nodes[n - 1] == stops[m - 1];
    for j in (0..m - 1).rev() {
        let mut avail = false;
        for k in (0..n).rev() {
            suf[j * n + k] = avail && nodes[k] == stops[j];
            let next = suf[(j + 1) * n + k];
            avail = if is_must[k] { next } else { avail || next };
        }
    }
    if !suf[0] {
        return None;
    }
    let mut out = vec![0usize];
    for j in 1..m {
        let prev = out[j - 1];
        let mut k = prev + 1;
        while k < n && !suf[j * n + k] {
            if is_must[k] {
                return None;
            }
            k += 1;
        }
        if k == n {
            return None;
        }
        out.push(k);
    }
    Some(out)
}

impl Network {
    /// The network's frame (origin and kind).
    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// T-2607: the trains line `l` runs at level `lev` over a round trip of `trip_s`: its count, but
    /// on a line keeping a headway no more than the headway needs (a fraction), the time left over
    /// spent waiting at the ends, so the headway set is the headway run.
    pub fn running_trains(&self, l: usize, lev: usize, trip_s: f64) -> f64 {
        let n = self.line_trains.get(l).copied().flatten().map_or(0.0, |c| c[lev] as f64);
        match self.line_auto.get(l).copied().flatten() {
            Some(a) if a[lev] > 0 && trip_s > 0.0 => n.min(trip_s / a[lev] as f64),
            _ => n,
        }
    }

    /// T-1925: a line's departure phase per level, seconds (`line_phase`; 0 where none is set).
    pub fn line_phase_of(&self, l: u32) -> [f32; 3] {
        self.line_phase.get(l as usize).copied().unwrap_or([0.0; 3])
    }
    /// Seconds a line's local time runs ahead of the game clock (T-309): the region of its first
    /// stop's. 0 without a time zone table or stops.
    pub fn line_shift(&self, l: u32) -> f64 {
        if !self.time_zones.set || !self.line_ok(l) {
            return 0.0;
        }
        if let Some(p) = self.route_points(l) {
            return p.first().map_or(0.0, |p| self.line_shift_at(p[0], p[1]));
        }
        let Some(&s) = self.line_stops(l).first() else { return 0.0 };
        self.line_shift_at(self.node_x[s as usize], self.node_y[s as usize])
    }
    /// Seconds local time runs ahead of the game clock at a plane point (T-309).
    pub fn line_shift_at(&self, x: f64, y: f64) -> f64 {
        if !self.time_zones.set {
            return 0.0;
        }
        let (lon, lat) = self.frame.to_lonlat(x, y);
        self.time_zones.shift_at(lon, lat)
    }

    /// Plane metres per ground metre at plane `y`: `cosh(y / R)` in the Mercator frame, exactly 1
    /// in the others (frame.rs). Metre figures entering the plane are multiplied by it.
    #[inline]
    pub fn k_at(&self, y: f64) -> f64 {
        if self.frame.scaled() {
            super::frame::merc_k(y)
        } else {
            1.0
        }
    }

    /// Fit an alignment in this network's frame (ground lengths in the Mercator frame, geom.rs).
    pub fn fit(&self, v: &[Pi]) -> geom::Fit {
        geom::fit_in(v, self.frame.scaled())
    }

    /// True if a node may be placed here: anywhere in the unscaled frames; in the Mercator frame
    /// within 85 degrees of the equator and not across the antimeridian (docs/world.md).
    pub fn inside_world(&self, x: f64, y: f64) -> bool {
        !self.frame.scaled() || super::frame::merc_inside(x, y)
    }

    /// An empty network in the given frame kind.
    pub fn new_in(kind: FrameKind, origin_lon: f64, origin_lat: f64, right_hand: bool) -> Network {
        let mut n = Network::new(origin_lon, origin_lat, right_hand);
        n.frame_kind = kind;
        n.frame = Frame::new(kind, origin_lon, origin_lat);
        n
    }

    /// An empty network in the legacy equirectangular frame.
    pub fn new(origin_lon: f64, origin_lat: f64, right_hand: bool) -> Network {
        Network {
            origin_lon,
            origin_lat,
            frame_kind: FrameKind::Equirect,
            frame: Frame::new(FrameKind::Equirect, origin_lon, origin_lat),
            right_hand,
            node_alive: vec![],
            node_x: vec![],
            node_y: vec![],
            node_level: vec![],
            node_flying: vec![],
            node_platform: vec![],
            node_name: vec![],
            node_named: vec![],
            node_built: vec![],
            node_day: vec![],
            node_off: vec![],
            node_upgrades: vec![],
            node_berth: vec![],
            node_ports: vec![],
            node_heading: vec![],
            node_free: vec![],
            edge_alive: vec![],
            edge_a: vec![],
            edge_b: vec![],
            edge_tracks: vec![],
            edge_built: vec![],
            edge_day: vec![],
            edge_off: vec![],
            edge_pis: vec![],
            edge_thru: vec![],
            edge_crossover: vec![],
            edge_hsr: vec![],
            edge_sep: vec![],
            edge_len: vec![],
            edge_pieces: vec![],
            edge_vert: vec![],
            edge_water: vec![],
            edge_terrain: vec![],
            edge_roads: vec![],
            edge_cost: vec![],
            edge_time: vec![],
            edge_free: vec![],
            pis: Pool::default(),
            pieces: Pool::default(),
            verts: Pool::default(),
            water: Pool::default(),
            terrain_spans: Pool::default(),
            road_spans: Pool::default(),
            line_alive: vec![],
            line_day: vec![],
            line_hand: vec![],
            line_name: vec![],
            line_symbol: vec![],
            next_line_number: 1,
            today: DAY_UNKNOWN,
            line_colour: vec![],
            line_stops: vec![],
            line_path: vec![],
            line_tph: vec![],
            line_trains: vec![],
            line_auto: vec![],
            rates_settled: false,
            rates_from_save: false,
            settle_kept: vec![],
            line_dwell: vec![],
            line_turn: vec![],
            line_cars: vec![],
            line_fast: vec![],
            line_bus: vec![],
            line_ferry: vec![],
            line_group: vec![],
            line_group_name: vec![],
            line_group_look: vec![],
            line_rank: vec![],
            node_rank: vec![],
            line_phase: vec![],
            phases_placed: false,
            line_broken: vec![],
            line_free: vec![],
            stops: Pool::default(),
            paths: Pool::default(),
            links: vec![],
            buildings: vec![],
            crossings: vec![],
            grid: Grid::default(),
            edge_lines_off: vec![0],
            edge_lines: vec![],
            water_mask: Box::new(NoWater),
            terrain: Terrain::default(),
            roads: Roads::default(),
            streets: Default::default(),
            time_zones: Default::default(),
            touch: Touch::default(),
            names_changed: vec![],
            names_all: true,
            repriced: false,
            node_wet: Default::default(),
            mask_epoch: 0,
            wet_cache: true,
        }
    }

    /// Whether the node lies on water (`water_mask.is_water` at it), asked of the mask once while the
    /// node and the mask stay as they are: stations are priced on every edit and every state.
    pub fn node_is_wet(&self, n: usize) -> bool {
        let (x, y) = (self.node_x[n], self.node_y[n]);
        if !self.wet_cache {
            return self.water_mask.is_water(x, y);
        }
        let mut kept = self.node_wet.borrow_mut();
        if kept.len() <= n {
            kept.resize(n + 1, (f64::NAN, f64::NAN, u32::MAX, false));
        }
        let k = &mut kept[n];
        if k.2 == self.mask_epoch && k.0 == x && k.1 == y {
            return k.3;
        }
        let wet = self.water_mask.is_water(x, y);
        *k = (x, y, self.mask_epoch, wet);
        wet
    }

    pub fn take_repriced(&mut self) -> bool {
        std::mem::take(&mut self.repriced)
    }

    pub fn repriced(&self) -> bool {
        self.repriced
    }

    /// The nodes whose station name was set (or cleared) since the last call, ascending; `[u32::MAX]`
    /// when every name is new to the caller (a network not asked before).
    pub fn take_name_changes(&mut self) -> Vec<u32> {
        let mut v = std::mem::take(&mut self.names_changed);
        if std::mem::replace(&mut self.names_all, false) {
            return vec![u32::MAX];
        }
        v.sort_unstable();
        v.dedup();
        v
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
    /// A live node with a station on it.
    pub fn is_station(&self, n: u32) -> bool {
        self.node_ok(n) && self.node_platform[n as usize] > 0
    }
    /// The player's links (T-197) whose two ends are both stations now.
    pub fn links_live(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.links.iter().copied().filter(|&[a, b]| self.is_station(a) && self.is_station(b))
    }
    /// Link two stations into one complex (T-197). False if either is not a station or they are
    /// the same; linking a linked pair again changes nothing.
    pub fn link(&mut self, a: u32, b: u32) -> bool {
        if a == b || !self.is_station(a) || !self.is_station(b) {
            return false;
        }
        let p = [a.min(b), a.max(b)];
        if !self.links.contains(&p) {
            self.links.push(p);
        }
        true
    }
    /// Remove the link between two stations, if there is one. True if there was.
    pub fn unlink(&mut self, a: u32, b: u32) -> bool {
        let p = [a.min(b), a.max(b)];
        let n = self.links.len();
        self.links.retain(|&q| q != p);
        self.links.len() != n
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

    fn grow_nodes(&mut self) {
        self.node_alive.push(false);
        self.node_x.push(0.0);
        self.node_y.push(0.0);
        self.node_level.push(0);
        self.node_flying.push(false);
        self.node_platform.push(0);
        self.node_name.push(String::new());
        self.node_named.push(false);
        self.node_built.push(false);
        self.node_day.push(0);
        self.node_off.push(false);
        self.node_upgrades.push(0);
        self.node_berth.push(None);
        self.node_ports.push(Ports::default());
        self.node_heading.push(0.0);
    }
    fn grow_edges(&mut self) {
        self.edge_alive.push(false);
        self.edge_a.push(0);
        self.edge_b.push(0);
        self.edge_tracks.push(2);
        self.edge_built.push(false);
        self.edge_day.push(0);
        self.edge_off.push(false);
        self.edge_pis.push(Span::default());
        self.edge_thru.push(vec![]);
        self.edge_crossover.push(false);
        self.edge_hsr.push(false);
        self.edge_sep.push(false);
        self.edge_len.push(0.0);
        self.edge_pieces.push(Span::default());
        self.edge_vert.push(Span::default());
        self.edge_water.push(Span::default());
        self.edge_terrain.push(Span::default());
        self.edge_roads.push(Span::default());
        self.edge_cost.push(0.0);
        self.edge_time.push(0.0);
        self.edge_lines_off.push(*self.edge_lines_off.last().unwrap());
    }
    fn grow_lines(&mut self) {
        self.line_alive.push(false);
        self.line_name.push(String::new());
        self.line_symbol.push(String::new());
        self.line_colour.push(0);
        self.line_stops.push(Span::default());
        self.line_path.push(Span::default());
        self.line_tph.push([0.0; 3]);
        self.line_trains.push(None);
        self.line_auto.push(None);
        self.line_dwell.push(DEFAULT_DWELL_S);
        self.line_turn.push(DEFAULT_TURNAROUND_S);
        self.line_cars.push(0);
        self.line_fast.push(false);
        self.line_bus.push(None);
        self.line_ferry.push(None);
        self.line_group.push(0);
        self.line_group_name.push(String::new());
        self.line_group_look.push(GroupLook::default());
        self.line_broken.push(false);
        self.line_day.push(0);
        self.line_hand.push(false);
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
            EdgeData { a: self.edge_a[i], b: self.edge_b[i], tracks: self.edge_tracks[i], pis: self.pis.get(self.edge_pis[i]).to_vec(), built: self.edge_built[i], off: self.edge_off[i], thru: self.edge_thru[i].clone(), crossover: self.edge_crossover[i], hsr: self.edge_hsr[i], sep: self.edge_sep[i], day: self.edge_day[i] }
        })
    }
    pub fn station_data(&self, n: u32) -> Option<StationData> {
        (self.node_ok(n) && self.node_platform[n as usize] > 0)
            .then(|| StationData { platform: self.node_platform[n as usize], name: self.node_name[n as usize].clone(), named: self.node_named[n as usize], built: self.node_built[n as usize], off: self.node_off[n as usize], upgrades: self.node_upgrades[n as usize], berth: self.node_berth[n as usize], day: self.node_day[n as usize] })
    }
    pub fn line_data(&self, l: u32) -> Option<LineData> {
        self.line_ok(l).then(|| {
            let i = l as usize;
            LineData {
                name: self.line_name[i].clone(),
                symbol: self.line_symbol[i].clone(),
                colour: self.line_colour[i],
                stops: self.stops.get(self.line_stops[i]).to_vec(),
                path: self.paths.get(self.line_path[i]).to_vec(),
                tph: self.line_tph[i],
                trains: self.line_trains[i],
                auto: self.line_auto[i],
                dwell_s: self.line_dwell[i],
                turnaround_s: self.line_turn[i],
                cars: self.line_cars[i],
                fast: self.line_fast[i],
                bus: self.line_bus[i].clone(),
                ferry: self.line_ferry[i].clone(),
                group: self.line_group[i],
                group_name: self.line_group_name[i].clone(),
                group_look: self.line_group_look[i].clone(),
                day: self.line_day[i],
                hand: self.line_hand[i],
            }
        })
    }
    /// The car's local street speed at a point of the plane, km/h (T-1577): open country's where no
    /// street grid covers it.
    pub fn car_kmh_at(&self, x: f64, y: f64) -> f64 {
        let (lon, lat) = self.frame.to_lonlat(x, y);
        self.streets.car_kmh_at(lon, lat).unwrap_or(super::streets::CAR_TOWN_KMH)
    }
    /// A live bus line (T-1464).
    pub fn is_bus(&self, l: u32) -> bool {
        self.line_ok(l) && self.line_bus[l as usize].is_some()
    }
    /// A rail line has no route of its own to build; a bus or ferry line's route is planned until constructed (T-2871).
    pub fn route_built(&self, l: u32) -> bool {
        let i = l as usize;
        match (&self.line_bus[i], &self.line_ferry[i]) {
            (Some(r), _) => r.built,
            (_, Some(r)) => r.built,
            _ => true,
        }
    }
    /// The live bus lines.
    pub fn bus_lines(&self) -> Vec<u32> {
        (0..self.line_count() as u32).filter(|&l| self.is_bus(l)).collect()
    }
    /// A port (T-2193): a station with a berth, on no track.
    pub fn is_port(&self, n: u32) -> bool {
        self.is_station(n) && self.node_berth[n as usize].is_some()
    }
    /// A live ferry line (T-2193).
    pub fn is_ferry(&self, l: u32) -> bool {
        self.line_ok(l) && self.line_ferry[l as usize].is_some()
    }
    /// The live ferry lines.
    pub fn ferry_lines(&self) -> Vec<u32> {
        (0..self.line_count() as u32).filter(|&l| self.is_ferry(l)).collect()
    }
    /// A line with no track, a bus or a ferry line: the drawn route's points.
    pub fn route_points(&self, l: u32) -> Option<&[[f64; 2]]> {
        let i = l as usize;
        if !self.line_ok(l) {
            return None;
        }
        match (self.line_bus.get(i).and_then(|b| b.as_ref()), self.line_ferry.get(i).and_then(|f| f.as_ref())) {
            (Some(r), _) => Some(&r.points),
            (None, Some(r)) => Some(&r.points),
            _ => None,
        }
    }
    /// A ferry route's length, ground m (0 for any other line).
    pub fn ferry_m(&self, l: u32) -> f64 {
        match self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            Some(r) if self.line_ok(l) => super::ferry::ground_cum(self, &r.points).last().copied().unwrap_or(0.0),
            _ => 0.0,
        }
    }
    /// A station's price, US$M: a port's flat price, any other by its level, platform, tracks and the
    /// water under it (`wet`, asked only for those).
    fn station_price(&self, n: usize, wet: impl FnOnce() -> bool) -> f64 {
        if self.node_berth[n].is_some() {
            PORT_COST
        } else {
            cost::station_cost(self.node_level[n], self.node_platform[n], self.node_tracks(n as u32), wet())
        }
    }
    /// A bus route's road length, m (its drawn length, times `bus::ROAD_FACTOR` where it is straight).
    pub fn bus_road_m(&self, l: u32) -> f64 {
        match self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            Some(r) if self.line_ok(l) => super::bus::Poly::new(self, &r.points, &r.kinds).road_len(),
            _ => 0.0,
        }
    }
    /// Build cost of every bus route, US$M: all of them, or the constructed ones.
    fn bus_cost(&self, built: bool) -> f64 {
        (0..self.line_count() as u32)
            .filter(|&l| self.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| self.line_ok(l) && (!built || r.built)))
            .map(|l| self.bus_road_m(l) / 1000.0 * BUS_ROUTE_COST_PER_KM)
            .sum()
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
    pub fn edge_terrain(&self, e: u32) -> &[[f64; 3]] {
        self.terrain_spans.get(self.edge_terrain[e as usize])
    }
    pub fn edge_roads(&self, e: u32) -> &[[f64; 2]] {
        self.road_spans.get(self.edge_roads[e as usize])
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
    /// High-speed track meets the node (T-2231: passing tracks there are priced as it).
    pub fn node_hsr(&self, n: u32) -> bool {
        self.node_ports[n as usize].ends().iter().any(|&c| self.edge_hsr[(c >> 1) as usize])
    }
    /// A station's upgrades, US$M (`upgrades::cost` at its level, water, platform and track).
    pub fn station_upgrades_price(&self, n: u32, bits: u8, wet: bool) -> f64 {
        let i = n as usize;
        super::upgrades::cost(bits, self.node_level[i], wet, self.node_platform[i], self.node_hsr(n))
    }
    /// The speed limit an edge puts on every piece of it, m/s: `V_CROSSOVER` on a crossover, 320
    /// km/h on high-speed track, 160 on any other (T-817).
    pub fn v_cap(&self, e: u32) -> f64 {
        if self.edge_crossover[e as usize] { V_CROSSOVER } else if self.edge_hsr[e as usize] { V_HSR } else { V_TOP }
    }
    /// A line's trains' top speed, m/s: high-speed trains or ordinary ones (T-817).
    pub fn line_top(&self, l: u32) -> f64 {
        if self.line_fast[l as usize] { V_HSR } else { V_TOP }
    }
    /// Edge ends at a node that are not crossovers and pass `keep`. A crossover pays for its own
    /// turnouts (T-338), so the nodes at its ends are priced as junctions only with three others.
    fn priced_ends(&self, n: u32, keep: impl Fn(usize) -> bool) -> usize {
        self.node_ports[n as usize].ends().iter().map(|&c| (c >> 1) as usize).filter(|&e| !self.edge_crossover[e] && keep(e)).count()
    }
    /// A flat crossing between the two diagonals of one crossover is part of its price.
    fn priced_crossing(&self, x: &Crossing) -> bool {
        !(self.edge_crossover[x.e1 as usize] && self.edge_crossover[x.e2 as usize])
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

    /// T-2871: an element's build day after an edit, given what it held before (0 if nothing or a
    /// blueprint): a set day stands, so undo and redo restore exactly; a built element already
    /// dated keeps its date; only construction takes today.
    fn day_for(&self, built: bool, given: u32, before: u32) -> u32 {
        match (built, given, before) {
            (false, ..) => 0,
            (true, 0, 0) => self.today,
            (true, 0, d) => d,
            (true, g, _) => g,
        }
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
            if (d.tracks != 1 && d.tracks != 2) || (d.hsr && (d.tracks != 2 || d.crossover)) {
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
                self.edge_day[i] = self.day_for(d.built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.edge_built[i] = d.built;
                self.edge_off[i] = d.off && !d.built;
                self.edge_pis[i] = self.pis.replace(self.edge_pis[i], &d.pis);
                self.edge_thru[i] = d.thru;
                self.edge_crossover[i] = d.crossover;
                self.edge_hsr[i] = d.hsr;
                self.edge_sep[i] = d.sep;
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
                self.edge_day[i] = 0;
                self.edge_off[i] = false;
                self.pis.free(self.edge_pis[i]);
                self.edge_pis[i] = Span::default();
                self.edge_thru[i] = vec![];
                self.edge_crossover[i] = false;
                self.edge_hsr[i] = false;
                self.edge_sep[i] = false;
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
                self.node_day[i] = 0;
                self.node_name[i].clear();
                self.node_named[i] = false;
                self.node_built[i] = false;
                self.node_off[i] = false;
                self.node_upgrades[i] = 0;
                self.node_berth[i] = None;
            }
            Some(d) => {
                self.node_day[i] = self.day_for(d.built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.node_platform[i] = d.platform;
                self.node_name[i] = d.name;
                self.node_named[i] = d.named;
                self.node_built[i] = d.built;
                self.node_off[i] = d.off && !d.built;
                self.node_upgrades[i] = d.upgrades & super::upgrades::ALL;
                self.node_berth[i] = d.berth;
            }
        }
        self.touch.stations.push(node);
        if self.names_changed.len() > 1 << 20 {
            self.names_changed.clear();
            self.names_all = true;
        }
        self.names_changed.push(node);
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
                let built = d.bus.as_ref().map(|r| r.built).or(d.ferry.as_ref().map(|r| r.built)).unwrap_or(true);
                self.line_day[i] = self.day_for(built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.line_alive[i] = true;
                self.line_name[i] = d.name;
                self.line_symbol[i] = d.symbol;
                self.line_colour[i] = d.colour;
                self.line_stops[i] = self.stops.replace(self.line_stops[i], &d.stops);
                self.line_path[i] = self.paths.replace(self.line_path[i], &d.path);
                self.line_tph[i] = d.tph;
                self.line_trains[i] = d.trains;
                self.line_auto[i] = d.auto.filter(|_| d.trains.is_some());
                self.line_dwell[i] = d.dwell_s;
                self.line_turn[i] = d.turnaround_s;
                self.line_cars[i] = d.cars;
                self.line_fast[i] = d.fast;
                self.line_hand[i] = d.hand && d.bus.is_none() && d.ferry.is_none();
                self.line_bus[i] = d.bus;
                self.line_ferry[i] = d.ferry;
                self.line_group[i] = d.group;
                self.line_group_name[i] = if d.group == 0 { String::new() } else { d.group_name };
                self.line_group_look[i] = if d.group == 0 { GroupLook::default() } else { d.group_look };
            }
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                self.line_alive[i] = false;
                self.line_day[i] = 0;
                self.line_hand[i] = false;
                self.line_bus[i] = None;
                self.line_ferry[i] = None;
                self.line_group[i] = 0;
                self.line_group_look[i] = GroupLook::default();
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

    /// A line's trains per hour as the settle starts from them (the counts' rates follow).
    pub fn set_tph(&mut self, id: u32, tph: [f32; 3]) -> Result<[f32; 3], Issue> {
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

    /// T-625: switch a line between a hand-set count and a headway per level, with the counts it
    /// starts from. Returns the line's previous headways and counts.
    pub fn set_auto(&mut self, id: u32, auto: Option<[u32; 3]>, trains: Option<[u32; 3]>) -> Result<(Option<[u32; 3]>, Option<[u32; 3]>), Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        let old = (self.line_auto[id as usize], self.line_trains[id as usize]);
        self.line_trains[id as usize] = trains;
        self.line_auto[id as usize] = auto.filter(|_| trains.is_some());
        self.touch.schedules.push(id);
        Ok(old)
    }

    pub fn set_water_mask(&mut self, mask: Box<dyn WaterMask>) {
        self.water_mask = mask;
        self.mask_epoch = self.mask_epoch.wrapping_add(1);
        for e in 0..self.edge_count() as u32 {
            if self.edge_ok(e) {
                self.touch.edges.push(e);
            }
        }
    }

    /// A water mask that adds water only inside `b` (`[xmin, ymin, xmax, ymax]`) to the one in
    /// use: only the edges reaching into `b` have their water and price worked out again, as
    /// `derive` does (T-1039: deriving the whole network for each zone's mask took 100 ms and more
    /// a mask with long track drawn). Nothing else an edge derives depends on water.
    pub fn add_water(&mut self, mask: Box<dyn WaterMask>, b: [f64; 4]) {
        self.water_mask = mask;
        self.mask_epoch = self.mask_epoch.wrapping_add(1);
        self.rework_ground(b, Ground::Water);
    }

    /// T-1270: a terrain grid added to those in use, its bounds `b` in the network frame: only the
    /// edges reaching into `b` have their terrain and price worked out again, as for water. Nothing
    /// but the price depends on terrain.
    pub fn add_terrain(&mut self, grid: std::sync::Arc<cost::ClassGrid>, b: [f64; 4]) {
        self.terrain.grids.push(grid);
        self.rework_ground(b, Ground::Terrain);
    }

    /// T-1929: a road tile added to those in use (`roads::Roads::key`), its bounds `b` in the
    /// network frame. T-2230: each crossing is found in one tile, so only the new tile's are looked
    /// for, on the stretches of the edges near `b` (the grid's), and added to those each edge had; an
    /// edge with none new keeps its price. Only prices change, which no line's rows hold, so
    /// `repriced` is left alone. (Every edge worked out again took 2-11 ms a tile on New York's real save.)
    pub fn add_roads(&mut self, key: u32, tile: std::sync::Arc<roads::RoadTile>, b: [f64; 4]) {
        self.roads.tiles.insert(key, tile.clone());
        let (tx, ty) = (key / roads::TILES, key % roads::TILES);
        let frame = self.frame;
        let edges = self.grid.in_box(b).unwrap_or_else(|| (0..self.edge_count() as u32).collect());
        let mut cross = vec![];
        for e in edges {
            let i = e as usize;
            // (as `derive`: a crossover's price has no crossings, and track off the ground keeps none)
            if !self.edge_ok(e) || self.edge_crossover[i] || roads::off_ground(self.verts.get(self.edge_vert[i])) {
                continue;
            }
            let pieces = self.pieces.get(self.edge_pieces[i]);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in pieces {
                let q = p.bbox();
                if q[0] <= b[2] && q[2] >= b[0] && q[1] <= b[3] && q[3] >= b[1] {
                    lo = lo.min(p.s0);
                    hi = hi.max(p.s0 + p.len);
                }
            }
            cross.clear();
            roads::crossings_in_tile(self.edge_len[i], lo, hi, |s| { let (x, y, _) = geom::pos_at(pieces, s); frame.to_lonlat(x, y) }, &tile, tx, ty, &mut cross);
            if cross.is_empty() {
                continue;
            }
            let mut all = self.road_spans.get(self.edge_roads[i]).to_vec();
            all.extend_from_slice(&cross);
            all.sort_by(|x, y| x[0].total_cmp(&y[0]));
            self.edge_roads[i] = self.road_spans.replace(self.edge_roads[i], &all);
            self.edge_cost[i] = self.edge_price(i, self.verts.get(self.edge_vert[i]), self.water.get(self.edge_water[i]), self.terrain_spans.get(self.edge_terrain[i]), self.road_spans.get(self.edge_roads[i]));
        }
    }

    /// Work out again the water or the terrain, and the price, of the edges reaching into `b`.
    fn rework_ground(&mut self, b: [f64; 4], what: Ground) {
        self.repriced = true;
        let frame = self.frame;
        for e in 0..self.edge_count() as u32 {
            let i = e as usize;
            if !self.edge_ok(e) {
                continue;
            }
            let pieces = self.pieces.get(self.edge_pieces[i]);
            if !pieces.iter().any(|p| { let q = p.bbox(); q[0] <= b[2] && q[2] >= b[0] && q[1] <= b[3] && q[3] >= b[1] }) {
                continue;
            }
            let pieces = pieces.to_vec();
            match what {
                Ground::Water => {
                    let mut wet = vec![];
                    cost::water_spans(self.edge_len[i], |s| { let (x, y, _) = geom::pos_at(&pieces, s); (x, y) }, self.water_mask.as_ref(), &mut wet);
                    self.edge_water[i] = self.water.replace(self.edge_water[i], &wet);
                }
                Ground::Terrain => {
                    let mut ground = vec![];
                    cost::terrain_spans(self.edge_len[i], |s| { let (x, y, _) = geom::pos_at(&pieces, s); frame.to_lonlat(x, y) }, &self.terrain, &mut ground);
                    self.edge_terrain[i] = self.terrain_spans.replace(self.edge_terrain[i], &ground);
                }
            }
            self.edge_cost[i] = self.edge_price(i, self.verts.get(self.edge_vert[i]), self.water.get(self.edge_water[i]), self.terrain_spans.get(self.edge_terrain[i]), self.road_spans.get(self.edge_roads[i]));
        }
    }

    /// An edge's price from its derived parts: a crossover's switches, else its track and the roads
    /// it crosses at ground level (T-1929).
    fn edge_price(&self, i: usize, vert: &[[f64; 2]], wet: &[[f64; 2]], terrain: &[[f64; 3]], cross: &[[f64; 2]]) -> f64 {
        if self.edge_crossover[i] {
            cost::junction_cost(self.node_level[self.edge_a[i] as usize], false, !wet.is_empty())
        } else {
            cost::track_cost_t(self.edge_len[i], vert, wet, terrain, self.edge_tracks[i], self.edge_hsr[i]) + roads::charge(vert, cross).1
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
            }
            if !self.edge_alive[i] {
                self.pieces.free(old);
                self.verts.free(self.edge_vert[i]);
                self.water.free(self.edge_water[i]);
                self.terrain_spans.free(self.edge_terrain[i]);
                self.road_spans.free(self.edge_roads[i]);
                self.edge_pieces[i] = Span::default();
                self.edge_vert[i] = Span::default();
                self.edge_water[i] = Span::default();
                self.edge_terrain[i] = Span::default();
                self.edge_roads[i] = Span::default();
                self.edge_len[i] = 0.0;
                self.edge_cost[i] = 0.0;
                self.edge_time[i] = 0.0;
                continue;
            }
            let f = self.fit(&self.edge_vertices(e));
            let mut wet = vec![];
            cost::water_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); (x, y) }, self.water_mask.as_ref(), &mut wet);
            let mut ground = vec![];
            let frame = self.frame;
            cost::terrain_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.terrain, &mut ground);
            let mut cross = vec![];
            if !self.edge_crossover[i] && !roads::off_ground(&f.vert) {
                roads::crossings(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.roads, &mut cross);
            }
            self.edge_len[i] = f.len;
            self.edge_cost[i] = self.edge_price(i, &f.vert, &wet, &ground, &cross);
            self.edge_terrain[i] = self.terrain_spans.replace(self.edge_terrain[i], &ground);
            self.edge_roads[i] = self.road_spans.replace(self.edge_roads[i], &cross);
            let cap = self.v_cap(e);
            self.edge_time[i] = f.pieces.iter().map(|p| p.len / p.v_curve().min(cap)).sum::<f64>().max(1e-3);
            self.edge_pieces[i] = self.pieces.replace(old, &f.pieces);
            self.edge_vert[i] = self.verts.replace(self.edge_vert[i], &f.vert);
            self.edge_water[i] = self.water.replace(self.edge_water[i], &wet);
            self.grid.insert(e, &f.pieces);
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
        if self.edge_sep[ie] && self.edge_sep[i_f] {
            return;
        }
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
            if shared.iter().any(|&n| (self.node_x[n as usize] - x).hypot(self.node_y[n as usize] - y) < SHARED_NODE_CLEAR * self.k_at(y)) {
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
        if self.terrain_spans.needs_compact() {
            self.terrain_spans.compact(&mut self.edge_terrain);
        }
        if self.road_spans.needs_compact() {
            self.road_spans.compact(&mut self.edge_roads);
        }
        if self.stops.needs_compact() {
            self.stops.compact(&mut self.line_stops);
        }
        if self.paths.needs_compact() {
            self.paths.compact(&mut self.line_path);
        }
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
        let verts = self.edge_vertices(e);
        if let Some(p) = verts.iter().find(|p| !self.inside_world(p.x, p.y)) {
            out.push(Issue::OutsideWorld { x: p.x, y: p.y });
        }
        let f = self.fit(&verts);
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
        if !self.edge_water(e).is_empty() && !self.edge_crossover[e as usize] && self.bridged_water_through(e) > WATER_SPAN_MAX {
            out.push(Issue::WaterTooLong { edge: e });
        }
    }

    /// The parts of an edge's water spans that are not bored tunnel (level -1 or above), as
    /// chainage intervals sampled every `BRIDGED_STEP` (T-1930).
    fn bridged_water(&self, e: u32) -> Vec<[f64; 2]> {
        const BRIDGED_STEP: f64 = 25.0;
        let vert = self.edge_vert(e);
        let mut out: Vec<[f64; 2]> = vec![];
        for w in self.edge_water(e) {
            let n = ((w[1] - w[0]) / BRIDGED_STEP).ceil().max(1.0) as usize;
            let step = (w[1] - w[0]) / n as f64;
            let mut open: Option<f64> = None;
            for j in 0..n {
                let bridged = cost::level_of(geom::height_at(vert, w[0] + (j as f64 + 0.5) * step)) >= -1;
                match (bridged, open) {
                    (true, None) => open = Some(w[0] + j as f64 * step),
                    (false, Some(s0)) => {
                        out.push([s0, w[0] + j as f64 * step]);
                        open = None;
                    }
                    _ => {}
                }
            }
            if let Some(s0) = open {
                out.push([s0, w[1]]);
            }
        }
        out
    }

    /// The longest stretch of viaduct or cut and cover over unbroken water that runs through edge
    /// `e`, m, counting the same stretch on the edges meeting it at a node in the water (T-1930).
    /// Stops adding past `WATER_SPAN_MAX`.
    fn bridged_water_through(&self, e: u32) -> f64 {
        const TOL: f64 = 1.0;
        let len = self.edge_len[e as usize];
        let mut best = 0.0f64;
        for r in self.bridged_water(e) {
            let mut total = r[1] - r[0];
            for (which, touches) in [(0u32, r[0] <= TOL), (1, r[1] >= len - TOL)] {
                if touches && total <= WATER_SPAN_MAX {
                    total += self.bridged_water_beyond(e, which, &mut vec![e], WATER_SPAN_MAX - total);
                }
            }
            best = best.max(total);
        }
        best
    }

    /// The most water run an end of edge `e` can add, m: of the other edges at that end's node, the
    /// one whose run from the node is longest, through its far end in turn where it spans the edge.
    /// `room` is how much more matters.
    fn bridged_water_beyond(&self, e: u32, which: u32, seen: &mut Vec<u32>, room: f64) -> f64 {
        const TOL: f64 = 1.0;
        let node = self.end_node(e, which) as usize;
        let mut best = 0.0f64;
        for &c in self.node_ports[node].ends() {
            let (o, ow) = (c >> 1, c & 1);
            if o == e || seen.contains(&o) || self.edge_crossover[o as usize] || !self.edge_ok(o) {
                continue;
            }
            let olen = self.edge_len[o as usize];
            let Some(r) = self.bridged_water(o).into_iter().find(|r| if ow == 0 { r[0] <= TOL } else { r[1] >= olen - TOL }) else { continue };
            let mut total = r[1] - r[0];
            if total < room && r[0] <= TOL && r[1] >= olen - TOL {
                seen.push(o);
                total += self.bridged_water_beyond(o, 1 - ow, seen, room - total);
                seen.pop();
            }
            best = best.max(total);
            if best >= room {
                break;
            }
        }
        best
    }

    fn check_node(&self, n: u32, out: &mut Vec<Issue>) {
        let i = n as usize;
        if !self.inside_world(self.node_x[i], self.node_y[i]) {
            out.push(Issue::OutsideWorld { x: self.node_x[i], y: self.node_y[i] });
        }
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
        if let Some(b) = self.node_berth[i] {
            // a port's landing is on the shore, which the coarse masks may call water
            if self.node_ports[i].n > 0 || self.node_upgrades[i] != 0 || self.node_level[i] != 0 {
                out.push(Issue::NotAtPort { node: n });
            }
            if !super::ferry::berth_ok(self, [self.node_x[i], self.node_y[i]], b) {
                out.push(Issue::PortBerth { node: n });
            }
            return;
        }
        let plat = self.node_platform[i];
        if !(PLATFORM_MIN..=PLATFORM_MAX).contains(&plat) || plat % PLATFORM_STEP != 0 {
            out.push(Issue::PlatformLength { node: n });
        }
        let p = self.node_ports[i];
        // A station whose track was removed waits for new track (its lines are broken).
        if p.n > 2 {
            out.push(Issue::StationJunction { node: n });
            return;
        }
        let reach = self.platform_reach(n);
        for &c in p.ends() {
            let (e, which) = (c >> 1, c & 1);
            let len = self.edge_len[e as usize];
            let other = self.end_node(e, 1 - which);
            let other_reach = if self.node_platform[other as usize] > 0 { self.platform_reach(other) } else { 0.0 };
            if reach + other_reach > len - 1.0 {
                let ports = self.node_ports[other as usize].n;
                out.push(if other_reach > 0.0 {
                    Issue::PlatformStation { node: n }
                } else if ports == 2 {
                    Issue::PlatformJoint { node: n }
                } else if ports > 2 {
                    Issue::PlatformJunction { node: n }
                } else {
                    Issue::PlatformTooLong { node: n }
                });
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
        if let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            return if super::bus::route_ok(self, &r.points, &r.kinds) { Ok(vec![]) } else { Err(Issue::LinePath { line: l }) };
        }
        if let Some(r) = self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            return super::ferry::check(self, l, self.line_stops(l), r).map(|_| vec![]);
        }
        let path = self.line_path(l);
        let stops = self.line_stops(l);
        if stops.len() < 2 && path.is_empty() {
            if stops.iter().all(|&s| self.node_ok(s) && self.node_platform[s as usize] > 0) { return Ok(stops.to_vec()); }
            return Err(Issue::LineStops { line: l });
        }
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
                // Through a node: arrive on one side, leave on the other, or turn back at a
                // stop or a bare track end (T-797, checked below).
                if self.port_side(path_edge(prev), 1 - path_dir(prev)).is_none() {
                    return Err(Issue::LinePath { line: l });
                }
            }
            nodes.push(to);
        }
        if stops[0] != nodes[0] || *stops.last().unwrap() != *nodes.last().unwrap() {
            return Err(Issue::LineStops { line: l });
        }
        if stops.iter().any(|&s| !self.node_ok(s) || self.node_platform[s as usize] == 0) {
            return Err(Issue::LineStops { line: l });
        }
        let turns = self.stop_reversals(path);
        if match_stops(&nodes, stops, &turns).is_none() {
            return Err(if turns.is_empty() { Issue::LineStops { line: l } } else { Issue::LinePath { line: l } });
        }
        Ok(nodes)
    }

    /// Whether a path turns back between entries `a` and `b`: it arrives at their node and leaves
    /// on the same side (T-797).
    pub fn turns_back(&self, a: u32, b: u32) -> bool {
        let arr = self.port_side(path_edge(a), 1 - path_dir(a));
        arr.is_some() && arr == self.port_side(path_edge(b), path_dir(b))
    }

    /// The indices into a path's node list (`path.len() + 1` nodes) where it turns back.
    pub fn path_reversals(&self, path: &[u32]) -> Vec<usize> {
        (1..path.len()).filter(|&k| self.turns_back(path[k - 1], path[k])).collect()
    }

    /// A track end with no station (T-797 B3): a train may reverse there without stopping for
    /// riders, as in a headshunt.
    pub fn bare_end(&self, n: u32) -> bool {
        self.node_ports[n as usize].n == 1 && self.node_platform[n as usize] == 0
    }

    /// `path_reversals` that must fall on a stop: all but those at a bare track end.
    pub fn stop_reversals(&self, path: &[u32]) -> Vec<usize> {
        self.path_reversals(path).into_iter().filter(|&k| !self.bare_end(self.path_nodes(path[k]).0)).collect()
    }

    /// What kind of loop stops and a path make (T-798).
    pub fn loop_kind(&self, stops: &[u32], path: &[u32]) -> LoopKind {
        match (stops.first(), stops.last(), path.first(), path.last()) {
            (Some(a), Some(b), Some(&p0), Some(&p1)) if stops.len() >= 3 && a == b => {
                if self.turns_back(p1, p0) { LoopKind::Balloon } else { LoopKind::Circle }
            }
            _ => LoopKind::None,
        }
    }

    pub fn line_loop(&self, l: u32) -> LoopKind {
        self.loop_kind(self.line_stops(l), self.line_path(l))
    }

    /// Where a line's trains reverse: partway, and where a balloon loop closes.
    pub fn turn_count(&self, stops: &[u32], path: &[u32]) -> usize {
        self.path_reversals(path).len() + (self.loop_kind(stops, path) == LoopKind::Balloon) as usize
    }

    // ---------------------------------------------------------------- cost

    /// Unbuilt costs grouped by kind, charged level, track count (`HSR_ROW` for high-speed track),
    /// water and ramp. Seven numbers per row: those five keys, metres/count, US$M. Kind 6 is what
    /// the terrain adds to track (T-1270), its level field holding the terrain class; kind 7 the
    /// roads crossed at ground level, weighted (T-1929).
    pub fn blueprint_cost_items(&self) -> Vec<f64> {
        // the fifth key field is a track row's terrain class (T-1839; a ramp is no price factor, it is
        // charged at its dearer end, so ramps are not told apart), 0 for every other kind
        let mut rows = std::collections::BTreeMap::<(u8, i8, u8, bool, u8), (f64, f64)>::new();
        let mut add = |kind, level, tracks, wet, class: u8, qty, cost| {
            let r = rows.entry((kind, level, tracks, wet, class)).or_default();
            r.0 += qty;
            r.1 += cost;
        };
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && !self.edge_built[e] && !self.edge_off[e] && self.edge_crossover[e] {
                let a = self.edge_a[e] as usize;
                add(2, self.node_level[a], 0, !self.edge_water(e as u32).is_empty(), 0, 1.0, self.edge_cost[e]);
            } else if self.edge_alive[e] && !self.edge_built[e] && !self.edge_off[e] {
                let (tracks, hsr) = (self.edge_tracks[e], self.edge_hsr[e]);
                cost::track_cost_parts_t(self.edge_len[e], self.edge_vert(e as u32), self.edge_water(e as u32), self.edge_terrain(e as u32), tracks, hsr, |level, wet, _ramp, class, m, c| {
                    // T-1839: the whole price, keyed by terrain class (0 where it does not count)
                    let class = if cost::terrain_surcharge(level, class) == 0.0 { 0 } else { class };
                    add(0, level, if hsr { HSR_ROW } else { tracks }, wet, class, m, c);
                });
                // kind 7: roads crossed at ground level (T-1929), a count
                let (n, c) = roads::charge(self.edge_vert(e as u32), self.edge_roads(e as u32));
                if n > 0.0 {
                    add(7, 0, 0, false, 0, n, c);
                }
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] { continue; }
            // asked only where a price needs it: most nodes need none
            let is_wet = || self.node_is_wet(n);
            let level = self.node_level[n];
            // kind 12: ports (T-2193), a flat price
            if self.node_platform[n] > 0 && !self.node_built[n] && !self.node_off[n] && self.node_berth[n].is_some() {
                add(12, 0, 0, false, 0, 1.0, PORT_COST);
            } else if self.node_platform[n] > 0 && !self.node_built[n] && !self.node_off[n] {
                let (tracks, wet) = (self.node_tracks(n as u32), is_wet());
                add(1, level, tracks, wet, 0, 1.0, cost::station_cost(level, self.node_platform[n], tracks, wet));
                let ups = self.node_upgrades[n];
                if ups != 0 {
                    add(8, level, 0, wet, 0, ups.count_ones() as f64, self.station_upgrades_price(n as u32, ups, wet));
                }
            }
            let nn = n as u32;
            if self.priced_ends(nn, |_| true) >= 3
                && self.priced_ends(nn, |e| self.edge_built[e]) < 3
                && self.priced_ends(nn, |e| self.edge_built[e] || !self.edge_off[e]) >= 3
            {
                let wet = is_wet();
                let base = cost::junction_cost(level, false, wet);
                add(2, level, 0, wet, 0, 1.0, base);
                if self.node_flying[n] {
                    add(3, level, 0, wet, 0, 1.0, cost::junction_cost(level, true, wet) - base);
                }
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                let (e1, e2) = (x.e1 as usize, x.e2 as usize);
                if (!self.edge_built[e1] || !self.edge_built[e2]) && !self.edge_off[e1] && !self.edge_off[e2] && self.priced_crossing(x) {
                    let wet = self.water_mask.is_water(x.x, x.y);
                    add(4, level, 0, wet, 0, 1.0, cost::crossing_cost(level, wet));
                }
            }
        }
        // passages to stations still to be constructed (T-1281), built with them
        for ([a, b], price) in self.passages() {
            let (a, b) = (a as usize, b as usize);
            let ready = |n: usize| self.node_built[n] || !self.node_off[n];
            if !(self.node_built[a] && self.node_built[b]) && ready(a) && ready(b) {
                add(9, 0, 0, false, 0, 1.0, price);
            }
        }
        // bus routes still to be constructed (T-1464): road metres
        for l in self.bus_lines() {
            if self.line_bus[l as usize].as_ref().is_some_and(|r| !r.built) {
                let m = self.bus_road_m(l);
                add(10, 0, 0, false, 0, m, m / 1000.0 * BUS_ROUTE_COST_PER_KM);
            }
        }
        // kind 14: ferry routes still to be constructed (T-2193): ground metres, free
        for l in self.ferry_lines() {
            if self.line_ferry[l as usize].as_ref().is_some_and(|r| !r.built) {
                add(14, 0, 0, false, 0, self.ferry_m(l), 0.0);
            }
        }
        rows.into_iter().flat_map(|((kind, level, tracks, wet, ramp), (qty, cost))|
            [kind as f64, level as f64, tracks as f64, wet as u8 as f64, ramp as u8 as f64, qty, cost]).collect()
    }

    /// T-2269: the blueprint's track, stations, junctions, flat crossings and passages priced one by
    /// one, switched-off ones too, for the Build tab's breakdown by stretch. Five numbers a row: kind
    /// (0 a piece of track, 1 a station or port with its upgrades, 2 a junction with any flyover, 3 a
    /// flat crossing, 4 a passage), two ids (a crossing's two edges, a passage's two stations, else
    /// the edge or node twice), US$M, and 1 when `blueprint_cost_items` counts it (the same tests).
    pub fn blueprint_piece_costs(&self) -> Vec<f64> {
        let mut out = vec![];
        let mut row = |kind: u8, a: usize, b: usize, cost: f64, on: bool| out.extend([kind as f64, a as f64, b as f64, cost, on as u8 as f64]);
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && !self.edge_built[e] {
                row(0, e, e, self.edge_cost[e], !self.edge_off[e]);
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] { continue; }
            let is_wet = || self.node_is_wet(n);
            let level = self.node_level[n];
            if self.node_platform[n] > 0 && !self.node_built[n] {
                let price = if self.node_berth[n].is_some() { PORT_COST } else {
                    let (tracks, wet) = (self.node_tracks(n as u32), is_wet());
                    let ups = self.node_upgrades[n];
                    cost::station_cost(level, self.node_platform[n], tracks, wet) + if ups != 0 { self.station_upgrades_price(n as u32, ups, wet) } else { 0.0 }
                };
                row(1, n, n, price, !self.node_off[n]);
            }
            let nn = n as u32;
            if self.priced_ends(nn, |_| true) >= 3 && self.priced_ends(nn, |e| self.edge_built[e]) < 3 {
                let on = self.priced_ends(nn, |e| self.edge_built[e] || !self.edge_off[e]) >= 3;
                row(2, n, n, cost::junction_cost(level, self.node_flying[n], is_wet()), on);
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                let (e1, e2) = (x.e1 as usize, x.e2 as usize);
                if (!self.edge_built[e1] || !self.edge_built[e2]) && self.priced_crossing(x) {
                    let wet = self.water_mask.is_water(x.x, x.y);
                    row(3, e1, e2, cost::crossing_cost(level, wet), !self.edge_off[e1] && !self.edge_off[e2]);
                }
            }
        }
        for ([a, b], price) in self.passages() {
            let (a, b) = (a as usize, b as usize);
            if !(self.node_built[a] && self.node_built[b]) {
                let ready = |n: usize| self.node_built[n] || !self.node_off[n];
                row(4, a, b, price, ready(a) && ready(b));
            }
        }
        out
    }

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
                c += self.station_price(n, wet);
                c += self.upgrades_cost(n);
            }
            if self.priced_ends(n as u32, |_| true) >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.priced_crossing(x) {
                    c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                }
            }
        }
        c + self.passages_cost(false) + self.bus_cost(false) + super::building::value(self)
    }

    /// `(total_cost(), built_cost())` in one pass: the same sums in the same order, so equal to the
    /// bit, with a station priced once and the water under a node asked of the mask only when the
    /// node or the mask is new (`node_is_wet`; `total_cost` and `built_cost` ask every time, and the
    /// rows check compares them with this) (T-1540).
    pub fn total_and_built_cost(&self) -> (f64, f64) {
        let (mut t, mut b) = (0.0, 0.0);
        for e in 0..self.edge_count() {
            if self.edge_alive[e] {
                t += self.edge_cost[e];
                if self.edge_built[e] {
                    b += self.edge_cost[e];
                }
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.node_is_wet(n);
            if self.node_platform[n] > 0 {
                let station = self.station_price(n, wet);
                let upgrades = self.upgrades_cost(n);
                t += station;
                t += upgrades;
                if self.node_built[n] {
                    b += station;
                    b += upgrades;
                }
            }
            let (all, built) = (self.priced_ends(n as u32, |_| true) >= 3, self.priced_ends(n as u32, |e| self.edge_built[e]) >= 3);
            if all || built {
                let junction = cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
                if all {
                    t += junction;
                }
                if built {
                    b += junction;
                }
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.priced_crossing(x) {
                    let price = cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                    t += price;
                    if self.edge_built[x.e1 as usize] && self.edge_built[x.e2 as usize] {
                        b += price;
                    }
                }
            }
        }
        let halls = super::building::value(self);
        (t + self.passages_cost(false) + self.bus_cost(false) + halls, b + self.passages_cost(true) + self.bus_cost(true) + halls)
    }

    /// Build cost of what is constructed, US$M (SPEC 6.4): constructed edges and stations, a
    /// junction once three of its edge ends are constructed, a flat crossing once both its edges
    /// are, a passage once both its stations are, station buildings always (T-2217).
    /// `total_cost() - built_cost()` is what the
    /// blueprint would cost to construct.
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
                c += self.station_price(n, wet);
                c += self.upgrades_cost(n);
            }
            if self.priced_ends(n as u32, |e| self.edge_built[e]) >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.edge_built[x.e1 as usize] && self.edge_built[x.e2 as usize] && self.priced_crossing(x) {
                    c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                }
            }
        }
        c + self.passages_cost(true) + self.bus_cost(true) + super::building::value(self)
    }

    /// Ground metres between two nodes in a straight line: a passage's length (T-1281).
    pub fn passage_m(&self, a: u32, b: u32) -> f64 {
        let (a, b) = (a as usize, b as usize);
        let d = (self.node_x[a] - self.node_x[b]).hypot(self.node_y[a] - self.node_y[b]);
        if self.frame().scaled() { d / super::frame::merc_k(0.5 * (self.node_y[a] + self.node_y[b])) } else { d }
    }

    /// The passages (the player's links) and their prices, US$M (T-1281): a passage is constructed
    /// once both its stations are.
    pub fn passages(&self) -> impl Iterator<Item = ([u32; 2], f64)> + '_ {
        self.links_live().map(|p| (p, super::upgrades::passage_price(self.passage_m(p[0], p[1]))))
    }

    fn passages_cost(&self, built: bool) -> f64 {
        self.passages().filter(|(p, _)| !built || p.iter().all(|&n| self.node_built[n as usize])).map(|x| x.1).sum()
    }

    /// A station's upgrades, US$M (T-1281).
    pub fn upgrades_cost(&self, n: usize) -> f64 {
        let ups = self.node_upgrades[n];
        if ups == 0 {
            return 0.0;
        }
        self.station_upgrades_price(n as u32, ups, self.water_mask.is_water(self.node_x[n], self.node_y[n]))
    }

    /// Edge ends at a node whose edge is constructed.
    pub fn built_ports(&self, n: u32) -> usize {
        self.node_ports[n as usize].ends().iter().filter(|&&c| self.edge_built[(c >> 1) as usize]).count()
    }

    /// The constructed station or edge set is complete for a line: it can carry trains.
    pub fn line_constructed(&self, l: u32) -> bool {
        if let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            return r.built;
        }
        if let Some(r) = self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            return r.built && self.line_stops(l).iter().all(|&s| self.node_ok(s) && self.node_built[s as usize]);
        }
        self.line_path(l).iter().all(|&p| self.edge_ok(path_edge(p)) && self.edge_built[path_edge(p) as usize])
            && self.line_stops(l).iter().all(|&s| self.node_ok(s) && self.node_built[s as usize])
    }

    /// The point of an edge nearest to (x, y): chainage and distance, m.
    pub fn project(&self, e: u32, x: f64, y: f64) -> (f64, f64) {
        geom::project(self.edge_pieces(e), x, y)
    }

    // ---------------------------------------------------------------- routing

    /// Fastest path from `from` to `to` over valid moves, by speed-limit running time. `arrived`
    /// is the side the train came in on at `from`, or None at a line's first stop. It leaves on
    /// the other side, or turns back there at `REVERSAL_SEARCH_S` extra; on the way it may turn
    /// back at a bare track end at the same cost (T-797: never at a station it passes or a
    /// junction). Returns the path and the side it arrives on at `to`.
    pub fn find_path(&self, from: u32, to: u32, arrived: Option<u8>) -> Option<(Vec<u32>, u8)> {
        self.find_path_side(from, to, arrived, None)
    }

    /// `find_path` arriving at `to` on the side `want` when given (T-798: a loop's closing hop
    /// arrives so the train carries on round).
    pub fn find_path_side(&self, from: u32, to: u32, arrived: Option<u8>, want: Option<u8>) -> Option<(Vec<u32>, u8)> {
        let ne = self.edge_count();
        let mut dist = vec![f64::INFINITY; 2 * ne];
        let mut prev = vec![u32::MAX; 2 * ne];
        let mut heap = BinaryHeap::new();
        let p = &self.node_ports[from as usize];
        for (k, &c) in p.ends().iter().enumerate() {
            let turn = if arrived == Some(p.side[k]) { REVERSAL_SEARCH_S } else { 0.0 };
            let (e, which) = (c >> 1, c & 1);
            let st = e << 1 | which; // leaving from end a = direction 0
            dist[st as usize] = self.edge_time[e as usize] + turn;
            heap.push(Key(dist[st as usize], st));
        }
        while let Some(Key(d, st)) = heap.pop() {
            if d > dist[st as usize] {
                continue;
            }
            let (_, node) = self.path_nodes(st);
            let arr_side = self.port_side(path_edge(st), 1 - path_dir(st)).unwrap_or(0);
            if node == to && want.map_or(true, |w| w == arr_side) {
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
            let bare = self.bare_end(node);
            for (k, &c) in p.ends().iter().enumerate() {
                if p.side[k] == arr_side && !bare {
                    continue;
                }
                let (e, which) = (c >> 1, c & 1);
                let nst = e << 1 | which;
                let nd = d + self.edge_time[e as usize] + if bare { REVERSAL_SEARCH_S } else { 0.0 };
                if nd < dist[nst as usize] {
                    dist[nst as usize] = nd;
                    prev[nst as usize] = st;
                    heap.push(Key(nd, nst));
                }
            }
        }
        None
    }

    /// Fastest path through a list of stops, hop by hop; it may turn back at a stop.
    pub fn route(&self, stops: &[u32]) -> Option<Vec<u32>> {
        if stops.iter().any(|&s| !self.node_ok(s) || self.node_platform[s as usize] == 0) { return None; }
        self.route_hops(stops).ok()
    }

    /// `route` over stops that are all stations, or the first hop with no path (T-2895: hop k runs
    /// from stop k to stop k + 1).
    pub fn route_hops(&self, stops: &[u32]) -> Result<Vec<u32>, usize> {
        let mut path: Vec<u32> = vec![];
        let mut side = None;
        let closes = stops.len() >= 3 && stops[0] == stops[stops.len() - 1];
        for (k, w) in stops.windows(2).enumerate() {
            let want = if closes && k + 2 == stops.len() { self.carry_on_side(&path) } else { None };
            let (p, s) = want.and_then(|_| self.find_path_side(w[0], w[1], side, want)).or_else(|| self.find_path(w[0], w[1], side)).ok_or(k)?;
            path.extend(p);
            side = Some(s);
        }
        Ok(path)
    }

    /// Whether any track joins two nodes, whatever way a train would have to turn on it (T-2895).
    pub fn joined(&self, a: u32, b: u32) -> bool {
        let mut seen = vec![false; self.node_count()];
        let mut queue = vec![a];
        seen[a as usize] = true;
        while let Some(n) = queue.pop() {
            if n == b {
                return true;
            }
            for &c in self.node_ports[n as usize].ends() {
                let m = self.path_nodes(c).1;
                if !seen[m as usize] {
                    seen[m as usize] = true;
                    queue.push(m);
                }
            }
        }
        false
    }

    /// The side a path ending at its first node must arrive on to carry straight on into its first
    /// entry (T-798: a circle line).
    pub fn carry_on_side(&self, path: &[u32]) -> Option<u8> {
        let &p0 = path.first()?;
        self.port_side(path_edge(p0), path_dir(p0)).map(|s| 1 - s)
    }

    /// Where to split an edge at chainage `s` without changing its geometry (notes/T-008.md 4.1):
    /// the new node, the first part (keeps the edge's id) and the second part. A split inside an
    /// arc gets a new PI on each side with the arc's radius; the neighbouring PIs whose legs
    /// change keep their current radius as a set one.
    pub fn split_plan(&self, e: u32, s: f64) -> Result<(NodeData, EdgeData, EdgeData), Issue> {
        if !self.edge_ok(e) {
            return Err(Issue::NoSuchEdge { edge: e });
        }
        if self.edge_crossover[e as usize] {
            return Err(Issue::Crossover { edge: e });
        }
        let v = self.edge_vertices(e);
        let f = self.fit(&v);
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
            // `r` is the ground radius the new PIs are given; the tangents are plane geometry.
            let scaled = self.frame.scaled();
            let r = quant(piece.ground_radius());
            let rt = if scaled { piece.radius() } else { r };
            let (d1, d2) = if scaled {
                let d1 = piece.plane_d(s - piece.s0);
                (d1, piece.plen - d1)
            } else {
                (s - piece.s0, piece.s0 + piece.len - s)
            };
            let t1 = rt * (d1 * piece.k.abs() / 2.0).tan();
            let t2 = rt * (d2 * piece.k.abs() / 2.0).tan();
            let mut pi1 = Pi::new(quant(x - t1 * th.cos()), quant(y - t1 * th.sin()), r, v[m].level);
            let mut pi2 = Pi::new(quant(x + t2 * th.cos()), quant(y + t2 * th.sin()), r, v[m].level);
            // On the millimetre grid a new PI a few metres from the node turns by a slightly
            // different angle than the arc did, and at the arc's radius its curve would overrun
            // the leg to the node ("does not fit", T-258). Each takes the radius that ends its
            // curve at the node; that is the arc's to well under 1% unless the PI is within a few
            // metres of the node, and then the split goes to the arc's end instead.
            // the radius at `p` (between `a` and `b`) whose tangent is `leg` long
            let fits = |a: (f64, f64), p: &Pi, b: (f64, f64), leg: f64| -> f64 {
                let half = geom::wrap((b.1 - p.y).atan2(b.0 - p.x) - (p.y - a.1).atan2(p.x - a.0)).abs() / 2.0;
                if half < 1e-12 { f64::INFINITY } else { leg / half.tan() }
            };
            let to_node = |p: &Pi| (node.x - p.x).hypot(node.y - p.y);
            // In ground metres (plane radius over k at the PI; k is 1 unscaled).
            let r1 = fits((v[m - 1].x, v[m - 1].y), &pi1, (node.x, node.y), to_node(&pi1)) / self.k_at(pi1.y);
            let r2 = fits((node.x, node.y), &pi2, (v[m + 1].x, v[m + 1].y), to_node(&pi2)) / self.k_at(pi2.y);
            if r1.min(r2) < 0.99 * r {
                let end = if s - piece.s0 < piece.s0 + piece.len - s { piece.s0 } else { piece.s0 + piece.len };
                return self.split_plan(e, end);
            }
            pi1.radius = r.min((r1 * 0.99999 / QUANT).floor() * QUANT);
            pi2.radius = r.min((r2 * 0.99999 / QUANT).floor() * QUANT);
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
            // A PI at the split point (a joint kept for its level, T-622) is the node now.
            let at_node = |p: &Pi| (p.x - node.x).hypot(p.y - node.y) < 0.01;
            let drop_prev = i > 0 && at_node(&v[i]);
            let drop_next = i + 1 < v.len() - 1 && at_node(&v[i + 1]);
            p1 = v[..i + !drop_prev as usize].to_vec();
            p1.push(marker);
            p2 = vec![marker];
            p2.extend_from_slice(&v[i + 1 + drop_next as usize..]);
            if !drop_prev {
                freeze(&mut p1, i, i);
            }
            if !drop_next {
                freeze(&mut p2, 1, i + 1);
            }
            // T-622: a ramp is centred between the two vertices of the leg, so cutting the leg
            // would move it. Where the ramp lies on one straight, a vertex at each end of it (on
            // the track, no turn) holds it where it was, on whichever side of the cut it is.
            if !drop_prev && !drop_next && v[i].level != v[i + 1].level {
                let (l0, l1) = (v[i].level, v[i + 1].level);
                let half = RAMP_PER_LEVEL * (l1 as f64 - l0 as f64).abs() / 2.0 + 0.01;
                let c = (f.vert_s[i] + f.vert_s[i + 1]) / 2.0;
                let (r0, r1) = (c - half, c + half);
                let (j0, j1) = (geom::piece_at(&f.pieces, r0), geom::piece_at(&f.pieces, r1));
                if r0 > 0.5 && r1 < f.len - 0.5 && j0 == j1 && f.pieces[j0].k == 0.0 {
                    let at = |r: f64, level: i8| {
                        let (x, y, _) = geom::pos_at(&f.pieces, r);
                        Pi::new(quant(x), quant(y), 0.0, level)
                    };
                    let ramp = [at(r0, l0), at(r1, l1)];
                    if r1 < s {
                        p1.splice(p1.len() - 1..p1.len() - 1, ramp);
                    } else if r0 > s {
                        p2.splice(1..1, ramp);
                    }
                }
            }
        }
        // T-622: two curves back to back (a joined pair of tangent arcs) have no straight between
        // them to spare, and the new node and PIs sit on the millimetre grid, which turns the
        // legs by a hair (a 1,796 m curve's tangent grew 24.413 to 24.426 m). A curve with a set
        // radius that no longer fits gives up 0.05% of it until it does.
        for part in [&mut p1, &mut p2] {
            for _ in 0..10 {
                let bad = self.fit(part).clamped;
                if bad.is_empty() {
                    break;
                }
                for j in bad {
                    part[j].radius = (part[j].radius * 0.9995 / QUANT).floor() * QUANT;
                }
            }
        }
        let i = e as usize;
        let (tracks, built, off, hsr, sep) = (self.edge_tracks[i], self.edge_built[i], self.edge_off[i], self.edge_hsr[i], self.edge_sep[i]);
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
        let a = EdgeData { a: self.edge_a[i], b: u32::MAX, tracks, pis: p1[1..p1.len() - 1].to_vec(), built, off, thru: t1, crossover: false, hsr, sep, day: self.edge_day[i] };
        let b = EdgeData { a: u32::MAX, b: self.edge_b[i], tracks, pis: p2[1..p2.len() - 1].to_vec(), built, off, thru: t2, crossover: false, hsr, sep, day: self.edge_day[i] };
        Ok((node, a, b))
    }
}
