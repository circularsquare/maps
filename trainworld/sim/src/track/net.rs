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
use super::lanes::{ends_in, Moves};
use super::params::*;
use super::roads::{self, Roads};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

/// Most edge ends one node can have.
pub const MAX_PORTS: usize = 8;

/// The least track a split leaves on each side of its node, m (`split_plan`), so no sliver piece
/// or PI a hair from a node is made.
pub const SPLIT_ROOM: f64 = 5.0;

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
    /// The kind of track (`TrackKind`): high speed (T-817) is always double.
    pub kind: TrackKind,
    /// Grade-separated from every edge with this set, itself included (T-1751): where two meet there is no
    /// crossing at all, flat or bad. Set on an imported real network's edges, whose lines were
    /// each given one of six levels and cannot all avoid one another.
    pub sep: bool,
    /// The day this was first constructed (T-2871), as `DAY_UNKNOWN` documents; 0 on a blueprint.
    pub day: u32,
    /// One-way single track (T-3094), the player's: 0 not set (the way its run takes by itself,
    /// `Network::edge_way`), 1 a to b only, 2 b to a only, 3 both ways. Single track only.
    pub oneway: u8,
    /// T-3137: per end (a, b), which track of the double track at that junction single track meets,
    /// looking out from the node along this piece: 0 none (the centre, both, as before joins), 1
    /// the left, 2 the right. Single track only; read only where the node has double track on the
    /// end's other side (`Network::join_away`).
    pub join: [u8; 2],
    /// T-3405: the player's speed limit on this piece, km/h; 0 none. Never on a crossover, always below
    /// the kind's top speed: a limit at or over the top is none (T-3525), so the piece joins unlimited
    /// track (`Network::set_edge` keeps it so).
    pub limit: u16,
    /// T-3027: the ramp per level this piece's ramps aim for, m (`geom::Ramps::of`): the drawing
    /// tool's for its target speed, shorter where it had no room; 0 on track from before, whose
    /// ramps are 200 m with no speed limit.
    pub ramp: u16,
}

/// The speed limit a piece of `kind` keeps when asked for `kmh` (T-3405): none on a crossover, for 0,
/// or at or over the kind's top speed (T-3525); else at least `EDGE_LIMIT_MIN`.
pub fn piece_limit(kmh: u16, kind: TrackKind, crossover: bool) -> u16 {
    let top = (kind.v_cap() * 3.6).round() as u16;
    if crossover || kmh == 0 || kmh >= top { 0 } else { kmh.max(EDGE_LIMIT_MIN) }
}

/// T-3332: `StationData::one` for trains travelling heading `th`: 1 eastward (northward where the
/// track runs due north-south), else 2. Stored against this fixed axis, as a renumbered edge can
/// turn a node's heading round.
pub fn one_for_travel(th: f64) -> u8 {
    let along = if th.cos().abs() > 1e-6 { th.cos() > 0.0 } else { th.sin() > 0.0 };
    if along { 1 } else { 2 }
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
    /// T-3332: a platform on one track only, 0 for both. On double track it serves one way, 1 trains
    /// running along the node's axis (`Network::axis`), 2 against it, and trains the other way run past;
    /// on single track (either value) it is a stop with no passing loop.
    pub one: u8,
    /// T-3740: its fare zone, set by the player: 0 none, else the fare area's id (the fares'
    /// `zones`, the app's) << 8 | the zone number, 1-255.
    pub fare: u32,
    /// T-3759: the name it shows when it is not bound to its complex's name; empty while bound. `name`
    /// stays its part in the complex's name (the app's `nameSource`) either way.
    pub own: String,
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
    /// T-3423: the groups `group` sits in, nearest first, each named and drawn as on every line of it.
    pub outer: Vec<OuterGroup>,
    /// T-2871: the day this line was added, or for a bus or ferry line its route constructed, as
    /// `EdgeData::day` (0 for a planned route).
    pub day: u32,
    /// T-2895: the player set the stop order by hand, so no stop edit puts the stops in a
    /// straighter order (`straighten`).
    pub hand: bool,
    /// T-3711: the stops the player made one way (a shift-click): station and the port side of it
    /// trains arrive by where they stop (`settle_ones` in wasm/lines.rs picks it); 2 while not yet
    /// fixed (set on a line with no route). Served once a round trip, as a one-way station is
    /// (`Network::legs`). Only stations among `stops`; saved in `LOS1`.
    pub ones: Vec<(u32, u8)>,
    /// T-2981: a loop runs one way, round in the direction its stops or route were drawn (run 0
    /// alone); otherwise both ways. Kept on an open line, which ignores it, so it carries over a
    /// close. A new bus or ferry line starts one way, a train line both ways.
    pub one_way: bool,
    /// T-2987: the player's speed limit for the line's trains, km/h; 0 none. Applies to rail lines only.
    pub limit: u16,
    /// T-3333: the line runs on both pairs of quad track (`Parent`), or is the hidden twin that runs
    /// the other pair for one (`Of`).
    pub twin: super::twin::Twin,
    /// T-3521: the line's trains on weekends (`params::WEEKEND`). Saved in `LWE1`.
    pub weekend: Weekend,
    /// T-3542: the rows the line sets its trains in (3, 4 or 5), 0 for the game's (`Network::rows`).
    pub rows: u8,
    /// T-3542: its midday and afternoon rows, read only on a line of four or five rows.
    pub x: Extra,
    /// T-3758: its rides pay the distance fare where the game has fare zones (else zone fares).
    /// Saved in `LFD1`.
    pub fare_dist: bool,
}

/// T-3542: a line's midday and afternoon rows (places 3 and 4, `params::places`): its rates, trains
/// and headways as `LineData::tph`, `trains` and `auto` (read only beside those), and its weekend
/// counts and headways set by hand as `Weekend::counts` and `headways`. Saved in `LPD1`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Extra {
    pub tph: [f32; 2],
    pub trains: [u32; 2],
    pub auto: [u32; 2],
    pub weekend_counts: [u32; 2],
    pub weekend_headways: [u32; 2],
    /// T-3542: the line's own hours (`params::Cuts`), all 0 for the game's (`Network::hours`).
    /// Saved in `LHR1`.
    pub hours: Cuts,
}

impl Extra {
    /// Midday and afternoon as the high row of `d` (a new line, a row split off high).
    pub fn of_high(d: &LineData) -> Extra {
        let h = |a: Option<[u32; 3]>| [a.map_or(0, |a| a[0]); 2];
        Extra { tph: [d.tph[0]; 2], trains: h(d.trains), auto: h(d.auto), weekend_counts: h(d.weekend.counts), weekend_headways: [d.weekend.headways[0]; 2], hours: d.x.hours }
    }
}

/// T-3521: how a line runs on weekends and public holidays. The default, 100% and no counts of its
/// own, runs the weekdays' trains and costs no settle of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weekend {
    /// A percent of the weekdays': counts rounded up, a headway line's headway divided by it and
    /// rounded down to a whole minute (`TrackWorld::weekend_counts`). Kept under `counts`, so the
    /// panel shows the percent the line had.
    pub share: u16,
    /// Whole trains at high, medium and low demand set by hand, in place of the share.
    pub counts: Option<[u32; 3]>,
    /// T-3557: the headway the player stepped a headway line's weekends to, seconds per level, 0 where
    /// none; read only beside `counts`, which then hold the trains that run it.
    pub headways: [u32; 3],
}

impl Default for Weekend {
    fn default() -> Self {
        Weekend { share: 100, counts: None, headways: [0; 3] }
    }
}

/// T-3542: the slots `period_slot` uses, in order, and how many (`Network::slots`).
fn slot_list(p: [usize; 5]) -> ([usize; SLOTS], usize) {
    let (mut l, mut n) = ([0; SLOTS], 0);
    for s in (0..SLOTS).filter(|s| p.contains(s)) {
        l[n] = s;
        n += 1;
    }
    (l, n)
}

/// T-3542: how a line's hours (`Network::hours`) fall into the slots, kept where they differ from
/// the default hours' (one row alone in each slot, its periods' own).
#[derive(Clone, Debug, PartialEq)]
pub struct Mix {
    /// Per slot, each place's share of the slot's hours.
    pub w: [[f64; 5]; SLOTS],
    /// Per slot, whether more than one place runs some of its hours.
    pub mixed: [bool; SLOTS],
    /// Per slot, the place with the most of its hours (the first in the day of equals).
    pub main: [u8; SLOTS],
    /// The line's day in pieces (`params::day_pieces`).
    pub pieces: [DayPiece; PIECES_MAX],
    pub n_pieces: usize,
    /// Half the shortest row, seconds (`params::shortest_row_h`).
    pub lead_max_s: f64,
}

impl Mix {
    /// The mix of hours `c` on a line of `n` rows with the game's `period_slot`, None where its day
    /// splits into the pieces the default hours give.
    pub fn of(c: Cuts, n: usize, period_slot: [usize; 5]) -> Option<Mix> {
        // (every edit asks for every line's: the default hours answer at once)
        if c == CUTS_DEFAULT {
            return None;
        }
        let (pieces, m) = day_pieces(c, n);
        if (pieces, m) == day_pieces(CUTS_DEFAULT, n) {
            return None;
        }
        let mut hrs = [[0u32; 5]; SLOTS];
        let mut first = [[usize::MAX; 5]; SLOTS];
        for (i, p) in pieces[..m].iter().enumerate() {
            let s = period_slot[p.q as usize];
            hrs[s][p.k as usize] += (p.b - p.a) as u32;
            first[s][p.k as usize] = first[s][p.k as usize].min(i);
        }
        let mut w = [[0.0; 5]; SLOTS];
        let mut mixed = [false; SLOTS];
        let mut main = [0u8; SLOTS];
        for s in 0..SLOTS {
            let all: u32 = hrs[s].iter().sum();
            if all == 0 { continue; }
            for k in 0..5 { w[s][k] = hrs[s][k] as f64 / all as f64; }
            mixed[s] = hrs[s].iter().filter(|&&h| h > 0).count() > 1;
            main[s] = (0..5).filter(|&k| hrs[s][k] > 0).max_by(|&a, &b| hrs[s][a].cmp(&hrs[s][b]).then(first[s][b].cmp(&first[s][a]))).unwrap_or(0) as u8;
        }
        Some(Mix { w, mixed, main, pieces, n_pieces: m, lead_max_s: 0.5 * shortest_row_h(c, n) as f64 * 3600.0 })
    }
}

/// T-3521: a headway on weekends at `share` percent of the weekdays': `headway_s` divided by the
/// share, rounded down to a whole minute (a headway under a minute to the second); 0 stays none.
pub fn weekend_headway(headway_s: u32, share: u16) -> u32 {
    if headway_s == 0 || share == 0 { return 0; }
    let raw = headway_s as u64 * 100 / share as u64;
    (if raw >= 60 { raw / 60 * 60 } else { raw.max(1) }).min(u32::MAX as u64) as u32
}

/// A line group's bullet set by hand (T-1813): its text (empty for the name's own) and its colour
/// (None for the colour most of its lines share).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GroupLook {
    pub symbol: String,
    pub colour: Option<u32>,
}

/// T-3423: a group a line's own group sits in, with its name (empty for its oldest line's) and bullet.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct OuterGroup {
    pub group: u32,
    pub name: String,
    pub look: GroupLook,
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
    /// T-3889: a station where tram track meets rail.
    StationFamily { node: u32 },
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
    /// T-3889: a line with no tram stop (a train line) would run on tram track.
    TramTrack { line: u32 },
    /// T-3911: a train or tram line has a ferry port among its stops.
    PortStop { line: u32 },
    /// A line's return run has no track to come back on: one-way track runs against it (T-3094).
    NoWayBack { line: u32 },
    /// One-way single track (T-3094) that a line cannot go round: it has no other way between its stations.
    OneWayNoWay { line: u32 },
    /// T-3332: a line stops at a one-track station only on runs that pass it on the other track, or
    /// ends there arriving on the other track.
    OneTrackStop { line: u32, node: u32 },
    /// T-3332: a one-track station placed or switched where it cannot go: anywhere but track running
    /// through with the same tracks both sides, or on one-way single track.
    OneTrackHere { node: u32 },
    LineSchedule { line: u32 },
    /// A split point on a ramp, too near a node, or outside the edge.
    SplitPoint { edge: u32 },
    /// A route end that would branch off a ramp: a node has one level, so a ramp is not split.
    SplitRamp { edge: u32 },
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
    /// T-3321: double track that cannot be rebuilt as single, and the thing that needs it double
    /// (`rebuild.rs` `single_refusal`).
    NoSingle { edge: u32, why: &'static str },
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
    /// T-3521: lines whose weekend setting changed (`set_weekend`), which settle only the weekends
    pub weekend: Vec<u32>,
    pub paths_changed: bool,
}

/// What `derive` rebuilt.
#[derive(Clone, Debug, Default)]
pub struct Derived {
    pub edges: Vec<u32>,
    pub nodes: Vec<u32>,
    /// Positions of crossings removed or added, and their tram tracks (`crossing_trams`).
    pub crossings: Vec<[f64; 3]>,
    /// Single track whose way (`Network::edge_way`) changed (T-3137); also in `edges`.
    pub ways: Vec<u32>,
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
    /// `StationData::one` (T-3332).
    pub node_one: Vec<u8>,
    /// `StationData::fare` (T-3740).
    pub node_fare: Vec<u32>,
    /// `StationData::own` (T-3759).
    pub node_own: Vec<String>,
    /// A port's berth (`StationData::berth`, T-2193); None for every other node.
    pub node_berth: Vec<Option<[f64; 2]>>,
    pub node_ports: Vec<Ports>,
    /// Derived with the ports: where each end's lanes lie and the moves through the node (T-3137).
    pub node_moves: Vec<Moves>,
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
    pub edge_kind: Vec<TrackKind>,
    pub edge_sep: Vec<bool>,
    pub edge_oneway: Vec<u8>,
    pub edge_join: Vec<[u8; 2]>,
    /// T-3405: `EdgeData::limit`.
    pub edge_limit: Vec<u16>,
    /// T-3027: `EdgeData::ramp`.
    pub edge_ramp: Vec<u16>,
    /// Derived (T-3137): the way single track's run takes by itself (`lanes.rs` `derive_ways`): 0
    /// both, 1 a to b, 2 b to a.
    pub edge_infer: Vec<u8>,
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
    /// T-3711: per line, its one-way stops the player set (`LineData::ones`).
    pub line_ones: Vec<Vec<(u32, u8)>>,
    pub line_one_way: Vec<bool>,
    pub line_limit: Vec<u16>,
    /// T-3758: `LineData::fare_dist`.
    pub line_fare_dist: Vec<bool>,
    /// T-3521: `LineData::weekend`.
    pub line_weekend: Vec<Weekend>,
    /// T-3521: each line's settled rates on weekends, as `line_tph` on weekdays
    /// (`TrackWorld::settle_counts`): every line's, since a line running its weekday trains then
    /// still shares the track with lines that do not. Read only where `rest_solved` says so.
    pub line_tph_rest: Vec<[f32; SLOTS]>,
    /// T-3521: weekends were settled on their own, as some line runs other trains then. Otherwise
    /// every line runs its weekday rates then, at no settle's cost.
    pub rest_solved: bool,
    /// `settle_kept` of the weekend settle.
    pub settle_kept_rest: Vec<(u32, usize, u32, u64, u32)>,
    /// T-3333: `LineData::twin`.
    pub line_twin: Vec<super::twin::Twin>,
    pub line_name: Vec<String>,
    pub line_symbol: Vec<String>,
    pub next_line_number: u32,
    /// What is stamped on an element constructed now (T-2871): `DAY_UNKNOWN` until the clock
    /// worker sets it (`TrackApi::set_game_day`).
    pub today: u32,
    pub line_colour: Vec<u32>,
    pub line_stops: Vec<Span>,
    pub line_path: Vec<Span>,
    pub line_tph: Vec<[f32; SLOTS]>,
    pub line_trains: Vec<Option<[u32; 3]>>,
    pub line_auto: Vec<Option<[u32; 3]>>,
    /// T-1787: `line_tph` holds settled rates (a settle has run, or a save carried them), so a save
    /// writes them to the bit in `LRT1`
    pub rates_settled: bool,
    /// T-1787: the rates came from a save's `LRT1`, so loading settles from them with an edit's
    /// quick settle, a no-op when nothing changed; without it a load settles fully, as before
    pub rates_from_save: bool,
    /// T-3233: the pacing came from a save's `LPC1` with its rates, so a reload keeps it rather than
    /// pacing again (the pacing can flip lines in and out of step every round, T-2973)
    pub pace_from_save: bool,
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
    pub line_group_outer: Vec<Vec<OuterGroup>>,
    /// T-1813: the Lines and Stations tabs' hand order, by line and by node id: 1 first, 0 for none.
    /// Outside the undo history, like the sort it replaces.
    pub line_rank: Vec<u32>,
    pub node_rank: Vec<u32>,
    /// T-1925: by line id, how many seconds after its pattern's own times each level's departures
    /// leave, so lines sharing track or a flat junction do not leave together (`spacing::place`);
    /// shorter than a missing entry is 0. Outside the undo history; saved in `LPH1`.
    pub line_phase: Vec<[f32; SLOTS]>,
    /// T-2972: by line id, the share of a headway line's spare time (`Service::spare`) its trains
    /// stand at the end of run 0 (its last stop) per level, the rest at its first: where it stands
    /// times the reverse departures (`spacing::place`); 0.5 where none is set. Outside the undo
    /// history; saved in `LSP1`.
    pub line_split: Vec<[f32; SLOTS]>,
    /// T-2973: derived, by line id and level: the headway a line set by train count keeps so it runs
    /// in step with the lines it meets at a merge (`TrackWorld::pace_merges`), s; 0 none. Its trains
    /// stand the time over at its ends, as a headway line's do. Not saved: every settle works it out.
    pub line_pace: Vec<[f32; SLOTS]>,
    /// T-3735: `line_phase` and `line_split` of the weekends, placed on the weekends' rates where a
    /// line's differ from its weekdays' (`TrackWorld::place_weekend`), a copy of the weekdays' for
    /// every other line. Read only where `rest_solved` says so; empty (or short): the weekdays'.
    /// Saved in `LWP1`.
    pub line_phase_rest: Vec<[f32; SLOTS]>,
    pub line_split_rest: Vec<[f32; SLOTS]>,
    /// T-3584: `line_pace` of the weekends (`TrackWorld::settle_weekend`), saved in `LWP1`; empty:
    /// the weekdays'.
    pub line_pace_rest: Vec<[f32; SLOTS]>,
    /// T-1925: `line_phase` is placed (a save carried it, or a load placed every line), so a
    /// reload keeps it rather than placing every line
    pub phases_placed: bool,
    /// T-3542: the slot serving each period (`period_slot`). Every line sets three rows, so it is
    /// `PERIOD_SLOT_3`; a test may set every period its own slot to work all five out
    /// (`set_period_slot`). With the slots it uses, in order, and how many, as `slots` reads them.
    period_slot: [usize; 5],
    slot_list: ([usize; SLOTS], usize),
    /// T-3542: the rows (3, 4 or 5) a line following the game's sets its trains in.
    pub rows_default: u8,
    /// `LineData::rows` and `x` by line.
    pub line_rows: Vec<u8>,
    pub line_x: Vec<Extra>,
    /// T-3542: the hours (`params::Cuts`) a line following the game's splits its day at.
    pub hours_default: Cuts,
    /// Derived (`refresh_slots`), by line: how its hours fall into the slots, None where every slot
    /// is one of its rows alone, the one its periods have at the default hours (the default hours
    /// always), so those lines take the paths they took before hours could move.
    line_mix: Vec<Option<Box<Mix>>>,
    /// The lines whose `Mix` the last `refresh_slots` changed.
    pub mix_moved: Vec<u32>,
    #[cfg(test)]
    slots_pinned: bool,
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

/// The part of a rail line's plan its trains run (T-3474, `Network::line_stretch`): stops
/// `stop0..stop1` and path entries `path0..path1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stretch {
    pub stop0: u32,
    pub stop1: u32,
    pub path0: u32,
    pub path1: u32,
}

impl Stretch {
    pub fn stops<'a>(&self, stops: &'a [u32]) -> &'a [u32] {
        &stops[self.stop0 as usize..self.stop1 as usize]
    }
    pub fn path<'a>(&self, path: &'a [u32]) -> &'a [u32] {
        &path[self.path0 as usize..self.path1 as usize]
    }
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

// the track-level modules the methods name as `super::x`, now one level down
use super::{building, bus, ferry, frame, streets, twin, upgrades};

mod service;
mod store;
mod stations;
mod edit;
mod derive;
mod validate;
mod costs;
mod paths;
