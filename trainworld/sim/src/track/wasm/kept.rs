//! T-4299: what a state reads of the model's money (the upkeep, the costs, the fleets' needs, the running costs, each
//! line's upkeep share and each station's upgrade upkeep), kept per record and worked out again only for the records
//! named since the last read (notes/T-4299.md "What names a change"):
//!  - an edge or node: in an applied edit's `Dirty` (undo, redo and owed services too), re-priced outside an edit
//!    (`Network::take_priced`), or at an end of an edge named (a node's junction, kind and one-platform rule read the
//!    edges at it);
//!  - a line: noted by the service (`Service::take_kept`: rebuilt, rescheduled, re-timed, a weekend setting), or whose
//!    price factor the app moved (`set_line_run_factors`, `note_lines`);
//!  - the passages and station buildings, set outside an edit (`link_stations`, `add_building`...: `note_extras`);
//!  - everything: a new model, a load or reload (`Service` noting every line), the water mask changed.
//! The edge and node totals are exact sums (`Fx`), so taking a record's old part off and adding its new one gives
//! what adding every record's part afresh gives: `TrackApi::kept_check` compares the two. The state's first read also
//! compares every line's money numbers (`LineMoney`) and the passages and buildings, a net for a change nothing named:
//! one it catches is a namer missing (`unnamed`, which the check reports). What is still worked out
//! whole on a change is said where it is: the lines' money (a pass over the lines), the track limits the weekend
//! counts read (`Capacity::line_limits`, once a change), the flat crossings, passages and station buildings.

use super::lines::Limits;
use super::*;
use crate::track::timing::PROFILE_VARS;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

/// An exact sum of US$M figures: each term in whole steps of 2^-64 in an i128, so terms come off and go on in any
/// order to the total a whole sum gives (a term is off by at most 2^-65 US$M).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Fx(i128);
const FX_ONE: f64 = 18_446_744_073_709_551_616.0;
impl Fx {
    fn of(x: f64) -> Fx {
        Fx((x * FX_ONE).round() as i128)
    }
    fn get(self) -> f64 {
        self.0 as f64 / FX_ONE
    }
}
impl std::ops::AddAssign for Fx {
    fn add_assign(&mut self, o: Fx) {
        self.0 += o.0;
    }
}
impl std::ops::SubAssign for Fx {
    fn sub_assign(&mut self, o: Fx) {
        self.0 -= o.0;
    }
}

/// a part of a share going to no line (`line_upkeep`'s `[-1, rest]`)
const REST: u32 = u32::MAX;

/// An edge's parts, as last worked out.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EdgeK {
    /// its ends while alive
    ends: Option<[u32; 2]>,
    /// its upkeep and its route-km where upkept (`edge_upkeep`)
    up: Fx,
    km: Fx,
    /// its price to the total and the built cost
    cost: [Fx; 2],
}

/// A node's parts, as last worked out.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NodeK {
    /// 1 an upkept station, 2 an upkept port
    kind: u8,
    /// a station's upkeep (`station_upkeep`) where an upkept station
    station: Fx,
    /// the price of the upgrades of a constructed station with some (`upkept_upgrades_value`), and whether at a port
    ups: Fx,
    port: bool,
    /// its upgrades' price where it is a station (`upgrade_upkeep_by_node`'s own part, a blueprint station's too)
    own: f64,
    /// its station, upgrades and junction to the total and the built cost (`Network::node_cost_parts`)
    cost: [Fx; 2],
}

/// What a line's money reads of it, compared at every refresh: a change names it.
#[derive(Clone, Debug, Default, PartialEq)]
struct LineMoney {
    key: [u64; MONEY_KEY],
    /// trains a day, weighting its share of the upkeep (`line_upkeep_periods`)
    weight: f64,
}

/// A line's share units, as last worked out: the edges of its plan and the stations it is counted at.
#[derive(Clone, Debug, Default, PartialEq)]
struct LineK {
    edges: Vec<u32>,
    stops: Vec<u32>,
    /// a bus line's road metres (`Network::bus_road_m`), NaN for any other line
    road: f64,
    /// the edges its trains run (`capacity::edges_in_use`: an ok rail line's live runs), ascending
    used: Vec<u32>,
    /// its moves through nodes (`capacity::move_conflicts`: node, arriving entry, leaving entry), ascending
    moves: Vec<(u32, u32, u32)>,
    /// what these were made from (`units_key`)
    key: u64,
}

#[derive(Debug)]
pub(super) struct Kept {
    /// what was named since the last refresh; `all`: everything (a new model, a load); `lines` named outside the
    /// service (a price factor moved); `extras`: the passages or buildings changed
    edges: Vec<u32>,
    nodes: Vec<u32>,
    lines: Vec<u32>,
    extras: bool,
    all: bool,
    /// what the state's full read last found moved with nothing naming it, for `TrackApi::kept_check`
    unnamed: Option<&'static str>,
    /// the water mask the parts were worked out under
    mask: u32,
    e: Vec<EdgeK>,
    n: Vec<NodeK>,
    money: Vec<LineMoney>,
    l: Vec<LineK>,
    // the edge and node totals
    track: Fx,
    km: Fx,
    stations: Fx,
    count: i64,
    port_count: i64,
    own: Fx,
    at_ports: Fx,
    cost: [Fx; 2],
    /// moves on whenever a refresh found anything named: the lines' money and the track limits are of one model state
    pub(super) gen: u64,
    lines_gen: u64,
    fleet: [u32; FLEETS],
    /// per kind of day (`DAY_ROWS`) and period, the running cost an hour (`running_cost_period`)
    running: [f64; DAY_ROWS * 5],
    limits: Option<(u64, Vec<[(f64, u32); SLOTS]>)>,
    /// the shares, for the hours a day of each period they were asked with
    hours: Vec<f64>,
    share: Vec<Fx>,
    rest: Fx,
    unit_e: Vec<Vec<(u32, Fx)>>,
    unit_n: Vec<Vec<(u32, Fx)>>,
    unit_b: Vec<Vec<(u32, Fx)>>,
    /// the lines counted stopping at each station (`upkeep_shares`' stopping), ascending
    stop_at: HashMap<u32, Vec<u32>>,
    /// per edge, how many lines' trains run it (`LineK::used`)
    in_use: Vec<u32>,
    /// per node, the moves the lines take through it (`LineK::moves`: arriving and leaving entries, how many lines),
    /// and where two of those cross (`capacity::move_conflicts`' pairs), by node
    node_moves: HashMap<u32, BTreeMap<(u32, u32), u32>>,
    pairs: BTreeMap<u32, Vec<[u32; 5]>>,
    pairs_dirty: Vec<u32>,
    /// the lines whose weight or share units moved in this refresh: their every edge and stop has its share worked out
    /// again
    weighed: Vec<u32>,

    /// `profile_meta`: per line its profiles' rows (phase count, trip seconds, path length), the slots worked out they
    /// were made for, the rows laid out with each one's first phase, and the first line to lay out again
    meta_rows: Vec<Vec<[f32; 3]>>,
    meta_slots: Vec<usize>,
    meta: Vec<f32>,
    meta_from: Option<usize>,
    /// the list flags the clock worker's rows read, per edge and node (`edge_flags`, `node_flags`), and the records whose
    /// flags or upgrade upkeep moved since `take_flags` (`flags_all`: every one)
    ef: Vec<[u32; 8]>,
    nf: Vec<[u32; 7]>,
    flag_edges: Vec<u32>,
    flag_nodes: Vec<u32>,
    flags_all: bool,
    /// what was read last, for this `gen`: the upkeep, the costs and the shares
    out_upkeep: Option<(u64, Vec<f64>)>,
    out_costs: Option<(u64, [f64; 2])>,
    out_shares: Option<(u64, Vec<f64>)>,
    /// each BRT line's part of the lanes' upkeep (`Network::brt_upkeep_m`), as the bus units were worked out with
    lanes: Vec<f64>,
    /// each station's upgrade upkeep a game day (`upgrade_upkeep_by_node`), those over 0, and the stations with a passage's
    /// or building's part, with what the passages and buildings were as last read
    up: BTreeMap<u32, f64>,
    up_of: Vec<f64>,
    extra: Vec<u32>,
    extra_key: u64,
    /// each station building's members as last worked out (`building::members`, a pass over the stations), by its place in
    /// `Network::buildings`: worked out again when the buildings change, or for a station named that is or was a member
    members: Vec<Vec<u32>>,
    empty: Fx,
}

impl Default for Kept {
    fn default() -> Self {
        Kept { edges: vec![], nodes: vec![], lines: vec![], extras: false, all: true, unnamed: None, mask: 0, e: vec![], n: vec![], money: vec![], l: vec![], track: Fx::default(), km: Fx::default(), stations: Fx::default(), count: 0, port_count: 0, own: Fx::default(), at_ports: Fx::default(), cost: [Fx::default(); 2], gen: 0, lines_gen: u64::MAX, fleet: [0; FLEETS], running: [0.0; DAY_ROWS * 5], limits: None, hours: vec![], share: vec![], rest: Fx::default(), unit_e: vec![], unit_n: vec![], unit_b: vec![], stop_at: HashMap::new(), in_use: vec![], node_moves: HashMap::new(), pairs: BTreeMap::new(), pairs_dirty: vec![], weighed: vec![], meta_rows: vec![], meta_slots: vec![], meta: vec![], meta_from: Some(0), ef: vec![], nf: vec![], flag_edges: vec![], flag_nodes: vec![], flags_all: true, out_upkeep: None, out_costs: None, out_shares: None, lanes: vec![], up: BTreeMap::new(), up_of: vec![], extra: vec![], extra_key: 0, members: vec![], empty: Fx::default() }
    }
}

fn edge_now(n: &Network, e: usize) -> EdgeK {
    if e >= n.edge_alive.len() || !n.edge_alive[e] {
        return EdgeK::default();
    }
    let (up, km) = match edge_upkeep(n, e) {
        Some(u) => (Fx::of(u), Fx::of(n.edge_len[e] / 1000.0)),
        None => (Fx::default(), Fx::default()),
    };
    let c = Fx::of(n.edge_cost[e]);
    EdgeK { ends: Some([n.edge_a[e], n.edge_b[e]]), up, km, cost: [c, if n.edge_built[e] { c } else { Fx::default() }] }
}

fn node_now(n: &Network, i: usize) -> NodeK {
    if i >= n.node_alive.len() || !n.node_alive[i] {
        return NodeK::default();
    }
    let kind = if upkept_station(n, i) { 1 } else if upkept_port(n, i) { 2 } else { 0 };
    let station = if kind == 1 { Fx::of(station_upkeep(n, i)) } else { Fx::default() };
    let with_ups = n.node_platform[i] > 0 && n.node_built[i] && n.node_upgrades[i] != 0;
    let ups = if with_ups { Fx::of(n.upgrades_cost(i)) } else { Fx::default() };
    let own = if n.node_platform[i] > 0 { n.upgrades_cost(i) } else { 0.0 };
    let (t, b) = n.node_cost_parts(i);
    let sum = |v: [f64; 3]| {
        let mut s = Fx::default();
        for x in v {
            s += Fx::of(x);
        }
        s
    };
    NodeK { kind, station, ups, port: n.node_berth[i].is_some(), own, cost: [sum(t), sum(b)] }
}

/// the numbers in a line's money key (`line_money`)
const MONEY_KEY: usize = 3 * SLOTS + 11;

/// what a line's money reads of it, each number whole (its key), and its weight for the shares at `slot_hours`
fn line_money(w: &TrackWorld, run_factor: &[f64], slot_hours: &[f64; SLOTS], l: usize) -> LineMoney {
    let (n, s) = (&w.net, &w.svc.lines[l]);
    let mut key = [0u64; MONEY_KEY];
    let mut k = 0;
    let mut put = |x: u64| {
        key[k] = x;
        k += 1;
    };
    put(s.ok as u64 | (s.running as u64) << 1 | (s.bus.is_some() as u64) << 2 | (s.ferry as u64) << 3 | (s.live_runs() as u64) << 4);
    put(s.cars as u64);
    for x in s.trains {
        put(x as u64);
    }
    for x in s.round_trip {
        put(x.to_bits());
    }
    // (not its local time: the kept running costs are every line's, `running_cost_period`)
    put(s.runs[0].len.to_bits());
    put(run_factor.get(l).copied().unwrap_or(1.0).to_bits());
    if l < n.line_count() {
        put(n.line_ok(l as u32) as u64 | (fleet_kind(n, l) as u64) << 1);
        let t = n.trains_of(l);
        put(t.is_some() as u64);
        for x in t.unwrap_or_default() {
            put(x as u64);
        }
        for x in n.line_tph[l] {
            put(x.to_bits() as u64);
        }
    }
    let weight = if !s.running || l >= n.line_count() { 0.0 } else { n.slots().map(|k| n.line_tph[l][k] as f64 * slot_hours[k]).sum() };
    LineMoney { key, weight }
}

/// line `l`'s share units: the edges of its plan (where `lines_on_edge` lists it) and, while it weighs, the stations it is
/// counted at (`upkeep_shares`' stopping)
fn line_units(w: &TrackWorld, l: usize, weight: f64) -> LineK {
    let (n, svc) = (&w.net, &w.svc.lines);
    let ok = l < n.line_count() && n.line_ok(l as u32);
    let mut edges: Vec<u32> = if ok { n.line_path(l as u32).iter().map(|&p| crate::track::net::path_edge(p)).collect() } else { vec![] };
    edges.sort_unstable();
    edges.dedup();
    let mut stops = vec![];
    if weight > 0.0 && ok {
        stops = if svc[l].part.is_some() { svc[l].stops.clone() } else { n.line_stops(l as u32).to_vec() };
        if !svc[l].is_bus() {
            for r in 0..2 {
                stops.extend((0..svc[l].runs[r].stop_idx.len()).filter_map(|k| svc[l].stop_node(r, k)));
            }
        }
        stops.sort_unstable();
        stops.dedup();
    }
    let road = if ok && n.is_bus(l as u32) { n.bus_road_m(l as u32) } else { f64::NAN };
    let s = &svc[l];
    let mut used: Vec<u32> = if s.ok && !s.is_bus() { s.runs.iter().take(s.live_runs()).flat_map(|r| r.segs.iter().map(|&p| crate::track::net::path_edge(p))).collect() } else { vec![] };
    used.sort_unstable();
    used.dedup();
    let mut moves = vec![];
    if s.ok && !s.is_bus() {
        for run in s.runs.iter().take(s.live_runs()) {
            for k in 1..run.nodes.len().saturating_sub(1) {
                moves.push((run.nodes[k], run.segs[k - 1], run.segs[k]));
            }
        }
    }
    moves.sort_unstable();
    moves.dedup();
    LineK { edges, stops, road, used, moves, key: units_key(w, l, weight) }
}

/// T-4299: an edge's list flags as the clock worker's rows read them (`off_edges`, `hsr_edges`, `crossover_edges` and
/// `crossover_partners`, `oneway_edges`, `join_edges`, `edge_limits`, `edge_ramps`): off, high speed, crossover, its
/// partner (`u32::MAX` none), way, joins, limit, ramp; a gone edge's all none
pub(super) fn edge_flags(n: &Network, e: u32) -> [u32; 8] {
    if !n.edge_ok(e) {
        return [0, 0, 0, u32::MAX, 0, 0, 0, 0];
    }
    let i = e as usize;
    let cross = n.edge_crossover[i];
    let counted = |which: u32| if n.join_away(e << 1 | which).is_some() { n.edge_join[i][which as usize] as u32 } else { 0 };
    let join = if n.edge_tracks[i] == 1 { counted(0) | counted(1) << 2 } else { 0 };
    [n.edge_off[i] as u32, n.edge_hsr(e) as u32, cross as u32, if cross { crossover_partner(n, e).unwrap_or(u32::MAX) } else { u32::MAX }, n.edge_way(e) as u32, join, n.edge_limit[i] as u32, n.edge_ramp[i] as u32]
}

/// T-4299: a node's list flags as the rows read them (`off_stations`, `named_stations`, `unbound_stations`,
/// `one_track_stations`, `one_platform_stations`, `station_fares`, `station_upgrades`): off, named, unbound, one track (0,
/// 1, 2: `station_one`), one platform, fare zone, upgrades; a gone node's all 0
pub(super) fn node_flags(n: &Network, v: u32) -> [u32; 7] {
    if !n.node_ok(v) {
        return [0; 7];
    }
    let i = v as usize;
    let station = n.is_station(v);
    let one = if !n.one_track(v) { 0 } else if n.one_side(v).is_some() { 1 } else { 2 };
    [(n.node_platform[i] > 0 && n.node_off[i]) as u32, (station && n.node_named[i]) as u32, (station && !n.node_own[i].is_empty()) as u32, one, n.one_platform(v) as u32, if station { n.node_fare[i] } else { 0 }, if station { n.node_upgrades[i] as u32 } else { 0 }]
}

/// what is in one of two ascending lists and not the other, into `out`
fn symdiff<T: Ord + Copy>(a: &[T], b: &[T], out: &mut Vec<T>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x == y => {
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => {
                out.push(*x);
                i += 1;
            }
            (Some(_), Some(y)) => {
                out.push(*y);
                j += 1;
            }
            (Some(x), None) => {
                out.push(*x);
                i += 1;
            }
            (None, Some(y)) => {
                out.push(*y);
                j += 1;
            }
            (None, None) => break,
        }
    }
}

/// what line `l`'s share units are made from, as one number: they are worked out again only when it moves
fn units_key(w: &TrackWorld, l: usize, weight: f64) -> u64 {
    let (n, s) = (&w.net, &w.svc.lines[l]);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut add = |x: u64| h = (h.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    add(s.ok as u64 | (s.is_bus() as u64) << 1 | (s.one_way as u64) << 2 | (s.part.is_some() as u64) << 3 | ((weight > 0.0) as u64) << 4);
    if l < n.line_count() {
        add(n.line_ok(l as u32) as u64);
        for xs in [n.line_path(l as u32), n.line_stops(l as u32)] {
            add(xs.len() as u64);
            for &x in xs {
                add(x as u64);
            }
        }
        if let Some(r) = n.line_bus.get(l).and_then(|b| b.as_ref()) {
            add(r.points.len() as u64);
            for p in &r.points {
                add(p[0].to_bits());
                add(p[1].to_bits());
            }
            for &k in &r.kinds {
                add(k as u64);
            }
        }
    }
    add(s.stops.len() as u64);
    for &x in &s.stops {
        add(x as u64);
    }
    for r in &s.runs {
        add(r.segs.len() as u64);
        for &x in &r.segs {
            add(x as u64);
        }
        add(r.nodes.len() as u64);
        for &x in &r.nodes {
            add(x as u64);
        }
        add(r.stop_idx.len() as u64);
        for &x in &r.stop_idx {
            add(x as u64);
        }
        for &(k, v) in &r.away {
            add(k as u64);
            add(v as u64);
        }
    }
    h
}

/// line `li`'s rows of `profile_meta` but their first phase: per profile (run, slot, variant) its phase count, trip
/// seconds and path length, zeros where it has none
fn meta_rows_of(w: &TrackWorld, li: usize) -> Vec<[f32; 3]> {
    let l = &w.svc.lines[li];
    let mut out = Vec::with_capacity(2 * SLOTS * PROFILE_VARS);
    for r in 0..2 {
        for k in 0..SLOTS {
            for v in 0..PROFILE_VARS {
                out.push(if l.ok && w.net.slot_worked(k) && (v == 0 || v <= l.runs[r].vars[k].len()) {
                    [l.prof_of(r, k, v as u8).phases.len() as f32, l.trip_len_var(&w.net, li as u32, r, k, v as u8) as f32, l.runs[r].len as f32]
                } else {
                    [0.0; 3]
                });
            }
        }
    }
    out
}

impl Kept {
    /// What an applied edit touched.
    pub(super) fn note(&mut self, d: &crate::track::world::Dirty) {
        self.note_records(&d.edges, &d.nodes);
    }

    pub(super) fn note_records(&mut self, edges: &[u32], nodes: &[u32]) {
        if self.all {
            return;
        }
        if self.edges.len() + self.nodes.len() > 1 << 18 {
            self.everything();
            return;
        }
        self.edges.extend_from_slice(edges);
        self.nodes.extend_from_slice(nodes);
    }

    /// Lines whose money moved with no service note (`set_line_run_factors`).
    pub(super) fn note_lines(&mut self, lines: &[u32]) {
        if !self.all {
            self.lines.extend_from_slice(lines);
        }
    }

    /// The passages or station buildings changed (they are set outside an edit).
    pub(super) fn note_extras(&mut self) {
        self.extras = true;
    }

    pub(super) fn everything(&mut self) {
        self.edges.clear();
        self.nodes.clear();
        self.lines.clear();
        self.all = true;
    }

    /// what the state's full read found moved that nothing named, since the last call
    pub(super) fn take_unnamed(&mut self) -> Option<&'static str> {
        self.unnamed.take()
    }

    /// Bring every part up to the model: `noted` the lines the service noted (None: every one), `priced` the edges
    /// re-priced outside an edit.
    /// `full`: also every line's money numbers and the passages and buildings compared (a pass over the lines), a net for
    /// a change nothing named; the state's first read (`take_changed_lines`) asks it, every other read takes the named
    /// changes alone, so a change it catches is recorded as a namer missing (`unnamed`).
    pub(super) fn refresh(&mut self, w: &TrackWorld, run_factor: &[f64], noted: Option<Vec<u32>>, full: bool) {
        let n = &w.net;
        if noted.is_none() || n.mask_epoch() != self.mask {
            self.all = true;
        }
        if self.all {
            let hours = std::mem::take(&mut self.hours);
            *self = Kept { all: false, mask: n.mask_epoch(), gen: self.gen + 1, ..Kept::default() };
            self.whole(w, run_factor, hours);
            return;
        }
        let mut edges = std::mem::take(&mut self.edges);
        let mut nodes = std::mem::take(&mut self.nodes);
        let mut noted = noted.unwrap_or_default();
        noted.append(&mut self.lines);
        noted.sort_unstable();
        noted.dedup();
        let extras = std::mem::take(&mut self.extras);
        if edges.is_empty() && nodes.is_empty() && noted.is_empty() && !extras {
            if !full {
                return;
            }
            self.unnamed = if self.money_moved(w, run_factor) { Some("a line's money") } else if self.extras_moved(n) { Some("the passages or buildings") } else { return };
        }
        self.gen += 1;
        edges.sort_unstable();
        edges.dedup();
        if self.e.len() < n.edge_alive.len() {
            self.e.resize(n.edge_alive.len(), EdgeK::default());
        }
        if self.n.len() < n.node_alive.len() {
            self.n.resize(n.node_alive.len(), NodeK::default());
        }
        // the edges' parts, their ends before and after named
        for &e in &edges {
            let (i, now) = (e as usize, edge_now(n, e as usize));
            let Some(was) = self.e.get(i).copied() else { continue };
            for k in [was.ends, now.ends].into_iter().flatten() {
                nodes.extend(k);
            }
            if now == was {
                continue;
            }
            self.track -= was.up;
            self.km -= was.km;
            self.cost[0] -= was.cost[0];
            self.cost[1] -= was.cost[1];
            self.track += now.up;
            self.km += now.km;
            self.cost[0] += now.cost[0];
            self.cost[1] += now.cost[1];
            self.e[i] = now;
        }
        nodes.sort_unstable();
        nodes.dedup();
        let mut named_nodes = vec![];
        for &v in &nodes {
            let i = v as usize;
            let Some(was) = self.n.get(i).copied() else { continue };
            let now = node_now(n, i);
            if now == was {
                continue;
            }
            self.add_node(&was, -1);
            self.add_node(&now, 1);
            self.n[i] = now;
            named_nodes.push(v);
        }
        self.flags_named(n, &edges, &nodes);
        // the lines whose money moved join the noted ones; the share units of those whose edges or stops moved, before and
        // after, are worked out again
        let (mut unit_edges, mut unit_nodes) = (edges, named_nodes);
        self.lines_moved(w, run_factor, &mut noted);
        self.lines_noted(w, &noted, &mut unit_edges, &mut unit_nodes);
        self.meta_noted(w, &noted);
        let mut at = std::mem::take(&mut self.pairs_dirty);
        at.extend(&nodes);
        at.sort_unstable();
        at.dedup();
        self.pairs_at(n, &at);
        // (a station's upgrade upkeep moves with its own record, or a passage's or building's part: `extras`; a station
        // named may join or leave a building)
        let mut named = nodes.clone();
        let k = named.len();
        self.extras(w, &mut named);
        unit_nodes.extend(&named[k..]);
        if !self.hours.is_empty() {
            self.shares_named(w, &noted, unit_edges, unit_nodes);
        }
        self.weighed.clear();
    }

    /// The parts of `w` worked out from nothing (`TrackApi::kept_check`'s twin).
    pub(super) fn from_whole(w: &TrackWorld, run_factor: &[f64], hours: Vec<f64>) -> Kept {
        let mut k = Kept { all: false, mask: w.net.mask_epoch(), ..Kept::default() };
        k.whole(w, run_factor, hours);
        k
    }

    /// Every part worked out from nothing (`refresh` after a load, `from_whole`).
    pub(super) fn whole(&mut self, w: &TrackWorld, run_factor: &[f64], hours: Vec<f64>) {
        let n = &w.net;
        self.hours = hours;
        self.e = (0..n.edge_alive.len()).map(|e| edge_now(n, e)).collect();
        self.n = (0..n.node_alive.len()).map(|i| node_now(n, i)).collect();
        for k in self.e.clone() {
            self.track += k.up;
            self.km += k.km;
            self.cost[0] += k.cost[0];
            self.cost[1] += k.cost[1];
        }
        for k in self.n.clone() {
            self.add_node(&k, 1);
        }
        self.lines_moved(w, run_factor, &mut vec![]);
        self.weighed.clear();
        self.lines_all(w);
        self.meta_all(w);
        self.ef = (0..n.edge_alive.len() as u32).map(|e| edge_flags(n, e)).collect();
        self.nf = (0..n.node_alive.len() as u32).map(|v| node_flags(n, v)).collect();
        self.flags_all = true;
        let mut all: Vec<u32> = (0..self.n.len() as u32).collect();
        self.extras(w, &mut all);
        if !self.hours.is_empty() {
            self.units_all(w);
        }
    }

    /// every line's share units and the stations' stopping lines from nothing
    fn lines_all(&mut self, w: &TrackWorld) {
        self.l = (0..w.svc.lines.len()).map(|l| line_units(w, l, self.weight(l as u32))).collect();
        self.stop_at.clear();
        self.in_use = vec![0; w.net.edge_alive.len()];
        for (l, k) in self.l.iter().enumerate() {
            for &s in &k.stops {
                self.stop_at.entry(s).or_default().push(l as u32);
            }
            for &e in &k.used {
                if let Some(c) = self.in_use.get_mut(e as usize) {
                    *c += 1;
                }
            }
        }
        self.node_moves.clear();
        for k in &self.l {
            for &(v, p, q) in &k.moves {
                *self.node_moves.entry(v).or_default().entry((p, q)).or_default() += 1;
            }
        }
        self.pairs.clear();
        let nodes: Vec<u32> = self.node_moves.keys().copied().collect();
        self.pairs_at(&w.net, &nodes);
    }

    /// the crossing pairs at `nodes` from the moves through them (`capacity::move_conflicts`, node by node)
    fn pairs_at(&mut self, n: &Network, nodes: &[u32]) {
        for &v in nodes {
            self.pairs.remove(&v);
            let Some(m) = self.node_moves.get(&v) else { continue };
            if !n.node_ok(v) || n.node_ports[v as usize].n < 3 || n.node_flying[v as usize] {
                continue;
            }
            let moves: Vec<(f64, f64, u32, u32)> = m.keys().filter(|&&(p, q)| !n.turns_back(p, q)).map(|&(p, q)| {
                let (pa, pb) = n.move_at(v, p, q);
                (pa, pb, p, q)
            }).collect();
            let mut out = vec![];
            for a in 0..moves.len() {
                for b in a + 1..moves.len() {
                    let (x, y) = (&moves[a], &moves[b]);
                    if (x.0 - y.0) * (x.1 - y.1) < 0.0 {
                        out.push([v, x.2, x.3, y.2, y.3]);
                    }
                }
            }
            if !out.is_empty() {
                self.pairs.insert(v, out);
            }
        }
    }

    /// The list flags of the edges and nodes named, the edges at the nodes named (a join counts by the track beyond an
    /// end, a one-platform station by its tracks) and a crossover's partner before and after; those that moved go in
    /// the flags' log.
    fn flags_named(&mut self, n: &Network, edges: &[u32], nodes: &[u32]) {
        if self.ef.len() < n.edge_alive.len() {
            self.ef.resize(n.edge_alive.len(), [0, 0, 0, u32::MAX, 0, 0, 0, 0]);
        }
        if self.nf.len() < n.node_alive.len() {
            self.nf.resize(n.node_alive.len(), [0; 7]);
        }
        let mut es: Vec<u32> = edges.to_vec();
        for &v in nodes {
            if n.node_ok(v) {
                es.extend(n.node_ports[v as usize].ends().iter().map(|&c| c >> 1));
            }
        }
        es.sort_unstable();
        es.dedup();
        let mut partners = vec![];
        for &e in &es {
            let Some(was) = self.ef.get(e as usize).copied() else { continue };
            let now = edge_flags(n, e);
            partners.extend([was[3], now[3]].into_iter().filter(|&p| p != u32::MAX && es.binary_search(&p).is_err()));
            if now != was {
                self.ef[e as usize] = now;
                self.flag_edges.push(e);
            }
        }
        for e in partners {
            let Some(was) = self.ef.get(e as usize).copied() else { continue };
            let now = edge_flags(n, e);
            if now != was {
                self.ef[e as usize] = now;
                self.flag_edges.push(e);
            }
        }
        for &v in nodes {
            let Some(was) = self.nf.get(v as usize).copied() else { continue };
            let now = node_flags(n, v);
            if now != was {
                self.nf[v as usize] = now;
                self.flag_nodes.push(v);
            }
        }
    }

    /// The records whose list flags or upgrade upkeep moved since the last call: `[edge count, edges..., nodes...]`
    /// ascending, or `[u32::MAX]` for every one.
    pub(super) fn take_flags(&mut self) -> Vec<u32> {
        let (mut e, mut v) = (std::mem::take(&mut self.flag_edges), std::mem::take(&mut self.flag_nodes));
        if std::mem::take(&mut self.flags_all) {
            return vec![u32::MAX];
        }
        e.sort_unstable();
        e.dedup();
        v.sort_unstable();
        v.dedup();
        let mut out = vec![e.len() as u32];
        out.extend(e);
        out.extend(v);
        out
    }

    /// the list flags and upgrade upkeep that differ from `whole`'s (worked out from nothing): "" for none
    pub(super) fn flags_differ(&self, whole: &Kept) -> String {
        let ne = self.ef.len().max(whole.ef.len());
        let e: Vec<usize> = (0..ne).filter(|&i| self.ef.get(i).copied().unwrap_or([0, 0, 0, u32::MAX, 0, 0, 0, 0]) != whole.ef.get(i).copied().unwrap_or([0, 0, 0, u32::MAX, 0, 0, 0, 0])).take(6).collect();
        let nn = self.nf.len().max(whole.nf.len());
        let v: Vec<usize> = (0..nn).filter(|&i| self.nf.get(i).copied().unwrap_or([0; 7]) != whole.nf.get(i).copied().unwrap_or([0; 7]) || self.up_of(i as u32).to_bits() != whole.up_of(i as u32).to_bits()).take(6).collect();
        if e.is_empty() && v.is_empty() { String::new() } else { format!("list flags of edges {e:?} and nodes {v:?}") }
    }

    /// a node's upgrade upkeep a game day as kept (`upgrade_upkeep`'s figure, 0 for none)
    pub(super) fn up_of(&self, v: u32) -> f64 {
        self.up_of.get(v as usize).copied().unwrap_or(0.0)
    }

    fn add_node(&mut self, k: &NodeK, sign: i64) {
        let add = |t: &mut Fx, v: Fx| if sign > 0 { *t += v } else { *t -= v };
        add(&mut self.stations, k.station);
        if k.port { add(&mut self.at_ports, k.ups) } else { add(&mut self.own, k.ups) }
        add(&mut self.cost[0], k.cost[0]);
        add(&mut self.cost[1], k.cost[1]);
        self.count += sign * (k.kind == 1) as i64;
        self.port_count += sign * (k.kind == 2) as i64;
    }

    /// whether any line's money numbers moved since they were last worked out (a pass over the lines)
    fn money_moved(&self, w: &TrackWorld, run_factor: &[f64]) -> bool {
        let sh = self.slot_hours(&w.net);
        w.svc.lines.len() != self.money.len() || (0..w.svc.lines.len()).any(|l| line_money(w, run_factor, &sh, l) != self.money[l])
    }

    fn slot_hours(&self, n: &Network) -> [f64; SLOTS] {
        let mut sh = [0.0; SLOTS];
        for (p, &s) in n.period_slot().iter().enumerate() {
            sh[s] += self.hours.get(p).copied().unwrap_or(0.0);
        }
        sh
    }

    /// Every line's money worked out again (a pass over the lines: each one's few numbers) and the lines whose money
    /// numbers moved added to `moved`; the fleets' needs and the running costs with them.
    fn lines_moved(&mut self, w: &TrackWorld, run_factor: &[f64], moved: &mut Vec<u32>) {
        let sh = self.slot_hours(&w.net);
        let nl = w.svc.lines.len();
        let now: Vec<LineMoney> = (0..nl).map(|l| line_money(w, run_factor, &sh, l)).collect();
        for l in 0..nl.max(self.money.len()) {
            if now.get(l) != self.money.get(l) {
                moved.push(l as u32);
            }
            if now.get(l).map(|m| m.weight.to_bits()) != self.money.get(l).map(|m| m.weight.to_bits()) {
                self.weighed.push(l as u32);
            }
        }
        moved.sort_unstable();
        moved.dedup();
        self.money = now;
        let mut most = self.take_limits();
        self.fleet = cars_needed_with(w, &mut most);
        self.put_limits(most);
        let mut running = [0.0; DAY_ROWS * 5];
        for (k, r) in running.iter_mut().enumerate() {
            let row = (k / 5).min(DAY_ROWS - 1);
            *r = super::money::running_cost_with(w, run_factor, row, w.net.period_slot()[k % 5], f64::NAN);
        }
        self.running = running;
        self.lines_gen = self.gen;
    }

    /// The track limits (`Capacity::line_limits`) as worked out for this model state (`gen`), None until a weekend count
    /// asks: at most once a state. To be handed back with `put_limits`.
    pub(super) fn take_limits(&mut self) -> Limits {
        match self.limits.take() {
            Some((g, v)) if g == self.gen => Some(v),
            _ => None,
        }
    }
    pub(super) fn put_limits(&mut self, most: Limits) {
        if let Some(v) = most {
            self.limits = Some((self.gen, v));
        }
    }

    fn meta_all(&mut self, w: &TrackWorld) {
        self.meta_rows = (0..w.svc.lines.len()).map(|l| meta_rows_of(w, l)).collect();
        self.meta_slots = w.net.slots().collect();
        self.meta_from = Some(0);
    }

    /// the noted lines' rows of `profile_meta`; every line's when the slots worked out changed
    fn meta_noted(&mut self, w: &TrackWorld, noted: &[u32]) {
        if w.svc.lines.len() != self.meta_rows.len() || !w.net.slots().eq(self.meta_slots.iter().copied()) {
            return self.meta_all(w);
        }
        for &l in noted {
            let l = l as usize;
            if l >= self.meta_rows.len() {
                continue;
            }
            let now = meta_rows_of(w, l);
            if now != self.meta_rows[l] {
                self.meta_rows[l] = now;
                self.meta_from = Some(self.meta_from.map_or(l, |f| f.min(l)));
            }
        }
    }

    /// `profile_meta`: the rows laid out again from the first line whose rows moved (each row's first phase follows
    /// every row before it), then a copy
    pub(super) fn meta(&mut self) -> Vec<f32> {
        if let Some(from) = self.meta_from.take() {
            let block = 2 * SLOTS * PROFILE_VARS;
            self.meta.resize(self.meta_rows.len() * block * 4, 0.0);
            let mut first: usize = self.meta_rows[..from].iter().flatten().map(|r| r[0] as usize).sum();
            for (l, rows) in self.meta_rows.iter().enumerate().skip(from) {
                for (j, r) in rows.iter().enumerate() {
                    let at = (l * block + j) * 4;
                    self.meta[at..at + 4].copy_from_slice(&[first as f32, r[0], r[1], r[2]]);
                    first += r[0] as usize;
                }
            }
        }
        self.meta.clone()
    }

    /// the share units of the noted lines: their edges and stops before and after go in `edges` and `nodes`
    fn lines_noted(&mut self, w: &TrackWorld, noted: &[u32], edges: &mut Vec<u32>, nodes: &mut Vec<u32>) {
        if self.l.len() < w.svc.lines.len() {
            self.l.resize(w.svc.lines.len(), LineK::default());
        }
        if self.in_use.len() < w.net.edge_alive.len() {
            self.in_use.resize(w.net.edge_alive.len(), 0);
        }
        for &l in noted {
            let i = l as usize;
            if i >= self.l.len() {
                continue;
            }
            let weight = self.money.get(i).map_or(0.0, |m| m.weight);
            // (what the units are made from as one number first: a noted line whose path, stops and runs stand is passed by)
            if i < w.svc.lines.len() && units_key(w, i, weight) == self.l[i].key && self.l[i].key != 0 {
                continue;
            }
            let now = if i < w.svc.lines.len() { line_units(w, i, weight) } else { LineK::default() };
            if now == self.l[i] {
                continue;
            }
            let was = std::mem::replace(&mut self.l[i], now);
            let now = &self.l[i];
            // only what came or went moves anything: an edge or stop kept is shared as before (the line's weight the same)
            let mut used = vec![];
            symdiff(&was.used, &now.used, &mut used);
            for &e in &used {
                let c = &mut self.in_use[e as usize];
                if now.used.binary_search(&e).is_ok() { *c += 1 } else { *c = c.saturating_sub(1) }
            }
            let mut moves = vec![];
            symdiff(&was.moves, &now.moves, &mut moves);
            for &(v, p, q) in &moves {
                let m = self.node_moves.entry(v).or_default();
                if now.moves.binary_search(&(v, p, q)).is_ok() {
                    *m.entry((p, q)).or_default() += 1;
                } else if let Some(c) = m.get_mut(&(p, q)) {
                    *c -= 1;
                    if *c == 0 {
                        m.remove(&(p, q));
                    }
                }
                if m.is_empty() {
                    self.node_moves.remove(&v);
                }
                self.pairs_dirty.push(v);
            }
            // (a line running part of its plan shares only the track it runs: `used` moves that)
            symdiff(&was.edges, &now.edges, edges);
            edges.extend(&used);
            let mut stops = vec![];
            symdiff(&was.stops, &now.stops, &mut stops);
            for &s in &stops {
                if now.stops.binary_search(&s).is_ok() {
                    let v = self.stop_at.entry(s).or_default();
                    if let Err(k) = v.binary_search(&l) {
                        v.insert(k, l);
                    }
                } else if let Some(v) = self.stop_at.get_mut(&s) {
                    v.retain(|&x| x != l);
                    if v.is_empty() {
                        self.stop_at.remove(&s);
                    }
                }
            }
            nodes.extend(stops);
            if was.road.to_bits() != now.road.to_bits() {
                self.weighed.push(l);
            }
        }
    }

    /// what the passages and the station buildings were as last read, as one number
    fn extras_key(n: &Network) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        n.links.hash(&mut h);
        for b in n.buildings.iter() {
            match b {
                Some(b) => {
                    1u8.hash(&mut h);
                    for c in &b.corners {
                        c[0].to_bits().hash(&mut h);
                        c[1].to_bits().hash(&mut h);
                    }
                }
                None => 0u8.hash(&mut h),
            }
        }
        h.finish()
    }

    fn extras_moved(&self, n: &Network) -> bool {
        Self::extras_key(n) != self.extra_key
    }

    /// Each station's upgrade upkeep (`upgrade_upkeep_by_node`) where its own upgrades, a passage's half or a building's
    /// share moved; those whose figure moved go in `nodes` (their share units). The passages and buildings are read whole
    /// (a building's members are a pass over the stations) while there are any and something was named.
    fn extras(&mut self, w: &TrackWorld, nodes: &mut Vec<u32>) {
        let n = &w.net;
        if self.up_of.len() < self.n.len() {
            self.up_of.resize(self.n.len(), 0.0);
        }
        let mut parts: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
        let mut empty = 0.0;
        for ([a, b], price) in n.passages() {
            if n.node_built[a as usize] && n.node_built[b as usize] {
                parts.entry(a).or_default().push(price / 2.0);
                parts.entry(b).or_default().push(price / 2.0);
            }
        }
        // the buildings' members: all of them when the buildings changed, else those a station named can join or leave
        let key = Self::extras_key(n);
        if key != self.extra_key || self.members.len() != n.buildings.len() {
            self.members = n.buildings.iter().map(|b| b.as_ref().map_or(vec![], |b| crate::track::building::members(n, &b.corners))).collect();
        } else {
            for (k, b) in n.buildings.iter().enumerate() {
                let Some(b) = b else { continue };
                if nodes.iter().any(|&v| self.members[k].binary_search(&v).is_ok() || crate::track::building::member(n, &b.corners, v)) {
                    self.members[k] = crate::track::building::members(n, &b.corners);
                }
            }
        }
        for (k, b) in n.buildings.iter().enumerate() {
            let Some(b) = b else { continue };
            let price = crate::track::upgrades::building_price(crate::track::building::area_m2(n, &b.corners));
            let on: Vec<u32> = self.members[k].iter().copied().filter(|&m| n.node_built[m as usize]).collect();
            if on.is_empty() {
                empty += price;
            }
            for &m in &on {
                parts.entry(m).or_default().push(price / on.len() as f64);
            }
        }
        self.empty = Fx::of(crate::track::upgrades::upkeep(empty));
        self.extra_key = key;
        let mut check: Vec<u32> = std::mem::take(&mut self.extra);
        check.extend(parts.keys());
        check.extend(nodes.iter());
        check.sort_unstable();
        check.dedup();
        for v in check {
            let i = v as usize;
            if i >= self.n.len() {
                continue;
            }
            let mut x = self.n[i].own;
            for p in parts.get(&v).map_or(&[][..], |p| &p[..]) {
                x += p;
            }
            let u = crate::track::upgrades::upkeep(x);
            if u.to_bits() != self.up_of[i].to_bits() {
                self.up_of[i] = u;
                nodes.push(v);
                self.flag_nodes.push(v);
            }
            if u > 0.0 { self.up.insert(v, u); } else { self.up.remove(&v); }
        }
        self.extra = parts.into_keys().collect();
    }

    // ---- the shares (`line_upkeep_periods`)

    fn weight(&self, l: u32) -> f64 {
        self.money.get(l as usize).map_or(0.0, |m| m.weight)
    }

    fn edge_unit(&self, w: &TrackWorld, e: usize) -> Vec<(u32, Fx)> {
        let n = &w.net;
        if e >= n.edge_alive.len() {
            return vec![];
        }
        let Some(u) = edge_upkeep(n, e) else { return vec![] };
        let svc = &w.svc.lines;
        let weight_on = |l: u32| if svc.get(l as usize).map_or(true, |s| s.runs_on(e as u32)) { self.weight(l) } else { 0.0 };
        let on = n.lines_on_edge(e as u32);
        let total: f64 = on.iter().map(|&l| weight_on(l)).sum();
        if total > 0.0 { on.iter().map(|&l| (l, Fx::of(u * weight_on(l) / total))).collect() } else { vec![(REST, Fx::of(u))] }
    }

    fn node_unit(&self, w: &TrackWorld, i: usize) -> Vec<(u32, Fx)> {
        let n = &w.net;
        let Some(k) = self.n.get(i) else { return vec![] };
        let up = self.up_of.get(i).copied().unwrap_or(0.0);
        let u = up + match k.kind {
            1 => station_upkeep(n, i),
            2 => port_upkeep(),
            _ => return vec![],
        };
        match self.stop_at.get(&(i as u32)) {
            Some(lines) => {
                let total: f64 = lines.iter().map(|&l| self.weight(l)).sum();
                lines.iter().map(|&l| (l, Fx::of(u * self.weight(l) / total))).collect()
            }
            None => vec![(REST, Fx::of(u))],
        }
    }

    fn bus_unit(&self, w: &TrackWorld, l: usize) -> Vec<(u32, Fx)> {
        let road = self.l.get(l).map_or(f64::NAN, |k| k.road);
        let Some(u) = bus_upkeep_with(&w.net, l as u32, road, &self.lanes) else { return vec![] };
        if self.weight(l as u32) > 0.0 { vec![(l as u32, Fx::of(u))] } else { vec![(REST, Fx::of(u))] }
    }

    fn put_unit(share: &mut Vec<Fx>, rest: &mut Fx, unit: &[(u32, Fx)], sign: i8) {
        for &(l, x) in unit {
            let t: &mut Fx = if l == REST {
                &mut *rest
            } else {
                if share.len() <= l as usize {
                    share.resize(l as usize + 1, Fx::default());
                }
                &mut share[l as usize]
            };
            if sign > 0 { *t += x } else { *t -= x }
        }
    }

    /// the shares for other hours: the weights, the stations' stopping lines and every unit again
    fn shares_all(&mut self, w: &TrackWorld, hours: Vec<f64>) {
        self.hours = hours;
        let sh = self.slot_hours(&w.net);
        let n = &w.net;
        for (l, m) in self.money.iter_mut().enumerate() {
            m.weight = if l >= n.line_count() || !w.svc.lines.get(l).is_some_and(|s| s.running) { 0.0 } else { n.slots().map(|k| n.line_tph[l][k] as f64 * sh[k]).sum() };
        }
        self.lines_all(w);
        self.units_all(w);
    }

    /// every share unit from nothing
    fn units_all(&mut self, w: &TrackWorld) {
        self.lanes = w.net.brt_upkeep_m();
        self.unit_e = (0..w.net.edge_alive.len()).map(|e| self.edge_unit(w, e)).collect();
        self.unit_n = (0..self.n.len()).map(|i| self.node_unit(w, i)).collect();
        self.unit_b = (0..w.svc.lines.len()).map(|l| self.bus_unit(w, l)).collect();
        let (mut share, mut rest) = (vec![Fx::default(); w.svc.lines.len()], Fx::default());
        for u in self.unit_e.iter().chain(&self.unit_n).chain(&self.unit_b) {
            Self::put_unit(&mut share, &mut rest, u, 1);
        }
        self.share = share;
        self.rest = rest;
    }

    /// the share units of the edges and nodes named (and of the lines' moved: `noted`, their bus units)
    fn shares_named(&mut self, w: &TrackWorld, noted: &[u32], mut edges: Vec<u32>, mut nodes: Vec<u32>) {
        let lanes = w.net.brt_upkeep_m();
        let mut buses: Vec<u32> = noted.to_vec();
        for l in 0..lanes.len().max(self.lanes.len()) {
            if lanes.get(l).copied().unwrap_or(0.0).to_bits() != self.lanes.get(l).copied().unwrap_or(0.0).to_bits() {
                buses.push(l as u32);
            }
        }
        self.lanes = lanes;
        // a line whose weight or units moved: every edge and stop of it (a noted line whose neither moved shares as it did)
        for l in std::mem::take(&mut self.weighed) {
            if let Some(k) = self.l.get(l as usize) {
                edges.extend(&k.edges);
                nodes.extend(&k.stops);
            }
        }
        edges.sort_unstable();
        edges.dedup();
        nodes.sort_unstable();
        nodes.dedup();
        buses.sort_unstable();
        buses.dedup();
        let (mut share, mut rest) = (std::mem::take(&mut self.share), self.rest);
        if share.len() < w.svc.lines.len() {
            share.resize(w.svc.lines.len(), Fx::default());
        }
        if self.unit_e.len() < w.net.edge_alive.len() {
            self.unit_e.resize(w.net.edge_alive.len(), vec![]);
        }
        if self.unit_n.len() < self.n.len() {
            self.unit_n.resize(self.n.len(), vec![]);
        }
        if self.unit_b.len() < w.svc.lines.len() {
            self.unit_b.resize(w.svc.lines.len(), vec![]);
        }
        for e in edges {
            let e = e as usize;
            if e >= self.unit_e.len() {
                continue;
            }
            let now = self.edge_unit(w, e);
            Self::put_unit(&mut share, &mut rest, &self.unit_e[e], -1);
            Self::put_unit(&mut share, &mut rest, &now, 1);
            self.unit_e[e] = now;
        }
        for v in nodes {
            let i = v as usize;
            if i >= self.unit_n.len() {
                continue;
            }
            let now = self.node_unit(w, i);
            Self::put_unit(&mut share, &mut rest, &self.unit_n[i], -1);
            Self::put_unit(&mut share, &mut rest, &now, 1);
            self.unit_n[i] = now;
        }
        for l in buses {
            let l = l as usize;
            if l >= self.unit_b.len() {
                continue;
            }
            let now = self.bus_unit(w, l);
            Self::put_unit(&mut share, &mut rest, &self.unit_b[l], -1);
            Self::put_unit(&mut share, &mut rest, &now, 1);
            self.unit_b[l] = now;
        }
        self.share = share;
        self.rest = rest;
    }

    // ---- what is read

    pub(super) fn upkeep(&mut self, w: &TrackWorld) -> Vec<f64> {
        if let Some((g, v)) = &self.out_upkeep {
            if *g == self.gen {
                return v.clone();
            }
        }
        let v = self.upkeep_now(w);
        self.out_upkeep = Some((self.gen, v.clone()));
        v
    }

    fn upkeep_now(&self, w: &TrackWorld) -> Vec<f64> {
        let n = &w.net;
        let own = crate::track::upgrades::upkeep(self.own.get());
        let at_ports = crate::track::upgrades::upkeep(self.at_ports.get());
        let passages = crate::track::upgrades::upkeep(n.passages_cost(true));
        let buildings = crate::track::upgrades::upkeep(crate::track::building::value(n));
        let ups = own + at_ports + passages + buildings;
        let stations = self.stations.get() + own + passages + buildings;
        let lanes = n.brt_upkeep_m();
        let (mut bus_km, mut buses) = (0.0, 0.0);
        for l in n.bus_lines() {
            let road = self.l.get(l as usize).map_or(f64::NAN, |k| k.road);
            if let Some(u) = bus_upkeep_with(n, l, road, &lanes) {
                bus_km += road / 1000.0;
                buses += u;
            }
        }
        let track = self.track.get();
        let port_count = self.port_count as f64;
        let ports = port_count * port_upkeep() + at_ports;
        vec![track + stations + buses + ports, track, stations, self.km.get(), self.count as f64, buses, bus_km, ports, port_count, ups]
    }

    pub(super) fn costs(&mut self, w: &TrackWorld) -> [f64; 2] {
        if let Some((g, v)) = self.out_costs {
            if g == self.gen {
                return v;
            }
        }
        let n = &w.net;
        let road = |l: u32| self.l.get(l as usize).map_or(f64::NAN, |k| k.road);
        let (t, b) = n.other_cost_parts(&road);
        let v = [self.cost[0].get() + t, self.cost[1].get() + b];
        self.out_costs = Some((self.gen, v));
        v
    }

    pub(super) fn fleet(&self) -> [u32; FLEETS] {
        self.fleet
    }

    pub(super) fn running(&self, k: usize) -> f64 {
        self.running[k.min(DAY_ROWS * 5 - 1)]
    }

    /// whether the shares are kept for these hours (else `shares_for` works them out whole for them first)
    pub(super) fn shares_for(&mut self, w: &TrackWorld, hours: &[f64]) -> Vec<f64> {
        if self.hours.len() != hours.len() || self.hours.iter().zip(hours).any(|(a, b)| a.to_bits() != b.to_bits()) {
            self.shares_all(w, hours.to_vec());
            self.out_shares = None;
        }
        if let Some((g, v)) = &self.out_shares {
            if *g == self.gen {
                return v.clone();
            }
        }
        let mut out: Vec<f64> = self.share.iter().enumerate().filter(|(l, _)| *l < w.svc.lines.len()).map(|(l, s)| (l, s.get())).filter(|(_, s)| *s > 0.0).flat_map(|(l, s)| [l as f64, s]).collect();
        out.extend([-1.0, self.rest.get() + self.empty.get()]);
        self.out_shares = Some((self.gen, out.clone()));
        out
    }

    /// `guard_conflicts` with the junction pairs as kept: the flat crossings (few, read whole), then the pairs by node
    pub(super) fn conflicts(&self, w: &TrackWorld) -> Vec<f64> {
        let net = &w.net;
        let mut out = vec![0.0];
        for c in net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))) {
            let at = |e: u32, s: f64| (s / net.edge_len[e as usize].max(1e-9)).clamp(0.0, 1.0);
            out.extend([c.e1 as f64, at(c.e1, c.s1), c.e2 as f64, at(c.e2, c.s2)]);
            out[0] += 1.0;
        }
        let k = out.len();
        out.push(0.0);
        for p in self.pairs.values().flatten() {
            out.extend(p.iter().map(|&v| v as f64));
            out[k] += 1.0;
        }
        out
    }

    /// whether two moves some line takes cross at node `v` (`capacity::conflict_nodes`)
    pub(super) fn busy(&self, v: u32) -> bool {
        self.pairs.contains_key(&v)
    }

    /// `crossings_info` with the edges in use as kept (a pass over the crossings, which are few)
    pub(super) fn crossings(&self, w: &TrackWorld) -> Vec<f64> {
        let n = &w.net;
        let used = |e: u32| self.in_use.get(e as usize).is_some_and(|&c| c > 0);
        let mut out = vec![];
        for c in &n.crossings {
            let (kind, level) = match c.kind {
                CrossKind::Flat(l) => (0.0, l as f64),
                CrossKind::Bad => (1.0, 0.0),
                CrossKind::Overlap => continue,
            };
            let built = n.edge_built[c.e1 as usize] && n.edge_built[c.e2 as usize];
            // T-3745: a built flat crossing shows only with a line on both its edges
            if built && kind == 0.0 && !(used(c.e1) && used(c.e2)) {
                continue;
            }
            out.extend([c.x, c.y, kind, level, built as u8 as f64, n.crossing_trams(c.e1, c.e2) as f64]);
        }
        out
    }

    pub(super) fn upgrade_upkeep(&self) -> Vec<f64> {
        self.up.iter().flat_map(|(&n, &u)| [n as f64, u]).collect()
    }

}
