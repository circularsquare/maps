use super::*;

// ---------------------------------------------------------------------------------------------
// 4. Network graph and station-to-station search

pub struct Graph {
    pub n_st: usize,
    pub n: usize,
    /// Route nodes (index node - n_st): one per stop per direction per line.
    pub rn_station: Vec<u32>,
    pub rn_line: Vec<u32>,
    /// Real ride minutes to the next stop in this direction (0 at the end).
    pub rn_ride: Vec<f32>,
    /// Distance fare in perceived minutes, kept separate from the crowding multiplier.
    pub rn_fare: Vec<f32>,
    /// Departures an hour for each route node in this period.
    pub rn_tph: Vec<f32>,
    /// Average extra real minutes a rider boarding at this route node waits because full trains
    /// leave riders on the platform (T-188, `left_behind`), plus `rn_span` and `rn_bus`. Added to the
    /// boarding's cost.
    pub rn_left: Vec<f32>,
    /// Perceived minutes added to boarding where only a share `phi` of the period's riders have a
    /// train due within their tolerance (T-842, `Network::span`): -ln(phi) / `Params::beta`, the
    /// logit's own form for an option open to that share. 0 with service all period.
    pub rn_span: Vec<f32>,
    /// Perceived minutes added to boarding on a bus line (T-1576): `BUS_BOARD_MIN` on every route
    /// node of a line with a bus-only stop, 0 elsewhere. Part of `rn_left`, but not real minutes.
    pub rn_bus: Vec<f32>,
    /// Each route node's platform less its station's position, metres (`Network::stop_x`): a change
    /// of trains walks from the platform it arrives at to the one it leaves from (T-717).
    pub rn_dx: Vec<f32>,
    pub rn_dy: Vec<f32>,
    /// Stations with a platform away from their position, where a change can walk.
    pub st_spread: Vec<bool>,
    /// Per station, how much of its minutes a wait there counts (`wait_factor`, T-1281).
    pub st_wait: Vec<f32>,
    /// T-2220: per station, whether it is in a station building (`upgrades::IN_BUILDING`). Empty
    /// when none is.
    pub st_bld: Vec<bool>,
    /// Incoming edges, CSR, for destination-based strategy search.
    pub rev_off: Vec<u32>,
    pub rev_from: Vec<u32>,
    pub rev_edge: Vec<u32>,
    pub adj_off: Vec<u32>,
    pub adj_to: Vec<u32>,
    pub adj_cost: Vec<f32>,
    /// For ride edges, the route node the segment starts at; NONE otherwise.
    pub adj_seg: Vec<u32>,
    /// Each route node's ride edge's end (a node id), NONE at a line's end: a rider who boards
    /// there rides at least to it (T-1055).
    pub rn_next: Vec<u32>,
    /// The fare curve's departure from `rn_fare`'s straight line (T-883, T-1143), and each route
    /// node's fare km to the next stop, which it needs; both empty while the curve is that line.
    pub fare_curve: FareCurve,
    pub rn_km: Vec<f32>,
    /// T-1626: each route node's fare multiplier (its line's high-speed one, else 1), and its fare
    /// km to the next stop times that; both empty while every line's is 1. A trip pays the base
    /// once and the rest of its fare times its km-weighted multiplier.
    pub rn_fare_mul: Vec<f32>,
    pub rn_kmw: Vec<f32>,
}

impl Graph {
    /// T-1626: the fare multiplier of route node `r`.
    #[inline]
    pub fn fare_mul(&self, r: usize) -> f32 {
        self.rn_fare_mul.get(r).copied().unwrap_or(1.0)
    }
    /// How much of a change's fixed minutes count at station `v` (T-2220).
    #[inline]
    pub fn change_f(&self, v: usize) -> f32 {
        change_f_of(&self.st_bld, v)
    }
}

/// The walks stay whole: with them at 0.8 too, a building leaving one line's platforms outside
/// swung the parallel line's riders 83% (T-2220, DECISIONS 2026-11-03).
#[inline]
pub(super) fn change_f_of(st_bld: &[bool], v: usize) -> f32 {
    if st_bld.get(v).copied().unwrap_or(false) { upgrades::BUILDING_CHANGE } else { 1.0 }
}

/// Whether each station is in a station building; empty when none is (T-2220).
pub(super) fn in_buildings(net: &Network) -> Vec<bool> {
    let s = net.n_stations();
    if !(0..s).any(|v| net.ups(v) & upgrades::IN_BUILDING != 0) {
        return vec![];
    }
    (0..s).map(|v| net.ups(v) & upgrades::IN_BUILDING != 0).collect()
}

/// The player's fare curve (T-883, T-1143): knots of trip km and perceived minutes, joined by
/// straight stretches and carried on past the last at its slope. `base` and `rate` are what a trip
/// is already charged (the base once and the rate on each ride edge), so a trip costs `extra(km)`
/// more than that. Route choice sees only the straight line; the curve enters each station pair's
/// cost on the expected km of its strategy (`DestSearch::run`), and the assignment puts each pair's
/// extra on the route nodes its riders ride from, by km (`Loads::rn_fare_x`, T-1516), so fare
/// income can count it.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct FareCurve {
    pub km: Vec<f32>,
    pub min: Vec<f32>,
    pub base: f32,
    pub rate: f32,
}

impl FareCurve {
    pub fn is_line(&self) -> bool {
        self.km.len() < 2
    }
    /// The curve's fare for a trip of `km`, perceived minutes.
    pub fn at(&self, km: f32) -> f32 {
        let (k, m, n) = (&self.km, &self.min, self.km.len());
        if n < 2 {
            return self.base + self.rate * km;
        }
        let i = k.partition_point(|&x| x <= km).clamp(1, n - 1);
        (m[i - 1] + (m[i] - m[i - 1]) * (km - k[i - 1]) / (k[i] - k[i - 1])).max(0.0)
    }
    pub fn extra(&self, km: f32) -> f32 {
        if self.is_line() { 0.0 } else { self.at(km) - self.base - self.rate * km }
    }
}

/// A wait is plain minutes, half the headway (Anita, T-585), times the station's `wait_factor`;
/// the change's fixed cost times its `Graph::change_f`.
pub fn board_cost(headway_min: f32, wait_f: f32, change_f: f32, p: &Params) -> f32 {
    wait_f * headway_min / 2.0 + p.change_at(change_f)
}

/// T-1430: an inter-city trip's first wait, planned around the timetable: half the headway, but at
/// most `PLANNED_EARLY_MIN` plus a sixth of it (on average a quarter of the headway from the time
/// wanted, at two thirds of a platform wait's weight) and at most `PLANNED_WAIT_CAP_MIN`.
pub const PLANNED_EARLY_MIN: f32 = 5.0;
pub const PLANNED_WAIT_CAP_MIN: f32 = 30.0;
pub fn planned_board_cost(headway_min: f32, wait_f: f32, change_f: f32, p: &Params) -> f32 {
    wait_f * (headway_min / 2.0).min(PLANNED_EARLY_MIN + headway_min / 6.0).min(PLANNED_WAIT_CAP_MIN) + p.change_at(change_f)
}

pub fn build_graph(net: &Network, p: &Params) -> Graph {
    let s = net.n_stations();
    let mut rn_station = vec![];
    let mut rn_line = vec![];
    let mut rn_ride = vec![];
    let mut rn_next = vec![];
    let (mut rn_dx, mut rn_dy) = (vec![], vec![]);
    let mut st_spread = vec![false; s];
    let st_wait: Vec<f32> = (0..s).map(|v| wait_factor(net, v)).collect();
    let st_bld = in_buildings(net);
    let platforms = net.stop_x.len() == net.line_stops.len() && net.stop_y.len() == net.line_stops.len();
    // route nodes where a circle's train rides on round to the first stop (T-798): nobody boards
    // there, since the first stop's own route node has the same trains
    let mut rn_wrap: Vec<bool> = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        let wrap = net.line_wrap.get(l).copied().unwrap_or(false);
        for dir in 0..2 {
            let base = s + rn_station.len();
            let len = b - a;
            for i in 0..len {
                let k = if dir == 0 { a + i } else { b - 1 - i };
                let st = net.line_stops[k] as usize;
                rn_station.push(st as u32);
                rn_line.push(l as u32);
                let (dx, dy) = if platforms { (net.stop_x[k] - net.st_x[st], net.stop_y[k] - net.st_y[st]) } else { (0.0, 0.0) };
                // under a metre is the merge's rounding, not a walk
                let (dx, dy) = if dx.hypot(dy) < 1.0 { (0.0, 0.0) } else { (dx, dy) };
                st_spread[st] |= dx != 0.0 || dy != 0.0;
                rn_dx.push(dx);
                rn_dy.push(dy);
                let ride = if i + 1 < len {
                    if dir == 0 {
                        net.hop_min[k]
                    } else if net.hop_back.is_empty() {
                        net.hop_min[k - 1]
                    } else {
                        net.hop_back[k]
                    }
                } else if wrap {
                    if dir == 0 || net.hop_back.is_empty() { net.hop_min[b - 1] } else { net.hop_back[a] }
                } else {
                    0.0
                };
                rn_ride.push(ride);
                rn_next.push(if i + 1 < len { (base + i + 1) as u32 } else if wrap { base as u32 } else { NONE });
                rn_wrap.push(wrap && i + 1 == len);
            }
        }
    }
    let n = s + rn_station.len();
    let mut edges: Vec<(u32, u32, f32, u32)> = vec![];
    for (r, &st) in rn_station.iter().enumerate() {
        let node = (s + r) as u32;
        let l = rn_line[r] as usize;
        if rn_next[r] != NONE {
            if !rn_wrap[r] {
                edges.push((st, node, board_cost(net.line_headway_min[l][p.period], st_wait[st as usize], change_f_of(&st_bld, st as usize), p), NONE));
            }
            edges.push((node, rn_next[r], rn_ride[r], node));
        }
        edges.push((node, st, 0.0, NONE));
    }
    // walking transfers: from each route node (having ridden) to the stations nearby, and to every
    // other station of its complex however far (T-161), from its platform to the other station's
    // position
    let cut2 = p.xfer_walk_m * p.xfer_walk_m;
    let d2 = |a: usize, b: usize| (net.st_x[b] - net.st_x[a]).powi(2) + (net.st_y[b] - net.st_y[a]).powi(2);
    let mut near: Vec<Vec<u32>> = vec![vec![]; s];
    if p.xfer_walk_m > 0.0 && s > 1 {
        let grid = StationGrid::new(net, p.xfer_walk_m, p.xfer_walk_m);
        for a in 0..s {
            grid.near(net.st_x[a], net.st_y[a], |b| {
                if b as usize != a && d2(a, b as usize) <= cut2 {
                    near[a].push(b);
                }
            });
        }
    }
    if net.st_group.len() == s {
        for members in complex_members(&net.st_group) {
            for &a in &members {
                for &b in &members {
                    // within the walk limit the edge is there already
                    if a != b && (p.xfer_walk_m <= 0.0 || d2(a as usize, b as usize) > cut2) {
                        near[a as usize].push(b);
                    }
                }
            }
        }
    }
    // a passage (T-1281) walks the straight line, on moving walkways
    let passage = |a: u32, b: u32| !net.st_passage.is_empty() && net.st_passage.binary_search(&[a.min(b), a.max(b)]).is_ok();
    for (r, &st) in rn_station.iter().enumerate() {
        let (x, y) = (net.st_x[st as usize] + rn_dx[r], net.st_y[st as usize] + rn_dy[r]);
        for &b in &near[st as usize] {
            let d = ((net.st_x[b as usize] - x).powi(2) + (net.st_y[b as usize] - y).powi(2)).sqrt();
            let walk = if passage(st, b) { d / upgrades::PASSAGE_MPS / 60.0 } else { p.xfer_min(d) };
            edges.push(((s + r) as u32, b, walk * p.w_xfer, NONE));
        }
    }
    edges.sort_by_key(|e| e.0);
    let mut adj_off = vec![0u32; n + 1];
    for e in &edges {
        adj_off[e.0 as usize + 1] += 1;
    }
    for i in 0..n {
        adj_off[i + 1] += adj_off[i];
    }
    let mut incoming: Vec<(u32, u32, u32)> = edges.iter().enumerate().map(|(e, x)| (x.1, x.0, e as u32)).collect();
    incoming.sort_by_key(|x| x.0);
    let mut rev_off = vec![0u32; n + 1];
    for x in &incoming { rev_off[x.0 as usize + 1] += 1; }
    for i in 0..n { rev_off[i + 1] += rev_off[i]; }
    let nr = rn_line.len();
    let phi = |r: usize| if net.span.len() == super::super::network::PERIODS * nr { net.span[p.period * nr + r] } else { 1.0 };
    // a share that rounds to 0 is no service at that stop (T-842)
    let rn_tph = rn_line.iter().enumerate().map(|(r, &l)| {
        let h = net.line_headway_min[l as usize][p.period];
        if h > 0.0 && h.is_finite() && phi(r) > 0.0 { 60.0 / h } else { 0.0 }
    }).collect();
    let rn_span: Vec<f32> = (0..nr).map(|r| { let f = phi(r); if f > 0.0 && f < 1.0 { -f.ln() / p.beta } else { 0.0 } }).collect();
    let bus_line: Vec<bool> = (0..net.n_lines()).map(|l| net.bus_line(l)).collect();
    let rn_bus: Vec<f32> = rn_line.iter().map(|&l| if bus_line[l as usize] { BUS_BOARD_MIN } else { 0.0 }).collect();
    Graph {
        rn_left: rn_span.iter().zip(&rn_bus).map(|(a, b)| a + b).collect(),
        rn_span,
        rn_bus,
        rn_dx,
        rn_dy,
        st_spread,
        st_wait,
        st_bld,
        rn_tph,
        rev_off,
        rev_from: incoming.iter().map(|x| x.1).collect(),
        rev_edge: incoming.iter().map(|x| x.2).collect(),
        n_st: s,
        n,
        rn_station,
        rn_line,
        rn_fare: vec![0.0; rn_ride.len()],
        rn_ride,
        adj_off,
        adj_to: edges.iter().map(|e| e.1).collect(),
        adj_cost: edges.iter().map(|e| e.2).collect(),
        adj_seg: edges.iter().map(|e| e.3).collect(),
        rn_next,
        fare_curve: FareCurve::default(),
        rn_km: vec![],
        rn_fare_mul: vec![],
        rn_kmw: vec![],
    }
}

/// The stations of each complex with more than one (`Network::st_group`), in station order.
pub fn complex_members(group: &[u32]) -> Vec<Vec<u32>> {
    let mut by: HashMap<u32, Vec<u32>> = HashMap::new();
    for (st, &g) in group.iter().enumerate() {
        by.entry(g).or_default().push(st as u32);
    }
    let mut out: Vec<Vec<u32>> = by.into_values().filter(|m| m.len() > 1).collect();
    out.sort_unstable_by_key(|m| m[0]);
    out
}

/// Ride edges cost real minutes times a crowding factor per segment (indexed by route node).
pub fn set_crowding(g: &mut Graph, factor: &[f32]) {
    for e in 0..g.adj_to.len() {
        let seg = g.adj_seg[e];
        if seg != NONE {
            let r = seg as usize - g.n_st;
            g.adj_cost[e] = g.rn_ride[r] * factor[r] + g.rn_fare[r];
        }
    }
}

/// Station-to-station strategies towards some of the stations. Each destination's search is
/// independent, so a caller that knows which stations anyone can alight at (`skims_for`) skips the
/// rest: a population whose commuters live near two stations of a 900-station network needs two
/// searches, not 900 (T-119). A destination that was not searched has no slot, `r` infinite in its
/// column, and no flow may be assigned to it.
#[derive(Clone)]
pub struct Skims {
    pub s: usize,
    pub n: usize,
    /// Generalised minutes station to station (expected wait + ride + transfers), S x S.
    pub r: Vec<f32>,
    /// Real minutes station to station, as `r` (waits, rides and the walks between stations and
    /// platforms; T-2265, `walk_column`), in metres walked at walking's pace (`Params::walk_eq_m`),
    /// saturating at `u16::MAX` where there is no route.
    pub walk: Vec<u16>,
    /// Per station, the most `walk` from it to any searched destination: what lets the walking
    /// rule pass trips without looking at their station pairs.
    pub walk_max: Vec<u16>,
    /// Each destination's slot in the arrays below, `NONE` if it was not searched.
    pub slot: Vec<u32>,
    /// Per searched destination (by slot), kept apart so adding one never moves the rest:
    /// the best next node towards it for route nodes (N), which boarding route nodes' departures
    /// are accepted (a bit per route node; an accepted one's departures an hour are its
    /// `Graph::rn_tph`, and boarding flow splits by them, so a rider takes the first useful
    /// train), and the search's settle order, nearest node first (as many as were reached).
    /// Node ids are u16 below 65,535 nodes (T-398: the tables are most of a worker's peak).
    pub next: Vec<NodeIds>,
    pub board: Vec<Box<[u64]>>,
    pub order: Vec<NodeIds>,
    /// Per searched destination, its change nodes (`ChangeNodes`): node ids from `n` on.
    pub change: Vec<ChangeNodes>,
    /// Per searched destination, the stations whose cost as an inter-city trip's start
    /// (`DestStrategy::entry`) is below `r`'s, (station, cost); `entry_r` puts them in.
    pub entry: Vec<Box<[(u32, f32)]>>,
}

/// A station as reached off one of the lines its own strategy boards (T-575): a rider never changes
/// onto the line they left towards the stop they came from or the one their train goes on to, so
/// arriving on that line they choose among the station's other lines and, where the line passes
/// the station again, its departures towards other stops (T-1268). Change node k is node `n + k`;
/// a route node of the line that leaves the train there, or walks there, has it as its next node. Each holds its station and a bit per
/// boarding edge of the station for the departures it accepts, `bits[off[k]..off[k + 1]]` (bit i
/// of word w is the station's edge 64 w + i).
#[derive(Clone, Default)]
pub struct ChangeNodes {
    pub st: Box<[u32]>,
    pub off: Box<[u32]>,
    pub bits: Box<[u64]>,
}

impl ChangeNodes {
    pub fn len(&self) -> usize {
        self.st.len()
    }
    /// Departures an hour change node `k` accepts at route node `r`.
    pub fn tph(&self, g: &Graph, k: usize, r: usize) -> f32 {
        let st = self.st[k] as usize;
        let e0 = g.adj_off[st] as usize;
        match (e0..g.adj_off[st + 1] as usize).position(|e| g.adj_to[e] as usize == g.n_st + r) {
            Some(i) if self.bits[self.off[k] as usize + i / 64] >> (i % 64) & 1 == 1 => g.rn_tph[r],
            _ => 0.0,
        }
    }
}

/// Node ids, as small as the graph allows (`NONE` is the type's largest value).
#[derive(Clone)]
pub enum NodeIds {
    Short(Box<[u16]>),
    Long(Box<[u32]>),
}

impl NodeIds {
    pub(super) fn of(v: &[u32], n: usize) -> NodeIds {
        if n < u16::MAX as usize {
            NodeIds::Short(v.iter().map(|&x| if x == NONE { u16::MAX } else { x as u16 }).collect())
        } else {
            NodeIds::Long(v.into())
        }
    }
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            NodeIds::Short(v) => v.iter().map(|&x| if x == u16::MAX { NONE } else { x as u32 }).collect(),
            NodeIds::Long(v) => v.to_vec(),
        }
    }
}

/// A node id as an index (`NodeIds`); an absent one is at least the node count.
pub(super) trait Id: Copy {
    fn ix(self) -> usize;
}
impl Id for u16 {
    #[inline(always)]
    fn ix(self) -> usize {
        self as usize
    }
}
impl Id for u32 {
    #[inline(always)]
    fn ix(self) -> usize {
        self as usize
    }
}

/// Departures an hour accepted at boarding route node `r` towards a destination (`Skims::board`).
#[inline(always)]
pub(super) fn accepted(bits: &[u64], g: &Graph, r: usize) -> f32 {
    if bits[r >> 6] >> (r & 63) & 1 == 1 { g.rn_tph[r] } else { 0.0 }
}

impl Skims {
    pub fn empty(g: &Graph) -> Skims {
        Skims { s: g.n_st, n: g.n, r: vec![f32::INFINITY; g.n_st * g.n_st], walk: vec![u16::MAX; g.n_st * g.n_st], walk_max: vec![0; g.n_st], slot: vec![NONE; g.n_st], next: vec![], board: vec![], order: vec![], change: vec![], entry: vec![] }
    }
    /// `r` with the first boarding at the planned wait (T-1430), as the inter-city trips cost it.
    pub fn entry_r(&self) -> std::borrow::Cow<'_, [f32]> {
        if self.entry.iter().all(|e| e.is_empty()) {
            return std::borrow::Cow::Borrowed(&self.r);
        }
        let mut r = self.r.clone();
        for (dst, &slot) in self.slot.iter().enumerate() {
            if slot != NONE {
                for &(src, c) in self.entry[slot as usize].iter() {
                    r[src as usize * self.s + dst] = c;
                }
            }
        }
        std::borrow::Cow::Owned(r)
    }
    /// The most change nodes of any searched destination.
    pub fn most_change_nodes(&self) -> usize {
        self.change.iter().map(|c| c.len()).max().unwrap_or(0)
    }
    /// The accepted departures an hour per boarding route node towards the destination in `slot`.
    pub fn board_tph(&self, g: &Graph, slot: usize) -> Vec<f32> {
        (0..g.n - g.n_st).map(|r| accepted(&self.board[slot], g, r)).collect()
    }
}

/// Destination-based optimal strategies for every station. At a station, useful lines share their
/// frequency and riders take the first arrival. A slower line joins only when its added frequency
/// lowers the expected trip cost; a line going down the wrong branch contributes no useful
/// frequency.
pub fn skims(g: &Graph, p: &Params) -> Skims {
    skims_for(g, p, &vec![true; g.n_st])
}

/// `skims` towards the stations marked in `want` only.
pub fn skims_for(g: &Graph, p: &Params, want: &[bool]) -> Skims {
    let mut sk = Skims::empty(g);
    skims_extend(&mut sk, g, p, want);
    sk
}

/// Search the destinations marked in `want` that `sk` does not have yet.
pub fn skims_extend(sk: &mut Skims, g: &Graph, p: &Params, want: &[bool]) {
    let todo: Vec<usize> = (0..g.n_st).filter(|&d| want.get(d).copied().unwrap_or(false) && sk.slot[d] == NONE).collect();
    if todo.is_empty() {
        return;
    }
    let mut q = DestSearch::new(g, p);
    let mut wm = vec![];
    let mut dep = vec![];
    for &dst in &todo {
        sk.slot[dst] = sk.next.len() as u32;
        let d = q.run(dst, Some(&mut sk.r));
        walk_column(g, p, d.view(), dst, &mut wm, &mut dep);
        let s = g.n_st;
        for (a, &m) in wm[..s].iter().enumerate() {
            if m.is_finite() {
                let w = p.walk_eq_m(m).round().min(u16::MAX as f32 - 1.0) as u16;
                sk.walk[a * s + dst] = w;
                sk.walk_max[a] = sk.walk_max[a].max(w);
            } else {
                sk.walk[a * s + dst] = u16::MAX;
            }
        }
        sk.next.push(d.next);
        sk.board.push(d.board);
        sk.order.push(d.order);
        sk.change.push(d.change);
        sk.entry.push(d.entry);
    }
}

/// The real minutes from each node to `dst` under strategy `st` (T-2265), into `wm` (node ids,
/// change nodes from `g.n`; infinite where the strategy does not reach): the wait for the first
/// useful train (half the accepted departures' headway), the rides, a walk to another station
/// from the platform left, and at a change the walk from where the rider stands to the platform
/// of the first train, all expected over the departures by frequency (T-717). No crowding, fare,
/// left-behind wait or fixed change minutes. A boarding rides a stop first (T-1055). Read in
/// settle order, so every next node is done first.
pub fn walk_column(g: &Graph, p: &Params, st: Strategy, dst: usize, wm: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => walk_column_in(g, p, st, dst, next, order, wm, dep),
        (NodeIds::Long(next), NodeIds::Long(order)) => walk_column_in(g, p, st, dst, next, order, wm, dep),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn walk_column_in<I: Id>(g: &Graph, p: &Params, st: Strategy, dst: usize, next: &[I], order: &[I], wm: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    let (s, n) = (g.n_st, g.n);
    wm.clear();
    wm.resize(n + st.change.len(), f32::INFINITY);
    // a walk edge's perceived minutes back to real ones (`build_graph`: `xfer_min(d) * w_xfer`)
    let to_min = 1.0 / p.w_xfer;
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    for &id in order {
        let id = id.ix();
        if id == dst {
            wm[id] = 0.0;
        } else if id < s || id >= n {
            departures(g, st.board, st.change, id, dep);
            let (mut f, mut w) = (0f32, 0f32);
            for &(e, a) in dep.iter() {
                let r = g.adj_to[e as usize] as usize - s;
                let k = if on_train(s + r) { wm[s + r] } else { g.rn_ride[r] + wm[g.rn_next[r] as usize] };
                if k.is_finite() {
                    f += a;
                    w += a * k;
                }
            }
            if f > 0.0 {
                wm[id] = 30.0 / f + w / f;
            }
        } else {
            let nx = next[id].ix();
            if nx >= n + st.change.len() {
                continue;
            }
            if on_train(id) {
                wm[id] = g.rn_ride[id - s] + wm[nx];
                continue;
            }
            let r = id - s;
            let here = g.rn_station[r] as usize;
            let to = if nx < s { nx } else { st.change.st[nx - n] as usize };
            let walk = if to != here {
                (g.adj_off[id] as usize..g.adj_off[id + 1] as usize).find(|&e| g.adj_to[e] as usize == to).map_or(0.0, |e| g.adj_cost[e] * to_min)
            } else {
                0.0
            };
            let (ax, ay) = if to == here { (g.rn_dx[r], g.rn_dy[r]) } else { (0.0, 0.0) };
            let mut plat = 0.0;
            if nx != dst && g.st_spread[to] {
                departures(g, st.board, st.change, nx, dep);
                let (mut f, mut w) = (0f32, 0f32);
                for &(e, a) in dep.iter() {
                    let r2 = g.adj_to[e as usize] as usize - s;
                    f += a;
                    w += a * (g.rn_dx[r2] - ax).hypot(g.rn_dy[r2] - ay);
                }
                if f > 0.0 {
                    plat = p.xfer_min(w / f);
                }
            }
            wm[id] = walk + plat + wm[nx];
        }
    }
}

/// Each node's expected fare km to `dst` under strategy `st` (T-883), into `km` (node ids, change
/// nodes from `g.n`): a station or change node averages the departures it accepts by frequency, as
/// the riders split, and a boarding rides a stop first (T-1055). Read in settle order, so every
/// next node is done first. Needs `g.rn_km` (a curve that is not a straight line).
pub fn trip_km(g: &Graph, st: Strategy, dst: usize, km: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    trip_km_on(g, st, dst, &g.rn_km, km, dep)
}

/// `trip_km` with each route node's fare km to the next stop given (`Graph::rn_km`'s layout).
pub fn trip_km_on(g: &Graph, st: Strategy, dst: usize, hop_km: &[f32], km: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => trip_km_in(g, st, dst, next, order, hop_km, km, dep),
        (NodeIds::Long(next), NodeIds::Long(order)) => trip_km_in(g, st, dst, next, order, hop_km, km, dep),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trip_km_in<I: Id>(g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], hop_km: &[f32], km: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    let (s, n) = (g.n_st, g.n);
    let end = n + st.change.len();
    km.clear();
    km.resize(end, 0.0);
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    let hop = |r: usize| hop_km.get(r).copied().unwrap_or(0.0);
    for &id in order {
        let id = id.ix();
        if id == dst {
            continue;
        }
        if id < s || id >= n {
            departures(g, st.board, st.change, id, dep);
            let (mut f, mut w) = (0f32, 0f32);
            for &(e, a) in dep.iter() {
                let r = g.adj_to[e as usize] as usize - s;
                let k = if on_train(s + r) { km[s + r] } else { hop(r) + km[g.rn_next[r] as usize] };
                f += a;
                w += a * k;
            }
            if f > 0.0 {
                km[id] = w / f;
            }
        } else {
            let nx = next[id].ix();
            if nx >= end {
                continue;
            }
            km[id] = if on_train(id) { hop(id - s) + km[nx] } else { km[nx] };
        }
    }
}

/// T-1626: a trip's km-weighted fare multiplier from its fare km and those km times their
/// multipliers.
#[inline]
pub fn fare_mul_of(km: f32, kmw: f32) -> f32 {
    if km > 0.0 { kmw / km } else { 1.0 }
}

/// The fare curve's extra on the trips towards `dst` (`col(src)` riders from each station), each
/// trip's at its station pair's expected km (`trip_km`, already in `km`): the curve's part of the
/// fare that the base and rate leave out, perceived minutes. It is added into `out` per route node
/// by the km ridden from it, so each line a trip rides takes its share by km and none goes below
/// its base and rate (T-1516): each trip's extra a km goes along its strategy as `assign` splits
/// the riders and as `trip_km` counts the km. `q`: scratch.
pub fn board_extra(g: &Graph, st: Strategy, dst: usize, col: impl Fn(usize) -> f32, km: &[f32], dep: &mut Vec<(u32, f32)>, q: &mut Vec<f32>, out: &mut [f32]) {
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => board_extra_in(g, st, dst, next, order, col, km, dep, q, out),
        (NodeIds::Long(next), NodeIds::Long(order)) => board_extra_in(g, st, dst, next, order, col, km, dep, q, out),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn board_extra_in<I: Id>(g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], col: impl Fn(usize) -> f32, km: &[f32], dep: &mut Vec<(u32, f32)>, q: &mut Vec<f32>, out: &mut [f32]) {
    let (s, n) = (g.n_st, g.n);
    let end = n + st.change.len();
    q.clear();
    q.resize(end, 0.0);
    let mut any = false;
    for src in 0..s {
        let f = col(src);
        if f == 0.0 || src == dst || !(km[src] > 0.0) {
            continue;
        }
        let x = f * g.fare_curve.extra(km[src]);
        if x != 0.0 {
            q[src] = x / km[src];
            any = true;
        }
    }
    if !any {
        return;
    }
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    // T-1626: a trip's extra a km lands on each km times its multiplier, so the trip pays it times
    // its km-weighted multiplier, as `DestSearch::run` priced it
    let hops = if g.rn_kmw.is_empty() { &g.rn_km } else { &g.rn_kmw };
    let hop = |r: usize| hops.get(r).copied().unwrap_or(0.0);
    // the reverse of the settle order: every node is reached after all that feed it
    for &id in order.iter().rev() {
        let id = id.ix();
        let v = q[id];
        if v == 0.0 || id == dst {
            continue;
        }
        if id < s || id >= n {
            departures(g, st.board, st.change, id, dep);
            let sum: f32 = dep.iter().map(|d| d.1).sum();
            if !(sum > 0.0) {
                continue;
            }
            for &(e, a) in dep.iter() {
                let r = g.adj_to[e as usize] as usize - s;
                let w = v * a / sum;
                if on_train(s + r) {
                    q[s + r] += w;
                } else {
                    // a boarding rides a stop first (T-1055)
                    out[r] += w * hop(r);
                    let nx = g.rn_next[r] as usize;
                    if nx < end {
                        q[nx] += w;
                    }
                }
            }
        } else {
            let nx = next[id].ix();
            if nx >= end {
                continue;
            }
            if on_train(id) {
                out[id - s] += v * hop(id - s);
            }
            q[nx] += v;
        }
    }
}


/// One destination's strategy, as a `Skims` slot holds it.
pub struct DestStrategy {
    pub next: NodeIds,
    pub board: Box<[u64]>,
    pub order: NodeIds,
    pub change: ChangeNodes,
    /// The stations whose route cost as an inter-city trip's start, the first boarding at the
    /// planned wait (`planned_board_cost`, T-1430), is below the plain one, (station, cost in
    /// `Skims::r`'s terms); empty unless the search wrote route costs.
    pub entry: Box<[(u32, f32)]>,
}

/// A destination's strategy, borrowed from a `Skims` slot or a `DestStrategy`.
#[derive(Clone, Copy)]
pub struct Strategy<'a> {
    pub next: &'a NodeIds,
    pub board: &'a [u64],
    pub order: &'a NodeIds,
    pub change: &'a ChangeNodes,
}

impl NodeIds {
    pub(super) fn bytes(&self) -> usize {
        match self {
            NodeIds::Short(v) => 2 * v.len(),
            NodeIds::Long(v) => 4 * v.len(),
        }
    }
}

impl DestStrategy {
    pub fn bytes(&self) -> usize {
        self.next.bytes() + self.order.bytes() + 8 * self.board.len() + 4 * (self.change.st.len() + self.change.off.len()) + 8 * self.change.bits.len() + 8 * self.entry.len()
    }
    pub fn view(&self) -> Strategy<'_> {
        Strategy { next: &self.next, board: &self.board, order: &self.order, change: &self.change }
    }
}

impl Skims {
    pub fn strategy(&self, slot: usize) -> Strategy<'_> {
        Strategy { next: &self.next[slot], board: &self.board[slot], order: &self.order[slot], change: &self.change[slot] }
    }
}

/// About the bytes one destination's strategy holds on `g` (`DestStrategy::bytes`, change nodes
/// left out): its next nodes and settle order, a bit per route node.
pub fn strategy_bytes(g: &Graph) -> usize {
    let id = if g.n < u16::MAX as usize { 2 } else { 4 };
    2 * id * g.n + (g.n - g.n_st).div_ceil(64) * 8
}

/// The strategy search one destination at a time, for a caller that does not hold every
/// destination's strategy at once (T-761).
pub struct DestSearch<'a> {
    pub(super) q: Search<'a>,
    pub(super) km: Vec<f32>,
    pub(super) kmw: Vec<f32>,
    pub(super) dep: Vec<(u32, f32)>,
    /// T-2652: with `TW_T2652_CHECK` set, every search is made again with no leaf stations and
    /// the two compared (`check_plain`).
    #[cfg(test)]
    pub(super) plain: Option<Box<Search<'a>>>,
}

impl<'a> DestSearch<'a> {
    pub fn new(g: &'a Graph, p: &'a Params) -> DestSearch<'a> {
        DestSearch {
            q: Search::new(g, p),
            km: vec![],
            kmw: vec![],
            dep: vec![],
            #[cfg(test)]
            plain: None,
        }
    }
    /// T-2652: queue pops so far, leaf stations and the route nodes at them.
    #[cfg(test)]
    pub fn counts(&self) -> (u64, usize, usize) {
        (self.q.pops, self.q.leaves.len(), self.q.lf_rn.len())
    }
    /// T-2652: the search just made towards `dst` against the same search with no leaf stations:
    /// costs, next nodes, accepted departures, offers and change nodes to the bit, the settle order
    /// the same nodes with every node after its next.
    #[cfg(test)]
    fn check_plain(&mut self, dst: usize) {
        if self.q.leaves.is_empty() || !(CHECK_PLAIN.with(|c| c.get()) || std::env::var_os("TW_T2652_CHECK").is_some()) {
            return;
        }
        let (g, p) = (self.q.g, self.q.p);
        let leaves = self.q.leaves.len();
        let b = self.plain.get_or_insert_with(|| {
            eprintln!("T-2652 check: {} stations, {leaves} leaves", g.n_st);
            let was = PLAIN_SEARCH.with(|c| c.replace(true));
            let b = Box::new(Search::new(g, p));
            PLAIN_SEARCH.with(|c| c.set(was));
            b
        });
        b.run(dst);
        let a = &self.q;
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        let sorted = |x: &[u32]| {
            let mut v = x.to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(bits(&a.dist), bits(&b.dist), "T-2652: costs towards {dst}");
        assert_eq!(a.next, b.next, "T-2652: next nodes towards {dst}");
        assert_eq!(a.board, b.board, "T-2652: accepted departures towards {dst}");
        assert_eq!(bits(&a.offer), bits(&b.offer), "T-2652: offers towards {dst}");
        assert_eq!(sorted(&a.order), sorted(&b.order), "T-2652: settled nodes towards {dst}");
        let cn = |q: &Search| q.cn.iter().map(|c| (c.st, c.line, c.ban, c.frequency.to_bits(), c.weighted.to_bits(), c.dist.to_bits(), c.done, c.feed, c.bits)).collect::<Vec<_>>();
        assert_eq!(cn(a), cn(b), "T-2652: change nodes towards {dst}");
        assert_eq!((&a.cn_bits, &a.cn_feed), (&b.cn_bits, &b.cn_feed), "T-2652: change nodes' departures towards {dst}");
        let mut at = vec![usize::MAX; a.n + a.cn.len()];
        for (k, &v) in a.order.iter().enumerate() {
            at[v as usize] = k;
        }
        for &v in &a.order[1..] {
            let v = v as usize;
            if v < a.n {
                assert!(at[a.next[v] as usize] < at[v], "T-2652: node {v} settled before its next towards {dst}");
            }
        }
    }
    /// Search towards `dst`: its strategy, and its column of route costs written into `r` (S x S,
    /// `Skims::r`'s layout) when given.
    pub fn run(&mut self, dst: usize, r: Option<&mut [f32]>) -> DestStrategy {
        match r {
            None => self.run_by(dst, None::<fn(usize, f32)>),
            Some(r) => {
                let s = self.q.s;
                self.run_by(dst, Some(|src: usize, v: f32| r[src * s + dst] = v))
            }
        }
    }
    /// `run` with the route costs written into `col`, by the station the trip starts from (T-1545).
    pub fn run_col(&mut self, dst: usize, col: &mut [f32]) -> DestStrategy {
        self.run_by(dst, Some(|src: usize, v: f32| col[src] = v))
    }
    pub(super) fn run_by(&mut self, dst: usize, put: Option<impl FnMut(usize, f32)>) -> DestStrategy {
        self.q.run(dst);
        #[cfg(test)]
        self.check_plain(dst);
        let q = &mut self.q;
        let (s, n) = (q.s, q.n);
        let ids = n + q.cn.len();
        let mut d = DestStrategy { next: NodeIds::of(&q.next, ids), board: std::mem::take(&mut q.board), order: NodeIds::of(&q.order, ids), change: q.change_nodes(), entry: Box::new([]) };
        if let Some(mut put) = put {
            let curve = &q.g.fare_curve;
            let bent = !curve.is_line();
            let weighed = bent && !q.g.rn_kmw.is_empty();
            if bent {
                trip_km(q.g, d.view(), dst, &mut self.km, &mut self.dep);
            }
            if weighed {
                trip_km_on(q.g, d.view(), dst, &q.g.rn_kmw, &mut self.kmw, &mut self.dep);
            }
            q.entry_costs(dst);
            let mut entry = vec![];
            for src in 0..s {
                if src == dst {
                    put(src, 0.0);
                    continue;
                }
                // The first boarding is not a transfer; the trip's base fare is added later.
                let x = if bent { curve.extra(self.km[src]) * if weighed { fare_mul_of(self.km[src], self.kmw[src]) } else { 1.0 } } else { 0.0 };
                let change = q.p.change_at(q.g.change_f(src));
                put(src, q.dist[src] - change + x);
                if q.entry[src] < q.dist[src] - 1e-4 {
                    entry.push((src as u32, q.entry[src] - change + x));
                }
            }
            d.entry = entry.into_boxed_slice();
        }
        d
    }
}

/// The strategy search's queue: a 4-ary min-heap of (cost bits, id), popped by cost, then id
/// (costs are never negative, so their bits sort as they do). A lower cost for an id already
/// queued adds a copy; the stale copy pops after it and is skipped, the node or change node being
/// settled by then (a boarding's turn is queued once). Cheaper than keeping each id's place to
/// move it up (T-1647).
pub(super) struct IdHeap {
    pub(super) h: Vec<u64>,
}

impl IdHeap {
    pub(super) fn new(ids: usize) -> IdHeap {
        IdHeap { h: Vec::with_capacity(ids) }
    }
    #[inline(always)]
    pub(super) fn push(&mut self, cost: f32, id: u32) {
        let x = (cost.to_bits() as u64) << 32 | id as u64;
        let h = &mut self.h;
        let mut i = h.len();
        h.push(x);
        while i > 0 {
            let up = (i - 1) / 4;
            let y = h[up];
            if y <= x { break; }
            h[i] = y;
            i = up;
        }
        h[i] = x;
    }
    #[inline(always)]
    pub(super) fn pop(&mut self) -> Option<usize> {
        let h = &mut self.h;
        let top = *h.first()?;
        let x = h.pop().unwrap();
        let len = h.len();
        if len > 0 {
            // the hole goes down to a leaf by the least child, then the last entry rises from it
            let mut i = 0;
            loop {
                let c = 4 * i + 1;
                if c + 3 < len {
                    let (a, b, e, f) = (h[c], h[c + 1], h[c + 2], h[c + 3]);
                    let (i1, v1) = if b < a { (c + 1, b) } else { (c, a) };
                    let (i2, v2) = if f < e { (c + 3, f) } else { (c + 2, e) };
                    let (m, y) = if v2 < v1 { (i2, v2) } else { (i1, v1) };
                    h[i] = y;
                    i = m;
                } else {
                    if c < len {
                        let (mut m, mut y) = (c, h[c]);
                        for k in c + 1..len {
                            if h[k] < y {
                                m = k;
                                y = h[k];
                            }
                        }
                        h[i] = y;
                        i = m;
                    }
                    break;
                }
            }
            while i > 0 {
                let up = (i - 1) / 4;
                let y = h[up];
                if y <= x { break; }
                h[i] = y;
                i = up;
            }
            h[i] = x;
        }
        Some(top as u32 as usize)
    }
}

/// A change node being searched (`ChangeNodes`): its station, the line it was reached off and which
/// of that line's departures it bars (`barred`), the pooled frequency and frequency-weighted cost
/// of the lines it accepts, its cost, settled, its arrivals (a range of `Search::cn_feed`: the
/// station's incoming edges off that line) and where its accepted departures' bits start in
/// `Search::cn_bits`.
pub(super) struct Change {
    pub(super) st: u32,
    pub(super) line: u32,
    pub(super) ban: [u32; 2],
    pub(super) frequency: f32,
    pub(super) weighted: f32,
    pub(super) dist: f32,
    pub(super) done: bool,
    pub(super) feed: (u32, u32),
    pub(super) bits: u32,
}

/// One destination's strategy search at a time (`skims_extend`). Heap ids: nodes below `n`; from
/// `n`, a boarding onto route node `id - n` that waited its turn (T-552); from `n + nr`, change
/// node `id - n - nr`.
pub(super) struct Search<'a> {
    pub(super) g: &'a Graph,
    pub(super) p: &'a Params,
    pub(super) s: usize,
    pub(super) n: usize,
    pub(super) nr: usize,
    pub(super) dist: Vec<f32>,
    pub(super) done: Vec<bool>,
    pub(super) frequency: Vec<f32>,
    pub(super) weighted: Vec<f32>,
    pub(super) heap: IdHeap,
    pub(super) next: Vec<u32>,
    pub(super) order: Vec<u32>,
    pub(super) board: Box<[u64]>,
    /// What each boarding route node offered its station, infinite if it has not.
    pub(super) offer: Vec<f32>,
    /// Each route node's cost by its ride edge, infinite until its next stop is settled: what a
    /// boarding there costs, since a rider who boards rides at least one stop (T-1055), however
    /// cheap leaving the train there is for a rider already on it.
    pub(super) ride: Vec<f32>,
    pub(super) cn: Vec<Change>,
    /// Each station's change nodes, a range of `cn`.
    pub(super) cn_of: Vec<(u32, u32)>,
    /// The change nodes' accepted departures, a bit per boarding edge of their station.
    pub(super) cn_bits: Vec<u64>,
    pub(super) cn_feed: Vec<u32>,
    pub(super) lines: Vec<u32>,
    /// A settling station's change nodes to be, (line, `ban`), and (index in `keys`, incoming
    /// edge) for its arrivals that need one.
    pub(super) keys: Vec<(u32, [u32; 2])>,
    pub(super) feed: Vec<(u32, u32)>,
    /// Each route node's next and previous stop, past a circle's stand at its own station.
    pub(super) nst: Vec<u32>,
    pub(super) pst: Vec<u32>,
    pub(super) cand: Vec<(f32, u32)>,
    /// Per station, the one line with departures from it, `MANY` for several, `NONE` for none.
    pub(super) lone: Vec<u32>,
    /// Each route node's place among its station's boarding edges.
    pub(super) edge_ix: Vec<u32>,
    /// The platforms (relative to the station) and frequencies a settling node boards (T-717).
    pub(super) pool: Vec<(f32, f32, f32)>,
    /// Each station's cost as an inter-city trip's start (`entry_costs`).
    pub(super) entry: Vec<f32>,
    /// Each incoming edge's cost (`Graph::rev_edge`'s), read in edge order.
    pub(super) rev_cost: Vec<f32>,
    /// Per route node, what settling it reads, in one place (T-1647).
    pub(super) rn: Vec<RnStep>,
    /// Per route node, a range of `banned`: the departures with trains of its line from its station
    /// that `ban_of` bars an arrival off it from; and whether they are all the line's there.
    pub(super) banned_off: Vec<u32>,
    pub(super) banned: Vec<u32>,
    pub(super) banned_all: Vec<bool>,
    /// T-3295: per route node, its stop's place in its line's list, its way along it (1 out, -1
    /// back) and, for a circle, its number of stops (0 otherwise), for `walk_ban`.
    pub(super) at: Vec<(i32, i32, i32)>,
    /// T-2652: the leaf stations (`leaves`; per station its place there, `NONE` if none): one
    /// line's departures, no walk in or out, every arrival barred that line whole. Only their own
    /// route nodes' offers reach them and nothing is reached from them, so their strategies are
    /// joined after the queue empties (`leaf_join`). Their route nodes (`lf_rn`, a range per leaf
    /// from `lf_off`; `chain`) are reached only by riding on, so they are settled along the line
    /// (`chain_step`). The destination is no leaf in its own search.
    pub(super) leaf_ix: Vec<u32>,
    pub(super) leaves: Vec<u32>,
    pub(super) lf_off: Vec<u32>,
    pub(super) lf_rn: Vec<u32>,
    pub(super) chain: Vec<bool>,
    pub(super) lf_cand: Vec<(u64, u32)>,
    /// Queue pops so far, for the benchmarks (T-2652).
    #[cfg(test)]
    pub(super) pops: u64,
}

#[cfg(test)]
thread_local! {
    /// T-2652: this thread's searches are checked against the plain search (`check_plain`).
    pub(super) static CHECK_PLAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// T-2652: searches made on this thread from now on find no leaf stations, so they run every
    /// node through the queue as before (for A/B runs; `set_plain_search`).
    static PLAIN_SEARCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// T-2652: whether the strategy searches made on this thread from now on leave the leaf stations
/// in the queue (the same strategies, slower); off by default.
pub fn set_plain_search(on: bool) {
    PLAIN_SEARCH.with(|c| c.set(on));
}

/// A route node's edges in as `Search::run` reads them: whether riders board it from its station
/// (a boarding edge and trains), its `Graph::rn_left`, and the route node riding into it, `NONE`
/// for none, with that ride's cost, its `rn_left` and whether riders board there.
#[derive(Clone, Copy)]
pub(super) struct RnStep {
    pub(super) prev: u32,
    pub(super) prev_cost: f32,
    pub(super) left: f32,
    pub(super) prev_left: f32,
    pub(super) boards: bool,
    pub(super) prev_boards: bool,
}

pub(super) const MANY: u32 = NONE - 1;
/// A change node's `ban` barring every departure of its line.
pub(super) const WHOLE: [u32; 2] = [MANY, MANY];

/// The stops a rider alighting off route node `a` may not ride on towards on its line: the one
/// they came from and the one their train goes on to (`Search::ban`).
#[inline(always)]
pub(super) fn ban_of(pst: &[u32], nst: &[u32], a: usize) -> [u32; 2] {
    let (x, y) = (pst[a], nst[a]);
    [x.min(y), x.max(y)]
}

/// Whether a change node off `line` with `ban` bars route node `r`: every departure of the line
/// for `WHOLE`, else those whose next stop is in `ban` (T-1268).
#[inline(always)]
pub(super) fn barred(g: &Graph, nst: &[u32], r: usize, line: u32, ban: [u32; 2]) -> bool {
    g.rn_line[r] == line && (ban == WHOLE || ban.contains(&nst[r]))
}

impl Search<'_> {
    pub(super) fn new<'a>(g: &'a Graph, p: &'a Params) -> Search<'a> {
        let (s, n) = (g.n_st, g.n);
        let nr = n - s;
        let fwd: Vec<u32> = g.rn_next.iter().map(|&x| if x == NONE { NONE } else { x - s as u32 }).collect();
        let mut back = vec![NONE; nr];
        for (r, &x) in fwd.iter().enumerate() {
            if x != NONE { back[x as usize] = r as u32; }
        }
        let stop = |step: &[u32], r: usize| {
            let mut x = step[r];
            for _ in 0..2 {
                if x == NONE || g.rn_station[x as usize] != g.rn_station[r] { break; }
                x = step[x as usize];
            }
            if x == NONE || g.rn_station[x as usize] == g.rn_station[r] { NONE } else { g.rn_station[x as usize] }
        };
        let nst: Vec<u32> = (0..nr).map(|r| stop(&fwd, r)).collect();
        let pst: Vec<u32> = (0..nr).map(|r| stop(&back, r)).collect();
        let mut edge_ix = vec![NONE; nr];
        let mut lone = vec![NONE; s];
        for v in 0..s {
            let e0 = g.adj_off[v] as usize;
            for e in e0..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                edge_ix[r] = (e - e0) as u32;
                if g.rn_tph[r] > 0.0 {
                    lone[v] = match lone[v] {
                        NONE => g.rn_line[r],
                        l if l == g.rn_line[r] => l,
                        _ => MANY,
                    };
                }
            }
        }
        // per route node, its line's departures with trains from its station that an arrival off
        // it may not take, and whether those are all of them
        let (mut banned_off, mut banned, mut banned_all) = (vec![0u32; nr + 1], vec![], vec![true; nr]);
        for a in 0..nr {
            let (v, line, ban) = (g.rn_station[a] as usize, g.rn_line[a], ban_of(&pst, &nst, a));
            for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                if g.rn_line[r] != line || g.rn_tph[r] == 0.0 { continue; }
                if ban.contains(&nst[r]) {
                    banned.push(r as u32);
                } else {
                    banned_all[a] = false;
                }
            }
            banned_off[a + 1] = banned.len() as u32;
        }
        // a line's route nodes are its stops out, then back in reverse (`build_graph`)
        let mut at = vec![(0, 1, 0); nr];
        let mut r0 = 0;
        while r0 < nr {
            let r1 = r0 + g.rn_line[r0..].iter().take_while(|&&l| l == g.rn_line[r0]).count();
            let len = (r1 - r0) / 2;
            let circle = if g.rn_next[r0 + len - 1] != NONE { len as i32 } else { 0 };
            for k in 0..len {
                at[r0 + k] = (k as i32, 1, circle);
                at[r0 + len + k] = ((len - 1 - k) as i32, -1, circle);
            }
            r0 = r1;
        }
        let rev_cost: Vec<f32> = g.rev_edge.iter().map(|&e| g.adj_cost[e as usize]).collect();
        let boards = |r: usize| edge_ix[r] != NONE && g.rn_tph[r] > 0.0;
        let rn = (0..nr)
            .map(|r| {
                let v = s + r;
                let mut x = RnStep { prev: NONE, prev_cost: 0.0, left: g.rn_left[r], prev_left: 0.0, boards: boards(r), prev_boards: false };
                for j in g.rev_off[v] as usize..g.rev_off[v + 1] as usize {
                    let from = g.rev_from[j] as usize;
                    if from >= s {
                        debug_assert!(x.prev == NONE, "a route node ridden into from two");
                        let p = from - s;
                        (x.prev, x.prev_cost, x.prev_left, x.prev_boards) = (p as u32, rev_cost[j], g.rn_left[p], boards(p));
                    }
                }
                x
            })
            .collect();
        // T-2652: a station is a leaf when every arrival is off one of its own route nodes, barred
        // its line's every departure here (its one line), and none of them walks anywhere
        let (mut leaf_ix, mut leaves, mut lf_off, mut lf_rn) = (vec![NONE; s], vec![], vec![0u32], vec![]);
        if !PLAIN_SEARCH.with(|c| c.get()) {
            for u in 0..s {
                let (j0, j1) = (g.rev_off[u] as usize, g.rev_off[u + 1] as usize);
                if lone[u] == NONE || lone[u] == MANY || j0 == j1 {
                    continue;
                }
                let leaf = g.rev_from[j0..j1].iter().all(|&from| {
                    let from = from as usize;
                    from >= s && g.rn_station[from - s] as usize == u && {
                        let a = from - s;
                        banned_all[a] && banned_off[a + 1] > banned_off[a] && g.adj_to[g.adj_off[from] as usize..g.adj_off[from + 1] as usize].iter().all(|&to| to as usize == u || to == g.rn_next[a])
                    }
                });
                if leaf {
                    // every route node has its edge to its station (`build_graph`)
                    leaf_ix[u] = leaves.len() as u32;
                    leaves.push(u as u32);
                    lf_rn.extend(g.rev_from[j0..j1].iter().map(|&from| from - s as u32));
                    lf_off.push(lf_rn.len() as u32);
                }
            }
        }
        let chain = (0..nr).map(|r| leaf_ix[g.rn_station[r] as usize] != NONE).collect();
        Search {
            leaf_ix,
            leaves,
            lf_off,
            lf_rn,
            chain,
            lf_cand: vec![],
            #[cfg(test)]
            pops: 0,
            g,
            p,
            s,
            n,
            nr,
            dist: vec![f32::INFINITY; n],
            done: vec![false; n],
            frequency: vec![0f32; s],
            weighted: vec![0f32; s],
            // a change node per station and line its strategy boards: fewer than route nodes
            heap: IdHeap::new(n + 2 * nr),
            next: vec![NONE; n],
            order: Vec::with_capacity(n),
            board: Box::new([]),
            offer: vec![f32::INFINITY; nr],
            ride: vec![f32::INFINITY; nr],
            cn: vec![],
            cn_of: vec![(0, 0); s],
            cn_bits: vec![],
            cn_feed: vec![],
            lines: vec![],
            keys: vec![],
            feed: vec![],
            nst,
            pst,
            cand: vec![],
            lone,
            edge_ix,
            pool: vec![],
            entry: vec![],
            rn,
            rev_cost,
            banned_off,
            banned,
            banned_all,
            at,
        }
    }

    /// T-1430: after `run`, each station's cost towards `dst` as an inter-city trip's start: the
    /// departures offered to it, less their tolerance penalty (`Graph::rn_span`), pooled as
    /// `join_line` pools them but at the planned wait (`planned_board_cost`). Every departure was
    /// offered by the search's end; the pool takes them cheapest first and stops at the first that
    /// would not lower it, as `join_line` would refuse every later one too.
    pub(super) fn entry_costs(&mut self, dst: usize) {
        let (g, s) = (self.g, self.s);
        self.entry.clear();
        self.entry.resize(s, f32::INFINITY);
        for v in 0..s {
            if v == dst || !self.dist[v].is_finite() {
                continue;
            }
            self.cand.clear();
            for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                if g.rn_tph[r] > 0.0 && self.offer[r].is_finite() {
                    self.cand.push((self.offer[r] - g.rn_span[r], r as u32));
                }
            }
            self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let (mut f, mut w, mut cost) = (0f32, 0f32, f32::INFINITY);
            for &(d, r) in &self.cand {
                let sum = f + g.rn_tph[r as usize];
                let c = (w + g.rn_tph[r as usize] * d) / sum + planned_board_cost(60.0 / sum, g.st_wait[v], g.change_f(v), self.p);
                if c >= cost {
                    break;
                }
                (f, w, cost) = (sum, w + g.rn_tph[r as usize] * d, c);
            }
            self.entry[v] = cost;
        }
    }

    pub(super) fn run(&mut self, dst: usize) {
        let (g, s, n, nr) = (self.g, self.s, self.n, self.nr);
        self.dist.fill(f32::INFINITY);
        self.done.fill(false);
        self.frequency.fill(0.0);
        self.weighted.fill(0.0);
        self.next.fill(NONE);
        self.order.clear();
        self.board = vec![0u64; nr.div_ceil(64)].into_boxed_slice();
        self.offer.fill(f32::INFINITY);
        self.ride.fill(f32::INFINITY);
        self.cn.clear();
        self.cn_of.fill((0, 0));
        self.cn_bits.clear();
        self.cn_feed.clear();
        // the destination's route nodes are reached by alighting there
        let dl = self.leaf_ix[dst] as usize;
        if dl != NONE as usize {
            for i in self.lf_off[dl] as usize..self.lf_off[dl + 1] as usize {
                self.chain[self.lf_rn[i] as usize] = false;
            }
        }
        self.dist[dst] = 0.0;
        self.heap.push(0.0, dst as u32);
        while let Some(id) = self.heap.pop() {
            #[cfg(test)]
            {
                self.pops += 1;
            }
            if id >= n + nr {
                self.settle_change(id - n - nr);
                continue;
            }
            if id >= n {
                // a boarding at its turn: onto a line that leaves riders behind (T-552), or at a
                // route node whose riders already on board do better to leave there (T-1055)
                let r = id - n;
                self.offer_line(r, self.ride[r] + g.rn_left[r]);
                continue;
            }
            let v = id;
            if self.done[v] { continue; }
            self.done[v] = true;
            self.order.push(v as u32);
            let d = self.dist[v];
            if v < s {
                self.settle_station(v);
                continue;
            }
            if self.chain[v - s] {
                self.chain_step(v - s, d);
                continue;
            }
            let x = self.rn[v - s];
            if x.boards {
                // Lines join a station's strategy in order of their cost from boarding, which
                // for a line leaving riders behind includes the extra wait (T-188): it waits
                // its turn, or a full train would join ahead of better lines and stay (T-552).
                // A boarding rides on, so where leaving the train is cheaper than staying on,
                // it waits for the ride's cost; unknown yet, it is queued when that comes.
                let on = (self.next[v] as usize) >= s && (self.next[v] as usize) < n;
                if !on {
                    if self.ride[v - s].is_finite() {
                        self.heap.push(self.ride[v - s] + x.left, (n + v - s) as u32);
                    }
                } else if x.left > 0.0 {
                    self.heap.push(d + x.left, (n + v - s) as u32);
                } else {
                    self.offer_line(v - s, d);
                }
            }
            if x.prev != NONE {
                let r = x.prev as usize;
                let rd = d + x.prev_cost;
                self.ride[r] = rd;
                if self.done[s + r] {
                    if x.prev_boards {
                        self.heap.push(rd + x.prev_left, (n + r) as u32);
                    }
                } else {
                    self.relax(s + r, rd, v as u32);
                }
            }
        }
        for i in 0..self.leaves.len() {
            if i != dl {
                self.leaf_join(i);
            }
        }
        if dl != NONE as usize {
            for i in self.lf_off[dl] as usize..self.lf_off[dl + 1] as usize {
                self.chain[self.lf_rn[i] as usize] = true;
            }
        }
    }

    /// T-2652: route node `r` at a leaf station, settled at `d`: its offer to its station noted for
    /// `leaf_join`, and the route nodes riding into it settled here in turn while they too are at a
    /// leaf station and so is the one riding into them. A run's last such node goes through the
    /// queue, so the node riding into it is reached when the queue would reach it.
    pub(super) fn chain_step(&mut self, r: usize, d: f32) {
        let (s, n) = (self.s, self.n);
        let (mut r, mut d) = (r, d);
        loop {
            let x = self.rn[r];
            if x.boards {
                // what `offer_line` would be given at this node's turn
                self.offer[r] = if x.left > 0.0 { d + x.left } else { d };
            }
            if x.prev == NONE {
                return;
            }
            let p = x.prev as usize;
            let pd = d + x.prev_cost;
            self.ride[p] = pd;
            let pp = self.rn[p].prev;
            if self.chain[p] && (pp == NONE || self.chain[pp as usize]) && !self.done[s + p] {
                self.dist[s + p] = pd;
                self.next[s + p] = (s + r) as u32;
                self.done[s + p] = true;
                self.order.push((s + p) as u32);
                (r, d) = (p, pd);
                continue;
            }
            if self.done[s + p] {
                if x.prev_boards {
                    self.heap.push(pd + x.prev_left, (n + p) as u32);
                }
            } else {
                self.relax(s + p, pd, (s + r) as u32);
            }
            return;
        }
    }

    /// T-2652: leaf station `leaves[i]`'s strategy, after the queue: its route nodes' offers joined
    /// in the order the queue would have brought them (by its key: cost, then the id of the node,
    /// or of the boarding's turn for a line with a wait of its own), up to the first that would
    /// have come after the station itself was settled. Settled last if reached.
    pub(super) fn leaf_join(&mut self, i: usize) {
        let (g, s, n) = (self.g, self.s, self.n);
        let u = self.leaves[i] as usize;
        self.lf_cand.clear();
        for k in self.lf_off[i] as usize..self.lf_off[i + 1] as usize {
            let r = self.lf_rn[k] as usize;
            let o = self.offer[r];
            if self.rn[r].boards && o.is_finite() {
                let id = if g.rn_left[r] > 0.0 { n + r } else { s + r };
                self.lf_cand.push(((o.to_bits() as u64) << 32 | id as u64, r as u32));
            }
        }
        if self.lf_cand.is_empty() {
            return;
        }
        if self.lf_cand.len() > 1 {
            self.lf_cand.sort_unstable();
        }
        let (mut f, mut w, mut cur, mut nx) = (0f32, 0f32, f32::INFINITY, NONE);
        for k in 0..self.lf_cand.len() {
            let (key, r) = self.lf_cand[k];
            if ((cur.to_bits() as u64) << 32 | u as u64) < key {
                break;
            }
            let r = r as usize;
            if let Some(c) = join_line(&mut f, &mut w, cur, g.rn_tph[r], self.offer[r], g.st_wait[u], g.change_f(u), self.p) {
                self.board[r >> 6] |= 1 << (r & 63);
                cur = c;
                nx = (s + r) as u32;
            }
        }
        self.dist[u] = cur;
        self.next[u] = nx;
        self.done[u] = true;
        self.order.push(u as u32);
    }

    #[inline(always)]
    pub(super) fn relax(&mut self, from: usize, nd: f32, to: u32) {
        if nd < self.dist[from] {
            self.dist[from] = nd;
            self.next[from] = to;
            self.heap.push(nd, from as u32);
        }
    }

    /// Route node `r`'s departures, at cost `d` from boarding, offered to its station's strategy,
    /// or once that is settled, to the station's change nodes that do not bar it.
    pub(super) fn offer_line(&mut self, r: usize, d: f32) {
        let g = self.g;
        let (from, f) = (g.rn_station[r] as usize, g.rn_tph[r]);
        self.offer[r] = d;
        if !self.done[from] {
            if let Some(cost) = join_line(&mut self.frequency[from], &mut self.weighted[from], self.dist[from], f, d, g.st_wait[from], g.change_f(from), self.p) {
                self.board[r >> 6] |= 1 << (r & 63);
                self.dist[from] = cost;
                self.next[from] = (self.s + r) as u32;
                self.heap.push(cost, from as u32);
            }
            return;
        }
        let (a, b) = self.cn_of[from];
        let i = self.edge_ix[r] as usize;
        for k in a as usize..b as usize {
            let c = &mut self.cn[k];
            // a line costing more from boarding than the pool does already never joins it
            if c.done || d >= c.dist || barred(g, &self.nst, r, c.line, c.ban) { continue; }
            if let Some(cost) = join_line(&mut c.frequency, &mut c.weighted, c.dist, f, d, g.st_wait[from], g.change_f(from), self.p) {
                c.dist = cost;
                self.cn_bits[c.bits as usize + i / 64] |= 1 << (i % 64);
                self.heap.push(cost, (self.n + self.nr + k) as u32);
            }
        }
    }

    /// The departures of its own line that an arrival at `v` off route node `a` may not take, as a
    /// change node's (line, `ban`), or `None` where it may take every one the strategy boards.
    /// Nobody changes onto the line they left towards the stop they came from (T-575), or towards
    /// the one their train goes on to; a line passing `v` again leaves it for other stops, so a rider
    /// off a lollipop's loop may change at its junction onto the line on round (T-1268). A walk from
    /// another station takes the line only away from where they rode (`walk_ban`, T-3295).
    pub(super) fn ban(&self, a: usize, v: usize) -> Option<(u32, [u32; 2])> {
        let g = self.g;
        let line = g.rn_line[a];
        if !self.lines.contains(&line) { return None; }
        if g.rn_station[a] as usize != v { return self.walk_ban(a, v, line); }
        let boarded = self.banned[self.banned_off[a] as usize..self.banned_off[a + 1] as usize].iter().any(|&r| self.board[r as usize >> 6] >> (r & 63) & 1 == 1);
        if !boarded {
            None
        } else if self.banned_all[a] {
            Some((line, WHOLE))
        } else {
            Some((line, ban_of(&self.pst, &self.nst, a)))
        }
    }

    /// `ban` for a rider walking to `v` off route node `a` of `line` (T-3295): barred are the
    /// line's departures whose next stop is the one they left or behind it along the way they rode
    /// (a circle's the nearer way round), so a walk never rides the line back over itself, and
    /// open the rest: across the bend of a line that loops back near itself, on either way. Keyed
    /// by the barred departures' next stops; past two of them, the whole line.
    fn walk_ban(&self, a: usize, v: usize, line: u32) -> Option<(u32, [u32; 2])> {
        let (g, s) = (self.g, self.s);
        let (i, d, _) = self.at[a];
        let (mut key, mut open, mut boarded) = ([NONE; 2], false, false);
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let r = g.adj_to[e] as usize - s;
            if g.rn_line[r] != line || g.rn_tph[r] == 0.0 { continue; }
            let (j, dir, circle) = self.at[r];
            let mut ahead = (j + dir - i) * d;
            if circle > 0 {
                ahead = ahead.rem_euclid(circle);
                if 2 * ahead > circle { ahead -= circle; }
            }
            if ahead > 0 {
                open = true;
                continue;
            }
            boarded |= self.board[r >> 6] >> (r & 63) & 1 == 1;
            let x = self.nst[r];
            if x == NONE || key[1] != NONE && !key.contains(&x) {
                key = WHOLE;
            } else if key != WHOLE && !key.contains(&x) {
                key[(key[0] != NONE) as usize] = x;
            }
        }
        if !boarded {
            None
        } else if !open || key == WHOLE {
            Some((line, WHOLE))
        } else {
            let k = if key[1] == NONE { [key[0]; 2] } else { [key[0].min(key[1]), key[0].max(key[1])] };
            Some((line, k))
        }
    }

    /// Station `v` is settled with its strategy. Riders reach it as they would start there unless
    /// the strategy boards a departure `ban` bars them; those come to a change node, made here
    /// from the departures offered so far less the barred ones.
    pub(super) fn settle_station(&mut self, v: usize) {
        let (g, s) = (self.g, self.s);
        let (j0, j1) = (g.rev_off[v] as usize, g.rev_off[v + 1] as usize);
        // most arrivals are settled first, riding on being cheaper than changing here
        if g.rev_from[j0..j1].iter().all(|&from| self.done[from as usize]) {
            let k = self.cn.len() as u32;
            self.cn_of[v] = (k, k);
            return;
        }
        let d = self.dist[v];
        let (e0, e1) = (g.adj_off[v] as usize, g.adj_off[v + 1] as usize);
        self.lines.clear();
        for e in e0..e1 {
            let r = g.adj_to[e] as usize - s;
            if self.board[r >> 6] >> (r & 63) & 1 == 1 && !self.lines.contains(&g.rn_line[r]) {
                self.lines.push(g.rn_line[r]);
            }
        }
        let spread = g.st_spread[v];
        if spread {
            self.pool.clear();
            for e in e0..e1 {
                let r = g.adj_to[e] as usize - s;
                if self.board[r >> 6] >> (r & 63) & 1 == 1 {
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], g.rn_tph[r]));
                }
            }
        }
        // one pass over the arrivals: those free to take the strategy go on, the others wait,
        // but for those barred their only line here whole, which have nothing to change to
        let lone = self.lone[v];
        self.feed.clear();
        self.keys.clear();
        for j in j0..j1 {
            let from = g.rev_from[j] as usize;
            if self.done[from] { continue; }
            match self.ban(from - s, v) {
                Some((line, WHOLE)) if line == lone => {}
                Some(key) => {
                    let k = self.keys.iter().position(|&x| x == key).unwrap_or_else(|| {
                        self.keys.push(key);
                        self.keys.len() - 1
                    });
                    self.feed.push((k as u32, j as u32));
                }
                None => {
                    let walk = if spread { platform_walk(g, self.p, &self.pool, from, v) } else { 0.0 };
                    self.relax(from, d + self.rev_cost[j] + walk, v as u32)
                }
            }
        }
        let first = self.cn.len() as u32;
        if !self.feed.is_empty() && lone != NONE {
            for ki in 0..self.keys.len() {
                let (line, ban) = self.keys[ki];
                let f0 = self.cn_feed.len();
                self.cn_feed.extend(self.feed.iter().filter(|f| f.0 == ki as u32).map(|f| f.1));
                if self.cn_feed.len() == f0 { continue; }
                self.cand.clear();
                for e in e0..e1 {
                    let r = g.adj_to[e] as usize - s;
                    if !barred(g, &self.nst, r, line, ban) && self.offer[r].is_finite() {
                        self.cand.push((self.offer[r], r as u32));
                    }
                }
                if self.cand.len() > 1 {
                    self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                }
                let k = self.cn.len();
                let b0 = self.cn_bits.len();
                self.cn_bits.resize(b0 + (e1 - e0).div_ceil(64), 0);
                let mut c = Change { st: v as u32, line, ban, frequency: 0.0, weighted: 0.0, dist: f32::INFINITY, done: false, feed: (f0 as u32, self.cn_feed.len() as u32), bits: b0 as u32 };
                for &(od, r) in &self.cand {
                    if let Some(cost) = join_line(&mut c.frequency, &mut c.weighted, c.dist, g.rn_tph[r as usize], od, g.st_wait[v], g.change_f(v), self.p) {
                        c.dist = cost;
                        let i = self.edge_ix[r as usize] as usize;
                        self.cn_bits[b0 + i / 64] |= 1 << (i % 64);
                    }
                }
                if c.dist.is_finite() {
                    self.heap.push(c.dist, (self.n + self.nr + k) as u32);
                }
                self.cn.push(c);
            }
        }
        self.cn_of[v] = (first, self.cn.len() as u32);
    }

    pub(super) fn settle_change(&mut self, k: usize) {
        let g = self.g;
        if self.cn[k].done { return; }
        self.cn[k].done = true;
        let id = (self.n + k) as u32;
        self.order.push(id);
        let (d, (f0, f1)) = (self.cn[k].dist, self.cn[k].feed);
        let v = self.cn[k].st as usize;
        let spread = g.st_spread[v];
        if spread {
            self.pool.clear();
            let (s, e0, b0) = (self.s, g.adj_off[v] as usize, self.cn[k].bits as usize);
            for e in e0..g.adj_off[v + 1] as usize {
                let i = e - e0;
                if self.cn_bits[b0 + i / 64] >> (i % 64) & 1 == 1 {
                    let r = g.adj_to[e] as usize - s;
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], g.rn_tph[r]));
                }
            }
        }
        for i in f0 as usize..f1 as usize {
            let j = self.cn_feed[i] as usize;
            let from = g.rev_from[j] as usize;
            if self.done[from] { continue; }
            let walk = if spread { platform_walk(g, self.p, &self.pool, from, v) } else { 0.0 };
            self.relax(from, d + self.rev_cost[j] + walk, id);
        }
    }

    /// This destination's change nodes, settled or not (an unsettled one is nobody's next node).
    pub(super) fn change_nodes(&self) -> ChangeNodes {
        ChangeNodes {
            st: self.cn.iter().map(|c| c.st).collect(),
            off: self.cn.iter().map(|c| c.bits).chain([self.cn_bits.len() as u32]).collect(),
            bits: self.cn_bits.as_slice().into(),
        }
    }
}

/// A line with `f` departures an hour and cost `d` from boarding joins a station's strategy if it
/// lowers the expected cost there below `cur`: the new cost, with the station's pooled frequency
/// and frequency-weighted cost updated.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn join_line(frequency: &mut f32, weighted: &mut f32, cur: f32, f: f32, d: f32, wait_f: f32, change_f: f32, p: &Params) -> Option<f32> {
    let sum = *frequency + f;
    let cost = (*weighted + f * d) / sum + board_cost(60.0 / sum, wait_f, change_f, p);
    if cost >= cur {
        return None;
    }
    *frequency = sum;
    *weighted += f * d;
    Some(cost)
}

/// Where a rider coming in by edge from route node `from` stands at station `v`, relative to its
/// position: the platform they alighted at, or the station's position after a walk from another.
#[inline]
pub(crate) fn arrival_at(g: &Graph, from: usize, v: usize) -> (f32, f32) {
    let r = from - g.n_st;
    if g.rn_station[r] as usize == v { (g.rn_dx[r], g.rn_dy[r]) } else { (0.0, 0.0) }
}

/// Perceived minutes walking from (x, y) to the platform at (px, py), both relative to their
/// station (T-717).
#[inline]
pub(crate) fn platform_walk_to(p: &Params, (x, y): (f32, f32), px: f32, py: f32) -> f32 {
    if px == x && py == y { 0.0 } else { p.xfer_min(((px - x).powi(2) + (py - y).powi(2)).sqrt()) * p.w_xfer }
}

/// A change at a station of several platforms walks from where the rider arrives to the platform
/// of the train they take: over `pool` (platform, departures an hour), the expected walk as the
/// first departure decides (T-717). Added to the arrival's cost rather than to each line's in the
/// pool, so which lines join the station's strategy does not depend on the platform arrived at.
#[inline]
pub(super) fn platform_walk(g: &Graph, p: &Params, pool: &[(f32, f32, f32)], from: usize, v: usize) -> f32 {
    let at = arrival_at(g, from, v);
    let (mut w, mut f) = (0.0, 0.0);
    for &(px, py, fr) in pool {
        w += fr * platform_walk_to(p, at, px, py);
        f += fr;
    }
    if f > 0.0 { w / f } else { 0.0 }
}

/// The stations anyone can alight at: those in the access lists of subzones with jobs. Nobody's
/// rail trip ends anywhere else, so the strategy searches towards other stations are not needed.
pub fn egress_stations(sz: &Subzones, n_st: usize) -> Vec<bool> {
    let mut want = vec![false; n_st];
    for v in 0..sz.n {
        if sz.jobs[v] > 0.0 {
            for k in sz.walk(v) {
                want[sz.acc_st[k] as usize] = true;
            }
        }
    }
    want
}

/// What a skims table depends on: the graph's edges and the boarding cost's parameters.
pub(super) struct SkimKey {
    pub(super) n_st: usize,
    pub(super) n: usize,
    pub(super) adj_off: Vec<u32>,
    pub(super) adj_to: Vec<u32>,
    pub(super) adj_cost: Vec<f32>,
    pub(super) rn_tph: Vec<f32>,
    pub(super) rn_left: Vec<f32>,
    pub(super) rn_dx: Vec<f32>,
    pub(super) rn_dy: Vec<f32>,
    pub(super) transfer_min: f32,
    pub(super) st_wait: Vec<f32>,
    pub(super) st_bld: Vec<bool>,
    pub(super) fare_curve: FareCurve,
    pub(super) rn_km: Vec<f32>,
    pub(super) rn_kmw: Vec<f32>,
}

impl SkimKey {
    pub(super) fn of(g: &Graph, p: &Params) -> SkimKey {
        SkimKey { n_st: g.n_st, n: g.n, adj_off: g.adj_off.clone(), adj_to: g.adj_to.clone(), adj_cost: g.adj_cost.clone(), rn_tph: g.rn_tph.clone(), rn_left: g.rn_left.clone(), rn_dx: g.rn_dx.clone(), rn_dy: g.rn_dy.clone(), transfer_min: p.change_min(), st_wait: g.st_wait.clone(), st_bld: g.st_bld.clone(), fare_curve: g.fare_curve.clone(), rn_km: g.rn_km.clone(), rn_kmw: g.rn_kmw.clone() }
    }
    pub(super) fn same(&self, g: &Graph, p: &Params) -> bool {
        self.n_st == g.n_st
            && self.n == g.n
            && self.transfer_min == p.change_min()
            && self.st_wait == g.st_wait
            && self.st_bld == g.st_bld
            && self.fare_curve == g.fare_curve
            && self.rn_km == g.rn_km
            && self.rn_kmw == g.rn_kmw
            && self.rn_dx == g.rn_dx
            && self.rn_dy == g.rn_dy
            && self.adj_off == g.adj_off
            && self.adj_to == g.adj_to
            && self.adj_cost == g.adj_cost
            && self.rn_tph == g.rn_tph
            && self.rn_left == g.rn_left
    }
}

/// Skims tables a thread keeps (`with_skims`). Each holds stations² route costs plus about 4
/// bytes a node per searched destination (u16 ids): on the real New York save 18 MB for New
/// York's table (T-398).
pub(super) const SKIM_TABLES: usize = 2;

thread_local! {
    /// Skims of the graphs a worker solved most recently, newest last (T-119).
    static SKIM_CACHE: std::cell::RefCell<Vec<(SkimKey, Skims)>> = std::cell::RefCell::new(vec![]);
}

/// Run `f` on the skims of `g` towards the stations in `want`, searching only what no earlier
/// caller on this thread has searched for the same graph. A worker solves several populations
/// (two cities and the longer-trip market) on one service, and a period's graph is the same for
/// all of them once crowding is set, so the searches are done once per period and round, not once
/// per population (T-119). The cities do not always agree to the last metre on where two stations
/// are, so a period can bring a graph per city: `SKIM_TABLES` are kept, and which go is chosen by
/// what they cost to search again, so the order the populations come in does not decide (T-405).
pub fn with_skims<R>(g: &Graph, p: &Params, want: &[bool], f: impl FnOnce(&Skims) -> R) -> R {
    SKIM_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        match c.iter().position(|(k, _)| k.same(g, p)) {
            Some(i) => {
                let e = c.remove(i);
                c.push(e);
            }
            None => {
                if c.len() >= SKIM_TABLES {
                    // the table cheapest to search again (fewest destinations, the oldest of
                    // equals) gives way, and only to one that will cost at least as much: a
                    // cheaper one is searched for this call alone
                    let todo = (0..g.n_st).filter(|&d| want.get(d).copied().unwrap_or(false)).count();
                    let (i, kept) = c.iter().enumerate().map(|(i, e)| (i, e.1.next.len())).min_by_key(|e| e.1).unwrap();
                    if todo < kept {
                        drop(c);
                        return f(&skims_for(g, p, want));
                    }
                    c.remove(i);
                }
                c.push((SkimKey::of(g, p), Skims::empty(g)));
            }
        }
        let sk = &mut c.last_mut().unwrap().1;
        skims_extend(sk, g, p, want);
        f(sk)
    })
}

/// Whether this thread holds skims of `g` (`with_skims`).
pub fn skims_cached(g: &Graph, p: &Params) -> bool {
    SKIM_CACHE.with(|c| c.borrow().iter().any(|(k, _)| k.same(g, p)))
}

/// Forget the cached skims (a worker calls this when it has finished a period's stage, so the
/// tables do not sit in memory while it waits).
pub fn clear_skim_cache() {
    SKIM_CACHE.with(|c| c.borrow_mut().clear());
}
