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
    /// leave riders on the platform (T-188, `left_behind`), plus `rn_span`. Added to the
    /// boarding's cost.
    pub rn_left: Vec<f32>,
    /// Perceived minutes per real minute of each route node's ride (a bus line's `BUS_RIDE_WEIGHT`); empty when none is a bus.
    pub rn_busw: Vec<f32>,
    /// Perceived minutes added to boarding where only a share `phi` of the period's riders have a
    /// train due within their tolerance (T-842, `Network::span`): -ln(phi) / `Params::beta`, the
    /// logit's own form for an option open to that share. 0 with service all period.
    pub rn_span: Vec<f32>,
    /// T-3888: perceived minutes a trip pays where its first boarding is at route node r
    /// (`Params::train_in_min` on a train or tram but at a tram stop on the street, T-4041, else 0), and where its last ride leaves at r (`train_out_min`); both empty where nothing pays. In the station pair's cost after the
    /// search (`trip_ends`), so route choice between lines does not see them.
    pub rn_in: Vec<f32>,
    pub rn_out: Vec<f32>,
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
    /// T-3718: per station, the stations a trip starting there may walk to before boarding, as a
    /// trip may walk to one after its last ride (`xfer_walk_m`, its complex), CSR by `st_walk_off`,
    /// with the walk's real minutes from position to position (`StartWalk`).
    pub st_walk_off: Vec<u32>,
    pub st_walk_to: Vec<u32>,
    pub st_walk_min: Vec<f32>,
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
    /// T-3578: each route node's line's own fare multiplier, which the base takes too: a trip pays
    /// the base times the largest of the lines it rides (`trip_base_mul`). Empty while every
    /// line's is 1; with it, `rn_km` is filled whatever the curve.
    pub rn_base_mul: Vec<f32>,
    /// The base fare `rn_base_mul` scales, perceived minutes (`Params::fare_min`).
    pub base_min: f32,
    /// T-3784: the transfer fare in the boarding edges' costs (`Params::xfer_fare`), which a change
    /// inside a fare zone run does not pay (`ZoneWork`)
    pub xfer_fare: f32,
    /// T-4050: per route node, its class of departures run as one evenly spaced service
    /// (`even_classes`): the least route node of the class, `NONE` in none. Empty when none is.
    pub rn_even: Vec<u32>,
    /// T-4514: per route node, its class's factor on the evenly spaced wait this period (`Network::gap_f`, the real
    /// gaps between its lines' departures), 1 or more; empty where every class's is 1.
    pub rn_gap: Vec<f32>,
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
    /// T-3740, T-3758: the fare zones, which take the curve's place on zone rides (`ZoneWork`); empty
    /// with none. `base` and `rate` are set whenever these are.
    pub zones: FareZones,
}

/// T-3740, T-3758: the player's fare zones. Per station 0 for none, else (area index + 1) << 8 | zone
/// number; per area its prices for 1, 2, ... zones (`min[off[a]..off[a + 1]]`, the last for any
/// more), US$ as set or perceived minutes on a graph. A ride on a zone line between two stations of
/// one area is a zone ride; consecutive zone rides in one area pay one price, for the zones from the
/// lowest to the highest of that area's stations they stop at or pass (`ZoneWork`).
#[derive(Clone, Default, PartialEq, Debug)]
pub struct FareZones {
    pub st: Vec<u32>,
    pub off: Vec<u32>,
    pub min: Vec<f32>,
    /// T-3758: per route node, whether its line pays the distance fare; empty while none does.
    pub dist: Vec<bool>,
    /// T-3758: per route node, the codes of the stations with a zone its trains pass without
    /// stopping on the way to the next stop, `pass[pass_off[r]..pass_off[r + 1]]`; both empty while
    /// none passes one.
    pub pass_off: Vec<u32>,
    pub pass: Vec<u32>,
}

impl FareZones {
    pub fn is_empty(&self) -> bool {
        self.st.is_empty()
    }
    /// Station `v`'s code, 0 for none.
    #[inline]
    pub fn code(&self, v: usize) -> u32 {
        self.st.get(v).copied().unwrap_or(0)
    }
    /// Whether route node `r`'s line pays the distance fare.
    #[inline]
    pub fn is_dist(&self, r: usize) -> bool {
        self.dist.get(r).copied().unwrap_or(false)
    }
    /// The codes of the stations passed after route node `r`.
    #[inline]
    pub fn passed(&self, r: usize) -> &[u32] {
        match (self.pass_off.get(r), self.pass_off.get(r + 1)) {
            (Some(&a), Some(&b)) if a <= b && b as usize <= self.pass.len() => &self.pass[a as usize..b as usize],
            _ => &[],
        }
    }
    /// Area `a`'s (index + 1) price for zones `lo` to `hi` (both counted); between two counts, as
    /// averaged zones are, the line between their prices. None for no such area.
    pub fn price_of(&self, a: u32, lo: f32, hi: f32) -> Option<f32> {
        let area = (a as usize).checked_sub(1)?;
        let (o, e) = (*self.off.get(area)? as usize, *self.off.get(area + 1)? as usize);
        if e <= o || e > self.min.len() {
            return None;
        }
        let k = (hi - lo).max(0.0);
        let i = k.floor() as usize;
        let at = |j: usize| self.min[(o + j).min(e - 1)];
        let (p0, p1) = (at(i), at(i + 1));
        Some(p0 + (p1 - p0) * (k - i as f32))
    }
}

impl FareCurve {
    pub fn is_line(&self) -> bool {
        self.km.len() < 2
    }
    /// Whether every trip pays the straight line (no curve, no fare zones): its fare km are then
    /// never needed.
    pub fn flat(&self) -> bool {
        self.is_line() && self.zones.is_empty()
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

/// T-4050: how much two lines' rides to the next stop may differ and still be one service there:
/// the track side's "one pattern" (`track::span::gaps::SAME_RIDE_S`, 30 s, its shared stretches' too).
pub const EVEN_RIDE_MIN: f32 = (crate::track::span::gaps::SAME_RIDE_S / 60.0) as f32;

/// T-4050 (Anita 11-22: "lines like the 4 and 5 trains that are on the same track and same stops,
/// no passing rails ... assumed as evenly spaced. but the 4 and 6 should be independent"): departures
/// whose trains leave one platform (station and `rn_dx`, `rn_dy`) for one platform at the next stop
/// in about the same time (`EVEN_RIDE_MIN`) share that track with no stop between to pass at, and
/// the timetable spaces such lines evenly (track-and-service.md "Spaced from other lines"). A line
/// skipping a stop has another next stop. Buses and ferries can pass, so are in no class. Per route
/// node, the least route node of its class of two or more lines, else `NONE`; empty when no class is.
#[allow(clippy::too_many_arguments)]
fn even_classes(net: &Network, s: usize, rn_station: &[u32], rn_line: &[u32], rn_next: &[u32], rn_ride: &[f32], rn_dx: &[f32], rn_dy: &[f32], rn_wrap: &[bool], rn_tph: &[f32]) -> Vec<u32> {
    let nr = rn_line.len();
    let track: Vec<bool> = (0..net.n_lines()).map(|l| net.stairs_line(l)).collect();
    // (station, next station, platform here, platform there, ride, route node)
    let mut by: Vec<([u32; 6], u32, u32)> = vec![];
    for r in 0..nr {
        if rn_tph[r] <= 0.0 || rn_wrap[r] || rn_next[r] == NONE || !track[rn_line[r] as usize] {
            continue;
        }
        let x = rn_next[r] as usize - s;
        by.push(([rn_station[r], rn_station[x], rn_dx[r].to_bits(), rn_dy[r].to_bits(), rn_dx[x].to_bits(), rn_dy[x].to_bits()], rn_ride[r].to_bits(), r as u32));
    }
    if by.len() < 2 {
        return vec![];
    }
    // (rides are positive, so their bits sort as they do)
    by.sort_unstable();
    let mut out = vec![NONE; nr];
    let mut any = false;
    let mut i = 0;
    while i < by.len() {
        let mut j = i + 1;
        while j < by.len() && by[j].0 == by[i].0 && f32::from_bits(by[j].1) - f32::from_bits(by[i].1) <= EVEN_RIDE_MIN {
            j += 1;
        }
        // (one line's two runs, as a balloon's down its trunk, are not spaced against each other)
        if by[i..j].iter().any(|x| rn_line[x.2 as usize] != rn_line[by[i].2 as usize]) {
            let least = by[i..j].iter().map(|x| x.2).min().unwrap();
            for x in &by[i..j] {
                out[x.2 as usize] = least;
            }
            any = true;
        }
        i = j;
    }
    if any { out } else { vec![] }
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
    // T-3888: each route node's stop (an index into `line_stops`), for its stairs
    let mut rn_stop: Vec<u32> = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        let wrap = net.line_wrap.get(l).copied().unwrap_or(false);
        for dir in 0..2 {
            let base = s + rn_station.len();
            let len = b - a;
            for i in 0..len {
                let k = if dir == 0 { a + i } else { b - 1 - i };
                // T-3348: the reverse run may stop at a platform beside the listed stop, its own
                // station's (so at that station's position)
                let st = if dir == 0 { net.line_stops[k] } else { net.back_stop(k) } as usize;
                let listed = st as u32 == net.line_stops[k];
                rn_station.push(st as u32);
                rn_line.push(l as u32);
                let (dx, dy) = if platforms && listed { (net.stop_x[k] - net.st_x[st], net.stop_y[k] - net.st_y[st]) } else { (0.0, 0.0) };
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
                rn_stop.push(k as u32);
            }
        }
    }
    let n = s + rn_station.len();
    let nr = rn_line.len();
    // the share, whether riders board, whether they get off (`network::stop_span`): T-3332, a stop
    // its trains pass on the other track, nobody boarding, alighting or changing there; T-4013, a
    // stop the player keeps to boarding only or getting off only
    let at = |r: usize| super::super::network::stop_span(if net.span.len() == super::super::network::PERIODS * nr { net.span[p.period * nr + r] } else { 1.0 });
    let phi = |r: usize| at(r).0;
    let boards = |r: usize| at(r).1;
    let alights = |r: usize| at(r).2;
    // T-4224: a bus line's headway counts as at least twice `BUS_MIN_WAIT_MIN`, and its rides weigh `BUS_RIDE_WEIGHT`
    let is_bus: Vec<bool> = (0..net.n_lines()).map(|l| net.bus_line(l)).collect();
    let headway = |l: usize| {
        let h = net.line_headway_min[l][p.period];
        if is_bus[l] { h.max(2.0 * super::super::network::BUS_MIN_WAIT_MIN) } else { h }
    };
    let rn_busw: Vec<f32> = if is_bus.iter().any(|&b| b) {
        rn_line.iter().map(|&l| if is_bus[l as usize] { super::super::network::BUS_RIDE_WEIGHT } else { 1.0 }).collect()
    } else {
        vec![]
    };
    let mut edges: Vec<(u32, u32, f32, u32)> = vec![];
    for (r, &st) in rn_station.iter().enumerate() {
        let node = (s + r) as u32;
        let l = rn_line[r] as usize;
        if rn_next[r] != NONE {
            if !rn_wrap[r] && boards(r) {
                edges.push((st, node, board_cost(headway(l), st_wait[st as usize], change_f_of(&st_bld, st as usize), p), NONE));
            }
            edges.push((node, rn_next[r], rn_ride[r] * rn_busw.get(r).copied().unwrap_or(1.0), node));
        }
        if alights(r) {
            edges.push((node, st, 0.0, NONE));
        }
    }
    // walking transfers: from each route node (having ridden) to the stations nearby, and to every
    // other station of its complex however far (T-161), from its platform to the other station's
    // position
    // T-4079: a bus stop's walks reach less far
    let cut2 = |a: usize, b: usize| net.walk_reach_m(a, b, p.xfer_walk_m).powi(2);
    // T-4094: a walk with a merged group of bus stops at either end is measured between the real stops
    let bw = net.bus_walks.as_deref();
    let merged = |a: usize| bw.is_some_and(|w| w.merged(a));
    let d2 = |a: usize, b: usize| match bw.filter(|_| merged(a) || merged(b)) {
        Some(w) => w.between(a as u32, b as u32).map_or(f32::INFINITY, |d| d * d),
        None => (net.st_x[b] - net.st_x[a]).powi(2) + (net.st_y[b] - net.st_y[a]).powi(2),
    };
    let mut near: Vec<Vec<u32>> = vec![vec![]; s];
    // T-4048: two stations the player separated are no walk apart
    let apart = |a: usize, b: u32| !net.st_apart.is_empty() && net.st_apart.binary_search(&[(a as u32).min(b), (a as u32).max(b)]).is_ok();
    if p.xfer_walk_m > 0.0 && s > 1 {
        let grid = StationGrid::new(net, p.xfer_walk_m, p.xfer_walk_m);
        for a in 0..s {
            grid.near(net.st_x[a], net.st_y[a], |b| {
                if b as usize != a && !merged(a) && !merged(b as usize) && d2(a, b as usize) <= cut2(a, b as usize) && !apart(a, b) {
                    near[a].push(b);
                }
            });
        }
        for (a, b) in bw.into_iter().flat_map(|w| w.pairs()) {
            if d2(a as usize, b as usize) <= cut2(a as usize, b as usize) && !apart(a as usize, b) {
                near[a as usize].push(b);
            }
        }
    }
    if net.st_group.len() == s {
        for members in complex_members(&net.st_group) {
            for &a in &members {
                for &b in &members {
                    // any two of a complex walk once: the walk limit's edge where it gave one, else this one (T-4048:
                    // a separated pair that a passage, a building or a third station joins again has none from it)
                    if a != b && !near[a as usize].contains(&b) {
                        near[a as usize].push(b);
                    }
                }
            }
        }
    }
    // a passage (T-1281) walks the straight line, on moving walkways
    let passage = |a: u32, b: u32| !net.st_passage.is_empty() && net.st_passage.binary_search(&[a.min(b), a.max(b)]).is_ok();
    let walk_min = |a: u32, b: u32, d: f32| if passage(a, b) { d / upgrades::PASSAGE_MPS / 60.0 } else { p.xfer_min(d) + net.connection_min(a as usize, b as usize) };
    let (mut st_walk_off, mut st_walk_to, mut st_walk_min) = (vec![0u32], vec![], vec![]);
    for (a, list) in near.iter().enumerate() {
        for &b in list {
            st_walk_to.push(b);
            st_walk_min.push(walk_min(a as u32, b, d2(a, b as usize).sqrt()));
        }
        st_walk_off.push(st_walk_to.len() as u32);
    }
    for (r, &st) in rn_station.iter().enumerate() {
        // a walk on from a stop is getting off there
        if !alights(r) {
            continue;
        }
        let (x, y) = (net.st_x[st as usize] + rn_dx[r], net.st_y[st as usize] + rn_dy[r]);
        for &b in &near[st as usize] {
            let d = if merged(st as usize) || merged(b as usize) { d2(st as usize, b as usize).sqrt() } else { ((net.st_x[b as usize] - x).powi(2) + (net.st_y[b as usize] - y).powi(2)).sqrt() };
            edges.push(((s + r) as u32, b, walk_min(st, b, d) * p.w_xfer, NONE));
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
    // a share that rounds to 0 is no service at that stop (T-842)
    let rn_tph: Vec<f32> = rn_line.iter().enumerate().map(|(r, &l)| {
        let h = headway(l as usize);
        if h > 0.0 && h.is_finite() && phi(r) > 0.0 && boards(r) { 60.0 / h } else { 0.0 }
    }).collect();
    let rn_even = even_classes(net, s, &rn_station, &rn_line, &rn_next, &rn_ride, &rn_dx, &rn_dy, &rn_wrap, &rn_tph);
    // T-4514: each class's factor on its evenly spaced wait this period, where any is past 1
    let rn_gap: Vec<f32> = if rn_even.is_empty() || net.gap_f.len() != super::super::network::PERIODS * nr {
        vec![]
    } else {
        let at = |r: usize| {
            let f = net.gap_f[p.period * nr + r];
            if f.is_finite() && f > 1.0 { f } else { 1.0 }
        };
        if (0..nr).any(|r| rn_even[r] != NONE && at(r) > 1.0) { (0..nr).map(at).collect() } else { vec![] }
    };
    let rn_span: Vec<f32> = (0..nr).map(|r| { let f = phi(r); if f > 0.0 && f < 1.0 { -f.ln() / p.beta } else { 0.0 } }).collect();
    // T-3888, T-3910: the boarding cost at a train's or tram's stop (`pays_ends`); only where any stop pays
    let stairs: Vec<bool> = (0..net.n_lines()).map(|l| net.stairs_line(l)).collect();
    let pays: Vec<bool> = (0..nr).map(|r| stairs[rn_line[r] as usize] && net.pays_ends(rn_stop[r] as usize)).collect();
    let (rn_in, rn_out) = if (p.train_in_min > 0.0 || p.train_out_min > 0.0) && pays.iter().any(|&t| t) {
        let by = |m: f32| pays.iter().map(|&t| if t { m } else { 0.0 }).collect::<Vec<f32>>();
        (by(p.train_in_min), by(p.train_out_min))
    } else {
        (vec![], vec![])
    };
    Graph {
        rn_even,
        rn_gap,
        rn_in,
        rn_out,
        rn_left: rn_span.clone(),
        rn_busw,
        rn_span,
        rn_dx,
        rn_dy,
        st_spread,
        st_wait,
        st_bld,
        st_walk_off,
        st_walk_to,
        st_walk_min,
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
        rn_base_mul: vec![],
        base_min: 0.0,
        xfer_fare: p.xfer_fare,
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
            g.adj_cost[e] = g.rn_ride[r] * factor[r] * g.rn_busw.get(r).copied().unwrap_or(1.0) + g.rn_fare[r];
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
    /// Per searched destination, its start walks (`DestStrategy::start`).
    pub start: Vec<Box<[(u32, u32)]>>,
    /// Per searched destination, its departures' shares (`DestStrategy::share`).
    pub share: Vec<Shares>,
    /// Per searched destination, its inter-city starts' departures (`DestStrategy::entry_dep`).
    pub entry_dep: Vec<EntryDeps>,
    /// Whether `entry` and `entry_dep` are filled (else empty per destination): only a market reads
    /// them, and at night they were a third of Tokyo's table (T-3160, `with_skims`).
    pub entries: bool,
}

/// A station as reached off one of the lines its own strategy boards (T-575): a rider never changes
/// onto the line they left towards the stop they came from or the one their train goes on to, so
/// arriving on that line they choose among the station's other lines and, where the line passes
/// the station again, its departures towards other stops (T-1268). Change node k is node `n + k`;
/// a route node of the line that leaves the train there, or walks there, has it as its next node. Each holds its station and a bit per
/// boarding edge of the station for the departures it accepts, `bits[off[k]..off[k + 1]]` (bit i
/// of word w is the station's edge 64 w + i). `off` is empty where every change node has one word (T-4179: a
/// station with more than 64 edges is rare, and the offsets were a quarter of the skims tables' change nodes).
#[derive(Clone, Default, PartialEq, Debug)]
pub struct ChangeNodes {
    pub st: Ix,
    pub off: Box<[u32]>,
    pub bits: Box<[u64]>,
}

impl ChangeNodes {
    pub fn len(&self) -> usize {
        self.st.len()
    }
    /// Change node `k`'s station.
    #[inline(always)]
    pub fn station(&self, k: usize) -> usize {
        self.st.get(k)
    }
    /// Change node `k`'s words of accepted departures.
    #[inline(always)]
    pub fn words(&self, k: usize) -> &[u64] {
        if self.off.is_empty() { std::slice::from_ref(&self.bits[k]) } else { &self.bits[self.off[k] as usize..self.off[k + 1] as usize] }
    }
    pub fn bytes(&self) -> usize {
        self.st.bytes() + 4 * self.off.len() + 8 * self.bits.len()
    }
    /// Departures an hour change node `k` accepts at route node `r`.
    pub fn tph(&self, g: &Graph, k: usize, r: usize) -> f32 {
        let st = self.station(k);
        let e0 = g.adj_off[st] as usize;
        match (e0..g.adj_off[st + 1] as usize).position(|e| g.adj_to[e] as usize == g.n_st + r) {
            Some(i) if self.words(k)[i / 64] >> (i % 64) & 1 == 1 => g.rn_tph[r],
            _ => 0.0,
        }
    }
}

/// T-4179: ascending or plain lists of station, node or offset numbers, as u16 where every value fits (most
/// networks), else u32: the skims tables' change nodes, shares and inter-city starts hold them.
#[derive(Clone, PartialEq, Debug)]
pub enum Ix {
    Short(Box<[u16]>),
    Long(Box<[u32]>),
}

impl Default for Ix {
    fn default() -> Self {
        Ix::Short(Box::new([]))
    }
}

impl Ix {
    pub fn of(v: impl ExactSizeIterator<Item = u32> + Clone) -> Ix {
        if v.clone().all(|x| x < u16::MAX as u32) { Ix::Short(v.map(|x| x as u16).collect()) } else { Ix::Long(v.collect()) }
    }
    #[inline(always)]
    pub fn get(&self, i: usize) -> usize {
        match self {
            Ix::Short(v) => v[i] as usize,
            Ix::Long(v) => v[i] as usize,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Ix::Short(v) => v.len(),
            Ix::Long(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self) -> usize {
        match self {
            Ix::Short(v) => 2 * v.len(),
            Ix::Long(v) => 4 * v.len(),
        }
    }
    /// Where `x` is in an ascending list.
    #[inline(always)]
    pub fn find(&self, x: usize) -> Option<usize> {
        match self {
            Ix::Short(v) => if x < u16::MAX as usize { v.binary_search(&(x as u16)).ok() } else { None },
            Ix::Long(v) => v.binary_search(&(x as u32)).ok(),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get(i))
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
        Skims { s: g.n_st, n: g.n, r: vec![f32::INFINITY; g.n_st * g.n_st], walk: vec![u16::MAX; g.n_st * g.n_st], walk_max: vec![0; g.n_st], slot: vec![NONE; g.n_st], next: vec![], board: vec![], order: vec![], change: vec![], entry: vec![], start: vec![], share: vec![], entry_dep: vec![], entries: true }
    }
    /// `r` with the first boarding at the planned wait (T-1430), as the inter-city trips cost it.
    pub fn entry_r(&self) -> std::borrow::Cow<'_, [f32]> {
        debug_assert!(self.entries, "a table without inter-city starts");
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
        // the column by the station a trip starts from: a start walk's (T-3718) from where it boards
        let mut from: std::borrow::Cow<[f32]> = std::borrow::Cow::Borrowed(&wm[..s]);
        if !d.start.is_empty() {
            let f = from.to_mut();
            for &(a, b) in d.start.iter() {
                f[a as usize] = start_walk_min(g, &d.board, &d.share, a as usize, b as usize, p) + wm[b as usize];
            }
        }
        for (a, &m) in from.iter().enumerate() {
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
        sk.entry.push(if sk.entries { d.entry } else { Box::default() });
        sk.start.push(d.start);
        sk.share.push(d.share.without_waits());
        sk.entry_dep.push(if sk.entries { d.entry_dep } else { EntryDeps::default() });
    }
}

/// T-3718: a trip's walk from station `a` to station `b` before its first boarding, real minutes:
/// between the stations' positions (`Graph::st_walk_min`), then at `b` to the platforms of the
/// departures `board` accepts there, by their shares, as a walk in after a ride does (T-717).
/// Infinite where `b` is not one of `a`'s.
pub fn start_walk_min(g: &Graph, board: &[u64], share: &Shares, a: usize, b: usize, p: &Params) -> f32 {
    let (i0, i1) = (g.st_walk_off[a] as usize, g.st_walk_off[a + 1] as usize);
    match (i0..i1).find(|&i| g.st_walk_to[i] as usize == b) {
        Some(i) => g.st_walk_min[i] + start_platform_min(g, board, share, b, p),
        None => f32::INFINITY,
    }
}

fn start_platform_min(g: &Graph, board: &[u64], share: &Shares, b: usize, p: &Params) -> f32 {
    if !g.st_spread[b] {
        return 0.0;
    }
    let s = g.n_st;
    let at = share.first(b);
    let (mut f, mut w, mut j) = (0f32, 0f32, 0);
    for e in g.adj_off[b] as usize..g.adj_off[b + 1] as usize {
        let r = g.adj_to[e] as usize - s;
        if accepted(board, g, r) != 0.0 {
            let a = at.map_or(1.0, |i| share.val[i + j]);
            j += 1;
            f += a;
            w += a * g.rn_dx[r].hypot(g.rn_dy[r]);
        }
    }
    if f > 0.0 { p.xfer_min(w / f) } else { 0.0 }
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
            departures(g, st.board, st.change, st.share, id, dep);
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
                wm[id] = st.share.wait(g, id, dep) + w / f;
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
            let to = if nx < s { nx } else { st.change.station(nx - n) };
            let walk = if to != here {
                (g.adj_off[id] as usize..g.adj_off[id + 1] as usize).find(|&e| g.adj_to[e] as usize == to).map_or(0.0, |e| g.adj_cost[e] * to_min)
            } else {
                0.0
            };
            let (ax, ay) = if to == here { (g.rn_dx[r], g.rn_dy[r]) } else { (0.0, 0.0) };
            let mut plat = 0.0;
            if nx != dst && g.st_spread[to] {
                departures(g, st.board, st.change, st.share, nx, dep);
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
            departures(g, st.board, st.change, st.share, id, dep);
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

/// T-3578: each node's base multiplier towards `dst` under strategy `st`, into `mb` (node ids, change
/// nodes from `g.n`): the largest own multiplier (`Graph::rn_base_mul`) of the lines ridden from it
/// on, a station or change node averaging the departures it accepts by frequency as `trip_km` does;
/// 0 at `dst`, where nothing is ridden. Exact on a single route; where riders split after a line, the
/// mean of the branches' largest is taken before that line's own, a little under the mean of each
/// rider's largest. Needs `g.rn_base_mul`.
pub fn trip_base_mul(g: &Graph, st: Strategy, dst: usize, mb: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => trip_base_mul_in(g, st, dst, next, order, mb, dep),
        (NodeIds::Long(next), NodeIds::Long(order)) => trip_base_mul_in(g, st, dst, next, order, mb, dep),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

pub(super) fn trip_base_mul_in<I: Id>(g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], mb: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    let (s, n) = (g.n_st, g.n);
    let end = n + st.change.len();
    mb.clear();
    mb.resize(end, 0.0);
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    let own = |r: usize| g.rn_base_mul.get(r).copied().unwrap_or(1.0);
    for &id in order {
        let id = id.ix();
        if id == dst {
            continue;
        }
        if id < s || id >= n {
            departures(g, st.board, st.change, st.share, id, dep);
            let (mut f, mut w) = (0f32, 0f32);
            for &(e, a) in dep.iter() {
                let r = g.adj_to[e as usize] as usize - s;
                // a boarding rides a stop first (T-1055)
                let k = if on_train(s + r) { mb[s + r] } else { own(r).max(mb[g.rn_next[r] as usize]) };
                f += a;
                w += a * k;
            }
            if f > 0.0 {
                mb[id] = w / f;
            }
        } else {
            let nx = next[id].ix();
            if nx >= end {
                continue;
            }
            mb[id] = own(id - s).max(mb[nx]);
        }
    }
}

/// T-3888: each station's way in and out towards `dst` under strategy `st`, into `out` (by station):
/// `Graph::rn_in` of its first boarding and `rn_out` of its last ride, each the mean over the
/// departures the strategy accepts by frequency as the riders split (`trip_km`). `last`: scratch,
/// per node the last ride's `rn_out` from it on. Needs `g.rn_in`.
pub fn trip_ends(g: &Graph, st: Strategy, dst: usize, out: &mut Vec<f32>, last: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => trip_ends_in(g, st, dst, next, order, out, last, dep),
        (NodeIds::Long(next), NodeIds::Long(order)) => trip_ends_in(g, st, dst, next, order, out, last, dep),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trip_ends_in<I: Id>(g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], out: &mut Vec<f32>, last: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
    let (s, n) = (g.n_st, g.n);
    let end = n + st.change.len();
    last.clear();
    last.resize(end, 0.0);
    out.clear();
    out.resize(s, 0.0);
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    for &id in order {
        let id = id.ix();
        if id == dst {
            continue;
        }
        if id < s || id >= n {
            departures(g, st.board, st.change, st.share, id, dep);
            let (mut f, mut w, mut wi) = (0f32, 0f32, 0f32);
            for &(e, a) in dep.iter() {
                let r = g.adj_to[e as usize] as usize - s;
                // a boarding rides a stop first (T-1055)
                let k = if on_train(s + r) { last[s + r] } else { last[g.rn_next[r] as usize] };
                f += a;
                w += a * k;
                wi += a * g.rn_in[r];
            }
            if f > 0.0 {
                last[id] = w / f;
                if id < s {
                    out[id] = wi / f + w / f;
                }
            }
        } else {
            let nx = next[id].ix();
            if nx >= end {
                continue;
            }
            // off the train at `dst`, or walking there after it: this is the last ride
            last[id] = if on_train(id) { last[nx] } else if nx == dst { g.rn_out[id - s] } else { last[nx] };
        }
    }
}

/// T-3578: what the base multiplier adds to a trip's fare at its start station: the base times the
/// multiplier less 1, perceived minutes (negative below 1).
#[inline]
pub fn base_extra(g: &Graph, mb: f32) -> f32 {
    g.base_min * (mb - 1.0)
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
    // T-1626: a trip's extra a km lands on each km times its multiplier, so the trip pays it times
    // its km-weighted multiplier, as `DestSearch::run` priced it
    let hops = if g.rn_kmw.is_empty() { &g.rn_km } else { &g.rn_kmw };
    let extra = |src: usize| g.fare_curve.extra(km[src]);
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => board_extra_in(g, st, dst, next, order, col, km, extra, hops, dep, q, out),
        (NodeIds::Long(next), NodeIds::Long(order)) => board_extra_in(g, st, dst, next, order, col, km, extra, hops, dep, q, out),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

/// T-3578: the base multiplier's part of the trips towards `dst` (`base_extra` on each start
/// station's multiplier in `mb`, `trip_base_mul`), into `out` as `board_extra` puts the curve's: by
/// the plain km ridden from each route node, so a line at 0 still takes its share.
#[allow(clippy::too_many_arguments)]
pub fn board_base(g: &Graph, st: Strategy, dst: usize, col: impl Fn(usize) -> f32, km: &[f32], mb: &[f32], dep: &mut Vec<(u32, f32)>, q: &mut Vec<f32>, out: &mut [f32]) {
    let extra = |src: usize| base_extra(g, mb[src]);
    match (st.next, st.order) {
        (NodeIds::Short(next), NodeIds::Short(order)) => board_extra_in(g, st, dst, next, order, col, km, extra, &g.rn_km, dep, q, out),
        (NodeIds::Long(next), NodeIds::Long(order)) => board_extra_in(g, st, dst, next, order, col, km, extra, &g.rn_km, dep, q, out),
        _ => unreachable!("a destination's ids in two widths"),
    }
}

/// Each trip's `extra(src)` spread along its strategy by `hops` a km of its `km`.
#[allow(clippy::too_many_arguments)]
pub(super) fn board_extra_in<I: Id>(g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], col: impl Fn(usize) -> f32, km: &[f32], extra: impl Fn(usize) -> f32, hops: &[f32], dep: &mut Vec<(u32, f32)>, q: &mut Vec<f32>, out: &mut [f32]) {
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
        let x = f * extra(src);
        if x != 0.0 {
            q[src] = x / km[src];
            any = true;
        }
    }
    if !any {
        return;
    }
    let on_train = |v: usize| (s..n).contains(&next[v].ix());
    let hop = |r: usize| hops.get(r).copied().unwrap_or(0.0);
    // the reverse of the settle order: every node is reached after all that feed it
    for &id in order.iter().rev() {
        let id = id.ix();
        let v = q[id];
        if v == 0.0 || id == dst {
            continue;
        }
        if id < s || id >= n {
            departures(g, st.board, st.change, st.share, id, dep);
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
    /// T-3718: the stations whose trips do better to walk to a station nearby and board there
    /// (`Search::start_walks`), (station, station walked to), by station. A trip may start with
    /// the walk it may end with; its riders are put at the station walked to, and every value read
    /// by the station a trip starts from (route cost, real minutes, fare km) is that station's
    /// plus the walk (`start_walk_min`). Nobody changing trains takes it.
    pub start: Box<[(u32, u32)]>,
    /// T-4041: the shares riders take its departures in where a node accepts more than one.
    pub share: Shares,
    /// T-4041: the departures an inter-city trip's first boarding takes, by the planned waits
    /// (`Search::entry_costs`), where they are not the plain strategy's one departure.
    pub entry_dep: EntryDeps,
}

/// T-4041: per station, the departures a trip starting there boards at the planned waits, (route
/// node, share) in `rn` and `share`, station `st[i]`'s from `off[i]` to the next station's; stations ascending.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct EntryDeps {
    pub st: Ix,
    pub off: Ix,
    pub rn: Ix,
    pub share: Box<[f32]>,
}

impl EntryDeps {
    pub(super) fn of_lists(at: &[(u32, u32)], dep: &[(u32, f32)]) -> EntryDeps {
        EntryDeps { st: Ix::of(at.iter().map(|x| x.0)), off: Ix::of(at.iter().map(|x| x.1)), rn: Ix::of(dep.iter().map(|x| x.0)), share: dep.iter().map(|x| x.1).collect() }
    }
    pub fn bytes(&self) -> usize {
        self.st.bytes() + self.off.bytes() + self.rn.bytes() + 4 * self.share.len()
    }
    /// Station `v`'s, (route node, share), or None where a trip starting there boards as the plain strategy does.
    pub fn of(&self, v: usize) -> Option<impl Iterator<Item = (usize, f32)> + '_> {
        let i = self.st.find(v)?;
        let end = if i + 1 < self.off.len() { self.off.get(i + 1) } else { self.share.len() };
        Some((self.off.get(i)..end).map(|k| (self.rn.get(k), self.share[k])))
    }
}

/// A destination's strategy, borrowed from a `Skims` slot or a `DestStrategy`.
#[derive(Clone, Copy)]
pub struct Strategy<'a> {
    pub next: &'a NodeIds,
    pub board: &'a [u64],
    pub order: &'a NodeIds,
    pub change: &'a ChangeNodes,
    pub start: &'a [(u32, u32)],
    pub share: &'a Shares,
    pub entry_dep: &'a EntryDeps,
}

impl Strategy<'_> {
    /// T-3718: the riders starting at each station moved to the station they walk to first, where
    /// they board (all taken before any is put, so none walks on from a station that walks too).
    pub fn fold_start(&self, flow: &mut [f32]) {
        if self.start.is_empty() {
            return;
        }
        let took: Vec<f32> = self.start.iter().map(|&(a, _)| std::mem::take(&mut flow[a as usize])).collect();
        for (&(_, b), f) in self.start.iter().zip(took) {
            flow[b as usize] += f;
        }
    }
    /// T-3718: a column by node (a strategy's fare km, say) made a column by the station a trip
    /// starts from: a start walk's takes the station walked to's own (read before any is written).
    /// Only for a column read by trip start, never one read again along the strategy.
    pub fn start_column(&self, col: &mut [f32]) {
        if self.start.is_empty() {
            return;
        }
        let to: Vec<f32> = self.start.iter().map(|&(_, b)| col[b as usize]).collect();
        for (&(a, _), v) in self.start.iter().zip(to) {
            col[a as usize] = v;
        }
    }
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
        self.next.bytes() + self.order.bytes() + 8 * self.board.len() + self.change.bytes() + 8 * self.entry.len() + 8 * self.start.len() + self.share.bytes() + self.entry_dep.bytes()
    }
    pub fn view(&self) -> Strategy<'_> {
        Strategy { next: &self.next, board: &self.board, order: &self.order, change: &self.change, start: &self.start, share: &self.share, entry_dep: &self.entry_dep }
    }
}

impl Skims {
    pub fn strategy(&self, slot: usize) -> Strategy<'_> {
        Strategy { next: &self.next[slot], board: &self.board[slot], order: &self.order[slot], change: &self.change[slot], start: &self.start[slot], share: &self.share[slot], entry_dep: &self.entry_dep[slot] }
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
    pub(super) mb: Vec<f32>,
    /// T-3888: `trip_ends`' result and scratch
    pub(super) ends: Vec<f32>,
    pub(super) ends_last: Vec<f32>,
    pub(super) dep: Vec<(u32, f32)>,
    pub(super) col: Vec<f32>,
    pub(super) ent: Vec<f32>,
    pub(super) fold: Vec<(usize, f32, f32)>,
    /// T-3758: the fares under fare zones
    pub(super) zw: ZoneWork,
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
            mb: vec![],
            ends: vec![],
            ends_last: vec![],
            dep: vec![],
            col: vec![],
            ent: vec![],
            fold: vec![],
            zw: ZoneWork::default(),
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
        let cn = |q: &Search| q.cn.iter().map(|c| (c.st, c.line, c.ban, c.back, c.even, c.cut.to_bits(), c.n, c.dist.to_bits(), c.done, c.feed, c.bits)).collect::<Vec<_>>();
        assert_eq!(cn(a), cn(b), "T-2652: change nodes towards {dst}");
        assert_eq!((&a.cn_bits, &a.cn_feed), (&b.cn_bits, &b.cn_feed), "T-2652: change nodes' departures towards {dst}");
        assert_eq!(a.share, b.share, "T-2652: departures' shares towards {dst}");
        let mut at = vec![usize::MAX; a.n + a.cn.len()];
        for (k, &v) in a.order.iter().enumerate() {
            at[v as usize] = k;
        }
        for &v in &a.order[1..] {
            let v = v as usize;
            if v < a.n && (v >= a.s || a.next[v] != NONE) {
                assert!(at[a.next[v] as usize] < at[v], "T-2652: node {v} settled before its next towards {dst}");
            }
            // T-4041: a station or change node after its departures, late lines too
            if v < a.s || v >= a.n {
                let (st, on): (usize, Box<dyn Fn(usize, usize) -> bool>) = if v < a.s {
                    (v, Box::new(|_, r: usize| a.board[r >> 6] >> (r & 63) & 1 == 1))
                } else {
                    let b0 = a.cn[v - a.n].bits as usize;
                    (a.cn[v - a.n].st as usize, Box::new(move |i: usize, _| a.cn_bits[b0 + i / 64] >> (i % 64) & 1 == 1))
                };
                let e0 = g.adj_off[st] as usize;
                for e in e0..g.adj_off[st + 1] as usize {
                    let r = g.adj_to[e] as usize - a.s;
                    if on(e - e0, r) {
                        assert!(at[a.s + r] < at[v], "T-4041: node {v} settled before its departure {r} towards {dst}");
                    }
                }
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
        let start = q.start_walks(dst).into_boxed_slice();
        q.entry_costs(dst);
        let entry_dep = EntryDeps::of_lists(&q.entry_at, &q.entry_dep);
        let mut d = DestStrategy { next: NodeIds::of(&q.next, ids), board: std::mem::take(&mut q.board), order: NodeIds::of(&q.order, ids), change: q.change_nodes(), entry: Box::new([]), start, share: std::mem::take(&mut q.share), entry_dep };
        if let Some(mut put) = put {
            let curve = &q.g.fare_curve;
            // T-3758: under fare zones a trip's fare follows its rides (`ZoneWork`)
            let zoned = !curve.zones.is_empty();
            let bent = !curve.flat() && !zoned;
            let weighed = bent && !q.g.rn_kmw.is_empty();
            if zoned {
                self.zw.run(q.g, d.view(), dst, &q.g.rn_km, &q.g.rn_kmw, &mut self.dep);
            }
            if bent {
                trip_km(q.g, d.view(), dst, &mut self.km, &mut self.dep);
            }
            if weighed {
                trip_km_on(q.g, d.view(), dst, &q.g.rn_kmw, &mut self.kmw, &mut self.dep);
            }
            // T-3578: the base times the trip's largest own multiplier, less the base mode choice adds
            let based = !q.g.rn_base_mul.is_empty();
            if based {
                trip_base_mul(q.g, d.view(), dst, &mut self.mb, &mut self.dep);
            }
            // T-3888: a train's way in and out
            let ends = !q.g.rn_in.is_empty();
            if ends {
                trip_ends(q.g, d.view(), dst, &mut self.ends, &mut self.ends_last, &mut self.dep);
            }
            // each station's route cost, and its cost as an inter-city trip's start where lower
            let (col, ent) = (&mut self.col, &mut self.ent);
            col.clear();
            ent.clear();
            for src in 0..s {
                if src == dst {
                    col.push(0.0);
                    ent.push(f32::INFINITY);
                    continue;
                }
                // The first boarding is not a transfer; the trip's base fare is added later.
                let x = if zoned { self.zw.extra_less_free(src, q.g.xfer_fare) } else if bent { curve.extra(self.km[src]) * if weighed { fare_mul_of(self.km[src], self.kmw[src]) } else { 1.0 } } else { 0.0 };
                let change = q.p.change_at(q.g.change_f(src));
                let b = if based { base_extra(q.g, self.mb[src]) } else { 0.0 };
                let b = if ends { b + self.ends[src] } else { b };
                col.push(q.dist[src] - change + x + b);
                ent.push(if q.entry[src] < q.dist[src] - 1e-4 { q.entry[src] - change + x + b } else { f32::INFINITY });
            }
            // T-3718: a start walk's costs are the station walked to's plus the walk, from the
            // stations' own (no walk follows another)
            self.fold.clear();
            for &(a, b) in d.start.iter() {
                let w = start_walk_min(q.g, &d.board, &d.share, a as usize, b as usize, q.p) * q.p.w_xfer;
                let (b, a) = (b as usize, a as usize);
                self.fold.push((a, col[b] + w, ent[b] + w));
            }
            for &(a, c, e) in &self.fold {
                col[a] = c;
                ent[a] = e;
            }
            let mut entry = vec![];
            for src in 0..s {
                put(src, col[src]);
                if ent[src] < col[src] {
                    entry.push((src as u32, ent[src]));
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
/// of that line's departures it bars (`barred`), the least of its accepted lines' cost from boarding
/// plus their spread (`cut`: a line costing that from boarding is never first), its cost, settled, its arrivals (a range of `Search::cn_feed`: the
/// station's incoming edges off that line) and where its accepted departures' bits start in
/// `Search::cn_bits`.
pub(super) struct Change {
    pub(super) st: u32,
    pub(super) line: u32,
    pub(super) ban: [u32; 2],
    /// T-4064: the station whose way every departure is barred, and the evenly spaced service whose
    /// departures are (`Search::xbar`); `NONE` for none
    pub(super) back: u32,
    pub(super) even: u32,
    pub(super) cut: f32,
    /// How many lines it accepts, and their list in `Search::lines_at` (cost from boarding,
    /// departures an hour, route node, next; `NONE` ends it).
    pub(super) n: u32,
    pub(super) head: u32,
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
    /// Per station, `Change::cut` for its strategy (T-4041), and how many lines it has joined.
    pub(super) cut: Vec<f32>,
    pub(super) joined: Vec<u32>,
    /// Per station, its joined lines' list in `lines_at`, as `Change::head`.
    pub(super) st_head: Vec<u32>,
    pub(super) lines_at: Vec<(f32, f32, u32, u32)>,
    pub(super) by_edge: Vec<(u32, f32, f32, u32)>,
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
    pub(super) keys: Vec<Bar>,
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
    /// T-4041: the lines a node being joined accepts, (cost from boarding, departures an hour), its
    /// departures' shares, `pool_cost`'s scratch, and the shares of the nodes settled so far
    pub(super) pl: Vec<(f32, f32)>,
    pub(super) sh: Vec<f32>,
    pub(super) pw: PoolWork,
    pub(super) shares: SharesBuild,
    /// T-4050: each line in `pl`'s route node; `pl` with the evenly spaced services taken as one
    /// (`merge_even`), each line's place there, and their shares
    pub(super) pk: Vec<u32>,
    pub(super) pm: Vec<(f32, f32)>,
    /// T-4514: each `pm` entry's trains an hour as they run, which its lines' riders split by; `pm`'s own are fewer
    /// where the service's real gaps make its wait longer than evenly spaced (`Graph::rn_gap`)
    pub(super) pmf: Vec<f32>,
    pub(super) pmap: Vec<u32>,
    pub(super) shm: Vec<f32>,
    pub(super) pord: Vec<u32>,
    /// T-4050: per route node settled, where a rider on its train gets off, (station, stops ridden
    /// on first); empty while nothing is evenly spaced (`Graph::rn_even`).
    pub(super) alight: Vec<(u32, u32)>,
    /// After `run`, the departures' shares towards its destination.
    pub(super) share: Shares,
    /// T-4041: the lines that joined a node already settled, (station, or change node from `n`,
    /// route node), and `late_fix`'s scratch: per node its state in the walk for the settle order
    /// and its first entry in `late` (`NONE` for none), the walk's stack and the departures it has
    /// yet to go through, (node, late line or `NONE`).
    pub(super) late: Vec<(u32, u32)>,
    pub(super) late_out: Vec<bool>,
    pub(super) mark: Vec<u8>,
    pub(super) late_at: Vec<u32>,
    pub(super) stack: Vec<(u32, u32, u32, u32, u32)>,
    pub(super) succ: Vec<(u32, u32)>,
    pub(super) reorder: Vec<u32>,
    /// After `entry_costs`, `EntryDeps`' two lists.
    pub(super) entry_at: Vec<(u32, u32)>,
    pub(super) entry_dep: Vec<(u32, f32)>,
    /// The stations `entry_costs` goes through: a line there is slower than 15 minutes or left behind.
    pub(super) entry_st: Vec<u32>,
    /// Each incoming edge's cost (`Graph::rev_edge`'s), read in edge order.
    pub(super) rev_cost: Vec<f32>,
    /// Per route node, what settling it reads, in one place (T-1647).
    pub(super) rn: Vec<RnStep>,
    /// Per route node, a range of `banned`: the departures with trains of its line from its station
    /// that `ban_of` bars an arrival off it from; and whether they are all the line's there.
    pub(super) banned_off: Vec<u32>,
    pub(super) banned: Vec<u32>,
    pub(super) banned_all: Vec<bool>,
    /// T-4064: per route node, what an arrival off it at its own station is barred beyond its own
    /// line's departures (`Bar`'s `back` and `even`, `NONE` where nothing there is), and per route
    /// node the evenly spaced service of the route node riding into it (empty while none is)
    pub(super) xbar: Vec<(u32, u32)>,
    pub(super) prev_even: Vec<u32>,
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

/// What a change node bars: its line's departures (`line`, `ban`: `barred`) and, T-4064 (Anita
/// 11-23: "makes more sense to just wait for a train that they can take all the way"), any line's
/// towards the station the rider came from (`back`), and the departures of the evenly spaced service
/// they rode in on (`even`, T-4050: that train is right behind theirs). `NONE` bars nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Bar {
    pub(super) line: u32,
    pub(super) ban: [u32; 2],
    pub(super) back: u32,
    pub(super) even: u32,
}

impl Search<'_> {
    /// Whether `b` bars route node `r`.
    #[inline(always)]
    pub(super) fn bars(&self, r: usize, b: Bar) -> bool {
        barred(self.g, &self.nst, r, b.line, b.ban) || (b.back != NONE && self.nst[r] == b.back) || (b.even != NONE && self.prev_even[r] == b.even)
    }
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
        // T-4064: per arrival at its own station, the bars beyond its own line's departures that
        // some other line's departure there meets
        let prev_even: Vec<u32> = if g.rn_even.is_empty() { vec![] } else { back.iter().map(|&b| if b == NONE { NONE } else { g.rn_even[b as usize] }).collect() };
        let xbar: Vec<(u32, u32)> = (0..nr)
            .map(|a| {
                let (v, line, from) = (g.rn_station[a] as usize, g.rn_line[a], pst[a]);
                let even = if prev_even.is_empty() { NONE } else { prev_even[a] };
                let (mut xb, mut xe) = (NONE, NONE);
                for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                    let r = g.adj_to[e] as usize - s;
                    if g.rn_line[r] == line || g.rn_tph[r] == 0.0 { continue; }
                    if from != NONE && nst[r] == from { xb = from; }
                    if even != NONE && prev_even[r] == even { xe = even; }
                }
                (xb, xe)
            })
            .collect();
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
        // the stations where a planned pool can differ from the plain one (`entry_costs`)
        let entry_st: Vec<u32> = (0..s as u32)
            .filter(|&v| {
                let v = v as usize;
                (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).any(|e| {
                    let r = g.adj_to[e] as usize - s;
                    g.rn_tph[r] > 0.0 && (g.rn_span[r] != 0.0 || !same_spread(g.rn_tph[r], g.st_wait[v]))
                })
            })
            .collect();
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
            cut: vec![f32::INFINITY; s],
            joined: vec![0; s],
            st_head: vec![NONE; s],
            lines_at: vec![],
            by_edge: vec![],
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
            pl: vec![],
            pk: vec![],
            pm: vec![],
            pmf: vec![],
            pmap: vec![],
            shm: vec![],
            pord: vec![],
            alight: if g.rn_even.is_empty() { vec![] } else { vec![(NONE, 0); nr] },
            sh: vec![],
            pw: PoolWork::default(),
            shares: SharesBuild::default(),
            share: Shares::default(),
            entry_at: vec![],
            entry_dep: vec![],
            entry_st,
            late: vec![],
            late_out: vec![],
            mark: vec![],
            late_at: vec![],
            stack: vec![],
            succ: vec![],
            reorder: vec![],
            rn,
            rev_cost,
            banned_off,
            banned,
            banned_all,
            xbar,
            prev_even,
            at,
        }
    }

    /// T-1430: after `run`, each station's cost towards `dst` as an inter-city trip's start: the
    /// departures offered to it, less their tolerance penalty (`Graph::rn_span`), pooled as
    /// `pool_cost` pools them but at the planned waits. Every departure was offered by the search's
    /// end; the pool takes every one that can be first.
    pub(super) fn entry_costs(&mut self, dst: usize) {
        let (g, s) = (self.g, self.s);
        self.entry.clear();
        self.entry.resize(s, f32::INFINITY);
        self.entry_at.clear();
        self.entry_dep.clear();
        #[cfg(test)]
        {
            let mut skipped = vec![true; s];
            for &v in &self.entry_st {
                skipped[v as usize] = false;
            }
            for v in 0..s {
                if skipped[v] && v != dst && self.dist[v].is_finite() {
                    self.check_not_kept(v);
                }
            }
        }
        for i in 0..self.entry_st.len() {
            let v = self.entry_st[i] as usize;
            if v == dst || !self.dist[v].is_finite() {
                continue;
            }
            self.cand.clear();
            // where every line's planned spread is its plain one and none is left behind, the
            // planned pool is the plain pool (the station's own cost), never kept
            let mut same_all = true;
            for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                if g.rn_tph[r] > 0.0 && self.offer[r].is_finite() {
                    self.cand.push((self.offer[r] - g.rn_span[r], r as u32));
                    same_all &= g.rn_span[r] == 0.0 && same_spread(g.rn_tph[r], g.st_wait[v]);
                }
            }
            #[cfg(test)]
            if same_all { self.check_not_kept(v); }
            if same_all {
                continue;
            }
            self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            // T-4041: every line that can be first, at the planned waits
            self.pl.clear();
            self.pk.clear();
            let mut cut = f32::INFINITY;
            let mut same = true;
            for &(d, r) in &self.cand {
                if !(d < cut) {
                    break;
                }
                let f = g.rn_tph[r as usize];
                same &= g.rn_span[r as usize] == 0.0 && same_spread(f, g.st_wait[v]);
                cut = cut.min(d + wait_spread(f, g.st_wait[v], true));
                self.pl.push((d, f));
                self.pk.push(r);
            }
            if self.pl.is_empty() {
                continue;
            }
            #[cfg(test)]
            if same { self.check_not_kept(v); }
            if same {
                continue;
            }
            self.sh.clear();
            self.sh.resize(self.pl.len(), 1.0);
            self.entry[v] = self.pool_lines(v, true, true).0;
            // kept where the planned waits cost less (else the trip is costed, so loaded, as any)
            // and the first boarding is not the plain strategy's one departure
            if !(self.entry[v] < self.dist[v] - 1e-4) {
                continue;
            }
            let r0 = self.cand[0].1 as usize;
            let plain_one = self.pl.len() == 1 && self.board[r0 >> 6] >> (r0 & 63) & 1 == 1 && self.share.first(v).is_none()
                && (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).filter(|&e| { let r = g.adj_to[e] as usize - s; self.board[r >> 6] >> (r & 63) & 1 == 1 }).count() == 1;
            if !plain_one {
                self.entry_at.push((v as u32, self.entry_dep.len() as u32));
                for (k, &x) in self.sh.iter().enumerate() {
                    if x > 0.0 {
                        self.entry_dep.push((self.cand[k].1, x));
                    }
                }
            }
        }
    }

    /// Test only: the entry of a station skipped for having the same planned and plain pool is not kept.
    #[cfg(test)]
    fn check_not_kept(&mut self, v: usize) {
        let (g, s) = (self.g, self.s);
        let (pl, pk, sh) = (self.pl.clone(), self.pk.clone(), self.sh.clone());
        self.cand.clear();
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let r = g.adj_to[e] as usize - s;
            if g.rn_tph[r] > 0.0 && self.offer[r].is_finite() {
                self.cand.push((self.offer[r] - g.rn_span[r], r as u32));
            }
        }
        self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        self.pl.clear();
        self.pk.clear();
        let mut cut = f32::INFINITY;
        for &(d, r) in &self.cand {
            if !(d < cut) {
                break;
            }
            let f = g.rn_tph[r as usize];
            cut = cut.min(d + wait_spread(f, g.st_wait[v], true));
            self.pl.push((d, f));
            self.pk.push(r);
        }
        if !self.pl.is_empty() {
            self.sh.clear();
            self.sh.resize(self.pl.len(), 1.0);
            let e = self.pool_lines(v, true, false).0;
            assert!(!(e < self.dist[v] - 1e-4), "entry kept at station {v}: {e} against {}", self.dist[v]);
        }
        (self.pl, self.pk, self.sh) = (pl, pk, sh);
    }

    /// T-3718: after `run`, the stations whose trips towards `dst` do better to walk first to a
    /// station of theirs (`Graph::st_walk_to`) and board there, (station, station walked to): by
    /// the boarding pool's cost less its first boarding's change minutes, plus the walk weighed as a
    /// walking transfer (`start_walk_min`), lower by more than a rounding. A walk to `dst` itself is
    /// no rail trip. The fare curve and base multiplier are left out of the choice, so a search
    /// that writes no route costs chooses the same.
    pub(super) fn start_walks(&self, dst: usize) -> Vec<(u32, u32)> {
        let (g, p) = (self.g, self.p);
        let mut out = vec![];
        if g.st_walk_to.is_empty() {
            return out;
        }
        let own = |v: usize| self.dist[v] - p.change_at(g.change_f(v));
        for a in 0..self.s {
            if a == dst {
                continue;
            }
            let (mut best, mut to) = (own(a) - 1e-4, NONE);
            for i in g.st_walk_off[a] as usize..g.st_walk_off[a + 1] as usize {
                let b = g.st_walk_to[i] as usize;
                if b == dst || !self.dist[b].is_finite() {
                    continue;
                }
                let c = (g.st_walk_min[i] + start_platform_min(g, &self.board, &self.share, b, p)) * p.w_xfer + own(b);
                if c < best {
                    (best, to) = (c, b as u32);
                }
            }
            if to != NONE {
                out.push((a as u32, to));
            }
        }
        out
    }

    pub(super) fn run(&mut self, dst: usize) {
        let (g, s, n, nr) = (self.g, self.s, self.n, self.nr);
        self.dist.fill(f32::INFINITY);
        self.done.fill(false);
        self.cut.fill(f32::INFINITY);
        // (no line joins the destination)
        self.cut[dst] = f32::NEG_INFINITY;
        self.late.clear();
        self.joined.fill(0);
        self.st_head.fill(NONE);
        self.lines_at.clear();
        self.shares.clear();
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
                self.note_station(v);
                self.settle_station(v);
                continue;
            }
            self.note_alight(v - s);
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
        self.late_fix(dst);
        self.share = self.shares.finish();
    }

    /// T-4050: route node `r` settled: where a rider on its train gets off (`alight`).
    #[inline]
    fn note_alight(&mut self, r: usize) {
        if self.alight.is_empty() {
            return;
        }
        let (s, n) = (self.s, self.n);
        let x = self.next[s + r] as usize;
        self.alight[r] = if x == NONE as usize {
            (NONE, 0)
        } else if x < s {
            (x as u32, 0)
        } else if x < n {
            let (st, k) = self.alight[x - s];
            (st, k + 1)
        } else {
            (self.cn[x - n].st, 0)
        };
    }

    /// The cost at station `v` (the wait's and change's factors) of the lines in `pl`.
    #[inline]
    fn pool_at(&mut self, v: usize) -> f32 {
        self.pool_lines(v, false, false).0
    }

    /// `pool_cost` of the lines in `pl` at station `v`, with each evenly spaced service taken as
    /// one (`merge_even`); with `want`, the lines' shares into `sh` (as long as `pl`).
    fn pool_lines(&mut self, v: usize, planned: bool, want: bool) -> (f32, f32) {
        let g = self.g;
        let (wf, cf) = (g.st_wait[v], g.change_f(v));
        if !self.merge_even(wf, planned) {
            return pool_cost(&self.pl, wf, cf, planned, self.p, &mut self.pw, if want { Some(&mut self.sh) } else { None });
        }
        let out = pool_cost(&self.pm, wf, cf, planned, self.p, &mut self.pw, if want { Some(&mut self.shm) } else { None });
        if want {
            for i in 0..self.pl.len() {
                let m = self.pmap[i];
                self.sh[i] = if m == NONE { 0.0 } else { self.shm[m as usize] * self.pl[i].1 / self.pmf[m as usize] };
            }
        }
        out
    }

    /// T-4050: the lines in `pl` whose departures are one evenly spaced service (`Graph::rn_even`)
    /// for riders who get off where they would (`alight`: the trip the lines serve alike) into one
    /// entry of `pm`, as riders take evenly spaced trains (the common-lines rule): from the cheapest,
    /// each next joins while its cost from boarding is below the service's so far (the mean of its
    /// lines' by trains an hour, plus half their combined headway times `wait_f`); trains an hour
    /// summed, so riders split by trains an hour. A line that does not join is never taken there
    /// (`pmap` `NONE`): its trains come between the service's. Coupled lines whose riders get off
    /// elsewhere (a stop on one branch) stay apart, as any. T-4514: two or more joined wait by their real
    /// gaps (`Graph::rn_gap`, the largest of theirs): the service's trains an hour in `pm` over that factor, the
    /// ones they run in `pmf`. False, `pm` untouched, when no two lines are one service.
    fn merge_even(&mut self, wait_f: f32, planned: bool) -> bool {
        let (g, s) = (self.g, self.s);
        let k = self.pl.len();
        if g.rn_even.is_empty() || k < 2 {
            return false;
        }
        let key = |q: &Search, i: usize| {
            let r = q.pk[i] as usize;
            let c = g.rn_even[r];
            if c == NONE { None } else { Some((c, q.alight[g.rn_next[r] as usize - s])) }
        };
        let mut any = false;
        'find: for i in 0..k {
            if let Some(a) = key(self, i) {
                for j in i + 1..k {
                    if key(self, j) == Some(a) {
                        any = true;
                        break 'find;
                    }
                }
            }
        }
        if !any {
            return false;
        }
        self.pm.clear();
        self.pmf.clear();
        // each line's first line with its key, then its entry in `pm`
        self.pmap.clear();
        for i in 0..k {
            let ki = key(self, i);
            let lead = ki.and_then(|a| (0..i).find(|&j| key(self, j) == Some(a))).unwrap_or(i);
            self.pmap.push(lead as u32);
        }
        let mut ord = std::mem::take(&mut self.pord);
        for i in 0..k {
            if self.pmap[i] as usize != i {
                continue;
            }
            let at = self.pm.len() as u32;
            ord.clear();
            ord.extend((i..k).filter(|&j| self.pmap[j] as usize == i).map(|j| j as u32));
            if ord.len() == 1 {
                self.pm.push(self.pl[i]);
                self.pmf.push(self.pl[i].1);
                self.pmap[i] = at;
                continue;
            }
            ord.sort_unstable_by(|&a, &b| self.pl[a as usize].0.total_cmp(&self.pl[b as usize].0).then(a.cmp(&b)));
            // (sums of cost x trains an hour, then the mean; the joined lines' largest factor on their wait)
            let (mut df, mut sf, mut joined, mut gf) = (0f64, 0f64, 0u32, 1f32);
            for &j in ord.iter() {
                let (d, f) = (self.pl[j as usize].0 as f64, self.pl[j as usize].1 as f64);
                let joins = sf == 0.0 || d < df / sf + wait_spread(sf as f32, wait_f, planned) as f64 / 2.0;
                if joins {
                    (df, sf, joined) = (df + d * f, sf + f, joined + 1);
                    if let Some(&x) = g.rn_gap.get(self.pk[j as usize] as usize) {
                        gf = gf.max(x);
                    }
                }
                self.pmap[j as usize] = if joins { at } else { NONE };
            }
            let gf = if joined > 1 { gf as f64 } else { 1.0 };
            self.pm.push(((df / sf) as f32, (sf / gf) as f32));
            self.pmf.push(sf as f32);
        }
        self.pord = ord;
        self.shm.clear();
        self.shm.resize(self.pm.len(), 1.0);
        true
    }

    /// Node `id` (a station, or change node `id - n`) at station `v` settled with the lines in `pl`:
    /// their shares into `sh`, and where two or more, into `shares`.
    fn note_shares(&mut self, id: usize, v: usize) -> f32 {
        let g = self.g;
        self.sh.clear();
        self.sh.resize(self.pl.len(), 1.0);
        if self.pl.is_empty() {
            return f32::INFINITY;
        }
        // (`pool_cost` gives the same to the bit whatever the lines' order)
        let (cost, wait) = self.pool_lines(v, false, true);
        if self.pl.len() > 1 {
            self.shares.push(id, wait / g.st_wait[v], &self.sh);
        }
        cost
    }

    fn note_station(&mut self, v: usize) {
        if self.joined[v] > 1 {
            self.list_by_edge(self.st_head[v]);
            let _ = self.note_shares(v, v);
        } else {
            self.sh.clear();
            self.sh.resize(self.joined[v] as usize, 1.0);
        }
    }

    /// Station `v`'s cost after line `r` (cost `d` from boarding) joins it: alone, as `pool_cost`
    /// gives one line, else over every line it has joined.
    #[inline]
    fn join_cost(&mut self, v: usize, r: usize, d: f32) -> f32 {
        self.joined[v] += 1;
        self.st_head[v] = self.add_line(self.st_head[v], d, r);
        if self.joined[v] == 1 {
            return self.one_cost(v, r, d);
        }
        self.list_lines(self.st_head[v]);
        self.pool_at(v)
    }

    /// A line onto the front of a list in `lines_at`: the list's new head.
    #[inline]
    fn add_line(&mut self, head: u32, d: f32, r: usize) -> u32 {
        self.lines_at.push((d, self.g.rn_tph[r], r as u32, head));
        (self.lines_at.len() - 1) as u32
    }

    /// `list_lines` in edge order (as `station_lines` and `change_lines` give them).
    fn list_by_edge(&mut self, head: u32) {
        self.by_edge.clear();
        let mut i = head;
        while i != NONE {
            let (d, f, r, nx) = self.lines_at[i as usize];
            self.by_edge.push((self.edge_ix[r as usize], d, f, r));
            i = nx;
        }
        self.by_edge_into_pl();
    }

    /// `by_edge` sorted into `pl` and `pk`.
    fn by_edge_into_pl(&mut self) {
        self.by_edge.sort_unstable_by_key(|x| x.0);
        self.pl.clear();
        self.pl.extend(self.by_edge.iter().map(|x| (x.1, x.2)));
        self.pk.clear();
        self.pk.extend(self.by_edge.iter().map(|x| x.3));
    }

    /// The lines of the list from `head` into `pl` (in no particular order: `pool_cost` sorts them).
    #[inline]
    fn list_lines(&mut self, head: u32) {
        self.pl.clear();
        self.pk.clear();
        let mut i = head;
        while i != NONE {
            let (d, f, r, nx) = self.lines_at[i as usize];
            self.pl.push((d, f));
            self.pk.push(r);
            i = nx;
        }
    }

    /// Line `r` alone at station `v`, `d` from boarding: what `pool_cost` gives one line.
    #[inline]
    fn one_cost(&self, v: usize, r: usize, d: f32) -> f32 {
        let g = self.g;
        d + board_cost(60.0 / g.rn_tph[r], g.st_wait[v], g.change_f(v), self.p)
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
                self.note_alight(p);
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
    /// or of the boarding's turn for a line with a wait of its own), each that can be first, as
    /// `offer_line` joins them before and after the station is settled. Settled last if reached.
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
        // as `offer_line` joins them while the station waits in the queue
        // as `offer_line` joins them, while the station waits in the queue and after (its cost is
        // the last)
        let (mut cut, mut cur, mut nx) = (f32::INFINITY, f32::INFINITY, NONE);
        for k in 0..self.lf_cand.len() {
            let r = self.lf_cand[k].1 as usize;
            let o = self.offer[r];
            if o < cut {
                self.board[r >> 6] |= 1 << (r & 63);
                cut = cut.min(o + wait_spread(g.rn_tph[r], g.st_wait[u], false));
                cur = self.join_cost(u, r, o);
                nx = (s + r) as u32;
            }
        }
        self.dist[u] = cur;
        self.next[u] = nx;
        self.done[u] = true;
        self.order.push(u as u32);
        self.note_station(u);
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
    /// or once that is settled, to the station's change nodes that do not bar it and are still in
    /// the queue (a change node takes no late lines: T-4054, DECISIONS 2026-11-23).
    pub(super) fn offer_line(&mut self, r: usize, d: f32) {
        let g = self.g;
        let (from, f) = (g.rn_station[r] as usize, g.rn_tph[r]);
        self.offer[r] = d;
        // T-4041: a line joins while its station waits in the queue, unless it is never first
        let spread = wait_spread(f, g.st_wait[from], false);
        if !self.done[from] {
            if d < self.cut[from] {
                self.board[r >> 6] |= 1 << (r & 63);
                self.cut[from] = self.cut[from].min(d + spread);
                // the cost can rise as a slow line joins (`pool_cost`); the station keeps its
                // place in the queue by the least
                let cost = self.join_cost(from, r, d);
                self.next[from] = (self.s + r) as u32;
                if cost < self.dist[from] {
                    self.heap.push(cost, from as u32);
                }
                self.dist[from] = cost;
            }
            return;
        }
        if d < self.cut[from] {
            // a line that can be first, offered after its station left the queue: it joins all
            // the same (its cost and shares are made in `late_fix`)
            self.board[r >> 6] |= 1 << (r & 63);
            self.cut[from] = self.cut[from].min(d + spread);
            self.joined[from] += 1;
            self.st_head[from] = self.add_line(self.st_head[from], d, r);
            self.next[from] = (self.s + r) as u32;
            self.late.push((from as u32, r as u32));
        }
        let (a, b) = self.cn_of[from];
        let i = self.edge_ix[r] as usize;
        for k in a as usize..b as usize {
            let c = &self.cn[k];
            if c.done || !(d < c.cut) || self.bars(r, Bar { line: c.line, ban: c.ban, back: c.back, even: c.even }) { continue; }
            let c = &mut self.cn[k];
            c.cut = c.cut.min(d + spread);
            c.n += 1;
            self.cn_bits[c.bits as usize + i / 64] |= 1 << (i % 64);
            let head = self.cn[k].head;
            self.cn[k].head = self.add_line(head, d, r);
            let cost = if self.cn[k].n == 1 {
                self.one_cost(from, r, d)
            } else {
                self.list_lines(self.cn[k].head);
                self.pool_at(from)
            };
            let c = &mut self.cn[k];
            if cost < c.dist {
                self.heap.push(cost, (self.n + self.nr + k) as u32);
            }
            c.dist = cost;
        }
    }

    /// The departures of its own line that an arrival at `v` off route node `a` may not take, as a
    /// change node's (line, `ban`), or `None` where it may take every one the strategy boards.
    /// Nobody changes onto the line they left towards the stop they came from (T-575), or towards
    /// the one their train goes on to; a line passing `v` again leaves it for other stops, so a rider
    /// off a lollipop's loop may change at its junction onto the line on round (T-1268). A walk from
    /// another station takes the line only away from where they rode (`walk_ban`, T-3295).
    pub(super) fn ban(&self, a: usize, v: usize) -> Option<Bar> {
        let g = self.g;
        let line = g.rn_line[a];
        let own = g.rn_station[a] as usize == v;
        let (back, even) = if own { self.xbar[a] } else { (NONE, NONE) };
        let lb = if self.banned_all[a] { WHOLE } else { ban_of(&self.pst, &self.nst, a) };
        if back != NONE || even != NONE {
            // T-4064: barred beyond its own line's departures, where the strategy boards one
            let b = Bar { line, ban: lb, back, even };
            let s = self.s;
            let hit = (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).any(|e| {
                let r = g.adj_to[e] as usize - s;
                self.board[r >> 6] >> (r & 63) & 1 == 1 && self.bars(r, b)
            });
            return hit.then_some(b);
        }
        if !self.lines.contains(&line) { return None; }
        if !own { return self.walk_ban(a, v, line).map(|(line, ban)| Bar { line, ban, back: NONE, even: NONE }); }
        let boarded = self.banned[self.banned_off[a] as usize..self.banned_off[a + 1] as usize].iter().any(|&r| self.board[r as usize >> 6] >> (r & 63) & 1 == 1);
        boarded.then_some(Bar { line, ban: lb, back: NONE, even: NONE })
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

    /// T-4041: after the queue, the lines that joined nodes already settled (`late`). A line can be
    /// first though its cost from boarding is above the station's cost, so it is offered once the
    /// station has left the queue; without it a line slower by more than about half the faster
    /// one's headway would get nobody. Here: riders who came off that line straight into the
    /// station get a change node barring it (as `settle_station` makes them), the settle order is
    /// made again so every node comes after those it sends riders to (a late line whose riders
    /// would come back to it is left out), and the nodes' costs and shares are made with their
    /// late lines. Their costs reach no node settled before (a cost a little high there).
    pub(super) fn late_fix(&mut self, dst: usize) {
        if self.late.is_empty() {
            return;
        }
        let (s, n) = (self.s, self.n);
        let cn0 = self.cn.len();
        // the late lines' own riders: per station, those that came off a line late there
        self.late.sort_unstable();
        let mut i = 0;
        while i < self.late.len() && (self.late[i].0 as usize) < s {
            let v = self.late[i].0 as usize;
            let mut j = i;
            while j < self.late.len() && self.late[j].0 as usize == v {
                j += 1;
            }
            self.change_off_late(v, i, j);
            i = j;
        }
        self.late.sort_unstable();
        self.late_out.clear();
        self.late_out.resize(self.late.len(), false);
        self.settle_order(dst);
        // costs and shares where lines joined late, without those left out (a node made before
        // whose late lines were all left out keeps the costs and shares it was settled with)
        let mut k = 0;
        while k < self.late.len() {
            let x = self.late[k].0;
            let k0 = k;
            while k < self.late.len() && self.late[k].0 == x {
                k += 1;
            }
            if (x as usize) < n + cn0 && self.late_out[k0..k].iter().all(|&o| o) {
                continue;
            }
            let x = x as usize;
            if x < s {
                self.list_by_edge_kept(self.st_head[x], |q, r| q.board[r >> 6] >> (r & 63) & 1 == 1);
                self.dist[x] = self.note_shares(x, x);
            } else {
                let c = x - n;
                let (v, b0, head) = (self.cn[c].st as usize, self.cn[c].bits as usize, self.cn[c].head);
                let on = move |q: &Search, r: usize| {
                    let e = q.edge_ix[r] as usize;
                    q.cn_bits[b0 + e / 64] >> (e % 64) & 1 == 1
                };
                self.list_by_edge_kept(head, on);
                self.cn[c].n = self.pl.len() as u32;
                self.cn[c].dist = self.note_shares(x, v);
            }
        }
    }

    /// `late_fix` at station `v`, its late lines `late[i..j]`: the arrivals that came straight in
    /// off one of their lines, and would be barred it at a change node now (`ban`), to a change
    /// node of the station's departures less the barred, its late lines late there too.
    fn change_off_late(&mut self, v: usize, i: usize, j: usize) {
        let (g, s, n) = (self.g, self.s, self.n);
        let (e0, e1) = (g.adj_off[v] as usize, g.adj_off[v + 1] as usize);
        self.lines.clear();
        for e in e0..e1 {
            let r = g.adj_to[e] as usize - s;
            if self.board[r >> 6] >> (r & 63) & 1 == 1 && !self.lines.contains(&g.rn_line[r]) {
                self.lines.push(g.rn_line[r]);
            }
        }
        self.keys.clear();
        self.feed.clear();
        for jj in g.rev_off[v] as usize..g.rev_off[v + 1] as usize {
            let from = g.rev_from[jj] as usize;
            if from < s || self.next[from] as usize != v {
                continue;
            }
            let line = g.rn_line[from - s];
            if !self.late[i..j].iter().any(|x| g.rn_line[x.1 as usize] == line) {
                continue;
            }
            match self.ban(from - s, v) {
                Some(Bar { line: l, ban: WHOLE, .. }) if l == self.lone[v] => {}
                Some(key) => {
                    let k = self.keys.iter().position(|&x| x == key).unwrap_or_else(|| {
                        self.keys.push(key);
                        self.keys.len() - 1
                    });
                    self.feed.push((k as u32, from as u32));
                }
                None => {}
            }
        }
        for ki in 0..self.keys.len() {
            let bar = self.keys[ki];
            let k = self.cn.len();
            let b0 = self.cn_bits.len();
            self.cn_bits.resize(b0 + (e1 - e0).div_ceil(64), 0);
            let mut c = Change { st: v as u32, line: bar.line, ban: bar.ban, back: bar.back, even: bar.even, cut: f32::INFINITY, n: 0, head: NONE, dist: f32::INFINITY, done: true, feed: (0, 0), bits: b0 as u32 };
            let mut at = self.st_head[v];
            while at != NONE {
                let (d, f, r, nx) = self.lines_at[at as usize];
                at = nx;
                let ru = r as usize;
                if self.board[ru >> 6] >> (ru & 63) & 1 == 0 || self.bars(ru, bar) {
                    continue;
                }
                let e = self.edge_ix[ru] as usize;
                self.cn_bits[b0 + e / 64] |= 1 << (e % 64);
                c.n += 1;
                self.lines_at.push((d, f, r, c.head));
                c.head = (self.lines_at.len() - 1) as u32;
            }
            if c.n == 0 {
                self.cn_bits.truncate(b0);
                continue;
            }
            self.cn.push(c);
            // its costs and shares are made with the late nodes'; its late lines can be left out
            let id = (n + k) as u32;
            let mut any = false;
            for x in i..j {
                let r = self.late[x].1 as usize;
                let e = self.edge_ix[r] as usize;
                if self.cn_bits[b0 + e / 64] >> (e % 64) & 1 == 1 {
                    self.late.push((id, r as u32));
                    any = true;
                }
            }
            if !any {
                // (no late line of its own: still its costs and shares)
                self.late.push((id, NONE));
            }
            for f in 0..self.feed.len() {
                if self.feed[f].0 == ki as u32 {
                    self.next[self.feed[f].1 as usize] = id;
                }
            }
        }
    }

    /// `list_by_edge`, the lines whose departure `keep` says is still accepted.
    fn list_by_edge_kept(&mut self, head: u32, keep: impl Fn(&Search, usize) -> bool) {
        self.by_edge.clear();
        let mut i = head;
        while i != NONE {
            let (d, f, r, nx) = self.lines_at[i as usize];
            if keep(self, r as usize) {
                self.by_edge.push((self.edge_ix[r as usize], d, f, r));
            }
            i = nx;
        }
        self.by_edge_into_pl();
    }

    /// `late_fix`'s settle order: each node after every node it sends riders to (a route node its
    /// next; a station or change node its departures, and where a boarding rides a stop first,
    /// that stop), by a depth-first walk from the nodes in the order they were settled. A late
    /// line found on a loop is left out (`late_out`, its departure's bit cleared), and the walk
    /// goes back to before it. A node is placed at once where nothing it sends riders to is left
    /// to place (T-4054): a route node whose next is placed, and between walks, a station or change
    /// node no line joined late (it sends riders only to nodes settled before it, placed by then).
    fn settle_order(&mut self, dst: usize) {
        let (s, n) = (self.s, self.n);
        let ids = n + self.cn.len();
        self.mark.clear();
        self.mark.resize(ids, 0);
        self.late_at.clear();
        self.late_at.resize(ids, NONE);
        for k in (0..self.late.len()).rev() {
            self.late_at[self.late[k].0 as usize] = k as u32;
        }
        self.reorder.clear();
        self.stack.clear();
        self.succ.clear();
        for o in 0..self.order.len() {
            let root = self.order[o] as usize;
            if self.mark[root] != 0 {
                continue;
            }
            let at_once = if s <= root && root < n { self.placed_next(root) } else { self.late_at[root] == NONE };
            if at_once {
                self.mark[root] = 2;
                self.reorder.push(root as u32);
                continue;
            }
            self.enter(root, NONE, dst);
            while let Some(&(_, _, pos, end, _)) = self.stack.last() {
                if pos == end {
                    let (x, start, _, _, _) = self.stack.pop().unwrap();
                    self.succ.truncate(start as usize);
                    self.mark[x as usize] = 2;
                    self.reorder.push(x);
                    continue;
                }
                self.stack.last_mut().unwrap().2 += 1;
                let (y, li) = self.succ[pos as usize];
                if li != NONE && self.late_out[li as usize] {
                    continue;
                }
                match self.mark[y as usize] {
                    0 if self.placed_next(y as usize) => {
                        self.mark[y as usize] = 2;
                        self.reorder.push(y);
                    }
                    0 => self.enter(y as usize, li, dst),
                    2 => {}
                    _ => {
                        // a loop: back to before its last late line
                        let out = if li != NONE {
                            Some((li, self.stack.len()))
                        } else {
                            let from = self.stack.iter().rposition(|f| f.0 == y).unwrap();
                            (from + 1..self.stack.len()).rev().find(|&f| self.stack[f].4 != NONE).map(|f| (self.stack[f].4, f))
                        };
                        let Some((li, f)) = out else {
                            debug_assert!(false, "T-4041: a loop with no late line");
                            continue;
                        };
                        self.drop_late(li as usize);
                        while self.stack.len() > f {
                            let (x, start, _, _, _) = self.stack.pop().unwrap();
                            self.succ.truncate(start as usize);
                            self.mark[x as usize] = 0;
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut self.order, &mut self.reorder);
    }

    /// `settle_order`: whether `x` is a route node whose next node is placed (or that has none).
    #[inline]
    fn placed_next(&self, x: usize) -> bool {
        if x < self.s || x >= self.n {
            return false;
        }
        let nx = self.next[x];
        nx == NONE || self.mark[nx as usize] == 2
    }

    /// `settle_order`: node `x`, reached by late line `li` (or `NONE`), onto the walk's stack with
    /// the nodes it sends riders to.
    fn enter(&mut self, x: usize, li: u32, dst: usize) {
        let (g, s, n) = (self.g, self.s, self.n);
        self.mark[x] = 1;
        let start = self.succ.len() as u32;
        if s <= x && x < n {
            let nx = self.next[x];
            if nx != NONE {
                self.succ.push((nx, NONE));
            }
        } else if x != dst {
            let (v, b0) = if x < s { (x, NONE) } else { (self.cn[x - n].st as usize, self.cn[x - n].bits) };
            let e0 = g.adj_off[v] as usize;
            let l0 = self.late_at[x];
            for e in e0..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                let on = if b0 == NONE { self.board[r >> 6] >> (r & 63) & 1 == 1 } else { self.cn_bits[b0 as usize + (e - e0) / 64] >> ((e - e0) % 64) & 1 == 1 };
                if !on {
                    continue;
                }
                // (`x`'s late lines run from `late_at[x]` in `late`, sorted; a node placed already
                // is passed over whatever line leads to it, so it is left off)
                let mut li = NONE;
                if l0 != NONE {
                    let mut k = l0 as usize;
                    while k < self.late.len() && self.late[k].0 as usize == x {
                        if self.late[k].1 as usize == r {
                            li = k as u32;
                            break;
                        }
                        k += 1;
                    }
                }
                if self.mark[s + r] != 2 {
                    self.succ.push(((s + r) as u32, li));
                }
                let nx = self.next[s + r] as usize;
                if !(nx >= s && nx < n) && self.mark[g.rn_next[r] as usize] != 2 {
                    self.succ.push((g.rn_next[r], li));
                }
            }
        }
        let end = self.succ.len() as u32;
        self.stack.push((x as u32, start, start, end, li));
    }

    /// Late line `late[li]` left out: its departure no longer accepted there.
    fn drop_late(&mut self, li: usize) {
        let (s, n) = (self.s, self.n);
        self.late_out[li] = true;
        let (x, r) = (self.late[li].0 as usize, self.late[li].1 as usize);
        if x < s {
            self.board[r >> 6] &= !(1 << (r & 63));
            self.joined[x] -= 1;
            if self.next[x] as usize == s + r {
                // (a station's next is one of its departures)
                let g = self.g;
                self.next[x] = (g.adj_off[x] as usize..g.adj_off[x + 1] as usize).map(|e| g.adj_to[e]).find(|&to| {
                    let q = to as usize - s;
                    self.board[q >> 6] >> (q & 63) & 1 == 1
                }).unwrap_or(NONE);
            }
        } else {
            let (b0, e) = (self.cn[x - n].bits as usize, self.edge_ix[r] as usize);
            self.cn_bits[b0 + e / 64] &= !(1 << (e % 64));
            self.cn[x - n].n -= 1;
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
            // by the departures' shares (`note_station`)
            self.pool.clear();
            let mut j = 0;
            for e in e0..e1 {
                let r = g.adj_to[e] as usize - s;
                if self.board[r >> 6] >> (r & 63) & 1 == 1 {
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], self.sh[j]));
                    j += 1;
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
                Some(Bar { line, ban: WHOLE, .. }) if line == lone => {}
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
                let bar = self.keys[ki];
                let f0 = self.cn_feed.len();
                self.cn_feed.extend(self.feed.iter().filter(|f| f.0 == ki as u32).map(|f| f.1));
                if self.cn_feed.len() == f0 { continue; }
                self.cand.clear();
                for e in e0..e1 {
                    let r = g.adj_to[e] as usize - s;
                    if !self.bars(r, bar) && self.offer[r].is_finite() {
                        self.cand.push((self.offer[r], r as u32));
                    }
                }
                if self.cand.len() > 1 {
                    self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                }
                let k = self.cn.len();
                let b0 = self.cn_bits.len();
                self.cn_bits.resize(b0 + (e1 - e0).div_ceil(64), 0);
                // T-4041: every line offered so far that can be first (all settled already)
                let mut c = Change { st: v as u32, line: bar.line, ban: bar.ban, back: bar.back, even: bar.even, cut: f32::INFINITY, n: 0, head: NONE, dist: f32::INFINITY, done: false, feed: (f0 as u32, self.cn_feed.len() as u32), bits: b0 as u32 };
                // (the cost does not depend on the lines' order: `pool_cost`)
                self.pl.clear();
                self.pk.clear();
                for &(od, r) in &self.cand {
                    if !(od < c.cut) {
                        break;
                    }
                    let f = g.rn_tph[r as usize];
                    c.cut = c.cut.min(od + wait_spread(f, g.st_wait[v], false));
                    let i = self.edge_ix[r as usize] as usize;
                    self.cn_bits[b0 + i / 64] |= 1 << (i % 64);
                    self.pl.push((od, f));
                    self.pk.push(r);
                    self.lines_at.push((od, f, r, c.head));
                    c.head = (self.lines_at.len() - 1) as u32;
                }
                c.n = self.pl.len() as u32;
                self.cn.push(c);
                if !self.pl.is_empty() {
                    let cost = if self.pl.len() == 1 { self.one_cost(v, self.cand[0].1 as usize, self.cand[0].0) } else { self.pool_at(v) };
                    self.cn[k].dist = cost;
                    self.heap.push(cost, (self.n + self.nr + k) as u32);
                }
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
        if self.cn[k].n > 1 {
            self.list_by_edge(self.cn[k].head);
            let _ = self.note_shares(id as usize, v);
        } else {
            self.sh.clear();
            self.sh.resize(self.cn[k].n as usize, 1.0);
        }
        let spread = g.st_spread[v];
        if spread {
            self.pool.clear();
            let (s, e0, b0) = (self.s, g.adj_off[v] as usize, self.cn[k].bits as usize);
            let mut j = 0;
            for e in e0..g.adj_off[v + 1] as usize {
                let i = e - e0;
                if self.cn_bits[b0 + i / 64] >> (i % 64) & 1 == 1 {
                    let r = g.adj_to[e] as usize - s;
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], self.sh[j]));
                    j += 1;
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
        let one = self.cn.iter().enumerate().all(|(k, c)| c.bits as usize == k) && self.cn_bits.len() == self.cn.len();
        ChangeNodes {
            st: Ix::of(self.cn.iter().map(|c| c.st)),
            off: if one { Box::new([]) } else { self.cn.iter().map(|c| c.bits).chain([self.cn_bits.len() as u32]).collect() },
            bits: self.cn_bits.as_slice().into(),
        }
    }
}

/// A line's planned spread is its plain one (a headway of 15 minutes or less).
#[inline(always)]
fn same_spread(f: f32, wait_f: f32) -> bool {
    wait_spread(f, wait_f, true) == wait_spread(f, wait_f, false)
}

/// T-4041: how far a wait for a line with `f` departures an hour spreads, perceived minutes: its
/// headway times the station's wait factor, or with `planned` the headway whose half is the
/// planned first wait (`planned_board_cost`), so one line costs what `board_cost` or
/// `planned_board_cost` says.
#[inline(always)]
pub fn wait_spread(f: f32, wait_f: f32, planned: bool) -> f32 {
    let h = 60.0 / f;
    wait_f * if planned { h.min(2.0 * PLANNED_EARLY_MIN + h / 3.0).min(2.0 * PLANNED_WAIT_CAP_MIN) } else { h }
}

/// T-4041 (Anita 11-22: "some people would probably just take the first train that comes ... take
/// us like 75% of the way"): the riders at a station who take the line that gets them to their
/// stop first; the rest board the first train that comes among the lines that can be first,
/// unless it is sure to arrive after the latest the best line would (`pool_cost`).
pub const FIRST_ARRIVAL_SHARE: f32 = 0.75;

/// Gauss-Legendre nodes and weights on [-1, 1], 1 and 2 points (exact to degree 3). Two at most
/// (T-4054, DECISIONS 2026-11-23): pools of up to three lines that can be first are integrated
/// exactly, larger ones nearly.
const GAUSS: [&[(f64, f64)]; 2] = [
    &[(0.0, 2.0)],
    &[(-0.577_350_269_189_625_8, 1.0), (0.577_350_269_189_625_8, 1.0)],
];

/// Scratch for `pool_cost`: per line, (earliest arrival, latest, 1 / spread, place in `lines`).
#[derive(Default)]
pub(super) struct PoolWork {
    v: Vec<(f64, f64, f64, usize)>,
    acc: Vec<f64>,
    /// the first-train riders' part: per line how far into the wait its departure counts, the
    /// same sorted, and each line's chance
    ej: Vec<f64>,
    e: Vec<(f64, usize)>,
    lz: Vec<f64>,
}

/// T-4041 (Anita 11-22: "evenly distribute the passengers by when they arrive"): a rider who turns
/// up at a station at a random time takes whichever of its accepted `lines` gets them to their stop
/// first. Each line, (cost from boarding, departures an hour), has its wait uniform over its spread
/// (`wait_spread`) and independent of the others'. A line whose cost from boarding is at least
/// another's plus that one's spread is never first and is left out (share 0); the least such sum
/// is the latest anyone arrives. `FIRST_ARRIVAL_SHARE` of the riders take the line that arrives
/// first, E[min(d_i + U(0, spread_i))]; the others board the first departure that can still
/// arrive by that latest time. The cost is the two groups' expected arrival, weighed, plus the
/// change's fixed minutes, so it can rise when a slow line joins; with `share` (as long as
/// `lines`) each line's share of the riders, and then also the expected perceived wait (else 0),
/// so the cost is the shares' rides plus the wait. One line costs `board_cost` (or
/// `planned_board_cost`) exactly. The result does not depend on the order of `lines`.
#[allow(clippy::too_many_arguments)]
pub(super) fn pool_cost(lines: &[(f32, f32)], wait_f: f32, change_f: f32, planned: bool, p: &Params, w: &mut PoolWork, share: Option<&mut [f32]>) -> (f32, f32) {
    if lines.len() == 1 {
        let (d, f) = lines[0];
        let h = 60.0 / f;
        let c = if planned { planned_board_cost(h, wait_f, change_f, p) } else { board_cost(h, wait_f, change_f, p) };
        if let Some(sh) = share {
            sh[0] = 1.0;
        }
        return (d + c, c - p.change_at(change_f));
    }
    if lines.len() == 2 {
        return pool_two(lines, wait_f, change_f, planned, p, share);
    }
    w.v.clear();
    for (i, &(d, f)) in lines.iter().enumerate() {
        let sp = (wait_spread(f, wait_f, planned) as f64).max(1e-9);
        w.v.push((d as f64, d as f64 + sp, 1.0 / sp, i));
    }
    // by cost from boarding, then spread, then place, so equal lines come out equal whatever order
    w.v.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(y.2.total_cmp(&x.2)).then(x.3.cmp(&y.3)));
    let last = w.v.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let m = w.v.iter().take_while(|x| x.0 < last).count();
    let want = share.is_some();
    w.acc.clear();
    w.acc.resize(if want { m } else { 0 }, 0.0);
    let mut tail = 0.0f64;
    // each quadrature's nodes and weights on its stretch: the nodes are gone through together,
    // line by line, so their products run side by side
    let (mut nt, mut nq) = ([0f64; 2], [0f64; 2]);
    // between one line's earliest arrival and the next's, the first k + 1 lines can be first
    for k in 0..m {
        let t0 = w.v[k].0;
        let t1 = if k + 1 < m { w.v[k + 1].0 } else { last };
        let len = t1 - t0;
        if len <= 0.0 {
            continue;
        }
        let nn = nodes_on(GAUSS[((k + 1) / 2).min(GAUSS.len() - 1)], t0, len, &mut nt, &mut nq);
        let mut surv = [1.0f64; 2];
        for l in &w.v[..=k] {
            for i in 0..nn {
                surv[i] *= (l.1 - nt[i]) * l.2;
            }
        }
        for i in 0..nn {
            surv[i] *= nq[i];
            tail += surv[i];
        }
        if want {
            for (j, l) in w.v[..=k].iter().enumerate() {
                let mut a = w.acc[j];
                for i in 0..nn {
                    let own = (l.1 - nt[i]) * l.2;
                    if own > 0.0 {
                        a += surv[i] / own * l.2;
                    }
                }
                w.acc[j] = a;
            }
        }
    }
    let first = w.v[0].0 + tail;
    // the riders who board the first train that comes (`FIRST_ARRIVAL_SHARE`): line j's
    // departure counts while it can still arrive by `last`, so up to e_j = min(spread, last - d_j)
    // into the wait; the first such departure wins. Between the e_j's, the lines whose e_j is
    // still ahead are uniform factors, the others constant.
    w.ej.clear();
    w.ej.extend(w.v[..m].iter().map(|l| (1.0 / l.2).min(last - l.0)));
    w.e.clear();
    w.e.extend(w.ej.iter().enumerate().map(|(j, &e)| (e, j)));
    w.e.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    w.lz.clear();
    w.lz.resize(m, 0.0);
    let mut lazy_wait = 0.0f64;
    let mut t0 = 0.0f64;
    for k in 0..m {
        let t1 = w.e[k].0;
        let len = t1 - t0;
        if len > 0.0 {
            let nn = nodes_on(GAUSS[((m - k + 1) / 2).min(GAUSS.len() - 1)], t0, len, &mut nt, &mut nq);
            let mut surv = [1.0f64; 2];
            for (j, l) in w.v[..m].iter().enumerate() {
                for i in 0..nn {
                    surv[i] *= 1.0 - nt[i].min(w.ej[j]) * l.2;
                }
            }
            for i in 0..nn {
                surv[i] *= nq[i];
                lazy_wait += surv[i];
            }
            for &(_, j) in &w.e[k..] {
                let l = w.v[j];
                let mut a = w.lz[j];
                for i in 0..nn {
                    a += surv[i] / (1.0 - nt[i] * l.2) * l.2;
                }
                w.lz[j] = a;
            }
        }
        t0 = t1;
        // a line whose whole spread counts ends every wait
        if w.e[k].0 * w.v[w.e[k].1].2 >= 1.0 - 1e-12 {
            break;
        }
    }
    let a = FIRST_ARRIVAL_SHARE as f64;
    let lz_total: f64 = w.lz.iter().sum();
    let lazy_ride: f64 = w.v[..m].iter().enumerate().map(|(j, l)| w.lz[j] / lz_total * l.0).sum();
    let cost = a * first + (1.0 - a) * (lazy_ride + lazy_wait);
    let mut wait = 0.0f64;
    if let Some(sh) = share {
        let total: f64 = w.acc.iter().sum();
        let mut ride = 0.0f64;
        sh.fill(0.0);
        for (j, l) in w.v[..m].iter().enumerate() {
            ride += w.acc[j] / total * l.0;
            sh[l.3] = (a * w.acc[j] / total + (1.0 - a) * w.lz[j] / lz_total) as f32;
        }
        wait = (a * (first - ride) + (1.0 - a) * lazy_wait).max(0.0);
    }
    (cost as f32 + p.change_at(change_f), wait as f32)
}

/// `pool_cost`'s quadrature on [t0, t0 + len]: its nodes into `t` and weights into `q`; how many.
#[inline(always)]
fn nodes_on(nodes: &[(f64, f64)], t0: f64, len: f64, t: &mut [f64; 2], q: &mut [f64; 2]) -> usize {
    for (i, &(x, wt)) in nodes.iter().enumerate() {
        t[i] = t0 + (x + 1.0) * len / 2.0;
        q[i] = wt * len / 2.0;
    }
    nodes.len()
}

/// `pool_cost` for two lines, in closed form (most stations and change nodes that take more than
/// one): the earlier line alone until the later's earliest arrival, then both.
fn pool_two(lines: &[(f32, f32)], wait_f: f32, change_f: f32, planned: bool, p: &Params, share: Option<&mut [f32]>) -> (f32, f32) {
    let line = |i: usize| {
        let (d, f) = lines[i];
        (d as f64, (wait_spread(f, wait_f, planned) as f64).max(1e-9))
    };
    let (x, y) = (line(0), line(1));
    // by cost from boarding, then spread, then place, as `pool_cost` sorts them
    let swap = y.0 < x.0 || (y.0 == x.0 && y.1 < x.1);
    let ((a1, h1), (a2, h2)) = if swap { (y, x) } else { (x, y) };
    let (b1, b2) = (a1 + h1, a2 + h2);
    let a = FIRST_ARRIVAL_SHARE as f64;
    let (cost, later, wait) = if a2 >= b1 {
        (a1 + h1 / 2.0, 0.0, h1 / 2.0)
    } else {
        let end = b1.min(b2);
        // to a2 the earlier line alone: its survival (b1 - t) / h1; then both, (b1 - t)(b2 - t) / h1 h2
        let alone = ((b1 - a1) * (b1 - a1) - (b1 - a2) * (b1 - a2)) / (2.0 * h1);
        let (u0, u1) = (b1 - a2, b1 - end);
        let k = b2 - b1;
        // ∫ u (u + k) du over u from b1 - end to b1 - a2
        let both = ((u0 * u0 * u0 - u1 * u1 * u1) / 3.0 + k * (u0 * u0 - u1 * u1) / 2.0) / (h1 * h2);
        let first = a1 + alone + both;
        let s2 = (u0 * u0 - u1 * u1) / (2.0 * h1 * h2);
        // the first-train riders: line i's departure counts up to e_i = end - a_i into the wait
        // (e2 <= e1), so the later line is taken when it comes first within e2
        let (e1, e2) = (end - a1, end - a2);
        let l2 = (e2 - e2 * e2 / (2.0 * h1)) / h2;
        let lw = e2 - e2 * e2 * (1.0 / h1 + 1.0 / h2) / 2.0 + e2 * e2 * e2 / (3.0 * h1 * h2) + (1.0 - e2 / h2) * ((e1 - e2) - (e1 * e1 - e2 * e2) / (2.0 * h1));
        let lazy = (1.0 - l2) * a1 + l2 * a2 + lw;
        let later = a * s2 + (1.0 - a) * l2;
        let wait = a * (first - (1.0 - s2) * a1 - s2 * a2) + (1.0 - a) * lw;
        (a * first + (1.0 - a) * lazy, later, wait)
    };
    let mut out_wait = 0.0f64;
    if let Some(sh) = share {
        let (i1, i2) = if swap { (1, 0) } else { (0, 1) };
        sh[i1] = (1.0 - later) as f32;
        sh[i2] = later as f32;
        out_wait = wait.max(0.0);
    }
    (cost as f32 + p.change_at(change_f), out_wait as f32)
}

/// T-4041: the departures' shares at the nodes of a destination's strategy that accept two or
/// more (stations; change nodes from `Graph::n`): each node's first value in `val`, by node,
/// ascending. There the real expected wait (minutes), then each accepted departure's share in edge
/// order. A node accepting one departure is not listed: its share is 1, its wait half its headway.
/// T-4179: the nodes in `node`, each's first share at `off` in `val` and its wait in `wait`. A skims table
/// keeps no waits (`without_waits`, a ninth of the table on the real New York save): only a search's own
/// readers (`walk_column`) and the trip explorer (`trip::routes`, which searches its destination again) read them.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Shares {
    pub node: Ix,
    pub off: Ix,
    pub val: Box<[f32]>,
    pub wait: Box<[f32]>,
}

impl Shares {
    /// Where node `v`'s first share is in `val`, if it has shares.
    #[inline]
    pub fn first(&self, v: usize) -> Option<usize> {
        self.node.find(v).map(|i| self.off.get(i))
    }
    pub fn bytes(&self) -> usize {
        self.node.bytes() + self.off.bytes() + 4 * (self.val.len() + self.wait.len())
    }
    pub fn is_empty(&self) -> bool {
        self.node.is_empty()
    }
    pub fn without_waits(mut self) -> Shares {
        self.wait = Box::new([]);
        self
    }
    /// From the entries as a search notes them (`SharesBuild`): node, then its wait and shares from that offset.
    fn of(at: &[(u32, u32)], inline: &[f32]) -> Shares {
        let mut val = Vec::with_capacity(inline.len() - at.len());
        let mut off = Vec::with_capacity(at.len());
        let mut wait = Vec::with_capacity(at.len());
        for (k, &(_, i)) in at.iter().enumerate() {
            let end = at.get(k + 1).map_or(inline.len(), |x| x.1 as usize);
            wait.push(inline[i as usize]);
            off.push(val.len() as u32);
            val.extend_from_slice(&inline[i as usize + 1..end]);
        }
        Shares { node: Ix::of(at.iter().map(|x| x.0)), off: Ix::of(off.into_iter()), val: val.into(), wait: wait.into() }
    }
    /// The real expected wait at node `v`, its accepted departures `dep` (`departures`).
    #[inline]
    pub fn wait(&self, g: &Graph, v: usize, dep: &[(u32, f32)]) -> f32 {
        match dep {
            [] => f32::INFINITY,
            [(e, _)] => 30.0 / g.rn_tph[g.adj_to[*e as usize] as usize - g.n_st],
            _ => self.node.find(v).map_or_else(|| 30.0 / dep.iter().map(|&(e, _)| g.rn_tph[g.adj_to[e as usize] as usize - g.n_st]).sum::<f32>(), |i| self.wait[i]),
        }
    }
}

/// `Shares` as a search puts them together, in settle order (a node again after its late lines).
#[derive(Default)]
pub(super) struct SharesBuild {
    at: Vec<(u32, u32)>,
    val: Vec<f32>,
    /// `finish`'s scratch: per node its last entry in `at`
    last: Vec<u32>,
}

impl SharesBuild {
    pub(super) fn clear(&mut self) {
        self.at.clear();
        self.val.clear();
    }
    pub(super) fn push(&mut self, v: usize, wait: f32, share: &[f32]) {
        self.at.push((v as u32, self.val.len() as u32));
        self.val.push(wait);
        self.val.extend_from_slice(share);
    }
    pub(super) fn finish(&mut self) -> Shares {
        if self.at.windows(2).all(|x| x[0].0 < x[1].0) {
            return Shares::of(&self.at, &self.val);
        }
        // each entry's values run to the next entry's, in push order; a node noted again (its late
        // lines, `Search::late_fix`) keeps its last
        let n = self.at.len();
        let ids = self.at.iter().map(|x| x.0 as usize + 1).max().unwrap_or(0);
        self.last.clear();
        self.last.resize(ids, NONE);
        for (k, x) in self.at.iter().enumerate() {
            self.last[x.0 as usize] = k as u32;
        }
        let mut val = Vec::with_capacity(self.val.len());
        let mut out = Vec::with_capacity(n);
        for (v, &k) in self.last.iter().enumerate() {
            if k == NONE {
                continue;
            }
            let k = k as usize;
            let (i, end) = (self.at[k].1 as usize, if k + 1 < n { self.at[k + 1].1 as usize } else { self.val.len() });
            out.push((v as u32, val.len() as u32));
            val.extend_from_slice(&self.val[i..end]);
        }
        Shares::of(&out, &val)
    }
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

/// The stations a commute can alight at: the walks of subzones with jobs (trips to work), and the
/// walks and rides of subzones with homes (trips home, which end where the trip out started,
/// T-3916). Nobody's rail trip ends anywhere else, so the strategy searches towards other stations
/// are not needed.
pub fn egress_stations(sz: &Subzones, n_st: usize) -> Vec<bool> {
    let mut want = vec![false; n_st];
    for v in 0..sz.n {
        let r = if sz.pop[v] > 0.0 { sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize } else if sz.jobs[v] > 0.0 { sz.walk(v) } else { 0..0 };
        for k in r {
            want[sz.acc_st[k] as usize] = true;
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
    /// T-3784: the transfer fare alone, which zone runs take off the route costs
    pub(super) xfer_fare: f32,
    pub(super) st_wait: Vec<f32>,
    pub(super) st_bld: Vec<bool>,
    pub(super) fare_curve: FareCurve,
    pub(super) rn_km: Vec<f32>,
    pub(super) rn_kmw: Vec<f32>,
    pub(super) rn_base_mul: Vec<f32>,
    pub(super) base_min: f32,
    pub(super) st_walk_to: Vec<u32>,
    pub(super) st_walk_min: Vec<f32>,
    pub(super) rn_in: Vec<f32>,
    pub(super) rn_out: Vec<f32>,
    pub(super) rn_even: Vec<u32>,
    pub(super) rn_gap: Vec<f32>,
}

impl SkimKey {
    pub(super) fn of(g: &Graph, p: &Params) -> SkimKey {
        SkimKey { n_st: g.n_st, n: g.n, adj_off: g.adj_off.clone(), adj_to: g.adj_to.clone(), adj_cost: g.adj_cost.clone(), rn_tph: g.rn_tph.clone(), rn_left: g.rn_left.clone(), rn_dx: g.rn_dx.clone(), rn_dy: g.rn_dy.clone(), transfer_min: p.change_min(), xfer_fare: g.xfer_fare, st_wait: g.st_wait.clone(), st_bld: g.st_bld.clone(), fare_curve: g.fare_curve.clone(), rn_km: g.rn_km.clone(), rn_kmw: g.rn_kmw.clone(), rn_base_mul: g.rn_base_mul.clone(), base_min: g.base_min, st_walk_to: g.st_walk_to.clone(), st_walk_min: g.st_walk_min.clone(), rn_in: g.rn_in.clone(), rn_out: g.rn_out.clone(), rn_even: g.rn_even.clone(), rn_gap: g.rn_gap.clone() }
    }
    pub(super) fn same(&self, g: &Graph, p: &Params) -> bool {
        self.n_st == g.n_st
            && self.n == g.n
            && self.st_walk_to == g.st_walk_to
            && self.st_walk_min == g.st_walk_min
            && self.transfer_min == p.change_min()
            && self.xfer_fare.to_bits() == g.xfer_fare.to_bits()
            && self.st_wait == g.st_wait
            && self.st_bld == g.st_bld
            && self.fare_curve == g.fare_curve
            && self.rn_km == g.rn_km
            && self.rn_kmw == g.rn_kmw
            && self.rn_base_mul == g.rn_base_mul
            && self.base_min.to_bits() == g.base_min.to_bits()
            && self.rn_dx == g.rn_dx
            && self.rn_dy == g.rn_dy
            && self.adj_off == g.adj_off
            && self.adj_to == g.adj_to
            && self.adj_cost == g.adj_cost
            && self.rn_tph == g.rn_tph
            && self.rn_left == g.rn_left
            && self.rn_in == g.rn_in
            && self.rn_out == g.rn_out
            && self.rn_even == g.rn_even
            && self.rn_gap == g.rn_gap
    }
}

/// Skims tables a thread keeps (`with_skims`). Each holds stations² route costs plus about 11
/// bytes a node per searched destination (u16 ids, change nodes, shares): on the real New York
/// save 48 MB for New York's table (T-398, T-4179).
pub(super) const SKIM_TABLES: usize = 2;

/// A thread's skims (`with_skims`): `kept`, the graphs solved most recently, newest last (T-119);
/// `spare`, the last table searched for one call alone, for the next caller if it brings the same
/// graph (a zone's everyday right after its commutes), dropped by any other caller before it
/// searches, so it never sits beside a new table (T-4123).
struct SkimCache {
    kept: Vec<(SkimKey, Skims)>,
    spare: Option<(SkimKey, Skims)>,
}

thread_local! {
    static SKIM_CACHE: std::cell::RefCell<SkimCache> = const { std::cell::RefCell::new(SkimCache { kept: Vec::new(), spare: None }) };
    static MARKETS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One live market (`IntercityApi`) on this thread, while held. A table `with_skims` makes with none
/// live leaves out the inter-city starts (`Skims::entries`), which only a market reads.
pub struct MarketLive(());

impl MarketLive {
    pub fn hold() -> MarketLive {
        MARKETS.with(|m| m.set(m.get() + 1));
        MarketLive(())
    }
}

impl Drop for MarketLive {
    fn drop(&mut self) {
        MARKETS.with(|m| m.set(m.get() - 1));
    }
}

/// Whether the cached table `sk` of a graph serves this thread's callers: it has the inter-city
/// starts, or no market is live to read them.
fn serves(sk: &Skims) -> bool {
    sk.entries || MARKETS.with(|m| m.get()) == 0
}

/// A new table for `with_skims`, with the inter-city starts only while a market is live.
fn cache_table(g: &Graph) -> Skims {
    let mut sk = Skims::empty(g);
    sk.entries = MARKETS.with(|m| m.get()) > 0;
    sk
}

/// Run `f` on the skims of `g` towards the stations in `want`, searching only what no earlier
/// caller on this thread has searched for the same graph. A worker solves several populations
/// (two cities and the longer-trip market) on one service, and a period's graph is the same for
/// all of them once crowding is set, so the searches are done once per period and round, not once
/// per population (T-119). The cities do not always agree to the last metre on where two stations
/// are, so a period can bring a graph per city: `SKIM_TABLES` are kept, and which go is chosen by
/// what they cost to search again, so the order the populations come in does not decide (T-405).
pub fn with_skims<R>(g: &Graph, p: &Params, want: &[bool], f: impl FnOnce(&Skims) -> R) -> R {
    SKIM_CACHE.with(|cell| {
        let mut c = cell.borrow_mut();
        // a table without the inter-city starts goes once a market is live
        c.kept.retain(|(_, sk)| serves(sk));
        let spare = c.spare.take().filter(|(k, sk)| k.same(g, p) && serves(sk));
        match c.kept.iter().position(|(k, _)| k.same(g, p)) {
            Some(i) => {
                let e = c.kept.remove(i);
                c.kept.push(e);
            }
            None => {
                if c.kept.len() >= SKIM_TABLES {
                    // the table cheapest to search again (fewest destinations, the oldest of
                    // equals) gives way, and only to one that will cost at least as much: a
                    // cheaper one is searched for this call alone (and kept as the spare)
                    let todo = (0..g.n_st).filter(|&d| want.get(d).copied().unwrap_or(false)).count();
                    let (i, kept) = c.kept.iter().enumerate().map(|(i, e)| (i, e.1.next.len())).min_by_key(|e| e.1).unwrap();
                    if todo < kept {
                        drop(c);
                        let (k, mut sk) = spare.unwrap_or_else(|| (SkimKey::of(g, p), cache_table(g)));
                        skims_extend(&mut sk, g, p, want);
                        let r = f(&sk);
                        cell.borrow_mut().spare = Some((k, sk));
                        return r;
                    }
                    c.kept.remove(i);
                }
                let e = spare.unwrap_or_else(|| (SkimKey::of(g, p), cache_table(g)));
                c.kept.push(e);
            }
        }
        let sk = &mut c.kept.last_mut().unwrap().1;
        skims_extend(sk, g, p, want);
        f(sk)
    })
}

/// Whether this thread holds skims of `g` (`with_skims`).
pub fn skims_cached(g: &Graph, p: &Params) -> bool {
    SKIM_CACHE.with(|c| c.borrow().kept.iter().any(|(k, sk)| k.same(g, p) && serves(sk)))
}

/// Forget the cached skims (a worker calls this when it has finished a period's stage, so the
/// tables do not sit in memory while it waits).
pub fn clear_skim_cache() {
    SKIM_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.kept.clear();
        c.spare = None;
    });
}

