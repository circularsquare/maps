//! The demand workers' API (T-026, T-021): one city's local demand on the player's network, solved
//! period by period. Each demand worker owns one `DemandApi` (its own copy of the city) and
//! solves the periods it was given; periods are independent, so a pool runs them side by side
//! (SPEC 4.4, 7; notes/T-026.md).
//!
//! Per worker: `new` (parse the pack, rebuild the zone gravity), then for every network snapshot
//! `set_network` (station access and subzones, shared by all periods), then per period `solve`
//! (the free-flow first estimate) and `crowd` (one crowding round, method of successive
//! averages). Results per period are read with `summary`, `seg`, `board`, `alight` and
//! `load_of_crush`, all per route node: line by line, the stops in order (run 0) and then in
//! reverse (run 1), the same order `kernel::build_graph` makes.
//!
//! Commutes only (SPEC 4.1's other trips are not modelled yet). A period carries its share of the
//! day's trips to work and trips home. Mode choice runs home -> work; the trips home are the same
//! station pairs reversed (riders go back the way they came, which is exact for park-and-ride),
//! so a period's rail matrix is `to_work x OD + to_home x OD^T`, scaled from one solve.

use super::clock::now_ms;
use super::days::{day_rate, load_factor, share_out, share_ret, WEEKDAY, WORK};
use super::kernel::*;
use super::network::{Network, PERIODS};
use super::pack::{self, City};
use crate::track::params::{LEVELS, PERIOD_HOURS, PERIOD_LEVEL};
use wasm_bindgen::prelude::*;

/// Share of the day's trips to work that start in each period (order of `PERIOD_HOURS`: 6-10,
/// 10-16, 16-20, 20-24, 0-6). Judgement after US travel-survey commute start times; notes/T-026.md.
pub const TO_WORK: [f32; PERIODS] = [0.70, 0.12, 0.06, 0.04, 0.08];
/// Same for trips home.
pub const TO_HOME: [f32; PERIODS] = [0.02, 0.18, 0.58, 0.16, 0.06];
/// Busiest hour of a period against its mean hour (crowding is judged at the busiest hour).
pub const PEAK_HOUR_FACTOR: [f32; PERIODS] = [1.4, 1.15, 1.4, 1.3, 1.5];
/// A 20 m car: seats and riders at crush load (a New York subway car: ~44 seats, ~160 at crush).
pub const SEATS_PER_CAR: f32 = 44.0;
pub const CRUSH_PER_CAR: f32 = 160.0;
/// The per-city rail propensity (SPEC 4.3). 0 for New York since the fast walk (T-078): rail and
/// driving are judged on perceived time and cost alone, which puts the real network's rail share
/// at 20.5% against ACS's 20.3% with T-090's soft reach and drive (notes/T-090.md).
pub const ASC_RAIL: f32 = 0.0;
/// US value of time: $20/hour, or 3 perceived minutes per dollar (T-067).
pub const FARE_MIN_PER_USD: f32 = 3.0;
/// `flows` leaves out the far cells holding this share of the selection's commuters, the
/// smallest first (they go to the cut totals), to keep the answer small (T-098).
pub const FLOW_CUT: f64 = 0.01;

/// One demand station for snapshot stations within `merge_m` of each other (single linkage), so
/// two lines' platforms built side by side or on top of each other (a station node takes one
/// route's track, so crossing lines need a node each) are one station for access and change
/// trains there without a walk. Two stations next to each other on one line are never merged.
/// Returns the merged positions (means) and the map snapshot station -> demand station.
pub(crate) fn merge_stations(st_xy: &[f32], line_off: &[u32], stops: &[u32], merge_m: f32) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let n = st_xy.len() / 2;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    if merge_m > 0.0 && n > 1 {
        let mut adjacent = std::collections::HashSet::new();
        for l in 0..line_off.len() - 1 {
            for k in line_off[l] as usize + 1..line_off[l + 1] as usize {
                let (a, b) = (stops[k - 1], stops[k]);
                adjacent.insert((a.min(b), a.max(b)));
            }
        }
        // a grid of merge_m cells: candidates are in the 3 x 3 block around a station
        let mut cells: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
        let key = |i: usize| ((st_xy[2 * i] / merge_m).floor() as i32, (st_xy[2 * i + 1] / merge_m).floor() as i32);
        for i in 0..n {
            cells.entry(key(i)).or_default().push(i);
        }
        let m2 = merge_m * merge_m;
        for i in 0..n {
            let (cx, cy) = key(i);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(js) = cells.get(&(cx + dx, cy + dy)) else { continue };
                    for &j in js {
                        if j <= i || adjacent.contains(&(i as u32, j as u32)) {
                            continue;
                        }
                        let d2 = (st_xy[2 * i] - st_xy[2 * j]).powi(2) + (st_xy[2 * i + 1] - st_xy[2 * j + 1]).powi(2);
                        if d2 <= m2 {
                            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                            if a != b {
                                parent[a.max(b)] = a.min(b);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut id = vec![u32::MAX; n];
    let mut map = vec![0u32; n];
    let (mut sx, mut sy, mut cnt) = (vec![], vec![], vec![]);
    for i in 0..n {
        let r = find(&mut parent, i);
        if id[r] == u32::MAX {
            id[r] = sx.len() as u32;
            sx.push(0.0f32);
            sy.push(0.0f32);
            cnt.push(0.0f32);
        }
        let k = id[r] as usize;
        map[i] = k as u32;
        sx[k] += st_xy[2 * i];
        sy[k] += st_xy[2 * i + 1];
        cnt[k] += 1.0;
    }
    for k in 0..sx.len() {
        sx[k] /= cnt[k];
        sy[k] /= cnt[k];
    }
    (sx, sy, map)
}

/// Each demand station's complex (T-161): the snapshot stations' complexes (`complex`, one
/// number per snapshot station, game/complexes.ts with the player's links) carried over to the
/// demand stations they were merged into (`st_map`); two complexes sharing a demand station become
/// one. Numbered by the complex's first demand station. Empty when `complex` does not match.
pub(crate) fn demand_groups(st_map: &[u32], n: usize, complex: &[u32]) -> Vec<u32> {
    if complex.len() != st_map.len() || n == 0 {
        return vec![];
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut first: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (i, &c) in complex.iter().enumerate() {
        let d = st_map[i] as usize;
        let f = *first.entry(c).or_insert(d);
        let (a, b) = (find(&mut parent, f), find(&mut parent, d));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    (0..n).map(|d| find(&mut parent, d) as u32).collect()
}

/// The network as the snapshot packs it (`set_network`).
pub(crate) struct Service {
    /// Demand stations: the snapshot's stations, those within `Params::station_merge_m` of each
    /// other merged into one (`st_map`: snapshot station -> demand station).
    pub(crate) st_x: Vec<f32>,
    pub(crate) st_y: Vec<f32>,
    pub(crate) st_map: Vec<u32>,
    pub(crate) line_off: Vec<u32>,
    pub(crate) stops: Vec<u32>,
    /// Seconds, 6 per stop: per run and level, `(run * 3 + level) * n + k` from the line's
    /// offset x 6. Run 0: stop k to k + 1 (0 at the last); run 1: stop k to k - 1 (0 at the first).
    pub(crate) times: Vec<f32>,
    pub(crate) tph: Vec<[f32; LEVELS]>,
    pub(crate) cars: Vec<f32>,
    /// Each demand station's complex (`demand_groups`), empty if none were given.
    pub(crate) group: Vec<u32>,
}

pub(crate) struct Period {
    pub(crate) net: Network,
    pub(crate) p: Params,
    pub(crate) g: Graph,
    /// Loads averaged over the rounds so far.
    avg: Loads,
    rail: f64,
    walk: f64,
    trips: f64,
    pub(crate) round: usize,
    ms: f64,
    pairs: u64,
    /// Where the rail trips come from and go to, averaged like the loads (T-078's demand views).
    pub(crate) sub: SubOut,
    /// The crowding factors the last pass was solved with, and whether every pass so far used
    /// these same ones (free flow is all ones). Then another pass with them would give exactly
    /// the loads already averaged in, so the round is skipped (T-119).
    fac: Vec<f32>,
    uniform: bool,
    /// Transfers platform to platform (T-176): the slots of `g`, averaged like the loads.
    xi: XferIndex,
    xfer: Vec<f32>,
}

/// A pass's rail trips by place (`kernel::ModeOut`), both directions of the period's commutes
/// counted at their home and work ends.
pub(crate) struct SubOut {
    /// Rail trips by home/work subzone and legacy zero walking-loss arrays (`ModeOut`).
    pub(crate) rail_home: Vec<f32>,
    rail_work: Vec<f32>,
    walk_lost_home: Vec<f32>,
    walk_lost_work: Vec<f32>,
    /// Per access entry (`Subzones::acc_st`): rail trips using that station from that subzone
    /// at the home end, and at the work end.
    acc_home: Vec<f32>,
    acc_work: Vec<f32>,
    /// Rail trips by (home station, work station), S x S, before the trips home are reversed.
    pub(crate) od: Vec<f32>,
    /// Rail trips by the perceived minutes of both walks together (`ModeOut::rail_walksum`).
    walksum: Vec<f32>,
}

impl SubOut {
    fn empty() -> SubOut {
        SubOut { rail_home: vec![], rail_work: vec![], walk_lost_home: vec![], walk_lost_work: vec![], acc_home: vec![], acc_work: vec![], od: vec![], walksum: vec![] }
    }
    fn msa(&mut self, new: &SubOut, round: usize) {
        msa(&mut self.rail_home, &new.rail_home, round);
        msa(&mut self.rail_work, &new.rail_work, round);
        msa(&mut self.walk_lost_home, &new.walk_lost_home, round);
        msa(&mut self.walk_lost_work, &new.walk_lost_work, round);
        msa(&mut self.acc_home, &new.acc_home, round);
        msa(&mut self.acc_work, &new.acc_work, round);
        msa(&mut self.od, &new.od, round);
        msa(&mut self.walksum, &new.walksum, round);
    }
}

/// T-276: `DemandApi::home_k` and `work_k`. With a joint gravity shipped for these zones, each
/// zone's trips kept in the pack over its weight; otherwise the pack's trips over its total weight
/// (every commute stays in the pack).
fn zone_trip_factors(city: &City, z: &Zones, gr: &Gravity, from_pack: bool) -> (Vec<f32>, Vec<f32>) {
    let (tp, tj) = city.totals();
    match city.gravity.as_ref().filter(|g| from_pack && g.home.len() == z.n && g.work.len() == z.n) {
        Some(g) => (
            (0..z.n).map(|i| if z.pop[i] > 0.0 { (g.home[i] / z.pop[i]) as f32 } else { 0.0 }).collect(),
            (0..z.n).map(|i| if z.jobs[i] > 0.0 { (g.work[i] / z.jobs[i]) as f32 } else { 0.0 }).collect(),
        ),
        None => (vec![(gr.total / tp.max(1e-9)) as f32; z.n], vec![(gr.total / tj.max(1e-9)) as f32; z.n]),
    }
}

#[wasm_bindgen]
pub struct DemandApi {
    pub(crate) city: City,
    pub(crate) z: Zones,
    pub(crate) gr: Gravity,
    pub(crate) base: Params,
    /// All-day home -> work trips that would walk with no rail (`Gravity::walk_trips`), and the
    /// same by home zone and by work zone.
    walk_all: f64,
    walk_home_zone: Vec<f64>,
    walk_work_zone: Vec<f64>,
    /// T-276: per zone, the pack's trips a day per unit of home weight (`city.pop`) and of work
    /// weight (`city.jobs`). One number everywhere for a pack's own gravity; with a joint gravity
    /// each zone's share kept in the pack (`PackGravity::home`, `work`): the rest commute to
    /// another pack, in the regional layer.
    home_k: Vec<f32>,
    work_k: Vec<f32>,
    pub(crate) svc: Option<Service>,
    pub(crate) sz: Option<Subzones>,
    /// The stations a trip can end at (`egress_stations`): the only ones strategies are searched to.
    want: Vec<bool>,
    pub(crate) per: Vec<Option<Period>>,
    pub(crate) fare_per_km: f32,
    pub(crate) fare_km: Vec<f32>,
    open_ms: f64,
    /// The complex of each station of the next `set_network` (`set_complexes`, T-161).
    complex_in: Vec<u32>,
}

/// Per period: start hour, end hour, demand level (0 high, 1 medium, 2 low), share of trips to
/// work, share of trips home, peak-hour factor. Six numbers each, from the one definition in
/// `track::params` plus this module's shares.
#[wasm_bindgen]
pub fn demand_periods() -> Vec<f64> {
    (0..PERIODS)
        .flat_map(|q| {
            let (h0, h1) = PERIOD_HOURS[q];
            [h0, h1, PERIOD_LEVEL[q] as f64, TO_WORK[q] as f64, TO_HOME[q] as f64, PEAK_HOUR_FACTOR[q] as f64]
        })
        .collect()
}

/// Multiply the riders of `transfer_pairs` triples (route node in, out, riders) by `f` (T-260).
pub(crate) fn scale_triples(v: &mut [f32], f: f32) {
    if f == 1.0 {
        return;
    }
    for t in v.chunks_exact_mut(3) {
        t[2] *= f;
    }
}

/// Free the strategy tables the populations on this thread share (`kernel::with_skims`); a worker
/// calls it when it has finished a period's stage (T-119).
#[wasm_bindgen]
pub fn release_skims() {
    clear_skim_cache();
}

impl DemandApi {
    fn params(&self, q: usize) -> Params {
        let (h0, h1) = PERIOD_HOURS[q];
        Params {
            period: q,
            period_share: TO_WORK[q] + TO_HOME[q],
            period_hours: (h1 - h0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.base.clone()
        }
    }

    /// The kernel's network for period `q`: that period's demand level's run times, and every
    /// period's headway (a line with no trains in a period has an infinite headway, so nobody
    /// boards it, and keeps its route nodes so results line up across periods).
    pub(crate) fn period_net(svc: &Service, q: usize) -> Network {
        let lev = PERIOD_LEVEL[q];
        let nl = svc.cars.len();
        let mut net = Network {
            st_x: svc.st_x.clone(),
            st_y: svc.st_y.clone(),
            line_name: vec![String::new(); nl],
            line_off: svc.line_off.clone(),
            line_stops: svc.stops.clone(),
            hop_min: vec![0.0; svc.stops.len()],
            hop_back: vec![0.0; svc.stops.len()],
            line_speed_kmh: vec![0.0; nl],
            line_headway_min: vec![],
            line_seats: svc.cars.iter().map(|&c| c * SEATS_PER_CAR).collect(),
            line_crush: svc.cars.iter().map(|&c| c * CRUSH_PER_CAR).collect(),
            st_group: svc.group.clone(),
        };
        for l in 0..nl {
            let (a, b) = (svc.line_off[l] as usize, svc.line_off[l + 1] as usize);
            let n = b - a;
            for k in 0..n {
                net.hop_min[a + k] = svc.times[6 * a + lev * n + k] / 60.0;
                net.hop_back[a + k] = svc.times[6 * a + (3 + lev) * n + k] / 60.0;
            }
            let hw: [f32; PERIODS] = std::array::from_fn(|p| {
                let t = svc.tph[l][PERIOD_LEVEL[p]];
                if t > 0.0 { 60.0 / t } else { f32::INFINITY }
            });
            net.line_headway_min.push(hw);
        }
        net
    }

    /// Mode choice and assignment for one period on the graph as it stands (free flow, or with
    /// the crowding factors already set). Returns the rail trips (both directions), the walk
    /// trips rail took, the loads, the pairs and the rail trips by place. Adds the transfers into
    /// `xfer` when given (T-176).
    fn pass(&self, g: &Graph, p: &Params, q: usize, xfer: Option<(&XferIndex, &mut [f32])>) -> (f64, f64, Loads, u64, SubOut) {
        let sz = self.sz.as_ref().unwrap();
        // the strategies are shared with the other populations on the same service (T-119)
        let (mo, l) = with_skims(g, p, &self.want, |sk| {
            let er = station_exp(sk, p);
            let mo = mode_choice(&self.z, &self.gr, sz, &er, g.n_st, p);
            drop(er);
            // the trips home: the same station pairs reversed
            let s = g.n_st;
            let (w, h) = (TO_WORK[q], TO_HOME[q]);
            let fw = w / (w + h);
            let fh = h / (w + h);
            let mut od = mo.od.clone();
            for a in 0..s {
                for t in a + 1..s {
                    let (x, y) = (od[a * s + t], od[t * s + a]);
                    od[a * s + t] = fw * x + fh * y;
                    od[t * s + a] = fw * y + fh * x;
                }
            }
            let l = match xfer {
                Some((ix, x)) => assign_with_transfers(g, sk, &od, ix, x),
                None => assign(g, sk, &od),
            };
            (mo, l)
        });
        let sub = SubOut {
            rail_home: mo.rail_by_sub,
            rail_work: mo.rail_to_sub,
            walk_lost_home: mo.walk_lost_by_sub,
            walk_lost_work: mo.walk_lost_to_sub,
            acc_home: mo.acc_home,
            acc_work: mo.acc_work,
            od: mo.od,
            walksum: mo.rail_walksum.iter().map(|&v| v as f32).collect(),
        };
        (mo.rail, mo.rail_walk, l, mo.pairs, sub)
    }
}

/// Checks on a solved period (T-007's real-network comparison); not exported to the app.
impl DemandApi {
    pub(crate) fn period(&self, q: usize) -> Option<&Period> {
        self.per.get(q).and_then(|p| p.as_ref())
    }
    /// Rail trips starting at each station (first boarding) in period `q`.
    pub fn entries(&self, q: usize) -> Vec<f32> {
        self.period(q).map_or(vec![], |st| st.avg.entry.clone())
    }
    /// Rail trips ending at each station (last alighting) in period `q`.
    pub fn exits(&self, q: usize) -> Vec<f32> {
        self.period(q).map_or(vec![], |st| st.avg.exit.clone())
    }
    /// Boardings and alightings per station, transfers included.
    pub fn station_board(&self, q: usize) -> (Vec<f32>, Vec<f32>) {
        self.period(q).map_or((vec![], vec![]), |st| (st.avg.board.clone(), st.avg.alight.clone()))
    }
    /// Rail trips in period `q` by home cell (each subzone's spread over its cells by commuters).
    pub fn rail_by_cell(&self, q: usize) -> Vec<f32> {
        let (Some(st), Some(sz)) = (self.period(q), self.sz.as_ref()) else { return vec![] };
        sz.cell
            .iter()
            .enumerate()
            .map(|(c, &u)| if u == u32::MAX || sz.pop[u as usize] <= 0.0 || st.sub.rail_home.is_empty() { 0.0 } else { st.sub.rail_home[u as usize] * self.city.pop[c] / sz.pop[u as usize] })
            .collect()
    }
    /// All trips (both directions) in period `q` by home cell.
    pub fn trips_by_cell(&self, q: usize) -> Vec<f32> {
        let share = self.params(q).period_share;
        self.city.pop.iter().zip(&self.z.cell_zone).map(|(&v, &z)| v * self.home_k[z as usize] * share).collect()
    }
    /// The route-node layout: per route node, its station and line (kernel order).
    pub fn route_nodes(&self, q: usize) -> (Vec<u32>, Vec<u32>) {
        self.period(q).map_or((vec![], vec![]), |st| (st.g.rn_station.clone(), st.g.rn_line.clone()))
    }
    /// Rail trips in period `q` by the straight-line distance between the home and the station
    /// boarded there, and between the work end and the station left there (`kernel::DIST_BANDS_KM`;
    /// a subzone's mean distance to the station). T-090: how much rail sits near the bound.
    pub fn access_bands(&self, q: usize) -> (Vec<f64>, Vec<f64>) {
        let (Some(st), Some(sz)) = (self.period(q), self.sz.as_ref()) else { return (vec![], vec![]) };
        let (mut h, mut w) = (vec![0f64; N_BANDS], vec![0f64; N_BANDS]);
        if st.sub.acc_home.len() == sz.acc_d.len() {
            for (k, &d) in sz.acc_d.iter().enumerate() {
                h[dist_band(d)] += st.sub.acc_home[k] as f64;
                w[dist_band(d)] += st.sub.acc_work[k] as f64;
            }
        }
        (h, w)
    }
    /// Rail trips in period `q` by the demand station boarded at the home end: those who walked
    /// there, and those who rode (T-371).
    pub fn home_access_modes(&self, q: usize) -> (Vec<f32>, Vec<f32>) {
        let (Some(st), Some(sz)) = (self.period(q), self.sz.as_ref()) else { return (vec![], vec![]) };
        let s = self.svc.as_ref().map_or(0, |v| v.st_x.len());
        let (mut walk, mut ride) = (vec![0f32; s], vec![0f32; s]);
        if st.sub.acc_home.len() == sz.acc_st.len() {
            for u in 0..sz.n {
                for k in sz.walk(u) {
                    walk[sz.acc_st[k] as usize] += st.sub.acc_home[k];
                }
                for k in sz.ride(u) {
                    ride[sz.acc_st[k] as usize] += st.sub.acc_home[k];
                }
            }
        }
        (walk, ride)
    }
    /// Rail trips to work in period `q` by the perceived minutes of the cheapest walks at both
    /// ends together (`kernel::WALKSUM_BAND_MIN` bands).
    pub fn walksum_bands(&self, q: usize) -> Vec<f64> {
        self.period(q).map_or(vec![], |st| st.sub.walksum.iter().map(|&v| v as f64).collect())
    }
    /// Snapshot station -> demand station (stations within `station_merge_m` are one).
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    /// The parameters every period starts from (to change before `set_network`).
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.base
    }
}

#[wasm_bindgen]
impl DemandApi {
    /// Parse a city pack (header text and .bin bytes) and rebuild its zone gravity.
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str, bin: &[u8]) -> Result<DemandApi, String> {
        let t0 = now_ms();
        let city = pack::parse(header, bin)?;
        let mut base = Params { asc_rail: ASC_RAIL, ..Params::default() };
        // T-277: a pack fitted on its own country's commuting (Toronto) ships its own distance
        // curve; the default is the US one, so take the pack's rather than re-solve with that
        if let Some(pg) = &city.gravity {
            if (pg.zone_m - base.zone_m).abs() < 0.5 {
                base.decay_km = pg.decay_km;
                base.decay_pow = pg.decay_pow;
            }
        }
        let (z, mut gr, gravity_from_pack) = city_gravity(&city, &base);
        // T-275: the pack's routed walks and drives, if it has them for these zones
        if let Some(r) = super::routes::Routes::from_pack(header, bin, &z)? {
            base.routes = Some(std::sync::Arc::new(r));
            gr.set_walk(&z, &base);
        }
        let (walk_home_zone, walk_work_zone) = gr.walk_trips_by_zone(&z);
        let walk_all = walk_home_zone.iter().sum();
        let (home_k, work_k) = zone_trip_factors(&city, &z, &gr, gravity_from_pack);
        Ok(DemandApi { city, z, gr, base, walk_all, walk_home_zone, walk_work_zone, home_k, work_k, svc: None, sz: None, want: vec![], per: (0..PERIODS).map(|_| None).collect(), fare_per_km: 0.0, fare_km: vec![], open_ms: now_ms() - t0, complex_in: vec![] })
    }

    /// The station complex of each station of the networks `set_network` is given from now on
    /// (T-161): one number per station, equal numbers one complex (game/complexes.ts, with the
    /// player's links). Riders change between any two stations of a complex on foot, the walk
    /// costed by distance like any walking transfer. Empty (the default) or a list of another
    /// length: no complexes, only the walks within `xfer_walk_m`.
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.complex_in = complex.to_vec();
    }

    /// Cells, zones, home -> work trips a day, the walk share of all trips with no rail at all,
    /// and the ms the constructor took.
    pub fn info(&self) -> Vec<f64> {
        vec![self.city.len() as f64, self.z.n as f64, self.gr.total, self.walk_all / self.gr.total, self.open_ms]
    }

    /// The network to solve on, packed: station x, y pairs (metres in the pack's frame); stops
    /// per line; the stops (station indices) line after line; 6 times per stop (seconds, see
    /// `Service::times`); trains an hour, 3 per line (high, medium, low); cars per line.
    /// Rebuilds station access and subzones (shared by every period) and forgets all results.
    /// Returns the ms it took.
    pub fn set_network(&mut self, st_xy: &[f32], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        let t0 = now_ms();
        let ns = st_xy.len() / 2;
        let nl = line_n.len();
        let mut line_off = vec![0u32];
        for &n in line_n {
            line_off.push(line_off.last().unwrap() + n);
        }
        let nstops = *line_off.last().unwrap() as usize;
        if stops.len() != nstops || times.len() != 6 * nstops || tph.len() != 3 * nl || cars.len() != nl {
            return Err(format!("network arrays disagree: {nl} lines, {nstops} stops, {} stop entries, {} times, {} tph, {} cars", stops.len(), times.len(), tph.len(), cars.len()));
        }
        if stops.iter().any(|&s| s as usize >= ns) {
            return Err("a stop names a station that is not in the list".into());
        }
        let (st_x, st_y, st_map) = merge_stations(st_xy, &line_off, stops, self.base.station_merge_m);
        let group = demand_groups(&st_map, st_x.len(), &self.complex_in);
        let svc = Service {
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            cars: cars.to_vec(),
            group,
        };
        self.per.iter_mut().for_each(|p| *p = None);
        self.fare_km.clear();
        self.sz = None;
        self.want.clear();
        if ns > 0 && nl > 0 {
            let net = Self::period_net(&svc, 0);
            let acc = build_access(&self.city, &net, &self.base);
            let sz = build_subzones(&self.city, &self.z, &acc, &net, &self.base);
            self.want = egress_stations(&sz, svc.st_x.len());
            self.sz = Some(sz);
        }
        self.svc = Some(svc);
        Ok(now_ms() - t0)
    }

    /// Set the player's price. Base is paid once per trip, including transfers; distance is
    /// charged on ride edges, so route choice and mode choice both see the actual route's fare.
    /// `hop_km` follows route nodes: each line's forward stops then reverse stops, 0 at ends.
    pub fn set_fares(&mut self, base_usd: f32, per_km_usd: f32, hop_km: &[f32]) -> Result<(), String> {
        let count = self.svc.as_ref().map_or(0, |s| 2 * s.stops.len());
        if !base_usd.is_finite() || base_usd < 0.0 || !per_km_usd.is_finite() || per_km_usd < 0.0
            || hop_km.len() != count || hop_km.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("invalid fare or segment distances".into());
        }
        self.base.fare_min = base_usd * FARE_MIN_PER_USD;
        self.fare_per_km = per_km_usd * FARE_MIN_PER_USD;
        self.fare_km = hop_km.to_vec();
        self.per.iter_mut().for_each(|p| *p = None);
        Ok(())
    }

    /// Run one solve and one crowding round on a small cross of two lines at the pack origin,
    /// then forget it. V8 runs a wasm function's first calls in its baseline compiler and swaps
    /// in optimised code only for later calls, so the first real solve took 3x as long as the
    /// next ones (notes/T-026.md); a worker calls this once after opening the city. Returns ms.
    pub fn warm_up(&mut self) -> f64 {
        let t0 = now_ms();
        let mut xy = vec![];
        for i in -6..=6 {
            xy.extend([i as f32 * 1000.0, 0.0]);
        }
        let mut b = vec![];
        for i in -6..=6i32 {
            if i == 0 {
                b.push(6u32);
            } else {
                b.push((xy.len() / 2) as u32);
                xy.extend([0.0, i as f32 * 1000.0]);
            }
        }
        let stops: Vec<u32> = (0..13).chain(b).collect();
        let times: Vec<f32> = (0..2).flat_map(|_| (0..6).flat_map(|r| (0..13).map(move |k| if (r < 3 && k == 12) || (r >= 3 && k == 0) { 0.0 } else { 90.0 }))).collect();
        if self.set_network(&xy, &[13, 13], &stops, &times, &[12.0, 6.0, 3.0, 12.0, 6.0, 3.0], &[8.0, 8.0]).is_ok() {
            self.solve(0);
            self.crowd(0);
        }
        self.per.iter_mut().for_each(|p| *p = None);
        self.svc = None;
        self.sz = None;
        self.want.clear();
        clear_skim_cache();
        now_ms() - t0
    }

    /// The free-flow first estimate for period `q` (SPEC 4.4). False if there is no network.
    pub fn solve(&mut self, q: usize) -> bool {
        let t0 = now_ms();
        let Some(svc) = &self.svc else { return false };
        if q >= PERIODS {
            return false;
        }
        let p = self.params(q);
        let trips = self.gr.total * p.period_share as f64;
        let walk0 = self.walk_all * p.period_share as f64;
        let net = Self::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        for (r, fare) in g.rn_fare.iter_mut().enumerate() {
            *fare = self.fare_km.get(r).copied().unwrap_or(0.0) * self.fare_per_km;
        }
        let no_crowding = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &no_crowding);
        let xi = xfer_index(&g);
        let mut xfer = vec![0.0; xi.len];
        let (rail, rail_walk, avg, pairs, sub) = if self.sz.is_some() {
            self.pass(&g, &p, q, Some((&xi, &mut xfer)))
        } else {
            (0.0, 0.0, Loads::zero(g.n_st, g.n - g.n_st), 0, SubOut::empty())
        };
        self.per[q] = Some(Period { net, p, g, avg, rail, walk: walk0 - rail_walk, trips, round: 0, ms: now_ms() - t0, pairs, sub, fac: no_crowding, uniform: true, xi, xfer });
        true
    }

    /// One crowding round for period `q` (a penalty per segment from the averaged loads, then
    /// search, mode choice and assignment again, averaged in). False if `q` has no solve yet.
    pub fn crowd(&mut self, q: usize) -> bool {
        let Some(Some(st)) = self.per.get(q) else { return false };
        let total = [&st.avg.seg[..], &st.avg.rn_board[..], &st.avg.rn_alight[..]].concat();
        self.crowd_with_external(q, &total)
    }

    /// One population's assignment under the complete shared-world loads. `total` includes this
    /// population; it is frozen before ANY population advances its round. It is the segment loads
    /// per route node, then the boardings and then the alightings per route node (riders left
    /// behind, T-188, `kernel::round_costs`); the segment loads alone price crowding only.
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(Some(st)) = self.per.get(q) else { return false };
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else { return false };
        let t0 = now_ms();
        let Some(mut st) = self.per.get_mut(q).and_then(|p| p.take()) else { return false };
        if self.sz.is_some() {
            if st.uniform && fac == st.fac && left == st.g.rn_left && !legacy() {
                // Nothing is crowded enough to change a cost, so this round would solve what the
                // round before did and average in the same loads: exactly nothing changes (T-119).
                st.round += 1;
                st.ms = now_ms() - t0;
                self.per[q] = Some(st);
                return true;
            }
            set_crowding(&mut st.g, &fac);
            st.g.rn_left = left;
            st.uniform = false;
            st.fac = fac;
            let mut xfer = vec![0.0; st.xi.len];
            let (rail, rail_walk, l, _, sub) = self.pass(&st.g, &st.p, q, Some((&st.xi, &mut xfer)));
            st.round += 1;
            let r = st.round;
            msa(&mut st.xfer, &xfer, r);
            msa(&mut st.avg.seg, &l.seg, r);
            msa(&mut st.avg.board, &l.board, r);
            msa(&mut st.avg.alight, &l.alight, r);
            msa(&mut st.avg.rn_board, &l.rn_board, r);
            msa(&mut st.avg.rn_alight, &l.rn_alight, r);
            msa(&mut st.avg.entry, &l.entry, r);
            msa(&mut st.avg.exit, &l.exit, r);
            st.sub.msa(&sub, r);
            let w = 1.0 / (r as f64 + 1.0);
            st.rail += (rail - st.rail) * w;
            let walk = self.walk_all * st.p.period_share as f64 - rail_walk;
            st.walk += (walk - st.walk) * w;
        } else {
            st.round += 1;
        }
        st.ms = now_ms() - t0;
        self.per[q] = Some(st);
        true
    }

    /// Period `q`: round (0 = free flow), trips, rail trips, walked trips, ms of the last stage,
    /// subzone pairs, segments over seats at the busiest hour, worst load as a share of crush,
    /// rider-weighted p50 and p90 crowding penalty. Empty if `q` has no solve.
    pub fn summary(&self, q: usize) -> Vec<f64> {
        let Some(Some(st)) = self.per.get(q) else { return vec![] };
        let (_, cs) = crowd_factors(&st.g, &st.net, &st.avg, &st.p);
        vec![
            st.round as f64,
            st.trips,
            st.rail,
            st.walk,
            st.ms,
            st.pairs as f64,
            cs.segs_over_seats as f64,
            cs.max_load_of_crush as f64,
            cs.p50 as f64,
            cs.p90 as f64,
        ]
    }

    /// Riders in period `q` on the segment leaving each route node (0 at a run's last stop).
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|p| p.as_ref()).map_or(vec![], |st| st.avg.seg.clone())
    }
    /// Riders in period `q` boarding at each route node.
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|p| p.as_ref()).map_or(vec![], |st| st.avg.rn_board.clone())
    }
    /// Riders in period `q` leaving the train at each route node.
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|p| p.as_ref()).map_or(vec![], |st| st.avg.rn_alight.clone())
    }
    /// Transfers over this worker's periods (T-176) with either end at a route node marked in
    /// `sel` (one byte per route node): triples (route node in, route node out, riders), each
    /// period's in turn, not added up (the caller sums them by pair). Riders arriving at the
    /// first and boarding the second, walking between stations included; averaged over the
    /// crowding rounds like the loads.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    /// `transfers` on a kind of day (T-260): each period's riders times the commute's load factor.
    pub fn transfers_day(&self, sel: &[u8], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (q, st) in self.per.iter().enumerate() {
            let Some(st) = st else { continue };
            let at = out.len();
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
            scale_triples(&mut out[at..], load_factor(day as usize, WORK, q));
        }
        out
    }
    /// Load at the period's busiest hour on the segment leaving each route node, as a share of
    /// the crush capacity of the trains running then (0 where none run).
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        let Some(Some(st)) = self.per.get(q) else { return vec![] };
        (0..st.avg.seg.len())
            .map(|r| {
                let l = st.g.rn_line[r] as usize;
                let tph = 60.0 / st.net.line_headway_min[l][q];
                let crush = tph * st.net.line_crush[l];
                if crush > 0.0 { st.avg.seg[r] / st.p.period_hours * st.p.peak_hour_factor / crush } else { 0.0 }
            })
            .collect()
    }

    // --- Demand views (T-078 for T-084; notes/T-078.md). A worker holds only its own periods,
    // so each answers for the periods it has solved and the main thread adds them up.

    /// Periods this worker has a result for, as a bit mask (bit q = period q).
    pub fn solved_mask(&self) -> u32 {
        (0..PERIODS).filter(|&q| self.period(q).is_some()).fold(0, |m, q| m | 1 << q)
    }

    /// Selected commute trip legs: outward then return, each mode-major, five periods.
    /// Same endpoint allocation as cell_modes; this is a period-specific mode query.
    pub fn purpose_timing(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        self.purpose_timing_day(end, cells, WEEKDAY as u32)
    }

    /// `purpose_timing` on a kind of day (T-260): each period's legs scaled by the commute's load
    /// factor and split into to work and home by that day's shares.
    pub fn purpose_timing_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        self.purpose_timing_in(end, cells, day, u32::MAX)
    }

    /// `purpose_timing_day` over the solved periods in `mask` alone (bit q = period q; T-336): a
    /// zone no station reaches, held by the auxiliary worker alone, answers for each solving
    /// worker's periods in turn, so a selection's figures add up as a solve's would.
    pub fn purpose_timing_mask_day(&self, end: u32, cells: &[u32], mask: u32, day: u32) -> Vec<f32> {
        self.purpose_timing_in(end, cells, day, mask)
    }

    fn purpose_timing_in(&self, end: u32, cells: &[u32], day: u32, mask: u32) -> Vec<f32> {
        let day = day as usize;
        let n = self.city.len();
        let mut out = vec![0.0; 30];
        for q in 0..PERIODS {
            if mask >> q & 1 == 0 { continue; }
            let Some(st) = self.period(q) else { continue };
            let mut sums = vec![0.0; 4 * self.sz.as_ref().map_or(0, |s| s.n)];
            let ns = sums.len() / 4;
            if ns > 0 {
                sums[..ns].copy_from_slice(&st.sub.rail_home);
                sums[ns..2*ns].copy_from_slice(&st.sub.rail_work);
                sums[2*ns..3*ns].copy_from_slice(&st.sub.walk_lost_home);
                sums[3*ns..].copy_from_slice(&st.sub.walk_lost_work);
            }
            let modes = self.cell_modes(&sums);
            let share = TO_WORK[q] + TO_HOME[q];
            for &c in cells {
                let c = c as usize;
                if c >= n { continue; }
                let base = (end.min(1) as usize * 3) * n + c;
                let rail = 2.0 * modes[base];
                // cell_modes includes all-day walking/driving baselines: take this period's
                // baseline and subtract its actual rail diversion, rather than scaling daily rail.
                let z = self.z.cell_zone[c] as usize;
                let daily_total = if end == 0 { self.city.pop[c] * self.home_k[z] } else { self.city.jobs[c] * self.work_k[z] };
                let (mass, zmass, walkz) = if end == 0 { (self.city.pop[c], self.z.pop[z], self.walk_home_zone[z]) }
                    else { (self.city.jobs[c], self.z.jobs[z], self.walk_work_zone[z]) };
                let lost = if let Some(sz) = &self.sz {
                    let u = sz.cell[c];
                    if u == u32::MAX { 0.0 } else {
                        let u = u as usize;
                        let denom = if end == 0 { sz.pop[u] } else { sz.jobs[u] };
                        let v = if end == 0 { st.sub.walk_lost_home[u] } else { st.sub.walk_lost_work[u] };
                        if denom > 0.0 { v * mass / denom } else { 0.0 }
                    }
                } else { 0.0 };
                let walk = ((walkz / zmass.max(1e-9)) as f32 * mass * share - lost).max(0.0);
                let drive = (daily_total * share - rail - walk).max(0.0);
                let f = load_factor(day, WORK, q);
                let (tw, th) = (share_out(day, WORK, q), share_ret(day, WORK, q));
                let sh = tw + th;
                for (m, v) in [rail, walk, drive].into_iter().enumerate() {
                    out[m*5+q] += v * f * tw / sh;
                    out[15+m*5+q] += v * f * th / sh;
                }
            }
        }
        out
    }

    /// Subzones of the current network (0 with none).
    pub fn n_subzones(&self) -> u32 {
        self.sz.as_ref().map_or(0, |sz| sz.n as u32)
    }

    /// Rail trips by subzone summed over this worker's solved periods, four blocks of
    /// `n_subzones`: rail by home/work, then legacy zero walking-loss blocks. Add the blocks of
    /// every worker and pass the sum to `cell_modes`.
    pub fn sub_sums(&self) -> Vec<f32> {
        self.sub_sums_day(WEEKDAY as u32)
    }

    /// `sub_sums` on a kind of day (T-260, `days.rs`): each period's rail weighed by the commute's
    /// load factor that day (exactly 1 on a weekday).
    pub fn sub_sums_day(&self, day: u32) -> Vec<f32> {
        let ns = self.n_subzones() as usize;
        let mut out = vec![0f32; 4 * ns];
        for q in 0..PERIODS {
            let Some(st) = self.period(q) else { continue };
            if st.sub.rail_home.len() != ns {
                continue;
            }
            let f = load_factor(day as usize, WORK, q);
            for (b, v) in [&st.sub.rail_home, &st.sub.rail_work, &st.sub.walk_lost_home, &st.sub.walk_lost_work].into_iter().enumerate() {
                for (o, &x) in out[b * ns..(b + 1) * ns].iter_mut().zip(v) {
                    *o += x * f;
                }
            }
        }
        out
    }

    /// Commuters a day per cell by how they travel, from the day's `sub_sums` of every worker:
    /// six blocks of one value per cell (pack order), at the home end rail, walk, drive, then at
    /// the work end rail, walk, drive. Empty if `sums` does not fit the current network. A
    /// commuter is one person making a trip to work and one home.
    pub fn cell_modes(&self, sums: &[f32]) -> Vec<f32> {
        self.cell_modes_day(sums, WEEKDAY as u32)
    }

    /// `cell_modes` on a kind of day (T-260), from that day's `sub_sums_day`: everyone's trips
    /// and the walkers scaled by the day's commute rate.
    pub fn cell_modes_day(&self, sums: &[f32], day: u32) -> Vec<f32> {
        let r = day_rate(day as usize, WORK);
        let n = self.city.len();
        // No subzones (a network with no running line): nobody by rail, everyone walks or drives.
        // This used to answer empty, and the demand views kept the previous network's answer
        // (T-099: the real New York save's bubbles stayed after a new game).
        let sz = self.sz.as_ref();
        let ns = sz.map_or(0, |s| s.n);
        if sums.len() != 4 * ns {
            return vec![];
        }
        let (scell, spop, sjobs): (&[u32], &[f32], &[f32]) = match sz {
            Some(s) => (&s.cell, &s.pop, &s.jobs),
            None => (&[], &[], &[]),
        };
        let mut out = vec![0f32; 6 * n];
        for c in 0..n {
            let zc = self.z.cell_zone[c] as usize;
            let (pop, jobs) = (self.city.pop[c], self.city.jobs[c]);
            let u = scell.get(c).copied().unwrap_or(u32::MAX);
            // home end
            let total = pop * self.home_k[zc] * r;
            let mut walk = if self.z.pop[zc] > 0.0 { (self.walk_home_zone[zc] / self.z.pop[zc]) as f32 * pop * r } else { 0.0 };
            let mut rail = 0.0;
            if u != u32::MAX && spop[u as usize] > 0.0 {
                let f = pop / spop[u as usize] / 2.0;
                rail = sums[u as usize] * f;
                walk -= sums[2 * ns + u as usize] * f;
            }
            let walk = walk.max(0.0);
            out[c] = rail;
            out[n + c] = walk;
            out[2 * n + c] = (total - rail - walk).max(0.0);
            // work end
            let total = jobs * self.work_k[zc] * r;
            let mut walk = if self.z.jobs[zc] > 0.0 { (self.walk_work_zone[zc] / self.z.jobs[zc]) as f32 * jobs * r } else { 0.0 };
            let mut rail = 0.0;
            if u != u32::MAX && sjobs[u as usize] > 0.0 {
                let f = jobs / sjobs[u as usize] / 2.0;
                rail = sums[ns + u as usize] * f;
                walk -= sums[3 * ns + u as usize] * f;
            }
            let walk = walk.max(0.0);
            out[3 * n + c] = rail;
            out[4 * n + c] = walk;
            out[5 * n + c] = (total - rail - walk).max(0.0);
        }
        out
    }

    /// One station's riders over this worker's solved periods, commuters a day, packed (`station`
    /// is an index into `set_network`'s station list; stations merged for demand answer as one):
    /// `[n_home, n_work, walked, rode, home cells.., their riders.., work cells.., their riders..,
    /// work zones of the home riders (one per zone).., home zones of the work riders (one per
    /// zone)..]`. Home riders live in those cells and use this station at the home end of their
    /// commute, `walked` of them on foot and `rode` by a ride (T-371, T-374; commutes only, the
    /// same unit as the riders); work riders work there and use it at the work end. Cell and zone indices are exact in f32
    /// (under 2^24). Empty if there is no network or no such station.
    pub fn station_riders(&self, station: u32) -> Vec<f32> {
        self.station_riders_day(station, WEEKDAY as u32)
    }

    /// `station_riders` on a kind of day (T-260): each period weighed by the commute's load factor.
    pub fn station_riders_day(&self, station: u32, day: u32) -> Vec<f32> {
        let (Some(sz), Some(svc)) = (self.sz.as_ref(), self.svc.as_ref()) else { return vec![] };
        let Some(&a) = svc.st_map.get(station as usize) else { return vec![] };
        let a = a as usize;
        let s = svc.st_x.len();
        let nz = self.z.n;
        let (mut home_r, mut work_r) = (vec![0f32; sz.n], vec![0f32; sz.n]);
        let (mut to_zone, mut from_zone) = (vec![0f32; nz], vec![0f32; nz]);
        let (mut dep, mut arr) = (vec![0f32; s], vec![0f32; s]);
        let (mut walked, mut rode) = (0f32, 0f32);
        for q in 0..PERIODS {
            let Some(st) = self.period(q) else { continue };
            let sub = &st.sub;
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != s * s {
                continue;
            }
            dep.iter_mut().for_each(|v| *v = 0.0);
            arr.iter_mut().for_each(|v| *v = 0.0);
            let f = load_factor(day as usize, WORK, q);
            for (k, &t) in sz.acc_st.iter().enumerate() {
                dep[t as usize] += sub.acc_home[k];
                arr[t as usize] += sub.acc_work[k];
            }
            for u in 0..sz.n {
                for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                    let t = sz.acc_st[k] as usize;
                    if t == a {
                        home_r[u] += sub.acc_home[k] * f;
                        work_r[u] += sub.acc_work[k] * f;
                        if k < sz.acc_mid[u] as usize {
                            walked += sub.acc_home[k] * f;
                        } else {
                            rode += sub.acc_home[k] * f;
                        }
                    }
                    // where this station's home riders work: their trips to station t, spread
                    // over the subzones arriving at t; and where its work riders live
                    let zu = sz.zone[u] as usize;
                    if arr[t] > 0.0 {
                        to_zone[zu] += sub.od[a * s + t] * sub.acc_work[k] / arr[t] * f;
                    }
                    if dep[t] > 0.0 {
                        from_zone[zu] += sub.od[t * s + a] * sub.acc_home[k] / dep[t] * f;
                    }
                }
            }
        }
        let cells = |r: &[f32], w: &[f32], wsub: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let (mut c, mut v) = (vec![], vec![]);
            for u in 0..sz.n {
                if r[u] <= 0.0 || wsub[u] <= 0.0 {
                    continue;
                }
                for &m in &sz.mem[sz.mem_off[u] as usize..sz.mem_off[u + 1] as usize] {
                    let x = r[u] / 2.0 * w[m as usize] / wsub[u];
                    if x > 0.0 {
                        c.push(m as f32);
                        v.push(x);
                    }
                }
            }
            (c, v)
        };
        let (hc, hv) = cells(&home_r, &self.city.pop, &sz.pop);
        let (wc, wv) = cells(&work_r, &self.city.jobs, &sz.jobs);
        let mut out = Vec::with_capacity(4 + 2 * (hc.len() + wc.len()) + 2 * nz);
        out.extend([hc.len() as f32, wc.len() as f32, walked / 2.0, rode / 2.0]);
        out.extend(hc);
        out.extend(hv);
        out.extend(wc);
        out.extend(wv);
        out.extend(to_zone.iter().map(|&v| v / 2.0));
        out.extend(from_zone.iter().map(|&v| v / 2.0));
        out
    }

    // --- Flows (T-098 for T-097's bubbles; notes/T-098.md): for a set of cells at one end of the
    // commute (homes, or jobs), all their commuters by mode spread over the other end. Two steps,
    // like `sub_sums` and `cell_modes`: every worker gives the rail part of its periods
    // (`flow_rail`), the main thread adds them up, one worker spreads everything (`flows`).

    /// The selection's share of each subzone at its end (`end` 0: homes, 1: jobs), by the
    /// commuters living or working in the selected cells. Repeated or unknown cells are ignored.
    fn selection(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        let Some(sz) = self.sz.as_ref() else { return vec![] };
        let (w, wsub) = if end == 0 { (&self.city.pop, &sz.pop) } else { (&self.city.jobs, &sz.jobs) };
        let mut seen = vec![false; self.city.len()];
        let mut su = vec![0f32; sz.n];
        for &c in cells {
            let c = c as usize;
            if c >= seen.len() || seen[c] {
                continue;
            }
            seen[c] = true;
            let u = sz.cell[c];
            if u != u32::MAX && wsub[u as usize] > 0.0 {
                su[u as usize] += w[c] / wsub[u as usize];
            }
        }
        su
    }

    /// Step 1, every worker: the selected cells' rail trips over this worker's solved periods,
    /// three blocks of `n_subzones`: by subzone at the far end; and for the selected subzones at
    /// their own end, rail trips and legacy zero walking-loss values (whole subzones, not
    /// scaled to the selection; `flows` does that per cell, as `cell_modes` does). Trips, both
    /// directions: add every worker's blocks and pass the sum to `flows`. Riders boarding at a
    /// station are assumed to spread over the stations they ride to like all its riders, and
    /// riders leaving at a station over the places they walk to like all of them (as
    /// `station_riders`): exact at the near end, a little blurred at the far end. Empty with no
    /// network.
    pub fn flow_rail(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        self.flow_rail_day(end, cells, WEEKDAY as u32)
    }

    /// `flow_rail` on a kind of day (T-260): each period weighed by the commute's load factor.
    pub fn flow_rail_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        let (Some(sz), Some(svc)) = (self.sz.as_ref(), self.svc.as_ref()) else { return vec![] };
        let su = self.selection(end, cells);
        let (ns, s) = (sz.n, svc.st_x.len());
        let mut out = vec![0f32; 3 * ns];
        let (mut dep, mut arr, mut q) = (vec![0f32; s], vec![0f32; s], vec![0f32; s]);
        let mut far_st = vec![0f32; s];
        for p in 0..PERIODS {
            let Some(st) = self.period(p) else { continue };
            let sub = &st.sub;
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != s * s {
                continue;
            }
            let (near_rail, near_lost) = if end == 0 { (&sub.rail_home, &sub.walk_lost_home) } else { (&sub.rail_work, &sub.walk_lost_work) };
            let f = load_factor(day as usize, WORK, p);
            for u in 0..ns {
                if su[u] > 0.0 {
                    out[ns + u] += near_rail[u] * f;
                    out[2 * ns + u] += near_lost[u] * f;
                }
            }
            dep.iter_mut().for_each(|v| *v = 0.0);
            arr.iter_mut().for_each(|v| *v = 0.0);
            q.iter_mut().for_each(|v| *v = 0.0);
            far_st.iter_mut().for_each(|v| *v = 0.0);
            // stations used at the near end by the selection, and by everyone
            let near_acc = if end == 0 { &sub.acc_home } else { &sub.acc_work };
            for u in 0..ns {
                for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                    let t = sz.acc_st[k] as usize;
                    dep[t] += sub.acc_home[k];
                    arr[t] += sub.acc_work[k];
                    if su[u] > 0.0 {
                        q[t] += su[u] * near_acc[k];
                    }
                }
            }
            // over the station pairs to the stations used at the far end
            for a in 0..s {
                if q[a] <= 0.0 {
                    continue;
                }
                if end == 0 {
                    let f = q[a] / dep[a];
                    for (o, &x) in far_st.iter_mut().zip(&sub.od[a * s..(a + 1) * s]) {
                        *o += f * x;
                    }
                } else {
                    let f = q[a] / arr[a];
                    for (b, o) in far_st.iter_mut().enumerate() {
                        *o += f * sub.od[b * s + a];
                    }
                }
            }
            // and over the places around them
            let (far_acc, far_tot) = if end == 0 { (&sub.acc_work, &arr) } else { (&sub.acc_home, &dep) };
            for v in 0..ns {
                let mut x = 0.0f32;
                for k in sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize {
                    let t = sz.acc_st[k] as usize;
                    if far_st[t] > 0.0 {
                        x += far_st[t] * far_acc[k] / far_tot[t];
                    }
                }
                out[v] += x * f;
            }
        }
        out
    }

    /// Step 2, one worker: the selected cells' commuters a day by mode, from `sums` (every
    /// worker's `flow_rail` added up), packed: `[rail, walk, drive, cut rail, cut walk, cut drive,
    /// n, n far cells (pack order).., their rail.., walk.., drive..]`. The first three are the
    /// selection's totals and match `cell_modes` summed over the selected cells. Far cells are
    /// every cell at the other end with commuters from (or to) the selection, rail included;
    /// cells that together hold the last `FLOW_CUT` of them, the smallest, are left out and
    /// their commuters added to the cut totals, so the far cells plus the cut add up to the
    /// totals. Cell indices are exact in f32. Empty if `sums` does not fit the current network.
    pub fn flows(&self, end: u32, cells: &[u32], sums: &[f32]) -> Vec<f32> {
        self.flows_day(end, cells, sums, WEEKDAY as u32)
    }

    /// `flows` on a kind of day (T-260), from that day's `flow_rail_day`: everyone and the
    /// walkers scaled by the day's commute rate.
    pub fn flows_day(&self, end: u32, cells: &[u32], sums: &[f32], day: u32) -> Vec<f32> {
        let rate = day_rate(day as usize, WORK);
        // No subzones (no running line): no rail, everyone walks or drives (T-099; this answered
        // empty before, and a selection's far end never came after a new game).
        let (scell, szone, spop, sjobs, ns): (&[u32], &[u32], &[f32], &[f32], usize) = match self.sz.as_ref() {
            Some(s) => (&s.cell, &s.zone, &s.pop, &s.jobs, s.n),
            None => (&[], &[], &[], &[], 0),
        };
        let (n, nz) = (self.city.len(), self.z.n);
        if sums.len() != 3 * ns {
            return vec![];
        }
        let home = end == 0;
        let (w, wsub, wzone, walk_zone) = if home { (&self.city.pop, spop, &self.z.pop, &self.walk_home_zone) } else { (&self.city.jobs, sjobs, &self.z.jobs, &self.walk_work_zone) };
        let (fw, fwsub, fzone) = if home { (&self.city.jobs, sjobs, &self.z.jobs) } else { (&self.city.pop, spop, &self.z.pop) };
        // T-276: each zone's trips kept in the pack per unit of weight
        let kz = if home { &self.home_k } else { &self.work_k };
        // the near end, cell by cell as `cell_modes` does
        let (mut rail, mut walk, mut drive, mut lost) = (0f64, 0f64, 0f64, 0f64);
        let mut wz = vec![0f64; nz];
        let mut seen = vec![false; n];
        for &c in cells {
            let c = c as usize;
            if c >= n || seen[c] || w[c] <= 0.0 {
                continue;
            }
            seen[c] = true;
            let zc = self.z.cell_zone[c] as usize;
            if wzone[zc] <= 0.0 {
                continue;
            }
            wz[zc] += w[c] as f64 / wzone[zc];
            let total = w[c] * kz[zc] * rate;
            let mut wk = (walk_zone[zc] / wzone[zc]) as f32 * w[c] * rate;
            let mut r = 0.0;
            let u = scell.get(c).copied().unwrap_or(u32::MAX);
            if u != u32::MAX && wsub[u as usize] > 0.0 {
                let f = w[c] / wsub[u as usize] / 2.0;
                r = sums[ns + u as usize] * f;
                wk -= sums[2 * ns + u as usize] * f;
                lost += (sums[2 * ns + u as usize] * f) as f64;
            }
            let wk = wk.max(0.0);
            rail += r as f64;
            walk += wk as f64;
            drive += (total - r - wk).max(0.0) as f64;
        }
        // the far end by zone: all commuters and those who would walk with no rail (the gravity),
        // and rail (from the far subzones)
        let (mut t_z, mut w_z, mut r_z) = (vec![0f64; nz], vec![0f64; nz], vec![0f64; nz]);
        let wt = !self.gr.wtab.is_empty();
        for a in 0..nz {
            if wz[a] <= 0.0 {
                continue;
            }
            for b in 0..nz {
                // home end: a is the home zone; job end: a is the work zone
                let (i, j) = if home { (a, b) } else { (b, a) };
                let idx = (self.gr.base(&self.z, i) + self.gr.key(&self.z, j)) as usize;
                let t = wz[a] * self.gr.row[i] * self.gr.col[j] * (self.gr.ftab[idx] * self.gr.k(&self.z, i, j)) as f64 * rate as f64;
                t_z[b] += t;
                if wt {
                    w_z[b] += t * self.gr.walk_share_at(i, j, idx) as f64;
                }
            }
        }
        for v in 0..ns {
            r_z[szone[v] as usize] += sums[v] as f64 / 2.0;
        }
        // Legacy loss reconciliation (zero since T-153); layout retained for worker compatibility.
        let lost_w: f64 = (0..nz).filter(|&b| t_z[b] > 0.0).map(|b| r_z[b] * w_z[b] / t_z[b]).sum();
        let lf = if lost_w > 0.0 { lost / lost_w } else { 0.0 };
        // per far cell: its zone's commuters by its people, rail from its subzone, the rest split
        // as its zone's walk and drive
        let mut fc: Vec<(u32, f32, f32, f32)> = vec![];
        for c in 0..n {
            let j = self.z.cell_zone[c] as usize;
            if fw[c] <= 0.0 || t_z[j] <= 0.0 || fzone[j] <= 0.0 {
                continue;
            }
            let tot = (t_z[j] * fw[c] as f64 / fzone[j]) as f32;
            let v = scell.get(c).copied().unwrap_or(u32::MAX);
            let r = if v != u32::MAX && fwsub[v as usize] > 0.0 { sums[v as usize] / 2.0 * fw[c] / fwsub[v as usize] } else { 0.0 };
            let non_z = (t_z[j] - r_z[j]).max(1e-12);
            let walk_z = (w_z[j] - r_z[j] * w_z[j] / t_z[j] * lf).max(0.0);
            let share = (walk_z / non_z).min(1.0) as f32;
            let non = (tot - r).max(0.0);
            if tot > 0.0 || r > 0.0 {
                fc.push((c as u32, r, non * share, non * (1.0 - share)));
            }
        }
        // The rail chain is exact in total but blurred per place, so a far cell can get more rail
        // than the gravity sends it commuters (its non-rail goes to 0, and the far end adds up to
        // more than the near end: 1.7% at a line's end on the test network). Scale walk and drive
        // to the near end's totals, so every total is exact.
        let (sw, sd) = fc.iter().fold((0f64, 0f64), |a, e| (a.0 + e.2 as f64, a.1 + e.3 as f64));
        let (kw, kd) = ((if sw > 0.0 { walk / sw } else { 0.0 }) as f32, (if sd > 0.0 { drive / sd } else { 0.0 }) as f32);
        for e in fc.iter_mut() {
            e.2 *= kw;
            e.3 *= kd;
        }
        // leave out the smallest far cells holding the last FLOW_CUT of the commuters: a
        // histogram on log2 of each cell's commuters, from the top down
        let mass: f64 = fc.iter().map(|e| (e.1 + e.2 + e.3) as f64).sum();
        const BINS: usize = 512;
        let top = fc.iter().map(|e| e.1 + e.2 + e.3).fold(0f32, f32::max).max(1e-30);
        let (hi, lo) = (top.log2() + 1e-3, top.log2() - 40.0);
        let bin = |x: f32| if x <= 0.0 { 0 } else { (((x.log2() - lo) / (hi - lo) * BINS as f32) as isize).clamp(0, BINS as isize - 1) as usize };
        let mut hist = vec![0f64; BINS];
        for e in &fc {
            hist[bin(e.1 + e.2 + e.3)] += (e.1 + e.2 + e.3) as f64;
        }
        let (mut cum, mut keep_from) = (0f64, 0usize);
        for b in (0..BINS).rev() {
            cum += hist[b];
            keep_from = b;
            if cum >= (1.0 - FLOW_CUT) * mass {
                break;
            }
        }
        let (mut kept, mut cut) = (vec![], [0f64; 3]);
        for e in fc {
            if bin(e.1 + e.2 + e.3) >= keep_from {
                kept.push(e);
            } else {
                cut[0] += e.1 as f64;
                cut[1] += e.2 as f64;
                cut[2] += e.3 as f64;
            }
        }
        // the cut is what the kept cells leave of the totals (the same as adding up the cells left
        // out, but free of rounding)
        let far = |m: usize| kept.iter().map(|e| [e.1, e.2, e.3][m] as f64).sum::<f64>();
        cut = [(rail - far(0)).max(0.0), (walk - far(1)).max(0.0), (drive - far(2)).max(0.0)];
        let mut out = Vec::with_capacity(7 + 4 * kept.len());
        out.extend([rail as f32, walk as f32, drive as f32, cut[0] as f32, cut[1] as f32, cut[2] as f32, kept.len() as f32]);
        out.extend(kept.iter().map(|e| e.0 as f32));
        out.extend(kept.iter().map(|e| e.1));
        out.extend(kept.iter().map(|e| e.2));
        out.extend(kept.iter().map(|e| e.3));
        out
    }

    /// The city's cells: x, y per cell (metres east and north of the pack origin, pack order).
    pub fn cell_xy(&self) -> Vec<f32> {
        self.city.x.iter().zip(&self.city.y).flat_map(|(&x, &y)| [x, y]).collect()
    }
    /// H3 index of each cell (resolution 9).
    pub fn cell_h3(&self) -> Vec<u64> {
        self.city.h3.clone()
    }
    /// The weights of local commuting per cell: commuters living there, then commuters working
    /// there (T-118: a selection's share of a 600 m square of the longer-trip market).
    pub fn cell_commuters(&self) -> Vec<f32> {
        self.city.pop.iter().chain(self.city.jobs.iter()).copied().collect()
    }
    /// Gravity zone of each cell.
    pub fn cell_zone(&self) -> Vec<u32> {
        self.z.cell_zone.clone()
    }
    /// The zones: side in metres, then the centre x, y of each zone (metres from the pack origin).
    /// A square a region border cuts holds a zone per region (T-162): those are placed at the
    /// mean of their own cells instead, so they do not sit on top of each other.
    pub fn zone_xy(&self) -> Vec<f32> {
        let zm = self.z.zone_m;
        let (off, _) = self.z.by_square();
        let shared = |i: usize| {
            let s = (self.z.gx[i] * self.z.h + self.z.gy[i]) as usize;
            off[s + 1] - off[s] > 1
        };
        let (mut sx, mut sy, mut sn) = (vec![0f64; self.z.n], vec![0f64; self.z.n], vec![0u32; self.z.n]);
        for c in 0..self.city.len() {
            let i = self.z.cell_zone[c] as usize;
            if shared(i) {
                sx[i] += self.city.x[c] as f64;
                sy[i] += self.city.y[c] as f64;
                sn[i] += 1;
            }
        }
        let mut out = vec![zm];
        for i in 0..self.z.n {
            if sn[i] > 0 {
                out.push((sx[i] / sn[i] as f64) as f32);
                out.push((sy[i] / sn[i] as f64) as f32);
            } else {
                out.push(self.z.x0 + (self.z.gx[i] as f32 + 0.5) * zm);
                out.push(self.z.y0 + (self.z.gy[i] as f32 + 0.5) * zm);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::{kernel, pack, synth};

    /// A small pack (the synthetic city's inner 12 km, gravity solved and shipped) and two
    /// crossing lines.
    fn small() -> (DemandApi, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let full = synth::synth_city(1);
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 12_000.0).collect();
        let mut city = pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: vec![], gravity: None,
        };
        let p = Params::default();
        let z = kernel::build_zones(&city, p.zone_m);
        let g = kernel::gravity(&z, &p, 100, 1e-5);
        city.gravity = Some(kernel::to_pack_gravity(&z, &g, &p));
        let (h, b) = pack::write(&city);
        let api = DemandApi::new(&h, &b).unwrap();
        // stations every 1 km along x and along y through the centre (shared at the origin)
        let mut xy = vec![];
        for i in -8..=8 {
            xy.extend([i as f32 * 1000.0, 0.0]);
        }
        let centre = 8u32;
        let mut line_b = vec![];
        for i in -8..=8i32 {
            if i == 0 {
                line_b.push(centre);
            } else {
                line_b.push((xy.len() / 2) as u32);
                xy.extend([0.0, i as f32 * 1000.0]);
            }
        }
        let line_a: Vec<u32> = (0..17).collect();
        let stops: Vec<u32> = line_a.iter().chain(&line_b).cloned().collect();
        // 90 s a hop in both directions at every level, 0 past the ends
        let mut times = vec![];
        for _ in 0..2 {
            for run in 0..2 {
                for _lev in 0..3 {
                    for k in 0..17 {
                        let end = if run == 0 { k == 16 } else { k == 0 };
                        times.push(if end { 0.0 } else { 90.0 });
                    }
                }
            }
        }
        let tph = vec![20.0, 10.0, 4.0, 12.0, 6.0, 0.0];
        let cars = vec![10.0, 8.0];
        (api, xy, vec![17, 17], stops, times, tph, cars)
    }

    /// T-161: snapshot complexes carried to demand stations; a demand station in two complexes
    /// joins them; a list of the wrong length means none.
    #[test]
    fn complexes_follow_merged_stations() {
        // snapshot stations 0..5 -> demand stations 0, 0, 1, 2, 3; complexes {0, 2}, {1, 3}, {4}
        let map = [0, 0, 1, 2, 3];
        assert_eq!(demand_groups(&map, 4, &[7, 9, 7, 9, 4]), vec![0, 0, 0, 3]);
        assert_eq!(demand_groups(&map, 4, &[7, 8, 7, 9, 4]), vec![0, 0, 2, 3]);
        assert!(demand_groups(&map, 4, &[1, 2]).is_empty());
        let (mut api, xy, n, stops, times, tph, cars) = small();
        // the two ends of line a, 16 km apart, as one complex
        let mut complex: Vec<u32> = (0..xy.len() as u32 / 2).collect();
        complex[16] = 0;
        api.set_complexes(&complex);
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        assert!(api.solve(0));
        let st = api.period(0).unwrap();
        let (a, b) = (api.station_map()[0], api.station_map()[16]);
        assert_eq!(st.net.st_group[a as usize], st.net.st_group[b as usize]);
        let g = &st.g;
        let walks = (0..g.adj_to.len()).filter(|&e| g.adj_seg[e] == u32::MAX && g.adj_to[e] < g.n_st as u32).count();
        api.set_complexes(&[]);
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        assert!(api.solve(0));
        let g = &api.period(0).unwrap().g;
        let plain = (0..g.adj_to.len()).filter(|&e| g.adj_seg[e] == u32::MAX && g.adj_to[e] < g.n_st as u32).count();
        // each end has one route node a run (the line's first or last stop), both runs: 4 walks
        assert_eq!(walks, plain + 4);
    }

    #[test]
    fn shared_crowding_uses_complete_loads_without_overwriting_own_assignment() {
        let (mut own, xy, n, stops, times, tph, cars) = small();
        let (mut external, ..) = small();
        let (mut legacy, ..) = small();
        for api in [&mut own, &mut external, &mut legacy] {
            api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
            assert!(api.solve(0));
        }
        let loads = own.seg(0);
        // segment loads, boardings and alightings (T-188), the population's own
        let own_all = [own.seg(0), own.board(0), own.alight(0)].concat();
        assert!(own.crowd_with_external(0, &own_all));
        assert!(legacy.crowd(0));
        assert_eq!(own.seg(0), legacy.seg(0), "own-only external input exactly preserves old solver");
        let shared: Vec<f32> = loads.iter().map(|v| v * 50.0).collect();
        assert!(external.crowd_with_external(0, &shared));
        assert!(external.summary(0)[2] < own.summary(0)[2], "other populations must reduce this population's rail choice when trains are full");
        assert!(external.seg(0).iter().sum::<f32>() < shared.iter().sum::<f32>(), "assignment remains this population, not the summed feedback vector");
        let before = external.seg(0);
        assert!(!external.crowd_with_external(0, &[f32::NAN]));
        let mut invalid = shared; invalid[0] = -1.0;
        assert!(!external.crowd_with_external(0, &invalid));
        assert_eq!(external.seg(0), before, "bad shared loads leave period intact");
        assert_eq!(external.summary(0)[0], 1.0);
    }

    /// T-119: strategies are searched only towards stations someone can alight at. A line out in the
    /// country (no cell within the walk's bound of its stations) is not searched towards, and the
    /// pass gives what a search towards every station gives, bit for bit.
    #[test]
    fn searching_only_the_egress_stations_changes_nothing() {
        let (mut api, mut xy, mut n, mut stops, mut times, mut tph, mut cars) = small();
        // a third line of two stations 60 km out, 4 minutes between them
        let far = (xy.len() / 2) as u32;
        xy.extend([60_000.0, 0.0, 62_000.0, 0.0]);
        n.push(2);
        stops.extend([far, far + 1]);
        for _ in 0..2 {
            times.extend([240.0, 0.0, 240.0, 0.0, 240.0, 0.0]);
        }
        // run 1 of a two-stop line: stop 1 to stop 0
        let at = times.len() - 6;
        times[at..].copy_from_slice(&[0.0, 240.0, 0.0, 240.0, 0.0, 240.0]);
        tph.extend([6.0, 6.0, 6.0]);
        cars.push(8.0);
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        assert!(api.want.iter().any(|&w| !w) && api.want.iter().any(|&w| w), "some stations searched, some not");
        assert!(!api.want[far as usize] && !api.want[far as usize + 1]);
        api.solve(0);
        api.crowd(0);
        let st = api.per[0].as_ref().unwrap();
        clear_skim_cache();
        let (rail, rail_walk, l, pairs, sub) = api.pass(&st.g, &st.p, 0, None);
        // the reference: every station searched, nothing cached
        let sz = api.sz.as_ref().unwrap();
        let sk = kernel::skims(&st.g, &st.p);
        let er = kernel::station_exp(&sk, &st.p);
        let mo = kernel::mode_choice(&api.z, &api.gr, sz, &er, st.g.n_st, &st.p);
        let s = st.g.n_st;
        let (w, h) = (TO_WORK[0], TO_HOME[0]);
        let (fw, fh) = (w / (w + h), h / (w + h));
        let mut od = mo.od.clone();
        for a in 0..s {
            for t in a + 1..s {
                let (x, y) = (od[a * s + t], od[t * s + a]);
                od[a * s + t] = fw * x + fh * y;
                od[t * s + a] = fw * y + fh * x;
            }
        }
        let want = kernel::assign(&st.g, &sk, &od);
        assert_eq!((rail, rail_walk, pairs), (mo.rail, mo.rail_walk, mo.pairs));
        assert_eq!(l.seg, want.seg);
        assert_eq!(l.board, want.board);
        assert_eq!(l.rn_board, want.rn_board);
        assert_eq!(l.rn_alight, want.rn_alight);
        assert_eq!(l.entry, want.entry);
        assert_eq!(sub.od, mo.od);
        assert!(rail > 0.0);
        clear_skim_cache();
    }

    /// T-119: when nothing is crowded a crowding round solves what the free-flow estimate did, so it
    /// is skipped, and the skip changes nothing: the loads, the rail trips and the views' sums are
    /// the free-flow estimate's to the last bit, round by round, and the round count still moves.
    #[test]
    fn uncrowded_rounds_repeat_the_estimate_exactly() {
        let (mut api, xy, n, stops, times, tph, _) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &[400.0, 400.0]).unwrap();
        api.solve(0);
        let (seg, rail, sums) = (api.seg(0), api.summary(0)[2], api.sub_sums());
        for r in 1..=3 {
            assert!(api.crowd(0));
            assert_eq!(api.summary(0)[0], r as f64);
            assert_eq!(api.seg(0), seg);
            assert_eq!(api.summary(0)[2], rail);
            assert_eq!(api.sub_sums(), sums);
        }
        // a skipped round is the pass it replaces: the same network crowded the legacy way gives the same
        let (mut crowded, xy, n, stops, times, tph, cars) = small();
        crowded.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        crowded.solve(0);
        let free = crowded.seg(0);
        crowded.crowd(0);
        assert_ne!(crowded.seg(0), free, "a crowded network moves in a round");
    }

    /// T-188: a line cut to too few trains leaves riders behind, and with their extra wait priced
    /// into the crowding rounds it loses riders (here to the car: the two lines cross, so they
    /// are not alternatives, and line a loses the riders who changed from b); the extra wait is
    /// a single line's, at its own stops; with room on every train nothing changes.
    #[test]
    fn a_line_with_too_few_trains_loses_riders_to_the_wait() {
        let run = |left_behind: bool, tph_b: f32, cars_b: f32| {
            let (mut api, xy, n, stops, times, mut tph, mut cars) = small();
            api.params_mut().left_behind = left_behind;
            tph[3] = tph_b;
            cars[1] = cars_b;
            // line a with room to spare (with rides to stations, T-371, 20 cars leave riders
            // behind mid-peak on this city)
            cars[0] = 40.0;
            api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
            api.solve(0);
            for _ in 0..3 {
                assert!(api.crowd(0));
            }
            let st = api.period(0).unwrap();
            let b: f64 = api.board(0).iter().zip(&st.g.rn_line).filter(|p| *p.1 == 1).map(|p| *p.0 as f64).sum();
            let a: f64 = api.board(0).iter().zip(&st.g.rn_line).filter(|p| *p.1 == 0).map(|p| *p.0 as f64).sum();
            let left_a = st.g.rn_left.iter().zip(&st.g.rn_line).filter(|p| *p.1 == 0).map(|p| *p.0).fold(0.0f32, f32::max);
            let left_b = st.g.rn_left.iter().zip(&st.g.rn_line).filter(|p| *p.1 == 1).map(|p| *p.0).fold(0.0f32, f32::max);
            (b, a, api.summary(0)[2], left_a, left_b)
        };
        // line b with room: 12 trains an hour of 64 cars; nobody is left behind and the waits
        // change nothing
        let (b_full_off, _, rail_full_off, _, _) = run(false, 12.0, 64.0);
        let (b_full, _, rail_full, _, lb_full) = run(true, 12.0, 64.0);
        assert_eq!(lb_full, 0.0);
        assert_eq!((b_full, rail_full), (b_full_off, rail_full_off));
        // cut to 4 trains of 8 cars (a twelfth of the room): crowding alone takes some riders
        // off; the wait for full trains takes visibly more
        let (b_off, a_off, rail_off, _, lb_off) = run(false, 4.0, 8.0);
        let (b_on, a_on, rail_on, la_on, lb_on) = run(true, 4.0, 8.0);
        assert_eq!(lb_off, 0.0, "no waits priced with the switch off");
        assert!(lb_on > 5.0, "line b's riders wait {lb_on} minutes more at worst");
        assert_eq!(la_on, 0.0, "line a has room: {la_on}");
        assert!(b_off < b_full, "crowding alone: {b_off:.0} of {b_full:.0}");
        assert!(b_on < 0.95 * b_off, "line b keeps {b_on:.0} of {b_off:.0} riders");
        // line a loses the riders who changed from b; the rest drive
        assert!(rail_on < rail_off, "and some drive: rail {rail_on:.0} against {rail_off:.0}");
        println!("line b full {b_full:.0}; cut, crowding only {b_off:.0}, with the wait {b_on:.0} riders (worst extra wait {lb_on:.1} min); line a {a_off:.0} -> {a_on:.0}; rail {rail_off:.0} -> {rail_on:.0}");
    }

    /// T-188: `kernel::left_behind` on one line: riders within the trains' room wait nothing
    /// extra; past it, the wait grows with the overload, falls on the stops where riders board
    /// into full trains, and is capped.
    #[test]
    fn left_behind_waits_grow_with_the_overload() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        api.solve(0);
        let st = api.period(0).unwrap();
        let (g, net, p) = (&st.g, &st.net, &st.p);
        let nr = g.rn_line.len();
        // line a run 0 (route nodes 0..17): riders board at the first 4 stops, leave at the last
        let crush_hour = 20.0 * 10.0 * CRUSH_PER_CAR; // 32,000 an hour
        let waits = |per_hour: f32| {
            let mut board = vec![0.0f32; nr];
            let mut alight = vec![0.0f32; nr];
            for k in 0..4 {
                board[k] = per_hour / 4.0 * p.period_hours;
            }
            alight[16] = per_hour * p.period_hours;
            kernel::left_behind(g, net, p, &board, &alight)
        };
        let (w, s) = waits(0.5 * crush_hour);
        assert!(w.iter().all(|&v| v == 0.0) && s.left == 0.0, "half full: {s:?}");
        // the busiest hour is 1.4 times the mean: at 0.8 of crush on average the middle is over
        let (w08, s08) = waits(0.8 * crush_hour);
        let (w12, s12) = waits(1.2 * crush_hour);
        assert!(s08.left > 0.0 && s12.left > s08.left, "{s08:?} {s12:?}");
        assert!(w12[3] > w08[3] && w08[3] > 0.0, "{} {}", w08[3], w12[3]);
        // the first stops find empty trains; the fourth boards into trains three stops full
        assert!(w12[0] < w12[3], "{:?}", &w12[..4]);
        assert!(w12[4..].iter().all(|&v| v == 0.0));
        let (w5, _) = waits(5.0 * crush_hour);
        assert!(w5[3] == kernel::LEFT_WAIT_CAP_MIN, "{}", w5[3]);
        // the rider-minutes divided by the riders are the boardings' average (below the cap)
        assert!(w08.iter().all(|&v| v < kernel::LEFT_WAIT_CAP_MIN));
        let tot: f64 = (0..4).map(|k| w08[k] as f64 * (0.8 * crush_hour / 4.0 * p.period_hours) as f64).sum();
        assert!((tot - s08.rider_min).abs() < 1e-3 * s08.rider_min, "{tot} {}", s08.rider_min);
    }

    #[test]
    fn fares_change_ridership_and_preserve_legacy_price() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        let mut km = vec![];
        for &count in &n { for _ in 0..2 { for k in 0..count { km.push(if k + 1 == count { 0.0 } else { 1.0 }); } } }
        api.solve(0);
        let legacy = api.summary(0)[2];
        api.set_fares(2.0, 0.0, &km).unwrap();
        assert_eq!(api.solved_mask(), 0);
        api.solve(0);
        assert_eq!(legacy, api.summary(0)[2], "legacy $2 flat fare is exactly 6 minutes");
        api.set_fares(0.0, 0.0, &km).unwrap(); api.solve(0);
        let free = api.summary(0)[2];
        api.set_fares(1.5, 0.1, &km).unwrap(); api.solve(0);
        let default = api.summary(0)[2];
        api.set_fares(20.0, 0.0, &km).unwrap(); api.solve(0);
        let expensive = api.summary(0)[2];
        api.set_fares(0.0, 10.0, &km).unwrap(); api.solve(0);
        let distance = api.summary(0)[2];
        assert!(free > default && default > expensive && free > distance,
            "free {free}, default {default}, expensive {expensive}, distance {distance}");
        // Crowding stretches riding time, never the fare; transfers do not add another base.
        api.crowd(0);
        let st = api.per[0].as_ref().unwrap();
        for e in 0..st.g.adj_to.len() {
            let seg = st.g.adj_seg[e];
            if seg != u32::MAX { let r = seg as usize - st.g.n_st; assert_eq!(st.g.rn_fare[r], km[r] * 30.0); }
        }
        assert!(api.set_fares(f32::NAN, 0.0, &km).is_err());
        assert!(api.set_fares(0.0, -1.0, &km).is_err());
        assert!(api.set_fares(0.0, 0.0, &[]).is_err());
        println!("fare riders: legacy={legacy:.0} free={free:.0} default={default:.0} expensive={expensive:.0} distance={distance:.0}");
    }

    #[test]
    fn fare_distances_follow_each_runs_track_and_base_is_once() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        let mut km = vec![0.0; 2 * stops.len()];
        // First line: 2 km forward segments and 3 km reverse segments.
        km[..16].fill(2.0);
        km[17..33].fill(3.0);
        api.set_fares(0.0, 0.0, &km).unwrap();
        api.solve(0);
        let st = api.per[0].as_ref().unwrap();
        let free = skims(&st.g, &st.p);
        api.set_fares(5.0, 0.5, &km).unwrap();
        api.solve(0);
        let st = api.per[0].as_ref().unwrap();
        let priced = skims(&st.g, &st.p);
        // Stations 0 and 1 are joined by the first forward/reverse segment respectively.
        assert!((priced.r[1] - free.r[1] - 3.0).abs() < 1e-5);
        assert!((priced.r[priced.s] - free.r[free.s] - 4.5).abs() < 1e-5);
        assert_eq!(st.p.fare_min, 15.0);
        // A two-line trip through the centre also sees no extra $5 in its route graph.
        let other = stops[17] as usize;
        assert!(priced.r[other].is_finite());
        let sk_before = priced.r.clone();
        api.set_fares(50.0, 0.5, &km).unwrap();
        api.solve(0);
        let st = api.per[0].as_ref().unwrap();
        assert_eq!(skims(&st.g, &st.p).r, sk_before);
        // Explicit multiplier check: fares stay unscaled even at double riding disutility.
        let mut graph = build_graph(&st.net, &st.p);
        graph.rn_fare = km.iter().map(|v| v * 1.5).collect();
        let factor = vec![2.0; graph.rn_ride.len()];
        set_crowding(&mut graph, &factor);
        for e in 0..graph.adj_to.len() {
            if graph.adj_seg[e] != u32::MAX {
                let r = graph.adj_seg[e] as usize - graph.n_st;
                assert_eq!(graph.adj_cost[e], 2.0 * graph.rn_ride[r] + km[r] * 1.5);
            }
        }
    }

    #[test]
    fn short_commutes_reserve_walking_before_rail_choice() {
        let city = pack::City { name: "short".into(), h3: vec![1, 2],
            x: vec![0.0, 300.0], y: vec![0.0, 0.0],
            pop: vec![100.0, 0.0], jobs: vec![0.0, 100.0], region: vec![], gravity: None };
        let (h, b) = pack::write(&city);
        let mut api = DemandApi::new(&h, &b).unwrap();
        let baseline = api.info()[3];
        assert!(baseline > 0.8, "same-zone trips overwhelmingly walk");
        api.params_mut().asc_rail = 20.0; // even overwhelmingly attractive rail cannot take walkers
        let times = [10.0,0.0,10.0,0.0,10.0,0.0,0.0,10.0,0.0,10.0,0.0,10.0];
        api.set_network(&[0.0,0.0,300.0,0.0], &[2], &[0,1], &times, &[60.0;3], &[4.0]).unwrap();
        api.set_fares(0.0, 0.0, &[0.3,0.0,0.0,0.3]).unwrap();
        for q in 0..5 {
            api.solve(q);
            let v = api.summary(q);
            assert!((v[3]/v[1]-baseline).abs()<1e-6);
            assert!(v[2] > 0.0 && v[2]/v[1] <= 1.0-baseline+1e-5);
            api.crowd(q);
            let crowded = api.summary(q);
            assert!((crowded[3]/crowded[1]-baseline).abs()<1e-6);
        }
        let sums = api.flow_rail(1, &[1]);
        let flows = api.flows(1, &[1], &sums);
        assert!((flows[1] as f64/(flows[0]+flows[1]+flows[2]) as f64-baseline).abs()<1e-5);
        println!("300m same-zone commute: walk {:.2}%, rail capped {:.2}%", baseline*100.0,(1.0-baseline)*100.0);
    }

    #[test]
    fn periods_solve_and_add_up() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        let info = api.info();
        assert!(info[3] > 0.0 && info[3] < 0.5, "walk share with no rail {}", info[3]);
        let mut day_trips = 0.0;
        for q in 0..PERIODS {
            assert!(api.solve(q));
            let s0 = api.summary(q);
            assert!(api.crowd(q));
            let s = api.summary(q);
            assert_eq!(s[0], 1.0);
            let (trips, rail, walk) = (s[1], s[2], s[3]);
            day_trips += trips;
            assert!(rail > 0.0 && rail < trips && walk > 0.0 && rail + walk < trips, "period {q}: {s:?}");
            assert!(s0[2] >= rail * 0.5);
            // every rail trip boards once and alights once (plus transfers, on both lines)
            let boards: f64 = api.board(q).iter().map(|&v| v as f64).sum();
            let alights: f64 = api.alight(q).iter().map(|&v| v as f64).sum();
            assert!(boards >= rail * 0.999 && (boards / alights - 1.0).abs() < 1e-3, "{boards} {alights} {rail}");
            assert_eq!(api.seg(q).len(), 68);
            // line B has no trains at night: nobody on it
            if q == 4 {
                assert!(api.board(q)[34..].iter().all(|&v| v == 0.0));
            }
        }
        assert!((day_trips / (2.0 * info[2]) - 1.0).abs() < 1e-6);
        // the trips home in the evening peak load the opposite direction to the morning's
        let lc = api.load_of_crush(0);
        assert!(lc.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn views_with_no_network() {
        // T-099: no running line means no rail, not no answer
        let (mut api, ..) = small();
        api.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
        let mut walk = 0.0;
        for q in 0..PERIODS {
            api.solve(q);
            walk += api.summary(q)[3];
        }
        let cm = api.cell_modes(&api.sub_sums());
        let nc = api.cell_xy().len() / 2;
        assert_eq!(cm.len(), 6 * nc);
        let tot = |b: usize| cm[b * nc..(b + 1) * nc].iter().map(|&v| v as f64).sum::<f64>();
        assert_eq!(tot(0), 0.0);
        assert!((tot(1) / (walk / 2.0) - 1.0).abs() < 2e-3, "walk {} vs {}", tot(1), walk / 2.0);
        let cell = (0..nc).find(|&c| cm[c] + cm[nc + c] + cm[2 * nc + c] > 0.0).unwrap() as u32;
        let rail = api.flow_rail(0, &[cell]);
        assert!(rail.is_empty());
        assert!(!api.flows(0, &[cell], &rail).is_empty());
    }

    /// The demand views (T-078): per-cell modes add up to the day's rail and walk, a station's
    /// riders add up the same by cell and by zone at the far end, and stations far apart differ.
    #[test]
    fn views_add_up() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        let (mut rail, mut walk) = (0.0, 0.0);
        for q in 0..PERIODS {
            api.solve(q);
            api.crowd(q);
            let s = api.summary(q);
            rail += s[2];
            walk += s[3];
        }
        assert_eq!(api.solved_mask(), (1 << PERIODS) - 1);
        let cm = api.cell_modes(&api.sub_sums());
        let nc = api.cell_xy().len() / 2;
        assert_eq!(cm.len(), 6 * nc);
        let tot = |b: usize| cm[b * nc..(b + 1) * nc].iter().map(|&v| v as f64).sum::<f64>();
        let commuters = api.info()[2];
        for (end, b) in [("home", 0), ("work", 3)] {
            assert!((tot(b) / (rail / 2.0) - 1.0).abs() < 1e-3, "{end} rail {} vs {}", tot(b), rail / 2.0);
            assert!((tot(b + 1) / (walk / 2.0) - 1.0).abs() < 2e-3, "{end} walk {} vs {}", tot(b + 1), walk / 2.0);
            assert!(((tot(b) + tot(b + 1) + tot(b + 2)) / commuters - 1.0).abs() < 2e-3, "{end} all");
        }
        assert!(api.cell_modes(&[1.0]).is_empty());
        let nz = api.zone_xy().len() / 2;
        let riders = |st: u32| {
            let r = api.station_riders(st);
            let (nh, nw) = (r[0] as usize, r[1] as usize);
            let sum = |a: usize, k: usize| r[a..a + k].iter().map(|&v| v as f64).sum::<f64>();
            let (home, work) = (sum(4 + nh, nh), sum(4 + 2 * nh + nw, nw));
            let o = 4 + 2 * nh + 2 * nw;
            assert_eq!(r.len(), o + 2 * nz);
            assert!((sum(o, nz) / home - 1.0).abs() < 1e-3 && (sum(o + nz, nz) / work - 1.0).abs() < 1e-3, "station {st}");
            // T-374: the walked and rode riders are the home riders, split
            assert!(((r[2] + r[3]) as f64 / home - 1.0).abs() < 1e-3, "station {st}: walked {} rode {} of {home}", r[2], r[3]);
            (home, work, r[4] as usize)
        };
        // the line's end and the shared centre station
        let (h_end, _, c_end) = riders(0);
        let (h_mid, w_mid, _) = riders(8);
        assert!(h_end > 0.0 && h_mid > 0.0 && w_mid > 0.0);
        let (x, y) = (api.cell_xy()[2 * c_end], api.cell_xy()[2 * c_end + 1]);
        // home cells ride from up to `ride_max_m` from their group's centre (T-371)
        assert!((x - xy[0]).hypot(y - xy[1]) < api.base.ride_max_m + 9000.0, "a catchment cell beyond the ride's bound");
        assert!(api.station_riders(9999).is_empty());
    }

    /// Flows (T-098): a selection's totals match `cell_modes` over its cells, at both ends; the
    /// far cells plus the cut add up to them; the cut is small.
    #[test]
    fn flows_add_up() {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        for q in 0..PERIODS {
            api.solve(q);
            api.crowd(q);
        }
        let cm = api.cell_modes(&api.sub_sums());
        let cxy = api.cell_xy();
        let nc = cxy.len() / 2;
        let near = |x: f32, y: f32, r: f32| -> Vec<u32> { (0..nc as u32).filter(|&c| (cxy[2 * c as usize] - x).hypot(cxy[2 * c as usize + 1] - y) < r).collect() };
        // around the line's end, the shared centre (one cell, and a bubble), and far from both lines
        let one = (0..nc as u32).min_by(|&a, &b| cxy[2 * a as usize].hypot(cxy[2 * a as usize + 1]).partial_cmp(&cxy[2 * b as usize].hypot(cxy[2 * b as usize + 1])).unwrap()).unwrap();
        let sels = [near(xy[0], xy[1], 1500.0), vec![one, one], near(0.0, 0.0, 2000.0), near(-6500.0, 6500.0, 1000.0)];
        for (si, sel) in sels.iter().enumerate() {
            assert!(!sel.is_empty());
            for end in 0..2u32 {
                let f = api.flows(end, sel, &api.flow_rail(end, sel));
                let b = 3 * end as usize;
                // a cell named twice counts once
                let mut uniq = sel.clone();
                uniq.dedup();
                let want: Vec<f64> = (0..3).map(|m| uniq.iter().map(|&c| cm[(b + m) * nc + c as usize] as f64).sum()).collect();
                let total: f64 = want.iter().sum();
                let k = f[6] as usize;
                for m in 0..3 {
                    let got = f[m] as f64;
                    assert!((got - want[m]).abs() <= 1e-3 * total.max(1.0), "selection {si} end {end} mode {m}: {got} vs cell_modes {}", want[m]);
                    let far: f64 = f[7 + (m + 1) * k..7 + (m + 2) * k].iter().map(|&v| v as f64).sum();
                    assert!((far + f[3 + m] as f64 - got).abs() <= 1e-3 * total.max(1.0), "selection {si} end {end} mode {m}: far {far} + cut {} vs {got}", f[3 + m]);
                    assert!(f[3 + m] as f64 >= -2e-3 * total.max(1.0), "selection {si} end {end} mode {m}: cut {}", f[3 + m]);
                }
                let cut: f64 = (3..6).map(|m| f[m] as f64).sum();
                assert!(cut <= 0.03 * total, "selection {si} end {end}: cut {cut} of {total}");
                if si == 0 {
                    assert!(f[0] > 0.0, "the line's end has rail");
                }
                // far from the lines: homes ride to them (T-371), jobs have no walk from them
                if si == 3 && end == 0 {
                    assert!(f[0] > 0.0, "homes far from the lines ride");
                }
                if si == 3 && end == 1 {
                    assert_eq!(f[0], 0.0, "no rail to jobs far from the lines");
                }
            }
        }
        assert!(api.flows(0, &sels[0], &[1.0]).is_empty());
    }

    /// The real New York pack with T-005's hand-made network: prints the day's split and the
    /// per-period timings. `cargo test --release --features demand nyc_day -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn nyc_day() {
        let (h, b) = pack::load_raw(concat!(env!("CARGO_MANIFEST_DIR"), "/../data/packs/nyc.json")).unwrap();
        let t0 = now_ms();
        let mut api = DemandApi::new(&h, &b).unwrap();
        // TW_LEFT_OFF=1: without the waits for full trains (T-188), to compare
        api.params_mut().left_behind = std::env::var("TW_LEFT_OFF").is_err();
        let info = api.info();
        println!("open {:.0} ms: {} cells, {} zones, {:.0} commutes a day, walk with no rail {:.2}%", now_ms() - t0, info[0], info[1], info[2], 100.0 * info[3]);
        let net = crate::demand::network::synth_network();
        let xy: Vec<f32> = (0..net.n_stations()).flat_map(|s| [net.st_x[s], net.st_y[s]]).collect();
        let n: Vec<u32> = (0..net.n_lines()).map(|l| net.stops(l).len() as u32).collect();
        let mut times = vec![];
        for l in 0..net.n_lines() {
            let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
            for run in 0..2 {
                for _ in 0..3 {
                    for k in a..b {
                        times.push(60.0 * if run == 0 { net.hop_min[k] } else if k > a { net.hop_min[k - 1] } else { 0.0 });
                    }
                }
            }
        }
        let tph: Vec<f32> = net.line_headway_min.iter().flat_map(|h| [60.0 / h[0], 60.0 / h[1], 60.0 / h[4]]).collect();
        // cars from the crush load (seats per car differ by kind in network.rs; here one car fits all)
        let cars: Vec<f32> = net.line_crush.iter().map(|c| (c / CRUSH_PER_CAR).round()).collect();
        let ms = api.set_network(&xy, &n, &net.line_stops, &times, &tph, &cars).unwrap();
        println!("set_network {ms:.0} ms ({} stations, {} lines)", net.n_stations(), net.n_lines());
        let (mut trips, mut rail, mut walk) = (0.0, 0.0, 0.0);
        for q in 0..PERIODS {
            api.solve(q);
            let s0 = api.summary(q);
            api.crowd(q);
            let s = api.summary(q);
            println!(
                "period {q}: free flow {:.0} ms, crowding round {:.0} ms; {:.0} pairs; trips {:.0}, rail {:.0} -> {:.0} ({:.1}%), walk {:.0}; worst {:.0}% of crush",
                s0[4], s[4], s[5], s[1], s0[2], s[2], 100.0 * s[2] / s[1], s[3], 100.0 * s[7]
            );
            trips += s[1];
            rail += s[2];
            walk += s[3];
        }
        println!("day: {:.0} trips, train {:.1}%, walk {:.1}%, drive {:.1}%", trips, 100.0 * rail / trips, 100.0 * walk / trips, 100.0 * (1.0 - (rail + walk) / trips));
    }

    /// T-119: the real combined Northeast network (nyc_real_track.json plus one Philadelphia line)
    /// solved as the app's workers do (both city populations and the regional market, five periods,
    /// free flow and three crowding rounds), with timings per stage. With `TW_DUMP=<file>` the
    /// summed results are written; with `TW_COMPARE=<file>` they are compared to such a file.
    /// `cargo test --release --features demand northeast_profile -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn northeast_profile() {
        use crate::demand::intercity::IntercityApi;
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let nyc = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let phl = pack::load_raw(&format!("{root}/data/packs/phl.json")).unwrap();
        let ic_header = std::fs::read_to_string(format!("{root}/data/packs/intercity.json")).unwrap();
        let t0 = now_ms();
        let mut pops = [DemandApi::new(&nyc.0, &nyc.1).unwrap(), DemandApi::new(&phl.0, &phl.1).unwrap()];
        let mut ic = IntercityApi::new(&ic_header).unwrap();
        println!("open {:.0} ms", now_ms() - t0);
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (mut st_xy, mut line_n, mut stops, mut times, mut tph, mut cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        // one Philadelphia line: two stations 4 km apart under City Hall's world position
        const R: f64 = 6371008.8;
        let rad = std::f64::consts::PI / 180.0;
        let wx = R * (-75.1636 + 73.985) * rad * (40.758 * rad).cos();
        let wy = R * (39.9526 - 40.758) * rad;
        let a = (st_xy.len() / 2) as u32;
        st_xy.extend([wx as f32, (wy - 2000.0) as f32, wx as f32, (wy + 2000.0) as f32]);
        line_n.push(2);
        stops.extend([a, a + 1]);
        for run in 0..2 {
            for _ in 0..3 {
                times.extend(if run == 0 { [600.0, 0.0] } else { [0.0, 600.0] });
            }
        }
        // `times` is laid out per line as 6 blocks of n values; the NYC lines were pushed in that order
        tph.extend([4.0, 4.0, 2.0]);
        cars.push(8.0);
        let ns = st_xy.len() / 2;
        println!("{ns} stations, {} lines, {} stops", line_n.len(), stops.len());
        let hop_km: Vec<f32> = {
            let mut v = vec![];
            let mut off = 0usize;
            for &n in &line_n {
                for dir in 0..2 {
                    for i in 0..n as usize {
                        let k = if dir == 0 { off + i } else { off + n as usize - 1 - i };
                        let next = if dir == 0 { k + 1 } else { k.wrapping_sub(1) };
                        v.push(if i + 1 < n as usize {
                            let (s0, s1) = (stops[k] as usize, stops[next] as usize);
                            ((st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1])) / 1000.0
                        } else {
                            0.0
                        });
                    }
                }
                off += n as usize;
            }
            v
        };
        // the game's start fare is 0.1 a km; TW_PERKM=0 shows the changes that must be exact
        if std::env::var("TW_LEGACY").is_ok() {
            kernel::LEGACY.store(true, std::sync::atomic::Ordering::Relaxed);
            println!("LEGACY: the solver as it was before T-119");
        }
        let per_km: f32 = std::env::var("TW_PERKM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1);
        let origin = [(-73.985f64, 40.758f64), (-75.1636, 39.9526)];
        for (k, a) in pops.iter_mut().enumerate() {
            let (lon0, lat0) = origin[k];
            let mut xy = st_xy.clone();
            for i in 0..ns {
                let lon = -73.985 + xy[2 * i] as f64 / (R * (40.758 * rad).cos() * rad);
                let lat = 40.758 + xy[2 * i + 1] as f64 / (R * rad);
                xy[2 * i] = (R * (lon - lon0) * rad * (lat0 * rad).cos()) as f32;
                xy[2 * i + 1] = (R * (lat - lat0) * rad) as f32;
            }
            let ms = a.set_network(&xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            a.set_fares(2.9, per_km, &hop_km).unwrap();
            println!("pop {k}: set_network {ms:.0} ms, {} subzones", a.n_subzones());
        }
        let ms = ic.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        ic.set_fares(2.9, per_km, &hop_km).unwrap();
        println!("intercity set_network {ms:.0} ms");
        // stage breakdown of the big population, period 0
        {
            let a = &mut pops[0];
            a.solve(0);
            let st = a.per[0].as_ref().unwrap();
            let t = now_ms();
            let g = kernel::build_graph(&st.net, &st.p);
            let t_graph = now_ms() - t;
            let t = now_ms();
            let sk = kernel::skims(&g, &st.p);
            let t_skims = now_ms() - t;
            let t = now_ms();
            let er = kernel::station_exp(&sk, &st.p);
            let t_er = now_ms() - t;
            let t = now_ms();
            let mo = kernel::mode_choice(&a.z, &a.gr, a.sz.as_ref().unwrap(), &er, g.n_st, &st.p);
            let t_mode = now_ms() - t;
            let t = now_ms();
            let _l = kernel::assign(&g, &sk, &mo.od);
            let t_assign = now_ms() - t;
            println!("NYC period 0 breakdown: graph {t_graph:.0}, skims {t_skims:.0}, station_exp {t_er:.0}, mode_choice {t_mode:.0} ({} pairs), assign {t_assign:.0} ms; S={} N={}", mo.pairs, g.n_st, g.n);
        }
        for a in pops.iter_mut() {
            a.per.iter_mut().for_each(|p| *p = None);
        }
        let mut result = serde_json::Map::new();
        crate::demand::alloc::reset_peak();
        let live0 = crate::demand::alloc::current();
        let tstart = now_ms();
        let mut stage_ms = vec![[0.0f64; 3]; 4];
        for round in 0..4 {
            let tr = now_ms();
            let mut per_period: Vec<f64> = vec![];
            for q in 0..PERIODS {
                let sum = |pops: &[DemandApi; 2], ic: &IntercityApi| -> Vec<f32> {
                    let mut out = pops[0].seg(q);
                    for (o, v) in out.iter_mut().zip(pops[1].seg(q)) {
                        *o += v;
                    }
                    for (o, v) in out.iter_mut().zip(ic.seg(q)) {
                        *o += v;
                    }
                    out
                };
                if round == 0 {
                    for (k, a) in pops.iter_mut().enumerate() {
                        let t = now_ms();
                        a.solve(q);
                        stage_ms[round][k] += now_ms() - t;
                    }
                    let t = now_ms();
                    ic.solve(q);
                    stage_ms[round][2] += now_ms() - t;
                } else {
                    let total = sum(&pops, &ic);
                    for (k, a) in pops.iter_mut().enumerate() {
                        let t = now_ms();
                        assert!(a.crowd_with_external(q, &total));
                        stage_ms[round][k] += now_ms() - t;
                    }
                    let t = now_ms();
                    assert!(ic.crowd_with_external(q, &total));
                    stage_ms[round][2] += now_ms() - t;
                }
                release_skims();
                print!("[p{q} {:.0}] ", now_ms() - tr - per_period.iter().sum::<f64>());
                per_period.push(now_ms() - tr - per_period.iter().sum::<f64>());
            }
            println!();
            per_period.clear();
            println!("round {round}: all five periods one after another {:.0} ms (nyc {:.0}, phl {:.0}, intercity {:.0})", now_ms() - tr, stage_ms[round][0], stage_ms[round][1], stage_ms[round][2]);
        }
        println!("total {:.0} ms; heap {:.0} MB before, peak {:.0} MB", now_ms() - tstart, crate::demand::alloc::mb(live0), crate::demand::alloc::mb(crate::demand::alloc::peak()));
        for q in 0..PERIODS {
            let mut seg = pops[0].seg(q);
            let mut board = pops[0].board(q);
            let mut alight = pops[0].alight(q);
            for (o, v) in seg.iter_mut().zip(pops[1].seg(q).iter().zip(ic.seg(q)).map(|(a, b)| a + b)) {
                *o += v;
            }
            for (o, v) in board.iter_mut().zip(pops[1].board(q).iter().zip(ic.board(q)).map(|(a, b)| a + b)) {
                *o += v;
            }
            for (o, v) in alight.iter_mut().zip(pops[1].alight(q).iter().zip(ic.alight(q)).map(|(a, b)| a + b)) {
                *o += v;
            }
            result.insert(format!("seg{q}"), serde_json::json!(seg));
            result.insert(format!("board{q}"), serde_json::json!(board));
            result.insert(format!("alight{q}"), serde_json::json!(alight));
            result.insert(format!("icboard{q}"), serde_json::json!(ic.board(q)));
            result.insert(format!("entries{q}"), serde_json::json!(pops[0].entries(q)));
            result.insert(format!("sum_nyc{q}"), serde_json::json!(pops[0].summary(q)));
            result.insert(format!("sum_phl{q}"), serde_json::json!(pops[1].summary(q)));
            result.insert(format!("sum_ic{q}"), serde_json::json!(ic.summary(q)));
            result.insert(format!("cm_nyc{q}"), serde_json::json!(pops[0].sub_sums().iter().map(|&v| v as f64).sum::<f64>()));
        }
        let sums: Vec<f64> = (0..PERIODS).map(|q| pops[0].summary(q)[2] + pops[1].summary(q)[2] + ic.summary(q)[2]).collect();
        println!("rail trips by period (all populations): {sums:?}");
        let now = serde_json::Value::Object(result);
        if let Ok(path) = std::env::var("TW_DUMP") {
            std::fs::write(&path, serde_json::to_string(&now).unwrap()).unwrap();
            println!("dumped to {path}");
        }
        if let Ok(path) = std::env::var("TW_COMPARE") {
            let base: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let flat = |v: &serde_json::Value| -> Vec<f64> {
                match v {
                    serde_json::Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect(),
                    other => vec![other.as_f64().unwrap_or(0.0)],
                }
            };
            let mut worst = 0.0f64;
            for (k, v) in now.as_object().unwrap() {
                let (a, b) = (flat(v), flat(&base[k]));
                assert_eq!(a.len(), b.len(), "{k} length");
                let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-9);
                let mut kmax = 0.0f64;
                let mut rel_total = 0.0;
                let (sa, sb): (f64, f64) = (a.iter().sum(), b.iter().sum());
                for (x, y) in a.iter().zip(&b) {
                    kmax = kmax.max((x - y).abs());
                }
                if sb.abs() > 0.0 {
                    rel_total = (sa - sb).abs() / sb.abs();
                }
                if k.starts_with("sum_") {
                    // ms (index 4) differs run to run
                    let mut m = 0.0f64;
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        if i != 4 {
                            m = m.max((x - y).abs() / y.abs().max(1.0));
                        }
                    }
                    kmax = m * scale;
                }
                worst = worst.max(kmax / scale);
                println!("{k}: max abs diff {kmax:.4e} = {:.2e} of max {scale:.3e}; sum differs {rel_total:.2e}", kmax / scale);
            }
            println!("WORST relative to each array's max: {worst:.3e}");
        }
    }

    /// Two crossing lines whose crossing stations are `gap` metres apart: riders change trains by
    /// walking (300 m by default) or, within 100 m, at one merged station.
    #[test]
    fn walking_transfers_and_merged_stations() {
        let run = |gap: f32, xfer: f32| {
            let (mut api, mut xy, n, stops, times, tph, cars) = small();
            api.params_mut().xfer_walk_m = xfer;
            // move line B (stations 17..) sideways and give it its own crossing station
            let b_cross = (xy.len() / 2) as u32;
            xy.extend([gap, 0.0]);
            let mut stops = stops;
            for s in stops[17..].iter_mut() {
                if *s == 8 {
                    *s = b_cross;
                }
            }
            for i in 17..xy.len() / 2 - 1 {
                xy[2 * i] = gap;
            }
            api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
            api.solve(0);
            let entries: f64 = api.entries(0).iter().map(|&v| v as f64).sum();
            let boards: f64 = api.board(0).iter().map(|&v| v as f64).sum();
            let n_st = api.station_map().iter().max().unwrap() + 1;
            (boards / entries, n_st)
        };
        let (none, st_far) = run(200.0, 0.0);
        let (walk, _) = run(200.0, 300.0);
        let (merged, st_near) = run(50.0, 0.0);
        assert!((none - 1.0).abs() < 1e-3, "no transfers without walking: {none}");
        assert!(walk > 1.02, "walking transfers: boardings / entries {walk}");
        assert!(merged > 1.02, "merged station transfers: {merged}");
        assert_eq!(st_far, st_near + 1);
    }

    /// T-176: transfers on the real New York network (nyc_real_track.json), commutes and the
    /// everyday purposes, periods 0 and 2, free flow and three crowding rounds, counted and not:
    /// the loads must be the same to the bit; prints the stage times both ways, the transfer
    /// arrays' size, and the biggest transfers at Penn Station and Grand Central.
    /// `cargo test --release --features demand transfers_real_network -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn transfers_real_network() {
        use crate::demand::everyday::EverydayApi;
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let mut a = DemandApi::new(&h, &b).unwrap();
        let mut e = EverydayApi::new(&h, &b).unwrap();
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (st_xy, line_n, stops, times, tph, cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        let names: Vec<String> = j["stations"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
        let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
        // route node -> snapshot station and line, and the fares' km
        let (mut rn_st, mut rn_line, mut hop_km) = (vec![], vec![], vec![]);
        let mut off = 0usize;
        for (l, &n) in line_n.iter().enumerate() {
            let n = n as usize;
            for dir in 0..2 {
                for i in 0..n {
                    let k = if dir == 0 { off + i } else { off + n - 1 - i };
                    rn_st.push(stops[k] as usize);
                    rn_line.push(l);
                    hop_km.push(if i + 1 < n {
                        let next = if dir == 0 { k + 1 } else { k - 1 };
                        let (s0, s1) = (stops[k] as usize, stops[next] as usize);
                        (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                    } else {
                        0.0
                    });
                }
            }
            off += n;
        }
        let nr = rn_st.len();
        let periods = [0usize, 2];
        let mut runs = vec![];
        for off in [true, false] {
            kernel::XFER_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
            a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            a.set_fares(2.9, 0.1, &hop_km).unwrap();
            e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            e.set_fares(2.9, 0.1, &hop_km).unwrap();
            let mut ms = vec![];
            for &q in &periods {
                for round in 0..4 {
                    let total: Vec<f32> = a.seg(q).iter().zip(e.seg(q)).map(|(x, y)| x + y).collect();
                    let t = now_ms();
                    if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                    let tc = now_ms() - t;
                    let t = now_ms();
                    if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                    ms.push((q, round, tc, now_ms() - t));
                    release_skims();
                }
            }
            let loads: Vec<Vec<f32>> = periods.iter().flat_map(|&q| [a.seg(q), a.board(q), a.alight(q), e.seg(q), e.board(q), e.alight(q)]).collect();
            runs.push((ms, loads));
        }
        kernel::XFER_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(runs[0].1, runs[1].1, "loads with and without transfers counted");
        println!("stage ms (period, round): commute off -> on, everyday off -> on");
        let (mut sum_off, mut sum_on) = (0.0, 0.0);
        for (x, y) in runs[0].0.iter().zip(&runs[1].0) {
            println!("  p{} r{}: commute {:.0} -> {:.0}, everyday {:.0} -> {:.0}", x.0, x.1, x.2, y.2, x.3, y.3);
            sum_off += x.2 + x.3;
            sum_on += y.2 + y.3;
        }
        println!("  all stages {sum_off:.0} -> {sum_on:.0} ms ({:+.1}%)", 100.0 * (sum_on / sum_off - 1.0));
        let st = a.per[0].as_ref().unwrap();
        // the assignment alone, period 0's last graph and the commute's matrix, best of 7 each
        {
            let sk = kernel::skims_for(&st.g, &st.p, &a.want);
            let er = kernel::station_exp(&sk, &st.p);
            let mo = kernel::mode_choice(&a.z, &a.gr, a.sz.as_ref().unwrap(), &er, st.g.n_st, &st.p);
            let (mut plain, mut counted) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..7 {
                let t = now_ms();
                let l0 = kernel::assign(&st.g, &sk, &mo.od);
                plain = plain.min(now_ms() - t);
                let mut x = vec![0.0; st.xi.len];
                let t = now_ms();
                let l1 = kernel::assign_with_transfers(&st.g, &sk, &mo.od, &st.xi, &mut x);
                counted = counted.min(now_ms() - t);
                assert_eq!(l0.seg, l1.seg);
            }
            println!("commute assignment, period 0: {plain:.1} ms, counting transfers {counted:.1} ms");
        }
        // whole free-flow stages, alternating, best of 5 (the round-by-round times above are noisy)
        let mut best = [[f64::INFINITY; 2]; 2];
        for _ in 0..5 {
            for (i, off) in [true, false].into_iter().enumerate() {
                kernel::XFER_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
                let t = now_ms();
                a.solve(0);
                best[0][i] = best[0][i].min(now_ms() - t);
                let t = now_ms();
                e.solve(0);
                best[1][i] = best[1][i].min(now_ms() - t);
                release_skims();
            }
        }
        kernel::XFER_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
        println!("free-flow stage, period 0, best of 5: commute {:.0} -> {:.0} ms, everyday {:.0} -> {:.0} ms (strategy search counted in the commute's)", best[0][0], best[0][1], best[1][0], best[1][1]);
        let st = a.per[0].as_ref().unwrap();
        let nz = st.xfer.iter().filter(|&&v| v > 0.0).count();
        println!("transfer slots {} ({} kB a period and population, plus {} kB index), {} carry riders in period 0 (commute)", st.xi.len, st.xi.len * 4 / 1024, st.xi.at.len() * 4 / 1024, nz);
        // every boarding after the first is a transfer (commutes, both periods)
        let all = vec![1u8; nr];
        let pairs = a.transfers(&all);
        let xfer_sum: f64 = pairs.chunks(3).map(|c| c[2] as f64).sum();
        let walked: f64 = pairs.chunks(3).filter(|c| rn_st[c[0] as usize] != rn_st[c[1] as usize] && a.svc.as_ref().unwrap().st_map[rn_st[c[0] as usize]] != a.svc.as_ref().unwrap().st_map[rn_st[c[1] as usize]]).map(|c| c[2] as f64).sum();
        let mut want = 0.0f64;
        for &q in &periods {
            want += a.board(q).iter().map(|&v| v as f64).sum::<f64>() - a.entries(q).iter().map(|&v| v as f64).sum::<f64>();
        }
        println!("commute transfers {xfer_sum:.0} ({walked:.0} walking between demand stations), boardings after the first {want:.0}");
        assert!((xfer_sum / want - 1.0).abs() < 1e-3);
        // the biggest at two hubs, both populations, both periods
        for hub in ["Penn Station", "Grand Central"] {
            let sel: Vec<u8> = rn_st.iter().map(|&s| names[s].contains(hub) as u8).collect();
            let mut by: std::collections::HashMap<(usize, usize), f64> = Default::default();
            let mut total = 0.0;
            for p in [a.transfers(&sel), e.transfers(&sel)] {
                for c in p.chunks(3) {
                    let (x, y) = (c[0] as usize, c[1] as usize);
                    if sel[x] != 0 && sel[y] != 0 {
                        *by.entry((rn_line[x], rn_line[y])).or_default() += c[2] as f64;
                        total += c[2] as f64;
                    }
                }
            }
            let mut v: Vec<_> = by.into_iter().collect();
            v.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap());
            println!("{hub}: {total:.0} transfers inside (periods 0 and 2), {} line pairs; biggest:", v.len());
            for ((x, y), r) in v.iter().take(8) {
                println!("  {} -> {}: {r:.0}", lines[*x], lines[*y]);
            }
        }
    }

    /// T-161: the real New York network (nyc_real_track.json) with and without its station
    /// complexes, commutes and the everyday purposes, every period, free flow and three crowding
    /// rounds. The complexes are T-007-real-nyc.save's as the game makes them
    /// (`checks/t161-complexes.mjs` writes them to ../artifacts/trainworld/t161/). Prints rail trips,
    /// boardings and transfers both ways, the transfers inside Penn Station, Grand Central and
    /// Times Sq, and the lines whose riders move most.
    /// `cargo test --release --features demand complexes_real_network -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn complexes_real_network() {
        use crate::demand::everyday::EverydayApi;
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let mut a = DemandApi::new(&h, &b).unwrap();
        let mut e = EverydayApi::new(&h, &b).unwrap();
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/../artifacts/trainworld/t161/nyc_real_complex.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (st_xy, line_n, stops, times, tph, cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        let stations = j["stations"].as_array().unwrap();
        let names: Vec<String> = stations.iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
        let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
        // the save's stations to the network's, by position to the millimetre
        let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
        let mut of: std::collections::HashMap<(i64, i64), u32> = Default::default();
        for s in cj["stations"].as_array().unwrap() {
            of.insert(key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap()), s["complex"].as_u64().unwrap() as u32);
        }
        let ns = stations.len();
        let mut unmatched = 0;
        let complex: Vec<u32> = stations.iter().enumerate().map(|(i, s)| match of.get(&key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap())) {
            Some(&c) => c,
            None => { unmatched += 1; (ns + i) as u32 }
        }).collect();
        println!("{ns} stations, {unmatched} not found in the save; {} save stations", of.len());
        assert_eq!(unmatched, 0);
        let (mut rn_st, mut rn_line, mut hop_km) = (vec![], vec![], vec![]);
        let mut off = 0usize;
        for (l, &n) in line_n.iter().enumerate() {
            let n = n as usize;
            for dir in 0..2 {
                for i in 0..n {
                    let k = if dir == 0 { off + i } else { off + n - 1 - i };
                    rn_st.push(stops[k] as usize);
                    rn_line.push(l);
                    hop_km.push(if i + 1 < n {
                        let next = if dir == 0 { k + 1 } else { k - 1 };
                        let (s0, s1) = (stops[k] as usize, stops[next] as usize);
                        (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                    } else {
                        0.0
                    });
                }
            }
            off += n;
        }
        let nr = rn_st.len();
        // a hub is the biggest complex holding a station of that name
        let hub = |name: &str| -> Vec<u8> {
            let mut size: std::collections::HashMap<u32, usize> = Default::default();
            for (i, n) in names.iter().enumerate() {
                if n.contains(name) {
                    *size.entry(complex[i]).or_default() += 0;
                }
            }
            for c in &complex {
                if let Some(v) = size.get_mut(c) { *v += 1; }
            }
            let best = size.into_iter().max_by_key(|&(c, n)| (n, std::cmp::Reverse(c))).unwrap().0;
            rn_st.iter().map(|&s| (complex[s] == best) as u8).collect()
        };
        let hubs: Vec<(&str, Vec<u8>)> = ["Penn Station", "Grand Central", "Times Sq"].into_iter().map(|n| (n, hub(n))).collect();
        // how many pairs of demand stations in one complex are beyond the walk limit: the walks
        // the complexes add
        {
            a.set_complexes(&complex);
            a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            let svc = a.svc.as_ref().unwrap();
            let lim = a.base.xfer_walk_m;
            let (mut pairs, mut beyond, mut far) = (0, 0, 0.0f32);
            for m in kernel::complex_members(&svc.group) {
                for (i, &p) in m.iter().enumerate() {
                    for &q in &m[i + 1..] {
                        let d = (svc.st_x[p as usize] - svc.st_x[q as usize]).hypot(svc.st_y[p as usize] - svc.st_y[q as usize]);
                        pairs += 1;
                        far = far.max(d);
                        if d > lim { beyond += 1; }
                    }
                }
            }
            println!("{} demand stations; {pairs} pairs of demand stations inside complexes, {beyond} beyond the {lim} m walk (farthest pair {far:.0} m)", svc.st_x.len());
            let ncx = complex.iter().collect::<std::collections::HashSet<_>>().len();
            println!("{ncx} complexes; complexes of more than one demand station: {}", kernel::complex_members(&svc.group).len());
            for (name, sel) in &hubs {
                let snap: std::collections::HashSet<usize> = rn_st.iter().zip(sel).filter(|p| *p.1 != 0).map(|p| *p.0).collect();
                let dem: std::collections::HashSet<u32> = snap.iter().map(|&s| svc.st_map[s]).collect();
                let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for &s in &snap { x0 = x0.min(st_xy[2 * s]); x1 = x1.max(st_xy[2 * s]); y0 = y0.min(st_xy[2 * s + 1]); y1 = y1.max(st_xy[2 * s + 1]); }
                println!("  {name}: {} stations with a line, {} demand station(s), centres spread {:.0} x {:.0} m", snap.len(), dem.len(), x1 - x0, y1 - y0);
            }
        }
        struct Run { rail: [f64; 2], board: f64, entries: f64, xfer: f64, walked: f64, hubs: Vec<(f64, Vec<((usize, usize), f64)>)>, line_riders: Vec<f64>, seg: Vec<f64>, max_crush: f64 }
        let mut runs: Vec<Run> = vec![];
        for with in [false, true] {
            let c: &[u32] = if with { &complex } else { &[] };
            a.set_complexes(c);
            e.set_complexes(c);
            a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            a.set_fares(2.9, 0.1, &hop_km).unwrap();
            e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            e.set_fares(2.9, 0.1, &hop_km).unwrap();
            let t0 = now_ms();
            for q in 0..PERIODS {
                for round in 0..4 {
                    let total: Vec<f32> = a.seg(q).iter().zip(e.seg(q)).map(|(x, y)| x + y).collect();
                    if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                    if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                    release_skims();
                }
            }
            println!("{} complexes: a day in {:.1} s", if with { "with" } else { "without" }, (now_ms() - t0) / 1000.0);
            let svc = a.svc.as_ref().unwrap();
            let (mut rail, mut board, mut entries, mut seg, mut line_riders, mut max_crush) = ([0.0; 2], 0.0, 0.0, vec![0.0f64; nr], vec![0.0f64; line_n.len()], 0.0f64);
            for q in 0..PERIODS {
                rail[0] += a.summary(q)[2];
                rail[1] += e.summary(q)[2];
                for v in [a.board(q), e.board(q)] {
                    for (r, &x) in v.iter().enumerate() {
                        board += x as f64;
                        line_riders[rn_line[r]] += x as f64;
                    }
                }
                entries += a.entries(q).iter().map(|&v| v as f64).sum::<f64>();
                for v in [a.seg(q), e.seg(q)] {
                    for (r, &x) in v.iter().enumerate() { seg[r] += x as f64; }
                }
                max_crush = max_crush.max(a.summary(q)[7]);
            }
            let all = vec![1u8; nr];
            let (mut xfer, mut walked) = (0.0, 0.0);
            for p in [a.transfers(&all), e.transfers(&all)] {
                for c in p.chunks(3) {
                    let (x, y) = (c[0] as usize, c[1] as usize);
                    xfer += c[2] as f64;
                    if svc.st_map[rn_st[x]] != svc.st_map[rn_st[y]] { walked += c[2] as f64; }
                }
            }
            let mut hub_out = vec![];
            for (_, sel) in &hubs {
                let mut by: std::collections::HashMap<(usize, usize), f64> = Default::default();
                let mut total = 0.0;
                for p in [a.transfers(sel), e.transfers(sel)] {
                    for c in p.chunks(3) {
                        let (x, y) = (c[0] as usize, c[1] as usize);
                        if sel[x] != 0 && sel[y] != 0 {
                            let k = (rn_line[x].min(rn_line[y]), rn_line[x].max(rn_line[y]));
                            *by.entry(k).or_default() += c[2] as f64;
                            total += c[2] as f64;
                        }
                    }
                }
                let mut v: Vec<_> = by.into_iter().collect();
                v.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap());
                hub_out.push((total, v));
            }
            runs.push(Run { rail, board, entries, xfer, walked, hubs: hub_out, line_riders, seg, max_crush });
        }
        let (o, n) = (&runs[0], &runs[1]);
        let pct = |x: f64, y: f64| 100.0 * (y / x - 1.0);
        println!("rail trips a day: commutes {:.0} -> {:.0} ({:+.2}%), everyday {:.0} -> {:.0} ({:+.2}%)", o.rail[0], n.rail[0], pct(o.rail[0], n.rail[0]), o.rail[1], n.rail[1], pct(o.rail[1], n.rail[1]));
        println!("boardings a day (both): {:.0} -> {:.0} ({:+.2}%); commute first boardings {:.0} -> {:.0}", o.board, n.board, pct(o.board, n.board), o.entries, n.entries);
        println!("transfers a day (both): {:.0} -> {:.0} ({:+.2}%), walking between demand stations {:.0} -> {:.0}", o.xfer, n.xfer, pct(o.xfer, n.xfer), o.walked, n.walked);
        println!("worst load of crush (commutes): {:.2} -> {:.2}", o.max_crush, n.max_crush);
        for (k, (name, _)) in hubs.iter().enumerate() {
            let ((t0, v0), (t1, v1)) = (&o.hubs[k], &n.hubs[k]);
            println!("{name}: {t0:.0} -> {t1:.0} changes a day inside the complex ({:+.1}%); biggest with complexes:", pct(*t0, *t1));
            for ((x, y), r) in v1.iter().take(6) {
                let before = v0.iter().find(|p| p.0 == (*x, *y)).map_or(0.0, |p| p.1);
                println!("  {} with {}: {before:.0} -> {r:.0}", lines[*x], lines[*y]);
            }
        }
        let mut moved: Vec<(usize, f64, f64)> = (0..line_n.len()).map(|l| (l, o.line_riders[l], n.line_riders[l])).collect();
        moved.sort_by(|p, q| (q.2 - q.1).abs().partial_cmp(&(p.2 - p.1).abs()).unwrap());
        println!("lines whose boardings a day moved most:");
        for &(l, x, y) in moved.iter().take(10) {
            println!("  {}: {x:.0} -> {y:.0} ({:+.1}%)", lines[l], pct(x, y));
        }
        let mut rel: Vec<f64> = moved.iter().filter(|m| m.1 > 1000.0).map(|m| (m.2 / m.1 - 1.0).abs()).collect();
        rel.sort_by(|p, q| p.partial_cmp(q).unwrap());
        println!("lines over 1,000 boardings a day: median change {:.2}%, 90th percentile {:.2}%, largest {:.1}%", 100.0 * rel[rel.len() / 2], 100.0 * rel[rel.len() * 9 / 10], 100.0 * rel[rel.len() - 1]);
        let mut sd: Vec<(usize, f64, f64)> = (0..nr).map(|r| (r, o.seg[r], n.seg[r])).collect();
        sd.sort_by(|p, q| (q.2 - q.1).abs().partial_cmp(&(p.2 - p.1).abs()).unwrap());
        println!("segments whose riders a day moved most:");
        for &(r, x, y) in sd.iter().take(6) {
            println!("  {} leaving {}: {x:.0} -> {y:.0}", lines[rn_line[r]], names[rn_st[r]]);
        }
    }

    /// T-188: the real New York network as the save has it (nyc_real_track.json with the save's
    /// complexes), commutes and the everyday purposes together as the workers solve them, every
    /// period, free flow and three crowding rounds, with and without the waits for full trains.
    /// Prints rail trips, riders left behind and their extra minutes (from the final averaged
    /// loads, by `kernel::left_behind`), the busiest lines and the lines that move most.
    /// `cargo test --release --features demand left_behind_real_network -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn left_behind_real_network() {
        use crate::demand::everyday::EverydayApi;
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let mut a = DemandApi::new(&h, &b).unwrap();
        let mut e = EverydayApi::new(&h, &b).unwrap();
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/../artifacts/trainworld/t161/nyc_real_complex.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (st_xy, line_n, stops, times, tph, cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        let stations = j["stations"].as_array().unwrap();
        let names: Vec<String> = stations.iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
        let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
        let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
        let mut of: std::collections::HashMap<(i64, i64), u32> = Default::default();
        for s in cj["stations"].as_array().unwrap() {
            of.insert(key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap()), s["complex"].as_u64().unwrap() as u32);
        }
        let ns = stations.len();
        let complex: Vec<u32> = stations.iter().enumerate().map(|(i, s)| of.get(&key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap())).copied().unwrap_or((ns + i) as u32)).collect();
        let (mut rn_st, mut rn_line, mut hop_km) = (vec![], vec![], vec![]);
        let mut off = 0usize;
        for (l, &n) in line_n.iter().enumerate() {
            let n = n as usize;
            for dir in 0..2 {
                for i in 0..n {
                    let k = if dir == 0 { off + i } else { off + n - 1 - i };
                    rn_st.push(stops[k] as usize);
                    rn_line.push(l);
                    hop_km.push(if i + 1 < n {
                        let next = if dir == 0 { k + 1 } else { k - 1 };
                        let (s0, s1) = (stops[k] as usize, stops[next] as usize);
                        (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                    } else {
                        0.0
                    });
                }
            }
            off += n;
        }
        struct Run { rail: [f64; 2], line: Vec<f64>, left: Vec<kernel::LeftStats>, worst: Vec<(f32, usize)>, ms: f64 }
        let mut runs = vec![];
        for on in [false, true] {
            a.params_mut().left_behind = on;
            e.params_mut().left_behind = on;
            a.set_complexes(&complex);
            e.set_complexes(&complex);
            a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            a.set_fares(2.9, 0.1, &hop_km).unwrap();
            e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            e.set_fares(2.9, 0.1, &hop_km).unwrap();
            let t0 = now_ms();
            for q in 0..PERIODS {
                for round in 0..4 {
                    let add = |x: Vec<f32>, y: Vec<f32>| -> Vec<f32> { x.iter().zip(y).map(|(p, q)| p + q).collect() };
                    let total = [add(a.seg(q), e.seg(q)), add(a.board(q), e.board(q)), add(a.alight(q), e.alight(q))].concat();
                    if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                    if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                    release_skims();
                }
            }
            let ms = now_ms() - t0;
            let (mut rail, mut line, mut left, mut worst) = ([0.0; 2], vec![0.0f64; line_n.len()], vec![], vec![]);
            for q in 0..PERIODS {
                rail[0] += a.summary(q)[2];
                rail[1] += e.summary(q)[2];
                let (board, alight): (Vec<f32>, Vec<f32>) = (a.board(q).iter().zip(e.board(q)).map(|(x, y)| x + y).collect(), a.alight(q).iter().zip(e.alight(q)).map(|(x, y)| x + y).collect());
                for (r, &x) in board.iter().enumerate() {
                    line[rn_line[r]] += x as f64;
                }
                let st = a.period(q).unwrap();
                let (w, s) = kernel::left_behind(&st.g, &st.net, &st.p, &board, &alight);
                left.push(s);
                let (mut wv, mut wr) = (0.0f32, 0usize);
                for (r, &v) in w.iter().enumerate() {
                    if v > wv { wv = v; wr = r; }
                }
                worst.push((wv, wr));
            }
            runs.push(Run { rail, line, left, worst, ms });
        }
        let (o, n) = (&runs[0], &runs[1]);
        let pct = |x: f64, y: f64| 100.0 * (y / x - 1.0);
        println!("a day: {:.1} s without, {:.1} s with the waits", o.ms / 1000.0, n.ms / 1000.0);
        println!("rail trips a day: commutes {:.0} -> {:.0} ({:+.3}%), everyday {:.0} -> {:.0} ({:+.3}%)", o.rail[0], n.rail[0], pct(o.rail[0], n.rail[0]), o.rail[1], n.rail[1], pct(o.rail[1], n.rail[1]));
        for q in 0..PERIODS {
            let (x, y) = (o.left[q], n.left[q]);
            println!(
                "period {q}: left behind {:.0} -> {:.0} riders, {:.0} -> {:.0} extra rider-minutes, at {} -> {} route nodes; worst average extra wait {:.1} min ({} leaving {}) -> {:.1} min ({} leaving {})",
                x.left, y.left, x.rider_min, y.rider_min, x.stops, y.stops,
                o.worst[q].0, lines[rn_line[o.worst[q].1]], names[rn_st[o.worst[q].1]], n.worst[q].0, lines[rn_line[n.worst[q].1]], names[rn_st[n.worst[q].1]]
            );
        }
        let mut busiest: Vec<usize> = (0..line_n.len()).collect();
        busiest.sort_by(|&x, &y| o.line[y].partial_cmp(&o.line[x]).unwrap());
        println!("busiest lines, boardings a day (both populations):");
        for &l in busiest.iter().take(12) {
            println!("  {}: {:.0} -> {:.0} ({:+.2}%)", lines[l], o.line[l], n.line[l], pct(o.line[l], n.line[l]));
        }
        let mut moved: Vec<usize> = (0..line_n.len()).filter(|&l| o.line[l] > 1000.0).collect();
        moved.sort_by(|&x, &y| (n.line[y] - o.line[y]).abs().partial_cmp(&(n.line[x] - o.line[x]).abs()).unwrap());
        println!("lines whose boardings moved most:");
        for &l in moved.iter().take(10) {
            println!("  {}: {:.0} -> {:.0} ({:+.2}%)", lines[l], o.line[l], n.line[l], pct(o.line[l], n.line[l]));
        }
        let by_feed = |r: &Run, f: &str| (0..line_n.len()).filter(|&l| lines[l].starts_with(f)).map(|l| r.line[l]).sum::<f64>();
        for f in ["subway", "lirr", "mnr", "njt", "path"] {
            println!("{f}: boardings {:.0} -> {:.0} ({:+.2}%)", by_feed(o, f), by_feed(n, f), pct(by_feed(o, f), by_feed(n, f)));
        }
    }

    #[test]
    fn empty_network_is_all_road_and_walk() {
        let (mut api, ..) = small();
        assert!(api.warm_up() > 0.0);
        assert!(api.summary(0).is_empty() && !api.solve(0));
        api.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
        assert!(api.solve(0));
        let s = api.summary(0);
        assert_eq!(s[2], 0.0);
        assert!(s[3] > 0.0);
        assert!(api.set_network(&[0.0, 0.0], &[2], &[0, 1], &[0.0; 12], &[1.0, 1.0, 1.0], &[1.0]).is_err());
    }
}
