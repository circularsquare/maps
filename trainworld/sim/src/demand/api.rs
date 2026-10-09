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
/// `flows` leaves out the far cells holding this share of the selection's commuters, the
/// smallest first (they go to the cut totals), to keep the answer small (T-098).
pub const FLOW_CUT: f64 = 0.01;

/// One demand station for snapshot stations within `merge_m` of each other (single linkage), so
/// two lines' platforms built side by side or on top of each other (a station node takes one
/// route's track, so crossing lines need a node each) are one station for access and change
/// trains there without a walk. Two stations next to each other on one line are never merged.
/// Returns the merged positions (means) and the map snapshot station -> demand station.
fn merge_stations(st_xy: &[f32], line_off: &[u32], stops: &[u32], merge_m: f32) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
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

/// The network as the snapshot packs it (`set_network`).
struct Service {
    /// Demand stations: the snapshot's stations, those within `Params::station_merge_m` of each
    /// other merged into one (`st_map`: snapshot station -> demand station).
    st_x: Vec<f32>,
    st_y: Vec<f32>,
    st_map: Vec<u32>,
    line_off: Vec<u32>,
    stops: Vec<u32>,
    /// Seconds, 6 per stop: per run and level, `(run * 3 + level) * n + k` from the line's
    /// offset x 6. Run 0: stop k to k + 1 (0 at the last); run 1: stop k to k - 1 (0 at the first).
    times: Vec<f32>,
    tph: Vec<[f32; LEVELS]>,
    cars: Vec<f32>,
}

struct Period {
    net: Network,
    p: Params,
    g: Graph,
    /// Loads averaged over the rounds so far.
    avg: Loads,
    rail: f64,
    walk: f64,
    trips: f64,
    round: usize,
    ms: f64,
    pairs: u64,
    /// Where the rail trips come from and go to, averaged like the loads (T-078's demand views).
    sub: SubOut,
}

/// A pass's rail trips by place (`kernel::ModeOut`), both directions of the period's commutes
/// counted at their home and work ends.
struct SubOut {
    /// Rail trips by home subzone and by work subzone, and the walk trips they took (`ModeOut`).
    rail_home: Vec<f32>,
    rail_work: Vec<f32>,
    walk_lost_home: Vec<f32>,
    walk_lost_work: Vec<f32>,
    /// Per access entry (`Subzones::acc_st`): rail trips using that station from that subzone
    /// at the home end, and at the work end.
    acc_home: Vec<f32>,
    acc_work: Vec<f32>,
    /// Rail trips by (home station, work station), S x S, before the trips home are reversed.
    od: Vec<f32>,
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

#[wasm_bindgen]
pub struct DemandApi {
    city: City,
    z: Zones,
    gr: Gravity,
    base: Params,
    /// All-day home -> work trips that would walk with no rail (`Gravity::walk_trips`), and the
    /// same by home zone and by work zone.
    walk_all: f64,
    walk_home_zone: Vec<f64>,
    walk_work_zone: Vec<f64>,
    svc: Option<Service>,
    sz: Option<Subzones>,
    per: Vec<Option<Period>>,
    open_ms: f64,
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
    fn period_net(svc: &Service, q: usize) -> Network {
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
    /// trips rail took, the loads, the pairs and the rail trips by place.
    fn pass(&self, g: &Graph, p: &Params, q: usize) -> (f64, f64, Loads, u64, SubOut) {
        let sz = self.sz.as_ref().unwrap();
        let sk = skims(g, p);
        let er = station_exp(&sk, p);
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
        let l = assign(g, &sk, &od);
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
    fn period(&self, q: usize) -> Option<&Period> {
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
        let p = self.params(q);
        let tp: f64 = self.city.pop.iter().map(|&v| v as f64).sum();
        let k = (self.gr.total / tp * p.period_share as f64) as f32;
        self.city.pop.iter().map(|&v| v * k).collect()
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
        let base = Params { asc_rail: ASC_RAIL, ..Params::default() };
        let (z, gr, _) = city_gravity(&city, &base);
        let (walk_home_zone, walk_work_zone) = gr.walk_trips_by_zone(&z);
        let walk_all = walk_home_zone.iter().sum();
        Ok(DemandApi { city, z, gr, base, walk_all, walk_home_zone, walk_work_zone, svc: None, sz: None, per: (0..PERIODS).map(|_| None).collect(), open_ms: now_ms() - t0 })
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
        let svc = Service {
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            cars: cars.to_vec(),
        };
        self.per.iter_mut().for_each(|p| *p = None);
        self.sz = None;
        if ns > 0 && nl > 0 {
            let net = Self::period_net(&svc, 0);
            let acc = build_access(&self.city, &net, &self.base);
            self.sz = Some(build_subzones(&self.city, &self.z, &acc, &net, &self.base));
        }
        self.svc = Some(svc);
        Ok(now_ms() - t0)
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
        let g = build_graph(&net, &p);
        let (rail, rail_walk, avg, pairs, sub) = if self.sz.is_some() {
            self.pass(&g, &p, q)
        } else {
            (0.0, 0.0, Loads::zero(g.n_st, g.n - g.n_st), 0, SubOut::empty())
        };
        self.per[q] = Some(Period { net, p, g, avg, rail, walk: walk0 - rail_walk, trips, round: 0, ms: now_ms() - t0, pairs, sub });
        true
    }

    /// One crowding round for period `q` (a penalty per segment from the averaged loads, then
    /// search, mode choice and assignment again, averaged in). False if `q` has no solve yet.
    pub fn crowd(&mut self, q: usize) -> bool {
        let t0 = now_ms();
        let Some(mut st) = self.per.get_mut(q).and_then(|p| p.take()) else { return false };
        if self.sz.is_some() {
            let (fac, _) = crowd_factors(&st.g, &st.net, &st.avg, &st.p);
            set_crowding(&mut st.g, &fac);
            let (rail, rail_walk, l, _, sub) = self.pass(&st.g, &st.p, q);
            st.round += 1;
            let r = st.round;
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

    /// Subzones of the current network (0 with none).
    pub fn n_subzones(&self) -> u32 {
        self.sz.as_ref().map_or(0, |sz| sz.n as u32)
    }

    /// Rail trips by subzone summed over this worker's solved periods, four blocks of
    /// `n_subzones`: by home, by work, walk trips they took by home, by work. Add the blocks of
    /// every worker and pass the sum to `cell_modes`.
    pub fn sub_sums(&self) -> Vec<f32> {
        let ns = self.n_subzones() as usize;
        let mut out = vec![0f32; 4 * ns];
        for q in 0..PERIODS {
            let Some(st) = self.period(q) else { continue };
            if st.sub.rail_home.len() != ns {
                continue;
            }
            for (b, v) in [&st.sub.rail_home, &st.sub.rail_work, &st.sub.walk_lost_home, &st.sub.walk_lost_work].into_iter().enumerate() {
                for (o, &x) in out[b * ns..(b + 1) * ns].iter_mut().zip(v) {
                    *o += x;
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
        let n = self.city.len();
        // No subzones (a network with no running line): nobody by rail, everyone walks or drives.
        // This used to answer empty, and the demand views kept the previous network's answer
        // (T-099: the real New York save's bubbles stayed after a new game).
        let sz = self.sz.as_ref();
        let ns = sz.map_or(0, |s| s.n);
        if sums.len() != 4 * ns {
            return vec![];
        }
        let tp: f64 = self.city.pop.iter().map(|&v| v as f64).sum();
        let tj: f64 = self.city.jobs.iter().map(|&v| v as f64).sum();
        let (kh, kw) = ((self.gr.total / tp.max(1e-9)) as f32, (self.gr.total / tj.max(1e-9)) as f32);
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
            let total = pop * kh;
            let mut walk = if self.z.pop[zc] > 0.0 { (self.walk_home_zone[zc] / self.z.pop[zc]) as f32 * pop } else { 0.0 };
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
            let total = jobs * kw;
            let mut walk = if self.z.jobs[zc] > 0.0 { (self.walk_work_zone[zc] / self.z.jobs[zc]) as f32 * jobs } else { 0.0 };
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
    /// `[n_home, n_work, home cells.., their riders.., work cells.., their riders.., work zones
    /// of the home riders (one per zone).., home zones of the work riders (one per zone)..]`.
    /// Home riders live in those cells and use this station at the home end of their commute;
    /// work riders work there and use it at the work end. Cell and zone indices are exact in f32
    /// (under 2^24). Empty if there is no network or no such station.
    pub fn station_riders(&self, station: u32) -> Vec<f32> {
        let (Some(sz), Some(svc)) = (self.sz.as_ref(), self.svc.as_ref()) else { return vec![] };
        let Some(&a) = svc.st_map.get(station as usize) else { return vec![] };
        let a = a as usize;
        let s = svc.st_x.len();
        let nz = self.z.n;
        let (mut home_r, mut work_r) = (vec![0f32; sz.n], vec![0f32; sz.n]);
        let (mut to_zone, mut from_zone) = (vec![0f32; nz], vec![0f32; nz]);
        let (mut dep, mut arr) = (vec![0f32; s], vec![0f32; s]);
        for q in 0..PERIODS {
            let Some(st) = self.period(q) else { continue };
            let sub = &st.sub;
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != s * s {
                continue;
            }
            dep.iter_mut().for_each(|v| *v = 0.0);
            arr.iter_mut().for_each(|v| *v = 0.0);
            for (k, &t) in sz.acc_st.iter().enumerate() {
                dep[t as usize] += sub.acc_home[k];
                arr[t as usize] += sub.acc_work[k];
            }
            for u in 0..sz.n {
                for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                    let t = sz.acc_st[k] as usize;
                    if t == a {
                        home_r[u] += sub.acc_home[k];
                        work_r[u] += sub.acc_work[k];
                    }
                    // where this station's home riders work: their trips to station t, spread
                    // over the subzones arriving at t; and where its work riders live
                    let zu = sz.zone[u] as usize;
                    if arr[t] > 0.0 {
                        to_zone[zu] += sub.od[a * s + t] * sub.acc_work[k] / arr[t];
                    }
                    if dep[t] > 0.0 {
                        from_zone[zu] += sub.od[t * s + a] * sub.acc_home[k] / dep[t];
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
        let mut out = Vec::with_capacity(2 + 2 * (hc.len() + wc.len()) + 2 * nz);
        out.extend([hc.len() as f32, wc.len() as f32]);
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
    /// their own end, all their rail trips and the walk trips rail took (whole subzones, not
    /// scaled to the selection; `flows` does that per cell, as `cell_modes` does). Trips, both
    /// directions: add every worker's blocks and pass the sum to `flows`. Riders boarding at a
    /// station are assumed to spread over the stations they ride to like all its riders, and
    /// riders leaving at a station over the places they walk to like all of them (as
    /// `station_riders`): exact at the near end, a little blurred at the far end. Empty with no
    /// network.
    pub fn flow_rail(&self, end: u32, cells: &[u32]) -> Vec<f32> {
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
            for u in 0..ns {
                if su[u] > 0.0 {
                    out[ns + u] += near_rail[u];
                    out[2 * ns + u] += near_lost[u];
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
                out[v] += x;
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
        let tot_w: f64 = w.iter().map(|&v| v as f64).sum();
        let k = (self.gr.total / tot_w.max(1e-9)) as f32;
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
            let total = w[c] * k;
            let mut wk = (walk_zone[zc] / wzone[zc]) as f32 * w[c];
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
                let t = wz[a] * self.gr.row[i] * self.gr.col[j] * self.gr.ftab[idx] as f64;
                t_z[b] += t;
                if wt {
                    w_z[b] += t * self.gr.wtab[idx] as f64;
                }
            }
        }
        for v in 0..ns {
            r_z[szone[v] as usize] += sums[v] as f64 / 2.0;
        }
        // the walk trips rail took, spread by rail and the walk share to each far zone
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
    /// Gravity zone of each cell.
    pub fn cell_zone(&self) -> Vec<u32> {
        self.z.cell_zone.clone()
    }
    /// The zones: side in metres, then the centre x, y of each zone (metres from the pack origin).
    pub fn zone_xy(&self) -> Vec<f32> {
        let zm = self.z.zone_m;
        let mut out = vec![zm];
        for i in 0..self.z.n {
            out.push(self.z.x0 + (self.z.gx[i] as f32 + 0.5) * zm);
            out.push(self.z.y0 + (self.z.gy[i] as f32 + 0.5) * zm);
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
            gravity: None,
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
            let (home, work) = (sum(2 + nh, nh), sum(2 + 2 * nh + nw, nw));
            let o = 2 + 2 * nh + 2 * nw;
            assert_eq!(r.len(), o + 2 * nz);
            assert!((sum(o, nz) / home - 1.0).abs() < 1e-3 && (sum(o + nz, nz) / work - 1.0).abs() < 1e-3, "station {st}");
            (home, work, r[2] as usize)
        };
        // the line's end and the shared centre station
        let (h_end, _, c_end) = riders(0);
        let (h_mid, w_mid, _) = riders(8);
        assert!(h_end > 0.0 && h_mid > 0.0 && w_mid > 0.0);
        let (x, y) = (api.cell_xy()[2 * c_end], api.cell_xy()[2 * c_end + 1]);
        assert!((x - xy[0]).hypot(y - xy[1]) < api.base.walk_cutoff_m + 100.0, "a catchment cell beyond the walk's bound");
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
                if si == 3 {
                    assert_eq!(f[0], 0.0, "no rail far from the lines");
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
