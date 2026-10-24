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
//! Commutes only; the everyday outings are `everyday.rs`, the regional market `intercity.rs`. A
//! period carries its share of the day's trips to work and trips home. Mode choice runs home -> work; the trips home are the same
//! station pairs reversed (riders go back the way they came, which is exact for park-and-ride),
//! so a period's rail matrix is `to_work x OD + to_home x OD^T`, scaled from one solve.

use super::clock::now_ms;
use super::days::{day_rate, load_factor, share_out, share_ret, WEEKDAY, WORK};
use super::kernel::*;
use super::network::{Network, PERIODS};
use super::pack::{self, City};
use crate::track::params::{LEVELS, PERIOD_HOURS, PERIOD_LEVEL};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Share of the day's trips to work that start in each period (order of `PERIOD_HOURS`: 6-10,
/// 10-16, 16-20, 20-24, 0-6). Judgement after US travel-survey commute start times; notes/T-026.md.
pub const TO_WORK: [f32; PERIODS] = [0.70, 0.12, 0.06, 0.04, 0.08];
/// Same for trips home.
pub const TO_HOME: [f32; PERIODS] = [0.02, 0.18, 0.58, 0.16, 0.06];
/// Busiest hour of a period against its mean hour (crowding is judged at the busiest hour).
pub const PEAK_HOUR_FACTOR: [f32; PERIODS] = crate::track::params::PEAK_HOUR_FACTOR;
/// A 20 m car: seats and riders at crush load (a New York subway car: ~44 seats, ~160 at crush).
pub const SEATS_PER_CAR: f32 = 44.0;
pub const CRUSH_PER_CAR: f32 = 160.0;
/// The rail propensity (docs/demand.md 4.3), one worldwide value fitted on six cities' real
/// networks (T-509), 0 since T-636 (the rail share comes from time and money alone); the everyday
/// layer adds `Params::everyday_asc_rail`, 0 as well.
pub const ASC_RAIL: f32 = 0.0;
/// What a country adds to `ASC_RAIL` (ISO 3166-1 alpha-2, the pack's `country`), for its
/// commutes, everyday trips and the regional layer's commutes from its zones; any other country
/// adds 0. Empty since T-636 (Japan's +0.5 went); the hook stays for a later country fit.
pub const COUNTRY_ASC_RAIL: &[(&str, f32)] = &[];
pub fn country_asc_rail(country: &str) -> f32 {
    COUNTRY_ASC_RAIL.iter().find(|(c, _)| *c == country).map_or(0.0, |&(_, v)| v)
}
/// US value of time: $20/hour, or 3 perceived minutes per dollar (T-067).
pub const FARE_MIN_PER_USD: f32 = 3.0;

/// Checks a fare curve's knots (T-883): pairs of trip km and US$, km rising, fares never falling;
/// fewer than two pairs is the straight line of `set_fares`.
pub(crate) fn check_fare_curve(knots: &[f32]) -> Result<(), String> {
    let ok = knots.len() % 2 == 0
        && knots.iter().all(|v| v.is_finite() && *v >= 0.0)
        && knots.chunks(2).zip(knots.chunks(2).skip(1)).all(|(a, b)| b[0] > a[0] && b[1] >= a[1]);
    if ok { Ok(()) } else { Err("invalid fare curve".into()) }
}

/// Gives `g` the fare curve `knots` (US$, `check_fare_curve`) over the straight line the ride
/// edges and base fare already charge (perceived minutes), with each route node's fare km.
pub(crate) fn put_fare_curve(g: &mut Graph, knots: &[f32], hop_km: &[f32], fare_min: f32, per_km: f32) {
    if knots.len() < 4 {
        return;
    }
    g.fare_curve = FareCurve {
        km: knots.iter().step_by(2).copied().collect(),
        min: knots.iter().skip(1).step_by(2).map(|v| v * FARE_MIN_PER_USD).collect(),
        base: fare_min,
        rate: per_km,
    };
    g.rn_km = (0..g.rn_fare.len()).map(|r| hop_km.get(r).copied().unwrap_or(0.0)).collect();
}

/// `Loads::rn_fare_x` in US$.
pub(crate) fn fare_x_usd(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / FARE_MIN_PER_USD).collect()
}

/// `flows` leaves out the far cells holding this share of the selection's commuters, the
/// smallest first (they go to the cut totals), to keep the answer small (T-098).
pub const FLOW_CUT: f64 = 0.01;
/// Upper edges of `DemandApi::commute_walk_bands`'s distance bands, km (the everyday table's).
pub const WALK_BANDS_KM: [f32; 8] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

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
    /// Each stop's platform, x and y: its snapshot station's position (`Network::stop_x`).
    pub(crate) stop_xy: Vec<f32>,
    /// Seconds, 6 per stop: per run and level, `(run * 3 + level) * n + k` from the line's
    /// offset x 6. Run 0: stop k to k + 1 (0 at the last); run 1: stop k to k - 1 (0 at the first).
    /// On a circle (T-798) the last of run 0 and the first of run 1 are the stand at the first stop,
    /// riding on round.
    pub(crate) times: Vec<f32>,
    pub(crate) tph: Vec<[f32; LEVELS]>,
    pub(crate) cars: Vec<f32>,
    /// Each demand station's complex (`demand_groups`), empty if none were given.
    pub(crate) group: Vec<u32>,
    /// `Network::span` (T-842), empty for 1 everywhere.
    pub(crate) span: Vec<f32>,
}

/// `span` if it is one per period and route node of `stops` stops, else empty (T-842).
pub(crate) fn span_for(span: &[f32], stops: usize) -> Vec<f32> {
    if span.len() == PERIODS * 2 * stops && span.iter().any(|&v| v != 1.0) { span.iter().map(|v| v.clamp(0.0, 1.0)).collect() } else { vec![] }
}

/// Equal to the bit (T-124: a kept period must be exactly what a fresh solve would give).
pub(crate) fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

impl Service {
    /// Stations, lines, stops, cars and complexes equal: station access, subzones and the
    /// served pairs come out the same, since they never read run times or trains an hour.
    pub(crate) fn same_layout(&self, o: &Service) -> bool {
        same_bits(&self.st_x, &o.st_x)
            && same_bits(&self.st_y, &o.st_y)
            && self.st_map == o.st_map
            && self.line_off == o.line_off
            && self.stops == o.stops
            && same_bits(&self.stop_xy, &o.stop_xy)
            && same_bits(&self.cars, &o.cars)
            && self.group == o.group
    }
    /// T-124: the periods (bit q) whose inputs are the same in `o`: the same layout, and the run
    /// times and trains an hour of the period's level. A period reads no other level's.
    pub(crate) fn same_periods(&self, o: &Service) -> u32 {
        if !self.same_layout(o) {
            return 0;
        }
        let level_same = |lev: usize| {
            self.tph.iter().zip(&o.tph).all(|(a, b)| a[lev].to_bits() == b[lev].to_bits())
                && self.line_off.windows(2).all(|w| {
                    let (a, n) = (w[0] as usize, (w[1] - w[0]) as usize);
                    [lev, 3 + lev].iter().all(|&r| same_bits(&self.times[6 * a + r * n..6 * a + (r + 1) * n], &o.times[6 * a + r * n..6 * a + (r + 1) * n]))
                })
        };
        let same: [bool; LEVELS] = std::array::from_fn(level_same);
        let nr = 2 * self.stops.len();
        let span_same = |q: usize| {
            fn at(s: &Service, q: usize, nr: usize) -> Option<&[f32]> {
                if s.span.is_empty() { None } else { Some(&s.span[q * nr..(q + 1) * nr]) }
            }
            match (at(self, q, nr), at(o, q, nr)) {
                (None, None) => true,
                (Some(a), Some(b)) => same_bits(a, b),
                (Some(a), None) | (None, Some(a)) => a.iter().all(|&v| v == 1.0),
            }
        };
        (0..PERIODS).filter(|&q| same[PERIOD_LEVEL[q]] && span_same(q)).fold(0, |m, q| m | 1 << q)
    }
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
    /// The segment loads the last crowding round was priced on (`kernel::settled`, T-434).
    seen: Vec<f32>,
    /// Transfers platform to platform (T-176): the slots of `g`, averaged like the loads.
    xi: XferIndex,
    xfer: Vec<f32>,
}

/// A pass's rail trips by place (`kernel::ModeOut`), both directions of the period's commutes
/// counted at their home and work ends.
pub(crate) struct SubOut {
    /// Rail trips by home/work subzone, and the walkers rail takes by home/work subzone (`ModeOut`).
    pub(crate) rail_home: Vec<f32>,
    rail_work: Vec<f32>,
    walk_lost_home: Vec<f32>,
    walk_lost_work: Vec<f32>,
    /// Per access entry (`Subzones::acc_st`): rail trips using that station from that subzone
    /// at the home end, and at the work end.
    acc_home: Vec<f32>,
    acc_work: Vec<f32>,
    /// Rail trips by (home station, work station) before the trips home are reversed, over the
    /// stations in the subzones' access lists only (`DemandApi::od_st`; every other station's row
    /// and column are zero, and a city a few stations reach would hold a whole network's S x S
    /// otherwise, T-398).
    pub(crate) od: Vec<f32>,
}

impl SubOut {
    fn empty() -> SubOut {
        SubOut { rail_home: vec![], rail_work: vec![], walk_lost_home: vec![], walk_lost_work: vec![], acc_home: vec![], acc_work: vec![], od: vec![] }
    }
    fn msa(&mut self, new: &SubOut, round: usize) {
        msa(&mut self.rail_home, &new.rail_home, round);
        msa(&mut self.rail_work, &new.rail_work, round);
        msa(&mut self.walk_lost_home, &new.walk_lost_home, round);
        msa(&mut self.walk_lost_work, &new.walk_lost_work, round);
        msa(&mut self.acc_home, &new.acc_home, round);
        msa(&mut self.acc_work, &new.acc_work, round);
        msa(&mut self.od, &new.od, round);
    }
}

/// T-558: all-day trips with no rail that walk, by home zone and by work zone
/// (`Gravity::no_rail_by_zone`).
fn no_rail_trips(z: &Zones, gr: &Gravity, p: &Params, nr: &NoRail) -> (Vec<f64>, Vec<f64>) {
    let [wh, ww] = gr.no_rail_by_zone(z, p, nr);
    (wh, ww)
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
    /// Shared with the city's everyday layer (`with_weights`, T-399).
    pub(crate) city: Arc<City>,
    pub(crate) z: Arc<Zones>,
    pub(crate) gr: Arc<Gravity>,
    /// The home and work weights station access and subzones are built on in place of
    /// `city.pop` and `city.jobs` (`with_weights`); None for the commutes.
    weights: Option<(Vec<f32>, Vec<f32>)>,
    pub(crate) base: Params,
    /// What the zone pairs' choice with no rail reads (T-558): the commutes' walking with no
    /// rail, by zone pair, for the totals and the pair loop alike.
    pub(crate) nr: Arc<NoRail>,
    /// All-day home -> work trips that would walk with no rail (`Gravity::no_rail_by_zone`), and
    /// the same by home zone and by work zone.
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
    /// The stations in the subzones' access lists, ascending, and each station's place among them
    /// (`u32::MAX` if absent): the rows and columns of `SubOut::od`.
    od_st: Vec<u32>,
    od_at: Vec<u32>,
    pub(crate) per: Vec<Option<Period>>,
    pub(crate) fare_per_km: f32,
    pub(crate) fare_km: Vec<f32>,
    /// The fare curve's knots, km and US$ in pairs (T-883, `set_fare_curve`); empty for the straight line.
    pub(crate) fare_curve: Vec<f32>,
    open_ms: f64,
    /// The complex of each station of the next `set_network` (`set_complexes`, T-161).
    complex_in: Vec<u32>,
    /// The service span of the next `set_network` (`set_span`, T-842).
    span_in: Vec<f32>,
    /// T-124: the periods (bit q) whose inputs the last `set_network` and `set_fares` left as they
    /// were, so their results were kept; and whether the layout was, so access was kept.
    pub(crate) same: u32,
    pub(crate) layout_same: bool,
    /// Whether `svc` and the periods were built with the parameters as they stand.
    reusable: bool,
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
            stop_x: svc.stop_xy.iter().step_by(2).copied().collect(),
            stop_y: svc.stop_xy.iter().skip(1).step_by(2).copied().collect(),
            line_wrap: vec![false; nl],
            span: svc.span.clone(),
        };
        for l in 0..nl {
            let (a, b) = (svc.line_off[l] as usize, svc.line_off[l + 1] as usize);
            let n = b - a;
            for k in 0..n {
                net.hop_min[a + k] = svc.times[6 * a + lev * n + k] / 60.0;
                net.hop_back[a + k] = svc.times[6 * a + (3 + lev) * n + k] / 60.0;
            }
            // T-798: a circle is a line whose last stop is its first, with a time to ride on
            // round from it (`Service::times`)
            net.line_wrap[l] = n >= 3 && svc.stops[a] == svc.stops[b - 1] && svc.times[6 * a + lev * n + n - 1] > 0.0;
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
            let mo = mode_choice(&self.z, &self.gr, sz, &er, &sk.walk, &sk.walk_max, g.n_st, p, &self.nr);
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
            od: self.compact_od(&mo.od, g.n_st),
        };
        (mo.rail, mo.rail_walk, l, mo.pairs, sub)
    }

    /// An S x S station-pair table over `od_st` only (`SubOut::od`).
    pub(crate) fn compact_od(&self, od: &[f32], s: usize) -> Vec<f32> {
        let k = self.od_st.len();
        let mut out = Vec::with_capacity(k * k);
        for &a in &self.od_st {
            let row = &od[a as usize * s..(a as usize + 1) * s];
            out.extend(self.od_st.iter().map(|&t| row[t as usize]));
        }
        out
    }

    /// `sub.od` at stations (a, t): 0 where either is in no access list.
    #[inline]
    fn sub_od(&self, sub: &SubOut, a: usize, t: usize) -> f32 {
        let (i, j) = (self.od_at[a], self.od_at[t]);
        if i == u32::MAX || j == u32::MAX { 0.0 } else { sub.od[i as usize * self.od_st.len() + j as usize] }
    }
}

/// Readers of a solved period for tests and the bins; not exported to the app.
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
    /// Trips walking the whole way in period `q` by home cell: the home zone's walking with no
    /// rail spread by people, less what rail takes from each subzone (T-621's calibration).
    pub fn walk_by_cell(&self, q: usize) -> Vec<f32> {
        let Some(st) = self.period(q) else { return vec![] };
        let share = self.params(q).period_share;
        (0..self.city.len())
            .map(|c| {
                let z = self.z.cell_zone[c] as usize;
                let walk = (self.walk_home_zone[z] / self.z.pop[z].max(1e-9)) as f32 * self.city.pop[c] * share;
                let lost = match self.sz.as_ref().map(|sz| (sz, sz.cell[c])) {
                    Some((sz, u)) if u != u32::MAX && sz.pop[u as usize] > 0.0 && !st.sub.walk_lost_home.is_empty() => st.sub.walk_lost_home[u as usize] * self.city.pop[c] / sz.pop[u as usize],
                    _ => 0.0,
                };
                (walk - lost).max(0.0)
            })
            .collect()
    }
    /// The route-node layout: per route node, its station and line (kernel order).
    pub fn route_nodes(&self, q: usize) -> (Vec<u32>, Vec<u32>) {
        self.period(q).map_or((vec![], vec![]), |st| (st.g.rn_station.clone(), st.g.rn_line.clone()))
    }
    /// Snapshot station -> demand station (stations within `station_merge_m` are one).
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    /// The parameters every period starts from (to change before `set_network`, which then
    /// keeps nothing).
    pub fn params_mut(&mut self) -> &mut Params {
        self.reusable = false;
        &mut self.base
    }
    /// Rebuild the walk's costs and the trips walking with no rail after changing
    /// their parameters through `params_mut` (the gravity holds them from `new`); before
    /// `EverydayApi::beside`.
    pub fn refresh_walk(&mut self) {
        let mut gr = (*self.gr).clone();
        gr.set_walk(&self.z, &self.base);
        self.nr = Arc::new(NoRail::new(&self.z, &self.base));
        gr.set_walk_fine(&self.city, &self.z, &self.base, &self.nr);
        let (home, work) = no_rail_trips(&self.z, &gr, &self.base, &self.nr);
        self.walk_all = home.iter().sum();
        self.walk_home_zone = home;
        self.walk_work_zone = work;
        self.gr = Arc::new(gr);
        self.reusable = false;
    }
    /// Commutes with no rail between zones within `WALK_FINE_R` squares, by the straight-line
    /// distance of six cell pairs a zone pair (`walk_points`; bands `WALK_BANDS_KM`, then beyond):
    /// trips, then those walking: each cell pair's own share when the pair walks between cells
    /// (`Params::walk_samples`), the pair's when between centres (T-587's check).
    pub fn commute_walk_bands(&self) -> Vec<f64> {
        let (z, gr, p) = (&*self.z, &*self.gr, &self.base);
        let nb = WALK_BANDS_KM.len() + 1;
        let mut out = vec![0f64; 2 * nb];
        let ns = 6;
        let (home, work) = walk_points(&self.city, z, ns);
        let fine = !gr.wf_off.is_empty();
        let sq = z.by_square();
        let (mut d, mut e) = (vec![0f32; ns * ns], vec![0f32; ns * ns]);
        for i in 0..z.n {
            if gr.row[i] == 0.0 || home[i * ns] == u32::MAX {
                continue;
            }
            z.near(&sq, i, WALK_FINE_R, |j| {
                if work[j * ns] == u32::MAX {
                    return;
                }
                let k = (gr.base(z, i) + gr.key(z, j)) as usize;
                let t = gr.t(z, i, j) / (ns * ns) as f64;
                let w = self.nr.split(gr, p, i, j, k);
                let (road, paid, free) = self.nr.road(gr, p, i, j, k);
                for (n, (&a, &b)) in home[i * ns..(i + 1) * ns].iter().flat_map(|a| work[j * ns..(j + 1) * ns].iter().map(move |b| (a, b))).enumerate() {
                    d[n] = cell_walk_m(&self.city, a, b) / 1000.0;
                    let f = gr.walk_factor_cells(z, a as usize, b as usize);
                    e[n] = if fine { p.walk_no_rail(road, paid, free, gr.wcost.at(d[n] * f)) } else { w };
                }
                for n in 0..ns * ns {
                    let band = WALK_BANDS_KM.partition_point(|&b| b <= d[n]);
                    out[band] += t;
                    out[nb + band] += t * e[n] as f64;
                }
            });
        }
        out
    }
    /// The same city on other home and work weights (the everyday layer's residents and jobs,
    /// T-399): the cells, zone grid, gravity and routes are shared, the zones' sums are its own,
    /// and station access and subzones are built on these weights. Its network starts unset.
    pub(crate) fn with_weights(&self, pop: Vec<f32>, jobs: Vec<f32>) -> DemandApi {
        let mut z = (*self.z).clone();
        z.pop.fill(0.0);
        z.jobs.fill(0.0);
        for c in 0..self.city.len() {
            let zi = z.cell_zone[c] as usize;
            z.pop[zi] += pop[c] as f64;
            z.jobs[zi] += jobs[c] as f64;
        }
        DemandApi {
            city: self.city.clone(),
            z: Arc::new(z),
            gr: self.gr.clone(),
            weights: Some((pop, jobs)),
            base: self.base.clone(),
            nr: self.nr.clone(),
            walk_all: self.walk_all,
            walk_home_zone: self.walk_home_zone.clone(),
            walk_work_zone: self.walk_work_zone.clone(),
            home_k: self.home_k.clone(),
            work_k: self.work_k.clone(),
            svc: None,
            sz: None,
            want: vec![],
            od_st: vec![],
            od_at: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            fare_per_km: 0.0,
            fare_km: vec![],
            fare_curve: vec![],
            open_ms: 0.0,
            complex_in: vec![],
            span_in: vec![],
            same: 0,
            layout_same: false,
            reusable: false,
        }
    }
    /// The home weights access and subzones are built on (`with_weights`).
    pub(crate) fn pop(&self) -> &[f32] {
        self.weights.as_ref().map_or(&self.city.pop, |w| &w.0)
    }
    pub(crate) fn jobs(&self) -> &[f32] {
        self.weights.as_ref().map_or(&self.city.jobs, |w| &w.1)
    }
    /// Subzones of the current network (0 with none).
    pub fn n_subzones(&self) -> u32 {
        self.sz.as_ref().map_or(0, |sz| sz.n as u32)
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
    /// Step 1, every worker: the selected cells' rail trips over this worker's solved periods,
    /// three blocks of `n_subzones`: by subzone at the far end; and for the selected subzones at
    /// their own end, rail trips and the walkers rail takes (whole subzones, not
    /// scaled to the selection; `flows` does that per cell, as `cell_modes` does). Trips, both
    /// directions: add every worker's blocks and pass the sum to `flows`. Riders boarding at a
    /// station are assumed to spread over the stations they ride to like all its riders, and
    /// riders leaving at a station over the places they walk to like all of them (as
    /// `station_riders`): exact at the near end, a little blurred at the far end. Empty with no
    /// network.
    pub fn flow_rail(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        self.flow_rail_day(end, cells, WEEKDAY as u32)
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
                base.decay_flat_km = pg.flat_km;
            }
        }
        let (z, mut gr, gravity_from_pack) = city_gravity(&city, &base);
        let h: serde_json::Value = serde_json::from_str(header).map_err(|e| e.to_string())?;
        if let (Some(lon), Some(lat)) = (h["origin"]["lon"].as_f64(), h["origin"]["lat"].as_f64()) {
            base.cordon = cordon_by_zone(&city, &z, (lon, lat)).map(Arc::new);
        }
        // T-275: the pack's routed walks and drives, if it has them for these zones
        if let Some(r) = super::routes::Routes::from_pack(header, bin, &z)? {
            base.routes = Some(Arc::new(r));
            gr.set_walk(&z, &base);
        }
        // T-854: residents per km² by zone, for parking at the home end (the pack's `nocar` is
        // not read)
        if let Some(pop) = pack::cell_weights(header, bin, "pop", city.len())? {
            base.home_dens = Some(Arc::new(z.home_density(&pop)));
        }
        // T-509 stage 4: the everyday rates of the pack's country; an older block's `decay_m` is
        // ignored, the decay lengths being one worldwide set (T-637)
        for (kind, v) in h["everyday"]["rate"].as_object().into_iter().flatten() {
            if let Some(x) = v.as_f64() {
                base.set_extra(&format!("rate_{kind}"), x as f32)?;
            }
        }
        // the pack's country: the header's own, else the one its everyday block was written for
        let country = h["country"].as_str().or_else(|| h["everyday"]["country"].as_str()).unwrap_or("");
        base.set_extra("country_asc_rail", country_asc_rail(country))?;
        let nr = Arc::new(NoRail::new(&z, &base));
        gr.set_walk_fine(&city, &z, &base, &nr);
        let (walk_home_zone, walk_work_zone) = no_rail_trips(&z, &gr, &base, &nr);
        let walk_all = walk_home_zone.iter().sum();
        let (home_k, work_k) = zone_trip_factors(&city, &z, &gr, gravity_from_pack);
        Ok(DemandApi { city: Arc::new(city), z: Arc::new(z), gr: Arc::new(gr), weights: None, base, nr, walk_all, walk_home_zone, walk_work_zone, home_k, work_k, svc: None, sz: None, want: vec![], od_st: vec![], od_at: vec![], per: (0..PERIODS).map(|_| None).collect(), fare_per_km: 0.0, fare_km: vec![], fare_curve: vec![], open_ms: now_ms() - t0, complex_in: vec![], span_in: vec![], same: 0, layout_same: false, reusable: false })
    }

    /// The station complex of each station of the networks `set_network` is given from now on
    /// (T-161): one number per station, equal numbers one complex (game/complexes.ts, with the
    /// player's links). Riders change between any two stations of a complex on foot, the walk
    /// costed by distance like any walking transfer. Empty (the default) or a list of another
    /// length: no complexes, only the walks within `xfer_walk_m`.
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.complex_in = complex.to_vec();
    }

    /// The service span of the networks `set_network` is given from now on (T-842): per period
    /// and route node (`q * 2 x stops + r`, each line's stops in order then reversed, as `hopKm`),
    /// the share of the period's riders there with a train due within their tolerance
    /// (`track::span`). Empty or another length: 1 everywhere.
    pub fn set_span(&mut self, span: &[f32]) {
        self.span_in = span.to_vec();
    }

    /// T-874: commute zone pairs under `COMMUTE_CHUNK` trips a period dealt in chunks
    /// (`Params::commute_chunk`), or every pair solved (`?perfOff=chunks`).
    pub fn set_chunked(&mut self, on: bool) {
        let c = if on { COMMUTE_CHUNK } else { 0.0 };
        if c != self.base.commute_chunk {
            self.params_mut().commute_chunk = c;
        }
    }

    /// Cells, zones, home -> work trips a day, the walk share of all trips with no rail at all,
    /// and the ms the constructor took.
    pub fn info(&self) -> Vec<f64> {
        vec![self.city.len() as f64, self.z.n as f64, self.gr.total, self.walk_all / self.gr.total, self.open_ms]
    }

    /// The network to solve on, packed: station x, y pairs (metres in the pack's frame); stops
    /// per line; the stops (station indices) line after line; 6 times per stop (seconds, see
    /// `Service::times`); trains an hour, 3 per line (high, medium, low); cars per line.
    /// Rebuilds station access and subzones (shared by every period) unless stations, lines,
    /// stops, cars and complexes are unchanged, and forgets the results of every period whose
    /// level's run times or trains an hour changed (T-124; all of them with a new layout).
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
            stop_xy: stops.iter().flat_map(|&s| [st_xy[2 * s as usize], st_xy[2 * s as usize + 1]]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            cars: cars.to_vec(),
            group,
            span: span_for(&self.span_in, nstops),
        };
        let old = self.svc.as_ref().filter(|_| self.reusable);
        self.reusable = true;
        self.layout_same = old.is_some_and(|o| svc.same_layout(o));
        self.same = old.map_or(0, |o| svc.same_periods(o));
        for (q, p) in self.per.iter_mut().enumerate() {
            match p {
                Some(st) if self.same >> q & 1 == 1 => st.net = Self::period_net(&svc, q),
                _ => *p = None,
            }
        }
        if !self.layout_same {
            self.fare_km.clear();
            self.sz = None;
            self.want.clear();
            self.od_st.clear();
            self.od_at.clear();
            if ns > 0 && nl > 0 {
                let net = Self::period_net(&svc, 0);
                let acc = build_access(&self.city, &net, &self.base);
                let sz = build_subzones_on(&self.city, self.pop(), self.jobs(), &self.z, &acc, &net, &self.base);
                self.want = egress_stations(&sz, svc.st_x.len());
                self.od_at = vec![u32::MAX; svc.st_x.len()];
                for &a in &sz.acc_st {
                    self.od_at[a as usize] = 0;
                }
                for (a, at) in self.od_at.iter_mut().enumerate() {
                    if *at == 0 {
                        *at = self.od_st.len() as u32;
                        self.od_st.push(a as u32);
                    }
                }
                self.sz = Some(sz);
            }
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
        let (fare_min, per_km) = (base_usd * FARE_MIN_PER_USD, per_km_usd * FARE_MIN_PER_USD);
        if fare_min.to_bits() != self.base.fare_min.to_bits() || per_km.to_bits() != self.fare_per_km.to_bits() || !same_bits(hop_km, &self.fare_km) {
            self.base.fare_min = fare_min;
            self.fare_per_km = per_km;
            self.fare_km = hop_km.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
            self.same = 0;
        }
        Ok(())
    }

    /// The player's fare curve (T-883, T-1143) over `set_fares`' straight line: trip km and US$ in
    /// pairs, km rising and fares never falling; empty when the curve is that line.
    pub fn set_fare_curve(&mut self, knots: &[f32]) -> Result<(), String> {
        check_fare_curve(knots)?;
        if !same_bits(knots, &self.fare_curve) {
            self.fare_curve = knots.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
            self.same = 0;
        }
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
        self.od_st.clear();
        self.od_at.clear();
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
        put_fare_curve(&mut g, &self.fare_curve, &self.fare_km, self.base.fare_min, self.fare_per_km);
        let no_crowding = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &no_crowding);
        let xi = xfer_index(&g);
        let mut xfer = vec![0.0; xi.len];
        let (rail, rail_walk, avg, pairs, sub) = if self.sz.is_some() {
            self.pass(&g, &p, q, Some((&xi, &mut xfer)))
        } else {
            (0.0, 0.0, Loads::zero(g.n_st, g.n - g.n_st), 0, SubOut::empty())
        };
        self.per[q] = Some(Period { net, p, g, avg, rail, walk: walk0 - rail_walk, trips, round: 0, ms: now_ms() - t0, pairs, sub, fac: no_crowding, uniform: true, seen: vec![], xi, xfer });
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
            if settled(&mut st.seen, &total[..st.g.rn_line.len()]) || (st.uniform && fac == st.fac && left == st.g.rn_left) {
                // The period's loads have settled (T-434), or nothing is crowded enough to change a
                // cost, so this round would solve what the round before did (T-119).
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
            msa(&mut st.avg.rn_fare_x, &l.rn_fare_x, r);
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
    /// The fare curve's extra in period `q` (T-1143), US$, at each route node where the trips paying
    /// it first board: what the trips pay beyond the base and the first stretch's rate per km.
    pub fn fare_extra(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|p| p.as_ref()).map_or(vec![], |st| fare_x_usd(&st.avg.rn_fare_x))
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


    /// Rail trips by subzone summed over this worker's solved periods, four blocks of
    /// `n_subzones`: rail by home/work, then the walkers rail takes by home/work. Add the blocks of
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
        // An empty answer would leave the demand views on the previous network's (T-099).
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
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != self.od_st.len() * self.od_st.len() {
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
                        to_zone[zu] += self.sub_od(sub, a, t) * sub.acc_work[k] / arr[t] * f;
                    }
                    if dep[t] > 0.0 {
                        from_zone[zu] += self.sub_od(sub, t, a) * sub.acc_home[k] / dep[t] * f;
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
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != self.od_st.len() * self.od_st.len() {
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
                    let k = self.od_st.len();
                    let i = self.od_at[a] as usize;
                    for (&t, &x) in self.od_st.iter().zip(&sub.od[i * k..(i + 1) * k]) {
                        far_st[t as usize] += f * x;
                    }
                } else {
                    let f = q[a] / arr[a];
                    for (b, o) in far_st.iter_mut().enumerate() {
                        *o += f * self.sub_od(sub, b, a);
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
        let wt = self.gr.walk_set;
        let nr = &self.nr;
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
                if wt && self.gr.walk_c(idx) < NO_WALK {
                    w_z[b] += t * nr.split(&self.gr, &self.base, i, j, idx) as f64;
                }
            }
        }
        for v in 0..ns {
            r_z[szone[v] as usize] += sums[v] as f64 / 2.0;
        }
        // the near end counted the walkers rail takes exactly (`lost`); the far end's estimate, from
        // rail by zone, is scaled to match
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
                out.push(self.z.x0 + self.z.hx(i) as f32 * 0.5 * zm);
                out.push(self.z.y0 + self.z.hy(i) as f32 * 0.5 * zm);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests;
