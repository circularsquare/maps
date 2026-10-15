//! T-195: the trip explorer's query. One trip between two points, as the commute model sees it in
//! one solved period: the two places' subzones, the walk share, the drive, the rail logsum over
//! access and egress stations, the logit between them, and the routes the rail riders take
//! through the period's strategies. Every number is the model's own: the choice is computed with
//! the same expressions and in the same order as `kernel::mode_choice` (so the same f32 results),
//! on the period's own graph (its last crowding round's costs), and the routes are walked through
//! the same strategies `kernel::assign` loads. Nothing here changes any state: a query between
//! solves, like the demand views' (SPEC 4.4, docs/demand.md "The trip explorer").
//!
//! The commute model chooses a mode per (home subzone, work subzone) pair, so a trip is read as a
//! commute from a home at the first point to work at the second; its return rides the same
//! stations back (`api.rs`). Rail is a choice among strategies, not one path: the trip's riders
//! spread over access and egress stations by a logit (`station_theta`), and at each station over
//! the useful departures by frequency. The answer lists the station pairs with their shares and
//! the routes (one per sequence of boardings) with theirs, biggest first.

use super::api::{DemandApi, Period, FARE_MIN_PER_USD};
use super::kernel::*;
use wasm_bindgen::prelude::*;

const NONE: u32 = u32::MAX;
/// A point takes the nearest cell whose centre is within this many metres; farther, it is outside
/// the city's cells (water, or past the pack's edge).
pub const NEAR_CELL_M: f32 = 600.0;
/// Station pairs carrying less than this share of the trip's rail riders are not walked for routes.
const PAIR_MIN: f32 = 0.005;
/// Routes carrying less than this share of the trip's rail riders are left out.
const ROUTE_MIN: f32 = 0.005;
/// At most this many routes are listed.
const MAX_ROUTES: usize = 8;

/// Numbers in the answer's head, per station pair, and per leg (`DemandApi::trip`).
pub const HEAD: usize = 32;
pub const PAIR_LEN: usize = 14;
pub const LEG_LEN: usize = 12;

/// Why rail is not offered for a trip (`HEAD` slot 26).
pub const RAIL_OFFERED: u32 = 0;
pub const NO_STATION_AT_START: u32 = 1;
pub const NO_STATION_AT_END: u32 = 2;
pub const WALKS_TOO_LONG: u32 = 3;
pub const NO_ROUTE: u32 = 4;

/// The model's choice for one (origin subzone, destination subzone) pair in a period, computed as
/// `kernel::mode_choice` computes it.
pub(crate) struct PairChoice {
    /// Share of all trips reserved for walking (by the two zones' distance).
    pub pw: f32,
    /// Straight line between the two subzones' centres, metres: the drive's distance.
    pub d_m: f32,
    /// Real minutes of the drive, and its parking at the destination (perceived minutes).
    pub drive_min: f32,
    pub park: f32,
    /// The drive's whole cost, perceived minutes.
    pub road: f32,
    /// `RAIL_OFFERED`, `WALKS_TOO_LONG` or `NO_ROUTE`.
    pub why: u32,
    /// The cheapest perceived walk at each end (the logsum's offsets).
    pub m_u: f32,
    pub m_v: f32,
    /// The logsum's sum over station pairs, and the rail cost it gives (perceived minutes).
    pub ssum: f32,
    pub rail: f32,
    /// Utility of rail against the road, and rail's share of the trips that do not walk.
    pub util: f32,
    pub r: f32,
    /// (access entry, egress entry, weight) over every station pair (`Subzones::acc_st` entries):
    /// weight / `ssum` is the pair's share of the trip's rail riders.
    pub pairs: Vec<(usize, usize, f32)>,
}

/// Homes and jobs per km² of land in zone `i`, as `Zones::density`.
fn zone_density(z: &Zones, i: usize) -> f32 {
    ((z.pop[i] + z.jobs[i]) / (z.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32
}

/// Parking at a destination in zone `j`, perceived minutes, as `mode_choice`.
fn park_at(z: &Zones, p: &Params, j: usize) -> f32 {
    let zone_area = (z.zone_m / 1000.0).powi(2);
    let dens = z.jobs[j] as f32 / zone_area;
    p.park_min + (dens / p.park_jobs_per_km2_per_min).min(p.park_cost_cap_min)
}

/// The walk share between zones `i` and `j` (`Gravity::wtab`; zero before it is set).
fn walk_share(api: &DemandApi, i: usize, j: usize) -> f32 {
    // T-275: around water with the pack's routes
    api.gr.walk_share_pair(&api.z, i, j)
}

/// The choice for subzones `u` (home) and `v` (work) with period parameters `p` and strategies
/// `sk` (searched at least towards `v`'s egress stations). Same expressions, same order, same
/// f32 results as `kernel::mode_choice`'s pair loop.
pub(crate) fn pair_choice(api: &DemandApi, p: &Params, sk: &Skims, u: usize, v: usize) -> PairChoice {
    let sz = api.sz.as_ref().expect("a network with subzones");
    let z = &api.z;
    let th = p.station_theta;
    let s = sk.s;
    let (i, j) = (sz.zone[u] as usize, sz.zone[v] as usize);
    let ends = |r: std::ops::Range<usize>| {
        let m = sz.acc_w[r.clone()].iter().cloned().fold(f32::INFINITY, f32::min);
        (r, m)
    };
    // T-371: a long commute's home end walks or rides
    let by_ride = !sz.ride(u).is_empty() && ride_offered(z, p, i, j);
    let (ru, m_u) = ends(if by_ride { sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize } else { sz.walk(u) });
    let (rv, m_v) = ends(sz.walk(v));
    let acc_e = |k: usize, m: f32| fast_exp(-th * (sz.acc_w[k] - m));
    // `station_exp`: exp(-theta r), 0 where there is no route and on the diagonal
    let er = |a: usize, t: usize| {
        let r = sk.r[a * s + t];
        if a == t || !r.is_finite() { 0.0 } else { fast_exp(-th * r) }
    };
    // the origin's row of the logsum, at the destination's egress stations only
    let mut ssum = 0.0f32;
    let mut pairs = Vec::with_capacity(ru.len() * rv.len());
    for kt in rv.clone() {
        let t = sz.acc_st[kt] as usize;
        let mut a_row = 0.0f32;
        for ka in ru.clone() {
            a_row += acc_e(ka, m_u) * er(sz.acc_st[ka] as usize, t);
        }
        let eg = acc_e(kt, m_v);
        ssum += a_row * eg;
        for ka in ru.clone() {
            pairs.push((ka, kt, acc_e(ka, m_u) * er(sz.acc_st[ka] as usize, t) * eg));
        }
    }
    let d = ((sz.cx[u] - sz.cx[v]).powi(2) + (sz.cy[u] - sz.cy[v]).powi(2)).sqrt();
    let drive_min = p.drive_min_f(d, p.drive_factor(i, j), p.place_min_per_km(zone_density(z, i)), p.place_min_per_km(zone_density(z, j)));
    let park = park_at(z, p, j);
    let road = drive_min * p.w_drive + park;
    let pw = walk_share(api, i, j);
    let why = if m_v > p.pair_walk_max - m_u {
        WALKS_TOO_LONG
    } else if ssum < 1e-30 {
        NO_ROUTE
    } else {
        RAIL_OFFERED
    };
    let (rail, util, r) = if why == RAIL_OFFERED {
        let rail = m_u + m_v + p.fare_min - fast_ln(ssum) / th;
        let util = p.beta * (road - rail) + p.asc_rail;
        (rail, util, 1.0 / (1.0 + fast_exp(-util)))
    } else {
        (f32::INFINITY, f32::NEG_INFINITY, 0.0)
    };
    PairChoice { pw, d_m: d, drive_min, park, road, why, m_u, m_v, ssum, rail, util, r, pairs }
}

/// One leg of a route: a ride (kind 1) from boarding to alighting route node, or a walk between
/// stations (kind 2) after a ride.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leg {
    pub kind: u8,
    /// route nodes (from 0, the order of the results' arrays): a ride's boarding and alighting
    /// ones; a walk's alighting one and one at the station walked to
    pub rn_from: u32,
    pub rn_to: u32,
    /// demand stations at the two ends
    pub st_from: u32,
    pub st_to: u32,
    /// real minutes riding or walking, and their perceived minutes (a ride's with its crowding,
    /// without its distance fare)
    pub min: f32,
    pub perc: f32,
    /// a ride's distance fare, perceived minutes
    pub fare: f32,
    /// a ride's wait at its boarding station: real minutes (half the combined headway of the
    /// useful departures there, plus the line's average extra wait for full trains there, T-188)
    /// and perceived minutes (`board_cost` less the transfer penalty, plus that extra wait)
    pub wait_min: f32,
    pub wait_perc: f32,
    /// stops ridden
    pub stops: u32,
}

/// Every route of the strategy from station `a` to station `t` carrying at least `min_share` of
/// what leaves `a` (each with its share): at a station riders split over the useful departures by
/// their frequency, and a route node follows its best edge, as `kernel::assign` loads them.
pub(crate) fn routes(g: &Graph, sk: &Skims, p: &Params, a: usize, t: usize, min_share: f32) -> Vec<(f32, Vec<Leg>)> {
    let mut out = vec![];
    let slot = sk.slot[t];
    if slot == NONE || a == t {
        return out;
    }
    let (next, board) = (&sk.next[slot as usize].to_vec(), &sk.board_tph(g, slot as usize));
    let mut rep = vec![NONE; g.n_st];
    for (r, &st) in g.rn_station.iter().enumerate().rev() {
        rep[st as usize] = r as u32;
    }
    let mut legs = vec![];
    walk(g, p, next, board, &rep, t, a, 1.0, min_share, &mut legs, &mut out);
    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    out
}

#[allow(clippy::too_many_arguments)]
fn walk(g: &Graph, p: &Params, next: &[u32], board: &[f32], rep: &[u32], t: usize, v: usize, share: f32, min_share: f32, legs: &mut Vec<Leg>, out: &mut Vec<(f32, Vec<Leg>)>) {
    if v == t {
        out.push((share, legs.clone()));
        return;
    }
    if legs.len() > 40 {
        return; // a guard: strategies have no cycles
    }
    let s = g.n_st;
    let edges = g.adj_off[v] as usize..g.adj_off[v + 1] as usize;
    let sum: f32 = edges.clone().map(|e| board[g.adj_to[e] as usize - s]).sum();
    if sum == 0.0 {
        return;
    }
    let headway = 60.0 / sum;
    let wait_min = headway / 2.0;
    let wait_perc = board_cost(headway, p) - p.transfer_min;
    for e in edges {
        let r0 = g.adj_to[e] as usize;
        let sh = share * board[r0 - s] / sum;
        if board[r0 - s] == 0.0 || sh < min_share {
            continue;
        }
        let (mut r, mut min, mut perc, mut fare, mut stops) = (r0, 0.0f32, 0.0f32, 0.0f32, 0u32);
        let mut nx = next[r];
        while nx != NONE && (nx as usize) >= s && stops as usize <= g.n {
            let ride = (g.adj_off[r] as usize..g.adj_off[r + 1] as usize).find(|&k| g.adj_seg[k] != NONE).unwrap();
            min += g.rn_ride[r - s];
            fare += g.rn_fare[r - s];
            perc += g.adj_cost[ride] - g.rn_fare[r - s];
            stops += 1;
            r = nx as usize;
            nx = next[r];
        }
        if nx == NONE {
            continue;
        }
        let here = g.rn_station[r - s];
        // and the extra wait where full trains leave riders behind (T-188)
        let left = g.rn_left[r0 - s];
        legs.push(Leg { kind: 1, rn_from: (r0 - s) as u32, rn_to: (r - s) as u32, st_from: v as u32, st_to: here, min, perc, fare, wait_min: wait_min + left, wait_perc: wait_perc + left * p.w_wait, stops });
        let mut pushed = 1;
        if nx != here {
            // a walking transfer from this route node to a station nearby
            let cost = (g.adj_off[r] as usize..g.adj_off[r + 1] as usize).find(|&k| g.adj_to[k] == nx).map_or(0.0, |k| g.adj_cost[k]);
            legs.push(Leg { kind: 2, rn_from: (r - s) as u32, rn_to: rep[nx as usize], st_from: here, st_to: nx, min: cost / p.w_xfer, perc: cost, fare: 0.0, wait_min: 0.0, wait_perc: 0.0, stops: 0 });
            pushed += 1;
        }
        walk(g, p, next, board, rep, t, nx as usize, sh, min_share, legs, out);
        legs.truncate(legs.len() - pushed);
    }
}

impl DemandApi {
    /// The cell nearest (x, y) (pack frame) within `NEAR_CELL_M`.
    pub(crate) fn nearest_cell(&self, x: f32, y: f32) -> Option<usize> {
        let mut best = (NEAR_CELL_M * NEAR_CELL_M, None);
        for c in 0..self.city.len() {
            let d2 = (self.city.x[c] - x).powi(2) + (self.city.y[c] - y).powi(2);
            if d2 <= best.0 {
                best = (d2, Some(c));
            }
        }
        best.1
    }

    fn explain(&self, st: &Period, q: usize, o: usize, d: usize) -> Vec<f64> {
        let p = &st.p;
        let city = &self.city;
        let mut h = vec![0f64; HEAD];
        h[0] = 1.0;
        h[1] = st.round as f64;
        h[2] = o as f64;
        h[3] = d as f64;
        h[6] = ((city.x[o] - city.x[d]).powi(2) + (city.y[o] - city.y[d]).powi(2)).sqrt() as f64;
        h[22] = p.fare_min as f64;
        h[23] = FARE_MIN_PER_USD as f64;
        h[27] = city.x[o] as f64;
        h[28] = city.y[o] as f64;
        h[29] = city.x[d] as f64;
        h[30] = city.y[d] as f64;
        let _ = q;
        let (u, v) = match self.sz.as_ref() {
            Some(sz) => (sz.cell[o], sz.cell[d]),
            None => (NONE, NONE),
        };
        h[4] = if u == NONE { -1.0 } else { u as f64 };
        h[5] = if v == NONE { -1.0 } else { v as f64 };
        if u == NONE || v == NONE {
            // no rail: the drive between the two cells, as the model would price it
            let (i, j) = (self.z.cell_zone[o] as usize, self.z.cell_zone[d] as usize);
            let dm = h[6] as f32;
            let drive = p.drive_min_f(dm, p.drive_factor(i, j), p.place_min_per_km(zone_density(&self.z, i)), p.place_min_per_km(zone_density(&self.z, j)));
            let park = park_at(&self.z, p, j);
            let pw = walk_share(self, i, j);
            h[7] = dm as f64;
            h[8] = pw as f64;
            h[9] = drive as f64;
            h[10] = (drive * p.w_drive) as f64;
            h[11] = park as f64;
            h[12] = (drive * p.w_drive + park) as f64;
            h[17] = pw as f64;
            h[19] = (1.0 - pw) as f64;
            h[26] = if u == NONE { NO_STATION_AT_START } else { NO_STATION_AT_END } as f64;
            h[31] = 1.0;
            return h;
        }
        let (u, v) = (u as usize, v as usize);
        let sz = self.sz.as_ref().unwrap();
        // strategies towards the destination's egress stations only (each search is independent,
        // so they are the model's, T-119)
        let mut want = vec![false; st.g.n_st];
        for k in sz.walk(v) {
            want[sz.acc_st[k] as usize] = true;
        }
        let sk = skims_for(&st.g, p, &want);
        let pc = pair_choice(self, p, &sk, u, v);
        let rail = (1.0 - pc.pw) * pc.r;
        h[7] = pc.d_m as f64;
        h[8] = pc.pw as f64;
        h[9] = pc.drive_min as f64;
        h[10] = (pc.drive_min * p.w_drive) as f64;
        h[11] = pc.park as f64;
        h[12] = pc.road as f64;
        h[13] = (pc.why == RAIL_OFFERED) as u32 as f64;
        h[14] = if pc.rail.is_finite() { pc.rail as f64 } else { -1.0 };
        h[15] = pc.r as f64;
        h[16] = if pc.util.is_finite() { pc.util as f64 } else { -1e9 };
        h[17] = pc.pw as f64;
        h[18] = rail as f64;
        h[19] = ((1.0 - pc.pw) * (1.0 - pc.r)) as f64;
        h[20] = pc.m_u as f64;
        h[21] = pc.m_v as f64;
        h[26] = pc.why as f64;
        if pc.why != RAIL_OFFERED {
            h[31] = 1.0;
            return h;
        }
        // station pairs, biggest share first
        let mut pairs: Vec<(usize, usize, f32)> = pc.pairs.iter().filter(|e| e.2 > 0.0).map(|&(a, t, w)| (a, t, w / pc.ssum)).collect();
        pairs.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
        let s = st.g.n_st;
        let mut rep = vec![-1f64; s];
        for (r, &stn) in st.g.rn_station.iter().enumerate().rev() {
            rep[stn as usize] = r as f64;
        }
        let mut body = vec![];
        for &(ka, kt, share) in &pairs {
            let (a, t) = (sz.acc_st[ka] as usize, sz.acc_st[kt] as usize);
            // a ride entry (T-371): its perceived minutes less the drive's are the ride's fixed cost and the station end
            let ride = sz.ride(u).contains(&ka);
            let station_end = if ride { sz.acc_w[ka] - p.w_drive * sz.acc_t[ka] } else { 0.0 };
            body.extend([a as f64, t as f64, rep[a], rep[t], sz.acc_d[ka] as f64, sz.acc_w[ka] as f64, sz.acc_d[kt] as f64, sz.acc_w[kt] as f64, sk.r[a * s + t] as f64, share as f64, sz.acc_t[ka] as f64, sz.acc_t[kt] as f64, ride as u8 as f64, station_end as f64]);
        }
        // routes over the bigger pairs, by their share of the whole trip's rail riders
        let mut all: Vec<(usize, f32, Vec<Leg>)> = vec![];
        for (pi, &(ka, kt, share)) in pairs.iter().enumerate() {
            if share < PAIR_MIN {
                break;
            }
            let (a, t) = (sz.acc_st[ka] as usize, sz.acc_st[kt] as usize);
            for (rs, legs) in routes(&st.g, &sk, p, a, t, ROUTE_MIN / share) {
                all.push((pi, rs * share, legs));
            }
        }
        all.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        all.truncate(MAX_ROUTES);
        all.retain(|r| r.1 >= ROUTE_MIN);
        h[24] = pairs.len() as f64;
        h[25] = all.len() as f64;
        h[31] = (1.0 - all.iter().map(|r| r.1 as f64).sum::<f64>()).max(0.0);
        let mut out = h;
        out.extend(body);
        for (pi, share, legs) in &all {
            out.extend([*pi as f64, *share as f64, legs.len() as f64]);
            for l in legs {
                out.extend([l.kind as f64, l.rn_from as f64, if l.rn_to == NONE { -1.0 } else { l.rn_to as f64 }, l.st_from as f64, l.st_to as f64, l.min as f64, l.perc as f64, l.fare as f64, l.wait_min as f64, l.wait_perc as f64, l.stops as f64, 0.0]);
            }
        }
        out
    }
}

#[wasm_bindgen]
impl DemandApi {
    /// The trip explorer (T-195): one trip from (ox, oy) to (dx, dy), pack frame, as a commute
    /// from a home at the first to work at the second, in solved period `q`. Packed, f64:
    ///
    /// `HEAD` numbers: status (1 answered; 0 period `q` not solved here; -1 a point has no cell
    /// within `NEAR_CELL_M`), the period's crowding round, the two cells, their subzones (-1 none),
    /// the straight line between the cells and the drive's distance (between the subzones'
    /// centres), metres; the walk share; the drive's real and perceived minutes, its parking and
    /// its whole cost (perceived); rail offered (1/0), rail's cost (the logsum, perceived, -1 if
    /// none), rail's share of the trips that do not walk, rail's utility against the road; the
    /// chance of walking, rail and driving; the cheapest perceived walk at each end; the base
    /// fare (perceived minutes) and perceived minutes per dollar; station pairs, routes; why rail
    /// is not offered (`RAIL_OFFERED`..`NO_ROUTE`); the two cells' x, y; the share of the rail
    /// riders on routes not listed.
    ///
    /// Then per station pair, biggest share first, `PAIR_LEN` numbers: access and egress demand
    /// stations, a route node at each (-1 none), the access walk's straight-line metres and
    /// perceived minutes, the egress walk's, the station-to-station strategy cost (perceived
    /// minutes, the base fare and first boarding's transfer penalty not included), the pair's
    /// share of the trip's rail riders, the two walks' real minutes (the access entry's are a
    /// ride's drive minutes, its metres the straight line to the station), whether the access
    /// entry is a ride (1/0) and, for a ride, the perceived minutes beyond the drive's (the ride's
    /// fixed cost and the parking or drop-off at the station).
    ///
    /// Then per route, biggest first: its station pair's index, its share of the trip's rail
    /// riders, its legs; then `LEG_LEN` numbers a leg: kind (1 ride, 2 walk between stations),
    /// route nodes from and to, demand stations from and to, real minutes, perceived minutes,
    /// distance fare (perceived minutes), wait (real and perceived minutes), stops ridden, 0.
    pub fn trip(&self, q: usize, ox: f32, oy: f32, dx: f32, dy: f32) -> Vec<f64> {
        let mut h = vec![0f64; HEAD];
        let Some(st) = self.period(q) else { return h };
        h[1] = st.round as f64;
        let (Some(o), Some(d)) = (self.nearest_cell(ox, oy), self.nearest_cell(dx, dy)) else {
            h[0] = -1.0;
            return h;
        };
        self.explain(st, q, o, d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::{kernel, network::PERIODS, pack, synth};

    /// The synthetic city's inner 12 km with two crossing lines (as api.rs's tests).
    fn small() -> (DemandApi, Vec<f32>) {
        let full = synth::synth_city(1);
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 12_000.0).collect();
        let mut city = pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: vec![],
            gravity: None,
        };
        let p = Params::default();
        let z = kernel::build_zones(&city, p.zone_m);
        let g = kernel::gravity(&z, &p, 100, 1e-5);
        city.gravity = Some(kernel::to_pack_gravity(&z, &g, &p));
        let (h, b) = pack::write(&city);
        let mut api = DemandApi::new(&h, &b).unwrap();
        let mut xy = vec![];
        for i in -8..=8 {
            xy.extend([i as f32 * 1000.0, 0.0]);
        }
        let mut line_b = vec![];
        for i in -8..=8i32 {
            if i == 0 {
                line_b.push(8u32);
            } else {
                line_b.push((xy.len() / 2) as u32);
                xy.extend([0.0, i as f32 * 1000.0]);
            }
        }
        let stops: Vec<u32> = (0..17).chain(line_b).collect();
        let mut times = vec![];
        for _ in 0..2 {
            for run in 0..2 {
                for _ in 0..3 {
                    for k in 0..17 {
                        let end = if run == 0 { k == 16 } else { k == 0 };
                        times.push(if end { 0.0 } else { 90.0 });
                    }
                }
            }
        }
        api.set_network(&xy, &[17, 17], &stops, &times, &[20.0, 10.0, 4.0, 12.0, 6.0, 0.0], &[10.0, 8.0]).unwrap();
        let km: Vec<f32> = (0..2).flat_map(|_| (0..2).flat_map(|_| (0..17).map(|k| if k == 16 { 0.0 } else { 1.0 }))).collect();
        api.set_fares(2.9, 0.1, &km).unwrap();
        (api, xy)
    }

    /// The pair choice is mode choice's: over every destination, an origin subzone's rail trips add
    /// up to what the solve kept for it, and the station pairs' shares to the solve's station pair
    /// matrix. Free flow and after a crowding round.
    #[test]
    fn pair_choice_reproduces_mode_choice() {
        let (mut api, _) = small();
        for q in [0usize, 2] {
            api.solve(q);
            for round in 0..2 {
                if round == 1 {
                    // a crowded round's own pass only: the solve's sums are averaged with the first
                    let seg: Vec<f32> = api.seg(q).iter().map(|v| v * 30.0).collect();
                    assert!(api.crowd_with_external(q, &seg));
                }
                let st = api.period(q).unwrap();
                let (p, g) = (&st.p, &st.g);
                let sz = api.sz.as_ref().unwrap();
                let s = g.n_st;
                let sk = kernel::skims(g, p);
                let er = kernel::station_exp(&sk, p);
                let mo = kernel::mode_choice(&api.z, &api.gr, sz, &er, s, p);
                let (z, gr) = (&api.z, &api.gr);
                let share = p.period_share as f64;
                let mut od = vec![0f64; s * s];
                let (mut worst, mut tot) = (0f64, 0f64);
                let mut checked = 0;
                for u in 0..sz.n {
                    let i = sz.zone[u] as usize;
                    if sz.pop[u] <= 0.0 || z.pop[i] <= 0.0 {
                        continue;
                    }
                    let ou = sz.origin_weight(u, z, gr, share);
                    let mut rail = 0f64;
                    for v in 0..sz.n {
                        let j = sz.zone[v] as usize;
                        if sz.jobs[v] <= 0.0 || z.jobs[j] <= 0.0 {
                            continue;
                        }
                        let pc = pair_choice(&api, p, &sk, u, v);
                        if pc.why != RAIL_OFFERED {
                            continue;
                        }
                        let dv = (gr.col[j] * sz.jobs[v] as f64 / z.jobs[j]) as f32;
                        let f = gr.ftab[(gr.base(z, i) + gr.key(z, j)) as usize] * gr.k(z, i, j);
                        let x = ou * f * dv * (1.0 - pc.pw) / (1.0 + fast_exp(-pc.util));
                        rail += x as f64;
                        for &(ka, kt, w) in &pc.pairs {
                            od[sz.acc_st[ka] as usize * s + sz.acc_st[kt] as usize] += (x * w / pc.ssum) as f64;
                        }
                    }
                    let want = mo.rail_by_sub[u] as f64;
                    worst = worst.max((rail - want).abs() / want.max(1e-3));
                    tot += want;
                    if want > 0.0 {
                        checked += 1;
                    }
                    // the period's own copy agrees too (round 0: nothing averaged in yet)
                    if round == 0 {
                        assert_eq!(st.sub.rail_home[u], mo.rail_by_sub[u]);
                    }
                }
                assert!(checked > 100 && tot > 0.0);
                assert!(worst < 1e-4, "period {q} round {round}: worst origin off by {worst:.2e}");
                let (mut d, mut m) = (0f64, 0f64);
                for k in 0..s * s {
                    d += (od[k] - mo.od[k] as f64).abs();
                    m += mo.od[k] as f64;
                }
                assert!(d / m < 1e-4, "period {q} round {round}: station pairs off by {:.2e}", d / m);
                println!("period {q} round {round}: {checked} origins, worst {worst:.1e}, station pairs {:.1e}", d / m);
            }
        }
    }

    /// The query: a trip along one line rides it with no change; one between the two lines changes
    /// at the shared centre; the routes' shares add up; a lone route's costs add up to the
    /// strategy's; a point off the city answers -1; a period not solved 0.
    #[test]
    fn trip_answers() {
        let (mut api, _) = small();
        assert_eq!(api.trip(0, 0.0, 0.0, 1.0, 1.0)[0], 0.0, "nothing solved");
        for q in 0..PERIODS {
            api.solve(q);
        }
        assert_eq!(api.trip(0, 0.0, 0.0, 99_000.0, 0.0)[0], -1.0);
        let read = |v: &[f64]| {
            let np = v[24] as usize;
            let mut o = HEAD + np * PAIR_LEN;
            let mut routes = vec![];
            for _ in 0..v[25] as usize {
                let (pi, share, n) = (v[o] as usize, v[o + 1], v[o + 2] as usize);
                o += 3;
                let legs: Vec<Vec<f64>> = (0..n).map(|k| v[o + k * LEG_LEN..o + (k + 1) * LEG_LEN].to_vec()).collect();
                o += n * LEG_LEN;
                routes.push((pi, share, legs));
            }
            assert_eq!(o, v.len());
            routes
        };
        // along line A: 6 km west of the centre to 6 km east
        let v = api.trip(0, -6000.0, 0.0, 6000.0, 0.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[13], 1.0, "rail offered");
        let (w, r, dr) = (v[17], v[18], v[19]);
        assert!((w + r + dr - 1.0).abs() < 1e-6 && r > 0.0 && dr > 0.0);
        let routes = read(&v);
        let main = &routes[0];
        assert_eq!(main.2.len(), 1, "one ride");
        assert_eq!(main.2[0][0], 1.0);
        let listed: f64 = routes.iter().map(|r| r.1).sum();
        assert!((listed + v[31] - 1.0).abs() < 1e-5);
        // a route's own costs add up to its pair's strategy cost when it is the only one
        let pair = &v[HEAD + main.0 * PAIR_LEN..HEAD + (main.0 + 1) * PAIR_LEN];
        let st = api.period(0).unwrap();
        let (strategy, l) = (pair[8], &main.2[0]);
        let own = l[6] + l[7] + l[9];
        println!("along A: walk {w:.3} rail {r:.3} drive {dr:.3}; route {:.3}, strategy {strategy:.3} vs own {own:.3}", main.1);
        if (main.1 - pair[9]).abs() < 1e-6 {
            assert!((strategy - own).abs() < 1e-3, "strategy {strategy} vs the route's own {own}");
        }
        // across: west on A to north on B, changing at the centre
        let v = api.trip(0, -6000.0, 0.0, 0.0, 6000.0);
        let routes = read(&v);
        let main = &routes[0];
        let rides: Vec<&Vec<f64>> = main.2.iter().filter(|l| l[0] == 1.0).collect();
        assert_eq!(rides.len(), 2, "a change: {:?}", main.2);
        assert_eq!(st.g.rn_line[rides[0][1] as usize], 0);
        assert_eq!(st.g.rn_line[rides[1][1] as usize], 1);
        assert!(rides[1][8] > 0.0 && rides[1][9] > 0.0, "a wait at the change");
        // night: line B does not run, so the trip across has no route by B
        let v = api.trip(4, -6000.0, 0.0, 0.0, 6000.0);
        for (_, _, legs) in read(&v) {
            assert!(legs.iter().filter(|l| l[0] == 1.0).all(|l| st.g.rn_line[l[1] as usize] == 0));
        }
        // probabilities differ by period with the service
        let morning = api.trip(0, -6000.0, 0.0, 6000.0, 0.0)[18];
        let night = api.trip(4, -6000.0, 0.0, 6000.0, 0.0)[18];
        assert!(morning > night, "morning {morning} night {night}");
    }

    /// The real New York pack and network (nyc_real_track.json), free flow in the morning peak:
    /// for each origin of three trips, its rail trips over every destination add up to the solve's
    /// own for that origin subzone; prints the three trips as the explorer reads them.
    /// `cargo test --release --features demand trip_real_network -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn trip_real_network() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let mut a = DemandApi::new(&h, &b).unwrap();
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (st_xy, line_n, stops, times, tph, cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        let names: Vec<String> = j["stations"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
        let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
        let (mut rn_st, mut hop_km) = (vec![], vec![]);
        let mut off = 0usize;
        for &n in &line_n {
            let n = n as usize;
            for dir in 0..2 {
                for i in 0..n {
                    let k = if dir == 0 { off + i } else { off + n - 1 - i };
                    rn_st.push(stops[k] as usize);
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
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        a.solve(0);
        // lon/lat to the pack's equirectangular frame (origin -73.985, 40.758)
        const R: f64 = 6371008.8;
        let rad = std::f64::consts::PI / 180.0;
        let at = |lon: f64, lat: f64| ((R * (lon + 73.985) * rad * (40.758 * rad).cos()) as f32, (R * (lat - 40.758) * rad) as f32);
        let trips = [
            ("Park Slope to Midtown", at(-73.978, 40.672), at(-73.981, 40.757)),
            ("Jersey City to Wall St", at(-74.063, 40.733), at(-74.009, 40.706)),
            ("Mineola to Hicksville", at(-73.640, 40.749), at(-73.525, 40.768)),
        ];
        let st = a.period(0).unwrap();
        let line_of = |rn: f64| lines[st.g.rn_line[rn as usize] as usize].clone();
        let name_of = |rn: f64| if rn < 0.0 { "?".to_string() } else { names[rn_st[rn as usize]].clone() };
        let sz = a.sz.as_ref().unwrap();
        let sk = kernel::skims_for(&st.g, &st.p, &kernel::egress_stations(sz, st.g.n_st));
        let (z, gr) = (&a.z, &a.gr);
        // TW_TRIP_DUMP=<file>: the three answers, to compare with the WASM worker's
        let mut dump = serde_json::Map::new();
        for (what, o, dd) in trips {
            let v = a.trip(0, o.0, o.1, dd.0, dd.1);
            dump.insert(what.to_string(), serde_json::json!({ "from": [o.0, o.1], "to": [dd.0, dd.1], "answer": v }));
            println!("\n{what}: cells {} -> {}, subzones {} -> {}, {:.1} km ({:.1} km between the subzones' centres)", v[2], v[3], v[4], v[5], v[6] / 1000.0, v[7] / 1000.0);
            println!("  walk {:.1}%, train {:.1}%, drive {:.1}% (rail {:.1}% of those not walking; utility {:.2})", 100.0 * v[17], 100.0 * v[18], 100.0 * v[19], 100.0 * v[15], v[16]);
            println!("  drive {:.1} min (x1.33 = {:.1}) + parking {:.1} = {:.1} perceived; rail {:.1} perceived (walks {:.1} + {:.1}, base fare {:.1})", v[9], v[10], v[11], v[12], v[14], v[20], v[21], v[22]);
            let np = v[24] as usize;
            for k in 0..np.min(4) {
                let p = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
                println!("  pair {:.1}%: {} ({:.0} m, {:.1}) to {} ({:.0} m, {:.1}), strategy {:.1}", 100.0 * p[9], name_of(p[2]), p[4], p[5], name_of(p[3]), p[6], p[7], p[8]);
            }
            let mut o2 = HEAD + np * PAIR_LEN;
            for _ in 0..v[25] as usize {
                let (share, n) = (v[o2 + 1], v[o2 + 2] as usize);
                let legs: Vec<String> = (0..n)
                    .map(|k| {
                        let l = &v[o2 + 3 + k * LEG_LEN..o2 + 3 + (k + 1) * LEG_LEN];
                        if l[0] == 1.0 {
                            format!("{} {} to {} ({} stops, wait {:.1}, ride {:.1})", line_of(l[1]), name_of(l[1]), name_of(l[2]), l[10], l[8], l[5])
                        } else {
                            format!("walk to {} ({:.1} min)", name_of(l[2]), l[5])
                        }
                    })
                    .collect();
                println!("  route {:.1}%: {}", 100.0 * share, legs.join("; "));
                o2 += 3 + n * LEG_LEN;
            }
            // the check: this origin's rail trips over every destination are the solve's own
            if v[4] < 0.0 {
                continue;
            }
            let u = v[4] as usize;
            let i = sz.zone[u] as usize;
            let ou = sz.origin_weight(u, z, gr, st.p.period_share as f64);
            let mut rail = 0f64;
            for w in 0..sz.n {
                let jz = sz.zone[w] as usize;
                if sz.jobs[w] <= 0.0 || z.jobs[jz] <= 0.0 {
                    continue;
                }
                let pc = pair_choice(&a, &st.p, &sk, u, w);
                if pc.why != RAIL_OFFERED {
                    continue;
                }
                let dv = (gr.col[jz] * sz.jobs[w] as f64 / z.jobs[jz]) as f32;
                let f = gr.ftab[(gr.base(z, i) + gr.key(z, jz)) as usize] * gr.k(z, i, jz);
                rail += (ou * f * dv * (1.0 - pc.pw) / (1.0 + fast_exp(-pc.util))) as f64;
            }
            let want = st.sub.rail_home[u] as f64;
            println!("  check: origin subzone {u}'s rail trips over every destination {rail:.3}, the solve's {want:.3} (off by {:.1e})", (rail - want).abs() / want.max(1e-9));
            assert!((rail - want).abs() <= 1e-4 * want.max(1.0));
        }
        if let Ok(path) = std::env::var("TW_TRIP_DUMP") {
            std::fs::write(&path, serde_json::to_string(&dump).unwrap()).unwrap();
            println!("dumped to {path}");
        }
    }
}
