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
//!
//! An everyday purpose's trip (T-586, `EverydayApi::trip`) is answered the same way, as the
//! everyday pass chooses: its own period graph, the subzone pair's mean distance, the cells' walk.

use super::api::{DemandApi, FARE_MIN_PER_USD};
use super::everyday::{EverydayApi, TripInputs};
use super::kernel::*;
use super::network::PERIODS;
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
pub const HEAD: usize = 44;
pub const PAIR_LEN: usize = 20;
pub const LEG_LEN: usize = 13;

/// Why rail is not offered for a trip (`HEAD` slot 26).
pub const RAIL_OFFERED: u32 = 0;
pub const NO_STATION_AT_START: u32 = 1;
pub const NO_STATION_AT_END: u32 = 2;
pub const WALKS_TOO_LONG: u32 = 3;
pub const NO_ROUTE: u32 = 4;
/// Every station pair walks further than walking there (T-1059).
pub const WALK_FURTHER: u32 = 5;

/// Whether the home end could drive to a station (`HEAD` slot 41, T-910): the trip is shorter than
/// `Params::ride_min_km`, the drives were among its ways to a station, or it has no station
/// within a drive.
pub const RIDE_TOO_SHORT: u8 = 0;
pub const RIDE_TRIED: u8 = 1;
pub const RIDE_NONE: u8 = 2;
const fn ride_state(by_ride: bool, has_rides: bool) -> u8 {
    if by_ride { RIDE_TRIED } else if has_rides { RIDE_TOO_SHORT } else { RIDE_NONE }
}

/// The model's choice for one (origin subzone, destination subzone) pair in a period, computed as
/// `kernel::mode_choice` computes it.
pub(crate) struct PairChoice {
    /// Walking's share of all trips (T-558: walking the whole way by the two zones' distance).
    pub pw: f32,
    /// Straight line between the two subzones' centres, metres: the drive's distance.
    pub d_m: f32,
    /// Real minutes of the drive, and its parking at the destination (perceived minutes, the
    /// mean over drivers who park free and those who pay).
    pub drive_min: f32,
    pub park: f32,
    /// The drive's whole cost at that mean, perceived minutes.
    pub road: f32,
    /// `RAIL_OFFERED`, `WALKS_TOO_LONG` or `NO_ROUTE`.
    pub why: u32,
    /// The cheapest perceived walk at each end (the logsum's offsets).
    pub m_u: f32,
    pub m_v: f32,
    /// The home end's drive to a station (T-910): `RIDE_TOO_SHORT`, `RIDE_TRIED` or `RIDE_NONE`.
    pub ride: u8,
    /// The rail cost the logsum over station pairs gives (perceived minutes).
    pub rail: f32,
    /// Utility of rail against a driver who parks free, and rail's share of all trips
    /// (`Params::car_choice`).
    pub util: f32,
    pub r: f32,
    /// The walkers rail takes, a share of all trips (T-558, `ModeOut::rail_walk`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub walk_lost: f32,
    /// (access entry, egress entry, share of the trip's rail riders) over every station pair
    /// (`Subzones::acc_st` entries).
    pub pairs: Vec<(usize, usize, f32)>,
}

/// Homes and jobs per km² of land in zone `i`, as `Zones::density`.
fn zone_density(z: &Zones, i: usize) -> f32 {
    ((z.pop[i] + z.jobs[i]) / (z.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32
}

/// The cost of walking the whole way between zones `i` and `j` (`Gravity::walk_c`; `NO_WALK`
/// before it is set).
fn walk_cost(api: &DemandApi, i: usize, j: usize) -> f32 {
    // T-275: around water with the pack's routes
    api.gr.walk_cost_pair(&api.z, i, j)
}

/// The choice for subzones `u` (home) and `v` (work) with period parameters `p` and strategies
/// `sk` (searched at least towards `v`'s egress stations). Same expressions, same order, same
/// f32 results as `kernel::mode_choice`'s pair loop.
pub(crate) fn pair_choice(api: &DemandApi, sz: &Subzones, p: &Params, sk: &Skims, u: usize, v: usize) -> PairChoice {
    let z = &api.z;
    let s = sk.s;
    let (i, j) = (sz.zone[u] as usize, sz.zone[v] as usize);
    // T-710: the zone pair's distance band, as `mode_choice` takes it
    let (th, off) = p.band(p.band_of_d2(z.d2_m(i, j)));
    let ends = |r: std::ops::Range<usize>| {
        let m = sz.acc_w[r.clone()].iter().cloned().fold(f32::INFINITY, f32::min);
        (r, m)
    };
    // T-371: a long commute's home end walks or rides
    let by_ride = !sz.ride(u).is_empty() && ride_offered(z, p, i, j);
    let (ru, m_u) = ends(if by_ride { sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize } else { sz.walk(u) });
    let (rv, m_v) = ends(sz.walk(v));
    let acc_e = |k: usize, m: f32| fast_exp(-th * (sz.acc_w[k] - m));
    // `station_exp`: exp(-theta (r - offset)), 0 where there is no route and on the diagonal
    let er = |a: usize, t: usize| {
        let r = sk.r[a * s + t];
        if a == t || !r.is_finite() { 0.0 } else { fast_exp(-th * (r - off)) }
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
    // T-1059: the station pairs walking further than walking there are no option
    let routed = ssum >= 1e-30;
    if p.walk_rule > 0.0 && sk.walk.len() == s * s {
        let direct = d * api.gr.walk_factor(i, j);
        let acc: Vec<(u32, f32, f32, u32)> = ru.clone().map(|k| (sz.acc_st[k], acc_e(k, m_u), walked(sz, u, k), k as u32)).collect();
        let eg: Vec<(u32, f32, f32, u32)> = rv.clone().map(|k| (sz.acc_st[k], acc_e(k, m_v), sz.acc_d[k], k as u32)).collect();
        let kept = walk_kept(&acc, &eg, er, &sk.walk, s, direct);
        if kept < 1.0 {
            ssum *= kept;
            for e in pairs.iter_mut() {
                let (a, t) = (sz.acc_st[e.0] as usize, sz.acc_st[e.1] as usize);
                if !walk_ok(walked(sz, u, e.0), sk.walk[a * s + t], sz.acc_d[e.1], direct) {
                    e.2 = 0.0;
                }
            }
        }
    }
    let drive_min = p.drive_min_f(d, p.drive_factor(i, j), p.place_min_per_km(zone_density(z, i)), p.place_min_per_km(zone_density(z, j)));
    let (fixed, paid, free) = p.park_parts(z, j);
    let rk = d * 0.001 * p.drive_factor(i, j);
    let car = p.car_drive(drive_min, rk);
    // T-854: and parking at the home end
    let road_free = car + fixed + sz.home_park[u];
    let park = fixed + (1.0 - free) * paid + sz.home_park[u];
    let road = car + park;
    let cw = walk_cost(api, i, j);
    let b_u = if by_ride { sz.bound_min(p, u) } else { m_u };
    let why = if m_v > p.pair_walk_max - b_u {
        WALKS_TOO_LONG
    } else if !routed {
        NO_ROUTE
    } else if ssum < 1e-30 {
        WALK_FURTHER
    } else {
        RAIL_OFFERED
    };
    let bt = p.mode_beta(road_free);
    let uw = bt * (road_free - cw) + p.asc_walk();
    let (rail, util, r_car) = if why == RAIL_OFFERED {
        let rail = m_u + m_v + p.fare_min - fast_ln(ssum) / th + off;
        let util = bt * (road_free - rail) + p.asc_rail;
        (rail, util, p.car_choice(util, uw, bt, paid, free, rail, road_free))
    } else {
        (f32::INFINITY, f32::NEG_INFINITY, 0.0)
    };
    // the zone pair's walking with no rail, of which rail takes its share (`mode_choice`)
    let w_none = if cw < NO_WALK { api.nr.split(&api.gr, p, i, j, (api.gr.base(z, i) + api.gr.key(z, j)) as usize) } else { 0.0 };
    let r = r_car;
    let pairs: Vec<(usize, usize, f32)> = pairs.into_iter().map(|(a, t, w)| (a, t, if why == RAIL_OFFERED { w / ssum } else { 0.0 })).collect();
    let walk_lost = r * w_none;
    let pw = w_none - walk_lost;
    let ride = ride_state(by_ride, !sz.ride(u).is_empty());
    PairChoice { pw, d_m: d, drive_min, park, road, why, m_u, m_v, ride, rail, util, r, walk_lost, pairs }
}

/// The everyday layer's choice (T-586, T-201) for purpose inputs `ev`, home subzone `u` and
/// activity subzone `v`, outward: the same expressions and order as `EverydayApi::pass`, on the
/// subzone pair's mean distance (the cells' when no sample joins them), with the cells' own walk
/// share, of which rail takes its share as the views have it. With `cells` (`cell_subzones`), the
/// two cells' own distance and walk factor.
pub(crate) fn everyday_choice(ev: &TripInputs, sz: &Subzones, sk: &Skims, u: usize, v: usize, cells_m: f32, cells: bool) -> PairChoice {
    let api = ev.engine;
    let p = ev.p;
    let z = &api.z;
    let s = sk.s;
    let (i, j) = (sz.zone[u] as usize, sz.zone[v] as usize);
    let distance = if cells { cells_m } else { ev.pair.map_or(cells_m, |x| x.0) };
    let walk_factor = if cells { ev.walk_factor_cells } else { ev.walk_factor };
    // T-710: the pair's distance band, as the everyday pass takes it
    let (th, off) = p.band(p.band_of_d2(distance * distance));
    let ends = |r: std::ops::Range<usize>| {
        let m = sz.acc_w[r.clone()].iter().cloned().fold(f32::INFINITY, f32::min);
        (r, m)
    };
    let by_ride = distance >= p.ride_min_km * 1000.0 && !sz.ride(u).is_empty();
    let (ru, m_u) = ends(if by_ride { sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize } else { sz.walk(u) });
    let (rv, m_v) = ends(sz.walk(v));
    let acc_e = |k: usize, m: f32| fast_exp(-th * (sz.acc_w[k] - m));
    let er = |a: usize, t: usize| {
        let r = sk.r[a * s + t];
        if a == t || !r.is_finite() { 0.0 } else { fast_exp(-th * (r - off)) }
    };
    // T-1059: station pairs walking further than walking there are no option, as the pass has it
    let walk_on = distance * walk_factor;
    let rule = p.walk_rule > 0.0 && sk.walk.len() == s * s;
    let keep = |ka: usize, kb: usize| !rule || walk_ok(walked(sz, u, ka), sk.walk[sz.acc_st[ka] as usize * s + sz.acc_st[kb] as usize], walked(sz, v, kb), walk_on);
    let logsum = |a: &std::ops::Range<usize>, ma: f32, out: &mut Vec<(usize, usize, f32)>| {
        let (mut sum, mut all) = (0.0f32, 0.0f32);
        for ka in a.clone() {
            for kb in rv.clone() {
                let w = acc_e(ka, ma) * acc_e(kb, m_v) * er(sz.acc_st[ka] as usize, sz.acc_st[kb] as usize);
                all += w;
                if !keep(ka, kb) {
                    out.push((ka, kb, 0.0));
                    continue;
                }
                sum += w;
                out.push((ka, kb, w));
            }
        }
        (sum, all)
    };
    let mut pairs = Vec::with_capacity(ru.len() * rv.len());
    let (sum, all) = logsum(&ru, m_u, &mut pairs);
    let density = |k: usize| p.place_min_per_km(zone_density(z, k));
    let f = p.drive_factor(i, j);
    let drive_min = p.drive_min_f(distance, f, density(i), density(j));
    let (fixed, paid, free) = p.park_parts(z, j);
    let car = p.car_drive(drive_min, distance * 0.001 * f);
    let road_free = car + fixed + sz.home_park[u];
    let park = fixed + (1.0 - free) * paid + sz.home_park[u];
    let road = car + park;
    let cw = p.walk_trip_cost(walk_on / 1000.0);
    let b_u = if by_ride { sz.bound_min(p, u) } else { m_u };
    let why = if b_u + m_v > p.pair_walk_max {
        WALKS_TOO_LONG
    } else if all < 1e-30 {
        NO_ROUTE
    } else if sum < 1e-30 {
        WALK_FURTHER
    } else {
        RAIL_OFFERED
    };
    let (mut rail, mut util, mut r) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
    let w_none = ev.no_rail;
    if why == RAIL_OFFERED {
        rail = m_u + m_v + p.fare_min - fast_ln(sum) / th + off;
        let bt = p.mode_beta(road_free);
        util = bt * (road_free - rail) + p.asc_rail;
        let uw = bt * (road_free - cw) + p.asc_walk();
        r = p.car_choice(util, uw, bt, paid, free, rail, road_free);
        // each station pair's share of the trip's rail riders
        for e in pairs.iter_mut() {
            e.2 *= 1.0 / sum;
        }
    } else {
        for e in pairs.iter_mut() {
            e.2 = 0.0;
        }
    }
    let walk_lost = r * w_none;
    let ride = ride_state(by_ride, !sz.ride(u).is_empty());
    PairChoice { pw: w_none - walk_lost, d_m: distance, drive_min, park, road, why, m_u, m_v, ride, rail, util, r, walk_lost, pairs }
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
    /// a ride's distance fare, perceived minutes, with its share by km of what the fare curve
    /// adds over the base and rate (T-883)
    pub fare: f32,
    /// a ride's fare km, while the fare curve is not a straight line (`Graph::rn_km`)
    pub km: f32,
    /// a ride's wait at its boarding station: real minutes (half the combined headway of the
    /// useful departures there, plus the line's average extra wait for full trains there, T-188)
    /// and perceived minutes (the same, T-585, plus `Graph::rn_span` where some of the period's
    /// riders have no train due, T-842)
    pub wait_min: f32,
    pub wait_perc: f32,
    /// the part of both waits that is for a train with room (T-188's `Graph::rn_left`)
    pub full_wait: f32,
    /// stops ridden
    pub stops: u32,
    /// a ride's fixed cost of changing onto it (`Params::transfer_min`; 0 on the first ride)
    pub change: f32,
}

impl Leg {
    /// The leg in a trip answer (`LEG_LEN` numbers, `DemandApi::trip`).
    pub(crate) fn numbers(&self) -> [f64; LEG_LEN] {
        let to = if self.rn_to == NONE { -1.0 } else { self.rn_to as f64 };
        [self.kind as f64, self.rn_from as f64, to, self.st_from as f64, self.st_to as f64, self.min as f64, self.perc as f64, self.fare as f64, self.wait_min as f64, self.wait_perc as f64, self.stops as f64, self.change as f64, self.full_wait as f64]
    }
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
    let at = At { g, p, next, board, cn: &sk.change[slot as usize], rep: &rep, t, min_share };
    walk(&at, a, None, 1.0, &mut legs, &mut out);
    if !g.fare_curve.is_line() {
        for (_, legs) in &mut out {
            let total: f32 = legs.iter().filter(|l| l.kind == 1).map(|l| l.km).sum();
            let extra = g.fare_curve.extra(total);
            let rides = legs.iter().filter(|l| l.kind == 1).count() as f32;
            for l in legs.iter_mut().filter(|l| l.kind == 1) {
                l.fare += extra * if total > 0.0 { l.km / total } else { 1.0 / rides };
            }
        }
    }
    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    out
}

/// What `walk` reads: one destination's strategy.
struct At<'a> {
    g: &'a Graph,
    p: &'a Params,
    next: &'a [u32],
    board: &'a [f32],
    cn: &'a ChangeNodes,
    rep: &'a [u32],
    t: usize,
    min_share: f32,
}

/// From node `v`: a station, or a change node (T-575) boarding at its station; `came` the route
/// node the rider left by an alighting or a walk to it, None at the trip's start.
fn walk(at: &At, v: usize, came: Option<usize>, share: f32, legs: &mut Vec<Leg>, out: &mut Vec<(f32, Vec<Leg>)>) {
    let At { g, p, next, board, cn, rep, t, min_share } = *at;
    if v == t {
        out.push((share, legs.clone()));
        return;
    }
    if legs.len() > 40 {
        return; // a guard: strategies have no cycles
    }
    let s = g.n_st;
    let (st, k) = if v < s { (v, NONE as usize) } else { (cn.st[v - g.n] as usize, v - g.n) };
    let acc = |r: usize| if v < s { board[r] } else { cn.tph(g, k, r) };
    let v = st;
    let edges = g.adj_off[v] as usize..g.adj_off[v + 1] as usize;
    let sum: f32 = edges.clone().map(|e| acc(g.adj_to[e] as usize - s)).sum();
    if sum == 0.0 {
        return;
    }
    let headway = 60.0 / sum;
    let wait_min = headway / 2.0;
    let arrived = came.map(|c| arrival_at(g, c, v));
    for e in edges {
        let r0 = g.adj_to[e] as usize;
        let sh = share * acc(r0 - s) / sum;
        if acc(r0 - s) == 0.0 || sh < min_share {
            continue;
        }
        // T-717: a change walks to this train's platform, as part of the walk into the station if
        // there was one
        let plat = arrived.map_or(0.0, |a| platform_walk_to(p, a, g.rn_dx[r0 - s], g.rn_dy[r0 - s]));
        let (mut r, mut min, mut perc, mut fare, mut km, mut stops) = (r0, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0u32);
        // a boarding rides at least one stop, even where riders on board leave (T-1055)
        let mut nx = if (next[r] as usize) >= s && (next[r] as usize) < g.n { next[r] } else { g.rn_next[r - s] };
        while nx != NONE && (nx as usize) >= s && (nx as usize) < g.n && stops as usize <= g.n {
            let ride = (g.adj_off[r] as usize..g.adj_off[r + 1] as usize).find(|&k| g.adj_seg[k] != NONE).unwrap();
            min += g.rn_ride[r - s];
            fare += g.rn_fare[r - s];
            km += g.rn_km.get(r - s).copied().unwrap_or(0.0);
            perc += g.adj_cost[ride] - g.rn_fare[r - s];
            stops += 1;
            r = nx as usize;
            nx = next[r];
        }
        if nx == NONE {
            continue;
        }
        // T-717: a change walks to this train's platform, within the walk into the station if
        // there was one
        let mut kept = None;
        let mut pushed = 0;
        if plat > 0.0 {
            match legs.last_mut() {
                Some(l) if l.kind == 2 && l.st_to as usize == v => {
                    kept = Some(*l);
                    l.min += plat / p.w_xfer;
                    l.perc += plat;
                    l.rn_to = (r0 - s) as u32;
                }
                _ => {
                    let c = came.unwrap();
                    legs.push(Leg { kind: 2, rn_from: (c - s) as u32, rn_to: (r0 - s) as u32, st_from: v as u32, st_to: v as u32, min: plat / p.w_xfer, perc: plat, fare: 0.0, km: 0.0, wait_min: 0.0, wait_perc: 0.0, full_wait: 0.0, stops: 0, change: 0.0 });
                    pushed += 1;
                }
            }
        }
        let here = g.rn_station[r - s];
        // and the extra wait where full trains leave riders behind (T-188); riders with no train
        // due within their tolerance (T-842) weigh on the perceived wait only
        let (left, span) = (g.rn_left[r0 - s], g.rn_span[r0 - s]);
        let change = if came.is_some() { p.transfer_min } else { 0.0 };
        legs.push(Leg { kind: 1, rn_from: (r0 - s) as u32, rn_to: (r - s) as u32, st_from: v as u32, st_to: here, min, perc, fare, km, wait_min: wait_min + left - span, wait_perc: wait_min + left, full_wait: left - span, stops, change });
        pushed += 1;
        let to = if (nx as usize) < s { nx } else { cn.st[nx as usize - g.n] };
        if to != here {
            // a walking transfer from this route node to a station nearby
            let cost = (g.adj_off[r] as usize..g.adj_off[r + 1] as usize).find(|&k| g.adj_to[k] == to).map_or(0.0, |k| g.adj_cost[k]);
            legs.push(Leg { kind: 2, rn_from: (r - s) as u32, rn_to: rep[to as usize], st_from: here, st_to: to, min: cost / p.w_xfer, perc: cost, fare: 0.0, km: 0.0, wait_min: 0.0, wait_perc: 0.0, full_wait: 0.0, stops: 0, change: 0.0 });
            pushed += 1;
        }
        walk(at, nx as usize, Some(r), sh, legs, out);
        legs.truncate(legs.len() - pushed);
        if let Some(l) = kept {
            *legs.last_mut().unwrap() = l;
        }
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

    /// T-720: cells `o` (home, in subzone `u`) and `d` (in `v`) as subzones 0 and 1 of their own,
    /// as one subzone per cell (`Params::band_min` -1) builds them: each cell's own walk to its
    /// `K` nearest stations, and the home's rides those of `u`. Drives go between the cells.
    fn cell_subzones(&self, p: &Params, o: usize, u: usize, d: usize, v: usize) -> Subzones {
        let (sz, city) = (self.sz.as_ref().unwrap(), &self.city);
        let svc = self.svc.as_ref().unwrap();
        let cut = p.walk_cutoff_m;
        let walks = |c: usize| {
            let (x, y) = (city.x[c], city.y[c]);
            let mut w: Vec<(f32, u32, f32)> = vec![];
            for s in 0..svc.st_x.len() {
                let (sx, sy) = (svc.st_x[s], svc.st_y[s]);
                let straight = ((sx - x).powi(2) + (sy - y).powi(2)).sqrt();
                if straight > cut {
                    continue;
                }
                let m = p.routes.as_ref().map_or(straight, |r| r.walk_to_cell(sx, sy, cut, c as u32, x, y));
                if m <= cut {
                    w.push((p.access_cost(m), s as u32, m));
                }
            }
            w.sort_by(|a, b| a.2.total_cmp(&b.2));
            w.truncate(K);
            w
        };
        let home = self.pop()[o];
        let mut cz = Subzones {
            n: 2,
            zone: vec![sz.zone[u], sz.zone[v]],
            pop: vec![home, 0.0],
            jobs: vec![0.0, self.jobs()[d]],
            cx: vec![city.x[o], city.x[d]],
            cy: vec![city.y[o], city.y[d]],
            home_park: vec![if home > 0.0 { p.home_park_cell(o) } else { 0.0 }, 0.0],
            acc_off: vec![0],
            acc_mid: vec![],
            acc_st: vec![],
            acc_w: vec![],
            acc_d: vec![],
            acc_t: vec![],
            home_off: vec![0, 0, 0],
            home_zones: vec![],
            zo_off: vec![],
            zo: vec![],
            zd_off: vec![],
            zd: vec![],
            cell: vec![],
            mem_off: vec![0, 1, 2],
            mem: vec![o as u32, d as u32],
        };
        for (c, rides) in [(o, sz.ride(u)), (d, 0..0)] {
            for (w, st, m) in walks(c) {
                cz.acc_st.push(st);
                cz.acc_w.push(w);
                cz.acc_d.push(m);
                cz.acc_t.push(p.walk_min(m));
            }
            cz.acc_mid.push(cz.acc_st.len() as u32);
            for k in rides {
                cz.acc_st.push(sz.acc_st[k]);
                cz.acc_w.push(sz.acc_w[k]);
                cz.acc_d.push(sz.acc_d[k]);
                cz.acc_t.push(sz.acc_t[k]);
            }
            cz.acc_off.push(cz.acc_st.len() as u32);
        }
        cz
    }

    /// The answer for cells `o` and `d` on period graph `g` with parameters `p` (`trip`'s layout),
    /// as the commutes choose or as an everyday purpose does (`m`).
    fn explain(&self, g: &Graph, p: &Params, round: usize, o: usize, d: usize, m: &Model) -> Vec<f64> {
        let city = &self.city;
        let mut h = vec![0f64; HEAD];
        h[0] = 1.0;
        h[1] = round as f64;
        h[2] = o as f64;
        h[3] = d as f64;
        h[6] = ((city.x[o] - city.x[d]).powi(2) + (city.y[o] - city.y[d]).powi(2)).sqrt() as f64;
        // walking the whole way, real minutes (T-628: the explorer's walk row)
        h[33] = h[6] / 1000.0 * (p.walk_detour * 1000.0 / p.walk_mps / 60.0 * self.gr.walk_factor_cells(&self.z, o, d)) as f64;
        h[36] = p.walk_trip_w as f64;
        if let Model::Everyday(ev) = m {
            h[34] = ev.pair.map_or(0.0, |x| x.1 as f64);
        }
        h[22] = p.fare_min as f64;
        h[23] = FARE_MIN_PER_USD as f64;
        h[27] = city.x[o] as f64;
        h[28] = city.y[o] as f64;
        h[29] = city.x[d] as f64;
        h[30] = city.y[d] as f64;
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
            let hp = p.home_park_cell(o);
            let park = p.park_mean(&self.z, j) + hp;
            let rk = dm * 0.001 * p.drive_factor(i, j);
            let car = p.car_drive(drive, rk);
            // walking against that drive, as the zones' totals have it (`NoRail::split`)
            let (fixed, paid, free) = p.park_parts(&self.z, j);
            let pw = match m {
                Model::Commute => p.walk_no_rail(car + fixed + hp, paid, free, walk_cost(self, i, j)),
                Model::Everyday(ev) => ev.no_rail,
            };
            h[7] = dm as f64;
            h[8] = pw as f64;
            h[9] = drive as f64;
            h[10] = (drive * p.w_drive) as f64;
            h[11] = (park - p.car_own_min()) as f64;
            h[12] = (car + park) as f64;
            h[32] = (p.car_own_min() + rk * p.car_own_km_min()) as f64;
            h[35] = rk as f64;
            h[17] = pw as f64;
            h[19] = (1.0 - pw) as f64;
            (h[37], h[39], h[40]) = (h[17], h[19], -1.0);
            h[26] = if u == NONE { NO_STATION_AT_START } else { NO_STATION_AT_END } as f64;
            h[31] = 1.0;
            return h;
        }
        let (u, v) = (u as usize, v as usize);
        let grouped = self.sz.as_ref().unwrap();
        // T-720: the trip is the two cells' own; the subzones' answer is what the views count
        let cz = self.cell_subzones(p, o, u, d, v);
        // strategies towards the destination's egress stations only (each search is independent,
        // so they are the model's, T-119)
        let mut want = vec![false; g.n_st];
        for k in grouped.walk(v) {
            want[grouped.acc_st[k] as usize] = true;
        }
        for k in cz.walk(1) {
            want[cz.acc_st[k] as usize] = true;
        }
        let sk = skims_for(g, p, &want);
        let cells_m = h[6] as f32;
        let choose = |zs: &Subzones, a: usize, b: usize, cells: bool| match m {
            Model::Commute => pair_choice(self, zs, p, &sk, a, b),
            Model::Everyday(ev) => everyday_choice(ev, zs, &sk, a, b, cells_m, cells),
        };
        let gc = choose(grouped, u, v, false);
        h[37] = gc.pw as f64;
        h[38] = gc.r as f64;
        h[39] = (1.0 - gc.pw - gc.r).max(0.0) as f64;
        h[40] = grouped.walk(u).min_by(|&a, &b| grouped.acc_w[a].total_cmp(&grouped.acc_w[b])).map_or(-1.0, |k| grouped.acc_t[k] as f64);
        // the rides the two cells take are the home subzone's, its parking at home included
        let ride_home_park = grouped.home_park[u];
        let (sz, u, v) = (&cz, 0, 1);
        let pc = choose(sz, u, v, true);
        let rail = pc.r;
        h[7] = pc.d_m as f64;
        h[8] = pc.pw as f64;
        h[9] = pc.drive_min as f64;
        h[10] = (pc.drive_min * p.w_drive) as f64;
        h[11] = (pc.park - p.car_own_min()) as f64;
        h[12] = pc.road as f64;
        let rk = pc.d_m * 0.001 * p.drive_factor(sz.zone[u] as usize, sz.zone[v] as usize);
        h[32] = (p.car_own_min() + rk * p.car_own_km_min()) as f64;
        h[35] = rk as f64;
        h[13] = (pc.why == RAIL_OFFERED) as u32 as f64;
        h[14] = if pc.rail.is_finite() { pc.rail as f64 } else { -1.0 };
        // rail's share of the trips not walking
        h[15] = if pc.pw < 1.0 { (pc.r / (1.0 - pc.pw)) as f64 } else { 0.0 };
        h[16] = if pc.util.is_finite() { pc.util as f64 } else { -1e9 };
        h[17] = pc.pw as f64;
        h[18] = rail as f64;
        h[19] = (1.0 - pc.pw - pc.r).max(0.0) as f64;
        h[20] = pc.m_u as f64;
        h[21] = pc.m_v as f64;
        h[26] = pc.why as f64;
        h[41] = pc.ride as f64;
        h[42] = p.ride_min_km as f64;
        h[43] = p.pair_walk_max as f64;
        if pc.why != RAIL_OFFERED {
            h[31] = 1.0;
            return h;
        }
        // station pairs, biggest share first
        let mut pairs: Vec<(usize, usize, f32)> = pc.pairs.iter().filter(|e| e.2 > 0.0).copied().collect();
        pairs.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
        let s = g.n_st;
        let mut rep = vec![-1f64; s];
        for (r, &stn) in g.rn_station.iter().enumerate().rev() {
            rep[stn as usize] = r as f64;
        }
        let mut body = vec![];
        let mut sq = None;
        for &(ka, kt, share) in &pairs {
            let (a, t) = (sz.acc_st[ka] as usize, sz.acc_st[kt] as usize);
            let ride = sz.ride(u).contains(&ka);
            // a ride entry (T-371, T-915): its perceived minutes beyond the drive's, and their parts
            let mut parts = [0f32; 7];
            if ride {
                let svc = self.svc.as_ref().unwrap();
                let sq = sq.get_or_insert_with(|| self.z.by_square());
                let (zm, x0, y0) = (self.z.zone_m, self.z.x0, self.z.y0);
                let sz_j = self.z.zone_at(sq, ((svc.st_x[a] - x0) / zm).floor() as i32, ((svc.st_y[a] - y0) / zm).floor() as i32);
                let (end, money, drop) = p.ride_station_end(&self.z, sz_j, End::Home);
                let extra = sz.acc_w[ka] - p.ride_w() * sz.acc_t[ka];
                // the car's cost is what is left: its per-trip part and its part by distance on the road there
                parts = [extra, p.p_ride, end, money, drop as u8 as f32, ride_home_park, extra - p.p_ride - end - ride_home_park];
            }
            body.extend([a as f64, t as f64, rep[a], rep[t], sz.acc_d[ka] as f64, sz.acc_w[ka] as f64, sz.acc_d[kt] as f64, sz.acc_w[kt] as f64, sk.r[a * s + t] as f64, share as f64, sz.acc_t[ka] as f64, sz.acc_t[kt] as f64, ride as u8 as f64]);
            body.extend(parts.iter().map(|&x| x as f64));
        }
        // routes over the bigger pairs, by their share of the whole trip's rail riders
        let mut all: Vec<(usize, f32, Vec<Leg>)> = vec![];
        for (pi, &(ka, kt, share)) in pairs.iter().enumerate() {
            if share < PAIR_MIN {
                break;
            }
            let (a, t) = (sz.acc_st[ka] as usize, sz.acc_st[kt] as usize);
            for (rs, legs) in routes(g, &sk, p, a, t, ROUTE_MIN / share) {
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
                out.extend(l.numbers());
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
    /// Every figure is the two cells' own trip (T-720: each cell's own walks to stations, the drive
    /// between the cells), except slots 37-40, the subzones' answer the demand views count.
    ///
    /// `HEAD` numbers: status (1 answered; 0 period `q` not solved here; -1 a point has no cell
    /// within `NEAR_CELL_M`), the period's crowding round, the two cells, their subzones (-1 none),
    /// the straight line between the cells and the drive's distance, metres; the walk share; the drive's real and perceived minutes, its parking and
    /// its whole cost (perceived); rail offered (1/0), rail's cost (the logsum, perceived, -1 if
    /// none), rail's share of the trips that do not walk, rail's utility against the road; the
    /// chance of walking, rail and driving; the cheapest perceived walk at each end; the base
    /// fare (perceived minutes) and perceived minutes per dollar; station pairs, routes; why rail
    /// is not offered (`RAIL_OFFERED`..`NO_ROUTE`); the two cells' x, y; the share of the rail
    /// riders on routes not listed; the car's ownership cost per trip (perceived minutes, T-620),
    /// which the whole cost includes and the parking leaves out; walking the whole way, real
    /// minutes (T-628); an everyday purpose's subzone pair's trips a day (0: a commute, or no
    /// sample joins the two subzones); the drive's length by road, km, and the weight a minute of walking the whole way counts at (T-699);
    /// the subzones' chance of walking, rail and driving, and their cheapest walk at the home end,
    /// real minutes (-1 none); whether the home end could drive to a station (`RIDE_TOO_SHORT`..
    /// `RIDE_NONE`, T-910), `Params::ride_min_km` and `Params::pair_walk_max`.
    ///
    /// Then per station pair, biggest share first, `PAIR_LEN` numbers: access and egress demand
    /// stations, a route node at each (-1 none), the access walk's straight-line metres and
    /// perceived minutes, the egress walk's, the station-to-station strategy cost (perceived
    /// minutes, the base fare and first boarding's transfer penalty not included), the pair's
    /// share of the trip's rail riders, the two walks' real minutes (the access entry's are a
    /// ride's drive minutes, its metres the straight line to the station), whether the access
    /// entry is a ride (1/0) and, for a ride (else 0), the perceived minutes beyond the drive's and
    /// their parts (T-915): the ride's fixed cost (`p_ride`), the parking or drop-off at the
    /// station, the part of that which is money (perceived), 1 if it is the drop-off, parking at
    /// home, and the car's ownership cost.
    ///
    /// Then per route, biggest first: its station pair's index, its share of the trip's rail
    /// riders, its legs; then `LEG_LEN` numbers a leg: kind (1 ride, 2 walk between stations or
    /// between the platforms of one, T-717), route nodes from and to, demand stations from and
    /// to, real minutes, perceived minutes, distance fare (perceived minutes), wait (real and
    /// perceived minutes), stops ridden, a ride's fixed cost of the change onto it (perceived
    /// minutes, 0 on the first ride), and the part of its wait that is for a train with room.
    pub fn trip(&self, q: usize, ox: f32, oy: f32, dx: f32, dy: f32) -> Vec<f64> {
        let mut h = vec![0f64; HEAD];
        let Some(st) = self.period(q) else { return h };
        h[1] = st.round as f64;
        let (Some(o), Some(d)) = (self.nearest_cell(ox, oy), self.nearest_cell(dx, dy)) else {
            h[0] = -1.0;
            return h;
        };
        self.explain(&st.g, &st.p, st.round, o, d, &Model::Commute)
    }
}

/// Whose choice a trip answer reads.
pub(crate) enum Model<'a> {
    Commute,
    Everyday(&'a TripInputs<'a>),
}

#[wasm_bindgen]
impl EverydayApi {
    /// The trip explorer for everyday purpose `kind` (T-586, T-201): `DemandApi::trip`'s answer
    /// for a trip from a home at (ox, oy) to the activity at (dx, dy), as the everyday layer
    /// chooses it in solved period `q` (its own period graph and parameters, the subzone pair's
    /// mean distance, the cells' walk share).
    pub fn trip(&self, kind: u32, q: usize, ox: f32, oy: f32, dx: f32, dy: f32) -> Vec<f64> {
        let mut h = vec![0f64; HEAD];
        let api = self.engine();
        if q >= PERIODS || self.solved_mask() >> q & 1 == 0 {
            return h;
        }
        let (Some(o), Some(d)) = (api.nearest_cell(ox, oy), api.nearest_cell(dx, dy)) else {
            h[0] = -1.0;
            return h;
        };
        let Some(ev) = self.trip_inputs(kind as usize, q, o, d) else { return h };
        api.explain(ev.g, ev.p, ev.round, o, d, &Model::Everyday(&ev))
    }
}

#[cfg(test)]
mod tests;
