//! T-866: the trip explorer between zones. The longer trips between a selection of local cells
//! and a far place (or one of its 6 km zones), as the market chooses them in one solved period:
//! each 6 km zone pair's choice computed with `IntercityApi::demand`'s expressions (the zones'
//! access lists, the station pairs' logsum, the logit against the pair's car or plane), every
//! solvable pair whether the sampled solve carries it or not (T-780), added up by the pair's trips;
//! then the station pairs and the routes the rail riders take through the period's strategies
//! (`trip::routes`). Nothing here changes any state.

use super::*;
use crate::demand::trip::{routes, Leg};

/// Numbers in the answer's head and per station pair (`IntercityApi::market_trip`); a leg is
/// `trip::LEG_LEN` numbers.
pub const MT_HEAD: usize = 22;
pub const MT_PAIR_LEN: usize = 7;
/// Station pairs under this share of the rail riders are not walked for routes, and routes under
/// it are left out (as `trip.rs`).
const PAIR_MIN: f64 = 0.005;
const ROUTE_MIN: f32 = 0.005;
const MAX_ROUTES: usize = 8;
const MAX_PAIRS: usize = 40;

/// Why rail is not offered (head slot 14).
const OK: f64 = 0.;
const FAR_NOT_OPEN: f64 = 1.;
const NO_RAIL: f64 = 2.;
const NOT_MATCHED: f64 = 3.;

impl IntercityApi {
    /// The selection's zone weights (`selection`'s): by people (`dir` 0) or attraction (1).
    fn trip_weights(&self, market: u32, xy: &[f32], share: &[f32], dir: u32) -> Vec<f64> {
        let m = &self.m;
        let mut w = vec![0f64; m.zones.len()];
        if xy.len() != 2 * share.len() || market as usize >= m.places.len() {
            return w;
        }
        for (i, &f) in share.iter().enumerate() {
            let key = ((xy[2 * i] / SITE_M).floor() as i32, (xy[2 * i + 1] / SITE_M).floor() as i32, market as usize);
            let Some(&c) = m.site_key.get(&key) else { continue };
            let f = (f as f64).clamp(0., 1.);
            let z = m.cell_zone[c];
            let (g, ci) = m.site(c);
            let zz = &m.zones[z];
            w[z] += if dir == 0 { f * g.city.pop[ci] as f64 / zz.pop.max(1e-12) } else { f * g.city.jobs[ci] as f64 / zz.w_attr.max(1e-12) };
        }
        w
    }

    /// The zones of place `place`, or only its 6 km zone `gid` (global id; -1 the whole place).
    fn zones_of(&self, place: u32, gid: i32) -> Vec<usize> {
        let m = &self.m;
        if gid >= 0 {
            m.comp_of_gid.get(gid as usize).filter(|&&c| c != u32::MAX).map(|&c| vec![c as usize]).unwrap_or_default()
        } else {
            (0..m.zones.len()).filter(|&z| m.zones[z].place == place as usize).collect()
        }
    }
}

#[wasm_bindgen]
impl IntercityApi {
    /// T-885: `market_trip` between two places of the market, or 6 km zones of them (global ids;
    /// -1 the whole place), every zone of the near end counted whole: the trips from `near` to
    /// `far` (`dir` 0) or from `far` to `near` (1). The same answer, but the everyday outings
    /// are those between the two in both directions.
    pub fn market_trip_between(&self, q: usize, day: u32, near: u32, near_gid: i32, dir: u32, far: u32, far_gid: i32) -> Vec<f64> {
        let mut w = vec![0f64; self.m.zones.len()];
        for z in self.zones_of(near, near_gid) {
            w[z] = 1.;
        }
        self.market_trip_w(q, day, w, near, dir, far, far_gid, true)
    }

    /// The trip explorer between zones (T-866) in solved period `q` on kind of day `day`: the
    /// longer trips of a selection (`xy`, `share`: 600 m squares of place `market`, as
    /// `selection_flows`) starting there (`dir` 0) or ending there (1), with far place `far`, or
    /// only its 6 km zone `far_gid` (global id; -1 the whole place). Packed, f64:
    ///
    /// `MT_HEAD` numbers: status (1 answered, 0 period not solved here), the period's crowding
    /// round; the period's trips, rail, plane and car trips; rail's perceived cost (the pairs'
    /// logsums, trip weighted over the pairs rail can serve; -1 none); the car's perceived minutes
    /// (trip weighted; -1 no car); the zone pairs' mean distance, km; the base fare (perceived
    /// minutes) and perceived minutes per dollar; station pairs, routes; the rail riders' share on
    /// routes not listed; why rail is not offered (0 offered, 1 the far place is not open here, 2
    /// no zone pair rail can serve, 3 no square matched a site); the share of the trips on pairs
    /// rail can serve; the reference mode's perceived minutes (the car's, or the plane's where no
    /// car goes), trip weighted; zone pairs counted; then (T-1229) the everyday outings across a
    /// zone line between the two ends, as the pairs choose them: legs a day, rail legs and walking
    /// legs (the rest go by car). The trips, rail, plane and car above are the longer trips alone;
    /// the station pairs and routes carry both kinds of rail riders.
    ///
    /// Then per station pair, most rail riders first, `MT_PAIR_LEN` numbers: a route node at the
    /// access station and at the egress one (-1 none), the perceived minutes to the first and from
    /// the second (their access weights' mean cost), the station-to-station strategy cost
    /// (perceived, the base fare not included), the pair's share of the rail riders, 0.
    ///
    /// Then per route, biggest first, as `DemandApi::trip`'s: its station pair's index, its share
    /// of the rail riders, its legs, and `trip::LEG_LEN` numbers a leg.
    pub fn market_trip(&self, q: usize, day: u32, market: u32, xy: &[f32], share: &[f32], dir: u32, far: u32, far_gid: i32) -> Vec<f64> {
        let w = self.trip_weights(market, xy, share, dir);
        self.market_trip_w(q, day, w, market, dir, far, far_gid, false)
    }
}

impl IntercityApi {
    /// The head's distance (slot 8), car minutes (7) and reference minutes (17) between two places
    /// from their rows alone, centre to centre; left as they are where either has no row.
    fn head_from_rows(&self, h: &mut [f64], near: u32, far: u32, dir: u32) {
        let (m, a, b) = (&self.m, near as usize, far as usize);
        if a == b || a >= m.places.len() || b >= m.places.len() || !m.places[a].has || !m.places[b].has {
            return;
        }
        let r = m.pr(a, b);
        h[8] = r.km;
        h[17] = r.refm as f64;
        if r.car {
            let to = if dir == 0 { b } else { a };
            h[7] = (car_min(r.km as f32, m.places[to].park as f32) + r.cross) as f64;
        }
    }

    /// `market_trip` for near zone weights `w` of place `near_place`. The longer trips are the
    /// same each way; the everyday outings are those of `dir` (homes at the near end for 0,
    /// activities there for 1), or with `both` of both.
    fn market_trip_w(&self, q: usize, day: u32, w: Vec<f64>, near_place: u32, dir: u32, far: u32, far_gid: i32, both: bool) -> Vec<f64> {
        let mut h = vec![0f64; MT_HEAD];
        let Some(per) = self.per(q) else { return h };
        h[0] = 1.;
        h[1] = per.round as f64;
        let (g, p) = (&per.g, &per.p);
        h[9] = p.fare_min as f64;
        h[10] = FARE_MIN_PER_USD as f64;
        h[6] = -1.;
        h[7] = -1.;
        let m = &self.m;
        let near: Vec<usize> = (0..w.len()).filter(|&z| w[z] > 0.).collect();
        let fars = self.zones_of(far, far_gid);
        if near.is_empty() || fars.is_empty() {
            // T-991: a place with no zones here still has its row: the distance and the car's
            // minutes are known, the train's are not
            self.head_from_rows(&mut h, near_place, far, dir);
            let closed = |p: u32| !self.m.zones.iter().any(|z| z.place == p as usize);
            h[14] = if fars.is_empty() || closed(near_place) { FAR_NOT_OPEN } else { NOT_MATCHED };
            return h;
        }
        // T-1229: the everyday outings across a zone line between the two ends: the pairs whose
        // home zone (`dir` 0) or activity zone (1) is a near zone and the other end a far zone
        let mut is_far = vec![false; m.zones.len()];
        for &b in &fars {
            is_far[b] = true;
        }
        let mut e_hits: Vec<(usize, f64)> = vec![];
        for end in if both { 0..2 } else { dir..dir + 1 } {
            let (off, items) = if end == 0 { &m.e_home } else { &m.e_act };
            for &z in &near {
                for &i in &items[off[z] as usize..off[z + 1] as usize] {
                    let pair = &m.everyday[i as usize];
                    let other = if end == 0 { m.comp_of_gid[pair.a as usize] as usize } else { pair.h as usize };
                    if other < is_far.len() && is_far[other] {
                        e_hits.push((i as usize, w[z]));
                    }
                }
            }
        }
        // strategies towards the egress stations of the pairs' destinations
        let s = g.n_st;
        let mut want = vec![false; s];
        for &z in if dir == 0 { &fars } else { &near } {
            for &(st, _) in &self.access_work[z] {
                want[st] = true;
            }
        }
        for &(i, _) in &e_hits {
            let ei = self.epair_of.get(i).copied().unwrap_or(u32::MAX);
            if ei != u32::MAX && per.e_frac.get(ei as usize).is_some_and(|&f| f > 0.) {
                for &(st, _) in &self.access_ew[self.epair_zones(&self.epairs[ei as usize]).1] {
                    want[st] = true;
                }
            }
        }
        let sk = skims_for(g, p, &want);
        let sh = share_out(day as usize, LONGER, q) as f64;
        let (mut all, mut rail, mut air, mut car) = (0f64, 0f64, 0f64, 0f64);
        let (mut rc, mut rc_w, mut cc, mut cc_w, mut ref_c, mut km, mut served, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        // (access, egress) -> rail trips, rail trips x access minutes, x egress minutes
        let mut by: HashMap<(u32, u32), (f64, f64, f64)> = HashMap::new();
        let mut terms: Vec<(u32, u32, f64)> = vec![];
        let beta = p.beta as f64;
        for &z in &near {
            for &b in &fars {
                if z == b {
                    continue;
                }
                let (a, bb) = if dir == 0 { (z, b) } else { (b, z) };
                let (za, zb) = (&m.zones[a], &m.zones[bb]);
                let trips = m.eachway(za, zb);
                if trips < m.rule.min_trips {
                    continue;
                }
                let t = w[z] * trips * sh;
                let r = m.pr(za.place, zb.place);
                let d = m.zone_km(za, zb) as f32;
                let car_c = car_min(d, zb.park) + r.cross;
                let drive = if r.car { car_c } else { r.refm };
                all += t;
                km += t * d as f64;
                ref_c += t * drive as f64;
                n += 1.;
                if r.car {
                    cc += t * car_c as f64;
                    cc_w += t;
                }
                let mut frac = 0f64;
                if self.solvable(a, bb) {
                    let band = self.base.band_of_d2(d * d * 1e6);
                    let (ra, rb) = (&self.access[a], &self.access_work[bb]);
                    let (rail_c, sum) = rail_logsum(&sk.r, s, p.fare_min, ra, rb, p.beta, p.soften(p.band_unsoftened(band), car_c), &mut terms);
                    if sum > 0. {
                        let bt = p.mode_beta(car_c);
                        let v = ((-bt * (rail_c - drive)).clamp(-80., 80.) as f64).exp();
                        frac = v / (r.rest as f64 + v);
                        served += t;
                        rc += t * rail_c as f64;
                        rc_w += t;
                        let rt = t * frac;
                        let cost = |list: &[(usize, f32)], st: u32| list.iter().find(|e| e.0 == st as usize).map_or(0., |e| -(e.1.max(1e-30) as f64).ln() / beta);
                        for &(sa, sb, v) in &terms {
                            let x = rt * v / sum;
                            let e = by.entry((sa, sb)).or_default();
                            e.0 += x;
                            e.1 += x * cost(ra, sa);
                            e.2 += x * cost(rb, sb);
                        }
                    }
                }
                rail += t * frac;
                let rest = t * (1. - frac);
                air += rest * r.air as f64;
                car += rest * if r.car { 1. - r.air as f64 } else { 0. };
            }
        }
        // T-1229: each everyday pair's legs this period, its rail legs from the solve's share and the
        // walkers rail leaves; its riders join the station pairs
        let pe = commute_params().everyday();
        let (mut e_all, mut e_rail, mut e_walk) = (0f64, 0f64, 0f64);
        let weigh = |list: &[(usize, f32)], st: u32| list.iter().find(|e| e.0 == st as usize).map_or(0., |e| -(e.1.max(1e-30) as f64).ln() / pe.beta as f64);
        for &(i, wz) in &e_hits {
            let pair = &m.everyday[i];
            let k = pair.kind as usize;
            let purpose = everyday_purpose(k);
            let n = wz * pair.trips as f64;
            let legs = n * (share_out(day as usize, purpose, q) + share_ret(day as usize, purpose, q)) as f64;
            let ei = self.epair_of.get(i).copied().unwrap_or(u32::MAX);
            let frac = if ei == u32::MAX { 0. } else { per.e_frac.get(ei as usize).copied().unwrap_or(0.) as f64 };
            let rail_legs = n * (OUTWARD[k][q] + RETURN[k][q]) as f64 * load_factor(day as usize, purpose, q) as f64 * frac;
            e_all += legs;
            e_rail += rail_legs;
            e_walk += pair.walk as f64 * (legs - rail_legs).max(0.);
            if rail_legs <= 0. {
                continue;
            }
            let ep = &self.epairs[ei as usize];
            let (hz, az) = self.epair_zones(ep);
            let home = if ep.ride { &self.access_c[hz] } else { &self.access_c_walk[hz] };
            let (_, sum) = rail_logsum(&sk.r, s, p.fare_min, home, &self.access_ew[az], pe.beta, p.soften(pe.band_unsoftened(ep.band as usize), ep.car), &mut terms);
            if sum > 0. {
                for &(sa, sb, v) in &terms {
                    let x = rail_legs * v / sum;
                    let e = by.entry((sa, sb)).or_default();
                    e.0 += x;
                    e.1 += x * weigh(home, sa);
                    e.2 += x * weigh(&self.access_ew[az], sb);
                }
            }
        }
        h[19] = e_all;
        h[20] = e_rail;
        h[21] = e_walk;
        let rail_all = rail + e_rail;
        h[2] = all;
        h[3] = rail;
        h[4] = air;
        h[5] = car;
        if rc_w > 0. {
            h[6] = rc / rc_w;
        }
        if cc_w > 0. {
            h[7] = cc / cc_w;
        }
        if all > 0. {
            h[8] = km / all;
            h[15] = served / all;
            h[17] = ref_c / all;
        }
        h[18] = n;
        if rail_all <= 0. {
            h[14] = NO_RAIL;
            h[13] = 1.;
            return h;
        }
        let mut pairs: Vec<((u32, u32), (f64, f64, f64))> = by.into_iter().filter(|e| e.1 .0 > 0.).collect();
        pairs.sort_by(|x, y| y.1 .0.total_cmp(&x.1 .0).then(x.0.cmp(&y.0)));
        pairs.truncate(MAX_PAIRS);
        let mut rep = vec![-1f64; s];
        for (r, &stn) in g.rn_station.iter().enumerate().rev() {
            rep[stn as usize] = r as f64;
        }
        let mut body = vec![];
        for &((sa, sb), (x, xa, xb)) in &pairs {
            body.extend([rep[sa as usize], rep[sb as usize], xa / x, xb / x, sk.r[sa as usize * s + sb as usize] as f64, x / rail_all, 0.]);
        }
        let mut all_routes: Vec<(usize, f32, Vec<Leg>)> = vec![];
        for (pi, &((sa, sb), (x, _, _))) in pairs.iter().enumerate() {
            let share = (x / rail_all) as f32;
            if (share as f64) < PAIR_MIN {
                break;
            }
            for (rs, legs) in routes(g, &sk, p, sa as usize, sb as usize, ROUTE_MIN / share) {
                all_routes.push((pi, rs * share, legs));
            }
        }
        all_routes.sort_by(|x, y| y.1.total_cmp(&x.1));
        all_routes.truncate(MAX_ROUTES);
        all_routes.retain(|r| r.1 >= ROUTE_MIN);
        h[11] = pairs.len() as f64;
        h[12] = all_routes.len() as f64;
        h[13] = (1. - all_routes.iter().map(|r| r.1 as f64).sum::<f64>()).max(0.);
        h[14] = OK;
        let mut out = h;
        out.extend(body);
        for (pi, share, legs) in &all_routes {
            out.extend([*pi as f64, *share as f64, legs.len() as f64]);
            for l in legs {
                out.extend(l.numbers());
            }
        }
        out
    }
}
