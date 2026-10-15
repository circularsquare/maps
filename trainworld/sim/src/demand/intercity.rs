//! T-117: a separate, tunable non-commute long-distance market on the common rail graph.
//!
//! T-174: opening never visits every zone pair. The figures that need all of them (each zone's
//! gravity normaliser, the day's trips, each corridor's trips) are computed offline by the same
//! code (`derive`, run by `bin/market_pack.rs` from pipeline/intercity.py) into the pack's
//! `derived` block, and the zone pairs themselves are built only between zones a station serves,
//! when the network is set: no other pair can carry a rail trip. A pack whose block is missing or
//! stale (another formula, other sites) is derived at opening instead: the same result, slower.
//!
//! T-128: zone distances (the decay and the drive time) are great circles on the ground when the
//! pack names the frame its x and y are in (`frame`, written by pipeline/intercity.py since
//! T-128); a pack without one keeps the straight distance in its plane and its old formula.
use super::{
    api::{demand_groups, merge_stations, same_bits, scale_triples, DemandApi, Service, FARE_MIN_PER_USD, PEAK_HOUR_FACTOR, TO_HOME, TO_WORK},
    clock::now_ms,
    days::{load_factor, share_out, share_ret, LONGER, WEEKDAY, WORK},
    kernel::*,
    network::PERIODS,
    pack::City,
};
use crate::track::{
    frame::Frame,
    params::{EARTH_R, PERIOD_HOURS},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use wasm_bindgen::prelude::*;

pub const SHARES: [f32; PERIODS] = [0.20, 0.35, 0.25, 0.15, 0.05];
// Only the numeric runtime inputs are retained. Deserializing a generic Value builds
// an individually allocated JSON array for every site and keeps it alive while pairs
// are allocated; WASM cannot return that high-water memory to the browser.
#[derive(Deserialize)]
struct MarketPack {
    #[serde(default = "default_zone_m")]
    zone_m: f64,
    #[serde(default = "default_trip_rate")]
    trip_rate: f64,
    #[serde(default = "legacy_markets")]
    markets: Vec<String>,
    sites: Vec<[f64; 5]>,
    /// T-128: the frame the sites' x and y are in. With it, zone distances (and so the decay and
    /// drive times) are measured on the ground; without it (packs before T-128), straight in the
    /// plane as before, under the old formula, so their derived blocks stay current.
    #[serde(default)]
    frame: Option<PackFrame>,
    /// T-174: the all-pairs figures, written offline by `derive`; used only if still current (a
    /// block that does not read as `Derived` is ignored like a stale one).
    #[serde(default)]
    derived: Option<serde_json::Value>,
    /// T-276: the outside world, everyone living in no zone (pipeline/outside.py): x, y in the
    /// sites' frame and attraction. Part of every zone's normaliser, so a share of each zone's
    /// trips leaves the map. Empty: a closed world, as before.
    #[serde(default)]
    outside: Vec<[f64; 3]>,
    /// T-276: per site, the local layer's parking at a destination there (perceived minutes, its
    /// 2 km zone's jobs per km²); the car pays it on arrival. Empty: no parking.
    #[serde(default)]
    site_park: Vec<f32>,
    /// T-276: commutes between zones (the cross-pack part of the joint commute gravity,
    /// `bin/joint_gravity.rs`): home site, work site (any site of each 6 km zone) and commuters a
    /// day. Empty: none.
    #[serde(default)]
    commute_home: Vec<u32>,
    #[serde(default)]
    commute_work: Vec<u32>,
    #[serde(default)]
    commute_trips: Vec<f32>,
}
/// The pack's `frame`: `kind` "equirect" or "lcc" (as the registry writes frames), origin in degrees.
#[derive(Deserialize)]
struct PackFrame {
    kind: String,
    lon: f64,
    lat: f64,
}
/// Names the market's zone-pair formula (`zone_km`, `decay_km`, `pair_trips`, `MIN_TRIPS`, the
/// outside world's term, `car_min`). Change it with any of them, so packs derived under the old
/// formula are recomputed at opening (and `pipeline/intercity.py --derive` refreshes them).
/// `probe` also catches a changed `decay_km`.
const FORMULA: &str = "T-276 v3: great-circle zone distances through the pack's frame, onset (d-25)/25 km, max(d,25)^-2.37 (NHTS 2017), an outside world in every normaliser, mean of both directions' shares, pairs under 0.01 trips dropped, car 15 min + d at 68 km/h plus running cost and half the parking";
/// The formula of a pack without a `frame` (before T-128): distances straight in its plane.
const FORMULA_PLANE: &str = "T-276 v3 plane: as v3 with distances straight in the pack's plane";
fn formula(ground: bool) -> &'static str {
    if ground {
        FORMULA
    } else {
        FORMULA_PLANE
    }
}
/// The figures of the market that need every zone pair (T-174). O(zones) numbers. Floats are
/// written as their bit patterns (u64 integers), which JSON carries exactly: serde_json's default
/// float parser can be a unit in the last place off, and the game should open the same market
/// from the block as without it.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Derived {
    /// `FORMULA` when derived
    formula: String,
    /// `decay` at a few distances, to notice a changed decay that kept its `FORMULA` (to 1e-9)
    #[serde(with = "bits")]
    probe: Vec<f64>,
    /// sum of the zones' x, y, people and attraction: notices changed sites, zone size or order
    #[serde(with = "bit")]
    check: f64,
    /// per zone, its row normaliser: the sum of attraction times decay over every other zone
    #[serde(with = "bits")]
    totals: Vec<f64>,
    /// the day's trips (both directions of every pair)
    #[serde(with = "bit")]
    daily: f64,
    /// per corridor (market pair, `corridors` order), the day's trips
    #[serde(with = "bits")]
    corridor_trips: Vec<f64>,
    /// directed zone pairs in the market
    pairs: usize,
    /// T-276: the day's departures that leave the map (to the outside world)
    #[serde(default, with = "bit")]
    leaving: f64,
}
/// An f64 as its bit pattern in JSON (`Derived`).
mod bit {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(v.to_bits())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(f64::from_bits(u64::deserialize(d)?))
    }
}
/// f64s as their bit patterns in JSON (`Derived`).
mod bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<u64>::deserialize(d)?.into_iter().map(f64::from_bits).collect())
    }
}
fn legacy_markets() -> Vec<String> {
    ["New York", "Philadelphia", "Trenton / Mercer County"]
        .map(str::to_owned)
        .to_vec()
}
fn default_zone_m() -> f64 {
    6000.
}
fn default_trip_rate() -> f64 {
    0.015
}
struct Zone {
    x: f32,
    y: f32,
    pop: f64,
    attr: f64,
    market: usize,
    /// T-128: the zone's centre on the unit sphere (from x, y through the pack's frame); zeros
    /// in a pack without a frame
    u: [f64; 3],
    /// T-276: parking for a car arriving here, perceived minutes (its sites' `site_park`,
    /// weighted by their jobs); 0 in a pack without it
    park: f32,
}
/// T-276: a point of the outside world (`MarketPack::outside`), as a zone's distance needs it.
struct Outside {
    x: f32,
    y: f32,
    u: [f64; 3],
    attr: f64,
}
struct Pair {
    a: usize,
    b: usize,
    trips: f64,
    drive: f32,
    corridor: usize,
}
/// T-276: a commute between two zones some station serves: its index in `commutes`, and the
/// car's perceived minutes from home to work (the local layer's commute rule).
struct CPair {
    c: u32,
    car: f32,
    corridor: usize,
    /// its zones' centres at least `ride_min_km` apart: the home end may ride (T-373)
    ride: bool,
}
/// T-276: the commutes between zones are the local layer's commuters, so they choose as it does:
/// a logit at its scale, the drive weighted as congested driving plus parking at work, and the
/// local rail propensity (`kernel::Params` defaults, as `DemandApi`'s).
fn commute_params() -> Params {
    Params::default()
}
struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    /// every part's loads together (crowding reads these), and the commutes' alone (T-276)
    avg: Loads,
    c_avg: Loads,
    /// the longer trips' rail and trips
    rail: f64,
    trips: f64,
    /// T-276: the commutes' rail legs and legs, and each commute pair's rail fraction (`cpairs`
    /// order), averaged over rounds like the loads
    c_rail: f64,
    c_trips: f64,
    c_frac: Vec<f32>,
    /// per corridor, the longer trips' rail
    corridors: Vec<f64>,
    /// T-276: per corridor, the commutes' rail legs
    c_corridors: Vec<f64>,
    /// Rail trips of each directed pair in this period (T-118), averaged over rounds like the loads.
    pair_rail: Vec<f32>,
    /// Per station, `2 * zones` numbers (T-118, T-120): rail trips starting here by destination
    /// zone, then ending here by origin zone. f32: it is one cell per station and zone.
    st: Vec<f32>,
    round: usize,
    ms: f64,
    /// As `api::Period`: the factors of the last pass, and whether every pass used them.
    fac: Vec<f32>,
    uniform: bool,
    /// As `api::Period`: transfers platform to platform (T-176), averaged like the loads.
    xi: XferIndex,
    xfer: Vec<f32>,
}
// Legacy fixtures retain the three-market layout. Runtime widths follow the pack.
#[cfg(test)]
const ST_W: usize = 18;
/// Side of the endpoint sites in metres (pipeline/intercity.py groups cells into 600 m squares).
const SITE_M: f32 = 600.;
#[cfg(test)]
const MARKETS: usize = 3;
/// What `pass` hands back besides the loads.
struct Pass {
    /// both parts together, and the commutes alone
    loads: Loads,
    c_loads: Loads,
    rail: f64,
    c_rail: f64,
    c_frac: Vec<f32>,
    cor: Vec<f64>,
    c_cor: Vec<f64>,
    pair_rail: Vec<f32>,
    st: Vec<f32>,
    /// laid out by the `XferIndex` the pass was given
    xfer: Vec<f32>,
}
#[wasm_bindgen]
pub struct IntercityApi {
    city: City,
    zones: Vec<Zone>,
    cell_zone: Vec<usize>,
    /// The zone pairs between zones some station serves (T-174), both directions, in the order
    /// the whole market lists them (by lower zone, then higher). Built by `set_network`.
    pairs: Vec<Pair>,
    /// The zones `pairs` was built for (ascending).
    served: Vec<usize>,
    /// (origin zone, destination zone) to its index in `pairs`.
    pair_index: HashMap<(u32, u32), u32>,
    /// The all-pairs figures: from the pack, or derived at opening (`from_pack` false).
    derived: Derived,
    from_pack: bool,
    /// T-128: zone distances on the ground (the pack names its frame), else in its plane
    ground: bool,
    rate: f64,
    corridor_of: Vec<usize>,
    base: Params,
    svc: Option<Service>,
    access: Vec<Vec<(usize, f32)>>,
    access_work: Vec<Vec<(usize, f32)>>,
    /// T-276: the same at the commute's logit scale (`commute_params`), homes then jobs; T-373:
    /// homes with rides, and by walking alone for commutes too short for a ride
    access_c: Vec<Vec<(usize, f32)>>,
    access_cw: Vec<Vec<(usize, f32)>>,
    access_c_walk: Vec<Vec<(usize, f32)>>,
    /// T-373: the sites' 2 km zones (homes and jobs, as the local layer's), whose centres the
    /// rides start from (`access_options`), and those centres
    kz: Zones,
    kplace: Vec<(usize, f32, f32)>,
    /// T-276: commutes between zones: (home zone, work zone, commuters a day), from the pack
    commutes: Vec<(u32, u32, f32)>,
    /// per zone, its commutes as the home end and as the work end (CSR offsets and `commutes` indices)
    c_home: (Vec<u32>, Vec<u32>),
    c_work: (Vec<u32>, Vec<u32>),
    /// per corridor, commute legs a day (both legs of every commuter)
    c_corridor: Vec<f64>,
    /// the commutes between served zones (built with `pairs`), and each commute's index in it
    cpairs: Vec<CPair>,
    cpair_of: Vec<u32>,
    /// The stations a trip can end at: the only ones strategies are searched to (T-119).
    want: Vec<bool>,
    per: Vec<Option<Period>>,
    fare_km: Vec<f32>,
    fare_per_km: f32,
    corridors: Vec<(usize, usize)>,
    markets: usize,
    daily: f64,
    /// Site of a 600 m square of a market, to find the sites a selection of local cells holds.
    site_key: HashMap<(i32, i32, usize), usize>,
    /// The complex of each station of the next `set_network` (T-161, `DemandApi::set_complexes`).
    complex_in: Vec<u32>,
}

impl IntercityApi {
    fn pass(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> Pass {
        // the strategies are shared with the local populations on the same service (T-119)
        with_skims(g, p, &self.want, |sk| self.pass_on(g, p, q, sk, ix))
    }
    fn pass_on(&self, g: &Graph, p: &Params, q: usize, sk: &Skims, ix: &XferIndex) -> Pass {
        let s = g.n_st;
        let mut od = vec![0.; s * s];
        let mut rail = 0.;
        let mut cor = vec![0.; self.corridors.len()];
        let mut pair_rail = vec![0f32; self.pairs.len()];
        let nz = self.zones.len();
        let w = 2 * nz;
        let mut st = vec![0f32; s * w];
        for (pi, pair) in self.pairs.iter().enumerate() {
            let trips = pair.trips * SHARES[q] as f64;
            let (a, b) = (&self.access[pair.a], &self.access_work[pair.b]);
            let mut terms = Vec::with_capacity(a.len() * b.len());
            let mut sum = 0f64;
            for &(sa, wa) in a {
                for &(sb, wb) in b {
                    if sa == sb {
                        continue;
                    } // A rail journey must actually board a train.
                    let cost = sk.r[sa * s + sb] + p.fare_min;
                    if !cost.is_finite() {
                        continue;
                    }
                    let v = wa as f64
                        * wb as f64
                        * ((-p.beta * (cost - pair.drive)).clamp(-80., 80.) as f64).exp();
                    sum += v;
                    terms.push((sa * s + sb, v));
                }
            }
            let fraction = sum / (1. + sum);
            let r = trips * fraction;
            rail += r;
            cor[pair.corridor] += r;
            pair_rail[pi] = r as f32;
            if sum > 0. {
                for (k, v) in terms {
                    let rr = r * v / sum;
                    od[k] += rr as f32;
                    let (sa, sb) = (k / s, k % s);
                    st[sa * w + pair.b] += rr as f32;
                    st[sb * w + nz + pair.a] += rr as f32;
                }
            }
        }
        // T-276: commutes between zones, chosen as the local layer chooses: once per commuter on
        // this period's service from home to work, the trip home on the same station pair reversed
        // (docs/demand.md 4.2), each leg at its own period share
        let pc = commute_params();
        let mut od_c = vec![0f32; if self.cpairs.is_empty() { 0 } else { s * s }];
        let (mut c_rail, mut c_frac) = (0f64, vec![0f32; self.cpairs.len()]);
        let mut c_cor = vec![0f64; self.corridors.len()];
        let mut terms = vec![];
        for (ci, cp) in self.cpairs.iter().enumerate() {
            let (h, wz, n) = self.commutes[cp.c as usize];
            let (h, wz) = (h as usize, wz as usize);
            let (out, ret) = (n as f64 * TO_WORK[q] as f64, n as f64 * TO_HOME[q] as f64);
            terms.clear();
            let mut sum = 0f64;
            let home = if cp.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            for &(sa, wa) in home {
                for &(sb, wb) in &self.access_cw[wz] {
                    if sa == sb {
                        continue;
                    }
                    let cost = sk.r[sa * s + sb] + p.fare_min;
                    if !cost.is_finite() {
                        continue;
                    }
                    let v = wa as f64 * wb as f64 * ((-pc.beta * (cost - cp.car) + pc.asc_rail).clamp(-80., 80.) as f64).exp();
                    sum += v;
                    terms.push((sa, sb, v));
                }
            }
            if sum <= 0. {
                continue;
            }
            let frac = sum / (1. + sum);
            c_frac[ci] = frac as f32;
            let (ro, rr_) = (out * frac, ret * frac);
            c_rail += ro + rr_;
            c_cor[cp.corridor] += ro + rr_;
            for &(sa, sb, v) in &terms {
                let (a, b) = ((ro * v / sum) as f32, (rr_ * v / sum) as f32);
                od_c[sa * s + sb] += a;
                od_c[sb * s + sa] += b;
                st[sa * w + wz] += a;
                st[sb * w + nz + h] += a;
                st[sb * w + h] += b;
                st[sa * w + nz + wz] += b;
            }
        }
        let c_loads = if od_c.is_empty() {
            Loads::zero(g.n_st, g.n - g.n_st)
        } else {
            let l = assign(g, sk, &od_c);
            for (a, b) in od.iter_mut().zip(&od_c) {
                *a += b;
            }
            l
        };
        let mut xfer = vec![0.0; ix.len];
        Pass {
            loads: assign_with_transfers(g, sk, &od, ix, &mut xfer),
            c_loads,
            rail,
            c_rail,
            c_frac,
            cor,
            c_cor,
            pair_rail,
            st,
            xfer,
        }
    }
    fn per(&self, q: usize) -> Option<&Period> {
        self.per.get(q).and_then(|v| v.as_ref())
    }
    /// The pairs between the `served` zones (ascending), in the whole market's order (T-174).
    fn set_served(&mut self, served: Vec<usize>) {
        let mut pairs = vec![];
        for (i, &a) in served.iter().enumerate() {
            pair_row(&self.zones, self.ground, &self.derived.totals, self.rate, &self.corridor_of, self.markets, a, &served[i + 1..], &mut pairs);
        }
        pairs.shrink_to_fit();
        self.pair_index = pairs.iter().enumerate().map(|(i, p)| ((p.a as u32, p.b as u32), i as u32)).collect();
        self.pairs = pairs;
        // T-276: the commutes between served zones, in the pack's order
        let mut is_served = vec![false; self.zones.len()];
        for &z in &served {
            is_served[z] = true;
        }
        let pc = commute_params();
        self.cpairs.clear();
        self.cpair_of = vec![u32::MAX; self.commutes.len()];
        for (c, &(h, w, _)) in self.commutes.iter().enumerate() {
            let (h, w) = (h as usize, w as usize);
            if !is_served[h] || !is_served[w] {
                continue;
            }
            let (zh, zw) = (&self.zones[h], &self.zones[w]);
            let km = zone_km(zh, zw, self.ground);
            let car = drive_min(km) * pc.w_drive + zw.park;
            let corridor = self.corridor_of[zh.market.min(zw.market) * self.markets + zh.market.max(zw.market)];
            self.cpair_of[c] = self.cpairs.len() as u32;
            self.cpairs.push(CPair { c: c as u32, car, corridor, ride: km >= pc.ride_min_km });
        }
        self.cpairs.shrink_to_fit();
        self.served = served;
    }
    /// Every pair of the market, as the served pairs would be with every zone served.
    #[cfg(test)]
    fn all_pairs(&self) -> Vec<Pair> {
        let all: Vec<usize> = (0..self.zones.len()).collect();
        let mut pairs = vec![];
        for a in 0..all.len() {
            pair_row(&self.zones, self.ground, &self.derived.totals, self.rate, &self.corridor_of, self.markets, a, &all[a + 1..], &mut pairs);
        }
        pairs
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intercity_prunes_symmetric_pairs_after_averaging() {
        // Directed departures are .03 and .0015: only one passes the old threshold.
        let keep = IntercityApi::new(r#"{"sites":[[0,0,2,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
        let pairs = keep.all_pairs();
        assert_eq!(pairs.len(), 2);
        assert_eq!(keep.derived.pairs, 2);
        assert_eq!(pairs[0].trips, pairs[1].trips);
        assert!((pairs[0].trips - 0.01575).abs() < 1e-12);
        assert_eq!((pairs[0].a, pairs[0].b), (pairs[1].b, pairs[1].a));
        // .015 and .0015 average below .01: both directions must disappear.
        let drop = IntercityApi::new(r#"{"sites":[[0,0,1,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
        assert!(drop.all_pairs().is_empty());
        assert_eq!((drop.derived.pairs, drop.daily), (0, 0.));
    }
    fn api() -> IntercityApi {
        let header = r#"{"zone_m":6000,"trip_rate":0.015,"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]]}"#;
        IntercityApi::new(header).unwrap()
    }
    fn network(api: &mut IntercityApi, freq: f32, slow: f32) {
        // Local feeder at each end, through intercity line, Mercer intermediate stop.
        let xy = [0., 0., 1000., 0., 60000., 0., 119000., 0., 120000., 0.];
        let n = [2, 3, 2];
        let stops = [0, 1, 1, 2, 3, 3, 4];
        let mut times = vec![0.; 42];
        for (a, len, sec) in [(0, 2, 120.), (2, 3, 1800. * slow), (5, 2, 120.)] {
            for lev in 0..3 {
                for k in 0..len - 1 {
                    times[6 * a + lev * len + k] = sec;
                    times[6 * a + (3 + lev) * len + k + 1] = sec;
                }
            }
        }
        api.set_network(
            &xy,
            &n,
            &stops,
            &times,
            &[12., 12., 12., freq, freq, freq, 12., 12., 12.],
            &[4., 4., 4.],
        )
        .unwrap();
        api.set_fares(
            0.,
            0.,
            &[1., 0., 0., 1., 59., 59., 0., 0., 59., 59., 1., 0., 0., 1.],
        )
        .unwrap();
    }
    #[test]
    fn intercity_market_is_symmetric_and_conserved() {
        let a = api();
        assert!((a.daily - 3600.).abs() < 1e-5);
        let pairs = a.all_pairs();
        assert_eq!(pairs.len(), a.derived.pairs);
        assert_eq!(a.daily, pairs.iter().map(|p| p.trips).sum::<f64>());
        for p in &pairs {
            let reverse = pairs.iter().find(|q| q.a == p.b && q.b == p.a).unwrap();
            assert_eq!(p.trips, reverse.trips);
        }
        assert!((SHARES.iter().sum::<f32>() - 1.).abs() < 1e-6);
    }
    #[test]
    fn intercity_uses_feeders_fares_service_and_shared_crowding() {
        let mut a = api();
        network(&mut a, 6., 1.);
        a.solve(0);
        let good = a.summary(0)[2];
        assert!(good > 100.);
        let board = a.board(0);
        assert!(
            board[0] > 0. && board[5] > 0.,
            "local feeder boardings: {board:?}"
        );
        assert!((a.entries(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
        assert!((a.exits(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
        let sum = a
            .corridor_summary(0)
            .chunks_exact(4)
            .map(|v| v[3])
            .sum::<f64>();
        assert!((sum - good).abs() < 1e-6);
        let loads = vec![1e7; a.seg(0).len()];
        assert!(a.crowd_with_external(0, &loads));
        assert!(a.summary(0)[2] < good);
        a.set_fares(80., 0., &[0.; 14]).unwrap();
        a.solve(0);
        assert!(a.summary(0)[2] < good * 0.1);
        network(&mut a, 6., 3.);
        a.solve(0);
        assert!(a.summary(0)[2] < good);
        network(&mut a, 0., 1.);
        a.solve(0);
        assert_eq!(a.summary(0)[2], 0.);
    }
    #[test]
    fn intercity_access_respects_home_and_destination_weights() {
        let header = r#"{"sites":[[0,0,100000,0,0],[4000,0,0,100000,0],[100000,0,100000,0,1],[104000,0,0,100000,1]]}"#;
        let mut a = IntercityApi::new(header).unwrap();
        let mut times = vec![0.; 24];
        for lev in 0..3 {
            for k in 0..3 {
                let sec = if k == 1 { 2400. } else { 120. };
                times[lev * 4 + k] = sec;
                times[(3 + lev) * 4 + k + 1] = sec;
            }
        }
        a.set_network(
            &[0., 0., 4000., 0., 100000., 0., 104000., 0.],
            &[4],
            &[0, 1, 2, 3],
            &times,
            &[6., 6., 6.],
            &[4.],
        )
        .unwrap();
        a.solve(0);
        let e = a.entries(0);
        let x = a.exits(0);
        assert!(e[0] > e[1], "origins follow homes: {e:?}");
        assert!(x[1] > x[0], "destinations favour jobs: {x:?}");
    }
    fn solve_all(a: &mut IntercityApi) {
        for q in 0..PERIODS {
            assert!(a.solve(q));
        }
    }
    fn sum_markets(v: &[f64], off: usize) -> f64 {
        (0..MARKETS).map(|m| v[1 + 8 * m + off]).sum()
    }
    #[test]
    fn intercity_selection_places_conserve_and_rail_is_within_all() {
        let mut a = api();
        network(&mut a, 6., 1.);
        solve_all(&mut a);
        assert_eq!(a.solved_mask(), 31);
        // The whole of market 0 (one site at the origin): starts equal its zone's departures.
        let all = a.selection_flows(0, &[0., 0.], &[1.]);
        assert_eq!(all[0], 1.);
        let pairs = a.all_pairs();
        let want: f64 = pairs
            .iter()
            .filter(|p| a.zones[p.a].market == 0)
            .map(|p| p.trips)
            .sum();
        assert!(want > 0.);
        assert!(
            (sum_markets(&all, 0) - want).abs() < 1e-6 * want.max(1.),
            "{all:?} vs {want}"
        );
        let want_in: f64 = pairs
            .iter()
            .filter(|p| a.zones[p.b].market == 0)
            .map(|p| p.trips)
            .sum();
        assert!((sum_markets(&all, 4) - want_in).abs() < 1e-6 * want_in.max(1.));
        for m in 0..MARKETS {
            for e in [0, 4] {
                let (trips, rail) = (all[1 + 8 * m + e], all[2 + 8 * m + e]);
                assert!(
                    rail <= trips + 1e-9,
                    "market {m}: rail {rail} above all {trips}"
                );
            }
        }
        assert!(
            sum_markets(&all, 1) > 0.,
            "service carries some of the selection"
        );
        // A part of the site is that part of everything; a point off every site matches nothing.
        let half = a.selection_flows(0, &[0., 0.], &[0.5]);
        for k in 1..1 + 8 * MARKETS {
            assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
        }
        let none = a.selection_flows(0, &[9e6, 9e6], &[1.]);
        assert_eq!(none[0], 0.);
        let head = 1 + 8 * MARKETS;
        let nz = a.zones.len();
        assert!(none[1..head + 4 * nz].iter().all(|v| *v == 0.));
        // per far zone (T-120): the sums equal the market sums, rail <= all, positions are the zones' own
        assert_eq!(all.len(), head + 7 * nz + 4 * MARKETS + 4 * nz);
        for d in 0..2 {
            let (mut tot, mut rail, mut sx) = (0., 0., 0.);
            for z in 0..nz {
                let (t, r) = (all[head + 4 * z + 2 * d], all[head + 4 * z + 2 * d + 1]);
                assert!(r <= t + 1e-9, "zone {z}: rail {r} above all {t}");
                tot += t;
                rail += r;
                sx += t * a.zones[z].x as f64;
            }
            let (mt, mr) = (sum_markets(&all, 4 * d), sum_markets(&all, 4 * d + 1));
            assert!((tot - mt).abs() < 1e-9 * mt.max(1.) && (rail - mr).abs() < 1e-9 * mr.max(1.));
            assert!((sx - sum_markets(&all, 4 * d + 2)).abs() < 1e-6 * sx.abs().max(1.));
        }
        for z in 0..nz {
            assert_eq!(all[head + 4 * nz + 3 * z], a.zones[z].x as f64);
            assert_eq!(all[head + 4 * nz + 3 * z + 1], a.zones[z].y as f64);
            assert_eq!(all[head + 4 * nz + 3 * z + 2], a.zones[z].market as f64);
        }
        // a half share gives half of every zone too
        for k in head..head + 4 * nz {
            assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
        }
        // The far centroid of New York's trips lies between the other two places.
        let (sx, tot) = (all[1 + 8 + 2], all[1 + 8]);
        assert!(
            tot > 0. && sx / tot > 100_000.,
            "Philadelphia sits 120 km east: {}",
            sx / tot
        );
    }
    #[test]
    fn intercity_station_places_equal_entries_and_exits() {
        let mut a = api();
        network(&mut a, 6., 1.);
        solve_all(&mut a);
        // the network's five stations: 0 and 1 are local feeders, 2 is Mercer, 3 and 4 Philadelphia's
        let map = a.station_map();
        let mut starts = 0.;
        let mut ends = 0.;
        for st in 0..5u32 {
            let row = a.station_flows(st);
            let nz = a.zones.len();
            assert_eq!(row.len(), ST_W + 5 * nz);
            // the zone rows add up to the market sums, and rail sits at the zones' own positions
            for d in 0..2 {
                let zsum: f64 = row[ST_W + d * nz..ST_W + (d + 1) * nz].iter().sum();
                let msum: f64 = row[d * MARKETS..(d + 1) * MARKETS].iter().sum();
                assert!((zsum - msum).abs() < 1e-9 * msum.max(1.));
            }
            for z in 0..nz {
                assert_eq!(row[ST_W + 2 * nz + 3 * z], a.zones[z].x as f64);
                assert_eq!(row[ST_W + 2 * nz + 3 * z + 1], a.zones[z].y as f64);
                assert_eq!(row[ST_W + 2 * nz + 3 * z + 2], a.zones[z].market as f64);
            }
            let (s, e) = (
                row[..MARKETS].iter().sum::<f64>(),
                row[MARKETS..2 * MARKETS].iter().sum::<f64>(),
            );
            let at = map[st as usize] as usize;
            let entries: f64 = (0..PERIODS).map(|q| a.entries(q)[at] as f64).sum();
            let exits: f64 = (0..PERIODS).map(|q| a.exits(q)[at] as f64).sum();
            assert!(
                (s - entries).abs() < 1e-3 * entries.max(1.),
                "station {st}: {s} vs {entries}"
            );
            assert!(
                (e - exits).abs() < 1e-3 * exits.max(1.),
                "station {st}: {e} vs {exits}"
            );
            starts += s;
            ends += e;
        }
        let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        assert!(
            rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail
        );
        assert!(a.station_flows(99).is_empty());
        // T-326: a station of another cluster: no riders, the zones as any station's answer has them
        let (none, row) = (a.no_station_flows(), a.station_flows(0));
        let nz = a.zones.len();
        assert_eq!(none.len(), row.len());
        assert!(none[..ST_W + 2 * nz].iter().all(|&v| v == 0.));
        assert_eq!(none[ST_W + 2 * nz..], row[ST_W + 2 * nz..]);
        // the same after crowding rounds (loads are averaged, not replaced)
        for q in 0..PERIODS {
            let loads = a.seg(q);
            assert!(a.crowd_with_external(q, &loads));
        }
        let rail2: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        let starts2: f64 = (0..5u32)
            .map(|s| a.station_flows(s)[..MARKETS].iter().sum::<f64>())
            .sum();
        assert!((starts2 - rail2).abs() < 1e-3 * rail2.max(1.));
    }
    #[test]
    #[ignore]
    fn intercity_real_pack_profile() {
        let header = std::fs::read_to_string("../data/packs/intercity.json").unwrap();
        let t = std::time::Instant::now();
        let mut a = IntercityApi::new(&header).unwrap();
        eprintln!(
            "intercity info {:?}, pairs {}, open {:?}",
            a.info(),
            a.derived.pairs,
            t.elapsed()
        );
        let x = [0., 0., -67000., -62000., -99000., -89000.];
        let mut times = vec![0.; 18];
        for lev in 0..3 {
            times[lev * 3] = 2400.;
            times[lev * 3 + 1] = 1200.;
            times[(3 + lev) * 3 + 1] = 2400.;
            times[(3 + lev) * 3 + 2] = 1200.;
        }
        a.set_network(&x, &[3], &[0, 1, 2], &times, &[6., 6., 6.], &[8.])
            .unwrap();
        let t = std::time::Instant::now();
        a.solve(0);
        eprintln!(
            "intercity solve {:?}, summary {:?}, corridors {:?}",
            t.elapsed(),
            a.summary(0),
            a.corridor_summary(0)
        );
        assert!(a.summary(0)[2] > 0.);
        assert!(a.summary(0).iter().all(|v| v.is_finite()));
    }
    /// T-174: the market alone on the real New York network plus one Philadelphia line, five
    /// periods, free flow and three crowding rounds on its own loads; every output the app reads.
    /// `TW_IC_DUMP=<file>` writes them, `TW_IC_COMPARE=<file>` compares with such a file.
    /// `cargo test --release --features demand intercity_real_network_outputs -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn intercity_real_network_outputs() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let header = std::fs::read_to_string(format!("{root}/data/packs/intercity.json")).unwrap();
        let t = now_ms();
        let mut a = IntercityApi::new(&header).unwrap();
        eprintln!("open {:.0} ms, info {:?}", now_ms() - t, a.info());
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (mut st_xy, mut line_n, mut stops, mut times, mut tph, mut cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        // one Philadelphia line (as api.rs's northeast_profile), so a second market is served
        const R: f64 = 6371008.8;
        let rad = std::f64::consts::PI / 180.0;
        let wx = R * (-75.1636 + 73.985) * rad * (40.758 * rad).cos();
        let wy = R * (39.9526 - 40.758) * rad;
        let s0 = (st_xy.len() / 2) as u32;
        st_xy.extend([wx as f32, (wy - 2000.0) as f32, wx as f32, (wy + 2000.0) as f32]);
        line_n.push(2);
        stops.extend([s0, s0 + 1]);
        for run in 0..2 {
            for _ in 0..3 {
                times.extend(if run == 0 { [600.0, 0.0] } else { [0.0, 600.0] });
            }
        }
        tph.extend([4.0, 4.0, 2.0]);
        cars.push(8.0);
        let mut hop_km = vec![];
        let mut off = 0usize;
        for &n in &line_n {
            for dir in 0..2 {
                for i in 0..n as usize {
                    let k = if dir == 0 { off + i } else { off + n as usize - 1 - i };
                    let next = if dir == 0 { k + 1 } else { k.wrapping_sub(1) };
                    hop_km.push(if i + 1 < n as usize {
                        let (p, q) = (stops[k] as usize, stops[next] as usize);
                        (st_xy[2 * p] - st_xy[2 * q]).hypot(st_xy[2 * p + 1] - st_xy[2 * q + 1]) / 1000.0
                    } else {
                        0.0
                    });
                }
            }
            off += n as usize;
        }
        let t = now_ms();
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        eprintln!("set_network and fares {:.0} ms; pairs served and all {:?}, zones served {}", now_ms() - t, a.pair_counts(), a.served.len());
        let t = now_ms();
        solve_all(&mut a);
        for _ in 0..3 {
            for q in 0..PERIODS {
                let own = a.seg(q);
                assert!(a.crowd_with_external(q, &own));
            }
        }
        eprintln!("free flow and three rounds {:.0} ms", now_ms() - t);
        let mut out = serde_json::Map::new();
        out.insert("info".into(), serde_json::json!(a.info()[..3]));
        for q in 0..PERIODS {
            let mut s = a.summary(q);
            s[4] = 0.; // ms
            out.insert(format!("summary{q}"), serde_json::json!(s));
            out.insert(format!("corridors{q}"), serde_json::json!(a.corridor_summary(q)));
            out.insert(format!("seg{q}"), serde_json::json!(a.seg(q)));
            out.insert(format!("board{q}"), serde_json::json!(a.board(q)));
            out.insert(format!("alight{q}"), serde_json::json!(a.alight(q)));
            out.insert(format!("entries{q}"), serde_json::json!(a.entries(q)));
            out.insert(format!("exits{q}"), serde_json::json!(a.exits(q)));
            out.insert(format!("crush{q}"), serde_json::json!(a.load_of_crush(q)));
        }
        let ns = st_xy.len() as u32 / 2;
        for st in [0, 1, 10, 100, 300, 500, ns - 3, ns - 1, ns + 5] {
            out.insert(format!("station{st}"), serde_json::json!(a.station_flows(st)));
        }
        // selections: Midtown and Lower Manhattan squares, Philadelphia's line, and the first
        // sites of every market (most of them far from any station)
        let pack: serde_json::Value = serde_json::from_str(&header).unwrap();
        let sites = pack["sites"].as_array().unwrap();
        let mut sels: Vec<(u32, Vec<f32>, Vec<f32>)> = vec![
            (0, vec![0., 0., 1500., -800., -3000., -7000.], vec![1., 0.5, 1.]),
            (1, vec![wx as f32, wy as f32, wx as f32 + 700., wy as f32 + 900.], vec![1., 1.]),
        ];
        for m in 0..a.market_count() {
            let xy: Vec<f32> = sites
                .iter()
                .filter(|s| s[4].as_u64() == Some(m as u64))
                .step_by(97)
                .take(6)
                .flat_map(|s| [s[0].as_f64().unwrap() as f32, s[1].as_f64().unwrap() as f32])
                .collect();
            let n = xy.len() / 2;
            sels.push((m as u32, xy, vec![1.; n]));
        }
        for (k, (m, xy, share)) in sels.iter().enumerate() {
            let t = now_ms();
            let v = a.selection_flows(*m, xy, share);
            eprintln!("selection {k} (market {m}, {} points): {:.1} ms, starts {:.1}", share.len(), now_ms() - t, (0..a.market_count()).map(|p| v[1 + 8 * p]).sum::<f64>());
            out.insert(format!("selection{k}"), serde_json::json!(v));
        }
        let now = serde_json::Value::Object(out);
        if let Ok(path) = std::env::var("TW_IC_DUMP") {
            std::fs::write(&path, serde_json::to_string(&now).unwrap()).unwrap();
            eprintln!("dumped to {path}");
        }
        if let Ok(path) = std::env::var("TW_IC_COMPARE") {
            let base: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let flat = |v: &serde_json::Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
            let (mut worst, mut same, mut keys) = (0f64, 0usize, 0usize);
            for (k, v) in now.as_object().unwrap() {
                let (x, y) = (flat(v), flat(&base[k]));
                assert_eq!(x.len(), y.len(), "{k} length");
                let scale = y.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-12);
                let diff = x.iter().zip(&y).fold(0f64, |m, (p, q)| m.max((p - q).abs()));
                keys += 1;
                if x == y {
                    same += 1;
                } else {
                    eprintln!("{k}: max diff {diff:.3e} of max {scale:.3e}");
                }
                worst = worst.max(diff / scale);
            }
            eprintln!("compared {keys} outputs: {same} identical, worst relative difference {worst:.3e}");
            assert!(worst < 1e-6);
        }
    }
    #[test]
    fn intercity_no_stations_and_invalid_input_are_safe() {
        let mut a = api();
        a.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
        assert!(a.solve(0));
        assert_eq!(a.summary(0)[2], 0.);
        assert!(a.seg(0).is_empty());
        assert!(!a.crowd_with_external(0, &[f32::NAN]));
        assert!(!a.solve(5));
        assert!(IntercityApi::new(r#"{"sites":[[0,0,-1,0,0]]}"#).is_err());
    }
    #[test]
    fn expanded_markets_route_boston_and_washington_and_keep_flow_layouts() {
        let mut a = IntercityApi::new(r#"{"markets":["New York","Philadelphia","Mercer","Boston","Washington DC"],"sites":[[0,0,100000,50000,0],[60000,0,100000,50000,1],[120000,0,100000,50000,2],[180000,0,100000,50000,3],[240000,0,100000,50000,4]]}"#).unwrap();
        assert_eq!(a.market_count(), 5);
        assert_eq!(a.corridors.len(), 15);
        let mut times = vec![0.; 30];
        for lev in 0..3 {
            for k in 0..4 {
                times[lev * 5 + k] = 1200.;
                times[(lev + 3) * 5 + k + 1] = 1200.;
            }
        }
        a.set_network(
            &[0., 0., 60000., 0., 120000., 0., 180000., 0., 240000., 0.],
            &[5],
            &[0, 1, 2, 3, 4],
            &times,
            &[6., 6., 6.],
            &[8.],
        )
        .unwrap();
        solve_all(&mut a);
        let nz = a.zones.len();
        let selected = a.selection_flows(3, &[180000., 0.], &[1.]);
        assert_eq!(selected.len(), 41 + 7 * nz + 20 + 4 * nz);
        assert_eq!(selected[0], 1.);
        assert!(
            selected[1 + 8 * 4 + 1] > 0.,
            "Boston to Washington is served"
        );
        let washington = a.selection_flows(4, &[240000., 0.], &[1.]);
        assert_eq!(washington[0], 1.);
        assert!(
            washington[1 + 8 * 3 + 1] > 0.,
            "Washington to Boston is served"
        );
        let mut starts = 0.;
        let mut ends = 0.;
        for st in 0..5 {
            let row = a.station_flows(st);
            assert_eq!(row.len(), 30 + 5 * nz);
            for d in 0..2 {
                let by_market: f64 = row[d * 5..(d + 1) * 5].iter().sum();
                let by_zone: f64 = row[30 + d * nz..30 + (d + 1) * nz].iter().sum();
                assert!((by_market - by_zone).abs() < 1e-9 * by_market.max(1.));
            }
            starts += row[..5].iter().sum::<f64>();
            ends += row[5..10].iter().sum::<f64>();
        }
        let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        assert!(
            rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail
        );
        assert_eq!(
            api().market_count(),
            3,
            "legacy packs retain their output layout"
        );
    }
    #[test]
    fn compact_pack_accepts_metadata_and_rejects_malformed_sites() {
        let base = api();
        let reordered = IntercityApi::new(r#"{"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]],"description":"unused","origin":{"lon":-73.985,"lat":40.758},"zone_m":6000,"trip_rate":0.015}"#).unwrap();
        assert_eq!(base.info(), reordered.info());
        for (a, b) in base.all_pairs().iter().zip(&reordered.all_pairs()) {
            assert_eq!(
                (a.a, a.b, a.trips, a.drive, a.corridor),
                (b.a, b.b, b.trips, b.drive, b.corridor)
            );
        }
        for bad in [
            r#"{}"#,
            r#"{"sites":[[0,0,1,0]]}"#,
            r#"{"sites":[[0,0,1,0,0,0]]}"#,
            r#"{"sites":[[0,0,null,0,0]]}"#,
            r#"{"sites":[[0,0,"1",0,0]]}"#,
            r#"{"zone_m":0,"sites":[]}"#,
            r#"{"markets":[],"sites":[]}"#,
            r#"{"markets":["A"],"sites":[[0,0,1,0,1]]}"#,
            r#"{"sites":[[0,0,1,0,0.5]]}"#,
        ] {
            assert!(IntercityApi::new(bad).is_err(), "accepted {bad}");
        }
    }
    /// Five markets on a line, a station at each; a zone 10 km from New York's that only a ride
    /// reaches (T-373), and one 300 km out that nothing reaches.
    const FIVE: &str = r#"{"markets":["New York","Philadelphia","Mercer","Boston","Washington DC"],"sites":[[0,0,100000,50000,0],[10000,0,30000,10000,0],[60000,0,100000,50000,1],[120000,0,100000,50000,2],[180000,0,100000,50000,3],[240000,0,100000,50000,4],[300000,50000,50000,20000,4]]}"#;
    fn five_network(a: &mut IntercityApi) {
        let mut times = vec![0.; 30];
        for lev in 0..3 {
            for k in 0..4 {
                times[lev * 5 + k] = 1200.;
                times[(lev + 3) * 5 + k + 1] = 1200.;
            }
        }
        let xy = [0., 0., 60000., 0., 120000., 0., 180000., 0., 240000., 0.];
        a.set_network(&xy, &[5], &[0, 1, 2, 3, 4], &times, &[6., 6., 6.], &[8.]).unwrap();
        a.set_fares(2., 0.1, &[60., 60., 60., 60., 0., 60., 60., 60., 60., 0.]).unwrap();
    }
    /// Everything the app reads of a solved market, bar timings.
    fn outputs(a: &IntercityApi) -> Vec<Vec<f64>> {
        let mut v = vec![a.info()[..3].to_vec()];
        for q in 0..PERIODS {
            let mut s = a.summary(q);
            s[4] = 0.;
            v.push(s);
            v.push(a.corridor_summary(q));
            v.push(a.seg(q).iter().chain(&a.board(q)).chain(&a.entries(q)).chain(&a.exits(q)).map(|&x| x as f64).collect());
        }
        for st in 0..5 {
            v.push(a.station_flows(st));
        }
        v.push(a.selection_flows(0, &[0., 0., 10000., 0.], &[1., 0.5]));
        v.push(a.selection_flows(4, &[300000., 50000.], &[1.]));
        v.push(a.selection_flows(2, &[120000., 0.], &[1.]));
        v
    }
    #[test]
    fn derived_block_from_the_pack_gives_the_same_market() {
        let block = derive_pack(FIVE).unwrap();
        let mut pack: serde_json::Value = serde_json::from_str(FIVE).unwrap();
        pack["derived"] = serde_json::to_value(&block).unwrap();
        let mut with = IntercityApi::new(&pack.to_string()).unwrap();
        let mut without = IntercityApi::new(FIVE).unwrap();
        assert_eq!((with.info()[3], without.info()[3]), (1., 0.));
        assert_eq!(with.derived, without.derived);
        assert!(with.derived.pairs > 0 && with.daily > 0.);
        assert!(with.pairs.is_empty(), "no pairs before a network");
        five_network(&mut with);
        five_network(&mut without);
        // only pairs between served zones, in the whole list's order
        let served: Vec<bool> = (0..with.zones.len()).map(|z| with.served.contains(&z)).collect();
        assert_eq!(served.iter().filter(|s| !**s).count(), 1);
        assert!(served[with.cell_zone[1]], "the zone 10 km out rides to New York's station");
        let want: Vec<_> = with.all_pairs().into_iter().filter(|p| served[p.a] && served[p.b]).map(|p| (p.a, p.b, p.trips)).collect();
        assert_eq!(with.pairs.iter().map(|p| (p.a, p.b, p.trips)).collect::<Vec<_>>(), want);
        assert!(with.pairs.len() < with.derived.pairs);
        for a in [&mut with, &mut without] {
            solve_all(a);
            for q in 0..PERIODS {
                let own = a.seg(q);
                assert!(a.crowd_with_external(q, &own));
            }
        }
        assert_eq!(outputs(&with), outputs(&without));
        // the unserved zones still have their trips (all modes), and no rail
        let far = with.selection_flows(4, &[300000., 50000.], &[1.]);
        let (all, rail): (f64, f64) = ((0..5).map(|m| far[1 + 8 * m]).sum(), (0..5).map(|m| far[2 + 8 * m]).sum());
        assert!(all > 0. && rail == 0., "{all} {rail}");
        // a stale block (another formula or decay, other sites) is derived again at opening
        let stale = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut p = pack.clone();
            edit(&mut p);
            IntercityApi::new(&p.to_string()).unwrap()
        };
        assert_eq!(stale(&|p| p["derived"]["formula"] = "older".into()).info()[3], 0.);
        assert_eq!(stale(&|p| p["derived"]["probe"][1] = 12345.into()).info()[3], 0.);
        assert_eq!(stale(&|p| p["derived"]["totals"] = serde_json::json!([1])).info()[3], 0.);
        assert_eq!(stale(&|p| p["derived"]["totals"] = "unreadable".into()).info()[3], 0.);
        assert_eq!(stale(&|p| p["derived"]["daily"] = (-1.0f64).to_bits().into()).info()[3], 0.);
        let moved = stale(&|p| p["sites"][1][2] = 31000.into());
        assert_eq!(moved.info()[3], 0.);
        assert_ne!(moved.daily, with.daily);
        // pairs follow the served zones as the network changes
        with.set_network(&[0., 0., 60000., 0.], &[2], &[0, 1], &[600., 0., 600., 0., 600., 0., 0., 600., 0., 600., 0., 600.], &[6., 6., 6.], &[8.]).unwrap();
        assert!(with.pairs.iter().all(|p| with.zones[p.a].market <= 1 && with.zones[p.b].market <= 1));
        with.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
        assert!(with.pairs.is_empty());
    }
    /// T-373: rides nest with a site's walks, so a dear ride adds little and a cheap one a lot.
    #[test]
    fn rides_nest_with_walks() {
        let total = |walks: &[(usize, f32)], rides: &[Ride]| {
            let mut m = BTreeMap::new();
            add_access(&mut m, walks, rides, 0.025, 0.2, 1.);
            m.values().sum::<f32>()
        };
        let walks = [(0, 20.), (1, 30.)];
        let alone = total(&walks, &[]);
        assert!((alone - ((-0.5f32).exp() + (-0.75f32).exp()) / 2.).abs() < 1e-6);
        let dear = total(&walks, &[(2, 60., 0., 0.)]);
        assert!(dear > alone && dear < alone * 1.01, "{alone} {dear}");
        let cheap = total(&walks, &[(2, 10., 0., 0.)]);
        assert!(cheap > (-0.25f32).exp(), "{cheap}");
        assert!((total(&[], &[(2, 20., 0., 0.)]) - (-0.5f32).exp()).abs() < 1e-6);
    }
    /// T-128: a pack that names its frame measures zone distances on the ground; one without
    /// keeps its plane (and its derived block); blocks of one kind are stale for the other.
    #[test]
    fn zone_distances_are_on_the_ground_when_the_pack_names_its_frame() {
        use crate::track::frame::great_circle;
        let (lon0, lat0) = (-73.985, 40.758);
        let eq = Frame::equirect(lon0, lat0);
        // New York, Philadelphia, Boston, Washington city halls
        let places = [(-73.985, 40.758), (-75.1636, 39.9526), (-71.0589, 42.3601), (-77.0369, 38.9072)];
        let sites: Vec<String> = places
            .iter()
            .enumerate()
            .map(|(m, &(lon, lat))| {
                let (x, y) = eq.from_lonlat(lon, lat);
                format!("[{x},{y},100000,50000,{m}]")
            })
            .collect();
        let body = format!(r#""markets":["NY","PHL","BOS","WDC"],"sites":[{}]"#, sites.join(","));
        let plane = IntercityApi::new(&format!("{{{body}}}")).unwrap();
        let frame = r#""frame":{"kind":"equirect","lon":-73.985,"lat":40.758}"#;
        let ground = IntercityApi::new(&format!("{{{frame},{body}}}")).unwrap();
        assert!(!plane.ground && ground.ground);
        let km = |a: usize, b: usize| great_circle(places[a].0, places[a].1, places[b].0, places[b].1) / 1000.;
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)] {
            let (zi, zj) = (&ground.zones[i], &ground.zones[j]);
            let d = zone_km(zi, zj, true) as f64;
            assert!((d - km(i, j)).abs() < 1e-5 * km(i, j), "{i}-{j}: {d} vs {}", km(i, j));
            let p = zone_km(&plane.zones[i], &plane.zones[j], false) as f64;
            assert!((p - d).abs() > 1e-4 * d, "{i}-{j}: the plane differs: {p} vs {d}");
        }
        // Boston: an east-west plane mile there is 2.7% short; the pair's drive time follows the ground
        let drive = |a: &IntercityApi| a.all_pairs().iter().find(|p| p.a == 0 && p.b == 2).unwrap().drive;
        assert_eq!(drive(&ground), car_min(zone_km(&ground.zones[0], &ground.zones[2], true), 0.));
        assert!(drive(&ground) != drive(&plane));
        // the same frame in the registry's world-frame kind gives the same ground within the conic's error
        let lcc = Frame::lcc(lon0, lat0);
        let lsites: Vec<String> = places
            .iter()
            .enumerate()
            .map(|(m, &(lon, lat))| {
                let (x, y) = lcc.from_lonlat(lon, lat);
                format!("[{x},{y},100000,50000,{m}]")
            })
            .collect();
        let l = IntercityApi::new(&format!(r#"{{"frame":{{"kind":"lcc","lon":-73.985,"lat":40.758}},"markets":["NY","PHL","BOS","WDC"],"sites":[{}]}}"#, lsites.join(","))).unwrap();
        let (a, b) = (zone_km(&l.zones[0], &l.zones[2], true), zone_km(&ground.zones[0], &ground.zones[2], true));
        assert!((a - b).abs() < 1e-3, "{a} {b}");
        // blocks: each opens its own, and one kind's is stale for the other
        for (text, api) in [(format!("{{{body}}}"), &plane), (format!("{{{frame},{body}}}"), &ground)] {
            let mut pack: serde_json::Value = serde_json::from_str(&text).unwrap();
            pack["derived"] = serde_json::to_value(derive_pack(&text).unwrap()).unwrap();
            let again = IntercityApi::new(&pack.to_string()).unwrap();
            assert_eq!((again.info()[3], again.derived.clone()), (1., api.derived.clone()));
            let other = if api.ground { pack.as_object_mut().unwrap().remove("frame"); pack } else {
                pack["frame"] = serde_json::json!({"kind": "equirect", "lon": lon0, "lat": lat0});
                pack
            };
            assert_eq!(IntercityApi::new(&other.to_string()).unwrap().info()[3], 0.);
        }
        // a moved origin makes the block stale
        let mut pack: serde_json::Value = serde_json::from_str(&format!("{{{frame},{body}}}")).unwrap();
        pack["derived"] = serde_json::to_value(&ground.derived).unwrap();
        pack["frame"]["lat"] = 41.0.into();
        assert_eq!(IntercityApi::new(&pack.to_string()).unwrap().info()[3], 0.);
        for bad in [r#""frame":{"kind":"mercator","lon":0,"lat":0}"#, r#""frame":{"kind":"equirect","lon":0,"lat":95}"#, r#""frame":{"lon":0,"lat":0}"#] {
            assert!(IntercityApi::new(&format!("{{{bad},{body}}}")).is_err(), "accepted {bad}");
        }
    }
}
/// The distance between two zones in km (part of `FORMULA`). On the ground (T-128): the great
/// circle between their centres, from the chord of their unit vectors, the same as
/// `track::frame::great_circle` to rounding. In a pack without a frame: straight in its plane (the
/// New York equirectangular frame, 1.2% short east-west at Philadelphia, 2.7% at Boston).
#[inline(always)]
fn zone_km(a: &Zone, b: &Zone, ground: bool) -> f32 {
    km_between(&a.u, a.x, a.y, &b.u, b.x, b.y, ground)
}
/// As `zone_km` for any two points given by unit vector (ground) or plane position.
#[inline(always)]
fn km_between(ua: &[f64; 3], xa: f32, ya: f32, ub: &[f64; 3], xb: f32, yb: f32, ground: bool) -> f32 {
    if ground {
        let c = ((ua[0] - ub[0]).powi(2) + (ua[1] - ub[1]).powi(2) + (ua[2] - ub[2]).powi(2)).sqrt();
        (2. * EARTH_R / 1000. * (c / 2.).min(1.).asin()) as f32
    } else {
        (xa - xb).hypot(ya - yb) / 1000.
    }
}
/// T-276: the onset, where the market takes over from the local layer's outings (0 at
/// `ONSET_KM`, 1 at twice it; everyday.rs tapers its outings by the complement).
pub const ONSET_KM: f32 = 25.;
/// T-276: the decay's power, fitted to NHTS 2017 with the outside world in place
/// (research/intercity/market_fit.py: Northeast residents' weekday non-commute trips by ground,
/// 50-1,000 km; an exponential length was free in the fit and went to infinity, so there is none).
const DECAY_A: f64 = 2.37;
/// The market's decay over `d` km, or None under the onset (no trips there; the local layer's
/// outings have them). A smooth onset, including long trips within a metro, avoids a
/// discontinuity at city boundaries. Part of `FORMULA`.
#[inline(always)]
fn decay_km(d: f32) -> Option<f64> {
    let onset = ((d - ONSET_KM) / ONSET_KM).clamp(0., 1.) as f64;
    // no trips under the onset: skip the power
    if onset == 0. {
        return None;
    }
    Some(onset / (d as f64).max(ONSET_KM as f64).powf(DECAY_A))
}
/// T-276: zone `z`'s outside-world term of its normaliser: the attraction of every outside point
/// times its decay. Part of `FORMULA`.
fn outside_total(z: &Zone, outside: &[Outside], ground: bool) -> f64 {
    outside.iter().map(|o| decay_km(km_between(&z.u, z.x, z.y, &o.u, o.x, o.y, ground)).map_or(0., |f| o.attr * f)).sum()
}
/// The decay between two zones, the lower index first: the distance in km and the factor.
#[inline(always)]
fn decay(a: &Zone, b: &Zone, ground: bool) -> Option<(f32, f64)> {
    let d = zone_km(a, b, ground);
    decay_km(d).map(|f| (d, f))
}
/// A pair's trips a day each way (`a` < `b`, `f` their decay): the mean of a's departures
/// shared to b and b's to a, each normalised over every zone. Part of `FORMULA`.
#[inline(always)]
fn pair_trips(zones: &[Zone], totals: &[f64], rate: f64, a: usize, b: usize, f: f64) -> f64 {
    let ab = if totals[a] > 0. {
        rate * zones[a].pop * zones[b].attr * f / totals[a]
    } else {
        0.
    };
    let ba = if totals[b] > 0. {
        rate * zones[b].pop * zones[a].attr * f / totals[b]
    } else {
        0.
    };
    (ab + ba) * 0.5
}
/// Pairs with fewer trips a day each way are dropped, both directions together (part of `FORMULA`).
const MIN_TRIPS: f64 = 0.01;
/// Driving minutes over `d` km (part of `FORMULA`).
fn drive_min(d: f32) -> f32 {
    15. + d * ROAD_DETOUR / 85. * 60.
}
/// The road is this times the straight line (part of `FORMULA`).
const ROAD_DETOUR: f32 = 1.25;
/// T-276: what a car costs to run a km, dollars (AAA "Your Driving Costs" 2024: fuel about 8.5
/// cents a km, maintenance and tyres about 6), and the people sharing it on a longer trip (NHTS
/// vehicle occupancy for long and leisure trips is about two).
const CAR_USD_PER_KM: f32 = 0.15;
const LONG_OCCUPANCY: f32 = 2.;
/// T-276: a longer trip's car as perceived minutes over `d` km to a zone where parking costs
/// `park` (the local layer's rule, per car): the drive, the running cost and the parking, both
/// split among the occupants, at the game's value of time (part of `FORMULA`).
fn car_min(d: f32, park: f32) -> f32 {
    drive_min(d) + (d * ROAD_DETOUR * CAR_USD_PER_KM * FARE_MIN_PER_USD + park) / LONG_OCCUPANCY
}
/// Distances (km) `Derived::probe` holds the decay at.
const PROBE_KM: [f32; 4] = [40., 75., 200., 900.];
fn probe() -> Vec<f64> {
    PROBE_KM.iter().map(|&km| decay_km(km).unwrap_or(0.)).collect()
}
/// `Derived::check` of these zones and the outside world; on the ground (T-128) it adds
/// `frame_id`, so a pack whose frame moved is derived again. (Not the zones' unit vectors: their
/// sines and cosines can differ in the last bits between the pipeline and the browser, and the
/// check is compared exactly.)
fn zone_check(zones: &[Zone], outside: &[Outside], frame_id: Option<f64>) -> f64 {
    let plane: f64 = zones.iter().map(|z| z.x as f64 + z.y as f64 + z.pop + z.attr + z.park as f64).sum();
    let out: f64 = outside.iter().map(|o| o.x as f64 + o.y as f64 + o.attr).sum();
    plane + out + frame_id.unwrap_or(0.)
}
/// The decay from zone `a` to every later zone with any trips, as (zone, km, factor), into `out`.
/// One row per call, out of line (T-172): a browser optimises a WebAssembly function only between
/// calls, so a single loop over every zone pair would run unoptimised throughout.
#[inline(never)]
fn decay_row(zones: &[Zone], ground: bool, a: usize, out: &mut Vec<(usize, f32, f64)>) {
    out.clear();
    for b in a + 1..zones.len() {
        if let Some((d, f)) = decay(&zones[a], &zones[b], ground) {
            out.push((b, d, f));
        }
    }
}
/// The figures that need every zone pair (T-174): two passes over all of them, O(zones^2). The
/// pipeline runs this once per pack (`bin/market_pack.rs`); the game only when the pack's block
/// is missing or stale.
fn derive(zones: &[Zone], outside: &[Outside], frame_id: Option<f64>, rate: f64, corridor_of: &[usize], markets: usize) -> Derived {
    let ground = frame_id.is_some();
    // Store only row normalisers, not a full directed matrix or a reverse-pair map.
    let mut totals = vec![0.; zones.len()];
    let mut row = vec![];
    for a in 0..zones.len() {
        decay_row(zones, ground, a, &mut row);
        for &(b, _, f) in &row {
            totals[a] += zones[b].attr * f;
            totals[b] += zones[a].attr * f;
        }
    }
    // T-276: the outside world's part of each normaliser, and the departures it takes
    let mut leaving = 0.;
    for (a, z) in zones.iter().enumerate() {
        let out = outside_total(z, outside, ground);
        totals[a] += out;
        if totals[a] > 0. {
            leaving += rate * z.pop * out / totals[a];
        }
    }
    // Sums in the order the whole pair list has (each pair, then its reverse), so they are the
    // same to the last bit as summing that list.
    let (mut daily, mut pairs) = (0f64, 0usize);
    let mut corridor_trips = vec![0f64; markets * (markets + 1) / 2];
    for a in 0..zones.len() {
        decay_row(zones, ground, a, &mut row);
        for &(b, _, f) in &row {
            let trips = pair_trips(zones, &totals, rate, a, b, f);
            // Prune both directions together after averaging, including pairs whose
            // original flow was above the threshold in only one direction.
            if trips < MIN_TRIPS {
                continue;
            }
            let c = corridor_of[zones[a].market.min(zones[b].market) * markets + zones[a].market.max(zones[b].market)];
            daily += trips;
            daily += trips;
            corridor_trips[c] += trips;
            corridor_trips[c] += trips;
            pairs += 2;
        }
    }
    Derived {
        formula: formula(ground).into(),
        probe: probe(),
        check: zone_check(zones, outside, frame_id),
        totals,
        daily,
        corridor_trips,
        pairs,
        leaving,
    }
}
/// The pairs from zone `a` to each of `later` (higher zones, ascending), both directions, onto
/// `out`. Out of line like `decay_row`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn pair_row(zones: &[Zone], ground: bool, totals: &[f64], rate: f64, corridor_of: &[usize], markets: usize, a: usize, later: &[usize], out: &mut Vec<Pair>) {
    for &b in later {
        let Some((d, f)) = decay(&zones[a], &zones[b], ground) else { continue };
        let trips = pair_trips(zones, totals, rate, a, b, f);
        if trips < MIN_TRIPS {
            continue;
        }
        let corridor = corridor_of[zones[a].market.min(zones[b].market) * markets + zones[a].market.max(zones[b].market)];
        // T-276: each direction's car parks at its own end
        out.push(Pair { a, b, trips, drive: car_min(d, zones[b].park), corridor });
        out.push(Pair { a: b, b: a, trips, drive: car_min(d, zones[a].park), corridor });
    }
}
/// Zone `z`'s pairs with every other zone, partner ascending, as (partner, trips each way); the
/// order the whole market's pairs of one zone have. Out of line like `decay_row`.
#[inline(never)]
fn zone_row(zones: &[Zone], ground: bool, totals: &[f64], rate: f64, z: usize) -> Vec<(usize, f64)> {
    let mut out = vec![];
    for b in 0..zones.len() {
        if b == z {
            continue;
        }
        let (lo, hi) = (z.min(b), z.max(b));
        let Some((_, f)) = decay(&zones[lo], &zones[hi], ground) else { continue };
        let trips = pair_trips(zones, totals, rate, lo, hi, f);
        if trips >= MIN_TRIPS {
            out.push((b, trips));
        }
    }
    out
}
/// What the market's sites make: the local city (one cell per site), the zones, each site's zone.
struct Opened {
    city: City,
    zones: Vec<Zone>,
    cell_zone: Vec<usize>,
    markets: usize,
    rate: f64,
    derived: Option<Derived>,
    corridors: Vec<(usize, usize)>,
    corridor_of: Vec<usize>,
    /// T-128: with the pack's frame, its kind (0 equirect, 1 lcc) plus its origin's degrees
    /// (`zone_check`); None for a pack without one, whose distances are in its plane
    frame_id: Option<f64>,
    /// T-276: the outside world, and the commutes between zones by zone
    outside: Vec<Outside>,
    commutes: Vec<(u32, u32, f32)>,
}
fn open(header: &str) -> Result<Opened, String> {
    let pack: MarketPack = serde_json::from_str(header).map_err(|e| e.to_string())?;
    let zm = pack.zone_m as f32;
    let rate = pack.trip_rate;
    let markets = pack.markets.len();
    if markets == 0 || !zm.is_finite() || zm <= 0. || !rate.is_finite() || rate < 0. {
        return Err("invalid market parameters".into());
    }
    let frame = match &pack.frame {
        None => None,
        Some(f) if !f.lon.is_finite() || !f.lat.is_finite() || f.lat.abs() >= 90. => return Err("invalid market frame origin".into()),
        Some(f) => match f.kind.as_str() {
            "equirect" => Some(Frame::equirect(f.lon, f.lat)),
            "lcc" => Some(Frame::lcc(f.lon, f.lat)),
            k => return Err(format!("unknown market frame {k:?}")),
        },
    };
    let mut city = City {
        name: "Northeast intercity".into(),
        h3: vec![],
        x: vec![],
        y: vec![],
        pop: vec![],
        jobs: vec![],
        region: vec![], gravity: None,
    };
    let mut keys = BTreeMap::new();
    let mut zones: Vec<Zone> = vec![];
    let mut cell_zone = vec![];
    if !pack.site_park.is_empty() && (pack.site_park.len() != pack.sites.len() || pack.site_park.iter().any(|p| !p.is_finite() || *p < 0.)) {
        return Err("invalid site parking".into());
    }
    // T-276: each zone's parking, weighted by its sites' jobs (the jobs-weight summed alongside)
    let mut park_w: Vec<f64> = vec![];
    for (si, nums) in pack.sites.into_iter().enumerate() {
        if nums.iter().any(|x| !x.is_finite())
            || nums[2] < 0.
            || nums[3] < 0.
            || nums[4] < 0.
            || nums[4].fract() != 0.
            || nums[4] >= markets as f64
        {
            return Err("invalid intercity weights".into());
        }
        let (x, y, pop, jobs, m) = (
            nums[0] as f32,
            nums[1] as f32,
            nums[2],
            nums[3],
            nums[4] as usize,
        );
        let key = ((x / zm).floor() as i32, (y / zm).floor() as i32, m);
        let z = *keys.entry(key).or_insert_with(|| {
            zones.push(Zone {
                x: 0.,
                y: 0.,
                pop: 0.,
                attr: 0.,
                market: m,
                u: [0.; 3],
                park: 0.,
            });
            park_w.push(0.);
            zones.len() - 1
        });
        let attr = pop + 2. * jobs;
        let w = pop + attr;
        zones[z].x += x * w as f32;
        zones[z].y += y * w as f32;
        zones[z].pop += pop;
        zones[z].attr += attr;
        if let Some(&p) = pack.site_park.get(si) {
            // weight 1e-9 for a site without jobs, so a zone of homes only still gets its parking
            let jw = jobs + 1e-9;
            zones[z].park += (p as f64 * jw) as f32;
            park_w[z] += jw;
        }
        city.x.push(x);
        city.y.push(y);
        city.pop.push(pop as f32);
        city.jobs.push(attr as f32);
        cell_zone.push(z);
    }
    let sphere = |f: &Frame, x: f32, y: f32| {
        let (lon, lat) = f.to_lonlat(x as f64, y as f64);
        let (lon, lat) = (lon.to_radians(), lat.to_radians());
        [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
    };
    for (zi, z) in zones.iter_mut().enumerate() {
        let w = (z.pop + z.attr).max(1e-12) as f32;
        z.x /= w;
        z.y /= w;
        if park_w[zi] > 0. {
            z.park = (z.park as f64 / park_w[zi]) as f32;
        }
        // T-128: the centre on the ground. An equirectangular frame is affine in lon/lat, so
        // the plane's weighted centre is the weighted centre of the sites' lon/lat.
        if let Some(f) = &frame {
            z.u = sphere(f, z.x, z.y);
        }
    }
    // T-276: the commutes between zones, by zone (a pair named by two sites of the same zones
    // twice is one pair)
    let nc = pack.commute_trips.len();
    if pack.commute_home.len() != nc || pack.commute_work.len() != nc {
        return Err("commute arrays disagree".into());
    }
    let mut by_pair: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for i in 0..nc {
        let (h, w, n) = (pack.commute_home[i] as usize, pack.commute_work[i] as usize, pack.commute_trips[i]);
        if h >= cell_zone.len() || w >= cell_zone.len() || !n.is_finite() || n < 0. {
            return Err("invalid commute between zones".into());
        }
        let (zh, zw) = (cell_zone[h] as u32, cell_zone[w] as u32);
        if zh != zw {
            *by_pair.entry((zh, zw)).or_default() += n as f64;
        }
    }
    let commutes: Vec<(u32, u32, f32)> = by_pair.into_iter().map(|((h, w), n)| (h, w, n as f32)).collect();
    let mut outside = Vec::with_capacity(pack.outside.len());
    for o in &pack.outside {
        if o.iter().any(|v| !v.is_finite()) || o[2] < 0. {
            return Err("invalid outside world".into());
        }
        let (x, y) = (o[0] as f32, o[1] as f32);
        outside.push(Outside { x, y, u: frame.as_ref().map_or([0.; 3], |f| sphere(f, x, y)), attr: o[2] });
    }
    let corridors: Vec<_> = (0..markets)
        .flat_map(|a| (a..markets).map(move |b| (a, b)))
        .collect();
    let mut corridor_of = vec![0; markets * markets];
    for (i, &(a, b)) in corridors.iter().enumerate() {
        corridor_of[a * markets + b] = i;
    }
    let derived = pack.derived.and_then(|v| serde_json::from_value(v).ok());
    let frame_id = frame.map(|f| f.kind as u8 as f64 + f.lon0 + f.lat0);
    Ok(Opened { city, zones, cell_zone, markets, rate, derived, corridors, corridor_of, frame_id, outside, commutes })
}
/// T-276: the market's zones as the game builds them from a pack's sites, for the joint commute
/// gravity (`bin/joint_gravity.rs`): each site's zone, the first site of each zone, and each
/// site's 600 m square key (x, y in the sites' frame / 600, floored; the market).
pub fn market_site_zones(header: &str) -> Result<(Vec<u32>, Vec<u32>, Vec<(i32, i32, usize)>), String> {
    let o = open(header)?;
    let mut first = vec![u32::MAX; o.zones.len()];
    for (s, &z) in o.cell_zone.iter().enumerate() {
        if first[z] == u32::MAX {
            first[z] = s as u32;
        }
    }
    let keys = (0..o.city.len())
        .map(|c| ((o.city.x[c] / SITE_M).floor() as i32, (o.city.y[c] / SITE_M).floor() as i32, o.zones[o.cell_zone[c]].market))
        .collect();
    Ok((o.cell_zone.iter().map(|&z| z as u32).collect(), first, keys))
}
/// T-373: the 2 km zones of the market's sites (`city`, whose `jobs` are attractions) with their
/// homes and jobs as the local layer's zones have them, so a ride's drive and parking are the
/// local layer's; and each zone's homes-and-jobs-weighted centre, where its rides start.
fn ride_zones(city: &City) -> (Zones, Vec<(usize, f32, f32)>) {
    let jobs: Vec<f32> = city.pop.iter().zip(&city.jobs).map(|(&p, &a)| ((a - p) / 2.).max(0.)).collect();
    let sites = City { name: String::new(), h3: vec![], x: city.x.clone(), y: city.y.clone(), pop: city.pop.clone(), jobs, region: vec![], gravity: None };
    if sites.is_empty() {
        return (build_zones(&sites, 2000.), vec![]);
    }
    let mut z = build_zones(&sites, 2000.);
    // a site is a 600 m square, not an H3 cell: its land in cells for `Zones::density`
    let per_site = (SITE_M / 1000.).powi(2) / CELL_KM2;
    let mut c = vec![(0f64, 0f64, 0f64); z.n];
    for s in 0..sites.len() {
        let zi = z.cell_zone[s] as usize;
        let w = (sites.pop[s] + sites.jobs[s]) as f64 + 1e-9;
        c[zi].0 += w;
        c[zi].1 += w * sites.x[s] as f64;
        c[zi].2 += w * sites.y[s] as f64;
    }
    let mut places = Vec::with_capacity(z.n);
    for (zi, &(w, x, y)) in c.iter().enumerate() {
        z.ncell[zi] = ((z.ncell[zi] as f32 * per_site).round() as u32).max(1);
        places.push((zi, (x / w) as f32, (y / w) as f32));
    }
    (z, places)
}
/// T-373: a site's access weights at logit scale `beta`, by station, added into `out` times
/// `share`: its walks (station, perceived minutes) as the mean of their exp(-beta x cost) as
/// before rides, nested with its rides at the station choice's scale `theta`, so a ride dearer
/// than the walk adds little; without rides, the walks' weights alone.
fn add_access(out: &mut BTreeMap<usize, f32>, walks: &[(usize, f32)], rides: &[Ride], beta: f32, theta: f32, share: f32) {
    let (beta, theta) = (beta as f64, theta as f64);
    let n = walks.len() as f64;
    let mean: f64 = walks.iter().map(|w| (-beta * w.1 as f64).exp() / n).sum();
    if rides.is_empty() {
        for &(st, c) in walks {
            *out.entry(st).or_default() += share * ((-beta * c as f64).exp() / n) as f32;
        }
        return;
    }
    let cw = if walks.is_empty() { f64::INFINITY } else { -mean.ln() / beta };
    let c0 = rides.iter().map(|r| r.1 as f64).fold(cw, f64::min);
    let aw = if walks.is_empty() { 0. } else { (-theta * (cw - c0)).exp() };
    let sum = aw + rides.iter().map(|r| (-theta * (r.1 as f64 - c0)).exp()).sum::<f64>();
    let total = (-beta * (c0 - sum.ln() / theta)).exp() / sum;
    for &(st, c) in walks {
        *out.entry(st).or_default() += share * (total * aw * (-beta * c as f64).exp() / n / mean) as f32;
    }
    for r in rides {
        *out.entry(r.0 as usize).or_default() += share * (total * (-theta * (r.1 as f64 - c0)).exp()) as f32;
    }
}
/// The pack's `derived` block, computed from its sites (T-174; `bin/market_pack.rs`).
pub fn derive_pack(header: &str) -> Result<Derived, String> {
    let o = open(header)?;
    Ok(derive(&o.zones, &o.outside, o.frame_id, o.rate, &o.corridor_of, o.markets))
}

#[wasm_bindgen]
impl IntercityApi {
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str) -> Result<IntercityApi, String> {
        let Opened { city, zones, cell_zone, markets, rate, derived, corridors, corridor_of, frame_id, outside, commutes } = open(header)?;
        let ground = frame_id.is_some();
        // T-174: the pack's all-pairs figures if they are this formula's for these zones;
        // otherwise derive them here (two passes over every zone pair, the same result)
        // The probe is compared to 1e-9: the pipeline derives natively, and the browser's exp
        // and powf (Rust's libm in WebAssembly) can differ from the platform's in the last bits.
        let current = derived.filter(|d| {
            d.formula == formula(ground)
                && d.probe.len() == PROBE_KM.len()
                && d.probe.iter().zip(probe()).all(|(a, b)| (a - b).abs() <= 1e-9 * b.abs())
                && d.check == zone_check(&zones, &outside, frame_id)
                && d.totals.len() == zones.len()
                && d.corridor_trips.len() == corridors.len()
                && d.totals.iter().chain(&d.corridor_trips).all(|v| v.is_finite() && *v >= 0.)
                && d.daily.is_finite()
                && d.daily >= 0.
        });
        let from_pack = current.is_some();
        let derived = current.unwrap_or_else(|| derive(&zones, &outside, frame_id, rate, &corridor_of, markets));
        let mut site_key = HashMap::new();
        for c in 0..city.len() {
            let key = (
                (city.x[c] / SITE_M).floor() as i32,
                (city.y[c] / SITE_M).floor() as i32,
                zones[cell_zone[c]].market,
            );
            site_key.entry(key).or_insert(c);
        }
        let base = Params {
            beta: 0.025,
            asc_rail: 0.,
            fare_min: 0.,
            ..Params::default()
        };
        // T-276: each zone's commutes from its homes and to its jobs, and each corridor's legs
        let csr = |end: fn(&(u32, u32, f32)) -> u32| {
            let mut off = vec![0u32; zones.len() + 1];
            for c in &commutes {
                off[end(c) as usize + 1] += 1;
            }
            for z in 0..zones.len() {
                off[z + 1] += off[z];
            }
            let mut fill = off.clone();
            let mut items = vec![0u32; commutes.len()];
            for (i, c) in commutes.iter().enumerate() {
                let z = end(c) as usize;
                items[fill[z] as usize] = i as u32;
                fill[z] += 1;
            }
            (off, items)
        };
        let (c_home, c_work) = (csr(|c| c.0), csr(|c| c.1));
        let (kz, kplace) = ride_zones(&city);
        let mut c_corridor = vec![0f64; corridors.len()];
        for &(h, w, n) in &commutes {
            let (a, b) = (zones[h as usize].market, zones[w as usize].market);
            c_corridor[corridor_of[a.min(b) * markets + a.max(b)]] += 2. * n as f64;
        }
        Ok(Self {
            city,
            zones,
            cell_zone,
            access_c: vec![],
            access_cw: vec![],
            access_c_walk: vec![],
            kz,
            kplace,
            commutes,
            c_home,
            c_work,
            c_corridor,
            cpairs: vec![],
            cpair_of: vec![],
            pairs: vec![],
            served: vec![],
            pair_index: HashMap::new(),
            daily: derived.daily,
            derived,
            from_pack,
            ground,
            rate,
            corridor_of,
            base,
            svc: None,
            access: vec![],
            access_work: vec![],
            want: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            fare_km: vec![],
            fare_per_km: 0.,
            corridors,
            markets,
            site_key,
            complex_in: vec![],
        })
    }
    /// The station complexes of the networks `set_network` is given from now on (T-161), as
    /// `DemandApi::set_complexes`.
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.complex_in = complex.to_vec();
    }
    /// Number of market slots in selection and station flow headers.
    pub fn market_count(&self) -> usize {
        self.markets
    }
    /// Sites, zones, the day's trips, 1 if the all-pairs figures came from the pack (T-174;
    /// 0: derived at opening, the pack's block missing or stale), and the day's departures that
    /// leave the map (T-276).
    pub fn info(&self) -> Vec<f64> {
        vec![self.city.len() as f64, self.zones.len() as f64, self.daily, self.from_pack as u8 as f64, self.derived.leaving]
    }
    /// The zone pairs the network's stations serve (T-174), and the market's whole count.
    pub fn pair_counts(&self) -> Vec<f64> {
        vec![self.pairs.len() as f64, self.derived.pairs as f64]
    }
    pub fn set_network(
        &mut self,
        st_xy: &[f32],
        line_n: &[u32],
        stops: &[u32],
        times: &[f32],
        tph: &[f32],
        cars: &[f32],
    ) -> Result<f64, String> {
        let t = now_ms();
        let mut off = vec![0];
        for &n in line_n {
            off.push(off.last().unwrap() + n)
        }
        if st_xy.len() % 2 != 0
            || stops.len() != *off.last().unwrap() as usize
            || times.len() != 6 * stops.len()
            || tph.len() != 3 * cars.len()
            || cars.len() != line_n.len()
            || stops.iter().any(|&s| s as usize >= st_xy.len() / 2)
        {
            return Err("intercity network arrays disagree".into());
        }
        let (st_x, st_y, st_map) = merge_stations(st_xy, &off, stops, self.base.station_merge_m);
        let group = demand_groups(&st_map, st_x.len(), &self.complex_in);
        let svc = Service {
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off: off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect(),
            cars: cars.to_vec(),
            group,
        };
        // T-124: as `DemandApi::set_network`, access is kept with the layout and a period with
        // its level's inputs
        let old = self.svc.as_ref();
        let same = old.map_or(0, |o| svc.same_periods(o));
        if !old.is_some_and(|o| svc.same_layout(o)) {
            self.set_access(&svc);
        }
        for (q, p) in self.per.iter_mut().enumerate() {
            match p {
                Some(st) if same >> q & 1 == 1 => st.net = DemandApi::period_net(&svc, q),
                _ => *p = None,
            }
        }
        self.svc = Some(svc);
        Ok(now_ms() - t)
    }
    /// Station access by zone and the zone pairs a station serves, for `svc`'s layout.
    fn set_access(&mut self, svc: &Service) {
        let net = DemandApi::period_net(svc, 0);
        let acc = build_access(&self.city, &net, &self.base);
        let mut weights: Vec<BTreeMap<usize, f32>> =
            (0..self.zones.len()).map(|_| BTreeMap::new()).collect();
        let mut work = weights.clone();
        // T-276: the commutes' access at their own logit scale; the work end by jobs alone
        let commuting = !self.commutes.is_empty();
        let (mut cw_home, mut cw_work, mut cw_walk) = (weights.clone(), weights.clone(), weights.clone());
        let beta_c = commute_params().beta;
        // T-373: rides from each site's 2 km zone, parking or a drop-off at the home end of both
        // parts, a drop-off at a longer trip's far end; a commute's work end walks
        let (h_off, h_rides) = access_options(&self.kz, &net, &self.base, &self.kplace, End::Home);
        let (f_off, f_rides) = access_options(&self.kz, &net, &self.base, &self.kplace, End::Far);
        let theta = self.base.station_theta;
        let mut walks: Vec<(usize, f32)> = Vec::with_capacity(K);
        for c in 0..self.city.len() {
            let z = self.cell_zone[c];
            let w = self.city.pop[c] / self.zones[z].pop.max(1e-12) as f32;
            let jw = self.city.jobs[c] / self.zones[z].attr.max(1e-12) as f32;
            let zjobs = ((self.zones[z].attr - self.zones[z].pop) / 2.).max(1e-12);
            let cjw = ((self.city.jobs[c] - self.city.pop[c]) / 2.).max(0.) / zjobs as f32;
            let kz = self.kz.cell_zone[c] as usize;
            let home = &h_rides[h_off[kz] as usize..h_off[kz + 1] as usize];
            let far = &f_rides[f_off[kz] as usize..f_off[kz + 1] as usize];
            walks.clear();
            for i in c * K..c * K + acc.n[c] as usize {
                let walk = acc.walk[i];
                let cost = walk.min(self.base.walk_easy_min) * self.base.w_walk + (walk - self.base.walk_easy_min).max(0.) * self.base.w_walk_far;
                walks.push((acc.st[i] as usize, cost));
            }
            add_access(&mut weights[z], &walks, home, self.base.beta, theta, w);
            add_access(&mut work[z], &walks, far, self.base.beta, theta, jw);
            if commuting {
                add_access(&mut cw_home[z], &walks, home, beta_c, theta, w);
                add_access(&mut cw_walk[z], &walks, &[], beta_c, theta, w);
                add_access(&mut cw_work[z], &walks, &[], beta_c, theta, cjw);
            }
        }
        let top8 = |maps: Vec<BTreeMap<usize, f32>>, positive: bool| -> Vec<Vec<(usize, f32)>> {
            maps.into_iter()
                .map(|m| {
                    let mut v: Vec<_> = m.into_iter().filter(|e| !positive || e.1 > 0.).collect();
                    v.sort_by(|a, b| b.1.total_cmp(&a.1));
                    v.truncate(8);
                    v
                })
                .collect()
        };
        self.access = top8(weights, false);
        self.access_work = top8(work, false);
        self.access_c = if commuting { top8(cw_home, true) } else { vec![] };
        self.access_cw = if commuting { top8(cw_work, true) } else { vec![] };
        self.access_c_walk = if commuting { top8(cw_walk, true) } else { vec![] };
        self.want = vec![false; svc.st_x.len()];
        // a commute's trip home ends at its home station
        for list in self.access_work.iter().chain(&self.access_cw).chain(&self.access_c).chain(&self.access_c_walk) {
            for &(st, _) in list {
                self.want[st] = true;
            }
        }
        // T-174: only pairs between zones a station serves can carry a rail trip (any other has
        // no access at one end and gets none), so only those are built, when that set changes
        let served: Vec<usize> = (0..self.zones.len())
            .filter(|&z| !self.access[z].is_empty() || !self.access_work[z].is_empty())
            .collect();
        if served != self.served {
            self.set_served(served);
        }
    }
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, km: &[f32]) -> Result<(), String> {
        if !base.is_finite()
            || base < 0.
            || !per_km.is_finite()
            || per_km < 0.
            || km.iter().any(|x| !x.is_finite() || *x < 0.)
            || km.len() != self.svc.as_ref().map_or(0, |s| 2 * s.stops.len())
        {
            return Err("invalid intercity fare".into());
        }
        let (fare_min, per_km) = (base * FARE_MIN_PER_USD, per_km * FARE_MIN_PER_USD);
        if fare_min.to_bits() != self.base.fare_min.to_bits() || per_km.to_bits() != self.fare_per_km.to_bits() || !same_bits(km, &self.fare_km) {
            self.base.fare_min = fare_min;
            self.fare_per_km = per_km;
            self.fare_km = km.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    pub fn solve(&mut self, q: usize) -> bool {
        let Some(svc) = &self.svc else { return false };
        if q >= PERIODS {
            return false;
        }
        let t = now_ms();
        let p = Params {
            period: q,
            period_share: SHARES[q],
            period_hours: (PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.base.clone()
        };
        let net = DemandApi::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        for (r, f) in g.rn_fare.iter_mut().enumerate() {
            *f = self.fare_km.get(r).copied().unwrap_or(0.) * self.fare_per_km;
        }
        // The fare enters the graph as it does for the local populations (T-121, T-172): before,
        // this estimate routed on ride time alone, so its graph differed from theirs and it ran a
        // strategy search of its own, the larger part of a big network's first estimate.
        let no_crowding = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &no_crowding);
        let xi = xfer_index(&g);
        let pass = self.pass(&g, &p, q, &xi);
        let commuters: f64 = self.commutes.iter().map(|c| c.2 as f64).sum();
        self.per[q] = Some(Period {
            net,
            p,
            g,
            avg: pass.loads,
            c_avg: pass.c_loads,
            rail: pass.rail,
            trips: self.daily * SHARES[q] as f64,
            c_rail: pass.c_rail,
            c_trips: commuters * (TO_WORK[q] + TO_HOME[q]) as f64,
            c_frac: pass.c_frac,
            corridors: pass.cor,
            c_corridors: pass.c_cor,
            pair_rail: pass.pair_rail,
            st: pass.st,
            round: 0,
            ms: now_ms() - t,
            fac: no_crowding,
            // a crowding round with no crowding only repeats this estimate
            uniform: true,
            xi,
            xfer: pass.xfer,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per(q) else { return false };
        // the crowding and the waits for full trains (T-188) from every population's loads
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            return false;
        };
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        if st.uniform && fac == st.fac && left == st.g.rn_left && !legacy() {
            // as `DemandApi::crowd_with_external`: nothing is crowded, the round changes nothing
            st.round += 1;
            st.ms = now_ms() - t;
            self.per[q] = Some(st);
            return true;
        }
        set_crowding(&mut st.g, &fac);
        st.g.rn_left = left;
        st.uniform = false;
        st.fac = fac;
        let Pass {
            loads: l,
            c_loads,
            rail,
            c_rail,
            c_frac,
            cor,
            c_cor,
            pair_rail,
            st: st_new,
            xfer,
        } = self.pass(&st.g, &st.p, q, &st.xi);
        st.round += 1;
        let r = st.round;
        msa(&mut st.xfer, &xfer, r);
        msa(&mut st.pair_rail, &pair_rail, r);
        msa(&mut st.c_frac, &c_frac, r);
        msa(&mut st.c_avg.seg, &c_loads.seg, r);
        msa(&mut st.c_avg.rn_board, &c_loads.rn_board, r);
        msa(&mut st.c_avg.rn_alight, &c_loads.rn_alight, r);
        st.c_rail += (c_rail - st.c_rail) / (r as f64 + 1.);
        for (a, b) in st.st.iter_mut().zip(st_new) {
            *a += (b - *a) / (r as f32 + 1.);
        }
        msa(&mut st.avg.seg, &l.seg, r);
        msa(&mut st.avg.board, &l.board, r);
        msa(&mut st.avg.alight, &l.alight, r);
        msa(&mut st.avg.rn_board, &l.rn_board, r);
        msa(&mut st.avg.rn_alight, &l.rn_alight, r);
        msa(&mut st.avg.entry, &l.entry, r);
        msa(&mut st.avg.exit, &l.exit, r);
        let w = 1. / (r as f64 + 1.);
        st.rail += (rail - st.rail) * w;
        for (a, b) in st.corridors.iter_mut().zip(cor) {
            *a += (b - *a) * w;
        }
        for (a, b) in st.c_corridors.iter_mut().zip(c_cor) {
            *a += (b - *a) * w;
        }
        st.ms = now_ms() - t;
        self.per[q] = Some(st);
        true
    }
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let (_, c) = crowd_factors(&s.g, &s.net, &s.avg, &s.p);
            vec![
                s.round as f64,
                s.trips + s.c_trips,
                s.rail + s.c_rail,
                0.,
                s.ms,
                self.derived.pairs as f64,
                c.segs_over_seats as f64,
                c.max_load_of_crush as f64,
                c.p50 as f64,
                c.p90 as f64,
            ]
        })
    }
    /// The regional layer's parts in period `q` (T-276): trips and rail trips of the longer trips,
    /// then legs and rail legs of the commutes between zones.
    pub fn part_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| vec![s.trips, s.rail, s.c_trips, s.c_rail])
    }
    /// T-276: the commutes' segment loads, boardings and alightings per route node in period `q`
    /// (part of `seg`, `board`, `alight`, which hold both parts); empty with no commutes.
    pub fn commute_loads(&self, q: usize) -> Vec<f32> {
        self.per(q).filter(|_| !self.commutes.is_empty()).map_or(vec![], |s| [&s.c_avg.seg[..], &s.c_avg.rn_board[..], &s.c_avg.rn_alight[..]].concat())
    }
    /// T-276: commute pairs between zones some station serves, and their commuters a day.
    pub fn served_commutes(&self) -> Vec<f64> {
        vec![self.cpairs.len() as f64, self.cpairs.iter().map(|c| self.commutes[c.c as usize].2 as f64).sum()]
    }
    /// T-276: commuters a day between zones, all of them (not only those a station serves).
    pub fn commuters(&self) -> f64 {
        self.commutes.iter().map(|c| c.2 as f64).sum()
    }
    /// Per corridor (market pair): its two places, longer trips in period `q` and their rail trips.
    pub fn corridor_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            self.corridors
                .iter()
                .enumerate()
                .flat_map(|(i, &(a, b))| {
                    [
                        a as f64,
                        b as f64,
                        self.derived.corridor_trips[i] * SHARES[q] as f64,
                        s.corridors[i],
                    ]
                })
                .collect()
        })
    }
    /// T-276: per corridor, in `corridor_summary`'s order, the commutes between zones in period
    /// `q`: legs (both directions) and rail legs. Empty with no commutes.
    pub fn commute_corridors(&self, q: usize) -> Vec<f64> {
        let legs = 0.5 * (TO_WORK[q] + TO_HOME[q]) as f64;
        self.per(q).filter(|_| !self.commutes.is_empty()).map_or(vec![], |s| {
            self.c_corridor.iter().zip(&s.c_corridors).flat_map(|(&c, &r)| [c * legs, r]).collect()
        })
    }
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.seg.clone())
    }
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_board.clone())
    }
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_alight.clone())
    }
    /// As `DemandApi::transfers` (T-176): this market's transfers over the periods held.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    /// `transfers` on a kind of day (T-260): each period's riders times the market's load factor.
    pub fn transfers_day(&self, sel: &[u8], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (q, st) in self.per.iter().enumerate() {
            let Some(st) = st else { continue };
            let at = out.len();
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
            scale_triples(&mut out[at..], load_factor(day as usize, LONGER, q));
        }
        out
    }
    pub fn entries(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.entry.clone())
    }
    pub fn exits(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.exit.clone())
    }
    /// Bit q set for each period solved.
    pub fn solved_mask(&self) -> u32 {
        (0..PERIODS)
            .filter(|&q| self.per(q).is_some())
            .fold(0, |m, q| m | 1 << q)
    }
    /// Where the longer trips of a selection of local cells go and come from (T-118).
    ///
    /// The selection is points (`xy`: x, y per 600 m square, metres in the New York frame) of one
    /// `market` (the index in the pack's market list) with the `share` (0..1) of that square's people, or
    /// jobs, that are selected. Each point is the endpoint site of the same square (the pipeline
    /// built the sites from the same cells), and a site's share of its zone's departures is its
    /// population over the zone's, of its arrivals its attraction over the zone's. So a zone's
    /// trips are split to the sites by the same weights the market used to give them out, and a
    /// selection that is part of a square counts that part of it.
    ///
    /// Returns, over the periods solved (`solved_mask`): the matched share of the points, then
    /// 8 numbers for each of the markets at the far end: trips starting in the selection that
    /// end there (all modes, by rail, and the all-mode trips times far-zone x and y), then the
    /// same for trips ending in the selection that start there. Trips are a day's share for the
    /// periods solved, so rail is never above all.
    pub fn selection_flows(&self, market: u32, xy: &[f32], share: &[f32]) -> Vec<f64> {
        self.selection_flows_day(market, xy, share, WEEKDAY as u32)
    }
    /// `selection_flows` on a kind of day (T-260, `days.rs`): that day's trips and period shares.
    pub fn selection_flows_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        let day = day as usize;
        let nz = self.zones.len();
        let head = 1 + 8 * self.markets;
        // T-276: then the commutes between zones, 4 numbers per market and 4 per zone (`tail`)
        let mut out = vec![0.; head + 7 * nz + 4 * self.markets + 4 * nz];
        self.write_zone_xy(&mut out[head + 4 * nz..]);
        if xy.len() != 2 * share.len() || market as usize >= self.markets {
            return out;
        }
        let (mut pop_w, mut attr_w, mut job_w) = (vec![0f64; nz], vec![0f64; nz], vec![0f64; nz]);
        let mut matched = 0.;
        for (i, &f) in share.iter().enumerate() {
            let key = (
                (xy[2 * i] / SITE_M).floor() as i32,
                (xy[2 * i + 1] / SITE_M).floor() as i32,
                market as usize,
            );
            let Some(&c) = self.site_key.get(&key) else {
                continue;
            };
            let f = (f as f64).clamp(0., 1.);
            matched += 1.;
            let z = self.cell_zone[c];
            pop_w[z] += f * self.city.pop[c] as f64 / self.zones[z].pop.max(1e-12);
            attr_w[z] += f * self.city.jobs[c] as f64 / self.zones[z].attr.max(1e-12);
            // T-276: the commutes' work end is jobs alone (attraction is people + 2 x jobs)
            let zjobs = (self.zones[z].attr - self.zones[z].pop) / 2.;
            if zjobs > 0. {
                job_w[z] += f * ((self.city.jobs[c] - self.city.pop[c]) as f64 / 2.).max(0.) / zjobs;
            }
        }
        out[0] = if share.is_empty() {
            1.
        } else {
            matched / share.len() as f64
        };
        if self.solved_mask() == 0 {
            return out;
        }
        // T-174: each selected zone's pairs with every other zone (partner ascending, the order
        // the whole market's pair list gives), computed here once per zone; rail is on the served
        // pairs only, every other pair has none. (partner, trips, served pair out, served pair in)
        const NONE: u32 = u32::MAX;
        let mut rows: HashMap<usize, Vec<(u32, u32, u32, f64)>> = HashMap::new();
        for z in (0..nz).filter(|&z| pop_w[z] > 0. || attr_w[z] > 0.) {
            let row = zone_row(&self.zones, self.ground, &self.derived.totals, self.rate, z)
                .into_iter()
                .map(|(b, trips)| {
                    let at = |a: usize, b: usize| self.pair_index.get(&(a as u32, b as u32)).copied().unwrap_or(NONE);
                    (b as u32, at(z, b), at(b, z), trips)
                })
                .collect();
            rows.insert(z, row);
        }
        let rail = |per: &Period, i: u32| if i == NONE { 0. } else { per.pair_rail[i as usize] };
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let sh = share_out(day, LONGER, q) as f64;
            let f = load_factor(day, LONGER, q) as f64;
            for (z, &w) in pop_w.iter().enumerate().filter(|(_, w)| **w > 0.) {
                for &(b, out_i, _, trips) in &rows[&z] {
                    let b = b as usize;
                    let (m, all) = (self.zones[b].market, w * trips * sh);
                    let r = w * rail(per, out_i) as f64 * f;
                    let o = 1 + 8 * m;
                    out[o] += all;
                    out[o + 1] += r;
                    out[o + 2] += all * self.zones[b].x as f64;
                    out[o + 3] += all * self.zones[b].y as f64;
                    out[head + 4 * b] += all;
                    out[head + 4 * b + 1] += r;
                }
            }
            for (z, &w) in attr_w.iter().enumerate().filter(|(_, w)| **w > 0.) {
                for &(a, _, in_i, trips) in &rows[&z] {
                    let a = a as usize;
                    let (m, all) = (self.zones[a].market, w * trips * sh);
                    let r = w * rail(per, in_i) as f64 * f;
                    let o = 1 + 8 * m + 4;
                    out[o] += all;
                    out[o + 1] += r;
                    out[o + 2] += all * self.zones[a].x as f64;
                    out[o + 3] += all * self.zones[a].y as f64;
                    out[head + 4 * a + 2] += all;
                    out[head + 4 * a + 3] += r;
                }
            }
        }
        // T-276: the commutes between zones. Per far market, then per far zone: legs a day (both
        // directions) of the commuters living in the selection, by where they work, and rail legs;
        // then the same for the commuters working in the selection, by where they live.
        let tail = head + 7 * nz;
        let mut add = |far: usize, side: usize, all: f64, rail: f64| {
            let m = self.zones[far].market;
            out[tail + 4 * m + 2 * side] += all;
            out[tail + 4 * m + 2 * side + 1] += rail;
            let o = tail + 4 * self.markets + 4 * far + 2 * side;
            out[o] += all;
            out[o + 1] += rail;
        };
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let legs = (share_out(day, WORK, q) + share_ret(day, WORK, q)) as f64;
            let rail_legs = (TO_WORK[q] + TO_HOME[q]) as f64 * load_factor(day, WORK, q) as f64;
            let frac = |c: u32| match self.cpair_of.get(c as usize) {
                Some(&i) if i != u32::MAX => per.c_frac.get(i as usize).copied().unwrap_or(0.) as f64,
                _ => 0.,
            };
            for (end, weights) in [(0usize, &pop_w), (1usize, &job_w)] {
                let (off, items) = if end == 0 { &self.c_home } else { &self.c_work };
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    for &c in &items[off[z] as usize..off[z + 1] as usize] {
                        let (h, wz, n) = self.commutes[c as usize];
                        let far = if end == 0 { wz } else { h } as usize;
                        let n = w * n as f64;
                        add(far, end, n * legs, n * rail_legs * frac(c));
                    }
                }
            }
        }
        out
    }
    /// A station's longer-trip rail riders over the periods solved (T-118; `station` indexes the
    /// network handed to `set_network`): per market, riders starting here bound for it, riders
    /// ending here from it, and those counts times the far zone's x and y (starts x, starts y,
    /// ends x, ends y; `market_count()` numbers each). Empty with no such station.
    pub fn station_flows(&self, station: u32) -> Vec<f64> {
        self.station_flows_day(station, WEEKDAY as u32)
    }
    /// `station_flows` on a kind of day (T-260): each period weighed by the market's load factor.
    pub fn station_flows_day(&self, station: u32, day: u32) -> Vec<f64> {
        let Some(svc) = &self.svc else { return vec![] };
        let Some(&s) = svc.st_map.get(station as usize) else {
            return vec![];
        };
        let nz = self.zones.len();
        let head = 6 * self.markets;
        let mut out = vec![0.; head + 5 * nz];
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let f = load_factor(day as usize, LONGER, q) as f64;
            if let Some(row) = per.st.get(s as usize * 2 * nz..(s as usize + 1) * 2 * nz) {
                for (z, &v) in row.iter().enumerate() {
                    out[head + z] += v as f64 * f;
                }
            }
        }
        // the market sums, from the zones' sums: counts, then counts times the zone's x and y
        for d in 0..2 {
            for z in 0..nz {
                let (v, zone) = (out[head + d * nz + z], &self.zones[z]);
                out[d * self.markets + zone.market] += v;
                out[(2 + 2 * d) * self.markets + zone.market] += v * zone.x as f64;
                out[(3 + 2 * d) * self.markets + zone.market] += v * zone.y as f64;
            }
        }
        self.write_zone_xy(&mut out[head + 2 * nz..]);
        out
    }
    /// `station_flows` of a station this network does not hold but the whole one does (T-326: a
    /// station in another cluster): no riders, the zones as always.
    pub fn no_station_flows(&self) -> Vec<f64> {
        let nz = self.zones.len();
        let head = 6 * self.markets;
        let mut out = vec![0.; head + 5 * nz];
        self.write_zone_xy(&mut out[head + 2 * nz..]);
        out
    }
    /// Every zone's position (x, y in the New York frame, metres) and exact market membership.
    fn write_zone_xy(&self, out: &mut [f64]) {
        for (z, zone) in self.zones.iter().enumerate() {
            out[3 * z] = zone.x as f64;
            out[3 * z + 1] = zone.y as f64;
            out[3 * z + 2] = zone.market as f64;
        }
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| {
            s.avg
                .seg
                .iter()
                .enumerate()
                .map(|(r, &v)| {
                    let l = s.g.rn_line[r] as usize;
                    let cap = 60. / s.net.line_headway_min[l][q] * s.net.line_crush[l];
                    if cap > 0. {
                        v / s.p.period_hours * s.p.peak_hour_factor / cap
                    } else {
                        0.
                    }
                })
                .collect()
        })
    }
}
