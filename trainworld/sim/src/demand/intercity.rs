//! T-117: the regional market of longer trips and commutes between zones, on the common rail graph.
//!
//! T-377 (T-362 stage 3, docs/regional.md 5.1): one world market. Every registered zone is a
//! *place*; its part of the market is its own pack (`data/packs/<id>.market.json`, written by
//! pipeline/market_zones.py): its 600 m sites in its own frame, its 6 km zones and each zone's two
//! normalisers against the world field (ground and tail, notes/T-376.md). The registry carries a
//! row per place (the zones' sums, its centre, country, income and landmass, and the trips to each
//! place under `near_km`), so any pair of places has its trips with no pack open. A market here is
//! a *composition*: the world rows plus the packs of the places a worker has open; pairs are built
//! only between 6 km zones some station of its network serves, on one track component, on one
//! landmass or with air between them. T-378: air is a third mode between far places each with an
//! airport, by place (the same whatever the network); it is not built, has no capacity and loads
//! nothing.
//!
//! A pack in the old single-pack form (`sites` with a market column: the Northeast's
//! data/packs/intercity.json, the joint commute gravity's input, and the tests' fixtures) opens as
//! a closed world over its own zones with the ground curve alone.
use super::{
    api::{check_fare_curve, demand_groups, demand_passages, demand_ups, fare_x_usd, merge_stations, put_fare_curve, same_bits, span_for, scale_triples, DemandApi, Service, FARE_MIN_PER_USD, PEAK_HOUR_FACTOR, TO_HOME, TO_WORK},
    clock::now_ms,
    days::{everyday as everyday_purpose, load_factor, share_out, share_ret, LONGER, WEEKDAY, WORK},
    everyday::{KIND_COUNT, OUTWARD, RETURN},
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
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

mod drives;
mod market_trip;
use drives::{Drive, Drives};

pub const SHARES: [f32; PERIODS] = [0.20, 0.35, 0.25, 0.15, 0.05];

/// T-377: the market's rule (pipeline/longhaul.json, notes/T-376.md), as the registry's
/// `market.rule` carries it. Departures a weekday per resident: ground `r x s`, tail
/// `rt x s x u^tho / u_mean_us`, each shared by its own normaliser; `s` the origin's income
/// factor (`inc^eps`), `u` its r5 cell's people over `u_ref_people`.
#[derive(Deserialize, Serialize, Clone, Debug)]
struct Rule {
    #[serde(rename = "R")]
    r: f64,
    #[serde(rename = "A")]
    a: f64,
    onset_km: [f64; 2],
    d0_km: f64,
    eps: f64,
    #[serde(rename = "G")]
    g: f64,
    #[serde(rename = "D1_km")]
    d1_km: f64,
    #[serde(rename = "TH")]
    th: f64,
    eps_t: f64,
    #[serde(rename = "THo")]
    tho: f64,
    tail_onset_km: [f64; 2],
    border_eps: f64,
    u_mean_us: f64,
    u_ref_people: f64,
    border: f64,
    #[serde(rename = "Rt")]
    rt: f64,
    near_km: f64,
    min_trips: f64,
}
impl Rule {
    /// A closed world's rule (the old single pack): the ground curve at the pack's trip rate, no
    /// tail, no border, one income.
    fn closed(rate: f64) -> Rule {
        Rule {
            r: rate,
            a: DECAY_A,
            onset_km: [ONSET_KM as f64, 2. * ONSET_KM as f64],
            d0_km: ONSET_KM as f64,
            eps: 0.,
            g: 0.7,
            d1_km: 171.,
            th: 0.,
            eps_t: 0.,
            tho: 0.,
            tail_onset_km: [300., 700.],
            border_eps: 0.,
            u_mean_us: 1.,
            u_ref_people: 1e5,
            border: 1.,
            rt: 0.,
            near_km: NEAR_KM,
            min_trips: MIN_TRIPS,
        }
    }
    fn ok(&self) -> bool {
        let v = [self.r, self.a, self.d0_km, self.eps, self.g, self.d1_km, self.th, self.eps_t, self.tho, self.border_eps, self.u_mean_us, self.u_ref_people, self.border, self.rt, self.near_km, self.min_trips];
        v.iter().chain(&self.onset_km).chain(&self.tail_onset_km).all(|x| x.is_finite() && *x >= 0.)
            && self.onset_km[1] > self.onset_km[0]
            && self.tail_onset_km[1] > self.tail_onset_km[0]
            && self.u_mean_us > 0.
            && self.u_ref_people > 0.
    }
    /// The ground curve at `d` km: the onset times `max(d, d0)^-A`; 0 under the onset.
    #[inline(always)]
    fn ground(&self, d: f64) -> f64 {
        let on = ((d - self.onset_km[0]) / (self.onset_km[1] - self.onset_km[0])).clamp(0., 1.);
        if on == 0. {
            0.
        } else {
            on / d.max(self.d0_km).powf(self.a)
        }
    }
    /// The long-haul tail at `d` km: its onset times `max(d, D1)^-G`; 0 under the onset.
    #[inline(always)]
    fn tail(&self, d: f64) -> f64 {
        let on = ((d - self.tail_onset_km[0]) / (self.tail_onset_km[1] - self.tail_onset_km[0])).clamp(0., 1.);
        if on == 0. || self.rt == 0. {
            0.
        } else {
            on / d.max(self.d1_km).powf(self.g)
        }
    }
}
/// T-809: the open-border area, the EU 27, Schengen's other members and the UK (ISO 3166 codes as
/// the rows carry them). A pair in two of its countries takes `OPEN_AREA_BORDER` in place of the
/// rule's border factor, whatever the origin's income; checked on Channel and Benelux corridors
/// (notes/T-809.md). pipeline/market_zones.py's normalisers must use the same rule.
const OPEN_AREA: [&str; 32] = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS", "CH", "LI", "GB",
];
const OPEN_AREA_BORDER: f64 = 0.5;
/// The border factor from a place in country `a` with income `inc` to one in country `b`.
fn border_factor(rule: &Rule, a: &str, inc: f64, b: &str) -> f64 {
    let general = rule.border * inc.powf(rule.border_eps);
    if a == b {
        1.
    } else if OPEN_AREA.contains(&a) && OPEN_AREA.contains(&b) {
        OPEN_AREA_BORDER.max(general)
    } else {
        general
    }
}
/// T-276: the onset, where the market takes over from the local layer's outings (0 at
/// `ONSET_KM`, 1 at twice it; everyday.rs tapers its outings by the complement).
pub const ONSET_KM: f32 = 25.;
/// T-276: the ground curve's power, fitted to NHTS 2017 (research/intercity/market_fit.py).
const DECAY_A: f64 = 2.37;
/// Pairs with fewer trips a day each way are dropped, both directions together.
const MIN_TRIPS: f64 = 0.01;
/// T-362 decision 6: places whose centres are nearer than this pair 6 km zone to 6 km zone;
/// farther ones place to place, spread inside each end.
const NEAR_KM: f64 = 250.;

/// A frame as the registry and the packs write it: `kind` "equirect" or "lcc", origin in degrees.
#[derive(Deserialize, Clone)]
struct PackFrame {
    kind: String,
    lon: f64,
    lat: f64,
}
fn make_frame(f: &PackFrame) -> Result<Frame, String> {
    if !f.lon.is_finite() || !f.lat.is_finite() || f.lat.abs() >= 90. {
        return Err("invalid market frame origin".into());
    }
    match f.kind.as_str() {
        "equirect" => Ok(Frame::equirect(f.lon, f.lat)),
        "lcc" => Ok(Frame::lcc(f.lon, f.lat)),
        k => Err(format!("unknown market frame {k:?}")),
    }
}

/// The old single pack: sites `[x, y, people, jobs, market]` in one frame (or one plane).
#[derive(Deserialize)]
struct LegacyPack {
    #[serde(default = "default_zone_m")]
    zone_m: f64,
    #[serde(default = "default_trip_rate")]
    trip_rate: f64,
    #[serde(default = "legacy_markets")]
    markets: Vec<String>,
    sites: Vec<[f64; 5]>,
    #[serde(default)]
    frame: Option<PackFrame>,
    #[serde(default)]
    site_park: Vec<f32>,
    #[serde(default)]
    commute_home: Vec<u32>,
    #[serde(default)]
    commute_work: Vec<u32>,
    #[serde(default)]
    commute_trips: Vec<f32>,
}
fn legacy_markets() -> Vec<String> {
    ["New York", "Philadelphia", "Trenton / Mercer County"].map(str::to_owned).to_vec()
}
fn default_zone_m() -> f64 {
    6000.
}
fn default_trip_rate() -> f64 {
    0.015
}
/// A place's part of the market (`data/packs/<id>.market.json`, pipeline/market_zones.py).
#[derive(Deserialize, Default)]
struct ZoneIn {
    place: usize,
    #[serde(default)]
    zone: String,
    frame: Option<PackFrame>,
    #[serde(default = "default_zone_m")]
    zone_m: f64,
    #[serde(default)]
    country: String,
    #[serde(default = "one")]
    inc: f64,
    #[serde(default)]
    road: i64,
    /// its 6 km zones' squares, and their ground and tail normalisers
    zones: Vec<[i32; 2]>,
    #[serde(default)]
    zone_ng: Vec<f64>,
    #[serde(default)]
    zone_nt: Vec<f64>,
    /// `[x, y, people, jobs, zone]`, x and y in `frame`
    sites: Vec<[f64; 5]>,
    #[serde(default)]
    site_park: Vec<f32>,
    /// people of each site's H3 r5 cell (the tail's density)
    #[serde(default)]
    cell_pop: Vec<f64>,
    /// commutes from this place: home zone, work place, work zone (that place's own), commuters a day
    #[serde(default)]
    commute_home: Vec<u32>,
    #[serde(default)]
    commute_place: Vec<u32>,
    #[serde(default)]
    commute_work: Vec<u32>,
    #[serde(default)]
    commute_trips: Vec<f32>,
    /// T-1175: everyday outings from homes here across a zone line: home zone, activity place,
    /// activity zone (that place's own), purpose (`everyday` kind), outings a day and the share
    /// of them walking the whole way with no rail
    #[serde(default)]
    everyday_home: Vec<u32>,
    #[serde(default)]
    everyday_place: Vec<u32>,
    #[serde(default)]
    everyday_zone: Vec<u32>,
    #[serde(default)]
    everyday_kind: Vec<u8>,
    #[serde(default)]
    everyday_trips: Vec<f32>,
    #[serde(default)]
    everyday_walk: Vec<f32>,
    /// T-378: the airports its people fly from (longitude, latitude), and for each the places it
    /// has direct flights to (pipeline/market_zones.py)
    #[serde(default)]
    airports: Vec<[f64; 2]>,
    #[serde(default)]
    air_direct: Vec<Vec<u32>>,
}
fn one() -> f64 {
    1.
}
/// A place's market row (app/src/game/market-rows.json, T-776), written by `bin/market_pack.rs`.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Row {
    place: usize,
    #[serde(default)]
    zone: String,
    lon: f64,
    lat: f64,
    country: String,
    inc: f64,
    road: i64,
    /// people times ground and tail departures over their normalisers (`sum O/N`), attraction for
    /// each (ground, tail), people, departures a day, 6 km zones
    g: f64,
    t: f64,
    a: f64,
    ta: f64,
    pop: f64,
    dep: f64,
    zones: u32,
    /// trips a day each way to each place under `near_km` (itself included)
    #[serde(default)]
    near: Vec<(u32, f64)>,
    /// commuters a day from here to each other place
    #[serde(default)]
    commute: Vec<(u32, f64)>,
    /// T-378: per airport its people fly from, the drive there in minutes (people-weighted over
    /// its zones) and the places it flies to directly; and a car's parking here (perceived
    /// minutes, attraction-weighted)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    air: Vec<(f64, Vec<u32>)>,
    #[serde(default)]
    park: f64,
    /// T-1277: perceived minutes a fare dollar costs people leaving here (the app adds it to the
    /// row from its country's income, game/income.ts); 0 or absent: `FARE_MIN_PER_USD`. Never
    /// written to market-rows.json
    #[serde(default, skip_serializing_if = "is_zero")]
    fw: f64,
}
fn is_zero(v: &f64) -> bool {
    *v == 0.
}
/// The world: the output frame (positions the app reads), the rule, the place count and the rows.
#[derive(Deserialize, Clone)]
struct WorldIn {
    frame: PackFrame,
    rule: Rule,
    places: usize,
    #[serde(default)]
    rows: Option<Vec<Row>>,
}
#[derive(Deserialize)]
struct ComposeIn {
    world: WorldIn,
    zones: Vec<ZoneIn>,
}

/// A place: its row's numbers as the pair formula reads them.
#[derive(Clone, Default)]
struct Place {
    has: bool,
    open: bool,
    /// its centre on the unit sphere (ground), and in the output frame
    u: [f64; 3],
    x: f32,
    y: f32,
    lon: f64,
    lat: f64,
    country: u32,
    inc: f64,
    road: i64,
    g: f64,
    t: f64,
    a: f64,
    ta: f64,
    pop: f64,
    dep: f64,
    zones: u32,
    off: u32,
    zone: String,
    air: Vec<(f64, Vec<u32>)>,
    park: f64,
    /// its row's `fw` (0: `FARE_MIN_PER_USD`)
    fw: f64,
}
/// Per ordered pair of places: the border factor, the tail's income factor, near or not, and
/// the distance between their centres; T-378: whether a car can make the trip, the share of the
/// trips not by rail that fly, and the weight of every mode but rail against the reference
/// (`refm`: the car's minutes, or with no car the plane's).
#[derive(Clone, Copy, Default)]
struct PairRule {
    b: f64,
    rel: f64,
    near: bool,
    km: f64,
    car: bool,
    air: f32,
    rest: f32,
    refm: f32,
    /// the car's perceived minutes for water it is carried over (`crossing_min`, a ferry)
    cross: f32,
    /// T-279: the routed drive's minutes and road km over the rule's (`Drive`)
    rt: f32,
    rk: f32,
}
impl PairRule {
    /// A pair the solve may build: a car or a plane to compete with.
    fn has_alt(&self) -> bool {
        self.car || self.air > 0.
    }
    /// A longer trip's car over a zone pair of this place pair, `d` km apart, to parking `park`.
    fn car_min(&self, d: f32, park: f32) -> f32 {
        routed_car_min(d, park, self.rt, self.rk) + self.cross
    }
}
/// The ordered pair (i, j)'s border factor, the tail's income factor, near or not, and the
/// distance between their centres (`pair_rule`'s first part).
#[inline]
fn pair_trip_rule(places: &[Place], countries: &[String], rule: &Rule, ground: bool, i: usize, j: usize) -> (f64, f64, bool, f64) {
    let (p, q) = (&places[i], &places[j]);
    let km = if i == j {
        0.
    } else if ground {
        chord_km(&p.u, &q.u)
    } else {
        ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
    };
    let b = border_factor(rule, &countries[p.country as usize], p.inc, &countries[q.country as usize]);
    let rel = if p.inc > 0. { (q.inc / p.inc).min(1.).powf(rule.eps_t) } else { 1. };
    (b, rel, km < rule.near_km, km)
}
/// The ordered pair (i, j)'s rule (`PairRule`).
fn pair_rule(places: &[Place], countries: &[String], rule: &Rule, ground: bool, mp: &Params, drives: &Drives, i: usize, j: usize) -> PairRule {
    let (b, rel, near, km) = pair_trip_rule(places, countries, rule, ground, i, j);
    let (car, air, rest, refm, cross, d) = pair_mode_rule(places, countries, mp, drives, i, j, near, km);
    PairRule { b, rel, near, km, car, air, rest, refm, cross, rt: d.rt, rk: d.rk }
}
/// `pair_rule`'s modes for the ordered pair (i, j), near or not, `km` apart: whether a car can go,
/// the share of the trips not by rail that fly, the weight of every mode but rail, the reference
/// minutes, the crossings' minutes and the routed drive.
#[inline]
fn pair_mode_rule(places: &[Place], countries: &[String], mp: &Params, drives: &Drives, i: usize, j: usize, near: bool, km: f64) -> (bool, f32, f32, f32, f32, Drive) {
    let (p, q) = (&places[i], &places[j]);
    // T-378: a car within a landmass; a plane between far places each with an airport;
    // both by place, the same whoever builds what (very far pairs coarse, T-362)
    let car = p.road == q.road;
    // the best pair of airports: the drives to them, and a direct flight or a change
    let flight = if i == j || near {
        None
    } else {
        let mut best: Option<f64> = None;
        for (x, xd) in &p.air {
            for (y, yd) in &q.air {
                let direct = xd.contains(&(j as u32)) || yd.contains(&(i as u32));
                let v = air_min(km, x + y, direct, p.fw);
                best = Some(best.map_or(v, |b| b.min(v)));
            }
        }
        best.map(|v| v as f32)
    };
    let d = if car { drives.get(i, j) } else { Drive::RULE };
    let cross = if car && i != j { crossing_min((&countries[p.country as usize], p.lat), (&countries[q.country as usize], q.lat)) + d.ferry_min() } else { 0. };
    let car_m = routed_car_min(km as f32, (0.5 * (p.park + q.park)) as f32, d.rt, d.rk) + cross;
    let (air, rest, refm) = match (car, flight) {
        (true, Some(f)) => {
            let w = (-mp.mode_beta(car_m) * (f - car_m)).clamp(-80., 80.).exp();
            (w / (1. + w), 1. + w, car_m)
        }
        (false, Some(f)) => (1., 1., f),
        _ => (0., 1., car_m),
    };
    (car, air, rest, refm, cross, d)
}
struct Zone {
    /// centre in the output frame, and on the unit sphere (zeros in a plane pack)
    x: f32,
    y: f32,
    u: [f64; 3],
    /// people; people plus twice the jobs and the jobs (the shares a site takes of the zone)
    pop: f64,
    w_attr: f64,
    jobs: f64,
    /// people over the normalisers (ground, tail) and attraction (ground, tail)
    g: f64,
    t: f64,
    a: f64,
    ta: f64,
    place: usize,
    /// parking for a car arriving here, perceived minutes (its sites' `site_park`, jobs-weighted)
    park: f32,
    gid: u32,
    /// departures a day, to anywhere (T-378: those to places with no zone are what is left)
    dep: f64,
}
/// One place's sites, in its pack's frame: the access search and the rides work in its plane.
struct Group {
    frame: Option<Frame>,
    /// global site indices `lo..lo + city.len()`
    lo: usize,
    city: City,
    kz: Zones,
    kplace: Vec<(usize, f32, f32)>,
}
/// The market's own data, shared (`Rc`) by every market `beside` it (T-409).
struct Market {
    rule: Rule,
    /// zone distances on the sphere; false: a plane pack (straight in its plane)
    ground: bool,
    /// stations come as longitude and latitude (`set_network_ll`), else in the pack's plane
    world: bool,
    /// rows from the world (1) or derived at opening (0)
    from_rows: bool,
    anchor: Option<Frame>,
    places: Vec<Place>,
    countries: Vec<String>,
    /// T-1204: the pair rules of the open places (every pair of places was 300 MB a worker)
    rules: Rules,
    /// `market_params()`, for the pair rules worked out when asked
    mp: Params,
    /// T-279: the routed drives between places
    drives: Rc<Drives>,
    near: HashMap<(u32, u32), f64>,
    zones: Vec<Zone>,
    n_gid: usize,
    gid_place: Vec<u32>,
    comp_of_gid: Vec<u32>,
    groups: Vec<Group>,
    cell_zone: Vec<usize>,
    site_key: HashMap<(i32, i32, usize), usize>,
    /// commutes: home zone (here), work zone (global id), commuters a day; by home zone and by
    /// work zone (only those open), CSR
    commutes: Vec<(u32, u32, f32)>,
    c_home: (Vec<u32>, Vec<u32>),
    c_work: (Vec<u32>, Vec<u32>),
    /// T-1175: everyday outings across a zone line, by home zone (here), activity zone (global
    /// id) and purpose
    everyday: Vec<EPair>,
    /// `everyday` indices by home zone and by activity zone (only those open), CSR
    e_home: (Vec<u32>, Vec<u32>),
    e_act: (Vec<u32>, Vec<u32>),
    /// per home place and kind: the pairs' outings a day and the part walking with no rail
    e_base: Vec<(f64, f64)>,
    /// T-1204: the place pairs' figures no network changes, shared by every composition of a
    /// world with rows (`MarketPacks`)
    cw: Rc<Corridors>,
}
/// T-1204: the pair rules with an open place at one end: each open place's row (its pairs with
/// every place, `out`) and column (`inn`). Any other pair's rule is worked out when asked, which
/// only `corridor_modes` and the trip explorer between two places that are not open do (the
/// corridors' trips, `Market::corridors_of`, work out their factors themselves).
#[derive(Default)]
struct Rules {
    /// per place, its index among the open places, or `u32::MAX`
    slot: Vec<u32>,
    /// the open places, ascending
    open: Vec<usize>,
    out: Vec<PairRule>,
    inn: Vec<PairRule>,
}
/// T-1204: the market's corridors (unordered place pairs a <= b, numbered as a loop over a then
/// b: `Market::corridor`) and their figures that depend on the places alone: their longer trips a
/// day, both ways; the commutes between zones on each, both ways (corridor ascending, those with
/// any); each place's departures to places with no zone; the totals.
struct Corridors {
    /// distinct for each set worked out (`IntercityApi::corridors_id`)
    id: u32,
    trips: Vec<f64>,
    commute: Vec<(u32, f64)>,
    commuters: f64,
    leaving_by: Vec<f64>,
    daily: f64,
    leaving: f64,
    place_pairs: usize,
}

/// `drive` is the reference alternative's perceived minutes (the car's, or with no car the
/// plane's) and `rest` the weight of every mode but rail against it (T-378).
struct Pair {
    a: u32,
    b: u32,
    trips: f64,
    drive: f32,
    rest: f32,
    corridor: u32,
    /// the car's perceived minutes even where no car can go, which set the scale (T-495)
    car: f32,
    /// the station choice's band by the zones' distance (`Params::band`, T-718)
    band: u8,
}
/// T-276: a commute between two zones some station serves: its index in `commutes`, and the
/// car's perceived minutes from home to work (the local layer's commute rule).
struct CPair {
    c: u32,
    car: f32,
    corridor: usize,
    /// its zones' centres at least `ride_min_km` apart: the home end may ride (T-373)
    ride: bool,
    /// the station choice's band by the zones' distance (`Params::band`, T-718)
    band: u8,
}
/// T-1175: everyday outings between two zones across a zone line: home zone, activity zone (global
/// id), purpose (`everyday` kind), outings a day, and their share walking with no rail.
#[derive(Clone, Copy)]
struct EPair {
    h: u32,
    a: u32,
    kind: u8,
    trips: f32,
    walk: f32,
}
/// T-1229: the everyday kinds that cross zone lines (errands, social, school, university, medical),
/// which `IntercityApi::selection` lists by kind.
pub const E_CROSS: [usize; 5] = [0, 1, 2, 3, 5];
/// T-1175: an everyday pair some station serves: its index in `everyday`, the car's perceived
/// minutes (the commutes' car to the activity zone), the walk's weight against it, as `exp(uw)` of
/// the local layer's logit, from the pair's share walking with no rail.
struct EPairOn {
    e: u32,
    car: f32,
    walk_e: f32,
    ride: bool,
    band: u8,
}
/// T-276: the commutes between zones are the local layer's commuters, so they choose as it does.
fn commute_params() -> Params {
    Params::default()
}
/// The longer trips' choice (`IntercityApi::base` before any fare is set): the local layer's, with
/// no rail propensity; the length of the trip damps it (`Params::mode_beta`, T-495). Their car
/// pays no ownership cost (T-528), so neither does a drive to their stations.
fn market_params() -> Params {
    Params { asc_rail: 0., fare_min: 0., car_own_usd_trip: 0., car_own_usd_km: 0., ..Params::default() }
}
/// T-718: rail between two zones as the local layers choose it (T-710): the station pairs from
/// access list `a` to `b` (weights exp(-`beta` x access), `add_access`) at the station choice's
/// scale `th`, each pair's weight into `terms`. Returns rail's perceived cost, their logsum (the
/// first boarding's, access and egress included, as the mode choice then damps it, T-495), and the
/// sum of the weights (0: no route).
#[cfg(test)]
fn rail_logsum(r: &[f32], s: usize, fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, terms: &mut Vec<(u32, u32, f64)>) -> (f32, f64) {
    rail_logsum_x(r, s, fare, a, b, beta, th, None, terms)
}
/// T-1286: the fares of trips from one country weighed at a factor of the market's (the player's
/// fare scale there over its income's), on each station pair's strategy as searched: its expected
/// fare km (`trip_km_on`, S x S as `Skims::r`) back to the fare's perceived minutes.
struct FareX<'a> {
    km: &'a [f32],
    base: f32,
    per_km: f32,
    curve: &'a FareCurve,
}
impl FareX<'_> {
    /// The perceived minutes of the fare from station `a` to `b` in the market's weight.
    #[inline]
    fn fare(&self, a: usize, b: usize, s: usize) -> f32 {
        let k = self.km[a * s + b];
        self.base + self.per_km * k + self.curve.extra(k)
    }
}
/// `rail_logsum` with the fares weighed at `1 + x.1` (`FareX`).
#[allow(clippy::too_many_arguments)]
fn rail_logsum_x(r: &[f32], s: usize, fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, x: Option<(&FareX, f32)>, terms: &mut Vec<(u32, u32, f64)>) -> (f32, f64) {
    terms.clear();
    let k = (th / beta) as f64;
    let mut top = f64::NEG_INFINITY;
    // the egress weights' logs, once a pair rather than once a station pair (T-761)
    let mut lb = [0f64; 8];
    for (l, &(_, wb)) in lb.iter_mut().zip(b) {
        *l = k * (wb as f64).ln();
    }
    for &(sa, wa) in a {
        if wa <= 0. {
            continue;
        }
        let la = k * (wa as f64).ln();
        let row = &r[sa * s..(sa + 1) * s];
        for (j, &(sb, wb)) in b.iter().enumerate() {
            // a rail journey must actually board a train
            let cost = match x {
                None => row[sb] + fare,
                Some((fx, d)) => row[sb] + fare + d * fx.fare(sa, sb, s),
            };
            if sa == sb || wb <= 0. || !cost.is_finite() {
                continue;
            }
            let lw = la + (if j < 8 { lb[j] } else { k * (wb as f64).ln() }) - (th * cost) as f64;
            top = top.max(lw);
            terms.push((sa as u32, sb as u32, lw));
        }
    }
    if terms.is_empty() {
        return (f32::INFINITY, 0.);
    }
    let mut sum = 0f64;
    for t in terms.iter_mut() {
        t.2 = (t.2 - top).exp();
        sum += t.2;
    }
    (-((top + sum.ln()) / th as f64) as f32, sum)
}
struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    /// every part's loads together (crowding reads these), and the commutes' alone (T-276)
    avg: Loads,
    c_avg: Loads,
    rail: f64,
    /// T-1300: `Demand::rail_d`, averaged like the rail trips
    rail_d: FareSense,
    trips: f64,
    c_rail: f64,
    c_trips: f64,
    c_frac: Vec<f32>,
    corridors: Vec<f64>,
    c_corridors: Vec<f64>,
    /// Rail trips of each directed pair in this period (T-118), averaged over rounds like the loads.
    pair_rail: Vec<f32>,
    /// Per station, rail trips starting here by destination zone and ending here by origin zone
    /// (T-118, T-120), the entries of `IntercityApi::st_pat` (T-761).
    st: Vec<f32>,
    round: usize,
    ms: f64,
    fac: Vec<f32>,
    uniform: bool,
    /// `kernel::settled` (T-434)
    seen: Vec<f32>,
    xi: XferIndex,
    xfer: Vec<f32>,
    /// T-1175: the everyday pairs' loads by purpose, averaged like the commutes'; their legs, rail
    /// legs and walking legs in this period (every pair, served or not), and each served pair's
    /// rail share
    e_avg: Vec<Option<Loads>>,
    e_trips: f64,
    e_rail: f64,
    e_walk: f64,
    e_walk_base: f64,
    e_frac: Vec<f32>,
}
#[cfg(test)]
const ST_W: usize = 18;
/// Side of the endpoint sites in metres.
const SITE_M: f32 = 600.;
#[cfg(test)]
const MARKETS: usize = 3;
/// The bytes of strategies a market's pass holds at once (T-761): past this its destinations are
/// searched one at a time (`IntercityApi::pass_streamed`), this much of them kept between sweeps.
/// Not more (T-780): in WASM the heap grows a piece at a time for the strategies kept, which cost
/// minutes on all of Japan's first pass, and growing it at once raised the peak by 600 MB.
const MARKET_STRATEGY_BYTES: usize = 256 << 20;
/// T-780: the trips a day each way a sampled zone pair carries (`solved_pairs`).
const MARKET_CHUNK: f64 = 1.;
/// A destination's strategy for `IntercityApi::assigned`: a table's, or searched for it alone.
enum Held<'a> {
    Ref(Strategy<'a>),
    Own(DestStrategy),
}
impl Held<'_> {
    fn view(&self) -> Strategy<'_> {
        match self {
            Held::Ref(s) => *s,
            Held::Own(d) => d.view(),
        }
    }
}
/// The commutes' station to station trips by destination (T-761: about 1.6 million of Japan's
/// 106 million station pairs carry any): destination d's origins `src[off[d]..off[d + 1]]`
/// ascending, their riders in `v`.
struct SparseOd {
    off: Vec<u32>,
    src: Vec<u32>,
    v: Vec<f32>,
}
impl SparseOd {
    /// From (destination, origin, riders) in the order a dense table would have added them up:
    /// each pair's sum is the dense table's, added in the same order.
    fn of(s: usize, mut adds: Vec<(u32, u32, f32)>) -> SparseOd {
        // stable: a pair's riders stay in the order they were added
        adds.sort_by_key(|e| (e.0, e.1));
        let mut od = SparseOd { off: vec![0; s + 1], src: vec![], v: vec![] };
        let mut i = 0;
        while i < adds.len() {
            let (d, o) = (adds[i].0, adds[i].1);
            let mut sum = 0f32;
            while i < adds.len() && adds[i].0 == d && adds[i].1 == o {
                sum += adds[i].2;
                i += 1;
            }
            od.src.push(o);
            od.v.push(sum);
            od.off[d as usize + 1] += 1;
        }
        for d in 0..s {
            od.off[d + 1] += od.off[d];
        }
        od
    }
}
struct Demand {
    od: Vec<f32>,
    od_c: SparseOd,
    st: Vec<f32>,
    rail: f64,
    cor: Vec<f64>,
    pair_rail: Vec<f32>,
    c_rail: f64,
    c_frac: Vec<f32>,
    c_cor: Vec<f64>,
    e: EDemand,
    /// T-1300: every part's rail trips per perceived minute of the market's fare dearer (negative)
    rail_d: FareSense,
}
/// T-1175: the everyday pairs' part of a pass: station to station trips by purpose (empty for a
/// purpose with none), rail trips, the walkers rail takes, and each served pair's rail share.
#[derive(Default)]
struct EDemand {
    od: Vec<SparseOd>,
    rail: f64,
    walk_lost: f64,
    frac: Vec<f32>,
}
struct Pass {
    loads: Loads,
    c_loads: Loads,
    rail: f64,
    c_rail: f64,
    c_frac: Vec<f32>,
    cor: Vec<f64>,
    c_cor: Vec<f64>,
    pair_rail: Vec<f32>,
    st: Vec<f32>,
    xfer: Vec<f32>,
    /// T-1175: the everyday pairs' loads by purpose (`everyday` kind; none where it has no pair)
    e_loads: Vec<Option<Loads>>,
    e_rail: f64,
    e_walk_lost: f64,
    e_frac: Vec<f32>,
    rail_d: FareSense,
}

thread_local! {
    static CORRIDORS_SEEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
fn next_corridors_id() -> u32 {
    CORRIDORS_SEEN.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}
/// The index of pair (a, b), a <= b < n, among all such pairs numbered a then b.
#[inline(always)]
fn tri(n: usize, a: usize, b: usize) -> usize {
    a * (2 * n - a + 1) / 2 + (b - a)
}
#[inline(always)]
fn chord_km(ua: &[f64; 3], ub: &[f64; 3]) -> f64 {
    let c = ((ua[0] - ub[0]).powi(2) + (ua[1] - ub[1]).powi(2) + (ua[2] - ub[2]).powi(2)).sqrt();
    2. * EARTH_R / 1000. * (c / 2.).min(1.).asin()
}
fn unit(lon: f64, lat: f64) -> [f64; 3] {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}
fn lonlat_of(u: &[f64; 3]) -> (f64, f64) {
    (u[1].atan2(u[0]).to_degrees(), u[2].atan2(u[0].hypot(u[1])).to_degrees())
}
/// Driving minutes over `d` km.
fn drive_min(d: f32) -> f32 {
    15. + d * ROAD_DETOUR / 85. * 60.
}
/// The road is this times the straight line.
const ROAD_DETOUR: f32 = 1.25;
/// T-276: what a car costs to run a km, dollars (AAA 2024), and the people sharing it on a
/// longer trip (NHTS: about two).
const CAR_USD_PER_KM: f32 = 0.15;
const LONG_OCCUPANCY: f32 = 2.;
/// T-276: a longer trip's car as perceived minutes over `d` km to a zone where parking costs `park`.
#[cfg(test)]
fn car_min(d: f32, park: f32) -> f32 {
    routed_car_min(d, park, 1., 1.)
}
/// `car_min` with the rule's minutes and road km times `rt` and `rk` (T-279, `Drive`).
fn routed_car_min(d: f32, park: f32, rt: f32, rk: f32) -> f32 {
    drive_min(d) * rt + (d * ROAD_DETOUR * rk * CAR_USD_PER_KM * FARE_MIN_PER_USD + park) / LONG_OCCUPANCY
}
/// A car carried over water by a shuttle or ferry where the landmass's fixed link takes no cars
/// at its own price (T-712): the fare per car in dollars and the minutes beyond the straight drive.
struct Crossing {
    usd: f32,
    min: f32,
}
/// The Channel (Le Shuttle and the Dover ferries) between Great Britain and the continent.
const CHANNEL: Crossing = Crossing { usd: 150., min: 60. };
/// The Tsugaru Strait: the Seikan Tunnel carries no cars, the Aomori-Hakodate ferry (3 h 40) does.
const TSUGARU: Crossing = Crossing { usd: 170., min: 240. };
/// The car's perceived minutes for the crossings between two places (country, latitude) on one
/// landmass; the fare is shared like the car's running cost.
fn crossing_min(a: (&str, f64), b: (&str, f64)) -> f32 {
    let hokkaido = |p: (&str, f64)| p.0 == "JP" && p.1 > 41.4;
    let mut m = 0.;
    for (on, c) in [((a.0 == "GB") != (b.0 == "GB"), &CHANNEL), (hokkaido(a) != hokkaido(b), &TSUGARU)] {
        if on {
            m += c.min + c.usd * FARE_MIN_PER_USD / LONG_OCCUPANCY;
        }
    }
    m
}
/// T-378 (notes/T-362.md "Modes"), T-712: air door to door, minutes at the airports, cruise speed
/// and the flight's fixed minutes on the great circle, the extra minutes with no direct flight, the
/// fare (a flat base: published average fares barely rise with distance, notes/T-712.md).
const AIR_TERMINAL_MIN: f64 = 90.;
const AIR_KMH: f64 = 750.;
const AIR_FIXED_MIN: f64 = 30.;
const AIR_NO_DIRECT_MIN: f64 = 60.;
const AIR_FARE_USD: f64 = 100.;
const AIR_USD_PER_KM: f64 = 0.05;
/// A flight's perceived minutes over `km` with `access` minutes of driving at the two ends, its
/// fare at `fw` perceived minutes a dollar, the departing place's (T-1277; 0: `FARE_MIN_PER_USD`).
fn air_min(km: f64, access: f64, direct: bool, fw: f64) -> f64 {
    let fw = if fw > 0. { fw } else { FARE_MIN_PER_USD as f64 };
    access + AIR_TERMINAL_MIN + km / AIR_KMH * 60. + AIR_FIXED_MIN + if direct { 0. } else { AIR_NO_DIRECT_MIN } + (AIR_FARE_USD + AIR_USD_PER_KM * km) * fw
}

impl Market {
    #[inline(always)]
    fn zone_km(&self, a: &Zone, b: &Zone) -> f64 {
        if self.ground {
            chord_km(&a.u, &b.u)
        } else {
            ((a.x - b.x).hypot(a.y - b.y) / 1000.) as f64
        }
    }
    /// The ordered pair (i, j)'s rule: an open place's row or column, else worked out (T-1204).
    #[inline(always)]
    fn pr(&self, i: usize, j: usize) -> PairRule {
        let np = self.places.len();
        let r = &self.rules;
        let (si, sj) = (r.slot[i], r.slot[j]);
        if si != u32::MAX {
            r.out[si as usize * np + j]
        } else if sj != u32::MAX {
            r.inn[sj as usize * np + i]
        } else {
            pair_rule(&self.places, &self.countries, &self.rule, self.ground, &self.mp, &self.drives, i, j)
        }
    }
    /// Trips a day from zone-like sums `(g, t)` at place `i` to `(a, ta)` at place `j`, `d` km.
    #[inline(always)]
    fn directed(&self, i: usize, j: usize, g: f64, t: f64, a: f64, ta: f64, d: f64) -> f64 {
        let r = self.pr(i, j);
        self.directed_by(r.b, r.rel, g, t, a, ta, d)
    }
    /// `directed` with the pair's border and income factors given.
    #[inline(always)]
    fn directed_by(&self, b: f64, rel: f64, g: f64, t: f64, a: f64, ta: f64, d: f64) -> f64 {
        let ground = if g > 0. { g * self.rule.ground(d) * a } else { 0. };
        let tail = if t > 0. { t * self.rule.tail(d) * ta * rel } else { 0. };
        b * (ground + tail)
    }
    /// A 6 km zone pair's trips a day each way: the mean of the two directions. Near places' zones
    /// at their own distance, far places' at their centres' (so their pairs add up to the places').
    #[inline(always)]
    fn eachway(&self, a: &Zone, b: &Zone) -> f64 {
        let r = self.pr(a.place, b.place);
        let d = if r.near { self.zone_km(a, b) } else { r.km };
        (self.directed(a.place, b.place, a.g, a.t, b.a, b.ta, d) + self.directed(b.place, a.place, b.g, b.t, a.a, a.ta, d)) * 0.5
    }
    /// `eachway` with the curves at the pair's distance given (`curves`), so a run of far zones works
    /// them out once (T-780): the same figure to the bit.
    #[inline(always)]
    fn eachway_with(&self, a: &Zone, b: &Zone, (gc, tc): (f64, f64)) -> f64 {
        let one = |r: PairRule, g: f64, t: f64, at: f64, ta: f64| r.b * ((if g > 0. { g * gc * at } else { 0. }) + if t > 0. { t * tc * ta * r.rel } else { 0. });
        (one(self.pr(a.place, b.place), a.g, a.t, b.a, b.ta) + one(self.pr(b.place, a.place), b.g, b.t, a.a, a.ta)) * 0.5
    }
    /// The ground and tail curves of zones a and b (`eachway_with`).
    #[inline(always)]
    fn curves(&self, a: &Zone, b: &Zone) -> (f64, f64) {
        let r = self.pr(a.place, b.place);
        let d = if r.near { self.zone_km(a, b) } else { r.km };
        (self.rule.ground(d), self.rule.tail(d))
    }
    /// Two places' trips a day each way: near ones from the rows (every zone pair added up),
    /// far ones from their centres.
    fn place_eachway(&self, i: usize, j: usize) -> f64 {
        let r = self.pr(i, j);
        if r.near {
            return self.near.get(&(i.min(j) as u32, i.max(j) as u32)).copied().unwrap_or(0.);
        }
        let (p, q) = (&self.places[i], &self.places[j]);
        (self.directed(i, j, p.g, p.t, q.a, q.ta, r.km) + self.directed(j, i, q.g, q.t, p.a, p.ta, r.km)) * 0.5
    }
    /// A zone's trips a day each way with a place that is not open: far ones exactly; near ones
    /// the places' trips by the zone's share of its place's departures and attraction (judgement,
    /// T-377: the place's zones are not known here).
    fn zone_place_eachway(&self, z: &Zone, j: usize) -> f64 {
        let r = self.pr(z.place, j);
        if r.near {
            let p = &self.places[z.place];
            let share = 0.5 * (if p.g > 0. { z.g / p.g } else { 0. } + if p.a > 0. { z.a / p.a } else { 0. });
            return self.place_eachway(z.place, j) * share;
        }
        let q = &self.places[j];
        (self.directed(z.place, j, z.g, z.t, q.a, q.ta, r.km) + self.directed(j, z.place, q.g, q.t, z.a, z.ta, r.km)) * 0.5
    }
    /// The corridor of places i and j (either order): pairs a <= b numbered a then b.
    fn corridor(&self, i: usize, j: usize) -> usize {
        tri(self.places.len(), i.min(j), i.max(j))
    }
    fn n_corridors(&self) -> usize {
        let np = self.places.len();
        np * (np + 1) / 2
    }
    /// T-1204: the corridor of two open places among the open places' corridors (the periods'
    /// rail figures are kept for those alone), in the same order as `corridor`.
    #[inline]
    fn open_corridor(&self, i: usize, j: usize) -> usize {
        let (a, b) = (self.rules.slot[i] as usize, self.rules.slot[j] as usize);
        tri(self.rules.open.len(), a.min(b), a.max(b))
    }
    fn n_open_corridors(&self) -> usize {
        let k = self.rules.open.len();
        k * (k + 1) / 2
    }
    /// The open places' corridors: (index among them, corridor), in corridor order.
    fn open_corridors(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (np, open) = (self.places.len(), &self.rules.open);
        (0..open.len()).flat_map(move |a| (a..open.len()).map(move |b| tri(np, open[a], open[b]))).enumerate()
    }
    /// Every corridor's two places, in corridor order.
    fn each_corridor(&self) -> impl Iterator<Item = (usize, usize)> {
        let np = self.places.len();
        (0..np).flat_map(move |a| (a..np).map(move |b| (a, b)))
    }
    /// Site `s` (global): its group's city and its index there.
    #[inline(always)]
    fn site(&self, s: usize) -> (&Group, usize) {
        let g = match self.groups.binary_search_by(|g| g.lo.cmp(&s)) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (&self.groups[g], s - self.groups[g].lo)
    }
    fn n_sites(&self) -> usize {
        self.cell_zone.len()
    }
    /// T-1204: the corridors' figures (`Corridors`): each corridor's longer trips a day (as
    /// `place_eachway` gives them, its pair factors worked out here rather than kept), the
    /// commutes on it from the rows (`commute_rows`, by place) or else from the composition's.
    fn corridors_of(&self, commute_rows: Option<&[Vec<(u32, f64)>]>) -> Corridors {
        let np = self.places.len();
        let mut commute: BTreeMap<u32, f64> = BTreeMap::new();
        let mut commuters = 0.;
        match commute_rows {
            Some(rows) => {
                for (p, list) in rows.iter().enumerate() {
                    for &(j, n) in list {
                        if (j as usize) < np && n.is_finite() && n >= 0. {
                            *commute.entry(self.corridor(p, j as usize) as u32).or_insert(0.) += 2. * n;
                            commuters += n;
                        }
                    }
                }
            }
            None => {
                for &(h, w, n) in &self.commutes {
                    let c = self.corridor(self.zones[h as usize].place, self.gid_place[w as usize] as usize);
                    *commute.entry(c as u32).or_insert(0.) += 2. * n as f64;
                    commuters += n as f64;
                }
            }
        }
        // what `pair_trip_rule` reads of each place, once a place: the border factor's general
        // part, and whether its country is in the open area (country ids name one country each)
        let rule = &self.rule;
        let general: Vec<f64> = self.places.iter().map(|p| rule.border * p.inc.powf(rule.border_eps)).collect();
        let open_area: Vec<bool> = self.countries.iter().map(|c| OPEN_AREA.contains(&c.as_str())).collect();
        let factors = |i: usize, j: usize| {
            let (p, q) = (&self.places[i], &self.places[j]);
            let b = if p.country == q.country {
                1.
            } else if open_area[p.country as usize] && open_area[q.country as usize] {
                OPEN_AREA_BORDER.max(general[i])
            } else {
                general[i]
            };
            // a ratio of 1 or more is held at 1, and 1 to any power is 1
            let rel = if p.inc > 0. {
                let x = (q.inc / p.inc).min(1.);
                if x == 1. { 1. } else { x.powf(rule.eps_t) }
            } else {
                1.
            };
            (b, rel)
        };
        let mut ct = vec![0f64; self.n_corridors()];
        let mut to_places = vec![0f64; np];
        let mut pairs = 0usize;
        for i in 0..np {
            let p = &self.places[i];
            if !p.has {
                continue;
            }
            for j in i..np {
                let q = &self.places[j];
                if !q.has {
                    continue;
                }
                let km = if i == j {
                    0.
                } else if self.ground {
                    chord_km(&p.u, &q.u)
                } else {
                    ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
                };
                let t = if km < rule.near_km {
                    self.near.get(&(i as u32, j as u32)).copied().unwrap_or(0.)
                } else {
                    let ((bi, ri), (bj, rj)) = (factors(i, j), factors(j, i));
                    (self.directed_by(bi, ri, p.g, p.t, q.a, q.ta, km) + self.directed_by(bj, rj, q.g, q.t, p.a, p.ta, km)) * 0.5
                };
                if t > 0. {
                    ct[tri(np, i, j)] += 2. * t;
                    // a place's own pairs both start there
                    to_places[i] += t;
                    to_places[j] += t;
                    pairs += if i == j { 1 } else { 2 };
                }
            }
        }
        let leaving_by: Vec<f64> = (0..np).map(|p| if self.places[p].has { (self.places[p].dep - to_places[p]).max(0.) } else { 0. }).collect();
        Corridors { id: next_corridors_id(), daily: ct.iter().sum(), leaving: leaving_by.iter().sum(), trips: ct, commute: commute.into_iter().collect(), commuters, leaving_by, place_pairs: pairs }
    }
    /// Each open place's run of zones (zones are laid out place by place): (place, first, end).
    fn place_spans(&self) -> Vec<(usize, usize, usize)> {
        let mut spans: Vec<(usize, usize, usize)> = vec![];
        for (z, zz) in self.zones.iter().enumerate() {
            match spans.last_mut() {
                Some(s) if s.0 == zz.place => s.2 = z + 1,
                _ => spans.push((zz.place, z, z + 1)),
            }
        }
        spans
    }
    /// Every global zone's position (x, y in the output frame) and place: open ones their own,
    /// others their place's centre.
    fn write_zone_xy(&self, out: &mut [f64]) {
        for g in 0..self.n_gid {
            let c = self.comp_of_gid[g];
            let (x, y) = if c != u32::MAX {
                let z = &self.zones[c as usize];
                (z.x, z.y)
            } else {
                let p = &self.places[self.gid_place[g] as usize];
                (p.x, p.y)
            };
            out[3 * g] = x as f64;
            out[3 * g + 1] = y as f64;
            out[3 * g + 2] = self.gid_place[g] as f64;
        }
    }
    /// Zone `z`'s pairs with every other open zone (trips each way), partner ascending.
    #[inline(never)]
    fn zone_row(&self, z: usize) -> Vec<(usize, f64)> {
        let (a, min) = (&self.zones[z], self.rule.min_trips);
        let mut out = vec![];
        for (b, zb) in self.zones.iter().enumerate() {
            if b == z {
                continue;
            }
            let t = self.eachway(a, zb);
            if t >= min {
                out.push((b, t));
            }
        }
        out
    }
}

/// What a site table needs to be built into the market: one input per place.
struct Built {
    zones: Vec<Rc<ZoneIn>>,
    rule: Rule,
    out: Option<PackFrame>,
    places: usize,
    rows: Option<Vec<Row>>,
    world: bool,
    closed: bool,
    /// legacy: each pack site's place and index in its place's input (commutes name pack sites)
    legacy_commutes: Vec<(u32, u32, f32)>,
    drives: Rc<Drives>,
}
fn single_pack(pack: LegacyPack) -> Result<Built, String> {
    let zm = pack.zone_m;
    let markets = pack.markets.len();
    if markets == 0 || !zm.is_finite() || zm <= 0. || !pack.trip_rate.is_finite() || pack.trip_rate < 0. {
        return Err("invalid market parameters".into());
    }
    if let Some(f) = &pack.frame {
        make_frame(f)?;
    }
    if !pack.site_park.is_empty() && (pack.site_park.len() != pack.sites.len() || pack.site_park.iter().any(|p| !p.is_finite() || *p < 0.)) {
        return Err("invalid site parking".into());
    }
    let mut zones: Vec<ZoneIn> = (0..markets).map(|m| ZoneIn { place: m, frame: pack.frame.clone(), zone_m: zm, inc: 1., ..Default::default() }).collect();
    let mut keys: Vec<HashMap<(i32, i32), usize>> = vec![HashMap::new(); markets];
    let mut at = Vec::with_capacity(pack.sites.len());
    for (si, s) in pack.sites.iter().enumerate() {
        if s.iter().any(|x| !x.is_finite()) || s[2] < 0. || s[3] < 0. || s[4] < 0. || s[4].fract() != 0. || s[4] >= markets as f64 {
            return Err("invalid intercity weights".into());
        }
        let m = s[4] as usize;
        let k = ((s[0] as f32 / zm as f32).floor() as i32, (s[1] as f32 / zm as f32).floor() as i32);
        let zi = &mut zones[m];
        let n = zi.zones.len();
        let z = *keys[m].entry(k).or_insert(n);
        if z == n {
            zi.zones.push([k.0, k.1]);
        }
        at.push((m as u32, zi.sites.len() as u32));
        zi.sites.push([s[0], s[1], s[2], s[3], z as f64]);
        if let Some(&p) = pack.site_park.get(si) {
            zi.site_park.push(p);
        }
    }
    let nc = pack.commute_trips.len();
    if pack.commute_home.len() != nc || pack.commute_work.len() != nc {
        return Err("commute arrays disagree".into());
    }
    let mut legacy_commutes = Vec::with_capacity(nc);
    for i in 0..nc {
        let (h, w, n) = (pack.commute_home[i] as usize, pack.commute_work[i] as usize, pack.commute_trips[i]);
        if h >= at.len() || w >= at.len() || !n.is_finite() || n < 0. {
            return Err("invalid commute between zones".into());
        }
        legacy_commutes.push((h as u32, w as u32, n));
    }
    // the commutes name pack sites: keep each site's place and place-local index for them
    let site_at = at;
    let zones = zones.into_iter().map(Rc::new).collect();
    let mut built = Built { zones, rule: Rule::closed(pack.trip_rate), out: pack.frame.clone(), places: markets, rows: None, world: false, closed: true, legacy_commutes: vec![], drives: Rc::default() };
    built.legacy_commutes = legacy_commutes
        .into_iter()
        .map(|(h, w, n)| {
            let (hp, hs) = site_at[h as usize];
            let (wp, ws) = site_at[w as usize];
            // packed: place in the top 8 bits is enough for the old packs (at most 255 places)
            ((hp << 24) | hs, (wp << 24) | ws, n)
        })
        .collect();
    Ok(built)
}
fn compose(c: ComposeIn) -> Result<Built, String> {
    if !c.world.rule.ok() {
        return Err("invalid market rule".into());
    }
    make_frame(&c.world.frame)?;
    Ok(Built { zones: c.zones.into_iter().map(Rc::new).collect(), rule: c.world.rule, out: Some(c.world.frame), places: c.world.places, rows: c.world.rows, world: true, closed: false, legacy_commutes: vec![], drives: Rc::default() })
}

fn build(b: Built) -> Result<Market, String> {
    build_for(b, None, None)
}
/// `build`, with the near totals added up only for pairs of places with an end in `near_for`
/// (indexed by place), and only those places' commutes (T-929: the rows of those places alone).
/// `reuse`: the corridors' figures of an earlier composition of the same world with rows.
fn build_for(b: Built, near_for: Option<&[bool]>, reuse: Option<Rc<Corridors>>) -> Result<Market, String> {
    let Built { mut zones, rule, out, places: np, rows, world, closed, legacy_commutes, drives } = b;
    let out_frame = match &out {
        Some(f) => Some(make_frame(f)?),
        None => None,
    };
    let ground = out_frame.is_some();
    zones.sort_by_key(|z| z.place);
    if zones.windows(2).any(|w| w[0].place == w[1].place) || zones.iter().any(|z| z.place >= np) {
        return Err("market zones: a place twice or out of range".into());
    }
    if let Some(rows) = &rows {
        if rows.iter().any(|r| r.place >= np) {
            return Err("market row out of range".into());
        }
    }
    let mut countries: Vec<String> = vec![];
    let mut country_id = |s: &str| -> u32 {
        match countries.iter().position(|c| c == s) {
            Some(i) => i as u32,
            None => {
                countries.push(s.to_owned());
                countries.len() as u32 - 1
            }
        }
    };
    let mut places: Vec<Place> = vec![Place::default(); np];
    // global zone offsets: the rows' zone counts in place order, or (no rows) the packs'
    let counts: Vec<u32> = match &rows {
        Some(rows) => {
            let mut c = vec![0u32; np];
            for r in rows {
                c[r.place] = r.zones;
            }
            c
        }
        None => {
            let mut c = vec![0u32; np];
            for z in &zones {
                c[z.place] = z.zones.len() as u32;
            }
            c
        }
    };
    let mut off = 0u32;
    for (p, pl) in places.iter_mut().enumerate() {
        pl.off = off;
        pl.zones = counts[p];
        off += counts[p];
    }
    let n_gid = off as usize;
    let mut gid_place = vec![0u32; n_gid];
    for (p, pl) in places.iter().enumerate() {
        for g in pl.off..pl.off + pl.zones {
            gid_place[g as usize] = p as u32;
        }
    }
    // ---- sites and zones, grouped by frame
    let mut mz: Vec<Zone> = vec![];
    let mut cell_zone: Vec<usize> = vec![];
    let mut groups: Vec<Group> = vec![];
    let mut site_park: Vec<f32> = vec![];
    let mut site_u: Vec<f64> = vec![];
    let mut site_mult: Vec<(f64, f64)> = vec![];
    let mut pack_first_zone: Vec<usize> = vec![0; np];
    let mut pack_first_site: Vec<usize> = vec![0; np];
    // sites are laid out group by group, so one group's are contiguous
    let mut order: Vec<usize> = (0..zones.len()).collect();
    let frame_of = |z: &ZoneIn| -> Result<Option<Frame>, String> {
        match &z.frame {
            Some(f) => Ok(Some(make_frame(f)?)),
            None if !world => Ok(None),
            None => Err("a market zone without a frame".into()),
        }
    };
    let frames: Vec<Option<Frame>> = zones.iter().map(|z| frame_of(z)).collect::<Result<_, _>>()?;
    // one group per place: its access and its rides' 2 km zones depend on its own sites alone, so
    // a place's figures are the same whichever other packs are open (T-377)
    let gkeys: Vec<Option<Frame>> = frames.clone();
    let gidx: Vec<usize> = (0..zones.len()).collect();
    order.sort_by_key(|&i| zones[i].place);
    let mut group_cities: Vec<City> = gkeys.iter().map(|_| City { name: String::new(), h3: vec![], x: vec![], y: vec![], pop: vec![], jobs: vec![], region: vec![], gravity: None }).collect();
    let mut group_lo = vec![usize::MAX; gkeys.len()];
    for &zi_ in &order {
        let zin = &zones[zi_];
        let g = gidx[zi_];
        let gc = &mut group_cities[g];
        let place = zin.place;
        let nz = zin.zones.len();
        if !zin.zone_m.is_finite() || zin.zone_m <= 0. || !zin.inc.is_finite() || zin.inc <= 0. {
            return Err("invalid market zone parameters".into());
        }
        if world && (zin.zone_ng.len() != nz || zin.zone_nt.len() != nz || zin.zone_ng.iter().chain(&zin.zone_nt).any(|v| !v.is_finite() || *v < 0.)) {
            return Err(format!("place {place}: normalisers do not match its zones"));
        }
        if !zin.site_park.is_empty() && zin.site_park.len() != zin.sites.len() {
            return Err("invalid site parking".into());
        }
        if !zin.cell_pop.is_empty() && zin.cell_pop.len() != zin.sites.len() {
            return Err("invalid cell people".into());
        }
        if counts[place] as usize != nz {
            return Err(format!("place {place}: {nz} zones in its pack, {} in its row (stale registry row: run pipeline/market_zones.py)", counts[place]));
        }
        if group_lo[g] == usize::MAX {
            group_lo[g] = cell_zone.len();
        } else if group_lo[g] + gc.len() != cell_zone.len() {
            return Err("market groups out of order".into());
        }
        let first = mz.len();
        pack_first_zone[place] = first;
        pack_first_site[place] = cell_zone.len();
        let f = frames[zi_];
        for k in 0..nz {
            mz.push(Zone { x: 0., y: 0., u: [0.; 3], pop: 0., w_attr: 0., jobs: 0., g: 0., t: 0., a: 0., ta: 0., place, park: 0., gid: places[place].off + k as u32, dep: 0. });
        }
        let s_inc = zin.inc.min(1.).powf(rule.eps);
        let total_pop: f64 = zin.sites.iter().map(|s| s[2]).sum();
        let total_attr: f64 = zin.sites.iter().map(|s| s[2] + 2. * s[3]).sum();
        // T-377: a pack's attraction rescaled so its total is its people (times income), as the
        // world field's cells count people
        let k_attr = if closed || total_attr <= 0. { 1. } else { total_pop / total_attr * s_inc };
        let mut park_w = vec![0f64; nz];
        let mut park = vec![0f64; nz];
        for (si, s) in zin.sites.iter().enumerate() {
            if s.iter().any(|x| !x.is_finite()) || s[2] < 0. || s[3] < 0. || s[4] < 0. || s[4].fract() != 0. || s[4] as usize >= nz {
                return Err("invalid intercity weights".into());
            }
            let (x, y, pop, jobs, z) = (s[0] as f32, s[1] as f32, s[2], s[3], s[4] as usize);
            let attr = pop + 2. * jobs;
            let zz = &mut mz[first + z];
            let w = pop + attr;
            zz.x += x * w as f32;
            zz.y += y * w as f32;
            zz.pop += pop;
            zz.w_attr += attr;
            zz.jobs += jobs;
            zz.a += attr * k_attr;
            let u = (zin.cell_pop.get(si).copied().unwrap_or(0.).max(1.) / rule.u_ref_people).max(1e-12);
            zz.ta += attr * k_attr * u.powf(rule.th);
            site_u.push(u);
            let (sg, st) = (s_inc, s_inc * u.powf(rule.tho) / rule.u_mean_us);
            site_mult.push((sg, st));
            zz.dep += pop * (rule.r * sg + rule.rt * st);
            if let Some(&p) = zin.site_park.get(si) {
                if !p.is_finite() || p < 0. {
                    return Err("invalid site parking".into());
                }
                let jw = jobs + 1e-9;
                park[z] += p as f64 * jw;
                park_w[z] += jw;
            }
            site_park.push(zin.site_park.get(si).copied().unwrap_or(0.));
            gc.x.push(x);
            gc.y.push(y);
            gc.pop.push(pop as f32);
            gc.jobs.push(attr as f32);
            cell_zone.push(first + z);
        }
        for k in 0..nz {
            let z = &mut mz[first + k];
            let w = (z.pop + z.w_attr).max(1e-12) as f32;
            z.x /= w;
            z.y /= w;
            if park_w[k] > 0. {
                z.park = (park[k] / park_w[k]) as f32;
            }
            // the centre on the ground (an equirectangular frame is affine in lon/lat, so the
            // plane's weighted centre is the sites' weighted lon/lat), and in the output frame
            if let Some(f) = &f {
                let (lon, lat) = f.to_lonlat(z.x as f64, z.y as f64);
                z.u = unit(lon, lat);
                if let Some(o) = &out_frame {
                    let (ox, oy) = o.from_lonlat(lon, lat);
                    z.x = ox as f32;
                    z.y = oy as f32;
                }
            }
        }
        // departures over the normalisers (a closed world's come after every zone is known)
        if !closed {
            let lo = pack_first_site[place];
            for (si, s) in zin.sites.iter().enumerate() {
                let z = first + s[4] as usize;
                let (sg, st) = site_mult[lo + si];
                let (ng, nt) = (zin.zone_ng[s[4] as usize], zin.zone_nt[s[4] as usize]);
                let zz = &mut mz[z];
                if ng > 0. {
                    zz.g += s[2] * rule.r * sg / ng;
                }
                if nt > 0. {
                    zz.t += s[2] * rule.rt * st / nt;
                }
            }
        }
        // T-378: the drive from each zone's centre to each airport, people-weighted
        if !zin.air_direct.is_empty() && zin.air_direct.len() != zin.airports.len() || zin.airports.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("place {place}: invalid airports"));
        }
        let zs = &mz[first..first + nz];
        let w: f64 = zs.iter().map(|z| z.pop).sum();
        let air: Vec<(f64, Vec<u32>)> = if !ground {
            vec![]
        } else {
            zin.airports
                .iter()
                .enumerate()
                .map(|(k, a)| {
                    let u = unit(a[0], a[1]);
                    let s: f64 = zs.iter().map(|z| z.pop * drive_min(chord_km(&z.u, &u) as f32) as f64).sum();
                    (if w > 0. { s / w } else { drive_min(0.) as f64 }, zin.air_direct.get(k).cloned().unwrap_or_default())
                })
                .collect()
        };
        let aw: f64 = zs.iter().map(|z| z.w_attr).sum();
        let park = if aw > 0. { zs.iter().map(|z| z.park as f64 * z.w_attr).sum::<f64>() / aw } else { 0. };
        let pl = &mut places[place];
        pl.open = true;
        pl.country = country_id(&zin.country);
        pl.inc = zin.inc.min(1.);
        pl.road = zin.road;
        pl.zone = zin.zone.clone();
        pl.air = air;
        pl.park = park;
    }
    for (g, c) in group_cities.into_iter().enumerate() {
        let (kz, kplace) = ride_zones(&c);
        groups.push(Group { frame: gkeys[g], lo: group_lo[g], city: c, kz, kplace });
    }
    groups.sort_by_key(|g| g.lo);
    // ---- a closed world: each zone's normaliser over the others (the old pack's market)
    let ground_km = |a: &Zone, b: &Zone| if ground { chord_km(&a.u, &b.u) } else { ((a.x - b.x).hypot(a.y - b.y) / 1000.) as f64 };
    if closed {
        let mut n = vec![0f64; mz.len()];
        for a in 0..mz.len() {
            for b in a + 1..mz.len() {
                let f = rule.ground(ground_km(&mz[a], &mz[b]));
                if f > 0. {
                    n[a] += mz[b].a * f;
                    n[b] += mz[a].a * f;
                }
            }
        }
        for (z, zz) in mz.iter_mut().enumerate() {
            zz.g = if n[z] > 0. { zz.pop * rule.r / n[z] } else { 0. };
        }
    }
    // ---- the places' rows: the world's, or added up here
    let mut near: HashMap<(u32, u32), f64> = HashMap::new();
    let mut commute_rows: Vec<Vec<(u32, f64)>> = vec![vec![]; np];
    let from_rows = rows.is_some();
    match rows {
        Some(rows) => {
            for r in rows {
                if [r.lon, r.lat, r.inc, r.g, r.t, r.a, r.ta, r.pop, r.dep].iter().any(|v| !v.is_finite()) || r.inc <= 0. {
                    return Err("invalid market row".into());
                }
                let pl = &mut places[r.place];
                if pl.open && (pl.country != country_id(&r.country) || pl.road != r.road) {
                    return Err(format!("place {}: its pack and its row disagree (run pipeline/market_zones.py)", r.place));
                }
                pl.has = true;
                pl.lon = r.lon;
                pl.lat = r.lat;
                pl.u = unit(r.lon, r.lat);
                pl.country = country_id(&r.country);
                pl.inc = r.inc.min(1.);
                pl.road = r.road;
                pl.g = r.g;
                pl.t = r.t;
                pl.a = r.a;
                pl.ta = r.ta;
                pl.pop = r.pop;
                pl.dep = r.dep;
                pl.zone = r.zone.clone();
                pl.air = r.air.iter().filter(|a| a.0.is_finite() && a.0 >= 0.).cloned().collect();
                pl.park = if r.park.is_finite() { r.park.max(0.) } else { 0. };
                pl.fw = if r.fw.is_finite() && r.fw > 0. { r.fw } else { 0. };
                for &(j, t) in &r.near {
                    if (j as usize) < np && t.is_finite() && t >= 0. {
                        near.insert((r.place.min(j as usize) as u32, r.place.max(j as usize) as u32), t);
                    }
                }
                commute_rows[r.place] = r.commute.clone();
            }
        }
        None => {
            for z in &mz {
                let pl = &mut places[z.place];
                pl.has = true;
                pl.g += z.g;
                pl.t += z.t;
                pl.a += z.a;
                pl.ta += z.ta;
                pl.pop += z.pop;
            }
            for (p, pl) in places.iter_mut().enumerate() {
                if !pl.has {
                    continue;
                }
                let zs = &mz[if pl.zones > 0 { pack_first_zone[p] } else { 0 }..if pl.zones > 0 { pack_first_zone[p] + pl.zones as usize } else { 0 }];
                let w: f64 = zs.iter().map(|z| z.pop + z.w_attr).sum::<f64>().max(1e-12);
                if ground {
                    let mut u = [0f64; 3];
                    for z in zs {
                        for k in 0..3 {
                            u[k] += z.u[k] * (z.pop + z.w_attr) / w;
                        }
                    }
                    let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12);
                    pl.u = [u[0] / n, u[1] / n, u[2] / n];
                    (pl.lon, pl.lat) = lonlat_of(&pl.u);
                } else {
                    pl.x = (zs.iter().map(|z| z.x as f64 * (z.pop + z.w_attr)).sum::<f64>() / w) as f32;
                    pl.y = (zs.iter().map(|z| z.y as f64 * (z.pop + z.w_attr)).sum::<f64>() / w) as f32;
                }
            }
            // departures: each site's ground and tail rate
            for (s, &z) in cell_zone.iter().enumerate() {
                let (sg, st) = site_mult[s];
                let gr = &groups[match groups.binary_search_by(|g| g.lo.cmp(&s)) {
                    Ok(i) => i,
                    Err(i) => i - 1,
                }];
                let pop = gr.city.pop[s - gr.lo] as f64;
                places[mz[z].place].dep += pop * (rule.r * sg + rule.rt * st);
            }
        }
    }
    for pl in places.iter_mut() {
        if pl.has {
            if let Some(o) = &out_frame {
                let (x, y) = o.from_lonlat(pl.lon, pl.lat);
                pl.x = x as f32;
                pl.y = y as f32;
            }
        }
    }
    // ---- per pair of places with an open one (T-1204: every pair was 300 MB a worker)
    let mp = market_params();
    let mut rules = Rules { slot: vec![u32::MAX; np], ..Rules::default() };
    for (p, pl) in places.iter().enumerate() {
        if pl.open {
            rules.slot[p] = rules.open.len() as u32;
            rules.open.push(p);
        }
    }
    rules.out.reserve_exact(rules.open.len() * np);
    rules.inn.reserve_exact(rules.open.len() * np);
    for &i in &rules.open {
        rules.out.extend((0..np).map(|j| pair_rule(&places, &countries, &rule, ground, &mp, &drives, i, j)));
        rules.inn.extend((0..np).map(|j| pair_rule(&places, &countries, &rule, ground, &mp, &drives, j, i)));
    }
    let mut m = Market {
        rule,
        ground,
        world,
        from_rows,
        anchor: groups.first().and_then(|g| g.frame),
        places,
        countries,
        rules,
        mp,
        drives,
        near,
        zones: mz,
        n_gid,
        gid_place,
        comp_of_gid: vec![u32::MAX; n_gid],
        groups,
        cell_zone,
        site_key: HashMap::new(),
        commutes: vec![],
        c_home: (vec![], vec![]),
        c_work: (vec![], vec![]),
        everyday: vec![],
        e_home: (vec![], vec![]),
        e_act: (vec![], vec![]),
        e_base: vec![],
        cw: Rc::new(Corridors { id: 0, trips: vec![], commute: vec![], commuters: 0., leaving_by: vec![], daily: 0., leaving: 0., place_pairs: 0 }),
    };
    for (c, z) in m.zones.iter().enumerate() {
        m.comp_of_gid[z.gid as usize] = c as u32;
    }
    // ---- near totals added up from the zones (no rows: the pipeline, a closed world)
    if !from_rows {
        // near places' zone pairs only (T-929): each pair of places sums its zone pairs (a, b > a)
        // in the order a scan of every zone pair would, so the totals are the same to the bit
        let spans = m.place_spans();
        for (k, &(p, lo, hi)) in spans.iter().enumerate() {
            for &(q, qlo, qhi) in &spans[k..] {
                if !m.pr(p, q).near || near_for.is_some_and(|f| !f[p] && !f[q]) {
                    continue;
                }
                let mut sum = None;
                for a in lo..hi {
                    for b in qlo.max(a + 1)..qhi {
                        let t = m.eachway(&m.zones[a], &m.zones[b]);
                        if t >= m.rule.min_trips {
                            *sum.get_or_insert(0f64) += t;
                        }
                    }
                }
                if let Some(t) = sum {
                    m.near.insert((p as u32, q as u32), t);
                }
            }
        }
    }
    // ---- commutes
    let mut by_pair: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    if closed {
        for &(h, w, n) in &legacy_commutes {
            let site = |v: u32| pack_first_site[(v >> 24) as usize] + (v & 0xff_ffff) as usize;
            let (zh, zw) = (m.cell_zone[site(h)], m.cell_zone[site(w)]);
            if zh != zw {
                *by_pair.entry((zh as u32, m.zones[zw].gid)).or_default() += n as f64;
            }
        }
    } else {
        for zin in zones.iter().filter(|z| near_for.is_none_or(|f| f[z.place])) {
            let nc = zin.commute_trips.len();
            if zin.commute_home.len() != nc || zin.commute_place.len() != nc || zin.commute_work.len() != nc {
                return Err("commute arrays disagree".into());
            }
            let first = pack_first_zone[zin.place];
            for i in 0..nc {
                let (h, wp, w, n) = (zin.commute_home[i] as usize, zin.commute_place[i] as usize, zin.commute_work[i], zin.commute_trips[i]);
                if h >= zin.zones.len() || wp >= np || w >= m.places[wp].zones || !n.is_finite() || n < 0. {
                    return Err("invalid commute between zones".into());
                }
                let wg = m.places[wp].off + w;
                if m.zones[first + h].gid != wg {
                    *by_pair.entry(((first + h) as u32, wg)).or_default() += n as f64;
                }
            }
        }
    }
    m.commutes = by_pair.into_iter().map(|((h, w), n)| (h, w, n as f32)).collect();
    // T-1175: everyday outings across a zone line, from the open places' own files
    let mut by_kind: BTreeMap<(u32, u32, u8), (f64, f64)> = BTreeMap::new();
    for zin in zones.iter().filter(|z| !closed && near_for.is_none_or(|f| f[z.place])) {
        let ne = zin.everyday_trips.len();
        if [zin.everyday_home.len(), zin.everyday_place.len(), zin.everyday_zone.len(), zin.everyday_kind.len(), zin.everyday_walk.len()].iter().any(|&l| l != ne) {
            return Err("everyday arrays disagree".into());
        }
        let first = pack_first_zone[zin.place];
        for i in 0..ne {
            let (h, ap, a, k, n, w) = (zin.everyday_home[i] as usize, zin.everyday_place[i] as usize, zin.everyday_zone[i], zin.everyday_kind[i], zin.everyday_trips[i], zin.everyday_walk[i]);
            if h >= zin.zones.len() || ap >= np || a >= m.places[ap].zones || k as usize >= super::everyday::KIND_COUNT || !n.is_finite() || n < 0. || !(0.0..=1.0).contains(&w) {
                return Err("invalid everyday pair between zones".into());
            }
            let ag = m.places[ap].off + a;
            if m.zones[first + h].gid != ag {
                let e = by_kind.entry(((first + h) as u32, ag, k)).or_default();
                e.0 += n as f64;
                e.1 += (n * w) as f64;
            }
        }
    }
    m.everyday = by_kind.into_iter().map(|((h, a, kind), (n, w))| EPair { h, a, kind, trips: n as f32, walk: (w / n.max(1e-12)) as f32 }).collect();
    let csr = |key: &dyn Fn(&(u32, u32, f32)) -> Option<u32>, n: usize, list: &[(u32, u32, f32)]| {
        let mut off = vec![0u32; n + 1];
        for c in list {
            if let Some(k) = key(c) {
                off[k as usize + 1] += 1;
            }
        }
        for z in 0..n {
            off[z + 1] += off[z];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; off[n] as usize];
        for (i, c) in list.iter().enumerate() {
            if let Some(k) = key(c) {
                items[fill[k as usize] as usize] = i as u32;
                fill[k as usize] += 1;
            }
        }
        (off, items)
    };
    let nz = m.zones.len();
    m.c_home = csr(&|c| Some(c.0), nz, &m.commutes);
    let cog = m.comp_of_gid.clone();
    m.c_work = csr(&|c| Some(cog[c.1 as usize]).filter(|&w| w != u32::MAX), nz, &m.commutes);
    // T-1229: the everyday pairs by home zone and by activity zone, and per home place and kind
    let csr_e = |key: &dyn Fn(&EPair) -> Option<u32>| {
        let mut off = vec![0u32; nz + 1];
        for e in &m.everyday {
            if let Some(k) = key(e) {
                off[k as usize + 1] += 1;
            }
        }
        for z in 0..nz {
            off[z + 1] += off[z];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; off[nz] as usize];
        for (i, e) in m.everyday.iter().enumerate() {
            if let Some(k) = key(e) {
                items[fill[k as usize] as usize] = i as u32;
                fill[k as usize] += 1;
            }
        }
        (off, items)
    };
    let (e_home, e_act) = (csr_e(&|e| Some(e.h)), csr_e(&|e| Some(cog[e.a as usize]).filter(|&w| w != u32::MAX)));
    m.e_home = e_home;
    m.e_act = e_act;
    m.e_base = vec![(0., 0.); np * KIND_COUNT];
    for e in &m.everyday {
        let b = &mut m.e_base[m.zones[e.h as usize].place * KIND_COUNT + e.kind as usize];
        b.0 += e.trips as f64;
        b.1 += (e.trips * e.walk) as f64;
    }
    // ---- the corridors' figures: with rows the same for every composition, so kept (T-1204)
    m.cw = match reuse {
        Some(c) if from_rows => c,
        _ => Rc::new(m.corridors_of(from_rows.then_some(&commute_rows[..]))),
    };
    // ---- each 600 m square's site
    let mut site_key = HashMap::new();
    for s in 0..m.n_sites() {
        let (g, i) = m.site(s);
        let key = ((g.city.x[i] / SITE_M).floor() as i32, (g.city.y[i] / SITE_M).floor() as i32, m.zones[m.cell_zone[s]].place);
        site_key.entry(key).or_insert(s);
    }
    m.site_key = site_key;
    Ok(m)
}
/// T-377: the packs of the places a worker has open, parsed once, and the world they belong to;
/// `market` composes them into a market (the worker does so whenever a pack opens or closes).
#[wasm_bindgen]
pub struct MarketPacks {
    world: WorldIn,
    zones: BTreeMap<usize, Rc<ZoneIn>>,
    /// T-1204, T-1220: the corridors' figures, worked out at the first composition and kept for
    /// the others (with rows they depend on the places alone)
    corridors: RefCell<Option<Rc<Corridors>>>,
    drives: Rc<Drives>,
}
#[wasm_bindgen]
impl MarketPacks {
    /// `world`: the registry's market frame, rule, place count and rows.
    #[wasm_bindgen(constructor)]
    pub fn new(world: &str) -> Result<MarketPacks, String> {
        let world: WorldIn = serde_json::from_str(world).map_err(|e| e.to_string())?;
        if !world.rule.ok() {
            return Err("invalid market rule".into());
        }
        make_frame(&world.frame)?;
        Ok(MarketPacks { world, zones: BTreeMap::new(), corridors: RefCell::new(None), drives: Rc::default() })
    }
    /// T-279: the routed drives between places (`market-drives.bin`, pipeline/market_drives.py),
    /// for the compositions after; the pairs placed. Without them every drive is the rule's.
    pub fn set_drives(&mut self, bytes: &[u8]) -> Result<u32, String> {
        let rows = self.world.rows.as_deref().unwrap_or(&[]);
        let place_of: HashMap<&str, usize> = rows.iter().map(|r| (r.zone.as_str(), r.place)).collect();
        let d = Drives::decode(bytes, &place_of, self.world.places)?;
        let n = d.len() as u32;
        self.drives = Rc::new(d);
        Ok(n)
    }
    /// Add (or replace) a place's pack; its place.
    pub fn add(&mut self, pack: &str) -> Result<u32, String> {
        let z: ZoneIn = serde_json::from_str(pack).map_err(|e| e.to_string())?;
        if z.place >= self.world.places {
            return Err("market pack's place out of range".into());
        }
        let p = z.place;
        self.zones.insert(p, Rc::new(z));
        Ok(p as u32)
    }
    pub fn remove(&mut self, place: u32) -> bool {
        self.zones.remove(&(place as usize)).is_some()
    }
    pub fn places(&self) -> Vec<u32> {
        self.zones.keys().map(|&p| p as u32).collect()
    }
    /// T-1174: drop the corridors kept from the first composition, so the next composition works
    /// them out from the packs held now, as a worker that opened them all at once does (they differ
    /// with the packs open at the first composition: T-1354).
    pub fn forget_corridors(&self) {
        *self.corridors.borrow_mut() = None;
    }
    /// The market of every pack held, with no network yet.
    pub fn market(&self) -> Result<IntercityApi, String> {
        let w = self.world.clone();
        let rows = w.rows.is_some();
        let b = Built { zones: self.zones.values().cloned().collect(), rule: w.rule, out: Some(w.frame), places: w.places, rows: w.rows, world: true, closed: false, legacy_commutes: vec![], drives: self.drives.clone() };
        let m = build_for(b, None, self.corridors.borrow().clone())?;
        if rows {
            *self.corridors.borrow_mut() = Some(m.cw.clone());
        }
        Ok(IntercityApi::on(Rc::new(m)))
    }
}
fn parse(header: &str) -> Result<Market, String> {
    build(parse_built(header)?)
}
fn parse_built(header: &str) -> Result<Built, String> {
    // a composition as bin/market_pack.rs writes it, straight into its struct (no Value between)
    if header.starts_with("{\"world\":") {
        return compose(serde_json::from_str(header).map_err(|e| e.to_string())?);
    }
    let v: serde_json::Value = serde_json::from_str(header).map_err(|e| e.to_string())?;
    if v.get("sites").is_some() {
        single_pack(serde_json::from_value(v).map_err(|e| e.to_string())?)
    } else {
        compose(serde_json::from_value(v).map_err(|e| e.to_string())?)
    }
}
/// T-276: the market's zones as the game built them from an old single pack's sites, for the
/// joint commute gravity (`bin/joint_gravity.rs`): each site's zone, the first site of each zone,
/// and each site's 600 m square key (x, y / 600, floored; the market).
pub fn market_site_zones(header: &str) -> Result<(Vec<u32>, Vec<u32>, Vec<(i32, i32, usize)>), String> {
    let pack: LegacyPack = serde_json::from_str(header).map_err(|e| e.to_string())?;
    let zm = pack.zone_m as f32;
    let mut keys = BTreeMap::new();
    let (mut site_zone, mut first, mut sq) = (vec![], vec![], vec![]);
    for (s, nums) in pack.sites.iter().enumerate() {
        let (x, y, m) = (nums[0] as f32, nums[1] as f32, nums[4] as usize);
        let n = keys.len();
        let z = *keys.entry(((x / zm).floor() as i32, (y / zm).floor() as i32, m)).or_insert(n);
        if z == first.len() {
            first.push(s as u32);
        }
        site_zone.push(z as u32);
        sq.push(((x / SITE_M).floor() as i32, (y / SITE_M).floor() as i32, m));
    }
    Ok((site_zone, first, sq))
}
/// T-377: the registry rows of every place a composition holds (the world given without rows):
/// what `bin/market_pack.rs` writes into the registry.
pub fn world_rows(header: &str) -> Result<Vec<Row>, String> {
    world_rows_for(header, None)
}
/// T-929: `world_rows` of the places in `only` (by place) alone. Each is the same to the bit as in
/// a composition of every place, given the composition holds every place under `near_km` of it and
/// every place it commutes to (a missing commute place is an error; a missing near one is not).
pub fn world_rows_for(header: &str, only: Option<&[bool]>) -> Result<Vec<Row>, String> {
    let b = parse_built(header)?;
    if only.is_some_and(|f| f.len() != b.places) {
        return Err("rows wanted for a place list of another length".into());
    }
    let m = build_for(b, only, None)?;
    if m.from_rows {
        return Err("the composition carries rows already".into());
    }
    let np = m.places.len();
    let mut commutes: Vec<BTreeMap<u32, f64>> = vec![BTreeMap::new(); np];
    for &(h, w, n) in &m.commutes {
        *commutes[m.zones[h as usize].place].entry(m.gid_place[w as usize]).or_default() += n as f64;
    }
    let mut out = vec![];
    for (p, commute) in commutes.into_iter().enumerate() {
        let pl = &m.places[p];
        if !pl.open || only.is_some_and(|f| !f[p]) {
            continue;
        }
        let near: Vec<(u32, f64)> = (0..np).filter(|&j| m.places[j].has && m.pr(p, j).near).filter_map(|j| m.near.get(&(p.min(j) as u32, p.max(j) as u32)).map(|&t| (j as u32, t))).collect();
        out.push(Row {
            place: p,
            zone: pl.zone.clone(),
            lon: pl.lon,
            lat: pl.lat,
            country: m.countries[pl.country as usize].clone(),
            inc: pl.inc,
            road: pl.road,
            g: pl.g,
            t: pl.t,
            a: pl.a,
            ta: pl.ta,
            pop: pl.pop,
            dep: pl.dep,
            zones: pl.zones,
            near,
            commute: commute.into_iter().collect(),
            air: pl.air.clone(),
            park: pl.park,
            fw: 0.,
        });
    }
    Ok(out)
}
/// T-377: trips a day each way between two places of a composition (rows given or not), and
/// that pair's trips over `min_km` (zone centre to zone centre; T-299's comparison), both from
/// the zones when both places are open.
pub fn place_trips(header: &str, pairs: &[(usize, usize)], min_km: f64) -> Result<Vec<(f64, f64)>, String> {
    Ok(parse(header)?.place_trips(pairs, min_km))
}
impl Market {
    fn place_trips(&self, pairs: &[(usize, usize)], min_km: f64) -> Vec<(f64, f64)> {
        let spans = self.place_spans();
        let span = |p: usize| spans.iter().find(|s| s.0 == p).map_or(0..0, |s| s.1..s.2);
        pairs
            .iter()
            .map(|&(i, j)| {
                let all = self.place_eachway(i, j);
                let mut over = 0.;
                if self.places[i].open && self.places[j].open {
                    for a in span(i) {
                        for b in span(j) {
                            let (za, zb) = (&self.zones[a], &self.zones[b]);
                            if (i == j && b <= a) || self.zone_km(za, zb) < min_km {
                                continue;
                            }
                            let t = self.eachway(za, zb);
                            if t >= self.rule.min_trips {
                                over += t;
                            }
                        }
                    }
                }
                (all, over)
            })
            .collect()
    }
}
impl IntercityApi {
    /// `place_trips` on this market, with no second parse (`bin/market_pack.rs`).
    pub fn place_trips(&self, pairs: &[(usize, usize)], min_km: f64) -> Vec<(f64, f64)> {
        self.m.place_trips(pairs, min_km)
    }
    /// Number of places in selection and station flow headers.
    pub fn market_count(&self) -> usize {
        self.m.places.len()
    }
    /// T-377: the 6 km zones of the whole world (every place's, open or not): the length of the
    /// zone blocks of every answer.
    pub fn zone_count(&self) -> usize {
        self.m.n_gid
    }
    /// The places this market holds the packs of (ascending).
    pub fn open_places(&self) -> Vec<u32> {
        (0..self.m.places.len()).filter(|&p| self.m.places[p].open).map(|p| p as u32).collect()
    }
    /// The zone pairs the network's stations serve, and the place pairs with trips.
    pub fn pair_counts(&self) -> Vec<f64> {
        vec![self.pairs.len() as f64, self.m.cw.place_pairs as f64]
    }
    /// The network with stations in the pack's plane (an old single pack).
    pub fn set_network(&mut self, st_xy: &[f32], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        self.set_network_in(st_xy, line_n, stops, times, tph, cars)
    }
    /// T-276: per corridor, the commutes between zones in period `q`: legs and rail legs.
    pub fn commute_corridors(&self, q: usize) -> Vec<f64> {
        let legs = 0.5 * (TO_WORK[q] + TO_HOME[q]) as f64;
        let m = &self.m;
        self.per(q).filter(|_| m.cw.commuters > 0.).map_or(vec![], |s| {
            let mut out = vec![0f64; 2 * m.n_corridors()];
            for &(c, n) in &m.cw.commute {
                out[2 * c as usize] = n * legs;
            }
            for (k, c) in m.open_corridors() {
                out[2 * c + 1] = s.c_corridors[k];
            }
            out
        })
    }
    /// As `DemandApi::transfers` (T-176): this market's transfers over the periods held.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    /// Where the longer trips of a selection of local cells go and come from (T-118).
    ///
    /// The selection is points (`xy`: x, y per 600 m square, metres in that place's pack frame)
    /// of one `market` (place) with the `share` (0..1) of that square's people, or jobs, that are
    /// selected. A site's share of its zone's departures is its people over the zone's, of its
    /// arrivals its attraction over the zone's.
    ///
    /// Returns, over the periods solved: the matched share of the points, then 8 numbers per
    /// place at the far end (trips starting in the selection that end there: all modes, by rail,
    /// all-mode trips times far x and y; then the same for trips ending in the selection), then
    /// per global 6 km zone (`zone_count`) all and rail of starts and of ends, every zone's x, y
    /// and place, then the commutes' tail (4 per place, 4 per zone), then (T-378) the trips with
    /// places that have no zone, starting and ending. Places with no pack open here are counted by
    /// place only.
    pub fn selection_flows(&self, market: u32, xy: &[f32], share: &[f32]) -> Vec<f64> {
        self.selection_flows_day(market, xy, share, WEEKDAY as u32)
    }
    /// A station's longer-trip rail riders over the periods solved (T-118): per place, riders
    /// starting here bound for it, riders ending here from it, and those counts times the far
    /// zone's x and y (starts x, starts y, ends x, ends y); then per global 6 km zone the riders
    /// starting, then ending, then every zone's x, y and place. Empty with no such station.
    pub fn station_flows(&self, station: u32) -> Vec<f64> {
        self.station_flows_day(station, WEEKDAY as u32)
    }
}

/// T-373: the 2 km zones of a group's sites (`city`, whose `jobs` are attractions) with their
/// homes and jobs as the local layer's zones have them, and each zone's centre.
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
/// `share`: its walks as the mean of their exp(-beta x cost), nested with its rides at the
/// station choice's scale `theta`.
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
/// Each station's component of the graph (edges taken both ways), named by its lowest station.
fn station_components(g: &Graph) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..g.n as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for a in 0..g.n {
        for e in g.adj_off[a] as usize..g.adj_off[a + 1] as usize {
            let (ra, rb) = (find(&mut parent, a as u32), find(&mut parent, g.adj_to[e]));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    (0..g.n_st as u32).map(|s| find(&mut parent, s)).collect()
}

/// T-409: the pack's own data is shared (`Rc`) by every market `beside` it; the rest is one
/// network's.
#[wasm_bindgen]
pub struct IntercityApi {
    m: Rc<Market>,
    /// The zone pairs between zones some station serves (T-174), two by two: (lower zone, higher
    /// zone), then the same reversed, in the order of (lower, higher) (`pair_at` searches it).
    pairs: Vec<Pair>,
    /// The zones `pairs` was built for (ascending), and each zone's place in it (`u32::MAX` if not).
    served: Vec<usize>,
    served_at: Vec<u32>,
    base: Params,
    svc: Option<Service>,
    access: Vec<Vec<(usize, f32)>>,
    access_work: Vec<Vec<(usize, f32)>>,
    access_c: Vec<Vec<(usize, f32)>>,
    access_cw: Vec<Vec<(usize, f32)>>,
    access_c_walk: Vec<Vec<(usize, f32)>>,
    cpairs: Vec<CPair>,
    cpair_of: Vec<u32>,
    /// T-1175: the activity end's walks of the everyday pairs (attraction-weighted), and the
    /// everyday pairs some station serves
    access_ew: Vec<Vec<(usize, f32)>>,
    epairs: Vec<EPairOn>,
    /// the index in `epairs` of each of the market's everyday pairs, `u32::MAX` if none
    epair_of: Vec<u32>,
    want: Vec<bool>,
    per: Vec<Option<Period>>,
    fare_km: Vec<f32>,
    fare_per_km: f32,
    /// `DemandApi::set_fare_curve`'s knots (T-883)
    fare_curve: Vec<f32>,
    /// T-1286: per zone, its country's fare factor less 1 (`set_fare_factors`); empty while every
    /// factor is 1.
    zone_fx: Vec<f32>,
    complex_in: Vec<u32>,
    /// `DemandApi::set_upgrades` (T-1281).
    ups_in: Vec<u8>,
    /// The passages of the next `set_network` (`set_passages`, T-1281).
    passages_in: Vec<u32>,
    /// `DemandApi::set_span` (T-842).
    span_in: Vec<f32>,
    /// T-409: per served zone, the track components its stations lie in.
    served_comp: Vec<Vec<u32>>,
    st_pat: StPattern,
    /// T-780: trips a day each way a sampled zone pair carries (`solved_pairs`); 0 solves every pair.
    chunk: f64,
}

/// T-761: the entries of the per-station far-zone table (`Period::st`) a pass can make non-zero,
/// row by station (CSR): row s's columns ascending in `col[off[s]..off[s + 1]]`, a column being a
/// served zone's place in `served` for trips starting at s, or that plus the served zones' count
/// for trips ending there. A station's riders go to and come from the zones its own zones trade
/// with (about a tenth of all on Japan's whole network), so the table holds those alone.
#[derive(Default)]
struct StPattern {
    off: Vec<u32>,
    col: Cols,
}

/// Columns, as u16 while every one fits (Japan's whole network: 18,228 columns).
enum Cols {
    Short(Vec<u16>),
    Long(Vec<u32>),
}
impl Default for Cols {
    fn default() -> Self {
        Cols::Short(vec![])
    }
}
impl Cols {
    fn len(&self) -> usize {
        match self {
            Cols::Short(v) => v.len(),
            Cols::Long(v) => v.len(),
        }
    }
    fn get(&self, i: usize) -> usize {
        match self {
            Cols::Short(v) => v[i] as usize,
            Cols::Long(v) => v[i] as usize,
        }
    }
}

impl StPattern {
    /// Where entry (station `s`, column `c`) sits, which must be one of the pattern's, searched
    /// from `cur` (a place in the row) outwards and left there: `demand` looks a row up in runs of
    /// one column, or of columns rising, so most lookups take a step or two.
    #[inline(always)]
    fn at(&self, s: usize, c: usize, cur: &mut u32) -> usize {
        let (lo, hi) = (self.off[s] as usize, self.off[s + 1] as usize);
        lo + match &self.col {
            Cols::Short(v) => find_from(&v[lo..hi], c as u16, cur),
            Cols::Long(v) => find_from(&v[lo..hi], c as u32, cur),
        }
    }
    fn row(&self, s: usize) -> std::ops::Range<usize> {
        self.off[s] as usize..self.off[s + 1] as usize
    }
}

/// `StPattern::at` in one row.
#[inline(always)]
fn find_from<T: Copy + Ord>(row: &[T], c: T, cur: &mut u32) -> usize {
    {
        let i = (*cur as usize).min(row.len() - 1);
        let k = if row[i] == c {
            i
        } else if row[i] < c {
            let (mut a, mut step, mut b) = (i, 1, i + 1);
            while b < row.len() && row[b] < c {
                a = b;
                step *= 2;
                b = a + step;
            }
            a + 1 + row[a + 1..(b + 1).min(row.len())].partition_point(|&x| x < c)
        } else {
            let (mut b, mut step) = (i, 1);
            let a = loop {
                if b < step {
                    break 0;
                }
                let p = b - step;
                if row[p] <= c {
                    break p;
                }
                b = p;
                step *= 2;
            };
            a + row[a..b].partition_point(|&x| x < c)
        };
        assert!(row.get(k) == Some(&c), "a far-zone entry outside the pattern");
        *cur = k as u32;
        k
    }
}

/// Which pairs of the `served` zones (ascending) are solved, a bit each for (i, k > i) in that
/// order, and their trips a day each way, counted before `set_served` makes their table so it is
/// made once at its size (T-761: grown by doubling it peaked at twice that, 500 MB on Japan).
/// A pair needs a car or a plane to compete with (T-377, T-378: else its trips stay fixed), track
/// joining its zones, and `MIN_TRIPS`.
///
/// With `chunk` (T-780) each zone's pairs with the zones of one place after it are dealt that
/// place's trips in chunks of about `chunk`: systematic sampling over the place's zones in their
/// order, by trips, from a start fixed by the two zones' ids, so the same zones give the same pairs
/// on any network and each zone's trips with each later place are kept exactly. A pair of at least
/// a chunk is always solved; smaller ones are solved for a chunk or not at all. The chunks are dealt
/// over every zone, served or not, so a station serving another zone adds pairs without dealing
/// the rest again.
fn solved_pairs(m: &Market, served: &[usize], joined: &impl Fn(usize, usize) -> bool, chunk: f64) -> (Vec<u64>, Vec<f64>) {
    let min = m.rule.min_trips;
    let n = served.len();
    let mut any = vec![0u64; (n as u64 * n.saturating_sub(1) as u64 / 2).div_ceil(64) as usize];
    let mut each = vec![];
    let row = |i: usize| i * n.saturating_sub(1) - i * i.saturating_sub(1) / 2;
    let mut keep = |any: &mut [u64], i: usize, k: usize, trips: f64| {
        let bit = row(i) + k - i - 1;
        any[bit / 64] |= 1 << (bit % 64);
        each.push(trips);
    };
    if chunk <= 0. {
        for (i, &a) in served.iter().enumerate() {
            let za = &m.zones[a];
            for (k, &b) in served.iter().enumerate().skip(i + 1) {
                let zb = &m.zones[b];
                if !m.pr(za.place, zb.place).has_alt() || !joined(i, k) {
                    continue;
                }
                let trips = m.eachway_with(za, zb, m.curves(za, zb));
                if trips >= min {
                    keep(&mut any, i, k, trips);
                }
            }
        }
        return (any, each);
    }
    let mut at = vec![u32::MAX; m.zones.len()];
    let mut some = vec![false; m.places.len()];
    for (k, &z) in served.iter().enumerate() {
        at[z] = k as u32;
        some[m.zones[z].place] = true;
    }
    let mut run: Vec<(usize, f64)> = vec![];
    for (i, &a) in served.iter().enumerate() {
        let za = &m.zones[a];
        let mut b = a + 1;
        while b < m.zones.len() {
            // one place's zones after a: a run of the zones' order
            let place = m.zones[b].place;
            run.clear();
            if !some[place] || !m.pr(za.place, place).has_alt() {
                while b < m.zones.len() && m.zones[b].place == place {
                    b += 1;
                }
                continue;
            }
            // far places' zones are all at the places' distance
            let fixed = (!m.pr(za.place, place).near).then(|| m.curves(za, &m.zones[b]));
            while b < m.zones.len() && m.zones[b].place == place {
                let zb = &m.zones[b];
                let t = m.eachway_with(za, zb, fixed.unwrap_or_else(|| m.curves(za, zb)));
                if t >= min {
                    run.push((b, t));
                }
                b += 1;
            }
            let Some(&(first, _)) = run.first() else { continue };
            let total: f64 = run.iter().map(|e| e.1).sum();
            let draws = (total / chunk).ceil();
            let mut solve = |zb: usize, trips: f64| {
                let k = at[zb];
                if k != u32::MAX && joined(i, k as usize) {
                    keep(&mut any, i, k as usize, trips);
                }
            };
            if draws >= run.len() as f64 {
                for &(zb, t) in &run {
                    solve(zb, t);
                }
                continue;
            }
            let step = total / draws;
            let u = unit_hash(za.gid, m.zones[first].gid);
            // draws before trips x: those at (j + u) x step below x
            let before = |x: f64| (x / step - u).ceil().clamp(0., draws);
            let mut lo = 0.;
            for &(zb, t) in &run {
                let hi = lo + t;
                let got = before(hi) - before(lo);
                lo = hi;
                if got > 0. {
                    solve(zb, got * step);
                }
            }
        }
    }
    (any, each)
}
/// A number in [0, 1) fixed by two ids (splitmix64).
pub(crate) fn unit_hash(a: u32, b: u32) -> f64 {
    let mut z = ((a as u64) << 32 | b as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

impl IntercityApi {
    fn on(m: Rc<Market>) -> Self {
        IntercityApi {
            m,
            pairs: vec![],
            served: vec![],
            served_at: vec![],
            base: market_params(),
            svc: None,
            access: vec![],
            access_work: vec![],
            access_c: vec![],
            access_cw: vec![],
            access_c_walk: vec![],
            cpairs: vec![],
            cpair_of: vec![],
            access_ew: vec![],
            epairs: vec![],
            epair_of: vec![],
            want: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            fare_km: vec![],
            fare_per_km: 0.,
            fare_curve: vec![],
            zone_fx: vec![],
            complex_in: vec![],
            ups_in: vec![],
            passages_in: vec![],
            span_in: vec![],
            chunk: 0.,
            served_comp: vec![],
            st_pat: StPattern::default(),
        }
    }
    fn pass(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> Pass {
        let todo = self.want.iter().filter(|&&w| w).count();
        if todo * strategy_bytes(g) <= MARKET_STRATEGY_BYTES || skims_cached(g, p) {
            // the strategies are shared with the local populations on the same service (T-119)
            return with_skims(g, p, &self.want, |sk| self.pass_on(g, p, q, sk, ix));
        }
        self.pass_streamed(g, p, q, ix, MARKET_STRATEGY_BYTES)
    }
    fn pass_on(&self, g: &Graph, p: &Params, q: usize, sk: &Skims, ix: &XferIndex) -> Pass {
        let s = g.n_st;
        let km = self.pair_km(g, &self.want, |dst| (sk.slot[dst] != u32::MAX).then(|| sk.strategy(sk.slot[dst] as usize)));
        let d = self.demand(g, &sk.r, p, q, &km);
        self.assigned(g, ix, d, |dst| {
            let slot = sk.slot[dst];
            debug_assert!(slot != u32::MAX, "flow towards station {dst}, which was not searched");
            (slot != u32::MAX).then(|| Held::Ref(sk.strategy(slot as usize)))
        }, s)
    }
    /// The pass's loads from its trips: each destination's column of the longer trips and the
    /// commutes together (`assign_with_transfers`'s), and of the commutes alone, assigned in station
    /// order along the destination's strategy (`strategy`; `None`: nobody can ride there).
    fn assigned<'a>(&self, g: &Graph, ix: &XferIndex, d: Demand, mut strategy: impl FnMut(usize) -> Option<Held<'a>>, s: usize) -> Pass {
        let Demand { od, od_c, st, rail, cor, pair_rail, c_rail, c_frac, c_cor, e, rail_d } = d;
        let counting = !(cfg!(test) && XFER_OFF.load(std::sync::atomic::Ordering::Relaxed));
        let (mut all, mut com) = (DestAssign::new(g, 0, counting), DestAssign::new(g, 0, false));
        com.fares = false;
        // T-1175: the everyday pairs' loads, a purpose at a time
        let mut ev: Vec<Option<DestAssign>> = e
            .od
            .iter()
            .map(|o| {
                (!o.src.is_empty()).then(|| {
                    let mut a = DestAssign::new(g, 0, false);
                    a.fares = false;
                    a
                })
            })
            .collect();
        let mut took = vec![false; ev.len()];
        let mut xfer = vec![0.0; ix.len];
        // a destination's commutes as a column (the riders from each station), and its everyday
        // pairs' (every purpose's together, then each purpose's)
        let mut col = vec![0f32; s];
        let (mut ecol, mut kcol) = (vec![0f32; s], vec![0f32; s]);
        for dst in 0..s {
            let ks = od_c.off[dst] as usize..od_c.off[dst + 1] as usize;
            for k in ks.clone() {
                col[od_c.src[k] as usize] = od_c.v[k];
            }
            for o in &e.od {
                for k in o.off[dst] as usize..o.off[dst + 1] as usize {
                    ecol[o.src[k] as usize] += o.v[k];
                }
            }
            let c = com.take(dst, |src| col[src]);
            let a = all.take(dst, |src| od[src * s + dst] + col[src] + ecol[src]);
            for k in ks {
                col[od_c.src[k] as usize] = 0.;
            }
            for (i, o) in e.od.iter().enumerate() {
                took[i] = false;
                let Some(ea) = ev[i].as_mut() else { continue };
                let r = o.off[dst] as usize..o.off[dst + 1] as usize;
                if r.is_empty() {
                    continue;
                }
                for k in r.clone() {
                    kcol[o.src[k] as usize] = o.v[k];
                }
                took[i] = ea.take(dst, |src| kcol[src]);
                for k in r {
                    kcol[o.src[k] as usize] = 0.;
                    ecol[o.src[k] as usize] = 0.;
                }
            }
            if !(a || c || took.contains(&true)) {
                continue;
            }
            let Some(h) = strategy(dst) else { continue };
            let st = h.view();
            if c {
                com.walk(g, dst, st, None);
            }
            for (i, ea) in ev.iter_mut().enumerate() {
                if let (true, Some(ea)) = (took[i], ea.as_mut()) {
                    ea.walk(g, dst, st, None);
                }
            }
            if a {
                all.walk(g, dst, st, counting.then_some((ix, &mut xfer[..])));
            }
        }
        let e_loads = ev.into_iter().map(|a| a.map(|a| a.l)).collect();
        Pass { loads: all.l, c_loads: com.l, rail, c_rail, c_frac, cor, c_cor, pair_rail, st, xfer, e_loads, e_rail: e.rail, e_walk_lost: e.walk_lost, e_frac: e.frac, rail_d }
    }
    /// `pass_on` with the destinations searched one at a time (T-761), so no table of every
    /// destination's strategy is held: a first sweep keeps each one's route costs (the logsums need
    /// every station pair's) and the first `keep_bytes` of strategies; the second assigns each
    /// destination's column of both ODs, searching it again unless kept. The same figures to the bit.
    fn pass_streamed(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex, keep_bytes: usize) -> Pass {
        // the market's part solves last in its cluster, so the tables the cities searched for this
        // period and round have no further use, and a search for the market shares none of them
        clear_skim_cache();
        let s = g.n_st;
        let mut r = vec![f32::INFINITY; s * s];
        let mut search = DestSearch::new(g, p);
        let mut kept: Vec<Option<DestStrategy>> = (0..s).map(|_| None).collect();
        let mut held = 0;
        let mut km = if self.zone_fx.is_empty() { vec![] } else { vec![0f32; s * s] };
        let (mut col, mut dep) = (vec![], vec![]);
        for dst in (0..s).filter(|&d| self.want[d]) {
            let d = search.run(dst, Some(&mut r));
            if !km.is_empty() {
                self.put_km(g, d.view(), dst, &mut col, &mut dep, &mut km);
            }
            if held + d.bytes() <= keep_bytes {
                held += d.bytes();
                kept[dst] = Some(d);
            }
        }
        let d = self.demand(g, &r, p, q, &km);
        drop(r);
        drop(km);
        self.assigned(g, ix, d, |dst| {
            debug_assert!(self.want[dst], "flow towards station {dst}, which was not searched");
            self.want[dst].then(|| Held::Own(kept[dst].take().unwrap_or_else(|| search.run(dst, None))))
        }, s)
    }
    /// T-1286: each station pair's expected fare km (`FareX::km`) on the strategies towards the
    /// stations `want`ed that `strategy` has; empty while every fare factor is 1.
    fn pair_km<'s>(&self, g: &Graph, want: &[bool], strategy: impl Fn(usize) -> Option<Strategy<'s>>) -> Vec<f32> {
        if self.zone_fx.is_empty() {
            return vec![];
        }
        let s = g.n_st;
        let mut km = vec![0f32; s * s];
        let (mut col, mut dep) = (vec![], vec![]);
        for dst in (0..s).filter(|&d| want[d]) {
            if let Some(st) = strategy(dst) {
                self.put_km(g, st, dst, &mut col, &mut dep, &mut km);
            }
        }
        km
    }
    /// The fares' weighing (`FareX`) on `km` (`pair_km` on `g`), none while it is empty.
    fn fare_x<'x>(&self, g: &'x Graph, p: &Params, km: &'x [f32]) -> Option<FareX<'x>> {
        (!km.is_empty()).then(|| FareX { km, base: p.fare_min, per_km: self.fare_per_km, curve: &g.fare_curve })
    }
    /// What `rail_logsum_x` takes for trips from `zone`: none where its factor is 1.
    fn fx_of<'x>(&self, fx: &'x Option<FareX<'x>>, zone: usize) -> Option<(&'x FareX<'x>, f32)> {
        fx.as_ref().and_then(|x| self.zone_fx.get(zone).filter(|&&d| d != 0.).map(|&d| (x, d)))
    }
    /// `pair_km`'s column towards `dst`.
    fn put_km(&self, g: &Graph, st: Strategy, dst: usize, col: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>, km: &mut [f32]) {
        let s = g.n_st;
        trip_km_on(g, st, dst, &self.fare_km, col, dep);
        for src in 0..s {
            km[src * s + dst] = col[src];
        }
    }
    /// The market's rail trips in period `q` from the route costs `r` (S x S, `Skims::r`): the
    /// station to station trips of the longer trips and of the commutes, and the pair, corridor and
    /// station figures. `km`: `pair_km` on `g`.
    fn demand(&self, g: &Graph, r: &[f32], p: &Params, q: usize, km: &[f32]) -> Demand {
        let s = g.n_st;
        let fx = self.fare_x(g, p, km);
        let fx_of = |zone: usize| self.fx_of(&fx, zone);
        // T-1300: the riders' fare sensitivity: each trip's rail share with its fare dearer by
        // `FARE_D_MIN` perceived minutes of the market's (times its fare factor), less its share
        let mut rail_d = FareSense::default();
        let zkm = |a: usize, b: usize| self.m.zone_km(&self.m.zones[a], &self.m.zones[b]);
        let add = |s: &mut FareSense, dd: f64, rail: f64, km: f64| {
            s.d += dd / FARE_D_MIN as f64;
            s.d_km += dd / FARE_D_MIN as f64 * km;
            s.km += rail * km;
        };
        let step = |zone: usize| FARE_D_MIN * (1. + self.zone_fx.get(zone).copied().unwrap_or(0.));
        let mut od = vec![0.; s * s];
        let mut rail = 0.;
        let mut cor = vec![0.; self.m.n_open_corridors()];
        let mut pair_rail = vec![0f32; self.pairs.len()];
        let nz = self.served.len();
        let col = &self.served_at;
        let pat = &self.st_pat;
        let mut st = vec![0f32; pat.col.len()];
        // per station, where its row was last looked up, for starts and for ends
        let mut cur = vec![0u32; 2 * s];
        let mut put = |st: &mut [f32], x: usize, c: usize, v: f32| st[pat.at(x, c, &mut cur[2 * x + (c >= nz) as usize])] += v;
        let mut terms: Vec<(u32, u32, f64)> = vec![];
        for (pi, pair) in self.pairs.iter().enumerate() {
            let trips = pair.trips * SHARES[q] as f64;
            let (a, b) = (&self.access[pair.a as usize], &self.access_work[pair.b as usize]);
            let (rail_c, sum) = rail_logsum_x(r, s, p.fare_min, a, b, p.beta, p.soften(p.band_unsoftened(pair.band as usize), pair.car), fx_of(pair.a as usize), &mut terms);
            let bt = p.mode_beta(pair.car);
            let v_rail = if sum > 0. { ((-bt * (rail_c - pair.drive)).clamp(-80., 80.) as f64).exp() } else { 0. };
            let fraction = v_rail / (pair.rest as f64 + v_rail);
            let r = trips * fraction;
            rail += r;
            if sum > 0. {
                let v = ((-bt * (rail_c + step(pair.a as usize) - pair.drive)).clamp(-80., 80.) as f64).exp();
                add(&mut rail_d, trips * (v / (pair.rest as f64 + v) - fraction), r, zkm(pair.a as usize, pair.b as usize));
            }
            cor[pair.corridor as usize] += r;
            pair_rail[pi] = r as f32;
            if sum > 0. {
                let (ca, cb) = (nz + col[pair.a as usize] as usize, col[pair.b as usize] as usize);
                for &(sa, sb, v) in &terms {
                    let (sa, sb) = (sa as usize, sb as usize);
                    let rr = (r * v / sum) as f32;
                    od[sa * s + sb] += rr;
                    put(&mut st, sa, cb, rr);
                    put(&mut st, sb, ca, rr);
                }
            }
        }
        // T-276: commutes between zones, chosen as the local layer chooses: once per commuter on
        // this period's service from home to work, the trip home on the same station pair reversed
        let pc = commute_params();
        // (destination, origin, riders) as they come
        let mut od_c: Vec<(u32, u32, f32)> = vec![];
        let (mut c_rail, mut c_frac) = (0f64, vec![0f32; self.cpairs.len()]);
        let mut c_cor = vec![0f64; self.m.n_open_corridors()];
        for (ci, cp) in self.cpairs.iter().enumerate() {
            let (h, wg, n) = self.m.commutes[cp.c as usize];
            let (h, wz) = (h as usize, self.m.comp_of_gid[wg as usize] as usize);
            let (out, ret) = (n as f64 * TO_WORK[q] as f64, n as f64 * TO_HOME[q] as f64);
            let home = if cp.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (rail_c, sum) = rail_logsum_x(r, s, p.fare_min, home, &self.access_cw[wz], pc.beta, p.soften(pc.band_unsoftened(cp.band as usize), cp.car), fx_of(h), &mut terms);
            if sum <= 0. {
                continue;
            }
            // T-918: rail against the car on the cost ratio, as the local commutes
            let frac = pc.ratio_share(rail_c, cp.car) as f64;
            c_frac[ci] = frac as f32;
            add(&mut rail_d, (out + ret) * (pc.ratio_share(rail_c + step(h), cp.car) as f64 - frac), (out + ret) * frac, zkm(h, wz));
            let (ro, rr_) = (out * frac, ret * frac);
            c_rail += ro + rr_;
            c_cor[cp.corridor] += ro + rr_;
            let (hc, wc) = (col[h] as usize, col[wz] as usize);
            for &(sa, sb, v) in &terms {
                let (a, b) = ((ro * v / sum) as f32, (rr_ * v / sum) as f32);
                od_c.push((sb, sa, a));
                od_c.push((sa, sb, b));
                let (sa, sb) = (sa as usize, sb as usize);
                put(&mut st, sa, wc, a);
                put(&mut st, sb, nz + hc, a);
                put(&mut st, sb, hc, b);
                put(&mut st, sa, nz + wc, b);
            }
        }
        // T-1175: everyday outings across a zone line, chosen as the everyday layer chooses: the
        // pair's walkers with no rail first (their weight against the car from that share), rail
        // and the car splitting the rest on their cost ratio, rail taking its share of the
        // walkers; out at the purpose's outward timing, home on the same station pairs reversed
        let pe = commute_params().everyday();
        let mut e = EDemand { frac: vec![0f32; self.epairs.len()], ..EDemand::default() };
        let mut od_e: Vec<Vec<(u32, u32, f32)>> = vec![vec![]; if self.epairs.is_empty() { 0 } else { KIND_COUNT }];
        for (ei, ep) in self.epairs.iter().enumerate() {
            let pair = self.m.everyday[ep.e as usize];
            let k = pair.kind as usize;
            let (h, az) = self.epair_zones(ep);
            let (out, ret) = (pair.trips as f64 * OUTWARD[k][q] as f64, pair.trips as f64 * RETURN[k][q] as f64);
            let home = if ep.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (rail_c, sum) = rail_logsum_x(r, s, p.fare_min, home, &self.access_ew[az], pe.beta, p.soften(pe.band_unsoftened(ep.band as usize), ep.car), fx_of(h), &mut terms);
            if sum <= 0. {
                continue;
            }
            let util = (pe.mode_beta(ep.car) * (ep.car - rail_c) + pe.asc_rail).clamp(-80., 80.) as f64;
            let w0 = ep.walk_e as f64 / (util.exp() + 1. + ep.walk_e as f64);
            let frac = (1. - w0) * pe.ratio_share(rail_c, ep.car) as f64;
            e.frac[ei] = frac as f32;
            let rd = rail_c + step(h);
            let ud = (pe.mode_beta(ep.car) * (ep.car - rd) + pe.asc_rail).clamp(-80., 80.) as f64;
            add(&mut rail_d, (out + ret) * ((1. - ep.walk_e as f64 / (ud.exp() + 1. + ep.walk_e as f64)) * pe.ratio_share(rd, ep.car) as f64 - frac), (out + ret) * frac, zkm(h, az));
            let (ro, rr_) = (out * frac, ret * frac);
            e.rail += ro + rr_;
            e.walk_lost += (ro + rr_) * pair.walk as f64;
            let (hc, ac) = (col[h] as usize, col[az] as usize);
            for &(sa, sb, v) in &terms {
                let (a, b) = ((ro * v / sum) as f32, (rr_ * v / sum) as f32);
                od_e[k].push((sb, sa, a));
                od_e[k].push((sa, sb, b));
                let (sa, sb) = (sa as usize, sb as usize);
                put(&mut st, sa, ac, a);
                put(&mut st, sb, nz + hc, a);
                put(&mut st, sb, hc, b);
                put(&mut st, sa, nz + ac, b);
            }
        }
        e.od = od_e.into_iter().map(|v| SparseOd::of(s, v)).collect();
        Demand { od, od_c: SparseOd::of(s, od_c), st, rail, cor, pair_rail, c_rail, c_frac, c_cor, e, rail_d }
    }
    /// T-1229: each served everyday pair's kind, home place, rail legs in period `q` and the
    /// walking legs those riders were taken from (the period's average over crowding rounds).
    fn each_e_rail(&self, s: &Period, q: usize, mut f: impl FnMut(usize, usize, f64, f64)) {
        for (ei, ep) in self.epairs.iter().enumerate() {
            let pair = &self.m.everyday[ep.e as usize];
            let k = pair.kind as usize;
            let rail = pair.trips as f64 * (OUTWARD[k][q] + RETURN[k][q]) as f64 * s.e_frac.get(ei).copied().unwrap_or(0.) as f64;
            f(k, self.m.zones[pair.h as usize].place, rail, rail * pair.walk as f64);
        }
    }
    /// T-1175: an everyday pair's home zone and activity zone (open zones).
    fn epair_zones(&self, ep: &EPairOn) -> (usize, usize) {
        let pair = &self.m.everyday[ep.e as usize];
        (pair.h as usize, self.m.comp_of_gid[pair.a as usize] as usize)
    }
    fn per(&self, q: usize) -> Option<&Period> {
        self.per.get(q).and_then(|v| v.as_ref())
    }
    /// The index in `pairs` of the pair from zone a to zone b, `u32::MAX` if none.
    fn pair_at(&self, a: usize, b: usize) -> u32 {
        let key = (a.min(b) as u32, a.max(b) as u32);
        let at = |m: usize| (self.pairs[2 * m].a, self.pairs[2 * m].b);
        let n = self.pairs.len() / 2;
        let (mut k, mut hi) = (0, n);
        while k < hi {
            let m = (k + hi) / 2;
            if at(m) < key { k = m + 1 } else { hi = m }
        }
        if a != b && k < n && at(k) == key { (2 * k + (a > b) as usize) as u32 } else { u32::MAX }
    }
    /// Whether zones a and b could make a pair (`solved_pairs` with every pair solved).
    fn solvable(&self, a: usize, b: usize) -> bool {
        let m = &self.m;
        let at = |z: usize| self.served_at.get(z).copied().unwrap_or(u32::MAX);
        let (i, k) = (at(a), at(b));
        let (za, zb) = (&m.zones[a], &m.zones[b]);
        a != b && i != u32::MAX && k != u32::MAX && m.pr(za.place, zb.place).has_alt() && m.eachway(za, zb) >= m.rule.min_trips && self.served_comp[i as usize].iter().any(|c| self.served_comp[k as usize].binary_search(c).is_ok())
    }
    /// The pairs between the `served` zones (ascending) joined by track, with a car or a plane to
    /// compete with, in the whole market's order (T-174, T-409, T-377). `comp` is `served_comp`.
    fn set_served(&mut self, served: Vec<usize>, comp: Vec<Vec<u32>>) {
        let m = self.m.clone();
        let joined = |i: usize, k: usize| comp[i].iter().any(|c| comp[k].binary_search(c).is_ok());
        self.pairs = vec![];
        let (any, each) = solved_pairs(&m, &served, &joined, self.chunk);
        let mut pairs = Vec::with_capacity(2 * each.len());
        let (mut bit, mut next) = (0, each.into_iter());
        for (i, &a) in served.iter().enumerate() {
            let za = &m.zones[a];
            for &b in &served[i + 1..] {
                let zb = &m.zones[b];
                bit += 1;
                if any[(bit - 1) / 64] >> ((bit - 1) % 64) & 1 == 0 {
                    continue;
                }
                let r = m.pr(za.place, zb.place);
                let trips = next.next().unwrap();
                let d = m.zone_km(za, zb) as f32;
                let corridor = m.open_corridor(za.place, zb.place) as u32;
                let car = |park: f32| r.car_min(d, park);
                let drive = |park: f32| if r.car { car(park) } else { r.refm };
                let (a, b) = (a as u32, b as u32);
                let band = self.base.band_of_d2(d * d * 1e6) as u8;
                pairs.push(Pair { a, b, trips, drive: drive(zb.park), rest: r.rest, corridor, car: car(zb.park), band });
                pairs.push(Pair { a: b, b: a, trips, drive: drive(za.park), rest: r.rest, corridor, car: car(za.park), band });
            }
        }
        self.pairs = pairs;
        let mut at = vec![u32::MAX; m.zones.len()];
        for (k, &z) in served.iter().enumerate() {
            at[z] = k as u32;
        }
        let pc = commute_params();
        self.cpairs.clear();
        self.cpair_of = vec![u32::MAX; m.commutes.len()];
        for (c, &(h, wg, _)) in m.commutes.iter().enumerate() {
            let w = m.comp_of_gid[wg as usize];
            if w == u32::MAX {
                continue;
            }
            let (h, w) = (h as usize, w as usize);
            if at[h] == u32::MAX || at[w] == u32::MAX || !joined(at[h] as usize, at[w] as usize) {
                continue;
            }
            let (zh, zw) = (&m.zones[h], &m.zones[w]);
            let km = m.zone_km(zh, zw) as f32;
            let r = m.pr(zh.place, zw.place);
            let car = drive_min(km) * r.rt * pc.w_drive + zw.park + pc.car_own_min() + km * ROAD_DETOUR * r.rk * pc.car_own_km_min() + r.cross;
            let corridor = m.open_corridor(zh.place, zw.place);
            self.cpair_of[c] = self.cpairs.len() as u32;
            let band = pc.band_of_d2(km * km * 1e6) as u8;
            self.cpairs.push(CPair { c: c as u32, car, corridor, ride: km >= pc.ride_min_km, band });
        }
        self.cpairs.shrink_to_fit();
        // T-1175: the everyday pairs, as the commutes
        self.epairs.clear();
        self.epair_of = vec![u32::MAX; m.everyday.len()];
        for (e, ep) in m.everyday.iter().enumerate() {
            let a = m.comp_of_gid[ep.a as usize];
            if a == u32::MAX {
                continue;
            }
            let (h, a) = (ep.h as usize, a as usize);
            if at[h] == u32::MAX || at[a] == u32::MAX || !joined(at[h] as usize, at[a] as usize) {
                continue;
            }
            let (zh, za) = (&m.zones[h], &m.zones[a]);
            let km = m.zone_km(zh, za) as f32;
            let r = m.pr(zh.place, za.place);
            let car = drive_min(km) * r.rt * pc.w_drive + za.park + pc.car_own_min() + km * ROAD_DETOUR * r.rk * pc.car_own_km_min() + r.cross;
            let w = ep.walk.min(0.999);
            self.epair_of[e] = self.epairs.len() as u32;
            self.epairs.push(EPairOn { e: e as u32, car, walk_e: w / (1. - w), ride: km >= pc.ride_min_km, band: pc.band_of_d2(km * km * 1e6) as u8 });
        }
        self.epairs.shrink_to_fit();
        self.served_at = at;
        self.served = served;
        self.served_comp = comp;
    }
    /// Every pair of the market, as the served pairs would be with every zone served.
    #[cfg(test)]
    fn all_pairs(&self) -> Vec<Pair> {
        let mut a = IntercityApi::on(self.m.clone());
        let all: Vec<usize> = (0..self.m.zones.len()).collect();
        let comp = vec![vec![0u32]; all.len()];
        a.set_served(all, comp);
        a.pairs
    }
    /// Station access by zone and the zone pairs a station serves, for `svc`'s layout. Each group
    /// of sites sees the stations in its own frame.
    fn set_access(&mut self, svc: &Service) {
        let m = self.m.clone();
        let net0 = DemandApi::period_net(svc, 0);
        let nzz = m.zones.len();
        let mut weights: Vec<BTreeMap<usize, f32>> = (0..nzz).map(|_| BTreeMap::new()).collect();
        let mut work = weights.clone();
        // T-1175: the everyday pairs' home ends walk and ride as the commutes' do
        let commuting = !m.commutes.is_empty() || !m.everyday.is_empty();
        let outings = !m.everyday.is_empty();
        let (mut cw_home, mut cw_work, mut cw_walk) = (weights.clone(), weights.clone(), weights.clone());
        let mut ew = weights.clone();
        let beta_c = commute_params().beta;
        // T-718: walks against rides to a station at the station choice's scale of the longest
        // band, where nearly every trip between zones falls
        let theta = self.base.band_unsoftened(self.base.n_bands() - 1);
        let mut walks: Vec<(usize, f32)> = Vec::with_capacity(K);
        for g in &m.groups {
            // the stations in this group's frame
            let mut net = net0.clone();
            if let (Some(a), Some(f)) = (&m.anchor, &g.frame) {
                if a != f {
                    for s in 0..net.st_x.len() {
                        let (lon, lat) = a.to_lonlat(net.st_x[s] as f64, net.st_y[s] as f64);
                        let (x, y) = f.from_lonlat(lon, lat);
                        net.st_x[s] = x as f32;
                        net.st_y[s] = y as f32;
                    }
                }
            }
            let acc = build_access(&g.city, &net, &self.base);
            // T-373: rides from each site's 2 km zone
            let (h_off, h_rides) = access_options(&g.kz, &net, &self.base, &g.kplace, End::Home);
            let (f_off, f_rides) = access_options(&g.kz, &net, &self.base, &g.kplace, End::Far);
            for c in 0..g.city.len() {
                let z = m.cell_zone[g.lo + c];
                let zz = &m.zones[z];
                let w = g.city.pop[c] / zz.pop.max(1e-12) as f32;
                let jw = g.city.jobs[c] / zz.w_attr.max(1e-12) as f32;
                let cjw = ((g.city.jobs[c] - g.city.pop[c]) / 2.).max(0.) / zz.jobs.max(1e-12) as f32;
                let kz = g.kz.cell_zone[c] as usize;
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
                if outings {
                    add_access(&mut ew[z], &walks, &[], beta_c, theta, jw);
                }
            }
        }
        let top8 = |maps: Vec<BTreeMap<usize, f32>>, positive: bool| -> Vec<Vec<(usize, f32)>> {
            maps.into_iter()
                .map(|mm| {
                    let mut v: Vec<_> = mm.into_iter().filter(|e| !positive || e.1 > 0.).collect();
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
        self.access_ew = if outings { top8(ew, true) } else { vec![] };
        self.want = vec![false; svc.st_x.len()];
        for list in self.access_work.iter().chain(&self.access_cw).chain(&self.access_c).chain(&self.access_c_walk).chain(&self.access_ew) {
            for &(st, _) in list {
                self.want[st] = true;
            }
        }
        // T-174: only pairs between zones a station serves can carry a rail trip
        let served: Vec<usize> = (0..nzz).filter(|&z| !self.access[z].is_empty() || !self.access_work[z].is_empty()).collect();
        // T-409: and only pairs whose zones reach one track component
        let comp = station_components(&build_graph(&net0, &self.base));
        let comp: Vec<Vec<u32>> = served
            .iter()
            .map(|&z| {
                let mut v: Vec<u32> = [&self.access, &self.access_work, &self.access_c, &self.access_cw, &self.access_c_walk, &self.access_ew].iter().filter_map(|l| l.get(z)).flatten().map(|&(st, _)| comp[st]).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        if served != self.served || comp != self.served_comp {
            self.set_served(served, comp);
        }
        self.set_st_pattern(svc.st_x.len());
    }
    /// The far-zone table's pattern (`StPattern`) for the access lists and pairs as they stand:
    /// every entry `demand` can add to.
    fn set_st_pattern(&mut self, s: usize) {
        let nz = self.served.len();
        let words = (2 * nz).div_ceil(64);
        let mut bits = vec![0u64; s * words];
        let mut set = |st: usize, c: usize| bits[st * words + c / 64] |= 1 << (c % 64);
        let col = &self.served_at;
        for pair in &self.pairs {
            let (ca, cb) = (nz + col[pair.a as usize] as usize, col[pair.b as usize] as usize);
            for &(sa, _) in &self.access[pair.a as usize] {
                set(sa, cb);
            }
            for &(sb, _) in &self.access_work[pair.b as usize] {
                set(sb, ca);
            }
        }
        for cp in &self.cpairs {
            let (h, wg, _) = self.m.commutes[cp.c as usize];
            let (h, wz) = (h as usize, self.m.comp_of_gid[wg as usize] as usize);
            let home = if cp.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (hc, wc) = (col[h] as usize, col[wz] as usize);
            for &(sa, _) in home {
                set(sa, wc);
                set(sa, nz + wc);
            }
            for &(sb, _) in &self.access_cw[wz] {
                set(sb, nz + hc);
                set(sb, hc);
            }
        }
        for ep in &self.epairs {
            let (h, az) = self.epair_zones(ep);
            let home = if ep.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (hc, ac) = (col[h] as usize, col[az] as usize);
            for &(sa, _) in home {
                set(sa, ac);
                set(sa, nz + ac);
            }
            for &(sb, _) in &self.access_ew[az] {
                set(sb, nz + hc);
                set(sb, hc);
            }
        }
        let rows = || bits.chunks_exact(words.max(1)).take(s);
        let mut off = Vec::with_capacity(s + 1);
        off.push(0u32);
        for row in rows() {
            off.push(off.last().unwrap() + row.iter().map(|b| b.count_ones()).sum::<u32>());
        }
        off.resize(s + 1, *off.last().unwrap());
        let cols = rows().flat_map(|row| {
            row.iter().enumerate().flat_map(|(k, &b)| {
                let mut b = b;
                std::iter::from_fn(move || {
                    (b != 0).then(|| {
                        let c = 64 * k as u32 + b.trailing_zeros();
                        b &= b - 1;
                        c
                    })
                })
            })
        });
        let n = *off.last().unwrap() as usize;
        let col = if 2 * nz <= u16::MAX as usize {
            let mut v = Vec::with_capacity(n);
            v.extend(cols.map(|c| c as u16));
            Cols::Short(v)
        } else {
            let mut v = Vec::with_capacity(n);
            v.extend(cols);
            Cols::Long(v)
        };
        self.st_pat = StPattern { off, col };
    }
    fn set_network_in(&mut self, st_xy: &[f32], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        let t = now_ms();
        let mut off = vec![0];
        for &n in line_n {
            off.push(off.last().unwrap() + n)
        }
        if st_xy.len() % 2 != 0 || stops.len() != *off.last().unwrap() as usize || times.len() != 6 * stops.len() || tph.len() != 3 * cars.len() || cars.len() != line_n.len() || stops.iter().any(|&s| s as usize >= st_xy.len() / 2) {
            return Err("intercity network arrays disagree".into());
        }
        let (st_x, st_y, st_map) = merge_stations(st_xy, &off, stops, self.base.station_merge_m);
        let group = demand_groups(&st_map, st_x.len(), &self.complex_in);
        let ups = demand_ups(&st_map, st_x.len(), &self.ups_in);
        let svc = Service {
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off: off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            stop_xy: stops.iter().flat_map(|&s| [st_xy[2 * s as usize], st_xy[2 * s as usize + 1]]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect(),
            cars: cars.to_vec(),
            group,
            span: span_for(&self.span_in, stops.len()),
            ups,
            passages: demand_passages(&st_map, &self.passages_in),
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
}

#[wasm_bindgen]
impl IntercityApi {
    /// A market from a world composition (`{"world": registry rule and rows, "zones": [place
    /// packs]}`) or an old single pack (`sites`: a closed world).
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str) -> Result<IntercityApi, String> {
        Ok(IntercityApi::on(Rc::new(parse(header)?)))
    }
    /// T-409: the same market with no network yet, sharing this one's data, for a network of its
    /// own: one per cluster the market reaches.
    pub fn beside(&self) -> IntercityApi {
        IntercityApi::on(self.m.clone())
    }
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.complex_in = complex.to_vec();
    }
    /// `DemandApi::set_upgrades` (T-1281).
    pub fn set_upgrades(&mut self, ups: &[u8]) {
        self.ups_in = ups.to_vec();
    }
    /// `DemandApi::set_passages` (T-1281).
    pub fn set_passages(&mut self, pairs: &[u32]) {
        self.passages_in = pairs.to_vec();
    }
    /// `DemandApi::set_span` (T-842): riders between cities read the same service span.
    pub fn set_span(&mut self, span: &[f32]) {
        self.span_in = span.to_vec();
    }
    /// T-780: zone pairs sampled in chunks of `MARKET_CHUNK` trips (`solved_pairs`), or every pair
    /// solved; the next network sets its pairs up again.
    pub fn set_sampled(&mut self, on: bool) {
        self.set_chunk(if on { MARKET_CHUNK } else { 0. });
    }
}
impl IntercityApi {
    fn set_chunk(&mut self, chunk: f64) {
        if chunk != self.chunk {
            self.chunk = chunk;
            self.svc = None;
            self.served.clear();
        }
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// Sites, open zones, the day's trips between places, 1 if the places' rows came from the
    /// world (0: added up at opening, a closed world or the pipeline), the day's departures to
    /// places with no zone.
    pub fn info(&self) -> Vec<f64> {
        vec![self.m.n_sites() as f64, self.m.zones.len() as f64, self.m.cw.daily, self.m.from_rows as u8 as f64, self.m.cw.leaving]
    }
    /// Per open 6 km zone: its global id, then the box its sites lie in (west, south, east, north;
    /// degrees in a world composition, else metres in the pack's plane). A station far from a
    /// zone's box is in no access list of that zone (T-409).
    pub fn zone_boxes(&self) -> Vec<f64> {
        let m = &self.m;
        let mut out = vec![f64::INFINITY; 5 * m.zones.len()];
        for (z, zz) in m.zones.iter().enumerate() {
            out[5 * z] = zz.gid as f64;
            out[5 * z + 3] = f64::NEG_INFINITY;
            out[5 * z + 4] = f64::NEG_INFINITY;
        }
        for s in 0..m.n_sites() {
            let (g, i) = m.site(s);
            let (x, y) = (g.city.x[i] as f64, g.city.y[i] as f64);
            let (x, y) = match (&g.frame, m.world) {
                (Some(f), true) => f.to_lonlat(x, y),
                _ => (x, y),
            };
            let b = &mut out[5 * m.cell_zone[s] + 1..5 * m.cell_zone[s] + 5];
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        }
        out
    }
    /// T-377: the network with stations as longitude and latitude (a world composition): put in
    /// the frame of its first group of sites for the graph, and in each group's own for access.
    pub fn set_network_ll(&mut self, st_ll: &[f64], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        if st_ll.len() % 2 != 0 || st_ll.iter().any(|v| !v.is_finite()) {
            return Err("intercity station positions".into());
        }
        let xy: Vec<f32> = match &self.m.anchor {
            Some(a) => st_ll.chunks_exact(2).flat_map(|p| {
                let (x, y) = a.from_lonlat(p[0], p[1]);
                [x as f32, y as f32]
            }).collect(),
            None => st_ll.iter().map(|&v| v as f32).collect(),
        };
        self.set_network_in(&xy, line_n, stops, times, tph, cars)
    }
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, km: &[f32]) -> Result<(), String> {
        if !base.is_finite() || base < 0. || !per_km.is_finite() || per_km < 0. || km.iter().any(|x| !x.is_finite() || *x < 0.) || km.len() != self.svc.as_ref().map_or(0, |s| 2 * s.stops.len()) {
            return Err("invalid intercity fare".into());
        }
        let w = self.base.fare_w;
        let (fare_min, per_km) = (base * w, per_km * w);
        if fare_min.to_bits() != self.base.fare_min.to_bits() || per_km.to_bits() != self.fare_per_km.to_bits() || !same_bits(km, &self.fare_km) {
            self.base.fare_min = fare_min;
            self.fare_per_km = per_km;
            self.fare_km = km.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    /// T-1277: perceived minutes a dollar of the fares the market is given (`DemandApi::
    /// set_fare_weight`); before `set_fares`. The plane's fare weighs each departing country's
    /// own (`WorldIn::fare_w`).
    pub fn set_fare_weight(&mut self, w: f32) -> Result<(), String> {
        if !(w.is_finite() && w > 0.) {
            return Err("invalid fare weight".into());
        }
        if w.to_bits() != self.base.fare_w.to_bits() {
            self.base.fare_w = w;
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    pub fn set_fare_curve(&mut self, knots: &[f32]) -> Result<(), String> {
        check_fare_curve(knots)?;
        if !same_bits(knots, &self.fare_curve) {
            self.fare_curve = knots.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    /// T-1286: the factor each country's riders weigh the market's fares at, as JSON
    /// `{"US": 1.25, ...}` (countries left out: 1): the player's fare scale there over its income's.
    /// Each trip takes its origin's (a commute or outing its home's).
    pub fn set_fare_factors(&mut self, json: &str) -> Result<(), String> {
        let by: HashMap<String, f32> = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if by.values().any(|f| !(f.is_finite() && *f > 0.)) {
            return Err("invalid fare factor".into());
        }
        let m = &self.m;
        let of_country: Vec<f32> = m.countries.iter().map(|c| by.get(c).copied().unwrap_or(1.) - 1.).collect();
        let fx: Vec<f32> = if of_country.iter().all(|&d| d == 0.) {
            vec![]
        } else {
            m.zones.iter().map(|z| of_country[m.places[z.place].country as usize]).collect()
        };
        if !same_bits(&fx, &self.zone_fx) {
            self.zone_fx = fx;
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
        put_fare_curve(&mut g, &self.fare_curve, &self.fare_km, self.base.fare_min, self.fare_per_km, self.base.fare_w);
        let no_crowding = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &no_crowding);
        let xi = xfer_index(&g);
        let pass = self.pass(&g, &p, q, &xi);
        let (e_trips, e_walk_base) = self.m.everyday.iter().fold((0f64, 0f64), |(t, w), e| {
            let legs = e.trips as f64 * (OUTWARD[e.kind as usize][q] + RETURN[e.kind as usize][q]) as f64;
            (t + legs, w + legs * e.walk as f64)
        });
        self.per[q] = Some(Period {
            e_avg: pass.e_loads,
            e_trips,
            e_rail: pass.e_rail,
            e_walk: (e_walk_base - pass.e_walk_lost).max(0.),
            e_walk_base,
            e_frac: pass.e_frac,
            net,
            p,
            g,
            avg: pass.loads,
            c_avg: pass.c_loads,
            rail: pass.rail,
            rail_d: pass.rail_d,
            trips: self.m.cw.daily * SHARES[q] as f64,
            c_rail: pass.c_rail,
            c_trips: self.m.cw.commuters * (TO_WORK[q] + TO_HOME[q]) as f64,
            c_frac: pass.c_frac,
            corridors: pass.cor,
            c_corridors: pass.c_cor,
            pair_rail: pass.pair_rail,
            st: pass.st,
            round: 0,
            ms: now_ms() - t,
            fac: no_crowding,
            uniform: true,
            seen: vec![],
            xi,
            xfer: pass.xfer,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per(q) else { return false };
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            return false;
        };
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        if settled(&mut st.seen, &total[..st.g.rn_line.len()]) || (st.uniform && fac == st.fac && left == st.g.rn_left) {
            st.round += 1;
            st.ms = now_ms() - t;
            self.per[q] = Some(st);
            return true;
        }
        set_crowding(&mut st.g, &fac);
        st.g.rn_left = left;
        st.uniform = false;
        st.fac = fac;
        let Pass { loads: l, c_loads, rail, c_rail, c_frac, cor, c_cor, pair_rail, st: st_new, xfer, e_loads, e_rail, e_walk_lost, e_frac, rail_d } = self.pass(&st.g, &st.p, q, &st.xi);
        st.round += 1;
        let r = st.round;
        msa(&mut st.xfer, &xfer, r);
        msa(&mut st.pair_rail, &pair_rail, r);
        msa(&mut st.c_frac, &c_frac, r);
        msa(&mut st.c_avg.seg, &c_loads.seg, r);
        msa(&mut st.c_avg.rn_board, &c_loads.rn_board, r);
        msa(&mut st.c_avg.rn_alight, &c_loads.rn_alight, r);
        st.c_rail += (c_rail - st.c_rail) / (r as f64 + 1.);
        msa(&mut st.e_frac, &e_frac, r);
        for (a, b) in st.e_avg.iter_mut().zip(e_loads) {
            if let (Some(a), Some(b)) = (a.as_mut(), b) {
                msa(&mut a.seg, &b.seg, r);
                msa(&mut a.rn_board, &b.rn_board, r);
                msa(&mut a.rn_alight, &b.rn_alight, r);
            }
        }
        st.e_rail += (e_rail - st.e_rail) / (r as f64 + 1.);
        let e_walk = (st.e_walk_base - e_walk_lost).max(0.);
        st.e_walk += (e_walk - st.e_walk) / (r as f64 + 1.);
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
        msa(&mut st.avg.rn_fare_x, &l.rn_fare_x, r);
        let w = 1. / (r as f64 + 1.);
        st.rail += (rail - st.rail) * w;
        st.rail_d.toward(rail_d, w);
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
    /// `DemandApi::cap_with_external` (T-1070). The longer trips' figures by pair, corridor and
    /// station fall by the share of all the market's rail trips that still fit.
    pub fn cap_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per.get_mut(q).and_then(|p| p.as_mut()) else { return false };
        let Some(fit) = capacity_fit(&st.g, &st.net, &st.p, total) else { return false };
        if !fit.any {
            return true;
        }
        let (_, all) = cap_loads(&st.g, &fit, &st.xi, &mut st.avg, &mut st.xfer);
        let c = &mut st.c_avg;
        cap_part(&fit, &mut c.seg, &mut c.rn_board, &mut c.rn_alight);
        for e in st.e_avg.iter_mut().flatten() {
            cap_part(&fit, &mut e.seg, &mut e.rn_board, &mut e.rn_alight);
        }
        st.rail *= all;
        st.c_rail *= all;
        st.e_rail *= all;
        st.rail_d.scale(all);
        let f = all as f32;
        for v in st.pair_rail.iter_mut().chain(st.c_frac.iter_mut()).chain(st.e_frac.iter_mut()).chain(st.st.iter_mut()) {
            *v *= f;
        }
        for v in st.corridors.iter_mut().chain(st.c_corridors.iter_mut()) {
            *v *= all;
        }
        true
    }
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let (_, c) = crowd_factors(&s.g, &s.net, &s.avg, &s.p);
            vec![s.round as f64, s.trips + s.c_trips + s.e_trips, s.rail + s.c_rail + s.e_rail, s.e_walk, s.ms, self.m.cw.place_pairs as f64, c.segs_over_seats as f64, c.max_load_of_crush as f64, c.p50 as f64, c.p90 as f64, s.rail_d.d, s.rail_d.d_km, s.rail_d.km]
        })
    }
    /// The regional layer's parts in period `q` (T-276): trips and rail trips of the longer trips,
    /// then legs and rail legs of the commutes between zones, then (T-1175) legs, rail legs and
    /// walking legs of the everyday outings across a zone line, and their walking legs with no
    /// rail (the same on any network).
    ///
    /// Then (T-1229) the same four for each everyday purpose, `KIND_COUNT` blocks in `everyday`
    /// kind order (legs, rail legs, walking legs, walking legs with no rail), which add up to the
    /// first four.
    pub fn part_summary(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let mut out = vec![s.trips, s.rail, s.c_trips, s.c_rail, s.e_trips, s.e_rail, s.e_walk, s.e_walk_base];
        let by = self.everyday_by_place(q);
        for k in 0..KIND_COUNT {
            let mut f = [0f64; 4];
            for p in 0..self.m.places.len() {
                let b = &by[3 * (p * KIND_COUNT + k)..][..3];
                let base = self.m.e_base[p * KIND_COUNT + k].1 * (OUTWARD[k][q] + RETURN[k][q]) as f64;
                f[0] += b[0];
                f[1] += b[1];
                f[2] += b[2];
                f[3] += base;
            }
            out.extend(f);
        }
        out
    }
    /// T-1229: the everyday outings across a zone line by home place and purpose in period `q`:
    /// per place (market order), per `everyday` kind, legs, rail legs and walking legs (the
    /// walkers rail takes already out). Empty before the period is solved.
    fn everyday_by_place(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let np = self.m.places.len();
        let mut out = vec![0f64; 3 * np * KIND_COUNT];
        for (i, &(n, w)) in self.m.e_base.iter().enumerate() {
            let k = i % KIND_COUNT;
            let share = (OUTWARD[k][q] + RETURN[k][q]) as f64;
            out[3 * i] = n * share;
            out[3 * i + 2] = w * share;
        }
        self.each_e_rail(s, q, |k, p, rail, lost| {
            out[3 * (p * KIND_COUNT + k) + 1] += rail;
            out[3 * (p * KIND_COUNT + k) + 2] -= lost;
        });
        for w in out.iter_mut().skip(2).step_by(3) {
            *w = w.max(0.);
        }
        out
    }
    /// T-276: the commutes' segment loads, boardings and alightings per route node in period `q`.
    pub fn commute_loads(&self, q: usize) -> Vec<f32> {
        self.per(q).filter(|_| !self.m.commutes.is_empty()).map_or(vec![], |s| [&s.c_avg.seg[..], &s.c_avg.rn_board[..], &s.c_avg.rn_alight[..]].concat())
    }
    /// T-1175: the everyday outings across a zone line, as `commute_loads` per everyday purpose
    /// in `everyday` kind order (`KIND_COUNT` blocks of three, zeros for a purpose with none);
    /// empty when the market has none.
    pub fn everyday_loads(&self, q: usize) -> Vec<f32> {
        let Some(s) = self.per(q).filter(|s| !s.e_avg.is_empty()) else { return vec![] };
        let m = s.c_avg.seg.len();
        let zero = vec![0f32; m];
        let mut out = Vec::with_capacity(3 * KIND_COUNT * m);
        for k in 0..KIND_COUNT {
            match s.e_avg.get(k).and_then(|l| l.as_ref()) {
                Some(l) => out.extend([&l.seg[..], &l.rn_board[..], &l.rn_alight[..]].concat()),
                None => (0..3).for_each(|_| out.extend_from_slice(&zero)),
            }
        }
        out
    }
    /// Per corridor (place pair): its two places, longer trips in period `q` and their rail trips.
    pub fn corridor_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let m = &self.m;
            let mut out: Vec<f64> = m.each_corridor().enumerate().flat_map(|(i, (a, b))| [a as f64, b as f64, m.cw.trips[i] * SHARES[q] as f64, 0.]).collect();
            for (k, c) in m.open_corridors() {
                out[4 * c + 3] = s.corridors[k];
            }
            out
        })
    }
    /// T-378: per corridor, the shares of its longer trips not by rail that fly and that drive
    /// (neither: near places on two landmasses). The same whatever the network.
    pub fn corridor_modes(&self) -> Vec<f32> {
        // (an open place's pairs from its row; the others' modes alone worked out, T-1204)
        let m = &self.m;
        let mut out = Vec::with_capacity(2 * m.n_corridors());
        // T-1277: a plane's fare weighs the departing country's, so the two ways can differ; a
        // corridor's split is their mean
        let way = |a: usize, b: usize| -> (bool, f32) {
            if m.rules.slot[a] != u32::MAX || m.rules.slot[b] != u32::MAX {
                let r = m.pr(a, b);
                (r.car, r.air)
            } else {
                let (p, q) = (&m.places[a], &m.places[b]);
                let km = if a == b {
                    0.
                } else if m.ground {
                    chord_km(&p.u, &q.u)
                } else {
                    ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
                };
                let (car, air, ..) = pair_mode_rule(&m.places, &m.countries, &m.mp, &m.drives, a, b, km < m.rule.near_km, km);
                (car, air)
            }
        };
        for (a, b) in m.each_corridor() {
            let ((car, x), (_, y)) = (way(a, b), way(b, a));
            let air = 0.5 * (x + y);
            out.extend([air, if car { 1. - air } else { 0. }]);
        }
        out
    }
    /// T-378: per place, its longer trips in period `q` with places that have no zone, each way.
    pub fn elsewhere(&self, q: usize) -> Vec<f64> {
        let s = SHARES.get(q).copied().unwrap_or(0.) as f64;
        self.m.cw.leaving_by.iter().map(|v| v * s).collect()
    }
    /// T-1008: the corridors' two places each, in `corridor_summary`'s order. With
    /// `corridor_trips`, `commute_base` and `corridor_modes` the part of the corridors no network or
    /// period changes, which the main thread is sent once.
    pub fn corridor_pairs(&self) -> Vec<u32> {
        self.m.each_corridor().flat_map(|(a, b)| [a as u32, b as u32]).collect()
    }
    /// T-1008: per corridor, its longer trips a day; period `q`'s are these times `corridor_shares(q)[0]`.
    pub fn corridor_trips(&self) -> Vec<f64> {
        self.m.cw.trips.clone()
    }
    /// T-1008: per corridor, the commutes between zones that `commute_corridors` counts, before the
    /// period's legs (`corridor_shares(q)[1]`); empty where it gives none.
    pub fn commute_base(&self) -> Vec<f64> {
        let m = &self.m;
        if m.cw.commuters <= 0. {
            return vec![];
        }
        let mut out = vec![0f64; m.n_corridors()];
        for &(c, n) in &m.cw.commute {
            out[c as usize] = n;
        }
        out
    }
    /// T-1008: period `q`'s factors of `corridor_trips` and `commute_base`; empty where the period
    /// is not solved here, as `corridor_summary` is.
    pub fn corridor_shares(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |_| vec![SHARES[q] as f64, 0.5 * (TO_WORK[q] + TO_HOME[q]) as f64])
    }
    /// T-1008: period `q`'s corridors with rail, as (corridor, rail trips, commute rail legs): the
    /// pairs of `corridor_summary` and `commute_corridors` with rail, without the rest.
    pub fn corridor_rail(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let commutes = self.m.cw.commuters > 0.;
        let mut out = vec![];
        // (the open places' corridors come in corridor order)
        for (k, i) in self.m.open_corridors() {
            let r = s.corridors[k];
            let c = if commutes { s.c_corridors.get(k).copied().unwrap_or(0.) } else { 0. };
            if r != 0. || c != 0. {
                out.extend([i as f64, r, c]);
            }
        }
        out
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
    /// As `DemandApi::fare_extra`.
    pub fn fare_extra(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| fare_x_usd(&s.avg.rn_fare_x, self.base.fare_w))
    }
    /// `transfers` on a kind of day (T-260).
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
        (0..PERIODS).filter(|&q| self.per(q).is_some()).fold(0, |m, q| m | 1 << q)
    }
    /// `selection_flows` on a kind of day (T-260, `days.rs`): that day's trips and period shares.
    pub fn selection_flows_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        self.selection(market, xy, share, day, self.solved_mask(), true)
    }
    /// T-377: `selection_flows_day`'s all-mode numbers alone (rail zero) over the periods in
    /// `mask`, solved or not: a market composed of every open pack answers them, its parts add
    /// their rail (`selection_rail_day`).
    pub fn selection_all_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32, mask: u32) -> Vec<f64> {
        self.selection(market, xy, share, day, mask, false)
    }
    /// T-1204: which corridors' figures this market reads; compositions of one world with rows
    /// (and their parts) share them, so `corridor_pairs`, `corridor_trips`, `commute_base` and
    /// `corridor_modes` are the same for every market with the same id.
    pub fn corridors_id(&self) -> u32 {
        self.m.cw.id
    }
    /// T-377: whether `other` is this market's composition (a part beside it, or itself).
    pub fn same_market(&self, other: &IntercityApi) -> bool {
        Rc::ptr_eq(&self.m, &other.m)
    }
}
impl IntercityApi {
    fn selection(&self, market: u32, xy: &[f32], share: &[f32], day: u32, mask: u32, with_rail: bool) -> Vec<f64> {
        let m = &self.m;
        let day = day as usize;
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 1 + 8 * np;
        let etail = head + 7 * ng + 4 * np + 4 * ng;
        let mut out = vec![0.; etail + E_CROSS.len() * 6 * ng + 2];
        let away = out.len() - 2;
        m.write_zone_xy(&mut out[head + 4 * ng..head + 7 * ng]);
        if xy.len() != 2 * share.len() || market as usize >= np {
            return out;
        }
        let nz = m.zones.len();
        let (mut pop_w, mut attr_w, mut job_w) = (vec![0f64; nz], vec![0f64; nz], vec![0f64; nz]);
        let mut matched = 0.;
        for (i, &f) in share.iter().enumerate() {
            let key = ((xy[2 * i] / SITE_M).floor() as i32, (xy[2 * i + 1] / SITE_M).floor() as i32, market as usize);
            let Some(&c) = m.site_key.get(&key) else { continue };
            let f = (f as f64).clamp(0., 1.);
            matched += 1.;
            let z = m.cell_zone[c];
            let (g, ci) = m.site(c);
            let zz = &m.zones[z];
            pop_w[z] += f * g.city.pop[ci] as f64 / zz.pop.max(1e-12);
            attr_w[z] += f * g.city.jobs[ci] as f64 / zz.w_attr.max(1e-12);
            if zz.jobs > 0. {
                job_w[z] += f * ((g.city.jobs[ci] - g.city.pop[ci]) as f64 / 2.).max(0.) / zz.jobs;
            }
        }
        out[0] = if share.is_empty() { 1. } else { matched / share.len() as f64 };
        if mask == 0 {
            return out;
        }
        let per_of = |q: usize| if mask >> q & 1 == 0 { None } else { Some(if with_rail { self.per(q) } else { None }) };
        // each selected zone's pairs with every other open zone, and with each place not open
        const NONE: u32 = u32::MAX;
        let closed: Vec<usize> = (0..np).filter(|&p| m.places[p].has && !m.places[p].open).collect();
        let mut rows: HashMap<usize, (Vec<(u32, u32, u32, f64)>, Vec<(usize, f64)>)> = HashMap::new();
        // T-378: each zone's trips each way with places that have no zone: its departures less
        // its trips with every place, as `Market::leaving_by` is a place's
        let mut away_z: HashMap<usize, f64> = HashMap::new();
        for z in (0..nz).filter(|&z| pop_w[z] > 0. || attr_w[z] > 0.) {
            let at = |a: usize, b: usize| self.pair_at(a, b);
            let row: Vec<_> = m.zone_row(z).into_iter().map(|(b, trips)| (b as u32, at(z, b), at(b, z), trips)).collect();
            let far: Vec<_> = closed.iter().map(|&p| (p, m.zone_place_eachway(&m.zones[z], p))).filter(|(_, t)| *t > 0.).collect();
            let known = row.iter().map(|r| r.3).sum::<f64>() + far.iter().map(|f| f.1).sum::<f64>();
            away_z.insert(z, (m.zones[z].dep - known).max(0.));
            rows.insert(z, (row, far));
        }
        let rail = |per: Option<&Period>, i: u32| match per {
            Some(p) if i != NONE => p.pair_rail[i as usize],
            _ => 0.,
        };
        // T-780: with sampled pairs a zone's rail with a place is its sampled pairs', spread over the
        // place's zones it could ride to by their trips, so no far zone shows more rail than trips
        let mut spread = vec![(0f64, 0f64); if self.chunk > 0. { np } else { 0 }];
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            let sh = share_out(day, LONGER, q) as f64;
            let f = load_factor(day, LONGER, q) as f64;
            for (dir, weights) in [(0usize, &pop_w), (1usize, &attr_w)] {
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    let (row, far) = &rows[&z];
                    out[away + dir] += w * away_z[&z] * sh;
                    if !spread.is_empty() {
                        spread.fill((0., 0.));
                        for &(b, out_i, in_i, trips) in row {
                            let e = &mut spread[m.zones[b as usize].place];
                            e.0 += rail(per, if dir == 0 { out_i } else { in_i }) as f64;
                            e.1 += if self.solvable(z, b as usize) { trips } else { 0. };
                        }
                    }
                    for &(b, out_i, in_i, trips) in row {
                        let zb = &m.zones[b as usize];
                        let all = w * trips * sh;
                        let pair_r = match spread.get(zb.place) {
                            Some(&(r, t)) if t > 0. && self.solvable(z, b as usize) => (r * trips / t) as f32,
                            Some(_) => 0.,
                            None => rail(per, if dir == 0 { out_i } else { in_i }),
                        };
                        let r = w * pair_r as f64 * f;
                        let o = 1 + 8 * zb.place + 4 * dir;
                        out[o] += all;
                        out[o + 1] += r;
                        out[o + 2] += all * zb.x as f64;
                        out[o + 3] += all * zb.y as f64;
                        out[head + 4 * zb.gid as usize + 2 * dir] += all;
                        out[head + 4 * zb.gid as usize + 2 * dir + 1] += r;
                    }
                    for &(p, trips) in far {
                        let all = w * trips * sh;
                        let o = 1 + 8 * p + 4 * dir;
                        out[o] += all;
                        out[o + 2] += all * m.places[p].x as f64;
                        out[o + 3] += all * m.places[p].y as f64;
                    }
                }
            }
        }
        // T-276: the commutes between zones, per far place, then per far zone: legs a day of the
        // commuters living in the selection, by where they work, and rail legs; then the same for
        // the commuters working in the selection, by where they live
        let tail = head + 7 * ng;
        let mut add = |far_gid: usize, side: usize, all: f64, rail: f64| {
            let p = m.gid_place[far_gid] as usize;
            out[tail + 4 * p + 2 * side] += all;
            out[tail + 4 * p + 2 * side + 1] += rail;
            let o = tail + 4 * np + 4 * far_gid + 2 * side;
            out[o] += all;
            out[o + 1] += rail;
        };
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            let legs = (share_out(day, WORK, q) + share_ret(day, WORK, q)) as f64;
            let rail_legs = (TO_WORK[q] + TO_HOME[q]) as f64 * load_factor(day, WORK, q) as f64;
            let frac = |c: u32| match (per, self.cpair_of.get(c as usize)) {
                (Some(per), Some(&i)) if i != u32::MAX => per.c_frac.get(i as usize).copied().unwrap_or(0.) as f64,
                _ => 0.,
            };
            for (end, weights) in [(0usize, &pop_w), (1usize, &job_w)] {
                let (off, items) = if end == 0 { &m.c_home } else { &m.c_work };
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    for &c in &items[off[z] as usize..off[z + 1] as usize] {
                        let (h, wg, n) = m.commutes[c as usize];
                        let far = if end == 0 { wg as usize } else { m.zones[h as usize].gid as usize };
                        let n = w * n as f64;
                        add(far, end, n * legs, n * rail_legs * frac(c));
                    }
                }
            }
        }
        // T-1229: the everyday outings across a zone line, by purpose and far zone (home end:
        // outings of the selection's residents by where they go; activity end: by where they live):
        // legs a day, rail legs and walking legs. Without rail (`with_rail` off) the walking legs are
        // the pairs' own, and with it the walkers rail takes, negative.
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            for (end, weights) in [(0usize, &pop_w), (1usize, &attr_w)] {
                let (off, items) = if end == 0 { &m.e_home } else { &m.e_act };
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    for &i in &items[off[z] as usize..off[z + 1] as usize] {
                        let pair = &m.everyday[i as usize];
                        let k = pair.kind as usize;
                        let Some(ki) = E_CROSS.iter().position(|&c| c == k) else { continue };
                        let far = if end == 0 { pair.a as usize } else { m.zones[pair.h as usize].gid as usize };
                        let n = w * pair.trips as f64;
                        let legs = n * (share_out(day, everyday_purpose(k), q) + share_ret(day, everyday_purpose(k), q)) as f64;
                        let o = etail + ki * 6 * ng + 6 * far + 3 * end;
                        out[o] += legs;
                        match per {
                            None => out[o + 2] += legs * pair.walk as f64,
                            Some(per) => {
                                let ei = self.epair_of.get(i as usize).copied().unwrap_or(u32::MAX);
                                let frac = if ei == u32::MAX { 0. } else { per.e_frac.get(ei as usize).copied().unwrap_or(0.) as f64 };
                                let rail = n * (OUTWARD[k][q] + RETURN[k][q]) as f64 * load_factor(day, everyday_purpose(k), q) as f64 * frac;
                                out[o + 1] += rail;
                                out[o + 2] -= rail * pair.walk as f64;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// T-409: `selection_flows_day` with only its rail numbers, the rest zero: what a market beside
    /// another adds to that one's answer.
    pub fn selection_rail_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        let mut out = self.selection_flows_day(market, xy, share, day);
        let (np, ng) = (self.m.places.len(), self.m.n_gid);
        let (head, tail) = (1 + 8 * np, 1 + 8 * np + 7 * ng);
        let etail = tail + 4 * np + 4 * ng;
        let away = out.len() - 2;
        // T-1229: the everyday pairs' blocks hold (legs, rail legs, walkers rail takes): the last
        // two are rail's
        let rail = |i: usize| {
            if i == 0 || i >= away {
                false
            } else if i < head {
                (i - 1) % 4 == 1
            } else if i < head + 4 * ng {
                (i - head) % 2 == 1
            } else if i < tail {
                false
            } else if i < etail {
                (i - tail) % 2 == 1
            } else {
                (i - etail) % 3 != 0
            }
        };
        for (i, v) in out.iter_mut().enumerate() {
            if !rail(i) {
                *v = 0.;
            }
        }
        out
    }
    /// `station_flows` on a kind of day (T-260): each period weighed by the market's load factor.
    pub fn station_flows_day(&self, station: u32, day: u32) -> Vec<f64> {
        let Some(svc) = &self.svc else { return vec![] };
        let Some(&s) = svc.st_map.get(station as usize) else {
            return vec![];
        };
        let m = &self.m;
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 6 * np;
        let mut out = vec![0.; head + 5 * ng];
        let ns = self.served.len();
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let f = load_factor(day as usize, LONGER, q) as f64;
            let pat = &self.st_pat;
            if (s as usize) + 1 < pat.off.len() && per.st.len() == pat.col.len() {
                for i in pat.row(s as usize) {
                    let k = pat.col.get(i);
                    let (end, k) = if k < ns { (0, k) } else { (ng, k - ns) };
                    out[head + end + m.zones[self.served[k]].gid as usize] += per.st[i] as f64 * f;
                }
            }
        }
        for d in 0..2 {
            for &z in &self.served {
                let zone = &m.zones[z];
                let v = out[head + d * ng + zone.gid as usize];
                out[d * np + zone.place] += v;
                out[(2 + 2 * d) * np + zone.place] += v * zone.x as f64;
                out[(3 + 2 * d) * np + zone.place] += v * zone.y as f64;
            }
        }
        m.write_zone_xy(&mut out[head + 2 * ng..]);
        out
    }
    /// `station_flows` of a station this network does not hold but the whole one does (T-326).
    pub fn no_station_flows(&self) -> Vec<f64> {
        let (np, ng) = (self.m.places.len(), self.m.n_gid);
        let head = 6 * np;
        let mut out = vec![0.; head + 5 * ng];
        self.m.write_zone_xy(&mut out[head + 2 * ng..]);
        out
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

#[cfg(test)]
mod tests;
