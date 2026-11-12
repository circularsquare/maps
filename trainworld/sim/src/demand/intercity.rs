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
//! data/work/intercity.json, the joint commute gravity's input, and the tests' fixtures) opens as
//! a closed world over its own zones with the ground curve alone.
use super::{
    api::{check_fare_curve, demand_catches, demand_groups, demand_passages, demand_ups, fare_x_usd, merge_stations, put_fares, check_fare_mul, same_bits, span_for, scale_triples, DemandApi, Service, FARE_MIN_PER_USD, PEAK_HOUR_FACTOR, TO_HOME, TO_WORK},
    clock::now_ms,
    days::{everyday as everyday_purpose, lean_sign, load_factor, share_out, share_ret, LONGER, WEEKDAY, WORK},
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
    /// T-3240 (pipeline/market_pull.json): a destination place's attraction, ground and tail alike, times
    /// `exp(gamma x px)`, `px` its excess (its row's), faded in over `pull_phase_km`; `gamma` is
    /// `pull_gamma` times `pull_country_gamma`'s factor for the country within it, `pull_gamma_x` across a
    /// border; none for residents of `pull_off`. Absent: no pull.
    #[serde(default)]
    pull_gamma: f64,
    #[serde(default)]
    pull_gamma_x: f64,
    #[serde(default = "default_pull_phase")]
    pull_phase_km: [f64; 2],
    #[serde(default)]
    pull_off: Vec<String>,
    #[serde(default)]
    pull_country_gamma: BTreeMap<String, f64>,
}
fn default_pull_phase() -> [f64; 2] {
    [50., 150.]
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
            pull_gamma: 0.,
            pull_gamma_x: 0.,
            pull_phase_km: default_pull_phase(),
            pull_off: vec![],
            pull_country_gamma: BTreeMap::new(),
        }
    }
    fn ok(&self) -> bool {
        let v = [self.r, self.a, self.d0_km, self.eps, self.g, self.d1_km, self.th, self.eps_t, self.tho, self.border_eps, self.u_mean_us, self.u_ref_people, self.border, self.rt, self.near_km, self.min_trips, self.pull_gamma, self.pull_gamma_x];
        v.iter().chain(&self.onset_km).chain(&self.tail_onset_km).chain(&self.pull_phase_km).chain(self.pull_country_gamma.values()).all(|x| x.is_finite() && *x >= 0.)
            && self.onset_km[1] > self.onset_km[0]
            && self.tail_onset_km[1] > self.tail_onset_km[0]
            && self.pull_phase_km[1] > self.pull_phase_km[0]
            && self.u_mean_us > 0.
            && self.u_ref_people > 0.
    }
    /// T-3240: the pull's fade at `d` km: 0 under its window, 1 from its end.
    #[inline(always)]
    fn pull_on(&self, d: f64) -> f64 {
        ((d - self.pull_phase_km[0]) / (self.pull_phase_km[1] - self.pull_phase_km[0])).clamp(0., 1.)
    }
    /// T-3240: the gamma of a trip from a resident of country `a` to a place in `b`; 0: no pull.
    fn pull_gamma_of(&self, a: &str, b: &str) -> f64 {
        if self.pull_off.iter().any(|c| c == a) {
            0.
        } else if a == b {
            self.pull_gamma * self.pull_country_gamma.get(b).copied().unwrap_or(1.)
        } else {
            self.pull_gamma_x
        }
    }
    /// The tail's onset at `d` km: 0 under its first bound, 1 from its second.
    #[inline(always)]
    fn tail_on(&self, d: f64) -> f64 {
        ((d - self.tail_onset_km[0]) / (self.tail_onset_km[1] - self.tail_onset_km[0])).clamp(0., 1.)
    }
    /// The ground curve at `d` km: the onset times `max(d, d0)^-A`, 0 under the onset. T-1603: where
    /// a tail runs it takes over across its own onset window, so the ground fades over it and is 0
    /// from its second bound.
    #[inline(always)]
    fn ground(&self, d: f64) -> f64 {
        let on = ((d - self.onset_km[0]) / (self.onset_km[1] - self.onset_km[0])).clamp(0., 1.);
        if on == 0. {
            return 0.;
        }
        let keep = if self.rt == 0. { 1. } else { 1. - self.tail_on(d) };
        if keep == 0. {
            0.
        } else {
            on * keep / d.max(self.d0_km).powf(self.a)
        }
    }
    /// The long-haul tail at `d` km: its onset times `max(d, D1)^-G`; 0 under the onset.
    #[inline(always)]
    fn tail(&self, d: f64) -> f64 {
        let on = self.tail_on(d);
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
/// A place's part of the market (`data/packs/<id>.market.json`, pipeline/market_zones.py) as the
/// file has it; `ZoneIn` keeps it in less room.
#[derive(Deserialize, Default)]
struct ZoneRaw {
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
    /// T-3240: the place's pull excess (`Rule`'s pull)
    #[serde(default)]
    px: f64,
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
/// A column of numbers kept as 4 bytes each when every one is exact so (a float32, or whole
/// thousandths, as the packs write people and jobs), else as they came (T-1559).
enum Col {
    F32(Vec<f32>),
    Milli(Vec<u32>),
    F64(Vec<f64>),
}
impl Col {
    fn of(v: Vec<f64>) -> Col {
        if v.iter().all(|&x| (x as f32) as f64 == x) {
            return Col::F32(v.iter().map(|&x| x as f32).collect());
        }
        let milli = |x: f64| {
            let k = (x * 1000.).round();
            (x.is_sign_positive() && k <= u32::MAX as f64 && k / 1000. == x).then_some(k as u32)
        };
        if let Some(k) = v.iter().map(|&x| milli(x)).collect::<Option<Vec<u32>>>() {
            return Col::Milli(k);
        }
        Col::F64(v)
    }
    fn len(&self) -> usize {
        match self {
            Col::F32(v) => v.len(),
            Col::Milli(v) => v.len(),
            Col::F64(v) => v.len(),
        }
    }
    /// The number, as the file gave it; `None` past the end.
    #[inline]
    fn at(&self, i: usize) -> Option<f64> {
        match self {
            Col::F32(v) => v.get(i).map(|&x| x as f64),
            Col::Milli(v) => v.get(i).map(|&k| k as f64 / 1000.),
            Col::F64(v) => v.get(i).copied(),
        }
    }
    fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len()).map(|i| self.at(i).unwrap())
    }
}
/// A column of indices kept in 2 bytes each when they fit (T-1559).
enum Ix {
    U16(Vec<u16>),
    U32(Vec<u32>),
}
impl Ix {
    fn of(v: Vec<u32>) -> Ix {
        if v.iter().all(|&x| x <= u16::MAX as u32) {
            Ix::U16(v.iter().map(|&x| x as u16).collect())
        } else {
            Ix::U32(v)
        }
    }
    fn len(&self) -> usize {
        match self {
            Ix::U16(v) => v.len(),
            Ix::U32(v) => v.len(),
        }
    }
    #[inline]
    fn at(&self, i: usize) -> u32 {
        match self {
            Ix::U16(v) => v[i] as u32,
            Ix::U32(v) => v[i],
        }
    }
}
/// A place's sites: `[x, y, people, jobs, zone]` of the file, x and y as the float32 the
/// composition reads, the zone as its index (`u32::MAX` where the file's is not one).
struct Sites {
    x: Vec<f32>,
    y: Vec<f32>,
    pop: Col,
    jobs: Col,
    zone: Ix,
}
impl Sites {
    fn len(&self) -> usize {
        self.x.len()
    }
    /// Site `i`: x, y, people, jobs, zone.
    #[inline]
    fn at(&self, i: usize) -> (f32, f32, f64, f64, u32) {
        (self.x[i], self.y[i], self.pop.at(i).unwrap(), self.jobs.at(i).unwrap(), self.zone.at(i))
    }
}
/// A place's part of the market as the composition reads it: `ZoneRaw`, its long arrays in 2 or 4
/// bytes an entry (T-1559). Everything read from them is the number the file gave, to the bit.
#[derive(Deserialize)]
#[serde(from = "ZoneRaw")]
struct ZoneIn {
    place: usize,
    zone: String,
    frame: Option<PackFrame>,
    zone_m: f64,
    country: String,
    inc: f64,
    road: i64,
    zones: Vec<[i32; 2]>,
    zone_ng: Vec<f64>,
    zone_nt: Vec<f64>,
    sites: Sites,
    site_park: Vec<f32>,
    cell_pop: Col,
    px: f64,
    commute_home: Ix,
    commute_place: Ix,
    commute_work: Ix,
    commute_trips: Vec<f32>,
    everyday_home: Ix,
    everyday_place: Ix,
    everyday_zone: Ix,
    everyday_kind: Vec<u8>,
    everyday_trips: Vec<f32>,
    everyday_walk: Vec<f32>,
    airports: Vec<[f64; 2]>,
    air_direct: Vec<Vec<u32>>,
}
/// A vector with no spare room (parsing grows each by doubling).
fn tight<T>(mut v: Vec<T>) -> Vec<T> {
    v.shrink_to_fit();
    v
}
impl From<ZoneRaw> for ZoneIn {
    fn from(z: ZoneRaw) -> ZoneIn {
        let site_zone = |v: f64| if v >= 0. && v.fract() == 0. && v < u32::MAX as f64 { v as u32 } else { u32::MAX };
        let sites = Sites {
            x: z.sites.iter().map(|s| s[0] as f32).collect(),
            y: z.sites.iter().map(|s| s[1] as f32).collect(),
            pop: Col::of(z.sites.iter().map(|s| s[2]).collect()),
            jobs: Col::of(z.sites.iter().map(|s| s[3]).collect()),
            zone: Ix::of(z.sites.iter().map(|s| site_zone(s[4])).collect()),
        };
        ZoneIn {
            place: z.place,
            zone: z.zone,
            frame: z.frame,
            zone_m: z.zone_m,
            country: z.country,
            inc: z.inc,
            road: z.road,
            zones: tight(z.zones),
            zone_ng: z.zone_ng,
            zone_nt: z.zone_nt,
            sites,
            site_park: tight(z.site_park),
            cell_pop: Col::of(z.cell_pop),
            px: z.px,
            commute_home: Ix::of(z.commute_home),
            commute_place: Ix::of(z.commute_place),
            commute_work: Ix::of(z.commute_work),
            commute_trips: tight(z.commute_trips),
            everyday_home: Ix::of(z.everyday_home),
            everyday_place: Ix::of(z.everyday_place),
            everyday_zone: Ix::of(z.everyday_zone),
            everyday_kind: tight(z.everyday_kind),
            everyday_trips: tight(z.everyday_trips),
            everyday_walk: tight(z.everyday_walk),
            airports: z.airports,
            air_direct: z.air_direct,
        }
    }
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
    /// T-3240: its pull excess (`Rule`'s pull); absent: 0
    #[serde(default, skip_serializing_if = "is_zero")]
    px: f64,
    /// T-1277: perceived minutes a fare dollar costs people leaving here (the app adds it to the
    /// row from its country's income, game/income.ts); 0 or absent: `FARE_MIN_PER_USD`. Never
    /// written to market-rows.json
    #[serde(default, skip_serializing_if = "is_zero")]
    fw: f64,
    /// T-1521: what a car costs to run a km of road leaving here, US$ (the app adds it from the
    /// country's fuel and tolls, game/income.ts); 0 or absent: `CAR_USD_PER_KM`. Never written to
    /// market-rows.json
    #[serde(default, skip_serializing_if = "is_zero")]
    cu: f64,
    /// T-1431: its people's income groups as the app gives them (game/income.ts, never written to
    /// market-rows.json): per group its share of the longer trips and the factor on `fw` its riders
    /// weigh every mode's money at, up to `GROUPS` pairs; absent: one group at `fw`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    fg: Vec<f64>,
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
    /// T-1374: the world's attractions as the app gives them (attractions.json): longitude,
    /// latitude, visitors a year
    #[serde(default)]
    attractions: Vec<[f64; 3]>,
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
    /// its row's `cu` (0: `CAR_USD_PER_KM`)
    cu: f64,
    /// T-1431: its income groups (share of the longer trips, factor on `fw`), the first `ng`;
    /// none: one group at `fw`
    gr: [(f32, f32); GROUPS],
    ng: u8,
    /// T-1374: how touristy it is (`touristy`), 0 with no pack open
    tour: f32,
    /// T-3240: its pull excess (its row's or pack's)
    px: f64,
}
/// T-1431: the most income groups a place's travellers come in.
pub const GROUPS: usize = 3;
const ONE_GROUP: [(f32, f32); 1] = [(1., 1.)];
impl Place {
    /// Perceived minutes a dollar costs a rider of average income leaving here.
    fn w(&self) -> f32 {
        if self.fw > 0. { self.fw as f32 } else { FARE_MIN_PER_USD }
    }
    /// What a car costs to run a km of road from here, US$.
    fn car_usd(&self) -> f32 {
        if self.cu > 0. { self.cu as f32 } else { CAR_USD_PER_KM }
    }
    /// Its income groups: each one's share of the longer trips and the factor on `w` its riders
    /// weigh money at.
    fn groups(&self) -> &[(f32, f32)] {
        if self.ng == 0 { &ONE_GROUP } else { &self.gr[..self.ng as usize] }
    }
    /// T-1902: perceived minutes a dollar costs its longer trips' travellers on average (`w` times
    /// the groups' trip weighted factor), what the trip explorer prices the car and plane at.
    fn trip_w(&self) -> f32 {
        self.w() * self.groups().iter().map(|g| g.0 * g.1).sum::<f32>()
    }
}
/// T-1431: a row's `fg` as a place keeps it: the shares made to add up to 1; none for no groups.
fn income_groups(fg: &[f64]) -> Result<([(f32, f32); GROUPS], u8), String> {
    let mut gr = [(0f32, 1f32); GROUPS];
    if fg.is_empty() {
        return Ok((gr, 0));
    }
    let n = fg.len() / 2;
    let sum: f64 = fg.iter().step_by(2).sum();
    if fg.len() % 2 != 0 || n > GROUPS || fg.iter().any(|v| !v.is_finite() || *v < 0.) || fg.iter().skip(1).step_by(2).any(|&f| f <= 0.) || sum <= 0. {
        return Err("invalid market row income groups".into());
    }
    for (k, g) in gr.iter_mut().enumerate().take(n) {
        *g = ((fg[2 * k] / sum) as f32, fg[2 * k + 1] as f32);
    }
    Ok((gr, n as u8))
}
/// A mode's cost in minutes and in dollars, the dollars at the rider's minutes a dollar (T-1431:
/// every mode at the same weight for a given rider).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Cost {
    min: f32,
    usd: f32,
}
impl Cost {
    const NONE: Cost = Cost { min: f32::INFINITY, usd: 0. };
    #[inline(always)]
    fn at(self, w: f32) -> f32 {
        self.min + self.usd * w
    }
    fn plus(self, o: Cost) -> Cost {
        Cost { min: self.min + o.min, usd: self.usd + o.usd }
    }
    fn is_some(self) -> bool {
        self.min.is_finite()
    }
}
/// Per ordered pair of places: the border factor, the tail's income factor, near or not, and
/// the distance between their centres; T-378: whether a car can make the trip, the share of the
/// trips not by rail that fly (over the first place's income groups, T-1431), and the flight by
/// the best pair of airports (`Cost::NONE` with none).
#[derive(Clone, Copy, Default)]
struct PairRule {
    b: f64,
    rel: f64,
    /// T-3240: the second place's pull on trips from the first at full strength, less 1 (0: none)
    pull: f64,
    near: bool,
    km: f64,
    car: bool,
    air: f32,
    fl: Cost,
    /// the car's cost for water it is carried over (`crossing_cost`, a ferry)
    cross: Cost,
    /// T-279: the routed drive's minutes and road km over the rule's (`Drive`)
    rt: f32,
    rk: f32,
    /// T-1521: the car's running cost a km of road, US$, the two places' mean (0: `CAR_USD_PER_KM`)
    cu: f32,
}
/// T-1431: the modes but rail of one zone pair for one rider: the reference mode's perceived minutes
/// (the car's, or with no car the plane's), the weight of every mode but rail against it, the
/// share of those trips that fly, and the car's minutes, which set the choice's scale (T-495) even
/// where no car goes.
#[derive(Clone, Copy, Debug)]
struct Alt {
    drive: f32,
    rest: f32,
    air: f32,
    car: f32,
}
impl PairRule {
    /// A pair the solve may build: a car or a plane to compete with.
    fn has_alt(&self) -> bool {
        self.car || self.fl.is_some()
    }
    /// A longer trip's car over a zone pair of this place pair, `d` km apart, to parking `park`.
    fn car_cost(&self, d: f32, park: f32) -> Cost {
        self.zone_car(d, park, (self.rt, self.rk))
    }
    /// `car_cost` with the zone pair's own routed ratios (`Market::zone_ratios`).
    fn zone_car(&self, d: f32, park: f32, (rt, rk): (f32, f32)) -> Cost {
        routed_car(d, park, rt, rk, if self.cu > 0. { self.cu } else { CAR_USD_PER_KM }).plus(self.cross)
    }
    /// The car the place pair's split is decided on: centre to centre, the two places' mean parking.
    fn split_car(&self, p: &Place, q: &Place) -> Cost {
        self.car_cost(self.km as f32, (0.5 * (p.park + q.park)) as f32)
    }
    /// The modes but rail for a rider weighing a dollar at `w` minutes, over a zone pair whose car
    /// costs `car` (`car_cost`), the split's car being `split` (`split_car`).
    fn alt(&self, car: Cost, split: Cost, w: f32, mp: &Params) -> Alt {
        let car_m = car.at(w);
        match (self.car, self.fl.is_some()) {
            (true, true) => {
                let s = split.at(w);
                let v = (-mp.mode_beta(s) * (self.fl.at(w) - s)).clamp(-80., 80.).exp();
                Alt { drive: car_m, rest: 1. + v, air: v / (1. + v), car: car_m }
            }
            (false, true) => Alt { drive: self.fl.at(w), rest: 1., air: 1., car: car_m },
            _ => Alt { drive: car_m, rest: 1., air: 0., car: car_m },
        }
    }
    /// The share of the trips not by rail from place `p` to `q` that fly, over `p`'s income groups
    /// by their trips (T-1431).
    fn flown(&self, p: &Place, q: &Place, mp: &Params) -> f32 {
        let (split, w) = (self.split_car(p, q), p.w());
        p.groups().iter().map(|&(t, f)| t * self.alt(split, split, w * f, mp).air).sum()
    }
    /// The reference mode's perceived minutes between the two places for a rider of average income
    /// from the first (the car's centre to centre, or with no car the plane's).
    fn refm(&self, p: &Place, q: &Place) -> f32 {
        let (split, w) = (self.split_car(p, q), p.w());
        if !self.car && self.fl.is_some() { self.fl.at(w) } else { split.at(w) }
    }
}
/// The ordered pair (i, j)'s border factor, the tail's income factor, j's pull on i's trips (less 1),
/// near or not, and the distance between their centres (`pair_rule`'s first part).
#[inline]
fn pair_trip_rule(places: &[Place], countries: &[String], rule: &Rule, ground: bool, i: usize, j: usize) -> (f64, f64, f64, bool, f64) {
    let (p, q) = (&places[i], &places[j]);
    let km = if i == j {
        0.
    } else if ground {
        chord_km(&p.u, &q.u)
    } else {
        ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
    };
    let (pc, qc) = (&countries[p.country as usize], &countries[q.country as usize]);
    let b = border_factor(rule, pc, p.inc, qc);
    let rel = if p.inc > 0. { (q.inc / p.inc).min(1.).powf(rule.eps_t) } else { 1. };
    let pull = if q.px > 0. { (rule.pull_gamma_of(pc, qc) * q.px).exp_m1() } else { 0. };
    (b, rel, pull, km < rule.near_km, km)
}
/// The ordered pair (i, j)'s rule (`PairRule`).
fn pair_rule(places: &[Place], countries: &[String], rule: &Rule, ground: bool, mp: &Params, drives: &Drives, i: usize, j: usize) -> PairRule {
    let (b, rel, pull, near, km) = pair_trip_rule(places, countries, rule, ground, i, j);
    pair_mode_rule(places, countries, mp, drives, i, j, near, km, PairRule { b, rel, pull, near, km, ..PairRule::default() })
}
/// `pair_rule`'s modes for the ordered pair (i, j), near or not, `km` apart, into `r`: whether a car
/// can go, the flight, the crossings, the routed drive, and the share of the trips not by rail that
/// fly, over i's income groups by their trips (T-1431).
#[inline]
#[allow(clippy::too_many_arguments)]
fn pair_mode_rule(places: &[Place], countries: &[String], mp: &Params, drives: &Drives, i: usize, j: usize, near: bool, km: f64, r: PairRule) -> PairRule {
    let (p, q) = (&places[i], &places[j]);
    // (the airports' direct places are sorted)
    let direct = |x: usize, y: usize| p.air[x].1.binary_search(&(j as u32)).is_ok() || q.air[y].1.binary_search(&(i as u32)).is_ok();
    let mut r = pair_ways(places, countries, drives, i, j, near, km, r, direct);
    r.air = r.flown(p, q, mp);
    r
}
/// `pair_mode_rule` but the share that flies: the car, the flight, the crossings, the routed drive
/// and the car's running cost, each the same both ways to the bit (T-1849), so a corridor works
/// them out once for its two ways. `direct(x, y)`: whether i's airport x and j's airport y have a
/// direct flight between them.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pair_ways(places: &[Place], countries: &[String], drives: &Drives, i: usize, j: usize, near: bool, km: f64, mut r: PairRule, direct: impl Fn(usize, usize) -> bool) -> PairRule {
    let (p, q) = (&places[i], &places[j]);
    // T-378: a car within a landmass; a plane between far places each with an airport;
    // both by place, the same whoever builds what (very far pairs coarse, T-362)
    r.car = p.road == q.road;
    // the best pair of airports: the drives to them, and a direct flight or a change (the fare is
    // the same for every pair, so the least minutes)
    r.fl = Cost::NONE;
    if i != j && !near {
        for (kx, (x, _)) in p.air.iter().enumerate() {
            for (ky, (y, _)) in q.air.iter().enumerate() {
                let c = air_cost(km, x + y, direct(kx, ky));
                if c.min < r.fl.min {
                    r.fl = c;
                }
            }
        }
    }
    let d = if r.car { drives.get(i, j) } else { Drive::RULE };
    r.cross = if r.car && i != j { crossing_cost((&countries[p.country as usize], p.lon, p.lat), (&countries[q.country as usize], q.lon, q.lat)).plus(d.ferry_cost()) } else { Cost::default() };
    (r.rt, r.rk) = (d.rt, d.rk);
    r.cu = 0.5 * (p.car_usd() + q.car_usd());
    r
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
    /// T-1374: how touristy it is (`touristy`)
    tour: f32,
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
    /// T-1431: some place's riders come in income groups (their fares then need `pair_km`)
    grouped: bool,
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
    site_key: SiteKeys,
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
/// Each 600 m square's first site (global index), the squares (x, y, place) in order, searched by
/// bisection on keys worked out from the sites (`Market::site_at`): 4 bytes a square (T-1558).
#[derive(Default)]
struct SiteKeys(Vec<u32>);
impl SiteKeys {
    /// From each site's key in site order; a square's first site is kept.
    fn of(keys: impl Iterator<Item = (i32, i32, usize)>) -> SiteKeys {
        let mut v: Vec<_> = keys.enumerate().map(|(s, (x, y, p))| (x, y, p as u32, s as u32)).collect();
        v.sort_unstable();
        v.dedup_by_key(|e| (e.0, e.1, e.2));
        SiteKeys(v.iter().map(|e| e.3).collect())
    }
}
/// T-1204, T-1558: the pair rules of the open places among themselves (`out`, the open places' k x k
/// table). Any other pair's rule is worked out when asked, which `corridor_modes`, the trip
/// explorer and the rows' near lists do (the corridors' trips, `Market::corridors_of`, work out
/// their factors themselves).
#[derive(Default)]
struct Rules {
    /// per place, its index among the open places, or `u32::MAX`
    slot: Vec<u32>,
    /// the open places, ascending
    open: Vec<usize>,
    out: Vec<PairRule>,
}
/// T-1204: the market's corridors (unordered place pairs a <= b, numbered as a loop over a then
/// b: `Market::corridor`) and their figures that depend on the places alone: their longer trips a
/// day, both ways; the commutes between zones on each, both ways (corridor ascending, those with
/// any); each place's departures to places with no zone; the totals.
struct Corridors {
    /// distinct for each set worked out (`IntercityApi::corridors_id`)
    id: u32,
    /// the corridors with longer trips, ascending, and their trips a day (T-2122: dense, it was 8
    /// bytes for each of the world's 12.7M corridors in every solving worker, two thirds of them 0)
    trips_at: Vec<u32>,
    trips: Vec<f64>,
    commute: Vec<(u32, f64)>,
    commuters: f64,
    leaving_by: Vec<f64>,
    daily: f64,
    leaving: f64,
    place_pairs: usize,
    /// T-1849: how many corridors have trips or commutes (`IntercityApi::table_rows`), once counted
    rows: std::cell::OnceCell<usize>,
}

/// A longer trips' zone pair, both ways apart: the modes but rail come from its place pair's rule
/// and `car` (`PairRule::alt`, T-378, T-1431).
#[derive(Clone)]
struct Pair {
    a: u32,
    b: u32,
    trips: f64,
    corridor: u32,
    /// the car's cost even where no car can go, which sets the scale (T-495)
    car: Cost,
    /// the station choice's band by the zones' distance (`Params::band`, T-718)
    band: u8,
}
/// T-276: a commute between two zones some station serves: its index in `commutes`, and the
/// car's perceived minutes from home to work (the local layer's commute rule).
#[derive(Clone)]
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
#[derive(Clone)]
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

mod logsum;
mod costs;
mod market;
mod build;
mod api;
mod pass;
mod setup;
mod exports;
mod selection;
use logsum::*;
pub use costs::*;
pub use build::*;
pub use api::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod t2652;
