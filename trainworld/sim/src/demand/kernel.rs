//! The demand kernel stages (SPEC 4.4). Flat arrays throughout; no per-cell structs.
//!
//! Scheme (see notes/T-005.md, T-019.md, T-020.md for why):
//! 1. Gravity, doubly constrained, between square zones of `zone_m` (2 km): homes -> jobs by
//!    distance only, so it does not depend on the network. The pipeline solves it once
//!    (bin/pack_gravity.rs) and ships the factors in the pack; `city_gravity` rebuilds it.
//! 2. Per cell, up to K=6 stations within reach of the walk.
//! 3. Access subzones: the cells of one zone that share their nearest station (and walk band, if
//!    bands are on). They carry the pop/jobs-weighted walk time to each of their stations, used
//!    at both ends of a trip, and their zone's rides to stations, used at the home end of long
//!    commutes (T-371). Homes beyond every walk are one subzone per zone with rides only; other
//!    cells with no station within reach are in no subzone: no rail.
//! 4. Station-to-station generalised times: one reverse strategy search per destination; useful
//!    lines share their frequency, with expected ride cost and the combined waiting time.
//! 5. Mode choice per (origin subzone, destination subzone) pair: zone trips spread by pop/jobs
//!    weight; reserve walking, then rail/road logit using the access/egress rail logsum.
//! 6. Rail trips by station pair, loaded onto stored strategies; common-line boardings split by
//!    departure frequency, while route nodes follow their best ride or alight edge.
//! 7. Crowding: penalty per segment from the loads, repeat 4-6, average loads (MSA).

use super::lanes::{F4, F8};
use super::network::Network;
use super::pack::{City, PackGravity};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Stations kept per cell and per subzone.
pub const K: usize = 6;
/// Dummy slots after each row of the commute pair loop, one per lane and egress step (T-036).
const PADS: usize = 8 * K;
const NONE: u32 = u32::MAX;
/// `Params::other_asc`'s default, perceived minutes (T-513: notes/T-513.md; T-509's joint fit kept it).
pub const OTHER_ASC: f32 = 10.0;

#[derive(Clone, Debug)]
pub struct Params {
    /// The walk to and from stations (SPEC 4.3, T-371), at both ends. The path is `walk_detour`
    /// x the straight line, covered at `walk_mps`; its first `walk_easy_min` minutes weigh
    /// `w_walk` each and the rest `w_walk_far`. `walk_cutoff_m` is only a computational bound
    /// (straight line, metres), set where `pair_walk_max` leaves little: a far station loses
    /// riders because the whole trip gets worse than driving, not at an edge.
    pub walk_cutoff_m: f32,
    pub walk_detour: f32,
    pub walk_mps: f32,
    pub walk_easy_min: f32,
    pub w_walk_far: f32,
    /// Gravity zone size, metres.
    pub zone_m: f32,
    /// Walk band width for subzones, minutes. 0 = no bands; < 0 = one subzone per cell (exact).
    pub band_min: f32,
    /// Gravity distance decay: f(d) = d^-decay_pow x exp(-d / decay_km), d in km (T-006: fitted
    /// to LODES OD for New York; decay_pow 0 is the plain exponential). T-162 refitted decay_km
    /// (29 -> 40) jointly with the US border factor on New York and Philadelphia LODES; T-198
    /// flattened the near end to d^-0.8 exp(-d/30 km) (T-185: job-rich homes' trips were too
    /// short). These are the US packs' values: a pack's gravity is used as shipped only when
    /// they match it (`city_gravity`), so they change together with `build_city.py`'s.
    pub decay_km: f32,
    pub decay_pow: f32,
    /// The decay's factor between home and work zones in different first-level regions (T-162:
    /// US states). Only for a gravity solved here or by `pack_gravity`: a pack's own gravity
    /// carries its factor (`PackGravity::border_factor`), so each country sets its own.
    pub border_factor: f32,
    pub w_walk: f32,
    pub w_wait: f32,
    pub w_home_wait: f32,
    /// Waiting beyond this many minutes (half headway) counts at `w_home_wait`.
    pub platform_wait_cap_min: f32,
    /// A fixed penalty per change of train, perceived minutes. 0 (Anita, T-557): a change costs
    /// its wait and any walk between platforms; kept for calibration runs (`realnet --set`).
    pub transfer_min: f32,
    pub fare_min: f32,
    /// Logit scale per perceived minute, and the rail propensity: the worldwide value plus
    /// `country_asc_rail`, which `set_extra` keeps in step (setting `asc_rail` by name sets the
    /// worldwide part).
    pub beta: f32,
    pub asc_rail: f32,
    pub country_asc_rail: f32,
    /// Cost damping (T-495, `mode_beta`): past this many perceived minutes of the car's trip the
    /// mode choice's scale falls as the square root of the car's minutes.
    pub damp_min: f32,
    /// Driving door to door (T-090), the alternative for the whole trip: the road is
    /// `road_detour` x the straight line; its first and last `road_local_km` (at most half the
    /// road each) are local streets, at `road_town_kmh` in open country, and the rest is main
    /// roads at `road_far_kmh`; both slow towards `road_city_kmh` where it is dense (half way at
    /// `road_dens50` commuters living and working per km² of land in the zone;
    /// `place_min_per_km`). Each end's main roads last `road_end_km`, and any middle beyond runs
    /// on open highway at `road_open_kmh` (T-370), so short trips and trips in the city crawl and
    /// long ones are quick.
    pub road_detour: f32,
    pub road_local_km: f32,
    pub road_end_km: f32,
    pub road_open_kmh: f32,
    pub road_city_kmh: f32,
    pub road_town_kmh: f32,
    pub road_dens50: f32,
    pub road_far_kmh: f32,
    pub w_drive: f32,
    /// Parking at a destination (T-380, `park_parts`): `park_min` to park for every driver; a
    /// paid space `park_paid_min` x ln(d / `park_paid_d0`) more, capped at `park_paid_cap_min`,
    /// d jobs per km² of land in the zone; a share 1 / (1 + d / `park_free_d`) parks free.
    pub park_min: f32,
    pub park_paid_min: f32,
    pub park_paid_d0: f32,
    pub park_paid_cap_min: f32,
    pub park_free_d: f32,
    /// A congestion charge per zone (`cordon_by_zone`), perceived minutes a one-way trip, paid
    /// by every driver arriving there. None: no charge anywhere.
    pub cordon: Option<std::sync::Arc<Vec<f32>>>,
    pub period: usize,
    /// Share of all home->work trips in this period.
    pub period_share: f32,
    pub period_hours: f32,
    pub peak_hour_factor: f32,
    pub crowd_alpha: f32,
    /// Zone pairs carrying fewer trips than this are skipped in mode choice.
    pub prune_trips: f32,
    /// Subzone pairs whose cheapest walks at the two ends together cost more than this many
    /// perceived minutes are not offered rail (T-090: a computational bound like
    /// `walk_cutoff_m`, set where little rail is left).
    pub pair_walk_max: f32,
    /// Logit scale of the choice among access and egress stations (and access modes), per
    /// perceived minute. notes/T-020.md. The pair loop keeps exp(-theta x station-to-station
    /// minutes) in f32, so theta x the longest such trip must stay under ~85 (0.2 allows 7 h);
    /// it cannot be pushed towards all-or-nothing.
    pub station_theta: f32,
    /// Share of all trips reserved for walking before rail/drive choice, by zone distance d:
    /// 1 / (1 + exp((d - walk_d0_km) / walk_s_km)). The remainder choose rail or drive (or bus). Judgement
    /// (notes/T-026.md): ~84% within a 2 km zone, 32% to the next zone, 6% two zones away.
    pub walk_d0_km: f32,
    pub walk_s_km: f32,
    /// Walking transfers (SPEC 6.1): a rider leaving a train may walk to another station within
    /// this straight-line distance and board there, at the walk's perceived time plus the usual
    /// boarding cost. Only after a ride, so walking between two stations is never a rail trip.
    /// 0 turns it off. T-007: crossing lines at different levels cannot share a station node.
    /// Stations of one complex (`Network::st_group`, T-161) are walks whatever the distance.
    pub xfer_walk_m: f32,
    /// The transfer walk is on foot: speed and perceived weight (Subway Builder's 1.39).
    pub xfer_mps: f32,
    pub w_xfer: f32,
    /// Stations of the snapshot this close together are one demand station (`api::merge_stations`).
    pub station_merge_m: f32,
    /// Riders left behind by full trains wait longer (T-188, `left_behind`): each crowding round
    /// adds the average extra wait at each line, direction and stop to that boarding's cost.
    pub left_behind: bool,
    /// The pack's routed walks and drives (T-275, `routes`): walks to stations around water,
    /// the walk share and the drive's road between zones by real paths. None: straight lines
    /// x `walk_detour` and x `road_detour`.
    pub routes: Option<std::sync::Arc<super::routes::Routes>>,
    /// The routed factor's multiplier (T-275): shortest-distance routes over the pack's roads
    /// are on average shorter than `road_detour`, which the density speeds were fitted with (to
    /// ACS car commute times on New York, T-090), and than the faster roads drivers take. 1.3 /
    /// 1.210, New York's commute-weighted routed factor, keeps New York's drives at the fitted
    /// level; water, bays and sparse roads still lengthen a pair against the others.
    pub road_routed_scale: f32,
    /// The ride to a station (car, bus, bike or a lift as one mode, T-371), at the home end of
    /// commutes whose zones' centres are at least `ride_min_km` apart: `p_ride` + `w_drive` x the
    /// drive from the home zone's centre + at the station the cheaper of parking there (as at a
    /// destination) and a drop-off, `d_drop`. Per home zone, the cheapest station of each of the
    /// `ride_lines` cheapest lines within `ride_max_m` (straight line); 0 lines turns it off.
    pub p_ride: f32,
    pub d_drop: f32,
    pub ride_min_km: f32,
    pub ride_max_m: f32,
    pub ride_lines: usize,
    /// Per cell, the share of its residents living in a household with no car (the pack's
    /// `nocar` over `pop`, T-511). None: everyone can drive. In the local layers their choice is
    /// rail, walking to stations, against walking the whole way (`walk_trip_min`) and "other"
    /// (bus, taxi, bike; T-513): the drive's road at `other_kmh` door to door, plus a wait of
    /// `other_wait_min` at `w_wait` and `other_asc`, no parking (`nocar_choice`). `other_kmh` and
    /// `other_wait_min` are fixed; `other_asc` is the one to fit.
    pub nocar: Option<std::sync::Arc<Vec<f32>>>,
    pub other_kmh: f32,
    pub other_wait_min: f32,
    pub other_asc: f32,
    /// The everyday layer's outings per resident (per enrolled student for school and
    /// university) and destination decay lengths in metres, by `everyday::KINDS` (T-509 stage
    /// 4). Read when `EverydayApi::beside` opens; the pack's `everyday` block sets them.
    pub everyday_rate: [f32; 6],
    pub everyday_decay_m: [f32; 6],
    /// The everyday layer's own mode choice (T-509): its rail constant is `asc_rail` plus this,
    /// and its walk share curve takes this `walk_d0_km` (with `walk_s_km`). Read when
    /// `EverydayApi::beside` opens.
    pub everyday_asc_rail: f32,
    pub everyday_walk_d0_km: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            // T-371: a true walk at 6 km/h, linear (19.5 perceived minutes a straight km at
            // `w_walk`); past a short trip the ride takes over. Walks over 3 km take a pair past
            // `pair_walk_max`
            walk_cutoff_m: 3000.0,
            walk_detour: 1.3,
            walk_mps: 6.0 / 3.6,
            walk_easy_min: 10.0,
            w_walk_far: 1.5,
            zone_m: 2000.0,
            band_min: 0.0,
            decay_km: 30.0,
            decay_pow: 0.8,
            border_factor: 1.0,
            // T-371, T-372: 1.5 rather than the usual 2.0 (notes/T-371.md, T-372.md)
            w_walk: 1.5,
            w_wait: 1.37,
            w_home_wait: 0.4,
            platform_wait_cap_min: 5.0,
            transfer_min: 0.0,
            fare_min: 6.0,
            beta: 0.05,
            asc_rail: super::api::ASC_RAIL,
            country_asc_rail: 0.0,
            damp_min: 60.0,
            road_detour: 1.3,
            road_local_km: 2.0,
            road_end_km: 8.0,
            road_open_kmh: 100.0,
            road_city_kmh: 15.0,
            road_town_kmh: 40.0,
            road_dens50: 5000.0,
            road_far_kmh: 80.0,
            w_drive: 1.33,
            park_min: 3.0,
            // T-380: within about half of real prices from Croydon to Midtown (notes/T-380.md)
            park_paid_min: 15.0,
            park_paid_d0: 3000.0,
            park_paid_cap_min: 50.0,
            park_free_d: 8000.0,
            cordon: None,
            period: 0,
            period_share: 0.45,
            period_hours: 3.0,
            peak_hour_factor: 1.25,
            crowd_alpha: 2.0,
            prune_trips: 0.0,
            // T-090: holds the pairs at what the 2.5 km edge had; on the real New York network
            // 0.3% of rail trips have walks of 70-80 perceived minutes together and 0.2% were
            // beyond it (a lone line in Manhattan, the worst case: 2.3%, and the bounds cost it
            // 4% of its riders)
            pair_walk_max: 80.0,
            station_theta: 0.2,
            walk_d0_km: 1.7,
            walk_s_km: 0.4,
            xfer_walk_m: 300.0,
            xfer_mps: 1.2,
            w_xfer: 1.39,
            station_merge_m: 100.0,
            left_behind: true,
            routes: None,
            road_routed_scale: 1.3 / 1.210,
            // T-372: fitted on the real New York network (notes/T-372.md)
            p_ride: 15.0,
            d_drop: 10.0,
            ride_min_km: 5.0,
            ride_max_m: 15_000.0,
            ride_lines: 6,
            nocar: None,
            // T-513: city buses' door-to-door speed and wait (DECISIONS 2026-10-16)
            other_kmh: 12.0,
            other_wait_min: 8.0,
            other_asc: OTHER_ASC,
            everyday_rate: super::everyday::RATE,
            everyday_decay_m: super::everyday::DECAY,
            // T-509, with `asc_rail`
            everyday_asc_rail: -1.1,
            everyday_walk_d0_km: 1.9,
        }
    }
}

impl Params {
    /// Set a parameter by name, for the checking tools (`realnet --set`); not every field.
    pub fn set_extra(&mut self, k: &str, v: f32) -> Result<(), String> {
        match k {
            "station_theta" => self.station_theta = v,
            "asc_rail" => self.asc_rail = v + self.country_asc_rail,
            "country_asc_rail" => {
                self.asc_rail += v - self.country_asc_rail;
                self.country_asc_rail = v;
            }
            "beta" => self.beta = v,
            "w_drive" => self.w_drive = v,
            "park_min" => self.park_min = v,
            "fare_min" => self.fare_min = v,
            "transfer_min" => self.transfer_min = v,
            "crowd_alpha" => self.crowd_alpha = v,
            "xfer_walk_m" => self.xfer_walk_m = v,
            "station_merge_m" => self.station_merge_m = v,
            // the gravity holds the walk curve: `DemandApi::refresh_walk` after these
            "walk_d0_km" => self.walk_d0_km = v,
            "walk_s_km" => self.walk_s_km = v,
            "damp_min" => self.damp_min = v,
            "walk_mps" => self.walk_mps = v,
            "walk_cutoff_m" => self.walk_cutoff_m = v,
            "w_wait" => self.w_wait = v,
            "w_walk" => self.w_walk = v,
            "walk_easy_min" => self.walk_easy_min = v,
            "w_walk_far" => self.w_walk_far = v,
            "pair_walk_max" => self.pair_walk_max = v,
            "road_detour" => self.road_detour = v,
            "road_routed_scale" => self.road_routed_scale = v,
            "road_local_km" => self.road_local_km = v,
            "road_end_km" => self.road_end_km = v,
            "road_open_kmh" => self.road_open_kmh = v,
            "road_city_kmh" => self.road_city_kmh = v,
            "road_town_kmh" => self.road_town_kmh = v,
            "road_dens50" => self.road_dens50 = v,
            "road_far_kmh" => self.road_far_kmh = v,
            "left_behind" => self.left_behind = v != 0.0,
            "p_ride" => self.p_ride = v,
            "d_drop" => self.d_drop = v,
            "ride_min_km" => self.ride_min_km = v,
            "ride_max_m" => self.ride_max_m = v,
            "ride_lines" => self.ride_lines = v as usize,
            "park_paid_min" => self.park_paid_min = v,
            "park_paid_d0" => self.park_paid_d0 = v,
            "park_paid_cap_min" => self.park_paid_cap_min = v,
            "park_free_d" => self.park_free_d = v,
            // scales every zone's cordon charge (0 turns them off)
            "cordon" => {
                if let Some(c) = &self.cordon {
                    self.cordon = Some(std::sync::Arc::new(c.iter().map(|x| x * v).collect()));
                }
            }
            "other_asc" => self.other_asc = v,
            "everyday_asc_rail" => self.everyday_asc_rail = v,
            "everyday_walk_d0_km" => self.everyday_walk_d0_km = v,
            // scales every cell's share without a car (0: everyone drives); before `set_network`
            "nocar" => {
                if let Some(c) = &self.nocar {
                    self.nocar = Some(std::sync::Arc::new(c.iter().map(|x| (x * v).min(1.0)).collect()));
                }
            }
            // the everyday layer's, by purpose (`rate_errands`, `decay_social` in metres); before
            // `EverydayApi::beside`
            _ if k.starts_with("rate_") || k.starts_with("decay_") => {
                let (what, kind) = k.split_once('_').unwrap();
                let i = super::everyday::KINDS.iter().position(|&n| n == kind).ok_or(format!("unknown parameter {k}"))?;
                if what == "rate" { self.everyday_rate[i] = v } else { self.everyday_decay_m[i] = v }
            }
            // the drive before T-090: one speed everywhere
            "road_kmh" => {
                self.road_city_kmh = v;
                self.road_town_kmh = v;
                self.road_far_kmh = v;
                self.road_open_kmh = v;
            }
            _ => return Err(format!("unknown parameter {k}")),
        }
        Ok(())
    }
    /// Real minutes of the fast walk to a station `d_m` metres away in a straight line.
    pub fn walk_min(&self, d_m: f32) -> f32 {
        d_m * self.walk_detour / self.walk_mps / 60.0
    }
    /// Perceived minutes of that walk: the first `walk_easy_min` at `w_walk`, the rest at
    /// `w_walk_far`.
    pub fn access_cost(&self, d_m: f32) -> f32 {
        let t = self.walk_min(d_m);
        t.min(self.walk_easy_min) * self.w_walk + (t - self.walk_easy_min).max(0.0) * self.w_walk_far
    }
    /// Minutes a km at a place with `dens` homes and jobs per km² of land: on its local streets,
    /// and on the main roads near it. Both fall from the open-country speeds (`road_town_kmh`,
    /// `road_far_kmh`) towards `road_city_kmh` as it gets dense, half way at `road_dens50`.
    pub fn place_min_per_km(&self, dens: f32) -> (f32, f32) {
        let f = 1.0 / (1.0 + dens / self.road_dens50.max(1e-6));
        let c = self.road_city_kmh;
        (60.0 / (c + (self.road_town_kmh - c) * f), 60.0 / (c + (self.road_far_kmh - c) * f))
    }
    /// Real minutes to drive `d_m` metres (straight line) between places a and b
    /// (`place_min_per_km` of each): the first and last `road_local_km` of road (at most half
    /// each) on each end's local streets, then up to `road_end_km` (at most half the rest) on
    /// each end's main roads, and any middle beyond at `road_open_kmh`.
    #[inline]
    pub fn drive_min(&self, d_m: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
        self.drive_min_f(d_m, self.road_detour, a, b)
    }
    /// `drive_min` with the road `f` x the straight line (T-275: the zones' routed factor,
    /// `drive_factor`).
    #[inline]
    pub fn drive_min_f(&self, d_m: f32, f: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
        let road = d_m * 0.001 * f;
        let e = (0.5 * road).min(self.road_local_km);
        let main = road - 2.0 * e;
        let k = (0.5 * main).min(self.road_end_km);
        e * (a.0 + b.0) + k * (a.1 + b.1) + (main - 2.0 * k) * 60.0 / self.road_open_kmh
    }
    /// The road's length over the straight line between zones i and j: the pack's routed factor
    /// (T-275), else `road_detour`.
    #[inline]
    pub fn drive_factor(&self, i: usize, j: usize) -> f32 {
        match &self.routes {
            Some(r) if r.has_drive() => r.drive_factor(i, j) * self.road_routed_scale,
            _ => self.road_detour,
        }
    }
    /// Parking at a destination in zone `j`, perceived minutes: what every driver pays (parking
    /// and any cordon charge), what a paid space costs on top, and the share who park free.
    pub fn park_parts(&self, z: &Zones, j: usize) -> (f32, f32, f32) {
        let land = (z.ncell[j].max(1) as f32 * CELL_KM2).min((z.zone_m / 1000.0).powi(2));
        let d = z.jobs[j] as f32 / land;
        let paid = if d > self.park_paid_d0 { (self.park_paid_min * (d / self.park_paid_d0).ln()).min(self.park_paid_cap_min) } else { 0.0 };
        let fixed = self.park_min + self.cordon.as_ref().and_then(|c| c.get(j)).map_or(0.0, |&c| c);
        (fixed, paid, 1.0 / (1.0 + d / self.park_free_d))
    }
    /// The mean of `park_parts` over drivers.
    pub fn park_mean(&self, z: &Zones, j: usize) -> f32 {
        let (fixed, paid, free) = self.park_parts(z, j);
        fixed + (1.0 - free) * paid
    }
    /// The mode choice's logit scale for a trip whose car costs `car` perceived minutes (the
    /// regional layer's car even where none can go, so rail against air never depends on it): `beta` up to `damp_min`, then `beta` x sqrt(`damp_min` / `car`), so a
    /// minute weighs less on a long trip (T-495). The same expressions as `mode_choice`'s lanes.
    #[inline]
    pub fn mode_beta(&self, car: f32) -> f32 {
        self.beta * (self.damp_min / car).min(1.0).sqrt()
    }
    /// Rail's share against the car, `util` being rail's utility against a driver who parks free
    /// at scale `beta` (`mode_beta`): drivers parking free and those paying `paid` for a space
    /// choose apart (T-380), the same expressions as `mode_choice`'s lanes.
    pub fn rail_share_parked(&self, util: f32, beta: f32, paid: f32, free: f32) -> f32 {
        let r0 = 1.0 / (1.0 + fast_exp(-util));
        if paid < 1e-30 {
            return r0;
        }
        let r1 = 1.0 / (1.0 + fast_exp(-(util + beta * paid)));
        r0 + (1.0 - free) * (r1 - r0)
    }
    /// Perceived minutes a km of walking the whole trip, the alternative of those without a car
    /// (T-511): along the drive's road, at `walk_mps` and `w_walk_far`.
    #[inline]
    pub fn walk_trip_min(&self) -> f32 {
        self.w_walk_far * 1000.0 / (60.0 * self.walk_mps)
    }
    /// Perceived minutes a km of "other" along the drive's road, and its fixed part (the wait
    /// at `w_wait`), T-513; `other_asc` apart.
    #[inline]
    pub fn other_min_km(&self) -> f32 {
        60.0 / self.other_kmh
    }
    #[inline]
    pub fn other_wait(&self) -> f32 {
        self.w_wait * self.other_wait_min
    }
    /// The choice of those without a car (T-511, T-513) among rail at `rail` perceived minutes
    /// (None: not offered), walking the whole way at `wk` and other at `oth` (+ `other_asc`), on
    /// the scale `mode_beta(oth)`: rail's share, and other's share of those not taking rail (a
    /// logit's, so the same whether rail is offered or not). The same expressions as
    /// `mode_choice`'s lanes.
    pub fn nocar_choice(&self, wk: f32, oth: f32, rail: Option<f32>) -> (f32, f32) {
        let bt = self.mode_beta(oth);
        let q = 1.0 / (1.0 + fast_exp(bt * (oth + self.other_asc - wk)));
        let r = rail.map_or(0.0, |rail| {
            let ew = fast_exp(-(bt * (wk - rail) + self.asc_rail));
            let eo = fast_exp(-(bt * (oth + self.other_asc - rail) + self.asc_rail));
            1.0 / (1.0 + ew + eo)
        });
        (r, q)
    }
    /// Real minutes of the walk between two stations on a transfer, `d_m` metres straight line.
    pub fn xfer_min(&self, d_m: f32) -> f32 {
        d_m * self.walk_detour / self.xfer_mps / 60.0
    }
    /// Share of all trips over `d_km` reserved for walking, independent of rail service.
    pub fn walk_share(&self, d_km: f32) -> f32 {
        1.0 / (1.0 + ((d_km - self.walk_d0_km) / self.walk_s_km).exp())
    }
    pub fn everyday_walk_share(&self, d_km: f32) -> f32 {
        1.0 / (1.0 + ((d_km - self.everyday_walk_d0_km) / self.walk_s_km).exp())
    }
}

/// Other's share of the no-car trips not by rail (`Params::nocar_choice`) by km of the drive's
/// road, tabled every 50 m and read linearly, for the totals that sum it over every zone pair or
/// trip sample (T-513); 1 past the table (within 1e-5).
pub struct OtherShare {
    tab: Vec<f32>,
}

impl OtherShare {
    const STEP_KM: f32 = 0.05;
    pub fn new(p: &Params) -> OtherShare {
        let (kw, ko, kow) = (p.walk_trip_min(), p.other_min_km(), p.other_wait());
        let mut tab = vec![];
        loop {
            let rk = tab.len() as f32 * Self::STEP_KM;
            let q = p.nocar_choice(rk * kw, rk * ko + kow, None).1;
            tab.push(q);
            if (1.0 - q < 1e-5 && rk > 1.0) || tab.len() > 100_000 {
                return OtherShare { tab };
            }
        }
    }
    #[inline]
    pub fn at(&self, road_km: f32) -> f32 {
        let x = road_km.max(0.0) / Self::STEP_KM;
        let i = x as usize;
        if i + 1 >= self.tab.len() {
            return 1.0;
        }
        let f = x - i as f32;
        self.tab[i] + (self.tab[i + 1] - self.tab[i]) * f
    }
}

// ---------------------------------------------------------------------------------------------
// 1. Zones and gravity

#[derive(Clone)]
pub struct Zones {
    pub zone_m: f32,
    pub n: usize,
    pub cell_zone: Vec<u32>,
    pub gx: Vec<i32>,
    pub gy: Vec<i32>,
    /// First-level region of each zone (T-162). A square cut by a region border is one zone per
    /// region, all at the square's position: two zones can share `gx`, `gy`.
    pub region: Vec<u8>,
    /// Grid size in zones, and its lower-left corner in metres.
    pub w: i32,
    pub h: i32,
    pub x0: f32,
    pub y0: f32,
    pub pop: Vec<f64>,
    pub jobs: Vec<f64>,
    /// Resolution 9 cells in each zone (its land, at `CELL_KM2` a cell; `res9_cells`).
    pub ncell: Vec<u32>,
}

/// Mean area of an H3 resolution 9 cell, km².
pub const CELL_KM2: f32 = 0.1053;

impl Zones {
    /// Homes and jobs per km² of land, per zone (the drive's local street speed).
    pub fn density(&self) -> Vec<f32> {
        (0..self.n).map(|i| ((self.pop[i] + self.jobs[i]) / (self.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32).collect()
    }
    /// The zones of each grid square, CSR by square index gx * h + gy (usually one; two or more
    /// where a region border cuts the square).
    pub fn by_square(&self) -> (Vec<u32>, Vec<u32>) {
        let ns = (self.w * self.h) as usize;
        let mut off = vec![0u32; ns + 1];
        for i in 0..self.n {
            off[(self.gx[i] * self.h + self.gy[i]) as usize + 1] += 1;
        }
        for s in 0..ns {
            off[s + 1] += off[s];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; self.n];
        for i in 0..self.n {
            let s = (self.gx[i] * self.h + self.gy[i]) as usize;
            items[fill[s] as usize] = i as u32;
            fill[s] += 1;
        }
        (off, items)
    }
}

/// Congestion charges at a destination (T-372): the charging area (lon, lat ring, a rough outline
/// at 2 km zones' scale) and its charge in perceived minutes a one-way trip, half the day's charge
/// at 3 minutes a US dollar.
pub const CORDONS: &[(&str, f32, &[(f32, f32)])] = &[
    // Manhattan south of 60th St, $9 at peak (2025)
    ("Manhattan", 13.5, &[(-74.020, 40.700), (-74.016, 40.715), (-74.012, 40.730), (-74.009, 40.745), (-74.002, 40.760), (-73.994, 40.774), (-73.959, 40.759), (-73.968, 40.748), (-73.972, 40.738), (-73.974, 40.726), (-73.978, 40.711), (-73.999, 40.707), (-74.012, 40.701)]),
    // the London congestion charge, £18 from 2026 at $1.30
    ("London", 35.1, &[(-0.159, 51.513), (-0.163, 51.520), (-0.126, 51.528), (-0.105, 51.530), (-0.088, 51.526), (-0.077, 51.524), (-0.072, 51.515), (-0.075, 51.5055), (-0.100, 51.495), (-0.125, 51.487), (-0.144, 51.495), (-0.152, 51.503)]),
];

fn in_ring(ring: &[(f32, f32)], x: f32, y: f32) -> bool {
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let ((xi, yi), (xj, yj)) = (ring[i], ring[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Each zone's cordon charge (`CORDONS`), as the share of its jobs inside the charging area times
/// the charge; None where no charging area holds any jobs. `origin` is the pack's (lon, lat), its
/// cells' metres equirectangular about it.
pub fn cordon_by_zone(city: &City, z: &Zones, origin: (f64, f64)) -> Option<Vec<f32>> {
    const R_EARTH: f64 = 6371008.8;
    let (lon0, lat0) = origin;
    let kx = 180.0 / std::f64::consts::PI / R_EARTH / lat0.to_radians().cos();
    let ky = 180.0 / std::f64::consts::PI / R_EARTH;
    let mut charged = vec![0f64; z.n];
    let mut any = false;
    for c in 0..city.len() {
        if city.jobs[c] <= 0.0 {
            continue;
        }
        let (lon, lat) = ((lon0 + city.x[c] as f64 * kx) as f32, (lat0 + city.y[c] as f64 * ky) as f32);
        if let Some(&(_, min, _)) = CORDONS.iter().find(|(_, _, ring)| in_ring(ring, lon, lat)) {
            charged[z.cell_zone[c] as usize] += (city.jobs[c] * min) as f64;
            any = true;
        }
    }
    any.then(|| (0..z.n).map(|j| if z.jobs[j] > 0.0 { (charged[j] / z.jobs[j]) as f32 } else { 0.0 }).collect())
}

/// Zones are the cells' 2 km squares, one per region in a square a region border cuts (T-162).
/// Zone ids in order of each zone's first cell, so a pack with one region gets the zones it had
/// before regions.
pub fn build_zones(city: &City, zone_m: f32) -> Zones {
    if city.len() == 0 {
        // the extent folds below would be infinite and w * h overflow (T-449)
        return Zones { zone_m, n: 0, cell_zone: vec![], gx: vec![], gy: vec![], region: vec![], w: 0, h: 0, x0: 0., y0: 0., pop: vec![], jobs: vec![], ncell: vec![] };
    }
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / zone_m) as i32 + 1;
    let h = ((maxy - miny) / zone_m) as i32 + 1;
    let nreg = city.region.iter().copied().max().map_or(1, |r| r as usize + 1);
    let mut grid = vec![NONE; (w * h) as usize * nreg];
    let mut z = Zones { zone_m, n: 0, cell_zone: vec![0; city.len()], gx: vec![], gy: vec![], region: vec![], w, h, x0: minx, y0: miny, pop: vec![], jobs: vec![], ncell: vec![] };
    for c in 0..city.len() {
        let gx = ((city.x[c] - minx) / zone_m) as i32;
        let gy = ((city.y[c] - miny) / zone_m) as i32;
        let r = city.region.get(c).copied().unwrap_or(0);
        let g = (gx * h + gy) as usize * nreg + r as usize;
        if grid[g] == NONE {
            grid[g] = z.n as u32;
            z.n += 1;
            z.gx.push(gx);
            z.gy.push(gy);
            z.region.push(r);
            z.pop.push(0.0);
            z.jobs.push(0.0);
            z.ncell.push(0);
        }
        let zi = grid[g] as usize;
        z.cell_zone[c] = zi as u32;
        z.ncell[zi] += res9_cells(city, c);
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    z
}

#[derive(Clone)]
pub struct Gravity {
    /// T_IJ = row[I] * col[J] * f(I, J) * k(I, J), with row = A_I O_I and col = B_J D_J, and k
    /// = `border` when zones I and J are in different regions, else 1 (`Gravity::k`, T-162).
    pub row: Vec<f64>,
    pub col: Vec<f64>,
    /// The decay's factor across a region border.
    pub border: f32,
    /// f by grid offset: index (dx + offx) * th + (dy + offy).
    pub ftab: Vec<f32>,
    /// Walk share of all trips by grid offset, same index as `ftab` (`Params::walk_share`);
    /// empty until `set_walk` fills it.
    pub wtab: Vec<f32>,
    /// T-275: the zone-centre distances (km) by grid offset and the routes whose walk factors
    /// lengthen the walk between nearby zones around water (`walk_share_pair`); set by `set_walk`.
    pub wdtab: Vec<f32>,
    pub wroutes: Option<std::sync::Arc<super::routes::Routes>>,
    pub wcurve: (f32, f32),
    pub th: i32,
    pub offx: i32,
    pub offy: i32,
    pub iters: usize,
    pub max_err: f64,
    pub mean_km: f64,
    pub total: f64,
}

impl Gravity {
    /// Index offset to add to `key(J)` for origin zone I.
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i]
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j]
    }
    /// The region factor between zones i and j: 1 in one region, `border` across.
    #[inline]
    pub fn k(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if z.region[i] == z.region[j] { 1.0 } else { self.border }
    }
    pub fn t(&self, z: &Zones, i: usize, j: usize) -> f64 {
        self.row[i] * self.col[j] * (self.ftab[(self.base(z, i) + self.key(z, j)) as usize] * self.k(z, i, j)) as f64
    }

    /// Fill `wtab` from the zone-centre distances (the same ones `decay_table` uses).
    pub fn set_walk(&mut self, z: &Zones, p: &Params) {
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        self.wtab = dtab.iter().map(|&d| p.walk_share(d)).collect();
        self.wroutes = p.routes.clone().filter(|r| r.walk_window() > 0);
        self.wcurve = (p.walk_d0_km, p.walk_s_km);
        self.wdtab = if self.wroutes.is_some() { dtab } else { vec![] };
    }

    /// The walk share between zones i and j at table index `k` (`base(i) + key(j)`): `wtab`,
    /// or with the pack's routes (T-275) the walk around water between them, the zone-centre
    /// distance times their walk factor.
    #[inline]
    pub fn walk_share_at(&self, i: usize, j: usize, k: usize) -> f32 {
        if let Some(r) = &self.wroutes {
            let f = r.walk_factor(i, j);
            if f > 1.0 {
                return 1.0 / (1.0 + ((self.wdtab[k] * f - self.wcurve.0) / self.wcurve.1).exp());
            }
        }
        self.wtab[k]
    }

    /// The same for any zone pair (0 before `set_walk`).
    pub fn walk_share_pair(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if self.wtab.is_empty() {
            return 0.0;
        }
        self.walk_share_at(i, j, (self.base(z, i) + self.key(z, j)) as usize)
    }

    /// T-513: all-day trips by other with no rail at all, by home zone and by work zone: over
    /// zone pairs, T_IJ times the share not walking, the home zone's share without a car `cz`,
    /// and other's share against walking the whole way (`Params::nocar_choice`) on the
    /// zone-centre distance along the drive's road.
    pub fn other_trips_by_zone(&self, z: &Zones, p: &Params, cz: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let (mut home, mut work) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        let oq = OtherShare::new(p);
        // Every trip of the group first, then near pairs take off what walks or walks the whole
        // way: past `r` squares other's share is 1 within 1e-3 (the road at least the straight
        // line), so the far pairs need no logit (one per pair took seconds to open a pack).
        let missing = |d_km: f32| 1.0 - (1.0 - p.walk_share(d_km)) * oq.at(d_km);
        let mut r = 1;
        while r < z.w.max(z.h) && missing((r - 1) as f32 * z.zone_m / 1000.0) > 1e-3 {
            r += 1;
        }
        let (off, at) = z.by_square();
        let keys: Vec<i32> = (0..z.n).map(|j| self.key(z, j)).collect();
        for i in 0..z.n {
            if self.row[i] == 0.0 || cz[i] <= 0.0 {
                continue;
            }
            let base = self.base(z, i);
            let ri = self.row[i] * cz[i] as f64;
            let reg = z.region[i];
            let mut hi = 0.0;
            for ((w, (&c, &key)), &rj) in work.iter_mut().zip(self.col.iter().zip(&keys)).zip(&z.region) {
                let f = self.ftab[(base + key) as usize] * if rj == reg { 1.0 } else { self.border };
                let t = ri * c * f as f64;
                hi += t;
                *w += t;
            }
            home[i] += hi;
            for gx in (z.gx[i] - r).max(0)..=(z.gx[i] + r).min(z.w - 1) {
                for gy in (z.gy[i] - r).max(0)..=(z.gy[i] + r).min(z.h - 1) {
                    let s = (gx * z.h + gy) as usize;
                    let k = (base + gx * self.th + gy) as usize;
                    for &j in &at[off[s] as usize..off[s + 1] as usize] {
                        let j = j as usize;
                        let w = if self.wtab.is_empty() { 0.0 } else { self.walk_share_at(i, j, k) };
                        let q = oq.at(dtab[k] * p.drive_factor(i, j));
                        let t = ri * self.col[j] * (self.ftab[k] * self.k(z, i, j) * (1.0 - (1.0 - w) * q)) as f64;
                        home[i] -= t;
                        work[j] -= t;
                    }
                }
            }
        }
        (home, work)
    }

    /// All-day trips that would walk with no rail at all: sum over zone pairs of T_IJ times the
    /// walk share at their distance.
    pub fn walk_trips(&self, z: &Zones) -> f64 {
        self.walk_trips_by_zone(z).0.iter().sum()
    }

    /// The same by home zone and by work zone. Only offsets where the share is not negligible
    /// are visited.
    pub fn walk_trips_by_zone(&self, z: &Zones) -> (Vec<f64>, Vec<f64>) {
        let (mut home, mut work) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        if self.wtab.is_empty() {
            return (home, work);
        }
        let (off, at) = z.by_square();
        // reach: the largest offset (in zones) along either axis whose share is above 1e-6
        let mut r = 0;
        while r < z.w.max(z.h) && self.wtab[((r + self.offx) * self.th + self.offy) as usize] > 1e-6 {
            r += 1;
        }
        for i in 0..z.n {
            if self.row[i] == 0.0 {
                continue;
            }
            let base = self.base(z, i);
            for gx in (z.gx[i] - r).max(0)..=(z.gx[i] + r).min(z.w - 1) {
                for gy in (z.gy[i] - r).max(0)..=(z.gy[i] + r).min(z.h - 1) {
                    let s = (gx * z.h + gy) as usize;
                    let k = (base + gx * self.th + gy) as usize;
                    for &j in &at[off[s] as usize..off[s + 1] as usize] {
                        let j = j as usize;
                        let t = self.row[i] * self.col[j] * (self.ftab[k] * self.k(z, i, j) * self.walk_share_at(i, j, k)) as f64;
                        home[i] += t;
                        work[j] += t;
                    }
                }
            }
        }
        (home, work)
    }
}

/// Decay f and distance d (km) by zone grid offset, index (dx + w - 1) * (2h - 1) + (dy + h - 1).
/// Zone-centre distance; within a zone 0.52 x the zone side (mean distance between two random
/// points of a square). The one definition of the decay: the pipeline's solve and the pack
/// loader both use it, so shipped factors and rebuilt trips agree exactly.
pub fn decay_table(w: i32, h: i32, zone_m: f32, decay_km: f32, decay_pow: f32) -> (Vec<f32>, Vec<f32>) {
    let (tw, th) = (2 * w - 1, 2 * h - 1);
    let (offx, offy) = (w - 1, h - 1);
    let zk = zone_m / 1000.0;
    let mut ftab = vec![0f32; (tw * th) as usize];
    let mut dtab = vec![0f32; (tw * th) as usize];
    for dx in -offx..=offx {
        for dy in -offy..=offy {
            let d = if dx == 0 && dy == 0 { 0.52 * zk } else { zk * ((dx * dx + dy * dy) as f32).sqrt() };
            let i = ((dx + offx) * th + dy + offy) as usize;
            ftab[i] = (-d / decay_km).exp() * if decay_pow != 0.0 { d.powf(-decay_pow) } else { 1.0 };
            dtab[i] = d;
        }
    }
    (ftab, dtab)
}

/// Zones and gravity for a city: from the pack when it ships them for these parameters (city
/// open, a few ms), otherwise solved here (seconds). Returns whether the pack's were used.
pub fn city_gravity(city: &City, p: &Params) -> (Zones, Gravity, bool) {
    if let Some(pg) = &city.gravity {
        if (pg.zone_m - p.zone_m).abs() < 0.5 && (pg.decay_km - p.decay_km).abs() < 1e-6 && (pg.decay_pow - p.decay_pow).abs() < 1e-6 {
            let z = zones_from_pack(city, pg);
            let mut g = gravity_from_pack(&z, pg);
            g.set_walk(&z, p);
            return (z, g, true);
        }
    }
    let z = build_zones(city, p.zone_m);
    let mut g = gravity(&z, p, 100, 1e-3);
    g.set_walk(&z, p);
    (z, g, false)
}

pub fn zones_from_pack(city: &City, pg: &PackGravity) -> Zones {
    let n = pg.gx.len();
    let mut z = Zones {
        zone_m: pg.zone_m,
        n,
        cell_zone: pg.cell_zone.clone(),
        gx: pg.gx.clone(),
        gy: pg.gy.clone(),
        region: if pg.zone_region.len() == n { pg.zone_region.clone() } else { vec![0; n] },
        w: pg.grid_w,
        h: pg.grid_h,
        x0: pg.grid_x0,
        y0: pg.grid_y0,
        pop: vec![0.0; n],
        jobs: vec![0.0; n],
        ncell: vec![0; n],
    };
    for c in 0..city.len() {
        let zi = z.cell_zone[c] as usize;
        z.ncell[zi] += res9_cells(city, c);
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    if pg.cells9.len() == n {
        z.ncell.clone_from(&pg.cells9);
    }
    z
}

/// Cell `c`'s land in resolution 9 cells: 7 for a merged res-8 cell (T-044), else 1 (synthetic
/// cities' ids are not H3).
fn res9_cells(city: &City, c: usize) -> u32 {
    match city.h3.get(c).map(|&h| (h >> 52) & 0xf) {
        Some(8) => 7,
        _ => 1,
    }
}

pub fn gravity_from_pack(z: &Zones, pg: &PackGravity) -> Gravity {
    let (ftab, _) = decay_table(z.w, z.h, z.zone_m, pg.decay_km, pg.decay_pow);
    Gravity {
        row: pg.row.clone(),
        col: pg.col.clone(),
        border: pg.border_factor,
        ftab,
        wtab: vec![],
        wdtab: vec![],
        wroutes: None,
        wcurve: (0.0, 1.0),
        th: 2 * z.h - 1,
        offx: z.w - 1,
        offy: z.h - 1,
        iters: pg.iters,
        max_err: pg.max_err,
        mean_km: pg.mean_km,
        total: pg.trips,
    }
}

/// What the pipeline ships (pack format 1).
pub fn to_pack_gravity(z: &Zones, g: &Gravity, p: &Params) -> PackGravity {
    PackGravity {
        zone_m: z.zone_m,
        decay_km: p.decay_km,
        decay_pow: p.decay_pow,
        border_factor: g.border,
        grid_x0: z.x0,
        grid_y0: z.y0,
        grid_w: z.w,
        grid_h: z.h,
        cell_zone: z.cell_zone.clone(),
        gx: z.gx.clone(),
        gy: z.gy.clone(),
        // one region: nothing to ship (and the pack reads as before T-162)
        zone_region: if z.region.iter().any(|&r| r != z.region[0]) { z.region.clone() } else { vec![] },
        row: g.row.clone(),
        col: g.col.clone(),
        iters: g.iters,
        max_err: g.max_err,
        mean_km: g.mean_km,
        trips: g.total,
        // the pack's own gravity: every commute stays in it (T-276's joint solve sets these)
        home: vec![],
        work: vec![],
        joint: None,
        cells9: vec![],
    }
}

/// Doubly constrained gravity by Furness balancing. Workers = population scaled to total jobs.
/// The decay depends on zone-centre distance only, so f is a small table by grid offset; it is
/// multiplied by `p.border_factor` between zones of different regions (T-162).
pub fn gravity(z: &Zones, p: &Params, max_iters: usize, tol: f64) -> Gravity {
    let (w, h) = (z.w, z.h);
    let th = 2 * h - 1;
    let (offx, offy) = (w - 1, h - 1);
    let (ftab, dtab) = decay_table(w, h, z.zone_m, p.decay_km, p.decay_pow);
    let tp: f64 = z.pop.iter().sum();
    let tj: f64 = z.jobs.iter().sum();
    let o: Vec<f64> = z.pop.iter().map(|&v| v * tj / tp).collect();
    let rows: Vec<usize> = (0..z.n).filter(|&i| o[i] > 0.0).collect();
    let cols: Vec<usize> = (0..z.n).filter(|&j| z.jobs[j] > 0.0).collect();
    let ckey: Vec<i32> = cols.iter().map(|&j| z.gx[j] * th + z.gy[j]).collect();
    let creg: Vec<u8> = cols.iter().map(|&j| z.region[j]).collect();
    // the factor by "regions differ" (index 0 same, 1 different)
    let kf = [1.0f64, p.border_factor as f64];
    let mut g = Gravity { row: vec![0.0; z.n], col: z.jobs.clone(), border: p.border_factor, ftab, wtab: vec![], wdtab: vec![], wroutes: None, wcurve: (0.0, 1.0), th, offx, offy, iters: 0, max_err: 0.0, mean_km: 0.0, total: 0.0 };
    let mut colc: Vec<f64> = cols.iter().map(|&j| g.col[j]).collect();
    let mut acc = vec![0f64; cols.len()];
    for it in 0..max_iters {
        // rows: A_I = 1 / sum_J col_J f_IJ k_IJ
        let mut err: f64 = 0.0;
        for &i in &rows {
            let base = g.base(z, i);
            let ri = z.region[i];
            let mut s = 0.0f64;
            for (k, &key) in ckey.iter().enumerate() {
                s += colc[k] * g.ftab[(base + key) as usize] as f64 * kf[(creg[k] != ri) as usize];
            }
            if it > 0 {
                err = err.max((g.row[i] * s / o[i] - 1.0).abs());
            }
            g.row[i] = o[i] / s;
        }
        g.iters = it + 1;
        g.max_err = err;
        if it > 0 && err < tol {
            break;
        }
        // columns: B_J = 1 / sum_I row_I f_IJ k_IJ
        acc.iter_mut().for_each(|v| *v = 0.0);
        for &i in &rows {
            let base = g.base(z, i);
            let ri = z.region[i];
            let r = g.row[i];
            for (k, &key) in ckey.iter().enumerate() {
                acc[k] += r * g.ftab[(base + key) as usize] as f64 * kf[(creg[k] != ri) as usize];
            }
        }
        for (k, &j) in cols.iter().enumerate() {
            colc[k] = z.jobs[j] / acc[k];
            g.col[j] = colc[k];
        }
    }
    // mean trip length
    let (mut sum, mut sumd) = (0.0f64, 0.0f64);
    for &i in &rows {
        let base = g.base(z, i);
        let ri = z.region[i];
        for (k, &key) in ckey.iter().enumerate() {
            let idx = (base + key) as usize;
            let t = g.row[i] * colc[k] * g.ftab[idx] as f64 * kf[(creg[k] != ri) as usize];
            sum += t;
            sumd += t * dtab[idx] as f64;
        }
    }
    g.total = sum;
    g.mean_km = sumd / sum;
    g
}

// ---------------------------------------------------------------------------------------------
// 2. Station access per cell

pub struct Access {
    pub n: Vec<u8>,
    /// K per cell, nearest first.
    pub st: Vec<u32>,
    /// Real walk minutes, K per cell.
    pub walk: Vec<f32>,
    pub cells_with_access: usize,
    /// Per station, the cells whose walk to it goes around water (T-275,
    /// `routes::Routes::station_walks`), with their walking metres; None without routes.
    pub ex: Vec<Option<std::sync::Arc<Vec<(u32, f32)>>>>,
}

/// Station lookup grid with buckets the size of the walk cutoff.
struct StationGrid {
    minx: f32,
    miny: f32,
    size: f32,
    w: i32,
    h: i32,
    off: Vec<u32>,
    items: Vec<u32>,
}

impl StationGrid {
    /// Buckets of `size` metres over the stations' extent plus `pad` metres all round.
    fn new(net: &Network, size: f32, pad: f32) -> Self {
        let minx = net.st_x.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let miny = net.st_y.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let maxx = net.st_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let maxy = net.st_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let w = ((maxx - minx) / size) as i32 + 1;
        let h = ((maxy - miny) / size) as i32 + 1;
        let b = |s: usize| (((net.st_x[s] - minx) / size) as i32 * h + ((net.st_y[s] - miny) / size) as i32) as usize;
        let mut off = vec![0u32; (w * h + 1) as usize];
        for s in 0..net.n_stations() {
            off[b(s) + 1] += 1;
        }
        for i in 0..(w * h) as usize {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; net.n_stations()];
        for s in 0..net.n_stations() {
            let k = b(s);
            items[fill[k] as usize] = s as u32;
            fill[k] += 1;
        }
        StationGrid { minx, miny, size, w, h, off, items }
    }
    fn near(&self, x: f32, y: f32, mut f: impl FnMut(u32)) {
        let bx = ((x - self.minx) / self.size).floor() as i32;
        let by = ((y - self.miny) / self.size).floor() as i32;
        for gx in (bx - 1).max(0)..=(bx + 1).min(self.w - 1) {
            for gy in (by - 1).max(0)..=(by + 1).min(self.h - 1) {
                let k = (gx * self.h + gy) as usize;
                for &s in &self.items[self.off[k] as usize..self.off[k + 1] as usize] {
                    f(s);
                }
            }
        }
    }
}

/// Bucket size of the cell grid the access search scatters stations over.
const ACCESS_BUCKET_M: f32 = 1000.0;

/// K nearest stations per cell within `walk_cutoff_m`, by scattering each station over the
/// cells near it (a station touches the few hundred cells within the bound, so a far bound
/// costs little; searching outwards from every cell cost 5-15x as much where stations are
/// sparse).
pub fn build_access(city: &City, net: &Network, p: &Params) -> Access {
    let n = city.len();
    let mut a = Access { n: vec![0; n], st: vec![NONE; n * K], walk: vec![f32::INFINITY; n * K], cells_with_access: 0, ex: vec![None; net.n_stations()] };
    if net.n_stations() == 0 || n == 0 {
        return a;
    }
    // cells bucketed on a grid (CSR)
    let size = ACCESS_BUCKET_M;
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / size) as i32 + 1;
    let h = ((maxy - miny) / size) as i32 + 1;
    let bucket = |x: f32, y: f32| ((((x - minx) / size) as i32).clamp(0, w - 1) * h + (((y - miny) / size) as i32).clamp(0, h - 1)) as usize;
    let mut off = vec![0u32; (w * h + 1) as usize];
    for c in 0..n {
        off[bucket(city.x[c], city.y[c]) + 1] += 1;
    }
    for i in 0..(w * h) as usize {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut cells = vec![0u32; n];
    for c in 0..n {
        let b = bucket(city.x[c], city.y[c]);
        cells[fill[b] as usize] = c as u32;
        fill[b] += 1;
    }
    drop(fill);
    // each station into the sorted K lists (distance squared for now) of the cells near it
    let cut = p.walk_cutoff_m;
    let cut2 = cut * cut;
    for s in 0..net.n_stations() {
        let (sx, sy) = (net.st_x[s], net.st_y[s]);
        let gx0 = (((sx - cut - minx) / size).floor() as i32).max(0);
        let gx1 = (((sx + cut - minx) / size).floor() as i32).min(w - 1);
        let gy0 = (((sy - cut - miny) / size).floor() as i32).max(0);
        let gy1 = (((sy + cut - miny) / size).floor() as i32).min(h - 1);
        // T-275: the cells whose walk to this station goes around water (searched once a station
        // position, kept for the thread)
        let ex = p.routes.as_ref().and_then(|r| {
            r.station_walks(sx, sy, cut, |f| {
                for gx in gx0..=gx1 {
                    for gy in gy0..=gy1 {
                        let k = (gx * h + gy) as usize;
                        for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                            f(c, city.x[c as usize], city.y[c as usize]);
                        }
                    }
                }
            })
        });
        a.ex[s] = ex.clone();
        let exl: &[(u32, f32)] = ex.as_deref().map_or(&[], |v| &v[..]);
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                let k = (gx * h + gy) as usize;
                for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                    let mut d2 = (city.x[c as usize] - sx).powi(2) + (city.y[c as usize] - sy).powi(2);
                    if !exl.is_empty() && d2 <= cut2 {
                        if let Ok(e) = exl.binary_search_by_key(&c, |e| e.0) {
                            let m = exl[e].1;
                            if m > cut {
                                continue; // around the water it is beyond the walk's bound
                            }
                            d2 = m * m;
                        }
                    }
                    let c = c as usize;
                    let cnt = a.n[c] as usize;
                    let wk = &mut a.walk[c * K..c * K + K];
                    if d2 > cut2 || (cnt == K && d2 >= wk[K - 1]) {
                        continue;
                    }
                    let st = &mut a.st[c * K..c * K + K];
                    let mut i = if cnt < K { cnt } else { K - 1 };
                    while i > 0 && wk[i - 1] > d2 {
                        wk[i] = wk[i - 1];
                        st[i] = st[i - 1];
                        i -= 1;
                    }
                    wk[i] = d2;
                    st[i] = s as u32;
                    if cnt < K {
                        a.n[c] += 1;
                    }
                }
            }
        }
    }
    for c in 0..n {
        let cnt = a.n[c] as usize;
        for wv in a.walk[c * K..c * K + cnt].iter_mut() {
            *wv = p.walk_min(wv.sqrt());
        }
        if cnt > 0 {
            a.cells_with_access += 1;
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------
// 3. Access subzones

pub struct Subzones {
    pub n: usize,
    pub zone: Vec<u32>,
    pub pop: Vec<f32>,
    pub jobs: Vec<f32>,
    pub cx: Vec<f32>,
    pub cy: Vec<f32>,
    /// Share of its homes without a car (`Params::nocar`, home-weighted; T-511).
    pub nocar: Vec<f32>,
    /// CSR access list, per subzone its walk entries (`walk`) then its ride entries (`ride`,
    /// T-371): station, perceived minutes, the distance in metres and the real minutes. A walk
    /// entry's minutes are the members' `Params::access_cost` (a pop/jobs-weighted logsum) and
    /// its distance their weighted mean (straight, or around water with the pack's routes,
    /// T-275); a ride entry's are its zone's (`zone_rides`). Walk entries serve both ends of a
    /// trip, ride entries only the home end.
    pub acc_off: Vec<u32>,
    pub acc_mid: Vec<u32>,
    pub acc_st: Vec<u32>,
    pub acc_w: Vec<f32>,
    pub acc_d: Vec<f32>,
    pub acc_t: Vec<f32>,
    /// CSR: a ride-only subzone's homes by zone (zone, homes); empty for the others, whose homes
    /// are all in `zone`.
    pub home_off: Vec<u32>,
    pub home_zones: Vec<(u32, f32)>,
    /// Per zone, subzones with commuters living there (origins) and with jobs (destinations).
    pub zo_off: Vec<u32>,
    pub zo: Vec<u32>,
    pub zd_off: Vec<u32>,
    pub zd: Vec<u32>,
    /// Subzone of each cell (NONE for cells with no station within reach). The cells with homes
    /// beyond every walk but within a ride are one subzone per `RIDE_GROUP` square and region,
    /// with no walk entries and no jobs: origins only, in their most populous zone (`zone`).
    pub cell: Vec<u32>,
    /// CSR: the cells of each subzone (`mem[mem_off[u]..mem_off[u + 1]]`).
    pub mem_off: Vec<u32>,
    pub mem: Vec<u32>,
}

impl Subzones {
    /// Subzone `u`'s walk entries (`acc_st` indices).
    #[inline]
    pub fn walk(&self, u: usize) -> std::ops::Range<usize> {
        self.acc_off[u] as usize..self.acc_mid[u] as usize
    }
    /// Its ride entries, home end only (T-371).
    #[inline]
    pub fn ride(&self, u: usize) -> std::ops::Range<usize> {
        self.acc_mid[u] as usize..self.acc_off[u + 1] as usize
    }
    /// Subzone `u`'s part of the home -> work trips of the period (`share` of the day): its
    /// homes' share of their zones' gravity row factors. A ride-only subzone's homes keep their
    /// own zones' totals, and its trips take its `zone`'s distance decay.
    pub fn origin_weight(&self, u: usize, z: &Zones, gr: &Gravity, share: f64) -> f32 {
        let h = &self.home_zones[self.home_off[u] as usize..self.home_off[u + 1] as usize];
        if h.is_empty() {
            let zi = self.zone[u] as usize;
            return if z.pop[zi] > 0.0 { (gr.row[zi] * self.pop[u] as f64 / z.pop[zi] * share) as f32 } else { 0.0 };
        }
        h.iter().map(|&(zi, pop)| if z.pop[zi as usize] > 0.0 { gr.row[zi as usize] * pop as f64 / z.pop[zi as usize] * share } else { 0.0 }).sum::<f64>() as f32
    }
}

/// Whether a commute between zones `i` and `j` is long enough to be offered the ride (T-371).
#[inline]
pub fn ride_offered(z: &Zones, p: &Params, i: usize, j: usize) -> bool {
    let (dx, dy) = ((z.gx[i] - z.gx[j]) as f32, (z.gy[i] - z.gy[j]) as f32);
    (dx * dx + dy * dy) * z.zone_m * z.zone_m >= (p.ride_min_km * 1000.0).powi(2)
}

/// A ride entry (T-371): station, perceived minutes, straight metres, real drive minutes.
pub type Ride = (u32, f32, f32, f32);

/// The end of a trip a ride serves (T-373): at home the rider may park at the station or be
/// dropped off; at the far end there is no car to park, only a drop-off (`d_drop`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum End {
    Home,
    Far,
}

/// The ride options from each place (its zone, x, y), CSR, for every layer (local commutes and
/// everyday trips, the regional market, T-362's long trips): walks come from `build_access` per
/// cell, rides from here. For each line, its station with the cheapest ride within `ride_max_m`;
/// then the stations of the `ride_lines` cheapest lines. The drive is the whole-trip drive's
/// (`drive_min_f` with the zones' routed factor) from the place, in its zone, to the station's
/// zone; at the station, the cheaper of parking there and `d_drop` at the home end, `d_drop` at
/// the far end.
pub fn access_options(zones: &Zones, net: &Network, p: &Params, places: &[(usize, f32, f32)], end: End) -> (Vec<u32>, Vec<Ride>) {
    let mut off = vec![0u32; places.len() + 1];
    let mut out = vec![];
    if p.ride_lines == 0 || net.n_stations() == 0 {
        return (off, out);
    }
    let dens = zones.density();
    let place = |z: Option<usize>| p.place_min_per_km(z.map_or(0.0, |z| dens[z]));
    // parking as at a destination, its mean over drivers
    let station_end = |z: Option<usize>| match end {
        End::Far => p.d_drop,
        End::Home => z.map_or(p.park_min, |z| p.park_mean(zones, z)).min(p.d_drop),
    };
    let (sq_off, sq) = zones.by_square();
    let st_zone: Vec<Option<usize>> = (0..net.n_stations())
        .map(|s| {
            let gx = ((net.st_x[s] - zones.x0) / zones.zone_m).floor() as i32;
            let gy = ((net.st_y[s] - zones.y0) / zones.zone_m).floor() as i32;
            if gx < 0 || gy < 0 || gx >= zones.w || gy >= zones.h {
                return None;
            }
            let k = (gx * zones.h + gy) as usize;
            (sq_off[k] < sq_off[k + 1]).then(|| sq[sq_off[k] as usize] as usize)
        })
        .collect();
    let mut st_lines: Vec<Vec<u32>> = vec![vec![]; net.n_stations()];
    for l in 0..net.n_lines() {
        for &s in net.stops(l) {
            if st_lines[s as usize].last() != Some(&(l as u32)) {
                st_lines[s as usize].push(l as u32);
            }
        }
    }
    let grid = StationGrid::new(net, p.ride_max_m, p.ride_max_m);
    let max2 = p.ride_max_m * p.ride_max_m;
    // per line: (cost, station, metres, minutes), or none yet from this place
    let mut best: Vec<(f32, u32, f32, f32)> = vec![(f32::INFINITY, NONE, 0.0, 0.0); net.n_lines()];
    let mut seen: Vec<u32> = vec![];
    for (pi, &(i, x, y)) in places.iter().enumerate() {
        {
            let a = place(Some(i));
            grid.near(x, y, |s| {
                let s = s as usize;
                let d2 = (net.st_x[s] - x).powi(2) + (net.st_y[s] - y).powi(2);
                if d2 > max2 || st_lines[s].is_empty() {
                    return;
                }
                let d = d2.sqrt();
                let j = st_zone[s];
                let f = j.map_or(p.road_detour, |j| p.drive_factor(i, j));
                let t = p.drive_min_f(d, f, a, place(j));
                let cost = p.p_ride + p.w_drive * t + station_end(j);
                for &l in &st_lines[s] {
                    let b = &mut best[l as usize];
                    if b.1 == NONE {
                        seen.push(l);
                    }
                    if cost < b.0 {
                        *b = (cost, s as u32, d, t);
                    }
                }
            });
            seen.sort_by(|&a, &b| best[a as usize].0.total_cmp(&best[b as usize].0));
            let start = out.len();
            for &l in seen.iter().take(p.ride_lines) {
                let (cost, s, d, t) = best[l as usize];
                if !out[start..].iter().any(|e: &Ride| e.0 == s) {
                    out.push((s, cost, d, t));
                }
            }
            for &l in &seen {
                best[l as usize] = (f32::INFINITY, NONE, 0.0, 0.0);
            }
            seen.clear();
        }
        off[pi + 1] = out.len() as u32;
    }
    (off, out)
}

/// Ride-only subzones group homes beyond every walk by squares of this many zones a side
/// (T-371: 6 km, as the market's sites; one per 2 km zone cost the real New York day a third
/// more solve time).
const RIDE_GROUP: i32 = 3;

pub fn build_subzones(city: &City, zones: &Zones, acc: &Access, net: &Network, p: &Params) -> Subzones {
    build_subzones_on(city, &city.pop, &city.jobs, zones, acc, net, p)
}

/// `build_subzones` on home and work weights other than the city's own (`zones` summing them).
pub fn build_subzones_on(city: &City, home_w: &[f32], work_w: &[f32], zones: &Zones, acc: &Access, net: &Network, p: &Params) -> Subzones {
    let n = city.len();
    let mut cell_sub = vec![NONE; n];
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut nsub = 0u32;
    // homes beyond every walk, by group square and region (T-371): per group its homes, their
    // centre, and its homes by zone
    let mut group_of: HashMap<u64, u32> = HashMap::new();
    let mut cell_group = vec![NONE; n];
    let mut gsum: Vec<(f64, f64, f64)> = vec![];
    let mut gzone: std::collections::BTreeMap<(u32, u32), f64> = Default::default();
    let (mut zx, mut zy) = (vec![0f64; zones.n], vec![0f64; zones.n]);
    for c in 0..n {
        let zi = zones.cell_zone[c] as usize;
        let pop = home_w[c] as f64;
        zx[zi] += pop * city.x[c] as f64;
        zy[zi] += pop * city.y[c] as f64;
        if acc.n[c] == 0 {
            if pop > 0.0 && p.ride_lines > 0 {
                let key = ((zones.gx[zi] / RIDE_GROUP) as u64) << 40 | ((zones.gy[zi] / RIDE_GROUP) as u64) << 8 | zones.region[zi] as u64;
                let g = *group_of.entry(key).or_insert_with(|| {
                    gsum.push((0.0, 0.0, 0.0));
                    gsum.len() as u32 - 1
                });
                cell_group[c] = g;
                let e = &mut gsum[g as usize];
                e.0 += pop;
                e.1 += pop * city.x[c] as f64;
                e.2 += pop * city.y[c] as f64;
                *gzone.entry((g, zi as u32)).or_default() += pop;
            }
            continue;
        }
        if home_w[c] + work_w[c] <= 0.0 {
            continue; // nobody commutes from or to here
        }
        let key = if p.band_min < 0.0 {
            c as u64
        } else {
            let band = if p.band_min > 0.0 { (acc.walk[c * K] / p.band_min) as u64 } else { 0 };
            ((zones.cell_zone[c] as u64) << 32) | ((acc.st[c * K] as u64) << 8) | band.min(255)
        };
        let id = *map.entry(key).or_insert_with(|| {
            nsub += 1;
            nsub - 1
        });
        cell_sub[c] = id;
    }
    drop(map);
    let n_walk = nsub as usize;
    // rides from each zone's home-weighted centre (for its walk subzones), then from each group's
    let mut places: Vec<(usize, f32, f32)> = (0..zones.n)
        .map(|i| {
            let w = zones.pop[i].max(1e-9);
            (i, (zx[i] / w) as f32, (zy[i] / w) as f32)
        })
        .collect();
    // a group's zone is its most populous one, its homes by zone `gz`
    let mut g_rep = vec![(NONE, 0f64); gsum.len()];
    let mut gz_off = vec![0u32; gsum.len() + 1];
    let mut gz: Vec<(u32, f32)> = vec![];
    for (&(g, zi), &pop) in &gzone {
        let r = &mut g_rep[g as usize];
        if pop > r.1 {
            *r = (zi, pop);
        }
        gz_off[g as usize + 1] += 1;
        gz.push((zi, pop as f32));
    }
    for g in 0..gsum.len() {
        gz_off[g + 1] += gz_off[g];
    }
    for (g, &(pop, x, y)) in gsum.iter().enumerate() {
        places.push((g_rep[g].0 as usize, (x / pop) as f32, (y / pop) as f32));
    }
    let (ride_off, rides) = access_options(zones, net, p, &places, End::Home);
    let place_ride = |pl: usize| &rides[ride_off[pl] as usize..ride_off[pl + 1] as usize];
    // a group with rides is a subzone
    let mut group_sub = vec![NONE; gsum.len()];
    for g in 0..gsum.len() {
        if !place_ride(zones.n + g).is_empty() {
            group_sub[g] = nsub;
            nsub += 1;
        }
    }
    for c in 0..n {
        if cell_group[c] != NONE {
            cell_sub[c] = group_sub[cell_group[c] as usize];
        }
    }
    let ns = nsub as usize;
    let mut sub_group = vec![NONE; ns - n_walk];
    for (g, &u) in group_sub.iter().enumerate() {
        if u != NONE {
            sub_group[u as usize - n_walk] = g as u32;
        }
    }
    // members CSR
    let mut moff = vec![0u32; ns + 1];
    for c in 0..n {
        if cell_sub[c] != NONE {
            moff[cell_sub[c] as usize + 1] += 1;
        }
    }
    for i in 0..ns {
        moff[i + 1] += moff[i];
    }
    let mut fill = moff.clone();
    let mut mem = vec![0u32; moff[ns] as usize];
    for c in 0..n {
        let s = cell_sub[c];
        if s != NONE {
            mem[fill[s as usize] as usize] = c as u32;
            fill[s as usize] += 1;
        }
    }
    let mut sz = Subzones {
        n: ns,
        zone: vec![0; ns],
        pop: vec![0.0; ns],
        jobs: vec![0.0; ns],
        cx: vec![0.0; ns],
        cy: vec![0.0; ns],
        nocar: vec![0.0; ns],
        acc_off: vec![0],
        acc_mid: vec![],
        acc_st: vec![],
        acc_w: vec![],
        acc_d: vec![],
        acc_t: vec![],
        home_off: vec![0],
        home_zones: vec![],
        zo_off: vec![],
        zo: vec![],
        zd_off: vec![],
        zd: vec![],
        cell: vec![],
        mem_off: vec![],
        mem: vec![],
    };
    // candidates: (mean perceived minutes, station, mean straight-line metres)
    let mut cand: Vec<(f32, u32, f32)> = Vec::with_capacity(32);
    for s in 0..ns {
        let m = &mem[moff[s] as usize..moff[s + 1] as usize];
        let walks = acc.n[m[0] as usize] > 0;
        let (mut wsum, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
        for &c in m {
            let c = c as usize;
            let jobs = if walks { work_w[c] } else { 0.0 };
            let wt = (home_w[c] + jobs).max(1e-6) as f64;
            wsum += wt;
            x += wt * city.x[c] as f64;
            y += wt * city.y[c] as f64;
            sz.pop[s] += home_w[c];
            sz.jobs[s] += jobs;
            if let Some(nc) = &p.nocar {
                sz.nocar[s] += home_w[c] * nc[c];
            }
        }
        sz.nocar[s] = if sz.pop[s] > 0.0 { (sz.nocar[s] / sz.pop[s]).min(1.0) } else { 0.0 };
        sz.cx[s] = (x / wsum) as f32;
        sz.cy[s] = (y / wsum) as f32;
        if !walks {
            let g = sub_group[s - n_walk] as usize;
            sz.zone[s] = g_rep[g].0;
            sz.acc_mid.push(sz.acc_st.len() as u32);
            for &(st, w, d, t) in place_ride(zones.n + g) {
                sz.acc_st.push(st);
                sz.acc_w.push(w);
                sz.acc_d.push(d);
                sz.acc_t.push(t);
            }
            sz.acc_off.push(sz.acc_st.len() as u32);
            sz.home_zones.extend_from_slice(&gz[gz_off[g] as usize..gz_off[g + 1] as usize]);
            sz.home_off.push(sz.home_zones.len() as u32);
            continue;
        }
        let zi = zones.cell_zone[m[0] as usize];
        sz.zone[s] = zi;
        sz.home_off.push(sz.home_zones.len() as u32);
        let zone_rides = || place_ride(zi as usize).iter();
        // candidates: union of the members' lists; distance = weighted mean over members
        cand.clear();
        for &c in m {
            let c = c as usize;
            for k in 0..acc.n[c] as usize {
                let st = acc.st[c * K + k];
                if !cand.iter().any(|&(_, t, _)| t == st) {
                    cand.push((0.0, st, 0.0));
                }
            }
        }
        for e in cand.iter_mut() {
            let (sx, sy) = (net.st_x[e.1 as usize], net.st_y[e.1 as usize]);
            let ex = acc.ex.get(e.1 as usize).and_then(|v| v.as_ref());
            let (mut wd, mut wc) = (0.0f64, 0.0f64);
            for &c in m {
                let wt = (home_w[c as usize] + work_w[c as usize]).max(1e-6) as f64;
                let d = ((city.x[c as usize] - sx).powi(2) + (city.y[c as usize] - sy).powi(2)).sqrt();
                // T-275: around water where the walk must go round (cut off: no weight)
                let d = super::routes::walk_m(ex, c, d);
                if !d.is_finite() {
                    wd += wt * p.walk_cutoff_m as f64;
                    continue;
                }
                wd += wt * d as f64;
                wc += wt * (-(p.beta * p.access_cost(d)) as f64).exp();
            }
            e.2 = (wd / wsum) as f32;
            // the members' perceived walks averaged as the mode choice would weigh them (a
            // logsum at its scale): the cost grows faster than the distance, and the near
            // members carry most of the rail (T-090: a plain mean put rail 3.3% under one
            // subzone per cell on the hand-made network, this 0.1% over)
            e.0 = (-(wc / wsum).ln() / p.beta as f64) as f32;
        }
        cand.retain(|e| e.2 <= p.walk_cutoff_m && e.0.is_finite());
        if cand.is_empty() {
            // every member is within reach of its own nearest station, so the mean distance to
            // that station is too; this only guards against rounding
            let c = m[0] as usize;
            let d = acc.walk[c * K] * p.walk_mps * 60.0 / p.walk_detour;
            cand.push((p.access_cost(d), acc.st[c * K], d));
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(wm, st, d) in cand.iter().take(K) {
            sz.acc_st.push(st);
            sz.acc_w.push(wm);
            sz.acc_d.push(d);
            sz.acc_t.push(p.walk_min(d));
        }
        sz.acc_mid.push(sz.acc_st.len() as u32);
        if sz.pop[s] > 0.0 {
            for &(st, w, d, t) in zone_rides() {
                sz.acc_st.push(st);
                sz.acc_w.push(w);
                sz.acc_d.push(d);
                sz.acc_t.push(t);
            }
        }
        sz.acc_off.push(sz.acc_st.len() as u32);
    }
    // zone -> subzone lists
    let csr = |want: &dyn Fn(usize) -> bool| {
        let mut off = vec![0u32; zones.n + 1];
        for s in 0..ns {
            if want(s) {
                off[sz.zone[s] as usize + 1] += 1;
            }
        }
        for i in 0..zones.n {
            off[i + 1] += off[i];
        }
        let mut f = off.clone();
        let mut l = vec![0u32; off[zones.n] as usize];
        for s in 0..ns {
            if want(s) {
                let z = sz.zone[s] as usize;
                l[f[z] as usize] = s as u32;
                f[z] += 1;
            }
        }
        (off, l)
    };
    let (zo_off, zo) = csr(&|s| sz.pop[s] > 0.0);
    let (zd_off, mut zd) = csr(&|s| sz.jobs[s] > 0.0);
    // destinations in each zone by their cheapest walk, so mode choice can stop at
    // `Params::pair_walk_max`
    for j in 0..zones.n {
        zd[zd_off[j] as usize..zd_off[j + 1] as usize].sort_by(|&a, &b| sz.acc_w[sz.acc_off[a as usize] as usize].partial_cmp(&sz.acc_w[sz.acc_off[b as usize] as usize]).unwrap());
    }
    sz.zo_off = zo_off;
    sz.zo = zo;
    sz.zd_off = zd_off;
    sz.zd = zd;
    sz.cell = cell_sub;
    sz.mem_off = moff;
    sz.mem = mem;
    sz
}

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
    /// leave riders on the platform (T-188, `left_behind`); 0 at free flow. Added to the
    /// boarding at the platform wait's weight.
    pub rn_left: Vec<f32>,
    /// Incoming edges, CSR, for destination-based strategy search.
    pub rev_off: Vec<u32>,
    pub rev_from: Vec<u32>,
    pub rev_edge: Vec<u32>,
    pub adj_off: Vec<u32>,
    pub adj_to: Vec<u32>,
    pub adj_cost: Vec<f32>,
    /// For ride edges, the route node the segment starts at; NONE otherwise.
    pub adj_seg: Vec<u32>,
}

pub fn board_cost(headway_min: f32, p: &Params) -> f32 {
    let half = headway_min / 2.0;
    half.min(p.platform_wait_cap_min) * p.w_wait + (half - p.platform_wait_cap_min).max(0.0) * p.w_home_wait + p.transfer_min
}

pub fn build_graph(net: &Network, p: &Params) -> Graph {
    let s = net.n_stations();
    let mut rn_station = vec![];
    let mut rn_line = vec![];
    let mut rn_ride = vec![];
    let mut rn_next = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        for dir in 0..2 {
            let base = s + rn_station.len();
            let len = b - a;
            for i in 0..len {
                let k = if dir == 0 { a + i } else { b - 1 - i };
                rn_station.push(net.line_stops[k]);
                rn_line.push(l as u32);
                let ride = if i + 1 < len {
                    if dir == 0 {
                        net.hop_min[k]
                    } else if net.hop_back.is_empty() {
                        net.hop_min[k - 1]
                    } else {
                        net.hop_back[k]
                    }
                } else {
                    0.0
                };
                rn_ride.push(ride);
                rn_next.push(if i + 1 < len { (base + i + 1) as u32 } else { NONE });
            }
        }
    }
    let n = s + rn_station.len();
    let mut edges: Vec<(u32, u32, f32, u32)> = vec![];
    for (r, &st) in rn_station.iter().enumerate() {
        let node = (s + r) as u32;
        let l = rn_line[r] as usize;
        if rn_next[r] != NONE {
            edges.push((st, node, board_cost(net.line_headway_min[l][p.period], p), NONE));
            edges.push((node, rn_next[r], rn_ride[r], node));
        }
        edges.push((node, st, 0.0, NONE));
    }
    // walking transfers: from each route node (having ridden) to the stations nearby, and to every
    // other station of its complex however far (T-161), at the same walk cost by distance
    let cut2 = p.xfer_walk_m * p.xfer_walk_m;
    let walk = |a: usize, b: usize| {
        let d2 = (net.st_x[b] - net.st_x[a]).powi(2) + (net.st_y[b] - net.st_y[a]).powi(2);
        (d2, p.xfer_min(d2.sqrt()) * p.w_xfer)
    };
    let mut near: Vec<Vec<(u32, f32)>> = vec![vec![]; s];
    if p.xfer_walk_m > 0.0 && s > 1 {
        let grid = StationGrid::new(net, p.xfer_walk_m, p.xfer_walk_m);
        for a in 0..s {
            grid.near(net.st_x[a], net.st_y[a], |b| {
                let (d2, c) = walk(a, b as usize);
                if b as usize != a && d2 <= cut2 {
                    near[a].push((b, c));
                }
            });
        }
    }
    if net.st_group.len() == s {
        for members in complex_members(&net.st_group) {
            for &a in &members {
                for &b in &members {
                    let (d2, c) = walk(a as usize, b as usize);
                    // within the walk limit the edge is there already
                    if a != b && (p.xfer_walk_m <= 0.0 || d2 > cut2) {
                        near[a as usize].push((b, c));
                    }
                }
            }
        }
    }
    for (r, &st) in rn_station.iter().enumerate() {
        for &(b, c) in &near[st as usize] {
            edges.push(((s + r) as u32, b, c, NONE));
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
    let rn_tph = rn_line.iter().map(|&l| {
        let h = net.line_headway_min[l as usize][p.period];
        if h > 0.0 && h.is_finite() { 60.0 / h } else { 0.0 }
    }).collect();
    Graph {
        rn_left: vec![0.0; rn_ride.len()],
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
            g.adj_cost[e] = g.rn_ride[r] * factor[r] + g.rn_fare[r];
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
}

/// Node ids, as small as the graph allows (`NONE` is the type's largest value).
#[derive(Clone)]
pub enum NodeIds {
    Short(Box<[u16]>),
    Long(Box<[u32]>),
}

impl NodeIds {
    fn of(v: &[u32], n: usize) -> NodeIds {
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
trait Id: Copy {
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
fn accepted(bits: &[u64], g: &Graph, r: usize) -> f32 {
    if bits[r >> 6] >> (r & 63) & 1 == 1 { g.rn_tph[r] } else { 0.0 }
}

impl Skims {
    pub fn empty(g: &Graph) -> Skims {
        Skims { s: g.n_st, n: g.n, r: vec![f32::INFINITY; g.n_st * g.n_st], slot: vec![NONE; g.n_st], next: vec![], board: vec![], order: vec![] }
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
    let (s, n) = (g.n_st, g.n);
    let nr = n - s;
    let todo: Vec<usize> = (0..s).filter(|&d| want.get(d).copied().unwrap_or(false) && sk.slot[d] == NONE).collect();
    if todo.is_empty() {
        return;
    }
    let mut dist = vec![f32::INFINITY; n];
    let mut done = vec![false; n];
    let mut frequency = vec![0f32; s];
    let mut weighted = vec![0f32; s];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(n);
    let mut next = vec![NONE; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for &dst in &todo {
        sk.slot[dst] = sk.next.len() as u32;
        dist.fill(f32::INFINITY);
        done.fill(false);
        frequency.fill(0.0);
        weighted.fill(0.0);
        next.fill(NONE);
        order.clear();
        let mut board = vec![0u64; nr.div_ceil(64)].into_boxed_slice();
        dist[dst] = 0.0;
        heap.push(Reverse((0f32.to_bits(), dst as u32)));
        while let Some(Reverse((_, id))) = heap.pop() {
            let id = id as usize;
            if id >= n {
                // a boarding onto a line that leaves riders behind, at its turn (T-552)
                let v = id - n + s;
                let from = g.rn_station[v - s] as usize;
                let d = dist[v] + g.rn_left[v - s] * p.w_wait;
                if !done[from] {
                    if let Some(cost) = join_line(&mut frequency[from], &mut weighted[from], dist[from], g.rn_tph[v - s], d, p) {
                        board[(v - s) >> 6] |= 1 << ((v - s) & 63);
                        dist[from] = cost;
                        next[from] = v as u32;
                        heap.push(Reverse((cost.to_bits(), from as u32)));
                    }
                }
                continue;
            }
            let v = id;
            if done[v] { continue; }
            done[v] = true;
            order.push(v as u32);
            let d = dist[v];
            for j in g.rev_off[v] as usize..g.rev_off[v + 1] as usize {
                let from = g.rev_from[j] as usize;
                if done[from] { continue; }
                let e = g.rev_edge[j] as usize;
                let nd = if from < s && v >= s {
                    let f = g.rn_tph[v - s];
                    if f == 0.0 { continue; }
                    // Lines join a station's strategy in order of their cost from boarding, which
                    // for a line leaving riders behind includes the extra wait (T-188): it waits
                    // its turn, or a full train would join ahead of better lines and stay (T-552).
                    let left = g.rn_left[v - s];
                    if left > 0.0 {
                        heap.push(Reverse(((d + left * p.w_wait).to_bits(), (n + v - s) as u32)));
                        continue;
                    }
                    let Some(cost) = join_line(&mut frequency[from], &mut weighted[from], dist[from], f, d, p) else { continue };
                    board[(v - s) >> 6] |= 1 << ((v - s) & 63);
                    cost
                } else { d + g.adj_cost[e] };
                if nd < dist[from] {
                    dist[from] = nd;
                    next[from] = v as u32;
                    heap.push(Reverse((nd.to_bits(), from as u32)));
                }
            }
        }
        sk.next.push(NodeIds::of(&next, n));
        sk.board.push(board);
        sk.order.push(NodeIds::of(&order, n));
        for src in 0..s {
            // The first boarding is not a transfer; the trip's base fare is added later.
            sk.r[src * s + dst] = if src == dst { 0.0 } else { dist[src] - p.transfer_min };
        }
    }
}

/// A line with `f` departures an hour and cost `d` from boarding joins a station's strategy if it
/// lowers the expected cost there below `cur`: the new cost, with the station's pooled frequency
/// and frequency-weighted cost updated.
#[inline(always)]
fn join_line(frequency: &mut f32, weighted: &mut f32, cur: f32, f: f32, d: f32, p: &Params) -> Option<f32> {
    let sum = *frequency + f;
    let cost = (*weighted + f * d) / sum + board_cost(60.0 / sum, p);
    if cost >= cur {
        return None;
    }
    *frequency = sum;
    *weighted += f * d;
    Some(cost)
}

/// Test switch (`TW_LEGACY=1` in api.rs's `northeast_profile`): behave as before T-119 (searches
/// towards every station for every population, every crowding round solved), to compare the
/// results of the two.
pub static LEGACY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn legacy() -> bool {
    cfg!(test) && LEGACY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The stations anyone can alight at: those in the access lists of subzones with jobs. Nobody's
/// rail trip ends anywhere else, so the strategy searches towards other stations are not needed.
pub fn egress_stations(sz: &Subzones, n_st: usize) -> Vec<bool> {
    let mut want = vec![false; n_st];
    for v in 0..sz.n {
        if sz.jobs[v] > 0.0 {
            for k in sz.walk(v) {
                want[sz.acc_st[k] as usize] = true;
            }
        }
    }
    want
}

/// What a skims table depends on: the graph's edges and the boarding cost's parameters.
struct SkimKey {
    n_st: usize,
    n: usize,
    adj_off: Vec<u32>,
    adj_to: Vec<u32>,
    adj_cost: Vec<f32>,
    rn_tph: Vec<f32>,
    rn_left: Vec<f32>,
    par: [f32; 4],
}

impl SkimKey {
    fn of(g: &Graph, p: &Params) -> SkimKey {
        SkimKey { n_st: g.n_st, n: g.n, adj_off: g.adj_off.clone(), adj_to: g.adj_to.clone(), adj_cost: g.adj_cost.clone(), rn_tph: g.rn_tph.clone(), rn_left: g.rn_left.clone(), par: [p.platform_wait_cap_min, p.w_wait, p.w_home_wait, p.transfer_min] }
    }
    fn same(&self, g: &Graph, p: &Params) -> bool {
        self.n_st == g.n_st
            && self.n == g.n
            && self.par == [p.platform_wait_cap_min, p.w_wait, p.w_home_wait, p.transfer_min]
            && self.adj_off == g.adj_off
            && self.adj_to == g.adj_to
            && self.adj_cost == g.adj_cost
            && self.rn_tph == g.rn_tph
            && self.rn_left == g.rn_left
    }
}

/// Skims tables a thread keeps (`with_skims`). Each holds stations² route costs plus about 4
/// bytes a node per searched destination (u16 ids): on the real New York save 18 MB for New
/// York's table (T-398).
const SKIM_TABLES: usize = 2;

thread_local! {
    /// Skims of the graphs a worker solved most recently, newest last (T-119).
    static SKIM_CACHE: std::cell::RefCell<Vec<(SkimKey, Skims)>> = std::cell::RefCell::new(vec![]);
}

/// Run `f` on the skims of `g` towards the stations in `want`, searching only what no earlier
/// caller on this thread has searched for the same graph. A worker solves several populations
/// (two cities and the longer-trip market) on one service, and a period's graph is the same for
/// all of them once crowding is set, so the searches are done once per period and round, not once
/// per population (T-119). The cities do not always agree to the last metre on where two stations
/// are, so a period can bring a graph per city: `SKIM_TABLES` are kept, and which go is chosen by
/// what they cost to search again, so the order the populations come in does not decide (T-405).
pub fn with_skims<R>(g: &Graph, p: &Params, want: &[bool], f: impl FnOnce(&Skims) -> R) -> R {
    if legacy() {
        return f(&skims(g, p));
    }
    SKIM_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        match c.iter().position(|(k, _)| k.same(g, p)) {
            Some(i) => {
                let e = c.remove(i);
                c.push(e);
            }
            None => {
                if c.len() >= SKIM_TABLES {
                    // the table cheapest to search again (fewest destinations, the oldest of
                    // equals) gives way, and only to one that will cost at least as much: a
                    // cheaper one is searched for this call alone
                    let todo = (0..g.n_st).filter(|&d| want.get(d).copied().unwrap_or(false)).count();
                    let (i, kept) = c.iter().enumerate().map(|(i, e)| (i, e.1.next.len())).min_by_key(|e| e.1).unwrap();
                    if todo < kept {
                        drop(c);
                        return f(&skims_for(g, p, want));
                    }
                    c.remove(i);
                }
                c.push((SkimKey::of(g, p), Skims::empty(g)));
            }
        }
        let sk = &mut c.last_mut().unwrap().1;
        skims_extend(sk, g, p, want);
        f(sk)
    })
}

/// Forget the cached skims (a worker calls this when it has finished a period's stage, so the
/// tables do not sit in memory while it waits).
pub fn clear_skim_cache() {
    SKIM_CACHE.with(|c| c.borrow_mut().clear());
}

// ---------------------------------------------------------------------------------------------
// 5. Mode choice

/// exp(x) to ~2e-4 relative: 2^int by exponent bits, 2^frac by a degree-5 polynomial. wasm has
/// no exp instruction, and libm's exp was the larger part of the mode choice loop there.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let t = x.clamp(-80.0, 80.0) * std::f32::consts::LOG2_E;
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355_8))));
    f32::from_bits(((i as i32 + 127) << 23) as u32) * p
}

/// ln(x) for normal x > 0, to 3e-6 absolute: exponent bits, then a degree-6 polynomial in
/// t = m - 1 with the mantissa m in [0.71, 1.41) (least-squares fit, notes/T-020.md). Same reason
/// as `fast_exp`: wasm has no log instruction. Branch- and division-free: shifting the bits by
/// those of 1/sqrt(2) folds the mantissa without a data-dependent branch, which in the pair loop
/// mispredicted half the time. In a logsum cost, 3e-6 is 1.5e-5 perceived minutes.
#[inline]
pub fn fast_ln(x: f32) -> f32 {
    const HALF_SQRT2: u32 = 0x3f35_04f3; // 0.70710677
    let b = x.to_bits().wrapping_sub(HALF_SQRT2);
    let e = (b as i32) >> 23;
    let t = f32::from_bits((b & 0x007f_ffff).wrapping_add(HALF_SQRT2)) - 1.0;
    let q = 1.000_004_68 + t * (-0.499_903_616 + t * (0.332_567_778 + t * (-0.253_978_318 + t * (0.220_702_996 + t * -0.143_383_126))));
    e as f32 * std::f32::consts::LN_2 + t * q
}

/// exp(-theta x station-to-station cost), S x S, with a zero diagonal: boarding and alighting at
/// the same station is not a rail trip. The access logsum per origin subzone is built from it.
pub fn station_exp(sk: &Skims, p: &Params) -> Vec<f32> {
    let s = sk.s;
    let mut er: Vec<f32> = sk.r.iter().map(|&r| if r.is_finite() { fast_exp(-p.station_theta * r) } else { 0.0 }).collect();
    for a in 0..s {
        er[a * s + a] = 0.0;
    }
    er
}

/// Distance bands (km, upper edges) for rail trips by how far home is from the boarding station.
pub const DIST_BANDS_KM: [f32; N_BANDS] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, f32::INFINITY];
pub const N_BANDS: usize = 13;
pub const N_WALKSUM: usize = 16;
pub const WALKSUM_BAND_MIN: f32 = 10.0;

/// The band of `DIST_BANDS_KM` a straight-line distance in metres falls in.
pub fn dist_band(d_m: f32) -> usize {
    DIST_BANDS_KM.iter().position(|&e| d_m < e * 1000.0).unwrap_or(N_BANDS - 1)
}

pub struct ModeOut {
    /// Rail trips by (access station, egress station), S x S, in the home -> work direction.
    pub od: Vec<f32>,
    pub rail: f64,
    /// Rail trips by straight-line distance from home (subzone centroid) to the boarding station,
    /// bands `DIST_BANDS_KM`.
    pub rail_band: [f64; N_BANDS],
    /// Rail trips by the perceived minutes of the cheapest walks at both ends together, in
    /// bands of `WALKSUM_BAND_MIN` (the last one open-ended): how much rail sits near
    /// `Params::pair_walk_max`.
    pub rail_walksum: [f64; N_WALKSUM],
    /// Trips between subzones offered rail: both ends within the walk's bound of a station, and
    /// at least one pair of the two zones under `Params::pair_walk_max`.
    pub trips_access: f64,
    /// All home->work trips in the period.
    pub trips_total: f64,
    /// Legacy walking-loss field, kept zero for the worker/query layout. Walking is reserved
    /// before rail/drive choice (T-153), so rail never subtracts from walking.
    pub rail_walk: f64,
    pub zone_pairs: u64,
    pub pairs: u64,
    pub pruned_trips: f64,
    /// Rail trips by home (origin) subzone and by work (destination) subzone.
    pub rail_by_sub: Vec<f32>,
    pub rail_to_sub: Vec<f32>,
    /// `rail_walk` by home subzone and by work subzone.
    pub walk_lost_by_sub: Vec<f32>,
    pub walk_lost_to_sub: Vec<f32>,
    /// T-513: rail trips of those without a car times other's share of their trips not by rail
    /// (`Params::nocar_choice`): what rail takes from other, in all and by home subzone. Other's
    /// trips are those with no rail (`Gravity::other_trips`) less these.
    pub other_lost: f64,
    pub other_lost_by_sub: Vec<f32>,
    /// Rail trips per access entry (`Subzones::acc_st`): at the home end, boarding at that
    /// station from that subzone; at the work end, leaving the train there for that subzone.
    pub acc_home: Vec<f32>,
    pub acc_work: Vec<f32>,
}

/// Reserve the zone-pair walking share, then choose rail/road for the remainder. Rail mode
/// choice per (origin subzone, destination subzone) pair uses a logit over access and
/// egress stations (SPEC 4.3, 4.4; notes/T-020.md, T-078.md).
///
/// For a pair (u, v) the rail alternatives are every (access station a of u, egress station t of
/// v); each costs access_a + R[a, t] + egress_t. Their logsum separates:
///     S_uv = sum_t A_u[t] E_v[t],  A_u[t] = sum_a exp(-theta (access_a - m_u)) exp(-theta R[a, t]),
///     E_v[t] = exp(-theta (egress_t - m_v)),  rail cost = m_u + m_v + fare - ln(S_uv) / theta.
/// A_u is one row of S per origin subzone, built once per subzone (access options x S
/// multiply-adds), so the pair loop costs one product per egress station plus one ln and one
/// exp. Rail trips are kept by (u, egress station) as Y_u[t] = sum_v x_uv E_v[t] / S_uv and split
/// back onto access stations once per subzone: od[a, t] += Y_u[t] w_a ER[a, t]. With rides
/// (T-371) an origin has a second row, B_u, over its walks and rides at the cheapest of either's
/// offset, used for the commutes `ride_offered`, with its own Y.
///
/// An origin first lists its pairs within the walk bound, then takes them eight at a time in
/// `lanes::F8` (T-036): every lane runs the scalar expressions in their order and every sum keeps
/// its order, so the results are those of one pair at a time to the bit (`trip::pair_choice`).
pub fn mode_choice(z: &Zones, gr: &Gravity, sz: &Subzones, er: &[f32], s: usize, p: &Params) -> ModeOut {
    let share = p.period_share as f64;
    let th = p.station_theta;
    // per subzone: its part of the zone's row / column factor
    let ou: Vec<f32> = (0..sz.n).map(|u| sz.origin_weight(u, z, gr, share)).collect();
    let dv: Vec<f32> = (0..sz.n)
        .map(|v| {
            let zj = sz.zone[v] as usize;
            if z.jobs[zj] > 0.0 { (gr.col[zj] * sz.jobs[v] as f64 / z.jobs[zj]) as f32 } else { 0.0 }
        })
        .collect();
    let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
    // per subzone and access entry: weights exp(-theta (cost - m)), a walk entry's with m the
    // cheapest walk (`acc_m`, the same at both ends; infinite with none), a ride entry's with m
    // the cheapest of walks and rides (`ride_m`)
    let mut acc_m = vec![f32::INFINITY; sz.n];
    let mut ride_m = vec![f32::INFINITY; sz.n];
    let mut acc_e = vec![0f32; sz.acc_st.len()];
    let min_of = |r: std::ops::Range<usize>| sz.acc_w[r].iter().cloned().fold(f32::INFINITY, f32::min);
    for v in 0..sz.n {
        let m = min_of(sz.walk(v));
        let mr = m.min(min_of(sz.ride(v)));
        acc_m[v] = m;
        ride_m[v] = mr;
        for k in sz.walk(v) {
            acc_e[k] = fast_exp(-th * (sz.acc_w[k] - m));
        }
        for k in sz.ride(v) {
            acc_e[k] = fast_exp(-th * (sz.acc_w[k] - mr));
        }
    }
    // per subzone: minutes a km on its local streets and main roads (by its zone's density)
    let dens = z.density();
    let local: Vec<(f32, f32)> = (0..sz.n).map(|v| p.place_min_per_km(dens[sz.zone[v] as usize])).collect();
    let mut out = ModeOut {
        od: vec![0.0; s * s],
        rail: 0.0,
        rail_band: [0.0; N_BANDS],
        rail_walksum: [0.0; N_WALKSUM],
        trips_access: 0.0,
        trips_total: gr.total * share,
        rail_walk: 0.0,
        zone_pairs: 0,
        pairs: 0,
        pruned_trips: 0.0,
        rail_by_sub: vec![0.0; sz.n],
        rail_to_sub: vec![0.0; sz.n],
        walk_lost_by_sub: vec![0.0; sz.n],
        walk_lost_to_sub: vec![0.0; sz.n],
        other_lost: 0.0,
        other_lost_by_sub: vec![0.0; sz.n],
        acc_home: vec![0.0; sz.acc_st.len()],
        acc_work: vec![0.0; sz.acc_st.len()],
    };
    let ozones: Vec<usize> = (0..z.n).filter(|&i| sz.zo_off[i + 1] > sz.zo_off[i]).collect();
    // Destination zones by their cheapest walk (their first subzone: `build_subzones` sorts
    // them), so an origin visits only the prefix within `pair_walk_max` of its own walk: in
    // 10-minute steps of that walk, in zone order within a step (zone order keeps the decay
    // table lookups near each other; sorting by the walk alone was 20% slower).
    let mut dzones: Vec<usize> = (0..z.n).filter(|&j| sz.zd_off[j + 1] > sz.zd_off[j]).collect();
    let zmin = |j: usize| acc_m[sz.zd[sz.zd_off[j] as usize] as usize];
    dzones.sort_by_key(|&j| ((zmin(j) * 0.1) as u32, j));
    let dstep: Vec<u32> = dzones.iter().map(|&j| (zmin(j) * 0.1) as u32).collect();
    // how many of them origin subzone u visits: past this, the cheapest walk there takes every
    // pair over `pair_walk_max` (u's own cheapest walk, or walk or ride when it has rides)
    let reach = |u: usize| {
        let lim = p.pair_walk_max - if sz.ride(u).is_empty() { acc_m[u] } else { ride_m[u] };
        dstep.partition_point(|&m| m as f32 * 10.0 <= lim)
    };
    // Everything the pair loop reads about a destination, packed in visiting order: per zone,
    // then per subzone, then its egress stations.
    struct DZone {
        key: i32,
        reg: u8,
        zid: u32,
        min: f32,
        max: f32,
        dv: f32,
        q0: u32,
        q1: u32,
        // its square and drive supercell, for the zone pair's factors without reading `z`
        gx: i32,
        gy: i32,
        sup: u16,
    }
    let mut dz: Vec<DZone> = Vec::with_capacity(dzones.len());
    // per destination: [x, y, local streets, main roads], [parking every driver pays, trips
    // factor, walk, a paid space], [the share parking free, -, -, -]
    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());
    // and its egress stations, their weights and `Subzones::acc_st` entries, padded to K with
    // station `s` (read as a dummy past the row's end, 0) at weight 0: a pad adds exactly 0 to
    // every sum it enters, so eight destinations go through the same steps
    let (mut eg_st, mut eg_e, mut eg_k, mut eg_n): (Vec<[u32; K]>, Vec<[f32; K]>, Vec<[u32; K]>, Vec<u8>) = (vec![], vec![], vec![], vec![]);
    for &j in &dzones {
        let q0 = dd.len() as u32;
        let mut dvz = 0.0f32;
        for &v in &sz.zd[sz.zd_off[j] as usize..sz.zd_off[j + 1] as usize] {
            let v = v as usize;
            let (mut st, mut e, mut kk) = ([s as u32; K], [0f32; K], [0u32; K]);
            for (n, k) in sz.walk(v).enumerate() {
                (st[n], e[n], kk[n]) = (sz.acc_st[k], acc_e[k], k as u32);
            }
            eg_st.push(st);
            eg_e.push(e);
            eg_k.push(kk);
            eg_n.push(sz.walk(v).len() as u8);
            dvz += dv[v];
            let (fixed, paid, free) = park[j];
            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);
        }
        let q1 = dd.len() as u32;
        dz.push(DZone { key: gr.key(z, j), reg: z.region[j], zid: j as u32, min: zmin(j), max: dd[q1 as usize - 1][1][2], dv: dvz, q0, q1, gx: z.gx[j], gy: z.gy[j], sup: p.routes.as_ref().map_or(0, |r| r.sup_of(j)) });
    }
    // the decay and the walk share by zone offset side by side: one lookup per zone pair
    let fw: Vec<[f32; 2]> = gr.ftab.iter().enumerate().map(|(k, &f)| [f, if gr.wtab.is_empty() { 0.0 } else { gr.wtab[k] }]).collect();
    // with a third: the road's length over the straight line between the zones (T-275); and a
    // fourth, 1 where the commute is long enough for the ride (T-371)
    let mut fz = vec![[0f32, 0.0, p.road_detour, 0.0]; dz.len()];
    let routed_walk = gr.wroutes.is_some();
    let drive = p.routes.as_deref().filter(|r| r.has_drive());
    let ride_min2 = (p.ride_min_km * 1000.0).powi(2);
    let kw = p.walk_trip_min();
    let (ko, kow) = (p.other_min_km(), p.other_wait());
    // the origin's two rows (walks; walks and rides) one after the other, each followed by
    // `PADS` dummies (0), and their rail trips by egress station the same way
    let h = s + PADS;
    let mut ab_row = vec![0f32; 2 * h];
    let mut ab_y = vec![0f32; 2 * h];
    // rail trips reaching each destination's egress entries, by its egress step
    let mut acc_q = vec![[[0f32; 4]; 2]; dd.len()];
    // the origin's pairs within the walk bound, in visiting order: destination, and the zone
    // visit (`visit`: [trips at the origin x decay, 1 - walk share, road factor, origin walk]),
    // its top bit set for the walk-and-ride row
    let mut cand: Vec<[u32; 2]> = Vec::with_capacity(dd.len() + 7);
    let mut visit: Vec<[f32; 4]> = Vec::with_capacity(dz.len());
    let mut walksum = [0f64; N_WALKSUM];
    for &i in &ozones {
        let base = gr.base(z, i);
        let us = &sz.zo[sz.zo_off[i] as usize..sz.zo_off[i + 1] as usize];
        let ou_zone: f32 = us.iter().map(|&u| ou[u as usize]).sum();
        // the decay (times the region factor) and walk share to every destination zone one of
        // this zone's origins reaches, once for all of them
        let ri = z.region[i];
        let nzi = us.iter().map(|&u| reach(u as usize)).max().unwrap_or(0);
        let (gxi, gyi) = (z.gx[i], z.gy[i]);
        for (o, zj) in fz[..nzi].iter_mut().zip(&dz) {
            let k = (base + zj.key) as usize;
            let [f, w] = fw[k];
            // `ride_offered`
            let (dx, dy) = (zj.gx - gxi, zj.gy - gyi);
            let (fx, fy) = (dx as f32, dy as f32);
            let long = ((fx * fx + fy * fy) * z.zone_m * z.zone_m >= ride_min2) as u32 as f32;
            *o = [f, w, o[2], long];
            if zj.reg != ri {
                o[0] *= gr.border;
            }
            if routed_walk && w > 0.0 {
                o[1] = gr.walk_share_at(i, zj.zid as usize, k);
            }
            if let Some(r) = drive {
                // `Params::drive_factor`
                o[2] = r.drive_factor_by(i, dx, dy, zj.reg, zj.sup) * p.road_routed_scale;
            }
        }
        for (ui, &u) in us.iter().enumerate() {
            let u = u as usize;
            let (walk, ride) = (sz.walk(u), sz.ride(u));
            let (m_u, m_r) = (acc_m[u], ride_m[u]);
            let rides = !ride.is_empty();
            // the walk entries' weight at the rides' offset
            let c_u = if walk.is_empty() { 0.0 } else { fast_exp(-th * (m_u - m_r)) };
            {
                let (a_row, b_row) = ab_row.split_at_mut(h);
                let (a_row, b_row) = (&mut a_row[..s], &mut b_row[..s]);
                a_row.fill(0.0);
                for k in walk.clone() {
                    let (a, w) = (sz.acc_st[k] as usize, acc_e[k]);
                    for (av, &e) in a_row.iter_mut().zip(&er[a * s..(a + 1) * s]) {
                        *av += w * e;
                    }
                }
                // the row with the rides too, for the long commutes
                if rides {
                    for (bv, &av) in b_row.iter_mut().zip(a_row.iter()) {
                        *bv = c_u * av;
                    }
                    for k in ride.clone() {
                        let (a, w) = (sz.acc_st[k] as usize, acc_e[k]);
                        for (bv, &e) in b_row.iter_mut().zip(&er[a * s..(a + 1) * s]) {
                            *bv += w * e;
                        }
                    }
                }
            }
            ab_y.fill(0.0);
            let (ux8, uy8, lu08, lu18) = (F8::splat(sz.cx[u]), F8::splat(sz.cy[u]), F8::splat(local[u].0), F8::splat(local[u].1));
            let mut rail_sum = 0.0f32;
            let mut other_sum = 0.0f32;
            let mut all_sum = 0.0f32;
            let mut n_pairs = 0usize;
            // past this, a destination's walk takes the pair over `pair_walk_max` (by walk, and
            // by walk or ride)
            let wmax = p.pair_walk_max - m_u;
            let wmax_r = p.pair_walk_max - m_r;
            // T-511: the share of this origin's homes without a car
            let cu = sz.nocar[u];
            let wmax_n = F8::splat(wmax);
            let nz = reach(u);
            for (zj, &[f, pw, df, long]) in dz[..nz].iter().zip(&fz) {
                let by_ride = rides && long > 0.0;
                let (m_o, wmax) = if by_ride { (m_r, wmax_r) } else { (m_u, wmax) };
                if zj.min > wmax {
                    continue;
                }
                if p.prune_trips > 0.0 && ou_zone * zj.dv * f < p.prune_trips {
                    out.pruned_trips += (ou[u] * zj.dv * f) as f64;
                    continue;
                }
                if ui == 0 {
                    out.zone_pairs += 1;
                }
                let tu = ou[u] * f;
                all_sum += tu * zj.dv;
                // cheapest walk first: past the bound, every later one is too
                let nv = if zj.max <= wmax { zj.q1 - zj.q0 } else { dd[zj.q0 as usize..zj.q1 as usize].partition_point(|d| d[1][2] <= wmax) as u32 };
                n_pairs += nv as usize;
                let zv = visit.len() as u32 | (by_ride as u32) << 31;
                visit.push([tu, 1.0 - pw, df, m_o]);
                cand.extend((zj.q0..zj.q0 + nv).map(|q| [q, zv]));
            }
            out.pairs += n_pairs as u64;
            // The pairs eight at a time, each lane the scalar expressions in their order
            // (`trip::pair_choice` gives the same bits one pair at a time), then each pair's rail
            // trips onto its egress stations, lane by lane in the pairs' order.
            let n_cand = cand.len();
            while cand.len() % 8 != 0 {
                cand.push(cand[0]);
            }
            let (mut wb, mut wacc) = (0, walksum[0]);
            for (ci, c) in cand.chunks_exact(8).enumerate() {
                let q: [usize; 8] = std::array::from_fn(|l| c[l][0] as usize);
                let ro: [u32; 8] = std::array::from_fn(|l| if c[l][1] >> 31 != 0 { h as u32 } else { 0 });
                let kn = q.iter().map(|&q| eg_n[q]).max().unwrap() as usize;
                let mut ix = [[0u32; 8]; K];
                let mut ev = [F8::splat(0.0); K];
                let mut rv = [F8::splat(0.0); K];
                let mut ssum = F8::splat(0.0);
                for k in 0..kn {
                    // a pad goes to its own dummy (lane and step), so pads chain nothing
                    ix[k] = std::array::from_fn(|l| {
                        let t = eg_st[q[l]][k] as usize;
                        (if t == s { s + l * K + k } else { t }) as u32 + ro[l]
                    });
                    ev[k] = F8::new(std::array::from_fn(|l| eg_e[q[l]][k]));
                    rv[k] = F8::gather(&ab_row, &ix[k]);
                    ssum = ssum + rv[k] * ev[k];
                }
                // no rail route (or only the same station at both ends) where under 1e-30
                let ok = ((1u32 << (n_cand - 8 * ci).min(8)) - 1) & !ssum.lt_mask(F8::splat(1e-30));
                if ok == 0 {
                    continue;
                }
                let [x8, y8, l08, l18] = F8::transpose(std::array::from_fn(|l| dd[q[l]][0]));
                let [park8, dv8, m8, paid8] = F8::transpose(std::array::from_fn(|l| dd[q[l]][1]));
                let [tu8, dpw8, df8, mo8] = F8::transpose(std::array::from_fn(|l| visit[(c[l][1] & 0x7fff_ffff) as usize]));
                let (dx, dy) = (ux8 - x8, uy8 - y8);
                let d = (dx * dx + dy * dy).sqrt();
                // `Params::drive_min_f`
                let rd = d * F8::splat(0.001) * df8;
                let e = (F8::splat(0.5) * rd).min(F8::splat(p.road_local_km));
                let main = rd - F8::splat(2.0) * e;
                let k = (F8::splat(0.5) * main).min(F8::splat(p.road_end_km));
                let drive = e * (lu08 + l08) + k * (lu18 + l18) + (main - F8::splat(2.0) * k) * F8::splat(60.0) / F8::splat(p.road_open_kmh);
                let road = drive * F8::splat(p.w_drive) + park8;
                let rail = mo8 + m8 + F8::splat(p.fare_min) - ssum.ln() / F8::splat(th);
                // `Params::mode_beta`
                let bt = F8::splat(p.beta) * (F8::splat(p.damp_min) / road).min(F8::splat(1.0)).sqrt();
                let util = bt * (road - rail) + F8::splat(p.asc_rail);
                // T-380: rail's share among drivers parking free, and among those paying for a
                // space (`rail_share_parked`; where no lane pays, the second share equals the first)
                let r0 = F8::splat(1.0) / (F8::splat(1.0) + util.neg().exp());
                let r = if paid8.lt_mask(F8::splat(1e-30)) == 0xff {
                    r0
                } else {
                    let [free8, _, _, _] = F8::transpose(std::array::from_fn(|l| dd[q[l]][2]));
                    let r1 = F8::splat(1.0) / (F8::splat(1.0) + (util + bt * paid8).neg().exp());
                    r0 + (F8::splat(1.0) - free8) * (r1 - r0)
                };
                // T-511, T-513: those without a car choose among rail, walking to stations,
                // walking the whole way and other; a lane on the rides' row prices their rail on
                // the walks' row
                let ride_l: u32 = (0..8).fold(0, |m, l| m | ((ro[l] != 0) as u32) << l);
                let split = cu > 0.0 && ride_l != 0;
                let mut ix0 = [[0u32; 8]; K];
                let mut rv0 = [F8::splat(0.0); K];
                // their share choosing rail, and their logsum's sum on the walks' row
                let mut rn = F8::splat(0.0);
                // other's share of their trips not by rail
                let mut qn = F8::splat(0.0);
                let mut sn = ssum;
                let mut okn = 0xffu32;
                let r = if cu > 0.0 {
                    let mut rail_n = rail;
                    if split {
                        sn = F8::splat(0.0);
                        for k in 0..kn {
                            ix0[k] = std::array::from_fn(|l| ix[k][l] - ro[l]);
                            rv0[k] = F8::gather(&ab_row, &ix0[k]);
                            sn = sn + rv0[k] * ev[k];
                        }
                        let rw = (F8::splat(m_u) + m8 + F8::splat(p.fare_min) - sn.ln() / F8::splat(th)).to_array();
                        let ra = rail.to_array();
                        rail_n = F8::new(std::array::from_fn(|l| if ride_l >> l & 1 != 0 { rw[l] } else { ra[l] }));
                        // walks over the bound, or no station on foot: no rail for them
                        okn = !(ride_l & (wmax_n.lt_mask(m8) | sn.lt_mask(F8::splat(1e-30)))) & 0xff;
                    }
                    // `Params::nocar_choice`: walking and other along the drive's road
                    let wk = rd * F8::splat(kw);
                    let oth = rd * F8::splat(ko) + F8::splat(kow);
                    let btn = F8::splat(p.beta) * (F8::splat(p.damp_min) / oth).min(F8::splat(1.0)).sqrt();
                    let oa = oth + F8::splat(p.other_asc);
                    qn = F8::splat(1.0) / (F8::splat(1.0) + (btn * (oa - wk)).exp());
                    let ew = (btn * (wk - rail_n) + F8::splat(p.asc_rail)).neg().exp();
                    let eo = (btn * (oa - rail_n) + F8::splat(p.asc_rail)).neg().exp();
                    let pn = (F8::splat(1.0) / (F8::splat(1.0) + ew + eo)).to_array();
                    rn = F8::new(std::array::from_fn(|l| if okn >> l & 1 != 0 { pn[l] } else { 0.0 }));
                    r + F8::splat(cu) * (rn - r)
                } else {
                    r
                };
                let x = tu8 * dv8 * dpw8 * r;
                // on a rides' lane their trips go on the walks' row, the rest on the rides'
                let (g, gn) = if split {
                    let xn = tu8 * dv8 * dpw8 * (F8::splat(cu) * rn);
                    let (xna, sna) = (xn.to_array(), sn.to_array());
                    let gn = F8::new(std::array::from_fn(|l| if ride_l >> l & 1 != 0 && okn >> l & 1 != 0 { xna[l] / sna[l] } else { 0.0 }));
                    let xr = F8::new(std::array::from_fn(|l| if ride_l >> l & 1 != 0 { xna[l] } else { 0.0 }));
                    ((x - xr) / ssum, gn)
                } else {
                    (x / ssum, F8::splat(0.0))
                };
                // trips by egress station y, and y x the row: what reaches each egress entry
                let mut ya = [[0f32; 8]; K];
                let mut yna = [[0f32; 8]; K];
                let mut yw = [F8::splat(0.0); 8];
                let mut xw = F8::splat(0.0);
                for k in 0..kn {
                    let y = g * ev[k];
                    yw[k] = y * rv[k];
                    if split {
                        let yn = gn * ev[k];
                        yw[k] = yw[k] + yn * rv0[k];
                        yna[k] = yn.to_array();
                    }
                    xw = xw + yw[k];
                    ya[k] = y.to_array();
                }
                // y x the row by lane, egress steps 0-3 and 4-7 (zeros past each list's end)
                let yw_lo = F8::by_lane([yw[0], yw[1], yw[2], yw[3]]);
                let yw_hi = if kn > 4 { F8::by_lane([yw[4], yw[5], yw[6], yw[7]]) } else { yw_lo };
                let (xa, xwa) = (x.to_array(), xw.to_array());
                // the no-car rail trips that would otherwise go by other (`other_lost_by_sub`)
                let la = if cu > 0.0 { (tu8 * dv8 * dpw8 * (F8::splat(cu) * rn) * qn).to_array() } else { [0.0; 8] };
                // each lane's walk-sum band, as the scalar (walk at home + at work) / band
                let bw = ((mo8 + m8) * F8::splat(1.0 / WALKSUM_BAND_MIN)).to_array();
                for l in 0..8 {
                    if ok >> l & 1 == 0 {
                        continue;
                    }
                    rail_sum += xa[l];
                    other_sum += la[l];
                    // pads add 0 to their own dummies: the block's one count predicts well
                    for k in 0..kn {
                        ab_y[ix[k][l] as usize] += ya[k][l];
                    }
                    if split && ride_l >> l & 1 != 0 {
                        for k in 0..kn {
                            ab_y[ix0[k][l] as usize] += yna[k][l];
                        }
                    }
                    let a = &mut acc_q[q[l]];
                    a[0] = (F4::new(a[0]) + yw_lo[l]).to_array();
                    if kn > 4 {
                        a[1] = (F4::new(a[1]) + yw_hi[l]).to_array();
                    }
                    // the band's running sum held while the band stays the same: the same adds
                    let b = (bw[l] as usize).min(N_WALKSUM - 1);
                    if b != wb {
                        walksum[wb] = wacc;
                        (wb, wacc) = (b, walksum[b]);
                    }
                    wacc += xwa[l] as f64;
                }
            }
            walksum[wb] = wacc;
            cand.clear();
            visit.clear();
            out.trips_access += all_sum as f64;
            if rail_sum == 0.0 {
                continue;
            }
            out.other_lost += other_sum as f64;
            out.other_lost_by_sub[u] = other_sum;
            out.rail += rail_sum as f64;
            out.rail_by_sub[u] = rail_sum;
            // split this subzone's rail trips back onto its access stations: a walk entry carries
            // its share of both rows' trips, a ride entry of the long commutes' only
            let (y_row, yb_row) = (&ab_y[..s], &ab_y[h..h + s]);
            for k in walk.start..ride.end {
                let (a, w) = (sz.acc_st[k] as usize, acc_e[k]);
                let row = &mut out.od[a * s..(a + 1) * s];
                let mut tot = 0.0f32;
                let er_a = &er[a * s..(a + 1) * s];
                if k < walk.end {
                    if rides {
                        for (((o, &y), &yb), &e) in row.iter_mut().zip(y_row).zip(yb_row).zip(er_a) {
                            let f = (y + c_u * yb) * w * e;
                            *o += f;
                            tot += f;
                        }
                    } else {
                        for ((o, &y), &e) in row.iter_mut().zip(y_row).zip(er_a) {
                            let f = y * w * e;
                            *o += f;
                            tot += f;
                        }
                    }
                } else {
                    for ((o, &yb), &e) in row.iter_mut().zip(yb_row).zip(er_a) {
                        let f = yb * w * e;
                        *o += f;
                        tot += f;
                    }
                }
                out.acc_home[k] += tot;
                out.rail_band[dist_band(sz.acc_d[k])] += tot as f64;
            }
        }
    }
    for (q, a) in acc_q.iter().enumerate() {
        for k in 0..eg_n[q] as usize {
            out.acc_work[eg_k[q][k] as usize] = a[k / 4][k % 4];
        }
    }
    // rail trips by work subzone: what its egress stations took
    for v in 0..sz.n {
        out.rail_to_sub[v] = out.acc_work[sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize].iter().sum();
    }
    out.rail_walksum = walksum;
    out
}

// ---------------------------------------------------------------------------------------------
// 6. Assignment

#[derive(Clone)]
pub struct Loads {
    /// Riders per period on the segment starting at each route node.
    pub seg: Vec<f32>,
    /// Boardings and alightings per station, transfers included.
    pub board: Vec<f32>,
    pub alight: Vec<f32>,
    /// The same per route node: riders boarding or leaving that line, direction and stop.
    pub rn_board: Vec<f32>,
    pub rn_alight: Vec<f32>,
    /// Rail trips starting (first boarding) and ending (last alighting) at each station: entries
    /// and exits, transfers not counted.
    pub entry: Vec<f32>,
    pub exit: Vec<f32>,
    pub pax_min: f64,
}

impl Loads {
    pub fn zero(n_st: usize, n_rn: usize) -> Loads {
        Loads { seg: vec![0.0; n_rn], board: vec![0.0; n_st], alight: vec![0.0; n_st], rn_board: vec![0.0; n_rn], rn_alight: vec![0.0; n_rn], entry: vec![0.0; n_st], exit: vec![0.0; n_st], pax_min: 0.0 }
    }
}

/// Each destination's strategy is walked from its farthest nodes towards it. Station flow
/// splits over useful departures; route flow follows its best ride/alight/walking-transfer edge.
/// Work is O(edges) per destination, independent of the number of OD pairs carrying flow.
pub fn assign(g: &Graph, sk: &Skims, od: &[f32]) -> Loads {
    assign_inner(g, sk, od, None)
}

/// `assign`, also adding the period's transfers into `xfer` (laid out by `ix`, T-176). The loads
/// are `assign`'s to the bit: the transfers are counted beside them, never mixed in.
pub fn assign_with_transfers(g: &Graph, sk: &Skims, od: &[f32], ix: &XferIndex, xfer: &mut [f32]) -> Loads {
    debug_assert_eq!(xfer.len(), ix.len);
    if cfg!(test) && XFER_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return assign(g, sk, od);
    }
    assign_inner(g, sk, od, Some((ix, xfer)))
}

/// Test switch (T-176's cost measurement): `assign_with_transfers` counts nothing.
pub static XFER_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Where transfers are counted (T-176): one slot per (arriving edge, departure) pair a rider can
/// change between. An arriving edge is a route node's edge to a station: alighting at its own
/// station, or a walking transfer to one nearby; the departures are that station's boarding
/// edges. So a slot is a pair of route nodes (line, direction and stop in, and out), which is
/// platform to platform. Depends only on the graph's edges, the same in every period and round.
pub struct XferIndex {
    /// Per edge: for an arriving edge, its first slot (its station's departures follow in edge
    /// order); `NONE` for every other edge.
    pub at: Vec<u32>,
    pub len: usize,
}

pub fn xfer_index(g: &Graph) -> XferIndex {
    let s = g.n_st;
    let mut at = vec![NONE; g.adj_to.len()];
    let mut len = 0usize;
    for v in s..g.n {
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let to = g.adj_to[e] as usize;
            if to < s {
                at[e] = len as u32;
                len += (g.adj_off[to + 1] - g.adj_off[to]) as usize;
            }
        }
    }
    XferIndex { at, len }
}

/// The transfers in `xfer` (laid out by `ix`) with either end at a route node marked in `sel`
/// (one byte per route node), as triples (route node in, route node out, riders), appended to
/// `out`. Route nodes are indices from 0 (node - n_st), the order of `Loads::seg`.
pub fn transfer_pairs(g: &Graph, ix: &XferIndex, xfer: &[f32], sel: &[u8], out: &mut Vec<f32>) {
    let s = g.n_st;
    if xfer.len() != ix.len || sel.len() != g.n - s {
        return;
    }
    for v in s..g.n {
        let a = sel[v - s] != 0;
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let slot = ix.at[e];
            if slot == NONE {
                continue;
            }
            let to = g.adj_to[e] as usize;
            for (k, d) in (g.adj_off[to] as usize..g.adj_off[to + 1] as usize).enumerate() {
                let x = xfer[slot as usize + k];
                let w = g.adj_to[d] as usize - s;
                if x > 0.0 && (a || sel[w] != 0) {
                    out.extend([(v - s) as f32, w as f32, x]);
                }
            }
        }
    }
}

fn assign_inner(g: &Graph, sk: &Skims, od: &[f32], mut xf: Option<(&XferIndex, &mut [f32])>) -> Loads {
    let (s, n) = (sk.s, sk.n);
    let nr = n - s;
    let mut l = Loads::zero(s, nr);
    let mut flow = vec![0f32; n];
    // T-176: riders arriving at each station towards this destination, as a list per station
    // (first record, then each record's next): the arriving edge's first slot and the riders.
    // A station is walked after every node that feeds it, so its list is whole by then.
    let counting = xf.is_some();
    let mut head = if counting { vec![NONE; s] } else { vec![] };
    let mut rec: Vec<(u32, f32, u32)> = vec![];
    let mut use_k: Vec<(u32, f32)> = vec![];
    for dst in 0..s {
        flow.fill(0.0);
        let mut total = 0.0;
        for src in 0..s {
            let v = od[src * s + dst];
            flow[src] = v;
            l.entry[src] += v;
            total += v;
        }
        l.exit[dst] += total;
        if total == 0.0 { continue; }
        let slot = sk.slot[dst] as usize;
        debug_assert!(slot != NONE as usize, "flow towards station {dst}, which was not searched");
        if slot == NONE as usize { continue; }
        let bits = &sk.board[slot][..];
        if counting {
            head.fill(NONE);
            rec.clear();
        }
        let mut w = DestWalk { g, s, n, dst, bits, flow: &mut flow, l: &mut l, head: &mut head, rec: &mut rec, use_k: &mut use_k, xf: xf.as_mut().map(|(ix, x)| (*ix, &mut **x)) };
        match (&sk.next[slot], &sk.order[slot]) {
            (NodeIds::Short(next), NodeIds::Short(order)) => w.run(next, order),
            (NodeIds::Long(next), NodeIds::Long(order)) => w.run(next, order),
            _ => unreachable!("a destination's ids in two widths"),
        }
    }
    l
}

/// One destination's part of `assign_inner`: its flows walked back along the settle order.
struct DestWalk<'a> {
    g: &'a Graph,
    s: usize,
    n: usize,
    dst: usize,
    bits: &'a [u64],
    flow: &'a mut [f32],
    l: &'a mut Loads,
    head: &'a mut [u32],
    rec: &'a mut Vec<(u32, f32, u32)>,
    use_k: &'a mut Vec<(u32, f32)>,
    xf: Option<(&'a XferIndex, &'a mut [f32])>,
}

impl DestWalk<'_> {
    #[inline(always)]
    fn run<I: Id>(&mut self, next: &[I], order: &[I]) {
        let DestWalk { g, s, n, dst, bits, ref mut flow, ref mut l, ref mut head, ref mut rec, ref mut use_k, ref mut xf } = *self;
        for &v in order.iter().skip(1).rev() {
            let v = v.ix();
            let f = flow[v];
            if f == 0.0 { continue; }
            if v < s {
                let edges = g.adj_off[v] as usize..g.adj_off[v + 1] as usize;
                let sum: f32 = edges.clone().map(|e| accepted(bits, g, g.adj_to[e] as usize - s)).sum();
                if sum == 0.0 { continue; }
                l.board[v] += f;
                for e in edges.clone() {
                    let to = g.adj_to[e] as usize;
                    let share = f * accepted(bits, g, to - s) / sum;
                    flow[to] += share;
                    l.rn_board[to - s] += share;
                }
                if head.get(v).is_some_and(|&h| h != NONE) {
                    // the riders who arrived here leave by the departures as everyone here does
                    // (only the useful ones: about half go the wrong way for this destination)
                    use_k.clear();
                    for (k, e) in edges.clone().enumerate() {
                        let b = accepted(bits, g, g.adj_to[e] as usize - s);
                        if b > 0.0 {
                            use_k.push((k as u32, b / sum));
                        }
                    }
                    let x = &mut xf.as_mut().unwrap().1;
                    let mut i = head[v];
                    while i != NONE {
                        let (at, fin, nx) = rec[i as usize];
                        for &(k, share) in use_k.iter() {
                            x[(at + k) as usize] += fin * share;
                        }
                        i = nx;
                    }
                }
            } else {
                let to = next[v].ix();
                if to >= n { continue; }
                flow[to] += f;
                if to >= s {
                    l.seg[v - s] += f;
                    l.pax_min += (f * g.rn_ride[v - s]) as f64;
                } else {
                    // Walking transfers alight at the route node's station, not the walk's end.
                    l.alight[g.rn_station[v - s] as usize] += f;
                    l.rn_alight[v - s] += f;
                    if to != dst {
                        if let Some((ix, _)) = xf.as_ref() {
                            let e = (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).find(|&e| g.adj_to[e] as usize == to).unwrap();
                            rec.push((ix.at[e], f, head[to]));
                            head[to] = (rec.len() - 1) as u32;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 7. Crowding

pub struct CrowdStats {
    pub segs_over_seats: usize,
    pub segs: usize,
    pub max_load_of_crush: f32,
    /// Rider-weighted percentiles of the penalty factor minus one.
    pub p50: f32,
    pub p90: f32,
}

/// Penalty per segment: 1 + alpha x(1+x)/2, x = max(load - seats, 0) / (crush - seats), per hour
/// at the peak hour. Starts where seats run out, never saturates (memory note on crowding).
pub fn crowd_factors(g: &Graph, net: &Network, loads: &Loads, p: &Params) -> (Vec<f32>, CrowdStats) {
    let nr = g.rn_line.len();
    let mut fac = vec![1.0f32; nr];
    let mut st = CrowdStats { segs_over_seats: 0, segs: 0, max_load_of_crush: 0.0, p50: 0.0, p90: 0.0 };
    let mut wv: Vec<(f32, f32)> = vec![];
    for r in 0..nr {
        if g.rn_ride[r] == 0.0 {
            continue;
        }
        st.segs += 1;
        let l = g.rn_line[r] as usize;
        let tph = 60.0 / net.line_headway_min[l][p.period];
        let seats = tph * net.line_seats[l];
        let crush = tph * net.line_crush[l];
        let hour = loads.seg[r] / p.period_hours * p.peak_hour_factor;
        let x = ((hour - seats) / (crush - seats)).max(0.0);
        fac[r] = 1.0 + p.crowd_alpha * x * (1.0 + x) / 2.0;
        if hour > seats {
            st.segs_over_seats += 1;
        }
        st.max_load_of_crush = st.max_load_of_crush.max(hour / crush);
        if loads.seg[r] > 0.0 {
            wv.push((fac[r] - 1.0, loads.seg[r] * g.rn_ride[r]));
        }
    }
    wv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let tot: f32 = wv.iter().map(|e| e.1).sum();
    let (mut acc, mut p50, mut p90) = (0.0f32, None, None);
    for &(v, w) in &wv {
        acc += w;
        if p50.is_none() && acc >= 0.5 * tot {
            p50 = Some(v);
        }
        if p90.is_none() && acc >= 0.9 * tot {
            p90 = Some(v);
        }
    }
    st.p50 = p50.unwrap_or(0.0);
    st.p90 = p90.unwrap_or(0.0);
    (fac, st)
}

/// Method of successive averages: `avg += (new - avg) / (round + 1)`.
pub fn msa(avg: &mut [f32], new: &[f32], round: usize) {
    let w = 1.0 / (round as f32 + 1.0);
    for (a, &n) in avg.iter_mut().zip(new) {
        *a += (n - *a) * w;
    }
}

// ---------------------------------------------------------------------------------------------
// 7b. Riders left behind by full trains (T-188)

/// The longest average extra wait, in real minutes, a boarding is given for full trains: a line
/// whose riders arrive faster than its trains take them all period long has a queue that never
/// clears, and its wait would grow without bound.
pub const LEFT_WAIT_CAP_MIN: f32 = 60.0;

/// What `left_behind` found over a period's network.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeftStats {
    /// Riders who could not get on the first train to leave after they arrived (the boarding
    /// account's "left behind", `RunAccount::left_h`).
    pub left: f64,
    /// Extra minutes waited for full trains, all riders together.
    pub rider_min: f64,
    /// Route nodes where anyone is left behind.
    pub stops: usize,
}

/// The boarding account's rule (track/boarding.rs, docs/track-and-service.md 6.6) run on one
/// period's own trains, to price riders left behind into the next crowding round (T-188): per
/// route node, the average extra real minutes a rider boarding there waits because full trains
/// left them on the platform.
///
/// Each direction of each line runs its period's trains at an even headway; riders arrive at each
/// stop over the period by the account's shape for that period (scaled so the busiest hour is
/// the busiest-hour factor times the mean, so riders arrive fastest mid-period); each train lets
/// off the stop's share alighting (demand's alightings over the riders arriving there) and takes
/// whoever waits up to its crush load. Whoever does not fit waits a headway more for the next
/// train of the same line and direction. A queue still there when the period ends is cleared at
/// the period's average spare room (a queue that grows for half an hour and clears in half an
/// hour adds its riders' half hour). Trains leaving before the period carry riders arriving at
/// the period's opening rate, so they reach the later stops with a load. `board` and `alight`
/// are the period's riders per route node, every population together.
pub fn left_behind(g: &Graph, net: &Network, p: &Params, board: &[f32], alight: &[f32]) -> (Vec<f32>, LeftStats) {
    use crate::track::boarding::{shape_cum, shape_g, shape_strength, SHAPE};
    let nr = g.rn_line.len();
    let mut out = vec![0.0f32; nr];
    let mut stats = LeftStats::default();
    if board.len() != nr || alight.len() != nr || p.period_hours <= 0.0 {
        return (out, stats);
    }
    let shape = SHAPE[p.period.min(SHAPE.len() - 1)];
    let hours = p.period_hours as f64;
    let c = shape_strength(shape, hours, p.peak_hour_factor as f64);
    let len = hours * 3600.0;
    let rate0 = 1.0 + c * shape_g(shape, 0.0);
    let mut alpha: Vec<f64> = vec![];
    let mut off: Vec<f64> = vec![];
    let mut carry: Vec<f64> = vec![];
    let mut r0 = 0usize;
    for l in 0..net.n_lines() {
        let n = (net.line_off[l + 1] - net.line_off[l]) as usize;
        for _dir in 0..2 {
            let run = r0..r0 + n;
            r0 += n;
            let tph = g.rn_tph[run.start] as f64;
            let cap = net.line_crush[l] as f64;
            let b = &board[run.clone()];
            if n < 2 || tph <= 0.0 || cap <= 0.0 || b.iter().map(|&v| v as f64).sum::<f64>() < 1.0 {
                continue;
            }
            let a = &alight[run.clone()];
            // the share of the riders aboard leaving at each stop, from the period's totals
            alpha.clear();
            let mut on = 0.0f64;
            for k in 0..n {
                let off_k = a[k] as f64;
                alpha.push(if k + 1 == n { 1.0 } else if on > 1e-6 { (off_k / on).clamp(0.0, 1.0) } else { 0.0 });
                on = (on + b[k] as f64 - off_k).max(0.0);
            }
            // seconds from the first stop to each stop
            off.clear();
            let mut t = 0.0f64;
            for k in 0..n {
                off.push(t);
                t += g.rn_ride[run.start + k] as f64 * 60.0;
            }
            let h = 3600.0 / tph;
            // trains leaving the first stop from a run and a headway before the period to its end
            let i0 = -((off[n - 1] + h) / h).ceil() as i64;
            let i1 = (len / h).ceil() as i64;
            carry.clear();
            carry.resize((i1 - i0).max(0) as usize, 0.0);
            for k in 0..n {
                let bk = b[k] as f64;
                let cum = |t: f64| if t <= 0.0 { bk / len * rate0 * t } else if t >= len { bk } else { bk * shape_cum(shape, c, t / len) };
                let (mut q, mut q_end, mut rider_s, mut left) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let (mut room_sum, mut trains) = (0.0f64, 0usize);
                for (j, i) in (i0..i1).enumerate() {
                    let t = (i as f64 + 0.5) * h + off[k];
                    let arrived = (cum(t) - cum(t - h)).max(0.0);
                    let on = carry[j] * (1.0 - alpha[k]);
                    let room = (cap - on).max(0.0);
                    let waiting = q + arrived;
                    let got_on = waiting.min(room);
                    q = waiting - got_on;
                    carry[j] = if k + 1 < n { on + got_on } else { 0.0 };
                    if (0.0..len).contains(&t) {
                        rider_s += q * h;
                        left += arrived.min(q);
                        room_sum += room;
                        trains += 1;
                        q_end = q;
                    }
                }
                if bk < 1.0 || trains == 0 {
                    continue;
                }
                if q_end > 0.0 {
                    // what is left clears at the room the period's trains had to spare on average
                    let spare = room_sum / trains as f64 * tph / 3600.0 - bk / len;
                    rider_s += if spare > 0.0 { (q_end * q_end / (2.0 * spare)).min(q_end * len) } else { q_end * len };
                }
                if rider_s > 0.0 {
                    out[run.start + k] = ((rider_s / bk / 60.0) as f32).min(LEFT_WAIT_CAP_MIN);
                    stats.left += left;
                    stats.rider_min += rider_s / 60.0;
                    stats.stops += 1;
                }
            }
        }
    }
    (out, stats)
}

/// The costs a crowding round is solved with, from the frozen loads of every population together
/// (`total`): the crowding factor per segment from the segment loads, and, when `total` also
/// carries the boardings and alightings per route node after them (`3 x` route nodes), the extra
/// wait for riders left behind (`left_behind`, T-188; none with the segment loads alone or with
/// `Params::left_behind` off). None if `total` has another length or a bad number.
pub fn round_costs(g: &Graph, net: &Network, p: &Params, total: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> {
    let nr = g.rn_line.len();
    if (total.len() != nr && total.len() != 3 * nr) || total.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return None;
    }
    let mut loads = Loads::zero(g.n_st, nr);
    loads.seg.copy_from_slice(&total[..nr]);
    let (fac, _) = crowd_factors(g, net, &loads, p);
    let left = if total.len() == 3 * nr && p.left_behind { left_behind(g, net, p, &total[nr..2 * nr], &total[2 * nr..]).0 } else { vec![0.0; nr] };
    Some((fac, left))
}

/// A period's crowding rounds stop once one moves its summed segment loads by under this share,
/// rider weighted (T-434); the driver caps them at three.
pub const SETTLED: f64 = 0.01;

/// Whether a period's next crowding round is skipped (T-434): `seg`, the segment loads it would be
/// priced on, is within `SETTLED` of those the round before was priced on, kept in `seen`. The first
/// crowding round always runs, as `seen` starts empty. Every population of a cluster gets the same
/// loads, so they all stop together.
pub fn settled(seen: &mut Vec<f32>, seg: &[f32]) -> bool {
    if seen.len() == seg.len() {
        let (mut moved, mut base) = (0f64, 0f64);
        for (a, b) in seg.iter().zip(seen.iter()) {
            moved += (a - b).abs() as f64;
            base += *b as f64;
        }
        if moved < SETTLED * base {
            return true;
        }
    }
    seen.clear();
    seen.extend_from_slice(seg);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::{network, synth};

    /// T-434: the first round always runs; a later one is skipped once the loads move under 1%.
    #[test]
    fn crowding_rounds_stop_once_settled() {
        let mut seen = vec![];
        assert!(!settled(&mut seen, &[100.0, 100.0]));
        assert!(!settled(&mut seen, &[100.0, 98.0]));
        assert!(settled(&mut seen, &[100.0, 96.5]));
        assert!(seen == [100.0, 98.0]);
        assert!(!settled(&mut seen, &[100.0, 95.0]));
    }

    fn common_test_net(lines: &[&[u32]], tph: &[f32], ride: &[f32]) -> Network {
        let mut net = Network {
            st_x: vec![0.0, 2000.0, 4000.0, 4000.0], st_y: vec![0.0, 0.0, 0.0, 2000.0],
            line_name: vec![String::new(); lines.len()], line_off: vec![0], line_stops: vec![],
            hop_min: vec![], hop_back: vec![], line_speed_kmh: vec![0.0; lines.len()],
            line_headway_min: tph.iter().map(|t| [if *t > 0.0 { 60.0 / t } else { f32::INFINITY }; network::PERIODS]).collect(),
            line_seats: vec![100.0; lines.len()], line_crush: vec![200.0; lines.len()],
            st_group: vec![],
        };
        for (l, stops) in lines.iter().enumerate() {
            net.line_stops.extend_from_slice(stops);
            net.line_off.push(net.line_stops.len() as u32);
            net.hop_min.extend((0..stops.len()).map(|k| if k + 1 < stops.len() { ride[l] } else { 0.0 }));
        }
        net
    }

    fn common_test_params() -> Params {
        Params { xfer_walk_m: 0.0, w_wait: 1.0, w_home_wait: 1.0, platform_wait_cap_min: 100.0, ..Params::default() }
    }

    #[test]
    fn everyday_rates_by_name() {
        let mut p = Params::default();
        p.set_extra("rate_social", 0.42).unwrap();
        p.set_extra("decay_medical", 3500.0).unwrap();
        assert_eq!((p.everyday_rate[1], p.everyday_decay_m[5]), (0.42, 3500.0));
        assert_eq!(p.everyday_rate[0], super::super::everyday::RATE[0]);
        assert!(p.set_extra("rate_shopping", 0.1).is_err());
    }

    /// Setting the worldwide rail constant by name keeps the country's part on top.
    #[test]
    fn country_rail_constant() {
        let mut p = Params::default();
        p.set_extra("country_asc_rail", super::super::api::country_asc_rail("JP")).unwrap();
        assert_eq!(p.asc_rail, super::super::api::ASC_RAIL + 0.5);
        p.set_extra("asc_rail", 0.0).unwrap();
        assert_eq!(p.asc_rail, 0.5);
        p.set_extra("country_asc_rail", super::super::api::country_asc_rail("US")).unwrap();
        assert_eq!(p.asc_rail, 0.0);
    }

    /// T-380's parking: by jobs per km² of land, the paid price, the free share and the cordon;
    /// with nothing to pay the two-logit mix is the one logit to the bit.
    #[test]
    fn parking_by_land_density() {
        let n = 5;
        let land = 4.0 / CELL_KM2;
        let ncell = vec![land.round() as u32, land.round() as u32, land.round() as u32, (land / 4.0).round() as u32, land.round() as u32];
        let km2: Vec<f64> = ncell.iter().map(|&c| c as f64 * CELL_KM2 as f64).collect();
        let dens = [2000.0, 10000.0, 200000.0, 10000.0, 8000.0];
        let z = Zones {
            zone_m: 2000.0, n, cell_zone: vec![], gx: vec![0; n], gy: vec![0; n], region: vec![0; n], w: 1, h: 1, x0: 0.0, y0: 0.0,
            pop: vec![0.0; n], jobs: (0..n).map(|j| dens[j] * km2[j]).collect(), ncell,
        };
        let mut p = Params::default();
        let near = |a: f32, b: f32| (a - b).abs() < 0.05;
        let (f0, paid0, free0) = p.park_parts(&z, 0);
        assert!(f0 == 3.0 && paid0 == 0.0 && near(free0, 0.8));
        let (_, paid1, _) = p.park_parts(&z, 1);
        assert!(near(paid1, 15.0 * (10000.0f32 / 3000.0).ln()));
        assert_eq!(p.park_parts(&z, 2).1, 50.0);
        // a quarter of the square on land: the same jobs per km² of land, the same price
        assert!(near(p.park_parts(&z, 3).1, paid1));
        assert!(near(p.park_parts(&z, 4).2, 0.5));
        p.cordon = Some(std::sync::Arc::new(vec![0.0, 0.0, 13.5, 0.0, 0.0]));
        assert_eq!(p.park_parts(&z, 2).0, 16.5);
        let ring = |name: &str| CORDONS.iter().find(|c| c.0 == name).unwrap().2;
        // Times Square and Wall St in, Downtown Brooklyn and the Upper East Side out
        assert!(in_ring(ring("Manhattan"), -73.985, 40.758) && in_ring(ring("Manhattan"), -74.009, 40.706));
        assert!(!in_ring(ring("Manhattan"), -73.990, 40.693) && !in_ring(ring("Manhattan"), -73.958, 40.775));
        // Bank and Oxford Circus in, Canary Wharf out
        assert!(in_ring(ring("London"), -0.089, 51.513) && in_ring(ring("London"), -0.142, 51.515));
        assert!(!in_ring(ring("London"), -0.024, 51.505));
        for util in [-3.0f32, 0.0, 2.5] {
            assert_eq!(p.rail_share_parked(util, p.beta, 0.0, 0.3), 1.0 / (1.0 + fast_exp(-util)));
            let r = p.rail_share_parked(util, p.beta, 20.0, 0.3);
            let (r0, r1) = (1.0 / (1.0 + fast_exp(-util)), 1.0 / (1.0 + fast_exp(-(util + p.beta * 20.0))));
            assert!(r > r0 && r < r1 && (r - (0.3 * r0 + 0.7 * r1)).abs() < 1e-6);
        }
    }

    #[test]
    fn common_lines_pool_wait_and_split_by_frequency() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 2.0]);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        assert!((sk.r[1] - (2.0 + 30.0 / 18.0)).abs() < 1e-5);
        let mut od = vec![0.0; 16]; od[1] = 90.0; od[4] = 30.0;
        let loads = assign(&g, &sk, &od);
        assert!((loads.seg[0] - 30.0).abs() < 1e-5);
        assert!((loads.seg[4] - 60.0).abs() < 1e-5);
        assert!((loads.seg[2] - 10.0).abs() < 1e-5);
        assert!((loads.seg[6] - 20.0).abs() < 1e-5);
        assert_eq!(loads.board.iter().sum::<f32>(), 120.0);
        assert_eq!(loads.alight.iter().sum::<f32>(), 120.0);
        assert_eq!(loads.rn_board.iter().sum::<f32>(), 120.0);
        assert_eq!(loads.rn_alight.iter().sum::<f32>(), 120.0);
        // Two identical 6-tph lines give exactly the same cost as one 12-tph line.
        let one = skims(&build_graph(&common_test_net(&[&[0, 1]], &[12.0], &[2.0]), &p), &p);
        let two = skims(&build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]), &p), &p);
        assert_eq!(one.r, two.r);
    }

    #[test]
    fn common_lines_are_destination_specific_and_ignore_unhelpful_service() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
        let g = build_graph(&net, &p); let sk = skims(&g, &p);
        assert!((sk.r[1] - 4.5).abs() < 1e-5, "shared trunk has combined 12 tph");
        assert!((sk.r[2] - 9.0).abs() < 1e-5, "branch has only its useful 6 tph");
        let mut od = vec![0.0; 16]; od[2] = 100.0;
        let loads = assign(&g, &sk, &od);
        assert_eq!(loads.rn_board[0], 100.0);
        assert_eq!(loads.rn_board[6], 0.0);
        // A train that is off, or too slow, adds no attractive departures.
        for (frequency, slow) in [(0.0, 2.0), (6.0, 100.0)] {
            let g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, frequency], &[2.0, slow]), &p);
            let sk = skims(&g, &p);
            assert!((sk.r[1] - 7.0).abs() < 1e-5);
            assert_eq!(sk.board_tph(&g, 1)[4], 0.0);
        }
    }

    /// T-552: a line leaving riders behind joins a station's strategy at its cost with the extra
    /// wait, so a full train ahead of better options is no longer kept. Line 0-1-2-3, 12 tph, 2
    /// minutes a stop, riders 1 -> 3; the southbound at 1 leaves riders behind for 60 minutes.
    #[test]
    fn a_full_line_joins_at_its_cost_with_the_wait() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[2.0]);
        let mut g = build_graph(&net, &p);
        let (south, north) = (1, 6);
        g.rn_left[south] = 60.0;
        let sk = skims(&g, &p);
        let tph = sk.board_tph(&g, sk.slot[3] as usize);
        // north to 0 and back: two waits of 2.5, 2 + 6 riding
        assert!((sk.r[1 * 4 + 3] - 13.0).abs() < 1e-4, "{}", sk.r[1 * 4 + 3]);
        assert_eq!((tph[south], tph[north]), (0.0, 12.0));
        // a shorter wait leaves the direct train the better choice, the back ride unused
        g.rn_left[south] = 2.0;
        let sk = skims(&g, &p);
        let tph = sk.board_tph(&g, sk.slot[3] as usize);
        assert!((sk.r[1 * 4 + 3] - (2.0 + 2.5 + 4.0)).abs() < 1e-4, "{}", sk.r[1 * 4 + 3]);
        assert_eq!((tph[south], tph[north]), (12.0, 0.0));
    }

    #[test]
    fn common_lines_use_expected_ride_cost_and_respect_fare_and_crowding() {
        let p = common_test_params();
        let mut g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 3.0]), &p);
        let sk = skims(&g, &p);
        assert!((sk.r[1] - (30.0 / 18.0 + (6.0 * 2.0 + 12.0 * 3.0) / 18.0)).abs() < 1e-5);
        g.rn_fare[4] = 100.0;
        let uncrowded = vec![1.0; g.n - g.n_st];
        set_crowding(&mut g, &uncrowded);
        let expensive = skims(&g, &p);
        assert!((expensive.r[1] - 7.0).abs() < 1e-5);
        g.rn_fare[4] = 0.0;
        let mut factor = vec![1.0; g.n - g.n_st]; factor[4] = 100.0;
        set_crowding(&mut g, &factor);
        let crowded = skims(&g, &p);
        assert!((crowded.r[1] - 7.0).abs() < 1e-5);
    }

    /// T-119: strategies searched towards some stations only are the full ones for those stations,
    /// the cache extends what it has instead of starting again, and flows into searched stations
    /// load the same as with every station searched.
    #[test]
    fn partial_skims_equal_full_skims_where_searched() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
        let g = build_graph(&net, &p);
        let full = skims(&g, &p);
        let s = g.n_st;
        let part = skims_for(&g, &p, &[false, true, false, true]);
        for dst in 0..s {
            for src in 0..s {
                let (a, b) = (part.r[src * s + dst], full.r[src * s + dst]);
                if dst == 1 || dst == 3 {
                    assert_eq!(a, b, "r {src} -> {dst}");
                } else {
                    assert!(a.is_infinite(), "r {src} -> {dst} was not searched");
                }
            }
        }
        assert_eq!(part.order.len(), 2);
        let mut od = vec![0.0; s * s];
        od[1] = 40.0; // station 0 to 1
        od[3] = 25.0; // station 0 to 3
        od[s * 2 + 3] = 5.0;
        let (lf, lp) = (assign(&g, &full, &od), assign(&g, &part, &od));
        assert_eq!(lf.seg, lp.seg);
        assert_eq!(lf.board, lp.board);
        assert_eq!(lf.rn_alight, lp.rn_alight);
        // the cache: ask for 1 and 3, then for 2 as well; every answer is the full table's column
        clear_skim_cache();
        with_skims(&g, &p, &[false, true, false, true], |sk| assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 2));
        with_skims(&g, &p, &[false, false, true, false], |sk| {
            assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 3);
            for dst in [1, 2, 3] {
                for src in 0..s {
                    assert_eq!(sk.r[src * s + dst], full.r[src * s + dst]);
                }
            }
            assert_eq!(assign(&g, sk, &od).seg, lf.seg);
        });
        // a different graph (another crowding) is a different entry
        let mut g2 = build_graph(&net, &p);
        let mut factor = vec![1.0; g2.n - g2.n_st];
        factor[0] = 3.0;
        set_crowding(&mut g2, &factor);
        let full2 = skims(&g2, &p);
        with_skims(&g2, &p, &[true; 4], |sk| assert_eq!(sk.r, full2.r));
        with_skims(&g, &p, &[true; 4], |sk| assert_eq!(sk.r, full.r));
        // T-405: a third graph wanting fewer searches than either kept table evicts neither
        let mut g3 = build_graph(&net, &p);
        factor[0] = 2.0;
        set_crowding(&mut g3, &factor);
        let full3 = skims(&g3, &p);
        with_skims(&g3, &p, &[false; 4], |sk| assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 0));
        with_skims(&g3, &p, &[false, true, false, false], |sk| {
            assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 1);
            assert_eq!((0..s).map(|src| sk.r[src * s + 1]).collect::<Vec<_>>(), (0..s).map(|src| full3.r[src * s + 1]).collect::<Vec<_>>());
        });
        with_skims(&g2, &p, &[false; 4], |sk| assert_eq!(sk.r, full2.r));
        with_skims(&g, &p, &[false; 4], |sk| assert_eq!(sk.r, full.r));
        clear_skim_cache();
    }

    /// T-176: transfers are counted platform to platform. A to C changes at B from line 0 to line 1
    /// (and back on the way home); riders who end at B do not transfer.
    #[test]
    fn transfers_between_route_nodes() {
        let p = common_test_params();
        // line 0: 0-1-2, line 1: 1-3 (stations 1 and 2 at 2 and 4 km on x, 3 is 2 km north of 2)
        let net = common_test_net(&[&[0, 1, 2], &[1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let s = g.n_st;
        let mut od = vec![0.0; s * s];
        od[3] = 100.0; // 0 to 3: line 0 to B, line 1 on
        od[s * 3] = 40.0; // 3 to 0: line 1 to B, line 0 back
        od[1] = 7.0; // 0 to 1: no transfer
        let ix = xfer_index(&g);
        let mut x = vec![0.0; ix.len];
        let l = assign_with_transfers(&g, &sk, &od, &ix, &mut x);
        let plain = assign(&g, &sk, &od);
        assert_eq!((l.seg, l.board, l.rn_board, l.rn_alight), (plain.seg.clone(), plain.board.clone(), plain.rn_board.clone(), plain.rn_alight.clone()));
        let mut out = vec![];
        transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
        // route nodes: line 0 run 0 = 0,1,2 (stations 0,1,2), run 1 = 3,4,5 (2,1,0); line 1 run 0 =
        // 6,7 (1,3), run 1 = 8,9 (3,1)
        let mut pairs: Vec<(u32, u32, f32)> = out.chunks(3).map(|c| (c[0] as u32, c[1] as u32, c[2])).collect();
        pairs.sort_by_key(|p| (p.0, p.1));
        assert_eq!(pairs, vec![(1, 6, 100.0), (9, 4, 40.0)]);
        // only pairs touching the selection
        let mut sel = vec![0u8; g.n - s];
        sel[4] = 1;
        let mut out = vec![];
        transfer_pairs(&g, &ix, &x, &sel, &mut out);
        assert_eq!(out, vec![9.0, 4.0, 40.0]);
    }

    /// T-161: stations of one complex are walking transfers whatever the distance, costed like any
    /// walk; without the complex 2 km is out of reach and the trip has no rail.
    #[test]
    fn complex_members_are_walking_transfers() {
        let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
        // line 0: 0-1, line 1: 2-3; stations 1 and 2 are 2 km apart
        let mut net = common_test_net(&[&[0, 1], &[2, 3]], &[6.0, 6.0], &[2.0, 2.0]);
        let walk_edge = |g: &Graph, r: usize, to: u32| (g.adj_off[g.n_st + r] as usize..g.adj_off[g.n_st + r + 1] as usize).find(|&e| g.adj_to[e] == to).map(|e| g.adj_cost[e]);
        let g = build_graph(&net, &p);
        assert_eq!(walk_edge(&g, 1, 2), None);
        assert!(skims(&g, &p).r[3].is_infinite());
        net.st_group = vec![0, 1, 1, 3];
        let g = build_graph(&net, &p);
        // route nodes: line 0 run 0 = 0, 1 (stations 0, 1); line 1 run 0 = 4, 5 (2, 3)
        assert_eq!(walk_edge(&g, 1, 2), Some(p.xfer_min(2000.0) * p.w_xfer));
        assert_eq!(walk_edge(&g, 4, 1), Some(p.xfer_min(2000.0) * p.w_xfer));
        assert_eq!(walk_edge(&g, 1, 0), None, "a station outside the complex stays out of reach");
        let sk = skims(&g, &p);
        assert!(sk.r[3].is_finite());
        let s = g.n_st;
        let mut od = vec![0.0; s * s];
        od[3] = 100.0;
        let ix = xfer_index(&g);
        let mut x = vec![0.0; ix.len];
        assign_with_transfers(&g, &sk, &od, &ix, &mut x);
        let mut out = vec![];
        transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
        assert_eq!(out, vec![1.0, 4.0, 100.0]);
        // a group within the walk limit adds nothing: no edge twice
        net.st_x = vec![0.0, 2000.0, 2100.0, 4000.0];
        let a = build_graph(&net, &p).adj_to.len();
        net.st_group = vec![];
        assert_eq!(build_graph(&net, &p).adj_to.len(), a);
        assert_eq!(complex_members(&[5, 2, 5, 7, 2, 2]), vec![vec![0, 2], vec![1, 4, 5]]);
    }

    /// T-176 on a small city with walking transfers: the loads are `assign`'s to the bit, and at
    /// every station the transfers leaving are the boardings that are not first boardings.
    #[test]
    fn transfers_account_for_every_boarding_after_the_first() {
        let full = synth::synth_city(1);
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 15_000.0).collect();
        let city = crate::demand::pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: vec![], gravity: None,
        };
        let net = network::synth_network();
        let p = Params::default();
        assert!(p.xfer_walk_m > 0.0);
        let z = build_zones(&city, p.zone_m);
        let mut gr = gravity(&z, &p, 100, 1e-4);
        gr.set_walk(&z, &p);
        let acc = build_access(&city, &net, &p);
        let sz = build_subzones(&city, &z, &acc, &net, &p);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let mo = mode_choice(&z, &gr, &sz, &station_exp(&sk, &p), g.n_st, &p);
        let ix = xfer_index(&g);
        let mut x = vec![0.0; ix.len];
        let l = assign_with_transfers(&g, &sk, &mo.od, &ix, &mut x);
        let plain = assign(&g, &sk, &mo.od);
        assert_eq!(l.seg, plain.seg);
        assert_eq!(l.board, plain.board);
        assert_eq!(l.alight, plain.alight);
        assert_eq!(l.rn_board, plain.rn_board);
        assert_eq!(l.rn_alight, plain.rn_alight);
        assert_eq!(l.entry, plain.entry);
        assert_eq!(l.exit, plain.exit);
        let s = g.n_st;
        let mut out = vec![];
        transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
        let mut leaving = vec![0f64; s];
        let mut from = vec![0f64; g.n - s];
        let mut walked = 0.0;
        for c in out.chunks(3) {
            let (a, b) = (c[0] as usize, c[1] as usize);
            leaving[g.rn_station[b] as usize] += c[2] as f64;
            from[a] += c[2] as f64;
            if g.rn_station[a] != g.rn_station[b] {
                walked += c[2] as f64;
            }
        }
        let total: f64 = leaving.iter().sum();
        // (the synthetic network has no stations close enough to walk between: the real network
        // test `transfers_real_network` counts walks)
        assert!(total > 0.0, "transfers {total}, walked {walked}");
        for st in 0..s {
            let want = (l.board[st] - l.entry[st]) as f64;
            assert!((leaving[st] - want).abs() <= 1e-3 * (l.board[st] as f64).max(1.0), "station {st}: {} leaving, {want} boardings after the first", leaving[st]);
        }
        for r in 0..g.n - s {
            assert!(from[r] <= l.rn_alight[r] as f64 * 1.0001 + 1e-3, "route node {r}: {} transfer, {} alight", from[r], l.rn_alight[r]);
        }
    }

    #[test]
    fn fast_exp_is_close() {
        let mut x = -30.0f32;
        while x < 30.0 {
            let (a, b) = (fast_exp(x), x.exp());
            assert!(((a - b) / b).abs() < 3e-4, "x {x}: {a} vs {b}");
            x += 0.0137;
        }
    }

    /// Small end-to-end run: gravity conserves trips, rail trips are a share of them, every rail
    /// trip shows up as a boarding.
    #[test]
    fn kernel_runs_on_a_small_city() {
        let full = synth::synth_city(1);
        // keep the inner 15 km so the test is quick
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 15_000.0).collect();
        let city = crate::demand::pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: vec![], gravity: None,
        };
        let net = network::synth_network();
        let p = Params::default();
        let z = build_zones(&city, p.zone_m);
        let mut gr = gravity(&z, &p, 100, 1e-4);
        gr.set_walk(&z, &p);
        let (_, tj) = city.totals();
        assert!((gr.total / tj - 1.0).abs() < 1e-3);
        let acc = build_access(&city, &net, &p);
        let sz = build_subzones(&city, &z, &acc, &net, &p);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let er = station_exp(&sk, &p);
        let mo = mode_choice(&z, &gr, &sz, &er, g.n_st, &p);
        assert!(mo.rail > 0.0 && mo.rail < mo.trips_access && mo.trips_access <= mo.trips_total * 1.0001);
        let l = assign(&g, &sk, &mo.od);
        let od_sum: f64 = mo.od.iter().map(|&v| v as f64).sum();
        // the split back onto access stations conserves rail trips
        assert!((od_sum / mo.rail - 1.0).abs() < 1e-3, "od {od_sum} vs rail {}", mo.rail);
        let bands: f64 = mo.rail_band.iter().sum();
        assert!((bands / mo.rail - 1.0).abs() < 1e-3);
        // rail trips add up the same by home subzone, work subzone and access entry at each end
        let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
        for (what, v) in [("by", &mo.rail_by_sub), ("to", &mo.rail_to_sub), ("home", &mo.acc_home), ("work", &mo.acc_work)] {
            assert!((sum(v) / mo.rail - 1.0).abs() < 1e-3, "{what}: {} vs {}", sum(v), mo.rail);
        }
        let (wb, wt) = (sum(&mo.walk_lost_by_sub), sum(&mo.walk_lost_to_sub));
        assert_eq!((wb, wt, mo.rail_walk), (0.0, 0.0, 0.0), "rail cannot take reserved walking trips");
        let boards: f64 = l.board.iter().map(|&v| v as f64).sum();
        assert!(boards >= od_sum * 0.999);
    }

    /// T-162: a region border (here a north-south line at x = 300 m, through the middle of a
    /// column of zones) splits the squares it cuts, the factor cuts trips across it, both margins
    /// still balance, and the pack round trip rebuilds exactly the trips solved.
    #[test]
    fn region_border_factor() {
        let full = synth::synth_city(1);
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 15_000.0).collect();
        let mut city = crate::demand::pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: vec![],
            gravity: None,
        };
        let p1 = Params::default();
        let z0 = build_zones(&city, p1.zone_m);
        let g0 = gravity(&z0, &p1, 200, 1e-6);
        city.region = city.x.iter().map(|&x| (x > 300.0) as u8).collect();
        let z = build_zones(&city, p1.zone_m);
        assert!(z.n > z0.n, "the border splits squares");
        let (off, _) = z.by_square();
        assert_eq!(off.windows(2).filter(|w| w[1] > w[0]).count(), z0.n, "same squares");
        let cross = |z: &Zones, g: &Gravity| -> f64 {
            let mut s = 0.0;
            for i in 0..z.n {
                for j in 0..z.n {
                    if z.region[i] != z.region[j] {
                        s += g.t(z, i, j);
                    }
                }
            }
            s
        };
        // factor 1: the split changes nothing but where the intrazonal trips of cut squares go
        let g1 = gravity(&z, &p1, 200, 1e-6);
        assert!((g1.total / g0.total - 1.0).abs() < 1e-6);
        let pk = Params { border_factor: 0.25, ..Params::default() };
        let gk = gravity(&z, &pk, 200, 1e-6);
        assert!((gk.total / g0.total - 1.0).abs() < 1e-6);
        let (c1, ck) = (cross(&z, &g1), cross(&z, &gk));
        assert!(ck < 0.6 * c1, "cross-border trips {ck} with the factor, {c1} without");
        // both margins hold with the factor
        let (tp, tj): (f64, f64) = (z.pop.iter().sum(), z.jobs.iter().sum());
        for i in 0..z.n {
            let (r, c): (f64, f64) = ((0..z.n).map(|j| gk.t(&z, i, j)).sum(), (0..z.n).map(|j| gk.t(&z, j, i)).sum());
            assert!((r - z.pop[i] * tj / tp).abs() <= 1e-3 * z.pop[i] * tj / tp + 1e-6, "row {i}");
            assert!((c - z.jobs[i]).abs() <= 1e-3 * z.jobs[i] + 1e-6, "column {i}");
        }
        // shipped and rebuilt: the same zones, factor and trips
        city.gravity = Some(to_pack_gravity(&z, &gk, &pk));
        let (h, b) = crate::demand::pack::write(&city);
        let back = crate::demand::pack::parse(&h, &b).unwrap();
        let (zb, gb, from_pack) = city_gravity(&back, &p1);
        assert!(from_pack && zb.n == z.n && zb.region == z.region && gb.border == 0.25);
        for (i, j) in [(0, 1), (3, z.n - 1), (z.n / 2, z.n / 3)] {
            assert!((gb.t(&zb, i, j) / gk.t(&z, i, j).max(1e-30) - 1.0).abs() < 1e-5 || gk.t(&z, i, j) < 1e-12);
        }
    }

    /// The ring search finds the same K nearest stations within the bound as a full scan.
    #[test]
    fn access_matches_a_full_scan() {
        let city = synth::synth_city(3);
        let net = network::synth_network();
        for cut in [800.0f32, 2500.0, 5000.0] {
            let p = Params { walk_cutoff_m: cut, ..Params::default() };
            let a = build_access(&city, &net, &p);
            for c in (0..city.len()).step_by(37) {
                let mut d: Vec<(f32, u32)> = (0..net.n_stations())
                    .map(|s| ((net.st_x[s] - city.x[c]).hypot(net.st_y[s] - city.y[c]), s as u32))
                    .filter(|e| e.0 <= cut)
                    .collect();
                d.sort_by(|x, y| x.partial_cmp(y).unwrap());
                d.truncate(K);
                assert_eq!(a.n[c] as usize, d.len(), "cell {c} cut {cut}");
                for (k, e) in d.iter().enumerate() {
                    assert!((a.walk[c * K + k] - p.walk_min(e.0)).abs() < 1e-3, "cell {c} k {k}");
                }
            }
        }
    }

    /// The walk's perceived cost: `w_walk` a minute for the first `walk_easy_min`, `w_walk_far`
    /// after (T-371: the same, a linear walk); the drive is slower in dense places and on short
    /// trips.
    #[test]
    fn access_and_drive_costs() {
        let p = Params::default();
        let easy_m = p.walk_easy_min * 60.0 * p.walk_mps / p.walk_detour;
        assert!((p.access_cost(easy_m) - p.walk_easy_min * p.w_walk).abs() < 1e-3);
        let km = |d: f32| p.access_cost(d + 1000.0) - p.access_cost(d);
        assert!((km(easy_m + 100.0) / km(0.0) - p.w_walk_far / p.w_walk).abs() < 1e-3);
        // 13 real minutes a straight km at 6 km/h on 1.3x the line
        assert!((p.walk_min(1000.0) - 13.0).abs() < 1e-3);
        let (dense, open) = (p.place_min_per_km(50_000.0), p.place_min_per_km(100.0));
        assert!(dense.0 > open.0 && dense.1 > open.1);
        // 30 km between open places is much quicker a km than 3 km
        assert!(p.drive_min(30_000.0, open, open) / 30.0 < 0.6 * p.drive_min(3_000.0, open, open) / 3.0);
        assert!(p.drive_min(10_000.0, dense, dense) > 2.0 * p.drive_min(10_000.0, open, open));
    }

    /// T-370: a long drive into a dense place is slow only near its ends (real peak drives of
    /// 40-60 km into Manhattan, Philadelphia or Boston take 50-90 minutes); a short one is all
    /// at its ends' speeds.
    #[test]
    fn long_drive_slow_only_near_its_ends() {
        let p = Params::default();
        let (city, suburb) = (p.place_min_per_km(30_000.0), p.place_min_per_km(1_000.0));
        let t = p.drive_min(60_000.0, suburb, city);
        assert!((60.0..90.0).contains(&t), "60 km into a 30,000/km² place: {t} min");
        let road = 60.0 * p.road_detour;
        let t_end = p.road_local_km * (suburb.0 + city.0) + p.road_end_km * (suburb.1 + city.1);
        let open = (road - 2.0 * (p.road_local_km + p.road_end_km)) * 60.0 / p.road_open_kmh;
        assert!((t - t_end - open).abs() < 1e-3);
        let road = 15.0 * p.road_detour;
        let short = p.road_local_km * (suburb.0 + city.0) + (road - 2.0 * p.road_local_km) * 0.5 * (suburb.1 + city.1);
        assert!((p.drive_min(15_000.0, suburb, city) - short).abs() < 1e-3);
    }

    #[test]
    fn fast_ln_is_close() {
        let mut x = 1e-30f32;
        while x < 1e30 {
            let (a, b) = (fast_ln(x), x.ln());
            assert!((a - b).abs() < 4e-6 * b.abs().max(1.0), "x {x}: {a} vs {b}");
            x *= 1.0137;
        }
    }
}
