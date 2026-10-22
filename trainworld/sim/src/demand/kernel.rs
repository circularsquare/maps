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
use std::collections::HashMap;

/// Stations kept per cell and per subzone.
pub const K: usize = 6;
/// Dummy slots after each row of the commute pair loop, one per lane and egress step (T-036).
const PADS: usize = 8 * K;
const NONE: u32 = u32::MAX;

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
    /// A fixed cost per change of train, perceived minutes (Anita, T-717): a change costs this, its
    /// wait and the walk between the two platforms.
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
    /// The choice curve's floor and ceiling (T-854, `rail_bounds`), on rail's share of the trips
    /// not walking where rail is offered: `rail_floor` of them always take rail while it costs at
    /// most the car's perceived minutes, fading linearly to none at `rail_floor_ratio` times the
    /// car's; mirrored, as many never take it while the car costs at most rail's, fading to none
    /// where rail costs 1 / `rail_floor_ratio` of the car's. 0: the plain curve.
    pub rail_floor: f32,
    pub rail_floor_ratio: f32,
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
    /// Parking at the home end (T-854, `home_park_at`): every car trip of the local layers from a
    /// home pays `home_park_f` x the paid-space rule above on its zone's residents per km² of land
    /// (`home_dens`, per cell; None: none anywhere). The ride to a station pays it too.
    pub home_park_f: f32,
    pub home_dens: Option<std::sync::Arc<Vec<f32>>>,
    /// Owning a car, amortized over its trips (T-620, T-621): every car trip pays
    /// `car_own_usd_trip` dollars (`car_own_min`, in `park_parts`' fixed part) plus
    /// `car_own_usd_km` a km of its own road (`car_drive`), at `FARE_MIN_PER_USD` perceived
    /// minutes a dollar; a drive to a station pays both on its own road.
    pub car_own_usd_trip: f32,
    pub car_own_usd_km: f32,
    /// A congestion charge per zone (`cordon_by_zone`), perceived minutes a one-way trip, paid
    /// by every driver arriving there. None: no charge anywhere.
    pub cordon: Option<std::sync::Arc<Vec<f32>>>,
    pub period: usize,
    /// Share of all home->work trips in this period.
    pub period_share: f32,
    pub period_hours: f32,
    pub peak_hour_factor: f32,
    pub crowd_alpha: f32,
    /// The crowding penalty starts at this share of crush, or at the seats if they are more
    /// (T-710, `crowd_factors`).
    pub crowd_start: f32,
    /// Zone pairs carrying fewer trips than this are skipped in mode choice.
    pub prune_trips: f32,
    /// T-874: an origin zone's pairs carrying fewer trips than this in the period are dealt
    /// those trips in chunks of about this size, per block of `CHUNK_BLOCK` destination zones
    /// (`mode_choice`); 0 solves every pair.
    pub commute_chunk: f32,
    /// Subzone pairs whose cheapest walks at the two ends together cost more than this many
    /// perceived minutes are not offered rail (T-090: a computational bound like
    /// `walk_cutoff_m`, set where little rail is left).
    pub pair_walk_max: f32,
    /// Logit scale of the choice among access and egress stations (and access modes), per
    /// perceived minute, where `rel_gamma` is 0 (every layer then, the regional one too, T-718).
    /// notes/T-020.md.
    pub station_theta: f32,
    /// T-710: the local layers' scale is `rel_gamma` / min(T, `rel_tmax`), T = `rel_t0` +
    /// `rel_tkm` x the straight-line km of the trip (a rough rail trip in perceived minutes), by
    /// the distance bands `REL_EDGES` (`Params::band`). 0: `station_theta` for every trip.
    pub rel_gamma: f32,
    pub rel_t0: f32,
    pub rel_tkm: f32,
    pub rel_tmax: f32,
    /// Anita, 2026-10-19 (T-717): with 1, past `rel_tmax` the scale falls further as
    /// sqrt(`rel_tmax` / T), so a long trip is less sensitive to a minute (`Params::soften`); 0 off.
    pub rel_soft: f32,
    /// Walking the whole way, a mode in the choice beside rail and the car (or other) (T-558,
    /// `walk_trip_cost`): the walk's real minutes (its distance x `walk_detour` at `walk_mps`)
    /// weigh `walk_trip_w` each, at one rate. `walk_bonus_min`, in perceived minutes, is what the
    /// car's trip costs beyond its road time against a walk (the logit's `asc_walk`, T-636).
    /// Walks over `WALK_REACH_KM` are not offered.
    pub walk_trip_w: f32,
    pub walk_bonus_min: f32,
    /// Commutes walk between cells, not zone centres (T-587): each zone's homes and its jobs are
    /// this many cells, at even steps of their weight, and a zone pair within `WALK_FINE_R`
    /// squares walks at the logsum (scale `beta`) of its cell pairs' walk costs
    /// (`Gravity::set_walk`). 0: at the zone-centre distance.
    pub walk_samples: usize,
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
    /// The everyday layer's outings per resident (per enrolled student for school and
    /// university) and destination decay lengths in metres, by `everyday::KINDS` (T-509 stage
    /// 4). Read when `EverydayApi::beside` opens; the pack's `everyday` block sets the rates.
    pub everyday_rate: [f32; 6],
    pub everyday_decay_m: [f32; 6],
    /// The everyday layer's rail constant is `asc_rail` plus this (`Params::everyday`); 0: the
    /// two layers share their constants (T-636), the gap kept for a later fit.
    pub everyday_asc_rail: f32,
}

/// Walks longer than this straight line (km, round water where the routes say) are not offered
/// (T-558): a computational bound, past which walking takes under one trip in a million.
pub const WALK_REACH_KM: f32 = 10.0;
/// T-710: edges (straight-line km) of the station choice's distance bands (`Params::band`); from
/// 12 km on the trip counts as 60 minutes or more, where the scale stops changing.
pub const REL_EDGES: [f32; 5] = [0.0, 3.0, 7.0, 12.0, f32::INFINITY];
/// A walk's cost past `WALK_REACH_KM`: its share is 0 in every logit.
pub const NO_WALK: f32 = 1e6;
/// Zone pairs up to this many squares apart each way walk between cells (T-587); past it (cells
/// at least 6 km apart for 2 km zones) almost nobody walks and the zone centres do.
pub const WALK_FINE_R: i32 = 3;

/// `Params::walk_trip_cost` as the gravity keeps it.
#[derive(Clone, Copy, Debug, Default)]
pub struct WalkCost {
    min_per_km: f32,
    w: f32,
}

impl WalkCost {
    #[inline]
    pub fn at(&self, d_km: f32) -> f32 {
        if d_km > WALK_REACH_KM {
            return NO_WALK;
        }
        d_km * self.min_per_km * self.w
    }
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
            // T-720: a zone's cells 1 to 30 minutes from one station are not one walk
            band_min: 15.0,
            decay_km: 30.0,
            decay_pow: 0.8,
            border_factor: 1.0,
            // T-371, T-372: 1.5 rather than the usual 2.0 (notes/T-371.md, T-372.md)
            w_walk: 1.5,
            transfer_min: 2.0,
            fare_min: 6.0,
            // T-636: Anita's set (notes/T-636.md)
            beta: 0.10,
            asc_rail: super::api::ASC_RAIL,
            country_asc_rail: 0.0,
            damp_min: 60.0,
            // T-854: 5%, gone where rail costs 3 times the car's (notes/T-854.md "Stage 2")
            rail_floor: 0.05,
            rail_floor_ratio: 3.0,
            road_detour: 1.3,
            road_local_km: 2.0,
            road_end_km: 8.0,
            road_open_kmh: 100.0,
            road_city_kmh: 15.0,
            road_town_kmh: 40.0,
            road_dens50: 5000.0,
            road_far_kmh: 80.0,
            w_drive: 1.33,
            park_min: 7.0,
            // T-380: within about half of real prices from Croydon to Midtown (notes/T-380.md)
            park_paid_min: 15.0,
            park_paid_d0: 3000.0,
            park_paid_cap_min: 50.0,
            park_free_d: 8000.0,
            // T-854: dense cores keep about the rail share the car-free split gave them
            home_park_f: 0.4,
            home_dens: None,
            car_own_usd_trip: 1.5,
            car_own_usd_km: 0.1,
            cordon: None,
            period: 0,
            period_share: 0.45,
            period_hours: 3.0,
            peak_hour_factor: 1.25,
            crowd_alpha: 2.0,
            // T-710: Anita's 70% (notes/T-710.md)
            crowd_start: 0.7,
            prune_trips: 0.0,
            commute_chunk: COMMUTE_CHUNK,
            // T-090: holds the pairs at what the 2.5 km edge had; on the real New York network
            // 0.3% of rail trips have walks of 70-80 perceived minutes together and 0.2% were
            // beyond it (a lone line in Manhattan, the worst case: 2.3%, and the bounds cost it
            // 4% of its riders)
            pair_walk_max: 80.0,
            station_theta: 0.2,
            // T-710: Anita's relative rule, capped at a 60 minute trip (notes/T-710.md)
            rel_gamma: 20.0,
            rel_t0: 25.0,
            rel_tkm: 3.0,
            rel_tmax: 60.0,
            rel_soft: 1.0,
            // T-558: the cost as Anita approved it; the constants hold New York's and London's
            // walking as it was (notes/T-558.md; commutes refitted for the cells' walks, T-587)
            walk_trip_w: 1.5,
            walk_bonus_min: 10.0,
            walk_samples: 6,
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
            everyday_rate: super::everyday::RATE,
            everyday_decay_m: super::everyday::DECAY,
            everyday_asc_rail: 0.0,
        }
    }
}

impl Params {
    /// Set a parameter by name, for the checking tools (`realnet --set`); not every field.
    pub fn set_extra(&mut self, k: &str, v: f32) -> Result<(), String> {
        match k {
            "station_theta" => self.station_theta = v,
            "rel_gamma" => self.rel_gamma = v,
            "rel_t0" => self.rel_t0 = v,
            "rel_tkm" => self.rel_tkm = v,
            "rel_tmax" => self.rel_tmax = v,
            "rel_soft" => self.rel_soft = v,
            "crowd_start" => self.crowd_start = v,
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
            // the gravity holds the walk's costs: `DemandApi::refresh_walk` after these
            "walk_trip_w" => self.walk_trip_w = v,
            "walk_bonus_min" => self.walk_bonus_min = v,
            "walk_samples" => self.walk_samples = v as usize,
            "damp_min" => self.damp_min = v,
            "rail_floor" => self.rail_floor = v,
            "rail_floor_ratio" => self.rail_floor_ratio = v,
            "walk_mps" => self.walk_mps = v,
            "walk_cutoff_m" => self.walk_cutoff_m = v,
            "w_walk" => self.w_walk = v,
            "walk_easy_min" => self.walk_easy_min = v,
            "w_walk_far" => self.w_walk_far = v,
            "pair_walk_max" => self.pair_walk_max = v,
            "prune_trips" => self.prune_trips = v,
            "commute_chunk" => self.commute_chunk = v,
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
            "home_park_f" => self.home_park_f = v,
            "car_own_usd_trip" => self.car_own_usd_trip = v,
            "car_own_usd_km" => self.car_own_usd_km = v,
            // scales every zone's cordon charge (0 turns them off)
            "cordon" => {
                if let Some(c) = &self.cordon {
                    self.cordon = Some(std::sync::Arc::new(c.iter().map(|x| x * v).collect()));
                }
            }
            "everyday_asc_rail" => self.everyday_asc_rail = v,
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
    /// The local layers' station-choice bands (T-710): `REL_EDGES`' four, or one where
    /// `rel_gamma` is 0.
    pub fn n_bands(&self) -> usize {
        if self.rel_gamma > 0.0 { REL_EDGES.len() - 1 } else { 1 }
    }
    /// The band of a trip whose straight line squared is `d2` square metres (zone centres for
    /// the commutes, the pair's mean distance for everyday trips).
    pub fn band_of_d2(&self, d2: f32) -> usize {
        if self.rel_gamma <= 0.0 {
            return 0;
        }
        REL_EDGES[1..REL_EDGES.len() - 1].iter().filter(|&&e| d2 >= e * e * 1e6).count()
    }
    /// Band `b`'s scale and offset (perceived minutes). Exp tables hold exp(-theta (r - offset))
    /// and each logsum adds the offset back. With the offset at 60 / theta a band keeps costs
    /// from 0 to 129 / theta inside f32 (`fast_exp` and the pair loop's 1e-30 floor): 190
    /// perceived minutes in the sharpest band, 387 from 12 km on; without it the sharpest would
    /// lose every pair over 100 (New York's station-to-station costs reach 700).
    pub fn band(&self, b: usize) -> (f32, f32) {
        if self.rel_gamma <= 0.0 {
            return (self.station_theta, 0.0);
        }
        let (lo, hi) = (REL_EDGES[b], REL_EDGES[b + 1]);
        let km = if hi.is_finite() { 0.5 * (lo + hi) } else { 1.5 * lo };
        let t = self.rel_t0 + self.rel_tkm * km;
        let th = self.soften(self.band_unsoftened(b), t);
        (th, 60.0 / th)
    }
    /// Band `b`'s scale before `soften`, for a layer that softens by its own trip length (the
    /// regional layer, by the car's minutes).
    pub fn band_unsoftened(&self, b: usize) -> f32 {
        if self.rel_gamma <= 0.0 {
            return self.station_theta;
        }
        let (lo, hi) = (REL_EDGES[b], REL_EDGES[b + 1]);
        let km = if hi.is_finite() { 0.5 * (lo + hi) } else { 1.5 * lo };
        self.rel_gamma / (self.rel_t0 + self.rel_tkm * km).min(self.rel_tmax)
    }
    /// Scale `th` for a trip `t` perceived minutes long under `rel_soft`: from `rel_tmax` on,
    /// times sqrt(`rel_tmax` / t) (the mode choice's damping shape, T-495). The local layers pass
    /// their band's rough rail length, the regional layer the car's perceived minutes.
    pub fn soften(&self, th: f32, t: f32) -> f32 {
        if self.rel_soft > 0.0 && self.rel_gamma > 0.0 && t > self.rel_tmax { th * (self.rel_tmax / t).sqrt() } else { th }
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
    /// The walking constant of the logit (units of `beta`): `walk_bonus_min` perceived minutes at
    /// the undamped scale.
    #[inline]
    pub fn asc_walk(&self) -> f32 {
        self.walk_bonus_min * self.beta
    }
    /// The ownership cost's part every car trip pays whatever its length, perceived minutes.
    #[inline]
    pub fn car_own_min(&self) -> f32 {
        self.car_own_usd_trip * super::api::FARE_MIN_PER_USD
    }
    /// The ownership cost's part a km of road, perceived minutes.
    #[inline]
    pub fn car_own_km_min(&self) -> f32 {
        self.car_own_usd_km * super::api::FARE_MIN_PER_USD
    }
    /// A drive of `drive_min` real minutes over `road_km` of road, perceived: its time and the
    /// ownership cost's part by distance (`mode_choice`'s lanes use the same expression).
    #[inline]
    pub fn car_drive(&self, drive_min: f32, road_km: f32) -> f32 {
        drive_min * self.w_drive + road_km * self.car_own_km_min()
    }
    /// Parking at a destination in zone `j`, perceived minutes: what every driver pays (parking,
    /// any cordon charge and the ownership cost), what a paid space costs on top, and the share
    /// who park free.
    pub fn park_parts(&self, z: &Zones, j: usize) -> (f32, f32, f32) {
        let land = (z.ncell[j].max(1) as f32 * CELL_KM2).min(z.square_km2(j));
        let d = z.jobs[j] as f32 / land;
        let paid = if d > self.park_paid_d0 { (self.park_paid_min * (d / self.park_paid_d0).ln()).min(self.park_paid_cap_min) } else { 0.0 };
        let fixed = self.park_min + self.car_own_min() + self.cordon.as_ref().and_then(|c| c.get(j)).map_or(0.0, |&c| c);
        (fixed, paid, 1.0 / (1.0 + d / self.park_free_d))
    }
    /// Parking at the home end of a car trip from a zone of `d` residents per km² of land, perceived
    /// minutes (T-854).
    #[inline]
    pub fn home_park_at(&self, d: f32) -> f32 {
        if d > self.park_paid_d0 { self.home_park_f * (self.park_paid_min * (d / self.park_paid_d0).ln()).min(self.park_paid_cap_min) } else { 0.0 }
    }
    /// The same for a home in cell `c`.
    #[inline]
    pub fn home_park_cell(&self, c: usize) -> f32 {
        self.home_dens.as_ref().map_or(0.0, |d| self.home_park_at(d[c]))
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
    /// Perceived minutes of walking the whole trip, `d_km` straight line (round water where the
    /// routes lengthen it), T-558; `NO_WALK` past `WALK_REACH_KM`.
    #[inline]
    pub fn walk_trip_cost(&self, d_km: f32) -> f32 {
        self.walk_cost().at(d_km)
    }
    pub fn walk_cost(&self) -> WalkCost {
        WalkCost { min_per_km: self.walk_detour * 1000.0 / self.walk_mps / 60.0, w: self.walk_trip_w }
    }
    /// The everyday layer's parameters: its rail and walking constants on top (T-509, T-558).
    pub fn everyday(&self) -> Params {
        Params { asc_rail: self.asc_rail + self.everyday_asc_rail, ..self.clone() }
    }
    /// With a car, the choice among rail (`util`, its utility against a driver who parks free at
    /// scale `beta`, `mode_beta` of that driver's cost), walking (`uw`, the same against that
    /// driver) and the car, for drivers parking free and those paying `paid` for a space, mixed
    /// by the share `free` (T-380, T-558): rail's share. The same expressions as `mode_choice`'s
    /// lanes.
    #[inline]
    pub fn car_choice(&self, util: f32, uw: f32, beta: f32, paid: f32, free: f32) -> f32 {
        let e = fast_exp(uw - util);
        let r0 = 1.0 / (1.0 + fast_exp(-util) + e);
        if paid < 1e-30 {
            return r0;
        }
        let r1 = 1.0 / (1.0 + fast_exp(-(util + beta * paid)) + e);
        r0 + (1.0 - free) * (r1 - r0)
    }
    /// The curve's floor and ceiling (`rail_floor`, T-854) on `r`, rail's share of all trips, with
    /// walking's share `e` x `r` (`e` as `car_choice`'s), rail costing `rail` perceived minutes and
    /// the car `car` (its mean over drivers): walking keeps its share and rail's share of the rest
    /// R becomes lo + (1 - lo - hi) R. The same expressions as `mode_choice`'s lanes.
    #[inline]
    pub fn rail_bounds(&self, r: f32, e: f32, rail: f32, car: f32) -> f32 {
        if self.rail_floor <= 0.0 {
            return r;
        }
        let k = 1.0 / (self.rail_floor_ratio - 1.0);
        let lo = self.rail_floor * ((self.rail_floor_ratio - rail / car) * k).max(0.0).min(1.0);
        let hi = self.rail_floor * ((self.rail_floor_ratio - car / rail) * k).max(0.0).min(1.0);
        lo + r * (1.0 - lo - hi - lo * e)
    }
    /// With no rail: walking's share against the car costing `road` perceived minutes (a driver
    /// who parks free; `paid` more for a space, `free` parking free), the walk `cw` (0 past
    /// `WALK_REACH_KM`).
    #[inline]
    pub fn walk_no_rail(&self, road: f32, paid: f32, free: f32, cw: f32) -> f32 {
        if cw >= NO_WALK {
            return 0.0;
        }
        let bt = self.mode_beta(road);
        let uw = bt * (road - cw) + self.asc_walk();
        let w0 = 1.0 / (1.0 + fast_exp(-uw));
        if paid < 1e-30 {
            return w0;
        }
        let w1 = 1.0 / (1.0 + fast_exp(-(uw + bt * paid)));
        w0 + (1.0 - free) * (w1 - w0)
    }
    /// Real minutes of the walk between two stations on a transfer, `d_m` metres straight line.
    pub fn xfer_min(&self, d_m: f32) -> f32 {
        d_m * self.walk_detour / self.xfer_mps / 60.0
    }
}

/// What the zone pairs' choice with no rail reads (`Gravity::no_rail_by_zone`, T-558): per zone
/// its streets' and main roads' minutes a km, its parking (`Params::park_parts`) and the home
/// end's parking.
pub struct NoRail {
    place: Vec<(f32, f32)>,
    park: Vec<(f32, f32, f32)>,
    /// parking at the home end, by home zone (T-854)
    home: Vec<f32>,
}

impl NoRail {
    pub fn new(z: &Zones, p: &Params) -> NoRail {
        let place = z.density().iter().map(|&d| p.place_min_per_km(d)).collect();
        let park = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let mut home = vec![0f32; z.n];
        for (c, &zi) in z.cell_zone.iter().enumerate() {
            home[zi as usize] = p.home_park_cell(c);
        }
        NoRail { place, park, home }
    }
    /// The share walking between zones i (home) and j at table index k (`Params::walk_no_rail`):
    /// a near pair's cell pairs' (`Gravity::set_walk_fine`), else at the zone-centre distance.
    #[inline]
    pub fn split(&self, gr: &Gravity, p: &Params, i: usize, j: usize, k: usize) -> f32 {
        if let Some(n) = gr.fine_at(i, j, k) {
            return gr.wf_s[n];
        }
        let cw = gr.walk_cost_at(i, j, k);
        if cw >= NO_WALK {
            return 0.0;
        }
        let (road, paid, free) = self.road(gr, p, i, j, k);
        p.walk_no_rail(road, paid, free, cw)
    }
    /// The drive between zone centres: perceived minutes to a driver parking free (with the home
    /// end's parking), a paid space's extra and the share parking free.
    #[inline]
    pub fn road(&self, gr: &Gravity, p: &Params, i: usize, j: usize, k: usize) -> (f32, f32, f32) {
        let d_km = gr.dist_km(k);
        let f = p.drive_factor(i, j);
        let (fixed, paid, free) = self.park[j];
        (p.car_drive(p.drive_min_f(d_km * 1000.0, f, self.place[i], self.place[j]), d_km * f) + fixed + self.home[i], paid, free)
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
    /// T-852: each zone's side in squares, 1 or 2 (a 2x2 block, `gx`, `gy` its lower-left
    /// square, in one region); empty: every zone one square.
    pub size: Vec<u8>,
}

/// Mean area of an H3 resolution 9 cell, km².
pub const CELL_KM2: f32 = 0.1053;

impl Zones {
    /// Whether the pack has blocks (T-852).
    #[inline]
    pub fn mixed(&self) -> bool {
        !self.size.is_empty()
    }
    /// Whether zone i is a 2x2 block.
    #[inline]
    pub fn big(&self, i: usize) -> bool {
        self.size.get(i).is_some_and(|&s| s == 2)
    }
    /// Zone i's side in squares.
    #[inline]
    pub fn side(&self, i: usize) -> i32 {
        1 + self.big(i) as i32
    }
    /// Zone i's centre in half squares from the grid's corner.
    #[inline]
    pub fn hx(&self, i: usize) -> i32 {
        2 * self.gx[i] + self.side(i)
    }
    #[inline]
    pub fn hy(&self, i: usize) -> i32 {
        2 * self.gy[i] + self.side(i)
    }
    /// Squared metres between the centres of zones i and j (for one-square zones exactly the
    /// squares' offset squared x `zone_m`²).
    #[inline]
    pub fn d2_m(&self, i: usize, j: usize) -> f32 {
        let (fx, fy) = ((self.hx(j) - self.hx(i)) as f32 * 0.5, (self.hy(j) - self.hy(i)) as f32 * 0.5);
        (fx * fx + fy * fy) * self.zone_m * self.zone_m
    }
    /// Zone i's whole square in km², the cap on its land.
    #[inline]
    pub fn square_km2(&self, i: usize) -> f32 {
        (self.side(i) as f32 * self.zone_m / 1000.0).powi(2)
    }
    /// Every zone with a square within `r` squares (each way) of one of zone i's, once, in the
    /// order of their lower-left squares (`by_square`'s index `sq`).
    #[inline]
    pub fn near(&self, sq: &(Vec<u32>, Vec<u32>), i: usize, r: i32, mut f: impl FnMut(usize)) {
        let (off, at) = sq;
        let (m, e) = (self.mixed() as i32, self.side(i) - 1);
        for gx in (self.gx[i] - r - m).max(0)..=(self.gx[i] + e + r).min(self.w - 1) {
            for gy in (self.gy[i] - r - m).max(0)..=(self.gy[i] + e + r).min(self.h - 1) {
                let s = (gx * self.h + gy) as usize;
                for &j in &at[off[s] as usize..off[s + 1] as usize] {
                    let j = j as usize;
                    if m == 1 {
                        let ej = self.side(j) - 1;
                        let gap = |a: i32, b: i32| (b - a - e).max(a - b - ej).max(0);
                        if gap(self.gx[i], gx) > r || gap(self.gy[i], gy) > r {
                            continue;
                        }
                    }
                    f(j);
                }
            }
        }
    }
    /// The zone holding square (gx, gy) (`by_square`'s index `sq`): the first one at that square,
    /// else a block covering it.
    pub fn zone_at(&self, sq: &(Vec<u32>, Vec<u32>), gx: i32, gy: i32) -> Option<usize> {
        let (off, at) = sq;
        let first = |gx: i32, gy: i32, block: bool| {
            if gx < 0 || gy < 0 || gx >= self.w || gy >= self.h {
                return None;
            }
            let s = (gx * self.h + gy) as usize;
            at[off[s] as usize..off[s + 1] as usize].iter().map(|&j| j as usize).find(|&j| !block || self.big(j))
        };
        first(gx, gy, false).or_else(|| if self.mixed() { first(gx - 1, gy, true).or_else(|| first(gx, gy - 1, true)).or_else(|| first(gx - 1, gy - 1, true)) } else { None })
    }
    /// Homes and jobs per km² of land, per zone (the drive's local street speed).
    pub fn density(&self) -> Vec<f32> {
        (0..self.n).map(|i| ((self.pop[i] + self.jobs[i]) / (self.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32).collect()
    }
    /// Per cell, its zone's residents (`pop`, people per cell) per km² of land, measured as
    /// `Params::park_parts` measures jobs (`Params::home_dens`, T-854).
    pub fn home_density(&self, pop: &[f32]) -> Vec<f32> {
        let mut p = vec![0f64; self.n];
        for (c, &zi) in self.cell_zone.iter().enumerate() {
            p[zi as usize] += pop.get(c).copied().unwrap_or(0.0) as f64;
        }
        let d: Vec<f32> = (0..self.n).map(|i| (p[i] / (self.ncell[i].max(1) as f32 * CELL_KM2).min(self.square_km2(i)) as f64) as f32).collect();
        self.cell_zone.iter().map(|&zi| d[zi as usize]).collect()
    }
    /// The zones of each grid square, CSR by square index gx * h + gy (usually one; two or more
    /// where a region border cuts the square); a block under its lower-left square only
    /// (`zone_at` finds it from the others).
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
        return Zones { zone_m, n: 0, cell_zone: vec![], gx: vec![], gy: vec![], region: vec![], w: 0, h: 0, x0: 0., y0: 0., pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
    }
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / zone_m) as i32 + 1;
    let h = ((maxy - miny) / zone_m) as i32 + 1;
    let nreg = city.region.iter().copied().max().map_or(1, |r| r as usize + 1);
    let mut grid = vec![NONE; (w * h) as usize * nreg];
    let mut z = Zones { zone_m, n: 0, cell_zone: vec![0; city.len()], gx: vec![], gy: vec![], region: vec![], w, h, x0: minx, y0: miny, pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
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

/// T-852: one-square zones `z` with each 2x2 block of squares (aligned to the grid, even `gx`
/// and `gy` at its lower-left) as one zone where its zones of one region are at least two and
/// each makes under `under` trip ends a weekday (`ends`, per zone). Zone ids in order of each
/// zone's first cell, as `build_zones`; the grid grows to an even size so every block is in it.
pub fn coarsen(z: &Zones, ends: &[f64], under: f64) -> Zones {
    assert!(!z.mixed() && ends.len() == z.n);
    let key = |i: usize| ((z.gx[i] / 2) as u64) << 40 | ((z.gy[i] / 2) as u64) << 8 | z.region[i] as u64;
    let mut blocks: HashMap<u64, (u32, bool)> = HashMap::new();
    for i in 0..z.n {
        let e = blocks.entry(key(i)).or_insert((0, true));
        e.0 += 1;
        e.1 &= ends[i] < under;
    }
    let coarse = |i: usize| blocks[&key(i)].0 >= 2 && blocks[&key(i)].1;
    if !(0..z.n).any(coarse) {
        return z.clone();
    }
    let mut out = Zones { zone_m: z.zone_m, n: 0, cell_zone: vec![0; z.cell_zone.len()], gx: vec![], gy: vec![], region: vec![], w: z.w + z.w % 2, h: z.h + z.h % 2, x0: z.x0, y0: z.y0, pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
    let mut new_of = vec![NONE; z.n];
    let mut block_id: HashMap<u64, u32> = HashMap::new();
    for c in 0..z.cell_zone.len() {
        let i = z.cell_zone[c] as usize;
        if new_of[i] == NONE {
            let big = coarse(i);
            let id = if big { block_id.get(&key(i)).copied() } else { None };
            new_of[i] = id.unwrap_or_else(|| {
                out.gx.push(if big { z.gx[i] & !1 } else { z.gx[i] });
                out.gy.push(if big { z.gy[i] & !1 } else { z.gy[i] });
                out.region.push(z.region[i]);
                out.size.push(1 + big as u8);
                out.n += 1;
                if big {
                    block_id.insert(key(i), out.n as u32 - 1);
                }
                out.n as u32 - 1
            });
        }
        out.cell_zone[c] = new_of[i];
    }
    out.pop = vec![0.0; out.n];
    out.jobs = vec![0.0; out.n];
    out.ncell = vec![0; out.n];
    for i in 0..z.n {
        if let Some(&k) = new_of.get(i).filter(|&&k| k != NONE) {
            out.pop[k as usize] += z.pop[i];
            out.jobs[k as usize] += z.jobs[i];
            out.ncell[k as usize] += z.ncell[i];
        }
    }
    out
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
    /// The squares' side in km, for the zone-centre distance of a table index (`dist_km`), and
    /// whether `set_walk` has set the walk's cost (`walk_c`) and (T-275) the routes whose walk
    /// factors lengthen the walk between nearby zones around water (`walk_cost_at`).
    pub zk: f32,
    pub walk_set: bool,
    pub wroutes: Option<std::sync::Arc<super::routes::Routes>>,
    pub wcost: WalkCost,
    /// Walks between cells (T-587, `set_walk_fine`), per home zone CSR over the zones within
    /// `WALK_FINE_R` squares, ascending: zone, the walk's cost for the choice with rail, and the
    /// share walking with no rail. Empty: zone centres only.
    pub wf_off: Vec<u32>,
    pub wf_j: Vec<u32>,
    pub wf_c: Vec<f32>,
    pub wf_s: Vec<f32>,
    pub th: i32,
    pub offx: i32,
    pub offy: i32,
    /// A block's key shift (T-852, `table_layout`); 0 without blocks.
    pub kb: i32,
    pub iters: usize,
    pub max_err: f64,
    pub mean_km: f64,
    pub total: f64,
}

impl Gravity {
    /// Index offset to add to `key(J)` for origin zone I.
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i] - 2 * self.kb * z.big(i) as i32
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j] + self.kb * z.big(j) as i32
    }
    /// The region factor between zones i and j: 1 in one region, `border` across.
    #[inline]
    pub fn k(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if z.region[i] == z.region[j] { 1.0 } else { self.border }
    }
    pub fn t(&self, z: &Zones, i: usize, j: usize) -> f64 {
        self.row[i] * self.col[j] * (self.ftab[(self.base(z, i) + self.key(z, j)) as usize] * self.k(z, i, j)) as f64
    }

    /// The zone-centre distance (km) of table index `k`, as `decay_table` and `decay_tables` give
    /// it (within a zone 0.52 of its side).
    #[inline]
    pub fn dist_km(&self, k: usize) -> f32 {
        let (r, y) = (k as i32 / self.th - self.offx, k as i32 % self.th);
        let h = self.kb / 2;
        let (dx, dy, own) = if self.kb == 0 || y < 2 * h - 1 {
            (r as f32, (y - self.offy) as f32, 1.0)
        } else if y < 4 * h - 1 {
            (r as f32 + 0.5, (y - 3 * h + 1) as f32 + 0.5, 1.0)
        } else {
            ((r + 1) as f32, (y - 5 * h + 2) as f32, 2.0)
        };
        if dx == 0.0 && dy == 0.0 { 0.52 * self.zk * own } else { self.zk * (dx * dx + dy * dy).sqrt() }
    }
    /// The cost of walking the whole way between zone centres at table index `k`
    /// (`Params::walk_trip_cost`; `NO_WALK` before `set_walk`).
    #[inline]
    pub fn walk_c(&self, k: usize) -> f32 {
        if self.walk_set { self.wcost.at(self.dist_km(k)) } else { NO_WALK }
    }

    /// Set the walk's cost between zone centres (`walk_c`); the walks between cells go
    /// (`set_walk_fine` after).
    pub fn set_walk(&mut self, z: &Zones, p: &Params) {
        self.wcost = p.walk_cost();
        self.zk = z.zone_m / 1000.0;
        self.walk_set = true;
        self.wroutes = p.routes.clone().filter(|r| r.walk_window() > 0);
        self.wf_off.clear();
        self.wf_j.clear();
        self.wf_c.clear();
        self.wf_s.clear();
    }

    /// `wf_*` (T-587): each near zone pair's commuters as the pairs of its zones' points
    /// (`walk_points`), each walking `cell_walk_m` times the zones' walk factor round water. The
    /// pair's share walking with no rail is its cell pairs' mean (`Params::walk_no_rail`); its cost for the choice with rail is the one walk whose share
    /// against a driver parking free is that mean.
    pub fn set_walk_fine(&mut self, city: &City, z: &Zones, p: &Params, nr: &NoRail) {
        let ns = p.walk_samples;
        if ns == 0 || !self.walk_set || city.len() != z.cell_zone.len() {
            return;
        }
        let (home, work) = walk_points(city, z, ns);
        let sq = z.by_square();
        let w2 = 1.0 / (ns * ns) as f32;
        let mut row: Vec<(u32, f32, f32)> = vec![];
        let mut c = vec![0f32; ns * ns];
        for i in 0..z.n {
            self.wf_off.push(self.wf_j.len() as u32);
            let hi = &home[i * ns..(i + 1) * ns];
            if hi[0] == NONE {
                continue;
            }
            row.clear();
            z.near(&sq, i, WALK_FINE_R, |ju| {
                let wj = &work[ju * ns..(ju + 1) * ns];
                if wj[0] == NONE {
                    return;
                }
                let k = (self.base(z, i) + self.key(z, ju)) as usize;
                for (n, &a) in hi.iter().enumerate() {
                    for (m, &b) in wj.iter().enumerate() {
                        let f = self.walk_factor_cells(z, a as usize, b as usize) * 0.001;
                        c[n * ns + m] = self.wcost.at(cell_walk_m(city, a, b) * f);
                    }
                }
                let (road, paid, free) = nr.road(self, p, i, ju, k);
                let bt = p.mode_beta(road);
                let (mut m0, mut sw) = (0f32, 0f32);
                for &x in &c {
                    sw += p.walk_no_rail(road, paid, free, x);
                    if x < NO_WALK {
                        m0 += 1.0 / (1.0 + fast_exp(-(bt * (road - x) + p.asc_walk())));
                    }
                }
                let cw = if m0 > 0.0 {
                    let m0 = (m0 * w2).clamp(1e-7, 1.0 - 1e-6);
                    road - ((m0 / (1.0 - m0)).ln() - p.asc_walk()) / bt
                } else {
                    NO_WALK
                };
                row.push((ju as u32, cw, sw * w2));
            });
            row.sort_unstable_by_key(|x| x.0);
            self.wf_j.extend(row.iter().map(|x| x.0));
            self.wf_c.extend(row.iter().map(|x| x.1));
            self.wf_s.extend(row.iter().map(|x| x.2));
        }
        self.wf_off.push(self.wf_j.len() as u32);
    }

    /// The entry of `wf_*` for zones i and j at table index `k`, if they are a near pair.
    #[inline]
    pub fn fine_at(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        if self.wf_off.is_empty() {
            return None;
        }
        let (dx, dy) = self.square_offset(k);
        // a block's pairs reach a square further (`Zones::near`)
        let r = WALK_FINE_R + (self.kb != 0) as i32;
        if dx.abs() > r || dy.abs() > r {
            return None;
        }
        let a = self.wf_off[i] as usize;
        self.wf_j[a..self.wf_off[i + 1] as usize].binary_search(&(j as u32)).ok().map(|n| a + n)
    }

    /// The offset in squares of table index `k`'s pair (with blocks, T-852, within a square of
    /// their centres' offset).
    #[inline]
    fn square_offset(&self, k: usize) -> (i32, i32) {
        let (r, y) = (k as i32 / self.th - self.offx, k as i32 % self.th);
        let h = self.kb / 2;
        if self.kb == 0 || y < 2 * h - 1 {
            (r, y - self.offy)
        } else if y < 4 * h - 1 {
            (r, y - 3 * h + 1)
        } else {
            (r + 1, y - 5 * h + 2)
        }
    }

    /// The walk's factor round water between zones i and j (T-275), at least 1.
    #[inline]
    pub fn walk_factor(&self, i: usize, j: usize) -> f32 {
        self.wroutes.as_ref().map_or(1.0, |r| r.walk_factor(i, j)).max(1.0)
    }

    /// The same between cells a and b, from the parts of their zones they are in (T-700).
    #[inline]
    pub fn walk_factor_cells(&self, z: &Zones, a: usize, b: usize) -> f32 {
        self.wroutes.as_ref().map_or(1.0, |r| r.walk_factor_cells(z, a, b)).max(1.0)
    }

    /// The cost of walking the whole way between zones i and j at table index `k` (`base(i) +
    /// key(j)`): between cells for near pairs (`wf_*`, T-587), else `walk_c`, or with the pack's
    /// routes (T-275) the walk around water between them, the zone-centre distance times their
    /// walk factor.
    #[inline]
    pub fn walk_cost_at(&self, i: usize, j: usize, k: usize) -> f32 {
        if let Some(n) = self.fine_at(i, j, k) {
            return self.wf_c[n];
        }
        if let Some(r) = &self.wroutes {
            let f = r.walk_factor(i, j);
            if f > 1.0 {
                return self.wcost.at(self.dist_km(k) * f);
            }
        }
        self.walk_c(k)
    }

    /// The same for any zone pair (`NO_WALK` before `set_walk`).
    pub fn walk_cost_pair(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if !self.walk_set {
            return NO_WALK;
        }
        self.walk_cost_at(i, j, (self.base(z, i) + self.key(z, j)) as usize)
    }

    /// T-558: all-day trips with no rail at all that walk, by home zone and by work zone: over the
    /// zone pairs within `WALK_REACH_KM`, T_IJ times `NoRail::split`.
    pub fn no_rail_by_zone(&self, z: &Zones, p: &Params, nr: &NoRail) -> [Vec<f64>; 2] {
        let (mut wh, mut ww) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        if !self.walk_set {
            return [wh, ww];
        }
        // the walk factor round water is at most this (`routes`' byte tables), so past this many
        // squares no walk is offered
        let r = (WALK_REACH_KM * 1000.0 / z.zone_m).ceil() as i32 + 1;
        let sq = z.by_square();
        for i in 0..z.n {
            if self.row[i] == 0.0 {
                continue;
            }
            let base = self.base(z, i);
            z.near(&sq, i, r, |j| {
                let k = (base + self.key(z, j)) as usize;
                if self.walk_c(k) >= NO_WALK {
                    return;
                }
                let w = nr.split(self, p, i, j, k);
                if w <= 0.0 {
                    return;
                }
                let t = self.row[i] * self.col[j] * (self.ftab[k] * self.k(z, i, j)) as f64;
                wh[i] += t * w as f64;
                ww[j] += t * w as f64;
            });
        }
        [wh, ww]
    }
}

/// Decay f and distance d (km) by zone grid offset, index (dx + w - 1) * (2h - 1) + (dy + h - 1).
/// Zone-centre distance; within a zone 0.52 x the zone side (mean distance between two random
/// points of a square). The one definition of the decay: the pipeline's solve and the pack
/// loader both use it, so shipped factors and rebuilt trips agree exactly.
/// Each zone's homes and jobs as `ns` cells (`Params::walk_samples`, T-587): the cells where the
/// cumulative commute weight, in pack order, crosses (s + 0.5) / ns of the zone's; NONE for a
/// zone with no weight. Zone by zone, `ns` each.
pub fn walk_points(city: &City, z: &Zones, ns: usize) -> (Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; z.n + 1];
    for &zi in &z.cell_zone {
        off[zi as usize + 1] += 1;
    }
    for i in 0..z.n {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut cells = vec![0u32; z.cell_zone.len()];
    for (c, &zi) in z.cell_zone.iter().enumerate() {
        cells[fill[zi as usize] as usize] = c as u32;
        fill[zi as usize] += 1;
    }
    let points = |w: &[f32]| {
        let mut out = vec![NONE; z.n * ns];
        for i in 0..z.n {
            let cs = &cells[off[i] as usize..off[i + 1] as usize];
            let tot: f64 = cs.iter().map(|&c| w[c as usize] as f64).sum();
            if tot <= 0.0 {
                continue;
            }
            let (mut acc, mut s) = (0.0f64, 0usize);
            for &c in cs {
                acc += w[c as usize] as f64;
                while s < ns && acc >= (s as f64 + 0.5) / ns as f64 * tot {
                    out[i * ns + s] = c;
                    s += 1;
                }
            }
            // rounding can leave the last on no cell: the last cell with weight
            let last = *cs.iter().rev().find(|&&c| w[c as usize] > 0.0).unwrap();
            out[i * ns + s..(i + 1) * ns].fill(last);
        }
        out
    };
    (points(&city.pop), points(&city.jobs))
}

/// Straight-line metres between cells a and b; a cell to itself 0.52 of its square's side, the
/// mean distance within a square.
#[inline]
pub fn cell_walk_m(city: &City, a: u32, b: u32) -> f32 {
    let (a, b) = (a as usize, b as usize);
    if a == b {
        0.52 * (res9_cells(city, a) as f32 * CELL_KM2).sqrt() * 1000.0
    } else {
        (city.x[a] - city.x[b]).hypot(city.y[a] - city.y[b])
    }
}

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

/// The pair tables' layout for zones `z` (`Gravity`'s `th`, `offx`, `offy`, `kb`), and its
/// rows. One-square zones: `decay_table`'s. With blocks (T-852) each row holds three runs,
/// one per kind of pair: two one-square zones by their squares' offset, a square and a block
/// by the half-square offset of their centres, two blocks by their blocks' offset; a block's key
/// is `kb` on and its base `2 kb` off, which lands each pair in its run (`Gravity::base`).
pub fn table_layout(z: &Zones) -> (i32, i32, i32, i32, i32) {
    let (w, h) = (z.w.max(1), z.h.max(1));
    if z.mixed() {
        (6 * h - 1, w - 1, h - 1, 2 * h, 2 * w - 1)
    } else {
        (2 * h - 1, w - 1, h - 1, 0, 2 * w - 1)
    }
}

/// `Gravity::base` and `key` alone, for other tables of `table_layout`'s.
#[derive(Clone, Copy, Debug)]
pub struct PairKey {
    th: i32,
    offx: i32,
    offy: i32,
    kb: i32,
}

impl PairKey {
    pub fn of(z: &Zones) -> PairKey {
        let (th, offx, offy, kb, _) = table_layout(z);
        PairKey { th, offx, offy, kb }
    }
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i] - 2 * self.kb * z.big(i) as i32
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j] + self.kb * z.big(j) as i32
    }
}

/// Each entry of a mixed layout (`table_layout`): `f` gets its index, the centres' offset in
/// squares and its run (0 squares, 1 a square and a block, 2 blocks).
fn mixed_entries(z: &Zones, mut f: impl FnMut(usize, f32, f32, u8)) {
    let (th, offx, offy, _, tw) = table_layout(z);
    let h = z.h.max(1);
    for r in 0..tw {
        let dx = r - offx;
        for y in 0..th {
            let k = (r * th + y) as usize;
            if y < 2 * h - 1 {
                f(k, dx as f32, (y - offy) as f32, 0);
            } else if y < 4 * h - 1 {
                f(k, dx as f32 + 0.5, (y - 3 * h + 1) as f32 + 0.5, 1);
            } else {
                f(k, (dx + 1) as f32, (y - 5 * h + 2) as f32, 2);
            }
        }
    }
}

/// A pair table over zones `z` of `decay(d_km)`: by `decay_table`'s rule for one-square zones;
/// with blocks (T-852), a pair with a block takes the mean of the decay between its squares (each
/// square's centre, one square's own 0.52 of a side), so a block draws as its four squares would
/// with its homes and jobs spread evenly over them.
pub fn pair_decay(z: &Zones, decay: impl Fn(f32) -> f32) -> Vec<f32> {
    let zk = z.zone_m / 1000.0;
    let (th, _, _, _, tw) = table_layout(z);
    // by absolute square offset, one more each way than the grid for a block's far squares
    let (qw, qh) = (z.w.max(1) as usize + 2, z.h.max(1) as usize + 2);
    let mut q = vec![0f32; qw * qh];
    for ax in 0..qw {
        for ay in 0..qh {
            let d = if ax == 0 && ay == 0 { 0.52 * zk } else { zk * ((ax * ax + ay * ay) as f32).sqrt() };
            q[ax * qh + ay] = decay(d);
        }
    }
    let fq = |dx: i32, dy: i32| q[(dx.unsigned_abs() as usize).min(qw - 1) * qh + (dy.unsigned_abs() as usize).min(qh - 1)];
    let mut out = vec![0f32; (tw * th) as usize];
    mixed_entries(z, |k, dx, dy, kind| {
        out[k] = match kind {
            0 => fq(dx as i32, dy as i32),
            1 => {
                let (x, y) = ((dx - 0.5) as i32, (dy - 0.5) as i32);
                0.25 * (fq(x, y) + fq(x + 1, y) + fq(x, y + 1) + fq(x + 1, y + 1))
            }
            _ => {
                let (x, y) = (dx as i32, dy as i32);
                let wt = [0.25f32, 0.5, 0.25];
                let mut s = 0.0;
                for (a, wa) in wt.iter().enumerate() {
                    for (b, wb) in wt.iter().enumerate() {
                        s += wa * wb * fq(x + a as i32 - 1, y + b as i32 - 1);
                    }
                }
                s
            }
        };
    });
    out
}

/// `decay_tables`' decay alone.
pub fn decay_ftab(z: &Zones, decay_km: f32, decay_pow: f32) -> Vec<f32> {
    if !z.mixed() {
        return decay_table(z.w, z.h, z.zone_m, decay_km, decay_pow).0;
    }
    pair_decay(z, |d| (-d / decay_km).exp() * if decay_pow != 0.0 { d.powf(-decay_pow) } else { 1.0 })
}

/// The pair tables of `decay_table` for zones `z`: decay and centre distance (km). With blocks
/// (T-852) the decay is `pair_decay`'s and a block's own pair is 0.52 of its side apart.
pub fn decay_tables(z: &Zones, decay_km: f32, decay_pow: f32) -> (Vec<f32>, Vec<f32>) {
    if !z.mixed() {
        return decay_table(z.w, z.h, z.zone_m, decay_km, decay_pow);
    }
    let ftab = pair_decay(z, |d| (-d / decay_km).exp() * if decay_pow != 0.0 { d.powf(-decay_pow) } else { 1.0 });
    let zk = z.zone_m / 1000.0;
    let mut dtab = vec![0f32; ftab.len()];
    mixed_entries(z, |k, dx, dy, kind| {
        dtab[k] = if dx == 0.0 && dy == 0.0 { 0.52 * zk * if kind == 2 { 2.0 } else { 1.0 } } else { zk * (dx * dx + dy * dy).sqrt() };
    });
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
        size: if pg.size.len() == n && pg.size.iter().any(|&s| s == 2) { pg.size.clone() } else { vec![] },
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
    let ftab = decay_ftab(z, pg.decay_km, pg.decay_pow);
    let (th, offx, offy, kb, _) = table_layout(z);
    Gravity {
        row: pg.row.clone(),
        col: pg.col.clone(),
        border: pg.border_factor,
        ftab,
        zk: z.zone_m / 1000.0,
        walk_set: false,
        wroutes: None,
        wcost: WalkCost::default(),
        wf_off: vec![],
        wf_j: vec![],
        wf_c: vec![], wf_s: vec![],
        th,
        offx,
        offy,
        kb,
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
        size: z.size.clone(),
    }
}

/// Doubly constrained gravity by Furness balancing. Workers = population scaled to total jobs.
/// The decay depends on zone-centre distance only, so f is a small table by grid offset; it is
/// multiplied by `p.border_factor` between zones of different regions (T-162).
pub fn gravity(z: &Zones, p: &Params, max_iters: usize, tol: f64) -> Gravity {
    let (th, offx, offy, kb, _) = table_layout(z);
    let (ftab, dtab) = decay_tables(z, p.decay_km, p.decay_pow);
    let tp: f64 = z.pop.iter().sum();
    let tj: f64 = z.jobs.iter().sum();
    let o: Vec<f64> = z.pop.iter().map(|&v| v * tj / tp).collect();
    let rows: Vec<usize> = (0..z.n).filter(|&i| o[i] > 0.0).collect();
    let cols: Vec<usize> = (0..z.n).filter(|&j| z.jobs[j] > 0.0).collect();
    let ckey: Vec<i32> = cols.iter().map(|&j| z.gx[j] * th + z.gy[j] + kb * z.big(j) as i32).collect();
    let creg: Vec<u8> = cols.iter().map(|&j| z.region[j]).collect();
    // the factor by "regions differ" (index 0 same, 1 different)
    let kf = [1.0f64, p.border_factor as f64];
    let mut g = Gravity { row: vec![0.0; z.n], col: z.jobs.clone(), border: p.border_factor, ftab, zk: z.zone_m / 1000.0, walk_set: false, wroutes: None, wcost: WalkCost::default(), wf_off: vec![], wf_j: vec![], wf_c: vec![], wf_s: vec![], th, offx, offy, kb, iters: 0, max_err: 0.0, mean_km: 0.0, total: 0.0 };
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
    /// Parking at the home end of its homes' car trips (`Params::home_park_cell`, home-weighted;
    /// T-854), perceived minutes. Its ride entries include it.
    pub home_park: Vec<f32>,
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
    z.d2_m(i, j) >= (p.ride_min_km * 1000.0).powi(2)
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
    // parking as at a destination, its mean over drivers (the ownership cost is the drive's, on
    // its own road, whether the rider parks or is dropped off; none at the far end: T-621)
    let station_end = |z: Option<usize>| match end {
        End::Far => p.d_drop,
        End::Home => z.map_or(p.park_min, |z| p.park_mean(zones, z) - p.car_own_min()).min(p.d_drop),
    };
    let own = |road_km: f32| match end {
        End::Far => 0.0,
        End::Home => p.car_own_min() + road_km * p.car_own_km_min(),
    };
    let sq = zones.by_square();
    let st_zone: Vec<Option<usize>> = (0..net.n_stations())
        .map(|s| {
            let gx = ((net.st_x[s] - zones.x0) / zones.zone_m).floor() as i32;
            let gy = ((net.st_y[s] - zones.y0) / zones.zone_m).floor() as i32;
            zones.zone_at(&sq, gx, gy)
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
                let cost = p.p_ride + p.w_drive * t + own(d * 0.001 * f) + station_end(j);
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
        home_park: vec![0.0; ns],
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
            sz.home_park[s] += home_w[c] * p.home_park_cell(c);
        }
        sz.home_park[s] = if sz.pop[s] > 0.0 { sz.home_park[s] / sz.pop[s] } else { 0.0 };
        let hp = sz.home_park[s];
        sz.cx[s] = (x / wsum) as f32;
        sz.cy[s] = (y / wsum) as f32;
        if !walks {
            let g = sub_group[s - n_walk] as usize;
            sz.zone[s] = g_rep[g].0;
            sz.acc_mid.push(sz.acc_st.len() as u32);
            for &(st, w, d, t) in place_ride(zones.n + g) {
                sz.acc_st.push(st);
                sz.acc_w.push(w + hp);
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
                sz.acc_w.push(w + hp);
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
    /// leave riders on the platform (T-188, `left_behind`), plus `rn_span`. Added to the
    /// boarding's cost.
    pub rn_left: Vec<f32>,
    /// Perceived minutes added to boarding where only a share `phi` of the period's riders have a
    /// train due within their tolerance (T-842, `Network::span`): -ln(phi) / `Params::beta`, the
    /// logit's own form for an option open to that share. 0 with service all period.
    pub rn_span: Vec<f32>,
    /// Each route node's platform less its station's position, metres (`Network::stop_x`): a change
    /// of trains walks from the platform it arrives at to the one it leaves from (T-717).
    pub rn_dx: Vec<f32>,
    pub rn_dy: Vec<f32>,
    /// Stations with a platform away from their position, where a change can walk.
    pub st_spread: Vec<bool>,
    /// Incoming edges, CSR, for destination-based strategy search.
    pub rev_off: Vec<u32>,
    pub rev_from: Vec<u32>,
    pub rev_edge: Vec<u32>,
    pub adj_off: Vec<u32>,
    pub adj_to: Vec<u32>,
    pub adj_cost: Vec<f32>,
    /// For ride edges, the route node the segment starts at; NONE otherwise.
    pub adj_seg: Vec<u32>,
    /// The fare curve's departure from `rn_fare`'s straight line (T-883), and each route node's
    /// fare km to the next stop, which it needs; both empty while the curve is that line.
    pub fare_curve: FareCurve,
    pub rn_km: Vec<f32>,
}

/// The player's fare curve (T-883): knots of trip km and perceived minutes, joined by straight
/// stretches and carried on past each end at that end's slope, never below 0. `base` and `rate` are
/// what a trip is already charged (the base fare once and the rate on each ride edge), so a trip
/// costs `extra(km)` more than that. Route choice sees only the straight line; the curve enters
/// each station pair's cost on the expected km of its strategy (`DestSearch::run`).
#[derive(Clone, Default, PartialEq, Debug)]
pub struct FareCurve {
    pub km: Vec<f32>,
    pub min: Vec<f32>,
    pub base: f32,
    pub rate: f32,
}

impl FareCurve {
    pub fn is_line(&self) -> bool {
        self.km.len() < 2
    }
    /// The curve's fare for a trip of `km`, perceived minutes.
    pub fn at(&self, km: f32) -> f32 {
        let (k, m, n) = (&self.km, &self.min, self.km.len());
        if n < 2 {
            return self.base + self.rate * km;
        }
        let i = k.partition_point(|&x| x <= km).clamp(1, n - 1);
        (m[i - 1] + (m[i] - m[i - 1]) * (km - k[i - 1]) / (k[i] - k[i - 1])).max(0.0)
    }
    pub fn extra(&self, km: f32) -> f32 {
        if self.is_line() { 0.0 } else { self.at(km) - self.base - self.rate * km }
    }
}

/// A wait is plain minutes, half the headway (Anita, T-585).
pub fn board_cost(headway_min: f32, p: &Params) -> f32 {
    headway_min / 2.0 + p.transfer_min
}

pub fn build_graph(net: &Network, p: &Params) -> Graph {
    let s = net.n_stations();
    let mut rn_station = vec![];
    let mut rn_line = vec![];
    let mut rn_ride = vec![];
    let mut rn_next = vec![];
    let (mut rn_dx, mut rn_dy) = (vec![], vec![]);
    let mut st_spread = vec![false; s];
    let platforms = net.stop_x.len() == net.line_stops.len() && net.stop_y.len() == net.line_stops.len();
    // route nodes where a circle's train rides on round to the first stop (T-798): nobody boards
    // there, since the first stop's own route node has the same trains
    let mut rn_wrap: Vec<bool> = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        let wrap = net.line_wrap.get(l).copied().unwrap_or(false);
        for dir in 0..2 {
            let base = s + rn_station.len();
            let len = b - a;
            for i in 0..len {
                let k = if dir == 0 { a + i } else { b - 1 - i };
                let st = net.line_stops[k] as usize;
                rn_station.push(st as u32);
                rn_line.push(l as u32);
                let (dx, dy) = if platforms { (net.stop_x[k] - net.st_x[st], net.stop_y[k] - net.st_y[st]) } else { (0.0, 0.0) };
                // under a metre is the merge's rounding, not a walk
                let (dx, dy) = if dx.hypot(dy) < 1.0 { (0.0, 0.0) } else { (dx, dy) };
                st_spread[st] |= dx != 0.0 || dy != 0.0;
                rn_dx.push(dx);
                rn_dy.push(dy);
                let ride = if i + 1 < len {
                    if dir == 0 {
                        net.hop_min[k]
                    } else if net.hop_back.is_empty() {
                        net.hop_min[k - 1]
                    } else {
                        net.hop_back[k]
                    }
                } else if wrap {
                    if dir == 0 || net.hop_back.is_empty() { net.hop_min[b - 1] } else { net.hop_back[a] }
                } else {
                    0.0
                };
                rn_ride.push(ride);
                rn_next.push(if i + 1 < len { (base + i + 1) as u32 } else if wrap { base as u32 } else { NONE });
                rn_wrap.push(wrap && i + 1 == len);
            }
        }
    }
    let n = s + rn_station.len();
    let mut edges: Vec<(u32, u32, f32, u32)> = vec![];
    for (r, &st) in rn_station.iter().enumerate() {
        let node = (s + r) as u32;
        let l = rn_line[r] as usize;
        if rn_next[r] != NONE {
            if !rn_wrap[r] {
                edges.push((st, node, board_cost(net.line_headway_min[l][p.period], p), NONE));
            }
            edges.push((node, rn_next[r], rn_ride[r], node));
        }
        edges.push((node, st, 0.0, NONE));
    }
    // walking transfers: from each route node (having ridden) to the stations nearby, and to every
    // other station of its complex however far (T-161), from its platform to the other station's
    // position
    let cut2 = p.xfer_walk_m * p.xfer_walk_m;
    let d2 = |a: usize, b: usize| (net.st_x[b] - net.st_x[a]).powi(2) + (net.st_y[b] - net.st_y[a]).powi(2);
    let mut near: Vec<Vec<u32>> = vec![vec![]; s];
    if p.xfer_walk_m > 0.0 && s > 1 {
        let grid = StationGrid::new(net, p.xfer_walk_m, p.xfer_walk_m);
        for a in 0..s {
            grid.near(net.st_x[a], net.st_y[a], |b| {
                if b as usize != a && d2(a, b as usize) <= cut2 {
                    near[a].push(b);
                }
            });
        }
    }
    if net.st_group.len() == s {
        for members in complex_members(&net.st_group) {
            for &a in &members {
                for &b in &members {
                    // within the walk limit the edge is there already
                    if a != b && (p.xfer_walk_m <= 0.0 || d2(a as usize, b as usize) > cut2) {
                        near[a as usize].push(b);
                    }
                }
            }
        }
    }
    for (r, &st) in rn_station.iter().enumerate() {
        let (x, y) = (net.st_x[st as usize] + rn_dx[r], net.st_y[st as usize] + rn_dy[r]);
        for &b in &near[st as usize] {
            let d = ((net.st_x[b as usize] - x).powi(2) + (net.st_y[b as usize] - y).powi(2)).sqrt();
            edges.push(((s + r) as u32, b, p.xfer_min(d) * p.w_xfer, NONE));
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
    let nr = rn_line.len();
    let phi = |r: usize| if net.span.len() == super::network::PERIODS * nr { net.span[p.period * nr + r] } else { 1.0 };
    // a share that rounds to 0 is no service at that stop (T-842)
    let rn_tph = rn_line.iter().enumerate().map(|(r, &l)| {
        let h = net.line_headway_min[l as usize][p.period];
        if h > 0.0 && h.is_finite() && phi(r) > 0.0 { 60.0 / h } else { 0.0 }
    }).collect();
    let rn_span: Vec<f32> = (0..nr).map(|r| { let f = phi(r); if f > 0.0 && f < 1.0 { -f.ln() / p.beta } else { 0.0 } }).collect();
    Graph {
        rn_left: rn_span.clone(),
        rn_span,
        rn_dx,
        rn_dy,
        st_spread,
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
        fare_curve: FareCurve::default(),
        rn_km: vec![],
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
    /// Per searched destination, its change nodes (`ChangeNodes`): node ids from `n` on.
    pub change: Vec<ChangeNodes>,
}

/// A station as reached off one of the lines its own strategy boards (T-575): a rider never changes
/// onto the line they left, in either direction, so arriving on that line they choose among the
/// station's other lines. Change node k is node `n + k`; a route node of the line that leaves the
/// train there, or walks there, has it as its next node. Each holds its station and a bit per
/// boarding edge of the station for the departures it accepts, `bits[off[k]..off[k + 1]]` (bit i
/// of word w is the station's edge 64 w + i).
#[derive(Clone, Default)]
pub struct ChangeNodes {
    pub st: Box<[u32]>,
    pub off: Box<[u32]>,
    pub bits: Box<[u64]>,
}

impl ChangeNodes {
    pub fn len(&self) -> usize {
        self.st.len()
    }
    /// Departures an hour change node `k` accepts at route node `r`.
    pub fn tph(&self, g: &Graph, k: usize, r: usize) -> f32 {
        let st = self.st[k] as usize;
        let e0 = g.adj_off[st] as usize;
        match (e0..g.adj_off[st + 1] as usize).position(|e| g.adj_to[e] as usize == g.n_st + r) {
            Some(i) if self.bits[self.off[k] as usize + i / 64] >> (i % 64) & 1 == 1 => g.rn_tph[r],
            _ => 0.0,
        }
    }
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
        Skims { s: g.n_st, n: g.n, r: vec![f32::INFINITY; g.n_st * g.n_st], slot: vec![NONE; g.n_st], next: vec![], board: vec![], order: vec![], change: vec![] }
    }
    /// The most change nodes of any searched destination.
    pub fn most_change_nodes(&self) -> usize {
        self.change.iter().map(|c| c.len()).max().unwrap_or(0)
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
    let todo: Vec<usize> = (0..g.n_st).filter(|&d| want.get(d).copied().unwrap_or(false) && sk.slot[d] == NONE).collect();
    if todo.is_empty() {
        return;
    }
    let mut q = DestSearch::new(g, p);
    for &dst in &todo {
        sk.slot[dst] = sk.next.len() as u32;
        let d = q.run(dst, Some(&mut sk.r));
        sk.next.push(d.next);
        sk.board.push(d.board);
        sk.order.push(d.order);
        sk.change.push(d.change);
    }
}

/// One destination's strategy, as a `Skims` slot holds it.
pub struct DestStrategy {
    pub next: NodeIds,
    pub board: Box<[u64]>,
    pub order: NodeIds,
    pub change: ChangeNodes,
}

/// A destination's strategy, borrowed from a `Skims` slot or a `DestStrategy`.
#[derive(Clone, Copy)]
pub struct Strategy<'a> {
    pub next: &'a NodeIds,
    pub board: &'a [u64],
    pub order: &'a NodeIds,
    pub change: &'a ChangeNodes,
}

impl NodeIds {
    fn bytes(&self) -> usize {
        match self {
            NodeIds::Short(v) => 2 * v.len(),
            NodeIds::Long(v) => 4 * v.len(),
        }
    }
}

impl DestStrategy {
    pub fn bytes(&self) -> usize {
        self.next.bytes() + self.order.bytes() + 8 * self.board.len() + 4 * (self.change.st.len() + self.change.off.len()) + 8 * self.change.bits.len()
    }
    pub fn view(&self) -> Strategy<'_> {
        Strategy { next: &self.next, board: &self.board, order: &self.order, change: &self.change }
    }
}

impl Skims {
    pub fn strategy(&self, slot: usize) -> Strategy<'_> {
        Strategy { next: &self.next[slot], board: &self.board[slot], order: &self.order[slot], change: &self.change[slot] }
    }
}

/// About the bytes one destination's strategy holds on `g` (`DestStrategy::bytes`, change nodes
/// left out): its next nodes and settle order, a bit per route node.
pub fn strategy_bytes(g: &Graph) -> usize {
    let id = if g.n < u16::MAX as usize { 2 } else { 4 };
    2 * id * g.n + (g.n - g.n_st).div_ceil(64) * 8
}

/// The strategy search one destination at a time, for a caller that does not hold every
/// destination's strategy at once (T-761).
pub struct DestSearch<'a> {
    q: Search<'a>,
}

impl<'a> DestSearch<'a> {
    pub fn new(g: &'a Graph, p: &'a Params) -> DestSearch<'a> {
        DestSearch { q: Search::new(g, p) }
    }
    /// Search towards `dst`: its strategy, and its column of route costs written into `r` (S x S,
    /// `Skims::r`'s layout) when given.
    pub fn run(&mut self, dst: usize, r: Option<&mut [f32]>) -> DestStrategy {
        let q = &mut self.q;
        let (s, n) = (q.s, q.n);
        q.run(dst);
        let ids = n + q.cn.len();
        let g = q.g;
        let curve = &g.fare_curve;
        let km = if r.is_none() || curve.is_line() { vec![] } else { q.trip_km() };
        let d = DestStrategy { next: NodeIds::of(&q.next, ids), board: std::mem::take(&mut q.board), order: NodeIds::of(&q.order, ids), change: q.change_nodes() };
        if let Some(r) = r {
            for src in 0..s {
                // The first boarding is not a transfer; the trip's base fare is added later.
                r[src * s + dst] = if src == dst { 0.0 } else { q.dist[src] - q.p.transfer_min + km.get(src).map_or(0.0, |&k| curve.extra(k)) };
            }
        }
        d
    }
}

/// The strategy search's queue (T-581): a 4-ary min-heap of (cost bits, id), each id in it at
/// most once. A lower cost for an id already queued moves it up instead of adding a copy, so it
/// pops the ids in the order a heap of copies did (by cost, then id; costs are never negative, so
/// their bits sort as they do), with no stale copies to pop and skip.
struct IdHeap {
    h: Vec<u64>,
    /// Each id's place in `h`, `NONE` when not queued.
    pos: Vec<u32>,
}

impl IdHeap {
    fn new(ids: usize) -> IdHeap {
        IdHeap { h: Vec::with_capacity(ids), pos: vec![NONE; ids] }
    }
    #[inline(always)]
    fn push(&mut self, cost: f32, id: u32) {
        let x = (cost.to_bits() as u64) << 32 | id as u64;
        let mut i = match self.pos[id as usize] {
            NONE => {
                self.h.push(x);
                self.h.len() - 1
            }
            i if x < self.h[i as usize] => i as usize,
            _ => return,
        };
        while i > 0 {
            let up = (i - 1) / 4;
            let y = self.h[up];
            if y <= x { break; }
            self.h[i] = y;
            self.pos[y as u32 as usize] = i as u32;
            i = up;
        }
        self.h[i] = x;
        self.pos[id as usize] = i as u32;
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<usize> {
        let top = *self.h.first()?;
        self.pos[top as u32 as usize] = NONE;
        let x = self.h.pop().unwrap();
        let len = self.h.len();
        if len > 0 {
            let mut i = 0;
            loop {
                let c = 4 * i + 1;
                if c >= len { break; }
                let (mut m, mut y) = (c, self.h[c]);
                for k in c + 1..(c + 4).min(len) {
                    if self.h[k] < y {
                        m = k;
                        y = self.h[k];
                    }
                }
                if y >= x { break; }
                self.h[i] = y;
                self.pos[y as u32 as usize] = i as u32;
                i = m;
            }
            self.h[i] = x;
            self.pos[x as u32 as usize] = i as u32;
        }
        Some(top as u32 as usize)
    }
}

/// A change node being searched (`ChangeNodes`): its station and the line it was reached off, the
/// pooled frequency and frequency-weighted cost of the lines it accepts, its cost, settled, its
/// arrivals (a range of `Search::cn_feed`: the station's incoming edges off that line) and where
/// its accepted departures' bits start in `Search::cn_bits`.
struct Change {
    st: u32,
    line: u32,
    frequency: f32,
    weighted: f32,
    dist: f32,
    done: bool,
    feed: (u32, u32),
    bits: u32,
}

/// One destination's strategy search at a time (`skims_extend`). Heap ids: nodes below `n`; from
/// `n`, a boarding onto route node `id - n` that waited its turn (T-552); from `n + nr`, change
/// node `id - n - nr`.
struct Search<'a> {
    g: &'a Graph,
    p: &'a Params,
    s: usize,
    n: usize,
    nr: usize,
    dist: Vec<f32>,
    done: Vec<bool>,
    frequency: Vec<f32>,
    weighted: Vec<f32>,
    heap: IdHeap,
    next: Vec<u32>,
    order: Vec<u32>,
    board: Box<[u64]>,
    /// What each boarding route node offered its station, infinite if it has not.
    offer: Vec<f32>,
    cn: Vec<Change>,
    /// Each station's change nodes, a range of `cn`.
    cn_of: Vec<(u32, u32)>,
    /// The change nodes' accepted departures, a bit per boarding edge of their station.
    cn_bits: Vec<u64>,
    cn_feed: Vec<u32>,
    lines: Vec<u32>,
    /// (index in `lines`, incoming edge) for a settling station's arrivals off its own lines.
    feed: Vec<(u32, u32)>,
    cand: Vec<(f32, u32)>,
    /// Per station, the one line with departures from it, `MANY` for several, `NONE` for none.
    lone: Vec<u32>,
    /// Each route node's place among its station's boarding edges.
    edge_ix: Vec<u32>,
    /// The platforms (relative to the station) and frequencies a settling node boards (T-717).
    pool: Vec<(f32, f32, f32)>,
}

const MANY: u32 = NONE - 1;

impl Search<'_> {
    fn new<'a>(g: &'a Graph, p: &'a Params) -> Search<'a> {
        let (s, n) = (g.n_st, g.n);
        let nr = n - s;
        let mut edge_ix = vec![NONE; nr];
        let mut lone = vec![NONE; s];
        for v in 0..s {
            let e0 = g.adj_off[v] as usize;
            for e in e0..g.adj_off[v + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                edge_ix[r] = (e - e0) as u32;
                if g.rn_tph[r] > 0.0 {
                    lone[v] = match lone[v] {
                        NONE => g.rn_line[r],
                        l if l == g.rn_line[r] => l,
                        _ => MANY,
                    };
                }
            }
        }
        Search {
            g,
            p,
            s,
            n,
            nr,
            dist: vec![f32::INFINITY; n],
            done: vec![false; n],
            frequency: vec![0f32; s],
            weighted: vec![0f32; s],
            // a change node per station and line its strategy boards: fewer than route nodes
            heap: IdHeap::new(n + 2 * nr),
            next: vec![NONE; n],
            order: Vec::with_capacity(n),
            board: Box::new([]),
            offer: vec![f32::INFINITY; nr],
            cn: vec![],
            cn_of: vec![(0, 0); s],
            cn_bits: vec![],
            cn_feed: vec![],
            lines: vec![],
            feed: vec![],
            cand: vec![],
            lone,
            edge_ix,
            pool: vec![],
        }
    }

    fn run(&mut self, dst: usize) {
        let (g, s, n, nr) = (self.g, self.s, self.n, self.nr);
        self.dist.fill(f32::INFINITY);
        self.done.fill(false);
        self.frequency.fill(0.0);
        self.weighted.fill(0.0);
        self.next.fill(NONE);
        self.order.clear();
        self.board = vec![0u64; nr.div_ceil(64)].into_boxed_slice();
        self.offer.fill(f32::INFINITY);
        self.cn.clear();
        self.cn_of.fill((0, 0));
        self.cn_bits.clear();
        self.cn_feed.clear();
        self.dist[dst] = 0.0;
        self.heap.push(0.0, dst as u32);
        while let Some(id) = self.heap.pop() {
            if id >= n + nr {
                self.settle_change(id - n - nr);
                continue;
            }
            if id >= n {
                // a boarding onto a line that leaves riders behind, at its turn (T-552)
                let r = id - n;
                self.offer_line(r, self.dist[s + r] + g.rn_left[r]);
                continue;
            }
            let v = id;
            if self.done[v] { continue; }
            self.done[v] = true;
            self.order.push(v as u32);
            let d = self.dist[v];
            if v < s {
                self.settle_station(v);
                continue;
            }
            for j in g.rev_off[v] as usize..g.rev_off[v + 1] as usize {
                let from = g.rev_from[j] as usize;
                if from < s {
                    let f = g.rn_tph[v - s];
                    if f == 0.0 { continue; }
                    // Lines join a station's strategy in order of their cost from boarding, which
                    // for a line leaving riders behind includes the extra wait (T-188): it waits
                    // its turn, or a full train would join ahead of better lines and stay (T-552).
                    let left = g.rn_left[v - s];
                    if left > 0.0 {
                        self.heap.push(d + left, (n + v - s) as u32);
                    } else {
                        self.offer_line(v - s, d);
                    }
                    continue;
                }
                if self.done[from] { continue; }
                self.relax(from, d + g.adj_cost[g.rev_edge[j] as usize], v as u32);
            }
        }
    }

    #[inline(always)]
    fn relax(&mut self, from: usize, nd: f32, to: u32) {
        if nd < self.dist[from] {
            self.dist[from] = nd;
            self.next[from] = to;
            self.heap.push(nd, from as u32);
        }
    }

    /// Route node `r`'s departures, at cost `d` from boarding, offered to its station's strategy,
    /// or once that is settled, to the station's change nodes off other lines.
    fn offer_line(&mut self, r: usize, d: f32) {
        let g = self.g;
        let (from, f) = (g.rn_station[r] as usize, g.rn_tph[r]);
        self.offer[r] = d;
        if !self.done[from] {
            if let Some(cost) = join_line(&mut self.frequency[from], &mut self.weighted[from], self.dist[from], f, d, self.p) {
                self.board[r >> 6] |= 1 << (r & 63);
                self.dist[from] = cost;
                self.next[from] = (self.s + r) as u32;
                self.heap.push(cost, from as u32);
            }
            return;
        }
        let (a, b) = self.cn_of[from];
        let i = self.edge_ix[r] as usize;
        for k in a as usize..b as usize {
            let c = &mut self.cn[k];
            // a line costing more from boarding than the pool does already never joins it
            if c.done || d >= c.dist || c.line == g.rn_line[r] { continue; }
            if let Some(cost) = join_line(&mut c.frequency, &mut c.weighted, c.dist, f, d, self.p) {
                c.dist = cost;
                self.cn_bits[c.bits as usize + i / 64] |= 1 << (i % 64);
                self.heap.push(cost, (self.n + self.nr + k) as u32);
            }
        }
    }

    /// Station `v` is settled with its strategy. Riders reach it off the lines that strategy does
    /// not board as they would start there; off one it boards, they come to its change node off
    /// that line, made here from the departures offered so far less that line's (T-575).
    fn settle_station(&mut self, v: usize) {
        let (g, s) = (self.g, self.s);
        let d = self.dist[v];
        let (e0, e1) = (g.adj_off[v] as usize, g.adj_off[v + 1] as usize);
        self.lines.clear();
        for e in e0..e1 {
            let r = g.adj_to[e] as usize - s;
            if self.board[r >> 6] >> (r & 63) & 1 == 1 && !self.lines.contains(&g.rn_line[r]) {
                self.lines.push(g.rn_line[r]);
            }
        }
        let spread = g.st_spread[v];
        if spread {
            self.pool.clear();
            for e in e0..e1 {
                let r = g.adj_to[e] as usize - s;
                if self.board[r >> 6] >> (r & 63) & 1 == 1 {
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], g.rn_tph[r]));
                }
            }
        }
        // one pass over the arrivals: those off other lines go on, those off its own wait
        self.feed.clear();
        for j in g.rev_off[v] as usize..g.rev_off[v + 1] as usize {
            let from = g.rev_from[j] as usize;
            if self.done[from] { continue; }
            match self.lines.iter().position(|&l| l == g.rn_line[from - s]) {
                Some(li) => self.feed.push((li as u32, j as u32)),
                None => {
                    let walk = if spread { platform_walk(g, self.p, &self.pool, from, v) } else { 0.0 };
                    self.relax(from, d + g.adj_cost[g.rev_edge[j] as usize] + walk, v as u32)
                }
            }
        }
        let first = self.cn.len() as u32;
        // a station with no other line has no change node
        let lone = self.lone[v];
        if !self.feed.is_empty() && lone != NONE {
            for li in 0..self.lines.len() {
                let line = self.lines[li];
                if lone == line { continue; }
                let f0 = self.cn_feed.len();
                self.cn_feed.extend(self.feed.iter().filter(|f| f.0 == li as u32).map(|f| f.1));
                if self.cn_feed.len() == f0 { continue; }
                self.cand.clear();
                for e in e0..e1 {
                    let r = g.adj_to[e] as usize - s;
                    if g.rn_line[r] != line && self.offer[r].is_finite() {
                        self.cand.push((self.offer[r], r as u32));
                    }
                }
                if self.cand.len() > 1 {
                    self.cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                }
                let k = self.cn.len();
                let b0 = self.cn_bits.len();
                self.cn_bits.resize(b0 + (e1 - e0).div_ceil(64), 0);
                let mut c = Change { st: v as u32, line, frequency: 0.0, weighted: 0.0, dist: f32::INFINITY, done: false, feed: (f0 as u32, self.cn_feed.len() as u32), bits: b0 as u32 };
                for &(od, r) in &self.cand {
                    if let Some(cost) = join_line(&mut c.frequency, &mut c.weighted, c.dist, g.rn_tph[r as usize], od, self.p) {
                        c.dist = cost;
                        let i = self.edge_ix[r as usize] as usize;
                        self.cn_bits[b0 + i / 64] |= 1 << (i % 64);
                    }
                }
                if c.dist.is_finite() {
                    self.heap.push(c.dist, (self.n + self.nr + k) as u32);
                }
                self.cn.push(c);
            }
        }
        self.cn_of[v] = (first, self.cn.len() as u32);
    }

    fn settle_change(&mut self, k: usize) {
        let g = self.g;
        if self.cn[k].done { return; }
        self.cn[k].done = true;
        let id = (self.n + k) as u32;
        self.order.push(id);
        let (d, (f0, f1)) = (self.cn[k].dist, self.cn[k].feed);
        let v = self.cn[k].st as usize;
        let spread = g.st_spread[v];
        if spread {
            self.pool.clear();
            let (s, e0, b0) = (self.s, g.adj_off[v] as usize, self.cn[k].bits as usize);
            for e in e0..g.adj_off[v + 1] as usize {
                let i = e - e0;
                if self.cn_bits[b0 + i / 64] >> (i % 64) & 1 == 1 {
                    let r = g.adj_to[e] as usize - s;
                    self.pool.push((g.rn_dx[r], g.rn_dy[r], g.rn_tph[r]));
                }
            }
        }
        for i in f0 as usize..f1 as usize {
            let j = self.cn_feed[i] as usize;
            let from = g.rev_from[j] as usize;
            if self.done[from] { continue; }
            let walk = if spread { platform_walk(g, self.p, &self.pool, from, v) } else { 0.0 };
            self.relax(from, d + g.adj_cost[g.rev_edge[j] as usize] + walk, id);
        }
    }

    /// Each node's expected fare km to the destination under its strategy (T-883), change node k
    /// at `n + k`: a station or change node averages the lines it boards by their departures, as
    /// the riders split. Read in settle order, so every next node is done first.
    fn trip_km(&self) -> Vec<f32> {
        let (g, s, n) = (self.g, self.s, self.n);
        let mut km = vec![0f32; n + self.cn.len()];
        let pooled = |km: &[f32], st: usize, on: &dyn Fn(usize) -> bool| {
            let (mut f, mut w) = (0f32, 0f32);
            let e0 = g.adj_off[st] as usize;
            for e in e0..g.adj_off[st + 1] as usize {
                let r = g.adj_to[e] as usize - s;
                if on(e - e0) {
                    f += g.rn_tph[r];
                    w += g.rn_tph[r] * km[s + r];
                }
            }
            if f > 0.0 { w / f } else { 0.0 }
        };
        for &id in &self.order {
            let id = id as usize;
            let v = if id >= n {
                let c = &self.cn[id - n];
                let b0 = c.bits as usize;
                pooled(&km, c.st as usize, &|i| self.cn_bits[b0 + i / 64] >> (i % 64) & 1 == 1)
            } else if id < s {
                let e0 = g.adj_off[id] as usize;
                pooled(&km, id, &|i| {
                    let r = g.adj_to[e0 + i] as usize - s;
                    self.board[r >> 6] >> (r & 63) & 1 == 1
                })
            } else {
                let nx = self.next[id] as usize;
                if nx == NONE as usize { 0.0 } else if nx >= s && nx < n { g.rn_km.get(id - s).copied().unwrap_or(0.0) + km[nx] } else { km[nx] }
            };
            km[id] = v;
        }
        km
    }

    /// This destination's change nodes, settled or not (an unsettled one is nobody's next node).
    fn change_nodes(&self) -> ChangeNodes {
        ChangeNodes {
            st: self.cn.iter().map(|c| c.st).collect(),
            off: self.cn.iter().map(|c| c.bits).chain([self.cn_bits.len() as u32]).collect(),
            bits: self.cn_bits.as_slice().into(),
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

/// Where a rider coming in by edge from route node `from` stands at station `v`, relative to its
/// position: the platform they alighted at, or the station's position after a walk from another.
#[inline]
pub(crate) fn arrival_at(g: &Graph, from: usize, v: usize) -> (f32, f32) {
    let r = from - g.n_st;
    if g.rn_station[r] as usize == v { (g.rn_dx[r], g.rn_dy[r]) } else { (0.0, 0.0) }
}

/// Perceived minutes walking from (x, y) to the platform at (px, py), both relative to their
/// station (T-717).
#[inline]
pub(crate) fn platform_walk_to(p: &Params, (x, y): (f32, f32), px: f32, py: f32) -> f32 {
    if px == x && py == y { 0.0 } else { p.xfer_min(((px - x).powi(2) + (py - y).powi(2)).sqrt()) * p.w_xfer }
}

/// A change at a station of several platforms walks from where the rider arrives to the platform
/// of the train they take: over `pool` (platform, departures an hour), the expected walk as the
/// first departure decides (T-717). Added to the arrival's cost rather than to each line's in the
/// pool, so which lines join the station's strategy does not depend on the platform arrived at.
#[inline]
fn platform_walk(g: &Graph, p: &Params, pool: &[(f32, f32, f32)], from: usize, v: usize) -> f32 {
    let at = arrival_at(g, from, v);
    let (mut w, mut f) = (0.0, 0.0);
    for &(px, py, fr) in pool {
        w += fr * platform_walk_to(p, at, px, py);
        f += fr;
    }
    if f > 0.0 { w / f } else { 0.0 }
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
    rn_dx: Vec<f32>,
    rn_dy: Vec<f32>,
    transfer_min: f32,
    fare_curve: FareCurve,
    rn_km: Vec<f32>,
}

impl SkimKey {
    fn of(g: &Graph, p: &Params) -> SkimKey {
        SkimKey { n_st: g.n_st, n: g.n, adj_off: g.adj_off.clone(), adj_to: g.adj_to.clone(), adj_cost: g.adj_cost.clone(), rn_tph: g.rn_tph.clone(), rn_left: g.rn_left.clone(), rn_dx: g.rn_dx.clone(), rn_dy: g.rn_dy.clone(), transfer_min: p.transfer_min, fare_curve: g.fare_curve.clone(), rn_km: g.rn_km.clone() }
    }
    fn same(&self, g: &Graph, p: &Params) -> bool {
        self.n_st == g.n_st
            && self.n == g.n
            && self.transfer_min == p.transfer_min
            && self.fare_curve == g.fare_curve
            && self.rn_km == g.rn_km
            && self.rn_dx == g.rn_dx
            && self.rn_dy == g.rn_dy
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

/// Whether this thread holds skims of `g` (`with_skims`).
pub fn skims_cached(g: &Graph, p: &Params) -> bool {
    !legacy() && SKIM_CACHE.with(|c| c.borrow().iter().any(|(k, _)| k.same(g, p)))
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

/// exp(-theta (station-to-station cost - offset)), S x S per band of `Params::band` one after
/// the other, with a zero diagonal: boarding and alighting at the same station is not a rail
/// trip. The access logsum per origin subzone is built from it.
pub fn station_exp(sk: &Skims, p: &Params) -> Vec<f32> {
    let s = sk.s;
    let mut er: Vec<f32> = Vec::with_capacity(p.n_bands() * s * s);
    for b in 0..p.n_bands() {
        let (th, off) = p.band(b);
        let at = er.len();
        er.extend(sk.r.iter().map(|&r| if r.is_finite() { fast_exp(-th * (r - off)) } else { 0.0 }));
        for a in 0..s {
            er[at + a * s + a] = 0.0;
        }
    }
    er
}

/// Distance bands (km, upper edges) for rail trips by how far home is from the boarding station.
pub const DIST_BANDS_KM: [f32; N_BANDS] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, f32::INFINITY];
pub const N_BANDS: usize = 13;
pub const N_WALKSUM: usize = 16;
pub const WALKSUM_BAND_MIN: f32 = 10.0;
/// T-874: `Params::commute_chunk`'s value in the game (notes/T-874.md).
pub const COMMUTE_CHUNK: f32 = 0.1;
/// T-874: the side of the blocks of destination zones (16 km for 2 km zones) within which an
/// origin zone's small commute pairs are dealt their trips in chunks (`Params::commute_chunk`).
const CHUNK_BLOCK: i32 = 8;

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
    /// Trips walking the whole way with no rail that rail takes (T-558): over the pairs offered
    /// rail, rail's share times walking's share with no rail (`Params::car_choice`). Walking's
    /// trips are those with no rail (`Gravity::no_rail_by_zone`) less these.
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
    /// Rail trips per access entry (`Subzones::acc_st`): at the home end, boarding at that
    /// station from that subzone; at the work end, leaving the train there for that subzone.
    pub acc_home: Vec<f32>,
    pub acc_work: Vec<f32>,
}

/// Rail, the car and walking the whole way (T-558; the zone pair's walk) in one logit per
/// (origin subzone, destination subzone) pair; rail's cost is a logit over access and egress
/// stations (SPEC 4.3, 4.4; notes/T-020.md, T-078.md).
///
/// For a pair (u, v) the rail alternatives are every (access station a of u, egress station t of
/// v); each costs access_a + R[a, t] + egress_t. Their logsum separates:
///     S_uv = sum_t A_u[t] E_v[t],  A_u[t] = sum_a exp(-theta (access_a - m_u)) exp(-theta R[a, t]),
///     E_v[t] = exp(-theta (egress_t - m_v)),  rail cost = m_u + m_v + fare - ln(S_uv) / theta.
/// Theta is the pair's distance band's (T-710, `Params::band`), its exp table offset by the
/// band's offset, which the cost adds back; an origin builds its rows for the bands it uses.
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
pub fn mode_choice(z: &Zones, gr: &Gravity, sz: &Subzones, er: &[f32], s: usize, p: &Params, nr: &NoRail) -> ModeOut {
    let share = p.period_share as f64;
    // T-710: the station choice's scale and offset by distance band (`er` holds a table per band)
    let nb = p.n_bands();
    let bands: Vec<(f32, f32)> = (0..nb).map(|b| p.band(b)).collect();
    let (ss, na) = (s * s, sz.acc_st.len());
    // per subzone: its part of the zone's row / column factor
    let ou: Vec<f32> = (0..sz.n).map(|u| sz.origin_weight(u, z, gr, share)).collect();
    let dv: Vec<f32> = (0..sz.n)
        .map(|v| {
            let zj = sz.zone[v] as usize;
            if z.jobs[zj] > 0.0 { (gr.col[zj] * sz.jobs[v] as f64 / z.jobs[zj]) as f32 } else { 0.0 }
        })
        .collect();
    let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
    // per band, subzone and access entry: weights exp(-theta (cost - m)), a walk entry's with m
    // the cheapest walk (`acc_m`, the same at both ends; infinite with none), a ride entry's with
    // m the cheapest of walks and rides (`ride_m`)
    let mut acc_m = vec![f32::INFINITY; sz.n];
    let mut ride_m = vec![f32::INFINITY; sz.n];
    let mut acc_e = vec![0f32; nb * na];
    let min_of = |r: std::ops::Range<usize>| sz.acc_w[r].iter().cloned().fold(f32::INFINITY, f32::min);
    for v in 0..sz.n {
        let m = min_of(sz.walk(v));
        let mr = m.min(min_of(sz.ride(v)));
        acc_m[v] = m;
        ride_m[v] = mr;
        for (b, &(th, _)) in bands.iter().enumerate() {
            for k in sz.walk(v) {
                acc_e[b * na + k] = fast_exp(-th * (sz.acc_w[k] - m));
            }
            for k in sz.ride(v) {
                acc_e[b * na + k] = fast_exp(-th * (sz.acc_w[k] - mr));
            }
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
        // its lower-left square, centre in half squares and drive supercell, for the zone
        // pair's factors without reading `z`
        gx: i32,
        gy: i32,
        hx: i32,
        hy: i32,
        sup: u16,
    }
    let mut dz: Vec<DZone> = Vec::with_capacity(dzones.len());
    // per destination: [x, y, local streets, main roads], [parking every driver pays, trips
    // factor, walk, a paid space], [the share parking free, -, -, -]
    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());
    // its subzone, and the walkers rail takes there (`walk_lost_to_sub`)
    let mut dd_sub: Vec<u32> = Vec::with_capacity(sz.zd.len());
    // and its egress stations, their weights and `Subzones::acc_st` entries, padded to K with
    // station `s` (read as a dummy past the row's end, 0) at weight 0: a pad adds exactly 0 to
    // every sum it enters, so eight destinations go through the same steps
    let (mut eg_st, mut eg_e, mut eg_k, mut eg_n): (Vec<[u32; K]>, Vec<[f32; K]>, Vec<[u32; K]>, Vec<u8>) = (vec![], vec![], vec![], vec![]);
    for &j in &dzones {
        let q0 = dd.len() as u32;
        let mut dvz = 0.0f32;
        for &v in &sz.zd[sz.zd_off[j] as usize..sz.zd_off[j + 1] as usize] {
            let v = v as usize;
            let (mut st, mut kk) = ([s as u32; K], [0u32; K]);
            for (n, k) in sz.walk(v).enumerate() {
                (st[n], kk[n]) = (sz.acc_st[k], k as u32);
            }
            // its weights per band: destination q's band b at `eg_e[q * nb + b]`
            for b in 0..nb {
                let mut e = [0f32; K];
                for (n, k) in sz.walk(v).enumerate() {
                    e[n] = acc_e[b * na + k];
                }
                eg_e.push(e);
            }
            eg_st.push(st);
            eg_k.push(kk);
            eg_n.push(sz.walk(v).len() as u8);
            dvz += dv[v];
            let (fixed, paid, free) = park[j];
            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);
            dd_sub.push(v as u32);
        }
        let q1 = dd.len() as u32;
        dz.push(DZone { key: gr.key(z, j), reg: z.region[j], zid: j as u32, min: zmin(j), max: dd[q1 as usize - 1][1][2], dv: dvz, q0, q1, gx: z.gx[j], gy: z.gy[j], hx: z.hx(j), hy: z.hy(j), sup: p.routes.as_ref().map_or(0, |r| r.sup_of(j)) });
    }
    // T-874: destination zones by block of `CHUNK_BLOCK` square, in zone order within one: the
    // order an origin zone's small pairs are dealt their chunks in, so an edit re-deals only the
    // blocks whose zones' trips it changed
    let blk = |j: usize| (z.gx[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff) << 16 | (z.gy[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff);
    let mut dorder: Vec<(u32, u32)> = dz.iter().enumerate().map(|(k, zj)| (blk(zj.zid as usize), k as u32)).collect();
    dorder.sort_by_key(|&(b, k)| (b, dz[k as usize].zid));
    // each destination's factor on its trips from the origin zone: 1, or what the chunks dealt
    // it over its trips (0 for none)
    let mut fm = vec![1f32; dz.len()];
    let mut run: Vec<(usize, f64)> = vec![];
    let mut lost_q = vec![0f32; dd.len()];
    // the decay and the walk's cost by zone offset side by side: one lookup per zone pair
    let fw: Vec<[f32; 2]> = gr.ftab.iter().enumerate().map(|(k, &f)| [f, gr.walk_c(k)]).collect();
    // with a third: the road's length over the straight line between the zones (T-275); and a
    // fourth, 1 where the commute is long enough for the ride (T-371)
    let mut fz = vec![[0f32, 0.0, p.road_detour, 0.0]; dz.len()];
    // and the zone pair's walking with no rail (`NoRail::split`, T-558), and its distance band
    let mut fzw = vec![0f32; dz.len()];
    let mut fzb = vec![0u8; dz.len()];
    let pair_walk = gr.wroutes.is_some() || !gr.wf_off.is_empty();
    let drive = p.routes.as_deref().filter(|r| r.has_drive());
    let ride_min2 = (p.ride_min_km * 1000.0).powi(2);
    // the origin's two rows (walks; walks and rides) one after the other, each followed by
    // `PADS` dummies (0), and their rail trips by egress station the same way; a pair per band,
    // band b's at 2 b h
    let h = s + PADS;
    let mut ab_row = vec![0f32; 2 * nb * h];
    let mut ab_y = vec![0f32; 2 * nb * h];
    // rail trips reaching each destination's egress entries, by its egress step
    let mut acc_q = vec![[[0f32; 4]; 2]; dd.len()];
    // the origin's pairs within the walk bound, in visiting order: destination, and the zone
    // visit (`visit`: [trips at the origin x decay, walking's cost, road factor, origin walk]),
    // its top bit set for the walk-and-ride row
    let mut cand: Vec<[u32; 2]> = Vec::with_capacity(dd.len() + 7);
    let mut visit: Vec<[f32; 4]> = Vec::with_capacity(dz.len());
    let mut visit_w: Vec<f32> = Vec::with_capacity(dz.len());
    let mut visit_b: Vec<u8> = Vec::with_capacity(dz.len());
    let mut walksum = [0f64; N_WALKSUM];
    for &i in &ozones {
        let base = gr.base(z, i);
        let us = &sz.zo[sz.zo_off[i] as usize..sz.zo_off[i + 1] as usize];
        let ou_zone: f32 = us.iter().map(|&u| ou[u as usize]).sum();
        // the decay (times the region factor) and walk's cost to every destination zone one of
        // this zone's origins reaches, once for all of them
        let ri = z.region[i];
        let nzi = us.iter().map(|&u| reach(u as usize)).max().unwrap_or(0);
        let (gxi, gyi, hxi, hyi) = (z.gx[i], z.gy[i], z.hx(i), z.hy(i));
        for ((o, zb), zj) in fz[..nzi].iter_mut().zip(fzb.iter_mut()).zip(&dz) {
            let k = (base + zj.key) as usize;
            let [f, w] = fw[k];
            // `ride_offered`
            let (dx, dy) = (zj.gx - gxi, zj.gy - gyi);
            let (fx, fy) = ((zj.hx - hxi) as f32 * 0.5, (zj.hy - hyi) as f32 * 0.5);
            let d2 = (fx * fx + fy * fy) * z.zone_m * z.zone_m;
            let long = (d2 >= ride_min2) as u32 as f32;
            *zb = p.band_of_d2(d2) as u8;
            *o = [f, w, o[2], long];
            if zj.reg != ri {
                o[0] *= gr.border;
            }
            if pair_walk && w < NO_WALK {
                o[1] = gr.walk_cost_at(i, zj.zid as usize, k);
            }
            if let Some(r) = drive {
                // `Params::drive_factor`
                o[2] = r.drive_factor_by(i, dx, dy, zj.reg, zj.sup) * p.road_routed_scale;
            }
        }
        for (wz, (o, zj)) in fzw[..nzi].iter_mut().zip(fz.iter().zip(&dz)) {
            *wz = if o[1] < NO_WALK { nr.split(gr, p, i, zj.zid as usize, (base + zj.key) as usize) } else { 0.0 };
        }
        // T-874: per block, the pairs under `commute_chunk` trips are dealt their total in chunks
        // of about that size, as the market's (`intercity::solved_pairs`): systematic sampling by
        // trips from a start fixed by the zone and block, so the block's trips stay exact and a
        // network gives the same pairs every pass. Trips alone decide it, never rail's costs, so
        // crowding rounds and timetable edits deal the same chunks.
        fm[..nzi].fill(1.0);
        if p.commute_chunk > 0.0 {
            let c = p.commute_chunk as f64;
            let mut g = 0;
            while g < dorder.len() {
                let b = dorder[g].0;
                run.clear();
                while g < dorder.len() && dorder[g].0 == b {
                    let k = dorder[g].1 as usize;
                    g += 1;
                    if k < nzi {
                        let t = (ou_zone * dz[k].dv * fz[k][0]) as f64;
                        if t > 0.0 && t < c {
                            run.push((k, t));
                        }
                    }
                }
                let total: f64 = run.iter().map(|e| e.1).sum();
                let draws = (total / c).ceil();
                if draws >= run.len() as f64 {
                    continue;
                }
                let step = total / draws;
                let at = super::intercity::unit_hash(i as u32, b);
                // draws before trips x: those at (n + at) x step below x
                let before = |x: f64| (x / step - at).ceil().clamp(0., draws);
                let mut lo = 0.;
                for &(k, t) in &run {
                    let hi = lo + t;
                    fm[k] = ((before(hi) - before(lo)) * step / t) as f32;
                    lo = hi;
                }
            }
        }
        for (ui, &u) in us.iter().enumerate() {
            let u = u as usize;
            let (walk, ride) = (sz.walk(u), sz.ride(u));
            let (m_u, m_r) = (acc_m[u], ride_m[u]);
            let rides = !ride.is_empty();
            // the walk entries' weight at the rides' offset, per band
            let c_u: Vec<f32> = bands.iter().map(|&(th, _)| if walk.is_empty() { 0.0 } else { fast_exp(-th * (m_u - m_r)) }).collect();
            let (ux8, uy8, lu08, lu18) = (F8::splat(sz.cx[u]), F8::splat(sz.cy[u]), F8::splat(local[u].0), F8::splat(local[u].1));
            let mut rail_sum = 0.0f32;
            let mut walk_sum = 0.0f32;
            let mut all_sum = 0.0f32;
            let mut n_pairs = 0usize;
            // past this, a destination's walk takes the pair over `pair_walk_max` (by walk, and
            // by walk or ride)
            let wmax = p.pair_walk_max - m_u;
            let wmax_r = p.pair_walk_max - m_r;
            let hp8 = F8::splat(sz.home_park[u]);
            let nz = reach(u);
            // the bands this origin's pairs fall in
            let mut used = 0u32;
            for ((((zj, &[f, cw, df, long]), &w0), &zb), &m) in dz[..nz].iter().zip(&fz).zip(&fzw).zip(&fzb).zip(&fm) {
                let by_ride = rides && long > 0.0;
                let (m_o, wmax) = if by_ride { (m_r, wmax_r) } else { (m_u, wmax) };
                if zj.min > wmax {
                    continue;
                }
                if p.prune_trips > 0.0 && ou_zone * zj.dv * f < p.prune_trips {
                    out.pruned_trips += (ou[u] * zj.dv * f) as f64;
                    continue;
                }
                if m == 0.0 {
                    continue;
                }
                if ui == 0 {
                    out.zone_pairs += 1;
                }
                let tu = ou[u] * f * m;
                all_sum += tu * zj.dv;
                // cheapest walk first: past the bound, every later one is too
                let nv = if zj.max <= wmax { zj.q1 - zj.q0 } else { dd[zj.q0 as usize..zj.q1 as usize].partition_point(|d| d[1][2] <= wmax) as u32 };
                n_pairs += nv as usize;
                let zv = visit.len() as u32 | (by_ride as u32) << 31;
                visit.push([tu, cw, df, m_o]);
                visit_w.push(w0);
                visit_b.push(zb);
                used |= 1 << zb;
                cand.extend((zj.q0..zj.q0 + nv).map(|q| [q, zv]));
            }
            out.pairs += n_pairs as u64;
            // the origin's rows in each band its pairs use
            for (b, &c_ub) in c_u.iter().enumerate() {
                if used >> b & 1 == 0 {
                    continue;
                }
                let at = 2 * b * h;
                ab_y[at..at + 2 * h].fill(0.0);
                let er_b = &er[b * ss..(b + 1) * ss];
                let (a_row, b_row) = ab_row[at..at + 2 * h].split_at_mut(h);
                let (a_row, b_row) = (&mut a_row[..s], &mut b_row[..s]);
                a_row.fill(0.0);
                for k in walk.clone() {
                    let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                    for (av, &e) in a_row.iter_mut().zip(&er_b[a * s..(a + 1) * s]) {
                        *av += w * e;
                    }
                }
                // the row with the rides too, for the long commutes
                if rides {
                    for (bv, &av) in b_row.iter_mut().zip(a_row.iter()) {
                        *bv = c_ub * av;
                    }
                    for k in ride.clone() {
                        let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                        for (bv, &e) in b_row.iter_mut().zip(&er_b[a * s..(a + 1) * s]) {
                            *bv += w * e;
                        }
                    }
                }
            }
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
                // each lane's band, and its row: the band's pair, walks or walks and rides
                let bl: [usize; 8] = std::array::from_fn(|l| visit_b[(c[l][1] & 0x7fff_ffff) as usize] as usize);
                let rr: [u32; 8] = std::array::from_fn(|l| if c[l][1] >> 31 != 0 { h as u32 } else { 0 });
                let ro: [u32; 8] = std::array::from_fn(|l| (2 * bl[l] * h) as u32 + rr[l]);
                let th8 = F8::new(std::array::from_fn(|l| bands[bl[l]].0));
                let off8 = F8::new(std::array::from_fn(|l| bands[bl[l]].1));
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
                    ev[k] = F8::new(std::array::from_fn(|l| eg_e[q[l] * nb + bl[l]][k]));
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
                let [tu8, cw8, df8, mo8] = F8::transpose(std::array::from_fn(|l| visit[(c[l][1] & 0x7fff_ffff) as usize]));
                let (dx, dy) = (ux8 - x8, uy8 - y8);
                let d = (dx * dx + dy * dy).sqrt();
                // `Params::drive_min_f`
                let rd = d * F8::splat(0.001) * df8;
                let e = (F8::splat(0.5) * rd).min(F8::splat(p.road_local_km));
                let main = rd - F8::splat(2.0) * e;
                let k = (F8::splat(0.5) * main).min(F8::splat(p.road_end_km));
                let drive = e * (lu08 + l08) + k * (lu18 + l18) + (main - F8::splat(2.0) * k) * F8::splat(60.0) / F8::splat(p.road_open_kmh);
                // `Params::car_drive`, parking at the far end and at home (T-854)
                let road = drive * F8::splat(p.w_drive) + rd * F8::splat(p.car_own_km_min()) + park8 + hp8;
                let rail = mo8 + m8 + F8::splat(p.fare_min) - ssum.ln() / th8 + off8;
                // `Params::mode_beta`
                let bt = F8::splat(p.beta) * (F8::splat(p.damp_min) / road).min(F8::splat(1.0)).sqrt();
                let util = bt * (road - rail) + F8::splat(p.asc_rail);
                // T-558: and walking the whole way, against the same driver
                let uw = bt * (road - cw8) + F8::splat(p.asc_walk());
                // T-380: rail's share among drivers parking free, and among those paying for a
                // space (`Params::car_choice`; where no lane pays, the second share equals the first)
                let e = (uw - util).exp();
                let r0 = F8::splat(1.0) / (F8::splat(1.0) + util.neg().exp() + e);
                let r = if paid8.lt_mask(F8::splat(1e-30)) == 0xff {
                    r0
                } else {
                    let [free8, _, _, _] = F8::transpose(std::array::from_fn(|l| dd[q[l]][2]));
                    let r1 = F8::splat(1.0) / (F8::splat(1.0) + (util + bt * paid8).neg().exp() + e);
                    r0 + (F8::splat(1.0) - free8) * (r1 - r0)
                };
                // T-854: the curve's floor and ceiling (`Params::rail_bounds`)
                let r = if p.rail_floor > 0.0 {
                    let [free8, _, _, _] = F8::transpose(std::array::from_fn(|l| dd[q[l]][2]));
                    let car = road + (F8::splat(1.0) - free8) * paid8;
                    let (a, rm, k) = (F8::splat(p.rail_floor), F8::splat(p.rail_floor_ratio), F8::splat(1.0 / (p.rail_floor_ratio - 1.0)));
                    let (zero, one) = (F8::splat(0.0), F8::splat(1.0));
                    let lo = a * ((rm - rail / car) * k).max(zero).min(one);
                    let hi = a * ((rm - car / rail) * k).max(zero).min(one);
                    lo + r * (one - lo - hi - lo * e)
                } else {
                    r
                };
                let x = tu8 * dv8 * r;
                let g = x / ssum;
                // trips by egress station y, and y x the row: what reaches each egress entry
                let mut ya = [[0f32; 8]; K];
                let mut yw = [F8::splat(0.0); 8];
                let mut xw = F8::splat(0.0);
                for k in 0..kn {
                    let y = g * ev[k];
                    yw[k] = y * rv[k];
                    xw = xw + yw[k];
                    ya[k] = y.to_array();
                }
                // y x the row by lane, egress steps 0-3 and 4-7 (zeros past each list's end)
                let yw_lo = F8::by_lane([yw[0], yw[1], yw[2], yw[3]]);
                let yw_hi = if kn > 4 { F8::by_lane([yw[4], yw[5], yw[6], yw[7]]) } else { yw_lo };
                let (xa, xwa) = (x.to_array(), xw.to_array());
                // and the walkers it takes: its trips times the zone pair's walking with no rail
                // (`rail_walk`), as the zones' sums have it
                let wl = (x * F8::new(std::array::from_fn(|l| visit_w[(c[l][1] & 0x7fff_ffff) as usize]))).to_array();
                // each lane's walk-sum band, as the scalar (walk at home + at work) / band
                let bw = ((mo8 + m8) * F8::splat(1.0 / WALKSUM_BAND_MIN)).to_array();
                for l in 0..8 {
                    if ok >> l & 1 == 0 {
                        continue;
                    }
                    rail_sum += xa[l];
                    walk_sum += wl[l];
                    lost_q[q[l]] += wl[l];
                    // pads add 0 to their own dummies: the block's one count predicts well
                    for k in 0..kn {
                        ab_y[ix[k][l] as usize] += ya[k][l];
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
            visit_w.clear();
            visit_b.clear();
            out.trips_access += all_sum as f64;
            if rail_sum == 0.0 {
                continue;
            }
            out.rail_walk += walk_sum as f64;
            out.walk_lost_by_sub[u] = walk_sum;
            out.rail += rail_sum as f64;
            out.rail_by_sub[u] = rail_sum;
            // split this subzone's rail trips back onto its access stations, band by band: a walk
            // entry carries its share of both rows' trips, a ride entry of the long commutes' only
            for (b, &c_ub) in c_u.iter().enumerate() {
                if used >> b & 1 == 0 {
                    continue;
                }
                let at = 2 * b * h;
                let (y_row, yb_row) = (&ab_y[at..at + s], &ab_y[at + h..at + h + s]);
                for k in walk.start..ride.end {
                    let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                    let row = &mut out.od[a * s..(a + 1) * s];
                    let mut tot = 0.0f32;
                    let er_a = &er[b * ss + a * s..b * ss + (a + 1) * s];
                    if k < walk.end {
                        if rides {
                            for (((o, &y), &yb), &e) in row.iter_mut().zip(y_row).zip(yb_row).zip(er_a) {
                                let f = (y + c_ub * yb) * w * e;
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
    for (q, &v) in dd_sub.iter().enumerate() {
        out.walk_lost_to_sub[v as usize] = lost_q[q];
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
    let s = sk.s;
    let mut a = DestAssign::new(g, sk.most_change_nodes(), xf.is_some());
    for dst in 0..s {
        if !a.take(dst, |src| od[src * s + dst]) { continue; }
        let slot = sk.slot[dst] as usize;
        debug_assert!(slot != NONE as usize, "flow towards station {dst}, which was not searched");
        if slot == NONE as usize { continue; }
        a.walk(g, dst, sk.strategy(slot), xf.as_mut().map(|(ix, x)| (*ix, &mut **x)));
    }
    a.l
}

/// `assign`'s work one destination at a time, each destination's strategy handed in (T-761):
/// destinations taken in station order give `assign`'s loads to the bit.
pub struct DestAssign {
    pub l: Loads,
    flow: Vec<f32>,
    // T-176: riders arriving at each station towards this destination, as a list per station
    // and change node (first record, then each record's next): the arriving edge's first slot and
    // the riders. A station is walked after every node that feeds it, so its list is whole by then.
    head: Vec<u32>,
    rec: Vec<(u32, f32, u32)>,
    use_k: Vec<(u32, f32)>,
    dep: Vec<(u32, f32)>,
    counting: bool,
}

impl DestAssign {
    /// `change_nodes`: the most change nodes of a destination to come (more grows the tables);
    /// `counting`: transfers will be counted.
    pub fn new(g: &Graph, change_nodes: usize, counting: bool) -> DestAssign {
        let (s, n) = (g.n_st, g.n);
        DestAssign {
            l: Loads::zero(s, n - s),
            flow: vec![0f32; n + change_nodes],
            head: if counting { vec![NONE; s + change_nodes] } else { vec![] },
            rec: vec![],
            use_k: vec![],
            dep: vec![],
            counting,
        }
    }
    /// Take destination `dst`'s column of the OD, `col(src)` riders from each station; false
    /// when nobody rides there (there is nothing to walk).
    #[inline(always)]
    pub fn take(&mut self, dst: usize, col: impl Fn(usize) -> f32) -> bool {
        let s = self.l.entry.len();
        self.flow.fill(0.0);
        let mut total = 0.0;
        for src in 0..s {
            let v = col(src);
            self.flow[src] = v;
            self.l.entry[src] += v;
            total += v;
        }
        self.l.exit[dst] += total;
        total != 0.0
    }
    /// Walk the column taken towards `dst` along its strategy.
    pub fn walk(&mut self, g: &Graph, dst: usize, st: Strategy, xf: Option<(&XferIndex, &mut [f32])>) {
        let (s, n) = (g.n_st, g.n);
        let np = st.change.len();
        if self.flow.len() < n + np {
            self.flow.resize(n + np, 0.0);
        }
        if self.counting {
            if self.head.len() < s + np {
                self.head.resize(s + np, NONE);
            }
            self.head.fill(NONE);
            self.rec.clear();
        }
        let mut w = DestWalk { g, s, n, dst, bits: st.board, cn: st.change, flow: &mut self.flow, l: &mut self.l, head: &mut self.head, rec: &mut self.rec, use_k: &mut self.use_k, dep: &mut self.dep, xf };
        match (st.next, st.order) {
            (NodeIds::Short(next), NodeIds::Short(order)) => w.run(next, order),
            (NodeIds::Long(next), NodeIds::Long(order)) => w.run(next, order),
            _ => unreachable!("a destination's ids in two widths"),
        }
    }
}

/// One destination's part of `assign_inner`: its flows walked back along the settle order.
struct DestWalk<'a> {
    g: &'a Graph,
    s: usize,
    n: usize,
    dst: usize,
    bits: &'a [u64],
    cn: &'a ChangeNodes,
    flow: &'a mut [f32],
    l: &'a mut Loads,
    head: &'a mut [u32],
    rec: &'a mut Vec<(u32, f32, u32)>,
    use_k: &'a mut Vec<(u32, f32)>,
    /// The departures a station or change node accepts, (edge, trains an hour), in edge order.
    dep: &'a mut Vec<(u32, f32)>,
    xf: Option<(&'a XferIndex, &'a mut [f32])>,
}

/// The departures station or change node `v` accepts towards a destination, (boarding edge, trains
/// an hour) in edge order, into `dep`; returns the station. The other edges' shares are 0 and add
/// nothing, so these alone give the sums over every edge, to the bit (T-581).
#[inline(always)]
pub(crate) fn departures(g: &Graph, bits: &[u64], cn: &ChangeNodes, v: usize, dep: &mut Vec<(u32, f32)>) -> usize {
    let s = g.n_st;
    dep.clear();
    if v < s {
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let a = accepted(bits, g, g.adj_to[e] as usize - s);
            if a != 0.0 {
                dep.push((e as u32, a));
            }
        }
        v
    } else {
        let k = v - g.n;
        let st = cn.st[k] as usize;
        let e0 = g.adj_off[st] as usize;
        for (w, &b) in cn.bits[cn.off[k] as usize..cn.off[k + 1] as usize].iter().enumerate() {
            let mut b = b;
            while b != 0 {
                let e = e0 + 64 * w + b.trailing_zeros() as usize;
                b &= b - 1;
                dep.push((e as u32, g.rn_tph[g.adj_to[e] as usize - s]));
            }
        }
        st
    }
}

impl DestWalk<'_> {
    #[inline(always)]
    fn run<I: Id>(&mut self, next: &[I], order: &[I]) {
        let DestWalk { g, s, n, dst, bits, cn, ref mut flow, ref mut l, ref mut head, ref mut rec, ref mut use_k, ref mut dep, ref mut xf } = *self;
        let end = n + cn.len();
        for &v in order.iter().skip(1).rev() {
            let v = v.ix();
            let f = flow[v];
            if f == 0.0 { continue; }
            if v < s || v >= n {
                // a station, or a change node boarding at its station (head slot s + k)
                let st = departures(g, bits, cn, v, dep);
                let hx = if v < s { v } else { s + v - n };
                let sum: f32 = dep.iter().map(|x| x.1).sum();
                if sum == 0.0 { continue; }
                l.board[st] += f;
                for &(e, a) in dep.iter() {
                    let to = g.adj_to[e as usize] as usize;
                    let share = f * a / sum;
                    flow[to] += share;
                    l.rn_board[to - s] += share;
                }
                if head.get(hx).is_some_and(|&h| h != NONE) {
                    // the riders who arrived here leave by the departures as everyone here does
                    // (only the useful ones: about half go the wrong way for this destination)
                    use_k.clear();
                    for &(e, b) in dep.iter() {
                        if b > 0.0 {
                            use_k.push((e - g.adj_off[st], b / sum));
                        }
                    }
                    let x = &mut xf.as_mut().unwrap().1;
                    let mut i = head[hx];
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
                if to >= end { continue; }
                flow[to] += f;
                if to >= s && to < n {
                    l.seg[v - s] += f;
                    l.pax_min += (f * g.rn_ride[v - s]) as f64;
                } else {
                    // Walking transfers alight at the route node's station, not the walk's end.
                    l.alight[g.rn_station[v - s] as usize] += f;
                    l.rn_alight[v - s] += f;
                    let (st, hx) = if to < s { (to, to) } else { (cn.st[to - n] as usize, s + to - n) };
                    if st != dst {
                        if let Some((ix, _)) = xf.as_ref() {
                            let e = (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).find(|&e| g.adj_to[e] as usize == st).unwrap();
                            rec.push((ix.at[e], f, head[hx]));
                            head[hx] = (rec.len() - 1) as u32;
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

/// Penalty per segment: 1 + alpha x(1+x)/2, x = max(load - start, 0) / (crush - start), per hour
/// at the peak hour, start the larger of the seats and `crowd_start` x crush (T-710: riders mind
/// crowding only near full). Never saturates (memory note on crowding).
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
        let start = seats.max(p.crowd_start * crush);
        let hour = loads.seg[r] / p.period_hours * p.peak_hour_factor;
        let x = ((hour - start) / (crush - start)).max(0.0);
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
            // a stop with no train due (T-842) has none in `rn_tph`; the line's trains still run
            let tph = g.rn_tph[run.clone()].iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
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
    let mut left = if total.len() == 3 * nr && p.left_behind { left_behind(g, net, p, &total[nr..2 * nr], &total[2 * nr..]).0 } else { vec![0.0; nr] };
    for (l, s) in left.iter_mut().zip(&g.rn_span) {
        *l += s;
    }
    Some((fac, left))
}

/// A period's crowding rounds stop once one moves its summed segment loads by under this share,
/// rider weighted (T-434); the driver caps them at three.
pub const SETTLED: f64 = 0.01;

/// Whether a period's next crowding round is skipped (T-434): `seg`, the segment loads it would be
/// priced on, is within `SETTLED` of those the round before was priced on, kept in `seen`. The first
/// crowding round always runs, as `seen` starts empty. Each population of a cluster gets the
/// cluster's loads on its own route nodes (T-756), so it stops when those settle.
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
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![], span: vec![],
        };
        for (l, stops) in lines.iter().enumerate() {
            net.line_stops.extend_from_slice(stops);
            net.line_off.push(net.line_stops.len() as u32);
            net.hop_min.extend((0..stops.len()).map(|k| if k + 1 < stops.len() { ride[l] } else { 0.0 }));
        }
        net
    }

    fn common_test_params() -> Params {
        Params { xfer_walk_m: 0.0, ..Params::default() }
    }

    /// T-883: a fare capped beyond 30 km lowers only the longer pairs' costs, by the rate on the km
    /// past 30, with two lines pooled at the shared stations.
    #[test]
    fn a_fare_cap_changes_long_trips_only() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1, 2, 3], &[0, 1, 2]], &[6.0, 12.0], &[2.0, 2.0]);
        let (base, rate) = (4.5f32, 0.3f32);
        let graph = |knots: &[(f32, f32)]| {
            let mut g = build_graph(&net, &p);
            g.rn_fare = vec![20.0 * rate; g.rn_ride.len()];
            let ones = vec![1.0; g.rn_ride.len()];
            set_crowding(&mut g, &ones);
            if knots.len() >= 2 {
                g.fare_curve = FareCurve { km: knots.iter().map(|k| k.0).collect(), min: knots.iter().map(|k| k.1).collect(), base, rate };
                g.rn_km = vec![20.0; g.rn_ride.len()];
            }
            g
        };
        let line = skims(&graph(&[]), &p).r;
        let straight = skims(&graph(&[(1.0, base + rate), (100.0, base + 100.0 * rate)]), &p).r;
        let cap = base + 30.0 * rate;
        let capped = skims(&graph(&[(1.0, base + rate), (30.0, cap), (100.0, cap)]), &p).r;
        for (a, b) in line.iter().zip(&straight) {
            assert!(a == b || (a - b).abs() < 1e-3, "{a} {b}");
        }
        let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
        assert!((at(&capped, 0, 1) - at(&line, 0, 1)).abs() < 1e-3);
        assert!((at(&line, 0, 2) - at(&capped, 0, 2) - rate * 10.0).abs() < 1e-3);
        assert!((at(&line, 0, 3) - at(&capped, 0, 3) - rate * 30.0).abs() < 1e-3);
        assert!((at(&line, 3, 0) - at(&capped, 3, 0) - rate * 30.0).abs() < 1e-3);
    }

    /// T-798: a circle 0-1-2-3-4-5-0, 12 tph, 2 minutes a stop and half a minute standing at 0.
    /// Riders take the short way round through 0 without changing; a line that is not a circle
    /// sends them the long way.
    #[test]
    fn a_circle_rides_the_short_way_through_its_first_stop() {
        let p = common_test_params();
        let ring = |wrap: bool| {
            let st: Vec<(f32, f32)> = (0..6).map(|i| { let a = i as f32 * std::f32::consts::PI / 3.0; (3000.0 * a.cos(), 3000.0 * a.sin()) }).collect();
            let stops = [0u32, 1, 2, 3, 4, 5, 0];
            let ride: Vec<f32> = (0..7).map(|k| if k < 6 { 2.0 } else if wrap { 0.5 } else { 0.0 }).collect();
            let back: Vec<f32> = (0..7).map(|k| if k > 0 { 2.0 } else if wrap { 0.5 } else { 0.0 }).collect();
            Network {
                st_x: st.iter().map(|s| s.0).collect(), st_y: st.iter().map(|s| s.1).collect(),
                line_name: vec![String::new()], line_off: vec![0, 7], line_stops: stops.to_vec(),
                hop_min: ride, hop_back: back, line_speed_kmh: vec![0.0],
                line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_crush: vec![200.0],
                st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![wrap], span: vec![],
            }
        };
        let g = build_graph(&ring(true), &p);
        let s = g.n_st;
        assert_eq!(g.n - s, 14, "two route nodes a stop a direction, as for any line");
        let sk = skims(&g, &p);
        // 5 to 1: half a headway's wait at 5, two stops and the stand at 0
        let short = 2.5 + 2.0 + 0.5 + 2.0;
        assert!((sk.r[5 * s + 1] - short).abs() < 1e-4, "{} against {short}", sk.r[5 * s + 1]);
        assert!((sk.r[1 * s + 5] - short).abs() < 1e-4, "the other way: {}", sk.r[1 * s + 5]);
        let mut od = vec![0.0; s * s];
        od[5 * s + 1] = 100.0;
        od[1 * s + 5] = 40.0;
        let loads = assign(&g, &sk, &od);
        // route nodes: run 0 = 0..7 (stations 0,1,2,3,4,5,0), run 1 = 7..14 (0,5,4,3,2,1,0)
        assert_eq!(loads.seg[6], 100.0, "riders stay aboard through 0 on run 0");
        assert_eq!(loads.seg[13], 40.0, "and on run 1");
        assert_eq!(loads.rn_board.iter().sum::<f32>(), 140.0, "one boarding each");
        assert_eq!((loads.rn_board[5], loads.rn_alight[1]), (100.0, 100.0));
        assert_eq!((loads.rn_board[12], loads.rn_alight[8]), (40.0, 40.0));
        // nobody boards at a wrapped last stop: the first stop's route node has the same trains
        assert_eq!(loads.rn_board[6] + loads.rn_board[13], 0.0);
        // without the wrap the trip is the long way: four stops
        let plain = skims(&build_graph(&ring(false), &p), &p);
        assert!((plain.r[5 * s + 1] - (2.5 + 8.0)).abs() < 1e-4, "{}", plain.r[5 * s + 1]);
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

    /// T-710: the station choice's scale is 20 over the trip's rough length (25 + 3 x km
    /// perceived minutes), held from 60 minutes on, by four distance bands; off, one band at
    /// `station_theta` with no offset.
    #[test]
    fn station_choice_scale_follows_the_trip_length() {
        let p = Params::default();
        let km = |d: f32| p.band_of_d2(d * d * 1e6);
        assert_eq!((km(0.0), km(2.9), km(3.0), km(6.0), km(11.9), km(12.0), km(80.0)), (0, 0, 1, 1, 2, 3, 3));
        let (th0, off0) = p.band(0);
        assert!((th0 - 20.0 / 29.5).abs() < 1e-6 && (off0 - 60.0 / th0).abs() < 1e-4);
        assert!((p.band(1).0 - 0.5).abs() < 1e-6);
        assert!((p.band(2).0 - 20.0 / 53.5).abs() < 1e-6);
        // past 60 minutes the scale softens as sqrt(60 / T): the band from 12 km is 79 minutes
        assert!((p.band(3).0 - 20.0 / 60.0 * (60.0f32 / 79.0).sqrt()).abs() < 1e-6);
        assert_eq!(p.soften(0.5, 50.0), 0.5);
        assert!((p.soften(1.0 / 3.0, 240.0) - 1.0 / 6.0).abs() < 1e-6);
        // `rel_soft` 0: past 60 minutes a minute counts the same however long the trip
        let hard = Params { rel_soft: 0.0, ..Params::default() };
        assert!((hard.band(3).0 - 20.0 / 60.0).abs() < 1e-6 && hard.soften(0.5, 240.0) == 0.5);
        let off = Params { rel_gamma: 0.0, ..Params::default() };
        assert_eq!((off.n_bands(), off.band_of_d2(1e12), off.band(0)), (1, 0, (0.2, 0.0)));
    }

    /// T-710: a train costs nothing extra until 70% of crush (or its seats, if more), then
    /// 1 + 2 x(1+x)/2 with x from there to crush: 3 at crush.
    #[test]
    fn crowding_starts_near_full() {
        let p = common_test_params();
        let net = common_test_net(&[&[0, 1]], &[6.0], &[2.0]);
        let g = build_graph(&net, &p);
        // an hour's load at the busiest hour from the period's riders
        let at = |hour: f32, p: &Params| {
            let mut l = Loads::zero(g.n_st, g.rn_line.len());
            l.seg[0] = hour * p.period_hours / p.peak_hour_factor;
            crowd_factors(&g, &net, &l, p).0[0]
        };
        // 6 trains an hour: 600 seats, 1,200 at crush
        assert_eq!(at(720.0, &p), 1.0);
        assert_eq!(at(840.0, &p), 1.0);
        assert!((at(1200.0, &p) - 3.0).abs() < 1e-4);
        let seats = Params { crowd_start: 0.0, ..common_test_params() };
        assert!((at(720.0, &seats) - 1.24).abs() < 1e-4);
    }

    /// T-710: a band's offset keeps a long trip's rail inside f32 with a sharp scale: a pair whose
    /// station-to-station cost is 300 perceived minutes is still priced, at its own cost.
    #[test]
    fn a_long_trip_keeps_its_rail_cost_in_a_sharp_band() {
        let p = Params::default();
        let (th, off) = p.band(3);
        // 300 minutes at this scale is e^-100 without the offset: clamped, as if 240
        assert_eq!(fast_exp(-th * 300.0), fast_exp(-80.0));
        let e = fast_exp(-th * (300.0 - off));
        assert!(e > 1e-30);
        assert!((-fast_ln(e) / th + off - 300.0).abs() < 0.05);
        // and a short one in the same band does not overflow
        assert!(fast_exp(-th * (0.0 - off)).is_finite());
    }

    /// Setting the worldwide rail constant by name keeps the country's part on top.
    #[test]
    fn country_rail_constant() {
        let mut p = Params::default();
        assert_eq!(super::super::api::country_asc_rail("JP"), 0.0);
        p.set_extra("country_asc_rail", 0.5).unwrap();
        assert_eq!(p.asc_rail, super::super::api::ASC_RAIL + 0.5);
        p.set_extra("asc_rail", 0.0).unwrap();
        assert_eq!(p.asc_rail, 0.5);
        p.set_extra("country_asc_rail", 0.0).unwrap();
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
            pop: vec![0.0; n], jobs: (0..n).map(|j| dens[j] * km2[j]).collect(), ncell, size: vec![],
        };
        let mut p = Params::default();
        let near = |a: f32, b: f32| (a - b).abs() < 0.05;
        let (f0, paid0, free0) = p.park_parts(&z, 0);
        assert!(f0 == p.park_min + p.car_own_min() && paid0 == 0.0 && near(free0, 0.8));
        let (_, paid1, _) = p.park_parts(&z, 1);
        assert!(near(paid1, 15.0 * (10000.0f32 / 3000.0).ln()));
        assert_eq!(p.park_parts(&z, 2).1, 50.0);
        // a quarter of the square on land: the same jobs per km² of land, the same price
        assert!(near(p.park_parts(&z, 3).1, paid1));
        assert!(near(p.park_parts(&z, 4).2, 0.5));
        p.cordon = Some(std::sync::Arc::new(vec![0.0, 0.0, 13.5, 0.0, 0.0]));
        assert_eq!(p.park_parts(&z, 2).0, 13.5 + p.park_min + p.car_own_min());
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

    /// T-621: ownership is dollars a trip in the destination's fixed part plus dollars a km of
    /// road in the drive, FARE_MIN_PER_USD minutes a dollar; at 0 both change nothing to the bit.
    #[test]
    fn car_ownership_adds_dollars_per_trip_and_km() {
        let z = Zones {
            zone_m: 2000.0, n: 1, cell_zone: vec![], gx: vec![0], gy: vec![0], region: vec![0], w: 1, h: 1, x0: 0.0, y0: 0.0,
            pop: vec![0.0], jobs: vec![0.0], ncell: vec![38], size: vec![],
        };
        let mut p = Params::default();
        p.set_extra("car_own_usd_trip", 0.0).unwrap();
        p.set_extra("car_own_usd_km", 0.0).unwrap();
        let base = p.park_parts(&z, 0);
        assert_eq!(p.car_drive(17.3, 12.1).to_bits(), (17.3f32 * p.w_drive).to_bits());
        p.set_extra("car_own_usd_trip", 1.5).unwrap();
        p.set_extra("car_own_usd_km", 0.1).unwrap();
        let usd = crate::demand::api::FARE_MIN_PER_USD;
        assert!((p.car_own_min() - 1.5 * usd).abs() < 1e-5);
        let with = p.park_parts(&z, 0);
        assert!((with.0 - base.0 - 1.5 * usd).abs() < 1e-5 && with.1 == base.1 && with.2 == base.2);
        // a 15 km road: $1.50 + $1.50 = 9 perceived minutes on top of the drive's time
        assert!((p.car_drive(20.0, 15.0) - 20.0 * p.w_drive - 1.5 * usd).abs() < 1e-4);
        assert!((p.car_own_min() + 15.0 * p.car_own_km_min() - 9.0).abs() < 1e-4);
    }

    /// T-854: home parking is 0.4 of the paid-space rule on residents; the curve's floor and ceiling
    /// hold rail's share of the trips not walking between 5% and 95% while rail and the car cost
    /// about the same, and fade to nothing at 3 times either way, walking keeping its share.
    #[test]
    fn home_parking_and_the_curves_bounds() {
        let p = Params::default();
        assert_eq!(p.home_park_at(3000.0), 0.0);
        assert!((p.home_park_at(40_000.0) - 0.4 * 15.0 * (40.0f32 / 3.0).ln()).abs() < 1e-4);
        let rest = |r: f32, e: f32| r / (1.0 - e * r);
        // rail as costly as the car: within 5% and 95%, walking (e r) kept
        for (r, e) in [(0.0, 0.0), (1.0, 0.0), (0.001, 2.0), (0.3, 1.0)] {
            let b = p.rail_bounds(r, e, 30.0, 30.0);
            let rb = b / (1.0 - e * r);
            assert!(rb >= 0.05 - 1e-6 && rb <= 0.95 + 1e-6, "{r} {e}: {rb}");
            assert!((rb - (0.05 + 0.9 * rest(r, e))).abs() < 1e-5);
        }
        // twice the car: half the floor; three times: none; and the mirror for the car
        assert!((p.rail_bounds(0.0, 0.0, 60.0, 30.0) - 0.025).abs() < 1e-6);
        assert_eq!(p.rail_bounds(0.0, 0.0, 90.0, 30.0), 0.0);
        assert!((p.rail_bounds(1.0, 0.0, 10.0, 30.0) - 1.0).abs() < 1e-6);
        let off = Params { rail_floor: 0.0, ..Params::default() };
        assert_eq!(off.rail_bounds(0.37, 0.5, 60.0, 30.0), 0.37);
    }

    /// T-636: the walking constant is `walk_bonus_min` perceived minutes whatever the scale, and a
    /// walk costs `walk_trip_w` a minute at every length.
    #[test]
    fn walk_constant_is_in_minutes_and_the_walk_cost_is_flat() {
        let mut p = Params::default();
        assert!((p.asc_walk() - 10.0 * p.beta).abs() < 1e-6);
        p.set_extra("beta", 0.05).unwrap();
        assert!((p.asc_walk() - 0.5).abs() < 1e-6);
        p.set_extra("walk_bonus_min", 20.0).unwrap();
        assert!((p.asc_walk() - 1.0).abs() < 1e-6);
        // 1 km straight is 1.3 km at 6 km/h: 13 minutes, and 2 km twice that
        let (a, b) = (p.walk_trip_cost(1.0), p.walk_trip_cost(2.0));
        assert!((a - 13.0 * p.walk_trip_w).abs() < 1e-3 && (b - 2.0 * a).abs() < 1e-3);
        assert_eq!(p.walk_trip_cost(WALK_REACH_KM + 0.1), NO_WALK);
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

    /// T-552, T-575: riders never change onto the line they left, so where full trains leave them
    /// behind they wait for the direct train rather than ride the other way to board upstream.
    /// Line 0-1-2-3, 12 tph, 2 minutes a stop, riders 1 -> 3; the southbound at 1 leaves riders
    /// behind for 60 minutes.
    #[test]
    fn riders_wait_for_a_full_line_rather_than_ride_back() {
        let p = common_test_params();
        {
            let net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[2.0]);
            let mut g = build_graph(&net, &p);
            let (south, north) = (1, 6);
            g.rn_left[south] = 60.0;
            let sk = skims(&g, &p);
            let s = g.n_st;
            let tph = sk.board_tph(&g, sk.slot[3] as usize);
            // the direct train with its wait: 2.5 a headway, 60 for the full trains, 4 riding
            let direct = 2.5 + 60.0 + 4.0;
            assert!((sk.r[1 * s + 3] - direct).abs() < 1e-3, "{} against {direct}", sk.r[1 * s + 3]);
            assert_eq!((tph[south], tph[north]), (12.0, 0.0));
            let mut od = vec![0.0; s * s];
            od[1 * s + 3] = 100.0;
            let loads = assign(&g, &sk, &od);
            assert_eq!(loads.rn_board[north], 0.0, "nobody rides back");
            assert_eq!(loads.rn_board[south], 100.0);
            assert_eq!(loads.board.iter().sum::<f32>(), 100.0, "one boarding each");
            // the trip explorer's route: one ride, its wait including the full trains'
            let r = super::super::trip::routes(&g, &sk, &p, 1, 3, 0.005);
            assert_eq!(r.len(), 1);
            let legs = &r[0].1;
            assert_eq!(legs.len(), 1);
            assert!((legs[0].wait_min - 62.5).abs() < 1e-4, "{}", legs[0].wait_min);
            // a shorter wait: still the direct train, its cost with the shorter wait
            g.rn_left[south] = 2.0;
            let sk = skims(&g, &p);
            let tph = sk.board_tph(&g, sk.slot[3] as usize);
            assert!((sk.r[1 * s + 3] - (2.5 + 2.0 + 4.0)).abs() < 1e-3, "{}", sk.r[1 * s + 3]);
            assert_eq!((tph[south], tph[north]), (12.0, 0.0));
        }
    }

    /// T-575: a rider leaving a line at a station whose strategy boards that line's other direction
    /// changes there onto the station's other lines only. Line L 0-1-2 (12 tph, half a minute a
    /// stop), C 0-3 (4 tph, 2 minutes), M 2-3 (4 tph, 3.5 minutes); riders 1 -> 3. At 2 the
    /// strategy pools M with L back towards C; a rider off L at 2 takes M alone. Changes cost no
    /// fixed minutes here, or the change onto C would keep L back out of the pool.
    #[test]
    fn a_change_off_a_line_takes_the_stations_other_lines() {
        let p = Params { transfer_min: 0.0, ..common_test_params() };
        let net = common_test_net(&[&[0, 1, 2], &[0, 3], &[2, 3]], &[12.0, 4.0, 4.0], &[0.5, 2.0, 3.5]);
        let g = build_graph(&net, &p);
        let s = g.n_st;
        let sk = skims(&g, &p);
        let slot = sk.slot[3] as usize;
        let tph = sk.board_tph(&g, slot);
        let (l_up, l_back_at_2, m_at_2) = (1, 3, 10);
        assert_eq!((tph[l_back_at_2], tph[m_at_2]), (12.0, 4.0), "2 pools M with L back");
        // from 1: L back to C (10 from boarding) and L on to M at 2 (0.5 + 11), both 12 tph
        let want = (10.0 + 11.5) / 2.0 + 1.25;
        assert!((sk.r[1 * s + 3] - want).abs() < 1e-4, "{} against {want}", sk.r[1 * s + 3]);
        assert_eq!(tph[l_up], 12.0);
        let mut od = vec![0.0; s * s];
        od[1 * s + 3] = 100.0;
        let loads = assign(&g, &sk, &od);
        assert_eq!(loads.rn_board[m_at_2], 50.0);
        assert_eq!(loads.rn_board[l_back_at_2], 0.0, "nobody off L at 2 rides L back");
        assert_eq!(loads.board[2], 50.0);
        assert_eq!(loads.alight[3], 100.0);
        let r = super::super::trip::routes(&g, &sk, &p, 1, 3, 0.005);
        assert_eq!(r.len(), 2);
        for (share, legs) in &r {
            assert!((share - 0.5).abs() < 1e-6 && legs.len() == 2);
        }
        let via_m = r.iter().find(|x| x.1[1].st_from == 2).unwrap();
        assert!((via_m.1[1].wait_min - 7.5).abs() < 1e-4, "M's own headway: {}", via_m.1[1].wait_min);
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

    /// T-717: a change costs the fixed minutes, its wait and the walk between its two platforms.
    #[test]
    fn a_change_costs_fixed_minutes_and_the_walk_between_platforms() {
        let p = common_test_params();
        assert_eq!(p.transfer_min, 2.0);
        // line A 0-1, B 1-2, 12 tph and 2 minutes each; riders 0 -> 2 change at 1
        let mut net = common_test_net(&[&[0, 1], &[1, 2]], &[12.0, 12.0], &[2.0, 2.0]);
        let s = net.n_stations();
        let one_platform = skims(&build_graph(&net, &p), &p).r[2];
        assert!((one_platform - (2.5 + 2.0 + 2.0 + 2.5 + 2.0)).abs() < 1e-4, "{one_platform}");
        // station 1 is two platforms 80 m apart, A's at y 0 and B's at y 80, the station between
        net.st_y[1] = 40.0;
        net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
        net.stop_y = vec![0.0, 0.0, 80.0, 0.0];
        let g = build_graph(&net, &p);
        assert!(g.st_spread[1] && !g.st_spread[0]);
        let sk = skims(&g, &p);
        let walk = p.xfer_min(80.0) * p.w_xfer;
        assert!((sk.r[2] - (one_platform + walk)).abs() < 1e-4, "{} against {}", sk.r[2], one_platform + walk);
        assert!((sk.r[s + 2] - 4.5).abs() < 1e-4, "a trip starting at the station walks no platform: {}", sk.r[s + 2]);
        // the trip explorer: ride, the walk between the platforms, ride with the change's 2 minutes
        let r = super::super::trip::routes(&g, &sk, &p, 0, 2, 0.005);
        let legs = &r[0].1;
        assert_eq!(legs.iter().map(|l| l.kind).collect::<Vec<_>>(), vec![1, 2, 1]);
        assert!((legs[1].perc - walk).abs() < 1e-5 && legs[1].st_from == 1 && legs[1].st_to == 1);
        assert_eq!((legs[0].change, legs[2].change), (0.0, 2.0));
        let total: f32 = legs.iter().map(|l| l.perc + l.wait_perc + l.change).sum();
        assert!((total - sk.r[2]).abs() < 1e-4, "the legs add up to the strategy: {total}");
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
        let mo = mode_choice(&z, &gr, &sz, &station_exp(&sk, &p), g.n_st, &p, &NoRail::new(&z, &p));
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
        let mo = mode_choice(&z, &gr, &sz, &er, g.n_st, &p, &NoRail::new(&z, &p));
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
        // T-558: rail takes walkers, the same by home and by work subzone, and never more than
        // walk with no rail
        let (wb, wt) = (sum(&mo.walk_lost_by_sub), sum(&mo.walk_lost_to_sub));
        assert!(mo.rail_walk > 0.0 && (wb / mo.rail_walk - 1.0).abs() < 1e-3 && (wt / mo.rail_walk - 1.0).abs() < 1e-3, "{wb} {wt} {}", mo.rail_walk);
        let nr = NoRail::new(&z, &p);
        let walk0: f64 = gr.no_rail_by_zone(&z, &p, &nr)[0].iter().sum::<f64>() * p.period_share as f64;
        assert!(mo.rail_walk < walk0, "{} of {walk0}", mo.rail_walk);
        let boards: f64 = l.board.iter().map(|&v| v as f64).sum();
        assert!(boards >= od_sum * 0.999);
        // T-874: small pairs dealt in chunks visit fewer pairs, keep the trips offered rail and
        // rail itself close, and deal the same chunks every pass
        let every = Params { commute_chunk: 0.0, ..p.clone() };
        let big = Params { commute_chunk: 50.0, ..p.clone() };
        let mo0 = mode_choice(&z, &gr, &sz, &er, g.n_st, &every, &nr);
        let (mo1, mo2) = (mode_choice(&z, &gr, &sz, &er, g.n_st, &big, &nr), mode_choice(&z, &gr, &sz, &er, g.n_st, &big, &nr));
        assert!(mo1.pairs < mo0.pairs * 9 / 10, "pairs {} of {}", mo1.pairs, mo0.pairs);
        assert!((mo1.trips_access / mo0.trips_access - 1.0).abs() < 0.01, "offered {} vs {}", mo1.trips_access, mo0.trips_access);
        assert!((mo1.rail / mo0.rail - 1.0).abs() < 0.02, "rail {} vs {}", mo1.rail, mo0.rail);
        assert!(mo1.od == mo2.od && mo1.rail_by_sub == mo2.rail_by_sub);
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

    /// T-852: the synthetic city's inner 15 km, two regions (border at x = 300 m), with every
    /// aligned block of zones whose centres are over 6 km out made one zone.
    fn mixed_city() -> (crate::demand::pack::City, Zones, Zones) {
        let full = synth::synth_city(1);
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 15_000.0).collect();
        let city = crate::demand::pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            region: keep.iter().map(|&c| (full.x[c] > 300.0) as u8).collect(),
            gravity: None,
        };
        let z0 = build_zones(&city, 2000.0);
        let out = |i: usize| (z0.x0 + (z0.gx[i] as f32 + 0.5) * 2000.0).hypot(z0.y0 + (z0.gy[i] as f32 + 0.5) * 2000.0) > 6000.0;
        let ends: Vec<f64> = (0..z0.n).map(|i| if out(i) { 10.0 } else { 1e6 }).collect();
        let z = coarsen(&z0, &ends, 1000.0);
        (city, z0, z)
    }

    /// T-852: zones of two sizes. Every pair's table entry is the mean of the decay between their
    /// squares and its distance that of their centres; the gravity balances to the same trips as
    /// the squares'; near pairs, a square's zone and the pack round trip hold.
    #[test]
    fn mixed_zones() {
        let (mut city, z0, z) = mixed_city();
        let blocks = z.size.iter().filter(|&&s| s == 2).count();
        assert!(blocks > 20 && z.n < z0.n - blocks && z.w % 2 == 0, "{blocks} blocks, {} of {} zones", z.n, z0.n);
        assert!((z.pop.iter().sum::<f64>() / z0.pop.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for c in 0..city.len() {
            let i = z.cell_zone[c] as usize;
            let (gx, gy) = (((city.x[c] - z.x0) / 2000.0) as i32, ((city.y[c] - z.y0) / 2000.0) as i32);
            assert!(gx - z.gx[i] < z.side(i) && gx >= z.gx[i] && gy - z.gy[i] < z.side(i) && gy >= z.gy[i] && z.region[i] == city.region[c]);
        }
        let p = Params::default();
        let g = gravity(&z, &p, 300, 1e-7);
        let g0 = gravity(&z0, &p, 300, 1e-7);
        assert!((g.total / g0.total - 1.0).abs() < 1e-5 && (g.mean_km / g0.mean_km - 1.0).abs() < 0.04, "{} km against {}", g.mean_km, g0.mean_km);
        let (_, dtab) = decay_tables(&z, 1.0, 0.0);
        let f = |d: f32| (-d / p.decay_km).exp() * d.powf(-p.decay_pow);
        let sq = |i: usize| -> Vec<(i32, i32)> {
            let (s, gx, gy) = (z.side(i), z.gx[i], z.gy[i]);
            (0..s).flat_map(|a| (0..s).map(move |b| (gx + a, gy + b))).collect()
        };
        for i in 0..z.n {
            for j in 0..z.n {
                let k = g.base(&z, i) + g.key(&z, j);
                assert!(k >= 0 && (k as usize) < g.ftab.len());
                let (si, sj) = (sq(i), sq(j));
                let mean = si.iter().flat_map(|a| sj.iter().map(move |b| (a, b))).map(|(a, b)| {
                    let d = if a == b { 1.04 } else { 2.0 * (((a.0 - b.0).pow(2) + (a.1 - b.1).pow(2)) as f32).sqrt() };
                    f(d)
                }).sum::<f32>() / (si.len() * sj.len()) as f32;
                let got = g.ftab[k as usize];
                assert!((got / mean - 1.0).abs() < 1e-5, "{i} {j}: {got} vs {mean}");
                let d = (z.d2_m(i, j).sqrt() / 1000.0).max(if z.d2_m(i, j) == 0.0 { 1.04 * z.side(i) as f32 } else { 0.0 });
                assert!((dtab[k as usize] - d).abs() < 1e-4 && (g.dist_km(k as usize) - d).abs() < 1e-4, "{i} {j}: {} km vs {d}", dtab[k as usize]);
            }
            // both margins
            let r: f64 = (0..z.n).map(|j| g.t(&z, i, j)).sum();
            let c: f64 = (0..z.n).map(|j| g.t(&z, j, i)).sum();
            assert!((c - z.jobs[i]).abs() <= 1e-4 * z.jobs[i] + 1e-6 && r >= 0.0, "column {i}");
        }
        // near pairs by squares, and the walk between cells for each of them
        let mut gw = g.clone();
        gw.set_walk(&z, &p);
        gw.set_walk_fine(&city, &z, &p, &NoRail::new(&z, &p));
        let byq = z.by_square();
        for i in 0..z.n {
            let mut got = vec![];
            z.near(&byq, i, 2, |j| got.push(j));
            let (si, mut want) = (sq(i), vec![]);
            for j in 0..z.n {
                if sq(j).iter().any(|b| si.iter().any(|a| (a.0 - b.0).abs() <= 2 && (a.1 - b.1).abs() <= 2)) {
                    want.push(j);
                }
            }
            got.sort();
            assert_eq!(got, want, "near {i}");
            let fine = &gw.wf_j[gw.wf_off[i] as usize..gw.wf_off[i + 1] as usize];
            for &j in fine {
                let k = (gw.base(&z, i) + gw.key(&z, j as usize)) as usize;
                assert!(gw.fine_at(i, j as usize, k).is_some(), "fine {i} {j}");
            }
        }
        // a square's zone (one region: the squares west of the border)
        for c in (0..city.len()).filter(|&c| city.region[c] == 0) {
            let (gx, gy) = (((city.x[c] - z.x0) / 2000.0) as i32, ((city.y[c] - z.y0) / 2000.0) as i32);
            let at = z.zone_at(&byq, gx, gy).unwrap();
            assert!(at == z.cell_zone[c] as usize || z.region[at] != 0);
        }
        // shipped and rebuilt
        city.gravity = Some(to_pack_gravity(&z, &g, &p));
        let (h, b) = crate::demand::pack::write(&city);
        assert!(h.contains("\"zone_size\""));
        let back = crate::demand::pack::parse(&h, &b).unwrap();
        let (zb, gb, from_pack) = city_gravity(&back, &p);
        assert!(from_pack && zb.size == z.size && zb.w == z.w);
        for (i, j) in [(0, 1), (3, z.n - 1), (z.n / 2, z.n / 3), (z.n - 1, z.n - 1)] {
            assert!((gb.t(&zb, i, j) / g.t(&z, i, j).max(1e-30) - 1.0).abs() < 1e-5 || g.t(&z, i, j) < 1e-12);
        }
    }

    /// T-852: commutes on the synthetic network with the outer zones in blocks keep their rail
    /// and their trips offered rail close to the squares'.
    #[test]
    fn mixed_zones_ride_like_squares() {
        let (city, z0, z) = mixed_city();
        let net = network::synth_network();
        let p = Params::default();
        let run = |z: &Zones| {
            let mut gr = gravity(z, &p, 200, 1e-5);
            gr.set_walk(z, &p);
            let nr = NoRail::new(z, &p);
            gr.set_walk_fine(&city, z, &p, &nr);
            let acc = build_access(&city, &net, &p);
            let sz = build_subzones(&city, z, &acc, &net, &p);
            let g = build_graph(&net, &p);
            let sk = skims(&g, &p);
            let er = station_exp(&sk, &p);
            let mo = mode_choice(z, &gr, &sz, &er, g.n_st, &p, &nr);
            let walk: f64 = gr.no_rail_by_zone(z, &p, &nr)[0].iter().sum();
            (mo.rail, mo.trips_access, walk)
        };
        let (r0, a0, w0) = run(&z0);
        let (r, a, w) = run(&z);
        assert!(r0 > 0.0 && (r / r0 - 1.0).abs() < 0.05 && (a / a0 - 1.0).abs() < 0.05 && (w / w0 - 1.0).abs() < 0.05, "rail {r} vs {r0}, offered {a} vs {a0}, walk {w} vs {w0}");
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
