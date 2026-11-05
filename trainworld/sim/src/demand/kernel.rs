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
use crate::track::upgrades;
use std::collections::HashMap;

/// Stations kept per cell and per subzone.
pub const K: usize = 6;
/// Dummy slots after each row of the commute pair loop, one per lane and egress step (T-036).
/// A subzone's walk entries at most: K train stations and BUS_SLOTS bus stops (T-1464).
pub const KE: usize = K + BUS_SLOTS;
const PADS: usize = 8 * KE;
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
    /// The decay is held at its value here inside this many km (T-1091, `Curve`); 0 none. A
    /// pack's own gravity carries its value (`PackGravity::flat_km`), set by `flat_near_km`.
    pub decay_flat_km: f32,
    /// The decay's factor between home and work zones in different first-level regions (T-162:
    /// US states). Only for a gravity solved here or by `pack_gravity`: a pack's own gravity
    /// carries its factor (`PackGravity::border_factor`), so each country sets its own.
    pub border_factor: f32,
    pub w_walk: f32,
    /// A fixed cost per change of train, perceived minutes (Anita, T-717): a change costs this, its
    /// wait and the walk between the two platforms.
    pub transfer_min: f32,
    /// The player's transfer fare (T-1801), perceived minutes, paid on each boarding after a
    /// trip's first; 0 unless set.
    pub xfer_fare: f32,
    pub fare_min: f32,
    /// Perceived minutes a fare dollar costs the riders of this population (T-1277: by their
    /// country's income, set by the app; the car's money stays at `api::FARE_MIN_PER_USD`).
    pub fare_w: f32,
    /// Logit scale per perceived minute, and the rail propensity: the worldwide value plus
    /// `country_asc_rail`, which `set_extra` keeps in step (setting `asc_rail` by name sets the
    /// worldwide part).
    pub beta: f32,
    pub asc_rail: f32,
    pub country_asc_rail: f32,
    /// Cost damping (T-495, `mode_beta`): past this many perceived minutes of the car's trip the
    /// logit's scale falls as the square root of the car's minutes (walking, and the regional
    /// layer; rail against the car is `ratio_share`'s).
    pub damp_min: f32,
    /// Rail against the car (T-918, `ratio_share`): rail's share of the two is 1 / (1 + q^3), q
    /// rail's cost over the car's, fading to nothing from `ratio_fade_lo` to `ratio_fade_hi` times
    /// the car's cost, and the car's the same way where it costs that many times rail's.
    pub ratio_fade_lo: f32,
    pub ratio_fade_hi: f32,
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
    /// times the country's income factor (`park_income`, T-2278), d jobs per km² of land in the
    /// zone; a share 1 / (1 + d / `park_free_d`) parks free.
    pub park_min: f32,
    pub park_paid_min: f32,
    pub park_paid_d0: f32,
    pub park_paid_cap_min: f32,
    pub park_free_d: f32,
    /// T-2278: the paid parts' factor for the pack country's income (`park_income_of`); 1 rich.
    pub park_income: f32,
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
    /// The crowding penalty starts at this share of the rated load, or at the seats if they are
    /// more (T-710, T-2186, `crowd_factors`).
    pub crowd_start: f32,
    /// Zone pairs carrying fewer trips than this are skipped in mode choice.
    pub prune_trips: f32,
    /// T-874: an origin zone's pairs carrying fewer trips than this in the period are dealt
    /// those trips in chunks of about this size, per block of `CHUNK_BLOCK` destination zones
    /// (`mode_choice`); 0 solves every pair.
    pub commute_chunk: f32,
    /// Mixed into where the chunks start (0 in the game; the level test's other draws, T-1222).
    pub chunk_salt: u32,
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
    /// Anita's walking rule (T-1059, in real minutes since T-2265): a station pair is no rail
    /// option for a trip within `WALK_REACH_KM` when its real minutes (the walk or drive to its
    /// first station, `Skims::walk` between, the walk from its last; no fare, crowding or fixed
    /// change minutes) are more than walking the whole way's, around water where the pack routes
    /// walks. All in metres walked at walking's pace (`walk_eq_m`). 0 turns it off.
    pub walk_rule: f32,
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
    /// The pack's land parts (T-2214, `Parts`): no car and no walk between two, access only
    /// within one. None: one part.
    pub parts: Option<std::sync::Arc<Parts>>,
    /// The routed factor's multiplier (T-275): shortest-distance routes over the pack's roads
    /// are on average shorter than `road_detour`, which the density speeds were fitted with (to
    /// ACS car commute times on New York, T-090), and than the faster roads drivers take. 1.3 /
    /// 1.210, New York's commute-weighted routed factor, keeps New York's drives at the fitted
    /// level; water, bays and sparse roads still lengthen a pair against the others.
    pub road_routed_scale: f32,
    /// The ride to a station (car, bus, bike or a lift as one mode, T-371), at the home end of
    /// commutes whose zones' centres are at least `ride_min_km` apart: `p_ride` + `ride_drive_f` x
    /// `w_drive` x the drive from the home zone's centre + at the station the cheaper of parking
    /// there (as at a destination) and a drop-off, `d_drop`. Per home zone, the cheapest station of
    /// each of the `ride_lines` cheapest lines within `ride_max_m` (straight line); 0 lines turns
    /// it off. `ride_drive_f` 2 (T-916): long drives to a station became rare, short ones stayed.
    pub p_ride: f32,
    pub ride_drive_f: f32,
    pub d_drop: f32,
    pub ride_min_km: f32,
    pub ride_max_m: f32,
    pub ride_lines: usize,
    /// The everyday layer's outings per resident (per enrolled student for school and
    /// university) and destination decay lengths in metres, by `everyday::KINDS` (T-509 stage
    /// 4). Read when `EverydayApi::beside` opens; the pack's `everyday` block sets the rates.
    pub everyday_rate: [f32; super::everyday::KIND_COUNT],
    pub everyday_decay_m: [f32; super::everyday::KIND_COUNT],
    /// T-1092: the share of errand and social outings drawn to central destinations, and their
    /// decay length in metres (`everyday::CENTRAL_SHARE`; 0 turns them off).
    pub everyday_central_share: f32,
    pub everyday_central_m: f32,
    /// The everyday layer's rail constant is `asc_rail` plus this (`Params::everyday`); 0: the
    /// two layers share their constants (T-636), the gap kept for a later fit.
    pub everyday_asc_rail: f32,
}

/// Walks longer than this straight line (km, round water where the routes say) are not offered
/// (T-558): a computational bound, past which walking takes under one trip in a million.
pub const WALK_REACH_KM: f32 = 10.0;
/// T-2307: the walking rule leaves a trip whole when the station pairs it would refuse carry at
/// most this share of the trip's station-pair weight: worth under 0.1 perceived minutes of rail,
/// and loading and taking off such slivers was most of the rule's cost.
pub const WALK_SLIVER: f32 = 0.01;

/// T-2278 (`park_income_of`, notes/T-2278.md): CBD parking prices rise about with GDP per head
/// (market US$) below this and not above it, and a dollar weighs GDP^-0.7 minutes (the fares'
/// rule, app game/income.ts), so paid parking's minutes go as GDP^0.3 below it.
pub const PARK_INCOME_KNEE_USD: f32 = 35_000.0;
pub const PARK_INCOME_E: f32 = 0.3;

/// The factor on paid parking for a country of `gdp` US$ a head at market rates (T-2278): 1 from
/// `PARK_INCOME_KNEE_USD` up, (gdp / knee)^`PARK_INCOME_E` below; 1 with no figure.
pub fn park_income_of(gdp: Option<f32>) -> f32 {
    match gdp {
        Some(y) if y > 0.0 => (y / PARK_INCOME_KNEE_USD).powf(PARK_INCOME_E).min(1.0),
        _ => 1.0,
    }
}
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
            decay_km: US_DECAY_KM,
            decay_pow: US_DECAY_POW,
            decay_flat_km: 0.0,
            border_factor: 1.0,
            // T-371, T-372: 1.5 rather than the usual 2.0 (notes/T-371.md, T-372.md)
            w_walk: 1.5,
            transfer_min: 2.0,
            xfer_fare: 0.0,
            fare_min: 6.0,
            fare_w: super::api::FARE_MIN_PER_USD,
            // T-636: Anita's set (notes/T-636.md)
            beta: 0.10,
            asc_rail: super::api::ASC_RAIL,
            country_asc_rail: 0.0,
            damp_min: 60.0,
            // T-918: Anita's fade, so a detour line cannot be farmed (notes/T-918.md)
            ratio_fade_lo: 3.0,
            ratio_fade_hi: 4.0,
            road_detour: 1.3,
            road_local_km: 2.0,
            road_end_km: 8.0,
            // T-938: T-090's fitted speeds and T-370's 100 km/h, each 10% quicker (Anita)
            road_open_kmh: 110.0,
            road_city_kmh: 16.5,
            road_town_kmh: 44.0,
            road_dens50: 5000.0,
            road_far_kmh: 88.0,
            w_drive: 1.33,
            park_min: 7.0,
            // T-380: within about half of real prices from Croydon to Midtown (notes/T-380.md)
            park_paid_min: 15.0,
            park_paid_d0: 3000.0,
            park_paid_cap_min: 50.0,
            park_free_d: 8000.0,
            park_income: 1.0,
            // T-854: dense cores keep about the rail share the car-free split gave them
            home_park_f: 0.4,
            home_dens: None,
            // T-938 then T-1755: T-621's $1.50 + $0.10 raised about 30%, then 5% (Anita)
            car_own_usd_trip: 2.1,
            car_own_usd_km: 0.1365,
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
            chunk_salt: 0,
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
            walk_rule: 1.0,
            xfer_mps: 1.2,
            w_xfer: 1.39,
            station_merge_m: 100.0,
            left_behind: true,
            routes: None,
            parts: None,
            road_routed_scale: 1.3 / 1.210,
            // T-372: fitted on the real New York network (notes/T-372.md)
            p_ride: 15.0,
            ride_drive_f: 2.0,
            d_drop: 10.0,
            ride_min_km: 5.0,
            ride_max_m: 15_000.0,
            ride_lines: 6,
            everyday_rate: super::everyday::RATE,
            everyday_decay_m: super::everyday::DECAY,
            everyday_central_share: super::everyday::CENTRAL_SHARE,
            everyday_central_m: super::everyday::CENTRAL_DECAY_M,
            everyday_asc_rail: 0.0,
        }
    }
}

impl Params {
    /// What every boarding adds besides its wait, perceived minutes: the change's fixed cost and
    /// the transfer fare; the trip's first boarding takes it off again (`DestSearch::run_by`).
    #[inline]
    pub fn change_min(&self) -> f32 {
        self.transfer_min + self.xfer_fare
    }
    /// `change_min` at a station whose change counts `f` of its fixed cost (`Graph::change_f`,
    /// T-2220); the fare is whole. At 1 exactly `change_min`.
    #[inline]
    pub fn change_at(&self, f: f32) -> f32 {
        f * self.transfer_min + self.xfer_fare
    }
    /// Set a parameter by name, for the checking tools (`calib --set`); not every field.
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
            "xfer_fare" => self.xfer_fare = v,
            // before `set_fares`, which reads it
            "fare_w" => self.fare_w = v,
            "transfer_min" => self.transfer_min = v,
            "crowd_alpha" => self.crowd_alpha = v,
            "xfer_walk_m" => self.xfer_walk_m = v,
            "walk_rule" => self.walk_rule = v,
            "station_merge_m" => self.station_merge_m = v,
            // the gravity holds the walk's costs: `DemandApi::refresh_walk` after these
            "walk_trip_w" => self.walk_trip_w = v,
            "walk_bonus_min" => self.walk_bonus_min = v,
            "walk_samples" => self.walk_samples = v as usize,
            "damp_min" => self.damp_min = v,
            "ratio_fade_lo" => self.ratio_fade_lo = v,
            "ratio_fade_hi" => self.ratio_fade_hi = v,
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
            "ride_drive_f" => self.ride_drive_f = v,
            "d_drop" => self.d_drop = v,
            "ride_min_km" => self.ride_min_km = v,
            "ride_max_m" => self.ride_max_m = v,
            "ride_lines" => self.ride_lines = v as usize,
            "park_paid_min" => self.park_paid_min = v,
            "park_paid_d0" => self.park_paid_d0 = v,
            "park_paid_cap_min" => self.park_paid_cap_min = v,
            "park_free_d" => self.park_free_d = v,
            "park_income" => self.park_income = v,
            "home_park_f" => self.home_park_f = v,
            "car_own_usd_trip" => self.car_own_usd_trip = v,
            "car_own_usd_km" => self.car_own_usd_km = v,
            // T-2214: 0 drops the pack's land parts (checks: a pack solved as one part)
            "parts" => {
                if v == 0.0 {
                    self.parts = None;
                }
            }
            // scales every zone's cordon charge (0 turns them off)
            "cordon" => {
                if let Some(c) = &self.cordon {
                    self.cordon = Some(std::sync::Arc::new(c.iter().map(|x| x * v).collect()));
                }
            }
            "everyday_asc_rail" => self.everyday_asc_rail = v,
            "central_share" => self.everyday_central_share = v,
            "central_m" => self.everyday_central_m = v,
            // the everyday layer's, by purpose (`rate_errands`, `decay_social` in metres); before
            // `EverydayApi::beside`
            _ if k.starts_with("rate_") || k.starts_with("decay_") => {
                let (what, kind) = k.split_once('_').unwrap();
                let i = super::everyday::KINDS.iter().position(|&n| n == kind).ok_or(format!("unknown parameter {k}"))?;
                if what == "rate" { self.everyday_rate[i] = v } else { self.everyday_decay_m[i] = v }
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
        self.access_cost_min(self.walk_min(d_m))
    }
    /// `access_cost` of a walk of `t` real minutes.
    pub fn access_cost_min(&self, t: f32) -> f32 {
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
        let paid = if d > self.park_paid_d0 { (self.park_paid_min * (d / self.park_paid_d0).ln()).min(self.park_paid_cap_min) * self.park_income() } else { 0.0 };
        let fixed = self.park_min + self.car_own_min() + self.cordon.as_ref().and_then(|c| c.get(j)).map_or(0.0, |&c| c);
        (fixed, paid, 1.0 / (1.0 + d / self.park_free_d))
    }
    /// Parking at the home end of a car trip from a zone of `d` residents per km² of land, perceived
    /// minutes (T-854).
    #[inline]
    pub fn home_park_at(&self, d: f32) -> f32 {
        if d > self.park_paid_d0 { self.home_park_f * (self.park_paid_min * (d / self.park_paid_d0).ln()).min(self.park_paid_cap_min) * self.park_income() } else { 0.0 }
    }
    /// T-2278: the factor on paid parking (at the destination and at home) for the pack country's
    /// income (`park_income_of`).
    #[inline]
    pub fn park_income(&self) -> f32 {
        self.park_income
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
    /// A ride's cost at the station in zone `z` (perceived minutes), the part of it that is money
    /// (paid parking and any cordon charge), and whether it is the drop-off. At the home end the
    /// cheaper of parking there, as at a destination (its mean over drivers, less the ownership
    /// cost, which the drive pays on its own road), and a drop-off, `d_drop`; at the far end there
    /// is no car to park, only a drop-off (T-621).
    pub fn ride_station_end(&self, zones: &Zones, z: Option<usize>, end: End) -> (f32, f32, bool) {
        if end == End::Far {
            return (self.d_drop, 0.0, true);
        }
        let (park, money) = match z {
            Some(z) => {
                let (fixed, paid, free) = self.park_parts(zones, z);
                let cordon = fixed - self.park_min - self.car_own_min();
                (fixed + (1.0 - free) * paid - self.car_own_min(), cordon + (1.0 - free) * paid)
            }
            None => (self.park_min, 0.0),
        };
        if park <= self.d_drop { (park, money, false) } else { (self.d_drop, 0.0, true) }
    }
    /// What a minute of the drive to a station counts, perceived minutes.
    #[inline]
    pub fn ride_w(&self) -> f32 {
        self.ride_drive_f * self.w_drive
    }
    /// A ride entry's perceived minutes `w` (its drive `t` real minutes) as the access bound sees
    /// them, the drive at `w_drive` (`Subzones::bound_min`).
    #[inline]
    pub fn ride_bound_cost(&self, w: f32, t: f32) -> f32 {
        w - (self.ride_w() - self.w_drive) * t
    }
    /// A ride's ownership cost over `road_km` of road to the station: none at the far end.
    #[inline]
    pub fn ride_own(&self, road_km: f32, end: End) -> f32 {
        match end {
            End::Far => 0.0,
            End::Home => self.car_own_min() + road_km * self.car_own_km_min(),
        }
    }
    /// The mode choice's logit scale for a trip whose car costs `car` perceived minutes (the
    /// regional layer's car even where none can go, so rail against air never depends on it): `beta` up to `damp_min`, then `beta` x sqrt(`damp_min` / `car`), so a
    /// minute weighs less on a long trip (T-495). The same expressions as `mode_choice`'s lanes.
    #[inline]
    pub fn mode_beta(&self, car: f32) -> f32 {
        self.beta * (self.damp_min / car).min(1.0).sqrt()
    }
    /// Rail's share of rail + car, rail costing `rail` perceived minutes and the car `car`
    /// (T-918): 1 / (1 + q^3) for q = rail / car, so q and 1 / q mirror each other; rail fades to
    /// nothing from `ratio_fade_lo` to `ratio_fade_hi` times the car's cost, and the car the same
    /// where it costs that many times rail's. The same expressions as `mode_choice`'s lanes.
    #[inline]
    pub fn ratio_share(&self, rail: f32, car: f32) -> f32 {
        let q = rail.max(0.01) / car;
        let s = 1.0 / (1.0 + q * q * q);
        let k = 1.0 / (self.ratio_fade_hi - self.ratio_fade_lo);
        let fr = ((self.ratio_fade_hi - q) * k).max(0.0).min(1.0);
        let fc = ((self.ratio_fade_hi - 1.0 / q) * k).max(0.0).min(1.0);
        1.0 - (1.0 - s * fr) * fc
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
    /// With a car, rail's share of all trips among rail, walking and the car, for drivers parking
    /// free (the car costing `road`) and those paying `paid` more for a space, mixed by the share
    /// `free` (T-380): walking takes its share of the three-way logit (`util` rail's utility and
    /// `uw` walking's against a driver who parks free, at scale `beta`, T-558), and rail and the
    /// car split the rest by `ratio_share` on rail's cost `rail` (T-918). The same expressions as
    /// `mode_choice`'s lanes.
    #[inline]
    pub fn car_choice(&self, util: f32, uw: f32, beta: f32, paid: f32, free: f32, rail: f32, road: f32) -> f32 {
        let e = fast_exp(uw - util);
        let w0 = e / (1.0 + fast_exp(-util) + e);
        let s0 = (1.0 - w0) * self.ratio_share(rail, road);
        if paid < 1e-30 {
            return s0;
        }
        let w1 = e / (1.0 + fast_exp(-(util + beta * paid)) + e);
        let s1 = (1.0 - w1) * self.ratio_share(rail, road + paid);
        s0 + (1.0 - free) * (s1 - s0)
    }
    /// Real minutes as the straight metres walking the whole way covers in them, the unit of the
    /// walking rule (`walk_rule`, `Skims::walk`).
    #[inline]
    pub fn walk_eq_m(&self, min: f32) -> f32 {
        min * self.walk_mps * 60.0 / self.walk_detour
    }
    /// Whether the walking rule applies to a trip whose ends are `direct` metres apart on foot: where
    /// walking the whole way is offered (`WALK_REACH_KM`).
    #[inline]
    pub fn walk_rule_at(&self, direct: f32) -> bool {
        self.walk_rule > 0.0 && direct <= WALK_REACH_KM * 1000.0
    }
    /// T-2214: rail's share of a trip between two land parts, where it is the only way: all of
    /// them up to `ratio_fade_lo` times the cost of a car on a bridge that is not there (`road`,
    /// to a driver parking free; `paid` more for a space, `free` parking free), none from
    /// `ratio_fade_hi`, linear between. The same expressions as `mode_choice`'s lanes.
    #[inline]
    pub fn bound_share(&self, rail: f32, road: f32, paid: f32, free: f32) -> f32 {
        let (hi, kf) = (self.ratio_fade_hi, 1.0 / (self.ratio_fade_hi - self.ratio_fade_lo));
        let rail_c = rail.max(0.01);
        let f0 = ((hi - rail_c / road) * kf).max(0.0).min(1.0);
        if paid < 1e-30 {
            return f0;
        }
        let f1 = ((hi - rail_c / (road + paid)) * kf).max(0.0).min(1.0);
        f0 + (1.0 - free) * (f1 - f0)
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

/// T-2214: the pack's land parts (its `part` per cell, T-2213; demand-mode.md 4.3 "Land parts"):
/// between two there is no car and no walk, and stations are reached only within one. Each
/// zone's cells by part are its slots, CSR by zone in part order.
#[derive(Debug)]
pub struct Parts {
    pub cell: Vec<u16>,
    /// The part with the most commuters' homes and jobs.
    pub main: u16,
    pub z_off: Vec<u32>,
    pub s_part: Vec<u16>,
    /// Each slot's home and work weight (`City::pop`, `City::jobs` over its cells).
    pub s_pop: Vec<f64>,
    pub s_jobs: Vec<f64>,
    pub cell_slot: Vec<u32>,
    /// The zones with a slot off `main`, ascending.
    pub mixed: Vec<u32>,
}

impl Parts {
    /// None where every cell is on one part (or `part` is not one per cell).
    pub fn of(part: Vec<u16>, city: &City, z: &Zones) -> Option<Parts> {
        if part.len() != city.len() || part.iter().all(|&p| p == part[0]) {
            return None;
        }
        let np = *part.iter().max().unwrap() as usize + 1;
        let mut mass = vec![0f64; np];
        for c in 0..city.len() {
            mass[part[c] as usize] += (city.pop[c] + city.jobs[c]) as f64;
        }
        let main = (0..np).max_by(|&a, &b| mass[a].total_cmp(&mass[b]).then(b.cmp(&a))).unwrap() as u16;
        let mut by_zone: Vec<Vec<u16>> = vec![vec![]; z.n];
        for c in 0..city.len() {
            let l = &mut by_zone[z.cell_zone[c] as usize];
            if !l.contains(&part[c]) {
                l.push(part[c]);
            }
        }
        let mut pp = Parts { cell: vec![], main, z_off: vec![0], s_part: vec![], s_pop: vec![], s_jobs: vec![], cell_slot: vec![0; city.len()], mixed: vec![] };
        for (i, l) in by_zone.iter_mut().enumerate() {
            l.sort_unstable();
            pp.s_part.extend_from_slice(l);
            pp.z_off.push(pp.s_part.len() as u32);
            if l.iter().any(|&p| p != main) {
                pp.mixed.push(i as u32);
            }
        }
        pp.s_pop = vec![0.0; pp.s_part.len()];
        pp.s_jobs = vec![0.0; pp.s_part.len()];
        for c in 0..city.len() {
            let zi = z.cell_zone[c] as usize;
            let s = pp.z_off[zi] as usize + by_zone[zi].binary_search(&part[c]).unwrap();
            pp.cell_slot[c] = s as u32;
            pp.s_pop[s] += city.pop[c] as f64;
            pp.s_jobs[s] += city.jobs[c] as f64;
        }
        pp.cell = part;
        Some(pp)
    }
    /// Zone i's slots.
    #[inline]
    pub fn slots(&self, i: usize) -> std::ops::Range<usize> {
        self.z_off[i] as usize..self.z_off[i + 1] as usize
    }
    /// The share of zone i's jobs on part `q` (`z` the zones the slots were made on).
    pub fn job_share(&self, z: &Zones, i: usize, q: u16) -> f64 {
        if z.jobs[i] <= 0.0 {
            return 0.0;
        }
        self.slots(i).find(|&s| self.s_part[s] == q).map_or(0.0, |s| self.s_jobs[s] / z.jobs[i])
    }
    pub fn home_share(&self, z: &Zones, i: usize, q: u16) -> f64 {
        if z.pop[i] <= 0.0 {
            return 0.0;
        }
        self.slots(i).find(|&s| self.s_part[s] == q).map_or(0.0, |s| self.s_pop[s] / z.pop[i])
    }
}

/// T-2214: per slot of `Parts` (or per zone with none), the day's commutes with no rail that walk
/// and that no way serves (`Gravity::no_rail_by_slot`), at the home end and at the work end.
#[derive(Clone, Default)]
pub struct NoRailSlots {
    pub walk_home: Vec<f64>,
    pub walk_work: Vec<f64>,
    /// Empty with no parts.
    pub unr_home: Vec<f64>,
    pub unr_work: Vec<f64>,
}

mod gravity;
mod access;
mod subzones;
mod search;
mod mode;
mod assign;
mod stalls;
pub use gravity::*;
pub use access::*;
pub use subzones::*;
pub use search::*;
pub use mode::*;
pub use assign::*;
pub use stalls::*;

#[cfg(test)]
mod tests;
