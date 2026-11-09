//! Every tuning constant of the track model in one place (SPEC 6, notes/T-008.md, notes/T-040.md).

// ---- geometry and speed (SPEC 6.1) ----

/// Lateral acceleration allowed in curves, m/s² (LGV design: 7,000 m for 320 km/h).
pub const A_LAT: f64 = 1.1;
/// Smallest radius a player may build, m (27 km/h; T-2799: about the tightest metro curves in use).
pub const MIN_RADIUS: f64 = 50.0;
/// Smallest radius a traced real line takes, m (38 km/h): a jagged source must not slow its trains.
pub const TRACE_MIN_RADIUS: f64 = 100.0;
/// Top speed of an ordinary train, m/s (160 km/h), and the cap on track other than high speed.
pub const V_TOP: f64 = 160.0 / 3.6;
/// Top speed of a high-speed train and the cap on high-speed track, m/s (T-817).
pub const V_HSR: f64 = 320.0 / 3.6;
/// Auto radius cap: the smallest whole-metre radius that already allows V_TOP
/// (ceil(44.44² / 1.1) = 1,796 m; T-008 rounds it to 1,800).
pub const R_CAP: f64 = 1796.0;
/// Design speeds the drawing tool offers, km/h (T-817): the auto radius of track it draws is
/// capped at `design_radius` of the one chosen. 160 is `R_CAP`.
pub const DESIGN_SPEEDS: [f64; 4] = [160.0, 200.0, 250.0, 320.0];

/// The smallest whole-metre radius that allows `kmh`: 1,796 m at 160, 2,806 at 200, 4,385 at 250,
/// 7,183 at 320.
pub fn design_radius(kmh: f64) -> f64 {
    ((kmh / 3.6).powi(2) / A_LAT).ceil()
}
/// Positions and radii are quantised to this (1 mm) when the player places them, so derived
/// geometry is a deterministic f64 function of the save.
pub const QUANT: f64 = 0.001;
/// Every edge end at a node must leave along the node's heading or its reverse, within this (rad).
pub const HEADING_TOL: f64 = 0.003;
/// Slack when checking that two tangent lengths fit on one leg, m.
pub const FIT_TOL: f64 = 0.01;
/// Intersections closer than this to a node both edges share are the node itself, not a crossing.
pub const SHARED_NODE_CLEAR: f64 = 25.0;
/// A crossover (T-338): its length along the track, m, the farthest apart the two tracks it joins
/// may be (centre to centre, m; the nearest is `CROSSOVER_MIN_GAP`), and its speed limit, m/s.
pub const CROSSOVER_LEN: f64 = 120.0;
pub const CROSSOVER_REACH: f64 = 15.0;
pub const CROSSOVER_MIN_GAP: f64 = 6.0;
pub const V_CROSSOVER: f64 = 50.0 / 3.6;

// ---- levels (SPEC 6.1) ----

pub const MIN_LEVEL: i8 = -3;
pub const MAX_LEVEL: i8 = 3;
/// Height between levels, m.
pub const LEVEL_H: f64 = 8.0;
pub const MAX_GRADE: f64 = 0.04;
/// Ramp length per level of height change, m (8 m at 4%).
pub const RAMP_PER_LEVEL: f64 = LEVEL_H / MAX_GRADE;

// ---- stations and trains (SPEC 6.1, 6.3) ----

pub const PLATFORM_MIN: u16 = 60;
pub const PLATFORM_MAX: u16 = 400;
pub const PLATFORM_STEP: u16 = 20;
pub const CAR_LEN: f64 = 20.0;
pub const DEFAULT_DWELL_S: f32 = 30.0;
pub const DEFAULT_TURNAROUND_S: f32 = 180.0;

/// Traction: 1.0 m/s² up to 60 km/h, 0.5 m/s² to 200 km/h, 0.3 m/s² above, which only high-speed
/// trains reach (T-817). A constant-power EMU falls off with speed; bands keep every phase at
/// constant acceleration, so keyframes stay exact. Braking 0.9 m/s² (normal service braking;
/// 1.1-1.3 is the usual maximum service rate). notes/T-040.md.
pub const ACCEL_LOW: f64 = 1.0;
pub const ACCEL_HIGH: f64 = 0.5;
pub const ACCEL_TOP: f64 = 0.3;
pub const ACCEL_SWITCH_V: f64 = 60.0 / 3.6;
pub const ACCEL_TOP_V: f64 = 200.0 / 3.6;
pub const BRAKE: f64 = 0.9;

// ---- costs, US$M (SPEC 6.4) ----

/// Per route-km of double track at multiplier 1, which is the first viaduct and cut and cover (T-1928,
/// Anita 2026-11-02, notes/T-1897.md): ground is 0.5 of it (US$37.5M; T-2017), bored tunnels 1.6 and 2.0, a
/// bored tunnel near cheap-country metros (Korea, Spain: about $100-150M a tunnelled km with stations).
pub const BASE_COST_PER_KM: f64 = 75.0;
/// Multiplier per level, index `level + 3` (-3 .. +3): up and down mirror each other.
pub const LEVEL_MULT: [f64; 7] = [2.0, 1.6, 1.0, 0.5, 1.0, 1.5, 2.0];
pub const SINGLE_TRACK_COST: f64 = 0.6;
/// High-speed track's price over double track at the same level (T-817, notes/T-817.md).
pub const HSR_COST: f64 = 1.5;
/// The track count a blueprint quote row gives high-speed track (`blueprint_cost_items`).
pub const HSR_ROW: u8 = 3;
pub const WATER_COST: f64 = 2.0;
/// 200 m station with two platform tracks: the part every station has wherever it is (platforms,
/// halls, systems), plus its structure's part by level, index `level + 3`: 50 + 25 a level down,
/// 25 + 15 a level up. Ground to -2 is 75 to 150, half, where track is 0.45 (T-1267).
pub const STATION_FIXED_200: f64 = 50.0;
pub const STATION_LEVEL_200: [f64; 7] = [125.0, 100.0, 75.0, 25.0, 40.0, 55.0, 70.0];
pub const SINGLE_TRACK_STATION: f64 = 0.7;
pub const JUNCTION_KM: f64 = 0.25;
pub const FLYOVER_KM: f64 = 0.6;
pub const FLAT_CROSSING_KM: f64 = 0.1;
/// Water mask sampling step along track, m: half the pack mask's 25 m pixel (notes/T-004.md).
pub const WATER_STEP: f64 = 12.5;
/// What each terrain class adds to the level's multiplier (T-1890, Anita 2026-10-31): flat, low
/// hills, high hills, mountains. Added, so at grade in mountains is 0.5 + 3 = 3.5x.
pub const TERRAIN_SURCHARGE: [f64; 4] = [0.0, 0.5, 1.2, 3.0];
/// What water adds to track's multiplier at any level (T-1890); stations, junctions and crossings
/// over water take `WATER_COST` times their price.
pub const WATER_SURCHARGE: f64 = 2.5;
/// The most unbroken water a viaduct or a cut and cover tunnel may cross, m (T-1930): Hangzhou Bay's
/// rail bridge is 29 km. Bored tunnels have no limit.
pub const WATER_SPAN_MAX: f64 = 30_000.0;
/// The lowest level the terrain surcharge applies to: bored tunnels (-2, -3) cost the same anywhere.
pub const TERRAIN_MIN_LEVEL: i8 = -1;
/// Terrain sampling step along track, m: under a quarter of the class grid's ~115 m cell (T-1854).
pub const TERRAIN_STEP: f64 = 25.0;
/// Per road that track at ground level crosses, US$M (T-1929; 10 since T-2732, Anita, standing in for
/// the buildings cleared: a viaduct is cheaper to build from about 3.75 crossings a km, open country's
/// under 1 a km stays cheapest at grade). Each road comes with its weight, a motorway's 3 (app `roadTiles.ts`).
pub const ROAD_CROSSING_COST: f64 = 10.0;
/// Step along track for finding road crossings, m: chords this long stray under 20 cm from a
/// 300 m curve.
pub const ROAD_STEP: f64 = 20.0;
/// Price of one 20 m car, US$M (T-028; a New York R211 car is about 2.6). Trains are bought when
/// the running lines need more cars than the network owns; spare cars are kept, never sold.
pub const CAR_PRICE: f64 = 2.5;
/// Running cost per train-km, US$, and per car-km on top: the crew and the trains. Track and
/// stations pay their own upkeep (`UPKEEP_*`, T-867), so a 9-car train costs 22 a train-km and a
/// 4-car one 12. Under real figures so fares can cover it at 100x (research/running-costs.md).
pub const RUN_COST_TRAIN_KM: f64 = 4.0;
pub const RUN_COST_CAR_KM: f64 = 2.0;
/// A high-speed car's price and running costs (T-817, T-867): 1.5x a car, 5 + 3.
pub const HS_CAR_PRICE: f64 = 3.75;
pub const HS_RUN_COST_TRAIN_KM: f64 = 5.0;
pub const HS_RUN_COST_CAR_KM: f64 = 3.0;
/// Upkeep a day, real US$ (US$35k, 50k, 100k, 130k, 50k and 200k a game day at `ECONOMY`), charged
/// whether trains run or not (T-867, notes/T-867.md): per route-km of constructed double track,
/// times `track_factor` (so quad track, two pairs, pays twice), and per constructed station.
/// Track pays by structure (T-1931, Anita 2026-11-02): ground, viaduct (+1 to +3), cut and cover
/// (-1), bored tunnel (-2, -3); an edge pays the dearest it touches (`cost::track_upkeep_km`). A
/// station pays `UPKEEP_STATION` at grade or above and `UPKEEP_STATION_UNDER` below ground, -1 to
/// -3, where its construction price takes the dearer per-level step (`cost::station_upkeep`).
/// Anita's figures (T-2619, 2026-11-06; research/upkeep-france.md).
pub const UPKEEP_GROUND_KM: f64 = 350.0;
pub const UPKEEP_VIADUCT_KM: f64 = 500.0;
pub const UPKEEP_CUT_KM: f64 = 1000.0;
pub const UPKEEP_BORED_KM: f64 = 1300.0;
pub const UPKEEP_STATION: f64 = 500.0;
pub const UPKEEP_STATION_UNDER: f64 = 2000.0;
/// The economy runs at 100x (T-081; Anita, 2026-10-09, as Subway Builder does): fares and running
/// costs per game day are this many times the real figures, so a line pays back in about a game
/// week. Construction and train prices stay real. The one constant: running costs here, fares in
/// app/src/game/money.ts (which reads it through `TrackApi.money_params`).
pub const ECONOMY: f64 = 100.0;

// ---- buses (T-1464, notes/T-1445.md) ----

/// Per road-km of bus route (stops, shelters, signs), US$M, paid on Construct.
pub const BUS_ROUTE_COST_PER_KM: f64 = 1.0;
/// Price of one bus, US$M; buses are a fleet of their own, spares shared between bus lines.
pub const BUS_PRICE: f64 = 0.6;
/// Running cost per bus-km, real US$: two-thirds of real cost, nearer rail's discount (T-1718).
pub const RUN_COST_BUS_KM: f64 = 6.0;
/// Upkeep a day per road-km of constructed route, real US$ (US$3k a game day at `ECONOMY`).
pub const UPKEEP_BUS_KM: f64 = 30.0;
/// A bus's riders at crush.
pub const BUS_CRUSH: f64 = 80.0;

// ---- ferries (T-2193, notes/T-2177.md 3.1, 3.4, 3.5) ----

/// Price of a port, US$M, paid on Construct (a NYC Ferry landing barge).
pub const PORT_COST: f64 = 10.0;
/// Upkeep a day per constructed port, real US$ (US$10k a game day at `ECONOMY`, a fifth of a station's).
pub const UPKEEP_PORT: f64 = 100.0;
/// Price of one boat, US$M (a NYC Ferry 350-seat boat is US$7-9M); boats are a fleet of their own,
/// spares shared between ferry lines.
pub const BOAT_PRICE: f64 = 8.0;
/// Running cost per boat-km, real US$.
pub const RUN_COST_BOAT_KM: f64 = 40.0;
/// A boat's riders at crush: two train cars (T-2195; demand's `BOAT_DEMAND_CARS`), 320 at the rated load.
pub const BOAT_CRUSH: f64 = 2.0 * super::boarding::CRUSH_PER_CAR;

pub fn level_mult(level: i8) -> f64 {
    LEVEL_MULT[(level.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize]
}

// ---- capacity (SPEC 6.2) ----

/// With `v / SIGNAL_BRAKE`, the most trains an hour a track takes (T-517: published maxima).
pub const SECTION_BASE_S: f64 = 70.0;
/// Signalling braking rate behind the block length term `v / 0.6`.
pub const SIGNAL_BRAKE: f64 = 0.6;
pub const SINGLE_TRACK_MARGIN_S: f64 = 60.0;
pub const PLATFORM_MARGIN_S: f64 = 60.0;
pub const PASSING_S: f64 = 40.0;
pub const TERMINUS_MARGIN_S: f64 = 60.0;
pub const CONFLICT_S: f64 = 90.0;
/// Every train reversing partway holds its station side's limiter this long: 12 an hour (T-797,
/// a stopgap until T-657's reversing design).
pub const REVERSAL_LIMIT_S: f64 = 300.0;
/// A reversal's cost to the route search (s) and its least cost to the stop insertion (m; more on
/// a longer route, `order_cost`), so a route without one wins unless it is much longer (T-797).
pub const REVERSAL_SEARCH_S: f64 = 600.0;
pub const REVERSAL_INSERT_M: f64 = 5000.0;
/// How far short of a bare track end a reversing train's tail stops, m (T-797 B3).
pub const BARE_END_CLEAR_M: f64 = 20.0;
pub const KINGMAN_C0: f64 = 0.2;
pub const RHO_KNEE: f64 = 0.9;
pub const OVERLOAD_WINDOW_S: f64 = 3600.0;
/// A resource at this utilisation counts as full: count schedules are cut to fit (T-517).
pub const FULL_RHO: f64 = 0.9999;
/// Delays are rounded to whole seconds; under this they are dropped.
pub const HOLD_MIN_S: f64 = 2.0;
/// A hold point closer than this to the previous stop becomes extra dwell there.
pub const HOLD_NEAR_STOP_M: f64 = 50.0;
/// Length of the slow approach for a short hold, m.
pub const CRAWL_ZONE_M: f64 = 300.0;
/// The train stops this far before a conflict point (plus half its length).
pub const HOLD_MARGIN_M: f64 = 20.0;
/// T-1924: a slower line on shared track counts for a faster one's following loss only when it
/// takes at least this much longer over the stretch, s.
pub const FOLLOW_MIN_S: f64 = 1.0;
/// T-2018: the slowest a following loss lowers a stretch's speed to, m/s; a loss too big for it
/// waits at the stop before the stretch.
pub const FOLLOW_CRAWL_MIN_V: f64 = 0.5;

// ---- service (SPEC 6.3, 6.5) ----

/// A station-to-station time moving this much or more wakes the city's demand.
pub const DEMAND_TIME_EPS_S: f64 = 5.0;
/// A rate an edit's settle moves on another line by this much or more (trains an hour) wakes demand
/// for that line (T-1805: a train every 100 hours changes no wait anyone could see).
pub const DEMAND_TPH_EPS: f32 = 0.01;
/// Demand levels: index into a line's `tph`.
pub const HIGH: usize = 0;
pub const MEDIUM: usize = 1;
pub const LOW: usize = 2;
pub const LEVELS: usize = 3;
/// The five periods of SPEC 4.2 in the demand kernel's order (morning peak, midday, evening
/// peak, evening, night): start and end hour, and which demand level schedules them. High runs
/// 6-20, medium 20-24, low 0-6 (Anita, 2026-10-27, T-1536). A level is a set of whole periods:
/// demand solves each period on its own level's trains.
pub const PERIOD_HOURS: [(f64, f64); 5] = [(6.0, 10.0), (10.0, 16.0), (16.0, 20.0), (20.0, 24.0), (0.0, 6.0)];
pub const PERIOD_LEVEL: [usize; 5] = [HIGH, HIGH, HIGH, MEDIUM, LOW];
/// Each period's busiest hour against its mean hour (T-087): how riders arrive over it (the
/// boarding account's shape, demand's crowding).
pub const PEAK_HOUR_FACTOR: [f32; 5] = [1.4, 1.15, 1.4, 1.3, 1.5];
/// Kinds of day (T-260, T-262, T-264, T-1373): weekday 0, Saturday 1, Sunday 2, a superholiday's
/// away day 3, going-out day 4 and coming-back day 5. Game day 1, a new game's first, is a Monday,
/// so days 6 and 7 of each week are Saturday and Sunday where a line's country says nothing else
/// (`Boarding::set_holidays`). The main thread's copy is app/src/game/days.ts.
pub const DAY_KINDS: usize = 6;
pub const AWAY: usize = 3;
pub const BACK: usize = 5;
pub fn day_kind_at(t: f64) -> usize {
    let day = (t / 86400.0).floor() as i64;
    match (day - 1).rem_euclid(7) {
        5 => 1,
        6 => 2,
        _ => 0,
    }
}

pub const EARTH_R: f64 = 6371008.8;

/// Round to the placement grid (1 mm).
pub fn quant(v: f64) -> f64 {
    (v / QUANT).round() * QUANT
}
