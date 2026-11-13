//! Every tuning constant of the track model in one place (SPEC 6, notes/T-008.md, notes/T-040.md).

// ---- geometry and speed (SPEC 6.1) ----

/// Lateral acceleration allowed in curves, m/s² (LGV design: 7,000 m for 320 km/h).
pub const A_LAT: f64 = 1.1;
/// Smallest radius a player may build on rail, m (19 km/h; T-3082: the Chicago Loop's corners are
/// about 27 m).
pub const MIN_RADIUS: f64 = 25.0;
/// High-speed track's smallest radius, m (27 km/h); unchanged by T-3082, which lowered rail's only.
pub const HSR_MIN_RADIUS: f64 = 50.0;
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
/// T-3121: the lowest design speed the drawing tool takes, km/h: its radius (29 m) is over rail's
/// smallest.
pub const DESIGN_MIN_KMH: f64 = 20.0;
/// T-3169: the flyover target max speed before the player sets one, km/h: the speed a real
/// flyover's crossing leg is laid out for where it has the room (DECISIONS.md).
pub const FLYOVER_KMH: f64 = 80.0;

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
/// Double track's two tracks lie this far either side of its centre line, m (4 m apart, T-319);
/// `lanes.rs`.
pub const LANE_HALF_M: f64 = 2.0;
/// Each edge end's slot across a node, m (T-3137): wider than a double end's two lanes, so the
/// ends' lanes never interleave.
pub const PORT_PITCH_M: f64 = 8.0;

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
/// the same for tram track (T-2926): a quote row's price is the tram table's, not rail's
pub const TRAM_ROW: u8 = 4;
/// and for street tram track (T-2927)
pub const STREET_TRAM_ROW: u8 = 5;
/// T-3352: a single-track row of those kinds is the kind's row plus this
pub const SINGLE_ROW_OFFSET: u8 = 3;
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
/// Width of a terrain hex across its flats, in world widths on the mercator plane (T-2890): 1/768 of
/// a degree of longitude, about 110 m at 40 N. The app draws the same lattice (`game/terrain.ts`).
pub const TERRAIN_HEX: f64 = 1.0 / (360.0 * 768.0);
/// How far east the hex lattice is shifted, world widths: 1/16 of a class grid cell at 960 a degree,
/// so no hex centre falls exactly on a cell edge, where rounding could pick different cells here and
/// in the app.
pub const TERRAIN_HEX_SHIFT: f64 = 1.0 / (360.0 * 15360.0);
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
/// A tram section's price, US$M (T-2926; section 8: about 2 for 20 m).
pub const TRAM_CAR_PRICE: f64 = 2.0;
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
/// A bus's riders at crush: a third of a train car (T-2919; demand's `BUS_CARS`), 53 at the rated load.
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

// ---- track kinds (T-2921 section 5, T-2925) ----

/// What an edge of track is. The number is what a save stores (edge flags bits 8-10; 0 reads bit 6).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[repr(u8)]
pub enum TrackKind {
    #[default]
    Regular = 0,
    HighSpeed = 1,
    StreetTram = 2,
    Tram = 3,
}

pub const TRACK_KINDS: usize = 4;

/// T-2997: tram track's smallest curve, centre line, m: one every common modern low-floor tram takes
/// (Citadis and Skoda 18, Combino 15); old street networks go down to 11-13 (notes/T-2921.md).
pub const TRAM_MIN_RADIUS: f64 = 18.0;

/// T-3004: the share of the terrain surcharge tram track pays, street and off-street (Anita, 2026-11-10).
pub const TRAM_TERRAIN: f64 = 0.25;

/// One car of a kind: its price (US$M), running costs (real US$ a train-km and a car-km), the share
/// of a rail car's riders it holds, and the fleet it is bought from.
pub struct CarSpec {
    pub price: f64,
    pub run_train_km: f64,
    pub run_car_km: f64,
    pub riders_scale: f64,
    pub fleet: usize,
}

/// Everything that differs between kinds of track.
pub struct TrackSpec {
    /// Nodes join only within a family: 0 rail, 1 tram (the check comes with T-2926).
    pub family: u8,
    pub v_cap_kmh: f64,
    pub min_radius: f64,
    /// Price multiplier per level, index `level + 3`, in the units of `LEVEL_MULT`.
    pub level_mult: [f64; 7],
    /// What terrain (hills, mountains) and what water add to a level's multiplier are scaled by.
    pub terrain_mult: f64,
    pub water_mult: f64,
    /// What the preview's price multiplier is measured against (`cost::price_scale`).
    pub price_scale: f64,
    /// Upkeep over ground rail's.
    pub upkeep: f64,
    pub road_crossings: bool,
    pub ramps_over: bool,
    pub section_s: f64,
    pub conflict_s: f64,
    pub platform_margin_s: f64,
    /// Shortest and longest platform, m.
    pub platform: (u16, u16),
    /// A stop's flat price at ground, US$M, where it is not the rail formula.
    pub station_flat: Option<f64>,
    /// A stop's upkeep a day, real US$, where it is not `UPKEEP_STATION`.
    pub station_upkeep: Option<f64>,
    /// What a new line of this kind starts with.
    pub dwell_s: f32,
    pub turnaround_s: f32,
    pub car: CarSpec,
}

const fn scaled(m: [f64; 7], k: f64) -> [f64; 7] {
    [m[0] * k, m[1] * k, m[2] * k, m[3] * k, m[4] * k, m[5] * k, m[6] * k]
}

pub static TRACK_SPECS: [TrackSpec; TRACK_KINDS] = [
    TrackSpec {
        family: 0,
        v_cap_kmh: 160.0,
        min_radius: MIN_RADIUS,
        level_mult: LEVEL_MULT,
        terrain_mult: 1.0,
        water_mult: 1.0,
        price_scale: 1.0,
        upkeep: 1.0,
        road_crossings: true,
        ramps_over: true,
        section_s: SECTION_BASE_S,
        conflict_s: CONFLICT_S,
        platform_margin_s: PLATFORM_MARGIN_S,
        platform: (PLATFORM_MIN, PLATFORM_MAX),
        station_flat: None,
        station_upkeep: None,
        dwell_s: DEFAULT_DWELL_S,
        turnaround_s: DEFAULT_TURNAROUND_S,
        car: CarSpec { price: CAR_PRICE, run_train_km: RUN_COST_TRAIN_KM, run_car_km: RUN_COST_CAR_KM, riders_scale: 1.0, fleet: 0 },
    },
    TrackSpec {
        family: 0,
        v_cap_kmh: 320.0,
        min_radius: HSR_MIN_RADIUS,
        level_mult: scaled(LEVEL_MULT, HSR_COST),
        terrain_mult: HSR_COST,
        water_mult: HSR_COST,
        price_scale: HSR_COST,
        upkeep: HSR_COST,
        road_crossings: true,
        ramps_over: true,
        section_s: SECTION_BASE_S,
        conflict_s: CONFLICT_S,
        platform_margin_s: PLATFORM_MARGIN_S,
        platform: (PLATFORM_MIN, PLATFORM_MAX),
        station_flat: None,
        station_upkeep: None,
        dwell_s: DEFAULT_DWELL_S,
        turnaround_s: DEFAULT_TURNAROUND_S,
        car: CarSpec { price: HS_CAR_PRICE, run_train_km: HS_RUN_COST_TRAIN_KM, run_car_km: HS_RUN_COST_CAR_KM, riders_scale: 1.0, fleet: 1 },
    },
    TrackSpec {
        family: 1,
        v_cap_kmh: 60.0,
        min_radius: TRAM_MIN_RADIUS,
        level_mult: [1.6, 1.2, 0.6, 0.1, 0.6, 1.1, 1.6],
        // T-2928: half the water street tram track pays (Anita, 2026-11-09)
        terrain_mult: TRAM_TERRAIN,
        water_mult: 0.5,
        price_scale: 1.0,
        upkeep: 0.6,
        road_crossings: false,
        ramps_over: false,
        section_s: 60.0,
        conflict_s: 45.0,
        platform_margin_s: 60.0,
        platform: (20, 80),
        station_flat: Some(2.0),
        station_upkeep: Some(50.0),
        dwell_s: 15.0,
        turnaround_s: 120.0,
        car: CarSpec { price: TRAM_CAR_PRICE, run_train_km: RUN_COST_TRAIN_KM, run_car_km: RUN_COST_CAR_KM, riders_scale: 0.75, fleet: 4 },
    },
    TrackSpec {
        family: 1,
        v_cap_kmh: 80.0,
        min_radius: TRAM_MIN_RADIUS,
        level_mult: [1.6, 1.2, 0.6, 0.5, 0.6, 1.1, 1.6],
        terrain_mult: TRAM_TERRAIN,
        water_mult: 1.0,
        price_scale: 1.0,
        upkeep: 1.0,
        road_crossings: false,
        ramps_over: false,
        section_s: 60.0,
        conflict_s: 45.0,
        platform_margin_s: 60.0,
        platform: (20, 80),
        station_flat: Some(2.0),
        station_upkeep: Some(50.0),
        dwell_s: 15.0,
        turnaround_s: 120.0,
        car: CarSpec { price: TRAM_CAR_PRICE, run_train_km: RUN_COST_TRAIN_KM, run_car_km: RUN_COST_CAR_KM, riders_scale: 0.75, fleet: 4 },
    },
];

impl TrackKind {
    pub const ALL: [TrackKind; TRACK_KINDS] = [TrackKind::Regular, TrackKind::HighSpeed, TrackKind::StreetTram, TrackKind::Tram];

    pub fn spec(self) -> &'static TrackSpec {
        &TRACK_SPECS[self as usize]
    }

    pub fn from_u8(n: u8) -> Option<TrackKind> {
        Self::ALL.get(n as usize).copied()
    }

    /// Top speed of the track, m/s.
    pub fn v_cap(self) -> f64 {
        self.spec().v_cap_kmh / 3.6
    }

    pub fn is_hsr(self) -> bool {
        self == TrackKind::HighSpeed
    }

    /// The kind a high-speed flag stands for.
    pub fn from_hsr(hsr: bool) -> TrackKind {
        if hsr { TrackKind::HighSpeed } else { TrackKind::Regular }
    }
}

// ---- capacity (SPEC 6.2) ----

/// With `v / SIGNAL_BRAKE`, the most trains an hour a track takes (T-517: published maxima).
pub const SECTION_BASE_S: f64 = 70.0;
/// Signalling braking rate behind the block length term `v / 0.6`.
pub const SIGNAL_BRAKE: f64 = 0.6;
pub const SINGLE_TRACK_MARGIN_S: f64 = 240.0;
pub const PLATFORM_MARGIN_S: f64 = 60.0;
pub const PASSING_S: f64 = 40.0;
pub const TERMINUS_MARGIN_S: f64 = 60.0;
pub const CONFLICT_S: f64 = 120.0;
/// A reversal's cost to the route search (s) and its least cost to the stop insertion (m; more on
/// a longer route, `order_cost`), so a route without one wins unless it is much longer (T-797).
pub const REVERSAL_SEARCH_S: f64 = 600.0;
pub const REVERSAL_INSERT_M: f64 = 5000.0;
/// How much more track time a loop's closing hop may take to carry on round rather than turn back
/// at the first stop (T-3021): about the turnaround it saves and two minutes, the slack a turn
/// round a loop at a line's end is allowed (`turn_allowance`).
pub const CARRY_SLACK_S: f64 = 300.0;
/// A turn round a loop at a line's end may make its round trip this much longer, s, or 10%
/// (T-3021).
pub const TURN_SLACK_S: f64 = 120.0;
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

#[cfg(test)]
mod kind_tests {
    use super::*;

    #[test]
    fn regular_and_high_speed_rows_are_the_old_constants() {
        let (r, h) = (TrackKind::Regular.spec(), TrackKind::HighSpeed.spec());
        assert_eq!(r.level_mult, LEVEL_MULT);
        for i in 0..7 {
            assert_eq!(h.level_mult[i], LEVEL_MULT[i] * HSR_COST);
        }
        assert_eq!((TrackKind::Regular.v_cap(), TrackKind::HighSpeed.v_cap()), (V_TOP, V_HSR));
        for s in [r, h] {
            assert_eq!((s.section_s, s.conflict_s, s.platform_margin_s), (SECTION_BASE_S, CONFLICT_S, PLATFORM_MARGIN_S));
            assert_eq!(s.platform, (PLATFORM_MIN, PLATFORM_MAX));
            assert!(s.road_crossings && s.ramps_over && s.family == 0);
        }
        assert_eq!((r.upkeep, h.upkeep), (1.0, HSR_COST));
    }

    #[test]
    fn tram_rows_hold_annitas_prices() {
        let (s, t) = (TrackKind::StreetTram.spec(), TrackKind::Tram.spec());
        assert_eq!(s.level_mult, [1.6, 1.2, 0.6, 0.1, 0.6, 1.1, 1.6]);
        assert_eq!(t.level_mult, [1.6, 1.2, 0.6, 0.5, 0.6, 1.1, 1.6]);
        assert_eq!((s.upkeep, t.upkeep), (0.6, 1.0));
        assert_eq!((s.v_cap_kmh, t.v_cap_kmh), (60.0, 80.0));
        assert!(!s.road_crossings && !t.road_crossings);
        assert!(s.family == 1 && t.family == 1);
    }

    #[test]
    fn a_kind_reads_back_from_its_number() {
        for k in TrackKind::ALL {
            assert_eq!(TrackKind::from_u8(k as u8), Some(k));
        }
        assert_eq!(TrackKind::from_u8(TRACK_KINDS as u8), None);
    }

    #[test]
    fn tram_track_prices_by_its_own_table() {
        use super::super::cost::track_cost;
        let flat = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat, &[], 2, TrackKind::StreetTram) - 7.5).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat, &[], 2, TrackKind::Tram) - 37.5).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat, &[], 2, TrackKind::Regular) - 37.5).abs() < 1e-9);
        // T-2928: over water street tram track pays half water's 2.5, off-street tram track all of it
        let wet = [[0.0, 1000.0]];
        assert!((track_cost(1000.0, &flat, &wet, 2, TrackKind::StreetTram) - 75.0 * (0.1 + 1.25)).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat, &wet, 2, TrackKind::Tram) - 75.0 * (0.5 + 2.5)).abs() < 1e-9);
    }

    fn line_net(kinds: [TrackKind; 2]) -> super::super::net::Network {
        use super::super::net::{EdgeData, Network, NodeData};
        use super::super::frame::FrameKind;
        let mut net = Network::new_in(FrameKind::Merc, 0.0, 0.0, false);
        for x in [0.0f64, 500.0, 1000.0] {
            let n = net.alloc_node();
            net.set_node(n, Some(NodeData { x, y: 0.0, level: 0, flying: false })).unwrap();
        }
        for (k, kind) in kinds.into_iter().enumerate() {
            let e = net.alloc_edge();
            net.set_edge(e, Some(EdgeData { a: k as u32, b: k as u32 + 1, tracks: 2, pis: vec![], built: false, off: false, thru: vec![], crossover: false, kind, sep: false, day: 0, oneway: 0, join: [0; 2] })).unwrap();
        }
        net
    }

    #[test]
    fn rail_and_tram_track_may_not_meet_at_a_node() {
        use super::super::net::Issue;
        let mixed = line_net([TrackKind::Regular, TrackKind::Tram]);
        assert!(mixed.validate(&[], &[1], &[]).iter().any(|i| matches!(i, Issue::TrackFamily { node: 1 })));
        for pair in [[TrackKind::Regular, TrackKind::HighSpeed], [TrackKind::StreetTram, TrackKind::Tram]] {
            assert!(!line_net(pair).validate(&[], &[1], &[]).iter().any(|i| matches!(i, Issue::TrackFamily { .. })), "{pair:?}");
        }
    }
}
