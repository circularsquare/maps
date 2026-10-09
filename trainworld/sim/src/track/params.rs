//! Every tuning constant of the track model in one place (SPEC 6, notes/T-008.md, notes/T-040.md).

// ---- geometry and speed (SPEC 6.1) ----

/// Lateral acceleration allowed in curves, m/s² (LGV design: 7,000 m for 320 km/h).
pub const A_LAT: f64 = 1.1;
/// Smallest radius a player may build, m (38 km/h).
pub const MIN_RADIUS: f64 = 100.0;
/// Top speed of the M1 train, m/s (160 km/h). The track itself has no cap.
pub const V_TOP: f64 = 160.0 / 3.6;
/// Auto radius cap: the smallest whole-metre radius that already allows V_TOP
/// (ceil(44.44² / 1.1) = 1,796 m; T-008 rounds it to 1,800).
pub const R_CAP: f64 = 1796.0;
/// Positions and radii are quantised to this (1 mm) when the player places them, so derived
/// geometry is a deterministic f64 function of the save.
pub const QUANT: f64 = 0.001;
/// Every edge end at a node must leave along the node's heading or its reverse, within this (rad).
pub const HEADING_TOL: f64 = 0.003;
/// Slack when checking that two tangent lengths fit on one leg, m.
pub const FIT_TOL: f64 = 0.01;
/// Intersections closer than this to a node both edges share are the node itself, not a crossing.
pub const SHARED_NODE_CLEAR: f64 = 25.0;
/// Two double-track routes' tracks are this far apart (drawing only).
pub const TRACK_SPACING: f64 = 4.0;

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

/// Traction: 1.0 m/s² up to 60 km/h, 0.5 m/s² above (a constant-power EMU falls off with speed;
/// two bands keep every phase at constant acceleration, so keyframes stay exact). Braking 0.9 m/s²
/// (normal service braking; 1.1-1.3 is the usual maximum service rate). notes/T-040.md.
pub const ACCEL_LOW: f64 = 1.0;
pub const ACCEL_HIGH: f64 = 0.5;
pub const ACCEL_SWITCH_V: f64 = 60.0 / 3.6;
pub const BRAKE: f64 = 0.9;

// ---- costs, US$M (SPEC 6.4) ----

/// Per route-km of double track at multiplier 1.
pub const BASE_COST_PER_KM: f64 = 90.0;
/// Multiplier per level, index `level + 3` (-3 .. +3). Anita, 2026-10-08 (+2 and +3 lowered from
/// 1.4 and 2.0 the same day).
pub const LEVEL_MULT: [f64; 7] = [1.2, 1.1, 1.0, 0.3, 0.8, 1.3, 1.8];
pub const SINGLE_TRACK_COST: f64 = 0.6;
pub const WATER_COST: f64 = 2.0;
/// 200 m station with two platform tracks, per level, index `level + 3`.
pub const STATION_COST_200: [f64; 7] = [140.0, 100.0, 60.0, 10.0, 30.0, 40.0, 50.0];
pub const SINGLE_TRACK_STATION: f64 = 0.7;
pub const JUNCTION_KM: f64 = 0.25;
pub const FLYOVER_KM: f64 = 0.6;
pub const FLAT_CROSSING_KM: f64 = 0.1;
/// Water mask sampling step along track, m: half the pack mask's 25 m pixel (notes/T-004.md).
pub const WATER_STEP: f64 = 12.5;
/// Price of one 20 m car, US$M (T-028; a New York R211 car is about 2.6). Trains are bought when
/// the running lines need more cars than the network owns; spare cars are kept, never sold.
pub const CAR_PRICE: f64 = 2.5;
/// Running cost per car-km, US$. Set for play, not realism (Anita, 2026-10-09): T-028's real-ish
/// 6 (New York's subway spends about 9 a car-km all in) made running costs beat fares even at
/// 100x, so the real New York network lost $0.65B a game day. A quarter of that leaves it about
/// $1.1B a day ahead and a well-filled line paying back in days.
pub const RUN_COST_CAR_KM: f64 = 1.5;
/// The economy runs at 100x (T-081; Anita, 2026-10-09, as Subway Builder does): fares and running
/// costs per game day are this many times the real figures, so a line pays back in about a game
/// week. Construction and train prices stay real. The one constant: running costs here, fares in
/// app/src/game/money.ts (which reads it through `TrackApi.money_params`).
pub const ECONOMY: f64 = 100.0;

pub fn level_mult(level: i8) -> f64 {
    LEVEL_MULT[(level.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize]
}

// ---- capacity (SPEC 6.2) ----

pub const SECTION_BASE_S: f64 = 90.0;
/// Signalling braking rate behind the block length term `v / 0.6`.
pub const SIGNAL_BRAKE: f64 = 0.6;
pub const SINGLE_TRACK_MARGIN_S: f64 = 60.0;
pub const PLATFORM_MARGIN_S: f64 = 60.0;
pub const PASSING_S: f64 = 40.0;
pub const TERMINUS_MARGIN_S: f64 = 60.0;
pub const CONFLICT_S: f64 = 90.0;
pub const KINGMAN_C0: f64 = 0.2;
pub const RHO_KNEE: f64 = 0.9;
pub const OVERLOAD_WINDOW_S: f64 = 3600.0;
/// Delays are rounded to whole seconds; under this they are dropped.
pub const HOLD_MIN_S: f64 = 2.0;
/// A hold point closer than this to the previous stop becomes extra dwell there.
pub const HOLD_NEAR_STOP_M: f64 = 50.0;
/// Length of the slow approach for a short hold, m.
pub const CRAWL_ZONE_M: f64 = 300.0;
/// The train stops this far before a conflict point (plus half its length).
pub const HOLD_MARGIN_M: f64 = 20.0;

// ---- service (SPEC 6.3, 6.5) ----

/// A station-to-station time moving this much or more wakes the city's demand.
pub const DEMAND_TIME_EPS_S: f64 = 5.0;
/// Demand levels: index into a line's `tph`.
pub const HIGH: usize = 0;
pub const MEDIUM: usize = 1;
pub const LOW: usize = 2;
pub const LEVELS: usize = 3;
/// The five periods of SPEC 4.2 in the demand kernel's order (morning peak, midday, evening
/// peak, evening, night): start and end hour, and which demand level schedules them.
pub const PERIOD_HOURS: [(f64, f64); 5] = [(6.0, 10.0), (10.0, 16.0), (16.0, 20.0), (20.0, 24.0), (0.0, 6.0)];
pub const PERIOD_LEVEL: [usize; 5] = [HIGH, MEDIUM, HIGH, MEDIUM, LOW];

// ---- render tiles (SPEC 6.5) ----

pub const TILE_ZOOM: u32 = 12;
pub const EARTH_R: f64 = 6371008.8;

/// Round to the placement grid (1 mm).
pub fn quant(v: f64) -> f64 {
    (v / QUANT).round() * QUANT
}
