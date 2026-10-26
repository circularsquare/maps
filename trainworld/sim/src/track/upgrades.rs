//! Station upgrades (T-1281, notes/T-1281.md): bits of a station's saved flags, their prices and
//! the turning rules they change. Bought on a blueprint station with it, or paid at once on a
//! constructed one; removing a constructed one refunds nothing.

use super::params::{level_mult, BASE_COST_PER_KM, WATER_COST};

/// Turnback sidings past the platform: a line turning here holds its platforms only for the dwell.
pub const TERMINAL: u8 = 1;
/// Car park, bus bays and bike racks: a ride to the station costs `PARK_RIDE_MIN` perceived minutes less.
pub const PARK_RIDE: u8 = 2;
/// Waits at the station count at `WAITING_ROOM_WAIT` of their minutes in mode choice.
pub const WAITING_ROOM: u8 = 4;
pub const ALL: u8 = TERMINAL | PARK_RIDE | WAITING_ROOM;

/// The terminal's price: this much double track at the station's level.
pub const TERMINAL_KM: f64 = 1.0;
pub const PARK_RIDE_COST: f64 = 20.0;
pub const WAITING_ROOM_COST: f64 = 5.0;
pub const PARK_RIDE_MIN: f32 = 10.0;
pub const WAITING_ROOM_WAIT: f32 = 0.8;

/// A passage between two stations (the player's link, T-197): a base price plus this a km of the
/// straight line between them, at most `PASSAGE_MAX_M` long; changing through it walks that line
/// at `PASSAGE_MPS` (moving walkways).
pub const PASSAGE_BASE: f64 = 10.0;
pub const PASSAGE_PER_KM: f64 = 90.0;
pub const PASSAGE_MAX_M: f64 = 1000.0;
pub const PASSAGE_MPS: f32 = 2.0;

/// A passage's price, US$M, for its straight line of `m` metres.
pub fn passage_price(m: f64) -> f64 {
    PASSAGE_BASE + PASSAGE_PER_KM * m / 1000.0
}

/// Sidings a terminal turns trains in, pooled.
pub const TERMINAL_SIDINGS: f64 = 2.0;
/// The reversing-partway limiter's hold per train at a station with a terminal: 30 an hour a side.
pub const TERMINAL_REVERSAL_S: f64 = 120.0;
/// Least a turn holds, summed over the tracks it is pooled across (turnaround + margin, at least
/// this): 18 an hour on one track, 36 on two (the Victoria line's two-platform termini), so a
/// short turnaround never turns more than a real terminus does.
pub const POOLED_TURN_MIN_S: f64 = 200.0;

/// Occupancy per train of a turn pooled over `tracks` (a dead end's platform tracks, a terminal's
/// sidings), from the line's turnaround plus `margin`.
pub fn pooled_turn_s(turnaround_s: f64, margin_s: f64, tracks: f64) -> f64 {
    (turnaround_s + margin_s).max(POOLED_TURN_MIN_S) / tracks.max(1.0)
}

/// One upgrade's price, US$M.
pub fn price(bit: u8, level: i8, wet: bool) -> f64 {
    match bit {
        TERMINAL => TERMINAL_KM * BASE_COST_PER_KM * level_mult(level) * if wet { WATER_COST } else { 1.0 },
        PARK_RIDE => PARK_RIDE_COST,
        WAITING_ROOM => WAITING_ROOM_COST,
        _ => 0.0,
    }
}

/// All of a station's upgrades, US$M.
pub fn cost(bits: u8, level: i8, wet: bool) -> f64 {
    [TERMINAL, PARK_RIDE, WAITING_ROOM].iter().filter(|&&b| bits & b != 0).map(|&b| price(b, level, wet)).sum()
}
