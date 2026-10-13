//! Kinds of day (T-260, docs/demand.md 4.2): weekday and weekend.
//!
//! The solve is a weekday's. A weekend is worked out from it, not solved: every purpose's trips
//! are a fixed multiple of a weekday's, departing by their own period shares, and without
//! crowding a trip's mode and route depend on its period's service only, not on how many make
//! it. So a weekend's riders in period q are the weekday's riders of each purpose in that period
//! times `load_factor(WEEKEND, purpose, q)`, and the queries (bubbles, catchments, selections)
//! weigh each period the same way. The weekend keeps the weekday's crowded route and mode
//! choices (fewer riders at the weekend would crowd less), and the same trains run every day.
//!
//! Sources (national, so every US city alike): NHTS 2017 person trips by purpose, weekday
//! (Mon-Fri) against Saturday and Sunday pooled, person-weighted, linked trips; England's NTS
//! 0504b agrees (commuting 0.28, shopping 1.26, leisure 1.57). Airport trips follow TSA checkpoint
//! throughput by day of week, 2025 (weekend 1.01 of a weekday). Weekend period shares are the
//! model's weekday shares times NHTS's weekend/weekday ratio in each period, renormalised; school,
//! university, airport and medical keep their weekday timing (NHTS's weekend counts for them are a
//! few hundred trips). notes/T-170.md.

use super::api::{TO_HOME, TO_WORK};
use super::everyday::{OUTWARD, RETURN};
use super::intercity::SHARES;
use super::network::PERIODS;
use wasm_bindgen::prelude::*;

pub const DAY_KINDS: usize = 2;
pub const WEEKDAY: usize = 0;
pub const WEEKEND: usize = 1;
/// Purposes as the demand results split them (demand.worker.ts `purposeParts`): work, the
/// everyday layer's six in its own order (shopping, social, school, university, airport,
/// medical), then the regional market's longer trips.
pub const PURPOSES: usize = 8;
pub const WORK: usize = 0;
pub const LONGER: usize = 7;
/// The purpose index of an everyday kind (`everyday.rs` `RATE` order).
pub const fn everyday(kind: usize) -> usize {
    1 + kind
}

/// A weekend day's trips per weekday's, by purpose (NHTS 2017; airport TSA 2025).
pub const WEEKEND_RATE: [f32; PURPOSES] = [0.30, 1.22, 1.60, 0.08, 0.22, 1.01, 0.24, 1.58];

/// Weekend period shares, outward (to work, to the activity; the market's one share), periods in
/// `PERIOD_HOURS` order. `None`: the weekday's.
const WEEKEND_OUT: [Option<[f32; PERIODS]>; PURPOSES] = [
    Some([0.546, 0.161, 0.116, 0.087, 0.090]),
    Some([0.098, 0.632, 0.230, 0.032, 0.008]),
    Some([0.076, 0.478, 0.307, 0.134, 0.005]),
    None,
    None,
    None,
    None,
    Some([0.151, 0.349, 0.235, 0.239, 0.026]),
];
const WEEKEND_RET: [Option<[f32; PERIODS]>; PURPOSES] = [
    Some([0.045, 0.175, 0.366, 0.289, 0.125]),
    Some([0.066, 0.586, 0.277, 0.066, 0.005]),
    Some([0.015, 0.388, 0.381, 0.182, 0.034]),
    None,
    None,
    None,
    None,
    None,
];

/// A weekday's period shares of purpose `p`: (outward, return). The market has one share.
fn weekday_shares(p: usize, q: usize) -> (f32, f32) {
    match p {
        WORK => (TO_WORK[q], TO_HOME[q]),
        LONGER => (SHARES[q], 0.0),
        k => (OUTWARD[k - 1][q], RETURN[k - 1][q]),
    }
}

/// Trips of purpose `p` a day on `day` per trip on a weekday.
pub fn day_rate(day: usize, p: usize) -> f32 {
    if day == WEEKDAY { 1.0 } else { WEEKEND_RATE[p] }
}

/// Outward trips of purpose `p` departing in period `q` on `day`, as a share of a weekday's
/// outward trips: on a weekday exactly the model's own table.
pub fn share_out(day: usize, p: usize, q: usize) -> f32 {
    let wd = weekday_shares(p, q).0;
    if day == WEEKDAY {
        return wd;
    }
    WEEKEND_RATE[p] * WEEKEND_OUT[p].map_or(wd, |s| s[q])
}

/// As `share_out` for the trips home.
pub fn share_ret(day: usize, p: usize, q: usize) -> f32 {
    let wd = weekday_shares(p, q).1;
    if day == WEEKDAY {
        return wd;
    }
    WEEKEND_RATE[p] * WEEKEND_RET[p].map_or(wd, |s| s[q])
}

/// Riders of purpose `p` in period `q` on `day` per rider on a weekday, both directions
/// together: exactly 1 on a weekday.
pub fn load_factor(day: usize, p: usize, q: usize) -> f32 {
    if day == WEEKDAY {
        return 1.0;
    }
    let (o, r) = weekday_shares(p, q);
    if o + r <= 0.0 {
        return 0.0;
    }
    (share_out(day, p, q) + share_ret(day, p, q)) / (o + r)
}

/// Every load factor for the main thread: `[day][purpose][period]`, then `day_rate` per day and
/// purpose (`DAY_KINDS * PURPOSES` more).
#[wasm_bindgen]
pub fn day_factors() -> Vec<f32> {
    let mut out = vec![];
    for d in 0..DAY_KINDS {
        for p in 0..PURPOSES {
            for q in 0..PERIODS {
                out.push(load_factor(d, p, q));
            }
        }
    }
    for d in 0..DAY_KINDS {
        for p in 0..PURPOSES {
            out.push(day_rate(d, p));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekend_shares_sum_to_one_and_weekday_is_the_tables() {
        for p in 0..PURPOSES {
            for t in [&WEEKEND_OUT[p], &WEEKEND_RET[p]].into_iter().flatten() {
                assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5, "purpose {p}");
            }
            let out: f32 = (0..PERIODS).map(|q| share_out(WEEKEND, p, q)).sum();
            let wd: f32 = (0..PERIODS).map(|q| share_out(WEEKDAY, p, q)).sum();
            assert!((out - WEEKEND_RATE[p] * wd).abs() < 1e-4, "purpose {p}");
            for q in 0..PERIODS {
                assert_eq!(load_factor(WEEKDAY, p, q), 1.0);
            }
        }
        assert_eq!(share_out(WEEKDAY, WORK, 0), TO_WORK[0]);
        assert_eq!(share_ret(WEEKDAY, everyday(1), 3), RETURN[1][3]);
        // the day's riders: weekend over weekday is the rate, both directions together
        for p in 0..PURPOSES {
            let (mut wd, mut we) = (0.0, 0.0);
            for q in 0..PERIODS {
                let (o, r) = weekday_shares(p, q);
                wd += o + r;
                we += (o + r) * load_factor(WEEKEND, p, q);
            }
            assert!((we / wd - WEEKEND_RATE[p]).abs() < 1e-4, "purpose {p}");
        }
    }
}
