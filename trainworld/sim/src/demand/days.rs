//! Kinds of day (T-260, T-262, docs/demand-timing.md): weekday, Saturday, Sunday and a
//! superholiday's away day (T-264).
//!
//! The solve is a weekday's. A Saturday or a Sunday is worked out from it, not solved: every
//! purpose's trips are a fixed multiple of a weekday's, departing by their own period shares, and
//! without crowding a trip's mode and route depend on its period's service only, not on how many
//! make it. So a Saturday's riders in period q are the weekday's riders of each purpose in that
//! period times `load_factor(SATURDAY, purpose, q)`, and the queries (bubbles, catchments,
//! selections) weigh each period the same way. Both keep the weekday's crowded route and mode
//! choices (fewer riders would crowd less), and the same trains run every day. The average day
//! (T-464) is 5 weekdays, a Saturday and a Sunday over 7, taken on the shares and rates, so every
//! figure that is linear in them is the same blend.
//!
//! Sources (national, so every US city alike): NHTS 2017 person trips by purpose, weekday
//! (Mon-Fri) against Saturday and against Sunday, person-weighted, linked trips, each rate the mean
//! of the outward and return ratios; England's NTS 0504b agrees on the weekend as a whole
//! (commuting 0.28, shopping 1.26, leisure 1.57). Airport trips follow TSA checkpoint throughput
//! by day of week, 2025 (Saturday 0.92, Sunday 1.11 of a weekday). A day's period shares are the
//! model's weekday shares times NHTS's day/weekday ratio in each period, renormalised
//! (research/weekend/tables.py); school, university, airport and medical keep their weekday
//! timing (NHTS's weekend counts for them are a few hundred trips). notes/T-170.md.

use super::api::{TO_HOME, TO_WORK};
use super::everyday::{OUTWARD, RETURN};
use super::intercity::SHARES;
use super::network::PERIODS;
use wasm_bindgen::prelude::*;

/// The days a query can be asked for: the four kinds, then the average over a week (T-464).
pub const DAY_KINDS: usize = 5;
pub const WEEKDAY: usize = 0;
pub const SATURDAY: usize = 1;
pub const SUNDAY: usize = 2;
/// A superholiday's days away (T-264): a Sunday with people gone home and visitors in town.
pub const AWAY: usize = 3;
pub const AVERAGE: usize = 4;
/// Weekdays, Saturdays and Sundays in a week, the average's weights.
const WEEK: [f32; 3] = [5.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0];
/// Purposes as the demand results split them (demand.worker.ts `purposeParts`): work, the
/// everyday layer's seven in its own order (shopping, social, school, university, airport,
/// medical, attractions), then the regional market's longer trips.
pub const PURPOSES: usize = 9;
pub const WORK: usize = 0;
pub const LONGER: usize = 8;
/// The purpose index of an everyday kind (`everyday.rs` `RATE` order).
pub const fn everyday(kind: usize) -> usize {
    1 + kind
}

/// A Saturday's and a Sunday's trips per weekday's, by purpose (NHTS 2017; airport TSA 2025;
/// attractions social's, T-263).
pub const DAY_RATE: [[f32; PURPOSES]; 2] = [
    [0.37, 1.39, 1.54, 0.11, 0.30, 0.92, 0.30, 1.54, 1.69],
    [0.24, 1.05, 1.67, 0.06, 0.14, 1.11, 0.18, 1.67, 1.48],
];

/// An away day's trips per Sunday's, by purpose (T-264, notes/T-264.md): local outings 0.7 with
/// people gone home, attractions 1.5 with visitors in town; work, airport and the longer trips keep
/// a Sunday's. Not in the average day.
pub const AWAY_RATE: [f32; PURPOSES] = [1.0, 0.7, 0.7, 0.7, 0.7, 1.0, 0.7, 1.5, 1.0];

type Shares =[Option<[f32; PERIODS]>; PURPOSES];
/// Social outings' Saturday and Sunday period shares, which attractions take too.
const SOCIAL_OUT: [Option<[f32; PERIODS]>; 2] = [Some([0.059, 0.440, 0.335, 0.160, 0.006]), Some([0.096, 0.516, 0.276, 0.108, 0.004])];
const SOCIAL_RET: [Option<[f32; PERIODS]>; 2] = [Some([0.016, 0.310, 0.395, 0.221, 0.058]), Some([0.014, 0.456, 0.368, 0.147, 0.015])];
/// A Saturday's and a Sunday's period shares, outward (to work, to the activity; the market's one
/// share), periods in `PERIOD_HOURS` order. `None`: the weekday's.
const DAY_OUT: [Shares; 2] = [
    [
        Some([0.586, 0.150, 0.115, 0.063, 0.086]),
        Some([0.110, 0.623, 0.228, 0.031, 0.008]),
        Some([0.059, 0.440, 0.335, 0.160, 0.006]),
        None,
        None,
        None,
        None,
        SOCIAL_OUT[0],
        Some([0.166, 0.314, 0.196, 0.295, 0.029]),
    ],
    [
        Some([0.483, 0.177, 0.117, 0.127, 0.096]),
        Some([0.082, 0.643, 0.232, 0.035, 0.008]),
        Some([0.096, 0.516, 0.276, 0.108, 0.004]),
        None,
        None,
        None,
        None,
        SOCIAL_OUT[1],
        Some([0.133, 0.394, 0.285, 0.166, 0.022]),
    ],
];
const DAY_RET: [Shares; 2] = [
    [
        Some([0.040, 0.184, 0.371, 0.304, 0.101]),
        Some([0.070, 0.572, 0.283, 0.067, 0.008]),
        Some([0.016, 0.310, 0.395, 0.221, 0.058]),
        None,
        None,
        None,
        None,
        SOCIAL_RET[0],
        None,
    ],
    [
        Some([0.051, 0.163, 0.359, 0.267, 0.160]),
        Some([0.060, 0.599, 0.271, 0.065, 0.005]),
        Some([0.014, 0.456, 0.368, 0.147, 0.015]),
        None,
        None,
        None,
        None,
        SOCIAL_RET[1],
        None,
    ],
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
    match day {
        WEEKDAY => 1.0,
        AVERAGE => WEEK[0] + WEEK[1] * DAY_RATE[0][p] + WEEK[2] * DAY_RATE[1][p],
        AWAY => AWAY_RATE[p] * DAY_RATE[1][p],
        d => DAY_RATE[d.min(SUNDAY) - 1][p],
    }
}

/// Departures of purpose `p` in period `q` on `day` per weekday trip, from a weekday's share `wd`
/// and the day tables `tables` (outward or return).
fn share(day: usize, p: usize, q: usize, wd: f32, tables: &[Shares; 2]) -> f32 {
    let on = |d: usize| DAY_RATE[d][p] * tables[d][p].map_or(wd, |s| s[q]);
    match day {
        WEEKDAY => wd,
        AVERAGE => WEEK[0] * wd + WEEK[1] * on(0) + WEEK[2] * on(1),
        AWAY => AWAY_RATE[p] * on(1),
        d => on(d.min(SUNDAY) - 1),
    }
}

/// Outward trips of purpose `p` departing in period `q` on `day`, as a share of a weekday's
/// outward trips: on a weekday exactly the model's own table.
pub fn share_out(day: usize, p: usize, q: usize) -> f32 {
    share(day, p, q, weekday_shares(p, q).0, &DAY_OUT)
}

/// As `share_out` for the trips home.
pub fn share_ret(day: usize, p: usize, q: usize) -> f32 {
    share(day, p, q, weekday_shares(p, q).1, &DAY_RET)
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
    fn day_shares_sum_to_one_and_weekday_is_the_tables() {
        for (k, day) in [SATURDAY, SUNDAY].into_iter().enumerate() {
            for p in 0..PURPOSES {
                for t in [&DAY_OUT[k][p], &DAY_RET[k][p]].into_iter().flatten() {
                    assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5, "day {day} purpose {p}");
                }
                let out: f32 = (0..PERIODS).map(|q| share_out(day, p, q)).sum();
                let wd: f32 = (0..PERIODS).map(|q| share_out(WEEKDAY, p, q)).sum();
                assert!((out - DAY_RATE[k][p] * wd).abs() < 1e-4, "day {day} purpose {p}");
                // the day's riders over a weekday's are the rate, both directions together
                let (mut wd, mut dd) = (0.0, 0.0);
                for q in 0..PERIODS {
                    let (o, r) = weekday_shares(p, q);
                    wd += o + r;
                    dd += (o + r) * load_factor(day, p, q);
                }
                assert!((dd / wd - DAY_RATE[k][p]).abs() < 1e-4, "day {day} purpose {p}");
            }
        }
        for p in 0..PURPOSES {
            for q in 0..PERIODS {
                assert_eq!(load_factor(WEEKDAY, p, q), 1.0);
            }
        }
        assert_eq!(share_out(WEEKDAY, WORK, 0), TO_WORK[0]);
        assert_eq!(share_ret(WEEKDAY, everyday(1), 3), RETURN[1][3]);
    }

    #[test]
    fn an_away_day_is_a_scaled_sunday() {
        for p in 0..PURPOSES {
            assert!((day_rate(AWAY, p) - AWAY_RATE[p] * day_rate(SUNDAY, p)).abs() < 1e-6, "purpose {p}");
            for q in 0..PERIODS {
                assert!((load_factor(AWAY, p, q) - AWAY_RATE[p] * load_factor(SUNDAY, p, q)).abs() < 1e-5, "purpose {p} period {q}");
            }
        }
        assert_eq!(AWAY_RATE[everyday(1)], 0.7);
        assert_eq!(AWAY_RATE[everyday(6)], 1.5);
        assert_eq!(AWAY_RATE[WORK], 1.0);
        assert_eq!(AWAY_RATE[LONGER], 1.0);
    }

    #[test]
    fn the_average_day_is_five_weekdays_a_saturday_and_a_sunday_over_seven() {
        for p in 0..PURPOSES {
            assert!((day_rate(AVERAGE, p) - (5.0 + DAY_RATE[0][p] + DAY_RATE[1][p]) / 7.0).abs() < 1e-6, "purpose {p}");
            for q in 0..PERIODS {
                let want = (5.0 + load_factor(SATURDAY, p, q) + load_factor(SUNDAY, p, q)) / 7.0;
                assert!((load_factor(AVERAGE, p, q) - want).abs() < 1e-5, "purpose {p} period {q}");
            }
        }
    }
}
