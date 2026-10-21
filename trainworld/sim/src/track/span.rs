//! Each stop's service span for demand (T-842, notes/T-842.md, docs/demand-kernel.md step 4).
//!
//! Riders differ in how long they would wait for the next train rather than go another way: the
//! tolerance is spread log-uniformly between `TOL_MIN_S` and `TOL_MAX_S`. For each line, direction,
//! stop and period, `phi` is the share of the period's riders (arriving by the boarding account's
//! shape, in the line's local time) whose wait for the next departure is within their tolerance,
//! from a steady day's departures (no eras). 1 wherever a train is never more than `TOL_MIN_S` away.

use super::boarding::{shape_cum, shape_strength, PERIODS, SHAPE};
use super::params::*;
use super::service::{lattice_deps_local, Lattice, LineSvc};

/// The shortest and longest wait any rider tolerates, seconds (Anita, 2026-10-21: "ramp from like
/// 0.5-6 hrs"). Log-uniform between: a business rider and a leisure one differ by a factor.
pub const TOL_MIN_S: f64 = 1800.0;
pub const TOL_MAX_S: f64 = 6.0 * 3600.0;
/// The step the tolerance ramp is summed in, seconds.
const STEP_S: f64 = 60.0;
const DAY: f64 = 86400.0;

/// The share of riders who would wait `wait` seconds.
pub fn within(wait: f64) -> f64 {
    if wait <= TOL_MIN_S {
        1.0
    } else if wait >= TOL_MAX_S {
        0.0
    } else {
        (TOL_MAX_S / wait).ln() / (TOL_MAX_S / TOL_MIN_S).ln()
    }
}

/// One stop's day as the tolerance sees it: from `t[i]` to the next start (the last to the end of
/// the day) a share `s[i]` of the riders arriving have a train due within their tolerance; the
/// segments split at period boundaries, and `phi` per period is that share over the period's riders
/// (unrounded). The boarding account spreads riders by it (T-842b).
#[derive(Clone, Debug, Default)]
pub struct Mask {
    pub t: Vec<f64>,
    pub s: Vec<f64>,
    pub phi: [f64; PERIODS],
}

/// The period's arrival shape: the share of its riders arrived by `t` (seconds of the day).
pub fn period_cum(q: usize, t: f64) -> f64 {
    let (h0, h1) = PERIOD_HOURS[q];
    let c = shape_strength(SHAPE[q], h1 - h0, PEAK_HOUR_FACTOR[q] as f64);
    shape_cum(SHAPE[q], c, ((t - h0 * 3600.0) / ((h1 - h0) * 3600.0)).clamp(0.0, 1.0))
}

/// `Mask` for one stop from its departures in a day (seconds of local time, sorted, in [0, DAY)).
pub fn stop_mask(deps: &[f64]) -> Mask {
    let ext: Vec<f64> = deps.iter().copied().chain(deps.iter().map(|d| d + DAY)).collect();
    let mut seg: Vec<(f64, f64)> = vec![];
    let mut phi = [0.0; PERIODS];
    let mut order: [usize; PERIODS] = std::array::from_fn(|q| q);
    order.sort_by(|&a, &b| PERIOD_HOURS[a].0.total_cmp(&PERIOD_HOURS[b].0));
    for q in order {
        let (a, b) = (PERIOD_HOURS[q].0 * 3600.0, PERIOD_HOURS[q].1 * 3600.0);
        // a new period starts a segment of its own
        let mut fresh = true;
        let mut push = |u0: f64, u1: f64, s: f64, tot: &mut f64| {
            if u1 > u0 {
                *tot += (period_cum(q, u1) - period_cum(q, u0)) * s;
                match seg.last() {
                    Some(&(_, x)) if x == s && !fresh => {}
                    _ => seg.push((u0, s)),
                }
                fresh = false;
            }
        };
        let mut j = ext.partition_point(|&d| d <= a);
        let (mut t, mut tot) = (a, 0.0f64);
        while t < b {
            let Some(&d) = ext.get(j) else {
                push(t, b, 0.0, &mut tot);
                break;
            };
            let end = d.min(b);
            push(t, end.min(d - TOL_MAX_S), 0.0, &mut tot);
            let (p0, p1) = (t.max(d - TOL_MAX_S), end.min(d - TOL_MIN_S));
            if p1 > p0 {
                let m = ((p1 - p0) / STEP_S).ceil().max(1.0) as usize;
                let h = (p1 - p0) / m as f64;
                for i in 0..m {
                    let u0 = p0 + i as f64 * h;
                    push(u0, u0 + h, within(d - u0 - 0.5 * h), &mut tot);
                }
            }
            // a train due within the shortest tolerance: everyone
            push(t.max(d - TOL_MIN_S), end, 1.0, &mut tot);
            t = d;
            j += 1;
        }
        phi[q] = tot.clamp(0.0, 1.0);
    }
    Mask { t: seg.iter().map(|x| x.0).collect(), s: seg.iter().map(|x| x.1).collect(), phi }
}

/// `phi` per period for one stop (`stop_mask`), rounded to hundredths so a share that does not
/// move compares equal to the bit.
pub fn stop_phi(deps: &[f64]) -> [f32; PERIODS] {
    stop_mask(deps).phi.map(|p| ((p * 100.0).round() / 100.0) as f32)
}

/// Each stop's departures in a steady day of pattern `lat` for run `r`, local time, sorted.
pub fn stop_deps(ls: &LineSvc, lat: &Lattice, r: usize) -> Vec<Vec<f64>> {
    let mut first = vec![];
    lattice_deps_local(lat, r, DAY, 2.0 * DAY, &mut first);
    (0..ls.stops.len())
        .map(|k| {
            let mut deps: Vec<f64> = first.iter().map(|&(d, lev, _)| (d + ls.runs[r].prof[lev as usize].dep.get(k).copied().unwrap_or(0.0)).rem_euclid(DAY)).collect();
            deps.sort_by(f64::total_cmp);
            deps
        })
        .collect()
}

/// A line's `phi`, `[(period * 2 + run) * n + stop]` with each run's stops in its travel order (the
/// boarding account's layout), from its steady pattern `lat`.
pub fn line_span(ls: &LineSvc, lat: &Lattice) -> Vec<f32> {
    let n = ls.stops.len();
    let mut out = vec![0.0f32; PERIODS * 2 * n];
    for r in 0..2 {
        for (k, deps) in stop_deps(ls, lat, r).iter().enumerate() {
            for (q, phi) in stop_phi(deps).into_iter().enumerate() {
                out[(q * 2 + r) * n + k] = phi;
            }
        }
    }
    out
}

/// What `line_span` reads, hashed (FNV-1a), so the clock worker works out again only lines whose
/// departures changed.
pub fn span_key(ls: &LineSvc, lat: &Lattice) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut add = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    add(ls.stops.len() as u64);
    for lev in 0..LEVELS {
        add(lat.tph[lev].to_bits() as u64);
        add(lat.off[lev].to_bits());
    }
    add(lat.lead.to_bits());
    for r in 0..2 {
        for lev in 0..LEVELS {
            for &v in ls.runs[r].prof[lev].dep.iter() {
                add(v.to_bits());
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every(hw: f64, from: f64, to: f64) -> Vec<f64> {
        let mut v = vec![];
        let mut t = from;
        while t < to {
            v.push(t);
            t += hw;
        }
        v
    }

    #[test]
    fn the_tolerance_ramps_from_half_an_hour_to_six() {
        assert_eq!(within(600.0), 1.0);
        assert_eq!(within(TOL_MIN_S), 1.0);
        assert!((within((TOL_MIN_S * TOL_MAX_S).sqrt()) - 0.5).abs() < 1e-12, "half at the geometric middle");
        assert_eq!(within(TOL_MAX_S), 0.0);
    }

    #[test]
    fn a_line_all_day_has_every_rider_and_a_gap_loses_them() {
        // every 20 minutes round the clock: 1 everywhere, exactly
        assert_eq!(stop_phi(&every(1200.0, 0.0, DAY)), [1.0; PERIODS]);
        // peaks only, 06-10 and 16-20 with a last train at each end, every 15 minutes
        let mut peak = every(900.0, 6.0 * 3600.0, 10.0 * 3600.0 + 1.0);
        peak.extend(every(900.0, 16.0 * 3600.0, 20.0 * 3600.0 + 1.0));
        let phi = stop_phi(&peak);
        assert_eq!(phi[0], 1.0, "{phi:?}");
        assert_eq!(phi[2], 1.0, "{phi:?}");
        // midday: waits from 6 h at 10:00 down to none at 16:00, 0.37 with riders arriving evenly
        assert!(phi[1] > 0.25 && phi[1] < 0.45, "{phi:?}");
        // evening: 10 h to the next morning for everyone after 20:00
        assert_eq!(phi[3], 0.0, "{phi:?}");
        // night: the same 6 h ramp up to the first train at 06:00
        assert!(phi[4] > 0.25 && phi[4] < 0.55, "{phi:?}");
        // no departures: nobody
        assert_eq!(stop_phi(&[]), [0.0; PERIODS]);
    }
}
