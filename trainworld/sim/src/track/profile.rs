//! Run-time profiles (SPEC 2, 6.2, 6.3): the fastest motion along a line's path under speed
//! limits, stopping at every stop, with capacity delays baked in as holds.
//!
//! The result is a list of constant-acceleration phases `(t, s, v, a)`: from time `t` the train is
//! at path offset `s` with speed `v` and acceleration `a` until the next phase starts. Those are
//! the keyframes (notes/T-040.md): position at any time is closed form.
//!
//! Method: per stretch between two standstills, the minimum-time profile is the pointwise minimum
//! of the limit, the forward acceleration envelope and the backward braking envelope. In v² against
//! distance both envelopes are straight lines (two slopes for the two traction bands), so every
//! crossing point is solved exactly.

use super::params::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase {
    pub t: f64,
    pub s: f64,
    pub v: f64,
    pub a: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub phases: Vec<Phase>,
    /// Arrival and departure time at each stop, seconds from the trip's start (standing at the
    /// first stop). `dep[0]` is later than 0 only if a hold was put there.
    pub arr: Vec<f64>,
    pub dep: Vec<f64>,
    /// Arrival at the last stop.
    pub duration: f64,
    /// Dwell at intermediate stops (scheduled, without holds).
    pub dwell: f64,
    /// Capacity delay actually added (holds).
    pub hold: f64,
}

/// A capacity delay to absorb at path offset `s`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub s: f64,
    pub delay: f64,
}

const VS2: f64 = ACCEL_SWITCH_V * ACCEL_SWITCH_V;

/// Distance to accelerate from u² to v².
fn acc_dist(u2: f64, v2: f64) -> f64 {
    if v2 <= u2 {
        0.0
    } else if u2 >= VS2 {
        (v2 - u2) / (2.0 * ACCEL_HIGH)
    } else if v2 <= VS2 {
        (v2 - u2) / (2.0 * ACCEL_LOW)
    } else {
        (VS2 - u2) / (2.0 * ACCEL_LOW) + (v2 - VS2) / (2.0 * ACCEL_HIGH)
    }
}

/// v² after accelerating `d` metres from u².
fn acc_v2(u2: f64, d: f64) -> f64 {
    if u2 >= VS2 {
        return u2 + 2.0 * ACCEL_HIGH * d;
    }
    let d1 = (VS2 - u2) / (2.0 * ACCEL_LOW);
    if d <= d1 {
        u2 + 2.0 * ACCEL_LOW * d
    } else {
        VS2 + 2.0 * ACCEL_HIGH * (d - d1)
    }
}

fn push(out: &mut Vec<Phase>, t: f64, s: f64, v: f64, a: f64) {
    if let Some(l) = out.last() {
        if l.a == a && (l.v + l.a * (t - l.t) - v).abs() < 1e-6 {
            return;
        }
    }
    out.push(Phase { t, s, v, a });
}

/// Accelerate from offset x0 (speed² u2) to x1.
fn accel(out: &mut Vec<Phase>, mut t: f64, x0: f64, x1: f64, u2: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let v0 = u2.max(0.0).sqrt();
    if u2 < VS2 {
        let d1 = (VS2 - u2) / (2.0 * ACCEL_LOW);
        let e = x1.min(x0 + d1);
        let v1 = acc_v2(u2, e - x0).sqrt();
        push(out, t, x0, v0, ACCEL_LOW);
        t += (v1 - v0) / ACCEL_LOW;
        if x1 > x0 + d1 {
            let v2 = acc_v2(u2, x1 - x0).sqrt();
            push(out, t, x0 + d1, ACCEL_SWITCH_V, ACCEL_HIGH);
            t += (v2 - ACCEL_SWITCH_V) / ACCEL_HIGH;
        }
    } else {
        let v1 = acc_v2(u2, x1 - x0).sqrt();
        push(out, t, x0, v0, ACCEL_HIGH);
        t += (v1 - v0) / ACCEL_HIGH;
    }
    t
}

/// Brake from x0 to x1, ending at speed² w2.
fn brake(out: &mut Vec<Phase>, t: f64, x0: f64, x1: f64, w2: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let v0 = (w2 + 2.0 * BRAKE * (x1 - x0)).sqrt();
    let v1 = w2.max(0.0).sqrt();
    push(out, t, x0, v0, -BRAKE);
    t + (v0 - v1) / BRAKE
}

/// From standstill at the first interval's start to standstill at the last one's end, through
/// intervals `(x0, x1, v_limit)`. Returns the end time.
fn segment(iv: &[(f64, f64, f64)], mut t: f64, out: &mut Vec<Phase>) -> f64 {
    let m = iv.len();
    if m == 0 {
        return t;
    }
    let mut fw = vec![0.0; m];
    let mut f = 0.0f64;
    for k in 0..m {
        let v2 = iv[k].2 * iv[k].2;
        let u = f.min(v2);
        fw[k] = u;
        f = acc_v2(u, iv[k].1 - iv[k].0).min(v2);
    }
    let mut bend = vec![0.0; m];
    let mut b = 0.0f64;
    for k in (0..m).rev() {
        let v2 = iv[k].2 * iv[k].2;
        let w = b.min(v2);
        bend[k] = w;
        b = (w + 2.0 * BRAKE * (iv[k].1 - iv[k].0)).min(v2);
    }
    for k in 0..m {
        let (xa, xb, vl) = iv[k];
        let (v2, u2, w2, len) = (vl * vl, fw[k], bend[k], xb - xa);
        let s_fv = xa + acc_dist(u2, v2);
        let s_bv = xb - (v2 - w2) / (2.0 * BRAKE);
        if s_fv <= s_bv {
            t = accel(out, t, xa, s_fv, u2);
            if s_bv > s_fv {
                push(out, t, s_fv, vl, 0.0);
                t += (s_bv - s_fv) / vl;
            }
            t = brake(out, t, s_bv, xb, w2);
        } else {
            let b_at_xa = w2 + 2.0 * BRAKE * len;
            let sx = if u2 >= b_at_xa {
                xa
            } else if acc_v2(u2, len) <= w2 {
                xb
            } else {
                // acc_v2(u2, d) = w2 + 2B(len - d), solved per traction band.
                let rhs = w2 + 2.0 * BRAKE * len;
                let d1 = if u2 < VS2 { (VS2 - u2) / (2.0 * ACCEL_LOW) } else { 0.0 };
                let d = if u2 < VS2 && (rhs - u2) / (2.0 * (ACCEL_LOW + BRAKE)) <= d1 {
                    (rhs - u2) / (2.0 * (ACCEL_LOW + BRAKE))
                } else {
                    // Band 2: v² = c + 2 A2 d.
                    let c = if u2 < VS2 { VS2 - 2.0 * ACCEL_HIGH * d1 } else { u2 };
                    (rhs - c) / (2.0 * (ACCEL_HIGH + BRAKE))
                };
                xa + d.clamp(0.0, len)
            };
            t = accel(out, t, xa, sx, u2);
            t = brake(out, t, sx, xb, w2);
        }
    }
    t
}

/// Speed limit at offset `s` from breakpoints `[s, v]`.
pub fn limit_at(limits: &[[f64; 2]], s: f64) -> f64 {
    limits[limits.partition_point(|l| l[0] <= s).saturating_sub(1)][1]
}

/// Elementary intervals over `[a, b]`: the limits, lowered inside slow zones `(z0, z1, v)`.
fn intervals(limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    let mut cuts = vec![a, b];
    for l in limits {
        if l[0] > a && l[0] < b {
            cuts.push(l[0]);
        }
    }
    for z in zones {
        for c in [z.0, z.1] {
            if c > a && c < b {
                cuts.push(c);
            }
        }
    }
    cuts.sort_by(|x, y| x.total_cmp(y));
    cuts.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
    let mut out: Vec<(f64, f64, f64)> = Vec::with_capacity(cuts.len());
    for w in cuts.windows(2) {
        let m = 0.5 * (w[0] + w[1]);
        let mut v = limit_at(limits, m);
        for z in zones {
            if m > z.0 && m < z.1 {
                v = v.min(z.2);
            }
        }
        match out.last_mut() {
            Some(l) if l.2 == v => l.1 = w[1],
            _ => out.push((w[0], w[1], v)),
        }
    }
    out
}

/// One hop from a stop at `a` to a stop at `b`, with slow zones and intermediate standstills
/// `(s, stand seconds)` (sorted). Returns the arrival time.
fn hop(limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)], mids: &[(f64, f64)], mut t: f64, out: &mut Vec<Phase>) -> f64 {
    let mut from = a;
    for &(s, stand) in mids {
        t = segment(&intervals(limits, from, s, zones), t, out);
        if stand > 0.0 {
            push(out, t, s, 0.0, 0.0);
            t += stand;
        }
        from = s;
    }
    segment(&intervals(limits, from, b, zones), t, out)
}

fn hop_time(limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)], mids: &[(f64, f64)], scratch: &mut Vec<Phase>) -> f64 {
    scratch.clear();
    hop(limits, a, b, zones, mids, 0.0, scratch)
}

/// The trip's profile. `limits` are speed-limit breakpoints `[s, v]` from 0 (already widened by
/// half the train length); `stops` are the stop offsets in order; `dwell[i]` is the standing
/// time at stop i (extra at the first stop delays departure; the last is ignored); `holds` are
/// capacity delays at track points, sorted by offset.
pub fn run(limits: &[[f64; 2]], stops: &[f64], dwell: &[f64], holds: &[Hold]) -> Profile {
    let n = stops.len();
    let mut p = Profile::default();
    if n < 2 {
        return p;
    }
    let mut dw = dwell.to_vec();
    dw.resize(n, 0.0);
    p.dwell = dw[1..n - 1].iter().sum();
    let mut per_hop: Vec<Vec<Hold>> = vec![vec![]; n - 1];
    for h in holds {
        if h.delay < HOLD_MIN_S {
            continue;
        }
        let i = stops.partition_point(|&s| s <= h.s).saturating_sub(1).min(n - 2);
        if h.s - stops[i] < HOLD_NEAR_STOP_M || h.s >= stops[i + 1] - 1.0 {
            // At or by a stop: wait there (at the stop it is leaving).
            dw[i] += h.delay;
            p.hold += h.delay;
        } else {
            per_hop[i].push(*h);
        }
    }
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    let mut t = 0.0;
    p.arr.push(0.0);
    if dw[0] > 0.0 {
        push(&mut out, 0.0, stops[0], 0.0, 0.0);
        t = dw[0];
    }
    p.dep.push(t);
    for i in 0..n - 1 {
        let (a, b) = (stops[i], stops[i + 1]);
        let mut zones: Vec<(f64, f64, f64)> = vec![];
        let mut mids: Vec<(f64, f64)> = vec![];
        for h in &per_hop[i] {
            let base = hop_time(limits, a, b, &zones, &mids, &mut scratch);
            let mut with_stop = mids.clone();
            let k = with_stop.partition_point(|m| m.0 < h.s);
            with_stop.insert(k, (h.s, 0.0));
            let loss = hop_time(limits, a, b, &zones, &with_stop, &mut scratch) - base;
            let z0 = (h.s - CRAWL_ZONE_M).max(mids.iter().map(|m| m.0).filter(|&s| s < h.s).fold(a, f64::max) + 1.0);
            let crawl = |vc: f64, scratch: &mut Vec<Phase>| {
                let mut z = zones.clone();
                z.push((z0, h.s, vc));
                hop_time(limits, a, b, &z, &mids, scratch) - base
            };
            if h.delay >= loss || z0 >= h.s - 1.0 || crawl(0.5, &mut scratch) < h.delay {
                with_stop[k].1 = (h.delay - loss).max(0.0);
                mids = with_stop;
            } else {
                // Slow approach: bisect the zone speed for the delay (extra time falls with speed).
                let (mut lo, mut hi) = (0.5, V_TOP);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let extra = crawl(mid, &mut scratch);
                    if (extra - h.delay).abs() < 0.1 {
                        (lo, hi) = (mid, mid);
                        break;
                    }
                    if extra > h.delay {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                zones.push((z0, h.s, 0.5 * (lo + hi)));
            }
            p.hold += h.delay;
        }
        t = hop(limits, a, b, &zones, &mids, t, &mut out);
        p.arr.push(t);
        if i + 1 < n - 1 {
            if dw[i + 1] > 0.0 {
                push(&mut out, t, b, 0.0, 0.0);
            }
            t += dw[i + 1];
        }
        p.dep.push(t);
    }
    p.duration = t;
    p.phases = out;
    p
}

/// Index of the phase in force at offset `s` (the last one starting at or before it).
fn phase_at_s(ph: &[Phase], s: f64) -> Option<&Phase> {
    let i = ph.partition_point(|p| p.s <= s);
    (i > 0).then(|| &ph[i - 1])
}

/// Time the train is at offset `s` (leaving it, at a stand).
pub fn time_at(ph: &[Phase], s: f64) -> f64 {
    let Some(p) = phase_at_s(ph, s) else { return 0.0 };
    let ds = s - p.s;
    if ds <= 0.0 {
        return p.t;
    }
    if p.a == 0.0 {
        if p.v > 0.0 {
            p.t + ds / p.v
        } else {
            p.t
        }
    } else {
        p.t + (-p.v + (p.v * p.v + 2.0 * p.a * ds).max(0.0).sqrt()) / p.a
    }
}

/// Speed at offset `s`.
pub fn speed_at(ph: &[Phase], s: f64) -> f64 {
    match phase_at_s(ph, s) {
        Some(p) => (p.v * p.v + 2.0 * p.a * (s - p.s)).max(0.0).sqrt(),
        None => 0.0,
    }
}

/// Highest speed reached over `[s0, s1]`.
pub fn v_max(ph: &[Phase], s0: f64, s1: f64) -> f64 {
    let mut v = speed_at(ph, s0).max(speed_at(ph, s1));
    let a = ph.partition_point(|p| p.s <= s0);
    let b = ph.partition_point(|p| p.s < s1);
    for p in &ph[a..b.max(a)] {
        v = v.max(p.v);
    }
    v
}

/// Offset and speed at time `t` (clamped to the trip).
pub fn state_at(ph: &[Phase], t: f64) -> (f64, f64) {
    let i = ph.partition_point(|p| p.t <= t);
    if i == 0 {
        return ph.first().map_or((0.0, 0.0), |p| (p.s, p.v));
    }
    let p = &ph[i - 1];
    let mut tau = t - p.t;
    if p.a < 0.0 {
        tau = tau.min(p.v / -p.a);
    }
    (p.s + p.v * tau + 0.5 * p.a * tau * tau, p.v + p.a * tau)
}

/// Widen every restriction by `half` on each side (the whole train must be under a limit while
/// any of it is on the restricted piece; positions are of the train's middle). `raw` are
/// breakpoints `[s, v]` from 0 to `len`.
pub fn widen(raw: &[[f64; 2]], len: f64, half: f64) -> Vec<[f64; 2]> {
    if half <= 0.0 || raw.is_empty() {
        return raw.to_vec();
    }
    let end = |k: usize| if k + 1 < raw.len() { raw[k + 1][0] } else { len };
    let mut cuts = vec![0.0, len];
    for r in raw {
        for c in [r[0] - half, r[0] + half] {
            if c > 0.0 && c < len {
                cuts.push(c);
            }
        }
    }
    cuts.sort_by(|a, b| a.total_cmp(b));
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(cuts.len());
    let mut lo = 0;
    for w in cuts.windows(2) {
        let m = 0.5 * (w[0] + w[1]);
        while lo + 1 < raw.len() && end(lo) <= m - half {
            lo += 1;
        }
        let mut v = f64::INFINITY;
        let mut k = lo;
        while k < raw.len() && raw[k][0] < m + half {
            v = v.min(raw[k][1]);
            k += 1;
        }
        if out.last().map_or(true, |l| l[1] != v) {
            out.push([w[0], v]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top() -> Vec<[f64; 2]> {
        vec![[0.0, V_TOP]]
    }

    #[test]
    fn one_km_straight_matches_hand_calculation() {
        let p = run(&top(), &[0.0, 1000.0], &[0.0, 0.0], &[]);
        // By hand: 1.0 m/s² to 60 km/h, then 0.5 m/s², then braking at 0.9 m/s² to a stand.
        let vs: f64 = 60.0 / 3.6;
        let d1 = vs * vs / 2.0;
        // d1 + (v² - vs²)/(2 x 0.5) + v²/(2 x 0.9) = 1000
        let v2 = (1000.0 - d1 + vs * vs) / (1.0 + 1.0 / 1.8);
        let v = v2.sqrt();
        let want = vs / 1.0 + (v - vs) / 0.5 + v / 0.9;
        assert!((p.duration - want).abs() < 1e-9, "{} vs {}", p.duration, want);
        assert!((p.duration - 67.5).abs() < 0.1);
        assert_eq!(p.phases.len(), 3);
        let (s, v_end) = state_at(&p.phases, p.duration);
        assert!((s - 1000.0).abs() < 1e-6 && v_end.abs() < 1e-6);
        // Reaches top speed on a long run.
        let p = run(&top(), &[0.0, 10000.0], &[0.0, 0.0], &[]);
        assert!((v_max(&p.phases, 0.0, 10000.0) - V_TOP).abs() < 1e-9);
    }

    #[test]
    fn respects_limits_and_widening() {
        let raw = vec![[0.0, V_TOP], [1000.0, 10.0], [1300.0, V_TOP]];
        let lim = widen(&raw, 3000.0, 50.0);
        assert_eq!(lim, vec![[0.0, V_TOP], [950.0, 10.0], [1350.0, V_TOP]]);
        let p = run(&lim, &[0.0, 3000.0], &[0.0, 0.0], &[]);
        for j in 0..3000 {
            let s = j as f64;
            assert!(speed_at(&p.phases, s) <= limit_at(&lim, s) + 1e-6, "at {s}");
        }
        assert!((speed_at(&p.phases, 1100.0) - 10.0).abs() < 1e-9);
        // Monotonic time, continuous position.
        let mut last = -1.0;
        for j in 0..=300 {
            let t = time_at(&p.phases, j as f64 * 10.0);
            assert!(t >= last);
            last = t;
        }
    }

    #[test]
    fn dwell_and_holds_add_their_time() {
        let stops = [0.0, 2000.0, 5000.0];
        let base = run(&top(), &stops, &[0.0, 30.0, 0.0], &[]);
        assert!((base.dep[1] - base.arr[1] - 30.0).abs() < 1e-9);
        for delay in [5.0, 20.0, 45.0, 120.0] {
            let p = run(&top(), &stops, &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay }]);
            assert!((p.duration - base.duration - delay).abs() < 0.5, "delay {delay}: {} vs {}", p.duration, base.duration);
            assert!((p.arr[1] - base.arr[1]).abs() < 1e-9);
        }
        // A long hold is a stop in front of the point; a short one is not.
        let p = run(&top(), &stops, &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay: 120.0 }]);
        assert!(p.phases.iter().any(|ph| ph.v == 0.0 && (ph.s - 3500.0).abs() < 1e-9));
        let p = run(&top(), &stops, &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay: 5.0 }]);
        assert!(!p.phases.iter().any(|ph| ph.v == 0.0 && (ph.s - 3500.0).abs() < 1e-9));
        // A hold right after a stop becomes dwell there.
        let p = run(&top(), &stops, &[0.0, 30.0, 0.0], &[Hold { s: 2010.0, delay: 40.0 }]);
        assert!((p.dep[1] - p.arr[1] - 70.0).abs() < 1e-9);
    }
}
