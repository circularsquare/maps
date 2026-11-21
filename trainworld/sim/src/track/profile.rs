//! Run-time profiles (SPEC 2, 6.2, 6.3): the fastest motion along a line's path under speed
//! limits, stopping at every stop, with capacity delays baked in as holds.
//!
//! The result is a list of constant-acceleration phases `(t, s, v, a)`: from time `t` the train is
//! at path offset `s` with speed `v` and acceleration `a` until the next phase starts. Those are
//! the keyframes (notes/T-040.md): position at any time is closed form.
//!
//! Method: per stretch between two standstills, the minimum-time profile is the pointwise minimum
//! of the limit, the forward acceleration envelope and the backward braking envelope. In v² against
//! distance both envelopes are piecewise straight lines (one slope per traction band), so every
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

/// A capacity delay to absorb at path offset `s`, or with `span` over `[s, s + span]` as slower
/// running (T-2018: a following loss, so the train never stands between stops for it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub s: f64,
    pub delay: f64,
    pub span: f64,
}

/// Traction bands: (speed where the band ends, acceleration). The last band is open; only
/// high-speed trains go past 200 km/h, so ordinary trains never reach it (T-817).
const BANDS: [(f64, f64); 3] = [(ACCEL_SWITCH_V, ACCEL_LOW), (ACCEL_TOP_V, ACCEL_HIGH), (f64::INFINITY, ACCEL_TOP)];

/// The band a train at speed² u2 accelerates in, and the speed² where that band ends.
fn band(u2: f64) -> (usize, f64) {
    let k = BANDS.iter().position(|b| u2 < b.0 * b.0).unwrap_or(BANDS.len() - 1);
    (k, BANDS[k].0 * BANDS[k].0)
}

/// Distance to accelerate from u² to v².
fn acc_dist(u2: f64, v2: f64) -> f64 {
    if v2 <= u2 {
        return 0.0;
    }
    let (mut k, mut end) = band(u2);
    let mut from = u2;
    let mut d = 0.0;
    loop {
        if v2 <= end {
            return d + (v2 - from) / (2.0 * BANDS[k].1);
        }
        d += (end - from) / (2.0 * BANDS[k].1);
        from = end;
        k += 1;
        end = BANDS[k].0 * BANDS[k].0;
    }
}

/// v² after accelerating `d` metres from u².
fn acc_v2(u2: f64, d: f64) -> f64 {
    let (mut k, mut end) = band(u2);
    let (mut from, mut at) = (u2, 0.0);
    loop {
        let dk = (end - from) / (2.0 * BANDS[k].1);
        if d - at <= dk {
            return from + 2.0 * BANDS[k].1 * (d - at);
        }
        at += dk;
        from = end;
        k += 1;
        end = BANDS[k].0 * BANDS[k].0;
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

struct Sink<'a>(Option<&'a mut Vec<Phase>>);

impl Sink<'_> {
    fn push(&mut self, t: f64, s: f64, v: f64, a: f64) {
        if let Some(out) = self.0.as_deref_mut() {
            push(out, t, s, v, a);
        }
    }
}

#[derive(Default)]
struct Scratch {
    cuts: Vec<f64>,
    iv: Vec<(f64, f64, f64)>,
    fw: Vec<f64>,
    bend: Vec<f64>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = Default::default();
}

fn with_scratch<T>(f: impl FnOnce(&mut Scratch) -> T) -> T {
    let mut s = SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let out = f(&mut s);
    SCRATCH.with(|c| *c.borrow_mut() = s);
    out
}

/// Accelerate from offset x0 (speed² u2) to x1.
fn accel(out: &mut Sink, mut t: f64, x0: f64, x1: f64, u2: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let (mut k, mut end) = band(u2);
    let (mut v0, mut s) = (u2.max(0.0).sqrt(), x0);
    // each band's end as a distance from x0, as acc_v2 measures it, so speeds are taken from u2
    let (mut from, mut reach) = (u2, 0.0);
    loop {
        let a = BANDS[k].1;
        reach += (end - from) / (2.0 * a);
        let e = x1.min(x0 + reach);
        let v1 = acc_v2(u2, e - x0).sqrt();
        out.push(t, s, v0, a);
        t += (v1 - v0) / a;
        if x1 <= x0 + reach {
            return t;
        }
        s = x0 + reach;
        v0 = BANDS[k].0;
        from = end;
        k += 1;
        end = BANDS[k].0 * BANDS[k].0;
    }
}

/// Brake at `br` m/s² from x0 to x1, ending at speed² w2.
fn brake(out: &mut Sink, t: f64, x0: f64, x1: f64, w2: f64, br: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let v0 = (w2 + 2.0 * br * (x1 - x0)).sqrt();
    let v1 = w2.max(0.0).sqrt();
    out.push(t, x0, v0, -br);
    t + (v0 - v1) / br
}

/// From standstill at the first interval's start to standstill at the last one's end, through
/// intervals `(x0, x1, v_limit)`, braking at `br`. Returns the end time.
fn segment(iv: &[(f64, f64, f64)], fw: &mut Vec<f64>, bend: &mut Vec<f64>, mut t: f64, out: &mut Sink, br: f64) -> f64 {
    let m = iv.len();
    if m == 0 {
        return t;
    }
    fw.clear();
    fw.resize(m, 0.0);
    bend.clear();
    bend.resize(m, 0.0);
    let mut f = 0.0f64;
    for k in 0..m {
        let v2 = iv[k].2 * iv[k].2;
        let u = f.min(v2);
        fw[k] = u;
        f = acc_v2(u, iv[k].1 - iv[k].0).min(v2);
    }
    let mut b = 0.0f64;
    for k in (0..m).rev() {
        let v2 = iv[k].2 * iv[k].2;
        let w = b.min(v2);
        bend[k] = w;
        b = (w + 2.0 * br * (iv[k].1 - iv[k].0)).min(v2);
    }
    for k in 0..m {
        t = cross(out, t, iv[k], fw[k], bend[k], br);
    }
    t
}

/// Across one interval `(xa, xb, v_limit)`, entering at speed² u2 and leaving at w2 (the forward
/// and braking envelopes there), braking at `br`. Returns the time at its end.
fn cross(out: &mut Sink, mut t: f64, (xa, xb, vl): (f64, f64, f64), u2: f64, w2: f64, br: f64) -> f64 {
    let (v2, len) = (vl * vl, xb - xa);
    let s_fv = xa + acc_dist(u2, v2);
    let s_bv = xb - (v2 - w2) / (2.0 * br);
    if s_fv <= s_bv {
        t = accel(out, t, xa, s_fv, u2);
        if s_bv > s_fv {
            out.push(t, s_fv, vl, 0.0);
            t += (s_bv - s_fv) / vl;
        }
        brake(out, t, s_bv, xb, w2, br)
    } else {
        let b_at_xa = w2 + 2.0 * br * len;
        let sx = if u2 >= b_at_xa {
            xa
        } else if acc_v2(u2, len) <= w2 {
            xb
        } else {
            // acc_v2(u2, d) = w2 + 2B(len - d), solved per traction band: v² = c + 2 a d there.
            let rhs = w2 + 2.0 * br * len;
            let (mut k, mut end) = band(u2);
            let (mut from, mut at) = (u2, 0.0);
            let d = loop {
                let a = BANDS[k].1;
                let dk = (end - from) / (2.0 * a);
                let c = from - 2.0 * a * at;
                let d = (rhs - c) / (2.0 * (a + br));
                if d <= at + dk || k + 1 == BANDS.len() {
                    break d;
                }
                at += dk;
                from = end;
                k += 1;
                end = BANDS[k].0 * BANDS[k].0;
            };
            xa + d.clamp(0.0, len)
        };
        t = accel(out, t, xa, sx, u2);
        brake(out, t, sx, xb, w2, br)
    }
}

/// Speed limit at offset `s` from breakpoints `[s, v]`.
pub fn limit_at(limits: &[[f64; 2]], s: f64) -> f64 {
    limits[limits.partition_point(|l| l[0] <= s).saturating_sub(1)][1]
}

/// Elementary intervals over `[a, b]`: the limits, lowered inside slow zones `(z0, z1, v)`.
fn intervals(sc: &mut Scratch, limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)]) {
    let cuts = &mut sc.cuts;
    cuts.clear();
    cuts.push(a);
    cuts.push(b);
    // the breakpoints strictly inside (a, b); a line has a few hundred, a hop a few
    let i0 = limits.partition_point(|l| l[0] <= a);
    let i1 = limits.partition_point(|l| l[0] < b).max(i0);
    cuts.extend(limits[i0..i1].iter().map(|l| l[0]));
    for z in zones {
        for c in [z.0, z.1] {
            if c > a && c < b {
                cuts.push(c);
            }
        }
    }
    cuts.sort_by(|x, y| x.total_cmp(y));
    cuts.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
    let out = &mut sc.iv;
    out.clear();
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
}

/// One hop from a stop at `a` to a stop at `b`, with slow zones and intermediate standstills
/// `(s, stand seconds)` (sorted), braking at `br`. Returns the arrival time.
fn hop(sc: &mut Scratch, limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)], mids: &[(f64, f64)], mut t: f64, out: &mut Sink, br: f64) -> f64 {
    let mut from = a;
    for &(s, stand) in mids {
        t = {
            intervals(sc, limits, from, s, zones);
            segment(&sc.iv, &mut sc.fw, &mut sc.bend, t, out, br)
        };
        if stand > 0.0 {
            out.push(t, s, 0.0, 0.0);
            t += stand;
        }
        from = s;
    }
    intervals(sc, limits, from, b, zones);
    segment(&sc.iv, &mut sc.fw, &mut sc.bend, t, out, br)
}

/// How near a crawl speed's added time comes to its hold's delay, seconds.
const CRAWL_TOL_S: f64 = 0.01;

/// The last stretch of a hop (from its last standstill to its end) as `segment` works it out:
/// each interval, the speed² entering and leaving it and its time. Kept so that a crawl zone or
/// a stand in it can be timed by working out only the intervals they change (T-3550).
#[derive(Default)]
struct Stretch {
    iv: Vec<(f64, f64, f64)>,
    fw: Vec<f64>,
    bend: Vec<f64>,
    dt: Vec<f64>,
    /// The hop's time before the stretch (earlier stretches and stands), and the stretch's own.
    pre: f64,
    total: f64,
    /// Braking rate, m/s².
    br: f64,
}

/// A piece of track standing in for some of a stretch's intervals in `retime`: `(xa, xb, v_limit,
/// stands at xa)`.
type Piece = (f64, f64, f64, bool);

#[derive(Default)]
struct Retime {
    pieces: Vec<Piece>,
    pfw: Vec<f64>,
    pbend: Vec<f64>,
    nfw: Vec<f64>,
    nbend: Vec<f64>,
}

impl Stretch {
    /// The hop from `a` to `b` with these zones and standstills between: the stretches before the
    /// last one timed, the last one kept; braking at `br`.
    fn of_hop(&mut self, sc: &mut Scratch, limits: &[[f64; 2]], a: f64, b: f64, zones: &[(f64, f64, f64)], mids: &[(f64, f64)], br: f64) {
        let (mut from, mut t) = (a, 0.0);
        for &(s, stand) in mids {
            intervals(sc, limits, from, s, zones);
            t = segment(&sc.iv, &mut sc.fw, &mut sc.bend, t, &mut Sink(None), br) + stand;
            from = s;
        }
        intervals(sc, limits, from, b, zones);
        self.br = br;
        self.pre = t;
        self.iv.clone_from(&sc.iv);
        let m = self.iv.len();
        self.fw.clear();
        self.bend.clear();
        self.dt.clear();
        let mut f = 0.0f64;
        for &(xa, xb, v) in &self.iv {
            let u = f.min(v * v);
            self.fw.push(u);
            f = acc_v2(u, xb - xa).min(v * v);
        }
        self.bend.resize(m, 0.0);
        let mut w_next = 0.0f64;
        for k in (0..m).rev() {
            let (xa, xb, v) = self.iv[k];
            let w = w_next.min(v * v);
            self.bend[k] = w;
            w_next = (w + 2.0 * br * (xb - xa)).min(v * v);
        }
        self.total = 0.0;
        for k in 0..m {
            let d = cross(&mut Sink(None), 0.0, self.iv[k], self.fw[k], self.bend[k], br);
            self.dt.push(d);
            self.total += d;
        }
    }

    fn time(&self) -> f64 {
        self.pre + self.total
    }

    /// The intervals overlapping `(x0, x1)`.
    fn over(&self, x0: f64, x1: f64) -> (usize, usize) {
        let k0 = self.iv.partition_point(|iv| iv.1 <= x0).min(self.iv.len() - 1);
        let k1 = self.iv.partition_point(|iv| iv.0 < x1).saturating_sub(1).max(k0);
        (k0, k1)
    }

    /// The time the train takes over `[z0, z1]` and its top speed there.
    fn through(&self, z0: f64, z1: f64, buf: &mut Vec<Phase>) -> (f64, f64) {
        buf.clear();
        if self.iv.is_empty() {
            return (0.0, 0.0);
        }
        let (k0, k1) = self.over(z0, z1);
        let mut sink = Sink(Some(buf));
        let mut t = 0.0;
        for k in k0..=k1 {
            t = cross(&mut sink, t, self.iv[k], self.fw[k], self.bend[k], self.br);
        }
        (time_at(buf, z1) - time_at(buf, z0), v_max(buf, z0, z1))
    }

    /// The hop's time with the stretch's speed limit lowered to `vc` over `[z0, z1]`.
    fn with_zone(&self, rt: &mut Retime, z0: f64, z1: f64, vc: f64) -> f64 {
        if self.iv.is_empty() {
            return self.time();
        }
        let (k0, k1) = self.over(z0, z1);
        rt.pieces.clear();
        for &(xa, xb, v) in &self.iv[k0..=k1] {
            let mut x = xa;
            for c in [z0, z1] {
                if c > x + 1e-9 && c < xb - 1e-9 {
                    rt.pieces.push((x, c, v, false));
                    x = c;
                }
            }
            rt.pieces.push((x, xb, v, false));
        }
        for p in rt.pieces.iter_mut() {
            let m = 0.5 * (p.0 + p.1);
            if m > z0 && m < z1 {
                p.2 = p.2.min(vc);
            }
        }
        self.pre + self.retime(rt, k0, k1)
    }

    /// The hop's time with a standstill (of no time) at `x`.
    fn with_stand(&self, rt: &mut Retime, x: f64) -> f64 {
        if self.iv.is_empty() {
            return self.time();
        }
        let (k0, _) = self.over(x, x);
        let (xa, xb, v) = self.iv[k0];
        rt.pieces.clear();
        if x > xa + 1e-9 {
            rt.pieces.push((xa, x, v, false));
        }
        rt.pieces.push((x.max(xa), xb, v, true));
        self.pre + self.retime(rt, k0, k0)
    }

    /// The stretch's time with intervals k0..=k1 replaced by `rt.pieces` (the same ground). Only
    /// the pieces and the intervals whose envelopes they move are worked out again: the forward
    /// envelope after the pieces until it meets the old one (where both reach a speed limit), the
    /// braking one before them likewise.
    fn retime(&self, rt: &mut Retime, k0: usize, k1: usize) -> f64 {
        let (m, br) = (self.iv.len(), self.br);
        let (pfw, pbend, nfw, nbend) = (&mut rt.pfw, &mut rt.pbend, &mut rt.nfw, &mut rt.nbend);
        pfw.clear();
        nfw.clear();
        nbend.clear();
        let mut f = match k0.checked_sub(1) {
            Some(k) => {
                let (xa, xb, v) = self.iv[k];
                acc_v2(self.fw[k], xb - xa).min(v * v)
            }
            None => 0.0,
        };
        for &(xa, xb, v, stand) in &rt.pieces {
            if stand {
                f = 0.0;
            }
            let u = f.min(v * v);
            pfw.push(u);
            f = acc_v2(u, xb - xa).min(v * v);
        }
        let mut kf = k1 + 1;
        while kf < m {
            let (xa, xb, v) = self.iv[kf];
            let u = f.min(v * v);
            if u == self.fw[kf] {
                break;
            }
            nfw.push(u);
            f = acc_v2(u, xb - xa).min(v * v);
            kf += 1;
        }
        let mut b = match self.iv.get(k1 + 1) {
            Some(&(xa, xb, v)) => (self.bend[k1 + 1] + 2.0 * br * (xb - xa)).min(v * v),
            None => 0.0,
        };
        pbend.resize(rt.pieces.len(), 0.0);
        for (j, &(xa, xb, v, stand)) in rt.pieces.iter().enumerate().rev() {
            let w = b.min(v * v);
            pbend[j] = w;
            b = if stand { 0.0 } else { (w + 2.0 * br * (xb - xa)).min(v * v) };
        }
        let mut kb = k0;
        while kb > 0 {
            let (xa, xb, v) = self.iv[kb - 1];
            let w = b.min(v * v);
            if w == self.bend[kb - 1] {
                break;
            }
            nbend.push(w);
            b = (w + 2.0 * br * (xb - xa)).min(v * v);
            kb -= 1;
        }
        let none = &mut Sink(None);
        let mut t = self.total - self.dt[kb..kf].iter().sum::<f64>();
        for k in kb..k0 {
            t += cross(none, 0.0, self.iv[k], self.fw[k], nbend[k0 - 1 - k], br);
        }
        for (j, &(xa, xb, v, _)) in rt.pieces.iter().enumerate() {
            t += cross(none, 0.0, (xa, xb, v), pfw[j], pbend[j], br);
        }
        for k in k1 + 1..kf {
            t += cross(none, 0.0, self.iv[k], nfw[k - k1 - 1], self.bend[k], br);
        }
        t
    }
}

/// The speed for a crawl zone that adds `delay` to a run taking `base` (T-3550): `time(v)` is the
/// run's time with the zone at v, falling as v rises, met to within `CRAWL_TOL_S`. Before the zone
/// the train takes `t_in` over its `len` metres, at most `v_in`: at `v_in` nothing changes, and at
/// `len / (t_in + delay)` the zone alone adds the delay, so the speed lies between. The time
/// across the zone is close to straight in 1/v, so a secant in 1/v from there takes 2-4 tries,
/// kept inside the bracket and halving it when it stalls. `slow` is the time at `v_slow`, the
/// lowest speed, when known.
fn crawl_speed(mut time: impl FnMut(f64) -> f64, base: f64, delay: f64, len: f64, t_in: f64, v_in: f64, v_slow: f64, slow: Option<f64>) -> f64 {
    let want = base + delay;
    // g(u) = time(1/u) - want rises with u = 1/v: below zero at lo, at or above it at hi
    let (mut lo, mut g_lo, mut hi) = (1.0 / v_in.max(v_slow), -delay, 1.0 / v_slow);
    let mut g_hi = slow.map(|t| t - want);
    if !(hi > lo) {
        return v_slow;
    }
    let len = len.max(1.0);
    let mut u = (t_in + delay) / len;
    let mut prev: Option<(f64, f64)> = None;
    if !(u > lo && u < hi) {
        // the zone alone cannot take the delay above the lowest speed: start from there
        u = match g_hi {
            Some(g) => hi - g / len,
            None => 0.5 * (lo + hi),
        };
        prev = g_hi.map(|g| (hi, g));
        if !(u > lo && u < hi) {
            u = 0.5 * (lo + hi);
        }
    }
    let mut width = hi - lo;
    for k in 0..40 {
        let g = time(1.0 / u) - want;
        if g.abs() < CRAWL_TOL_S {
            return 1.0 / u;
        }
        if g > 0.0 {
            (hi, g_hi) = (u, Some(g));
        } else {
            (lo, g_lo) = (u, g);
        }
        let mut next = match prev {
            // At the first try the zone's own time is the delay, so g is what slowing to v and
            // back costs, about c (v_in - v)² (constant braking and acceleration). The speed
            // where len / v - t_in + c (v_in - v)² is the delay, by Newton's method (convex and
            // falling, so it creeps up from v without overshooting).
            None if g > 0.0 && 1.0 / u < v_in => {
                let c = g / (v_in - 1.0 / u).powi(2);
                let mut v = 1.0 / u;
                for _ in 0..6 {
                    let f = len / v - t_in + c * (v_in - v).powi(2) - delay;
                    v -= f / (-len / (v * v) - 2.0 * c * (v_in - v));
                }
                1.0 / v
            }
            None if g > 0.0 => u - g / len,
            Some((up, gp)) if g != gp => u - g * (u - up) / (g - gp),
            _ => f64::NAN,
        };
        prev = Some((u, g));
        if !(next > lo && next < hi) {
            next = match g_hi {
                Some(gh) => lo + (hi - lo) * -g_lo / (gh - g_lo),
                None => 0.5 * (lo + hi),
            };
        }
        if k % 3 == 2 {
            if hi - lo > 0.5 * width {
                next = 0.5 * (lo + hi);
            }
            width = hi - lo;
        }
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        u = next;
    }
    2.0 / (lo + hi)
}

/// Each hop's motion with no holds, standstill to standstill, timed from 0, and its duration:
/// `run_with` takes these for every hop no hold falls in (T-1283: a line's profiles are worked out
/// again whenever its holds move, mostly over hops the holds leave alone).
#[derive(Clone, Debug, Default)]
pub struct Hops(Vec<(Vec<Phase>, f64)>);

pub fn hops(limits: &[[f64; 2]], stops: &[f64], jump: &[f64], br: f64) -> Hops {
    let leave = |i: usize| stops[i] + jump.get(i).copied().unwrap_or(0.0);
    Hops((0..stops.len().saturating_sub(1)).map(|i| {
        let mut out = vec![];
        let d = {
            let mut sink = Sink(Some(&mut out));
            with_scratch(|sc| hop(sc, limits, leave(i), stops[i + 1], &[], &[], 0.0, &mut sink, br))
        };
        (out, d)
    }).collect())
}

/// A train's trip profile (braking at `BRAKE`). `limits` are speed-limit breakpoints `[s, v]` from 0 (already widened by
/// half the train length); `stops` are the stop offsets in order; `jump[i]` (empty for none)
/// moves stop i's departure on from its arrival (a reversal, T-797: the train is drawn jumping
/// once as it stops); `dwell[i]` is the standing time at stop i (extra at the first stop delays
/// departure; the last is ignored); `holds` are capacity delays at track points, sorted by offset.
pub fn run(limits: &[[f64; 2]], stops: &[f64], jump: &[f64], dwell: &[f64], holds: &[Hold]) -> Profile {
    run_with(limits, stops, jump, dwell, holds, &Hops::default(), BRAKE)
}

/// `run`, braking at `br` (T-3921: `TRAM_BRAKE` for a tram line), with `hops` from `hops` over the
/// same limits, stops, jumps and braking (or none).
pub fn run_with(limits: &[[f64; 2]], stops: &[f64], jump: &[f64], dwell: &[f64], holds: &[Hold], hops: &Hops, br: f64) -> Profile {
    let n = stops.len();
    let mut p = Profile::default();
    if n < 2 {
        return p;
    }
    let mut dw = dwell.to_vec();
    dw.resize(n, 0.0);
    let leave = |i: usize| stops[i] + jump.get(i).copied().unwrap_or(0.0);
    p.dwell = dw[1..n - 1].iter().sum();
    let mut per_hop: Vec<Vec<Hold>> = vec![vec![]; n - 1];
    let (mut rt, mut buf) = (Retime::default(), Vec::new());
    let mut st: Vec<Stretch> = vec![];
    // spread holds: one lowered speed over their span, the same in every hop they cover
    let mut hop_zones: Vec<Vec<(f64, f64, f64)>> = vec![vec![]; n - 1];
    for h in holds.iter().filter(|h| h.span > 0.0 && h.delay >= HOLD_MIN_S) {
        let (s0, s1) = (h.s, h.s + h.span);
        let parts: Vec<(usize, f64, f64)> = (0..n - 1)
            .filter_map(|i| {
                let (z0, z1) = (s0.max(leave(i)), s1.min(stops[i + 1]));
                (z1 > z0 + 1.0).then_some((i, z0, z1))
            })
            .collect();
        let Some(&(i_first, ..)) = parts.first() else {
            // no running in it: wait at the stop before
            let i = stops.partition_point(|&s| s <= h.s).saturating_sub(1).min(n - 2);
            dw[i] += h.delay;
            p.hold += h.delay;
            continue;
        };
        if st.len() < parts.len() {
            st.resize_with(parts.len(), Stretch::default);
        }
        let (mut base, mut t_in, mut v_in, mut len) = (0.0, 0.0, 0.0f64, 0.0);
        for (j, &(i, z0, z1)) in parts.iter().enumerate() {
            with_scratch(|sc| st[j].of_hop(sc, limits, leave(i), stops[i + 1], &hop_zones[i], &[], br));
            let (ti, vi) = st[j].through(z0, z1, &mut buf);
            base += st[j].time();
            t_in += ti;
            v_in = v_in.max(vi);
            len += z1 - z0;
        }
        let mut total = |vc: f64| parts.iter().enumerate().map(|(j, &(_, z0, z1))| st[j].with_zone(&mut rt, z0, z1, vc)).sum::<f64>();
        // at the lowest speed the span takes at least len / v - t_in more: only near that is it run
        let slow = (len / FOLLOW_CRAWL_MIN_V - t_in < h.delay + 1.0).then(|| total(FOLLOW_CRAWL_MIN_V));
        let vc = match slow.map(|t| t - base) {
            Some(most) if most <= h.delay => {
                // too much to run off: the rest waits at the stop before the span
                dw[i_first] += h.delay - most;
                FOLLOW_CRAWL_MIN_V
            }
            _ => {
                crawl_speed(total, base, h.delay, len, t_in, v_in, FOLLOW_CRAWL_MIN_V, slow)
            }
        };
        for &(i, z0, z1) in &parts {
            hop_zones[i].push((z0, z1, vc));
        }
        p.hold += h.delay;
    }
    for h in holds {
        if h.delay < HOLD_MIN_S || h.span > 0.0 {
            continue;
        }
        let i = stops.partition_point(|&s| s <= h.s).saturating_sub(1).min(n - 2);
        if h.s - leave(i) < HOLD_NEAR_STOP_M || h.s >= stops[i + 1] - 1.0 {
            // At or by a stop: wait there (at the stop it is leaving).
            dw[i] += h.delay;
            p.hold += h.delay;
        } else {
            per_hop[i].push(*h);
        }
    }
    if st.is_empty() && per_hop.iter().any(|h| !h.is_empty()) {
        st.push(Stretch::default());
    }
    let mut out = Vec::new();
    let mut t = 0.0;
    p.arr.push(0.0);
    if dw[0] > 0.0 {
        push(&mut out, 0.0, stops[0], 0.0, 0.0);
        t = dw[0];
    }
    p.dep.push(t);
    for i in 0..n - 1 {
        let (a, b) = (leave(i), stops[i + 1]);
        let mut zones: Vec<(f64, f64, f64)> = std::mem::take(&mut hop_zones[i]);
        let mut mids: Vec<(f64, f64)> = vec![];
        for h in &per_hop[i] {
            // every point hold lies after the stands so far (holds are sorted), so in the last stretch
            let sr = &mut st[0];
            with_scratch(|sc| sr.of_hop(sc, limits, a, b, &zones, &mids, br));
            let base = sr.time();
            let loss = sr.with_stand(&mut rt, h.s) - base;
            let k = mids.partition_point(|m| m.0 < h.s);
            let z0 = (h.s - CRAWL_ZONE_M).max(mids.iter().map(|m| m.0).filter(|&s| s < h.s).fold(a, f64::max) + 1.0);
            let mut stand = h.delay >= loss || z0 >= h.s - 1.0;
            let (mut t_in, mut v_in, mut slow) = (0.0, 0.0, None);
            if !stand {
                (t_in, v_in) = sr.through(z0, h.s, &mut buf);
                // at 0.5 m/s the zone takes at least (h.s - z0) / 0.5 - t_in more: only near that is it run
                if (h.s - z0) / 0.5 - t_in < h.delay + 1.0 {
                    let ts = sr.with_zone(&mut rt, z0, h.s, 0.5);
                    stand = ts - base < h.delay;
                    slow = Some(ts);
                }
            }
            if stand {
                mids.insert(k, (h.s, (h.delay - loss).max(0.0)));
            } else {
                // Slow approach: the zone speed for the delay (extra time falls with speed).
                let vc = crawl_speed(|v| sr.with_zone(&mut rt, z0, h.s, v), base, h.delay, h.s - z0, t_in, v_in, 0.5, slow);
                zones.push((z0, h.s, vc));
            }
            p.hold += h.delay;
        }
        match hops.0.get(i).filter(|_| per_hop[i].is_empty() && zones.is_empty() && hops.0.len() == n - 1) {
            Some((ph, d)) => {
                for q in ph {
                    push(&mut out, t + q.t, q.s, q.v, q.a);
                }
                t += d;
            }
            None => {
                let mut sink = Sink(Some(&mut out));
                t = with_scratch(|sc| hop(sc, limits, a, b, &zones, &mids, t, &mut sink, br));
            }
        }
        p.arr.push(t);
        if i + 1 < n - 1 {
            let at = leave(i + 1);
            if dw[i + 1] > 0.0 || at > b {
                push(&mut out, t, at, 0.0, 0.0);
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
        let p = run(&top(), &[0.0, 1000.0], &[], &[0.0, 0.0], &[]);
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
        let p = run(&top(), &[0.0, 10000.0], &[], &[0.0, 0.0], &[]);
        assert!((v_max(&p.phases, 0.0, 10000.0) - V_TOP).abs() < 1e-9);
    }

    #[test]
    fn respects_limits_and_widening() {
        let raw = vec![[0.0, V_TOP], [1000.0, 10.0], [1300.0, V_TOP]];
        let lim = widen(&raw, 3000.0, 50.0);
        assert_eq!(lim, vec![[0.0, V_TOP], [950.0, 10.0], [1350.0, V_TOP]]);
        let p = run(&lim, &[0.0, 3000.0], &[], &[0.0, 0.0], &[]);
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

    /// T-3550: a stretch's times with a zone or a stand, worked out from the intervals they
    /// change, are the hop's times worked out whole.
    #[test]
    fn stretch_retimes_match_whole_hops() {
        let raw = vec![[0.0, V_TOP], [800.0, 12.0], [1100.0, 30.0], [2600.0, 20.0], [3900.0, V_TOP]];
        let lim = widen(&raw, 6000.0, 60.0);
        let whole = |zones: &[(f64, f64, f64)], mids: &[(f64, f64)], br: f64| with_scratch(|sc| hop(sc, &lim, 0.0, 6000.0, zones, mids, 0.0, &mut Sink(None), br));
        let mut rt = Retime::default();
        // a train's braking and a tram's (T-3921)
        let cases = [BRAKE, TRAM_BRAKE].into_iter().flat_map(|br| [(br, vec![]), (br, vec![(1500.0, 20.0)])]);
        for (br, mids) in cases {
            for zones in [vec![], vec![(2000.0, 2300.0, 9.0)]] {
                let mut st = Stretch::default();
                with_scratch(|sc| st.of_hop(sc, &lim, 0.0, 6000.0, &zones, &mids, br));
                assert!((st.time() - whole(&zones, &mids, br)).abs() < 1e-9);
                for x in [1700.0, 2150.0, 3000.0, 3950.0, 5990.0] {
                    let mut m = mids.clone();
                    m.push((x, 0.0));
                    let (a, b) = (st.with_stand(&mut rt, x), whole(&zones, &m, br));
                    assert!((a - b).abs() < 1e-9, "stand at {x}, {mids:?} {zones:?}: {a} vs {b}");
                }
                for (z0, z1) in [(1600.0, 1900.0), (2100.0, 2400.0), (3700.0, 4000.0), (5700.0, 5999.0), (1501.0, 5000.0)] {
                    for vc in [0.5, 5.0, 11.0, 19.0, 25.0, 45.0] {
                        let mut z = zones.clone();
                        z.push((z0, z1, vc));
                        let (a, b) = (st.with_zone(&mut rt, z0, z1, vc), whole(&z, &mids, br));
                        assert!((a - b).abs() < 1e-9, "zone {z0}-{z1} at {vc}, {mids:?} {zones:?}: {a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn dwell_and_holds_add_their_time() {
        let stops = [0.0, 2000.0, 5000.0];
        let base = run(&top(), &stops, &[], &[0.0, 30.0, 0.0], &[]);
        assert!((base.dep[1] - base.arr[1] - 30.0).abs() < 1e-9);
        for delay in [5.0, 20.0, 45.0, 120.0] {
            let p = run(&top(), &stops, &[], &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay, span: 0.0 }]);
            assert!((p.duration - base.duration - delay).abs() < 0.5, "delay {delay}: {} vs {}", p.duration, base.duration);
            assert!((p.arr[1] - base.arr[1]).abs() < 1e-9);
        }
        // A long hold is a stop in front of the point; a short one is not.
        let p = run(&top(), &stops, &[], &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay: 120.0, span: 0.0 }]);
        assert!(p.phases.iter().any(|ph| ph.v == 0.0 && (ph.s - 3500.0).abs() < 1e-9));
        let p = run(&top(), &stops, &[], &[0.0, 30.0, 0.0], &[Hold { s: 3500.0, delay: 5.0, span: 0.0 }]);
        assert!(!p.phases.iter().any(|ph| ph.v == 0.0 && (ph.s - 3500.0).abs() < 1e-9));
        // T-3550: crawls (several point holds in one hop, spread holds over two hops) add their
        // delays to within the crawl speed's tolerance.
        let lim = widen(&[[0.0, V_TOP], [2600.0, 20.0], [3300.0, V_TOP]], 5000.0, 60.0);
        let none = run(&lim, &stops, &[], &[0.0, 30.0, 0.0], &[]);
        for hs in [
            vec![Hold { s: 2700.0, delay: 3.0, span: 0.0 }, Hold { s: 3000.0, delay: 8.0, span: 0.0 }, Hold { s: 4200.0, delay: 2.5, span: 0.0 }],
            vec![Hold { s: 500.0, delay: 12.0, span: 3000.0 }, Hold { s: 2500.0, delay: 4.0, span: 900.0 }],
            vec![Hold { s: 900.0, delay: 6.0, span: 400.0 }, Hold { s: 3600.0, delay: 9.0, span: 0.0 }],
        ] {
            let p = run(&lim, &stops, &[], &[0.0, 30.0, 0.0], &hs);
            let want: f64 = hs.iter().map(|h| h.delay).sum();
            assert!((p.duration - none.duration - want).abs() < 0.011 * hs.len() as f64, "{hs:?}: {} vs {}", p.duration - none.duration, want);
            assert!(!p.phases.iter().any(|ph| ph.v == 0.0 && ph.s > 0.0 && ph.s < 5000.0 && ph.s != 2000.0), "{hs:?}: stands between stops");
        }
        // A hold right after a stop becomes dwell there.
        let p = run(&top(), &stops, &[], &[0.0, 30.0, 0.0], &[Hold { s: 2010.0, delay: 40.0, span: 0.0 }]);
        assert!((p.dep[1] - p.arr[1] - 70.0).abs() < 1e-9);
    }
}
