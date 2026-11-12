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
    /// the segments in time order, a ramp's held as the departure it leads to (T-2722: a stop with
    /// hourly trains has ~600 segments, nearly all of them ramp)
    p: Vec<Piece>,
    len: usize,
    /// marks in all (`Piece::mark`)
    pub marks: usize,
    /// every share 1: riders arrive as if unmasked
    pub all_one: bool,
    pub phi: [f64; PERIODS],
}

/// A segment's place in a mask: piece and segment within it.
pub type At = (usize, usize);

/// A sum of riders the boarding account keeps (`Masked::pre`) at every piece's first segment and
/// every `MARK_EVERY`th of a ramp; the segments between are added up again when needed.
pub const MARK_EVERY: usize = 8;

/// One segment (`h` 0: from `t0`, share `x`), or a ramp of `n` segments before the departure at
/// `x`, the `j`th from `t0 + j * h` with share `within(x - start - h / 2)`, the same arithmetic
/// `stop_mask` made them with. `mark`: the index of its first mark.
#[derive(Clone, Copy, Debug)]
struct Piece {
    mark: u32,
    n: u32,
    t0: f64,
    h: f64,
    x: f64,
}

impl Piece {
    fn start(&self, j: usize) -> f64 {
        if self.h == 0.0 { self.t0 } else { self.t0 + j as f64 * self.h }
    }
    fn share(&self, start: f64) -> f64 {
        if self.h == 0.0 { self.x } else { within(self.x - start - 0.5 * self.h) }
    }
}

impl Mask {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The last segment starting at or before `s` (the first if none).
    pub fn find(&self, s: f64) -> At {
        let k = self.p.partition_point(|p| p.t0 <= s).saturating_sub(1);
        let Some(p) = self.p.get(k) else { return (0, 0) };
        let n = p.n as usize;
        if p.h == 0.0 || n == 1 {
            return (k, 0);
        }
        let mut j = (((s - p.t0) / p.h).floor().max(0.0) as usize).min(n - 1);
        while j + 1 < n && p.start(j + 1) <= s {
            j += 1;
        }
        while j > 0 && p.start(j) > s {
            j -= 1;
        }
        (k, j)
    }
    /// A segment's start and share.
    pub fn seg(&self, (k, j): At) -> (f64, f64) {
        let p = &self.p[k];
        let t = p.start(j);
        (t, p.share(t))
    }
    /// The mark at or before a segment, and that mark's segment.
    pub fn mark_before(&self, (k, j): At) -> (usize, At) {
        let i = j / MARK_EVERY;
        (self.p[k].mark as usize + i, (k, i * MARK_EVERY))
    }
    /// The segments from `at` on: start, end (the next one's start, the last's the end of the
    /// day), share, and whether a mark falls on it.
    pub fn segs_from(&self, (k, j): At) -> Segs<'_> {
        let next = self.p.get(k).map(|p| p.start(j));
        Segs { m: self, k, j, next }
    }
    #[cfg(test)]
    fn expand(&self) -> Vec<(f64, f64)> {
        self.segs_from((0, 0)).map(|(t, _, s, _)| (t, s)).collect()
    }
    /// Heap bytes held (benches).
    pub fn heap_bytes(&self) -> usize {
        self.p.capacity() * std::mem::size_of::<Piece>()
    }
}

pub struct Segs<'a> {
    m: &'a Mask,
    k: usize,
    j: usize,
    /// the current segment's start
    next: Option<f64>,
}

impl Iterator for Segs<'_> {
    type Item = (f64, f64, f64, bool);
    fn next(&mut self) -> Option<Self::Item> {
        let t = self.next?;
        let p = &self.m.p[self.k];
        let s = p.share(t);
        let mark = self.j % MARK_EVERY == 0;
        self.j += 1;
        if self.j >= p.n as usize {
            self.k += 1;
            self.j = 0;
        }
        self.next = self.m.p.get(self.k).map(|q| q.start(self.j));
        Some((t, self.next.unwrap_or(DAY), s, mark))
    }
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
    // where each whole ramp starts in `seg`: its first segment's index, start, step and departure
    let mut ramps: Vec<(usize, usize, f64, f64, f64)> = vec![];
    let mut phi = [0.0; PERIODS];
    let mut order: [usize; PERIODS] = std::array::from_fn(|q| q);
    order.sort_by(|&a, &b| PERIOD_HOURS[a].0.total_cmp(&PERIOD_HOURS[b].0));
    for q in order {
        let (a, b) = (PERIOD_HOURS[q].0 * 3600.0, PERIOD_HOURS[q].1 * 3600.0);
        // a new period starts a segment of its own
        let mut fresh = true;
        // true when it adds a segment
        let mut push = |seg: &mut Vec<(f64, f64)>, u0: f64, u1: f64, s: f64, tot: &mut f64| {
            let mut added = false;
            if u1 > u0 {
                *tot += (period_cum(q, u1) - period_cum(q, u0)) * s;
                match seg.last() {
                    Some(&(_, x)) if x == s && !fresh => {}
                    _ => {
                        seg.push((u0, s));
                        added = true;
                    }
                }
                fresh = false;
            }
            added
        };
        let mut j = ext.partition_point(|&d| d <= a);
        let (mut t, mut tot) = (a, 0.0f64);
        while t < b {
            let Some(&d) = ext.get(j) else {
                push(&mut seg, t, b, 0.0, &mut tot);
                break;
            };
            let end = d.min(b);
            push(&mut seg, t, end.min(d - TOL_MAX_S), 0.0, &mut tot);
            let (p0, p1) = (t.max(d - TOL_MAX_S), end.min(d - TOL_MIN_S));
            if p1 > p0 {
                let m = ((p1 - p0) / STEP_S).ceil().max(1.0) as usize;
                let h = (p1 - p0) / m as f64;
                let mut added = 0;
                let at = seg.len();
                for i in 0..m {
                    let u0 = p0 + i as f64 * h;
                    added += push(&mut seg, u0, u0 + h, within(d - u0 - 0.5 * h), &mut tot) as usize;
                }
                // every step its own segment: kept as the ramp
                if added == m && m > 1 && h > 0.0 {
                    ramps.push((at, m, p0, h, d));
                }
            }
            // a train due within the shortest tolerance: everyone
            push(&mut seg, t.max(d - TOL_MIN_S), end, 1.0, &mut tot);
            t = d;
            j += 1;
        }
        phi[q] = tot.clamp(0.0, 1.0);
    }
    let mut p: Vec<Piece> = Vec::with_capacity(seg.len().min(4 * ramps.len() + 8));
    let mut ramps = ramps.into_iter().peekable();
    let (mut i, mut marks) = (0, 0);
    while i < seg.len() {
        match ramps.peek() {
            Some(&(at, n, t0, h, x)) if at == i => {
                p.push(Piece { mark: marks as u32, n: n as u32, t0, h, x });
                ramps.next();
                i += n;
                marks += n.div_ceil(MARK_EVERY);
            }
            _ => {
                p.push(Piece { mark: marks as u32, n: 1, t0: seg[i].0, h: 0.0, x: seg[i].1 });
                i += 1;
                marks += 1;
            }
        }
    }
    p.shrink_to_fit();
    let mask = Mask { p, len: seg.len(), marks, all_one: seg.iter().all(|x| x.1 == 1.0), phi };
    #[cfg(test)]
    {
        let e = mask.expand();
        assert!(e.len() == seg.len() && e.iter().zip(&seg).all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()), "a mask's pieces give back its segments");
    }
    mask
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
/// boarding account's layout), from its steady pattern `lat`; -1 at a stop the run passes on the other
/// track (T-3332), where demand boards and alights nobody.
pub fn line_span(ls: &LineSvc, lat: &Lattice) -> Vec<f32> {
    let n = ls.stops.len();
    let mut out = vec![0.0f32; PERIODS * 2 * n];
    for r in 0..2 {
        for (k, deps) in stop_deps(ls, lat, r).iter().enumerate() {
            for (q, phi) in stop_phi(deps).into_iter().enumerate() {
                out[(q * 2 + r) * n + k] = if ls.runs[r].passes(k) { -1.0 } else { phi };
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
        add(lat.phase[lev].to_bits());
    }
    add(lat.lead.to_bits());
    for r in 0..2 {
        for (k, _) in ls.runs[r].pass.iter().enumerate().filter(|x| *x.1) {
            add(k as u64 | 1 << 40);
        }
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

    #[test]
    fn a_masks_ramps_find_the_segment_a_list_would() {
        // stop_mask itself checks its pieces give back its segments
        let m = stop_mask(&[5.0 * 3600.0 + 17.0, 7.5 * 3600.0, 8.0 * 3600.0 + 0.3, 13.0 * 3600.0, 21.0 * 3600.0 + 59.0]);
        let e = m.expand();
        assert!(e.len() > 100 && m.p.len() < 40, "{} segments in {} pieces", e.len(), m.p.len());
        for k in 0..20000 {
            let s = k as f64 * 4.3201 - 10.0;
            let i = e.partition_point(|x| x.0 <= s).saturating_sub(1);
            let at = m.find(s);
            assert_eq!(m.seg(at), e[i], "at {s}");
            assert_eq!(m.segs_from(at).count(), e.len() - i, "at {s}");
        }
        assert_eq!(m.segs_from((0, 0)).filter(|x| x.3).count(), m.marks);
    }
}
