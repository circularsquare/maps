//! A real railway's line as our track (T-727c, notes/T-727.md): the corners (PIs) whose fitted
//! alignment, straights and circular arcs as `geom::fit` makes them, stays within a tolerance of
//! the real line, for the track tool to place through `add_route`'s clicked corners.
//!
//! The real line is thinned (Douglas-Peucker at 0.6 of the tolerance), fitted with auto radii, and
//! measured every 10 m against the fit; a span still out of tolerance gets the real vertex nearest
//! its worst point as a corner, until none is. Then every curve under the minimum radius (the
//! route changing between two parallel tracks, a station throat) is widened to it where its legs
//! have room, else its corner is dropped, or a neighbour with it, whichever leaves the line
//! nearest. Distances are ground metres; in a scaled frame the plane ones are divided by `k`.

use super::frame::merc_k;
use super::geom::{self, Fit, Pi};
use super::params::*;
use std::f64::consts::PI;

/// The share of the tolerance the first thinning keeps to: the arcs cut corners, so the fit
/// lies a little further from the line than the thinned polyline does.
const THIN_SHARE: f64 = 0.6;
/// Real line sampled this often (ground m) to measure the fit.
const SAMPLE_M: f64 = 10.0;
/// Rounds of adding corners where the fit is out of tolerance.
const MAX_ROUNDS: usize = 40;

/// What `trace` made of a real line.
#[derive(Clone, Debug)]
pub struct Traced {
    /// The corners between the two ends, at `level`; radius 0 (auto) except where widened.
    pub pis: Vec<Pi>,
    /// The fit's largest and mean distance from the real line, ground m.
    pub worst: f64,
    pub mean: f64,
    /// Curves widened to the minimum radius, and corners dropped for one.
    pub widened: usize,
    pub dropped: usize,
}

/// `line`: the real line in plane metres, its first and last points the route's two ends. `tol`
/// the distance the fit may lie from it (ground m), `r_min` the smallest radius (ground m; 100,
/// or 108 for quad track's inner pair). None if the line is too short to fit or the result does
/// not fit (it never should).
pub fn trace(line: &[(f64, f64)], scaled: bool, tol: f64, r_min: f64, level: i8) -> Option<Traced> {
    let mut t = Tracer::new(line, scaled, tol, r_min, level);
    while !t.step() {}
    t.take()
}

/// `trace` in steps, each a round of adding corners or one tight curve eased (tens of ms at
/// most on a 5,000 km route), so a caller can give up on it between steps (T-754).
pub struct Tracer {
    line: Vec<(f64, f64)>,
    samples: Vec<((f64, f64), usize)>,
    scaled: bool,
    tol: f64,
    r_min: f64,
    level: i8,
    keep: Vec<usize>,
    rounds: usize,
    corners: Option<Corners>,
    tried: usize,
    widened: usize,
    dropped: usize,
    sweep: Option<Sweep>,
    pis: Vec<Pi>,
    done: Option<Option<Traced>>,
}

enum Eased {
    Widened(Corners),
    Dropped(Corners),
    Done,
    Failed,
}

impl Tracer {
    pub fn new(line: &[(f64, f64)], scaled: bool, tol: f64, r_min: f64, level: i8) -> Tracer {
        let line = dedup(line);
        let mut t = Tracer { line, samples: vec![], scaled, tol, r_min, level, keep: vec![], rounds: 0, corners: None, tried: 0, widened: 0, dropped: 0, sweep: None, pis: vec![], done: None };
        if t.line.len() < 2 {
            t.done = Some(None);
            return t;
        }
        t.samples = densify(&t.line, scaled);
        t.keep = thin(&t.line, &|y| tol * THIN_SHARE * if scaled { merc_k(y) } else { 1.0 });
        t
    }

    /// One step; true once the result is ready.
    pub fn step(&mut self) -> bool {
        if self.done.is_some() {
            return true;
        }
        if let Some(s) = &mut self.sweep {
            if !s.run(&self.samples, self.scaled) {
                return false;
            }
            let off = self.sweep.take().map(|s| s.off).unwrap_or_default();
            if self.corners.is_none() {
                // a round of adding corners where the fit is out of tolerance
                self.rounds += 1;
                let add = worst_per_span(&self.keep, &self.samples, &off, self.tol);
                let more = !add.is_empty();
                if more {
                    self.keep.extend(add);
                    self.keep.sort_unstable();
                    self.keep.dedup();
                }
                if !more || self.rounds >= MAX_ROUNDS {
                    let keep = std::mem::take(&mut self.keep);
                    self.corners = Some(Corners { radius: vec![0.0; keep.len()], at: vec![None; keep.len()], keep });
                }
                return false;
            }
            let worst = off.iter().cloned().fold(0.0, f64::max);
            let mean = off.iter().sum::<f64>() / off.len().max(1) as f64;
            self.done = Some(Some(Traced { pis: std::mem::take(&mut self.pis), worst, mean, widened: self.widened, dropped: self.dropped }));
            return true;
        }
        let Some(c) = &self.corners else {
            let v: Vec<Pi> = self.keep.iter().map(|&i| Pi::new(self.line[i].0, self.line[i].1, 0.0, self.level).quantised()).collect();
            self.sweep = Some(Sweep::new(&geom::fit_in(&v, self.scaled), SWEEP_CHUNK));
            return false;
        };
        if self.tried >= c.keep.len() {
            return self.finish();
        }
        self.tried += 1;
        match self.ease(c) {
            Eased::Widened(c2) => {
                self.corners = Some(c2);
                self.widened += 1;
                false
            }
            Eased::Dropped(c2) => {
                self.corners = Some(c2);
                self.dropped += 1;
                false
            }
            Eased::Done => self.finish(),
            Eased::Failed => {
                self.done = Some(None);
                true
            }
        }
    }

    /// The result, once `step` has said it is ready.
    pub fn take(&mut self) -> Option<Traced> {
        self.done.take().flatten()
    }

    /// The tightest curve under the minimum radius widened to it, or its corners dropped or merged.
    fn ease(&self, c: &Corners) -> Eased {
        let (line, samples, scaled, level, r_min, tol) = (&self.line[..], &self.samples[..], self.scaled, self.level, self.r_min, self.tol);
        let f = c.fit(line, scaled, level);
        let tight_at = |f: &Fit, j: usize| f.radius[j] > 0.0 && f.radius[j] < r_min - 1e-6;
        let Some(i) = (1..c.keep.len() - 1).filter(|&j| tight_at(&f, j)).min_by(|&a, &b| f.radius[a].total_cmp(&f.radius[b])) else { return Eased::Done };
        // widen in place when the legs have room beside the neighbours' tangents
        let v = c.verts(line, level);
        let leg = |a: usize, b: usize| (v[b].x - v[a].x).hypot(v[b].y - v[a].y);
        let turn = turn_at(&v, i);
        let need = r_min * if scaled { merc_k(v[i].y) } else { 1.0 } * (turn / 2.0).tan();
        if turn < PI - 1e-3 && need + f.tangent[i - 1] <= leg(i - 1, i) - FIT_TOL && need + f.tangent[i + 1] <= leg(i, i + 1) - FIT_TOL {
            let mut c2 = c.clone();
            c2.radius[i] = r_min + 1e-3;
            if c2.near(line, scaled, level, i).issues(|_| true) == 0 && c2.worst_near(line, samples, scaled, level, i) <= 2.0 * tol {
                return Eased::Widened(c2);
            }
        }
        // else drop it with up to three neighbours each side, or put one corner where the legs
        // either side of those meet, at the minimum radius; keep what leaves the line nearest
        let before = (i.saturating_sub(6).max(1)..(i + 7).min(f.radius.len() - 1)).filter(|&j| tight_at(&f, j)).count();
        let mut best: Option<(f64, Corners)> = None;
        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 3)] {
            let (lo, hi) = (i.saturating_sub(a), i + b);
            if lo < 1 || hi > c.keep.len() - 2 {
                continue;
            }
            for merge in [false, true] {
                let Some(c2) = (if merge { c.merged(&v, lo, hi, r_min + 1e-3) } else { Some(c.without(lo, hi)) }) else { continue };
                let at = lo.min(c2.keep.len() - 2).max(1);
                let w2 = c2.near(line, scaled, level, at);
                if w2.tight(at.saturating_sub(6), at + 7, r_min) >= before || w2.issues(|g| !matches!(g, geom::GeomIssue::RadiusTooSmall(_))) > 0 {
                    continue;
                }
                let w = c2.worst_near(line, samples, scaled, level, at);
                if best.as_ref().map_or(true, |bst| w < bst.0) {
                    best = Some((w, c2));
                }
            }
        }
        match best {
            Some((_, c2)) => Eased::Dropped(c2),
            None => Eased::Failed,
        }
    }

    /// Checks the eased alignment and starts measuring it; true if it does not fit.
    fn finish(&mut self) -> bool {
        let Some(c) = &self.corners else { return true };
        let v = c.verts(&self.line, self.level);
        let f = geom::fit_in(&v, self.scaled);
        if !f.issues.is_empty() {
            self.done = Some(None);
            return true;
        }
        self.sweep = Some(Sweep::new(&f, SWEEP_CHUNK));
        self.pis = v[1..v.len() - 1].to_vec();
        false
    }
}

/// The alignment's vertices: real vertices by index (ends included), each with its radius (0
/// auto) and, for a corner put where two legs meet, its own position.
#[derive(Clone)]
struct Corners {
    keep: Vec<usize>,
    radius: Vec<f64>,
    at: Vec<Option<(f64, f64)>>,
}

impl Corners {
    fn verts(&self, line: &[(f64, f64)], level: i8) -> Vec<Pi> {
        (0..self.keep.len())
            .map(|j| {
                let (x, y) = self.at[j].unwrap_or(line[self.keep[j]]);
                Pi::new(x, y, self.radius[j], level).quantised()
            })
            .collect()
    }
    fn fit(&self, line: &[(f64, f64)], scaled: bool, level: i8) -> Fit {
        geom::fit_in(&self.verts(line, level), scaled)
    }
    /// The fit of the corners around corner `at` alone: a corner's arc depends only on its two
    /// legs, so away from the slice's ends it is the whole alignment's.
    fn near(&self, line: &[(f64, f64)], scaled: bool, level: i8, at: usize) -> Near {
        let last = self.keep.len() - 1;
        let (a, b) = (at.saturating_sub(NEAR), (at + NEAR).min(last));
        let fit = geom::fit_in(&self.slice(a, b).verts(line, level), scaled);
        // the slice's first and last corners are its ends, so the corners beside them lose the
        // half-leg rule, and an arc set beside those may then be blamed
        let lo = if a == 0 { 1 } else { a + 3 };
        let hi = if b == last { last - 1 } else { b - 3 };
        Near { fit, a, lo, hi }
    }
    fn slice(&self, lo: usize, hi: usize) -> Corners {
        Corners { keep: self.keep[lo..=hi].to_vec(), radius: self.radius[lo..=hi].to_vec(), at: self.at[lo..=hi].to_vec() }
    }
    fn without(&self, lo: usize, hi: usize) -> Corners {
        let mut c = self.clone();
        c.keep.drain(lo..=hi);
        c.radius.drain(lo..=hi);
        c.at.drain(lo..=hi);
        c
    }
    /// Corners lo..=hi as one where the leg into lo and the leg out of hi cross, radius `r`.
    fn merged(&self, v: &[Pi], lo: usize, hi: usize, r: f64) -> Option<Corners> {
        let (p, d1) = ((v[lo - 1].x, v[lo - 1].y), (v[lo].x - v[lo - 1].x, v[lo].y - v[lo - 1].y));
        let (q, d2) = ((v[hi + 1].x, v[hi + 1].y), (v[hi].x - v[hi + 1].x, v[hi].y - v[hi + 1].y));
        let den = d1.0 * d2.1 - d1.1 * d2.0;
        if den.abs() < 1e-9 {
            return None;
        }
        // p + s d1 = q + t d2, both s and t forward (towards the corner)
        let s = ((q.0 - p.0) * d2.1 - (q.1 - p.1) * d2.0) / den;
        let t = ((q.0 - p.0) * d1.1 - (q.1 - p.1) * d1.0) / den;
        if s <= 0.0 || t <= 0.0 {
            return None;
        }
        let mut c = if hi > lo { self.without(lo + 1, hi) } else { self.clone() };
        c.at[lo] = Some((p.0 + s * d1.0, p.1 + s * d1.1));
        c.radius[lo] = r;
        c.keep[lo] = (self.keep[lo] + self.keep[hi]) / 2;
        Some(c)
    }
    /// The worst offset over the samples between the corners two either side of corner `at`.
    fn worst_near(&self, line: &[(f64, f64)], samples: &[((f64, f64), usize)], scaled: bool, level: i8, at: usize) -> f64 {
        let lo = at.saturating_sub(4);
        let hi = (at + 4).min(self.keep.len() - 1);
        let (a, b) = (self.keep[at.saturating_sub(2)], self.keep[(at + 2).min(self.keep.len() - 1)]);
        let sub = self.slice(lo, hi);
        // the slice's ends are corners in the whole alignment, so only its middle is compared
        let mut v = sub.verts(line, level);
        let n = v.len();
        v[0].radius = 0.0;
        v[n - 1].radius = 0.0;
        // samples are in order along the line
        let (from, to) = (samples.partition_point(|s| s.1 < a), samples.partition_point(|s| s.1 < b));
        sweep(&geom::fit_in(&v, scaled), &samples[from..to], scaled).into_iter().fold(0.0, f64::max)
    }
}

/// Corners either side of the one `Corners::near` fits around.
const NEAR: usize = 12;

/// A fit of some corners, indexed as the whole alignment is, trusted from corner `lo` to `hi`.
struct Near {
    fit: Fit,
    a: usize,
    lo: usize,
    hi: usize,
}

impl Near {
    /// Issues at the trusted corners (or legs) that `pick` picks.
    fn issues(&self, pick: impl Fn(&geom::GeomIssue) -> bool) -> usize {
        use geom::GeomIssue::*;
        self.fit
            .issues
            .iter()
            .filter(|g| {
                let (ZeroLeg(j) | Reversal(j) | RadiusTooSmall(j) | ArcDoesNotFit(j) | RampTooShort(j) | LevelRange(j)) = **g;
                (self.lo..=self.hi).contains(&(self.a + j)) && pick(g)
            })
            .count()
    }
    /// Corners from..to (whole-alignment indices, within the trusted ones) under `r_min`.
    fn tight(&self, from: usize, to: usize, r_min: f64) -> usize {
        (from.max(self.lo)..to.min(self.hi + 1)).filter(|&j| {
            let r = self.fit.radius[j - self.a];
            r > 0.0 && r < r_min - 1e-6
        }).count()
    }
}

fn turn_at(v: &[Pi], i: usize) -> f64 {
    geom::wrap((v[i + 1].y - v[i].y).atan2(v[i + 1].x - v[i].x) - (v[i].y - v[i - 1].y).atan2(v[i].x - v[i - 1].x)).abs()
}

fn dedup(line: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(line.len());
    for &p in line {
        if out.last().map_or(true, |q| (p.0 - q.0).hypot(p.1 - q.1) > 1e-3) {
            out.push(p);
        }
    }
    out
}

/// Douglas-Peucker: the indices kept, ends included, sorted.
fn thin(line: &[(f64, f64)], tol: &dyn Fn(f64) -> f64) -> Vec<usize> {
    let n = line.len();
    let mut keep = vec![0, n - 1];
    let mut stack = vec![(0, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (a, b) = (line[i], line[j]);
        let (mut far, mut at) = (0.0, 0);
        for (m, &p) in line.iter().enumerate().take(j).skip(i + 1) {
            let d = seg_dist(p, a, b) - tol(p.1);
            if d > far {
                far = d;
                at = m;
            }
        }
        if far > 0.0 {
            keep.push(at);
            stack.push((i, at));
            stack.push((at, j));
        }
    }
    keep.sort_unstable();
    keep
}

fn seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

/// The real line every `SAMPLE_M` ground metres: point and the line segment it lies on.
fn densify(line: &[(f64, f64)], scaled: bool) -> Vec<((f64, f64), usize)> {
    let mut out = vec![(line[0], 0)];
    for i in 0..line.len() - 1 {
        let (a, b) = (line[i], line[i + 1]);
        let k = if scaled { merc_k((a.1 + b.1) / 2.0) } else { 1.0 };
        let n = (((b.0 - a.0).hypot(b.1 - a.1)) / (SAMPLE_M * k)).ceil().max(1.0) as usize;
        for s in 1..=n {
            let t = s as f64 / n as f64;
            out.push(((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)), i));
        }
    }
    out
}

/// Distances from points in order along the line to pieces in order along the fit: each point
/// looks at the pieces near the last one it matched.
fn sweep(f: &Fit, samples: &[((f64, f64), usize)], scaled: bool) -> Vec<f64> {
    let mut s = Sweep::new(f, usize::MAX);
    s.run(samples, scaled);
    s.off
}

/// Samples measured in one `Sweep::run`: a few ms of a step.
const SWEEP_CHUNK: usize = 150_000;

/// `sweep` a chunk of samples at a time (`Tracer::step`).
struct Sweep {
    shapes: Vec<Shape>,
    cur: usize,
    chunk: usize,
    off: Vec<f64>,
}

impl Sweep {
    fn new(f: &Fit, chunk: usize) -> Sweep {
        Sweep { shapes: f.pieces.iter().map(Shape::of).collect(), cur: 0, chunk, off: vec![] }
    }
    /// The next chunk of `samples` (the same every call); true once all are measured.
    fn run(&mut self, samples: &[((f64, f64), usize)], scaled: bool) -> bool {
        let (from, n) = (self.off.len(), self.shapes.len());
        let to = from.saturating_add(self.chunk).min(samples.len());
        for &((x, y), _) in &samples[from..to] {
            if n == 0 {
                self.off.push(f64::INFINITY);
                continue;
            }
            // pieces are hundreds of metres and samples 10 m apart, so the match moves slowly; the
            // nearest of the window, the first of equals, with the last match measured first so the
            // bounding circles rule most of the others out
            let cur = self.cur;
            let (lo, hi) = (cur.saturating_sub(1), (cur + 8).min(n));
            let mut best = (self.shapes[cur].dist(x, y), cur);
            for (j, s) in self.shapes.iter().enumerate().take(hi).skip(lo) {
                if j == cur || s.beyond(x, y, best.0) {
                    continue;
                }
                let d = s.dist(x, y);
                if d < best.0 || (d == best.0 && j < best.1) {
                    best = (d, j);
                }
            }
            self.cur = best.1;
            self.off.push(best.0 / if scaled { merc_k(y) } else { 1.0 });
        }
        to == samples.len()
    }
}

/// A piece as the distance test wants it: its ends, an arc's centre, radius, start angle and
/// signed turn, and a circle holding the whole piece.
struct Shape {
    a: (f64, f64),
    b: (f64, f64),
    arc: Option<(f64, f64, f64, f64, f64)>,
    mid: (f64, f64),
    reach: f64,
}

impl Shape {
    fn of(p: &geom::Piece) -> Shape {
        let (ex, ey, _) = p.end();
        let arc = (p.k != 0.0).then(|| {
            let (cx, cy) = p.center();
            (cx, cy, 1.0 / p.k.abs(), (p.y0 - cy).atan2(p.x0 - cx), p.k * p.plen)
        });
        // an arc turning under half a circle lies inside the circle on its chord (Thales)
        let wide = arc.is_some_and(|a| a.4.abs() > PI - 1e-6);
        let mid = ((p.x0 + ex) / 2.0, (p.y0 + ey) / 2.0);
        let reach = if wide { f64::INFINITY } else { (ex - p.x0).hypot(ey - p.y0) / 2.0 + 1e-6 };
        Shape { a: (p.x0, p.y0), b: (ex, ey), arc, mid, reach }
    }
    /// Whether every point of the piece is further than `d` from (x, y).
    fn beyond(&self, x: f64, y: f64, d: f64) -> bool {
        let (dx, dy, r) = (x - self.mid.0, y - self.mid.1, self.reach + d + 1e-9);
        r.is_finite() && dx * dx + dy * dy > r * r
    }
    fn dist(&self, x: f64, y: f64) -> f64 {
        let Some((cx, cy, r, a0, turn)) = self.arc else { return seg_dist((x, y), self.a, self.b) };
        let mut t = geom::wrap((y - cy).atan2(x - cx) - a0) * turn.signum();
        if t < 0.0 {
            t += 2.0 * PI;
        }
        if t <= turn.abs() {
            ((x - cx).hypot(y - cy) - r).abs()
        } else {
            (x - self.a.0).hypot(y - self.a.1).min((x - self.b.0).hypot(y - self.b.1))
        }
    }
}

/// For each span between kept corners with a sample out of tolerance, the real vertex nearest
/// its worst sample that lies strictly inside the span.
fn worst_per_span(keep: &[usize], samples: &[((f64, f64), usize)], off: &[f64], tol: f64) -> Vec<usize> {
    let mut add = vec![];
    let mut span_worst: Option<(usize, f64, usize)> = None;
    let flush = |w: Option<(usize, f64, usize)>, add: &mut Vec<usize>| {
        if let Some((sp, _, seg)) = w {
            let (lo, hi) = (keep[sp], keep[(sp + 1).min(keep.len() - 1)]);
            if let Some(c) = [seg + 1, seg].into_iter().find(|&c| lo < c && c < hi) {
                add.push(c);
            }
        }
    };
    for (s, &d) in samples.iter().zip(off) {
        if d <= tol {
            continue;
        }
        let sp = keep.partition_point(|&k| k <= s.1).saturating_sub(1);
        match span_worst {
            Some((p, w, _)) if p == sp => {
                if d > w {
                    span_worst = Some((sp, d, s.1));
                }
            }
            _ => {
                flush(span_worst.take(), &mut add);
                span_worst = Some((sp, d, s.1));
            }
        }
    }
    flush(span_worst, &mut add);
    add
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A straight, a 600 m curve through 90 degrees sampled every 5 m, and a straight.
    fn bend() -> Vec<(f64, f64)> {
        let mut l: Vec<(f64, f64)> = (0..=200).map(|i| (i as f64 * 10.0, 0.0)).collect();
        let r = 600.0;
        for i in 1..=188 {
            let a = i as f64 * 5.0 / r;
            if a > PI / 2.0 {
                break;
            }
            l.push((2000.0 + r * a.sin(), r - r * a.cos()));
        }
        for i in 1..=200 {
            l.push((2600.0, 600.0 + i as f64 * 10.0));
        }
        l
    }

    #[test]
    fn a_bend_within_tolerance() {
        let t = trace(&bend(), false, 25.0, MIN_RADIUS, 0).unwrap();
        assert!(t.worst <= 25.0, "worst {}", t.worst);
        assert!(t.pis.len() <= 6, "{} corners", t.pis.len());
        let mut v = vec![Pi::new(0.0, 0.0, 0.0, 0)];
        v.extend(t.pis.iter().cloned());
        v.push(Pi::new(2600.0, 2600.0, 0.0, 0));
        let f = geom::fit_in(&v, false);
        assert!(f.issues.is_empty());
        let rmin = f.radius.iter().cloned().filter(|&r| r > 0.0).fold(f64::INFINITY, f64::min);
        assert!(rmin > 400.0, "tightest {rmin}");
    }

    #[test]
    fn a_jog_between_tracks_is_smoothed() {
        // two straights 4 m apart joined by a 15 m sideways step, as a route changing track does
        let mut l: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64 * 10.0, 0.0)).collect();
        l.push((1015.0, 4.0));
        l.extend((1..=100).map(|i| (1015.0 + i as f64 * 10.0, 4.0)));
        let t = trace(&l, false, 25.0, MIN_RADIUS, 0).unwrap();
        let mut v = vec![Pi::new(0.0, 0.0, 0.0, 0)];
        v.extend(t.pis.iter().cloned());
        v.push(Pi::new(2015.0, 4.0, 0.0, 0));
        assert!(geom::fit_in(&v, false).issues.is_empty());
        assert!(t.worst <= 25.0);
    }
}
