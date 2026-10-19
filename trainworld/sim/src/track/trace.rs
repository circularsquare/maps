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
    let line = dedup(line);
    if line.len() < 2 {
        return None;
    }
    let k = |y: f64| if scaled { merc_k(y) } else { 1.0 };
    let samples = densify(&line, scaled);
    let mut keep = thin(&line, &|y| tol * THIN_SHARE * k(y));
    for _ in 0..MAX_ROUNDS {
        let off = offsets(&line, &keep, &samples, scaled, level);
        let add = worst_per_span(&keep, &samples, &off, tol);
        if add.is_empty() {
            break;
        }
        keep.extend(add);
        keep.sort_unstable();
        keep.dedup();
    }
    let mut c = Corners { radius: vec![0.0; keep.len()], at: vec![None; keep.len()], keep };
    let (mut widened, mut dropped) = (0, 0);
    let mut tried = 0;
    while tried < c.keep.len() {
        tried += 1;
        let f = c.fit(&line, scaled, level);
        let tight_at = |f: &Fit, j: usize| f.radius[j] > 0.0 && f.radius[j] < r_min - 1e-6;
        let Some(i) = (1..c.keep.len() - 1).filter(|&j| tight_at(&f, j)).min_by(|&a, &b| f.radius[a].total_cmp(&f.radius[b])) else { break };
        // widen in place when the legs have room beside the neighbours' tangents
        let v = c.verts(&line, level);
        let leg = |a: usize, b: usize| (v[b].x - v[a].x).hypot(v[b].y - v[a].y);
        let turn = turn_at(&v, i);
        let need = r_min * k(v[i].y) * (turn / 2.0).tan();
        if turn < PI - 1e-3 && need + f.tangent[i - 1] <= leg(i - 1, i) - FIT_TOL && need + f.tangent[i + 1] <= leg(i, i + 1) - FIT_TOL {
            let mut c2 = c.clone();
            c2.radius[i] = r_min + 1e-3;
            if c2.fit(&line, scaled, level).issues.is_empty() && c2.worst_near(&line, &samples, scaled, level, i) <= 2.0 * tol {
                c = c2;
                widened += 1;
                continue;
            }
        }
        // else drop it with up to three neighbours each side, or put one corner where the legs
        // either side of those meet, at the minimum radius; keep what leaves the line nearest
        let tight_near = |f: &Fit, a: usize, b: usize| (a.max(1)..b.min(f.radius.len() - 1)).filter(|&j| tight_at(f, j)).count();
        let before = tight_near(&f, i.saturating_sub(6), i + 7);
        let mut best: Option<(f64, Corners)> = None;
        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 3)] {
            let (lo, hi) = (i.saturating_sub(a), i + b);
            if lo < 1 || hi > c.keep.len() - 2 {
                continue;
            }
            for merge in [false, true] {
                let Some(c2) = (if merge { c.merged(&v, lo, hi, r_min + 1e-3) } else { Some(c.without(lo, hi)) }) else { continue };
                let f2 = c2.fit(&line, scaled, level);
                let at = lo.min(c2.keep.len() - 2).max(1);
                if tight_near(&f2, at.saturating_sub(6), at + 7) >= before || f2.issues.iter().any(|g| !matches!(g, geom::GeomIssue::RadiusTooSmall(_))) {
                    continue;
                }
                let w = c2.worst_near(&line, &samples, scaled, level, at);
                if best.as_ref().map_or(true, |bst| w < bst.0) {
                    best = Some((w, c2));
                }
            }
        }
        match best {
            Some((_, c2)) => {
                c = c2;
                dropped += 1;
            }
            None => return None,
        }
    }
    let v = c.verts(&line, level);
    if !geom::fit_in(&v, scaled).issues.is_empty() {
        return None;
    }
    let off = sweep(&c.fit(&line, scaled, level), &samples, scaled);
    let worst = off.iter().cloned().fold(0.0, f64::max);
    let mean = off.iter().sum::<f64>() / off.len().max(1) as f64;
    Some(Traced { pis: v[1..v.len() - 1].to_vec(), worst, mean, widened, dropped })
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
        let s: Vec<((f64, f64), usize)> = samples.iter().filter(|s| s.1 >= a && s.1 < b).cloned().collect();
        sweep(&geom::fit_in(&v, scaled), &s, scaled).into_iter().fold(0.0, f64::max)
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

/// Ground distance from each sample to the alignment through the real vertices `keep`, auto radii.
fn offsets(line: &[(f64, f64)], keep: &[usize], samples: &[((f64, f64), usize)], scaled: bool, level: i8) -> Vec<f64> {
    let v: Vec<Pi> = keep.iter().map(|&i| Pi::new(line[i].0, line[i].1, 0.0, level).quantised()).collect();
    sweep(&geom::fit_in(&v, scaled), samples, scaled)
}

/// Distances from points in order along the line to pieces in order along the fit: each point
/// looks at the pieces near the last one it matched.
fn sweep(f: &Fit, samples: &[((f64, f64), usize)], scaled: bool) -> Vec<f64> {
    let shapes: Vec<Shape> = f.pieces.iter().map(Shape::of).collect();
    let mut cur = 0usize;
    let n = shapes.len();
    samples
        .iter()
        .map(|&((x, y), _)| {
            // pieces are hundreds of metres and samples 10 m apart, so the match moves slowly
            let (lo, hi) = (cur.saturating_sub(1), (cur + 8).min(n));
            let mut best = (f64::INFINITY, cur);
            for (j, s) in shapes.iter().enumerate().take(hi).skip(lo) {
                let d = s.dist(x, y);
                if d < best.0 {
                    best = (d, j);
                }
            }
            cur = best.1;
            best.0 / if scaled { merc_k(y) } else { 1.0 }
        })
        .collect()
}

/// A piece as the distance test wants it: its ends, and an arc's centre, radius, start angle and
/// signed turn.
struct Shape {
    a: (f64, f64),
    b: (f64, f64),
    arc: Option<(f64, f64, f64, f64, f64)>,
}

impl Shape {
    fn of(p: &geom::Piece) -> Shape {
        let (ex, ey, _) = p.end();
        let arc = (p.k != 0.0).then(|| {
            let (cx, cy) = p.center();
            (cx, cy, 1.0 / p.k.abs(), (p.y0 - cy).atan2(p.x0 - cx), p.k * p.plen)
        });
        Shape { a: (p.x0, p.y0), b: (ex, ey), arc }
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
