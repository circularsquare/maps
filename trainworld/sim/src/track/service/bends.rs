//! T-4075: the points a train is drawn through between two of its line's samples where the track bends
//! between them. The renderer places a train by its offset along the line, between samples `step` m apart
//! (`TrackApi::run_polyline`); on a tight curve the straight between two samples lies metres off the track
//! (7 m on New York's subway, more on a tram's street corner). An interval whose track strays from that
//! straight by more than `tol` is cut into 2, 4, 8, 16 or 32 equal parts, and its inner points are drawn
//! through, so the renderer finds a point in O(1) still.

/// The finest cut: an interval in this many equal parts.
pub const FINE: usize = 32;

type P = (f64, f64);

fn lerp(a: P, b: P, f: f64) -> P {
    (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f)
}

fn dist(a: P, b: P) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// The furthest any of the interval's fine points `q` lies from where it is drawn when cut in `parts`.
fn worst(q: &[P; FINE + 1], parts: usize) -> f64 {
    let w = FINE / parts;
    (0..=FINE).fold(0.0, |m: f64, i| {
        let j = (i / w).min(parts - 1);
        m.max(dist(q[i], lerp(q[j * w], q[(j + 1) * w], (i - j * w) as f64 / w as f64)))
    })
}

/// For a run `len` m long sampled every `step` m, `pos` its point (plane units) at an offset (ground m):
/// per interval that bends, its index, how many inner points (1, 3, 7, 15 or 31), then the points, x and y
/// each, the interval's ends excluded. `tol`: ground metres.
pub fn bends(len: f64, step: f64, tol: f64, pos: impl Fn(f64) -> P) -> Vec<f64> {
    bends_in(len, step, tol, 0, usize::MAX, pos)
}

/// T-4281: `bends` of the intervals `from..to` alone (clamped to the run's), the same numbers as the whole run's give
/// there.
pub fn bends_in(len: f64, step: f64, tol: f64, from: usize, to: usize, pos: impl Fn(f64) -> P) -> Vec<f64> {
    let mut out = vec![];
    if !(len > 0.0 && step > 0.0) {
        return out;
    }
    let n = (len / step).ceil() as usize;
    let (from, to) = (from.min(n), to.min(n));
    // (the interval's start as the whole run reaches it: the end of the one before, worked out as there)
    let mut p0 = if from == 0 {
        pos(0.0)
    } else {
        let d0 = (from - 1) as f64 * step;
        pos(d0 + ((from as f64 * step).min(len) - d0))
    };
    let mut q = [(0.0, 0.0); FINE + 1];
    for k in from..to {
        let d0 = k as f64 * step;
        let h = ((k + 1) as f64 * step).min(len) - d0;
        let p1 = pos(d0 + h);
        let (a, p) = (p0, p1);
        p0 = p1;
        if h <= 1e-6 {
            continue;
        }
        let quarter = [pos(d0 + h * 0.25), pos(d0 + h * 0.5), pos(d0 + h * 0.75)];
        // plane units a ground metre here, from the path through the quarters (a curve's chord would be short)
        let path = dist(a, quarter[0]) + dist(quarter[0], quarter[1]) + dist(quarter[1], quarter[2]) + dist(quarter[2], p);
        let tp = tol * (path / h).max(1e-9);
        if quarter.iter().zip([0.25, 0.5, 0.75]).all(|(&c, f)| dist(c, lerp(a, p, f)) <= tp) {
            continue;
        }
        q[0] = a;
        q[FINE] = p;
        const Q: usize = FINE / 4;
        for (i, x) in q.iter_mut().enumerate().take(FINE).skip(1) {
            *x = if i % Q == 0 { quarter[i / Q - 1] } else { pos(d0 + h * i as f64 / FINE as f64) };
        }
        let mut parts = 2;
        while parts < FINE && worst(&q, parts) > tp {
            parts *= 2;
        }
        out.extend([k as f64, (parts - 1) as f64]);
        let w = FINE / parts;
        for j in 1..parts {
            out.extend([q[j * w].0, q[j * w].1]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point drawn at offset `d`: between the samples, through any bend points (the train shader's `lineAt`).
    fn drawn(len: f64, step: f64, b: &[f64], pos: &impl Fn(f64) -> P, d: f64) -> P {
        let n = (len / step).ceil() as usize;
        let k = ((d / step).floor() as usize).min(n - 1);
        let (d0, d1) = (k as f64 * step, ((k + 1) as f64 * step).min(len));
        let f = (d - d0) / (d1 - d0);
        let mut pts = vec![pos(d0)];
        let mut i = 0;
        while i < b.len() {
            let (kk, m) = (b[i] as usize, b[i + 1] as usize);
            if kk == k {
                pts.extend((0..m).map(|j| (b[i + 2 + 2 * j], b[i + 3 + 2 * j])));
            }
            i += 2 + 2 * m;
        }
        pts.push(pos(d1));
        let u = f * (pts.len() - 1) as f64;
        let j = (u.floor() as usize).min(pts.len() - 2);
        lerp(pts[j], pts[j + 1], u - j as f64)
    }

    fn worst_off(len: f64, step: f64, tol: f64, pos: impl Fn(f64) -> P) -> (f64, usize) {
        let b = bends(len, step, tol, &pos);
        let mut w: f64 = 0.0;
        let mut d = 0.0;
        while d <= len {
            w = w.max(dist(drawn(len, step, &b, &pos, d), pos(d)));
            d += 0.1;
        }
        let mut cut = 0;
        let mut i = 0;
        while i < b.len() {
            cut += 1;
            i += 2 + 2 * b[i + 1] as usize;
        }
        (w, cut)
    }

    #[test]
    fn a_straight_has_no_bends() {
        let (w, cut) = worst_off(437.0, 50.0, 0.25, |s| (3.0 + s * 0.6, -2.0 + s * 0.8));
        assert_eq!(cut, 0);
        assert!(w < 1e-9, "{w}");
    }

    #[test]
    fn a_tram_curve_is_drawn_on_its_track() {
        // a tram's tightest curve, 18 m, half a turn and back onto a straight
        let r = 18.0;
        let arc = std::f64::consts::PI * r;
        let pos = |s: f64| if s < arc { (r * (s / r).sin(), r - r * (s / r).cos()) } else { (-(s - arc), 2.0 * r) };
        let (before, _) = worst_off(arc + 60.0, 50.0, 1e9, pos);
        let (after, cut) = worst_off(arc + 60.0, 50.0, 0.25, pos);
        assert!(before > 10.0, "{before}");
        assert!(after <= 0.25, "{after}");
        assert!(cut >= 2, "{cut}");
    }

    #[test]
    fn a_scaled_plane_keeps_ground_tolerance() {
        // plane units twice ground metres (a Mercator plane at 60 degrees): 0.25 ground m is 0.5 plane
        let r = 300.0;
        let pos = |s: f64| (2.0 * r * (s / r).sin(), 2.0 * (r - r * (s / r).cos()));
        let (after, _) = worst_off(400.0, 50.0, 0.25, pos);
        assert!(after <= 0.5 + 1e-9, "{after}");
    }

    #[test]
    fn a_corner_is_cut_finest() {
        // a street corner at a right angle 20 m into an interval
        let pos = |s: f64| if s < 20.0 { (s, 0.0) } else { (20.0, s - 20.0) };
        let (w, cut) = worst_off(100.0, 50.0, 0.25, pos);
        assert_eq!(cut, 1);
        assert!(w < 1.2, "{w}");
    }
}
