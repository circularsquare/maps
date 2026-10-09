//! Alignment geometry (SPEC 6.1, notes/T-008.md 1-2): points of intersection (PIs) with a circular
//! arc fitted at each, straights between, ramps centred between vertices of different levels.
//!
//! Coordinates are local metres, x east and y north of the city origin (the city pack's frame).
//! Headings are radians counter-clockwise from east. Curvature `k` is signed: positive turns left.

use super::params::*;
use std::f64::consts::PI;

/// A vertex of an alignment: an end node or a PI. `radius` 0 = auto; ignored at the ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pi {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub level: i8,
}

impl Pi {
    pub fn new(x: f64, y: f64, radius: f64, level: i8) -> Pi {
        Pi { x, y, radius, level }
    }
    /// The same PI on the placement grid.
    pub fn quantised(self) -> Pi {
        Pi { x: quant(self.x), y: quant(self.y), radius: if self.radius > 0.0 { quant(self.radius) } else { 0.0 }, level: self.level }
    }
}

/// One straight (`k` = 0) or circular arc, `s0` = chainage of its start along the edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub s0: f64,
    pub len: f64,
    pub x0: f64,
    pub y0: f64,
    pub th0: f64,
    pub k: f64,
}

impl Piece {
    /// Position and heading `ds` metres into the piece.
    #[inline]
    pub fn at(&self, ds: f64) -> (f64, f64, f64) {
        let th = self.th0 + self.k * ds;
        if self.k == 0.0 {
            (self.x0 + ds * self.th0.cos(), self.y0 + ds * self.th0.sin(), th)
        } else {
            (self.x0 + (th.sin() - self.th0.sin()) / self.k, self.y0 - (th.cos() - self.th0.cos()) / self.k, th)
        }
    }
    pub fn end(&self) -> (f64, f64, f64) {
        self.at(self.len)
    }
    pub fn radius(&self) -> f64 {
        if self.k == 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.k.abs()
        }
    }
    /// Curve speed limit, capped at the train's top speed (SPEC 6.1).
    pub fn v_limit(&self) -> f64 {
        if self.k == 0.0 {
            V_TOP
        } else {
            curve_speed(1.0 / self.k.abs())
        }
    }
    /// Centre of an arc.
    pub fn center(&self) -> (f64, f64) {
        (self.x0 - self.th0.sin() / self.k, self.y0 + self.th0.cos() / self.k)
    }
    /// Bounding box `[xmin, ymin, xmax, ymax]` of the part `[d0, d1]` of the piece, exact.
    pub fn bbox_part(&self, d0: f64, d1: f64) -> [f64; 4] {
        let (xa, ya, tha) = self.at(d0);
        let (xb, yb, thb) = self.at(d1);
        let mut b = [xa.min(xb), ya.min(yb), xa.max(xb), ya.max(yb)];
        if self.k != 0.0 {
            // Extremes of a circle are where the heading is a multiple of pi/2.
            let (lo, hi) = if tha < thb { (tha, thb) } else { (thb, tha) };
            let mut j = (lo / (PI / 2.0)).ceil() as i64;
            while (j as f64) * (PI / 2.0) <= hi {
                let (x, y, _) = self.at(((j as f64) * (PI / 2.0) - self.th0) / self.k);
                b[0] = b[0].min(x);
                b[1] = b[1].min(y);
                b[2] = b[2].max(x);
                b[3] = b[3].max(y);
                j += 1;
            }
        }
        b
    }
    pub fn bbox(&self) -> [f64; 4] {
        self.bbox_part(0.0, self.len)
    }
}

/// `v = sqrt(a_lat R)`, capped at the top speed.
pub fn curve_speed(radius: f64) -> f64 {
    (A_LAT * radius).sqrt().min(V_TOP)
}

/// Wrap an angle to (-pi, pi].
pub fn wrap(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a <= -PI {
        a += 2.0 * PI;
    } else if a > PI {
        a -= 2.0 * PI;
    }
    a
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeomIssue {
    /// Two consecutive vertices at the same place (index of the leg).
    ZeroLeg(usize),
    /// The alignment turns back on itself at this vertex.
    Reversal(usize),
    /// The arc at this vertex is under the minimum radius.
    RadiusTooSmall(usize),
    /// The arc at this (player-set) vertex does not fit its legs.
    ArcDoesNotFit(usize),
    /// The ramp on this leg is longer than the leg.
    RampTooShort(usize),
    /// A vertex level outside -3..+3.
    LevelRange(usize),
}

/// A fitted alignment.
#[derive(Clone, Debug, Default)]
pub struct Fit {
    pub pieces: Vec<Piece>,
    /// Chainage of each vertex: 0 and `len` at the ends, the arc midpoint at a PI.
    pub vert_s: Vec<f64>,
    /// Radius actually used at each vertex (0 where there is no arc).
    pub radius: Vec<f64>,
    /// Tangent length at each vertex.
    pub tangent: Vec<f64>,
    pub len: f64,
    /// Per piece: the leg it lies on (a straight, leg i joins vertex i and i+1) or the vertex
    /// whose arc it is.
    pub piece_at_vertex: Vec<u32>,
    /// Height breakpoints `(s, z)`, piecewise linear.
    pub vert: Vec<[f64; 2]>,
    pub issues: Vec<GeomIssue>,
}

/// Fit arcs at the interior vertices of `v` (`v[0]` and the last are the end nodes) and lay the
/// ramps. Never fails: problems are listed in `issues`, and the geometry is still produced (with a
/// clamped straight where an arc does not fit) so the drawing tool can show it red.
pub fn fit(v: &[Pi]) -> Fit {
    let n = v.len();
    let mut f = Fit { vert_s: vec![0.0; n], radius: vec![0.0; n], tangent: vec![0.0; n], ..Default::default() };
    if n < 2 {
        return f;
    }
    for (i, p) in v.iter().enumerate() {
        if p.level < MIN_LEVEL || p.level > MAX_LEVEL {
            f.issues.push(GeomIssue::LevelRange(i));
        }
    }
    let mut leg_len = vec![0.0; n - 1];
    let mut leg_th = vec![0.0; n - 1];
    let mut zero = false;
    for i in 0..n - 1 {
        let (dx, dy) = (v[i + 1].x - v[i].x, v[i + 1].y - v[i].y);
        leg_len[i] = dx.hypot(dy);
        leg_th[i] = dy.atan2(dx);
        if leg_len[i] < 1e-6 {
            f.issues.push(GeomIssue::ZeroLeg(i));
            zero = true;
        }
    }
    let mut delta = vec![0.0; n];
    if !zero {
        for i in 1..n - 1 {
            let d = wrap(leg_th[i] - leg_th[i - 1]);
            if d.abs() < 1e-9 {
                continue;
            }
            if d.abs() > PI - 1e-6 {
                f.issues.push(GeomIssue::Reversal(i));
                continue;
            }
            let t = (d.abs() / 2.0).tan();
            let avail_in = if i == 1 { leg_len[0] } else { leg_len[i - 1] / 2.0 };
            let avail_out = if i == n - 2 { leg_len[i] } else { leg_len[i] / 2.0 };
            let r = if v[i].radius > 0.0 { v[i].radius } else { (avail_in.min(avail_out) / t).min(R_CAP) };
            if r < MIN_RADIUS - 1e-6 {
                f.issues.push(GeomIssue::RadiusTooSmall(i));
            }
            delta[i] = d;
            f.radius[i] = r;
            f.tangent[i] = r * t;
        }
        for i in 0..n - 1 {
            if f.tangent[i] + f.tangent[i + 1] > leg_len[i] + FIT_TOL {
                for j in [i, i + 1] {
                    let blame = v[j].radius > 0.0 && j > 0 && j < n - 1;
                    if blame && !f.issues.contains(&GeomIssue::ArcDoesNotFit(j)) {
                        f.issues.push(GeomIssue::ArcDoesNotFit(j));
                    }
                }
            }
        }
    }
    let mut s = 0.0;
    for i in 0..n - 1 {
        let (c, sn) = (leg_th[i].cos(), leg_th[i].sin());
        let (t0, t1) = (f.tangent[i], f.tangent[i + 1]);
        let straight = (leg_len[i] - t0 - t1).max(0.0);
        if straight > 1e-9 {
            f.pieces.push(Piece { s0: s, len: straight, x0: v[i].x + t0 * c, y0: v[i].y + t0 * sn, th0: leg_th[i], k: 0.0 });
            f.piece_at_vertex.push(i as u32);
            s += straight;
        }
        if i + 1 < n - 1 {
            if f.radius[i + 1] > 0.0 {
                let r = f.radius[i + 1];
                let alen = r * delta[i + 1].abs();
                f.pieces.push(Piece {
                    s0: s,
                    len: alen,
                    x0: v[i + 1].x - t1 * c,
                    y0: v[i + 1].y - t1 * sn,
                    th0: leg_th[i],
                    k: delta[i + 1].signum() / r,
                });
                f.piece_at_vertex.push(i as u32 + 1);
                f.vert_s[i + 1] = s + alen / 2.0;
                s += alen;
            } else {
                f.vert_s[i + 1] = s;
            }
        }
    }
    f.vert_s[n - 1] = s;
    f.len = s;
    f.vert = vertical(v, &f.vert_s, &mut f.issues);
    f
}

/// Height breakpoints: level at every vertex, a ramp of 200 m per level centred between two
/// vertices of different levels (so the result does not depend on the drawing direction).
fn vertical(v: &[Pi], vs: &[f64], issues: &mut Vec<GeomIssue>) -> Vec<[f64; 2]> {
    let n = v.len();
    let z = |i: usize| v[i].level as f64 * LEVEL_H;
    let mut pts = vec![[0.0, z(0)]];
    for k in 0..n - 1 {
        if v[k].level == v[k + 1].level {
            continue;
        }
        let r = RAMP_PER_LEVEL * (v[k + 1].level as f64 - v[k].level as f64).abs();
        let c = (vs[k] + vs[k + 1]) / 2.0;
        let (mut a, mut b) = (c - r / 2.0, c + r / 2.0);
        if a < vs[k] - 1e-6 || b > vs[k + 1] + 1e-6 {
            issues.push(GeomIssue::RampTooShort(k));
            a = a.max(vs[k]);
            b = b.min(vs[k + 1]).max(a);
        }
        pts.push([a, z(k)]);
        pts.push([b, z(k + 1)]);
    }
    pts.push([vs[n - 1], z(n - 1)]);
    pts
}

/// Height at chainage `s` from breakpoints.
pub fn height_at(vert: &[[f64; 2]], s: f64) -> f64 {
    let i = vert.partition_point(|p| p[0] <= s);
    if i == 0 {
        return vert[0][1];
    }
    if i >= vert.len() {
        return vert[vert.len() - 1][1];
    }
    let (a, b) = (vert[i - 1], vert[i]);
    if b[0] - a[0] < 1e-9 {
        return b[1];
    }
    a[1] + (b[1] - a[1]) * (s - a[0]) / (b[0] - a[0])
}

/// True if any part of `[s0, s1]` is on a ramp.
pub fn on_ramp(vert: &[[f64; 2]], s0: f64, s1: f64) -> bool {
    vert.windows(2).any(|w| w[0][1] != w[1][1] && w[1][0] > s0 + 1e-6 && w[0][0] < s1 - 1e-6)
}

/// Index of the piece holding chainage `s`.
pub fn piece_at(pieces: &[Piece], s: f64) -> usize {
    pieces.partition_point(|p| p.s0 <= s).saturating_sub(1).min(pieces.len().saturating_sub(1))
}

/// Position and heading at chainage `s` (clamped to the alignment).
pub fn pos_at(pieces: &[Piece], s: f64) -> (f64, f64, f64) {
    if pieces.is_empty() {
        // A degenerate alignment (two vertices in one place): refused by validation, but the
        // checks still ask where it is.
        return (0.0, 0.0, 0.0);
    }
    let p = &pieces[piece_at(pieces, s)];
    p.at((s - p.s0).clamp(0.0, p.len))
}

/// Points along the alignment for drawing: piece ends, and arcs split so no step is longer than
/// `max_step` or strays more than `chord_tol` from the arc. `lateral` offsets the line to the left
/// (positive) or right, e.g. +-2 m for the two tracks of a double-track route; arcs stay arcs.
pub fn sample(pieces: &[Piece], max_step: f64, chord_tol: f64, lateral: f64, out: &mut Vec<[f64; 2]>) {
    let push = |out: &mut Vec<[f64; 2]>, (x, y, th): (f64, f64, f64)| out.push([x - lateral * th.sin(), y + lateral * th.cos()]);
    for p in pieces {
        let step = if p.k == 0.0 { max_step } else { max_step.min((8.0 * p.radius() * chord_tol).sqrt()) };
        let n = if p.k == 0.0 { 1 } else { (p.len / step).ceil().max(1.0) as usize };
        for j in 0..n {
            push(out, p.at(p.len * j as f64 / n as f64));
        }
    }
    if let Some(p) = pieces.last() {
        push(out, p.end());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pi(x: f64, y: f64) -> Pi {
        Pi::new(x, y, 0.0, 0)
    }

    #[test]
    fn auto_radius_uses_half_legs_and_caps() {
        // Right angle, legs of 1000 m, both ends are nodes: the whole leg is available, so the
        // radius would be 1000 / tan(45) = 1000 m.
        let f = fit(&[pi(0.0, 0.0), pi(1000.0, 0.0), pi(1000.0, 1000.0)]);
        assert!(f.issues.is_empty(), "{:?}", f.issues);
        assert!((f.radius[1] - 1000.0).abs() < 1e-9);
        // Arc length R * pi/2, total = 2 straights of 0 + arc.
        assert!((f.len - 1000.0 * PI / 2.0).abs() < 1e-6);
        let end = f.pieces.last().unwrap().end();
        assert!((end.0 - 1000.0).abs() < 1e-6 && (end.1 - 1000.0).abs() < 1e-6);
        // Two interior PIs: each may use half of the middle leg.
        let f = fit(&[pi(0.0, 0.0), pi(2000.0, 0.0), pi(2000.0, 600.0), pi(5000.0, 600.0)]);
        assert!((f.tangent[1] - 300.0).abs() < 1e-9 && (f.tangent[2] - 300.0).abs() < 1e-9);
        // A gentle bend with long legs is capped at the 160 km/h radius.
        let f = fit(&[pi(0.0, 0.0), pi(5000.0, 0.0), pi(10000.0, 500.0)]);
        assert_eq!(f.radius[1], R_CAP);
        assert!(curve_speed(R_CAP) >= V_TOP);
        assert!(curve_speed(R_CAP - 1.0) < V_TOP);
    }

    #[test]
    fn manual_radius_too_small_or_too_big() {
        let mut v = [pi(0.0, 0.0), pi(1000.0, 0.0), pi(1000.0, 1000.0)];
        v[1].radius = 80.0;
        assert!(fit(&v).issues.contains(&GeomIssue::RadiusTooSmall(1)));
        v[1].radius = 1500.0; // tangent 1500 > 1000 m leg
        assert!(fit(&v).issues.contains(&GeomIssue::ArcDoesNotFit(1)));
        v[1].radius = 400.0;
        let f = fit(&v);
        assert!(f.issues.is_empty());
        // 600 m straight, quarter circle, 600 m straight.
        assert!((f.len - (1200.0 + 400.0 * PI / 2.0)).abs() < 1e-6);
        assert!((f.pieces[1].v_limit() - (1.1f64 * 400.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn pieces_are_continuous_and_tangent() {
        let v = [pi(0.0, 0.0), pi(800.0, 300.0), pi(1500.0, -200.0), pi(2600.0, 100.0), pi(3000.0, 900.0)];
        let f = fit(&v);
        assert!(f.issues.is_empty(), "{:?}", f.issues);
        for w in f.pieces.windows(2) {
            let (x, y, th) = w[0].end();
            assert!((x - w[1].x0).abs() < 1e-6 && (y - w[1].y0).abs() < 1e-6);
            assert!(wrap(th - w[1].th0).abs() < 1e-9);
            assert!((w[0].s0 + w[0].len - w[1].s0).abs() < 1e-9);
        }
        let (x, y, _) = f.pieces.last().unwrap().end();
        assert!((x - 3000.0).abs() < 1e-6 && (y - 900.0).abs() < 1e-6);
        // Heading at the start is the first leg's.
        assert!((pos_at(&f.pieces, 0.0).2 - (300.0f64).atan2(800.0)).abs() < 1e-12);
    }

    #[test]
    fn speeds_match_the_spec_table() {
        let kmh = |r: f64| curve_speed(r) * 3.6;
        assert_eq!(kmh(100.0).round(), 38.0);
        assert_eq!(kmh(300.0).round(), 65.0);
        assert_eq!(kmh(1000.0).round(), 119.0);
        assert_eq!(kmh(1800.0).round(), 160.0);
        assert_eq!(kmh(5000.0), 160.0);
    }

    #[test]
    fn ramps_are_centred_and_must_fit() {
        // 0 -> -2 over a 1000 m leg: a 400 m ramp from 300 to 700.
        let v = [Pi::new(0.0, 0.0, 0.0, 0), Pi::new(1000.0, 0.0, 0.0, -2)];
        let f = fit(&v);
        assert!(f.issues.is_empty());
        assert_eq!(height_at(&f.vert, 299.0), 0.0);
        assert!((height_at(&f.vert, 500.0) + 8.0).abs() < 1e-9);
        assert_eq!(height_at(&f.vert, 701.0), -16.0);
        assert!(on_ramp(&f.vert, 250.0, 320.0) && !on_ramp(&f.vert, 0.0, 300.0));
        // Drawn the other way: the same ramp.
        let r = fit(&[v[1], v[0]]);
        assert!((height_at(&r.vert, 1000.0 - 500.0) + 8.0).abs() < 1e-9);
        // 0 -> +3 needs 600 m.
        let f = fit(&[Pi::new(0.0, 0.0, 0.0, 0), Pi::new(500.0, 0.0, 0.0, 3)]);
        assert!(f.issues.contains(&GeomIssue::RampTooShort(0)));
    }

    #[test]
    fn bbox_covers_arc() {
        let f = fit(&[pi(0.0, 0.0), pi(1000.0, 0.0), pi(1000.0, 1000.0)]);
        let arc = f.pieces.iter().find(|p| p.k != 0.0).unwrap();
        let b = arc.bbox();
        for j in 0..=100 {
            let (x, y, _) = arc.at(arc.len * j as f64 / 100.0);
            assert!(x >= b[0] - 1e-9 && x <= b[2] + 1e-9 && y >= b[1] - 1e-9 && y <= b[3] + 1e-9);
        }
        let mut pts = vec![];
        sample(&f.pieces, 50.0, 0.05, 2.0, &mut pts);
        assert!(pts.len() > 10);
    }
}
