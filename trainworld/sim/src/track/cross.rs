//! Where track crosses track: exact straight/arc intersections and a 1 km grid of edges for
//! finding candidates (notes/T-008.md 4.2-4.3).

use super::geom::Piece;
use std::collections::HashMap;

/// Grid cell size, m.
pub const CELL: f64 = 1000.0;
/// Long pieces are put in the grid in chunks of at most this, so a 10 km straight does not cover
/// its whole bounding box.
const CHUNK: f64 = 500.0;
const EPS: f64 = 1e-6;

/// One intersection of two pieces: offsets into each, and whether it is a stretch of overlap
/// (collinear straights or arcs of one circle) rather than a point.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub dp: f64,
    pub dq: f64,
    pub overlap: bool,
}

fn cross2(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ax * by - ay * bx
}

/// Offset along arc `p` of a point on its circle, in (-pi R, pi R].
fn arc_param(p: &Piece, x: f64, y: f64) -> f64 {
    let (cx, cy) = p.center();
    let phi = (y - cy).atan2(x - cx);
    let phi0 = (p.y0 - cy).atan2(p.x0 - cx);
    super::geom::wrap(phi - phi0) / p.k
}

fn in_range(d: f64, len: f64) -> bool {
    d >= -EPS && d <= len + EPS
}

/// All intersections of two pieces.
pub fn intersect(p: &Piece, q: &Piece, out: &mut Vec<Hit>) {
    match (p.k == 0.0, q.k == 0.0) {
        (true, true) => line_line(p, q, out),
        (true, false) => line_arc(p, q, out, false),
        (false, true) => line_arc(q, p, out, true),
        (false, false) => arc_arc(p, q, out),
    }
}

fn line_line(p: &Piece, q: &Piece, out: &mut Vec<Hit>) {
    let (d1x, d1y) = (p.th0.cos(), p.th0.sin());
    let (d2x, d2y) = (q.th0.cos(), q.th0.sin());
    let (wx, wy) = (q.x0 - p.x0, q.y0 - p.y0);
    let den = cross2(d1x, d1y, d2x, d2y);
    if den.abs() < 1e-12 {
        if cross2(d1x, d1y, wx, wy).abs() > 0.01 {
            return;
        }
        // Collinear: overlap of the two intervals measured along p.
        let t0 = wx * d1x + wy * d1y;
        let t1 = t0 + q.len * (d1x * d2x + d1y * d2y);
        let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(p.len));
        if hi - lo > 0.01 {
            for t in [lo, hi] {
                let (x, y) = (p.x0 + t * d1x, p.y0 + t * d1y);
                let u = (x - q.x0) * d2x + (y - q.y0) * d2y;
                out.push(Hit { dp: t, dq: u, overlap: true });
            }
        }
        return;
    }
    let t = cross2(wx, wy, d2x, d2y) / den;
    let u = cross2(wx, wy, d1x, d1y) / den;
    if in_range(t, p.len) && in_range(u, q.len) {
        out.push(Hit { dp: t.clamp(0.0, p.len), dq: u.clamp(0.0, q.len), overlap: false });
    }
}

/// Straight `l` against arc `a`; `swap` reports the hit as (arc, line).
fn line_arc(l: &Piece, a: &Piece, out: &mut Vec<Hit>, swap: bool) {
    let (dx, dy) = (l.th0.cos(), l.th0.sin());
    let (cx, cy) = a.center();
    let r = a.radius();
    let (fx, fy) = (l.x0 - cx, l.y0 - cy);
    let b = fx * dx + fy * dy;
    let c = fx * fx + fy * fy - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return;
    }
    let sq = disc.sqrt();
    let roots = if sq < 1e-12 { [-b, f64::NAN] } else { [-b - sq, -b + sq] };
    for t in roots {
        if !t.is_finite() || !in_range(t, l.len) {
            continue;
        }
        let (x, y) = (l.x0 + t * dx, l.y0 + t * dy);
        let u = arc_param(a, x, y);
        if in_range(u, a.len) {
            let (t, u) = (t.clamp(0.0, l.len), u.clamp(0.0, a.len));
            out.push(if swap { Hit { dp: u, dq: t, overlap: false } } else { Hit { dp: t, dq: u, overlap: false } });
        }
    }
}

fn arc_arc(p: &Piece, q: &Piece, out: &mut Vec<Hit>) {
    let (c1x, c1y) = p.center();
    let (c2x, c2y) = q.center();
    let (r1, r2) = (p.radius(), q.radius());
    let (ex, ey) = (c2x - c1x, c2y - c1y);
    let d = ex.hypot(ey);
    if d < 0.01 {
        if (r1 - r2).abs() > 0.01 {
            return;
        }
        // One circle: overlap where q's ends fall on p or p's ends fall on q.
        let mut ds: Vec<(f64, f64)> = vec![];
        for (x, y, _) in [q.at(0.0), q.end()] {
            let u = arc_param(p, x, y);
            if in_range(u, p.len) {
                ds.push((u, arc_param(q, x, y)));
            }
        }
        for (x, y, _) in [p.at(0.0), p.end()] {
            let u = arc_param(q, x, y);
            if in_range(u, q.len) {
                ds.push((arc_param(p, x, y), u));
            }
        }
        if ds.len() >= 2 {
            ds.sort_by(|a, b| a.0.total_cmp(&b.0));
            if ds[ds.len() - 1].0 - ds[0].0 > 0.01 {
                for (a, b) in [ds[0], ds[ds.len() - 1]] {
                    out.push(Hit { dp: a.clamp(0.0, p.len), dq: b.clamp(0.0, q.len), overlap: true });
                }
            }
        }
        return;
    }
    if d > r1 + r2 || d < (r1 - r2).abs() {
        return;
    }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h = (r1 * r1 - a * a).max(0.0).sqrt();
    let (mx, my) = (c1x + a * ex / d, c1y + a * ey / d);
    let pts = if h < 1e-9 { vec![(mx, my)] } else { vec![(mx - h * ey / d, my + h * ex / d), (mx + h * ey / d, my - h * ex / d)] };
    for (x, y) in pts {
        let (u, w) = (arc_param(p, x, y), arc_param(q, x, y));
        if in_range(u, p.len) && in_range(w, q.len) {
            out.push(Hit { dp: u.clamp(0.0, p.len), dq: w.clamp(0.0, q.len), overlap: false });
        }
    }
}

fn boxes_touch(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] + 0.01 && b[0] <= a[2] + 0.01 && a[1] <= b[3] + 0.01 && b[1] <= a[3] + 0.01
}

/// Intersections between two alignments as `(chainage on p, chainage on q, overlap)`, deduplicated
/// (a hit at a piece boundary is found from both pieces). With `same`, `p` and `q` are one edge and
/// hits of a piece with itself or near the diagonal are skipped.
pub fn intersect_alignments(p: &[Piece], q: &[Piece], same: bool, out: &mut Vec<(f64, f64, bool)>) {
    let mut hits = Vec::new();
    let qb: Vec<[f64; 4]> = q.iter().map(|x| x.bbox()).collect();
    for (i, a) in p.iter().enumerate() {
        let ab = a.bbox();
        for (j, b) in q.iter().enumerate() {
            if same && j <= i + 1 {
                continue;
            }
            if !boxes_touch(&ab, &qb[j]) {
                continue;
            }
            hits.clear();
            intersect(a, b, &mut hits);
            for h in &hits {
                let (s, t) = (a.s0 + h.dp, b.s0 + h.dq);
                if same && (s - t).abs() < 1.0 {
                    continue;
                }
                if !out.iter().any(|o: &(f64, f64, bool)| (o.0 - s).abs() < 0.01 && (o.1 - t).abs() < 0.01) {
                    out.push((s, t, h.overlap));
                }
            }
        }
    }
}

/// Cells of the grid a piece passes through (chunked, conservative by the arc's sagitta).
pub fn piece_cells(p: &Piece, out: &mut Vec<(i32, i32)>) {
    let n = (p.len / CHUNK).ceil().max(1.0) as usize;
    for j in 0..n {
        let (d0, d1) = (p.len * j as f64 / n as f64, p.len * (j + 1) as f64 / n as f64);
        let b = p.bbox_part(d0, d1);
        let (x0, y0) = ((b[0] / CELL).floor() as i32, (b[1] / CELL).floor() as i32);
        let (x1, y1) = ((b[2] / CELL).floor() as i32, (b[3] / CELL).floor() as i32);
        for cx in x0..=x1 {
            for cy in y0..=y1 {
                out.push((cx, cy));
            }
        }
    }
}

/// Edges per 1 km cell. Lookups only; nothing iterates the map, so its order never leaks out.
#[derive(Default, Clone)]
pub struct Grid {
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl Grid {
    fn cells_of(pieces: &[Piece]) -> Vec<(i32, i32)> {
        let mut c = Vec::new();
        for p in pieces {
            piece_cells(p, &mut c);
        }
        c.sort_unstable();
        c.dedup();
        c
    }
    pub fn insert(&mut self, edge: u32, pieces: &[Piece]) {
        for c in Self::cells_of(pieces) {
            self.cells.entry(c).or_default().push(edge);
        }
    }
    pub fn remove(&mut self, edge: u32, pieces: &[Piece]) {
        for c in Self::cells_of(pieces) {
            if let Some(v) = self.cells.get_mut(&c) {
                v.retain(|&e| e != edge);
                if v.is_empty() {
                    self.cells.remove(&c);
                }
            }
        }
    }
    /// Edges sharing a cell with these pieces, sorted, without duplicates.
    pub fn near(&self, pieces: &[Piece]) -> Vec<u32> {
        let mut out = Vec::new();
        for c in Self::cells_of(pieces) {
            if let Some(v) = self.cells.get(&c) {
                out.extend_from_slice(v);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::geom::{fit, Pi};
    use super::*;

    fn line(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Piece> {
        fit(&[Pi::new(x0, y0, 0.0, 0), Pi::new(x1, y1, 0.0, 0)]).pieces
    }

    #[test]
    fn straight_cross() {
        let mut out = vec![];
        intersect_alignments(&line(0.0, 0.0, 1000.0, 0.0), &line(500.0, -500.0, 500.0, 500.0), false, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0].0 - 500.0).abs() < 1e-9 && (out[0].1 - 500.0).abs() < 1e-9);
    }

    #[test]
    fn arc_cross_and_overlap() {
        // Quarter circle of radius 1000 centred on (0, 1000) from (0,0) to (1000,1000).
        let arc = fit(&[Pi::new(0.0, 0.0, 0.0, 0), Pi::new(1000.0, 0.0, 0.0, 0), Pi::new(1000.0, 1000.0, 0.0, 0)]).pieces;
        let mut out = vec![];
        // A vertical line at x = 600 meets the circle at y = 1000 - 800 = 200.
        intersect_alignments(&arc, &line(600.0, -100.0, 600.0, 900.0), false, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!((out[0].1 - 300.0).abs() < 1e-6);
        // The same arc twice: overlap.
        out.clear();
        intersect_alignments(&arc, &arc, false, &mut out);
        assert!(out.iter().all(|h| h.2) && !out.is_empty());
        // Collinear straights overlapping.
        out.clear();
        intersect_alignments(&line(0.0, 0.0, 1000.0, 0.0), &line(500.0, 0.0, 1500.0, 0.0), false, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out[0].2);
        // Two arcs crossing.
        let arc2 = fit(&[Pi::new(1000.0, 0.0, 0.0, 0), Pi::new(0.0, 0.0, 0.0, 0), Pi::new(0.0, 1000.0, 0.0, 0)]).pieces;
        out.clear();
        intersect_alignments(&arc, &arc2, false, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        let (x, y, _) = super::super::geom::pos_at(&arc, out[0].0);
        assert!((x - 500.0).abs() < 1e-6, "{x} {y}");
    }

    #[test]
    fn grid_finds_neighbours() {
        let mut g = Grid::default();
        let a = line(0.0, 0.0, 10000.0, 0.0);
        let b = line(5000.0, -300.0, 5000.0, 300.0);
        let c = line(0.0, 8000.0, 10000.0, 8000.0);
        g.insert(1, &a);
        g.insert(2, &b);
        g.insert(3, &c);
        assert_eq!(g.near(&b), vec![1, 2]);
        g.remove(1, &a);
        assert_eq!(g.near(&b), vec![2]);
    }
}
