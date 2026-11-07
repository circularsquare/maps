//! Station buildings (T-2208, T-2217; notes/T-2208.md 3): an outline the player draws round
//! stations. It joins every station whose platform touches it into one complex, at any level
//! (`game/buildings.ts` works the members out for the app's complexes from the drawn platform
//! lines; `members` here is the same rule over the track model's own track, for the refusals and
//! the panel). Paid at once and outside the undo history, as passages are (`wasm/buildings.rs`);
//! saved in the network's `BLD1` block. Members are never saved.

use super::geom;
use super::net::{Issue, Network};
use std::f64::consts::PI;

pub const MIN_CORNERS: usize = 3;
pub const MAX_CORNERS: usize = 24;
/// The least floor a building has, m².
pub const MIN_M2: f64 = 1000.0;
/// The farthest two corners may be apart, ground m: a passage's reach.
pub const REACH_M: f64 = super::upgrades::PASSAGE_MAX_M;
/// A platform's centre line this near the outline touches it, ground m: half a drawn box's width
/// (game/complexes.ts `BOX_TOUCH_M`), so a box that touches the outline belongs.
pub const TOUCH_M: f64 = 8.0;
/// Corners nearer each other than this are one corner placed twice, ground m.
pub const MIN_EDGE_M: f64 = 1.0;
/// Two buildings sharing less floor than this only touch, m².
pub const OVERLAP_M2: f64 = 1.0;
/// Points a platform's centre line is sampled at (the app's `PLATFORM_POINTS`).
const PLATFORM_POINTS: usize = 13;
/// Collinear, in m² of the cross product.
const EPS: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct Building {
    pub name: String,
    /// Plane x, y on the 1 mm grid, in the order drawn.
    pub corners: Vec<[f64; 2]>,
}

type P = [f64; 2];

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn cross(a: P, b: P) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn orient(a: P, b: P, c: P) -> f64 {
    cross(sub(b, a), sub(c, a))
}
fn dist(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Plane units per ground metre over these corners (at their mean y).
fn scale(net: &Network, c: &[P]) -> f64 {
    let y = c.iter().map(|p| p[1]).sum::<f64>() / c.len().max(1) as f64;
    net.k_at(y)
}

/// The corners as ground metres from `o`.
fn ground(c: &[P], o: P, k: f64) -> Vec<P> {
    c.iter().map(|p| [(p[0] - o[0]) / k, (p[1] - o[1]) / k]).collect()
}

/// The shoelace area, positive when the corners run anticlockwise.
pub fn signed_area(p: &[P]) -> f64 {
    let n = p.len();
    (0..n).map(|i| cross(p[i], p[(i + 1) % n])).sum::<f64>() / 2.0
}

/// A building's floor, m².
pub fn area_m2(net: &Network, c: &[P]) -> f64 {
    if c.len() < 3 {
        return 0.0;
    }
    signed_area(&ground(c, c[0], scale(net, c))).abs()
}

/// p lies on segment a-b, given it is collinear with it.
fn within(a: P, b: P, p: P) -> bool {
    p[0] >= a[0].min(b[0]) - 1e-9 && p[0] <= a[0].max(b[0]) + 1e-9 && p[1] >= a[1].min(b[1]) - 1e-9 && p[1] <= a[1].max(b[1]) + 1e-9
}

/// Segments a-b and c-d meet, ends included.
fn segs_meet(a: P, b: P, c: P, d: P) -> bool {
    let (d1, d2, d3, d4) = (orient(c, d, a), orient(c, d, b), orient(a, b, c), orient(a, b, d));
    if ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS)) && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS)) {
        return true;
    }
    (d1.abs() <= EPS && within(c, d, a)) || (d2.abs() <= EPS && within(c, d, b)) || (d3.abs() <= EPS && within(a, b, c)) || (d4.abs() <= EPS && within(a, b, d))
}

/// Two edges of the outline cross or touch (or one folds back along the next).
pub fn edges_cross(p: &[P]) -> bool {
    let n = p.len();
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        for j in i + 1..n {
            let (c, d) = (p[j], p[(j + 1) % n]);
            if j == i + 1 || (i == 0 && j == n - 1) {
                // neighbours share a corner: they cross only by folding back along each other
                let (u, s, v) = if j == i + 1 { (a, b, d) } else { (b, a, c) };
                let (su, sv) = (sub(u, s), sub(v, s));
                if cross(su, sv).abs() <= EPS * (1.0 + dist(u, s) * dist(v, s)) && su[0] * sv[0] + su[1] * sv[1] > 0.0 {
                    return true;
                }
                continue;
            }
            if segs_meet(a, b, c, d) {
                return true;
            }
        }
    }
    false
}

/// x inside triangle a, b, c (anticlockwise), its edges included.
fn in_tri(a: P, b: P, c: P, x: P) -> bool {
    orient(a, b, x) >= -EPS && orient(b, c, x) >= -EPS && orient(c, a, x) >= -EPS
}

/// A simple polygon cut into anticlockwise triangles (ear clipping).
fn triangles(p: &[P]) -> Vec<[P; 3]> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    if signed_area(p) < 0.0 {
        idx.reverse();
    }
    let mut out = vec![];
    while idx.len() > 3 {
        let m = idx.len();
        let corner = |k: usize| (p[idx[(k + m - 1) % m]], p[idx[k]], p[idx[(k + 1) % m]]);
        let ear = (0..m).find(|&k| {
            let (a, b, c) = corner(k);
            orient(a, b, c) > EPS && idx.iter().all(|&q| { let x = p[q]; x == a || x == b || x == c || !in_tri(a, b, c, x) })
        });
        // none (corners in a line): the flattest corner goes
        let k = ear.unwrap_or_else(|| (0..m).min_by(|&i, &j| { let (a, b, c) = corner(i); let (d, e, f) = corner(j); orient(a, b, c).abs().total_cmp(&orient(d, e, f).abs()) }).unwrap());
        let (a, b, c) = corner(k);
        if orient(a, b, c) > EPS {
            out.push([a, b, c]);
        }
        idx.remove(k);
    }
    if idx.len() == 3 && orient(p[idx[0]], p[idx[1]], p[idx[2]]) > EPS {
        out.push([p[idx[0]], p[idx[1]], p[idx[2]]]);
    }
    out
}

/// A convex polygon clipped to an anticlockwise triangle (Sutherland-Hodgman).
fn clip(mut poly: Vec<P>, t: &[P; 3]) -> Vec<P> {
    for i in 0..3 {
        let (a, b) = (t[i], t[(i + 1) % 3]);
        let inp = std::mem::take(&mut poly);
        for j in 0..inp.len() {
            let (p, q) = (inp[j], inp[(j + 1) % inp.len()]);
            let (sp, sq) = (orient(a, b, p), orient(a, b, q));
            if sp >= 0.0 {
                poly.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                poly.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
            }
        }
        if poly.is_empty() {
            break;
        }
    }
    poly
}

/// The floor two simple outlines share, in the square of their units.
pub fn shared_area(a: &[P], b: &[P]) -> f64 {
    let (ta, tb) = (triangles(a), triangles(b));
    ta.iter().flat_map(|x| tb.iter().map(move |y| signed_area(&clip(x.to_vec(), y)).abs())).sum()
}

/// Floor in `new` that `old` did not cover, m²: what reshaping a building pays for.
pub fn added_m2(net: &Network, old: &[P], new: &[P]) -> f64 {
    if new.len() < 3 {
        return 0.0;
    }
    let k = scale(net, new);
    let (o, n) = (ground(old, new[0], k), ground(new, new[0], k));
    let fresh = signed_area(&n).abs() - if old.len() >= 3 { shared_area(&o, &n) } else { 0.0 };
    fresh.max(0.0)
}

/// x inside the outline (even-odd).
pub fn inside(c: &[P], x: P) -> bool {
    let n = c.len();
    let mut odd = false;
    for i in 0..n {
        let (a, b) = (c[i], c[(i + n - 1) % n]);
        if (a[1] > x[1]) != (b[1] > x[1]) && x[0] < a[0] + (b[0] - a[0]) * (x[1] - a[1]) / (b[1] - a[1]) {
            odd = !odd;
        }
    }
    odd
}

/// Shortest distance between segments p0-p1 and q0-q1 (game/complexes.ts `segmentGap`).
fn seg_gap(p0: P, p1: P, q0: P, q1: P) -> f64 {
    let (u, v, w) = (sub(p1, p0), sub(q1, q0), sub(p0, q0));
    let (a, b, c, d, e) = (u[0] * u[0] + u[1] * u[1], u[0] * v[0] + u[1] * v[1], v[0] * v[0] + v[1] * v[1], u[0] * w[0] + u[1] * w[1], v[0] * w[0] + v[1] * w[1]);
    let den = a * c - b * b;
    let (mut s, mut t) = (0.0, 0.0);
    if a < 1e-12 && c < 1e-12 {
        return w[0].hypot(w[1]);
    }
    if a < 1e-12 {
        t = (e / c).clamp(0.0, 1.0);
    } else if c < 1e-12 {
        s = (-d / a).clamp(0.0, 1.0);
    } else {
        s = if den > 1e-9 * a * c { ((b * e - c * d) / den).clamp(0.0, 1.0) } else { 0.0 };
        t = (b * s + e) / c;
        if t < 0.0 {
            t = 0.0;
            s = (-d / a).clamp(0.0, 1.0);
        } else if t > 1.0 {
            t = 1.0;
            s = ((b - d) / a).clamp(0.0, 1.0);
        }
    }
    (w[0] + s * u[0] - t * v[0]).hypot(w[1] + s * u[1] - t * v[1])
}

/// A station's platform centre line, plane points: along its track from the node, half its length
/// each way through the node, its whole length from the node at a dead end, straight on past the end
/// of the track where it stops short (as the app draws it, map/stationGeometry.ts `platformShape`);
/// with no track, its length along its heading centred on the node.
pub fn platform_line(net: &Network, n: u32) -> Vec<P> {
    let i = n as usize;
    let (x, y) = (net.node_x[i], net.node_y[i]);
    let k = net.k_at(y);
    let plat = net.node_platform[i] as f64;
    // the point `d` ground m from the node along edge end `c`
    let way = |c: u32, d: f64| -> P {
        let (e, which) = (c >> 1, c & 1);
        let pcs = net.edge_pieces(e);
        let len = net.edge_len[e as usize];
        if pcs.is_empty() {
            return [x, y];
        }
        let s = d.min(len);
        let (px, py, th) = geom::pos_at(pcs, if which == 0 { s } else { len - s });
        let th = if which == 0 { th } else { th + PI };
        let past = (d - len).max(0.0) * net.k_at(py);
        [px + past * th.cos(), py + past * th.sin()]
    };
    let ends = net.node_ports[i].ends();
    let step = |j: usize| j as f64 / (PLATFORM_POINTS - 1) as f64;
    match ends.len() {
        0 => {
            let (dx, dy) = (net.node_heading[i].cos() * plat / 2.0 * k, net.node_heading[i].sin() * plat / 2.0 * k);
            vec![[x - dx, y - dy], [x + dx, y + dy]]
        }
        1 => (0..PLATFORM_POINTS).map(|j| way(ends[0], plat * step(j))).collect(),
        _ => (0..PLATFORM_POINTS)
            .map(|j| {
                let t = plat * (step(j) - 0.5);
                if t < 0.0 { way(ends[1], -t) } else { way(ends[0], t) }
            })
            .collect(),
    }
}

/// A platform line touches the outline: a point of it inside, or within `tol` plane units of an edge.
pub fn touches(c: &[P], line: &[P], tol: f64) -> bool {
    if line.iter().any(|&p| inside(c, p)) {
        return true;
    }
    let n = c.len();
    let segs: Vec<(P, P)> = if line.len() < 2 { line.iter().map(|&p| (p, p)).collect() } else { line.windows(2).map(|w| (w[0], w[1])).collect() };
    segs.iter().any(|&(p, q)| (0..n).any(|i| seg_gap(p, q, c[i], c[(i + 1) % n]) <= tol))
}

/// The stations whose platform touches the outline, any level, constructed or not, by node id.
pub fn members(net: &Network, c: &[P]) -> Vec<u32> {
    if c.len() < 3 {
        return vec![];
    }
    let k = scale(net, c);
    let tol = TOUCH_M * k;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in c {
        (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
    }
    (0..net.node_count() as u32)
        .filter(|&n| {
            let i = n as usize;
            if !net.is_station(n) {
                return false;
            }
            // a platform reaches at most its length from its node (more only on a sharp curve)
            let r = net.node_platform[i] as f64 * 1.5 * k + tol;
            let (x, y) = (net.node_x[i], net.node_y[i]);
            x >= x0 - r && x <= x1 + r && y >= y0 - r && y <= y1 + r && touches(c, &platform_line(net, n), tol)
        })
        .collect()
}

/// Why these corners cannot be a building (building `id` reshaped, `u32::MAX` for a new one); None
/// if they can.
pub fn refusal(net: &Network, id: u32, c: &[P]) -> Option<Issue> {
    if c.len() < MIN_CORNERS || c.len() > MAX_CORNERS || c.iter().any(|p| !p[0].is_finite() || !p[1].is_finite() || !net.inside_world(p[0], p[1])) {
        return Some(Issue::BuildingCorners);
    }
    let k = scale(net, c);
    let g = ground(c, c[0], k);
    let n = g.len();
    if (0..n).any(|i| dist(g[i], g[(i + 1) % n]) < MIN_EDGE_M) {
        return Some(Issue::BuildingCorners);
    }
    if (0..n).any(|i| (i + 1..n).any(|j| dist(g[i], g[j]) > REACH_M + 1e-6)) {
        return Some(Issue::BuildingTooWide);
    }
    if edges_cross(&g) {
        return Some(Issue::BuildingEdgesCross);
    }
    if signed_area(&g).abs() < MIN_M2 {
        return Some(Issue::BuildingTooSmall);
    }
    let bbox = |c: &[P]| c.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]);
    let me = bbox(c);
    for (j, b) in net.buildings.iter().enumerate() {
        let Some(b) = b.as_ref().filter(|_| j as u32 != id) else { continue };
        let o = bbox(&b.corners);
        if o[0] > me[2] || o[2] < me[0] || o[1] > me[3] || o[3] < me[1] {
            continue;
        }
        if shared_area(&ground(&b.corners, c[0], k), &g) > OVERLAP_M2 {
            return Some(Issue::BuildingOverlap { building: j as u32 });
        }
    }
    if !members(net, c).iter().any(|&m| net.node_built[m as usize]) {
        return Some(Issue::BuildingNoStation);
    }
    None
}

/// A building's name when the player gave none: the first member whose name the player typed, else
/// the first constructed member's, the part before any slash.
pub fn default_name(net: &Network, members: &[u32]) -> String {
    members
        .iter()
        .find(|&&m| net.node_named[m as usize])
        .or_else(|| members.iter().find(|&&m| net.node_built[m as usize]))
        .or(members.first())
        .map(|&m| net.node_name[m as usize].split('/').next().unwrap_or("").trim().to_string())
        .unwrap_or_default()
}

/// What the buildings standing cost, US$M: their floor at today's price (in the built value).
pub fn value(net: &Network) -> f64 {
    net.buildings.iter().flatten().map(|b| super::upgrades::building_price(area_m2(net, &b.corners))).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(x: f64, y: f64, s: f64) -> Vec<P> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    #[test]
    fn areas_and_shared_floor() {
        assert!((signed_area(&sq(0.0, 0.0, 10.0)) - 100.0).abs() < 1e-9);
        let mut cw = sq(0.0, 0.0, 10.0);
        cw.reverse();
        assert!((signed_area(&cw) + 100.0).abs() < 1e-9);
        assert!((shared_area(&sq(0.0, 0.0, 10.0), &sq(5.0, 5.0, 10.0)) - 25.0).abs() < 1e-9);
        assert!((shared_area(&sq(0.0, 0.0, 10.0), &cw) - 100.0).abs() < 1e-9, "the same square, either way round");
        assert!(shared_area(&sq(0.0, 0.0, 10.0), &sq(10.0, 0.0, 10.0)).abs() < 1e-9, "a shared wall is no floor");
        // an L (concave) against a square in its notch: nothing shared; one over its arm: the arm's part
        let l: Vec<P> = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0], [10.0, 20.0], [0.0, 20.0]];
        assert!((signed_area(&l) - 300.0).abs() < 1e-9);
        assert!(shared_area(&l, &sq(12.0, 12.0, 6.0)).abs() < 1e-9);
        assert!((shared_area(&l, &sq(15.0, 5.0, 10.0)) - 25.0).abs() < 1e-9);
        assert!((shared_area(&l, &l) - 300.0).abs() < 1e-9);
    }

    #[test]
    fn crossing_edges() {
        assert!(!edges_cross(&sq(0.0, 0.0, 10.0)));
        assert!(edges_cross(&[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]), "a bow tie");
        assert!(edges_cross(&[[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]]), "folding back");
        assert!(edges_cross(&[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 0.0], [0.0, 10.0]]), "a corner on another edge");
        assert!(!edges_cross(&[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 1.0], [0.0, 10.0]]));
    }

    #[test]
    fn inside_and_gaps() {
        let c = sq(0.0, 0.0, 10.0);
        assert!(inside(&c, [5.0, 5.0]) && !inside(&c, [15.0, 5.0]));
        assert!(touches(&c, &[[15.0, 5.0], [30.0, 5.0]], 5.0) && !touches(&c, &[[15.1, 5.0], [30.0, 5.0]], 5.0));
        assert!(touches(&c, &[[-5.0, 5.0], [15.0, 5.0]], 0.0), "running right through");
    }
}
