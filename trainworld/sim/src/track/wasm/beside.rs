//! Drawing beside track (T-1685): a route laid parallel to existing track at the express pair's
//! spacing, its corners concentric with the track's (`express_verts`).

use super::*;

/// Pieces walked from the start's piece, all told, looking for the end's.
const BESIDE_WALK: usize = 400;

impl TrackApi {
    /// `route_op`'s ends and PIs for a route beside track: both ends moved onto the line
    /// `QUAD_SPACING` from the track (a node end stays put), on the end's side, and the corners
    /// between concentric with the track's. None if an end is on track, or the two pieces are not
    /// on one run.
    pub(super) fn beside_ends(&self, ends: &[f64], ea: u32, eb: u32) -> Option<(Vec<f64>, Vec<f64>)> {
        let net = &self.w.net;
        let at = |c: &[f64]| -> Option<(f64, f64, bool)> {
            match c[0] as i32 {
                0 => Some((c[2], c[3], false)),
                1 if net.node_ok(c[1] as u32) => Some((net.node_x[c[1] as usize], net.node_y[c[1] as usize], true)),
                _ => None,
            }
        };
        let (a, b) = (at(&ends[0..5])?, at(&ends[5..10])?);
        let scaled = net.frame().scaled();
        let mut v = self.beside_run(ea, eb, scaled)?;
        let mut pa = project(&v, a.0, a.1);
        let mut pb = project(&v, b.0, b.1);
        if pb.0 < pa.0 {
            v.reverse();
            (pa, pb) = (project(&v, a.0, a.1), project(&v, b.0, b.1));
        }
        // the side the pointer is on; a start already built on the other side is no run beside
        let side = if pb.1 >= 0.0 { 1.0 } else { -1.0 };
        if a.2 && pa.1 * side < 0.0 {
            return None;
        }
        let point = |s: f64| -> Pi {
            let i = (s.floor() as usize).min(v.len() - 2);
            let t = s - i as f64;
            Pi::new(v[i].x + (v[i + 1].x - v[i].x) * t, v[i].y + (v[i + 1].y - v[i].y) * t, 0.0, v[i].level)
        };
        let mut sub = vec![point(pa.0)];
        let first = pa.0.floor() as usize + 1;
        let last = (pb.0.ceil() as usize).saturating_sub(1);
        for p in v.iter().take(last + 1).skip(first) {
            if (p.x - sub[sub.len() - 1].x).hypot(p.y - sub[sub.len() - 1].y) > 0.01 {
                sub.push(*p);
            }
        }
        let end = point(pb.0);
        while sub.len() > 1 && (end.x - sub[sub.len() - 1].x).hypot(end.y - sub[sub.len() - 1].y) < 0.01 {
            sub.pop();
        }
        sub.push(end);
        if (end.x - sub[0].x).hypot(end.y - sub[0].y) < 1.0 {
            return None;
        }
        let own = |k: usize, p: (f64, f64, bool)| if p.2 { (p.0, p.1) } else { express_end(&sub, k, side, scaled) };
        let (oa, ob) = (own(0, a), own(1, b));
        let (ex, _, _) = express_verts(&sub, 1, sub.len() - 2, scaled, side, Some(oa), Some(ob))?;
        let mut e = ends.to_vec();
        for (k, o) in [(0, oa), (5, ob)] {
            if e[k] as i32 == 0 {
                (e[k + 2], e[k + 3]) = o;
            }
        }
        let pis = ex[1..ex.len() - 1].iter().flat_map(|p| [p.x, p.y, p.radius, p.level as f64]).collect();
        Some((e, pis))
    }

    /// The run of track through piece `ea`, through plain joints and stations, as one vertex list
    /// with every curve's radius as fitted (so a cut leg keeps its curves' size). None if `eb` is
    /// not on it.
    fn beside_run(&self, ea: u32, eb: u32, scaled: bool) -> Option<Vec<Pi>> {
        let net = &self.w.net;
        if !net.edge_ok(ea) || !net.edge_ok(eb) {
            return None;
        }
        let mut run = vec![(ea, false)];
        let mut nodes = vec![net.edge_a[ea as usize], net.edge_b[ea as usize]];
        for back in [false, true] {
            while run.len() < BESIDE_WALK {
                let (node, from) = if back { (nodes[0], run[0].0) } else { (*nodes.last().unwrap(), run.last().unwrap().0) };
                let p = &net.node_ports[node as usize];
                if p.n != 2 || p.side[0] == p.side[1] {
                    break;
                }
                let Some(&c) = p.ends().iter().find(|&&c| c >> 1 != from) else { break };
                let e = c >> 1;
                if run.iter().any(|r| r.0 == e) {
                    break;
                }
                let far = net.end_node(e, 1 - (c & 1));
                if back {
                    run.insert(0, (e, c & 1 == 0));
                    nodes.insert(0, far);
                } else {
                    run.push((e, c & 1 == 1));
                    nodes.push(far);
                }
            }
        }
        if !run.iter().any(|r| r.0 == eb) {
            return None;
        }
        let mut v: Vec<Pi> = vec![];
        for &(e, rev) in &run {
            let mut d = self.editable(e)?;
            if rev {
                d.pis.reverse();
                std::mem::swap(&mut d.a, &mut d.b);
            }
            let mut s = vertices_of(net, &d, &[]);
            let f = geom::fit_in(&s, scaled);
            for i in 1..s.len() - 1 {
                if f.radius[i] > 0.0 {
                    s[i].radius = quant(f.radius[i]);
                }
            }
            if !v.is_empty() {
                s.remove(0);
            }
            v.extend(s);
        }
        Some(v)
    }
}

/// Where (x, y) falls along polyline `v`: leg index plus the share of that leg, and which side
/// (positive left of the polyline's direction).
fn project(v: &[Pi], x: f64, y: f64) -> (f64, f64) {
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for i in 0..v.len() - 1 {
        let (dx, dy) = (v[i + 1].x - v[i].x, v[i + 1].y - v[i].y);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((x - v[i].x) * dx + (y - v[i].y) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        let (fx, fy) = (v[i].x + dx * t, v[i].y + dy * t);
        let d = (x - fx).hypot(y - fy);
        if d < best.0 {
            best = (d, i as f64 + t, dx * (y - fy) - dy * (x - fx));
        }
    }
    (best.1, best.2)
}
