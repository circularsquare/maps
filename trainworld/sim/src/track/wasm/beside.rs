//! Drawing beside track (T-1685): a route laid parallel to existing track at the express pair's
//! spacing, its corners concentric with the track's (`express_verts`).

use super::*;

/// Pieces walked from the start's piece, all told, looking for the end's.
const BESIDE_WALK: usize = 400;

impl TrackApi {
    /// `route_op`'s ends and PIs for a route beside the run of track through piece `eb` (the one
    /// the end is beside), and whether its start is on that run's line already. That line is
    /// `QUAD_SPACING` from the track on the end's side, its corners concentric with the track's.
    /// A start on the line (within half the spacing of it) runs along it to the end, an open-ground
    /// start moving onto it, a node start on the other side giving None. Any other start comes in
    /// (T-1721): its route turns onto the line where it has run along the track the start's
    /// distance from it (at most half the way to the end), then runs along it. The end moves onto
    /// the line. None if an end is on track.
    pub(super) fn beside_ends(&self, ends: &[f64], eb: u32) -> Option<(Vec<f64>, Vec<f64>, bool)> {
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
        let mut v = self.beside_run(eb, eb, scaled)?;
        let mut pa = project(&v, a.0, a.1);
        let mut pb = project(&v, b.0, b.1);
        if pb.0 < pa.0 {
            v.reverse();
            (pa, pb) = (project(&v, a.0, a.1), project(&v, b.0, b.1));
        }
        // the side the pointer is on
        let side = if pb.1 >= 0.0 { 1.0 } else { -1.0 };
        let gap = QUAD_SPACING * kk(scaled, a.1);
        let on_line = (pa.2 - gap).abs() <= gap / 2.0 || !a.2 && pa.2 <= gap * 3.0;
        if on_line && a.2 && pa.1 * side < 0.0 {
            return None;
        }
        let mut cum = vec![0.0];
        for w in v.windows(2) {
            cum.push(cum[cum.len() - 1] + (w[1].x - w[0].x).hypot(w[1].y - w[0].y));
        }
        let along = |s: f64| { let i = (s.floor() as usize).min(v.len() - 2); cum[i] + (cum[i + 1] - cum[i]) * (s - i as f64) };
        let from = if on_line {
            pa.0
        } else {
            let (la, le) = (along(pa.0), along(pb.0));
            let l = la + pa.2.min((le - la) / 2.0);
            let i = (0..v.len() - 1).find(|&i| cum[i + 1] >= l).unwrap_or(v.len() - 2);
            i as f64 + if cum[i + 1] > cum[i] { (l - cum[i]) / (cum[i + 1] - cum[i]) } else { 0.0 }
        };
        let point = |s: f64| -> Pi {
            let i = (s.floor() as usize).min(v.len() - 2);
            let t = s - i as f64;
            Pi::new(v[i].x + (v[i + 1].x - v[i].x) * t, v[i].y + (v[i + 1].y - v[i].y) * t, 0.0, v[i].level)
        };
        let mut sub = vec![point(from)];
        let first = from.floor() as usize + 1;
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
        let own = |k: usize, p: (f64, f64, bool)| if p.2 && on_line && k == 0 { (p.0, p.1) } else { express_end(&sub, k, side, scaled) };
        let (oa, ob) = (own(0, a), own(1, b));
        let (ex, _, _) = express_verts(&sub, 1, sub.len() - 2, scaled, side, Some(oa), Some(ob))?;
        let mut e = ends.to_vec();
        if e[0] as i32 == 0 && on_line {
            (e[2], e[3]) = oa;
        }
        if e[5] as i32 == 0 {
            (e[7], e[8]) = ob;
        }
        // coming in, the line's first point is a corner of the route (auto radius)
        let lo = if on_line { 1 } else { 0 };
        // at the end's level throughout: the track's own ramps are not copied (T-1722), the route
        // ramps over what it crosses by itself
        let pis = ex[lo..ex.len() - 1].iter().flat_map(|p| [p.x, p.y, p.radius, ends[9]]).collect();
        Some((e, pis, on_line))
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
/// (positive left of the polyline's direction), and how far from it.
fn project(v: &[Pi], x: f64, y: f64) -> (f64, f64, f64) {
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
    (best.1, best.2, best.0)
}
