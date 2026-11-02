//! Drawing beside track (T-1685): a route laid parallel to existing track at `BESIDE_SPACING`, its
//! corners concentric with the track's (`express_verts`).

use super::*;

/// Pieces walked from the start's piece, all told, looking for the end's.
const BESIDE_WALK: usize = 400;
/// Centre line to centre line between track and a route drawn beside it, m (T-1685; T-1895).
pub const BESIDE_SPACING: f64 = 12.0;

impl TrackApi {
    /// `route_op`'s ends and PIs for a route beside the run of track through piece `eb` (the one
    /// the end is beside), whether its start is on that run's line already, and the run's pieces.
    /// That line is `BESIDE_SPACING` from the track, its corners concentric with the track's, on
    /// the start's side (T-1895: never across the track), or the pointer's where the start is on
    /// the track or is open ground near the line. A start on the line (within half the spacing of
    /// it) runs along it to the end, an open-ground start moving onto it. Any other start comes in
    /// (T-1721): its route turns onto the line somewhere from where it has run along the track the
    /// start's distance from it (at most half the way to the end). The end moves onto the line.
    /// Distances and sides are taken from the track as fitted, curves and all, and the line is cut
    /// where the start and end are, inside a curve too. None if an end is on track, or every way
    /// onto the line would cross the track.
    pub(super) fn beside_ends(&self, ends: &[f64], eb: u32) -> Option<(Vec<f64>, Vec<f64>, bool, Vec<u32>)> {
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
        let (mut v, run) = self.beside_run(eb, eb, scaled)?;
        let mut f = geom::fit_in(&v, scaled);
        if f.pieces.is_empty() {
            return None;
        }
        // chainage along the run's track, and the offset from it (plane units, left positive)
        let place = |f: &geom::Fit, x: f64, y: f64| {
            let (s, d) = geom::project(&f.pieces, x, y);
            let (px, py, th) = geom::pos_at(&f.pieces, s);
            (s, if th.cos() * (y - py) - th.sin() * (x - px) >= 0.0 { d } else { -d })
        };
        let (mut pa, mut pb) = (place(&f, a.0, a.1), place(&f, b.0, b.1));
        if pb.0 < pa.0 {
            v.reverse();
            f = geom::fit_in(&v, scaled);
            (pa, pb) = (place(&f, a.0, a.1), place(&f, b.0, b.1));
        }
        let k = kk(scaled, a.1);
        let gap = BESIDE_SPACING * k;
        let da = pa.1.abs();
        let on_line = (da - gap).abs() <= gap / 2.0 || !a.2 && da <= gap * 3.0;
        let sign = |o: f64| if o >= 0.0 { 1.0 } else { -1.0 };
        let side = if da > gap / 2.0 && (a.2 || !on_line) { sign(pa.1) } else { sign(pb.1) };
        // `express_verts` takes the spacing as a multiple of the quad's
        let scale = side * BESIDE_SPACING / QUAD_SPACING;
        let level = ends[9] as i8;
        // the line from chainage `from` to the end, and its own ends
        let line = |from: f64| -> Option<(Vec<Pi>, (f64, f64), (f64, f64))> {
            if pb.0 - from < 1.0 {
                return None;
            }
            let sub = cut(&f, from, pb.0, level);
            let own = |k: usize| if a.2 && on_line && k == 0 { (a.0, a.1) } else { express_end(&sub, k, scale, scaled) };
            let (oa, ob) = (own(0), own(1));
            let (ex, _, _) = express_verts(&sub, 1, sub.len() - 2, scaled, scale, Some(oa), Some(ob))?;
            Some((ex, oa, ob))
        };
        // a start on the track meets it there, which is no crossing
        let crosses = |route: &[Pi]| route_crosses(route, &f.pieces, scaled, if da <= gap / 2.0 { gap } else { 0.0 });
        let (ex, oa, ob) = if on_line {
            let c = line(pa.0)?;
            if crosses(&c.0) {
                return None;
            }
            c
        } else {
            // T-1781: coming in, the turn onto the line where its auto radius is largest (to the
            // design radius, then the nearest), the line's corners still fitting and the way in
            // clear of the track; at the least the start's own distance along
            let near = pa.0 + (da / k).min((pb.0 - pa.0) / 2.0);
            let mut best: Option<(f64, (Vec<Pi>, (f64, f64), (f64, f64)))> = None;
            for t in 0..=TURN_IN_TRIES {
                let l = near + (pb.0 - near) * 0.9 * t as f64 / TURN_IN_TRIES as f64;
                let Some(c) = line(l) else { continue };
                // the route's own fit sizes the turn, the line's corners keeping their radii
                let mut route = vec![Pi::new(a.0, a.1, 0.0, level)];
                route.extend(c.0.iter().map(|p| Pi { level, ..*p }));
                route[1].radius = 0.0;
                let g = geom::fit_capped(&route, scaled, self.route_cap);
                if (!g.issues.is_empty() || !g.clamped.is_empty()) && t > 0 || crosses(&route) {
                    continue;
                }
                let r = g.radius[1];
                if best.as_ref().map_or(true, |b| r > b.0 + 1.0) {
                    best = Some((r, c));
                }
            }
            best?.1
        };
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
        Some((e, pis, on_line, run))
    }

    /// The run of track through piece `ea`, through plain joints and stations, as one vertex list
    /// with every curve's radius as fitted (so a cut leg keeps its curves' size), and its pieces.
    /// None if `eb` is not on it.
    pub(super) fn beside_run(&self, ea: u32, eb: u32, scaled: bool) -> Option<(Vec<Pi>, Vec<u32>)> {
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
        Some((v, run.iter().map(|r| r.0).collect()))
    }
}

/// Turn-in points tried along the track coming in to the line beside it (T-1781).
const TURN_IN_TRIES: usize = 16;
/// A curve cut shorter than this at an end of the line is left out, m.
const MIN_CUT_ARC: f64 = 0.5;

/// Alignment `f` from chainage `s0` to `s1` as corners at `level`: a point at each end, and for
/// each curve (or part of one) between, a corner where the tangents at that part's ends cross,
/// with the curve's radius set. Fitted, it is the same track, cut inside a curve if need be.
fn cut(f: &geom::Fit, s0: f64, s1: f64, level: i8) -> Vec<Pi> {
    let pt = |s: f64| {
        let (x, y, _) = geom::pos_at(&f.pieces, s);
        Pi::new(quant(x), quant(y), 0.0, level)
    };
    let mut out = vec![pt(s0)];
    for p in &f.pieces {
        let (a, b) = (p.s0.max(s0), (p.s0 + p.len).min(s1));
        if p.k == 0.0 || b - a < MIN_CUT_ARC {
            continue;
        }
        let (da, db) = (p.plane_d(a - p.s0), p.plane_d(b - p.s0));
        let (x, y, th) = p.at_plane(da);
        let t = p.radius() * ((db - da) * p.k.abs() / 2.0).tan();
        out.push(Pi::new(quant(x + t * th.cos()), quant(y + t * th.sin()), quant(p.ground_radius()), level));
    }
    out.push(pt(s1));
    out
}

/// Whether `route`, fitted, crosses the track `pieces` anywhere past `skip` along it.
pub(super) fn route_crosses(route: &[Pi], pieces: &[geom::Piece], scaled: bool, skip: f64) -> bool {
    let g = geom::fit_in(route, scaled);
    let mut hits = vec![];
    crate::track::cross::intersect_alignments(&g.pieces, pieces, false, &mut hits);
    hits.iter().any(|h| h.0 > skip)
}
