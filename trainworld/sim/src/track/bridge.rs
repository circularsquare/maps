//! A segment drawn at ground level across water becomes a bridge by itself (T-656).
//!
//! The drawing tool's alignment is re-expressed as the same shape with more vertices: each stretch
//! over water is raised to level 1, with ramps that end on land before the water and start on
//! land after it. A ramp is centred between two vertices of different levels (geom.rs), so a ramp
//! centred at `c` is placed by a vertex at `c - d` and another at `c + d` (d at least half a ramp).
//! On a straight a vertex is a PI with no turn; on an arc the arc is cut into arcs of the same
//! radius, each its own PI at the tangents' crossing, whose vertex chainage is the middle of its
//! arc. The preview and the edit both go through `bridge`, so they cannot disagree.

use super::cost::{water_spans, WaterMask};
use super::frame::merc_k;
use super::geom::{self, Fit, GeomIssue, Piece, Pi};
use super::params::*;

/// The level a bridge runs at.
const BRIDGE_LEVEL: i8 = 1;
/// A ramp up to the bridge, m.
const RAMP: f64 = RAMP_PER_LEVEL;
/// Land kept between a ramp and the water's edge: the mask is read every `WATER_STEP`.
const SHORE: f64 = WATER_STEP + 2.0;
/// Ramps that a curve in the way keeps off their place move back by this, up to `MAX_SHIFT`.
const SHIFT_STEP: f64 = 25.0;
const MAX_SHIFT: f64 = 400.0;
/// Half the distance between a ramp's two vertices, tried in turn: a ramp is the same wherever
/// they are, so a curve that has no room for them close in may have it further out.
const HALF_SPANS: [f64; 11] = [100.0, 125.0, 150.0, 200.0, 260.0, 340.0, 450.0, 600.0, 800.0, 1100.0, 1500.0];
/// The shortest arc an arc is cut into, m.
const MIN_CUT: f64 = 5.0;

/// There is no room on land for the ramps.
#[derive(Debug, PartialEq, Eq)]
pub struct NoRoom;

/// A ramp centred at `c`, with its vertices `d` either side. It goes up from the level before it
/// (even index: 0, odd: 1).
#[derive(Clone, Copy, Debug)]
struct Ramp {
    c: f64,
    d: f64,
}

/// One vertex a ramp needs. `follow` is the second vertex of its ramp.
#[derive(Clone, Copy, Debug)]
struct Req {
    x: f64,
    level: i8,
    follow: bool,
    /// The other vertex of the same ramp, where there is one (the end node can stand in).
    partner: Option<f64>,
}

/// A vertex of the new alignment; `level` None takes the level of the stretch it lies in.
struct V {
    pi: Pi,
    mid: f64,
    level: Option<i8>,
}

/// `verts` (all at level 0) with a bridge over each stretch of water, as the same alignment.
/// Ok(None) where nothing needs doing: no water, any other level, or an alignment that is
/// invalid anyway.
pub fn bridge(verts: &[Pi], scaled: bool, mask: &dyn WaterMask) -> Result<Option<Vec<Pi>>, NoRoom> {
    if mask.is_empty() || verts.len() < 2 || verts.iter().any(|p| p.level != 0) {
        return Ok(None);
    }
    let f = geom::fit_in(verts, scaled);
    if !f.issues.is_empty() || f.pieces.is_empty() {
        return Ok(None);
    }
    let mut wet = vec![];
    water_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); (x, y) }, mask, &mut wet);
    if wet.is_empty() {
        return Ok(None);
    }
    // The stretch each bridge must cover. Two close together share one, since there is not room
    // for the ramp down and the ramp up between them.
    let mut zones: Vec<[f64; 2]> = vec![];
    for w in &wet {
        let z = [w[0] - SHORE, w[1] + SHORE];
        match zones.last_mut() {
            Some(last) if z[0] - last[1] < 2.0 * RAMP + 2.0 * SHORE => last[1] = z[1],
            _ => zones.push(z),
        }
    }
    let mut shift = 0.0;
    while shift <= MAX_SHIFT {
        // the ramp up ends at the zone's start and the ramp down starts at its end, moved out
        let centres: Vec<f64> = zones.iter().flat_map(|z| [z[0] - shift - RAMP / 2.0, z[1] + shift + RAMP / 2.0]).collect();
        if centres[0] < RAMP / 2.0 - 1e-9 || centres[centres.len() - 1] > f.len - RAMP / 2.0 + 1e-9 {
            break;
        }
        if let Some(nv) = place(&f, verts, scaled, &centres, &wet) {
            return Ok(Some(nv));
        }
        shift += SHIFT_STEP;
    }
    Err(NoRoom)
}

/// The ramps one after another, each with the nearest vertices that hold it; then the whole.
fn place(f: &Fit, verts: &[Pi], scaled: bool, centres: &[f64], wet: &[[f64; 2]]) -> Option<Vec<Pi>> {
    let mut ramps: Vec<Ramp> = vec![];
    for &c in centres {
        let cap = c.min(f.len - c);
        // Vertices the alignment has already are the nearest to use for one end of the ramp.
        let mut spans: Vec<f64> = HALF_SPANS.to_vec();
        spans.push(cap);
        spans.extend(verts.iter().enumerate().skip(1).take(verts.len() - 2).map(|(i, _)| (f.vert_s[i] - c).abs()));
        spans.sort_by(f64::total_cmp);
        let mut tried: Vec<f64> = vec![];
        let mut found = false;
        for d in spans {
            // a vertex next to the end node is the end node
            let d = if d > cap - 0.5 { cap } else { d };
            if d < RAMP / 2.0 - 1e-9 || tried.contains(&d) {
                continue;
            }
            tried.push(d);
            ramps.push(Ramp { c, d });
            if build(f, verts, scaled, &ramps).is_some_and(|(_, g)| ramp_in_place(&g, c, ramps.len() % 2 == 0)) {
                found = true;
                break;
            }
            ramps.pop();
        }
        if !found {
            return None;
        }
    }
    let (nv, g) = build(f, verts, scaled, &ramps)?;
    water_held(&g, wet).then_some(nv)
}

/// The ramp at `c` is where it should be: it spans a ramp's length about `c`.
fn ramp_in_place(g: &Fit, c: f64, down: bool) -> bool {
    let (lo, hi) = if down { (LEVEL_H, 0.0) } else { (0.0, LEVEL_H) };
    let h = |s: f64| geom::height_at(&g.vert, s);
    (h(c - RAMP / 2.0 - 0.02) - lo).abs() < 0.05 && (h(c + RAMP / 2.0 + 0.02) - hi).abs() < 0.05 && (h(c) - LEVEL_H / 2.0).abs() < 0.05
}

fn floor_mm(v: f64) -> f64 {
    ((v / QUANT) + 1e-6).floor() * QUANT
}

/// The alignment with the ramps `ramps` (in order along it) and no others, fitted.
fn build(f: &Fit, verts: &[Pi], scaled: bool, ramps: &[Ramp]) -> Option<(Vec<Pi>, Fit)> {
    let n = verts.len();
    // the level after k ramps is 0 or 1
    let after = |k: usize| (k % 2) as i8 * BRIDGE_LEVEL;
    let mut reqs: Vec<Req> = vec![];
    for (k, r) in ramps.iter().enumerate() {
        let (u, v) = (r.c - r.d, r.c + r.d);
        let (has_u, has_v) = (u > 0.5, v < f.len - 0.5);
        if has_u {
            reqs.push(Req { x: u, level: after(k), follow: false, partner: has_v.then_some(v).or(Some(f.len)) });
        }
        if has_v {
            reqs.push(Req { x: v, level: after(k + 1), follow: true, partner: has_u.then_some(u) });
        }
    }
    // Which vertex each lies on: a straight (a PI with no turn) or an arc (cut into arcs).
    let mut on_arc: Vec<Vec<Req>> = vec![vec![]; n];
    let mut items: Vec<(f64, Vec<V>)> = vec![];
    // an end of a ramp at a vertex the alignment has is that vertex, at the ramp's level
    let mut fixed: Vec<Option<i8>> = vec![None; n];
    for r in &reqs {
        if let Some(i) = (1..n - 1).find(|&i| (f.vert_s[i] - r.x).abs() < 1e-6) {
            fixed[i] = Some(r.level);
            continue;
        }
        let j = geom::piece_at(&f.pieces, r.x);
        let piece = &f.pieces[j];
        if piece.k != 0.0 {
            on_arc[f.piece_at_vertex[j] as usize].push(*r);
        } else {
            let (x, y, _) = piece.at(r.x - piece.s0);
            items.push((r.x, vec![V { pi: Pi::new(quant(x), quant(y), 0.0, r.level), mid: r.x, level: Some(r.level) }]));
        }
    }
    for i in 1..n - 1 {
        // Every radius is set, since a PI next to a new one no longer has the legs the auto radius
        // was taken from.
        let kept = Pi::new(verts[i].x, verts[i].y, if f.radius[i] > 0.0 { floor_mm(f.radius[i]) } else { 0.0 }, 0);
        if on_arc[i].is_empty() {
            items.push((f.vert_s[i], vec![V { pi: kept, mid: f.vert_s[i], level: fixed[i] }]));
            continue;
        }
        let j = (0..f.pieces.len()).find(|&j| f.pieces[j].k != 0.0 && f.piece_at_vertex[j] as usize == i)?;
        items.push((f.vert_s[i], slice(&f.pieces[j], scaled, &on_arc[i])?));
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    let level_at = |m: f64| after(ramps.iter().filter(|r| r.c < m).count());
    let mut nv = vec![Pi::new(verts[0].x, verts[0].y, 0.0, 0)];
    for (_, vs) in &items {
        for v in vs {
            nv.push(Pi { level: v.level.unwrap_or_else(|| level_at(v.mid)), ..v.pi });
        }
    }
    nv.push(Pi::new(verts[n - 1].x, verts[n - 1].y, 0.0, after(ramps.len())));
    let g = settle(f, &mut nv, scaled)?;
    Some((nv, g))
}

/// An arc cut so that each request is the middle of one piece of it, and no vertex lies between
/// the two ends of a ramp. Each piece becomes a PI where the tangents at its ends cross, with the
/// radius the arc has.
fn slice(piece: &Piece, scaled: bool, reqs: &[Req]) -> Option<Vec<V>> {
    let (a, b) = (piece.s0, piece.s0 + piece.len);
    // (from, to, level of the piece)
    let mut cuts: Vec<(f64, f64, Option<i8>)> = vec![];
    let mut cur = a;
    for r in reqs {
        let (left, right);
        if r.follow {
            let h = r.x - cur;
            (left, right) = (cur, r.x + h);
        } else if r.partner.is_some_and(|p| p < b - 1e-6) {
            // The second end is on this arc too, in the piece that follows: this one is as wide as
            // it can be, so that one is as narrow as it can, since the pair reaches that far past
            // the second end.
            let gap = r.x - cur;
            let h = if gap - (RAMP - MIN_CUT) < MIN_CUT { gap } else { RAMP - MIN_CUT };
            (left, right) = (r.x - h, r.x + h);
        } else {
            // the second end is further on: this piece runs to the end of the arc
            let h = b - r.x;
            (left, right) = (r.x - h, b);
        }
        if left < cur - 1e-9 || right > b + 1e-9 || right - left < MIN_CUT {
            return None;
        }
        if left - cur > 1e-9 {
            if left - cur < MIN_CUT {
                return None;
            }
            cuts.push((cur, left, None));
        }
        cuts.push((left, right, Some(r.level)));
        cur = right;
    }
    if b - cur > 1e-9 {
        if b - cur < MIN_CUT {
            return None;
        }
        cuts.push((cur, b, None));
    }
    let rp = piece.radius();
    Some(cuts.iter().map(|&(c0, c1, level)| {
        let (x0, y0, th0) = piece.at(c0 - a);
        let (_, _, th1) = piece.at(c1 - a);
        let t = rp * (geom::wrap(th1 - th0).abs() / 2.0).tan();
        let ymid = piece.at((c0 + c1) / 2.0 - a).1;
        let k = if scaled { merc_k(ymid) } else { 1.0 };
        let pi = Pi::new(quant(x0 + t * th0.cos()), quant(y0 + t * th0.sin()), floor_mm(rp / k), 0);
        V { pi, mid: (c0 + c1) / 2.0, level }
    }).collect())
}

/// The new alignment fits and is the old one. Radii that the millimetre grid left a hair too big
/// are eased.
fn settle(f: &Fit, nv: &mut [Pi], scaled: bool) -> Option<Fit> {
    let mut g = geom::fit_in(nv, scaled);
    for _ in 0..10 {
        let bad: Vec<usize> = g.issues.iter().filter_map(|i| if let GeomIssue::ArcDoesNotFit(j) = i { Some(*j) } else { None }).collect();
        if bad.is_empty() {
            break;
        }
        for j in bad {
            nv[j].radius = floor_mm(nv[j].radius * 0.9995);
        }
        g = geom::fit_in(nv, scaled);
    }
    if !g.issues.is_empty() || g.pieces.is_empty() || (g.len - f.len).abs() > 0.05 {
        return None;
    }
    let mut s = 0.0;
    while s < f.len {
        let (a, b) = (geom::pos_at(&f.pieces, s), geom::pos_at(&g.pieces, s));
        if (a.0 - b.0).hypot(a.1 - b.1) > 0.25 {
            return None;
        }
        s += 25.0;
    }
    Some(g)
}

/// The track is high over all the water.
fn water_held(g: &Fit, wet: &[[f64; 2]]) -> bool {
    wet.iter().all(|w| {
        let mut s = w[0];
        while s <= w[1] {
            if geom::height_at(&g.vert, s) < LEVEL_H - 0.01 {
                return false;
            }
            s += WATER_STEP / 2.0;
        }
        geom::height_at(&g.vert, w[1]) >= LEVEL_H - 0.01
    })
}
