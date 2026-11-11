//! T-3092: real flyovers (notes/T-3085.md "Real flyovers"; T-3120 "Round 2"). With the ramp switch on,
//! a double-track route that makes a new junction J on double track lays it as a flying junction of
//! real track: its first stretch, as drawn, to J3 one-way single track for the direction that crosses
//! nothing, and the other direction a one-way single leg from J itself, out on the main's far side,
//! a level over the main ahead of J and down between the main and the first stretch to J3. Both
//! legs leave the main at J, the point clicked, so a flyover needs room only ahead of it. T-3170:
//! high-speed track too, its legs one-way single high-speed track.

use super::*;
use crate::track::lanes::join_for;

/// The crossing leg's distance from the plain leg beside it, and from the main on its far side, m.
const GAP: f64 = beside::BESIDE_SPACING;
/// At J3 the leg leaves along the route, curving at once toward the main, and is GAP from the
/// route SPLIT_LEAD + SPLIT_LEN out, m.
const SPLIT_LEAD: f64 = 20.0;
const SPLIT_LEN: f64 = 60.0;
/// How sharply the leg jogs out from the main to its far side at J, rad (12 degrees).
const JOG: f64 = 0.2094;
/// Leg needed between an end and the crossing: a level's ramp, the clearance at the crossing, and
/// room for the ramp to move off a curve, m.
const RAMP_ROOM: f64 = RAMP_PER_LEVEL + WATER_STEP + 2.0 + 20.0;
/// The route at J3 leaves the main at between these, rad (4 and 80 degrees; T-3188: down to -4,
/// as near parallel, two GAPs off it); the leg crosses the main at no more than `CROSS_MAX` (86
/// degrees).
const SHALLOWEST: f64 = 0.0698;
const STEEPEST: f64 = 1.396;
const CROSS_MAX: f64 = 1.501;
/// The leg alignments tried at each J3, the neatest first: (case A, far side's swing out, rad). A
/// swing of 0 is a jog to GAP beside the main; 20 to 50 degrees make room for a ramp when the
/// crossing comes soon after J.
const LEGS: [(bool, f64); 7] = [(true, 0.0), (true, 0.349), (true, 0.524), (false, 0.0), (true, 0.698), (true, 0.873), (false, 0.524)];
/// The radius a swung far side leaves J on, at most, m.
const SWING_R: f64 = 150.0;
/// The nearest J3 lies along the route, and the step it moves out by, m.
const J3_MIN: f64 = 150.0;
const J3_STEP: f64 = 10.0;
/// Plain track kept clear past the crossing and round a station's platform, m.
const CLEAR: f64 = 40.0;
/// The most leg alignments a junction tries ramping before it falls back to flat.
const RAMP_TRIES: usize = 8;
/// The shortest arc the leg's own corners take over the track's smallest, m.
const LEG_MIN_OVER: f64 = 5.0;
/// T-3169: what a corner may fall short of the radius aimed for, a share (the radius is floored to
/// the millimetre, and the S-curves are drawn 2% inside their limit).
const AIM_SLACK: f64 = 0.99;
/// T-3169: with no room for the flyover target, the leg aims this much slower in turn, km/h.
const AIM_STEP: f64 = 20.0;

/// T-3169: an S-curve of two arcs of radius `r` that steps GAP sideways: each arc's turn, rad, at
/// most `most`. 2% under the turn whose two arcs meet with no straight between, so the leg between
/// its corners holds both tangents.
fn s_turn(r: f64, most: f64) -> f64 {
    most.min(0.98 * (1.0 - GAP / (2.0 * r)).max(-1.0).acos())
}

/// Track walked from a node: per edge (edge, the node it is entered from, distance at its start),
/// to the first node that is not a plain joint, or `max`.
struct Walk {
    segs: Vec<(u32, u32, f64)>,
    len: f64,
    /// the node it stopped at, when that is not a plain joint, and its distance
    stop: Option<(u32, f64)>,
}

fn walk(net: &Network, from: u32, first: u32, max: f64, ok: impl Fn(u32) -> bool) -> Walk {
    let (mut segs, mut len, mut n, mut e) = (vec![], 0.0, from, first);
    loop {
        if !ok(e) || net.edge_crossover[e as usize] {
            return Walk { segs, len, stop: Some((n, len)) };
        }
        segs.push((e, n, len));
        len += net.edge_len[e as usize];
        let m = if net.edge_a[e as usize] == n { net.edge_b[e as usize] } else { net.edge_a[e as usize] };
        let p = &net.node_ports[m as usize];
        if p.n != 2 || net.node_platform[m as usize] > 0 {
            return Walk { segs, len, stop: Some((m, len)) };
        }
        if len >= max {
            return Walk { segs, len, stop: None };
        }
        let Some(&next) = p.ends().iter().find(|&&c| c >> 1 != e) else { return Walk { segs, len, stop: Some((m, len)) } };
        (n, e) = (m, next >> 1);
    }
}

/// The segment of `w` at distance `s`: (edge, offset along the edge from its a, walked a to b).
fn seg_at(net: &Network, w: &Walk, s: f64) -> Option<(u32, f64, bool)> {
    let &(e, from, s0) = w.segs.iter().rev().find(|x| x.2 <= s)?;
    let len = net.edge_len[e as usize];
    let t = s - s0;
    if t > len {
        return None;
    }
    let fwd = net.edge_a[e as usize] == from;
    Some((e, if fwd { t } else { len - t }, fwd))
}

/// Position and heading (along the walk) at distance `s` of `w`.
fn at(net: &Network, w: &Walk, s: f64) -> Option<(f64, f64, f64)> {
    let (e, t, fwd) = seg_at(net, w, s)?;
    let (x, y, th) = geom::pos_at(net.edge_pieces(e), t);
    Some((x, y, if fwd { th } else { th + PI }))
}

/// The leg `v` (its corners at `level`, J to J3) a level up over the track it crosses at its own
/// level, or else down under it, by the bridge's ramps (`shaped` as a route is): None where neither
/// fits. Where each end leaves the track it starts beside is no crossing.
fn ramp_leg(net: &Network, v: &[Pi], level: i8) -> Option<Vec<Pi>> {
    let scaled = net.frame().scaled();
    let water = bridge::water(net.water_mask.as_ref());
    for up in [level >= 0, level < 0] {
        let wet = up && level == 0;
        let spans = |f: &geom::Fit, fine: bool| {
            let mut s = super::draw::flat_spans(net, f, level, 0.0);
            s.retain(|w| w[0] > SHARED_NODE_CLEAR + 5.0 && w[1] < f.len - SHARED_NODE_CLEAR - 5.0);
            if wet {
                s.extend(water(f, fine));
                s.sort_by(|a, b| a[0].total_cmp(&b[0]));
            }
            s
        };
        if let Ok(Some(out)) = bridge::ramp_over(v, scaled, &spans, level, up, [false; 2], false) {
            let to = level + if up { 1 } else { -1 };
            if super::draw::flat_spans(net, &geom::fit_in(&out, scaled), to, to as f64 * LEVEL_H).is_empty() {
                return Some(out);
            }
        }
    }
    None
}

/// T-3120: the leg `v` (J to J3, at `level`, crossing the main at chainage `s_x`) a level up or down
/// by raising a run of its own corners: `geom::vertical` lays each ramp between a corner at one level
/// and the next at the other, spreading over the corners either side (T-1895), so a ramp may hold
/// curves. The shortest run that keeps SHORE level either side of the crossing, crosses nothing else
/// flat at either level and keeps every ramp whole; None where none does.
fn ramp_levels(net: &Network, v: &[Pi], level: i8, s_x: f64) -> Option<Vec<Pi>> {
    let scaled = net.frame().scaled();
    let n = v.len();
    let ends = |s: &Vec<[f64; 2]>, len: f64| s.iter().any(|w| w[0] > SHARED_NODE_CLEAR + 5.0 && w[1] < len - SHARED_NODE_CLEAR - 5.0);
    let mut runs: Vec<(usize, usize)> = (1..n - 1).flat_map(|a| (a..n - 1).map(move |b| (a, b))).collect();
    runs.sort_by_key(|r| (r.1 - r.0, r.0));
    for dir in if level >= 0 { [1, -1] } else { [-1, 1] } {
        let to = level + dir;
        if !(MIN_LEVEL..=MAX_LEVEL).contains(&to) {
            continue;
        }
        let z = to as f64 * LEVEL_H;
        for &(a, b) in &runs {
            let mut w = v.to_vec();
            for p in &mut w[a..=b] {
                p.level = to;
            }
            let f = geom::fit_in(&w, scaled);
            if !f.issues.is_empty() {
                continue;
            }
            let mut s = s_x - SHORE;
            let mut level_there = true;
            while s <= s_x + SHORE {
                level_there &= (geom::height_at(&f.vert, s) - z).abs() < 0.01;
                s += 2.0;
            }
            if level_there && !ends(&super::draw::flat_spans(net, &f, to, z), f.len) && !ends(&super::draw::flat_spans(net, &f, level, level as f64 * LEVEL_H), f.len) {
                return Some(w);
            }
        }
    }
    None
}

/// Level track kept either side of where the leg crosses the main, as a bridge keeps from water, m.
const SHORE: f64 = WATER_STEP + 2.0;

/// `verts` with each corner's radius set to what `fit_capped` gives it at `cap` (an arc as wide as
/// its legs hold, up to the cap), so later fits at the usual cap keep it; None where one would be
/// under `floor` (T-3169: the radius aimed for, less `AIM_SLACK`).
fn set_radii(verts: &mut [Pi], scaled: bool, cap: f64, floor: f64) -> Option<geom::Fit> {
    let f = geom::fit_capped(verts, scaled, cap);
    if !f.issues.is_empty() {
        return None;
    }
    for i in 1..verts.len() - 1 {
        if f.radius[i] > 0.0 {
            if f.radius[i] < floor {
                return None;
            }
            verts[i].radius = ((f.radius[i] / QUANT).floor() - 1.0) * QUANT;
        }
    }
    let g = geom::fit_in(verts, scaled);
    (g.issues.is_empty() && g.clamped.is_empty()).then_some(g)
}

impl TrackApi {
    /// The route `r` (from `route_op_plain`) with a flyover at each new junction it makes on double
    /// track that has room; the others stay flat and are counted in `flyover_fell_back`. Each
    /// flyover is planned on the route sketched in, and kept only where the whole edit passes the
    /// checks and leaves no move crossing another at its junctions.
    pub(super) fn with_flyovers(&mut self, r: (Op, Vec<Pi>, Vec<u32>, Vec<u32>), tracks: u8) -> (Op, Vec<Pi>, Vec<u32>, Vec<u32>) {
        let (op, verts, mut nn, mut ne) = r;
        let js: Vec<u32> = self.route_junctions.iter().map(|j| j.0).collect();
        // T-3170: regular and high-speed double track
        if tracks != 2 || self.route_kind_of(2).spec().family != 0 || js.is_empty() {
            return (op, verts, nn, ne);
        }
        let mut ops = match op {
            Op::Batch(v) => v,
            o => vec![o],
        };
        // the route's own pieces: its new edges less the halves of the track it splits
        let halves: Vec<u32> = ops.iter().filter_map(|o| if let Op::Split { new_edge, .. } = o { Some(*new_edge) } else { None }).collect();
        let route_edges: Vec<u32> = ne.iter().copied().filter(|e| !halves.contains(e)).collect();
        // a route the checks refuse anyway gets no flyover and no note
        match self.w.sketch(Op::Batch(ops.clone()), true) {
            Ok(inv) => self.w.unsketch(inv),
            Err(_) => return (Op::Batch(ops), verts, nn, ne),
        }
        let (saved_bridge, saved_short) = (self.bridge_refused, self.bridge_short.clone());
        for j in js {
            let Ok(inv) = self.w.sketch(Op::Batch(ops.clone()), false) else { break };
            let plan = self.plan_flyover(j, &route_edges);
            self.w.unsketch(inv);
            let (more, pn, pe, leg) = match plan {
                Some(Some(p)) => p,
                Some(None) => {
                    self.flyover_fell_back += 1;
                    continue;
                }
                None => continue,
            };
            let mut all = ops.clone();
            all.extend(more);
            let laid = match self.w.sketch(Op::Batch(all.clone()), true) {
                Ok(inv) => {
                    let net = &self.w.net;
                    let clean = [j].iter().chain(&pn).all(|&n| !crate::track::capacity::junction_crosses(net, n));
                    self.w.unsketch(inv);
                    clean
                }
                Err(_) => false,
            };
            if laid {
                ops = all;
                nn.extend(pn);
                ne.extend(pe);
                self.route_legs.push(leg);
                self.route_junctions.retain(|x| x.0 != j);
            } else {
                self.release(&pn, &pe);
                self.flyover_fell_back += 1;
            }
        }
        (self.bridge_refused, self.bridge_short) = (saved_bridge, saved_short);
        (Op::Batch(ops), verts, nn, ne)
    }

    /// The ops laying a flyover at junction `j` of the route sketched in (`route_edges` its
    /// pieces), the nodes and edges they add, and the crossing leg's alignment. None where `j` is
    /// not a split of double track by a double route; Some(None) where it is but has no room.
    #[allow(clippy::type_complexity)]
    fn plan_flyover(&mut self, j: u32, route_edges: &[u32]) -> Option<Option<(Vec<Op>, Vec<u32>, Vec<u32>, Vec<Pi>)>> {
        let net = &self.w.net;
        let i = j as usize;
        let p = net.node_ports[i];
        if p.n != 3 || net.node_platform[i] > 0 || net.node_flying[i] {
            return None;
        }
        let ends: Vec<(u32, u8)> = (0..3).map(|k| (p.end[k] >> 1, p.side[k])).collect();
        let r = ends.iter().position(|e| route_edges.contains(&e.0))?;
        let (re, rside) = ends[r];
        let fwd = ends.iter().enumerate().find(|(k, e)| *k != r && e.1 == rside)?.1 .0;
        let back = ends.iter().enumerate().find(|(k, e)| *k != r && e.1 != rside)?.1 .0;
        let rail = |e: u32| net.edge_tracks[e as usize] == 2 && !net.edge_crossover[e as usize] && net.edge_kind[e as usize].spec().family == 0;
        if !rail(re) || !rail(fwd) || !rail(back) {
            return None;
        }
        Some(self.lay_flyover(j, re, fwd, rside, route_edges))
    }

    /// T-3120: the flyover at `j`, as tight as its curves allow (T-3169: each curve at least the
    /// flyover target's radius, else a quarter of it, else the leg's least). J3 is the first point out
    /// along the route where the crossing leg fits: from `j` it jogs GAP out to the main's far side and runs
    /// beside it, turns across the main once a ramp up has room, and comes down between the main and
    /// the route, GAP from it, to J3: along a line beside J3's tangent where that crosses the main
    /// far enough ahead (case A), else straight from just past J3 at the angle that puts the
    /// crossing there (case B). Its curves are as wide as their legs hold, up to the target speed's.
    #[allow(clippy::type_complexity)]
    fn lay_flyover(&mut self, j: u32, re: u32, fwd: u32, rside: u8, route_edges: &[u32]) -> Option<(Vec<Op>, Vec<u32>, Vec<u32>, Vec<Pi>)> {
        let net = &self.w.net;
        let i = j as usize;
        let level = net.node_level[i];
        let scaled = net.frame().scaled();
        let (jx, jy) = (net.node_x[i], net.node_y[i]);
        let k = net.k_at(jy);
        let h = net.node_heading[i] + if rside == 1 { 0.0 } else { PI };
        let u = (h.cos(), h.sin());
        let route = walk(net, j, re, 3000.0, |e| route_edges.contains(&e));
        // which side the branch leaves on; `nrm` points that way
        let probe = at(net, &route, route.len.min(150.0) - 1.0)?;
        let left = u.0 * (probe.1 - jy) - u.1 * (probe.0 - jx) > 0.0;
        let nrm = if left { (-u.1, u.0) } else { (u.1, -u.0) };
        // the main's frame at J, ground m: `a` along the main, `l` toward the branch's side
        let local = |x: f64, y: f64| (((x - jx) * u.0 + (y - jy) * u.1) / k, ((x - jx) * nrm.0 + (y - jy) * nrm.1) / k);
        let plane = |a: f64, l: f64| Pi::new(quant(jx + (u.0 * a + nrm.0 * l) * k), quant(jy + (u.1 * a + nrm.1 * l) * k), 0.0, level);
        // trains keep to the running side, so the move to the branch crosses the other direction's
        // track when the branch leaves on the other side; it gets the leg over the main
        let crossing_out = left == net.right_hand;
        let limit = match route.stop {
            Some((n, d)) if net.node_platform[n as usize] > 0 => d - net.node_platform[n as usize] as f64 / 2.0 - CLEAR,
            Some((_, d)) => d - 50.0,
            None => route.len - 50.0,
        };
        // the main ahead, as far as the next station, junction or end
        let main = walk(net, j, fwd, 3000.0, |_| true);
        let main_room = match main.stop {
            Some((n, d)) => d - net.node_platform[n as usize] as f64 / 2.0 - CLEAR,
            None => main.len,
        };
        let leg_min = net.edge_kind[re as usize].spec().min_radius + LEG_MIN_OVER;
        // T-3169: every curve of the leg at the flyover target's radius where there is the room, else
        // at the radius of each `AIM_STEP` km/h less in turn, else as tight as the leg goes
        // never faster than the route's own target max speed: its curves hold it to that anyway
        let v_aim = (A_LAT * self.flyover_r.min(self.route_cap)).sqrt() * 3.6;
        let mut aims: Vec<f64> = (0..).map(|s| v_aim - AIM_STEP * s as f64).take_while(|&v| v >= 2.0 * AIM_STEP).map(design_radius).filter(|&r| r > leg_min).collect();
        aims.push(leg_min);
        let mut laid = None;
        for r_aim in aims {
            let floor = leg_min.max(r_aim * AIM_SLACK);
            let cap = self.route_cap.max(r_aim);
            // where a curve under the aim wants a crossing further on (case B), how much further, m
            let grow = (r_aim > leg_min).then_some(0.25 * r_aim);
            // The leg's far side, J to the corner where it turns across the main at `phi`, crossing at
            // `x_c`. `swing` 0: a jog out to -GAP, then beside the main. Else straight out at `swing`
            // (rad) to that corner, more track for a crossing close to J.
            let jog = s_turn(r_aim, JOG);
            let jog_t = GAP / jog.sin() / 2.0;
            let jog_d = (jog_t / (jog / 2.0).tan()).min(cap) * (jog / 2.0).tan();
            let swing_r = SWING_R.max(r_aim).min(cap);
            // At J3 the leg runs `lead` along the route, then parts from it by GAP over `split_len`.
            let split = s_turn(r_aim, (GAP / SPLIT_LEN).atan());
            let split_len = GAP / split.tan();
            let lead = SPLIT_LEAD.max(1.02 * r_aim * (split / 2.0).tan());
            let far_side = |x_c: f64, phi: f64, swing: f64| -> Option<Vec<(f64, f64)>> {
                if swing == 0.0 {
                    let (e, c) = (jog_d + GAP / jog.tan(), x_c - GAP / phi.tan());
                    return (c - e > 1.0).then(|| vec![(0.0, 0.0), (jog_d, 0.0), (e, -GAP), (c, -GAP)]);
                }
                let d = swing_r * (swing / 2.0).tan();
                let l1 = (x_c - d) * phi.sin() / (phi + swing).sin();
                (x_c - d > 2.0 * d).then(|| vec![(0.0, 0.0), (d, 0.0), (d + l1 * swing.cos(), -l1 * swing.sin())])
            };
            // The leg from J through `far` and `near` (the near side's corners, from the crossing on) to
            // J3 at `p3` (plane `at3`), its corners rounded, and where it crosses the main. Err(Some(m)):
            // m more metres of far side wanted (less, where m is under 0).
            let leg = |far: Vec<(f64, f64)>, near: &[(f64, f64)], p3: (f64, f64), at3: (f64, f64), d3: f64| -> Result<(Vec<Pi>, f64), Option<f64>> {
                let mut v: Vec<Pi> = far.iter().chain(near).chain([&p3]).map(|&(a, l)| plane(a, l)).collect();
                let last = v.len() - 1;
                (v[0].x, v[0].y, v[last].x, v[last].y) = (jx, jy, at3.0, at3.1);
                let fit = set_radii(&mut v, scaled, cap, floor).ok_or(grow)?;
                let mut hits = vec![];
                let mut s_x = f64::MAX;
                for &(e, _, _) in &main.segs {
                    hits.clear();
                    crate::track::cross::intersect_alignments(&fit.pieces, net.edge_pieces(e), false, &mut hits);
                    s_x = hits.iter().map(|h| h.0).filter(|&s| s > SHARED_NODE_CLEAR + 5.0).fold(s_x, f64::min);
                }
                if s_x == f64::MAX {
                    return Err(None);
                }
                // T-3188: a crossing too near J3 wants that much less far side
                if fit.len - s_x < RAMP_ROOM {
                    return Err(Some(fit.len - s_x - RAMP_ROOM));
                }
                if s_x < RAMP_ROOM {
                    return Err(Some(RAMP_ROOM - s_x));
                }
                // GAP or more from the route from where it is beside the main to J3's split, and so
                // never across it
                let mut s = 2.0 * jog_d + GAP / jog.tan();
                while s < fit.len - lead - split_len {
                    let (lx, ly, _) = geom::pos_at(&fit.pieces, s);
                    if route.segs.iter().filter(|g| g.2 < d3).any(|g| net.project(g.0, lx, ly).1 < (GAP - 0.5) * k) {
                        return Err(None);
                    }
                    s += 10.0;
                }
                Ok((v, s_x))
            };
            let mut tries = 0;
            let step = J3_STEP.max(r_aim / 60.0);
            let mut d3 = J3_MIN.max(lead + split_len);
            laid = 'search: loop {
                d3 += step;
                if d3 > limit || tries >= RAMP_TRIES {
                    break None;
                }
                let Some((e, t, f)) = seg_at(net, &route, d3) else { continue };
                if t < 10.0 || t > net.edge_len[e as usize] - 10.0 {
                    continue;
                }
                let Some((x, y, th)) = at(net, &route, d3) else { break None };
                let p3 = local(x, y);
                let psi = (th.cos() * nrm.0 + th.sin() * nrm.1).atan2(th.cos() * u.0 + th.sin() * u.1);
                // T-3188: a route nearly parallel to the main takes a flyover too, J3 two GAPs off it
                let shallow = psi < SHALLOWEST;
                if !(-SHALLOWEST..=STEEPEST).contains(&psi) || p3.1 < if shallow { 2.0 * GAP } else { GAP + 5.0 } {
                    continue;
                }
                // back along the route from J3, and toward the main square to it
                let b = (-psi.cos(), -psi.sin());
                let m = (psi.sin(), -psi.cos());
                for &(beside, swing) in &LEGS {
                    let found = if beside {
                        // case A: GAP beside J3's tangent from the split back, then straight to the
                        // main at `phi`: along that tangent where it meets the main far enough
                        // ahead, else (T-3169, T-3188) further ahead and steeper, so the crossing
                        // curve has the room its radius wants
                        let n0 = (p3.0 + (lead + split_len) * b.0 + GAP * m.0, p3.1 + (lead + split_len) * b.1 + GAP * m.1);
                        let near = [n0, (p3.0 + lead * b.0, p3.1 + lead * b.1)];
                        // the far side's last corner, GAP across the main
                        let mut c = (n0.0 - (n0.1 + GAP) / psi.max(SHALLOWEST).tan()).max(jog_d + GAP / jog.tan() + 2.0);
                        let mut got = None;
                        for _ in 0..if grow.is_some() { 12 } else { 4 } {
                            let phi = (n0.1 + GAP).atan2(n0.0 - c);
                            let x_c = c + GAP / phi.tan();
                            if phi > CROSS_MAX || x_c > main_room {
                                break;
                            }
                            let Some(far) = far_side(x_c, phi, swing) else {
                                c += 0.25 * r_aim;
                                continue;
                            };
                            match leg(far, &near, p3, (x, y), d3) {
                                Ok(r) => {
                                    got = Some(r);
                                    break;
                                }
                                Err(Some(more)) => c += more + more.signum(),
                                Err(None) => break,
                            }
                        }
                        got
                    } else {
                        // case B: from just past J3 straight to a crossing far enough ahead, a little
                        // further out where the turn is sharp, so its arc holds the radius aimed for
                        let mut x_need = RAMP_ROOM;
                        let mut got = None;
                        for _ in 0..if grow.is_some() { 12 } else { 4 } {
                            let mut lead = SPLIT_LEAD;
                            for _ in 0..3 {
                                let phi = (p3.1 + lead * b.1).atan2(p3.0 + lead * b.0 - x_need);
                                lead = SPLIT_LEAD.max(1.2 * r_aim * ((phi - psi).max(0.0) / 2.0).tan());
                            }
                            let p1 = (p3.0 + lead * b.0, p3.1 + lead * b.1);
                            let phi = p1.1.atan2(p1.0 - x_need);
                            if phi - psi < 0.03 || phi > CROSS_MAX || x_need > main_room {
                                break;
                            }
                            let Some(far) = far_side(x_need, phi, swing) else { break };
                            match leg(far, &[p1], p3, (x, y), d3) {
                                Ok(r) => {
                                    got = Some(r);
                                    break;
                                }
                                Err(Some(more)) if more > 0.0 => x_need += more + 1.0,
                                Err(_) => break,
                            }
                        }
                        got
                    };
                    let Some((v, s_x)) = found else { continue };
                    tries += 1;
                    if let Some(shape) = ramp_levels(net, &v, level, s_x).or_else(|| ramp_leg(net, &v, level)) {
                        break 'search Some((e, t, f, shape, d3));
                    }
                    if tries >= RAMP_TRIES {
                        break 'search None;
                    }
                }
            };
            if laid.is_some() {
                break;
            }
        }
        let (e3, t3, fwd3, shape, d3) = laid?;
        let (nd3, mut first3, mut second3) = net.split_plan(e3, t3).ok()?;
        if nd3.level != level {
            return None;
        }
        // the ops: J3 splits the route, its J side one-way single track, then the leg from J to J3
        let net = &mut self.w.net;
        let (j3, e3b, leg) = (net.alloc_node(), net.alloc_edge(), net.alloc_edge());
        let net = &self.w.net;
        let plain_out = !crossing_out;
        let mut ops = vec![Op::Split { edge: e3, s: t3, node: j3, new_edge: e3b }];
        (first3.b, second3.a) = (j3, j3);
        for &(e, from, _) in route.segs.iter().filter(|x| x.2 < d3) {
            let fwd = net.edge_a[e as usize] == from;
            let (id, mut d) = if e == e3 {
                if fwd3 { (e3, first3.clone()) } else { (e3b, second3.clone()) }
            } else {
                (e, net.edge_data(e)?)
            };
            d.tracks = 1;
            d.oneway = if fwd == plain_out { 1 } else { 2 };
            // T-3137: its ends at J and J3 join the track that runs its way
            for (k, n) in [(0, d.a), (1, d.b)] {
                if n == j || n == j3 {
                    d.join[k] = join_for((n == j) == plain_out, net.right_hand);
                }
            }
            ops.push(Op::Edge { id, data: Some(d) });
        }
        let pis = shape[1..shape.len() - 1].to_vec();
        ops.push(Op::Edge { id: leg, data: Some(EdgeData { a: j, b: j3, tracks: 1, pis, built: false, off: false, thru: vec![], crossover: false, kind: net.edge_kind[re as usize], sep: false, day: 0, oneway: if crossing_out { 1 } else { 2 }, join: [join_for(crossing_out, net.right_hand), join_for(!crossing_out, net.right_hand)] }) });
        Some((ops, vec![j3], vec![e3b, leg], shape))
    }
}
