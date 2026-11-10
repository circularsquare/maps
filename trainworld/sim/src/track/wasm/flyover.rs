//! T-3092: real flyovers (notes/T-3085.md "Real flyovers"). With the ramp switch on, a double-track
//! route that makes a new junction J1 on double track lays it as a flying junction of real track:
//! its first stretch to J3 one-way single track for the direction that crosses nothing, and the
//! other direction a one-way single leg from J3 over the main to J2, a split on the main.

use super::*;

/// The least route from J1 to J3, m.
const PLAIN_MIN: f64 = 300.0;
/// The leg's track from J3 to where it crosses the main and from there to J2: a level's ramp, the
/// clearance at the node and at the crossing, and a margin, m.
const LEG_RAMP_ROOM: f64 = SHARED_NODE_CLEAR + RAMP_PER_LEVEL + 15.0 + 30.0;
/// Each leg end's lead along the track it leaves, m.
const LEAD: f64 = 80.0;
/// Plain track kept clear past J2 and round a station's platform, m.
const CLEAR: f64 = 40.0;
/// How steeply the leg comes back in to the main at J2, across over along.
const BACK_SLOPE: f64 = 0.25;
/// The nearest J2 lies to J1, m.
const J2_MIN: f64 = 60.0;

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

/// The leg `v` (its corners at `level`, J3 to J2) a level up over the track it crosses at its own
/// level, or else down under it, by the bridge's ramps (`shaped` as a route is): None where neither
/// fits. Where each end's lead leaves the track it starts on is no crossing.
fn ramp_leg(net: &Network, v: &[Pi], level: i8) -> Option<Vec<Pi>> {
    let scaled = net.frame().scaled();
    let water = bridge::water(net.water_mask.as_ref());
    for up in [level >= 0, level < 0] {
        let wet = up && level == 0;
        let spans = |f: &geom::Fit, fine: bool| {
            let mut s = super::draw::flat_spans(net, f, level, 0.0);
            s.retain(|w| w[0] > LEAD + 5.0 && w[1] < f.len - LEAD - 5.0);
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

impl TrackApi {
    /// The route `r` (from `route_op_plain`) with a flyover at each new junction it makes on double
    /// track that has room; the others stay flat and are counted in `flyover_fell_back`. Each
    /// flyover is planned on the route sketched in, and kept only where the whole edit passes the
    /// checks.
    pub(super) fn with_flyovers(&mut self, r: (Op, Vec<Pi>, Vec<u32>, Vec<u32>), tracks: u8) -> (Op, Vec<Pi>, Vec<u32>, Vec<u32>) {
        let (op, verts, mut nn, mut ne) = r;
        let js: Vec<u32> = self.route_junctions.iter().map(|j| j.0).collect();
        if tracks != 2 || self.route_kind_of(2) != TrackKind::Regular || js.is_empty() {
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
            match self.w.sketch(Op::Batch(all.clone()), true) {
                Ok(inv) => {
                    self.w.unsketch(inv);
                    ops = all;
                    nn.extend(pn);
                    ne.extend(pe);
                    self.route_legs.push(leg);
                    self.route_junctions.retain(|x| x.0 != j);
                }
                Err(_) => {
                    self.release(&pn, &pe);
                    self.flyover_fell_back += 1;
                }
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
        if !rail(re) || !rail(fwd) || !rail(back) || net.edge_kind[re as usize] != TrackKind::Regular {
            return None;
        }
        Some(self.lay_flyover(j, re, fwd, back, rside, route_edges))
    }

    #[allow(clippy::type_complexity)]
    fn lay_flyover(&mut self, j: u32, re: u32, fwd: u32, back: u32, rside: u8, route_edges: &[u32]) -> Option<(Vec<Op>, Vec<u32>, Vec<u32>, Vec<Pi>)> {
        let net = &self.w.net;
        let i = j as usize;
        let level = net.node_level[i];
        let (jx, jy) = (net.node_x[i], net.node_y[i]);
        let k = net.k_at(jy);
        let h = net.node_heading[i] + if rside == 1 { 0.0 } else { PI };
        let u = (h.cos(), h.sin());
        let route = walk(net, j, re, 3000.0, |e| route_edges.contains(&e));
        // which side the branch leaves on; `nrm` points that way
        let probe = at(net, &route, route.len.min(150.0) - 1.0)?;
        let left = u.0 * (probe.1 - jy) - u.1 * (probe.0 - jx) > 0.0;
        let nrm = if left { (-u.1, u.0) } else { (u.1, -u.0) };
        let lat = |x: f64, y: f64| ((x - jx) * nrm.0 + (y - jy) * nrm.1) / k;
        let along = |x: f64, y: f64| ((x - jx) * u.0 + (y - jy) * u.1) / k;
        // trains keep to the running side, so the move to the branch crosses the other direction's
        // track when the branch leaves on the other side; it gets the leg over the main
        let crossing_out = left == net.right_hand;
        let limit = match route.stop {
            Some((n, d)) if net.node_platform[n as usize] > 0 => d - net.node_platform[n as usize] as f64 / 2.0 - CLEAR,
            Some((_, d)) => d - 50.0,
            None => route.len - 50.0,
        };
        // J3: the first point PLAIN_MIN or more out whose tangent back toward J1 reaches the main
        // far enough off for a ramp. The leg runs straight on along it, so it stays on the main's
        // side of the plain leg (a curve away from the main lies off its tangents), and as far again
        // past the main for the ramp down.
        let mut d3 = PLAIN_MIN;
        let (e3, t3, fwd3, b3, reach) = loop {
            if d3 > limit {
                return None;
            }
            if let Some((e, t, f)) = seg_at(net, &route, d3) {
                if t > 10.0 && t < net.edge_len[e as usize] - 10.0 {
                    let (x, y, th) = at(net, &route, d3)?;
                    let b = (-th.cos(), -th.sin());
                    let rate = -(b.0 * nrm.0 + b.1 * nrm.1);
                    if rate > 0.05 && lat(x, y) / rate >= LEG_RAMP_ROOM {
                        break (e, t, f, b, lat(x, y) / rate);
                    }
                }
            }
            d3 += 25.0;
        };
        let (nd3, mut first3, mut second3) = net.split_plan(e3, t3).ok()?;
        if nd3.level != level {
            return None;
        }
        let far = (reach + LEG_RAMP_ROOM) * k;
        let c = (nd3.x + b3.0 * far, nd3.y + b3.1 * far);
        let mut v = vec![Pi::new(nd3.x, nd3.y, 0.0, level), Pi::new(quant(c.0), quant(c.1), 0.0, level)];
        // J2 on the main, far enough back that the leg comes in to it at a gentle angle
        let mut x2 = along(c.0, c.1) - (-lat(c.0, c.1) / BACK_SLOPE).max(150.0) - LEAD;
        if x2.abs() < J2_MIN {
            x2 = -J2_MIN;
        }
        let main = walk(net, j, if x2 < 0.0 { back } else { fwd }, x2.abs() + CLEAR, |_| true);
        if main.len < x2.abs() + CLEAR && main.stop.is_some() {
            return None;
        }
        let (e2, t2, _) = seg_at(net, &main, x2.abs())?;
        let (nd2, _, _) = net.split_plan(e2, t2).ok()?;
        if nd2.level != level {
            return None;
        }
        // clear of every station's platform
        for m in 0..net.node_count() {
            let pl = net.node_platform[m];
            if pl > 0 && net.node_ok(m as u32) && (net.node_x[m] - nd2.x).hypot(net.node_y[m] - nd2.y) / k < pl as f64 / 2.0 + CLEAR {
                return None;
            }
        }
        let (_, _, thw) = at(net, &main, x2.abs())?;
        let to = if x2 < 0.0 { thw + PI } else { thw };
        let tv = (to.cos(), to.sin());
        v.push(Pi::new(quant(nd2.x + tv.0 * LEAD * k), quant(nd2.y + tv.1 * LEAD * k), 0.0, level));
        v.push(Pi::new(nd2.x, nd2.y, 0.0, level));
        let shape = ramp_leg(net, &v, level);
        let shape = shape?;
        // the ops: J3 splits the route, its J1 side one-way single track, J2 splits the main, the leg
        let net = &mut self.w.net;
        let (j3, e3b, j2, e2b, leg) = (net.alloc_node(), net.alloc_edge(), net.alloc_node(), net.alloc_edge(), net.alloc_edge());
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
            ops.push(Op::Edge { id, data: Some(d) });
        }
        ops.push(Op::Split { edge: e2, s: t2, node: j2, new_edge: e2b });
        let pis = shape[1..shape.len() - 1].to_vec();
        ops.push(Op::Edge { id: leg, data: Some(EdgeData { a: j3, b: j2, tracks: 1, pis, built: false, off: false, thru: vec![], crossover: false, kind: TrackKind::Regular, sep: false, day: 0, oneway: if crossing_out { 2 } else { 1 } }) });
        Some((ops, vec![j3, j2], vec![e3b, e2b, leg], shape))
    }
}
