//! Service: stop times, profiles, trips, junctions, delays and markers.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- service
    /// Time of each stop from the start of the line (SPEC 6.3), seconds: arrival times.
    pub fn stop_times(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.cumulative(run, level).to_vec())
    }
    /// Departure time from each stop, seconds from the start of the trip.
    pub fn stop_departures(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].prof[level].dep.clone())
    }
    /// T-842: the share of each period's riders at each stop with a train due within their
    /// tolerance (`span::line_span`), `[(period * 2 + run) * n + stop]`, each run's stops in its
    /// travel order; empty when the line is not running. Worked out again only when the line's
    /// steady departures change.
    pub fn line_span(&mut self, line: u32) -> Vec<f32> {
        let l = line as usize;
        let (net, svc) = (&self.w.net, &self.w.svc);
        let Some(lat) = svc.lattice(net, l) else { return vec![] };
        let ls = &svc.lines[l];
        let key = crate::track::span::span_key(ls, &lat);
        if self.spans.len() <= l {
            self.spans.resize(l + 1, None);
        }
        match &self.spans[l] {
            Some((k, v)) if *k == key => v.clone(),
            _ => {
                let v = crate::track::span::line_span(ls, &lat);
                self.spans[l] = Some((key, v.clone()));
                v
            }
        }
    }
    /// All phase tables: 4 floats per phase (t, s, v, a). notes/T-040.md "Keyframes".
    pub fn phases(&self) -> Vec<f32> {
        let mut out = vec![];
        for l in &self.w.svc.lines {
            for r in 0..2 {
                for k in 0..LEVELS {
                    if l.ok {
                        out.extend(l.runs[r].prof[k].phases.iter().flat_map(|p| [p.t as f32, p.s as f32, p.v as f32, p.a as f32]));
                    }
                }
            }
        }
        out
    }
    /// Per profile `(line * 2 + run) * 3 + level`: first phase, phase count, trip seconds
    /// (run plus turnaround), path length m. Absent lines have count 0.
    pub fn profile_meta(&self) -> Vec<f32> {
        let mut out = vec![];
        let mut first = 0usize;
        for (li, l) in self.w.svc.lines.iter().enumerate() {
            for r in 0..2 {
                for k in 0..LEVELS {
                    if l.ok {
                        let n = l.runs[r].prof[k].phases.len();
                        out.extend([first as f32, n as f32, l.trip_len(&self.w.net, li as u32, r, k) as f32, l.runs[r].len as f32]);
                        first += n;
                    } else {
                        out.extend([first as f32, 0.0, 0.0, 0.0]);
                    }
                }
            }
        }
        out
    }
    /// Trips drawn in `[epoch, epoch + 3600)`: per trip, profile index, departure relative to
    /// `epoch` (may be negative: still running from before) and when it stops being drawn,
    /// relative to `epoch` (its train leaves service or takes its next departure, T-206), and the
    /// departure its train takes next from that terminal, relative to `epoch` (the other run's
    /// trip drawn from then; NaN when it leaves service there, T-245). Running lines only.
    pub fn trips(&self, epoch: f64) -> Vec<f32> {
        self.w
            .svc
            .trips(&self.w.net, epoch, epoch + 3600.0)
            .iter()
            .flat_map(|t| {
                let next = t.next.map_or(f32::NAN, |n| (n - epoch) as f32);
                [((t.line * 2 + t.run as u32) * 3 + t.level as u32) as f32, (t.draw - epoch) as f32, (t.end - epoch) as f32, next]
            })
            .collect()
    }
    /// A run's path resampled every `step` m (x, y pairs), for the renderer's line texture.
    pub fn run_polyline(&self, line: u32, run: usize, step: f64) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let len = l.runs[run].len;
        let n = (len / step).ceil() as usize;
        (0..=n)
            .flat_map(|i| {
                let (x, y) = self.w.svc.position(&self.w.net, line, run, (i as f64 * step).min(len));
                [x, y]
            })
            .collect()
    }
    /// The junction inspector (T-031): a junction's edge ends and the moves running lines make
    /// through it. Flat: `[ports, then per port: edge, end (0 a, 1 b), dx, dy (unit vector from the
    /// node to the track 1 km out, or its far end), moves, then per move: in port, out port, trains an hour x3
    /// (high, medium, low), delay per train s x3, utilisation x3, line count, lines..., crossing
    /// count, indices of the moves it crosses...]`. Delay and utilisation are the conflict
    /// resource's (0 for a move that crosses nothing, or at a flying junction).
    pub fn junction_info(&self, node: u32) -> Vec<f64> {
        let n = &self.w.net;
        let svc = &self.w.svc;
        if !n.node_ok(node) {
            return vec![0.0, 0.0];
        }
        let (nx, ny) = (n.node_x[node as usize], n.node_y[node as usize]);
        let mut ports: Vec<(u32, u32)> = vec![];
        let mut out = vec![0.0];
        for e in 0..n.edge_count() as u32 {
            if !n.edge_ok(e) {
                continue;
            }
            for (end, at) in [(0u32, n.edge_a[e as usize]), (1u32, n.edge_b[e as usize])] {
                if at != node {
                    continue;
                }
                let len = n.edge_len[e as usize];
                let s = if end == 0 { len.min(1000.0) } else { (len - 1000.0).max(0.0) };
                let (x, y, _) = geom::pos_at(n.edge_pieces(e), s);
                let d = (x - nx).hypot(y - ny).max(1e-9);
                ports.push((e, end));
                out.extend([e as f64, end as f64, (x - nx) / d, (y - ny) / d]);
            }
        }
        out[0] = ports.len() as f64;
        // moves: (in port, out port) -> line directions through it
        let mut moves: Vec<((usize, usize), Vec<(u32, u8)>)> = vec![];
        for (l, ls) in svc.lines.iter().enumerate() {
            if !ls.running {
                continue;
            }
            for (r, run) in ls.runs.iter().enumerate() {
                for k in 1..run.nodes.len().saturating_sub(1) {
                    if run.nodes[k] != node {
                        continue;
                    }
                    let (p, q) = (run.segs[k - 1], run.segs[k]);
                    let pin = (crate::track::net::path_edge(p), if crate::track::net::path_dir(p) == 0 { 1 } else { 0 });
                    let pout = (crate::track::net::path_edge(q), if crate::track::net::path_dir(q) == 0 { 0 } else { 1 });
                    let (Some(i), Some(o)) = (ports.iter().position(|&x| x == pin), ports.iter().position(|&x| x == pout)) else { continue };
                    match moves.iter_mut().find(|m| m.0 == (i, o)) {
                        Some(m) => m.1.push((l as u32, r as u8)),
                        None => moves.push(((i, o), vec![(l as u32, r as u8)])),
                    }
                }
            }
        }
        // each move's conflict resource: the junction resource it owns
        let cap = &svc.cap;
        let res_of = |m: &Vec<(u32, u8)>| {
            (0..cap.res.len()).find(|&r| {
                cap.res[r].kind == crate::track::capacity::ResKind::Junction && cap.res[r].node == node && cap.users_of(r).iter().any(|u| u.owner && m.contains(&(u.line, u.run)))
            })
        };
        out.push(moves.len() as f64);
        for (k, ((i, o), users)) in moves.iter().enumerate() {
            let mut tph = [0.0f64; 3];
            let mut lines: Vec<u32> = vec![];
            for &(l, _) in users {
                for (lev, t) in tph.iter_mut().enumerate() {
                    *t += n.line_tph[l as usize][lev] as f64;
                }
                if !lines.contains(&l) {
                    lines.push(l);
                }
            }
            let r = res_of(users);
            out.extend([*i as f64, *o as f64]);
            out.extend(tph);
            out.extend(r.map_or([0.0; 3], |r| cap.delay[r]));
            out.extend(r.map_or([0.0; 3], |r| cap.rho[r]));
            out.push(lines.len() as f64);
            out.extend(lines.iter().map(|&l| l as f64));
            let crossing: Vec<usize> = match r {
                Some(r) => (0..moves.len())
                    .filter(|&j| j != k && cap.users_of(r).iter().any(|u| !u.owner && moves[j].1.contains(&(u.line, u.run))))
                    .collect(),
                None => vec![],
            };
            out.push(crossing.len() as f64);
            out.extend(crossing.iter().map(|&j| j as f64));
        }
        out
    }

    /// Where a line loses time to capacity at a demand level (T-031): per resource it waits at,
    /// largest first, `[kind (ResKind as u8), node (or -1), x, y, delay per round trip s]`; delay
    /// counts each of its directions that waits there.
    pub fn line_delays(&self, line: u32, level: usize) -> Vec<f64> {
        let cap = &self.w.svc.cap;
        let lev = level.min(LEVELS - 1);
        let mut v: Vec<(usize, f64)> = vec![];
        for r in 0..cap.res.len() {
            let d = cap.delay[r][lev];
            if d <= 0.0 {
                continue;
            }
            let k = cap.users_of(r).iter().filter(|u| u.owner && u.line == line).count();
            if k > 0 {
                v.push((r, d * k as f64));
            }
        }
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.iter()
            .flat_map(|&(r, d)| {
                let x = &cap.res[r];
                [cap.wire_kind(r) as f64, if x.node == u32::MAX { -1.0 } else { x.node as f64 }, x.x, x.y, d]
            })
            .collect()
    }


    /// Capacity markers for a level (SPEC 6.2 shows 75% and up): x, y, utilisation, delay s, kind,
    /// trains an hour through it, and 1 when it is full and holds back some line's trains (T-517).
    pub fn markers(&self, level: usize) -> Vec<f32> {
        let (cap, net) = (&self.w.svc.cap, &self.w.net);
        self.w
            .busy(level)
            .iter()
            .flat_map(|&(r, _, rho, d)| {
                let res = &cap.res[r];
                let held = rho >= FULL_RHO && cap.limits(r) && cap.users_of(r).iter().any(|u| self.waiting(u.line, level) > 0.0);
                [res.x as f32, res.y as f32, rho as f32, d as f32, cap.wire_kind(r) as f32, cap.trains(net, r, level) as f32, held as u8 as f32]
            })
            .collect()
    }

    /// Track at capacity at a level (T-819, `FULL_RHO`): per edge of a full section or flat
    /// crossing (`Capacity::track_edges`), `[edge, utilisation, trains an hour through it]`, the
    /// fullest resource where several are.
    pub fn full_track(&self, level: usize) -> Vec<f64> {
        let (cap, net) = (&self.w.svc.cap, &self.w.net);
        let mut by: std::collections::HashMap<u32, (f64, f64)> = std::collections::HashMap::new();
        for r in 0..cap.res.len() {
            let rho = cap.rho[r][level];
            if rho < FULL_RHO {
                continue;
            }
            let edges = cap.track_edges(r);
            if edges.is_empty() {
                continue;
            }
            let tph = cap.trains(net, r, level);
            for k in edges {
                let e = by.entry(k).or_insert((0.0, 0.0));
                if rho > e.0 {
                    *e = (rho, tph);
                }
            }
        }
        let mut v: Vec<(u32, (f64, f64))> = by.into_iter().collect();
        v.sort_unstable_by_key(|x| x.0);
        v.into_iter().flat_map(|(e, (rho, tph))| [e as f64, rho, tph]).collect()
    }

    /// Trains of a whole-line fleet that full track keeps from running at a level (T-517), or 0.
    pub(super) fn waiting(&self, line: u32, level: usize) -> f64 {
        let (n, svc) = (&self.w.net, &self.w.svc);
        let (Some(Some(counts)), Some(ls)) = (n.line_trains.get(line as usize), svc.lines.get(line as usize)) else { return 0.0 };
        let short = counts[level] as f64 - n.line_tph[line as usize][level] as f64 * ls.round_trip[level] / 3600.0;
        if short > 0.01 { short } else { 0.0 }
    }

    /// Per level, where full track holds a line back (T-517): `[trains that wait, kind (ResKind
    /// as u8), node or -1, x, y, trains an hour through it, the most it takes]`, all zero where the
    /// line's whole fleet runs. Then per level what limits the line (T-679): `[shortest headway s
    /// with every other line as it runs (infinite with no room left), kind, node or -1, x, y]`,
    /// all zero where nothing does.
    pub fn line_held(&self, line: u32) -> Vec<f64> {
        let mut out = vec![0.0; 12 * LEVELS];
        let cap = &self.w.svc.cap;
        if line as usize >= self.w.net.line_count() {
            return out;
        }
        let most = cap.line_limits(&self.w.net)[line as usize];
        for lev in 0..LEVELS {
            let (tph, r) = most[lev];
            if r == u32::MAX {
                continue;
            }
            let res = &cap.res[r as usize];
            out[7 * LEVELS + 5 * lev..7 * LEVELS + 5 * lev + 5].copy_from_slice(&[if tph > 0.0 { 3600.0 / tph } else { f64::INFINITY },cap.wire_kind(r as usize) as f64, if res.node == u32::MAX { -1.0 } else { res.node as f64 }, res.x, res.y]);
        }
        let worst = cap.bottlenecks(self.w.net.line_count())[line as usize];
        for lev in 0..LEVELS {
            let (rho, r) = worst[lev];
            let wait = self.waiting(line, lev);
            if rho < FULL_RHO || r == u32::MAX || wait <= 0.0 {
                continue;
            }
            let res = &cap.res[r as usize];
            let tph = cap.trains(&self.w.net, r as usize, lev);
            out[7 * lev..7 * lev + 7].copy_from_slice(&[wait, cap.wire_kind(r as usize) as f64, if res.node == u32::MAX { -1.0 } else { res.node as f64 }, res.x, res.y, tph, tph / rho]);
        }
        out
    }

    /// The flat junctions (no flyover, no station) and flat crossings a line's route passes, in
    /// route order (T-629): per mark `[4 junction | 5 crossing, the stop it follows, how far on
    /// to the next stop 0..1, x, y, number of other lines there, those lines' numbers]`. Other
    /// lines: those through the junction's node, or on the crossing edge.
    pub fn line_marks(&self, line: u32) -> Vec<f64> {
        use crate::track::net::{path_dir, path_edge};
        let (net, svc) = (&self.w.net, &self.w.svc);
        let Some(ls) = svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let run = &ls.runs[0];
        if run.stop_idx.len() < 2 {
            return vec![];
        }
        let stop_s: Vec<f64> = run.stop_idx.iter().map(|&k| run.node_s[k as usize]).collect();
        let place = |s: f64| -> (usize, f64) {
            let g = stop_s.partition_point(|&a| a <= s).saturating_sub(1).min(stop_s.len() - 2);
            (g, ((s - stop_s[g]) / (stop_s[g + 1] - stop_s[g]).max(1e-9)).clamp(0.0, 1.0))
        };
        let others = |uses: &dyn Fn(&crate::track::service::Run) -> bool| -> Vec<f64> {
            (0..svc.lines.len()).filter(|&o| o != line as usize && svc.lines[o].ok && uses(&svc.lines[o].runs[0])).map(|o| o as f64).collect()
        };
        let mut marks: Vec<(f64, [f64; 5], Vec<f64>)> = vec![];
        for k in 1..run.nodes.len().saturating_sub(1) {
            let n = run.nodes[k];
            let i = n as usize;
            if net.node_ports[i].n < 3 || net.node_flying[i] || net.node_platform[i] > 0 {
                continue;
            }
            let s = run.node_s[k];
            if s <= stop_s[0] || s >= stop_s[stop_s.len() - 1] {
                continue;
            }
            let (g, f) = place(s);
            marks.push((s, [4.0, g as f64, f, net.node_x[i], net.node_y[i]], others(&|r| r.nodes.contains(&n))));
        }
        for c in net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))) {
            for (own, s_own, other) in [(c.e1, c.s1, c.e2), (c.e2, c.s2, c.e1)] {
                let Some(j) = run.segs.iter().position(|&p| path_edge(p) == own) else { continue };
                let len = net.edge_len[own as usize];
                let s = run.seg_off[j] + if path_dir(run.segs[j]) == 0 { s_own } else { len - s_own };
                if s <= stop_s[0] || s >= stop_s[stop_s.len() - 1] {
                    continue;
                }
                let (g, f) = place(s);
                marks.push((s, [5.0, g as f64, f, c.x, c.y], others(&|r| r.segs.iter().any(|&p| path_edge(p) == other))));
            }
        }
        marks.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = vec![];
        for (_, head, lines) in marks {
            out.extend_from_slice(&head);
            out.push(lines.len() as f64);
            out.extend(lines);
        }
        out
    }
}
