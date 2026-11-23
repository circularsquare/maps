//! Service: stop times, profiles, trips, junctions, delays and markers.

use super::*;
use crate::track::timing::PROFILE_VARS;

#[wasm_bindgen]
impl TrackApi {
    // ---- service
    /// Time of each stop from the start of the line (SPEC 6.3), seconds: arrival times, at row
    /// `level` (high, medium, low).
    pub fn stop_times(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        let lev = self.w.net.row_slot(line as usize, level.min(ROWS - 1));
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.cumulative(run, lev).to_vec())
    }
    /// Departure time from each stop, seconds from the start of the trip, at row `level`.
    pub fn stop_departures(&self, line: u32, run: usize, level: usize) -> Vec<f64> {
        let lev = self.w.net.row_slot(line as usize, level.min(ROWS - 1));
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].prof[lev].dep.clone())
    }
    /// T-3542: `stop_times` in demand period `period` (0-4, `PERIOD_HOURS`): its slot's.
    pub fn stop_times_in(&self, line: u32, run: usize, period: usize) -> Vec<f64> {
        let lev = self.w.net.period_slot()[period.min(4)];
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.cumulative(run, lev).to_vec())
    }
    /// T-3542: `stop_departures` in demand period `period`.
    pub fn stop_departures_in(&self, line: u32, run: usize, period: usize) -> Vec<f64> {
        let lev = self.w.net.period_slot()[period.min(4)];
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].prof[lev].dep.clone())
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
        for l in 0..self.w.svc.lines.len() {
            self.push_phases(l, &mut out);
        }
        out
    }
    /// One line's part of `phases`, for the clock worker to read only the lines an edit changed
    /// (T-1540): its tables laid end to end in the same order, empty when it is not ok.
    pub fn line_phases(&self, line: u32) -> Vec<f32> {
        let mut out = vec![];
        self.push_phases(line as usize, &mut out);
        out
    }
    /// T-3542: profiles a run has (`params::SLOTS`), the stride of a trip's profile index; the app's
    /// copy is map/profiles.ts `SLOTS`.
    pub fn profile_slots(&self) -> u32 {
        SLOTS as u32
    }
    /// T-3346 stage 2: profiles per (line, run, slot), the slot's and its train-by-train variants:
    /// the innermost stride of a trip's profile index; the app's copy is map/profiles.ts `PROFILE_VARS`.
    pub fn profile_vars(&self) -> u32 {
        PROFILE_VARS as u32
    }
    /// Per profile `((line * 2 + run) * SLOTS + slot) * PROFILE_VARS + var`: first phase, phase
    /// count, trip seconds (run plus turnaround), path length m. Absent lines, slots not worked out
    /// and variants a run has not have count 0.
    pub fn profile_meta(&self) -> Vec<f32> {
        let mut out = vec![];
        let mut first = 0usize;
        for (li, l) in self.w.svc.lines.iter().enumerate() {
            for r in 0..2 {
                for k in 0..SLOTS {
                    for v in 0..PROFILE_VARS {
                        if l.ok && self.w.net.slot_worked(k) && (v == 0 || v <= l.runs[r].vars[k].len()) {
                            let n = l.prof_of(r, k, v as u8).phases.len();
                            out.extend([first as f32, n as f32, l.trip_len_var(&self.w.net, li as u32, r, k, v as u8) as f32, l.runs[r].len as f32]);
                            first += n;
                        } else {
                            out.extend([first as f32, 0.0, 0.0, 0.0]);
                        }
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
                [profile_index(t) as f32, (t.draw - epoch) as f32, (t.end - epoch) as f32, next]
            })
            .collect()
    }
    /// T-2603: line `line`'s trips drawn in `[t0, t1)`, game seconds, in absolute time: per trip the
    /// profile index, departure, end and next departure as `trips` gives them (NaN for none).
    pub fn line_trips(&self, line: u32, t0: f64, t1: f64) -> Vec<f64> {
        self.w
            .svc
            .line_trips(&self.w.net, line as usize, t0, t1)
            .iter()
            .flat_map(|t| [profile_index(t) as f64, t.draw, t.end, t.next.unwrap_or(f64::NAN)])
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
    /// T-3346: the departures board. Line `line`'s trains leaving station node `node` in
    /// `[t0, t1)`, game seconds: per departure its run, the stop's index in that run and the time
    /// it leaves (its trip's `draw` plus its profile's `dep` there), sorted by time. A run's last
    /// stop is never a departure; a stop the run passes twice is two.
    pub fn station_departures(&self, line: u32, node: u32, t0: f64, t1: f64) -> Vec<f64> {
        let svc = &self.w.svc;
        let Some(ls) = svc.lines.get(line as usize).filter(|l| l.ok && !l.is_bus()) else { return vec![] };
        // the one place this reads a trip's timetable: its own (T-3346 stage 2)
        let prof = |t: &crate::track::service::Trip| ls.prof_of(t.run as usize, t.level as usize, t.var);
        let mut out: Vec<(f64, u8, usize)> = vec![];
        for t in svc.line_trips(&self.w.net, line as usize, t0, t1) {
            let p = prof(&t);
            let n = ls.runs[t.run as usize].stop_idx.len();
            for k in 0..n.saturating_sub(1) {
                // T-3565: nor where the run passes a one-way station on the other track
                if ls.stop_node(t.run as usize, k) != Some(node) || ls.runs[t.run as usize].away.iter().any(|x| x.0 as usize == k) {
                    continue;
                }
                let at = t.draw + p.dep.get(k).copied().unwrap_or(0.0);
                if at >= t0 && at < t1 {
                    out.push((at, t.run, k));
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out.iter().flat_map(|&(at, r, k)| [r as f64, k as f64, at]).collect()
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
            for (r, run) in ls.runs.iter().enumerate().take(ls.live_runs()) {
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
            let mut tph = [0.0f64; ROWS];
            let mut lines: Vec<u32> = vec![];
            for &(l, _) in users {
                for (row, t) in tph.iter_mut().enumerate() {
                    *t += n.line_tph[l as usize][n.row_slot(l as usize, row)] as f64;
                }
                if !lines.contains(&l) {
                    lines.push(l);
                }
            }
            let r = res_of(users);
            out.extend([*i as f64, *o as f64]);
            out.extend(tph);
            out.extend(r.map_or([0.0; ROWS], |r| slots_to_rows(cap.delay[r])));
            out.extend(r.map_or([0.0; ROWS], |r| slots_to_rows(cap.rho[r])));
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
    /// counts each of its directions that waits there. Behind slower trains (kind 11, T-1924) the
    /// second number is the slower line it follows (or -1), and the delay its one direction's loss
    /// on that stretch.
    pub fn line_delays(&self, line: u32, level: usize) -> Vec<f64> {
        self.line_delays_at(line, self.w.net.row_slot(line as usize, level.min(ROWS - 1)))
    }
    /// T-3542: `line_delays` in demand period `period` (0-4), at the slot serving it.
    pub fn line_delays_in(&self, line: u32, period: usize) -> Vec<f64> {
        self.line_delays_at(line, self.w.net.period_slot()[period.min(4)])
    }
    fn line_delays_at(&self, line: u32, lev: usize) -> Vec<f64> {
        let cap = &self.w.svc.cap;
        let mut v: Vec<(usize, f64)> = vec![];
        for r in 0..cap.res.len() {
            let d = cap.delay[r][lev];
            if d <= 0.0 {
                continue;
            }
            let k = cap.users_of(r).iter().filter(|u| u.owner && u.line == line).count();
            if k > 0 {
                // a following loss is one direction's, however many places it waits at
                v.push((r, if cap.wire_kind(r) == 11 { d } else { d * k as f64 }));
            }
        }
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.iter()
            .flat_map(|&(r, d)| {
                let x = &cap.res[r];
                let node = match cap.follow_leader(&self.w.net, r, lev) {
                    Some(l) => l as f64,
                    None if cap.wire_kind(r) == 11 || x.node == u32::MAX => -1.0,
                    None => x.node as f64,
                };
                [cap.wire_kind(r) as f64, node, x.x, x.y, d]
            })
            .collect()
    }


    /// Capacity markers for a level (SPEC 6.2 shows 75% and up): x, y, utilisation, delay s, kind,
    /// trains an hour through it, and 1 when it is full and holds back some line's trains (T-517).
    pub fn markers(&self, level: usize) -> Vec<f32> {
        // `level` is a row (high, medium, low); every line of three rows keeps it in one slot
        self.markers_at(ROW_SLOT[level.min(ROWS - 1)])
    }
    fn markers_at(&self, level: usize) -> Vec<f32> {
        let (cap, net) = (&self.w.svc.cap, &self.w.net);
        self.w
            .busy(level)
            .iter()
            .flat_map(|&(r, _, rho, d)| {
                let res = &cap.res[r];
                // T-2935: a whole train, as the line panel counts them; a part train still runs some of the time
                let held = rho >= FULL_RHO && cap.limits(r) && cap.users_of(r).iter().any(|u| cap.caps(r, u) && self.waiting(u.line, level) >= 0.99);
                [res.x as f32, res.y as f32, rho as f32, d as f32, cap.wire_kind(r) as f32, cap.trains(net, r, level) as f32, held as u8 as f32]
            })
            .collect()
    }

    /// Track at capacity at a level (T-819, `FULL_RHO`): per edge of a full section or flat
    /// crossing (`Capacity::track_edges`), `[edge, utilisation, trains an hour through it]`, the
    /// fullest resource where several are.
    pub fn full_track(&self, level: usize) -> Vec<f64> {
        self.full_track_at(ROW_SLOT[level.min(ROWS - 1)])
    }
    fn full_track_at(&self, level: usize) -> Vec<f64> {
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
        let (Some(Some(_)), Some(ls)) = (n.line_trains.get(line as usize), svc.lines.get(line as usize)) else { return 0.0 };
        let short = n.running_trains(line as usize, level, ls.round_trip[level]) - n.line_tph[line as usize][level] as f64 * ls.round_trip[level] / 3600.0;
        if short > 0.01 { short } else { 0.0 }
    }

    /// What limits a line at slot `lev` (`Capacity::line_limits`'s resource `r`): wire kind, node or -1, x, y; a full
    /// BRT lane (T-3378, `LANE_RES`) by its fullest way's middle.
    fn limit_place(&self, line: u32, lev: usize, r: u32) -> (f64, f64, f64, f64) {
        let cap = &self.w.svc.cap;
        if r == crate::track::capacity::LANE_RES {
            let (_, x, y) = self.lane_load(line, lev).unwrap_or((0.0, 0.0, 0.0));
            return (crate::track::capacity::LANE_KIND as f64, -1.0, x, y);
        }
        let res = &cap.res[r as usize];
        (cap.wire_kind(r as usize) as f64, if res.node == u32::MAX { -1.0 } else { res.node as f64 }, res.x, res.y)
    }

    /// T-3378: the fullest way of a BRT lane line `line` runs on at slot `lev`: buses an hour through it, x, y.
    fn lane_load(&self, line: u32, lev: usize) -> Option<(f64, f64, f64)> {
        let n = &self.w.net;
        n.brt_room().fullest(line, |l| n.line_tph.get(l as usize).map_or(0.0, |t| t[lev] as f64))
    }

    /// T-3378: `line_held`'s seven numbers where a full BRT lane keeps some of a bus line's buses from running at slot
    /// `lev`: [buses that wait, `LANE_KIND`, -1, x, y, buses an hour through it, the most it takes].
    fn lane_held(&self, line: u32, lev: usize) -> Option<[f64; 7]> {
        let wait = self.waiting(line, lev);
        let (load, x, y) = self.lane_load(line, lev)?;
        let most = crate::track::bus::lanes::LANE_BUSES_H;
        (wait > 0.0 && load >= most * FULL_RHO).then_some([wait, crate::track::capacity::LANE_KIND as f64, -1.0, x, y, load, most])
    }

    /// T-3378: per demand period (0-4), the buses an hour one way on the fullest stretch of BRT lane a line runs on (0
    /// where it has none), then `LANE_BUSES_H`.
    pub fn bus_lane_load(&self, line: u32) -> Vec<f64> {
        let ps = self.w.net.period_slot();
        let mut out: Vec<f64> = ps.iter().map(|&lev| self.lane_load(line, lev).map_or(0.0, |x| x.0)).collect();
        out.push(crate::track::bus::lanes::LANE_BUSES_H);
        out
    }

    /// Per level, where full track holds a line back (T-517): `[trains that wait, kind (ResKind
    /// as u8), node or -1, x, y, trains an hour through it, the most it takes]`, all zero where the
    /// line's whole fleet runs. Then per level what limits the line (T-679): `[shortest headway s
    /// with every other line as it runs (infinite with no room left), kind, node or -1, x, y]`,
    /// all zero where nothing does. Then per level the shortest headway a train count is capped by
    /// (T-3526): the same, but with single track's margin counted for a lone train too; 0 where
    /// nothing limits.
    pub fn line_held(&self, line: u32) -> Vec<f64> {
        let mut out = vec![0.0; 13 * ROWS];
        let cap = &self.w.svc.cap;
        if line as usize >= self.w.net.line_count() {
            return out;
        }
        let most = cap.line_limits(&self.w.net)[line as usize];
        // the lone case only differs for a line with one train or none
        let lone = self.w.net.line_trains[line as usize].map_or(true, |t| t.iter().any(|&n| n <= 1));
        let paired = if lone { cap.line_limits_lone(&self.w.net, false)[line as usize] } else { most };
        // per row (high, medium, low), each from its slot
        let slot = |row: usize| self.w.net.row_slot(line as usize, row);
        for row in 0..ROWS {
            let tph = paired[slot(row)].0;
            if paired[slot(row)].1 != u32::MAX {
                out[12 * ROWS + row] = if tph > 0.0 { 3600.0 / tph } else { f64::INFINITY };
            }
        }
        for row in 0..ROWS {
            let (tph, r) = most[slot(row)];
            if r == u32::MAX {
                continue;
            }
            let (kind, node, x, y) = self.limit_place(line, slot(row), r);
            out[7 * ROWS + 5 * row..7 * ROWS + 5 * row + 5].copy_from_slice(&[if tph > 0.0 { 3600.0 / tph } else { f64::INFINITY }, kind, node, x, y]);
        }
        let worst = cap.bottlenecks(self.w.net.line_count())[line as usize];
        for row in 0..ROWS {
            let lev = slot(row);
            let (rho, r) = worst[lev];
            let wait = self.waiting(line, lev);
            if let Some(h) = self.lane_held(line, lev).filter(|_| rho < FULL_RHO || r == u32::MAX) {
                out[7 * row..7 * row + 7].copy_from_slice(&h);
                continue;
            }
            if rho < FULL_RHO || r == u32::MAX || wait <= 0.0 {
                continue;
            }
            let res = &cap.res[r as usize];
            let tph = cap.trains(&self.w.net, r as usize, lev);
            out[7 * row..7 * row + 7].copy_from_slice(&[wait, cap.wire_kind(r as usize) as f64, if res.node == u32::MAX { -1.0 } else { res.node as f64 }, res.x, res.y, tph, tph / rho]);
        }
        out
    }

    /// T-3542: `line_held` per row of the line in the day's order (`line_period_count` rows): `7 *
    /// n` held numbers, `5 * n` limit numbers, `n` capped headways, laid out as `line_held`'s. A row
    /// over several slots takes the slot that holds the most trains back, and the tightest limit.
    pub fn line_row_held(&self, line: u32) -> Vec<f64> {
        let net = &self.w.net;
        if !net.line_ok(line) { return vec![] }
        let rows = net.row_slots(line as usize);
        let n = rows.len();
        let mut out = vec![0.0; 13 * n];
        let cap = &self.w.svc.cap;
        let most = cap.line_limits(net)[line as usize];
        let lone = net.line_trains[line as usize].map_or(true, |t| t.iter().any(|&n| n <= 1));
        let paired = if lone { cap.line_limits_lone(net, false)[line as usize] } else { most };
        let worst = cap.bottlenecks(net.line_count())[line as usize];
        // a limit's shortest headway, longest first: infinite for a full stretch
        let limit_s = |tph: f64| if tph > 0.0 { 3600.0 / tph } else { f64::INFINITY };
        for (row, (_, slots)) in rows.iter().enumerate() {
            let tight = slots.iter().copied().filter(|&s| paired[s].1 != u32::MAX).map(|s| limit_s(paired[s].0)).fold(0.0, f64::max);
            out[12 * n + row] = tight;
            if let Some(s) = slots.iter().copied().filter(|&s| most[s].1 != u32::MAX).max_by(|&a, &b| limit_s(most[a].0).total_cmp(&limit_s(most[b].0))) {
                let (tph, r) = most[s];
                let (kind, node, x, y) = self.limit_place(line, s, r);
                out[7 * n + 5 * row..7 * n + 5 * row + 5].copy_from_slice(&[limit_s(tph), kind, node, x, y]);
            }
            let held = slots.iter().copied().map(|s| (s, self.waiting(line, s))).filter(|&(s, w)| worst[s].0 >= FULL_RHO && worst[s].1 != u32::MAX && w > 0.0).max_by(|a, b| a.1.total_cmp(&b.1));
            // T-3378: else a full BRT lane, at the slot holding the most buses back
            let lane = slots.iter().copied().filter_map(|s| self.lane_held(line, s)).max_by(|a, b| a[0].total_cmp(&b[0]));
            if let (None, Some(h)) = (held, lane) {
                out[7 * row..7 * row + 7].copy_from_slice(&h);
            }
            if let Some((lev, wait)) = held {
                let (rho, r) = worst[lev];
                let res = &cap.res[r as usize];
                let tph = cap.trains(net, r as usize, lev);
                out[7 * row..7 * row + 7].copy_from_slice(&[wait, cap.wire_kind(r as usize) as f64, if res.node == u32::MAX { -1.0 } else { res.node as f64 }, res.x, res.y, tph, tph / rho]);
            }
        }
        out
    }

    /// T-3542: `markers` in demand period `period` (0-4): at the slot serving it.
    pub fn markers_in(&self, period: usize) -> Vec<f32> {
        self.markers_at(self.w.net.period_slot()[period.min(4)])
    }
    /// T-3542: `full_track` in demand period `period` (0-4).
    pub fn full_track_in(&self, period: usize) -> Vec<f64> {
        self.full_track_at(self.w.net.period_slot()[period.min(4)])
    }

    /// T-1966: the conflicts the map's train guard holds trains at: `[flat crossings, per crossing:
    /// edge 1, where it is along edge 1 from its node a (0..1), edge 2, along edge 2; junction
    /// pairs, per pair: node, one move's arriving and leaving path entries, the other's]`.
    pub fn guard_conflicts(&self) -> Vec<f64> {
        let net = &self.w.net;
        let mut out = vec![0.0];
        for c in net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))) {
            let at = |e: u32, s: f64| (s / net.edge_len[e as usize].max(1e-9)).clamp(0.0, 1.0);
            out.extend([c.e1 as f64, at(c.e1, c.s1), c.e2 as f64, at(c.e2, c.s2)]);
            out[0] += 1.0;
        }
        let pairs = crate::track::capacity::move_conflicts(net, &self.w.svc.lines);
        out.push(pairs.len() as f64);
        for p in pairs {
            out.extend(p.iter().map(|&v| v as f64));
        }
        out
    }

    /// T-3202: the two-way single-track stretches between passing points (the capacity's single-track
    /// sections) the map's train guard keeps trains of opposite directions off together: `[count, per
    /// stretch: number of edges, its edges]`.
    pub fn guard_singles(&self) -> Vec<f64> {
        let cap = &self.w.svc.cap;
        let mut out = vec![0.0];
        for r in 0..cap.res.len() {
            if cap.res[r].kind != crate::track::capacity::ResKind::SingleTrack {
                continue;
            }
            let mut es = cap.track_edges(r);
            es.sort_unstable();
            es.dedup();
            out.push(es.len() as f64);
            out.extend(es.iter().map(|&e| e as f64));
            out[0] += 1.0;
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
        // T-3745: only where something could hold this line's trains: a junction move of its own that
        // crosses another move some line runs, a crossing with a line on its other edge
        let pairs = crate::track::capacity::move_conflicts(net, &svc.lines);
        let used = crate::track::capacity::edges_in_use(&svc.lines);
        // its moves through junctions over every run, so its other direction's crossing counts too
        let mut mine = std::collections::HashSet::new();
        for r in ls.runs.iter().take(ls.live_runs()) {
            for k in 1..r.nodes.len().saturating_sub(1) {
                mine.insert((r.nodes[k], r.segs[k - 1], r.segs[k]));
            }
        }
        let mut marks: Vec<(f64, [f64; 5], Vec<f64>)> = vec![];
        // a looped route passes a junction or crossing twice: drawn once, at its first pass
        let mut seen_nodes = std::collections::HashSet::new();
        for k in 1..run.nodes.len().saturating_sub(1) {
            let n = run.nodes[k];
            let i = n as usize;
            if net.node_ports[i].n < 3 || net.node_flying[i] || net.node_platform[i] > 0 {
                continue;
            }
            let s = run.node_s[k];
            if !pairs.iter().any(|c| c[0] == n && [[c[1], c[2]], [c[3], c[4]]].iter().any(|m| mine.contains(&(n, m[0], m[1])))) {
                continue;
            }
            if s <= stop_s[0] || s >= stop_s[stop_s.len() - 1] || !seen_nodes.insert(n) {
                continue;
            }
            let (g, f) = place(s);
            marks.push((s, [4.0, g as f64, f, net.node_x[i], net.node_y[i]], others(&|r| r.nodes.contains(&n))));
        }
        for c in net.crossings.iter().filter(|c| matches!(c.kind, CrossKind::Flat(_))) {
            let mut best: Option<(f64, Vec<f64>)> = None;
            for (own, s_own, other) in [(c.e1, c.s1, c.e2), (c.e2, c.s2, c.e1)] {
                let Some(j) = run.segs.iter().position(|&p| path_edge(p) == own) else { continue };
                if used.binary_search(&other).is_err() {
                    continue;
                }
                let len = net.edge_len[own as usize];
                let s = run.seg_off[j] + if path_dir(run.segs[j]) == 0 { s_own } else { len - s_own };
                if s <= stop_s[0] || s >= stop_s[stop_s.len() - 1] || best.as_ref().is_some_and(|b| b.0 <= s) {
                    continue;
                }
                best = Some((s, others(&|r| r.segs.iter().any(|&p| path_edge(p) == other))));
            }
            if let Some((s, lines)) = best {
                let (g, f) = place(s);
                marks.push((s, [5.0, g as f64, f, c.x, c.y], lines));
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

impl TrackApi {
    fn push_phases(&self, line: usize, out: &mut Vec<f32>) {
        let Some(l) = self.w.svc.lines.get(line).filter(|l| l.ok) else { return };
        for r in 0..2 {
            for k in 0..SLOTS {
                out.extend(l.runs[r].prof[k].phases.iter().flat_map(|p| [p.t as f32, p.s as f32, p.v as f32, p.a as f32]));
                // T-3346 stage 2: its variants after it, as `profile_meta` counts them
                if self.w.net.slot_worked(k) {
                    for x in l.runs[r].vars[k].iter().take(PROFILE_VARS - 1) {
                        out.extend(x.prof.phases.iter().flat_map(|p| [p.t as f32, p.s as f32, p.v as f32, p.a as f32]));
                    }
                }
            }
        }
    }
}

/// A trip's place in `profile_meta` (T-3346 stage 2: its variant's).
fn profile_index(t: &crate::track::service::Trip) -> u32 {
    ((t.line * 2 + t.run as u32) * SLOTS as u32 + t.level as u32) * PROFILE_VARS as u32 + t.var as u32
}

/// T-4116: what bin/edit_fuzz checks after every edit (native builds only, not in the WASM interface).
#[cfg(not(target_arch = "wasm32"))]
impl TrackApi {
    /// Per running line whose live runs name a derived edge on no live piece: the line, the run and the entry's index.
    pub fn stale_runs(&self) -> Vec<(u32, usize, usize)> {
        let n = &self.w.net;
        let mut out = vec![];
        for (l, ls) in self.w.svc.lines.iter().enumerate() {
            if !ls.ok || ls.is_bus() {
                continue;
            }
            for (r, run) in ls.runs.iter().take(ls.live_runs()).enumerate() {
                if let Some(k) = run.segs.iter().position(|x| !n.edge_ok(*x >> 1)) {
                    out.push((l as u32, r, k));
                }
            }
        }
        out
    }

    /// Per line the network has (not broken) whose path names a track or part its piece does not have: the line, 9 and
    /// the entry's index (a colour change re-checks the path and is refused then).
    pub fn bad_paths(&self) -> Vec<(u32, usize, usize)> {
        let n = &self.w.net;
        (0..n.line_count() as u32)
            .filter(|&l| n.line_ok(l) && !n.line_broken[l as usize] && !n.is_bus(l) && !n.is_ferry(l))
            .filter_map(|l| n.line_path(l).iter().position(|&p| !n.edge_ok(p >> 1)).map(|k| (l, 9, k)))
            .collect()
    }
}
