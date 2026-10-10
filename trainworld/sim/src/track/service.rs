//! Lines as run (SPEC 6.3): each line's two directions ("runs") with their path, stops, speed
//! limits and profiles per demand level; stop times, round trip, trains needed; the capacity pass;
//! and the trips the renderer draws.

use super::capacity::{self, Capacity};
use super::geom;
use super::net::{path_dir, path_edge, Network};
use super::params::*;
use super::profile::{self, Hold, Profile};

/// One direction of a line. Offsets are metres along this direction's path, from the first
/// stop's node.
#[derive(Clone, Debug, Default)]
pub struct Run {
    /// Path entries in travel order, `edge << 1 | dir`.
    pub segs: Vec<u32>,
    /// Offset where each entry starts.
    pub seg_off: Vec<f64>,
    /// Nodes passed, `segs.len() + 1`, and their offsets.
    pub nodes: Vec<u32>,
    pub node_s: Vec<f64>,
    pub len: f64,
    /// Per stop: index into `nodes` and where the train's middle stands.
    pub stop_idx: Vec<u32>,
    pub stop_s: Vec<f64>,
    /// Speed-limit breakpoints `[s, v]`, widened by half the train length.
    pub limits: Vec<[f64; 2]>,
    /// No capacity delay.
    pub free: Profile,
    /// Per demand level: track holds and extra dwell per stop from the capacity pass.
    pub holds: [Vec<Hold>; 3],
    pub stop_holds: [Vec<f64>; 3],
    pub prof: [Profile; 3],
}

#[derive(Clone, Debug, Default)]
pub struct LineSvc {
    /// Alive, not broken, profiles built.
    pub ok: bool,
    /// `ok` and every edge and stop constructed: the line carries trains, loads capacity and
    /// has trips (SPEC 6.4). A line that is `ok` but not running shows its times as a preview.
    pub running: bool,
    /// The stops these profiles were built for (a change wakes demand).
    pub stops: Vec<u32>,
    pub runs: [Run; 2],
    pub cars: u32,
    pub train_len: f64,
    /// Per demand level: both directions with dwell, holds and two turnarounds.
    pub round_trip: [f64; 3],
    pub trains: [u32; 3],
    /// Per demand level: capacity delay over the round trip.
    pub delay: [f64; 3],
}

impl LineSvc {
    /// Time of each stop from the start of the line (SPEC 6.3's stops list), for one direction
    /// and demand level: the arrival time, 0 at the first stop.
    pub fn cumulative(&self, run: usize, level: usize) -> &[f64] {
        &self.runs[run].prof[level].arr
    }
    /// Running time from the previous stop's departure to each stop (0 at the first).
    pub fn stop_to_stop(&self, run: usize, level: usize) -> Vec<f64> {
        let p = &self.runs[run].prof[level];
        (0..p.arr.len()).map(|i| if i == 0 { 0.0 } else { p.arr[i] - p.dep[i - 1] }).collect()
    }
    /// Seconds a trip of this direction is drawn: run, then standing for the turnaround.
    pub fn trip_len(&self, net: &Network, line: u32, run: usize, level: usize) -> f64 {
        self.runs[run].prof[level].duration + net.line_turn[line as usize] as f64
    }
}

/// A departure the renderer draws: `profile(line, run, level)` from time `dep`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trip {
    pub line: u32,
    pub run: u8,
    pub level: u8,
    pub dep: f64,
}

/// What `update` recomputed.
#[derive(Clone, Debug, Default)]
pub struct ServiceDirty {
    /// Lines whose path, stops or geometry were rebuilt.
    pub rebuilt: Vec<u32>,
    /// Lines whose profiles were recomputed (rebuilt ones plus those whose holds moved).
    pub recomputed: Vec<u32>,
    /// Lines sharing a resource with a rebuilt or rescheduled line (SPEC 6.5's one step).
    pub shared: Vec<u32>,
    /// Lines whose service changed for demand: stops, frequency, or a stop-to-stop time moving
    /// 5 s or more.
    pub demand: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Service {
    pub lines: Vec<LineSvc>,
    pub cap: Capacity,
}

fn build_run(net: &Network, l: u32, rev: bool, train_len: f64, dwell: f64) -> Option<Run> {
    let path = net.line_path(l);
    let mut run = Run::default();
    run.segs = if rev { path.iter().rev().map(|p| p ^ 1).collect() } else { path.to_vec() };
    let mut s = 0.0;
    run.nodes.push(net.path_nodes(run.segs[0]).0);
    run.node_s.push(0.0);
    let mut raw: Vec<[f64; 2]> = vec![];
    for &p in &run.segs {
        let e = path_edge(p);
        let len = net.edge_len[e as usize];
        run.seg_off.push(s);
        let pcs = net.edge_pieces(e);
        let mut add = |a: f64, v: f64| match raw.last() {
            Some(l) if l[1] == v => {}
            _ => raw.push([a, v]),
        };
        if path_dir(p) == 0 {
            for pc in pcs {
                add(s + pc.s0, pc.v_limit());
            }
        } else {
            for pc in pcs.iter().rev() {
                add(s + len - (pc.s0 + pc.len), pc.v_limit());
            }
        }
        s += len;
        run.nodes.push(net.path_nodes(p).1);
        run.node_s.push(s);
    }
    run.len = s;
    if let Some(f) = raw.first_mut() {
        f[0] = 0.0;
    }
    let mut stops: Vec<u32> = net.line_stops(l).to_vec();
    if rev {
        stops.reverse();
    }
    let mut k = 0;
    for (j, &st) in stops.iter().enumerate() {
        while k < run.nodes.len() && run.nodes[k] != st {
            k += 1;
        }
        if k == run.nodes.len() {
            return None;
        }
        let mut at = run.node_s[k];
        // At a dead end the platform lies along the track: stop at its middle.
        if net.node_ports[st as usize].n == 1 {
            let half = net.node_platform[st as usize] as f64 / 2.0;
            at += if j == 0 { half } else { -half };
        }
        run.stop_idx.push(k as u32);
        run.stop_s.push(at);
        k += 1;
    }
    run.limits = profile::widen(&raw, run.len, train_len / 2.0);
    let n = run.stop_s.len();
    let dw: Vec<f64> = (0..n).map(|i| if i > 0 && i + 1 < n { dwell } else { 0.0 }).collect();
    run.free = profile::run(&run.limits, &run.stop_s, &dw, &[]);
    Some(run)
}

fn build_line(net: &Network, l: u32) -> LineSvc {
    let mut ls = LineSvc::default();
    if !net.line_ok(l) || net.line_broken[l as usize] {
        return ls;
    }
    ls.stops = net.line_stops(l).to_vec();
    let min_plat = net.line_stops(l).iter().map(|&s| net.node_platform[s as usize]).min().unwrap_or(0) as f64;
    let fit = (min_plat / CAR_LEN).floor().max(1.0) as u32;
    let want = net.line_cars[l as usize] as u32;
    ls.cars = if want == 0 { fit } else { want.min(fit) };
    ls.train_len = ls.cars as f64 * CAR_LEN;
    let dwell = net.line_dwell[l as usize] as f64;
    match (build_run(net, l, false, ls.train_len, dwell), build_run(net, l, true, ls.train_len, dwell)) {
        (Some(a), Some(b)) => {
            ls.runs = [a, b];
            ls.ok = true;
            ls.running = net.line_constructed(l);
        }
        _ => {}
    }
    ls
}

fn profile_for(net: &Network, l: u32, run: &Run, level: usize) -> Profile {
    let n = run.stop_s.len();
    let dwell = net.line_dwell[l as usize] as f64;
    let dw: Vec<f64> = (0..n)
        .map(|i| if i > 0 && i + 1 < n { dwell } else { 0.0 } + run.stop_holds[level].get(i).copied().unwrap_or(0.0))
        .collect();
    let mut p = profile::run(&run.limits, &run.stop_s, &dw, &run.holds[level]);
    p.hold += run.stop_holds[level].iter().sum::<f64>();
    p
}

impl Service {
    /// Bring everything up to date. `rebuilt` are lines whose path, stops, platforms or track
    /// geometry may have changed; `rescheduled` changed trains per hour. Stop changes are found
    /// by comparing with the stops the old profiles were built for.
    /// `capacity` says whether anything else the capacity pass reads changed (a crossing, a
    /// junction's ports or flying flag); without it and with no line changes, nothing runs.
    pub fn update(&mut self, net: &Network, rebuilt: &[u32], rescheduled: &[u32], capacity: bool) -> ServiceDirty {
        self.update_scoped(net, rebuilt, rescheduled, capacity, None)
    }

    /// `update`, with the capacity pass rebuilt only around `touched` (the edges and nodes the
    /// edit touched, T-050) instead of from scratch. `None` runs the global pass.
    pub fn update_scoped(&mut self, net: &Network, rebuilt: &[u32], rescheduled: &[u32], capacity: bool, touched: Option<(&[u32], &[u32])>) -> ServiceDirty {
        if !capacity && rebuilt.is_empty() && rescheduled.is_empty() {
            return ServiceDirty::default();
        }
        let sorted = |v: &[u32]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let (rebuilt, rescheduled) = (&sorted(rebuilt)[..], &sorted(rescheduled)[..]);
        let mut d = ServiceDirty { rebuilt: rebuilt.to_vec(), ..Default::default() };
        self.lines.resize(net.line_count(), LineSvc::default());
        let old_times: Vec<(u32, [Vec<Vec<f64>>; 2])> = rebuilt
            .iter()
            .map(|&l| (l, std::array::from_fn(|r| (0..LEVELS).map(|lev| self.lines[l as usize].stop_to_stop_safe(r, lev)).collect())))
            .collect();
        let mut demand: Vec<u32> = rescheduled.to_vec();
        // The incremental capacity pass needs the old and new paths of the rebuilt lines.
        let mut path_edges: Vec<u32> = vec![];
        let mut path_nodes: Vec<u32> = vec![];
        let mut add_path = |ls: &LineSvc| {
            if ls.running {
                path_edges.extend(ls.runs[0].segs.iter().map(|&p| path_edge(p)));
                path_nodes.extend_from_slice(&ls.runs[0].nodes);
            }
        };
        for &l in rebuilt {
            add_path(&self.lines[l as usize]);
        }
        for &l in rebuilt {
            let (old_ok, old_run) = (self.lines[l as usize].ok, self.lines[l as usize].running);
            let old_stops = std::mem::take(&mut self.lines[l as usize].stops);
            self.lines[l as usize] = build_line(net, l);
            let now = &self.lines[l as usize];
            if now.ok != old_ok || now.running != old_run || now.stops != old_stops {
                demand.push(l);
            }
        }
        // Lines whose holds may have changed: all of them after the global pass; after the
        // incremental one, the owners of resources rebuilt or re-solved, plus every rebuilt or
        // rescheduled line.
        let affected: Vec<bool> = match touched {
            None => {
                self.cap = capacity::build(net, &self.lines);
                capacity::solve(&mut self.cap, net);
                vec![true; self.lines.len()]
            }
            Some((edges, nodes)) => {
                for &l in rebuilt {
                    add_path(&self.lines[l as usize]);
                }
                path_edges.extend_from_slice(edges);
                path_nodes.extend_from_slice(nodes);
                path_edges.sort_unstable();
                path_edges.dedup();
                path_nodes.sort_unstable();
                path_nodes.dedup();
                let mut a = self.cap.update(net, &self.lines, &path_edges, &path_nodes, rescheduled);
                for &l in rebuilt.iter().chain(rescheduled) {
                    if let Some(x) = a.get_mut(l as usize) {
                        *x = true;
                    }
                }
                a
            }
        };
        let holds = capacity::holds_for(&self.cap, &self.lines, touched.map(|_| &affected[..]));
        for (l, ls) in self.lines.iter_mut().enumerate() {
            if !ls.ok || !affected[l] {
                continue;
            }
            let forced = rebuilt.binary_search(&(l as u32)).is_ok();
            let before: [Vec<Vec<f64>>; 2] = if forced {
                old_times.iter().find(|o| o.0 == l as u32).map(|o| o.1.clone()).unwrap_or_default()
            } else {
                std::array::from_fn(|r| (0..LEVELS).map(|lev| ls.stop_to_stop_safe(r, lev)).collect())
            };
            let mut changed = false;
            for r in 0..2 {
                for lev in 0..LEVELS {
                    let (h, sh) = &holds[l][r][lev];
                    let run = &mut ls.runs[r];
                    if forced || *h != run.holds[lev] || *sh != run.stop_holds[lev] {
                        run.holds[lev] = h.clone();
                        run.stop_holds[lev] = sh.clone();
                        run.prof[lev] = profile_for(net, l as u32, run, lev);
                        changed = true;
                    }
                }
            }
            let resched = rescheduled.binary_search(&(l as u32)).is_ok();
            if changed || resched {
                for lev in 0..LEVELS {
                    let turn = net.line_turn[l] as f64;
                    let (a, b) = (&ls.runs[0].prof[lev], &ls.runs[1].prof[lev]);
                    ls.round_trip[lev] = a.duration + b.duration + 2.0 * turn;
                    ls.delay[lev] = a.hold + b.hold;
                    let tph = net.line_tph[l][lev] as f64;
                    ls.trains[lev] = if tph > 0.0 { (ls.round_trip[lev] * tph / 3600.0 - 1e-9).ceil() as u32 } else { 0 };
                }
            }
            if changed {
                d.recomputed.push(l as u32);
                let moved = (0..2).any(|r| {
                    (0..LEVELS).any(|lev| {
                        let now = ls.stop_to_stop_safe(r, lev);
                        let was = before[r].get(lev).cloned().unwrap_or_default();
                        now.len() != was.len() || now.iter().zip(&was).any(|(a, b)| (a - b).abs() >= DEMAND_TIME_EPS_S)
                    })
                });
                if moved {
                    demand.push(l as u32);
                }
            }
        }
        // One step: lines sharing a resource with a line that changed.
        let mut seed = vec![false; self.lines.len()];
        for &l in rebuilt.iter().chain(rescheduled) {
            if let Some(x) = seed.get_mut(l as usize) {
                *x = true;
            }
        }
        for r in 0..self.cap.res.len() {
            let us = self.cap.users_of(r);
            if us.iter().any(|u| seed[u.line as usize]) {
                d.shared.extend(us.iter().map(|u| u.line));
            }
        }
        for v in [&mut d.shared, &mut demand] {
            v.sort_unstable();
            v.dedup();
        }
        d.demand = demand;
        d
    }

    /// Every line from scratch.
    pub fn update_all(&mut self, net: &Network) -> ServiceDirty {
        let all: Vec<u32> = (0..net.line_count() as u32).collect();
        self.update(net, &all, &all, true)
    }

    /// Departures overlapping `[t0, t1)` (game seconds), from each period's trains per hour.
    /// Departures repeat daily; both directions leave their first stop on the same clock.
    pub fn trips(&self, net: &Network, t0: f64, t1: f64) -> Vec<Trip> {
        let mut out = vec![];
        for (l, ls) in self.lines.iter().enumerate() {
            if !ls.running {
                continue;
            }
            for r in 0..2 {
                for (p, &(h0, h1)) in PERIOD_HOURS.iter().enumerate() {
                    let lev = PERIOD_LEVEL[p];
                    let tph = net.line_tph[l][lev] as f64;
                    if tph <= 0.0 {
                        continue;
                    }
                    let hw = 3600.0 / tph;
                    let dur = ls.trip_len(net, l as u32, r, lev);
                    let d0 = ((t0 - dur) / 86400.0).floor() as i64;
                    let d1 = (t1 / 86400.0).floor() as i64;
                    for day in d0..=d1 {
                        let (start, pe) = (day as f64 * 86400.0 + h0 * 3600.0, day as f64 * 86400.0 + h1 * 3600.0);
                        // The reverse departure follows the forward trip's turnaround, so an
                        // odd whole-line fleet stays evenly spaced around the complete circuit.
                        let offset = if r == 1 && net.line_trains[l].is_some() { ls.trip_len(net, l as u32, 0, lev) % hw } else { 0.0 };
                        let ps = start + offset;
                        let k0 = ((t0 - dur - ps) / hw).ceil().max(0.0) as i64;
                        let mut k = k0;
                        loop {
                            let dep = ps + k as f64 * hw;
                            if dep >= pe || dep >= t1 {
                                break;
                            }
                            if dep + dur > t0 {
                                out.push(Trip { line: l as u32, run: r as u8, level: lev as u8, dep });
                            }
                            k += 1;
                        }
                    }
                }
            }
        }
        out.sort_by(|a, b| a.dep.total_cmp(&b.dep).then((a.line, a.run).cmp(&(b.line, b.run))));
        out
    }

    /// Where a run's offset `s` is on the track: (edge, direction, offset along the edge from a).
    /// The `(segment, offset)` of SPEC 2.1's keyframes.
    pub fn segment_at(&self, net: &Network, line: u32, run: usize, s: f64) -> (u32, u32, f64) {
        let r = &self.lines[line as usize].runs[run];
        let i = r.seg_off.partition_point(|&o| o <= s).saturating_sub(1).min(r.segs.len() - 1);
        let p = r.segs[i];
        let e = path_edge(p);
        let d = (s - r.seg_off[i]).clamp(0.0, net.edge_len[e as usize]);
        (e, path_dir(p), if path_dir(p) == 0 { d } else { net.edge_len[e as usize] - d })
    }

    /// World position of a run's offset.
    pub fn position(&self, net: &Network, line: u32, run: usize, s: f64) -> (f64, f64) {
        let (e, _, d) = self.segment_at(net, line, run, s);
        let (x, y, _) = geom::pos_at(net.edge_pieces(e), d);
        (x, y)
    }
}

impl LineSvc {
    fn stop_to_stop_safe(&self, r: usize, lev: usize) -> Vec<f64> {
        if !self.ok {
            return vec![];
        }
        self.stop_to_stop(r, lev)
    }
}
