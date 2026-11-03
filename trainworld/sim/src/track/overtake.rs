//! Passing tracks (T-2231, notes/T-2228.md 5): where a slower line stops at a station with passing
//! tracks inside a stretch of track a faster line shares with it (`capacity::stretches`), the
//! timetable has its trains wait there for every faster train that would otherwise catch them before
//! the next such station. Worked out train by train over one cycle of both lines' departures, at the
//! relative phase with the least waiting and loss an hour (`Plan::outcome`): the slower line's mean
//! wait at each passing station becomes a hold at that stop (resource kind 13), the faster line's
//! remaining loss its following loss (kind 11).

use std::sync::{Arc, Mutex};

use super::capacity::{self, active_node_tracks, leave_at, reach_at};
use super::net::{path_edge, Network};
use super::service::{LineSvc, Run};
use super::upgrades;

/// A slower train standing at a passing station leaves this long after a faster one has passed (s):
/// the faster one clears the far turnout. Behind a slower train stepping aside there, a faster one
/// keeps only this, not the full spacing.
pub const PASS_CLEAR_S: f64 = 60.0;
/// Relative phases tried, per faster headway.
const PHASES: usize = 24;
/// The longest cycle of departures worked out (s), and the most slower headways one may span.
const CYCLE_MAX_S: f64 = 7200.0;
const CYCLE_MAX_K: usize = 12;
/// Results kept per plan, by the two lines' rates.
const MEMO: usize = 8;

/// Station `n` has passing tracks that work: on double track (constructed), with a platform.
pub fn passing_at(net: &Network, n: u32) -> bool {
    let i = n as usize;
    net.node_platform[i] > 0 && net.node_upgrades[i] & upgrades::PASSING != 0 && active_node_tracks(net, n) > 1
}

/// A place along a stretch where either run stops, or its end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Free times from reaching the stretch's start: arrival and departure (the same where the
    /// run does not stop), the slower run's and the faster run's.
    pub slow: [f64; 2],
    pub fast: [f64; 2],
    /// A passing station the slower run stops at (never the stretch's end).
    pub passing: bool,
    /// The slower run's stop index here, or -1.
    pub stop: i32,
    pub node: u32,
    /// Offset in the slower run.
    pub at: f64,
}

/// What a plan comes to at one pair of rates.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The faster run's mean loss over the stretch per train (s).
    pub loss: f64,
    /// The slower run's mean wait at each point per train (s).
    pub waits: Vec<f64>,
    /// How long after a slower train a faster one reaches the stretch's start (s, under its headway).
    pub offset: f64,
}

/// One faster run behind one slower run over a shared stretch with a passing station inside.
#[derive(Debug)]
pub struct Plan {
    pub fast: (u32, u8),
    pub slow: (u32, u8),
    /// Each run's stand at the stretch's start (s): slower, faster.
    pub start: [f64; 2],
    pub points: Vec<Point>,
    /// Per point: the next passing point, or the end.
    next: Vec<usize>,
    /// What a train keeps behind the one ahead (s), at the faster run's speed.
    pub spacing: f64,
    memo: Mutex<Vec<([u32; 2], Arc<Outcome>)>>,
}

/// The plan for path entries `i0..=i1` of the faster run `fr`, shared with the slower run `sr` from
/// its entry `j0`, or None where the slower run stops at no passing station strictly inside.
#[allow(clippy::too_many_arguments)]
pub fn plan(net: &Network, fast: (u32, u8), fr: &Run, slow: (u32, u8), sr: &Run, i0: usize, i1: usize, j0: usize, spacing: f64) -> Option<Plan> {
    let stops = |run: &Run, k: usize| run.stop_idx.iter().position(|&x| x as usize == k);
    let mut any = false;
    for k in i0 + 1..=i1 {
        any |= stops(sr, j0 + k - i0).is_some() && passing_at(net, fr.nodes[k]);
    }
    if !any {
        return None;
    }
    let j1 = j0 + (i1 - i0);
    let (fa, fb, sa, sb) = (fr.seg_off[i0], fr.node_s[i1 + 1], sr.seg_off[j0], sr.node_s[j1 + 1]);
    let (f0, s0) = (reach_at(fr, fa), reach_at(sr, sa));
    let times = |run: &Run, x: f64, t0: f64, stops: bool| {
        let leave = leave_at(run, x) - t0;
        [if stops { reach_at(run, x) - t0 } else { leave }, leave]
    };
    let mut points = vec![];
    for k in i0 + 1..=i1 {
        let j = j0 + k - i0;
        let (ss, fs) = (stops(sr, j), stops(fr, k));
        if ss.is_none() && fs.is_none() {
            continue;
        }
        let node = fr.nodes[k];
        points.push(Point {
            slow: times(sr, sr.node_s[j], s0, ss.is_some()),
            fast: times(fr, fr.node_s[k], f0, fs.is_some()),
            passing: ss.is_some() && passing_at(net, node),
            stop: ss.map_or(-1, |x| x as i32),
            node,
            at: sr.node_s[j],
        });
    }
    points.push(Point { slow: [reach_at(sr, sb) - s0; 2], fast: [reach_at(fr, fb) - f0; 2], passing: false, stop: -1, node: fr.nodes[i1 + 1], at: sb });
    let n = points.len();
    let next = (0..n).map(|p| (p + 1..n).find(|&q| points[q].passing).unwrap_or(n - 1)).collect();
    Some(Plan { fast, slow, start: [leave_at(sr, sa) - s0, leave_at(fr, fa) - f0], points, next, spacing, memo: Mutex::new(vec![]) })
}

/// The cycle the two headways repeat over: the shortest whole number of slower headways that is a
/// whole number of faster ones, or as many slower headways as fit `CYCLE_MAX_S`.
fn cycle(hs: f64, hf: f64) -> f64 {
    for k in 1..=CYCLE_MAX_K {
        let c = k as f64 * hs;
        if c > CYCLE_MAX_S {
            break;
        }
        let m = (c / hf).round();
        if m >= 1.0 && (c - m * hf).abs() < 0.5 {
            return c;
        }
    }
    (CYCLE_MAX_S / hs).floor().max(1.0) * hs
}

struct Train {
    slow: bool,
    t0: f64,
    arr: f64,
    dep: f64,
    measured: bool,
    /// Per point, how long it waited there for faster trains (slower trains only).
    wait: Vec<f64>,
}

impl Plan {
    /// The outcome at level `lev`, at the rates `net` holds now.
    pub fn outcome(&self, net: &Network, lev: usize) -> Arc<Outcome> {
        let (tf, ts) = (net.line_tph[self.fast.0 as usize][lev], net.line_tph[self.slow.0 as usize][lev]);
        let key = [tf.to_bits(), ts.to_bits()];
        if let Some(o) = self.memo.lock().unwrap().iter().find(|m| m.0 == key) {
            return o.1.clone();
        }
        let o = Arc::new(self.best(tf as f64, ts as f64));
        let mut m = self.memo.lock().unwrap();
        if m.len() >= MEMO {
            m.remove(0);
        }
        m.push((key, o.clone()));
        o
    }

    /// The relative phase (of `PHASES` a faster headway) with the least waiting and loss an hour.
    fn best(&self, tf: f64, ts: f64) -> Outcome {
        let n = self.points.len();
        if !(tf > 0.0 && ts > 0.0) {
            return Outcome { loss: 0.0, waits: vec![0.0; n], offset: 0.0 };
        }
        let (hf, hs) = (3600.0 / tf, 3600.0 / ts);
        let c = cycle(hs, hf);
        let mut best: Option<(f64, Outcome)> = None;
        for k in 0..PHASES {
            let (o, _) = self.sweep(hs, hf, hf * k as f64 / PHASES as f64, c);
            let cost = o.loss * tf + o.waits.iter().sum::<f64>() * ts;
            if best.as_ref().map_or(true, |b| cost < b.0 - 1e-9) {
                best = Some((cost, o));
            }
        }
        best.unwrap().1
    }

    /// The best relative phase at these headways (s) and, per slower train of an hour or more of
    /// whole cycles, its wait at each point: for checks.
    pub fn trains_at(&self, hs: f64, hf: f64) -> (Outcome, Vec<Vec<f64>>) {
        let o = self.best(3600.0 / hf, 3600.0 / hs);
        let c = cycle(hs, hf);
        self.sweep(hs, hf, o.offset, (3600.0 / c).ceil().max(1.0) * c)
    }

    /// One cycle of departures, the faster trains reaching the start `x` after the slower ones:
    /// point by point, every train keeps the spacing behind the one ahead (only `PASS_CLEAR_S` behind
    /// a slower one stepping aside at a passing station), and at each passing station a slower train
    /// waits for each faster one that would come within the spacing of it before its next passing
    /// point, unless that would hold it past its own headway less the spacing (it would block the
    /// next one). The outcome over the trains reaching the start in `[0, c)`, after a warm-up.
    fn sweep(&self, hs: f64, hf: f64, x: f64, c: f64) -> (Outcome, Vec<Vec<f64>>) {
        let n = self.points.len();
        let warm = self.points[n - 1].slow[0] + 2.0 * hs.max(hf);
        let mut tr: Vec<Train> = vec![];
        for (slow, h, x0, stand) in [(true, hs, 0.0, self.start[0]), (false, hf, x, self.start[1])] {
            let first = ((-warm - x0) / h).floor() as i64;
            let last = ((c + warm - x0) / h).ceil() as i64;
            for i in first..last {
                let t0 = x0 + i as f64 * h;
                tr.push(Train { slow, t0, arr: t0, dep: t0 + stand, measured: i >= 0 && (i as f64) * h < c - 1e-6, wait: if slow { vec![0.0; n] } else { vec![] } });
            }
        }
        let mut order: Vec<usize> = (0..tr.len()).collect();
        let mut fast: Vec<f64> = vec![];
        let (mut prev_s, mut prev_f) = (self.start[0], self.start[1]);
        for p in 0..n {
            let pt = self.points[p];
            order.sort_by(|&a, &b| tr[a].dep.total_cmp(&tr[b].dep).then(tr[b].slow.cmp(&tr[a].slow)).then(tr[a].t0.total_cmp(&tr[b].t0)));
            let mut ahead: Option<(f64, bool)> = None;
            for &i in &order {
                let t = &mut tr[i];
                let (free, stand) = if t.slow { (t.dep + pt.slow[0] - prev_s, pt.slow[1] - pt.slow[0]) } else { (t.dep + pt.fast[0] - prev_f, pt.fast[1] - pt.fast[0]) };
                let arr = match ahead {
                    Some((a, slow)) => free.max(a + if slow && !t.slow && pt.passing { PASS_CLEAR_S } else { self.spacing }),
                    None => free,
                };
                t.arr = arr;
                t.dep = arr + stand;
                ahead = Some((arr, t.slow));
            }
            if pt.passing {
                let b = self.points[self.next[p]];
                let (run_s, run_f) = (b.slow[0] - pt.slow[1], b.fast[0] - pt.fast[1]);
                fast.clear();
                fast.extend(tr.iter().filter(|t| !t.slow).map(|t| t.dep));
                fast.sort_by(f64::total_cmp);
                for t in tr.iter_mut().filter(|t| t.slow) {
                    let normal = t.dep;
                    for &xa in &fast {
                        if xa + PASS_CLEAR_S <= t.dep {
                            continue;
                        }
                        if xa + run_f >= t.dep + run_s + self.spacing || xa + PASS_CLEAR_S - normal > hs - self.spacing {
                            break;
                        }
                        t.dep = xa + PASS_CLEAR_S;
                    }
                    t.wait[p] = t.dep - normal;
                }
            }
            (prev_s, prev_f) = (pt.slow[1], pt.fast[1]);
        }
        let end = self.points[n - 1].fast[0];
        let (mut loss, mut nf) = (0.0f64, 0.0f64);
        let (mut waits, mut ns) = (vec![0.0f64; n], 0.0f64);
        let mut each = vec![];
        for t in tr.iter().filter(|t| t.measured) {
            if t.slow {
                ns += 1.0;
                waits.iter_mut().zip(&t.wait).for_each(|(a, w)| *a += w);
                each.push(t.wait.clone());
            } else {
                nf += 1.0;
                loss += t.arr - t.t0 - end;
            }
        }
        waits.iter_mut().for_each(|w| *w /= ns.max(1.0));
        (Outcome { loss: loss / nf.max(1.0), waits, offset: x }, each)
    }
}

/// T-2231: where line `l`'s trains get passed or would pass: the slower line's stops strictly inside
/// each stretch `l` shares with a slower line, or a faster line shares with it, on double track and
/// without passing tracks yet, in `l`'s order (the line panel's button, T-2232). Lines that are
/// planned count as well as running ones.
pub fn passing_spots(net: &Network, lines: &[LineSvc], l: u32) -> Vec<u32> {
    let Some(ls) = lines.get(l as usize).filter(|x| x.ok && !x.is_bus()) else { return vec![] };
    let live = |x: &LineSvc| x.ok;
    let mut out: Vec<u32> = vec![];
    let mut add = |fr: &Run, s: &capacity::Stretch| {
        let sr = &lines[s.line as usize].runs[s.run as usize];
        for k in s.i0 + 1..=s.i1 {
            let n = fr.nodes[k];
            if sr.stop_idx.contains(&((s.j0 + k - s.i0) as u32)) && !out.contains(&n) && net.is_station(n) && net.node_upgrades[n as usize] & upgrades::PASSING == 0 && net.node_tracks(n) > 1 {
                out.push(n);
            }
        }
    };
    for (fr, ss) in capacity::stretches(net, lines, l, &live, false).iter().enumerate() {
        ss.iter().for_each(|s| add(&ls.runs[fr], s));
    }
    let mut others: Vec<u32> = vec![];
    for &p in &ls.runs[0].segs {
        others.extend(net.lines_on_edge(path_edge(p)).iter().filter(|&&o| o != l));
    }
    others.sort_unstable();
    others.dedup();
    for f in others {
        for (fr, ss) in capacity::stretches(net, lines, f, &live, false).iter().enumerate() {
            ss.iter().filter(|s| s.line == l).for_each(|s| add(&lines[f as usize].runs[fr], s));
        }
    }
    let at = |n: u32| ls.runs[0].nodes.iter().position(|&x| x == n).unwrap_or(usize::MAX);
    out.sort_by_key(|&n| at(n));
    out
}

#[cfg(test)]
mod tests;
