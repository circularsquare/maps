//! T-4514: waits on shared stretches from the real gaps (docs/demand-kernel.md "Lines on one track and stops are one
//! service", notes/T-4514.md). Where rail lines leave one platform for one platform at the next stop in about the same
//! time (demand's `even_classes`, here by station node), a rider who takes whichever comes waits by the gaps between
//! their combined departures: the sum of the squared gaps over twice their sum (Anita 11-27). Demand takes it as a
//! factor on the wait it would have if they were evenly spaced: the gaps' count times the sum of their squares over
//! their sum squared, 1 where every gap is the same.

use std::collections::HashMap;

use crate::track::net::{Network, ALIGHT_ONLY};
use crate::track::params::PERIOD_HOURS;
use crate::track::service::{lattice_deps_local, Cal, Lattice, LineSvc, Service};

use super::{DAY, PERIODS};

/// Rides to the next stop within this of the shortest are one stretch, seconds (the track side's one pattern,
/// `capacity::FOLLOW_TIMED_MIN_S`; demand's `EVEN_RIDE_MIN` is this).
pub const SAME_RIDE_S: f64 = 30.0;
/// A shared stretch's wait that moved by more than this since demand's network calculates trips again (Anita 11-27),
/// seconds: a row's Offset or a re-spacing changes no run time or rate.
pub const MOVE_S: f64 = 5.0;
/// Departures are laid out from this long before the day, so a long run's later stops have the night's.
const LOOKBACK_S: f64 = 6.0 * 3600.0;

/// A run's stop on a shared stretch: the line, the run, and the stop's place in the run's travel order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Member {
    pub line: u32,
    pub run: u8,
    pub k: u32,
}

/// A stop's departures in a period: the headway, one departure (game seconds), and the ride on to the next stop as
/// demand's run times count it (the dwell there and the hop), seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pat {
    pub hw: f64,
    pub t: f64,
    pub ride: f64,
}

/// The lines leaving one platform for one platform at the next stop in one period, and their wait (s) and factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub q: u8,
    pub members: Vec<Member>,
    pub wait: f64,
    pub factor: f64,
}

/// Line `ls`'s pattern at each stop it is boarded at, per run and period, `[(q * 2 + r) * n + k]` (`line_span`'s
/// layout): from its weekday departures (demand's day), the departure nearest the period's middle and its headway.
/// None at a stop the run passes or only lets riders off at, at its last stop, and in a period it does not run.
pub fn stop_pats(ls: &LineSvc, lat: &Lattice) -> Vec<Option<Pat>> {
    let n = ls.stops.len();
    let mut out = vec![None; PERIODS * 2 * n];
    let lat = Lattice { rest_on: false, cal: Cal::NONE, ..*lat };
    let mut deps = vec![];
    for r in 0..2 {
        deps.clear();
        lattice_deps_local(&lat, r, DAY - LOOKBACK_S, 2.0 * DAY, &mut deps);
        deps.sort_by(|a, b| a.0.total_cmp(&b.0));
        let run = &ls.runs[r];
        for q in 0..PERIODS {
            let (h0, h1) = PERIOD_HOURS[q];
            let (a, b) = (DAY + h0 * 3600.0, DAY + h1 * 3600.0);
            let mid = 0.5 * (a + b);
            let p = &run.prof[lat.period_slot[q]];
            let m = p.dep.len().min(p.arr.len()).min(n);
            for k in 0..m.saturating_sub(1) {
                // run 1 lists its stops in its travel order, the line's backwards
                let only = ls.only.get(if r == 0 { k } else { n - 1 - k }).copied().unwrap_or(0);
                if run.passes(k) || only == ALIGHT_ONLY {
                    continue;
                }
                let want = mid - p.dep[k];
                let i = deps.partition_point(|d| d.0 < want);
                let near = [i.checked_sub(1), Some(i)].into_iter().flatten().filter_map(|j| deps.get(j)).min_by(|x, y| (x.0 - want).abs().total_cmp(&(y.0 - want).abs()));
                let Some(&(d, lev, hw, _)) = near else { continue };
                let Some(&at) = run.prof[lev as usize].dep.get(k) else { continue };
                let t = d + at;
                if !(hw.is_finite() && hw > 0.0 && t >= a && t < b) {
                    continue;
                }
                let ride = p.arr[k + 1] - p.dep[k] + if k > 0 { p.dep[k] - p.arr[k] } else { 0.0 };
                out[(q * 2 + r) * n + k] = Some(Pat { hw, t: t - lat.shift, ride });
            }
        }
    }
    out
}

/// `stop_pats` of line `l` as it runs now, empty for a line that is not a running train or tram line.
pub fn line_pats(net: &Network, svc: &Service, l: u32) -> Vec<Option<Pat>> {
    let Some(ls) = svc.lines.get(l as usize).filter(|ls| !ls.is_bus() && !ls.ferry) else { return vec![] };
    svc.lattice(net, l as usize).map_or(vec![], |lat| stop_pats(ls, &lat))
}

/// The factor on the even wait and the mean wait (s) of riders turning up at random against these patterns' combined
/// departures: over one headway, round, where all run at one; else over `span` s from the first's departure, the
/// gaps between its departures only.
pub fn real_wait(pats: &[Pat], span: f64) -> (f64, f64) {
    let Some(first) = pats.first() else { return (1.0, 0.0) };
    let hw = first.hw;
    let mut t: Vec<f64> = vec![];
    let round = pats.iter().all(|p| (p.hw - hw).abs() <= 1e-9 * hw);
    if round {
        t.extend(pats.iter().map(|p| p.t.rem_euclid(hw)));
    } else {
        let (a, b) = (first.t - 0.5 * span, first.t + 0.5 * span);
        for p in pats {
            let mut x = p.t + ((a - p.t) / p.hw).ceil() * p.hw;
            while x < b {
                t.push(x);
                x += p.hw;
            }
        }
    }
    t.sort_by(f64::total_cmp);
    let (mut n, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
    let mut gap = |g: f64| {
        n += 1.0;
        s1 += g;
        s2 += g * g;
    };
    for w in t.windows(2) {
        gap(w[1] - w[0]);
    }
    if round && !t.is_empty() {
        gap(t[0] + hw - t[t.len() - 1]);
    }
    if n == 0.0 || s1 <= 0.0 {
        return (1.0, 0.0);
    }
    ((n * s2 / (s1 * s1)).max(1.0), s2 / (2.0 * s1))
}

fn period_s(q: usize) -> f64 {
    (PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0) * 3600.0
}

/// The patterns of `members` in `pats` (by line, `line_pats`), where each still has one.
fn members_now(members: &[Member], q: usize, pats: &HashMap<u32, Vec<Option<Pat>>>, svc: &Service) -> Vec<Pat> {
    members
        .iter()
        .filter_map(|m| {
            let n = svc.lines.get(m.line as usize)?.stops.len();
            pats.get(&m.line)?.get((q * 2 + m.run as usize) * n + m.k as usize).copied().flatten()
        })
        .collect()
}

/// Every shared stretch of the running train and tram lines, per period: runs leaving one station node for one station
/// node at the next stop, rides within `SAME_RIDE_S` of the shortest, of two or more lines (one line's two runs, as
/// a balloon's down its trunk, are not spaced against each other), as demand's `even_classes` makes its classes.
pub fn groups(net: &Network, svc: &Service) -> Vec<Group> {
    // hops (node, next node) two or more lines run, the only ones that can be shared: per hop its first line, or
    // `SHARED` once another runs it too (none on a save of lines with their own stations, so no departures laid out)
    const SHARED: u32 = u32::MAX;
    let rail = |l: usize| svc.lines.get(l).filter(|ls| ls.running && !ls.is_bus() && !ls.ferry);
    let mut hops: HashMap<(u32, u32), u32> = HashMap::new();
    for l in 0..svc.lines.len() {
        let Some(ls) = rail(l) else { continue };
        for r in 0..2 {
            for k in 0..ls.stops.len().saturating_sub(1) {
                let (Some(a), Some(b)) = (ls.stop_node(r, k), ls.stop_node(r, k + 1)) else { continue };
                let h = hops.entry((a, b)).or_insert(l as u32);
                if *h != l as u32 {
                    *h = SHARED;
                }
            }
        }
    }
    let mut pats: HashMap<u32, Vec<Option<Pat>>> = HashMap::new();
    // (period, node, next node, ride, member)
    let mut by: Vec<(u8, u32, u32, f64, Member)> = vec![];
    for l in 0..svc.lines.len() as u32 {
        let Some(ls) = rail(l as usize) else { continue };
        let n = ls.stops.len();
        let hop = |r: usize, k: usize| Some((ls.stop_node(r, k)?, ls.stop_node(r, k + 1)?)).filter(|h| hops.get(h) == Some(&SHARED));
        if !(0..2).any(|r| (0..n.saturating_sub(1)).any(|k| hop(r, k).is_some())) {
            continue;
        }
        let v = line_pats(net, svc, l);
        if v.is_empty() {
            continue;
        }
        for q in 0..PERIODS {
            for r in 0..2 {
                for k in 0..n {
                    let Some(p) = v[(q * 2 + r) * n + k] else { continue };
                    let Some((a, b)) = hop(r, k) else { continue };
                    by.push((q as u8, a, b, p.ride, Member { line: l, run: r as u8, k: k as u32 }));
                }
            }
        }
        pats.insert(l, v);
    }
    by.sort_by(|x, y| (x.0, x.1, x.2).cmp(&(y.0, y.1, y.2)).then(x.3.total_cmp(&y.3)).then(x.4.cmp(&y.4)));
    let mut out = vec![];
    let mut i = 0;
    while i < by.len() {
        let mut j = i + 1;
        while j < by.len() && (by[j].0, by[j].1, by[j].2) == (by[i].0, by[i].1, by[i].2) && by[j].3 - by[i].3 <= SAME_RIDE_S {
            j += 1;
        }
        if by[i..j].iter().any(|x| x.4.line != by[i].4.line) {
            let q = by[i].0 as usize;
            let members: Vec<Member> = by[i..j].iter().map(|x| x.4).collect();
            let (factor, wait) = real_wait(&members_now(&members, q, &pats, svc), period_s(q));
            out.push(Group { q: q as u8, members, wait, factor });
        }
        i = j;
    }
    out
}

/// A factor as demand takes it: to hundredths, so a wait that does not move compares equal to the bit (as
/// `span::stop_phi`).
pub fn factor_sent(f: f64) -> f32 {
    ((f * 100.0).round() / 100.0) as f32
}

/// The shared stretches as demand's network was last sent them (`TrackApi::shared_gaps`), with each line's.
#[derive(Clone, Debug, Default)]
pub struct Sent {
    groups: Vec<Group>,
    of_line: HashMap<u32, Vec<u32>>,
}

impl Sent {
    pub fn of(groups: Vec<Group>) -> Sent {
        let mut of_line: HashMap<u32, Vec<u32>> = HashMap::new();
        for (g, grp) in groups.iter().enumerate() {
            for m in &grp.members {
                let v = of_line.entry(m.line).or_default();
                if v.last() != Some(&(g as u32)) {
                    v.push(g as u32);
                }
            }
        }
        Sent { groups, of_line }
    }

    /// Per stop whose factor demand takes past 1: line, `[(q * 2 + run) * n + k]`, the factor (`factor_sent`).
    pub fn triples(&self, svc: &Service) -> Vec<f32> {
        let mut out = vec![];
        for g in &self.groups {
            let f = factor_sent(g.factor);
            if f <= 1.0 {
                continue;
            }
            for m in &g.members {
                let n = svc.lines[m.line as usize].stops.len();
                out.extend([m.line as f32, ((g.q as usize * 2 + m.run as usize) * n + m.k as usize) as f32, f]);
            }
        }
        out
    }

    /// Of `lines`, those on a stretch sent whose wait now differs from the one sent by more than `MOVE_S`.
    pub fn moved(&self, net: &Network, svc: &Service, lines: &[u32]) -> Vec<u32> {
        let mut pats: HashMap<u32, Vec<Option<Pat>>> = HashMap::new();
        let mut out = vec![];
        for &l in lines {
            let Some(gs) = self.of_line.get(&l) else { continue };
            for &g in gs {
                let grp = &self.groups[g as usize];
                for m in &grp.members {
                    pats.entry(m.line).or_insert_with(|| line_pats(net, svc, m.line));
                }
                let q = grp.q as usize;
                let wait = real_wait(&members_now(&grp.members, q, &pats, svc), period_s(q)).1;
                if (wait - grp.wait).abs() > MOVE_S {
                    out.push(l);
                    break;
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests;
