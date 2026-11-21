//! Each train its own timetable (T-3346 stage 2, notes/T-3346.md). The capacity pass gives every
//! train of a run in a slot the same holds: the mean wait at each passing station (`overtake`, kind
//! 13) and the mean following loss behind a slower line it is timetabled with
//! (`capacity::follow_best`, kind 11). Here the timetable is worked out train by train, against
//! the other line's trains as they run:
//!
//! - a slower train at a passing station waits for exactly the faster trains that would catch it
//!   before its next one (`overtake::train_waits`);
//! - a faster train with nowhere to pass is timed behind the slower train ahead of it, running
//!   slower from where it would otherwise come within the spacing of it (`follow_one`).
//!
//! A run keeps at most `MAX_VARS` variants per slot, and each weekday departure names one by its
//! place in the day (`Run::pattern`, read only through `Service::run_deps` into `Trip::var`).
//! Weekend departures of a line with its own weekends run the slot's timetable, and a pair of lines
//! only one of which has its own weekends is not timed. Demand, round trips and fleets keep the
//! means; a train that waited longer stands less at its last stop (`LineSvc::trip_len_var`).
//! Variants are derived, never saved, and worked out again after every edit has settled and its
//! departures are placed (`retime`).

use std::collections::{BTreeMap, HashMap};

use super::capacity::{FOLLOW_PASSING, FOLLOW_TIMED};
use super::net::Network;
use super::overtake::{self, Pass, PASS_CLEAR_S};
use super::params::*;
use super::profile::{self, Hold, Profile};
use super::service::{lattice_deps, lattice_deps_local, profile_with, Cal, Dep, Lattice, Run, Service, Variant, NO_PLACE};

/// Variants per run and slot, besides the slot's own timetable.
pub const MAX_VARS: usize = 16;
/// Profiles per (line, run, slot) the renderer indexes: the slot's and its variants
/// (`((line * 2 + run) * SLOTS + slot) * PROFILE_VARS + var`).
pub const PROFILE_VARS: usize = MAX_VARS + 1;
/// The least a train stands at its last stop when its own timetable ran longer than the slot's
/// (s): it takes its next departure as the slot's train would where this allows.
pub const TURN_MIN_S: f64 = 60.0;
/// T-3929: the least a faster train that caught a slower one before the first passing station
/// reaches it after the slower one (s): about a 400 m train's length at line speed, so the slower
/// one is off the running line. (The plan's `PASS_CLEAR_S` there would hold every such train for
/// the share of its averaged loss the slot spreads past that station.)
const LEAD_IN_CLEAR_S: f64 = 10.0;
/// Waits and losses are rounded up to whole steps of this (s) before variants are told apart.
const STEP_S: f64 = 2.0;
const DAY: f64 = 86400.0;
/// The weekday the variants are worked out on (local day number; any day, every day around it
/// taken as a weekday).
const DAY_REF: i64 = 10;
/// How far around the reference day the other line's trains are worked out (s): a run, a stand
/// and a headway either side.
const AROUND_S: f64 = 10.0 * 3600.0;

type Memo = HashMap<(u32, u8, Vec<u64>), Profile>;

/// Profiles built this time, and those built last time to take again (`Service::timing_memo`).
struct Built {
    old: Memo,
    new: Memo,
}

impl Built {
    /// `profile_with`, or the same profile built before: by line, run, its free run (a rebuilt run
    /// differs there), the line's dwell and turnaround, and the holds.
    fn prof(&mut self, net: &Network, l: u32, r: usize, run: &Run, stop_holds: &[f64], holds: &[Hold]) -> Profile {
        let mut key: Vec<u64> = vec![run.free.duration.to_bits(), run.len.to_bits(), run.free.phases.len() as u64, net.line_dwell[l as usize].to_bits() as u64, net.line_turn[l as usize].to_bits() as u64];
        key.extend(stop_holds.iter().map(|x| x.to_bits()));
        key.push(u64::MAX);
        key.extend(holds.iter().flat_map(|h| [h.s.to_bits(), h.delay.to_bits(), h.span.to_bits()]));
        let k = (l, r as u8, key);
        if let Some(p) = self.new.get(&k) {
            return p.clone();
        }
        let p = self.old.remove(&k).unwrap_or_else(|| profile_with(net, l, run, stop_holds, holds));
        self.new.insert(k, p.clone());
        p
    }
}

/// One run's variants in one slot, and its pattern.
struct Timed {
    line: u32,
    run: usize,
    lev: usize,
    vars: Vec<Variant>,
    pattern: Vec<(u32, u8)>,
}

/// A line's pattern with every day around the reference day a weekday: what the variants are
/// worked out on.
fn weekday(lat: &Lattice) -> Lattice {
    Lattice { cal: Cal::all(DAY_REF, WEEKDAYS), ..*lat }
}

/// Work out every line's variants again (after an edit has settled and its phases are placed).
pub fn retime(net: &Network, svc: &mut Service) {
    let key = inputs(net, svc);
    if key == svc.timing_key {
        // nothing it reads moved: the variants stand (an update marks a line it touched untimed)
        for l in 0..svc.lines.len() {
            if svc.lines[l].runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty())) {
                svc.lines[l].timed = svc.lattice(net, l).map(|x| x.timing_key());
            }
        }
        return;
    }
    svc.timing_key = key;
    // T-3896: the variants are in the line's phase tables, which the page reads again only for the
    // lines noted as changed: any line timed otherwise than before is noted, edited or not
    let before: Vec<Vec<Vec<Variant>>> = svc.lines.iter().map(|ls| ls.runs.iter().flat_map(|r| r.vars.iter().cloned()).collect()).collect();
    clear(svc);
    if !svc.timing_key.is_empty() {
        let lats: Vec<Option<Lattice>> = (0..svc.lines.len()).map(|l| svc.lattice(net, l).map(|x| weekday(&x))).collect();
        let mut built = Built { old: std::mem::take(&mut svc.timing_memo), new: HashMap::new() };
        // (before the waits: a slower train waits for the faster ones as they run)
        let lead = lead_in(net, svc, &lats, &mut built);
        install(svc, lead, &lats);
        let passing = passing_waits(net, svc, &lats, &mut built);
        install(svc, passing, &lats);
        // (before the following: a faster line timed behind a slower one reads its trains as they run)
        floor(net, svc);
        let follow = following(net, svc, &lats, &mut built);
        install(svc, follow, &lats);
        svc.timing_memo = built.new;
    } else {
        svc.timing_memo.clear();
    }
    let moved: Vec<u32> = (0..svc.lines.len()).filter(|&l| before.get(l).is_some_and(|b| svc.lines[l].runs.iter().flat_map(|r| r.vars.iter()).ne(b.iter()))).map(|l| l as u32).collect();
    svc.note_changed(&moved);
}

/// Every run's variants and patterns dropped, and every line untimed.
fn clear(svc: &mut Service) {
    for ls in svc.lines.iter_mut() {
        for r in ls.runs.iter_mut() {
            for lev in 0..SLOTS {
                r.vars[lev].clear();
                r.pattern[lev].clear();
            }
        }
        ls.timed = None;
    }
}

/// The stand a run's trains have at its last stop before their next departure (s), on average over
/// a slot's weekday departures: the line's turnaround, the share of a headway line's spare time the
/// spacing put at that end, less how much longer the run's trains take on their own timetables
/// than on the slot's. None for a run without variants in that slot, a loop or a line run one way.
fn stand_at_end(net: &Network, svc: &Service, l: usize, r: usize, lev: usize) -> Option<f64> {
    let ls = &svc.lines[l];
    let run = &ls.runs[r];
    if ls.is_loop() || ls.one_way || run.pattern[lev].is_empty() {
        return None;
    }
    let n = run.pattern[lev].len() as f64;
    let excess = run.pattern[lev].iter().map(|&(_, v)| ls.prof_of(r, lev, v).duration - run.prof[lev].duration).sum::<f64>() / n;
    let split = net.line_split_of(l as u32)[lev] as f64;
    let share = if r == 0 { split } else { 1.0 - split };
    Some(ls.close_stand(net, l as u32) + share * svc.spare(net, l, lev) - excess)
}

/// T-3346 stage 2's floor, for the waits at passing stations: a run whose trains, on their own
/// timetables, would not on average make their next departure at its far end (a stand there under `TURN_MIN_S`, or the line's turnaround
/// where that is shorter) runs that slot's timetable instead, as before. Its round trip and fleet
/// are sized on the slot's (the averaged waits at the plan's best phase); where the trains really
/// run at another phase, timing them train by train would leave them standing a whole headway at
/// the end for their next departure, holding up the trains behind (t2233-passing-map's 30/10).
fn floor(net: &Network, svc: &mut Service) {
    for l in 0..svc.lines.len() {
        let need = TURN_MIN_S.min(svc.lines[l].close_stand(net, l as u32)) - 1.0;
        for lev in net.slots() {
            for r in 0..2 {
                if stand_at_end(net, svc, l, r, lev).is_some_and(|s| s < need) {
                    let run = &mut svc.lines[l].runs[r];
                    run.vars[lev].clear();
                    run.pattern[lev].clear();
                }
            }
        }
        if svc.lines[l].runs.iter().all(|r| r.vars.iter().all(|v| v.is_empty())) {
            svc.lines[l].timed = None;
        }
    }
}

/// Everything `retime` reads, as bits: the waits and following losses with their users and delays,
/// and each line they name's pattern, dwell, turnaround and timetables. Empty where there is none
/// (no line is timed).
fn inputs(net: &Network, svc: &Service) -> Vec<u64> {
    let cap = &svc.cap;
    let mut key = vec![];
    let mut lines: Vec<u32> = vec![];
    for r in (0..cap.res.len()).filter(|&r| cap.wait_point(r).is_some() || cap.is_follow(r)) {
        key.push(r as u64);
        for u in cap.users_of(r) {
            key.extend([u.line as u64, u.run as u64, u.stop as u64, u.at.to_bits(), u.occ.to_bits()]);
            lines.push(u.line);
        }
        key.extend(cap.delay[r].iter().map(|x| x.to_bits()));
        if let Some((_, k)) = cap.wait_point(r) {
            key.push(k as u64);
        }
    }
    if key.is_empty() {
        return key;
    }
    key.extend(net.period_slot().map(|s| s as u64));
    lines.sort_unstable();
    lines.dedup();
    for l in lines {
        key.push(u64::MAX - l as u64);
        if let Some(lat) = svc.lattice(net, l as usize) {
            key.extend(lat.tph.iter().map(|x| x.to_bits() as u64));
            key.extend(lat.off.iter().chain(&lat.phase).chain([&lat.lead, &lat.shift]).map(|x| x.to_bits()));
            key.push(lat.rest_on as u64);
            key.extend(lat.period_slot.map(|s| s as u64));
            key.extend(lat.hours.words());
        }
        key.extend([net.line_dwell[l as usize].to_bits() as u64, net.line_turn[l as usize].to_bits() as u64]);
        for run in &svc.lines[l as usize].runs {
            key.push(run.free.duration.to_bits());
            for lev in net.slots() {
                key.push(run.prof[lev].duration.to_bits());
                key.extend(run.stop_holds[lev].iter().map(|x| x.to_bits()));
                key.extend(run.holds[lev].iter().flat_map(|h| [h.s.to_bits(), h.delay.to_bits(), h.span.to_bits()]));
            }
        }
    }
    key
}

fn install(svc: &mut Service, done: Vec<Timed>, lats: &[Option<Lattice>]) {
    for t in done {
        let ls = &mut svc.lines[t.line as usize];
        ls.runs[t.run].vars[t.lev] = t.vars;
        ls.runs[t.run].pattern[t.lev] = t.pattern;
        ls.timed = lats[t.line as usize].map(|x| x.timing_key());
    }
}

/// A run's weekday departures in slot `lev` over the reference day (game time), with their
/// headways and places in the day: each place once, the day being the same every weekday.
fn day_deps(lat: &Lattice, r: usize, lev: usize) -> Vec<(f64, f64, u32)> {
    let mut buf: Vec<Dep> = vec![];
    lattice_deps_local(lat, r, DAY_REF as f64 * DAY - 3.0 * 3600.0, (DAY_REF + 1) as f64 * DAY, &mut buf);
    let mut seen: Vec<u32> = vec![];
    let mut out = vec![];
    for (d, l, hw, place) in buf {
        if l as usize != lev || place == NO_PLACE || seen.contains(&place) {
            continue;
        }
        seen.push(place);
        out.push((d - lat.shift, hw, place));
    }
    out
}

/// A run's departures around the reference day, every slot, game time.
fn around(lat: &Lattice, r: usize) -> Vec<Dep> {
    let mut buf = vec![];
    lattice_deps(lat, r, DAY_REF as f64 * DAY - AROUND_S, (DAY_REF + 1) as f64 * DAY + AROUND_S, &mut buf);
    buf
}

/// The game times another line's trains can meet trains leaving at `deps` that run `dur` (s):
/// from an hour before the first (less the other's run, which the caller takes off) to the last's
/// arrival and an hour.
fn window(deps: &[(f64, f64, u32)], dur: f64) -> (f64, f64) {
    let (a, b) = deps.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), d| (a.min(d.0), b.max(d.0)));
    (a - 3600.0, b + dur + 3600.0)
}

/// When a profile's train first reaches offset `s` (before standing there).
fn reach(p: &Profile, s: f64) -> f64 {
    profile::time_at(&p.phases, (s - 0.5).max(0.0))
}

/// When it leaves or passes `s`.
fn leave(p: &Profile, s: f64) -> f64 {
    profile::time_at(&p.phases, s)
}

/// Rounded up to whole `STEP_S`.
fn step(x: f64) -> i64 {
    if x < 0.5 { 0 } else { (x / STEP_S).ceil() as i64 }
}

/// At most `MAX_VARS` keys that every departure's own fits under (each part no smaller), and per
/// departure which: alike keys merge, the pair that adds the least waiting first, of any two
/// (T-3931: on lines whose trains drift against each other's, count lines, the keys that sort
/// together are rarely alike, and merging only those added enough waiting to put whole runs back
/// on the slot's timetable, `floor`). `ok(key, i)`: whether departure `i` may run that key's
/// timetable; a merge every departure of may is taken first, and only where none is left one
/// that may not.
fn cluster(each: &[Vec<i64>], ok: impl Fn(&[i64], usize) -> bool) -> (Vec<Vec<i64>>, Vec<usize>) {
    // (key, its departures)
    let mut keys: Vec<(Vec<i64>, Vec<usize>)> = vec![];
    let mut sorted: Vec<usize> = (0..each.len()).collect();
    sorted.sort_by(|&a, &b| each[a].cmp(&each[b]));
    for i in sorted {
        match keys.last_mut() {
            Some(x) if x.0 == each[i] => x.1.push(i),
            _ => keys.push((each[i].clone(), vec![i])),
        }
    }
    let max = |a: &[i64], b: &[i64]| a.iter().zip(b).map(|(x, y)| *x.max(y)).collect::<Vec<i64>>();
    // what a merge adds (the waiting every departure of both gains), past every merge all of whose
    // departures may run it where one may not
    const UNSAFE: f64 = 1e15;
    let cost = |a: &(Vec<i64>, Vec<usize>), b: &(Vec<i64>, Vec<usize>)| {
        let m = max(&a.0, &b.0);
        let add = |k: &(Vec<i64>, Vec<usize>)| m.iter().zip(&k.0).map(|(x, y)| (x - y) as f64).sum::<f64>() * k.1.len() as f64;
        add(a) + add(b) + if a.1.iter().chain(&b.1).all(|&i| ok(&m, i)) { 0.0 } else { UNSAFE }
    };
    if keys.len() <= MAX_VARS {
        let which = (0..each.len()).map(|i| keys.iter().position(|k| k.1.contains(&i)).unwrap()).collect();
        return (keys.into_iter().map(|k| k.0).collect(), which);
    }
    // every pair's cost, and per key the cheapest other (the scan per merge is then one per key)
    let mut c: Vec<Vec<f64>> = vec![vec![f64::INFINITY; keys.len()]; keys.len()];
    for i in 0..keys.len() {
        for j in i + 1..keys.len() {
            let x = cost(&keys[i], &keys[j]);
            (c[i][j], c[j][i]) = (x, x);
        }
    }
    let low = |row: &[f64]| row.iter().enumerate().fold((f64::INFINITY, 0), |b, (j, &x)| if x < b.0 { (x, j) } else { b });
    let mut near: Vec<(f64, usize)> = c.iter().map(|r| low(r)).collect();
    while keys.len() > MAX_VARS {
        let bi = (0..keys.len()).min_by(|&a, &b| near[a].0.total_cmp(&near[b].0)).unwrap();
        let (bi, bj) = (bi.min(near[bi].1), bi.max(near[bi].1));
        let b = keys.remove(bj);
        c.remove(bj);
        c.iter_mut().for_each(|row| {
            row.remove(bj);
        });
        near.remove(bj);
        keys[bi] = (max(&keys[bi].0, &b.0), keys[bi].1.iter().chain(&b.1).copied().collect());
        for j in (0..keys.len()).filter(|&j| j != bi) {
            let x = cost(&keys[bi], &keys[j]);
            (c[bi][j], c[j][bi]) = (x, x);
        }
        for i in 0..keys.len() {
            let (x, j) = near[i];
            near[i] = if i == bi || j == bi || j == bj {
                low(&c[i])
            } else {
                let j = if j > bj { j - 1 } else { j };
                if c[i][bi] < x { (c[i][bi], bi) } else { (x, j) }
            };
        }
    }
    let mut which = vec![0; each.len()];
    for (k, key) in keys.iter().enumerate() {
        for &i in &key.1 {
            which[i] = k;
        }
    }
    let reps: Vec<Vec<i64>> = keys.into_iter().map(|k| k.0).collect();
    for (i, e) in each.iter().enumerate() {
        // the representative it fits under and may run that adds the least, if not its own
        let add = |k: usize| reps[k].iter().zip(e).map(|(r, x)| r - x).sum::<i64>();
        if let Some(k) = (0..reps.len()).filter(|&k| reps[k].iter().zip(e).all(|(r, x)| r >= x) && ok(&reps[k], i)).min_by_key(|&k| add(k)) {
            if add(k) < add(which[i]) || !ok(&reps[which[i]], i) {
                which[i] = k;
            }
        }
    }
    (reps, which)
}

/// A pattern from each departure's place and variant (1-based), sorted by place.
fn pattern_of(deps: &[(f64, f64, u32)], which: &[usize]) -> Vec<(u32, u8)> {
    let mut p: Vec<(u32, u8)> = deps.iter().zip(which).map(|(d, &w)| (d.2, w as u8 + 1)).collect();
    p.sort_unstable();
    p
}

/// Slower runs at passing stations: each train waits for the faster trains that would catch it.
fn passing_waits(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built) -> Vec<Timed> {
    let cap = &svc.cap;
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for r in 0..cap.res.len() {
        if cap.wait_point(r).is_some() {
            if let Some(o) = cap.users_of(r).iter().find(|u| u.owner) {
                by.entry((o.line, o.run)).or_default().push(r);
            }
        }
    }
    let mut fast_deps: BTreeMap<(u32, u8), Vec<Dep>> = BTreeMap::new();
    let mut out = vec![];
    for (&(l, r), ws) in &by {
        let (ls, Some(lat)) = (&svc.lines[l as usize], lats[l as usize].as_ref()) else { continue };
        // a pair where one line runs its own weekends is left averaged: its weekend trains would
        // meet trains the variants were not worked out against
        if ws.iter().any(|&w| cap.wait_point(w).is_some_and(|(f, _)| lats[f as usize].as_ref().map_or(true, |x| x.rest_on != lat.rest_on))) {
            continue;
        }
        let run = &ls.runs[r as usize];
        for lev in net.slots() {
            let deps = day_deps(lat, r as usize, lev);
            if deps.is_empty() {
                continue;
            }
            // its stop holds without the averaged waits
            let mut base = run.stop_holds[lev].clone();
            for &w in ws {
                let o = cap.users_of(w).iter().find(|u| u.owner).unwrap();
                if let Some(x) = base.get_mut(o.stop as usize) {
                    *x -= cap.delay[w][lev];
                }
            }
            base.iter_mut().for_each(|x| if *x < 1e-6 { *x = 0.0 });
            let b = built.prof(net, l, r as usize, run, &base, &run.holds[lev]);
            // the faster trains that can meet these: from a run before the first to a run and an
            // hour after the last
            let (lo, hi) = window(&deps, b.duration);
            let mut stations: BTreeMap<usize, Vec<Pass>> = BTreeMap::new();
            let mut hcap = f64::INFINITY;
            for &w in ws {
                let (f, k) = cap.wait_point(w).unwrap();
                let o = cap.users_of(w).iter().find(|u| u.owner).unwrap();
                let Some(plan) = cap.plans_of(w).first() else { continue };
                let (fl, Some(flat)) = (&svc.lines[f as usize], lats[f as usize].as_ref()) else { continue };
                hcap = hcap.min(plan.headway_cap(net));
                let fr = plan.fast.1 as usize;
                let k = k as usize;
                let (pt, q, nb) = (plan.points[k], plan.points[k + 1], plan.points[plan.next_passing(k)]);
                let s_leave = b.dep[o.stop as usize];
                let run_s = reach(&b, nb.at) - s_leave;
                let s_to_q = reach(&b, q.at) - s_leave;
                let fdeps = fast_deps.entry((f, fr as u8)).or_insert_with(|| around(flat, fr));
                let passes = stations.entry(o.stop as usize).or_default();
                for &(d, flev, hf, place) in fdeps.iter().filter(|x| x.0 >= lo - fl.runs[fr].prof[x.1 as usize].duration && x.0 < hi) {
                    // each faster train as it runs (`lead_in`)
                    let fp = fl.prof_of(fr, flev as usize, fl.runs[fr].var_at(flev as usize, place));
                    let x = leave(fp, pt.fat);
                    let lead = PASS_CLEAR_S.max(reach(fp, q.fat) - x + plan.spacing - s_to_q);
                    passes.push(Pass { at: d + x, lead, run_f: reach(fp, nb.fat) - x, run_s, spacing: plan.spacing, most: hf.min(plan.headway_cap(net)) + lead });
                }
            }
            let stations: Vec<(usize, f64, Vec<Pass>)> = stations
                .into_iter()
                .map(|(j, mut v)| {
                    v.sort_by(|a, b| a.at.total_cmp(&b.at));
                    (j, b.dep[j], v)
                })
                .collect();
            let times: Vec<(f64, f64)> = deps.iter().map(|d| (d.0, d.1.min(hcap))).collect();
            let waits = overtake::train_waits(&times, &stations);
            let each: Vec<Vec<i64>> = waits.iter().map(|w| w.iter().map(|&x| step(x)).collect()).collect();
            // a train runs a merged timetable only where it meets no faster train at a station its
            // own does not
            let secs = |k: &[i64]| k.iter().map(|&x| x as f64 * STEP_S).collect::<Vec<f64>>();
            let own: Vec<u64> = each.iter().zip(&times).map(|(k, t)| overtake::conflicts(t.0, &stations, &secs(k))).collect();
            let (reps, which) = cluster(&each, |k, i| overtake::conflicts(times[i].0, &stations, &secs(k)) & !own[i] == 0);
            let vars = reps
                .iter()
                .map(|rep| {
                    let mut sh = base.clone();
                    for ((j, ..), &w) in stations.iter().zip(rep) {
                        sh[*j] += w as f64 * STEP_S;
                    }
                    let prof = built.prof(net, l, r as usize, run, &sh, &run.holds[lev]);
                    Variant { stop_holds: sh, holds: run.holds[lev].clone(), prof }
                })
                .collect();
            out.push(Timed { line: l, run: r as usize, lev, vars, pattern: pattern_of(&deps, &which) });
        }
    }
    out
}

/// T-3929: a faster run behind the one slower run on a stretch with passing stations
/// (`FOLLOW_PASSING`). The slower trains step aside only from the first passing station on, so a
/// faster train reaching the stretch close behind a slower one catches it on the way there,
/// whatever the two lines' phases (they drift wherever the rates are no simple ratio). Such a train
/// is timed behind it to that station (`follow_one`), reaching it `LEAD_IN_CLEAR_S` after the
/// slower one: a loss on top of the slot's timetable, no more than its stand at the far end allows.
/// The rest run the slot's timetable. A run with none to time is left alone.
fn lead_in(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built) -> Vec<Timed> {
    let cap = &svc.cap;
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for r in (0..cap.res.len()).filter(|&r| cap.is_follow(r)) {
        if let Some(o) = cap.users_of(r).iter().find(|u| u.owner) {
            by.entry((o.line, o.run)).or_default().push(r);
        }
    }
    let mut out = vec![];
    for (&(f, fr), rs) in &by {
        let [rr] = rs[..] else { continue };
        let us = cap.users_of(rr);
        let o = *us.iter().find(|u| u.owner).unwrap();
        let slow: Vec<_> = us.iter().filter(|u| !u.owner).collect();
        let ([su], [plan]) = (&slow[..], cap.plans_of(rr)) else { continue };
        if su.stop != FOLLOW_PASSING {
            continue;
        }
        let (fl, sl) = (&svc.lines[f as usize], &svc.lines[su.line as usize]);
        let (Some(flat), Some(slat)) = (lats[f as usize].as_ref(), lats[su.line as usize].as_ref()) else { continue };
        if flat.rest_on != slat.rest_on {
            continue;
        }
        let (frun, srun) = (&fl.runs[fr as usize], &sl.runs[su.run as usize]);
        let Some(p1) = plan.points.iter().find(|p| p.passing).copied() else { continue };
        // the offsets along the slower run where the faster run's are `x`: the same track
        let (a, delta) = (o.at, p1.at - p1.fat);
        let mut xs: Vec<f64> = vec![a];
        xs.extend(frun.node_s.iter().copied().filter(|&x| x > a + 1.0 && x < p1.fat - 1.0));
        xs.push(p1.fat);
        let slow_deps = around(slat, su.run as usize);
        for lev in net.slots() {
            if flat.tph[lev] <= 0.0 || slat.tph[lev] <= 0.0 {
                continue;
            }
            let deps = day_deps(flat, fr as usize, lev);
            if deps.is_empty() {
                continue;
            }
            // as `following`: a train losing more than its stand at the far end less `TURN_MIN_S`
            // would miss its next departure
            let close = fl.close_stand(net, f);
            let split = net.line_split_of(f)[lev] as f64;
            let at_end = close + if fr == 0 { split } else { 1.0 - split } * svc.spare(net, f as usize, lev);
            let slack = (at_end - TURN_MIN_S.min(close)).max(0.0);
            let room = slack - 2.0 * STEP_S;
            if room <= 0.0 {
                continue;
            }
            let e0 = &frun.prof[lev];
            let (lo, hi) = window(&deps, e0.duration);
            let e_at: Vec<f64> = xs.iter().map(|&x| reach(e0, x)).collect();
            // the slower trains leaving each node, and at the passing station their arrival and
            // `LEAD_IN_CLEAR_S` (in `follow_one`'s terms, less the spacing it adds); no spacing at
            // the stretch's start, as in the plan (`Plan::sweep`), so its offsets time no one
            let mut slow: Vec<(f64, Vec<f64>)> = slow_deps
                .iter()
                .filter(|x| x.0 >= lo - srun.prof[x.1 as usize].duration && x.0 < hi)
                .map(|&(d, slev, _, _)| {
                    let p = &srun.prof[slev as usize];
                    let mut t: Vec<f64> = xs.iter().map(|&x| d + leave(p, x + delta)).collect();
                    t[0] = f64::NEG_INFINITY;
                    *t.last_mut().unwrap() = d + reach(p, p1.at) + LEAD_IN_CLEAR_S - plan.spacing;
                    (d + reach(p, a + delta), t)
                })
                .collect();
            slow.sort_by(|x, y| x.0.total_cmp(&y.0));
            let each: Vec<(f64, f64, f64)> = deps
                .iter()
                .map(|&(d, _, _)| {
                    let e: Vec<f64> = e_at.iter().map(|t| d + t).collect();
                    let i = slow.partition_point(|s| s.0 <= e[0]);
                    if i == 0 {
                        return (0.0, a, 0.0);
                    }
                    let (st, x0, dl) = follow_one(&xs, &e, &slow[i - 1].1, plan.spacing);
                    let over = st + dl - room;
                    if over <= 0.0 {
                        return (st, x0, dl);
                    }
                    let d2 = (dl - over).max(0.0);
                    ((st - (over - (dl - d2))).max(0.0), x0, d2)
                })
                .collect();
            if each.iter().all(|x| step(x.0) == 0 && step(x.2) == 0) {
                continue;
            }
            let starts: Vec<f64> = {
                let mut v: Vec<f64> = each.iter().map(|x| x.1).collect();
                v.sort_by(f64::total_cmp);
                v.dedup();
                v
            };
            let keys: Vec<Vec<i64>> = each.iter().map(|&(st, x0, d)| vec![step(st), starts.iter().position(|&s| s == x0).unwrap() as i64, step(d)]).collect();
            let (reps, which) = cluster_follow(&keys);
            let vars = reps
                .iter()
                .map(|k| {
                    let (mut stand, x0, mut d) = (k[0] as f64 * STEP_S, starts[k[1] as usize], k[2] as f64 * STEP_S);
                    let mut tries = 0;
                    loop {
                        let mut h = frun.holds[lev].clone();
                        if stand > 0.0 {
                            h.push(Hold { s: a, delay: stand, span: 0.0 });
                        }
                        if d > 0.0 {
                            h.push(Hold { s: x0, delay: d, span: (p1.fat - x0).max(0.0) });
                        }
                        h.sort_by(|x, y| x.s.total_cmp(&y.s).then(x.delay.total_cmp(&y.delay)));
                        let prof = built.prof(net, f, fr as usize, frun, &frun.stop_holds[lev], &h);
                        let over = prof.duration - e0.duration - slack + 1.0;
                        tries += 1;
                        if over <= 0.0 || tries > 4 || stand + d <= 0.0 {
                            break Variant { stop_holds: frun.stop_holds[lev].clone(), holds: h, prof };
                        }
                        let d2 = (d - over).max(0.0);
                        stand = (stand - (over - (d - d2))).max(0.0);
                        d = d2;
                    }
                })
                .collect();
            out.push(Timed { line: f, run: fr as usize, lev, vars, pattern: pattern_of(&deps, &which) });
        }
    }
    out
}

/// One faster train behind the slower train ahead on a stretch with nowhere to pass: a stand at
/// the stretch's start where it reaches it too close, then from `x0` on slower running that loses
/// `delay`, so it is never within the spacing of the slower train at a node of the stretch. `e`:
/// its times at the stretch's start and at each node checked (`xs`, offsets in its run), with no
/// following loss; `s`: the slower train's there. Returns (stand, x0, delay).
fn follow_one(xs: &[f64], e: &[f64], s: &[f64], spacing: f64) -> (f64, f64, f64) {
    let stand = (s[0] + spacing - e[0]).max(0.0);
    let short: Vec<f64> = (0..xs.len()).map(|i| s[i] + spacing - e[i] - stand).collect();
    let Some(first) = (1..xs.len()).find(|&i| short[i] > 0.0) else { return (stand, xs[0], 0.0) };
    let end = xs[xs.len() - 1];
    // slower running from a node before it first comes too close, whichever loses least: it runs
    // at its own speed up to there (a loss spread from x0 to the end is taken as even by distance)
    let mut best = (f64::INFINITY, xs[0]);
    for j in 0..first {
        let x0 = xs[j];
        let d = (j + 1..xs.len()).map(|i| short[i] / ((xs[i] - x0) / (end - x0)).max(1e-6)).fold(0.0, f64::max);
        if d < best.0 {
            best = (d, x0);
        }
    }
    (stand, best.1, best.0)
}

/// Faster runs timetabled together with the one slower run on a stretch with no passing station
/// (`FOLLOW_TIMED`, T-2419): each train timed behind the slower train ahead of it.
fn following(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built) -> Vec<Timed> {
    let cap = &svc.cap;
    // per faster run, its following losses: only a run with one, behind one slower run, is timed
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for r in (0..cap.res.len()).filter(|&r| cap.is_follow(r)) {
        if let Some(o) = cap.users_of(r).iter().find(|u| u.owner) {
            by.entry((o.line, o.run)).or_default().push(r);
        }
    }
    let mut out = vec![];
    for (&(f, fr), rs) in &by {
        let [rr] = rs[..] else { continue };
        let us = cap.users_of(rr);
        let o = *us.iter().find(|u| u.owner).unwrap();
        let slow: Vec<_> = us.iter().filter(|u| !u.owner).collect();
        let [su] = slow[..] else { continue };
        if su.stop != FOLLOW_TIMED {
            continue;
        }
        let fl = &svc.lines[f as usize];
        let sl = &svc.lines[su.line as usize];
        let (Some(flat), Some(slat)) = (lats[f as usize].as_ref(), lats[su.line as usize].as_ref()) else { continue };
        if flat.rest_on != slat.rest_on {
            continue;
        }
        let (frun, srun) = (&fl.runs[fr as usize], &sl.runs[su.run as usize]);
        // a run already timed train by train at a passing station keeps that
        if frun.vars.iter().any(|v| !v.is_empty()) {
            continue;
        }
        let (a, b, spacing) = (o.at, o.occ, su.at);
        // the slower run's offset where the faster run's is `a`: the same track entry
        let Some(i0) = frun.seg_off.iter().position(|&x| x == a) else { continue };
        let js: Vec<usize> = (0..srun.segs.len()).filter(|&j| srun.segs[j] == frun.segs[i0] && (i0 + 1 >= frun.segs.len() || srun.segs.get(j + 1) == frun.segs.get(i0 + 1))).collect();
        let [j0] = js[..] else { continue };
        let delta = srun.seg_off[j0] - a;
        let mut xs: Vec<f64> = vec![a];
        xs.extend(frun.node_s.iter().copied().filter(|&x| x > a + 1.0 && x < b - 1.0));
        xs.push(b);
        let slow_deps = around(slat, su.run as usize);
        let slow_timed = sl.timed.is_some() && sl.timed == Some(slat.timing_key());
        for lev in net.slots() {
            if flat.tph[lev] <= 0.0 || slat.tph[lev] <= 0.0 {
                continue;
            }
            let deps = day_deps(flat, fr as usize, lev);
            if deps.is_empty() {
                continue;
            }
            // its holds without the averaged following loss
            let d_mean = cap.delay[rr][lev];
            let mut holds = frun.holds[lev].clone();
            if d_mean > 0.0 {
                let want = Hold { s: a, delay: d_mean, span: (b - a).max(0.0) };
                if let Some(i) = holds.iter().position(|h| *h == want) {
                    holds.remove(i);
                }
            }
            let e0 = built.prof(net, f, fr as usize, frun, &frun.stop_holds[lev], &holds);
            // the most a train may lose here: its round trip and fleet are sized on the averaged
            // loss, so one losing more than its stand at the far end allows over that (less
            // `TURN_MIN_S`) would miss its next departure and another train be drawn for it; past
            // this it is held behind the slower train on screen (the train guard), as before
            let close = fl.close_stand(net, f);
            let split = net.line_split_of(f)[lev] as f64;
            let at_end = close + if fr == 0 { split } else { 1.0 - split } * svc.spare(net, f as usize, lev);
            let slack = (at_end - TURN_MIN_S.min(close)).max(0.0);
            let room = d_mean + slack - 2.0 * STEP_S;
            let (lo, hi) = window(&deps, e0.duration);
            // the faster train reaching each node, the slower one leaving it
            let e_at: Vec<f64> = xs.iter().map(|&x| reach(&e0, x)).collect();
            // the slower trains as they run (their own variants included), by when they reach the start
            let mut slow: Vec<(f64, Vec<f64>)> = slow_deps
                .iter()
                .filter(|x| x.0 >= lo - srun.prof[x.1 as usize].duration && x.0 < hi)
                .map(|&(d, slev, _, place)| {
                    let var = if slow_timed { srun.var_at(slev as usize, place) } else { 0 };
                    let p = sl.prof_of(su.run as usize, slev as usize, var);
                    (d + reach(p, a + delta), xs.iter().map(|&x| d + leave(p, x + delta)).collect())
                })
                .collect();
            slow.sort_by(|x, y| x.0.total_cmp(&y.0));
            let each: Vec<(f64, f64, f64)> = deps
                .iter()
                .map(|&(d, _, _)| {
                    let e: Vec<f64> = e_at.iter().map(|t| d + t).collect();
                    let i = slow.partition_point(|s| s.0 <= e[0]);
                    if i == 0 {
                        return (0.0, a, 0.0);
                    }
                    let (st, x0, d) = follow_one(&xs, &e, &slow[i - 1].1, spacing);
                    let over = st + d - room.max(0.0);
                    if over <= 0.0 {
                        return (st, x0, d);
                    }
                    let d2 = (d - over).max(0.0);
                    ((st - (over - (d - d2))).max(0.0), x0, d2)
                })
                .collect();
            // told apart by stand and loss, each in whole steps, and where the loss starts
            let starts: Vec<f64> = {
                let mut v: Vec<f64> = each.iter().map(|x| x.1).collect();
                v.sort_by(f64::total_cmp);
                v.dedup();
                v
            };
            let keys: Vec<Vec<i64>> = each.iter().map(|&(st, x0, d)| vec![step(st), starts.iter().position(|&s| s == x0).unwrap() as i64, step(d)]).collect();
            let (reps, which) = cluster_follow(&keys);
            let vars = reps
                .iter()
                .map(|k| {
                    let (mut stand, x0, mut d) = (k[0] as f64 * STEP_S, starts[k[1] as usize], k[2] as f64 * STEP_S);
                    // (a loss spread over a stretch can cost a little more than itself: the cap is
                    // held on the built profile)
                    let mut tries = 0;
                    loop {
                        let mut h = holds.clone();
                        if stand > 0.0 {
                            h.push(Hold { s: a, delay: stand, span: 0.0 });
                        }
                        if d > 0.0 {
                            h.push(Hold { s: x0, delay: d, span: (b - x0).max(0.0) });
                        }
                        h.sort_by(|x, y| x.s.total_cmp(&y.s).then(x.delay.total_cmp(&y.delay)));
                        let prof = built.prof(net, f, fr as usize, frun, &frun.stop_holds[lev], &h);
                        let over = prof.duration - frun.prof[lev].duration - slack + 1.0;
                        tries += 1;
                        if over <= 0.0 || tries > 4 || stand + d <= 0.0 {
                            break Variant { stop_holds: frun.stop_holds[lev].clone(), holds: h, prof };
                        }
                        let d2 = (d - over).max(0.0);
                        stand = (stand - (over - (d - d2))).max(0.0);
                        d = d2;
                    }
                })
                .collect();
            out.push(Timed { line: f, run: fr as usize, lev, vars, pattern: pattern_of(&deps, &which) });
        }
    }
    out
}

/// `cluster` for following keys (stand, where the loss starts, loss): alike ones merge to the
/// larger stand and loss and the earlier start, so each train still keeps its spacing.
fn cluster_follow(each: &[Vec<i64>]) -> (Vec<Vec<i64>>, Vec<usize>) {
    // an earlier start is the larger key's part, as `cluster` merges by the larger part
    let flip: Vec<Vec<i64>> = each.iter().map(|k| vec![k[0], -k[1], k[2]]).collect();
    let (reps, which) = cluster(&flip, |_, _| true);
    (reps.into_iter().map(|k| vec![k[0], -k[1], k[2]]).collect(), which)
}

#[cfg(test)]
mod tests;
