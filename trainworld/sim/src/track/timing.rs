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
//! departures are placed (`retime`), for the clusters of lines timed together whose inputs moved.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use super::capacity::{FOLLOW_PASSING, FOLLOW_TIMED};
use super::net::Network;
use super::overtake::{self, Pass, PASS_CLEAR_S};
use super::params::*;
use super::profile::{self, Hold, Profile};
use super::moves::{self, Move, Moves};
use super::service::{TimingKey, hold_plan, lattice_deps_local, lattice_deps_local_moved, lattice_deps_moved, profile_with_plan, Cal, Dep, Lattice, Run, Service, Variant, NO_PLACE};

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
type Plans = HashMap<(u32, u8, Vec<u64>), Rc<profile::HoldPlan>>;

/// Profiles built this time, and those built last time to take again (`Service::timing_memo`).
struct Built {
    old: Memo,
    new: Memo,
    /// Each run's departures, worked out once for this retime (T-4449): the lattices are fixed
    /// through it, and each of the three passes asks for them again.
    around: HashMap<(u32, u8), Rc<Vec<Dep>>>,
    /// Each line's moves (T-4544), fixed through the retime as the lattices are.
    moves: Vec<Moves>,
    days: HashMap<(u32, u8), Rc<Vec<Vec<(f64, f64, u32)>>>>,
    /// Each run's stand lost to moves per weekday departure, as `days` lays them out (T-4546).
    lost: HashMap<(u32, u8), Rc<Vec<Vec<f64>>>>,
    /// What each run's holds do on its hops, once for all the timetables built with them.
    plans: Plans,
    /// Per run, the parts of those plans the holds have in common.
    plan_memos: HashMap<(u32, u8, Vec<u64>), profile::PlanMemo>,
}

impl Built {
    fn around(&mut self, lat: &Lattice, l: u32, r: usize) -> Rc<Vec<Dep>> {
        let mv = self.moves.get(l as usize).cloned().unwrap_or_default();
        self.around.entry((l, r as u8)).or_insert_with(|| Rc::new(around(lat, &mv, r))).clone()
    }

    /// `day_deps` of every slot.
    fn days(&mut self, lat: &Lattice, l: u32, r: usize) -> Rc<Vec<Vec<(f64, f64, u32)>>> {
        let mv = self.moves.get(l as usize).cloned().unwrap_or_default();
        self.days.entry((l, r as u8)).or_insert_with(|| Rc::new(day_deps(lat, &mv, r))).clone()
    }

    /// `train_lost` of every slot.
    fn lost(&mut self, net: &Network, svc: &Service, lat: &Lattice, l: u32, r: usize) -> Rc<Vec<Vec<f64>>> {
        let mv = self.moves.get(l as usize).cloned().unwrap_or_default();
        self.lost.entry((l, r as u8)).or_insert_with(|| Rc::new(train_lost(net, svc, lat, &mv, l as usize, r))).clone()
    }

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
        let p = match self.old.remove(&k) {
            Some(p) => p,
            None => {
                // the holds' part of a timetable reads the run and the holds only (not the stops' standing)
                let mut pk: Vec<u64> = vec![run.free.duration.to_bits(), run.len.to_bits(), run.free.phases.len() as u64];
                pk.extend(holds.iter().flat_map(|h| [h.s.to_bits(), h.delay.to_bits(), h.span.to_bits()]));
                let memo = self.plan_memos.entry((l, r as u8, pk[..3].to_vec())).or_default();
                let plan = self.plans.entry((l, r as u8, pk)).or_insert_with(|| Rc::new(hold_plan(run, holds, memo))).clone();
                profile_with_plan(net, l, run, stop_holds, &plan)
            }
        };
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

/// Lines `retime` times together: those joined through a wait at a passing station or a following
/// loss (owners, users and the plan's lines), and those resources in their order. Each cluster's
/// variants are worked out from its own lines alone.
struct Cluster {
    lines: Vec<u32>,
    res: Vec<usize>,
}

fn clusters(svc: &Service) -> Vec<Cluster> {
    let cap = &svc.cap;
    let n = svc.lines.len();
    let mut up: Vec<usize> = (0..n).collect();
    fn root(up: &mut [usize], mut x: usize) -> usize {
        while up[x] != x {
            up[x] = up[up[x]];
            x = up[x];
        }
        x
    }
    let mut res = vec![];
    let mut on = vec![false; n];
    for r in (0..cap.res.len()).filter(|&r| cap.wait_point(r).is_some() || cap.is_follow(r)) {
        let mut ls: Vec<usize> = cap.users_of(r).iter().map(|u| u.line as usize).collect();
        ls.extend(cap.wait_point(r).map(|(f, _)| f as usize));
        ls.extend(cap.plans_of(r).iter().flat_map(|p| [p.fast.0 as usize, p.slow.0 as usize]));
        ls.retain(|&l| l < n);
        let Some(&a) = ls.first() else { continue };
        for &b in &ls {
            on[b] = true;
            let (ra, rb) = (root(&mut up, a), root(&mut up, b));
            up[ra.max(rb)] = ra.min(rb);
        }
        res.push((r, a));
    }
    let mut by: BTreeMap<usize, Cluster> = BTreeMap::new();
    for l in (0..n).filter(|&l| on[l]) {
        let r = root(&mut up, l);
        by.entry(r).or_insert_with(|| Cluster { lines: vec![], res: vec![] }).lines.push(l as u32);
    }
    for (r, a) in res {
        let k = root(&mut up, a);
        by.get_mut(&k).unwrap().res.push(r);
    }
    by.into_values().collect()
}

/// Bits of every number in `xs`, folded (FNV-1a over the words).
fn fold(xs: impl Iterator<Item = u64>) -> u64 {
    xs.fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x100_0000_01b3))
}

fn fold_prof(p: &Profile) -> u64 {
    fold(p.phases.iter().flat_map(|x| [x.t, x.s, x.v, x.a]).chain(p.arr.iter().copied()).chain(p.dep.iter().copied()).chain([p.duration, p.dwell, p.hold]).map(f64::to_bits))
}

/// Everything `retime` reads for one cluster, as bits: its waits and following losses with their
/// users, delays and plans, and each line's pattern, dwell, turnaround, stands and timetables. The
/// resources are put in the order of what they hold (`c.res` too, which the passes read in): an
/// edit elsewhere renumbers them, and their order would otherwise move the key (and the last bits
/// of a wait summed from two at one stop).
fn cluster_key(net: &Network, svc: &Service, c: &mut Cluster, lats: &[Option<Lattice>]) -> Vec<u64> {
    let mut chunks: Vec<(Vec<u64>, usize)> = c.res.iter().map(|&r| (res_key(net, svc, r), r)).collect();
    if !check::index_order() {
        chunks.sort();
    }
    c.res = chunks.iter().map(|x| x.1).collect();
    let mut key: Vec<u64> = chunks.into_iter().flat_map(|x| x.0).collect();
    key.push(u64::MAX - 1);
    key.extend(net.period_slot().map(|s| s as u64));
    key.extend(net.slots().map(|s| s as u64));
    for &l in &c.lines {
        let li = l as usize;
        let ls = &svc.lines[li];
        key.push(u64::MAX - 2 - l as u64);
        if let Some(lat) = &lats[li] {
            key.extend(lat.tph.iter().map(|x| x.to_bits() as u64));
            key.extend(lat.off.iter().chain(&lat.phase).chain([&lat.lead, &lat.shift]).map(|x| x.to_bits()));
            key.push(lat.rest_on as u64);
            key.extend(lat.period_slot.map(|s| s as u64));
            key.extend(lat.hours.words());
            key.extend(moves::words(&svc.moves_of(li)));
        }
        key.extend([net.line_dwell[li].to_bits() as u64, net.line_turn[li].to_bits() as u64, ls.close_stand(net, l).to_bits(), ls.is_loop() as u64, ls.one_way as u64]);
        key.extend(net.line_split_of(l).iter().map(|x| x.to_bits() as u64));
        key.extend(net.slots().map(|lev| svc.spare(net, li, lev).to_bits()));
        for run in &ls.runs {
            key.extend([run.free.duration.to_bits(), run.len.to_bits(), run.free.phases.len() as u64, fold_prof(&run.free)]);
            key.extend([fold(run.segs.iter().map(|&x| x as u64)), fold(run.seg_off.iter().chain(&run.node_s).map(|x| x.to_bits()))]);
            for lev in net.slots() {
                key.extend([run.prof[lev].duration.to_bits(), fold_prof(&run.prof[lev])]);
                key.extend(run.stop_holds[lev].iter().map(|x| x.to_bits()));
                key.extend(run.holds[lev].iter().flat_map(|h| [h.s.to_bits(), h.delay.to_bits(), h.span.to_bits()]));
            }
        }
    }
    key
}

/// One wait or following loss as bits: its users, delays and plans.
fn res_key(net: &Network, svc: &Service, r: usize) -> Vec<u64> {
    let cap = &svc.cap;
    let mut key = vec![];
    for u in cap.users_of(r) {
        key.extend([u.line as u64, u.run as u64, u.stop as u64, u.at.to_bits(), u.occ.to_bits(), u.owner as u64]);
    }
    key.extend(cap.delay[r].iter().map(|x| x.to_bits()));
    match cap.wait_point(r) {
        Some((f, k)) => key.extend([1, f as u64, k as u64]),
        None => key.push(2),
    }
    for p in cap.plans_of(r) {
        p.key_into(net, &mut key);
    }
    key
}

/// What a cluster's lines' variants are now: anything else that sets them (a rebuilt line starts
/// with none) shows here, and its cluster is worked out again.
fn state(svc: &Service, lines: &[u32]) -> Vec<u64> {
    let mut out = vec![];
    for &l in lines {
        for run in &svc.lines[l as usize].runs {
            for lev in 0..SLOTS {
                out.extend([run.vars[lev].len() as u64, fold(run.pattern[lev].iter().map(|&(p, v)| (p as u64) << 8 | v as u64))]);
                out.extend(run.vars[lev].iter().map(|v| v.prof.duration.to_bits()));
            }
        }
    }
    out
}

/// Work out the variants again (after an edit has settled and its phases are placed): only for
/// the clusters whose inputs or variants moved since last time (T-4132), so an edit times the
/// lines sharing waits with what it changed and keeps the rest.
pub fn retime(net: &Network, svc: &mut Service) {
    if svc.timing_held.is_some() {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = std::time::Instant::now();
    let n = svc.lines.len();
    let mut cl = clusters(svc);
    let lats: Vec<Option<Lattice>> = {
        let mut on = vec![false; n];
        cl.iter().flat_map(|c| &c.lines).for_each(|&l| on[l as usize] = true);
        (0..n).map(|l| if on[l] { svc.lattice(net, l).map(|x| weekday(&x)) } else { None }).collect()
    };
    let keys: Vec<Vec<u64>> = cl.iter_mut().map(|c| cluster_key(net, svc, c, &lats)).collect();
    let mut redo = vec![false; n];
    let mut states: Vec<Vec<u64>> = cl.iter().map(|c| state(svc, &c.lines)).collect();
    let mut stood: Vec<bool> = cl.iter().enumerate().map(|(i, c)| svc.timing_keys.get(&c.lines[0]).is_some_and(|(k, s)| *k == keys[i] && *s == states[i])).collect();
    if check::full() && stood.contains(&false) {
        stood.iter_mut().for_each(|s| *s = false);
    }
    if !svc.timing_keys.is_empty() {
        check::skip_one(&mut stood);
    }
    for (c, _) in cl.iter().zip(&stood).filter(|x| !*x.1) {
        c.lines.iter().for_each(|&l| redo[l as usize] = true);
    }
    // a line in no cluster has no variants
    let mut in_cl = vec![false; n];
    cl.iter().flat_map(|c| &c.lines).for_each(|&l| in_cl[l as usize] = true);
    for l in (0..n).filter(|&l| !in_cl[l]) {
        let ls = &svc.lines[l];
        redo[l] |= ls.timed.is_some() || ls.runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty()) || r.pattern.iter().any(|p| !p.is_empty()));
    }
    // T-3896: the variants are in the line's phase tables, which the page reads again only for the
    // lines noted as changed: any line timed otherwise than before is noted, edited or not
    let redone: Vec<u32> = (0..n as u32).filter(|&l| redo[l as usize]).collect();
    let before: Vec<Vec<Vec<Variant>>> = redone.iter().map(|&l| svc.lines[l as usize].runs.iter().flat_map(|r| r.vars.iter().cloned()).collect()).collect();
    clear(svc, &redo);
    // the profiles built last time for the lines worked out again, to take again; the rest stay
    let mut memo = std::mem::take(&mut svc.timing_memo);
    let mut old = HashMap::new();
    memo.retain(|k, p| {
        let l = k.0 as usize;
        if l < n && in_cl[l] && !redo[l] {
            return true;
        }
        if l < n && redo[l] {
            old.insert(k.clone(), std::mem::take(p));
        }
        false
    });
    let mut built = Built { old, new: HashMap::new(), moves: svc.moves.clone(), around: HashMap::new(), days: HashMap::new(), lost: HashMap::new(), plans: HashMap::new(), plan_memos: HashMap::new() };
    // the waits and following losses of the clusters worked out again, each cluster's in its key's order
    let res: Vec<usize> = cl.iter().zip(&stood).filter(|x| !*x.1).flat_map(|x| x.0.res.iter().copied()).collect();
    if !res.is_empty() {
        // (before the waits: a slower train waits for the faster ones as they run)
        let lead = lead_in(net, svc, &lats, &mut built, &res);
        install(svc, lead, &lats);
        let passing = passing_waits(net, svc, &lats, &mut built, &res);
        install(svc, passing, &lats);
        // (before the following: a faster line timed behind a slower one reads its trains as they run)
        floor(net, svc, &redo, &lats, &mut built);
        let follow = following(net, svc, &lats, &mut built, &res);
        install(svc, follow, &lats);
    }
    memo.extend(built.new);
    svc.timing_memo = memo;
    // a cluster that stands: its lines timed again where they have variants (an update marks a line
    // it touched untimed)
    for l in (0..n).filter(|&l| in_cl[l] && !redo[l]) {
        if svc.lines[l].runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty())) {
            let mv = svc.moves_of(l);
            svc.lines[l].timed = lats[l].map(|x| x.timing_key(&mv));
        }
    }
    for (i, c) in cl.iter().enumerate().filter(|x| !stood[x.0]) {
        states[i] = state(svc, &c.lines);
    }
    svc.timing_keys = cl.iter().zip(keys).zip(states).map(|((c, k), s)| (c.lines[0], (k, s))).collect();
    let moved: Vec<u32> = redone.iter().zip(&before).filter(|(&l, b)| svc.lines[l as usize].runs.iter().flat_map(|r| r.vars.iter()).ne(b.iter())).map(|x| *x.0).collect();
    svc.note_changed(&moved);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let has = |l: &u32| svc.lines[*l as usize].runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty()));
        let mixed = stood.contains(&false) && cl.iter().zip(&stood).any(|(c, s)| *s && c.lines.iter().any(has));
        check::after(net, svc, t0, cl.len(), stood.iter().filter(|s| !**s).count(), redone.len(), mixed);
    }
}

/// A line's variants, patterns and timing, kept through a trial that times no train (`Service::timing_held`).
#[derive(Clone, Debug)]
pub struct Held {
    runs: Vec<([Vec<Variant>; SLOTS], [Vec<(u32, u8)>; SLOTS])>,
    timed: Option<TimingKey>,
}

/// A line's variants taken off it, None for a line with none.
pub fn take(ls: &mut super::service::LineSvc) -> Option<Held> {
    if !ls.runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty())) {
        return None;
    }
    let runs = ls.runs.iter_mut().map(|r| (std::mem::take(&mut r.vars), std::mem::take(&mut r.pattern))).collect();
    Some(Held { runs, timed: ls.timed.take() })
}

/// The variants a trial's rebuilds took (`Service::timing_held`) back on lines that have none
/// now; the next `retime` keeps them where what they read is as it was, and works the rest out.
pub fn put_back(svc: &mut Service, held: Vec<(u32, Held)>) {
    for (l, h) in held {
        let Some(ls) = svc.lines.get_mut(l as usize) else { continue };
        if ls.runs.iter().any(|r| r.vars.iter().any(|v| !v.is_empty())) {
            continue;
        }
        for (r, (vars, pattern)) in ls.runs.iter_mut().zip(h.runs) {
            (r.vars, r.pattern) = (vars, pattern);
        }
        ls.timed = h.timed;
    }
}

/// Each line's variants and patterns dropped, and the line untimed, where `redo`.
fn clear(svc: &mut Service, redo: &[bool]) {
    for (ls, _) in svc.lines.iter_mut().zip(redo).filter(|x| *x.1) {
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
///
/// T-4546: and a moved train, whose own waits (and its own move later, or the move earlier of the departure its train takes
/// next) would leave it short of that, runs the slot's timetable alone; the rest of the run keeps its variants.
fn floor(net: &Network, svc: &mut Service, redo: &[bool], lats: &[Option<Lattice>], built: &mut Built) {
    for l in (0..svc.lines.len()).filter(|&l| redo[l]) {
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
        if let (Some(lat), false) = (lats[l].as_ref(), built.moves.get(l).map_or(true, |m| m.is_empty())) {
            for r in 0..2 {
                let (days, lost) = (built.days(lat, l as u32, r), built.lost(net, svc, lat, l as u32, r));
                for lev in net.slots() {
                    let ls = &svc.lines[l];
                    let run = &ls.runs[r];
                    let split = net.line_split_of(l as u32)[lev] as f64;
                    let at_end = ls.close_stand(net, l as u32) + if r == 0 { split } else { 1.0 - split } * svc.spare(net, l, lev);
                    let drop: Vec<u32> = days[lev]
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| lost_at(&lost, lev, i) > 0.0)
                        .filter_map(|(i, d)| {
                            let v = run.var_at(lev, d.2);
                            let excess = ls.prof_of(r, lev, v).duration - run.prof[lev].duration;
                            (v != 0 && at_end - excess - lost_at(&lost, lev, i) < need).then_some(d.2)
                        })
                        .collect();
                    if !drop.is_empty() {
                        let run = &mut svc.lines[l].runs[r];
                        run.pattern[lev].retain(|x| !drop.contains(&x.0));
                        if run.pattern[lev].is_empty() {
                            run.vars[lev].clear();
                        }
                    }
                }
            }
        }
        if svc.lines[l].runs.iter().all(|r| r.vars.iter().all(|v| v.is_empty())) {
            svc.lines[l].timed = None;
        }
    }
}

fn install(svc: &mut Service, done: Vec<Timed>, lats: &[Option<Lattice>]) {
    for t in done {
        let mv = svc.moves_of(t.line as usize);
        let ls = &mut svc.lines[t.line as usize];
        ls.runs[t.run].vars[t.lev] = t.vars;
        ls.runs[t.run].pattern[t.lev] = t.pattern;
        ls.timed = lats[t.line as usize].map(|x| x.timing_key(&mv));
    }
}

/// A run's weekday departures over the reference day (game time), per slot, with their
/// headways and places in the day: each place once in its slot, the day being the same every weekday.
fn day_deps(lat: &Lattice, mv: &[Move], r: usize) -> Vec<Vec<(f64, f64, u32)>> {
    let mut buf: Vec<Dep> = vec![];
    lattice_deps_local_moved(lat, mv, r, DAY_REF as f64 * DAY - 3.0 * 3600.0, (DAY_REF + 1) as f64 * DAY, &mut buf);
    let mut seen: Vec<std::collections::HashSet<u32>> = vec![Default::default(); SLOTS];
    let mut out = vec![vec![]; SLOTS];
    for (d, l, hw, place) in buf {
        let lev = l as usize;
        if lev >= SLOTS || place == NO_PLACE || !seen[lev].insert(place) {
            continue;
        }
        out[lev].push((d - lat.shift, hw, place));
    }
    out
}

/// T-4546: per weekday departure of a run, laid out as `day_deps` has them, the seconds of its train's stand at the
/// far end that moves take: its own move later, and the move earlier of the departure that train takes next (the
/// other run's departure nearest where the train would be ready with no move). Empty for a line with no moves.
/// A train's own timetable and `retime`'s waits are fitted into what is left (`lead_in`, `following`, `floor`).
fn train_lost(net: &Network, svc: &Service, lat: &Lattice, mv: &[Move], l: usize, r: usize) -> Vec<Vec<f64>> {
    let none = vec![vec![]; SLOTS];
    let ls = &svc.lines[l];
    if mv.is_empty() || ls.is_loop() || ls.one_way || r > 1 {
        return none;
    }
    let (a, b) = (DAY_REF as f64 * DAY - 3.0 * 3600.0, (DAY_REF + 1) as f64 * DAY);
    // (unmoved local time, move) of a run's departures, as `day_deps` keeps them
    let moves_of = |r: usize| {
        let mut buf: Vec<Dep> = vec![];
        lattice_deps_local(lat, r, a - moves::MAX_MOVE_S, b + moves::MAX_MOVE_S, &mut buf);
        let mut seen: Vec<std::collections::HashSet<u32>> = vec![Default::default(); SLOTS];
        let mut out = vec![vec![]; SLOTS];
        for (d, l, hw, place) in buf {
            let lev = l as usize;
            let m = moves::secs(mv, r, d - lat.shift, hw);
            if lev >= SLOTS || place == NO_PLACE || !(d + m >= a && d + m < b) || !seen[lev].insert(place) {
                continue;
            }
            out[lev].push((d - lat.shift, m));
        }
        out
    };
    let (mine, other) = (moves_of(r), moves_of(1 - r));
    let mut next: Vec<(f64, f64)> = other.into_iter().flatten().collect();
    next.sort_by(|x, y| x.0.total_cmp(&y.0));
    if next.is_empty() {
        return none;
    }
    let run = &ls.runs[r];
    mine.iter()
        .enumerate()
        .map(|(lev, v)| {
            if v.is_empty() {
                return vec![];
            }
            let split = net.line_split_of(l as u32)[lev] as f64;
            let ready = run.prof[lev].duration + ls.close_stand(net, l as u32) + if r == 0 { split } else { 1.0 - split } * svc.spare(net, l, lev);
            v.iter()
                .map(|&(d, m)| {
                    let t = d + ready;
                    let i = next.partition_point(|x| x.0 < t).min(next.len() - 1);
                    let j = if i > 0 && (t - next[i - 1].0).abs() < (next[i].0 - t).abs() { i - 1 } else { i };
                    m.max(0.0) + (-next[j].1).max(0.0)
                })
                .collect()
        })
        .collect()
}

/// A train's lost stand (`train_lost`) in slot `lev`, its `i`th departure there.
fn lost_at(lost: &[Vec<f64>], lev: usize, i: usize) -> f64 {
    lost.get(lev).and_then(|v| v.get(i)).copied().unwrap_or(0.0)
}

/// A run's departures around the reference day, every slot, game time.
fn around(lat: &Lattice, mv: &[Move], r: usize) -> Vec<Dep> {
    let mut buf = vec![];
    lattice_deps_moved(lat, mv, r, DAY_REF as f64 * DAY - AROUND_S, (DAY_REF + 1) as f64 * DAY + AROUND_S, &mut buf);
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
/// on the slot's timetable, `floor`). `ok(key, is)`: whether every departure in `is` may run that
/// key's timetable; a merge every departure of may is taken first, and only where none is left one
/// that may not.
fn cluster(each: &[Vec<i64>], ok: impl Fn(&[i64], &[usize]) -> bool) -> (Vec<Vec<i64>>, Vec<usize>) {
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
        add(a) + add(b) + if ok(&m, &a.1) && ok(&m, &b.1) { 0.0 } else { UNSAFE }
    };
    if keys.len() <= MAX_VARS {
        let mut which = vec![0; each.len()];
        for (k, key) in keys.iter().enumerate() {
            key.1.iter().for_each(|&i| which[i] = k);
        }
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
        if let Some(k) = (0..reps.len()).filter(|&k| reps[k].iter().zip(e).all(|(r, x)| r >= x) && ok(&reps[k], &[i])).min_by_key(|&k| add(k)) {
            if add(k) < add(which[i]) || !ok(&reps[which[i]], &[i]) {
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
fn passing_waits(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built, res: &[usize]) -> Vec<Timed> {
    let cap = &svc.cap;
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for &r in res {
        if cap.wait_point(r).is_some() {
            if let Some(o) = cap.users_of(r).iter().find(|u| u.owner) {
                by.entry((o.line, o.run)).or_default().push(r);
            }
        }
    }
    let mut out = vec![];
    for (&(l, r), ws) in &by {
        let (ls, Some(lat)) = (&svc.lines[l as usize], lats[l as usize].as_ref()) else { continue };
        // a pair where one line runs its own weekends is left averaged: its weekend trains would
        // meet trains the variants were not worked out against
        if ws.iter().any(|&w| cap.wait_point(w).is_some_and(|(f, _)| lats[f as usize].as_ref().map_or(true, |x| x.rest_on != lat.rest_on))) {
            continue;
        }
        let run = &ls.runs[r as usize];
        let days = built.days(lat, l, r as usize);
        for lev in net.slots() {
            let deps = &days[lev];
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
                let cap_h = plan.headway_cap(net);
                hcap = hcap.min(cap_h);
                let fr = plan.fast.1 as usize;
                let k = k as usize;
                let (pt, q, nb) = (plan.points[k], plan.points[k + 1], plan.points[plan.next_passing(k)]);
                let s_leave = b.dep[o.stop as usize];
                let run_s = reach(&b, nb.at) - s_leave;
                let s_to_q = reach(&b, q.at) - s_leave;
                let fdeps = built.around(flat, f, fr);
                let passes = stations.entry(o.stop as usize).or_default();
                for &(d, flev, hf, place) in fdeps.iter().filter(|x| x.0 >= lo - fl.runs[fr].prof[x.1 as usize].duration && x.0 < hi) {
                    // each faster train as it runs (`lead_in`)
                    let fp = fl.prof_of(fr, flev as usize, fl.runs[fr].var_at(flev as usize, place));
                    let x = leave(fp, pt.fat);
                    let lead = PASS_CLEAR_S.max(reach(fp, q.fat) - x + plan.spacing - s_to_q);
                    passes.push(Pass { at: d + x, lead, run_f: reach(fp, nb.fat) - x, run_s, spacing: plan.spacing, most: hf.min(cap_h) + lead });
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
            let (reps, which) = cluster(&each, |k, is| {
                let k = secs(k);
                is.iter().all(|&i| overtake::conflicts(times[i].0, &stations, &k) & !own[i] == 0)
            });
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
fn lead_in(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built, res: &[usize]) -> Vec<Timed> {
    let cap = &svc.cap;
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for &r in res.iter().filter(|&&r| cap.is_follow(r)) {
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
        let slow_deps = built.around(slat, su.line, su.run as usize);
        let days = built.days(flat, f, fr as usize);
        let lost = built.lost(net, svc, flat, f, fr as usize);
        for lev in net.slots() {
            if flat.tph[lev] <= 0.0 || slat.tph[lev] <= 0.0 {
                continue;
            }
            let deps = &days[lev];
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
            // (a moved train's room is what its move leaves: `train_lost`)
            let each: Vec<(f64, f64, f64)> = deps
                .iter()
                .enumerate()
                .map(|(n, &(d, _, _))| {
                    let e: Vec<f64> = e_at.iter().map(|t| d + t).collect();
                    let i = slow.partition_point(|s| s.0 <= e[0]);
                    if i == 0 {
                        return (0.0, a, 0.0);
                    }
                    let (st, x0, dl) = follow_one(&xs, &e, &slow[i - 1].1, plan.spacing);
                    let over = st + dl - (room - step(lost_at(&lost, lev, n)) as f64 * STEP_S).max(0.0);
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
            let keys: Vec<Vec<i64>> = each.iter().enumerate().map(|(n, &(st, x0, d))| vec![step(st), starts.iter().position(|&s| s == x0).unwrap() as i64, step(d), step(lost_at(&lost, lev, n))]).collect();
            let (reps, which) = cluster_follow(&keys);
            let vars = reps
                .iter()
                .map(|k| {
                    let (mut stand, x0, mut d) = (k[0] as f64 * STEP_S, starts[k[1] as usize], k[2] as f64 * STEP_S);
                    let slack = (slack - k[3] as f64 * STEP_S).max(0.0);
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
fn following(net: &Network, svc: &Service, lats: &[Option<Lattice>], built: &mut Built, res: &[usize]) -> Vec<Timed> {
    let cap = &svc.cap;
    // per faster run, its following losses: only a run with one, behind one slower run, is timed
    let mut by: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for &r in res.iter().filter(|&&r| cap.is_follow(r)) {
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
        let slow_deps = built.around(slat, su.line, su.run as usize);
        let days = built.days(flat, f, fr as usize);
        let lost = built.lost(net, svc, flat, f, fr as usize);
        let slow_timed = sl.timed.is_some() && sl.timed == Some(slat.timing_key(&svc.moves_of(su.line as usize)));
        for lev in net.slots() {
            if flat.tph[lev] <= 0.0 || slat.tph[lev] <= 0.0 {
                continue;
            }
            let deps = &days[lev];
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
                .enumerate()
                .map(|(n, &(d, _, _))| {
                    let e: Vec<f64> = e_at.iter().map(|t| d + t).collect();
                    let i = slow.partition_point(|s| s.0 <= e[0]);
                    if i == 0 {
                        return (0.0, a, 0.0);
                    }
                    let (st, x0, d) = follow_one(&xs, &e, &slow[i - 1].1, spacing);
                    let over = st + d - (room - step(lost_at(&lost, lev, n)) as f64 * STEP_S).max(0.0);
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
            let keys: Vec<Vec<i64>> = each.iter().enumerate().map(|(n, &(st, x0, d))| vec![step(st), starts.iter().position(|&s| s == x0).unwrap() as i64, step(d), step(lost_at(&lost, lev, n))]).collect();
            let (reps, which) = cluster_follow(&keys);
            let vars = reps
                .iter()
                .map(|k| {
                    let (mut stand, x0, mut d) = (k[0] as f64 * STEP_S, starts[k[1] as usize], k[2] as f64 * STEP_S);
                    let slack = (slack - k[3] as f64 * STEP_S).max(0.0);
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
    // the fourth part, a moved train's lost stand (T-4546), is the room its variant is built with: a merge
    // of unlike ones is the last resort
    let flip: Vec<Vec<i64>> = each.iter().map(|k| vec![k[0], -k[1], k[2], k[3]]).collect();
    let (reps, which) = cluster(&flip, |m, is| is.iter().all(|&i| each[i][3] == m[3]));
    (reps.into_iter().map(|k| vec![k[0], -k[1], k[2], k[3]]).collect(), which)
}

pub mod check;

#[cfg(test)]
mod tests;
