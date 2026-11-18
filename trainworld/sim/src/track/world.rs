//! Edit operations with inverses, undo and redo, and what each edit dirties (SPEC 6.5,
//! notes/T-008.md 7). `TrackWorld` is what the clock worker owns: the network, the service
//! derived from it, and the undo history.

use super::capacity::ResKind;
use super::net::{path_edge, weekend_headway, EdgeData, Issue, LineData, Network, NodeData, StationData, Weekend};
use super::params::*;
use super::service::{Service, ServiceDirty};
use super::twin::Twin;

/// A line's trains, rates and stop-to-stop times (by run, then level), as demand reads them.
type DemandView = (Option<[u32; 3]>, [f32; 3], [Vec<Vec<f64>>; 2]);

/// One edit. The primitive setters replace an object whole (`None` removes it); `Split` and
/// `DeleteEdge` expand into primitives when applied. Every applied op yields its inverse, which
/// is again an `Op`.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Node { id: u32, data: Option<NodeData> },
    Edge { id: u32, data: Option<EdgeData> },
    Station { node: u32, data: Option<StationData> },
    Line { id: u32, data: Option<LineData> },
    /// Tests only: a line at these trains per hour, without trains; the next settle allocates the
    /// whole trains that rate needs (T-1146).
    #[cfg(test)]
    Schedule { id: u32, tph: [f32; 3] },
    TrainCounts { id: u32, trains: Option<[u32; 3]>, tph: [f32; 3] },
    /// T-625: set a line's headways in seconds per level (`None`: counts set by hand) with the
    /// counts it starts from; the settle then keeps the counts to the headways.
    AutoTrains { id: u32, auto: Option<[u32; 3]>, trains: Option<[u32; 3]> },
    /// T-3521: set a line's weekend trains (`LineData::weekend`).
    Weekend { id: u32, weekend: Weekend },
    /// Split `edge` at chainage `s` with a new node `node`; the far part becomes `new_edge`.
    /// Line paths through it are rewritten. Geometry does not change.
    Split { edge: u32, s: f64, node: u32, new_edge: u32 },
    /// Remove an edge; each line using it is rerouted by the fastest path or left broken.
    DeleteEdge { id: u32 },
    Batch(Vec<Op>),
}

/// Trains off its fleet a settle stops at: a line within it is settled.
const SETTLED: f64 = 0.0001;
/// Most rounds of one settle.
const SETTLE_ROUNDS: usize = 80;
/// T-3233: the longest repeat a settle looks for.
const CYCLE_MAX: usize = 4;
/// Within this many trains of its fleet, a line whose shortfall keeps turning is settled (T-1283).
const SWING_NEAR: f64 = 0.05;
/// Turns that do not shrink the shortfall by half before it counts as circling (T-1283).
const SWING_TURNS: u8 = 3;

/// Most rounds of "count from round trip, settle" a headway line's counts get per edit (T-625).
const AUTO_ROUNDS: usize = 5;
/// T-2973: count lines at a merge run one headway when theirs are within this share of the longest
/// (Anita, 2026-11-10: "3% seems like a reasonable cap"), and the pacing settles in at most this
/// many rounds.
const PACE_SHARE: f64 = 0.03;
/// T-2973: and keep the longest's headway this much over it (as a headway line may run 0.5% over).
const PACE_SLACK: f64 = 0.002;
/// T-2973: a merge this full is shared out evenly.
const PACE_FULL_RHO: f64 = 0.98;
const PACE_ROUNDS: usize = 3;
/// T-3388: a count line's headway moving by no more than this share keeps its merge's pacing, while
/// the line can still run its pace. A line on the edge of a set moves its own round trip by about
/// 0.2% as it joins or leaves it.
const PACE_HOLD: f64 = 0.005;

/// T-3388: what one line's pacing at one level was last worked out from (`TrackWorld::pace_in`):
/// its headway (0 when it took no part), whether its merge was full, its group (least line, line
/// count), the set it was put in (the set's longest line), and whether it was paced in this
/// settle, so its headway has not yet moved with its own pacing.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PaceIn {
    h: f64,
    full: bool,
    group: (u32, u32),
    set: u32,
    fresh: bool,
}
const NO_SET: u32 = u32::MAX;
impl Default for PaceIn {
    fn default() -> Self {
        PaceIn { h: 0.0, full: false, group: (NO_SET, 0), set: NO_SET, fresh: false }
    }
}

/// How far over the chosen headway the real one may run: a line's round trip moves a little with
/// its own train count, and a train more for a few seconds would be noise (T-625).
pub(crate) const AUTO_SLACK: f64 = 1.005;

/// The fewest whole trains that keep a headway of `headway_s` over a round trip of `trip_s`
/// (T-625); a headway of 0 is no service.
pub fn auto_trains(trip_s: f64, headway_s: u32) -> u32 {
    if headway_s == 0 || !(trip_s > 0.0) { return 0; }
    ((trip_s / (headway_s as f64 * AUTO_SLACK) - 1e-6).ceil().max(0.0) as u32).min(9999)
}

/// T-679: `want` trains for a headway of `headway_s`, clamped to the whole trains that fit when the
/// line's track takes at most `most_tph` trains an hour and the headway is at that limit or within
/// `AUTO_SLACK` of it: rounding up would cross the limit and hold a train out. At least one train
/// runs. A headway well under the limit keeps its count, so the trains that do not fit wait at the
/// ends and the panel warns.
pub fn clamp_auto(want: u32, trip_s: f64, headway_s: u32, most_tph: f64) -> u32 {
    if want == 0 || !(most_tph > 0.0) || !most_tph.is_finite() || headway_s as f64 * AUTO_SLACK < 3600.0 / most_tph {
        return want;
    }
    want.min(((most_tph * trip_s / 3600.0 + 1e-6).floor() as u32).max(1))
}

/// What an edit touched (SPEC 6.5).
#[derive(Clone, Debug, Default)]
pub struct Dirty {
    pub edges: Vec<u32>,
    pub nodes: Vec<u32>,
    /// Positions of crossings removed or added, and their tram tracks (`Network::crossing_trams`).
    pub crossings: Vec<[f64; 3]>,
    /// Resources (indices into `svc.cap.res`, valid until the next edit) on touched track or used
    /// by a directly changed line.
    pub resources: Vec<u32>,
    /// Lines whose path, stops, geometry or schedule changed.
    pub lines: Vec<u32>,
    pub service: ServiceDirty,
}

#[derive(Clone, Debug)]
pub struct EditResult {
    pub dirty: Dirty,
    /// Build cost of the whole network (blueprint included) before and after, US$M.
    pub cost_before: f64,
    pub cost_after: f64,
    /// Cost of what is constructed before and after (`Network::built_cost`).
    pub built_before: f64,
    pub built_after: f64,
    /// What the edit costs: the increase in constructed cost. Blueprint edits cost nothing and
    /// removing constructed track refunds nothing (SPEC 6.4, Anita 2026-10-09).
    pub charge: f64,
}

pub struct TrackWorld {
    pub net: Network,
    pub svc: Service,
    undo: Vec<Op>,
    redo: Vec<Op>,
    /// T-752: a line switching between counts and headways, whose counts this edit's settle keeps
    switching: Option<u32>,
    /// T-1863: the lines an edit's settle may step: those sharing track, directly or through others,
    /// with what the edit changed. None settles every line off its fleet (a load, a bulk edit).
    settle_scope: Option<Vec<bool>>,
    /// T-3521: the weekend service (`settle_weekend`) where weekends were settled on their own; not
    /// saved. None: worked out again from the weekdays' at the next weekend settle.
    svc_rest: Option<Box<Service>>,
    /// T-3521: a load keeps the save's weekend rates as they are (`reload`), to the bit, and the
    /// weekend service is worked out at the first edit that settles them.
    weekend_as_saved: bool,
    /// T-3388: per line and level, what `pace_merges` last paced it from. Not saved: a load takes
    /// the saved pacing as it is. Weekdays' only: the weekends run the weekdays' pacing.
    pace_in: Vec<[PaceIn; LEVELS]>,
}

/// Why an edit that changes what is constructed was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ChargeError {
    Issues(Vec<Issue>),
    /// The edit costs `need` US$M and there is only `have`.
    Money { need: f64, have: f64 },
}

impl TrackWorld {
    pub fn new(net: Network) -> TrackWorld {
        let mut w = TrackWorld { net, svc: Service::default(), undo: vec![], redo: vec![], switching: None, settle_scope: None, svc_rest: None, weekend_as_saved: false, pace_in: vec![] };
        w.reload();
        w
    }

    /// Derive everything from the inputs (after a load or a bulk build).
    pub fn reload(&mut self) {
        self.svc.changed_all = true;
        self.svc_rest = None;
        self.net.touch_all();
        let t = self.net.take_touch();
        self.net.derive(&t);
        self.svc.update_all(&self.net);
        self.weekend_as_saved = self.net.rates_from_save && self.net.rest_solved;
        self.pace_in.clear();
        if self.net.pace_from_save {
            // T-3233: the save's rates were settled with its pacing; pacing again would move them
            self.settle_unpaced(false);
            self.pace_merges(true);
            self.settle_weekend(false);
        } else {
            self.settle_train_counts(!self.net.rates_from_save);
        }
        self.weekend_as_saved = false;
        // T-1925: a network without phases (an older save, a new world) has every line placed once,
        // in id order; a reload for new water keeps them
        if !self.net.phases_placed {
            let all: Vec<u32> = (0..self.net.line_count() as u32).collect();
            self.place_departures(&all);
            self.pin_twin_phases();
            self.net.phases_placed = true;
        }
    }

    /// T-1925: place `lines`' departures (`spacing::place`) into `Network::line_phase`, and where
    /// their spare time stands into `line_split` (T-2972).
    fn place_departures(&mut self, lines: &[u32]) {
        let mut phase = std::mem::take(&mut self.net.line_phase);
        let mut split = std::mem::take(&mut self.net.line_split);
        super::spacing::place(&self.net, &self.svc, lines, &mut phase, &mut split);
        self.net.line_phase = phase;
        self.net.line_split = split;
    }

    /// The bus lines timed again (their street speeds changed, T-1577); nothing else is touched.
    pub fn rework_buses(&mut self) {
        let buses = self.net.bus_lines();
        if buses.is_empty() {
            return;
        }
        self.svc.update(&self.net, &buses, &[], false);
        self.svc_rest = None;
        self.settle_train_counts(!self.net.rates_from_save);
    }

    /// Solve headways from a fixed whole-line fleet and the capacity-adjusted round trip.
    /// A positive derivative bound damps feedback from shared resources.
    /// Runs on edits only, with a bounded number of global capacity solves.
    /// Levels share no capacity, so each level settles on its own and a level already within
    /// tolerance is left alone to the bit: demand keeps its periods only then (T-124, T-415).
    /// T-1146: a line a loaded save left with no train allocation (a frequency-scheduled line)
    /// takes the whole trains its trains per hour need over the round trip the service has worked
    /// out for it (`LineSvc::fleet`), then the counts settle as for any line. A broken line takes
    /// none. Nothing in the game makes such a line any more; this is the one way out of one.
    /// Returns whether any line was given trains.
    pub fn allocate_unallocated(&mut self) -> bool {
        let mut any = false;
        for l in 0..self.net.line_count() {
            if !self.net.line_ok(l as u32) || self.net.line_trains[l].is_some() { continue; }
            self.net.line_trains[l] = Some(self.svc.lines.get(l).filter(|s| s.ok).map_or([0; 3], |s| s.trains));
            any = true;
        }
        if any { self.settle_train_counts(true); }
        any
    }

    /// `full`: every line of an unsettled level is stepped, as a load without saved rates needs;
    /// otherwise (an edit) only the lines off their fleet are (T-1787). Then count lines meeting at
    /// a merge are put in step (T-2973) and settled again, until they hold.
    fn settle_train_counts(&mut self, full: bool) -> Vec<u32> {
        let mut affected = self.settle_unpaced(full);
        for _ in 0..PACE_ROUNDS {
            if !self.pace_merges(false) {
                break;
            }
            affected.extend(self.settle_unpaced(full));
        }
        self.pace_in.iter_mut().flatten().for_each(|p| p.fresh = false);
        affected.extend(self.settle_weekend(full));
        affected.sort_unstable();
        affected.dedup();
        affected
    }

    /// T-1863: `seeds` and every line that shares a capacity resource with one of them, or with such a
    /// line in turn.
    fn sharing_closure(&self, seeds: &[u32]) -> Vec<bool> {
        let n = self.net.line_count();
        let mut hit = vec![false; n];
        for &l in seeds {
            if (l as usize) < n { hit[l as usize] = true; }
        }
        let cap = &self.svc.cap;
        loop {
            let mut grew = false;
            for r in 0..cap.res.len() {
                let us = cap.users_of(r);
                if us.iter().any(|u| (u.line as usize) < n && hit[u.line as usize]) {
                    for u in us {
                        if (u.line as usize) < n && !hit[u.line as usize] {
                            hit[u.line as usize] = true;
                            grew = true;
                        }
                    }
                }
            }
            if !grew { break; }
        }
        hit
    }

    /// T-2973: count lines (a whole-line fleet) meeting at a merge, with or through each other,
    /// whose headways (round trip over trains) are within `PACE_SHARE` of the longest of them, all
    /// run that one: the faster stand the difference at their ends, as a headway line does, so the
    /// lines take even turns at the merge all day instead of drifting past each other. A headway
    /// line there counts with its set headway and is never moved. Into `Network::line_pace`; true
    /// when it changed.
    ///
    /// T-3388: a merge's lines at one level are paced again only when what they were paced from
    /// has changed: a line's headway by more than `PACE_HOLD` or past its pace, a line joining or
    /// leaving, the merge filling or emptying (`pace_in`). Then the lines that have not moved keep
    /// their sets and a set its pace where it still fits. Otherwise each pacing slows the lines a
    /// little and moves the next one's headways and set edges, and London's 130 lines at one merge
    /// never settle. `adopt`: take the pacing as it stands (a load's) and only note what it was
    /// paced from.
    fn pace_merges(&mut self, adopt: bool) -> bool {
        let n = self.net.line_count();
        let mut root: Vec<usize> = (0..n).collect();
        fn top(root: &mut [usize], mut x: usize) -> usize {
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        let cap = &self.svc.cap;
        let mut met = vec![false; n];
        for r in 0..cap.res.len() {
            if cap.wire_kind(r) != 12 {
                continue;
            }
            let us = cap.users_of(r);
            let Some(first) = us.first() else { continue };
            for u in us {
                if u.line != first.line && (u.line as usize) < n && (first.line as usize) < n {
                    let (a, b) = (top(&mut root, first.line as usize), top(&mut root, u.line as usize));
                    root[a] = b;
                    met[u.line as usize] = true;
                    met[first.line as usize] = true;
                }
            }
        }
        // the groups with a full merge, by level
        let mut full: Vec<(usize, usize)> = vec![];
        for r in 0..cap.res.len() {
            if cap.wire_kind(r) != 12 {
                continue;
            }
            if let Some(u) = cap.users_of(r).first().filter(|u| (u.line as usize) < n) {
                for lev in (0..LEVELS).filter(|&lev| cap.rho[r][lev] >= PACE_FULL_RHO) {
                    full.push((top(&mut root, u.line as usize), lev));
                }
            }
        }
        let was: Vec<[f32; LEVELS]> = (0..n).map(|l| self.net.line_pace_of(l as u32)).collect();
        let mut pace = vec![[0.0f32; LEVELS]; n];
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for l in (0..n).filter(|&l| met[l]) {
            let g = top(&mut root, l);
            groups.entry(g).or_default().push(l);
        }
        self.pace_in.resize(n, [PaceIn::default(); LEVELS]);
        for lines in groups.values() {
            let g = top(&mut root, lines[0]);
            // lines are pushed in id order, so the first is the group's least
            let group = (lines[0] as u32, lines.len() as u32);
            for lev in 0..LEVELS {
                let is_full = full.contains(&(g, lev));
                // (headway, line, set by the player, the set it was in when last paced: NO_SET
                // when it has moved since); a count line's headway within `PACE_HOLD` of the one it
                // was last paced from, and no longer than its pace, counts as that one
                let mut hs: Vec<(f64, usize, bool, u32)> = vec![];
                for &l in lines {
                    let ls = &self.svc.lines[l];
                    if !self.net.line_ok(l as u32) || !ls.ok || !ls.running || ls.is_loop() || ls.is_bus() {
                        continue;
                    }
                    let last = self.pace_in[l][lev];
                    if let Some(a) = self.net.line_auto[l] {
                        if a[lev] > 0 {
                            let h = a[lev] as f64;
                            hs.push((h, l, true, if last.h == h { last.set } else { NO_SET }));
                        }
                    } else if let Some(c) = self.net.line_trains[l] {
                        if c[lev] > 0 && ls.round_trip[lev] > 0.0 {
                            let h = ls.round_trip[lev] / c[lev] as f64;
                            let keeps = was[l][lev] == 0.0 || h <= was[l][lev] as f64;
                            if !adopt && last.fresh && last.h > 0.0 && keeps {
                                // just paced: its headway moved with its own pacing, which is what
                                // it is now paced from
                                self.pace_in[l][lev].h = h;
                                self.pace_in[l][lev].fresh = false;
                                hs.push((h, l, false, last.set));
                            } else if !adopt && last.h > 0.0 && keeps && (h - last.h).abs() <= PACE_HOLD * last.h {
                                hs.push((last.h, l, false, last.set));
                            } else {
                                hs.push((h, l, false, NO_SET));
                            }
                        }
                    }
                }
                let h_of = |l: usize| hs.iter().find(|x| x.1 == l).map_or(0.0, |x| x.0);
                let same = lines.iter().all(|&l| {
                    let p = self.pace_in[l][lev];
                    p.h == h_of(l) && p.full == is_full && p.group == group
                });
                if adopt {
                    // the sets the loaded pacing has: count lines at one pace, and a headway line
                    // whose headway is that pace
                    let mut set_of: std::collections::HashMap<u32, u32> = Default::default();
                    for x in hs.iter().filter(|x| !x.2 && was[x.1][lev] > 0.0) {
                        set_of.entry(was[x.1][lev].to_bits()).or_insert(x.1 as u32);
                    }
                    for &l in lines {
                        self.pace_in[l][lev] = PaceIn { h: 0.0, full: is_full, group, set: NO_SET, fresh: false };
                    }
                    for x in &hs {
                        let key = if x.2 { (x.0 as f32).to_bits() } else { was[x.1][lev].to_bits() };
                        let set = set_of.get(&key).copied().unwrap_or(x.1 as u32);
                        self.pace_in[x.1][lev] = PaceIn { h: x.0, full: is_full, group, set, fresh: false };
                    }
                }
                if adopt || same {
                    for &l in lines {
                        pace[l][lev] = was[l][lev];
                    }
                    continue;
                }
                for &l in lines {
                    self.pace_in[l][lev] = PaceIn { h: 0.0, full: is_full, group, set: NO_SET, fresh: false };
                }
                hs.sort_by(|a, b| b.0.total_cmp(&a.0));
                let mut i = 0;
                while i < hs.len() {
                    let top_h = hs[i].0;
                    // a line that has not moved stays out of a set it was not in: the sets' edges
                    // stay where they were, or one line moving shifts every edge below it
                    let mut set = hs[i].3;
                    let mut j = i + 1;
                    while j < hs.len() && hs[j].0 * (1.0 + PACE_SHARE) >= top_h {
                        if hs[j].3 != NO_SET {
                            if set == NO_SET {
                                set = hs[j].3;
                            } else if set != hs[j].3 {
                                break;
                            }
                        }
                        j += 1;
                    }
                    // every count line of the set, the slowest too, a little over the longest, so a
                    // round trip drifting by a second as the others' holds move keeps them in step
                    let mut keep = if hs[i].2 { top_h } else { top_h * (1.0 + PACE_SLACK) };
                    // a set that ran one pace keeps it while every line can run it and it is
                    // within `PACE_HOLD` of the new one
                    let old = was[hs[i].1][lev] as f64;
                    if !hs[i].2 && old >= top_h && old <= keep * (1.0 + PACE_HOLD) && (i..j).all(|k| !hs[k].2 && was[hs[k].1][lev] as f64 == old) {
                        keep = old;
                    }
                    for k in i..j {
                        if !hs[k].2 && j - i > 1 {
                            pace[hs[k].1][lev] = keep as f32;
                        }
                        self.pace_in[hs[k].1][lev] = PaceIn { h: hs[k].0, full: is_full, group, set: hs[i].1 as u32, fresh: true };
                    }
                    i = j;
                }
                // a full merge is shared out evenly between the count lines through it: each that
                // could run more often runs the even share of what they run together, so the full
                // track's queue is not all one line's (T-2960 saw 3:39 and 7:02 for two alike)
                if is_full {
                    let count: Vec<(usize, f64)> = hs.iter().filter(|x| !x.2).map(|x| (x.1, x.0)).collect();
                    let rate: f64 = count.iter().map(|&(l, _)| self.net.line_tph[l][lev] as f64).sum();
                    if count.len() > 1 && rate > 0.0 {
                        // `rate` is read after the pacing already slowed these lines, so the even
                        // headway is the one at which the lines, each no faster than its own, run
                        // that much together: the same one again on the next edit, where 3600 k / rate
                        // grew by what the lines left at their own headway ran under it (T-3252)
                        let (mut lo, mut hi) = (0.0f64, count.iter().map(|x| x.1).fold(3600.0 * count.len() as f64 / rate, f64::max));
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            let together: f64 = count.iter().map(|&(l, h)| {
                                let own = 3600.0 / h.max(mid);
                                // a line not paced yet may run less than its own headway, for the track's sake
                                if self.net.line_pace_of(l as u32)[lev] > 0.0 { own } else { own.min(self.net.line_tph[l][lev] as f64) }
                            }).sum();
                            if together > rate { lo = mid } else { hi = mid }
                        }
                        let even = hi;
                        for &(l, h) in &count {
                            if h < even * (1.0 - PACE_SLACK) {
                                pace[l][lev] = pace[l][lev].max(even as f32);
                            }
                        }
                    }
                }
            }
        }
        if adopt || was == pace {
            return false;
        }
        self.net.line_pace = pace;
        true
    }

    /// The settle without `pace_merges`.
    fn settle_unpaced(&mut self, full: bool) -> Vec<u32> {
        if !self.net.line_auto.iter().any(|a| a.is_some()) {
            return self.settle_counts(full);
        }
        // T-625: the counts of headway lines follow their round trips, which follow the counts
        // through shared track, so settle and ask again until they hold still (a few rounds).
        let mut affected = vec![];
        let first = self.net.line_trains.clone();
        for round in 0..AUTO_ROUNDS {
            let moved = self.auto_counts(round + 1 == AUTO_ROUNDS);
            affected.extend(self.settle_counts(full));
            if !moved && round > 0 { break; }
        }
        // counts the full track would keep waiting at the ends are not bought
        if self.trim_auto() { affected.extend(self.settle_counts(full)); }
        affected.extend((0..self.net.line_count() as u32).filter(|&l| self.net.line_trains[l as usize] != first[l as usize]));
        affected.sort_unstable(); affected.dedup(); affected
    }

    /// T-625: each headway line's counts, the fewest trains that keep its headway over the round
    /// trip as it stands. `last`: no further rounds follow, so only counts that come down are taken
    /// (a count that keeps rising round after round is on shared track that fills up).
    fn auto_counts(&mut self, last: bool) -> bool {
        let mut moved = false;
        let most = self.svc.cap.line_limits(&self.net);
        for l in 0..self.net.line_count() {
            let (Some(auto), Some(old)) = (self.net.line_auto[l], self.net.line_trains[l]) else { continue };
            if !self.net.line_ok(l as u32) || !self.svc.lines[l].ok { continue; }
            // T-752: a switch to headways keeps the trains the line has; T-849: a level with none
            // takes its headway's count, or a new line switched straight to headways never runs
            let switching = self.switching == Some(l as u32);
            let trip = self.svc.lines[l].round_trip;
            // a loop's two directions each keep the headway with half its trains (T-798)
            let each = if self.svc.lines[l].paired() { 2 } else { 1 };
            let mut counts = old;
            for lev in (0..LEVELS).filter(|&lev| !switching || old[lev] == 0) {
                let t = trip[lev] / each as f64;
                let want = each * clamp_auto(auto_trains(t, auto[lev]), t, auto[lev], most[l][lev].0);
                if want < old[lev] || !last { counts[lev] = want; }
            }
            if counts != old {
                self.net.line_trains[l] = Some(counts);
                moved = true;
            }
        }
        moved
    }

    /// T-625: drop from each headway line the whole trains that wait at the ends because the track
    /// they use is full (`short` as in `TrackApi::waiting`). Over-full track (other lines' rates
    /// past its capacity) keeps them, held out of service (T-752).
    fn trim_auto(&mut self) -> bool {
        let mut trimmed = false;
        let worst = self.svc.cap.bottlenecks(self.net.line_count());
        for l in 0..self.net.line_count() {
            let (Some(_), Some(mut counts)) = (self.net.line_auto[l], self.net.line_trains[l]) else { continue };
            if !self.net.line_ok(l as u32) || !self.svc.lines[l].ok || self.switching == Some(l as u32) { continue; }
            let trip = self.svc.lines[l].round_trip;
            for lev in (0..LEVELS).filter(|&lev| worst[l][lev].0 <= 2.0 - FULL_RHO) {
                let short = self.net.running_trains(l, lev, trip[lev]) - self.net.line_tph[l][lev] as f64 * trip[lev] / 3600.0;
                let mut held = if short > 0.01 { short.floor() as u32 } else { 0 };
                if self.svc.lines[l].paired() { held &= !1; }
                if held > 0 { counts[lev] -= held.min(counts[lev]); trimmed = true; }
            }
            self.net.line_trains[l] = Some(counts);
        }
        trimmed
    }

    /// T-3521: a line's whole trains on weekends (`LineData::weekend`) where they differ from its
    /// weekday counts; None where it runs those. A share scales the counts up to whole trains, or a
    /// headway line's headways (`weekend_headway`) over its weekday round trip.
    pub fn weekend_counts(&self, l: usize) -> Option<[u32; 3]> {
        let mut most = None;
        self.weekend_counts_in(l, &mut most)
    }

    /// `weekend_counts` with the track's limits per line (`line_limits`) worked out once into `most`
    /// for every line after it.
    fn weekend_counts_in(&self, l: usize, most: &mut Option<Vec<[(f64, u32); 3]>>) -> Option<[u32; 3]> {
        let t = self.net.line_trains.get(l).copied().flatten()?;
        // a headway line runs its weekend headway even on its weekday count of trains, which then
        // stand longer at the ends
        let own_headway = self.weekend_auto(l) != self.net.line_auto.get(l).copied().flatten();
        let c = match self.net.line_weekend.get(l).copied().unwrap_or_default() {
            Weekend { counts: Some(c), .. } => c,
            Weekend { share: 100, .. } => return None,
            Weekend { share: p, .. } => match (self.net.line_auto[l], self.svc.lines.get(l)) {
                (Some(auto), Some(ls)) => {
                    // a loop's two directions each keep the headway with half its trains (T-798),
                    // and no more than the track lets run, as the weekdays' (`auto_counts`)
                    let each = if ls.paired() { 2 } else { 1 };
                    let most = most.get_or_insert_with(|| self.svc.cap.line_limits(&self.net));
                    // under 100% never more than the weekdays run (a full track trims those, T-625)
                    std::array::from_fn(|lev| {
                        let (rt, h) = (ls.round_trip[lev] / each as f64, weekend_headway(auto[lev], p));
                        let n = each * clamp_auto(auto_trains(rt, h), rt, h, most.get(l).map_or(f64::INFINITY, |m| m[lev].0));
                        if p <= 100 { n.min(t[lev]) } else { n }
                    })
                }
                _ => t.map(|x| (x as u64 * p as u64).div_ceil(100).min(u32::MAX as u64) as u32),
            },
        };
        (c != t || own_headway).then_some(c)
    }

    /// T-3521: a headway line's headways on weekends (`line_auto` while the weekends settle): the
    /// weekdays' at its percent (`weekend_headway`), none where its weekend counts are set by hand
    /// (they run as a count line's, bar the headways stepped to, T-3557). None for a count line.
    pub(crate) fn weekend_auto(&self, l: usize) -> Option<[u32; 3]> {
        let a = self.net.line_auto.get(l).copied().flatten()?;
        match self.net.line_weekend.get(l).copied().unwrap_or_default() {
            // a headway stepped to by hand keeps the cap on the trains set to run it (T-3557); a level
            // without one runs as a count line
            Weekend { counts: Some(_), headways, .. } => (headways != [0; 3]).then_some(headways),
            Weekend { share: 100, .. } => Some(a),
            Weekend { share: p, .. } => Some(a.map(|h| weekend_headway(h, p))),
        }
    }

    /// T-3521: the weekends' headways in `line_auto` for the lines in `own` (`weekend_own`), so
    /// whatever reads a line's headway (`running_trains`, `Service::spare`, `pattern_tph`) reads the
    /// weekends'; returns the weekdays' to put back.
    fn swap_in_weekend_autos(&mut self, own: &[(usize, [u32; 3])]) -> Vec<Option<[u32; 3]>> {
        let week = self.net.line_auto.clone();
        for &(l, _) in own {
            if week[l].is_some() { self.net.line_auto[l] = self.weekend_auto(l); }
        }
        week
    }

    /// T-3521: each line's weekend counts where they differ from its weekday ones (`weekend_counts`).
    pub fn weekend_own(&self) -> Vec<(usize, [u32; 3])> {
        if !self.net.any_weekend() {
            return vec![];
        }
        let mut most = None;
        (0..self.net.line_count())
            .filter(|&l| self.net.line_ok(l as u32) && self.svc.lines.get(l).is_some_and(|s| s.ok))
            .filter_map(|l| self.weekend_counts_in(l, &mut most).map(|c| (l, c)))
            .collect()
    }

    /// T-3521: the weekend service (`svc_rest`) follows an edit's rebuild of the weekdays'
    /// (`Service::update_scoped`, the same lines and track) on the weekends' rates and counts, so the
    /// next weekend settle starts from it rather than working every line out again.
    fn mirror_weekend(&mut self, lines: &[u32], schedules: &[u32], cap_dirty: bool, edges: &[u32], nodes: &[u32]) {
        if !self.net.rest_solved {
            self.svc_rest = None;
        }
        if self.svc_rest.is_none() {
            return;
        }
        let own = self.weekend_own();
        let week_trains = self.net.line_trains.clone();
        for &(l, c) in &own { self.net.line_trains[l] = Some(c); }
        let week_auto = self.swap_in_weekend_autos(&own);
        std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
        if let Some(rest) = self.svc_rest.as_mut() {
            rest.update_scoped(&self.net, lines, schedules, cap_dirty, Some((edges, nodes)));
        }
        std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
        self.net.line_trains = week_trains;
        self.net.line_auto = week_auto;
    }

    /// T-3521: weekends' settle, after the weekdays' (`settle_train_counts`), where some line runs
    /// other trains then: every line runs at once on weekends too, so their rates come from a settle
    /// of their own over the same track (`settle_counts` on the weekend counts and service,
    /// `svc_rest`), kept in `line_tph_rest`. The weekdays' figures and service are left as they were
    /// to the bit: the rest of the game (demand, round trips, holds, capacity) reads those. With no
    /// line other on weekends this costs nothing.
    fn settle_weekend(&mut self, full: bool) -> Vec<u32> {
        let n = self.net.line_count();
        let own = self.weekend_own();
        if own.is_empty() && !self.net.rest_solved {
            self.svc_rest = None;
            return vec![];
        }
        if self.weekend_as_saved && !own.is_empty() {
            return vec![];
        }
        let before = (self.net.line_tph_rest.clone(), self.net.rest_solved);
        if own.is_empty() {
            self.net.rest_solved = false;
            self.net.settle_kept_rest.clear();
            self.svc_rest = None;
        } else {
            let week_trains = self.net.line_trains.clone();
            let week_tph = self.net.line_tph.clone();
            let week_kept = std::mem::replace(&mut self.net.settle_kept, std::mem::take(&mut self.net.settle_kept_rest));
            let settled = self.net.rates_settled;
            // the weekend service as the last weekend settle and the edits since left it; at first
            // (or after a load) the weekdays' worked out again on the weekend rates below
            let kept = self.svc_rest.take().filter(|_| before.1);
            let fresh = kept.is_none();
            let rest = kept.map_or_else(|| self.svc.clone(), |b| *b);
            let week_svc = std::mem::replace(&mut self.svc, rest);
            // a settle steps from where the last one left the weekends' rates, at first from the
            // weekdays', as a weekday count change steps from the old rates: a start already at its
            // fleet's rate would be taken as settled and never meet the track's capacity (`cap.fill`)
            if before.1 {
                for l in 0..n { self.net.line_tph[l] = self.net.line_tph_rest[l]; }
            }
            for &(l, c) in &own { self.net.line_trains[l] = Some(c); }
            let week_auto = self.swap_in_weekend_autos(&own);
            // the service (round trips, holds, how full each track is) on the weekends' rates before
            // the settle reads it: on the weekdays' a track full on weekdays held every line on it
            if fresh {
                let moved: Vec<u32> = (0..n as u32).filter(|&l| self.net.line_tph[l as usize] != week_tph[l as usize] && self.svc.lines.get(l as usize).is_some_and(|s| s.ok)).collect();
                if !moved.is_empty() { self.svc.update_scoped(&self.net, &[], &moved, true, Some((&[], &[]))); }
            }
            self.settle_counts(full);
            self.net.line_tph_rest.clone_from(&self.net.line_tph);
            // a loop's counts set by hand were made even (as its weekday counts are), which it keeps
            for &(l, _) in &own {
                if let (Some(c), Some(t)) = (&mut self.net.line_weekend[l].counts, self.net.line_trains[l]) { *c = t; }
            }
            self.net.settle_kept_rest = std::mem::replace(&mut self.net.settle_kept, week_kept);
            self.net.line_trains = week_trains;
            self.net.line_auto = week_auto;
            self.net.line_tph = week_tph;
            self.net.rates_settled = settled;
            self.svc_rest = Some(Box::new(std::mem::replace(&mut self.svc, week_svc)));
            self.net.rest_solved = true;
        }
        let on = |l: usize, t: &[[f32; 3]], s: bool| if s { t.get(l).copied() } else { self.net.line_tph.get(l).copied() };
        let moved: Vec<u32> = (0..n as u32).filter(|&l| on(l as usize, &before.0, before.1) != on(l as usize, &self.net.line_tph_rest, self.net.rest_solved)).collect();
        // their weekend departures are drawn again
        self.svc.note_changed(&moved);
        moved
    }

    /// The settle on the counts in `line_trains` (the weekends' too, `settle_weekend`).
    fn settle_counts(&mut self, full: bool) -> Vec<u32> {
        self.net.rates_settled = true;
        let ids: Vec<u32> = (0..self.net.line_count() as u32)
            .filter(|&l| self.net.line_ok(l) && self.net.line_trains[l as usize].is_some() && self.svc.lines[l as usize].ok).collect();
        if ids.is_empty() { return vec![]; }
        // a loop's trains split evenly between its two directions: an odd count rounds up (T-798)
        for &id in &ids {
            if self.svc.lines[id as usize].paired() {
                if let Some(c) = self.net.line_trains[id as usize].as_mut() {
                    c.iter_mut().for_each(|x| *x += *x % 2);
                }
            }
        }
        let before = self.net.line_tph.clone();
        let mut affected = vec![];
        // T-1283: per line and level, the sign its shortfall last had, how often it has turned
        // without shrinking by half, and its size at the last turn
        let mut swings: Vec<[(i8, u8, f64); LEVELS]> = vec![[(0, 0, 0.0); LEVELS]; ids.len()];
        let mut kept_now = vec![[false; LEVELS]; ids.len()];
        // T-3233: a round is a function of the rates, round trips and swings it starts from, so a
        // state seen `p` rounds before comes round every `p` rounds to the cap; the settle then runs
        // only the rounds that end it in the cap's state, to the bit
        let mut seen: std::collections::VecDeque<(Vec<[f32; 3]>, Vec<[f64; 3]>, Vec<[(i8, u8, f64); LEVELS]>)> = Default::default();
        let mut stop_at: Option<usize> = None;
        for round in 0..SETTLE_ROUNDS {
            let trips: Vec<[f64; 3]> = ids.iter().map(|&id| self.svc.lines[id as usize].round_trip).collect();
            let rates: Vec<[f32; 3]> = ids.iter().map(|&id| self.net.line_tph[id as usize]).collect();
            match stop_at {
                Some(s) if round >= s => break,
                Some(_) => {}
                None => {
                    if let Some(p) = (1..=seen.len()).find(|&p| { let x = &seen[seen.len() - p]; x.0 == rates && x.1 == trips && x.2 == swings }) {
                        stop_at = Some(round + (SETTLE_ROUNDS - round) % p);
                        if (SETTLE_ROUNDS - round) % p == 0 { break; }
                    } else {
                        if seen.len() == CYCLE_MAX { seen.pop_front(); }
                        seen.push_back((rates.clone(), trips.clone(), swings.clone()));
                    }
                }
            }
            let mut active = [false; LEVELS];
            let mut off = vec![[false; LEVELS]; ids.len()];
            let keep = &mut kept_now;
            keep.iter_mut().for_each(|k| *k = [false; LEVELS]);
            let worst = self.svc.cap.bottlenecks(self.net.line_count());
            let scope = self.settle_scope.as_deref();
            for (j, &id) in ids.iter().enumerate() {
                let counts = self.net.line_trains[id as usize].unwrap();
                for lev in 0..LEVELS {
                    let short = self.net.running_trains(id as usize, lev, trips[j][lev]) - rates[j][lev] as f64 * trips[j][lev] / 3600.0;
                    // T-517: a line short of its fleet's rate is settled when a track it uses is full.
                    // T-3526: not single track it alone runs over past full: only its own rate can
                    // overfill that (a lone train's rate, T-3201, when a second train comes), so it is
                    // stepped and the fill cuts it to the stretch.
                    let (wr, wres) = worst[id as usize][lev];
                    let own_over = wr > 1.0 + 1e-9 && wres != u32::MAX && {
                        let cap = &self.svc.cap;
                        cap.res[wres as usize].kind == ResKind::SingleTrack && cap.users_of(wres as usize).iter().all(|u| u.line == id)
                    };
                    let held = short > 0.0 && wr >= FULL_RHO && !own_over;
                    // T-1283: a round trip that jumps as the rate passes a threshold (a hold coming
                    // and going) has no rate that fits the fleet exactly, and the steps below circle
                    // it to the cap, each a pass over every line. Near the fit, a shortfall that
                    // keeps turning without shrinking is settled on its side with trains to spare.
                    let sign = if short >= SETTLED { 1 } else if short <= -SETTLED { -1 } else { 0 };
                    let sw = &mut swings[j][lev];
                    if short.abs() >= SWING_NEAR {
                        *sw = (0, 0, 0.0);
                    } else if sign != 0 && sw.0 != 0 && sign != sw.0 {
                        // counted only up to the test below, so a repeat is seen (T-3233)
                        if short.abs() >= 0.5 * sw.2 { sw.1 = (sw.1 + 1).min(SWING_TURNS); }
                        sw.2 = short.abs();
                    }
                    if sign != 0 { sw.0 = sign; }
                    let left = (id, lev, rates[j][lev].to_bits(), trips[j][lev].to_bits(), counts[lev]);
                    keep[j][lev] = (sw.1 >= SWING_TURNS && short > 0.0) || self.net.settle_kept.contains(&left);
                    // T-1787: on an edit a settled line is left as it is, not probed and stepped
                    // with the rest of its level: on shared track each line moved re-times its
                    // neighbours. A held line is stepped, as the fill shares full track by what each
                    // fleet would run.
                    off[j][lev] = (full || sign != 0) && !keep[j][lev] && scope.map_or(true, |s| s[id as usize]);
                    if sign != 0 && !held && !keep[j][lev] && scope.map_or(true, |s| s[id as usize]) { active[lev] = true; }
                }
            }
            if !active.contains(&true) { break; }
            // A common 1% upward probe measures the sum of positive congestion derivatives.
            // Dividing by this bound damps shared-track feedback even close to saturation.
            for (j, &id) in ids.iter().enumerate() {
                for lev in (0..LEVELS).filter(|&lev| active[lev] && (full || off[j][lev])) { self.net.line_tph[id as usize][lev] *= 1.01; }
            }
            // T-1787: an edit probes from the capacity solve alone (`Service::probe_trips`), the
            // profiles worked out once a round; a full settle probes with them, as before
            let probed = if full {
                let d = self.svc.update_scoped(&self.net, &[], &ids, true, Some((&[], &[])));
                affected.extend(d.recomputed);
                None
            } else {
                Some(self.svc.probe_trips(&self.net, &ids))
            };
            for (j, &id) in ids.iter().enumerate() {
                let i = id as usize;
                let counts = self.net.line_trains[i].unwrap();
                for lev in (0..LEVELS).filter(|&lev| active[lev]) {
                    if !off[j][lev] {
                        self.net.line_tph[i][lev] = rates[j][lev];
                        continue;
                    }
                    let trip = trips[j][lev];
                    let probe = probed.as_ref().map_or(self.svc.lines[i].round_trip[lev], |p| p[j][lev]);
                    let rise = ((probe - trip) / 0.01).max(0.0);
                    let old = rates[j][lev] as f64;
                    let runs = self.net.running_trains(i, lev, trip);
                    let residual = old * trip - runs * 3600.0;
                    self.net.line_tph[i][lev] = (if counts[lev] == 0 || trip <= 0.0 { 0.0 } else if runs < counts[lev] as f64 {
                        // T-2607: the headway set, whatever the round trip
                        runs * 3600.0 / trip
                    } else {
                        (old - residual / (trip + rise)).max(0.0)
                    }) as f32;
                }
            }
            // T-517: no track carries more trains than its capacity; a fleet's surplus waits at the ends
            for lev in (0..LEVELS).filter(|&lev| active[lev]) {
                let want: Vec<(u32, f64)> = ids.iter().map(|&id| (id, self.net.line_tph[id as usize][lev] as f64)).collect();
                for (&id, got) in ids.iter().zip(self.svc.cap.fill(&self.net, &want, lev)) {
                    self.net.line_tph[id as usize][lev] = got as f32;
                }
            }
            let d = self.svc.update_scoped(&self.net, &[], &ids, true, Some((&[], &[])));
            affected.extend(d.recomputed);
            // T-747: a round that ends where it began repeats itself to the cap, ~12 ms a round on
            // 143 lines
            let still = ids.iter().enumerate().all(|(j, &id)| self.net.line_tph[id as usize] == rates[j] && self.svc.lines[id as usize].round_trip == trips[j]);
            if still { break; }
        }
        for &id in &ids { self.svc.lines[id as usize].trains = self.net.line_trains[id as usize].unwrap(); }
        self.net.settle_kept.clear();
        for (j, &id) in ids.iter().enumerate() {
            for lev in (0..LEVELS).filter(|&lev| kept_now[j][lev]) {
                self.net.settle_kept.push((id, lev, self.net.line_tph[id as usize][lev].to_bits(), self.svc.lines[id as usize].round_trip[lev].to_bits(), self.net.line_trains[id as usize].unwrap()[lev]));
            }
        }
        affected.extend(ids.into_iter().filter(|&id| before[id as usize] != self.net.line_tph[id as usize]));
        affected.sort_unstable(); affected.dedup(); affected
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    /// T-1073: whether the edit an undo (or redo) would apply changes only train counts or headways.
    pub fn step_is_service(&self, redo: bool) -> bool {
        fn only_counts(op: &Op) -> bool {
            match op {
                Op::TrainCounts { .. } | Op::AutoTrains { .. } | Op::Weekend { .. } => true,
                Op::Batch(ops) => !ops.is_empty() && ops.iter().all(only_counts),
                _ => false,
            }
        }
        (if redo { self.redo.last() } else { self.undo.last() }).is_some_and(only_counts)
    }

    /// Apply, validate, recompute. On problems nothing changes and the problems come back.
    pub fn apply(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        let (r, inv) = self.commit(op)?;
        self.undo.push(inv);
        self.redo.clear();
        Ok(r)
    }

    /// Apply an edit and pay for it: its construction charge plus `extra`, evaluated on the
    /// edited network (the trains it makes the lines need, T-028). Refused, with nothing changed,
    /// if that comes to more than `cash`; an edit costing nothing is never refused for money, so
    /// a network in debt can still be edited. `final_` edits clear the undo history (SPEC 6.4);
    /// the others go on the undo stack. Returns the result and what `extra` came to.
    pub fn apply_paid(&mut self, op: Op, cash: f64, final_: bool, extra: &dyn Fn(&TrackWorld) -> f64) -> Result<(EditResult, f64), ChargeError> {
        self.apply_priced(op, cash, final_, extra, None)
    }

    /// `apply_paid` with the track's charge set by `price` on the edited network, not the rise in
    /// built value (T-3059: a rebuild pays the full new price).
    pub fn apply_priced(&mut self, op: Op, cash: f64, final_: bool, extra: &dyn Fn(&TrackWorld) -> f64, price: Option<&dyn Fn(&TrackWorld) -> f64>) -> Result<(EditResult, f64), ChargeError> {
        let phases = (self.net.line_phase.clone(), self.net.line_split.clone());
        let (mut r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        if let Some(price) = price {
            r.charge = price(self);
        }
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.restore(inv);
            // T-1925: and the departures as they were
            (self.net.line_phase, self.net.line_split) = phases;
            return Err(ChargeError::Money { need, have: cash });
        }
        if final_ {
            self.undo.clear();
        } else {
            self.undo.push(inv);
        }
        self.redo.clear();
        Ok((r, x))
    }

    /// T-3528: an edit the game makes for the player (the cap lowering on load): applied and paid as
    /// `apply_paid`, but on neither history, so an undo takes back the player's own last edit.
    pub fn apply_unrecorded(&mut self, op: Op, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64) -> Result<(EditResult, f64), ChargeError> {
        let phases = (self.net.line_phase.clone(), self.net.line_split.clone());
        let (r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.restore(inv);
            (self.net.line_phase, self.net.line_split) = phases;
            return Err(ChargeError::Money { need, have: cash });
        }
        Ok((r, x))
    }

    /// Undo (or redo) and pay `extra` as `apply_paid` does: undoing a line's removal can need
    /// trains again. Nothing changes if it is refused.
    pub fn step_paid(&mut self, redo: bool, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64) -> Option<Result<(EditResult, f64), ChargeError>> {
        let op = if redo { self.redo.pop()? } else { self.undo.pop()? };
        let back = |w: &mut TrackWorld, op: Op| if redo { w.redo.push(op) } else { w.undo.push(op) };
        let phases = (self.net.line_phase.clone(), self.net.line_split.clone());
        Some(match self.restore(op.clone()) {
            Ok((mut r, inv)) => {
                // what an undo or redo restores was paid for or removed before: only the trains
                // it needs are charged, never the track (a removed built stretch comes back free)
                let need = extra(self);
                if need > 1e-9 && need > cash + 1e-6 {
                    let _ = self.restore(inv);
                    (self.net.line_phase, self.net.line_split) = phases;
                    back(self, op);
                    Err(ChargeError::Money { need, have: cash })
                } else {
                    if redo {
                        self.undo.push(inv);
                    } else {
                        self.redo.push(inv);
                    }
                    r.charge = 0.0;
                    Ok((r, need))
                }
            }
            Err(e) => {
                back(self, op);
                Err(ChargeError::Issues(e))
            }
        })
    }

    /// What an edit would do to the track, without doing it: applied, measured and rolled back. The
    /// drawing tool's preview, with every check a real edit gets (crossings, ports, platforms). The
    /// services are left as they are (T-1779: rebuilding a long line's took 0.4-0.8 s each way on a
    /// shared real network), so the result's `dirty` has no lines, resources or service.
    pub fn trial(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        let (r, inv) = self.commit_with(op, true, false)?;
        // T-2700: the rollback as `commit_with(inv, false, false)` did it, without pricing the
        // whole network twice for a result nobody reads
        let _ = self.apply_and_derive(inv);
        Ok(r)
    }

    /// `trial_with` without the services (as `trial`): for a planner that tries many edits, on
    /// track that lines run on (T-3059).
    pub fn trial_light<T>(&mut self, op: Op, inspect: impl FnOnce(&TrackWorld) -> T) -> Result<(EditResult, T), Vec<Issue>> {
        let (r, inv) = self.commit_with(op, true, false)?;
        let value = inspect(self);
        let _ = self.apply_and_derive(inv);
        Ok((r, value))
    }

    /// Inspect the derived constructed state, then restore it without spending or undo edits.
    pub fn trial_with<T>(&mut self, op: Op, inspect: impl FnOnce(&TrackWorld) -> T) -> Result<(EditResult, T), Vec<Issue>> {
        let phases = (self.net.line_phase.clone(), self.net.line_split.clone());
        let (r, inv) = self.commit(op)?;
        let value = inspect(self);
        let _ = self.restore(inv);
        (self.net.line_phase, self.net.line_split) = phases;
        Ok((r, value))
    }

    /// Apply an edit with no services, undo or (unless `checked`) checks, to plan a follow-on edit
    /// against the edited track (T-733). Returns the inverse, for `unsketch`; a checked sketch
    /// with problems is rolled back.
    pub(crate) fn sketch(&mut self, op: Op, checked: bool) -> Result<Op, Vec<Issue>> {
        let (inv, t, der) = self.apply_and_derive(op)?;
        let issues = if checked { self.check(&t, &der) } else { vec![] };
        if issues.is_empty() {
            return Ok(inv);
        }
        self.unsketch(inv);
        Err(issues)
    }
    pub(crate) fn unsketch(&mut self, inv: Op) {
        let _ = self.apply_and_derive(inv);
    }

    pub fn undo(&mut self) -> Option<Result<EditResult, Vec<Issue>>> {
        let op = self.undo.pop()?;
        Some(match self.restore(op.clone()) {
            Ok((r, inv)) => {
                self.redo.push(inv);
                Ok(r)
            }
            Err(e) => {
                self.undo.push(op);
                Err(e)
            }
        })
    }

    pub fn redo(&mut self) -> Option<Result<EditResult, Vec<Issue>>> {
        let op = self.redo.pop()?;
        Some(match self.restore(op.clone()) {
            Ok((r, inv)) => {
                self.undo.push(inv);
                Ok(r)
            }
            Err(e) => {
                self.redo.push(op);
                Err(e)
            }
        })
    }

    fn commit(&mut self, op: Op) -> Result<(EditResult, Op), Vec<Issue>> {
        self.commit_with(op, true, true)
    }

    /// T-1649: an undo, a redo or a trial's rollback puts back what an edit took, so it is not
    /// checked afresh: track the saved game already had problems on (an old save under a newer
    /// water mask) would otherwise make the restore refuse, leaving a trial's deletions in place.
    fn restore(&mut self, op: Op) -> Result<(EditResult, Op), Vec<Issue>> {
        self.commit_with(op, false, true)
    }

    /// `service`: rebuild the services the edit touches (every real edit; a `trial` skips it).
    fn commit_with(&mut self, op: Op, checked: bool, service: bool) -> Result<(EditResult, Op), Vec<Issue>> {
        let (cost_before, built_before) = self.net.total_and_built_cost();
        let again = op.clone();
        let switch = match &op {
            Op::AutoTrains { id, auto, .. } if self.net.line_auto.get(*id as usize).is_some_and(|a| a.is_some() != auto.is_some()) => Some(*id),
            _ => None,
        };
        let (mut inv, mut t, mut der) = self.apply_rerouted(op, checked)?;
        let issues = if checked { self.check(&t, &der) } else { vec![] };
        if !issues.is_empty() {
            // Refuse only what the edit adds: problems already there (an old save under a newer
            // water mask, say) must not block fixing things nearby.
            let _ = self.apply_raw(inv);
            let t2 = self.net.take_touch();
            let der2 = self.net.derive(&t2);
            self.note_lines(&t2, &der2);
            let before = self.check(&t, &der);
            let added: Vec<Issue> = issues.into_iter().filter(|i| !before.iter().any(|b| same_issue(i, b))).collect();
            if !added.is_empty() {
                return Err(added);
            }
            (inv, t, der) = self.apply_rerouted(again, checked)?;
        }
        if !service {
            let (cost_after, built_after) = self.net.total_and_built_cost();
            let dirty = Dirty { edges: der.edges, nodes: der.nodes, crossings: der.crossings, ..Default::default() };
            let charge = if built_after - built_before > 1e-9 { built_after - built_before } else { 0.0 };
            return Ok((EditResult { dirty, cost_before, cost_after, built_before, built_after, charge }, inv));
        }
        let lines = self.lines_touched(&t, &der);
        let junction_touched = der.nodes.iter().any(|&n| self.net.node_ok(n) && self.net.node_ports[n as usize].n >= 3);
        // A station touched with no line stopping there still changes passing lines' platforms.
        let cap_dirty = !der.crossings.is_empty() || junction_touched || !t.stations.is_empty();
        // The capacity pass rebuilds only around what was touched (T-050).
        let mut cap_nodes = der.nodes.clone();
        cap_nodes.extend_from_slice(&t.stations);
        let settles = cap_dirty || !lines.is_empty() || !t.schedules.is_empty();
        let before = settles.then(|| self.demand_view());
        let mut svc = self.svc.update_scoped(&self.net, &lines, &t.schedules, cap_dirty, Some((&der.edges, &cap_nodes)));
        if settles { self.mirror_weekend(&lines, &t.schedules, cap_dirty, &der.edges, &cap_nodes); }
        if let Some(before) = before {
            // T-752: a switch between counts and headways never buys or sells a train
            self.switching = switch;
            let mut seeds = lines.clone();
            seeds.extend(svc.rebuilt.iter().chain(&svc.recomputed).chain(&svc.shared).chain(&t.weekend).copied());
            self.settle_scope = Some(self.sharing_closure(&seeds));
            let counts = self.settle_train_counts(false);
            self.settle_scope = None;
            self.switching = None;
            for id in counts {
                // T-1805: what the settle moved on other lines counts for demand once it shows
                // there: a train more or less, a rate or a stop-to-stop time past demand's tolerance
                if self.demand_moved(id, &before) {
                    svc.demand.push(id);
                }
                svc.recomputed.push(id);
            }
            svc.demand.sort_unstable(); svc.demand.dedup();
            svc.recomputed.sort_unstable(); svc.recomputed.dedup();
        } else if !t.weekend.is_empty() {
            // T-3521: an edit of weekends alone settles only them, the weekdays left to the bit
            self.settle_scope = Some(self.sharing_closure(&t.weekend));
            svc.recomputed.extend(self.settle_weekend(false));
            self.settle_scope = None;
            svc.recomputed.sort_unstable(); svc.recomputed.dedup();
        }
        let mut direct = lines.clone();
        direct.extend_from_slice(&t.schedules);
        direct.sort_unstable();
        direct.dedup();
        // T-1925: the lines the edit changed take departure slots among the others
        self.place_departures(&direct);
        self.pin_twin_phases();
        let mut resources = vec![];
        for (r, res) in self.svc.cap.res.iter().enumerate() {
            let on_track = (res.edge != u32::MAX && der.edges.binary_search(&res.edge).is_ok())
                || (res.node != u32::MAX && der.nodes.binary_search(&res.node).is_ok());
            if on_track || self.svc.cap.users_of(r).iter().any(|u| direct.binary_search(&u.line).is_ok()) {
                resources.push(r as u32);
            }
        }
        let (cost_after, built_after) = self.net.total_and_built_cost();
        let r = EditResult {
            dirty: Dirty { edges: der.edges, nodes: der.nodes, crossings: der.crossings, resources, lines: direct, service: svc },
            cost_before,
            cost_after,
            built_before,
            built_after,
            // below a millionth of a dollar is rounding from re-summing the same shapes (T-624)
            charge: if built_after - built_before > 1e-9 { built_after - built_before } else { 0.0 },
        };
        Ok((r, inv))
    }

    /// Lines to rebuild after an edit: set directly, on touched track, or stopping at a touched station.
    fn lines_touched(&self, t: &super::net::Touch, der: &super::net::Derived) -> Vec<u32> {
        let mut lines: Vec<u32> = t.lines.clone();
        for &e in &der.edges {
            lines.extend_from_slice(self.net.lines_on_edge(e));
        }
        lines.extend(self.net.back_lines());
        if !t.stations.is_empty() {
            for l in 0..self.net.line_count() as u32 {
                if self.net.line_ok(l) && self.net.line_stops(l).iter().any(|s| t.stations.binary_search(s).is_ok()) {
                    lines.push(l);
                }
            }
        }
        // a bus line's stops follow the stations near its route and the routes it crosses (T-1464)
        if !t.stations.is_empty() || !t.lines.is_empty() {
            lines.extend(self.net.bus_lines());
        }
        lines.sort_unstable();
        lines.dedup();
        lines
    }

    /// T-1382: what the net was changed under these lines, whether or not their service is rebuilt
    /// (an edit refused after its check, a sketch, a batch rolled back leave the net touched and
    /// derived but the service as it was): the clock worker reads their rows again.
    /// Per line, what demand reads of its service: trains, rates, stop-to-stop times by run and level.
    fn demand_view(&self) -> Vec<DemandView> {
        (0..self.net.line_count())
            .map(|l| {
                let ls = self.svc.lines.get(l);
                let times = std::array::from_fn(|r| (0..LEVELS).map(|lev| ls.map_or(vec![], |s| s.stop_to_stop_safe(r, lev))).collect());
                (self.net.line_trains[l], self.net.line_tph[l], times)
            })
            .collect()
    }

    /// Whether line `l` moved for demand since `before` (`demand_view`): its trains, a rate by
    /// `DEMAND_TPH_EPS` or a stop-to-stop time by `DEMAND_TIME_EPS_S`.
    fn demand_moved(&self, l: u32, before: &[DemandView]) -> bool {
        let Some((trains, tph, times)) = before.get(l as usize) else { return true };
        let i = l as usize;
        if self.net.line_trains[i] != *trains || self.net.line_tph[i].iter().zip(tph).any(|(a, b)| (a - b).abs() >= DEMAND_TPH_EPS) {
            return true;
        }
        let Some(ls) = self.svc.lines.get(i) else { return true };
        (0..2).any(|r| (0..LEVELS).any(|lev| {
            let (now, was) = (ls.stop_to_stop_safe(r, lev), &times[r][lev]);
            now.len() != was.len() || now.iter().zip(was).any(|(a, b)| (a - b).abs() >= DEMAND_TIME_EPS_S)
        }))
    }

    fn note_lines(&mut self, t: &super::net::Touch, der: &super::net::Derived) {
        let lines = self.lines_touched(t, der);
        self.svc.note_changed(&lines);
    }

    fn apply_and_derive(&mut self, op: Op) -> Result<(Op, super::net::Touch, super::net::Derived), Vec<Issue>> {
        let inv = match self.apply_raw(op) {
            Ok(i) => i,
            Err(e) => {
                let t = self.net.take_touch();
                self.note_lines(&t, &Default::default());
                return Err(vec![e]);
            }
        };
        let t = self.net.take_touch();
        let der = self.net.derive(&t);
        self.note_lines(&t, &der);
        Ok((inv, t, der))
    }

    /// `apply_and_derive`, and where `rerouted` and the edit changes the way single track runs
    /// (T-3137), the lines it left running against that way, which it did not break otherwise,
    /// moved through their same stops in the same edit (`way_reroutes`).
    fn apply_rerouted(&mut self, op: Op, rerouted: bool) -> Result<(Op, super::net::Touch, super::net::Derived), Vec<Issue>> {
        let broken = rerouted.then(|| self.net.line_broken.clone());
        let (mut inv, mut t, mut der) = self.apply_and_derive(op)?;
        let ops = match &broken {
            Some(b) => self.way_reroutes(&der, b),
            None => vec![],
        };
        if !ops.is_empty() {
            if let Ok((inv2, t2, der2)) = self.apply_and_derive(Op::Batch(ops)) {
                merge_derived(&mut t, &mut der, t2, der2);
                inv = Op::Batch(vec![inv2, inv]);
            }
        }
        // T-3333: lines run on both pairs follow the edit in it, as the reroutes do
        let ops = if rerouted { self.twin_ops() } else { vec![] };
        if !ops.is_empty() {
            if let Ok((inv2, t2, der2)) = self.apply_and_derive(Op::Batch(ops)) {
                merge_derived(&mut t, &mut der, t2, der2);
                inv = Op::Batch(vec![inv2, inv]);
            }
        }
        Ok((inv, t, der))
    }

    /// T-3333: ops keeping each line run on both pairs of quad track in step with its hidden twin
    /// (`twin.rs`): a parent that can run on both is split into one pair's trains and its twin made
    /// or brought up to date; one that can no longer runs alone with the whole line's trains and loses
    /// its twin; a twin whose parent is gone or switched off goes.
    fn twin_ops(&self) -> Vec<Op> {
        let net = &self.net;
        if net.line_twin.iter().all(|t| *t == Twin::None) {
            return vec![];
        }
        let mut ops = vec![];
        let mut kept: Vec<u32> = vec![];
        let mut made = 0;
        for l in 0..net.line_count() as u32 {
            let Twin::Parent { split } = net.line_twin_of(l) else { continue };
            let Some(mut d) = net.line_data(l) else { continue };
            let twin = net.twin_line(l);
            match net.twin_plan(l) {
                Ok((stops, path)) => {
                    if !split {
                        super::twin::halve(&mut d);
                        d.twin = Twin::Parent { split: true };
                        ops.push(Op::Line { id: l, data: Some(d.clone()) });
                    }
                    let was = twin.and_then(|t| net.line_data(t));
                    let mut want = super::twin::twin_data(l, &d, stops, path, was.as_ref().filter(|_| split));
                    match twin {
                        Some(t) => {
                            kept.push(t);
                            if let Some(w) = &was {
                                want.day = w.day;
                            }
                            // T-3521: a change of weekends alone is a weekend edit, which settles only them
                            if was.as_ref().is_some_and(|w| *w != want && LineData { weekend: want.weekend, ..w.clone() } == want) {
                                ops.push(Op::Weekend { id: t, weekend: want.weekend });
                            } else if was.as_ref() != Some(&want) {
                                ops.push(Op::Line { id: t, data: Some(want) });
                            }
                        }
                        None => {
                            let id = net.spare_line_ids(made + 1)[made];
                            made += 1;
                            ops.push(Op::Line { id, data: Some(want) });
                        }
                    }
                }
                Err(_) if split => {
                    super::twin::double(&mut d);
                    d.twin = Twin::Parent { split: false };
                    ops.push(Op::Line { id: l, data: Some(d) });
                }
                Err(_) => {}
            }
        }
        for t in 0..net.line_count() as u32 {
            if net.line_ok(t) && net.is_twin(t) && !kept.contains(&t) {
                ops.push(Op::Line { id: t, data: None });
            }
        }
        ops
    }

    /// T-3333: each twin's departures half a headway after its parent's, level by level, so the two
    /// pairs' trains take even turns (the spacing places them as separate lines, which share nothing).
    fn pin_twin_phases(&mut self) {
        let n = self.net.line_count();
        if !self.net.line_twin.iter().any(|t| matches!(t, Twin::Of(_))) {
            return;
        }
        self.net.line_phase.resize(self.net.line_phase.len().max(n), [0.0; LEVELS]);
        for t in 0..n {
            let Twin::Of(p) = self.net.line_twin_of(t as u32) else { continue };
            if !self.net.line_ok(t as u32) || !self.net.line_ok(p) {
                continue;
            }
            let pp = self.net.line_phase_of(p);
            let tph = self.net.line_tph[p as usize];
            self.net.line_phase[t] = std::array::from_fn(|lev| if tph[lev] > 0.0 { pp[lev] + 1800.0 / tph[lev] } else { pp[lev] });
        }
    }

    /// T-3297: the whole network to the other running side, in place (ids kept): the single track
    /// `Network::flip_reversals` names turned round, every line the flip leaves broken moved through
    /// its same stops where it can be, and everything derived again. Not undoable: the history is
    /// dropped, as what it holds was drawn for the other side. The pieces turned round, the lines
    /// moved, and the lines left broken.
    pub fn flip_right_hand(&mut self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let was_broken = self.net.line_broken.clone();
        let reversed = self.net.flip_reversals();
        for &e in &reversed {
            let w = &mut self.net.edge_oneway[e as usize];
            *w = 3 - *w;
        }
        // T-3332: a one-track station keeps its platform on its track, which now carries the other way
        for n in 0..self.net.node_count() as u32 {
            if self.net.one_side(n).is_some() {
                self.net.node_one[n as usize] = 3 - self.net.node_one[n as usize];
            }
        }
        self.net.right_hand = !self.net.right_hand;
        // junction capacity moves with the side: rates are settled again, not kept from the save
        self.net.rates_from_save = false;
        self.net.pace_from_save = false;
        self.reload();
        self.undo.clear();
        self.redo.clear();
        let (mut ops, mut moved, mut broken) = (vec![], vec![], vec![]);
        let net = &self.net;
        for l in 0..net.line_count() as u32 {
            let li = l as usize;
            if !net.line_ok(l) || !net.line_broken[li] || was_broken.get(li).copied().unwrap_or(false) {
                continue;
            }
            let stops = net.line_stops(l);
            let once = net.line_runs_once(l);
            let path = net.route_hops(stops).ok().filter(|p| once || net.back_path_of(stops, p).is_some());
            match (path, net.line_data(l)) {
                (Some(path), Some(mut d)) => {
                    d.path = path;
                    ops.push(Op::Line { id: l, data: Some(d) });
                    moved.push(l);
                }
                _ => broken.push(l),
            }
        }
        if !ops.is_empty() {
            if self.apply_and_derive(Op::Batch(ops)).is_ok() {
                self.reload();
            } else {
                broken.append(&mut moved);
            }
            self.undo.clear();
            self.redo.clear();
        }
        // T-3333: lines on both pairs follow the flip, as after any edit
        let tw = self.twin_ops();
        if !tw.is_empty() && self.apply_and_derive(Op::Batch(tw)).is_ok() {
            self.reload();
            self.undo.clear();
            self.redo.clear();
        }
        (reversed, moved, broken)
    }

    /// T-3137: ops moving each rail line on track whose way the edit changed (`der.ways`) that it
    /// left broken (`broken`: as they were before) onto a path through its same stops that runs every
    /// track's way, as `TrackApi::set_edge_oneway` does; one with no such path stays broken.
    /// T-3541: so too a line that turned back at a track end the edit built on past (no longer a
    /// track end, T-3595: a station there or not), never with more turns back than it had.
    fn way_reroutes(&self, der: &super::net::Derived, broken: &[bool]) -> Vec<Op> {
        let net = &self.net;
        let mut lines: Vec<u32> = der.ways.iter().flat_map(|&e| net.lines_on_edge(e).iter().copied()).collect();
        if !der.ways.is_empty() {
            lines.extend(net.back_lines());
        }
        let turned: Vec<u32> = (0..net.line_count() as u32)
            .filter(|&l| net.line_ok(l) && net.line_broken[l as usize] && {
                let path = net.line_path(l);
                net.path_reversals(path).into_iter().any(|k| {
                    let n = net.path_nodes(path[k]).0;
                    // T-3595: a station there too, when the line does not stop at it
                    der.nodes.contains(&n) && !net.line_stops(l).contains(&n) && !net.turn_end(n)
                })
            })
            .collect();
        lines.extend(&turned);
        lines.sort_unstable();
        lines.dedup();
        let mut ops = vec![];
        for l in lines {
            let li = l as usize;
            if !net.line_ok(l) || !net.line_broken[li] || broken.get(li).copied().unwrap_or(false) || net.is_bus(l) || net.is_ferry(l) || !net.line_path(l).iter().all(|&p| net.edge_ok(path_edge(p))) {
                continue;
            }
            let stops = net.line_stops(l);
            let once = net.line_runs_once(l);
            let turns = net.path_reversals(net.line_path(l)).len();
            let Some(path) = net.route_hops(stops).ok().filter(|p| (once || net.back_path_of(stops, p).is_some()) && (!turned.contains(&l) || net.path_reversals(p).len() <= turns)) else { continue };
            if let Some(mut d) = net.line_data(l).filter(|d| d.path != path) {
                d.path = path;
                ops.push(Op::Line { id: l, data: Some(d) });
            }
        }
        ops
    }

    /// Problems on what an edit touched: its edges, their nodes and stations, their crossings,
    /// and lines it set. A line whose path runs over removed track is left broken, not refused
    /// (removing a station it stopped at with that track, T-859).
    fn check(&self, t: &super::net::Touch, der: &super::net::Derived) -> Vec<Issue> {
        let alive: Vec<u32> = der.edges.iter().copied().filter(|&e| self.net.edge_ok(e)).collect();
        let mut issues = self.net.validate(&alive, &der.nodes, &[]);
        let mut lines = t.lines.clone();
        // T-3332: a one-track station set or turned round is checked against the lines stopping there
        if t.stations.iter().any(|&n| self.net.one_track(n)) {
            lines.extend((0..self.net.line_count() as u32).filter(|&l| self.net.line_ok(l) && self.net.line_stops(l).iter().any(|&s| self.net.one_track(s) && t.stations.contains(&s))));
            lines.sort_unstable();
            lines.dedup();
        }
        for l in lines {
            if self.net.line_ok(l) && self.net.line_path(l).iter().all(|&p| self.net.edge_ok(path_edge(p))) {
                if let Err(i) = self.net.check_line(l) {
                    issues.push(i);
                }
            }
        }
        issues
    }

    /// Apply without deriving or validating; returns the inverse. A failing batch is rolled back.
    fn apply_raw(&mut self, op: Op) -> Result<Op, Issue> {
        let n = &mut self.net;
        Ok(match op {
            Op::Node { id, data } => Op::Node { id, data: n.set_node(id, data)? },
            Op::Edge { id, data } => Op::Edge { id, data: n.set_edge(id, data)? },
            Op::Station { node, data } => Op::Station { node, data: n.set_station(node, data)? },
            Op::Line { id, data } => Op::Line { id, data: n.set_line(id, data)? },
            #[cfg(test)]
            Op::Schedule { id, tph } => {
                let old = n.line_data(id).ok_or(Issue::NoSuchLine { line: id })?;
                n.set_tph(id, tph)?;
                n.line_trains[id as usize] = None;
                n.line_auto[id as usize] = None;
                Op::TrainCounts { id, trains: old.trains, tph: old.tph }
            },
            Op::TrainCounts { id, trains, tph } => {
                let old = n.line_data(id).ok_or(Issue::NoSuchLine { line: id })?;
                n.set_tph(id, tph)?;
                n.line_trains[id as usize] = trains;
                n.line_auto[id as usize] = old.auto.filter(|_| trains.is_some());
                Op::TrainCounts { id, trains: old.trains, tph: old.tph }
            },
            Op::AutoTrains { id, auto, trains } => {
                let (old_auto, old_trains) = n.set_auto(id, auto, trains)?;
                Op::AutoTrains { id, auto: old_auto, trains: old_trains }
            },
            Op::Weekend { id, weekend } => Op::Weekend { id, weekend: n.set_weekend(id, weekend)? },
            Op::Split { edge, s, node, new_edge } => {
                if n.node_ok(node) {
                    return Err(Issue::NoSuchNode { node });
                }
                if n.edge_ok(new_edge) {
                    return Err(Issue::NoSuchEdge { edge: new_edge });
                }
                let (nd, mut first, mut second) = n.split_plan(edge, s)?;
                first.b = node;
                second.a = node;
                let mut ops = vec![Op::Node { id: node, data: Some(nd) }, Op::Edge { id: edge, data: Some(first) }, Op::Edge { id: new_edge, data: Some(second) }];
                for l in lines_using(n, edge) {
                    let mut d = n.line_data(l).unwrap();
                    let mut path = Vec::with_capacity(d.path.len() + 1);
                    for &p in &d.path {
                        if path_edge(p) == edge {
                            let dir = p & 1;
                            if dir == 0 {
                                path.push(edge << 1);
                                path.push(new_edge << 1);
                            } else {
                                path.push(new_edge << 1 | 1);
                                path.push(edge << 1 | 1);
                            }
                        } else {
                            path.push(p);
                        }
                    }
                    d.path = path;
                    ops.push(Op::Line { id: l, data: Some(d) });
                }
                self.apply_raw(Op::Batch(ops))?
            }
            Op::DeleteEdge { id } => {
                if !n.edge_ok(id) {
                    return Err(Issue::NoSuchEdge { edge: id });
                }
                let users = lines_using(n, id);
                // a reroute never adds a reversal the player did not choose (T-797)
                let turns: Vec<usize> = users.iter().map(|&l| n.turn_count(n.line_stops(l), n.line_path(l))).collect();
                let mut inv = vec![Op::Edge { id, data: n.set_edge(id, None)? }];
                for (l, turns) in users.into_iter().zip(turns) {
                    let mut d = self.net.line_data(l).unwrap();
                    if let Some(path) = self.net.route(&d.stops).filter(|p| self.net.turn_count(&d.stops, p) <= turns) {
                        d.path = path;
                        let old = self.net.set_line(l, Some(d))?;
                        inv.push(Op::Line { id: l, data: old });
                    }
                }
                inv.reverse();
                Op::Batch(inv)
            }
            Op::Batch(ops) => {
                let mut inv = Vec::with_capacity(ops.len());
                for o in ops {
                    match self.apply_raw(o) {
                        Ok(i) => inv.push(i),
                        Err(e) => {
                            for i in inv.into_iter().rev() {
                                let _ = self.apply_raw(i);
                            }
                            return Err(e);
                        }
                    }
                }
                inv.reverse();
                Op::Batch(inv)
            }
        })
    }

    /// Cost and validity of an alignment not yet built (the drawing tool's preview runs the same
    /// code): the fitted geometry, its track cost (water included) and its geometry problems.
    pub fn preview(&self, vertices: &[super::geom::Pi], tracks: u8, kind: super::params::TrackKind) -> (super::geom::Fit, f64, bool) {
        let f = self.net.fit(vertices);
        let mut wet = vec![];
        super::cost::water_spans(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); (x, y) }, self.net.water_mask.as_ref(), &mut wet);
        let ground = self.ground_of(&f);
        let c = super::cost::track_cost_t(f.len, &f.vert, &wet, &ground, tracks, kind);
        let ground_wet = wet.iter().any(|w| {
            let mut s = w[0];
            while s <= w[1] {
                if super::geom::height_at(&f.vert, s).abs() < LEVEL_H / 2.0 {
                    return true;
                }
                s += WATER_STEP / 2.0;
            }
            false
        });
        let ok = f.issues.is_empty() && !ground_wet;
        (f, c, ok)
    }

    /// The terrain spans of a fitted alignment (T-1270, `cost::terrain_spans`).
    pub fn ground_of(&self, f: &super::geom::Fit) -> Vec<[f64; 3]> {
        let mut ground = vec![];
        let frame = self.net.frame();
        super::cost::terrain_spans(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.net.terrain, &mut ground);
        ground
    }

    /// The roads a fitted alignment crosses at ground level, weighted, and their price, US$M (T-1929).
    pub fn road_charge_of(&self, f: &super::geom::Fit, kind: TrackKind) -> (f64, f64) {
        if !kind.spec().road_crossings {
            return (0.0, 0.0);
        }
        let mut cross = vec![];
        let frame = self.net.frame();
        super::roads::crossings(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.net.roads, &mut cross);
        super::roads::charge(&f.vert, &cross)
    }

    /// Metres of a fitted alignment for the track tool's tag: in each terrain class where the class
    /// is charged (T-1270: off water, at elevation -1 and above), then over water at any level
    /// (T-1887).
    pub fn ground_metres(&self, f: &super::geom::Fit) -> [f64; 5] {
        let mut out = [0.0; 5];
        let ground = self.ground_of(f);
        let mut wet = vec![];
        super::cost::water_spans(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); (x, y) }, self.net.water_mask.as_ref(), &mut wet);
        if ground.is_empty() && wet.is_empty() {
            return out;
        }
        super::cost::track_cost_parts_t(f.len, &f.vert, &wet, &ground, 2, super::params::TrackKind::Regular, |level, w, _, class, m, _| {
            if w {
                out[4] += m;
            } else if level >= TERRAIN_MIN_LEVEL {
                out[class.min(3) as usize] += m;
            }
        });
        out
    }

    /// Resources at or above 75% for a level (what T-031 marks on the map).
    pub fn busy(&self, level: usize) -> Vec<(usize, ResKind, f64, f64)> {
        let c = &self.svc.cap;
        (0..c.res.len()).filter(|&r| c.rho[r][level] >= 0.75).map(|r| (r, c.res[r].kind, c.rho[r][level], c.delay[r][level])).collect()
    }
}

/// Two problems are the same one, ignoring where exactly a moved crossing now is.
/// What a second batch inside one edit touched and derived, put with the first's.
fn merge_derived(t: &mut super::net::Touch, der: &mut super::net::Derived, t2: super::net::Touch, der2: super::net::Derived) {
    for (v, w) in [(&mut t.nodes, t2.nodes), (&mut t.edges, t2.edges), (&mut t.stations, t2.stations), (&mut t.lines, t2.lines), (&mut t.schedules, t2.schedules), (&mut t.weekend, t2.weekend), (&mut der.edges, der2.edges), (&mut der.nodes, der2.nodes), (&mut der.ways, der2.ways)] {
        v.extend(w);
        v.sort_unstable();
        v.dedup();
    }
    t.paths_changed |= t2.paths_changed;
    der.crossings.extend(der2.crossings);
}

fn same_issue(a: &Issue, b: &Issue) -> bool {
    match (a, b) {
        (Issue::Crossing { e1, e2, overlap, .. }, Issue::Crossing { e1: f1, e2: f2, overlap: o2, .. }) => (e1, e2, overlap) == (f1, f2, o2),
        (Issue::GroundOverWater { edge, .. }, Issue::GroundOverWater { edge: f, .. }) => edge == f,        _ => a == b,
    }
}

fn lines_using(n: &Network, e: u32) -> Vec<u32> {
    (0..n.line_count() as u32).filter(|&l| n.line_ok(l) && n.line_path(l).iter().any(|&p| path_edge(p) == e)).collect()
}
