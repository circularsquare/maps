//! Edit operations with inverses, undo and redo, and what each edit dirties (SPEC 6.5,
//! notes/T-008.md 7). `TrackWorld` is what the clock worker owns: the network, the service
//! derived from it, and the undo history.

use super::capacity::ResKind;
use super::net::{path_edge, EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::service::{Service, ServiceDirty};

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
    /// Split `edge` at chainage `s` with a new node `node`; the far part becomes `new_edge`.
    /// Line paths through it are rewritten. Geometry does not change.
    Split { edge: u32, s: f64, node: u32, new_edge: u32 },
    /// Remove an edge; each line using it is rerouted by the fastest path or left broken.
    DeleteEdge { id: u32 },
    Batch(Vec<Op>),
}

/// Trains off its fleet a settle stops at: a line within it is settled.
const SETTLED: f64 = 0.0001;
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
    /// Positions of crossings removed or added.
    pub crossings: Vec<[f64; 2]>,
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
        let mut w = TrackWorld { net, svc: Service::default(), undo: vec![], redo: vec![], switching: None };
        w.reload();
        w
    }

    /// Derive everything from the inputs (after a load or a bulk build).
    pub fn reload(&mut self) {
        self.svc.changed_all = true;
        self.net.touch_all();
        let t = self.net.take_touch();
        self.net.derive(&t);
        self.svc.update_all(&self.net);
        self.settle_train_counts(!self.net.rates_from_save);
        // T-1925: a network without phases (an older save, a new world) has every line placed once,
        // in id order; a reload for new water keeps them
        if !self.net.phases_placed {
            let all: Vec<u32> = (0..self.net.line_count() as u32).collect();
            self.place_departures(&all);
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
            if !self.pace_merges() {
                break;
            }
            affected.extend(self.settle_unpaced(full));
        }
        affected.sort_unstable();
        affected.dedup();
        affected
    }

    /// T-2973: count lines (a whole-line fleet) meeting at a merge, with or through each other,
    /// whose headways (round trip over trains) are within `PACE_SHARE` of the longest of them, all
    /// run that one: the faster stand the difference at their ends, as a headway line does, so the
    /// lines take even turns at the merge all day instead of drifting past each other. A headway
    /// line there counts with its set headway and is never moved. Into `Network::line_pace`; true
    /// when it changed.
    fn pace_merges(&mut self) -> bool {
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
        let mut pace = vec![[0.0f32; LEVELS]; n];
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for l in (0..n).filter(|&l| met[l]) {
            let g = top(&mut root, l);
            groups.entry(g).or_default().push(l);
        }
        for lines in groups.values() {
            for lev in 0..LEVELS {
                // (headway, line, set by the player)
                let mut hs: Vec<(f64, usize, bool)> = vec![];
                for &l in lines {
                    let ls = &self.svc.lines[l];
                    if !self.net.line_ok(l as u32) || !ls.ok || !ls.running || ls.is_loop() || ls.is_bus() {
                        continue;
                    }
                    if let Some(a) = self.net.line_auto[l] {
                        if a[lev] > 0 {
                            hs.push((a[lev] as f64, l, true));
                        }
                        continue;
                    }
                    let Some(c) = self.net.line_trains[l] else { continue };
                    if c[lev] > 0 && ls.round_trip[lev] > 0.0 {
                        hs.push((ls.round_trip[lev] / c[lev] as f64, l, false));
                    }
                }
                hs.sort_by(|a, b| b.0.total_cmp(&a.0));
                let mut i = 0;
                while i < hs.len() {
                    let top_h = hs[i].0;
                    let mut j = i + 1;
                    while j < hs.len() && hs[j].0 * (1.0 + PACE_SHARE) >= top_h {
                        j += 1;
                    }
                    // every count line of the set, the slowest too, a little over the longest, so a
                    // round trip drifting by a second as the others' holds move keeps them in step
                    let keep = if hs[i].2 { top_h } else { top_h * (1.0 + PACE_SLACK) };
                    for k in i..j {
                        if !hs[k].2 && j - i > 1 {
                            pace[hs[k].1][lev] = keep as f32;
                        }
                    }
                    i = j;
                }
                // a full merge is shared out evenly between the count lines through it: each that
                // could run more often runs the even share of what they run together, so the full
                // track's queue is not all one line's (T-2960 saw 3:39 and 7:02 for two alike)
                let g = top(&mut root, lines[0]);
                if full.contains(&(g, lev)) {
                    let count: Vec<(usize, f64)> = hs.iter().filter(|x| !x.2).map(|x| (x.1, x.0)).collect();
                    let rate: f64 = count.iter().map(|&(l, _)| self.net.line_tph[l][lev] as f64).sum();
                    if count.len() > 1 && rate > 0.0 {
                        let even = 3600.0 * count.len() as f64 / rate;
                        for &(l, h) in &count {
                            if h < even * (1.0 - PACE_SLACK) {
                                pace[l][lev] = pace[l][lev].max(even as f32);
                            }
                        }
                    }
                }
            }
        }
        let was: Vec<[f32; LEVELS]> = (0..n).map(|l| self.net.line_pace_of(l as u32)).collect();
        if was == pace {
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
        for _ in 0..80 {
            let trips: Vec<[f64; 3]> = ids.iter().map(|&id| self.svc.lines[id as usize].round_trip).collect();
            let rates: Vec<[f32; 3]> = ids.iter().map(|&id| self.net.line_tph[id as usize]).collect();
            let mut active = [false; LEVELS];
            let mut off = vec![[false; LEVELS]; ids.len()];
            let keep = &mut kept_now;
            keep.iter_mut().for_each(|k| *k = [false; LEVELS]);
            let worst = self.svc.cap.bottlenecks(self.net.line_count());
            for (j, &id) in ids.iter().enumerate() {
                let counts = self.net.line_trains[id as usize].unwrap();
                for lev in 0..LEVELS {
                    let short = self.net.running_trains(id as usize, lev, trips[j][lev]) - rates[j][lev] as f64 * trips[j][lev] / 3600.0;
                    // T-517: a line short of its fleet's rate is settled when a track it uses is full
                    let held = short > 0.0 && worst[id as usize][lev].0 >= FULL_RHO;
                    // T-1283: a round trip that jumps as the rate passes a threshold (a hold coming
                    // and going) has no rate that fits the fleet exactly, and the steps below circle
                    // it to the cap, each a pass over every line. Near the fit, a shortfall that
                    // keeps turning without shrinking is settled on its side with trains to spare.
                    let sign = if short >= SETTLED { 1 } else if short <= -SETTLED { -1 } else { 0 };
                    let sw = &mut swings[j][lev];
                    if short.abs() >= SWING_NEAR {
                        *sw = (0, 0, 0.0);
                    } else if sign != 0 && sw.0 != 0 && sign != sw.0 {
                        if short.abs() >= 0.5 * sw.2 { sw.1 = sw.1.saturating_add(1); }
                        sw.2 = short.abs();
                    }
                    if sign != 0 { sw.0 = sign; }
                    let left = (id, lev, rates[j][lev].to_bits(), trips[j][lev].to_bits(), counts[lev]);
                    keep[j][lev] = (sw.1 >= SWING_TURNS && short > 0.0) || self.net.settle_kept.contains(&left);
                    // T-1787: on an edit a settled line is left as it is, not probed and stepped
                    // with the rest of its level: on shared track each line moved re-times its
                    // neighbours. A held line is stepped, as the fill shares full track by what each
                    // fleet would run.
                    off[j][lev] = (full || sign != 0) && !keep[j][lev];
                    if sign != 0 && !held && !keep[j][lev] { active[lev] = true; }
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
        let (mut inv, mut t, mut der) = self.apply_and_derive(op)?;
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
            (inv, t, der) = self.apply_and_derive(again)?;
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
        if let Some(before) = before {
            // T-752: a switch between counts and headways never buys or sells a train
            self.switching = switch;
            let counts = self.settle_train_counts(false);
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
        }
        let mut direct = lines.clone();
        direct.extend_from_slice(&t.schedules);
        direct.sort_unstable();
        direct.dedup();
        // T-1925: the lines the edit changed take departure slots among the others
        self.place_departures(&direct);
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

    /// Problems on what an edit touched: its edges, their nodes and stations, their crossings,
    /// and lines it set. A line whose path runs over removed track is left broken, not refused
    /// (removing a station it stopped at with that track, T-859).
    fn check(&self, t: &super::net::Touch, der: &super::net::Derived) -> Vec<Issue> {
        let alive: Vec<u32> = der.edges.iter().copied().filter(|&e| self.net.edge_ok(e)).collect();
        let mut issues = self.net.validate(&alive, &der.nodes, &[]);
        for &l in &t.lines {
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
fn same_issue(a: &Issue, b: &Issue) -> bool {
    match (a, b) {
        (Issue::Crossing { e1, e2, overlap, .. }, Issue::Crossing { e1: f1, e2: f2, overlap: o2, .. }) => (e1, e2, overlap) == (f1, f2, o2),
        (Issue::GroundOverWater { edge, .. }, Issue::GroundOverWater { edge: f, .. }) => edge == f,        _ => a == b,
    }
}

fn lines_using(n: &Network, e: u32) -> Vec<u32> {
    (0..n.line_count() as u32).filter(|&l| n.line_ok(l) && n.line_path(l).iter().any(|&p| path_edge(p) == e)).collect()
}
