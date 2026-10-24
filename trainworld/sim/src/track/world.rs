//! Edit operations with inverses, undo and redo, and what each edit dirties (SPEC 6.5,
//! notes/T-008.md 7). `TrackWorld` is what the clock worker owns: the network, the service
//! derived from it, and the undo history.

use super::capacity::ResKind;
use super::net::{path_edge, EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::service::{Service, ServiceDirty};

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

/// Most rounds of "count from round trip, settle" a headway line's counts get per edit (T-625).
const AUTO_ROUNDS: usize = 5;

/// How far over the chosen headway the real one may run: a line's round trip moves a little with
/// its own train count, and a train more for a few seconds would be noise (T-625).
const AUTO_SLACK: f64 = 1.005;

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
        self.net.touch_all();
        let t = self.net.take_touch();
        self.net.derive(&t);
        self.svc.update_all(&self.net);
        self.settle_train_counts();
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
        if any { self.settle_train_counts(); }
        any
    }

    fn settle_train_counts(&mut self) -> Vec<u32> {
        if !self.net.line_auto.iter().any(|a| a.is_some()) {
            return self.settle_counts();
        }
        // T-625: the counts of headway lines follow their round trips, which follow the counts
        // through shared track, so settle and ask again until they hold still (a few rounds).
        let mut affected = vec![];
        let first = self.net.line_trains.clone();
        for round in 0..AUTO_ROUNDS {
            let moved = self.auto_counts(round + 1 == AUTO_ROUNDS);
            affected.extend(self.settle_counts());
            if !moved && round > 0 { break; }
        }
        // counts the full track would keep waiting at the ends are not bought
        if self.trim_auto() { affected.extend(self.settle_counts()); }
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
            let each = if self.svc.lines[l].is_loop() { 2 } else { 1 };
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
                let short = counts[lev] as f64 - self.net.line_tph[l][lev] as f64 * trip[lev] / 3600.0;
                let mut held = if short > 0.01 { short.floor() as u32 } else { 0 };
                if self.svc.lines[l].is_loop() { held &= !1; }
                if held > 0 { counts[lev] -= held.min(counts[lev]); trimmed = true; }
            }
            self.net.line_trains[l] = Some(counts);
        }
        trimmed
    }

    fn settle_counts(&mut self) -> Vec<u32> {
        let ids: Vec<u32> = (0..self.net.line_count() as u32)
            .filter(|&l| self.net.line_ok(l) && self.net.line_trains[l as usize].is_some() && self.svc.lines[l as usize].ok).collect();
        if ids.is_empty() { return vec![]; }
        // a loop's trains split evenly between its two directions: an odd count rounds up (T-798)
        for &id in &ids {
            if self.svc.lines[id as usize].is_loop() {
                if let Some(c) = self.net.line_trains[id as usize].as_mut() {
                    c.iter_mut().for_each(|x| *x += *x % 2);
                }
            }
        }
        let before = self.net.line_tph.clone();
        let mut affected = vec![];
        for _ in 0..80 {
            let trips: Vec<[f64; 3]> = ids.iter().map(|&id| self.svc.lines[id as usize].round_trip).collect();
            let rates: Vec<[f32; 3]> = ids.iter().map(|&id| self.net.line_tph[id as usize]).collect();
            let mut active = [false; LEVELS];
            let worst = self.svc.cap.bottlenecks(self.net.line_count());
            for (j, &id) in ids.iter().enumerate() {
                let counts = self.net.line_trains[id as usize].unwrap();
                for lev in 0..LEVELS {
                    let short = counts[lev] as f64 - rates[j][lev] as f64 * trips[j][lev] / 3600.0;
                    // T-517: a line short of its fleet's rate is settled when a track it uses is full
                    let held = short > 0.0 && worst[id as usize][lev].0 >= FULL_RHO;
                    if short.abs() >= 0.0001 && !held { active[lev] = true; }
                }
            }
            if !active.contains(&true) { break; }            // A common 1% upward probe measures the sum of positive congestion derivatives.
            // Dividing by this bound damps shared-track feedback even close to saturation.
            for &id in &ids {
                for lev in (0..LEVELS).filter(|&lev| active[lev]) { self.net.line_tph[id as usize][lev] *= 1.01; }
            }
            let d = self.svc.update_scoped(&self.net, &[], &ids, true, Some((&[], &[])));
            affected.extend(d.recomputed);
            for (j, &id) in ids.iter().enumerate() {
                let i = id as usize;
                let counts = self.net.line_trains[i].unwrap();
                for lev in (0..LEVELS).filter(|&lev| active[lev]) {
                    let trip = trips[j][lev];
                    let rise = ((self.svc.lines[i].round_trip[lev] - trip) / 0.01).max(0.0);
                    let old = rates[j][lev] as f64;
                    let residual = old * trip - counts[lev] as f64 * 3600.0;
                    self.net.line_tph[i][lev] = (if counts[lev] == 0 || trip <= 0.0 { 0.0 } else {
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
        let (r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.commit(inv);
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
        Some(match self.commit(op.clone()) {
            Ok((r, inv)) => {
                let x = extra(self);
                let need = r.charge + x;
                if need > 1e-9 && need > cash + 1e-6 {
                    let _ = self.commit(inv);
                    back(self, op);
                    Err(ChargeError::Money { need, have: cash })
                } else {
                    if redo {
                        self.undo.push(inv);
                    } else {
                        self.redo.push(inv);
                    }
                    Ok((r, x))
                }
            }
            Err(e) => {
                back(self, op);
                Err(ChargeError::Issues(e))
            }
        })
    }

    /// What an edit would do, without doing it: applied, measured and rolled back. The drawing
    /// tool's preview, with every check a real edit gets (crossings, ports, platforms).
    pub fn trial(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        self.trial_with(op, |_| ()).map(|(r, _)| r)
    }

    /// Inspect the derived constructed state, then restore it without spending or undo edits.
    pub fn trial_with<T>(&mut self, op: Op, inspect: impl FnOnce(&TrackWorld) -> T) -> Result<(EditResult, T), Vec<Issue>> {
        let (r, inv) = self.commit(op)?;
        let value = inspect(self);
        let _ = self.commit(inv);
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
        Some(match self.commit(op.clone()) {
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
        Some(match self.commit(op.clone()) {
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
        let cost_before = self.net.total_cost();
        let built_before = self.net.built_cost();
        let again = op.clone();
        let switch = match &op {
            Op::AutoTrains { id, auto, .. } if self.net.line_auto.get(*id as usize).is_some_and(|a| a.is_some() != auto.is_some()) => Some(*id),
            _ => None,
        };
        let (mut inv, mut t, mut der) = self.apply_and_derive(op)?;
        let issues = self.check(&t, &der);
        if !issues.is_empty() {
            // Refuse only what the edit adds: problems already there (an old save under a newer
            // water mask, say) must not block fixing things nearby.
            let _ = self.apply_raw(inv);
            let t2 = self.net.take_touch();
            self.net.derive(&t2);
            let before = self.check(&t, &der);
            let added: Vec<Issue> = issues.into_iter().filter(|i| !before.iter().any(|b| same_issue(i, b))).collect();
            if !added.is_empty() {
                return Err(added);
            }
            (inv, t, der) = self.apply_and_derive(again)?;
        }
        // Lines to rebuild: set directly, on touched track, or stopping at a touched station.
        let mut lines: Vec<u32> = t.lines.clone();
        for &e in &der.edges {
            lines.extend_from_slice(self.net.lines_on_edge(e));
        }
        if !t.stations.is_empty() {
            for l in 0..self.net.line_count() as u32 {
                if self.net.line_ok(l) && self.net.line_stops(l).iter().any(|s| t.stations.binary_search(s).is_ok()) {
                    lines.push(l);
                }
            }
        }
        lines.sort_unstable();
        lines.dedup();
        let junction_touched = der.nodes.iter().any(|&n| self.net.node_ok(n) && self.net.node_ports[n as usize].n >= 3);
        // A station touched with no line stopping there still changes passing lines' platforms.
        let cap_dirty = !der.crossings.is_empty() || junction_touched || !t.stations.is_empty();
        // The capacity pass rebuilds only around what was touched (T-050).
        let mut cap_nodes = der.nodes.clone();
        cap_nodes.extend_from_slice(&t.stations);
        let mut svc = self.svc.update_scoped(&self.net, &lines, &t.schedules, cap_dirty, Some((&der.edges, &cap_nodes)));
        if cap_dirty || !lines.is_empty() || !t.schedules.is_empty() {
            // T-752: a switch between counts and headways never buys or sells a train
            self.switching = switch;
            let counts = self.settle_train_counts();
            self.switching = None;
            for id in counts {
                svc.demand.push(id);
                svc.recomputed.push(id);
            }
            svc.demand.sort_unstable(); svc.demand.dedup();
            svc.recomputed.sort_unstable(); svc.recomputed.dedup();
        }
        let mut direct = lines.clone();
        direct.extend_from_slice(&t.schedules);
        direct.sort_unstable();
        direct.dedup();
        let mut resources = vec![];
        for (r, res) in self.svc.cap.res.iter().enumerate() {
            let on_track = (res.edge != u32::MAX && der.edges.binary_search(&res.edge).is_ok())
                || (res.node != u32::MAX && der.nodes.binary_search(&res.node).is_ok());
            if on_track || self.svc.cap.users_of(r).iter().any(|u| direct.binary_search(&u.line).is_ok()) {
                resources.push(r as u32);
            }
        }
        let cost_after = self.net.total_cost();
        let built_after = self.net.built_cost();
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

    fn apply_and_derive(&mut self, op: Op) -> Result<(Op, super::net::Touch, super::net::Derived), Vec<Issue>> {
        let inv = match self.apply_raw(op) {
            Ok(i) => i,
            Err(e) => {
                self.net.take_touch();
                return Err(vec![e]);
            }
        };
        let t = self.net.take_touch();
        let der = self.net.derive(&t);
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
    pub fn preview(&self, vertices: &[super::geom::Pi], tracks: u8, hsr: bool) -> (super::geom::Fit, f64, bool) {
        let f = self.net.fit(vertices);
        let mut wet = vec![];
        super::cost::water_spans(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); (x, y) }, self.net.water_mask.as_ref(), &mut wet);
        let c = super::cost::track_cost(f.len, &f.vert, &wet, tracks, hsr);
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
        (Issue::GroundOverWater { edge, .. }, Issue::GroundOverWater { edge: f, .. }) => edge == f,
        _ => a == b,
    }
}

fn lines_using(n: &Network, e: u32) -> Vec<u32> {
    (0..n.line_count() as u32).filter(|&l| n.line_ok(l) && n.line_path(l).iter().any(|&p| path_edge(p) == e)).collect()
}
