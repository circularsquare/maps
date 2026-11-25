//! T-4113: what the edits applied since the clock worker last asked touched, for the state it builds to read
//! again only there (notes/T-4113.md). Collected from every applied edit's `Dirty` (undo and redo too, previews
//! not: they take themselves back), since one state can follow several edits.

use super::*;
use crate::track::world::Dirty;

/// T-4299: a stamp's hash, a word at a time (rustc's FxHasher step). Only the lines a settle noted are stamped, and a
/// line's phase tables reach 276,000 floats on her Asia save: SipHash, a call a float, took most of a state's model time.
/// A real change moves many words of a line's row, so two rows that differ hash alike only by chance (one in 2^64).
#[derive(Default)]
struct StampHash(u64);
impl StampHash {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// What a line's stamp is made of, each value written whole (floats by their bits).
trait Stamp {
    fn stamp(&self, h: &mut StampHash);
}
macro_rules! stamp_int {
    ($($t:ty),*) => { $(impl Stamp for $t { fn stamp(&self, h: &mut StampHash) { h.add(*self as i64 as u64); } })* };
}
stamp_int!(u8, u32, i32, usize, bool);
impl Stamp for f64 {
    fn stamp(&self, h: &mut StampHash) { h.add(self.to_bits()); }
}
impl Stamp for f32 {
    fn stamp(&self, h: &mut StampHash) { h.add(self.to_bits() as u64); }
}
impl Stamp for String {
    fn stamp(&self, h: &mut StampHash) {
        h.add(self.len() as u64);
        for c in self.as_bytes().chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            h.add(u64::from_le_bytes(b));
        }
    }
}
impl<T: Stamp> Stamp for Vec<T> {
    fn stamp(&self, h: &mut StampHash) { h.add(self.len() as u64); for x in self { x.stamp(h); } }
}

#[derive(Debug)]
pub(super) struct Touched {
    edges: Vec<u32>,
    nodes: Vec<u32>,
    /// the caller must take everything: a new or loaded world, or more noted than is worth keeping
    all: bool,
}

impl Default for Touched {
    fn default() -> Self {
        Touched { edges: vec![], nodes: vec![], all: true }
    }
}

impl Touched {
    pub(super) fn note(&mut self, d: &Dirty) {
        if self.all {
            return;
        }
        if self.edges.len() + self.nodes.len() > 1 << 18 {
            self.everything();
            return;
        }
        self.edges.extend_from_slice(&d.edges);
        self.nodes.extend_from_slice(&d.nodes);
    }

    pub(super) fn everything(&mut self) {
        self.edges.clear();
        self.nodes.clear();
        self.all = true;
    }

    /// T-4299: edges whose numbers moved with no edit (re-priced, `Network::take_priced`).
    pub(super) fn note_edges(&mut self, edges: &[u32]) {
        if !self.all {
            self.edges.extend_from_slice(edges);
        }
    }
}

impl TrackApi {
    /// T-4113: a line's stamp: everything the clock worker reads of its row and phase tables, through the same
    /// calls (workers/stateRows.ts `heavy`, its own calls, `line_phases`), so equal stamps are equal rows.
    fn line_stamp(&self, l: u32, most: &mut super::lines::Limits) -> u64 {
        let mut h = StampHash::default();
        macro_rules! s { ($($e:expr),* $(,)?) => { $( $e.stamp(&mut h); )* }; }
        s!(self.line_part(l), self.line_run_stops(l), self.line_run_path(l), self.line_stops(l), self.line_path(l), self.line_name(l), self.line_symbol(l),
            self.line_group(l), self.line_group_name(l), self.line_group_symbol(l), self.line_group_colour(l));
        let outer = self.line_group_outer(l);
        for k in 0..outer.len() as u32 {
            s!(self.line_group_outer_name(l, k), self.line_group_outer_symbol(l, k), self.line_group_outer_colour(l, k));
        }
        s!(outer, self.back_stamp(l), self.line_back_stops(l), self.stop_times(l, 0, 0), self.stop_times(l, 0, 1), self.stop_times(l, 0, 2),
            self.line_stop_s(l, 0), self.line_stop_s(l, 1), self.line_stretch_speeds(l), self.passed_stations(l), self.line_reversals(l),
            self.line_end_turns(l), self.line_stop_ways(l), self.line_one_stops(l), self.line_stop_only(l), self.line_hand(l), self.line_auto(l),
            self.line_fast(l), self.line_tram(l), self.line_loop(l), self.line_one_way(l), self.line_one_way_by_track(l), self.line_limit(l),
            self.line_fare_dist(l), self.line_weekend_in(l, most), self.line_tph_rest(l), self.line_shift(l), self.line_period_count(l), self.line_periods_own(l),
            self.line_tph_by_period(l), self.line_hours(l), self.line_hours_own(l), self.line_row_headways(l), self.line_rows(l), self.line_row_trains(l),
            self.line_row_weekend_in(l, most), self.line_pairs(l), self.line_twin_of(l));
        self.stamp_phases(l, &mut h);
        h.0
    }

    /// T-4299: what `line_back_path` is made from, as the service ran it: the return run's path and whether it runs one way.
    /// `line_back_path` routes the way back again on every call (1.4 ms over the 9 lines a station noted on her Asia save);
    /// the service built run 1 from that same route (`build_run`) and builds it again whenever it could move. A line the
    /// service does not run takes `line_back_path` itself (its way back is still routed from the plan).
    fn back_stamp(&self, l: u32) -> Vec<u32> {
        match self.w.svc.lines.get(l as usize).filter(|x| x.ok && !x.is_bus()) {
            Some(ls) => std::iter::once(ls.one_way as u32).chain(ls.runs[1].segs.iter().copied()).collect(),
            None => self.line_back_path(l),
        }
    }

    /// T-4299: `line_phases` into the stamp as it reads them (f32), from the profiles in place: the copy was most of a
    /// long line's stamp (276,000 floats on her Asia save)
    fn stamp_phases(&self, l: u32, h: &mut StampHash) {
        let Some(ls) = self.w.svc.lines.get(l as usize).filter(|x| x.ok) else { return h.add(0) };
        let mut one = |p: &crate::track::profile::Profile| {
            h.add(p.phases.len() as u64);
            for x in &p.phases {
                h.add((x.t as f32).to_bits() as u64 | ((x.s as f32).to_bits() as u64) << 32);
                h.add((x.v as f32).to_bits() as u64 | ((x.a as f32).to_bits() as u64) << 32);
            }
        };
        for r in 0..2 {
            for k in 0..SLOTS {
                one(&ls.runs[r].prof[k]);
                if self.w.net.slot_worked(k) {
                    for x in ls.runs[r].vars[k].iter().take(crate::track::timing::PROFILE_VARS - 1) {
                        one(&x.prof);
                    }
                }
            }
        }
    }

    /// T-4113: of the lines the model noted (`svc.take_changed`), those whose stamp moved since the last take (a bus
    /// or ferry line, or one gone, always), and the stamps taken now, for `keep_stamps`.
    pub(super) fn stamped(&self, noted: Vec<u32>, most: &mut super::lines::Limits) -> (Vec<u32>, Vec<(u32, Option<u64>)>) {
        if self.stamps_off {
            return (noted, vec![]);
        }
        // (T-4149: the track limits the weekend reads take, a pass over the whole network, once for every line stamped;
        // T-4299: shared with the kept money's, `Kept::take_limits`)
        if noted.first() == Some(&u32::MAX) {
            let all = (0..self.w.net.line_count() as u32).map(|l| (l, self.live_stamp(l, most))).collect();
            return (noted, all);
        }
        let mut out = Vec::with_capacity(noted.len());
        let mut now = Vec::with_capacity(noted.len());
        for l in noted {
            let s = self.live_stamp(l, most);
            if s.is_none() || self.stamps.get(l as usize).copied().flatten() != s {
                out.push(l);
            }
            now.push((l, s));
        }
        (out, now)
    }

    pub(super) fn keep_stamps(&mut self, now: Vec<(u32, Option<u64>)>) {
        for (l, s) in now {
            if self.stamps.len() <= l as usize {
                self.stamps.resize(l as usize + 1, None);
            }
            self.stamps[l as usize] = s;
        }
    }

    /// T-4299: the edges re-priced outside an edit (a road tile, water, terrain) named to both readers: the rows (an
    /// edge's row holds its price; T-4291 named none of them) and the kept money.
    pub(super) fn drain_priced(&mut self) {
        let p = self.w.net.take_priced();
        if !p.is_empty() {
            self.touched.note_edges(&p);
            self.kept.note_records(&p, &[]);
        }
    }

    /// T-4299: the kept money (`wasm/kept.rs`) brought up to the model.
    pub(super) fn kept_now(&mut self) {
        self.kept_with(false);
    }

    /// `kept_now` with every line's money numbers compared too (`Kept::refresh`'s `full`).
    pub(super) fn kept_with(&mut self, full: bool) {
        self.drain_priced();
        let noted = self.w.svc.take_kept();
        self.kept.refresh(&self.w, &self.run_factor, noted, full);
    }

    /// a live train line's stamp; none for a bus or ferry line (whose rows read more) or a line gone
    fn live_stamp(&self, l: u32, most: &mut super::lines::Limits) -> Option<u64> {
        (self.w.net.line_ok(l) && !self.is_bus(l) && !self.is_ferry(l)).then(|| self.line_stamp(l, most))
    }
}

#[wasm_bindgen]
impl TrackApi {
    /// T-4249: whether an edit since the last call rebuilt, rescheduled or re-timed any line's service (the edits of one
    /// player action all count); false: no train runs otherwise, so the trips and the boarding account stand.
    pub fn take_service_moved(&mut self) -> bool {
        std::mem::take(&mut self.service_moved)
    }

    /// T-4249: true makes every edit rebuild every bus line and way back, as before edits took only those they reached
    /// (to time and compare against).
    pub fn set_rebuild_all(&mut self, on: bool) {
        self.w.rebuild_all = on;
    }

    /// T-4113: false hands `take_changed_lines` every line the model noted, stamped or not (`?patch=walk`, to time
    /// against).
    pub fn set_line_stamps(&mut self, on: bool) {
        self.stamps_off = !on;
        if !on {
            self.stamps.clear();
        }
    }

    /// T-4113: the edges and nodes the edits applied since the last call added, removed or changed, as
    /// `[edge count, edges..., nodes...]` ascending within each part, or `[u32::MAX]` for everything (a new or
    /// loaded world). What changes on a record without an edit touching it (a line's run moving its one-way
    /// arrows, a station's platform line) is the caller's to find.
    pub fn take_touched(&mut self) -> Vec<u32> {
        self.drain_priced();
        let t = std::mem::replace(&mut self.touched, Touched { edges: vec![], nodes: vec![], all: false });
        Self::touched_list(t.edges, t.nodes, t.all)
    }

    /// T-4149: the lines whose runs lie anywhere new on the track since the last call, ascending, or `[u32::MAX]` for every
    /// line (`Service::take_moved`). Of `dirty_lines`, only these have runs drawn anywhere new: the others had their
    /// profiles worked out again (a settle's timing) or were rebuilt onto the same track.
    pub fn take_moved_lines(&mut self) -> Vec<u32> {
        self.w.svc.take_moved()
    }

    /// T-4281: of the lines the last `take_moved_lines` listed, where each run moved, as `[line, run, lo, hi, drift, len]`
    /// each (`service::Span`: offsets of the run now, m; `len` the run's length now); a listed line's run not here may lie
    /// anywhere new
    pub fn take_moved_spans(&mut self) -> Vec<f64> {
        let spans = self.w.svc.take_spans();
        let svc = &self.w.svc;
        let len = |l: u32, r: u8| svc.lines.get(l as usize).map_or(f64::NAN, |ls| ls.runs[r as usize].len);
        spans.into_iter().flat_map(|(l, r, s)| [l as f64, r as f64, s.lo, s.hi, s.drift, len(l, r)]).collect()
    }

    /// T-4299: the edges and nodes whose list flags (the sets the rows read: an edge's off, high speed, crossover and
    /// partner, way, joins, limit and ramp; a station's off, named, unbound, one track, one platform, fare zone, upgrades
    /// and their upkeep) moved since the last call, as `[edge count, edges..., nodes...]` ascending, or `[u32::MAX]` for
    /// every one: the rows patch their sets by these (`edge_flags_of`, `node_flags_of`) instead of reading every list.
    pub fn take_flag_changes(&mut self) -> Vec<u32> {
        self.kept_now();
        self.kept.take_flags()
    }

    /// T-4299: per live edge among `ids`, 9 numbers: id, off, high speed, crossover, its partner (`u32::MAX` none), way,
    /// joins, limit, ramp (`kept::edge_flags`).
    pub fn edge_flags_of(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter().filter(|&&e| self.w.net.edge_ok(e)).flat_map(|&e| std::iter::once(e).chain(super::kept::edge_flags(&self.w.net, e))).collect()
    }

    /// T-4299: per live node among `ids`, 9 numbers: id, off, named, unbound, one track (0, 1, 2), one platform, fare zone,
    /// upgrades, upgrade upkeep a game day US$M (`kept::node_flags`, `station_upgrade_upkeep`).
    pub fn node_flags_of(&mut self, ids: &[u32]) -> Vec<f64> {
        self.kept_now();
        let n = &self.w.net;
        ids.iter().filter(|&&v| n.node_ok(v)).flat_map(|&v| std::iter::once(v as f64).chain(super::kept::node_flags(n, v).map(|x| x as f64)).chain([self.kept.up_of(v)])).collect()
    }

    /// T-4214: `take_touched` without taking: what the edits since the last take touched, for the page's clickable map
    /// patched right after an edit (workers/clockPatch.ts), while the next state still reads them.
    pub fn peek_touched(&self) -> Vec<u32> {
        let t = &self.touched;
        Self::touched_list(t.edges.clone(), t.nodes.clone(), t.all)
    }
}

impl TrackApi {
    fn touched_list(mut e: Vec<u32>, mut n: Vec<u32>, all: bool) -> Vec<u32> {
        if all {
            return vec![u32::MAX];
        }
        for v in [&mut e, &mut n] {
            v.sort_unstable();
            v.dedup();
        }
        let mut out = Vec::with_capacity(1 + e.len() + n.len());
        out.push(e.len() as u32);
        out.extend(e);
        out.extend(n);
        out
    }
}
