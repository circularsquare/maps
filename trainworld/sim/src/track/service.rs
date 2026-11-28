//! Lines as run (SPEC 6.3): each line's two directions ("runs") with their path, stops, speed
//! limits and profiles per demand level; stop times, round trip, trains needed; the capacity pass;
//! and the trips the renderer draws.

use super::capacity::{self, Capacity};
use super::geom;
use super::net::{path_dir, path_edge, DayFleet, LoopKind, Network};
use super::params::*;
use super::profile::{self, Hold, Profile};

pub mod bends;
pub mod check;

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
    /// Per stop: trains reverse there (T-797), standing the line's turnaround; and how far on the
    /// train's middle leaves from (`stop_s + jump`: the same place on the ground, past the turn in
    /// this run's unfolded offsets; 0 where it does not reverse).
    pub turn: Vec<bool>,
    pub jump: Vec<f64>,
    /// Per stop: trains pass it without stopping, on the other track from its one platform (T-3332,
    /// `Network::run_passes`). It keeps its place in every per-stop array, timed as the train passes
    /// (`arr` and `dep` alike), so the runs' stops stay the line's; empty for none.
    pub pass: Vec<bool>,
    /// T-3565: the one-way stations on the other run's track (`Network::match_run`), as (stop index,
    /// station): passed (`pass`), each placed at the node of this run's path nearest it.
    pub away: Vec<(u32, u32)>,
    /// Track ends where trains reverse between stops (T-797 B3; a station there no stop, T-3595): index into `nodes`, where
    /// the train's middle stands on arrival and how far on it leaves from, as `stop_s`/`jump`.
    /// Not stops: the profile stands the turnaround there and leaves them out of `arr`/`dep`.
    pub ends: Vec<(u32, f64, f64)>,
    /// Speed-limit breakpoints `[s, v]`, widened by half the train length.
    pub limits: Vec<[f64; 2]>,
    /// The hops between stops and ends with no holds (`run_profile`).
    pub hops: profile::Hops,
    /// A tram line's run (`Network::line_tram`): it brakes at `TRAM_BRAKE` (T-3921).
    pub tram: bool,
    /// No capacity delay.
    pub free: Profile,
    /// Per slot (`params::SLOTS`): track holds and extra dwell per stop from the capacity pass.
    pub holds: [Vec<Hold>; SLOTS],
    pub stop_holds: [Vec<f64>; SLOTS],
    pub prof: [Profile; SLOTS],
    /// T-3346 stage 2 (`timing`): per slot, the train-by-train variants of this run's timetable,
    /// at most `timing::MAX_VARS`; a trip's `var` k > 0 runs `vars[slot][k - 1]`. Empty where every
    /// train runs `prof`.
    pub vars: [Vec<Variant>; SLOTS],
    /// Per slot, each weekday departure's `var` by its place in the day (`Dep`'s place), sorted by
    /// place; a place not listed runs `prof`.
    pub pattern: [Vec<(u32, u8)>; SLOTS],
}

/// One train-by-train version of a run's timetable (T-3346 stage 2): its own extra dwell per stop
/// and track holds in place of the slot's averaged ones, and the profile they give.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Variant {
    pub stop_holds: Vec<f64>,
    pub holds: Vec<Hold>,
    pub prof: Profile,
}

impl Run {
    /// The variant a departure at `place` (`Dep`'s) of slot `lev` runs: 0 for the slot's own.
    pub fn var_at(&self, lev: usize, place: u32) -> u8 {
        let p = &self.pattern[lev];
        p.binary_search_by_key(&place, |x| x.0).map_or(0, |i| p[i].1)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LineSvc {
    /// Alive, not broken, profiles built.
    pub ok: bool,
    /// `ok` and the stops it runs constructed with the track between (all of its plan, or the part
    /// in `part`): the line carries trains, loads capacity and has trips (SPEC 6.4). A line that is
    /// `ok` but not running shows its times as a preview.
    pub running: bool,
    /// The stops these profiles were built for (a change wakes demand): the part of the plan its
    /// trains run (`part`), else the whole plan.
    pub stops: Vec<u32>,
    /// T-3474: where the line runs only part of its plan (`Network::line_stretch`), that part; None
    /// when it runs or previews all of it.
    pub part: Option<super::net::Stretch>,
    /// A loop (T-798): each run goes from the first stop round to it again, and its trains take
    /// that run's next departure there.
    pub kind: LoopKind,
    /// T-2981: a loop run one way: run 1 carries no trains (`Lattice::off` infinite).
    pub one_way: bool,
    pub runs: [Run; 2],
    pub cars: u32,
    pub train_len: f64,
    /// Per slot: both runs with dwell, holds and their closing stands (`LineSvc::lap`);
    /// for a loop, twice the longer lap.
    pub round_trip: [f64; SLOTS],
    pub trains: [u32; SLOTS],
    /// Per slot: capacity delay over the round trip.
    pub delay: [f64; SLOTS],
    /// Seconds the line's local time runs ahead of the game clock (T-309, `Network::line_shift`):
    /// its periods and its boarding account's hours are its own region's.
    pub shift: f64,
    /// A bus line (T-1464): its runs have no path (empty `segs`, `nodes`, `stop_idx`), `stops`
    /// holds its stops' ids (`bus::BusStop::id`), and `bus` its plan in run 0's order.
    pub bus: Option<super::bus::Plan>,
    /// A ferry line (T-2193): run like a bus line (`bus` holds its ports as stops), every stop a port.
    pub ferry: bool,
    /// T-3892: a ferry line's boat (Medium for any other line).
    pub boat: super::ferry::Boat,
    /// T-3346 stage 2: the pattern the runs' variants were worked out for (`timing::retime`, as
    /// `Lattice::timing_key`); a trip takes one only while the line still runs it.
    pub timed: Option<Lattice>,
    /// T-4013: per stop (parallel to `stops`), `LineData::only`'s kind there (0 both ways on and
    /// off); empty for a line with none. Demand reads it through `span::line_span`.
    pub only: Vec<u8>,
}

impl Run {
    /// Trains pass stop `j` without stopping (T-3332).
    pub fn passes(&self, j: usize) -> bool {
        self.pass.get(j).copied().unwrap_or(false)
    }

    /// Its trains' braking, m/s².
    pub fn brake(&self) -> f64 {
        if self.tram { TRAM_BRAKE } else { BRAKE }
    }
}

impl LineSvc {
    /// A line with no track: a bus or a ferry line.
    pub fn is_bus(&self) -> bool {
        self.bus.is_some()
    }
    /// Riders a train, bus or boat takes at crush.
    pub fn crush(&self) -> f64 {
        if self.ferry { super::boarding::CRUSH_PER_CAR * self.boat.cars() as f64 } else if self.is_bus() { BUS_CRUSH } else { super::boarding::CRUSH_PER_CAR * self.cars.max(1) as f64 }
    }
    /// Riders a train, bus or boat takes at its rated load (100%, T-2186): two thirds of its crush
    /// for all three (T-2919).
    pub fn rated(&self) -> f64 {
        self.crush() * super::boarding::RATED_PER_CAR / super::boarding::CRUSH_PER_CAR
    }
    /// The station node of stop `k` of run `run`, where its riders wait: a rail line's or a ferry line's
    /// (a port); None for a bus line, whose stops have no platform.
    pub fn stop_node(&self, run: usize, k: usize) -> Option<u32> {
        if let Some(plan) = self.bus.as_ref() {
            let n = plan.stops.len();
            return if self.ferry && k < n { plan.stops[if run == 0 { k } else { n - 1 - k }].station } else { None };
        }
        let r = self.runs.get(run)?;
        if let Some(&(_, s)) = r.away.iter().find(|x| x.0 as usize == k) {
            return Some(s);
        }
        r.nodes.get(*r.stop_idx.get(k)? as usize).copied()
    }
    /// T-1067, T-4052: the most riders waiting at stop `k` of run `run`: a platform side's `WAITING_CAP` at a
    /// station or a port; at a bus line's stop, which has no platform, `BUS_STOP_WAITING`, or `BRT_STATION_WAITING`
    /// at a BRT station.
    pub fn waiting_cap(&self, net: &Network, run: usize, k: usize) -> f64 {
        use super::boarding::{BRT_STATION_WAITING, BUS_STOP_WAITING, WAITING_CAP};
        let Some(plan) = self.bus.as_ref().filter(|_| !self.ferry) else { return WAITING_CAP };
        let n = plan.stops.len();
        let at = plan.stops.get(if run == 0 { k } else { n.wrapping_sub(1 + k) }).and_then(|s| s.station);
        if at.is_some_and(|s| net.is_brt_station(s)) { BRT_STATION_WAITING } else { BUS_STOP_WAITING }
    }
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
    /// Seconds a trip of this direction is drawn: run, then the closing stand.
    pub fn trip_len(&self, net: &Network, line: u32, run: usize, level: usize) -> f64 {
        self.runs[run].prof[level].duration + self.close_stand(net, line)
    }
    /// The profile a trip with variant `var` runs (T-3346 stage 2): the slot's, or its own.
    pub fn prof_of(&self, run: usize, level: usize, var: u8) -> &Profile {
        let r = &self.runs[run];
        match var {
            0 => &r.prof[level],
            v => r.vars[level].get(v as usize - 1).map_or(&r.prof[level], |x| &x.prof),
        }
    }
    /// `trip_len` for a trip with variant `var`: a train that ran its own timetable still takes
    /// its next departure as the slot's would, standing less at the end where it waited longer on
    /// the way, down to `timing::TURN_MIN_S` (T-3346 stage 2).
    pub fn trip_len_var(&self, net: &Network, line: u32, run: usize, level: usize, var: u8) -> f64 {
        let mean = self.trip_len(net, line, run, level);
        if var == 0 {
            return mean;
        }
        let stand = self.close_stand(net, line).min(super::timing::TURN_MIN_S);
        mean.max(self.prof_of(run, level, var).duration + stand)
    }
    /// What a train stands at the end of a run: the turnaround, or the dwell where a circle carries
    /// on round (T-798).
    pub fn close_stand(&self, net: &Network, line: u32) -> f64 {
        if self.kind == LoopKind::Circle { net.line_dwell[line as usize] as f64 } else { net.line_turn[line as usize] as f64 }
    }
    pub fn is_loop(&self) -> bool {
        self.kind != LoopKind::None
    }
    /// A loop run both ways: its trains are half each way, so its count is even (T-798).
    pub fn paired(&self) -> bool {
        self.is_loop() && !self.one_way
    }
    /// The runs that carry trains, from run 0: both, or run 0 alone on a one-way loop (T-2981).
    pub fn live_runs(&self) -> usize {
        if self.one_way { 1 } else { 2 }
    }
    /// Whether its trains take edge `e` that its plan has (T-3474): always, unless it runs only part
    /// of its plan.
    pub fn runs_on(&self, e: u32) -> bool {
        self.part.is_none() || self.runs.iter().take(self.live_runs()).any(|r| r.segs.iter().any(|&p| path_edge(p) == e))
    }
    /// The round trip from each run's duration: a one-way loop's lap, twice a two-way loop's
    /// longer lap, or both runs and their turnarounds.
    fn round_trip_of(&self, a: f64, b: f64, stand: f64) -> f64 {
        if self.one_way { a + stand } else if self.is_loop() { 2.0 * (a.max(b) + stand) } else { a + b + 2.0 * stand }
    }
    /// Trains a round trip needs at `tph` each way; a two-way loop's count is even, the same each way.
    pub fn fleet(&self, round_trip: f64, tph: f64) -> u32 {
        if tph <= 0.0 {
            0
        } else if self.paired() {
            2 * (0.5 * round_trip * tph / 3600.0 - 1e-9).ceil() as u32
        } else {
            (round_trip * tph / 3600.0 - 1e-9).ceil() as u32
        }
    }
}

/// A departure the renderer draws: `profile(line, run, level)` from time `dep`, drawn until `end`:
/// the end of its run and turnaround, or later when the train stands at the terminal for its next
/// departure (T-206). `hw` is the headway of the pattern it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trip {
    pub line: u32,
    pub run: u8,
    pub level: u8,
    /// The departure in the line's pattern (the boarding account's key).
    pub dep: f64,
    /// When it is drawn leaving: `dep`, or moved so a train running when its line's profile
    /// changed carries on from where it was (T-206).
    pub draw: f64,
    pub end: f64,
    pub hw: f64,
    /// The departure its train takes next, from the terminal it ends at (that trip's `draw`,
    /// `hand_over`); None when the train leaves service there (T-245: a selected train is
    /// followed onto its next trip).
    pub next: Option<f64>,
    /// T-3346 stage 2: the timetable variant it runs (`LineSvc::prof_of`), 0 for the slot's.
    /// Every reader takes it from here.
    pub var: u8,
}

/// A train running when an edit is applied: where it is on its profile (T-206).
#[derive(Clone, Copy, Debug)]
pub struct InFlight {
    line: u32,
    run: u8,
    level: u8,
    var: u8,
    dep: f64,
    s: f64,
    /// Standing at a stop: seconds before it would have left; with `s` infinite, standing at its
    /// last stop: seconds it has stood there.
    stopped_left: Option<f64>,
    /// The place on the ground when `s` is finite: the path entry it is on, which time that entry
    /// comes up in the run, and the metres into it. A changed run can start somewhere else, so
    /// the offset alone would move the train.
    at: Option<(u32, usize, f64)>,
    /// T-4133: what `note_changes` reads of its profile and run (`train_key`): the same after the edit,
    /// the train stands where it was.
    key: u64,
}

/// T-4133: the bits of what moving a train again reads: its profile's phases and length, and its
/// run's path and offsets.
fn train_key(r: &Run, p: &Profile) -> u64 {
    let bits = p.phases.iter().flat_map(|x| [x.t, x.s, x.v, x.a]).chain([p.duration, r.len]).chain(r.seg_off.iter().copied()).map(f64::to_bits);
    bits.chain(r.segs.iter().map(|&x| x as u64)).fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x100_0000_01b3))
}

/// T-4133: `train_key` once per line, run, slot and variant (per line, the few it has).
struct TrainKeys(Vec<Vec<([u8; 3], u64)>>);

impl TrainKeys {
    fn new(lines: usize) -> TrainKeys {
        TrainKeys(vec![vec![]; lines])
    }
    fn of(&mut self, ls: &LineSvc, l: u32, run: u8, level: u8, var: u8) -> u64 {
        let v = &mut self.0[l as usize];
        let at = [run, level, var];
        if let Some(&(_, k)) = v.iter().find(|x| x.0 == at) {
            return k;
        }
        let k = train_key(&ls.runs[run as usize], ls.prof_of(run as usize, level as usize, var));
        v.push((at, k));
        k
    }
}

/// Where offset `s` is on a run's path, as `InFlight::at` keeps it.
fn ground_at(r: &Run, s: f64) -> Option<(u32, usize, f64)> {
    if r.segs.is_empty() {
        return None;
    }
    let i = r.seg_off.partition_point(|&o| o <= s).saturating_sub(1).min(r.segs.len() - 1);
    let nth = r.segs[..i].iter().filter(|&&p| p == r.segs[i]).count();
    Some((r.segs[i], nth, s - r.seg_off[i]))
}

/// The offset on a run of a place `ground_at` found on its old path: None when the new path no
/// longer has that entry.
fn offset_at(r: &Run, at: (u32, usize, f64)) -> Option<f64> {
    let i = (0..r.segs.len()).filter(|&i| r.segs[i] == at.0).nth(at.1)?;
    let end = r.seg_off.get(i + 1).copied().unwrap_or(r.len);
    Some((r.seg_off[i] + at.2).min(end))
}

/// The service as an edit found it (`Service::before_edit`, then `note_changes`).
#[derive(Clone, Debug, Default)]
pub struct Before {
    lat: Vec<Option<Lattice>>,
    flight: Vec<InFlight>,
    lens: Vec<[f64; 2 * SLOTS]>,
    /// T-4133: how many lines `changed` held: what the edit notes after this is what it touched
    /// (`Service::touched_since`).
    noted: usize,
}

/// The earliest time a profile's train has reached `s` (it never goes back), within `[0, dur]`.
fn time_reaching(ph: &[profile::Phase], dur: f64, s: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, dur);
    if profile::state_at(ph, hi).0 < s {
        return hi;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if profile::state_at(ph, mid).0 >= s { hi = mid } else { lo = mid }
    }
    hi
}

/// A line's departure pattern (T-206): trains an hour per demand level, and where run 1's
/// departures sit after each period's start (a whole-line fleet's reverse trips follow the forward
/// ones round the circuit; 0 otherwise; infinite on a one-way loop, which has none, T-2981), and `shift`, the line's local time ahead of the game
/// clock (T-309): its periods start on its own region's hours. `lead`: seconds every period's
/// departures leave the ends before its start, so a train belongs to the period in force when it
/// reaches the line's middle (T-835); one figure for both runs keeps a whole-line fleet aligned.
/// `phase`: seconds every departure of a level leaves after that, both runs, under a headway, so
/// lines sharing track leave apart (T-1925, `spacing::place`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    pub tph: [f32; SLOTS],
    pub off: [f64; SLOTS],
    pub lead: f64,
    pub shift: f64,
    pub phase: [f64; SLOTS],
    /// T-3521: weekends' trains an hour, offsets and phases per level (`LineData::weekend`), as
    /// `tph`, `off` and `phase` are the weekdays'; `rest_on` false: every day runs the weekdays'.
    pub rest_tph: [f32; SLOTS],
    pub rest_off: [f64; SLOTS],
    pub rest_phase: [f64; SLOTS],
    pub rest_on: bool,
    /// The line's own calendar around now (holidays, its country's weekend): the kind of those days.
    /// `Cal::NONE` without `rest_on`, so the pattern is the same whatever day it is.
    pub cal: Cal,
    /// T-3542: the slot serving each period (`Network::period_slot`): the periods of one slot
    /// join into one stretch of departures.
    pub period_slot: [usize; 5],
    /// T-3542: the line's day in pieces where its hours differ from the default's (`n` 0: the
    /// periods, as at the default hours).
    pub hours: Hours,
}

impl Lattice {
    /// T-3346 stage 2: what a run's variants are worked out for: the pattern without its calendar,
    /// which moves on day by day without changing a weekday's departures.
    pub fn timing_key(&self) -> Lattice {
        Lattice { cal: Cal::NONE, ..*self }
    }
}

/// T-3542: a line's day cut at its own hours as well as the periods' (`params::day_pieces`), each
/// piece run by its row: in a slot of one row at the slot's rate, offset and phase; in a slot its
/// hours share between rows (`Network::mixed`) at that row's own (`own`, per kind of day).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hours {
    pub n: u8,
    pub pieces: [DayPiece; PIECES_MAX],
    /// Per piece: whether it runs `own` rather than its slot's.
    pub own_on: [bool; PIECES_MAX],
    /// Per piece, weekdays then weekends: trains an hour, offset, phase.
    pub own: [[(f32, f64, f64); DAY_ROWS]; PIECES_MAX],
}

impl Hours {
    pub const NONE: Hours = Hours { n: 0, pieces: [DayPiece { a: 0, b: 0, q: 0, k: 0 }; PIECES_MAX], own_on: [false; PIECES_MAX], own: [[(0.0, 0.0, 0.0); DAY_ROWS]; PIECES_MAX] };
    /// What a hash of a pattern reads of its hours (`span::span_key`): nothing at `n` 0.
    pub fn words(&self) -> Vec<u64> {
        let mut out = vec![];
        for i in 0..self.n as usize {
            let p = self.pieces[i];
            out.push(1 << 48 | (p.a as u64) << 24 | (p.b as u64) << 16 | (p.q as u64) << 8 | p.k as u64);
            if self.own_on[i] {
                for &(t, o, ph) in &self.own[i] { out.extend([t.to_bits() as u64, o.to_bits(), ph.to_bits()]); }
            }
        }
        out
    }
    /// T-3931: the piece whose period's slot piece `i` runs the timetable of (its waits, round
    /// trip, spare time and phase): in a slot its hours share between rows (`own_on`), the nearest
    /// piece of its own row, through the day, that has a slot to itself, so a moved hour of the
    /// morning runs as the morning does; itself where its row has none.
    pub fn row_piece(&self, i: usize) -> usize {
        let n = self.n as usize;
        if !self.own_on[i] || n == 0 {
            return i;
        }
        let k = self.pieces[i].k;
        // (the day's last piece runs on into its first: the night crosses midnight)
        let find = |step: usize| {
            let mut j = i;
            for d in 1..n {
                j = (j + step) % n;
                if self.pieces[j].k != k {
                    return None;
                }
                if !self.own_on[j] {
                    return Some((d, j));
                }
            }
            None
        };
        match (find(1), find(n - 1)) {
            (Some(a), Some(b)) => if b.0 < a.0 { b.1 } else { a.1 },
            (Some(a), None) | (None, Some(a)) => a.1,
            (None, None) => i,
        }
    }
}

/// One stretch of a day's departures (`day_stretches`): local hours, level (`Lattice::levels_on`),
/// trains an hour, offset and phase.
#[derive(Clone, Copy, Debug)]
struct Stretch {
    h0: f64,
    h1: f64,
    lev: usize,
    tph: f32,
    off: f64,
    phase: f64,
}

impl Lattice {
    pub const NONE: Lattice = Lattice { tph: [0.0; SLOTS], off: [0.0; SLOTS], lead: 0.0, shift: 0.0, phase: [0.0; SLOTS], rest_tph: [0.0; SLOTS], rest_off: [0.0; SLOTS], rest_phase: [0.0; SLOTS], rest_on: false, cal: Cal::NONE, period_slot: PERIOD_SLOT_3, hours: Hours::NONE };
    /// Each period's slot on local day `day` (`period_slot`) plus `SLOTS` times the day's kind
    /// (weekdays or weekends; always weekdays without `rest_on`).
    pub fn levels_on(&self, day: i64) -> [usize; 5] {
        let row = if self.rest_on { self.cal.row(day) } else { WEEKDAYS };
        std::array::from_fn(|q| self.period_slot[q] + SLOTS * row)
    }
    /// A slot's trains an hour, offset and phase (`levels_on`).
    pub fn slot(&self, s: usize) -> (f32, f64, f64) {
        let lev = s % SLOTS;
        if s < SLOTS { (self.tph[lev], self.off[lev], self.phase[lev]) } else { (self.rest_tph[lev], self.rest_off[lev], self.rest_phase[lev]) }
    }
}

/// Days of a line's own calendar kept around now (`Cal`): the clock worker sends from yesterday to
/// eight days ahead.
pub const CAL_DAYS: usize = 12;
/// A row of `Cal::rows` that follows the week (`params::row_of_day`).
pub const CAL_WEEK: u8 = u8::MAX;

/// A line's own calendar (holidays, its country's weekend, weekend days worked): the kind of day of
/// local days `day0 ..`, `CAL_WEEK` where the week decides; days outside follow the week. A plain
/// value so a `Lattice` stays one, and departures stay a function of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cal {
    pub day0: i64,
    pub rows: [u8; CAL_DAYS],
}

impl Cal {
    pub const NONE: Cal = Cal { day0: 0, rows: [CAL_WEEK; CAL_DAYS] };
    /// Every day around local day `day` of kind `row`, as if the line ran that kind of day for a
    /// while (`Service::run_deps_as`).
    pub fn all(day: i64, row: usize) -> Cal {
        Cal { day0: day - 2, rows: [row.min(DAY_ROWS - 1) as u8; CAL_DAYS] }
    }
    /// The kind of local day `day` (`params::WEEKDAYS`, `WEEKEND`).
    pub fn row(&self, day: i64) -> usize {
        let k = day - self.day0;
        match (0..CAL_DAYS as i64).contains(&k).then(|| self.rows[k as usize]) {
            Some(r) if (r as usize) < DAY_ROWS => r as usize,
            _ => row_of_day(day),
        }
    }
    /// From `(day, row)` pairs (any order; rows past `DAY_ROWS` ignored), the window from the first.
    pub fn from_days(days: &[(i64, u8)]) -> Cal {
        let Some(day0) = days.iter().map(|d| d.0).min() else { return Cal::NONE };
        let mut c = Cal { day0, rows: [CAL_WEEK; CAL_DAYS] };
        for &(d, r) in days {
            let k = d - day0;
            if (0..CAL_DAYS as i64).contains(&k) && (r as usize) < DAY_ROWS {
                c.rows[k as usize] = r;
            }
        }
        c
    }
}

/// The most a period's departures lead its start (a run over four hours): half the shortest period,
/// so the periods stay in order.
const LEAD_MAX_S: f64 = 2.0 * 3600.0;

/// The pattern a line ran before a change at `until` (T-206): departures before `until` keep it,
/// so trains already running keep their places; later departures follow the next pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct Era {
    pub until: f64,
    pub lat: Lattice,
    /// The trains drawn at `until` (run, departure): only these may stand at a terminal across
    /// the change for a departure of the new pattern, so a train that had left service never
    /// comes back.
    pub drawn: Vec<(u8, f64)>,
    /// T-3346 stage 2: the timetable variant each drawn train was running (run, departure, var):
    /// it keeps it where the line still has one by that number, so a change moves no train's end.
    /// Not saved (a loaded era's trains run their slot's).
    pub vars: Vec<(u8, f64, u8)>,
}

/// T-3346 stage 2: the variant an era's departure runs: the one its train was running when the era
/// ended, where the run still has one by that number.
fn era_var(e: &Era, run: &Run, r: usize, d: f64, lev: u8) -> u8 {
    e.vars.iter().find(|x| x.0 as usize == r && x.1 == d).map_or(0, |x| if (x.2 as usize) <= run.vars[lev as usize].len() { x.2 } else { 0 })
}

/// A departure: game (or local) time, slot, headway, and its place in the day (`place_of`), the
/// key of a run's variant pattern (T-3346 stage 2).
pub(crate) type Dep = (f64, u8, f64, u32);

/// T-3346 stage 2: a departure's place in its day: the stretch (`day_stretches`' index) and its
/// number from the stretch's first point. Weekend stretches have none (`NO_PLACE`): their trains run
/// their slot's timetable.
fn place_of(stretch: usize, k: i64, weekend: bool) -> u32 {
    if weekend || !(0..1 << 16).contains(&k) { NO_PLACE } else { (stretch as u32) << 16 | k as u32 }
}
/// A place no variant pattern holds.
pub const NO_PLACE: u32 = u32::MAX;

/// The longest a train stands at a terminal for its next departure before it leaves service.
pub const HANDOVER_MAX_S: f64 = 3600.0;
/// A train ready this little after a departure still takes it (a whole-line fleet's round trip is
/// solved to about 0.04 s of its trains times the headway).
const HANDOVER_TOL_S: f64 = 1.0;
/// At a change of pattern (a new period, an edit), a departure closer than this share of its
/// headway after the last one of the old pattern is left out, so trains never leave in pairs.
const MIN_GAP_SHARE: f64 = 0.5;
/// An era's pattern is looked back over this long for its last departure.
const ERA_LOOKBACK_S: f64 = 4.0 * 3600.0;
/// An era may end this far after the clock: an edit takes effect a moment ahead (T-1071).
const ERA_AHEAD_S: f64 = 3600.0;
const DAY_S: f64 = 86400.0;

/// The periods in the order of the day.
fn day_order() -> [usize; 5] {
    let mut o = [0, 1, 2, 3, 4];
    o.sort_by(|&a, &b| PERIOD_HOURS[a].0.total_cmp(&PERIOD_HOURS[b].0));
    o
}

/// The day's stretches of one level: runs of consecutive periods at the same level, as (first,
/// last) period, in day order. Departures are laid out per stretch, so a level that spans several
/// periods keeps one headway across their boundaries (T-1536).
/// `levels`: each period's slot on that day (`Lattice::levels_on`).
pub(crate) fn level_stretches(levels: &[usize; 5]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = vec![];
    for p in day_order() {
        match out.last_mut() {
            Some((_, last)) if levels[*last] == levels[p] && PERIOD_HOURS[*last].1 == PERIOD_HOURS[p].0 => *last = p,
            _ => out.push((p, p)),
        }
    }
    out
}

/// Local day `day`'s stretches of departures from midnight: runs of periods of one level
/// (`level_stretches`), or where the line's hours moved (`Lattice::hours`) runs of pieces of one
/// row, level and rate.
fn day_stretches(lat: &Lattice, day: i64) -> Vec<Stretch> {
    let levels = lat.levels_on(day);
    let h = &lat.hours;
    if h.n == 0 {
        return level_stretches(&levels).into_iter().map(|(first, last)| {
            let (tph, off, phase) = lat.slot(levels[first]);
            Stretch { h0: PERIOD_HOURS[first].0, h1: PERIOD_HOURS[last].1, lev: levels[first], tph, off, phase }
        }).collect();
    }
    let row = levels[0] / SLOTS;
    let mut out: Vec<(Stretch, u8)> = Vec::with_capacity(h.n as usize);
    for i in 0..h.n as usize {
        let p = h.pieces[i];
        let lev = levels[h.pieces[h.row_piece(i)].q as usize];
        let (tph, off, phase) = if h.own_on[i] { h.own[i][row] } else { lat.slot(lev) };
        match out.last_mut() {
            Some((x, k)) if *k == p.k && x.lev == lev && x.tph.to_bits() == tph.to_bits() && x.off.to_bits() == off.to_bits() && x.phase.to_bits() == phase.to_bits() => x.h1 = p.b as f64,
            _ => out.push((Stretch { h0: p.a as f64, h1: p.b as f64, lev, tph, off, phase }, p.k)),
        }
    }
    out.into_iter().map(|x| x.0).collect()
}

/// A stretch (`day_stretches`) of `day` for run `r` of a pattern: (first departure, headway,
/// departures), or None without service.
/// `next`: the trains an hour of the stretch after (the next day's first for the day's last).
fn stretch_points(lat: &Lattice, r: usize, day: i64, st: &Stretch, next: f32) -> Option<(f64, f64, i64)> {
    let (tph, off, phase) = (st.tph as f64, st.off, st.phase);
    if tph <= 0.0 || (r == 1 && off.is_infinite()) {
        return None;
    }
    let hw = 3600.0 / tph;
    let (h0, h1) = (st.h0, st.h1);
    // a period followed by no service runs to its own end, so a line's last trains of the evening
    // or of a peak are not pulled early (T-835)
    let end_lead = if next > 0.0 { lat.lead } else { 0.0 };
    let (start, end) = (day as f64 * DAY_S + h0 * 3600.0 - lat.lead, day as f64 * DAY_S + h1 * 3600.0 - end_lead);
    let ps = start + phase + if r == 1 { off } else { 0.0 };
    // the points ps + k hw before the period's end
    let mut n = ((end - ps) / hw).ceil() as i64;
    while n > 0 && ps + (n - 1) as f64 * hw >= end {
        n -= 1;
    }
    while ps + n as f64 * hw < end {
        n += 1;
    }
    if n <= 0 { None } else { Some((ps, hw, n)) }
}

/// One run's departures from a pattern in `[a, b)` of game time, with level and headway: the
/// pattern's periods are laid out in the line's local time (T-309, `Lattice::shift`), and each
/// departure is the same game time from any window.
pub(crate) fn lattice_deps(lat: &Lattice, r: usize, a: f64, b: f64, out: &mut Vec<Dep>) {
    let s = lat.shift;
    if s == 0.0 {
        return lattice_deps_local(lat, r, a, b, out);
    }
    // a second either side, so rounding in `a + s` never loses a departure on the window's edge
    let start = out.len();
    lattice_deps_local(lat, r, a + s - 1.0, b + s + 1.0, out);
    let mut k = start;
    for i in start..out.len() {
        let (d, lev, hw, place) = out[i];
        let d = d - s;
        if d >= a && d < b {
            out[k] = (d, lev, hw, place);
            k += 1;
        }
    }
    out.truncate(k);
}

/// One run's departures from a pattern in `[a, b)` of the line's local time, with level and
/// headway: per stretch of one level (`level_stretches`), `start - lead + off + k * headway`,
/// leaving out any closer than half a headway after the stretch before's last (MIN_GAP_SHARE). A
/// pure function of the pattern, so every window agrees.
pub(crate) fn lattice_deps_local(lat: &Lattice, r: usize, a: f64, b: f64, out: &mut Vec<Dep>) {
    if !(b > a) {
        return;
    }
    let d0 = (a / DAY_S).floor() as i64;
    let d1 = (b / DAY_S).floor() as i64;
    // the last departure before day d0: the day before's last stretch with service
    let mut last: Option<f64> = None;
    let mut next_day = day_stretches(lat, d0);
    let after = |st: &[Stretch], i: usize, next: &[Stretch]| st.get(i + 1).or(next.first()).map_or(0.0, |x| x.tph);
    {
        let st = day_stretches(lat, d0 - 1);
        for (i, s) in st.iter().enumerate().rev() {
            if let Some((ps, hw, n)) = stretch_points(lat, r, d0 - 1, s, after(&st, i, &next_day)) {
                last = Some(ps + (n - 1) as f64 * hw);
                break;
            }
        }
    }
    // the next day's first stretch starts `lead` before it
    for day in d0..=d1 + 1 {
        let st = std::mem::replace(&mut next_day, day_stretches(lat, day + 1));
        for (i, s) in st.iter().enumerate() {
            let Some((ps, hw, n)) = stretch_points(lat, r, day, s, after(&st, i, &next_day)) else { continue };
            let mut k0 = 0i64;
            if let Some(l) = last {
                while k0 < n && ps + k0 as f64 * hw - l < MIN_GAP_SHARE * hw {
                    k0 += 1;
                }
            }
            if k0 >= n {
                continue;
            }
            let lev = (s.lev % SLOTS) as u8;
            let weekend = s.lev >= SLOTS;
            let mut k = k0.max(((a - ps) / hw).floor() as i64 - 1).max(k0);
            while k < n {
                let d = ps + k as f64 * hw;
                if d >= b {
                    break;
                }
                if d >= a {
                    out.push((d, lev, hw, place_of(i, k, weekend)));
                }
                k += 1;
            }
            last = Some(ps + (n - 1) as f64 * hw);
        }
    }
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
    /// Per line, the patterns it ran before recent changes, oldest first (T-206). Never saved:
    /// kept only while trains of an old pattern may still be drawn, then pruned.
    pub eras: Vec<Vec<Era>>,
    /// Per line, trains running when an edit changed their profile: (run, departure, seconds
    /// their drawing is moved by), so each carries on from where it was (T-206). Pruned with eras.
    pub shifts: Vec<Vec<(u8, f64, f64)>>,
    /// T-1382: the lines whose service was rebuilt, rescheduled or had its profiles recomputed, or
    /// that the net was changed under (`note_changed`), since `take_changed`, and whether to take
    /// them all (a world just made or reloaded).
    changed: Vec<u32>,
    pub changed_all: bool,
    /// T-4299: the same lines for the model's kept money (`wasm/kept.rs`, `take_kept`), its own cursor
    kept_changed: Vec<u32>,
    kept_all: bool,
    /// Per line id, its own calendar (`Cal`): the kind of day of its holidays and its country's
    /// weekend. Not saved; the clock worker sends it (`TrackApi::set_line_rows`).
    pub cal: Vec<Cal>,
    /// T-3346 stage 2: the profiles `timing::retime` built last time, by what they were built
    /// from, so an edit elsewhere builds none of them again.
    pub timing_memo: std::collections::HashMap<(u32, u8, Vec<u64>), Profile>,
    /// T-4132: per cluster of lines `timing::retime` times together, by its first line: what it read
    /// last time and the state it left its lines' variants in; both the same again keep them.
    pub timing_keys: std::collections::HashMap<u32, (Vec<u64>, Vec<u64>)>,
    /// T-4132: while a trial times no train (`TrackWorld::trial_with`), the variants each line it
    /// rebuilds had before it, to put back after (`timing::put_back`); None otherwise.
    pub timing_held: Option<Vec<(u32, super::timing::Held)>>,
    /// T-4149: the lines whose runs lie anywhere new on the track since `take_moved` (`update_scoped`), for the clock
    /// worker's drawn runs; None for every line (a service just made, never taken from). Put back after a trial
    /// (`TrackWorld::trial_with`), with `spans`.
    pub(crate) moved: Option<Vec<u32>>,
    /// T-4281: per moved line and run, where it moved since `take_moved` (`Span`), None where that is not known; put
    /// back after a trial with `moved`.
    pub(crate) spans: std::collections::HashMap<(u32, u8), Option<Span>>,
    /// T-4281: the spans of the lines the last `take_moved` listed (`take_spans`)
    taken_spans: Vec<(u32, u8, Span)>,
}

/// T-4281: where a rebuilt line's run lies anywhere new: from offset `lo` to `hi` (offsets of the run now); before
/// `lo` its points are where they were, bit for bit; past `hi` they are the same track's, each moved along it by
/// at most `drift` metres (the run's length changed by about that much before them: a station splitting an edge moves
/// the pieces' lengths by a few mm). Several rebuilds since a take add up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub lo: f64,
    pub hi: f64,
    pub drift: f64,
}

/// T-4281: how far apart two offsets of the same entry past a change may be and still count as moved by the run's
/// length change alone, m
const SPAN_EPS_M: f64 = 1e-6;

/// T-4281: run `x`'s span against what it was (`segs`, `off`, `len`), its edges in `touched` (sorted) counted as
/// changed; None when nothing past the start is alike.
fn span_of(x: &Run, segs: &[u32], off: &[f64], len: f64, touched: &[u32]) -> Option<Span> {
    let same_edge = |p: u32| touched.binary_search(&path_edge(p)).is_err();
    let (n0, n1) = (segs.len().min(off.len()), x.segs.len().min(x.seg_off.len()));
    let mut i = 0;
    while i < n0 && i < n1 && segs[i] == x.segs[i] && off[i].to_bits() == x.seg_off[i].to_bits() && same_edge(x.segs[i]) {
        i += 1;
    }
    let shift = x.len - len;
    if !shift.is_finite() {
        return None;
    }
    // (the most an entry past the change moved along the run: the length change, give or take `SPAN_EPS_M`)
    let (mut j, mut drift) = (0, shift.abs());
    while j < n0 - i && j < n1 - i {
        let (a, b) = (n0 - 1 - j, n1 - 1 - j);
        let moved = x.seg_off[b] - off[a];
        if segs[a] != x.segs[b] || !same_edge(x.segs[b]) || (moved - shift).abs() > SPAN_EPS_M {
            break;
        }
        drift = drift.max(moved.abs());
        j += 1;
    }
    let lo = if i < n1 { x.seg_off[i] } else { x.len };
    let hi = if j > 0 { x.seg_off[n1 - j] } else { x.len };
    (i > 0 || j > 0).then_some(Span { lo, hi: hi.max(lo), drift })
}

/// One run of the stops and path its trains run (`line`: the whole plan, or the part of it a
/// partly built line runs, T-3474).
struct Served<'a> {
    stops: &'a [u32],
    path: &'a [u32],
    running: bool,
    circle: bool,
}

fn build_run(net: &Network, l: u32, on: &Served, rev: bool, train_len: f64, dwell: f64) -> Option<Run> {
    let path = on.path;
    let running = on.running;
    let mut run = Run { tram: net.line_tram(l), ..Default::default() };
    // T-3348: each run's own stop nodes, run 1's in its order
    let stops = if rev {
        let (stops, segs) = net.back_run_of(on.stops, path, net.line_ones(l))?;
        run.segs = segs;
        stops
    } else {
        run.segs = path.to_vec();
        net.fwd_stops_of(on.stops, path, net.line_ones(l))
    };
    let mut s = 0.0;
    run.nodes.push(net.path_nodes(run.segs[0]).0);
    run.node_s.push(0.0);
    let mut raw: Vec<[f64; 2]> = vec![];
    let top = net.line_run_top(l);
    for &p in &run.segs {
        let e = path_edge(p);
        let len = net.edge_len[e as usize];
        run.seg_off.push(s);
        // T-817: the lowest of the curve's speed, the track's cap and the trains' top speed; T-3027:
        // and a ramp's
        let cap = net.v_cap(e).min(top);
        let lims = net.edge_limits(e, cap);
        let mut add = |a: f64, v: f64| match raw.last() {
            Some(l) if l[1] == v => {}
            _ => raw.push([a, v]),
        };
        if path_dir(p) == 0 {
            for l in &lims {
                add(s + l.s0, l.v);
            }
        } else {
            for l in lims.iter().rev() {
                add(s + len - l.s1, l.v);
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
    let turns = net.stop_reversals(&run.segs);
    let (at, stopped) = net.match_run(&run.nodes, &stops, &turns, net.line_ones(l), &run.segs)?;
    run.stop_idx = at.into_iter().map(|k| k as u32).collect();
    run.away = stopped.iter().enumerate().filter(|x| !*x.1).map(|(j, _)| (j as u32, stops[j])).collect();
    if stops.iter().any(|&s| net.one_side(s).is_some()) || !run.away.is_empty() {
        run.pass = net.run_passes(&run.segs, &run.stop_idx, on.circle);
        for &(j, _) in &run.away {
            run.pass[j as usize] = true;
        }
        if !run.pass.contains(&true) {
            run.pass.clear();
        }
    }
    let node_s: Vec<f64> = run.stop_idx.iter().map(|&i| run.node_s[i as usize]).collect();
    // where it turns back at a stop it stands its turnaround there; anywhere else (a track end, T-797 B3, T-3595) too, as no stop
    let reversals = net.path_reversals(&run.segs);
    let stops_at: Vec<u32> = (0..stops.len()).filter(|&j| !run.away.iter().any(|x| x.0 as usize == j)).map(|j| run.stop_idx[j]).collect();
    let bare: Vec<usize> = reversals.iter().copied().filter(|&k| !stops_at.contains(&(k as u32))).collect();
    // Where trains stand still, in path order: a reversal's train takes at most half the gap to
    // its neighbours on each side.
    let mut marks: Vec<f64> = node_s.iter().copied().chain(bare.iter().map(|&k| run.node_s[k])).collect();
    marks.sort_by(f64::total_cmp);
    let room = |at: f64| {
        let i = marks.partition_point(|&m| m < at);
        let before = if i > 0 { at - marks[i - 1] } else { f64::INFINITY };
        let after = marks.get(i + 1).map_or(f64::INFINITY, |&m| m - at);
        before.min(after) / 2.0
    };
    for &k in &bare {
        let at = run.node_s[k];
        let d = (train_len / 2.0 + BARE_END_CLEAR_M).min(room(at));
        run.ends.push((k as u32, at - d, 2.0 * d));
    }
    for (j, &st) in stops.iter().enumerate() {
        let node_at = |i: usize| node_s[i];
        let mut at = node_at(j);
        // At a dead end the platform lies along the track: stop at its middle.
        // Unused blueprints do not move the stops of a constructed service. Planned
        // services still preview the complete planned station geometry.
        let ports = if running { net.built_ports(st) } else { net.node_ports[st as usize].n as usize };
        let turn = reversals.contains(&(run.stop_idx[j] as usize)) && !run.away.iter().any(|x| x.0 as usize == j);
        run.turn.push(turn);
        run.jump.push(0.0);
        if turn {
            // T-797: the train's middle stands d short of the node on arrival and d past it in
            // the unfolded offsets on leaving, the same place on the ground.
            let d = if ports == 1 { net.node_platform[st as usize] as f64 / 2.0 } else { train_len / 2.0 };
            let d = d.min(room(at));
            run.stop_s.push(at - d);
            run.jump[j] = 2.0 * d;
            continue;
        }
        if ports == 1 {
            let half = net.node_platform[st as usize] as f64 / 2.0;
            // never past halfway to the next stop (T-760: two stops closer than their platforms)
            at = if j == 0 {
                (at + half).min(if stops.len() > 1 { (at + node_at(1)) / 2.0 } else { at + half })
            } else {
                (at - half).max((at + node_at(j - 1)) / 2.0)
            };
        }
        run.stop_s.push(at);
    }
    run.limits = profile::widen(&raw, run.len, train_len / 2.0);
    let turn = net.line_turn[l as usize] as f64;
    let dw = stand_times(&run, dwell, turn);
    run.hops = match with_ends(&run, &dw, turn) {
        None => profile::hops(&run.limits, &run.stop_s, &run.jump, run.brake()),
        Some((s, jump, ..)) => profile::hops(&run.limits, &s, &jump, run.brake()),
    };
    run.free = run_profile(&run, &dw, turn, &[]);
    Some(run)
}

/// `profile::run` over a run's stops and the bare track ends it reverses at, standing `turn` at
/// each end, leaving out the stops it passes (T-3332); `arr`/`dep` keep every stop, a passed one
/// timed as the train goes by.
fn run_profile(run: &Run, dw: &[f64], turn: f64, holds: &[Hold]) -> Profile {
    let Some((s, jump, stand, keep)) = with_ends(run, dw, turn) else {
        return profile::run_with(&run.limits, &run.stop_s, &run.jump, dw, holds, &run.hops, run.brake());
    };
    let mut p = profile::run_with(&run.limits, &s, &jump, &stand, holds, &run.hops, run.brake());
    let n = run.stop_s.len();
    let (mut arr, mut dep) = (vec![0.0; n], vec![0.0; n]);
    for (i, k) in keep.iter().enumerate() {
        if let (Some(j), Some(&a), Some(&d)) = (*k, p.arr.get(i), p.dep.get(i)) {
            (arr[j], dep[j]) = (a, d);
        }
    }
    for j in (0..n).filter(|&j| run.passes(j)) {
        let t = profile::time_at(&p.phases, run.stop_s[j]);
        (arr[j], dep[j]) = (t, t);
    }
    (p.arr, p.dep) = (arr, dep);
    p
}

/// A run's stops with the bare track ends it reverses at put in and the stops it passes left out:
/// offsets, jumps, standing times and the stop each is (None for an end). None for a run with
/// neither.
#[allow(clippy::type_complexity)]
fn with_ends(run: &Run, dw: &[f64], turn: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<Option<usize>>)> {
    if run.ends.is_empty() && run.pass.is_empty() {
        return None;
    }
    let n = run.stop_s.len() + run.ends.len();
    let (mut s, mut jump, mut stand, mut keep) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut e = 0;
    for j in 0..run.stop_s.len() {
        while e < run.ends.len() && run.ends[e].0 < run.stop_idx[j] {
            s.push(run.ends[e].1);
            jump.push(run.ends[e].2);
            stand.push(turn);
            keep.push(None);
            e += 1;
        }
        if run.passes(j) {
            continue;
        }
        s.push(run.stop_s[j]);
        jump.push(run.jump[j]);
        stand.push(dw.get(j).copied().unwrap_or(0.0));
        keep.push(Some(j));
    }
    Some((s, jump, stand, keep))
}

/// Scheduled standing time per stop: the dwell, or the turnaround where trains reverse (T-797);
/// none at the ends or where trains pass (T-3332).
fn stand_times(run: &Run, dwell: f64, turn: f64) -> Vec<f64> {
    let n = run.stop_s.len();
    (0..n).map(|i| if i == 0 || i + 1 == n || run.passes(i) { 0.0 } else if run.turn[i] { turn } else { dwell }).collect()
}

fn build_line(net: &Network, l: u32) -> LineSvc {
    if !net.line_ok(l) || net.line_broken[l as usize] {
        return LineSvc::default();
    }
    if let Some(r) = net.line_bus[l as usize].as_ref() {
        return build_bus(net, l, r);
    }
    if let Some(r) = net.line_ferry[l as usize].as_ref() {
        return build_ferry(net, l, r);
    }
    // T-3474: a line not wholly constructed runs the stretch of its plan that is, and previews the
    // whole plan while none is; should that stretch not build, it previews as before
    let stretch = net.line_stretch(l);
    if let Some(s) = stretch.filter(|s| net.stretch_is_part(l, s)) {
        let ls = build_on(net, l, s.stops(net.line_stops(l)), s.path(net.line_path(l)), true);
        if ls.ok {
            return LineSvc { part: Some(s), ..ls };
        }
        return build_on(net, l, net.line_stops(l), net.line_path(l), false);
    }
    build_on(net, l, net.line_stops(l), net.line_path(l), stretch.is_some())
}

/// A rail line's service over `stops` and `path` (its plan, or the part of it it runs).
fn build_on(net: &Network, l: u32, stops: &[u32], path: &[u32], running: bool) -> LineSvc {
    let mut ls = LineSvc::default();
    ls.stops = stops.to_vec();
    if ls.stops.len() < 2 { return ls; }
    if !net.line_only(l).is_empty() {
        ls.only = stops.iter().map(|&s| net.stop_only(l, s)).collect();
    }
    ls.kind = net.loop_kind(stops, path);
    ls.one_way = net.runs_once(l, stops, path);
    // T-3348: the platforms each run stops at
    let ways = net.way_stops(stops, path, net.line_ones(l)).unwrap_or_default();
    let min_plat = stops.iter().chain(&ways.0).chain(&ways.1).map(|&s| net.node_platform[s as usize]).min().unwrap_or(0) as f64;
    let fit = (min_plat / CAR_LEN).floor().max(1.0) as u32;
    let want = net.line_cars[l as usize] as u32;
    ls.cars = if want == 0 { fit } else { want.min(fit) };
    ls.train_len = ls.cars as f64 * CAR_LEN;
    let dwell = net.line_dwell[l as usize] as f64;
    let on = Served { stops, path, running, circle: ls.kind == LoopKind::Circle };
    let first = build_run(net, l, &on, false, ls.train_len, dwell);
    // a loop that runs one way has no return run to build, whatever one-way track it goes round (T-3094)
    let second = build_run(net, l, &on, true, ls.train_len, dwell).or_else(|| if ls.one_way { first.clone() } else { None });
    match (first, second) {
        (Some(a), Some(b)) => {
            ls.runs = [a, b];
            ls.ok = true;
            ls.running = running;
            ls.shift = net.line_shift(l);
        }
        _ => {}
    }
    ls
}

/// A bus line's service (T-1464): its stops from `bus::plan`, each hop timed by the road it crosses
/// (`bus::Pace`, T-2904), standing the line's dwell at the train stations and a short stand at each
/// street stop (T-1798), with no capacity holds.
fn build_bus(net: &Network, l: u32, r: &super::bus::BusRoute) -> LineSvc {
    let mut ls = LineSvc::default();
    let plan = super::bus::plan_route(net, l, r, r.built);
    if plan.stops.len() < 2 {
        return ls;
    }
    let dwell = net.line_dwell[l as usize] as f64;
    let n = plan.stops.len();
    let fwd_s: Vec<f64> = plan.stops.iter().map(|s| s.s).collect();
    // T-3378: a BRT line at a BRT station stands no dwell and its stop costs `BRT_STATION_STOP_S`
    let brt_at = |s: &super::bus::BusStop| r.brt && s.station.is_some_and(|n| net.is_brt_station(n));
    let fwd_dw: Vec<f64> = plan.stops.iter().map(|s| if s.station.is_some() && !brt_at(s) { dwell } else { 0.0 }).collect();
    let fwd_cut: Vec<f64> = plan.stops.iter().map(|s| if brt_at(s) { super::streets::BUS_STOP_S - BRT_STATION_STOP_S } else { 0.0 }).collect();
    let len = plan.len;
    let pace = super::bus::Pace::along(net, &r.points, &plan.cum, &r.kinds, len).brt(r.brt);
    let rev_s: Vec<f64> = fwd_s.iter().rev().map(|s| len - s).collect();
    let rev_dw: Vec<f64> = fwd_dw.iter().rev().copied().collect();
    let rev_cut: Vec<f64> = fwd_cut.iter().rev().copied().collect();
    let rev_street: Vec<f64> = plan.street.iter().rev().map(|s| len - s).collect();
    // T-4380: a by-hand line whose start is no stop still runs from it: the trip is worked out from the route's end (a point
    // that is no stop, `lead`), and the stops' own times are kept
    let lead = plan.first_off && plan.stops[0].s > 0.0;
    let make = |s: Vec<f64>, dw: &[f64], cut: &[f64], street: &[f64], pace: &super::bus::Pace, pre: bool, post: bool| {
        let (mut ps, mut pd, mut pc) = (s.clone(), dw.to_vec(), cut.to_vec());
        if pre {
            ps.insert(0, 0.0);
            pd.insert(0, 0.0);
            pc.insert(0, 0.0);
        }
        if post {
            ps.push(len);
            pd.push(0.0);
            pc.push(0.0);
        }
        let mut p = super::bus::profile_cut(&ps, &pd, &pc, street, pace, r.rule() != super::bus::StopRule::Auto);
        if pre {
            p.arr.remove(0);
            p.dep.remove(0);
        }
        if post {
            p.arr.pop();
            p.dep.pop();
        }
        let top = p.phases.iter().map(|x| x.v).fold(0.0, f64::max);
        Run {
            len,
            turn: vec![false; n],
            jump: vec![0.0; n],
            limits: vec![[0.0, top]],
            free: p.clone(),
            holds: Default::default(),
            stop_holds: std::array::from_fn(|_| vec![0.0; n]),
            prof: std::array::from_fn(|s| if net.slot_worked(s) { p.clone() } else { Profile::default() }),
            stop_s: s,
            ..Default::default()
        }
    };
    ls.stops = plan.stops.iter().map(|s| s.id).collect();
    // T-2899: a route that ends where it starts is a circle
    if super::bus::is_closed(&r.points) {
        ls.kind = LoopKind::Circle;
        ls.one_way = net.line_one_way[l as usize];
    }
    ls.runs = [make(fwd_s, &fwd_dw, &fwd_cut, &plan.street, &pace, lead, false), make(rev_s, &rev_dw, &rev_cut, &rev_street, &pace.reversed(), false, lead)];
    ls.cars = 1;
    ls.train_len = super::bus::BUS_LEN_M;
    ls.ok = true;
    ls.running = r.built;
    ls.shift = net.line_shift(l);
    ls.bus = Some(plan);
    ls
}

/// A ferry line's service (T-2193): its ports in order, the boat accelerating and braking at
/// `ferry::BOAT_A` under the route's speeds, standing the line's dwell at the ports between its ends,
/// with no capacity holds.
fn build_ferry(net: &Network, l: u32, r: &super::ferry::FerryRoute) -> LineSvc {
    use super::ferry;
    let mut ls = LineSvc::default();
    let ports = net.line_stops(l);
    let plan = ferry::plan(net, ports, r);
    if plan.stops.len() < 2 {
        return ls;
    }
    let (n, len) = (plan.stops.len(), plan.len);
    let dwell = net.line_dwell[l as usize] as f64;
    let fwd_s: Vec<f64> = plan.stops.iter().map(|s| s.s).collect();
    let rev_s: Vec<f64> = fwd_s.iter().rev().map(|s| len - s).collect();
    let dw: Vec<f64> = (0..n).map(|i| if i == 0 || i + 1 == n { 0.0 } else { dwell }).collect();
    let fwd = ferry::limits(r, &plan.cum);
    let rev = ferry::reversed(&fwd, len);
    let make = |s: Vec<f64>, iv: &[(f64, f64, f64)]| {
        let p = ferry::profile(iv, &s, &dw);
        Run {
            len,
            turn: vec![false; n],
            jump: vec![0.0; n],
            limits: iv.iter().map(|x| [x.0, x.2]).collect(),
            free: p.clone(),
            holds: Default::default(),
            stop_holds: std::array::from_fn(|_| vec![0.0; n]),
            prof: std::array::from_fn(|s| if net.slot_worked(s) { p.clone() } else { Profile::default() }),
            stop_s: s,
            ..Default::default()
        }
    };
    ls.stops = ports.to_vec();
    if ferry::is_closed(ports) {
        ls.kind = LoopKind::Circle;
        ls.one_way = net.line_one_way[l as usize];
    }
    ls.runs = [make(fwd_s, &fwd), make(rev_s, &rev)];
    ls.cars = 1;
    ls.train_len = r.boat.len_m();
    ls.ok = true;
    ls.running = net.line_constructed(l);
    ls.shift = net.line_shift(l);
    ls.bus = Some(plan);
    ls.ferry = true;
    ls.boat = r.boat;
    ls
}

fn profile_for(net: &Network, l: u32, run: &Run, level: usize) -> Profile {
    if net.line_bus[l as usize].is_some() || net.line_ferry[l as usize].is_some() {
        return run.free.clone();
    }
    profile_with(net, l, run, &run.stop_holds[level], &run.holds[level])
}

/// A rail run's profile with these extra dwells per stop and track holds (T-3346 stage 2: a
/// variant's, or the slot's).
pub(crate) fn profile_with(net: &Network, l: u32, run: &Run, stop_holds: &[f64], holds: &[Hold]) -> Profile {
    let turn = net.line_turn[l as usize] as f64;
    let mut dw = stand_times(run, net.line_dwell[l as usize] as f64, turn);
    for (d, h) in dw.iter_mut().zip(stop_holds) {
        *d += h;
    }
    let mut p = run_profile(run, &dw, turn, holds);
    p.hold += stop_holds.iter().sum::<f64>();
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
        // T-4149: what places each rebuilt line's runs on the track, to tell the lines that run anywhere new
        let old_geo: Vec<(bool, [(Vec<u32>, Vec<f64>, f64); 2])> = rebuilt
            .iter()
            .map(|&l| { let ls = &self.lines[l as usize]; (ls.ok, std::array::from_fn(|r| (ls.runs[r].segs.clone(), ls.runs[r].seg_off.clone(), ls.runs[r].len))) })
            .collect();
        let old_times: Vec<(u32, [Vec<Vec<f64>>; 2])> = rebuilt
            .iter()
            .map(|&l| (l, std::array::from_fn(|r| (0..SLOTS).map(|lev| self.lines[l as usize].stop_to_stop_safe(r, lev)).collect())))
            .collect();
        let mut demand: Vec<u32> = rescheduled.to_vec();
        // The incremental capacity pass needs the old and new paths of the rebuilt lines.
        let mut path_edges: Vec<u32> = vec![];
        let mut path_nodes: Vec<u32> = vec![];
        // T-3945: both runs, since the way back can call at platforms the way out does not (a street
        // tram's stop on each side of the road): a platform only one run visits otherwise keeps the
        // old line's stop indices
        let mut add_path = |ls: &LineSvc| {
            if ls.running {
                for run in &ls.runs {
                    path_edges.extend(run.segs.iter().map(|&p| path_edge(p)));
                    path_nodes.extend_from_slice(&run.nodes);
                }
            }
        };
        for &l in rebuilt {
            add_path(&self.lines[l as usize]);
        }
        for &l in rebuilt {
            let (old_ok, old_run) = (self.lines[l as usize].ok, self.lines[l as usize].running);
            let old_stops = std::mem::take(&mut self.lines[l as usize].stops);
            let old_only = std::mem::take(&mut self.lines[l as usize].only);
            if let Some(held) = &mut self.timing_held {
                if !held.iter().any(|h| h.0 == l) {
                    held.extend(super::timing::take(&mut self.lines[l as usize]).map(|h| (l, h)));
                }
            }
            self.lines[l as usize] = build_line(net, l);
            let now = &self.lines[l as usize];
            if now.ok != old_ok || now.running != old_run || now.stops != old_stops || now.only != old_only {
                demand.push(l);
            }
        }
        self.note_moved(net, rebuilt, &old_geo, touched.map(|t| t.0));
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
                std::array::from_fn(|r| (0..SLOTS).map(|lev| ls.stop_to_stop_safe(r, lev)).collect())
            };
            let mut changed = false;
            for r in 0..2 {
                for lev in net.slots() {
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
                // its variants are worked out again once the edit has settled (`timing::retime`)
                ls.timed = None;
                let stand = ls.close_stand(net, l as u32);
                for lev in net.slots() {
                    let (a, b) = (&ls.runs[0].prof[lev], &ls.runs[1].prof[lev]);
                    ls.round_trip[lev] = ls.round_trip_of(a.duration, b.duration, stand);
                    ls.delay[lev] = a.hold + if ls.one_way { 0.0 } else { b.hold };
                    ls.trains[lev] = ls.fleet(ls.round_trip[lev], net.line_tph[l][lev] as f64);
                }
            }
            if changed {
                d.recomputed.push(l as u32);
                let moved = (0..2).any(|r| {
                    net.slots().any(|lev| {
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
        for r in self.cap.resources_of_lines(rebuilt.iter().chain(rescheduled).copied(), false) {
            d.shared.extend(self.cap.users_of(r as usize).iter().map(|u| u.line));
        }
        for v in [&mut d.shared, &mut demand] {
            v.sort_unstable();
            v.dedup();
        }
        d.demand = demand;
        self.note_changed(rebuilt);
        self.note_changed(rescheduled);
        self.note_changed(&d.recomputed);
        d
    }

    /// T-4132: the next `timing::retime` works every line out again, as from a load.
    pub fn forget_timing(&mut self) {
        self.timing_keys.clear();
        self.timing_memo.clear();
    }

    /// T-4149: of the lines just rebuilt (`was`: each one's ok and runs' entries, offsets and length before), those whose
    /// runs can lie anywhere new: a train line whose entries, offsets or length moved, or with an edge under it in
    /// `edges` (touched: its shape may have), every bus and ferry line (laid along its route), every line after a global
    /// pass (`edges` None). An edit rebuilds every bus line and every line with a way back on most edits; most run where
    /// they ran.
    fn note_moved(&mut self, net: &Network, rebuilt: &[u32], was: &[(bool, [(Vec<u32>, Vec<f64>, f64); 2])], edges: Option<&[u32]>) {
        let Some(moved) = self.moved.as_mut() else { return };
        let mut touched = edges.map(|e| e.to_vec());
        if let Some(t) = touched.as_mut() {
            t.sort_unstable();
            t.dedup();
        }
        for (&l, (ok, runs)) in rebuilt.iter().zip(was) {
            let ls = &self.lines[l as usize];
            let bits = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            let same = touched.as_ref().is_some_and(|t| {
                ls.ok == *ok && ls.bus.is_none() && net.route_points(l).is_none()
                    && (0..2).all(|r| {
                        let (x, (segs, off, len)) = (&ls.runs[r], &runs[r]);
                        x.segs == *segs && x.len.to_bits() == len.to_bits() && bits(&x.seg_off, off) && !x.segs.iter().any(|&p| t.binary_search(&path_edge(p)).is_ok())
                    })
            });
            if !same {
                moved.push(l);
                // T-4281: a train line that stays ok, on edges an edit touched (not a global pass): where each run moved
                let local = ls.ok && *ok && ls.bus.is_none() && net.route_points(l).is_none();
                for (r, (segs, off, len)) in runs.iter().enumerate() {
                    let s = touched.as_ref().filter(|_| local).and_then(|t| span_of(&ls.runs[r], segs, off, *len, t));
                    let key = (l, r as u8);
                    let merged = match self.spans.get(&key) {
                        None => s,
                        Some(None) => None,
                        // (the earlier span in the offsets of now: it moved by at most the later drift)
                        Some(Some(a)) => s.map(|b| Span { lo: a.lo.min(b.lo) - b.drift, hi: a.hi.max(b.hi) + b.drift, drift: a.drift + b.drift }),
                    };
                    self.spans.insert(key, merged);
                }
            }
        }
        if moved.len() > 1 << 16 {
            self.moved = None;
            self.spans.clear();
        }
    }

    /// T-4149: the lines whose runs lie anywhere new on the track since the last call (`note_moved`), ascending;
    /// `[u32::MAX]` for all of them.
    pub fn take_moved(&mut self) -> Vec<u32> {
        let spans = std::mem::take(&mut self.spans);
        match self.moved.replace(vec![]) {
            None => {
                self.taken_spans.clear();
                vec![u32::MAX]
            }
            Some(mut v) => {
                v.sort_unstable();
                v.dedup();
                self.taken_spans = spans.into_iter().filter_map(|((l, r), s)| s.map(|s| (l, r, s))).collect();
                self.taken_spans.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
                v
            }
        }
    }

    /// T-4281: the spans of the runs the last `take_moved` listed that are known (`Span`), by line and run; a listed
    /// line with no span here may lie anywhere new
    pub fn take_spans(&mut self) -> Vec<(u32, u8, Span)> {
        std::mem::take(&mut self.taken_spans)
    }

    pub fn note_changed(&mut self, lines: &[u32]) {
        if self.changed.len() > 1 << 20 {
            self.changed.clear();
            self.changed_all = true;
        }
        self.changed.extend_from_slice(lines);
        if self.kept_changed.len() > 1 << 20 {
            self.kept_changed.clear();
            self.kept_all = true;
        }
        self.kept_changed.extend_from_slice(lines);
    }

    /// Every line changed (a world just made or reloaded), for both readers.
    pub fn note_all(&mut self) {
        self.changed_all = true;
        self.kept_all = true;
    }

    /// T-4299: the lines noted since the last call (`note_changed`), ascending, or None for every line.
    pub fn take_kept(&mut self) -> Option<Vec<u32>> {
        let mut v = std::mem::take(&mut self.kept_changed);
        if std::mem::take(&mut self.kept_all) {
            return None;
        }
        v.sort_unstable();
        v.dedup();
        Some(v)
    }

    /// The lines whose service changed since the last call, ascending; `[u32::MAX]` for all of them.
    pub fn take_changed(&mut self) -> Vec<u32> {
        let mut v = std::mem::take(&mut self.changed);
        if std::mem::take(&mut self.changed_all) {
            return vec![u32::MAX];
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// `take_changed` without taking.
    pub fn peek_changed(&self) -> Vec<u32> {
        if self.changed_all {
            return vec![u32::MAX];
        }
        let mut v = self.changed.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The round trips `lines` would have at the rates `net` holds now, from the capacity solve
    /// alone, leaving the service as it is (T-1787: an edit's settle probes with these). A hold adds
    /// its delay to the run, so each run's duration moves by the change in its holds' delays; the
    /// profiles, most of a settle's cost, are not worked out.
    pub fn probe_trips(&self, net: &Network, lines: &[u32]) -> Vec<[f64; SLOTS]> {
        let (delay, affected) = self.cap.probe(net, &self.lines, lines);
        let holds = capacity::holds_with(&self.cap, &delay, &self.lines, Some(&affected));
        let total = |h: &[Hold], stops: &[f64]| h.iter().filter(|h| h.delay >= HOLD_MIN_S).map(|h| h.delay).sum::<f64>() + stops.iter().sum::<f64>();
        lines.iter().map(|&l| {
            let ls = &self.lines[l as usize];
            if !ls.ok || !affected[l as usize] {
                return ls.round_trip;
            }
            let stand = ls.close_stand(net, l);
            std::array::from_fn(|lev| {
                if !net.slot_worked(lev) {
                    return ls.round_trip[lev];
                }
                let [a, b] = [0, 1].map(|r| {
                    let run = &ls.runs[r];
                    let (h, sh) = &holds[l as usize][r][lev];
                    run.prof[lev].duration + total(h, sh) - total(&run.holds[lev], &run.stop_holds[lev])
                });
                ls.round_trip_of(a, b, stand)
            })
        }).collect()
    }

    /// Every line from scratch.
    pub fn update_all(&mut self, net: &Network) -> ServiceDirty {
        let all: Vec<u32> = (0..net.line_count() as u32).collect();
        self.update(net, &all, &all, true)
    }

    /// T-2607: the time a headway line's round trip leaves its trains at `lev`, stood at its ends
    /// (1e-3 trains: what a settle leaves of a round trip that is whole headways), and a count line's
    /// kept in step at a merge (T-2973); 0 for any other fleet set by count and a loop.
    pub fn spare(&self, net: &Network, l: usize, lev: usize) -> f64 {
        let t = net.line_tph.get(l).map_or(0.0, |x| x[lev] as f64);
        self.spare_with(net, l, lev, t, net.auto_at(l, lev))
    }
    /// `spare` at `t` trains an hour for a place keeping headway `auto` (T-3542: a row's own rate in
    /// a slot its hours share with another).
    pub fn spare_with(&self, net: &Network, l: usize, lev: usize, t: f64, auto: Option<u32>) -> f64 {
        self.spare_paced(net, l, lev, t, auto, net.line_pace_of(l as u32)[lev])
    }
    /// `spare_with` for a line paced at `pace` (T-3735: the weekends' pacing).
    fn spare_paced(&self, net: &Network, l: usize, lev: usize, t: f64, auto: Option<u32>, pace: f32) -> f64 {
        let Some(ls) = self.lines.get(l) else { return 0.0 };
        let paced = auto.is_some_and(|a| a > 0) || pace > 0.0;
        if t <= 0.0 || ls.is_loop() || !paced {
            return 0.0;
        }
        let h = 3600.0 / t;
        let trip = ls.trip_len(net, l as u32, 0, lev) + ls.trip_len(net, l as u32, 1, lev);
        ((trip / h - 1e-3).ceil() * h - trip).clamp(0.0, h)
    }

    /// T-3542: line `l`'s trains an hour in each demand period as demand reads them: its slot's
    /// rate, but in a period its hours share between rows (`Network::mixed`) the rate of the mean
    /// headway its riders meet there. Riders arrive by the period's shape (`span::period_cum`), each
    /// row's piece of the period waits its own headway, and a piece with no trains counts for
    /// nobody: those riders are the span's (`span::line_span`, the share with a train due).
    pub fn demand_tph(&self, net: &Network, l: usize) -> [f32; 5] {
        let ps = net.period_slot();
        let Some(t) = net.line_tph.get(l) else { return [0.0; 5] };
        let mut out: [f32; 5] = std::array::from_fn(|q| t[ps[q]]);
        let (Some(m), Some(ls)) = (net.mix(l), self.lines.get(l)) else { return out };
        for q in 0..5 {
            let s = ps[q];
            if !m.mixed[s] { continue; }
            let (mut riders, mut wait) = (0.0, 0.0);
            for p in m.pieces[..m.n_pieces].iter().filter(|p| p.q as usize == q) {
                let r = net.place_tph(l, s, p.k as usize, t[s], ls.round_trip[s]) as f64;
                let c = super::span::period_cum(q, p.b as f64 * 3600.0) - super::span::period_cum(q, p.a as f64 * 3600.0);
                if r > 0.0 && c > 0.0 {
                    riders += c;
                    wait += c * 3600.0 / r;
                }
            }
            out[q] = if riders > 0.0 && wait > 0.0 { (3600.0 * riders / wait) as f32 } else { 0.0 };
        }
        out
    }

    /// T-4227: line `l`'s fleet on kind of day `row` (`params::WEEKDAYS`, `WEEKEND`), as its
    /// departures share a slot its hours mix out by (`Network::place_tph_in`): on weekends the
    /// weekend fleet where it runs weekend rates of its own (`rest_fleet`), else the weekdays'.
    pub fn fleet_on(&self, net: &Network, l: usize, row: usize) -> Option<DayFleet> {
        if row == WEEKEND && net.runs_own_weekend(l) { self.rest_fleet(net, l) } else { net.fleet(l) }
    }

    /// T-4227: line `l`'s weekend fleet: its weekend trains at each place (`Network::weekend_places_by`;
    /// a headway line at a percent the trains its weekend headway takes over the round trip, not cut
    /// to what the track lets run, which cuts the slot's rate instead), its weekend headways, and its
    /// settled weekend rates and pacing.
    pub fn rest_fleet(&self, net: &Network, l: usize) -> Option<DayFleet> {
        let ls = self.lines.get(l);
        let trains = net.weekend_places_by(l, |slots, h| {
            let ls = ls?;
            let each = if ls.paired() { 2 } else { 1 };
            Some(slots.iter().map(|&s| each * super::world::auto_trains(ls.round_trip[s] / each as f64, h)).max().unwrap_or(0))
        })?;
        Some(DayFleet { trains, auto: net.weekend_auto(l), tph: std::array::from_fn(|s| net.line_tph_on(l, WEEKEND, s)), pace: net.line_pace_rest_of(l as u32) })
    }

    /// A running line's departure pattern now (None when it is not running).
    pub fn lattice(&self, net: &Network, l: usize) -> Option<Lattice> {
        let ls = self.lines.get(l)?;
        if !ls.running || l >= net.line_count() {
            return None;
        }
        let tph = net.line_tph[l];
        // The reverse departure follows the forward trip's turnaround, so an odd whole-line fleet
        // stays evenly spaced around the complete circuit.
        let off_at = |lev: usize, t: f32, split: &[f32; SLOTS], spare: &dyn Fn(usize, f64) -> f64| {
            let t = t as f64;
            if ls.one_way {
                return f64::INFINITY;
            }
            if net.line_trains[l].is_none() || t <= 0.0 || ls.is_loop() {
                return 0.0;
            }
            let (h, there) = (3600.0 / t, ls.trip_len(net, l as u32, 0, lev));
            // T-2607, T-2972: a headway line's trains wait out the time its round trip leaves them,
            // the share the spacing chose at its last stop, the rest at its first
            (there + split[lev] as f64 * spare(lev, t)) % h
        };
        let split = net.line_split_of(l as u32);
        let off_of = |lev: usize, t: f32| off_at(lev, t, &split, &|lev, _| self.spare(net, l, lev));
        // (a slot no period uses keeps 0, as one read back from a save's eras)
        let off = std::array::from_fn(|lev| if net.slot_worked(lev) { off_of(lev, tph[lev]) } else { 0.0 });
        // a loop's middle is half a lap from its first stop (T-798): the same figure
        let lead = (0.25 * (ls.runs[0].prof[MEDIUM].duration + ls.runs[1].prof[MEDIUM].duration)).clamp(0.0, net.mix(l).map_or(LEAD_MAX_S, |m| m.lead_max_s.min(LEAD_MAX_S)));
        // T-1925: under a headway, whatever the headway has become since it was placed
        let phase_in = |p: &[f32; SLOTS], lev: usize, t: f32| if t > 0.0 { (p[lev] as f64).rem_euclid(3600.0 / t as f64) } else { 0.0 };
        let p = net.line_phase_of(l as u32);
        let phase = std::array::from_fn(|lev| phase_in(&p, lev, tph[lev]));
        // weekends at their own settled rates, which a line running its weekday counts then has too
        // where others' counts change what the shared track lets it run
        let rest_tph: [f32; SLOTS] = std::array::from_fn(|lev| if net.rest_solved { net.line_tph_on(l, WEEKEND, lev) } else { tph[lev] });
        // T-3735: a line running other rates on weekends lays its weekend departures out on its own
        // headways, pacing, spare time split and phases (`TrackWorld::place_weekend`), over the run
        // times its weekend trains are drawn on (the weekdays'); any other its weekday pattern
        let own = rest_tph != tph;
        let (split_r, p_r, auto_r, pace_r) = if own {
            (net.line_split_rest_of(l as u32), net.line_phase_rest_of(l as u32), net.weekend_auto(l), net.line_pace_rest_of(l as u32))
        } else {
            (split, p, net.auto_of(l), net.line_pace_of(l as u32))
        };
        // T-4227: a slot its hours share between rows shares its weekend rate out by the weekend's
        // trains (`fleet_on`), so a row with none there on weekdays runs its weekend ones
        let fleet_r = net.mix(l).and_then(|_| self.fleet_on(net, l, WEEKEND));
        let spare_r = |lev: usize, t: f64| self.spare_paced(net, l, lev, t, auto_r.map(|a| a[net.place(l, lev)]), pace_r[lev]);
        // T-3542: pieces of a slot the line's hours share between rows run each row's own rate
        let hours_of = |rest: &[f32; SLOTS]| -> Hours {
            let Some(m) = net.mix(l) else { return Hours::NONE };
            let (auto, ps) = (net.auto_of(l), net.period_slot());
            let mut h = Hours { n: m.n_pieces as u8, pieces: m.pieces, ..Hours::NONE };
            for (i, pc) in m.pieces[..m.n_pieces].iter().enumerate() {
                h.own_on[i] = m.mixed[ps[pc.q as usize]];
            }
            for (i, pc) in m.pieces[..m.n_pieces].iter().enumerate() {
                if !h.own_on[i] { continue; }
                let s = ps[pc.q as usize];
                // T-3931: laid out on the slot whose timetable it runs (`Hours::row_piece`)
                let sr = ps[m.pieces[h.row_piece(i)].q as usize];
                let k = pc.k as usize;
                let (a, ar) = (auto.map(|a| a[k]), auto_r.map(|a| a[k]));
                let tk = net.place_tph(l, s, k, tph[s], ls.round_trip[s]);
                h.own[i][WEEKDAYS] = (tk, off_at(sr, tk, &split, &|lev, t| self.spare_with(net, l, lev, t, a)), phase_in(&p, sr, tk));
                let tk = fleet_r.as_ref().map_or(rest[s], |f| net.place_tph_in(l, s, k, rest[s], ls.round_trip[s], f));
                h.own[i][WEEKEND] = (tk, off_at(sr, tk, &split_r, &|lev, t| self.spare_paced(net, l, lev, t, ar, pace_r[lev])), phase_in(&p_r, sr, tk));
            }
            h
        };
        let week = Lattice { tph, off, lead, shift: ls.shift, phase, rest_tph: tph, rest_off: off, rest_phase: phase, rest_on: false, cal: Cal::NONE, period_slot: net.period_slot(), hours: hours_of(&tph) };
        // T-3521: no line runs other trains on weekends: every day the weekdays' pattern, at no cost
        if !net.rest_solved {
            return Some(self.joined(net, l, week));
        }
        let rest_off = std::array::from_fn(|lev| if net.slot_worked(lev) { off_at(lev, rest_tph[lev], &split_r, &spare_r) } else { 0.0 });
        // T-3333: a twin's phase is pinned half its parent's headway on (`pin_twin_phases`), which
        // another headway turns into any other share: on weekends it is half their own
        let twin_of = match net.line_twin_of(l as u32) { crate::track::twin::Twin::Of(p) => Some(p as usize), _ => None };
        let rest_phase: [f64; SLOTS] = std::array::from_fn(|lev| {
            let t = rest_tph[lev];
            match twin_of {
                Some(q) if t > 0.0 => {
                    let pt = net.line_tph_on(q, WEEKEND, lev);
                    let qp = if own { net.line_phase_rest_of(q as u32) } else { net.line_phase_of(q as u32) };
                    let pp = qp[lev] as f64 + if pt > 0.0 { 1800.0 / pt as f64 } else { 0.0 };
                    pp.rem_euclid(3600.0 / t as f64)
                }
                _ => phase_in(&p_r, lev, t),
            }
        });
        // rates equal to the weekdays' run the weekdays' pattern, so they keep `rest_on` off and the
        // lattice equal to one read back from a save, whose eras keep no weekend rates
        if rest_tph == tph && rest_off == off && rest_phase == phase {
            return Some(self.joined(net, l, week));
        }
        let cal = self.cal.get(l).copied().unwrap_or(Cal::NONE);
        Some(self.joined(net, l, Lattice { rest_tph, rest_off, rest_phase, rest_on: true, cal, hours: hours_of(&rest_tph), ..week }))
    }

    /// T-3542: `lat` with the day's periods of one row of line `l` that run alike joined into one
    /// slot, so they lay out one stretch of departures with one headway (`level_stretches`): a line
    /// of three rows in a game of five keeps one high stretch 6-20 while its morning, midday and
    /// afternoon slots settle to the same rates, offsets, phases and round trips. Only consecutive
    /// periods of a row join (morning, midday, afternoon), never two rows.
    fn joined(&self, net: &Network, l: usize, mut lat: Lattice) -> Lattice {
        let Some(ls) = self.lines.get(l) else { return lat };
        let ps = &mut lat.period_slot;
        for (q, p) in [(0, 1), (1, 2)] {
            let (j, s) = (ps[q], ps[p]);
            let same = |a: &[f64; SLOTS]| a[j].to_bits() == a[s].to_bits();
            if s != j && net.mixed(l, s).is_none() && net.mixed(l, j).is_none() && net.place(l, s) == net.place(l, j) && lat.tph[j].to_bits() == lat.tph[s].to_bits() && lat.rest_tph[j].to_bits() == lat.rest_tph[s].to_bits()
                && same(&lat.off) && same(&lat.phase) && same(&lat.rest_off) && same(&lat.rest_phase) && same(&ls.round_trip) {
                ps[p] = j;
            }
        }
        lat
    }

    /// Every line's local time again, after the time zone table changed (T-309).
    pub fn reshift(&mut self, net: &Network) {
        for (l, ls) in self.lines.iter_mut().enumerate() {
            ls.shift = if ls.ok { net.line_shift(l as u32) } else { 0.0 };
        }
    }

    /// Every line's pattern and every train running at `now`, before an edit (`note_changes`
    /// after it).
    pub fn before_edit(&self, net: &Network, now: f64) -> Before {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let lat = (0..self.lines.len()).map(|l| self.lattice(net, l)).collect();
        let mut flight = vec![];
        if now.is_finite() {
            let mut keys = TrainKeys::new(self.lines.len());
            for t in self.trips(net, now, now + 1e-3) {
                let ls = &self.lines[t.line as usize];
                let p = ls.prof_of(t.run as usize, t.level as usize, t.var);
                let key = keys.of(ls, t.line, t.run, t.level, t.var);
                let tau = now - t.draw;
                if tau < 0.0 {
                    continue;
                }
                if tau >= p.duration {
                    // standing at the terminal: as long again after the new profile's arrival
                    flight.push(InFlight { line: t.line, run: t.run, level: t.level, var: t.var, dep: t.dep, s: f64::INFINITY, stopped_left: Some(tau - p.duration), at: None, key });
                    continue;
                }
                let (s, v) = profile::state_at(&p.phases, tau);
                let stopped_left = (v < 0.01).then(|| time_reaching(&p.phases, p.duration, s + 0.05) - tau);
                let at = ground_at(&ls.runs[t.run as usize], s);
                flight.push(InFlight { line: t.line, run: t.run, level: t.level, var: t.var, dep: t.dep, s, stopped_left, at, key });
            }
        }
        let lens = (0..self.lines.len()).map(|l| self.trip_lens(net, l)).collect();
        #[cfg(not(target_arch = "wasm32"))]
        check::before(t0, flight.len());
        Before { lat, flight, lens, noted: self.changed.len() }
    }

    /// T-4133: per line, whether the edit since `before` noted it (`note_changed`: rebuilt,
    /// rescheduled, its profiles recomputed or its variants moved by `timing::retime`, or the net
    /// changed under it): what the clock worker reads again. A line it did not note runs the
    /// profiles and variants it ran, so its trains stand where they were.
    fn touched_since(&self, before: &Before) -> Vec<bool> {
        let n = self.lines.len();
        if self.changed_all || self.changed.len() < before.noted {
            return vec![true; n];
        }
        let mut on = vec![false; n];
        for &l in &self.changed[before.noted..] {
            if let Some(x) = on.get_mut(l as usize) {
                *x = true;
            }
        }
        on
    }

    /// A line's trip lengths (run and turnaround) per run and slot.
    fn trip_lens(&self, net: &Network, l: usize) -> [f64; 2 * SLOTS] {
        let ls = &self.lines[l];
        std::array::from_fn(|k| if ls.ok && l < net.line_count() && net.slot_worked(k % SLOTS) { ls.trip_len(net, l as u32, k / SLOTS, k % SLOTS) } else { 0.0 })
    }

    /// After an edit at game time `now` (T-206): a running line whose pattern changed keeps its
    /// old one for departures before `now`, so its trains in service run on; a line not running
    /// before had no departures. A line that stops running forgets its eras. A train running
    /// when its profile changed is drawn from where it was (`shifts`): at the same place, or
    /// standing at the same stop for the time it had left there.
    pub fn note_changes(&mut self, net: &Network, before: &Before, now: f64) {
        if !now.is_finite() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let (t0, shadow) = (std::time::Instant::now(), check::shadow(self));
        // T-4133: the trains to move again: those of lines the edit touched or whose pattern moved
        let mut moved = if check::full() { vec![true; self.lines.len()] } else { self.touched_since(before) };
        // (a line noted is often one a settle scheduled again as it was: `train_key` below finds its
        // trains unchanged)
        self.eras.resize(self.lines.len(), vec![]);
        self.shifts.resize(self.lines.len(), vec![]);
        for l in 0..self.lines.len() {
            let Some(after) = self.lattice(net, l) else {
                self.eras[l].clear();
                self.shifts[l].clear();
                continue;
            };
            let was = before.lat.get(l).copied().flatten().unwrap_or(Lattice::NONE);
            let lens = self.trip_lens(net, l);
            let reshaped = before.lens.get(l).is_some_and(|b| b.iter().zip(&lens).any(|(x, y)| (x - y).abs() > 1e-6));
            moved[l] |= was != after || reshaped;
            let v = &mut self.eras[l];
            match v.last() {
                // a change at the same instant: the pattern before the instant stands for
                // everything before it
                Some(e) if e.until >= now => {}
                _ if was != after || reshaped => {
                    let drawn = before.flight.iter().filter(|f| f.line as usize == l).map(|f| (f.run, f.dep)).collect();
                    let vars = before.flight.iter().filter(|f| f.line as usize == l && f.var > 0).map(|f| (f.run, f.dep, f.var)).collect();
                    v.push(Era { until: now, lat: was, drawn, vars })
                }
                _ => {}
            }
        }
        let (full, mut keys) = (check::full(), TrainKeys::new(self.lines.len()));
        #[cfg(not(target_arch = "wasm32"))]
        let (mut walked, mut skip) = (0, u32::MAX);
        for f in before.flight.iter().filter(|f| moved.get(f.line as usize).copied().unwrap_or(true)) {
            let l = f.line as usize;
            let Some(ls) = self.lines.get(l).filter(|ls| ls.running && l < net.line_count()) else { continue };
            // T-3346 stage 2: the timetable that departure runs now (after the eras above)
            let var = self.var_now(net, l, f.run as usize, f.dep);
            // T-4133: on the profile and run it had, a train stands where it was (moving it again
            // would only round its shift)
            if !full && keys.of(ls, f.line, f.run, f.level, var) == f.key {
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                if check::skip_line(&mut skip, f.line) {
                    continue;
                }
                walked += 1;
            }
            let p = ls.prof_of(f.run as usize, f.level as usize, var);
            let moved = f.at.and_then(|a| offset_at(&ls.runs[f.run as usize], a));
            let f = &InFlight { s: moved.unwrap_or(f.s), ..*f };
            let tau = match f.stopped_left {
                Some(stood) if f.s.is_infinite() => p.duration + stood,
                Some(left) => {
                    let arrive = time_reaching(&p.phases, p.duration, f.s - 0.05);
                    let leave = time_reaching(&p.phases, p.duration, f.s + 0.05);
                    (leave - left).max(arrive).min(leave)
                }
                None => time_reaching(&p.phases, p.duration, f.s),
            };
            let by = now - tau - f.dep;
            let v = &mut self.shifts[l];
            v.retain(|x| !(x.0 == f.run && x.1 == f.dep));
            if by.abs() > 1e-3 {
                v.push((f.run, f.dep, by));
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        check::after(net, self, shadow, before, walked, now, t0);
    }

    /// Forget eras whose trains can no longer be drawn at `now`, or that lie more than an hour
    /// after it (the clock went back; an edit takes effect a moment ahead, T-1071).
    pub fn prune_eras(&mut self, net: &Network, now: f64) {
        if !now.is_finite() {
            return;
        }
        for l in 0..self.eras.len().max(self.shifts.len()) {
            if self.eras_of(l).is_empty() && self.shifts.get(l).map_or(true, |v| v.is_empty()) {
                continue;
            }
            let (dur, hcap) = if l < self.lines.len() { self.reach(net, l) } else { (0.0, 0.0) };
            let keep = now - 3600.0 - dur - 2.0 * hcap - 60.0;
            if let Some(v) = self.eras.get_mut(l) {
                v.retain(|e| e.until > keep && e.until <= now + ERA_AHEAD_S);
            }
            if let Some(v) = self.shifts.get_mut(l) {
                v.retain(|x| x.1 > keep && x.1 <= now + ERA_AHEAD_S);
            }
        }
    }

    /// The eras and shifts for a save (T-1071), so a game saved while trains change over loads
    /// with them where they were: per line that has any, its place among the live lines (as
    /// `save::encode` numbers them), then each era and each shift, little-endian. Empty when no
    /// line has any.
    /// T-3542 (`ERA3`, `wide`): every slot of each era and the slot of each period, where some era
    /// runs other slots than three rows' (`eras_wide`); otherwise each row's slot, as before slots.
    pub fn encode_eras(&self, net: &Network) -> Vec<u8> {
        let hours = self.eras_hours(net);
        let wide = hours || self.eras_wide(net);
        let mut out = vec![];
        let mut n = 0u32;
        let mut body = vec![];
        for (i, l) in (0..net.line_count()).filter(|&l| net.line_ok(l as u32)).enumerate() {
            let (eras, shifts) = (self.eras_of(l), self.shifts.get(l).map_or(&[][..], |v| &v[..]));
            if eras.is_empty() && shifts.is_empty() {
                continue;
            }
            n += 1;
            body.extend((i as u32).to_le_bytes());
            body.extend((eras.len() as u32).to_le_bytes());
            for e in eras {
                body.extend(e.until.to_le_bytes());
                // T-3542: per row (a 3-row line's slots, `ROW_SLOT`), as before slots, or every slot
                if wide {
                    e.lat.tph.iter().for_each(|x| body.extend(x.to_le_bytes()));
                    e.lat.off.iter().for_each(|x| body.extend(x.to_le_bytes()));
                } else {
                    slots_to_rows(e.lat.tph).iter().for_each(|x| body.extend(x.to_le_bytes()));
                    slots_to_rows(e.lat.off).iter().for_each(|x| body.extend(x.to_le_bytes()));
                }
                body.extend(e.lat.lead.to_le_bytes());
                body.extend(e.lat.shift.to_le_bytes());
                // T-1925 (`ERA2`): the departure phases
                if wide {
                    e.lat.phase.iter().for_each(|x| body.extend(x.to_le_bytes()));
                    body.extend(e.lat.period_slot.map(|s| s as u8));
                    // T-3542 (`ERA4`): the pieces of moved hours and their own rates, weekdays'
                    if hours {
                        let h = &e.lat.hours;
                        body.push(h.n);
                        for i in 0..h.n as usize {
                            let p = h.pieces[i];
                            body.extend([p.a, p.b, p.q, p.k, h.own_on[i] as u8]);
                            if h.own_on[i] {
                                let (t, o, ph) = h.own[i][WEEKDAYS];
                                body.extend(t.to_le_bytes());
                                body.extend(o.to_le_bytes());
                                body.extend(ph.to_le_bytes());
                            }
                        }
                    }
                } else {
                    slots_to_rows(e.lat.phase).iter().for_each(|x| body.extend(x.to_le_bytes()));
                }
                body.extend((e.drawn.len() as u32).to_le_bytes());
                for &(r, d) in &e.drawn {
                    body.push(r);
                    body.extend(d.to_le_bytes());
                }
            }
            body.extend((shifts.len() as u32).to_le_bytes());
            for &(r, d, by) in shifts {
                body.push(r);
                body.extend(d.to_le_bytes());
                body.extend(by.to_le_bytes());
            }
        }
        if n > 0 {
            out.extend(n.to_le_bytes());
            out.extend(body);
        }
        out
    }

    /// Eras and shifts from `encode_eras`, onto a service just built from the saved network.
    /// `phases`: the eras carry departure phases (`ERA2`, T-1925); older ones (`ERA1`) do not.
    /// T-3542: some era of a live line runs other slots than a line of three rows in a game of three
    /// (`encode_eras` then writes `ERA3`).
    pub fn eras_wide(&self, net: &Network) -> bool {
        (0..net.line_count()).filter(|&l| net.line_ok(l as u32)).any(|l| self.eras_of(l).iter().any(|e| e.lat.period_slot != PERIOD_SLOT_3))
    }

    /// T-3542: some era of a live line runs moved hours (`Lattice::hours`; `encode_eras` then writes`ERA4`).
    pub fn eras_hours(&self, net: &Network) -> bool {
        (0..net.line_count()).filter(|&l| net.line_ok(l as u32)).any(|l| self.eras_of(l).iter().any(|e| e.lat.hours.n > 0))
    }

    /// `wide`: `ERA3` eras, with every slot and each period's slot (T-3542); `hours`: `ERA4`, with
    /// the pieces of moved hours too.
    pub fn decode_eras(&mut self, bytes: &[u8], phases: bool, wide: bool, hours: bool) -> Result<(), String> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let b = bytes.get(at..at + n).ok_or("truncated eras")?;
            at += n;
            Ok(b)
        };
        macro_rules! num {
            ($t:ty) => {
                <$t>::from_le_bytes(take(std::mem::size_of::<$t>())?.try_into().unwrap())
            };
        }
        let lines = self.lines.len();
        let mut eras = vec![vec![]; lines];
        let mut shifts = vec![vec![]; lines];
        for _ in 0..num!(u32) {
            let l = num!(u32) as usize;
            let mut v = vec![];
            for _ in 0..num!(u32) {
                let until = num!(f64);
                let (mut tph, mut off, mut phase, mut period_slot) = ([0.0f32; SLOTS], [0.0; SLOTS], [0.0; SLOTS], PERIOD_SLOT_3);
                if wide {
                    for v in tph.iter_mut() { *v = num!(f32); }
                    for v in off.iter_mut() { *v = num!(f64); }
                } else {
                    tph = rows_to_slots([num!(f32), num!(f32), num!(f32)], 0.0);
                    off = rows_to_slots([num!(f64), num!(f64), num!(f64)], 0.0);
                }
                let (lead, shift) = (num!(f64), num!(f64));
                if wide {
                    for v in phase.iter_mut() { *v = num!(f64); }
                    for v in period_slot.iter_mut() { *v = (num!(u8) as usize).min(SLOTS - 1); }
                } else if phases {
                    phase = rows_to_slots([num!(f64), num!(f64), num!(f64)], 0.0);
                }
                let mut h = Hours::NONE;
                if hours {
                    h.n = num!(u8).min(PIECES_MAX as u8);
                    for i in 0..h.n as usize {
                        let (a, b, q, k, on) = (num!(u8), num!(u8), num!(u8).min(4), num!(u8).min(4), num!(u8));
                        h.pieces[i] = DayPiece { a, b, q, k };
                        h.own_on[i] = on != 0;
                        if on != 0 {
                            let o = (num!(f32), num!(f64), num!(f64));
                            h.own[i] = [o; DAY_ROWS];
                        }
                    }
                }
                let mut drawn = vec![];
                for _ in 0..num!(u32) {
                    drawn.push((num!(u8), num!(f64)));
                }
                // T-3521: weekend trains are not in the era bytes: an old pattern loads running its
                // weekday trains every day
                v.push(Era { until, lat: Lattice { tph, off, lead, shift, phase, rest_tph: tph, rest_off: off, rest_phase: phase, rest_on: false, cal: Cal::NONE, period_slot, hours: h }, drawn, vars: vec![] });
            }
            let mut s = vec![];
            for _ in 0..num!(u32) {
                s.push((num!(u8), num!(f64), num!(f64)));
            }
            if l < lines {
                eras[l] = v;
                shifts[l] = s;
            }
        }
        self.eras = eras;
        self.shifts = shifts;
        Ok(())
    }

    /// Each line's own calendar (`Cal`), by line id, from the clock worker (`TrackApi::set_line_rows`).
    /// A line whose rows changed on any day is noted as changed (`note_changed`); a window that only
    /// moved on is not. `noted`: false while no line runs other trains on weekends, so a calendar
    /// moves nothing (T-3521).
    pub fn set_cal(&mut self, cal: Vec<Cal>, noted: bool) {
        let n = self.cal.len().max(cal.len());
        let at = |c: &[Cal], l: usize| c.get(l).copied().unwrap_or(Cal::NONE);
        let changed: Vec<u32> = (0..n)
            .filter(|&l| {
                let (a, b) = (at(&self.cal, l), at(&cal, l));
                let lo = a.day0.min(b.day0);
                a != b && (lo..lo.max(a.day0.max(b.day0)) + CAL_DAYS as i64).any(|d| a.row(d) != b.row(d))
            })
            .map(|l| l as u32)
            .collect();
        self.cal = cal;
        if noted && !changed.is_empty() {
            self.note_changed(&changed);
        }
    }

    /// A line's eras (empty for most).
    pub fn eras_of(&self, l: usize) -> &[Era] {
        self.eras.get(l).map_or(&[], |v| &v[..])
    }

    /// T-3346 stage 2: the variant run `r`'s departure at `dep` (game time) takes now, as
    /// `run_deps` gives it: an era's train the one it ran, else its place's in the pattern the line's
    /// variants were worked out for; 0 for a line not timed train by train.
    pub fn var_now(&self, net: &Network, l: usize, r: usize, dep: f64) -> u8 {
        let Some(ls) = self.lines.get(l) else { return 0 };
        let Some(cur) = self.lattice(net, l) else { return 0 };
        let mut buf: Vec<Dep> = vec![];
        if let Some(e) = self.eras_of(l).iter().find(|e| dep < e.until) {
            lattice_deps(&e.lat, r, dep - 1.0, dep + 1.0, &mut buf);
            return buf.iter().find(|x| x.0 == dep).map_or(0, |x| era_var(e, &ls.runs[r], r, dep, x.1));
        }
        if ls.timed != Some(cur.timing_key()) {
            return 0;
        }
        lattice_deps(&cur, r, dep - 1.0, dep + 1.0, &mut buf);
        buf.iter().find(|x| x.0 == dep).map_or(0, |&(_, lev, _, place)| ls.runs[r].var_at(lev as usize, place))
    }

    /// The longest trip a line draws (run and turnaround) and the longest it may stand at a
    /// terminal for its next departure.
    fn reach(&self, net: &Network, l: usize) -> (f64, f64) {
        let ls = &self.lines[l];
        if !ls.ok {
            return (0.0, 0.0);
        }
        let mut dur = 0.0f64;
        for r in 0..2 {
            for lev in net.slots() {
                dur = dur.max(ls.trip_len(net, l as u32, r, lev));
                for v in 1..=ls.runs[r].vars[lev].len() {
                    dur = dur.max(ls.trip_len_var(net, l as u32, r, lev, v as u8));
                }
            }
        }
        let mut hw = 0.0f64;
        // T-3521: a weekend's longer headway too
        let cur = [net.line_tph[l], std::array::from_fn(|lev| net.line_tph_on(l, WEEKEND, lev))];
        for tph in cur.iter().chain(self.eras_of(l).iter().map(|e| &e.lat.tph)) {
            for &t in tph {
                if t > 0.0 {
                    hw = hw.max(3600.0 / t as f64);
                }
            }
        }
        (dur, hw.min(HANDOVER_MAX_S))
    }

    /// One direction's departures in `[a, b)` (T-164, T-206), unsorted, appended to `out` with
    /// `end` at the end of its run and turnaround. Before each era's `until` the era's pattern;
    /// after the last, the line's own. The boarding account uses it too, so its departures are the
    /// drawn ones to the bit.
    pub fn run_deps(&self, net: &Network, l: usize, r: usize, a: f64, b: f64, out: &mut Vec<Trip>) {
        self.run_deps_as(net, l, r, a, b, None, out)
    }

    /// `run_deps`, with every day from `a` to `b` of kind `row` (`params::WEEKDAYS`, `WEEKEND`)
    /// when given, whatever the line's calendar says: the boarding account runs each
    /// kind of day's trains on one reference day (`Boarding::kind_row`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_deps_as(&self, net: &Network, l: usize, r: usize, a: f64, b: f64, row: Option<usize>, out: &mut Vec<Trip>) {
        let Some(mut cur) = self.lattice(net, l) else { return };
        let ls0 = &self.lines[l];
        let as_row = |lat: &Lattice| -> Lattice {
            let mut x = *lat;
            if let Some(row) = row { x.cal = Cal::all(((a + ls0.shift) / 86400.0).floor() as i64, row); }
            x
        };
        cur = as_row(&cur);
        let ls = &self.lines[l];
        let eras = self.eras_of(l);
        let shifts = self.shifts.get(l).map_or(&[][..], |v| &v[..]);
        let mut seg_start = f64::NEG_INFINITY;
        let mut last: Option<f64> = None;
        let mut buf: Vec<Dep> = vec![];
        // T-3346 stage 2: the variants were worked out for the pattern the line runs now; an era's
        // trains run the one they ran at the change
        let timed = ls.timed.is_some() && ls.timed == Some(cur.timing_key());
        for i in 0..=eras.len() {
            let era = if i < eras.len() { as_row(&eras[i].lat) } else { cur };
            let (lat, seg_end) = (&era, if i < eras.len() { eras[i].until } else { f64::INFINITY });
            // an era is followed by another pattern, whose first departures need its last one
            let lo = if i < eras.len() { seg_start.max(a.min(seg_end - ERA_LOOKBACK_S)) } else { seg_start.max(a) };
            let hi = if i < eras.len() { seg_end } else { b };
            buf.clear();
            lattice_deps(lat, r, lo, hi, &mut buf);
            let prev = last;
            for &(d, lev, hw, place) in &buf {
                if prev.is_some_and(|p| d - p < MIN_GAP_SHARE * hw) {
                    continue;
                }
                last = Some(d);
                if d >= a && d < b {
                    let draw = d + shifts.iter().find(|x| x.0 as usize == r && x.1 == d).map_or(0.0, |x| x.2);
                    let var = if i < eras.len() {
                        era_var(&eras[i], &ls.runs[r], r, d, lev)
                    } else if timed {
                        ls.runs[r].var_at(lev as usize, place)
                    } else {
                        0
                    };
                    let end = draw + ls.trip_len_var(net, l as u32, r, lev as usize, var);
                    out.push(Trip { line: l as u32, run: r as u8, level: lev, dep: d, draw, end, hw, next: None, var });
                }
            }
            seg_start = seg_end;
        }
    }

    /// Hand trains over at a line's two terminals (T-206). A train that has finished its run and
    /// turnaround takes the next departure from that terminal if it comes within that
    /// departure's headway (at most HANDOVER_MAX_S), and stands there until it leaves: its trip's
    /// `end` moves to that departure. A departure no train is waiting for is a train entering
    /// service there; a train that finds another already waiting for the next departure, or no
    /// departure soon enough, leaves service at the end of its turnaround. In steady service with
    /// a whole-line fleet every train is ready exactly at its next departure, so nothing changes.
    /// On a loop (T-798) a train finishing a run takes that run's next departure, which is where
    /// it stands; its stand there is the timing hold.
    fn hand_over(trips: &mut [Trip], eras: &[Era], is_loop: bool) {
        let ready: Vec<f64> = trips.iter().map(|t| t.end).collect();
        // T-2429: a train standing at the terminal across a change, which its new, quicker run would
        // have had ready before it, waits from the change, and up to the half headway the new
        // pattern's first departure may be put back (MIN_GAP_SHARE)
        let from: Vec<f64> = trips
            .iter()
            .zip(&ready)
            .map(|(t, &r)| eras.iter().filter(|e| r < e.until && e.drawn.contains(&(t.run, t.dep))).fold(r, |m, e| m.max(e.until)))
            .collect();
        // whether train `w` may stand until `d`: within the departure's headway, and not across a
        // change unless it was drawn then
        let can = |trips: &[Trip], w: usize, d: f64, hw: f64| {
            let span = if from[w] > ready[w] { 1.0 + MIN_GAP_SHARE } else { 1.0 };
            d >= ready[w] - HANDOVER_TOL_S
                && d - from[w] <= HANDOVER_MAX_S.min(span * hw.max(1.0)) + HANDOVER_TOL_S
                && eras.iter().all(|e| !(ready[w] < e.until && d > e.until) || e.drawn.contains(&(trips[w].run, trips[w].dep)))
        };
        for (t, &f) in trips.iter_mut().zip(&from) {
            t.end = f;
        }
        for r in 0..2u8 {
            // trains finishing run r stand at the first stop of the other run, or of run r on a loop
            let mut ev: Vec<(f64, u8, usize)> = vec![];
            let mut deps: Vec<(f64, f64)> = vec![];
            for (i, t) in trips.iter().enumerate() {
                if t.run == r {
                    ev.push((ready[i] - HANDOVER_TOL_S, 0, i));
                }
                // a train running at a change and drawn on from where it was (`shifts`) is that
                // train still, even drawn from its first stop: no train standing there takes it
                let moved_on = t.draw != t.dep && eras.iter().any(|e| t.dep < e.until && e.drawn.contains(&(t.run, t.dep)));
                if (t.run == r) == is_loop && !moved_on {
                    ev.push((t.draw, 1, i));
                    deps.push((t.draw, t.hw));
                }
            }
            ev.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            deps.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut wait: Option<usize> = None;
            for &(_, kind, i) in &ev {
                if kind == 0 {
                    // two trains here: the first keeps its place unless it could not take the
                    // next departure anyway
                    if let Some(w) = wait {
                        let next = deps.get(deps.partition_point(|d| d.0 < ready[i] - HANDOVER_TOL_S)).copied();
                        if !next.is_some_and(|(d, hw)| can(trips, w, d, hw)) {
                            wait = Some(i);
                        }
                    } else {
                        wait = Some(i);
                    }
                } else if let Some(w) = wait.take() {
                    let (d, hw) = (trips[i].draw, trips[i].hw);
                    if can(trips, w, d, hw) {
                        trips[w].end = d.max(ready[w]);
                        trips[w].next = Some(d);
                    }
                }
            }
        }
    }

    /// Trips drawn in `[t0, t1)` (game seconds): each running line's departures, the trains
    /// handed over at the terminals (`hand_over`), those drawn at some time in the window.
    pub fn trips(&self, net: &Network, t0: f64, t1: f64) -> Vec<Trip> {
        let mut out: Vec<Trip> = vec![];
        let mut buf: Vec<Trip> = vec![];
        for l in 0..self.lines.len() {
            self.line_window(net, l, t0, t1, &mut buf, &mut out);
        }
        out.sort_by(|a, b| a.draw.total_cmp(&b.draw).then((a.line, a.run).cmp(&(b.line, b.run))));
        out
    }

    /// One running line's trips drawn in `[t0, t1)`, as `trips` gives them (T-2603, the line's
    /// timetable chart, whose window can be hours long).
    pub fn line_trips(&self, net: &Network, l: usize, t0: f64, t1: f64) -> Vec<Trip> {
        let mut out: Vec<Trip> = vec![];
        if l < self.lines.len() {
            self.line_window(net, l, t0, t1, &mut vec![], &mut out);
        }
        out.sort_by(|a, b| a.draw.total_cmp(&b.draw).then(a.run.cmp(&b.run)));
        out
    }

    /// Line `l`'s trips drawn in `[t0, t1)` onto `out`; `buf` is scratch.
    fn line_window(&self, net: &Network, l: usize, t0: f64, t1: f64, buf: &mut Vec<Trip>, out: &mut Vec<Trip>) {
        let ls = &self.lines[l];
        if !ls.running {
            return;
        }
        // far enough back that every train drawn in the window is handed over as it would be
        // from any other window
        let (dur, hcap) = self.reach(net, l);
        let (a, b) = (t0 - dur - 2.0 * hcap - 60.0, t1 + dur + hcap + HANDOVER_TOL_S + 1.0);
        buf.clear();
        for r in 0..2 {
            self.run_deps(net, l, r, a, b, buf);
        }
        let eras = self.eras_of(l);
        Self::hand_over(buf, eras, ls.is_loop());
        // a train that had left service before a change stays gone, whatever the new profile
        // says about how long its trip would have taken
        let gone = |t: &Trip| eras.iter().any(|e| t.draw < e.until && t.end > e.until && !e.drawn.contains(&(t.run, t.dep)));
        out.extend(buf.iter().filter(|t| t.draw < t1 && t.end > t0 && !gone(t)));
    }

    /// Where a line's boarding account starts its day, in game time (T-164, T-206): 03:00 of a
    /// quiet day in the line's local time (T-309), or, while it has eras, six hours before its
    /// latest change, so the account's day holds the departures actually drawn around it.
    pub fn account_start(&self, l: usize, start_hour: f64) -> f64 {
        match self.eras_of(l).last() {
            Some(e) => ((e.until - 6.0 * 3600.0) / 60.0).floor() * 60.0,
            None => DAY_S + start_hour * 3600.0 - self.lines.get(l).map_or(0.0, |ls| ls.shift),
        }
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
        if let (Some(pts), Some(plan)) = (net.route_points(line), self.lines[line as usize].bus.as_ref()) {
            let road = if run == 0 { s } else { plan.len - s };
            return super::bus::at_road(pts, &plan.cum, road);
        }
        let (e, _, d) = self.segment_at(net, line, run, s);
        let (x, y, _) = geom::pos_at(net.edge_pieces(e), d);
        (x, y)
    }
}

impl LineSvc {
    pub(crate) fn stop_to_stop_safe(&self, r: usize, lev: usize) -> Vec<f64> {
        if !self.ok {
            return vec![];
        }
        self.stop_to_stop(r, lev)
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    /// T-4285: an entry past a station's split that moved by less than `SPAN_EPS_M` while the run's length came out the
    /// same bit for bit is counted in the span's drift, so the clock worker's kept samples are checked within it
    #[test]
    fn drift_covers_every_kept_entry() {
        let was_segs = [2, 4, 6];
        let was_off = [0.0, 10.0, 20.0];
        let moved = 1e-7;
        let x = Run { segs: vec![2, 8, 10, 6], seg_off: vec![0.0, 10.0, 15.0, 20.0 + moved], len: 30.0, ..Default::default() };
        let s = span_of(&x, &was_segs, &was_off, 30.0, &[2, 4, 5]).expect("a span");
        assert_eq!((s.lo, s.hi), (10.0, 20.0 + moved));
        assert!(s.drift >= (20.0 + moved) - 20.0, "drift {}", s.drift);
    }
}
