//! Edit operations with inverses, undo and redo, and what each edit dirties (SPEC 6.5,
//! notes/T-008.md 7). `TrackWorld` is what the clock worker owns: the network, the service
//! derived from it, and the undo history.

use super::capacity::ResKind;
use super::net::{path_edge, weekend_headway, EdgeData, Extra, Issue, LineData, Network, NodeData, StationData, Weekend};
use super::params::*;
use super::service::{Service, ServiceDirty};
use super::twin::Twin;

/// A line's trains, rates and stop-to-stop times (by run, then level), as demand reads them.
type DemandView = (Option<[u32; 5]>, [f32; SLOTS], [Vec<Vec<f64>>; 2]);
/// The weekdays' trains, headways and midday and afternoon rows, put back after a weekend settle.
type WeekdayRows = (Vec<Option<[u32; ROWS]>>, Vec<Option<[u32; ROWS]>>, Vec<Extra>);

/// One edit. The primitive setters replace an object whole (`None` removes it); `Split` and
/// `DeleteEdge` expand into primitives when applied. Every applied op yields its inverse, which
/// is again an `Op`.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Node { id: u32, data: Option<NodeData> },
    Edge { id: u32, data: Option<EdgeData> },
    Station { node: u32, data: Option<StationData> },
    Line { id: u32, data: Option<LineData> },
    TrainCounts { id: u32, trains: Option<[u32; ROWS]>, tph: [f32; ROWS] },
    /// T-625: set a line's headways in seconds per level (`None`: counts set by hand) with the
    /// counts it starts from; the settle then keeps the counts to the headways.
    AutoTrains { id: u32, auto: Option<[u32; ROWS]>, trains: Option<[u32; ROWS]> },
    /// T-3521: set a line's weekend trains (`LineData::weekend`).
    Weekend { id: u32, weekend: Weekend },
    /// T-3542: the rows (3, 4 or 5) the game's lines set their trains in (`line` None), or line
    /// `line`'s own (0: the game's). `TrackWorld::rows_ops` adds what a switch does to the counts.
    Rows { line: Option<u32>, rows: u8 },
    /// T-3542: set a line's midday and afternoon rows (`LineData::x`).
    Extra { id: u32, x: Extra },
    /// T-3542: the hours (`params::Cuts`) the game's lines split their day at (`line` None), or
    /// line `line`'s own (all 0: the game's).
    Hours { line: Option<u32>, cuts: Cuts },
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
/// T-3482: trains an hour a round may move a line the edit changed by before its share of full
/// track counts as found.
const EDITED_SETTLED_TPH: f32 = 0.01;
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

/// T-3646: what a settle and the departures' placing leave on one line, weekdays and weekends, as
/// an undo or redo puts it back: re-settling from the edited state lands elsewhere, since held
/// shares of full track and the pacing's sets depend on the way there. A table not reaching the
/// line reads as its default, as the code that grows it fills it.
#[derive(Clone, Debug, PartialEq)]
struct LineRates {
    tph: [f32; SLOTS],
    tph_rest: [f32; SLOTS],
    pace: [f32; SLOTS],
    pace_rest: [f32; SLOTS],
    pace_in: [PaceIn; SLOTS],
    pace_in_rest: [PaceIn; SLOTS],
    trains: Option<[u32; 5]>,
    /// phase, split, and the weekends' (`TrackWorld::phases`)
    phases: [[f32; SLOTS]; 4],
    /// its `settle_kept` and `settle_kept_rest` entries
    kept: [Vec<(u32, usize, u32, u64, u32)>; 2],
}

/// T-3646: an edit's effect on the settled state, to undo exactly: per line it moved, its state
/// after the edit (what the undo expects to find) and before it (what the undo puts back), and the
/// weekends' `rest_solved` the same way. An undo that finds anything else (an edit off the history
/// came between, the load's cap lowering) settles instead.
#[derive(Clone, Debug, Default)]
struct Rates {
    lines: Vec<(u32, Option<LineRates>, Option<LineRates>)>,
    rest_solved: (bool, bool),
}

/// One entry of the undo history: the step back (the inverse and what it puts back) and, for an edit
/// off the history (the load's cap lowering, T-3528), the edit itself, which the redo of the edit
/// before it makes again (`TrackWorld::step`). Such an entry is never undone by itself.
#[derive(Clone, Debug)]
struct Back {
    step: (Op, Rates),
    again: Option<Op>,
}

/// The lines an op sets, where it sets nothing but lines' trains, headways and rows; None for any
/// other op (track, stations, the game's rows or hours, a line's path).
fn op_lines(op: &Op) -> Option<Vec<u32>> {
    match op {
        Op::TrainCounts { id, .. } | Op::AutoTrains { id, .. } | Op::Weekend { id, .. } | Op::Extra { id, .. } => Some(vec![*id]),
        Op::Batch(v) => v.iter().map(op_lines).collect::<Option<Vec<_>>>().map(|v| v.concat()),
        _ => None,
    }
}

/// `next` folded into `all`: what either touched, costs from the first before and the last after.
fn merge_results(all: &mut Option<EditResult>, next: EditResult) {
    let Some(a) = all.as_mut() else {
        *all = Some(next);
        return;
    };
    fn join(a: &mut Vec<u32>, b: Vec<u32>) {
        a.extend(b);
        a.sort_unstable();
        a.dedup();
    }
    let (d, n) = (&mut a.dirty, next.dirty);
    join(&mut d.edges, n.edges);
    join(&mut d.nodes, n.nodes);
    d.crossings.extend(n.crossings);
    join(&mut d.resources, n.resources);
    join(&mut d.lines, n.lines);
    join(&mut d.service.rebuilt, n.service.rebuilt);
    join(&mut d.service.recomputed, n.service.recomputed);
    join(&mut d.service.shared, n.service.shared);
    join(&mut d.service.demand, n.service.demand);
    a.cost_after = next.cost_after;
    a.built_after = next.built_after;
    a.charge += next.charge;
}

/// A line's departure split where the table does not reach it (as a load grows it).
const SPLIT_NONE: [f32; SLOTS] = [0.5; SLOTS];

/// `v[l] = x`, growing `v` with `none` only where `x` is not `none` (`LineRates` reads a table
/// that does not reach a line as `none`).
fn put_at<T: Copy + PartialEq>(v: &mut Vec<T>, l: usize, x: T, none: T) {
    if v.len() <= l {
        if x == none { return; }
        v.resize(l + 1, none);
    }
    v[l] = x;
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
    undo: Vec<Back>,
    /// each with the undo stack's length right after its undo (the entries above that are edits off
    /// the history made since, which a redo takes back first) and the edits off the history its undo
    /// took back, which its redo makes again (`step`)
    redo: Vec<((Op, Rates), usize, Vec<Op>)>,
    /// T-3646: the lines' states an undo or redo puts back in this commit (`Rates`), when it found
    /// the state the step left.
    restore_to: Option<(Vec<(u32, LineRates)>, bool)>,
    /// T-752: a line switching between counts and headways, whose counts this edit's settle keeps,
    /// and the levels it had no trains at (T-849: they take the headway's count, which the track
    /// trims as any headway line's)
    switching: Option<(u32, [bool; 5])>,
    /// T-1863: the lines an edit's settle may step: those sharing track, directly or through others,
    /// with what the edit changed. None settles every line off its fleet (a load, a bulk edit).
    settle_scope: Option<Vec<bool>>,
    /// T-3482: the lines the edit itself changed (their path, stops, trains or headways), sorted, with
    /// the levels they ran no trains at before it (every level of a line the edit made or brought
    /// back): no fill has given them a share there, so being short on full track does not settle them
    /// in the edit's first settle (`settle_unpaced`, up to its trim). Empty outside an edit.
    settle_edited: Vec<(u32, [bool; SLOTS])>,
    /// T-3482: `settle_edited` for the weekends' settle (`settle_weekend`), by the weekend rates.
    settle_edited_rest: Vec<(u32, [bool; SLOTS])>,
    /// T-3521: the weekend service (`settle_weekend`) where weekends were settled on their own; not
    /// saved. None: worked out again from the weekdays' at the next weekend settle.
    svc_rest: Option<Box<Service>>,
    /// T-3521: a load keeps the save's weekend rates as they are (`reload`), to the bit, and the
    /// weekend service is worked out at the first edit that settles them.
    weekend_as_saved: bool,
    /// T-3388: per line and level, what `pace_merges` last paced it from. Not saved: a load takes
    /// the saved pacing as it is.
    pace_in: Vec<[PaceIn; SLOTS]>,
    /// T-3584: what the weekends' pacing (`Network::line_pace_rest`) was paced from, kept with
    /// `svc_rest` (the weekdays' start them at the first weekend settle). Not saved.
    pace_in_rest: Vec<[PaceIn; SLOTS]>,
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
        let mut w = TrackWorld { net, svc: Service::default(), undo: vec![], redo: vec![], restore_to: None, switching: None, settle_scope: None, settle_edited: vec![], settle_edited_rest: vec![], svc_rest: None, weekend_as_saved: false, pace_in: vec![], pace_in_rest: vec![] };
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
        // T-3735: the weekends' too, where a save carried none (`LWP1`)
        if !self.net.rest_solved || self.net.line_phase_rest.is_empty() {
            let all: Vec<u32> = (0..self.net.line_count() as u32).collect();
            self.place_weekend(&all);
        }
        self.retime();
    }

    /// T-3346 stage 2: each train's own timetable worked out again (`timing::retime`), once the
    /// service has settled and the departures are placed.
    pub fn retime(&mut self) {
        super::timing::retime(&self.net, &mut self.svc);
    }

    /// T-3542: after an edit split or joined the slots (`Network::refresh_slots`, from `old`): the
    /// slots it split off start from what their periods ran (`Network::seed_slots`), those it joined
    /// go back to as made, and everything is worked out again as a load with saved rates does, the
    /// rates and pacing taken as they stand and the departures kept. The weekends keep their rates
    /// until the next edit settles them, as after a load (London's weekends at 70% took 9 s natively).
    fn reslot(&mut self, old: [usize; 5]) {
        self.net.seed_slots(old);
        let settled = self.net.rates_settled;
        (self.net.rates_from_save, self.net.pace_from_save, self.net.phases_placed) = (settled, settled, true);
        // trains already running keep their patterns (each era carries its own slots)
        let (eras, shifts) = (std::mem::take(&mut self.svc.eras), std::mem::take(&mut self.svc.shifts));
        self.reload();
        (self.svc.eras, self.svc.shifts) = (eras, shifts);
    }

    /// T-3542: the edit that sets the game's rows (`line` None) or line `line`'s own (0: the game's)
    /// to `rows`, with what the switch does to each line it moves: a row split in two gives both its
    /// trains (and headway); rows joined into one take the most trains of them (the shortest headway),
    /// so no period quietly loses service. Weekend counts set by hand alike. None if nothing changes.
    pub fn rows_ops(&self, line: Option<u32>, rows: u8) -> Option<Op> {
        let net = &self.net;
        let now = match line { None => net.rows_default, Some(l) => *net.line_rows.get(l as usize)? };
        if now == rows {
            return None;
        }
        let mut ops = vec![Op::Rows { line, rows }];
        // a line back to the game's rows takes the game's hours too; one taking its own rows keeps
        // the game's hours as its own, so the game's next hours leave it as it is
        let own_hours = |l: u32| net.line_x.get(l as usize).is_some_and(|x| x.hours != [0; 5]);
        let hours_after = match (line, rows) {
            (Some(l), 0) if own_hours(l) => Some([0; 5]),
            (Some(l), _) if rows != 0 && !own_hours(l) && net.line_rows.get(l as usize) == Some(&0) => Some(net.hours_default),
            _ => None,
        };
        if let Some(cuts) = hours_after { ops.push(Op::Hours { line, cuts }); }
        // a line's rows after the edit
        let after = |l: usize| -> usize {
            let own = if line.is_some() { rows } else { net.line_rows[l] };
            let game = if line.is_none() { rows } else { net.rows_default };
            (if own == 0 { game } else { own } as usize).clamp(ROWS_MIN, ROWS_MAX)
        };
        let lines: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l) && !net.is_twin(l) && match line {
            None => net.line_rows[l as usize] == 0,
            Some(id) => l == id,
        }).collect();
        for l in lines {
            let (i, from) = (l as usize, net.rows(l as usize));
            let to = after(i);
            if from == to { continue; }
            // each new row's value from the old places of its periods: the most, or the shortest
            // headway that is some
            let remap = |p: [u32; 5], most: bool| -> [u32; 5] {
                let mut out = p;
                for &k in ROW_PLACES[to - ROWS_MIN] {
                    let olds = (0..5).filter(|&q| PERIOD_PLACE[to - ROWS_MIN][q] == k).map(|q| p[PERIOD_PLACE[from - ROWS_MIN][q]]);
                    out[k] = if most { olds.max().unwrap_or(0) } else { olds.filter(|&h| h > 0).min().unwrap_or(0) };
                }
                out
            };
            let Some(d) = net.line_data(l) else { continue };
            let mut x = d.x;
            if let Some(c) = hours_after { x.hours = c; }
            if let Some(t) = net.trains_of(i) {
                let (m, tx) = unplace(remap(t, true));
                x.trains = tx;
                match net.auto_of(i) {
                    Some(a) => {
                        let (am, ax) = unplace(remap(a, false));
                        x.auto = ax;
                        ops.push(Op::AutoTrains { id: l, auto: Some(am), trains: Some(m) });
                    }
                    None => ops.push(Op::TrainCounts { id: l, trains: Some(m), tph: d.tph }),
                }
            }
            if let Some(c) = d.weekend.counts {
                let (m, cx) = unplace(remap(places(c, x.weekend_counts), true));
                let (h, hx) = unplace(remap(places(d.weekend.headways, x.weekend_headways), false));
                (x.weekend_counts, x.weekend_headways) = (cx, hx);
                ops.push(Op::Weekend { id: l, weekend: Weekend { counts: Some(m), headways: h, ..d.weekend } });
            }
            if x != d.x {
                ops.push(Op::Extra { id: l, x });
            }
        }
        Some(Op::Batch(ops))
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

    /// T-3735: `place_departures` on weekends, into `line_phase_rest` and `line_split_rest`: those of
    /// `lines` that run other rates then (`Network::runs_own_weekend`) are placed among every line's
    /// weekend trains, on the weekends' rates, headways and pacing swapped in as `settle_weekend`
    /// swaps them. The service placed on is the weekdays': weekend trains run its timetables. Every
    /// other line keeps a copy of its weekday phases and splits, as it runs its weekday departures.
    fn place_weekend(&mut self, lines: &[u32]) {
        if !self.net.rest_solved {
            self.net.line_phase_rest.clear();
            self.net.line_split_rest.clear();
            return;
        }
        let n = self.net.line_count();
        let own: Vec<bool> = (0..n).map(|l| self.net.runs_own_weekend(l)).collect();
        let before = (std::mem::take(&mut self.net.line_phase_rest), std::mem::take(&mut self.net.line_split_rest));
        let copy = |net: &Network, own: &[bool], phase: &mut Vec<[f32; SLOTS]>, split: &mut Vec<[f32; SLOTS]>, from: &(Vec<[f32; SLOTS]>, Vec<[f32; SLOTS]>)| {
            phase.resize(n, [0.0; SLOTS]);
            split.resize(n, [0.5; SLOTS]);
            for l in 0..n {
                let (p, s) = if own[l] { (from.0.get(l), from.1.get(l)) } else { (None, None) };
                phase[l] = p.copied().unwrap_or_else(|| net.line_phase_of(l as u32));
                split[l] = s.copied().unwrap_or_else(|| net.line_split_of(l as u32));
            }
        };
        let (mut phase, mut split) = (vec![], vec![]);
        copy(&self.net, &own, &mut phase, &mut split, &before);
        let todo: Vec<u32> = lines.iter().copied().filter(|&l| own.get(l as usize) == Some(&true)).collect();
        let twins = self.net.line_twin.iter().enumerate().any(|(t, x)| matches!(x, Twin::Of(_)) && own.get(t) == Some(&true));
        if !todo.is_empty() || twins {
            let counts = self.weekend_own();
            let week = self.swap_in_weekend(&counts);
            std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
            let paced = !self.net.line_pace_rest.is_empty();
            if paced { std::mem::swap(&mut self.net.line_pace, &mut self.net.line_pace_rest); }
            std::mem::swap(&mut self.net.line_phase, &mut phase);
            std::mem::swap(&mut self.net.line_split, &mut split);
            // (the lattices read while placing are the weekends', in the weekdays' fields)
            self.net.rest_solved = false;
            self.place_departures(&todo);
            self.pin_twin_phases();
            self.net.rest_solved = true;
            std::mem::swap(&mut self.net.line_phase, &mut phase);
            std::mem::swap(&mut self.net.line_split, &mut split);
            if paced { std::mem::swap(&mut self.net.line_pace, &mut self.net.line_pace_rest); }
            std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
            self.put_back_weekdays(week);
            // a line placed alongside (its spare time moved) that runs its weekday pattern keeps it
            let placed = (phase, split);
            let (mut p2, mut s2) = (vec![], vec![]);
            copy(&self.net, &own, &mut p2, &mut s2, &placed);
            (phase, split) = (p2, s2);
        }
        let moved: Vec<u32> = (0..n as u32).filter(|&l| {
            let i = l as usize;
            before.0.get(i) != phase.get(i) || before.1.get(i) != split.get(i)
        }).filter(|&l| own[l as usize]).collect();
        self.net.line_phase_rest = phase;
        self.net.line_split_rest = split;
        // their weekend departures are drawn again
        self.svc.note_changed(&moved);
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
        self.retime();
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
            let t = self.svc.lines.get(l).filter(|s| s.ok).map_or([0; 5], |s| self.net.places_most(l, s.trains));
            self.net.set_trains_of(l, Some(t));
            any = true;
        }
        if any { self.settle_train_counts(true); self.retime(); }
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
        // T-3378: and BRT lines sharing a lane, whose buses share its room as trains share track; after an edit of any
        // bus line every BRT line, as one taken off BRT or redrawn shares no lane any more with the lines it held back
        let room = self.net.brt_room();
        if seeds.iter().any(|&l| (l as usize) < n && self.net.is_bus(l)) {
            for l in self.net.brt_order(true) {
                hit[l as usize] = true;
            }
        }
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
            for w in &room.ways {
                if !w.lines.iter().any(|&l| (l as usize) < n && hit[l as usize]) {
                    continue;
                }
                for &l in &w.lines {
                    if (l as usize) < n && !hit[l as usize] {
                        hit[l as usize] = true;
                        grew = true;
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
                for lev in self.net.slots().filter(|&lev| cap.rho[r][lev] >= PACE_FULL_RHO) {
                    full.push((top(&mut root, u.line as usize), lev));
                }
            }
        }
        let was: Vec<[f32; SLOTS]> = (0..n).map(|l| self.net.line_pace_of(l as u32)).collect();
        let mut pace = vec![[0.0f32; SLOTS]; n];
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for l in (0..n).filter(|&l| met[l]) {
            let g = top(&mut root, l);
            groups.entry(g).or_default().push(l);
        }
        self.pace_in.resize(n, [PaceIn::default(); SLOTS]);
        for lines in groups.values() {
            let g = top(&mut root, lines[0]);
            // lines are pushed in id order, so the first is the group's least
            let group = (lines[0] as u32, lines.len() as u32);
            for lev in self.net.slots() {
                let is_full = full.contains(&(g, lev));
                // (headway, line, set by the player, the set it was in when last paced: NO_SET
                // when it has moved since); a count line's headway within `PACE_HOLD` of the one it
                // was last paced from, and no longer than its pace, counts as that one
                let mut hs: Vec<(f64, usize, bool, u32)> = vec![];
                for &l in lines {
                    let ls = &self.svc.lines[l];
                    // T-3542: a slot its hours share between rows runs no one headway to pace
                    if !self.net.line_ok(l as u32) || !ls.ok || !ls.running || ls.is_loop() || ls.is_bus() || self.net.mixed(l, lev).is_some() {
                        continue;
                    }
                    let last = self.pace_in[l][lev];
                    let c = self.net.trains_at(l, lev);
                    if let Some(a) = self.net.auto_at(l, lev) {
                        if a > 0 {
                            let h = a as f64;
                            hs.push((h, l, true, if last.h == h { last.set } else { NO_SET }));
                        }
                    } else if self.net.line_trains[l].is_some() {
                        if c > 0 && ls.round_trip[lev] > 0.0 {
                            let h = ls.round_trip[lev] / c as f64;
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

    /// T-3378 stage 2: cut the bus lines on a BRT lane past its room to fit (`Capacity::fill`, which counts the lanes),
    /// at each slot where one is: a settle steps only lines off their fleet, and lines at their headways can overfill a
    /// lane without being off it. Then settled again, the cut lines are short of their fleet and held to it as track
    /// holds trains. The lines it moved.
    fn fit_lanes(&mut self) -> Vec<u32> {
        let room = self.net.brt_room();
        let mut ids: Vec<u32> = room.ways.iter().flat_map(|w| w.lines.iter().copied()).filter(|&l| self.net.line_ok(l) && self.svc.lines.get(l as usize).is_some_and(|s| s.ok)).collect();
        ids.sort_unstable();
        ids.dedup();
        let mut moved = vec![];
        for lev in self.net.slots() {
            let over = room.ways.iter().any(|w| w.lines.iter().map(|&l| self.net.line_tph[l as usize][lev] as f64).sum::<f64>() > super::bus::lanes::LANE_BUSES_H + 1e-6);
            if !over {
                continue;
            }
            let want: Vec<(u32, f64)> = ids.iter().map(|&l| (l, self.net.line_tph[l as usize][lev] as f64)).collect();
            for (&l, got) in ids.iter().zip(self.svc.cap.fill(&self.net, &want, lev)) {
                if (got as f32) != self.net.line_tph[l as usize][lev] {
                    self.net.line_tph[l as usize][lev] = got as f32;
                    moved.push(l);
                }
            }
        }
        if !moved.is_empty() {
            moved.sort_unstable();
            moved.dedup();
            self.svc.update_scoped(&self.net, &[], &moved, true, Some((&[], &[])));
        }
        moved
    }

    /// The settle without `pace_merges`.
    fn settle_unpaced(&mut self, full: bool) -> Vec<u32> {
        if !self.net.line_auto.iter().any(|a| a.is_some()) {
            let mut affected = self.settle_counts(full);
            // T-3378: and to the BRT lanes' room
            let cut = self.fit_lanes();
            if !cut.is_empty() {
                affected.extend(cut);
                affected.extend(self.settle_counts(full));
                affected.sort_unstable(); affected.dedup();
            }
            self.settle_edited.clear();
            return affected;
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
        // T-3378: and to the BRT lanes' room, before the buses it leaves waiting are trimmed
        let cut = self.fit_lanes();
        if !cut.is_empty() {
            affected.extend(cut);
            affected.extend(self.settle_counts(full));
        }
        // T-3482: the edited lines have their share of full track, with the counts that took it; the
        // trains it leaves waiting are trimmed from them as from any other line
        self.settle_edited.clear();
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
            let (Some(auto), Some(old)) = (self.net.auto_of(l), self.net.trains_of(l)) else { continue };
            if !self.net.line_ok(l as u32) || !self.svc.lines[l].ok { continue; }
            // T-752: a switch to headways keeps the trains the line has; T-849: a level with none
            // takes its headway's count, or a new line switched straight to headways never runs,
            // and follows its round trip as any headway line's (T-3482)
            let keep = match self.switching { Some((s, empty)) if s == l as u32 => empty.map(|e| !e), _ => [false; 5] };
            let trip = self.svc.lines[l].round_trip;
            // a loop's two directions each keep the headway with half its trains (T-798)
            let each = if self.svc.lines[l].paired() { 2 } else { 1 };
            let mut counts = old;
            for (k, slots) in self.net.row_slots(l).into_iter().filter(|(k, _)| !keep[*k]) {
                // T-3542: a row over several slots takes the most any of them needs
                let want = slots.iter().map(|&lev| {
                    let t = trip[lev] / each as f64;
                    each * clamp_auto(auto_trains(t, auto[k]), t, auto[k], most[l][lev].0)
                }).max().unwrap_or(0);
                if want < old[k] || !last { counts[k] = want; }
            }
            if counts != old {
                self.net.set_trains_of(l, Some(counts));
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
            let (Some(_), Some(mut counts)) = (self.net.line_auto[l], self.net.trains_of(l)) else { continue };
            if !self.net.line_ok(l as u32) || !self.svc.lines[l].ok { continue; }
            let empty = match self.switching { Some((s, e)) if s == l as u32 => e, Some(_) | None => [true; 5] };
            let trip = self.svc.lines[l].round_trip;
            for (k, slots) in self.net.row_slots(l).into_iter().filter(|(k, _)| empty[*k]) {
                // T-3542: a row over several slots keeps what its busiest needs: the fewest any of
                // them leaves waiting (none where its track is over full)
                let held = slots.iter().map(|&lev| {
                    if worst[l][lev].0 > 2.0 - FULL_RHO { return 0; }
                    let short = self.net.running_trains(l, lev, trip[lev]) - self.net.line_tph[l][lev] as f64 * trip[lev] / 3600.0;
                    let held = if short > 0.01 { short.floor() as u32 } else { 0 };
                    if self.svc.lines[l].paired() { held & !1 } else { held }
                }).min().unwrap_or(0);
                if held > 0 { counts[k] -= held.min(counts[k]); trimmed = true; }
            }
            self.net.set_trains_of(l, Some(counts));
        }
        trimmed
    }

    /// T-3521: a line's whole trains on weekends (`LineData::weekend`) where they differ from its
    /// weekday counts; None where it runs those. A share scales the counts up to whole trains, or a
    /// headway line's headways (`weekend_headway`) over its weekday round trip.
    pub fn weekend_counts(&self, l: usize) -> Option<[u32; ROWS]> {
        self.weekend_places(l).map(|c| unplace(c).0)
    }

    /// T-3542: `weekend_counts` at the line's five places (`params::places`).
    pub fn weekend_places(&self, l: usize) -> Option<[u32; 5]> {
        let mut most = None;
        self.weekend_counts_in(l, &mut most)
    }

    /// `weekend_places` with the track's limits per line (`line_limits`) worked out once into `most`
    /// for every line after it.
    fn weekend_counts_in(&self, l: usize, most: &mut Option<Vec<[(f64, u32); SLOTS]>>) -> Option<[u32; 5]> {
        let t = self.net.trains_of(l)?;
        // a headway line runs its weekend headway even on its weekday count of trains, which then
        // stand longer at the ends
        let own_headway = match (self.weekend_auto(l), self.net.auto_of(l)) {
            (Some(a), Some(b)) => !self.net.same_places(l, a, b),
            (a, b) => a.is_some() != b.is_some(),
        };
        let x = self.net.line_x[l];
        let mut c = t;
        match self.net.line_weekend.get(l).copied().unwrap_or_default() {
            Weekend { counts: Some(m), .. } => c = places(m, x.weekend_counts),
            Weekend { share: 100, .. } => return None,
            Weekend { share: p, .. } => match (self.net.auto_of(l), self.svc.lines.get(l)) {
                (Some(auto), Some(ls)) => {
                    // a loop's two directions each keep the headway with half its trains (T-798),
                    // and no more than the track lets run, as the weekdays' (`auto_counts`)
                    let each = if ls.paired() { 2 } else { 1 };
                    let most = most.get_or_insert_with(|| self.svc.cap.line_limits(&self.net));
                    // under 100% never more than the weekdays run (a full track trims those, T-625);
                    // a row over several slots the most any of them needs (T-3542)
                    for (k, slots) in self.net.row_slots(l) {
                        let h = weekend_headway(auto[k], p);
                        let n = slots.iter().map(|&lev| {
                            let rt = ls.round_trip[lev] / each as f64;
                            each * clamp_auto(auto_trains(rt, h), rt, h, most.get(l).map_or(f64::INFINITY, |m| m[lev].0))
                        }).max().unwrap_or(0);
                        c[k] = if p <= 100 { n.min(t[k]) } else { n };
                    }
                }
                _ => for (k, _) in self.net.row_slots(l) { c[k] = (t[k] as u64 * p as u64).div_ceil(100).min(u32::MAX as u64) as u32 },
            },
        };
        (!self.net.same_places(l, c, t) || own_headway).then_some(c)
    }

    /// `Network::weekend_auto`.
    pub(crate) fn weekend_auto(&self, l: usize) -> Option<[u32; 5]> {
        self.net.weekend_auto(l)
    }

    /// T-3521: the weekends' headways in `line_auto` for the lines in `own` (`weekend_own`), so
    /// whatever reads a line's headway (`running_trains`, `Service::spare`, `pattern_tph`) reads the
    /// weekends'; and their trains in `line_trains`. Returns the weekdays' headways, trains and
    /// midday and afternoon rows to put back (`put_back_weekdays`).
    fn swap_in_weekend(&mut self, own: &[(usize, [u32; 5])]) -> WeekdayRows {
        let week = (self.net.line_trains.clone(), self.net.line_auto.clone(), self.net.line_x.clone());
        for &(l, c) in own {
            let auto = week.1[l].is_some().then(|| self.weekend_auto(l)).flatten();
            self.net.set_trains_of(l, Some(c));
            if week.1[l].is_some() { self.net.set_auto_of(l, auto); }
        }
        week
    }

    /// `swap_in_weekend` undone.
    fn put_back_weekdays(&mut self, week: WeekdayRows) {
        (self.net.line_trains, self.net.line_auto, self.net.line_x) = week;
    }

    /// T-3521: each line's weekend counts where they differ from its weekday ones (`weekend_counts`).
    pub fn weekend_own(&self) -> Vec<(usize, [u32; 5])> {
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
        let week = self.swap_in_weekend(&own);
        std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
        if let Some(rest) = self.svc_rest.as_mut() {
            rest.update_scoped(&self.net, lines, schedules, cap_dirty, Some((edges, nodes)));
        }
        std::mem::swap(&mut self.net.line_tph, &mut self.net.line_tph_rest);
        self.put_back_weekdays(week);
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
            let week = self.swap_in_weekend(&own);
            // the service (round trips, holds, how full each track is) on the weekends' rates before
            // the settle reads it: on the weekdays' a track full on weekdays held every line on it
            if fresh {
                let moved: Vec<u32> = (0..n as u32).filter(|&l| self.net.line_tph[l as usize] != week_tph[l as usize] && self.svc.lines.get(l as usize).is_some_and(|s| s.ok)).collect();
                if !moved.is_empty() { self.svc.update_scoped(&self.net, &[], &moved, true, Some((&[], &[]))); }
            }
            // T-3584: lines sharing track are paced on the weekends' headways too, from the weekdays'
            // pacing at first
            let (week_pace, week_pace_in) = (std::mem::take(&mut self.net.line_pace), std::mem::take(&mut self.pace_in));
            if fresh {
                self.net.line_pace = week_pace.clone();
                self.pace_in = week_pace_in.clone();
            } else {
                self.net.line_pace = std::mem::take(&mut self.net.line_pace_rest);
                self.pace_in = std::mem::take(&mut self.pace_in_rest);
            }
            self.settle_edited = std::mem::take(&mut self.settle_edited_rest);
            self.settle_counts(full);
            self.settle_edited.clear();
            for _ in 0..PACE_ROUNDS {
                if !self.pace_merges(false) {
                    break;
                }
                self.settle_counts(full);
            }
            self.pace_in.iter_mut().flatten().for_each(|p| p.fresh = false);
            self.net.line_pace_rest = std::mem::replace(&mut self.net.line_pace, week_pace);
            self.pace_in_rest = std::mem::replace(&mut self.pace_in, week_pace_in);
            self.net.line_tph_rest.clone_from(&self.net.line_tph);
            // a loop's counts set by hand were made even (as its weekday counts are), which it keeps
            let even: Vec<(usize, [u32; 5])> = own.iter().filter(|&&(l, _)| self.net.line_weekend[l].counts.is_some()).filter_map(|&(l, _)| self.net.trains_of(l).map(|t| (l, t))).collect();
            self.net.settle_kept_rest = std::mem::replace(&mut self.net.settle_kept, week_kept);
            self.put_back_weekdays(week);
            for (l, t) in even {
                let (m, x) = unplace(t);
                self.net.line_weekend[l].counts = Some(m);
                self.net.line_x[l].weekend_counts = x;
            }
            self.net.line_tph = week_tph;
            self.net.rates_settled = settled;
            self.svc_rest = Some(Box::new(std::mem::replace(&mut self.svc, week_svc)));
            self.net.rest_solved = true;
        }
        let on = |l: usize, t: &[[f32; SLOTS]], s: bool| if s { t.get(l).copied() } else { self.net.line_tph.get(l).copied() };
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
                    self.net.line_x[id as usize].trains.iter_mut().for_each(|x| *x += *x % 2);
                }
            }
        }
        let before = self.net.line_tph.clone();
        let mut affected = vec![];
        // T-1283: per line and level, the sign its shortfall last had, how often it has turned
        // without shrinking by half, and its size at the last turn
        let mut swings: Vec<[(i8, u8, f64); SLOTS]> = vec![[(0, 0, 0.0); SLOTS]; ids.len()];
        let mut kept_now = vec![[false; SLOTS]; ids.len()];
        // T-3233: a round is a function of the rates, round trips and swings it starts from, so a
        // state seen `p` rounds before comes round every `p` rounds to the cap; the settle then runs
        // only the rounds that end it in the cap's state, to the bit
        let mut seen: std::collections::VecDeque<(Vec<[f32; SLOTS]>, Vec<[f64; SLOTS]>, Vec<[(i8, u8, f64); SLOTS]>)> = Default::default();
        let mut stop_at: Option<usize> = None;
        let mut last_rates: Option<Vec<[f32; SLOTS]>> = None;
        for round in 0..SETTLE_ROUNDS {
            let trips: Vec<[f64; SLOTS]> = ids.iter().map(|&id| self.svc.lines[id as usize].round_trip).collect();
            let rates: Vec<[f32; SLOTS]> = ids.iter().map(|&id| self.net.line_tph[id as usize]).collect();
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
            let mut active = [false; SLOTS];
            let mut off = vec![[false; SLOTS]; ids.len()];
            let keep = &mut kept_now;
            keep.iter_mut().for_each(|k| *k = [false; SLOTS]);
            let worst = self.svc.cap.bottlenecks(self.net.line_count());
            let scope = self.settle_scope.as_deref();
            for (j, &id) in ids.iter().enumerate() {
                for lev in self.net.slots() {
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
                    // T-3482: not a line the edit changed, whose rate no fill has given it yet (a new
                    // line on track others fill would keep the rate it came with), until its share
                    // stops moving
                    let edited = self.settle_edited.binary_search_by_key(&id, |e| e.0).is_ok_and(|k| self.settle_edited[k].1[lev]) && last_rates.as_ref().map_or(true, |p| (rates[j][lev] - p[j][lev]).abs() > EDITED_SETTLED_TPH);
                    let held = short > 0.0 && wr >= FULL_RHO && !own_over && !edited;
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
                    let left = (id, lev, rates[j][lev].to_bits(), trips[j][lev].to_bits(), self.net.trains_key(id as usize, lev));
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
                for lev in (0..SLOTS).filter(|&lev| active[lev] && (full || off[j][lev])) { self.net.line_tph[id as usize][lev] *= 1.01; }
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
                for lev in (0..SLOTS).filter(|&lev| active[lev]) {
                    let count = self.net.trains_in(i, lev);
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
                    self.net.line_tph[i][lev] = (if count == 0.0 || trip <= 0.0 { 0.0 } else if runs < count {
                        // T-2607: the headway set, whatever the round trip
                        runs * 3600.0 / trip
                    } else {
                        (old - residual / (trip + rise)).max(0.0)
                    }) as f32;
                }
            }
            // T-517: no track carries more trains than its capacity; a fleet's surplus waits at the ends
            for lev in (0..SLOTS).filter(|&lev| active[lev]) {
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
            last_rates = Some(rates);
        }
        for &id in &ids { self.svc.lines[id as usize].trains = self.net.slot_trains(id as usize); }
        self.net.settle_kept.clear();
        for (j, &id) in ids.iter().enumerate() {
            for lev in (0..SLOTS).filter(|&lev| kept_now[j][lev]) {
                self.net.settle_kept.push((id, lev, self.net.line_tph[id as usize][lev].to_bits(), self.svc.lines[id as usize].round_trip[lev].to_bits(), self.net.trains_key(id as usize, lev)));
            }
        }
        affected.extend(ids.into_iter().filter(|&id| before[id as usize] != self.net.line_tph[id as usize]));
        affected.sort_unstable(); affected.dedup(); affected
    }

    pub fn can_undo(&self) -> bool {
        self.undo.iter().any(|b| b.again.is_none())
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    /// T-1073: whether the edit an undo (or redo) would apply changes only train counts or headways.
    pub fn step_is_service(&self, redo: bool) -> bool {
        fn only_counts(op: &Op) -> bool {
            match op {
                Op::TrainCounts { .. } | Op::AutoTrains { .. } | Op::Weekend { .. } | Op::Rows { .. } | Op::Extra { .. } | Op::Hours { .. } => true,
                Op::Batch(ops) => !ops.is_empty() && ops.iter().all(only_counts),
                _ => false,
            }
        }
        if redo {
            self.redo.last().is_some_and(|s| only_counts(&s.0 .0))
        } else {
            self.undo.iter().rev().find(|b| b.again.is_none()).is_some_and(|b| only_counts(&b.step.0))
        }
    }

    /// Apply, validate, recompute. On problems nothing changes and the problems come back.
    pub fn apply(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        let (r, inv) = self.commit(op)?;
        self.undo.push(Back { step: inv, again: None });
        self.redo.clear();
        Ok(r)
    }

    /// The departures as placed (phases and splits, weekdays and weekends), to put back where an
    /// edit is rolled back (T-1925).
    fn phases(&self) -> [Vec<[f32; SLOTS]>; 4] {
        let n = &self.net;
        [n.line_phase.clone(), n.line_split.clone(), n.line_phase_rest.clone(), n.line_split_rest.clone()]
    }
    fn set_phases(&mut self, p: [Vec<[f32; SLOTS]>; 4]) {
        let n = &mut self.net;
        [n.line_phase, n.line_split, n.line_phase_rest, n.line_split_rest] = p;
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
        let phases = self.phases();
        let (mut r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        if let Some(price) = price {
            r.charge = price(self);
        }
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.restore(inv);
            // T-1925: and the departures as they were
            self.set_phases(phases);
            return Err(ChargeError::Money { need, have: cash });
        }
        if final_ {
            self.undo.clear();
        } else {
            self.undo.push(Back { step: inv, again: None });
        }
        self.redo.clear();
        Ok((r, x))
    }

    /// T-3528: an edit the game makes for the player (the cap lowering on load): applied and paid as
    /// `apply_paid`, but on neither history, so an undo takes back the player's own last edit.
    pub fn apply_unrecorded(&mut self, op: Op, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64) -> Result<(EditResult, f64), ChargeError> {
        self.apply_off_history(op, cash, extra, false)
    }

    /// `apply_unrecorded`, and T-3646, where `with_edit`: belonging to the player's edit before it, so
    /// that edit's undo takes it back and its redo makes it again (the load's cap lowering, which can
    /// come after the first edits). Without, it stays whatever the history does (a removed fare area).
    pub fn apply_off_history(&mut self, op: Op, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64, with_edit: bool) -> Result<(EditResult, f64), ChargeError> {
        let phases = self.phases();
        let again = op.clone();
        let (r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.restore(inv);
            self.set_phases(phases);
            return Err(ChargeError::Money { need, have: cash });
        }
        if with_edit && (!self.undo.is_empty() || !self.redo.is_empty()) {
            self.undo.push(Back { step: inv, again: Some(again) });
        }
        Ok((r, x))
    }

    /// Undo (or redo) and pay `extra` as `apply_paid` does: undoing a line's removal can need
    /// trains again. Nothing changes if it is refused.
    pub fn step_paid(&mut self, redo: bool, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64) -> Option<Result<(EditResult, f64), ChargeError>> {
        let phases = self.phases();
        Some(match self.step(redo)? {
            Ok(mut r) => {
                // what an undo or redo restores was paid for or removed before: only the trains
                // it needs are charged, never the track (a removed built stretch comes back free)
                let need = extra(self);
                if need > 1e-9 && need > cash + 1e-6 {
                    // the step the other way takes it back as exactly
                    let _ = self.step(!redo);
                    self.set_phases(phases);
                    Err(ChargeError::Money { need, have: cash })
                } else {
                    r.charge = 0.0;
                    Ok((r, need))
                }
            }
            Err(e) => Err(ChargeError::Issues(e)),
        })
    }

    /// One undo (or redo). T-3646: the edits off the history made since the step's edit (above it on
    /// the undo stack) are taken back exactly first, so the step finds the state it left and puts back
    /// exactly what it moved. An undo then makes them again (T-3528: an undo takes back the player's
    /// edit, not the game's), but for those setting a line the undone edit set, or all where that
    /// edit changed more than lines, as they were made for the edit's state; its redo takes back the
    /// ones made since and makes them all again after the edit, on the state they were made on.
    /// The result covers all of it. On problems nothing changes.
    fn step(&mut self, redo: bool) -> Option<Result<EditResult, Vec<Issue>>> {
        let base = if redo { self.redo.last()?.1 } else { self.undo.iter().rposition(|b| b.again.is_none())? + 1 };
        let mut all: Option<EditResult> = None;
        let mut again = vec![];
        while self.undo.len() > base {
            let Some(b) = self.undo.pop() else { break };
            let Some(op) = b.again.clone() else { self.undo.push(b); break };
            match self.restore(b.step.clone()) {
                Ok((r, _)) => {
                    merge_results(&mut all, r);
                    again.push(op);
                }
                Err(e) => {
                    self.undo.push(b);
                    self.make_again(&mut all, again);
                    return Some(Err(e));
                }
            }
        }
        again.reverse();
        let (step, after) = if redo { self.redo.pop().map(|s| (s.0, s.2))? } else { (self.undo.pop()?.step, vec![]) };
        match self.restore(step.clone()) {
            Ok((r, inv)) => {
                merge_results(&mut all, r);
                if redo {
                    self.undo.push(Back { step: inv, again: None });
                    self.make_again(&mut all, after);
                } else {
                    let set = op_lines(&step.0);
                    let keep: Vec<Op> = again.iter().filter(|o| set.as_ref().is_some_and(|s| op_lines(o).is_some_and(|l| l.iter().all(|x| !s.contains(x))))).cloned().collect();
                    self.redo.push((inv, self.undo.len(), again));
                    self.make_again(&mut all, keep);
                }
                Some(Ok(all.expect("a step's result")))
            }
            Err(e) => {
                if redo { self.redo.push((step, base, after)) } else { self.undo.push(Back { step, again: None }) }
                self.make_again(&mut all, again);
                Some(Err(e))
            }
        }
    }

    /// Edits off the history made again in order, each kept on the undo stack.
    fn make_again(&mut self, all: &mut Option<EditResult>, again: Vec<Op>) {
        for op in again {
            if let Ok((r, inv)) = self.commit(op.clone()) {
                merge_results(all, r);
                self.undo.push(Back { step: inv, again: Some(op) });
            }
        }
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
        let phases = self.phases();
        let (r, inv) = self.commit(op)?;
        let value = inspect(self);
        let _ = self.restore(inv);
        self.set_phases(phases);
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
        self.step(false)
    }

    pub fn redo(&mut self) -> Option<Result<EditResult, Vec<Issue>>> {
        self.step(true)
    }

    fn commit(&mut self, op: Op) -> Result<(EditResult, (Op, Rates)), Vec<Issue>> {
        self.commit_recorded(op, true, None)
    }

    /// T-1649: an undo, a redo or a trial's rollback puts back what an edit took, so it is not
    /// checked afresh: track the saved game already had problems on (an old save under a newer
    /// water mask) would otherwise make the restore refuse, leaving a trial's deletions in place.
    /// T-3646: and the rates, pacing and departures the edit moved, as they were, where it finds
    /// them as the edit left them.
    fn restore(&mut self, step: (Op, Rates)) -> Result<(EditResult, (Op, Rates)), Vec<Issue>> {
        let (op, rates) = step;
        self.commit_recorded(op, false, Some(&rates))
    }

    /// `commit_with` with services, and what it moved (`Rates`) for its inverse to put back;
    /// `back`: what this commit itself puts back (an undo, redo or rollback).
    fn commit_recorded(&mut self, op: Op, checked: bool, back: Option<&Rates>) -> Result<(EditResult, (Op, Rates)), Vec<Issue>> {
        let before = self.line_rates_all();
        let solved = self.net.rest_solved;
        self.restore_to = back.and_then(|b| self.found(b, &before));
        let r = self.commit_with(op, checked, true);
        self.restore_to = None;
        let (r, inv) = r?;
        let after = self.line_rates_all();
        let n = before.len().max(after.len());
        let lines = (0..n).filter_map(|l| {
            let (b, a) = (before.get(l).cloned().flatten(), after.get(l).cloned().flatten());
            (b != a).then_some((l as u32, a, b))
        }).collect();
        Ok((r, (inv, Rates { lines, rest_solved: (self.net.rest_solved, solved) })))
    }

    /// T-3646: what `back` puts back, if the lines it names are as it expects (`now`, from
    /// `line_rates_all`); None otherwise, and the commit settles as an edit does.
    fn found(&self, back: &Rates, now: &[Option<LineRates>]) -> Option<(Vec<(u32, LineRates)>, bool)> {
        if back.rest_solved.0 != self.net.rest_solved || !back.lines.iter().all(|(l, want, _)| now.get(*l as usize).cloned().flatten() == *want) {
            return None;
        }
        Some((back.lines.iter().filter_map(|(l, _, set)| set.clone().map(|s| (*l, s))).collect(), back.rest_solved.1))
    }

    /// T-3646: each line's `LineRates`, None for a line not there.
    fn line_rates_all(&self) -> Vec<Option<LineRates>> {
        let n = &self.net;
        let mut kept: Vec<[Vec<(u32, usize, u32, u64, u32)>; 2]> = vec![Default::default(); n.line_count()];
        for (k, list) in [&n.settle_kept, &n.settle_kept_rest].into_iter().enumerate() {
            for e in list {
                if let Some(v) = kept.get_mut(e.0 as usize) { v[k].push(*e); }
            }
        }
        let z = [0.0; SLOTS];
        let p = [PaceIn::default(); SLOTS];
        (0..n.line_count()).map(|l| n.line_ok(l as u32).then(|| LineRates {
            tph: n.line_tph[l],
            tph_rest: n.line_tph_rest.get(l).copied().unwrap_or(z),
            pace: n.line_pace.get(l).copied().unwrap_or(z),
            pace_rest: n.line_pace_rest.get(l).copied().unwrap_or(z),
            pace_in: self.pace_in.get(l).copied().unwrap_or(p),
            pace_in_rest: self.pace_in_rest.get(l).copied().unwrap_or(p),
            trains: n.trains_of(l),
            phases: [(&n.line_phase, z), (&n.line_split, SPLIT_NONE), (&n.line_phase_rest, z), (&n.line_split_rest, SPLIT_NONE)].map(|(v, d)| v.get(l).copied().unwrap_or(d)),
            kept: std::mem::take(&mut kept[l]),
        })).collect()
    }

    /// T-3646: `restore_to`'s rates, pacing, trains and what the settle kept put back; the lines.
    fn put_rates(&mut self, set: &[(u32, LineRates)]) -> Vec<u32> {
        let count = self.net.line_count();
        let lines: Vec<u32> = set.iter().map(|e| e.0).filter(|&l| (l as usize) < count).collect();
        for (l, r) in set.iter().filter(|e| (e.0 as usize) < count) {
            let i = *l as usize;
            let n = &mut self.net;
            n.line_tph[i] = r.tph;
            put_at(&mut n.line_tph_rest, i, r.tph_rest, [0.0; SLOTS]);
            put_at(&mut n.line_pace, i, r.pace, [0.0; SLOTS]);
            put_at(&mut n.line_pace_rest, i, r.pace_rest, [0.0; SLOTS]);
            put_at(&mut self.pace_in, i, r.pace_in, [PaceIn::default(); SLOTS]);
            put_at(&mut self.pace_in_rest, i, r.pace_in_rest, [PaceIn::default(); SLOTS]);
            match r.trains {
                Some(t) => self.net.set_trains_of(i, Some(t)),
                None => self.net.line_trains[i] = None,
            }
        }
        let n = &mut self.net;
        n.settle_kept.retain(|e| !lines.contains(&e.0));
        n.settle_kept_rest.retain(|e| !lines.contains(&e.0));
        for (_, r) in set.iter().filter(|e| lines.contains(&e.0)) {
            n.settle_kept.extend_from_slice(&r.kept[0]);
            n.settle_kept_rest.extend_from_slice(&r.kept[1]);
        }
        lines
    }

    /// T-3646: the departures `put_rates` left to the placing, as they were.
    fn put_phases(&mut self, set: &[(u32, LineRates)]) {
        let count = self.net.line_count();
        let n = &mut self.net;
        for (l, r) in set.iter().filter(|e| (e.0 as usize) < count) {
            let i = *l as usize;
            for ((v, d), x) in [(&mut n.line_phase, [0.0; SLOTS]), (&mut n.line_split, SPLIT_NONE), (&mut n.line_phase_rest, [0.0; SLOTS]), (&mut n.line_split_rest, SPLIT_NONE)].into_iter().zip(r.phases) {
                put_at(v, i, x, d);
            }
        }
    }

    /// `service`: rebuild the services the edit touches (every real edit; a `trial` skips it).
    fn commit_with(&mut self, op: Op, checked: bool, service: bool) -> Result<(EditResult, Op), Vec<Issue>> {
        let (cost_before, built_before) = self.net.total_and_built_cost();
        let again = op.clone();
        // T-3542: the game's hours move every line (see `reslot` below)
        let game_hours = |o: &Op| matches!(o, Op::Hours { line: None, .. });
        let hours_game = match &op { Op::Batch(v) => v.iter().any(game_hours), o => game_hours(o) };
        // T-3482: each line's rates before the edit, weekdays' and weekends', None for a line not there
        let had: Vec<Option<[[f32; SLOTS]; 2]>> = (0..self.net.line_count()).map(|l| self.net.line_ok(l as u32).then(|| [WEEKDAYS, WEEKEND].map(|row| std::array::from_fn(|lev| self.net.line_tph_on(l, row, lev))))).collect();
        let worked: [bool; SLOTS] = std::array::from_fn(|s| self.net.slot_worked(s));
        let none_at = |lines: &[u32], row: usize| -> Vec<(u32, [bool; SLOTS])> {
            lines.iter().map(|&l| (l, had.get(l as usize).copied().flatten().map_or(worked, |r| std::array::from_fn(|s| worked[s] && r[row][s] == 0.0)))).filter(|e| e.1.contains(&true)).collect()
        };
        // (T-3542: also inside a batch, as a line of four or five rows sets its midday and afternoon with it)
        let auto_op = match &op {
            Op::Batch(ops) => ops.iter().find(|o| matches!(o, Op::AutoTrains { .. })),
            o => Some(o),
        };
        let switch = match auto_op {
            Some(Op::AutoTrains { id, auto, .. }) if self.net.line_auto.get(*id as usize).is_some_and(|a| a.is_some() != auto.is_some()) => {
                Some((*id, self.net.trains_of(*id as usize).unwrap_or_default().map(|n| n == 0)))
            }
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
        // T-3542: rows that split or join the slots work every line out again, as a load does
        let old_slots = self.net.period_slot();
        let reslotted = service && self.net.refresh_slots();
        // T-3542: a line whose hours fall into the slots otherwise (its hours, its rows) settles
        // afresh: what kept it no longer says where its rates go
        let moved = if service { std::mem::take(&mut self.net.mix_moved) } else { vec![] };
        if !moved.is_empty() {
            self.net.settle_kept.retain(|k| !moved.contains(&k.0));
            self.net.settle_kept_rest.retain(|k| !moved.contains(&k.0));
        }
        // the game's hours move every line: worked out again as a periods switch is (London 0.9 s
        // natively against 1.8-6.8 s as an edit of every line's schedule)
        let hours_all = hours_game && !moved.is_empty();
        if reslotted || hours_all {
            self.reslot(old_slots);
            let (cost_after, built_after) = self.net.total_and_built_cost();
            let all: Vec<u32> = (0..self.net.line_count() as u32).filter(|&l| self.net.line_ok(l)).collect();
            let service = super::service::ServiceDirty { rebuilt: all.clone(), recomputed: all.clone(), shared: vec![], demand: all.clone() };
            let dirty = Dirty { edges: der.edges, nodes: der.nodes, crossings: der.crossings, resources: (0..self.svc.cap.res.len() as u32).collect(), lines: all, service };
            let charge = if built_after - built_before > 1e-9 { built_after - built_before } else { 0.0 };
            return Ok((EditResult { dirty, cost_before, cost_after, built_before, built_after, charge }, inv));
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
        // T-3646: an undo or redo that finds the state its step left puts back the rates, pacing and
        // trains the step moved, so the settle below starts from a settled state and keeps it
        let set = self.restore_to.take();
        let mut schedules = t.schedules.clone();
        if let Some((set, _)) = &set {
            schedules.extend(set.iter().map(|e| e.0).filter(|&l| (l as usize) < self.net.line_count()));
            schedules.sort_unstable();
            schedules.dedup();
        }
        // T-3646: the lines sharing track with what the edit changed, as they did before it: a line
        // removed, or moved off their track, shares none with them after it, yet they may run more
        let mut changed: Vec<u32> = lines.iter().chain(&t.schedules).copied().collect();
        changed.sort_unstable();
        changed.dedup();
        let mut was_sharing = vec![];
        for r in 0..self.svc.cap.res.len() {
            let us = self.svc.cap.users_of(r);
            if us.iter().any(|u| changed.binary_search(&u.line).is_ok()) {
                was_sharing.extend(us.iter().map(|u| u.line));
            }
        }
        let settles = cap_dirty || !lines.is_empty() || !schedules.is_empty();
        let before = settles.then(|| self.demand_view());
        // after the demand view, so what is put back counts for demand where it moved
        let restored = set.map(|(set, solved)| {
            let l = self.put_rates(&set);
            // weekends settled on their own or not, as before the step: where that changes, the
            // weekend service is worked out again at the next weekend settle, as after a load
            if solved != self.net.rest_solved {
                self.net.rest_solved = solved;
                self.svc_rest = None;
            }
            (set, l)
        });
        let mut svc = self.svc.update_scoped(&self.net, &lines, &schedules, cap_dirty, Some((&der.edges, &cap_nodes)));
        if settles { self.mirror_weekend(&lines, &schedules, cap_dirty, &der.edges, &cap_nodes); }
        if let Some(before) = before {
            // T-752: a switch between counts and headways never buys or sells a train
            self.switching = switch;
            // the rescheduled lines themselves too: a bus or ferry line uses no capacity resource, so
            // `svc.shared` never names it, and its new count would never be settled into a rate (T-3704)
            let mut seeds = lines.clone();
            seeds.extend(svc.rebuilt.iter().chain(&svc.recomputed).chain(&svc.shared).chain(&t.weekend).chain(&schedules).chain(&was_sharing).copied());
            self.settle_scope = Some(self.sharing_closure(&seeds));
            // the levels a line ran no trains at, or every level of a line the edit made (or brought back)
            self.settle_edited = none_at(&changed, 0);
            self.settle_edited_rest = none_at(&changed, 1);
            let counts = match &restored {
                // T-3646: put back as settled, so not settled again: a settle from there is no
                // no-op where a line is kept off its fleet by having been left out of a scope (a
                // switch to headways keeps its counts, T-752, until a settle steps it)
                Some((_, l)) => {
                    // as a settle leaves every line's (a rebuilt service starts from the counts set)
                    for id in (0..self.net.line_count()).filter(|&i| self.net.line_ok(i as u32) && self.net.line_trains[i].is_some()) {
                        let t = self.net.slot_trains(id);
                        if let Some(s) = self.svc.lines.get_mut(id) { s.trains = t; }
                    }
                    self.svc.note_changed(l);
                    l.clone()
                }
                None => self.settle_train_counts(false),
            };
            self.settle_scope = None;
            self.settle_edited.clear();
            self.settle_edited_rest.clear();
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
            self.settle_edited_rest = none_at(&t.weekend, 1);
            svc.recomputed.extend(self.settle_weekend(false));
            self.settle_edited_rest.clear();
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
        // T-3735: and on weekends, with the lines whose weekends alone the edit changed
        let mut weekend = direct.clone();
        weekend.extend_from_slice(&t.weekend);
        weekend.sort_unstable();
        weekend.dedup();
        self.place_weekend(&weekend);
        if let Some((set, _)) = &restored {
            self.put_phases(set);
        }
        self.retime();
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
                let times = std::array::from_fn(|r| (0..SLOTS).map(|lev| ls.map_or(vec![], |s| s.stop_to_stop_safe(r, lev))).collect());
                (self.net.trains_of(l), self.net.line_tph[l], times)
            })
            .collect()
    }

    /// Whether line `l` moved for demand since `before` (`demand_view`): its trains, a rate by
    /// `DEMAND_TPH_EPS` or a stop-to-stop time by `DEMAND_TIME_EPS_S`.
    fn demand_moved(&self, l: u32, before: &[DemandView]) -> bool {
        let Some((trains, tph, times)) = before.get(l as usize) else { return true };
        let i = l as usize;
        if self.net.trains_of(i) != *trains || self.net.line_tph[i].iter().zip(tph).any(|(a, b)| (a - b).abs() >= DEMAND_TPH_EPS) {
            return true;
        }
        let Some(ls) = self.svc.lines.get(i) else { return true };
        (0..2).any(|r| self.net.slots().any(|lev| {
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
        self.net.line_phase.resize(self.net.line_phase.len().max(n), [0.0; SLOTS]);
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
            let ones = net.line_ones(l);
            let path = net.route_hops(stops, ones).ok().filter(|p| once || net.back_path_of(stops, p, ones).is_some());
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
            let ones = net.line_ones(l);
            let Some(path) = net.route_hops(stops, ones).ok().filter(|p| (once || net.back_path_of(stops, p, ones).is_some()) && (!turned.contains(&l) || net.path_reversals(p).len() <= turns)) else { continue };
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
            // T-3911: an edit of a rail line with a port among its stops is refused; loading such a save
            // (`derive`) keeps the line as it is
            if self.net.line_ok(l) && !self.net.is_bus(l) && !self.net.is_ferry(l) &&self.net.line_stops(l).iter().any(|&s| self.net.is_port(s)) {
                issues.push(Issue::PortStop { line: l });
            } else if self.net.line_ok(l) && self.net.line_path(l).iter().all(|&p| self.net.edge_ok(path_edge(p))) {
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
            Op::Rows { line, rows } => Op::Rows { line, rows: n.set_rows(line, rows)? },
            Op::Extra { id, x } => Op::Extra { id, x: n.set_extra(id, x)? },
            Op::Hours { line, cuts } => Op::Hours { line, cuts: n.set_hours(line, cuts)? },
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
                    if let Some(path) = self.net.route(&d.stops, &d.ones).filter(|p| self.net.turn_count(&d.stops, p) <= turns) {
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
