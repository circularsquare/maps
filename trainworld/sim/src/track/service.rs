//! Lines as run (SPEC 6.3): each line's two directions ("runs") with their path, stops, speed
//! limits and profiles per demand level; stop times, round trip, trains needed; the capacity pass;
//! and the trips the renderer draws.

use super::capacity::{self, Capacity};
use super::geom;
use super::net::{path_dir, path_edge, LoopKind, Network};
use super::params::*;
use super::profile::{self, Hold, Profile};

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
    /// Bare track ends where trains reverse between stops (T-797 B3): index into `nodes`, where
    /// the train's middle stands on arrival and how far on it leaves from, as `stop_s`/`jump`.
    /// Not stops: the profile stands the turnaround there and leaves them out of `arr`/`dep`.
    pub ends: Vec<(u32, f64, f64)>,
    /// Speed-limit breakpoints `[s, v]`, widened by half the train length.
    pub limits: Vec<[f64; 2]>,
    /// The hops between stops and ends with no holds (`run_profile`).
    pub hops: profile::Hops,
    /// No capacity delay.
    pub free: Profile,
    /// Per demand level: track holds and extra dwell per stop from the capacity pass.
    pub holds: [Vec<Hold>; 3],
    pub stop_holds: [Vec<f64>; 3],
    pub prof: [Profile; 3],
}

#[derive(Clone, Debug, Default)]
pub struct LineSvc {
    /// Alive, not broken, profiles built.
    pub ok: bool,
    /// `ok` and every edge and stop constructed: the line carries trains, loads capacity and
    /// has trips (SPEC 6.4). A line that is `ok` but not running shows its times as a preview.
    pub running: bool,
    /// The stops these profiles were built for (a change wakes demand).
    pub stops: Vec<u32>,
    /// A loop (T-798): each run goes from the first stop round to it again, and its trains take
    /// that run's next departure there.
    pub kind: LoopKind,
    pub runs: [Run; 2],
    pub cars: u32,
    pub train_len: f64,
    /// Per demand level: both runs with dwell, holds and their closing stands (`LineSvc::lap`);
    /// for a loop, twice the longer lap.
    pub round_trip: [f64; 3],
    pub trains: [u32; 3],
    /// Per demand level: capacity delay over the round trip.
    pub delay: [f64; 3],
    /// Seconds the line's local time runs ahead of the game clock (T-309, `Network::line_shift`):
    /// its periods and its boarding account's hours are its own region's.
    pub shift: f64,
    /// A bus line (T-1464): its runs have no path (empty `segs`, `nodes`, `stop_idx`), `stops`
    /// holds its stops' ids (`bus::BusStop::id`), and `bus` its plan in run 0's order.
    pub bus: Option<super::bus::Plan>,
}

impl LineSvc {
    pub fn is_bus(&self) -> bool {
        self.bus.is_some()
    }
    /// Riders a train or bus takes at crush.
    pub fn crush(&self) -> f64 {
        if self.is_bus() { BUS_CRUSH } else { super::boarding::CRUSH_PER_CAR * self.cars.max(1) as f64 }
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
    /// What a train stands at the end of a run: the turnaround, or the dwell where a circle carries
    /// on round (T-798).
    pub fn close_stand(&self, net: &Network, line: u32) -> f64 {
        if self.kind == LoopKind::Circle { net.line_dwell[line as usize] as f64 } else { net.line_turn[line as usize] as f64 }
    }
    pub fn is_loop(&self) -> bool {
        self.kind != LoopKind::None
    }
    /// Trains a round trip needs at `tph` each way; a loop's count is even, the same each way.
    pub fn fleet(&self, round_trip: f64, tph: f64) -> u32 {
        if tph <= 0.0 {
            0
        } else if self.is_loop() {
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
}

/// A train running when an edit is applied: where it is on its profile (T-206).
#[derive(Clone, Copy, Debug)]
pub struct InFlight {
    line: u32,
    run: u8,
    level: u8,
    dep: f64,
    s: f64,
    /// Standing at a stop: seconds before it would have left; with `s` infinite, standing at its
    /// last stop: seconds it has stood there.
    stopped_left: Option<f64>,
}

/// The service as an edit found it (`Service::before_edit`, then `note_changes`).
#[derive(Clone, Debug, Default)]
pub struct Before {
    lat: Vec<Option<Lattice>>,
    flight: Vec<InFlight>,
    lens: Vec<[f64; 6]>,
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
/// ones round the circuit; 0 otherwise), and `shift`, the line's local time ahead of the game
/// clock (T-309): its periods start on its own region's hours. `lead`: seconds every period's
/// departures leave the ends before its start, so a train belongs to the period in force when it
/// reaches the line's middle (T-835); one figure for both runs keeps a whole-line fleet aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    pub tph: [f32; LEVELS],
    pub off: [f64; LEVELS],
    pub lead: f64,
    pub shift: f64,
}

impl Lattice {
    pub const NONE: Lattice = Lattice { tph: [0.0; LEVELS], off: [0.0; LEVELS], lead: 0.0, shift: 0.0 };
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
}

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
pub(crate) fn level_stretches() -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = vec![];
    for p in day_order() {
        match out.last_mut() {
            Some((_, last)) if PERIOD_LEVEL[*last] == PERIOD_LEVEL[p] && PERIOD_HOURS[*last].1 == PERIOD_HOURS[p].0 => *last = p,
            _ => out.push((p, p)),
        }
    }
    out
}

/// A stretch (`level_stretches`) of `day` for run `r` of a pattern: (first departure, headway,
/// departures), or None without service.
fn stretch_points(lat: &Lattice, r: usize, day: i64, (first, last): (usize, usize)) -> Option<(f64, f64, i64)> {
    let lev = PERIOD_LEVEL[first];
    let tph = lat.tph[lev] as f64;
    if tph <= 0.0 {
        return None;
    }
    let hw = 3600.0 / tph;
    let (h0, h1) = (PERIOD_HOURS[first].0, PERIOD_HOURS[last].1);
    // a period followed by no service runs to its own end, so a line's last trains of the evening
    // or of a peak are not pulled early (T-835)
    let next = PERIOD_HOURS.iter().position(|&(a, _)| a == h1 % 24.0).map_or(0.0, |q| lat.tph[PERIOD_LEVEL[q]]);
    let end_lead = if next > 0.0 { lat.lead } else { 0.0 };
    let (start, end) = (day as f64 * DAY_S + h0 * 3600.0 - lat.lead, day as f64 * DAY_S + h1 * 3600.0 - end_lead);
    let ps = start + if r == 1 { lat.off[lev] } else { 0.0 };
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
fn lattice_deps(lat: &Lattice, r: usize, a: f64, b: f64, out: &mut Vec<(f64, u8, f64)>) {
    let s = lat.shift;
    if s == 0.0 {
        return lattice_deps_local(lat, r, a, b, out);
    }
    // a second either side, so rounding in `a + s` never loses a departure on the window's edge
    let start = out.len();
    lattice_deps_local(lat, r, a + s - 1.0, b + s + 1.0, out);
    let mut k = start;
    for i in start..out.len() {
        let (d, lev, hw) = out[i];
        let d = d - s;
        if d >= a && d < b {
            out[k] = (d, lev, hw);
            k += 1;
        }
    }
    out.truncate(k);
}

/// One run's departures from a pattern in `[a, b)` of the line's local time, with level and
/// headway: per stretch of one level (`level_stretches`), `start - lead + off + k * headway`,
/// leaving out any closer than half a headway after the stretch before's last (MIN_GAP_SHARE). A
/// pure function of the pattern, so every window agrees.
pub(crate) fn lattice_deps_local(lat: &Lattice, r: usize, a: f64, b: f64, out: &mut Vec<(f64, u8, f64)>) {
    if !(b > a) {
        return;
    }
    let stretches = level_stretches();
    let d0 = (a / DAY_S).floor() as i64;
    let d1 = (b / DAY_S).floor() as i64;
    // the last departure before day d0: the day before's last stretch with service
    let mut last: Option<f64> = None;
    for &s in stretches.iter().rev() {
        if let Some((ps, hw, n)) = stretch_points(lat, r, d0 - 1, s) {
            last = Some(ps + (n - 1) as f64 * hw);
            break;
        }
    }
    // the next day's first stretch starts `lead` before it
    for day in d0..=d1 + 1 {
        for &s in &stretches {
            let Some((ps, hw, n)) = stretch_points(lat, r, day, s) else { continue };
            let mut k0 = 0i64;
            if let Some(l) = last {
                while k0 < n && ps + k0 as f64 * hw - l < MIN_GAP_SHARE * hw {
                    k0 += 1;
                }
            }
            if k0 >= n {
                continue;
            }
            let lev = PERIOD_LEVEL[s.0] as u8;
            let mut k = k0.max(((a - ps) / hw).floor() as i64 - 1).max(k0);
            while k < n {
                let d = ps + k as f64 * hw;
                if d >= b {
                    break;
                }
                if d >= a {
                    out.push((d, lev, hw));
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
}

fn build_run(net: &Network, l: u32, rev: bool, train_len: f64, dwell: f64) -> Option<Run> {
    let path = net.line_path(l);
    let running = net.line_constructed(l);
    let mut run = Run::default();
    run.segs = if rev { path.iter().rev().map(|p| p ^ 1).collect() } else { path.to_vec() };
    let mut s = 0.0;
    run.nodes.push(net.path_nodes(run.segs[0]).0);
    run.node_s.push(0.0);
    let mut raw: Vec<[f64; 2]> = vec![];
    let top = net.line_top(l);
    for &p in &run.segs {
        let e = path_edge(p);
        let len = net.edge_len[e as usize];
        run.seg_off.push(s);
        let pcs = net.edge_pieces(e);
        // T-817: the lowest of the curve's speed, the track's cap and the trains' top speed
        let cap = net.v_cap(e).min(top);
        let mut add = |a: f64, v: f64| match raw.last() {
            Some(l) if l[1] == v => {}
            _ => raw.push([a, v]),
        };
        if path_dir(p) == 0 {
            for pc in pcs {
                add(s + pc.s0, pc.v_curve().min(cap));
            }
        } else {
            for pc in pcs.iter().rev() {
                add(s + len - (pc.s0 + pc.len), pc.v_curve().min(cap));
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
    let mut stops: Vec<u32> = net.line_stops(l).to_vec();
    if rev {
        stops.reverse();
    }
    let turns = net.stop_reversals(&run.segs);
    run.stop_idx = super::net::match_stops(&run.nodes, &stops, &turns)?.into_iter().map(|k| k as u32).collect();
    let node_s: Vec<f64> = run.stop_idx.iter().map(|&i| run.node_s[i as usize]).collect();
    let bare: Vec<usize> = net.path_reversals(&run.segs).into_iter().filter(|&k| net.bare_end(run.nodes[k])).collect();
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
        let turn = turns.contains(&(run.stop_idx[j] as usize));
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
        None => profile::hops(&run.limits, &run.stop_s, &run.jump),
        Some((s, jump, ..)) => profile::hops(&run.limits, &s, &jump),
    };
    run.free = run_profile(&run, &dw, turn, &[]);
    Some(run)
}

/// `profile::run` over a run's stops and the bare track ends it reverses at, standing `turn` at
/// each end; `arr`/`dep` keep the stops only.
fn run_profile(run: &Run, dw: &[f64], turn: f64, holds: &[Hold]) -> Profile {
    let Some((s, jump, stand, keep)) = with_ends(run, dw, turn) else {
        return profile::run_with(&run.limits, &run.stop_s, &run.jump, dw, holds, &run.hops);
    };
    let mut p = profile::run_with(&run.limits, &s, &jump, &stand, holds, &run.hops);
    let pick = |v: &[f64]| v.iter().zip(&keep).filter(|x| *x.1).map(|x| *x.0).collect::<Vec<f64>>();
    (p.arr, p.dep) = (pick(&p.arr), pick(&p.dep));
    p
}

/// A run's stops with the bare track ends it reverses at put in: offsets, jumps, standing times
/// and whether each is a stop. None for a run with no such ends.
#[allow(clippy::type_complexity)]
fn with_ends(run: &Run, dw: &[f64], turn: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>)> {
    if run.ends.is_empty() {
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
            keep.push(false);
            e += 1;
        }
        s.push(run.stop_s[j]);
        jump.push(run.jump[j]);
        stand.push(dw.get(j).copied().unwrap_or(0.0));
        keep.push(true);
    }
    Some((s, jump, stand, keep))
}

/// Scheduled standing time per stop: the dwell, or the turnaround where trains reverse (T-797);
/// none at the ends.
fn stand_times(run: &Run, dwell: f64, turn: f64) -> Vec<f64> {
    let n = run.stop_s.len();
    (0..n).map(|i| if i == 0 || i + 1 == n { 0.0 } else if run.turn[i] { turn } else { dwell }).collect()
}

fn build_line(net: &Network, l: u32) -> LineSvc {
    let mut ls = LineSvc::default();
    if !net.line_ok(l) || net.line_broken[l as usize] {
        return ls;
    }
    if let Some(r) = net.line_bus[l as usize].as_ref() {
        return build_bus(net, l, r);
    }
    ls.stops = net.line_stops(l).to_vec();
    if ls.stops.len() < 2 { return ls; }
    ls.kind = net.line_loop(l);
    let min_plat = net.line_stops(l).iter().map(|&s| net.node_platform[s as usize]).min().unwrap_or(0) as f64;
    let fit = (min_plat / CAR_LEN).floor().max(1.0) as u32;
    let want = net.line_cars[l as usize] as u32;
    ls.cars = if want == 0 { fit } else { want.min(fit) };
    ls.train_len = ls.cars as f64 * CAR_LEN;
    let dwell = net.line_dwell[l as usize] as f64;
    match (build_run(net, l, false, ls.train_len, dwell), build_run(net, l, true, ls.train_len, dwell)) {
        (Some(a), Some(b)) => {
            ls.runs = [a, b];
            ls.ok = true;
            ls.running = net.line_constructed(l);
            ls.shift = net.line_shift(l);
        }
        _ => {}
    }
    ls
}

/// A bus line's service (T-1464): its stops from `bus::plan`, each run at a flat speed, standing the
/// line's dwell at the train stations, with no capacity holds.
fn build_bus(net: &Network, l: u32, r: &super::bus::BusRoute) -> LineSvc {
    let mut ls = LineSvc::default();
    let plan = super::bus::plan(net, l, &r.points, &r.kinds, r.built);
    if plan.stops.len() < 2 {
        return ls;
    }
    let dwell = net.line_dwell[l as usize] as f64;
    let n = plan.stops.len();
    let fwd_s: Vec<f64> = plan.stops.iter().map(|s| s.s).collect();
    let fwd_dw: Vec<f64> = plan.stops.iter().map(|s| if s.station.is_some() { dwell } else { 0.0 }).collect();
    let len = plan.len;
    let rev_s: Vec<f64> = fwd_s.iter().rev().map(|s| len - s).collect();
    let rev_dw: Vec<f64> = fwd_dw.iter().rev().copied().collect();
    let make = |s: Vec<f64>, dw: &[f64]| {
        let p = super::bus::profile(&s, dw);
        Run {
            len,
            turn: vec![false; n],
            jump: vec![0.0; n],
            limits: vec![[0.0, super::bus::BUS_V]],
            free: p.clone(),
            holds: Default::default(),
            stop_holds: std::array::from_fn(|_| vec![0.0; n]),
            prof: std::array::from_fn(|_| p.clone()),
            stop_s: s,
            ..Default::default()
        }
    };
    ls.stops = plan.stops.iter().map(|s| s.id).collect();
    ls.runs = [make(fwd_s, &fwd_dw), make(rev_s, &rev_dw)];
    ls.cars = 1;
    ls.train_len = super::bus::BUS_LEN_M;
    ls.ok = true;
    ls.running = r.built;
    ls.shift = net.line_shift(l);
    ls.bus = Some(plan);
    ls
}

fn profile_for(net: &Network, l: u32, run: &Run, level: usize) -> Profile {
    if net.line_bus[l as usize].is_some() {
        return run.free.clone();
    }
    let turn = net.line_turn[l as usize] as f64;
    let mut dw = stand_times(run, net.line_dwell[l as usize] as f64, turn);
    for (d, h) in dw.iter_mut().zip(&run.stop_holds[level]) {
        *d += h;
    }
    let mut p = run_profile(run, &dw, turn, &run.holds[level]);
    p.hold += run.stop_holds[level].iter().sum::<f64>();
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
        let old_times: Vec<(u32, [Vec<Vec<f64>>; 2])> = rebuilt
            .iter()
            .map(|&l| (l, std::array::from_fn(|r| (0..LEVELS).map(|lev| self.lines[l as usize].stop_to_stop_safe(r, lev)).collect())))
            .collect();
        let mut demand: Vec<u32> = rescheduled.to_vec();
        // The incremental capacity pass needs the old and new paths of the rebuilt lines.
        let mut path_edges: Vec<u32> = vec![];
        let mut path_nodes: Vec<u32> = vec![];
        let mut add_path = |ls: &LineSvc| {
            if ls.running {
                path_edges.extend(ls.runs[0].segs.iter().map(|&p| path_edge(p)));
                path_nodes.extend_from_slice(&ls.runs[0].nodes);
            }
        };
        for &l in rebuilt {
            add_path(&self.lines[l as usize]);
        }
        for &l in rebuilt {
            let (old_ok, old_run) = (self.lines[l as usize].ok, self.lines[l as usize].running);
            let old_stops = std::mem::take(&mut self.lines[l as usize].stops);
            self.lines[l as usize] = build_line(net, l);
            let now = &self.lines[l as usize];
            if now.ok != old_ok || now.running != old_run || now.stops != old_stops {
                demand.push(l);
            }
        }
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
                std::array::from_fn(|r| (0..LEVELS).map(|lev| ls.stop_to_stop_safe(r, lev)).collect())
            };
            let mut changed = false;
            for r in 0..2 {
                for lev in 0..LEVELS {
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
                let stand = ls.close_stand(net, l as u32);
                for lev in 0..LEVELS {
                    let (a, b) = (&ls.runs[0].prof[lev], &ls.runs[1].prof[lev]);
                    ls.round_trip[lev] = if ls.is_loop() { 2.0 * (a.duration.max(b.duration) + stand) } else { a.duration + b.duration + 2.0 * stand };
                    ls.delay[lev] = a.hold + b.hold;
                    ls.trains[lev] = ls.fleet(ls.round_trip[lev], net.line_tph[l][lev] as f64);
                }
            }
            if changed {
                d.recomputed.push(l as u32);
                let moved = (0..2).any(|r| {
                    (0..LEVELS).any(|lev| {
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
        let mut seed = vec![false; self.lines.len()];
        for &l in rebuilt.iter().chain(rescheduled) {
            if let Some(x) = seed.get_mut(l as usize) {
                *x = true;
            }
        }
        for r in 0..self.cap.res.len() {
            let us = self.cap.users_of(r);
            if us.iter().any(|u| seed[u.line as usize]) {
                d.shared.extend(us.iter().map(|u| u.line));
            }
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

    pub fn note_changed(&mut self, lines: &[u32]) {
        if self.changed.len() > 1 << 20 {
            self.changed.clear();
            self.changed_all = true;
        }
        self.changed.extend_from_slice(lines);
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

    /// Every line from scratch.
    pub fn update_all(&mut self, net: &Network) -> ServiceDirty {
        let all: Vec<u32> = (0..net.line_count() as u32).collect();
        self.update(net, &all, &all, true)
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
        let off = std::array::from_fn(|lev| {
            let t = tph[lev] as f64;
            if net.line_trains[l].is_some() && t > 0.0 && !ls.is_loop() { ls.trip_len(net, l as u32, 0, lev) % (3600.0 / t) } else { 0.0 }
        });
        // a loop's middle is half a lap from its first stop (T-798): the same figure
        let lead = (0.25 * (ls.runs[0].prof[1].duration + ls.runs[1].prof[1].duration)).clamp(0.0, LEAD_MAX_S);
        Some(Lattice { tph, off, lead, shift: ls.shift })
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
        let lat = (0..self.lines.len()).map(|l| self.lattice(net, l)).collect();
        let mut flight = vec![];
        if now.is_finite() {
            for t in self.trips(net, now, now + 1e-3) {
                let p = &self.lines[t.line as usize].runs[t.run as usize].prof[t.level as usize];
                let tau = now - t.draw;
                if tau < 0.0 {
                    continue;
                }
                if tau >= p.duration {
                    // standing at the terminal: as long again after the new profile's arrival
                    flight.push(InFlight { line: t.line, run: t.run, level: t.level, dep: t.dep, s: f64::INFINITY, stopped_left: Some(tau - p.duration) });
                    continue;
                }
                let (s, v) = profile::state_at(&p.phases, tau);
                let stopped_left = (v < 0.01).then(|| time_reaching(&p.phases, p.duration, s + 0.05) - tau);
                flight.push(InFlight { line: t.line, run: t.run, level: t.level, dep: t.dep, s, stopped_left });
            }
        }
        let lens = (0..self.lines.len()).map(|l| self.trip_lens(net, l)).collect();
        Before { lat, flight, lens }
    }

    /// A line's trip lengths (run and turnaround) per run and level.
    fn trip_lens(&self, net: &Network, l: usize) -> [f64; 6] {
        let ls = &self.lines[l];
        std::array::from_fn(|k| if ls.ok && l < net.line_count() { ls.trip_len(net, l as u32, k / 3, k % 3) } else { 0.0 })
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
        self.eras.resize(self.lines.len(), vec![]);
        self.shifts.resize(self.lines.len(), vec![]);
        for f in &before.flight {
            let l = f.line as usize;
            let Some(ls) = self.lines.get(l).filter(|ls| ls.running) else { continue };
            let p = &ls.runs[f.run as usize].prof[f.level as usize];
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
        for l in 0..self.lines.len() {
            let Some(after) = self.lattice(net, l) else {
                self.eras[l].clear();
                self.shifts[l].clear();
                continue;
            };
            let was = before.lat.get(l).copied().flatten().unwrap_or(Lattice::NONE);
            let lens = self.trip_lens(net, l);
            let reshaped = before.lens.get(l).is_some_and(|b| b.iter().zip(&lens).any(|(x, y)| (x - y).abs() > 1e-6));
            let v = &mut self.eras[l];
            match v.last() {
                // a change at the same instant: the pattern before the instant stands for
                // everything before it
                Some(e) if e.until >= now => {}
                _ if was != after || reshaped => {
                    let drawn = before.flight.iter().filter(|f| f.line as usize == l).map(|f| (f.run, f.dep)).collect();
                    v.push(Era { until: now, lat: was, drawn })
                }
                _ => {}
            }
        }
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
    pub fn encode_eras(&self, net: &Network) -> Vec<u8> {
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
                e.lat.tph.iter().for_each(|x| body.extend(x.to_le_bytes()));
                e.lat.off.iter().for_each(|x| body.extend(x.to_le_bytes()));
                body.extend(e.lat.lead.to_le_bytes());
                body.extend(e.lat.shift.to_le_bytes());
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
    pub fn decode_eras(&mut self, bytes: &[u8]) -> Result<(), String> {
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
                let tph = [num!(f32), num!(f32), num!(f32)];
                let off = [num!(f64), num!(f64), num!(f64)];
                let (lead, shift) = (num!(f64), num!(f64));
                let mut drawn = vec![];
                for _ in 0..num!(u32) {
                    drawn.push((num!(u8), num!(f64)));
                }
                v.push(Era { until, lat: Lattice { tph, off, lead, shift }, drawn });
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

    /// A line's eras (empty for most).
    pub fn eras_of(&self, l: usize) -> &[Era] {
        self.eras.get(l).map_or(&[], |v| &v[..])
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
            for lev in 0..LEVELS {
                dur = dur.max(ls.trip_len(net, l as u32, r, lev));
            }
        }
        let mut hw = 0.0f64;
        let cur = [net.line_tph[l]];
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
        let Some(cur) = self.lattice(net, l) else { return };
        let ls = &self.lines[l];
        let eras = self.eras_of(l);
        let shifts = self.shifts.get(l).map_or(&[][..], |v| &v[..]);
        let mut seg_start = f64::NEG_INFINITY;
        let mut last: Option<f64> = None;
        let mut buf: Vec<(f64, u8, f64)> = vec![];
        for i in 0..=eras.len() {
            let (lat, seg_end) = if i < eras.len() { (&eras[i].lat, eras[i].until) } else { (&cur, f64::INFINITY) };
            // an era is followed by another pattern, whose first departures need its last one
            let lo = if i < eras.len() { seg_start.max(a.min(seg_end - ERA_LOOKBACK_S)) } else { seg_start.max(a) };
            let hi = if i < eras.len() { seg_end } else { b };
            buf.clear();
            lattice_deps(lat, r, lo, hi, &mut buf);
            let prev = last;
            for &(d, lev, hw) in &buf {
                if prev.is_some_and(|p| d - p < MIN_GAP_SHARE * hw) {
                    continue;
                }
                last = Some(d);
                if d >= a && d < b {
                    let draw = d + shifts.iter().find(|x| x.0 as usize == r && x.1 == d).map_or(0.0, |x| x.2);
                    let end = draw + ls.trip_len(net, l as u32, r, lev as usize);
                    out.push(Trip { line: l as u32, run: r as u8, level: lev, dep: d, draw, end, hw, next: None });
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
        // whether train `w` may stand until `d`: within the departure's headway, and not across a
        // change unless it was drawn then
        let can = |trips: &[Trip], w: usize, d: f64, hw: f64| {
            d >= ready[w] - HANDOVER_TOL_S
                && d - ready[w] <= HANDOVER_MAX_S.min(hw.max(1.0)) + HANDOVER_TOL_S
                && eras.iter().all(|e| !(ready[w] < e.until && d > e.until) || e.drawn.contains(&(trips[w].run, trips[w].dep)))
        };
        for r in 0..2u8 {
            // trains finishing run r stand at the first stop of the other run, or of run r on a loop
            let mut ev: Vec<(f64, u8, usize)> = vec![];
            let mut deps: Vec<(f64, f64)> = vec![];
            for (i, t) in trips.iter().enumerate() {
                if t.run == r {
                    ev.push((ready[i] - HANDOVER_TOL_S, 0, i));
                }
                if (t.run == r) == is_loop {
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
        for (l, ls) in self.lines.iter().enumerate() {
            if !ls.running {
                continue;
            }
            // far enough back that every train drawn in the window is handed over as it would be
            // from any other window
            let (dur, hcap) = self.reach(net, l);
            let (a, b) = (t0 - dur - 2.0 * hcap - 60.0, t1 + dur + hcap + HANDOVER_TOL_S + 1.0);
            buf.clear();
            for r in 0..2 {
                self.run_deps(net, l, r, a, b, &mut buf);
            }
            let eras = self.eras_of(l);
            Self::hand_over(&mut buf, eras, ls.is_loop());
            // a train that had left service before a change stays gone, whatever the new profile
            // says about how long its trip would have taken
            let gone = |t: &Trip| eras.iter().any(|e| t.draw < e.until && t.end > e.until && !e.drawn.contains(&(t.run, t.dep)));
            out.extend(buf.iter().filter(|t| t.draw < t1 && t.end > t0 && !gone(t)));
        }
        out.sort_by(|a, b| a.draw.total_cmp(&b.draw).then((a.line, a.run).cmp(&(b.line, b.run))));
        out
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
        if let (Some(r), Some(plan)) = (net.line_bus.get(line as usize).and_then(|b| b.as_ref()), self.lines[line as usize].bus.as_ref()) {
            let road = if run == 0 { s } else { plan.len - s };
            return super::bus::at_road(&r.points, &plan.cum, road);
        }
        let (e, _, d) = self.segment_at(net, line, run, s);
        let (x, y, _) = geom::pos_at(net.edge_pieces(e), d);
        (x, y)
    }
}

impl LineSvc {
    fn stop_to_stop_safe(&self, r: usize, lev: usize) -> Vec<f64> {
        if !self.ok {
            return vec![];
        }
        self.stop_to_stop(r, lev)
    }
}
