//! The boarding account (T-164, notes/T-164.md, docs/track-and-service.md 6.3): demand's riders per
//! period turned into a load per train and a crowd per platform.
//!
//! Demand decides who travels and by which line (SPEC 4); it reports, per line, direction, stop and
//! period, how many board and how many alight. Here those boardings arrive at the platform over the
//! period (a simple in-period shape, below), and each train of the line, in the order it reaches the
//! stop, lets off the line's share alighting there and takes on whoever is waiting, up to its crush
//! load. Whoever does not fit waits for the next train of the same line and direction. Trains do
//! not block each other (no signals, SPEC 2.1), so each direction of each line is one independent
//! loop over its departures and stops.
//!
//! The account is a pure function of the service and the riders: nothing is saved. It is worked
//! out for a whole day whenever either changes (a line's inputs are hashed, so an edit redoes only
//! the lines it changed): a day of departures from 03:00 from empty platforms, or, for a direction
//! where someone is still waiting at the end of that day, two days with the second kept, so the
//! queue carries round. Each game hour's slice (`Boarding::hour`) is then read from it for the
//! trips drawn that hour.

use super::net::Network;
use super::params::*;
use super::service::{LineSvc, Service, Trip};
use std::collections::HashMap;

/// Seats and riders at crush load per 20 m car: the demand kernel's `SEATS_PER_CAR` and
/// `CRUSH_PER_CAR` (sim/src/demand/api.rs, a New York subway car).
pub const SEATS_PER_CAR: f64 = 44.0;
pub const CRUSH_PER_CAR: f64 = 160.0;
pub const DAY: f64 = 86400.0;
pub const PERIODS: usize = PERIOD_HOURS.len();

/// How each period's boardings spread over it: the peaks and midday rise to the middle and fall
/// away (a tent), the evening falls from its start, the night is busiest at its two ends. Scaled so
/// the busiest hour has the demand model's busiest-hour factor (T-087) times the period's mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Tent,
    Falling,
    Valley,
}
pub const SHAPE: [Shape; PERIODS] = [Shape::Tent, Shape::Tent, Shape::Tent, Shape::Falling, Shape::Valley];

/// The shape's deviation from flat, `g(x)` on [0, 1), mean 0, between -1 and 1.
pub fn shape_g(s: Shape, x: f64) -> f64 {
    let tent = 1.0 - (2.0 * x - 1.0).abs(); // 0 at the ends, 1 in the middle
    match s {
        Shape::Tent => 2.0 * tent - 1.0,
        Shape::Falling => 1.0 - 2.0 * x,
        Shape::Valley => 1.0 - 2.0 * tent,
    }
}

/// `x + c * integral of g from 0 to x`: the share of a period's arrivals by fraction `x` of it.
pub fn shape_cum(s: Shape, c: f64, x: f64) -> f64 {
    let tent = if x <= 0.5 { 2.0 * x * x - x } else { 3.0 * x - 2.0 * x * x - 1.0 };
    let g = match s {
        Shape::Tent => tent,
        Shape::Falling => x - x * x,
        Shape::Valley => -tent,
    };
    x + c * g
}

/// How strongly a period's shape is applied: `c` such that the busiest clock hour is `factor`
/// times the mean hour, at most 1 (no negative rates). `busiest` is the mean of `g` over that
/// hour: a tent's middle hour (beside the middle on an even number of hours), the falling shape's
/// first, the valley's first or last.
pub fn shape_strength(s: Shape, hours: f64, factor: f64) -> f64 {
    let even = (hours.round() as i64) % 2 == 0;
    let busiest = match s {
        Shape::Tent if even => 1.0 - 2.0 / hours,
        Shape::Tent | Shape::Falling => 1.0 - 1.0 / hours,
        Shape::Valley => 1.0 - 2.0 / hours,
    };
    if busiest <= 0.0 { 0.0 } else { ((factor - 1.0) / busiest).clamp(0.0, 1.0) }
}

/// The period at game time `t`.
pub fn period_at(t: f64) -> usize {
    let h = (t / 3600.0).floor().rem_euclid(24.0);
    PERIOD_HOURS.iter().position(|&(a, b)| h >= a && h < b).unwrap_or(0)
}

/// Riders arriving at one platform (one stop of one direction of one line) over the day.
#[derive(Clone, Debug, Default)]
pub struct Arrivals {
    /// boardings in each period
    pub per: [f64; PERIODS],
    pub c: [f64; PERIODS],
    pub day: f64,
}

impl Arrivals {
    pub fn new(per: [f64; PERIODS], peak: &[f64; PERIODS]) -> Self {
        let c = std::array::from_fn(|p| {
            let (h0, h1) = PERIOD_HOURS[p];
            shape_strength(SHAPE[p], h1 - h0, peak[p])
        });
        Arrivals { per, c, day: per.iter().sum() }
    }
    /// Riders arrived since the start of game day 0 (negative before it).
    pub fn cum(&self, t: f64) -> f64 {
        let d = (t / DAY).floor();
        let s = t - d * DAY;
        let mut tot = d * self.day;
        for p in 0..PERIODS {
            let (a, b) = (PERIOD_HOURS[p].0 * 3600.0, PERIOD_HOURS[p].1 * 3600.0);
            if s >= b {
                tot += self.per[p];
            } else if s > a {
                tot += self.per[p] * shape_cum(SHAPE[p], self.c[p], (s - a) / (b - a));
            }
        }
        tot
    }
    /// Riders arriving per second at `t`.
    pub fn rate(&self, t: f64) -> f64 {
        let p = period_at(t);
        let (a, b) = (PERIOD_HOURS[p].0 * 3600.0, PERIOD_HOURS[p].1 * 3600.0);
        let s = t.rem_euclid(DAY);
        self.per[p] / (b - a) * (1.0 + self.c[p] * shape_g(SHAPE[p], ((s - a) / (b - a)).clamp(0.0, 1.0)))
    }
}

/// What demand puts on one line: per period, direction and stop (in that direction's order),
/// boardings and alightings over the period. `n` is the line's stop count.
#[derive(Clone, Debug, Default)]
pub struct LineRiders {
    pub n: usize,
    /// `[(period * 2 + run) * n + stop]`
    pub board: Vec<f64>,
    pub alight: Vec<f64>,
}

/// One direction of one line over the day.
#[derive(Clone, Debug, Default)]
pub struct RunAccount {
    pub n: usize,
    /// Departures from the first stop, seconds into the day, sorted (one periodic day).
    pub deps: Vec<f64>,
    pub levels: Vec<u8>,
    /// The departure times from each stop, from the trip's start, per demand level.
    pub stop_dep: [Vec<f64>; LEVELS],
    /// `[trip * n + stop]`: riders aboard leaving the stop (0 at the last); riders left on the
    /// platform as it leaves; riders on the platform as it arrived.
    pub load: Vec<f32>,
    pub q_after: Vec<f32>,
    /// Per stop, the trips in the order they leave it in a day (`stop * trips + k`), and the time
    /// of day each leaves it.
    pub order: Vec<u32>,
    pub order_t: Vec<f64>,
    /// Per stop and hour of the day (`stop * 24 + hour`): riders who could not get on the first
    /// train to leave after they arrived, and the longest anyone waited for a train leaving then.
    pub left_h: Vec<f32>,
    pub wait_h: Vec<f32>,
    pub arrivals: Vec<Arrivals>,
    /// Totals over the day, for checks: riders who arrived, boarded, alighted.
    pub arrived: f64,
    pub boarded: f64,
    pub alighted: f64,
    /// Riders still waiting at a stop once the day's last train has left it.
    pub leftover: f64,
    /// When the line's trains started, in its local time, if within this account's day (T-680):
    /// its platforms are empty then. Departures by time of day before it are the day wrapping
    /// round, not trains that ran. `NEG_INFINITY` otherwise.
    pub open: f64,
    /// Per period and stop (`p * n + k`), the km a boarder there rides on this line (`ride_km`).
    pub ride_km: Vec<f64>,
}

/// One line's counts over a window of the day (`Boarding::window`).
#[derive(Default)]
struct Tally {
    stops: Vec<f64>,
    boarded: f64,
    left: f64,
    fullest: f64,
    full: Vec<f64>,
    /// trips that reach a stop in the window, the count `full` is out of (T-481)
    trains: f64,
}

#[derive(Clone, Debug, Default)]
pub struct LineAccount {
    key: u64,
    /// The line's local time ahead of the game clock (T-309, `LineSvc::shift`): the account's
    /// times of day are its local times, so game time `t` is `t + shift` here.
    pub shift: f64,
    pub runs: [RunAccount; 2],
}

/// The account for the whole network, and the riders it was worked out from: one of each per
/// kind of day (T-260, `day_kind_at`), as the same trains carry a weekday's or a weekend's riders.
/// A kind with no riders reads the weekday's.
#[derive(Clone, Debug, Default)]
pub struct Boarding {
    /// busiest-hour factor per period (demand's, T-087)
    pub peak: [f64; PERIODS],
    /// per kind of day, per line id
    pub riders: [Vec<Option<LineRiders>>; DAY_KINDS],
    generation: [u64; DAY_KINDS],
    pub accounts: [Vec<Option<LineAccount>>; DAY_KINDS],
    /// lines worked out again since `take_redone`
    redone: usize,
    /// Riders a day changing onto each platform (line, run, stop in the run's order) from other
    /// platforms (T-554, `set_transfers`). Only the timing of drawn crowds reads them.
    pub xfer: HashMap<(u32, u8, u32), Vec<Feeder>>,
}

/// A platform riders change from, and riders a day making that change (T-554).
#[derive(Clone, Copy, Debug)]
pub struct Feeder {
    pub line: u32,
    pub run: u8,
    pub stop: u32,
    pub riders: f64,
}

/// Demand's transfer walk (kernel `Params::walk_detour`, `xfer_mps`): detour on the straight line
/// and walking speed, m/s.
const XFER_DETOUR: f64 = 1.3;
const XFER_MPS: f64 = 1.2;

struct Hash(u64);
impl Hash {
    fn new() -> Self {
        Hash(0xcbf29ce484222325)
    }
    fn add(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn f(&mut self, v: f64) {
        self.add(v.to_bits());
    }
}

impl Boarding {
    /// Replace the riders. `packed`: 5 busiest-hour factors, then per line: line id, stop count
    /// `n`, then for each period, boardings at each route node (run 0's stops, then run 1's, each
    /// in travel order: `2n`), then alightings likewise. The demand result's own layout.
    pub fn set_riders(&mut self, packed: &[f32]) {
        self.set_riders_day(packed, 0);
    }

    /// `set_riders` for one kind of day (T-260).
    pub fn set_riders_day(&mut self, packed: &[f32], day: usize) {
        let day = day.min(DAY_KINDS - 1);
        self.generation[day] += 1;
        self.riders[day].clear();
        if packed.len() < PERIODS {
            return;
        }
        for p in 0..PERIODS {
            self.peak[p] = packed[p] as f64;
        }
        let mut i = PERIODS;
        while i + 2 <= packed.len() {
            let (id, n) = (packed[i] as usize, packed[i + 1] as usize);
            i += 2;
            let len = PERIODS * 2 * n;
            if i + 2 * len > packed.len() {
                break;
            }
            let mut lr = LineRiders { n, board: vec![0.0; len], alight: vec![0.0; len] };
            for p in 0..PERIODS {
                for k in 0..2 * n {
                    lr.board[p * 2 * n + k] = packed[i + p * 4 * n + k].max(0.0) as f64;
                    lr.alight[p * 2 * n + k] = packed[i + p * 4 * n + 2 * n + k].max(0.0) as f64;
                }
            }
            i += 2 * len;
            let riders = &mut self.riders[day];
            if riders.len() <= id {
                riders.resize(id + 1, None);
            }
            riders[id] = Some(lr);
        }
    }

    /// Demand's transfers between platforms (T-554): per pair 7 numbers, the line, run and stop
    /// (in the run's travel order) riders leave a train at, the same for the platform they board
    /// at, and riders a day.
    pub fn set_transfers(&mut self, packed: &[f32]) {
        self.xfer.clear();
        for c in packed.chunks_exact(7) {
            let to = (c[3] as u32, c[4] as u8, c[5] as u32);
            let f = Feeder { line: c[0] as u32, run: c[1] as u8, stop: c[2] as u32, riders: c[6] as f64 };
            if f.riders > 0.0 && (f.line, f.run, f.stop) != to {
                self.xfer.entry(to).or_default().push(f);
            }
        }
    }

    pub fn clear(&mut self) {
        for d in 0..DAY_KINDS {
            self.generation[d] += 1;
            self.riders[d].clear();
            self.accounts[d].clear();
        }
        self.xfer.clear();
    }

    pub fn has_riders(&self) -> bool {
        !self.riders[0].is_empty()
    }

    /// The kind of day whose account game time `t` reads: its own, or the weekday's while it
    /// has no riders.
    pub fn kind_at(&self, t: f64) -> usize {
        self.kind(day_kind_at(t))
    }
    fn kind(&self, day: usize) -> usize {
        let day = day.min(DAY_KINDS - 1);
        if self.riders[day].is_empty() { 0 } else { day }
    }

    /// What a line's account depends on, hashed.
    fn key(&self, net: &Network, svc: &Service, l: usize, day: usize) -> u64 {
        let ls = &svc.lines[l];
        let mut h = Hash::new();
        h.add(self.generation[day]);
        h.add(ls.cars as u64);
        h.add(net.line_trains[l].is_some() as u64);
        h.f(net.line_turn[l] as f64);
        h.f(ls.shift);
        for t in net.line_tph[l] {
            h.f(t as f64);
        }
        // the patterns before recent changes (T-206): the account's day holds their departures
        for e in svc.eras_of(l) {
            h.f(e.until);
            for lev in 0..LEVELS {
                h.f(e.lat.tph[lev] as f64);
                h.f(e.lat.off[lev]);
            }
            h.f(e.lat.shift);
        }
        for r in 0..2 {
            for lev in 0..LEVELS {
                let p = &ls.runs[r].prof[lev];
                h.f(p.duration);
                for &v in p.dep.iter() {
                    h.f(v);
                }
            }
        }
        h.0
    }

    /// Bring every running line's account up to date with the service and the riders.
    pub fn ensure(&mut self, net: &Network, svc: &Service) {
        self.ensure_some(net, svc, usize::MAX);
    }

    /// `ensure`, working out at most `max_lines` lines again; true if more are out of date. The
    /// clock worker calls it a few lines at a time after new riders, so it never blocks long.
    pub fn ensure_some(&mut self, net: &Network, svc: &Service, max_lines: usize) -> bool {
        let nl = svc.lines.len();
        let mut done = 0;
        for day in 0..DAY_KINDS {
            self.accounts[day].resize(nl, None);
            self.accounts[day].truncate(nl);
            for l in 0..nl {
                let ls = &svc.lines[l];
                let lr = self.riders[day].get(l).and_then(|r| r.as_ref());
                let ok = ls.running && l < net.line_count() && lr.map_or(false, |r| r.n == ls.stops.len() && r.n >= 2);
                if !ok {
                    self.accounts[day][l] = None;
                    continue;
                }
                let key = self.key(net, svc, l, day);
                if self.accounts[day][l].as_ref().map_or(false, |a| a.key == key) {
                    continue;
                }
                if done == max_lines {
                    return true;
                }
                done += 1;
                let lr = lr.unwrap();
                // One day from the quiet of the night; a platform still waiting at its end means a
                // queue that carries round, so that direction is run again from a day earlier.
                let runs = [0, 1].map(|r| {
                    let a = simulate(net, svc, l, r, lr, &self.peak, false);
                    if a.leftover > 0.5 { simulate(net, svc, l, r, lr, &self.peak, true) } else { a }
                });
                self.accounts[day][l] = Some(LineAccount { key, shift: ls.shift, runs });
                self.redone += 1;
            }
        }
        false
    }

    /// Lines worked out again since the last call (checks and the debug readout).
    pub fn take_redone(&mut self) -> usize {
        std::mem::take(&mut self.redone)
    }

    /// The trips drawn in `[epoch, epoch + 3600)` (`Service::trips`, in its order) with their
    /// loads, and the platforms' crowds that hour. One flat array:
    ///
    /// `[trips, loads, queues, events, pulses, then per trip the index of its first load or -1 and
    /// the seconds from its departure to its last stop, then the loads, then per queue
    /// QUEUE_STRIDE numbers, then per event 2, then per pulse 2]`.
    ///
    /// A trip's loads are riders aboard on each stop-to-stop stretch, numbered as on run 0 (a
    /// run 1 trip's are reversed), so the renderer looks them up by the stretch it draws.
    /// A queue is one platform: one stop of one direction of one line: node, line, run, its
    /// offset along run 0 (m), riders arriving steadily a second at the middle of the hour,
    /// riders left behind by a full train this hour, the longest wait for a train leaving this
    /// hour (s), riders arriving this hour, first event, event count, first pulse, pulse count.
    /// Its events are the trains leaving it from the last one before the hour to the first one
    /// after: seconds from `epoch`, and riders left on the platform as it went. Its pulses are
    /// riders changing from other lines' trains (`pulses`): seconds from `epoch` and riders.
    /// Between two events, the crowd grows at the steady rate plus the pulses.
    pub fn hour(&mut self, net: &Network, svc: &Service, epoch: f64) -> Vec<f32> {
        self.ensure(net, svc);
        let game = epoch;
        let mut pulses: Vec<(f32, f32)> = vec![];
        // the hour's kind of day (T-260) in each line's local time (T-309): every trip and
        // platform of the line reads that day's account, at its local hour
        let shift = |l: usize| svc.lines.get(l).map_or(0.0, |ls| ls.shift);
        let trips = svc.trips(net, epoch, epoch + 3600.0);
        let mut offs: Vec<f32> = Vec::with_capacity(trips.len() * 2);
        let mut loads: Vec<f32> = vec![];
        for t in &trips {
            let day = self.kind_at(epoch + shift(t.line as usize));
            offs.push(self.trip_loads(t, day, &mut loads));
            offs.push(svc.lines[t.line as usize].runs[t.run as usize].prof[t.level as usize].duration as f32);
        }
        let mut queues: Vec<f32> = vec![];
        let mut events: Vec<f32> = vec![];
        let nl = self.accounts.iter().map(|a| a.len()).max().unwrap_or(0);
        for l in 0..nl {
            // the hour in the line's local time
            let epoch = epoch + shift(l);
            let day = self.kind_at(epoch);
            let Some(Some(acc)) = self.accounts[day].get(l) else { continue };
            let h0 = epoch.rem_euclid(DAY);
            let hour = ((h0 / 3600.0).floor() as usize).min(23);
            let ls = &svc.lines[l];
            for (r, ra) in acc.runs.iter().enumerate() {
                let m = ra.deps.len();
                if m == 0 {
                    continue;
                }
                let run = &ls.runs[r];
                for k in 0..ra.n {
                    let arr = &ra.arrivals[k];
                    if arr.day <= 0.0 {
                        continue;
                    }
                    let ord = &ra.order[k * m..(k + 1) * m];
                    let ts = &ra.order_t[k * m..(k + 1) * m];
                    let first = ts.partition_point(|&t| t < h0);
                    let end = ts.partition_point(|&t| t < h0 + 3600.0);
                    let start = events.len() / 2;
                    // the last train before the hour, the hour's, and the first after it
                    let at = |j: isize| -> (f64, usize) {
                        let jj = j.rem_euclid(m as isize) as usize;
                        let wrap = j.div_euclid(m as isize) as f64 * DAY;
                        (ts[jj] + wrap - h0, ord[jj] as usize)
                    };
                    for j in first as isize - 1..=end as isize {
                        let (t, i) = at(j);
                        events.push(t as f32);
                        events.push(ra.q_after[i * ra.n + k]);
                    }
                    // T-680: empty when the line's trains started, and no train before then
                    let open = ra.open - epoch;
                    if (events[start * 2] as f64) < open {
                        let after = (start..events.len() / 2).find(|&e| events[e * 2] as f64 >= open).unwrap_or(events.len() / 2);
                        events.splice(start * 2..after * 2, [open as f32, 0.0]);
                    }
                    let node = run.nodes[run.stop_idx[k] as usize];
                    let d0 = if r == 0 { ls.runs[0].stop_s[k] } else { ls.runs[0].stop_s[ra.n - 1 - k] };
                    let mid = epoch + 1800.0;
                    let p0 = pulses.len();
                    let carried = self.pulses(net, svc, game, (l, r, k), arr, node, &mut pulses);
                    // before the hour, only those still waiting when it starts
                    let since = events[start * 2];
                    let kept = p0 + pulses[p0..].partition_point(|p| p.0 <= since);
                    pulses.drain(p0..kept);
                    queues.extend_from_slice(&[
                        node as f32,
                        l as f32,
                        r as f32,
                        d0 as f32,
                        (arr.rate(mid) * (1.0 - carried)) as f32,
                        ra.left_h[k * 24 + hour],
                        ra.wait_h[k * 24 + hour],
                        (arr.cum(epoch + 3600.0) - arr.cum(epoch.max(ra.open).min(epoch + 3600.0))) as f32,
                        start as f32,
                        (events.len() / 2 - start) as f32,
                        p0 as f32,
                        (pulses.len() - p0) as f32,
                    ]);
                }
            }
        }
        let mut out = Vec::with_capacity(5 + offs.len() + loads.len() + queues.len() + events.len() + pulses.len() * 2);
        out.extend_from_slice(&[trips.len() as f32, loads.len() as f32, (queues.len() / QUEUE_STRIDE) as f32, (events.len() / 2) as f32, pulses.len() as f32]);
        out.extend_from_slice(&offs);
        out.extend_from_slice(&loads);
        out.extend_from_slice(&queues);
        out.extend_from_slice(&events);
        for (t, v) in pulses {
            out.extend([t, v]);
        }
        out
    }

    /// The riders changing onto platform `q` (line, run, stop) from other lines' trains (T-554),
    /// in the hour before game time `epoch` and the hour from it: each hour, a feeding
    /// platform's share of `q`'s boardings (demand's transfers a day over its boardings a day) of
    /// the riders arriving at `q` that hour, split over the trains leaving the feeding platform
    /// by the riders each brings in, reaching `q` a transfer walk after the train leaves.
    /// Pushed onto `out` in time order as (seconds from `epoch`, riders). Returns the share of
    /// the hour's arrivals they carry; the rest arrive steadily. A feeding platform with no
    /// train bringing riders in an hour, or under one rider a train, leaves its share steady.
    fn pulses(&self, net: &Network, svc: &Service, epoch: f64, q: (usize, usize, usize), arr: &Arrivals, node: u32, out: &mut Vec<(f32, f32)>) -> f64 {
        let (l, r, k) = q;
        let Some(fs) = self.xfer.get(&(l as u32, r as u8, k as u32)) else { return 0.0 };
        let Some(ls) = svc.lines.get(l) else { return 0.0 };
        let local = epoch + ls.shift;
        // shares of the platform's boardings, never more than all of them
        let per = 1.0 / arr.day.max(fs.iter().map(|f| f.riders).sum());
        let (x0, y0) = (net.node_x[node as usize], net.node_y[node as usize]);
        let start = out.len();
        let mut carried = 0.0;
        let mut trains: Vec<(f64, f64)> = vec![];
        for f in fs {
            let fl = f.line as usize;
            let Some(fls) = svc.lines.get(fl) else { continue };
            let fe = epoch + fls.shift;
            let Some(Some(acc)) = self.accounts[self.kind_at(fe)].get(fl) else { continue };
            let ra = &acc.runs[(f.run as usize).min(1)];
            let (m, n, kf) = (ra.deps.len(), ra.n, f.stop as usize);
            // nobody arrives at a first stop
            if m == 0 || kf == 0 || kf >= n {
                continue;
            }
            let run = &fls.runs[(f.run as usize).min(1)];
            let fnode = run.nodes[run.stop_idx[kf] as usize] as usize;
            let d = (net.node_x[fnode] - x0).hypot(net.node_y[fnode] - y0) / net.k_at(y0);
            let walk = d * XFER_DETOUR / XFER_MPS;
            let ts = &ra.order_t[kf * m..(kf + 1) * m];
            let ord = &ra.order[kf * m..(kf + 1) * m];
            let h0 = fe.rem_euclid(DAY);
            for a in [-3600.0, 0.0] {
                // the trains whose riders reach `q` in [a, a + 3600), weighed by riders aboard
                trains.clear();
                let (lo, hi) = (h0 + a - walk, h0 + a + 3600.0 - walk);
                for d in (lo / DAY).floor() as i64..=(hi / DAY).floor() as i64 {
                    let off = d as f64 * DAY;
                    let (i0, i1) = (ts.partition_point(|&t| t + off < lo), ts.partition_point(|&t| t + off < hi));
                    for j in i0..i1 {
                        let w = ra.load[ord[j] as usize * n + kf - 1] as f64;
                        if w > 0.0 {
                            trains.push((ts[j] + off - h0 + walk, w));
                        }
                    }
                }
                let wsum: f64 = trains.iter().map(|t| t.1).sum();
                let riders = f.riders * per * (arr.cum(local + a + 3600.0) - arr.cum(local + a));
                // under one rider a train would not show: kept steady, and the slice small
                if wsum <= 0.0 || riders < trains.len() as f64 {
                    continue;
                }
                out.extend(trains.iter().map(|&(t, w)| (t as f32, (riders * w / wsum) as f32)));
                if a == 0.0 {
                    carried += f.riders * per;
                }
            }
        }
        out[start..].sort_by(|a, b| a.0.total_cmp(&b.0));
        carried
    }

    /// Append a trip's loads (run 0's stretch order) and return where they start, or -1 when it
    /// has no account.
    fn trip_loads(&self, t: &Trip, day: usize, loads: &mut Vec<f32>) -> f32 {
        let Some(Some(acc)) = self.accounts[day].get(t.line as usize) else { return -1.0 };
        let ra = &acc.runs[t.run as usize];
        let Some(i) = ra.find(t.dep + acc.shift) else { return -1.0 };
        let n = ra.n;
        let at = loads.len();
        loads.resize(at + n - 1, 0.0);
        for k in 0..n - 1 {
            let s = if t.run == 0 { k } else { n - 2 - k };
            loads[at + s] = ra.load[i * n + k];
        }
        at as f32
    }

    /// A line's day in the account, for its panel (T-230): `[fullest train's riders as a share of
    /// crush, trains packed full, riders left behind by a full train, longest wait (s), spans, then
    /// per span the first and last time (s into the day) a full train leaves the stop where it is
    /// fullest]`. Full trains less than half an hour apart are one span. Empty without an account.
    pub fn line_summary(&self, svc: &Service, l: usize) -> Vec<f64> {
        self.line_summary_day(svc, l, 0)
    }

    /// `line_summary` on a kind of day (T-260).
    pub fn line_summary_day(&self, svc: &Service, l: usize, day: usize) -> Vec<f64> {
        let Some(Some(acc)) = self.accounts[self.kind(day)].get(l) else { return vec![] };
        let Some(ls) = svc.lines.get(l) else { return vec![] };
        let cap = CRUSH_PER_CAR * ls.cars.max(1) as f64;
        let (mut fullest, mut left, mut wait) = (0.0f64, 0.0f64, 0.0f64);
        let mut full: Vec<f64> = vec![];
        for ra in &acc.runs {
            let n = ra.n;
            for j in 0..ra.deps.len() {
                let (mut best, mut at) = (0.0f64, 0usize);
                for k in 0..n {
                    let v = ra.load[j * n + k] as f64;
                    if v > best {
                        best = v;
                        at = k;
                    }
                }
                fullest = fullest.max(best / cap);
                if best >= cap - 0.5 {
                    let off = ra.stop_dep[ra.levels[j] as usize].get(at).copied().unwrap_or(0.0);
                    full.push((ra.deps[j] + off).rem_euclid(DAY));
                }
            }
            left += ra.left_h.iter().map(|&v| v as f64).sum::<f64>();
            wait = wait.max(ra.wait_h.iter().fold(0.0f32, |a, &b| a.max(b)) as f64);
        }
        full.sort_by(f64::total_cmp);
        let mut spans: Vec<(f64, f64)> = vec![];
        for &t in &full {
            match spans.last_mut() {
                Some(s) if t - s.1 <= 1800.0 => s.1 = t,
                _ => spans.push((t, t)),
            }
        }
        let mut out = vec![fullest, full.len() as f64, left, wait, spans.len() as f64];
        for (a, b) in spans {
            out.extend([a, b]);
        }
        out
    }

    /// Today so far (T-293): every line's departures from midnight to game time `t`, read from the
    /// account of `t`'s kind of day (the account's day is periodic, so the hours after midnight
    /// are its own). One flat array: the line count, then per line with an account its id, stop
    /// count `n`, riders boarded, trains that left packed full, riders left behind by a full
    /// train, the fullest train's riders as a share of crush, the number of spans of full trains,
    /// per span its first and last time (s into the day), then boardings at each stop in the
    /// line's own stop order (both directions), then its trains in service at `t` and how many
    /// of them are packed full on their trip (`under_way`, T-472), then the trips counted, the
    /// number the full ones are out of (T-481), then for the period now on `t`'s kind of day the
    /// rider-seconds waited after full trains, riders arriving and rider-seconds aboard
    /// (`crowding`, T-491), and the first two times the km those riders ride (T-818).
    ///
    /// A stop's boardings since midnight telescope: the riders who have arrived by its last
    /// departure, less those still waiting after it, less the same at its last departure before
    /// midnight. Riders left behind are counted per departure as in `simulate` (who arrived since
    /// the train before and is still waiting).
    pub fn today(&self, svc: &Service, t: f64) -> Vec<f64> {
        self.window(svc, t, false)
    }

    /// Trains of a line on the move or turning at local time of day `tod`, and how many of them
    /// are packed full somewhere on their trip (T-472). A trip counts from its departure until the
    /// next one could leave the far end, so the count is about the line's trains in service.
    fn under_way(acc: &LineAccount, ls: &LineSvc, tod: f64) -> (f64, f64) {
        let cap = CRUSH_PER_CAR * ls.cars.max(1) as f64;
        let (mut on, mut packed) = (0.0, 0.0);
        for (r, ra) in acc.runs.iter().enumerate() {
            let n = ra.n;
            for j in 0..ra.deps.len() {
                let lv = ra.levels[j] as usize;
                let turn = ((ls.round_trip[lv] - ls.runs[0].prof[lv].duration - ls.runs[1].prof[lv].duration) / 2.0).max(0.0);
                if (tod - ra.deps[j]).rem_euclid(DAY) >= ls.runs[r].prof[lv].duration + turn {
                    continue;
                }
                on += 1.0;
                if ra.load[j * n..(j + 1) * n].iter().any(|&v| v as f64 >= cap - 0.5) {
                    packed += 1.0;
                }
            }
        }
        (on, packed)
    }

    /// A line's day by period (T-477, T-491), on a kind of day: per period `PERIOD_STRIDE`
    /// numbers (`crowding`), then per period the rider-seconds waited after full trains at each
    /// stop, both directions, in the line's stop order (T-818: where they wait). Empty without an
    /// account.
    pub fn line_periods(&self, svc: &Service, l: usize, day: usize) -> Vec<f64> {
        let Some(Some(acc)) = self.accounts[self.kind(day)].get(l) else { return vec![] };
        let Some(ls) = svc.lines.get(l) else { return vec![] };
        let mut out = Self::crowding(acc, ls).concat();
        let n = acc.runs[0].n;
        let mut at = vec![0.0f64; PERIODS * n];
        for (r, ra) in acc.runs.iter().enumerate() {
            let m = ra.deps.len();
            if ra.n != n {
                continue;
            }
            for k in 0..n {
                let ts = &ra.order_t[k * m..(k + 1) * m];
                let ord = &ra.order[k * m..(k + 1) * m];
                let s = if r == 0 { k } else { n - 1 - k };
                for i in 0..m {
                    let q = ra.q_after[ord[i] as usize * n + k] as f64;
                    if q > 0.0 {
                        let gap = if i + 1 < m { ts[i + 1] - ts[i] } else { ts[0] + DAY - ts[i] };
                        at[period_at(ts[i]) * n + s] += q * gap.max(0.0);
                    }
                }
            }
        }
        out.extend(at);
        out
    }

    /// Per period: trains leaving in it, those packed full somewhere on their trip, riders
    /// arriving at the line's platforms, riders a full train left behind, rider-seconds spent
    /// waiting for a later train after a full one (each left behind waits the gap to the next
    /// departure), and rider-seconds aboard (T-491: the ride the wait is weighed against); then the
    /// waiting rider-seconds and the riders arriving each times the km they ride (`ride_km`), so the
    /// app can weigh riders by the fare they pay (T-818).
    fn crowding(acc: &LineAccount, ls: &LineSvc) -> [[f64; PERIOD_STRIDE]; PERIODS] {
        let cap = CRUSH_PER_CAR * ls.cars.max(1) as f64;
        let mut out = [[0.0f64; PERIOD_STRIDE]; PERIODS];
        for ra in &acc.runs {
            let (n, m) = (ra.n, ra.deps.len());
            if m == 0 {
                continue;
            }
            for j in 0..m {
                let p = &mut out[period_at(ra.deps[j])];
                p[0] += 1.0;
                let load = &ra.load[j * n..(j + 1) * n];
                if load.iter().any(|&v| v as f64 >= cap - 0.5) {
                    p[1] += 1.0;
                }
                let sd = &ra.stop_dep[ra.levels[j] as usize];
                for k in 0..n.saturating_sub(1).min(sd.len().saturating_sub(1)) {
                    p[5] += load[k] as f64 * (sd[k + 1] - sd[k]).max(0.0);
                }
            }
            let km = |p: usize, k: usize| ra.ride_km.get(p * n + k).copied().unwrap_or(0.0);
            for k in 0..n {
                for (p, o) in out.iter_mut().enumerate() {
                    o[2] += ra.arrivals[k].per[p];
                    o[7] += ra.arrivals[k].per[p] * km(p, k);
                }
                for h in 0..24 {
                    out[period_at(h as f64 * 3600.0)][3] += ra.left_h[k * 24 + h] as f64;
                }
                let ts = &ra.order_t[k * m..(k + 1) * m];
                let ord = &ra.order[k * m..(k + 1) * m];
                for i in 0..m {
                    let q = ra.q_after[ord[i] as usize * n + k] as f64;
                    if q > 0.0 {
                        let gap = if i + 1 < m { ts[i + 1] - ts[i] } else { ts[0] + DAY - ts[i] };
                        let p = period_at(ts[i]);
                        out[p][4] += q * gap.max(0.0);
                        out[p][6] += q * gap.max(0.0) * km(p, k);
                    }
                }
            }
        }
        out
    }

    /// The past 24 hours to game time `t` (T-446), in `today`'s layout: the rest of yesterday from
    /// the time of day now, read from yesterday's kind of day, then today so far. Span times are s
    /// from today's local midnight, so yesterday's are negative.
    pub fn past_day(&self, svc: &Service, t: f64) -> Vec<f64> {
        self.window(svc, t, true)
    }

    fn window(&self, svc: &Service, t: f64, past: bool) -> Vec<f64> {
        let mut out = vec![0.0];
        let mut count = 0;
        let nl = self.accounts.iter().map(|a| a.len()).max().unwrap_or(0);
        for l in 0..nl {
            let Some(ls) = svc.lines.get(l) else { continue };
            // the line's own day and time of day (T-309: its local time)
            let t = t + ls.shift;
            let tod = t.rem_euclid(DAY);
            let Some(Some(acc)) = self.accounts[self.kind_at(t)].get(l) else { continue };
            let n = acc.runs[0].n;
            let mut w = Tally { stops: vec![0.0f64; n], ..Default::default() };
            if past {
                if let Some(Some(ya)) = self.accounts[self.kind_at(t - DAY)].get(l) {
                    if ya.runs[0].n == n {
                        self.tally(ya, ls, tod, DAY, -DAY, t - tod - DAY, &mut w);
                    }
                }
            }
            self.tally(acc, ls, f64::NEG_INFINITY, tod, 0.0, t - tod, &mut w);
            let Tally { stops, boarded, left, fullest, mut full, trains } = w;
            full.sort_by(f64::total_cmp);
            let mut spans: Vec<(f64, f64)> = vec![];
            for &s in &full {
                match spans.last_mut() {
                    Some(p) if s - p.1 <= 1800.0 => p.1 = s,
                    _ => spans.push((s, s)),
                }
            }
            out.extend([l as f64, n as f64, boarded, full.len() as f64, left, fullest, spans.len() as f64]);
            for (a, b) in spans {
                out.extend([a, b]);
            }
            out.extend(stops);
            let (on, packed) = Self::under_way(acc, ls, tod);
            out.extend([on, packed, trains]);
            let c = Self::crowding(acc, ls)[period_at(tod)];
            out.extend([c[4], c[2], c[5], c[6], c[7]]);
            count += 1;
        }
        out[0] = count as f64;
        out
    }

    /// Add one account's departures with time of day in (`from`, `to`] to `w`; full trains' times
    /// are shifted by `off`. A stop's boardings telescope: the riders who have arrived by its last
    /// departure in the window, less those still waiting after it, less the same at the departure
    /// before the window (the day's last, a day earlier, when `from` is before the first).
    /// `day0` is the local time of the window's midnight.
    fn tally(&self, acc: &LineAccount, ls: &LineSvc, from: f64, to: f64, off: f64, day0: f64, w: &mut Tally) {
        let cap = CRUSH_PER_CAR * ls.cars.max(1) as f64;
        let n = w.stops.len();
        for (r, ra) in acc.runs.iter().enumerate() {
            let m = ra.deps.len();
            if m == 0 || ra.n != n {
                continue;
            }
            // T-680: nothing ran before the line's trains started
            let open = ra.open - day0;
            let from = if open > from { open - 1e-6 } else { from };
            for k in 0..n {
                let arr = &ra.arrivals[k];
                let ts = &ra.order_t[k * m..(k + 1) * m];
                let ord = &ra.order[k * m..(k + 1) * m];
                let q = |j: usize| ra.q_after[ord[j] as usize * n + k] as f64;
                let lo = ts.partition_point(|&s| s <= from);
                let done = ts.partition_point(|&s| s <= to);
                if done <= lo {
                    continue;
                }
                let (t_before, q_before) = if lo > 0 { (ts[lo - 1], q(lo - 1)) } else { (ts[m - 1] - DAY, q(m - 1)) };
                let last = done - 1;
                let mut b = (arr.cum(ts[last]) - q(last)) - (arr.cum(t_before) - q_before);
                let (mut prev, mut q_prev) = (t_before, q_before);
                for j in lo..done {
                    let mut a = (arr.cum(ts[j]) - arr.cum(prev)).max(0.0);
                    if prev < open && open <= ts[j] {
                        // whoever was waiting then never was: the platform starts empty
                        b -= q_prev + arr.cum(open) - arr.cum(prev);
                        a = (arr.cum(ts[j]) - arr.cum(open)).max(0.0);
                    }
                    w.left += a.min(q(j));
                    (prev, q_prev) = (ts[j], q(j));
                }
                let b = b.max(0.0);
                w.stops[if r == 0 { k } else { n - 1 - k }] += b;
                w.boarded += b;
            }
            // trains: the fullest stretch each has reached in the window (a trip from before
            // midnight counts at the stops it reaches after it)
            for j in 0..m {
                let (mut best, mut at_t, mut seen) = (0.0f64, f64::NAN, false);
                for k in 0..n {
                    let o = ra.stop_dep[ra.levels[j] as usize].get(k).copied().unwrap_or(0.0);
                    let s = (ra.deps[j] + o).rem_euclid(DAY);
                    if s > to || s <= from {
                        continue;
                    }
                    seen = true;
                    let v = ra.load[j * n + k] as f64;
                    if v > best {
                        best = v;
                        at_t = s;
                    }
                }
                w.fullest = w.fullest.max(best / cap);
                if seen {
                    w.trains += 1.0;
                }
                if best >= cap - 0.5 && !at_t.is_nan() {
                    w.full.push(at_t + off);
                }
            }
        }
    }

    /// Riders aboard a trip leaving each stop, in its own stop order (checks and inspectors).
    pub fn trip_account(&self, line: u32, run: usize, dep: f64) -> Option<Vec<f32>> {
        // the line's local time (T-309): both kinds of day's accounts carry the same shift
        let shift = self.accounts[0].get(line as usize).and_then(|a| a.as_ref()).map_or(0.0, |a| a.shift);
        let dep = dep + shift;
        let ra = &self.accounts[self.kind_at(dep)].get(line as usize)?.as_ref()?.runs[run];
        let i = ra.find(dep)?;
        Some(ra.load[i * ra.n..(i + 1) * ra.n].to_vec())
    }
}

pub const QUEUE_STRIDE: usize = 12;
/// Numbers per period in `Boarding::line_periods` (T-491).
pub const PERIOD_STRIDE: usize = 8;

impl RunAccount {
    /// The account's index of a trip leaving its first stop at game time `dep` (any day).
    pub fn find(&self, dep: f64) -> Option<usize> {
        let rel = dep.rem_euclid(DAY);
        let j = self.deps.partition_point(|&d| d < rel);
        let mut best: Option<(usize, f64)> = None;
        for c in [j.wrapping_sub(1), j] {
            if let Some(&d) = self.deps.get(c) {
                let e = (d - rel).abs();
                if e < 0.5 && best.map_or(true, |b| e < b.1) {
                    best = Some((c, e));
                }
            }
        }
        best.map(|b| b.0)
    }
}

/// Alighting share per period and stop: demand's alightings over the riders it has arriving.
fn alight_shares(lr: &LineRiders, r: usize) -> Vec<f64> {
    let n = lr.n;
    let mut out = vec![0.0; PERIODS * n];
    let mut day_in = vec![0.0; n];
    let mut day_off = vec![0.0; n];
    for p in 0..PERIODS {
        let base = (p * 2 + r) * n;
        let mut on = 0.0;
        for k in 0..n {
            let a = lr.alight[base + k];
            day_in[k] += on;
            day_off[k] += a;
            out[p * n + k] = if on > 1e-6 { (a / on).clamp(0.0, 1.0) } else { -1.0 };
            on = (on + lr.board[base + k] - a).max(0.0);
        }
    }
    for p in 0..PERIODS {
        for k in 0..n {
            if out[p * n + k] < 0.0 {
                out[p * n + k] = if day_in[k] > 1e-6 { (day_off[k] / day_in[k]).clamp(0.0, 1.0) } else { 0.0 };
            }
        }
        out[p * n + n - 1] = 1.0;
    }
    out
}

/// Per period and stop (`p * n + k`), the km a rider boarding there rides on average (T-818): each
/// stop lets off its share of whoever is aboard, as `simulate` does, so a boarder rides on past
/// stop j with the chance `1 - alpha` of every stop before it.
fn ride_km(alpha: &[f64], stop_s: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; PERIODS * n];
    if stop_s.len() < n {
        return out;
    }
    for p in 0..PERIODS {
        for k in (0..n.saturating_sub(1)).rev() {
            let on = if k + 2 < n { (1.0 - alpha[p * n + k + 1]) * out[p * n + k + 1] } else { 0.0 };
            out[p * n + k] = (stop_s[k + 1] - stop_s[k]).max(0.0) / 1000.0 + on;
        }
    }
    out
}

/// The account's day starts at this hour, when platforms are usually empty.
pub const START_HOUR: f64 = 3.0;

/// One direction of one line for a day of departures from `START_HOUR`, from empty platforms;
/// with `warm`, from a day earlier, keeping the second day. `leftover` says whether anyone was
/// still waiting when the day ended (then the warm run is the right one).
pub fn simulate(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool) -> RunAccount {
    let ls = &svc.lines[l];
    let n = lr.n;
    let cap = CRUSH_PER_CAR * ls.cars.max(1) as f64;
    let t0 = svc.account_start(l, START_HOUR);
    let from = if warm { t0 - DAY } else { t0 };
    let mut trips: Vec<Trip> = vec![];
    svc.run_deps(net, l, r, from, t0 + DAY, &mut trips);
    // T-680: a change from a pattern with no trains is when the line's trains started; nobody
    // waited for them before it
    let open = svc.eras_of(l).iter().rev().find(|e| e.lat.tph.iter().all(|&t| t <= 0.0)).map(|e| e.until).filter(|&u| u > t0);
    let open = open.map_or(f64::NEG_INFINITY, |u| u + ls.shift);
    // T-309: from here on, times are the line's local time (its periods, hours and days)
    let t0 = t0 + ls.shift;
    for t in &mut trips {
        t.dep += ls.shift;
    }
    trips.sort_by(|a, b| a.dep.total_cmp(&b.dep));
    let m = trips.len();
    let stop_dep: [Vec<f64>; LEVELS] = std::array::from_fn(|lev| ls.runs[r].prof[lev].dep.clone());
    let arrivals: Vec<Arrivals> = (0..n)
        .map(|k| Arrivals::new(std::array::from_fn(|p| lr.board[(p * 2 + r) * n + k]), peak))
        .collect();
    let alpha = alight_shares(lr, r);
    let ride_km = ride_km(&alpha, &ls.runs[r].stop_s, n);
    let time = |i: usize, k: usize| trips[i].dep + stop_dep[trips[i].level as usize].get(k).copied().unwrap_or(0.0);
    // the last day's trips are the account, by time of day
    let kept = |i: usize| trips[i].dep >= t0;
    let mut keep: Vec<usize> = (0..m).filter(|&i| kept(i)).collect();
    keep.sort_by(|&a, &b| trips[a].dep.rem_euclid(DAY).total_cmp(&trips[b].dep.rem_euclid(DAY)));
    let mut slot = vec![u32::MAX; m];
    for (j, &i) in keep.iter().enumerate() {
        slot[i] = j as u32;
    }
    let mk = keep.len();
    let mut acc = RunAccount {
        n,
        deps: keep.iter().map(|&i| trips[i].dep.rem_euclid(DAY)).collect(),
        levels: keep.iter().map(|&i| trips[i].level).collect(),
        stop_dep: stop_dep.clone(),
        load: vec![0.0; mk * n],
        q_after: vec![0.0; mk * n],
        order: vec![0; mk * n],
        order_t: vec![0.0; mk * n],
        left_h: vec![0.0; n * 24],
        wait_h: vec![0.0; n * 24],
        arrivals: vec![],
        arrived: 0.0,
        boarded: 0.0,
        alighted: 0.0,
        leftover: 0.0,
        open,
        ride_km,
    };
    let mut carry = vec![0.0f64; m];
    let mut order: Vec<usize> = (0..m).collect();
    let mut hist: Vec<(f64, f64)> = Vec::with_capacity(m);
    for k in 0..n {
        order.sort_by(|&a, &b| time(a, k).total_cmp(&time(b, k)));
        let arr = &arrivals[k];
        let mut q = 0.0f64;
        let mut c_prev = f64::NAN;
        let mut t_prev = f64::NEG_INFINITY;
        hist.clear();
        // Without the warm-up day the first train takes whoever arrived since the day's last
        // train, a day before (the platform empty then: `leftover` checks it).
        if !warm && m > 0 {
            t_prev = time(order[m - 1], k) - DAY;
            c_prev = arr.cum(t_prev);
            hist.push((t_prev, c_prev));
        }
        let mut head = 0usize;
        for &i in &order {
            let t = time(i, k);
            if open > t_prev && open <= t {
                q = 0.0;
                c_prev = arr.cum(open);
                hist.clear();
                hist.push((open, c_prev));
                head = 0;
            }
            let c_now = arr.cum(t);
            let a = if c_prev.is_nan() { 0.0 } else { (c_now - c_prev).max(0.0) };
            let q_before = q + a;
            // head of the queue: the first rider not yet on a train, and when they arrived
            let first_rider = c_now - q_before;
            while head + 1 < hist.len() && hist[head + 1].1 <= first_rider {
                head += 1;
            }
            let since = match hist.get(head) {
                Some(&(t0, c0)) => {
                    let (t1, c1) = hist.get(head + 1).copied().unwrap_or((t, c_now));
                    if c1 > c0 && first_rider >= c0 { t0 + (first_rider - c0).min(c1 - c0) / (c1 - c0) * (t1 - t0) } else { t0 }
                }
                None => t,
            };
            let p = period_at(t);
            let on = carry[i];
            let off = on * alpha[p * n + k];
            let on = on - off;
            let room = (cap - on).max(0.0);
            let b = q_before.min(room);
            let load = on + b;
            let q_after = q_before - b;
            carry[i] = if k + 1 < n { load } else { 0.0 };
            if kept(i) {
                let j = slot[i] as usize;
                acc.load[j * n + k] = if k + 1 < n { load as f32 } else { 0.0 };
                acc.q_after[j * n + k] = q_after as f32;
                let h = ((t.rem_euclid(DAY) / 3600.0).floor() as usize).min(23);
                acc.left_h[k * 24 + h] += a.min(q_after) as f32;
                if b > 0.0 || q_before > 0.0 {
                    let w = (t - since).max(0.0) as f32;
                    if w > acc.wait_h[k * 24 + h] {
                        acc.wait_h[k * 24 + h] = w;
                    }
                }
                acc.arrived += a;
                acc.boarded += b;
                acc.alighted += off + if k + 1 < n { 0.0 } else { load };
            }
            hist.push((t, c_now));
            c_prev = c_now;
            t_prev = t;
            q = q_after;
        }
        acc.leftover = acc.leftover.max(q);
        // this stop's departures in a day, by time of day
        let mut day: Vec<(f64, u32)> = order.iter().filter(|&&i| kept(i)).map(|&i| (time(i, k).rem_euclid(DAY), slot[i])).collect();
        day.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (j, (t, s)) in day.into_iter().enumerate() {
            acc.order[k * mk + j] = s;
            acc.order_t[k * mk + j] = t;
        }
    }
    acc.arrivals = arrivals;
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::tests::{edge, line, node, station, world};

    #[test]
    fn ride_km_follows_the_alighting_shares() {
        // stops 0, 10, 20 and 120 km; half of whoever is aboard gets off at each middle stop
        let n = 4;
        let alpha: Vec<f64> = (0..PERIODS).flat_map(|_| [0.0, 0.5, 0.5, 1.0]).collect();
        let km = ride_km(&alpha, &[0.0, 10_000.0, 20_000.0, 120_000.0], n);
        let want = [10.0 + 0.5 * (10.0 + 0.5 * 100.0), 10.0 + 0.5 * 100.0, 100.0, 0.0];
        for p in 0..PERIODS {
            for k in 0..n {
                assert!((km[p * n + k] - want[k]).abs() < 1e-9, "{km:?}");
            }
        }
    }

    #[test]
    fn shapes_keep_the_period_total_and_reach_the_busiest_hour() {
        for s in [Shape::Tent, Shape::Falling, Shape::Valley] {
            for hours in [4.0, 6.0] {
                let c = shape_strength(s, hours, 1.4);
                assert!((shape_cum(s, c, 1.0) - 1.0).abs() < 1e-12);
                // the busiest whole hour, against the mean hour
                let best = (0..(hours as usize))
                    .map(|h| (shape_cum(s, c, (h as f64 + 1.0) / hours) - shape_cum(s, c, h as f64 / hours)) * hours)
                    .fold(0.0f64, f64::max);
                assert!((best - 1.4).abs() < 1e-9, "{s:?} {hours}: {best}");
            }
        }
        // the rate is the derivative of the cumulative curve
        let a = Arrivals::new([1000.0, 600.0, 900.0, 300.0, 50.0], &[1.4, 1.15, 1.4, 1.3, 1.5]);
        for t in [7.2 * 3600.0, 13.0 * 3600.0, 21.5 * 3600.0, 2.0 * 3600.0, 5.9 * 3600.0] {
            let d = (a.cum(t + 1.0) - a.cum(t - 1.0)) / 2.0;
            assert!((d - a.rate(t)).abs() < 1e-6, "{t}: {d} {}", a.rate(t));
        }
        assert!((a.cum(DAY) - a.cum(0.0) - 2850.0).abs() < 1e-9);
    }

    /// A - B - C, a line every 6 minutes at high demand.
    fn setup(tph: f32) -> (crate::track::world::TrackWorld, u32) {
        let mut w = world();
        let a = node(&mut w, 0.0, 0.0, 0);
        let b = node(&mut w, 3000.0, 0.0, 0);
        let c = node(&mut w, 6000.0, 0.0, 0);
        edge(&mut w, a, b, &[], 0).unwrap();
        edge(&mut w, b, c, &[], 0).unwrap();
        for n in [a, b, c] {
            station(&mut w, n, 200);
        }
        let l = line(&mut w, &[a, b, c], [tph, tph / 2.0, tph / 4.0]);
        (w, l)
    }

    /// Riders boarding at A for C (run 0) and at C for A (run 1), `per_hour` an hour in every
    /// period.
    fn riders(l: u32, per_hour: [f32; PERIODS]) -> Vec<f32> {
        let mut v = vec![1.0; PERIODS]; // flat: the peak factor 1
        v.extend([l as f32, 3.0]);
        for p in 0..PERIODS {
            let (h0, h1) = PERIOD_HOURS[p];
            let x = per_hour[p] * (h1 - h0) as f32;
            v.extend([x, 0.0, 0.0, x, 0.0, 0.0]); // board: run 0 at A, run 1 at C
            v.extend([0.0, 0.0, x, 0.0, 0.0, x]); // alight: run 0 at C, run 1 at A
        }
        v
    }

    #[test]
    fn a_weekend_hour_reads_the_weekend_account() {
        // T-260: game day 1 is a Monday, days 6 and 7 the weekend
        assert_eq!(day_kind_at(86400.0 + 7.0 * 3600.0), 0);
        assert_eq!(day_kind_at(5.0 * 86400.0 + 23.0 * 3600.0), 0);
        assert_eq!(day_kind_at(6.0 * 86400.0), 1);
        assert_eq!(day_kind_at(7.0 * 86400.0 + 3600.0), 1);
        assert_eq!(day_kind_at(8.0 * 86400.0), 0);
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders_day(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]), 0);
        // before the weekend's riders, a weekend hour reads the weekday's
        let sat = 6.0 * 86400.0 + 8.0 * 3600.0;
        assert_eq!(b.kind_at(sat), 0);
        b.set_riders_day(&riders(l, [100.0, 300.0, 200.0, 200.0, 50.0]), 1);
        assert_eq!(b.kind_at(sat), 1);
        let mon = 8.0 * 86400.0 + 8.0 * 3600.0;
        let (h_sat, h_mon) = (b.hour(&w.net, &w.svc, sat), b.hour(&w.net, &w.svc, mon));
        let arriving = |h: &[f32]| {
            let (nt, nl, nq) = (h[0] as usize, h[1] as usize, h[2] as usize);
            let q0 = 5 + 2 * nt + nl;
            (0..nq).map(|k| h[q0 + k * QUEUE_STRIDE + 7]).sum::<f32>()
        };
        // 8-9 in the morning peak: 500 an hour at each end on a weekday, 100 on Saturday
        assert!((arriving(&h_mon) / arriving(&h_sat) - 5.0).abs() < 0.01, "{} {}", arriving(&h_mon), arriving(&h_sat));
        assert_ne!(b.line_summary_day(&w.svc, l as usize, 1)[0], b.line_summary_day(&w.svc, l as usize, 0)[0]);
    }

    #[test]
    fn a_line_with_room_carries_demands_load_and_leaves_no_one() {
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
        b.ensure(&w.net, &w.svc);
        let acc = b.accounts[0][l as usize].as_ref().unwrap();
        for ra in &acc.runs {
            // every rider in the day boards and gets off
            assert!((ra.arrived - ra.boarded).abs() < 1e-6 * ra.arrived.max(1.0), "{} {}", ra.arrived, ra.boarded);
            assert!((ra.boarded - ra.alighted).abs() < 1e-6 * ra.boarded.max(1.0));
            assert!(ra.q_after.iter().all(|&q| q < 1e-3));
            // the day's arrivals are demand's
            assert!((ra.arrived - (500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0)).abs() < 1e-6);
        }
        // a morning-peak train carries a headway's riders: 500 an hour, 10 trains an hour
        let ra = &acc.runs[0];
        let i = ra.find(8.0 * 3600.0 + 60.0).or_else(|| ra.deps.iter().position(|&d| d > 8.0 * 3600.0)).unwrap();
        assert!((ra.load[i * 3] - 50.0).abs() < 1.0, "{}", ra.load[i * 3]);
        assert_eq!(ra.load[i * 3 + 1], ra.load[i * 3]);
        assert_eq!(ra.load[i * 3 + 2], 0.0);
    }

    #[test]
    fn today_counts_up_to_the_days_riders() {
        // T-293: boardings since midnight grow through the day to the day's riders
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
        b.ensure(&w.net, &w.svc);
        let day = 500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0;
        let at = |h: f64| {
            let v = b.today(&w.svc, 2.0 * DAY + h * 3600.0);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[1], l as f64);
            v[3]
        };
        let (night, noon, end) = (at(0.0), at(12.0), at(23.999));
        // at midnight: at most the trains leaving at 00:00 (the night's 50 an hour at each end)
        assert!(night < 150.0, "{night}");
        assert!(noon > 0.3 * 2.0 * day && noon < 0.7 * 2.0 * day, "{noon}");
        // both directions, everyone who arrived by the last train has boarded
        assert!((end - 2.0 * day).abs() < 0.02 * day, "{end} {}", 2.0 * day);
        // stops: A and C board, B nobody
        let v = b.today(&w.svc, 2.0 * DAY + 23.999 * 3600.0);
        let spans = v[7] as usize;
        let stops = &v[8 + 2 * spans..8 + 2 * spans + 3];
        assert!(stops[1] < 1e-6 && (stops[0] - day).abs() < 0.02 * day, "{stops:?}");
    }

    #[test]
    fn past_day_holds_a_whole_day_at_any_hour() {
        // T-446: the past 24 hours on weekdays are a whole weekday's riders, whatever the hour
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
        b.ensure(&w.net, &w.svc);
        let day = 2.0 * (500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0);
        for h in [0.0, 0.5, 7.25, 12.0, 18.9, 23.999] {
            let v = b.past_day(&w.svc, 2.0 * DAY + h * 3600.0);
            assert_eq!(v[0], 1.0);
            assert!((v[3] - day).abs() < 0.02 * day, "{h}: {} {day}", v[3]);
        }
    }

    #[test]
    fn period_totals_match_demand_with_the_busiest_hour_shapes() {
        let (w, l) = setup(12.0);
        let mut b = Boarding::default();
        let per_hour = [900.0, 400.0, 1000.0, 300.0, 40.0];
        let mut packed = riders(l, per_hour);
        packed[..PERIODS].copy_from_slice(&[1.4, 1.15, 1.4, 1.3, 1.5]);
        b.set_riders(&packed);
        b.ensure(&w.net, &w.svc);
        let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
        let mut by_period = [0.0f64; PERIODS];
        for (i, &d) in ra.deps.iter().enumerate() {
            by_period[period_at(d)] += ra.load[i * 3] as f64;
        }
        for p in 0..PERIODS {
            let (h0, h1) = PERIOD_HOURS[p];
            let want = per_hour[p] as f64 * (h1 - h0);
            // riders arriving in the last headway of a period board the next period's first train
            let hw_riders = per_hour[p].max(per_hour[(p + PERIODS - 1) % PERIODS]) as f64 * 3600.0 / (12.0 / 4.0) / 3600.0;
            assert!((by_period[p] - want).abs() <= hw_riders, "period {p}: {} against {want}", by_period[p]);
        }
        // and the morning peak's busiest train carries about 1.4 times the mean train
        let mean = per_hour[0] as f64 / 12.0;
        let peak = ra.deps.iter().enumerate().filter(|(_, &d)| d >= 6.0 * 3600.0 && d < 10.0 * 3600.0).map(|(i, _)| ra.load[i * 3] as f64).fold(0.0, f64::max);
        assert!(peak > 1.45 * mean && peak < 1.85 * mean, "{peak} against a mean of {mean}");
    }

    #[test]
    fn an_overloaded_line_leaves_people_behind_and_the_queue_clears() {
        // 10 trains an hour of crush 1,600 (10 cars) = 16,000 an hour; 20,000 an hour arrive in
        // the morning peak
        let (w, l) = setup(10.0);
        let cap = CRUSH_PER_CAR * w.svc.lines[l as usize].cars as f64;
        assert_eq!(cap, 1600.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
        b.ensure(&w.net, &w.svc);
        let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
        let m = ra.deps.len();
        // the queue at A, train by train through the day
        let q: Vec<(f64, f32, f32)> = (0..m).map(|j| (ra.order_t[j], ra.q_after[ra.order[j] as usize * 3], ra.load[ra.order[j] as usize * 3])).collect();
        let at = |h: f64| q.iter().filter(|e| e.0 < h * 3600.0).last().unwrap().1;
        assert!(at(7.0) > 0.0 && at(9.0) > at(7.0) && at(10.0) > at(9.0), "{} {} {}", at(7.0), at(9.0), at(10.0));
        // 4 hours of 4,000 an hour too many: 16,000 left at 10:00, give or take a headway
        assert!((at(10.0) - 16000.0).abs() < 1700.0, "{}", at(10.0));
        // every train in the peak leaves full
        assert!(q.iter().filter(|e| e.0 > 6.5 * 3600.0 && e.0 < 10.0 * 3600.0).all(|e| (e.2 as f64 - cap).abs() < 1e-3));
        // midday trains (5 an hour, 8,000 of room against 2,000 arriving) clear it within about
        // 2.7 hours
        assert!(at(12.0) > 0.0 && at(13.5) < 1.0, "{} {}", at(12.0), at(13.5));
        // nobody is lost: in = on + still waiting at the end of the day (none)
        assert!((ra.arrived - ra.boarded).abs() < 1e-6 * ra.arrived);
        // riders left behind and long waits are reported for the peak's hours
        assert!(ra.left_h[8] > 1000.0);
        assert!(ra.wait_h[9] > 1800.0, "{}", ra.wait_h[9]);
        // once it has cleared, the longest wait is a headway (5 trains an hour at midday)
        assert!(ra.wait_h[15] < 730.0, "{}", ra.wait_h[15]);
    }

    #[test]
    fn the_line_summary_names_the_full_trains_and_when() {
        // T-230: 20,000 an hour against 16,000 in the morning peak only
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
        b.ensure(&w.net, &w.svc);
        let s = b.line_summary(&w.svc, l as usize);
        assert!((s[0] - 1.0).abs() < 1e-6, "fullest {}", s[0]);
        assert!(s[1] >= 60.0, "{} full trains", s[1]); // 10 an hour each way through the peak
        assert!(s[2] > 10000.0 && s[3] > 1800.0);
        // one span: from the peak's start until midday trains have cleared the queue (by 13:30)
        let spans: Vec<(f64, f64)> = (0..s[4] as usize).map(|k| (s[5 + 2 * k], s[6 + 2 * k])).collect();
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!(spans[0].0 >= 6.0 * 3600.0 && spans[0].0 < 7.0 * 3600.0 && spans[0].1 > 10.0 * 3600.0 && spans[0].1 < 13.5 * 3600.0, "{spans:?}");
        // a line with room: no full trains, nobody left
        let (w2, l2) = setup(10.0);
        let mut b2 = Boarding::default();
        b2.set_riders(&riders(l2, [500.0, 200.0, 500.0, 200.0, 50.0]));
        b2.ensure(&w2.net, &w2.svc);
        let s2 = b2.line_summary(&w2.svc, l2 as usize);
        assert!(s2[0] < 0.1 && s2[1] == 0.0 && s2[2] < 1e-3 && s2[4] == 0.0, "{s2:?}");
        // T-477, T-491: by period, the full trains, riders left behind and their wait fall in the
        // morning peak, spilling into midday while the queue clears; none at night
        let v = b.line_periods(&w.svc, l as usize, 0);
        let p = |p: usize, i: usize| v[p * PERIOD_STRIDE + i];
        assert!(p(0, 1) >= 60.0 && p(0, 3) > 10000.0, "{v:?}");
        assert!(p(4, 1) == 0.0 && p(4, 0) > 0.0 && p(4, 4) == 0.0, "{v:?}");
        // the average morning rider waits minutes for a train with room; everyone rides
        assert!(p(0, 4) / p(0, 2) > 300.0 && p(0, 5) > 0.0, "{v:?}");
        assert!(p(1, 4) > 0.0, "{v:?}");
        let v2 = b2.line_periods(&w2.svc, l2 as usize, 0);
        assert!(v2[..PERIODS * PERIOD_STRIDE].chunks(PERIOD_STRIDE).all(|c| c[1] == 0.0 && c[3] < 1e-3 && c[4] < 1e-3 && c[5] > 0.0), "{v2:?}");
        // T-818: every rider rides some km, so the km-weighted sums are positive and no larger than
        // the plain ones times the line's length; the stops' waits add up to the period's
        let len_km = (w.svc.lines[l as usize].runs[0].stop_s.last().unwrap() - w.svc.lines[l as usize].runs[0].stop_s[0]) / 1000.0;
        assert!(p(0, 6) > 0.0 && p(0, 6) <= p(0, 4) * len_km + 1e-6 && p(0, 7) > 0.0 && p(0, 7) <= p(0, 2) * len_km + 1e-6, "{v:?}");
        let n = w.svc.lines[l as usize].stops.len();
        let at: f64 = v[PERIODS * PERIOD_STRIDE..PERIODS * PERIOD_STRIDE + n].iter().sum();
        assert!(v.len() == PERIODS * (PERIOD_STRIDE + n) && (at - p(0, 4)).abs() < 1e-6 * p(0, 4).max(1.0), "{at} {}", p(0, 4));
        // T-472: at 08:00 the trains in service are about the line's peak trains, every one full
        let v = b.past_day(&w.svc, 2.0 * DAY + 8.0 * 3600.0);
        // T-491: the period now's wait, arrivals and ride end the block, then T-818's two
        let (wait, arrived) = (v[v.len() - 5], v[v.len() - 4]);
        assert!((wait - p(0, 4)).abs() < 1e-6 && (arrived - p(0, 2)).abs() < 1e-6);
        let (on, packed, trains) = (v[v.len() - 8], v[v.len() - 7], v[v.len() - 6]);
        assert!(on > 0.0 && (on - w.svc.lines[l as usize].trains[HIGH] as f64).abs() <= 2.0 && packed == on, "{on} {packed}");
        // T-481: the full trains are counted out of every trip in the 24 hours
        assert!(trains >= v[4] && trains > 100.0, "{trains} trips, {} full", v[4]);
    }

    #[test]
    fn a_queue_that_lasts_the_night_carries_round_the_day() {
        // the night (2.5 trains an hour, 4,000 of room) gets 6,000 an hour: still waiting at 03:00
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [2000.0, 2000.0, 2000.0, 2000.0, 6000.0]));
        b.ensure(&w.net, &w.svc);
        let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
        let quick = simulate(&w.net, &w.svc, l as usize, 0, b.riders[0][l as usize].as_ref().unwrap(), &b.peak, false);
        assert!(quick.leftover > 1000.0, "{}", quick.leftover);
        // the kept account is the warm one: the first train after 03:00 already finds a queue
        let i = ra.deps.iter().position(|&d| d >= START_HOUR * 3600.0).unwrap();
        assert!(ra.q_after[i * 3] > 1000.0, "{}", ra.q_after[i * 3]);
        let j = quick.deps.iter().position(|&d| d >= START_HOUR * 3600.0).unwrap();
        assert!(ra.q_after[i * 3] > quick.q_after[j * 3] + 1000.0);
        // and every departure of the day is in it once, in time order
        assert_eq!(ra.deps.len(), quick.deps.len());
        assert!(ra.deps.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn a_change_mid_peak_keeps_the_account_on_the_drawn_trains() {
        // T-206: 10 an hour cut to 4 at 08:20; trains already running keep their loads, and
        // every train drawn around the change has its own account
        let (mut w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [3000.0, 2000.0, 2000.0, 2000.0, 100.0]));
        let now = 2.0 * DAY + 8.0 * 3600.0 + 1200.0;
        b.ensure(&w.net, &w.svc);
        let before_trips = w.svc.trips(&w.net, now - 600.0, now);
        let loads_before: Vec<Option<Vec<f32>>> = before_trips.iter().map(|t| b.trip_account(t.line, t.run as usize, t.dep)).collect();
        let before = w.svc.before_edit(&w.net, now);
        w.apply(crate::track::world::Op::Schedule { id: l, tph: [4.0, 5.0, 2.5] }).unwrap();
        w.svc.note_changes(&w.net, &before, now);
        b.ensure(&w.net, &w.svc);
        for epoch in [now - 1200.0, now + 2400.0] {
            for t in w.svc.trips(&w.net, epoch, epoch + 3600.0) {
                assert!(b.trip_account(t.line, t.run as usize, t.dep).is_some(), "{t:?} has no account");
            }
        }
        // the trains that left before the change were already full or not: the same riders
        for (t, was) in before_trips.iter().zip(&loads_before) {
            if t.dep < now - 900.0 {
                assert_eq!(&b.trip_account(t.line, t.run as usize, t.dep), was, "{t:?}");
            }
        }
        // fewer trains: riders now wait, on the drawn departures
        let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
        let late = ra.deps.iter().enumerate().filter(|(_, &d)| d > (now + 600.0).rem_euclid(DAY) && d < (now + 3000.0).rem_euclid(DAY)).count();
        assert!(late <= 5, "{late} departures in 40 minutes at 4 an hour");
    }

    /// T-309: a London line and a New York line in one Mercator world, on the registry's time
    /// zones (New York's and Toronto's; London is in none, so its longitude's UTC+0). The same
    /// schedule and riders run on each city's own hours.
    #[test]
    fn a_london_line_runs_on_londons_hours() {
        use crate::track::frame::FrameKind;
        use crate::track::localtime::TimeZones;
        use crate::track::net::Network;
        use crate::track::world::TrackWorld;
        let mut net = Network::new_in(FrameKind::Merc, -73.985, 40.758, true);
        net.time_zones = TimeZones::from_table(&[-5.0, 120_000.0, -73.985, 40.758, -5.0, -79.3832, 43.6534, -5.0]);
        let mut w = TrackWorld::new(net);
        let city = |w: &mut TrackWorld, lon: f64, lat: f64| {
            let k = 1.0 / (lat * std::f64::consts::PI / 180.0).cos();
            let (x, y) = w.net.frame().from_lonlat(lon, lat);
            let (x, y) = (quant(x), quant(y));
            let s: Vec<u32> = (0..3).map(|i| node(w, x + i as f64 * 3000.0 * k, y, 0)).collect();
            edge(w, s[0], s[1], &[], 0).unwrap();
            edge(w, s[1], s[2], &[], 0).unwrap();
            for &n in &s {
                station(w, n, 200);
            }
            line(w, &s, [10.0, 5.0, 2.5])
        };
        let nyc = city(&mut w, -73.99, 40.75);
        let lon = city(&mut w, -0.12, 51.51);
        assert_eq!(w.svc.lines[nyc as usize].shift, 0.0);
        assert_eq!(w.svc.lines[lon as usize].shift, 5.0 * 3600.0);
        // service periods: the morning peak's ten an hour start at 06:00 local, 01:00 game time
        let day = 3.0 * DAY;
        let levels = |l: u32, h: f64| -> Vec<u8> {
            let mut v: Vec<u8> = w.svc.trips(&w.net, day + h * 3600.0, day + (h + 1.0) * 3600.0).iter().filter(|t| t.line == l && t.draw >= day + h * 3600.0).map(|t| t.level).collect();
            v.dedup();
            v
        };
        assert_eq!(levels(lon, 0.0), vec![LOW as u8]);
        assert_eq!(levels(lon, 1.0), vec![HIGH as u8]);
        assert_eq!(levels(nyc, 1.0), vec![LOW as u8]);
        assert_eq!(levels(nyc, 6.0), vec![HIGH as u8]);
        // every window agrees on the departures, across the change of period too
        let deps = |a: f64, b: f64| {
            let mut v = vec![];
            w.svc.run_deps(&w.net, lon as usize, 1, day + a * 3600.0, day + b * 3600.0, &mut v);
            v.iter().map(|t| t.dep).collect::<Vec<f64>>()
        };
        let mut two = deps(0.5, 1.0);
        two.extend(deps(1.0, 1.7));
        assert_eq!(deps(0.5, 1.7), two);
        // the boarding account: London at 03:00 game time is its 08:00, New York's at 08:00
        let mut b = Boarding::default();
        let mut wk = riders(nyc, [500.0, 200.0, 500.0, 200.0, 50.0]);
        wk.extend_from_slice(&riders(lon, [500.0, 200.0, 500.0, 200.0, 50.0])[PERIODS..]);
        b.set_riders_day(&wk, 0);
        let mut we = riders(nyc, [100.0, 300.0, 200.0, 200.0, 10.0]);
        we.extend_from_slice(&riders(lon, [100.0, 300.0, 200.0, 200.0, 10.0])[PERIODS..]);
        b.set_riders_day(&we, 1);
        let arriving = |h: &[f32], l: u32| {
            let (nt, nl, nq) = (h[0] as usize, h[1] as usize, h[2] as usize);
            let q0 = 5 + 2 * nt + nl;
            (0..nq).map(|k| &h[q0 + k * QUEUE_STRIDE..q0 + (k + 1) * QUEUE_STRIDE]).filter(|q| q[1] as u32 == l).map(|q| q[7]).sum::<f32>()
        };
        let at = |b: &mut Boarding, h: f64| b.hour(&w.net, &w.svc, day + h * 3600.0);
        let (h3, h8) = (at(&mut b, 3.0), at(&mut b, 8.0));
        // 500 an hour at each end in the morning peak, 50 at night
        assert!((arriving(&h3, lon) - 1000.0).abs() < 1.0, "{}", arriving(&h3, lon));
        assert!((arriving(&h3, nyc) - 100.0).abs() < 1.0, "{}", arriving(&h3, nyc));
        assert!((arriving(&h8, nyc) - 1000.0).abs() < 1.0, "{}", arriving(&h8, nyc));
        // every London trip of the hour has its load in the account
        let trips = w.svc.trips(&w.net, day + 3.0 * 3600.0, day + 4.0 * 3600.0);
        assert!(trips.iter().filter(|t| t.line == lon).all(|t| b.trip_account(t.line, t.run as usize, t.dep).is_some()));
        // weekends by the local day: game day 6 (Saturday) starts five hours early in London,
        // at 19:00 on Friday game time (day 5), where New York is still on Friday evening
        let fri = 5.0 * DAY + 20.0 * 3600.0 - day;
        let h = at(&mut b, fri / 3600.0);
        assert!((arriving(&h, lon) - 20.0).abs() < 0.5, "{}", arriving(&h, lon)); // Saturday 01:00
        assert!((arriving(&h, nyc) - 400.0).abs() < 1.0, "{}", arriving(&h, nyc)); // Friday 20:00
        // fares follow: today's boardings count from London's midnight
        let v = b.today(&w.svc, day + 3.0 * 3600.0);
        assert_eq!(v[0], 2.0);
    }

    /// T-309: the real New York save (trainworld/T-007-real-nyc.save, Anita's) runs exactly as
    /// before local time with the registry's time zones: every line is on New York's time, and a
    /// week of drawn trips, every hour's boarding slice, today's counts and the line summaries are
    /// bit-identical to the network without a time zone table. Runs on request, on the save's
    /// track bytes (gunzipped, after its JSON header): `T309_TRACK=<file> cargo test --manifest-path
    /// sim/Cargo.toml --release t309 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn t309_real_save_runs_as_before() {
        use crate::track::localtime::TimeZones;
        use crate::track::world::TrackWorld;
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let bytes = std::fs::read(std::env::var("T309_TRACK").expect("T309_TRACK: the save's track bytes")).unwrap();
        // the registry's table, as app/src/game/localTime.ts builds it
        let reg = std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap();
        let num = |s: &str| s.trim().trim_end_matches(',').trim().parse::<f64>().unwrap();
        let clock = num(reg.split("\"clockUtcOffset\":").nth(1).unwrap().split('\n').next().unwrap());
        let mut table = vec![clock, 120_000.0];
        for city in reg.split("\"center\": [").skip(1) {
            let mut it = city.split(']').next().unwrap().split(',');
            let (lon, lat) = (num(it.next().unwrap()), num(it.next().unwrap()));
            let off = num(city.split("\"utcOffset\":").nth(1).unwrap().split('\n').next().unwrap());
            table.extend([lon, lat, off]);
        }
        println!("time zone table: {} zones, clock UTC{clock}", (table.len() - 2) / 3);
        let a = TrackWorld::new(crate::track::save::decode(&bytes).unwrap());
        let mut net = crate::track::save::decode(&bytes).unwrap();
        net.time_zones = TimeZones::from_table(&table);
        let b = TrackWorld::new(net);
        let running: Vec<usize> = (0..b.svc.lines.len()).filter(|&l| b.svc.lines[l].running).collect();
        assert!(running.iter().all(|&l| b.svc.lines[l].shift == 0.0));
        assert!(b.net.time_zones.set);
        println!("{} running lines, all on New York's time", running.len());
        let day = 2.0 * DAY;
        let mut trips = 0;
        for h in 0..(7 * 24) {
            let (t0, t1) = (day + h as f64 * 3600.0, day + (h + 1) as f64 * 3600.0);
            let (x, y) = (a.svc.trips(&a.net, t0, t1), b.svc.trips(&b.net, t0, t1));
            assert_eq!(x, y, "hour {h}");
            trips += x.len();
        }
        // made-up riders on every running line (the account's arithmetic is what is compared)
        let packed = |scale: f32| {
            let mut v = vec![1.4, 1.15, 1.4, 1.3, 1.5];
            for &l in &running {
                let n = b.svc.lines[l].stops.len();
                v.extend([l as f32, n as f32]);
                for p in 0..PERIODS {
                    let x = scale * (p as f32 + 1.0) * 37.0;
                    v.extend((0..2 * n).map(|k| if k % n + 1 < n { x } else { 0.0 }));
                    v.extend((0..2 * n).map(|k| if k % n > 0 { x } else { 0.0 }));
                }
            }
            v
        };
        let (mut ba, mut bb) = (Boarding::default(), Boarding::default());
        for bo in [&mut ba, &mut bb] {
            bo.set_riders_day(&packed(1.0), 0);
            bo.set_riders_day(&packed(0.6), 1);
        }
        for h in 0..(7 * 24) {
            let t = day + h as f64 * 3600.0;
            let (x, y) = (ba.hour(&a.net, &a.svc, t), bb.hour(&b.net, &b.svc, t));
            assert!(x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "hour {h}");
            let (x, y) = (ba.today(&a.svc, t + 1234.5), bb.today(&b.svc, t + 1234.5));
            assert!(x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "today {h}");
        }
        for &l in &running {
            for d in 0..DAY_KINDS {
                assert_eq!(ba.line_summary_day(&a.svc, l, d), bb.line_summary_day(&b.svc, l, d));
            }
        }
        println!("a week: {trips} trips, every hourly boarding slice and today's counts bit-identical");
    }

    /// T-554: riders changing from a train reach their next platform a transfer walk after it
    /// leaves, the hour's arrivals stay the same, and the account itself does not move.
    #[test]
    fn riders_changing_lines_arrive_with_their_train() {
        // line 1 A - B - C every 6 minutes at the peak; line 2 E - F - G every 15, ending at G,
        // 60 m from B
        let (mut w, l1) = setup(10.0);
        let e = node(&mut w, 3000.0, 6060.0, 0);
        let f = node(&mut w, 3000.0, 3060.0, 0);
        let g = node(&mut w, 3000.0, 60.0, 0);
        edge(&mut w, e, f, &[], 0).unwrap();
        edge(&mut w, f, g, &[], 0).unwrap();
        for n in [e, f, g] {
            station(&mut w, n, 200);
        }
        let l2 = line(&mut w, &[e, f, g], [4.0, 2.0, 1.0]);
        let mut v = vec![1.0; PERIODS];
        let hours = |p: usize| (PERIOD_HOURS[p].1 - PERIOD_HOURS[p].0) as f32;
        v.extend([l1 as f32, 3.0]);
        for p in 0..PERIODS {
            let (x, y) = (300.0 * hours(p), 200.0 * hours(p));
            v.extend([x, y, 0.0, x, y, 0.0, 0.0, 0.0, x + y, 0.0, 0.0, x + y]);
        }
        v.extend([l2 as f32, 3.0]);
        for p in 0..PERIODS {
            let z = 200.0 * hours(p);
            v.extend([z, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * z, 0.0, 0.0, 0.0]);
        }
        let day_b: f32 = (0..PERIODS).map(|p| 200.0 * hours(p)).sum();
        let mut b = Boarding::default();
        b.set_riders(&v);
        let epoch = 3.0 * DAY + 8.0 * 3600.0;
        let before = b.hour(&w.net, &w.svc, epoch);
        // half of B's boarders each way come off line 2 at G
        b.set_transfers(&[l2 as f32, 0.0, 2.0, l1 as f32, 0.0, 1.0, day_b / 2.0, l2 as f32, 0.0, 2.0, l1 as f32, 1.0, 1.0, day_b / 2.0]);
        let after = b.hour(&w.net, &w.svc, epoch);
        let parts = |h: &[f32]| {
            let (nt, nl, nq, ne) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3] as usize);
            let q0 = 5 + 2 * nt + nl;
            let e0 = q0 + nq * QUEUE_STRIDE;
            (h[..q0].to_vec(), h[q0..e0].to_vec(), h[e0..e0 + 2 * ne].to_vec(), h[e0 + 2 * ne..].to_vec())
        };
        let (ta, qa, ea, pa) = parts(&before);
        let (tb, qb, eb, pb) = parts(&after);
        // the trains' loads and every train leaving a platform are as before
        assert_eq!(ta[..4], tb[..4]);
        assert_eq!(ta[5..], tb[5..]);
        assert_eq!(ea, eb);
        assert!(pa.is_empty() && !pb.is_empty());
        let mut seen = 0;
        for (x, y) in qa.chunks(QUEUE_STRIDE).zip(qb.chunks(QUEUE_STRIDE)) {
            assert_eq!(x[..4], y[..4]);
            assert_eq!(x[5..10], y[5..10]); // left behind, waits, the hour's arrivals, events
            if x[1] as u32 != l1 || x[0] as u32 != w.svc.lines[l1 as usize].runs[0].nodes[w.svc.lines[l1 as usize].runs[0].stop_idx[1] as usize] {
                assert_eq!(x[4], y[4]);
                assert_eq!(y[11], 0.0);
                continue;
            }
            seen += 1;
            // at B: half arrive steadily, half in four pulses an hour, one per line 2 train
            assert!((y[4] - x[4] / 2.0).abs() < 1e-6, "{} {}", x[4], y[4]);
            let p = &pb[y[10] as usize * 2..(y[10] + y[11]) as usize * 2];
            let hour: Vec<&[f32]> = p.chunks(2).filter(|c| c[0] >= 0.0 && c[0] < 3600.0).collect();
            assert_eq!(hour.len(), 4);
            let sum: f32 = hour.iter().map(|c| c[1]).sum();
            assert!((sum - x[7] / 2.0).abs() < 0.01, "{sum} {}", x[7]);
            // before the hour, only after the last train to leave before it
            let since = eb[y[8] as usize * 2];
            assert!(p.chunks(2).all(|c| c[0] > since && c[0] < 3600.0));
            // each a 65 s walk (60 m x 1.3 at 1.2 m/s) after a line 2 train reaches G
            let g_t: Vec<f64> = {
                let ra = &b.accounts[0][l2 as usize].as_ref().unwrap().runs[0];
                let m = ra.deps.len();
                ra.order_t[2 * m..3 * m].to_vec()
            };
            let h0 = epoch.rem_euclid(DAY);
            for c in &hour {
                let t = c[0] as f64 - 65.0 + h0;
                assert!(g_t.iter().any(|&x| (x - t).abs() < 0.01), "{}", c[0]);
            }
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn the_hour_lines_up_with_the_drawn_trips() {
        let (w, l) = setup(10.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
        let epoch = 3.0 * DAY + 8.0 * 3600.0;
        let out = b.hour(&w.net, &w.svc, epoch);
        let trips = w.svc.trips(&w.net, epoch, epoch + 3600.0);
        let (nt, nl, nq, ne, np) = (out[0] as usize, out[1] as usize, out[2] as usize, out[3] as usize, out[4] as usize);
        assert_eq!(nt, trips.len());
        assert_eq!(np, 0); // no transfers set
        assert_eq!(out.len(), 5 + 2 * nt + nl + nq * QUEUE_STRIDE + ne * 2);
        let offs: Vec<f32> = out[5..5 + 2 * nt].chunks(2).map(|c| c[0]).collect();
        assert!(offs.iter().all(|&o| o >= 0.0));
        let run_s = w.svc.lines[l as usize].runs[0].prof[HIGH].duration as f32;
        assert!(out[5..5 + 2 * nt].chunks(2).all(|c| c[1] == run_s), "{:?} {run_s}", &out[5..9]);
        let out = &out[nt + 1..]; // the rest as if one number per trip and a 4-number header
        let loads = &out[4 + nt..4 + nt + nl];
        for (t, o) in trips.iter().zip(offs.iter().copied()) {
            let own = b.trip_account(t.line, t.run as usize, t.dep).unwrap();
            let o = o as usize;
            if t.run == 0 {
                assert_eq!(&loads[o..o + 2], &own[0..2]);
            } else {
                assert_eq!(loads[o], own[1]);
                assert_eq!(loads[o + 1], own[0]);
            }
        }
        // platforms: A and C have riders (the line's two boarding stops), B none
        assert_eq!(nq, 2);
        let qs = &out[4 + nt + nl..4 + nt + nl + nq * QUEUE_STRIDE];
        let ev = &out[4 + nt + nl + nq * QUEUE_STRIDE..];
        for q in qs.chunks(QUEUE_STRIDE) {
            let (first, count) = (q[8] as usize, q[9] as usize);
            assert!(count >= 11, "{count}"); // 10 an hour, one before and one after
            let e = &ev[first * 2..(first + count) * 2];
            assert!(e[0] < 0.0 && e[(count - 1) * 2] >= 3600.0);
            assert!(e.chunks(2).all(|p| p[1] > 0.0)); // people left behind at every train
            assert!(q[5] > 0.0);
            assert!((q[7] - 20000.0 * 1.0).abs() < 1.0); // flat shape: 20,000 this hour
        }
    }

    #[test]
    fn a_new_lines_platforms_fill_from_when_its_trains_start() {
        // T-680: trains set on a line at 08:20; its platforms are empty then and fill at the
        // arrival rate, not with everyone who would have arrived in the six hours before
        let (mut w, l) = setup(0.0);
        let mut b = Boarding::default();
        b.set_riders(&riders(l, [3000.0, 2000.0, 2000.0, 2000.0, 100.0]));
        let now = 2.0 * DAY + 8.0 * 3600.0 + 1200.0;
        let before = w.svc.before_edit(&w.net, now);
        w.apply(crate::track::world::Op::Schedule { id: l, tph: [10.0, 5.0, 2.5] }).unwrap();
        w.svc.note_changes(&w.net, &before, now);
        let epoch = now - 1200.0;
        let out = b.hour(&w.net, &w.svc, epoch);
        let (nt, nl, nq) = (out[0] as usize, out[1] as usize, out[2] as usize);
        let q0 = 5 + 2 * nt + nl;
        let ev = &out[q0 + nq * QUEUE_STRIDE..];
        assert_eq!(nq, 2);
        for q in out[q0..q0 + nq * QUEUE_STRIDE].chunks(QUEUE_STRIDE) {
            let (rate, first, count) = (q[4], q[8] as usize, q[9] as usize);
            let e = &ev[first * 2..(first + count) * 2];
            // as game/boarding.ts's crowdAt: the last train to leave at or before t, plus arrivals
            let crowd = |t: f32| {
                let j = e.chunks(2).rposition(|p| p[0] <= t).unwrap_or(0);
                e[j * 2 + 1] + (t - e[j * 2]).max(0.0) * rate
            };
            let mut t = 1200.0f32;
            while t <= 3600.0 {
                assert!(crowd(t) <= rate * (t - 1200.0) + 1.0, "{} waiting at {t} s of the hour, {} a second", crowd(t), rate);
                t += 30.0;
            }
            assert!(crowd(1199.0) < 1.0);
            // riders arriving this hour: from 08:20 only
            assert!((q[7] - 3000.0 * 40.0 / 60.0).abs() < 1.0, "{}", q[7]);
        }
        // the line's riders today: those since 08:20 at most
        let today = b.today(&w.svc, now + 1800.0);
        assert!(today[3] <= 2.0 * 3000.0 * 0.5 + 1.0, "{} riders today", today[3]);
        assert!(today[3] > 0.0);
    }
}
