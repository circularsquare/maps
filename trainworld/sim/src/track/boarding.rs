//! The boarding account (T-164, notes/T-164.md, docs/track-and-service.md 6.3): demand's riders per
//! period turned into a load per train and a crowd per platform.
//!
//! Demand decides who travels and by which line (SPEC 4); it reports, per line, direction, stop and
//! period, how many board and how many alight. Here those boardings arrive at the platform over the
//! period (a simple in-period shape, below), and each train of the line, in the order it reaches the
//! stop, lets off the line's share alighting there and takes on whoever is waiting, up to its crush
//! load. Whoever does not fit waits for the next train of the same line and direction, until full
//! trains have kept them `GIVE_UP_S` or the platform holds `WAITING_CAP` (T-1067). Trains do
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
use super::span::{self, Mask};
use std::collections::HashMap;

/// Seats and riders at crush load per 20 m car: the demand kernel's `SEATS_PER_CAR` and
/// `CRUSH_PER_CAR` (sim/src/demand/api.rs, a New York subway car).
pub const SEATS_PER_CAR: f64 = 44.0;
pub const CRUSH_PER_CAR: f64 = 160.0;
pub const DAY: f64 = 86400.0;
pub const PERIODS: usize = PERIOD_HOURS.len();
/// T-1067: a rider whom full trains have kept this long past the first train they came for goes
/// another way (DECISIONS 2026-10-23).
pub const GIVE_UP_S: f64 = 5400.0;
/// T-1067: the most riders waiting on one platform side (a direction of a line at a stop), what a
/// 300 m platform's side 5 m deep holds at 0.3 m² a rider; whoever comes when it is full goes
/// another way. `WAITING_CAP` in app/src/game/boarding.ts.
pub const WAITING_CAP: f64 = 5000.0;
/// T-1255: a rider comes and waits for a train with room only when the riders ahead all get on
/// within this long (DECISIONS 2026-10-25); demand's cap (`kernel::capacity_fit`) is the same queue.
pub const WAIT_FOR_ROOM_S: f64 = 1800.0;

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
    /// T-842b: riders arrive only as the tolerance lets them (`span::Mask`: none while no train is
    /// due for hours), each period's total kept; None: over the whole period by its shape.
    masked: Option<Box<Masked>>,
}

#[derive(Clone, Debug, Default)]
struct Masked {
    m: Mask,
    /// per period, the mask's share over its riders by this platform's shape (0: no train due all
    /// period, and the period's riders arrive as if unmasked)
    norm: [f64; PERIODS],
    /// riders arrived from the start of the day to each segment's start
    pre: Vec<f64>,
}

impl Arrivals {
    pub fn new(per: [f64; PERIODS], peak: &[f64; PERIODS]) -> Self {
        let c = std::array::from_fn(|p| {
            let (h0, h1) = PERIOD_HOURS[p];
            shape_strength(SHAPE[p], h1 - h0, peak[p])
        });
        Arrivals { per, c, day: per.iter().sum(), masked: None }
    }
    /// `new`, with riders spread only where `m` lets them arrive; a mask letting everyone arrive
    /// all day changes nothing.
    pub fn with_mask(per: [f64; PERIODS], peak: &[f64; PERIODS], m: Mask) -> Self {
        let mut a = Self::new(per, peak);
        let ns = m.t.len();
        if ns == 0 || m.s.iter().all(|&s| s == 1.0) {
            return a;
        }
        let ends: Vec<f64> = (0..ns).map(|i| if i + 1 < ns { m.t[i + 1] } else { DAY }).collect();
        let mut norm = [0.0; PERIODS];
        for i in 0..ns {
            let p = period_at(m.t[i]);
            norm[p] += m.s[i] * (a.w(p, ends[i]) - a.w(p, m.t[i]));
        }
        let mut x = Masked { m, norm, pre: Vec::with_capacity(ns) };
        let mut tot = 0.0;
        for i in 0..ns {
            x.pre.push(tot);
            tot += a.seg(&x, i, x.m.t[i], ends[i]);
        }
        a.day = tot;
        a.masked = Some(Box::new(x));
        a
    }
    /// Period `p`'s shape: the share of its riders (unmasked) arrived by `s` seconds into the day.
    fn w(&self, p: usize, s: f64) -> f64 {
        let (a, b) = (PERIOD_HOURS[p].0 * 3600.0, PERIOD_HOURS[p].1 * 3600.0);
        shape_cum(SHAPE[p], self.c[p], ((s - a) / (b - a)).clamp(0.0, 1.0))
    }
    /// The mask's weight on riders in segment `i`.
    fn eff(&self, x: &Masked, i: usize) -> f64 {
        let p = period_at(x.m.t[i]);
        if x.norm[p] > 1e-9 { x.m.s[i] / x.norm[p] } else { 1.0 }
    }
    /// Riders arriving in segment `i` between `u0` and `u1` seconds into the day.
    fn seg(&self, x: &Masked, i: usize, u0: f64, u1: f64) -> f64 {
        let p = period_at(x.m.t[i]);
        self.per[p] * self.eff(x, i) * (self.w(p, u1) - self.w(p, u0))
    }
    /// Riders arrived since the start of game day 0 (negative before it).
    pub fn cum(&self, t: f64) -> f64 {
        let d = (t / DAY).floor();
        let s = t - d * DAY;
        let mut tot = d * self.day;
        if let Some(x) = &self.masked {
            let i = x.m.t.partition_point(|&u| u <= s).saturating_sub(1);
            return tot + x.pre[i] + self.seg(x, i, x.m.t[i], s);
        }
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
        let r = self.per[p] / (b - a) * (1.0 + self.c[p] * shape_g(SHAPE[p], ((s - a) / (b - a)).clamp(0.0, 1.0)));
        match &self.masked {
            Some(x) => r * self.eff(x, x.m.t.partition_point(|&u| u <= s).saturating_sub(1)),
            None => r,
        }
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
    /// T-1255: riders coming to each platform, laid out as `board` (at least its boardings: those
    /// who would ride if the trains had room); empty when demand sent none, and then the boardings
    pub come: Vec<f64>,
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
    /// `[trip * n + stop]`: riders who went another way as the train left (T-1067: `GIVE_UP_S`,
    /// `WAITING_CAP`), never boarding.
    pub gone: Vec<f32>,
    /// Per stop, the trips in the order they leave it in a day (`stop * trips + k`), and the time
    /// of day each leaves it.
    pub order: Vec<u32>,
    pub order_t: Vec<f64>,
    /// Per stop and hour of the day (`stop * 24 + hour`): riders who could not get on the first
    /// train to leave after they arrived, and the longest anyone waited for a train leaving then.
    pub left_h: Vec<f32>,
    pub wait_h: Vec<f32>,
    pub arrivals: Vec<Arrivals>,
    /// Totals over the day, for checks: riders who arrived, boarded, alighted, went another way.
    pub arrived: f64,
    pub boarded: f64,
    pub alighted: f64,
    pub gave_up: f64,
    /// Riders still waiting at a stop once the day's last train has left it.
    pub leftover: f64,
    /// When the line's trains started, in its local time, if within this account's day (T-680):
    /// its platforms are empty then. Departures by time of day before it are the day wrapping
    /// round, not trains that ran. `NEG_INFINITY` otherwise.
    pub open: f64,
    /// Per stop, when its platform starts filling on the day the line opens (T-848): one headway
    /// before the first train to reach it, never before `open`. `NEG_INFINITY` otherwise.
    pub stop_open: Vec<f64>,
    /// Per period and stop (`p * n + k`), the km a boarder there rides on this line (`ride_km`).
    pub ride_km: Vec<f64>,
    /// Per stretch (stop k to k + 1 in this run's order), the share of the day's trains that leave
    /// it packed to crush (`full_shares`, T-1125, T-1152).
    pub full: Vec<f32>,
    /// Per stretch, the extra wait (s) of the day's riders arriving at its first stop for a train
    /// with room, on average (`stop_waits`, T-1160).
    pub wait: Vec<f32>,
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
/// kind of day (T-260, `day_kind_at`), as the same trains carry a weekday's, a Saturday's or a
/// Sunday's riders.
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
    /// Per line id, the local game days its country runs as another kind than `day_kind_at`'s, and
    /// that kind: holidays, and its own weekend where that is not Saturday and Sunday (T-872,
    /// T-262, `set_holidays`). The clock worker keeps a window around now.
    pub holidays: HashMap<u32, Vec<(i64, usize)>>,
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
            let mut lr = LineRiders { n, board: vec![0.0; len], alight: vec![0.0; len], come: vec![] };
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

    /// The riders coming to each platform on a kind of day (T-1255), after `set_riders_day`: per
    /// line its id, stop count `n`, then for each period the riders at each route node (run 0's
    /// stops, then run 1's, in travel order: `2n`). Those beyond the boardings are riders who would
    /// ride if the trains had room; the account turns away whoever finds the queue too long.
    pub fn set_arrivals_day(&mut self, packed: &[f32], day: usize) {
        let day = day.min(DAY_KINDS - 1);
        self.generation[day] += 1;
        let mut i = 0;
        while i + 2 <= packed.len() {
            let (id, n) = (packed[i] as usize, packed[i + 1] as usize);
            i += 2;
            let len = PERIODS * 2 * n;
            if i + len > packed.len() {
                break;
            }
            if let Some(Some(lr)) = self.riders[day].get_mut(id) {
                if lr.n == n {
                    lr.come = packed[i..i + len].iter().map(|&v| v.max(0.0) as f64).collect();
                }
            }
            i += len;
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

    /// Each line's days of another kind (T-872, T-262): per line, its id, a count `n`, then `n`
    /// pairs of a local game day and its kind.
    pub fn set_holidays(&mut self, packed: &[f64]) {
        self.holidays.clear();
        let mut i = 0;
        while i + 2 <= packed.len() {
            let (id, n) = (packed[i] as u32, packed[i + 1] as usize);
            let days = &packed[(i + 2).min(packed.len())..(i + 2 + 2 * n).min(packed.len())];
            if days.len() >= 2 {
                self.holidays.insert(id, days.chunks_exact(2).map(|d| (d[0] as i64, d[1] as usize)).collect());
            }
            i += 2 + 2 * n;
        }
    }

    /// The kind of day whose account line `line`'s local time `t` reads: its own, or the kind its
    /// country gives that day (a holiday, its own weekend, T-872, T-262), or the weekday's while
    /// that kind has no riders (an away day the Sunday's first: the clock worker sends its riders
    /// only while one is near, T-264).
    pub fn kind_at(&self, t: f64, line: usize) -> usize {
        let day = (t / DAY).floor() as i64;
        let given = self.holidays.get(&(line as u32)).and_then(|d| d.iter().find(|x| x.0 == day)).map(|x| x.1);
        self.kind(given.unwrap_or_else(|| day_kind_at(t)))
    }
    fn kind(&self, day: usize) -> usize {
        let day = day.min(DAY_KINDS - 1);
        if !self.riders[day].is_empty() {
            day
        } else if day == AWAY && !self.riders[2].is_empty() {
            2
        } else {
            0
        }
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
        if ls.is_loop() {
            h.add(ls.kind as u64 + 100);
        }
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
            h.f(e.lat.lead);
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
    /// QUEUE_STRIDE numbers, then per event 2, then per pulse 2, then per event its train's
    /// arrival (T-1123), then per pulse the station node its riders come from and their walk (s,
    /// T-1124), then the count of full stretches, per one its line and run 0 stretch (T-1125), then
    /// per one the share of the day's trains packed to crush there (T-1152), then per one the
    /// riders' average wait there for a train with room (s, T-1160)]`.
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
        let mut pulses: Vec<(f32, f32, f32, f32)> = vec![];
        // the hour's kind of day (T-260) in each line's local time (T-309): every trip and
        // platform of the line reads that day's account, at its local hour
        let shift = |l: usize| svc.lines.get(l).map_or(0.0, |ls| ls.shift);
        let trips = svc.trips(net, epoch, epoch + 3600.0);
        let mut offs: Vec<f32> = Vec::with_capacity(trips.len() * 2);
        let mut loads: Vec<f32> = vec![];
        for t in &trips {
            let day = self.kind_at(epoch + shift(t.line as usize), t.line as usize);
            offs.push(self.trip_loads(t, day, &mut loads));
            offs.push(svc.lines[t.line as usize].runs[t.run as usize].prof[t.level as usize].duration as f32);
        }
        let mut queues: Vec<f32> = vec![];
        let mut events: Vec<f32> = vec![];
        let mut arrive: Vec<f32> = vec![];
        let nl = self.accounts.iter().map(|a| a.len()).max().unwrap_or(0);
        for l in 0..nl {
            // the hour in the line's local time
            let epoch = epoch + shift(l);
            let day = self.kind_at(epoch, l);
            let Some(Some(acc)) = self.accounts[day].get(l) else { continue };
            let h0 = epoch.rem_euclid(DAY);
            let hour = ((h0 / 3600.0).floor() as usize).min(23);
            let ls = &svc.lines[l];
            // a bus stop has no platform (T-1464)
            if ls.is_bus() {
                continue;
            }
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
                    // T-1123: each train's arrival too, when its riders get off and on: its stand
                    // there before it leaves (the closing stand at a first stop), never before
                    // the train ahead left
                    let close = ls.close_stand(net, l as u32);
                    let a0 = arrive.len();
                    for j in first as isize - 1..=end as isize {
                        let (t, i) = at(j);
                        events.push(t as f32);
                        events.push(ra.q_after[i * ra.n + k]);
                        let stand = if k == 0 { close } else { run.prof.get(ra.levels[i] as usize).and_then(|p| Some(p.dep.get(k)? - p.arr.get(k)?)).unwrap_or(0.0) };
                        arrive.push((t - stand.max(0.0)) as f32);
                    }
                    // T-680: empty when the line's trains started, and no train before then
                    let open_k = ra.stop_open.get(k).copied().unwrap_or(ra.open);
                    let open = open_k - epoch;
                    if (events[start * 2] as f64) < open {
                        let after = (start..events.len() / 2).find(|&e| events[e * 2] as f64 >= open).unwrap_or(events.len() / 2);
                        events.splice(start * 2..after * 2, [open as f32, 0.0]);
                        arrive.splice(a0..a0 + (after - start), [open as f32]);
                    }
                    for e in a0 + 1..arrive.len() {
                        arrive[e] = arrive[e].max(events[(e - 1) * 2]).min(events[e * 2]);
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
                        (arr.cum(epoch + 3600.0) - arr.cum(epoch.max(open_k).min(epoch + 3600.0))) as f32,
                        start as f32,
                        (events.len() / 2 - start) as f32,
                        p0 as f32,
                        (pulses.len() - p0) as f32,
                    ]);
                }
            }
        }
        // T-1125, T-1152, T-1160: the stretches where riders wait FULL_WAIT_MIN_S or more for a
        // train with room, either way, in run 0's numbering; the share of the day's trains packed
        // to crush there and that wait (s), each the fuller way's
        let (mut full, mut shares, mut waits): (Vec<f32>, Vec<f32>, Vec<f32>) = (vec![], vec![], vec![]);
        for l in 0..nl {
            let day = self.kind_at(epoch + shift(l), l);
            let Some(Some(acc)) = self.accounts[day].get(l) else { continue };
            let back = |v: &[f32], s: usize| v.len().checked_sub(1 + s).and_then(|i| v.get(i)).copied().unwrap_or(0.0);
            let (r0, r1) = (&acc.runs[0], acc.runs.get(1));
            for s in 0..r0.full.len().max(r1.map_or(0, |r| r.full.len())) {
                let wait = r0.wait.get(s).copied().unwrap_or(0.0).max(r1.map_or(0.0, |r| back(&r.wait, s)));
                if wait >= FULL_WAIT_MIN_S {
                    full.extend([l as f32, s as f32]);
                    shares.push(r0.full.get(s).copied().unwrap_or(0.0).max(r1.map_or(0.0, |r| back(&r.full, s))));
                    waits.push(wait);
                }
            }
        }
        let mut out = Vec::with_capacity(6 + offs.len() + loads.len() + queues.len() + events.len() + arrive.len() + pulses.len() * 4 + full.len());
        out.extend_from_slice(&[trips.len() as f32, loads.len() as f32, (queues.len() / QUEUE_STRIDE) as f32, (events.len() / 2) as f32, pulses.len() as f32]);
        out.extend_from_slice(&offs);
        out.extend_from_slice(&loads);
        out.extend_from_slice(&queues);
        out.extend_from_slice(&events);
        for p in &pulses {
            out.extend([p.0, p.1]);
        }
        out.extend_from_slice(&arrive);
        for p in &pulses {
            out.extend([p.2, p.3]);
        }
        out.push((full.len() / 2) as f32);
        out.extend_from_slice(&full);
        out.extend_from_slice(&shares);
        out.extend_from_slice(&waits);
        out
    }

    /// The riders changing onto platform `q` (line, run, stop) from other lines' trains (T-554),
    /// in the hour before game time `epoch` and the hour from it: each hour, a feeding
    /// platform's share of `q`'s boardings (demand's transfers a day over its boardings a day) of
    /// the riders arriving at `q` that hour, split over the trains leaving the feeding platform
    /// by the riders each brings in, reaching `q` a transfer walk after the train stops (T-1124).
    /// Pushed onto `out` in time order as (seconds from `epoch`, riders, the feeding station's
    /// node, the walk in seconds). Returns the share of
    /// the hour's arrivals they carry; the rest arrive steadily. A feeding platform with no
    /// train bringing riders in an hour, or under one rider a train, leaves its share steady.
    fn pulses(&self, net: &Network, svc: &Service, epoch: f64, q: (usize, usize, usize), arr: &Arrivals, node: u32, out: &mut Vec<(f32, f32, f32, f32)>) -> f64 {
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
            let Some(fls) = svc.lines.get(fl).filter(|x| !x.is_bus()) else { continue };
            let fe = epoch + fls.shift;
            let Some(Some(acc)) = self.accounts[self.kind_at(fe, fl)].get(fl) else { continue };
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
            // T-1124: riders get off as the train stops, its stand before it leaves
            let stand = |i: usize| run.prof.get(ra.levels[i] as usize).and_then(|p| Some(p.dep.get(kf)? - p.arr.get(kf)?)).unwrap_or(0.0).max(0.0);
            let most = run.prof.iter().filter_map(|p| Some(p.dep.get(kf)? - p.arr.get(kf)?)).fold(0.0f64, f64::max);
            for a in [-3600.0, 0.0] {
                // the trains whose riders reach `q` in [a, a + 3600), weighed by riders aboard
                trains.clear();
                let (lo, hi) = (h0 + a - walk, h0 + a + 3600.0 - walk);
                for d in ((lo / DAY).floor() as i64)..=(((hi + most) / DAY).floor() as i64) {
                    let off = d as f64 * DAY;
                    let (i0, i1) = (ts.partition_point(|&t| t + off < lo), ts.partition_point(|&t| t + off < hi + most));
                    for j in i0..i1 {
                        let reach = ts[j] + off - stand(ord[j] as usize) - h0 + walk;
                        let w = ra.load[ord[j] as usize * n + kf - 1] as f64;
                        if w > 0.0 && reach >= a && reach < a + 3600.0 {
                            trains.push((reach, w));
                        }
                    }
                }
                let wsum: f64 = trains.iter().map(|t| t.1).sum();
                let riders = f.riders * per * (arr.cum(local + a + 3600.0) - arr.cum(local + a));
                // under one rider a train would not show: kept steady, and the slice small
                if wsum <= 0.0 || riders < trains.len() as f64 {
                    continue;
                }
                out.extend(trains.iter().map(|&(t, w)| (t as f32, (riders * w / wsum) as f32, fnode as f32, walk as f32)));
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
        let cap = ls.crush();
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
    /// midnight, less those who went another way (`tally`). Riders left behind are counted per departure as in `simulate` (who arrived since
    /// the train before and is still waiting).
    pub fn today(&self, svc: &Service, t: f64) -> Vec<f64> {
        self.window(svc, t, false)
    }

    /// Trains of a line on the move or turning at local time of day `tod`, and how many of them
    /// are packed full somewhere on their trip (T-472). A trip counts from its departure until the
    /// next one could leave the far end, so the count is about the line's trains in service.
    fn under_way(acc: &LineAccount, ls: &LineSvc, tod: f64) -> (f64, f64) {
        let cap = ls.crush();
        let (mut on, mut packed) = (0.0, 0.0);
        for (r, ra) in acc.runs.iter().enumerate() {
            let n = ra.n;
            for j in 0..ra.deps.len() {
                let lv = ra.levels[j] as usize;
                // a loop's train is out until the next one of its run could leave (T-798)
                let span = if ls.is_loop() {
                    ls.round_trip[lv] / 2.0
                } else {
                    ls.runs[r].prof[lv].duration + ((ls.round_trip[lv] - ls.runs[0].prof[lv].duration - ls.runs[1].prof[lv].duration) / 2.0).max(0.0)
                };
                if (tod - ra.deps[j]).rem_euclid(DAY) >= span {
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
    /// stop, both directions, in the line's stop order (T-818: where they wait; T-1255: with each
    /// rider turned away at `WAIT_FOR_ROOM_S`), then per period the riders arriving at each stop
    /// likewise (T-1255). Empty without an account.
    pub fn line_periods(&self, svc: &Service, l: usize, day: usize) -> Vec<f64> {
        let Some(Some(acc)) = self.accounts[self.kind(day)].get(l) else { return vec![] };
        let Some(ls) = svc.lines.get(l) else { return vec![] };
        let mut out = Self::crowding(acc, ls).concat();
        let n = acc.runs[0].n;
        let mut at = vec![0.0f64; PERIODS * n];
        // T-1255: per period and stop, the riders arriving there, both directions
        let mut came = vec![0.0f64; PERIODS * n];
        for (r, ra) in acc.runs.iter().enumerate() {
            if ra.n == n {
                for k in 0..n {
                    let s = if r == 0 { k } else { n - 1 - k };
                    for p in 0..PERIODS {
                        came[p * n + s] += ra.arrivals[k].per[p];
                    }
                }
            }
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
                    let gone = ra.gone[ord[i] as usize * n + k] as f64;
                    let gap = if i + 1 < m { ts[i + 1] - ts[i] } else { ts[0] + DAY - ts[i] };
                    at[period_at(ts[i]) * n + s] += q * gap.max(0.0) + gone * WAIT_FOR_ROOM_S;
                }
            }
        }
        out.extend(at);
        out.extend(came);
        out
    }

    /// Per period: trains leaving in it, those packed full somewhere on their trip, riders
    /// arriving at the line's platforms, riders a full train left behind, rider-seconds spent
    /// waiting for a later train after a full one (each left behind waits the gap to the next
    /// departure; T-1255: each rider who went another way counts `WAIT_FOR_ROOM_S`), and
    /// rider-seconds aboard (T-491: the ride the wait is weighed against); then the waiting
    /// rider-seconds and the riders arriving each times the km they ride (`ride_km`), so the app
    /// can weigh riders by the fare they pay (T-818); then the riders who went another way; then
    /// the waiting rider-seconds and the riders arriving at the platforms where trains come full
    /// (`BUSY_PLATFORM_S`, T-1255: the line panel's crowding).
    fn crowding(acc: &LineAccount, ls: &LineSvc) -> [[f64; PERIOD_STRIDE]; PERIODS] {
        let cap = ls.crush();
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
                let mut here = [0.0f64; PERIODS];
                for i in 0..m {
                    let q = ra.q_after[ord[i] as usize * n + k] as f64;
                    let gone = ra.gone[ord[i] as usize * n + k] as f64;
                    let gap = if i + 1 < m { ts[i + 1] - ts[i] } else { ts[0] + DAY - ts[i] };
                    let p = period_at(ts[i]);
                    // T-1255: a rider who went another way counts the wait they would not take
                    let w = q * gap.max(0.0) + gone * WAIT_FOR_ROOM_S;
                    out[p][4] += w;
                    out[p][6] += w * km(p, k);
                    out[p][8] += gone;
                    here[p] += w;
                }
                // T-1255: the platforms where trains come full, those whose riders wait
                // `BUSY_PLATFORM_S` or more on average
                for (p, o) in out.iter_mut().enumerate() {
                    let came = ra.arrivals[k].per[p];
                    if came > 0.0 && here[p] >= BUSY_PLATFORM_S * came {
                        o[9] += here[p];
                        o[10] += came;
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
            let Some(Some(acc)) = self.accounts[self.kind_at(t, l)].get(l) else { continue };
            let n = acc.runs[0].n;
            let mut w = Tally { stops: vec![0.0f64; n], ..Default::default() };
            if past {
                if let Some(Some(ya)) = self.accounts[self.kind_at(t - DAY, l)].get(l) {
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
            // T-1255: the crowding at the platforms where trains come full, every rider alike (the
            // km-weighted pair stays for the layout)
            out.extend([c[9], c[10], c[5], 0.0, 0.0]);
            count += 1;
        }
        out[0] = count as f64;
        out
    }

    /// Add one account's departures with time of day in (`from`, `to`] to `w`; full trains' times
    /// are shifted by `off`. A stop's boardings telescope: the riders who have arrived by its last
    /// departure in the window, less those still waiting after it, less the same at the departure
    /// before the window (the day's last, a day earlier, when `from` is before the first), less
    /// those who went another way in the window.
    /// `day0` is the local time of the window's midnight.
    fn tally(&self, acc: &LineAccount, ls: &LineSvc, from: f64, to: f64, off: f64, day0: f64, w: &mut Tally) {
        let cap = ls.crush();
        let n = w.stops.len();
        for (r, ra) in acc.runs.iter().enumerate() {
            let m = ra.deps.len();
            if m == 0 || ra.n != n {
                continue;
            }
            // T-680: nothing ran before the line's trains started
            let open = ra.open - day0;
            let line_from = if open > from { open - 1e-6 } else { from };
            for k in 0..n {
                // T-848: a stop along the line opens with its first train
                let open = ra.stop_open.get(k).map_or(open, |o| o - day0);
                let from = if open > from { open - 1e-6 } else { from };
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
                    // T-1067: those who went another way never boarded
                    b -= ra.gone[ord[j] as usize * n + k] as f64;
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
                    let from = ra.stop_open.get(k).map_or(line_from, |o| if o - day0 > line_from { o - day0 - 1e-6 } else { line_from });
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
        let ra = &self.accounts[self.kind_at(dep, line as usize)].get(line as usize)?.as_ref()?.runs[run];
        let i = ra.find(dep)?;
        Some(ra.load[i * ra.n..(i + 1) * ra.n].to_vec())
    }
}

pub const QUEUE_STRIDE: usize = 12;
/// Numbers per period in `Boarding::line_periods` (T-491).
pub const PERIOD_STRIDE: usize = 11;

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

    /// T-1125, T-1152: per stretch, the share of the day's trains leaving its first stop packed to
    /// crush (`cap`).
    pub fn full_shares(&self, cap: f64) -> Vec<f32> {
        let (n, m) = (self.n, self.deps.len());
        let mut out = vec![0.0f32; n.saturating_sub(1)];
        if m == 0 {
            return out;
        }
        for (k, f) in out.iter_mut().enumerate() {
            let full = (0..m).filter(|&j| self.load[self.order[k * m + j] as usize * n + k] as f64 >= cap - 0.5).count();
            *f = full as f32 / m as f32;
        }
        out
    }

    /// T-1160: per stretch, the rider-seconds waited at its first stop after full trains (each
    /// rider left behind waits the gap to the next departure, as `Boarding::crowding` counts them)
    /// over the riders arriving there.
    pub fn stop_waits(&self) -> Vec<f32> {
        let (n, m) = (self.n, self.deps.len());
        let mut out = vec![0.0f32; n.saturating_sub(1)];
        if m == 0 {
            return out;
        }
        for (k, w) in out.iter_mut().enumerate() {
            let riders = self.arrivals.get(k).map_or(0.0, |a| a.day);
            if !(riders > 0.0) {
                continue;
            }
            let ts = &self.order_t[k * m..(k + 1) * m];
            let ord = &self.order[k * m..(k + 1) * m];
            let mut waited = 0.0f64;
            for i in 0..m {
                let q = self.q_after[ord[i] as usize * n + k] as f64;
                if q > 0.0 {
                    let gap = if i + 1 < m { ts[i + 1] - ts[i] } else { ts[0] + DAY - ts[i] };
                    waited += q * gap.max(0.0);
                }
            }
            *w = (waited / riders) as f32;
        }
        out
    }
}

/// Alighting share per period and stop: demand's alightings over the riders it has arriving.
/// On a circle (`wrap`) the last stop lets off its share too; elsewhere everyone.
fn alight_shares(lr: &LineRiders, r: usize, wrap: bool) -> Vec<f64> {
    let n = lr.n;
    let mut out = vec![0.0; PERIODS * n];
    let mut day_in = vec![0.0; n];
    let mut day_off = vec![0.0; n];
    for p in 0..PERIODS {
        let base = (p * 2 + r) * n;
        // a circle's train reaches its first stop with riders from the lap before: at least
        // enough that no stop sees fewer than nobody aboard
        let mut on = 0.0f64;
        if wrap {
            let (mut x, mut low) = (0.0f64, 0.0f64);
            for k in 0..n {
                x -= lr.alight[base + k];
                low = low.min(x);
                x += lr.board[base + k];
            }
            on = -low;
        }
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
        if !wrap {
            out[p * n + n - 1] = 1.0;
        }
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

/// A stretch is marked full (T-1125, T-1160) where the riders arriving at its first stop wait at
/// least this long (s) on average for a train with room: 0.2 min, where the crowding colour
/// (app/src/game/palette.ts `CROWD_STOPS`) leaves green.
pub const FULL_WAIT_MIN_S: f32 = 12.0;

/// T-1255: a platform where trains come full, for the line panel's crowding: its riders wait at
/// least this long (s) on average for a train with room in the period, those turned away counting
/// `WAIT_FOR_ROOM_S` (DECISIONS 2026-10-25).
pub const BUSY_PLATFORM_S: f64 = 60.0;

/// One direction of one line for a day of departures from `START_HOUR`, from empty platforms;
/// with `warm`, from a day earlier, keeping the second day. `leftover` says whether anyone was
/// still waiting when the day ended (then the warm run is the right one).
pub fn simulate(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool) -> RunAccount {
    // T-798: on a circle the riders still aboard at the last stop ride on round: a second pass
    // queues them at the first stop when their train arrives there, ahead of later arrivals
    let wrap = svc.lines[l].kind == crate::track::net::LoopKind::Circle;
    let mut through = vec![];
    let mut acc = simulate_carry(net, svc, l, r, lr, peak, warm, wrap, &[], &mut through);
    if wrap && !through.is_empty() {
        through.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sum = 0.0;
        for x in &mut through {
            sum += x.1;
            x.1 = sum;
        }
        acc = simulate_carry(net, svc, l, r, lr, peak, warm, wrap, &through, &mut vec![]);
    }
    acc.full = acc.full_shares(svc.lines[l].crush());
    acc.wait = acc.stop_waits();
    acc
}

/// Riders arrived at a platform by a time.
struct Cum<'a>(&'a dyn Fn(f64) -> f64);
impl Cum<'_> {
    fn cum(&self, t: f64) -> f64 {
        (self.0)(t)
    }
}

/// `simulate`, with `carry_in` (time, riders so far, in time order) joining the first stop's queue,
/// and on a circle (`wrap`) the riders staying aboard at the last stop pushed to `carry_out`.
#[allow(clippy::too_many_arguments)]
fn simulate_carry(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool, wrap: bool, carry_in: &[(f64, f64)], carry_out: &mut Vec<(f64, f64)>) -> RunAccount {
    let ls = &svc.lines[l];
    let n = lr.n;
    let cap = ls.crush();
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
    // T-842b: riders arrive while a train is due within their tolerance, by the steady day's
    // departures, as demand prices them
    let deps = svc.lattice(net, l).map(|lat| span::stop_deps(ls, &lat, r)).filter(|d| d.len() == n);
    let arrivals: Vec<Arrivals> = (0..n)
        .map(|k| {
            let per = std::array::from_fn(|p| {
                let i = (p * 2 + r) * n + k;
                lr.board[i].max(lr.come.get(i).copied().unwrap_or(0.0))
            });
            match &deps {
                Some(d) => Arrivals::with_mask(per, peak, span::stop_mask(&d[k])),
                None => Arrivals::new(per, peak),
            }
        })
        .collect();
    let alpha = alight_shares(lr, r, wrap);
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
        gone: vec![0.0; mk * n],
        order: vec![0; mk * n],
        order_t: vec![0.0; mk * n],
        left_h: vec![0.0; n * 24],
        wait_h: vec![0.0; n * 24],
        arrivals: vec![],
        arrived: 0.0,
        boarded: 0.0,
        alighted: 0.0,
        gave_up: 0.0,
        leftover: 0.0,
        open,
        stop_open: vec![f64::NEG_INFINITY; n],
        ride_km,
        full: vec![],
        wait: vec![],
    };
    let mut carry = vec![0.0f64; m];
    let mut order: Vec<usize> = (0..m).collect();
    let mut hist: Vec<(f64, f64)> = Vec::with_capacity(m);
    for k in 0..n {
        order.sort_by(|&a, &b| time(a, k).total_cmp(&time(b, k)));
        // T-848: trains enter at the ends, so a stop along a long line sees its first an hour or
        // more after the opening; its riders come for that train as they would on any day
        let open = match order.get(order.partition_point(|&i| time(i, k) < open)) {
            Some(&i) if open.is_finite() => (time(i, k) - trips[i].hw).max(open),
            _ => open,
        };
        acc.stop_open[k] = open;
        let arrs = &arrivals[k];
        let carried = |t: f64| if k > 0 { 0.0 } else { carry_in[..carry_in.partition_point(|x| x.0 <= t)].last().map_or(0.0, |x| x.1) };
        let arr = Cum(&|t| arrs.cum(t) + carried(t));
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
        // riders turned away by a full platform so far: `hist` and `c_prev` count only those who
        // joined the queue, so it stays one run of arrivals, oldest first
        let mut turned = 0.0f64;
        for &i in &order {
            let t = time(i, k);
            if open > t_prev && open <= t {
                q = 0.0;
                c_prev = arr.cum(open) - turned;
                hist.clear();
                hist.push((open, c_prev));
                head = 0;
            }
            let mut c_now = arr.cum(t) - turned;
            let a = if c_prev.is_nan() { 0.0 } else { (c_now - c_prev).max(0.0) };
            let p = period_at(t);
            let on = carry[i];
            let off = on * alpha[p * n + k];
            let on = on - off;
            let room = (cap - on).max(0.0);
            // T-1067: whoever was waiting when a train left more than GIVE_UP_S ago has been left
            // behind that long, and went (a second's slack: a train exactly GIVE_UP_S on is still
            // taken); then the platform holds whoever this train and the trains due in the next
            // WAIT_FOR_ROOM_S take, each with this one's room (T-1255), at most WAITING_CAP: the
            // latest arrivals are turned away
            let cut = hist.partition_point(|x| x.0 < t - GIVE_UP_S - 1.0);
            let gave = if cut > 0 { (hist[cut - 1].1 - (c_now - q - a)).clamp(0.0, q) } else { 0.0 };
            let gap = if t_prev.is_finite() { t - t_prev } else { trips[i].hw };
            let hold = (room * (1.0 + (WAIT_FOR_ROOM_S / gap.max(1.0) + 1e-9).floor())).min(WAITING_CAP);
            let over = (q - gave + a - hold).clamp(0.0, a);
            turned += over;
            c_now -= over;
            let q_before = q - gave + a - over;
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
            let b = q_before.min(room);
            let load = on + b;
            let q_after = q_before - b;
            carry[i] = if k + 1 < n { load } else { 0.0 };
            if wrap && k + 1 == n && load > 0.0 {
                carry_out.push((t, load));
            }
            if kept(i) {
                let j = slot[i] as usize;
                acc.load[j * n + k] = if k + 1 < n { load as f32 } else { 0.0 };
                acc.q_after[j * n + k] = q_after as f32;
                acc.gone[j * n + k] = (gave + over) as f32;
                let h = ((t.rem_euclid(DAY) / 3600.0).floor() as usize).min(23);
                // the riders who came since the last train and did not get on this one (those
                // waiting before them board first)
                acc.left_h[k * 24 + h] += (a - (b - (q - gave)).max(0.0)).max(0.0) as f32;
                if b > 0.0 || q_before > 0.0 {
                    let w = (t - since).max(0.0) as f32;
                    if w > acc.wait_h[k * 24 + h] {
                        acc.wait_h[k * 24 + h] = w;
                    }
                }
                acc.arrived += a;
                acc.boarded += b;
                acc.gave_up += gave + over;
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
mod tests;
