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
use super::timing::PROFILE_VARS;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Seats, riders at the rated load (100%) and at crush (150%, T-2186) per 20 m car: the demand
/// kernel's `SEATS_PER_CAR`, `RATED_PER_CAR` and `CRUSH_PER_CAR` (sim/src/demand/api.rs).
pub const SEATS_PER_CAR: f64 = 44.0;
pub const RATED_PER_CAR: f64 = 160.0;
pub const CRUSH_PER_CAR: f64 = 240.0;
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
    /// the same for every kind of day's account of the service (T-2707)
    m: Arc<Mask>,
    /// per period, the mask's share over its riders by this platform's shape (0: no train due all
    /// period, and the period's riders arrive as if unmasked)
    norm: [f64; PERIODS],
    /// riders arrived from the start of the day to each of the mask's marks (`span::MARK_EVERY`);
    /// `cum` adds the segments after one in the order this sum did, so its sums are the same to the
    /// bit (T-2707)
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
        Self::with_shared(per, peak, Arc::new(m))
    }
    /// `with_mask` on a mask other platforms' accounts hold too.
    pub fn with_shared(per: [f64; PERIODS], peak: &[f64; PERIODS], m: Arc<Mask>) -> Self {
        let mut a = Self::new(per, peak);
        if m.is_empty() || m.all_one {
            return a;
        }
        // the shares worked out once (a ramp's are a logarithm each)
        let segs: Vec<(f64, f64, f64, bool)> = m.segs_from((0, 0)).collect();
        let mut x = Masked { norm: [0.0; PERIODS], pre: Vec::with_capacity(m.marks), m };
        for &(t, e, s, _) in &segs {
            let p = period_at(t);
            x.norm[p] += s * (a.w(p, e) - a.w(p, t));
        }
        let mut tot = 0.0;
        for &(t, e, s, mark) in &segs {
            if mark {
                x.pre.push(tot);
            }
            tot += a.seg(&x, t, s, t, e);
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
    /// The mask's weight on riders in the segment starting at `t` with share `s`.
    fn eff(x: &Masked, t: f64, s: f64) -> f64 {
        let p = period_at(t);
        if x.norm[p] > 1e-9 { s / x.norm[p] } else { 1.0 }
    }
    /// Riders arriving in that segment between `u0` and `u1` seconds into the day.
    fn seg(&self, x: &Masked, t: f64, s: f64, u0: f64, u1: f64) -> f64 {
        let p = period_at(t);
        self.per[p] * Self::eff(x, t, s) * (self.w(p, u1) - self.w(p, u0))
    }
    /// Riders arrived since the start of game day 0 (negative before it).
    pub fn cum(&self, t: f64) -> f64 {
        let d = (t / DAY).floor();
        let s = t - d * DAY;
        let mut tot = d * self.day;
        if let Some(x) = &self.masked {
            let at = x.m.find(s);
            let (c, from) = x.m.mark_before(at);
            let mut pre = x.pre[c];
            let mut segs = x.m.segs_from(from);
            for _ in from.1..at.1 {
                let (t, e, sh, _) = segs.next().unwrap();
                pre += self.seg(x, t, sh, t, e);
            }
            let (t, _, sh, _) = segs.next().unwrap();
            return tot + pre + self.seg(x, t, sh, t, s);
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
            Some(x) => {
                let (u, sh) = x.m.seg(x.m.find(s));
                r * Self::eff(x, u, sh)
            }
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
    pub vars: Vec<u8>,
    /// The departure times from each stop, from the trip's start, per slot and variant
    /// (`slot * PROFILE_VARS + var`; empty for a variant the run has not). T-3346 stage 2: `vars`,
    /// each departure's timetable variant (`Trip::var`).
    pub stop_dep: Vec<Vec<f64>>,
    /// `[trip * n + stop]`: riders aboard leaving the stop (0 at the last).
    pub load: Vec<f32>,
    /// `[trip * n + stop]`: riders left on the platform as the train leaves (`q_after`), and riders who went another
    /// way then (T-1067: `GIVE_UP_S`, `WAITING_CAP`), never boarding (`gone`).
    pub queues: Queues,
    /// Per stop, the trips in the order they leave it in a day (`stop * trips + k`); `stop_times`
    /// gives the time of day each leaves it.
    pub order: TripOrder,
    /// Per departure, as `deps`, its time leaving the first stop not wrapped into the day: with
    /// `stop_dep_of` the time it leaves each stop (`leaves`). T-3939: worked out where read rather
    /// than kept per trip and stop, a third of the account's memory.
    pub dep0: Vec<f64>,
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
    /// Per stretch (stop k to k + 1 in this run's order) and hour of the day (`k * 24 + h`), the
    /// mean riders aboard the trains leaving its first stop in that hour, 0 with none
    /// (`hour_loads`, T-2236).
    pub hour_load: Vec<f32>,
}

/// T-3939: `RunAccount::queues`, kept only where either number is not 0 (most trains of most lines leave nobody
/// behind): a bit per trip and stop, the count of set bits before each word, and the pairs. Read back to the bit.
#[derive(Clone, Debug, Default)]
pub struct Queues {
    bits: Vec<u64>,
    rank: Vec<u32>,
    pairs: Vec<[f32; 2]>,
}

impl Queues {
    pub fn new(q_after: &[f32], gone: &[f32]) -> Queues {
        let words = q_after.len().div_ceil(64);
        let (mut bits, mut rank, mut pairs) = (vec![0u64; words], Vec::with_capacity(words), vec![]);
        for (w, word) in bits.iter_mut().enumerate() {
            rank.push(pairs.len() as u32);
            for b in 0..64 {
                let i = w * 64 + b;
                let (Some(&q), Some(&g)) = (q_after.get(i), gone.get(i)) else { break };
                if q.to_bits() != 0 || g.to_bits() != 0 {
                    *word |= 1 << b;
                    pairs.push([q, g]);
                }
            }
        }
        pairs.shrink_to_fit();
        Queues { bits, rank, pairs }
    }
    #[inline]
    fn pair(&self, i: usize) -> [f32; 2] {
        let (w, b) = (i / 64, i % 64);
        let word = self.bits.get(w).copied().unwrap_or(0);
        if word >> b & 1 == 0 {
            return [0.0; 2];
        }
        self.pairs[self.rank[w] as usize + (word & ((1u64 << b) - 1)).count_ones() as usize]
    }
    /// Riders left on the platform as trip and stop `i` (`trip * n + stop`) leaves.
    #[inline]
    pub fn q_after(&self, i: usize) -> f32 {
        self.pair(i)[0]
    }
    /// Riders who went another way as it left.
    #[inline]
    pub fn gone(&self, i: usize) -> f32 {
        self.pair(i)[1]
    }
    /// Heap bytes held.
    pub fn heap_bytes(&self) -> usize {
        self.bits.capacity() * 8 + self.rank.capacity() * 4 + self.pairs.capacity() * 8
    }
    /// Trip-and-stop pairs kept (benches).
    pub fn kept(&self) -> usize {
        self.pairs.len()
    }
}

/// `x.rem_euclid(DAY)` to the bit, without the division's library call for a time above 0: `x - k * DAY` is
/// exact there (Sterbenz: `k * DAY` is within a factor of two of `x` once `k` is 1 or more), and a `k` one off
/// is put right by one exact step.
#[inline]
fn day_rem(x: f64) -> f64 {
    // (zeros too: -0.0 keeps its sign)
    if !(x > 0.0) || x >= 1e15 {
        return x.rem_euclid(DAY);
    }
    let mut r = x - (x / DAY).floor() * DAY;
    if r < 0.0 {
        r += DAY;
    } else if r >= DAY {
        r -= DAY;
    }
    r
}

/// T-4184: `RunAccount::order`, each entry a departure's index in the account: 16 bits wide when the run has at most
/// 65,536 departures a day (every line the game makes), else 32.
#[derive(Clone, Debug)]
pub enum TripOrder {
    Short(Vec<u16>),
    Wide(Vec<u32>),
}

impl Default for TripOrder {
    fn default() -> Self {
        TripOrder::Short(vec![])
    }
}

impl TripOrder {
    fn zeros(len: usize, trips: usize) -> TripOrder {
        if trips <= 1 << 16 { TripOrder::Short(vec![0; len]) } else { TripOrder::Wide(vec![0; len]) }
    }
    #[inline]
    pub fn get(&self, i: usize) -> usize {
        match self {
            TripOrder::Short(v) => v[i] as usize,
            TripOrder::Wide(v) => v[i] as usize,
        }
    }
    #[inline]
    pub fn try_get(&self, i: usize) -> Option<usize> {
        match self {
            TripOrder::Short(v) => v.get(i).map(|&x| x as usize),
            TripOrder::Wide(v) => v.get(i).map(|&x| x as usize),
        }
    }
    fn set(&mut self, i: usize, s: u32) {
        match self {
            TripOrder::Short(v) => v[i] = s as u16,
            TripOrder::Wide(v) => v[i] = s,
        }
    }
    fn slice(&self, a: usize, b: usize) -> OrdSlice<'_> {
        match self {
            TripOrder::Short(v) => OrdSlice::Short(&v[a..b]),
            TripOrder::Wide(v) => OrdSlice::Wide(&v[a..b]),
        }
    }
    pub fn heap_bytes(&self) -> usize {
        match self {
            TripOrder::Short(v) => v.capacity() * 2,
            TripOrder::Wide(v) => v.capacity() * 4,
        }
    }
}

/// One stop's part of a `TripOrder`.
#[derive(Clone, Copy)]
pub enum OrdSlice<'a> {
    Short(&'a [u16]),
    Wide(&'a [u32]),
}

impl OrdSlice<'_> {
    #[inline]
    pub fn get(&self, i: usize) -> usize {
        match self {
            OrdSlice::Short(v) => v[i] as usize,
            OrdSlice::Wide(v) => v[i] as usize,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            OrdSlice::Short(v) => v.len(),
            OrdSlice::Wide(v) => v.len(),
        }
    }
}

/// One stop's departures in the order they leave it in a day, sorted by time of day (`RunAccount::stop_times`).
pub struct StopTimes<'a> {
    ra: &'a RunAccount,
    k: usize,
    /// the departures' account indices, in that order
    pub ord: OrdSlice<'a>,
}

impl StopTimes<'_> {
    /// The time of day the `i`th of them leaves.
    #[inline]
    pub fn at(&self, i: usize) -> f64 {
        self.ra.leaves(self.ord.get(i), self.k)
    }
    /// As `slice::partition_point` over the times.
    pub fn partition_point(&self, pred: impl Fn(f64) -> bool) -> usize {
        let (mut lo, mut hi) = (0, self.ord.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.at(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
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
    /// T-4149: this account among every one this `Boarding` made (from 1), so a line's day read from it can be kept
    /// while it stands (`Boarding::window`)
    id: u64,
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
    /// T-4149: accounts made so far (`LineAccount::id`)
    made: u64,
    /// T-4149: per line, its last block of `window` (past day, then today so far) and what it was read from, given again
    /// while all of that stands: after an edit only the lines it touched are read again (`past_day` over every line was
    /// 125-445 ms on a 2,647-station game, asked twice an edit)
    day_memo: RefCell<[Vec<Option<(Vec<u64>, Vec<f64>)>>; 2]>,
    /// T-4149: per line and kind of day, its last `line_periods` and what it was read from (the panels ask every line's
    /// after each edit, `linesPeriods`)
    periods_memo: RefCell<HashMap<(usize, usize), ([u64; 3], Vec<f64>)>>,
    /// where `ensure_some` goes on from: day kind times lines plus line
    cursor: usize,
    /// T-2707: per line id, the service key (`key`) its arrival masks were made for, and per run
    /// the masks (`line_masks`), which every kind of day's account shares
    masks: Vec<Option<(u64, [RunMasks; 2])>>,
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
    /// To the microsecond (or millionth of a train an hour): a plain joint cut into a line's track
    /// moves its times by about 1e-10 s, which must not redo its account (T-1780).
    fn f(&mut self, v: f64) {
        self.add(((v * 1e6).round() as i64) as u64);
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
        self.masks.clear();
        self.xfer.clear();
        *self.day_memo.get_mut() = Default::default();
        self.periods_memo.get_mut().clear();
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
    /// that kind has no riders (an away or coming-back day the Sunday's first: the clock worker sends
    /// a superholiday's riders only while one is near, T-264, T-1373).
    pub fn kind_at(&self, t: f64, line: usize) -> usize {
        let day = (t / DAY).floor() as i64;
        let given = self.holidays.get(&(line as u32)).and_then(|d| d.iter().find(|x| x.0 == day)).map(|x| x.1);
        self.kind(given.unwrap_or_else(|| day_kind_at(t)))
    }
    /// T-3521: which of a line's trains a kind of day's account runs (`Service::run_deps_as`): a
    /// Saturday's and a Sunday's (a public holiday's too) the weekends', every other kind (a
    /// superholiday's days among them) the weekdays' (`rowOfIn` in the app's
    /// game/holidayCalendar.ts gives the days the same way).
    pub fn kind_row(kind: usize) -> usize {
        match kind {
            1 | 2 => WEEKEND,
            _ => WEEKDAYS,
        }
    }

    fn kind(&self, day: usize) -> usize {
        let day = day.min(DAY_KINDS - 1);
        if !self.riders[day].is_empty() {
            day
        } else if (day == AWAY || day == BACK) && !self.riders[2].is_empty() {
            2
        } else {
            0
        }
    }

    /// What a line's account depends on in the service, hashed (the riders: `generation`).
    fn key(net: &Network, svc: &Service, l: usize) -> u64 {
        let ls = &svc.lines[l];
        let mut h = Hash::new();
        h.add(ls.cars as u64);
        h.add(net.line_trains[l].is_some() as u64);
        h.f(net.line_turn[l] as f64);
        // the lattice's waits under a headway (T-2607), which the arrival masks read
        for lev in 0..ROWS {
            h.add(net.line_auto[l].map_or(0, |a| a[lev] as u64 + 1));
        }
        h.f(ls.shift);
        // T-3542: moved hours, which cut the departures elsewhere (nothing at the default hours)
        if net.mix(l).is_some() {
            for c in net.hours(l) { h.add(c as u64 + 1000); }
            h.add(net.rows(l) as u64);
        }
        if ls.is_loop() {
            h.add(ls.kind as u64 + 100 + 10 * ls.one_way as u64);
        }
        for t in net.line_tph[l] {
            h.f(t as f64);
        }
        // T-3521: the weekend rates, which the weekend accounts run
        if net.rest_solved {
            for lev in net.slots() {
                h.f(net.line_tph_on(l, WEEKEND, lev) as f64);
            }
            // T-3735: and where they differ, the weekends' own departures
            if net.runs_own_weekend(l) {
                for p in net.line_phase_rest_of(l as u32).into_iter().chain(net.line_split_rest_of(l as u32)) {
                    h.f(p as f64);
                }
            }
        }
        // T-1925: the departure phases
        for p in net.line_phase_of(l as u32) {
            h.f(p as f64);
        }
        // the patterns before recent changes (T-206): the account's day holds their departures
        for e in svc.eras_of(l) {
            h.f(e.until);
            for lev in net.slots() {
                h.f(e.lat.tph[lev] as f64);
                h.f(e.lat.off[lev]);
                h.f(e.lat.phase[lev]);
            }
            h.f(e.lat.lead);
            h.f(e.lat.shift);
            for v in e.lat.hours.words() { h.add(v); }
        }
        for r in 0..2 {
            for lev in net.slots() {
                let p = &ls.runs[r].prof[lev];
                h.f(p.duration);
                for &v in p.dep.iter() {
                    h.f(v);
                }
                // T-3346 stage 2: each train's own timetable
                let run = &ls.runs[r];
                if !run.vars[lev].is_empty() {
                    h.add(ls.timed.is_some() as u64);
                    for x in &run.vars[lev] {
                        x.prof.dep.iter().for_each(|&v| h.f(v));
                    }
                    run.pattern[lev].iter().for_each(|&(k, v)| h.add((k as u64) << 8 | v as u64));
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
        for day in 0..DAY_KINDS {
            self.accounts[day].resize(nl, None);
            self.accounts[day].truncate(nl);
        }
        // T-2699: the scan goes on from where the last call stopped, wrapping round, so a round of
        // steps hashes each line about once (from the start every call it hashed lines squared);
        // only a call that finds nothing out of date looks at every line. A line's service hash is
        // the same on every kind of day.
        let total = DAY_KINDS * nl;
        let mut svc_key: Vec<Option<u64>> = vec![None; nl];
        let mut done = 0;
        for i in 0..total {
            let at = (self.cursor + i) % total;
            let (day, l) = (at / nl, at % nl);
            let ls = &svc.lines[l];
            let lr = self.riders[day].get(l).and_then(|r| r.as_ref());
            let ok = ls.running && l < net.line_count() && lr.map_or(false, |r| r.n == ls.stops.len() && r.n >= 2);
            if !ok {
                self.accounts[day][l] = None;
                continue;
            }
            let base = *svc_key[l].get_or_insert_with(|| Self::key(net, svc, l));
            let key = {
                let mut h = Hash(base);
                h.add(self.generation[day]);
                h.0
            };
            if self.accounts[day][l].as_ref().map_or(false, |a| a.key == key) {
                continue;
            }
            if done == max_lines {
                self.cursor = at;
                return true;
            }
            done += 1;
            let lr = lr.unwrap();
            if self.masks.len() < nl {
                self.masks.resize(nl, None);
            }
            if self.masks[l].as_ref().map_or(true, |m| m.0 != base) {
                self.masks[l] = Some((base, [0, 1].map(|r| line_masks(net, svc, l, r))));
            }
            let masks = &self.masks[l].as_ref().unwrap().1;
            // One day from the quiet of the night; a platform still waiting at its end means a
            // queue that carries round, so that direction is run again from a day earlier.
            let runs = [0, 1].map(|r| {
                let m = masks[r].as_deref();
                let row = Some(Self::kind_row(day));
                let a = simulate_with(net, svc, l, r, lr, &self.peak, false, m, row);
                if a.leftover > 0.5 { simulate_with(net, svc, l, r, lr, &self.peak, true, m, row) } else { a }
            });
            self.made += 1;
            self.accounts[day][l] = Some(LineAccount { key, id: self.made, shift: ls.shift, runs });
            self.redone += 1;
        }
        // every line seen: no masks kept for a line without an account
        self.masks.truncate(nl);
        for (l, m) in self.masks.iter_mut().enumerate() {
            if self.accounts.iter().all(|a| a[l].is_none()) {
                *m = None;
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
            offs.push(svc.lines[t.line as usize].prof_of(t.run as usize, t.level as usize, t.var).duration as f32);
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
            // a bus stop has no platform (T-1464); a port has (T-2193)
            if ls.is_bus() && !ls.ferry {
                continue;
            }
            for (r, ra) in acc.runs.iter().enumerate() {
                let m = ra.deps.len();
                if m == 0 {
                    continue;
                }
                for k in 0..ra.n {
                    let arr = &ra.arrivals[k];
                    if arr.day <= 0.0 {
                        continue;
                    }
                    let ts = ra.stop_times(k);
                    let first = ts.partition_point(|t| t < h0);
                    let end = ts.partition_point(|t| t < h0 + 3600.0);
                    let start = events.len() / 2;
                    // the last train before the hour, the hour's, and the first after it
                    let at = |j: isize| -> (f64, usize) {
                        let jj = j.rem_euclid(m as isize) as usize;
                        let wrap = j.div_euclid(m as isize) as f64 * DAY;
                        (ts.at(jj) + wrap - h0, ts.ord.get(jj))
                    };
                    // T-1123: each train's arrival too, when its riders get off and on: its stand
                    // there before it leaves (the closing stand at a first stop), never before
                    // the train ahead left
                    let close = ls.close_stand(net, l as u32);
                    let a0 = arrive.len();
                    for j in first as isize - 1..=end as isize {
                        let (t, i) = at(j);
                        events.push(t as f32);
                        events.push(ra.queues.q_after(i * ra.n + k));
                        let stand = if k == 0 { close } else { Some(ls.prof_of(r, ra.levels[i] as usize, ra.vars[i])).and_then(|p| Some(p.dep.get(k)? - p.arr.get(k)?)).unwrap_or(0.0) };
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
                    let Some(node) = ls.stop_node(r, k) else { continue };
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
        // T-2236: the crowded stretches, in run 0's numbering: those whose busiest hour's trains
        // run at `CROWDED_FROM` of the rated load or more, either way; that load as a share of the
        // rated, and the hour of the day it is in (the line's local time)
        let (mut full, mut peaks, mut hours): (Vec<f32>, Vec<f32>, Vec<f32>) = (vec![], vec![], vec![]);
        for l in 0..nl {
            let day = self.kind_at(epoch + shift(l), l);
            let (Some(Some(acc)), Some(ls)) = (self.accounts[day].get(l), svc.lines.get(l)) else { continue };
            let sh = Self::stretch_hours(acc, ls);
            for (s, by_hour) in sh.chunks(24).enumerate() {
                let (h, &v) = by_hour.iter().enumerate().fold((0, &0.0), |b, x| if *x.1 > *b.1 { x } else { b });
                if v >= CROWDED_FROM {
                    full.extend([l as f32, s as f32]);
                    peaks.push(v as f32);
                    hours.push(h as f32);
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
        out.extend_from_slice(&peaks);
        out.extend_from_slice(&hours);
        out
    }

    /// T-2236: per stretch in run 0's numbering and hour of the day (`t * 24 + h`), the mean load
    /// of the trains leaving its first stop in the hour as a share of the rated load, the fuller
    /// way's.
    fn stretch_hours(acc: &LineAccount, ls: &LineSvc) -> Vec<f64> {
        let n = acc.runs.first().map_or(0, |r| r.n);
        let s = n.saturating_sub(1);
        let rated = ls.rated();
        let mut out = vec![0.0f64; s * 24];
        for (r, ra) in acc.runs.iter().enumerate() {
            if ra.n != n || ra.hour_load.len() != s * 24 {
                continue;
            }
            for k in 0..s {
                let t = if r == 0 { k } else { s - 1 - k };
                for h in 0..24 {
                    let v = ra.hour_load[k * 24 + h] as f64 / rated;
                    if v > out[t * 24 + h] {
                        out[t * 24 + h] = v;
                    }
                }
            }
        }
        out
    }

    /// T-2236: per period and stretch (`p * (n - 1) + t`), the busiest hour's load in the period
    /// as a share of the rated load (`stretch_hours`).
    fn period_peaks(acc: &LineAccount, ls: &LineSvc) -> Vec<f64> {
        let sh = Self::stretch_hours(acc, ls);
        let s = sh.len() / 24;
        let mut out = vec![0.0f64; PERIODS * s];
        for t in 0..s {
            for h in 0..24 {
                let p = period_at(h as f64 * 3600.0);
                out[p * s + t] = out[p * s + t].max(sh[t * 24 + h]);
            }
        }
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
            let Some(fls) = svc.lines.get(fl).filter(|x| !x.is_bus() || x.ferry) else { continue };
            let fe = epoch + fls.shift;
            let Some(Some(acc)) = self.accounts[self.kind_at(fe, fl)].get(fl) else { continue };
            let ra = &acc.runs[(f.run as usize).min(1)];
            let (m, n, kf) = (ra.deps.len(), ra.n, f.stop as usize);
            // nobody arrives at a first stop
            if m == 0 || kf == 0 || kf >= n {
                continue;
            }
            let run = &fls.runs[(f.run as usize).min(1)];
            let Some(fnode) = fls.stop_node((f.run as usize).min(1), kf) else { continue };
            let fnode = fnode as usize;
            let d = (net.node_x[fnode] - x0).hypot(net.node_y[fnode] - y0) / net.k_at(y0);
            let walk = d * XFER_DETOUR / XFER_MPS;
            let ts = ra.stop_times(kf);
            let ord = ts.ord;
            let h0 = fe.rem_euclid(DAY);
            // T-1124: riders get off as the train stops, its stand before it leaves
            let stand = |i: usize| Some(fls.prof_of((f.run as usize).min(1), ra.levels[i] as usize, ra.vars[i])).and_then(|p| Some(p.dep.get(kf)? - p.arr.get(kf)?)).unwrap_or(0.0).max(0.0);
            let most = run.prof.iter().chain(run.vars.iter().flatten().map(|x| &x.prof)).filter_map(|p| Some(p.dep.get(kf)? - p.arr.get(kf)?)).fold(0.0f64, f64::max);
            for a in [-3600.0, 0.0] {
                // the trains whose riders reach `q` in [a, a + 3600), weighed by riders aboard
                trains.clear();
                let (lo, hi) = (h0 + a - walk, h0 + a + 3600.0 - walk);
                for d in ((lo / DAY).floor() as i64)..=(((hi + most) / DAY).floor() as i64) {
                    let off = d as f64 * DAY;
                    let (i0, i1) = (ts.partition_point(|t| t + off < lo), ts.partition_point(|t| t + off < hi + most));
                    for j in i0..i1 {
                        let reach = ts.at(j) + off - stand(ord.get(j)) - h0 + walk;
                        let w = ra.load[ord.get(j) * n + kf - 1] as f64;
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
                    let off = ra.stop_dep_of(j).get(at).copied().unwrap_or(0.0);
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
    /// (`crowding`, T-491), 0, and the busiest hour's load in the period on the fullest stretch as
    /// a share of the rated load (`period_peaks`, T-2236).
    ///
    /// A stop's boardings since midnight telescope: the riders who have arrived by its last
    /// departure, less those still waiting after it, less the same at its last departure before
    /// midnight, less those who went another way (`tally`). Riders left behind are counted per departure as in `simulate` (who arrived since
    /// the train before and is still waiting).
    pub fn today(&self, svc: &Service, t: f64) -> Vec<f64> {
        self.window(svc, t, false, true)
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
                    ls.round_trip[lv] / ls.live_runs() as f64
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
    /// likewise (T-1255), then per period the rider-seconds waited after full trains at the first
    /// stop of each of the `n - 1` stretches, and the riders arriving there, each the fuller way's
    /// (T-1926), then per period the busiest hour's load on each stretch as a share of the rated
    /// load (`period_peaks`, T-2236: the line panel's crowding). Empty without an account.
    pub fn line_periods(&self, svc: &Service, l: usize, day: usize) -> Vec<f64> {
        let kind = self.kind(day);
        let Some(Some(acc)) = self.accounts[kind].get(l) else { return vec![] };
        let Some(ls) = svc.lines.get(l) else { return vec![] };
        // T-4149: kept while its account (made once, never changed) and the loads it is measured against stand
        let key = [acc.id, ls.crush().to_bits(), ls.rated().to_bits()];
        if let Some((k, v)) = self.periods_memo.borrow().get(&(l, kind)) {
            if *k == key {
                return v.clone();
            }
        }
        let out = Self::line_periods_of(acc, ls);
        self.periods_memo.borrow_mut().insert((l, kind), (key, out.clone()));
        out
    }

    /// T-4149: `line_periods` worked out, nothing kept used (`?patch=check`).
    pub fn line_periods_whole(&self, svc: &Service, l: usize, day: usize) -> Vec<f64> {
        let Some(Some(acc)) = self.accounts[self.kind(day)].get(l) else { return vec![] };
        let Some(ls) = svc.lines.get(l) else { return vec![] };
        Self::line_periods_of(acc, ls)
    }

    fn line_periods_of(acc: &LineAccount, ls: &LineSvc) -> Vec<f64> {
        let mut out = Self::crowding(acc, ls).concat();
        let n = acc.runs[0].n;
        let mut stretch_here = vec![0.0f64; PERIODS * n.saturating_sub(1)];
        let mut stretch_came = vec![0.0f64; PERIODS * n.saturating_sub(1)];
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
                let ts = ra.stop_times(k);
                let ord = ts.ord;
                let s = if r == 0 { k } else { n - 1 - k };
                let mut here = [0.0f64; PERIODS];
                let t0 = if m > 0 { ts.at(0) } else { 0.0 };
                let mut ti = t0;
                for i in 0..m {
                    let q = ra.queues.q_after(ord.get(i) * n + k) as f64;
                    let gone = ra.queues.gone(ord.get(i) * n + k) as f64;
                    let next = if i + 1 < m { ts.at(i + 1) } else { t0 + DAY };
                    let gap = next - ti;
                    let w = q * gap.max(0.0) + gone * WAIT_FOR_ROOM_S;
                    at[period_at(ti) * n + s] += w;
                    here[period_at(ti)] += w;
                    ti = next;
                }
                // T-1926: the stretch leaving this stop, in run 0's numbering; the fuller way's
                if k + 1 < n {
                    let t = if r == 0 { k } else { n - 2 - k };
                    for p in 0..PERIODS {
                        let c = ra.arrivals[k].per[p];
                        let at_t = p * (n - 1) + t;
                        if c > 0.0 && here[p] * stretch_came[at_t] >= stretch_here[at_t] * c {
                            stretch_here[at_t] = here[p];
                            stretch_came[at_t] = c;
                        }
                    }
                }
            }
        }
        out.extend(at);
        out.extend(came);
        out.extend(stretch_here);
        out.extend(stretch_came);
        out.extend(Self::period_peaks(acc, ls));
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
                let sd = &ra.stop_dep_of(j);
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
                let ts = ra.stop_times(k);
                let ord = ts.ord;
                let mut here = [0.0f64; PERIODS];
                let t0 = ts.at(0);
                let mut ti = t0;
                for i in 0..m {
                    let q = ra.queues.q_after(ord.get(i) * n + k) as f64;
                    let gone = ra.queues.gone(ord.get(i) * n + k) as f64;
                    let next = if i + 1 < m { ts.at(i + 1) } else { t0 + DAY };
                    let gap = next - ti;
                    let p = period_at(ti);
                    ti = next;
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
        self.window(svc, t, true, true)
    }

    /// T-4149: `past_day` read whole, nothing kept used or kept: what the kept blocks must equal (`?patch=check`).
    pub fn past_day_whole(&self, svc: &Service, t: f64) -> Vec<f64> {
        self.window(svc, t, true, false)
    }

    fn window(&self, svc: &Service, t: f64, past: bool, keep: bool) -> Vec<f64> {
        let mut out = vec![0.0];
        let mut count = 0;
        let nl = self.accounts.iter().map(|a| a.len()).max().unwrap_or(0);
        let mut memo = self.day_memo.borrow_mut();
        let memo = &mut memo[past as usize];
        memo.resize(nl, None);
        for l in 0..nl {
            let Some(ls) = svc.lines.get(l) else { continue };
            // the line's own day and time of day (T-309: its local time)
            let t = t + ls.shift;
            let tod = t.rem_euclid(DAY);
            let Some(Some(acc)) = self.accounts[self.kind_at(t, l)].get(l) else { continue };
            // T-4149: what the block reads: the time, the accounts (each made once, never changed), and the service's
            // parts read below
            let ya_id = if past { self.accounts[self.kind_at(t - DAY, l)].get(l).and_then(|a| a.as_ref()).map_or(0, |a| a.id) } else { 0 };
            let key = Self::day_key(ls, t, acc.id, ya_id);
            if let Some((k, block)) = memo[l].as_ref().filter(|_| keep) {
                if *k == key {
                    out.extend_from_slice(block);
                    count += 1;
                    continue;
                }
            }
            let start = out.len();
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
            let p = period_at(tod);
            let c = Self::crowding(acc, ls)[p];
            let pk = Self::period_peaks(acc, ls);
            let s = pk.len() / PERIODS;
            let peak = pk[p * s..(p + 1) * s].iter().fold(0.0f64, |a, &b| a.max(b));
            // T-1255: the crowding at the platforms where trains come full, every rider alike; T-2236:
            // the load past rated in the slot the km-weighted wait had
            out.extend([c[9], c[10], c[5], 0.0, peak]);
            if keep {
                memo[l] = Some((key, out[start..].to_vec()));
            }
            count += 1;
        }
        out[0] = count as f64;
        out
    }

    /// T-4149: what line `ls`'s block of `window` at its local time `t` is read from besides its accounts: the service's
    /// parts `tally`, `under_way`, `crowding` and `period_peaks` read, as their bits.
    fn day_key(ls: &LineSvc, t: f64, acc: u64, ya: u64) -> Vec<u64> {
        let mut k = vec![t.to_bits(), acc, ya, ls.shift.to_bits(), ls.crush().to_bits(), ls.rated().to_bits(), ls.is_loop() as u64, ls.live_runs() as u64];
        k.extend(ls.round_trip.iter().map(|v| v.to_bits()));
        for r in &ls.runs {
            k.extend(r.prof.iter().map(|p| p.duration.to_bits()));
        }
        k
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
                let ts = ra.stop_times(k);
                let ord = ts.ord;
                let q = |j: usize| ra.queues.q_after(ord.get(j) * n + k) as f64;
                let lo = ts.partition_point(|s| s <= from);
                let done = ts.partition_point(|s| s <= to);
                if done <= lo {
                    continue;
                }
                let (t_before, q_before) = if lo > 0 { (ts.at(lo - 1), q(lo - 1)) } else { (ts.at(m - 1) - DAY, q(m - 1)) };
                let last = done - 1;
                // T-1815: each departure's `cum` is the next one's `cum(prev)`, worked out once
                let c_before = arr.cum(t_before);
                let mut b = (arr.cum(ts.at(last)) - q(last)) - (c_before - q_before);
                let (mut prev, mut q_prev, mut c_prev) = (t_before, q_before, c_before);
                for j in lo..done {
                    let tj = ts.at(j);
                    let c_j = arr.cum(tj);
                    let mut a = (c_j - c_prev).max(0.0);
                    if prev < open && open <= tj {
                        // whoever was waiting then never was: the platform starts empty
                        b -= q_prev + arr.cum(open) - c_prev;
                        a = (c_j - arr.cum(open)).max(0.0);
                    }
                    w.left += a.min(q(j));
                    // T-1067: those who went another way never boarded
                    b -= ra.queues.gone(ord.get(j) * n + k) as f64;
                    (prev, q_prev, c_prev) = (tj, q(j), c_j);
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
                    let o = ra.stop_dep_of(j).get(k).copied().unwrap_or(0.0);
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

    /// T-2707: heap bytes held, by structure (benches).
    pub fn mem_bytes(&self) -> Vec<(&'static str, usize)> {
        use std::mem::size_of;
        let v = |c: usize, s: usize| c * s;
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<(&'static str, usize)> = vec![];
        let mut add = |k: &'static str, b: usize| match out.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += b,
            None => out.push((k, b)),
        };
        for d in 0..DAY_KINDS {
            add("riders", v(self.riders[d].capacity(), size_of::<Option<LineRiders>>()));
            for lr in self.riders[d].iter().flatten() {
                add("riders", v(lr.board.capacity() + lr.alight.capacity() + lr.come.capacity(), 8));
            }
            add("accounts", v(self.accounts[d].capacity(), size_of::<Option<LineAccount>>()));
            for acc in self.accounts[d].iter().flatten() {
                for ra in &acc.runs {
                    add("deps, levels, stop_dep", v(ra.deps.capacity(), 8) + 2 * ra.levels.capacity() + ra.stop_dep.iter().map(|x| x.capacity() * 8).sum::<usize>());
                    add("load", v(ra.load.capacity(), 4));
                    add("queues", ra.queues.heap_bytes());
                    add("order, dep0", ra.order.heap_bytes() + v(ra.dep0.capacity(), 8));
                    add("left_h, wait_h, hour_load", v(ra.left_h.capacity() + ra.wait_h.capacity() + ra.hour_load.capacity(), 4));
                    add("stop_open, ride_km", v(ra.stop_open.capacity() + ra.ride_km.capacity(), 8));
                    add("arrivals", v(ra.arrivals.capacity(), size_of::<Arrivals>()));
                    for a in &ra.arrivals {
                        if let Some(x) = &a.masked {
                            if seen.insert(Arc::as_ptr(&x.m)) {
                                add("mask segments", size_of::<Mask>() + x.m.heap_bytes());
                            }
                            add("mask pre", size_of::<Masked>() + v(x.pre.capacity(), 8));
                        }
                    }
                }
            }
        }
        add("masks kept", v(self.masks.capacity(), size_of::<Option<(u64, [RunMasks; 2])>>()));
        for runs in self.masks.iter().flatten() {
            for m in runs.1.iter().flatten().flatten() {
                add("masks kept", size_of::<Arc<Mask>>());
                if seen.insert(Arc::as_ptr(m)) {
                    add("mask segments", size_of::<Mask>() + m.heap_bytes());
                }
            }
        }
        out
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
    /// Departure `j`'s times leaving each stop, from its start (its variant's, T-3346 stage 2).
    pub fn stop_dep_of(&self, j: usize) -> &[f64] {
        self.stop_dep.get(self.levels[j] as usize * PROFILE_VARS + self.vars.get(j).copied().unwrap_or(0) as usize).map_or(&[], |v| &v[..])
    }

    /// The time of day departure `j` leaves stop `k` (the same sum `simulate_carry` orders by).
    #[inline]
    pub fn leaves(&self, j: usize, k: usize) -> f64 {
        day_rem(self.dep0[j] + self.stop_dep_of(j).get(k).copied().unwrap_or(0.0))
    }

    /// Stop `k`'s departures in the order they leave it in a day.
    pub fn stop_times(&self, k: usize) -> StopTimes<'_> {
        let m = self.deps.len();
        StopTimes { ra: self, k, ord: self.order.slice(k * m, (k + 1) * m) }
    }

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

    /// T-2236: per stretch and hour of the day (`k * 24 + h`), the mean riders aboard the trains
    /// leaving its first stop in that hour, 0 with none.
    pub fn hour_loads(&self) -> Vec<f32> {
        let (n, m) = (self.n, self.deps.len());
        let s = n.saturating_sub(1);
        let (mut sum, mut trains) = (vec![0.0f64; s * 24], vec![0u32; s * 24]);
        for k in 0..s {
            for i in 0..m {
                let Some(j) = self.order.try_get(k * m + i) else { continue };
                let t = self.leaves(j as usize, k);
                let h = ((t.rem_euclid(DAY) / 3600.0) as usize).min(23);
                sum[k * 24 + h] += self.load.get(j as usize * n + k).copied().unwrap_or(0.0) as f64;
                trains[k * 24 + h] += 1;
            }
        }
        sum.iter().zip(&trains).map(|(&a, &c)| if c > 0 { (a / c as f64) as f32 } else { 0.0 }).collect()
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

/// T-2236: a stretch is marked crowded where its busiest hour's trains run at this share of the
/// rated load or more on average: past 100% as the hover rounds it.
pub const CROWDED_FROM: f64 = 1.005;

/// T-1255: a platform where trains come full, for the line panel's crowding: its riders wait at
/// least this long (s) on average for a train with room in the period, those turned away counting
/// `WAIT_FOR_ROOM_S` (DECISIONS 2026-10-25).
pub const BUSY_PLATFORM_S: f64 = 60.0;

/// One direction of one line for a day of departures from `START_HOUR`, from empty platforms;
/// with `warm`, from a day earlier, keeping the second day. `leftover` says whether anyone was
/// still waiting when the day ended (then the warm run is the right one).
#[allow(clippy::too_many_arguments)]
pub fn simulate(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool) -> RunAccount {
    simulate_with(net, svc, l, r, lr, peak, warm, line_masks(net, svc, l, r).as_deref(), None)
}

/// Per stop of run `r` (in its travel order), when riders arrive by the tolerance (T-842b): by
/// the steady day's departures, as demand prices them; None for a line without a steady pattern.
pub type RunMasks = Option<Vec<Arc<Mask>>>;
pub fn line_masks(net: &Network, svc: &Service, l: usize, r: usize) -> RunMasks {
    let ls = &svc.lines[l];
    let deps = span::stop_deps(ls, &svc.lattice(net, l)?, r);
    Some(deps.iter().map(|d| Arc::new(span::stop_mask(d))).collect())
}

/// `simulate` with each stop's arrival mask (`line_masks`), and the trains of kind of day `row` when
/// given (`Service::run_deps_as`), else the line's calendar's.
#[allow(clippy::too_many_arguments)]
pub fn simulate_with(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool, masks: Option<&[Arc<Mask>]>, row: Option<usize>) -> RunAccount {
    // T-798: on a circle the riders still aboard at the last stop ride on round: a second pass
    // queues them at the first stop when their train arrives there, ahead of later arrivals
    let wrap = svc.lines[l].kind == crate::track::net::LoopKind::Circle;
    let masks = masks.filter(|m| m.len() == lr.n);
    let mut through = vec![];
    let mut acc = simulate_carry(net, svc, l, r, lr, peak, warm, wrap, masks, row, &[], &mut through);
    if wrap && !through.is_empty() {
        through.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sum = 0.0;
        for x in &mut through {
            sum += x.1;
            x.1 = sum;
        }
        acc = simulate_carry(net, svc, l, r, lr, peak, warm, wrap, masks, row, &through, &mut vec![]);
    }
    acc.hour_load = acc.hour_loads();
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
fn simulate_carry(net: &Network, svc: &Service, l: usize, r: usize, lr: &LineRiders, peak: &[f64; PERIODS], warm: bool, wrap: bool, masks: Option<&[Arc<Mask>]>, row: Option<usize>, carry_in: &[(f64, f64)], carry_out: &mut Vec<(f64, f64)>) -> RunAccount {
    let ls = &svc.lines[l];
    let n = lr.n;
    let cap = ls.crush();
    let t0 = svc.account_start(l, START_HOUR);
    let from = if warm { t0 - DAY } else { t0 };
    let mut trips: Vec<Trip> = vec![];
    svc.run_deps_as(net, l, r, from, t0 + DAY, row, &mut trips);
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
    // T-3346 stage 2: each trip's own timetable
    let stop_dep = |t: &Trip| &ls.prof_of(r, t.level as usize, t.var).dep;
    let arrivals: Vec<Arrivals> = (0..n)
        .map(|k| {
            let per = std::array::from_fn(|p| {
                let i = (p * 2 + r) * n + k;
                lr.board[i].max(lr.come.get(i).copied().unwrap_or(0.0))
            });
            match masks {
                Some(m) => Arrivals::with_shared(per, peak, m[k].clone()),
                None => Arrivals::new(per, peak),
            }
        })
        .collect();
    let alpha = alight_shares(lr, r, wrap);
    let ride_km = ride_km(&alpha, &ls.runs[r].stop_s, n);
    let time = |i: usize, k: usize| trips[i].dep + stop_dep(&trips[i]).get(k).copied().unwrap_or(0.0);
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
        vars: keep.iter().map(|&i| trips[i].var).collect(),
        stop_dep: (0..SLOTS * PROFILE_VARS).map(|x| { let (lev, v) = (x / PROFILE_VARS, x % PROFILE_VARS); if v == 0 || v <= ls.runs[r].vars[lev].len() { ls.prof_of(r, lev, v as u8).dep.clone() } else { vec![] } }).collect(),
        load: vec![0.0; mk * n],
        queues: Queues::default(),
        order: TripOrder::zeros(mk * n, mk),
        dep0: keep.iter().map(|&i| trips[i].dep).collect(),
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
        hour_load: vec![],
    };
    let (mut q_kept, mut gone_kept) = (vec![0.0f32; mk * n], vec![0.0f32; mk * n]);
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
                q_kept[j * n + k] = q_after as f32;
                gone_kept[j * n + k] = (gave + over) as f32;
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
        for (j, (_, s)) in day.into_iter().enumerate() {
            acc.order.set(k * mk + j, s);
        }
    }
    acc.arrivals = arrivals;
    acc.queues = Queues::new(&q_kept, &gone_kept);
    acc
}

#[cfg(test)]
mod tests;
