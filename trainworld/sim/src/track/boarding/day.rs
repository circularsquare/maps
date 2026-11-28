//! T-4281: each line's block of `Boarding::window` (the panels' past 24 hours, or today so far) kept and moved along as
//! the clock runs. A whole read goes over every departure at every stop (about 0.1 s on a 3,400-stop game, asked every
//! second or so while the clock runs); moved along, a read costs the departures the clock passed since the last one,
//! plus a pass over the line's stops and trips.
//!
//! The window is the rest of yesterday from the time of day now, read from yesterday's kind of day, then today so far.
//! As the clock runs, each stop's departures leave yesterday's part and join today's in the order they leave it
//! (`RunAccount::stop_times`), so each stop keeps its bounds in that order and the sums over them, and each trip keeps
//! its stops in the window and the first of them (by stop) with its most riders aboard.

use super::{period_at, Boarding, LineAccount, RunAccount, DAY, PERIODS};
use crate::track::service::{LineSvc, Service};

/// A trip with no stop in the window carrying riders.
const NONE: u32 = u32::MAX;

/// One trip of a run in one part of the window: its stops in it, and the first of them (by stop) with the most riders
/// aboard, while any has riders.
#[derive(Clone, Copy, Debug)]
struct Trip {
    best: f32,
    at: u32,
    count: u32,
}

/// One stop of a run in one part of the window: its departures `lo..done` in the order they leave it, and the sums
/// `tally` reads over them.
#[derive(Clone, Debug, Default)]
struct Stop {
    lo: u32,
    done: u32,
    /// riders arrived by the window's last departure less those still waiting after it, and the same at the departure
    /// before the window
    x: f64,
    y: f64,
    /// of the window's departures, the riders who went another way and those left behind: today's summed from its
    /// first departure, yesterday's from the day's last
    gone: f64,
    left: f64,
    /// whoever was waiting when the line opened in the window never was (T-680)
    open: f64,
    /// yesterday's: each departure from the window's first on with riders gone or left, and those summed from it to the
    /// day's last; `next` is the first still in the window
    tail: Vec<(u32, f64, f64)>,
    next: u32,
}

impl Stop {
    fn boarded(&self) -> f64 {
        (((self.x - self.y) - self.gone) - self.open).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct Run {
    r: usize,
    stops: Vec<Stop>,
    trips: Vec<Trip>,
}

/// Yesterday's part (`past`) or today's, of one account.
#[derive(Clone, Debug)]
struct Part {
    past: bool,
    runs: Vec<Run>,
}

/// One line's block of `window`, with what it reads and where it stands.
#[derive(Clone, Debug, Default)]
pub(super) struct DayKept {
    /// what the block reads besides the time of day: the local day, the accounts, the service's parts (`day_key`)
    key: Vec<u64>,
    /// the local time of day it stands at
    tod: f64,
    /// no opening (`RunAccount::open`, `stop_open`) within a day of either part: only the clock moves its bounds
    simple: bool,
    parts: Vec<Part>,
    /// per run of today's account, each departure's time out in service (by its level) and whether it is packed full
    /// on its trip (`under_way`)
    span: [Vec<f64>; 2],
    packed: [Vec<bool>; 2],
    /// per period, the crowding numbers the block shows and the busiest stretch's hour (`crowding`, `period_peaks`),
    /// with the account and loads they were read from
    crowd: Vec<[f64; 4]>,
    crowd_key: [u64; 3],
    pub(super) block: Vec<f64>,
}

/// What line `ls`'s block reads besides the time of day (T-4149: the service's parts `tally`, `under_way`, `crowding`
/// and `period_peaks` read, as their bits) and the accounts, on the local day starting at `day0`.
pub(super) fn day_key(ls: &LineSvc, day0: f64, acc: u64, ya: u64) -> Vec<u64> {
    let mut k = vec![day0.to_bits(), acc, ya, ls.shift.to_bits(), ls.crush().to_bits(), ls.rated().to_bits(), ls.is_loop() as u64, ls.live_runs() as u64];
    k.extend(ls.round_trip.iter().map(|v| v.to_bits()));
    for r in &ls.runs {
        k.extend(r.prof.iter().map(|p| p.duration.to_bits()));
    }
    k
}

impl DayKept {
    /// Line `l`'s block read whole at its local time `t` (time of day `tod`) from today's account `acc` and, for the past
    /// 24 hours, yesterday's `ya`. `was`: the line's last, whose crowding is taken over if read from the same.
    pub(super) fn read(l: usize, ls: &LineSvc, acc: &LineAccount, ya: Option<&LineAccount>, t: f64, tod: f64, key: Vec<u64>, was: Option<DayKept>) -> DayKept {
        let n = acc.runs[0].n;
        let day0 = t - tod;
        let mut parts = vec![];
        if let Some(ya) = ya.filter(|ya| ya.runs[0].n == n) {
            parts.push(Part::read(ya, n, tod, DAY, day0 - DAY, true));
        }
        parts.push(Part::read(acc, n, f64::NEG_INFINITY, tod, day0, false));
        // an opening counts only within a day before the window's (`tally`'s `prev` is never before -DAY)
        let far = |a: &LineAccount, d0: f64| a.runs.iter().all(|ra| ra.open - d0 < -DAY && ra.stop_open.iter().all(|&o| o - d0 < -DAY));
        let simple = far(acc, day0) && ya.is_none_or(|ya| far(ya, day0 - DAY));
        let cap = ls.crush();
        let mut span: [Vec<f64>; 2] = Default::default();
        let mut packed: [Vec<bool>; 2] = Default::default();
        for (r, ra) in acc.runs.iter().enumerate() {
            let n = ra.n;
            for &lv in &ra.trains.levels {
                let lv = lv as usize;
                if span[r].len() <= lv {
                    span[r].resize(lv + 1, f64::NAN);
                }
                // a loop's train is out until the next one of its run could leave (T-798)
                span[r][lv] = if ls.is_loop() {
                    ls.round_trip[lv] / ls.live_runs() as f64
                } else {
                    ls.runs[r].prof[lv].duration + ((ls.round_trip[lv] - ls.runs[0].prof[lv].duration - ls.runs[1].prof[lv].duration) / 2.0).max(0.0)
                };
            }
            packed[r] = (0..ra.trains.deps.len()).map(|j| ra.load[j * n..(j + 1) * n].iter().any(|&v| v as f64 >= cap - 0.5)).collect();
        }
        let crowd_key = [acc.id, cap.to_bits(), ls.rated().to_bits()];
        let crowd = match was {
            Some(w) if w.crowd_key == crowd_key && w.crowd.len() == PERIODS => w.crowd,
            _ => {
                let c = Boarding::crowding(acc, ls);
                let pk = Boarding::period_peaks(acc, ls);
                let s = pk.len() / PERIODS;
                (0..PERIODS).map(|p| [c[p][9], c[p][10], c[p][5], pk[p * s..(p + 1) * s].iter().fold(0.0f64, |a, &b| a.max(b))]).collect()
            }
        };
        let mut d = DayKept { key, tod, simple, parts, span, packed, crowd, crowd_key, block: vec![] };
        d.block = d.block(l, n, acc, ya, cap);
        d
    }

    pub(super) fn key(&self) -> &[u64] {
        &self.key
    }

    /// Heap bytes held (benches, `Boarding::mem_bytes`).
    pub(super) fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        let runs = self.parts.iter().flat_map(|p| &p.runs);
        let parts = self.parts.capacity() * size_of::<Part>() + self.parts.iter().map(|p| p.runs.capacity() * size_of::<Run>()).sum::<usize>();
        let runs: usize = runs.map(|r| r.stops.capacity() * size_of::<Stop>() + r.stops.iter().map(|s| s.tail.capacity() * size_of::<(u32, f64, f64)>()).sum::<usize>() + r.trips.capacity() * size_of::<Trip>()).sum();
        let under: usize = self.span.iter().map(|s| s.capacity() * 8).sum::<usize>() + self.packed.iter().map(|p| p.capacity()).sum::<usize>();
        self.key.capacity() * 8 + parts + runs + under + self.crowd.capacity() * size_of::<[f64; 4]>() + self.block.capacity() * 8
    }

    /// Move the block on to time of day `tod` of the same day, from the same accounts; false when it cannot (an opening
    /// within reach, or the clock went back), and it is then read whole.
    pub(super) fn advance(&mut self, l: usize, ls: &LineSvc, acc: &LineAccount, ya: Option<&LineAccount>, tod: f64) -> bool {
        if !self.simple || !(tod >= self.tod) {
            return false;
        }
        if tod.to_bits() == self.tod.to_bits() {
            return true;
        }
        let n = acc.runs[0].n;
        for part in &mut self.parts {
            let a = if part.past { ya.expect("yesterday's part reads yesterday's account") } else { acc };
            for run in &mut part.runs {
                run.advance(&a.runs[run.r], part.past, tod);
            }
        }
        self.tod = tod;
        self.block = self.block(l, n, acc, ya, ls.crush());
        true
    }

    /// The block (`Boarding::today` has the layout).
    fn block(&self, l: usize, n: usize, acc: &LineAccount, ya: Option<&LineAccount>, cap: f64) -> Vec<f64> {
        let mut stops = vec![0.0f64; n];
        let (mut boarded, mut left, mut fullest, mut trains) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut full: Vec<f64> = vec![];
        for part in &self.parts {
            let (a, off) = if part.past { (ya.expect("yesterday's part reads yesterday's account"), -DAY) } else { (acc, 0.0) };
            for run in &part.runs {
                for (k, st) in run.stops.iter().enumerate() {
                    if st.done <= st.lo {
                        continue;
                    }
                    let b = st.boarded();
                    stops[if run.r == 0 { k } else { n - 1 - k }] += b;
                    boarded += b;
                    left += st.left;
                }
                // trains: the fullest stretch each has reached in the window (a trip from before midnight counts at the
                // stops it reaches after it)
                let ra = &a.runs[run.r];
                for (j, tr) in run.trips.iter().enumerate() {
                    let best = tr.best as f64;
                    fullest = fullest.max(best / cap);
                    if tr.count > 0 {
                        trains += 1.0;
                    }
                    if best >= cap - 0.5 && tr.at != NONE {
                        full.push(ra.leaves(j, tr.at as usize) + off);
                    }
                }
            }
        }
        full.sort_by(f64::total_cmp);
        let mut spans: Vec<(f64, f64)> = vec![];
        for &s in &full {
            match spans.last_mut() {
                Some(p) if s - p.1 <= 1800.0 => p.1 = s,
                _ => spans.push((s, s)),
            }
        }
        let mut out = vec![l as f64, n as f64, boarded, full.len() as f64, left, fullest, spans.len() as f64];
        for (a, b) in spans {
            out.extend([a, b]);
        }
        out.extend(stops);
        let (on, packed) = self.under_way(acc);
        out.extend([on, packed, trains]);
        // T-1255: the crowding at the platforms where trains come full, every rider alike; T-2236: the load past rated
        // in the slot the km-weighted wait had
        let c = self.crowd[period_at(self.tod)];
        out.extend([c[0], c[1], c[2], 0.0, c[3]]);
        out
    }

    /// Trains of the line on the move or turning now, and how many of them are packed full somewhere on their trip
    /// (T-472). A trip counts from its departure until the next one could leave the far end, so the count is about the
    /// line's trains in service.
    fn under_way(&self, acc: &LineAccount) -> (f64, f64) {
        let tod = self.tod;
        let (mut on, mut packed) = (0.0, 0.0);
        for (r, ra) in acc.runs.iter().enumerate() {
            for (j, &dep) in ra.trains.deps.iter().enumerate() {
                // `(tod - dep).rem_euclid(DAY)` to the bit: both are times of day
                let d = tod - dep;
                let d = if d > -DAY && d < DAY { if d < 0.0 { d + DAY } else { d } } else { d.rem_euclid(DAY) };
                if d >= self.span[r][ra.trains.levels[j] as usize] {
                    continue;
                }
                on += 1.0;
                if self.packed[r][j] {
                    packed += 1.0;
                }
            }
        }
        (on, packed)
    }
}

impl Part {
    /// One account's departures with time of day in (`from`, `to`] (`day0` the local time of the window's midnight), as
    /// `tally` counted them.
    fn read(acc: &LineAccount, n: usize, from: f64, to: f64, day0: f64, past: bool) -> Part {
        let mut runs = vec![];
        for (r, ra) in acc.runs.iter().enumerate() {
            let m = ra.trains.deps.len();
            if m == 0 || ra.n != n {
                continue;
            }
            // T-680: nothing ran before the line's trains started
            let open = ra.open - day0;
            let stops = (0..n).map(|k| Stop::read(ra, k, from, to, day0, open, past)).collect();
            let line_from = if open > from { open - 1e-6 } else { from };
            let trips = (0..m)
                .map(|j| {
                    let mut t = Trip { best: 0.0, at: NONE, count: 0 };
                    for k in 0..n {
                        let s = ra.leaves(j, k);
                        // T-848: a stop along the line opens with its first train
                        let from = ra.stop_open.get(k).map_or(line_from, |o| if o - day0 > line_from { o - day0 - 1e-6 } else { line_from });
                        if s > to || s <= from {
                            continue;
                        }
                        t.count += 1;
                        let v = ra.load[j * n + k];
                        if v > t.best {
                            t.best = v;
                            t.at = k as u32;
                        }
                    }
                    t
                })
                .collect();
            runs.push(Run { r, stops, trips });
        }
        Part { past, runs }
    }
}

impl Stop {
    /// Stop `k`'s departures with time of day in (`from`, `to`]. A stop's boardings telescope: the riders who have
    /// arrived by its last departure in the window, less those still waiting after it, less the same at the departure
    /// before the window (the day's last, a day earlier, when `from` is before the first), less those who went another
    /// way in the window.
    fn read(ra: &RunAccount, k: usize, from: f64, to: f64, day0: f64, line_open: f64, past: bool) -> Stop {
        let n = ra.n;
        // T-848: a stop along the line opens with its first train
        let open = ra.stop_open.get(k).map_or(line_open, |o| o - day0);
        let from = if open > from { open - 1e-6 } else { from };
        let arr = &ra.arrivals[k];
        let ts = ra.stop_times(k);
        let ord = ts.ord;
        let m = ord.len();
        let q = |j: usize| ra.queues.q_after(ord.get(j) * n + k) as f64;
        let (lo, done) = (ts.partition_point(|s| s <= from), ts.partition_point(|s| s <= to));
        let (t_before, q_before) = if lo > 0 { (ts.at(lo - 1), q(lo - 1)) } else { (ts.at(m - 1) - DAY, q(m - 1)) };
        let c_before = arr.cum(t_before);
        let mut st = Stop { lo: lo as u32, done: done as u32, y: c_before - q_before, ..Default::default() };
        if done > lo {
            st.x = arr.cum(ts.at(done - 1)) - q(done - 1);
        }
        let at = |j: usize| ord.get(j) * n + k;
        if past {
            for j in (lo..done).rev() {
                let (g, l) = (ra.queues.gone(at(j)), ra.queues.left(at(j)));
                st.gone += g as f64;
                st.left += l as f64;
                if g != 0.0 || l != 0.0 {
                    st.tail.push((j as u32, st.gone, st.left));
                }
            }
            st.tail.reverse();
        } else {
            for j in lo..done {
                st.gone += ra.queues.gone(at(j)) as f64;
                st.left += ra.queues.left(at(j)) as f64;
            }
        }
        // whoever was waiting when the stop opened never was: the platform starts empty (the departure before it is
        // before the opening, the one at it at or after)
        if open > f64::NEG_INFINITY && done > lo {
            let j = ts.partition_point(|s| s < open).max(lo);
            if j < done {
                let (prev, q_prev, c_prev) = if j == lo { (t_before, q_before, c_before) } else { (ts.at(j - 1), q(j - 1), arr.cum(ts.at(j - 1))) };
                if prev < open {
                    st.open = q_prev + arr.cum(open) - c_prev;
                }
            }
        }
        st
    }
}

impl Run {
    /// Move the run's part of the window on to time of day `tod` (later on the same day, no opening within reach):
    /// today's takes in each stop's departures up to it, yesterday's lets them go.
    fn advance(&mut self, ra: &RunAccount, past: bool, tod: f64) {
        let n = ra.n;
        let m = ra.trains.deps.len();
        for (k, st) in self.stops.iter_mut().enumerate() {
            let ts = ra.stop_times(k);
            let ord = ts.ord;
            let q = |j: usize| ra.queues.q_after(ord.get(j) * n + k) as f64;
            let arr = &ra.arrivals[k];
            if past {
                let lo0 = st.lo as usize;
                let mut lo = lo0;
                while lo < m && ts.at(lo) <= tod {
                    let j = ord.get(lo);
                    let tr = &mut self.trips[j];
                    tr.count -= 1;
                    if tr.count == 0 {
                        (tr.best, tr.at) = (0.0, NONE);
                    } else if tr.at == k as u32 {
                        // its fullest stop left the window: the fullest of those still in it, by stop
                        (tr.best, tr.at) = (0.0, NONE);
                        for k2 in 0..n {
                            let v = ra.load[j * n + k2];
                            if ra.leaves(j, k2) > tod && v > tr.best {
                                tr.best = v;
                                tr.at = k2 as u32;
                            }
                        }
                    }
                    lo += 1;
                }
                if lo != lo0 {
                    st.lo = lo as u32;
                    while (st.next as usize) < st.tail.len() && (st.tail[st.next as usize].0 as usize) < lo {
                        st.next += 1;
                    }
                    (st.gone, st.left) = st.tail.get(st.next as usize).map_or((0.0, 0.0), |e| (e.1, e.2));
                    st.y = arr.cum(ts.at(lo - 1)) - q(lo - 1);
                }
            } else {
                let d0 = st.done as usize;
                let mut d = d0;
                while d < m && ts.at(d) <= tod {
                    let j = ord.get(d);
                    let i = j * n + k;
                    st.gone += ra.queues.gone(i) as f64;
                    st.left += ra.queues.left(i) as f64;
                    let tr = &mut self.trips[j];
                    tr.count += 1;
                    // the first stop (by stop) with the most riders aboard, as a pass over the stops in order finds it
                    let v = ra.load[i];
                    if v > tr.best || (v == tr.best && tr.at != NONE && (k as u32) < tr.at) {
                        tr.best = v;
                        tr.at = k as u32;
                    }
                    d += 1;
                }
                if d != d0 {
                    st.done = d as u32;
                    st.x = arr.cum(ts.at(d - 1)) - q(d - 1);
                }
            }
        }
    }
}

impl Boarding {
    /// Every line's block at game time `t`: the past 24 hours (`past`) or today so far. `keep`: each line's block is
    /// kept and moved along while its accounts, service and local day stand (T-4149, T-4281); otherwise each is read
    /// whole, nothing kept used (`?patch=check`'s whole read).
    pub(super) fn window(&self, svc: &Service, t: f64, past: bool, keep: bool) -> Vec<f64> {
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
            let ya = if past { self.accounts[self.kind_at(t - DAY, l)].get(l).and_then(|a| a.as_ref()) } else { None };
            let key = day_key(ls, t - tod, acc.id, ya.map_or(0, |a| a.id));
            count += 1;
            if !keep {
                out.extend(DayKept::read(l, ls, acc, ya, t, tod, key, None).block);
                continue;
            }
            if let Some(k) = memo[l].as_mut().filter(|k| k.key() == key.as_slice()) {
                if k.advance(l, ls, acc, ya, tod) {
                    out.extend_from_slice(&k.block);
                    continue;
                }
            }
            let k = DayKept::read(l, ls, acc, ya, t, tod, key, memo[l].take());
            out.extend_from_slice(&k.block);
            memo[l] = Some(k);
        }
        out[0] = count as f64;
        out
    }
}
