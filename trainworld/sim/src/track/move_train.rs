//! T-4545 (T-4511 stage 3): moving one train. The edit is `Op::MoveTrain` (a line's whole list of moves, undone by the list it
//! found), a service edit like a nudge: nothing to settle or place, trains running carry on (`note_changes` leaves an era
//! at the edit), and the line's rows holding a moved train are not placed again (`Service::move_holds`).
//!
//! How far a train moves (notes/T-4511.md "How far it moves"), worked out on the slot's own timetable:
//! 1. its train makes its next departure, standing at the far end no less than `TURN_MIN_S` (and the train that runs it
//!    stood at the first stop that long before): a line set by train counts stands exactly its turnaround, so about 2
//!    minutes either way; a Headway or paced line adds its spare time;
//! 2. it keeps clear of its own line's trains before and after it, by the longest time one of its trains holds a place it
//!    uses that way (`User::occ`) and at least a minute;
//! 3. its line runs a timetable: a running rail or tram line, not a circle.
//! `hand_over` pairs trains by the unmoved times (`Trip::ready`), so a move in range never brings a train into service
//! or sends one away.

use super::moves::{self, Move, MoveKind, MAX_MOVE_S};
use super::service::move_dep;
use super::timing::TURN_MIN_S;
use super::net::LoopKind;
use super::world::{Op, TrackWorld};

const DAY: f64 = 86400.0;
/// Fewest seconds between two trains of one line leaving one way.
const MIN_APART_S: f64 = 60.0;
/// A step that moves a train less than this is none.
const MIN_STEP_S: f64 = 0.5;

/// What stops a train moving further one way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Why {
    /// Its train has to leave `node` again at game time `at` (rule 1, later).
    Leaves { node: u32, at: f64 },
    /// The train that runs it only gets to `node` at `at` (rule 1, earlier).
    Arrives { node: u32, at: f64 },
    /// Too close to its line's train leaving at `at` (rule 2).
    Close { at: f64 },
    /// The furthest any train moves.
    Far,
    /// Its line runs no timetable of its own (rule 3).
    NoTimetable,
    /// The line has no such departure.
    NoTrain,
}

/// A train's range: the seconds it is moved now, and the furthest it may be (total, from where its pattern puts it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub now: f64,
    pub lo: f64,
    pub hi: f64,
    pub lo_why: Why,
    pub hi_why: Why,
    /// the ends before a train past a limit (its neighbours moved since) is let stay where it is: where the stored seconds
    /// are clamped to (`clamp_moves`)
    pub raw_lo: f64,
    pub raw_hi: f64,
}

/// A move refused: the total it could go to that way instead (`limit`), and why it can't go further.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refused {
    pub limit: f64,
    pub why: Why,
}

fn none(why: Why) -> Refused {
    Refused { limit: 0.0, why }
}

impl TrackWorld {
    /// The longest time one of line `l`'s trains holds a place it uses running run `run` (`User::occ`).
    fn line_occ(&self, l: u32, run: u8) -> f64 {
        let cap = &self.svc.cap;
        // (a following loss's users hold no place: the owner's `occ` is where its stretch ends)
        (0..cap.res.len()).filter(|&r| !cap.is_follow(r)).flat_map(|r| cap.users_of(r)).filter(|u| u.line == l && u.run == run).map(|u| u.occ).fold(0.0, f64::max)
    }

    /// Where the train of line `line` leaving at `dep` (its pattern's time, `Trip::dep`) running `run` may be moved to, and
    /// what stops it each way (rules 1 to 3).
    pub fn move_range(&self, line: u32, run: u8, dep: f64) -> Result<Range, Refused> {
        let net = &self.net;
        let l = line as usize;
        let ls = self.svc.lines.get(l).filter(|x| x.ok && x.running && !x.is_bus() && x.kind != LoopKind::Circle && (run as usize) < x.live_runs());
        let ls = match ls {
            Some(x) if net.line_ok(line) && !net.is_ferry(line) => x,
            _ => return Err(none(Why::NoTimetable)),
        };
        let (dur, hcap) = self.svc.reach(net, l);
        let trips = self.svc.line_trips(net, l, dep - dur - hcap - 120.0, dep + 2.0 * hcap + MAX_MOVE_S + 120.0);
        let Some(me) = trips.iter().find(|t| t.run == run && (t.dep - dep).abs() < 0.5) else { return Err(none(Why::NoTrain)) };
        let (cur, base) = (me.moved as f64, me.draw - me.moved as f64);
        let stand = ls.close_stand(net, line).min(TURN_MIN_S);
        let stops = net.line_stops(line);
        let (first, last) = if run == 0 { (stops.first(), stops.last()) } else { (stops.last(), stops.first()) };
        let (first, last) = (first.copied().unwrap_or(u32::MAX), last.copied().unwrap_or(u32::MAX));
        let (mut lo, mut lo_why) = (-MAX_MOVE_S, Why::Far);
        let (mut hi, mut hi_why) = (MAX_MOVE_S, Why::Far);
        // rule 1: later, the train reaches the far end that much later and still takes its next departure
        if let Some(next) = me.next {
            let arrive = base + ls.prof_of(run as usize, me.level as usize, 0).duration;
            let room = next - arrive - stand;
            if room < hi {
                (hi, hi_why) = (room, Why::Leaves { node: last, at: next });
            }
        }
        // and earlier, the train that runs it only got to the first stop on its last journey that long before
        if let Some(prev) = trips.iter().find(|t| t.next == Some(me.draw) && (t.run, t.dep) != (me.run, me.dep)) {
            let arrive = prev.draw + ls.prof_of(prev.run as usize, prev.level as usize, 0).duration;
            let room = arrive + stand - base;
            if room > lo {
                (lo, lo_why) = (room, Why::Arrives { node: first, at: arrive });
            }
        }
        // rule 2: clear of its own line's trains before and after it, as they run
        let gap = self.line_occ(line, run).max(MIN_APART_S);
        let same = |t: &&super::service::Trip| t.run == run;
        if let Some(p) = trips.iter().filter(same).filter(|t| t.dep < me.dep).max_by(|a, b| a.dep.total_cmp(&b.dep)) {
            let room = p.draw + gap - base;
            if room > lo {
                (lo, lo_why) = (room, Why::Close { at: p.draw });
            }
        }
        if let Some(n) = trips.iter().filter(same).filter(|t| t.dep > me.dep).min_by(|a, b| a.dep.total_cmp(&b.dep)) {
            let room = n.draw - gap - base;
            if room < hi {
                (hi, hi_why) = (room, Why::Close { at: n.draw });
            }
        }
        let (raw_lo, raw_hi) = (lo, hi);
        // a train already past a limit (its neighbours moved since) stays where it is and moves back only
        if lo > cur {
            (lo, lo_why) = (cur, lo_why);
        }
        if hi < cur {
            (hi, hi_why) = (cur, hi_why);
        }
        Ok(Range { now: cur, lo, hi, lo_why, hi_why, raw_lo, raw_hi })
    }

    /// The trains that move with the one of line `line` leaving at `dep` (T-4548): those `retime` timed directly with it, as
    /// (line, run, departure in their pattern) on the day of `dep`, with their ranges. The train itself comes first.
    fn group(&self, line: u32, run: u8, dep: f64, own: Range) -> Vec<(u32, u8, f64, Range)> {
        let mut out = vec![(line, run, dep, own)];
        let net = &self.net;
        let Some(lat) = self.svc.lattice(net, line as usize) else { return out };
        let Some(me) = self.svc.train_at(net, line as usize, run, moves::key_of(dep, lat.shift + lat.lead)) else { return out };
        for p in self.svc.partners(me) {
            let (Some(k), Some(pl)) = (self.svc.place_key(net, p), self.svc.lattice(net, p.line as usize)) else { continue };
            let ofs = pl.shift + pl.lead;
            // the partner's departure nearest this one's, on the same day
            let g = dep + (k - ofs - dep + 0.5 * DAY).rem_euclid(DAY) - 0.5 * DAY;
            let trips = self.svc.line_trips(net, p.line as usize, g - 7200.0, g + 7200.0);
            let Some(t) = trips.iter().filter(|t| t.run == p.run && moves::apart(moves::key_of(t.dep, ofs), k) < 0.5 * t.hw).min_by(|a, b| moves::apart(moves::key_of(a.dep, ofs), k).total_cmp(&moves::apart(moves::key_of(b.dep, ofs), k))) else { continue };
            if let Ok(r) = self.move_range(p.line, p.run, t.dep) {
                out.push((p.line, p.run, t.dep, r));
            }
        }
        out
    }

    /// The edit that moves that train `by` seconds later (earlier when negative) than it runs now, as far as its range
    /// allows. None for no step; refused, with the reason, where it can go no further that way (the step's end is the range's).
    /// T-4548: the trains timed with it move by the same seconds (stored as moves of their lines marked `With`), the step
    /// ending at the tightest of their ranges, and a train back at 0 takes its partners back to where they were.
    pub fn move_op(&self, line: u32, run: u8, dep: f64, by: f64) -> Result<Option<Op>, Refused> {
        let r = self.move_range(line, run, dep)?;
        if by.abs() < MIN_STEP_S {
            return Ok(None);
        }
        let group = self.group(line, run, dep, r);
        // the furthest every train of the group goes each way, as seconds more than it is moved now
        let (mut lo, mut lo_why, mut hi, mut hi_why) = (f64::NEG_INFINITY, Why::Far, f64::INFINITY, Why::Far);
        for g in &group {
            if g.3.lo - g.3.now > lo {
                (lo, lo_why) = (g.3.lo - g.3.now, g.3.lo_why);
            }
            if g.3.hi - g.3.now < hi {
                (hi, hi_why) = (g.3.hi - g.3.now, g.3.hi_why);
            }
        }
        let delta = by.clamp(lo, hi);
        if delta.abs() < MIN_STEP_S {
            return Err(if by > 0.0 { Refused { limit: r.now + hi, why: hi_why } } else { Refused { limit: r.now + lo, why: lo_why } });
        }
        let to = r.now + delta;
        let delta = if to.abs() < MIN_STEP_S { -r.now } else { delta };
        let to = if to.abs() < MIN_STEP_S { 0.0 } else { to };
        let mut lists: std::collections::BTreeMap<u32, Vec<Move>> = Default::default();
        for (i, &(l, run, dep, rg)) in group.iter().enumerate() {
            let total = if i == 0 { to } else { rg.now + delta };
            let total = if total.abs() < MIN_STEP_S { 0.0 } else { total };
            // the headway of that departure: the moves of its run within half of it are its own
            let hw = self.svc.line_trips(&self.net, l as usize, dep - 1.0, dep + 1.0).iter().find(|t| t.run == run && (t.dep - dep).abs() < 0.5).map_or(0.0, |t| t.hw);
            let Some(lat) = self.svc.lattice(&self.net, l as usize) else {
                if i == 0 {
                    return Err(none(Why::NoTimetable));
                }
                continue;
            };
            let key = moves::key_of(dep, lat.shift + lat.lead);
            let list = lists.entry(l).or_insert_with(|| self.svc.stored_of(l as usize).to_vec());
            let near = |m: &Move| m.run == run && moves::apart(m.at, key) < 0.5 * hw.max(1.0);
            let kind = if i > 0 && !list.iter().any(|m| near(m) && m.kind == MoveKind::Own) { MoveKind::With } else { MoveKind::Own };
            list.retain(|m| !near(m));
            if total != 0.0 {
                list.push(Move { run, at: key, secs: total as f32, kind });
            }
        }
        let mut ops: Vec<Op> = lists.into_iter().map(|(line, moves)| Op::MoveTrain { line, moves }).collect();
        Ok(Some(if ops.len() == 1 { ops.pop().unwrap() } else { Op::Batch(ops) }))
    }
}

impl TrackWorld {
    /// T-4547: each line's stored moves clamped to the room its trains have now (`Range::raw_lo`, `raw_hi`; a train is always
    /// let stay unmoved), as they run (`Service::moves`). A move with no departure to take it, or on a line that runs no
    /// timetable, runs as stored. Run before `retime`, which reads what runs; a train's room depends on its neighbours'
    /// moves, so it goes round until they stop changing (three times at most, the next edit goes on from there).
    pub(super) fn clamp_moves(&mut self) {
        if self.svc.stored.iter().all(|s| s.is_empty()) {
            return;
        }
        for l in 0..self.svc.stored.len() {
            let st = self.svc.stored_of(l);
            if st.is_empty() {
                continue;
            }
            for _ in 0..3 {
                let eff: Vec<Move> = st.iter().map(|m| Move { secs: self.clamped(l as u32, m), ..*m }).collect();
                if *moves::sorted(eff.clone()) == *self.svc.moves_of(l) {
                    break;
                }
                self.svc.set_eff(l, eff);
            }
        }
    }

    fn clamped(&self, line: u32, m: &Move) -> f32 {
        let l = line as usize;
        let Some(lat) = self.svc.lattice(&self.net, l) else { return m.secs };
        let Some((d, ..)) = move_dep(&lat, m, self.svc.ref_day(l), &mut vec![]) else { return m.secs };
        match self.move_range(line, m.run, d) {
            Ok(r) => (m.secs as f64).max(r.raw_lo.min(0.0)).min(r.raw_hi.max(0.0)) as f32,
            Err(_) => m.secs,
        }
    }
}

#[cfg(test)]
mod tests;
