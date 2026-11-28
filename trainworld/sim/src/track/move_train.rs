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
        (0..cap.res.len()).flat_map(|r| cap.users_of(r)).filter(|u| u.line == l && u.run == run).map(|u| u.occ).fold(0.0, f64::max)
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

    /// The edit that moves that train `by` seconds later (earlier when negative) than it runs now, as far as its range
    /// allows. None for no step; refused, with the reason, where it can go no further that way (the step's end is the range's).
    pub fn move_op(&self, line: u32, run: u8, dep: f64, by: f64) -> Result<Option<Op>, Refused> {
        let r = self.move_range(line, run, dep)?;
        if by.abs() < MIN_STEP_S {
            return Ok(None);
        }
        let to = (r.now + by).clamp(r.lo, r.hi);
        if (to - r.now).abs() < MIN_STEP_S {
            return Err(if by > 0.0 { Refused { limit: r.hi, why: r.hi_why } } else { Refused { limit: r.lo, why: r.lo_why } });
        }
        let to = if to.abs() < MIN_STEP_S { 0.0 } else { to };
        // the headway of that departure: the moves of its run within half of it are its own
        let hw = self.svc.line_trips(&self.net, line as usize, dep - 1.0, dep + 1.0).iter().find(|t| t.run == run && (t.dep - dep).abs() < 0.5).map_or(0.0, |t| t.hw);
        let Some(lat) = self.svc.lattice(&self.net, line as usize) else { return Err(none(Why::NoTimetable)) };
        let key = moves::key_of(dep, lat.shift + lat.lead);
        let mut list: Vec<Move> = self.svc.stored_of(line as usize).iter().copied().filter(|m| !(m.run == run && moves::apart(m.at, key) < 0.5 * hw.max(1.0))).collect();
        if to != 0.0 {
            list.push(Move { run, at: key, secs: to as f32, kind: MoveKind::Own });
        }
        Ok(Some(Op::MoveTrain { line, moves: list }))
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
