//! T-3742: nudging a line's departures (notes/T-3742.md). A row of a rail line is held at an offset from where the
//! spacing placed it: its slots are never placed again (`spacing::place`), and the offset is in its `line_phase`
//! itself, so every reader of the phase (departures, the drawn trains, the board, boarding, `timing::retime`, the
//! spacing of other lines) runs the nudged times with nothing to settle. A line timed behind a slower one
//! (`Capacity::follow_pairs`) holds no offset of its own: a nudge of either moves the slower line's (its anchor's),
//! and every line timed with that one by the same seconds, so their passes and waits stay as sized.
//! T-4510: weekends are nudged on their own (`weekend` below): a weekend row's offset and phase are held apart
//! (`Network::line_offset_rest`, `line_hold_rest`), read over whatever the weekend would run otherwise
//! (`Network::weekend_phase_of`), so a line running its weekday departures on weekends has its own weekend phase
//! only for the slots held, and a weekday nudge no longer moves the weekends.

use super::net::{Issue, Network};
use super::params::{SLOTS, WEEKEND};
use super::world::{Op, TrackWorld};

/// One slot of a line as `Op::Nudge` sets it (on weekends when `weekend`); a None field is left as it stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NudgeSet {
    pub line: u32,
    pub slot: u8,
    pub weekend: bool,
    /// the offset it is held at, s (`Network::line_offset`; Some(None): placed by the spacing again)
    pub offset: Option<Option<f32>>,
    /// its departure phase (`line_phase`; on weekends the held one, `line_hold_rest`)
    pub phase: Option<f32>,
}

/// An offset as shown and stepped: within half a headway `h` either way as set, else (the headway is shorter since)
/// wrapped into it, as a whole headway's shift is the same pattern.
pub fn shown(offset: f32, h: f64) -> f64 {
    let o = offset as f64;
    if h <= 0.0 { return 0.0 }
    if o.abs() <= 0.5 * h { return o }
    (o + 0.5 * h).rem_euclid(h) - 0.5 * h
}

/// Below this many seconds an offset is 0: the row is placed by the spacing again.
const ZERO_S: f64 = 0.5;

/// The line `l` is timed behind in `pairs` (faster, slower), up the chain: the one holding its offset.
fn anchor(pairs: &[(u32, u32)], l: u32) -> u32 {
    let mut at = l;
    let mut seen = vec![l];
    while let Some(&(_, slow)) = pairs.iter().find(|p| p.0 == at) {
        if seen.contains(&slow) { break; }
        seen.push(slow);
        at = slow;
    }
    at
}

/// Every line timed with `l` in `pairs`, either way and through others, `l` first.
fn group(pairs: &[(u32, u32)], l: u32) -> Vec<u32> {
    let mut out = vec![l];
    let mut i = 0;
    while i < out.len() {
        let at = out[i];
        for &(a, b) in pairs {
            let other = if a == at { b } else if b == at { a } else { continue };
            if !out.contains(&other) { out.push(other); }
        }
        i += 1;
    }
    out
}

/// Line `l`'s held offset in slot `s` on weekdays, or on weekends.
fn held_at(net: &Network, l: u32, s: usize, weekend: bool) -> Option<f32> {
    if weekend { net.offset_rest_at(l, s) } else { net.offset_at(l, s) }
}

impl TrackWorld {
    /// The slots row `row` of line `l` (its rows in the day's order) sets the departures of: those whose
    /// timetable is that row's (`Network::place`).
    fn row_slots_of(&self, l: u32, row: usize) -> Vec<usize> {
        let net = &self.net;
        let Some(&(k, _)) = net.row_slots(l as usize).get(row) else { return vec![] };
        net.slots().filter(|&s| net.place(l as usize, s) == k).collect()
    }

    /// A rail line the spacing places, running: what a nudge may move.
    fn nudgeable(&self, l: u32) -> bool {
        self.net.line_ok(l) && !self.net.is_twin(l) && self.svc.lines.get(l as usize).is_some_and(|x| x.running && !x.is_bus())
    }

    /// Trains an hour of line `a` in slot `s` on the day asked.
    fn tph_on_day(&self, a: u32, s: usize, weekend: bool) -> f64 {
        if weekend { self.net.line_tph_on(a as usize, WEEKEND, s) as f64 } else { self.net.line_tph.get(a as usize).map_or(0.0, |t| t[s]) as f64 }
    }

    /// T-3742: the edit that nudges row `row` of line `line` `by` seconds later (earlier when negative), within half
    /// a headway of where the spacing placed it, on weekdays or (T-4510) on weekends. Per slot of the row: the line's
    /// anchor (`anchor`) holds the offset, and it and every line timed with it move by the same seconds (on weekends
    /// each is held, as a line running its weekday departures there has no weekend phase to move otherwise). None where
    /// nothing moves (the end of the range, no trains in the row).
    pub fn nudge_op(&self, line: u32, row: usize, by: f64, weekend: bool) -> Result<Option<Op>, Issue> {
        if !self.nudgeable(line) { return Err(Issue::NoSuchLine { line }) }
        let net = &self.net;
        let mut set = vec![];
        for s in self.row_slots_of(line, row) {
            let pairs = self.svc.cap.follow_pairs(net, s);
            let a = anchor(&pairs, line);
            let tph = self.tph_on_day(a, s, weekend);
            if tph <= 0.0 { continue }
            let h = 3600.0 / tph;
            let now = shown(held_at(net, a, s, weekend).unwrap_or(0.0), h);
            let d = ((now + by).clamp(-0.5 * h, 0.5 * h) - now) as f32;
            if d.abs() < 1e-3 { continue }
            // on weekends every line timed with it in any slot is held, so a timed pair keeps one `rest_on` and is not
            // left averaged (`timing::retime`)
            let timed = if weekend { net.slots().flat_map(|s2| self.svc.cap.follow_pairs(net, s2)).collect() } else { pairs.clone() };
            for m in group(&timed, a) {
                let held = held_at(net, m, s, weekend);
                let offset = if m == a || held.is_some() || weekend {
                    let o = held.unwrap_or(0.0) + d;
                    Some((shown(o, h).abs() >= ZERO_S).then_some(o))
                } else {
                    None
                };
                let from = if weekend { net.weekend_phase_of(m)[s] } else { net.line_phase_of(m)[s] };
                set.push(NudgeSet { line: m, slot: s as u8, weekend, offset, phase: Some(from + d) });
            }
        }
        Ok((!set.is_empty()).then_some(Op::Nudge(set)))
    }

    /// T-3742: what row `row` of line `line` shows for its offset: the offset of its anchor in its first slot that
    /// runs (wrapped, s), half that headway (the range either way, s; 0: no trains, nothing to nudge), whether it is
    /// held, and the other lines timed with it (they move with it). None for a line a nudge cannot move.
    pub fn row_nudge(&self, line: u32, row: usize, weekend: bool) -> Option<(f64, f64, bool, Vec<u32>)> {
        if !self.nudgeable(line) { return None }
        let net = &self.net;
        for s in self.row_slots_of(line, row) {
            let pairs = self.svc.cap.follow_pairs(net, s);
            let a = anchor(&pairs, line);
            let tph = self.tph_on_day(a, s, weekend);
            if tph <= 0.0 { continue }
            let h = 3600.0 / tph;
            let held = held_at(net, a, s, weekend);
            let others = group(&pairs, a).into_iter().filter(|&m| m != line).collect();
            return Some((held.map_or(0.0, |o| shown(o, h)), 0.5 * h, held.is_some(), others));
        }
        Some((0.0, 0.0, false, vec![]))
    }
}

/// `Op::Nudge` applied: each entry set as given, in order; the inverse sets back what each found. Refused, with
/// nothing changed, for a line not there or a slot out of range.
pub(super) fn apply(n: &mut Network, set: Vec<NudgeSet>) -> Result<Op, Issue> {
    if let Some(x) = set.iter().find(|x| !n.line_ok(x.line) || x.slot as usize >= SLOTS) {
        return Err(Issue::NoSuchLine { line: x.line });
    }
    let mut back = Vec::with_capacity(set.len());
    for x in set {
        let (l, s) = (x.line as usize, x.slot as usize);
        let mut b = NudgeSet { line: x.line, slot: x.slot, weekend: x.weekend, offset: None, phase: None };
        let (offsets, phases) = if x.weekend { (&mut n.line_offset_rest, &mut n.line_hold_rest) } else { (&mut n.line_offset, &mut n.line_phase) };
        if let Some(o) = x.offset {
            if offsets.len() <= l { offsets.resize(l + 1, [None; SLOTS]); }
            b.offset = Some(std::mem::replace(&mut offsets[l][s], o));
        }
        if let Some(p) = x.phase {
            if phases.len() <= l { phases.resize(l + 1, [0.0; SLOTS]); }
            b.phase = Some(std::mem::replace(&mut phases[l][s], p));
        }
        back.push(b);
    }
    back.reverse();
    Ok(Op::Nudge(back))
}

/// The held slots of line `l` released, as entries that hold them again (a removed line's, for its undo).
pub(super) fn release(n: &mut Network, l: u32) -> Vec<NudgeSet> {
    let mut out = vec![];
    for weekend in [false, true] {
        let offsets = if weekend { &mut n.line_offset_rest } else { &mut n.line_offset };
        let Some(o) = offsets.get_mut(l as usize) else { continue };
        let had = std::mem::replace(o, [None; SLOTS]);
        let phases = if weekend { n.line_hold_rest.get(l as usize).copied() } else { None };
        out.extend((0..SLOTS).filter_map(|s| had[s].map(|x| NudgeSet { line: l, slot: s as u8, weekend, offset: Some(Some(x)), phase: phases.map(|p| p[s]) })));
    }
    out
}

/// The lines `op`'s nudges move, into `out`.
pub(super) fn lines_of(op: &Op, out: &mut Vec<u32>) {
    match op {
        Op::Nudge(v) => out.extend(v.iter().map(|x| x.line)),
        Op::Batch(v) => v.iter().for_each(|o| lines_of(o, out)),
        _ => {}
    }
}

#[cfg(test)]
mod tests;
