//! T-3742: nudging a line's departures (notes/T-3742.md). A row of a rail line is held at an offset from where the
//! spacing placed it: its slots are never placed again (`spacing::place`), and the offset is in its `line_phase`
//! itself, so every reader of the phase (departures, the drawn trains, the board, boarding, `timing::retime`, the
//! spacing of other lines) runs the nudged times with nothing to settle. A line timed behind a slower one
//! (`Capacity::follow_pairs`) holds no offset of its own: a nudge of either moves the slower line's (its anchor's),
//! and every line timed with that one by the same seconds, so their passes and waits stay as sized. Weekends follow
//! the weekdays: a line running other rates then has its weekend phase moved and held with them.

use super::net::{Issue, Network};
use super::params::SLOTS;
use super::world::{Op, TrackWorld};

/// One slot of a line as `Op::Nudge` sets it; a None field is left as it stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NudgeSet {
    pub line: u32,
    pub slot: u8,
    /// the offset it is held at, s (`Network::line_offset`; Some(None): placed by the spacing again)
    pub offset: Option<Option<f32>>,
    /// its departure phase on weekdays and on weekends (`line_phase`, `line_phase_rest`)
    pub phase: Option<f32>,
    pub rest: Option<f32>,
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

    /// T-3742: the edit that nudges row `row` of line `line` `by` seconds later (earlier when negative), within half
    /// a headway of where the spacing placed it. Per slot of the row: the line's anchor (`anchor`) holds the offset,
    /// and it and every line timed with it move by the same seconds. None where nothing moves (the end of the range,
    /// no trains in the row).
    pub fn nudge_op(&self, line: u32, row: usize, by: f64) -> Result<Option<Op>, Issue> {
        if !self.nudgeable(line) { return Err(Issue::NoSuchLine { line }) }
        let net = &self.net;
        let mut set = vec![];
        for s in self.row_slots_of(line, row) {
            let pairs = self.svc.cap.follow_pairs(net, s);
            let a = anchor(&pairs, line);
            let tph = net.line_tph.get(a as usize).map_or(0.0, |t| t[s]) as f64;
            if tph <= 0.0 { continue }
            let h = 3600.0 / tph;
            let now = shown(net.offset_at(a, s).unwrap_or(0.0), h);
            let d = ((now + by).clamp(-0.5 * h, 0.5 * h) - now) as f32;
            if d.abs() < 1e-3 { continue }
            for m in group(&pairs, a) {
                let held = net.offset_at(m, s);
                let offset = if m == a || held.is_some() {
                    let o = held.unwrap_or(0.0) + d;
                    Some((shown(o, h).abs() >= ZERO_S).then_some(o))
                } else {
                    None
                };
                let rest = (net.runs_own_weekend(m as usize) && net.rest_solved).then(|| net.line_phase_rest.get(m as usize).map(|p| p[s] + d)).flatten();
                set.push(NudgeSet { line: m, slot: s as u8, offset, phase: Some(net.line_phase_of(m)[s] + d), rest });
            }
        }
        Ok((!set.is_empty()).then_some(Op::Nudge(set)))
    }

    /// T-3742: what row `row` of line `line` shows for its offset: the offset of its anchor in its first slot that
    /// runs (wrapped, s), half that headway (the range either way, s; 0: no trains, nothing to nudge), whether it is
    /// held, and the other lines timed with it (they move with it). None for a line a nudge cannot move.
    pub fn row_nudge(&self, line: u32, row: usize) -> Option<(f64, f64, bool, Vec<u32>)> {
        if !self.nudgeable(line) { return None }
        let net = &self.net;
        for s in self.row_slots_of(line, row) {
            let pairs = self.svc.cap.follow_pairs(net, s);
            let a = anchor(&pairs, line);
            let tph = net.line_tph.get(a as usize).map_or(0.0, |t| t[s]) as f64;
            if tph <= 0.0 { continue }
            let h = 3600.0 / tph;
            let held = net.offset_at(a, s);
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
        let mut b = NudgeSet { line: x.line, slot: x.slot, offset: None, phase: None, rest: None };
        if let Some(o) = x.offset {
            if n.line_offset.len() <= l { n.line_offset.resize(l + 1, [None; SLOTS]); }
            b.offset = Some(std::mem::replace(&mut n.line_offset[l][s], o));
        }
        if let Some(p) = x.phase {
            if n.line_phase.len() <= l { n.line_phase.resize(l + 1, [0.0; SLOTS]); }
            b.phase = Some(std::mem::replace(&mut n.line_phase[l][s], p));
        }
        if let Some(p) = x.rest {
            if let Some(v) = n.line_phase_rest.get_mut(l) { b.rest = Some(std::mem::replace(&mut v[s], p)); }
        }
        back.push(b);
    }
    back.reverse();
    Ok(Op::Nudge(back))
}

/// The held slots of line `l` released, as entries that hold them again (a removed line's, for its undo).
pub(super) fn release(n: &mut Network, l: u32) -> Vec<NudgeSet> {
    let Some(o) = n.line_offset.get_mut(l as usize) else { return vec![] };
    let had = std::mem::replace(o, [None; SLOTS]);
    (0..SLOTS).filter_map(|s| had[s].map(|x| NudgeSet { line: l, slot: s as u8, offset: Some(Some(x)), phase: None, rest: None })).collect()
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
