//! T-4544 (T-4511 stage 2): single trains moved off their pattern. A line's moves are a list beside its
//! pattern (`Lattice` stays a pure, `Copy` pattern); whatever reads the timetable applies them to the
//! departures the pattern lays out, so the board, the table, the drawn trains, `retime` and the boarding account
//! agree. The 90 s gap and auto-space (later stages) write to the same list under their own kinds.

use std::sync::Arc;

/// Who made a move: the player for the train (`Own`) or with another train it is timed to (`With`); the game's
/// 90 s gap pass (`Gap`) or a line's auto-space (`Auto`). Only the first two are ever saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    Own,
    With,
    Gap,
    Auto,
}

/// One departure moved: run `run`'s departure the pattern lays out nearest `at` leaves `secs` later (negative:
/// earlier). `at` is a key (T-4547): the time of day, in the line's local time, of the pattern's point for that
/// departure (the departure from the first stop plus the line's `lead`), so it names "the 7:32" on every day and
/// does not change when a reroute changes `lead`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub run: u8,
    pub at: f64,
    pub secs: f32,
    pub kind: MoveKind,
    /// T-4549: the kind of day it belongs to (`params::WEEKDAYS`, `WEEKEND`): a move runs on days of its kind only.
    pub row: u8,
}

/// A train by its place in its line's day (`Dep`'s level and place: the pattern's own numbering, the table's column), which no
/// move or retime changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Train {
    pub line: u32,
    pub run: u8,
    pub lev: u8,
    pub place: u32,
}

/// T-4548: two trains `retime` timed directly with each other on a weekday: `timed` runs a timetable of its own worked out
/// against `other`'s train as it runs (a faster train behind a slower one it can't pass, a slower train waiting at a passing
/// station for a faster one). A move of either takes the other with it by the same seconds, one step only: never through
/// a partner's partners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub timed: Train,
    pub other: Train,
}

/// A line's moves, sorted by `at`, shared by the service and the eras that ran them.
pub type Moves = Arc<[Move]>;

const DAY: f64 = 86400.0;

/// No train moves further than this: the window a lookup widens by to find a departure moved into it.
pub const MAX_MOVE_S: f64 = 1800.0;

/// A list of moves, sorted.
pub fn sorted(mut v: Vec<Move>) -> Moves {
    v.sort_by(|a, b| a.at.total_cmp(&b.at).then(a.run.cmp(&b.run)));
    v.into()
}

/// The key of a departure at game time `d`: its time of day in local time plus the lead, `ofs` being the lattice's
/// `shift + lead`.
pub fn key_of(d: f64, ofs: f64) -> f64 {
    (d + ofs).rem_euclid(DAY)
}

/// How far apart two keys are, round the clock.
pub fn apart(a: f64, b: f64) -> f64 {
    let x = (a - b).rem_euclid(DAY);
    x.min(DAY - x)
}

/// The seconds run `r`'s departure at game time `d` of headway `hw` is moved: that of the move whose key is nearest it
/// within half a headway on a day of kind `row` (the day of the departure, `Lattice::row_at`), so two moves for one departure leave the nearer, and a move no departure is within half a
/// headway of waits. Reads only the departure and the moves, so every window agrees.
pub fn secs(moves: &[Move], r: usize, d: f64, hw: f64, ofs: f64, row: usize) -> f64 {
    if moves.is_empty() {
        return 0.0;
    }
    let (half, k) = (0.5 * hw, key_of(d, ofs));
    let mut best: Option<(f64, f64)> = None;
    for m in moves.iter().filter(|m| m.run as usize == r && m.row as usize == row) {
        let dist = apart(m.at, k);
        if dist < half && best.map_or(true, |b| dist < b.0) {
            best = Some((dist, m.secs as f64));
        }
    }
    best.map_or(0.0, |b| b.1.clamp(-MAX_MOVE_S, MAX_MOVE_S))
}

/// What a hash of a line's departures reads of its moves (nothing for none, so their keys stay as they were).
pub fn words(moves: &[Move]) -> Vec<u64> {
    moves.iter().flat_map(|m| [m.at.to_bits(), (m.row as u64) << 48 | (m.run as u64) << 40 | (m.kind as u64) << 32 | m.secs.to_bits() as u64]).collect()
}
