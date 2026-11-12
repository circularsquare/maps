//! T-3333: a rail line run on both pairs of quad track (`track/twin.rs`): the switch, and what the
//! app asks to show one line for the two.

use super::*;
use crate::track::twin::{double, Twin};

#[wasm_bindgen]
impl TrackApi {
    /// Switch line `line`'s trains onto both pairs of quad track, or back onto one. On: the switch
    /// is stored, and where the line can run on both its hidden twin is made in the same edit, the
    /// line keeping one pair's trains; off: the twin goes and the line takes the whole line's back.
    /// One undoable edit.
    pub fn set_line_pairs(&mut self, line: u32, on: bool) -> u32 {
        let net = &self.w.net;
        let Some(mut d) = net.line_data(line).filter(|d| d.bus.is_none() && d.ferry.is_none() && !net.is_twin(line)) else {
            return self.refuse(Issue::NoSuchLine { line });
        };
        match (d.twin, on) {
            (Twin::Parent { .. }, true) | (Twin::None, false) => return 0,
            (Twin::Parent { split }, false) => {
                if split {
                    double(&mut d);
                }
                d.twin = Twin::None;
            }
            _ => d.twin = Twin::Parent { split: false },
        }
        self.run(Op::Line { id: line, data: Some(d) })
    }

    /// 0 the switch off, 1 the line runs on both pairs, 2 the switch is on and it runs on one pair
    /// for now (`line_pairs_why`).
    pub fn line_pairs(&self, line: u32) -> u32 {
        match self.w.net.line_twin_of(line) {
            Twin::Parent { split: true } => 1,
            Twin::Parent { split: false } => 2,
            _ => 0,
        }
    }

    /// Whether line `line` could run on both pairs now (the app offers the switch only then, or while
    /// it is on).
    pub fn line_pairs_ok(&self, line: u32) -> bool {
        let net = &self.w.net;
        !net.is_twin(line) && (net.line_twin_of(line) == (Twin::Parent { split: true }) || net.twin_plan(line).is_ok())
    }

    /// Why line `line`, its switch on, runs on one pair: `[kind, index]` (`twin::Why::code`: 1 stop
    /// `index` has no station beside it on another track, 2 no track joins the stations beside stops
    /// `index` and `index + 1`, 3 its route there turns back elsewhere or has no way back, 4 the
    /// track or stations beside are not built, 5 not a rail line); empty when it runs on both or the
    /// switch is off.
    pub fn line_pairs_why(&self, line: u32) -> Vec<f64> {
        self.w.net.twin_why(line).map_or(vec![], |w| w.code().to_vec())
    }

    /// The hidden twin of line `line` (a line run on both pairs), else -1.
    pub fn line_twin(&self, line: u32) -> i32 {
        self.w.net.twin_line(line).map_or(-1, |t| t as i32)
    }

    /// The line a hidden twin runs the other pair for, else -1.
    pub fn line_twin_of(&self, line: u32) -> i32 {
        match self.w.net.line_twin_of(line) {
            Twin::Of(p) if self.w.net.line_ok(line) => p as i32,
            _ => -1,
        }
    }
}
