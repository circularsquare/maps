//! T-3297: a game's running side turned over once track is laid (`TrackWorld::flip_right_hand`), and
//! what the player has to look at afterwards.

use super::*;
use crate::track::capacity::junction_crosses;

/// `TrackApi::flip_right_hand`'s sites, each `[kind, id]`: a piece of single track turned round
/// (edge), a line moved through its same stops (line), a line left broken (line), a junction whose
/// moves now cross flat where they did not (node), one-way track no train can use any more (node).
pub const FLIP_REVERSED: f64 = 1.0;
pub const FLIP_MOVED: f64 = 2.0;
pub const FLIP_BROKEN: f64 = 3.0;
pub const FLIP_FLAT: f64 = 4.0;
pub const FLIP_UNUSABLE: f64 = 5.0;
/// T-3332: a one-track station whose trains now run the other way (node).
pub const FLIP_ONE_TRACK: f64 = 6.0;

#[wasm_bindgen]
impl TrackApi {
    /// T-3297: trains, trams and buses to the other side for the whole network (`TrackWorld::flip_right_hand`:
    /// the single track set one way where it meets double track turned round, broken lines moved through
    /// their stops, the undo history dropped), then the sites, two numbers each (`FLIP_*`): what was fixed
    /// and what still needs the player. Ids are kept.
    pub fn flip_right_hand(&mut self) -> Vec<f64> {
        let net = &self.w.net;
        let live = |n: u32| net.node_ok(n);
        let flat_before: Vec<bool> = (0..net.node_count() as u32).map(|n| live(n) && junction_crosses(net, n)).collect();
        let unusable_before: Vec<bool> = (0..net.node_count() as u32).map(|n| live(n) && net.node_unusable(n) != 0).collect();
        let one_track: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.one_side(n).is_some()).collect();
        let (reversed, moved, broken) = self.w.flip_right_hand();
        let net = &self.w.net;
        let mut out = vec![];
        out.extend(reversed.iter().flat_map(|&e| [FLIP_REVERSED, e as f64]));
        out.extend(moved.iter().flat_map(|&l| [FLIP_MOVED, l as f64]));
        out.extend(broken.iter().flat_map(|&l| [FLIP_BROKEN, l as f64]));
        out.extend(one_track.iter().flat_map(|&n| [FLIP_ONE_TRACK, n as f64]));
        for n in 0..net.node_count() as u32 {
            if !net.node_ok(n) {
                continue;
            }
            if junction_crosses(net, n) && !flat_before.get(n as usize).copied().unwrap_or(false) {
                out.extend([FLIP_FLAT, n as f64]);
            }
            if net.node_unusable(n) != 0 && !unusable_before.get(n as usize).copied().unwrap_or(false) {
                out.extend([FLIP_UNUSABLE, n as f64]);
            }
        }
        out
    }
}
