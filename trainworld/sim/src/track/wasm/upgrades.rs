//! Station upgrades (T-1281): bought like a flyover, into the blueprint with a blueprint station or
//! paid at once on a constructed one.

use super::*;
use crate::track::upgrades;

/// T-3740: a new station within this many ground metres of a station with a fare zone takes the nearest one's
/// (DECISIONS 2026-11-19): about half the widest stop spacing inside a fare area, so infill joins and a line
/// drawn far out of the area does not.
pub(super) const NEIGHBOUR_FARE_M: f64 = 2000.0;

#[wasm_bindgen]
impl TrackApi {
    /// Add or remove one station upgrade (`upgrades::TERMINAL` and the like) at each of `nodes`, in
    /// one edit. On a constructed station it is paid at once and final, and removing one refunds
    /// nothing; at blueprint stations only it is an undoable blueprint edit. 1 when nothing changed.
    pub fn set_station_upgrade(&mut self, nodes: &[u32], bit: u8, on: bool) -> u32 {
        if bit.count_ones() != 1 || bit & upgrades::ALL == 0 {
            return self.refuse(Issue::NoSuchNode { node: nodes.first().copied().unwrap_or(u32::MAX) });
        }
        let (mut ops, mut built) = (vec![], false);
        for &node in nodes {
            let Some(mut d) = self.w.net.station_data(node) else { return self.refuse(Issue::NoSuchNode { node }) };
            // T-3332: a one-track station has no terminal and no passing tracks
            if on && d.one != 0 && bit & (upgrades::TERMINAL | upgrades::PASSING) != 0 {
                return self.refuse(Issue::OneTrackHere { node });
            }
            // T-3419: a port takes the waiting room and nothing else
            if on && d.berth.is_some() && bit != upgrades::WAITING_ROOM {
                return self.refuse(Issue::NotAtPort { node });
            }
            let ups = if on { d.upgrades | bit } else { d.upgrades & !bit };
            if ups != d.upgrades {
                built |= d.built;
                d.upgrades = ups;
                ops.push(Op::Station { node, data: Some(d) });
            }
        }
        if ops.is_empty() {
            self.money_short = None;
            self.issues.clear();
            return 1;
        }
        let op = if ops.len() == 1 { ops.pop().unwrap() } else { Op::Batch(ops) };
        if built { self.run_charged(op) } else { self.run(op) }
    }

    /// One upgrade's price at a station, US$M (0 for no station).
    pub fn station_upgrade_cost(&self, node: u32, bit: u8) -> f64 {
        let net = &self.w.net;
        if !net.is_station(node) {
            return 0.0;
        }
        let i = node as usize;
        upgrades::price(bit, net.node_level[i], net.water_mask.is_water(net.node_x[i], net.node_y[i]), net.node_platform[i], net.node_price_kind(node))
    }

    /// T-2231: the stations where `line`'s trains get passed or would pass a slower line's, with no
    /// passing tracks yet (`overtake::passing_spots`), in the line's order: what the line panel's
    /// button adds passing tracks at (one `set_station_upgrade` with `upgrades::PASSING`).
    pub fn passing_spots(&self, line: u32) -> Vec<u32> {
        crate::track::overtake::passing_spots(&self.w.net, &self.w.svc.lines, line)
    }

    /// T-2231: where `line` waits for faster trains at passing stations at a demand level (resource
    /// kind 13): per station, direction and faster line, `[station, faster line, wait per train s]`,
    /// flat, largest first.
    pub fn line_waits(&self, line: u32, level: usize) -> Vec<f64> {
        self.line_waits_at(line, self.w.net.row_slot(line as usize, level.min(crate::track::params::ROWS - 1)))
    }
    /// T-3542: `line_waits` in demand period `period` (0-4), at the slot serving it.
    pub fn line_waits_in(&self, line: u32, period: usize) -> Vec<f64> {
        self.line_waits_at(line, self.w.net.period_slot()[period.min(4)])
    }
    fn line_waits_at(&self, line: u32, lev: usize) -> Vec<f64> {
        let cap = &self.w.svc.cap;
        let mut v: Vec<[f64; 3]> = (0..cap.res.len())
            .filter_map(|r| {
                let f = cap.wait_for(r)?;
                (cap.delay[r][lev] > 0.0 && cap.users_of(r).iter().any(|u| u.owner && u.line == line)).then(|| [cap.res[r].node as f64, f as f64, cap.delay[r][lev]])
            })
            .collect();
        v.sort_by(|a, b| b[2].total_cmp(&a[2]));
        v.into_iter().flatten().collect()
    }

    /// Stations with upgrades: node, upgrade bits, flat.
    pub fn station_upgrades(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32)
            .filter(|&n| net.is_station(n) && net.node_upgrades[n as usize] != 0)
            .flat_map(|n| [n, net.node_upgrades[n as usize] as u32])
            .collect()
    }

    /// T-3922: stations whose upgrades pay upkeep (`Network::upgrade_upkeep_by_node`): node, upkeep a game day
    /// US$M, flat.
    pub fn station_upgrade_upkeep(&self) -> Vec<f64> {
        let (v, _) = self.w.net.upgrade_upkeep_by_node();
        v.iter().enumerate().filter(|(_, &u)| u > 0.0).flat_map(|(n, &u)| [n as f64, u]).collect()
    }

    /// T-3740: stations with a fare zone: node, code (`StationData::fare`), flat.
    pub fn station_fares(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32)
            .filter(|&n| net.is_station(n) && net.node_fare[n as usize] != 0)
            .flat_map(|n| [n, net.node_fare[n as usize]])
            .collect()
    }

    /// T-3740: give stations `nodes` the fare zone codes `codes` (one each, or one for all; 0 none),
    /// as one edit, free at constructed stations too. Nodes that are no station, and stations already
    /// so, are left out; 1 when none is left (no edit).
    pub fn set_station_fares(&mut self, nodes: &[u32], codes: &[u32]) -> u32 {
        if codes.len() != nodes.len() && codes.len() != 1 {
            return self.refuse(Issue::NoSuchNode { node: nodes.first().copied().unwrap_or(u32::MAX) });
        }
        let mut ops = vec![];
        for (k, &node) in nodes.iter().enumerate() {
            let fare = codes[if codes.len() == 1 { 0 } else { k }];
            let Some(old) = self.w.net.station_data(node) else { continue };
            if old.fare != fare && !ops.iter().any(|o| matches!(o, Op::Station { node: n, .. } if *n == node)) {
                ops.push(Op::Station { node, data: Some(StationData { fare, ..old }) });
            }
        }
        if ops.is_empty() {
            self.money_short = None;
            self.issues.clear();
            return 1;
        }
        self.run(if ops.len() == 1 { ops.pop().unwrap() } else { Op::Batch(ops) })
    }

    /// T-3740: take every station whose code names an area not in `areas` (the fares' area ids) out of its
    /// fare zone, as one edit on neither undo nor redo: removing an area is a setting, not undone, and an
    /// undo or redo may bring back a code from before it went. How many stations changed (0: none).
    pub fn tidy_station_fares(&mut self, areas: &[u32]) -> u32 {
        let net = &self.w.net;
        let ops: Vec<Op> = (0..net.node_count() as u32)
            .filter(|&n| net.is_station(n) && net.node_fare[n as usize] != 0 && !areas.contains(&(net.node_fare[n as usize] >> 8)))
            .filter_map(|node| net.station_data(node).map(|d| Op::Station { node, data: Some(StationData { fare: 0, ..d }) }))
            .collect();
        let n = ops.len() as u32;
        if n == 0 {
            return 0;
        }
        // the edit before (an undo this follows) stays the last one, whose changes the caller still reads
        let last = self.last.take();
        let r = self.run_unrecorded(Op::Batch(ops));
        self.last = last;
        if r != 0 { 0 } else { n }
    }

    /// T-3740: the fare zone code a new station at plane (x, y) takes: the code of the nearest station with
    /// one within `NEIGHBOUR_FARE_M` on the ground, else 0. Codes of removed areas are never left on the
    /// model (`tidy_station_fares`), so any code found is live.
    pub(super) fn neighbour_fare(&self, x: f64, y: f64) -> u32 {
        let net = &self.w.net;
        let reach = NEIGHBOUR_FARE_M * net.k_at(y);
        let (mut best, mut code) = (reach * reach, 0);
        for n in 0..net.node_count() {
            if net.node_fare[n] == 0 || !net.is_station(n as u32) {
                continue;
            }
            let d = (net.node_x[n] - x).powi(2) + (net.node_y[n] - y).powi(2);
            if d <= best {
                (best, code) = (d, net.node_fare[n]);
            }
        }
        code
    }

    /// Stations whose name the player typed (T-2683): their nodes.
    pub fn named_stations(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32).filter(|&n| net.is_station(n) && net.node_named[n as usize]).collect()
    }
}
