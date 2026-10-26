//! Station upgrades (T-1281): bought like a flyover, into the blueprint with a blueprint station or
//! paid at once on a constructed one.

use super::*;
use crate::track::upgrades;

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
        upgrades::price(bit, net.node_level[i], net.water_mask.is_water(net.node_x[i], net.node_y[i]))
    }

    /// Stations with upgrades: node, upgrade bits, flat.
    pub fn station_upgrades(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32)
            .filter(|&n| net.is_station(n) && net.node_upgrades[n as usize] != 0)
            .flat_map(|n| [n, net.node_upgrades[n as usize] as u32])
            .collect()
    }
}
