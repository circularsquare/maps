//! T-3332: one-track stations (`StationData::one`): a station's platform on one track only, switched
//! from the station panel. Placing one is `set_station_one` before `add_station` (draw.rs).

use super::*;
use crate::track::upgrades;

#[wasm_bindgen]
impl TrackApi {
    /// T-3332: put a station's platforms on one track (`one` true) or on both. On double track a
    /// station going to one track keeps the way that leaves every line stopping there able to (the
    /// way along the station's axis first), and loses a terminal or passing tracks; on single track it
    /// becomes a stop where trains cannot pass. A blueprint station changes free; a constructed one
    /// pays the difference going to both tracks and gets nothing back going to one. A change a line
    /// stopping there cannot take is refused (`OneTrackStop` names it). One undoable edit.
    pub fn set_station_tracks(&mut self, node: u32, one: bool) -> u32 {
        let Some(old) = self.w.net.station_data(node).filter(|d| d.berth.is_none()) else { return self.refuse(Issue::NoSuchNode { node }) };
        if (old.one != 0) == one {
            return 0;
        }
        if !one {
            return self.station_one_op(node, StationData { one: 0, ..old });
        }
        if !self.one_fits(node) {
            return self.refuse(Issue::OneTrackHere { node });
        }
        let d = StationData { upgrades: old.upgrades & !(upgrades::TERMINAL | upgrades::PASSING), ..old };
        if self.w.net.node_tracks(node) == 1 {
            return self.station_one_op(node, StationData { one: 1, ..d });
        }
        // the way a trial takes without refusal, else the first way's refusal
        let first = StationData { one: 1, ..d.clone() };
        if self.w.trial(Op::Station { node, data: Some(first.clone()) }).is_ok() {
            return self.station_one_op(node, first);
        }
        let second = StationData { one: 2, ..d };
        if self.w.trial(Op::Station { node, data: Some(second.clone()) }).is_ok() {
            return self.station_one_op(node, second);
        }
        self.station_one_op(node, first)
    }

    /// T-3332: put a station on double track on the one track that serves trains going `heading` (radians,
    /// world frame), from both tracks or from the other track: one undoable edit, charged as
    /// `set_station_tracks` is. A line stopping there that cannot take it refuses the change.
    pub fn set_station_way(&mut self, node: u32, heading: f64) -> u32 {
        let Some(old) = self.w.net.station_data(node).filter(|d| d.berth.is_none()) else { return self.refuse(Issue::NoSuchNode { node }) };
        if !self.one_fits(node) || self.w.net.node_tracks(node) != 2 {
            return self.refuse(Issue::OneTrackHere { node });
        }
        let way = crate::track::net::one_for_travel(heading);
        if old.one == way {
            return 0;
        }
        self.station_one_op(node, StationData { one: way, upgrades: old.upgrades & !(upgrades::TERMINAL | upgrades::PASSING), ..old })
    }

    /// T-3332: turn a one-track station on double track round: its platform to the other track, so its
    /// trains are those running the other way. Free, undoable; refused where a line stopping there
    /// cannot take it.
    pub fn reverse_station(&mut self, node: u32) -> u32 {
        let Some(old) = self.w.net.station_data(node).filter(|d| d.one != 0) else { return self.refuse(Issue::NoSuchNode { node }) };
        if self.w.net.one_side(node).is_none() {
            return 0;
        }
        self.station_one_op(node, StationData { one: 3 - old.one, ..old })
    }

    /// T-3332: what a constructed one-track station pays, US$M, going to platforms on both tracks (the
    /// price difference, as `set_station_tracks` charges it); 0 for any other node.
    pub fn station_tracks_cost(&self, node: u32) -> f64 {
        let net = &self.w.net;
        if !net.one_track(node) {
            return 0.0;
        }
        let i = node as usize;
        let wet = net.water_mask.is_water(net.node_x[i], net.node_y[i]);
        let price = |tracks: u8| crate::track::cost::station_cost_k(net.node_kind(node), net.node_level[i], net.node_platform[i], tracks, wet);
        (price(net.node_tracks(node)) - price(1)).max(0.0)
    }

    /// T-3332: the station's platforms are on one track: 0 both, 1 one track on double track
    /// (`station_one_heading` gives its trains' way), 2 a stop with no passing loop on single track.
    pub fn station_one(&self, node: u32) -> u8 {
        let net = &self.w.net;
        if !net.one_track(node) { 0 } else if net.one_side(node).is_some() { 1 } else { 2 }
    }

    /// T-3332: the heading (radians, world frame) of the trains a one-track station on double track
    /// serves; NaN for any other node.
    pub fn station_one_heading(&self, node: u32) -> f64 {
        let net = &self.w.net;
        net.one_side(node).map_or(f64::NAN, |s| net.node_heading[node as usize] + if s == 0 { 0.0 } else { std::f64::consts::PI })
    }

    /// The constructed-or-not stations with one platform track (`Network::one_platform`), for the
    /// upkeep the app shows.
    pub fn one_platform_stations(&self) -> Vec<u32> {
        let net = &self.w.net;
        (0..net.node_count() as u32).filter(|&n| net.one_platform(n)).collect()
    }

    /// T-3332: the one-track stations, `[node, one]` each (`StationData::one`), for the map's drawing.
    pub fn one_track_stations(&self) -> Vec<f64> {
        let net = &self.w.net;
        (0..net.node_count() as u32).filter(|&n| net.one_track(n)).flat_map(|n| [n as f64, self.station_one(n) as f64]).collect()
    }
}

impl TrackApi {
    /// A station change from the panel: charged when constructed (going to both tracks costs the
    /// difference; to one, nothing back).
    fn station_one_op(&mut self, node: u32, d: StationData) -> u32 {
        let op = Op::Station { node, data: Some(d.clone()) };
        if d.built { self.run_charged(op) } else { self.run(op) }
    }

    /// A one-track station can go on node `node`: through track with the same tracks both sides, not
    /// one-way single track.
    pub(super) fn one_fits(&self, node: u32) -> bool {
        let net = &self.w.net;
        let p = &net.node_ports[node as usize];
        if p.n != 2 || p.side[0] == p.side[1] {
            return false;
        }
        let (e0, e1) = (p.end[0] >> 1, p.end[1] >> 1);
        let t = net.edge_tracks[e0 as usize];
        t == net.edge_tracks[e1 as usize] && (t == 2 || (net.edge_way(e0) == 0 && net.edge_way(e1) == 0))
    }
}
