//! T-3348 stage 1 (notes/T-3348.md): per-direction stops. A line stops at a station, and each run
//! stops at the platform on its own track: where a stop's own platform is not on a run's way (one-way
//! single track running the other way, or a one-track station's platform on the other track), that
//! run stops at the nearest station of the stop's complex, or failing that within riders' change
//! walk, whose platform its trains pass on their own track. The line keeps one stop; nothing is saved.

use std::f64::consts::PI;

use super::net::{match_stops, path_dir, path_edge, Network};

/// Riders change on foot between stations up to this far apart (demand `Params::xfer_walk_m`),
/// ground m: how far a run's own platform may be from the stop's when the two are no complex.
const WAY_WALK_M: f64 = 300.0;
/// A station runs a run's way when its track there is within 60 degrees of the run's heading.
const WAY_COS: f64 = 0.5;

impl Network {
    /// Station `n`'s platform serves one way only: on one-way single track, or a one-track station
    /// on double track (T-3332).
    pub fn way_limited(&self, n: u32) -> bool {
        self.one_side(n).is_some()
            || self.node_ports[n as usize].ends().iter().any(|&c| self.edge_tracks[(c >> 1) as usize] == 1 && self.edge_way(c >> 1) != 0)
    }

    /// Whether trains running through station `m` along its heading (`along`), or against it, can
    /// stop at its platform.
    fn serves_way(&self, m: u32, along: bool) -> bool {
        let p = &self.node_ports[m as usize];
        if p.n != 2 {
            return false;
        }
        // a train arriving from side 0 runs along the heading
        let ws = if along { 0 } else { 1 };
        if self.one_side(m).is_some_and(|w| w != ws) {
            return false;
        }
        let (Some(i), Some(j)) = ((0..2).find(|&i| p.side[i] == ws), (0..2).find(|&j| p.side[j] != ws)) else { return false };
        self.node_moves[m as usize].to[i] >> j & 1 != 0
    }

    /// The stations a stop at `s` may use on a run its own platform does not serve, nearest first:
    /// the other stations of its complex, then any within `WAY_WALK_M`; of its family of track, no
    /// port.
    fn way_group(&self, s: u32) -> Vec<u32> {
        let i = s as usize;
        let family = self.node_kind(s).spec().family;
        let dist = |m: u32| (self.node_x[m as usize] - self.node_x[i]).hypot(self.node_y[m as usize] - self.node_y[i]);
        let ok = |m: u32| m != s && self.is_station(m) && self.node_berth[m as usize].is_none() && self.node_kind(m).spec().family == family;
        let mut out: Vec<u32> = self.complex_of(s).into_iter().filter(|&m| ok(m)).collect();
        out.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
        let reach = WAY_WALK_M * self.k_at(self.node_y[i]);
        let mut walk: Vec<u32> = (0..self.node_count() as u32).filter(|&m| ok(m) && !out.contains(&m) && dist(m) <= reach).collect();
        walk.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
        out.extend(walk);
        out
    }

    /// The station of `s`'s group (`way_group`) where a run heading `th` (radians, world frame) at
    /// `s` stops: the nearest whose platform its trains pass on their own track, none of `skip`.
    pub fn way_station(&self, s: u32, th: f64, skip: &[u32]) -> Option<u32> {
        self.way_group(s).into_iter().find(|&m| {
            let c = (th - self.node_heading[m as usize]).cos();
            !skip.contains(&m) && c.abs() >= WAY_COS && self.serves_way(m, c > 0.0)
        })
    }

    /// The stations a click adding station `s` to a line may add in its place (`best_slot`): where
    /// `s` serves one way only, the others of its group that do too, nearest first; so a line clicked
    /// onto the leg its trains do not run along stops on the one they do.
    pub fn way_others(&self, s: u32) -> Vec<u32> {
        if !self.is_station(s) || !self.way_limited(s) {
            return vec![];
        }
        self.way_group(s).into_iter().filter(|&m| self.way_limited(m)).collect()
    }

    /// Each run's stop nodes for a rail line's `stops` and forward `path`: run 0's, and run 1's in
    /// run 0's order (reversed, run 1's own). A stop between the ends that a run cannot stop at its
    /// way (its platform on the other track, or its track not running that way) is that run's
    /// `way_station` where there is one; run 0's only where the forward path passes it. Ends, and
    /// stops where trains turn back, keep theirs. None when the stops do not match the path.
    pub fn way_stops(&self, stops: &[u32], path: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
        if path.is_empty() || !stops.iter().any(|&s| self.node_ok(s) && self.way_limited(s)) {
            return Some((stops.to_vec(), stops.to_vec()));
        }
        let nodes: Vec<u32> = std::iter::once(self.path_nodes(path[0]).0).chain(path.iter().map(|&p| self.path_nodes(p).1)).collect();
        let turns = self.stop_reversals(path);
        let at = match_stops(&nodes, stops, &turns)?;
        let arrive = |p: u32| self.port_side(path_edge(p), 1 - path_dir(p));
        let (mut fwd, mut back) = (stops.to_vec(), stops.to_vec());
        for j in 1..stops.len().saturating_sub(1) {
            let (s, k) = (stops[j], at[j]);
            if !self.node_ok(s) || !self.way_limited(s) || k == 0 || k >= path.len() || turns.contains(&k) {
                continue;
            }
            let (pin, pout) = (path[k - 1], path[k]);
            let Some(side) = arrive(pin) else { continue };
            // the forward run's heading through the stop
            let th = self.node_heading[s as usize] + if side == 0 { 0.0 } else { PI };
            let one = self.one_side(s);
            if one.is_some_and(|w| w != side) {
                if let Some(m) = self.way_station(s, th, stops) {
                    fwd[j] = m;
                }
            }
            let back_ok = self.dir_ok(pout ^ 1) && self.dir_ok(pin ^ 1) && one.map_or(true, |w| arrive(pout ^ 1) == Some(w));
            if !back_ok {
                if let Some(m) = self.way_station(s, th + PI, stops) {
                    back[j] = m;
                }
            }
        }
        if fwd != stops && match_stops(&nodes, &fwd, &turns).is_none() {
            fwd = stops.to_vec();
        }
        Some((fwd, back))
    }

    /// Run 0's stop nodes in its order (`way_stops`).
    pub fn fwd_stops_of(&self, stops: &[u32], path: &[u32]) -> Vec<u32> {
        self.way_stops(stops, path).map_or_else(|| stops.to_vec(), |w| w.0)
    }

    /// Run 1's stop nodes in its own order and its path (T-3094, T-3348): the forward path run
    /// backward where every way allows it and it passes them in order; else the route through them;
    /// failing that, the route through the line's own stops reversed. None when there is no way back.
    pub fn back_run_of(&self, stops: &[u32], path: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
        let rs: Vec<u32> = stops.iter().rev().copied().collect();
        let mut own = self.way_stops(stops, path).map_or_else(|| stops.to_vec(), |w| w.1);
        own.reverse();
        let back: Vec<u32> = path.iter().rev().map(|p| p ^ 1).collect();
        // (an empty path has no stops to move, so `own == rs`)
        if back.iter().all(|&p| self.dir_ok(p)) {
            let fits = own == rs || {
                let nodes: Vec<u32> = std::iter::once(self.path_nodes(back[0]).0).chain(back.iter().map(|&p| self.path_nodes(p).1)).collect();
                match_stops(&nodes, &own, &self.stop_reversals(&back)).is_some()
            };
            if fits {
                return Some((own, back));
            }
        }
        if let Ok(p) = self.route_hops(&own) {
            return Some((own, p));
        }
        if own != rs {
            if let Ok(p) = self.route_hops(&rs) {
                return Some((rs, p));
            }
        }
        None
    }

    /// `back_run_of` for line `l`.
    pub fn back_run(&self, l: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        self.back_run_of(self.line_stops(l), self.line_path(l))
    }
}
