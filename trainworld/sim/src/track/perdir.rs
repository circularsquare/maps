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
/// T-3565 (Anita 2026-11-17): a one-way station's pair on the other leg of a one-way street pair
/// is at most this far, ground m: platforms side by side across the street.
const LEG_PAIR_M: f64 = 50.0;
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
        // T-3565: a one-way station pairs only with one whose platform is beside it, complex or not
        let leg = self.leg_heading(s).is_some();
        let reach = if leg { LEG_PAIR_M } else { WAY_WALK_M } * self.k_at(self.node_y[i]);
        let ok = |m: u32| m != s && self.is_station(m) && self.node_berth[m as usize].is_none() && self.node_kind(m).spec().family == family && (!leg || dist(m) <= reach);
        let mut out: Vec<u32> = self.complex_of(s).into_iter().filter(|&m| ok(m)).collect();
        out.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
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
        // T-3565: of a one-way station's, only those serving the other way: never a neighbour on its
        // own leg, which is a stop of its own
        let own = self.leg_heading(s);
        self.way_group(s).into_iter().filter(|&m| self.way_limited(m) && own.map_or(true, |th| self.serves_heading(m, th + PI))).collect()
    }

    /// Whether trains heading `th` (radians, world frame) stop at station `m`'s platform running
    /// within 60 degrees of it.
    fn serves_heading(&self, m: u32, th: f64) -> bool {
        let c = (th - self.node_heading[m as usize]).cos();
        c.abs() >= WAY_COS && self.serves_way(m, c > 0.0)
    }

    /// T-3565: the heading (radians, world frame) trains run through station `s` on one-way single
    /// track, a one-way station; None for any other node (a one-track station on double track,
    /// T-3332, is none).
    pub fn leg_heading(&self, s: u32) -> Option<f64> {
        let i = s as usize;
        if !self.node_ok(s) || !self.is_station(s) || self.node_berth[i].is_some() || self.one_side(s).is_some() {
            return None;
        }
        let p = &self.node_ports[i];
        if p.n != 2 || p.ends().iter().any(|&c| self.edge_tracks[(c >> 1) as usize] != 1 || self.edge_way(c >> 1) == 0) {
            return None;
        }
        // a train leaving by side 1 runs along the heading
        let k = (0..2).find(|&k| self.dir_ok(p.end[k]))?;
        Some(self.node_heading[i] + if p.side[k] == 1 { 0.0 } else { PI })
    }

    /// T-3565: whether one-way station `s` (`leg_heading`), a stop of a line with `stops`, is served
    /// once a round trip, by the run along its track. T-3348 keeps a lone one: one with a station of
    /// its group serving the other way one way only (the other leg's, `way_station`) that is no stop
    /// of the line is one stop, served each way at its own platform (`way_stops`). Dropping that
    /// rule is this function returning `leg_heading(s).is_some()`.
    pub fn leg_stop(&self, s: u32, stops: &[u32]) -> bool {
        self.leg_heading(s).is_some_and(|th| !self.way_group(s).into_iter().any(|m| !stops.contains(&m) && self.way_limited(m) && self.serves_heading(m, th + PI)))
    }

    /// Per stop of a line, whether it is a one-way station served once (`leg_stop`); never an end,
    /// nor any stop of a loop.
    pub fn legs(&self, stops: &[u32]) -> Vec<bool> {
        let n = stops.len();
        if n < 3 || stops[0] == stops[n - 1] || !stops[1..n - 1].iter().any(|&s| self.leg_heading(s).is_some()) {
            return vec![false; n];
        }
        (0..n).map(|j| j > 0 && j + 1 < n && self.leg_stop(stops[j], stops)).collect()
    }

    /// `match_stops` for one run of a line: where each of `stops` is among the run's path `nodes`
    /// (`must`: where it turns back), and whether the run stops there. A one-way station (`legs`)
    /// on the other run's way is passed: it keeps a place between its neighbours, at the node there
    /// nearest it, so every per-stop array keeps the line's stops. None when the stops do not match.
    pub fn match_run(&self, nodes: &[u32], stops: &[u32], must: &[usize]) -> Option<(Vec<usize>, Vec<bool>)> {
        if let Some(at) = match_stops(nodes, stops, must) {
            return Some((at, vec![true; stops.len()]));
        }
        let legs = self.legs(stops);
        if !legs.contains(&true) {
            return None;
        }
        // the rest must match; then each one-way station on the run's path, in the line's order, is
        // stopped at where the stops kept so far still match with it
        let mut keep: Vec<bool> = legs.iter().map(|&l| !l).collect();
        let kept = |keep: &[bool]| -> Vec<u32> { stops.iter().zip(keep).filter(|x| *x.1).map(|x| *x.0).collect() };
        let mut kat = match_stops(nodes, &kept(&keep), must)?;
        for j in 0..stops.len() {
            if !legs[j] || !nodes.contains(&stops[j]) {
                continue;
            }
            keep[j] = true;
            match match_stops(nodes, &kept(&keep), must) {
                Some(at) => kat = at,
                None => keep[j] = false,
            }
        }
        let (x, y) = (&self.node_x, &self.node_y);
        let mut it = kat.into_iter();
        let mut at: Vec<Option<usize>> = keep.iter().map(|&k| if k { it.next() } else { None }).collect();
        for j in 0..stops.len() {
            if keep[j] {
                continue;
            }
            // the ends are kept, so a stop before and after it is
            let lo = at[j - 1]?;
            let hi = (j + 1..stops.len()).find_map(|q| at[q])?;
            let s = stops[j] as usize;
            let d = |k: usize| (x[nodes[k] as usize] - x[s]).hypot(y[nodes[k] as usize] - y[s]);
            // never where the run turns back (a track end): that place is its turnaround's
            at[j] = Some((lo + 1..hi).filter(|&k| !must.contains(&k) && !self.turn_end(nodes[k])).min_by(|&a, &b| d(a).total_cmp(&d(b))).unwrap_or(lo));
        }
        Some((at.into_iter().collect::<Option<Vec<usize>>>()?, keep))
    }

    /// T-3565: a line's route (`route_hops`) where some stops are one-way stations (`legs`): through
    /// the rest, stopping at those it passes; the way back, routed through the rest backward, stops
    /// at the others (`match_run` on each). A one-way station neither way stops at (off both ways'
    /// track, or listed against the way its track runs) is routed through as any stop. None when no
    /// stop is a one-way station, or that still leaves one neither way stops at: then the route goes
    /// through every stop.
    pub fn route_legs(&self, stops: &[u32]) -> Option<Vec<u32>> {
        let mut legs = self.legs(stops);
        let nodes = |p: &[u32]| -> Vec<u32> { std::iter::once(self.path_nodes(p[0]).0).chain(p.iter().map(|&q| self.path_nodes(q).1)).collect() };
        let n = stops.len();
        let rs: Vec<u32> = stops.iter().rev().copied().collect();
        while legs.contains(&true) {
            let through: Vec<u32> = (0..n).filter(|&j| !legs[j]).map(|j| stops[j]).collect();
            let base = self.route_hops_plain(&through).ok().filter(|p| !p.is_empty())?;
            let back: Vec<u32> = through.iter().rev().copied().collect();
            let back = self.route_hops_plain(&back).ok().filter(|p| !p.is_empty())?;
            let (_, fwd) = self.match_run(&nodes(&base), stops, &self.stop_reversals(&base))?;
            let (_, rev) = self.match_run(&nodes(&back), &rs, &self.stop_reversals(&back))?;
            let missed: Vec<usize> = (0..n).filter(|&j| !fwd[j] && !rev[n - 1 - j]).collect();
            if missed.is_empty() {
                return Some(base);
            }
            for j in missed {
                if !legs[j] {
                    return None;
                }
                legs[j] = false;
            }
        }
        None
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
        let (at, stopped) = self.match_run(&nodes, stops, &turns)?;
        let rev = self.path_reversals(path);
        let legs = self.legs(stops);
        let arrive = |p: u32| self.port_side(path_edge(p), 1 - path_dir(p));
        let (mut fwd, mut back) = (stops.to_vec(), stops.to_vec());
        for j in 1..stops.len().saturating_sub(1) {
            let (s, k) = (stops[j], at[j]);
            if legs[j] || !stopped[j] || !self.node_ok(s) || !self.way_limited(s) || k == 0 || k >= path.len() || rev.contains(&k) {
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
        if fwd != stops && self.match_run(&nodes, &fwd, &turns).is_none() {
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
                self.match_run(&nodes, &own, &self.stop_reversals(&back)).is_some()
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
