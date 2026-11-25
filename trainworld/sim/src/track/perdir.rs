//! T-3348 stage 1 (notes/T-3348.md): per-direction stops. A line stops at a station, and each run
//! stops at the platform on its own track: where a stop's own platform is not on a run's way (a
//! one-track station's platform on the other track; a one-way station is a stop of its own, T-3605), that
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
    /// port. T-3605 (Anita 2026-11-17): none for a one-way station (`leg_heading`), nor is one in any
    /// group: it is a stop of its own, served one way, so the stop list says where trains stop.
    fn way_group(&self, s: u32) -> Vec<u32> {
        if self.leg_heading(s).is_some() {
            return vec![];
        }
        let i = s as usize;
        let family = self.node_kind(s).spec().family;
        let dist = |m: u32| (self.node_x[m as usize] - self.node_x[i]).hypot(self.node_y[m as usize] - self.node_y[i]);
        let reach = WAY_WALK_M * self.k_at(self.node_y[i]);
        let ok = |m: u32| m != s && self.is_station(m) && !self.trackless(m) && self.node_kind(m).spec().family == family && self.leg_heading(m).is_none();
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
        self.way_group(s).into_iter().filter(|&m| self.way_limited(m)).collect()
    }

    /// T-3565: the heading (radians, world frame) trains run through station `s` on one-way single
    /// track, a one-way station; None for any other node (a one-track station on double track,
    /// T-3332, is none).
    pub fn leg_heading(&self, s: u32) -> Option<f64> {
        let i = s as usize;
        if !self.node_ok(s) || !self.is_station(s) || self.trackless(s) || self.one_side(s).is_some() {
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

    /// T-3565: whether station `s` is a one-way station (`leg_heading`): a stop of its own, served
    /// once a round trip by the run along its track, never paired with one on the other leg (T-3605).
    pub fn leg_stop(&self, s: u32) -> bool {
        self.leg_heading(s).is_some()
    }

    /// T-3711: the side of station `s` (its port side) trains arrive by where a line's one-way stop
    /// the player set (`ones`, `LineData::ones`) serves them; None where `s` is none, its side is not
    /// fixed yet, or the station is a one-way station already (its track says the way) or no plain
    /// two-port station.
    pub fn one_stop_side(&self, ones: &[(u32, u8)], s: u32) -> Option<u8> {
        let &(_, w) = ones.iter().find(|x| x.0 == s && x.1 < 2)?;
        self.can_one_stop(s).then_some(w)
    }

    /// T-3711: station `s` can be made a one-way stop: a plain station through which track runs
    /// (two port sides), serving both ways (no one-way station, no platform on one track only).
    pub fn can_one_stop(&self, s: u32) -> bool {
        let i = s as usize;
        self.node_ok(s) && self.is_station(s) && !self.trackless(s) && self.one_side(s).is_none() && self.node_ports[i].n == 2 && !self.leg_stop(s)
    }

    /// The port side of the node a path's entry `p` arrives at, by which it arrives.
    pub fn arrive_side(&self, p: u32) -> Option<u8> {
        self.port_side(path_edge(p), 1 - path_dir(p))
    }

    /// Whether a run on `path` stops at stop `s` a line serves once (`legs`): its path passes the
    /// station, arriving by the side a one-way stop the player set serves (`one_stop_side`).
    pub fn path_serves(&self, path: &[u32], s: u32, ones: &[(u32, u8)]) -> bool {
        match self.one_stop_side(ones, s) {
            None => self.path_has_node(path, s),
            Some(w) => path.iter().any(|&p| self.path_nodes(p).1 == s && self.arrive_side(p) == Some(w)),
        }
    }

    /// Per stop of a line, whether it is served once a round trip: a one-way station (`leg_stop`)
    /// or a one-way stop the player set (`one_stop_side`, T-3711); never an end, nor any stop of a
    /// loop.
    pub fn legs(&self, stops: &[u32], ones: &[(u32, u8)]) -> Vec<bool> {
        let n = stops.len();
        let leg = |s: u32| self.leg_stop(s) || self.one_stop_side(ones, s).is_some();
        if n < 3 || stops[0] == stops[n - 1] || !stops[1..n - 1].iter().any(|&s| leg(s)) {
            return vec![false; n];
        }
        (0..n).map(|j| j > 0 && j + 1 < n && leg(stops[j])).collect()
    }

    /// `match_stops` for one run of a line: where each of `stops` is among the run's path `nodes`
    /// (`must`: where it turns back), and whether the run stops there. A stop served once (`legs`)
    /// on the other run's way is passed: it keeps a place between its neighbours, at the node there
    /// nearest it, so every per-stop array keeps the line's stops. A one-way stop the player set
    /// (`ones`) is stopped at only where the run arrives by its side, read from `path` (the run's
    /// path, whose nodes `nodes` are). None when the stops do not match.
    pub fn match_run(&self, nodes: &[u32], stops: &[u32], must: &[usize], ones: &[(u32, u8)], path: &[u32]) -> Option<(Vec<usize>, Vec<bool>)> {
        let legs = self.legs(stops, ones);
        // each kept one-way stop the player set arrived at by its side (`at` per kept stop)
        let sides_ok = |keep: &[bool], at: &[usize]| {
            let mut q = 0;
            (0..stops.len()).all(|j| {
                if !keep[j] {
                    return true;
                }
                let k = at[q];
                q += 1;
                !legs[j] || self.one_stop_side(ones, stops[j]).map_or(true, |w| k > 0 && k <= path.len() && self.arrive_side(path[k - 1]) == Some(w))
            })
        };
        let all = vec![true; stops.len()];
        if let Some(at) = match_stops(nodes, stops, must) {
            if sides_ok(&all, &at) {
                return Some((at, all));
            }
        }
        if !legs.contains(&true) {
            return None;
        }
        // the rest must match; then each one stopped at once on the run's path, in the line's order,
        // is stopped at where the stops kept so far still match with it
        let mut keep: Vec<bool> = legs.iter().map(|&l| !l).collect();
        let kept = |keep: &[bool]| -> Vec<u32> { stops.iter().zip(keep).filter(|x| *x.1).map(|x| *x.0).collect() };
        let mut kat = match_stops(nodes, &kept(&keep), must)?;
        for j in 0..stops.len() {
            if !legs[j] || !nodes.contains(&stops[j]) {
                continue;
            }
            keep[j] = true;
            match match_stops(nodes, &kept(&keep), must).filter(|at| sides_ok(&keep, at)) {
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
    /// at the others (`match_run` on each). A one-way station neither of those passes (faster track
    /// beside its leg, T-3651) is taken by the run that can without turning back more often
    /// (`leg_run`). One neither way stops at even so (off both ways' track, or listed against the
    /// way its track runs) is routed through as any stop. None when no stop is a one-way station, or
    /// that still leaves one neither way stops at: then the route goes through every stop.
    pub fn route_legs(&self, stops: &[u32], ones: &[(u32, u8)]) -> Option<Vec<u32>> {
        self.route_legs_t(stops, ones, self.stops_tram(stops))
    }

    /// `route_legs` for a tram line or a train line (`tram`, T-3889).
    pub fn route_legs_t(&self, stops: &[u32], ones: &[(u32, u8)], tram: bool) -> Option<Vec<u32>> {
        self.route_legs_with(stops, ones, |stops| self.route_hops_plain_t(stops, tram).ok())
    }

    pub(crate) fn route_legs_with(
        &self, stops: &[u32], ones: &[(u32, u8)], mut route: impl FnMut(&[u32]) -> Option<Vec<u32>>,
    ) -> Option<Vec<u32>> {
        let mut legs = self.legs(stops, ones);
        let nodes = |p: &[u32]| -> Vec<u32> { std::iter::once(self.path_nodes(p[0]).0).chain(p.iter().map(|&q| self.path_nodes(q).1)).collect() };
        let n = stops.len();
        let rs: Vec<u32> = stops.iter().rev().copied().collect();
        let on = |p: &[u32], s: u32| self.path_serves(p, s, ones);
        while legs.contains(&true) {
            let through: Vec<u32> = (0..n).filter(|&j| !legs[j]).map(|j| stops[j]).collect();
            let mut base = route(&through).filter(|p| !p.is_empty())?;
            let back: Vec<u32> = through.iter().rev().copied().collect();
            let mut back = route(&back).filter(|p| !p.is_empty())?;
            // the ones served once that neither run passes yet, which a run then takes where it can
            let off: Vec<bool> = (0..n).map(|j| legs[j] && !on(&base, stops[j]) && !on(&back, stops[j])).collect();
            if off.contains(&true) {
                base = self.leg_run(stops, &legs, &off, ones, &mut route)?;
                let roff: Vec<bool> = off.iter().rev().copied().collect();
                let rlegs: Vec<bool> = legs.iter().rev().copied().collect();
                let still: Vec<bool> = (0..n).map(|j| roff[j] && !on(&base, stops[n - 1 - j])).collect();
                if still.contains(&true) {
                    back = self.leg_run(&rs, &rlegs, &still, ones, &mut route)?;
                }
            }
            let (_, fwd) = self.match_run(&nodes(&base), stops, &self.stop_reversals(&base), ones, &base)?;
            let (_, rev) = self.match_run(&nodes(&back), &rs, &self.stop_reversals(&back), ones, &back)?;
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

    /// T-3651: one run's route through `stops`: every stop not a one-way station (`legs` false), and
    /// each one-way station of `want`, in the stops' order, that the run can take without turning
    /// back more often. The quickest route through the rest alone may pass none of them where faster
    /// track runs beside the legs (passing tracks between two side platforms), and then neither run
    /// would stop there.
    fn leg_run(
        &self, stops: &[u32], legs: &[bool], want: &[bool], ones: &[(u32, u8)],
        route: &mut impl FnMut(&[u32]) -> Option<Vec<u32>>,
    ) -> Option<Vec<u32>> {
        let mut keep: Vec<bool> = legs.iter().map(|&l| !l).collect();
        let kept = |keep: &[bool]| -> Vec<u32> { stops.iter().zip(keep).filter(|x| *x.1).map(|x| *x.0).collect() };
        let mut path = route(&kept(&keep)).filter(|p| !p.is_empty())?;
        let turns = self.path_reversals(&path).len();
        for j in 0..stops.len() {
            if !want[j] || self.path_serves(&path, stops[j], ones) {
                continue;
            }
            keep[j] = true;
            // a one-way stop the player set only where the run arrives by its side (T-3711)
            match route(&kept(&keep)).filter(|p| !p.is_empty() && self.path_reversals(p).len() <= turns && self.path_serves(p, stops[j], ones)) {
                Some(p) => path = p,
                None => keep[j] = false,
            }
        }
        Some(path)
    }

    /// Whether `path` runs through node `n` (its first node included).
    fn path_has_node(&self, path: &[u32], n: u32) -> bool {
        path.first().is_some_and(|&p| self.path_nodes(p).0 == n) || path.iter().any(|&p| self.path_nodes(p).1 == n)
    }

    /// T-3651: `stops` (a run's stops in its order) with each of `extra` (one-way stations, none of
    /// them on the route through `stops`) put in where that run takes it without turning back more
    /// often, at the place its route costs least; those it cannot take are left out. The stops and
    /// their route, or None where `stops` have none. `tram`: a tram line's (T-3889).
    pub fn take_legs(&self, stops: &[u32], extra: &[u32], tram: bool) -> Option<(Vec<u32>, Vec<u32>)> {
        // `route_hops_plain` hop by hop (stops here are never a loop's), each hop's path and the side
        // it arrives on, so a slot reroutes only its own two hops and, where it arrives otherwise,
        // the hops after it
        let hops_from = |stops: &[u32], from: usize, side: Option<u8>| -> Option<Vec<(Vec<u32>, u8)>> {
            let mut side = side;
            let mut out = vec![];
            for w in stops[from..].windows(2) {
                let h = self.find_path_side(w[0], w[1], side, None, tram)?;
                side = Some(h.1);
                out.push(h);
            }
            Some(out)
        };
        let joined = |hops: &[(Vec<u32>, u8)]| -> Vec<u32> { hops.iter().flat_map(|h| h.0.iter().copied()).collect() };
        let len = |p: &[u32]| -> f64 { p.iter().map(|&q| self.edge_len[path_edge(q) as usize]).sum() };
        let mut stops = stops.to_vec();
        let mut hops = hops_from(&stops, 0, None)?;
        let base = joined(&hops);
        if base.is_empty() {
            return None;
        }
        let turns = self.path_reversals(&base).len();
        for &s in extra {
            if self.path_has_node(&joined(&hops), s) {
                continue;
            }
            let mut best: Option<(f64, usize, Vec<(Vec<u32>, u8)>)> = None;
            for slot in 1..stops.len() {
                let side = if slot >= 2 { Some(hops[slot - 2].1) } else { None };
                let Some(a) = self.find_path_side(stops[slot - 1], s, side, None, tram) else { continue };
                let Some(b) = self.find_path_side(s, stops[slot], Some(a.1), None, tram) else { continue };
                let mut t: Vec<(Vec<u32>, u8)> = hops[..slot - 1].to_vec();
                let same = b.1 == hops[slot - 1].1;
                let side = b.1;
                t.push(a);
                t.push(b);
                if same {
                    t.extend(hops[slot..].iter().cloned());
                } else {
                    let Some(rest) = hops_from(&stops, slot, Some(side)) else { continue };
                    t.extend(rest);
                }
                let p = joined(&t);
                if self.path_reversals(&p).len() > turns {
                    continue;
                }
                let c = len(&p);
                if best.as_ref().map_or(true, |x| c < x.0 - 0.001) {
                    best = Some((c, slot, t));
                }
            }
            if let Some((_, slot, t)) = best {
                stops.insert(slot, s);
                hops = t;
            }
        }
        let path = joined(&hops);
        Some((stops, path))
    }

    /// Each run's stop nodes for a rail line's `stops` and forward `path`: run 0's, and run 1's in
    /// run 0's order (reversed, run 1's own). A stop between the ends that a run cannot stop at its
    /// way (its platform on the other track, or its track not running that way) is that run's
    /// `way_station` where there is one; run 0's only where the forward path passes it. Ends, and
    /// stops where trains turn back, keep theirs. None when the stops do not match the path.
    pub fn way_stops(&self, stops: &[u32], path: &[u32], ones: &[(u32, u8)]) -> Option<(Vec<u32>, Vec<u32>)> {
        if path.is_empty() || !stops.iter().any(|&s| self.node_ok(s) && self.way_limited(s)) {
            return Some((stops.to_vec(), stops.to_vec()));
        }
        let nodes: Vec<u32> = std::iter::once(self.path_nodes(path[0]).0).chain(path.iter().map(|&p| self.path_nodes(p).1)).collect();
        let turns = self.stop_reversals(path);
        let (at, stopped) = self.match_run(&nodes, stops, &turns, ones, path)?;
        let rev = self.path_reversals(path);
        let legs = self.legs(stops, ones);
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
        if fwd != stops && self.match_run(&nodes, &fwd, &turns, ones, path).is_none() {
            fwd = stops.to_vec();
        }
        Some((fwd, back))
    }

    /// Run 0's stop nodes in its order (`way_stops`).
    pub fn fwd_stops_of(&self, stops: &[u32], path: &[u32], ones: &[(u32, u8)]) -> Vec<u32> {
        self.way_stops(stops, path, ones).map_or_else(|| stops.to_vec(), |w| w.0)
    }

    /// Run 1's stop nodes in its own order and its path (T-3094, T-3348): the forward path run
    /// backward where every way allows it and it passes them in order; else the route through them;
    /// failing that, the route through the line's own stops reversed. None when there is no way back.
    pub fn back_run_of(&self, stops: &[u32], path: &[u32], ones: &[(u32, u8)]) -> Option<(Vec<u32>, Vec<u32>)> {
        let rs: Vec<u32> = stops.iter().rev().copied().collect();
        let mut own = self.way_stops(stops, path, ones).map_or_else(|| stops.to_vec(), |w| w.1);
        own.reverse();
        let back: Vec<u32> = path.iter().rev().map(|p| p ^ 1).collect();
        // T-3651: a one-way station off the forward path is the way back's to stop at, so the forward
        // path run backward (the same track) will not do
        let legs = self.legs(stops, ones);
        let legs_on = stops.iter().zip(&legs).all(|(&s, &l)| !l || self.path_serves(path, s, ones));
        // (an empty path has no stops to move, so `own == rs`)
        if legs_on && back.iter().all(|&p| self.dir_ok(p)) {
            let fits = own == rs || {
                let nodes: Vec<u32> = std::iter::once(self.path_nodes(back[0]).0).chain(back.iter().map(|&p| self.path_nodes(p).1)).collect();
                self.match_run(&nodes, &own, &self.stop_reversals(&back), ones, &back).is_some()
            };
            if fits {
                return Some((own, back));
            }
        }
        // T-3889: a tram line's, or a stretch of one on tram track
        let tram = self.stops_tram(stops) || self.path_tram(path);
        if let Ok(p) = self.route_hops_t(&own, ones, tram) {
            return Some((own, p));
        }
        if own != rs {
            if let Ok(p) = self.route_hops_t(&rs, ones, tram) {
                return Some((rs, p));
            }
        }
        None
    }

    /// `back_run_of` for line `l`.
    pub fn back_run(&self, l: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        self.back_run_of(self.line_stops(l), self.line_path(l), self.line_ones(l))
    }
}
