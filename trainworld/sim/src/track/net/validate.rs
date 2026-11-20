//! Validation of edges, nodes, stations and routes, and turns.

use super::*;

impl Network {
    /// Problems with the given edges, nodes (with their stations) and lines, and the crossings of
    /// those edges. Empty = valid.
    pub fn validate(&self, edges: &[u32], nodes: &[u32], lines: &[u32]) -> Vec<Issue> {
        let mut out = vec![];
        for &e in edges {
            if self.edge_ok(e) {
                self.check_edge(e, &mut out);
            }
        }
        for c in &self.crossings {
            if !matches!(c.kind, CrossKind::Flat(_)) && (edges.binary_search(&c.e1).is_ok() || edges.binary_search(&c.e2).is_ok()) {
                out.push(Issue::Crossing { e1: c.e1, e2: c.e2, x: c.x, y: c.y, overlap: c.kind == CrossKind::Overlap });
            }
        }
        for &n in nodes {
            if self.node_ok(n) {
                self.check_node(n, &mut out);
            }
        }
        for &l in lines {
            if self.line_ok(l) && !self.line_broken[l as usize] {
                if let Err(i) = self.check_line(l) {
                    out.push(i);
                }
            }
        }
        out
    }

    pub(super) fn check_edge(&self, e: u32, out: &mut Vec<Issue>) {
        let verts = self.edge_vertices(e);
        if let Some(p) = verts.iter().find(|p| !self.inside_world(p.x, p.y)) {
            out.push(Issue::OutsideWorld { x: p.x, y: p.y });
        }
        let f = geom::with_ramps(self.edge_ramps(e), || geom::fit_min(&verts, self.frame.scaled(), R_CAP, self.edge_kind[e as usize].spec().min_radius));
        for issue in f.issues {
            out.push(Issue::Geom { edge: e, issue });
        }
        let vert = self.edge_vert(e);
        // T-3106: street tram track runs at ground level over the bridge its street has
        let street = self.edge_kind[e as usize] == TrackKind::StreetTram;
        for w in self.edge_water(e).iter().filter(|_| !street) {
            // Ground level over water: anything nearer the ground than half a level.
            let mut s = w[0];
            while s <= w[1] {
                if geom::height_at(vert, s).abs() < LEVEL_H / 2.0 {
                    out.push(Issue::GroundOverWater { edge: e, s });
                    break;
                }
                s += WATER_STEP / 2.0;
            }
        }
        if !self.edge_water(e).is_empty() && !self.edge_crossover[e as usize] && self.bridged_water_through(e) > WATER_SPAN_MAX {
            out.push(Issue::WaterTooLong { edge: e });
        }
    }

    /// The parts of an edge's water spans that are not bored tunnel (level -1 or above), as
    /// chainage intervals sampled every `BRIDGED_STEP` (T-1930).
    pub(super) fn bridged_water(&self, e: u32) -> Vec<[f64; 2]> {
        const BRIDGED_STEP: f64 = 25.0;
        let vert = self.edge_vert(e);
        let mut out: Vec<[f64; 2]> = vec![];
        for w in self.edge_water(e) {
            let n = ((w[1] - w[0]) / BRIDGED_STEP).ceil().max(1.0) as usize;
            let step = (w[1] - w[0]) / n as f64;
            let mut open: Option<f64> = None;
            for j in 0..n {
                let bridged = cost::level_of(geom::height_at(vert, w[0] + (j as f64 + 0.5) * step)) >= -1;
                match (bridged, open) {
                    (true, None) => open = Some(w[0] + j as f64 * step),
                    (false, Some(s0)) => {
                        out.push([s0, w[0] + j as f64 * step]);
                        open = None;
                    }
                    _ => {}
                }
            }
            if let Some(s0) = open {
                out.push([s0, w[1]]);
            }
        }
        out
    }

    /// The longest stretch of viaduct or cut and cover over unbroken water that runs through edge
    /// `e`, m, counting the same stretch on the edges meeting it at a node in the water (T-1930).
    /// Stops adding past `WATER_SPAN_MAX`.
    pub(super) fn bridged_water_through(&self, e: u32) -> f64 {
        const TOL: f64 = 1.0;
        let len = self.edge_len[e as usize];
        let mut best = 0.0f64;
        for r in self.bridged_water(e) {
            let mut total = r[1] - r[0];
            for (which, touches) in [(0u32, r[0] <= TOL), (1, r[1] >= len - TOL)] {
                if touches && total <= WATER_SPAN_MAX {
                    total += self.bridged_water_beyond(e, which, &mut vec![e], WATER_SPAN_MAX - total);
                }
            }
            best = best.max(total);
        }
        best
    }

    /// The most water run an end of edge `e` can add, m: of the other edges at that end's node, the
    /// one whose run from the node is longest, through its far end in turn where it spans the edge.
    /// `room` is how much more matters.
    pub(super) fn bridged_water_beyond(&self, e: u32, which: u32, seen: &mut Vec<u32>, room: f64) -> f64 {
        const TOL: f64 = 1.0;
        let node = self.end_node(e, which) as usize;
        let mut best = 0.0f64;
        for &c in self.node_ports[node].ends() {
            let (o, ow) = (c >> 1, c & 1);
            if o == e || seen.contains(&o) || self.edge_crossover[o as usize] || !self.edge_ok(o) {
                continue;
            }
            let olen = self.edge_len[o as usize];
            let Some(r) = self.bridged_water(o).into_iter().find(|r| if ow == 0 { r[0] <= TOL } else { r[1] >= olen - TOL }) else { continue };
            let mut total = r[1] - r[0];
            if total < room && r[0] <= TOL && r[1] >= olen - TOL {
                seen.push(o);
                total += self.bridged_water_beyond(o, 1 - ow, seen, room - total);
                seen.pop();
            }
            best = best.max(total);
            if best >= room {
                break;
            }
        }
        best
    }

    pub(super) fn check_node(&self, n: u32, out: &mut Vec<Issue>) {
        let i = n as usize;
        if !self.inside_world(self.node_x[i], self.node_y[i]) {
            out.push(Issue::OutsideWorld { x: self.node_x[i], y: self.node_y[i] });
        }
        let p = &self.node_ports[i];
        let h = self.node_heading[i];
        for (k, &c) in p.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            if pcs.is_empty() {
                continue;
            }
            let th = if which == 0 { geom::pos_at(pcs, 0.0).2 } else { geom::pos_at(pcs, self.edge_len[e as usize]).2 + PI };
            let d = geom::wrap(th - h);
            let off = if p.side[k] == 1 { d.abs() } else { PI - d.abs() };
            if off > HEADING_TOL {
                out.push(Issue::NodeHeading { node: n, edge: e });
            }
        }
        let mut fam = p.ends().iter().map(|&c| self.edge_kind[(c >> 1) as usize].spec().family);
        if let Some(f0) = fam.next() {
            if fam.any(|f| f != f0) {
                out.push(Issue::TrackFamily { node: n });
            }
        }
        // Two ends leaving the same way with nothing behind is a harmless fork (e.g. a junction
        // whose stem was removed); a station must have its platform track run through.
        if p.n >= 2 && self.node_platform[i] > 0 {
            let b = p.side[..p.n as usize].iter().filter(|&&s| s == 1).count();
            if b == 0 || b == p.n as usize {
                out.push(Issue::NodeOneSided { node: n });
            }
        }
        if self.node_platform[i] > 0 {
            self.check_station(n, out);
        }
    }

    /// The node sequence of a line's path, or the first problem with it.
    pub fn check_line(&self, l: u32) -> Result<Vec<u32>, Issue> {
        if let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            return if super::bus::route_ok(self, &r.points, &r.kinds) { Ok(vec![]) } else { Err(Issue::LinePath { line: l }) };
        }
        if let Some(r) = self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            return super::ferry::check(self, l, self.line_stops(l), r).map(|_| vec![]);
        }
        self.check_route(l, self.line_stops(l), self.line_path(l), self.line_runs_once(l))
    }

    /// `check_line` for a rail line's stops and path not yet on a line (`once`: it runs its path once
    /// round, one way); `l` names the line in the issue (T-3333: a twin is checked before it is made).
    pub fn check_route(&self, l: u32, stops: &[u32], path: &[u32], once: bool) -> Result<Vec<u32>, Issue> {
        if stops.len() < 2 && path.is_empty() {
            if stops.iter().all(|&s| self.node_ok(s) && self.node_platform[s as usize] > 0) { return Ok(stops.to_vec()); }
            return Err(Issue::LineStops { line: l });
        }
        if path.is_empty() {
            return Err(Issue::LinePath { line: l });
        }
        if stops.len() < 2 || stops.windows(2).any(|w| w[0] == w[1]) {
            return Err(Issue::LineStops { line: l });
        }
        for &p in path {
            if !self.edge_ok(path_edge(p)) || !self.dir_ok(p) {
                return Err(Issue::LinePath { line: l });
            }
        }
        let mut nodes = Vec::with_capacity(path.len() + 1);
        nodes.push(self.path_nodes(path[0]).0);
        for k in 0..path.len() {
            let (from, to) = self.path_nodes(path[k]);
            if k > 0 {
                let prev = path[k - 1];
                if self.path_nodes(prev).1 != from {
                    return Err(Issue::LinePath { line: l });
                }
                // Through a node: a move its table holds, or a turn back at a stop or a bare track
                // end (T-797, checked below).
                if self.port_side(path_edge(prev), 1 - path_dir(prev)).is_none() || !(self.turns_back(prev, path[k]) || self.move_ok(prev, path[k])) {
                    return Err(Issue::LinePath { line: l });
                }
            }
            nodes.push(to);
        }
        if stops[0] != nodes[0] || *stops.last().unwrap() != *nodes.last().unwrap() {
            return Err(Issue::LineStops { line: l });
        }
        if stops.iter().any(|&s| !self.node_ok(s) || self.node_platform[s as usize] == 0) {
            return Err(Issue::LineStops { line: l });
        }
        let turns = self.stop_reversals(path);
        let ones = self.line_ones(l);
        let Some((at, _)) = self.match_run(&nodes, stops, &turns, ones, path) else {
            return Err(if turns.is_empty() { Issue::LineStops { line: l } } else { Issue::LinePath { line: l } });
        };
        // the return run goes round one-way track when the path back is against it (T-3094), and each
        // run stops at the platform on its own track (T-3348)
        let back = if once { None } else {
            match self.back_run_of(stops, path, ones) {
                Some(b) => Some(b),
                None => return Err(Issue::NoWayBack { line: l }),
            }
        };
        let fwd = self.fwd_stops_of(stops, path, ones);
        let one = |s: &[u32]| s.iter().any(|&s| self.one_side(s).is_some());
        if one(stops) || one(&fwd) || back.as_ref().is_some_and(|b| one(&b.0)) {
            let at = if fwd == stops { at } else { self.match_run(&nodes, &fwd, &turns, ones, path).map(|m| m.0).ok_or(Issue::LinePath { line: l })? };
            self.check_one_track(l, stops, path, &at, back.as_ref().map(|b| (&b.0[..], &b.1[..])))?;
        }
        Ok(nodes)
    }

    /// T-3332: a line's stops at one-track stations on double track: each stopped at by at least one of
    /// its runs, and its ends (a loop's first stop too) arrived at on the platform's track. `at`
    /// places run 0's stop nodes on `path`; `back` is run 1's stop nodes in its order and its path
    /// (T-3348).
    pub(super) fn check_one_track(&self, l: u32, stops: &[u32], path: &[u32], at: &[usize], back: Option<(&[u32], &[u32])>) -> Result<(), Issue> {
        let circle = self.loop_kind(stops, path) == LoopKind::Circle;
        let idx: Vec<u32> = at.iter().map(|&k| k as u32).collect();
        let fwd = self.run_passes(path, &idx, circle);
        let rev = match back {
            Some((rs, b)) => {
                let nodes: Vec<u32> = std::iter::once(self.path_nodes(b[0]).0).chain(b.iter().map(|&p| self.path_nodes(p).1)).collect();
                let ra = self.match_run(&nodes, rs, &self.stop_reversals(b), self.line_ones(l), b).map(|m| m.0).ok_or(Issue::LinePath { line: l })?;
                let mut r = self.run_passes(b, &ra.into_iter().map(|k| k as u32).collect::<Vec<_>>(), circle);
                r.reverse();
                Some(r)
            }
            None => None,
        };
        let n = stops.len();
        let turns = self.path_reversals(path);
        for j in 0..n {
            // a turn back partway arrives on the same track both ways
            let end = j == 0 || j + 1 == n || turns.contains(&at[j]);
            let passed = fwd[j] && rev.as_ref().map_or(true, |r| r[j]);
            if passed || (end && (fwd[j] || rev.as_ref().is_some_and(|r| r[j]))) {
                return Err(Issue::OneTrackStop { line: l, node: stops[j] });
            }
        }
        Ok(())
    }

    /// Whether a path turns back between entries `a` and `b`: it arrives at their node and leaves
    /// on the same side (T-797).
    pub fn turns_back(&self, a: u32, b: u32) -> bool {
        let arr = self.port_side(path_edge(a), 1 - path_dir(a));
        arr.is_some() && arr == self.port_side(path_edge(b), path_dir(b))
    }

    /// The indices into a path's node list (`path.len() + 1` nodes) where it turns back.
    pub fn path_reversals(&self, path: &[u32]) -> Vec<usize> {
        (1..path.len()).filter(|&k| self.turns_back(path[k - 1], path[k])).collect()
    }

    /// A track end with no station (T-797 B3): a train may reverse there without stopping for
    /// riders, as in a headshunt. T-3541: an end of constructed track with only blueprint beyond
    /// it is still one, so blueprint drawn on from it never moves a running line's turn there.
    pub fn bare_end(&self, n: u32) -> bool {
        self.node_platform[n as usize] == 0 && self.turn_end(n)
    }

    /// T-3595: a track end, station or not (one constructed end, as `bare_end`): a train may turn back
    /// there without stopping. A station there that is no stop of the line is run into and out of;
    /// building on from a loose end to a station so never leaves a line with nowhere to turn.
    pub fn turn_end(&self, n: u32) -> bool {
        let p = &self.node_ports[n as usize];
        p.n == 1 || (p.n > 1 && self.built_ports(n) == 1)
    }

    /// `path_reversals` that must fall on a stop: all but those at a track end (`turn_end`), where a
    /// train turns back whether the line stops there or not.
    pub fn stop_reversals(&self, path: &[u32]) -> Vec<usize> {
        self.path_reversals(path).into_iter().filter(|&k| !self.turn_end(self.path_nodes(path[k]).0)).collect()
    }

    /// What kind of loop stops and a path make (T-798).
    pub fn loop_kind(&self, stops: &[u32], path: &[u32]) -> LoopKind {
        match (stops.first(), stops.last(), path.first(), path.last()) {
            (Some(a), Some(b), Some(&p0), Some(&p1)) if stops.len() >= 3 && a == b => {
                if self.turns_back(p1, p0) { LoopKind::Balloon } else { LoopKind::Circle }
            }
            _ => LoopKind::None,
        }
    }

    pub fn line_loop(&self, l: u32) -> LoopKind {
        self.loop_kind(self.line_stops(l), self.line_path(l))
    }

    /// Where a line's trains reverse: partway, and where a balloon loop closes.
    pub fn turn_count(&self, stops: &[u32], path: &[u32]) -> usize {
        self.path_reversals(path).len() + (self.loop_kind(stops, path) == LoopKind::Balloon) as usize
    }

    // ---------------------------------------------------------------- cost
}
