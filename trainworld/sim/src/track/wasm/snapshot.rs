//! What the last edit changed, and the network for the main thread's snapshot.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- what the last edit changed
    /// What the last edit paid, US$M: construction plus trains (`last_train_charge`).
    pub fn last_charge(&self) -> f64 {
        self.last.as_ref().map_or(0.0, |r| r.charge + self.last_trains)
    }
    pub fn dirty_edges(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.edges.clone())
    }
    /// Lines whose profiles changed (re-upload their phase tables).
    pub fn dirty_lines(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.service.recomputed.clone())
    }
    /// T-1382: the lines whose rows the state must read again since the last call, ascending, or
    /// `[u32::MAX]` for all of them: a line the service rebuilt, rescheduled or recomputed, or that
    /// the net was changed under (every edit's, undo's, preview's, refused edit's and settle's,
    /// however many calls ago), or all after a load, a reload or edge prices worked out again for
    /// water or terrain. The caller keeps the other lines' rows.
    pub fn take_changed_lines(&mut self) -> Vec<u32> {
        let mut v = self.w.svc.take_changed();
        if self.w.net.take_repriced() && v.first() != Some(&u32::MAX) {
            v = vec![u32::MAX];
        }
        v
    }
    /// `take_changed_lines` without taking: what has changed since, for the phase tables to be read
    /// after the trials that pricing a planned line runs (T-1540).
    pub fn peek_changed_lines(&self) -> Vec<u32> {
        if self.w.net.repriced() {
            return vec![u32::MAX];
        }
        self.w.svc.peek_changed()
    }
    /// T-1382: the nodes whose station name was set or cleared since the last call, ascending, or
    /// `[u32::MAX]` when the caller knows none of them (a new network).
    pub fn take_name_changes(&mut self) -> Vec<u32> {
        self.w.net.take_name_changes()
    }
    /// Lines whose service changed for demand (stops, frequency, running, a time moving 5 s).
    pub fn dirty_demand_lines(&self) -> Vec<u32> {
        self.last.as_ref().map_or(vec![], |r| r.dirty.service.demand.clone())
    }

    // ---- the network, for the main thread's snapshot
    /// Per live edge, 10 numbers: id, a, b, tracks, constructed (1/0), length m, cost US$M,
    /// lowest speed limit km/h, lowest and highest level.
    pub fn edges_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for e in 0..n.edge_count() as u32 {
            if !n.edge_ok(e) {
                continue;
            }
            let i = e as usize;
            // the track's own: the player's limit on it (T-3405) is not a curve
            let vmin = n.edge_pieces(e).iter().map(|p| p.v_curve()).fold(n.v_kind_cap(e), f64::min);
            let zs = n.edge_vert(e).iter().map(|v| v[1]);
            let (lo, hi) = zs.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), z| (a.min(z), b.max(z)));
            out.extend([e as f64, n.edge_a[i] as f64, n.edge_b[i] as f64, n.edge_tracks[i] as f64, n.edge_built[i] as u8 as f64, n.edge_len[i], n.edge_cost[i], vmin * 3.6, (lo / LEVEL_H).round(), (hi / LEVEL_H).round()]);
        }
        out
    }
    /// An edge's centre line for drawing: x, y, height (m) triples.
    /// Curve-limited km/h for every consecutive segment in edge_render's point triples.
    pub fn edge_render_speeds(&self, e: u32) -> Vec<f32> {
        let n = &self.w.net;
        if !n.edge_ok(e) { return vec![]; }
        render_speeds(n.edge_pieces(e), n.edge_vert(e), n.v_cap(e))
    }

    pub fn edge_render(&self, e: u32) -> Vec<f64> {
        let mut out = vec![];
        if self.w.net.edge_ok(e) {
            render_samples(self.w.net.edge_pieces(e), self.w.net.edge_vert(e), &mut out);
        }
        out
    }
    /// An edge as a polyline (x, y pairs, local metres), `lateral` metres to the left.
    pub fn edge_polyline(&self, e: u32, lateral: f64) -> Vec<f32> {
        let mut pts = vec![];
        geom::sample(self.w.net.edge_pieces(e), 50.0, 0.05, lateral, &mut pts);
        pts.iter().flat_map(|p| [p[0] as f32, p[1] as f32]).collect()
    }
    /// Per live node, 10 numbers: id, x, y, level, edge ends, constructed edge ends, flags (1
    /// flying, 2 moves through it can cross, T-3103; 4 one-way track runs in with no way on, 8 one-way
    /// track runs out with no way in, T-3164), platform length (0 = no station), station
    /// constructed, heading (rad).
    pub fn nodes_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        let busy = crate::track::capacity::conflict_nodes(n, &self.w.svc.lines);
        for k in 0..n.node_count() as u32 {
            if !n.node_ok(k) {
                continue;
            }
            let i = k as usize;
            // T-3745: a built junction shows only where two moves some line runs cross; a blueprint one always
            let crosses = crate::track::capacity::junction_crosses(n, k) && (n.built_ports(k) < 3 || busy.binary_search(&k).is_ok());
            let flags = n.node_flying[i] as u8 | (crosses as u8) << 1 | n.node_unusable(k) << 2;
            out.extend([k as f64, n.node_x[i], n.node_y[i], n.node_level[i] as f64, n.node_ports[i].n as f64, n.built_ports(k) as f64, flags as f64, n.node_platform[i] as f64, n.node_built[i] as u8 as f64, n.node_heading[i]]);
        }
        out
    }
    /// Per crossing of two edges at one height, 6 numbers: x, y, kind (0 flat, 1 bad), level (a
    /// flat crossing's, else 0), built (1 when both its edges are constructed), tram tracks
    /// (`Network::crossing_trams`, T-3062). For the map's marks (T-629).
    pub fn crossings_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        let used = crate::track::capacity::edges_in_use(&self.w.svc.lines);
        for c in &n.crossings {
            let (kind, level) = match c.kind {
                CrossKind::Flat(l) => (0.0, l as f64),
                CrossKind::Bad => (1.0, 0.0),
                CrossKind::Overlap => continue,
            };
            let built = n.edge_built[c.e1 as usize] && n.edge_built[c.e2 as usize];
            // T-3745: a built flat crossing shows only with a line on both its edges
            if built && kind == 0.0 && !(used.binary_search(&c.e1).is_ok() && used.binary_search(&c.e2).is_ok()) {
                continue;
            }
            out.extend([c.x, c.y, kind, level, built as u8 as f64, n.crossing_trams(c.e1, c.e2) as f64]);
        }
        out
    }
    /// Where the last route, edge or node-move preview makes a crossing or a flat junction:
    /// x, y, kind (0 crossing, 1 junction), tram tracks of a crossing (`Network::crossing_trams`,
    /// T-3535) per mark (T-629).
    pub fn preview_marks(&self) -> Vec<f64> {
        self.preview_marks.clone()
    }
    /// Per segment of the last route or edge preview polyline (the segments `preview_render_speeds` has), its
    /// track: 0 single, 255 double (T-3134).
    pub fn preview_render_lanes(&self) -> Vec<u8> {
        self.preview_lanes.clone()
    }
    /// Per edge of the last node-move preview, its track: 0 single, 255 double (T-3134).
    pub fn preview_part_lanes(&self) -> Vec<u8> {
        self.preview_part_lanes.clone()
    }
    pub fn node_name(&self, id: u32) -> String {
        self.w.net.node_name.get(id as usize).cloned().unwrap_or_default()
    }
    /// Per live line, 22 numbers: id, colour (0xRRGGBB), ok, running, broken, tph high, medium,
    /// low, cars, path length m, round trip s x3, trains needed x3, delay s x3, dwell s,
    /// turnaround s, cars that fit.
    pub fn lines_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for l in 0..n.line_count() as u32 {
            if !n.line_ok(l) {
                continue;
            }
            let i = l as usize;
            let svc = self.w.svc.lines.get(i);
            let ok = svc.map_or(false, |s| s.ok);
            let running = svc.map_or(false, |s| s.running);
            out.extend([l as f64, n.line_colour[i] as f64, ok as u8 as f64, running as u8 as f64, n.line_broken[i] as u8 as f64]);
            // per row (high, medium, low), each from its slot
            out.extend(n.rows_of(n.line_tph[i]).iter().map(|&t| t as f64));
            out.push(n.line_cars[i] as f64);
            match svc.filter(|s| s.ok) {
                Some(s) => {
                    out.push(s.runs[0].len);
                    out.extend(n.rows_of(s.round_trip));
                    out.extend(n.rows_of(s.trains).iter().map(|&t| t as f64));
                    out.extend(n.rows_of(s.delay));
                    out.extend([n.line_dwell[i] as f64, n.line_turn[i] as f64, s.cars as f64]);
                }
                None => {
                    out.extend([0.0; 4]);
                    out.extend(n.line_trains[i].unwrap_or([0; 3]).iter().map(|&t| t as f64));
                    out.extend([0.0; 3]);
                    out.extend([n.line_dwell[i] as f64, n.line_turn[i] as f64, 0.0]);
                }
            }
        }
        out
    }
    pub fn line_name(&self, id: u32) -> String {
        self.w.net.line_name.get(id as usize).cloned().unwrap_or_default()
    }
    pub fn line_symbol(&self, id: u32) -> String {
        self.w.net.line_symbol.get(id as usize).cloned().unwrap_or_default()
    }
    /// A line's stops; a bus line's as its service places them (T-1464, `bus::plan`), empty while
    /// it has none.
    pub fn line_stops(&self, id: u32) -> Vec<u32> {
        if self.w.net.is_bus(id) {
            return self.w.svc.lines.get(id as usize).map_or(vec![], |l| l.stops.clone());
        }
        if self.w.net.line_ok(id) {
            self.w.net.line_stops(id).to_vec()
        } else {
            vec![]
        }
    }
    /// The line's path, `edge << 1 | direction`, first stop to last.
    pub fn line_path(&self, id: u32) -> Vec<u32> {
        if self.w.net.line_ok(id) {
            self.w.net.line_path(id).to_vec()
        } else {
            vec![]
        }
    }
    /// T-3474: where a rail line's trains run only part of its plan, that part as
    /// `[stop0, stop1, path0, path1]` (ranges of `line_stops` and `line_path`); else empty.
    pub fn line_part(&self, id: u32) -> Vec<u32> {
        self.part_of(id).map_or_else(Vec::new, |l| {
            let s = l.part.unwrap();
            vec![s.stop0, s.stop1, s.path0, s.path1]
        })
    }
    /// T-3474: the stops its trains serve: `line_stops`, or the part of them a partly built line runs.
    pub fn line_run_stops(&self, id: u32) -> Vec<u32> {
        self.part_of(id).map_or_else(|| self.line_stops(id), |l| l.stops.clone())
    }
    /// T-3474: the path its trains run forward, as `line_run_stops`.
    pub fn line_run_path(&self, id: u32) -> Vec<u32> {
        self.part_of(id).map_or_else(|| self.line_path(id), |l| l.runs[0].segs.clone())
    }
    fn part_of(&self, id: u32) -> Option<&crate::track::service::LineSvc> {
        self.w.svc.lines.get(id as usize).filter(|l| l.ok && l.part.is_some() && self.w.net.line_ok(id))
    }
    /// T-3117: the path of the line's return run, last stop to first, where it is not its path run
    /// backward (round one-way track, T-3094); else empty. T-3474: of the part it runs, for a partly
    /// built line.
    pub fn line_back_path(&self, id: u32) -> Vec<u32> {
        if let Some(l) = self.part_of(id) {
            let back = &l.runs[1].segs;
            let same = back.iter().eq(l.runs[0].segs.iter().rev().map(|p| p ^ 1).collect::<Vec<_>>().iter());
            return if l.one_way || same { vec![] } else { back.clone() };
        }
        let n = &self.w.net;
        let rail = n.line_bus.get(id as usize).map_or(true, |b| b.is_none()) && n.line_ferry.get(id as usize).map_or(true, |f| f.is_none());
        if !n.line_ok(id) || !rail || n.line_runs_once(id) {
            return vec![];
        }
        let back = n.back_path(id).unwrap_or_default();
        // T-3348: a return run to a one-track station beside may leave the path on two-way track too
        if back.iter().eq(n.line_path(id).iter().rev().map(|p| p ^ 1).collect::<Vec<_>>().iter()) { vec![] } else { back }
    }
    /// T-3348: the stations the return run stops at, first stop to last (the order of `line_stops`),
    /// where one is not the line's own stop (a platform on its own track beside it); else empty.
    pub fn line_back_stops(&self, id: u32) -> Vec<u32> {
        let Some(ls) = self.w.svc.lines.get(id as usize).filter(|l| l.ok && !l.is_bus()) else { return vec![] };
        let n = ls.runs[1].stop_idx.len();
        let back: Vec<u32> = (0..n).rev().filter_map(|k| ls.stop_node(1, k)).collect();
        if back.len() != ls.stops.len() || back == ls.stops { vec![] } else { back }
    }
    /// Where each stop is along a run's path, m (the train's middle when standing).
    pub fn line_stop_s(&self, line: u32, run: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].stop_s.clone())
    }

}
