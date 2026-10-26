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
            let vmin = n.edge_pieces(e).iter().map(|p| p.v_curve()).fold(n.v_cap(e), f64::min);
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
    /// Per live node, 10 numbers: id, x, y, level, edge ends, constructed edge ends, flying,
    /// platform length (0 = no station), station constructed, heading (rad).
    pub fn nodes_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for k in 0..n.node_count() as u32 {
            if !n.node_ok(k) {
                continue;
            }
            let i = k as usize;
            out.extend([k as f64, n.node_x[i], n.node_y[i], n.node_level[i] as f64, n.node_ports[i].n as f64, n.built_ports(k) as f64, n.node_flying[i] as u8 as f64, n.node_platform[i] as f64, n.node_built[i] as u8 as f64, n.node_heading[i]]);
        }
        out
    }
    /// Per crossing of two edges at one height, 5 numbers: x, y, kind (0 flat, 1 bad), level (a
    /// flat crossing's, else 0), built (1 when both its edges are constructed). For the map's
    /// marks (T-629).
    pub fn crossings_info(&self) -> Vec<f64> {
        let n = &self.w.net;
        let mut out = vec![];
        for c in &n.crossings {
            let (kind, level) = match c.kind {
                CrossKind::Flat(l) => (0.0, l as f64),
                CrossKind::Bad => (1.0, 0.0),
                CrossKind::Overlap => continue,
            };
            let built = n.edge_built[c.e1 as usize] && n.edge_built[c.e2 as usize];
            out.extend([c.x, c.y, kind, level, built as u8 as f64]);
        }
        out
    }
    /// Where the last route, edge or node-move preview makes a crossing or a flat junction:
    /// x, y, kind (0 crossing, 1 junction) per mark (T-629).
    pub fn preview_marks(&self) -> Vec<f64> {
        self.preview_marks.clone()
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
            out.extend(n.line_tph[i].iter().map(|&t| t as f64));
            out.push(n.line_cars[i] as f64);
            match svc.filter(|s| s.ok) {
                Some(s) => {
                    out.push(s.runs[0].len);
                    out.extend(s.round_trip);
                    out.extend(s.trains.iter().map(|&t| t as f64));
                    out.extend(s.delay);
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
    /// Where each stop is along a run's path, m (the train's middle when standing).
    pub fn line_stop_s(&self, line: u32, run: usize) -> Vec<f64> {
        self.w.svc.lines.get(line as usize).filter(|l| l.ok).map_or(vec![], |l| l.runs[run].stop_s.clone())
    }

}
