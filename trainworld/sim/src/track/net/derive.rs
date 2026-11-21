//! Derived data and crossings.

use super::*;

impl Network {
    pub fn take_touch(&mut self) -> Touch {
        let mut t = std::mem::take(&mut self.touch);
        for v in [&mut t.nodes, &mut t.edges, &mut t.stations, &mut t.lines, &mut t.schedules, &mut t.weekend] {
            v.sort_unstable();
            v.dedup();
        }
        t
    }

    /// Mark everything touched (after a bulk load).
    pub fn touch_all(&mut self) {
        self.touch.nodes = (0..self.node_count() as u32).filter(|&n| self.node_ok(n)).collect();
        self.touch.edges = (0..self.edge_count() as u32).filter(|&e| self.edge_ok(e)).collect();
        self.touch.lines = (0..self.line_count() as u32).filter(|&l| self.line_ok(l)).collect();
        self.touch.stations = self.touch.nodes.iter().copied().filter(|&n| self.node_platform[n as usize] > 0).collect();
        self.touch.paths_changed = true;
    }

    // ---------------------------------------------------------------- derive

    /// Rebuild what the touched inputs feed: edge geometry and grid, node ports, crossings, the
    /// edge-to-lines index and broken flags. `t` is the touch record (from `take_touch`).
    pub fn derive(&mut self, t: &Touch) -> Derived {
        let mut d = Derived::default();
        let mut edges = t.edges.clone();
        for &n in &t.nodes {
            if (n as usize) < self.node_count() {
                for &c in self.node_ports[n as usize].ends() {
                    edges.push(c >> 1);
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let mut fresh = vec![];
        for &e in &edges {
            let i = e as usize;
            let old = self.edge_pieces[i];
            if old.len > 0 {
                let p = self.pieces.get(old).to_vec();
                self.grid.remove(e, &p);
            }
            if !self.edge_alive[i] {
                self.pieces.free(old);
                self.verts.free(self.edge_vert[i]);
                self.water.free(self.edge_water[i]);
                self.terrain_spans.free(self.edge_terrain[i]);
                self.road_spans.free(self.edge_roads[i]);
                self.edge_pieces[i] = Span::default();
                self.edge_vert[i] = Span::default();
                self.edge_water[i] = Span::default();
                self.edge_terrain[i] = Span::default();
                self.edge_roads[i] = Span::default();
                self.edge_len[i] = 0.0;
                self.edge_cost[i] = 0.0;
                self.edge_time[i] = 0.0;
                continue;
            }
            let f = self.fit_edge(e, &self.edge_vertices(e));
            let mut wet = vec![];
            cost::water_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); (x, y) }, self.water_mask.as_ref(), &mut wet);
            let mut ground = vec![];
            let frame = self.frame;
            cost::terrain_spans(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.terrain, &mut ground);
            let mut cross = vec![];
            if !self.edge_crossover[i] && self.edge_kind[i].spec().road_crossings && !roads::off_ground(&f.vert) {
                roads::crossings(f.len, |s| { let (x, y, _) = geom::pos_at(&f.pieces, s); frame.to_lonlat(x, y) }, &self.roads, &mut cross);
            }
            self.edge_len[i] = f.len;
            self.edge_cost[i] = self.edge_price(i, &f.vert, &wet, &ground, &cross);
            self.edge_terrain[i] = self.terrain_spans.replace(self.edge_terrain[i], &ground);
            self.edge_roads[i] = self.road_spans.replace(self.edge_roads[i], &cross);
            let cap = self.v_cap(e);
            self.edge_time[i] = geom::run_time(&f.pieces, &f.vert, self.edge_ramp[i] == 0, cap).max(1e-3);
            self.edge_pieces[i] = self.pieces.replace(old, &f.pieces);
            self.edge_vert[i] = self.verts.replace(self.edge_vert[i], &f.vert);
            self.edge_water[i] = self.water.replace(self.edge_water[i], &wet);
            self.grid.insert(e, &f.pieces);
            fresh.push(e);
        }
        // Nodes: the touched ones and every node at a re-derived edge's ends.
        let mut nodes = t.nodes.clone();
        for &e in &fresh {
            nodes.push(self.edge_a[e as usize]);
            nodes.push(self.edge_b[e as usize]);
        }
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            if self.node_ok(n) {
                self.derive_node(n);
            }
        }
        // T-3137: single track's ways, then the moves they allow
        let mut seeds = nodes.clone();
        seeds.extend_from_slice(&t.stations);
        d.ways = self.derive_ways(&edges, &seeds);
        for &e in &d.ways {
            nodes.push(self.edge_a[e as usize]);
            nodes.push(self.edge_b[e as usize]);
        }
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            if self.node_ok(n) {
                self.derive_moves(n);
            }
        }
        // Crossings of touched edges.
        let before = self.crossings.len();
        let is_touched = |e: u32| edges.binary_search(&e).is_ok();
        let mut kept = Vec::with_capacity(before);
        for c in std::mem::take(&mut self.crossings) {
            if is_touched(c.e1) || is_touched(c.e2) {
                d.crossings.push([c.x, c.y, self.crossing_trams(c.e1, c.e2) as f64]);
            } else {
                kept.push(c);
            }
        }
        self.crossings = kept;
        for &e in &fresh {
            let near = self.grid.near(self.edge_pieces(e));
            for f in near {
                // Pairs of two touched edges are tested once, from the smaller id.
                if f < e && fresh.binary_search(&f).is_ok() {
                    continue;
                }
                self.find_crossings(e, f, &mut d.crossings);
            }
        }
        self.crossings.sort_by(|a, b| (a.e1, a.e2).cmp(&(b.e1, b.e2)).then(a.s1.total_cmp(&b.s1)));
        if t.paths_changed || edges.iter().any(|&e| !self.edge_alive[e as usize]) {
            self.rebuild_edge_lines();
        }
        edges.extend_from_slice(&d.ways);
        edges.sort_unstable();
        edges.dedup();
        // Broken flags of lines through touched edges and of touched lines.
        let mut lines: Vec<u32> = t.lines.clone();
        for &e in &edges {
            lines.extend_from_slice(self.lines_on_edge(e));
        }
        for &n in &t.stations {
            for l in 0..self.line_count() as u32 {
                if self.line_ok(l) && self.line_stops(l).contains(&n) {
                    lines.push(l);
                }
            }
        }
        lines.extend(self.back_lines());
        lines.sort_unstable();
        lines.dedup();
        for l in lines {
            if self.line_ok(l) {
                self.line_broken[l as usize] = self.check_line(l).is_err();
            }
        }
        self.maybe_compact();
        d.edges = edges;
        d.nodes = nodes;
        d
    }

    pub(super) fn derive_node(&mut self, n: u32) {
        let i = n as usize;
        let ports = self.node_ports[i];
        if ports.n == 0 {
            // T-3378: a BRT station's platform lies along its road
            self.node_heading[i] = self.node_road[i].unwrap_or(0.0);
            self.node_moves[i] = Moves::default();
            return;
        }
        let (nx, ny) = (self.node_x[i], self.node_y[i]);
        // Outward heading of each end; the lowest end defines the node's heading (its side B).
        let mut out_th = [0.0f64; MAX_PORTS];
        let mut lat = [0.0f64; MAX_PORTS];
        for (k, &c) in ports.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            if pcs.is_empty() {
                continue;
            }
            out_th[k] = if which == 0 { geom::pos_at(pcs, 0.0).2 } else { geom::pos_at(pcs, self.edge_len[e as usize]).2 + PI };
        }
        let h = out_th[0];
        self.node_heading[i] = h;
        let (hc, hs) = (h.cos(), h.sin());
        let mut p = ports;
        for (k, &c) in ports.ends().iter().enumerate() {
            let (e, which) = (c >> 1, c & 1);
            let pcs = self.edge_pieces(e);
            let len = self.edge_len[e as usize];
            p.side[k] = if (out_th[k] - h).cos() > 0.0 { 1 } else { 0 };
            if pcs.is_empty() {
                continue;
            }
            let d = len.min(200.0);
            let (px, py, _) = if which == 0 { geom::pos_at(pcs, d) } else { geom::pos_at(pcs, len - d) };
            lat[k] = hc * (py - ny) - hs * (px - nx);
        }
        for side in 0..2u8 {
            let mut idx: Vec<usize> = (0..ports.n as usize).filter(|&k| p.side[k] == side).collect();
            idx.sort_by(|&a, &b| lat[a].total_cmp(&lat[b]).then(ports.end[a].cmp(&ports.end[b])));
            for (r, &k) in idx.iter().enumerate() {
                p.rank[k] = r as u8;
            }
        }
        self.node_ports[i] = p;
    }

    pub(super) fn find_crossings(&mut self, e: u32, f: u32, changed: &mut Vec<[f64; 3]>) {
        let (ie, i_f) = (e as usize, f as usize);
        if self.edge_sep[ie] && self.edge_sep[i_f] {
            return;
        }
        let mut hits = vec![];
        cross::intersect_alignments(self.edge_pieces(e), self.edge_pieces(f), e == f, &mut hits);
        if hits.is_empty() {
            return;
        }
        let shared: Vec<u32> = [self.edge_a[ie], self.edge_b[ie]]
            .into_iter()
            .filter(|&n| e != f && (n == self.edge_a[i_f] || n == self.edge_b[i_f]))
            .collect();
        for (s1, s2, overlap) in hits {
            let (x, y, _) = geom::pos_at(self.edge_pieces(e), s1);
            if shared.iter().any(|&n| (self.node_x[n as usize] - x).hypot(self.node_y[n as usize] - y) < SHARED_NODE_CLEAR * self.k_at(y)) {
                continue;
            }
            let (v1, v2) = (self.edge_vert(e), self.edge_vert(f));
            let (z1, z2) = (geom::height_at(v1, s1), geom::height_at(v2, s2));
            let dz = (z1 - z2).abs();
            if dz >= LEVEL_H - 1e-6 {
                continue;
            }
            let kind = if overlap {
                CrossKind::Overlap
            } else if dz < 1e-6 && !geom::on_ramp(v1, s1 - 1.0, s1 + 1.0) && !geom::on_ramp(v2, s2 - 1.0, s2 + 1.0) {
                CrossKind::Flat(cost::level_of(z1))
            } else {
                CrossKind::Bad
            };
            let (e1, s1, e2, s2) = if e <= f { (e, s1, f, s2) } else { (f, s2, e, s1) };
            self.crossings.push(Crossing { e1, s1, e2, s2, x, y, kind });
            changed.push([x, y, self.crossing_trams(e1, e2) as f64]);
        }
    }

    /// T-3062: the tram tracks (0 to 2) of a crossing of edges `e1` and `e2`. At a flat crossing a tram
    /// gives way to a train, and the map draws its stroke of the mark broken.
    pub fn crossing_trams(&self, e1: u32, e2: u32) -> u8 {
        [e1, e2].into_iter().filter(|&e| self.edge_kind[e as usize].spec().family == 1).count() as u8
    }

    pub(super) fn rebuild_edge_lines(&mut self) {
        let mut pairs: Vec<(u32, u32)> = vec![];
        for l in 0..self.line_count() as u32 {
            if !self.line_ok(l) {
                continue;
            }
            for &p in self.line_path(l) {
                pairs.push((path_edge(p), l));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let ne = self.edge_count();
        let mut off = vec![0u32; ne + 1];
        for &(e, _) in &pairs {
            if (e as usize) < ne {
                off[e as usize + 1] += 1;
            }
        }
        for i in 0..ne {
            off[i + 1] += off[i];
        }
        self.edge_lines = pairs.iter().filter(|p| (p.0 as usize) < ne).map(|p| p.1).collect();
        self.edge_lines_off = off;
    }

    pub(super) fn maybe_compact(&mut self) {
        if self.pis.needs_compact() {
            self.pis.compact(&mut self.edge_pis);
        }
        if self.pieces.needs_compact() {
            self.pieces.compact(&mut self.edge_pieces);
        }
        if self.verts.needs_compact() {
            self.verts.compact(&mut self.edge_vert);
        }
        if self.water.needs_compact() {
            self.water.compact(&mut self.edge_water);
        }
        if self.terrain_spans.needs_compact() {
            self.terrain_spans.compact(&mut self.edge_terrain);
        }
        if self.road_spans.needs_compact() {
            self.road_spans.compact(&mut self.edge_roads);
        }
        if self.stops.needs_compact() {
            self.stops.compact(&mut self.line_stops);
        }
        if self.paths.needs_compact() {
            self.paths.compact(&mut self.line_path);
        }
    }

    // ---------------------------------------------------------------- validity
}
