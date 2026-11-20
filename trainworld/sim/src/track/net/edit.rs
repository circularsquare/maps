//! Setters, the ground (water, terrain, roads) and what a setter reprices.

use super::*;

impl Network {
    /// High-speed track meets the node (T-2231: passing tracks there are priced as it).
    pub fn node_hsr(&self, n: u32) -> bool {
        self.node_ports[n as usize].ends().iter().any(|&c| self.edge_kind[(c >> 1) as usize].is_hsr())
    }
    /// The kind of track at a node: its first edge end's (the family rule keeps them alike).
    pub fn node_kind(&self, n: u32) -> TrackKind {
        self.node_ports[n as usize].ends().first().map_or(TrackKind::Regular, |&c| self.edge_kind[(c >> 1) as usize])
    }
    pub fn edge_section_s(&self, e: u32) -> f64 {
        self.edge_kind[e as usize].spec().section_s
    }
    pub fn edge_conflict_s(&self, e: u32) -> f64 {
        self.edge_kind[e as usize].spec().conflict_s
    }
    pub fn node_conflict_s(&self, n: u32) -> f64 {
        self.node_kind(n).spec().conflict_s
    }
    pub fn node_margin_s(&self, n: u32) -> f64 {
        self.node_kind(n).spec().platform_margin_s
    }
    /// Whether a line runs on tram track: the kind of its first stop's track.
    pub fn line_tram(&self, l: u32) -> bool {
        self.line_stops(l).first().is_some_and(|&n| self.node_ok(n) && self.node_kind(n).spec().family == 1)
    }
    /// The edge is high-speed track.
    pub fn edge_hsr(&self, e: u32) -> bool {
        self.edge_kind[e as usize].is_hsr()
    }
    /// The kind a station's upgrades are priced at: high speed if any of its track is, else its track's.
    pub fn node_price_kind(&self, n: u32) -> TrackKind {
        if self.node_hsr(n) { TrackKind::HighSpeed } else { self.node_kind(n) }
    }
    /// The speed limit an edge puts on every piece of it, m/s: its kind's (`v_kind_cap`) under the
    /// player's limit on it (T-3405).
    pub fn v_cap(&self, e: u32) -> f64 {
        let cap = self.v_kind_cap(e);
        match self.edge_limit[e as usize] {
            0 => cap,
            lim => cap.min(lim as f64 / 3.6),
        }
    }
    /// The speed limit of an edge's kind of track, m/s: `V_CROSSOVER` on a crossover, 320 km/h on
    /// high-speed track, 160 on other rail, the tram kinds' own (T-817).
    pub fn v_kind_cap(&self, e: u32) -> f64 {
        if self.edge_crossover[e as usize] { V_CROSSOVER } else { self.edge_kind[e as usize].v_cap() }
    }
    /// A line's trains' top speed, m/s: high-speed trains or ordinary ones (T-817).
    pub fn line_top(&self, l: u32) -> f64 {
        if self.line_fast[l as usize] { V_HSR } else { V_TOP }
    }
    /// A line's trains' top speed under the player's limit for the line (T-2987), m/s.
    pub fn line_run_top(&self, l: u32) -> f64 {
        let lim = self.line_limit[l as usize];
        let top = self.line_top(l);
        if lim == 0 { top } else { top.min(lim as f64 / 3.6) }
    }
    /// Edge ends at a node that are not crossovers and pass `keep`. A crossover pays for its own
    /// turnouts (T-338), so the nodes at its ends are priced as junctions only with three others.
    pub(super) fn priced_ends(&self, n: u32, keep: impl Fn(usize) -> bool) -> usize {
        self.node_ports[n as usize].ends().iter().map(|&c| (c >> 1) as usize).filter(|&e| !self.edge_crossover[e] && keep(e)).count()
    }
    /// A flat crossing between the two diagonals of one crossover is part of its price.
    pub(super) fn priced_crossing(&self, x: &Crossing) -> bool {
        !(self.edge_crossover[x.e1 as usize] && self.edge_crossover[x.e2 as usize])
    }

    // ---------------------------------------------------------------- raw setters

    pub(super) fn touch_node(&mut self, n: u32) {
        self.touch.nodes.push(n);
    }

    pub fn set_node(&mut self, id: u32, data: Option<NodeData>) -> Result<Option<NodeData>, Issue> {
        let old = self.node_data(id);
        match data {
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                if self.node_ports[i].n > 0 || self.node_platform[i] > 0 {
                    return Err(Issue::NodeInUse { node: id });
                }
                self.node_alive[i] = false;
                self.node_free.push(id);
            }
            Some(d) => {
                if !(MIN_LEVEL..=MAX_LEVEL).contains(&d.level) {
                    return Err(Issue::NodeLevel { node: id });
                }
                self.claim_node(id);
                let i = id as usize;
                self.node_alive[i] = true;
                self.node_x[i] = d.x;
                self.node_y[i] = d.y;
                self.node_level[i] = d.level;
                self.node_flying[i] = d.flying;
                let ends: Vec<u32> = self.node_ports[i].ends().to_vec();
                for c in ends {
                    self.touch.edges.push(c >> 1);
                }
            }
        }
        self.touch_node(id);
        Ok(old)
    }

    /// T-2871: an element's build day after an edit, given what it held before (0 if nothing or a
    /// blueprint): a set day stands, so undo and redo restore exactly; a built element already
    /// dated keeps its date; only construction takes today.
    pub(super) fn day_for(&self, built: bool, given: u32, before: u32) -> u32 {
        match (built, given, before) {
            (false, ..) => 0,
            (true, 0, 0) => self.today,
            (true, 0, d) => d,
            (true, g, _) => g,
        }
    }

    pub fn set_edge(&mut self, id: u32, data: Option<EdgeData>) -> Result<Option<EdgeData>, Issue> {
        let old = self.edge_data(id);
        if let Some(d) = &data {
            if !self.node_ok(d.a) {
                return Err(Issue::NoSuchNode { node: d.a });
            }
            if !self.node_ok(d.b) {
                return Err(Issue::NoSuchNode { node: d.b });
            }
            if d.a == d.b {
                return Err(Issue::EdgeLoop { edge: id });
            }
            // T-3352: high-speed track has no crossover; T-3575: tram track takes them as rail does
            if (d.tracks != 1 && d.tracks != 2) || (d.kind.is_hsr() && d.crossover) {
                return Err(Issue::Tracks { edge: id });
            }
            for (node, which) in [(d.a, 0u32), (d.b, 1u32)] {
                let p = &self.node_ports[node as usize];
                let mine = old.as_ref().map_or(false, |o| (if which == 0 { o.a } else { o.b }) == node);
                if p.n as usize >= MAX_PORTS && !mine {
                    return Err(Issue::TooManyPorts { node });
                }
            }
        }
        if let Some(o) = &old {
            self.node_ports[o.a as usize].remove(id << 1);
            self.node_ports[o.b as usize].remove(id << 1 | 1);
            self.touch_node(o.a);
            self.touch_node(o.b);
        }
        match data {
            Some(d) => {
                self.claim_edge(id);
                let i = id as usize;
                self.edge_alive[i] = true;
                self.edge_a[i] = d.a;
                self.edge_b[i] = d.b;
                self.edge_tracks[i] = d.tracks;
                self.edge_day[i] = self.day_for(d.built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.edge_built[i] = d.built;
                self.edge_off[i] = d.off && !d.built;
                self.edge_pis[i] = self.pis.replace(self.edge_pis[i], &d.pis);
                self.edge_thru[i] = d.thru;
                self.edge_crossover[i] = d.crossover;
                self.edge_kind[i] = d.kind;
                self.edge_sep[i] = d.sep;
                self.edge_oneway[i] = if d.tracks == 1 { d.oneway } else { 0 };
                self.edge_join[i] = if d.tracks == 1 { d.join } else { [0; 2] };
                self.edge_limit[i] = piece_limit(d.limit, d.kind, d.crossover);
                self.edge_ramp[i] = d.ramp;
                self.node_ports[d.a as usize].add(id << 1);
                self.node_ports[d.b as usize].add(id << 1 | 1);
                self.touch_node(d.a);
                self.touch_node(d.b);
            }
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                self.edge_alive[i] = false;
                self.edge_built[i] = false;
                self.edge_day[i] = 0;
                self.edge_off[i] = false;
                self.pis.free(self.edge_pis[i]);
                self.edge_pis[i] = Span::default();
                self.edge_thru[i] = vec![];
                self.edge_crossover[i] = false;
                self.edge_kind[i] = TrackKind::Regular;
                self.edge_sep[i] = false;
                self.edge_oneway[i] = 0;
                self.edge_join[i] = [0; 2];
                self.edge_limit[i] = 0;
                self.edge_ramp[i] = 0;
                self.edge_infer[i] = 0;
                self.edge_free.push(id);
            }
        }
        self.touch.edges.push(id);
        Ok(old)
    }

    /// T-3141: a one-way rail circle whose trains turn back at one stop, and nowhere else (a branch off the
    /// loop), started at that stop: a balloon, so they turn there as at an end of the line rather than
    /// reversing partway. The same stops in the same order round the lap, and the same path. None for
    /// any other line.
    pub fn branch_end(&self, d: &LineData) -> Option<(Vec<u32>, Vec<u32>)> {
        let (stops, path) = (&d.stops, &d.path);
        if !d.one_way || d.bus.is_some() || d.ferry.is_some() || self.loop_kind(stops, path) != LoopKind::Circle {
            return None;
        }
        let [k] = self.path_reversals(path)[..] else { return None };
        let nodes: Vec<u32> = std::iter::once(self.path_nodes(path[0]).0).chain(path.iter().map(|&p| self.path_nodes(p).1)).collect();
        let at = match_stops(&nodes, stops, &[k])?;
        let i = at.iter().position(|&a| a == k)?;
        let n = stops.len();
        if i == 0 || i == n - 1 {
            return None;
        }
        let mut s: Vec<u32> = stops[i..n - 1].iter().chain(&stops[..i]).copied().collect();
        s.push(stops[i]);
        let p: Vec<u32> = path[k..].iter().chain(&path[..k]).copied().collect();
        Some((s, p))
    }

    pub fn set_line(&mut self, id: u32, data: Option<LineData>) -> Result<Option<LineData>, Issue> {
        let old = self.line_data(id);
        match data {
            Some(mut d) => {
                if let Some((stops, path)) = self.branch_end(&d) {
                    (d.stops, d.path) = (stops, path);
                }
                if d.tph.iter().chain(&d.x.tph).any(|t| !t.is_finite() || *t < 0.0) || d.dwell_s < 0.0 || d.turnaround_s < 0.0 || (d.rows != 0 && !(ROWS_MIN..=ROWS_MAX).contains(&(d.rows as usize))) {
                    return Err(Issue::LineSchedule { line: id });
                }
                if d.fast && d.stops.first().is_some_and(|&n| self.node_ok(n) && self.node_kind(n).spec().family != 0) {
                    return Err(Issue::LineSchedule { line: id });
                }
                self.claim_line(id);
                let i = id as usize;
                let built = d.bus.as_ref().map(|r| r.built).or(d.ferry.as_ref().map(|r| r.built)).unwrap_or(true);
                self.line_day[i] = self.day_for(built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.line_alive[i] = true;
                self.line_name[i] = d.name;
                self.line_symbol[i] = d.symbol;
                self.line_colour[i] = d.colour;
                self.line_stops[i] = self.stops.replace(self.line_stops[i], &d.stops);
                self.line_path[i] = self.paths.replace(self.line_path[i], &d.path);
                self.line_trains[i] = d.trains;
                self.line_auto[i] = d.auto.filter(|_| d.trains.is_some());
                self.line_rows[i] = d.rows;
                self.line_x[i] = d.x;
                self.line_dwell[i] = d.dwell_s;
                self.line_turn[i] = d.turnaround_s;
                self.line_cars[i] = d.cars;
                self.line_fast[i] = d.fast;
                self.line_hand[i] = d.hand && d.bus.is_none() && d.ferry.is_none();
                self.line_ones[i] = if d.bus.is_some() || d.ferry.is_some() { vec![] } else { d.ones.iter().copied().filter(|x| d.stops.contains(&x.0)).collect() };
                self.line_one_way[i] = d.one_way;
                self.line_limit[i] = d.limit;
                self.line_fare_dist[i] = d.fare_dist;
                self.line_weekend[i] = d.weekend;
                self.line_twin[i] = if d.bus.is_some() || d.ferry.is_some() { Default::default() } else { d.twin };
                self.line_bus[i] = d.bus;
                self.line_ferry[i] = d.ferry;
                // after its kind and twin, which decide its rows (`rows`)
                self.line_tph[i] = self.places_in(i, places(d.tph, d.x.tph), 0.0);
                self.line_group[i] = d.group;
                self.line_group_name[i] = if d.group == 0 { String::new() } else { d.group_name };
                self.line_group_look[i] = if d.group == 0 { GroupLook::default() } else { d.group_look };
                self.line_group_outer[i] = if d.group == 0 { vec![] } else { d.outer };
            }
            None => {
                if old.is_none() {
                    return Ok(None);
                }
                let i = id as usize;
                self.line_alive[i] = false;
                self.line_day[i] = 0;
                self.line_hand[i] = false;
                self.line_ones[i].clear();
                self.line_one_way[i] = false;
                self.line_limit[i] = 0;
                self.line_fare_dist[i] = false;
                self.line_weekend[i] = Weekend::default();
                self.line_twin[i] = Default::default();
                self.line_rows[i] = 0;
                self.line_x[i] = Extra::default();
                self.line_bus[i] = None;
                self.line_ferry[i] = None;
                self.line_group[i] = 0;
                self.line_group_look[i] = GroupLook::default();
                self.line_group_outer[i].clear();
                self.stops.free(self.line_stops[i]);
                self.paths.free(self.line_path[i]);
                self.line_stops[i] = Span::default();
                self.line_path[i] = Span::default();
                self.line_free.push(id);
            }
        }
        self.touch.lines.push(id);
        self.touch.paths_changed = true;
        Ok(old)
    }

    /// A line's trains per hour as the settle starts from them (the counts' rates follow).
    pub fn set_tph(&mut self, id: u32, tph: [f32; 3]) -> Result<[f32; 3], Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        if tph.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(Issue::LineSchedule { line: id });
        }
        let old = self.rows_of(self.line_tph[id as usize]);
        // T-3542: a line of four or five rows keeps its midday and afternoon rates
        let x = self.extra_of(self.line_tph[id as usize]);
        self.line_tph[id as usize] = self.places_in(id as usize, places(tph, x), 0.0);
        self.touch.schedules.push(id);
        Ok(old)
    }

    /// T-625: switch a line between a hand-set count and a headway per level, with the counts it
    /// starts from. Returns the line's previous headways and counts.
    pub fn set_auto(&mut self, id: u32, auto: Option<[u32; 3]>, trains: Option<[u32; 3]>) -> Result<(Option<[u32; 3]>, Option<[u32; 3]>), Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        let old = (self.line_auto[id as usize], self.line_trains[id as usize]);
        self.line_trains[id as usize] = trains;
        self.line_auto[id as usize] = auto.filter(|_| trains.is_some());
        self.touch.schedules.push(id);
        Ok(old)
    }

    /// T-3521: a line's weekend trains (`LineData::weekend`); returns the old setting.
    pub fn set_weekend(&mut self, id: u32, weekend: Weekend) -> Result<Weekend, Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        let old = std::mem::replace(&mut self.line_weekend[id as usize], weekend);
        self.touch.weekend.push(id);
        Ok(old)
    }

    /// T-3542: the rows the game's lines set (`line` None: 3 to 5) or line `line`'s own (0 for the
    /// game's); returns the old setting. Every line it moves is rescheduled.
    pub fn set_rows(&mut self, line: Option<u32>, rows: u8) -> Result<u8, Issue> {
        let ok = |r: u8| (ROWS_MIN..=ROWS_MAX).contains(&(r as usize));
        let old = match line {
            None if ok(rows) => std::mem::replace(&mut self.rows_default, rows),
            Some(id) if self.line_ok(id) && (rows == 0 || ok(rows)) => std::mem::replace(&mut self.line_rows[id as usize], rows),
            Some(id) if !self.line_ok(id) => return Err(Issue::NoSuchLine { line: id }),
            _ => return Err(Issue::LineSchedule { line: line.unwrap_or(u32::MAX) }),
        };
        // a twin follows its line (`rows`)
        let moved: Vec<u32> = (0..self.line_count() as u32).filter(|&l| self.line_ok(l) && match line {
            None => true,
            Some(id) => l == id || self.line_twin_of(l) == super::twin::Twin::Of(id),
        }).collect();
        self.touch.schedules.extend(moved);
        Ok(old)
    }

    /// T-3542: a line's midday and afternoon rows (`LineData::x`); returns the old ones.
    pub fn set_extra(&mut self, id: u32, x: Extra) -> Result<Extra, Issue> {
        if !self.line_ok(id) {
            return Err(Issue::NoSuchLine { line: id });
        }
        if x.tph.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(Issue::LineSchedule { line: id });
        }
        let old = std::mem::replace(&mut self.line_x[id as usize], x);
        self.touch.schedules.push(id);
        self.touch.weekend.push(id);
        Ok(old)
    }

    pub fn set_water_mask(&mut self, mask: Box<dyn WaterMask>) {
        self.water_mask = mask;
        self.mask_epoch = self.mask_epoch.wrapping_add(1);
        for e in 0..self.edge_count() as u32 {
            if self.edge_ok(e) {
                self.touch.edges.push(e);
            }
        }
    }

    /// A water mask that adds water only inside `b` (`[xmin, ymin, xmax, ymax]`) to the one in
    /// use: only the edges reaching into `b` have their water and price worked out again, as
    /// `derive` does (T-1039: deriving the whole network for each zone's mask took 100 ms and more
    /// a mask with long track drawn). Nothing else an edge derives depends on water.
    pub fn add_water(&mut self, mask: Box<dyn WaterMask>, b: [f64; 4]) {
        self.water_mask = mask;
        self.mask_epoch = self.mask_epoch.wrapping_add(1);
        self.rework_ground(b, Ground::Water);
    }

    /// T-1270, T-2901: a terrain block added to those in use, its bounds `b` in the network frame: only
    /// the edges reaching into `b` have their terrain and price worked out again, as for water. Nothing
    /// but the price depends on terrain.
    pub fn add_terrain(&mut self, grid: std::sync::Arc<cost::ClassGrid>, b: [f64; 4]) {
        self.terrain.add(grid);
        self.rework_ground(b, Ground::Terrain);
    }

    /// A terrain block taken out of those in use, unless an edge reaches into its bounds `b`: such a
    /// block's cells price that edge at its next re-derive (T-2901).
    pub fn drop_terrain(&mut self, key: i32, b: [f64; 4]) -> bool {
        let reaches = (0..self.edge_count() as u32).any(|e| {
            self.edge_ok(e) && self.pieces.get(self.edge_pieces[e as usize]).iter().any(|p| { let q = p.bbox(); q[0] <= b[2] && q[2] >= b[0] && q[1] <= b[3] && q[3] >= b[1] })
        });
        if !reaches {
            self.terrain.remove(key);
        }
        !reaches
    }

    /// The blocks with a terrain file that the terrain samples of live edges fall in (only the
    /// blueprint's with `blueprint`), ascending.
    pub fn terrain_blocks_under(&self, blueprint: bool) -> Vec<i32> {
        let mut seen = std::collections::HashSet::new();
        let frame = self.frame;
        for e in 0..self.edge_count() as u32 {
            let i = e as usize;
            if self.edge_ok(e) && !(blueprint && self.edge_built[i]) {
                let pieces = self.pieces.get(self.edge_pieces[i]);
                cost::terrain_blocks(self.edge_len[i], |s| { let (x, y, _) = geom::pos_at(pieces, s); frame.to_lonlat(x, y) }, &mut seen);
            }
        }
        let mut out: Vec<i32> = seen.into_iter().filter(|k| self.terrain.listed(*k)).collect();
        out.sort_unstable();
        out
    }

    /// T-1929: a road tile added to those in use (`roads::Roads::key`), its bounds `b` in the
    /// network frame. T-2230: each crossing is found in one tile, so only the new tile's are looked
    /// for, on the stretches of the edges near `b` (the grid's), and added to those each edge had; an
    /// edge with none new keeps its price. Only prices change, which no line's rows hold, so
    /// `repriced` is left alone. (Every edge worked out again took 2-11 ms a tile on New York's real save.)
    pub fn add_roads(&mut self, key: u32, tile: std::sync::Arc<roads::RoadTile>, b: [f64; 4]) {
        self.roads.tiles.insert(key, tile.clone());
        let (tx, ty) = (key / roads::TILES, key % roads::TILES);
        let frame = self.frame;
        let edges = self.grid.in_box(b).unwrap_or_else(|| (0..self.edge_count() as u32).collect());
        let mut cross = vec![];
        for e in edges {
            let i = e as usize;
            // (as `derive`: a crossover's price has no crossings, and track off the ground keeps none)
            if !self.edge_ok(e) || self.edge_crossover[i] || !self.edge_kind[i].spec().road_crossings || roads::off_ground(self.verts.get(self.edge_vert[i])) {
                continue;
            }
            let pieces = self.pieces.get(self.edge_pieces[i]);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in pieces {
                let q = p.bbox();
                if q[0] <= b[2] && q[2] >= b[0] && q[1] <= b[3] && q[3] >= b[1] {
                    lo = lo.min(p.s0);
                    hi = hi.max(p.s0 + p.len);
                }
            }
            cross.clear();
            roads::crossings_in_tile(self.edge_len[i], lo, hi, |s| { let (x, y, _) = geom::pos_at(pieces, s); frame.to_lonlat(x, y) }, &tile, tx, ty, &mut cross);
            if cross.is_empty() {
                continue;
            }
            let mut all = self.road_spans.get(self.edge_roads[i]).to_vec();
            all.extend_from_slice(&cross);
            all.sort_by(|x, y| x[0].total_cmp(&y[0]));
            self.edge_roads[i] = self.road_spans.replace(self.edge_roads[i], &all);
            self.edge_cost[i] = self.edge_price(i, self.verts.get(self.edge_vert[i]), self.water.get(self.edge_water[i]), self.terrain_spans.get(self.edge_terrain[i]), self.road_spans.get(self.edge_roads[i]));
        }
    }

    /// Work out again the water or the terrain, and the price, of the edges reaching into `b`.
    pub(super) fn rework_ground(&mut self, b: [f64; 4], what: Ground) {
        let frame = self.frame;
        for e in 0..self.edge_count() as u32 {
            let i = e as usize;
            if !self.edge_ok(e) {
                continue;
            }
            let pieces = self.pieces.get(self.edge_pieces[i]);
            if !pieces.iter().any(|p| { let q = p.bbox(); q[0] <= b[2] && q[2] >= b[0] && q[1] <= b[3] && q[3] >= b[1] }) {
                continue;
            }
            let pieces = pieces.to_vec();
            match what {
                Ground::Water => {
                    let mut wet = vec![];
                    cost::water_spans(self.edge_len[i], |s| { let (x, y, _) = geom::pos_at(&pieces, s); (x, y) }, self.water_mask.as_ref(), &mut wet);
                    self.edge_water[i] = self.water.replace(self.edge_water[i], &wet);
                }
                Ground::Terrain => {
                    let mut ground = vec![];
                    cost::terrain_spans(self.edge_len[i], |s| { let (x, y, _) = geom::pos_at(&pieces, s); frame.to_lonlat(x, y) }, &self.terrain, &mut ground);
                    self.edge_terrain[i] = self.terrain_spans.replace(self.edge_terrain[i], &ground);
                }
            }
            let price = self.edge_price(i, self.verts.get(self.edge_vert[i]), self.water.get(self.edge_water[i]), self.terrain_spans.get(self.edge_terrain[i]), self.road_spans.get(self.edge_roads[i]));
            // a block over edges whose ground is flat changes no line's rows, and a view brings many
            self.repriced |= (price - self.edge_cost[i]).abs() > 1e-9;
            self.edge_cost[i] = price;
        }
    }

    /// An edge's price from its derived parts: a crossover's switches, else its track and the roads
    /// it crosses at ground level (T-1929).
    pub(super) fn edge_price(&self, i: usize, vert: &[[f64; 2]], wet: &[[f64; 2]], terrain: &[[f64; 3]], cross: &[[f64; 2]]) -> f64 {
        if self.edge_crossover[i] {
            cost::crossover_cost(self.node_level[self.edge_a[i] as usize], !wet.is_empty(), self.edge_kind[i])
        } else {
            cost::track_cost_t(self.edge_len[i], vert, wet, terrain, self.edge_tracks[i], self.edge_kind[i]) + roads::charge(vert, cross).1
        }
    }
}
