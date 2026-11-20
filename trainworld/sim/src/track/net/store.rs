//! Storage: the node, edge and line arrays, ids, bus and ferry reads, edge accessors.

use super::*;

impl Network {
    /// Fit an alignment in this network's frame (ground lengths in the Mercator frame, geom.rs).
    pub fn fit(&self, v: &[Pi]) -> geom::Fit {
        geom::fit_in(v, self.frame.scaled())
    }

    /// T-3027: how edge `e`'s ramps are laid (`EdgeData::ramp`).
    pub fn edge_ramps(&self, e: u32) -> geom::Ramps {
        geom::Ramps::of(self.edge_ramp[e as usize], self.edge_kind[e as usize])
    }

    /// `fit` of `v` as edge `e`'s shape, its ramps laid as its own are.
    pub fn fit_edge(&self, e: u32, v: &[Pi]) -> geom::Fit {
        geom::with_ramps(self.edge_ramps(e), || self.fit(v))
    }

    /// T-3027: edge `e`'s speed limits from its a end, capped at `cap` (`geom::limits`).
    pub fn edge_limits(&self, e: u32, cap: f64) -> Vec<geom::Limit> {
        geom::limits(self.edge_pieces(e), self.edge_vert(e), self.edge_ramp[e as usize] == 0, cap)
    }

    /// True if a node may be placed here: anywhere in the unscaled frames; in the Mercator frame
    /// within 85 degrees of the equator and not across the antimeridian (docs/world.md).
    pub fn inside_world(&self, x: f64, y: f64) -> bool {
        !self.frame.scaled() || super::frame::merc_inside(x, y)
    }

    /// An empty network in the given frame kind.
    pub fn new_in(kind: FrameKind, origin_lon: f64, origin_lat: f64, right_hand: bool) -> Network {
        let mut n = Network::new(origin_lon, origin_lat, right_hand);
        n.frame_kind = kind;
        n.frame = Frame::new(kind, origin_lon, origin_lat);
        n
    }

    /// An empty network in the legacy equirectangular frame.
    pub fn new(origin_lon: f64, origin_lat: f64, right_hand: bool) -> Network {
        Network {
            origin_lon,
            origin_lat,
            frame_kind: FrameKind::Equirect,
            frame: Frame::new(FrameKind::Equirect, origin_lon, origin_lat),
            right_hand,
            node_alive: vec![],
            node_x: vec![],
            node_y: vec![],
            node_level: vec![],
            node_flying: vec![],
            node_platform: vec![],
            node_name: vec![],
            node_named: vec![],
            node_built: vec![],
            node_day: vec![],
            node_off: vec![],
            node_upgrades: vec![],
            node_one: vec![],
            node_fare: vec![],
            node_own: vec![],
            node_berth: vec![],
            node_ports: vec![],
            node_moves: vec![],
            node_heading: vec![],
            node_free: vec![],
            edge_alive: vec![],
            edge_a: vec![],
            edge_b: vec![],
            edge_tracks: vec![],
            edge_built: vec![],
            edge_day: vec![],
            edge_off: vec![],
            edge_pis: vec![],
            edge_thru: vec![],
            edge_crossover: vec![],
            edge_kind: vec![],
            edge_sep: vec![],
            edge_oneway: vec![],
            edge_join: vec![],
            edge_limit: vec![],
            edge_ramp: vec![],
            edge_infer: vec![],
            edge_len: vec![],
            edge_pieces: vec![],
            edge_vert: vec![],
            edge_water: vec![],
            edge_terrain: vec![],
            edge_roads: vec![],
            edge_cost: vec![],
            edge_time: vec![],
            edge_free: vec![],
            pis: Pool::default(),
            pieces: Pool::default(),
            verts: Pool::default(),
            water: Pool::default(),
            terrain_spans: Pool::default(),
            road_spans: Pool::default(),
            line_alive: vec![],
            line_day: vec![],
            line_hand: vec![],
            line_ones: vec![],
            line_one_way: vec![],
            line_limit: vec![],
            line_fare_dist: vec![],
            line_weekend: vec![],
            line_tph_rest: vec![],
            rest_solved: false,
            settle_kept_rest: vec![],
            line_twin: vec![],
            line_name: vec![],
            line_symbol: vec![],
            next_line_number: 1,
            today: DAY_UNKNOWN,
            line_colour: vec![],
            line_stops: vec![],
            line_path: vec![],
            line_tph: vec![],
            line_trains: vec![],
            line_auto: vec![],
            rates_settled: false,
            rates_from_save: false,
            pace_from_save: false,
            settle_kept: vec![],
            line_dwell: vec![],
            line_turn: vec![],
            line_cars: vec![],
            line_fast: vec![],
            line_bus: vec![],
            line_ferry: vec![],
            line_group: vec![],
            line_group_name: vec![],
            line_group_look: vec![],
            line_group_outer: vec![],
            line_rank: vec![],
            node_rank: vec![],
            line_phase: vec![],
            line_split: vec![],
            line_pace: vec![],
            line_phase_rest: vec![],
            line_split_rest: vec![],
            line_pace_rest: vec![],
            phases_placed: false,
            period_slot: PERIOD_SLOT_3,
            slot_list: slot_list(PERIOD_SLOT_3),
            rows_default: ROWS_MIN as u8,
            line_rows: vec![],
            line_x: vec![],
            hours_default: CUTS_DEFAULT,
            line_mix: vec![],
            mix_moved: vec![],
            #[cfg(test)]
            slots_pinned: false,
            line_broken: vec![],
            line_free: vec![],
            stops: Pool::default(),
            paths: Pool::default(),
            links: vec![],
            buildings: vec![],
            crossings: vec![],
            grid: Grid::default(),
            edge_lines_off: vec![0],
            edge_lines: vec![],
            water_mask: Box::new(NoWater),
            terrain: Terrain::default(),
            roads: Roads::default(),
            streets: Default::default(),
            time_zones: Default::default(),
            touch: Touch::default(),
            names_changed: vec![],
            names_all: true,
            repriced: false,
            node_wet: Default::default(),
            mask_epoch: 0,
            wet_cache: true,
        }
    }

    /// Whether the node lies on water (`water_mask.is_water` at it), asked of the mask once while the
    /// node and the mask stay as they are: stations are priced on every edit and every state.
    pub fn node_is_wet(&self, n: usize) -> bool {
        let (x, y) = (self.node_x[n], self.node_y[n]);
        if !self.wet_cache {
            return self.water_mask.is_water(x, y);
        }
        let mut kept = self.node_wet.borrow_mut();
        if kept.len() <= n {
            kept.resize(n + 1, (f64::NAN, f64::NAN, u32::MAX, false));
        }
        let k = &mut kept[n];
        if k.2 == self.mask_epoch && k.0 == x && k.1 == y {
            return k.3;
        }
        let wet = self.water_mask.is_water(x, y);
        *k = (x, y, self.mask_epoch, wet);
        wet
    }

    pub fn take_repriced(&mut self) -> bool {
        std::mem::take(&mut self.repriced)
    }

    pub fn repriced(&self) -> bool {
        self.repriced
    }

    /// The nodes whose station name was set (or cleared) since the last call, ascending; `[u32::MAX]`
    /// when every name is new to the caller (a network not asked before).
    pub fn take_name_changes(&mut self) -> Vec<u32> {
        let mut v = std::mem::take(&mut self.names_changed);
        if std::mem::replace(&mut self.names_all, false) {
            return vec![u32::MAX];
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    // ---------------------------------------------------------------- ids

    pub fn node_count(&self) -> usize {
        self.node_alive.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edge_alive.len()
    }
    pub fn line_count(&self) -> usize {
        self.line_alive.len()
    }
    pub fn node_ok(&self, n: u32) -> bool {
        self.node_alive.get(n as usize).copied().unwrap_or(false)
    }
    pub fn edge_ok(&self, e: u32) -> bool {
        self.edge_alive.get(e as usize).copied().unwrap_or(false)
    }
    pub fn line_ok(&self, l: u32) -> bool {
        self.line_alive.get(l as usize).copied().unwrap_or(false)
    }
    /// A live node with a station on it.
    pub fn is_station(&self, n: u32) -> bool {
        self.node_ok(n) && self.node_platform[n as usize] > 0
    }
    /// The player's links (T-197) whose two ends are both stations now.
    pub fn links_live(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.links.iter().copied().filter(|&[a, b]| self.is_station(a) && self.is_station(b))
    }
    /// Link two stations into one complex (T-197). False if either is not a station or they are
    /// the same; linking a linked pair again changes nothing.
    pub fn link(&mut self, a: u32, b: u32) -> bool {
        if a == b || !self.is_station(a) || !self.is_station(b) {
            return false;
        }
        let p = [a.min(b), a.max(b)];
        if !self.links.contains(&p) {
            self.links.push(p);
        }
        true
    }
    /// Remove the link between two stations, if there is one. True if there was.
    pub fn unlink(&mut self, a: u32, b: u32) -> bool {
        let p = [a.min(b), a.max(b)];
        let n = self.links.len();
        self.links.retain(|&q| q != p);
        self.links.len() != n
    }

    /// A free node id for a new node (claimed when the op adding it is applied).
    pub fn alloc_node(&mut self) -> u32 {
        self.node_free.pop().unwrap_or_else(|| {
            self.grow_nodes();
            self.node_alive.len() as u32 - 1
        })
    }
    pub fn alloc_edge(&mut self) -> u32 {
        self.edge_free.pop().unwrap_or_else(|| {
            self.grow_edges();
            self.edge_alive.len() as u32 - 1
        })
    }
    pub fn alloc_line(&mut self) -> u32 {
        self.line_free.pop().unwrap_or_else(|| {
            self.grow_lines();
            self.line_alive.len() as u32 - 1
        })
    }

    /// The ids the next `n` `alloc_line` calls would hand out, without taking them (T-3333: an edit
    /// planning new lines as ops).
    pub fn spare_line_ids(&self, n: usize) -> Vec<u32> {
        let mut out: Vec<u32> = self.line_free.iter().rev().copied().take(n).collect();
        let mut next = self.line_alive.len() as u32;
        while out.len() < n {
            out.push(next);
            next += 1;
        }
        out
    }

    /// Give back ids handed out by `alloc_*` that no op claimed (a refused or trial edit).
    pub fn release_node(&mut self, id: u32) {
        if !self.node_ok(id) && (id as usize) < self.node_count() && !self.node_free.contains(&id) {
            self.node_free.push(id);
        }
    }
    pub fn release_edge(&mut self, id: u32) {
        if !self.edge_ok(id) && (id as usize) < self.edge_count() && !self.edge_free.contains(&id) {
            self.edge_free.push(id);
        }
    }

    pub(super) fn grow_nodes(&mut self) {
        self.node_alive.push(false);
        self.node_x.push(0.0);
        self.node_y.push(0.0);
        self.node_level.push(0);
        self.node_flying.push(false);
        self.node_platform.push(0);
        self.node_name.push(String::new());
        self.node_named.push(false);
        self.node_built.push(false);
        self.node_day.push(0);
        self.node_off.push(false);
        self.node_upgrades.push(0);
        self.node_one.push(0);
        self.node_fare.push(0);
        self.node_own.push(String::new());
        self.node_berth.push(None);
        self.node_ports.push(Ports::default());
        self.node_moves.push(Moves::default());
        self.node_heading.push(0.0);
    }
    pub(super) fn grow_edges(&mut self) {
        self.edge_alive.push(false);
        self.edge_a.push(0);
        self.edge_b.push(0);
        self.edge_tracks.push(2);
        self.edge_built.push(false);
        self.edge_day.push(0);
        self.edge_off.push(false);
        self.edge_pis.push(Span::default());
        self.edge_thru.push(vec![]);
        self.edge_crossover.push(false);
        self.edge_kind.push(TrackKind::Regular);
        self.edge_sep.push(false);
        self.edge_oneway.push(0);
        self.edge_join.push([0; 2]);
        self.edge_limit.push(0);
        self.edge_ramp.push(0);
        self.edge_infer.push(0);
        self.edge_len.push(0.0);
        self.edge_pieces.push(Span::default());
        self.edge_vert.push(Span::default());
        self.edge_water.push(Span::default());
        self.edge_terrain.push(Span::default());
        self.edge_roads.push(Span::default());
        self.edge_cost.push(0.0);
        self.edge_time.push(0.0);
        self.edge_lines_off.push(*self.edge_lines_off.last().unwrap());
    }
    pub(super) fn grow_lines(&mut self) {
        self.line_alive.push(false);
        self.line_name.push(String::new());
        self.line_symbol.push(String::new());
        self.line_colour.push(0);
        self.line_stops.push(Span::default());
        self.line_path.push(Span::default());
        self.line_tph.push([0.0; SLOTS]);
        self.line_tph_rest.push([0.0; SLOTS]);
        self.line_trains.push(None);
        self.line_auto.push(None);
        self.line_dwell.push(DEFAULT_DWELL_S);
        self.line_turn.push(DEFAULT_TURNAROUND_S);
        self.line_cars.push(0);
        self.line_fast.push(false);
        self.line_bus.push(None);
        self.line_ferry.push(None);
        self.line_group.push(0);
        self.line_group_name.push(String::new());
        self.line_group_look.push(GroupLook::default());
        self.line_group_outer.push(vec![]);
        self.line_broken.push(false);
        self.line_day.push(0);
        self.line_hand.push(false);
        self.line_ones.push(vec![]);
        self.line_one_way.push(false);
        self.line_limit.push(0);
        self.line_fare_dist.push(false);
        self.line_weekend.push(Weekend::default());
        self.line_twin.push(Default::default());
        self.line_rows.push(0);
        self.line_x.push(Extra::default());
    }
    pub(super) fn claim_node(&mut self, id: u32) {
        while self.node_alive.len() <= id as usize {
            self.grow_nodes();
            let n = self.node_alive.len() as u32 - 1;
            if n != id {
                self.node_free.push(n);
            }
        }
        self.node_free.retain(|&x| x != id);
    }
    pub(super) fn claim_edge(&mut self, id: u32) {
        while self.edge_alive.len() <= id as usize {
            self.grow_edges();
            let n = self.edge_alive.len() as u32 - 1;
            if n != id {
                self.edge_free.push(n);
            }
        }
        self.edge_free.retain(|&x| x != id);
    }
    pub(super) fn claim_line(&mut self, id: u32) {
        while self.line_alive.len() <= id as usize {
            self.grow_lines();
            let n = self.line_alive.len() as u32 - 1;
            if n != id {
                self.line_free.push(n);
            }
        }
        self.line_free.retain(|&x| x != id);
    }

    // ---------------------------------------------------------------- reading inputs

    pub fn node_data(&self, n: u32) -> Option<NodeData> {
        self.node_ok(n).then(|| {
            let i = n as usize;
            NodeData { x: self.node_x[i], y: self.node_y[i], level: self.node_level[i], flying: self.node_flying[i] }
        })
    }
    pub fn edge_data(&self, e: u32) -> Option<EdgeData> {
        self.edge_ok(e).then(|| {
            let i = e as usize;
            EdgeData { a: self.edge_a[i], b: self.edge_b[i], tracks: self.edge_tracks[i], pis: self.pis.get(self.edge_pis[i]).to_vec(), built: self.edge_built[i], off: self.edge_off[i], thru: self.edge_thru[i].clone(), crossover: self.edge_crossover[i], kind: self.edge_kind[i], sep: self.edge_sep[i], day: self.edge_day[i], oneway: self.edge_oneway[i], join: self.edge_join[i], limit: self.edge_limit[i], ramp: self.edge_ramp[i] }
        })
    }
    pub fn line_data(&self, l: u32) -> Option<LineData> {
        self.line_ok(l).then(|| {
            let i = l as usize;
            LineData {
                name: self.line_name[i].clone(),
                symbol: self.line_symbol[i].clone(),
                colour: self.line_colour[i],
                stops: self.stops.get(self.line_stops[i]).to_vec(),
                path: self.paths.get(self.line_path[i]).to_vec(),
                tph: self.rows_of(self.line_tph[i]),
                trains: self.line_trains[i],
                auto: self.line_auto[i],
                dwell_s: self.line_dwell[i],
                turnaround_s: self.line_turn[i],
                cars: self.line_cars[i],
                fast: self.line_fast[i],
                bus: self.line_bus[i].clone(),
                ferry: self.line_ferry[i].clone(),
                group: self.line_group[i],
                group_name: self.line_group_name[i].clone(),
                group_look: self.line_group_look[i].clone(),
                outer: self.line_group_outer[i].clone(),
                day: self.line_day[i],
                hand: self.line_hand[i],
                ones: self.line_ones[i].clone(),
                one_way: self.line_one_way[i],
                limit: self.line_limit[i],
                twin: self.line_twin[i],
                weekend: self.line_weekend[i],
                rows: self.line_rows[i],
                x: Extra { tph: self.extra_of(self.line_tph[i]), ..self.line_x[i] },
                fare_dist: self.line_fare_dist[i],
            }
        })
    }
    /// The car's local street speed at a point of the plane, km/h (T-1577): open country's where no
    /// street grid covers it.
    pub fn car_kmh_at(&self, x: f64, y: f64) -> f64 {
        let (lon, lat) = self.frame.to_lonlat(x, y);
        self.streets.car_kmh_at(lon, lat).unwrap_or(super::streets::CAR_TOWN_KMH)
    }
    /// A live bus line (T-1464).
    pub fn is_bus(&self, l: u32) -> bool {
        self.line_ok(l) && self.line_bus[l as usize].is_some()
    }
    /// A rail line has no route of its own to build; a bus or ferry line's route is planned until constructed (T-2871).
    pub fn route_built(&self, l: u32) -> bool {
        let i = l as usize;
        match (&self.line_bus[i], &self.line_ferry[i]) {
            (Some(r), _) => r.built,
            (_, Some(r)) => r.built,
            _ => true,
        }
    }
    /// The live bus lines.
    pub fn bus_lines(&self) -> Vec<u32> {
        (0..self.line_count() as u32).filter(|&l| self.is_bus(l)).collect()
    }
    /// A port (T-2193): a station with a berth, on no track.
    pub fn is_port(&self, n: u32) -> bool {
        self.is_station(n) && self.node_berth[n as usize].is_some()
    }
    /// A live ferry line (T-2193).
    pub fn is_ferry(&self, l: u32) -> bool {
        self.line_ok(l) && self.line_ferry[l as usize].is_some()
    }
    /// The live ferry lines.
    pub fn ferry_lines(&self) -> Vec<u32> {
        (0..self.line_count() as u32).filter(|&l| self.is_ferry(l)).collect()
    }
    /// A line with no track, a bus or a ferry line: the drawn route's points.
    pub fn route_points(&self, l: u32) -> Option<&[[f64; 2]]> {
        let i = l as usize;
        if !self.line_ok(l) {
            return None;
        }
        match (self.line_bus.get(i).and_then(|b| b.as_ref()), self.line_ferry.get(i).and_then(|f| f.as_ref())) {
            (Some(r), _) => Some(&r.points),
            (None, Some(r)) => Some(&r.points),
            _ => None,
        }
    }
    /// A ferry route's length, ground m (0 for any other line).
    pub fn ferry_m(&self, l: u32) -> f64 {
        match self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            Some(r) if self.line_ok(l) => super::ferry::ground_cum(self, &r.points).last().copied().unwrap_or(0.0),
            _ => 0.0,
        }
    }
    /// What a station costs built new as it stands, US$M (T-3059: a rebuild is charged it).
    pub fn station_build_price(&self, n: u32) -> f64 {
        self.station_price(n as usize, || self.node_is_wet(n as usize))
    }
    /// A station's price, US$M: a port's flat price, any other by its level, platform, tracks and the
    /// water under it (`wet`, asked only for those).
    pub(super) fn station_price(&self, n: usize, wet: impl FnOnce() -> bool) -> f64 {
        if self.node_berth[n].is_some() {
            PORT_COST
        } else {
            cost::station_cost_k(self.node_kind(n as u32), self.node_level[n], self.node_platform[n], self.station_tracks(n as u32), wet())
        }
    }
    /// A bus route's road length, m (its drawn length, times `bus::ROAD_FACTOR` where it is straight).
    pub fn bus_road_m(&self, l: u32) -> f64 {
        match self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            Some(r) if self.line_ok(l) => super::bus::Poly::new(self, &r.points, &r.kinds).road_len(),
            _ => 0.0,
        }
    }
    /// Build cost of every bus route, US$M: all of them, or the constructed ones.
    pub(super) fn bus_cost(&self, built: bool) -> f64 {
        (0..self.line_count() as u32)
            .filter(|&l| self.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| self.line_ok(l) && (!built || r.built)))
            .map(|l| self.bus_road_m(l) / 1000.0 * BUS_ROUTE_COST_PER_KM)
            .sum()
    }
    pub fn edge_pieces(&self, e: u32) -> &[Piece] {
        self.pieces.get(self.edge_pieces[e as usize])
    }
    pub fn edge_vert(&self, e: u32) -> &[[f64; 2]] {
        self.verts.get(self.edge_vert[e as usize])
    }
    pub fn edge_water(&self, e: u32) -> &[[f64; 2]] {
        self.water.get(self.edge_water[e as usize])
    }
    pub fn edge_terrain(&self, e: u32) -> &[[f64; 3]] {
        self.terrain_spans.get(self.edge_terrain[e as usize])
    }
    pub fn edge_roads(&self, e: u32) -> &[[f64; 2]] {
        self.road_spans.get(self.edge_roads[e as usize])
    }
    pub fn line_path(&self, l: u32) -> &[u32] {
        self.paths.get(self.line_path[l as usize])
    }
    pub fn line_stops(&self, l: u32) -> &[u32] {
        self.stops.get(self.line_stops[l as usize])
    }
    /// T-3711: a line's one-way stops the player set (`LineData::ones`).
    pub fn line_ones(&self, l: u32) -> &[(u32, u8)] {
        self.line_ones.get(l as usize).map_or(&[], |v| &v[..])
    }
    pub fn lines_on_edge(&self, e: u32) -> &[u32] {
        let i = e as usize;
        if i + 1 >= self.edge_lines_off.len() {
            return &[];
        }
        &self.edge_lines[self.edge_lines_off[i] as usize..self.edge_lines_off[i + 1] as usize]
    }
    /// The edge's vertices: its a node, its PIs, its b node.
    pub fn edge_vertices(&self, e: u32) -> Vec<Pi> {
        let i = e as usize;
        let (a, b) = (self.edge_a[i] as usize, self.edge_b[i] as usize);
        let mut v = Vec::with_capacity(self.edge_pis[i].len as usize + 2);
        v.push(Pi::new(self.node_x[a], self.node_y[a], 0.0, self.node_level[a]));
        v.extend_from_slice(self.pis.get(self.edge_pis[i]));
        v.push(Pi::new(self.node_x[b], self.node_y[b], 0.0, self.node_level[b]));
        v
    }
    /// The node an edge end is at.
    pub fn end_node(&self, e: u32, which: u32) -> u32 {
        if which == 0 {
            self.edge_a[e as usize]
        } else {
            self.edge_b[e as usize]
        }
    }
    /// Entry and exit node of a path entry.
    pub fn path_nodes(&self, p: u32) -> (u32, u32) {
        let e = path_edge(p);
        if path_dir(p) == 0 {
            (self.edge_a[e as usize], self.edge_b[e as usize])
        } else {
            (self.edge_b[e as usize], self.edge_a[e as usize])
        }
    }
    /// Side (0 = A, 1 = B) of an edge end at its node.
    pub fn port_side(&self, e: u32, which: u32) -> Option<u8> {
        let n = self.end_node(e, which);
        let p = &self.node_ports[n as usize];
        p.find(e, which).map(|i| p.side[i])
    }
}
