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
            node_road: vec![],
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
            line_only: vec![],
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
            apart: vec![],
            buildings: vec![],
            keeps: vec![],
            keep_free: vec![],
            keep_epoch: 0,
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
            priced: vec![],
            node_wet: Default::default(),
            mask_epoch: 0,
            wet_cache: true,
            brt_room: Default::default(),
            brt_lanes: Default::default(),
            busways: Default::default(),
            busways_planned: Default::default(),
        }
    }

    /// T-3378 stage 4: the constructed busway edges.
    pub fn built_busways(&self) -> impl Iterator<Item = u32> + '_ {
        let planned = self.busways_planned.get();
        (0..self.edge_count() as u32).filter(move |&e| {
            let e = e as usize;
            self.edge_alive[e] && self.edge_kind[e].is_busway() && (self.edge_built[e] || planned && !self.edge_off[e])
        })
    }
    /// T-3378: whether the blueprint holds a busway Construct would build.
    pub fn busway_planned(&self) -> bool {
        (0..self.edge_count()).any(|e| self.edge_alive[e] && !self.edge_built[e] && !self.edge_off[e] && self.edge_kind[e].is_busway())
    }
    /// T-3378: `f` worked out as if every busway in the blueprint were constructed: Construct takes the BRT lanes they
    /// run along off the lanes it charges for, so the quote must too.
    pub fn with_busways_planned<T>(&self, f: impl FnOnce() -> T) -> T {
        self.busways_planned.set(true);
        let out = f();
        self.busways_planned.set(false);
        out
    }
    /// T-3378: a hash of the constructed busways: each one's id, length and ends.
    pub fn busway_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for e in self.built_busways() {
            e.hash(&mut h);
            let len = self.edge_len[e as usize];
            len.to_bits().hash(&mut h);
            let pcs = self.edge_pieces(e);
            for s in [0.0, 0.5 * len, len] {
                let (x, y, _) = super::geom::pos_at(pcs, s);
                (x.to_bits(), y.to_bits()).hash(&mut h);
            }
        }
        h.finish()
    }
    /// T-3378: the constructed busways a bus route runs along, worked out again only when `busway_key` changes.
    pub fn busways(&self) -> std::sync::Arc<super::bus::busway::Busways> {
        let key = self.busway_key();
        if let Some((k, b)) = self.busways.borrow().as_ref() {
            if *k == key {
                return b.clone();
            }
        }
        let b = std::sync::Arc::new(super::bus::busway::Busways::build(self));
        *self.busways.borrow_mut() = Some((key, b.clone()));
        b
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

    /// T-4299: the edges whose price moved outside an edit since the last call (a road tile's crossings, water,
    /// terrain), ascending: the only edge numbers that move with no edit's `Dirty` naming them.
    pub fn take_priced(&mut self) -> Vec<u32> {
        let mut v = std::mem::take(&mut self.priced);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// T-4299: counts the water masks installed (a node's water, under its station's and junction's price).
    pub fn mask_epoch(&self) -> u32 {
        self.mask_epoch
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
    /// T-4048: the separated pairs (`apart`) whose two ends are both stations now.
    pub fn apart_live(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.apart.iter().copied().filter(|&[a, b]| self.is_station(a) && self.is_station(b))
    }
    /// T-4048: whether the distance rule is kept from joining stations `a` and `b`.
    pub fn is_apart(&self, a: u32, b: u32) -> bool {
        !self.apart.is_empty() && self.apart.contains(&[a.min(b), a.max(b)])
    }
    /// T-4048: set whether `pair` is a link (`apart` false) or a separated pair (true); what it was.
    pub fn set_pair(&mut self, apart: bool, pair: [u32; 2], on: bool) -> bool {
        let p = [pair[0].min(pair[1]), pair[0].max(pair[1])];
        let list = if apart { &mut self.apart } else { &mut self.links };
        let was = list.contains(&p);
        if on && !was && p[0] != p[1] {
            list.push(p);
        } else if !on && was {
            list.retain(|&q| q != p);
        }
        was
    }
    /// T-4048: the links and separated pairs naming station `n`, taken off (a station removed takes
    /// them with it inside its edit, so its undo brings them back and a node id used again starts clean).
    pub fn take_pairs(&mut self, n: u32) -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
        let named = |p: &[u32; 2]| p[0] == n || p[1] == n;
        let links: Vec<[u32; 2]> = self.links.iter().copied().filter(named).collect();
        let apart: Vec<[u32; 2]> = self.apart.iter().copied().filter(named).collect();
        if !links.is_empty() { self.links.retain(|p| !named(p)); }
        if !apart.is_empty() { self.apart.retain(|p| !named(p)); }
        (links, apart)
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
        self.node_road.push(None);
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
        self.line_only.push(vec![]);
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
                only: self.line_only[i].clone(),
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
    /// T-3378: a BRT station, beside a road on no track.
    pub fn is_brt_station(&self, n: u32) -> bool {
        self.is_station(n) && self.node_road[n as usize].is_some()
    }
    /// A station on no track, a port or a BRT station: none of the track's station rules apply to it.
    pub fn trackless(&self, n: u32) -> bool {
        self.node_berth[n as usize].is_some() || self.node_road[n as usize].is_some()
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
    /// A station's price, US$M: a port's or a BRT station's flat price, any other by its level, platform, tracks
    /// and the water under it (`wet`, asked only for those).
    pub(super) fn station_price(&self, n: usize, wet: impl FnOnce() -> bool) -> f64 {
        if self.node_berth[n].is_some() {
            PORT_COST
        } else if self.node_road[n].is_some() {
            BRT_STATION_COST
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
    /// Build cost of every bus route, US$M: all of them, or the constructed ones. T-3378: lanes two BRT lines
    /// share are counted once (`brt_lane_m`).
    pub(super) fn bus_cost(&self, built: bool) -> f64 {
        self.bus_cost_by(built, &|l| self.bus_road_m(l))
    }

    /// `bus_cost` with each bus line's road metres from `road` (T-4299: the kept money's, `bus_road_m` as last read).
    pub(super) fn bus_cost_by(&self, built: bool, road: &dyn Fn(u32) -> f64) -> f64 {
        let road: f64 = (0..self.line_count() as u32)
            .filter(|&l| self.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| self.line_ok(l) && (!built || r.built)))
            .map(|l| road(l) / 1000.0 * BUS_ROUTE_COST_PER_KM)
            .sum();
        road + self.brt_lane_m(built) / 1000.0 * BRT_COST_PER_KM
    }
    /// T-3378: the BRT lines, those with constructed lanes first, each part by id; only those with `built`. T-3944:
    /// a constructed route with its lanes in the blueprint is among the planned ones.
    pub fn brt_order(&self, built: bool) -> Vec<u32> {
        let brt: Vec<u32> = (0..self.line_count() as u32).filter(|&l| self.line_ok(l) && self.bus_route(l).is_some_and(|r| r.brt_any())).collect();
        let (mut done, rest): (Vec<u32>, Vec<u32>) = brt.into_iter().partition(|&l| self.lanes_built(l));
        if !built {
            done.extend(rest);
        }
        done
    }
    /// T-3378: a hash of what the BRT lines' lanes are worked out from: `order`'s routes, whether each is
    /// constructed and a loop run one way, and the street grids.
    fn brt_key(&self, order: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for &l in order {
            l.hash(&mut h);
            self.line_one_way.get(l as usize).copied().unwrap_or(false).hash(&mut h);
            if let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
                (r.built, r.brt, r.brt_bp).hash(&mut h);
                r.points.len().hash(&mut h);
                r.points.iter().flatten().for_each(|v| v.to_bits().hash(&mut h));
                r.kinds.hash(&mut h);
            }
        }
        self.streets.grids.iter().for_each(|g| (std::sync::Arc::as_ptr(g) as usize).hash(&mut h));
        // T-3378: a lane on a busway is none (the busway paid), but its room counts
        if !order.is_empty() {
            self.busway_key().hash(&mut h);
        }
        h.finish()
    }
    /// T-3378: every BRT line's lanes in `brt_order(false)`, and how many of them are constructed (they come first),
    /// worked out again only when `brt_key` changes: the state rows ask it of every bus line.
    pub fn brt_lanes(&self) -> std::sync::Arc<(super::bus::lanes::Lanes, u32)> {
        let order = self.brt_order(false);
        let key = self.brt_key(&order);
        if let Some((k, lanes)) = self.brt_lanes.borrow().as_ref() {
            if *k == key {
                return lanes.clone();
            }
        }
        let built = order.iter().filter(|&&l| self.lanes_built(l)).count() as u32;
        let lanes = std::sync::Arc::new((super::bus::lanes::Lanes::build(self, &order), built));
        *self.brt_lanes.borrow_mut() = Some((key, lanes.clone()));
        lanes
    }
    /// T-3378: the road metres of BRT lanes in town, each shared stretch once: every BRT line's, or the
    /// constructed ones'. Constructed lines come first, so what the planned ones add is the blueprint's.
    pub fn brt_lane_m(&self, built: bool) -> f64 {
        let all = self.brt_lanes();
        let (lanes, n) = (&all.0, if built { all.1 as usize } else { usize::MAX });
        lanes.lines().iter().take(n).map(|&l| lanes.split(l).first).sum()
    }
    /// T-3378: the road metres of route `pts` in town that no other BRT line has lanes on: the constructed ones for
    /// a constructed line `l`, any for a planned one (or an id no line has).
    pub fn brt_new_m(&self, l: u32, pts: &[[f64; 2]], kinds: &[u8]) -> f64 {
        let all = self.brt_lanes();
        let built = self.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| r.built);
        let steps = super::bus::lanes::town_steps(self, pts, kinds).0;
        all.0.new_m_upto(&steps, all.0.position(l), if built { all.1 } else { u32::MAX })
    }
    /// T-3378: a bus line's lanes, road metres: [in town, not shared with another BRT line (what switching it on
    /// would pay), its part of the lanes' upkeep (0 off BRT)], beside the constructed BRT lines for a constructed
    /// line and all of them for a planned one.
    pub fn brt_split(&self, l: u32) -> [f64; 3] {
        let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()).filter(|_| self.line_ok(l)) else { return [0.0; 3] };
        let all = self.brt_lanes();
        let upto = if r.built { all.1 } else { u32::MAX };
        if all.0.position(l).is_some() {
            let s = all.0.split_upto(l, upto);
            return [s.town, s.alone, s.share];
        }
        let steps = super::bus::lanes::town_steps(self, &r.points, &r.kinds).0;
        [steps.iter().map(|s| s.m).sum(), all.0.new_m_upto(&steps, None, upto), 0.0]
    }
    /// T-3378 stage 2: the room on the constructed BRT lines' lanes, worked out again only when one of their routes,
    /// their one-way loops or the street grids have changed (it is read on every round of a settle).
    pub fn brt_room(&self) -> std::sync::Arc<super::bus::lanes::Room> {
        let order = self.brt_order(true);
        let key = self.brt_key(&order);
        if let Some((k, room)) = self.brt_room.borrow().as_ref() {
            if *k == key {
                return room.clone();
            }
        }
        let room = std::sync::Arc::new(if order.is_empty() { Default::default() } else { super::bus::lanes::Room::build(self, &order) });
        *self.brt_room.borrow_mut() = Some((key, room.clone()));
        room
    }
    /// T-3378: each line's part of the constructed BRT lanes' upkeep, road metres, by line index.
    pub fn brt_upkeep_m(&self) -> Vec<f64> {
        let all = self.brt_lanes();
        let mut out = vec![0.0; self.line_count()];
        for &l in all.0.lines().iter().take(all.1 as usize) {
            out[l as usize] = all.0.split_upto(l, all.1).share;
        }
        out
    }
    /// T-3907: a bus line's road metres in town for BRT (`bus::Pace::town_m`), on BRT or not (0 for any other line).
    pub fn bus_town_m(&self, l: u32) -> f64 {
        match self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            Some(r) if self.line_ok(l) => super::bus::town_m(self, &r.points, &r.kinds),
            _ => 0.0,
        }
    }
    /// T-3907: the road metres a BRT line pays its lanes over: `bus_town_m` on BRT, else 0.
    pub fn brt_town_m(&self, l: u32) -> f64 {
        if self.is_brt(l) { self.bus_town_m(l) } else { 0.0 }
    }
    /// Whether a bus line runs on BRT (T-3907).
    pub fn is_brt(&self, l: u32) -> bool {
        self.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| r.brt)
    }
    /// T-3944: whether a constructed bus line's BRT lanes are in the blueprint.
    pub fn brt_bp(&self, l: u32) -> bool {
        self.bus_route(l).is_some_and(|r| r.brt_bp)
    }
    /// T-3944: whether a bus line's BRT lanes are constructed.
    pub fn lanes_built(&self, l: u32) -> bool {
        self.bus_route(l).is_some_and(|r| r.lanes_built())
    }
    fn bus_route(&self, l: u32) -> Option<&super::bus::BusRoute> {
        self.line_bus.get(l as usize).and_then(|b| b.as_ref())
    }
    /// T-3944: a BRT line's lanes on the map: runs of its road in town, each `[kind, points, x, y, ...]` (frame), kind 0
    /// new lanes in the blueprint, 1 in the blueprint on lanes a constructed BRT line has (free), 2 constructed. The
    /// blueprint's kinds are against the constructed lanes, as its Construct price is. Empty off BRT.
    pub fn brt_marks(&self, l: u32) -> Vec<f64> {
        let Some(r) = self.bus_route(l).filter(|r| self.line_ok(l) && r.brt_any()) else { return vec![] };
        let all = self.brt_lanes();
        let built = r.lanes_built();
        super::bus::lanes::marks(self, &r.points, &r.kinds, |s| if built { 2 } else if all.0.sharers(s, all.0.position(l)).iter().any(|&j| j < all.1) { 1 } else { 0 })
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
    /// T-4013: a line's boarding-only and getting-off-only stops (`LineData::only`).
    pub fn line_only(&self, l: u32) -> &[(u32, u8)] {
        self.line_only.get(l as usize).map_or(&[], |v| &v[..])
    }
    /// T-4013: what riders may do at station `s` on line `l`: `BOARD_ONLY`, `ALIGHT_ONLY`, or 0 both.
    pub fn stop_only(&self, l: u32, s: u32) -> u8 {
        self.line_only(l).iter().find(|x| x.0 == s).map_or(0, |x| x.1)
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
