//! Blueprint and built costs.

use super::*;

impl Network {
    /// Unbuilt costs grouped by kind, charged level, track count (`HSR_ROW` for high-speed track, `TRAM_ROW` for tram, `STREET_TRAM_ROW` for street tram),
    /// water and ramp. Seven numbers per row: those five keys, metres/count, US$M. Kind 6 is what
    /// the terrain adds to track (T-1270), its level field holding the terrain class; kind 7 the
    /// roads crossed at ground level, weighted (T-1929).
    pub fn blueprint_cost_items(&self) -> Vec<f64> {
        // the fifth key field is a track row's terrain class (T-1839; a ramp is no price factor, it is
        // charged at its dearer end, so ramps are not told apart), 0 for every other kind
        let mut rows = std::collections::BTreeMap::<(u8, i8, u8, bool, u8), (f64, f64)>::new();
        let mut add = |kind, level, tracks, wet, class: u8, qty, cost| {
            let r = rows.entry((kind, level, tracks, wet, class)).or_default();
            r.0 += qty;
            r.1 += cost;
        };
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && !self.edge_built[e] && !self.edge_off[e] && self.edge_crossover[e] {
                let a = self.edge_a[e] as usize;
                add(2, self.node_level[a], 0, !self.edge_water(e as u32).is_empty(), 0, 1.0, self.edge_cost[e]);
            } else if self.edge_alive[e] && !self.edge_built[e] && !self.edge_off[e] {
                let (tracks, kind) = (self.edge_tracks[e], self.edge_kind[e]);
                cost::track_cost_parts_t(self.edge_len[e], self.edge_vert(e as u32), self.edge_water(e as u32), self.edge_terrain(e as u32), tracks, kind, |level, wet, _ramp, class, m, c| {
                    // T-1839: the whole price, keyed by terrain class (0 where it does not count)
                    let class = if cost::terrain_surcharge(level, class) == 0.0 { 0 } else { class };
                    let row = if kind.is_hsr() { HSR_ROW } else if kind == TrackKind::Tram { TRAM_ROW } else if kind == TrackKind::StreetTram { STREET_TRAM_ROW } else { tracks };
                    add(0, level, if tracks == 1 && row > 2 { row + SINGLE_ROW_OFFSET } else { row }, wet, class, m, c);
                });
                // kind 7: roads crossed at ground level (T-1929), a count
                let (n, c) = roads::charge(self.edge_vert(e as u32), self.edge_roads(e as u32));
                if n > 0.0 {
                    add(7, 0, 0, false, 0, n, c);
                }
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] { continue; }
            // asked only where a price needs it: most nodes need none
            let is_wet = || self.node_is_wet(n);
            let level = self.node_level[n];
            // kind 12: ports (T-2193), a flat price
            if self.node_platform[n] > 0 && !self.node_built[n] && !self.node_off[n] && self.node_berth[n].is_some() {
                add(12, 0, 0, false, 0, 1.0, PORT_COST);
                if self.node_upgrades[n] != 0 {
                    add(8, level, 0, false, 0, 1.0, self.station_upgrades_price(n as u32, self.node_upgrades[n], false));
                }
            } else if self.node_platform[n] > 0 && !self.node_built[n] && !self.node_off[n] {
                let (tracks, wet) = (self.station_tracks(n as u32), is_wet());
                add(1, level, tracks, wet, 0, 1.0, cost::station_cost_k(self.node_kind(n as u32), level, self.node_platform[n], self.station_tracks(n as u32), wet));
                let ups = self.node_upgrades[n];
                if ups != 0 {
                    add(8, level, 0, wet, 0, ups.count_ones() as f64, self.station_upgrades_price(n as u32, ups, wet));
                }
            }
            let nn = n as u32;
            if self.priced_ends(nn, |_| true) >= 3
                && self.priced_ends(nn, |e| self.edge_built[e]) < 3
                && self.priced_ends(nn, |e| self.edge_built[e] || !self.edge_off[e]) >= 3
            {
                let wet = is_wet();
                let base = cost::junction_cost(level, false, wet);
                add(2, level, 0, wet, 0, 1.0, base);
                if self.node_flying[n] {
                    add(3, level, 0, wet, 0, 1.0, cost::junction_cost(level, true, wet) - base);
                }
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                let (e1, e2) = (x.e1 as usize, x.e2 as usize);
                if (!self.edge_built[e1] || !self.edge_built[e2]) && !self.edge_off[e1] && !self.edge_off[e2] && self.priced_crossing(x) {
                    let wet = self.water_mask.is_water(x.x, x.y);
                    add(4, level, 0, wet, 0, 1.0, cost::crossing_cost(level, wet));
                }
            }
        }
        // passages to stations still to be constructed (T-1281), built with them
        for ([a, b], price) in self.passages() {
            let (a, b) = (a as usize, b as usize);
            let ready = |n: usize| self.node_built[n] || !self.node_off[n];
            if !(self.node_built[a] && self.node_built[b]) && ready(a) && ready(b) {
                add(9, 0, 0, false, 0, 1.0, price);
            }
        }
        // bus routes still to be constructed (T-1464): road metres
        for l in self.bus_lines() {
            if self.line_bus[l as usize].as_ref().is_some_and(|r| !r.built) {
                let m = self.bus_road_m(l);
                add(10, 0, 0, false, 0, m, m / 1000.0 * BUS_ROUTE_COST_PER_KM);
            }
        }
        // kind 14: ferry routes still to be constructed (T-2193): ground metres, free
        for l in self.ferry_lines() {
            if self.line_ferry[l as usize].as_ref().is_some_and(|r| !r.built) {
                add(14, 0, 0, false, 0, self.ferry_m(l), 0.0);
            }
        }
        rows.into_iter().flat_map(|((kind, level, tracks, wet, ramp), (qty, cost))|
            [kind as f64, level as f64, tracks as f64, wet as u8 as f64, ramp as u8 as f64, qty, cost]).collect()
    }

    /// T-2269: the blueprint's track, stations, junctions, flat crossings and passages priced one by
    /// one, switched-off ones too, for the Build tab's breakdown by stretch. Five numbers a row: kind
    /// (0 a piece of track, 1 a station or port with its upgrades, 2 a junction with any flyover, 3 a
    /// flat crossing, 4 a passage), two ids (a crossing's two edges, a passage's two stations, else
    /// the edge or node twice), US$M, and 1 when `blueprint_cost_items` counts it (the same tests).
    pub fn blueprint_piece_costs(&self) -> Vec<f64> {
        let mut out = vec![];
        let mut row = |kind: u8, a: usize, b: usize, cost: f64, on: bool| out.extend([kind as f64, a as f64, b as f64, cost, on as u8 as f64]);
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && !self.edge_built[e] {
                row(0, e, e, self.edge_cost[e], !self.edge_off[e]);
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] { continue; }
            let is_wet = || self.node_is_wet(n);
            let level = self.node_level[n];
            if self.node_platform[n] > 0 && !self.node_built[n] {
                let price = if self.node_berth[n].is_some() { PORT_COST + self.upgrades_cost(n) } else {
                    let wet = is_wet();
                    let ups = self.node_upgrades[n];
                    cost::station_cost_k(self.node_kind(n as u32), level, self.node_platform[n], self.station_tracks(n as u32), wet) + if ups != 0 { self.station_upgrades_price(n as u32, ups, wet) } else { 0.0 }
                };
                row(1, n, n, price, !self.node_off[n]);
            }
            let nn = n as u32;
            if self.priced_ends(nn, |_| true) >= 3 && self.priced_ends(nn, |e| self.edge_built[e]) < 3 {
                let on = self.priced_ends(nn, |e| self.edge_built[e] || !self.edge_off[e]) >= 3;
                row(2, n, n, cost::junction_cost(level, self.node_flying[n], is_wet()), on);
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                let (e1, e2) = (x.e1 as usize, x.e2 as usize);
                if (!self.edge_built[e1] || !self.edge_built[e2]) && self.priced_crossing(x) {
                    let wet = self.water_mask.is_water(x.x, x.y);
                    row(3, e1, e2, cost::crossing_cost(level, wet), !self.edge_off[e1] && !self.edge_off[e2]);
                }
            }
        }
        for ([a, b], price) in self.passages() {
            let (a, b) = (a as usize, b as usize);
            if !(self.node_built[a] && self.node_built[b]) {
                let ready = |n: usize| self.node_built[n] || !self.node_off[n];
                row(4, a, b, price, ready(a) && ready(b));
            }
        }
        out
    }

    /// Build cost of everything standing, US$M.
    pub fn total_cost(&self) -> f64 {
        let mut c = 0.0;
        for e in 0..self.edge_count() {
            if self.edge_alive[e] {
                c += self.edge_cost[e];
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.water_mask.is_water(self.node_x[n], self.node_y[n]);
            if self.node_platform[n] > 0 {
                c += self.station_price(n, wet);
                c += self.upgrades_cost(n);
            }
            if self.priced_ends(n as u32, |_| true) >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.priced_crossing(x) {
                    c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                }
            }
        }
        c + self.passages_cost(false) + self.bus_cost(false) + super::building::value(self)
    }

    /// `(total_cost(), built_cost())` in one pass: the same sums in the same order, so equal to the
    /// bit, with a station priced once and the water under a node asked of the mask only when the
    /// node or the mask is new (`node_is_wet`; `total_cost` and `built_cost` ask every time, and the
    /// rows check compares them with this) (T-1540).
    pub fn total_and_built_cost(&self) -> (f64, f64) {
        let (mut t, mut b) = (0.0, 0.0);
        for e in 0..self.edge_count() {
            if self.edge_alive[e] {
                t += self.edge_cost[e];
                if self.edge_built[e] {
                    b += self.edge_cost[e];
                }
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.node_is_wet(n);
            if self.node_platform[n] > 0 {
                let station = self.station_price(n, wet);
                let upgrades = self.upgrades_cost(n);
                t += station;
                t += upgrades;
                if self.node_built[n] {
                    b += station;
                    b += upgrades;
                }
            }
            let (all, built) = (self.priced_ends(n as u32, |_| true) >= 3, self.priced_ends(n as u32, |e| self.edge_built[e]) >= 3);
            if all || built {
                let junction = cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
                if all {
                    t += junction;
                }
                if built {
                    b += junction;
                }
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.priced_crossing(x) {
                    let price = cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                    t += price;
                    if self.edge_built[x.e1 as usize] && self.edge_built[x.e2 as usize] {
                        b += price;
                    }
                }
            }
        }
        let halls = super::building::value(self);
        (t + self.passages_cost(false) + self.bus_cost(false) + halls, b + self.passages_cost(true) + self.bus_cost(true) + halls)
    }

    /// Build cost of what is constructed, US$M (SPEC 6.4): constructed edges and stations, a
    /// junction once three of its edge ends are constructed, a flat crossing once both its edges
    /// are, a passage once both its stations are, station buildings always (T-2217).
    /// `total_cost() - built_cost()` is what the
    /// blueprint would cost to construct.
    pub fn built_cost(&self) -> f64 {
        let mut c = 0.0;
        for e in 0..self.edge_count() {
            if self.edge_alive[e] && self.edge_built[e] {
                c += self.edge_cost[e];
            }
        }
        for n in 0..self.node_count() {
            if !self.node_alive[n] {
                continue;
            }
            let wet = || self.water_mask.is_water(self.node_x[n], self.node_y[n]);
            if self.node_platform[n] > 0 && self.node_built[n] {
                c += self.station_price(n, wet);
                c += self.upgrades_cost(n);
            }
            if self.priced_ends(n as u32, |e| self.edge_built[e]) >= 3 {
                c += cost::junction_cost(self.node_level[n], self.node_flying[n], wet());
            }
        }
        for x in &self.crossings {
            if let CrossKind::Flat(level) = x.kind {
                if self.edge_built[x.e1 as usize] && self.edge_built[x.e2 as usize] && self.priced_crossing(x) {
                    c += cost::crossing_cost(level, self.water_mask.is_water(x.x, x.y));
                }
            }
        }
        c + self.passages_cost(true) + self.bus_cost(true) + super::building::value(self)
    }

    /// Ground metres between two nodes in a straight line: a passage's length (T-1281).
    pub fn passage_m(&self, a: u32, b: u32) -> f64 {
        let (a, b) = (a as usize, b as usize);
        let d = (self.node_x[a] - self.node_x[b]).hypot(self.node_y[a] - self.node_y[b]);
        if self.frame().scaled() { d / super::frame::merc_k(0.5 * (self.node_y[a] + self.node_y[b])) } else { d }
    }

    /// The passages (the player's links) and their prices, US$M (T-1281): a passage is constructed
    /// once both its stations are.
    pub fn passages(&self) -> impl Iterator<Item = ([u32; 2], f64)> + '_ {
        self.links_live().map(|p| (p, super::upgrades::passage_price(self.passage_m(p[0], p[1]))))
    }

    pub(super) fn passages_cost(&self, built: bool) -> f64 {
        self.passages().filter(|(p, _)| !built || p.iter().all(|&n| self.node_built[n as usize])).map(|x| x.1).sum()
    }

    /// A station's upgrades, US$M (T-1281).
    pub fn upgrades_cost(&self, n: usize) -> f64 {
        let ups = self.node_upgrades[n];
        if ups == 0 {
            return 0.0;
        }
        self.station_upgrades_price(n as u32, ups, self.water_mask.is_water(self.node_x[n], self.node_y[n]))
    }

    /// Edge ends at a node whose edge is constructed.
    pub fn built_ports(&self, n: u32) -> usize {
        self.node_ports[n as usize].ends().iter().filter(|&&c| self.edge_built[(c >> 1) as usize]).count()
    }
}
