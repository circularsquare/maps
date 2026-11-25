//! Stations and one-track queries, with set_station, check_station and platform_reach.

use super::*;

impl Network {
    pub fn station_data(&self, n: u32) -> Option<StationData> {
        (self.node_ok(n) && self.node_platform[n as usize] > 0)
            .then(|| StationData { platform: self.node_platform[n as usize], name: self.node_name[n as usize].clone(), named: self.node_named[n as usize], built: self.node_built[n as usize], off: self.node_off[n as usize], upgrades: self.node_upgrades[n as usize], berth: self.node_berth[n as usize], day: self.node_day[n as usize], one: self.node_one[n as usize], fare: self.node_fare[n as usize], own: self.node_own[n as usize].clone(), road: self.node_road[n as usize] })
    }
    /// Tracks at a station (2 if any of its edges is double).
    pub fn node_tracks(&self, n: u32) -> u8 {
        let p = &self.node_ports[n as usize];
        p.ends().iter().map(|&c| self.edge_tracks[(c >> 1) as usize]).max().unwrap_or(2)
    }
    /// Tracks a station's platforms are on, for its price: 1 for a one-track station (T-3332).
    pub fn station_tracks(&self, n: u32) -> u8 {
        if self.node_one[n as usize] != 0 { 1 } else { self.node_tracks(n) }
    }
    /// T-3332: a station with its platform on one track only (`StationData::one`).
    pub fn one_track(&self, n: u32) -> bool {
        self.node_ok(n) && self.node_platform[n as usize] > 0 && self.node_one[n as usize] != 0
    }
    /// A station whose platform is on one track: a one way station, or any stop on single track (a
    /// passing loop makes its tracks double). It pays `UPKEEP_ONE_PLATFORM_SHARE` of the upkeep.
    pub fn one_platform(&self, n: u32) -> bool {
        self.node_ok(n) && self.node_platform[n as usize] > 0 && (self.node_one[n as usize] != 0 || (self.node_tracks(n) == 1 && self.node_upgrades[n as usize] & crate::track::upgrades::PASSING == 0))
    }
    /// T-3332: a one-track station where trains cannot pass: on single track. A tram stop is the
    /// same (T-3563): one placed as a passing station is a loop its trams meet at.
    pub fn no_passing(&self, n: u32) -> bool {
        self.node_ok(n) && self.node_tracks(n) == 1 && self.one_track(n)
    }
    /// Whether the node's heading points along the axis a one-track station's way is stored against
    /// (`one_for_travel`).
    pub fn axis_along_heading(&self, n: u32) -> bool {
        one_for_travel(self.node_heading[n as usize]) == 1
    }
    /// T-3332: a one-track station on double track: the side (`port_side`) the trains it serves
    /// arrive from (a train arriving from side 0 runs along the heading). None for any other node.
    pub fn one_side(&self, n: u32) -> Option<u8> {
        if !self.one_track(n) || self.node_tracks(n) < 2 || self.node_ports[n as usize].n != 2 {
            return None;
        }
        Some(if (self.node_one[n as usize] == 1) == self.axis_along_heading(n) { 0 } else { 1 })
    }
    /// `StationData::one` for a one-track station at `n` serving trains that arrive from `side`.
    pub fn one_for_side(&self, n: u32, side: u8) -> u8 {
        if (side == 0) == self.axis_along_heading(n) { 1 } else { 2 }
    }
    /// T-3332: per stop of a run over `segs` (`stop_idx` into its nodes), whether its trains pass it
    /// on the other track from its platform. At a run's first stop it is the track a train arriving
    /// to turn there comes in on (the one it leaves by, the other way), or for a circle the one it
    /// comes round on.
    pub fn run_passes(&self, segs: &[u32], stop_idx: &[u32], circle: bool) -> Vec<bool> {
        let node_at = |k: usize| if k == 0 { self.path_nodes(segs[0]).0 } else { self.path_nodes(segs[k - 1]).1 };
        let arrive = |p: u32| self.port_side(path_edge(p), 1 - path_dir(p));
        stop_idx.iter().map(|&k| {
            let k = k as usize;
            let Some(want) = self.one_side(node_at(k)) else { return false };
            let side = if k > 0 { arrive(segs[k - 1]) } else if circle { segs.last().and_then(|&p| arrive(p)) } else { self.port_side(path_edge(segs[0]), path_dir(segs[0])) };
            side != Some(want)
        }).collect()
    }
    /// A station's upgrades, US$M (`upgrades::cost` at its level, water, platform and track).
    pub fn station_upgrades_price(&self, n: u32, bits: u8, wet: bool) -> f64 {
        let i = n as usize;
        super::upgrades::cost(bits, self.node_level[i], wet, self.node_platform[i], self.node_price_kind(n))
    }
    pub fn set_station(&mut self, node: u32, data: Option<StationData>) -> Result<Option<StationData>, Issue> {
        if !self.node_ok(node) {
            return Err(Issue::NoSuchNode { node });
        }
        let old = self.station_data(node);
        let i = node as usize;
        // T-4249: what a bus line's stops read of a station (`bus::reach`): there, constructed, a BRT station
        let seen = |s: Option<&StationData>| s.filter(|d| d.platform > 0).map(|d| (d.built, d.road));
        let bus_sees = seen(old.as_ref()) != seen(data.as_ref());
        let built = [&old, &data].iter().any(|s| seen(s.as_ref()).is_some_and(|x| x.0));
        match data {
            None => {
                for l in 0..self.line_count() as u32 {
                    if self.line_ok(l) && self.line_stops(l).contains(&node) {
                        return Err(Issue::StationInUse { node, line: l });
                    }
                }
                // T-3348: nor where a run stops in place of a stop of its line (the platform on its own
                // track beside it): without it the line has no way back
                if self.way_limited(node) {
                    for l in 0..self.line_count() as u32 {
                        if self.line_ok(l) && self.way_stops(self.line_stops(l), self.line_path(l), self.line_ones(l)).is_some_and(|(f, b)| f.contains(&node) || b.contains(&node)) {
                            return Err(Issue::StationInUse { node, line: l });
                        }
                    }
                }
                self.node_platform[i] = 0;
                self.node_day[i] = 0;
                self.node_name[i].clear();
                self.node_named[i] = false;
                self.node_built[i] = false;
                self.node_off[i] = false;
                self.node_upgrades[i] = 0;
                self.node_one[i] = 0;
                self.node_fare[i] = 0;
                self.node_own[i].clear();
                self.node_berth[i] = None;
                self.node_road[i] = None;
            }
            Some(d) => {
                self.node_day[i] = self.day_for(d.built, d.day, old.as_ref().map_or(0, |o| o.day));
                self.node_platform[i] = d.platform;
                self.node_name[i] = d.name;
                self.node_named[i] = d.named;
                self.node_built[i] = d.built;
                self.node_off[i] = d.off && !d.built;
                // T-4012: a BRT station's waiting room is built in (`BRT_STATION`), so one a save bought is dropped, not counted twice
                self.node_upgrades[i] = d.upgrades & super::upgrades::ALL & if d.road.is_some() { !super::upgrades::WAITING_ROOM } else { 0xff };
                self.node_one[i] = d.one.min(2);
                self.node_fare[i] = d.fare;
                self.node_own[i] = d.own;
                self.node_berth[i] = d.berth;
                self.node_road[i] = d.road;
            }
        }
        // T-4080: a kept stop's connection to this station starts or stops holding
        if self.keeps.iter().flatten().any(|k| k.links.contains(&bus::keep::KeepLink::Station(node))) {
            self.keep_epoch = self.keep_epoch.wrapping_add(1);
        }
        self.touch.stations.push(node);
        if bus_sees {
            self.touch.station_places.push(([self.node_x[i], self.node_y[i]], built));
        }
        if self.names_changed.len() > 1 << 20 {
            self.names_changed.clear();
            self.names_all = true;
        }
        self.names_changed.push(node);
        self.touch_node(node);
        Ok(old)
    }

    /// How far a station's platform reaches along each of its edges.
    pub fn platform_reach(&self, n: u32) -> f64 {
        let i = n as usize;
        let plat = self.node_platform[i] as f64;
        if self.node_ports[i].n <= 1 {
            plat
        } else {
            plat / 2.0
        }
    }

    pub(super) fn check_station(&self, n: u32, out: &mut Vec<Issue>) {
        let i = n as usize;
        if let Some(b) = self.node_berth[i] {
            // a port's landing is on the shore, which the coarse masks may call water
            // T-3419: a waiting room is the one upgrade a port takes
            if self.node_ports[i].n > 0 || self.node_upgrades[i] & !super::upgrades::WAITING_ROOM != 0 || self.node_level[i] != 0 {
                out.push(Issue::NotAtPort { node: n });
            }
            if !super::ferry::berth_ok(self, [self.node_x[i], self.node_y[i]], b) {
                out.push(Issue::PortBerth { node: n });
            }
            return;
        }
        if let Some(h) = self.node_road[i] {
            // T-3378: beside a road (the app's test; a road on a bridge stands over water); park and ride from an older
            // save still loads
            if self.node_ports[i].n > 0 || self.node_upgrades[i] & !super::upgrades::PARK_RIDE != 0 || self.node_level[i] != 0 || !h.is_finite() {
                out.push(Issue::NotAtBrtStation { node: n });
            }
            return;
        }
        let plat = self.node_platform[i];
        let (pmin, pmax) = self.node_kind(n).spec().platform;
        if !(pmin..=pmax).contains(&plat) || plat % PLATFORM_STEP != 0 {
            out.push(Issue::PlatformLength { node: n });
        }
        let p = self.node_ports[i];
        // A station whose track was removed waits for new track (its lines are broken).
        if p.n > 2 {
            out.push(Issue::StationJunction { node: n });
            return;
        }
        let reach = self.platform_reach(n);
        for &c in p.ends() {
            let (e, which) = (c >> 1, c & 1);
            let len = self.edge_len[e as usize];
            let other = self.end_node(e, 1 - which);
            let other_reach = if self.node_platform[other as usize] > 0 { self.platform_reach(other) } else { 0.0 };
            if reach + other_reach > len - 1.0 {
                let ports = self.node_ports[other as usize].n;
                out.push(if other_reach > 0.0 {
                    Issue::PlatformStation { node: n }
                } else if ports == 2 {
                    Issue::PlatformJoint { node: n }
                } else if ports > 2 {
                    Issue::PlatformJunction { node: n }
                } else {
                    Issue::PlatformTooLong { node: n }
                });
            }
            let (a, b) = if which == 0 { (0.0, reach) } else { (len - reach, len) };
            if geom::on_ramp(self.edge_vert(e), a, b) {
                out.push(Issue::PlatformNotLevel { node: n });
            }
            for x in &self.crossings {
                let s = if x.e1 == e { x.s1 } else if x.e2 == e { x.s2 } else { continue };
                if s >= a - 1.0 && s <= b + 1.0 {
                    out.push(Issue::PlatformCrossing { node: n });
                }
            }
        }
        if self.node_level[i] == 0 && self.water_mask.is_water(self.node_x[i], self.node_y[i]) {
            out.push(Issue::StationGroundOverWater { node: n });
        }
    }
}
