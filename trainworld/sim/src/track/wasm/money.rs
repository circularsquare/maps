//! Cash, the fleet, running costs, upkeep and the quotes for what an edit costs.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- money
    pub fn cash(&self) -> f64 {
        self.cash
    }
    pub fn set_cash(&mut self, v: f64) {
        self.cash = v;
    }
    pub fn set_unlimited(&mut self, on: bool) {
        self.unlimited = on;
    }
    /// What an edit may spend: the cash, or anything in a sandbox.
    pub(super) fn spendable(&self) -> f64 {
        if self.unlimited { f64::INFINITY } else { self.cash }
    }
    /// Cars owned (T-028).
    pub fn fleet(&self) -> u32 {
        self.fleet[0]
    }
    pub fn set_fleet(&mut self, cars: u32) {
        self.fleet[0] = cars;
    }
    /// High-speed cars owned (T-817).
    pub fn fleet_fast(&self) -> u32 {
        self.fleet[1]
    }
    pub fn set_fleet_fast(&mut self, cars: u32) {
        self.fleet[1] = cars;
    }
    /// Buses owned (T-1464).
    pub fn fleet_bus(&self) -> u32 {
        self.fleet[2]
    }
    pub fn set_fleet_bus(&mut self, buses: u32) {
        self.fleet[2] = buses;
    }
    /// Buses the running bus lines need now (T-1464).
    pub fn buses_needed(&self) -> u32 {
        cars_needed(&self.w)[2]
    }
    /// Ordinary cars the running lines need now.
    pub fn cars_needed(&self) -> u32 {
        cars_needed(&self.w)[0]
    }
    /// High-speed cars the running lines need now (T-817).
    pub fn cars_needed_fast(&self) -> u32 {
        cars_needed(&self.w)[1]
    }
    /// [price of a car US$M, running cost per car-km US$, economy scale, running cost per train-km
    /// US$, then a high-speed car's price, its car-km and train-km costs, upkeep a day per km of
    /// double track and per station US$, and single and high-speed track's upkeep over double
    /// track's (T-867)] (params.rs). The real running costs; the game counts them, and fares,
    /// `ECONOMY` times (T-081).
    pub fn money_params(&self) -> Vec<f64> {
        vec![
            CAR_PRICE, RUN_COST_CAR_KM, ECONOMY, RUN_COST_TRAIN_KM, HS_CAR_PRICE, HS_RUN_COST_CAR_KM, HS_RUN_COST_TRAIN_KM,
            UPKEEP_TRACK_KM, UPKEEP_STATION, crate::track::cost::track_factor(1, false), crate::track::cost::track_factor(2, true),
            // T-1464: a bus's price US$M, running cost per bus-km and a route's upkeep a day per
            // road-km US$, a route's price per road-km US$M
            BUS_PRICE, RUN_COST_BUS_KM, UPKEEP_BUS_KM, BUS_ROUTE_COST_PER_KM,
        ]
    }
    /// Running cost of the running lines per hour at a demand level, US$M: trains an hour in each
    /// direction times the line's length and cars, at the game's economy scale (T-081).
    pub fn running_cost(&self, level: usize) -> f64 {
        self.running_cost_in(level, f64::NAN)
    }
    /// `running_cost` of the lines on one local time (`running_shifts`, T-309); NaN: every line.
    pub fn running_cost_in(&self, level: usize, shift: f64) -> f64 {
        let n = &self.w.net;
        let (mut train_km, mut car_km) = ([0.0; 3], [0.0; 3]);
        for (l, s) in self.w.svc.lines.iter().enumerate() {
            if s.running && (shift.is_nan() || s.shift == shift) {
                let tk = n.line_tph[l][level.min(LEVELS - 1)] as f64 * 2.0 * s.runs[0].len / 1000.0;
                let k = fleet_kind(n, l);
                train_km[k] += tk;
                car_km[k] += tk * s.cars as f64;
            }
        }
        let ordinary = train_km[0] * RUN_COST_TRAIN_KM + car_km[0] * RUN_COST_CAR_KM;
        let fast = train_km[1] * HS_RUN_COST_TRAIN_KM + car_km[1] * HS_RUN_COST_CAR_KM;
        let bus = train_km[2] * RUN_COST_BUS_KM;
        (ordinary + fast + bus) * ECONOMY / 1e6
    }
    /// Upkeep a game day (T-867): [total US$M, track US$M, stations US$M, route-km of constructed
    /// track, constructed stations, bus routes US$M, road-km of constructed bus route (T-1464)].
    pub fn upkeep(&self) -> Vec<f64> {
        let n = &self.w.net;
        let (mut km, mut track) = (0.0, 0.0);
        for e in 0..n.edge_alive.len() {
            if let Some(u) = edge_upkeep(n, e) {
                km += n.edge_len[e] / 1000.0;
                track += u;
            }
        }
        let count = (0..n.node_alive.len()).filter(|&i| upkept_station(n, i)).count() as f64;
        let stations = count * station_upkeep();
        let bus_km: f64 = n.bus_lines().into_iter().filter_map(|l| bus_upkeep(n, l).map(|_| n.bus_road_m(l) / 1000.0)).sum();
        let buses = bus_km * UPKEEP_BUS_KM * ECONOMY / 1e6;
        vec![track + stations + buses, track, stations, km, count, buses, bus_km]
    }
    /// Each running line's share of the upkeep a game day, US$M, as `[line, share]` pairs, then
    /// `[-1, rest]` for the track and stations no running line uses (T-867). Each edge's upkeep is
    /// split between the lines on it by their trains a day, each station's between the lines
    /// stopping there. `hours`: hours a day at each demand level.
    pub fn line_upkeep(&self, hours: &[f64]) -> Vec<f64> {
        let n = &self.w.net;
        let w: Vec<f64> = self.w.svc.lines.iter().enumerate().map(|(l, s)| {
            if !s.running { return 0.0 }
            (0..LEVELS).map(|k| n.line_tph[l][k] as f64 * hours.get(k).copied().unwrap_or(0.0)).sum()
        }).collect();
        let weight = |l: u32| w.get(l as usize).copied().unwrap_or(0.0);
        let mut share = vec![0.0; w.len()];
        let mut rest = 0.0;
        for e in 0..n.edge_alive.len() {
            let Some(u) = edge_upkeep(n, e) else { continue };
            let on = n.lines_on_edge(e as u32);
            let total: f64 = on.iter().map(|&l| weight(l)).sum();
            if total > 0.0 {
                for &l in on { share[l as usize] += u * weight(l) / total; }
            } else {
                rest += u;
            }
        }
        let mut stopping: std::collections::HashMap<u32, (f64, Vec<u32>)> = Default::default();
        for l in 0..w.len() as u32 {
            if weight(l) <= 0.0 || !n.line_ok(l) { continue }
            let mut stops = n.line_stops(l).to_vec();
            stops.sort_unstable();
            stops.dedup();
            for s in stops {
                let e = stopping.entry(s).or_default();
                e.0 += weight(l);
                e.1.push(l);
            }
        }
        let u = station_upkeep();
        for i in 0..n.node_alive.len() {
            if !upkept_station(n, i) { continue }
            match stopping.get(&(i as u32)) {
                Some((total, lines)) => for &l in lines { share[l as usize] += u * weight(l) / total },
                None => rest += u,
            }
        }
        // a bus route's upkeep is its own line's (T-1464)
        for l in n.bus_lines() {
            let Some(u) = bus_upkeep(n, l) else { continue };
            if weight(l) > 0.0 { share[l as usize] += u } else { rest += u }
        }
        let mut out: Vec<f64> = share.iter().enumerate().filter(|(_, &s)| s > 0.0).flat_map(|(l, &s)| [l as f64, s]).collect();
        out.extend([-1.0, rest]);
        out
    }
    /// What the trains a planned line needs would cost once it runs, US$M, after the spare cars of
    /// its kind.
    pub fn line_train_cost(&self, line: u32) -> f64 {
        let Some(s) = self.w.svc.lines.get(line as usize).filter(|s| s.ok && !s.running) else { return 0.0 };
        let want = s.trains.iter().copied().max().unwrap_or(0) * s.cars;
        let k = fleet_kind(&self.w.net, line as usize);
        let spare = self.fleet[k].saturating_sub(cars_needed(&self.w)[k]);
        want.saturating_sub(spare) as f64 * CAR_PRICES[k]
    }
    /// Whether a line runs high-speed trains (T-817).
    pub fn line_fast(&self, line: u32) -> bool {
        self.w.net.line_fast.get(line as usize).copied().unwrap_or(false)
    }
    /// A line's trains: high-speed (true) or ordinary (T-817). Choosing high-speed buys the cars
    /// the line needs beyond the spare high-speed ones; its ordinary cars stay as spares, and the
    /// other way round.
    pub fn set_line_fast(&mut self, id: u32, fast: bool) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.fast = fast;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Per stretch between stops of run 0 (T-817): the highest speed the trains reach there, the
    /// highest the limits allow, km/h, and what sets that limit: 0 the trains' top speed, 1 the
    /// track (ordinary track's 160), 2 curves, 3 a crossover.
    pub fn line_stretch_speeds(&self, line: u32) -> Vec<f64> {
        let net = &self.w.net;
        let Some(s) = self.w.svc.lines.get(line as usize).filter(|s| s.ok) else { return vec![] };
        let run = &s.runs[0];
        let top = net.line_top(line);
        let mut out = vec![];
        for i in 0..run.stop_idx.len().saturating_sub(1) {
            let (n0, n1) = (run.stop_idx[i] as usize, run.stop_idx[i + 1] as usize);
            let (mut best, mut why) = (0.0f64, 0.0);
            for &p in &run.segs[n0..n1] {
                let e = crate::track::net::path_edge(p);
                let cap = net.v_cap(e);
                for pc in net.edge_pieces(e) {
                    let curve = pc.v_curve();
                    let v = curve.min(cap).min(top);
                    if v > best + 1e-9 {
                        best = v;
                        why = if top <= cap && top <= curve { 0.0 } else if cap <= curve { if net.edge_crossover[e as usize] { 3.0 } else { 1.0 } } else { 2.0 };
                    }
                }
            }
            let reached = profile::v_max(&run.free.phases, run.stop_s[i], run.stop_s[i + 1]);
            out.extend([reached * 3.6, best * 3.6, why]);
        }
        out
    }
    /// What the last edit paid for trains, US$M (also in `last_charge`).
    pub fn last_train_charge(&self) -> f64 {
        self.last_trains
    }
    /// Cost of everything drawn, blueprint included, US$M.
    pub fn total_cost(&self) -> f64 {
        self.w.net.total_cost()
    }
    /// Cost of what is constructed.
    pub fn built_cost(&self) -> f64 {
        self.w.net.built_cost()
    }
    /// What constructing the whole blueprint would cost now.
    pub fn blueprint_cost(&self) -> f64 {
        (self.w.net.total_cost() - self.w.net.built_cost()).max(0.0)
    }

    /// Cost rows for Construct all, including new cars after using the spare fleet.
    /// Infrastructure rows come from the same pricing intervals as track_cost.
    pub fn blueprint_cost_items(&mut self) -> Vec<f64> {
        let mut rows = self.w.net.blueprint_cost_items();
        if !rows.is_empty() && self.w.svc.lines.iter().any(|s| s.ok && !s.running) {
            let net = &self.w.net;
            let edges: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && !net.edge_built[e as usize] && !net.edge_off[e as usize]).collect();
            let stations: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && !net.node_built[n as usize] && !net.node_off[n as usize]).collect();
            let buses = net.bus_lines();
            let ops = self.construct_ops(&edges, &stations, &buses);
            let fleet = self.fleet;
            // Starting several lines together can change junction delays and their fleet needs.
            // Quote the derived constructed service, not a sum of the planned lines' estimates.
            let need = self.w.trial_with(Op::Batch(ops), cars_needed).map_or([0; 3], |(_, c)| c);
            let buy: [u32; 3] = std::array::from_fn(|k| need[k].saturating_sub(fleet[k]));
            let cost = buy[0] as f64 * CAR_PRICES[0] + buy[1] as f64 * CAR_PRICES[1];
            if cost > 0.0 {
                rows.extend([5.0, 0.0, 0.0, 0.0, 0.0, (buy[0] + buy[1]) as f64, cost]);
            }
            // T-1464: buses, kind 11
            if buy[2] > 0 {
                rows.extend([11.0, 0.0, 0.0, 0.0, 0.0, buy[2] as f64, buy[2] as f64 * CAR_PRICES[2]]);
            }
        }
        rows
    }

}
