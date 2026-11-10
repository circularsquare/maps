//! Bus lines (T-1464, `track::bus`): adding one, its route, its stops, and the drawing tool's quote.
//! A route crosses the interface as x, y, kind per point (`bus::K_CORNER`, `bus::K_ROAD`).

use super::*;
use crate::track::bus::{self, BusRoute};

/// The ground metres within which `bus_catches` thins the route a stop serves.
const CATCH_TOL_M: f64 = 15.0;

fn points_from(flat: &[f64]) -> (Vec<[f64; 2]>, Vec<u8>) {
    let pts = flat.chunks_exact(3).map(|c| [quant(c[0]), quant(c[1])]).collect();
    let kinds = flat.chunks_exact(3).map(|c| c[2] as u8).collect();
    (pts, kinds)
}

#[wasm_bindgen]
impl TrackApi {
    /// Add a planned bus line over a route (x, y, kind per point in the network frame), its buses
    /// following a headway in seconds per demand level (0: none); its stops place themselves
    /// (`bus::plan`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_bus_line(&mut self, id: u32, points: &[f64], high: u32, medium: u32, low: u32, dwell_s: f32, turnaround_s: f32, colour: u32, name: String) -> u32 {
        let (pts, kinds) = points_from(points);
        if !bus::route_ok(&self.w.net, &pts, &kinds) {
            return self.refuse(Issue::LinePath { line: id });
        }
        let data = LineData {
            name, symbol: String::new(), colour, stops: vec![], path: vec![], tph: [0.0; 3], trains: Some([0; 3]), auto: Some([high, medium, low]),
            dwell_s, turnaround_s, cars: 1, fast: false, bus: Some(BusRoute::new(pts, kinds, false)), ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, manual: false, one_way: true, limit: 0,
        };
        self.run(Op::Line { id, data: Some(data) })
    }
    /// Draw a bus line's route again, keeping everything else. A constructed route stays
    /// constructed: the edit pays for any longer road and is final, like constructed track.
    pub fn set_bus_route(&mut self, id: u32, points: &[f64]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let (pts, kinds) = points_from(points);
        let Some(r) = d.bus.as_mut() else { return self.refuse(Issue::LinePath { line: id }) };
        if !bus::route_ok(&self.w.net, &pts, &kinds) {
            return self.refuse(Issue::LinePath { line: id });
        }
        let (built, spacing, hand, kept) = (r.built, r.spacing, r.hand.take(), std::mem::take(&mut r.hand_kept));
        *r = BusRoute::new(pts, kinds, built);
        r.spacing = spacing;
        // T-2891: stops the new route still passes go onto it; the rest are gone with the old road
        let net = &self.w.net;
        let poly = bus::Poly::new(net, &r.points, &r.kinds);
        let onto = |p: &[f64; 2]| {
            let (s, d) = poly.project(net, p[0], p[1]);
            (d <= bus::STATION_REACH_M).then(|| { let (x, y) = poly.at_road(s); [quant(x), quant(y)] })
        };
        r.hand = hand.map(|h| h.iter().filter_map(onto).collect());
        r.hand_kept = kept.iter().filter_map(onto).collect();
        if built { self.run_charged(Op::Line { id, data: Some(d) }) } else { self.run(Op::Line { id, data: Some(d) }) }
    }
    /// Set how far apart a bus line's stops are, m (T-2885); 0 leaves them to place themselves. Free,
    /// on a constructed line too, and one undo step like any edit of a line.
    pub fn set_bus_spacing(&mut self, id: u32, metres: f64) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(r) = d.bus.as_mut() else { return self.refuse(Issue::LinePath { line: id }) };
        let spacing = bus::spacing_from(metres);
        if r.spacing == spacing && r.hand.is_none() {
            return 0;
        }
        r.spacing = spacing;
        if let Some(h) = r.hand.take().filter(|h| !h.is_empty()) {
            r.hand_kept = h;
        }
        self.run(Op::Line { id, data: Some(d) })
    }
    /// A bus line's stop spacing, m (0: its stops place themselves or are placed by hand, or any other line).
    pub fn bus_spacing(&self, line: u32) -> f64 {
        self.w.net.line_bus.get(line as usize).and_then(|b| b.as_ref()).map_or(0.0, |r| r.spacing as f64)
    }
    /// T-2891: the player places this bus line's stops, starting from those of its last time by hand
    /// (T-2939), else none between its ends. Free and one undo step, as `set_bus_spacing`; that function
    /// leaves the state.
    pub fn set_bus_hand(&mut self, id: u32) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(r) = d.bus.as_mut() else { return self.refuse(Issue::LinePath { line: id }) };
        if r.hand.is_some() {
            return 0;
        }
        r.spacing = 0;
        r.hand = Some(std::mem::take(&mut r.hand_kept));
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Whether the player places a bus line's stops (T-2891).
    pub fn bus_hand(&self, line: u32) -> bool {
        self.w.net.line_bus.get(line as usize).and_then(|b| b.as_ref()).is_some_and(|r| r.hand.is_some())
    }
    /// T-2891: put a stop on a by-hand bus line where the route is nearest (x, y) (`on`), or take out
    /// the stop nearest (x, y). The ends stay, and a click that changes no stop is no edit.
    pub fn set_bus_stop(&mut self, id: u32, x: f64, y: f64, on: bool) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(r) = d.bus.as_mut().filter(|r| r.hand.is_some()) else { return self.refuse(Issue::LinePath { line: id }) };
        let net = &self.w.net;
        let plan = bus::plan_route(net, id, r, r.built);
        let poly = bus::Poly::new(net, &r.points, &r.kinds);
        let mut hand = r.hand.clone().unwrap_or_default();
        if on {
            if hand.len() >= bus::HAND_MAX_STOPS {
                return 0;
            }
            let (px, py) = poly.at_road(poly.project(net, x, y).0);
            hand.push([quant(px), quant(py)]);
        } else {
            let n = plan.stops.len();
            let far = |k: usize| (plan.stops[k].x - x).hypot(plan.stops[k].y - y);
            let Some(k) = (0..n).min_by(|&a, &b| far(a).total_cmp(&far(b))) else { return 0 };
            if k == 0 || k + 1 == n {
                return 0;
            }
            // every point that made this stop goes, or a stop two points landed on would stay
            let nearest = |s: f64| (0..n).min_by(|&a, &b| (plan.stops[a].s - s).abs().total_cmp(&(plan.stops[b].s - s).abs()));
            hand.retain(|p| nearest(poly.project(net, p[0], p[1]).0) != Some(k));
        }
        r.hand = Some(hand);
        if bus::plan_route(net, id, r, r.built).stops.len() == plan.stops.len() {
            return 0;
        }
        self.run(Op::Line { id, data: Some(d) })
    }
    /// T-2891: the stations a bus line's route can stop at, a stop or not, 3 numbers each: node, x, y.
    pub fn bus_reach(&self, line: u32) -> Vec<f64> {
        let Some(plan) = self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()) else { return vec![] };
        plan.reach.iter().flat_map(|&n| [n as f64, self.w.net.node_x[n as usize], self.w.net.node_y[n as usize]]).collect()
    }
    /// A bus line's road length, m (0 for any other line).
    pub fn bus_road_m(&self, line: u32) -> f64 {
        self.w.net.bus_road_m(line)
    }
    /// Whether a line is a bus line.
    pub fn is_bus(&self, line: u32) -> bool {
        self.w.net.is_bus(line)
    }
    /// The live bus lines.
    pub fn bus_lines(&self) -> Vec<u32> {
        self.w.net.bus_lines()
    }
    /// A bus line's route: constructed (1/0), then x, y, kind per point; empty for any other line.
    pub fn bus_route(&self, line: u32) -> Vec<f64> {
        let Some(r) = self.w.net.line_bus.get(line as usize).and_then(|b| b.as_ref()).filter(|_| self.w.net.line_ok(line)) else { return vec![] };
        let mut out = vec![r.built as u8 as f64];
        out.extend(r.points.iter().zip(&r.kinds).flat_map(|(p, &k)| [p[0], p[1], k as f64]));
        out
    }
    /// A bus line's stops as its service has them, run 0's order, 8 numbers each: id (the station's
    /// node, or `bus::STOP_ID_BASE` and up), x, y, road m from the start, kind (`bus::KIND_*`),
    /// station node or -1, and the stretch of route it serves, road m from the start (NaN, NaN
    /// at a station). Empty when it has no service.
    pub fn bus_stops(&self, line: u32) -> Vec<f64> {
        let Some(plan) = self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()) else { return vec![] };
        stops_flat(plan)
    }
    /// Each of a bus line's stops' stretch of route (`bus::BusStop::cover`) as points, for demand's
    /// corridor: per stop in run 0's order its point count (0 at a station), then x, y per point.
    /// A road-following stretch is thinned to within `CATCH_TOL_M`.
    pub fn bus_catches(&self, line: u32) -> Vec<f64> {
        let (Some(plan), Some(r)) = (self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()), self.w.net.line_bus.get(line as usize).and_then(|b| b.as_ref())) else { return vec![] };
        let at = |s: f64| bus::at_road(&r.points, &plan.cum, s);
        let tol = CATCH_TOL_M * self.w.net.k_at(r.points.first().map_or(0.0, |p| p[1]));
        let mut out = vec![];
        for s in &plan.stops {
            let Some([g0, g1]) = s.cover else {
                out.push(0.0);
                continue;
            };
            let mut pts = vec![at(g0)];
            pts.extend((0..r.points.len()).filter(|&i| plan.cum[i] > g0 && plan.cum[i] < g1).map(|i| (r.points[i][0], r.points[i][1])));
            pts.push(at(g1));
            let pts = bus::simplify(&pts, tol);
            out.push(pts.len() as f64);
            out.extend(pts.into_iter().flat_map(|(x, y)| [x, y]));
        }
        out
    }
    /// A bus line's street stops (drawn only): x, y per stop.
    pub fn bus_street(&self, line: u32) -> Vec<f64> {
        let (Some(plan), Some(r)) = (self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()), self.w.net.line_bus.get(line as usize).and_then(|b| b.as_ref())) else { return vec![] };
        plan.street.iter().flat_map(|&s| { let (x, y) = bus::at_road(&r.points, &plan.cum, s); [x, y] }).collect()
    }
    /// The drawing tool's quote for a route not yet a line (`line`: the line it would replace, or
    /// any id no line has): [road m, US$M, 1 if it makes a line, stop count, then x, y, kind per
    /// stop, then street stop count and x, y per street stop].
    pub fn bus_quote(&self, line: u32, points: &[f64]) -> Vec<f64> {
        let net = &self.w.net;
        let (pts, kinds) = points_from(points);
        let ok = bus::route_ok(net, &pts, &kinds);
        let poly = bus::Poly::new(net, &pts, &kinds);
        let road = if pts.len() >= 2 { poly.road_len() } else { 0.0 };
        let rule = net.line_bus.get(line as usize).and_then(|b| b.as_ref()).map_or(bus::StopRule::Auto, |r| r.rule());
        let plan = bus::plan_by(net, line, &pts, &kinds, false, rule);
        let mut out = vec![road, road / 1000.0 * BUS_ROUTE_COST_PER_KM, (ok && plan.stops.len() >= 2) as u8 as f64, plan.stops.len() as f64];
        for s in &plan.stops {
            out.extend([s.x, s.y, s.kind as f64]);
        }
        out.push(plan.street.len() as f64);
        for &s in &plan.street {
            let (x, y) = poly.at_road(s);
            out.extend([x, y]);
        }
        out
    }
}

fn stops_flat(plan: &bus::Plan) -> Vec<f64> {
    plan.stops
        .iter()
        .flat_map(|s| {
            let [c0, c1] = s.cover.unwrap_or([f64::NAN, f64::NAN]);
            [s.id as f64, s.x, s.y, s.s, s.kind as f64, s.station.map_or(-1.0, |n| n as f64), c0, c1]
        })
        .collect()
}
