//! Ferry lines and ports (T-2193, `track::ferry`): adding a port, a ferry line and its route, and the
//! drawing tool's quote. A route crosses the interface as x, y, kind (`ferry::K_*`: 0 route, 1 port, 2
//! waypoint) and speed (m/s) per point, as T-2192's worker gives them (notes/T-2177.md "F1 -> F2 route
//! format"), with the line's ports as their own list in route order, one per `K_PORT` point.

use super::*;
use crate::track::ferry::{self, FerryRoute};

fn route_from(flat: &[f64]) -> FerryRoute {
    let c = flat.chunks_exact(4);
    let pts = c.clone().map(|c| [quant(c[0]), quant(c[1])]).collect();
    let kinds = c.clone().map(|c| c[2] as u8).collect();
    let v = c.map(|c| c[3]).collect();
    FerryRoute::new(pts, kinds, v, false)
}

#[wasm_bindgen]
impl TrackApi {
    /// Add a blueprint port: its landing at (x, y) on the shore, its berth at (bx, by) in the water,
    /// within `ferry::BERTH_MAX_M` of it. 0 when added; `last_port` is then its node.
    pub fn add_port(&mut self, x: f64, y: f64, bx: f64, by: f64, name: String) -> u32 {
        let node = self.w.net.alloc_node();
        let st = StationData { platform: ferry::PORT_PLATFORM, name, named: false, built: false, off: false, upgrades: 0, berth: Some([quant(bx), quant(by)]), day: 0 };
        let op = Op::Batch(vec![Op::Node { id: node, data: Some(NodeData { x: quant(x), y: quant(y), level: 0, flying: false }) }, Op::Station { node, data: Some(st) }]);
        let r = self.run(op);
        if r == 0 {
            self.last_port = node;
        } else {
            self.w.net.release_node(node);
            self.last_port = u32::MAX;
        }
        r
    }
    /// The node of the port the last `add_port` added (`u32::MAX` if it was refused).
    pub fn last_port(&self) -> u32 {
        self.last_port
    }
    /// Whether a node is a port.
    pub fn is_port(&self, node: u32) -> bool {
        self.w.net.is_port(node)
    }
    /// A port's berth, [x, y]; empty for any other node.
    pub fn port_berth(&self, node: u32) -> Vec<f64> {
        if !self.w.net.is_port(node) {
            return vec![];
        }
        self.w.net.node_berth[node as usize].map_or(vec![], |b| b.to_vec())
    }
    /// The live ports.
    pub fn ports(&self) -> Vec<u32> {
        let n = &self.w.net;
        (0..n.node_count() as u32).filter(|&k| n.is_port(k)).collect()
    }

    /// Add a planned ferry line over `ports` (in route order) and a route (x, y, kind, m/s per point
    /// in the network frame), its boats following a headway in seconds per demand level (0: none).
    /// What a new line starts with is the app's (`NEW_LINE` in app/src/game/newLine.ts).
    #[allow(clippy::too_many_arguments)]
    pub fn set_ferry_line(&mut self, id: u32, ports: &[u32], points: &[f64], high: u32, medium: u32, low: u32, dwell_s: f32, turnaround_s: f32, colour: u32, name: String) -> u32 {
        let r = route_from(points);
        if let Err(i) = ferry::check(&self.w.net, id, ports, &r) {
            return self.refuse(i);
        }
        let data = LineData {
            name, symbol: String::new(), colour, stops: ports.to_vec(), path: vec![], tph: [0.0; 3], trains: Some([0; 3]), auto: Some([high, medium, low]),
            dwell_s, turnaround_s, cars: 1, fast: false, bus: None, ferry: Some(r), group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false,
        };
        self.run(Op::Line { id, data: Some(data) })
    }
    /// Draw a ferry line's route again, with its ports, keeping everything else. The route is free;
    /// a constructed line stays constructed and its new ports are constructed with it, paid now.
    pub fn set_ferry_route(&mut self, id: u32, ports: &[u32], points: &[f64]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(built) = d.ferry.as_ref().map(|r| r.built) else { return self.refuse(Issue::LinePath { line: id }) };
        let mut r = route_from(points);
        r.built = built;
        if let Err(i) = ferry::check(&self.w.net, id, ports, &r) {
            return self.refuse(i);
        }
        d.stops = ports.to_vec();
        d.ferry = Some(r);
        // a loop names its first port twice (T-2899)
        let mut once = ports.to_vec();
        once.sort_unstable();
        once.dedup();
        let mut ops = if built { self.construct_ops(&[], &once, &[]) } else { vec![] };
        if ops.is_empty() {
            return self.run(Op::Line { id, data: Some(d) });
        }
        ops.push(Op::Line { id, data: Some(d) });
        self.run_charged(Op::Batch(ops))
    }
    /// Whether a line is a ferry line.
    pub fn is_ferry(&self, line: u32) -> bool {
        self.w.net.is_ferry(line)
    }
    /// A ferry line's route: constructed (1/0), then x, y, kind, m/s per point; empty for any other
    /// line. Its ports are `line_stops`; `bus_stops` gives them as its service has them.
    pub fn ferry_route(&self, line: u32) -> Vec<f64> {
        let Some(r) = self.w.net.line_ferry.get(line as usize).and_then(|f| f.as_ref()).filter(|_| self.w.net.line_ok(line)) else { return vec![] };
        let mut out = vec![r.built as u8 as f64];
        for ((p, &k), &v) in r.points.iter().zip(&r.kinds).zip(&r.v) {
            out.extend([p[0], p[1], k as f64, ferry::v_mm(v) as f64 / 1000.0]);
        }
        out
    }
    /// A ferry line's route length, ground m (0 for any other line).
    pub fn ferry_m(&self, line: u32) -> f64 {
        self.w.net.ferry_m(line)
    }
    /// The drawing tool's quote for a route not yet a line (`line`: the line it would replace, or any
    /// id no line has): [ground m, a boat's run from the first port to the last standing `dwell_s` at
    /// each port between (s), US$M for its ports not yet constructed, 1 if it makes a line].
    pub fn ferry_quote(&self, line: u32, ports: &[u32], points: &[f64], dwell_s: f64) -> Vec<f64> {
        let net = &self.w.net;
        let r = route_from(points);
        let ok = ferry::check(net, line, ports, &r).is_ok();
        let cum = ferry::ground_cum(net, &r.points);
        let len = cum.last().copied().unwrap_or(0.0);
        let run = if ok {
            let stops: Vec<f64> = r.port_points().iter().map(|&i| cum[i]).collect();
            let dw = vec![dwell_s.max(0.0); stops.len()];
            ferry::profile(&ferry::limits(&r, &cum), &stops, &dw).duration
        } else {
            0.0
        };
        let mut new_ports: Vec<u32> = ports.iter().copied().filter(|&p| net.is_port(p) && !net.node_built[p as usize]).collect();
        new_ports.sort_unstable();
        new_ports.dedup();
        vec![len, run, new_ports.len() as f64 * PORT_COST, ok as u8 as f64]
    }
}

impl TrackApi {
    /// The live ferry lines.
    pub fn ferry_lines(&self) -> Vec<u32> {
        self.w.net.ferry_lines()
    }
}
