//! Bus lines (T-1464, notes/T-1445.md): a line with no track, over a route the player draws. The
//! route runs straight between the clicked points; its road length is the drawn length times
//! `ROAD_FACTOR`, which sets the run time, the price and the running cost. A bus runs a flat
//! `BUS_V` door to door, standing the line's dwell only at the train stations it meets.
//!
//! Demand sees a few stops of each route (`plan`): the train stations it passes within
//! `STATION_REACH_M`, its ends, a stop where it crosses another bus route, and one every
//! `DEMAND_STOP_M` or so between, each on the grid of street stops every `STREET_STOP_M`, which are
//! otherwise only drawn. Every point of the route belongs to its nearest stop that is not a station
//! (the stop's `cover`), so a home near the route walks to the route and rides from there.

use super::net::Network;
use super::profile::{Phase, Profile};

/// A bus's speed, street stops and traffic included, m/s.
pub const BUS_V: f64 = 18.0 / 3.6;
/// Road metres per drawn metre: a route clicked along the streets runs about 1.0, a diagonal across
/// a grid 1.27-1.41.
pub const ROAD_FACTOR: f64 = 1.2;
/// A train station this near the route (ground m) is a stop.
pub const STATION_REACH_M: f64 = 200.0;
/// Street stops along the road, m.
pub const STREET_STOP_M: f64 = 400.0;
/// About one demand stop this far along the road, m.
pub const DEMAND_STOP_M: f64 = 2000.0;
/// Two stations closer than this along the road: the bus stops at the one nearer the route.
const STATION_GAP_M: f64 = 100.0;
/// Two stations this far apart along the road get a stop between them even under `DEMAND_STOP_M`.
const STATION_PAIR_GAP_M: f64 = 800.0;
/// A bus's length, m.
pub const BUS_LEN_M: f64 = 12.0;
/// The shortest route, ground m.
pub const MIN_ROUTE_M: f64 = 100.0;
/// The most points a route may have.
pub const MAX_ROUTE_POINTS: usize = 400;
/// Ids of the stops that are no station: `STOP_ID_BASE | line << 10 | index`, never a node id.
pub const STOP_ID_BASE: u32 = 1 << 30;

pub fn stop_id(line: u32, k: usize) -> u32 {
    STOP_ID_BASE | (line << 10) | (k as u32 & 1023)
}
pub fn is_stop_id(id: u32) -> bool {
    id >= STOP_ID_BASE
}

/// A bus line's route as the player drew it: plane points in the network's frame, and whether it
/// is constructed (paid for; it runs).
#[derive(Clone, Debug, PartialEq)]
pub struct BusRoute {
    pub points: Vec<[f64; 2]>,
    pub built: bool,
}

pub const KIND_END: u8 = 0;
pub const KIND_STATION: u8 = 1;
pub const KIND_MIDDLE: u8 = 2;
pub const KIND_CROSSING: u8 = 3;

/// One of the stops demand sees.
#[derive(Clone, Debug, PartialEq)]
pub struct BusStop {
    pub id: u32,
    /// Road metres from the route's start.
    pub s: f64,
    /// Where riders board: the station's node, or the point on the route.
    pub x: f64,
    pub y: f64,
    pub kind: u8,
    pub station: Option<u32>,
    /// The stretch of the route (ground metres from its start) whose homes walk to the route and
    /// ride from this stop; None at a station, which riders reach on foot.
    pub cover: Option<[f64; 2]>,
}

/// A route's stops: those demand sees, the street stops between (road m from the start), and the
/// road length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub stops: Vec<BusStop>,
    pub street: Vec<f64>,
    pub len: f64,
}

/// A route's points with their ground metres from the start.
pub struct Poly<'a> {
    pub pts: &'a [[f64; 2]],
    pub cum: Vec<f64>,
}

impl<'a> Poly<'a> {
    pub fn new(net: &Network, pts: &'a [[f64; 2]]) -> Poly<'a> {
        let mut cum = Vec::with_capacity(pts.len());
        let mut g = 0.0;
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                let q = pts[i - 1];
                g += (p[0] - q[0]).hypot(p[1] - q[1]) / net.k_at(0.5 * (p[1] + q[1]));
            }
            cum.push(g);
        }
        Poly { pts, cum }
    }
    /// Ground metres.
    pub fn len(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }
    /// The point `g` ground metres from the start.
    pub fn at(&self, g: f64) -> (f64, f64) {
        let n = self.pts.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        if n == 1 || g <= 0.0 {
            return (self.pts[0][0], self.pts[0][1]);
        }
        let i = self.cum.partition_point(|&c| c <= g).clamp(1, n - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let seg = self.cum[i] - self.cum[i - 1];
        let t = if seg > 0.0 { ((g - self.cum[i - 1]) / seg).clamp(0.0, 1.0) } else { 0.0 };
        (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    }
    /// The route's nearest point to (x, y): ground metres from the start and ground distance.
    pub fn project(&self, net: &Network, x: f64, y: f64) -> (f64, f64) {
        let mut best = (0.0, f64::INFINITY);
        for i in 1..self.pts.len() {
            let (a, b) = (self.pts[i - 1], self.pts[i]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 { (((x - a[0]) * dx + (y - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let (px, py) = (a[0] + t * dx, a[1] + t * dy);
            let d = (x - px).hypot(y - py) / net.k_at(py);
            if d < best.1 {
                best = (self.cum[i - 1] + t * (self.cum[i] - self.cum[i - 1]), d);
            }
        }
        best
    }
    /// Plane bounding box, `[xmin, ymin, xmax, ymax]`.
    pub fn bbox(&self) -> [f64; 4] {
        self.pts.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
    }
}

/// Where two routes cross: ground metres along `a` of each crossing.
fn crossings(a: &Poly, b: &Poly) -> Vec<f64> {
    let mut out = vec![];
    for i in 1..a.pts.len() {
        let (p, q) = (a.pts[i - 1], a.pts[i]);
        for j in 1..b.pts.len() {
            let (r, s) = (b.pts[j - 1], b.pts[j]);
            let (d1x, d1y, d2x, d2y) = (q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]);
            let den = d1x * d2y - d1y * d2x;
            if den.abs() < 1e-12 {
                continue;
            }
            let t = ((r[0] - p[0]) * d2y - (r[1] - p[1]) * d2x) / den;
            let u = ((r[0] - p[0]) * d1y - (r[1] - p[1]) * d1x) / den;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                out.push(a.cum[i - 1] + t * (a.cum[i] - a.cum[i - 1]));
            }
        }
    }
    out
}

/// Whether a route's points make a route at all: two or more, finite, inside the world, not too
/// many, and at least `MIN_ROUTE_M` long.
pub fn route_ok(net: &Network, pts: &[[f64; 2]]) -> bool {
    pts.len() >= 2
        && pts.len() <= MAX_ROUTE_POINTS
        && pts.iter().all(|p| p[0].is_finite() && p[1].is_finite() && net.inside_world(p[0], p[1]))
        && Poly::new(net, pts).len() >= MIN_ROUTE_M
}

/// Line `l`'s stops over route `pts`. `built_only`: a constructed route stops only at constructed
/// stations and crosses only constructed routes; a planned one previews them all.
pub fn plan(net: &Network, l: u32, pts: &[[f64; 2]], built_only: bool) -> Plan {
    if !route_ok(net, pts) {
        return Plan::default();
    }
    let poly = Poly::new(net, pts);
    let g_len = poly.len();
    let len = g_len * ROAD_FACTOR;
    let bb = poly.bbox();
    let pad = STATION_REACH_M * net.k_at(0.5 * (bb[1] + bb[3])) * 1.1 + 1.0;
    // stations within reach: (road s, node, distance)
    let mut st: Vec<(f64, u32, f64)> = vec![];
    for n in 0..net.node_count() as u32 {
        let i = n as usize;
        if !net.is_station(n) || (built_only && !net.node_built[i]) {
            continue;
        }
        let (x, y) = (net.node_x[i], net.node_y[i]);
        if x < bb[0] - pad || x > bb[2] + pad || y < bb[1] - pad || y > bb[3] + pad {
            continue;
        }
        let (g, d) = poly.project(net, x, y);
        if d <= STATION_REACH_M {
            st.push((g * ROAD_FACTOR, n, d));
        }
    }
    st.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut kept: Vec<(f64, u32, f64)> = vec![];
    for c in st {
        match kept.last_mut() {
            Some(k) if c.0 - k.0 < STATION_GAP_M => {
                if c.2 < k.2 {
                    *k = c;
                }
            }
            _ => kept.push(c),
        }
    }
    // fixed stops: (road s, kind, station)
    let mut fixed: Vec<(f64, u8, Option<u32>)> = vec![];
    let start_merged = kept.first().filter(|k| k.0 <= STATION_REACH_M).map(|k| k.1);
    let end_merged = kept.last().filter(|k| k.0 >= len - STATION_REACH_M && Some(k.1) != start_merged).map(|k| k.1);
    if start_merged.is_none() {
        fixed.push((0.0, KIND_END, None));
    }
    fixed.extend(kept.iter().map(|k| (k.0, KIND_STATION, Some(k.1))));
    if end_merged.is_none() {
        fixed.push((len, KIND_END, None));
    }
    let near_fixed = |s: f64, fixed: &[(f64, u8, Option<u32>)]| fixed.iter().any(|f| (f.0 - s).abs() < STATION_REACH_M);
    let mut cross: Vec<f64> = vec![];
    for o in 0..net.line_count() as u32 {
        if o == l || !net.line_ok(o) {
            continue;
        }
        let Some(r) = net.line_bus[o as usize].as_ref() else { continue };
        if built_only && !r.built {
            continue;
        }
        cross.extend(crossings(&poly, &Poly::new(net, &r.points)).into_iter().map(|g| g * ROAD_FACTOR));
    }
    cross.sort_by(f64::total_cmp);
    for s in cross {
        if !near_fixed(s, &fixed) {
            fixed.push((s, KIND_CROSSING, None));
        }
    }
    fixed.sort_by(|a, b| a.0.total_cmp(&b.0));
    // the street grid and the middles, gap by gap
    let mut street = vec![];
    let mut middles: Vec<f64> = vec![];
    for w in fixed.windows(2) {
        let (s0, s1) = (w[0].0, w[1].0);
        let g = s1 - s0;
        if g <= 0.0 {
            continue;
        }
        let n = ((g / STREET_STOP_M).round() as usize).max(1);
        let step = g / n as f64;
        let mut k = ((g / DEMAND_STOP_M - 1e-9).ceil() as usize).saturating_sub(1);
        if k == 0 && g >= STATION_PAIR_GAP_M && w[0].1 == KIND_STATION && w[1].1 == KIND_STATION {
            k = 1;
        }
        let mut picked: Vec<usize> = vec![];
        if n >= 2 {
            for j in 1..=k {
                let ideal = j as f64 * g / (k + 1) as f64;
                let i = ((ideal / step).round() as usize).clamp(1, n - 1);
                if !picked.contains(&i) {
                    picked.push(i);
                    middles.push(s0 + i as f64 * step);
                }
            }
        }
        for i in 1..n {
            if !picked.contains(&i) {
                street.push(s0 + i as f64 * step);
            }
        }
    }
    let mut all: Vec<(f64, u8, Option<u32>)> = fixed;
    all.extend(middles.into_iter().map(|s| (s, KIND_MIDDLE, None)));
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut stops: Vec<BusStop> = all
        .iter()
        .enumerate()
        .map(|(k, &(s, kind, station))| {
            let (x, y) = match station {
                Some(n) => (net.node_x[n as usize], net.node_y[n as usize]),
                None => poly.at(s / ROAD_FACTOR),
            };
            BusStop { id: station.unwrap_or_else(|| stop_id(l, k)), s, x, y, kind, station, cover: None }
        })
        .collect();
    let open: Vec<usize> = (0..stops.len()).filter(|&k| stops[k].station.is_none()).collect();
    for (j, &k) in open.iter().enumerate() {
        let g = stops[k].s / ROAD_FACTOR;
        let a = if j == 0 { 0.0 } else { 0.5 * (g + stops[open[j - 1]].s / ROAD_FACTOR) };
        let b = if j + 1 == open.len() { g_len } else { 0.5 * (g + stops[open[j + 1]].s / ROAD_FACTOR) };
        stops[k].cover = Some([a, b]);
    }
    Plan { stops, street, len }
}

/// A bus's trip over stops at road offsets `stop_s` (ascending), standing `dwell[i]` at each
/// stop between the ends: constant speed between stops, no accelerating.
pub fn profile(stop_s: &[f64], dwell: &[f64]) -> Profile {
    let n = stop_s.len();
    let mut p = Profile::default();
    if n < 2 {
        return p;
    }
    let mut t = 0.0;
    p.arr.push(0.0);
    p.dep.push(0.0);
    for i in 0..n - 1 {
        p.phases.push(Phase { t, s: stop_s[i], v: BUS_V, a: 0.0 });
        t += (stop_s[i + 1] - stop_s[i]).max(0.0) / BUS_V;
        p.arr.push(t);
        let stand = if i + 1 < n - 1 { dwell.get(i + 1).copied().unwrap_or(0.0).max(0.0) } else { 0.0 };
        if stand > 0.0 {
            p.phases.push(Phase { t, s: stop_s[i + 1], v: 0.0, a: 0.0 });
            p.dwell += stand;
            t += stand;
        }
        p.dep.push(t);
    }
    p.phases.push(Phase { t, s: stop_s[n - 1], v: 0.0, a: 0.0 });
    p.duration = t;
    p
}

#[cfg(test)]
mod tests;
