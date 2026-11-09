//! Bus lines (T-1464, T-1465, notes/T-1445.md): a line with no track, over a route the player
//! draws. Between two clicked points (its corners) the route follows the roads where the app found
//! a road path (`K_ROAD`: the road length is the drawn length) and runs straight elsewhere, where
//! the road length is the drawn length times `ROAD_FACTOR`. The road length sets the run time, the
//! price and the running cost. A hop between two stops takes the car's time over its road, by
//! demand's car model with the streets it crosses (`Pace`, `streets.rs`), at the bus's share of the
//! car's speed, plus `streets::BUS_STOP_S` for the stop (T-2904). Automatic stops count that cost
//! every `STREET_STOP_M` of road. The bus stands the line's dwell at the train stations it meets
//! and `STREET_DWELL_S` at each street stop (drawn only, within the hop's time).
//!
//! Demand sees a few stops of each route (`plan`): the train stations it passes within
//! `STATION_REACH_M`, its ends, a stop where it crosses another bus route, and one every
//! `DEMAND_STOP_M` or so between, each on the grid of street stops every `STREET_STOP_M`, which are
//! otherwise only drawn. Every point of the route belongs to its nearest stop that is not a station
//! (the stop's `cover`), so a home near the route walks to the route and rides from there.

use super::net::Network;
use super::profile::{Phase, Profile};
use super::streets::{bus_kmh, car_main_kmh, BUS_STOP_S, CAR_END_M, CAR_LOCAL_M, CAR_OPEN_KMH, CAR_TOWN_KMH, CAR_TRUNK_KMH};

/// Road metres of a route that share one speed (`Pace`).
const PACE_STEP_M: f64 = 100.0;
/// Road metres per drawn metre on a straight stretch of route: a diagonal across a grid runs
/// 1.27-1.41, a stretch drawn along the streets 1.0. A stretch the app laid along the roads
/// (`K_ROAD`) counts at 1.0.
pub const ROAD_FACTOR: f64 = 1.2;
/// A route point the player clicked (the app keeps these to draw the route again).
pub const K_CORNER: u8 = 1;
/// The route up to this point follows roads: its road length is its drawn length.
pub const K_ROAD: u8 = 2;
/// The stretch up to this point is on a trunk road (T-2904). A save from before has none.
pub const K_TRUNK: u8 = 4;
/// A train station this near the route (ground m) is a stop.
pub const STATION_REACH_M: f64 = 200.0;
/// Street stops along the road, m.
pub const STREET_STOP_M: f64 = 400.0;
/// About one demand stop this far along the road, m.
pub const DEMAND_STOP_M: f64 = 2000.0;
/// Seconds a bus stands at a street stop, taken out of the hop's running time (T-1798).
pub const STREET_DWELL_S: f64 = 20.0;
/// Two stations closer than this along the road: the bus stops at the one nearer the route.
const STATION_GAP_M: f64 = 100.0;
/// Two stations this far apart along the road get a stop between them even under `DEMAND_STOP_M`.
const STATION_PAIR_GAP_M: f64 = 800.0;
/// A stop on a line with its own spacing (`BusRoute::spacing`) serves this far of the route each
/// way, m: what stops further apart than twice this leave between them has no bus (T-2885).
const SPACED_COVER_M: f64 = 400.0;
/// The closest and farthest apart a player may set a line's stops, m.
pub const SPACING_MIN_M: u32 = 400;
pub const SPACING_MAX_M: u32 = 100_000;
/// The most stops `plan` gives a line with its own spacing: stop ids hold 1,024 a line.
const SPACED_MAX_STOPS: f64 = 1000.0;
/// A bus's length, m.
pub const BUS_LEN_M: f64 = 12.0;
/// The shortest route, ground m.
pub const MIN_ROUTE_M: f64 = 100.0;
/// The most points a route may have (a road-following route has many).
pub const MAX_ROUTE_POINTS: usize = 8000;
/// Ids of the stops that are no station: `STOP_ID_BASE | line << 10 | index`, never a node id.
pub const STOP_ID_BASE: u32 = 1 << 30;

pub fn stop_id(line: u32, k: usize) -> u32 {
    STOP_ID_BASE | (line << 10) | (k as u32 & 1023)
}
pub fn is_stop_id(id: u32) -> bool {
    id >= STOP_ID_BASE
}

/// A bus line's route: plane points in the network's frame, each with its `K_*` bits (`kinds`, as
/// long as `points`), whether it is constructed (paid for; it runs), and how far apart the player
/// set its stops, m (`spacing`, 0: they place themselves, T-2885).
#[derive(Clone, Debug, PartialEq)]
pub struct BusRoute {
    pub points: Vec<[f64; 2]>,
    pub kinds: Vec<u8>,
    pub built: bool,
    pub spacing: u32,
}

impl BusRoute {
    /// A route with `kinds` cut or padded (with plain corners) to its points.
    pub fn new(points: Vec<[f64; 2]>, mut kinds: Vec<u8>, built: bool) -> BusRoute {
        kinds.resize(points.len(), K_CORNER);
        BusRoute { points, kinds, built, spacing: 0 }
    }
}

/// `m` as a line's stop spacing: 0 stays 0 (automatic), anything else inside the limits.
pub fn spacing_from(m: f64) -> u32 {
    if m > 0.0 { (m.round() as u32).clamp(SPACING_MIN_M, SPACING_MAX_M) } else { 0 }
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
    /// The stretch of the route (road metres from its start) whose homes walk to the route and
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
    /// Road metres from the start at each of the route's points.
    pub cum: Vec<f64>,
}

/// A route's points with their ground and road metres from the start.
pub struct Poly<'a> {
    pub pts: &'a [[f64; 2]],
    /// ground metres
    pub cum: Vec<f64>,
    /// road metres: each stretch's ground length times 1.0 on road, `ROAD_FACTOR` otherwise
    pub road: Vec<f64>,
}

/// The point `s` road metres along a route whose points have road metres `cum`.
pub fn at_road(pts: &[[f64; 2]], cum: &[f64], s: f64) -> (f64, f64) {
    let n = pts.len().min(cum.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 || s <= 0.0 {
        return (pts[0][0], pts[0][1]);
    }
    let i = cum[..n].partition_point(|&c| c <= s).clamp(1, n - 1);
    let (a, b) = (pts[i - 1], pts[i]);
    let seg = cum[i] - cum[i - 1];
    let t = if seg > 0.0 { ((s - cum[i - 1]) / seg).clamp(0.0, 1.0) } else { 0.0 };
    (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
}

impl<'a> Poly<'a> {
    /// `kinds[i]` has `K_ROAD` when the stretch ending at point i follows roads; a missing entry is
    /// a straight stretch.
    pub fn new(net: &Network, pts: &'a [[f64; 2]], kinds: &[u8]) -> Poly<'a> {
        let mut cum = Vec::with_capacity(pts.len());
        let mut road = Vec::with_capacity(pts.len());
        let (mut g, mut r) = (0.0, 0.0);
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                let q = pts[i - 1];
                let d = (p[0] - q[0]).hypot(p[1] - q[1]) / net.k_at(0.5 * (p[1] + q[1]));
                g += d;
                r += d * if kinds.get(i).is_some_and(|k| k & K_ROAD != 0) { 1.0 } else { ROAD_FACTOR };
            }
            cum.push(g);
            road.push(r);
        }
        Poly { pts, cum, road }
    }
    /// Ground metres.
    pub fn len(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }
    /// Road metres.
    pub fn road_len(&self) -> f64 {
        self.road.last().copied().unwrap_or(0.0)
    }
    /// The point `s` road metres from the start.
    pub fn at_road(&self, s: f64) -> (f64, f64) {
        at_road(self.pts, &self.road, s)
    }
    /// The route's nearest point to (x, y): road metres from the start and ground distance.
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
                best = (self.road[i - 1] + t * (self.road[i] - self.road[i - 1]), d);
            }
        }
        best
    }
    /// Plane bounding box, `[xmin, ymin, xmax, ymax]`.
    pub fn bbox(&self) -> [f64; 4] {
        self.pts.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
    }
}

/// Segments per box in `crossings`.
const CHUNK: usize = 16;

/// Boxes of runs of `CHUNK` segments: `[xmin, ymin, xmax, ymax]`.
fn chunk_boxes(pts: &[[f64; 2]]) -> Vec<[f64; 4]> {
    (1..pts.len())
        .step_by(CHUNK)
        .map(|i| {
            let hi = (i + CHUNK).min(pts.len());
            pts[i - 1..hi].iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
        })
        .collect()
}

fn boxes_meet(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Where two routes cross: road metres along `a` of each crossing. Runs of segments are compared by
/// their boxes first, so two long road-following routes cost little.
fn crossings(a: &Poly, b: &Poly) -> Vec<f64> {
    let mut out = vec![];
    let (ca, cb) = (chunk_boxes(a.pts), chunk_boxes(b.pts));
    for (ia, ba) in ca.iter().enumerate() {
        for (ib, bb) in cb.iter().enumerate() {
            if !boxes_meet(ba, bb) {
                continue;
            }
            for i in 1 + ia * CHUNK..(1 + (ia + 1) * CHUNK).min(a.pts.len()) {
                let (p, q) = (a.pts[i - 1], a.pts[i]);
                for j in 1 + ib * CHUNK..(1 + (ib + 1) * CHUNK).min(b.pts.len()) {
                    let (r, s) = (b.pts[j - 1], b.pts[j]);
                    let (d1x, d1y, d2x, d2y) = (q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]);
                    let den = d1x * d2y - d1y * d2x;
                    if den.abs() < 1e-12 {
                        continue;
                    }
                    let t = ((r[0] - p[0]) * d2y - (r[1] - p[1]) * d2x) / den;
                    let u = ((r[0] - p[0]) * d1y - (r[1] - p[1]) * d1x) / den;
                    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                        out.push(a.road[i - 1] + t * (a.road[i] - a.road[i - 1]));
                    }
                }
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// A line of points thinned to within `tol` (plane units) of the original.
pub fn simplify(pts: &[(f64, f64)], tol: f64) -> Vec<(f64, f64)> {
    let n = pts.len();
    if n <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0, n - 1)];
    while let Some((a, b)) = stack.pop() {
        let (ax, ay, bx, by) = (pts[a].0, pts[a].1, pts[b].0, pts[b].1);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let (mut worst, mut at) = (0.0, 0);
        for (i, p) in pts.iter().enumerate().take(b).skip(a + 1) {
            let (px, py) = (p.0 - ax, p.1 - ay);
            let t = if l2 > 0.0 { ((px * dx + py * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = (px - t * dx).hypot(py - t * dy);
            if d > worst {
                (worst, at) = (d, i);
            }
        }
        if worst > tol {
            keep[at] = true;
            stack.push((a, at));
            stack.push((at, b));
        }
    }
    pts.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

/// Whether a route's points make a route at all: two or more, finite, inside the world, not too
/// many, and at least `MIN_ROUTE_M` long.
pub fn route_ok(net: &Network, pts: &[[f64; 2]], kinds: &[u8]) -> bool {
    pts.len() >= 2
        && pts.len() <= MAX_ROUTE_POINTS
        && pts.iter().all(|p| p[0].is_finite() && p[1].is_finite() && net.inside_world(p[0], p[1]))
        && Poly::new(net, pts, kinds).len() >= MIN_ROUTE_M
}

/// Stops `sp` road metres or so apart from the first of `fixed` to its last (T-2885): each lands on
/// the station nearest its place within half `sp`, else the crossing, else the place itself, and
/// no two come closer than half `sp`, so a line can pass a station without stopping.
fn spaced_stops(fixed: &[(f64, u8, Option<u32>)], sp: f64) -> Vec<(f64, u8, Option<u32>)> {
    let (Some(&first), Some(&last)) = (fixed.first(), fixed.last()) else { return vec![] };
    if fixed.len() < 2 {
        return fixed.to_vec();
    }
    let inner = &fixed[1..fixed.len() - 1];
    let mut out = vec![first];
    let mut cur = first.0;
    loop {
        let target = cur + sp;
        if target >= last.0 - 0.5 * sp {
            break;
        }
        let (lo, hi) = (cur + 0.5 * sp, (target + 0.5 * sp).min(last.0 - 0.5 * sp));
        let nearest = |kind: u8| inner.iter().filter(|c| c.1 == kind && c.0 >= lo && c.0 <= hi).min_by(|a, b| (a.0 - target).abs().total_cmp(&(b.0 - target).abs())).copied();
        let stop = nearest(KIND_STATION).or_else(|| nearest(KIND_CROSSING)).unwrap_or((target, KIND_MIDDLE, None));
        cur = stop.0;
        out.push(stop);
    }
    out.push(last);
    out
}

/// Line `l`'s stops over route `pts`, placed by themselves. `built_only`: a constructed route stops
/// only at constructed stations and crosses only constructed routes; a planned one previews them all.
pub fn plan(net: &Network, l: u32, pts: &[[f64; 2]], kinds: &[u8], built_only: bool) -> Plan {
    plan_spaced(net, l, pts, kinds, built_only, 0)
}

/// A route's stops by its own spacing (`BusRoute::spacing`).
pub fn plan_route(net: &Network, l: u32, r: &BusRoute, built_only: bool) -> Plan {
    plan_spaced(net, l, &r.points, &r.kinds, built_only, r.spacing)
}

/// As `plan`, the stops `spacing` metres or so apart unless that is 0 (T-2885): then every stop is one
/// demand sees, there are no street stops, and each stop serves `SPACED_COVER_M` of the route each way.
pub fn plan_spaced(net: &Network, l: u32, pts: &[[f64; 2]], kinds: &[u8], built_only: bool, spacing: u32) -> Plan {
    if !route_ok(net, pts, kinds) {
        return Plan::default();
    }
    let poly = Poly::new(net, pts, kinds);
    let len = poly.road_len();
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
            st.push((g, n, d));
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
        cross.extend(crossings(&poly, &Poly::new(net, &r.points, &r.kinds)));
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
    let spaced = spacing > 0;
    if spaced {
        fixed = spaced_stops(&fixed, (spacing as f64).max(len / SPACED_MAX_STOPS));
    }
    let auto_gaps: &[(f64, u8, Option<u32>)] = if spaced { &[] } else { &fixed };
    for w in auto_gaps.windows(2) {
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
                None => poly.at_road(s),
            };
            BusStop { id: station.unwrap_or_else(|| stop_id(l, k)), s, x, y, kind, station, cover: None }
        })
        .collect();
    let open: Vec<usize> = (0..stops.len()).filter(|&k| stops[k].station.is_none()).collect();
    for (j, &k) in open.iter().enumerate() {
        let g = stops[k].s;
        let a = if j == 0 { 0.0 } else { 0.5 * (g + stops[open[j - 1]].s) };
        let b = if j + 1 == open.len() { len } else { 0.5 * (g + stops[open[j + 1]].s) };
        stops[k].cover = Some(if spaced { [a.max(g - SPACED_COVER_M), b.min(g + SPACED_COVER_M)] } else { [a, b] });
    }
    Plan { stops, street, len, cum: poly.road.clone() }
}

/// What a route's road is like along it: per `step` road metres the car's local street speed there,
/// km/h, and whether it is a trunk road. Empty: open country and no trunk all the way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pace {
    step: f64,
    car: Vec<f64>,
    /// empty when the route has no trunk road
    trunk: Vec<bool>,
}

impl Pace {
    /// From the street grids and the route's `kinds`, at the middle of each `PACE_STEP_M` of road.
    /// `cum`: the road metres at each of `pts`.
    pub fn along(net: &Network, pts: &[[f64; 2]], cum: &[f64], kinds: &[u8], len: f64) -> Pace {
        let has_trunk = kinds.iter().any(|k| k & K_TRUNK != 0);
        if (net.streets.is_empty() && !has_trunk) || len <= 0.0 {
            return Pace::default();
        }
        let n = (len / PACE_STEP_M).ceil().max(1.0) as usize;
        let step = len / n as f64;
        let mid = |i: usize| (i as f64 + 0.5) * step;
        let car = (0..n)
            .map(|i| {
                let (x, y) = at_road(pts, cum, mid(i));
                net.car_kmh_at(x, y)
            })
            .collect();
        let trunk = if has_trunk {
            // kinds[k] is the stretch ending at point k
            (0..n).map(|i| kinds.get(cum.partition_point(|&c| c < mid(i))).is_some_and(|k| k & K_TRUNK != 0)).collect()
        } else {
            vec![]
        };
        Pace { step, car, trunk }
    }
    /// The same route run from its other end.
    pub fn reversed(&self) -> Pace {
        Pace { step: self.step, car: self.car.iter().rev().copied().collect(), trunk: self.trunk.iter().rev().copied().collect() }
    }
    /// Seconds to run from road metre `a` to `b` (a before b) at `kmh(car's local street speed, trunk)`.
    fn seconds(&self, a: f64, b: f64, kmh: impl Fn(f64, bool) -> f64) -> f64 {
        let d = (b - a).max(0.0);
        if d == 0.0 {
            return 0.0;
        }
        if self.car.is_empty() {
            return d * 3.6 / kmh(CAR_TOWN_KMH, false);
        }
        let last = self.car.len() - 1;
        let at = |i: usize| 3.6 / kmh(self.car[i], self.trunk.get(i).copied().unwrap_or(false));
        let (i0, i1) = (((a / self.step).floor().max(0.0) as usize).min(last), ((b / self.step).floor().max(0.0) as usize).min(last));
        if i0 == i1 {
            return d * at(i0);
        }
        let mut t = ((i0 + 1) as f64 * self.step - a) * at(i0) + (b - i1 as f64 * self.step) * at(i1);
        for i in i0 + 1..i1 {
            t += self.step * at(i);
        }
        t
    }
    /// Seconds for road metres `a` to `b` on automatic stops: all local streets, a stop every
    /// `STREET_STOP_M`.
    pub fn auto_time(&self, a: f64, b: f64) -> f64 {
        self.seconds(a, b, street_bus_kmh) + BUS_STOP_S * (b - a).max(0.0) / STREET_STOP_M
    }
    /// Seconds for a hop between two stops `a` and `b` road metres along: the car's model over the
    /// hop, local streets at each end, then main roads, then open highway, plus one stop.
    pub fn hop_time(&self, a: f64, b: f64) -> f64 {
        let d = (b - a).max(0.0);
        let local = (0.5 * d).min(CAR_LOCAL_M);
        let main = (0.5 * (d - 2.0 * local)).min(CAR_END_M);
        let (a1, a2, b2, b1) = (a + local, a + local + main, b - local - main, b - local);
        self.seconds(a, a1, street_bus_kmh) + self.seconds(a1, a2, main_bus_kmh) + self.seconds(a2, b2, |_, _| bus_kmh(CAR_OPEN_KMH)) + self.seconds(b2, b1, main_bus_kmh) + self.seconds(b1, b, street_bus_kmh) + BUS_STOP_S
    }
}

fn street_bus_kmh(car: f64, trunk: bool) -> f64 {
    bus_kmh(if trunk { car.max(CAR_TRUNK_KMH) } else { car })
}

fn main_bus_kmh(car: f64, trunk: bool) -> f64 {
    let main = car_main_kmh(car);
    bus_kmh(if trunk { main.max(CAR_TRUNK_KMH) } else { main })
}

/// A bus's trip on automatic stops at road offsets `stop_s` (ascending), standing `dwell[i]` at each
/// stop between the ends: constant speed between stops (the hop's mean), no accelerating. It also
/// stands `STREET_DWELL_S` at each street stop of `street` (ascending) inside a hop, which is drawn
/// only: the hop's time stays what `pace` gives, the bus running faster between the stands.
pub fn profile(stop_s: &[f64], dwell: &[f64], street: &[f64], pace: &Pace) -> Profile {
    run_profile(stop_s, dwell, street, pace, false)
}

/// The trip of a line with its own spacing (T-2885): each hop is `Pace::hop_time`, and the bus
/// stands `STREET_DWELL_S` inside it at each stop with no dwell of its own.
pub fn spaced_profile(stop_s: &[f64], dwell: &[f64], pace: &Pace) -> Profile {
    run_profile(stop_s, dwell, &[], pace, true)
}

fn run_profile(stop_s: &[f64], dwell: &[f64], street: &[f64], pace: &Pace, spaced: bool) -> Profile {
    let n = stop_s.len();
    let mut p = Profile::default();
    if n < 2 {
        return p;
    }
    let mut t = 0.0;
    p.arr.push(0.0);
    p.dep.push(0.0);
    for i in 0..n - 1 {
        let mut run = if spaced { pace.hop_time(stop_s[i], stop_s[i + 1]) } else { pace.auto_time(stop_s[i], stop_s[i + 1]) };
        let d = (stop_s[i + 1] - stop_s[i]).max(0.0);
        let mut own_stand = 0.0;
        if spaced && i + 1 < n - 1 && dwell.get(i + 1).copied().unwrap_or(0.0) <= 0.0 {
            own_stand = STREET_DWELL_S.min(BUS_STOP_S);
            run -= own_stand;
        }
        let lo = street.partition_point(|&s| s <= stop_s[i]);
        let hi = street.partition_point(|&s| s < stop_s[i + 1]);
        let ss = &street[lo..hi.max(lo)];
        // at most 40% of the hop standing; the rest is the same time spent moving
        let stand_at = if ss.is_empty() || run <= 0.0 { 0.0 } else { STREET_DWELL_S.min(0.4 * run / ss.len() as f64) };
        let moving = run - stand_at * ss.len() as f64;
        let v = if moving > 0.0 { d / moving } else { 0.0 };
        p.phases.push(Phase { t, s: stop_s[i], v, a: 0.0 });
        for (k, &s) in ss.iter().enumerate() {
            let arrive = t + (s - stop_s[i]) / v + stand_at * k as f64;
            p.phases.push(Phase { t: arrive, s, v: 0.0, a: 0.0 });
            p.phases.push(Phase { t: arrive + stand_at, s, v, a: 0.0 });
        }
        t += run;
        p.arr.push(t);
        let stand = own_stand + if i + 1 < n - 1 { dwell.get(i + 1).copied().unwrap_or(0.0).max(0.0) } else { 0.0 };
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
