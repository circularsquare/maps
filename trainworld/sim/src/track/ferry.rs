//! Ferry lines (T-2193, notes/T-2177.md 3): a line on water between ports. The app searches the route
//! (T-2192, notes/T-2177.md "F1 -> F2 route format") and hands over its points, each with a kind
//! (`K_*`) and the most the boat may do there; the sim takes them as they come and never sees the water.
//!
//! A port is a station of its own kind (`Network::is_port`): a node with no track, its landing on the
//! shore (the node, where riders walk to) and its berth in the water (`StationData::berth`), where
//! routes start and end. A ferry line's stops are its ports, in order; each is a `K_PORT` point of the
//! route at that port's berth.
//!
//! Between two points the boat runs at most the lower of their speeds; it accelerates and brakes at
//! `BOAT_A`, stops at each port and stands the line's dwell at those between its ends (`profile`).

use super::bus::{BusStop, Plan, KIND_STATION};
use super::net::{Issue, Network};
use super::profile::{Phase, Profile};

/// A point of the searched route.
pub const K_ROUTE: u8 = 0;
/// A port's berth: the boat stops.
pub const K_PORT: u8 = 1;
/// A waypoint: the route passes through it at its speed and does not stop.
pub const K_WAYPOINT: u8 = 2;

/// A boat's top speed, m/s: 25 km/h, 200 m or more from land (notes/T-2177.md 3.4).
pub const BOAT_TOP: f64 = 25.0 / 3.6;
/// The slowest speed a route point may ask for, m/s.
pub const BOAT_MIN: f64 = 0.25;
/// A boat's acceleration and braking, m/s²: 46 s and 160 m to reach 25 km/h.
pub const BOAT_A: f64 = 0.15;
/// A boat's length, m (drawn only).
pub const BOAT_LEN_M: f64 = 35.0;
/// A port's platform length (`Network::node_platform`, which marks every station): a port has none,
/// and no rule reads it.
pub const PORT_PLATFORM: u16 = 1;
/// The farthest a berth may be from its landing, ground m: the app looks within 300 m.
pub const BERTH_MAX_M: f64 = 320.0;
/// The farthest a route's `K_PORT` point may be from its port's berth, ground m.
pub const BERTH_TOL_M: f64 = 50.0;
/// The shortest route, ground m.
pub const MIN_ROUTE_M: f64 = 100.0;
/// The most points a route may have.
pub const MAX_ROUTE_POINTS: usize = 8000;
/// What a new ferry line starts with (the app passes them, `NEW_FERRY_*` in the clock worker): dwell
/// and turnaround, s, and a boat every 20 and 30 minutes at high and medium demand, none at low (T-2273).
pub const FERRY_DWELL_S: f32 = 60.0;
pub const FERRY_TURNAROUND_S: f32 = 300.0;
pub const FERRY_HEADWAYS_S: [u32; 3] = [1200, 1800, 0];

/// A ferry line's route: plane points in the network's frame, each with its `K_*` kind and the most
/// the boat may do there (m/s, on a mm/s grid so a save keeps it to the bit), and whether it is
/// constructed (it runs). Its ports are the line's stops.
#[derive(Clone, Debug, PartialEq)]
pub struct FerryRoute {
    pub points: Vec<[f64; 2]>,
    pub kinds: Vec<u8>,
    pub v: Vec<f32>,
    pub built: bool,
}

/// A speed as a route keeps it: m/s within `BOAT_MIN..=BOAT_TOP` on a mm/s grid; anything not a
/// positive number is the top speed.
pub fn quant_v(mps: f64) -> f32 {
    let v = if mps > 0.0 && mps.is_finite() { mps.clamp(BOAT_MIN, BOAT_TOP) } else { BOAT_TOP };
    ((v * 1000.0).round() / 1000.0) as f32
}

/// A kept speed in whole mm/s, as a save writes it.
pub fn v_mm(v: f32) -> u64 {
    (v as f64 * 1000.0).round().max(0.0) as u64
}

impl FerryRoute {
    /// A route with `kinds` and `v` cut or padded (route points at the top speed) to its points.
    pub fn new(points: Vec<[f64; 2]>, mut kinds: Vec<u8>, v: Vec<f64>, built: bool) -> FerryRoute {
        kinds.resize(points.len(), K_ROUTE);
        let mut v: Vec<f32> = v.into_iter().map(quant_v).collect();
        v.resize(points.len(), quant_v(BOAT_TOP));
        FerryRoute { points, kinds, v, built }
    }
    /// Indices of the `K_PORT` points.
    pub fn port_points(&self) -> Vec<usize> {
        (0..self.points.len()).filter(|&i| self.kinds[i] == K_PORT).collect()
    }
}

/// Ground metres from the route's start at each point.
pub fn ground_cum(net: &Network, pts: &[[f64; 2]]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(pts.len());
    let mut g = 0.0;
    for (i, p) in pts.iter().enumerate() {
        if i > 0 {
            let q = pts[i - 1];
            g += (p[0] - q[0]).hypot(p[1] - q[1]) / net.k_at(0.5 * (p[1] + q[1]));
        }
        cum.push(g);
    }
    cum
}

/// Ground metres between two plane points.
fn ground_m(net: &Network, a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1]) / net.k_at(0.5 * (a[1] + b[1]))
}

/// Whether a port's berth fits its landing: finite, inside the world, within `BERTH_MAX_M`.
pub fn berth_ok(net: &Network, landing: [f64; 2], berth: [f64; 2]) -> bool {
    berth.iter().all(|v| v.is_finite()) && net.inside_world(berth[0], berth[1]) && ground_m(net, landing, berth) <= BERTH_MAX_M
}

/// Whether a ferry line's stops (`ports`) and route make a line: two ports or more, no port twice in a
/// row, a route of 2 to `MAX_ROUTE_POINTS` points inside the world and `MIN_ROUTE_M` long, starting and
/// ending at a port, with one `K_PORT` point per stop within `BERTH_TOL_M` of that port's berth.
pub fn check(net: &Network, l: u32, ports: &[u32], r: &FerryRoute) -> Result<(), Issue> {
    let pts = &r.points;
    let path = Issue::LinePath { line: l };
    if pts.len() < 2 || pts.len() > MAX_ROUTE_POINTS || !pts.iter().all(|p| p[0].is_finite() && p[1].is_finite() && net.inside_world(p[0], p[1])) {
        return Err(path);
    }
    if ground_cum(net, pts).last().copied().unwrap_or(0.0) < MIN_ROUTE_M {
        return Err(path);
    }
    let stops = Issue::LineStops { line: l };
    if ports.len() < 2 || ports.windows(2).any(|w| w[0] == w[1]) || !ports.iter().all(|&p| net.is_port(p)) {
        return Err(stops);
    }
    let at = r.port_points();
    if at.len() != ports.len() || at.first() != Some(&0) || at.last() != Some(&(pts.len() - 1)) {
        return Err(stops);
    }
    for (&i, &p) in at.iter().zip(ports) {
        let Some(b) = net.node_berth[p as usize] else { return Err(stops) };
        if ground_m(net, pts[i], b) > BERTH_TOL_M {
            return Err(stops);
        }
    }
    Ok(())
}

/// The boat's speed limits along a route, run 0's way: `(from, to, m/s)` ground metres from the start,
/// covering `[0, len]`: each stretch between two points at the lower of their speeds.
pub fn limits(r: &FerryRoute, cum: &[f64]) -> Vec<(f64, f64, f64)> {
    let mut out: Vec<(f64, f64, f64)> = vec![];
    for i in 1..r.points.len().min(cum.len()) {
        let (a, b, v) = (cum[i - 1], cum[i], r.v[i - 1].min(r.v[i]) as f64);
        match out.last_mut() {
            Some(l) if l.2 == v => l.1 = b,
            _ if b > a => out.push((a, b, v)),
            _ => {}
        }
    }
    out
}

/// The same limits for the run from the other end.
pub fn reversed(iv: &[(f64, f64, f64)], len: f64) -> Vec<(f64, f64, f64)> {
    iv.iter().rev().map(|&(a, b, v)| (len - b, len - a, v)).collect()
}

fn push(out: &mut Vec<Phase>, t: f64, s: f64, v: f64, a: f64) {
    if let Some(l) = out.last() {
        if l.a == a && (l.v + l.a * (t - l.t) - v).abs() < 1e-6 {
            return;
        }
    }
    out.push(Phase { t, s, v, a });
}

fn accel(out: &mut Vec<Phase>, t: f64, x0: f64, x1: f64, u2: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let (v0, v1) = (u2.max(0.0).sqrt(), (u2 + 2.0 * BOAT_A * (x1 - x0)).sqrt());
    push(out, t, x0, v0, BOAT_A);
    t + (v1 - v0) / BOAT_A
}

fn brake(out: &mut Vec<Phase>, t: f64, x0: f64, x1: f64, w2: f64) -> f64 {
    if x1 <= x0 + 1e-12 {
        return t;
    }
    let (v0, v1) = ((w2 + 2.0 * BOAT_A * (x1 - x0)).sqrt(), w2.max(0.0).sqrt());
    push(out, t, x0, v0, -BOAT_A);
    t + (v0 - v1) / BOAT_A
}

/// From standstill to standstill through intervals `(x0, x1, v)`, accelerating and braking at
/// `BOAT_A`: `profile::segment` with one rate. Returns the end time.
fn segment(iv: &[(f64, f64, f64)], mut t: f64, out: &mut Vec<Phase>) -> f64 {
    let (m, a2) = (iv.len(), 2.0 * BOAT_A);
    if m == 0 {
        return t;
    }
    let mut fw = vec![0.0; m];
    let mut f = 0.0f64;
    for k in 0..m {
        let v2 = iv[k].2 * iv[k].2;
        fw[k] = f.min(v2);
        f = (fw[k] + a2 * (iv[k].1 - iv[k].0)).min(v2);
    }
    let mut bend = vec![0.0; m];
    let mut b = 0.0f64;
    for k in (0..m).rev() {
        let v2 = iv[k].2 * iv[k].2;
        bend[k] = b.min(v2);
        b = (bend[k] + a2 * (iv[k].1 - iv[k].0)).min(v2);
    }
    for k in 0..m {
        let (xa, xb, vl) = iv[k];
        let (v2, u2, w2, len) = (vl * vl, fw[k], bend[k], xb - xa);
        let s_fv = xa + (v2 - u2).max(0.0) / a2;
        let s_bv = xb - (v2 - w2).max(0.0) / a2;
        if s_fv <= s_bv {
            t = accel(out, t, xa, s_fv, u2);
            if s_bv > s_fv {
                push(out, t, s_fv, vl, 0.0);
                t += (s_bv - s_fv) / vl;
            }
            t = brake(out, t, s_bv, xb, w2);
        } else {
            let sx = if u2 >= w2 + a2 * len {
                xa
            } else if u2 + a2 * len <= w2 {
                xb
            } else {
                xa + ((w2 - u2 + a2 * len) / (2.0 * a2)).clamp(0.0, len)
            };
            t = accel(out, t, xa, sx, u2);
            t = brake(out, t, sx, xb, w2);
        }
    }
    t
}

/// The intervals of `iv` inside `[a, b]`.
fn clip(iv: &[(f64, f64, f64)], a: f64, b: f64) -> Vec<(f64, f64, f64)> {
    let lo = iv.partition_point(|x| x.1 <= a);
    iv[lo..].iter().take_while(|x| x.0 < b).map(|&(x0, x1, v)| (x0.max(a), x1.min(b), v)).filter(|x| x.1 > x.0).collect()
}

/// A boat's trip over stops at ground offsets `stop_s` (ascending) under limits `iv`, standing
/// `dwell[i]` at each stop between the ends.
pub fn profile(iv: &[(f64, f64, f64)], stop_s: &[f64], dwell: &[f64]) -> Profile {
    let n = stop_s.len();
    let mut p = Profile::default();
    if n < 2 {
        return p;
    }
    let mut t = 0.0;
    p.arr.push(0.0);
    p.dep.push(0.0);
    for i in 0..n - 1 {
        t = segment(&clip(iv, stop_s[i], stop_s[i + 1]), t, &mut p.phases);
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

/// A ferry line's stops as the service keeps them (`Plan`, as a bus line's): each port at its landing,
/// `s` its berth's ground metres along the route; `cum` the ground metres at each point.
pub fn plan(net: &Network, ports: &[u32], r: &FerryRoute) -> Plan {
    let cum = ground_cum(net, &r.points);
    let stops = r
        .port_points()
        .iter()
        .zip(ports)
        .map(|(&i, &p)| BusStop { id: p, s: cum[i], x: net.node_x[p as usize], y: net.node_y[p as usize], kind: KIND_STATION, station: Some(p), cover: None })
        .collect();
    Plan { stops, street: vec![], len: cum.last().copied().unwrap_or(0.0), cum }
}

#[cfg(test)]
mod tests;
