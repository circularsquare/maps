//! T-3333 (notes/T-3333.md): a rail line run on both pairs of quad track. The player's switch is on
//! the line (`Twin::Parent`); the sim keeps a hidden second line (`Twin::Of`) on the stations beside
//! its stops, on the other pair, in step with it inside every edit (`TrackWorld::twin_ops`). Each runs
//! one pair's trains, so the line's stored counts, rates and headways are one pair's while it is
//! split; the app shows the two added up. Capacity, timetables, boarding, money and demand see two
//! ordinary lines.

use super::building::{members, platform_line, seg_gap};
use super::geom;
use super::net::{path_edge, LineData, Network};


#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Twin {
    #[default]
    None,
    /// The switch is on. `split`: the twin runs, so this line holds one pair's trains, rates and
    /// headways; otherwise the line runs alone on one pair with the whole line's.
    Parent { split: bool },
    /// The hidden twin of this line.
    Of(u32),
}

/// Why a line with the switch on cannot run on both pairs (`Network::twin_plan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// Not a rail line with two stops or more.
    NotRail,
    /// Stop `k` has no station beside it on another track.
    Stop(usize),
    /// No track joins the stations beside stops `k` and `k + 1` in order.
    Hop(usize),
    /// The stations beside the stops are joined, but trains would turn back elsewhere or find no
    /// way back.
    Route,
    /// The line is built and the track or stations beside it are not.
    Unbuilt,
}

impl Why {
    /// `[kind, index]` for the app: 1 a stop, 2 a hop, 3 the route, 4 unbuilt, 5 not rail.
    pub fn code(self) -> [f64; 2] {
        match self {
            Why::Stop(k) => [1.0, k as f64],
            Why::Hop(k) => [2.0, k as f64],
            Why::Route => [3.0, 0.0],
            Why::Unbuilt => [4.0, 0.0],
            Why::NotRail => [5.0, 0.0],
        }
    }
}

/// A station is on the track beside a stop's when its node is at least this far off the stop's track,
/// ground m: half quad's 8 m between pairs (`wasm::QUAD_SPACING`); a station on the same track is on it.
const BESIDE_MIN_M: f64 = 4.0;
/// And at most this far: quad's pairs are 8 m apart, track drawn beside track 12 m
/// (`beside::BESIDE_SPACING`).
const BESIDE_MAX_M: f64 = 16.0;
/// The most the station's heading and the stop's track there may differ by (either way along).
const BESIDE_SIN: f64 = 0.1;
/// How many pieces of the stop's track either way are looked along for a station beside it: a
/// complex's platforms reach a few hundred metres at most, and pieces end at joints and stations.
const BESIDE_HOPS: usize = 8;
/// Platforms (centre lines) this close are one complex, ground m (game/complexes.ts `PLATFORM_GAP_M`).
const PLATFORM_GAP_M: f64 = 50.0;
/// Stations whose centres are this close are one complex, ground m (game/complexes.ts `DEMAND_MERGE_M`).
const MERGE_M: f64 = 100.0;

type P = [f64; 2];

/// The shortest distance between two polylines, plane units.
fn polyline_gap(a: &[P], b: &[P]) -> f64 {
    let segs = |p: &[P]| -> Vec<(P, P)> { if p.len() < 2 { p.iter().map(|&q| (q, q)).collect() } else { p.windows(2).map(|w| (w[0], w[1])).collect() } };
    let (sa, sb) = (segs(a), segs(b));
    sa.iter().flat_map(|&(p0, p1)| sb.iter().map(move |&(q0, q1)| seg_gap(p0, p1, q0, q1))).fold(f64::INFINITY, f64::min)
}

impl Network {
    pub fn line_twin_of(&self, l: u32) -> Twin {
        self.line_twin.get(l as usize).copied().unwrap_or_default()
    }

    /// The twin line of parent `l`, if one is made.
    pub fn twin_line(&self, l: u32) -> Option<u32> {
        (0..self.line_count() as u32).find(|&t| self.line_ok(t) && self.line_twin_of(t) == Twin::Of(l))
    }

    /// Whether `l` is a hidden twin.
    pub fn is_twin(&self, l: u32) -> bool {
        matches!(self.line_twin_of(l), Twin::Of(_))
    }

    /// The station beside station `n` on another track (T-3356): of the stations in `n`'s complex,
    /// of its family of track (rail or tram), the nearest whose node lies on a track running beside
    /// `n`'s (4 to 16 m off it and parallel there), or with `n` on a track beside its own.
    pub fn station_beside(&self, n: u32) -> Option<u32> {
        if !self.node_ok(n) || self.node_platform[n as usize] == 0 || self.node_berth[n as usize].is_some() {
            return None;
        }
        let i = n as usize;
        let family = self.node_kind(n).spec().family;
        let mut best: Option<(u32, f64)> = None;
        for m in self.complex_of(n) {
            let j = m as usize;
            if m == n || self.node_berth[j].is_some() || self.node_kind(m).spec().family != family {
                continue;
            }
            let d = (self.node_x[j] - self.node_x[i]).hypot(self.node_y[j] - self.node_y[i]);
            if best.is_some_and(|b| b.1 <= d) || !(self.on_track_beside(n, m) || self.on_track_beside(m, n)) {
                continue;
            }
            best = Some((m, d));
        }
        best.map(|b| b.0)
    }

    /// Whether station `m`'s node lies on a track beside the track through station `n`: 4 to 16 m off
    /// it, parallel there, within `BESIDE_HOPS` pieces of `n` either way.
    fn on_track_beside(&self, n: u32, m: u32) -> bool {
        let j = m as usize;
        let (x, y, th) = (self.node_x[j], self.node_y[j], self.node_heading[j]);
        let k = self.k_at(y);
        for &c in self.node_ports[n as usize].ends() {
            let (mut e, mut from) = (c >> 1, n);
            for _ in 0..BESIDE_HOPS {
                let (s, dist) = self.project(e, x, y);
                let d = dist / k;
                if (BESIDE_MIN_M..=BESIDE_MAX_M).contains(&d) && (geom::pos_at(self.edge_pieces(e), s).2 - th).sin().abs() <= BESIDE_SIN {
                    return true;
                }
                // on through a plain piece end to the next piece
                let to = if self.edge_a[e as usize] == from { self.edge_b[e as usize] } else { self.edge_a[e as usize] };
                let ends = self.node_ports[to as usize].ends();
                if ends.len() != 2 {
                    break;
                }
                let Some(&next) = ends.iter().find(|&&c2| c2 >> 1 != e) else { break };
                (e, from) = (next >> 1, to);
            }
        }
        false
    }

    /// Whether two stations are one complex by distance, as game/complexes.ts `nearEnough` has it:
    /// centres within 100 m, or platforms (drawn centre lines, or the straight segment of the
    /// platform's length along the heading) within 50 m. Levels are ignored.
    fn near_enough(&self, a: u32, b: u32) -> bool {
        let (i, j) = (a as usize, b as usize);
        let k = (self.k_at(self.node_y[i]) + self.k_at(self.node_y[j])) / 2.0;
        let d = (self.node_x[i] - self.node_x[j]).hypot(self.node_y[i] - self.node_y[j]);
        if d <= MERGE_M * k {
            return true;
        }
        // a platform reaches at most its length from its node (more only on a sharp curve)
        let (pa, pb) = (self.node_platform[i] as f64, self.node_platform[j] as f64);
        if d > (1.5 * (pa + pb) + PLATFORM_GAP_M) * k {
            return false;
        }
        let straight = |n: usize, p: f64| -> Vec<P> {
            let (dx, dy) = (self.node_heading[n].cos() * p / 2.0 * k, self.node_heading[n].sin() * p / 2.0 * k);
            vec![[self.node_x[n] - dx, self.node_y[n] - dy], [self.node_x[n] + dx, self.node_y[n] + dy]]
        };
        polyline_gap(&straight(i, pa), &straight(j, pb)) <= PLATFORM_GAP_M * k
            || polyline_gap(&platform_line(self, a), &platform_line(self, b)) <= PLATFORM_GAP_M * k
    }

    /// The stations of station `n`'s complex, `n` first (game/complexes.ts, the rule the map's labels
    /// and demand's transfers use): linked by distance (`near_enough`), by the player's links (T-197)
    /// or by touching one station building (T-2217), directly or through others.
    pub fn complex_of(&self, n: u32) -> Vec<u32> {
        if !self.is_station(n) {
            return vec![];
        }
        let stations: Vec<u32> = (0..self.node_count() as u32).filter(|&m| self.is_station(m)).collect();
        // each building's corners' box, its members found only once a member of the complex is near it
        let halls: Vec<(&[P], [f64; 4])> = self
            .buildings
            .iter()
            .flatten()
            .filter(|b| b.corners.len() >= 3)
            .map(|b| {
                let c = &b.corners[..];
                (c, c.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |r, p| [r[0].min(p[0]), r[1].min(p[1]), r[2].max(p[0]), r[3].max(p[1])]))
            })
            .collect();
        let mut hall_members: Vec<Option<Vec<u32>>> = vec![None; halls.len()];
        let links: Vec<[u32; 2]> = self.links_live().collect();
        let mut out = vec![n];
        let mut k = 0;
        while k < out.len() {
            let a = out[k];
            k += 1;
            let add = |m: u32, out: &mut Vec<u32>| {
                if !out.contains(&m) {
                    out.push(m);
                }
            };
            for &[p, q] in &links {
                if p == a {
                    add(q, &mut out);
                } else if q == a {
                    add(p, &mut out);
                }
            }
            let (x, y) = (self.node_x[a as usize], self.node_y[a as usize]);
            let r = (self.node_platform[a as usize] as f64 * 1.5 + super::building::TOUCH_M) * self.k_at(y);
            for (h, &(c, b)) in halls.iter().enumerate() {
                if x < b[0] - r || x > b[2] + r || y < b[1] - r || y > b[3] + r {
                    continue;
                }
                let got = hall_members[h].get_or_insert_with(|| members(self, c));
                if got.contains(&a) {
                    for &m in got.iter() {
                        add(m, &mut out);
                    }
                }
            }
            for &m in &stations {
                if !out.contains(&m) && self.near_enough(a, m) {
                    out.push(m);
                }
            }
        }
        out
    }

    /// The twin's stops and path for line `l`: the stations beside its stops, joined in order the way
    /// any line's are (`route_hops`), turning back where it does, with a way back, and built where `l` is.
    pub fn twin_plan(&self, l: u32) -> Result<(Vec<u32>, Vec<u32>), Why> {
        let i = l as usize;
        if !self.line_ok(l) || self.line_bus[i].is_some() || self.line_ferry[i].is_some() {
            return Err(Why::NotRail);
        }
        let stops = self.line_stops(l);
        if stops.len() < 2 || self.line_path(l).is_empty() {
            return Err(Why::NotRail);
        }
        let mut twin = Vec::with_capacity(stops.len());
        // T-3332: a twin serves the same ways: a one-track station's is one-track the same way
        let ways = |n: u32| if !self.one_track(n) { 0 } else if self.one_side(n).is_some() { self.node_one[n as usize] } else { 3 };
        for (k, &s) in stops.iter().enumerate() {
            match self.station_beside(s) {
                Some(t) if !stops.contains(&t) && ways(t) == ways(s) => twin.push(t),
                _ => return Err(Why::Stop(k)),
            }
        }
        let path = self.route_hops(&twin).map_err(Why::Hop)?;
        let once = self.line_runs_once(l);
        if self.check_route(l, &twin, &path, once).is_err() {
            return Err(Why::Route);
        }
        if self.stop_reversals(&path).len() != self.stop_reversals(self.line_path(l)).len() {
            return Err(Why::Route);
        }
        let built = path.iter().all(|&p| self.edge_built[path_edge(p) as usize]) && twin.iter().all(|&s| self.node_built[s as usize]);
        if self.line_constructed(l) && !built {
            return Err(Why::Unbuilt);
        }
        Ok((twin, path))
    }

    /// Why line `l`, its switch on, runs on one pair now; None when it runs on both or the switch
    /// is off.
    pub fn twin_why(&self, l: u32) -> Option<Why> {
        match self.line_twin_of(l) {
            Twin::Parent { split: false } => self.twin_plan(l).err(),
            _ => None,
        }
    }
}

/// One pair's trains from the whole line's: half each, an odd count rounding up (as a loop's
/// two directions, T-798), a headway twice as long.
pub fn halve(d: &mut LineData) {
    d.trains = d.trains.map(|t| t.map(|x| x.div_ceil(2)));
    d.tph = d.tph.map(|t| t / 2.0);
    d.auto = d.auto.map(|a| a.map(|h| h * 2));
}

/// The whole line's trains from one pair's: `halve` undone (an odd count stays even).
pub fn double(d: &mut LineData) {
    d.trains = d.trains.map(|t| t.map(|x| x * 2));
    d.tph = d.tph.map(|t| t * 2.0);
    d.auto = d.auto.map(|a| a.map(|h| h / 2));
}

/// The twin line's data for `parent` (its data now) on `stops` and `path`, keeping `was` (the
/// twin as it stands) where the settle owns a figure: a headway line's counts, every rate.
pub fn twin_data(parent: u32, p: &LineData, stops: Vec<u32>, path: Vec<u32>, was: Option<&LineData>) -> LineData {
    let mut d = p.clone();
    d.stops = stops;
    d.path = path;
    d.group = 0;
    d.group_name = String::new();
    d.group_look = Default::default();
    d.hand = false;
    d.twin = Twin::Of(parent);
    if let Some(w) = was {
        d.tph = w.tph;
        if p.auto.is_some() && w.auto.is_some() {
            d.trains = w.trains;
        }
    }
    d
}
