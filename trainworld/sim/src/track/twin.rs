//! T-3333 (notes/T-3333.md): a rail line run on both pairs of quad track. The player's switch is on
//! the line (`Twin::Parent`); the sim keeps a hidden second line (`Twin::Of`) on the stations beside
//! its stops, on the other pair, in step with it inside every edit (`TrackWorld::twin_ops`). Each runs
//! one pair's trains, so the line's stored counts, rates and headways are one pair's while it is
//! split; the app shows the two added up. Capacity, timetables, boarding, money and demand see two
//! ordinary lines.

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

/// Stations closer than this, in ground metres between their nodes, are not beside each other: half
/// quad's 8 m between pairs (`wasm::QUAD_SPACING`); none is that close on one track.
const BESIDE_MIN_M: f64 = 4.0;
/// Farther than this they are not: quad's pairs are 8 m apart, track drawn beside track 12 m
/// (`beside::BESIDE_SPACING`), with room for a twin placed a little off level.
const BESIDE_MAX_M: f64 = 16.0;
/// The most two stations' headings may differ by (either way along) to count as side by side.
const BESIDE_SIN: f64 = 0.1;

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

    /// The station beside station `n` on another track at the same spot: the nearest at its level,
    /// of its family of track (rail or tram), parallel to it, with its node 4 to 16 m from `n`'s.
    pub fn station_beside(&self, n: u32) -> Option<u32> {
        if !self.node_ok(n) || self.node_platform[n as usize] == 0 || self.node_berth[n as usize].is_some() {
            return None;
        }
        let i = n as usize;
        let (x, y, th) = (self.node_x[i], self.node_y[i], self.node_heading[i]);
        let k = self.k_at(y);
        let family = self.node_kind(n).spec().family;
        let mut best: Option<(u32, f64)> = None;
        for m in 0..self.node_count() as u32 {
            let j = m as usize;
            if m == n || !self.node_ok(m) || self.node_platform[j] == 0 || self.node_berth[j].is_some() || self.node_level[j] != self.node_level[i] {
                continue;
            }
            let d = (self.node_x[j] - x).hypot(self.node_y[j] - y) / k;
            if !(BESIDE_MIN_M..=BESIDE_MAX_M).contains(&d) || best.is_some_and(|b| b.1 <= d) {
                continue;
            }
            if (self.node_heading[j] - th).sin().abs() > BESIDE_SIN || self.node_kind(m).spec().family != family {
                continue;
            }
            best = Some((m, d));
        }
        best.map(|b| b.0)
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
        for (k, &s) in stops.iter().enumerate() {
            match self.station_beside(s) {
                Some(t) if !stops.contains(&t) => twin.push(t),
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
