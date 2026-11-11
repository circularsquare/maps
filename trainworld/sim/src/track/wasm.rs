//! The clock worker's API (T-025; notes/T-040.md has the buffer formats, notes/T-025.md the rest).
//!
//! The worker owns one `TrackApi`. Edits go in as calls that each build one op; every edit returns
//! 0 (applied) or the number of problems (nothing changed; `issue_kinds()` says what). After an
//! edit, `dirty_*` say what to refresh. Buffers come back as typed arrays (copies the worker can
//! transfer to the main thread).
//!
//! Blueprint and construct (T-055, SPEC 6.4): everything the player draws is a blueprint, free
//! and undoable. `construct` pays for blueprint edges and stations out of `cash` and makes them
//! real; it, removing constructed track or stations, and changing a constructed junction or
//! platform are final (they clear the undo history). Money is US$M.

use super::boarding::Boarding;
use super::bridge;
use super::cost::{BlockIndex, ClassGrid, GeoMask, MaskSet, NoWater, RunMask, TransformedMask, UnionMask, WaterMask, WorldWater};
use super::frame::FrameKind;
use super::geom::{self, design_radii, GeomIssue, Pi};
use super::net::{CrossKind, EdgeData, GroupLook, Issue, LineData, LoopKind, Network, NodeData, StationData};
use super::params::*;
use super::profile;
use super::roads::{self, RoadTile, Roads};
use super::save;
use super::streets::StreetGrid;
use super::world::{auto_trains, ChargeError, EditResult, Op, TrackWorld};
use std::f64::consts::PI;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// T-2871: the stored build days as the game days `edge_built_days` documents.
fn built_days(days: &[u32]) -> Vec<i32> {
    days.iter().map(|&d| match d {
        0 => -2,
        crate::track::net::DAY_UNKNOWN => -1,
        d => (d - 2).min(i32::MAX as u32) as i32,
    }).collect()
}

#[wasm_bindgen]
pub struct TrackApi {
    w: TrackWorld,
    last: Option<EditResult>,
    issues: Vec<Issue>,
    /// Set when the last refusal was for money: (need, have).
    money_short: Option<(f64, f64)>,
    /// Cash, US$M (SPEC 6.4: $10B to start).
    cash: f64,
    /// A sandbox game (T-728): nothing is refused for money; the cash still pays and may go negative.
    unlimited: bool,
    /// Cars the network owns (T-028). Running lines need `cars_needed`; an edit that needs more
    /// buys them, spare cars stay for later. Ordinary cars, then high-speed ones (T-817): spares of
    /// one kind never serve the other.
    fleet: [u32; FLEETS],
    /// What the last edit paid for trains, US$M.
    last_trains: f64,
    /// The last route the drawing tool asked about: (cost, length, issues), and its vertices.
    route_vertices: Vec<Pi>,
    /// The express pair of the last quad route asked about (T-255), else empty.
    route_express: Vec<Pi>,
    /// T-3092: the crossing legs of the flyovers the last route asked about lays, each J3 to J2.
    route_legs: Vec<Vec<Pi>>,
    /// T-3168: the last preview's crossing legs' vertex rows (`preview_leg_curves`).
    preview_leg_rows: Vec<f64>,
    /// T-3092: how many junctions of the last route asked about fell back to flat for want of room.
    flyover_fell_back: u32,
    /// T-656: the last route's alignment once bridged over water (what is built and drawn), else
    /// empty; `route_vertices` stays the shape the player drew.
    route_bridged: Vec<Pi>,
    /// T-656: the last route crossed water and had no room for a bridge's ramps.
    bridge_refused: bool,
    /// T-2414: what the refusal names, `BridgeNoRoom` or the end short of room.
    bridge_short: Issue,
    /// T-418: a route from a heading to a free end is one tangent arc (`set_route_tangent`).
    route_tangent: bool,
    /// T-817: the auto radius cap of the corners the drawing tool makes, ground m (`set_design_speed`).
    route_cap: f64,
    /// T-3169: the radius a real flyover's crossing leg aims for at every curve, ground m
    /// (`set_flyover_speed`).
    flyover_r: f64,
    /// T-817: double track the drawing tool lays is high-speed track (`set_route_hsr`).
    route_hsr: bool,
    /// T-2926: the kind of track the drawing tool lays when it is a tram kind (`set_route_kind`); Regular otherwise.
    route_kind: TrackKind,
    /// T-2927: the street line (plane) street tram track is drawn along (`set_route_street`); empty for none.
    route_street: Vec<(f64, f64)>,
    /// T-1615: routes ramp over or under track they would cross flat (`set_route_ramp`).
    route_ramp: bool,
    /// T-1685: the next route runs beside the run through this piece (`set_route_beside`).
    route_beside: Option<u32>,
    /// T-3137: the points the next route's ends join double track nearest (`set_route_join`), and
    /// while one is laid out (`route_op`) the ones it uses; NaN for the nearer track.
    route_join: Option<[f64; 4]>,
    join_at: [f64; 4],
    /// T-3137: the last route's joins at its start and end, and its pieces in the order drawn
    /// (`route_joins`).
    route_join_out: [u8; 2],
    route_pieces: Vec<u32>,
    route_ways: Vec<u8>,
    /// The route being laid out starts on the line beside track: its ends get no lead-ins.
    beside_on: bool,
    /// The way `route_op_as` tries: `Ramp::Flat`, or the preferred way or the other.
    ramp_try: Ramp,
    /// The route asked about crosses track at its own level (so a ramp was tried).
    ramp_found: bool,
    /// The ramp tried had no room.
    ramp_failed: bool,
    /// The last tangent route's circle: centre x, y and ground radius, else empty.
    route_arc: Vec<f64>,
    /// The end node of the last route asked about (new or existing).
    route_end: u32,
    /// Per rendered segment of the last route/edge preview, authoritative geometry speed km/h.
    preview_speeds: Vec<f32>,
    /// Per rendered segment of the last route/edge preview, its track as `Strokes.lane`: 0 single, 255 double (T-3134).
    preview_lanes: Vec<u8>,
    /// Per edge of the last node-move preview, as `preview_lanes`.
    preview_part_lanes: Vec<u8>,
    /// Where the last preview's track makes a crossing or a flat junction (T-629): x, y, kind (0
    /// crossing, 1 junction) per mark.
    preview_marks: Vec<f64>,
    /// The junctions the last route asked about makes: node, x, y per junction.
    route_junctions: Vec<(u32, [f64; 2])>,
    /// What the last station trial added to the blueprint, US$M (`station_cost`).
    station_cost: f64,
    /// The building the last `add_building` made (`u32::MAX` if none) and what the last add or
    /// reshape paid, US$M (T-2217).
    building_last: (u32, f64),
    /// The pack rasters installed with `add_pack_water`, kept so the water mask can be rebuilt
    /// when a loaded save is in another frame (T-125).
    rasters: Vec<PackWater>,
    /// The world's coarse water (`add_world_water`, T-1285), deciding where no pack raster does.
    world_water: Option<Arc<RunMask>>,
    /// Metres of the last preview by terrain class and over water (`preview_ground`, `ground_metres`),
    /// then the roads it crosses at ground level, weighted, and their price (T-1929).
    preview_ground: [f64; 7],
    /// The boarding account (T-164): demand's riders per period, and each train's load and each
    /// platform's crowd worked out from them. Never saved.
    boarding: Boarding,
    /// Per line id, its service span for demand and the key it was worked out for (`line_span`).
    spans: Vec<Option<(u64, Vec<f32>)>>,
    /// Game seconds now (`set_now`; NaN until the worker sets it: edits then re-space at once).
    now: f64,
    /// Game seconds after `now` the next edit takes effect (`set_edit_lead`, T-1071), and when the
    /// last one did (`edit_time`).
    lead: f64,
    edit_at: f64,
    /// Pieces the last batch level change left out (T-815).
    skipped: u32,
    /// What the last `link_stations` paid, US$M (T-1281).
    link_charge: f64,
    /// The node of the port the last `add_port` added (T-2193), `u32::MAX` if refused.
    last_port: u32,
    /// T-2669: per line id, its running cost over the world price list (its countries' prices,
    /// `set_line_run_factors`); 1 for a line not listed. Never saved: the app sends it.
    run_factor: Vec<f64>,
}

// T-2820: the last panic's message, as the demand module's (`demand::wasm`), for the clock worker's
// error. Not in a build with the demand module's own start function.
#[cfg(all(target_arch = "wasm32", not(feature = "demand")))]
thread_local! {
    static LAST_PANIC: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

#[cfg(all(target_arch = "wasm32", not(feature = "demand")))]
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        let text = info.to_string();
        LAST_PANIC.with(|p| if let Ok(mut p) = p.try_borrow_mut() { *p = text; });
    }));
}

/// T-2820: the last panic's message, empty if none. Borrows no exported object, so it can be called
/// after a panic has left a `TrackApi` unusable.
#[cfg(all(target_arch = "wasm32", not(feature = "demand")))]
#[wasm_bindgen]
pub fn last_panic() -> String {
    LAST_PANIC.with(|p| p.try_borrow().map(|p| p.clone()).unwrap_or_default())
}

/// T-1071: the end of a track save that carries the service's eras (`Service::encode_eras`):
/// the eras, their length as a u32, then this tag. The network's decoder stops before them.
/// `ERA2` (T-1925) eras carry departure phases; `ERA1`, older saves', do not.
const ERAS_TAG: &[u8; 4] = b"ERA2";
const ERAS_TAG_1: &[u8; 4] = b"ERA1";

/// A save's network bytes and its eras, if it has any, and whether they carry phases.
fn split_eras(bytes: &[u8]) -> (&[u8], Option<(&[u8], bool)>) {
    let n = bytes.len();
    if n < 8 || (&bytes[n - 4..] != ERAS_TAG && &bytes[n - 4..] != ERAS_TAG_1) {
        return (bytes, None);
    }
    let phases = &bytes[n - 4..] == ERAS_TAG;
    let len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    match (n - 8).checked_sub(len) {
        Some(start) => (&bytes[..start], Some((&bytes[start..n - 8], phases))),
        None => (bytes, None),
    }
}

/// A pack's water raster and the origin of the pack's own frame.
struct PackWater {
    mask: Arc<RunMask>,
    lon: f64,
    lat: f64,
    /// the raster seen from the network's frame and its bounds there, made again only when the
    /// frame changes (T-1039)
    seen: (Arc<dyn WaterMask>, [f64; 4]),
}

/// A pack raster seen from `frame`, and its bounds there (`[xmin, ymin, xmax, ymax]`).
fn pack_mask_in(frame: super::frame::Frame, mask: &Arc<RunMask>, lon: f64, lat: f64) -> (Arc<dyn WaterMask>, [f64; 4]) {
    if frame.kind == FrameKind::Equirect {
        let rad = PI / 180.0;
        let offset = [EARTH_R * (lon - frame.lon0) * rad * (frame.lat0 * rad).cos(), EARTH_R * (lat - frame.lat0) * rad];
        let scale = [(frame.lat0 * rad).cos() / (lat * rad).cos(), 1.0];
        let (x1, y1) = (mask.x0 + mask.w as f64 * mask.cell, mask.y0 + mask.h as f64 * mask.cell);
        let b = [offset[0] + scale[0] * mask.x0, offset[1] + scale[1] * mask.y0, offset[0] + scale[0] * x1, offset[1] + scale[1] * y1];
        (Arc::new(TransformedMask { mask: mask.clone(), offset, scale }), b)
    } else {
        let g = GeoMask::new(mask.clone(), frame, lon, lat);
        let b = g.bounds;
        (Arc::new(g), b)
    }
}

/// Cars the running lines need: each line's busiest schedule's trains times its cars (T-028).
/// Ordinary cars, then high-speed ones (T-817), then buses (T-1464), then boats (T-2193), then trams (T-2926).
fn cars_needed(w: &TrackWorld) -> [u32; FLEETS] {
    let mut out = [0; FLEETS];
    for (i, l) in w.svc.lines.iter().enumerate() {
        if l.running {
            out[fleet_kind(&w.net, i)] += l.trains.iter().copied().max().unwrap_or(0) * l.cars;
        }
    }
    out
}

/// The fleets: ordinary cars, high-speed cars, buses, boats, trams.
const FLEETS: usize = 5;

/// Which fleet a line's trains come from: 0 ordinary cars, 1 high-speed cars, 2 buses, 3 boats, 4 trams.
fn fleet_kind(n: &Network, l: usize) -> usize {
    if n.line_bus[l].is_some() { 2 } else if n.line_ferry[l].is_some() { 3 } else if n.line_tram(l as u32) { 4 } else { n.line_fast[l] as usize }
}

/// Price of a car of each kind (a bus or a boat counting as one), US$M.
const CAR_PRICES: [f64; FLEETS] = [CAR_PRICE, HS_CAR_PRICE, BUS_PRICE, BOAT_PRICE, TRAM_CAR_PRICE];

/// A constructed edge's upkeep a game day, US$M; None for blueprint, and for crossovers, which cost
/// nothing on top of their track (T-867).
fn edge_upkeep(n: &Network, e: usize) -> Option<f64> {
    (n.edge_alive[e] && n.edge_built[e] && !n.edge_crossover[e])
        .then(|| n.edge_len[e] / 1000.0 * edge_upkeep_km(n, e) * super::cost::track_factor(n.edge_tracks[e], n.edge_kind[e]) * ECONOMY / 1e6)
}
/// An edge's upkeep rate a route-km, real US$ (T-1931): the dearest structure between its lowest and
/// highest level. The app reads the same two levels from `edges_info`, so its figures match
/// (`edgeUpkeep`, game/regionMoney.ts).
fn edge_upkeep_km(n: &Network, e: usize) -> f64 {
    let level = |z: f64| (z / LEVEL_H).round() as i32;
    let zs = n.edge_vert(e as u32).iter().map(|v| level(v[1]));
    let (lo, hi) = zs.fold((i32::MAX, i32::MIN), |(a, b), l| (a.min(l), b.max(l)));
    super::cost::track_upkeep_km(lo, hi)
}
fn upkept_station(n: &Network, i: usize) -> bool {
    n.node_alive[i] && n.node_platform[i] > 0 && n.node_built[i] && n.node_berth[i].is_none()
}
/// A constructed port (T-2193), upkept at `UPKEEP_PORT`.
fn upkept_port(n: &Network, i: usize) -> bool {
    n.node_alive[i] && n.node_platform[i] > 0 && n.node_built[i] && n.node_berth[i].is_some()
}
fn port_upkeep() -> f64 {
    UPKEEP_PORT * ECONOMY / 1e6
}
/// A constructed bus route's upkeep a game day, US$M; None for a planned one (T-1464).
fn bus_upkeep(n: &Network, l: u32) -> Option<f64> {
    n.line_bus.get(l as usize)?.as_ref().filter(|r| r.built && n.line_ok(l)).map(|_| n.bus_road_m(l) / 1000.0 * UPKEEP_BUS_KM * ECONOMY / 1e6)
}
fn station_upkeep(n: &Network, i: usize) -> f64 {
    n.node_kind(i as u32).spec().station_upkeep.unwrap_or_else(|| super::cost::station_upkeep(n.node_level[i])) * ECONOMY / 1e6
}

fn pis_from(flat: &[f64]) -> Vec<Pi> {
    flat.chunks_exact(4).map(|c| Pi::new(c[0], c[1], c[2], c[3] as i8).quantised()).collect()
}

/// Where a drawn route starts or ends.
#[derive(Clone, Copy, Debug)]
enum End {
    /// A new node here, at this level.
    Free { x: f64, y: f64, level: i8 },
    /// An existing node (a track end, a junction, a station).
    Node(u32),
    /// A new node splitting this edge near (x, y).
    Edge { e: u32, x: f64, y: f64 },
}

fn end_from(c: &[f64]) -> End {
    match c[0] as i32 {
        1 => End::Node(c[1] as u32),
        2 => End::Edge { e: c[1] as u32, x: c[2], y: c[3] },
        _ => End::Free { x: quant(c[2]), y: quant(c[3]), level: c[4] as i8 },
    }
}

/// A resolved end: the node (new or existing), where it is, its level, and the directions a new
/// edge may leave it in (empty = any).
struct Resolved {
    node: u32,
    x: f64,
    y: f64,
    level: i8,
    dirs: Vec<f64>,
    /// The route turns it into a junction (it had two edge ends, or it splits track).
    new_junction: bool,
    /// Double track meets it there (T-3137: single track drawn from it joins one of the two).
    double: bool,
    /// Ops creating it (a split or a new node), applied before the edge.
    ops: Vec<Op>,
}

/// A reshape of a blueprint edge (`TrackApi::reshaped`): the edit, the edge's vertices after it,
/// and the neighbouring pieces whose joint moves with it (T-244), in the stroke's order: `before`
/// ends at the edge's a node, `after` starts at its b node (empty where none moves).
struct Reshape {
    op: Op,
    verts: Vec<Pi>,
    tracks: u8,
    kind: TrackKind,
    before: Vec<Pi>,
    after: Vec<Pi>,
}

#[wasm_bindgen]
impl TrackApi {
    /// An empty network for a city: its origin and running side (right in the US).
    #[wasm_bindgen(constructor)]
    pub fn new(origin_lon: f64, origin_lat: f64, right_hand: bool) -> TrackApi {
        TrackApi {
            w: TrackWorld::new(Network::new(origin_lon, origin_lat, right_hand)),
            last: None,
            issues: vec![],
            money_short: None,
            cash: 6000.0,
            unlimited: false,
            fleet: [0; FLEETS],
            last_trains: 0.0,
            route_vertices: vec![],
            route_express: vec![],
            route_legs: vec![],
            preview_leg_rows: vec![],
            flyover_fell_back: 0,
            route_bridged: vec![],
            bridge_refused: false,
            bridge_short: Issue::BridgeNoRoom,
            route_tangent: false,
            route_cap: R_CAP,
            flyover_r: design_radius(FLYOVER_KMH),
            route_hsr: false,
            route_kind: TrackKind::Regular,
            route_street: vec![],
            route_ramp: false,
            route_beside: None,
            route_join: None,
            join_at: [f64::NAN; 4],
            route_join_out: [0; 2],
            route_pieces: vec![],
            route_ways: vec![],
            beside_on: false,
            ramp_try: Ramp::Flat,
            ramp_found: false,
            ramp_failed: false,
            route_arc: vec![],
            route_end: u32::MAX,
            preview_speeds: vec![],
            preview_lanes: vec![],
            preview_part_lanes: vec![],
            preview_marks: vec![],
            route_junctions: vec![],
            station_cost: 0.0,
            building_last: (u32::MAX, 0.0),
            rasters: vec![],
            world_water: None,
            preview_ground: [0.0; 7],
            boarding: Boarding::default(),
            spans: vec![],
            now: f64::NAN,
            lead: 0.0,
            edit_at: f64::NAN,
            skipped: 0,
            link_charge: 0.0,
            last_port: u32::MAX,
            run_factor: vec![],
        }
    }

    /// Demand's riders for the boarding account (T-164), as `Boarding::set_riders` reads them:
    /// 5 busiest-hour factors, then per line its id, stop count `n`, and per period boardings
    /// then alightings at each of its `2n` route nodes. One kind of day's (T-260, T-262: 0
    /// weekday, 1 Saturday, 2 Sunday; each hour reads its own day's, `params::day_kind_at`).
    pub fn set_riders(&mut self, packed: Vec<f32>, day: u32) {
        self.boarding.set_riders_day(&packed, day as usize);
    }
    /// The riders coming to each platform (T-1255, `Boarding::set_arrivals_day`), after
    /// `set_riders` for the same kind of day: per line its id, stop count `n`, and per period the
    /// riders at each of its `2n` route nodes.
    pub fn set_arrivals(&mut self, packed: Vec<f32>, day: u32) {
        self.boarding.set_arrivals_day(&packed, day as usize);
    }
    /// Demand's transfers between platforms (T-554, `Boarding::set_transfers`): per pair the
    /// line, run and stop riders leave, the line, run and stop they board, and riders a day. Only
    /// the timing of drawn platform crowds reads them.
    pub fn set_transfers(&mut self, packed: Vec<f32>) {
        self.boarding.set_transfers(&packed);
    }
    /// Each line's days that run as the other kind (T-872, T-262, `Boarding::set_holidays`): per
    /// line its id, a count, then that many pairs of a local game day and its kind. Replaces the
    /// last list.
    pub fn set_line_holidays(&mut self, packed: Vec<f64>) {
        self.boarding.set_holidays(&packed);
    }
    /// A line's day in the boarding account for its panel (T-230, `Boarding::line_summary`), on
    /// a kind of day (T-260); empty before riders.
    pub fn line_account(&self, line: u32, day: u32) -> Vec<f64> {
        self.boarding.line_summary_day(&self.w.svc, line as usize, day as usize)
    }
    /// A line's day by period (T-477, T-491, `Boarding::line_periods`), on a kind of day.
    pub fn line_periods(&self, line: u32, day: u32) -> Vec<f64> {
        self.boarding.line_periods(&self.w.svc, line as usize, day as usize)
    }
    /// Every line's day so far at game time `t` (T-293, `Boarding::today` has the layout).
    pub fn today(&self, t: f64) -> Vec<f64> {
        self.boarding.today(&self.w.svc, t)
    }
    /// Every line's past 24 hours at game time `t` (T-446, `Boarding::past_day`).
    pub fn past_day(&self, t: f64) -> Vec<f64> {
        self.boarding.past_day(&self.w.svc, t)
    }
    /// Forget the riders (a game was loaded: its demand comes later).
    pub fn clear_riders(&mut self) {
        self.boarding.clear();
    }
    pub fn has_riders(&self) -> bool {
        self.boarding.has_riders()
    }
    /// The hour `[epoch, epoch + 3600)` of the boarding account, lined up with `trips(epoch)`
    /// (`Boarding::hour` has the layout). Brings the account up to date first.
    pub fn boarding_hour(&mut self, epoch: f64) -> Vec<f32> {
        self.boarding.hour(&self.w.net, &self.w.svc, epoch)
    }
    /// Work out at most `max_lines` out-of-date lines of the account; true if more remain. The
    /// clock worker steps through new riders this way between messages (T-164).
    pub fn boarding_step(&mut self, max_lines: u32) -> bool {
        self.boarding.ensure_some(&self.w.net, &self.w.svc, max_lines as usize)
    }
    /// Lines whose account was worked out again since the last call.
    pub fn boarding_redone(&mut self) -> u32 {
        self.boarding.take_redone() as u32
    }

    /// Make the network empty and put it in a frame (`frame_kind` numbers: 0 legacy, 1 conic, 2
    /// Mercator, `frame.rs`). False, changing nothing, if anything is drawn or the kind is not
    /// known: a network's metres cannot be
    /// reinterpreted, only converted (`convert_save_frame`).
    pub fn set_empty_frame(&mut self, kind: u8) -> bool {
        let n = &self.w.net;
        if (0..n.node_count() as u32).any(|i| n.node_ok(i)) || (0..n.line_count() as u32).any(|l| n.line_ok(l)) || n.buildings.iter().any(Option::is_some) {
            return false;
        }
        let Some(kind) = FrameKind::from_u8(kind) else { return false };
        let mut net = Network::new_in(kind, n.origin_lon, n.origin_lat, n.right_hand);
        std::mem::swap(&mut net.water_mask, &mut self.w.net.water_mask);
        net.time_zones = self.w.net.time_zones.clone();
        net.terrain = self.w.net.terrain.clone();
        net.roads = self.w.net.roads.clone();
        net.streets = self.w.net.streets.clone();
        self.w = TrackWorld::new(net);
        self.rebuild_water();
        true
    }

    /// Each region's local time (T-309, `localtime.rs`): the game clock's UTC offset (h), the
    /// reach of a zone's centre (m), then per registered zone its centre's lon, lat and UTC offset
    /// (h). Kept across loads; every line's periods follow it from now.
    pub fn set_time_zones(&mut self, table: &[f64]) {
        self.w.net.time_zones = super::localtime::TimeZones::from_table(table);
        self.w.svc.reshift(&self.w.net);
    }
    /// Seconds a line's local time runs ahead of the game clock (T-309); 0 for no such line.
    pub fn line_shift(&self, line: u32) -> f64 {
        self.w.svc.lines.get(line as usize).filter(|s| s.ok).map_or(0.0, |s| s.shift)
    }
    /// The local times the running lines run on, seconds ahead of the game clock, each once,
    /// ascending (T-309: money settles each one's hours by its own periods).
    pub fn running_shifts(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.w.svc.lines.iter().filter(|s| s.running).map(|s| s.shift).collect();
        v.sort_by(f64::total_cmp);
        v.dedup();
        v
    }

    /// The frame of the loaded network: 0 legacy equirectangular, 1 the Northeast conic (T-125),
    /// 2 the Mercator world (T-306).
    pub fn frame_kind(&self) -> u8 {
        self.w.net.frame_kind as u8
    }
    /// The frame's origin: [lon, lat] in degrees.
    pub fn frame_origin(&self) -> Vec<f64> {
        vec![self.w.net.origin_lon, self.w.net.origin_lat]
    }


    /// Reset coverage before installing the region's pack rasters (the world's coarse water stays).
    pub fn clear_water(&mut self) {
        self.rasters.clear();
        let set = self.mask_set();
        self.w.net.set_water_mask(Box::new(set));
        self.w.reload();
    }

    /// Append a pack raster transformed into the fixed network frame by an offset and scale
    /// (the legacy frame only; `add_pack_water` serves every frame).
    pub fn append_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16], offset_x: f64, offset_y: f64, scale_x: f64, scale_y: f64) {
        assert!(scale_x.is_finite() && scale_x > 0.0 && scale_y.is_finite() && scale_y > 0.0);
        let m = TransformedMask {
            mask: Arc::new(RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() }),
            offset: [offset_x, offset_y], scale: [scale_x, scale_y],
        };
        let previous = std::mem::replace(&mut self.w.net.water_mask, Box::new(NoWater));
        self.w.net.set_water_mask(Box::new(UnionMask(previous, Box::new(m))));
        self.w.reload();
    }

    /// Add a pack's water raster (T-125). `pack_lon`, `pack_lat` are the origin of the pack's own
    /// equirectangular frame, in which the raster's grid is laid out. The mask follows the loaded
    /// network's frame and is rebuilt when a save in another frame is loaded.
    pub fn add_pack_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16], pack_lon: f64, pack_lat: f64) {
        let mask = Arc::new(RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() });
        let seen = pack_mask_in(self.w.net.frame(), &mask, pack_lon, pack_lat);
        let b = seen.1;
        self.rasters.push(PackWater { mask, lon: pack_lon, lat: pack_lat, seen });
        let set = self.mask_set();
        self.w.net.add_water(Box::new(set), b);
    }

    /// `index.bin` of the terrain blocks (T-2901, `cost::BlockIndex`): which blocks have a file. False
    /// if the bytes are not one.
    pub fn set_terrain_index(&mut self, bytes: &[u8]) -> bool {
        let Some(i) = BlockIndex::decode(bytes) else { return false };
        self.w.net.terrain.set_index(i);
        true
    }

    /// `[west, south, east, north]` degrees of a block in the network frame, with a margin the sampled
    /// curve never leaves.
    fn terrain_box(&self, [w, s, e, n]: [f64; 4]) -> [f64; 4] {
        let frame = self.w.net.frame();
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        const STEPS: usize = 32;
        for i in 0..=STEPS {
            let t = i as f64 / STEPS as f64;
            let (lon, lat) = (w + (e - w) * t, s + (n - s) * t);
            for (lo, la) in [(lon, s), (lon, n), (w, lat), (e, lat)] {
                let (x, y) = frame.from_lonlat(lo, la);
                b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
            }
        }
        // the boundary's sampled curve bows inward by far less than this between samples
        let m = 0.01 * (b[2] - b[0]).max(b[3] - b[1]) + 1000.0;
        [b[0] - m, b[1] - m, b[2] + m, b[3] + m]
    }

    /// Add a terrain block (T-1270, T-2901: a `.terrain` file's bytes, `cost::ClassGrid`, inside one
    /// degree block): the blueprint and built track it reaches is priced again. False if the bytes are
    /// not one.
    pub fn add_terrain(&mut self, bytes: &[u8]) -> bool {
        let Some(g) = ClassGrid::decode(bytes) else { return false };
        let [w, s, e, n] = g.bounds();
        if e > w.floor() + 1.0 + 1e-9 || n > s.floor() + 1.0 + 1e-9 {
            return false;
        }
        let b = self.terrain_box([w, s, e, n]);
        self.w.net.add_terrain(Arc::new(g), b);
        true
    }

    /// Take a terrain block out of the model, to free its memory (T-2901). False if an edge reaches
    /// into it: it stays, so the edge prices the same at its next re-derive.
    pub fn drop_terrain(&mut self, key: i32) -> bool {
        let (la, lo) = (key.div_euclid(360) - 90, key.rem_euclid(360) - 180);
        let b = self.terrain_box([lo as f64, la as f64, lo as f64 + 1.0, la as f64 + 1.0]);
        self.w.net.drop_terrain(key, b)
    }

    /// The blocks asked for by a price since the last call and not installed, by `cost::block_key`;
    /// nothing until the index is in.
    pub fn take_terrain_wanted(&self) -> Vec<i32> {
        self.w.net.terrain.take_wanted()
    }

    /// The blocks with a file under the live edges, only the blueprint's with `blueprint` (T-2901: what
    /// Construct must have).
    pub fn terrain_blocks_under(&self, blueprint: bool) -> Vec<i32> {
        self.w.net.terrain_blocks_under(blueprint)
    }

    /// Add a pack's street speeds (T-1577, `streets::StreetGrid::from_zones`): a bus runs at its
    /// share of the car's street speed where zones say so, and the bus lines are timed again. The
    /// arrays are the pack's (`pop`, `jobs`, `zone` per cell; `zone_gx`, `zone_gy`, `zone_size`
    /// (or none), `zone_cells9` per zone), its grid from `gravity` in the header, and its origin.
    /// False if they do not fit each other.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pack_streets(&mut self, zone_m: f64, x0: f64, y0: f64, w: u32, h: u32, pop: &[f32], jobs: &[f32], zone: &[u32], gx: &[u16], gy: &[u16], size: &[u8], cells9: &[u32], pack_lon: f64, pack_lat: f64) -> bool {
        let Some(g) = StreetGrid::from_zones(zone_m, x0, y0, w, h, pop, jobs, zone, gx, gy, size, cells9, pack_lon, pack_lat) else { return false };
        self.w.net.streets.grids.push(Arc::new(g));
        self.w.rework_buses();
        true
    }

    /// Metres of the last route or edge preview in each terrain class where it is charged (flat,
    /// rolling, hilly, mountain; T-1270) and over water (T-1887), then the roads it crosses at
    /// ground level, weighted, and their price, US$M (T-1929), for the track tool's tag.
    pub fn preview_ground(&self) -> Vec<f64> {
        self.preview_ground.to_vec()
    }

    /// Add a zoom 14 road tile (T-1929, `roads::RoadTile::new`: `pts` x, y pairs in units of
    /// `extent` across the tile, line `i` from point `starts[i]`, weighted `weights[i]`): the
    /// blueprint and built track it reaches is priced again. A tile already in is kept as it is.
    pub fn add_road_tile(&mut self, tx: u32, ty: u32, extent: f64, pts: &[i32], starts: &[u32], weights: &[u8]) -> bool {
        let key = Roads::key(tx, ty);
        if self.w.net.roads.tiles.contains_key(&key) || tx >= roads::TILES || ty >= roads::TILES {
            return false;
        }
        let tile = Arc::new(RoadTile::new(extent, pts, starts, weights));
        let [w, s, e, n] = roads::tile_bounds(tx, ty);
        let frame = self.w.net.frame();
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for (lon, lat) in [(w, s), (w, n), (e, s), (e, n), (0.5 * (w + e), s), (0.5 * (w + e), n)] {
            let (x, y) = frame.from_lonlat(lon, lat);
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
        let m = 100.0;
        self.w.net.add_roads(key, tile, [b[0] - m, b[1] - m, b[2] + m, b[3] + m]);
        true
    }

    /// Install the world's coarse water mask (T-1285, a `world.water` file's bytes,
    /// `cost::WorldWater`): every edge's water and price is worked out again. False if the bytes are
    /// not one.
    pub fn add_world_water(&mut self, bytes: &[u8]) -> bool {
        let Some(m) = WorldWater::decode(bytes) else { return false };
        self.world_water = Some(Arc::new(m));
        let set = self.mask_set();
        self.w.net.add_water(Box::new(set), [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]);
        true
    }

    fn mask_set(&self) -> MaskSet {
        self.mask_set_in(self.w.net.frame())
    }
    fn mask_set_in(&self, frame: super::frame::Frame) -> MaskSet {
        MaskSet {
            rasters: self.rasters.iter().map(|r| r.seen.clone()).collect(),
            world: self.world_water.as_ref().map(|m| Arc::new(WorldWater { mask: m.clone(), frame }) as Arc<dyn WaterMask>),
        }
    }

    /// Replace the network with a saved one. False if the bytes are not a save.
    pub fn load(&mut self, bytes: &[u8]) -> bool {
        let (bytes, eras) = split_eras(bytes);
        match save::decode(bytes) {
            Ok(mut net) => {
                // saves made from a real network before T-1442 carry its untidy names
                super::names::tidy_network(&mut net);
                // T-3234: the masks in the save's frame, set before its one derive (set after it,
                // they would derive and settle the whole network again)
                let water = !self.rasters.is_empty() || self.world_water.is_some();
                if water {
                    let frame = net.frame();
                    for r in &mut self.rasters {
                        r.seen = pack_mask_in(frame, &r.mask, r.lon, r.lat);
                    }
                    net.set_water_mask(Box::new(self.mask_set_in(frame)));
                } else {
                    std::mem::swap(&mut net.water_mask, &mut self.w.net.water_mask);
                }
                net.time_zones = self.w.net.time_zones.clone();
                net.terrain = self.w.net.terrain.clone();
                net.roads = self.w.net.roads.clone();
                net.streets = self.w.net.streets.clone();
                self.w = TrackWorld::new(net);
                if stop_every_pass(&mut self.w.net) {
                    self.w.reload();
                }
                // T-2669: the last game's lines' prices are not this one's; the app sends them again
                self.run_factor.clear();
                // lines saved at a frequency, not trains, are given trains (T-1146), and derived
                // again with them
                if self.w.allocate_unallocated() && water {
                    self.w.reload();
                }
                // a save made while trains changed over loads with them where they were (T-1071)
                if let Some((e, phases)) = eras {
                    if self.w.svc.decode_eras(e, phases).is_err() {
                        self.w.svc.eras.clear();
                        self.w.svc.shifts.clear();
                    }
                }
                true
            }
            Err(_) => false,
        }
    }

    /// The save's bytes in another frame (`frame_kind` numbers), see `save::convert_frame`. Empty if
    /// the bytes are not a save.
    pub fn convert_save_frame(bytes: &[u8], kind: u8) -> Vec<u8> {
        let Some(to) = FrameKind::from_u8(kind) else { return vec![] };
        save::convert_frame(bytes, to).unwrap_or_default()
    }
    /// T-1826: how much of a save's network bytes this build does not read (`save::decode_read`,
    /// with a trailing eras tail it does not know counted too); 0 if it reads them all or they do
    /// not decode (the load says so).
    pub fn save_unread(bytes: &[u8]) -> u32 {
        let (bytes, _) = split_eras(bytes);
        save::decode_read(bytes).map_or(0, |(_, n)| n as u32)
    }
    /// T-1826: the save level this build writes and reads up to (`save::LEVEL`).
    pub fn save_level() -> u32 {
        save::LEVEL
    }
    /// A save's node and curve points, x, y flat, read without deriving it (T-737: the clock worker
    /// fetches the water masks they fall in before the one real load). Empty if it does not decode.
    pub fn save_points(bytes: &[u8]) -> Vec<f64> {
        let Ok(net) = save::decode(bytes) else { return vec![] };
        let mut out = Vec::with_capacity(2 * (net.node_x.len() + net.pis.data.len()));
        for (i, &alive) in net.node_alive.iter().enumerate() {
            if alive {
                out.extend([net.node_x[i], net.node_y[i]]);
            }
        }
        for (e, &alive) in net.edge_alive.iter().enumerate() {
            if alive {
                for p in net.pis.get(net.edge_pis[e]) {
                    out.extend([p.x, p.y]);
                }
            }
        }
        out
    }
    /// T-889: a save's built stations as `nodes_info` lists them, read without deriving it: the
    /// frame (kind, origin lon, lat), then x, y per station in node order. The demand pool opens the
    /// zones they reach while the clock worker works the network out. Empty if it does not decode.
    pub fn save_stations(bytes: &[u8]) -> Vec<f64> {
        let Ok(net) = save::decode(bytes) else { return vec![] };
        let mut out = vec![net.frame_kind as u8 as f64, net.origin_lon, net.origin_lat];
        for k in 0..net.node_count() as u32 {
            let i = k as usize;
            if net.node_ok(k) && net.node_platform[i] > 0 && net.node_built[i] {
                out.extend([net.node_x[i], net.node_y[i]]);
            }
        }
        out
    }
    /// T-2871: the game day, set by the clock worker before each edit: what is constructed now is
    /// dated with it.
    pub fn set_game_day(&mut self, day: u32) {
        self.w.net.today = day.min(1 << 30) + 2;
    }
    /// T-2871: per edge id, the game day it was first constructed; -1 built before recording
    /// began, -2 not constructed. `station_built_days` per node id and `line_built_days` per
    /// line id likewise (a line: the day it was added, a bus or ferry line its route constructed).
    pub fn edge_built_days(&self) -> Vec<i32> {
        built_days(&self.w.net.edge_day)
    }
    pub fn station_built_days(&self) -> Vec<i32> {
        built_days(&self.w.net.node_day)
    }
    pub fn line_built_days(&self) -> Vec<i32> {
        built_days(&self.w.net.line_day)
    }
    pub fn save(&self) -> Vec<u8> {
        let mut out = save::encode(&self.w.net);
        let eras = self.w.svc.encode_eras(&self.w.net);
        if !eras.is_empty() {
            let n = eras.len() as u32;
            out.extend(eras);
            out.extend(n.to_le_bytes());
            out.extend_from_slice(ERAS_TAG);
        }
        out
    }
}

// The rest of `TrackApi`'s exports, by area.
mod money;
mod edits;
mod draw;
mod flyover;
use draw::Ramp;
mod beside;
mod crossings;
mod links;
mod oneway;
mod rebuild;
mod lines;
mod gap;
mod snapshot;
mod service;
mod upgrades;
mod related;
mod bus;
mod ferry;
mod buildings;

/// What buying the cars the running lines need beyond `fleet` costs, US$M.
fn trains_to_buy(w: &TrackWorld, fleet: [u32; FLEETS]) -> f64 {
    let need = cars_needed(w);
    (0..FLEETS).map(|k| need[k].saturating_sub(fleet[k]) as f64 * CAR_PRICES[k]).sum()
}

/// An edge's vertices: its a node, its PIs, its b node; `moved` puts one node somewhere else.
fn vertices_of(net: &Network, d: &EdgeData, moved: &[(u32, f64, f64)]) -> Vec<Pi> {
    let at = |n: u32| -> Pi {
        let i = n as usize;
        match moved.iter().find(|m| m.0 == n) {
            Some(&(_, x, y)) => Pi::new(x, y, 0.0, net.node_level[i]),
            None => Pi::new(net.node_x[i], net.node_y[i], 0.0, net.node_level[i]),
        }
    };
    let mut v = vec![at(d.a)];
    v.extend_from_slice(&d.pis);
    v.push(at(d.b));
    v
}

/// A radius the edit itself sets (`set(pi index, pi)`) that its legs have no room for is stored
/// as the widest that fits (T-1895), as the fit draws it. Any other set radius is kept as it is: a
/// most, narrowed by the fit while the legs are short and wide again once they have room (T-1898;
/// going back to auto capped it at 160 km/h).
fn narrow_set_radii(net: &Network, d: &mut EdgeData, moved: &[(u32, f64, f64)], set: &dyn Fn(usize, &Pi) -> bool) {
    let verts = vertices_of(net, d, moved);
    let f = net.fit(&verts);
    for &j in &f.clamped {
        let p = &mut d.pis[j - 1];
        if set(j - 1, p) {
            p.radius = ((f.radius[j] / QUANT) * (1.0 - 1e-9)).floor() * QUANT;
        }
    }
}

/// T-1898: the auto corners of `d` at PI indices `made` (ones an edit placed or moved) laid out for
/// the design radius `cap`, as drawing lays its own (`design_radii`): wider than `R_CAP` where they
/// have room, set to it.
fn design_corners(net: &Network, d: &mut EdgeData, moved: &[(u32, f64, f64)], made: &[usize], cap: f64) {
    // T-3121: under 160 too, never under the kind's smallest curve
    let cap = cap.max(d.kind.spec().min_radius);
    if cap == R_CAP || made.is_empty() {
        return;
    }
    let mut verts = vertices_of(net, d, moved);
    let at: Vec<usize> = made.iter().map(|k| k + 1).collect();
    design_radii(&mut verts, &at, cap, net.frame().scaled());
    for &k in made {
        d.pis[k].radius = verts[k + 1].radius;
    }
}

/// Per vertex of a fitted alignment: x, y (the middle of its curve, or the vertex where it has
/// none), radius used (0 = no curve), speed limit km/h.
fn vertex_rows(f: &geom::Fit, verts: &[Pi], out: &mut Vec<f64>, cap: f64) {
    for (i, v) in verts.iter().enumerate() {
        let rad = f.radius.get(i).copied().unwrap_or(0.0);
        let (x, y) = if rad > 0.0 && !f.pieces.is_empty() { let (x, y, _) = geom::pos_at(&f.pieces, f.vert_s[i]); (x, y) } else { (v.x, v.y) };
        out.extend([x, y, rad, if rad > 0.0 { (A_LAT * rad).sqrt().min(cap) * 3.6 } else { cap * 3.6 }]);
    }
}

/// Hops already searched while trying stops (T-237): `(from, to, side arrived on at from, 2 for
/// none)` to `Network::find_path`'s answer. Valid for one state of the network.
type HopMemo = std::collections::HashMap<(u32, u32, u8, u8), Option<(Vec<u32>, u8)>>;

/// `Network::route`, with each hop's search kept in `memo`: the same answer, but trying a stop in
/// every slot searches each hop once. A loop's closing hop takes the side that carries on round
/// (T-798) when that is worth it (`Network::carry_worth`), as `route` does.
fn route_memo(net: &Network, stops: &[u32], memo: &mut HopMemo) -> Option<Vec<u32>> {
    if stops.iter().any(|&s| !net.node_ok(s) || net.node_platform[s as usize] == 0) {
        return None;
    }
    let mut path = vec![];
    let mut side = None;
    let closes = stops.len() >= 3 && stops[0] == stops[stops.len() - 1];
    for (k, w) in stops.windows(2).enumerate() {
        let want = if closes && k + 2 == stops.len() { net.carry_on_side(&path) } else { None };
        let found = |memo: &mut HopMemo, x: Option<u8>| memo.entry((w[0], w[1], side.unwrap_or(2), x.unwrap_or(2))).or_insert_with(|| net.find_path_side(w[0], w[1], side, x)).is_some();
        let key = |x: Option<u8>| (w[0], w[1], side.unwrap_or(2), x.unwrap_or(2));
        let x = if want.is_some() && found(memo, want) && (!found(memo, None) || net.carry_worth(&memo[&key(want)].as_ref().unwrap().0, &memo[&key(None)].as_ref().unwrap().0)) { want } else { None };
        found(memo, x);
        let (p, s) = memo[&(w[0], w[1], side.unwrap_or(2), x.unwrap_or(2))].as_ref()?;
        path.extend_from_slice(p);
        side = Some(*s);
    }
    Some(path)
}

/// Whether a line's stops are a loop's (T-798): its last stop is its first.
fn closed(stops: &[u32]) -> bool {
    stops.len() >= 3 && stops[0] == stops[stops.len() - 1]
}

/// The path a line would take closed into a circle at its first stop (T-798), when track leads
/// round from its last stop without turning back there (a train that has to go back the way it
/// came makes no loop). A closing that turns back at the first stop, or runs back over the line's
/// own track (past its stations without stopping), is not offered: a loop at an end is `turns`
/// (T-3021).
fn close_path(net: &Network, line: u32, memo: &mut HopMemo) -> Option<Vec<u32>> {
    let stops = net.line_stops(line);
    if stops.len() < 2 || closed(stops) || stops.len() == 2 && stops[0] == stops[1] {
        return None;
    }
    let mut s = stops.to_vec();
    s.push(stops[0]);
    let path = route_memo(net, &s, memo)?;
    let open = net.line_path(line);
    let back_over = path.len() < open.len() || path[open.len()..].iter().any(|&p| open.iter().any(|&q| q >> 1 == p >> 1));
    (!back_over && net.path_reversals(&path).len() == net.path_reversals(open).len() && net.loop_kind(&s, &path) == LoopKind::Circle).then_some(path)
}

/// T-3021: a line's end turned round a loop of track: the line's new stops and path, and the stop
/// before the branch, whose marker offers it.
#[derive(Debug, PartialEq)]
struct Turn {
    stops: Vec<u32>,
    path: Vec<u32>,
    branch: u32,
}

/// A path's node list, `path.len() + 1` nodes.
fn nodes_of(net: &Network, path: &[u32]) -> Vec<u32> {
    path.first().map_or(vec![], |&f| std::iter::once(net.path_nodes(f).0).chain(path.iter().map(|&p| net.path_nodes(p).1)).collect())
}

/// A rough round trip of `line` run over `stops` and `path` (track time, dwells, turnarounds; a
/// loop one way), to compare a turn with what it replaces.
fn round_guess(net: &Network, line: u32, stops: &[u32], path: &[u32]) -> f64 {
    let t: f64 = path.iter().map(|&p| net.edge_time[(p >> 1) as usize]).sum();
    let (dwell, turn) = (net.line_dwell[line as usize] as f64, net.line_turn[line as usize] as f64);
    let n = stops.len() as f64;
    let rev = net.path_reversals(path).len() as f64 * turn;
    match net.loop_kind(stops, path) {
        LoopKind::None => 2.0 * (t + (n - 2.0).max(0.0) * dwell + rev + turn),
        LoopKind::Balloon => t + (n - 2.0) * dwell + rev + turn,
        LoopKind::Circle => t + (n - 1.0) * dwell + rev,
    }
}

/// How much longer a turn's round trip may be than the line's now (T-3021).
fn turn_allowance(old: f64) -> f64 {
    (0.1 * old).max(TURN_SLACK_S)
}

/// The open line `stops`/`path` turned round a loop past its last stop: out to the end, round the
/// loop, and back over the stops before the point it rejoins, closing at the first stop.
fn turn_last(net: &Network, line: u32, stops: &[u32], path: &[u32], memo: &mut HopMemo) -> Option<Turn> {
    let n = stops.len();
    if n < 2 || path.is_empty() {
        return None;
    }
    let old = round_guess(net, line, stops, path);
    let nodes = nodes_of(net, path);
    let (_, k) = net.loop_back(path, true, nodes.len(), old + turn_allowance(old))?;
    let at = super::net::match_stops(&nodes, stops, &net.stop_reversals(path))?;
    let branch = (0..n).rev().find(|&i| at[i] <= k)?;
    let mut new = stops.to_vec();
    new.extend((0..n - 1).rev().filter(|&i| at[i] <= k).map(|i| stops[i]));
    turned(net, line, new, stops[branch], old, net.path_reversals(path).len(), memo)
}

/// A balloon turned round a loop past its first stop too, a circle: from the first stop after
/// the point it rejoins, round both loops.
fn turn_first(net: &Network, line: u32, stops: &[u32], path: &[u32], memo: &mut HopMemo) -> Option<Turn> {
    let n = stops.len();
    let nodes = nodes_of(net, path);
    let at = super::net::match_stops(&nodes, stops, &net.stop_reversals(path))?;
    let old = round_guess(net, line, stops, path);
    let (_, k) = net.loop_back(path, false, at[open_prefix(stops)], old + turn_allowance(old))?;
    let mut new: Vec<u32> = (1..n).filter(|&i| at[i] > k).map(|i| stops[i]).collect();
    let &first = new.first()?;
    new.push(first);
    turned(net, line, new, first, old, net.path_reversals(path).len(), memo)
}

/// A turn's stops routed, kept when they make a loop with no more reversals than before and a round
/// trip within the allowance; its stops at every pass (`every_pass`).
fn turned(net: &Network, line: u32, stops: Vec<u32>, branch: u32, old: f64, turns: usize, memo: &mut HopMemo) -> Option<Turn> {
    if !closed(&stops) || stops.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    let ok = |s: &[u32], path: &[u32]| net.path_reversals(path).len() == turns && net.loop_kind(s, path) != LoopKind::None && round_guess(net, line, s, path) <= old + turn_allowance(old);
    // the closing hop may carry on round a loop at the first stop's end too; if that is too long,
    // the turn alone (a balloon) is offered
    let take = |path: Vec<u32>| {
        let s = every_pass(net, &stops, &path);
        ok(&s, &path).then_some(Turn { stops: s, path, branch })
    };
    route_memo(net, &stops, memo).and_then(take).or_else(|| route_plain_close(net, &stops, memo).and_then(take))
}

/// `route_memo` for a loop's stops with the closing hop the quickest way, turning back at the
/// first stop if that is it.
fn route_plain_close(net: &Network, stops: &[u32], memo: &mut HopMemo) -> Option<Vec<u32>> {
    let n = stops.len();
    let mut path = route_memo(net, &stops[..n - 1], memo)?;
    let &last = path.last()?;
    let side = net.port_side(last >> 1, 1 - (last & 1));
    let hop = memo.entry((stops[n - 2], stops[n - 1], side.unwrap_or(2), 2)).or_insert_with(|| net.find_path_side(stops[n - 2], stops[n - 1], side, None));
    path.extend_from_slice(&hop.as_ref()?.0);
    Some(path)
}

/// The turns a line's ends can take (T-3021): either end of an open line, or the first stop's end
/// of a balloon. Bus and ferry lines take none.
fn turns(net: &Network, line: u32, memo: &mut HopMemo) -> Vec<Turn> {
    if !net.line_ok(line) || net.line_bus.get(line as usize).is_some_and(|b| b.is_some()) || net.line_ferry.get(line as usize).is_some_and(|f| f.is_some()) {
        return vec![];
    }
    let mut out = turns_of(net, line, net.line_stops(line), net.line_path(line), memo);
    if out.len() == 2 && out[0].branch == out[1].branch {
        out.pop();
    }
    out
}

/// `turns` for a line run over `stops` and `path`, each end's.
fn turns_of(net: &Network, line: u32, stops: &[u32], path: &[u32], memo: &mut HopMemo) -> Vec<Turn> {
    if closed(stops) {
        return if net.loop_kind(stops, path) == LoopKind::Balloon { turn_first(net, line, stops, path, memo).into_iter().collect() } else { vec![] };
    }
    let back: Vec<u32> = stops.iter().rev().copied().collect();
    let back_path: Vec<u32> = path.iter().rev().map(|&p| p ^ 1).collect();
    let mut out: Vec<Turn> = turn_last(net, line, stops, path, memo).into_iter().collect();
    out.extend(turn_last(net, line, &back, &back_path, memo));
    out
}

/// Where a loop's stops start coming back (T-3021): the least index i whose later stops all come
/// before it too, so `stops[..=i]` is the open line it was turned from (a circle less its closing
/// stop).
fn open_prefix(stops: &[u32]) -> usize {
    (0..stops.len()).find(|&i| stops[i + 1..].iter().all(|s| stops[..=i].contains(s))).unwrap_or(stops.len() - 1)
}

/// The stop of a balloon whose marker turns it back into the open line it was turned from, when
/// turning that line gives this balloon again (T-3021).
fn unturn_branch(net: &Network, line: u32, memo: &mut HopMemo) -> Option<u32> {
    let stops = net.line_stops(line);
    if !closed(stops) || net.line_loop(line) != LoopKind::Balloon {
        return None;
    }
    let open = &stops[..=open_prefix(stops)];
    let path = route_memo(net, open, memo)?;
    let t = turn_last(net, line, open, &path, memo)?;
    (t.stops == stops).then_some(t.branch)
}

/// T-3122: a loop's stops with a stop at every pass of its path over a station it stops at (a
/// turned line's stem out and back, a circle's way back), unless that stop would come straight
/// before or after another at the same station: a turn round a loop with no stop of its own between
/// stops there once. The path is kept. An open line, or stops that do not match the path, come back
/// as they are.
pub(super) fn every_pass(net: &Network, stops: &[u32], path: &[u32]) -> Vec<u32> {
    if !closed(stops) || path.is_empty() {
        return stops.to_vec();
    }
    let nodes = nodes_of(net, path);
    let last = nodes.len() - 1;
    let Some(at) = super::net::match_stops(&nodes, stops, &net.stop_reversals(path)).filter(|at| at[at.len() - 1] == last) else { return stops.to_vec() };
    // the walk round, closing stop left out: per pass of a stop's station, whether it is a stop now
    let mut walk: Vec<(u32, bool)> = vec![];
    let mut j = 0;
    for (k, &n) in nodes[..last].iter().enumerate() {
        if at[j] == k {
            walk.push((n, true));
            j += 1;
        } else if stops.contains(&n) {
            walk.push((n, false));
        }
    }
    loop {
        let m = walk.len();
        let Some(i) = (0..m).find(|&i| !walk[i].1 && (walk[(i + m - 1) % m].0 == walk[i].0 || walk[(i + 1) % m].0 == walk[i].0)) else { break };
        walk.remove(i);
    }
    let mut out: Vec<u32> = walk.iter().map(|w| w.0).collect();
    out.push(out[0]);
    out
}

/// T-3122: every loop line of a loaded save (derived) stopping at every pass (`every_pass`); a save
/// made before the rule can have a station a turned line stops at only one way. A one-way circle
/// with one branch comes to end there as `set_line` makes it (T-3141). Whether any changed.
fn stop_every_pass(net: &mut Network) -> bool {
    let mut changed = false;
    for l in 0..net.line_count() as u32 {
        if !net.line_ok(l) || net.line_bus[l as usize].is_some() || net.line_ferry[l as usize].is_some() {
            continue;
        }
        let stops = every_pass(net, net.line_stops(l), net.line_path(l));
        let mut d = net.line_data(l).unwrap();
        if stops.len() != d.stops.len() || net.branch_end(&d).is_some() {
            d.stops = stops;
            changed |= net.set_line(l, Some(d)).is_ok();
        }
    }
    changed
}

/// What a path costs the stop insertion search: its length, and per reversal twice that length, at
/// least `REVERSAL_INSERT_M` (T-1899). Stops that lie along one way through the track then run
/// straight along it unless that way is over three times as long: a flat 5 km let a long way round
/// lose to reversing at a stop, and clicks then built lines that went back and forth.
fn order_cost(net: &Network, path: &[u32]) -> f64 {
    let len: f64 = path.iter().map(|p| net.edge_len[(p >> 1) as usize]).sum();
    len + net.path_reversals(path).len() as f64 * REVERSAL_INSERT_M.max(2.0 * len)
}

/// The line's stops with `station` inserted where its path costs least (`order_cost`), and that
/// path (`TrackApi::insert_stop`). None if the line is not there.
fn best_insert(net: &Network, line: u32, station: u32, memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    if !net.line_ok(line) {
        return None;
    }
    let (stops, path) = best_slot(net, net.line_stops(line), station, memo)?;
    Some((every_pass(net, &stops, &path), path))
}

/// `old` with `station` inserted where the path costs least, and that path: every slot tried with
/// the route search `set_stops` uses. Between slots with the same path (a stretch the line runs out
/// and back over), the one whose stops lie closest together along it wins (T-1674), then the end.
/// None if no slot has a route, or it is a stop already.
fn best_slot(net: &Network, old: &[u32], station: u32, memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    if old.contains(&station) {
        return None;
    }
    let mut best: Option<(Vec<u32>, Vec<u32>)> = None;
    let mut best_len = f64::INFINITY;
    let mut best_spread = None;
    // a loop takes new stops between its first stop and its closing one (T-798); a balloon also
    // past the stop it turns back at, which the new stop then is (T-3122: its stem grows)
    let slots = if closed(old) { 1..=old.len() - 1 } else { 0..=old.len() };
    let mut tries: Vec<Vec<u32>> = slots.rev().map(|slot| {
        let mut stops = old.to_vec();
        stops.insert(slot, station);
        stops
    }).collect();
    if closed(old) && route_memo(net, old, memo).is_some_and(|p| net.loop_kind(old, &p) == LoopKind::Balloon) {
        tries.push(std::iter::once(station).chain(old.iter().copied()).chain(std::iter::once(station)).collect());
    }
    for stops in tries {
        if let Some(path) = route_memo(net, &stops, memo) {
            let len = order_cost(net, &path);
            let wins = if len < best_len - 0.001 {
                best_spread = None;
                true
            } else if len <= best_len + 0.001 && best.as_ref().is_some_and(|b| b.1 == path) {
                let b = best.as_ref().unwrap();
                let held = *best_spread.get_or_insert_with(|| stop_spread(net, &b.0, &b.1));
                let s = stop_spread(net, &stops, &path);
                if s < held - 0.001 { best_spread = Some(s); true } else { false }
            } else {
                false
            };
            if wins {
                best_len = best_len.min(len);
                best = Some((stops, path));
            }
        }
    }
    best
}

/// T-1899: an open line's stops whose path turns back, put in again one at a time in their order
/// as a click puts a stop in (`best_slot`): that order and its path if it turns back fewer times
/// and costs less. Catches lines an edit left going back and forth along one way, whichever
/// edit it was.
fn straighten(net: &Network, stops: &[u32], path: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
    let turns = net.path_reversals(path).len();
    if turns == 0 || stops.len() < 3 || closed(stops) {
        return None;
    }
    let mut memo = HopMemo::new();
    let mut best: (Vec<u32>, Vec<u32>) = (vec![], vec![]);
    for &s in stops {
        best = best_slot(net, &best.0, s, &mut memo)?;
    }
    (net.path_reversals(&best.1).len() < turns && order_cost(net, &best.1) < order_cost(net, path) - 0.001).then_some(best)
}

/// The sum of the distances along `path` between every two of `stops`, as the line's trains run
/// it: smaller when a stop on a stretch run out and back is served on the pass nearer the rest
/// of the line (T-1674). Infinite if the stops do not match the path.
fn stop_spread(net: &Network, stops: &[u32], path: &[u32]) -> f64 {
    let Some(&first) = path.first() else { return 0.0 };
    let nodes: Vec<u32> = std::iter::once(net.path_nodes(first).0).chain(path.iter().map(|&p| net.path_nodes(p).1)).collect();
    let Some(at) = super::net::match_stops(&nodes, stops, &net.stop_reversals(path)) else { return f64::INFINITY };
    let mut off = Vec::with_capacity(path.len() + 1);
    off.push(0.0);
    for &p in path {
        off.push(off[off.len() - 1] + net.edge_len[(p >> 1) as usize]);
    }
    let n = at.len() as f64;
    at.iter().enumerate().map(|(i, &k)| off[k] * (2.0 * i as f64 - n + 1.0)).sum()
}

/// The nodes where a path turns back, in order (T-797).
fn turn_nodes(net: &Network, path: &[u32]) -> Vec<u32> {
    net.path_reversals(path).into_iter().map(|k| net.path_nodes(path[k]).0).collect()
}

/// Nodes a train can reach from any of `stops`, leaving it either way, over valid moves and
/// turning back at bare track ends. Moves work both ways, so a node outside it has no route to or
/// from a stop either.
fn reachable_from(net: &Network, stops: &[u32]) -> Vec<bool> {
    reach_from(net, stops).0
}

/// `reachable_from`'s nodes, and per edge end code (`edge << 1 | end`) whether the search left
/// through it, so an edge is reached when either of its two codes is (T-1896).
fn reach_from(net: &Network, stops: &[u32]) -> (Vec<bool>, Vec<bool>) {
    let mut seen_node = vec![false; net.node_count()];
    let mut seen = vec![false; 2 * net.edge_count()];
    let mut queue: Vec<u32> = vec![];
    for &s in stops {
        if !net.node_ok(s) {
            continue;
        }
        seen_node[s as usize] = true;
        for &c in net.node_ports[s as usize].ends() {
            if !seen[c as usize] {
                seen[c as usize] = true;
                queue.push(c);
            }
        }
    }
    while let Some(st) = queue.pop() {
        let (_, node) = net.path_nodes(st);
        seen_node[node as usize] = true;
        let p = &net.node_ports[node as usize];
        let arr = p.find(super::net::path_edge(st), 1 - super::net::path_dir(st));
        let link = if net.bare_end(node) { 1 } else { arr.map_or(0, |i| net.node_moves[node as usize].link[i]) };
        for k in super::lanes::ends_in(link) {
            let c = p.end[k];
            if !seen[c as usize] {
                seen[c as usize] = true;
                queue.push(c);
            }
        }
    }
    (seen_node, seen)
}

/// A drawn stroke cut into pieces, one per corner clicked (T-244), so removing one removes only
/// that stretch. `verts` is the whole stroke (end, lead-in, the `clicks` corners from index
/// `first`, lead-in, end); each piece is a vertex list from end or joint to joint or end.
///
/// The cut is at the middle of the leg between two consecutive corners. That point is on the
/// straight (an auto curve takes at most half of a leg between corners), so each corner stays a
/// PI of its own piece with its auto radius, and the radius is the same as in the whole stroke: a
/// corner next to an end may use the whole leg, which is now the half leg it had anyway. A leg
/// that changes level is not cut, since a ramp is centred between its corners and the joint, a
/// node, must be at a whole level. Any radius that would still come out different (rounding to the
/// millimetre grid) is set to the whole stroke's, so the track is where it was drawn. A leg a ramp
/// spreads into from the next (T-1895) is not cut either.
/// `r_min` is the smallest radius of the track kind (`geom::fit_min`).
fn stroke_pieces(verts: &[Pi], first: usize, clicks: usize, scaled: bool, r_min: f64) -> Vec<Vec<Pi>> {
    let f = geom::fit_min(verts, scaled, R_CAP, r_min);
    let cuts = if f.issues.is_empty() { mid_cuts(verts, &f, first, clicks, scaled, &|_| true) } else { vec![] };
    cut_at(verts, &f, &cuts, scaled)
}

/// Whether leg `i` of a fitted stroke changes level or carries a ramp, where no joint may go.
fn ramped_leg(verts: &[Pi], f: &geom::Fit, i: usize) -> bool {
    verts[i].level != verts[i + 1].level || geom::on_ramp(&f.vert, f.vert_s[i], f.vert_s[i + 1])
}

/// `stroke_pieces`' cuts: the middle of each leg between consecutive clicked corners where it is on
/// the straight, and of the legs `ok` allows.
fn mid_cuts(verts: &[Pi], f: &geom::Fit, first: usize, clicks: usize, scaled: bool, ok: &dyn Fn(usize) -> bool) -> Vec<(usize, Pi)> {
    let mut cuts = vec![];
    for i in first..(first + clicks).saturating_sub(1) {
        let (p, q) = (verts[i], verts[i + 1]);
        let leg = (q.x - p.x).hypot(q.y - p.y);
        if !ok(i) || ramped_leg(verts, f, i) || leg < 2.0 * MIN_JOINT_LEG * kk(scaled, p.y) || f.tangent[i] > leg / 2.0 + 1e-6 || f.tangent[i + 1] > leg / 2.0 + 1e-6 {
            continue;
        }
        cuts.push((i, Pi::new(quant((p.x + q.x) / 2.0), quant((p.y + q.y) / 2.0), 0.0, p.level)));
    }
    cuts
}

/// T-2927: a curve of street tram track whose middle lies more than this from its two legs (the
/// street's line the route was fitted to) leaves the street, m.
pub(super) const OFF_STREET_M: f64 = 5.0;
/// The straight left on each side of a curve cut out as tram track of its own, m.
const OFF_STREET_PAD: f64 = 1.0;
/// T-2927: street tram track further than this from the street's line it was fitted to (where the
/// fit cut a corner or rounded a sharp one wide) is off the street, ground m: about half a street.
pub(super) const STREET_HALF_M: f64 = 10.0;
/// The straight of a leg is measured against the street every this many ground m.
const STREET_STEP_M: f64 = 5.0;

/// Ground distance from plane point `p` to the polyline `line`.
fn line_dist(line: &[(f64, f64)], p: (f64, f64), scaled: bool) -> f64 {
    let mut best = f64::INFINITY;
    for w in line.windows(2) {
        let ((ax, ay), (bx, by)) = (w[0], w[1]);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((p.0 - ax) * dx + (p.1 - ay) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((ax + t * dx - p.0).hypot(ay + t * dy - p.1));
    }
    best / kk(scaled, p.1)
}

/// T-2927: a street tram stroke as `stroke_pieces` cuts it, each curve that leaves the street
/// (`OFF_STREET_M`) cut out as a piece of tram track of its own from just before its arc to just
/// after; the rest street tram track. With `street` (the line the stroke was fitted to, plane), a
/// curve whose middle or a straight any part of which lies more than `STREET_HALF_M` from it is off
/// the street too, a straight from corner to corner. Where there is no room for a joint between
/// such a curve and the next (a street corner's arc, a ramp, under `MIN_JOINT_LEG` of straight to
/// an end or to the next such curve), the off-street piece takes it in too.
fn street_pieces(verts: &[Pi], first: usize, clicks: usize, scaled: bool, street: &[(f64, f64)]) -> Vec<(Vec<Pi>, TrackKind)> {
    let (on, own) = (TrackKind::StreetTram, TrackKind::Tram);
    let f = geom::fit_min(verts, scaled, R_CAP, on.spec().min_radius);
    let n = verts.len();
    if !f.issues.is_empty() || n < 2 {
        return cut_at(verts, &f, &[], scaled).into_iter().map(|p| (p, on)).collect();
    }
    let leg = |i: usize| (verts[i + 1].x - verts[i].x).hypot(verts[i + 1].y - verts[i].y);
    let heading = |i: usize| (verts[i + 1].y - verts[i].y).atan2(verts[i + 1].x - verts[i].x);
    let far = |p: (f64, f64)| street.len() > 1 && line_dist(street, p, scaled) > STREET_HALF_M;
    // each leg's straight, from tangent point to tangent point
    let leg_off: Vec<bool> = (0..n - 1)
        .map(|i| {
            let (p, l) = (verts[i], leg(i));
            let (from, to) = (f.tangent[i], l - f.tangent[i + 1]);
            let steps = ((to - from) / (STREET_STEP_M * kk(scaled, p.y))).ceil().max(1.0) as usize;
            to > from && (0..=steps).any(|s| {
                let t = (from + (to - from) * s as f64 / steps as f64) / l;
                far((p.x + t * (verts[i + 1].x - p.x), p.y + t * (verts[i + 1].y - p.y)))
            })
        })
        .collect();
    if n == 2 {
        return cut_at(verts, &f, &[], scaled).into_iter().map(|p| (p, if leg_off[0] { own } else { on })).collect();
    }
    let mut off = vec![false; n];
    for i in 1..n - 1 {
        let (h0, h1) = (heading(i - 1), heading(i));
        let turn = geom::wrap(h1 - h0).abs();
        let r = f.radius[i];
        // the arc's middle: from the corner towards the inside, r (sec - 1) along the bisector
        let bis = h0 + geom::wrap(h1 - h0) / 2.0 + if geom::wrap(h1 - h0) > 0.0 { PI / 2.0 } else { -PI / 2.0 };
        let mid = r * (1.0 / (turn / 2.0).cos() - 1.0) * kk(scaled, verts[i].y);
        off[i] = r > 0.0 && (r * (1.0 - (turn / 2.0).cos()) > OFF_STREET_M || far((verts[i].x + mid * bis.cos(), verts[i].y + mid * bis.sin())));
    }
    for i in 0..n - 1 {
        if leg_off[i] {
            (off[i], off[i + 1]) = (true, true);
        }
    }
    // the joints either side of each off-street run: (leg, plane distance from its first vertex);
    // `off` at an end means a run reaches it, with no joint there
    let mut bounds: Vec<(usize, f64)>;
    loop {
        bounds = vec![];
        let mut grew = false;
        for i in 0..n - 1 {
            let (a, b) = (off[i], off[i + 1]);
            if a == b {
                // two off-street curves: a street stretch between them only where it is worth a piece
                let k = kk(scaled, verts[i].y);
                let (pad, l) = (OFF_STREET_PAD * k, leg(i));
                if a && !leg_off[i] && i > 0 && i + 1 < n - 1 && !ramped_leg(verts, &f, i) && l - f.tangent[i] - f.tangent[i + 1] - 2.0 * pad >= MIN_JOINT_LEG * k {
                    bounds.push((i, f.tangent[i] + pad));
                    bounds.push((i, l - f.tangent[i + 1] - pad));
                }
                continue;
            }
            let k = kk(scaled, verts[i].y);
            let (pad, l) = (OFF_STREET_PAD * k, leg(i));
            let (at, far) = if a { (f.tangent[i] + pad, i + 1) } else { (l - f.tangent[i + 1] - pad, i) };
            // room beyond the joint for the far corner's curve, or for a piece to the end
            let room = if a { l - at } else { at };
            let need = if far == 0 || far == n - 1 { MIN_JOINT_LEG * k } else { f.tangent[far] };
            if ramped_leg(verts, &f, i) || room < need - 1e-6 {
                off[far] = true;
                grew = true;
                continue;
            }
            bounds.push((i, at));
        }
        if !grew {
            break;
        }
    }
    let in_run = |i: usize| off[i] && i > 0 && i < n - 1;
    let mut cuts: Vec<(usize, f64, Pi)> = bounds
        .iter()
        .map(|&(i, s)| {
            let (p, q) = (verts[i], verts[i + 1]);
            let t = s / leg(i);
            (i, s, Pi::new(quant(p.x + t * (q.x - p.x)), quant(p.y + t * (q.y - p.y)), 0.0, p.level))
        })
        .collect();
    for (i, j) in mid_cuts(verts, &f, first, clicks, scaled, &|i| !off[i] && !off[i + 1]) {
        cuts.push((i, leg(i) / 2.0, j));
    }
    cuts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let flat: Vec<(usize, Pi)> = cuts.iter().map(|c| (c.0, c.2)).collect();
    // each piece's kind: tram track if it holds an off-street curve
    let mut kinds = vec![];
    let mut start = 0;
    for &(i, _) in flat.iter().chain(std::iter::once(&(n - 1, verts[n - 1]))) {
        kinds.push(if (start + 1..=i.min(n - 2)).any(in_run) { own } else { on });
        start = i;
    }
    cut_at(verts, &f, &flat, scaled).into_iter().zip(kinds).collect()
}

/// T-3130: a street route's ground-level piece of tram track over water is street track: it rounds a
/// corner of a street that crosses there on a bridge of its own (T-3106), and as tram track it would be
/// refused at ground level, while the rest of the route may not leave the ground.
fn street_over_water(net: &Network, pieces: Vec<(Vec<Pi>, TrackKind)>) -> Vec<(Vec<Pi>, TrackKind)> {
    let water = bridge::water(net.water_mask.as_ref());
    let scaled = net.frame().scaled();
    pieces
        .into_iter()
        .map(|(p, k)| {
            let wet = k == TrackKind::Tram && p.iter().all(|v| v.level == 0) && !water(&geom::fit_in(&p, scaled), false).is_empty();
            (p, if wet { TrackKind::StreetTram } else { k })
        })
        .collect()
}

/// A stroke cut at `cuts` (leg index, joint; in order along it), each piece's corners keeping the
/// whole stroke's radii (`stroke_pieces`).
fn cut_at(verts: &[Pi], f: &geom::Fit, cuts: &[(usize, Pi)], scaled: bool) -> Vec<Vec<Pi>> {
    let mut out = vec![];
    let mut start = 0;
    let mut head = verts[0];
    let end = (verts.len() - 1, verts[verts.len() - 1]);
    for &(i, joint) in cuts.iter().chain(std::iter::once(&end)) {
        let mut p = vec![head];
        p.extend_from_slice(&verts[start + 1..=i.min(verts.len() - 2)]);
        p.push(joint);
        // what the whole stroke uses at each corner of the piece
        let pf = geom::fit_in(&p, scaled);
        for j in 1..p.len() - 1 {
            let g = start + j;
            if f.radius[g] > 0.0 && (pf.radius[j] - f.radius[g]).abs() > 0.01 {
                p[j].radius = quant(f.radius[g]);
            }
        }
        // T-727: in the Mercator frame a curve that ended exactly at the cut can overrun it by a
        // centimetre once its radius is set (the scale is taken at the arc's middle), so it is
        // eased by a hundredth of a percent at a time until it fits
        for _ in 0..4 {
            let over = geom::fit_in(&p, scaled).clamped;
            if over.is_empty() {
                break;
            }
            for j in over {
                if p[j].radius > 0.0 {
                    p[j].radius = quant(p[j].radius * (1.0 - 1e-4));
                }
            }
        }
        out.push(p);
        start = i;
        head = joint;
    }
    out
}

/// Centre line to centre line between the two double tracks of quad track, m (T-255): 4 m between
/// every pair of neighbouring tracks, as on the double track itself.
pub const QUAD_SPACING: f64 = 8.0;
/// How much nearer the drawn track an express curve may come where its concentric radius would
/// be under `MIN_RADIUS` and it is widened to it instead, m (T-1678).
const QUAD_EASE_TOL: f64 = 0.5;
/// How long the express pair takes to swing out to its place beside the drawn pair, m.
const QUAD_TAPER: f64 = 150.0;
/// A route through more vertices than this is bridged river by river (`bridge::bridge_long`).
const LONG_ROUTE_VERTS: usize = 16;

/// The express pair of a quad route (T-255): from the drawn route's `verts` (end, lead-in, the
/// `clicks` corners from index `first`, lead-in, end), a double track starting and ending at the
/// same nodes, leaving each end along its heading, swinging out to `QUAD_SPACING` to the left of
/// the drawn direction over `QUAD_TAPER` (less on a short leg), and running parallel between,
/// each corner concentric with the drawn one (radius smaller by the spacing on the inside of the
/// turn, larger outside). Returns its vertices, the index of its first corner-like vertex and how
/// many there are (the taper ends and the corners), for `stroke_pieces`. None if degenerate.
///
/// T-423: `side` is 1 for the express pair on the left, -1 on the right; another size scales the
/// spacing (drawing beside track, `BESIDE_SPACING`). An end given as
/// `own_a` or `own_b` (the express pair's own end node there, from `quad_beside`) has no swing:
/// the pair runs parallel right to that end, every PI of the drawn route concentric.
fn express_verts(verts: &[Pi], first: usize, clicks: usize, scaled: bool, side: f64, own_a: Option<(f64, f64)>, own_b: Option<(f64, f64)>) -> Option<(Vec<Pi>, usize, usize)> {
    let n = verts.len();
    let f = geom::fit_in(verts, scaled);
    let leg = |i: usize| -> Option<(f64, f64, f64)> {
        let (dx, dy) = (verts[i + 1].x - verts[i].x, verts[i + 1].y - verts[i].y);
        let l = dx.hypot(dy);
        (l > 1e-6).then(|| (dx / l, dy / l, l))
    };
    // Ground spacing for the radii; plane spacing (times k at the stroke's start) for positions.
    let kp = kk(scaled, verts[0].y);
    let dg = QUAD_SPACING * side.abs();
    let d = QUAD_SPACING * kp * side;
    // the corners offset: every PI between an own end and the next, else the clicked ones
    let lo = if own_a.is_some() { 1 } else { first };
    let hi = if own_b.is_some() { n - 1 } else { first + clicks };
    // the legs the tapers lie on: from the vertex before the first corner, into the one after the last
    let (s, e) = (lo - 1, hi);
    let (ux, uy, la) = leg(s)?;
    let (vx, vy, lb) = leg(e - 1)?;
    let share = if s == e - 1 { 0.3 } else { 0.4 };
    let ta = (QUAD_TAPER * kp).min(share * la);
    let tb = (QUAD_TAPER * kp).min(share * lb);
    let qa = Pi::new(quant(verts[s].x + ux * ta - uy * d), quant(verts[s].y + uy * ta + ux * d), 0.0, verts[s].level);
    let qb = Pi::new(quant(verts[e].x - vx * tb - vy * d), quant(verts[e].y - vy * tb + vx * d), 0.0, verts[e].level);
    let mut corners = vec![];
    for i in lo..hi {
        let (ax, ay, _) = leg(i - 1)?;
        let (bx, by, _) = leg(i)?;
        let (n1, n2) = ((-ay, ax), (-by, bx));
        let dot = n1.0 * n2.0 + n1.1 * n2.1;
        if f.radius[i] <= 0.0 || dot < -0.9 {
            continue; // no curve here (straight on), or turning back
        }
        let k = d / (1.0 + dot);
        let inside = (ax * by - ay * bx > 0.0) == (side > 0.0);
        let mut r = if inside { f.radius[i] - dg } else { f.radius[i] + dg };
        // T-1678: an inside corner under the minimum widens to it where that brings the arc's
        // middle at most QUAD_EASE_TOL nearer the drawn track (its PI stays, so it moves
        // (MIN_RADIUS - r)(sec(turn / 2) - 1) outwards)
        if inside && r < MIN_RADIUS && f.radius[i] >= MIN_RADIUS - 1e-6 && (MIN_RADIUS - r) * ((2.0 / (1.0 + dot)).sqrt() - 1.0) <= QUAD_EASE_TOL {
            r = MIN_RADIUS;
        }
        corners.push(Pi::new(quant(verts[i].x + k * (n1.0 + n2.0)), quant(verts[i].y + k * (n1.1 + n2.1)), quant(r.max(1.0)), verts[i].level));
    }
    let (a, b) = (verts[0], verts[n - 1]);
    let mut out = vec![];
    if let Some((x, y)) = own_a {
        out.push(Pi::new(x, y, 0.0, a.level));
    } else {
        out.push(Pi::new(a.x, a.y, 0.0, a.level));
        let (hx, hy, _) = leg(0)?;
        if let Some(p) = lead_in_along(a.x, a.y, a.level, hy.atan2(hx), (qa.x, qa.y), scaled) {
            out.push(p);
        }
    }
    let first_out = out.len();
    if own_a.is_none() {
        out.push(qa);
    }
    out.extend(corners.iter().copied());
    if own_b.is_none() {
        out.push(qb);
    }
    let count = out.len() - first_out;
    if let Some((x, y)) = own_b {
        out.push(Pi::new(x, y, 0.0, b.level));
    } else {
        let (gx, gy, _) = leg(n - 2)?;
        if let Some(p) = lead_in_along(b.x, b.y, b.level, (-gy).atan2(-gx), (qb.x, qb.y), scaled) {
            out.push(p);
        }
        out.push(Pi::new(b.x, b.y, 0.0, b.level));
    }
    // T-708: a corner whose curve fills its legs is ill-conditioned when the turn is slight (a
    // capped curve of a fraction of a degree has tangents of a metre or two): the millimetre grid
    // tilts its legs enough that the concentric radius overruns them. Auto fills them instead,
    // within millimetres of concentric. An auto curve takes half of each leg, so a set one beside
    // it may then not fit either (T-727f: one left in 5,692 kept the whole pair from being cut into
    // pieces, and the 5,400 km single edge took 3 s to add).
    for _ in 0..8 {
        let mut eased = false;
        for j in geom::fit_in(&out, scaled).clamped {
            out[j].radius = 0.0;
            eased = true;
        }
        if !eased {
            break;
        }
    }
    Some((out, first_out, count))
}

/// The other pair's loose end beside track end `node` on quad track drawn through points
/// (T-423): another track end at the same level, about `QUAD_SPACING` to one side, level with it
/// and carrying on the same way. With the side it lies on, 1 left of that way and -1 right.
fn quad_beside(net: &Network, node: u32) -> Option<(u32, f64)> {
    let carry = |n: usize| -> Option<f64> {
        let p = &net.node_ports[n];
        (p.n == 1).then(|| if p.side[0] == 1 { net.node_heading[n] + PI } else { net.node_heading[n] })
    };
    if !net.node_ok(node) {
        return None;
    }
    let i = node as usize;
    let h = carry(i)?;
    let (ux, uy) = (h.cos(), h.sin());
    let k = net.k_at(net.node_y[i]);
    let mut best: Option<(u32, f64, f64)> = None;
    for m in 0..net.node_count() {
        if m == i || !net.node_ok(m as u32) || net.node_level[m] != net.node_level[i] {
            continue;
        }
        let Some(hm) = carry(m) else { continue };
        let (dx, dy) = (net.node_x[m] - net.node_x[i], net.node_y[m] - net.node_y[i]);
        let (along, across) = (dx * ux + dy * uy, ux * dy - uy * dx);
        if along.abs() > 2.0 * k || !(QUAD_SPACING * 0.5 * k..=QUAD_SPACING * 1.5 * k).contains(&across.abs()) || geom::wrap(hm - h).abs() > 0.03 {
            continue;
        }
        if best.is_some_and(|b| b.2 <= across.abs()) {
            continue;
        }
        best = Some((m as u32, across.signum(), across.abs()));
    }
    best.map(|b| (b.0, b.1))
}

/// A track end within 2 m of (x, y), where the express pair's end beside `node` would go (T-708):
/// at the same level and running the same way, so a double-up carries on from the pair beside
/// that is there already.
fn loose_near(net: &Network, node: u32, x: f64, y: f64) -> Option<u32> {
    let h = net.node_heading[node as usize];
    let reach = 2.0 * net.k_at(y);
    (0..net.node_count() as u32).find(|&m| {
        let i = m as usize;
        m != node && net.node_ok(m) && net.node_ports[i].n == 1 && net.node_level[i] == net.node_level[node as usize] && (net.node_x[i] - x).hypot(net.node_y[i] - y) < reach && (net.node_heading[i] - h).sin().abs() < 0.03
    })
}

/// Where the express pair's own end goes beside the drawn route's end `at` (0 start, 1 end) on
/// open ground (T-423): `QUAD_SPACING` to `side` of the track's heading there.
fn express_end(verts: &[Pi], at: usize, side: f64, scaled: bool) -> (f64, f64) {
    let n = verts.len();
    let (p, q) = if at == 0 { (verts[0], verts[1]) } else { (verts[n - 2], verts[n - 1]) };
    let (dx, dy) = (q.x - p.x, q.y - p.y);
    let l = dx.hypot(dy).max(1e-9);
    let d = QUAD_SPACING * kk(scaled, verts[0].y) * side;
    let o = if at == 0 { p } else { q };
    (quant(o.x - dy / l * d), quant(o.y + dx / l * d))
}

/// Where a crossover's end goes (T-338): a split of an edge at a chainage, or a plain joint.
#[derive(Clone, Copy, Debug)]
enum Loc {
    Split(u32, f64),
    Node(u32),
}

/// How near a plain joint a crossover's end uses the joint instead of a split, m (a split needs
/// 5 m from a node).
const JOINT_SNAP: f64 = 8.0;

/// The point chainage `s` along edge `e`, which may be off either end: walked on through plain
/// joints (two edge ends, no station) onto the next piece. None where the track ends, meets a
/// junction or a station first, or the point would be within a few metres of one.
fn locate(net: &Network, e: u32, s: f64) -> Option<Loc> {
    locate_or_stop(net, e, s).ok()
}

/// `locate`, or the node it stopped at (u32::MAX: it walked too far).
fn locate_or_stop(net: &Network, mut e: u32, mut s: f64) -> Result<Loc, u32> {
    for _ in 0..16 {
        let len = net.edge_len[e as usize];
        if s >= JOINT_SNAP && s <= len - JOINT_SNAP {
            return Ok(Loc::Split(e, s));
        }
        let which = if s < len / 2.0 { 0 } else { 1 };
        let over = if which == 0 { -s } else { s - len };
        let n = net.end_node(e, which);
        let i = n as usize;
        let p = &net.node_ports[i];
        if p.n != 2 || net.node_platform[i] > 0 {
            return Err(n);
        }
        if over.abs() < JOINT_SNAP {
            return Ok(Loc::Node(n));
        }
        let &c = p.ends().iter().find(|&&c| c != (e << 1 | which)).ok_or(n)?;
        let (o, ow) = (c >> 1, c & 1);
        if net.edge_crossover[o as usize] {
            return Err(n);
        }
        // `over` metres on from `n` into `o`
        s = if ow == 0 { over } else { net.edge_len[o as usize] - over };
        e = o;
    }
    Err(u32::MAX)
}

/// The other diagonal of the crossover edge `e` is (T-338): the crossover edge crossing it.
fn crossover_partner(net: &Network, e: u32) -> Option<u32> {
    if !net.edge_ok(e) || !net.edge_crossover[e as usize] {
        return None;
    }
    net.crossings.iter().find_map(|c| {
        let o = if c.e1 == e { c.e2 } else if c.e2 == e { c.e1 } else { return None };
        (o != e && net.edge_crossover[o as usize]).then_some(o)
    })
}

/// T-656: a segment that needed a bridge and had no room for the ramps fails ground over water;
/// the player is told about the ramps instead, `short` naming the end that has no room (T-2414).
fn bridge_refusal(issues: Vec<Issue>, short: &Issue) -> Vec<Issue> {
    let mut out: Vec<Issue> = vec![];
    for i in issues {
        let i = if matches!(i, Issue::GroundOverWater { .. }) { short.clone() } else { i };
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

/// Plane metres per ground metre at plane `y` in a scaled (Mercator) frame, else exactly 1.
fn kk(scaled: bool, y: f64) -> f64 {
    if scaled { super::frame::merc_k(y) } else { 1.0 }
}

/// The shortest leg either side of a joint between pieces of a stroke, m (T-244).
const MIN_JOINT_LEG: f64 = 20.0;

/// A PI along a node's heading, so a new edge leaves the node the way its track runs (like a
/// turnout) and its first arc starts right at the node: on the main line's heading, a straight
/// would run on top of the main line (an overlap). `toward` is the next vertex of the route; of
/// the directions the node allows, the one nearer to it. None where the node has no track yet
/// or the route already leaves along the heading.
fn lead_in(r: &Resolved, toward: (f64, f64), scaled: bool, cap: f64) -> Option<Pi> {
    lead_in_capped(r.x, r.y, r.level, nearer_dir(r, toward)?, toward, scaled, cap)
}

/// Street corners `tram_ends` may drop at each end of a route.
const TRAM_DROP: usize = 3;

/// T-2997: a street tram route's ends at track with a heading (`lead_in`): each lead-in at least long
/// enough for a curve of `r_min` from the heading onto the street, and the street corners next to an
/// end dropped where that curve has no room before them (the track then leaves the street there and
/// is priced as off it, `street_pieces`). The fewest corners dropped whose alignment fits; if none
/// does, the plain lead-ins, refused as before.
fn tram_ends(ra: &Resolved, rb: &Resolved, pis: Vec<Pi>, scaled: bool, r_min: f64, cap: f64) -> (Vec<Pi>, Option<Pi>, Option<Pi>) {
    let lead = |r: &Resolved, toward: (f64, f64)| {
        let dir = nearer_dir(r, toward)?;
        let p = lead_in_capped(r.x, r.y, r.level, dir, toward, scaled, cap)?;
        // the shortest lead that leaves room for the smallest curve, a hair over: the curve at the
        // lead's end turns by more than the way to `toward` does from the node, so a few rounds
        let (r_k, d0) = (1.02 * r_min * kk(scaled, r.y), (p.x - r.x).hypot(p.y - r.y));
        let mut need = 0.0;
        for _ in 0..6 {
            let (lx, ly) = (r.x + dir.cos() * need, r.y + dir.sin() * need);
            let turn = geom::wrap((toward.1 - ly).atan2(toward.0 - lx) - dir).abs();
            need = r_k * (turn / 2.0).tan();
            if !need.is_finite() || need > (toward.0 - r.x).hypot(toward.1 - r.y) {
                return Some(p);
            }
        }
        Some(if need > d0 { Pi::new(r.x + dir.cos() * need, r.y + dir.sin() * need, 0.0, r.level).quantised() } else { p })
    };
    let n = pis.len();
    let (ha, hb) = (!ra.dirs.is_empty(), !rb.dirs.is_empty());
    for total in 0..=2 * TRAM_DROP {
        for da in 0..=total.min(TRAM_DROP) {
            let db = total - da;
            if db > TRAM_DROP || da + db > n || (da > 0 && !ha) || (db > 0 && !hb) {
                continue;
            }
            let rest = &pis[da..n - db];
            let la = lead(ra, rest.first().map_or((rb.x, rb.y), |p| (p.x, p.y)));
            let lb = lead(rb, rest.last().map_or((ra.x, ra.y), |p| (p.x, p.y)));
            let mut v = vec![Pi::new(ra.x, ra.y, 0.0, ra.level)];
            v.extend(la);
            v.extend_from_slice(rest);
            v.extend(lb);
            v.push(Pi::new(rb.x, rb.y, 0.0, rb.level));
            if geom::fit_min(&v, scaled, R_CAP, r_min).issues.is_empty() {
                return (rest.to_vec(), la, lb);
            }
        }
    }
    let la = lead_in(ra, pis.first().map_or((rb.x, rb.y), |p| (p.x, p.y)), scaled, cap);
    let lb = lead_in(rb, pis.last().map_or((ra.x, ra.y), |p| (p.x, p.y)), scaled, cap);
    (pis, la, lb)
}

/// Of the directions an end allows a new edge to leave in, the one nearer to `toward`.
fn nearer_dir(r: &Resolved, toward: (f64, f64)) -> Option<f64> {
    let want = (toward.1 - r.y).atan2(toward.0 - r.x);
    r.dirs.iter().copied().min_by(|a, b| geom::wrap(a - want).abs().total_cmp(&geom::wrap(b - want).abs()))
}

/// The furthest off its start's heading a tangent segment's end may be (T-442): a turn of up to
/// 350 degrees; right behind the start the circle grows without bound.
const TANGENT_MAX_OFF: f64 = 175.0 * PI / 180.0;

/// T-418: the PIs that make a segment from (x, y), leaving on heading `dir`, one circular arc
/// through `to`. For a chord d at angle a off the heading the arc turns 2a with radius
/// d / (2 sin a). Up to a quarter turn that is one PI d / (2 cos a) along the heading, both legs
/// that long, so the auto radius, which may use an end leg whole, is the arc's. A wider turn
/// (T-442) is split into n equal corners of at most 90 degrees, each PI r tan(2a / 2n) from the
/// arc's tangent points: the legs between are twice that, of which the auto radius uses half, so
/// it is the arc's again; the express pair of quad track offsets each corner concentric, which it
/// cannot do for one corner turning back. Empty where `to` is on the heading; Err where it is more
/// than `TANGENT_MAX_OFF` off it. An arc wider than `cap` (the design speed's radius, T-817)
/// becomes the capped curve then a straight (T-471). Also the curve's circle (centre x, y, ground
/// radius); None for a straight.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn tangent_arc(x: f64, y: f64, level: i8, level_to: i8, dir: f64, to: (f64, f64), scaled: bool, cap: f64) -> (Result<Vec<Pi>, ()>, Option<[f64; 3]>) {
    let (dx, dy) = (to.0 - x, to.1 - y);
    let d = dx.hypot(dy);
    let off = geom::wrap(dy.atan2(dx) - dir);
    if d < 1.0 || off.abs() < 0.002 {
        return (Ok(vec![]), None);
    }
    let a = off.abs();
    let k = kk(scaled, y);
    let (ux, uy) = (dir.cos(), dir.sin());
    let (nx, ny) = (-uy * off.signum(), ux * off.signum());
    let r = d / (2.0 * a.sin());
    let rc = cap * k;
    if a > TANGENT_MAX_OFF {
        return (Err(()), Some([x + nx * r, y + ny * r, r / k]));
    }
    // T-471: one round curve however far the click; at r = rc the straight is nil, so the shape
    // does not jump as the click crosses the cap
    let (radius, turn) = if r <= rc {
        (r, 2.0 * a)
    } else {
        let (fx, fy) = (d * a.cos(), d * a.sin() - rc); // the click from the circle's centre
        let reach = fx.hypot(fy);
        (rc, (fy.atan2(fx) + PI / 2.0 - (rc / reach).min(1.0).acos()).rem_euclid(2.0 * PI))
    };
    let circle = Some([x + nx * radius, y + ny * radius, radius / k]);
    let n = (turn / (PI / 2.0) - 1e-9).ceil().max(1.0) as usize;
    if n == 1 && r <= rc {
        let t = d / (2.0 * a.cos());
        return (Ok(vec![Pi::new(x + ux * t, y + uy * t, 0.0, level).quantised()]), circle);
    }
    let step = turn * off.signum() / n as f64;
    let t = radius * (step.abs() / 2.0).tan();
    let (mut px, mut py) = (x + ux * t, y + uy * t);
    let mut pis = vec![Pi::new(px, py, 0.0, level).quantised()];
    for j in 1..n {
        let h = dir + step * j as f64;
        px += 2.0 * t * h.cos();
        py += 2.0 * t * h.sin();
        // a level change ramps on the middle leg (2t), not the short last one
        pis.push(Pi::new(px, py, 0.0, if j < n / 2 { level } else { level_to }).quantised());
    }
    (Ok(pis), circle)
}

/// A lead-in PI leaving (x, y) on heading `dir` towards `toward` (`lead_in`), or None when the
/// way to `toward` is already within the node's tolerance of the heading.
fn lead_in_along(x: f64, y: f64, level: i8, dir: f64, toward: (f64, f64), scaled: bool) -> Option<Pi> {
    lead_in_capped(x, y, level, dir, toward, scaled, R_CAP)
}

/// `lead_in_along` for a curve of up to `cap` (ground m; the drawing tool's design radius, T-817).
fn lead_in_capped(x: f64, y: f64, level: i8, dir: f64, toward: (f64, f64), scaled: bool, cap: f64) -> Option<Pi> {
    let (dx, dy) = (toward.0 - x, toward.1 - y);
    let dist = dx.hypot(dy);
    if dist < 1.0 {
        return None;
    }
    let delta = geom::wrap(dy.atan2(dx) - dir).abs();
    if delta < 0.002 {
        return None;
    }
    // The arc at the lead-in uses the whole first leg (auto radius, notes/T-008.md 1.2) as long
    // as its radius stays under the cap; 0.3 of the way keeps it inside half the next leg.
    let d = (cap * kk(scaled, y) * (delta / 2.0).tan()).min(0.3 * dist).max(0.5);
    Some(Pi::new(x + dir.cos() * d, y + dir.sin() * d, 0.0, level).quantised())
}

/// Points along an alignment for drawing: x, y, height triples; arcs within 5 cm, straights
/// every 100 m (so ramps show), ramp ends included.
fn render_speeds(pieces: &[geom::Piece], vert: &[[f64; 2]], cap: f64) -> Vec<f32> {
    if pieces.is_empty() { return vec![]; }
    render_chainages(pieces, vert).windows(2).map(|pair| {
        let mid = (pair[0] + pair[1]) * 0.5;
        let i = pieces.partition_point(|p| p.s0 <= mid).saturating_sub(1);
        (pieces[i].v_curve().min(cap) * 3.6) as f32
    }).collect()
}

fn render_chainages(pieces: &[geom::Piece], vert: &[[f64; 2]]) -> Vec<f64> {
    let mut ss: Vec<f64> = vec![];
    for p in pieces {
        let step = if p.k == 0.0 { 100.0 } else { 50.0f64.min((8.0 * p.radius() * 0.05).sqrt()) };
        let n = (p.len / step).ceil().max(1.0) as usize;
        for j in 0..n {
            ss.push(p.s0 + p.len * j as f64 / n as f64);
        }
    }
    if let Some(p) = pieces.last() {
        ss.push(p.s0 + p.len);
    }
    for v in vert {
        ss.push(v[0]);
    }
    ss.sort_by(|a, b| a.total_cmp(b));
    ss.dedup_by(|a, b| (*a - *b).abs() < 0.5);
    ss
}

/// The crossings a trial edit added or removed (an accepted edit), or the ones a refusal names,
/// as `preview_marks` rows (x, y, 0). The refusal's are the bad ones.
fn trial_crossings(r: &Result<EditResult, Vec<Issue>>) -> Vec<f64> {
    let mut out = vec![];
    match r {
        Ok(r) => r.dirty.crossings.iter().for_each(|c| out.extend([c[0], c[1], 0.0])),
        Err(is) => is.iter().for_each(|i| {
            if let Issue::Crossing { x, y, overlap: false, .. } = i {
                out.extend([*x, *y, 0.0]);
            }
        }),
    }
    out
}

fn render_samples(pieces: &[geom::Piece], vert: &[[f64; 2]], out: &mut Vec<f64>) {
    for s in render_chainages(pieces, vert) {
        let (x, y, _) = geom::pos_at(pieces, s);
        out.extend([x, y, geom::height_at(vert, s)]);
    }
}

fn issue_kind(i: &Issue) -> String {
    match i {
        Issue::Geom { issue, .. } => format!(
            "Geom:{}",
            match issue {
                GeomIssue::ZeroLeg(_) => "ZeroLeg",
                GeomIssue::Reversal(_) => "Reversal",
                GeomIssue::RadiusTooSmall(_) => "RadiusTooSmall",
                GeomIssue::RampTooShort(_) => "RampTooShort",
                GeomIssue::LevelRange(_) => "LevelRange",
            }
        ),
        Issue::Crossing { overlap: true, .. } => "CrossingOverlap".into(),
        Issue::OneWayNoWay { line } => format!("OneWayNoWay:{line}"),
        other => {
            let d = format!("{other:?}");
            d.split(|c: char| c == ' ' || c == '{' || c == '(').next().unwrap_or("").to_string()
        }
    }
}

/// A real line being traced (`TrackApi::trace_start`): the router worker steps it between its
/// messages, so a newer request can drop it (T-754).
#[wasm_bindgen]
pub struct RouteTrace {
    t: super::trace::Tracer,
}

#[wasm_bindgen]
impl RouteTrace {
    /// One step, tens of ms at most; true once the result is ready.
    pub fn step(&mut self) -> bool {
        self.t.step()
    }

    /// `trace_route`'s result: [worst, mean distance (m), corner count, then each corner's x, y,
    /// radius (0 auto), level], or empty if it cannot be fitted.
    pub fn result(&mut self) -> Vec<f64> {
        let Some(t) = self.t.take() else { return vec![] };
        let mut out = vec![t.worst, t.mean, t.pis.len() as f64];
        for p in &t.pis {
            out.extend_from_slice(&[p.x, p.y, p.radius, p.level as f64]);
        }
        out
    }
}

impl TrackApi {
    /// T-2707: the boarding account's heap bytes by structure (benches; not exported).
    pub fn boarding_mem(&self) -> Vec<(&'static str, usize)> {
        self.boarding.mem_bytes()
    }

    /// Compose the water mask from the pack rasters for the loaded network's frame. In the legacy
    /// frame each raster is the affine `TransformedMask` earlier builds used (same arithmetic as
    /// the clock worker's, so the costs of legacy saves are unchanged); in the world frame it is
    /// a `GeoMask` through longitude and latitude.
    fn rebuild_water(&mut self) {
        let frame = self.w.net.frame();
        for r in &mut self.rasters {
            r.seen = pack_mask_in(frame, &r.mask, r.lon, r.lat);
        }
        let set = self.mask_set();
        self.w.net.set_water_mask(Box::new(set));
        self.w.reload();
    }

    // ---- not exported to the app: Rust tests and the `realsave` bin
    /// The network and service, read-only (`bin/lanes_parity.rs`).
    pub fn world(&self) -> &TrackWorld {
        &self.w
    }
    /// Plane metres to [lon, lat] in the loaded network's frame.
    pub fn to_lonlat(&self, x: f64, y: f64) -> Vec<f64> {
        let (lon, lat) = self.w.net.frame().to_lonlat(x, y);
        vec![lon, lat]
    }
    /// [lon, lat] to plane metres in the loaded network's frame.
    pub fn from_lonlat(&self, lon: f64, lat: f64) -> Vec<f64> {
        let (x, y) = self.w.net.frame().from_lonlat(lon, lat);
        vec![x, y]
    }
    /// The pack's water mask (notes/T-004.md): header `cell_m`, `origin_m`, `size`, and the
    /// `water_row` (u32) and `water_x` (u16) arrays.
    pub fn set_water(&mut self, cell: f64, x0: f64, y0: f64, w: u32, h: u32, row: &[u32], xs: &[u16]) {
        let m = RunMask { x0, y0, cell, w, h, row: row.to_vec(), xs: xs.to_vec() };
        self.rasters.clear();
        self.w.net.set_water_mask(Box::new(m));
        self.w.reload();
    }
    pub fn new_node_id(&mut self) -> u32 {
        self.w.net.alloc_node()
    }
    pub fn new_edge_id(&mut self) -> u32 {
        self.w.net.alloc_edge()
    }
    /// Add a blueprint edge, or replace one. Constructed track cannot be reshaped (remove it and
    /// draw again). `pis`: x, y, radius (0 = auto), level per PI.
    pub fn set_edge(&mut self, id: u32, a: u32, b: u32, tracks: u8, pis: &[f64]) -> u32 {
        if self.w.net.edge_ok(id) && self.w.net.edge_built[id as usize] {
            return self.refuse(Issue::NoSuchEdge { edge: id });
        }
        self.run(Op::Edge { id, data: Some(EdgeData { a, b, tracks, pis: pis_from(pis), built: false, off: false, thru: vec![], crossover: false, kind: TrackKind::Regular, sep: false, day: 0, oneway: 0, join: [0; 2] }) })
    }
    pub fn set_station(&mut self, node: u32, platform: u16, name: String) -> u32 {
        self.run(Op::Station { node, data: Some(StationData { platform, name, named: false, built: false, off: false, upgrades: 0, berth: None, day: 0 }) })
    }
    /// What constructing these would cost now, US$M (junctions and crossings they complete
    /// included).
    fn construct_cost(&mut self, edges: &[u32], stations: &[u32], buses: &[u32]) -> f64 {
        let ops = self.construct_ops(edges, stations, buses);
        if ops.is_empty() {
            return 0.0;
        }
        self.w.trial(Op::Batch(ops)).map_or(0.0, |r| r.charge)
    }
    /// The most cars `line`'s shortest platform fits, and the station with that platform (T-733).
    pub fn line_cars_cap(&self, line: u32) -> Vec<u32> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return vec![];
        }
        let Some(&s) = net.line_stops(line).iter().min_by_key(|&&s| net.node_platform[s as usize]) else { return vec![] };
        vec![((net.node_platform[s as usize] as f64 / CAR_LEN).floor().max(1.0)) as u32, s]
    }
    /// Per demand level (high, medium, low): round trip s, trains needed, delay s.
    pub fn line_summary(&self, line: u32) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        (0..LEVELS).flat_map(|k| [l.round_trip[k], l.trains[k] as f64, l.delay[k]]).collect()
    }
    /// Where a line's train is at game time `t` on a trip that left at `dep`: x, y (checks).
    pub fn train_at(&self, line: u32, run: usize, level: usize, dep: f64, t: f64) -> Vec<f64> {
        let Some(l) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let (s, _) = profile::state_at(&l.runs[run].prof[level].phases, t - dep);
        let (x, y) = self.w.svc.position(&self.w.net, line, run, s);
        vec![x, y]
    }
    /// A line's dwell over a round trip at a level, s: the dwell at every intermediate stop both
    /// ways (capacity holds at platforms are delay, not dwell), and the turnarounds: two at the
    /// ends and two at each stop or bare track end where trains reverse partway (T-797). A circle
    /// stands the dwell at its first stop and turns nowhere at its ends (T-798).
    pub fn line_dwell_split(&self, line: u32) -> Vec<f64> {
        let n = &self.w.net;
        let Some(ls) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![0.0, 0.0] };
        let turns = ls.runs[0].turn.iter().filter(|&&t| t).count() as f64;
        let circle = ls.kind == LoopKind::Circle;
        let stops = ls.runs[0].stop_s.len().saturating_sub(2) as f64 - turns + if circle { 1.0 } else { 0.0 };
        let ends = if circle { 0.0 } else { 2.0 };
        let turns = turns + ls.runs[0].ends.len() as f64;
        // T-2981: a one-way loop goes round once
        let ways = ls.live_runs() as f64;
        vec![ways * stops * n.line_dwell[line as usize] as f64, ways * (ends / 2.0 + turns) * n.line_turn[line as usize] as f64]
    }
}

#[cfg(test)]
impl TrackApi {
    /// Tests only: add or replace a line at trains per hour; its path is the fastest route through
    /// the stops, and the next settle gives it whole trains (T-1146).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_line(&mut self, id: u32, stops: &[u32], high: f32, medium: f32, low: f32, dwell_s: f32, turnaround_s: f32, cars: u8, colour: u32, name: String) -> u32 {
        let Some(path) = self.w.net.route(stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { symbol: String::new(), trains: None, auto: None, name, colour, stops: stops.to_vec(), path, tph: [high, medium, low], dwell_s, turnaround_s, cars, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0 }) })
    }
    pub(crate) fn set_schedule(&mut self, id: u32, high: f32, medium: f32, low: f32) -> u32 {
        self.run(Op::Schedule { id, tph: [high, medium, low] })
    }
    /// Detailed refusal text for native test failures; absent from the production WASM API.
    fn issues(&self) -> String {
        self.issues.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod bus_tests;
#[cfg(test)]
mod loop_tests;
#[cfg(test)]
mod ferry_tests;
/// T-656: ground track across water becomes a bridge by itself.
#[cfg(test)]
mod bridge_tests;

#[cfg(test)]
mod regional_water_tests;

#[cfg(test)]
mod quad_edit_tests;

/// T-1615: routes ramp over or under track they would cross flat.
#[cfg(test)]
mod ramp_tests;
#[cfg(test)]
mod beside_tests;

#[cfg(test)]
mod t1071_tests;

/// T-1155: track drawn from a station branches past its platform.
#[cfg(test)]
mod branch_tests;

/// T-1273: stop edits leave a line's ends as termini.
#[cfg(test)]
mod stop_edit_tests;
/// T-3021: turning a line's end round a loop.
#[cfg(test)]
mod turn_tests;
#[cfg(test)]
mod stop_pass_tests;
#[cfg(test)]
mod stop_order_tests;
#[cfg(test)]
mod oneway_tests;
/// T-3103, T-3092: flat-junction badges and real flyovers.
#[cfg(test)]
mod flyover_tests;
/// T-3137: lanes and the move table.
#[cfg(test)]
mod lanes_tests;

#[cfg(test)]
mod related_tests;

/// T-1281: station upgrades.
#[cfg(test)]
mod upgrade_tests;

/// T-1382: the changes the clock worker reads its state rows again for.
#[cfg(test)]
mod changed_tests;

/// T-1964: station placements across joints between pieces of track.
#[cfg(test)]
mod station_sweep_tests;

/// T-1898: drags keep the design speed; a selection's track type.
#[cfg(test)]
mod t1898_tests;
#[cfg(test)]
mod rebuild_tests;
#[cfg(test)]
mod tram_tests;
#[cfg(test)]
mod gap_tests;
/// T-2414: close crossings, ends near water and islands.
#[cfg(test)]
mod water_gap_tests;
#[cfg(test)]
mod built_day_tests;
#[cfg(test)]
mod lane_preview_tests;
