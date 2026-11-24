//! T-4116: random edits, undos and previews on a real save, natively, each followed by the reads the game makes
//! after it; every panic is logged with the ops that led to it, to replay and shrink.
//!
//!   cargo run --profile agent --bin edit_fuzz -- <track bytes> run <seed> <episodes> <steps> <out dir>
//!   cargo run --profile agent --bin edit_fuzz -- <track bytes> replay <ops file>
//!   cargo run --profile agent --bin edit_fuzz -- <track bytes> shrink <ops file> <out file>
//!
//! The track bytes are a save's network part (after the gzip, magic and JSON header, as for edit_bench). An episode
//! starts from the save and runs up to <steps> ops; one that panics writes <out dir>/panic-<seed>-<ep>.txt, its ops
//! one per line ("name v0 v1 ...", the panic on the first line), which replay and shrink read.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The watchdog's view of the run: when the op under way started (ms, 0 none), the episode's ops so far and where a
/// hang's ops go. An op that runs past TW_HANG_S (30) is written out as a hang and the process exits 3.
static STARTED: AtomicU64 = AtomicU64::new(0);
static OPS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static HANG_FILE: Mutex<String> = Mutex::new(String::new());
fn ms_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}
fn watchdog() {
    let limit: u64 = std::env::var("TW_HANG_S").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let s = STARTED.load(Ordering::SeqCst);
        if s != 0 && ms_now() - s > limit * 1000 {
            let ops = OPS.lock().unwrap().join("\n");
            let path = HANG_FILE.lock().unwrap().clone();
            let last = OPS.lock().unwrap().last().cloned().unwrap_or_default();
            std::fs::write(&path, format!("# HANG over {limit} s in '{last}'\n{ops}\n")).unwrap();
            println!("HANG over {limit} s in '{last}' -> {path}");
            std::process::exit(3);
        }
    });
}
use trainworld_sim::track::wasm::TrackApi;

thread_local! {
    static LAST: RefCell<String> = RefCell::new(String::new());
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> Option<T> {
        if v.is_empty() { None } else { Some(v[self.below(v.len())]) }
    }
}

#[derive(Clone)]
struct Op {
    k: String,
    v: Vec<f64>,
}
impl Op {
    fn new(k: &str, v: Vec<f64>) -> Op {
        Op { k: k.into(), v }
    }
    fn line(&self) -> String {
        let mut s = self.k.clone();
        for x in &self.v {
            s += &format!(" {x:?}");
        }
        s
    }
    fn parse(l: &str) -> Option<Op> {
        let mut it = l.split_whitespace();
        let k = it.next()?.to_string();
        Some(Op { k, v: it.map(|x| x.parse().unwrap()).collect() })
    }
}

/// What the generator picks from, read after every op.
struct View {
    nodes: Vec<(u32, f64, f64, f64)>,
    stations: Vec<(u32, f64, f64)>,
    edges: Vec<(u32, u32, u32, u32, bool)>,
    lines: Vec<u32>,
}
fn view(a: &TrackApi) -> View {
    let ni = a.nodes_info();
    let nodes: Vec<_> = ni.chunks(10).map(|c| (c[0] as u32, c[1], c[2], c[3])).collect();
    let stations = ni.chunks(10).filter(|c| c[7] > 0.0).map(|c| (c[0] as u32, c[1], c[2])).collect();
    let edges = a.edges_info().chunks(10).map(|c| (c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32, c[4] > 0.0)).collect();
    let lines = a.lines_info().chunks(22).map(|c| c[0] as u32).collect();
    View { nodes, stations, edges, lines }
}

/// The stations nearest first from one, for a line that can route.
fn near(vw: &View, from: (u32, f64, f64), k: usize) -> Vec<u32> {
    let mut d: Vec<(f64, u32)> = vw.stations.iter().filter(|s| s.0 != from.0).map(|s| ((s.1 - from.1).hypot(s.2 - from.2), s.0)).collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(k).map(|x| x.1).collect()
}

fn on_edge(a: &TrackApi, e: u32, t: f64) -> Option<(f64, f64)> {
    let r = a.edge_render(e);
    let n = r.len() / 3;
    if n < 2 {
        return None;
    }
    let i = ((n - 1) as f64 * t).round() as usize;
    Some((r[i * 3], r[i * 3 + 1]))
}

fn gen(a: &TrackApi, vw: &View, r: &mut Rng) -> Option<Op> {
    let f = |x: u32| x as f64;
    let st = r.pick(&vw.stations);
    let nd = r.pick(&vw.nodes);
    let ed = r.pick(&vw.edges);
    let ln = r.pick(&vw.lines);
    let roll = r.below(104);
    Some(match roll {
        0..=13 => Op::new("undo", vec![]),
        14..=16 => Op::new("redo", vec![]),
        17..=24 => {
            // draw track: from a node, an edge or open ground, to open ground or another node
            let (x0, y0) = match r.below(3) {
                0 => {
                    let n = nd?;
                    (n.1, n.2)
                }
                _ => {
                    let s = st.or(nd.map(|n| (n.0, n.1, n.2)))?;
                    (s.1, s.2)
                }
            };
            let ang = r.f() * std::f64::consts::TAU;
            let dist = 200.0 + r.f() * 3000.0;
            let (x1, y1) = (x0 + ang.cos() * dist, y0 + ang.sin() * dist);
            let from = match (r.below(3), nd, ed) {
                (0, Some(n), _) => [1.0, f(n.0), 0.0, 0.0, 0.0],
                (1, _, Some(e)) => {
                    let (x, y) = on_edge(a, e.0, 0.2 + 0.6 * r.f())?;
                    [2.0, f(e.0), x, y, 0.0]
                }
                _ => [0.0, 0.0, x0 + 30.0, y0 + 30.0, 0.0],
            };
            let to = if r.below(3) == 0 {
                let m = r.pick(&vw.nodes)?;
                [1.0, f(m.0), 0.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, x1, y1, [0.0, 0.0, 0.0, 1.0, -1.0][r.below(5)]]
            };
            let mut v = from.to_vec();
            v.extend(to);
            v.push([1.0, 2.0, 2.0, 4.0][r.below(4)]);
            v.push(r.below(5) as f64); // 3 tram, 4 high speed, 2 with ramps, else plain rail
            Op::new(if r.below(3) == 0 { "previewRoute" } else { "route" }, v)
        }
        25..=31 => {
            let e = ed?;
            let (x, y) = on_edge(a, e.0, 0.1 + 0.8 * r.f())?;
            Op::new(if r.below(3) == 0 { "previewStation" } else { "stationEdge" }, vec![f(e.0), x, y, [120.0, 200.0, 300.0][r.below(3)]])
        }
        32..=33 => {
            let n = nd?;
            Op::new("stationNode", vec![f(n.0), 200.0])
        }
        34..=36 => {
            let n = st.or(nd.map(|n| (n.0, n.1, n.2)))?;
            let ang = r.f() * std::f64::consts::TAU;
            let (cx, cy) = (n.1 + ang.cos() * (300.0 + 2000.0 * r.f()), n.2 + ang.sin() * (300.0 + 2000.0 * r.f()));
            let h = r.f() * std::f64::consts::PI;
            Op::new(if r.below(3) == 0 { "previewFree" } else { "freeStation" }, vec![cx - h.cos() * 100.0, cy - h.sin() * 100.0, cx + h.cos() * 100.0, cy + h.sin() * 100.0, 0.0, [1.0, 2.0, 2.0, 4.0][r.below(4)]])
        }
        37..=40 => Op::new("removeStation", vec![f(st?.0)]),
        41..=43 => Op::new("deleteEdge", vec![f(ed?.0)]),
        44..=45 => Op::new("deletePieces", vec![f(ed?.0)]),
        46..=49 => Op::new("addTrack", vec![f(ed?.0), [1.0, -1.0, 0.0][r.below(3)]]),
        50..=51 => Op::new("deleteTrack", vec![f(ed?.0), r.below(3) as f64]),
        52 => Op::new("edgesTracks", vec![f(ed?.0), (1 + r.below(2)) as f64]),
        53..=54 => Op::new("rebuild", vec![f(ed?.0), [99.0, -1.0, 0.0, 1.0][r.below(4)], r.below(4) as f64, r.below(3) as f64]),
        55..=57 => Op::new("construct", vec![f(ed?.0)]),
        58 => Op::new("constructAll", vec![]),
        59..=64 => {
            let s = st?;
            let k = 1 + r.below(4);
            let mut stops = vec![f(s.0)];
            let cand = near(vw, s, 6);
            for _ in 0..k {
                if let Some(c) = r.pick(&cand) {
                    if !stops.contains(&f(c)) {
                        stops.push(f(c));
                    }
                }
            }
            Op::new("newLine", stops)
        }
        65..=69 => {
            let l = ln?;
            let stops = a.line_stops(l);
            let s = st?;
            let pick = if !stops.is_empty() && r.below(2) == 0 {
                // a station near one of its stops
                let at = stops[r.below(stops.len())];
                let p = vw.stations.iter().find(|x| x.0 == at).copied().unwrap_or(s);
                r.pick(&near(vw, p, 5)).unwrap_or(s.0)
            } else {
                r.pick(&near(vw, s, 4)).unwrap_or(s.0)
            };
            Op::new(["addStop", "oneWayStop", "stopPassed", "turnEnd", "insertStopQ"][r.below(5)], vec![f(l), f(pick)])
        }
        70..=72 => {
            let l = ln?;
            let n = a.line_stops(l).len().max(1);
            Op::new(["dropStop", "endLineAt"][r.below(2)], vec![f(l), r.below(n) as f64])
        }
        73 => Op::new("openLoop", vec![f(ln?)]),
        74..=75 => Op::new("removeLine", vec![f(ln?)]),
        76 => Op::new("copyLine", vec![f(ln?)]),
        77 => {
            let l = ln?;
            let mut s: Vec<f64> = a.line_stops(l).iter().map(|&x| f(x)).collect();
            if s.len() > 2 {
                let i = r.below(s.len());
                let j = r.below(s.len());
                s.swap(i, j);
            }
            Op::new("setStops", [vec![f(l)], s].concat())
        }
        78 => Op::new("level", vec![f(ed?.0), [-1.0, 0.0, 1.0, 2.0][r.below(4)]]),
        79 => Op::new("kind", vec![f(ed?.0), (1 + r.below(3)) as f64]),
        80 => Op::new("oneWay", vec![f(ed?.0), r.below(3) as f64]),
        81 => Op::new("trackWay", vec![f(ed?.0), r.below(2) as f64, (1 + r.below(3)) as f64]),
        82..=83 => {
            let n = nd?;
            let (dx, dy) = ((r.f() - 0.5) * 200.0, (r.f() - 0.5) * 200.0);
            Op::new(if r.below(2) == 0 { "previewMove" } else { "moveNode" }, vec![f(n.0), n.1 + dx, n.2 + dy])
        }
        84 => Op::new("addPoint", vec![f(ed?.0)]),
        85 => Op::new("stationTracks", vec![f(st?.0), r.below(2) as f64]),
        86 => Op::new("reverseStation", vec![f(st?.0)]),
        87 => Op::new("stationWay", vec![f(st?.0), r.f() * std::f64::consts::TAU]),
        88..=89 => {
            let e = ed?;
            let (x, y) = on_edge(a, e.0, 0.2 + 0.6 * r.f())?;
            Op::new(if r.below(2) == 0 { "previewCrossover" } else { "crossover" }, vec![f(e.0), x, y])
        }
        90 => Op::new("pairs", vec![f(ln?), r.below(2) as f64]),
        91 => Op::new("lineOneWay", vec![f(ln?), r.below(2) as f64]),
        92 => Op::new("fast", vec![f(ln?), r.below(2) as f64]),
        93 => Op::new("trains", vec![f(ln?), r.below(6) as f64, r.below(6) as f64, r.below(4) as f64]),
        94 => Op::new("removalQ", vec![f(ed?.0)]),
        95 => Op::new("blueprintOff", vec![f(ed?.0), r.below(2) as f64]),
        96 => Op::new("stationProps", vec![f(st?.0), [100.0, 200.0, 400.0][r.below(3)]]),
        97 => Op::new("flying", vec![f(nd?.0), r.below(2) as f64]),
        98 => Op::new("sortStops", vec![f(ln?)]),
        99..=101 => Op::new("lineLook", vec![f(ln?), (r.next() & 0xffffff) as f64, r.below(2) as f64]),
        _ => Op::new("constructLine", vec![f(ln?)]),
    })
}

fn apply(a: &mut TrackApi, op: &Op, now: f64) {
    let v = &op.v;
    let u = |i: usize| v[i] as u32;
    a.set_game_day((now / 86400.0) as u32);
    match op.k.as_str() {
        "undo" => {
            a.undo();
        }
        "redo" => {
            a.redo();
        }
        "route" | "previewRoute" => {
            a.set_route_tangent(false);
            a.set_design_speed(160.0);
            a.set_flyover_speed(80.0);
            a.set_route_kind(if v[11] == 3.0 { 3 } else { 0 });
            a.set_route_street(&[]);
            a.set_route_hsr(v[11] == 4.0);
            a.set_route_ramp(v[11] == 2.0);
            a.set_route_beside(u32::MAX, u32::MAX);
            a.set_route_join(&[f64::NAN; 4]);
            let ends = &v[..10];
            if op.k == "route" {
                a.add_route(ends, &[], v[10] as u8, v[11] == 2.0);
            } else {
                a.preview_route(ends, &[], v[10] as u8, v[11] == 2.0);
                let _ = (a.preview_marks(), a.preview_leg_curves(), a.route_joins(), a.route_arc(), a.route_flyovers_flat(), a.preview_ground());
            }
        }
        "stationEdge" => {
            a.set_station_one(&[]);
            a.add_station(2, u(0), v[1], v[2], v[3] as u16, String::new());
        }
        "previewStation" => {
            a.set_station_one(&[]);
            a.preview_station(2, u(0), v[1], v[2], v[3] as u16);
            a.station_cost();
        }
        "stationNode" => {
            a.set_station_one(&[]);
            a.add_station(1, u(0), 0.0, 0.0, v[1] as u16, String::new());
        }
        "freeStation" | "previewFree" => {
            a.set_route_kind(0);
            a.set_route_hsr(false);
            let ends = [0.0, 0.0, v[0], v[1], v[4], 0.0, 0.0, v[2], v[3], v[4]];
            if op.k == "freeStation" {
                a.add_free_station(&ends, v[5] as u8, 200, String::new());
            } else {
                a.preview_free_station(&ends, v[5] as u8, 200);
                a.station_cost();
            }
        }
        "removeStation" => {
            a.remove_station(u(0));
        }
        "deleteEdge" => {
            a.delete_edge(u(0));
        }
        "deletePieces" => {
            a.delete_pieces(&[u(0)], &[]);
        }
        "addTrack" => {
            
        }
        "deleteTrack" => {
            
        }
        "edgesTracks" => {
            a.set_edges_tracks(&[u(0)], v[1] as u8);
        }
        "rebuild" => {
            a.rebuild_pieces(&[u(0)], &[], v[1] as i32, v[2] as i32, v[3] as i32);
        }
        "construct" => {
            let e = u(0);
            let ends: Vec<u32> = a.edges_info().chunks(10).filter(|c| c[0] as u32 == e).flat_map(|c| [c[1] as u32, c[2] as u32]).collect();
            a.construct_with_buses(&[e], &ends, &[]);
        }
        "constructAll" => {
            a.construct_all();
        }
        "constructLine" => {
            a.construct_line(u(0));
        }
        "blueprintOff" => {
            a.set_blueprint_off(&[u(0)], &[], v[1] > 0.0);
        }
        "newLine" => {
            let id = a.new_line_id();
            let stops: Vec<u32> = v.iter().map(|&x| x as u32).collect();
            let name = a.next_line_name();
            a.set_line_with_trains(id, &stops, 4, 3, 2, 30.0, 300.0, 6, 0x2e86de, name);
        }
        "addStop" => {
            a.add_stop(u(0), u(1));
        }
        "oneWayStop" => {
            a.one_way_stop(u(0), u(1));
        }
        "stopPassed" => {
            a.stop_passed(u(0), u(1));
        }
        "turnEnd" => {
            a.turn_end(u(0), u(1));
        }
        "insertStopQ" => {
            a.insert_stop(u(0), u(1));
            a.stop_gap(u(0), u(1));
        }
        "dropStop" => {
            a.drop_stop(u(0), u(1));
        }
        "endLineAt" => {
            a.end_line_at(u(0), u(1));
        }
        "openLoop" => {
            a.open_loop(u(0));
        }
        "removeLine" => {
            a.remove_line(u(0));
        }
        "copyLine" => {
            let id = a.new_line_id();
            a.copy_line(u(0), id, format!("copy {id}"));
        }
        "setStops" => {
            let s: Vec<u32> = v[1..].iter().map(|&x| x as u32).collect();
            a.set_stops(u(0), &s);
        }
        "sortStops" => {
            a.sort_stops(u(0));
        }
        "lineLook" => {
            let name = a.line_name(u(0));
            if v[2] > 0.0 {
                let sym = a.line_symbol(u(0));
                a.set_line_badge(u(0), name, u(1), sym);
            } else {
                a.set_line_look(u(0), name, u(1));
            }
        }
        "level" => {
            a.set_edges_level(&[u(0)], v[1] as i8);
        }
        "kind" => {
            a.set_edges_kind(&[u(0)], v[1] as i32);
        }
        "oneWay" => {
            a.oneway_effects_many(&[u(0)], &[v[1] as u8]);
            a.set_edges_oneway(&[u(0)], &[v[1] as u8]);
        }
        "trackWay" => {
        }
        "moveNode" => {
            a.set_design_speed(160.0);
            a.move_node(u(0), v[1], v[2]);
        }
        "previewMove" => {
            a.set_design_speed(160.0);
            a.preview_move_node(u(0), v[1], v[2]);
        }
        "addPoint" => {
            a.add_point_middle(u(0));
        }
        "stationTracks" => {
            a.station_tracks_cost(u(0));
            a.set_station_tracks(u(0), v[1] > 0.0);
        }
        "reverseStation" => {
            a.reverse_station(u(0));
        }
        "stationWay" => {
            a.set_station_way(u(0), v[1]);
        }
        "stationProps" => {
            a.platform_cost(&[u(0)], v[1] as u16);
            let name = a.node_name(u(0));
            a.set_station_props(u(0), v[1] as u16, name);
        }
        "flying" => {
            a.flyover_cost(u(0));
            a.set_flying(u(0), v[1] > 0.0);
        }
        "crossover" => {
            a.add_crossover(u(0), v[1], v[2]);
        }
        "previewCrossover" => {
            a.preview_crossover(u(0), v[1], v[2]);
        }
        "pairs" => {
            a.line_pairs_why(u(0));
            a.set_line_pairs(u(0), v[1] > 0.0);
        }
        "lineOneWay" => {
            a.set_line_one_way(u(0), v[1] > 0.0);
        }
        "fast" => {
            a.set_line_fast(u(0), v[1] > 0.0);
        }
        "trains" => {
            a.set_train_counts(u(0), u(1), u(2), u(3));
        }
        "removalQ" => {
            a.removal_lines(&[u(0)], &[]);
        }
        k => panic!("unknown op {k}"),
    }
}

/// The reads the game makes after an edit (its state rows, the clock's trips and markers), for the changed lines
/// and edges, or every line when all is set.
fn reads(a: &mut TrackApi, now: f64, all: bool) {
    let stale = a.stale_runs();
    assert!(stale.is_empty(), "stale service runs (line, run, entry): {stale:?}");
    let bad = a.bad_paths();
    assert!(bad.is_empty(), "line paths naming no track (line, 9, entry): {bad:?}");
    let changed = a.take_changed_lines();
    let lines: Vec<u32> = if all { a.lines_info().chunks(22).map(|c| c[0] as u32).collect() } else { changed };
    let _ = (a.lines_info(), a.edges_info(), a.nodes_info(), a.crossings_info(), a.hsr_edges());
    let _ = (a.crossover_edges(), a.crossover_partners(), a.join_edges(), a.oneway_edges(), a.off_edges(), a.off_stations());
    let _ = (a.one_track_stations(), a.one_platform_stations(), a.station_upgrades(), a.station_upgrade_upkeep(), a.named_stations(), a.unbound_stations());
    let _ = (a.edge_limits(), a.edge_ramps(), a.costs(), a.blueprint_piece_costs(), a.phases(), a.profile_meta(), a.guard_conflicts(), a.guard_singles());
    let _ = a.blueprint_cost_items();
    let _ = (a.dirty_demand_lines(), a.station_links(), a.link_charge(), a.cars_needed(), a.take_name_changes());
    for e in a.dirty_edges() {
        let _ = (a.edge_render(e), a.edge_render_speeds(e), a.edge_pis(e), a.edge_kind(e), a.edge_oneway(e), a.edge_way(e), a.edge_infer(e), a.edge_polyline(e, 2.0));
    }
    let periods = a.game_periods() as usize;
    for p in 0..periods.max(1) {
        let _ = (a.markers_in(p), a.full_track_in(p));
    }
    for t in [now, now + 1800.0, now + 7200.0] {
        let _ = a.trips(t);
    }
    for &l in &lines {
        line_reads(a, l, now);
    }
}

fn line_reads(a: &mut TrackApi, l: u32, now: f64) {
    let _ = (a.line_shift(l), a.line_cars_cap(l), a.line_summary(l), a.line_dwell_split(l), a.line_tram(l), a.line_run_factor(l), a.line_train_cost(l), a.line_fast(l));
    let _ = a.line_stretch_speeds(l);
    let _ = a.passed_stations(l);
    // stop_tiers is left out: tens of seconds a line on Vienna's longest lines natively (T-4116 notes), which
    // would stall the run; the line tool's single-station read (insert_stop) is an op of its own
    let _ = (a.line_loop(l), a.line_one_way(l), a.line_one_way_by_track(l), a.line_limit(l), a.line_fare_dist(l));
    let _ = (a.line_reversals(l), a.line_stop_ways(l), a.line_end_turns(l), a.line_phases(l), a.line_held(l), a.line_row_held(l), a.line_marks(l), a.passing_spots(l));
    let _ = (a.line_pairs(l), a.line_pairs_ok(l), a.line_pairs_why(l), a.line_twin(l), a.line_twin_of(l));
    let _ = (a.line_hand(l), a.line_group(l), a.line_one_stops(l), a.line_stop_only(l), a.line_weekend(l), a.line_tph_rest(l), a.line_period_count(l), a.line_periods_own(l));
    let _ = (a.line_row_trains(l), a.line_row_headways(l), a.line_tph_by_period(l), a.line_demand_tph(l), a.line_hours(l), a.line_hours_own(l), a.line_rows(l), a.line_row_weekend(l));
    let _ = (a.line_auto(l), a.line_stops(l), a.line_path(l), a.line_part(l), a.line_run_stops(l), a.line_run_path(l));
    let _ = (a.line_back_path(l), a.line_back_stops(l), a.line_build_cost(l));
    let stops = a.line_stops(l);
    for run in 0..2 {
        let _ = (a.line_stop_s(l, run), a.run_polyline(l, run, 50.0), a.stop_times_in(l, run, 0), a.stop_departures_in(l, run, 0));
        for lev in 0..3 {
            let _ = a.stop_times(l, run, lev);
        }
    }
    for lev in 0..3 {
        let _ = (a.line_delays(l, lev), a.line_waits(l, lev));
    }
    let _ = a.line_trips(l, now, now + 3600.0);
    if let Some(&s) = stops.first() {
        let _ = (a.station_departures(l, s, now, now + 3600.0), a.junction_info(s), a.station_own(s), a.station_one(s), a.station_one_heading(s));
    }
}

fn load(bytes: &[u8], now: f64) -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert!(a.load(bytes), "not a save");
    a.set_unlimited(true);
    a.set_now(now);
    a
}

/// One op and its reads; the panic's place and message when it panics.
fn step(a: &mut TrackApi, op: &Op, now: f64, all: bool) -> Result<(), String> {
    LAST.with(|l| l.borrow_mut().clear());
    let verbose = std::env::var("TW_VERBOSE").is_ok();
    OPS.lock().unwrap().push(op.line());
    STARTED.store(ms_now(), Ordering::SeqCst);
    let r = catch_unwind(AssertUnwindSafe(|| {
        if verbose {
            eprintln!("  op {}", op.line());
        }
        let s0 = trainworld_sim::track::timing::check::stats();
        let f0 = trainworld_sim::track::service::check::stats();
        let t = Instant::now();
        apply(a, op, now);
        let ta = t.elapsed().as_secs_f64() * 1e3;
        // T-4132: what retime did in the op (TW_RETIME_STATS): ms, its calls, clusters, worked out, lines worked out
        if std::env::var("TW_RETIME_STATS").is_ok() {
            let s1 = trainworld_sim::track::timing::check::stats();
            let d: Vec<u64> = (0..8).map(|i| s1[i] - s0[i]).collect();
            println!("RT\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}", op.k, ta, d[1] as f64 / 1e6, d[0], d[2], d[3], d[4]);
        }
        // T-4133: the trains the op moved again (TW_TRAINS_STATS): ms, before_edit ms, note_changes ms, trains running, walked
        if std::env::var("TW_TRAINS_STATS").is_ok() {
            let f1 = trainworld_sim::track::service::check::stats();
            let d: Vec<u64> = (0..6).map(|i| f1[i] - f0[i]).collect();
            println!("TR\t{}\t{:.2}\t{:.3}\t{:.3}\t{}\t{}", op.k, ta, d[1] as f64 / 1e6, d[4] as f64 / 1e6, d[2], d[5]);
        }
        LAST.with(|l| *l.borrow_mut() = "@reads".into());
        reads(a, now, all);
        if verbose {
            eprintln!("{:>8.1} ms + reads {:>8.1} ms  {}", ta, t.elapsed().as_secs_f64() * 1e3 - ta, op.line());
        }
    }));
    STARTED.store(0, Ordering::SeqCst);
    match r {
        Ok(()) => Ok(()),
        Err(_) => Err(LAST.with(|l| l.borrow().clone())),
    }
}

/// The end of a run saved, loaded and read in full, as an autosave and a reload would.
fn save_load(a: &TrackApi, now: f64) -> Result<(), String> {
    LAST.with(|l| l.borrow_mut().clear());
    catch_unwind(AssertUnwindSafe(|| {
        let b = a.save();
        let mut c = TrackApi::new(-73.985, 40.758, true);
        assert!(c.load(&b), "the saved bytes do not load");
        c.set_now(now);
        reads(&mut c, now, true);
    }))
    .map_err(|_| format!("{} (in the save and load at the end)", LAST.with(|l| l.borrow().clone())))
}

/// The ops replayed from the save; the index and panic of the first that panics.
fn replay(bytes: &[u8], now: f64, ops: &[Op]) -> Option<(usize, String)> {
    let mut a = load(bytes, now);
    let _ = a.take_changed_lines();
    for (i, op) in ops.iter().enumerate() {
        if let Err(p) = step(&mut a, op, now, false) {
            return Some((i, p));
        }
    }
    save_load(&a, now).err().map(|p| (ops.len(), p))
}

/// The place a panic names (file:line:col), what shrinking keeps.
fn place(p: &str) -> String {
    p.split(" | ").next().unwrap_or(p).to_string()
}

fn read_ops(path: &str) -> Vec<Op> {
    std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).filter_map(Op::parse).collect()
}

fn main() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        let at = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        if std::env::var("TW_BT").is_ok() {
            let bt = std::backtrace::Backtrace::force_capture().to_string();
            let keep: Vec<&str> = bt.lines().filter(|l| l.contains("trainworld_sim::") || l.contains("at .\\src") || l.contains("src\\track")).take(60).collect();
            eprintln!("BACKTRACE {at}\n{}", keep.join("\n"));
        }
        LAST.with(|l| {
            let mut l = l.borrow_mut();
            let phase = if l.as_str() == "@reads" { " (in the reads after)" } else { "" };
            *l = format!("{at} | {msg}{phase}");
        });
    }));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&args[0]).expect("track bytes");
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    let now: f64 = std::env::var("TW_NOW").ok().and_then(|s| s.parse().ok()).unwrap_or(111600.0);
    match args[1].as_str() {
        "run" => {
            let seed: u64 = args[2].parse().unwrap();
            let (ep0, episodes) = match args[3].split_once('-') {
                Some((a, b)) => (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()),
                None => (0, args[3].parse::<usize>().unwrap()),
            };
            let steps: usize = args[4].parse().unwrap();
            let out = &args[5];
            watchdog();
            std::fs::create_dir_all(out).unwrap();
            let t0 = Instant::now();
            let mut panics = 0;
            for ep in ep0..episodes {
                OPS.lock().unwrap().clear();
                *HANG_FILE.lock().unwrap() = format!("{out}/hang-{seed}-{ep}.txt");
                let mut r = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(ep as u64 * 7919 + 1) | 1);
                let mut a = load(&bytes, now);
                let _ = a.take_changed_lines();
                let mut ops: Vec<Op> = vec![];
                let mut fail = None;
                if std::env::var("TW_VERBOSE").is_ok() {
                    eprintln!("loaded ep {ep}, {:.1} s", t0.elapsed().as_secs_f64());
                }
                let mut vw = view(&a);
                if std::env::var("TW_VERBOSE").is_ok() {
                    eprintln!("first view, {:.1} s", t0.elapsed().as_secs_f64());
                }
                for s in 0..steps {
                    if std::env::var("TW_VERBOSE").is_ok() {
                        eprintln!("step {s}");
                    }
                    let op = match catch_unwind(AssertUnwindSafe(|| gen(&a, &vw, &mut r))) {
                        Ok(Some(op)) => op,
                        Ok(None) => continue,
                        Err(_) => {
                            fail = Some(format!("{} (in the generator's reads)", LAST.with(|l| l.borrow().clone())));
                            break;
                        }
                    };
                    ops.push(op.clone());
                    if let Err(p) = step(&mut a, &op, now, s % 25 == 24) {
                        fail = Some(p);
                        break;
                    }
                    // as an autosave and a recovery's reload would, every few edits
                    if s % 10 == 9 {
                        if let Err(p) = save_load(&a, now) {
                            fail = Some(p);
                            break;
                        }
                    }
                    let t = Instant::now();
                    vw = view(&a);
                    if std::env::var("TW_VERBOSE").is_ok() {
                        eprintln!("    view {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                if fail.is_none() {
                    fail = save_load(&a, now).err();
                }
                let last = ops.last().map(|o| o.line()).unwrap_or_default();
                if let Some(p) = fail {
                    panics += 1;
                    let path = format!("{out}/panic-{seed}-{ep}.txt");
                    let body: String = ops.iter().map(|o| o.line() + "\n").collect();
                    std::fs::write(&path, format!("# {p}\n{body}")).unwrap();
                    println!("PANIC ep {ep} after {} ops, last '{last}': {p}  -> {path}", ops.len());
                } else {
                    println!("ok ep {ep}: {} ops, {:.0} s", ops.len(), t0.elapsed().as_secs_f64());
                }
            }
            println!("done: {episodes} episodes, {panics} panicked, {:.0} s", t0.elapsed().as_secs_f64());
            // T-4132: retime's calls, ns, clusters, clusters worked out, lines worked out, mixed calls, calls checked
            println!("retime stats {:?}", trainworld_sim::track::timing::check::stats());
            // T-4133: before_edit calls, ns, trains; note_changes calls, ns, trains walked; lines off by rounding, most (f64 bits), calls checked
            let f = trainworld_sim::track::service::check::stats();
            println!("trains stats {:?} (most rounding {:e} s)", f, f64::from_bits(f[7]));
        }
        "replay" => {
            watchdog();
            *HANG_FILE.lock().unwrap() = format!("{}.hang", args[2]);
            let ops = read_ops(&args[2]);
            match replay(&bytes, now, &ops) {
                Some((i, p)) => println!("PANIC at op {i} of {} ('{}'): {p}", ops.len(), ops.get(i).map(|o| o.line()).unwrap_or("save".into())),
                None => println!("no panic over {} ops", ops.len()),
            }
        }
        "shrink" => {
            let mut ops = read_ops(&args[2]);
            let Some((i, p)) = replay(&bytes, now, &ops) else {
                println!("no panic to shrink");
                return;
            };
            let want = place(&p);
            ops.truncate(i + 1);
            println!("panic at op {i}: {p}; shrinking {} ops", ops.len());
            let same = |ops: &[Op]| replay(&bytes, now, ops).map_or(false, |(_, q)| place(&q) == want);
            let mut chunk = (ops.len() / 2).max(1);
            loop {
                let mut k = 0;
                let mut cut = false;
                while k < ops.len() {
                    let mut t = ops.clone();
                    let end = (k + chunk).min(t.len());
                    t.drain(k..end);
                    if !t.is_empty() && same(&t) {
                        ops = t;
                        cut = true;
                    } else {
                        k += chunk;
                    }
                }
                if !cut {
                    if chunk == 1 {
                        break;
                    }
                    chunk /= 2;
                }
            }
            let (_, q) = replay(&bytes, now, &ops).unwrap();
            let body: String = ops.iter().map(|o| o.line() + "\n").collect();
            std::fs::write(&args[3], format!("# {q}\n{body}")).unwrap();
            println!("shrunk to {} ops: {q}\n{body}", ops.len());
        }
        m => panic!("unknown mode {m}"),
    }
}
