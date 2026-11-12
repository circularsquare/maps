//! T-3137: a fingerprint of what the track model's lanes and move tables decide, to compare two
//! builds: every line's path and return path, fresh routes through its stops and between station
//! pairs, the flat-junction badges, each junction's inspector, the train guard's crossing pairs,
//! every capacity resource with its users and solved figures, and the lines' figures.
//!
//!   cargo run --release --bin lanes_parity -- <track bytes | game | game-left | mirror-right | mirror-left> <out file>
//!
//! The track bytes are a save's network part (`checks/t3137-lanes-parity.mjs` cuts them out of
//! the real save). `game` builds a small network of its own (flat and flying junctions, a
//! flyover, single track two-way and one-way, quad with a crossover, a circle line, a flat
//! crossing, stations on single track) with
//! right-hand running, `game-left` the same with left-hand. T-3297: `mirror-right` is that game
//! without its quad track, `mirror-left` its mirror image with left-hand running, which must
//! decide everything the same (`checks/t3297-mirror.mjs` compares them). Timings go to stderr.

use std::fmt::Write as _;
use std::time::Instant;
use trainworld_sim::track::capacity::junction_crosses;
use trainworld_sim::track::wasm::TrackApi;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
/// The live edge passing within 1 m of (x, y).
fn edge_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.world().net;
    (0..n.edge_count() as u32).find(|&e| n.edge_ok(e) && n.project(e, x, y).1 < 1.0).unwrap_or_else(|| panic!("no edge at {x} {y}"))
}
fn on_edge(a: &TrackApi, x: f64, y: f64) -> [f64; 5] {
    [2.0, edge_at(a, x, y) as f64, x, y, 0.0]
}
/// The track end nearest (x, y).
fn end_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.world().net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n == 1)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}
fn station_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.world().net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_platform[k as usize] > 0)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}
macro_rules! ok {
    ($a:ident, $call:expr, $what:expr) => {{
        let r = $call;
        assert_eq!(r, 0, "{}: {}", $what, $a.issue_kinds());
    }};
}

/// The small network `game` names, laid through the API as a player would. T-3297: `sx` -1 lays its
/// mirror image (x negated, which Mercator keeps exact: its scale goes with y only); `quad` false
/// leaves out the quad track, whose express pair goes left of the drawing direction whatever the
/// mirror, so a mirrored game is the mirror image only without it.
fn game(right_hand: bool, sx: f64, quad: bool) -> TrackApi {
    let x = |v: f64| v * sx;
    let mut a = TrackApi::new(-73.985, 40.758, right_hand);
    a.set_cash(1e9);
    // a double main, a flat double Y off it, a flyover double branch, a two-way single branch and
    // a single branch made one way, a double line crossing the Y branch flat
    ok!(a, a.add_route(&ends(free(x(0.0), 0.0), free(x(8000.0), 0.0)), &[], 2, false), "main");
    let r = a.add_route(&ends(on_edge(&a, x(1500.0), 0.0), free(x(4000.0), -1500.0)), &[], 2, false);
    ok!(a, r, "Y");
    a.set_route_ramp(true);
    let r = a.add_route(&ends(on_edge(&a, x(5000.0), 0.0), free(x(8500.0), 2500.0)), &[], 2, true);
    a.set_route_ramp(false);
    ok!(a, r, "flyover");
    let r = a.add_route(&ends(on_edge(&a, x(3000.0), 0.0), free(x(4500.0), 1500.0)), &[], 1, false);
    ok!(a, r, "single branch");
    let r = a.add_route(&ends(on_edge(&a, x(6500.0), 0.0), free(x(7500.0), -1400.0)), &[], 1, false);
    ok!(a, r, "one-way branch");
    let r = a.add_route(&ends(free(x(2600.0), -3000.0), free(x(2600.0), 3000.0)), &[], 2, false);
    ok!(a, r, "crossing line");
    // a single-track main with a single Y and a passing station
    ok!(a, a.add_route(&ends(free(x(0.0), 6000.0), free(x(6000.0), 6000.0)), &[], 1, false), "single main");
    let r = a.add_route(&ends(on_edge(&a, x(2000.0), 6000.0), free(x(3500.0), 7500.0)), &[], 1, false);
    ok!(a, r, "single Y");
    // a junction made flying
    let r = a.add_route(&ends(on_edge(&a, x(7000.0), 0.0), free(x(9000.0), -2500.0)), &[], 2, false);
    ok!(a, r, "flying Y");
    {
        let n = &a.world().net;
        let fly = (0..n.node_count() as u32)
            .filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n >= 3)
            .min_by(|&p, &q| {
                let d = |k: u32| (n.node_x[k as usize] - x(7000.0)).hypot(n.node_y[k as usize]);
                d(p).total_cmp(&d(q))
            })
            .unwrap();
        ok!(a, a.set_flying(fly, true), "flying");
    }
    // quad track (its express pair swings out and back in) with a crossover between the pairs, and
    // double track on from each end
    let mut express = None;
    if quad {
        ok!(a, a.add_route(&ends(free(x(0.0), -6000.0), free(x(6000.0), -6000.0)), &[], 4, false), "quad");
        let (qa, qb) = {
            let n = &a.world().net;
            let at = |v: f64| (0..n.node_count() as u32).find(|&k| n.node_ok(k) && (n.node_x[k as usize] - v).abs() < 1.0 && (n.node_y[k as usize] + 6000.0).abs() < 1.0).unwrap();
            (at(x(0.0)), at(x(6000.0)))
        };
        ok!(a, a.add_route(&ends([1.0, qa as f64, x(0.0), -6000.0, 0.0], free(x(-1500.0), -6000.0)), &[], 2, false), "quad west");
        ok!(a, a.add_route(&ends([1.0, qb as f64, x(6000.0), -6000.0, 0.0], free(x(7500.0), -6000.0)), &[], 2, false), "quad east");
        express = Some({
            let n = &a.world().net;
            (0..n.node_count() as u32).find(|&k| n.node_ok(k) && (n.node_x[k as usize] - x(3000.0)).abs() < 1.0 && (n.node_y[k as usize] + 6000.0).abs() > 1.0 && (n.node_y[k as usize] + 6000.0).abs() < 30.0).unwrap()
        });
        let r = a.add_crossover(edge_at(&a, x(2200.0), -6000.0), x(2200.0), -6000.0);
        ok!(a, r, "crossover");
    }
    // a loop: a straight, then a route from its east end round to its west end
    ok!(a, a.add_route(&ends(free(x(0.0), -10000.0), free(x(4000.0), -10000.0)), &[], 2, false), "loop straight");
    let (la, lb) = (end_at(&a, x(0.0), -10000.0), end_at(&a, x(4000.0), -10000.0));
    let r = a.add_route(&ends([1.0, lb as f64, x(4000.0), -10000.0, 0.0], [1.0, la as f64, x(0.0), -10000.0, 0.0]), &[x(5500.0), -12000.0, 0.0, 0.0, x(-1500.0), -12000.0, 0.0, 0.0], 2, false);
    ok!(a, r, "loop round");
    // stations: every track end, mid stations on the main, the single main, the loop
    let bare: Vec<u32> = {
        let n = &a.world().net;
        (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n == 1).collect()
    };
    for (k, &n) in bare.iter().enumerate() {
        let (nx, ny) = (a.world().net.node_x[n as usize], a.world().net.node_y[n as usize]);
        ok!(a, a.add_station(1, n, nx, ny, 200, format!("S{k}")), "end station");
    }
    let mids: &[(f64, f64)] = if quad { &[(4200.0, 0.0), (4000.0, 6000.0), (2000.0, -10000.0), (1200.0, -6000.0)] } else { &[(4200.0, 0.0), (4000.0, 6000.0), (2000.0, -10000.0)] };
    for (k, &(mx, my)) in mids.iter().enumerate() {
        let r = a.add_station(2, edge_at(&a, x(mx), my), x(mx), my, 200, format!("M{k}"));
        ok!(a, r, "mid station");
    }
    let named: Vec<(u32, &str)> = [(lb, "Loop east")].into_iter().chain(express.map(|e| (e, "Express"))).collect();
    for (n, name) in named {
        let (nx, ny) = (a.world().net.node_x[n as usize], a.world().net.node_y[n as usize]);
        ok!(a, a.add_station(1, n, nx, ny, 200, name.into()), "node station");
    }
    let south = {
        let n = &a.world().net;
        // the loop's far side: the point of its round route furthest south
        let at = |e: u32, f: f64| trainworld_sim::track::geom::pos_at(n.edge_pieces(e), n.edge_len[e as usize] * f);
        let lo = |e: u32| (0..=20).map(|k| at(e, k as f64 / 20.0).1).fold(f64::MAX, f64::min);
        let e = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).min_by(|&p, &q| lo(p).total_cmp(&lo(q))).unwrap();
        let (px, py, _) = at(e, 0.5);
        (e, px, py)
    };
    ok!(a, a.add_station(2, south.0, south.1, south.2, 200, "Loop south".into()), "loop south");
    // the one-way branch: its piece nearest the main runs toward the main
    {
        let n = &a.world().net;
        let e = (0..n.edge_count() as u32).find(|&e| n.edge_ok(e) && n.edge_tracks[e as usize] == 1 && n.project(e, x(6600.0), -140.0).1 < 60.0);
        if let Some(e) = e {
            ok!(a, a.set_edge_oneway(e, 1), "one way");
        }
    }
    let st = |a: &TrackApi, px: f64, py: f64| station_near(a, x(px), py);
    let lines: Vec<Vec<(f64, f64)>> = vec![
        vec![(0.0, 0.0), (4200.0, 0.0), (8000.0, 0.0)],
        vec![(0.0, 0.0), (4000.0, -1500.0)],
        vec![(0.0, 0.0), (8500.0, 2500.0)],
        vec![(4200.0, 0.0), (8500.0, 2500.0)],
        vec![(0.0, 0.0), (4500.0, 1500.0)],
        vec![(2600.0, -3000.0), (2600.0, 3000.0)],
        vec![(0.0, 6000.0), (4000.0, 6000.0), (6000.0, 6000.0)],
        vec![(0.0, 6000.0), (3500.0, 7500.0)],
        vec![(0.0, 0.0), (9000.0, -2500.0)],
    ];
    let mut lines: Vec<Vec<u32>> = lines.iter().map(|stops| stops.iter().map(|&(px, py)| st(&a, px, py)).collect()).collect();
    if let Some(express) = express {
        lines.extend([vec![(-1500.0, -6000.0), (1200.0, -6000.0), (7500.0, -6000.0)], vec![(-1500.0, -6000.0), (7500.0, -6000.0)]].iter().map(|stops| stops.iter().map(|&(px, py)| st(&a, px, py)).collect::<Vec<u32>>()));
        let (qw, ql, qe) = (st(&a, -1500.0, -6000.0), st(&a, 1200.0, -6000.0), st(&a, 7500.0, -6000.0));
        lines.extend([vec![qw, express, qe], vec![qw, ql, express, qe]]);
    }
    let (m2, le, ls) = (st(&a, 2000.0, -10000.0), lb, station_near(&a, south.1, south.2));
    lines.push(vec![m2, le, ls, m2]);
    for (k, ids) in lines.iter().enumerate() {
        let l = a.new_line_id();
        let r = a.set_line_with_trains(l, ids, 6, 4, 2, 30.0, 180.0, 0, 0, format!("L{k}"));
        ok!(a, r, &format!("line {k}"));
    }
    ok!(a, a.construct_all(), "construct");
    a
}

fn fingerprint(a: &TrackApi) -> (String, Vec<(String, f64)>) {
    let w = a.world();
    let net = &w.net;
    let mut times = vec![];
    let mut s = String::new();
    let lines: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l)).collect();
    let rail: Vec<u32> = lines.iter().copied().filter(|&l| !net.is_bus(l) && !net.is_ferry(l)).collect();

    s += "## paths\n";
    for &l in &lines {
        writeln!(s, "{l} stops {:?} path {:?} back {:?} broken {}", net.line_stops(l), net.line_path(l), a.line_back_path(l), net.line_broken[l as usize]).unwrap();
    }
    s += "## reroute\n";
    let t = Instant::now();
    for &l in &rail {
        writeln!(s, "{l} {:?}", net.route(net.line_stops(l))).unwrap();
    }
    times.push(("reroute every rail line".into(), ms(t)));
    s += "## pairs\n";
    let stations: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0).collect();
    let t = Instant::now();
    let pairs = stations.len().min(600);
    for i in 0..pairs {
        let (p, q) = (stations[i * stations.len() / pairs], stations[(i * 7919 + 13) % stations.len()]);
        if p == q {
            continue;
        }
        let arrived = [None, Some(0u8), Some(1u8)][i % 3];
        writeln!(s, "{p} {q} {arrived:?} {:?}", net.find_path(p, q, arrived)).unwrap();
    }
    times.push((format!("find_path for {pairs} station pairs"), ms(t)));
    s += "## badges\n";
    let t = Instant::now();
    let badges: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n) && junction_crosses(net, n)).collect();
    times.push(("junction_crosses at every node".into(), ms(t)));
    writeln!(s, "{badges:?}").unwrap();
    s += "## nodes_info\n";
    for c in a.nodes_info().chunks_exact(10) {
        writeln!(s, "{c:?}").unwrap();
    }
    s += "## junction_info\n";
    for n in 0..net.node_count() as u32 {
        if net.node_ok(n) && net.node_ports[n as usize].n >= 3 {
            writeln!(s, "{n} {:?}", a.junction_info(n)).unwrap();
        }
    }
    s += "## guard\n";
    for c in a.guard_conflicts().chunks(5) {
        writeln!(s, "{c:?}").unwrap();
    }
    s += "## capacity\n";
    for (r, us, rho, delay) in w.svc.cap.canonical() {
        writeln!(s, "{:?} node {} edge {} at {:?},{:?} rho {rho:?} delay {delay:?}", r.kind, r.node, r.edge, r.x, r.y).unwrap();
        for u in us {
            writeln!(s, "  {u:?}").unwrap();
        }
    }
    s += "## lines_info\n";
    for c in a.lines_info().chunks(22) {
        writeln!(s, "{c:?}").unwrap();
    }
    s += "## stop_s\n";
    for &l in &rail {
        writeln!(s, "{l} {:?} {:?}", a.line_stop_s(l, 0), a.line_stop_s(l, 1)).unwrap();
    }
    s += "## passed\n";
    for &l in &rail {
        writeln!(s, "{l} {:?}", a.passed_stations(l)).unwrap();
    }
    (s, times)
}

/// Median of `reps` timings of an edit and its undo: (edit ms, undo ms).
fn time_edit(a: &mut TrackApi, reps: usize, edit: &dyn Fn(&mut TrackApi) -> u32) -> Option<(f64, f64)> {
    let mut e = vec![];
    let mut u = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        if edit(a) != 0 {
            eprintln!("  edit refused: {}", a.issue_kinds());
            return None;
        }
        e.push(ms(t));
        let t = Instant::now();
        a.undo();
        u.push(ms(t));
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Some((med(&mut e), med(&mut u)))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (src, out) = (&args[0], &args[1]);
    let t = Instant::now();
    let mut a = match src.as_str() {
        "game" => game(true, 1.0, true),
        "game-left" => game(false, 1.0, true),
        "mirror-right" => game(true, 1.0, false),
        "mirror-left" => game(false, -1.0, false),
        path => {
            let bytes = std::fs::read(path).expect("track bytes");
            let c = TrackApi::convert_save_frame(&bytes, 2);
            let bytes = if c.is_empty() { bytes } else { c };
            let mut a = TrackApi::new(-73.985, 40.758, true);
            assert!(a.load(&bytes), "not a save");
            a.set_unlimited(true);
            a
        }
    };
    let load = ms(t);
    let (fp, times) = fingerprint(&a);
    std::fs::write(out, &fp).expect("write fingerprint");
    eprintln!("load or build {load:.1} ms");
    for (what, t) in times {
        eprintln!("{what} {t:.1} ms");
    }
    // timed edits on the busiest edge's middle: a station split into it, and a double branch off it
    let net = &a.world().net;
    let busiest = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && net.edge_len[e as usize] > 400.0).max_by_key(|&e| net.lines_on_edge(e).len());
    if let Some(e) = busiest {
        let (x, y, h) = trainworld_sim::track::geom::pos_at(net.edge_pieces(e), net.edge_len[e as usize] / 2.0);
        let k = net.k_at(y);
        let (bx, by) = (x + 600.0 * k * (h + 0.5).cos(), y + 600.0 * k * (h + 0.5).sin());
        if let Some((d, u)) = time_edit(&mut a, 5, &|a| a.add_station(2, e, x, y, 100, "Timing".into())) {
            eprintln!("station split into edge {e}: edit {d:.1} ms, undo {u:.1} ms");
        }
        if let Some((d, u)) = time_edit(&mut a, 5, &|a| a.add_route(&ends([2.0, e as f64, x, y, 0.0], free(bx, by)), &[], 2, false)) {
            eprintln!("double branch off edge {e}: edit {d:.1} ms, undo {u:.1} ms");
        }
    }
}
