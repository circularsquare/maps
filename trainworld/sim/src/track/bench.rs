//! A large synthetic network and timings of edits on it (T-040). Native: `src/bin/track_bench.rs`;
//! WASM: `track_bench` in `wasm.rs` (feature `track-bench`).
//!
//! The network: corridors drawn as random walks (a PI every 0.8-1.5 km, turns up to 20 degrees),
//! a third of them roots at random levels, the rest branching off an earlier corridor at a
//! junction (the branch curving away at the node, as the drawing tool will make it). Stations at
//! leg midpoints about every `km / stations`. One stopping line per corridor (a branch's line starts
//! on its parent), the rest express lines over a corridor stopping at every third station. Same
//! level corridors cross flat; the others pass over or under.

use super::geom::Pi;
use super::net::{EdgeData, LineData, Network, NodeData, StationData};
use super::params::*;
use super::save;
use super::world::{Op, TrackWorld};

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    #[wasm_bindgen::prelude::wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }
    now()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum V {
    Pi,
    Node(u32),
}

struct Corridor {
    pts: Vec<(f64, f64, V)>,
    level: i8,
    parent: Option<usize>,
}

/// Build the synthetic network (inputs only; `TrackWorld::new` derives it).
pub fn synth(km: f64, stations: usize, lines: usize, seed: u64) -> Network {
    // Wrapping: a plain multiply overflows in debug builds (release wraps silently).
    let mut rng = Rng(seed.max(1).wrapping_mul(0x9E3779B97F4A7C15));
    let mut net = Network::new(-73.985, 40.758, true);
    let n_corr = (lines * 2 / 3).max(1);
    let n_roots = (n_corr / 3).max(1);
    let per = km * 1000.0 / n_corr as f64;
    let spacing = km * 1000.0 / stations as f64;
    let half_box = (km * 1000.0 * 30.0).sqrt().max(5000.0);
    let mut cors: Vec<Corridor> = vec![];
    let new_node = |net: &mut Network, x: f64, y: f64, level: i8| {
        let id = net.alloc_node();
        net.set_node(id, Some(NodeData { x: quant(x), y: quant(y), level, flying: false })).unwrap();
        id
    };
    for c in 0..n_corr {
        let (mut x, mut y, mut th, level, mut pts);
        let mut parent = None;
        if c < n_roots {
            x = rng.range(-half_box, half_box);
            y = rng.range(-half_box, half_box);
            th = rng.range(0.0, std::f64::consts::TAU);
            level = [0i8, 0, 1, -1, -2][(rng.next() * 5.0) as usize % 5];
            let n = new_node(&mut net, x, y, level);
            pts = vec![(x, y, V::Node(n))];
        } else {
            // Branch off the middle of a free leg of an earlier corridor, along its heading.
            let p = (rng.next() * c as f64) as usize;
            parent = Some(p);
            let par = &cors[p];
            let mut legs: Vec<usize> = (1..par.pts.len() - 2)
                .filter(|&i| par.pts[i].2 == V::Pi && par.pts[i + 1].2 == V::Pi && par.pts[i - 1].2 == V::Pi)
                .collect();
            if legs.is_empty() {
                legs.push(1);
            }
            let i = legs[(rng.next() * legs.len() as f64) as usize % legs.len()];
            let (a, b) = (par.pts[i], par.pts[i + 1]);
            level = par.level;
            x = (a.0 + b.0) / 2.0;
            y = (a.1 + b.1) / 2.0;
            th = (b.1 - a.1).atan2(b.0 - a.0);
            let j = new_node(&mut net, x, y, level);
            cors[p].pts.insert(i + 1, (quant(x), quant(y), V::Node(j)));
            pts = vec![(quant(x), quant(y), V::Node(j))];
            x += 400.0 * th.cos();
            y += 400.0 * th.sin();
            pts.push((quant(x), quant(y), V::Pi));
            th += rng.range(0.55, 0.8) * if rng.next() < 0.5 { 1.0 } else { -1.0 };
            x += 1300.0 * th.cos();
            y += 1300.0 * th.sin();
            pts.push((quant(x), quant(y), V::Pi));
        }
        let mut len = 0.0;
        while len < per {
            th += rng.range(-0.35, 0.35);
            let step = rng.range(800.0, 1500.0);
            x += step * th.cos();
            y += step * th.sin();
            len += step;
            pts.push((quant(x), quant(y), V::Pi));
        }
        cors.push(Corridor { pts, level, parent });
    }
    // Stations at leg midpoints every ~spacing, and at both ends.
    for cor in &mut cors {
        let mut run = spacing / 2.0;
        let mut i = 1;
        while i + 2 < cor.pts.len() {
            let (a, b) = (cor.pts[i], cor.pts[i + 1]);
            let leg = (b.0 - a.0).hypot(b.1 - a.1);
            run += leg;
            if run >= spacing && a.2 == V::Pi && b.2 == V::Pi && cor.pts[i - 1].2 == V::Pi && cor.pts[i + 2].2 == V::Pi {
                let n = new_node(&mut net, (a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0, cor.level);
                net.set_station(n, Some(StationData { platform: 160, name: format!("S{n}"), built: true })).unwrap();
                cor.pts.insert(i + 1, (net.node_x[n as usize], net.node_y[n as usize], V::Node(n)));
                run = 0.0;
                i += 1;
            }
            i += 1;
        }
        let last = *cor.pts.last().unwrap();
        let n = new_node(&mut net, last.0, last.1, cor.level);
        cor.pts.pop();
        cor.pts.push((last.0, last.1, V::Node(n)));
    }
    for cor in &cors {
        // Ends are stations; a branch's first node is its junction.
        let ends = if cor.parent.is_none() { vec![cor.pts[0], *cor.pts.last().unwrap()] } else { vec![*cor.pts.last().unwrap()] };
        for p in ends {
            if let V::Node(n) = p.2 {
                if net.node_platform[n as usize] == 0 {
                    net.set_station(n, Some(StationData { platform: 160, name: format!("S{n}"), built: true })).unwrap();
                }
            }
        }
    }
    // Edges between consecutive nodes of each corridor.
    for cor in &cors {
        let mut from: Option<u32> = None;
        let mut pis = vec![];
        for &(x, y, v) in &cor.pts {
            match v {
                V::Pi => pis.push(Pi::new(x, y, 0.0, cor.level)),
                V::Node(n) => {
                    if let Some(a) = from {
                        let id = net.alloc_edge();
                        net.set_edge(id, Some(EdgeData { a, b: n, tracks: 2, pis: std::mem::take(&mut pis), built: true, thru: vec![] })).unwrap();
                    }
                    from = Some(n);
                    pis.clear();
                }
            }
        }
    }
    // Derive once so routing can see edge times and ports.
    let mut w = TrackWorld::new(net);
    let net = &mut w.net;
    let station_nodes = |net: &Network, path: &[u32]| {
        let mut v = vec![net.path_nodes(path[0]).0];
        v.extend(path.iter().map(|&p| net.path_nodes(p).1));
        v.into_iter().filter(|&n| net.node_platform[n as usize] > 0).collect::<Vec<u32>>()
    };
    let mut made = 0;
    let ends: Vec<(u32, u32)> = cors
        .iter()
        .map(|c| {
            // A branch's line starts at its root ancestor's first station, through the junctions.
            let mut root = c;
            while let Some(p) = root.parent {
                root = &cors[p];
            }
            let first = match root.pts[0].2 {
                V::Node(n) => n,
                _ => unreachable!(),
            };
            let last = match c.pts.last().unwrap().2 {
                V::Node(n) => n,
                _ => unreachable!(),
            };
            (first, last)
        })
        .collect();
    let mut k = 0;
    while made < lines && k < lines * 4 {
        let c = k % ends.len();
        let express = k >= ends.len();
        k += 1;
        let (a, b) = ends[c];
        let Some(path) = net.route(&[a, b]) else { continue };
        let mut stops = station_nodes(net, &path);
        if express {
            let last = *stops.last().unwrap();
            stops = stops.iter().copied().enumerate().filter(|(i, _)| i % 3 == 0).map(|x| x.1).collect();
            if *stops.last().unwrap() != last {
                stops.push(last);
            }
        }
        if stops.len() < 2 {
            continue;
        }
        let id = net.alloc_line();
        let tph = if express { [6.0, 3.0, 0.0] } else { [12.0, 6.0, 3.0] };
        net.set_line(id, Some(LineData { name: format!("L{id}"), colour: 0, stops, path, tph, dwell_s: 30.0, turnaround_s: 180.0, cars: 0 })).unwrap();
        made += 1;
    }
    let TrackWorld { net, .. } = w;
    net
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Build, then time edits (each applied and undone `reps` times). Returns a report.
pub fn run(km: f64, stations: usize, lines: usize, reps: usize) -> (String, Vec<u8>) {
    let mut out = String::new();
    let t0 = now_ms();
    let net = synth(km, stations, lines, 7);
    let t1 = now_ms();
    let mut w = TrackWorld::new(net);
    let t2 = now_ms();
    let n = &w.net;
    let route_km: f64 = (0..n.edge_count()).filter(|&e| n.edge_alive[e]).map(|e| n.edge_len[e]).sum::<f64>() / 1000.0;
    let n_st = (0..n.node_count()).filter(|&i| n.node_alive[i] && n.node_platform[i] > 0).count();
    let n_lines = (0..n.line_count() as u32).filter(|&l| n.line_ok(l)).count();
    let ok_lines = w.svc.lines.iter().filter(|l| l.ok).count();
    let edges: Vec<u32> = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).collect();
    let nodes: Vec<u32> = (0..n.node_count() as u32).filter(|&x| n.node_ok(x)).collect();
    let issues = n.validate(&edges, &nodes, &[]);
    let flat = n.crossings.iter().filter(|c| matches!(c.kind, super::net::CrossKind::Flat(_))).count();
    let phases: usize = w.svc.lines.iter().filter(|l| l.ok).map(|l| l.runs.iter().map(|r| r.prof.iter().map(|p| p.phases.len()).sum::<usize>()).sum::<usize>()).sum();
    let busy = (0..w.svc.cap.res.len()).filter(|&r| w.svc.cap.rho[r][HIGH] >= 0.75).count();
    out += &format!(
        "network: {route_km:.0} route-km, {} edges, {} pieces, {n_st} stations, {n_lines} lines ({ok_lines} running), {} crossings ({flat} flat), {} issues\n",
        edges.len(),
        n.pieces.data.len() - n.pieces.garbage,
        n.crossings.len(),
        issues.len()
    );
    out += &format!(
        "capacity: {} resources, {} at 75%+ in the peak; {} profile phases\n",
        w.svc.cap.res.len(),
        busy,
        phases
    );
    out += &format!("generate {:.1} ms, derive everything (geometry, crossings, profiles, capacity) {:.1} ms\n", t1 - t0, t2 - t1);
    let t = now_ms();
    w.svc.update_all(&w.net);
    out += &format!("service only (all profiles + capacity) {:.1} ms\n", now_ms() - t);
    let t = now_ms();
    let mut cap = super::capacity::build(&w.net, &w.svc.lines);
    let tb = now_ms();
    super::capacity::solve(&mut cap, &w.net);
    let ts = now_ms();
    let h = super::capacity::holds(&cap, &w.svc.lines);
    let th = now_ms();
    out += &format!(
        "capacity pass alone {:.2} ms (resources {:.2}, utilisation and delay {:.2}, holds per line {:.2}); {} users\n",
        th - t,
        tb - t,
        ts - tb,
        th - ts,
        cap.users.len()
    );
    drop(h);
    // T-050: the incremental pass for one edit, alone: an edge and the paths of its lines.
    let inc = |w: &TrackWorld, e: Option<u32>, resched: &[u32]| {
        let mut edges: Vec<u32> = vec![];
        let mut nodes: Vec<u32> = vec![];
        if let Some(e) = e {
            edges.push(e);
            for &l in w.net.lines_on_edge(e) {
                let run = &w.svc.lines[l as usize].runs[0];
                edges.extend(run.segs.iter().map(|&p| p >> 1));
                nodes.extend_from_slice(&run.nodes);
            }
        }
        edges.sort_unstable();
        edges.dedup();
        nodes.sort_unstable();
        nodes.dedup();
        let mut v = vec![];
        for _ in 0..reps.max(5) {
            let mut c = w.svc.cap.clone();
            let t = now_ms();
            c.update(&w.net, &w.svc.lines, &edges, &nodes, resched);
            v.push(now_ms() - t);
        }
        median(&mut v)
    };

    // The busiest edge (most lines) and an edge of a single line.
    let busiest = *edges.iter().filter(|&&e| w.net.edge_pis[e as usize].len >= 3).max_by_key(|&&e| (w.net.lines_on_edge(e).len(), e)).unwrap();
    let quiet = *edges.iter().filter(|&&e| w.net.edge_pis[e as usize].len >= 3).min_by_key(|&&e| (w.net.lines_on_edge(e).len(), e)).unwrap();
    let report = |name: &str, w: &mut TrackWorld, make: &dyn Fn(&mut TrackWorld) -> Op| {
        let mut apply = vec![];
        let mut undo = vec![];
        let mut info = String::new();
        for i in 0..reps {
            let op = make(w);
            let t = now_ms();
            let r = w.apply(op);
            apply.push(now_ms() - t);
            match r {
                Ok(r) => {
                    if i == 0 {
                        let d = &r.dirty;
                        info = format!(
                            "tiles {}, edges {}, lines {} (+{} sharing), profiles recomputed {}, demand woken {}",
                            d.tiles.len(),
                            d.edges.len(),
                            d.lines.len(),
                            d.service.shared.len(),
                            d.service.recomputed.len(),
                            d.service.demand.len()
                        );
                    }
                }
                Err(e) => {
                    info = format!("REFUSED {:?}", &e[..e.len().min(2)]);
                    break;
                }
            }
            let t = now_ms();
            w.undo().unwrap().unwrap();
            undo.push(now_ms() - t);
        }
        if apply.is_empty() || undo.is_empty() {
            return format!("{name}: {info}\n");
        }
        format!("{name}: apply {:.2} ms, undo {:.2} ms (median of {}); {info}\n", median(&mut apply), median(&mut undo), apply.len())
    };
    let nudge = |e: u32, w: &mut TrackWorld| {
        let mut d = w.net.edge_data(e).unwrap();
        // A middle PI: the first and last set the heading at the nodes.
        let k = d.pis.len() / 2;
        d.pis[k].x = quant(d.pis[k].x + 3.0);
        Op::Edge { id: e, data: Some(d) }
    };
    let l0 = (0..w.net.line_count() as u32).find(|&l| w.svc.lines[l as usize].ok).unwrap();
    out += &format!(
        "incremental capacity pass alone (T-050): busiest edge {:.2} ms, quiet edge {:.2} ms, one line's trains per hour {:.2} ms
",
        inc(&w, Some(busiest), &[]),
        inc(&w, Some(quiet), &[]),
        inc(&w, None, &[l0])
    );
    out += &report(&format!("move a PI 3 m on the busiest edge ({} lines)", w.net.lines_on_edge(busiest).len()), &mut w, &|w| nudge(busiest, w));
    out += &report(&format!("move a PI 3 m on a quiet edge ({} lines)", w.net.lines_on_edge(quiet).len()), &mut w, &|w| nudge(quiet, w));
    out += &report("change one line's trains per hour", &mut w, &|w| {
        let t = w.net.line_tph[l0 as usize];
        Op::Schedule { id: l0, tph: [t[0] + 2.0, t[1], t[2]] }
    });
    out += &report("split the busiest edge (new node)", &mut w, &|w| {
        let s = w.net.edge_len[busiest as usize] * 0.37;
        let node = w.net.alloc_node();
        let new_edge = w.net.alloc_edge();
        Op::Split { edge: busiest, s, node, new_edge }
    });
    let far = w.net.node_x.iter().cloned().fold(0.0f64, f64::max) + 5000.0;
    out += &report("draw 5 km of new track no line uses", &mut w, &|w| {
        let (a, b, e) = (w.net.alloc_node(), w.net.alloc_node(), w.net.alloc_edge());
        Op::Batch(vec![
            Op::Node { id: a, data: Some(NodeData { x: far, y: 0.0, level: 0, flying: false }) },
            Op::Node { id: b, data: Some(NodeData { x: far + 3000.0, y: 4000.0, level: 0, flying: false }) },
            Op::Edge { id: e, data: Some(EdgeData { a, b, tracks: 2, pis: vec![Pi::new(far + 3000.0, 0.0, 0.0, 0)], built: true, thru: vec![] }) },
        ])
    });
    out += &report("delete the busiest edge (lines reroute or break)", &mut w, &|_| Op::DeleteEdge { id: busiest });
    let bytes = save::encode(&w.net);
    out += &format!("save: {} bytes raw for {route_km:.0} route-km ({:.1} KB per 1,000 route-km)\n", bytes.len(), bytes.len() as f64 / 1024.0 / route_km * 1000.0);
    let t = now_ms();
    let back = save::decode(&bytes).unwrap();
    let w2 = TrackWorld::new(back);
    out += &format!(
        "load (decode + derive everything) {:.1} ms; cost {:.1} vs {:.1} US$M\n",
        now_ms() - t,
        w2.net.total_cost(),
        w.net.total_cost()
    );
    (out, bytes)
}
