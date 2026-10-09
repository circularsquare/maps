//! T-007: New York's real network as a track-model save, and the snapshot demand would get from it.
//!
//!   cargo run --release --features demand --bin realsave -- [--dir data/work/realnet]
//!
//! Reads `nyc_real.json` (stations, lines, trains per hour, cars: `pipeline/realnet.py net`) and
//! `nyc_real_geom.json` (per line a node per stop and the GTFS shape between stops, simplified, and
//! a level: `realnet.py geom`). Every line gets its own double track and its own station nodes at
//! its level (a station node takes one route's track, SPEC 6.1); lines meet at a station complex
//! by the demand side's merging and walking transfers. Everything is constructed.
//!
//! Writes `nyc_real.twt2` (the network part of a save, format TWT2, uncompressed) and
//! `nyc_real_track.json`: the stations and lines with the run times the track model gives them,
//! packed as `DemandApi::set_network` takes them (the clock worker's snapshot), for `realnet`.

use serde_json::{json, Value};
use trainworld_sim::track::geom::Pi;
use trainworld_sim::track::net::{EdgeData, LineData, Network, NodeData, StationData};
use trainworld_sim::track::params::*;
use trainworld_sim::track::save;
use trainworld_sim::track::world::TrackWorld;

const PALETTE: [u32; 10] = [0xd7263d, 0x1b998b, 0x2e86de, 0xf46036, 0x7d3c98, 0x2ca02c, 0xe1a100, 0x8c564b, 0x17becf, 0xe377c2];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dir = get("--dir").unwrap_or(format!("{root}/data/work/realnet"));
    let net_j: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/nyc_real.json")).unwrap()).unwrap();
    let geo_j: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/nyc_real_geom.json")).unwrap()).unwrap();
    let pack: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/packs/nyc.json")).unwrap()).unwrap();
    let (lon0, lat0) = (pack["origin"]["lon"].as_f64().unwrap(), pack["origin"]["lat"].as_f64().unwrap());
    let stations = net_j["stations"].as_array().unwrap();
    let lines = net_j["lines"].as_array().unwrap();
    let geoms = geo_j["lines"].as_array().unwrap();

    let mut net = Network::new(lon0, lat0, true);
    let mut placed: Vec<(f64, f64)> = vec![];
    // per line: its node ids; per node: the station key it stands for
    let mut node_key: Vec<String> = vec![];
    let mut node_names: Vec<String> = vec![];
    let mut line_ids = vec![];
    let mut n_lines_ok = 0;
    for (li, (l, g)) in lines.iter().zip(geoms).enumerate() {
        let stops: Vec<usize> = l["stops"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let level = g["level"].as_i64().unwrap() as i8;
        let pts: Vec<(f64, f64)> = g["nodes"].as_array().unwrap().iter().map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect();
        let cars = l["cars"].as_u64().unwrap().clamp(1, 20) as u8;
        let platform = ((cars as u16 * CAR_LEN as u16).clamp(PLATFORM_MIN, PLATFORM_MAX) / PLATFORM_STEP) * PLATFORM_STEP;
        let n = stops.len();
        // headings: along the line through each stop
        let head: Vec<f64> = (0..n)
            .map(|k| {
                let (a, b) = (pts[k.saturating_sub(1)], pts[(k + 1).min(n - 1)]);
                (b.1 - a.1).atan2(b.0 - a.0)
            })
            .collect();
        let mut nodes = vec![];
        for k in 0..n {
            let (mut x, mut y) = pts[k];
            // another line's node already here (lines sharing a platform in GTFS): step aside
            while placed.iter().any(|&(px, py)| (px - x).hypot(py - y) < 6.0) {
                x += 8.0 * (-head[k].sin());
                y += 8.0 * head[k].cos();
            }
            placed.push((x, y));
            let id = net.alloc_node();
            net.set_node(id, Some(NodeData { x: quant(x), y: quant(y), level, flying: false })).unwrap();
            let s = &stations[stops[k]];
            let name = s["name"].as_str().unwrap().to_string();
            net.set_station(id, Some(StationData { platform, name: name.clone(), built: true })).unwrap();
            node_key.push(s["key"].as_str().unwrap().to_string());
            node_names.push(name);
            nodes.push(id);
        }
        let mut path = vec![];
        for k in 0..n - 1 {
            let (a, b) = (nodes[k], nodes[k + 1]);
            let (ax, ay) = (net.node_x[a as usize], net.node_y[a as usize]);
            let (bx, by) = (net.node_x[b as usize], net.node_y[b as usize]);
            let len = (bx - ax).hypot(by - ay);
            // lead out of and into the station along its heading, so the track runs straight
            // through the platform; then the shape's points, those near the ends dropped
            let lead = (len / 4.0).min(80.0);
            let mut pis = vec![Pi::new(ax + lead * head[k].cos(), ay + lead * head[k].sin(), 0.0, level).quantised()];
            for p in g["hops"][k].as_array().unwrap() {
                let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
                if (x - ax).hypot(y - ay) > lead + 30.0 && (x - bx).hypot(y - by) > lead + 30.0 {
                    pis.push(Pi::new(x, y, 0.0, level).quantised());
                }
            }
            pis.push(Pi::new(bx - lead * head[k + 1].cos(), by - lead * head[k + 1].sin(), 0.0, level).quantised());
            let e = net.alloc_edge();
            net.set_edge(e, Some(EdgeData { a, b, tracks: 2, pis, built: true, thru: vec![] })).unwrap();
            path.push(e << 1);
        }
        let tph: Vec<f32> = l["tph"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let id = net.alloc_line();
        let name = format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap());
        let r = net.set_line(id, Some(LineData { name, colour: PALETTE[li % PALETTE.len()], stops: nodes.clone(), path, tph: [tph[0], tph[1], tph[2]], dwell_s: 30.0, turnaround_s: 180.0, cars }));
        if r.is_ok() {
            n_lines_ok += 1;
        }
        line_ids.push(id);
    }
    let bytes = save::encode(&net);
    std::fs::write(format!("{dir}/nyc_real.twt2"), &bytes).unwrap();
    let t0 = std::time::Instant::now();
    let w = TrackWorld::new(save::decode(&bytes).unwrap());
    eprintln!("save: {} bytes, {} nodes, {} lines set ({} ok); load + derive {:.0} ms", bytes.len(), node_key.len(), lines.len(), n_lines_ok, t0.elapsed().as_secs_f64() * 1000.0);
    let issues = w.net.validate(&(0..w.net.edge_count() as u32).collect::<Vec<_>>(), &(0..w.net.node_count() as u32).collect::<Vec<_>>(), &[]);
    let mut kinds = std::collections::BTreeMap::new();
    for i in &issues {
        *kinds.entry(format!("{:?}", i).split(|c| c == ' ' || c == '{' || c == '(').next().unwrap().to_string()).or_insert(0) += 1;
    }
    eprintln!("validation (not checked on load): {kinds:?}");
    let flat = w.net.crossings.iter().filter(|c| matches!(c.kind, trainworld_sim::track::net::CrossKind::Flat(_))).count();
    eprintln!("crossings {} ({} flat), total cost US${:.0}M", w.net.crossings.len(), flat, w.net.total_cost());

    // the snapshot (as clock.worker.ts buildSnapshot + demandClient.ts packService do)
    let mut st_xy = vec![];
    let mut st_out = vec![];
    let mut node_st = vec![u32::MAX; w.net.node_count()];
    for n in 0..w.net.node_count() {
        node_st[n] = (st_xy.len() / 2) as u32;
        st_xy.push(w.net.node_x[n] as f32);
        st_xy.push(w.net.node_y[n] as f32);
        st_out.push(json!({"key": node_key[n], "name": node_names[n], "feed": node_key[n].split(':').next().unwrap().replace("mta", "subway"), "x": w.net.node_x[n], "y": w.net.node_y[n]}));
    }
    let (mut line_n, mut stops, mut times, mut tph, mut cars) = (vec![], vec![], vec![], vec![], vec![]);
    let mut lines_out = vec![];
    let (mut gtfs_s, mut track_s) = (0.0, 0.0);
    let mut not_running = 0;
    let mut fleet = 0u64;
    for (li, &id) in line_ids.iter().enumerate() {
        let svc = &w.svc.lines[id as usize];
        if !svc.running {
            not_running += 1;
            continue;
        }
        let ls = w.net.line_stops(id).to_vec();
        let n = ls.len();
        line_n.push(n as u32);
        stops.extend(ls.iter().map(|&s| node_st[s as usize]));
        let mut t = vec![0f32; 6 * n];
        for lev in 0..3 {
            for run in 0..2 {
                let p = &svc.runs[run].prof[lev];
                for j in 0..n - 1 {
                    let dwell = if j == 0 { 0.0 } else { p.dep[j] - p.arr[j] };
                    let v = (p.arr[j + 1] - p.dep[j] + dwell) as f32;
                    let k = if run == 0 { j } else { n - 1 - j };
                    t[(run * 3 + lev) * n + k] = v;
                }
            }
        }
        let l = &lines[li];
        let g0: f64 = l["t0"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).sum();
        let t0s: f64 = t[..n].iter().map(|&v| v as f64).sum();
        gtfs_s += g0;
        track_s += t0s;
        times.extend(t);
        let d = w.net.line_data(id).unwrap();
        tph.extend(d.tph);
        cars.push(d.cars as f32);
        fleet += *svc.trains.iter().max().unwrap() as u64 * d.cars as u64;
        lines_out.push(json!({"feed": l["feed"], "name": l["name"], "route": l["route"], "gtfs_min": g0 / 60.0, "track_min": t0s / 60.0}));
    }
    eprintln!("{} lines running ({} not); one-way run times, sum over lines: GTFS {:.0} min, track model {:.0} min ({:+.0}%)", lines_out.len(), not_running, gtfs_s / 60.0, track_s / 60.0, 100.0 * (track_s / gtfs_s - 1.0));
    let out = json!({
        "what": "the real network through the track model (T-007): stations are nodes, run times from its profiles",
        "stations": st_out,
        "lines": lines_out,
        "fleet": fleet,
        "demand": {"st_xy": st_xy, "line_n": line_n, "stops": stops, "times": times, "tph": tph, "cars": cars},
    });
    std::fs::write(format!("{dir}/nyc_real_track.json"), serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("-> {dir}/nyc_real.twt2, {dir}/nyc_real_track.json");
}
