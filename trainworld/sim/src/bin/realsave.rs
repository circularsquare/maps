//! T-007: New York's real network as a track-model save, and the snapshot demand would get from it.
//!
//!   cargo run --release --features demand --bin realsave -- [--dir data/work/realnet] [--city nyc] [--out DIR] [--shared] [--left]
//!
//! `--left` (T-3297): the network keeps left (a city whose metro runs on the left, `pipeline/realauto.json`'s
//! `left`, `realnet.py`'s `LEFT`).
//!
//! `--shared` (T-1305) builds instead the network `pipeline/realcorridors.py` laid out in
//! `<city>_real_corridors.json` ("model": lines sharing track, one or two pairs a piece, junctions
//! and station nodes per pair, each line's stops and path, trains an hour after the trim), tracing
//! each edge's real line between its two leads.
//!
//! Reads `nyc_real.json` (stations, lines, trains per hour, cars: `pipeline/realnet.py net`) and
//! `nyc_real_geom.json` (per line a node per stop and the GTFS shape between stops, simplified, and
//! a level: `realnet.py geom`), and `nyc_real_route.json` where there is one (`realnet.py route`,
//! T-1303: the hops routed over the real track, which are traced here). Every line gets its own double track and its own station nodes at
//! its level (a station node takes one route's track, SPEC 6.1); lines meet at a station complex
//! by the demand side's merging and walking transfers. Everything is constructed.
//!
//! Station and line names are tidied as they go in (T-1442, `track::names`): the lines lose their
//! feed ("subway 6X" is "6X"), the stations their capitals and "STATION".
//!
//! Writes `nyc_real.twt2` (the network part of a save, uncompressed, in the game's Mercator frame
//! with its settled rates, T-3233; the lines get whole trains
//! from the real trains per hour when the network is derived, T-1146, unless the network JSON gives
//! a line `trains`, and `hsr` for high-speed track, T-1625; a line over half on high-speed track runs high-speed
//! trains, T-3272) and
//! `nyc_real_track.json`: the stations and lines with the run times the track model gives them,
//! packed as `DemandApi::set_network` takes them (the clock worker's snapshot), for `calib`.

use serde_json::{json, Value};
use trainworld_sim::track::geom::Pi;
use trainworld_sim::track::names;
use trainworld_sim::track::net::{EdgeData, LineData, Network, NodeData, StationData};
use trainworld_sim::track::params::*;
use trainworld_sim::track::frame::FrameKind;
use trainworld_sim::track::save;
use trainworld_sim::track::trace;
use trainworld_sim::track::upgrades;
use trainworld_sim::track::world::TrackWorld;

/// The fit's tolerance: well under the 6 m between lines set side by side (`realnet.py spread`),
/// so that two lines on one real track do not cross over and back (the track tool's is 25 m).
const TRACE_TOL_M: f64 = 3.0;
/// T-1773: a ramp gets its own end corners only this far clear of the leg's corners (their curves)
const RAMP_CLEAR_M: f64 = 30.0;
const RAMP_SLACK_M: f64 = 20.0;

const PALETTE: [u32; 10] = [0xd7263d, 0x1b998b, 0x2e86de, 0xf46036, 0x7d3c98, 0x2ca02c, 0xe1a100, 0x8c564b, 0x17becf, 0xe377c2];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dir = get("--dir").unwrap_or(format!("{root}/data/work/realnet"));
    // any realnet.py city (T-509): <city>_real.json, <city>_real_geom.json, its pack's origin
    let city = get("--city").unwrap_or("nyc".into());
    // T-1303: `--out` stages the outputs elsewhere; the geometry and routes are read from there first
    let out_dir = get("--out").unwrap_or(dir.clone());
    let first = |name: String| {
        let p = format!("{out_dir}/{name}");
        if std::path::Path::new(&p).exists() { p } else { format!("{dir}/{name}") }
    };
    // T-1770: `realnet.py colours` stages a network JSON with the lines' real colours beside the others
    let net_j: Value = serde_json::from_str(&std::fs::read_to_string(first(format!("{city}_real.json"))).unwrap()).unwrap();
    let geo_j: Value = serde_json::from_str(&std::fs::read_to_string(first(format!("{city}_real_geom.json"))).unwrap()).unwrap();
    let route_j: Option<Value> = std::fs::read_to_string(first(format!("{city}_real_route.json"))).ok().map(|s| serde_json::from_str(&s).unwrap());
    if route_j.is_none() && !args.iter().any(|a| a == "--shared") {
        eprintln!("no {city}_real_route.json: every hop from its shape or straight (realnet.py route lays them on real track)");
    }
    let (mut n_traced, mut n_untraced, mut worst_fit, mut trace_s) = (0usize, 0usize, 0f64, 0f64);
    // T-1773: level changes by how their ramp was placed (the kinds in the edge loop, in order)
    let mut ramps_by_how = [0usize; 5];
    let pack_name = net_j["pack"].as_str().unwrap_or(&city).to_string();
    let pack: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/packs/{pack_name}.json")).unwrap()).unwrap();
    let (lon0, lat0) = (pack["origin"]["lon"].as_f64().unwrap(), pack["origin"]["lat"].as_f64().unwrap());
    let stations = net_j["stations"].as_array().unwrap();
    let lines = net_j["lines"].as_array().unwrap();
    let geoms = geo_j["lines"].as_array().unwrap();

    let mut net = Network::new(lon0, lat0, !args.iter().any(|a| a == "--left"));
    let mut placed: Vec<(f64, f64)> = vec![];
    // per line: its node ids; per node: the station key it stands for
    let mut node_key: Vec<String> = vec![];
    let mut node_names: Vec<String> = vec![];
    // (network line index, track line id)
    let mut line_ids: Vec<(usize, u32)> = vec![];
    let mut n_lines_ok = 0;
    // T-1305: `--shared` builds the network `realcorridors.py` laid out (lines sharing track, one
    // or two pairs a piece) from `<city>_real_corridors.json` instead of a track per line
    let shared = args.iter().any(|a| a == "--shared");
    if shared {
        let corr: Value = serde_json::from_str(&std::fs::read_to_string(first(format!("{city}_real_corridors.json"))).unwrap()).unwrap();
        let m = &corr["model"];
        let xy = |p: &Value| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        for nd in m["nodes"].as_array().unwrap() {
            let (x, y) = xy(&nd["xy"]);
            let level = nd["level"].as_i64().unwrap() as i8;
            let id = net.alloc_node();
            let st = &nd["station"];
            net.set_node(id, Some(NodeData { x: quant(x), y: quant(y), level, flying: st.is_null() })).unwrap();
            if st.is_null() {
                node_key.push(String::new());
                node_names.push(String::new());
                continue;
            }
            let raw = st["name"].as_str().unwrap();
            let name = names::tidy_station(raw).unwrap_or_else(|| raw.to_string());
            let platform = st["platform"].as_u64().unwrap() as u16;
            net.set_station(id, Some(StationData { platform, name: name.clone(), named: false, built: true, off: false, upgrades: 0, berth: None, day: 0 })).unwrap();
            node_key.push(st["key"].as_str().unwrap().to_string());
            node_names.push(name);
        }
        // T-3272: an edge is high-speed track when every line on it is a high-speed line (`hsr` in the
        // network JSON)
        let n_edges = m["edges"].as_array().unwrap().len();
        let (mut on_hsr, mut on_other) = (vec![false; n_edges], vec![false; n_edges]);
        for (li, l) in lines.iter().enumerate() {
            if m["lines"][li].is_null() {
                continue;
            }
            let hsr = l["hsr"].as_bool().unwrap_or(false);
            for v in m["lines"][li]["path"].as_array().unwrap() {
                let e = (v.as_u64().unwrap() >> 1) as usize;
                if hsr { on_hsr[e] = true } else { on_other[e] = true }
            }
        }
        let mut edge_ids = vec![];
        let mut refused = 0;
        for (ei, ed) in m["edges"].as_array().unwrap().iter().enumerate() {
            let edge_hsr = on_hsr[ei] && !on_other[ei];
            let design = if edge_hsr { design_radius(320.0) } else { R_CAP };
            let (a, b) = (ed["a"].as_u64().unwrap() as u32, ed["b"].as_u64().unwrap() as u32);
            let mut pis = vec![];
            let node_level = |n: u32| m["nodes"][n as usize]["level"].as_i64().unwrap() as i8;
            let (lv_a, lv_b) = (node_level(a), node_level(b));
            if !ed["lead_a"].is_null() {
                let (la, lb) = (xy(&ed["lead_a"]), xy(&ed["lead_b"]));
                let inner: Vec<(f64, f64)> = ed["line"].as_array().unwrap().iter().map(xy).collect();
                let a_xy = xy(&m["nodes"][a as usize]["xy"]);
                let b_xy = xy(&m["nodes"][b as usize]["xy"]);
                let mut full = vec![a_xy, la];
                full.extend(inner.iter().copied());
                full.push(lb);
                full.push(b_xy);
                let mut cum = vec![0.0];
                for w in full.windows(2) {
                    cum.push(cum.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
                }
                // a point's distance along `full`, where it lies nearest
                let arc_of = |p: (f64, f64)| -> f64 {
                    let mut best = (f64::INFINITY, 0.0);
                    for i in 1..full.len() {
                        let (a, b) = (full[i - 1], full[i]);
                        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                        let l2 = dx * dx + dy * dy;
                        let t = if l2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
                        let d = (a.0 + dx * t - p.0).hypot(a.1 + dy * t - p.1);
                        if d < best.0 {
                            best = (d, cum[i - 1] + t * (cum[i] - cum[i - 1]));
                        }
                    }
                    best.1
                };
                let runs: Vec<(f64, f64, i8)> = ed["levels"].as_array().map(|r| r.iter().map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap(), x[2].as_i64().unwrap() as i8)).collect()).unwrap_or_default();
                let mut line = vec![la];
                line.extend(inner.iter().copied());
                line.push(lb);
                let t = std::time::Instant::now();
                let r = if inner.is_empty() { None } else { trace::trace(&line, false, TRACE_TOL_M, TRACE_MIN_RADIUS, design, lv_a) };
                trace_s += t.elapsed().as_secs_f64();
                let mid: Vec<Pi> = match r {
                    Some(t) => {
                        n_traced += 1;
                        worst_fit = worst_fit.max(t.worst);
                        t.pis
                    }
                    None => {
                        if !inner.is_empty() {
                            n_untraced += 1;
                        }
                        let mut v = vec![];
                        let mut last = la;
                        for &p in &inner {
                            if (p.0 - last.0).hypot(p.1 - last.1) > 30.0 && (p.0 - lb.0).hypot(p.1 - lb.1) > 30.0 {
                                v.push(Pi::new(p.0, p.1, 0.0, lv_a).quantised());
                                last = p;
                            }
                        }
                        v
                    }
                };
                // every corner on its stretch's level (pipeline/reallevels.py's `levels`, [from m, to
                // m, level] along a, lead a, line, lead b, b, and `flat`, the metres level at each end).
                // The game lays a ramp (RAMP_PER_LEVEL a level) centred between the two corners of
                // different levels, spreading over the corners either side at its two levels when
                // longer than its leg. Each change's ramp is centred where the stretches put it, moved
                // into the room it has (past the flat ends and the ramp before, RAMP_SLACK_M short of
                // each), by, in order: a new corner at each end on a leg with room clear of its corners'
                // curves (RAMP_CLEAR_M); one new corner making a leg centred there with an existing
                // corner, both clear; the existing leg whose middle is nearest inside the room; the
                // same new corner nearer a curve; else its own corners at the change anyway.
                let mut seq: Vec<(f64, Pi)> = vec![(cum[1], Pi::new(la.0, la.1, 0.0, lv_a).quantised())];
                for p in mid {
                    seq.push((arc_of((p.x, p.y)), p));
                }
                seq.push((cum[cum.len() - 2], Pi::new(lb.0, lb.1, 0.0, lv_b).quantised()));
                let len = cum[cum.len() - 1];
                let flat = |i: usize| ed["flat"][i].as_f64().unwrap_or(0.0);
                let pt = |seq: &Vec<(f64, Pi)>, j: usize, s: f64, lv: i8| {
                    let ((sp, p), (sq, q)) = (seq[j], seq[j + 1]);
                    let f = if sq > sp { ((s - sp) / (sq - sp)).clamp(0.0, 1.0) } else { 0.5 };
                    (s, Pi::new(p.x + (q.x - p.x) * f, p.y + (q.y - p.y) * f, 0.0, lv).quantised())
                };
                // where each change happens along the edge: corners before it take the level before
                let mut cuts: Vec<f64> = vec![];
                let mut room_lo = flat(0) + RAMP_SLACK_M;
                for k in 1..runs.len() {
                    let (l0, l1, b) = (runs[k - 1].2, runs[k].2, runs[k].0);
                    let r = RAMP_PER_LEVEL * (l1 - l0).abs() as f64;
                    let room_hi = if k + 1 == runs.len() { len - flat(1) } else { runs[k].1 - RAMP_PER_LEVEL / 2.0 * (runs[k + 1].2 - l1).abs() as f64 } - RAMP_SLACK_M;
                    let n = seq.len();
                    let in_room = |c: f64| c - r / 2.0 >= room_lo - 1e-6 && c + r / 2.0 <= room_hi + 1e-6;
                    // the middle wanted, inside the room where there is one
                    let c = if room_hi - room_lo >= r { b.clamp(room_lo + r / 2.0, room_hi - r / 2.0) } else { b };
                    // the leg holding it: corners i - 1 and i
                    let i = (1..n).find(|&i| seq[i].0 > c).unwrap_or(n - 1);
                    let (sp, sq) = (seq[i - 1].0, seq[i].0);
                    // (corners to add as (leg, arc, level), the ramp's middle, how)
                    let mut pick: Option<(Vec<(usize, f64, i8)>, f64, u8)> = None;
                    if in_room(c) && sp + RAMP_CLEAR_M <= c - r / 2.0 && c + r / 2.0 + RAMP_CLEAR_M <= sq {
                        pick = Some((vec![(i - 1, c - r / 2.0, l0), (i - 1, c + r / 2.0, l1)], c, 0));
                    }
                    // one new corner q with an existing one p centred on c: p after c (the first
                    // corner on l1; lead b only for the last change), or p before c (the last on l0;
                    // lead a only for the first change); how clear q is of both corners of its leg
                    let one = |p_after: bool| -> Option<((usize, f64, i8), f64)> {
                        let (p, other, ok) = if p_after { (sq, sp, i < n - 1 || k + 1 == runs.len()) } else { (sp, sq, i - 1 > 0 || k == 1) };
                        let q = 2.0 * c - p;
                        if !ok || !in_room(c) || (q - other).abs() < 1.0 || (q < sp.min(sq)) || (q > sp.max(sq)) {
                            return None;
                        }
                        Some(((i - 1, q, if p_after { l0 } else { l1 }), (q - p).abs().min((q - other).abs())))
                    };
                    let best_one = [one(true), one(false)].into_iter().flatten().max_by(|x, y| x.1.total_cmp(&y.1));
                    if pick.is_none() {
                        if let Some((add, clear)) = best_one {
                            if clear >= RAMP_CLEAR_M {
                                pick = Some((vec![add], c, 1));
                            }
                        }
                    }
                    if pick.is_none() {
                        pick = (0..n - 1)
                            .map(|j| (seq[j].0 + seq[j + 1].0) / 2.0)
                            .filter(|&m| in_room(m))
                            .min_by(|x, y| (x - c).abs().total_cmp(&(y - c).abs()))
                            .map(|m| (vec![], m, 2));
                    }
                    if pick.is_none() {
                        if let Some((add, clear)) = best_one {
                            if clear >= 5.0 {
                                pick = Some((vec![add], c, 3));
                            }
                        }
                    }
                    let (add, cut, how) = pick.unwrap_or_else(|| {
                        let j = (0..n - 1).find(|&j| seq[j + 1].0 >= b).unwrap_or(n - 2);
                        // only the ends that fall inside the leg, so no leg is left of zero length
                        let add = [(b - r / 2.0, l0), (b + r / 2.0, l1)].into_iter()
                            .filter(|&(s, _)| s > seq[j].0 + 1.0 && s < seq[j + 1].0 - 1.0)
                            .map(|(s, lv)| (j, s, lv)).collect();
                        (add, b, 4)
                    });
                    ramps_by_how[how as usize] += 1;
                    let z0 = cut - r / 2.0;
                    // added from the last, so the leg indices still hold
                    for &(j, s, lv) in add.iter().rev() {
                        let p = pt(&seq, j, s, lv);
                        seq.insert(j + 1, p);
                    }
                    cuts.push(cut);
                    room_lo = z0 + r + RAMP_SLACK_M;
                }
                let n = seq.len();
                for (i, (s, p)) in seq.iter_mut().enumerate() {
                    if i == 0 || i + 1 == n {
                        continue;
                    }
                    p.level = runs.get(cuts.iter().filter(|&&c| c < *s).count()).map(|r| r.2).unwrap_or(lv_b);
                }
                pis = seq.into_iter().map(|(_, p)| p).collect();
            }
            let e = net.alloc_edge();
            if net.set_edge(e, Some(EdgeData { a, b, tracks: 2, pis, built: true, off: false, thru: vec![], crossover: false, kind: TrackKind::from_hsr(edge_hsr), sep: true, day: 0, oneway: 0, join: [0; 2] })).is_err() {
                refused += 1;
            }
            edge_ids.push(e);
        }
        if refused > 0 {
            eprintln!("{refused} edges refused");
        }
        for (li, l) in lines.iter().enumerate() {
            let ml = &m["lines"][li];
            if ml.is_null() {
                continue;
            }
            let stops: Vec<u32> = ml["stops"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let path: Vec<u32> = ml["path"].as_array().unwrap().iter().map(|v| {
                let c = v.as_u64().unwrap() as u32;
                edge_ids[(c >> 1) as usize] << 1 | (c & 1)
            }).collect();
            let tph: Vec<f32> = ml["tph"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let cars = l["cars"].as_u64().unwrap().clamp(1, 20) as u8;
            let trains: Option<[u32; 3]> = l["trains"].as_array().map(|a| [0, 1, 2].map(|i| a[i].as_u64().unwrap() as u32));
            let id = net.alloc_line();
            let name = format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap());
            let name = names::tidy_line(&name, "").unwrap_or(name);
            let colour = l["colour"].as_str().and_then(|s| u32::from_str_radix(s.trim_start_matches('#'), 16).ok()).unwrap_or(PALETTE[li % PALETTE.len()]);
            let r = net.set_line(id, Some(LineData { symbol: String::new(), trains, auto: None, name, colour, stops, path, tph: [tph[0], tph[1], tph[2]], dwell_s: 30.0, turnaround_s: 180.0, cars, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0 }));
            if r.is_ok() {
                n_lines_ok += 1;
            }
            line_ids.push((li, id));
        }
    }
    for (li, (l, g)) in lines.iter().zip(geoms).enumerate() {
        if shared {
            break;
        }
        let stops: Vec<usize> = l["stops"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let xy = |p: &Value| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        // T-1303: the line routed over the real track (`realnet.py route`): its stations where the
        // routes reach the track, the track's heading there, per hop the real line or null, and a
        // level chosen on that geometry
        let route = route_j.as_ref().map(|r| &r["lines"][li]);
        let level = route.unwrap_or(g)["level"].as_i64().unwrap() as i8;
        let pts: Vec<(f64, f64)> = route.unwrap_or(g)["nodes"].as_array().unwrap().iter().map(xy).collect();
        let cars = l["cars"].as_u64().unwrap().clamp(1, 20) as u8;
        // T-1625: a Shinkansen route has high-speed track (double, traced at the 320 km/h design
        // radius) and runs high-speed trains; a line with whole trains set keeps them
        let hsr = l["hsr"].as_bool().unwrap_or(false);
        let design = if hsr { design_radius(320.0) } else { R_CAP };
        let trains: Option<[u32; 3]> = l["trains"].as_array().map(|a| [0, 1, 2].map(|i| a[i].as_u64().unwrap() as u32));
        let platform = ((cars as u16 * CAR_LEN as u16).clamp(PLATFORM_MIN, PLATFORM_MAX) / PLATFORM_STEP) * PLATFORM_STEP;
        let n = stops.len();
        // headings: the track's through each stop, else along the line through it
        let head: Vec<f64> = (0..n)
            .map(|k| {
                if let Some(h) = route.and_then(|r| r["heads"][k].as_f64()) {
                    return h;
                }
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
            let raw = s["name"].as_str().unwrap();
            let name = names::tidy_station(raw).unwrap_or_else(|| raw.to_string());
            net.set_station(id, Some(StationData { platform, name: name.clone(), named: false, built: true, off: false, upgrades: 0, berth: None, day: 0 })).unwrap();
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
            let lead_a = (ax + lead * head[k].cos(), ay + lead * head[k].sin());
            let lead_b = (bx - lead * head[k + 1].cos(), by - lead * head[k + 1].sin());
            let mut pis = vec![Pi::new(lead_a.0, lead_a.1, 0.0, level).quantised()];
            // T-1303: the real line between the two leads, traced as the track tool traces a snap
            let real = route.and_then(|r| r["hops"][k].as_array()).map(|h| h.iter().map(xy).collect::<Vec<_>>());
            let traced = real.and_then(|h| {
                let mut line = vec![lead_a];
                let (mut at, total) = (0.0, h.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum::<f64>());
                for (i, &p) in h.iter().enumerate() {
                    if i > 0 {
                        at += (p.0 - h[i - 1].0).hypot(p.1 - h[i - 1].1);
                    }
                    if at > lead + 10.0 && at < total - lead - 10.0 {
                        line.push(p);
                    }
                }
                line.push(lead_b);
                let t = std::time::Instant::now();
                // the design radius at 160 km/h, or at 320 for high-speed track (T-1290)
                let r = trace::trace(&line, false, TRACE_TOL_M, TRACE_MIN_RADIUS, design, level);
                trace_s += t.elapsed().as_secs_f64();
                r
            });
            match traced {
                Some(t) => {
                    n_traced += 1;
                    worst_fit = worst_fit.max(t.worst);
                    pis.extend(t.pis);
                }
                None => {
                    n_untraced += 1;
                    for p in g["hops"][k].as_array().unwrap() {
                        let (x, y) = xy(p);
                        if (x - ax).hypot(y - ay) > lead + 30.0 && (x - bx).hypot(y - by) > lead + 30.0 {
                            pis.push(Pi::new(x, y, 0.0, level).quantised());
                        }
                    }
                }
            }
            pis.push(Pi::new(lead_b.0, lead_b.1, 0.0, level).quantised());
            let e = net.alloc_edge();
            net.set_edge(e, Some(EdgeData { a, b, tracks: 2, pis, built: true, off: false, thru: vec![], crossover: false, kind: trainworld_sim::track::params::TrackKind::from_hsr(hsr), sep: true, day: 0, oneway: 0, join: [0; 2] })).unwrap();
            path.push(e << 1);
        }
        let tph: Vec<f32> = l["tph"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let id = net.alloc_line();
        let name = format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap());
        let name = names::tidy_line(&name, "").unwrap_or(name);
        // the line's real colour ("#rrggbb", `pipeline/realcolour.py`), else the palette
        let colour = l["colour"].as_str().and_then(|s| u32::from_str_radix(s.trim_start_matches('#'), 16).ok()).unwrap_or(PALETTE[li % PALETTE.len()]);
        let r = net.set_line(id, Some(LineData { symbol: String::new(), trains, auto: None, name, colour, stops: nodes.clone(), path, tph: [tph[0], tph[1], tph[2]], dwell_s: 30.0, turnaround_s: 180.0, cars, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0 }));
        if r.is_ok() {
            n_lines_ok += 1;
        }
        line_ids.push((li, id));
    }
    // T-3272: the game's own rule for a line on high-speed track (clock.worker's `hsrShare`): a line
    // runs high-speed trains when over half its length is on high-speed track
    let mut n_fast = 0;
    for &(_, id) in &line_ids {
        let (mut all, mut hsr) = (0.0, 0.0);
        for &p in net.line_path(id) {
            let Some(e) = net.edge_data(p >> 1) else { continue };
            let m: f64 = e.pis.windows(2).map(|w| (w[1].x - w[0].x).hypot(w[1].y - w[0].y)).sum();
            all += m;
            if e.kind == TrackKind::from_hsr(true) {
                hsr += m;
            }
        }
        if all > 0.0 && hsr > all * 0.5 {
            let mut d = net.line_data(id).unwrap();
            d.fast = true;
            net.set_line(id, Some(d)).unwrap();
            n_fast += 1;
        }
    }
    eprintln!("lines with over half their length on high-speed track, set to high-speed trains: {n_fast}");
    // T-3233: a real terminus has the tracks to turn trains on both its platforms, so every dead-end
    // station is given a terminal (without one it turns one train at a time, T-2986)
    let mut terminals = 0;
    for n in 0..net.node_count() as u32 {
        if net.node_ok(n) && net.is_station(n) && net.built_ports(n) <= 1 {
            let mut d = net.station_data(n).unwrap();
            if d.upgrades & upgrades::TERMINAL == 0 {
                d.upgrades |= upgrades::TERMINAL;
                net.set_station(n, Some(d)).unwrap();
                terminals += 1;
            }
        }
    }
    // T-3233: settled in the game's Mercator frame and saved in it, so the game loads the save
    // without converting it (a conversion drops the settled rates and the game settles again from
    // scratch, 40 s and more for London). Ids are kept; outputs keep the pack's local frame.
    let local_frame = net.frame();
    let net = save::decode(&save::convert_frame(&save::encode(&net), FrameKind::Merc).expect("the save converts to the game's frame")).unwrap();
    let merc_frame = net.frame();
    let local = |x: f64, y: f64| {
        let (lon, lat) = merc_frame.to_lonlat(x, y);
        local_frame.from_lonlat(lon, lat)
    };
    let t0 = std::time::Instant::now();
    let mut w = TrackWorld::new(net);
    eprintln!("derive {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
    w.allocate_unallocated();
    let bytes = save::encode(&w.net);
    std::fs::write(format!("{out_dir}/{city}_real.twt2"), &bytes).unwrap();
    eprintln!("terminals given to {terminals} dead-end stations");
    eprintln!("save: {} bytes, {} nodes, {} lines set ({} ok); load + derive {:.0} ms", bytes.len(), node_key.len(), lines.len(), n_lines_ok, t0.elapsed().as_secs_f64() * 1000.0);
    eprintln!("hops traced on real track {n_traced} (fit within {worst_fit:.1} m), from a shape or straight {n_untraced}; traced in {:.0} ms", trace_s * 1000.0);
    eprintln!("level changes: {} ramps with a new corner at each end, {} with one new corner, {} between existing corners, {} with one new corner near a curve, {} with corners anyway (no room)", ramps_by_how[0], ramps_by_how[1], ramps_by_how[2], ramps_by_how[3], ramps_by_how[4]);
    let issues = w.net.validate(&(0..w.net.edge_count() as u32).collect::<Vec<_>>(), &(0..w.net.node_count() as u32).collect::<Vec<_>>(), &[]);
    let mut kinds = std::collections::BTreeMap::new();
    for i in &issues {
        *kinds.entry(format!("{:?}", i).split(|c| c == ' ' || c == '{' || c == '(').next().unwrap().to_string()).or_insert(0) += 1;
    }
    eprintln!("validation (not checked on load): {kinds:?}");
    let flat = w.net.crossings.iter().filter(|c| matches!(c.kind, trainworld_sim::track::net::CrossKind::Flat(_))).count();
    eprintln!("crossings {} ({} flat), total cost US${:.0}M", w.net.crossings.len(), flat, w.net.total_cost());
    // T-1305: a line turning back partway (arriving and leaving a node on one side); real service has none
    for &(li, id) in &line_ids {
        let r = w.net.path_reversals(w.net.line_path(id));
        if !r.is_empty() {
            eprintln!("line {li} ({}) turns back partway at path nodes {r:?}", lines[li]["name"].as_str().unwrap_or(""));
        }
    }
    // where trains slow most: per edge, seconds a train loses to curves and caps below the top
    // speed over its pieces, times the lines on it (T-1305 stage 3)
    if args.iter().any(|a| a == "--slow") {
        let mut on = vec![0u32; w.net.edge_count()];
        for &(_, id) in &line_ids {
            for &p in w.net.line_path(id) {
                on[(p >> 1) as usize] += 1;
            }
        }
        let mut lost: Vec<(f64, usize, f64, f64)> = vec![];
        for e in 0..w.net.edge_count() {
            if !w.net.edge_alive[e] || on[e] == 0 {
                continue;
            }
            let cap = w.net.v_cap(e as u32);
            let (mut s, mut vmin) = (0.0, f64::INFINITY);
            for pc in w.net.edge_pieces(e as u32) {
                let v = pc.v_curve().min(cap).min(V_TOP);
                vmin = vmin.min(v);
                s += pc.len * (1.0 / v - 1.0 / V_TOP);
            }
            lost.push((s * on[e] as f64, e, vmin * 3.6, w.net.edge_len[e]));
        }
        lost.sort_by(|a, b| b.0.total_cmp(&a.0));
        // the capacity pass's holds, per demand level, against the free run
        let (mut free, mut held) = (0.0, [0.0f64; 3]);
        let mut worst: Vec<(f64, usize)> = vec![];
        let mut at_node: std::collections::HashMap<u32, f64> = Default::default();
        for &(li, id) in &line_ids {
            let svc = &w.svc.lines[id as usize];
            if !svc.running {
                continue;
            }
            let r = &svc.runs[0];
            let end = |p: &trainworld_sim::track::profile::Profile| p.arr.last().copied().unwrap_or(0.0) - p.dep.first().copied().unwrap_or(0.0);
            free += end(&r.free);
            for lev in 0..3 {
                held[lev] += end(&r.prof[lev]);
            }
            worst.push((end(&r.prof[0]) - end(&r.free), li));
            for h in &r.holds[0] {
                let k = r.node_s.partition_point(|&s| s <= h.s).saturating_sub(1);
                *at_node.entry(r.nodes[k]).or_insert(0.0) += h.delay;
            }
        }
        let cap = &w.svc.cap;
        // T-1774: every line's holds and every loaded resource, for a closer look (`<city>_holds.json`)
        let line_rows: Vec<Value> = line_ids.iter().filter(|&&(_, id)| w.svc.lines[id as usize].running).map(|&(li, id)| {
            let r = &w.svc.lines[id as usize].runs[0];
            let end = |p: &trainworld_sim::track::profile::Profile| p.arr.last().copied().unwrap_or(0.0) - p.dep.first().copied().unwrap_or(0.0);
            let held: Vec<f64> = (0..3).map(|lev| (end(&r.prof[lev]) - end(&r.free)) / 60.0).collect();
            let at: Vec<Value> = r.holds[0].iter().map(|h| {
                let k = r.node_s.partition_point(|&s| s <= h.s).saturating_sub(1);
                let n = r.nodes[k] as usize;
                let (x, y) = local(w.net.node_x[n], w.net.node_y[n]);
                json!([node_names[n], x.round(), y.round(), (h.delay * 10.0).round() / 10.0, h.s.round()])
            }).collect();
            json!({"index": li, "name": lines[li]["name"], "feed": lines[li]["feed"], "operator": lines[li]["operator"], "free_min": end(&r.free) / 60.0, "held_min": held, "holds": at})
        }).collect();
        let res_rows: Vec<Value> = (0..cap.res.len()).filter(|&r| cap.rho[r][0] > 0.5).map(|r| {
            let x = &cap.res[r];
            let names: std::collections::BTreeSet<String> = cap.users_of(r).iter().filter_map(|u| line_ids.iter().find(|l| l.1 == u.line).map(|l| lines[l.0]["name"].as_str().unwrap_or("").to_string())).collect();
            let (lx, ly) = local(x.x, x.y);
            json!({"kind": format!("{:?}", x.kind), "x": lx, "y": ly, "rho": cap.rho[r], "users": cap.users_of(r).len(), "lines": names})
        }).collect();
        std::fs::write(format!("{out_dir}/{city}_holds.json"), serde_json::to_string(&json!({"lines": line_rows, "resources": res_rows})).unwrap()).unwrap();
        for lev in 0..3 {
            let mut kinds = std::collections::BTreeMap::new();
            for (r, x) in cap.res.iter().enumerate() {
                if cap.rho[r][lev] > 0.9 {
                    *kinds.entry(format!("{:?}", x.kind)).or_insert(0) += 1;
                }
            }
            eprintln!("  level {lev}: resources over 90%: {kinds:?}");
        }
        let mut over: Vec<(f64, usize)> = (0..cap.res.len()).map(|r| (cap.rho[r][0], r)).filter(|x| x.0 > 0.9).collect();
        over.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (rho, r) in over.iter().take(12) {
            let x = &cap.res[*r];
            let names: std::collections::BTreeSet<String> = cap.users_of(*r).iter().filter_map(|u| line_ids.iter().find(|l| l.1 == u.line).map(|l| lines[l.0]["name"].as_str().unwrap_or("").to_string())).collect();
            let (lx, ly) = local(x.x, x.y);
            eprintln!("  {:?} at {lx:.0},{ly:.0} high {:.2}: {:?}", x.kind, rho, names);
        }
        let mut hot: Vec<(u32, f64)> = at_node.into_iter().collect();
        hot.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (n, d) in hot.iter().take(15) {
            let (x, y) = local(w.net.node_x[*n as usize], w.net.node_y[*n as usize]);
            eprintln!("  holds after node {n} ({}) at {x:.0},{y:.0}: {:.1} line-min (high demand)", node_names[*n as usize], d / 60.0);
        }
        eprintln!("slow: one-way minutes over lines, free run {:.0}, with the capacity pass's holds {:.0} / {:.0} / {:.0} (high, medium, low demand)", free / 60.0, held[0] / 60.0, held[1] / 60.0, held[2] / 60.0);
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (d, li) in worst.iter().take(10) {
            eprintln!("  held {:.1} min: line {li} {}", d / 60.0, lines[*li]["name"].as_str().unwrap_or(""));
        }
        eprintln!("slow: {:.0} line-minutes lost to curves in all", lost.iter().map(|l| l.0).sum::<f64>() / 60.0);
        for l in lost.iter().take(25) {
            let (a, b) = (w.net.edge_a[l.1], w.net.edge_b[l.1]);
            let (x, y) = local(w.net.node_x[a as usize], w.net.node_y[a as usize]);
            eprintln!("  edge {} ({}-{}) {:.0} line-s, slowest {:.0} km/h, {:.0} m, crossover {}, at {x:.0},{y:.0}", l.1, a, b, l.0, l.2, l.3, w.net.edge_crossover[l.1]);
        }
    }
    let live: Vec<usize> = (0..w.net.edge_count()).filter(|&e| w.net.edge_alive[e]).collect();
    eprintln!("{} edges, {:.0} km of double track", live.len(), live.iter().map(|&e| w.net.edge_len[e]).sum::<f64>() / 1000.0);

    // the snapshot (as clock.worker.ts buildSnapshot + demandClient.ts packService do)
    let mut st_xy = vec![];
    let mut st_out = vec![];
    let mut node_st = vec![u32::MAX; w.net.node_count()];
    for n in 0..w.net.node_count() {
        node_st[n] = (st_xy.len() / 2) as u32;
        let (x, y) = local(w.net.node_x[n], w.net.node_y[n]);
        st_xy.push(x as f32);
        st_xy.push(y as f32);
        st_out.push(json!({"key": node_key[n], "name": node_names[n], "feed": node_key[n].split(':').next().unwrap().replace("mta", "subway"), "x": x, "y": y}));
    }
    let (mut line_n, mut stops, mut times, mut tph, mut cars) = (vec![], vec![], vec![], vec![], vec![]);
    let mut lines_out = vec![];
    let (mut gtfs_s, mut track_s) = (0.0, 0.0);
    let mut not_running = 0;
    let mut fleet = 0u64;
    for &(li, id) in &line_ids {
        let svc = &w.svc.lines[id as usize];
        if !svc.running {
            not_running += 1;
            let bad: Vec<u32> = w.net.line_path(id).iter().map(|&p| p >> 1).filter(|&e| !w.net.edge_ok(e) || !w.net.edge_built[e as usize]).collect();
            let bad_stops: Vec<u32> = w.net.line_stops(id).iter().copied().filter(|&s| !w.net.node_ok(s) || !w.net.node_built[s as usize]).collect();
            eprintln!("line {li} ({}) not running: edges {bad:?}, stops {bad_stops:?}, path {} edges, {} stops", lines[li]["name"].as_str().unwrap_or(""), w.net.line_path(id).len(), w.net.line_stops(id).len());
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
    std::fs::write(format!("{out_dir}/{city}_real_track.json"), serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("-> {out_dir}/{city}_real.twt2, {out_dir}/{city}_real_track.json");
}
