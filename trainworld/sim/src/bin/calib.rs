//! T-509: one calibration run. A weekday on a city's real network (`pipeline/realnet.py net
//! <city>`) with every in-pack layer (commutes and the everyday purposes; the regional market
//! off), crowding rounds shared between the two, natively, with a parameter set passed in; it
//! writes what `pipeline/calib.py` scores against the city's targets.
//!
//!   cargo run --release --features demand --bin calib -- --net data/work/realnet/nyc_real.json
//!       --out run.json [--pack data/packs/nyc.json] [--rounds 3] [--areas cell_area.u8]
//!       [--params p.json] [--set name=value,..] [--fares 1.5,0.1 | --no-fares] [--no-everyday]
//!
//! The pack defaults to the network's `pack`. Parameters: `--params` a JSON object of
//! `Params::set_extra` names, then `--set` on top; both layers get them. Fares default to the
//! game's starting fares ($1.50 plus $0.10 a km); `--no-fares` keeps the flat `fare_min`.
//! `--areas`: one value per pack cell, the home area (u8, 255 none; or a .u16 file, 65535 none).
//!
//! Output: per demand station the day's entries (first boardings from the street) by layer and
//! in 6-10; per line the day's boardings and changes between two trains of the same line; per
//! operator (the lines' `operator`) trips using it, boardings less changes within it (T-171); per
//! home area the commutes' rail trips and all trips; totals by layer; milliseconds by stage.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use trainworld_sim::demand::api::DemandApi;
use trainworld_sim::demand::clock::now_ms;
use trainworld_sim::demand::everyday::EverydayApi;
use trainworld_sim::demand::network::PERIODS;
use trainworld_sim::demand::pack;

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_u64().unwrap() as u32).collect()
}
fn r1(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| (x * 10.0).round() / 10.0).collect()
}
fn add(acc: &mut Vec<f64>, v: &[f32]) {
    if acc.len() < v.len() {
        acc.resize(v.len(), 0.0);
    }
    for (a, &x) in acc.iter_mut().zip(v) {
        *a += x as f64;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let has = |k: &str| args.iter().any(|a| a == k);
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let net_path = get("--net").expect("--net <network json>");
    let out_path = get("--out").expect("--out <result json>");
    let rounds: usize = get("--rounds").map_or(3, |v| v.parse().unwrap());
    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let pack_path = get("--pack").unwrap_or_else(|| format!("{root}/data/packs/{}.json", net["pack"].as_str().unwrap_or("nyc")));

    // the parameter set: --params JSON, then --set
    let mut set: BTreeMap<String, f32> = BTreeMap::new();
    if let Some(p) = get("--params") {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        for (k, x) in v.as_object().expect("--params: a JSON object") {
            set.insert(k.clone(), x.as_f64().expect("--params: numbers") as f32);
        }
    }
    if let Some(v) = get("--set") {
        for kv in v.split(',') {
            let (k, x) = kv.split_once('=').expect("--set name=value");
            set.insert(k.to_string(), x.parse().unwrap());
        }
    }

    let t0 = now_ms();
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    for (k, &x) in &set {
        api.params_mut().set_extra(k, x).unwrap_or_else(|e| panic!("{e}"));
    }
    if set.contains_key("walk_d0_km") || set.contains_key("walk_s_km") {
        api.refresh_walk();
    }
    // the everyday layer takes the commutes' parameters and walk curve as they are now
    let mut ev = if has("--no-everyday") { None } else { Some(EverydayApi::beside(&api, &h, &b).unwrap()) };
    let t_open = now_ms() - t0;

    let d = &net["demand"];
    let (st_xy, line_n, stops, times, tph, cars) = (arr_f32(d, "st_xy"), arr_u32(d, "line_n"), arr_u32(d, "stops"), arr_f32(d, "times"), arr_f32(d, "tph"), arr_f32(d, "cars"));
    // each route node's hop to the next stop, km (the fare's distance), kernel order
    let mut hop_km = vec![];
    let mut off = 0usize;
    for &n in &line_n {
        let n = n as usize;
        for dir in 0..2 {
            for i in 0..n {
                let k = if dir == 0 { off + i } else { off + n - 1 - i };
                hop_km.push(if i + 1 < n {
                    let nx = if dir == 0 { k + 1 } else { k - 1 };
                    let (s0, s1) = (stops[k] as usize, stops[nx] as usize);
                    (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                } else {
                    0.0
                });
            }
        }
        off += n;
    }
    let t1 = now_ms();
    api.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    if let Some(e) = ev.as_mut() {
        e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    }
    let fares: Option<(f32, f32)> = if has("--no-fares") {
        None
    } else {
        let v = get("--fares").unwrap_or("1.5,0.1".into());
        let (a, k) = v.split_once(',').expect("--fares base,per_km");
        Some((a.parse().unwrap(), k.parse().unwrap()))
    };
    if let Some((base, km)) = fares {
        api.set_fares(base, km, &hop_km).unwrap();
        if let Some(e) = ev.as_mut() {
            e.set_fares(base, km, &hop_km).unwrap();
        }
    }
    let t_net = now_ms() - t1;

    let areas: Option<(Vec<u32>, u32)> = get("--areas").map(|p| {
        let raw = std::fs::read(&p).unwrap();
        if p.ends_with(".u16") {
            (raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(), 65535)
        } else {
            (raw.iter().map(|&v| v as u32).collect(), 255)
        }
    });
    let n_cells = api.trips_by_cell(0).len();
    if let Some((a, _)) = &areas {
        assert_eq!(a.len(), n_cells, "--areas has {} values, the pack {} cells", a.len(), n_cells);
    }
    let n_area = areas.as_ref().map_or(0, |(a, none)| a.iter().filter(|&v| v != none).map(|&v| v as usize + 1).max().unwrap_or(0));

    let t2 = now_ms();
    let (mut ent_c, mut ent_e, mut ent_am) = (vec![], vec![], vec![]);
    let (mut board, mut area_rail, mut area_trips) = (vec![], vec![0f64; n_area], vec![0f64; n_area]);
    let (mut c_tot, mut e_tot) = ([0f64; 3], [0f64; 3]);
    let mut kinds: Vec<f64> = vec![];
    let mut periods = vec![];
    for q in 0..PERIODS {
        api.solve(q);
        if let Some(e) = ev.as_mut() {
            e.solve(q);
        }
        for _ in 0..rounds {
            let nr = api.seg(q).len();
            let mut total = vec![0f32; 3 * nr];
            let mut put = |v: Vec<f32>, at: usize| {
                for (t, x) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                    *t += x;
                }
            };
            put(api.seg(q), 0);
            put(api.board(q), 1);
            put(api.alight(q), 2);
            if let Some(e) = &ev {
                put(e.seg(q), 0);
                put(e.board(q), 1);
                put(e.alight(q), 2);
            }
            assert!(api.crowd_with_external(q, &total));
            if let Some(e) = ev.as_mut() {
                assert!(e.crowd_with_external(q, &total));
            }
        }
        let s = api.summary(q);
        let es = ev.as_ref().map(|e| e.summary(q)).unwrap_or_default();
        for i in 0..3 {
            c_tot[i] += s[i + 1];
            e_tot[i] += es.get(i + 1).copied().unwrap_or(0.0);
        }
        if let Some(e) = &ev {
            let kt = e.kind_totals(q);
            if kinds.is_empty() {
                kinds = vec![0.0; kt.len()];
            }
            for (a, x) in kinds.iter_mut().zip(&kt) {
                *a += x;
            }
        }
        add(&mut ent_c, &api.entries(q));
        add(&mut board, &api.board(q));
        if let Some(e) = &ev {
            add(&mut ent_e, &e.entries(q));
            add(&mut board, &e.board(q));
        }
        if q == 0 {
            add(&mut ent_am, &api.entries(q));
            if let Some(e) = &ev {
                add(&mut ent_am, &e.entries(q));
            }
        }
        if let Some((a, none)) = &areas {
            let (rail, trips) = (api.rail_by_cell(q), api.trips_by_cell(q));
            for (c, &k) in a.iter().enumerate() {
                if k != *none && c < rail.len() {
                    area_rail[k as usize] += rail[c] as f64;
                    area_trips[k as usize] += trips[c] as f64;
                }
            }
        }
        periods.push(json!({"commute": s, "everyday": es}));
    }
    let t_solve = now_ms() - t2;

    // changes between two trains: same line, and same operator
    let lines = net["lines"].as_array().unwrap();
    let op_of: Vec<String> = lines.iter().map(|l| l["operator"].as_str().or(l["feed"].as_str()).unwrap_or("?").to_string()).collect();
    let rn_line: Vec<usize> = api.route_nodes(0).1.iter().map(|&l| l as usize).collect();
    assert_eq!(rn_line.len(), board.len());
    let sel = vec![1u8; rn_line.len()];
    let mut xf = api.transfers(&sel);
    if let Some(e) = &ev {
        xf.extend(e.transfers(&sel));
    }
    let n_lines = line_n.len();
    let mut same_line = vec![0f64; n_lines];
    let mut ops: BTreeMap<String, f64> = BTreeMap::new();
    let mut line_board = vec![0f64; n_lines];
    for (r, &v) in board.iter().enumerate() {
        line_board[rn_line[r]] += v;
        *ops.entry(op_of[rn_line[r]].clone()).or_default() += v;
    }
    for t in xf.chunks_exact(3) {
        let (li, lo) = (rn_line[t[0] as usize], rn_line[t[1] as usize]);
        if li == lo {
            same_line[li] += t[2] as f64;
        }
        if op_of[li] == op_of[lo] {
            *ops.get_mut(&op_of[li]).unwrap() -= t[2] as f64;
        }
    }
    let t_all = now_ms() - t0;
    eprintln!(
        "commutes {:.0} trips, rail {:.0} ({:.1}%); everyday {:.0}, rail {:.0} ({:.1}%); open {:.1} s, network {:.1} s, solve {:.1} s, all {:.1} s",
        c_tot[0], c_tot[1], 100.0 * c_tot[1] / c_tot[0].max(1e-9), e_tot[0], e_tot[1], 100.0 * e_tot[1] / e_tot[0].max(1e-9),
        t_open / 1e3, t_net / 1e3, t_solve / 1e3, t_all / 1e3
    );
    eprintln!("operators: {}", ops.iter().map(|(k, v)| format!("{k} {:.0}", v)).collect::<Vec<_>>().join(", "));
    let mut set_out = Map::new();
    for (k, v) in &set {
        set_out.insert(k.clone(), json!(v));
    }
    let out = json!({
        "net": net_path,
        "pack": pack_path,
        "set": set_out,
        "fares": fares.map(|(a, k)| vec![a, k]),
        "rounds": rounds,
        "params": format!("{:?}", api.params_mut()),
        "ms": {"open": t_open, "network": t_net, "solve": t_solve, "all": t_all},
        "day": {"commute": c_tot, "everyday": e_tot, "everyday_kinds": kinds},
        "st_map": api.station_map(),
        "entries": {"commute": r1(&ent_c), "everyday": r1(&ent_e), "am": r1(&ent_am)},
        "lines": {"board": r1(&line_board), "same_line_changes": r1(&same_line)},
        "operators": ops,
        "areas": {"rail": r1(&area_rail), "trips": r1(&area_trips)},
        "periods": periods,
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("-> {out_path}");
}
