//! T-276: a day on the real New York network with every layer the game solves there: New York's
//! commutes and everyday purposes, and the regional layer (longer trips, and commutes between
//! zones once the market pack carries them). The before/after check for the zone and intercity
//! changes: each layer's trips and rail, the market's corridors, New York's rail share.
//!
//!   cargo run --release --features demand --bin zones_day -- [--net data/work/realnet/nyc_real_track.json]
//!       [--pack data/packs/nyc.json] [--market data/packs/intercity.json] [--rounds 3]
//!       [--out result.json] [--no-everyday]
//!
//! Fares are the game's starting fares ($1.50 plus $0.10 a km). The New York pack's frame is the
//! market's (both the New York equirectangular frame), so station positions pass through as they are.

use serde_json::{json, Value};
use trainworld_sim::demand::api::DemandApi;
use trainworld_sim::demand::clock::now_ms;
use trainworld_sim::demand::everyday::EverydayApi;
use trainworld_sim::demand::intercity::IntercityApi;
use trainworld_sim::demand::network::PERIODS;
use trainworld_sim::demand::pack;

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let has = |k: &str| args.iter().any(|a| a == k);
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let net_path = get("--net").unwrap_or(format!("{root}/data/work/realnet/nyc_real_track.json"));
    let pack_path = get("--pack").unwrap_or(format!("{root}/data/packs/nyc.json"));
    let market_path = get("--market").unwrap_or(format!("{root}/data/packs/intercity.json"));
    let rounds: usize = get("--rounds").map_or(3, |v| v.parse().unwrap());

    let t0 = now_ms();
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    let mut ev = if has("--no-everyday") { None } else { Some(EverydayApi::new(&h, &b).unwrap()) };
    let mut ic = IntercityApi::new(&std::fs::read_to_string(&market_path).unwrap()).unwrap();
    eprintln!("opened in {:.0} ms; commute info {:?}; market info {:?}", now_ms() - t0, api.info(), ic.info());

    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
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
    let (fare_base, fare_km) = (1.5, 0.1);
    api.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    api.set_fares(fare_base, fare_km, &hop_km).unwrap();
    if let Some(e) = ev.as_mut() {
        e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        e.set_fares(fare_base, fare_km, &hop_km).unwrap();
    }
    ic.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    ic.set_fares(fare_base, fare_km, &hop_km).unwrap();
    eprintln!("market pairs served and all {:?}; commute pairs served and their commuters {:?} of {:.0}", ic.pair_counts(), ic.served_commutes(), ic.commuters());

    let t1 = now_ms();
    let mut periods = vec![];
    let (mut c_trips, mut c_rail, mut e_trips, mut e_rail, mut m_trips, mut m_rail) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut corridors: Vec<f64> = vec![];
    let mut market_parts: Vec<f64> = vec![];
    let mut commute_cor: Vec<f64> = vec![];
    for q in 0..PERIODS {
        api.solve(q);
        if let Some(e) = ev.as_mut() {
            e.solve(q);
        }
        ic.solve(q);
        for _ in 0..rounds {
            let nr = api.seg(q).len();
            let mut total = vec![0f32; 3 * nr];
            let mut add = |v: Vec<f32>, at: usize| {
                for (t, x) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                    *t += x;
                }
            };
            add(api.seg(q), 0);
            add(api.board(q), 1);
            add(api.alight(q), 2);
            if let Some(e) = &ev {
                add(e.seg(q), 0);
                add(e.board(q), 1);
                add(e.alight(q), 2);
            }
            add(ic.seg(q), 0);
            add(ic.board(q), 1);
            add(ic.alight(q), 2);
            assert!(api.crowd_with_external(q, &total));
            if let Some(e) = ev.as_mut() {
                assert!(e.crowd_with_external(q, &total));
            }
            assert!(ic.crowd_with_external(q, &total));
        }
        let s = api.summary(q);
        c_trips += s[1];
        c_rail += s[2];
        let es = ev.as_ref().map(|e| e.summary(q)).unwrap_or_default();
        if !es.is_empty() {
            e_trips += es[1];
            e_rail += es[2];
        }
        let ms = ic.summary(q);
        m_trips += ms[1];
        m_rail += ms[2];
        let cs = ic.corridor_summary(q);
        if corridors.is_empty() {
            corridors = cs.clone();
        } else {
            for (k, v) in cs.iter().enumerate() {
                if k % 4 >= 2 {
                    corridors[k] += v;
                }
            }
        }
        let cc = ic.commute_corridors(q);
        if commute_cor.is_empty() {
            commute_cor = cc.clone();
        } else {
            for (a, b) in commute_cor.iter_mut().zip(&cc) {
                *a += b;
            }
        }
        let mp = ic.part_summary(q);
        if market_parts.is_empty() {
            market_parts = mp.clone();
        } else {
            for (a, b) in market_parts.iter_mut().zip(&mp) {
                *a += b;
            }
        }
        eprintln!("period {q}: commute rail {:.0} of {:.0}, everyday rail {:.0} of {:.0}, market rail {:.0} of {:.0}", s[2], s[1], es.get(2).unwrap_or(&0.), es.get(1).unwrap_or(&0.), ms[2], ms[1]);
        periods.push(json!({"commute": s, "everyday": es, "market": ms, "market_parts": mp}));
    }
    eprintln!("solved in {:.0} ms", now_ms() - t1);
    let local_share = (c_rail + e_rail) / (c_trips + e_trips).max(1e-9);
    println!("New York commutes: {:.0} trips, rail {:.0} ({:.1}%)", c_trips, c_rail, 100.0 * c_rail / c_trips.max(1e-9));
    println!("New York everyday: {:.0} trips, rail {:.0} ({:.1}%)", e_trips, e_rail, 100.0 * e_rail / e_trips.max(1e-9));
    println!("New York local, all purposes: rail {:.1}%", 100.0 * local_share);
    println!("regional layer: {:.0} trips, rail {:.0} ({:.2}%); by part (trips, rail): {:?}", m_trips, m_rail, 100.0 * m_rail / m_trips.max(1e-9), market_parts.iter().map(|v| v.round()).collect::<Vec<_>>());
    println!("corridors with rail (from, to, all trips, rail):");
    for c in corridors.chunks_exact(4) {
        if c[3] >= 1.0 {
            println!("  {} - {}: {:.0}, rail {:.0}", c[0], c[1], c[2], c[3]);
        }
    }
    println!("commutes between zones by corridor (from, to, legs, rail legs):");
    for (c, v) in corridors.chunks_exact(4).zip(commute_cor.chunks_exact(2)) {
        if v[0] >= 1.0 {
            println!("  {} - {}: {:.0}, rail {:.0}", c[0], c[1], v[0], v[1]);
        }
    }
    if let Some(p) = get("--out") {
        let out = json!({"net": net_path, "pack": pack_path, "market": market_path, "rounds": rounds,
            "commute": [c_trips, c_rail], "everyday": [e_trips, e_rail], "market_total": [m_trips, m_rail],
            "market_parts": market_parts, "corridors": corridors, "periods": periods});
        std::fs::write(&p, serde_json::to_string(&out).unwrap()).unwrap();
        eprintln!("-> {p}");
    }
}
