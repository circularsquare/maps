//! T-377: the world market's registry rows from every place's pack (`data/packs/<id>.market.json`),
//! with the game's own code (`intercity::world_rows`): each place's sums of its zones, its centre,
//! and the trips to every place whose centre is under `near_km` (every 6 km zone pair added up).
//! pipeline/market_zones.py runs it and merges the rows into app/src/game/market-rows.json.
//!
//!   cargo run --release --features demand --bin market_pack -- --root . --out data/work/market_rows.json [--only FILE]
//!
//! It also prints the trips of a few pairs of places (all, and T-299's part over 80 km zone to zone).
//! `--only` (T-929): a JSON `{"rows": [ids], "compose": [ids]}`; only the `rows` zones' rows, from a
//! composition of the `compose` zones (market_zones.py names every zone those rows read), checked
//! against market-rows.json's other rows as the game opens them; no pairs table.

use serde_json::{json, Value};
use std::collections::HashSet;
use trainworld_sim::demand::{clock::now_ms, intercity};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = get("--root").unwrap_or_else(|| ".".into());
    let out = get("--out").expect("usage: market_pack --root DIR --out FILE [--only FILE]");
    let reg: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let only: Option<(HashSet<String>, HashSet<String>)> = get("--only").map(|f| {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let set = |k: &str| v[k].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_owned()).collect::<HashSet<_>>();
        (set("rows"), set("compose"))
    });
    let market = &reg["market"];
    let places = market["places"].as_array().unwrap().len();
    let mut packs = vec![];
    let mut ids = vec![];
    for c in reg["cities"].as_array().unwrap() {
        let id = c["id"].as_str().unwrap();
        let Some(place) = c["place"].as_u64() else { continue };
        if only.as_ref().is_some_and(|(r, s)| !r.contains(id) && !s.contains(id)) {
            continue;
        }
        let path = format!("{root}/data/packs/{id}.market.json");
        match std::fs::read_to_string(&path) {
            Ok(t) => {
                packs.push(t);
                ids.push((id.to_owned(), place as usize));
            }
            Err(e) => eprintln!("{path}: {e}; place {place} left out"),
        }
    }
    let want: Option<Vec<bool>> = only.as_ref().map(|(r, _)| {
        let mut w = vec![false; places];
        for (id, p) in &ids {
            w[*p] = r.contains(id);
        }
        w
    });
    let world = json!({"frame": market["frame"], "rule": market["rule"], "places": places});
    let text = format!("{{\"world\":{},\"zones\":[{}]}}", world, packs.join(","));
    let t0 = now_ms();
    let rows = intercity::world_rows_for(&text, want.as_deref()).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{} places' rows in {:.0} ms", rows.len(), now_ms() - t0);
    let by_id: serde_json::Map<String, Value> = rows
        .iter()
        .map(|r| {
            let v = serde_json::to_value(r).unwrap();
            (v["zone"].as_str().unwrap().to_owned(), v)
        })
        .collect();
    if let Some(w) = &want {
        // the rows open as the game opens them: every row (the others as market-rows.json has them)
        // with the new rows' packs
        let mut all: serde_json::Map<String, Value> = std::fs::read_to_string(format!("{root}/app/src/game/market-rows.json"))
            .map(|t| serde_json::from_str(&t).unwrap())
            .unwrap_or_default();
        for (k, v) in &by_id {
            all.insert(k.clone(), v.clone());
        }
        let mine: Vec<&str> = packs.iter().zip(&ids).filter(|(_, (_, p))| w[*p]).map(|(t, _)| t.as_str()).collect();
        let with = json!({"frame": market["frame"], "rule": market["rule"], "places": places, "rows": all.values().collect::<Vec<_>>()});
        let full = format!("{{\"world\":{},\"zones\":[{}]}}", with, mine.join(","));
        let api = intercity::IntercityApi::new(&full).unwrap_or_else(|e| panic!("the rows do not open: {e}"));
        assert_eq!(api.info()[3], 1.0);
        println!("{} rows from {} packs", by_id.len(), packs.len());
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"rows": by_id, "pairs": []})).unwrap()).unwrap();
        println!("wrote {out}");
        return;
    }
    // the rows open as the game opens them: every place's pack with the world's rows
    let with = json!({"frame": market["frame"], "rule": market["rule"], "places": places, "rows": rows});
    let full = format!("{{\"world\":{},\"zones\":[{}]}}", with, packs.join(","));
    let api = intercity::IntercityApi::new(&full).unwrap_or_else(|e| panic!("the rows do not open: {e}"));
    let info = api.info();
    assert_eq!(info[3], 1.0);
    println!("{} sites, {} zones, {:.0} trips a day between places, {:.0} departures a day to places with no zone", info[0], info[1], info[2], info[4]);
    let place = |id: &str| ids.iter().find(|(i, _)| i == id).map(|(_, p)| *p);
    let names = [("nyc", "phl"), ("nyc", "bos"), ("nyc", "wdc"), ("bos", "wdc"), ("wdc", "bal"), ("phl", "wdc"), ("nyc", "nhv"), ("nhv", "hfd"), ("bos", "pvd"), ("nyc", "tor"), ("lon", "nyc"), ("lon", "tor"), ("lon", "par"), ("par", "ber"), ("lon", "ber"), ("sha", "tyo"), ("mum", "lon"), ("nyc", "nyc"), ("lon", "lon")];
    let pairs: Vec<(usize, usize)> = names.iter().filter_map(|(a, b)| Some((place(a)?, place(b)?))).collect();
    let trips = api.place_trips(&pairs, 80.);
    // T-378: each corridor's shares of its trips by plane and by car with no rail
    let modes = api.corridor_modes();
    let corridor = |a: usize, b: usize| {
        let (a, b) = (a.min(b), a.max(b));
        // corridors run (0, 0), (0, 1) .. (0, n - 1), (1, 1) ..
        a * places - a * a.saturating_sub(1) / 2 + (b - a)
    };
    let mut table = vec![];
    for (((a, b), (all, over)), &(pa, pb)) in names.iter().filter(|(a, b)| place(a).is_some() && place(b).is_some()).zip(&trips).zip(&pairs) {
        let c = corridor(pa, pb);
        let (air, car) = (modes[2 * c], modes[2 * c + 1]);
        println!("  {a} - {b}: {:.0} trips a day both ways, {:.0} of them zone to zone over 80 km; with no rail {:.1}% fly, {:.1}% drive", 2. * all, 2. * over, 100. * air, 100. * car);
        table.push(json!({"a": a, "b": b, "both_ways": 2. * all, "over_80km": 2. * over, "air": air, "car": car}));
    }
    let doc = json!({"rows": by_id, "pairs": table});
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out}");
}
