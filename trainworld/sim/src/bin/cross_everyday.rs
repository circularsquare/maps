//! T-1163 (T-1175): everyday outings across a zone line for one solve unit (`demand::cross`).
//!
//!   cargo run --release --features demand --bin cross_everyday -- --packs a,b [--fixed c,d]
//!       --out FILE [--root ..] [--packs-dir DIR] [--threads 2]
//!
//! `--packs` are drawn from (their zones' kept shares and the pairs their draws make), `--fixed`
//! only drawn on, as `joint_gravity`'s solve units. Reads `<packs-dir>/<id>.json` and `.bin`
//! (default `<root>/data/packs`) and `<root>/pipeline/borders.json`, writes `--out` (pipeline/
//! cross_everyday.py puts the kept shares into the packs, market_zones.py the pairs into the
//! market packs). Prints, per solved pack, the outings that leave it by purpose.
use serde_json::Value;
use trainworld_sim::demand::{clock::now_ms, cross, everyday::KINDS, pack};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = get("--root").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/..").to_string());
    let dir = get("--packs-dir").unwrap_or(format!("{root}/data/packs"));
    let threads: usize = get("--threads").map_or(2, |v| v.parse().unwrap());
    let list = |k: &str| get(k).map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect::<Vec<_>>()).unwrap_or_default();
    let (solved, fixed) = (list("--packs"), list("--fixed"));
    let out = get("--out").expect("--out FILE");
    assert!(!solved.is_empty(), "--packs a,b");
    let borders: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/pipeline/borders.json")).unwrap()).unwrap();
    let t0 = now_ms();
    let mut packs: Vec<cross::CrossPack> = solved
        .iter()
        .map(|id| (id, true))
        .chain(fixed.iter().map(|id| (id, false)))
        .map(|(id, s)| {
            let (header, bin) = pack::load_raw(&format!("{dir}/{id}.json")).unwrap_or_else(|e| panic!("{e}"));
            cross::CrossPack::new(id, &header, &bin, s).unwrap_or_else(|e| panic!("{id}: {e}"))
        })
        .collect();
    let t1 = now_ms();
    let unit = cross::run(&mut packs, &borders, threads);
    let t2 = now_ms();
    let v = unit.to_json(&packs);
    std::fs::write(&out, serde_json::to_string(&v).unwrap() + "\n").unwrap();
    let names: Vec<&str> = cross::CROSS.iter().map(|&k| KINDS[k]).collect();
    for (pi, p) in packs.iter().enumerate().filter(|(_, p)| p.solved) {
        let (m, o) = (unit.moved[pi], unit.outings[pi]);
        let all: f64 = m.iter().sum::<f64>() / o.iter().sum::<f64>().max(1e-9);
        let by = names.iter().enumerate().map(|(s, n)| format!("{n} {:.0} ({:.2}%)", m[s], 100.0 * m[s] / o[s].max(1e-9))).collect::<Vec<_>>().join(", ");
        eprintln!("{}: {} zones with a kept share under 1; outings leaving {:.2}% ({by})", p.id, unit.kept[pi].len(), 100.0 * all);
    }
    let n = v["trips"].as_array().map_or(0, |a| a.len());
    let t: f64 = v["trips"].as_array().map_or(0.0, |a| a.iter().filter_map(|x| x.as_f64()).sum());
    eprintln!("{n} pairs, {t:.0} outings a day; loaded in {:.0} ms, drawn in {:.0} ms; {out}", t1 - t0, t2 - t1);
}
