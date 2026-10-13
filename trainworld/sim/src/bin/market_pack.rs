//! T-174: the regional market's all-pairs figures, written into its pack. pipeline/intercity.py
//! runs it after writing the sites; run it alone after changing the market's zone-pair formula
//! (`intercity::FORMULA`), or the game derives them at every opening (same result, seconds).
//!
//!   cargo run --release --features demand --bin market_pack -- data/packs/intercity.json
//!
//! Reads the pack, computes its `derived` block with the game's own code (`intercity::derive_pack`:
//! each zone's gravity normaliser, the day's trips, each corridor's trips; two passes over every
//! zone pair) and writes the pack back with the block replaced. Nothing else changes.

use trainworld_sim::demand::{clock::now_ms, intercity};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = <[String; 1]>::try_from(args).expect("usage: market_pack PACK.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let t0 = now_ms();
    let block = intercity::derive_pack(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    let ms = now_ms() - t0;
    let mut pack: serde_json::Value = serde_json::from_str(&text).unwrap();
    pack["derived"] = serde_json::to_value(&block).unwrap();
    let out = serde_json::to_string(&pack).unwrap();
    // the game opens it with the block in use, or this is wrong
    let api = intercity::IntercityApi::new(&out).unwrap_or_else(|e| panic!("written pack does not open: {e}"));
    let info = api.info();
    assert_eq!(info[3], 1.0, "the written block is not taken as current");
    let part = format!("{path}.part");
    std::fs::write(&part, &out).unwrap();
    std::fs::rename(&part, &path).unwrap();
    println!(
        "{path}: {} sites, {} zones, {} directed pairs, {:.0} trips a day; derived in {ms:.0} ms; {:.2} MB (was {:.2} MB)",
        info[0],
        info[1],
        api.pair_counts()[1],
        info[2],
        out.len() as f64 / 1e6,
        text.len() as f64 / 1e6
    );
}
