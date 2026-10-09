//! T-005 demand kernel benchmark, native.
//!
//!   cargo run --release --features demand --bin demand_bench -- [--pack data/packs/nyc.json]
//!       [--zone-m 2000] [--band 0] [--exact] [--sweep] [--no-edit] [--no-periods]
//!       [--rounds 1] [--prune 0] [--write-synth DIR] [--solve] [--theta 0.2]
//!       [--walk-m 3000] [--set name=value,..]
//!
//! Without --pack it generates the synthetic New York-sized city (and with --write-synth writes it
//! as a format-1 pack without gravity, then reloads it through the loader). --solve solves the
//! zone gravity even when the pack ships it (the city open before T-019).

use trainworld_sim::demand::{bench, clock::now_ms, network, pack, synth};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut o = bench::Opts::default();
    let mut pack_path: Option<String> = None;
    let mut write_synth: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().expect("missing value");
        match args[i].as_str() {
            "--pack" => {
                pack_path = Some(next(i));
                i += 1;
            }
            "--zone-m" => {
                o.zone_m = next(i).parse().unwrap();
                i += 1;
            }
            "--band" => {
                o.band_min = next(i).parse().unwrap();
                i += 1;
            }
            "--rounds" => {
                o.crowd_rounds = next(i).parse().unwrap();
                i += 1;
            }
            "--asc" => {
                o.asc_rail = next(i).parse().unwrap();
                i += 1;
            }
            "--prune" => {
                o.prune_trips = next(i).parse().unwrap();
                i += 1;
            }
            "--write-synth" => {
                write_synth = Some(next(i));
                i += 1;
            }
            "--theta" => {
                o.theta = Some(next(i).parse().unwrap());
                i += 1;
            }
            "--walk-m" => {
                o.walk_cutoff_m = Some(next(i).parse().unwrap());
                i += 1;
            }
            "--set" => {
                for kv in next(i).split(',') {
                    let (k, v) = kv.split_once('=').expect("--set name=value");
                    o.sets.push((k.to_string(), v.parse().unwrap()));
                }
                i += 1;
            }
            "--exact" => o.exact = true,
            "--solve" => o.solve_gravity = true,
            "--sweep" => {
                o.sweep = true;
                o.exact = true;
            }
            "--no-edit" => o.edit = false,
            "--no-periods" => o.periods = false,
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    let t0 = now_ms();
    let city = match &pack_path {
        Some(pth) => pack::load(pth).unwrap_or_else(|e| panic!("{e}")),
        None => {
            let c = synth::synth_city(5);
            if let Some(dir) = &write_synth {
                let (h, b) = pack::write(&c);
                std::fs::create_dir_all(dir).unwrap();
                let jp = format!("{dir}/synthetic-ny.json");
                std::fs::write(&jp, h).unwrap();
                std::fs::write(format!("{dir}/synthetic-ny.bin"), b).unwrap();
                let back = pack::load(&jp).unwrap();
                assert_eq!(back.len(), c.len());
                println!("wrote and reloaded {jp}");
                back
            } else {
                c
            }
        }
    };
    println!("load/generate {:.0} ms", now_ms() - t0);
    let net = network::synth_network();
    print!("{}", bench::run(&city, &net, &o));
}
