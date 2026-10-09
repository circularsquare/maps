//! T-019: the pipeline's gravity step. Reads a cells-only city pack (format 1), solves the
//! doubly constrained 2 km zone gravity with the kernel's own code, and writes the pack with the
//! zone arrays and `gravity` header block added (notes/T-004.md). pipeline/build_city.py runs it.
//!
//!   cargo run --release --features demand --bin pack_gravity -- IN.json OUT.json
//!       [--zone-m 2000] [--decay-km K] [--decay-pow A] [--tol 1e-4]   (defaults: kernel::Params)
//!
//! One implementation of the gravity: the game's loader rebuilds zone trips from these factors
//! with the same decay table, so what the pipeline balanced is exactly what the game uses.

use trainworld_sim::demand::{clock::now_ms, kernel, pack};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = vec![];
    let mut p = kernel::Params::default();
    let mut tol = 1e-4f64;
    let mut i = 0;
    while i < args.len() {
        let val = || args.get(i + 1).cloned().expect("missing value");
        match args[i].as_str() {
            "--zone-m" => {
                p.zone_m = val().parse().unwrap();
                i += 1;
            }
            "--decay-km" => {
                p.decay_km = val().parse().unwrap();
                i += 1;
            }
            "--decay-pow" => {
                p.decay_pow = val().parse().unwrap();
                i += 1;
            }
            "--tol" => {
                tol = val().parse().unwrap();
                i += 1;
            }
            a if a.starts_with("--") => panic!("unknown argument {a}"),
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let [inp, out] = <[String; 2]>::try_from(pos).expect("usage: pack_gravity IN.json OUT.json [--zone-m M] [--decay-km K] [--decay-pow A] [--tol T]");
    let (header, bin) = pack::load_raw(&inp).unwrap_or_else(|e| panic!("{e}"));
    let city = pack::parse(&header, &bin).unwrap_or_else(|e| panic!("{e}"));
    let t0 = now_ms();
    let z = kernel::build_zones(&city, p.zone_m);
    let g = kernel::gravity(&z, &p, 1000, tol);
    let ms = now_ms() - t0;
    if g.max_err > tol {
        eprintln!("WARNING: Furness stopped at max row error {:.2e} (> {tol:.0e}) after {} iterations", g.max_err, g.iters);
    }
    println!(
        "gravity: {} zones ({} x {} grid of {} m), decay d^-{} exp(-d/{} km); Furness {} iterations, max row error {:.1e}; mean trip {:.2} km; {:.0} trips; {:.0} ms",
        z.n, z.w, z.h, z.zone_m, p.decay_pow, p.decay_km, g.iters, g.max_err, g.mean_km, g.total, ms
    );
    let pg = kernel::to_pack_gravity(&z, &g, &p);
    let (h2, b2) = pack::with_gravity(&header, &bin, &pg).unwrap_or_else(|e| panic!("{e}"));
    // read back what is about to be written before replacing anything
    let back = pack::parse(&h2, &b2).unwrap_or_else(|e| panic!("written pack does not parse: {e}"));
    let bg = back.gravity.as_ref().expect("gravity missing after write");
    let worst = bg.row.iter().zip(&g.row).chain(bg.col.iter().zip(&g.col)).map(|(a, b)| if *b == 0.0 { a.abs() } else { (a / b - 1.0).abs() }).fold(0.0f64, f64::max);
    assert!(worst < 1e-6, "factors changed by {worst:e} in the f32 round trip");
    let stem = out.strip_suffix(".json").unwrap_or(&out).to_string();
    let (bin_path, json_path) = (format!("{stem}.bin"), format!("{stem}.json"));
    std::fs::write(format!("{bin_path}.part"), &b2).unwrap();
    std::fs::rename(format!("{bin_path}.part"), &bin_path).unwrap();
    std::fs::write(format!("{json_path}.part"), &h2).unwrap();
    std::fs::rename(format!("{json_path}.part"), &json_path).unwrap();
    println!("wrote {json_path} and .bin ({:.2} MB, {} bytes of zone arrays added)", b2.len() as f64 / 1e6, b2.len() - bin.len());
}
