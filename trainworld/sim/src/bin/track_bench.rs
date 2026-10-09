//! T-040 track model benchmark, native.
//!
//!   cargo run --release --bin track_bench -- [km] [stations] [lines] [reps] [save-out]
//!
//! Defaults: 2000 km, 500 stations, 60 lines, 21 repetitions per edit. With `save-out`, writes the
//! encoded network there (to check its compressed size).

use trainworld_sim::track::bench;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: f64| a.get(i).map_or(d, |s| s.parse().unwrap());
    let (out, bytes) = bench::run(num(0, 2000.0), num(1, 500.0) as usize, num(2, 60.0) as usize, num(3, 21.0) as usize);
    print!("{out}");
    if let Some(p) = a.get(4) {
        std::fs::write(p, bytes).unwrap();
        println!("wrote {p}");
    }
}
