//! T-1283: how long service edits take in the track model on a real save, natively.
//!
//!   cargo run --release --bin edit_bench -- <track bytes> [clock s] [reps]
//!
//! The track bytes are a save's network part (after the gzip and header; `checks/` scripts or
//! `node -e` can cut them out). Loads it as the clock worker does (converted to the Mercator frame,
//! no water or terrain masks), then on the line with the most Medium trains: a headway set, stepped
//! up and down, and train counts set, each timed. With `TW_DUMP=<file>`, writes `lines_info`
//! after the load to the file and after the edits to `<file>.end`, to compare two builds' figures.

use std::time::Instant;
use trainworld_sim::track::wasm::TrackApi;

/// A line's trains needed per level (`lines_info`).
fn trains(a: &TrackApi, line: u32) -> [f64; 3] {
    let info = a.lines_info();
    info.chunks(22).find(|c| c[0] as u32 == line).map_or([0.0; 3], |c| [c[13], c[14], c[15]])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&args[0]).expect("track bytes");
    let clock: f64 = args.get(1).map_or(111600.0, |s| s.parse().unwrap());
    let reps: usize = args.get(2).map_or(4, |s| s.parse().unwrap());
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    let mut a = TrackApi::new(-73.985, 40.758, true);
    let t = Instant::now();
    assert!(a.load(&bytes), "not a save");
    eprintln!("load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    a.set_unlimited(true);
    a.set_now(clock);
    let info = a.lines_info();
    let lines: Vec<u32> = info.chunks(22).map(|c| c[0] as u32).collect();
    let best = *lines.iter().max_by_key(|&&l| trains(&a, l)[1] as u32).unwrap();
    eprintln!("{} lines; busiest line {best}, trains {:?}, auto {:?}", lines.len(), trains(&a, best), a.line_auto(best));
    let mut now = clock;
    let mut time = |a: &mut TrackApi, what: &str, f: &dyn Fn(&mut TrackApi) -> u32| {
        // the clock runs between edits, as in the game
        now += 30.0;
        a.set_now(now);
        let t = Instant::now();
        let r = f(a);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("{what:<28} {ms:>8.1} ms  r={r}  trains {:?}", trains(a, best));
        ms
    };
    if let Ok(path) = std::env::var("TW_DUMP") {
        std::fs::write(&path, format!("{:?}\n", a.lines_info())).unwrap();
    }
    let mut all = vec![];
    all.push(time(&mut a, "headway 300/600/1200", &|a| a.set_auto_headway(best, 300, 600, 1200)));
    for i in 0..reps {
        let m = 600 - 60 * (i as u32 + 1);
        all.push(time(&mut a, &format!("headway medium {m}"), &|a| a.set_auto_headway(best, 300, m, 1200)));
    }
    all.push(time(&mut a, "back to counts", &|a| a.clear_auto_headway(best)));
    for i in 0..reps {
        let n = 10 + i as u32;
        all.push(time(&mut a, &format!("trains 20/{n}/4"), &|a| a.set_train_counts(best, 20, n, 4)));
    }
    all.push(time(&mut a, "undo", &|a| a.undo()));
    if let Ok(path) = std::env::var("TW_DUMP") {
        std::fs::write(format!("{path}.end"), format!("{:?}\n", a.lines_info())).unwrap();
    }
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let mut s = all.clone();
    s.sort_by(f64::total_cmp);
    eprintln!("edits {}: mean {mean:.1} ms, median {:.1} ms, max {:.1} ms", s.len(), s[s.len() / 2], s[s.len() - 1]);
}
