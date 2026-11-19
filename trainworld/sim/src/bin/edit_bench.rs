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

/// T-3542: figures to compare two builds to the bit: per line `line_summary`, the running cost per
/// hour at each level of both kinds of day, and every trip drawn over a weekday and a Saturday with
/// its train's position halfway (profile index cut to line and run, `TW_STRIDE` profiles a run).
fn figures(a: &TrackApi, lines: &[u32]) -> String {
    let stride: u32 = std::env::var("TW_STRIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(a.profile_slots());
    let slot_of = |k: u32| if stride == 3 { k } else { [0, 0, 0, 1, 2][k as usize] };
    let mut s = String::new();
    for &l in lines {
        s += &format!("summary {l} {:?}\n", a.line_summary(l));
    }
    s += &format!("running {:?}\n", (0..6).map(|k| a.running_cost(k)).collect::<Vec<_>>());
    // T-3542: the readouts that take or give a row (`TW_ROWS=1`; master's API may lack some)
    if std::env::var("TW_ROWS").is_ok() {
        for r in 0..3 {
            s += &format!("markers {r} {:?}\nfull {r} {:?}\n", a.markers(r), a.full_track(r));
        }
        s += &format!("upkeep {:?}\n", a.line_upkeep(&[14.0, 4.0, 6.0]));
        for &l in lines {
            s += &format!("held {l} {:?}\nrest {l} {:?}\nweekend {l} {:?}\n", a.line_held(l), a.line_tph_rest(l), a.line_weekend(l));
            for r in 0..3 {
                s += &format!("delays {l} {r} {:?}\nwaits {l} {r} {:?}\n", a.line_delays(l, r), a.line_waits(l, r));
            }
            for &n in a.line_stops(l).iter() {
                s += &format!("junction {n} {:?}\n", a.junction_info(n));
            }
        }
    }
    let meta = a.profile_meta();
    for day in [2.0, 6.0] {
        for h in 0..24 {
            let epoch = day * 86400.0 + h as f64 * 3600.0;
            for t in a.trips(epoch).chunks(4) {
                let p = t[0] as u32;
                let (line, run, lev) = (p / (2 * stride), (p / stride) % 2, p % stride);
                let (draw, end) = (epoch + t[1] as f64, epoch + t[2] as f64);
                let at = a.train_at(line, run as usize, lev as usize, draw, (draw + end) / 2.0);
                s += &format!("trip {line} {run} {} {:?} {:?} {:?}\n", slot_of(lev), &t[1..], &meta[p as usize * 4 + 1..p as usize * 4 + 4], at);
            }
        }
    }
    s
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
        // T-3542: the save bytes after the load and after the edits, to compare two builds' saves
        std::fs::write(format!("{path}.save"), a.save()).unwrap();
        std::fs::write(format!("{path}.figs"), figures(&a, &lines)).unwrap();
    }
    // T-1924: per line after the load, tab-separated: id, name, km, then per level (high, medium,
    // low) trains an hour, round trip s, delay s and trains, to compare two builds' run times
    if let Ok(path) = std::env::var("TW_LINES") {
        let mut s = String::from("id\tname\tkm\ttph\tround_trip\tdelay\ttrains\n");
        for c in a.lines_info().chunks(22) {
            let id = c[0] as u32;
            let f = |k: usize| format!("{:.2}/{:.2}/{:.2}", c[k], c[k + 1], c[k + 2]);
            s += &format!("{id}\t{}\t{:.2}\t{}\t{}\t{}\t{}\n", a.line_name(id), c[9] / 1000.0, f(5), f(10), f(16), f(13));
        }
        std::fs::write(&path, s).unwrap();
    }
    // T-3521: `TW_WEEKEND=<percent>` runs every line's weekends at that percent first (timed, not
    // in the mean), so every edit after it settles the weekends too
    if let Some(p) = std::env::var("TW_WEEKEND").ok().and_then(|s| s.parse::<u32>().ok()) {
        time(&mut a, &format!("weekends {p}% (all lines)"), &|a| a.set_weekend_share_all(p));
        // every line's trains an hour at each level, weekdays then weekends, summed
        let sum = |row: u32| -> Vec<f32> { (0..3).map(|lev| lines.iter().map(|&l| a.line_tph_on(l, row, lev)).sum()).collect() };
        eprintln!("trains an hour, all lines: weekdays {:?}, weekends {:?}", sum(0), sum(1));
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
        std::fs::write(format!("{path}.end.save"), a.save()).unwrap();
        std::fs::write(format!("{path}.end.figs"), figures(&a, &lines)).unwrap();
        // saved and loaded again (T-1787): the same figures, to the bit
        let mut b = TrackApi::new(-73.985, 40.758, true);
        let t = Instant::now();
        assert!(b.load(&a.save()), "not a save");
        eprintln!("reload {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        std::fs::write(format!("{path}.reload"), format!("{:?}\n", b.lines_info())).unwrap();
    }
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let mut s = all.clone();
    s.sort_by(f64::total_cmp);
    eprintln!("edits {}: mean {mean:.1} ms, median {:.1} ms, max {:.1} ms", s.len(), s[s.len() / 2], s[s.len() - 1]);
}
