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
    // T-3542: what demand reads per period (`TW_DEMAND=1`; the slots' rates and the riders' mean)
    if std::env::var("TW_DEMAND").is_ok() {
        for &l in lines {
            s += &format!("period {l} {:?}\ndemand {l} {:?}\nhours {l} {:?}\n", a.line_tph_by_period(l), a.line_demand_tph(l), a.line_hours(l));
        }
    }
    let meta = a.profile_meta();
    for day in [2.0, 6.0] {
        for h in 0..24 {
            let epoch = day * 86400.0 + h as f64 * 3600.0;
            for t in a.trips(epoch).chunks(4) {
                let p = t[0] as u32;
                // T-3346 stage 2: each slot's variants under it (a trip of a timed line prints its slot)
                let q = p / a.profile_vars();
                let (line, run, lev) = (q / (2 * stride), (q / stride) % 2, q % stride);
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
    // T-3542: `TW_PERIODS=<3|4|5>` sets the game's rows first (timed, not in the mean, the clock
    // left as it is), so every line runs that many; `TW_PERIODS_BACK=1` sets three again after it
    if let Some(p) = std::env::var("TW_PERIODS").ok().and_then(|s| s.parse::<u32>().ok()) {
        let t = Instant::now();
        let r = a.set_game_periods(p);
        eprintln!("periods {p} (game) {:.1} ms r={r}, slots {:?}", t.elapsed().as_secs_f64() * 1e3, a.period_slots());
        if std::env::var("TW_PERIODS_BACK").is_ok() {
            let t = Instant::now();
            let r = a.set_game_periods(3);
            eprintln!("periods 3 (game) {:.1} ms r={r}, slots {:?}", t.elapsed().as_secs_f64() * 1e3, a.period_slots());
        }
    }
    // `TW_PERIODS_LINE=<4|5>`: the busiest line alone sets that many rows, the rest three
    if let Some(p) = std::env::var("TW_PERIODS_LINE").ok().and_then(|s| s.parse::<u32>().ok()) {
        let t = Instant::now();
        let r = a.set_line_periods(best, p);
        eprintln!("periods {p} (line {best}) {:.1} ms r={r}, slots {:?}", t.elapsed().as_secs_f64() * 1e3, a.period_slots());
    }
    // T-3542: `TW_HOURS=<five hours>` sets the game's hours (timed, not in the mean);
    // `TW_HOURS_LINE=<five hours>` the busiest line's own, on five rows of its own
    let cuts = |v: &str| -> Vec<u32> { v.split(',').map(|x| x.trim().parse().unwrap()).collect() };
    if let Ok(v) = std::env::var("TW_HOURS") {
        let t = Instant::now();
        let r = a.set_game_hours(cuts(&v));
        eprintln!("hours {v} (game) {:.1} ms r={r}", t.elapsed().as_secs_f64() * 1e3);
    }
    if let Ok(v) = std::env::var("TW_HOURS_LINE") {
        a.set_line_periods(best, 5);
        let t = Instant::now();
        let r = a.set_line_hours(best, cuts(&v));
        eprintln!("hours {v} (line {best}) {:.1} ms r={r}", t.elapsed().as_secs_f64() * 1e3);
    }
    // T-3542: `TW_HOURS_CHECK=<from h>,<to h>`: per line, departures from its first stop in that local
    // hour and in 7-8, a weekday, where the first are more (a borrowed hour outrunning the peak)
    if let Ok(v) = std::env::var("TW_HOURS_CHECK") {
        let h = cuts(&v);
        let (mut more, mut seen) = (0, 0);
        for &l in &lines {
            let Some(&stop) = a.line_stops(l).first() else { continue };
            let n = |from: f64, to: f64| { let s = a.line_shift(l); a.station_departures(l, stop, 3.0 * 86400.0 + from * 3600.0 - s, 3.0 * 86400.0 + to * 3600.0 - s).len() / 3 };
            let (x, peak) = (n(h[0] as f64, h[1] as f64), n(7.0, 8.0));
            seen += 1;
            if x > peak {
                more += 1;
                eprintln!("line {l} {}: {x} departures {}-{} against {peak} in 7-8", a.line_name(l), h[0], h[1]);
            }
        }
        eprintln!("hours check: {more} of {seen} lines leave more often in {}-{} than in 7-8", h[0], h[1]);
    }
    let mut all = vec![];
    // T-4194: `TW_STOPS=1`: the line on headways (trains bought as its length changes), then a
    // middle stop removed and put back, each timed with the settle rounds it ran
    if std::env::var("TW_STOPS").is_ok() {
        let rounds = || trainworld_sim::track::world::SETTLE_UPDATES.with(|c| c.get());
        // T-4194: the line whose last stop comes off cleanly and has the most trains
        let mut best = best;
        let mut top = 0.0;
        for &l in &lines {
            let s = a.line_stops(l);
            if s.len() < 4 { continue; }
            let t = trains(&a, l)[1];
            if t > top && a.set_stops(l, &s[..s.len() - 1]) == 0 { a.set_stops(l, &s); best = l; top = t; }
        }
        eprintln!("stop line {best}: {top} medium trains");
        time(&mut a, "headways (not in the stop rows)", &|a| a.set_auto_headway(best, 300, 600, 1200));
        let full = a.line_stops(best);
        // a cut the track accepts: the last stop, else the first, else the one before the last
        let mut cut = full.clone();
        for k in [full.len() - 1, 0, full.len() - 2, full.len() / 2] {
            let mut c = full.clone();
            c.remove(k);
            let r = a.set_stops(best, &c);
            eprintln!("cut stop {k}: r={r}");
            if r == 0 { cut = c; a.set_stops(best, &full); break; }
        }
        for i in 0..reps.max(2) {
            for (what, stops) in [("remove stop", &cut), ("add stop", &full)] {
                let r0 = rounds();
                let rc = std::cell::Cell::new(0);
                let ms = time(&mut a, &format!("{what} {i}"), &|a| { let r = a.set_stops(best, stops); rc.set(r); r });
                eprintln!("STOPS {what} {i}: {ms:.1} ms, {} settle rounds, r={}, {} stops", rounds() - r0, rc.get(), stops.len());
            }
        }
        return;
    }
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
    // T-3542: with more than three rows, one row's trains alone (the first: rush or morning)
    if a.game_periods() > 3 {
        let mut one = vec![];
        for i in 0..reps as u32 {
            let mut t = a.line_row_trains(best);
            t[0] += 1 + i;
            one.push(time(&mut a, &format!("first row {}", t[0]), &|a| a.set_line_row_trains(best, t.clone())));
        }
        eprintln!("first row edits {}: mean {:.1} ms", one.len(), one.iter().sum::<f64>() / one.len() as f64);
    }
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
        // T-3542: and its save and figures, the same as `.end.save` and `.end.figs`
        if std::env::var("TW_RELOAD_FIGS").is_ok() {
            b.set_unlimited(true);
            b.set_now(now);
            std::fs::write(format!("{path}.reload.save"), b.save()).unwrap();
            std::fs::write(format!("{path}.reload.figs"), figures(&b, &lines)).unwrap();
        }
    }
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let mut s = all.clone();
    s.sort_by(f64::total_cmp);
    eprintln!("edits {}: mean {mean:.1} ms, median {:.1} ms, max {:.1} ms", s.len(), s[s.len() / 2], s[s.len() - 1]);
}
