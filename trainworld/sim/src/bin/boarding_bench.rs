//! T-2699: how long the boarding account's steps take on a real save, natively.
//!
//!   cargo run --release --bin boarding_bench -- <track bytes> [clock s] [rounds]
//!
//! Loads the save's network as `edit_bench` does, gives every running line made-up riders (a fixed
//! pattern per line, stop, period and direction, some lines crowded) for a weekday, a Saturday and a
//! Sunday, and steps through the account `BOARDING_STEP_LINES` lines at a time as the clock worker
//! does, timing each step; then new weekday riders, `rounds` times. With `TW_DUMP=<file>`, writes
//! every line's account readouts (`line_account`, `line_periods`, `today`, `boarding_hour`) after
//! the last round, to compare two builds' figures.

use std::fmt::Write as _;
use std::time::Instant;
use trainworld_sim::track::wasm::TrackApi;

/// `clock.worker.ts`'s.
const STEP_LINES: u32 = 3;
const PERIODS: usize = 5;

fn riders(lines: &[(u32, usize)], round: u32) -> Vec<f32> {
    let mut v = vec![1.6f32, 1.8, 1.3, 1.2, 1.5];
    for &(id, n) in lines {
        v.push(id as f32);
        v.push(n as f32);
        // one line in seven crowded, so some platforms still wait at the day's end
        let scale = if id % 7 == 3 { 4000.0 } else { 120.0 };
        for p in 0..PERIODS {
            for side in 0..2 {
                for r in 0..2 {
                    for k in 0..n {
                        let x = ((id as usize * 31 + k * 17 + p * 7 + r * 3 + side * 5 + round as usize) % 23) as f32 / 23.0;
                        let v_ = if side == 0 { if k + 1 < n { x * scale } else { 0.0 } } else if k > 0 { x * scale } else { 0.0 };
                        v.push(v_);
                    }
                }
            }
        }
    }
    v
}

/// T-3939: the native heap live now and at its peak since the last call, MB, when built with the `demand`
/// feature (its counting allocator); nothing otherwise.
fn heap(what: &str) {
    #[cfg(feature = "demand")]
    {
        use trainworld_sim::demand::alloc;
        eprintln!("  heap {what}: live {:.1} MB, peak {:.1} MB", alloc::mb(alloc::current()), alloc::mb(alloc::peak()));
        alloc::reset_peak();
    }
    #[cfg(not(feature = "demand"))]
    let _ = what;
}

fn round(a: &mut TrackApi, what: &str) {
    let mut ms = vec![];
    let t0 = Instant::now();
    loop {
        let t = Instant::now();
        let more = a.boarding_step(STEP_LINES);
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if !more {
            break;
        }
    }
    let total = t0.elapsed().as_secs_f64() * 1e3;
    let redone = a.boarding_redone();
    let mut s = ms.clone();
    s.sort_by(f64::total_cmp);
    let q = |f: f64| s[((s.len() - 1) as f64 * f) as usize];
    let first: f64 = ms.iter().take(10).sum::<f64>() / ms.len().min(10) as f64;
    let last: f64 = ms.iter().rev().take(10).sum::<f64>() / ms.len().min(10) as f64;
    eprintln!("{what}: {} steps, {redone} lines redone, {total:.0} ms; step p50 {:.2} p95 {:.2} max {:.2} ms; first ten {first:.2}, last ten {last:.2} ms", ms.len(), q(0.5), q(0.95), q(1.0));
    let t = Instant::now();
    a.boarding_step(0);
    eprintln!("  an up-to-date check (step 0): {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&args[0]).expect("track bytes");
    let clock: f64 = args.get(1).map_or(111600.0, |s| s.parse().unwrap());
    let rounds: u32 = args.get(2).map_or(2, |s| s.parse().unwrap());
    let bytes = {
        let c = TrackApi::convert_save_frame(&bytes, 2);
        if c.is_empty() { bytes } else { c }
    };
    heap("before the model");
    let mut a = TrackApi::new(-73.985, 40.758, true);
    let t = Instant::now();
    assert!(a.load(&bytes), "not a save");
    eprintln!("load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    heap("after load");
    a.set_unlimited(true);
    a.set_now(clock);
    let info = a.lines_info();
    let lines: Vec<(u32, usize)> = info.chunks(22).filter(|c| c[3] > 0.0).map(|c| (c[0] as u32, a.line_stops(c[0] as u32).len())).filter(|l| l.1 >= 2).collect();
    eprintln!("{} lines, {} running with 2 stops or more, {} stops in all", info.len() / 22, lines.len(), lines.iter().map(|l| l.1).sum::<usize>());
    heap("after lines_info");
    for day in 0..3 {
        a.set_riders(riders(&lines, day), day);
    }
    heap("after riders");
    round(&mut a, "three kinds of day");
    heap("after the account");
    for r in 1..=rounds {
        a.set_riders(riders(&lines, 10 + r), 0);
        round(&mut a, &format!("new weekday riders {r}"));
    }
    heap("after the rounds");
    let mem = a.boarding_mem();
    let tot: usize = mem.iter().map(|x| x.1).sum();
    eprintln!("account memory {:.1} MB:", tot as f64 / 1e6);
    for (k, b) in mem {
        eprintln!("  {k}: {:.1} MB", b as f64 / 1e6);
    }
    // T-3939: the readouts the panels poll, every line and kind of day, timed
    let t = Instant::now();
    for _ in 0..5 {
        for &(id, _) in &lines {
            for day in 0..3 {
                std::hint::black_box(a.line_account(id, day));
                std::hint::black_box(a.line_periods(id, day));
            }
        }
        std::hint::black_box(a.today(clock));
        std::hint::black_box(a.past_day(clock));
        std::hint::black_box(a.boarding_hour((clock / 3600.0).floor() * 3600.0));
    }
    eprintln!("readouts (every line's account and periods, today, past day, the hour) x5: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    for _ in 0..10 {
        std::hint::black_box(a.today(clock));
    }
    let today_ms = t.elapsed().as_secs_f64() * 1e2;
    let t = Instant::now();
    for _ in 0..10 {
        std::hint::black_box(a.boarding_hour((clock / 3600.0).floor() * 3600.0));
    }
    eprintln!("  today {today_ms:.1} ms a call, the hour's platforms {:.1} ms a call", t.elapsed().as_secs_f64() * 1e2);
    if let Ok(path) = std::env::var("TW_DUMP") {
        let mut s = String::new();
        for &(id, _) in &lines {
            for day in 0..3 {
                writeln!(s, "{id} {day} {:?}", a.line_account(id, day)).unwrap();
                writeln!(s, "{id} {day} {:?}", a.line_periods(id, day)).unwrap();
            }
        }
        writeln!(s, "today {:?}", a.today(clock)).unwrap();
        writeln!(s, "past {:?}", a.past_day(clock)).unwrap();
        writeln!(s, "hour {:?}", a.boarding_hour((clock / 3600.0).floor() * 3600.0)).unwrap();
        std::fs::write(&path, s).unwrap();
    }
}
