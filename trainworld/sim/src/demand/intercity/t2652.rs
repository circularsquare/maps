//! T-2652: the strategy search with leaf stations against the plain one (`set_plain_search`) on a
//! captured market network, for A/B runs (notes/T-2652.md).

use super::tests::t780_net;
use super::*;
use crate::demand::kernel::{set_plain_search, DestSearch};

/// This thread's CPU milliseconds so far, 0 off Windows.
fn cpu_ms() -> f64 {
    #[cfg(windows)]
    {
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn GetThreadTimes(h: isize, created: *mut u64, ended: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
        }
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        // SAFETY: four FILETIMEs written by the call; the pseudo-handle needs no closing
        unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
        (k + u) as f64 / 1e4
    }
    #[cfg(not(windows))]
    0.
}

fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: reads the time stamp counter
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// The searches alone on `TW_T780_NET` in each mode of `TW_MODES` (0 plain, 1 with leaf stations;
/// the first the reference), towards the market's destinations in turns of 64 (the order of the
/// modes rotating), so a busy machine's swings fall on all alike; `TW_REPS` rounds. Every destination's strategy compared to the bit on the first
/// round (route costs, planned-wait entries, accepted departures, next nodes, change nodes, the
/// settle order as the same nodes).
/// `TW_T780_NET=<net> TW_Q=0 cargo test --release --features demand --lib t2652_search -- --ignored --nocapture`
#[test]
#[ignore]
fn t2652_search() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let q: usize = env("TW_Q", "0").parse().unwrap();
    let reps: usize = env("TW_REPS", "1").parse().unwrap();
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), MARKET_CHUNK);
    assert!(a.solve(q));
    let per = a.per(q).unwrap();
    let (g, p) = (&per.g, &per.p);
    let want: Vec<usize> = (0..g.n_st).filter(|&d| a.want.get(d).copied().unwrap_or(false)).collect();
    let modes: Vec<u32> = env("TW_MODES", "0,1").split(',').map(|v| v.parse().unwrap()).collect();
    let m = modes.len();
    let mut qs: Vec<DestSearch> = modes
        .iter()
        .map(|&mode| {
            set_plain_search(mode == 0);
            DestSearch::new(g, p)
        })
        .collect();
    set_plain_search(false);
    let (_, leaves, at_leaves) = qs[m - 1].counts();
    eprintln!("graph: {} stations, {} route nodes; leaf stations {leaves} ({:.1}%), route nodes at them {at_leaves} ({:.1}%); {} destinations", g.n_st, g.n - g.n_st, 100. * leaves as f64 / g.n_st as f64, 100. * at_leaves as f64 / (g.n - g.n_st) as f64, want.len());
    let s = g.n_st;
    let mut t = vec![0u64; m];
    let mut pops = vec![0u64; m];
    let mut col = vec![vec![0f32; s]; m];
    let mut bad = vec![0; m];
    for rep in 0..reps {
        for (ci, chunk) in want.chunks(64).enumerate() {
            for o in 0..m {
                let k = (o + ci + rep) % m;
                let (p0, t0) = (qs[k].counts().0, ticks());
                for &dst in chunk {
                    std::hint::black_box(qs[k].run_col(dst, &mut col[k]));
                }
                t[k] += ticks() - t0;
                pops[k] += qs[k].counts().0 - p0;
            }
            if rep == 0 {
                for &dst in chunk {
                  let r = qs[0].run_col(dst, &mut col[0]);
                  for k in 1..m {
                    let d = qs[k].run_col(dst, &mut col[k]);
                    let (mut o0, mut o1) = (r.order.to_vec(), d.order.to_vec());
                    o0.sort_unstable();
                    o1.sort_unstable();
                    let same = col[k].iter().zip(&col[0]).all(|(x, y)| x.to_bits() == y.to_bits())
                        && d.entry.len() == r.entry.len()
                        && d.entry.iter().zip(r.entry.iter()).all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits())
                        && d.board == r.board
                        && d.next.to_vec() == r.next.to_vec()
                        && d.change.st == r.change.st
                        && d.change.off == r.change.off
                        && d.change.bits == r.change.bits
                        && o0 == o1;
                    if !same {
                        bad[k] += 1;
                        eprintln!("mode {}, destination {dst}: strategies differ", modes[k]);
                    }
                  }
                }
            }
        }
    }
    for k in 0..m {
        eprintln!("mode {}: {:.0} M ticks, {:.1}M pops; {:.3} of the first's time ({:.2}x); strategies differing {} of {}", modes[k], t[k] as f64 / 1e6, pops[k] as f64 / 1e6, t[k] as f64 / t[0] as f64, t[0] as f64 / t[k] as f64, bad[k], want.len());
    }
    assert!(bad.iter().all(|&b| b == 0));
}

/// The market's pass on `TW_T780_NET` in each mode of `TW_MODES` (0 plain, 1 with leaf stations),
/// `TW_REPS` rounds; every figure of each solve compared with the first one's to the bit, at free flow and (with `TW_CROWD=k`)
/// again at the crowding of k times the free-flow loads.
/// `TW_T780_NET=<net> TW_Q=0 TW_CROWD=3 cargo test --release --features demand --lib t2652_ab -- --ignored --nocapture`
#[test]
#[ignore]
fn t2652_ab() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let q: usize = env("TW_Q", "0").parse().unwrap();
    let reps: usize = env("TW_REPS", "1").parse().unwrap();
    let crowd: f32 = env("TW_CROWD", "0").parse().unwrap();
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), MARKET_CHUNK);
    let figs = |a: &IntercityApi, q: usize| -> Vec<(String, Vec<f64>)> {
        let per = a.per(q).unwrap();
        let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let mut out: Vec<(String, Vec<f64>)> = vec![];
        let mut loads = |name: &str, l: &Loads| {
            for (k, v) in [("seg", &l.seg), ("board", &l.board), ("alight", &l.alight), ("rn_board", &l.rn_board), ("rn_alight", &l.rn_alight), ("entry", &l.entry), ("exit", &l.exit), ("rn_fare_x", &l.rn_fare_x)] {
                out.push((format!("{name}.{k}"), f(v)));
            }
            out.push((format!("{name}.pax_min"), vec![l.pax_min]));
        };
        loads("all", &per.avg);
        loads("com", &per.c_avg);
        if let Some(l) = &per.r_avg {
            loads("lean", l);
        }
        for (i, l) in per.e_avg.iter().enumerate() {
            if let Some(l) = l {
                loads(&format!("ev{i}"), l);
            }
        }
        out.push(("rail".into(), vec![per.rail, per.c_rail, per.e_rail, per.e_walk]));
        out.push(("pair_rail".into(), f(&per.pair_rail)));
        out.push(("st".into(), f(&per.st)));
        out.push(("xfer".into(), f(&per.xfer)));
        out.push(("c_frac".into(), f(&per.c_frac)));
        out.push(("e_frac".into(), f(&per.e_frac)));
        out.push(("cor".into(), per.corridors.v.clone()));
        out.push(("cor_k".into(), per.corridors.k.iter().map(|&x| x as f64).collect()));
        out
    };
    let cmp = |base: &[(String, Vec<f64>)], now: &[(String, Vec<f64>)]| -> String {
        let mut diff = vec![];
        for ((k, a), (_, b)) in base.iter().zip(now) {
            if a.len() != b.len() {
                diff.push(format!("{k}: length {} -> {}", a.len(), b.len()));
                continue;
            }
            let (mut n, mut worst) = (0, 0f64);
            for (x, y) in a.iter().zip(b) {
                if x.to_bits() != y.to_bits() {
                    n += 1;
                    worst = worst.max((x - y).abs() / x.abs().max(y.abs()).max(1e-30));
                }
            }
            if n > 0 {
                diff.push(format!("{k}: {n} of {} differ, worst {worst:.2e}", a.len()));
            }
        }
        if diff.is_empty() { "the same to the bit".into() } else { diff.join("; ") }
    };
    let (mut base, mut base_c): (Option<Vec<(String, Vec<f64>)>>, Option<Vec<(String, Vec<f64>)>>) = (None, None);
    let mut total: Vec<f32> = vec![];
    let modes: Vec<u32> = env("TW_MODES", "0,1").split(',').map(|v| v.parse().unwrap()).collect();
    for rep in 0..reps {
        for &mode in &modes {
            for crowded in [false, true] {
                if crowded && crowd <= 0. {
                    continue;
                }
                set_plain_search(mode == 0);
                crate::demand::kernel::clear_skim_cache();
                let (t0, c0) = (now_ms(), cpu_ms());
                assert!(if crowded { a.solve_crowded(q, &total) } else { a.solve(q) });
                let (wall, cpu) = (now_ms() - t0, cpu_ms() - c0);
                let now = figs(&a, q);
                if !crowded && total.is_empty() {
                    let per = a.per(q).unwrap();
                    total = per.avg.seg.iter().chain(&per.avg.rn_board).chain(&per.avg.rn_alight).map(|&x| x * crowd.max(1.)).collect();
                }
                let slot = if crowded { &mut base_c } else { &mut base };
                let same = match slot {
                    None => {
                        *slot = Some(now);
                        "(reference)".to_string()
                    }
                    Some(b) => cmp(b, &now),
                };
                let tag = if crowded { "crowded" } else { "free flow" };
                eprintln!("rep {rep} {mode} {tag}: solve {wall:.0} ms wall, {cpu:.0} ms cpu; figures {same}");
            }
        }
    }
    set_plain_search(false);
}
