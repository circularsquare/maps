/// T-2642 (scratch only, included at the end of the copy's intercity/tests.rs): the market's pass on
/// a captured net in each `SC_MODE` of `TW_MODES`, back to back in one process (`TW_REPS` rounds of
/// them), every figure of each compared with the first mode's to the bit; with `TW_CROWD=k` each
/// is solved again at the crowding of k times the first mode's free-flow loads (left behind on).
/// `TW_T780_NET=<net> TW_Q=0 TW_MODES=0,1 cargo test --release --no-default-features --features demand --lib t2642_ab -- --ignored --nocapture`
#[test]
#[ignore]
fn t2642_ab() {
    use super::pass::PT;
    use crate::demand::kernel::{tsc, SC_MODE, SC_STATS};
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let q: usize = env("TW_Q", "0").parse().unwrap();
    let modes: Vec<u32> = env("TW_MODES", "0,1").split(',').map(|v| v.parse().unwrap()).collect();
    let reps: usize = env("TW_REPS", "1").parse().unwrap();
    let crowd: f32 = env("TW_CROWD", "0").parse().unwrap();
    let net = std::env::var("TW_T780_NET").unwrap();
    let t = now_ms();
    let mut a = t780_net(&net, MARKET_CHUNK);
    eprintln!("set up {:.0} ms, {} longer pairs", now_ms() - t, a.pairs.len());
    {
        // the graph's size and its leaf stations
        SC_MODE.with(|c| c.set(0));
        assert!(a.solve(q));
        let per = a.per(q).unwrap();
        let g = &per.g;
        let qq = crate::demand::kernel::DestSearch::new(g, &per.p);
        let (leaf, chain) = qq.counts();
        eprintln!("graph: {} stations, {} route nodes, {} nodes; leaf stations {} ({:.1}%), route nodes at them {} ({:.1}%)", g.n_st, g.n - g.n_st, g.n, leaf, 100. * leaf as f64 / g.n_st as f64, chain, 100. * chain as f64 / (g.n - g.n_st) as f64);
    }
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
    let mut base: Option<Vec<(String, Vec<f64>)>> = None;
    let mut base_c: Option<Vec<(String, Vec<f64>)>> = None;
    let mut total: Vec<f32> = vec![];
    for rep in 0..reps {
        for &mode in &modes {
            for crowded in [false, true] {
                if crowded && crowd <= 0. {
                    continue;
                }
                SC_MODE.with(|c| c.set(mode));
                // every solve searches afresh (no table kept from the solve before)
                crate::demand::kernel::clear_skim_cache();
                SC_STATS.with(|c| c.set([0; 32]));
                PT.with(|c| c.set([0; 16]));
                let (t0, c0, k0) = (now_ms(), thread_cpu_ms(), tsc());
                if crowded {
                    assert!(a.solve_crowded(q, &total));
                } else {
                    assert!(a.solve(q));
                }
                let (wall, cpu, ticks) = (now_ms() - t0, thread_cpu_ms() - c0, tsc() - k0);
                // (the machine is shared: sections are their share of the ticks times the CPU time)
                let ms = |t: u64| t as f64 / ticks as f64 * cpu;
                let st = SC_STATS.with(|c| c.get());
                let pt = PT.with(|c| c.get());
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
                eprintln!("rep {rep} mode {mode} {tag}: solve {wall:.0} ms wall, {cpu:.0} ms cpu; pass {:.0}: searches {:.0}, fare km {:.0}, longer {:.0} (logsum {:.0}, spread {:.0}; {} pairs, {:.1}M terms), commutes {:.0}, everyday {:.0}, assign {:.0} (again {:.0}), table by zone: index {:.0}, copy {:.0}", ms(pt[0]), ms(pt[1]), ms(pt[2]), ms(pt[3]), ms(pt[4]), ms(pt[5]), pt[11], pt[10] as f64 / 1e6, ms(pt[6]), ms(pt[7]), ms(pt[8]), ms(pt[9]), ms(pt[12]), ms(pt[13]));
                eprintln!("  search counts: {} searches, {:.1}M pops ({:.1}M stale, {:.1}M boarding turns, {:.2}M change), settled {:.1}M stations + {:.1}M leaf + {:.1}M route nodes popped + {:.1}M stepped, {:.2}M change nodes made, {:.1}M leaf joins, {:.1}M in order", st[0], st[1] as f64 / 1e6, st[2] as f64 / 1e6, st[7] as f64 / 1e6, st[8] as f64 / 1e6, st[3] as f64 / 1e6, st[4] as f64 / 1e6, st[5] as f64 / 1e6, st[6] as f64 / 1e6, st[9] as f64 / 1e6, st[10] as f64 / 1e6, st[12] as f64 / 1e6);
                if mode & crate::demand::kernel::SC_PROBE != 0 {
                    eprintln!("  search ms: run {:.0} (settle_station {:.0}, settle_change {:.0}, offer_line {:.0}, leaf after {:.0}), strategy copies {:.0}, trip_km {:.0}, entry_costs {:.0}, put {:.0}", ms(st[16]), ms(st[17]), ms(st[18]), ms(st[19]), ms(st[25]), ms(st[20]), ms(st[21]), ms(st[22]), ms(st[23]));
                }
                eprintln!("  figures: {same}");
            }
        }
    }
}

/// T-2642 (scratch): the searches alone, each `SC_MODE` of `TW_MODES` (the first the reference) run
/// towards the same destinations in turns of 64 (the order of the modes rotating), so a busy
/// machine's swings fall on all alike; every strategy compared with the reference's: route costs,
/// planned-wait entries, accepted departures, next nodes and change nodes to the bit, the settle
/// order as the same nodes. `TW_T780_NET=<net> TW_Q=0 TW_MODES=0,2 ... t2642_search`
#[test]
#[ignore]
fn t2642_search() {
    use crate::demand::kernel::{tsc, DestSearch, SC_MODE, SC_STATS};
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let q: usize = env("TW_Q", "0").parse().unwrap();
    let modes: Vec<u32> = env("TW_MODES", "0,2").split(',').map(|v| v.parse().unwrap()).collect();
    let reps: usize = env("TW_REPS", "1").parse().unwrap();
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), MARKET_CHUNK);
    SC_MODE.with(|c| c.set(0));
    assert!(a.solve(q));
    let per = a.per(q).unwrap();
    let (g, p) = (&per.g, &per.p);
    let want: Vec<usize> = (0..g.n_st).filter(|&d| a.want.get(d).copied().unwrap_or(false)).collect();
    let mut qs: Vec<DestSearch> = modes
        .iter()
        .map(|&m| {
            SC_MODE.with(|c| c.set(m));
            DestSearch::new(g, p)
        })
        .collect();
    SC_MODE.with(|c| c.set(0));
    SC_STATS.with(|c| c.set([0; 32]));
    let s = g.n_st;
    let mut ticks = vec![0u64; modes.len()];
    let mut col = vec![vec![0f32; s]; modes.len()];
    let (mut bad, mut checked) = (vec![0usize; modes.len()], 0usize);
    for rep in 0..reps {
        for (ci, chunk) in want.chunks(64).enumerate() {
            let m = modes.len();
            for o in 0..m {
                let k = (o + ci + rep) % m;
                let t = tsc();
                for &dst in chunk {
                    std::hint::black_box(qs[k].run_col(dst, &mut col[k]));
                }
                ticks[k] += tsc() - t;
            }
            if rep == 0 {
                // the strategies, compared once
                for &dst in chunk.iter().take(8) {
                    let r = qs[0].run_col(dst, &mut col[0]);
                    let mut order0 = r.order.to_vec();
                    order0.sort_unstable();
                    checked += 1;
                    for k in 1..m {
                        let d = qs[k].run_col(dst, &mut col[k]);
                        let mut order = d.order.to_vec();
                        order.sort_unstable();
                        let same = col[k].iter().zip(&col[0]).all(|(x, y)| x.to_bits() == y.to_bits())
                            && d.entry.len() == r.entry.len()
                            && d.entry.iter().zip(r.entry.iter()).all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits())
                            && d.board == r.board
                            && d.next.to_vec() == r.next.to_vec()
                            && d.change.st == r.change.st
                            && d.change.off == r.change.off
                            && d.change.bits == r.change.bits
                            && order == order0;
                        if !same {
                            bad[k] += 1;
                        }
                    }
                }
            }
        }
    }
    let st = SC_STATS.with(|c| c.get());
    for (k, &m) in modes.iter().enumerate() {
        eprintln!("mode {m}: {} searches x {reps}, {:.0} M ticks, {:.3} of the reference's; strategies differing {} of {checked}", want.len(), ticks[k] as f64 / 1e6, ticks[k] as f64 / ticks[0] as f64, bad[k]);
    }
    drop(qs);
    let st2 = SC_STATS.with(|c| c.get());
    eprintln!("(all modes: {} searches, {:.1}M pops; first stats {:?})", st2[0], st2[1] as f64 / 1e6, &st[..1]);
}
