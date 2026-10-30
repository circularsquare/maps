use super::*;
use crate::demand::api::release_skims;
use crate::demand::lod::{self, Lod, Sampling};
use crate::demand::network::RealNet;
use crate::demand::{alloc, kernel, synth};

/// Lines through a zone's busiest 2 km square: four, east-west, north-south and the two
/// diagonals, `reach` metres each way with a station every `spacing`, sharing the middle one;
/// 36 km/h between stations, 12, 8 and 4 trains an hour, 8 cars.
fn cross_net(x: &[f32], y: &[f32], pop: &[f32], reach: f32, spacing: f32) -> RealNet {
    let mut sq: std::collections::HashMap<(i32, i32), (f64, f64, f64)> = Default::default();
    for c in 0..x.len() {
        let e = sq.entry(((x[c] / 2000.0).floor() as i32, (y[c] / 2000.0).floor() as i32)).or_default();
        let w = pop[c] as f64;
        *e = (e.0 + w, e.1 + w * x[c] as f64, e.2 + w * y[c] as f64);
    }
    let (_, &(w, sx, sy)) = sq.iter().max_by(|a, b| a.1 .0.total_cmp(&b.1 .0).then(b.0.cmp(a.0))).unwrap();
    let (cx, cy) = ((sx / w) as f32, (sy / w) as f32);
    let m = (reach / spacing).round() as i32;
    let mut st_xy = vec![cx, cy];
    let (mut line_n, mut stops, mut times) = (vec![], vec![], vec![]);
    for a in 0..4 {
        let th = a as f32 * std::f32::consts::FRAC_PI_4;
        let mut line = vec![];
        for k in -m..=m {
            if k == 0 {
                line.push(0);
            } else {
                line.push((st_xy.len() / 2) as u32);
                st_xy.extend([cx + k as f32 * spacing * th.cos(), cy + k as f32 * spacing * th.sin()]);
            }
        }
        let n = line.len();
        let hop = spacing / 10.0;
        for run in 0..2 {
            for _lev in 0..3 {
                for k in 0..n {
                    let end = if run == 0 { k == n - 1 } else { k == 0 };
                    times.push(if end { 0.0 } else { hop });
                }
            }
        }
        line_n.push(n as u32);
        stops.extend(line);
    }
    RealNet { st_xy, line_n, stops, times, tph: [12.0, 8.0, 4.0].repeat(4), cars: vec![8.0; 4] }
}

/// A zone's own network for the measurements: the cross scaled by its people.
fn zone_net(a: &DemandApi) -> RealNet {
    let pop: f64 = a.city.pop.iter().map(|&v| v as f64).sum();
    let reach = (4000.0 * (pop / 1e6).sqrt() as f32).clamp(5000.0, 25000.0);
    cross_net(&a.city.x, &a.city.y, &a.city.pop, reach, if pop > 5e6 { 1200.0 } else { 1500.0 })
}

/// A zone's two populations opened at `lod`, with the commute chunks' start mixed by `salt`.
fn open(h: &str, b: &[u8], lod: Lod) -> (DemandApi, EverydayApi) {
    let mut a = DemandApi::new(h, b).unwrap();
    a.lod = lod;
    a.params_mut().commute_chunk = lod.chunk;
    a.params_mut().chunk_salt = lod.sampling.salt;
    let e = EverydayApi::beside_with(&a, h, b, &[]).unwrap();
    (a, e)
}

/// Every period through the worker's rounds (`demand.worker.ts` `step`): a free-flow solve,
/// `rounds` crowding rounds on both populations' loads, then the capacity fit. Returns the ms
/// each population took.
fn solve_day(a: &mut DemandApi, e: &mut EverydayApi, rounds: usize, mut each: impl FnMut(usize, &DemandApi, &EverydayApi)) -> [f64; 2] {
    let mut ms = [0.0; 2];
    let total = |a: &DemandApi, e: &EverydayApi, q: usize| -> Vec<f32> {
        let sum = |x: Vec<f32>, y: Vec<f32>| -> Vec<f32> { x.iter().zip(&y).map(|(p, q)| p + q).collect() };
        [sum(a.seg(q), e.seg(q)), sum(a.board(q), e.board(q)), sum(a.alight(q), e.alight(q))].concat()
    };
    for q in 0..PERIODS {
        for round in 0..=rounds {
            let t = (round > 0).then(|| total(a, e, q));
            let t0 = now_ms();
            match &t {
                None => assert!(a.solve(q)),
                Some(t) => assert!(a.crowd_with_external(q, t)),
            }
            release_skims();
            let t1 = now_ms();
            match &t {
                None => assert!(e.solve(q)),
                Some(t) => assert!(e.crowd_with_external(q, t)),
            }
            release_skims();
            ms[0] += t1 - t0;
            ms[1] += now_ms() - t1;
        }
        if rounds > 0 {
            let t = total(a, e, q);
            assert!(a.cap_with_external(q, &t));
            assert!(e.cap_with_external(q, &t));
        }
        each(q, a, e);
    }
    ms
}

/// A day's figures of one zone: per period commute and everyday trips, rail and walking, and
/// per station and line the boardings a day, both populations.
fn figures(a: &DemandApi, e: &EverydayApi) -> serde_json::Value {
    let (mut st, mut ln) = (vec![], vec![]);
    let mut per = vec![];
    for q in 0..PERIODS {
        let (sa, se) = (a.summary(q), e.summary(q));
        per.push(serde_json::json!([sa[1], sa[2], sa[3], se[1], se[2], se[3]]));
        let (rs, rl) = a.route_nodes(q);
        let ns = rs.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
        let nl = rl.iter().map(|&l| l as usize + 1).max().unwrap_or(0);
        st.resize(st.len().max(ns), 0.0f64);
        ln.resize(ln.len().max(nl), 0.0f64);
        for (r, (&b0, &b1)) in a.board(q).iter().zip(&e.board(q)).enumerate() {
            st[rs[r] as usize] += (b0 + b1) as f64;
            ln[rl[r] as usize] += (b0 + b1) as f64;
        }
    }
    serde_json::json!({"periods": per, "stations": st, "lines": ln})
}

/// What a solved zone holds by part, in MB, found by dropping the parts one by one: per
/// population its periods' networks and graphs and the rest of its periods; the everyday
/// layer's pair probabilities, samples and pairs, and its own engine (`with_weights`); then the
/// commutes' engine.
fn held_parts(a: DemandApi, e: EverydayApi) -> serde_json::Value {
    let mut last = alloc::current();
    let mut step = || {
        let now = alloc::current();
        let d = alloc::mb(last.saturating_sub(now));
        last = now;
        d
    };
    let (mut ev_graph, mut ev_prob, mut ev_period) = (0.0, 0.0, 0.0);
    let EverydayApi { engine, per, destinations, edge_pair, pairs, live, first, .. } = e;
    for st in per.into_iter().flatten() {
        ev_period += step();
        let Period { net, g, probability, .. } = st;
        drop((net, g));
        ev_graph += step();
        drop(probability);
        ev_prob += step();
    }
    ev_period += step();
    drop((destinations, live, first));
    let samples = step();
    drop((edge_pair, pairs));
    let ev_pairs = step();
    drop(engine);
    let ev_engine = step();
    let mut a = a;
    let (mut c_graph, mut c_period) = (0.0, 0.0);
    for st in a.per.iter_mut().filter_map(|v| v.take()) {
        c_period += step();
        let crate::demand::api::Period { net, g, .. } = st;
        drop((net, g));
        c_graph += step();
    }
    c_period += step();
    drop(a);
    let commute = step();
    serde_json::json!({
        "everyday_period_graphs": ev_graph, "everyday_probabilities": ev_prob, "everyday_period_rest": ev_period,
        "samples": samples, "everyday_pairs": ev_pairs, "everyday_engine": ev_engine,
        "commute_period_graphs": c_graph, "commute_period_rest": c_period, "commute_engine": commute,
    })
}

/// T-1222 stage 0: each zone's memory by part, its times and its figures at the recipes asked
/// for, a JSON line a run into `TW_LOD_OUT`. Zones: `TW_LOD_PACKS` (ids, in `TW_LOD_DIR`, or
/// `nyc` from data/packs on its real network); recipes: `TW_LOD_RECIPES`, `name:per:min:max:few:chunk`
/// separated by `;`; each recipe is drawn at salts 0..`TW_LOD_SEEDS` (fine only at 0 unless
/// `TW_LOD_FINE_SEEDS`). notes/T-1206.md.
/// `cargo test --release --features demand t1222_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn t1222_probe() {
    use std::io::Write;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let env = |k: &str| std::env::var(k).ok();
    let dir = env("TW_LOD_DIR").unwrap_or(format!("{root}/data/packs"));
    let out = env("TW_LOD_OUT").expect("TW_LOD_OUT");
    let rounds: usize = env("TW_LOD_ROUNDS").map_or(3, |v| v.parse().unwrap());
    let seeds: u32 = env("TW_LOD_SEEDS").map_or(4, |v| v.parse().unwrap());
    let fine_seeds: u32 = env("TW_LOD_FINE_SEEDS").map_or(1, |v| v.parse().unwrap());
    let recipes: Vec<(String, Sampling, f32)> = env("TW_LOD_RECIPES")
        .unwrap_or("fine:8:2:16:2:0.1;coarse:64:1:4:1:1".into())
        .split(';')
        .map(|r| {
            let f: Vec<&str> = r.split(':').collect();
            let n = |i: usize| f[i].parse::<f64>().unwrap();
            (f[0].to_string(), Sampling { per: n(1), min: n(2) as usize, max: n(3) as usize, max_few: n(4) as usize, site: if f[0] == "fine" { Sampling::FINE.site } else { None }, salt: 0 }, n(5) as f32)
        })
        .collect();
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&out).unwrap();
    for id in env("TW_LOD_PACKS").expect("TW_LOD_PACKS").split(',') {
        let (h, b) = if id == "nyc" { pack::load_raw(&format!("{root}/data/packs/nyc.json")) } else { pack::load_raw(&format!("{dir}/{id}.json")) }.unwrap();
        let net = {
            let a = DemandApi::new(&h, &b).unwrap();
            if id == "nyc" { RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json")).0 } else { zone_net(&a) }
        };
        let hop_km = net.hop_km();
        for (name, sampling, chunk) in &recipes {
            let n_salts = if name == "fine" { fine_seeds } else { seeds };
            for salt in 0..n_salts {
                let lod = Lod { level: 9, sampling: Sampling { salt, ..*sampling }, chunk: *chunk };
                alloc::reset_peak();
                let h0 = alloc::current();
                let t0 = now_ms();
                let mut a = DemandApi::new(&h, &b).unwrap();
                a.lod = lod;
                a.params_mut().commute_chunk = lod.chunk;
                a.params_mut().chunk_salt = salt;
                let (h1, t1) = (alloc::current(), now_ms());
                let mut e = EverydayApi::beside_with(&a, &h, &b, &[]).unwrap();
                let (h2, t2) = (alloc::current(), now_ms());
                let samples = 4 * (e.destinations.capacity() + e.live.iter().chain(&e.first).map(|v| v.capacity()).sum::<usize>());
                a.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
                let (h3, t3) = (alloc::current(), now_ms());
                e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
                let (h4, t4) = (alloc::current(), now_ms());
                let pairs = 20 * e.pairs.capacity() + 4 * e.edge_pair.capacity() + e.sz_part.capacity();
                a.set_fares(2.9, 0.1, &hop_km).unwrap();
                e.set_fares(2.9, 0.1, &hop_km).unwrap();
                let mut held = vec![];
                let ms = solve_day(&mut a, &mut e, rounds, |_, _, _| held.push(alloc::current()));
                let t5 = now_ms();
                let mb = |v: usize| alloc::mb(v);
                let (held_mb, peak_mb) = (mb(alloc::current() - h0), mb(alloc::peak() - h0));
                let pop: f64 = a.city.pop.iter().map(|&v| v as f64).sum();
                let dense: f64 = a.city.pop.iter().filter(|&&v| v / CELL_KM2 > 5000.0).map(|&v| v as f64).sum();
                let line = serde_json::json!({
                    "zone": id, "recipe": name, "salt": salt, "rounds": rounds,
                    "cells": a.city.len(), "zones": a.z.n, "pop": pop, "dense_share": dense / pop.max(1.0),
                    "stations": net.st_xy.len() / 2, "samples": e.destinations.len(), "pairs": e.pairs.len(), "subzones": a.n_subzones(), "everyday_subzones": e.engine.n_subzones(),
                    "mb": {
                        "commute_open": mb(h1 - h0), "everyday_open": mb(h2 - h1), "samples": mb(samples), "everyday_engine_open": mb(h2 - h1 - samples),
                        "commute_network": mb(h3 - h2), "everyday_network": mb(h4 - h3), "everyday_pairs": mb(pairs), "everyday_engine_network": mb(h4 - h3 - pairs),
                        "periods": held.iter().scan(h4, |last, &v| { let d = v as f64 - *last as f64; *last = v; Some(d / 1048576.0) }).collect::<Vec<f64>>(),
                        "held": held_mb, "peak": peak_mb,
                    },
                    "ms": {"commute_open": t1 - t0, "everyday_open": t2 - t1, "commute_network": t3 - t2, "everyday_network": t4 - t3, "commute_solve": ms[0], "everyday_solve": ms[1], "day": t5 - t0},
                    "figures": figures(&a, &e),
                    "parts": held_parts(a, e),
                });
                writeln!(file, "{line}").unwrap();
                println!("{id} {name} salt {salt}: {:.0} ms, held {held_mb:.0} MB, peak {peak_mb:.0} MB", t5 - t0);
            }
        }
    }
}

/// The synthetic city's inner 12 km as an everyday pack (errands at the jobs, leisure at a
/// third of the people), its gravity shipped, and the cross through its middle.
fn small_zone() -> (String, Vec<u8>, RealNet) {
    let mut city = synth::synth_city(1).inner(12_000.0);
    let p = Params::default();
    let z = kernel::build_zones(&city, p.zone_m);
    let g = kernel::gravity(&z, &p, 100, 1e-5);
    city.gravity = Some(kernel::to_pack_gravity(&z, &g, &p));
    let (h, mut b) = pack::write(&city);
    let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
    let leisure: Vec<f32> = city.pop.iter().map(|v| v / 3.0).collect();
    for (name, v) in [("errand_jobs", &city.jobs), ("leisure_jobs", &leisure)] {
        let offset = b.len();
        b.extend(v.iter().flat_map(|v| v.to_le_bytes()));
        h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":v.len(),"offset":offset}));
    }
    let net = cross_net(&city.x, &city.y, &city.pop, 8000.0, 1000.0);
    (h.to_string(), b, net)
}

fn day_of(a: &mut DemandApi, e: &mut EverydayApi, net: &RealNet) -> (Vec<f64>, Vec<f64>) {
    a.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
    e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
    let hop_km = net.hop_km();
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    e.set_fares(2.9, 0.1, &hop_km).unwrap();
    let (mut sums, mut loads) = (vec![], vec![]);
    solve_day(a, e, 2, |q, a, e| {
        let no_ms = |s: Vec<f64>| s.into_iter().enumerate().filter(|&(i, _)| i != 4).map(|(_, v)| v);
        sums.extend(no_ms(a.summary(q)).chain(no_ms(e.summary(q))));
        loads.extend(a.seg(q).iter().chain(&a.board(q)).chain(&e.seg(q)).chain(&e.board(q)).map(|&v| v as f64));
    });
    (sums, loads)
}

/// T-1222: a zone opened fine is today's to the bit; one opened coarse keeps its rail on
/// average over draws (notes/T-1206.md section 4), with far fewer samples.
#[test]
fn levels_fine_is_today_coarse_keeps_rail() {
    let (h, b, net) = small_zone();
    let today = {
        let mut a = DemandApi::new(&h, &b).unwrap();
        let mut e = EverydayApi::beside_with(&a, &h, &b, &[]).unwrap();
        (e.destinations.clone(), day_of(&mut a, &mut e, &net))
    };
    let mut a = DemandApi::at_level(&h, &b, lod::FINE).unwrap();
    assert_eq!(a.level(), lod::FINE);
    let mut e = EverydayApi::beside_with(&a, &h, &b, &[]).unwrap();
    assert_eq!(e.level(), lod::FINE);
    assert!(e.destinations == today.0, "the fine samples are today's");
    let fine = day_of(&mut a, &mut e, &net);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(bits(&fine.0) == bits(&today.1 .0) && bits(&fine.1) == bits(&today.1 .1), "fine figures to the bit");
    let rail = |s: &[f64]| -> [f64; 3] {
        // per period: the commutes' summary then the everyday one, 12 numbers each (ms left out)
        let n = 12;
        let (mut c, mut d, mut t) = (0.0, 0.0, 0.0);
        for p in s.chunks(2 * n) {
            c += p[2];
            d += p[n + 2];
            t += p[1] + p[n + 1];
        }
        [c, d, t]
    };
    let coarse = DemandApi::at_level(&h, &b, lod::COARSE).unwrap();
    assert_eq!(coarse.level(), lod::COARSE);
    assert_eq!(coarse.base.commute_chunk, lod::COARSE_CHUNK);
    // both levels over the same draws (draw 0 is the game's): one draw of fine is itself as far
    // from the mean as the bound
    let draws = 6;
    let mean = |level: u32| -> ([f64; 3], usize) {
        let (mut sum, mut samples) = ([0.0; 3], 0);
        for salt in 0..draws {
            let mut lod = Lod::of(level).unwrap();
            lod.sampling.salt = salt;
            let (mut a, mut e) = open(&h, &b, lod);
            samples += e.destinations.len();
            let r = rail(&day_of(&mut a, &mut e, &net).0);
            for k in 0..3 {
                sum[k] += r[k] / draws as f64;
            }
        }
        (sum, samples)
    };
    let (f, fine_samples) = mean(lod::FINE);
    let (c, coarse_samples) = mean(lod::COARSE);
    assert!(coarse_samples * 2 < fine_samples, "{coarse_samples} samples against {fine_samples}");
    assert!((c[2] / f[2] - 1.0).abs() < 0.002, "trips {} against {}", c[2], f[2]);
    let bias = (c[0] + c[1]) / (f[0] + f[1]) - 1.0;
    println!("coarse against fine over {draws} draws: rail {:+.3}% (commutes {:+.3}%, everyday {:+.3}%), {coarse_samples} samples against {fine_samples}", 100.0 * bias, 100.0 * (c[0] / f[0] - 1.0), 100.0 * (c[1] / f[1] - 1.0));
    assert!(DemandApi::at_level(&h, &b, 2).is_err());
    assert!(bias.abs() < 0.005, "rail {:+.3}% on average", 100.0 * bias);
}

/// T-1543: one period's free flow solved in `n` parts of the origins (and the other periods
/// estimated from each part) against the whole, figure by figure: the worst gap between the
/// parts' sum and the whole's, as a share of the figure's largest value. Then each part's copy
/// solves the period whole again, which must be a fresh solve's to the bit.
fn parts_against_whole(h: &str, b: &[u8], net: &RealNet, n: u32, from: usize) -> Vec<(String, usize, f64)> {
    let hop_km = net.hop_km();
    let setup = || {
        let mut a = DemandApi::new(h, b).unwrap();
        let mut e = EverydayApi::beside_with(&a, h, b, &[]).unwrap();
        a.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
        e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        e.set_fares(2.9, 0.1, &hop_km).unwrap();
        (a, e)
    };
    // every figure a result carries, per period, and the bubbles' sums over every period
    let figures = |a: &DemandApi, e: &EverydayApi, q: usize| -> Vec<(&'static str, Vec<f64>)> {
        let f = |v: Vec<f32>| v.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let pick = |s: Vec<f64>| [1, 2, 3, 5, 10, 11, 12].iter().map(|&k| s[k]).collect::<Vec<f64>>();
        vec![
            ("commute seg", f(a.seg(q))),
            ("commute board", f(a.board(q))),
            ("commute alight", f(a.alight(q))),
            ("commute transfers", f(a.transfers_in(q))),
            ("commute fare extra", f(a.fare_extra(q))),
            ("commute load of crush", f(a.load_of_crush(q))),
            ("commute summary", pick(a.summary(q))),
            ("everyday seg", f(e.seg(q))),
            ("everyday board", f(e.board(q))),
            ("everyday parts", f(e.part_loads(q))),
            ("everyday transfers", f(e.transfers_in(q))),
            ("everyday fare extra", f(e.fare_extra(q))),
            ("everyday summary", pick(e.summary(q))),
            ("everyday kinds", e.kind_totals(q)),
        ]
    };
    let (mut wa, mut we) = setup();
    // each copy finds its own strategies, as in its own worker
    release_skims();
    let t = now_ms();
    assert!(wa.solve(from));
    let ta = now_ms() - t;
    assert!(we.solve(from));
    println!("whole: commutes {ta:.0} ms, everyday {:.0} ms", now_ms() - t - ta);
    for q in (0..PERIODS).filter(|&q| q != from) {
        assert!(wa.estimate(from, q) && we.estimate(from, q));
    }
    let mut parts: Vec<(DemandApi, EverydayApi)> = (0..n).map(|_| setup()).collect();
    let t0 = now_ms();
    for (k, (a, e)) in parts.iter_mut().enumerate() {
        a.set_part(k as u32, n);
        e.set_part(k as u32, n);
        release_skims();
        let t = now_ms();
        assert!(a.solve(from));
        let ta = now_ms() - t;
        assert!(e.solve(from));
        println!("part {k} of {n}: commutes {ta:.0} ms, everyday {:.0} ms", now_ms() - t - ta);
        for q in (0..PERIODS).filter(|&q| q != from) {
            assert!(a.estimate(from, q) && e.estimate(from, q));
        }
    }
    println!("{n} parts {:.0} ms in all", now_ms() - t0);
    let mut out = vec![];
    for q in 0..PERIODS {
        let whole = figures(&wa, &we, q);
        let each: Vec<_> = parts.iter().map(|(a, e)| figures(a, e, q)).collect();
        for (i, (name, w)) in whole.iter().enumerate() {
            let scale = w.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-9);
            let gap = (0..w.len()).map(|j| (each.iter().map(|p| p[i].1[j]).sum::<f64>() - w[j]).abs()).fold(0.0, f64::max) / scale;
            out.push((name.to_string(), q, gap));
        }
    }
    // the bubbles: the commutes' sums and the everyday layer's legs per cell, over every period
    let sums = |v: Vec<Vec<f32>>| -> Vec<f64> { (0..v[0].len()).map(|j| v.iter().map(|x| x[j] as f64).sum()).collect() };
    for (name, w, p) in [
        ("commute sums", wa.sub_sums(), sums(parts.iter().map(|(a, _)| a.sub_sums()).collect())),
        ("everyday cell modes", we.cell_modes(), sums(parts.iter().map(|(_, e)| e.cell_modes()).collect())),
    ] {
        let scale = w.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-9) as f64;
        let gap = w.iter().zip(&p).map(|(&x, &y)| (x as f64 - y).abs()).fold(0.0, f64::max) / scale;
        out.push((name.to_string(), PERIODS, gap));
    }
    // each part's copy solves the period whole again, as after the parts' figures are out
    let (mut fa, mut fe) = setup();
    assert!(fa.solve(from) && fe.solve(from));
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let no_ms = |s: Vec<f64>| s.into_iter().enumerate().filter(|&(i, _)| i != 4).map(|(_, v)| v.to_bits()).collect::<Vec<_>>();
    for (a, e) in parts.iter_mut() {
        a.set_part(0, 1);
        e.set_part(0, 1);
        for q in (0..PERIODS).filter(|&q| q != from) {
            a.forget_period(q);
            e.forget_period(q);
        }
        assert!(a.solved_mask() == 1 << from && e.solved_mask() == 1 << from);
        assert!(a.solve(from) && e.solve(from));
        assert!(no_ms(a.summary(from)) == no_ms(fa.summary(from)) && no_ms(e.summary(from)) == no_ms(fe.summary(from)), "solved whole again: summaries to the bit");
        assert!(bits(a.seg(from)) == bits(fa.seg(from)) && bits(e.seg(from)) == bits(fe.seg(from)) && bits(a.transfers_in(from)) == bits(fa.transfers_in(from)) && bits(e.transfers_in(from)) == bits(fe.transfers_in(from)), "solved whole again: loads to the bit");
        assert!(bits(a.sub_sums()) == bits(fa.sub_sums()) && bits(e.cell_modes()) == bits(fe.cell_modes()), "solved whole again: sums to the bit");
    }
    out
}

/// T-1543: the parts of a period's origins add up to the whole period, and the parts' copies then
/// solve it whole to the bit. The everyday layer's estimated periods scale each part's transfers
/// and fare extras by its own pairs' rail (`EverydayApi::estimate`), so they are near, not equal.
#[test]
fn origin_parts_add_up_to_the_period() {
    let (h, b, net) = small_zone();
    for (name, q, gap) in parts_against_whole(&h, &b, &net, 3, 0) {
        let near = name.starts_with("everyday") && q != 0 && (name.ends_with("transfers") || name.ends_with("fare extra"));
        assert!(gap < if near { 0.05 } else { 1e-4 }, "{name}, period {q}: the parts' sum is {:.2e} of the largest figure off the whole", gap);
    }
}

/// T-1543 on New York's real network, timed (`TW_PARTS=3`, `--ignored --nocapture`).
#[test]
#[ignore]
fn origin_parts_on_new_york() {
    let n: u32 = std::env::var("TW_PARTS").map_or(3, |v| v.parse().unwrap());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let net = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json")).0;
    let gaps = parts_against_whole(&h, &b, &net, n, 0);
    for (name, q, gap) in gaps.iter().filter(|g| g.2 > 1e-6) {
        println!("{name}, period {q}: {gap:.2e}");
    }
    println!("worst: {:?}", gaps.iter().map(|g| g.2).fold(0.0, f64::max));
}

/// T-1874: New York's samples by purpose and its everyday opening and first network times, with
/// a campus's homes one per 30 trips (2 to 100) against T-397's one per 8 (2 to 16).
/// `cargo test --release --features demand t1874_samples -- --ignored --nocapture`
#[test]
#[ignore]
fn t1874_samples() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let net = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json")).0;
    for (name, sampling) in [("t397", Sampling { site: None, ..Sampling::FINE }), ("t1874", Sampling::FINE)] {
        let lod = Lod { level: 0, sampling, chunk: Lod::of(lod::FINE).unwrap().chunk };
        let (mut best_open, mut best_net) = (f64::MAX, f64::MAX);
        let mut last = None;
        for _ in 0..3 {
            let t = now_ms();
            let (_, mut e) = open(&h, &b, lod);
            best_open = best_open.min(now_ms() - t);
            let t = now_ms();
            e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
            best_net = best_net.min(now_ms() - t);
            last = Some(e);
        }
        let e = last.unwrap();
        let by: Vec<usize> = (0..e.totals.len()).map(|k| e.start[k + 1] - e.start[k]).collect();
        println!("{name}: samples {} {by:?}, pairs {}, totals {:?}, open {best_open:.0} ms, first network {best_net:.0} ms", e.destinations.len(), e.pairs.len(), e.totals.iter().map(|t| t.round()).collect::<Vec<_>>());
    }
}

/// T-1538: a period estimated from another's free flow has its own trips, and solving it after
/// replaces the estimate whole: its figures are a fresh solve's to the bit.
#[test]
fn estimate_keeps_trips_and_solve_replaces_it() {
    let (h, b, net) = small_zone();
    let hop_km = net.hop_km();
    let setup = || {
        let mut a = DemandApi::new(&h, &b).unwrap();
        let mut e = EverydayApi::beside_with(&a, &h, &b, &[]).unwrap();
        a.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
        e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        e.set_fares(2.9, 0.1, &hop_km).unwrap();
        (a, e)
    };
    let (mut a, mut e) = setup();
    let (mut fa, mut fe) = setup();
    assert!(!a.estimate(0, 1), "nothing to estimate from");
    a.solve(0);
    e.solve(0);
    assert!(!a.estimate(0, 0) && !a.estimate(0, PERIODS));
    let no_ms = |s: Vec<f64>| s.into_iter().enumerate().filter(|&(i, _)| i != 4).map(|(_, v)| v.to_bits()).collect::<Vec<_>>();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for q in 1..PERIODS {
        assert!(a.estimate(0, q) && e.estimate(0, q));
        fa.solve(q);
        fe.solve(q);
        let near = |x: f64, y: f64| (x - y).abs() <= 1e-6 * y.abs().max(1.0);
        assert!(near(a.summary(q)[1], fa.summary(q)[1]) && near(e.summary(q)[1], fe.summary(q)[1]), "period {q}: the estimate's trips are the period's");
        assert!(a.summary(q)[2] > 0.0 && e.summary(q)[2] > 0.0, "period {q}: the estimate has rail");
        a.solve(q);
        e.solve(q);
        assert!(no_ms(a.summary(q)) == no_ms(fa.summary(q)) && no_ms(e.summary(q)) == no_ms(fe.summary(q)), "period {q}: summaries to the bit");
        assert!(bits(a.seg(q)) == bits(fa.seg(q)) && bits(e.seg(q)) == bits(fe.seg(q)) && bits(a.board(q)) == bits(fa.board(q)), "period {q}: loads to the bit");
    }
}
