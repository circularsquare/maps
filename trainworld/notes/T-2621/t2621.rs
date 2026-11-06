/// T-2621 (scratch only, included at the end of a copy's intercity/tests.rs, with the scratch
/// timers and the coarse switches in that copy's pass.rs, search.rs and logsum.rs): the market at
/// free flow on a captured net, exact and in each coarse mode, every period in `TW_Q`; each pass's
/// time by part and the market's boardings by line.
/// `TW_T780_NET=<net> TW_T2621_OUT=<out.json> TW_Q=0,1,2,3,4 TW_MODES="exact;far=50,t=15,r=8,cont=1,bound=1,slack=30"
///   cargo test --release --features demand t2621_dump -- --ignored --nocapture`
#[test]
#[ignore]
fn t2621_dump() {
    use super::pass::{Coarse, COARSE, T2621};
    let qs: Vec<usize> = std::env::var("TW_Q").unwrap_or("1".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let modes: Vec<String> = std::env::var("TW_MODES").unwrap_or("exact".into()).split(';').map(|s| s.to_string()).collect();
    let net_path = std::env::var("TW_T780_NET").unwrap();
    let t = now_ms();
    let mut a = t780_net(&net_path, MARKET_CHUNK);
    let t_net = now_ms() - t;
    // the stations' longitude and latitude, as t780_net has them
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let fr = &j["frame"];
    let frame = Frame::equirect(fr["lon"].as_f64().unwrap(), fr["lat"].as_f64().unwrap()).with_kind(crate::track::frame::FrameKind::Merc);
    let xy: Vec<f64> = j["stXY"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let ll: Vec<(f64, f64)> = xy.chunks_exact(2).map(|p| frame.to_lonlat(p[0], p[1])).collect();
    let km = |a: usize, b: usize| {
        let (la1, la2) = (ll[a].1.to_radians(), ll[b].1.to_radians());
        let (dla, dlo) = (la2 - la1, (ll[b].0 - ll[a].0).to_radians());
        let h = (dla / 2.).sin().powi(2) + la1.cos() * la2.cos() * (dlo / 2.).sin().powi(2);
        2. * 6371.0 * h.sqrt().asin()
    };
    // the pairs' zones and distances, for the error by distance
    let m = &a.m;
    let pairs: Vec<serde_json::Value> = a.pairs.iter().map(|p| serde_json::json!([p.a, p.b, (m.zone_km(&m.zones[p.a as usize], &m.zones[p.b as usize]) * 10.).round() / 10.])).collect();
    let mut out = vec![];
    for mode in &modes {
        let kv: HashMap<String, f64> = mode
            .split(',')
            .filter_map(|e| e.split_once('='))
            .map(|(k, v)| (k.to_string(), v.parse::<f64>().unwrap()))
            .collect();
        let on = kv.contains_key("far");
        // a longer chunk with distance (`cd0` km, `ccap`): the pairs made again
        // and far place pairs from `pd` km dealt over both places' zones
        let ch = (kv.get("cd0").copied().unwrap_or(0.), kv.get("ccap").copied().unwrap_or(1.));
        let pd = kv.get("pd").copied().unwrap_or(0.);
        let reps = kv.get("reps").copied().unwrap_or(0.);
        if ch != super::api::CHUNK_BY_KM.with(|c| c.get()) || pd != super::api::PLACE_DEAL_KM.with(|c| c.get()) || reps != super::api::REP_PEOPLE.with(|c| c.get()) {
            super::api::CHUNK_BY_KM.with(|c| c.set(ch));
            super::api::PLACE_DEAL_KM.with(|c| c.set(pd));
            super::api::REP_PEOPLE.with(|c| c.set(reps));
            let t = now_ms();
            a = t780_net(&net_path, MARKET_CHUNK);
            eprintln!("{mode}: set up again, {} longer pairs, {:.0} ms", a.pairs.len(), now_ms() - t);
        }
        if mode == "searched" {
            std::env::set_var("TW_T2621_SEARCHED", "1");
        } else {
            std::env::remove_var("TW_T2621_SEARCHED");
        }
        let mut periods = vec![];
        let mut hubs_n = 0;
        let mut t_hubs = 0.;
        for &q in &qs {
            // hubs on this period's graph: the stations with the most lines first, each taking the
            // stations not yet taken within `r` km and `t` rail minutes of it
            COARSE.with(|c| *c.borrow_mut() = Coarse::default());
            if on {
                // (the graph of the last solve of q: the same at free flow)
                if a.per(q).is_none() {
                    assert!(a.solve(q));
                }
                let t0 = now_ms();
                let g = &a.per(q).unwrap().g;
                let s = g.n_st;
                let mut lines: Vec<Vec<u32>> = vec![vec![]; s];
                let mut tph = vec![0f32; s];
                for r in 0..g.rn_line.len() {
                    if g.rn_tph[r] > 0. {
                        let v = g.rn_station[r] as usize;
                        if !lines[v].contains(&g.rn_line[r]) {
                            lines[v].push(g.rn_line[r]);
                        }
                        tph[v] += g.rn_tph[r];
                    }
                }
                let mut order: Vec<usize> = (0..s).collect();
                order.sort_by(|&x, &y| lines[y].len().cmp(&lines[x].len()).then(tph[y].total_cmp(&tph[x])).then(x.cmp(&y)));
                let (tmax, rmax) = (kv.get("t").copied().unwrap_or(15.) as f32, kv.get("r").copied().unwrap_or(8.));
                let mut hub = vec![u32::MAX; s];
                let mut d = vec![f32::INFINITY; g.n];
                let mut seen: Vec<usize> = vec![];
                for &h in order.iter().take(if kv.contains_key("nohub") { 0 } else { usize::MAX }) {
                    if hub[h] != u32::MAX {
                        continue;
                    }
                    hub[h] = h as u32;
                    hubs_n += 1;
                    if tmax <= 0. {
                        continue;
                    }
                    // rail minutes from h, forward along the graph's edges, to `tmax`
                    let mut heap = std::collections::BinaryHeap::new();
                    d[h] = 0.;
                    seen.push(h);
                    heap.push((std::cmp::Reverse(0u32), h as u32));
                    while let Some((std::cmp::Reverse(k), v)) = heap.pop() {
                        let v = v as usize;
                        let dv = f32::from_bits(k);
                        if dv > d[v] {
                            continue;
                        }
                        if v < s && hub[v] == u32::MAX && km(h, v) <= rmax {
                            hub[v] = h as u32;
                        }
                        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                            let (to, c) = (g.adj_to[e] as usize, dv + g.adj_cost[e].max(0.));
                            if c <= tmax && c < d[to] {
                                if d[to].is_infinite() {
                                    seen.push(to);
                                }
                                d[to] = c;
                                heap.push((std::cmp::Reverse(c.to_bits()), to as u32));
                            }
                        }
                    }
                    for &v in &seen {
                        d[v] = f32::INFINITY;
                    }
                    seen.clear();
                }
                t_hubs += now_ms() - t0;
                COARSE.with(|c| {
                    *c.borrow_mut() = Coarse {
                        on: true,
                        far_km: kv.get("far").copied().unwrap_or(50.),
                        cont: kv.get("cont").copied().unwrap_or(1.) != 0.,
                        bound: kv.get("bound").copied().unwrap_or(1.) != 0.,
                        slack: kv.get("slack").copied().unwrap_or(30.) as f32,
                        hub: std::rc::Rc::new(if kv.contains_key("nohub") { vec![0; hub.len()] } else { hub }),
                    }
                });
            }
            T2621.with(|c| c.set([0.; 24]));
            let t = now_ms();
            assert!(a.solve(q));
            let t_solve = now_ms() - t;
            let tm = T2621.with(|c| c.get());
            let per = a.per(q).unwrap();
            let g = &per.g;
            let nl = g.rn_line.iter().copied().max().map_or(0, |l| l as usize + 1);
            let by_line = |v: &[f32]| {
                let mut o = vec![0f64; nl];
                for (r, &x) in v.iter().enumerate() {
                    o[g.rn_line[r] as usize] += x as f64;
                }
                o
            };
            let all = by_line(&per.avg.rn_board);
            let com = by_line(&per.c_avg.rn_board);
            let mut ev = vec![0f64; nl];
            for e in per.e_avg.iter().flatten() {
                for (o, x) in ev.iter_mut().zip(by_line(&e.rn_board)) {
                    *o += x;
                }
            }
            // rider-minutes by line, from the segments' loads and ride minutes
            let seg_min: Vec<f64> = {
                let mut o = vec![0f64; nl];
                for (r, &x) in per.avg.seg.iter().enumerate() {
                    o[g.rn_line[r] as usize] += x as f64 * g.rn_ride[r] as f64;
                }
                o
            };
            eprintln!(
                "{mode} q{q}: solve {t_solve:.0} ms (skims {:.0}, pass {:.0}: searches {:.0} [{} full, {} bounded, {:.1}M pops], fare km {:.0}, longer {:.0} [{} far, {} near, {:.1}M terms; set {:.0}, logsum {:.0}, choice {:.0}, spread {:.0}], commutes {:.0}, everyday {:.0}, assign {:.0} (again {:.0}), needs {:.0}); rail {:.1}, commutes {:.1}, everyday {:.1}",
                tm[0], tm[9], tm[1], tm[10], tm[11], tm[12] / 1e6, tm[2], tm[3], tm[13], tm[14], tm[15] / 1e6, tm[16], tm[17], tm[18], tm[19], tm[4], tm[5], tm[6] - tm[7], tm[7], tm[8], per.rail, per.c_rail, per.e_rail
            );
            periods.push(serde_json::json!({
                "q": q, "solve_ms": t_solve, "tm": tm.to_vec(), "rail": per.rail, "c_rail": per.c_rail, "e_rail": per.e_rail,
                "line_all": all, "line_com": com, "line_ev": ev, "line_min": seg_min,
                "pair_rail": per.pair_rail.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
            }));
        }
        out.push(serde_json::json!({"mode": mode, "hubs": hubs_n / qs.len().max(1), "hubs_ms": t_hubs, "periods": periods}));
    }
    COARSE.with(|c| *c.borrow_mut() = Coarse::default());
    let j = serde_json::json!({"net": net_path, "set_up_ms": t_net, "stations": ll.len(), "modes": out, "pairs": pairs});
    std::fs::write(std::env::var("TW_T2621_OUT").unwrap(), j.to_string()).unwrap();
    eprintln!("set up {t_net:.0} ms");
}
