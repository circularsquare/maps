/// T-2606 (scratch only, included at the end of a copy's intercity/tests.rs): the market at free
/// flow on a captured net, every period in `TW_Q` (default 0,1), each pair's rail, its zones and
/// their boxes, the stations each zone reaches, and each pass's time.
/// `TW_T780_NET=<net> TW_T2606_OUT=<out.json> TW_Q=0,1 cargo test --release --features demand t2606_dump -- --ignored --nocapture`
#[test]
#[ignore]
fn t2606_dump() {
    let qs: Vec<usize> = std::env::var("TW_Q").unwrap_or("0,1".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let t = now_ms();
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), MARKET_CHUNK);
    let t_net = now_ms() - t;
    let mut periods = vec![];
    for &q in &qs {
        let t = now_ms();
        assert!(a.solve(q));
        let t_solve = now_ms() - t;
        let m = &a.m;
        let per = a.per(q).unwrap();
        let mut long = vec![];
        for (pi, p) in a.pairs.iter().enumerate() {
            let r = per.pair_rail[pi];
            if r > 0. {
                long.push(serde_json::json!([p.a, p.b, p.trips * SHARES[q] as f64, r]));
            }
        }
        let mut com = vec![];
        for (ci, cp) in a.cpairs.iter().enumerate() {
            let (h, wg, n) = m.commutes[cp.c as usize];
            let w = m.comp_of_gid[wg as usize];
            let legs = n as f64 * (TO_WORK[q] + TO_HOME[q]) as f64;
            let f = per.c_frac[ci] as f64;
            if f > 0. {
                com.push(serde_json::json!([h, w, legs, legs * f]));
            }
        }
        let mut ev = vec![];
        for (ei, ep) in a.epairs.iter().enumerate() {
            let e = m.everyday[ep.e as usize];
            let w = m.comp_of_gid[e.a as usize];
            let k = e.kind as usize;
            let legs = e.trips as f64 * (OUTWARD[k][q] + RETURN[k][q]) as f64;
            let f = per.e_frac[ei] as f64;
            if f > 0. {
                ev.push(serde_json::json!([e.h, w, legs, legs * f]));
            }
        }
        let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        eprintln!("q{q}: solve {t_solve:.0} ms, rail {:.1}, commutes {:.1}, everyday {:.1}", per.rail, per.c_rail, per.e_rail);
        periods.push(serde_json::json!({
            "q": q, "solve_ms": t_solve, "rail": per.rail, "c_rail": per.c_rail, "e_rail": per.e_rail,
            "long": long, "com": com, "ev": ev, "seg": f(&per.avg.seg), "board": f(&per.avg.rn_board),
        }));
    }
    let m = &a.m;
    let per = a.per(qs[0]).unwrap();
    let zones: Vec<serde_json::Value> = m.zones.iter().map(|z| serde_json::json!([m.places[z.place].zone, z.x, z.y, z.pop, z.jobs])).collect();
    let access: Vec<Vec<usize>> = (0..m.zones.len())
        .map(|z| {
            let mut s: Vec<usize> = [&a.access, &a.access_work, &a.access_c, &a.access_cw, &a.access_ew]
                .iter()
                .filter_map(|v| v.get(z))
                .flat_map(|l| l.iter().map(|e| e.0))
                .collect();
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    let j = serde_json::json!({
        "access": access, "st_map": a.station_map(), "boxes": a.zone_boxes(),
        "set_up_ms": t_net, "stations": per.g.n_st, "wanted": a.want.iter().filter(|&&w| w).count(),
        "zones": zones, "rn_line": per.g.rn_line, "rn_station": per.g.rn_station, "periods": periods,
    });
    std::fs::write(std::env::var("TW_T2606_OUT").unwrap(), j.to_string()).unwrap();
    eprintln!("set up {t_net:.0} ms");
}
