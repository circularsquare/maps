//! T-710: the T-709 trips on a city's real network after a peak-period solve with its crowding
//! rounds (as `bin/calib.rs`: commutes and everyday trips share the loads), with a parameter set.
//!
//!   T710_CITY=nyc T710_SET=station_theta=0.5,station_ref=100 T710_OUT=out.json \
//!     cargo test --release --features demand t710_trips -- --ignored --nocapture

use crate::demand::api::DemandApi;
use crate::demand::everyday::EverydayApi;
use crate::demand::kernel;
use crate::demand::pack;
use crate::demand::trip::{HEAD, LEG_LEN, PAIR_LEN};
use serde_json::{json, Value};

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore]
fn t710_trips() {
    let city = std::env::var("T710_CITY").unwrap_or("nyc".into());
    let sets = std::env::var("T710_SET").unwrap_or_default();
    let out_path = std::env::var("T710_OUT").expect("T710_OUT");
    let root = std::env::var("T710_ROOT").expect("T710_ROOT: the trainworld folder");
    let net: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/{city}_real.json")).unwrap()).unwrap();
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/{}.json", net["pack"].as_str().unwrap())).unwrap();
    let hv: Value = serde_json::from_str(&h).unwrap();
    let (lon0, lat0) = (hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
    let mut api = DemandApi::new(&h, &b).unwrap();
    for kv in sets.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        api.params_mut().set_extra(k, v.parse().unwrap()).unwrap();
    }
    if !sets.is_empty() {
        api.refresh_walk();
    }
    let mut ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let d = &net["demand"];
    let (st_xy, line_n, stops, times, tph, cars) = (arr_f32(d, "st_xy"), arr_u32(d, "line_n"), arr_u32(d, "stops"), arr_f32(d, "times"), arr_f32(d, "tph"), arr_f32(d, "cars"));
    let mut hop_km = vec![];
    let mut off = 0usize;
    for &n in &line_n {
        let n = n as usize;
        for dir in 0..2 {
            for i in 0..n {
                let k = if dir == 0 { off + i } else { off + n - 1 - i };
                hop_km.push(if i + 1 < n {
                    let nx = if dir == 0 { k + 1 } else { k - 1 };
                    let (s0, s1) = (stops[k] as usize, stops[nx] as usize);
                    (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                } else {
                    0.0
                });
            }
        }
        off += n;
    }
    api.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    ev.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    api.set_fares(1.5, 0.1, &hop_km).unwrap();
    ev.set_fares(1.5, 0.1, &hop_km).unwrap();
    let q = 0;
    api.solve(q);
    ev.solve(q);
    for _ in 0..3 {
        let nr = api.seg(q).len();
        let mut total = vec![0f32; 3 * nr];
        for (at, v) in [(0, api.seg(q)), (1, api.board(q)), (2, api.alight(q)), (0, ev.seg(q)), (1, ev.board(q)), (2, ev.alight(q))] {
            for (t, x) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                *t += x;
            }
        }
        assert!(api.crowd_with_external(q, &total));
        assert!(ev.crowd_with_external(q, &total));
    }
    // names: demand station -> the first network station in it; route node -> line
    let smap = api.station_map();
    let names: Vec<String> = net["stations"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap_or("?").to_string()).collect();
    let mut dname = vec![String::from("?"); smap.iter().map(|&x| x as usize + 1).max().unwrap_or(0)];
    for (i, &ds) in smap.iter().enumerate().rev() {
        dname[ds as usize] = names[i].clone();
    }
    let lines: Vec<String> = net["lines"].as_array().unwrap().iter().map(|l| l["name"].as_str().unwrap_or("?").to_string()).collect();
    let (rn_st, rn_line) = api.route_nodes(q);

    // the station-to-station costs' range (f32 limit of a sharp theta, `station_ref`)
    let st = api.period(q).unwrap();
    let sz = api.sz.as_ref().unwrap();
    let sk = kernel::skims_for(&st.g, &st.p, &kernel::egress_stations(sz, st.g.n_st));
    let fin: Vec<f32> = sk.r.iter().cloned().filter(|r| r.is_finite()).collect();
    let mut srt = fin.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| srt[((srt.len() as f64 - 1.0) * p) as usize];
    let over = |m: f32| fin.iter().filter(|&&r| r > m).count() as f64 / fin.len().max(1) as f64;
    let range = json!({"n": fin.len(), "p50": pct(0.5), "p99": pct(0.99), "max": pct(1.0), "over140": over(140.0), "over200": over(200.0), "over240": over(240.0)});
    eprintln!("station-to-station costs: {range}");

    let at = |lon: f64, lat: f64| {
        const R: f64 = 6371008.8;
        let rad = std::f64::consts::PI / 180.0;
        ((R * (lon - lon0) * rad * (lat0 * rad).cos()) as f32, (R * (lat - lat0) * rad) as f32)
    };
    let trips: Vec<(&str, (f64, f64), (f64, f64))> = if city == "tor" {
        vec![("Bloor to Union", (-79.3857, 43.6709), (-79.3806, 43.6453)), ("Eglinton to Queen", (-79.3985, 43.7059), (-79.3791, 43.6525))]
    } else if city == "lon" {
        vec![
            ("Wimbledon to Waterloo", (-0.2064, 51.4214), (-0.1134, 51.5031)),
            ("Richmond to Waterloo", (-0.3016, 51.4633), (-0.1134, 51.5031)),
            ("Stratford to Bank", (-0.0035, 51.5413), (-0.0886, 51.5133)),
            ("Brixton to Oxford Circus", (-0.1146, 51.4626), (-0.1419, 51.5152)),
        ]
    } else {
        vec![
            ("E 98 St (Lexington) to Bleecker St", (-73.9475, 40.7864), (-73.9946, 40.7254)),
            ("W 98 St (Broadway) to Houston St", (-73.9700, 40.7960), (-74.0054, 40.7282)),
            ("Jamaica to Penn Station", (-73.8081, 40.7000), (-73.9935, 40.7506)),
            ("SoHo to Times Sq", (-74.0005, 40.7233), (-73.9866, 40.7559)),
            ("Union Sq to 23 St", (-73.9906, 40.7359), (-73.9893, 40.7415)),
            ("Brooklyn (Nostrand) to Rockefeller", (-73.9505, 40.6804), (-73.9790, 40.7587)),
            ("Hempstead to Rockefeller", (-73.6213, 40.7134), (-73.9790, 40.7587)),
            ("Queens (75 Av) to Wall St", (-73.8370, 40.7181), (-74.0090, 40.7068)),
            ("Bronx (Jackson Av) to 49 St", (-73.9078, 40.8165), (-73.9843, 40.7599)),
            ("Harlem to Union Sq", (-73.9452, 40.8076), (-73.9906, 40.7359)),
        ]
    };
    let mut out = vec![];
    for (what, o, dd) in trips {
        let (o, dd) = (at(o.0, o.1), at(dd.0, dd.1));
        let v = api.trip(q, o.0, o.1, dd.0, dd.1);
        let np = v[24] as usize;
        let shares: Vec<f64> = (0..np).map(|k| v[HEAD + k * PAIR_LEN + 9]).collect();
        let mut acc = 0.0;
        let mut n80 = 0;
        for &s in &shares {
            if acc >= 0.8 {
                break;
            }
            acc += s;
            n80 += 1;
        }
        let pairs: Vec<Value> = (0..np.min(5))
            .map(|k| {
                let p = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
                json!({"share": p[9], "from": dname[p[0] as usize], "to": dname[p[1] as usize], "walk_from": p[5], "walk_to": p[7], "strategy": p[8], "ride": p[12]})
            })
            .collect();
        // where riders leave the train: share and perceived walk by egress station, all pairs
        let mut egress: Vec<(String, f64, f64)> = vec![];
        for k in 0..np {
            let p = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
            let name = dname[p[1] as usize].clone();
            match egress.iter_mut().find(|e| e.0 == name) {
                Some(e) => e.1 += p[9],
                None => egress.push((name, p[9], p[7])),
            }
        }
        egress.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let egress: Vec<Value> = egress.iter().take(6).map(|e| json!({"station": e.0, "share": e.1, "walk": e.2})).collect();
        let mut by_rides = [0f64; 4];
        let mut o2 = HEAD + np * PAIR_LEN;
        let mut routes = vec![];
        for _ in 0..v[25] as usize {
            let (pi, share, n) = (v[o2] as usize, v[o2 + 1], v[o2 + 2] as usize);
            let pp = &v[HEAD + pi * PAIR_LEN..HEAD + (pi + 1) * PAIR_LEN];
            let mut cost = pp[5] + pp[7];
            for k in 0..n {
                let l = &v[o2 + 3 + k * LEG_LEN..o2 + 3 + (k + 1) * LEG_LEN];
                cost += if l[0] == 1.0 { l[6] + l[7] + l[9] } else { l[6] };
            }
            let legs: Vec<String> = (0..n)
                .map(|k| {
                    let l = &v[o2 + 3 + k * LEG_LEN..o2 + 3 + (k + 1) * LEG_LEN];
                    if l[0] == 1.0 {
                        format!("{} {}-{}", lines[rn_line[l[1] as usize] as usize], dname[rn_st[l[1] as usize] as usize], dname[l[4] as usize])
                    } else {
                        format!("walk to {}", dname[l[4] as usize])
                    }
                })
                .collect();
            let rides = (0..n).filter(|&k| v[o2 + 3 + k * LEG_LEN] == 1.0).count();
            by_rides[rides.min(3)] += share;
            routes.push(json!({"pair": pi, "share": share, "pair_share": shares.get(pi).copied().unwrap_or(0.0), "cost": cost, "legs": legs.join("; ")}));
            o2 += 3 + n * LEG_LEN;
        }
        let fastest = routes.iter().min_by(|a, b| a["cost"].as_f64().unwrap().partial_cmp(&b["cost"].as_f64().unwrap()).unwrap()).cloned();
        let r = json!({"trip": what, "fastest_share": fastest.as_ref().map_or(0.0, |f| f["share"].as_f64().unwrap()), "fastest_cost": fastest.as_ref().map_or(0.0, |f| f["cost"].as_f64().unwrap()), "walk": v[17], "rail": v[18], "rail_cost": v[14], "drive_cost": v[12], "pairs": np, "pairs_80": n80,
            "busiest_pair": shares.first().copied().unwrap_or(0.0), "main_route": routes.first().map_or(0.0, |r| r["share"].as_f64().unwrap()),
            "listed": routes.iter().map(|r| r["share"].as_f64().unwrap()).sum::<f64>(), "unlisted": v[31], "top_pairs": pairs, "egress": egress, "by_rides": by_rides, "routes": routes});
        eprintln!("{what}: rail {:.1}%, pairs {np} (80% in {n80}), busiest {:.1}%, main route {:.1}%, fastest listed {:.1}% at {:.1}, listed {:.1}%", 100.0 * v[18], 100.0 * r["busiest_pair"].as_f64().unwrap(), 100.0 * r["main_route"].as_f64().unwrap(), 100.0 * r["fastest_share"].as_f64().unwrap(), r["fastest_cost"].as_f64().unwrap(), 100.0 * r["listed"].as_f64().unwrap());
        out.push(r);
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&json!({"city": city, "set": sets, "range": range, "trips": out})).unwrap()).unwrap();
}
