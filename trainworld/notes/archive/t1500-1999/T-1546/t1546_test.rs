//! T-1546: rail against the plane at 4-5 hours. T-1541's corridor run (the regional layer alone,
//! every zone of every place, every period, three crowding rounds, the built fares), then for the
//! midday period every zone pair of each focus corridor as `demand` chooses it, dumped with what the
//! choice needs (per income group: trips, the rider's weight, rail's logsum; the car; the place
//! pair's plane and split car; the middle group's rail parts; each zone's drive to its place's
//! airports), so `t1546_eval.py` can rebuild the shares under other rules for the plane without a
//! new solve.
//!
//!   T1546_CASE=eu T1521_ROOT=<trainworld> T1546_OUT=out.json TW_ROWS=rows.json T1521_FARE=... \
//!     cargo test --release --features demand t1546_dump -- --ignored --nocapture

use super::tests::real_world;
use super::*;
use crate::demand::pack;

/// stations (lon, lat), minutes per hop, tph [three slots], cars
type Line = (Vec<(f64, f64)>, Vec<f32>, [f32; 3], f32);
/// place A, place B, A's station, B's station, the line's minutes between them
type Focus = (&'static str, &'static str, (f64, f64), (f64, f64), f32);

const NYP: (f64, f64) = (-73.9935, 40.7506);
const NWK: (f64, f64) = (-74.1645, 40.7346);
const TRE: (f64, f64) = (-74.7557, 40.2178);
const PHL: (f64, f64) = (-75.1817, 39.9557);
const WIL: (f64, f64) = (-75.5513, 39.7366);
const BAL: (f64, f64) = (-76.6158, 39.3074);
const WAS: (f64, f64) = (-77.0063, 38.8973);
const STM: (f64, f64) = (-73.5418, 41.0465);
const NHV: (f64, f64) = (-72.9267, 41.2977);
const PVD: (f64, f64) = (-71.4134, 41.8294);
const BOS: (f64, f64) = (-71.0553, 42.3523);
const PGL: (f64, f64) = (2.3730, 48.8443);
const LYO: (f64, f64) = (4.8594, 45.7606);
const MRS: (f64, f64) = (5.3806, 43.3027);
const MAD: (f64, f64) = (-3.6908, 40.4065);
const ZAZ: (f64, f64) = (-0.9112, 41.6587);
const BCN: (f64, f64) = (2.1404, 41.3790);
const KGX: (f64, f64) = (-0.1233, 51.5320);
const YRK: (f64, f64) = (-1.0932, 53.9580);
const NCL: (f64, f64) = (-1.6174, 54.9683);
const EDB: (f64, f64) = (-3.1890, 55.9521);
const BER: (f64, f64) = (13.3695, 52.5251);
const HAM: (f64, f64) = (10.0067, 53.5530);
const BJS: (f64, f64) = (116.3790, 39.8650);
const BJW: (f64, f64) = (116.3210, 39.8950);
const SHH: (f64, f64) = (121.3200, 31.1940);
const GZS: (f64, f64) = (113.2690, 22.9880);
const TYO: (f64, f64) = (139.7671, 35.6812);
const NGO: (f64, f64) = (136.8815, 35.1709);
const OSA: (f64, f64) = (135.5003, 34.7334);
const HKT: (f64, f64) = (130.4207, 33.5897);

fn nec_lines() -> Vec<Line> {
    vec![
        (vec![NYP, NWK, TRE, PHL, WIL, BAL, WAS], vec![15., 32., 28., 20., 50., 35.], [3., 2., 1.], 8.),
        (vec![NYP, STM, NHV, PVD, BOS], vec![50., 40., 95., 40.], [2., 1., 1.], 8.),
    ]
}
fn eu_lines() -> Vec<Line> {
    vec![
        (vec![PGL, LYO, MRS], vec![117., 68.], [4., 3., 1.], 16.),
        (vec![MAD, ZAZ, BCN], vec![80., 75.], [3., 2., 1.], 16.),
        (vec![KGX, YRK, NCL, EDB], vec![110., 55., 95.], [2., 2., 1.], 16.),
        (vec![BER, HAM], vec![105.], [2., 1., 1.], 16.),
    ]
}
fn china_lines() -> Vec<Line> {
    vec![
        (vec![BJS, (117.2100, 39.1360), (116.8830, 36.6700), (118.7980, 31.9690), SHH], vec![30., 60., 110., 70.], [4., 3., 1.], 16.),
        (vec![BJW, (114.4800, 38.0100), (113.7720, 34.7590), (114.4240, 30.6060), (113.0640, 28.1470), GZS], vec![70., 100., 95., 75., 110.], [3., 2., 1.], 16.),
    ]
}
/// T-1521's Nozomi: Tokyo - Shin-Osaka 2 h 23, Hakata 4 h 45
fn jpn_lines() -> Vec<Line> {
    vec![(
        vec![TYO, (139.7387, 35.6285), (139.6173, 35.5079), NGO, (135.7588, 34.9858), OSA, (133.9180, 34.6660), (132.4757, 34.3976), (130.8823, 33.8869), HKT],
        vec![7., 11., 77., 34., 14., 45., 35., 45., 17.],
        [8., 6., 3.],
        16.,
    )]
}

/// T-1541's stand-in for the metro a player builds: `n` radial lines from the inter-city station, a
/// stop every `step` km to `len` km at `kmh`, 12/8/6 trains an hour, 8 cars.
fn feeders(c: (f64, f64), n: usize, len: f64, step: f64, kmh: f64) -> Vec<Line> {
    let k = (len / step).round() as usize;
    (0..n)
        .map(|i| {
            let th = (i as f64 + 0.5) / n as f64 * std::f64::consts::TAU;
            let pts = (0..=k)
                .map(|j| {
                    let d = j as f64 * step;
                    (c.0 + d * th.cos() / (111.32 * c.1.to_radians().cos()), c.1 + d * th.sin() / 110.57)
                })
                .collect();
            (pts, vec![(step / kmh * 60. + 0.5) as f32; k], [12., 8., 6.], 8.)
        })
        .collect()
}

fn case(name: &str) -> (Vec<&'static str>, Vec<Line>, Vec<Focus>) {
    let feed = |cs: &[(f64, f64)]| cs.iter().flat_map(|&c| feeders(c, 12, 30., 1.5, 35.)).collect::<Vec<_>>();
    let nec_f = vec![("nyc", "wdc", NYP, WAS, 180.), ("nyc", "bos", NYP, BOS, 225.)];
    let eu_f = vec![("lon", "cit", KGX, EDB, 260.), ("par", "mar", PGL, MRS, 185.), ("par", "lyo", PGL, LYO, 117.), ("mad", "bar", MAD, BCN, 155.)];
    let cn_f = vec![("sha", "bei", SHH, BJS, 270.), ("bei", "gua", BJW, GZS, 450.)];
    let jp_f = vec![("tyo", "fuk", TYO, HKT, 285.), ("tyo", "osa", TYO, OSA, 143.), ("osa", "fuk", OSA, HKT, 142.), ("tyo", "ngo", TYO, NGO, 95.)];
    match name {
        // T-1521 round 3's cases
        "nec" => (vec!["nyc"], nec_lines(), nec_f),
        "eu" => (vec![], eu_lines(), eu_f),
        "china" => (vec![], china_lines(), cn_f),
        "jpn" => (vec!["jpn"], jpn_lines(), jp_f),
        // Europe as Japan is run: London's and Paris's real networks, a stand-in metro at the other
        // big stations (Edinburgh, Lyon, Marseille, Madrid, Barcelona)
        "eun" => (vec!["lon", "par"], [eu_lines(), feed(&[EDB, LYO, MRS, MAD, BCN])].concat(), eu_f),
        // China and the Northeast with stand-in metros where no real network is
        "china_feed" => (vec![], [china_lines(), feed(&[BJS, BJW, SHH, GZS])].concat(), cn_f),
        "nec_feed" => (vec!["nyc"], [nec_lines(), feed(&[WAS, PHL, BOS])].concat(), nec_f),
        _ => panic!("unknown case {name}"),
    }
}

fn hav_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (lo0, la0, lo1, la1) = (a.0.to_radians(), a.1.to_radians(), b.0.to_radians(), b.1.to_radians());
    let h = ((la1 - la0) / 2.).sin().powi(2) + la0.cos() * la1.cos() * ((lo1 - lo0) / 2.).sin().powi(2);
    2. * 6371.0 * h.sqrt().asin()
}

#[test]
#[ignore]
fn t1546_dump() {
    let name = std::env::var("T1546_CASE").unwrap_or("eu".into());
    let root = std::env::var("T1521_ROOT").expect("T1521_ROOT");
    let out_path = std::env::var("T1546_OUT").expect("T1546_OUT");
    let fare: Vec<f32> = std::env::var("T1521_FARE").expect("T1521_FARE").split(',').map(|v| v.parse().unwrap()).collect();
    let (bases, extra, focus) = case(&name);
    let (mut ll, mut line_n, mut stops, mut times, mut tph, mut cars) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for b in &bases {
        let net: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/{b}_real.json")).unwrap()).unwrap();
        let (h, _) = pack::load_raw(&format!("{root}/data/packs/{}.json", net["pack"].as_str().unwrap())).unwrap();
        let hv: serde_json::Value = serde_json::from_str(&h).unwrap();
        let frame = Frame::equirect(hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
        let d = &net["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let s0 = (ll.len() / 2) as u32;
        for p in f32s("st_xy").chunks_exact(2) {
            let (lon, lat) = frame.to_lonlat(p[0] as f64, p[1] as f64);
            ll.extend([lon, lat]);
        }
        line_n.extend(u32s("line_n"));
        stops.extend(u32s("stops").into_iter().map(|s| s + s0));
        times.extend(f32s("times"));
        tph.extend(f32s("tph"));
        cars.extend(f32s("cars"));
    }
    for (pts, mins, f, c) in &extra {
        let s0 = (ll.len() / 2) as u32;
        let n = pts.len();
        for p in pts {
            ll.extend([p.0, p.1]);
        }
        line_n.push(n as u32);
        stops.extend((0..n as u32).map(|k| s0 + k));
        for run in 0..2 {
            for _ in 0..3 {
                for k in 0..n {
                    let s = if run == 0 { if k + 1 < n { mins[k] * 60. } else { 0. } } else if k > 0 { mins[k - 1] * 60. } else { 0. };
                    times.push(s);
                }
            }
        }
        tph.extend(f);
        cars.push(*c);
    }
    let km = |a: usize, b: usize| hav_km((ll[2 * a], ll[2 * a + 1]), (ll[2 * b], ll[2 * b + 1])) as f32;
    let mut hop_km = vec![];
    let mut off = 0usize;
    for &n in &line_n {
        let n = n as usize;
        for dir in 0..2 {
            for i in 0..n {
                let k = if dir == 0 { off + i } else { off + n - 1 - i };
                hop_km.push(if i + 1 < n {
                    let nx = if dir == 0 { k + 1 } else { k - 1 };
                    km(stops[k] as usize, stops[nx] as usize)
                } else {
                    0.
                });
            }
        }
        off += n;
    }
    let t0 = now_ms();
    let mut a = IntercityApi::new(&real_world(&root, None)).unwrap();
    a.set_network_ll(&ll, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fare_weight(fare[0]).unwrap();
    a.set_fares(fare[1], fare[2], &hop_km).unwrap();
    a.set_fare_curve(&fare[3..]).unwrap();
    let m = a.m.clone();
    let cors: Vec<Option<(usize, usize, usize)>> = focus
        .iter()
        .map(|f| {
            let (pa, pb) = (m.places.iter().position(|p| p.zone == f.0), m.places.iter().position(|p| p.zone == f.1));
            match (pa, pb) {
                (Some(x), Some(y)) if x != y => Some((x, y, m.corridor(x, y))),
                _ => None,
            }
        })
        .collect();
    let nf = focus.len();
    let (mut ct, mut cr) = (vec![0f64; nf], vec![0f64; nf]);
    const QD: usize = 1;
    for q in 0..PERIODS {
        assert!(a.solve(q));
        for _ in 0..3 {
            let nr = a.seg(q).len();
            let mut total = vec![0f32; 3 * nr];
            for (at, v) in [(0, a.seg(q)), (1, a.board(q)), (2, a.alight(q))] {
                for (x, y) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                    *x += y;
                }
            }
            assert!(a.crowd_with_external(q, &total));
        }
        let cs = a.corridor_summary(q);
        for (i, c) in cors.iter().enumerate() {
            if let Some((_, _, c)) = c {
                ct[i] += cs[4 * c + 2];
                cr[i] += cs[4 * c + 3];
            }
        }
    }
    let solve_ms = now_ms() - t0;
    let modes = a.corridor_modes();
    let st_map = a.svc.as_ref().unwrap().st_map.clone();
    let st_of = |p: (f64, f64)| -> usize {
        let k = (0..ll.len() / 2).find(|&k| (ll[2 * k] - p.0).abs() < 1e-9 && (ll[2 * k + 1] - p.1).abs() < 1e-9).unwrap();
        st_map[k] as usize
    };
    let beta = a.base.beta as f64;
    let per = a.per[QD].as_ref().unwrap();
    let (g, p) = (&per.g, &per.p);
    let s = g.n_st;
    let mut out_pairs = vec![];
    with_skims(g, p, &a.want, |sk| {
        let re = sk.entry_r();
        let kmv = a.pair_km(g, &a.want, |dst| (sk.slot[dst] != u32::MAX).then(|| sk.strategy(sk.slot[dst] as usize)));
        let fx = a.fare_x(g, p, &kmv);
        let mut terms: Vec<(u32, u32, f64)> = vec![];
        for (fi, (f, c)) in focus.iter().zip(&cors).enumerate() {
            let Some((pa, pb, ci)) = *c else {
                out_pairs.push(serde_json::json!({"pair": [f.0, f.1], "missing": true}));
                continue;
            };
            let (sta, stb) = (st_of(f.2), st_of(f.3));
            // each zone of the two places: place, people, km to its focus station, lon/lat, and the
            // drive (`drive_min`, plain minutes) to each of its place's airports in the place's order
            let mut zones = serde_json::Map::new();
            for (pl, st) in [(pa, f.2), (pb, f.3)] {
                let pk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/packs/{}.market.json", m.places[pl].zone)).unwrap()).unwrap();
                let aps: Vec<[f64; 3]> = pk["airports"].as_array().unwrap().iter().map(|a| unit(a[0].as_f64().unwrap(), a[1].as_f64().unwrap())).collect();
                assert_eq!(aps.len(), m.places[pl].air.len());
                for (zi, z) in m.zones.iter().enumerate().filter(|(_, z)| z.place == pl) {
                    let ll_z = lonlat_of(&z.u);
                    let dr: Vec<f64> = aps.iter().map(|u| drive_min(chord_km(&z.u, u) as f32) as f64).collect();
                    let akm: Vec<f64> = aps.iter().map(|u| chord_km(&z.u, u)).collect();
                    zones.insert(zi.to_string(), serde_json::json!({"place": pl, "pop": z.pop, "st_km": hav_km(ll_z, st), "ll": ll_z, "air_min": dr, "air_km": akm}));
                }
            }
            let place_j = |pl: usize, other: usize| {
                let pk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/packs/{}.market.json", m.places[pl].zone)).unwrap()).unwrap();
                let pp = &m.places[pl];
                serde_json::json!({"zone": pp.zone, "w": pp.w(), "groups": pp.groups(), "pop": pp.pop,
                    "air": pp.air.iter().map(|(x, d)| serde_json::json!([x, d.contains(&(other as u32))])).collect::<Vec<_>>(),
                    "airports": pk["airports"]})
            };
            let rule_j = |x: usize, y: usize| {
                let r = m.pr(x, y);
                let sp = r.split_car(&m.places[x], &m.places[y]);
                serde_json::json!({"car": r.car, "km": r.km, "fl": [r.fl.min, r.fl.usd], "split": [sp.min, sp.usd], "air": r.air})
            };
            let mut rows = vec![];
            for (x_pl, y_pl) in [(pa, pb), (pb, pa)] {
                let xs: Vec<usize> = m.zones.iter().enumerate().filter(|(_, z)| z.place == x_pl).map(|(i, _)| i).collect();
                let ys: Vec<usize> = m.zones.iter().enumerate().filter(|(_, z)| z.place == y_pl).map(|(i, _)| i).collect();
                for &za in &xs {
                    for &zb in &ys {
                        let (fp, tp) = (&m.places[x_pl], &m.places[y_pl]);
                        let rule = m.pr(x_pl, y_pl);
                        let all = m.eachway(&m.zones[za], &m.zones[zb]) * SHARES[QD] as f64;
                        let pi = a.pair_at(za, zb);
                        let car_c = if pi != u32::MAX { a.pairs[pi as usize].car } else { rule.car_cost(m.zone_km(&m.zones[za], &m.zones[zb]) as f32, m.zones[zb].park) };
                        let band = if pi != u32::MAX { a.pairs[pi as usize].band } else { 4 };
                        let split = rule.split_car(fp, tp);
                        let ng = fp.groups().len();
                        let mut rail = vec![];
                        let mut parts = serde_json::Value::Null;
                        for (k, &(_, fgf)) in fp.groups().iter().enumerate() {
                            let w = fp.w() * fgf;
                            let alt = rule.alt(car_c, split, w, &m.mp);
                            let (rail_c, sum) = if pi != u32::MAX {
                                rail_logsum_x(&re, s, p.fare_min, &a.access[za], &a.access_work[zb], p.beta, p.soften(p.band_unsoftened(band as usize), alt.car), a.fx_of(&fx, za, fgf), &mut terms)
                            } else {
                                (f32::INFINITY, 0.)
                            };
                            rail.push(if sum > 0. { serde_json::json!(rail_c) } else { serde_json::Value::Null });
                            if k == ng / 2 && sum > 0. {
                                let top = terms.iter().cloned().fold((0u32, 0u32, -1f64), |b, t| if t.2 > b.2 { t } else { b });
                                let (sa, sb) = (top.0 as usize, top.1 as usize);
                                let wa = a.access[za].iter().find(|e| e.0 == sa).unwrap().1 as f64;
                                let wb = a.access_work[zb].iter().find(|e| e.0 == sb).unwrap().1 as f64;
                                let dd = (1. + a.zone_fx.get(za).copied().unwrap_or(0.)) * fgf;
                                let fare_one = fx.as_ref().map_or(p.fare_min, |x| x.fare(sa, sb, s));
                                let fare_tot = fare_one * dd;
                                let s2s = re[sa * s + sb] - (fare_one - p.fare_min);
                                let focus_pair = (sa == sta && sb == stb) || (sa == stb && sb == sta);
                                parts = serde_json::json!([-(wa.ln()) / beta, -(wb.ln()) / beta, s2s, fare_tot, focus_pair, top.2 / sum]);
                            }
                        }
                        rows.push(serde_json::json!([za, zb, all, car_c.min, car_c.usd, rail, parts]));
                    }
                }
            }
            out_pairs.push(serde_json::json!({
                "pair": [f.0, f.1], "pa": pa, "pb": pb, "ride_min": f.4,
                "day": {"trips": ct[fi], "rail": cr[fi], "air_of_nonrail": modes[2 * ci], "car_of_nonrail": modes[2 * ci + 1]},
                "places": {"A": place_j(pa, pb), "B": place_j(pb, pa)},
                "rules": {"AB": rule_j(pa, pb), "BA": rule_j(pb, pa)},
                "zones": zones, "rows": rows,
            }));
        }
    });
    let out = serde_json::json!({"case": name, "fare": fare, "solve_ms": solve_ms,
        "mp": {"beta": m.mp.beta, "damp_min": m.mp.damp_min}, "p": {"beta": p.beta, "damp_min": p.damp_min},
        "pairs": out_pairs});
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("wrote {out_path} in {:.0} s", solve_ms / 1e3);
}
