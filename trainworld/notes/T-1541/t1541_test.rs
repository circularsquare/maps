//! T-1541: why corridor rail is low where an inter-city line runs alone through a big city. T-1521
//! round 3's corridor run (the regional layer alone, every zone of every place, every period, three
//! crowding rounds), then, in the midday period, every zone pair of each focus corridor rebuilt as
//! `demand` chooses it, with rail's perceived minutes split into access, egress, wait and ride, and
//! fare, against the car and the plane. Zones are grouped by their distance from the focus line's
//! station in their place.
//!
//!   T1541_CASE=china T1521_ROOT=<trainworld> T1541_OUT=out.json TW_ROWS=rows.json T1521_FARE=... \
//!     cargo test --release --features demand t1541_access -- --ignored --nocapture
//!
//! Knobs (the market's own `Params`, set before the network): T1541_RIDE_MAX (m), T1541_RIDE_F,
//! T1541_P_RIDE, T1541_D_DROP, T1541_RIDE_LINES. In the copy only: T1541_AIRACC, T1521_AIR.

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

/// A stand-in for the metro a player builds: `n` radial lines from the inter-city station, a stop
/// every `step` km to `len` km at `kmh` with half a minute's dwell, 12/8/6 trains an hour, 8 cars.
/// The first stop is the station itself, so the lines share it (merged within 100 m).
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
    let nec_f = vec![("nyc", "wdc", NYP, WAS, 180.), ("nyc", "bos", NYP, BOS, 225.), ("nyc", "phl", NYP, PHL, 75.)];
    let eu_f = vec![("lon", "cit", KGX, EDB, 260.), ("par", "lyo", PGL, LYO, 117.), ("mad", "bar", MAD, BCN, 155.)];
    let cn_f = vec![("sha", "bei", SHH, BJS, 270.), ("bei", "gua", BJW, GZS, 450.)];
    match name {
        // New York's real network plus a Northeast Regional and a New York - Boston line (T-1521)
        "nec" => (vec!["nyc"], nec_lines(), nec_f),
        // the same, with a stand-in metro at Washington, Philadelphia and Boston
        "nec_feed" => (vec!["nyc"], [nec_lines(), feed(&[WAS, PHL, BOS])].concat(), nec_f),
        // lone lines (T-1521)
        "eu" => (vec![], eu_lines(), eu_f),
        // London's real network (T-712's lon_real) plus the lines
        "eu_lon" => (vec!["lon"], eu_lines(), eu_f),
        // a stand-in metro at King's Cross and Edinburgh
        "eu_feed" => (vec![], [eu_lines(), feed(&[KGX, EDB])].concat(), eu_f),
        // London's real network and a stand-in metro at Edinburgh
        "eu_lonfeed" => (vec!["lon"], [eu_lines(), feed(&[EDB])].concat(), eu_f),
        // lone lines (T-1521)
        "china" => (vec![], china_lines(), cn_f),
        // a stand-in metro at Beijing South, Beijing West, Hongqiao and Guangzhou South
        "china_feed" => (vec![], [china_lines(), feed(&[BJS, BJW, SHH, GZS])].concat(), cn_f),
        _ => panic!("unknown case {name}"),
    }
}

fn hav_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (lo0, la0, lo1, la1) = (a.0.to_radians(), a.1.to_radians(), b.0.to_radians(), b.1.to_radians());
    let h = ((la1 - la0) / 2.).sin().powi(2) + la0.cos() * la1.cos() * ((lo1 - lo0) / 2.).sin().powi(2);
    2. * 6371.0 * h.sqrt().asin()
}

const RINGS: [f64; 6] = [0., 5., 10., 15., 25., 40.];
fn ring(d: f64) -> usize {
    RINGS.iter().rposition(|&r| d >= r).unwrap()
}

fn env_f(k: &str) -> Option<f32> {
    std::env::var(k).ok().map(|v| v.parse().unwrap())
}

#[test]
#[ignore]
fn t1541_access() {
    let name = std::env::var("T1541_CASE").unwrap_or("china".into());
    let root = std::env::var("T1521_ROOT").expect("T1521_ROOT");
    let out_path = std::env::var("T1541_OUT").expect("T1541_OUT");
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
    let mut a = IntercityApi::new(&real_world(&root, None)).unwrap();
    if let Some(v) = env_f("T1541_RIDE_MAX") {
        a.base.ride_max_m = v;
    }
    if let Some(v) = env_f("T1541_RIDE_F") {
        a.base.ride_drive_f = v;
    }
    if let Some(v) = env_f("T1541_P_RIDE") {
        a.base.p_ride = v;
    }
    if let Some(v) = env_f("T1541_D_DROP") {
        a.base.d_drop = v;
    }
    if let Some(v) = env_f("T1541_RIDE_LINES") {
        a.base.ride_lines = v as usize;
    }
    a.set_network_ll(&ll, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fare_weight(fare[0]).unwrap();
    a.set_fares(fare[1], fare[2], &hop_km).unwrap();
    a.set_fare_curve(&fare[3..]).unwrap();
    let m = a.m.clone();
    let cors: Vec<Option<(usize, usize, usize)>> = focus.iter().map(|f| {
        let (pa, pb) = (m.places.iter().position(|p| p.zone == f.0), m.places.iter().position(|p| p.zone == f.1));
        match (pa, pb) {
            (Some(x), Some(y)) if x != y => Some((x, y, m.corridor(x, y))),
            _ => None,
        }
    }).collect();
    let nf = focus.len();
    let (mut ct, mut cr, mut cr1, mut ct1) = (vec![0f64; nf], vec![0f64; nf], vec![0f64; nf], vec![0f64; nf]);
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
                if q == QD {
                    ct1[i] = cs[4 * c + 2];
                    cr1[i] = cs[4 * c + 3];
                }
            }
        }
    }
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
            let Some((pa, pb, _)) = *c else {
                out_pairs.push(serde_json::json!({"pair": [f.0, f.1], "missing": true}));
                continue;
            };
            let (sta, stb) = (st_of(f.2), st_of(f.3));
            let zones_of = |pl: usize, st: (f64, f64)| -> Vec<(usize, f64)> {
                m.zones.iter().enumerate().filter(|(_, z)| z.place == pl).map(|(i, z)| (i, hav_km(lonlat_of(&z.u), st))).collect()
            };
            let (za_all, zb_all) = (zones_of(pa, f.2), zones_of(pb, f.3));
            let (from, to) = (&m.places[pa], &m.places[pb]);
            // T1541_ZAIR "mult,add": the plane's drive to each airport from the zone itself (the
            // pack's airports, in the place's order), counted mult x minutes + add per end, in
            // place of the place's people-weighted mean
            let zair: Option<(f64, f64)> = std::env::var("T1541_ZAIR").ok().map(|v| {
                let (x, y) = v.split_once(',').unwrap();
                (x.parse().unwrap(), y.parse().unwrap())
            });
            let mut zdrive: HashMap<usize, Vec<f64>> = HashMap::new();
            if zair.is_some() {
                for (pl, zs) in [(pa, &za_all), (pb, &zb_all)] {
                    let pk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/packs/{}.market.json", m.places[pl].zone)).unwrap()).unwrap();
                    let aps: Vec<[f64; 3]> = pk["airports"].as_array().unwrap().iter().map(|a| unit(a[0].as_f64().unwrap(), a[1].as_f64().unwrap())).collect();
                    assert_eq!(aps.len(), m.places[pl].air.len());
                    for &(z, _) in zs.iter() {
                        zdrive.insert(z, aps.iter().map(|u| drive_min(chord_km(&m.zones[z].u, u) as f32) as f64).collect());
                    }
                }
            }
            let zone_fl = |za: usize, zb: usize, rule: &PairRule| -> Cost {
                let Some((mul, add)) = zair.filter(|_| rule.fl.is_some()) else { return rule.fl };
                let (zpa, zpb) = (m.zones[za].place, m.zones[zb].place);
                let (p, q) = (&m.places[zpa], &m.places[zpb]);
                let (Some(da), Some(db)) = (zdrive.get(&za), zdrive.get(&zb)) else { return rule.fl };
                let mut best = Cost::NONE;
                for (k, (_, xd)) in p.air.iter().enumerate() {
                    for (l, (_, yd)) in q.air.iter().enumerate() {
                        let direct = xd.contains(&(zpb as u32)) || yd.contains(&(zpa as u32));
                        let c = air_cost(rule.km, mul * (da[k] + db[l]) + 2. * add, direct);
                        if c.min < best.min {
                            best = c;
                        }
                    }
                }
                best
            };
            // one zone pair, one way, as `demand` chooses it: per income group (share, factor) its
            // trips, rail, air, car, and for the middle group the parts
            let eval = |za: usize, zb: usize, terms: &mut Vec<(u32, u32, f64)>| -> (f64, f64, f64, f64, Option<serde_json::Value>) {
                let (zpa, zpb) = (m.zones[za].place, m.zones[zb].place);
                let (fp, tp) = (&m.places[zpa], &m.places[zpb]);
                let mut rule = m.pr(zpa, zpb);
                rule.fl = zone_fl(za, zb, &rule);
                let split = rule.split_car(fp, tp);
                let all = m.eachway(&m.zones[za], &m.zones[zb]) * SHARES[QD] as f64;
                let pi = a.pair_at(za, zb);
                let car_c = if pi != u32::MAX { a.pairs[pi as usize].car } else { rule.car_cost(m.zone_km(&m.zones[za], &m.zones[zb]) as f32, m.zones[zb].park) };
                let band = if pi != u32::MAX { a.pairs[pi as usize].band } else { 4 };
                let (mut t, mut r, mut ai, mut ca) = (0., 0., 0., 0.);
                let mut detail = None;
                let ng = fp.groups().len();
                for (k, &(share, fgf)) in fp.groups().iter().enumerate() {
                    let trips = all * share as f64;
                    let w = fp.w() * fgf;
                    let alt = rule.alt(car_c, split, w, &m.mp);
                    let (rail_c, sum) = if pi != u32::MAX {
                        rail_logsum_x(&re, s, p.fare_min, &a.access[za], &a.access_work[zb], p.beta, p.soften(p.band_unsoftened(band as usize), alt.car), a.fx_of(&fx, za, fgf), terms)
                    } else {
                        (f32::INFINITY, 0.)
                    };
                    let bt = p.mode_beta(alt.car);
                    let v_rail = if sum > 0. { ((-bt * (rail_c - alt.drive)).clamp(-80., 80.) as f64).exp() } else { 0. };
                    let frac = v_rail / (alt.rest as f64 + v_rail);
                    t += trips;
                    r += trips * frac;
                    ai += trips * (1. - frac) * alt.air as f64;
                    ca += trips * (1. - frac) * (1. - alt.air as f64);
                    if k == ng / 2 {
                        let car_t = car_c.at(w);
                        let fl_t = rule.fl.at(w);
                        let mut d = serde_json::json!({
                            "w": w, "served": pi != u32::MAX, "route": sum > 0.,
                            "car": {"min": car_c.min, "money": car_c.usd * w, "total": car_t},
                            "air": {"min": rule.fl.min, "money": rule.fl.usd * w, "total": fl_t, "share_of_nonrail": alt.air},
                            "group_shares": [frac, (1. - frac) * alt.air as f64, (1. - frac) * (1. - alt.air as f64)],
                            "mode_beta": bt,
                        });
                        if sum > 0. {
                            let top = terms.iter().cloned().fold((0u32, 0u32, -1f64), |b, t| if t.2 > b.2 { t } else { b });
                            let (sa, sb) = (top.0 as usize, top.1 as usize);
                            let wa = a.access[za].iter().find(|e| e.0 == sa).unwrap().1 as f64;
                            let wb = a.access_work[zb].iter().find(|e| e.0 == sb).unwrap().1 as f64;
                            let dd = (1. + a.zone_fx.get(za).copied().unwrap_or(0.)) * fgf;
                            let fare_one = fx.as_ref().map_or(p.fare_min, |x| x.fare(sa, sb, s));
                            let fare_tot = fare_one * dd;
                            let skim_fare = fare_one - p.fare_min;
                            let s2s = re[sa * s + sb] - skim_fare;
                            let acc = -(wa.ln()) / beta;
                            let egr = -(wb.ln()) / beta;
                            d["rail"] = serde_json::json!({
                                "logsum": rail_c, "best_share": top.2 / sum, "access": acc, "egress": egr,
                                "wait_ride": s2s, "fare": fare_tot, "best_total": acc + egr + s2s as f64 + fare_tot as f64,
                                "best_is_focus": sa == sta && sb == stb, "n_pairs": terms.len(),
                                "access_list": a.access[za].iter().map(|e| (e.0, -((e.1 as f64).ln()) / beta)).collect::<Vec<_>>(),
                                "egress_list": a.access_work[zb].iter().map(|e| (e.0, -((e.1 as f64).ln()) / beta)).collect::<Vec<_>>(),
                            });
                        }
                        detail = Some(d);
                    }
                }
                (t, r, ai, ca, detail)
            };
            // rings: trips and rail by the two ends' distance from the focus stations
            let nr = RINGS.len();
            let mut grid = vec![[0f64; 4]; nr * nr];
            let mut served_t = 0f64;
            let (mut tt, mut rr, mut aa, mut cc) = (0., 0., 0., 0.);
            for &(za, da) in &za_all {
                for &(zb, db) in &zb_all {
                    for (x, y) in [(za, zb), (zb, za)] {
                        let (t, r, ai, ca, _) = eval(x, y, &mut terms);
                        let gi = ring(da) * nr + ring(db);
                        grid[gi][0] += t;
                        grid[gi][1] += r;
                        grid[gi][2] += ai;
                        grid[gi][3] += ca;
                        tt += t;
                        rr += r;
                        aa += ai;
                        cc += ca;
                        if a.pair_at(x, y) != u32::MAX {
                            served_t += t;
                        }
                    }
                }
            }
            // samples: the station's zone, then the most populous zone near each ring's middle
            let pick = |zs: &[(usize, f64)]| -> Vec<(usize, f64)> {
                let mut v = vec![*zs.iter().min_by(|x, y| x.1.total_cmp(&y.1)).unwrap()];
                for want in [7.5, 12.5, 20., 32.] {
                    if let Some(z) = zs.iter().filter(|z| (z.1 - want).abs() < 2.5).max_by(|x, y| m.zones[x.0].pop.total_cmp(&m.zones[y.0].pop)) {
                        v.push(*z);
                    }
                }
                v
            };
            let (sa_z, sb_z) = (pick(&za_all), pick(&zb_all));
            let mut samples = vec![];
            for (end, list, other) in [("A", &sa_z, sb_z[0]), ("B", &sb_z, sa_z[0])] {
                for &(z, d) in list.iter() {
                    let (x, y) = if end == "A" { (z, other.0) } else { (other.0, z) };
                    let (_, _, _, _, det) = eval(x, y, &mut terms);
                    samples.push(serde_json::json!({"end": end, "zone_km": d, "pop": m.zones[z].pop, "ll": lonlat_of(&m.zones[z].u), "zone_pair_km": m.zone_km(&m.zones[x], &m.zones[y]), "detail": det}));
                }
            }
            let rings: Vec<serde_json::Value> = grid.iter().enumerate().filter(|(_, v)| v[0] > 0.).map(|(i, v)| serde_json::json!({"ra": RINGS[i / nr], "rb": RINGS[i % nr], "trips": v[0], "rail": v[1], "air": v[2], "car": v[3]})).collect();
            let pop_ring = |zs: &[(usize, f64)]| -> Vec<f64> {
                let mut v = vec![0f64; nr];
                for &(z, d) in zs {
                    v[ring(d)] += m.zones[z].pop;
                }
                v
            };
            let c = c.unwrap().2;
            out_pairs.push(serde_json::json!({
                "pair": [f.0, f.1], "km": m.pr(pa, pb).km, "ride_min": f.4,
                "day": {"trips": ct[fi], "rail": cr[fi], "air_of_nonrail": modes[2 * c], "car_of_nonrail": modes[2 * c + 1]},
                "midday": {"cs_trips": ct1[fi], "cs_rail": cr1[fi], "trips": tt, "rail": rr, "air": aa, "car": cc, "served_trips": served_t},
                "pop_by_ring": {"A": pop_ring(&za_all), "B": pop_ring(&zb_all)},
                "places": {"A": {"w": from.w(), "groups": from.groups(), "pop": za_all.iter().map(|z| m.zones[z.0].pop).sum::<f64>()}, "B": {"w": to.w(), "pop": zb_all.iter().map(|z| m.zones[z.0].pop).sum::<f64>()}},
                "rings": rings, "samples": samples,
            }));
        }
    });
    let out = serde_json::json!({"case": name, "fare": fare, "knobs": {
        "ride_max_m": a.base.ride_max_m, "ride_drive_f": a.base.ride_drive_f, "p_ride": a.base.p_ride, "d_drop": a.base.d_drop, "ride_lines": a.base.ride_lines,
        "airacc": std::env::var("T1541_AIRACC").unwrap_or("1,0".into()), "zair": std::env::var("T1541_ZAIR").ok(), "air": std::env::var("T1521_AIR").unwrap_or("100,0.05".into())}, "pairs": out_pairs});
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    eprintln!("wrote {out_path}");
}
