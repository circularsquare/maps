//! T-1521 round 3: T-712's corridor test brought up to date. The regional layer alone on real
//! networks plus test lines, every zone of every place, every period, three crowding rounds on its
//! own loads; per focus corridor its longer trips, their rail trips, and the split of the rest
//! between plane and car. The rows carry what the app adds (`TW_ROWS`, written by
//! corridor_rows.mjs); the fares are the market's (`T1521_FARE`: India's weight, base, per km and
//! the curve's knots).
//!
//!   T1521_CASE=eu T1521_ROOT=<trainworld> T1521_OUT=out.json TW_ROWS=rows.json \
//!     cargo test --release --features demand t1521_corridors -- --ignored --nocapture

use super::tests::real_world;
use super::*;
use crate::demand::pack;

/// stations (lon, lat), minutes per hop, tph [three slots], cars
type Line = (Vec<(f64, f64)>, Vec<f32>, [f32; 3], f32);

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

type Case = (Vec<&'static str>, Vec<Line>, Vec<(&'static str, &'static str)>);

fn case(name: &str) -> Case {
    match name {
        // New York's real network plus a Northeast Regional (Washington 3 h) and New York - Boston 3 h 45
        "nec" => (vec!["nyc"], vec![
            (vec![NYP, NWK, TRE, PHL, WIL, BAL, WAS], vec![15., 32., 28., 20., 50., 35.], [3., 2., 1.], 8.),
            (vec![NYP, STM, NHV, PVD, BOS], vec![50., 40., 95., 40.], [2., 1., 1.], 8.),
        ], vec![("nyc", "wdc"), ("nyc", "bos"), ("nyc", "phl")]),
        // Japan's real network (T-737) plus a Nozomi: Tokyo - Shin-Osaka 2 h 23, Hakata 4 h 45
        "jpn" => (vec!["jpn"], vec![
            (vec![(139.7671, 35.6812), (139.7387, 35.6285), (139.6173, 35.5079), (136.8815, 35.1709), (135.7588, 34.9858), (135.5003, 34.7334),
                  (133.9180, 34.6660), (132.4757, 34.3976), (130.8823, 33.8869), (130.4207, 33.5897)],
             vec![7., 11., 77., 34., 14., 45., 35., 45., 17.], [8., 6., 3.], 16.),
        ], vec![("tyo", "osa"), ("tyo", "fuk"), ("tyo", "ngo"), ("osa", "fuk")]),
        // lone lines, no city networks: Paris - Lyon - Marseille, Madrid - Barcelona, London - Edinburgh, Berlin - Hamburg
        "eu" => (vec![], vec![
            (vec![PGL, LYO, MRS], vec![117., 68.], [4., 3., 1.], 16.),
            (vec![MAD, ZAZ, BCN], vec![80., 75.], [3., 2., 1.], 16.),
            (vec![KGX, YRK, NCL, EDB], vec![110., 55., 95.], [2., 2., 1.], 16.),
            (vec![BER, HAM], vec![105.], [2., 1., 1.], 16.),
        ], vec![("par", "lyo"), ("par", "mar"), ("mad", "bar"), ("lon", "cit"), ("ber", "ham")]),
        // lone lines: Beijing South - Shanghai Hongqiao 4 h 30, Beijing West - Guangzhou South 7 h 30
        "china" => (vec![], vec![
            (vec![(116.3790, 39.8650), (117.2100, 39.1360), (116.8830, 36.6700), (118.7980, 31.9690), (121.3200, 31.1940)], vec![30., 60., 110., 70.], [4., 3., 1.], 16.),
            (vec![(116.3210, 39.8950), (114.4800, 38.0100), (113.7720, 34.7590), (114.4240, 30.6060), (113.0640, 28.1470), (113.2690, 22.9880)], vec![70., 100., 95., 75., 110.], [3., 2., 1.], 16.),
        ], vec![("sha", "bei"), ("bei", "gua")]),
        _ => panic!("unknown case {name}"),
    }
}

#[test]
#[ignore]
fn t1521_corridors() {
    let name = std::env::var("T1521_CASE").unwrap_or("eu".into());
    let root = std::env::var("T1521_ROOT").expect("T1521_ROOT");
    let out_path = std::env::var("T1521_OUT").expect("T1521_OUT");
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
    let km = |a: usize, b: usize| {
        let (lo0, la0, lo1, la1) = (ll[2 * a].to_radians(), ll[2 * a + 1].to_radians(), ll[2 * b].to_radians(), ll[2 * b + 1].to_radians());
        let h = ((la1 - la0) / 2.).sin().powi(2) + la0.cos() * la1.cos() * ((lo1 - lo0) / 2.).sin().powi(2);
        (2. * 6371.0 * h.sqrt().asin()) as f32
    };
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
    let t = now_ms();
    let mut a = IntercityApi::new(&real_world(&root, None)).unwrap();
    let open_ms = now_ms() - t;
    a.set_network_ll(&ll, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fare_weight(fare[0]).unwrap();
    a.set_fares(fare[1], fare[2], &hop_km).unwrap();
    a.set_fare_curve(&fare[3..]).unwrap();
    let cors: Vec<Option<(usize, usize, usize)>> = focus.iter().map(|f| {
        let (pa, pb) = (a.m.places.iter().position(|p| p.zone == f.0), a.m.places.iter().position(|p| p.zone == f.1));
        match (pa, pb) {
            (Some(x), Some(y)) if x != y => Some((x, y, a.m.corridor(x, y))),
            _ => None,
        }
    }).collect();
    let nf = focus.len();
    let (mut ct, mut cr, mut cc, mut ccr) = (vec![0f64; nf], vec![0f64; nf], vec![0f64; nf], vec![0f64; nf]);
    let t = now_ms();
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
        let ccs = a.commute_corridors(q);
        for (i, c) in cors.iter().enumerate() {
            if let Some((_, _, c)) = c {
                ct[i] += cs[4 * c + 2];
                cr[i] += cs[4 * c + 3];
                if !ccs.is_empty() {
                    cc[i] += ccs[2 * c];
                    ccr[i] += ccs[2 * c + 1];
                }
            }
        }
    }
    let solve_ms = now_ms() - t;
    let modes = a.corridor_modes();
    let pairs: Vec<serde_json::Value> = focus.iter().zip(&cors).enumerate().map(|(i, (f, c))| match c {
        None => serde_json::json!({"pair": [f.0, f.1], "missing": true}),
        Some((x, y, c)) => {
            let r = a.m.pr(*x, *y);
            let (air, car) = (modes[2 * c] as f64, modes[2 * c + 1] as f64);
            let rail = if ct[i] > 0. { cr[i] / ct[i] } else { 0. };
            serde_json::json!({"pair": [f.0, f.1], "km": r.km, "trips": ct[i], "rail": cr[i], "air_of_nonrail": air, "car_of_nonrail": car,
                "shares": [rail, (1. - rail) * air, (1. - rail) * car], "commute_legs": cc[i], "commute_rail": ccr[i]})
        }
    }).collect();
    let out = serde_json::json!({"case": name, "fare": fare, "pairs": pairs, "open_ms": open_ms, "solve_ms": solve_ms});
    eprintln!("{}", serde_json::to_string_pretty(&out).unwrap());
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
