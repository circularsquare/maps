//! T-436 (measurement): crowding rounds started from the last solve's loads against cold solves
//! (free flow and three rounds), on a real network through edits and their undos.
//! `TW_CITY=nyc TW_K=1 cargo test --release --features demand --lib t436 -- --ignored --nocapture`
use super::*;
use crate::demand::everyday::EverydayApi;
use crate::demand::network::{route_nodes, RealNet};
use crate::demand::pack;
use std::collections::HashMap;

#[derive(Clone)]
struct Net {
    st_xy: Vec<f32>,
    line_n: Vec<u32>,
    stops: Vec<u32>,
    times: Vec<f32>,
    tph: Vec<f32>,
    cars: Vec<f32>,
    id: Vec<u32>,
}

impl Net {
    fn remove(&self, l: usize) -> Net {
        let off: usize = self.line_n[..l].iter().map(|&n| n as usize).sum();
        let n = self.line_n[l] as usize;
        let mut m = self.clone();
        m.stops.drain(off..off + n);
        m.times.drain(6 * off..6 * (off + n));
        m.tph.drain(3 * l..3 * l + 3);
        m.cars.remove(l);
        m.line_n.remove(l);
        m.id.remove(l);
        m
    }
    fn scale_tph(&self, l: usize, k: f32) -> Net {
        let mut m = self.clone();
        for v in &mut m.tph[3 * l..3 * l + 3] {
            *v *= k;
        }
        m
    }
    /// per route node (kernel order): its line's id and its place among the line's route nodes
    fn keys(&self) -> Vec<(u32, u32)> {
        let mut out = vec![];
        for (l, &n) in self.line_n.iter().enumerate() {
            for i in 0..2 * n {
                out.push((self.id[l], i));
            }
        }
        out
    }
    fn at(&self, id: u32) -> usize {
        self.id.iter().position(|&x| x == id).unwrap()
    }
}

struct Out {
    /// per period: segment loads, boardings, alightings per route node, both populations
    total: Vec<Vec<f32>>,
    board: HashMap<u32, f64>,
    pkm: HashMap<u32, f64>,
    rail: f64,
    ms: f64,
    passes: usize,
}

fn run(a: &mut DemandApi, e: &mut EverydayApi, net: &Net, from: Option<(&Out, &Net)>, k: usize) -> Out {
    a.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
    e.set_network(&net.st_xy, &net.line_n, &net.stops, &net.times, &net.tph, &net.cars).unwrap();
    let (_, rn_line, hop_km) = route_nodes(&net.st_xy, &net.line_n, &net.stops);
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    e.set_fares(2.9, 0.1, &hop_km).unwrap();
    let nr = rn_line.len();
    let mapped: Option<Vec<Vec<f32>>> = from.map(|(o, old)| {
        let ok = old.keys();
        let onr = ok.len();
        let idx: HashMap<(u32, u32), usize> = ok.into_iter().enumerate().map(|(j, key)| (key, j)).collect();
        let keys = net.keys();
        (0..PERIODS)
            .map(|q| {
                let mut m = vec![0f32; 3 * nr];
                for (i, key) in keys.iter().enumerate() {
                    if let Some(&j) = idx.get(key) {
                        for f in 0..3 {
                            m[f * nr + i] = o.total[q][f * onr + j];
                        }
                    }
                }
                m
            })
            .collect()
    });
    let rounds = if from.is_some() { k } else { 3 };
    let t0 = now_ms();
    let mut passes = 0;
    let sum = |a: &DemandApi, e: &EverydayApi, q: usize| -> Vec<f32> {
        let add = |x: Vec<f32>, y: Vec<f32>| -> Vec<f32> { x.iter().zip(y).map(|(p, q)| p + q).collect() };
        [add(a.seg(q), e.seg(q)), add(a.board(q), e.board(q)), add(a.alight(q), e.alight(q))].concat()
    };
    for q in 0..PERIODS {
        for round in 0..=rounds {
            if round == 0 {
                let at = mapped.as_ref().map(|m| &m[q][..]);
                assert!(a.solve_from(q, at));
                assert!(e.solve_from(q, at));
                passes += 1;
            } else {
                let total = sum(a, e, q);
                let before = a.period(q).unwrap().ms;
                let _ = before;
                assert!(a.crowd_with_external(q, &total));
                assert!(e.crowd_with_external(q, &total));
                // a skipped round (settled or uncrowded) takes well under a ms
                if a.period(q).unwrap().ms > 5.0 {
                    passes += 1;
                }
            }
            release_skims();
        }
    }
    let ms = now_ms() - t0;
    let (mut board, mut pkm, mut rail) = (HashMap::new(), HashMap::new(), 0.0);
    let mut total = vec![];
    for q in 0..PERIODS {
        rail += a.summary(q)[2] + e.summary(q)[2];
        let t = sum(a, e, q);
        for r in 0..nr {
            let id = net.id[rn_line[r]];
            *board.entry(id).or_insert(0.0) += t[nr + r] as f64;
            *pkm.entry(id).or_insert(0.0) += t[r] as f64 * hop_km[r] as f64;
        }
        total.push(t);
    }
    Out { total, board, pkm, rail, ms, passes }
}

/// `x` against the reference `y`: lines with 1,000+ boardings a day in `y`
fn cmp(what: &str, x: &Out, y: &Out, names: &[String]) {
    let mut s = String::new();
    for (label, xm, ym) in [("boardings", &x.board, &y.board), ("passenger km", &x.pkm, &y.pkm)] {
        let (mut n, mut o1, mut o2, mut o5, mut worst, mut wid) = (0, 0, 0, 0, 0f64, 0u32);
        for (&id, &yv) in ym.iter() {
            if y.board.get(&id).copied().unwrap_or(0.0) < 1000.0 {
                continue;
            }
            let Some(&xv) = xm.get(&id) else { continue };
            let d = 100.0 * (xv / yv - 1.0);
            n += 1;
            if d.abs() > 1.0 { o1 += 1; }
            if d.abs() > 2.0 { o2 += 1; }
            if d.abs() > 5.0 { o5 += 1; }
            if d.abs() > worst.abs() { worst = d; wid = id; }
        }
        s += &format!("  {label}: of {n} lines {o1} over 1%, {o2} over 2%, {o5} over 5%, largest {worst:+.2}% ({})\n", names[wid as usize]);
    }
    println!("{what}: rail trips {:+.3}%\n{s}", 100.0 * (x.rail / y.rail - 1.0));
}

#[test]
#[ignore]
fn t436() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let city = std::env::var("TW_CITY").unwrap_or("nyc".into());
    let k: usize = std::env::var("TW_K").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/{city}.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let mut e = EverydayApi::new(&h, &b).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, j) = RealNet::load(&format!("{root}/data/work/realnet/{city}_real_track.json"));
    let names: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap_or(""), l["name"].as_str().unwrap_or(""))).collect();
    if city == "nyc" {
        let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/../artifacts/trainworld/t161/nyc_real_complex.json")).unwrap()).unwrap();
        let stations = j["stations"].as_array().unwrap();
        let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
        let mut of: HashMap<(i64, i64), u32> = Default::default();
        for s in cj["stations"].as_array().unwrap() {
            of.insert(key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap()), s["complex"].as_u64().unwrap() as u32);
        }
        let ns = stations.len();
        let complex: Vec<u32> = stations.iter().enumerate().map(|(i, s)| of.get(&key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap())).copied().unwrap_or((ns + i) as u32)).collect();
        a.set_complexes(&complex);
        e.set_complexes(&complex);
    }
    let id = (0..line_n.len() as u32).collect();
    let base = Net { st_xy, line_n, stops, times, tph, cars, id };
    println!("{city}: {} lines, {} stations; warm solves run {k} crowding round(s) after their first pass", base.line_n.len(), base.st_xy.len() / 2);
    let c0 = run(&mut a, &mut e, &base, None, 3);
    println!("cold base: {:.1} s, {} passes, rail trips {:.0}", c0.ms / 1000.0, c0.passes, c0.rail);
    let mut by: Vec<(u32, f64)> = c0.board.iter().map(|(&i, &v)| (i, v)).collect();
    by.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let (busiest, fifth) = (by[0].0, by[4].0);
    println!("edit 1: trains an hour on {} halved; edit 2: {} removed", names[busiest as usize], names[fifth as usize]);
    let s1 = base.scale_tph(base.at(busiest), 0.5);
    let s2 = s1.remove(s1.at(fifth));
    let c1 = run(&mut a, &mut e, &s1, None, 3);
    let c2 = run(&mut a, &mut e, &s2, None, 3);
    println!("cold after edit 1: {:.1} s, {} passes; after edit 2: {:.1} s, {} passes", c1.ms / 1000.0, c1.passes, c2.ms / 1000.0, c2.passes);
    cmp("edit 1's own effect (cold after edit 1 against cold base)", &c1, &c0, &names);
    cmp("edit 2's own effect (cold after edit 2 against cold after edit 1)", &c2, &c1, &names);
    let w0 = run(&mut a, &mut e, &base, Some((&c0, &base)), k);
    let w1 = run(&mut a, &mut e, &s1, Some((&c0, &base)), k);
    let w2 = run(&mut a, &mut e, &s2, Some((&w1, &s1)), k);
    let w1u = run(&mut a, &mut e, &s1, Some((&w2, &s2)), k);
    let w0u = run(&mut a, &mut e, &base, Some((&w1u, &s1)), k);
    for (what, w) in [("warm, nothing changed", &w0), ("warm edit 1", &w1), ("warm edit 2", &w2), ("warm undo 2", &w1u), ("warm undo 1", &w0u)] {
        println!("{what}: {:.1} s, {} passes", w.ms / 1000.0, w.passes);
    }
    cmp("warm re-solve of the base, nothing changed, against cold", &w0, &c0, &names);
    cmp("warm after edit 1 against cold", &w1, &c1, &names);
    cmp("warm after edit 2 against cold", &w2, &c2, &names);
    cmp("warm after undoing edit 2 against cold", &w1u, &c1, &names);
    cmp("warm after undoing edit 2 against warm before edit 2 (the same network)", &w1u, &w1, &names);
    cmp("warm after undoing both against cold base", &w0u, &c0, &names);
    cmp("warm after undoing both against warm nothing-changed", &w0u, &w0, &names);
}
