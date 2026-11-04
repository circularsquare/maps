//! T-1508: the market's longer trips by country and distance, from the world's rows alone (no packs).
//! For every ordered pair of places 100 km or more apart: the directed trips a day, ground and tail
//! apart (`directed_by` with one part zeroed), and the pair's km. Writes T1508_OUT.
//!
//!   T1521_ROOT=<trainworld> T1508_OUT=out.json cargo test --release --features demand t1508_dump -- --ignored --nocapture

use super::tests::real_world;
use super::*;

#[test]
#[ignore]
fn t1508_dump() {
    let root = std::env::var("T1521_ROOT").expect("T1521_ROOT");
    let out_path = std::env::var("T1508_OUT").expect("T1508_OUT");
    let a = IntercityApi::new(&real_world(&root, Some(&[] as &[&str]))).unwrap();
    let m = &a.m;
    let np = m.places.len();
    let places: Vec<_> = m
        .places
        .iter()
        .map(|p| serde_json::json!({"zone": p.zone, "country": m.countries[p.country as usize], "pop": p.pop, "inc": p.inc, "dep": p.dep, "has": p.has, "lon": p.lon, "lat": p.lat}))
        .collect();
    let mut pairs = vec![];
    for i in 0..np {
        let p = &m.places[i];
        if !p.has {
            continue;
        }
        for j in 0..np {
            let q = &m.places[j];
            if i == j || !q.has {
                continue;
            }
            let r = m.pr(i, j);
            if r.km < 100. {
                continue;
            }
            let gr = m.directed_by(r.b, r.rel, p.g, 0., q.a, q.ta, r.km);
            let tl = m.directed_by(r.b, r.rel, 0., p.t, q.a, q.ta, r.km);
            if gr + tl < 0.5 {
                continue;
            }
            pairs.push(serde_json::json!([i, j, (r.km * 10.).round() / 10., (gr * 100.).round() / 100., (tl * 100.).round() / 100.]));
        }
    }
    let out = serde_json::json!({"places": places, "pairs": pairs});
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("wrote {out_path}: {} pairs", pairs.len());
}
