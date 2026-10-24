//! T-1092 (scratch): everyday trip lengths and the share ending in the job core, by purpose, from
//! every destination sample. Included into everyday.rs for the run only:
//!
//!   T1092_PACK=<pack .json> T1092_OUT=out.json [T1092_SET=k=v,..] <test exe> t1092_meas --ignored

use super::*;
use crate::demand::api::DemandApi;
use crate::demand::pack;
use serde_json::json;

#[test]
#[ignore]
fn t1092_meas() {
    let pack_path = std::env::var("T1092_PACK").expect("T1092_PACK");
    let out_path = std::env::var("T1092_OUT").expect("T1092_OUT");
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    if let Ok(s) = std::env::var("T1092_SET") {
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, x) = kv.split_once('=').unwrap();
            api.params_mut().set_extra(k, x.parse().unwrap()).unwrap();
        }
    }
    let t0 = std::time::Instant::now();
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let open_s = t0.elapsed().as_secs_f64();
    let z = &*ev.engine.z;
    // the job core: the densest zones by jobs per km² of land holding the first 10% of jobs
    let land = |k: usize| (z.ncell[k].max(1) as f32 * CELL_KM2).min(z.square_km2(k)) as f64;
    let mut order: Vec<usize> = (0..z.n).collect();
    order.sort_by(|&a, &b| (z.jobs[b] / land(b)).partial_cmp(&(z.jobs[a] / land(a))).unwrap());
    let tj: f64 = z.jobs.iter().sum();
    let mut core = vec![false; z.n];
    let mut acc = 0.0;
    for &k in &order {
        core[k] = true;
        acc += z.jobs[k];
        if acc >= 0.1 * tj {
            break;
        }
    }
    let kinds = ev.totals.len();
    // per purpose (and all, last): trips, <2, 2-5, >5 km, km, no-rail walk, to core, >5 to core
    let mut s = vec![[0f64; 8]; kinds + 1];
    let mut dest = vec![vec![0f64; z.n]; kinds];
    for k in 0..ev.destinations.len() {
        let Some((kind, c, d, w, pw)) = ev.edge(k) else { continue };
        let dist = ev.distance(c, d) as f64 / 1000.0;
        let w = w as f64;
        let dz = z.cell_zone[d] as usize;
        dest[kind][dz] += w;
        for i in [kind, kinds] {
            let a = &mut s[i];
            a[0] += w;
            a[if dist < 2.0 { 1 } else if dist < 5.0 { 2 } else { 3 }] += w;
            a[4] += w * dist;
            a[5] += w * pw as f64;
            if core[dz] {
                a[6] += w;
                if dist >= 5.0 {
                    a[7] += w;
                }
            }
        }
    }
    let rows: Vec<_> = (0..=kinds)
        .map(|i| {
            let a = s[i];
            let t = a[0].max(1e-9);
            let top = if i < kinds { dest[i].iter().cloned().fold(0.0, f64::max) / t } else { 0.0 };
            // the five busiest destination zones: share, centre (mean of its cells, metres)
            let mut tops: Vec<serde_json::Value> = vec![];
            if i < kinds {
                let mut o: Vec<usize> = (0..z.n).collect();
                o.sort_by(|&a, &b| dest[i][b].partial_cmp(&dest[i][a]).unwrap());
                for &k in o.iter().take(5) {
                    let cells: Vec<usize> = (0..z.cell_zone.len()).filter(|&c| z.cell_zone[c] as usize == k).collect();
                    let m = cells.len().max(1) as f32;
                    let cx = cells.iter().map(|&c| ev.engine.city.x[c]).sum::<f32>() / m;
                    let cy = cells.iter().map(|&c| ev.engine.city.y[c]).sum::<f32>() / m;
                    tops.push(json!([100.0 * dest[i][k] / t, cx, cy, core[k], z.jobs[k] / land(k)]));
                }
            }
            json!({
                "purpose": if i < kinds { KINDS[i] } else { "all" },
                "trips": a[0], "under2": 100.0 * a[1] / t, "b2_5": 100.0 * a[2] / t, "over5": 100.0 * a[3] / t,
                "mean_km": a[4] / t, "walk_norail": 100.0 * a[5] / t, "to_core": 100.0 * a[6] / t,
                "over5_to_core": 100.0 * a[7] / t, "top_zone": 100.0 * top, "tops": tops,
            })
        })
        .collect();
    let out = json!({"pack": pack_path, "open_s": open_s, "core_zones": core.iter().filter(|&&x| x).count(), "rows": rows});
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
