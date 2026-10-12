//! NYC-first home-based outings. Job counts are attraction weights, never customer capacities.
//! Coarse destination gravity is integrated by deterministic stratified cell samples. Endpoint
//! distances, the 150 m exclusion, walking and regional overlap use the actual sampled cells.
use super::{
    api::{DemandApi, PEAK_HOUR_FACTOR},
    clock::now_ms,
    kernel::*,
    network::PERIODS,
    pack,
};
use crate::track::params::PERIOD_HOURS;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

const K: usize = 16;
const NONE: u32 = u32::MAX;
pub const MIN_DISTANCE_M: f32 = 150.0;
const RATE: [f32; 4] = [0.3, 0.2, 0.9, 0.6];
const DECAY: [f32; 4] = [3000.0, 6000.0, 2500.0, 6000.0];
pub const OUTWARD: [[f32; 5]; 4] = [
    [0.10, 0.55, 0.30, 0.04, 0.01],
    [0.05, 0.35, 0.42, 0.17, 0.01],
    [0.94, 0.04, 0.015, 0.004, 0.001],
    [0.65, 0.27, 0.07, 0.009, 0.001],
];
pub const RETURN: [[f32; 5]; 4] = [
    [0.06, 0.49, 0.36, 0.08, 0.01],
    [0.02, 0.25, 0.43, 0.27, 0.03],
    [0.01, 0.82, 0.15, 0.018, 0.002],
    [0.04, 0.42, 0.42, 0.11, 0.01],
];

fn unit(mut x: u32) -> f64 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x as f64 + 0.5) / 4294967296.0
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(distance: f32) -> (String, Vec<u8>) {
        let city = pack::City {
            name: "test".into(),
            h3: vec![1, 2],
            x: vec![0.0, distance],
            y: vec![0.0, 0.0],
            pop: vec![100.0, 0.0],
            jobs: vec![0.0, 100.0],
            gravity: None,
        };
        let (header, mut bin) = pack::write(&city);
        let mut h: serde_json::Value = serde_json::from_str(&header).unwrap();
        for name in ["errand_jobs", "leisure_jobs"] {
            let offset = bin.len();
            bin.extend([0.0f32, 100.0].into_iter().flat_map(|v| v.to_le_bytes()));
            h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":2,"offset":offset}));
        }
        (h.to_string(), bin)
    }
    fn network(a: &mut EverydayApi, distance: f32, tph: [f32; 3]) {
        let times = [
            30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0,
        ];
        a.set_network(
            &[0.0, 0.0, distance, 0.0],
            &[2],
            &[0, 1],
            &times,
            &tph,
            &[4.0],
        )
        .unwrap();
        a.set_fares(0.0, 0.0, &[distance / 1000.0, 0.0, 0.0, distance / 1000.0])
            .unwrap();
    }
    #[test]
    fn cutoff_removes_all_modes_without_replacement_and_keeps_boundary() {
        let (h, b) = fixture(149.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.totals, [0.0, 0.0]);
        network(&mut a, 149.0, [12.0; 3]);
        for q in 0..5 {
            assert!(a.solve(q));
            assert_eq!(a.summary(q)[1], 0.0);
        }
        assert!(a.cell_modes().iter().all(|&v| v == 0.0));
        assert!(a.selection(1, &[1]).iter().all(|&v| v == 0.0));
        let (h, b) = fixture(150.0);
        let a = EverydayApi::new(&h, &b).unwrap();
        assert!((a.totals[0] - 30.0).abs() < 1e-4);
        for k in 0..a.destinations.len() {
            if let Some((_, c, d, _, _)) = a.edge(k) {
                assert!(a.distance(c, d) >= 150.0);
            }
        }
    }
    #[test]
    fn purpose_periods_cells_and_assignment_conserve_trip_legs() {
        let (h, b) = fixture(4000.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        network(&mut a, 4000.0, [12.0; 3]);
        for q in 0..5 {
            assert!(a.solve(q));
        }
        let total = a.totals.iter().sum::<f64>() * 2.0;
        let summaries: Vec<_> = (0..5).map(|q| a.summary(q)).collect();
        assert!((summaries.iter().map(|v| v[1]).sum::<f64>() - total).abs() < 1e-3);
        let modes = a.cell_modes();
        assert!((modes[..6].iter().sum::<f32>() as f64 - total).abs() < 1e-3);
        assert!((modes[6..].iter().sum::<f32>() as f64 - total).abs() < 1e-3);
        let selected = a.selection(1, &[1]);
        let stride = 36;
        for kind in 0..2 {
            let v = &selected[kind * stride..(kind + 1) * stride];
            assert!((v[..30].iter().sum::<f32>() - v[30..].iter().sum::<f32>()).abs() < 1e-3);
        }
        let rail = summaries.iter().map(|v| v[2]).sum::<f64>();
        assert!(rail > 0.0);
        assert!((modes[0] as f64 - rail).abs() < 1e-3);
        assert!((a.board(0).iter().sum::<f32>() as f64 - summaries[0][2]).abs() < 1e-3);
        let before = a.summary(0)[2];
        let shared: Vec<_> = a.seg(0).iter().map(|v| v * 100000.0).collect();
        assert!(a.crowd_with_external(0, &shared));
        assert!(a.summary(0)[2] < before);
        assert!(!a.crowd_with_external(0, &[f32::NAN]));
    }
    #[test]
    fn schedules_and_directional_service_are_independent() {
        for k in 0..OUTWARD.len() {
            assert!((OUTWARD[k].iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!((RETURN[k].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let (h, b) = fixture(4000.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        network(&mut a, 4000.0, [0.0, 12.0, 0.0]);
        for q in 0..5 {
            assert!(a.solve(q));
        }
        assert_eq!(a.summary(0)[2], 0.0);
        assert!(a.summary(1)[2] > 0.0);
        assert_eq!(a.summary(4)[2], 0.0);
        // Return costs must see their own route: make homeward hops extremely slow.
        let times = [
            30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 0.0, 3600.0, 0.0, 3600.0, 0.0, 3600.0,
        ];
        a.set_network(
            &[0.0, 0.0, 4000.0, 0.0],
            &[2],
            &[0, 1],
            &times,
            &[12.0; 3],
            &[4.0],
        )
        .unwrap();
        a.solve(1);
        assert!(a.per[1]
            .as_ref()
            .unwrap()
            .probability
            .iter()
            .any(|p| p[0] > p[1] + 0.1));
    }

    fn school_fixture(distance: f32) -> (String, Vec<u8>) {
        let (h, mut b) = fixture(distance);
        let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
        for (name, values) in [
            ("school_children", [10.0f32, 0.0]),
            ("school_enrollment", [0.0, 100.0]),
        ] {
            let offset = b.len();
            b.extend(values.into_iter().flat_map(|v| v.to_le_bytes()));
            h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":2,"offset":offset}));
        }
        (h.to_string(), b)
    }
    #[test]
    fn school_enrollment_sets_budget_and_only_child_cells_supply_homes() {
        let (h, b) = school_fixture(4000.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.purpose_count(), 3);
        assert!(
            (a.totals[2] - 90.0).abs() < 1e-3,
            "attendance budget follows enrollment, not resident totals"
        );
        let mut school_edges = 0;
        for k in 2 * 2 * K..a.destinations.len() {
            if let Some((kind, home, school, _, _)) = a.edge(k) {
                assert_eq!((kind, home, school), (2, 0, 1));
                school_edges += 1;
            }
        }
        assert_eq!(school_edges, K);
        network(&mut a, 4000.0, [12.0; 3]);
        for q in 0..5 {
            assert!(a.solve(q));
        }
        let selected = a.selection(1, &[1]);
        let school = &selected[72..108];
        assert!((school[..30].iter().sum::<f32>() - 180.0).abs() < 1e-3);
        assert!((school[30..].iter().sum::<f32>() - 180.0).abs() < 1e-3);
        let outbound_midday = (0..3).map(|m| school[m * 5 + 1]).sum::<f32>();
        let homeward_midday = (0..3).map(|m| school[15 + m * 5 + 1]).sum::<f32>();
        assert!((homeward_midday / outbound_midday - 20.5).abs() < 1e-3);
        let home_selected = a.selection(0, &[0]);
        assert!((home_selected[72..102].iter().sum::<f32>() - 180.0).abs() < 1e-3);
        let (h, b) = school_fixture(149.0);
        let a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.totals[2], 0.0);
    }
    #[test]
    fn short_everyday_trips_keep_walking_and_disaggregate_motorized_budgets() {
        let city = pack::City { name: "short".into(), h3: vec![1,2,3],
            x: vec![0.0,100.0,400.0], y: vec![0.0;3],
            pop: vec![100.0,100.0,0.0], jobs: vec![0.0,0.0,100.0], gravity: None };
        let (h, mut b) = pack::write(&city);
        let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
        for name in ["errand_jobs", "leisure_jobs"] {
            let offset = b.len();
            b.extend([0.0f32,0.0,100.0].into_iter().flat_map(|v| v.to_le_bytes()));
            h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":3,"offset":offset}));
        }
        let mut a = EverydayApi::new(&h.to_string(), &b).unwrap();
        a.engine.params_mut().asc_rail = 20.0;
        network(&mut a, 400.0, [60.0;3]);
        assert!(a.pairs.len()<a.destinations.iter().filter(|&&v|v!=NONE).count());
        for q in 0..5 { a.solve(q); }
        let modes = a.cell_modes();
        for c in 0..2 {
            let total = modes[c]+modes[3+c]+modes[6+c];
            let pw = a.engine.base.walk_share((400.0-city.x[c])/1000.0);
            assert!((modes[3+c]/total-pw).abs()<1e-5);
            assert!(pw>0.96 && modes[c]/total<=1.0-pw+1e-5);
            assert!(modes[6+c]>=0.0);
        }
        let summaries: Vec<_>=(0..5).map(|q|a.summary(q)).collect();
        let rail: f64=summaries.iter().map(|v|v[2]).sum();
        assert!((rail-modes[..3].iter().sum::<f32>() as f64).abs()<1e-3);
        let before=a.summary(0)[3];
        let shared:Vec<_>=a.seg(0).iter().map(|v|v*1e6).collect();
        a.crowd_with_external(0,&shared);
        assert!((a.summary(0)[3]-before).abs()<1e-6);
        let selected=a.selection(1,&[2]);
        for kind in 0..2 {
            let slice=&selected[kind*39..(kind+1)*39];
            assert!((slice[..30].iter().sum::<f32>()-slice[30..].iter().sum::<f32>()).abs()<1e-3);
        }
        println!("Everyday baseline walking: 300m {:.2}%, 400m {:.2}%",a.engine.base.walk_share(0.3)*100.0,a.engine.base.walk_share(0.4)*100.0);
    }

    fn university_fixture(distance: f32) -> (String, Vec<u8>) {
        let (h, mut b) = school_fixture(distance);
        let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
        // College homes are opposite the school homes, proving independent origin weights.
        for (name, values) in [("university_adults", [0.0f32, 50.0]),
                               ("university_enrollment", [100.0, 0.0])] {
            let offset = b.len();
            b.extend(values.into_iter().flat_map(|v| v.to_le_bytes()));
            h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":2,"offset":offset}));
        }
        (h.to_string(), b)
    }
    #[test]
    fn university_is_optional_independent_and_conserves_purpose_timing() {
        let (oldh, oldb) = school_fixture(4000.0);
        let old = EverydayApi::new(&oldh, &oldb).unwrap();
        let (h, b) = university_fixture(4000.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.purpose_count(), 4);
        assert_eq!(&a.info()[..9], &old.info());
        assert_eq!(&a.destinations[..old.destinations.len()], &old.destinations);
        assert!((a.totals[3]-60.0).abs()<1e-4);
        for k in 3*2*K..a.destinations.len() {
            if let Some((kind, home, campus, _, _)) = a.edge(k) {
                assert_eq!((kind,home,campus),(3,1,0));
            }
        }
        network(&mut a, 4000.0, [12.0;3]);
        for q in 0..5 { a.solve(q); }
        let selected=a.selection(1,&[0]);
        let uni=&selected[108..144];
        assert!((uni[..30].iter().sum::<f32>()-120.0).abs()<1e-3);
        assert!((uni[30..].iter().sum::<f32>()-120.0).abs()<1e-3);
        let outward_morning=(0..3).map(|m|uni[m*5]).sum::<f32>();
        let return_evening=(0..3).map(|m|uni[15+m*5+2]).sum::<f32>();
        assert!((outward_morning/60.0-0.65).abs()<1e-5);
        assert!((return_evening/60.0-0.42).abs()<1e-5);
        let modes=a.cell_modes();
        for m in 0..3 {
            let total=(0..4).map(|kind|(0..5).map(|q|selected[kind*36+m*5+q]+selected[kind*36+15+m*5+q]).sum::<f32>()).sum::<f32>();
            assert!((total-modes[(m+3)*2]).abs()<1e-3);
        }
        let mut bad:serde_json::Value=serde_json::from_str(&h).unwrap();
        bad["arrays"].as_array_mut().unwrap().retain(|v|v["name"]!="university_adults");
        assert!(EverydayApi::new(&bad.to_string(),&b).is_err());
        let mut no_school:serde_json::Value=serde_json::from_str(&h).unwrap();
        no_school["arrays"].as_array_mut().unwrap().retain(|v|v["name"]!="school_children" && v["name"]!="school_enrollment");
        let only=EverydayApi::new(&no_school.to_string(),&b).unwrap();
        assert_eq!(only.purpose_count(),4);
        assert_eq!(only.totals[2],0.0);
        assert!((only.totals[3]-60.0).abs()<1e-4);
        let (h,b)=university_fixture(149.0);
        assert_eq!(EverydayApi::new(&h,&b).unwrap().totals[3],0.0);
    }
    #[test]
    #[ignore] // Real source pack check, invoked explicitly by T-152.
    fn university_real_pack_preserves_existing_generation() {
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/packs");
        let h=std::fs::read_to_string(root.join("nyc.json")).unwrap();
        let b=std::fs::read(root.join("nyc.bin")).unwrap();
        let mut oldh:serde_json::Value=serde_json::from_str(&h).unwrap();
        let enrolled=oldh["university"]["enrolled"].as_f64().unwrap();
        oldh["arrays"].as_array_mut().unwrap().retain(|v|v["name"]!="university_adults" && v["name"]!="university_enrollment");
        let old=EverydayApi::new(&oldh.to_string(),&b).unwrap();
        let new=EverydayApi::new(&h,&b).unwrap();
        assert_eq!(old.purpose_count(),3); assert_eq!(new.purpose_count(),4);
        assert_eq!(&new.info()[..9],&old.info());
        assert_eq!(&new.destinations[..old.destinations.len()],&old.destinations);
        assert!(new.totals[3]>0.0 && new.totals[3]<=enrolled*0.6+1.0);
        println!("University real pack: enrolled {enrolled}, outings {}, legs {}, walk share {}, mean km {}; prior generation identical",new.info()[9],new.info()[9]*2.0,new.info()[10],new.info()[11]);
    }

    #[test]
    fn optional_school_inputs_preserve_existing_layers_and_require_both_arrays() {
        let (h, b) = fixture(4000.0);
        let old = EverydayApi::new(&h, &b).unwrap();
        let (h, b) = school_fixture(4000.0);
        let new = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(
            &new.destinations[..old.destinations.len()],
            &old.destinations
        );
        assert_eq!(&new.info()[..6], &old.info());
        let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
        h["arrays"]
            .as_array_mut()
            .unwrap()
            .retain(|a| a["name"] != "school_children");
        assert!(EverydayApi::new(&h.to_string(), &b).is_err());
    }
}
fn local_share(d: f32) -> f32 {
    let x = ((d - 40000.0) / 40000.0).clamp(0.0, 1.0);
    1.0 - x * x * (3.0 - 2.0 * x)
}
#[derive(Clone, Copy)]
struct Pair {
    u: u32,
    v: u32,
    kind: usize,
    trips: f32,
    walk: f32,
    distance: f32,
}
struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    loads: Loads,
    probability: Vec<[f32; 2]>,
    round: usize,
    ms: f64,
    fac: Vec<f32>,
    uniform: bool,
    trips: f64,
    rail: f64,
    walk: f64,
}

#[wasm_bindgen]
pub struct EverydayApi {
    engine: DemandApi,
    destinations: Vec<u32>,
    edge_pair: Vec<u32>,
    pairs: Vec<Pair>,
    per: Vec<Option<Period>>,
    totals: Vec<f64>,
    walks: Vec<f64>,
    /// Optional enrollment budgets: school generation samples homes FROM each activity site.
    school_enrollment: Vec<f32>,
    university_enrollment: Vec<f32>,
}

impl EverydayApi {
    /// Pair probabilities are unconditional shares of all trips. Disaggregate rail by the
    /// sampled cell edge's motorized budget, not its total: colocated subzones can contain
    /// edges with very different walking shares. This preserves pair loads and cell totals.
    fn edge_motorized_scale(&self, pi: u32, walk: f32) -> f32 {
        let motorized = 1.0 - self.pairs[pi as usize].walk;
        if motorized > 0.0 { (1.0 - walk) / motorized } else { 0.0 }
    }
    fn distance(&self, c: usize, d: usize) -> f32 {
        let city = &self.engine.city;
        ((city.x[c] - city.x[d]).powi(2) + (city.y[c] - city.y[d]).powi(2)).sqrt()
    }
    fn edge(&self, k: usize) -> Option<(usize, usize, usize, f32, f32)> {
        let d = self.destinations[k];
        if d == NONE {
            return None;
        }
        let n = self.engine.city.len();
        let kind = k / (n * K);
        let fixed = (k / K) % n;
        let (c, d) = if kind >= 2 {
            (d as usize, fixed)
        } else {
            (fixed, d as usize)
        };
        let distance = self.distance(c, d);
        let budget = match kind {
            2 => self.school_enrollment[d],
            3 => self.university_enrollment[d],
            _ => self.engine.city.pop[c],
        };
        let weight = budget * RATE[kind] / K as f32 * local_share(distance);
        Some((
            kind,
            c,
            d,
            weight,
            self.engine.base.walk_share(distance / 1000.0),
        ))
    }
    fn pass(&self, g: &Graph, p: &Params, q: usize) -> (Vec<[f32; 2]>, Loads) {
        let mut probability = vec![[0.0; 2]; self.pairs.len()];
        let s = g.n_st;
        let Some(sz) = &self.engine.sz else {
            return (probability, Loads::zero(s, g.n - s));
        };
        let want = vec![true; s]; // outward destinations AND homeward destinations
        let loads = with_skims(g, p, &want, |sk| {
            let er = station_exp(sk, p);
            let mut od = vec![0.0; s * s];
            let th = p.station_theta;
            let mut min = vec![0.0; sz.n];
            let mut acc = vec![0.0; sz.acc_st.len()];
            for u in 0..sz.n {
                let r = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
                min[u] = sz.acc_w[r.clone()]
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min);
                for k in r {
                    acc[k] = fast_exp(-th * (sz.acc_w[k] - min[u]));
                }
            }
            let density = self.engine.z.density();
            for (i, pair) in self.pairs.iter().enumerate() {
                for direction in 0..2 {
                    let (u, v) = if direction == 0 {
                        (pair.u as usize, pair.v as usize)
                    } else {
                        (pair.v as usize, pair.u as usize)
                    };
                    if min[u] + min[v] > p.pair_walk_max {
                        continue;
                    }
                    let a = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
                    let b = sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize;
                    let mut sum = 0.0;
                    for ka in a.clone() {
                        for kb in b.clone() {
                            sum += acc[ka]
                                * acc[kb]
                                * er[sz.acc_st[ka] as usize * s + sz.acc_st[kb] as usize];
                        }
                    }
                    if sum < 1e-30 {
                        continue;
                    }
                    let road = p.drive_min(
                        pair.distance,
                        p.place_min_per_km(density[sz.zone[u] as usize]),
                        p.place_min_per_km(density[sz.zone[v] as usize]),
                    ) * p.w_drive
                        + p.park_min
                        + (self.engine.z.jobs[sz.zone[v] as usize] as f32
                            / (self.engine.z.zone_m / 1000.0).powi(2)
                            / p.park_jobs_per_km2_per_min)
                            .min(p.park_cost_cap_min);
                    let rail = min[u] + min[v] + p.fare_min - fast_ln(sum) / th;
                    let pr = (1.0 - pair.walk) / (1.0 + fast_exp(-(p.beta * (road - rail) + p.asc_rail)));
                    probability[i][direction] = pr;
                    let share = if direction == 0 {
                        OUTWARD[pair.kind][q]
                    } else {
                        RETURN[pair.kind][q]
                    };
                    let trips = pair.trips * share * pr / sum;
                    for ka in a.clone() {
                        for kb in b.clone() {
                            let ia = sz.acc_st[ka] as usize;
                            let ib = sz.acc_st[kb] as usize;
                            od[ia * s + ib] += trips * acc[ka] * acc[kb] * er[ia * s + ib];
                        }
                    }
                }
            }
            assign(g, sk, &od)
        });
        (probability, loads)
    }
    fn mode_totals(&self, q: usize, probability: &[[f32; 2]]) -> (f64, f64, f64) {
        let mut trips = 0.0;
        let mut walk = 0.0;
        let mut rail = 0.0;
        for kind in 0..self.totals.len() {
            let share = (OUTWARD[kind][q] + RETURN[kind][q]) as f64;
            trips += self.totals[kind] * share;
            walk += self.walks[kind] * share;
        }
        for (i, pair) in self.pairs.iter().enumerate() {
            let r = pair.trips as f64
                * (OUTWARD[pair.kind][q] as f64 * probability[i][0] as f64
                    + RETURN[pair.kind][q] as f64 * probability[i][1] as f64);
            rail += r;
        }
        (trips, rail, walk.max(0.0))
    }
}

#[wasm_bindgen]
impl EverydayApi {
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str, bin: &[u8]) -> Result<EverydayApi, String> {
        let mut engine = DemandApi::new(header, bin)?;
        let n = engine.city.len();
        let pop =
            pack::cell_weights(header, bin, "pop", n)?.ok_or("missing resident population")?;
        let retail = pack::cell_weights(header, bin, "errand_jobs", n)?
            .ok_or("missing errand attractors")?;
        let leisure = pack::cell_weights(header, bin, "leisure_jobs", n)?
            .ok_or("missing leisure attractors")?;
        let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
        let (tp, tl) = (sum(&pop), sum(&leisure));
        if tp <= 0.0 || tl <= 0.0 || sum(&retail) <= 0.0 {
            return Err("empty everyday weights".into());
        }
        let social: Vec<f32> = pop
            .iter()
            .zip(&leisure)
            .map(|(&p, &l)| (0.5 * p as f64 / tp + 0.5 * l as f64 / tl) as f32)
            .collect();
        let children = pack::cell_weights(header, bin, "school_children", n)?;
        let school_enrollment = pack::cell_weights(header, bin, "school_enrollment", n)?;
        if children.is_some() != school_enrollment.is_some() {
            return Err("incomplete school inputs".into());
        }
        if let (Some(children), Some(enrollment)) = (&children, &school_enrollment) {
            if sum(children) <= 0.0 || sum(enrollment) <= 0.0 {
                return Err("empty school inputs".into());
            }
        }
        let adults = pack::cell_weights(header, bin, "university_adults", n)?;
        let university_enrollment = pack::cell_weights(header, bin, "university_enrollment", n)?;
        if adults.is_some() != university_enrollment.is_some() {
            return Err("incomplete university inputs".into());
        }
        if let (Some(adults), Some(enrollment)) = (&adults, &university_enrollment) {
            if sum(adults) <= 0.0 || sum(enrollment) <= 0.0 {
                return Err("empty university inputs".into());
            }
        }
        // Purpose indices stay stable: university-only packs leave an empty school slot.
        let empty_school = vec![0.0; if children.is_none() && adults.is_some() { n } else { 0 }];
        let mut weights = vec![&retail, &social];
        if children.is_some() || adults.is_some() {
            weights.push(children.as_ref().unwrap_or(&empty_school));
        }
        if let Some(adults) = &adults { weights.push(adults); }
        let kinds = weights.len();
        engine.city.pop = pop;
        // Access centroids and road density use real residents/employment, not commuter weights.
        engine.city.jobs = pack::cell_weights(header, bin, "jobs", n)?.ok_or("missing jobs")?;
        engine.z.pop.fill(0.0);
        engine.z.jobs.fill(0.0);
        for c in 0..n {
            let z = engine.z.cell_zone[c] as usize;
            engine.z.pop[z] += engine.city.pop[c] as f64;
            engine.z.jobs[z] += engine.city.jobs[c] as f64;
        }
        let nz = engine.z.n;
        let mut members = vec![vec![]; nz];
        for c in 0..n {
            members[engine.z.cell_zone[c] as usize].push(c);
        }
        let mut destinations = vec![NONE; kinds * n * K];
        for (kind, weights) in weights.into_iter().enumerate() {
            let mut mass = vec![0.0; nz];
            let mut cell_cdf = vec![vec![]; nz];
            for z in 0..nz {
                for &c in &members[z] {
                    mass[z] += weights[c] as f64;
                    cell_cdf[z].push(mass[z]);
                }
            }
            let zones: Vec<usize> = (0..nz).filter(|&z| mass[z] > 0.0).collect();
            for oz in 0..nz {
                let x = engine.z.x0 + (engine.z.gx[oz] as f32 + 0.5) * engine.z.zone_m;
                let y = engine.z.y0 + (engine.z.gy[oz] as f32 + 0.5) * engine.z.zone_m;
                let mut cdf = Vec::with_capacity(zones.len());
                let mut total = 0.0;
                for &dz in &zones {
                    let dx = engine.z.x0 + (engine.z.gx[dz] as f32 + 0.5) * engine.z.zone_m - x;
                    let dy = engine.z.y0 + (engine.z.gy[dz] as f32 + 0.5) * engine.z.zone_m - y;
                    let d = if oz == dz {
                        0.52 * engine.z.zone_m
                    } else {
                        (dx * dx + dy * dy).sqrt()
                    };
                    total += mass[dz] * (-d / DECAY[kind]).exp() as f64;
                    cdf.push(total);
                }
                if total <= 0.0 {
                    continue;
                }
                for &c in &members[oz] {
                    let budget = match kind {
                        2 => school_enrollment.as_ref().map_or(0.0, |v| v[c]),
                        3 => university_enrollment.as_ref().unwrap()[c],
                        _ => engine.city.pop[c],
                    };
                    if budget <= 0.0 {
                        continue;
                    }
                    let seed = (engine.city.h3[c] ^ (engine.city.h3[c] >> 32)) as u32
                        ^ (kind as u32).wrapping_mul(0x9e3779b9);
                    let offset = unit(seed);
                    for k in 0..K {
                        let target = (k as f64 + offset) / K as f64 * total;
                        let j = cdf.partition_point(|&v| v <= target).min(zones.len() - 1);
                        let z = zones[j];
                        let target =
                            unit(seed.wrapping_add((k as u32 + 1).wrapping_mul(0x85ebca6b)))
                                * mass[z];
                        let j = cell_cdf[z]
                            .partition_point(|&v| v <= target)
                            .min(members[z].len() - 1);
                        let d = members[z][j];
                        let distance = ((engine.city.x[c] - engine.city.x[d]).powi(2)
                            + (engine.city.y[c] - engine.city.y[d]).powi(2))
                        .sqrt();
                        // Excluded trips vanish, including from all-mode denominators. No replacement.
                        if distance >= MIN_DISTANCE_M && local_share(distance) > 0.0 {
                            destinations[(kind * n + c) * K + k] = d as u32;
                        }
                    }
                }
            }
        }
        let mut api = EverydayApi {
            engine,
            destinations,
            edge_pair: vec![],
            pairs: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            totals: vec![0.0; kinds],
            walks: vec![0.0; kinds],
            school_enrollment: school_enrollment.unwrap_or_default(),
            university_enrollment: university_enrollment.unwrap_or_default(),
        };
        for k in 0..api.destinations.len() {
            if let Some((kind, _, _, w, pw)) = api.edge(k) {
                api.totals[kind] += w as f64;
                api.walks[kind] += (w * pw) as f64;
            }
        }
        Ok(api)
    }
    /// Outings retained, walking fraction, and mean distance by purpose.
    pub fn info(&self) -> Vec<f64> {
        let mut distance = vec![0.0; self.totals.len()];
        for k in 0..self.destinations.len() {
            if let Some((kind, c, d, w, _)) = self.edge(k) {
                distance[kind] += self.distance(c, d) as f64 * w as f64 / 1000.0;
            }
        }
        (0..self.totals.len())
            .flat_map(|k| {
                [
                    self.totals[k],
                    self.walks[k] / self.totals[k].max(1e-9),
                    distance[k] / self.totals[k].max(1e-9),
                ]
            })
            .collect()
    }
    pub fn purpose_count(&self) -> u32 {
        self.totals.len() as u32
    }
    pub fn set_network(
        &mut self,
        xy: &[f32],
        line_n: &[u32],
        stops: &[u32],
        times: &[f32],
        tph: &[f32],
        cars: &[f32],
    ) -> Result<f64, String> {
        let t = now_ms();
        self.engine
            .set_network(xy, line_n, stops, times, tph, cars)?;
        self.per.iter_mut().for_each(|v| *v = None);
        self.pairs.clear();
        self.edge_pair = vec![NONE; self.destinations.len()];
        let Some(sz) = &self.engine.sz else {
            return Ok(now_ms() - t);
        };
        let mut index = HashMap::new();
        for k in 0..self.destinations.len() {
            let Some((kind, c, d, w, pw)) = self.edge(k) else {
                continue;
            };
            let (u, v) = (sz.cell[c], sz.cell[d]);
            if u == NONE || v == NONE {
                continue;
            }
            let i = *index.entry((kind, u, v)).or_insert_with(|| {
                let i = self.pairs.len() as u32;
                self.pairs.push(Pair {
                    u,
                    v,
                    kind,
                    trips: 0.0,
                    walk: 0.0,
                    distance: 0.0,
                });
                i
            });
            self.edge_pair[k] = i;
            let distance = self.distance(c, d);
            let pair = &mut self.pairs[i as usize];
            pair.trips += w;
            pair.walk += w * pw;
            pair.distance += w * distance;
        }
        for p in &mut self.pairs {
            p.walk = (p.walk / p.trips).clamp(0.0, 1.0);
            p.distance /= p.trips;
        }
        Ok(now_ms() - t)
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, hop_km: &[f32]) -> Result<(), String> {
        self.engine.set_fares(base, per_km, hop_km)?;
        self.per.iter_mut().for_each(|v| *v = None);
        Ok(())
    }
    pub fn solve(&mut self, q: usize) -> bool {
        if q >= PERIODS {
            return false;
        }
        let Some(svc) = &self.engine.svc else {
            return false;
        };
        let t = now_ms();
        let (h0, h1) = PERIOD_HOURS[q];
        let p = Params {
            period: q,
            period_share: 1.0,
            period_hours: (h1 - h0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.engine.base.clone()
        };
        let net = DemandApi::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        for (r, fare) in g.rn_fare.iter_mut().enumerate() {
            *fare = self.engine.fare_km.get(r).copied().unwrap_or(0.0) * self.engine.fare_per_km;
        }
        let fac = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &fac);
        let (probability, loads) = self.pass(&g, &p, q);
        let (trips, rail, walk) = self.mode_totals(q, &probability);
        self.per[q] = Some(Period {
            net,
            p,
            g,
            loads,
            probability,
            round: 0,
            ms: now_ms() - t,
            fac,
            uniform: true,
            trips,
            rail,
            walk,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(Some(st)) = self.per.get(q) else {
            return false;
        };
        if total.len() != st.loads.seg.len() || total.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return false;
        }
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        let mut shared = st.loads.clone();
        shared.seg.copy_from_slice(total);
        let (fac, _) = crowd_factors(&st.g, &st.net, &shared, &st.p);
        st.round += 1;
        if !st.uniform || fac != st.fac {
            st.uniform = false;
            st.fac = fac;
            set_crowding(&mut st.g, &st.fac);
            let (probability, loads) = self.pass(&st.g, &st.p, q);
            for (old, new) in st.probability.iter_mut().zip(&probability) {
                msa(old, new, st.round);
            }
            msa(&mut st.loads.seg, &loads.seg, st.round);
            msa(&mut st.loads.rn_board, &loads.rn_board, st.round);
            msa(&mut st.loads.rn_alight, &loads.rn_alight, st.round);
            msa(&mut st.loads.entry, &loads.entry, st.round);
            msa(&mut st.loads.exit, &loads.exit, st.round);
            let (trips, rail, walk) = self.mode_totals(q, &st.probability);
            st.trips = trips;
            st.rail = rail;
            st.walk = walk;
        }
        st.ms = now_ms() - t;
        self.per[q] = Some(st);
        true
    }
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| {
                vec![
                    s.round as f64,
                    s.trips,
                    s.rail,
                    s.walk,
                    s.ms,
                    self.pairs.len() as f64,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            })
    }
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.seg.clone())
    }
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.rn_board.clone())
    }
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.rn_alight.clone())
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        (0..st.loads.seg.len())
            .map(|r| {
                let l = st.g.rn_line[r] as usize;
                let cap = 60.0 / st.net.line_headway_min[l][q] * st.net.line_crush[l];
                if cap > 0.0 {
                    st.loads.seg[r] / st.p.period_hours * st.p.peak_hour_factor / cap
                } else {
                    0.0
                }
            })
            .collect()
    }
    pub fn solved_mask(&self) -> u32 {
        (0..5)
            .filter(|&q| self.per[q].is_some())
            .fold(0, |m, q| m | 1 << q)
    }
    /// Six mode-major cell blocks, home and activity ends, trip legs over solved periods.
    pub fn cell_modes(&self) -> Vec<f32> {
        let n = self.engine.city.len();
        let mut out = vec![0.0; 6 * n];
        for k in 0..self.destinations.len() {
            let Some((kind, c, d, w, pw)) = self.edge(k) else {
                continue;
            };
            let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
            let mut total = 0.0;
            let mut rail = 0.0;
            for q in 0..5 {
                if let Some(st) = &self.per[q] {
                    total += w * (OUTWARD[kind][q] + RETURN[kind][q]);
                    if pi != NONE {
                        let pr = st.probability[pi as usize];
                        let scale = self.edge_motorized_scale(pi, pw);
                        rail += w * scale * (OUTWARD[kind][q] * pr[0] + RETURN[kind][q] * pr[1]);
                    }
                }
            }
            let walk = total * pw;
            let drive = (total - walk - rail).max(0.0);
            for (mode, v) in [rail, walk, drive].into_iter().enumerate() {
                out[mode * n + c] += v;
                out[(mode + 3) * n + d] += v;
            }
        }
        out
    }
    /// Each enabled purpose: 30 timing values + 3 far-cell mode blocks. All legs; no cut.
    pub fn selection(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        let n = self.engine.city.len();
        let stride = 30 + 3 * n;
        let mut out = vec![0.0; self.totals.len() * stride];
        let mut selected = vec![false; n];
        for &c in cells {
            if (c as usize) < n {
                selected[c as usize] = true;
            }
        }
        for k in 0..self.destinations.len() {
            let Some((kind, c, d, w, pw)) = self.edge(k) else {
                continue;
            };
            let (near, far) = if end == 0 { (c, d) } else { (d, c) };
            if !selected[near] {
                continue;
            }
            let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
            let base = kind * stride;
            for q in 0..5 {
                if let Some(st) = &self.per[q] {
                    for direction in 0..2 {
                        let share = if direction == 0 {
                            OUTWARD[kind][q]
                        } else {
                            RETURN[kind][q]
                        };
                        let pr = if pi == NONE {
                            0.0
                        } else {
                            st.probability[pi as usize][direction] * self.edge_motorized_scale(pi, pw)
                        };
                        let rail = w * share * pr;
                        let walk = w * share * pw;
                        let drive = (w * share - walk - rail).max(0.0);
                        for (m, v) in [rail, walk, drive]
                            .into_iter()
                            .enumerate()
                        {
                            out[base + direction * 15 + m * 5 + q] += v;
                            out[base + 30 + m * n + far] += v;
                        }
                    }
                }
            }
        }
        out
    }
}
