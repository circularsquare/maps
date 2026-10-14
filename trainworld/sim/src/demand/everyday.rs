//! NYC-first home-based outings. Job counts are attraction weights, never customer capacities.
//! Coarse destination gravity is integrated by deterministic stratified cell samples. Endpoint
//! distances, the 150 m exclusion, walking and regional overlap use the actual sampled cells.
use super::{
    api::{scale_triples, DemandApi, PEAK_HOUR_FACTOR},
    clock::now_ms,
    days::{everyday, load_factor, share_out, share_ret, WEEKDAY},
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
/// Purpose kinds, in this order everywhere (`info`, `part_loads`, `selection`; the app's
/// `EVERYDAY_KEYS`): errands, social, school, university, airport (T-142), medical (T-142).
/// Kinds 0, 1 and 5 are outings per resident; 2 and 3 attendance per enrolled student (samples
/// drawn from the site); 4 the airport's own daily ground trips (`airport_trips`), its homes and
/// visitors' ends drawn from each origin cell but normalised per airport.
const RATE: [f32; 6] = [0.3, 0.2, 0.9, 0.6, 1.0, 0.04];
const DECAY: [f32; 6] = [3000.0, 6000.0, 2500.0, 6000.0, 20000.0, 5000.0];
pub const OUTWARD: [[f32; 5]; 6] = [
    [0.10, 0.55, 0.30, 0.04, 0.01],
    [0.05, 0.35, 0.42, 0.17, 0.01],
    [0.94, 0.04, 0.015, 0.004, 0.001],
    [0.65, 0.27, 0.07, 0.009, 0.001],
    // to the airport: early flights put some trips before 6
    [0.30, 0.35, 0.22, 0.06, 0.07],
    // to appointments, and visitors
    [0.30, 0.55, 0.12, 0.02, 0.01],
];
pub const RETURN: [[f32; 5]; 6] = [
    [0.06, 0.49, 0.36, 0.08, 0.01],
    [0.02, 0.25, 0.43, 0.27, 0.03],
    [0.01, 0.82, 0.15, 0.018, 0.002],
    [0.04, 0.42, 0.42, 0.11, 0.01],
    // from the airport: arrivals run late into the evening
    [0.12, 0.33, 0.28, 0.24, 0.03],
    [0.08, 0.60, 0.25, 0.05, 0.02],
];
const AIRPORT: usize = 4;
/// Destination samples per cell, by purpose (T-142). Airports are a handful of destinations, so
/// four samples spread a cell's trips over them; medical visits go to many places but nobody
/// looks at one cell's medical trips alone, so eight. Sixteen for the four older purposes, whose
/// samples are unchanged. Fewer samples keep the opening cheap (each costs the same at opening
/// and in `set_network`).
const SAMPLES: [usize; 6] = [K, K, K, K, 2, 2];
/// No airport trip ends this close to a terminal (its own jobs are not its passengers' ends).
const AIRPORT_CLEAR_M: f32 = 1500.0;
/// T-196: the share of the social/leisure attraction that is parks (the pack's `park_draw`, OSM
/// parks by area and residents nearby), taken evenly from the population and leisure-jobs halves.
/// ATUS 2023: 7.6% of leisure outings (eating out, socialising, sport and recreation) end
/// outdoors away from home (notes/T-196.md). A pack without parks keeps 0.5 / 0.5 exactly.
const PARK_SHARE: f64 = 0.08;
/// School and university: samples are drawn from the activity site (homes), the site's
/// enrollment is the budget. Every other kind draws its samples from the home end.
fn site_anchored(kind: usize) -> bool {
    kind == 2 || kind == 3
}

fn unit(mut x: u32) -> f64 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x as f64 + 0.5) / 4294967296.0
}

/// What one purpose's destination sampling reads (T-172: the constructor's loops moved out of
/// line, so the browser optimises them; see `DecayTable::cdf`).
struct Sampler<'a> {
    kind: usize,
    /// samples per cell for this purpose (`SAMPLES`)
    k: usize,
    /// trips sent from each cell: residents, or a school's or university's enrollment
    budget: &'a [f32],
    h3: &'a [u64],
    x: &'a [f32],
    y: &'a [f32],
    /// the zones with any attraction, their mass and each one's cells' running mass
    zones: &'a [usize],
    mass: &'a [f64],
    cell_cdf: &'a [Vec<f64>],
    members: &'a [Vec<usize>],
}

impl Sampler<'_> {
    fn budget(&self, c: usize) -> f32 {
        self.budget.get(c).copied().unwrap_or(0.0)
    }
    /// `self.k` stratified destinations for each cell of one origin zone, from that zone's `cdf`
    /// over `zones`, into `out` (this purpose's slice of the destinations).
    #[inline(never)]
    fn sample(&self, cells: &[usize], cdf: &[f64], total: f64, out: &mut [u32]) {
        let n_k = self.k;
        for &c in cells {
            if self.budget(c) <= 0.0 {
                continue;
            }
            let seed = (self.h3[c] ^ (self.h3[c] >> 32)) as u32 ^ (self.kind as u32).wrapping_mul(0x9e3779b9);
            let offset = unit(seed);
            for k in 0..n_k {
                let target = (k as f64 + offset) / n_k as f64 * total;
                let j = cdf.partition_point(|&v| v <= target).min(self.zones.len() - 1);
                let z = self.zones[j];
                let target = unit(seed.wrapping_add((k as u32 + 1).wrapping_mul(0x85ebca6b))) * self.mass[z];
                let j = self.cell_cdf[z].partition_point(|&v| v <= target).min(self.members[z].len() - 1);
                let d = self.members[z][j];
                let distance = ((self.x[c] - self.x[d]).powi(2) + (self.y[c] - self.y[d]).powi(2)).sqrt();
                // Excluded trips vanish, including from all-mode denominators. No replacement.
                if distance >= MIN_DISTANCE_M && local_share(distance) > 0.0 {
                    out[c * n_k + k] = d as u32;
                }
            }
        }
    }
}

/// A multiply-rotate hasher for `set_network`'s pair index (T-172): its keys are one u64 packed
/// from the purpose and two subzones, so the default SipHash's guard against crafted keys only
/// costs time. Pairs are numbered in first-seen order, so the hasher changes no result.
#[derive(Default)]
struct FastHasher(u64);
impl std::hash::Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn finish(&self) -> u64 {
        // hashbrown takes buckets from the low bits and tags from the top ones: fold the high
        // bits of the product down
        self.0 ^ (self.0 >> 29)
    }
}
type FastHash = std::hash::BuildHasherDefault<FastHasher>;

/// exp(-d / decay) between two zone centres by their grid offset (T-172). Zones of one square
/// (the same zone, or zones a region border splits, T-162) are 0.52 of a zone apart.
struct DecayTable {
    span: usize,
    v: Vec<f32>,
}

impl DecayTable {
    fn new(gx: &[i32], gy: &[i32], zone_m: f32, decay: f32) -> DecayTable {
        let range = |g: &[i32]| g.iter().max().zip(g.iter().min()).map_or(1, |(a, b)| (a - b) as usize + 1);
        let (wx, wy) = (range(gx), range(gy));
        let mut v = vec![0.0; wx * wy];
        for ax in 0..wx {
            for ay in 0..wy {
                let (dx, dy) = (ax as f32 * zone_m, ay as f32 * zone_m);
                let d = if ax == 0 && ay == 0 { 0.52 * zone_m } else { (dx * dx + dy * dy).sqrt() };
                v[ax * wy + ay] = (-d / decay).exp();
            }
        }
        DecayTable { span: wy, v }
    }
    /// The running sum of `mass` x decay from the zone at (`gx`, `gy`) to each listed zone, into
    /// `cdf`; returns the total. Kept out of line: a browser optimises a WebAssembly function only
    /// between calls (no on-stack replacement), so this one is fast after the first few origins
    /// while one long loop inside the constructor would stay in the baseline compiler throughout.
    #[inline(never)]
    fn cdf(&self, gx: i32, gy: i32, zx: &[i32], zy: &[i32], mass: &[f64], cdf: &mut Vec<f64>) -> f64 {
        cdf.clear();
        let mut total = 0.0;
        for k in 0..zx.len() {
            let o = (zx[k] - gx).unsigned_abs() as usize * self.span + (zy[k] - gy).unsigned_abs() as usize;
            total += mass[k] * self.v[o] as f64;
            cdf.push(total);
        }
        total
    }
}

/// T-142: each airport cell's daily trips divided by its draw, the decay-weighted origin weight
/// around it (sum over zones of their origin weight x decay to the airport's zone), as the mass
/// origin cells sample airports by. An origin cell then sends `origin x` the sum over airports of
/// mass x decay from its zone, and every airport receives exactly its trips (before the 150 m
/// exclusion and the 40-80 km taper, which apply to every purpose).
fn airport_draw(z: &Zones, members: &[Vec<usize>], origin: &[f32], trips: &[f32]) -> Vec<f32> {
    let table = DecayTable::new(&z.gx, &z.gy, z.zone_m, DECAY[AIRPORT]);
    let w: Vec<f64> = members.iter().map(|m| m.iter().map(|&c| origin[c] as f64).sum()).collect();
    let zones: Vec<usize> = (0..z.n).filter(|&k| w[k] > 0.0).collect();
    let zx: Vec<i32> = zones.iter().map(|&k| z.gx[k]).collect();
    let zy: Vec<i32> = zones.iter().map(|&k| z.gy[k]).collect();
    let zw: Vec<f64> = zones.iter().map(|&k| w[k]).collect();
    let mut draw = vec![None; z.n];
    let mut cdf = Vec::with_capacity(zones.len());
    let mut mass = vec![0.0f32; trips.len()];
    for (c, &t) in trips.iter().enumerate() {
        if t <= 0.0 {
            continue;
        }
        let a = z.cell_zone[c] as usize;
        let d = *draw[a].get_or_insert_with(|| table.cdf(z.gx[a], z.gy[a], &zx, &zy, &zw, &mut cdf));
        if d > 0.0 {
            mass[c] = (t as f64 / d) as f32;
        }
    }
    mass
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
            region: vec![], gravity: None,
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
            pop: vec![100.0,100.0,0.0], jobs: vec![0.0,0.0,100.0], region: vec![], gravity: None };
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
    fn purpose_parts_add_up_to_loads_and_cell_modes() {
        let (h, b) = university_fixture(4000.0);
        let mut a = EverydayApi::new(&h, &b).unwrap();
        network(&mut a, 4000.0, [12.0; 3]);
        let kinds = a.purpose_count() as usize;
        let check = |a: &EverydayApi| {
            for q in 0..5 {
                let parts = a.part_loads(q);
                let whole = [a.seg(q), a.board(q), a.alight(q)];
                let nr = whole[0].len();
                assert_eq!(parts.len(), kinds * 3 * nr);
                for (block, w) in whole.iter().enumerate() {
                    for r in 0..nr {
                        let sum: f32 = (0..kinds).map(|k| parts[(3 * k + block) * nr + r]).sum();
                        assert!((sum - w[r]).abs() <= 1e-4 * w[r].max(1.0), "period {q} block {block} node {r}: {sum} against {}", w[r]);
                    }
                }
            }
            let all = a.cell_modes();
            let mut sum = vec![0.0f32; all.len()];
            for k in 0..kinds {
                for (s, v) in sum.iter_mut().zip(a.cell_modes_of(k as u32)) {
                    *s += v;
                }
            }
            for (s, v) in sum.iter().zip(&all) {
                assert!((s - v).abs() <= 1e-4 * v.max(1.0));
            }
            assert!(a.cell_modes_of(kinds as u32).iter().all(|&v| v == 0.0));
        };
        for q in 0..5 {
            assert!(a.solve(q));
        }
        check(&a);
        // school and university riders are both on the line: two purposes carry load
        let parts = a.part_loads(0);
        let nr = a.seg(0).len();
        assert!((0..kinds).filter(|&k| parts[3 * k * nr..(3 * k + 1) * nr].iter().sum::<f32>() > 0.0).count() >= 2);
        for q in 0..5 {
            let shared: Vec<_> = a.seg(q).iter().map(|v| v * 1000.0).collect();
            assert!(a.crowd_with_external(q, &shared));
        }
        check(&a);
    }

    /// T-158: the real New York pack's everyday layer on the real network (nyc_real_track.json),
    /// one period, free flow and a crowding round, with timings; the per-purpose loads must add up
    /// to the total. `cargo test --release --features demand everyday_real_network -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn everyday_real_network() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
        let t = now_ms();
        let mut a = EverydayApi::new(&h, &b).unwrap();
        println!("open {:.0} ms, {} purposes", now_ms() - t, a.purpose_count());
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/data/work/realnet/nyc_real_track.json")).unwrap()).unwrap();
        let d = &j["demand"];
        let f32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let u32s = |k: &str| d[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<u32>>();
        let (st_xy, line_n, stops, times, tph, cars) = (f32s("st_xy"), u32s("line_n"), u32s("stops"), f32s("times"), f32s("tph"), f32s("cars"));
        let mut hop_km = vec![];
        let mut off = 0usize;
        for &n in &line_n {
            let n = n as usize;
            for dir in 0..2 {
                for i in 0..n {
                    let k = if dir == 0 { off + i } else { off + n - 1 - i };
                    hop_km.push(if i + 1 < n {
                        let next = if dir == 0 { k + 1 } else { k - 1 };
                        let (s0, s1) = (stops[k] as usize, stops[next] as usize);
                        (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                    } else {
                        0.0
                    });
                }
            }
            off += n;
        }
        let ms = a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        println!("set_network {ms:.0} ms, {} pairs, {} route nodes", a.pairs.len(), 2 * stops.len());
        for q in [0usize, 1] {
            let t = now_ms();
            a.solve(q);
            let free = now_ms() - t;
            let (by_purpose, combined) = ASSIGN_MS.with(|c| c.get());
            let total: Vec<f32> = a.seg(q).iter().map(|v| v * 3.0).collect();
            let t = now_ms();
            a.crowd_with_external(q, &total);
            let round = now_ms() - t;
            println!("period {q}: free flow {free:.0} ms (assignment by purpose {by_purpose:.0} ms, one summed assignment would be {combined:.0} ms), crowding round {round:.0} ms, rail {:.0}", a.summary(q)[2]);
            let parts = a.part_loads(q);
            let (seg, board, alight) = (a.seg(q), a.board(q), a.alight(q));
            let nr = seg.len();
            assert_eq!(parts.len(), 3 * nr * a.purpose_count() as usize);
            for (block, whole) in [&seg, &board, &alight].into_iter().enumerate() {
                let mut worst = 0.0f32;
                for r in 0..nr {
                    let sum: f32 = (0..a.purpose_count() as usize).map(|k| parts[(3 * k + block) * nr + r]).sum();
                    worst = worst.max((sum - whole[r]).abs() / whole[r].max(1.0));
                }
                println!("  block {block}: parts against total, worst relative difference {worst:.2e}");
                assert!(worst < 1e-3);
            }
            let by_kind: Vec<f64> = (0..a.purpose_count() as usize).map(|k| parts[3 * k * nr..(3 * k + 1) * nr].iter().map(|&v| v as f64).sum()).collect();
            println!("  rider-segments by purpose {by_kind:.0?}");
        }
        // T-285: a selection's far end, visiting only its samples, against visiting them all
        let n = a.engine.city.len();
        let pop = &a.engine.city.pop;
        let busy: Vec<u32> = { let mut c: Vec<u32> = (0..n as u32).collect(); c.sort_by(|&x, &y| pop[y as usize].total_cmp(&pop[x as usize])); c };
        let picks: Vec<Vec<u32>> = vec![busy[..1].to_vec(), busy[100..107].to_vec(), busy[..2000].iter().step_by(3).copied().collect(), (0..n as u32).step_by(997).collect()];
        for cells in &picks {
            for end in 0..2 {
                for day in 0..2 {
                    let t = now_ms();
                    let old = selection_reference(&a, end, cells, day);
                    let old_ms = now_ms() - t;
                    let t = now_ms();
                    let new = a.selection_day(end, cells, day);
                    let new_ms = now_ms() - t;
                    let t = now_ms();
                    let sparse = a.selection_sparse_day(end, cells, day);
                    let sparse_ms = now_ms() - t;
                    assert!(old == new, "{} cells, end {end}, day {day}", cells.len());
                    println!("selection of {} cells, end {end}, day {day}: all samples {old_ms:.1} ms, its own {new_ms:.1} ms dense, {sparse_ms:.1} ms packed ({} numbers against {})", cells.len(), sparse.len(), new.len());
                }
            }
        }
    }

    /// `selection_day` as it was before T-285: every sample weighed, then the selected kept.
    fn selection_reference(a: &EverydayApi, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        let day = day as usize;
        let n = a.engine.city.len();
        let stride = 30 + 3 * n;
        let mut out = vec![0.0; a.totals.len() * stride];
        let mut selected = vec![false; n];
        for &c in cells {
            if (c as usize) < n {
                selected[c as usize] = true;
            }
        }
        for k in 0..a.destinations.len() {
            let Some((kind, c, d, w, pw)) = a.edge(k) else { continue };
            let (near, far) = if end == 0 { (c, d) } else { (d, c) };
            if !selected[near] {
                continue;
            }
            let pi = a.edge_pair.get(k).copied().unwrap_or(NONE);
            let base = kind * stride;
            for q in 0..5 {
                if let Some(st) = &a.per[q] {
                    for direction in 0..2 {
                        let share = if direction == 0 { share_out(day, everyday(kind), q) } else { share_ret(day, everyday(kind), q) };
                        let pr = if pi == NONE { 0.0 } else { st.probability[pi as usize][direction] * a.edge_motorized_scale(pi, pw) };
                        let rail = w * share * pr;
                        let walk = w * share * pw;
                        let drive = (w * share - walk - rail).max(0.0);
                        for (m, v) in [rail, walk, drive].into_iter().enumerate() {
                            out[base + direction * 15 + m * 5 + q] += v;
                            out[base + 30 + m * n + far] += v;
                        }
                    }
                }
            }
        }
        out
    }

    /// T-285 on the small fixtures: the selection reads only its samples, the same numbers.
    #[test]
    fn selection_visits_only_its_samples() {
        for (h, b) in [fixture(4000.0), school_fixture(4000.0)] {
            let mut a = EverydayApi::new(&h, &b).unwrap();
            network(&mut a, 4000.0, [12.0; 3]);
            for q in 0..5 {
                assert!(a.solve(q));
            }
            let n = a.engine.city.len() as u32;
            for cells in [vec![0], vec![1], (0..n).collect::<Vec<_>>(), vec![1, 1, 0, n + 5]] {
                for end in 0..2 {
                    for day in 0..2 {
                        assert_eq!(selection_reference(&a, end, &cells, day), a.selection_day(end, &cells, day));
                    }
                }
            }
        }
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

    /// T-142: one home cell, facilities at `x[1..]` with the given airport trips and medical sizes.
    fn facility_fixture(x: &[f32], airport: Option<&[f32]>, medical: Option<&[f32]>) -> (String, Vec<u8>) {
        let n = x.len();
        let one = |v: f32| (0..n).map(|c| if c == 0 { v } else { 0.0 }).collect::<Vec<f32>>();
        let city = pack::City {
            name: "facilities".into(),
            h3: (1..=n as u64).collect(),
            x: x.to_vec(),
            y: vec![0.0; n],
            pop: one(100.0),
            jobs: one(100.0),
            region: vec![],
            gravity: None,
        };
        let (h, mut b) = pack::write(&city);
        let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
        let mut arrays: Vec<(&str, Vec<f32>)> = vec![("errand_jobs", one(100.0)), ("leisure_jobs", one(100.0))];
        if let Some(a) = airport {
            arrays.push(("airport_trips", a.to_vec()));
        }
        if let Some(m) = medical {
            arrays.push(("medical_visits", m.to_vec()));
        }
        for (name, values) in arrays {
            let offset = b.len();
            b.extend(values.into_iter().flat_map(|v| v.to_le_bytes()));
            h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":n,"offset":offset}));
        }
        (h.to_string(), b)
    }

    #[test]
    fn airport_and_medical_purposes_conserve_and_stay_optional() {
        let x = [0.0, 4000.0];
        let (h, b) = facility_fixture(&x, Some(&[0.0, 50.0]), Some(&[0.0, 10.0]));
        let mut a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.purpose_count(), 6);
        assert_eq!((a.totals[2], a.totals[3]), (0.0, 0.0), "no school or university inputs: empty slots");
        assert!((a.totals[AIRPORT] - 50.0).abs() < 1e-3, "the airport keeps its own trips: {}", a.totals[AIRPORT]);
        assert!((a.totals[5] - 100.0 * 0.04).abs() < 1e-4, "medical: a rate per resident");
        for k in 0..a.destinations.len() {
            if let Some((kind, home, site, _, _)) = a.edge(k) {
                if kind >= AIRPORT {
                    assert_eq!((home, site), (0, 1));
                }
            }
        }
        // the layers before are untouched by the new arrays
        let (oh, ob) = facility_fixture(&x, None, None);
        let old = EverydayApi::new(&oh, &ob).unwrap();
        assert_eq!(old.purpose_count(), 2);
        assert_eq!(&a.info()[..6], &old.info()[..]);
        assert_eq!(&a.destinations[..old.destinations.len()], &old.destinations[..]);
        // timing and selections: legs to and from the airport, outward and return
        network(&mut a, 4000.0, [12.0; 3]);
        for q in 0..5 {
            assert!(a.solve(q));
        }
        let selected = a.selection(1, &[1]);
        let stride = 30 + 3 * 2;
        let air = &selected[AIRPORT * stride..(AIRPORT + 1) * stride];
        assert!((air[..15].iter().sum::<f32>() - 50.0).abs() < 1e-3);
        assert!((air[15..30].iter().sum::<f32>() - 50.0).abs() < 1e-3);
        let early = (0..3).map(|m| air[m * 5 + 4]).sum::<f32>();
        assert!((early / 50.0 - OUTWARD[AIRPORT][4]).abs() < 1e-5);
        let modes = a.cell_modes();
        let all = modes[..6].iter().sum::<f32>() as f64;
        assert!((all - 2.0 * a.totals.iter().sum::<f64>()).abs() < 1e-2);
        // optional: airport without medical, and a pack with no airport (Springfield) keeps the slot
        let (h, b) = facility_fixture(&x, Some(&[0.0, 50.0]), None);
        assert_eq!(EverydayApi::new(&h, &b).unwrap().purpose_count(), 5);
        let (h, b) = facility_fixture(&x, Some(&[0.0, 0.0]), Some(&[0.0, 10.0]));
        let a = EverydayApi::new(&h, &b).unwrap();
        assert_eq!(a.purpose_count(), 6);
        assert_eq!(a.totals[AIRPORT], 0.0);
        assert!((a.totals[5] - 4.0).abs() < 1e-4);
        // two airports share one home cell's trips by their own totals (to one sample's share)
        let (h, b) = facility_fixture(&[0.0, 3000.0, 10000.0], Some(&[0.0, 30.0, 20.0]), None);
        let a = EverydayApi::new(&h, &b).unwrap();
        assert!((a.totals[AIRPORT] - 50.0).abs() < 1e-3);
        let mut to = [0.0f64; 3];
        for k in 0..a.destinations.len() {
            if let Some((AIRPORT, _, site, w, _)) = a.edge(k) {
                to[site] += w as f64;
            }
        }
        let one = 50.0 / SAMPLES[AIRPORT] as f64 + 1e-3;
        assert!((to[1] - 30.0).abs() <= one && (to[2] - 20.0).abs() <= one, "{to:?}");
    }

    /// A pack's header and bin with one more f32 cell array.
    fn with_array(h: &str, mut b: Vec<u8>, name: &str, values: &[f32]) -> (String, Vec<u8>) {
        let mut h: serde_json::Value = serde_json::from_str(h).unwrap();
        let offset = b.len();
        b.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":values.len(),"offset":offset}));
        (h.to_string(), b)
    }

    /// T-196: parks take PARK_SHARE of the social attraction and nothing else; a pack without
    /// parks (or with an empty array) is unchanged.
    #[test]
    fn parks_share_the_social_destinations() {
        let x = [0.0, 3000.0];
        let (h0, b0) = facility_fixture(&x, None, None);
        let old = EverydayApi::new(&h0, &b0).unwrap();
        let (h, b) = with_array(&h0, b0.clone(), "park_draw", &[0.0, 0.0]);
        assert_eq!(EverydayApi::new(&h, &b).unwrap().destinations, old.destinations);
        let (h, b) = with_array(&h0, b0, "park_draw", &[0.0, 5.0]);
        let a = EverydayApi::new(&h, &b).unwrap();
        // errands unchanged; social outings that reach the park are the only ones over 150 m
        assert_eq!(&a.destinations[..a.start[1]], &old.destinations[..old.start[1]]);
        assert_eq!(old.totals[1], 0.0, "home, jobs and leisure all in the one cell: under 150 m");
        assert!(a.totals[1] > 0.0 && a.totals[1] < 0.25 * 100.0 * RATE[1] as f64, "{}", a.totals[1]);
        for k in a.start[1]..a.start[2] {
            if let Some((kind, home, site, _, _)) = a.edge(k) {
                assert_eq!((kind, home, site), (1, 0, 1));
            }
        }
    }

    /// T-196: each city's pack before parks (`data/work/pre-T196`) against now. Every other
    /// purpose's destinations must be identical; prints social outings per resident, walk share
    /// and mean length before and after, the share of social outings that go to parks, and the
    /// opening time. `cargo test --release --features demand parks_before_after -- --ignored
    /// --nocapture`
    #[test]
    #[ignore]
    fn parks_before_after() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
        for city in ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf"] {
            let open = |path: &str| {
                let (h, b) = pack::load_raw(path).unwrap();
                let mut best = f64::INFINITY;
                let mut api = None;
                for _ in 0..3 {
                    let t = now_ms();
                    let a = EverydayApi::new(&h, &b).unwrap();
                    best = best.min(now_ms() - t);
                    api = Some(a);
                }
                (api.unwrap(), best, h, b)
            };
            let (old, old_ms, _, _) = open(&format!("{root}/work/pre-T196/{city}.json"));
            let (new, new_ms, h, b) = open(&format!("{root}/packs/{city}.json"));
            assert_eq!(&new.destinations[..new.start[1]], &old.destinations[..old.start[1]], "{city}");
            assert_eq!(&new.destinations[new.start[2]..], &old.destinations[old.start[2]..], "{city}");
            let n = new.engine.city.len();
            let w = |name: &str| pack::cell_weights(&h, &b, name, n).unwrap().unwrap();
            let (pop, leisure, park) = (w("pop"), w("leisure_jobs"), w("park_draw"));
            let s = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
            let (tp, tl, tk, residents) = (s(&pop), s(&leisure), s(&park), s(&pop));
            let half = 0.5 * (1.0 - PARK_SHARE);
            // the park part of each social outing's destination cell, weighted by the outing
            let (mut all, mut to_park) = (0.0f64, 0.0f64);
            for k in new.start[1]..new.start[2] {
                if let Some((_, _, d, wt, _)) = new.edge(k) {
                    let pk = PARK_SHARE * park[d] as f64 / tk;
                    let tot = half * pop[d] as f64 / tp + half * leisure[d] as f64 / tl + pk;
                    all += wt as f64;
                    to_park += wt as f64 * if tot > 0.0 { pk / tot } else { 0.0 };
                }
            }
            let (oi, ni) = (old.info(), new.info());
            println!(
                "{city}: social {:.4} -> {:.4} outings per resident ({:.0} -> {:.0}), walk {:.1}% -> {:.1}%, mean {:.2} -> {:.2} km; to parks {:.1}% of social outings; open {old_ms:.0} -> {new_ms:.0} ms",
                oi[3] / residents, ni[3] / residents, oi[3], ni[3], 100.0 * oi[4], 100.0 * ni[4], oi[5], ni[5], 100.0 * to_park / all.max(1e-9)
            );
        }
    }

    /// T-142's airports and medical visits: each city's pack before them (`data/work/pre-T142b`)
    /// against now. The four older purposes' destinations and totals must be identical; prints
    /// the opening time of both (best of three) and the airport per-terminal totals' worst
    /// difference from the pack's input. `cargo test --release --features demand
    /// airport_medical_before_after -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn airport_medical_before_after() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
        for city in ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf"] {
            let open = |path: &str| {
                let (h, b) = pack::load_raw(path).unwrap();
                let mut best = f64::INFINITY;
                let mut api = None;
                for _ in 0..3 {
                    let t = now_ms();
                    let a = EverydayApi::new(&h, &b).unwrap();
                    best = best.min(now_ms() - t);
                    api = Some(a);
                }
                (api.unwrap(), best, h, b)
            };
            let (old, old_ms, _, _) = open(&format!("{root}/work/pre-T142b/{city}.json"));
            let (new, new_ms, h, b) = open(&format!("{root}/packs/{city}.json"));
            assert_eq!(old.purpose_count(), 4, "{city}");
            assert_eq!(new.purpose_count(), 6, "{city}");
            assert_eq!(&new.destinations[..old.destinations.len()], &old.destinations[..], "{city}");
            assert_eq!(&new.info()[..12], &old.info()[..], "{city}");
            // per airport cell: what the samples send there against the pack's input
            let n = new.engine.city.len();
            let input = pack::cell_weights(&h, &b, "airport_trips", n).unwrap().unwrap();
            let mut got = vec![0.0f64; n];
            for k in new.start[AIRPORT]..new.start[AIRPORT + 1] {
                if let Some((_, c, d, _, _)) = new.edge(k) {
                    // before the 40-80 km taper, as the input is (samples under 150 m or past
                    // 80 km are gone already)
                    got[d] += new.airport_origin[c] as f64 * new.airport_scale[new.engine.z.cell_zone[c] as usize] / SAMPLES[AIRPORT] as f64;
                }
            }
            // terminals of one airport (within 5 km of each other) together
            let sites: Vec<usize> = (0..n).filter(|&c| input[c] > 0.0).collect();
            let mut group: Vec<usize> = (0..sites.len()).collect();
            for i in 0..sites.len() {
                for j in 0..i {
                    if new.distance(sites[i], sites[j]) < 5000.0 {
                        group[i] = group[j];
                        break;
                    }
                }
            }
            let mut worst = 0.0f64;
            for g in 0..sites.len() {
                let members: Vec<usize> = (0..sites.len()).filter(|&i| group[i] == g).map(|i| sites[i]).collect();
                let want: f64 = members.iter().map(|&c| input[c] as f64).sum();
                let have: f64 = members.iter().map(|&c| got[c]).sum();
                // airports with over 1,000 outings a day: a small one at the pack's edge (Portsmouth,
                // Montauk) loses its ends past 80 km, which the samples drop
                if want > 1000.0 {
                    worst = worst.max((have - want).abs() / want);
                    if (have - want).abs() > 0.02 * want {
                        println!("  {city}: airport at cell {} wants {want:.0}, samples give {have:.0}", members[0]);
                    }
                }
            }
            let info = new.info();
            println!(
                "{city}: open {old_ms:.0} ms before, {new_ms:.0} ms now; airport {:.0} outings ({:.1}% walk, {:.1} km), each airport over 1,000 a day within {:.2}% of its input before the taper; medical {:.0} ({:.1}% walk, {:.1} km); destination samples {:.1} MB",
                info[12], 100.0 * info[13], info[14], 100.0 * worst, info[15], 100.0 * info[16], info[17],
                new.destinations.len() as f64 * 4.0 / 1e6
            );
        }
    }

    /// T-142: every shipped city pack's everyday layer opens; prints its opening time and, per
    /// purpose, outings a day, walk share and mean km against the pack's residents (the per-capita
    /// check in notes/T-142.md). `cargo test --release --features demand every_pack_everyday --
    /// --ignored --nocapture`
    #[test]
    #[ignore]
    fn every_pack_everyday() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/packs");
        for city in ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf", "tor"] {
            let (h, b) = pack::load_raw(&format!("{root}/{city}.json")).unwrap();
            let t = now_ms();
            let a = EverydayApi::new(&h, &b).unwrap();
            let ms = now_ms() - t;
            // residents, not the engine's `pop` (the commute home end)
            let pop: f64 = pack::cell_weights(&h, &b, "pop", a.engine.city.len()).unwrap().unwrap().iter().map(|&v| v as f64).sum();
            let info = a.info();
            let per: Vec<String> = info
                .chunks(3)
                .map(|v| format!("{:.0} ({:.3}/res, walk {:.1}%, {:.2} km)", v[0], v[0] / pop, 100.0 * v[1], v[2]))
                .collect();
            println!("{city}: open {ms:.0} ms, {} cells, {} zones, residents {pop:.0}; {}", a.engine.city.len(), a.engine.z.n, per.join("; "));
            // the airports' own trips against what the 150 m exclusion and 40-80 km taper keep
            let input: f64 = pack::cell_weights(&h, &b, "airport_trips", a.engine.city.len()).unwrap().map_or(0.0, |v| v.iter().map(|&x| x as f64).sum());
            println!("  {city} airport: pack {input:.0} outings a day, kept {:.0}", a.totals.get(AIRPORT).copied().unwrap_or(0.0));
            // six since T-142's airports and medical visits (Springfield's airport slot is empty)
            assert_eq!(a.purpose_count(), 6, "{city}");
        }
    }
}
/// 0..n in pieces of 16k, for loops over every destination sample (3.2M on New York): each piece
/// is one call of an out-of-line function, so the browser optimises it after the first few
/// pieces (T-172; WebAssembly has no on-stack replacement).
fn chunks(n: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    (0..n).step_by(1 << 14).map(move |a| a..(a + (1 << 14)).min(n))
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
/// One purpose's share of a period's loads (T-158), per route node. The purposes' parts add up
/// to the period's `loads` (assignment is linear in the trips for fixed strategies); only the
/// display filter reads them, crowding and mode choice always use the total.
#[derive(Clone)]
struct Part {
    seg: Vec<f32>,
    rn_board: Vec<f32>,
    rn_alight: Vec<f32>,
}

impl Part {
    fn of(l: &Loads) -> Part {
        Part { seg: l.seg.clone(), rn_board: l.rn_board.clone(), rn_alight: l.rn_alight.clone() }
    }
}

/// `total += l`, every field.
fn add_loads(total: &mut Loads, l: &Loads) {
    for (a, b) in [
        (&mut total.seg, &l.seg),
        (&mut total.board, &l.board),
        (&mut total.alight, &l.alight),
        (&mut total.rn_board, &l.rn_board),
        (&mut total.rn_alight, &l.rn_alight),
        (&mut total.entry, &l.entry),
        (&mut total.exit, &l.exit),
    ] {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }
    total.pax_min += l.pax_min;
}

#[cfg(test)]
thread_local! {
    /// Test probe (T-158): ms spent assigning by purpose in the last pass, and what one assignment
    /// of the summed trips would have taken.
    static ASSIGN_MS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((0.0, 0.0)) };
}

struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    loads: Loads,
    /// per purpose kind, as `totals`
    parts: Vec<Part>,
    probability: Vec<[f32; 2]>,
    round: usize,
    ms: f64,
    fac: Vec<f32>,
    uniform: bool,
    trips: f64,
    rail: f64,
    walk: f64,
    /// Transfers platform to platform, every purpose together (T-176), averaged like the loads.
    xi: XferIndex,
    xfer: Vec<f32>,
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
    /// trip kilometres by purpose, for `info` (summed once at opening, T-172)
    km: Vec<f64>,
    /// Optional enrollment budgets: school generation samples homes FROM each activity site.
    school_enrollment: Vec<f32>,
    university_enrollment: Vec<f32>,
    /// T-142, airports: each cell's share of the ground trips' home and visitor ends (sums to
    /// 1), and per origin zone the airports' normalised draw from there (`airport_draw`), so a
    /// cell's airport trips are `airport_origin[c] x airport_scale[zone]`. Empty without airports.
    airport_origin: Vec<f32>,
    airport_scale: Vec<f64>,
    /// Where each purpose's samples start in `destinations` (one more entry than purposes): a
    /// purpose holds cells x `SAMPLES[kind]`, purpose by purpose.
    start: Vec<usize>,
}

impl EverydayApi {
    /// The parameters every period starts from (checks; to change before `set_network`).
    pub fn params_mut(&mut self) -> &mut Params {
        self.engine.params_mut()
    }
}

impl EverydayApi {
    /// Trips, walking trips and trip kilometres by purpose, over destination samples `r` (see
    /// `chunks`).
    #[inline(never)]
    fn add_totals(&mut self, r: std::ops::Range<usize>) {
        for k in r {
            if let Some((kind, c, d, w, pw)) = self.edge(k) {
                self.totals[kind] += w as f64;
                self.walks[kind] += (w * pw) as f64;
                self.km[kind] += self.distance(c, d) as f64 * w as f64 / 1000.0;
            }
        }
    }
    /// Join destination samples `r` to the subzone pairs they fall in (for `set_network`).
    #[inline(never)]
    fn index_edges(&mut self, r: std::ops::Range<usize>, index: &mut HashMap<u64, u32, FastHash>) {
        for k in r {
            let Some((kind, c, d)) = self.ends(k) else {
                continue;
            };
            let Some(sz) = &self.engine.sz else { return };
            let (u, v) = (sz.cell[c], sz.cell[d]);
            // both ends in a subzone first: the weights cost an exp and a square root
            if u == NONE || v == NONE {
                continue;
            }
            let distance = self.distance(c, d);
            let (w, pw) = self.weight(kind, c, d, distance);
            let key = (kind as u64) << 58 | (u as u64) << 29 | v as u64;
            let pairs = &mut self.pairs;
            let i = *index.entry(key).or_insert_with(|| {
                pairs.push(Pair { u, v, kind, trips: 0.0, walk: 0.0, distance: 0.0 });
                (pairs.len() - 1) as u32
            });
            self.edge_pair[k] = i;
            let pair = &mut self.pairs[i as usize];
            pair.trips += w;
            pair.walk += w * pw;
            pair.distance += w * distance;
        }
    }
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
    /// Destination sample `k`'s purpose, home cell and activity cell.
    fn ends(&self, k: usize) -> Option<(usize, usize, usize)> {
        let d = self.destinations[k];
        if d == NONE {
            return None;
        }
        // at most six purposes: a short scan
        let mut kind = 0;
        while self.start[kind + 1] <= k {
            kind += 1;
        }
        let fixed = (k - self.start[kind]) / SAMPLES[kind];
        Some(if site_anchored(kind) { (kind, d as usize, fixed) } else { (kind, fixed, d as usize) })
    }
    /// A sample's trips and walking share, from its ends and their distance.
    fn weight(&self, kind: usize, c: usize, d: usize, distance: f32) -> (f32, f32) {
        let budget = match kind {
            2 => self.school_enrollment[d],
            3 => self.university_enrollment[d],
            AIRPORT => (self.airport_origin[c] as f64 * self.airport_scale[self.engine.z.cell_zone[c] as usize]) as f32,
            _ => self.engine.city.pop[c],
        };
        let weight = budget * RATE[kind] / SAMPLES[kind] as f32 * local_share(distance);
        // T-275: the walk goes around water between the two cells' zones
        let walk_f = match &self.engine.gr.wroutes {
            Some(r) => r.walk_factor(self.engine.z.cell_zone[c] as usize, self.engine.z.cell_zone[d] as usize),
            None => 1.0,
        };
        (weight, self.engine.base.walk_share(distance * walk_f / 1000.0))
    }
    fn edge(&self, k: usize) -> Option<(usize, usize, usize, f32, f32)> {
        let (kind, c, d) = self.ends(k)?;
        let (w, pw) = self.weight(kind, c, d, self.distance(c, d));
        Some((kind, c, d, w, pw))
    }
    /// Also returns the transfers, every purpose together, laid out by `ix` (T-176).
    fn pass(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> (Vec<[f32; 2]>, Loads, Vec<Part>, Vec<f32>) {
        let mut probability = vec![[0.0; 2]; self.pairs.len()];
        let s = g.n_st;
        let kinds = self.totals.len();
        let mut xfer = vec![0.0; ix.len];
        let Some(sz) = &self.engine.sz else {
            let zero = Loads::zero(s, g.n - s);
            return (probability, zero.clone(), vec![Part::of(&zero); kinds], xfer);
        };
        // Outward AND homeward destinations: every station either end of a pair can reach (T-142:
        // not every station, so a city with no pairs near the network, open in a worker beside
        // the one being built in, searches nothing; each destination's search is independent,
        // so this changes no result).
        let mut want = vec![false; s];
        let mut used = vec![false; sz.n];
        for pair in &self.pairs {
            used[pair.u as usize] = true;
            used[pair.v as usize] = true;
        }
        for u in (0..sz.n).filter(|&u| used[u]) {
            for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                want[sz.acc_st[k] as usize] = true;
            }
        }
        if !want.contains(&true) {
            let zero = Loads::zero(s, g.n - s);
            return (probability, zero.clone(), vec![Part::of(&zero); kinds], xfer);
        }
        let (loads, parts) = with_skims(g, p, &want, |sk| {
            let er = station_exp(sk, p);
            // One station-pair matrix, assigned and cleared each time the pairs' purpose changes
            // (they are stored purpose by purpose), so each purpose's loads are kept apart for the
            // display filter (T-158) without a matrix per purpose. Their sum is the total.
            let mut od = vec![0.0; s * s];
            let mut parts: Vec<Loads> = (0..kinds).map(|_| Loads::zero(s, g.n - s)).collect();
            let mut od_kind = usize::MAX;
            let mut od_used = false;
            #[cfg(test)]
            let mut probe = (vec![0.0f32; s * s], 0.0f64);
            #[cfg_attr(not(test), allow(unused_mut))]
            let mut flush = |od: &mut Vec<f32>, kind: usize, parts: &mut Vec<Loads>| {
                #[cfg(test)]
                let t = now_ms();
                let l = assign_with_transfers(g, sk, od, ix, &mut xfer);
                #[cfg(test)]
                {
                    probe.1 += now_ms() - t;
                    for (a, b) in probe.0.iter_mut().zip(od.iter()) {
                        *a += b;
                    }
                }
                add_loads(&mut parts[kind], &l);
                od.fill(0.0);
            };
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
                if pair.kind != od_kind {
                    if od_used {
                        flush(&mut od, od_kind, &mut parts);
                    }
                    od_kind = pair.kind;
                    od_used = false;
                }
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
                    let road = p.drive_min_f(
                        pair.distance,
                        p.drive_factor(sz.zone[u] as usize, sz.zone[v] as usize),
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
                    od_used = true;
                    for ka in a.clone() {
                        for kb in b.clone() {
                            let ia = sz.acc_st[ka] as usize;
                            let ib = sz.acc_st[kb] as usize;
                            od[ia * s + ib] += trips * acc[ka] * acc[kb] * er[ia * s + ib];
                        }
                    }
                }
            }
            if od_used {
                flush(&mut od, od_kind, &mut parts);
            }
            #[cfg(test)]
            {
                let t = now_ms();
                let _ = assign(g, sk, &probe.0);
                ASSIGN_MS.with(|c| c.set((probe.1, now_ms() - t)));
            }
            let mut total = Loads::zero(s, g.n - s);
            for l in &parts {
                add_loads(&mut total, l);
            }
            (total, parts.iter().map(Part::of).collect::<Vec<_>>())
        });
        (probability, loads, parts, xfer)
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
        // T-196: parks, where the pack has them, take PARK_SHARE of the social attraction
        let parks = pack::cell_weights(header, bin, "park_draw", n)?.filter(|v| sum(v) > 0.0);
        let social: Vec<f32> = match &parks {
            None => pop.iter().zip(&leisure).map(|(&p, &l)| (0.5 * p as f64 / tp + 0.5 * l as f64 / tl) as f32).collect(),
            Some(park) => {
                let (tk, half) = (sum(park), 0.5 * (1.0 - PARK_SHARE));
                (0..n).map(|c| (half * pop[c] as f64 / tp + half * leisure[c] as f64 / tl + PARK_SHARE * park[c] as f64 / tk) as f32).collect()
            }
        };
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
        // T-142: airport ground trips per terminal cell and medical visit sizes per cell. A pack
        // without them (or with no airport, Springfield) leaves the purpose empty.
        let nonzero = |v: Option<Vec<f32>>| v.filter(|v| sum(v) > 0.0);
        let airport_trips = nonzero(pack::cell_weights(header, bin, "airport_trips", n)?);
        let medical = nonzero(pack::cell_weights(header, bin, "medical_visits", n)?);
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
        // Airports: the home and visitor ends are half residents, a quarter jobs (business
        // visitors) and a quarter leisure jobs (hotels, restaurants, attractions), each normalised.
        // Each airport keeps its own total: its trips are divided by its draw over every origin
        // (`airport_draw`), so an origin cell's sampled destinations pick airports by that
        // normalised mass times the decay, and the cell sends its weight times their sum.
        let (airport_origin, airport_mass) = match &airport_trips {
            Some(trips) => {
                let tj = sum(&engine.city.jobs).max(1e-9);
                let mut origin: Vec<f32> = (0..n)
                    .map(|c| (0.5 * engine.city.pop[c] as f64 / tp + 0.25 * engine.city.jobs[c] as f64 / tj + 0.25 * leisure[c] as f64 / tl) as f32)
                    .collect();
                // the airport's own jobs are not where its passengers come from (a terminal's
                // block holds tens of thousands of airport jobs): no ends within 1.5 km of one
                let sites: Vec<usize> = (0..n).filter(|&c| trips[c] > 0.0).collect();
                for (c, o) in origin.iter_mut().enumerate() {
                    let (x, y) = (engine.city.x[c], engine.city.y[c]);
                    if sites.iter().any(|&a| (engine.city.x[a] - x).powi(2) + (engine.city.y[a] - y).powi(2) < AIRPORT_CLEAR_M * AIRPORT_CLEAR_M) {
                        *o = 0.0;
                    }
                }
                let mass = airport_draw(&engine.z, &members, &origin, trips);
                (origin, Some(mass))
            }
            None => (vec![], None),
        };
        // Purpose indices stay stable: a missing purpose before a present one is an empty slot.
        let zeros = vec![0.0f32; n];
        let optional = [children.as_ref(), adults.as_ref(), airport_mass.as_ref(), medical.as_ref()];
        let mut weights = vec![&retail, &social];
        if let Some(last) = optional.iter().rposition(|o| o.is_some()) {
            weights.extend(optional[..=last].iter().map(|o| o.unwrap_or(&zeros)));
        }
        let kinds = weights.len();
        let mut airport_scale = vec![0.0f64; if airport_mass.is_some() { nz } else { 0 }];
        let mut start = vec![0usize];
        for kind in 0..kinds {
            start.push(start[kind] + n * SAMPLES[kind]);
        }
        let mut destinations = vec![NONE; start[kinds]];
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
            // T-172: zone centres sit on the zone grid, so the decay between two zones depends
            // only on their grid offset: one table per purpose instead of a square root and an
            // exp for every zone pair (5 s for New York's four purposes, now a fraction of that).
            let zx: Vec<i32> = zones.iter().map(|&z| engine.z.gx[z]).collect();
            let zy: Vec<i32> = zones.iter().map(|&z| engine.z.gy[z]).collect();
            let zmass: Vec<f64> = zones.iter().map(|&z| mass[z]).collect();
            let table = DecayTable::new(&engine.z.gx, &engine.z.gy, engine.z.zone_m, DECAY[kind]);
            let budget: &[f32] = match kind {
                2 => school_enrollment.as_deref().unwrap_or(&[]),
                3 => university_enrollment.as_deref().unwrap_or(&[]),
                AIRPORT => &airport_origin,
                _ => &engine.city.pop,
            };
            let s = Sampler {
                kind,
                k: SAMPLES[kind],
                budget,
                h3: &engine.city.h3,
                x: &engine.city.x,
                y: &engine.city.y,
                zones: &zones,
                mass: &mass,
                cell_cdf: &cell_cdf,
                members: &members,
            };
            let out = &mut destinations[start[kind]..start[kind + 1]];
            let mut cdf = Vec::with_capacity(zones.len());
            for oz in 0..nz {
                // a zone with nothing to send needs no destinations
                if !members[oz].iter().any(|&c| s.budget(c) > 0.0) {
                    continue;
                }
                let total = table.cdf(engine.z.gx[oz], engine.z.gy[oz], &zx, &zy, &zmass, &mut cdf);
                if kind == AIRPORT {
                    airport_scale[oz] = total;
                }
                if total > 0.0 {
                    s.sample(&members[oz], &cdf, total, out);
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
            km: vec![0.0; kinds],
            school_enrollment: school_enrollment.unwrap_or_default(),
            university_enrollment: university_enrollment.unwrap_or_default(),
            airport_origin,
            airport_scale,
            start,
        };
        for r in chunks(api.destinations.len()) {
            api.add_totals(r);
        }
        Ok(api)
    }
    /// Outings retained, walking fraction, and mean distance by purpose.
    pub fn info(&self) -> Vec<f64> {
        let distance = &self.km;
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
    /// The station complexes of the networks `set_network` is given from now on (T-161), as
    /// `DemandApi::set_complexes`.
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.engine.set_complexes(complex);
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
        if self.engine.sz.is_none() {
            return Ok(now_ms() - t);
        }
        let mut index: HashMap<u64, u32, FastHash> = HashMap::default();
        for r in chunks(self.destinations.len()) {
            self.index_edges(r, &mut index);
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
        let xi = xfer_index(&g);
        let (probability, loads, parts, xfer) = self.pass(&g, &p, q, &xi);
        let (trips, rail, walk) = self.mode_totals(q, &probability);
        self.per[q] = Some(Period {
            net,
            p,
            g,
            loads,
            parts,
            probability,
            round: 0,
            ms: now_ms() - t,
            fac,
            uniform: true,
            trips,
            rail,
            walk,
            xi,
            xfer,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(Some(st)) = self.per.get(q) else {
            return false;
        };
        // the crowding and the waits for full trains (T-188) from every population's loads
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            return false;
        };
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        st.round += 1;
        if !st.uniform || fac != st.fac || left != st.g.rn_left {
            st.uniform = false;
            st.fac = fac;
            set_crowding(&mut st.g, &st.fac);
            st.g.rn_left = left;
            let (probability, loads, parts, xfer) = self.pass(&st.g, &st.p, q, &st.xi);
            for (old, new) in st.probability.iter_mut().zip(&probability) {
                msa(old, new, st.round);
            }
            msa(&mut st.xfer, &xfer, st.round);
            msa(&mut st.loads.seg, &loads.seg, st.round);
            msa(&mut st.loads.rn_board, &loads.rn_board, st.round);
            msa(&mut st.loads.rn_alight, &loads.rn_alight, st.round);
            msa(&mut st.loads.entry, &loads.entry, st.round);
            msa(&mut st.loads.exit, &loads.exit, st.round);
            // the same averaging, so the purposes still add up to the total
            for (old, new) in st.parts.iter_mut().zip(&parts) {
                msa(&mut old.seg, &new.seg, st.round);
                msa(&mut old.rn_board, &new.rn_board, st.round);
                msa(&mut old.rn_alight, &new.rn_alight, st.round);
            }
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
    /// As `DemandApi::transfers` (T-176): every purpose's transfers over the periods held.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    pub fn solved_mask(&self) -> u32 {
        (0..5)
            .filter(|&q| self.per[q].is_some())
            .fold(0, |m, q| m | 1 << q)
    }
    /// Each purpose's share of `seg`, `board` and `alight` (T-158): for each purpose kind in
    /// `info` order, three blocks of one value per route node (riders on the segment, boarding,
    /// alighting). They add up to the three totals. Empty before the period is solved.
    pub fn part_loads(&self, q: usize) -> Vec<f32> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        let mut out = Vec::with_capacity(st.parts.len() * 3 * st.loads.seg.len());
        for part in &st.parts {
            out.extend_from_slice(&part.seg);
            out.extend_from_slice(&part.rn_board);
            out.extend_from_slice(&part.rn_alight);
        }
        out
    }
    /// Six mode-major cell blocks, home and activity ends, trip legs over solved periods.
    pub fn cell_modes(&self) -> Vec<f32> {
        self.modes_of(None, WEEKDAY)
    }
    /// `cell_modes` for one purpose kind only (T-158's display filter).
    pub fn cell_modes_of(&self, kind: u32) -> Vec<f32> {
        self.modes_of(Some(kind as usize), WEEKDAY)
    }
    /// `cell_modes` on a kind of day (T-260, `days.rs`): that day's trips and period shares.
    pub fn cell_modes_day(&self, day: u32) -> Vec<f32> {
        self.modes_of(None, day as usize)
    }
    /// `cell_modes_of` on a kind of day.
    pub fn cell_modes_of_day(&self, kind: u32, day: u32) -> Vec<f32> {
        self.modes_of(Some(kind as usize), day as usize)
    }
    /// Each purpose kind's trips, rail trips and walked trips in period `q` (a weekday's), three
    /// numbers a kind in `info` order; the kinds add up to `summary`'s (T-260: the weekend's
    /// mode split is worked out from them). Empty before the period is solved.
    pub fn kind_totals(&self, q: usize) -> Vec<f64> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        let nk = self.totals.len();
        let mut out = vec![0.0; 3 * nk];
        for kind in 0..nk {
            let share = (OUTWARD[kind][q] + RETURN[kind][q]) as f64;
            out[3 * kind] = self.totals[kind] * share;
            out[3 * kind + 2] = (self.walks[kind] * share).max(0.0);
        }
        for (i, pair) in self.pairs.iter().enumerate() {
            let pr = st.probability[i];
            out[3 * pair.kind + 1] += pair.trips as f64
                * (OUTWARD[pair.kind][q] as f64 * pr[0] as f64 + RETURN[pair.kind][q] as f64 * pr[1] as f64);
        }
        out
    }
    /// `transfers` on a kind of day (T-260). The purposes' transfers are kept together, so each
    /// period's are scaled by its rail riders' mean load factor that day.
    pub fn transfers_day(&self, sel: &[u8], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (q, st) in self.per.iter().enumerate() {
            let Some(st) = st else { continue };
            let at = out.len();
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
            if day as usize != WEEKDAY {
                let k = self.kind_totals(q);
                let (mut rail, mut weighted) = (0.0, 0.0);
                for kind in 0..k.len() / 3 {
                    rail += k[3 * kind + 1];
                    weighted += k[3 * kind + 1] * load_factor(day as usize, everyday(kind), q) as f64;
                }
                scale_triples(&mut out[at..], if rail > 0.0 { (weighted / rail) as f32 } else { 0.0 });
            }
        }
        out
    }
}

#[wasm_bindgen]
impl EverydayApi {
    /// `cell_modes_day` over the periods in `mask` (bit q = period q) whether solved or not, no
    /// rail where a period has no solve (T-325): a zone no station reaches, held by the
    /// auxiliary worker alone, answers for each solving worker's periods in turn, so its bubbles
    /// add up exactly as a solve's would.
    pub fn cell_modes_mask_day(&self, mask: u32, day: u32) -> Vec<f32> {
        self.modes_in(None, day as usize, Some(mask))
    }
    /// `cell_modes_of_day` over the periods in `mask`, as `cell_modes_mask_day`.
    pub fn cell_modes_of_mask_day(&self, kind: u32, mask: u32, day: u32) -> Vec<f32> {
        self.modes_in(Some(kind as usize), day as usize, Some(mask))
    }
}

impl EverydayApi {
    fn modes_of(&self, only: Option<usize>, day: usize) -> Vec<f32> {
        self.modes_in(only, day, None)
    }
    /// `mask` None: the solved periods, each with its rail. Some: the periods in the mask; a
    /// period's rail only where it has a solve.
    fn modes_in(&self, only: Option<usize>, day: usize, mask: Option<u32>) -> Vec<f32> {
        let n = self.engine.city.len();
        let mut out = vec![0.0; 6 * n];
        // edges are stored purpose by purpose (`edge`)
        let range = match only {
            Some(kind) if kind < self.totals.len() => self.start[kind]..self.start[kind + 1],
            Some(_) => 0..0,
            None => 0..self.destinations.len(),
        };
        for k in range {
            let Some((kind, c, d, w, pw)) = self.edge(k) else {
                continue;
            };
            let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
            let mut total = 0.0;
            let mut rail = 0.0;
            let p = everyday(kind);
            for q in 0..5 {
                let held = match mask {
                    Some(m) => m >> q & 1 == 1,
                    None => self.per[q].is_some(),
                };
                if held {
                    let (o, r) = (share_out(day, p, q), share_ret(day, p, q));
                    total += w * (o + r);
                    if let (true, Some(st)) = (pi != NONE, &self.per[q]) {
                        let pr = st.probability[pi as usize];
                        let scale = self.edge_motorized_scale(pi, pw);
                        rail += w * scale * (o * pr[0] + r * pr[1]);
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
}

#[wasm_bindgen]
impl EverydayApi {
    /// Each enabled purpose: 30 timing values + 3 far-cell mode blocks. All legs; no cut.
    pub fn selection(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        self.selection_day(end, cells, WEEKDAY as u32)
    }
    /// `selection` on a kind of day (T-260): that day's trips and period shares.
    pub fn selection_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        let n = self.engine.city.len();
        let stride = 30 + 3 * n;
        let mut out = vec![0.0; self.totals.len() * stride];
        for (kind, (timing, far, legs)) in self.selection_parts(end, cells, day).into_iter().enumerate() {
            let base = kind * stride;
            out[base..base + 30].copy_from_slice(&timing);
            for (i, &c) in far.iter().enumerate() {
                for m in 0..3 {
                    out[base + 30 + m * n + c as usize] = legs[3 * i + m];
                }
            }
        }
        out
    }
    /// `selection_day` packed for the demand worker (T-285): only the far cells some sample of
    /// the selection reaches, the same numbers. Per purpose kind: 30 timing values, `m`, then `m`
    /// far cells (ascending), their rail, their walking and their driving legs.
    pub fn selection_sparse_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (timing, far, legs) in self.selection_parts(end, cells, day) {
            out.extend_from_slice(&timing);
            out.push(far.len() as f32);
            out.extend(far.iter().map(|&c| c as f32));
            for m in 0..3 {
                out.extend((0..far.len()).map(|i| legs[3 * i + m]));
            }
        }
        out
    }
}

impl EverydayApi {
    /// Per purpose kind: the selection's 30 timing values, the far cells its samples reach
    /// (ascending) and their rail, walking and driving legs (3 per cell). T-285: only the
    /// selection's samples are visited and weighed. A purpose whose samples are laid out by the
    /// selected end (homes for outings, sites for schools) reads their ranges directly; at the
    /// other end the sample list is scanned for selected cells, and only those are weighed. The
    /// samples are taken in the order of the whole list, so every sum is the same as visiting all.
    fn selection_parts(&self, end: u32, cells: &[u32], day: u32) -> Vec<([f32; 30], Vec<u32>, Vec<f32>)> {
        let day = day as usize;
        let n = self.engine.city.len();
        let mut selected = vec![false; n];
        for &c in cells {
            if (c as usize) < n {
                selected[c as usize] = true;
            }
        }
        let sorted: Vec<usize> = (0..n).filter(|&c| selected[c]).collect();
        // a far cell's slot in this kind's lists
        let mut slot = vec![NONE; n];
        let kinds = self.totals.len();
        let mut parts = Vec::with_capacity(kinds);
        for kind in 0..kinds {
            let (a, b) = (self.start[kind], self.start[kind + 1]);
            let mut timing = [0f32; 30];
            let mut far_cells: Vec<u32> = vec![];
            let mut legs: Vec<f32> = vec![];
            let mut visit = |k: usize| {
                let Some((kind, c, d, w, pw)) = self.edge(k) else { return };
                let (near, far) = if end == 0 { (c, d) } else { (d, c) };
                if !selected[near] {
                    return;
                }
                let s = if slot[far] == NONE {
                    slot[far] = far_cells.len() as u32;
                    far_cells.push(far as u32);
                    legs.extend([0.0; 3]);
                    far_cells.len() - 1
                } else {
                    slot[far] as usize
                };
                let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
                for q in 0..5 {
                    if let Some(st) = &self.per[q] {
                        for direction in 0..2 {
                            let share = if direction == 0 { share_out(day, everyday(kind), q) } else { share_ret(day, everyday(kind), q) };
                            let pr = if pi == NONE { 0.0 } else { st.probability[pi as usize][direction] * self.edge_motorized_scale(pi, pw) };
                            let rail = w * share * pr;
                            let walk = w * share * pw;
                            let drive = (w * share - walk - rail).max(0.0);
                            for (m, v) in [rail, walk, drive].into_iter().enumerate() {
                                timing[direction * 15 + m * 5 + q] += v;
                                legs[3 * s + m] += v;
                            }
                        }
                    }
                }
            };
            // the samples' fixed end: the home for outings, the site for site-anchored purposes
            let fixed_is_near = (end == 0) != site_anchored(kind);
            if fixed_is_near {
                let per = SAMPLES[kind];
                for &c in &sorted {
                    let k0 = a + c * per;
                    for k in k0..(k0 + per).min(b) {
                        visit(k);
                    }
                }
            } else {
                for k in a..b {
                    let d = self.destinations[k];
                    if d != NONE && selected[d as usize] {
                        visit(k);
                    }
                }
            }
            // ascending far cells, as the dense layout reads them
            let mut order: Vec<usize> = (0..far_cells.len()).collect();
            order.sort_unstable_by_key(|&i| far_cells[i]);
            let cells_out: Vec<u32> = order.iter().map(|&i| far_cells[i]).collect();
            let legs_out: Vec<f32> = order.iter().flat_map(|&i| [legs[3 * i], legs[3 * i + 1], legs[3 * i + 2]]).collect();
            for &c in &far_cells {
                slot[c as usize] = NONE;
            }
            parts.push((timing, cells_out, legs_out));
        }
        parts
    }
}
