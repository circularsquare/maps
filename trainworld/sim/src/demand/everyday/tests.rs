use super::*;
use crate::demand::lod::Sampling;
use crate::demand::network::{route_nodes, RealNet};
fn fixture(distance: f32) -> (String, Vec<u8>) {
    fixture_of(distance, 100.0)
}
/// `fixture` with this many residents (a thousand draw social samples, `sample_count`).
fn fixture_of(distance: f32, residents: f32) -> (String, Vec<u8>) {
    let city = pack::City {
        name: "test".into(),
        h3: vec![1, 2],
        x: vec![0.0, distance],
        y: vec![0.0, 0.0],
        pop: vec![residents, 0.0],
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
/// T-852: with blocks, a zone's draw on the others is the mean decay between their squares.
#[test]
fn decay_table_with_blocks() {
    let city = crate::demand::synth::synth_city(1).inner(12_000.0);
    let z0 = build_zones(&city, 2000.0);
    let ends: Vec<f64> = (0..z0.n).map(|i| (z0.gx[i] + z0.gy[i]) as f64).collect();
    let z = coarsen(&z0, &ends, 9.0);
    assert!(z.mixed());
    let t = DecayTable::new(&z, 3000.0);
    let all: Vec<usize> = (0..z.n).collect();
    let (zx, zy) = t.places(&z, &all);
    let mass: Vec<f64> = (0..z.n).map(|j| 1.0 + j as f64).collect();
    let sq = |i: usize| -> Vec<(i32, i32)> {
        let (s, gx, gy) = (z.side(i), z.gx[i], z.gy[i]);
        (0..s).flat_map(|a| (0..s).map(move |b| (gx + a, gy + b))).collect()
    };
    for o in [0, z.n / 2, z.n - 1] {
        let mut cdf = vec![];
        let got = t.cdf(&z, o, &zx, &zy, &mass, &mut cdf);
        let want: f64 = (0..z.n).map(|j| {
            let (a, b) = (sq(o), sq(j));
            let m: f64 = a.iter().flat_map(|p| b.iter().map(move |q| (p, q))).map(|(p, q)| {
                let d = if p == q { 1040.0 } else { 2000.0 * (((p.0 - q.0).pow(2) + (p.1 - q.1).pow(2)) as f64).sqrt() };
                (-d / 3000.0).exp()
            }).sum::<f64>() / (a.len() * b.len()) as f64;
            mass[j] * m
        }).sum();
        assert!((got / want - 1.0).abs() < 1e-5, "{got} vs {want}");
    }
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
    assert_eq!(a.summary(1)[2], 0.0);
    assert!(a.summary(3)[2] > 0.0);
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
    let mut w = vec![];
    for k in a.start[2]..a.destinations.len() {
        if let Some((kind, home, school, wk, _)) = a.edge(k) {
            assert_eq!((kind, home, school), (2, 0, 1));
            school_edges += 1;
            w.push(wk);
        }
    }
    assert_eq!(school_edges, sample_count(2, 90.0));
    // T-1874: one home per 30 trips, of uneven sizes adding up to the site's trips
    assert_eq!(school_edges, 3);
    assert!((w.iter().sum::<f32>() - 90.0).abs() < 1e-3 && w.iter().any(|&x| (x - w[0]).abs() > 1.0), "{w:?}");
    assert_eq!((sample_count(2, 59.0), sample_count(3, 450.0), sample_count(3, 40000.0), sample_count(0, 400.0)), (2, 15, 100, 16));
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
fn short_everyday_trips_walk_what_rail_leaves() {
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
    // T-558: with no rail these walk; a train this attractive takes walkers and drivers in
    // proportion, so each cell walks its share of what rail leaves
    for c in 0..2 {
        let total = modes[c]+modes[3+c]+modes[6+c];
        let pw = a.no_rail(c, 2, 400.0-city.x[c]);
        assert!(pw>0.9, "{pw}");
        assert!(modes[c]>0.1*total, "rail takes some of them");
        assert!((modes[3+c]-(total-modes[c])*pw).abs()<1e-4*total, "cell {c}: walk {} rail {} total {total} pw {pw}", modes[3+c], modes[c]);
        assert!(modes[6+c]>=0.0);
    }
    let summaries: Vec<_>=(0..5).map(|q|a.summary(q)).collect();
    let rail: f64=summaries.iter().map(|v|v[2]).sum();
    assert!((rail-modes[..3].iter().sum::<f32>() as f64).abs()<1e-3);
    let walk: f64=summaries.iter().map(|v|v[3]).sum();
    let trips: f64=summaries.iter().map(|v|v[1]).sum();
    assert!((walk-modes[3..6].iter().sum::<f32>() as f64).abs()<1e-5*trips, "{walk} against {}", modes[3..6].iter().sum::<f32>());
    let selected=a.selection(1,&[2]);
    for kind in 0..2 {
        let slice=&selected[kind*39..(kind+1)*39];
        assert!((slice[..30].iter().sum::<f32>()-slice[30..].iter().sum::<f32>()).abs()<1e-3);
    }
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
    for k in a.start[3]..a.destinations.len() {
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
/// The pack with the named per-cell arrays set to zero (a place where OSM or the source has
/// no sites for that purpose).
fn zeroed(h: &str, b: &[u8], names: &[&str]) -> (String, Vec<u8>) {
    let hv: serde_json::Value = serde_json::from_str(h).unwrap();
    let mut b = b.to_vec();
    for a in hv["arrays"].as_array().unwrap() {
        if names.contains(&a["name"].as_str().unwrap()) {
            let (o, n) = (a["offset"].as_u64().unwrap() as usize, a["count"].as_u64().unwrap() as usize);
            b[o..o + 4 * n].fill(0);
        }
    }
    (h.to_string(), b)
}
#[test]
fn a_purpose_with_no_sites_opens_empty() {
    let (h, b) = university_fixture(4000.0);
    let full = EverydayApi::new(&h, &b).unwrap();
    let (zh, zb) = zeroed(&h, &b, &["university_adults", "university_enrollment"]);
    let mut a = EverydayApi::new(&zh, &zb).unwrap();
    assert_eq!(a.purpose_count(), 3, "the school slot stays, the empty university is trimmed");
    assert_eq!(&a.info()[..9], &full.info()[..9]);
    assert_eq!(&a.destinations[..], &full.destinations[..a.destinations.len()]);
    network(&mut a, 4000.0, [12.0; 3]);
    for q in 0..5 { assert!(a.solve(q)); }
    assert!(a.selection(1, &[0]).iter().all(|v| v.is_finite()));
    // university empty but airports after it: the university slot is zeros
    let (fh, fb) = facility_fixture(&[0.0, 4000.0], Some(&[0.0, 50.0]), Some(&[0.0, 10.0]));
    let (fh, fb) = zeroed(&fh, &fb, &["errand_jobs"]);
    let mut a = EverydayApi::new(&fh, &fb).unwrap();
    assert_eq!(a.purpose_count(), 6);
    assert_eq!(a.totals[0], 0.0, "no errand sites: no errand trips");
    let (oh, ob) = facility_fixture(&[0.0, 4000.0], Some(&[0.0, 50.0]), Some(&[0.0, 10.0]));
    assert_eq!(a.totals[1], EverydayApi::new(&oh, &ob).unwrap().totals[1], "social is untouched");
    network(&mut a, 4000.0, [12.0; 3]);
    for q in 0..5 { assert!(a.solve(q)); }
    assert!(a.selection(1, &[1]).iter().all(|v| v.is_finite()));
    // no leisure sites: social falls back to residents, nothing divides by zero
    let (lh, lb) = facility_fixture(&[0.0, 4000.0], Some(&[0.0, 50.0]), None);
    let (lh, lb) = zeroed(&lh, &lb, &["leisure_jobs"]);
    let a = EverydayApi::new(&lh, &lb).unwrap();
    assert!(a.info().iter().all(|v| v.is_finite()));
    // no residents is still an error
    let (ph, pb) = facility_fixture(&[0.0, 4000.0], None, None);
    let (eh, eb) = zeroed(&ph, &pb, &["pop"]);
    assert!(EverydayApi::new(&eh, &eb).is_err());
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
    CHECK_LANES.with(|c| c.set(true));
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

/// T-433: the one walk for every purpose gives each purpose's loads as the kernel's assignment
/// of that purpose alone, to the bit (`check_lanes`), free flow and two crowded rounds.
#[test]
fn lanes_match_the_kernel_assignment() {
    CHECK_LANES.with(|c| c.set(true));
    let (h, b) = university_fixture(4000.0);
    let mut a = EverydayApi::new(&h, &b).unwrap();
    network(&mut a, 4000.0, [12.0; 3]);
    for q in 0..5 {
        assert!(a.solve(q));
    }
    for factor in [500.0, 2000.0] {
        for q in 0..5 {
            let shared: Vec<_> = a.seg(q).iter().map(|v| v * factor).collect();
            assert!(a.crowd_with_external(q, &shared));
        }
    }
    assert!(a.seg(0).iter().sum::<f32>() > 0.0, "riders to compare");
}

/// T-158: the real New York pack's everyday layer on the real network (nyc_real_track.json),
/// one period, free flow and a crowding round, with timings; the per-purpose loads must add up
/// to the total. `cargo test --release --features demand everyday_real_network -- --ignored --nocapture`
#[test]
#[ignore]
fn everyday_real_network() {
    CHECK_LANES.with(|c| c.set(true));
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let t = now_ms();
    let mut a = EverydayApi::new(&h, &b).unwrap();
    println!("open {:.0} ms, {} purposes", now_ms() - t, a.purpose_count());
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let hop_km = route_nodes(&st_xy, &line_n, &stops).2;
    let ms = a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    println!("set_network {ms:.0} ms, {} pairs, {} route nodes", a.pairs.len(), 2 * stops.len());
    for q in [0usize, 1] {
        let t = now_ms();
        a.solve(q);
        let free = now_ms() - t;
        let total: Vec<f32> = a.seg(q).iter().map(|v| v * 3.0).collect();
        let t = now_ms();
        a.crowd_with_external(q, &total);
        let round = now_ms() - t;
        println!("period {q}: free flow {free:.0} ms, crowding round {round:.0} ms, rail {:.0}", a.summary(q)[2]);
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
    let pop = a.engine.pop();
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
                    let pr = if pi == NONE { 0.0 } else { st.probability[pi as usize][direction] };
                    let rail = w * share * pr;
                    let walk = (w * share - rail) * pw;
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
    facility_fixture_of(x, airport, medical, 100.0)
}
fn facility_fixture_of(x: &[f32], airport: Option<&[f32]>, medical: Option<&[f32]>, residents: f32) -> (String, Vec<u8>) {
    let n = x.len();
    let one = |v: f32| (0..n).map(|c| if c == 0 { v } else { 0.0 }).collect::<Vec<f32>>();
    let city = pack::City {
        name: "facilities".into(),
        h3: (1..=n as u64).collect(),
        x: x.to_vec(),
        y: vec![0.0; n],
        pop: one(residents),
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
    let one = 50.0 / Sampling::FINE.max_few as f64 + 1e-3;
    assert!((to[1] - 30.0).abs() <= one && (to[2] - 20.0).abs() <= one, "{to:?}");
}

#[test]
fn attractions_keep_their_visits_and_stay_optional() {
    let x = [0.0, 4000.0, 9000.0];
    let (h, b) = facility_fixture(&x, Some(&[0.0, 50.0, 0.0]), None);
    let commutes = DemandApi::new(&h, &b).unwrap();
    // one a year figure near cell 2, one 5 km off the pack (no cell within reach)
    let year = 3_650_000.0;
    let a = EverydayApi::beside_with(&commutes, &h, &b, &[9100.0, 50.0, year, 30000.0, 0.0, year]).unwrap();
    assert_eq!(a.purpose_count() as usize, ATTRACTION + 1, "attractions after empty medical");
    let p = everyday(ATTRACTION);
    let weekday = year as f64 * 7.0 / 365.0 / (5.0 + day_rate(SATURDAY, p) + day_rate(SUNDAY, p)) as f64;
    assert!((a.totals[ATTRACTION] / weekday - 1.0).abs() < 1e-4, "{} against {weekday}", a.totals[ATTRACTION]);
    assert_eq!(a.attractions(), vec![0.0, 2.0, weekday as f32]);
    for k in 0..a.destinations.len() {
        if let Some((ATTRACTION, _, site, _, _)) = a.edge(k) {
            assert_eq!(site, 2);
        }
    }
    // without attractions the layer is as before
    let old = EverydayApi::beside(&commutes, &h, &b).unwrap();
    assert_eq!(old.purpose_count() as usize, AIRPORT + 1);
    assert_eq!(&a.info()[..3 * (AIRPORT + 1)], &old.info()[..]);
    assert_eq!(&a.destinations[..old.destinations.len()], &old.destinations[..]);
    let none = EverydayApi::beside_with(&commutes, &h, &b, &[30000.0, 0.0, year]).unwrap();
    assert_eq!(none.purpose_count() as usize, AIRPORT + 1);
}

/// T-1932: an attraction's staff come out of the leisure lumps within reach, nearest first, down
/// to the lump size; small cells and far attractions leave the weights alone.
#[test]
fn attraction_staff_leave_the_social_weight() {
    let (h, b) = facility_fixture(&[0.0, 500.0, 900.0, 3000.0], None, None);
    let commutes = DemandApi::new(&h, &b).unwrap();
    let leisure = [5000.0, 3000.0, 2500.0, 500.0];
    let out = social_leisure(&commutes, &[0.0, 0.0, 4_000_000.0], &leisure);
    assert_eq!(out, vec![2000.0, 2000.0, 2500.0, 500.0]);
    assert_eq!(social_leisure(&commutes, &[9000.0, 0.0, 4_000_000.0], &leisure), leisure.to_vec());
    assert_eq!(social_leisure(&commutes, &[], &leisure), leisure.to_vec());
}

/// A pack's header and bin with one more f32 cell array.
fn with_array(h: &str, mut b: Vec<u8>, name: &str, values: &[f32]) -> (String, Vec<u8>) {
    let mut h: serde_json::Value = serde_json::from_str(h).unwrap();
    let offset = b.len();
    b.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"dtype":"f32","per":"cell","count":values.len(),"offset":offset}));
    (h.to_string(), b)
}

/// T-1175: a zone's kept share scales its outings of that purpose alone; a pack without the array,
/// or with one for another zone count (a pack rebuilt since), keeps them all.
#[test]
fn kept_share_scales_a_zones_outings() {
    let (h, b) = fixture(4000.0);
    let a = EverydayApi::new(&h, &b).unwrap();
    let nz = a.engine.z.n;
    let per_zone = |n: usize| {
        let (h, b) = with_array(&h, b.clone(), "everyday_kept_errands", &vec![0.25; n]);
        let mut v: serde_json::Value = serde_json::from_str(&h).unwrap();
        v["arrays"].as_array_mut().unwrap().last_mut().unwrap()["per"] = "zone".into();
        (v.to_string(), b)
    };
    let (h2, b2) = per_zone(nz);
    let k = EverydayApi::new(&h2, &b2).unwrap();
    assert!(a.totals[0] > 0.0 && (k.totals[0] - 0.25 * a.totals[0]).abs() < 1e-6 * a.totals[0]);
    assert_eq!(k.totals[1], a.totals[1]);
    let (h3, b3) = per_zone(nz + 1);
    assert_eq!(EverydayApi::new(&h3, &b3).unwrap().totals[0], a.totals[0]);
}

/// T-1257: a zone that keeps none of a purpose's outings (every one crosses a zone line) gives
/// that purpose no subzone pairs, so its loads stay numbers and a crowding round takes them.
#[test]
fn zero_kept_share_leaves_finite_loads() {
    let (h, b) = fixture(4000.0);
    let nz = EverydayApi::new(&h, &b).unwrap().engine.z.n;
    let (h, b) = with_array(&h, b, "everyday_kept_errands", &vec![0.0; nz]);
    let mut v: serde_json::Value = serde_json::from_str(&h).unwrap();
    v["arrays"].as_array_mut().unwrap().last_mut().unwrap()["per"] = "zone".into();
    let mut a = EverydayApi::new(&v.to_string(), &b).unwrap();
    assert_eq!(a.totals[0], 0.0);
    network(&mut a, 4000.0, [12.0; 3]);
    for q in 0..5 {
        assert!(a.solve(q));
        let total = [a.seg(q), a.board(q), a.alight(q)].concat();
        assert!(total.iter().all(|v| v.is_finite()), "period {q}: {total:?}");
        assert!(a.summary(q).iter().all(|v| v.is_finite()));
        assert!(a.crowd_with_external(q, &total), "period {q}");
    }
    assert!(a.cell_modes().iter().all(|v| v.is_finite()));
}

/// T-196: parks take PARK_SHARE of the social attraction and nothing else; a pack without
/// parks (or with an empty array) is unchanged.
#[test]
fn parks_share_the_social_destinations() {
    let x = [0.0, 3000.0];
    // enough residents for sixteen social samples (`sample_count`)
    let (h0, b0) = facility_fixture_of(&x, None, None, 1000.0);
    let old = EverydayApi::new(&h0, &b0).unwrap();
    let (h, b) = with_array(&h0, b0.clone(), "park_draw", &[0.0, 0.0]);
    assert_eq!(EverydayApi::new(&h, &b).unwrap().destinations, old.destinations);
    let (h, b) = with_array(&h0, b0, "park_draw", &[0.0, 5.0]);
    let a = EverydayApi::new(&h, &b).unwrap();
    // errands unchanged; social outings that reach the park are the only ones over 150 m
    assert_eq!(&a.destinations[..a.start[1]], &old.destinations[..old.start[1]]);
    assert_eq!(old.totals[1], 0.0, "home, jobs and leisure all in the one cell: under 150 m");
    assert!(a.totals[1] > 0.0 && a.totals[1] < 0.25 * 1000.0 * RATE[1] as f64, "{}", a.totals[1]);
    for k in a.start[1]..a.start[2] {
        if let Some((kind, home, site, _, _)) = a.edge(k) {
            assert_eq!((kind, home, site), (1, 0, 1));
        }
    }
}

/// T-1092: the mixture's two parts keep their shares, and a part with nothing in reach
/// leaves the other the whole draw.
#[test]
fn central_mix_shares() {
    let mut cdf = vec![1.0, 3.0];
    assert_eq!(mix_central(&mut cdf, 3.0, &[2.0, 4.0], 4.0, 0.2), 1.0);
    let want = [0.8 / 3.0, 0.8, 0.9, 1.0];
    assert!(cdf.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{cdf:?}");
    let mut cdf = vec![1.0, 4.0];
    assert_eq!(mix_central(&mut cdf, 4.0, &[0.0, 0.0], 0.0, 0.2), 1.0);
    assert_eq!(cdf, [0.25, 1.0, 1.0, 1.0]);
    let mut cdf = vec![0.0, 0.0];
    assert_eq!(mix_central(&mut cdf, 0.0, &[0.0, 0.0], 0.0, 0.2), 0.0);
}

/// T-1092: a dense errand centre 12 km out draws a share of the outings that the local
/// draw sends to a sparse place 2 km away.
#[test]
fn central_destinations_reach_a_dense_centre() {
    let mut x = vec![0.0f32];
    let mut errands = vec![0.0f32];
    // empty land in the home square, so its residents' own density does not swamp it
    for i in 0..9 {
        x.push(100.0 + 180.0 * i as f32);
        errands.push(0.0);
    }
    for i in 0..10 {
        x.push(2100.0 + 90.0 * i as f32);
        errands.push(10.0);
    }
    x.push(12_100.0);
    errands.push(100.0);
    let n = x.len();
    let city = pack::City {
        name: "centre".into(),
        h3: (1..=n as u64).collect(),
        x: x.clone(),
        y: vec![100.0; n],
        pop: (0..n).map(|c| if c == 0 { 1000.0 } else { 0.0 }).collect(),
        jobs: errands.clone(),
        region: vec![],
        gravity: None,
    };
    let (h, b) = pack::write(&city);
    let (h, b) = with_array(&h, b, "errand_jobs", &errands);
    let (h, b) = with_array(&h, b, "leisure_jobs", &errands);
    let open = |share: f32| {
        let mut api = DemandApi::new(&h, &b).unwrap();
        api.params_mut().set_extra("central_share", share).unwrap();
        EverydayApi::beside(&api, &h, &b).unwrap()
    };
    let far = |a: &EverydayApi, kind: usize| {
        let (mut all, mut far) = (0.0, 0.0);
        for k in a.start[kind]..a.start[kind + 1] {
            if let Some((_, _, d, w, _)) = a.edge(k) {
                all += w as f64;
                far += if d == n - 1 { w as f64 } else { 0.0 };
            }
        }
        (all, far / all)
    };
    let (off, on) = (open(0.0), open(CENTRAL_SHARE));
    let ((t0, f0), (t1, f1)) = (far(&off, 0), far(&on, 0));
    assert!((t0 - t1).abs() < 1e-3 * t0, "the outings are conserved: {t0} {t1}");
    assert!(f0 < 0.01 && f1 > 0.08 && f1 < 0.2, "errands to the centre: {f0} without, {f1} with");
    // social: half its attraction is residents, all at home (under 150 m, dropped), so
    // its centre shows with the central draw alone
    let ((_, f0), (_, f1)) = (far(&off, 1), far(&open(1.0), 1));
    assert!(f0 < 0.01 && f1 > 0.3, "social to the centre: {f0} without, {f1} all central");
}

/// T-142: every registered pack's everyday layer opens; prints its opening time and, per
/// purpose, outings a day, walk share and mean km against the pack's residents (the per-capita
/// check in notes/T-142.md). `cargo test --release --features demand every_pack_everyday --
/// --ignored --nocapture`; `TW_PACKS=lag,seo` for some packs only. The `gate:` line is what
/// checks/pack_gate.py reads (T-538).
#[test]
#[ignore]
fn every_pack_everyday() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/packs");
    let registry = concat!(env!("CARGO_MANIFEST_DIR"), "/../app/src/game/registry.json");
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(registry).unwrap()).unwrap();
    let mut ids: Vec<String> = reg["cities"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_string()).collect();
    if let Ok(only) = std::env::var("TW_PACKS") {
        ids = only.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    }
    for city in &ids {
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
        let rate = |k: usize| info.get(3 * k).copied().unwrap_or(0.0) / pop;
        println!(
            "gate: {city} errands={:.4} social={:.4} school={:.4} university={:.5} medical={:.4} purposes={}",
            rate(0), rate(1), rate(2), rate(3), rate(5), a.purpose_count()
        );
        // the airports' own trips against what the 150 m exclusion and 40-80 km taper keep
        let input: f64 = pack::cell_weights(&h, &b, "airport_trips", a.engine.city.len()).unwrap().map_or(0.0, |v| v.iter().map(|&x| x as f64).sum());
        println!("  {city} airport: pack {input:.0} outings a day, kept {:.0}", a.totals.get(AIRPORT).copied().unwrap_or(0.0));
        // six since T-142's airports and medical visits (Springfield's airport slot is empty)
        assert_eq!(a.purpose_count(), 6, "{city}");
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// T-2196: on a line that is a ferry its fans ride it more: visits up, errands as before to the
/// bit; a network with no ferry (no flags, or all 0) as before to the bit, free flow and crowded.
#[test]
fn ferry_fans_ride_visits_only() {
    CHECK_LANES.with(|c| c.set(true));
    let (h, b) = fixture_of(4000.0, 1000.0);
    let solve = |ferry: &[u8], crowd: bool| {
        let mut a = EverydayApi::new(&h, &b).unwrap();
        network(&mut a, 4000.0, [12.0; 3]);
        a.set_ferry(ferry).unwrap();
        for q in 0..5 {
            assert!(a.solve(q));
            if crowd {
                let shared: Vec<_> = a.seg(q).iter().map(|v| v * 2000.0).collect();
                assert!(a.crowd_with_external(q, &shared));
            }
        }
        a
    };
    assert!(EverydayApi::new(&h, &b).map(|mut a| { network(&mut a, 4000.0, [12.0; 3]); a.set_ferry(&[1, 0]).is_err() }).unwrap());
    for crowd in [false, true] {
        let (plain, zero, fans) = (solve(&[], crowd), solve(&[0], crowd), solve(&[1], crowd));
        let nr = plain.seg(0).len();
        for q in 0..5 {
            assert_eq!(bits(&plain.part_loads(q)), bits(&zero.part_loads(q)));
            // (4 is the solve's ms)
            let (x, y) = (plain.summary(q), zero.summary(q));
            assert_eq!((&x[..4], &x[5..]), (&y[..4], &y[5..]));
            let (a, f) = (plain.part_loads(q), fans.part_loads(q));
            if !crowd {
                assert_eq!(bits(&a[..3 * nr]), bits(&f[..3 * nr]), "period {q}: errands");
            }
            let visits = |v: &[f32]| v[3 * nr..4 * nr].iter().sum::<f32>();
            assert!(visits(&f) > visits(&a), "period {q}: visits {} against {}", visits(&f), visits(&a));
            assert!(fans.summary(q)[2] > plain.summary(q)[2]);
        }
    }
}

/// T-2196: fans ride a ferry with trains to and from its ports: the ride is on the ferry's route
/// nodes, a trip is one entry and one exit however it changes, and the purposes add up.
#[test]
fn ferry_fans_change_to_and_from_the_ferry() {
    CHECK_LANES.with(|c| c.set(true));
    let (h, b) = fixture_of(8000.0, 1000.0);
    let solve = |ferry: &[u8]| {
        let mut a = EverydayApi::new(&h, &b).unwrap();
        // a train to a port 1.5 km out, a ferry 5 km on (20 minutes), a train on from its other port
        let line = |t: f32| [t, 0.0, t, 0.0, t, 0.0, 0.0, t, 0.0, t, 0.0, t];
        let times: Vec<f32> = [line(120.0), line(1200.0), line(120.0)].concat();
        a.set_network(&[0.0, 0.0, 1500.0, 0.0, 6500.0, 0.0, 8000.0, 0.0], &[2, 2, 2], &[0, 1, 1, 2, 2, 3], &times, &[12.0; 9], &[4.0, 2.0, 4.0]).unwrap();
        a.set_fares(1.0, 0.1, &[1.5, 0.0, 0.0, 1.5, 5.0, 0.0, 0.0, 5.0, 1.5, 0.0, 0.0, 1.5]).unwrap();
        a.set_ferry(ferry).unwrap();
        for q in 0..5 {
            assert!(a.solve(q));
        }
        a
    };
    let (plain, fans) = (solve(&[]), solve(&[0, 1, 0]));
    let nr = plain.seg(0).len();
    let (mut rode, mut legs) = (false, false);
    for q in 0..5 {
        let (a, f) = (plain.part_loads(q), fans.part_loads(q));
        assert_eq!(bits(&a[..3 * nr]), bits(&f[..3 * nr]), "period {q}: errands");
        // the ferry's route nodes are 4..8
        let ferry = |v: &[f32]| v[3 * nr + 4] + v[3 * nr + 6];
        rode |= ferry(&f) > ferry(&a) * 1.05;
        legs |= f[3 * nr] > a[3 * nr] && f[3 * nr + 8] > a[3 * nr + 8];
        for a in [&plain, &fans] {
            let entries: f32 = a.entries(q).iter().sum();
            let rail = a.summary(q)[2] as f32;
            assert!((entries - rail).abs() <= 1e-3 * rail.max(1.0), "period {q}: entries {entries} against rail trips {rail}");
            let whole = a.seg(q);
            let parts = a.part_loads(q);
            for r in 0..nr {
                let sum: f32 = (0..a.purpose_count() as usize).map(|k| parts[3 * k * nr + r]).sum();
                assert!((sum - whole[r]).abs() <= 1e-4 * whole[r].max(1.0));
            }
        }
        // the fans' trains to the port and from the other one carry them
        assert!(f[3 * nr] >= a[3 * nr] && f[3 * nr + 8] >= a[3 * nr + 8]);
    }
    assert!(rode && legs, "the fans ride the ferry, and the trains to and from it");
}

/// T-2196: the ferry fans' cost on the real New York pack and network, the shortest lines taken as
/// ferries (about as many ports as five harbour lines; a subway line at the fans' weight is a worst
/// case, cheaper for them than nearly anything). `cargo test --release --features demand
/// ferry_fans_real_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn ferry_fans_real_cost() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let hop_km = route_nodes(&st_xy, &line_n, &stops).2;
    let mut order: Vec<usize> = (0..line_n.len()).collect();
    order.sort_by_key(|&l| line_n[l]);
    let mut ferry = vec![0u8; line_n.len()];
    let mut ports = 0;
    for &l in &order {
        if ports >= 18 {
            break;
        }
        ferry[l] = 1;
        ports += line_n[l];
    }
    for flags in [vec![], ferry] {
        let mut a = EverydayApi::new(&h, &b).unwrap();
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        a.set_ferry(&flags).unwrap();
        let mut ms = 0.0;
        let mut rail = 0.0;
        for q in 0..5 {
            let t = now_ms();
            a.solve(q);
            let total: Vec<f32> = a.seg(q).iter().map(|v| v * 3.0).collect();
            a.crowd_with_external(q, &total);
            ms += now_ms() - t;
            rail += a.summary(q)[2];
        }
        println!("ferry lines {}: five periods, free flow and a round, {ms:.0} ms; rail {rail:.0}", flags.iter().filter(|&&f| f == 1).count());
    }
}
/// T-2365: the outings by length add up to the periods' summaries, block by block.
#[test]
fn trip_lengths_add_up_to_the_summary() {
    use crate::demand::days::AVERAGE;
    let (h, b) = fixture(4000.0);
    let mut a = EverydayApi::new(&h, &b).unwrap();
    network(&mut a, 4000.0, [12.0; 3]);
    for q in 0..5 {
        assert!(a.solve(q));
    }
    let shared: Vec<_> = a.seg(0).iter().map(|v| v * 100.0).collect();
    assert!(a.crowd_with_external(0, &shared));
    for day in [WEEKDAY, SATURDAY, AVERAGE] {
        let v = a.trip_lengths_day(day as u32, 0b11111);
        let block = |k: usize| v[k * N_LEN..(k + 1) * N_LEN].iter().sum::<f64>();
        let mut want = [0f64; 4];
        for q in 0..5 {
            let kt = a.kind_totals(q);
            for k in 0..kt.len() / 3 {
                let f = load_factor(day, everyday(k), q) as f64;
                for m in 0..3 {
                    want[m] += f * kt[3 * k + m];
                }
            }
        }
        for k in 0..3 {
            assert!((block(k) - want[k]).abs() <= 1e-6 * want[0].max(1.0), "day {day} block {k}: {} vs {}", block(k), want[k]);
        }
        assert!(want[1] > 0.0, "no rail");
        // the trips are 4 km long
        assert!((block(4) / block(0) - 4.0).abs() < 1e-3);
    }
}
