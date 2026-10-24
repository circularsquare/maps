use super::*;
use crate::demand::network::{route_nodes, RealNet, PERIODS};
use crate::demand::{kernel, pack, synth};

/// The synthetic city's inner 12 km with two crossing lines (as api.rs's tests).
fn small() -> (DemandApi, Vec<f32>) {
    let (h, b, n) = small_pack();
    let mut api = DemandApi::new(&h, &b).unwrap();
    // T-854: parking at home from none to most, cell by cell
    api.params_mut().home_dens = Some(std::sync::Arc::new((0..n).map(|c| (c % 5) as f32 * 10_000.0).collect()));
    let (xy, stops, times, km) = small_net();
    api.set_network(&xy, &[17, 17], &stops, &times, &[20.0, 10.0, 4.0, 12.0, 6.0, 0.0], &[10.0, 8.0]).unwrap();
    api.set_fares(2.9, 0.1, &km).unwrap();
    (api, xy)
}

/// The same city and network for the everyday purposes (T-586).
fn small_everyday() -> EverydayApi {
    let (h, mut b, n) = small_pack();
    // errand and leisure attractors, uneven from cell to cell
    let jobs: Vec<f32> = (0..n).map(|c| ((c * 7919) % 13) as f32).collect();
    let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
    for name in ["errand_jobs", "leisure_jobs"] {
        let offset = b.len();
        b.extend(jobs.iter().flat_map(|v| v.to_le_bytes()));
        h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name": name, "dtype": "f32", "per": "cell", "count": n, "offset": offset}));
    }
    let mut e = EverydayApi::new(&h.to_string(), &b).unwrap();
    e.params_mut().home_dens = Some(std::sync::Arc::new((0..n).map(|c| (c % 5) as f32 * 10_000.0).collect()));
    let (xy, stops, times, km) = small_net();
    e.set_network(&xy, &[17, 17], &stops, &times, &[20.0, 10.0, 4.0, 12.0, 6.0, 0.0], &[10.0, 8.0]).unwrap();
    e.set_fares(2.9, 0.1, &km).unwrap();
    e
}

fn small_pack() -> (String, Vec<u8>, usize) {
    let mut city = synth::synth_city(1).inner(12_000.0);
    let p = Params::default();
    let z = kernel::build_zones(&city, p.zone_m);
    let g = kernel::gravity(&z, &p, 100, 1e-5);
    city.gravity = Some(kernel::to_pack_gravity(&z, &g, &p));
    let (h, b) = pack::write(&city);
    (h, b, city.len())
}

/// Two crossing lines of 17 stations 1 km apart, and their fares' km.
fn small_net() -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut xy = vec![];
    for i in -8..=8 {
        xy.extend([i as f32 * 1000.0, 0.0]);
    }
    let mut line_b = vec![];
    for i in -8..=8i32 {
        if i == 0 {
            line_b.push(8u32);
        } else {
            line_b.push((xy.len() / 2) as u32);
            xy.extend([0.0, i as f32 * 1000.0]);
        }
    }
    let stops: Vec<u32> = (0..17).chain(line_b).collect();
    let mut times = vec![];
    for _ in 0..2 {
        for run in 0..2 {
            for _ in 0..3 {
                for k in 0..17 {
                    let end = if run == 0 { k == 16 } else { k == 0 };
                    times.push(if end { 0.0 } else { 90.0 });
                }
            }
        }
    }
    let km: Vec<f32> = (0..2).flat_map(|_| (0..2).flat_map(|_| (0..17).map(|k| if k == 16 { 0.0 } else { 1.0 }))).collect();
    (xy, stops, times, km)
}

/// The everyday choice is the everyday pass's (T-586): for one sample of every subzone pair,
/// its rail chance outward equals the free-flow solve's to the bit (after crowding the solve
/// keeps the rounds' average, the explorer the last round's costs); the station pairs' shares
/// add up to 1; walking is the cells' share less the walkers rail takes.
#[test]
fn everyday_choice_reproduces_the_pass() {
    let mut e = small_everyday();
    for q in [0usize, 2] {
        assert!(e.solve(q));
        let samples = e.pair_samples(q);
        let api = e.engine();
        let sz = api.sz.as_ref().unwrap();
        let (mut worst, mut rail, mut checked) = (0f32, 0f32, 0);
        let (k0, c0, d0, _) = samples[0];
        let sk = {
            let ev = e.trip_inputs(k0, q, c0, d0).unwrap();
            kernel::skims(ev.g, ev.p)
        };
        for &(kind, c, d, pr) in &samples {
            let ev = e.trip_inputs(kind, q, c, d).unwrap();
            assert!(ev.pair.is_some());
            let cells_m = (api.city.x[c] - api.city.x[d]).hypot(api.city.y[c] - api.city.y[d]);
            let pc = everyday_choice(&ev, sz, &sk, sz.cell[c] as usize, sz.cell[d] as usize, cells_m, false);
            worst = worst.max((pc.r - pr).abs());
            rail += pc.r;
            if pc.why == RAIL_OFFERED {
                checked += 1;
                let sum: f32 = pc.pairs.iter().map(|x| x.2).sum();
                assert!((sum - 1.0).abs() < 1e-4, "pair shares add to {sum}");
            }
            assert!((pc.pw - ev.no_rail * (1.0 - pc.r)).abs() < 1e-6);
        }
        println!("period {q}: {} pairs, {checked} with rail, mean rail {:.3}, worst {worst:.1e}", samples.len(), rail / samples.len() as f32);
        assert!(checked > 100 && rail > 0.0);
        assert!(worst < 1e-6, "period {q}: rail chance off by {worst:.2e}");
    }
    // the query: a trip along line A answers, with routes, and its chances add up
    let (sx, sy) = (-6000.0, 0.0);
    let v = e.trip(0, 0, sx, sy, 6000.0, 0.0);
    assert_eq!(v[0], 1.0);
    assert!((v[17] + v[18] + v[19] - 1.0).abs() < 1e-6);
    assert!(v[33] > 60.0, "a 12 km walk is over an hour: {}", v[33]);
    assert_eq!(e.trip(0, 1, 0.0, 0.0, 1.0, 1.0)[0], 0.0, "period 1 is not solved");
}

/// The pair choice is mode choice's: over every destination, an origin subzone's rail trips add
/// up to what the solve kept for it, and the station pairs' shares to the solve's station pair
/// matrix. Free flow and after a crowding round.
#[test]
fn pair_choice_reproduces_mode_choice() {
    let (mut api, _) = small();
    // every pair solved, none dealt in chunks (T-874), so each one's trips are its own
    api.params_mut().set_extra("commute_chunk", 0.0).unwrap();
    for q in [0usize, 2] {
        api.solve(q);
        for round in 0..2 {
            if round == 1 {
                // a crowded round's own pass only: the solve's sums are averaged with the first
                let seg: Vec<f32> = api.seg(q).iter().map(|v| v * 8.0).collect();
                assert!(api.crowd_with_external(q, &seg));
            }
            let st = api.period(q).unwrap();
            let (p, g) = (&st.p, &st.g);
            let sz = api.sz.as_ref().unwrap();
            let s = g.n_st;
            let sk = kernel::skims(g, p);
            let er = kernel::station_exp(&sk, p);
            let mo = kernel::mode_choice(&api.z, &api.gr, sz, &er, &sk.walk, &sk.walk_max, s, p, &api.nr);
            let (z, gr) = (&api.z, &api.gr);
            let share = p.period_share as f64;
            let mut od = vec![0f64; s * s];
            let (mut worst, mut tot, mut walk_tot) = (0f64, 0f64, 0f64);
            let mut checked = 0;
            for u in 0..sz.n {
                let i = sz.zone[u] as usize;
                if sz.pop[u] <= 0.0 || z.pop[i] <= 0.0 {
                    continue;
                }
                let ou = sz.origin_weight(u, z, gr, share);
                let mut rail = 0f64;
                let mut walk_lost = 0f64;
                for v in 0..sz.n {
                    let j = sz.zone[v] as usize;
                    if sz.jobs[v] <= 0.0 || z.jobs[j] <= 0.0 {
                        continue;
                    }
                    let pc = pair_choice(&api, sz, p, &sk, u, v);
                    if pc.why != RAIL_OFFERED {
                        continue;
                    }
                    let dv = (gr.col[j] * sz.jobs[v] as f64 / z.jobs[j]) as f32;
                    let f = gr.ftab[(gr.base(z, i) + gr.key(z, j)) as usize] * gr.k(z, i, j);
                    let x = ou * f * dv * pc.r;
                    rail += x as f64;
                    walk_lost += (ou * f * dv * pc.walk_lost) as f64;
                    for &(ka, kt, w) in &pc.pairs {
                        od[sz.acc_st[ka] as usize * s + sz.acc_st[kt] as usize] += (x * w) as f64;
                    }
                }
                let want = mo.rail_by_sub[u] as f64;
                worst = worst.max((rail - want).abs() / want.max(1e-3));
                // T-558: and the walkers it takes
                let want_w = mo.walk_lost_by_sub[u] as f64;
                worst = worst.max((walk_lost - want_w).abs() / want.max(1e-3));
                walk_tot += want_w;
                tot += want;
                if want > 0.0 {
                    checked += 1;
                }
                // the period's own copy agrees too (round 0: nothing averaged in yet)
                if round == 0 {
                    assert_eq!(st.sub.rail_home[u], mo.rail_by_sub[u]);
                }
            }
            assert!(checked > 100 && tot > 0.0 && walk_tot > 0.01 * tot, "walking must be in play: {checked} origins, rail {tot:.0}, walking {walk_tot:.0}");
            assert!(worst < 1e-4, "period {q} round {round}: worst origin off by {worst:.2e}");
            let (mut d, mut m) = (0f64, 0f64);
            for k in 0..s * s {
                d += (od[k] - mo.od[k] as f64).abs();
                m += mo.od[k] as f64;
            }
            assert!(d / m < 1e-4, "period {q} round {round}: station pairs off by {:.2e}", d / m);
            println!("period {q} round {round}: {checked} origins, worst {worst:.1e}, station pairs {:.1e}", d / m);
        }
    }
}

/// The query: a trip along one line rides it with no change; one between the two lines changes
/// at the shared centre; the routes' shares add up; a lone route's costs add up to the
/// strategy's; a point off the city answers -1; a period not solved 0.
#[test]
fn trip_answers() {
    let (mut api, _) = small();
    assert_eq!(api.trip(0, 0.0, 0.0, 1.0, 1.0)[0], 0.0, "nothing solved");
    for q in 0..PERIODS {
        api.solve(q);
    }
    assert_eq!(api.trip(0, 0.0, 0.0, 99_000.0, 0.0)[0], -1.0);
    let read = |v: &[f64]| {
        let np = v[24] as usize;
        let mut o = HEAD + np * PAIR_LEN;
        let mut routes = vec![];
        for _ in 0..v[25] as usize {
            let (pi, share, n) = (v[o] as usize, v[o + 1], v[o + 2] as usize);
            o += 3;
            let legs: Vec<Vec<f64>> = (0..n).map(|k| v[o + k * LEG_LEN..o + (k + 1) * LEG_LEN].to_vec()).collect();
            o += n * LEG_LEN;
            routes.push((pi, share, legs));
        }
        assert_eq!(o, v.len());
        routes
    };
    // along line A: 6 km west of the centre to 6 km east
    let v = api.trip(0, -6000.0, 0.0, 6000.0, 0.0);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[13], 1.0, "rail offered");
    let (w, r, dr) = (v[17], v[18], v[19]);
    assert!((w + r + dr - 1.0).abs() < 1e-6 && r > 0.0 && dr > 0.0);
    let routes = read(&v);
    let main = &routes[0];
    assert_eq!(main.2.len(), 1, "one ride");
    assert_eq!(main.2[0][0], 1.0);
    let listed: f64 = routes.iter().map(|r| r.1).sum();
    assert!((listed + v[31] - 1.0).abs() < 1e-5);
    // a route's own costs add up to its pair's strategy cost when it is the only one
    let pair = &v[HEAD + main.0 * PAIR_LEN..HEAD + (main.0 + 1) * PAIR_LEN];
    let st = api.period(0).unwrap();
    let (strategy, l) = (pair[8], &main.2[0]);
    let own = l[6] + l[7] + l[9];
    println!("along A: walk {w:.3} rail {r:.3} drive {dr:.3}; route {:.3}, strategy {strategy:.3} vs own {own:.3}", main.1);
    if (main.1 - pair[9]).abs() < 1e-6 {
        assert!((strategy - own).abs() < 1e-3, "strategy {strategy} vs the route's own {own}");
    }
    // across: west on A to north on B, changing at the centre
    let v = api.trip(0, -6000.0, 0.0, 0.0, 6000.0);
    let routes = read(&v);
    let main = &routes[0];
    let rides: Vec<&Vec<f64>> = main.2.iter().filter(|l| l[0] == 1.0).collect();
    assert_eq!(rides.len(), 2, "a change: {:?}", main.2);
    assert_eq!(st.g.rn_line[rides[0][1] as usize], 0);
    assert_eq!(st.g.rn_line[rides[1][1] as usize], 1);
    assert!(rides[1][8] > 0.0 && rides[1][9] > 0.0, "a wait at the change");
    // night: line B does not run, so the trip across has no route by B
    let v = api.trip(4, -6000.0, 0.0, 0.0, 6000.0);
    for (_, _, legs) in read(&v) {
        assert!(legs.iter().filter(|l| l[0] == 1.0).all(|l| st.g.rn_line[l[1] as usize] == 0));
    }
    // probabilities differ by period with the service
    let morning = api.trip(0, -6000.0, 0.0, 6000.0, 0.0)[18];
    let night = api.trip(4, -6000.0, 0.0, 6000.0, 0.0)[18];
    assert!(morning > night, "morning {morning} night {night}");
}

/// T-915, T-916: a drive to a station is priced as its parts say: the drive at `ride_w` a
/// minute, then the fixed cost, the station end, parking at home and the car's cost, which add
/// up to the entry's perceived minutes beyond the drive.
#[test]
fn ride_parts_add_up() {
    let (mut api, _) = small();
    api.solve(0);
    let p = api.period(0).unwrap().p.clone();
    assert_eq!(p.ride_w(), 2.0 * p.w_drive);
    let mut rides = 0;
    for (ox, oy) in [(-6000.0, 0.0), (-9000.0, 2000.0), (0.0, -8000.0), (-8000.0, -6000.0)] {
        for (dx, dy) in [(6000.0, 0.0), (0.0, 6000.0), (500.0, 500.0)] {
            let v = api.trip(0, ox, oy, dx, dy);
            if v[0] != 1.0 || v[13] != 1.0 {
                continue;
            }
            for k in 0..v[24] as usize {
                let e = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
                if e[12] != 1.0 {
                    assert!(e[13..].iter().all(|&x| x == 0.0));
                    continue;
                }
                rides += 1;
                let (extra, start, end, money, drop, home, car) = (e[13], e[14], e[15], e[16], e[17], e[18], e[19]);
                assert!((e[5] - extra - p.ride_w() as f64 * e[10]).abs() < 1e-3, "the drive at ride_w: {e:?}");
                assert!((start + end + home + car - extra).abs() < 1e-4);
                assert_eq!(start, p.p_ride as f64);
                assert!(car >= p.car_own_min() as f64 - 1e-3, "the car's cost {car}");
                assert!(end <= p.d_drop as f64 + 1e-6 && money >= 0.0 && money <= end);
                assert!(drop == 0.0 || (end == p.d_drop as f64 && money == 0.0));
            }
        }
    }
    assert!(rides > 0, "no drive to a station among the trips");
    // the access bound sees each drive at `w_drive`: under the priced cheapest by its extra weight
    let sz = api.sz.as_ref().unwrap();
    for u in (0..sz.n).filter(|&u| !sz.ride(u).is_empty()) {
        let k = sz.ride(u).min_by(|&a, &b| sz.acc_w[a].total_cmp(&sz.acc_w[b])).unwrap();
        let want = sz.acc_w[k] - (p.ride_w() - p.w_drive) * sz.acc_t[k];
        assert!(sz.bound_min(&p, u) <= want + 1e-4);
    }
}

/// The real New York pack and network (nyc_real_track.json), free flow in the morning peak:
/// for each origin of three trips, its rail trips over every destination add up to the solve's
/// own for that origin subzone; prints the three trips as the explorer reads them.
/// `cargo test --release --features demand trip_real_network -- --ignored --nocapture`
#[test]
#[ignore]
fn trip_real_network() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, j) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let names: Vec<String> = j["stations"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
    let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
    let (rn_st, _, hop_km) = route_nodes(&st_xy, &line_n, &stops);
    a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    a.solve(0);
    // lon/lat to the pack's equirectangular frame (origin -73.985, 40.758)
    const R: f64 = 6371008.8;
    let rad = std::f64::consts::PI / 180.0;
    let at = |lon: f64, lat: f64| ((R * (lon + 73.985) * rad * (40.758 * rad).cos()) as f32, (R * (lat - 40.758) * rad) as f32);
    let trips = [
        ("Park Slope to Midtown", at(-73.978, 40.672), at(-73.981, 40.757)),
        ("Jersey City to Wall St", at(-74.063, 40.733), at(-74.009, 40.706)),
        ("Mineola to Hicksville", at(-73.640, 40.749), at(-73.525, 40.768)),
    ];
    let st = a.period(0).unwrap();
    let line_of = |rn: f64| lines[st.g.rn_line[rn as usize] as usize].clone();
    let name_of = |rn: f64| if rn < 0.0 { "?".to_string() } else { names[rn_st[rn as usize]].clone() };
    let sz = a.sz.as_ref().unwrap();
    let sk = kernel::skims_for(&st.g, &st.p, &kernel::egress_stations(sz, st.g.n_st));
    let (z, gr) = (&a.z, &a.gr);
    // TW_TRIP_DUMP=<file>: the three answers, to compare with the WASM worker's
    let mut dump = serde_json::Map::new();
    for (what, o, dd) in trips {
        let v = a.trip(0, o.0, o.1, dd.0, dd.1);
        dump.insert(what.to_string(), serde_json::json!({ "from": [o.0, o.1], "to": [dd.0, dd.1], "answer": v }));
        println!("\n{what}: cells {} -> {}, subzones {} -> {}, {:.1} km ({:.1} km between the subzones' centres)", v[2], v[3], v[4], v[5], v[6] / 1000.0, v[7] / 1000.0);
        println!("  walk {:.1}%, train {:.1}%, drive {:.1}% (rail {:.1}% of those not walking; utility {:.2})", 100.0 * v[17], 100.0 * v[18], 100.0 * v[19], 100.0 * v[15], v[16]);
        println!("  drive {:.1} min (x1.33 = {:.1}) + parking {:.1} = {:.1} perceived; rail {:.1} perceived (walks {:.1} + {:.1}, base fare {:.1})", v[9], v[10], v[11], v[12], v[14], v[20], v[21], v[22]);
        let np = v[24] as usize;
        for k in 0..np.min(4) {
            let p = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
            println!("  pair {:.1}%: {} ({:.0} m, {:.1}) to {} ({:.0} m, {:.1}), strategy {:.1}", 100.0 * p[9], name_of(p[2]), p[4], p[5], name_of(p[3]), p[6], p[7], p[8]);
        }
        let mut o2 = HEAD + np * PAIR_LEN;
        for _ in 0..v[25] as usize {
            let (share, n) = (v[o2 + 1], v[o2 + 2] as usize);
            let legs: Vec<String> = (0..n)
                .map(|k| {
                    let l = &v[o2 + 3 + k * LEG_LEN..o2 + 3 + (k + 1) * LEG_LEN];
                    if l[0] == 1.0 {
                        format!("{} {} to {} ({} stops, wait {:.1}, ride {:.1})", line_of(l[1]), name_of(l[1]), name_of(l[2]), l[10], l[8], l[5])
                    } else {
                        format!("walk to {} ({:.1} min)", name_of(l[2]), l[5])
                    }
                })
                .collect();
            println!("  route {:.1}%: {}", 100.0 * share, legs.join("; "));
            o2 += 3 + n * LEG_LEN;
        }
        // the check: this origin's rail trips over every destination are the solve's own
        if v[4] < 0.0 {
            continue;
        }
        let u = v[4] as usize;
        let i = sz.zone[u] as usize;
        let ou = sz.origin_weight(u, z, gr, st.p.period_share as f64);
        let mut rail = 0f64;
        for w in 0..sz.n {
            let jz = sz.zone[w] as usize;
            if sz.jobs[w] <= 0.0 || z.jobs[jz] <= 0.0 {
                continue;
            }
            let pc = pair_choice(&a, sz, &st.p, &sk, u, w);
            if pc.why != RAIL_OFFERED {
                continue;
            }
            let dv = (gr.col[jz] * sz.jobs[w] as f64 / z.jobs[jz]) as f32;
            let f = gr.ftab[(gr.base(z, i) + gr.key(z, jz)) as usize] * gr.k(z, i, jz);
            rail += (ou * f * dv * pc.r) as f64;
        }
        let want = st.sub.rail_home[u] as f64;
        println!("  check: origin subzone {u}'s rail trips over every destination {rail:.3}, the solve's {want:.3} (off by {:.1e})", (rail - want).abs() / want.max(1e-9));
        assert!((rail - want).abs() <= 1e-4 * want.max(1.0));
    }
    if let Ok(path) = std::env::var("TW_TRIP_DUMP") {
        std::fs::write(&path, serde_json::to_string(&dump).unwrap()).unwrap();
        println!("dumped to {path}");
    }
}
