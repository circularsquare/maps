//! T-2216: place pairs with no road and no flight between them (islands), solved with rail as their
//! only way (`Params::bound_share` against a drive over a bridge that is not there), against the
//! solve with the rule off (`ISLANDS_OFF`, this thread only). notes/T-2216.md "Stage 1".

use super::tests::{composition, real_world, rows_json, solve_all, world_packs};
use super::*;
use crate::demand::kernel::skims_for;
use serde_json::json;

fn set_off(off: bool) {
    ISLANDS_OFF.with(|c| c.set(off));
}

/// `world_packs`, place 0's first zone with commuters and errands (a fifth walking) to place 3's,
/// across the sea.
fn island_packs() -> Vec<String> {
    let mut packs = world_packs();
    let mut v: serde_json::Value = serde_json::from_str(&packs[0]).unwrap();
    let add = [
        ("commute_home", json!([0])),
        ("commute_place", json!([3])),
        ("commute_work", json!([0])),
        ("commute_trips", json!([4000.])),
        ("everyday_home", json!([0])),
        ("everyday_place", json!([3])),
        ("everyday_zone", json!([0])),
        ("everyday_kind", json!([0])),
        ("everyday_trips", json!([3000.])),
        ("everyday_walk", json!([0.2])),
    ];
    for (k, x) in add {
        v[k] = x;
    }
    packs[0] = v.to_string();
    packs
}

/// `world_lines`' lines in places 0 and 1 and in place 2, and with `sea` a line from place 0's
/// centre to place 3 across the sea (111 km) taking `sea` seconds each way.
fn lines(a: &mut IntercityApi, sea: Option<f32>) {
    let mut ll = vec![0., 50., 0.15, 50., 1.4, 50., 1.5, 50.04];
    let mut n = vec![4u32];
    let mut stops = vec![0u32, 1, 2, 3];
    let mut times = vec![0f32; 24];
    for lev in 0..3 {
        for k in 0..3 {
            times[lev * 4 + k] = if k == 1 { 2400. } else { 300. };
            times[(3 + lev) * 4 + k + 1] = if k == 1 { 2400. } else { 300. };
        }
    }
    let mut tph = vec![6f32, 6., 6.];
    let mut cars = vec![8f32];
    let mut add = |pts: &[f64], s: f32| {
        let k = ll.len() as u32 / 2;
        ll.extend_from_slice(pts);
        n.push(2);
        stops.extend([k, k + 1]);
        times.extend([s, 0., s, 0., s, 0., 0., s, 0., s, 0., s]);
        tph.extend([6f32, 6., 6.]);
        cars.push(8.);
    };
    add(&[11., 50., 10.88, 50.02], 600.);
    if let Some(s) = sea {
        add(&[0., 50., 0., 49.], s);
    }
    a.set_network_ll(&ll, &n, &stops, &times, &tph, &cars).unwrap();
    let k = 2 * stops.len();
    a.set_fares(2., 0.1, &vec![20.; k]).unwrap();
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn load_bits(l: &Loads) -> [Vec<u32>; 7] {
    [&l.seg, &l.board, &l.alight, &l.rn_board, &l.rn_alight, &l.entry, &l.exit].map(|v| bits(v))
}
/// A solved period's figures that must not move, to the bit (its time and the walking left out).
fn period_bits(a: &IntercityApi, q: usize) -> (Vec<u64>, [Vec<u32>; 7], [Vec<u32>; 7], Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let s = a.per(q).unwrap();
    let head = [s.rail, s.trips, s.c_rail, s.c_trips, s.e_rail, s.e_trips, s.round as f64, s.rail_d.d, s.rail_d.d_km, s.rail_d.km].map(f64::to_bits).to_vec();
    (head, load_bits(&s.avg), load_bits(&s.c_avg), bits(&s.pair_rail), bits(&s.st), bits(&s.c_frac), bits(&s.e_frac))
}
/// Each pair's rail share of its trips this period.
fn share(a: &IntercityApi, pass: &Pass, i: usize) -> f64 {
    pass.pair_rail[i] as f64 / (a.pairs[i].trips * SHARES[0] as f64)
}
fn across(a: &IntercityApi, i: usize) -> bool {
    let (x, y) = (a.m.zones[a.pairs[i].a as usize].place, a.m.zones[a.pairs[i].b as usize].place);
    (x == 3) != (y == 3)
}

#[test]
fn island_pairs_ride_by_the_bound_and_every_other_pair_as_before() {
    let packs = island_packs();
    let world = composition(&packs, Some(&rows_json(&packs)));
    let open = |off: bool, sea: Option<f32>| {
        set_off(off);
        let mut a = IntercityApi::new(&world).unwrap();
        lines(&mut a, sea);
        solve_all(&mut a);
        set_off(false);
        a
    };
    // no line across the sea: the same solve to the bit, every period (but the errands' walkers
    // across the sea, who are gone)
    let (on, off) = (open(false, None), open(true, None));
    for q in 0..PERIODS {
        assert!(period_bits(&on, q) == period_bits(&off, q), "period {q}");
    }
    assert!(on.per(0).unwrap().e_walk < off.per(0).unwrap().e_walk);
    let e = on.m.everyday.iter().find(|e| on.m.gid_place[e.a as usize] == 3).unwrap();
    assert_eq!(e.walk, 0.);

    // a fast line across: the pairs across ride, all of them at the stations; every pair with a car
    // or a plane, and every commute and outing on one landmass, as without the rule to the bit
    for (sea, all_ride) in [(600., true), (3600., true), (72000., false)] {
        let (on, off) = (open(false, Some(sea)), open(true, Some(sea)));
        set_off(true);
        let per = off.per(0).unwrap();
        let (g, p, ix) = (&per.g, &per.p, &per.xi);
        let b = off.pass_on(g, p, 0, &skims_for(g, p, &off.want), ix);
        set_off(false);
        let a = on.pass_on(g, p, 0, &skims_for(g, p, &on.want), ix);
        assert!(off.pairs.len() < on.pairs.len() && (0..off.pairs.len()).all(|i| !across(&off, i)));
        for (i, pr) in off.pairs.iter().enumerate() {
            let j = on.pair_at(pr.a as usize, pr.b as usize) as usize;
            assert_eq!(a.pair_rail[j].to_bits(), b.pair_rail[i].to_bits(), "pair {} {}", pr.a, pr.b);
        }
        assert_eq!((on.cpairs.len(), on.epairs.len()), (off.cpairs.len(), off.epairs.len()));
        let road = |z: usize| on.m.places[on.m.zones[z].place].road;
        for (i, cp) in on.cpairs.iter().enumerate() {
            let (h, w, _) = on.m.commutes[cp.c as usize];
            assert_eq!(cp.apart, road(h as usize) != road(on.m.comp_of_gid[w as usize] as usize));
            if !cp.apart {
                assert_eq!(a.c_frac[i].to_bits(), b.c_frac[i].to_bits());
            }
        }
        for (i, ep) in on.epairs.iter().enumerate() {
            if !ep.apart {
                assert_eq!(a.e_frac[i].to_bits(), b.e_frac[i].to_bits());
            }
        }
        // the pairs across: rail by the bound, all of it up to 3 times the drive, none from 4
        let shares: Vec<f64> = (0..on.pairs.len()).filter(|&i| across(&on, i)).map(|i| share(&on, &a, i)).collect();
        let top = shares.iter().cloned().fold(0., f64::max);
        if all_ride {
            assert!((top - 1.).abs() < 1e-6, "sea {sea}: best share {top}");
        } else {
            assert_eq!(top, 0., "sea {sea}: {shares:?}");
        }
        // the commute and errands across: as the longer trips
        let ci = on.cpairs.iter().position(|c| c.apart).unwrap();
        let ei = on.epairs.iter().position(|e| e.apart).unwrap();
        if all_ride {
            assert!(a.c_frac[ci] == 1. && a.e_frac[ei] > 0.99, "commute {}, errands {}", a.c_frac[ci], a.e_frac[ei]);
        } else {
            assert_eq!((a.c_frac[ci], a.e_frac[ei]), (0., 0.));
        }
        // the trip explorer between the two stations' zones: the pass's share, and the cut-off's code
        let (z0, z3) = (on.m.places[0].off as i32, on.m.places[3].off as i32);
        let h = on.market_trip_between(0, WEEKDAY as u32, 0, z0, 0, 3, z3);
        let i = on.pair_at(on.m.comp_of_gid[z0 as usize] as usize, on.m.comp_of_gid[z3 as usize] as usize) as usize;
        let s_pass = on.per(0).unwrap().pair_rail[i] as f64 / (on.pairs[i].trips * SHARES[0] as f64);
        assert!((h[3] / h[2] - s_pass).abs() < 1e-4, "sea {sea}: explorer {} pass {s_pass}", h[3] / h[2]);
        assert!((h[3] / h[2] - p.bound_share(h[6] as f32, h[17] as f32, 0., 1.) as f64).abs() < 1e-3, "rail {} drive {}", h[6], h[17]);
        assert_eq!(h[14] == 8., !all_ride, "sea {sea}: code {}", h[14]);
        assert_eq!((h[4], h[5]), (0., 0.), "no plane, no car");
    }
    // a speed between: rail fades between 3 and 4 times the drive, as the explorer prices it
    let mut faded = false;
    for sea in [14400f32, 18000., 21600., 25200., 28800.] {
        let on = open(false, Some(sea));
        let (z0, z3) = (on.m.places[0].off as i32, on.m.places[3].off as i32);
        let h = on.market_trip_between(0, WEEKDAY as u32, 0, z0, 0, 3, z3);
        let s = h[3] / h[2];
        let q = h[6] / h[17];
        assert!((s - ((4. - q) as f64).clamp(0., 1.)).abs() < 1e-3, "sea {sea}: ratio {q}, share {s}");
        faded |= s > 0.01 && s < 0.99;
    }
    assert!(faded, "some speed between the two");
    // no network: no route, and with the rule off the old code
    for off in [false, true] {
        set_off(off);
        let mut a = IntercityApi::new(&world).unwrap();
        a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
        assert!(a.solve(0));
        let h = a.market_trip_between(0, WEEKDAY as u32, 0, -1, 0, 3, -1);
        assert_eq!(h[14], if off { 5. } else { 2. });
        set_off(false);
    }
}

/// A captured market network (`checks/t761-plain-net.mjs`), opened once from one read of the packs.
fn net_world(path: &str) -> (String, serde_json::Value) {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let ids: Vec<String> = j["markets"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let idr: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    (real_world(root, Some(&idr)), j)
}
fn net_on(world: &str, j: &serde_json::Value) -> IntercityApi {
    let f = |k: &str| j[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<f64>>();
    let mut a = IntercityApi::new(world).unwrap();
    let fr = &j["frame"];
    let frame = Frame::equirect(fr["lon"].as_f64().unwrap(), fr["lat"].as_f64().unwrap()).with_kind(crate::track::frame::FrameKind::Merc);
    let ll: Vec<f64> = f("stXY").chunks_exact(2).flat_map(|p| { let (x, y) = frame.to_lonlat(p[0], p[1]); [x, y] }).collect();
    let u = |k: &str| f(k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
    let g32 = |k: &str| f(k).into_iter().map(|v| v as f32).collect::<Vec<f32>>();
    a.set_complexes(&u("complex"));
    a.set_chunk(MARKET_CHUNK);
    a.set_network_ll(&ll, &u("lineN"), &u("stops"), &g32("times"), &g32("tph"), &g32("cars")).unwrap();
    a.set_fares(j["fare"]["base"].as_f64().unwrap() as f32, j["fare"]["perKm"].as_f64().unwrap() as f32, &g32("hopKm")).unwrap();
    a
}

/// The solve with the rule off and on, on a captured network: every period to the bit, and the
/// time of each (best of `TW_REPS`, alternating).
/// `TW_T780_NET=<plain net> TW_REPS=2 cargo test --release --features demand --lib t2216_ab -- --ignored --nocapture`
#[test]
#[ignore]
fn t2216_ab() {
    let path = std::env::var("TW_T780_NET").unwrap();
    let reps: usize = std::env::var("TW_REPS").unwrap_or("2".into()).parse().unwrap();
    let (world, j) = net_world(&path);
    let mut best = [[f64::INFINITY; PERIODS]; 2];
    let mut figs: [Vec<_>; 2] = [vec![], vec![]];
    for rep in 0..reps {
        for off in [true, false] {
            set_off(off);
            let mut a = net_on(&world, &j);
            let k = off as usize;
            if rep == 0 {
                let apart = a.cpairs.iter().filter(|c| c.apart).count() + a.epairs.iter().filter(|e| e.apart).count();
                let bounded = a.pairs.iter().filter(|p| a.m.pr(a.m.zones[p.a as usize].place, a.m.zones[p.b as usize].place).bounded()).count();
                eprintln!("{}: pairs {}, of them bounded {bounded}; commute and outing pairs apart {apart}", if off { "off" } else { "on" }, a.pairs.len());
            }
            for q in 0..PERIODS {
                let t = now_ms();
                assert!(a.solve(q));
                best[k][q] = best[k][q].min(now_ms() - t);
            }
            if rep == 0 {
                figs[k] = (0..PERIODS).map(|q| period_bits(&a, q)).collect();
                let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
                eprintln!("  rail all periods {rail:.3}");
            }
            set_off(false);
        }
    }
    for q in 0..PERIODS {
        eprintln!("period {q}: off {:.0} ms, on {:.0} ms, same to the bit: {}", best[1][q], best[0][q], figs[0][q] == figs[1][q]);
    }
    eprintln!("all periods: off {:.0} ms, on {:.0} ms", best[1].iter().sum::<f64>(), best[0].iter().sum::<f64>());
    assert!(figs[0] == figs[1], "not the same to the bit");
}

/// A world opened from the rows alone: the place pairs with no car and no plane, near and far, and
/// the island pairs the design names, with no network.
/// `cargo test --release --features demand --lib t2216_rows -- --ignored --nocapture`
#[test]
#[ignore]
fn t2216_rows() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let ids = ["van", "six", "sea", "sin", "bat", "tee", "gra"];
    let a = IntercityApi::new(&real_world(root, Some(&ids))).unwrap();
    let m = &a.m;
    let (rows, rp, rt, rc, rm) = (a.corridor_rows(), a.corridor_rows_pairs(), a.corridor_rows_trips(), a.corridor_rows_commutes(), a.corridor_rows_modes());
    let (mut near, mut far, mut cn, mut n_near, mut n_far) = (0f64, 0f64, 0f64, 0, 0);
    for i in 0..rows.len() {
        let (x, y) = (rp[2 * i] as usize, rp[2 * i + 1] as usize);
        if x == y || rm[2 * i + 1] != 0. {
            continue;
        }
        let c = rc.get(i).copied().unwrap_or(0.);
        cn += c;
        if rm[2 * i] > 0. {
            continue;
        }
        let km = chord_km(&m.places[x].u, &m.places[y].u);
        if km < m.rule.near_km {
            near += rt[i];
            n_near += 1;
        } else {
            far += rt[i];
            n_far += 1;
        }
    }
    eprintln!("no car, no plane: near {near:.0} trips a day over {n_near} place pairs, far {far:.0} over {n_far}; commute legs with no car {cn:.0}");
    let np = m.places.len();
    let s = a.table_summary(np as u32);
    let place = |id: &str| -> usize {
        let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
        reg["cities"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap()["place"].as_u64().unwrap() as usize
    };
    for (x, y) in [("van", "six"), ("sea", "six"), ("sin", "bat"), ("tee", "gra")] {
        let (px, py) = (place(x), place(y));
        let r = a.place_rows(&[px as u32]);
        for row in r.chunks_exact(7).filter(|v| v[1] as usize == px.min(py) && v[2] as usize == px.max(py)) {
            eprintln!("{x} - {y}: longer trips {:.0} a day, commute legs {:.0}, plane share {}, car share {}", row[3], row[4], row[5], row[6]);
        }
        eprintln!("  {x}: no way but rail, all its pairs: {:.0} of {:.0} trips and {:.0} commute legs", s[2 + 5 * np + px], s[2 + 3 * px], s[2 + 3 * px + 1]);
    }
}

/// Default ferries across the Salish Sea (the design's examples, T-3431): Vancouver - Victoria and
/// Seattle - Victoria, each 32 km/h on the water every 20 minutes at the starting fare ($2.50 +
/// $0.20 a km), stations at Vancouver's Waterfront, Victoria's Inner Harbour and Seattle's piers.
/// Their riders a day, and for the zones at the ports rail's cost over the drive and the fares at
/// which riders start to go and are all gone.
/// `cargo test --release --features demand --lib t2216_ferries -- --ignored --nocapture`
#[test]
#[ignore]
fn t2216_ferries() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let ids = ["van", "six", "sea"];
    let world = real_world(root, Some(&ids));
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let place = |id: &str| reg["cities"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap()["place"].as_u64().unwrap() as u32;
    let (van, vic, sea) = (place("van"), place("six"), place("sea"));
    // (lon, lat) and the boat's km on the water (the straight line x 1.15)
    let (p_van, p_vic, p_sea) = ([-123.1115, 49.2869], [-123.3707, 48.4222], [-122.3532, 47.6155]);
    let km = |a: [f64; 2], b: [f64; 2]| {
        let r = 6371.;
        let (la, lb) = (a[1].to_radians(), b[1].to_radians());
        let h = ((lb - la) / 2.).sin().powi(2) + la.cos() * lb.cos() * ((b[0] - a[0]).to_radians() / 2.).sin().powi(2);
        2. * r * h.sqrt().asin() * 1.15
    };
    let net = |base: f32, with_net: bool| -> IntercityApi {
        let mut a = IntercityApi::new(&world).unwrap();
        a.set_chunk(MARKET_CHUNK);
        if !with_net {
            a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
            return a;
        }
        let (k1, k2) = (km(p_van, p_vic), km(p_sea, p_vic));
        let ll = [p_van[0], p_van[1], p_vic[0], p_vic[1], p_sea[0], p_sea[1], p_vic[0], p_vic[1]];
        let s = |k: f64| (k / 32. * 3600.) as f32;
        let mut times = vec![];
        for k in [k1, k2] {
            let t = s(k);
            times.extend([t, 0., t, 0., t, 0., 0., t, 0., t, 0., t]);
        }
        a.set_network_ll(&ll, &[2, 2], &[0, 1, 2, 3], &times, &[3.; 6], &[8., 8.]).unwrap();
        let (k1, k2) = (k1 as f32, k2 as f32);
        a.set_fares(base, 0.2, &[k1, 0., k1, 0., k2, 0., k2, 0.]).unwrap();
        a
    };
    let run = |base: f32, with_net: bool| -> IntercityApi {
        let mut a = net(base, with_net);
        assert!(a.solve(0));
        a
    };
    // the solve with the rule off and on (period 0, best of three)
    let (mut best, mut pairs) = ([f64::INFINITY; 2], [0; 2]);
    for _ in 0..3 {
        for off in [true, false] {
            set_off(off);
            let mut a = net(2.5, true);
            let t = now_ms();
            assert!(a.solve(0));
            best[off as usize] = best[off as usize].min(now_ms() - t);
            pairs[off as usize] = a.pairs.len();
            set_off(false);
        }
    }
    eprintln!("solve(0): off {:.0} ms with {} pairs, on {:.0} ms with {} pairs", best[1], pairs[1], best[0], pairs[0]);
    let pair_line = |a: &IntercityApi, x: u32, y: u32| {
        let h = a.market_trip_between(0, WEEKDAY as u32, x, -1, 0, y, -1);
        let b = a.market_trip_between(0, WEEKDAY as u32, y, -1, 0, x, -1);
        (h[2] + b[2], h[3] + b[3], h[4] + b[4], h[5] + b[5], h[14], b[22] + h[22], b[23] + h[23])
    };
    // no network
    let a = run(2.5, false);
    for (n, x, y) in [("Vancouver - Saanich", van, vic), ("Seattle - Saanich", sea, vic)] {
        let (t, r, air, car, why, cl, cr) = pair_line(&a, x, y);
        eprintln!("no network, {n}, period 0 both ways: trips {t:.0}, rail {r:.0}, plane {air:.0}, car {car:.0}, code {why}; commute legs {cl:.0}, by rail {cr:.0}");
    }
    // the ferries at the starting fare
    let a = run(2.5, true);
    let share_q = SHARES[0] as f64;
    for (n, x, y) in [("Vancouver - Saanich", van, vic), ("Seattle - Saanich", sea, vic)] {
        let (t, r, air, car, why, cl, cr) = pair_line(&a, x, y);
        eprintln!("ferry, {n}, period 0 both ways: trips {t:.0}, rail {r:.0} ({:.1}%), plane {air:.0}, car {car:.0}, code {why}; a day about {:.0} rail of {:.0}; commute legs {cl:.0}, by rail {cr:.0}", 100. * r / t.max(1e-9), r / share_q, t / share_q);
    }
    eprintln!("rail all, period 0: {:.0}", a.summary(0)[2]);
    // the port zones: rail's cost against the drive, and the fares where riders start to go and are gone
    let port_zone = |pl: u32, p: [f64; 2]| -> i32 {
        let m = &a.m;
        let u = unit(p[0], p[1]);
        let pl = pl as usize;
        (m.places[pl].off..m.places[pl].off + m.places[pl].zones as u32).min_by(|&x, &y| {
            let (zx, zy) = (&m.zones[m.comp_of_gid[x as usize] as usize], &m.zones[m.comp_of_gid[y as usize] as usize]);
            chord_km(&zx.u, &u).total_cmp(&chord_km(&zy.u, &u))
        }).unwrap() as i32
    };
    for (n, x, px, y, py) in [("Vancouver - Saanich", van, p_van, vic, p_vic), ("Seattle - Saanich", sea, p_sea, vic, p_vic)] {
        let (zx, zy) = (port_zone(x, px), port_zone(y, py));
        let mut out = vec![];
        for base in [2.5f32, 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150.] {
            let b = run(base, true);
            let h = b.market_trip_between(0, WEEKDAY as u32, x, zx, 0, y, zy);
            out.push(format!("${base}: rail/drive {:.2} ({:.0}/{:.0}), share {:.3}", h[6] / h[17], h[6], h[17], h[3] / h[2].max(1e-12)));
            if h[3] <= 0. {
                break;
            }
        }
        eprintln!("port zones {n}:\n  {}", out.join("\n  "));
    }
}
