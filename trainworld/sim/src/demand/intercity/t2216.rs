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
    // the panels' figures with no line: no car and no plane between places 0 and 3, no way for the
    // trips and for every outing across (`everyday_place_legs`' 5th and 6th numbers: legs, rail on them)
    assert_eq!(on.m.corridor_mode(&Directs::of(&on.m.places), 0, 3), [0., 0.]);
    let legs = |a: &IntercityApi| (0..PERIODS).flat_map(|q| a.everyday_place_legs(q).chunks_exact(7).filter(|c| c[0] == 3.).map(|c| (c[1], c[5], c[6])).collect::<Vec<_>>()).collect::<Vec<_>>();
    let none = legs(&on);
    assert!(!none.is_empty() && none.iter().all(|&(all, apart, rail)| apart > 0. && apart <= all && rail == 0.), "{none:?}");

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
            let w = on.m.comp_of_gid[w as usize] as usize;
            assert_eq!(cp.apart, road(h as usize) != road(w) || on.m.zone_land(&on.m.zones[h as usize]) != on.m.zone_land(&on.m.zones[w]));
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
        // (what rail takes of the outings across is counted: all of them where all ride)
        let across = legs(&on);
        assert!(across.iter().all(|&(_, apart, rail)| rail <= apart + 1e-9) && across.iter().any(|&(_, apart, rail)| (rail > 0.) == all_ride && apart > 0.), "sea {sea}: {across:?}");
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

fn set_lands_off(off: bool) {
    ZONE_LANDS_OFF.with(|c| c.set(off));
}

/// `world_packs`, place 0's second zone (10 km east of its first) on an island of its own: a part
/// not reaching the grid's edge on the main part's world landmass, as Salamis. Its people commute
/// to place 1 and run errands to place 0's first zone (a third walking).
pub(super) fn zone_island_packs() -> Vec<String> {
    let mut packs = world_packs();
    let mut v: serde_json::Value = serde_json::from_str(&packs[0]).unwrap();
    let add = [
        ("zone_parts", json!([[[0, 1.0]], [[1, 1.0]], [[0, 1.0]]])),
        ("site_part", json!([0, 1, 0])),
        ("parts", json!({"people": [350000., 100000.], "jobs": [210000., 30000.], "open": [1, 0], "landmass": [0, 0]})),
        ("commute_home", json!([1])),
        ("commute_place", json!([1])),
        ("commute_work", json!([0])),
        ("commute_trips", json!([2000.])),
        ("everyday_home", json!([1])),
        ("everyday_place", json!([0])),
        ("everyday_zone", json!([0])),
        ("everyday_kind", json!([0])),
        ("everyday_trips", json!([1500.])),
        ("everyday_walk", json!([0.3])),
    ];
    for (k, x) in add {
        v[k] = x;
    }
    packs[0] = v.to_string();
    packs
}

/// T-2216 stage 2: a zone off its pack's main land has no car to any other zone, its pairs ride by
/// the bound (no plane between them); every pair of two zones on one land as before, to the bit.
#[test]
fn zones_off_their_main_land_ride_by_the_bound() {
    let packs = zone_island_packs();
    let world = composition(&packs, Some(&rows_json(&packs)));
    let open = |off: bool, net: bool| {
        set_lands_off(off);
        let mut a = IntercityApi::new(&world).unwrap();
        if net {
            lines(&mut a, None);
        } else {
            a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
        }
        solve_all(&mut a);
        set_lands_off(false);
        a
    };
    let (on, off) = (open(false, true), open(true, true));
    let isle = on.m.comp_of_gid[on.m.places[0].off as usize + 1] as usize;
    assert!(on.m.zone_land(&on.m.zones[isle]) != on.m.zone_land(&on.m.zones[isle - 1]));
    assert_eq!(on.m.zone_land(&on.m.zones[isle - 1]), on.m.zone_land(&on.m.zones[isle + 1]));
    let touches = |a: &IntercityApi, i: usize| a.pairs[i].a as usize == isle || a.pairs[i].b as usize == isle;
    // the station 0.7 km from the island's site is the island's: only the island reaches it, and
    // the island no other
    let st_isle = 1;
    for (z, list) in on.access.iter().chain(&on.access_work).enumerate().map(|(k, l)| (k % on.m.zones.len(), l)) {
        assert!(list.iter().all(|e| (e.0 == st_isle) == (z == isle)), "zone {z}: {list:?}");
    }
    // in a single pass, every pair not touching the island whose zones reach the same stations as
    // with the rule off to the bit
    let per = off.per(0).unwrap();
    let (g, p, ix) = (&per.g, &per.p, &per.xi);
    set_lands_off(true);
    let b = off.pass_on(g, p, 0, &skims_for(g, p, &off.want), ix);
    set_lands_off(false);
    let a = on.pass_on(g, p, 0, &skims_for(g, p, &on.want), ix);
    let same_access = |z: usize| on.access[z] == off.access[z] && on.access_work[z] == off.access_work[z];
    let (mut n_isle, mut n_same) = (0, 0);
    for i in 0..on.pairs.len() {
        let (za, zb) = (on.pairs[i].a as usize, on.pairs[i].b as usize);
        let j = off.pair_at(za, zb);
        if touches(&on, i) {
            n_isle += 1;
            let r = on.m.zone_rule(&on.m.zones[za], &on.m.zones[zb]);
            assert!(!r.car && r.bounded(), "pair {i}");
            // a short ride against a drive over a bridge: everyone rides
            assert!((share(&on, &a, i) - 1.).abs() < 1e-6, "pair {i}: on {}", share(&on, &a, i));
            if j != u32::MAX {
                assert!(share(&off, &b, j as usize) < 1., "pair {i}: off {}", share(&off, &b, j as usize));
            }
        } else if same_access(za) && same_access(zb) {
            n_same += 1;
            assert_eq!(a.pair_rail[i].to_bits(), b.pair_rail[j as usize].to_bits(), "pair {i}");
        }
    }
    assert!(n_isle >= 2 && n_same > 0, "the island's pairs with place 1 ({n_isle}), others the same ({n_same})");
    // its commute and errands: apart, no walkers, all ride
    let ci = on.cpairs.iter().position(|c| on.m.commutes[c.c as usize].0 as usize == isle).unwrap();
    let ei = on.epairs.iter().position(|e| on.m.everyday[e.e as usize].h as usize == isle).unwrap();
    let cj = off.cpairs.iter().position(|c| off.m.commutes[c.c as usize].0 as usize == isle).unwrap();
    let ej = off.epairs.iter().position(|e| off.m.everyday[e.e as usize].h as usize == isle).unwrap();
    assert!(on.cpairs[ci].apart && on.epairs[ei].apart && !off.cpairs[cj].apart && !off.epairs[ej].apart);
    assert_eq!(on.m.everyday[on.epairs[ei].e as usize].walk, 0.);
    assert!(off.m.everyday[off.epairs[ej].e as usize].walk > 0.29);
    assert!(a.c_frac[ci] == 1. && a.e_frac[ei] > 0.99, "commute {} errands {}", a.c_frac[ci], a.e_frac[ei]);
    // the panels' figures: the island's share of place 0's trips with place 1 has no car (and no plane
    // here, so no way but rail), and so has its share of the place's own row
    let (d_on, d_off) = (Directs::of(&on.m.places), Directs::of(&off.m.places));
    set_lands_off(true);
    let m_off = off.m.corridor_mode(&d_off, 0, 1);
    let off_empty = off.m.apart().own.is_empty() && off.m.apart().cross.is_empty();
    set_lands_off(false);
    let m_on = on.m.corridor_mode(&d_on, 0, 1);
    assert!(m_off[1] > 0.99 && m_on[1] > 0. && m_on[1] < m_off[1] - 1e-6, "car share with place 1: off {m_off:?} on {m_on:?}");
    assert!(m_on[0] == 0. && m_on[0] + m_on[1] < 1., "no plane: the rest has no way, {m_on:?}");
    assert!(on.m.apart().own.get(&0).is_some_and(|&f| f > 0. && f <= 1.), "own row share {:?}", on.m.apart().own);
    assert!(off_empty);
    // the trip explorer: from the island no car to place 1 or to its own place's first zone; from
    // that first zone the car as before
    let none = open(false, false);
    let (z0, z1) = (none.m.places[0].off as i32, none.m.places[0].off as i32 + 1);
    let h = none.market_trip_between(0, WEEKDAY as u32, 0, z1, 0, 1, -1);
    assert!(h[2] > 0. && h[5] == 0. && h[4] == 0. && h[14] == 2., "island to place 1: trips {} car {} code {}", h[2], h[5], h[14]);
    // (its third zone, 30 km off: the longer trips start at 25 km)
    let h = none.market_trip_between(0, WEEKDAY as u32, 0, z1, 0, 0, z0 + 2);
    assert!(h[2] > 0. && h[5] == 0., "island to its own mainland: trips {} car {}", h[2], h[5]);
    let h = none.market_trip_between(0, WEEKDAY as u32, 0, z0, 0, 0, z0 + 2);
    assert!(h[2] > 0. && h[5] > 0., "within the mainland: trips {} car {}", h[2], h[5]);
    let h = none.market_trip_between(0, WEEKDAY as u32, 0, z0, 0, 1, -1);
    assert!(h[5] > 0. && (h[5] - h[2]).abs() < 1e-9 * h[2], "mainland to place 1 drives: trips {} car {}", h[2], h[5]);
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

/// T-2216 stage 2 on a captured network, the zones' own lands off and on: the zones off their pack's
/// main land; in one pass of period 0 every pair not touching one to the bit, and the rail of those
/// that do; then all periods solved, the rail of each and, by the place of the zone off its main
/// land, the longer trips' rail a day on its pairs.
/// `TW_T780_NET=<plain net> cargo test --profile agent --features demand --lib t2216_zones_ab -- --ignored --nocapture`
#[test]
#[ignore]
fn t2216_zones_ab() {
    let path = std::env::var("TW_T780_NET").unwrap();
    let (world, j) = net_world(&path);
    let open = |off: bool| {
        set_lands_off(off);
        let mut a = net_on(&world, &j);
        let t = now_ms();
        for q in 0..PERIODS {
            assert!(a.solve(q));
        }
        let ms = now_ms() - t;
        set_lands_off(false);
        (a, ms)
    };
    let ((mut on, mut ms_on), (mut off, mut ms_off)) = (open(true), open(false));
    std::mem::swap(&mut on, &mut off);
    std::mem::swap(&mut ms_on, &mut ms_off);
    // (the times: best of `TW_REPS`, off and on alternating)
    let reps: usize = std::env::var("TW_REPS").unwrap_or("2".into()).parse().unwrap();
    for _ in 1..reps {
        ms_off = ms_off.min(open(true).1);
        ms_on = ms_on.min(open(false).1);
    }
    let m = &on.m;
    let isle = |z: usize| m.zones[z].land != MAIN_LAND;
    let (n_isle, pop_isle) = (0..m.zones.len()).filter(|&z| isle(z)).fold((0, 0.), |(n, p), z| (n + 1, p + m.zones[z].pop));
    eprintln!("open zones {}, off their main land {n_isle} ({pop_isle:.0} people); pairs {} (lands off {})", m.zones.len(), on.pairs.len(), off.pairs.len());
    let touches = |i: usize| isle(on.pairs[i].a as usize) || isle(on.pairs[i].b as usize);
    let same_access = |z: usize| on.access[z] == off.access[z] && on.access_work[z] == off.access_work[z];
    let lost: Vec<usize> = (0..m.zones.len()).filter(|&z| !isle(z) && !same_access(z)).collect();
    eprintln!("zones on their main land whose stations changed: {} ({:.0} people)", lost.len(), lost.iter().map(|&z| m.zones[z].pop).sum::<f64>());
    {
        let (net, anchor) = (&off.per(0).unwrap().net, m.anchor.unwrap());
        let stl = m.station_lands(net);
        for &z in lost.iter().take(20) {
            let zz = &m.zones[z];
            let (lon, lat) = lonlat_of(&zz.u);
            let gone: Vec<String> = off.access[z].iter().chain(&off.access_work[z]).filter(|e| !on.access[z].iter().chain(&on.access_work[z]).any(|f| f.0 == e.0)).take(3).map(|e| {
                let (x, y) = anchor.to_lonlat(net.st_x[e.0] as f64, net.st_y[e.0] as f64);
                format!("st {} ({x:.4},{y:.4}) land {:?} w {:.3}", e.0, stl[e.0], e.1)
            }).collect();
            eprintln!("  main zone {z} of {} ({lon:.4},{lat:.4}, {:.0} people, land {}) lost {}", m.places[zz.place].zone, zz.pop, m.zone_land(zz), gone.join("; "));
        }
    }
    // one pass of period 0 on the lands-off solve's graph
    let per = off.per(0).unwrap();
    let (g, p, ix) = (&per.g, &per.p, &per.xi);
    set_lands_off(true);
    let b = off.pass_on(g, p, 0, &skims_for(g, p, &off.want), ix);
    set_lands_off(false);
    let a = on.pass_on(g, p, 0, &skims_for(g, p, &on.want), ix);
    let (mut moved, mut same, mut n_t, mut r_on, mut r_off) = (0, 0, 0, 0f64, 0f64);
    for i in 0..on.pairs.len() {
        let (za, zb) = (on.pairs[i].a as usize, on.pairs[i].b as usize);
        let j = off.pair_at(za, zb);
        if touches(i) {
            n_t += 1;
            r_on += a.pair_rail[i] as f64;
        } else if same_access(za) && same_access(zb) {
            same += 1;
            if j == u32::MAX || a.pair_rail[i].to_bits() != b.pair_rail[j as usize].to_bits() {
                moved += 1;
                if moved <= 6 {
                    let pl = |z: usize| m.places[m.zones[z].place].zone.clone();
                    eprintln!("  moved: zones {za} ({}) - {zb} ({}): on {} off {}", pl(za), pl(zb), a.pair_rail[i], if j == u32::MAX { f32::NAN } else { b.pair_rail[j as usize] });
                }
            }
        }
    }
    for j in 0..off.pairs.len() {
        if isle(off.pairs[j].a as usize) || isle(off.pairs[j].b as usize) {
            r_off += b.pair_rail[j] as f64;
        }
    }
    eprintln!("one pass, period 0: pairs touching a zone off its main land {n_t}, their rail off {r_off:.1} on {r_on:.1}; other pairs with the same stations {same}, of them not to the bit: {moved}");
    let (c_on, c_off) = (on.cpairs.iter().filter(|c| c.apart).count(), off.cpairs.iter().filter(|c| c.apart).count());
    let (e_on, e_off) = (on.epairs.iter().filter(|e| e.apart).count(), off.epairs.iter().filter(|e| e.apart).count());
    eprintln!("commute pairs apart: off {c_off} on {c_on}; outing pairs apart: off {e_off} on {e_on}");
    // all periods
    let (mut all_on, mut all_off) = (0f64, 0f64);
    for q in 0..PERIODS {
        let (x, y) = (on.summary(q)[2], off.summary(q)[2]);
        all_on += x;
        all_off += y;
        eprintln!("period {q}: rail off {y:.1} on {x:.1} ({:+.4}%)", 100. * (x - y) / y.max(1e-9));
    }
    eprintln!("all periods: rail off {all_off:.1} on {all_on:.1} ({:+.4}%); solve {ms_off:.0} ms off, {ms_on:.0} ms on", 100. * (all_on - all_off) / all_off.max(1e-9));
    // by place of the zone off its main land: longer trips a day on its pairs, and their rail
    let mut by: std::collections::BTreeMap<String, (f64, f64, f64)> = Default::default();
    for i in (0..on.pairs.len()).filter(|&i| touches(i)) {
        let pr = &on.pairs[i];
        let z = if isle(pr.a as usize) { pr.a } else { pr.b } as usize;
        let e = by.entry(m.places[m.zones[z].place].zone.clone()).or_default();
        let j = off.pair_at(pr.a as usize, pr.b as usize);
        for q in 0..PERIODS {
            e.0 += pr.trips * SHARES[q] as f64;
            if j != u32::MAX {
                e.1 += off.per(q).unwrap().pair_rail[j as usize] as f64;
            }
            e.2 += on.per(q).unwrap().pair_rail[i] as f64;
        }
    }
    for (k, (t, x, y)) in &by {
        if *t > 0. {
            eprintln!("  {k}: longer trips a day on pairs off the main land {t:.0}, rail off {x:.0} on {y:.0}");
        }
    }
    // network set with lands off and on (`set_access` holds `station_lands`), best of 3
    let mut set_ms = [f64::INFINITY; 2];
    for _ in 0..3 {
        for off_ in [true, false] {
            set_lands_off(off_);
            let t = now_ms();
            let _a = net_on(&world, &j);
            set_ms[off_ as usize] = set_ms[off_ as usize].min(now_ms() - t);
            set_lands_off(false);
        }
    }
    eprintln!("opened and network set: off {:.0} ms, on {:.0} ms (best of 3)", set_ms[1], set_ms[0]);
    {
        let net = &on.per(0).unwrap().net;
        let t = now_ms();
        let _ = m.station_lands(net);
        eprintln!("station_lands alone: {:.0} ms for {} stations; sites {}, of them kept off the main part {}", now_ms() - t, net.st_x.len(), m.n_sites(), m.site_land.len());
    }
    // each station's nearest site and nearest site on another land (within 3 km), the close calls
    {
        let (net, anchor) = (&on.per(0).unwrap().net, m.anchor.unwrap());
        let mut close = vec![];
        for s in 0..net.st_x.len() {
            let (lon, lat) = anchor.to_lonlat(net.st_x[s] as f64, net.st_y[s] as f64);
            let u = unit(lon, lat);
            let (mut best, mut other) = ((f64::INFINITY, MAIN_LAND), f64::INFINITY);
            let mut seen: Vec<(f64, i64)> = vec![];
            for g in &m.groups {
                let Some(f) = g.frame else { continue };
                let (x, y) = f.from_lonlat(lon, lat);
                let (x, y) = (x as f32, y as f32);
                if g.city.x.is_empty() {
                    continue;
                }
                let lo = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = |v: &[f32]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if x < lo(&g.city.x) - 3000. || x > hi(&g.city.x) + 3000. || y < lo(&g.city.y) - 3000. || y > hi(&g.city.y) + 3000. {
                    continue;
                }
                for i in 0..g.city.len() {
                    let d = (((g.city.x[i] - x).powi(2) + (g.city.y[i] - y).powi(2)) as f64).sqrt();
                    if d < 3000. {
                        seen.push((d, m.site_land_of(g.lo + i)));
                    }
                }
            }
            for &(d, l) in &seen {
                if d < best.0 {
                    best = (d, l);
                }
            }
            for &(d, l) in &seen {
                if l != best.1 {
                    other = other.min(d);
                }
            }
            let _ = u;
            if other.is_finite() && (other - best.0 < 600. || s == 1854) {
                close.push(format!("st {s} ({lon:.4},{lat:.4}): nearest site {:.0} m (land {}), other land {:.0} m", best.0, best.1, other));
            }
        }
        eprintln!("stations with a site on another land under 600 m farther than the nearest: {}", close.len());
        for c in close.iter().take(30) {
            eprintln!("  {c}");
        }
    }
    // the rail a day by place, lands off and on (either end's place, the longer trips)
    {
        let mut by: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
        for (k, a) in [&off, &on].into_iter().enumerate() {
            for (i, pr) in a.pairs.iter().enumerate() {
                let r: f64 = (0..PERIODS).map(|q| a.per(q).unwrap().pair_rail[i] as f64).sum();
                let e = by.entry(m.places[m.zones[pr.a as usize].place].zone.clone()).or_default();
                if k == 0 { e.0 += r } else { e.1 += r }
            }
        }
        let mut v: Vec<_> = by.into_iter().map(|(k, (x, y))| (y - x, k, x, y)).collect();
        v.sort_by(|a, b| a.0.abs().total_cmp(&b.0.abs()).reverse());
        for (d, k, x, y) in v.iter().take(12) {
            eprintln!("  longer trips' rail a day from {k}: off {x:.0} on {y:.0} ({d:+.0})");
        }
    }
    // the zones off the main land whose pairs ride most, and the stations they reach
    let (net, anchor) = (&on.per(0).unwrap().net, m.anchor.unwrap());
    let st: Vec<(f64, f64)> = (0..net.st_x.len()).map(|s| anchor.to_lonlat(net.st_x[s] as f64, net.st_y[s] as f64)).collect();
    let mut by_zone: std::collections::BTreeMap<usize, f64> = Default::default();
    for i in (0..on.pairs.len()).filter(|&i| touches(i)) {
        let pr = &on.pairs[i];
        let z = if isle(pr.a as usize) { pr.a } else { pr.b } as usize;
        *by_zone.entry(z).or_default() += (0..PERIODS).map(|q| on.per(q).unwrap().pair_rail[i] as f64).sum::<f64>();
    }
    let mut top: Vec<(usize, f64)> = by_zone.into_iter().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    for &(z, r) in top.iter().take(12) {
        let zz = &m.zones[z];
        let (lon, lat) = lonlat_of(&zz.u);
        let acc: Vec<String> = on.access[z].iter().take(4).map(|&(s, w)| { let (x, y) = st[s]; format!("st {s} ({x:.4},{y:.4}) {:.1} km w {w:.3}", chord_km(&zz.u, &unit(x, y))) }).collect();
        eprintln!("  zone {z} of {} ({lon:.4},{lat:.4}, {:.0} people, land {}): rail a day {r:.0}; reaches {}", m.places[zz.place].zone, zz.pop, zz.land, acc.join("; "));
    }
}

/// T-2216 stage 2, real places: Sado (Niigata's pack) with Tokyo, Salamis and Aegina (Athens's), with
/// no network and with a default boat (32 km/h on the water, the straight line x 1.15, every 20
/// minutes, $2.50 + $0.20 a km), Sado's boat meeting a train to Tokyo (100 minutes, 3 an hour). For
/// the island's zones: their longer trips with the place named in period 0, how many go by car and
/// by rail, the zones' own lands off and on.
/// `cargo test --profile agent --features demand --lib t2216_zones_places -- --ignored --nocapture`
#[test]
#[ignore]
fn t2216_zones_places() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let place = |id: &str| reg["cities"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap()["place"].as_u64().unwrap() as u32;
    let km = |a: [f64; 2], b: [f64; 2]| {
        let (la, lb) = (a[1].to_radians(), b[1].to_radians());
        let h = ((lb - la) / 2.).sin().powi(2) + la.cos() * lb.cos() * ((b[0] - a[0]).to_radians() / 2.).sin().powi(2);
        2. * 6371. * h.sqrt().asin() * 1.15
    };
    // (name, packs, island's place, ports, train on from the mainland port, far place)
    let (ryotsu, niigata, tokyo) = ([138.4385, 38.0800], [139.0586, 37.9283], [139.7671, 35.6812]);
    let (paloukia, perama) = ([23.4795, 37.9615], [23.5640, 37.9640]);
    let (aegina, piraeus) = ([23.4285, 37.7460], [23.6400, 37.9425]);
    let cases: [(&str, &[&str], &str, [f64; 2], [f64; 2], Option<[f64; 2]>, &str); 4] = [
        ("Sado - Tokyo", &["nig", "tyo"], "nig", ryotsu, niigata, Some(tokyo), "tyo"),
        ("Sado - Niigata", &["nig", "tyo"], "nig", ryotsu, niigata, Some(tokyo), "nig"),
        ("Salamis - Athens", &["ath"], "ath", paloukia, perama, None, "ath"),
        ("Aegina - Athens", &["ath"], "ath", aegina, piraeus, None, "ath"),
    ];
    for (name, ids, home, port_i, port_m, train, far) in cases {
        let world = real_world(root, Some(ids));
        let (pi, pf) = (place(home), place(far));
        for off in [true, false] {
            for with_net in [false, true] {
                set_lands_off(off);
                let mut a = IntercityApi::new(&world).unwrap();
                a.set_chunk(MARKET_CHUNK);
                if with_net {
                    let k = km(port_i, port_m);
                    let s = (k / 32. * 3600.) as f32;
                    let mut ll = vec![port_i[0], port_i[1], port_m[0], port_m[1]];
                    let mut n = vec![2u32];
                    let mut stops = vec![0u32, 1];
                    let mut times = vec![s, 0., s, 0., s, 0., 0., s, 0., s, 0., s];
                    let mut tph = vec![3f32; 3];
                    let mut cars = vec![8f32];
                    let mut hop = vec![k as f32, 0., k as f32, 0.];
                    if let Some(t) = train {
                        ll.extend_from_slice(&t);
                        n.push(2);
                        stops.extend([1, 2]);
                        let s = 6000f32;
                        times.extend([s, 0., s, 0., s, 0., 0., s, 0., s, 0., s]);
                        tph.extend([3f32; 3]);
                        cars.push(8.);
                        let k = (km(port_m, t) / 1.15 * 1.2) as f32;
                        hop.extend([k, 0., k, 0.]);
                    }
                    a.set_network_ll(&ll, &n, &stops, &times, &tph, &cars).unwrap();
                    a.set_fares(2.5, 0.2, &hop).unwrap();
                } else {
                    a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
                }
                assert!(a.solve(0));
                let m = &a.m;
                let pl = &m.places[pi as usize];
                // the island: the zones off the main land nearest the island's port
                let u = unit(port_i[0], port_i[1]);
                let isle: Vec<usize> = (pl.off..pl.off + pl.zones).map(|g| m.comp_of_gid[g as usize] as usize).filter(|&z| z != u32::MAX as usize).collect();
                let near_land = isle.iter().copied().min_by(|&x, &y| chord_km(&m.zones[x].u, &u).total_cmp(&chord_km(&m.zones[y].u, &u))).map(|z| m.zones[z].land).unwrap();
                let isle: Vec<usize> = isle.into_iter().filter(|&z| m.zones[z].land == near_land).collect();
                // longer trips, rail, car; outings legs, rail, walking; commute legs, rail
                let mut s = [0f64; 8];
                let mut pop = 0.;
                for &z in &isle {
                    pop += m.zones[z].pop;
                    let h = a.market_trip_between(0, WEEKDAY as u32, pi, m.zones[z].gid as i32, 0, pf, -1);
                    for (k, i) in [2, 3, 5, 19, 20, 21, 22, 23].into_iter().enumerate() {
                        s[k] += h[i];
                    }
                }
                set_lands_off(false);
                eprintln!(
                    "{name}: lands {}, {}: island zones {} ({pop:.0} people, land {}), period 0 with {far}: longer trips {:.0}, rail {:.0}, car {:.0}; outing legs {:.0}, rail {:.0}, walking {:.0}, rest {:.0}; commute legs {:.0}, rail {:.0}, rest {:.0}",
                    if off { "off" } else { "on" },
                    if with_net { "boat" } else { "no network" },
                    isle.len(),
                    if near_land == MAIN_LAND { "main".into() } else { near_land.to_string() },
                    s[0], s[1], s[2], s[3], s[4], s[5], s[3] - s[4] - s[5], s[6], s[7], s[6] - s[7]
                );
            }
        }
    }
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
