use super::*;
use crate::demand::network::{route_nodes, RealNet};

#[test]
fn intercity_prunes_symmetric_pairs_after_averaging() {
    // Directed departures are .03 and .0015: only one passes the old threshold.
    let keep = IntercityApi::new(r#"{"sites":[[0,0,2,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
    let pairs = keep.all_pairs();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].trips, pairs[1].trips);
    assert!((pairs[0].trips - 0.01575).abs() < 1e-12);
    assert!((keep.m.cw.daily - 2. * 0.01575).abs() < 1e-12);
    assert_eq!((pairs[0].a, pairs[0].b), (pairs[1].b, pairs[1].a));
    // .015 and .0015 average below .01: both directions must disappear.
    let drop = IntercityApi::new(r#"{"sites":[[0,0,1,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
    assert!(drop.all_pairs().is_empty());
    assert_eq!(drop.m.cw.daily, 0.);
}
#[test]
fn pair_at_finds_every_pair() {
    let mut a = api();
    let n = a.m.zones.len();
    a.set_served((0..n).collect(), vec![vec![0u32]; n]);
    assert!(!a.pairs.is_empty());
    for x in 0..n {
        for y in 0..n {
            let want = a.pairs.iter().position(|p| (p.a as usize, p.b as usize) == (x, y)).map_or(u32::MAX, |i| i as u32);
            assert_eq!(a.pair_at(x, y), want);
        }
    }
}
fn api() -> IntercityApi {
    let header = r#"{"zone_m":6000,"trip_rate":0.015,"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]]}"#;
    IntercityApi::new(header).unwrap()
}
fn network(api: &mut IntercityApi, freq: f32, slow: f32) {
    // Local feeder at each end, through intercity line, Mercer intermediate stop.
    let xy = [0., 0., 1000., 0., 60000., 0., 119000., 0., 120000., 0.];
    let n = [2, 3, 2];
    let stops = [0, 1, 1, 2, 3, 3, 4];
    let mut times = vec![0.; 42];
    for (a, len, sec) in [(0, 2, 120.), (2, 3, 1800. * slow), (5, 2, 120.)] {
        for lev in 0..3 {
            for k in 0..len - 1 {
                times[6 * a + lev * len + k] = sec;
                times[6 * a + (3 + lev) * len + k + 1] = sec;
            }
        }
    }
    api.set_network(&xy, &n, &stops, &times, &[12., 12., 12., freq, freq, freq, 12., 12., 12.], &[4., 4., 4.]).unwrap();
    api.set_fares(0., 0., &[1., 0., 0., 1., 59., 59., 0., 0., 59., 59., 1., 0., 0., 1.]).unwrap();
}
#[test]
fn intercity_market_is_symmetric_and_conserved() {
    let a = api();
    assert!((a.m.cw.daily - 3600.).abs() < 1e-5);
    let pairs = a.all_pairs();
    assert!((a.m.cw.daily - pairs.iter().map(|p| p.trips).sum::<f64>()).abs() < 1e-9 * a.m.cw.daily);
    for p in &pairs {
        let reverse = pairs.iter().find(|q| q.a == p.b && q.b == p.a).unwrap();
        assert_eq!(p.trips, reverse.trips);
    }
    assert!((SHARES.iter().sum::<f32>() - 1.).abs() < 1e-6);
}
/// T-1008: the corridors sent once (pairs, trips a day) and per period (shares, the pairs with
/// rail) give `corridor_summary` and `commute_corridors` back.
#[test]
fn corridor_parts_rebuild_the_summary() {
    let mut a = api();
    network(&mut a, 6., 1.);
    for q in [0, 2] {
        a.solve(q);
    }
    let (pairs, trips, base) = (a.corridor_pairs(), a.corridor_trips(), a.commute_base());
    assert_eq!(pairs.len(), 2 * trips.len());
    for q in 0..SHARES.len() {
        let full = a.corridor_summary(q);
        let rail = a.corridor_rail(q);
        if full.is_empty() {
            assert!(a.corridor_shares(q).is_empty() && rail.is_empty());
            continue;
        }
        let s = a.corridor_shares(q);
        let mut railed = vec![0.; trips.len()];
        for v in rail.chunks_exact(3) {
            railed[v[0] as usize] = v[1];
        }
        assert!(rail.len() >= 3, "period {q} has rail");
        for (i, v) in full.chunks_exact(4).enumerate() {
            assert_eq!([v[0], v[1]], [pairs[2 * i] as f64, pairs[2 * i + 1] as f64]);
            assert_eq!(v[2], trips[i] * s[0]);
            assert_eq!(v[3], railed[i]);
        }
        let k = a.commute_corridors(q);
        assert_eq!(k.len(), 2 * base.len());
        for (i, v) in k.chunks_exact(2).enumerate() {
            assert_eq!(v[0], base[i] * s[1]);
        }
    }
}
#[test]
fn intercity_uses_feeders_fares_service_and_shared_crowding() {
    let mut a = api();
    network(&mut a, 6., 1.);
    a.solve(0);
    let good = a.summary(0)[2];
    assert!(good > 100.);
    let board = a.board(0);
    assert!(board[0] > 0. && board[5] > 0., "local feeder boardings: {board:?}");
    assert!((a.entries(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
    assert!((a.exits(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
    let sum = a.corridor_summary(0).chunks_exact(4).map(|v| v[3]).sum::<f64>();
    assert!((sum - good).abs() < 1e-6);
    let loads = vec![1e7; a.seg(0).len()];
    assert!(a.crowd_with_external(0, &loads));
    assert!(a.summary(0)[2] < good);
    a.set_fares(80., 0., &[0.; 14]).unwrap();
    a.solve(0);
    assert!(a.summary(0)[2] < good * 0.1);
    network(&mut a, 6., 3.);
    a.solve(0);
    assert!(a.summary(0)[2] < good);
    network(&mut a, 0., 1.);
    a.solve(0);
    assert_eq!(a.summary(0)[2], 0.);
}
#[test]
fn intercity_access_respects_home_and_destination_weights() {
    let header = r#"{"sites":[[0,0,100000,0,0],[4000,0,0,100000,0],[100000,0,100000,0,1],[104000,0,0,100000,1]]}"#;
    let mut a = IntercityApi::new(header).unwrap();
    let mut times = vec![0.; 24];
    for lev in 0..3 {
        for k in 0..3 {
            let sec = if k == 1 { 2400. } else { 120. };
            times[lev * 4 + k] = sec;
            times[(3 + lev) * 4 + k + 1] = sec;
        }
    }
    a.set_network(&[0., 0., 4000., 0., 100000., 0., 104000., 0.], &[4], &[0, 1, 2, 3], &times, &[6., 6., 6.], &[4.]).unwrap();
    a.solve(0);
    let e = a.entries(0);
    let x = a.exits(0);
    assert!(e[0] > e[1], "origins follow homes: {e:?}");
    assert!(x[1] > x[0], "destinations favour jobs: {x:?}");
}
/// T-409: two lines with no track between them, 300 km apart, on one network or each alone.
fn two_lines(api: &mut IntercityApi, which: [bool; 2]) {
    let (mut xy, mut n, mut stops, mut times, mut tph) = (vec![], vec![], vec![], vec![], vec![]);
    for (i, x0) in [0f32, 300000.].into_iter().enumerate().filter(|(i, _)| which[*i]) {
        let s = xy.len() as u32 / 2;
        xy.extend([x0, 0., x0 + 60000., 0.]);
        n.push(2);
        stops.extend([s, s + 1]);
        let mut t = [0f32; 12];
        for lev in 0..3 {
            t[lev * 2] = 1800. + 300. * i as f32;
            t[(3 + lev) * 2 + 1] = 1800. + 300. * i as f32;
        }
        times.extend(t);
        tph.extend([6f32, 6., 6.]);
    }
    let cars = vec![4f32; n.len()];
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    api.set_fares(1., 0.1, &vec![30.; 2 * stops.len()]).unwrap();
}
#[test]
fn markets_beside_split_by_network_and_pairs_need_track() {
    let header = r#"{"zone_m":6000,"trip_rate":0.015,"sites":[[0,0,100000,50000,0],[60000,0,80000,40000,1],[300000,0,90000,50000,2],[360000,0,70000,30000,2]]}"#;
    let solved = |a: &mut IntercityApi| {
        for q in 0..PERIODS {
            assert!(a.solve(q));
            let big = vec![3e5f32; 3 * a.seg(q).len()];
            assert!(a.crowd_with_external(q, &big));
        }
    };
    let mut both = IntercityApi::new(header).unwrap();
    two_lines(&mut both, [true, true]);
    solved(&mut both);
    let mut parts = [both.beside(), both.beside()];
    for (k, p) in parts.iter_mut().enumerate() {
        two_lines(p, [k == 0, k == 1]);
        solved(p);
    }
    // a part beside equals a market opened alone on the same network, to the bit
    let mut alone = IntercityApi::new(header).unwrap();
    two_lines(&mut alone, [false, true]);
    solved(&mut alone);
    for q in 0..PERIODS {
        assert_eq!(parts[1].summary(q)[1..4], alone.summary(q)[1..4]);
        assert_eq!(parts[1].seg(q), alone.seg(q));
        assert_eq!(parts[1].corridor_summary(q), alone.corridor_summary(q));
    }
    assert_eq!(parts[1].station_flows(0), alone.station_flows(0));
    // every zone is served on the joint network, but the pairs across have no track: none built
    assert_eq!(both.served.len(), 4);
    let (p0, p1) = (parts[0].pair_counts()[0], parts[1].pair_counts()[0]);
    assert!(p0 > 0. && p1 > 0.);
    assert_eq!(both.pair_counts()[0], p0 + p1);
    for q in 0..PERIODS {
        let (j, a, b) = (both.summary(q), parts[0].summary(q), parts[1].summary(q));
        assert!(a[2] > 0. && b[2] > 0.);
        assert!((j[2] - (a[2] + b[2])).abs() <= 1e-9 * j[2], "{j:?} {a:?} {b:?}");
        assert_eq!((j[1], a[1]), (b[1], b[1]));
    }
    for (sel, m) in [([0f32, 0.], 0u32), ([300000., 0.], 2)] {
        let j = both.selection_flows_day(m, &sel, &[1.], 0);
        let a = parts[0].selection_flows_day(m, &sel, &[1.], 0);
        let b = parts[1].selection_rail_day(m, &sel, &[1.], 0);
        assert!(j.iter().zip(a.iter().zip(&b)).all(|(j, (a, b))| (j - (a + b)).abs() <= 1e-9 * j.abs().max(1.)), "{j:?}\n{a:?}\n{b:?}");
        assert!(b.iter().any(|v| *v > 0.) == (m == 2));
    }
}
fn solve_all(a: &mut IntercityApi) {
    for q in 0..PERIODS {
        assert!(a.solve(q));
    }
}
fn sum_markets(v: &[f64], off: usize) -> f64 {
    (0..MARKETS).map(|m| v[1 + 8 * m + off]).sum()
}
#[test]
fn intercity_selection_places_conserve_and_rail_is_within_all() {
    let mut a = api();
    network(&mut a, 6., 1.);
    solve_all(&mut a);
    assert_eq!(a.solved_mask(), 31);
    let all = a.selection_flows(0, &[0., 0.], &[1.]);
    assert_eq!(all[0], 1.);
    let pairs = a.all_pairs();
    let want: f64 = pairs.iter().filter(|p| a.m.zones[p.a as usize].place == 0).map(|p| p.trips).sum();
    assert!(want > 0.);
    assert!((sum_markets(&all, 0) - want).abs() < 1e-6 * want.max(1.), "{all:?} vs {want}");
    let want_in: f64 = pairs.iter().filter(|p| a.m.zones[p.b as usize].place == 0).map(|p| p.trips).sum();
    assert!((sum_markets(&all, 4) - want_in).abs() < 1e-6 * want_in.max(1.));
    for m in 0..MARKETS {
        for e in [0, 4] {
            let (trips, rail) = (all[1 + 8 * m + e], all[2 + 8 * m + e]);
            assert!(rail <= trips + 1e-9, "market {m}: rail {rail} above all {trips}");
        }
    }
    assert!(sum_markets(&all, 1) > 0., "service carries some of the selection");
    let half = a.selection_flows(0, &[0., 0.], &[0.5]);
    for k in 1..1 + 8 * MARKETS {
        assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
    }
    let none = a.selection_flows(0, &[9e6, 9e6], &[1.]);
    assert_eq!(none[0], 0.);
    let head = 1 + 8 * MARKETS;
    let nz = a.zone_count();
    assert_eq!(nz, a.m.zones.len());
    assert!(none[1..head + 4 * nz].iter().all(|v| *v == 0.));
    assert_eq!(all.len(), head + 7 * nz + 4 * MARKETS + 4 * nz + 6 * E_CROSS.len() * nz + 2);
    for d in 0..2 {
        let (mut tot, mut rail, mut sx) = (0., 0., 0.);
        for z in 0..nz {
            let (t, r) = (all[head + 4 * z + 2 * d], all[head + 4 * z + 2 * d + 1]);
            assert!(r <= t + 1e-9, "zone {z}: rail {r} above all {t}");
            tot += t;
            rail += r;
            sx += t * a.m.zones[z].x as f64;
        }
        let (mt, mr) = (sum_markets(&all, 4 * d), sum_markets(&all, 4 * d + 1));
        assert!((tot - mt).abs() < 1e-9 * mt.max(1.) && (rail - mr).abs() < 1e-9 * mr.max(1.));
        assert!((sx - sum_markets(&all, 4 * d + 2)).abs() < 1e-6 * sx.abs().max(1.));
    }
    for z in 0..nz {
        assert_eq!(all[head + 4 * nz + 3 * z], a.m.zones[z].x as f64);
        assert_eq!(all[head + 4 * nz + 3 * z + 1], a.m.zones[z].y as f64);
        assert_eq!(all[head + 4 * nz + 3 * z + 2], a.m.zones[z].place as f64);
    }
    for k in head..head + 4 * nz {
        assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
    }
    let (sx, tot) = (all[1 + 8 + 2], all[1 + 8]);
    assert!(tot > 0. && sx / tot > 100_000., "Philadelphia sits 120 km east: {}", sx / tot);
}
/// T-1373: a rush day's longer trips lean from the larger place to the smaller, each pair's trips
/// in all kept: the loads by the lean's column, a selection's far ends by its places' lean.
#[test]
fn rush_days_lean_towards_smaller_places() {
    use crate::demand::days::{BACK, OUT, RUSH_LONGER};
    let mut a = api();
    network(&mut a, 6., 1.);
    solve_all(&mut a);
    let m = a.m.clone();
    assert_eq!(m.lean(0, 1), 0.);
    assert!(m.lean(0, 2) > 0. && m.lean(0, 2) <= LEAN && m.lean(2, 0) == -m.lean(0, 2));
    let (mut up, mut down) = (false, false);
    for q in 0..PERIODS {
        let (seg, lean) = (a.seg(q), a.lean_loads(q));
        assert_eq!(lean.len(), 3 * seg.len());
        for (r, &s) in seg.iter().enumerate() {
            assert!(lean[r].abs() <= LEAN * s + 1e-3, "period {q} node {r}: {} against {s}", lean[r]);
            up |= lean[r] > 1e-3;
            down |= lean[r] < -1e-3;
        }
    }
    assert!(up && down, "the line carries more one way going out");
    // the small place's residents' trips to place 0: out times 1 - lean, back times 1 + lean
    let at = |day: u32| a.selection_flows_day(2, &[60000., 0.], &[1.], day)[1];
    let (wd, out, back) = (at(0), at(OUT as u32), at(BACK as u32));
    let l = m.lean(2, 0) as f64;
    assert!(wd > 0. && (out - wd * RUSH_LONGER as f64 * (1. + l)).abs() < 1e-6 * wd);
    let sun = at(2) / crate::demand::days::day_rate(2, LONGER) as f64;
    assert!((back - sun * RUSH_LONGER as f64 * (1. - l)).abs() < 1e-6 * wd, "{back} vs {}", sun * 2. * (1. - l));
}
#[test]
fn intercity_station_places_equal_entries_and_exits() {
    let mut a = api();
    network(&mut a, 6., 1.);
    solve_all(&mut a);
    let map = a.station_map();
    let mut starts = 0.;
    let mut ends = 0.;
    let nz = a.zone_count();
    for st in 0..5u32 {
        let row = a.station_flows(st);
        assert_eq!(row.len(), ST_W + 5 * nz);
        for d in 0..2 {
            let zsum: f64 = row[ST_W + d * nz..ST_W + (d + 1) * nz].iter().sum();
            let msum: f64 = row[d * MARKETS..(d + 1) * MARKETS].iter().sum();
            assert!((zsum - msum).abs() < 1e-9 * msum.max(1.));
        }
        for z in 0..nz {
            assert_eq!(row[ST_W + 2 * nz + 3 * z], a.m.zones[z].x as f64);
            assert_eq!(row[ST_W + 2 * nz + 3 * z + 1], a.m.zones[z].y as f64);
            assert_eq!(row[ST_W + 2 * nz + 3 * z + 2], a.m.zones[z].place as f64);
        }
        let (s, e) = (row[..MARKETS].iter().sum::<f64>(), row[MARKETS..2 * MARKETS].iter().sum::<f64>());
        let at = map[st as usize] as usize;
        let entries: f64 = (0..PERIODS).map(|q| a.entries(q)[at] as f64).sum();
        let exits: f64 = (0..PERIODS).map(|q| a.exits(q)[at] as f64).sum();
        assert!((s - entries).abs() < 1e-3 * entries.max(1.), "station {st}: {s} vs {entries}");
        assert!((e - exits).abs() < 1e-3 * exits.max(1.), "station {st}: {e} vs {exits}");
        starts += s;
        ends += e;
    }
    let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
    assert!(rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail);
    assert!(a.station_flows(99).is_empty());
    let (none, row) = (a.no_station_flows(), a.station_flows(0));
    assert_eq!(none.len(), row.len());
    assert!(none[..ST_W + 2 * nz].iter().all(|&v| v == 0.));
    assert_eq!(none[ST_W + 2 * nz..], row[ST_W + 2 * nz..]);
    for q in 0..PERIODS {
        let loads = a.seg(q);
        assert!(a.crowd_with_external(q, &loads));
    }
    let rail2: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
    let starts2: f64 = (0..5u32).map(|s| a.station_flows(s)[..MARKETS].iter().sum::<f64>()).sum();
    assert!((starts2 - rail2).abs() < 1e-3 * rail2.max(1.));
}
#[test]
fn intercity_no_stations_and_invalid_input_are_safe() {
    let mut a = api();
    a.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
    assert!(a.solve(0));
    assert_eq!(a.summary(0)[2], 0.);
    assert!(a.seg(0).is_empty());
    assert!(!a.crowd_with_external(0, &[f32::NAN]));
    assert!(!a.solve(5));
    assert!(IntercityApi::new(r#"{"sites":[[0,0,-1,0,0]]}"#).is_err());
}
#[test]
fn expanded_markets_route_boston_and_washington_and_keep_flow_layouts() {
    let mut a = IntercityApi::new(r#"{"markets":["New York","Philadelphia","Mercer","Boston","Washington DC"],"sites":[[0,0,100000,50000,0],[60000,0,100000,50000,1],[120000,0,100000,50000,2],[180000,0,100000,50000,3],[240000,0,100000,50000,4]]}"#).unwrap();
    assert_eq!(a.market_count(), 5);
    assert_eq!(a.m.n_corridors(), 15);
    let mut times = vec![0.; 30];
    for lev in 0..3 {
        for k in 0..4 {
            times[lev * 5 + k] = 1200.;
            times[(lev + 3) * 5 + k + 1] = 1200.;
        }
    }
    a.set_network(&[0., 0., 60000., 0., 120000., 0., 180000., 0., 240000., 0.], &[5], &[0, 1, 2, 3, 4], &times, &[6., 6., 6.], &[8.]).unwrap();
    solve_all(&mut a);
    let nz = a.zone_count();
    let selected = a.selection_flows(3, &[180000., 0.], &[1.]);
    assert_eq!(selected.len(), 41 + 7 * nz + 20 + 4 * nz + 30 * nz + 2);
    assert_eq!(selected[0], 1.);
    assert!(selected[1 + 8 * 4 + 1] > 0., "Boston to Washington is served");
    let washington = a.selection_flows(4, &[240000., 0.], &[1.]);
    assert_eq!(washington[0], 1.);
    assert!(washington[1 + 8 * 3 + 1] > 0., "Washington to Boston is served");
    let mut starts = 0.;
    let mut ends = 0.;
    for st in 0..5 {
        let row = a.station_flows(st);
        assert_eq!(row.len(), 30 + 5 * nz);
        for d in 0..2 {
            let by_market: f64 = row[d * 5..(d + 1) * 5].iter().sum();
            let by_zone: f64 = row[30 + d * nz..30 + (d + 1) * nz].iter().sum();
            assert!((by_market - by_zone).abs() < 1e-9 * by_market.max(1.));
        }
        starts += row[..5].iter().sum::<f64>();
        ends += row[5..10].iter().sum::<f64>();
    }
    let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
    assert!(rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail);
    assert_eq!(api().market_count(), 3, "legacy packs retain their output layout");
}
#[test]
fn compact_pack_accepts_metadata_and_rejects_malformed_sites() {
    let base = api();
    let reordered = IntercityApi::new(r#"{"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]],"description":"unused","origin":{"lon":-73.985,"lat":40.758},"zone_m":6000,"trip_rate":0.015}"#).unwrap();
    assert_eq!(base.info(), reordered.info());
    for (a, b) in base.all_pairs().iter().zip(&reordered.all_pairs()) {
        assert_eq!((a.a, a.b, a.trips, a.car, a.corridor), (b.a, b.b, b.trips, b.car, b.corridor));
    }
    for bad in [
        r#"{}"#,
        r#"{"sites":[[0,0,1,0]]}"#,
        r#"{"sites":[[0,0,1,0,0,0]]}"#,
        r#"{"sites":[[0,0,null,0,0]]}"#,
        r#"{"sites":[[0,0,"1",0,0]]}"#,
        r#"{"zone_m":0,"sites":[]}"#,
        r#"{"markets":[],"sites":[]}"#,
        r#"{"markets":["A"],"sites":[[0,0,1,0,1]]}"#,
        r#"{"sites":[[0,0,1,0,0.5]]}"#,
    ] {
        assert!(IntercityApi::new(bad).is_err(), "accepted {bad}");
    }
}
/// T-373: rides nest with a site's walks, so a dear ride adds little and a cheap one a lot.
#[test]
fn rides_nest_with_walks() {
    let total = |walks: &[(usize, f32)], rides: &[Ride]| {
        let mut m = BTreeMap::new();
        add_access(&mut m, walks, rides, 0.025, 0.2, 1.);
        m.values().sum::<f32>()
    };
    let walks = [(0, 20.), (1, 30.)];
    let alone = total(&walks, &[]);
    assert!((alone - ((-0.5f32).exp() + (-0.75f32).exp()) / 2.).abs() < 1e-6);
    let dear = total(&walks, &[(2, 60., 0., 0.)]);
    assert!(dear > alone && dear < alone * 1.01, "{alone} {dear}");
    let cheap = total(&walks, &[(2, 10., 0., 0.)]);
    assert!(cheap > (-0.25f32).exp(), "{cheap}");
    assert!((total(&[], &[(2, 20., 0., 0.)]) - (-0.5f32).exp()).abs() < 1e-6);
}
/// T-128: a pack that names its frame measures zone distances on the ground; one without keeps
/// its plane.
#[test]
fn zone_distances_are_on_the_ground_when_the_pack_names_its_frame() {
    use crate::track::frame::great_circle;
    let (lon0, lat0) = (-73.985, 40.758);
    let eq = Frame::equirect(lon0, lat0);
    let places = [(-73.985, 40.758), (-75.1636, 39.9526), (-71.0589, 42.3601), (-77.0369, 38.9072)];
    let sites: Vec<String> = places
        .iter()
        .enumerate()
        .map(|(m, &(lon, lat))| {
            let (x, y) = eq.from_lonlat(lon, lat);
            format!("[{x},{y},100000,50000,{m}]")
        })
        .collect();
    let body = format!(r#""markets":["NY","PHL","BOS","WDC"],"sites":[{}]"#, sites.join(","));
    let plane = IntercityApi::new(&format!("{{{body}}}")).unwrap();
    let frame = r#""frame":{"kind":"equirect","lon":-73.985,"lat":40.758}"#;
    let ground = IntercityApi::new(&format!("{{{frame},{body}}}")).unwrap();
    assert!(!plane.m.ground && ground.m.ground);
    let km = |a: usize, b: usize| great_circle(places[a].0, places[a].1, places[b].0, places[b].1) / 1000.;
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)] {
        let (zi, zj) = (&ground.m.zones[i], &ground.m.zones[j]);
        let d = ground.m.zone_km(zi, zj);
        assert!((d - km(i, j)).abs() < 1e-5 * km(i, j), "{i}-{j}: {d} vs {}", km(i, j));
        let p = plane.m.zone_km(&plane.m.zones[i], &plane.m.zones[j]);
        assert!((p - d).abs() > 1e-4 * d, "{i}-{j}: the plane differs: {p} vs {d}");
    }
    let drive = |a: &IntercityApi| a.all_pairs().iter().find(|p| p.a == 0 && p.b == 2).unwrap().car.at(FARE_MIN_PER_USD);
    assert_eq!(drive(&ground), car_min(ground.m.zone_km(&ground.m.zones[0], &ground.m.zones[2]) as f32, 0.));
    assert!(drive(&ground) != drive(&plane));
    let lcc = Frame::lcc(lon0, lat0);
    let lsites: Vec<String> = places
        .iter()
        .enumerate()
        .map(|(m, &(lon, lat))| {
            let (x, y) = lcc.from_lonlat(lon, lat);
            format!("[{x},{y},100000,50000,{m}]")
        })
        .collect();
    let l = IntercityApi::new(&format!(r#"{{"frame":{{"kind":"lcc","lon":-73.985,"lat":40.758}},"markets":["NY","PHL","BOS","WDC"],"sites":[{}]}}"#, lsites.join(","))).unwrap();
    let (a, b) = (l.m.zone_km(&l.m.zones[0], &l.m.zones[2]), ground.m.zone_km(&ground.m.zones[0], &ground.m.zones[2]));
    assert!((a - b).abs() < 1e-3, "{a} {b}");
    for bad in [r#""frame":{"kind":"mercator","lon":0,"lat":0}"#, r#""frame":{"kind":"equirect","lon":0,"lat":95}"#, r#""frame":{"lon":0,"lat":0}"#] {
        assert!(IntercityApi::new(&format!("{{{bad},{body}}}")).is_err(), "accepted {bad}");
    }
}

// ---- T-377: the world market

const RULE: &str = r#"{"R":0.107,"A":2.37,"onset_km":[25,50],"d0_km":25,"eps":0.7,"G":0.703,"D1_km":171.2,"TH":0.75,"eps_t":1.0,"THo":0.503,"tail_onset_km":[300,700],"border_eps":1.503,"u_mean_us":1.2413,"u_ref_people":100000,"border":0.26,"y_us":76931.3,"Rt":0.00473,"near_km":250,"min_trips":0.01}"#;
/// A place's pack: sites at (km east, km north, people, jobs) of its own equirect origin, each
/// 6 km square a zone, made-up normalisers.
fn zpack(place: usize, (lon0, lat0): (f64, f64), country: &str, inc: f64, road: i64, sites: &[(f64, f64, f64, f64)]) -> String {
    let mut keys: Vec<(i32, i32)> = vec![];
    let mut rows = vec![];
    for &(x, y, p, j) in sites {
        let (x, y) = (x * 1000., y * 1000.);
        let k = ((x / 6000.).floor() as i32, (y / 6000.).floor() as i32);
        let z = keys.iter().position(|&q| q == k).unwrap_or_else(|| {
            keys.push(k);
            keys.len() - 1
        });
        rows.push(format!("[{x},{y},{p},{j},{z}]"));
    }
    let nz = keys.len();
    let ng: Vec<String> = (0..nz).map(|z| format!("{}", 2e4 + 1e3 * z as f64)).collect();
    let nt: Vec<String> = (0..nz).map(|z| format!("{}", 3e5 + 1e4 * z as f64)).collect();
    format!(
        r#"{{"place":{place},"zone":"p{place}","frame":{{"kind":"equirect","lon":{lon0},"lat":{lat0}}},"country":"{country}","inc":{inc},"road":{road},"zones":{},"zone_ng":[{}],"zone_nt":[{}],"sites":[{}],"cell_pop":[{}]}}"#,
        serde_json::to_string(&keys.iter().map(|k| [k.0, k.1]).collect::<Vec<_>>()).unwrap(),
        ng.join(","),
        nt.join(","),
        rows.join(","),
        vec!["200000"; sites.len()].join(",")
    )
}
/// Four places: 0 and 1 near each other (100 km) in one country, 2 far (about 800 km) in
/// another, 3 near 0 but across the sea (another landmass).
fn world_packs() -> Vec<String> {
    vec![
        zpack(0, (0., 50.), "AA", 1.0, 0, &[(0., 0., 300000., 200000.), (10., 0., 100000., 30000.), (40., 5., 50000., 10000.)]),
        zpack(1, (1.4, 50.), "AA", 1.0, 0, &[(0., 0., 200000., 90000.), (8., 3., 40000., 20000.)]),
        zpack(2, (11., 50.), "BB", 0.6, 0, &[(0., 0., 250000., 120000.), (-9., 2., 60000., 10000.)]),
        zpack(3, (0., 49.), "AA", 1.0, 7, &[(0., 0., 80000., 30000.)]),
    ]
}
fn composition(packs: &[String], rows: Option<&str>) -> String {
    let rows = rows.map_or(String::new(), |r| format!(r#","rows":{r}"#));
    format!(r#"{{"world":{{"frame":{{"kind":"equirect","lon":0,"lat":50}},"rule":{RULE},"places":5{rows}}},"zones":[{}]}}"#, packs.join(","))
}
fn rows_json(packs: &[String]) -> String {
    serde_json::to_string(&world_rows(&composition(packs, None)).unwrap()).unwrap()
}
#[test]
fn rows_of_some_places_are_those_of_every_place() {
    // T-929: a place's row from its own pack and its near places' packs, to the bit
    let packs = world_packs();
    // (compared as written: serde_json's own parse can be a bit off in the last place)
    let all = world_rows(&composition(&packs, None)).unwrap();
    let only = |want: &[usize], packs: &[String]| {
        let f: Vec<bool> = (0..5).map(|p| want.contains(&p)).collect();
        serde_json::to_string(&world_rows_for(&composition(packs, None), Some(&f)).unwrap()).unwrap()
    };
    let row = |p: usize| serde_json::to_string(&vec![all.iter().find(|r| r.place == p).unwrap()]).unwrap();
    assert_eq!(only(&[0], &[packs[0].clone(), packs[1].clone(), packs[3].clone()]), row(0));
    assert_eq!(only(&[2], &[packs[2].clone()]), row(2));
    assert_eq!(only(&[1], &packs), row(1));
    // a near place left out changes the row (the composition must hold every one)
    assert_ne!(only(&[0], &[packs[0].clone(), packs[3].clone()]), row(0));
}
#[test]
fn world_rows_add_up_and_open_as_the_game_opens_them() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let w = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    let no = IntercityApi::new(&composition(&packs, None)).unwrap();
    assert_eq!((w.info()[3], no.info()[3]), (1., 0.));
    // the same market from the rows as from the zones
    assert!((w.m.cw.daily - no.m.cw.daily).abs() < 1e-9 * no.m.cw.daily, "{} {}", w.m.cw.daily, no.m.cw.daily);
    assert_eq!(w.m.cw.trips.len(), 15);
    for (a, b) in w.m.cw.trips.iter().zip(&no.m.cw.trips) {
        assert!((a - b).abs() <= 1e-9 * b.max(1.));
    }
    assert!(w.m.cw.daily > 0. && w.m.cw.leaving > 0.);
    // far places' zone pairs add up to the places' trips; near places' rows are their zones'
    let m = &w.m;
    for (i, j) in [(0, 2), (1, 2), (0, 1), (0, 0), (0, 3)] {
        let mut sum = 0.;
        for (a, za) in m.zones.iter().enumerate().filter(|(_, z)| z.place == i) {
            for (b, zb) in m.zones.iter().enumerate().filter(|(_, z)| z.place == j) {
                if i == j && b <= a {
                    continue;
                }
                let t = m.eachway(za, zb);
                if m.pr(i, j).near && t < m.rule.min_trips {
                    continue;
                }
                sum += t;
            }
        }
        let p = m.place_eachway(i, j);
        assert!(p > 0. && (sum - p).abs() < 1e-9 * p, "{i}-{j}: zones {sum}, places {p}");
        assert_eq!(m.pr(i, j).near, i != 2 && j != 2);
    }
    // the border: place 2's residents abroad are fewer than at home in proportion
    let r = m.pr(2, 0);
    assert!((r.b - 0.26 * 0.6f64.powf(1.503)).abs() < 1e-12 && m.pr(0, 1).b == 1.);
    // a stale row (other zone count) does not open
    let mut bad: Vec<serde_json::Value> = serde_json::from_str(&rows).unwrap();
    bad[1]["zones"] = 9.into();
    assert!(IntercityApi::new(&composition(&packs, Some(&serde_json::to_string(&bad).unwrap()))).is_err());
}
/// T-809: a pair inside the open-border area takes its milder factor whatever the income; a pair
/// with an end outside keeps the rule's.
#[test]
fn open_border_area_takes_its_own_factor() {
    let b = |aa: &str, bb: &str| {
        let packs: Vec<String> = world_packs().iter().map(|p| p.replace("\"AA\"", &format!("\"{aa}\"")).replace("\"BB\"", &format!("\"{bb}\""))).collect();
        let w = IntercityApi::new(&composition(&packs, None)).unwrap();
        (w.m.pr(2, 0).b, w.m.pr(0, 2).b, w.m.pr(0, 1).b)
    };
    let general = (0.26 * 0.6f64.powf(1.503), 0.26);
    assert_eq!(b("FR", "GB"), (OPEN_AREA_BORDER, OPEN_AREA_BORDER, 1.));
    assert_eq!(b("CH", "DE"), (OPEN_AREA_BORDER, OPEN_AREA_BORDER, 1.));
    for (aa, bb) in [("US", "GB"), ("MA", "FR"), ("FR", "TR")] {
        let (x, y, z) = b(aa, bb);
        assert!((x - general.0).abs() < 1e-12 && (y - general.1).abs() < 1e-12 && z == 1., "{aa}-{bb}: {x} {y}");
    }
}
/// Lines in place 0 and place 1 (joined) and in place 2 (apart); stations as longitude and
/// latitude, each place in its own frame.
fn world_lines(a: &mut IntercityApi, with_far: bool, across_sea: bool) {
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
    let mut add = |pts: &[f64]| {
        let s = ll.len() as u32 / 2;
        ll.extend_from_slice(pts);
        n.push(2);
        stops.extend([s, s + 1]);
        times.extend([600f32, 0., 600., 0., 600., 0., 0., 600., 0., 600., 0., 600.]);
        tph.extend([6f32, 6., 6.]);
        cars.push(8.);
    };
    if with_far {
        add(&[11., 50., 10.88, 50.02]);
    }
    if across_sea {
        // from place 0's centre to place 3, joined to the first line at its first station
        add(&[0., 50., 0., 49.]);
    }
    a.set_network_ll(&ll, &n, &stops, &times, &tph, &cars).unwrap();
    let k = 2 * stops.len();
    a.set_fares(2., 0.1, &vec![20.; k]).unwrap();
}
#[test]
fn world_market_serves_zones_in_every_frame_and_no_car_keeps_trips_fixed() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let mut a = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    world_lines(&mut a, true, true);
    solve_all(&mut a);
    let m = a.m.clone();
    let served_places: std::collections::BTreeSet<usize> = a.served.iter().map(|&z| m.zones[z].place).collect();
    assert_eq!(served_places.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 3], "each frame's zones are reached");
    // pairs: 0-1 joined by track; 2 apart (no track); 3 joined by track but across the sea (no car)
    for p in &a.pairs {
        let (i, j) = (m.zones[p.a as usize].place, m.zones[p.b as usize].place);
        assert!(i != 2 && j != 2, "no track to place 2");
        assert!((i == 3) == (j == 3), "no pair across the sea: {i}-{j}");
    }
    assert!(a.pairs.iter().any(|p| m.zones[p.a as usize].place == 0 && m.zones[p.b as usize].place == 1));
    let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
    assert!(rail > 0.);
    // a selection in place 3: all its trips to place 0 are shown, none by rail
    let sel = a.selection_flows(3, &[0., 0.], &[1.]);
    assert!(sel[1 + 8 * 0] > 0. && sel[2 + 8 * 0] == 0., "{:?}", &sel[..9]);
    // the far place: all of place 0's trips there, none by rail; its zones in the zone block
    let s0 = a.selection_flows(0, &[0., 0.], &[1.]);
    assert!(s0[1 + 8 * 2] > 0. && s0[2 + 8 * 2] == 0.);
    let head = 1 + 8 * 5;
    let g2 = m.places[2].off as usize;
    assert!(s0[head + 4 * g2] > 0.);
    // the zone positions are in the output frame: place 2's first zone is about 790 km east
    let x = s0[head + 4 * a.zone_count() + 3 * g2];
    assert!((x - 787_000.).abs() < 15_000., "{x}");
}
/// Place 0's pack with everyday outings from its first zone to place 1's first zone: errands
/// (a fifth walking with no rail) and medical visits (none walking).
fn with_everyday(pack: &str, trips: [f32; 2], lens_ok: bool) -> String {
    let mut v: serde_json::Value = serde_json::from_str(pack).unwrap();
    v["everyday_home"] = serde_json::json!([0, 0]);
    v["everyday_place"] = serde_json::json!([1, 1]);
    v["everyday_zone"] = serde_json::json!([0, 0]);
    v["everyday_kind"] = serde_json::json!([0, 5]);
    v["everyday_trips"] = serde_json::json!(trips);
    v["everyday_walk"] = if lens_ok { serde_json::json!([0.2, 0.0]) } else { serde_json::json!([0.2]) };
    v.to_string()
}
/// T-1175: everyday outings across a zone line are solved beside the longer trips without moving
/// them, keep their legs whatever the network, load their own purposes and lose walkers to rail.
#[test]
fn everyday_pairs_across_a_zone_line() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let mut e_packs = packs.clone();
    e_packs[0] = with_everyday(&packs[0], [1000., 100.], true);
    let mut plain = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    let mut a = IntercityApi::new(&composition(&e_packs, Some(&rows))).unwrap();
    assert_eq!(a.m.everyday.len(), 2);
    world_lines(&mut plain, true, true);
    world_lines(&mut a, true, true);
    solve_all(&mut plain);
    solve_all(&mut a);
    assert_eq!(a.epairs.len(), 2, "both pairs are served");
    let (mut legs, mut rail, mut walk) = (0., 0., 0.);
    for q in 0..PERIODS {
        let (p, s) = (plain.part_summary(q), a.part_summary(q));
        assert_eq!((p[0], p[1]), (s[0], s[1]), "the longer trips do not move");
        assert_eq!((p[4], p[5], p[6]), (0., 0., 0.));
        legs += s[4];
        rail += s[5];
        walk += s[6];
        assert!(s[5] > 0. && s[5] < s[4] && s[6] < s[7] && s[6] > 0.);
        assert_eq!(a.summary(q)[3], s[6]);
        assert!((a.summary(q)[1] - plain.summary(q)[1] - s[4]).abs() < 1e-6);
        // their loads are the purposes' own: errands and medical boardings, nothing else
        let l = a.everyday_loads(q);
        let m = a.seg(q).len();
        assert_eq!(l.len(), 3 * KIND_COUNT * m);
        let board = |k: usize| l[(3 * k + 1) * m..(3 * k + 2) * m].iter().sum::<f32>() as f64;
        assert!(board(0) > 0. && board(5) > 0.);
        assert!((0..KIND_COUNT).filter(|&k| k != 0 && k != 5).all(|k| board(k) == 0.));
        assert!((board(0) + board(5) - s[5]).abs() < 1e-3 * s[5], "{} {}", board(0) + board(5), s[5]);
        assert!(plain.everyday_loads(q).is_empty());
    }
    assert!((legs - 2. * 1100.).abs() < 1e-3, "{legs}");
    assert!(rail > 0. && walk < 2. * 1000. * 0.2);
    // arrays that disagree do not open
    let mut bad = packs.clone();
    bad[0] = with_everyday(&packs[0], [1000., 100.], false);
    assert!(IntercityApi::new(&composition(&bad, Some(&rows))).is_err());
}
/// T-1229: the everyday pairs by purpose: in `part_summary` by kind and `everyday_by_place`, in a
/// selection's blocks (all modes from the market alone, rail and the walkers rail takes from the
/// network), and in the trip explorer between the two places.
#[test]
fn everyday_pairs_by_purpose_in_selections_and_the_explorer() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let mut e_packs = packs.clone();
    e_packs[0] = with_everyday(&packs[0], [1000., 100.], true);
    let mut a = IntercityApi::new(&composition(&e_packs, Some(&rows))).unwrap();
    world_lines(&mut a, true, true);
    solve_all(&mut a);
    let (np, ng) = (a.m.places.len(), a.m.n_gid);
    let (mut rail, mut walk) = ([0f64; KIND_COUNT], [0f64; KIND_COUNT]);
    for q in 0..PERIODS {
        let s = a.part_summary(q);
        assert_eq!(s.len(), 8 + 4 * KIND_COUNT);
        for i in 0..4 {
            let sum: f64 = (0..KIND_COUNT).map(|k| s[8 + 4 * k + i]).sum();
            assert!((sum - s[4 + i]).abs() <= 1e-4 * s[4 + i].max(1.), "figure {i}: {sum} {}", s[4 + i]);
        }
        for k in (0..KIND_COUNT).filter(|&k| k != 0 && k != 5) {
            assert!((0..4).all(|i| s[8 + 4 * k + i] == 0.), "kind {k} has no pairs");
        }
        let by = a.everyday_by_place(q);
        assert_eq!(by.len(), 3 * np * KIND_COUNT);
        for k in 0..KIND_COUNT {
            for i in 0..3 {
                assert_eq!(by[3 * k + i], s[8 + 4 * k + i], "place 0, kind {k}, figure {i}");
            }
            assert!((1..np).all(|p| by[3 * (p * KIND_COUNT + k)..][..3].iter().all(|&v| v == 0.)), "only place 0 has homes with pairs");
            rail[k] += s[8 + 4 * k + 1];
            walk[k] += s[8 + 4 * k + 2];
        }
    }
    assert!(rail[0] > 0. && rail[5] > 0. && walk[0] > 0.);
    // a selection of every site of place 0 (homes), and of place 1 (the first zone, activities)
    let sites = |place: usize| {
        let g = a.m.groups.iter().find(|g| a.m.zones[a.m.cell_zone[g.lo]].place == place).unwrap();
        let xy: Vec<f32> = (0..g.city.len()).flat_map(|i| [g.city.x[i], g.city.y[i]]).collect();
        (xy, vec![1f32; g.city.len()])
    };
    let etail = 1 + 8 * np + 11 * ng + 4 * np;
    let all_q = (1u32 << PERIODS) - 1;
    let far1 = a.m.places[1].off as usize;
    let far0 = a.m.places[0].off as usize;
    for (market, end, far) in [(0usize, 0usize, far1), (1, 1, far0)] {
        let (xy, share) = sites(market);
        let base = a.selection_all_day(market as u32, &xy, &share, WEEKDAY as u32, all_q);
        let with = a.selection_rail_day(market as u32, &xy, &share, WEEKDAY as u32);
        assert_eq!(base.len(), etail + E_CROSS.len() * 6 * ng + 2);
        for (ki, &k) in E_CROSS.iter().enumerate() {
            let o = etail + ki * 6 * ng + 6 * far + 3 * end;
            let want = if k == 0 { 2000. } else if k == 5 { 200. } else { 0. };
            assert!((base[o] - want).abs() < 1e-3 * want.max(1.), "kind {k} end {end}: legs {} not {want}", base[o]);
            assert_eq!(base[o + 1], 0., "no rail without a network");
            assert!((base[o + 2] - base[o] * if k == 0 { 0.2 } else { 0. }).abs() < 1e-3, "walkers with no rail");
            assert_eq!(with[o], 0., "legs are the market's alone");
            assert!((with[o + 1] - rail[k]).abs() < 1e-3 * rail[k].max(1.), "kind {k} end {end}: rail {} not {}", with[o + 1], rail[k]);
            // what is left walking is the explorer's and the summary's
            assert!((base[o + 2] + with[o + 2] - walk[k]).abs() < 1e-3 * walk[k].max(1.), "kind {k} end {end}: walking {} not {}", base[o + 2] + with[o + 2], walk[k]);
        }
        // the longer trips' and the commutes' blocks do not hold them
        assert!(base[etail..etail + E_CROSS.len() * 6 * ng].iter().sum::<f64>() > 0.);
    }
    // the explorer between the places, either way, one period
    for q in 0..PERIODS {
        let s = a.part_summary(q);
        for (near, far, dir) in [(0u32, 1u32, 0u32), (1, 0, 1)] {
            let h = a.market_trip_between(q, WEEKDAY as u32, near, -1, dir, far, -1);
            assert!(h.len() >= super::market_trip::MT_HEAD);
            assert!((h[19] - s[4]).abs() < 1e-3 * s[4].max(1.), "q {q} dir {dir}: legs {} not {}", h[19], s[4]);
            assert!((h[20] - s[5]).abs() < 1e-3 * s[5].max(1.), "rail {} not {}", h[20], s[5]);
            assert!((h[21] - s[6]).abs() < 1e-3 * s[6].max(1.), "walking {} not {}", h[21], s[6]);
            if h[20] > 0. {
                assert!(h[11] > 0. && h[14] == 0., "pairs {} why {}", h[11], h[14]);
            }
        }
    }
    // a selection of place 1 asks one way: nobody living there goes out to place 0, and the
    // activities there are place 0's residents'
    let (xy, share) = sites(1);
    let out = a.market_trip(0, WEEKDAY as u32, 1, &xy, &share, 0, 0, -1);
    assert_eq!((out[19], out[20], out[21]), (0., 0., 0.));
    let back = a.market_trip(0, WEEKDAY as u32, 1, &xy, &share, 1, 0, -1);
    let both = a.market_trip_between(0, WEEKDAY as u32, 1, -1, 1, 0, -1);
    assert!(back[19] > 0. && (back[19] - both[19]).abs() < 1e-6 * both[19]);
}
#[test]
fn places_not_open_answer_by_place() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let mut full = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    // places 1 and 2 not open here: their trips by place, none at their zones
    let mut some = IntercityApi::new(&composition(&[packs[0].clone(), packs[3].clone()], Some(&rows))).unwrap();
    assert_eq!(some.open_places(), vec![0, 3]);
    assert_eq!(some.zone_count(), full.zone_count());
    assert_eq!(some.m.cw.daily, full.m.cw.daily);
    for a in [&mut full, &mut some] {
        a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
        solve_all(a);
    }
    let (f, s) = (full.selection_flows(0, &[0., 0.], &[1.]), some.selection_flows(0, &[0., 0.], &[1.]));
    let head = 1 + 8 * 5;
    // far: exact
    assert!((f[1 + 8 * 2] - s[1 + 8 * 2]).abs() < 1e-9 * f[1 + 8 * 2], "{} {}", f[1 + 8 * 2], s[1 + 8 * 2]);
    // near: by the zone's shares, the same order
    assert!(s[1 + 8] > 0.3 * f[1 + 8] && s[1 + 8] < 3. * f[1 + 8], "{} {}", s[1 + 8], f[1 + 8]);
    let g1 = some.m.places[1].off as usize;
    assert_eq!(s[head + 4 * g1], 0.);
    // its zones sit at their place's centre
    let ng = some.zone_count();
    assert_eq!(s[head + 4 * ng + 3 * g1], some.m.places[1].x as f64);
    // the place's own trips are the same as with every pack open
    assert!((f[1] - s[1]).abs() < 1e-9 * f[1]);
}
/// T-378: a place's pack with airports (longitude, latitude), each flying direct to `direct`.
fn with_air(pack: &str, airports: &[(f64, f64)], direct: &[u32], road: Option<i64>) -> String {
    let mut v: serde_json::Value = serde_json::from_str(pack).unwrap();
    v["airports"] = serde_json::json!(airports.iter().map(|a| [a.0, a.1]).collect::<Vec<_>>());
    v["air_direct"] = serde_json::json!(airports.iter().map(|_| direct).collect::<Vec<_>>());
    if let Some(r) = road {
        v["road"] = r.into();
    }
    v.to_string()
}
/// Airports at places 0, 2 and 3 (place 1 has none), 0 and 2 flying direct; place 2 on another
/// landmass when `island`.
fn air_packs(island: bool) -> Vec<String> {
    let p = world_packs();
    vec![
        with_air(&p[0], &[(0.1, 50.)], &[2], None),
        p[1].clone(),
        with_air(&p[2], &[(11.1, 50.02)], &[], island.then_some(5)),
        with_air(&p[3], &[(0., 49.05)], &[], None),
    ]
}
/// A line between place 0's and place 2's centres taking `hours`, every 10 minutes, with fares
/// over 900 km at a long-distance rate (T-1521).
fn far_line(a: &mut IntercityApi, hours: f32) {
    let s = hours * 3600.;
    let times = [s, 0., s, 0., s, 0., 0., s, 0., s, 0., s];
    a.set_network_ll(&[0., 50., 11., 50.], &[2], &[0, 1], &times, &[6., 6., 6.], &[8.]).unwrap();
    a.set_fares(2., 0.15, &[900., 0., 0., 900.]).unwrap();
}
#[test]
fn the_car_pays_the_channel_and_the_tsugaru_ferry() {
    let (lon, par, man) = (("GB", 51.5), ("FR", 48.9), ("GB", 53.5));
    let (tyo, spk, hak) = (("JP", 35.7), ("JP", 43.1), ("JP", 41.8));
    let crossing_min = |a, b| crossing_cost(a, b).at(FARE_MIN_PER_USD);
    assert_eq!((crossing_min(lon, man), crossing_min(tyo, ("JP", 40.8)), crossing_min(spk, hak)), (0., 0., 0.));
    assert_eq!(crossing_min(lon, par), crossing_min(par, man));
    assert!(crossing_min(lon, par) > 200. && crossing_min(tyo, spk) > crossing_min(lon, par));
}
#[test]
fn air_flies_far_pairs_by_place_and_rail_takes_its_share_by_the_curve() {
    let packs = air_packs(false);
    let rows = rows_json(&packs);
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&rows).unwrap();
    assert!(parsed[0]["air"][0][0].as_f64().unwrap() > 15. && parsed[1].get("air").is_none(), "{rows}");
    let mut a = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    let m = a.m.clone();
    // far with airports at both ends: some fly, the rest drive; near or with no airport: none fly
    let r02 = m.pr(0, 2);
    assert!(r02.car && r02.air > 0.5 && r02.air < 1., "{}", r02.air);
    let alt = |r: &PairRule, i: usize, j: usize| {
        let split = r.split_car(&m.places[i], &m.places[j]);
        r.alt(split, split, m.places[i].w(), &m.mp)
    };
    let rest = alt(&r02, 0, 2).rest;
    assert!(((rest - 1.) / rest - r02.air).abs() < 1e-5, "{rest} {}", r02.air);
    assert_eq!((m.pr(0, 1).air, m.pr(1, 2).air, m.pr(0, 3).air), (0., 0., 0.));
    assert!(!m.pr(0, 3).has_alt(), "near and across the sea: no car, no plane");
    // the corridors' split is the pairs'
    let modes = a.corridor_modes();
    let c = m.corridor(0, 2);
    assert_eq!((modes[2 * c], modes[2 * c + 1]), (r02.air, 1. - r02.air));
    let c = m.corridor(0, 3);
    assert_eq!((modes[2 * c], modes[2 * c + 1]), (0., 0.));
    // the same trips over the same distance with no direct flight fly less
    let nod = air_packs(false).into_iter().enumerate().map(|(i, p)| if i == 0 { with_air(&p, &[(0.1, 50.)], &[], None) } else { p }).collect::<Vec<_>>();
    let b = IntercityApi::new(&composition(&nod, Some(&rows_json(&nod)))).unwrap();
    assert!(b.m.pr(0, 2).air < r02.air);
    // rail against air on a far pair (T-1521's plane): 3 hours on the
    // train takes about half, 6 hours few
    let ratio = |a: &mut IntercityApi, hours: f32| {
        far_line(a, hours);
        assert!(a.solve(0));
        let (za, zb) = (a.m.places[0].off as usize, a.m.places[2].off as usize);
        let (za, zb) = (a.m.comp_of_gid[za] as usize, a.m.comp_of_gid[zb] as usize);
        let i = a.pair_at(za, zb) as usize;
        let rail = a.per(0).unwrap().pair_rail[i] as f64 / (a.pairs[i].trips * SHARES[0] as f64);
        let air = (1. - rail) * a.m.pr(0, 2).air as f64;
        rail / (rail + air)
    };
    let (fast, slow) = (ratio(&mut a, 3.), ratio(&mut a, 6.));
    assert!(fast > 0.3 && fast < 0.9 && slow < 0.2 && slow < fast, "rail of rail and air: 3 h {fast}, 6 h {slow}");
    // across the sea the far pair is all air, and still served by rail where track joins it
    let packs = air_packs(true);
    let rows = rows_json(&packs);
    let mut s = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    let r = s.m.pr(0, 2);
    let split = r.split_car(&s.m.places[0], &s.m.places[2]);
    assert!(!r.car && r.air == 1. && r.alt(split, split, s.m.places[0].w(), &s.m.mp).rest == 1. && r.has_alt());
    // rail against air does not depend on the car (one logit: the plane's weight against the
    // car's cancels), up to the car's zone against place distance
    let island = ratio(&mut s, 3.);
    assert!((island - fast).abs() < 0.05, "{island} {fast}");
}
/// T-1431: a place's riders in income groups each choose with every mode's money at their own
/// minutes a dollar, and a pair's rail trips are the groups' by their shares of the trips.
#[test]
fn income_groups_choose_at_their_own_weight() {
    let packs = air_packs(false);
    let rows = rows_json(&packs);
    let rail = |fg: Option<&str>| {
        let mut v: Vec<serde_json::Value> = serde_json::from_str(&rows).unwrap();
        for r in v.iter_mut() {
            r["fw"] = 4.5.into();
            if let Some(fg) = fg {
                r["fg"] = serde_json::from_str(fg).unwrap();
            }
        }
        let mut a = IntercityApi::new(&composition(&packs, Some(&serde_json::to_string(&v).unwrap()))).unwrap();
        assert_eq!(a.m.grouped, fg.is_some_and(|f| f != "[1,1]"));
        far_line(&mut a, 4.);
        assert!(a.solve(0));
        let (za, zb) = (a.m.places[0].off as usize, a.m.places[2].off as usize);
        let (za, zb) = (a.m.comp_of_gid[za] as usize, a.m.comp_of_gid[zb] as usize);
        let i = a.pair_at(za, zb) as usize;
        a.per(0).unwrap().pair_rail[i] as f64 / (a.pairs[i].trips * SHARES[0] as f64)
    };
    let (none, one) = (rail(None), rail(Some("[1,1]")));
    assert_eq!(none, one);
    let (rich, poor, mix) = (rail(Some("[1,0.6]")), rail(Some("[1,1.6]")), rail(Some("[0.5,0.6,1.5,1.6]")));
    assert!((rich - poor).abs() > 1e-3, "{rich} {poor}");
    let want = 0.25 * rich + 0.75 * poor;
    assert!((mix - want).abs() < 1e-5 * want, "{mix} against {want}");
    assert!(IntercityApi::new(&composition(&packs, Some(&rows.replacen("\"park\"", "\"fg\":[1,0],\"park\"", 1)))).is_err());
}
/// T-1521: a place's row may give its country's car running cost; a pair's car costs the two
/// places' mean, and a row without one the default.
#[test]
fn a_rows_car_cost_sets_the_pairs() {
    let packs = air_packs(false);
    let rows = rows_json(&packs);
    let with = |cu: &[f64]| {
        let mut v: Vec<serde_json::Value> = serde_json::from_str(&rows).unwrap();
        for (r, &c) in v.iter_mut().zip(cu) {
            r["cu"] = c.into();
        }
        IntercityApi::new(&composition(&packs, Some(&serde_json::to_string(&v).unwrap()))).unwrap()
    };
    let usd = |a: &IntercityApi| a.m.pr(0, 2).car_cost(100., 0.).usd;
    let (none, same, mixed) = (usd(&with(&[])), usd(&with(&[0.15, 0.15, 0.15, 0.15])), usd(&with(&[0.3, 0.15, 0.1, 0.15])));
    assert!((none - same).abs() < 1e-6 && (mixed / none - (0.3 + 0.1) / 2. / 0.15).abs() < 1e-5, "{none} {same} {mixed}");
}
#[test]
fn trips_with_places_that_have_no_zone_add_up() {
    let packs = air_packs(false);
    let rows = rows_json(&packs);
    let mut a = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    a.set_network_ll(&[], &[], &[], &[], &[], &[]).unwrap();
    solve_all(&mut a);
    let by_q: f64 = (0..PERIODS).map(|q| a.elsewhere(q)[2]).sum();
    assert!(by_q > 0. && (by_q - a.m.cw.leaving_by[2]).abs() < 1e-6 * by_q, "{by_q} {}", a.m.cw.leaving_by[2]);
    // every site of place 2 selected: its trips with nowhere in the market are its row's
    let g = &a.m.groups.iter().find(|g| a.m.zones[a.m.cell_zone[g.lo]].place == 2).unwrap();
    let xy: Vec<f32> = (0..g.city.len()).flat_map(|i| [g.city.x[i], g.city.y[i]]).collect();
    let v = a.selection_flows(2, &xy, &vec![1.; g.city.len()]);
    let away = v[v.len() - 2];
    assert!((away - a.m.cw.leaving_by[2]).abs() < 1e-6 * away, "{away} {}", a.m.cw.leaving_by[2]);
    assert!(v[v.len() - 1] > 0.);
}
/// T-377: the worker's way: packs held once, composed; all modes from the composition with no
/// solve, rail from the parts.
#[test]
fn market_packs_compose_and_answer_all_modes_without_a_solve() {
    let packs = world_packs();
    let rows = rows_json(&packs);
    let world = format!(r#"{{"frame":{{"kind":"equirect","lon":0,"lat":50}},"rule":{RULE},"places":5,"rows":{rows}}}"#);
    let mut h = MarketPacks::new(&world).unwrap();
    for p in packs.iter().rev() {
        h.add(p).unwrap();
    }
    assert_eq!(h.places(), vec![0, 1, 2, 3]);
    let base = h.market().unwrap();
    let mut part = base.beside();
    let mut direct = IntercityApi::new(&composition(&packs, Some(&rows))).unwrap();
    assert!(base.same_market(&part) && !base.same_market(&direct));
    for a in [&mut part, &mut direct] {
        world_lines(a, true, true);
        solve_all(a);
    }
    let whole = direct.selection_flows(0, &[0., 0.], &[1.]);
    let all = base.selection_all_day(0, &[0., 0.], &[1.], 0, 31);
    let rail = part.selection_rail_day(0, &[0., 0.], &[1.], 0);
    assert!(rail.iter().any(|v| *v > 0.));
    for i in 0..whole.len() {
        assert!((whole[i] - (all[i] + rail[i])).abs() <= 1e-9 * whole[i].abs().max(1.), "{i}: {} vs {} + {}", whole[i], all[i], rail[i]);
    }
    assert!(h.remove(2) && !h.remove(2));
    assert_eq!(h.places(), vec![0, 1, 3]);
    assert_eq!(h.market().unwrap().open_places(), vec![0, 1, 3]);
}
/// T-174 and T-377: the market alone on the real New York network plus one Philadelphia line,
/// five periods, free flow and three crowding rounds; every output the app reads.
/// `TW_IC_DUMP=<file>` writes them, `TW_IC_COMPARE=<file>` compares with such a file.
/// `cargo test --release --features demand intercity_real_network_outputs -- --ignored --nocapture`
#[test]
#[ignore]
fn intercity_real_network_outputs() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let t = now_ms();
    let mut a = IntercityApi::new(&real_world(root, None)).unwrap();
    eprintln!("open {:.0} ms, info {:?}", now_ms() - t, a.info());
    let (RealNet { mut st_xy, mut line_n, mut stops, mut times, mut tph, mut cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    const R: f64 = 6371008.8;
    let rad = std::f64::consts::PI / 180.0;
    let wx = R * (-75.1636 + 73.985) * rad * (40.758 * rad).cos();
    let wy = R * (39.9526 - 40.758) * rad;
    let s0 = (st_xy.len() / 2) as u32;
    st_xy.extend([wx as f32, (wy - 2000.0) as f32, wx as f32, (wy + 2000.0) as f32]);
    line_n.push(2);
    stops.extend([s0, s0 + 1]);
    for run in 0..2 {
        for _ in 0..3 {
            times.extend(if run == 0 { [600.0, 0.0] } else { [0.0, 600.0] });
        }
    }
    tph.extend([4.0, 4.0, 2.0]);
    cars.push(8.0);
    let hop_km = route_nodes(&st_xy, &line_n, &stops).2;
    // the network is in the New York frame: to longitude and latitude
    let ny = Frame::equirect(-73.985, 40.758);
    let ll: Vec<f64> = st_xy.chunks_exact(2).flat_map(|p| {
        let (lon, lat) = ny.to_lonlat(p[0] as f64, p[1] as f64);
        [lon, lat]
    }).collect();
    let t = now_ms();
    a.set_network_ll(&ll, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    eprintln!("set_network and fares {:.0} ms; pairs served and place pairs {:?}, zones served {}", now_ms() - t, a.pair_counts(), a.served.len());
    let t = now_ms();
    solve_all(&mut a);
    for _ in 0..3 {
        for q in 0..PERIODS {
            let own = a.seg(q);
            assert!(a.crowd_with_external(q, &own));
        }
    }
    eprintln!("free flow and three rounds {:.0} ms", now_ms() - t);
    let mut out = serde_json::Map::new();
    out.insert("info".into(), serde_json::json!(a.info()[..3]));
    for q in 0..PERIODS {
        let mut s = a.summary(q);
        s[4] = 0.;
        out.insert(format!("summary{q}"), serde_json::json!(s));
        out.insert(format!("corridors{q}"), serde_json::json!(a.corridor_summary(q)));
        out.insert(format!("seg{q}"), serde_json::json!(a.seg(q)));
        out.insert(format!("board{q}"), serde_json::json!(a.board(q)));
        out.insert(format!("entries{q}"), serde_json::json!(a.entries(q)));
        out.insert(format!("crush{q}"), serde_json::json!(a.load_of_crush(q)));
    }
    let ns = st_xy.len() as u32 / 2;
    for st in [0, 1, 10, 100, 300, 500, ns - 3, ns - 1, ns + 5] {
        out.insert(format!("station{st}"), serde_json::json!(a.station_flows(st)));
    }
    for (k, (m, xy)) in [(0u32, vec![0f32, 0., 1500., -800., -3000., -7000.]), (1, vec![wx as f32, wy as f32, wx as f32 + 700., wy as f32 + 900.])].iter().enumerate() {
        let share = vec![1f32; xy.len() / 2];
        let t = now_ms();
        let v = a.selection_flows(*m, xy, &share);
        eprintln!("selection {k}: {:.1} ms, starts {:.1}", now_ms() - t, (0..a.market_count()).map(|p| v[1 + 8 * p]).sum::<f64>());
        out.insert(format!("selection{k}"), serde_json::json!(v));
    }
    let now = serde_json::Value::Object(out);
    if let Ok(path) = std::env::var("TW_IC_DUMP") {
        std::fs::write(&path, serde_json::to_string(&now).unwrap()).unwrap();
        eprintln!("dumped to {path}");
    }
    if let Ok(path) = std::env::var("TW_IC_COMPARE") {
        let base: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let flat = |v: &serde_json::Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
        let (mut worst, mut same, mut keys) = (0f64, 0usize, 0usize);
        for (k, v) in now.as_object().unwrap() {
            let (x, y) = (flat(v), flat(&base[k]));
            assert_eq!(x.len(), y.len(), "{k} length");
            let scale = y.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-12);
            let diff = x.iter().zip(&y).fold(0f64, |m, (p, q)| m.max((p - q).abs()));
            keys += 1;
            if x == y {
                same += 1;
            } else {
                eprintln!("{k}: max diff {diff:.3e} of max {scale:.3e}");
            }
            worst = worst.max(diff / scale);
        }
        eprintln!("compared {keys} outputs: {same} identical, worst relative difference {worst:.3e}");
        assert!(worst < 1e-6);
    }
}
/// The Northeast's old single pack opens as a closed world (`bin/joint_gravity.rs` checks what it
/// writes this way). `cargo test --release --features demand legacy_northeast_pack_opens -- --ignored`
#[test]
#[ignore]
fn legacy_northeast_pack_opens() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let text = std::fs::read_to_string(format!("{root}/data/packs/intercity.json")).unwrap();
    let t = now_ms();
    let a = IntercityApi::new(&text).unwrap();
    let (zones, first, keys) = market_site_zones(&text).unwrap();
    eprintln!("{:.0} ms, info {:?}, {} sites, {} zones", now_ms() - t, a.info(), zones.len(), first.len());
    assert_eq!(keys.len() as f64, a.info()[0]);
    assert!(a.info()[2] > 0. && a.m.cw.commuters > 0.);
}
/// T-378: rail against the plane on real far pairs, one fast line between two stations (no feeders:
/// walks and rides only), fares $1.50 + 10 cents a rail km, free flow in the busiest period.
/// `cargo test --release --features demand real_far_lines_share_rail_with_air -- --ignored --nocapture`
#[test]
#[ignore]
fn real_far_lines_share_rail_with_air() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    // T-495: and the pairs notes/T-495.md compares with published shares, at today's train times
    let pairs: [(&str, &str, (f64, f64), (f64, f64), f32, &[f32]); 6] = [
        ("lon", "par", (-0.1263, 51.5306), (2.3553, 48.8809), 492f32, &[2.27f32, 3., 4.]),
        ("nyc", "wdc", (-73.9935, 40.7506), (-77.0064, 38.8973), 362., &[2.75, 3.5, 4.5]),
        ("nyc", "phl", (-73.9935, 40.7506), (-75.1817, 39.9557), 146., &[1.2]),
        ("nyc", "bos", (-73.9935, 40.7506), (-71.0552, 42.3523), 370., &[3.75]),
        ("par", "ber", (2.3592, 48.8768), (13.3695, 52.5251), 1090., &[8., 4.]),
        ("tyo", "osa", (139.7671, 35.6812), (135.5003, 34.7334), 553., &[2.5]),
    ];
    for (a, b, sa, sb, km, hours) in pairs {
        let mut api = IntercityApi::new(&real_world(root, Some(&[a, b]))).unwrap();
        let (pa, pb) = (api.m.places.iter().position(|p| p.zone == a).unwrap(), api.m.places.iter().position(|p| p.zone == b).unwrap());
        let c = api.m.corridor(pa, pb);
        let (air, car) = (api.corridor_modes()[2 * c], api.corridor_modes()[2 * c + 1]);
        eprintln!("{a} - {b} with no rail: plane {:.1}%, car {:.1}% of the trips not by rail", 100. * air, 100. * car);
        for &h in hours {
            let s = h * 3600.;
            let times = [s, 0., s, 0., s, 0., 0., s, 0., s, 0., s];
            api.set_network_ll(&[sa.0, sa.1, sb.0, sb.1], &[2], &[0, 1], &times, &[2., 2., 2.], &[16.]).unwrap();
            api.set_fares(1.5, 0.1, &[km, 0., 0., km]).unwrap();
            assert!(api.solve(1));
            let cs = api.corridor_summary(1);
            let (trips, rail) = (cs[4 * c + 2], cs[4 * c + 3]);
            let rest = trips - rail;
            // the zone pair the line serves best: its stations' own zones
            let per = api.per(1).unwrap();
            let best = api.pairs.iter().enumerate().filter(|(_, p)| api.m.zones[p.a as usize].place == pa && api.m.zones[p.b as usize].place == pb).map(|(i, p)| per.pair_rail[i] as f64 / (p.trips * SHARES[1] as f64)).fold(0., f64::max);
            eprintln!(
                "{a} - {b}, {h} h on the train: {:.0} trips in the period, rail {:.1}%, car {:.1}%, plane {:.1}%, rail of rail and plane {:.0}%; best-served zone pair rail {:.0}%, of rail and plane {:.0}%",
                trips, 100. * rail / trips, 100. * rest * car as f64 / trips, 100. * rest * air as f64 / trips, 100. * rail / (rail + rest * air as f64),
                100. * best, 100. * best / (best + (1. - best) * air as f64)
            );
        }
    }
}
/// T-761: the pass with its destinations searched one at a time gives the pass's figures to the
/// bit, whatever share of the strategies it keeps between its sweeps. Two lines across Tokyo
/// changing at Tokyo station, so the commutes between zones and the transfers are in it.
#[test]
fn a_streamed_pass_is_the_pass_to_the_bit() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    // the commutes between zones here are Tokyo's to Gunma's
    let mut api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    let ll = [139.7671, 35.6812, 139.7387, 35.6285, 139.6970, 35.5313, 139.6222, 35.4660, 139.7006, 35.6896, 139.7774, 35.7138, 139.6239, 35.9064, 139.0124, 36.3221];
    let mut times = vec![];
    for len in [4, 5] {
        let mut t = vec![0f32; 6 * len];
        for lev in 0..3 {
            for k in 0..len - 1 {
                t[lev * len + k] = 600.;
                t[(3 + lev) * len + k + 1] = 600.;
            }
        }
        times.extend(t);
    }
    api.set_network_ll(&ll, &[4, 5], &[0, 1, 2, 3, 4, 0, 5, 6, 7], &times, &[12., 12., 12., 8., 8., 8.], &[10., 10.]).unwrap();
    api.set_fares(1.5, 0.1, &[8.; 18]).unwrap();
    assert!(api.solve(0));
    let per = api.per(0).unwrap();
    let (g, p, ix) = (&per.g, &per.p, &per.xi);
    let whole = api.pass_on(g, p, 0, &skims_for(g, p, &api.want), ix);
    assert!(whole.c_rail > 0. && whole.rail > 0. && whole.xfer.iter().any(|&x| x > 0.), "commutes {} ({} pairs of {}), longer trips {} and transfers {}", whole.c_rail, api.cpairs.len(), api.m.commutes.len(), whole.rail, whole.xfer.iter().sum::<f32>());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let loads = |l: &Loads| [&l.seg, &l.board, &l.alight, &l.rn_board, &l.rn_alight, &l.entry, &l.exit].map(|v| bits(v));
    let one = strategy_bytes(g);
    for keep in [0, 3 * one, usize::MAX] {
        let s = api.pass_streamed(g, p, 0, ix, keep);
        assert_eq!(loads(&s.loads), loads(&whole.loads), "keeping {keep} bytes");
        assert_eq!(loads(&s.c_loads), loads(&whole.c_loads));
        assert_eq!((s.loads.pax_min.to_bits(), s.c_loads.pax_min.to_bits()), (whole.loads.pax_min.to_bits(), whole.c_loads.pax_min.to_bits()));
        assert_eq!((s.rail.to_bits(), s.c_rail.to_bits()), (whole.rail.to_bits(), whole.c_rail.to_bits()));
        assert_eq!((bits(&s.c_frac), bits(&s.pair_rail), bits(&s.st), bits(&s.xfer)), (bits(&whole.c_frac), bits(&whole.pair_rail), bits(&whole.st), bits(&whole.xfer)));
        assert_eq!((s.cor, s.c_cor), (whole.cor.clone(), whole.c_cor.clone()));
    }
}
/// T-1286: fare factors of 1, or for a country outside the market, leave the pass alone to the bit;
/// a factor of 2 where the trips start costs rail riders, the same streamed or not.
#[test]
fn fare_factors_weigh_trips_by_their_origin() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let mut api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    let ll = [139.7671, 35.6812, 139.7387, 35.6285, 139.6970, 35.5313, 139.6222, 35.4660, 139.7006, 35.6896, 139.7774, 35.7138, 139.6239, 35.9064, 139.0124, 36.3221];
    let mut times = vec![];
    for len in [4, 5] {
        let mut t = vec![0f32; 6 * len];
        for lev in 0..3 {
            for k in 0..len - 1 {
                t[lev * len + k] = 600.;
                t[(3 + lev) * len + k + 1] = 600.;
            }
        }
        times.extend(t);
    }
    api.set_network_ll(&ll, &[4, 5], &[0, 1, 2, 3, 4, 0, 5, 6, 7], &times, &[12., 12., 12., 8., 8., 8.], &[10., 10.]).unwrap();
    api.set_fares(1.5, 0.1, &[8.; 18]).unwrap();
    api.set_fare_curve(&[0., 1.5, 10., 2.5, 100., 8.]).unwrap();
    let figures = |api: &mut IntercityApi| {
        assert!(api.solve(0));
        let per = api.per(0).unwrap();
        let p = api.pass_on(&per.g, &per.p, 0, &skims_for(&per.g, &per.p, &api.want), &per.xi);
        let s = api.pass_streamed(&per.g, &per.p, 0, &per.xi, 0);
        assert_eq!((p.rail.to_bits(), p.c_rail.to_bits(), p.e_rail.to_bits()), (s.rail.to_bits(), s.c_rail.to_bits(), s.e_rail.to_bits()));
        (p.rail, p.c_rail)
    };
    let base = figures(&mut api);
    assert!(base.0 > 0. && base.1 > 0.);
    for same in [r#"{"JP": 1}"#, r#"{"US": 2}"#, "{}"] {
        api.set_fare_factors(same).unwrap();
        let f = figures(&mut api);
        assert_eq!((f.0.to_bits(), f.1.to_bits()), (base.0.to_bits(), base.1.to_bits()), "{same}");
    }
    api.set_fare_factors(r#"{"JP": 2}"#).unwrap();
    let dear = figures(&mut api);
    assert!(dear.0 < base.0 && dear.1 < base.1, "{dear:?} against {base:?}");
    api.set_fare_factors(r#"{"JP": 0.5}"#).unwrap();
    let cheap = figures(&mut api);
    assert!(cheap.0 > base.0 && cheap.1 > base.1, "{cheap:?} against {base:?}");
    assert!(api.set_fare_factors(r#"{"JP": 0}"#).is_err());
    // T-1300: the rail trips' fare sensitivity is what a dearer base fare does (it moves no route),
    // with a fare factor too; on the base and rate's line, since a bent curve sets every fare itself
    api.set_fare_curve(&[]).unwrap();
    for fx in ["{}", r#"{"JP": 2}"#] {
        api.set_fare_factors(fx).unwrap();
        api.set_fares(1.5, 0.1, &[8.; 18]).unwrap();
        assert!(api.solve(0));
        let (r0, d) = (api.summary(0)[2], api.summary(0)[10]);
        api.set_fares(1.55, 0.1, &[8.; 18]).unwrap();
        assert!(api.solve(0));
        let r1 = api.summary(0)[2];
        let want = (r1 - r0) / (0.05 * api.base.fare_w as f64);
        assert!(d < 0. && (d - want).abs() < 0.03 * want.abs(), "{fx}: sensitivity {d}, a dearer base fare gives {want}");
    }
}
/// T-880: the trip explorer between zones counts the commutes between zones as a selection's
/// bubbles do: a selection of every Tokyo site against Gunma, legs and rail legs a day, each way.
#[test]
fn the_explorer_counts_commutes_as_the_selection() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let mut api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    let ll = [139.7671, 35.6812, 139.7006, 35.6896, 139.7774, 35.7138, 139.6239, 35.9064, 139.0124, 36.3221];
    let mut t = vec![0f32; 6 * 5];
    for lev in 0..3 {
        for k in 0..4 {
            t[lev * 5 + k] = 600.;
            t[(3 + lev) * 5 + k + 1] = 600.;
        }
    }
    api.set_network_ll(&ll, &[5], &[0, 1, 2, 3, 4], &t, &[12., 12., 12.], &[10.]).unwrap();
    api.set_fares(1.5, 0.1, &[8.; 10]).unwrap();
    solve_all(&mut api);
    let m = api.m.clone();
    let place_of = |g: &Group| m.zones[m.cell_zone[g.lo]].place;
    let gnm = m.zones[m.cell_zone[m.groups.last().unwrap().lo]].place;
    let tyo = place_of(&m.groups[0]);
    assert_ne!(tyo, gnm);
    let (mut xy, mut share) = (vec![], vec![]);
    for g in m.groups.iter().filter(|g| place_of(g) == tyo) {
        for i in 0..g.city.len() {
            xy.extend([g.city.x[i], g.city.y[i]]);
            share.push(1f32);
        }
    }
    let (np, ng) = (m.places.len(), m.n_gid);
    let tail = 1 + 8 * np + 7 * ng;
    let sel = api.selection_flows_day(tyo as u32, &xy, &share, WEEKDAY as u32);
    for dir in 0..2usize {
        let (legs, rail) = (sel[tail + 4 * gnm + 2 * dir], sel[tail + 4 * gnm + 2 * dir + 1]);
        let (mut e_legs, mut e_rail) = (0., 0.);
        for q in 0..PERIODS {
            let h = api.market_trip(q, WEEKDAY as u32, tyo as u32, &xy, &share, dir as u32, gnm as u32, -1);
            e_legs += h[22];
            e_rail += h[23];
        }
        assert!(legs > 0. && rail > 0., "dir {dir}: commutes {legs}, rail {rail}");
        assert!((e_legs - legs).abs() < 1e-6 * legs, "dir {dir}: explorer legs {e_legs}, selection {legs}");
        assert!((e_rail - rail).abs() < 1e-6 * rail.max(1.), "dir {dir}: explorer rail {e_rail}, selection {rail}");
    }
}
/// T-761: the market's pass on a captured network natively, by phase, with its tables' sizes.
/// `TW_T761_NET=<checks/t761-plain-net.mjs's output> cargo test --release --features demand t761_market_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn t761_market_bench() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("TW_T761_NET").unwrap()).unwrap()).unwrap();
    let f = |k: &str| j[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<f64>>();
    let ids: Vec<String> = j["markets"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let idr: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let t = now_ms();
    let mut a = IntercityApi::new(&real_world(root, Some(&idr))).unwrap();
    eprintln!("open {:.0} ms", now_ms() - t);
    let fr = &j["frame"];
    let frame = Frame::equirect(fr["lon"].as_f64().unwrap(), fr["lat"].as_f64().unwrap()).with_kind(crate::track::frame::FrameKind::Merc);
    let ll: Vec<f64> = f("stXY").chunks_exact(2).flat_map(|p| { let (x, y) = frame.to_lonlat(p[0], p[1]); [x, y] }).collect();
    let u = |k: &str| f(k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
    let g32 = |k: &str| f(k).into_iter().map(|v| v as f32).collect::<Vec<f32>>();
    let t = now_ms();
    a.set_network_ll(&ll, &u("lineN"), &u("stops"), &g32("times"), &g32("tph"), &g32("cars")).unwrap();
    a.set_fares(j["fare"]["base"].as_f64().unwrap() as f32, j["fare"]["perKm"].as_f64().unwrap() as f32, &g32("hopKm")).unwrap();
    eprintln!("set_network {:.0} ms: served zones {}, pairs {}, commute pairs {}, far-zone entries {}", now_ms() - t, a.served.len(), a.pairs.len(), a.cpairs.len(), a.st_pat.col.len());
    let t = now_ms();
    assert!(a.solve(0));
    eprintln!("solve(0) {:.0} ms, rail {:.1}", now_ms() - t, a.summary(0)[2]);
    let per = a.per(0).unwrap();
    let (g, p) = (&per.g, &per.p);
    let s = g.n_st;
    let mut r = vec![f32::INFINITY; s * s];
    let mut q = DestSearch::new(g, p);
    let t = now_ms();
    let (mut n, mut bytes) = (0, 0);
    for d in (0..s).filter(|&d| a.want[d]) {
        bytes += q.run(d, Some(&mut r)).bytes();
        n += 1;
    }
    eprintln!("a sweep: {n} searches {:.0} ms, {:.0} MB of strategies (estimate {:.0} MB)", now_ms() - t, bytes as f64 / 1048576., (n * strategy_bytes(g)) as f64 / 1048576.);
    let t = now_ms();
    let d = a.demand(g, &r, &r, p, 0, &[]);
    eprintln!("demand {:.0} ms; od non-zero {}, od_c non-zero {}", now_ms() - t, d.od.iter().filter(|&&v| v != 0.).count(), d.od_c.v.len());
}
/// T-1175: the market's solve with and without the everyday pairs across zone lines, on scratch
/// market packs (`TW_T1175_DIR`: `<id>.market.json` and `<id>.market.everyday.json` of `TW_T1175_IDS`)
/// and three made-up lines from Guangzhou into Foshan.
/// `cargo test --release --features demand t1175_market_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn t1175_market_bench() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dir = std::env::var("TW_T1175_DIR").unwrap();
    let ids = std::env::var("TW_T1175_IDS").unwrap_or("gua,fos,shz,dog".into());
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let by_id: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/market-rows.json")).unwrap()).unwrap();
    let m = &reg["market"];
    let rows: Vec<serde_json::Value> = reg["cities"].as_array().unwrap().iter().filter_map(|c| by_id.get(c["id"].as_str()?).cloned()).collect();
    let world = serde_json::json!({"frame": m["frame"], "rule": m["rule"], "places": m["places"].as_array().unwrap().len(), "rows": rows});
    // lines of stations about 2 km apart, (lon, lat) ends
    let lines = [((113.27, 23.13), (113.10, 23.02), 10), ((113.24, 23.20), (113.05, 23.15), 10), ((113.30, 23.05), (113.15, 22.92), 10)];
    let (mut ll, mut n, mut stops, mut times, mut tph, mut cars) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for &((x0, y0), (x1, y1), k) in &lines {
        let s0 = ll.len() as u32 / 2;
        for i in 0..k {
            let f = i as f64 / (k - 1) as f64;
            ll.extend([x0 + f * (x1 - x0), y0 + f * (y1 - y0)]);
            stops.push(s0 + i as u32);
        }
        n.push(k as u32);
        for dir in 0..2 {
            for _ in 0..3 {
                times.extend((0..k).map(|i| if (dir == 0 && i + 1 < k) || (dir == 1 && i > 0) { 150f32 } else { 0. }));
            }
        }
        tph.extend([12f32, 8., 6.]);
        cars.push(6f32);
    }
    for with in [false, true] {
        let packs: Vec<String> = ids.split(',').map(|id| std::fs::read_to_string(format!("{dir}/{id}.market{}.json", if with { ".everyday" } else { "" })).unwrap()).collect();
        let t = now_ms();
        let mut a = IntercityApi::new(&format!("{{\"world\":{},\"zones\":[{}]}}", world, packs.join(","))).unwrap();
        let open = now_ms() - t;
        let t = now_ms();
        a.set_network_ll(&ll, &n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(0.3, 0.02, &vec![2.; 2 * stops.len()]).unwrap();
        let set = now_ms() - t;
        let t = now_ms();
        solve_all(&mut a);
        let solve = now_ms() - t;
        let t = now_ms();
        for q in 0..PERIODS {
            let total = a.seg(q);
            assert!(a.crowd_with_external(q, &total));
        }
        let crowd = now_ms() - t;
        let s: Vec<f64> = (0..8).map(|i| (0..PERIODS).map(|q| a.part_summary(q)[i]).sum()).collect();
        eprintln!(
            "everyday pairs {with}: {} pairs ({} served), open {open:.0} ms, set_network {set:.0} ms, five solves {solve:.0} ms, a crowding round {crowd:.0} ms; longer {:.0} (rail {:.0}), commute legs {:.0} (rail {:.0}), everyday legs {:.0} (rail {:.0}, walking {:.0} of {:.0})",
            a.m.everyday.len(), a.epairs.len(), s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
        );
        if !with {
            continue;
        }
        // Anita's ceiling tests (notes/T-636.md) on the everyday pairs, every pair open: rail's share
        // of rail + car by the pairs' own ratio curve, with a straight train between their zones' centres,
        // a 3-minute walk at each end (1.5 a minute), 2.5 minutes' wait; A 140 km/h and free, B 100 km/h
        // and $1.50 + $0.05 a km
        let pc = commute_params();
        let pe = pc.everyday();
        let (mut t, mut sa, mut sb) = (0f64, 0f64, 0f64);
        let mut bands = [[0f64; 3]; 4];
        for e in &a.m.everyday {
            let zh = &a.m.zones[e.h as usize];
            let Some(zi) = (0..a.m.zones.len()).find(|&k| a.m.zones[k].gid == e.a) else { continue };
            let za = &a.m.zones[zi];
            let km = a.m.zone_km(zh, za) as f32;
            let car = drive_min(km) * pc.w_drive + za.park + pc.car_own_min() + km * ROAD_DETOUR * pc.car_own_km_min();
            let base = 2. * 3. * pc.w_walk + 2.5;
            let ra = base + km / 140. * 60.;
            let rb = base + km / 100. * 60. + FARE_MIN_PER_USD * (1.5 + 0.05 * km);
            let w = e.trips as f64;
            let (x, y) = (pe.ratio_share(ra, car) as f64, pe.ratio_share(rb, car) as f64);
            t += w;
            sa += w * x;
            sb += w * y;
            let b = [2., 5., 10., f32::INFINITY].iter().position(|&e| km < e).unwrap();
            bands[b][0] += w;
            bands[b][1] += w * x;
            bands[b][2] += w * y;
        }
        eprintln!("ceilings on the everyday pairs: A {:.1}%, B {:.1}% of rail + car over {t:.0} outings", 100. * sa / t, 100. * sb / t);
        for (b, n) in ["0-2", "2-5", "5-10", "10+"].iter().enumerate() {
            let v = bands[b];
            eprintln!("  {n} km: {:.0} outings, A {:.1}%, B {:.1}%", v[0], 100. * v[1] / v[0].max(1e-9), 100. * v[2] / v[0].max(1e-9));
        }
    }
}
/// The world composition as the app opens it: the market rows of the registry's zones (T-776:
/// market-rows.json, in registry order) and the packs of `ids` (all that exist with None).
pub(crate) fn real_world(root: &str, ids: Option<&[&str]>) -> String {
    // T-780: a copy of the registry and rows (`TW_REGISTRY`, `TW_ROWS`) while a batch is adding zones
    let reg = std::env::var("TW_REGISTRY").unwrap_or(format!("{root}/app/src/game/registry.json"));
    let rows = std::env::var("TW_ROWS").unwrap_or(format!("{root}/app/src/game/market-rows.json"));
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(reg).unwrap()).unwrap();
    let by_id: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(rows).unwrap()).unwrap();
    let m = &reg["market"];
    let rows: Vec<serde_json::Value> = reg["cities"].as_array().unwrap().iter().filter_map(|c| by_id.get(c["id"].as_str()?).cloned()).collect();
    let packs: Vec<String> = reg["cities"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["id"].as_str())
        .filter(|id| ids.map_or(true, |l| l.contains(id)))
        .filter_map(|id| std::fs::read_to_string(format!("{root}/data/packs/{id}.market.json")).ok())
        .collect();
    let world = serde_json::json!({"frame": m["frame"], "rule": m["rule"], "places": m["places"].as_array().unwrap().len(), "rows": rows});
    format!("{{\"world\":{},\"zones\":[{}]}}", world, packs.join(","))
}
/// T-718: the regional layer shares a trip's riders over station pairs at the local layers' scale
/// for its band (T-710), and rail's cost is their logsum at that scale.
#[test]
fn regional_station_pairs_take_the_local_scale() {
    // stations 0 and 1 near the home, 2 at work: 0 -> 2 costs 30, 1 -> 2 costs 33
    let inf = f32::INFINITY;
    let sk = Skims { s: 3, n: 3, r: vec![0., inf, 30., inf, 0., 33., inf, inf, 0.], walk: vec![], walk_max: vec![], slot: vec![], next: vec![], board: vec![], order: vec![], change: vec![], entry: vec![] };
    let p = Params::default();
    // a pair whose car takes 60 perceived minutes, past 12 km: the band's scale, not softened
    let th = p.soften(p.band_unsoftened(p.band_of_d2(20e3 * 20e3)), 60.);
    assert!((th - 1. / 3.).abs() < 1e-6);
    // 5 perceived minutes to either station
    let w = (-p.beta * 5.0f32).exp();
    let mut terms = vec![];
    let (c, sum) = rail_logsum(&sk.r, sk.s, 0., &[(0, w), (1, w)], &[(2, 1.)], p.beta, th, &mut terms);
    let share = terms[0].2 / sum;
    assert!((share - 1. / (1. + (-1f64).exp())).abs() < 1e-4, "3 minutes at a third: {share}");
    let want = 5. + 30. - (1. + (-1f32).exp()).ln() / th;
    assert!((c - want).abs() < 1e-3, "{c} against {want}");
    let (c, sum) = rail_logsum(&sk.r, sk.s, 0., &[(1, w)], &[(1, 1.)], p.beta, th, &mut terms);
    assert!(c.is_infinite() && sum == 0. && terms.is_empty(), "no route");
}
/// T-780: a captured network's market (`checks/t761-plain-net.mjs`'s output), set up natively.
#[cfg(test)]
pub(crate) fn t780_net(path: &str, chunk: f64) -> IntercityApi {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let f = |k: &str| j[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<f64>>();
    let ids: Vec<String> = j["markets"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let idr: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let mut a = IntercityApi::new(&real_world(root, Some(&idr))).unwrap();
    let fr = &j["frame"];
    let frame = Frame::equirect(fr["lon"].as_f64().unwrap(), fr["lat"].as_f64().unwrap()).with_kind(crate::track::frame::FrameKind::Merc);
    let ll: Vec<f64> = f("stXY").chunks_exact(2).flat_map(|p| { let (x, y) = frame.to_lonlat(p[0], p[1]); [x, y] }).collect();
    let u = |k: &str| f(k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
    let g32 = |k: &str| f(k).into_iter().map(|v| v as f32).collect::<Vec<f32>>();
    a.set_complexes(&u("complex"));
    a.set_chunk(chunk);
    a.set_network_ll(&ll, &u("lineN"), &u("stops"), &g32("times"), &g32("tph"), &g32("cars")).unwrap();
    a.set_fares(j["fare"]["base"].as_f64().unwrap() as f32, j["fare"]["perKm"].as_f64().unwrap() as f32, &g32("hopKm")).unwrap();
    a
}
/// T-780: the market's pairs by distance and size, with their station pairs and rail trips.
/// `TW_T780_NET=<plain net> cargo test --release --features demand t780_census -- --ignored --nocapture`
#[test]
#[ignore]
fn t780_census() {
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), 0.);
    eprintln!("served zones {}, pairs {}, commute pairs {}, stations {}, wanted {}", a.served.len(), a.pairs.len(), a.cpairs.len(), a.want.len(), a.want.iter().filter(|&&w| w).count());
    let t = now_ms();
    assert!(a.solve(1));
    eprintln!("solve(1) {:.0} ms", now_ms() - t);
    let per = a.per(1).unwrap();
    let bands = [25., 50., 100., 250., 500., 1e9];
    let mut by = vec![[0f64; 5]; bands.len() * 3];
    let mut sz = vec![[0f64; 5]; 8];
    let mut hist_terms = 0f64;
    for (pi, pair) in a.pairs.iter().enumerate() {
        let (za, zb) = (&a.m.zones[pair.a as usize], &a.m.zones[pair.b as usize]);
        let pr = a.m.pr(za.place, zb.place);
        let class = if za.place == zb.place { 0 } else if pr.near { 1 } else { 2 };
        let d = a.m.zone_km(za, zb);
        let k = bands.iter().position(|&b| d < b).unwrap();
        let terms = (a.access[pair.a as usize].len() * a.access_work[pair.b as usize].len()) as f64;
        hist_terms += terms;
        let rail = per.pair_rail[pi] as f64;
        let e = &mut by[class * bands.len() + k];
        e[0] += 1.; e[1] += pair.trips * SHARES[1] as f64; e[2] += terms; e[3] += rail;
        let t = pair.trips * SHARES[1] as f64;
        let k = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 1e18].iter().position(|&b| t < b).unwrap();
        let e = &mut sz[k];
        e[0] += 1.; e[1] += t; e[2] += terms; e[3] += rail;
    }
    eprintln!("terms {hist_terms:.0}; period 1 rail {:.0}, commutes' rail {:.0}", per.rail, per.c_rail);
    for c in 0..3 {
        for (k, b) in bands.iter().enumerate() {
            let e = by[c * bands.len() + k];
            eprintln!("class {c} under {b:>6} km: pairs {:>9} trips {:>10.0} terms {:>11.0} rail {:>9.1}", e[0], e[1], e[2], e[3]);
        }
    }
    for (k, b) in [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 1e18].iter().enumerate() {
        let e = sz[k];
        eprintln!("trips a period under {b:>6}: pairs {:>9} trips {:>10.0} terms {:>11.0} rail {:>9.1}", e[0], e[1], e[2], e[3]);
    }
    // the logsums alone against the whole of `demand`
    let (g, p) = (&per.g, &per.p);
    let s = g.n_st;
    let mut r = vec![f32::INFINITY; s * s];
    let mut q = DestSearch::new(g, p);
    let t = now_ms();
    for d in (0..s).filter(|&d| a.want[d]) {
        q.run(d, Some(&mut r));
    }
    eprintln!("a sweep {:.0} ms", now_ms() - t);
    let t = now_ms();
    let mut terms = vec![];
    let mut x = 0f64;
    for pair in &a.pairs {
        let (c, _) = rail_logsum(&r, s, p.fare_min, &a.access[pair.a as usize], &a.access_work[pair.b as usize], p.beta, p.soften(p.band_unsoftened(pair.band as usize), pair.car.at(FARE_MIN_PER_USD)), &mut terms);
        x += c as f64;
    }
    eprintln!("logsums alone {:.0} ms ({x:.0})", now_ms() - t);
    let t = now_ms();
    let d = a.demand(g, &r, &r, p, 1, &[]);
    eprintln!("demand {:.0} ms, rail {:.0}", now_ms() - t, d.rail);
}
/// T-780: the market's period 1 at free flow with its pairs sampled in chunks, against every pair.
/// `TW_T780_NET=<plain net> TW_T780_CHUNKS=0,0.3,1 cargo test --release --features demand t780_compare -- --ignored --nocapture`
#[test]
#[ignore]
fn t780_compare() {
    let net = std::env::var("TW_T780_NET").unwrap();
    let chunks: Vec<f64> = std::env::var("TW_T780_CHUNKS").unwrap_or("0,1".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let mut base: Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, f64)> = None;
    // how far b is from a: the total's change, the share of the whole moved, and the worst element
    // against the larger of itself and 1% of the biggest
    let moved = |a: &[f32], b: &[f32]| {
        let (ta, tb) = (a.iter().map(|&v| v as f64).sum::<f64>(), b.iter().map(|&v| v as f64).sum::<f64>());
        let l1: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum();
        let top = a.iter().fold(0f32, |m, &v| m.max(v)) as f64;
        let worst = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs() / (x as f64).max(y as f64).max(0.01 * top).max(1e-9)).fold(0., f64::max);
        format!("total {:+.3}%, moved {:.2}%, worst {:.1}%", 100. * (tb - ta) / ta.max(1e-9), 100. * l1 / ta.max(1e-9), 100. * worst)
    };
    for &c in &chunks {
        let t = now_ms();
        let mut a = t780_net(&net, c);
        let t_net = now_ms() - t;
        let t = now_ms();
        assert!(a.solve(1));
        let t_solve = now_ms() - t;
        let per = a.per(1).unwrap();
        let (board, seg, entries) = (a.board(1), a.seg(1), a.entries(1));
        let cor: Vec<f64> = a.corridor_summary(1).chunks_exact(4).map(|v| v[3]).collect();
        let rail = per.rail;
        eprintln!("chunk {c}: pairs {}, far-zone entries {}, set up {t_net:.0} ms, solve {t_solve:.0} ms, longer trips' rail {rail:.1}, commutes' {:.1}", a.pairs.len(), a.st_pat.col.len(), per.c_rail);
        if let Some((b0, s0, e0, c0, r0)) = &base {
            let cf = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            eprintln!("  rail {:+.3}%; boardings {}; hop loads {}; station entries {}; corridors {}", 100. * (rail - r0) / r0, moved(b0, &board), moved(s0, &seg), moved(e0, &entries), moved(&cf(c0), &cf(&cor)));
            let top = e0.iter().fold(0f32, |m, &v| m.max(v));
            let w = (0..e0.len()).max_by(|&i, &j| ((e0[i] - entries[i]).abs() / e0[i].max(entries[i]).max(0.01 * top)).total_cmp(&((e0[j] - entries[j]).abs() / e0[j].max(entries[j]).max(0.01 * top)))).unwrap();
            eprintln!("  worst station {w}: {:.1} against {:.1} entries (busiest {top:.1}, {} stations over 1% of it)", e0[w], entries[w], e0.iter().filter(|&&v| v > 0.01 * top).count());
        } else {
            base = Some((board, seg, entries, cor, rail));
        }
    }
}
/// T-780 (T-767): the streamed pass keeping more of its strategies between sweeps, with sampled pairs.
/// `TW_T780_NET=<plain net> TW_T780_KEEP=256,768 cargo test --release --features demand t780_keep -- --ignored --nocapture`
#[test]
#[ignore]
fn t780_keep() {
    let mut a = t780_net(&std::env::var("TW_T780_NET").unwrap(), MARKET_CHUNK);
    assert!(a.solve(1));
    let per = a.per(1).unwrap();
    let (g, p, ix) = (&per.g, &per.p, &per.xi);
    eprintln!("a strategy {:.0} KB, {} destinations", strategy_bytes(g) as f64 / 1024., a.want.iter().filter(|&&w| w).count());
    for mb in std::env::var("TW_T780_KEEP").unwrap_or("256,768".into()).split(',').map(|v| v.parse::<usize>().unwrap()) {
        let t = now_ms();
        let s = a.pass_streamed(g, p, 1, ix, mb << 20);
        eprintln!("keeping {mb} MB: pass {:.0} ms, rail {:.1}", now_ms() - t, s.rail);
    }
}
/// T-780: each curve worked out once gives every zone pair's trips to the bit.
#[test]
fn eachway_with_its_curves_is_eachway() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    let m = &api.m;
    for a in m.zones.iter().step_by(7) {
        for b in &m.zones {
            assert_eq!(m.eachway_with(a, b, m.curves(a, b)).to_bits(), m.eachway(a, b).to_bits());
        }
    }
}
/// T-780: sampled pairs keep each zone's trips with each place after it, always solve a pair of at
/// least a chunk, carry whole chunks or a big pair's own trips, and are the same every time.
#[test]
fn sampled_pairs_keep_each_zones_trips_with_each_place() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    let m = &api.m;
    let served: Vec<usize> = (0..m.zones.len()).collect();
    let joined = |_: usize, _: usize| true;
    let (any0, each0) = solved_pairs(m, &served, &joined, 0.);
    let chunk = 1.;
    let (any, each) = solved_pairs(m, &served, &joined, chunk);
    assert_eq!((&any, &each), (&solved_pairs(m, &served, &joined, chunk).0, &solved_pairs(m, &served, &joined, chunk).1));
    assert!(each.len() * 10 < each0.len() * 8, "{} sampled pairs of {}", each.len(), each0.len());
    // per zone and later place: (exact trips, sampled trips)
    let n = served.len();
    let mut by: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let (mut bit, mut k0, mut k1) = (0, 0, 0);
    for i in 0..n {
        for k in i + 1..n {
            let (on0, on1) = (any0[bit / 64] >> (bit % 64) & 1 == 1, any[bit / 64] >> (bit % 64) & 1 == 1);
            bit += 1;
            let e = by.entry((i, m.zones[k].place)).or_default();
            let t0 = if on0 { k0 += 1; each0[k0 - 1] } else { 0. };
            if on0 {
                e.0 += t0;
            }
            if on1 {
                k1 += 1;
                e.1 += each[k1 - 1];
                assert!(on0, "a sampled pair that is not a pair");
            } else if t0 >= chunk {
                panic!("a pair of {t0} trips left out");
            }
        }
    }
    for ((i, p), (t0, t1)) in by {
        assert!((t0 - t1).abs() <= 1e-9 * t0.max(1.), "zone {i} with place {p}: {t0} trips, {t1} sampled");
    }
}/// T-780: with sampled pairs no far zone of a selection shows more rail than trips, and its rail by
/// place is the zones' added up.
#[test]
fn sampled_selection_rail_stays_within_trips() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let mut api = IntercityApi::new(&real_world(root, Some(&["tyo", "gnm"]))).unwrap();
    api.set_chunk(1.);
    // the streamed pass's two lines across Tokyo, changing at Tokyo station
    let ll = [139.7671, 35.6812, 139.7387, 35.6285, 139.6970, 35.5313, 139.6222, 35.4660, 139.7006, 35.6896, 139.7774, 35.7138, 139.6239, 35.9064, 139.0124, 36.3221];
    let mut times = vec![];
    for len in [4, 5] {
        let mut t = vec![0f32; 6 * len];
        for lev in 0..3 {
            for k in 0..len - 1 {
                t[lev * len + k] = 600.;
                t[(3 + lev) * len + k + 1] = 600.;
            }
        }
        times.extend(t);
    }
    api.set_network_ll(&ll, &[4, 5], &[0, 1, 2, 3, 4, 0, 5, 6, 7], &times, &[12., 12., 12., 8., 8., 8.], &[10., 10.]).unwrap();
    api.set_fares(1.5, 0.1, &[8.; 18]).unwrap();
    solve_all(&mut api);
    let m = &api.m;
    let pop = |c: usize| { let (g, ci) = m.site(c); g.city.pop[ci] };
    let mut keys: Vec<((i32, i32, usize), usize)> = m.site_key.iter().map(|(&k, &c)| (k, c)).collect();
    keys.sort_by(|a, b| pop(b.1).total_cmp(&pop(a.1)).then(a.0.cmp(&b.0)));
    let (np, ng) = (m.places.len(), m.n_gid);
    let head = 1 + 8 * np;
    let mut any = 0.;
    for &(key, _) in keys.iter().take(12) {
        let xy = [(key.0 as f32 + 0.5) * SITE_M, (key.1 as f32 + 0.5) * SITE_M];
        let out = api.selection_flows(key.2 as u32, &xy, &[1.]);
        for d in 0..2 {
            let mut by = vec![0f64; np];
            for z in 0..ng {
                let (t, r) = (out[head + 4 * z + 2 * d], out[head + 4 * z + 2 * d + 1]);
                assert!(r <= t * (1. + 1e-6) + 1e-9, "zone {z}: rail {r} above all {t}");
                by[m.gid_place[z] as usize] += r;
                any += r;
            }
            for (p, &r) in by.iter().enumerate() {
                let place = out[2 + 8 * p + 4 * d];
                assert!((place - r).abs() <= 1e-6 * place.max(1.), "place {p}: {place} against its zones' {r}");
            }
        }
    }
    assert!(any > 0., "the lines carry some of the selections");
}
/// T-1204: a signature of every market figure on the world composition with `TW_T1204_IDS`'s packs
/// (default Paris, Lille, Brussels, London, Amsterdam) and three made-up lines, with the composing
/// time and heap. Run before and after a change to the composition: the hashes must agree.
/// `cargo test --release --features demand t1204_signature -- --ignored --nocapture`
#[test]
#[ignore]
fn t1204_signature() {
    use crate::demand::alloc;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let ids = std::env::var("TW_T1204_IDS").unwrap_or("par,lil,bru,lon,ams".into());
    let ids: Vec<&str> = ids.split(',').filter(|s| !s.is_empty()).collect();
    let reg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let by_id: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/market-rows.json")).unwrap()).unwrap();
    let m = &reg["market"];
    let rows: Vec<serde_json::Value> = reg["cities"].as_array().unwrap().iter().filter_map(|c| by_id.get(c["id"].as_str()?).cloned()).collect();
    let world = serde_json::json!({"frame": m["frame"], "rule": m["rule"], "places": m["places"].as_array().unwrap().len(), "rows": rows}).to_string();
    let packs: Vec<String> = ids.iter().map(|id| std::fs::read_to_string(format!("{root}/data/packs/{id}.market.json")).unwrap()).collect();
    let mut h = MarketPacks::new(&world).unwrap();
    let heap0 = alloc::current();
    alloc::reset_peak();
    let t = now_ms();
    let empty = h.market().unwrap();
    eprintln!("empty composition {:.0} ms, heap +{:.1} MB (peak +{:.1})", now_ms() - t, alloc::mb(alloc::current() - heap0), alloc::mb(alloc::peak() - heap0));
    drop(empty);
    for p in &packs {
        h.add(p).unwrap();
    }
    let heap1 = alloc::current();
    alloc::reset_peak();
    let t = now_ms();
    let mut a = h.market().unwrap();
    eprintln!("composition of {} packs {:.0} ms, heap +{:.1} MB (peak +{:.1})", ids.len(), now_ms() - t, alloc::mb(alloc::current() - heap1), alloc::mb(alloc::peak() - heap1));
    let t = now_ms();
    let again = h.market().unwrap();
    eprintln!("the same composition again {:.0} ms, heap +{:.1} MB with both held", now_ms() - t, alloc::mb(alloc::current() - heap1));
    drop(again);
    // lines of stations, (lon, lat) ends: Paris across, Paris to Brussels by Lille, London across
    let lines = [((2.25, 48.84), (2.45, 48.88), 10), ((2.36, 48.88), (4.35, 50.84), 12), ((-0.20, 51.50), (0.02, 51.53), 10)];
    let (mut ll, mut n, mut stops, mut times, mut tph, mut cars) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for &((x0, y0), (x1, y1), k) in &lines {
        let s0 = ll.len() as u32 / 2;
        for i in 0..k {
            let f = i as f64 / (k - 1) as f64;
            ll.extend([x0 + f * (x1 - x0), y0 + f * (y1 - y0)]);
            stops.push(s0 + i as u32);
        }
        n.push(k as u32);
        let sec = if k == 12 { 900f32 } else { 150. };
        for dir in 0..2 {
            for _ in 0..3 {
                times.extend((0..k).map(|i| if (dir == 0 && i + 1 < k) || (dir == 1 && i > 0) { sec } else { 0. }));
            }
        }
        tph.extend([12f32, 8., 6.]);
        cars.push(8f32);
    }
    let heap2 = alloc::current();
    let t = now_ms();
    a.set_network_ll(&ll, &n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(0.3, 0.02, &vec![2.; 2 * stops.len()]).unwrap();
    solve_all(&mut a);
    for q in 0..PERIODS {
        let total = a.seg(q);
        assert!(a.crowd_with_external(q, &total));
    }
    eprintln!("network, five solves and a round {:.0} ms, heap +{:.1} MB", now_ms() - t, alloc::mb(alloc::current() - heap2));
    let t = now_ms();
    let modes = a.corridor_modes();
    eprintln!("corridor_modes {:.0} ms; corridors with longer trips {} of {}, with commutes {}", now_ms() - t, a.corridor_trips().iter().filter(|&&t| t > 0.).count(), a.corridor_trips().len(), a.commute_base().iter().filter(|&&t| t > 0.).count());
    drop(modes);
    let mut sig: Vec<(String, u64)> = vec![];
    let fnv = |v: &mut dyn Iterator<Item = u64>| v.fold(0xcbf29ce484222325u64, |h, x| (h ^ x).wrapping_mul(0x100000001b3));
    let f64s = |v: &[f64]| fnv(&mut v.iter().map(|x| x.to_bits()));
    let f32s = |v: &[f32]| fnv(&mut v.iter().map(|x| x.to_bits() as u64));
    sig.push(("corridor_pairs".into(), fnv(&mut a.corridor_pairs().iter().map(|&x| x as u64))));
    sig.push(("corridor_trips".into(), f64s(&a.corridor_trips())));
    sig.push(("commute_base".into(), f64s(&a.commute_base())));
    sig.push(("corridor_modes".into(), f32s(&a.corridor_modes())));
    sig.push(("info".into(), f64s(&a.info())));
    sig.push(("pair_counts".into(), f64s(&a.pair_counts())));
    sig.push(("zone_boxes".into(), f64s(&a.zone_boxes())));
    sig.push(("open_places".into(), fnv(&mut a.open_places().iter().map(|&x| x as u64))));
    for q in 0..PERIODS {
        // (the summary's fifth figure is the solve's time)
        let mut sm = a.summary(q);
        if sm.len() > 4 {
            sm[4] = 0.;
        }
        for (k, v) in [("summary", sm), ("part_summary", a.part_summary(q)), ("corridor_summary", a.corridor_summary(q)), ("corridor_rail", a.corridor_rail(q)), ("commute_corridors", a.commute_corridors(q)), ("elsewhere", a.elsewhere(q)), ("corridor_shares", a.corridor_shares(q))] {
            sig.push((format!("{k} {q}"), f64s(&v)));
        }
        for (k, v) in [("seg", a.seg(q)), ("board", a.board(q)), ("alight", a.alight(q)), ("entries", a.entries(q)), ("exits", a.exits(q)), ("fare_extra", a.fare_extra(q)), ("commute_loads", a.commute_loads(q)), ("everyday_loads", a.everyday_loads(q))] {
            sig.push((format!("{k} {q}"), f32s(&v)));
        }
    }
    for day in 0..3u32 {
        for s in [0u32, 5, 12, 20, 25, 31] {
            sig.push((format!("station_flows_day {s} {day}"), f64s(&a.station_flows_day(s, day))));
        }
        // a selection: the sites of a few zones of each open place, in their place's plane
        for z in [0usize, 3, 10, 40, 80] {
            let Some(zz) = a.m.zones.get(z) else { continue };
            let (mut xy, mut share) = (vec![], vec![]);
            for s in 0..a.m.n_sites() {
                if a.m.cell_zone[s] == z {
                    let (g, i) = a.m.site(s);
                    xy.extend([g.city.x[i], g.city.y[i]]);
                    share.push(1f32);
                }
            }
            let p = zz.place as u32;
            sig.push((format!("selection_flows_day z{z} {day}"), f64s(&a.selection_flows_day(p, &xy, &share, day))));
            sig.push((format!("selection_all_day z{z} {day}"), f64s(&a.selection_all_day(p, &xy, &share, day, 31))));
            sig.push((format!("selection_rail_day z{z} {day}"), f64s(&a.selection_rail_day(p, &xy, &share, day))));
        }
    }
    let pt = a.place_trips(&[(0, 1), (100, 200), (5, 5)], 30.);
    sig.push(("place_trips".into(), f64s(&pt.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>())));
    let all = fnv(&mut sig.iter().map(|s| s.1));
    for (k, v) in &sig {
        println!("SIG {k} {v:016x}");
    }
    println!("SIG all {all:016x}");
    eprintln!("signature over {} parts: {all:016x}", sig.len());
}
/// T-1506: past the best `ACCESS_TOP`, an access list keeps a slot for each track component none of
/// those is on, best first, up to `ACCESS_MAX`; a station with no train line takes none.
#[test]
fn access_lists_reserve_slots_for_unlinked_track() {
    // stations 0-10 metro on one track; 11 a bus stop on its own; 12, 13 and 14 three unlinked
    // inter-city lines, 14 at two stations (14 and 15)
    let comp: Vec<u32> = (0..16).map(|s| if s >= 11 { s.min(14) as u32 } else { 0 }).collect();
    let mut on_train = vec![true; 16];
    on_train[11] = false;
    let w = |s: usize| match s {
        11 => 5.,
        12 => 0.3,
        13 => 0.2,
        14 => 0.1,
        15 => 0.25,
        _ => 10. - 0.5 * s as f32,
    };
    let all: BTreeMap<usize, f32> = (0..16).map(|s| (s, w(s))).collect();
    let got: Vec<usize> = access_list(all, false, &comp, &on_train).into_iter().map(|e| e.0).collect();
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7, 12, 15]);
    // track already listed takes no slot
    let some: BTreeMap<usize, f32> = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13].into_iter().map(|s| (s, w(s))).collect();
    let got: Vec<usize> = access_list(some, false, &comp, &on_train).into_iter().map(|e| e.0).collect();
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7, 13]);
    // up to `ACCESS_TOP` the list is the best stations, as before
    let few: BTreeMap<usize, f32> = [(12, 0.1), (3, 2.), (11, 1.), (5, 0.)].into_iter().collect();
    let got: Vec<usize> = access_list(few, true, &comp, &on_train).into_iter().map(|e| e.0).collect();
    assert_eq!(got, vec![3, 11, 12]);
}
/// T-1506: a zone among nine metro lines reaches an inter-city station that no metro line links to,
/// so the inter-city line carries its trips (with the best 8 stations alone it carried none).
#[test]
fn an_unlinked_inter_city_station_gets_a_slot() {
    let mut sites: Vec<String> = (0..10).map(|k| format!("[{},200,20000,10000,0]", 500 * k)).collect();
    sites.push("[2500,3000,200,100,0]".into());
    sites.push("[100000,0,200000,100000,1]".into());
    let mut a = IntercityApi::new(&format!(r#"{{"zone_m":6000,"trip_rate":0.015,"sites":[{}]}}"#, sites.join(","))).unwrap();
    // metro stations 0-9 500 m apart, lines between neighbours; the inter-city line 10 -> 11
    let mut xy: Vec<f32> = (0..10).flat_map(|k| [500. * k as f32, 0.]).collect();
    xy.extend([2500., 3200., 100000., 0.]);
    let (mut n, mut stops, mut times) = (vec![], vec![], vec![]);
    for (s0, s1, sec) in (0..9).map(|k| (k, k + 1, 90f32)).chain([(10, 11, 1800.)]) {
        n.push(2u32);
        stops.extend([s0 as u32, s1 as u32]);
        let mut t = [0f32; 12];
        for lev in 0..3 {
            t[lev * 2] = sec;
            t[(3 + lev) * 2 + 1] = sec;
        }
        times.extend(t);
    }
    let tph = vec![12f32; 3 * n.len()];
    let cars = vec![6f32; n.len()];
    a.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let zone_at = |x: f32, y: f32| {
        let g = &a.m.groups[0];
        let c = (0..g.city.len()).min_by(|&i, &j| ((g.city.x[i] - x).powi(2) + (g.city.y[i] - y).powi(2)).total_cmp(&((g.city.x[j] - x).powi(2) + (g.city.y[j] - y).powi(2)))).unwrap();
        a.m.cell_zone[g.lo + c]
    };
    let z = zone_at(0., 200.);
    assert_eq!(z, zone_at(2500., 3000.), "the metro's homes and the inter-city station's share a zone");
    let svc = a.svc.as_ref().unwrap();
    let h = svc.st_map[10] as usize;
    let list = &a.access[z];
    assert!(list.len() > ACCESS_TOP && list.len() <= ACCESS_MAX);
    assert!(list[..ACCESS_TOP].iter().all(|e| e.0 != h), "the metro fills the best 8: {list:?}");
    assert!(list.iter().any(|e| e.0 == h), "the inter-city station has a slot: {list:?}");
    a.solve(0);
    assert!(a.summary(0)[2] > 0., "rail between the zones");
}
/// T-1511: a zone among nine metro lines whose only way to an unlinked inter-city station is a
/// ride gets that ride, though the nine lines' rides are all cheaper, and so the station's slot.
#[test]
fn a_ride_reaches_unlinked_track_past_the_cheapest_lines() {
    let mut sites: Vec<String> = (0..10).map(|k| format!("[{},200,20000,10000,0]", 500 * k)).collect();
    sites.push("[100000,0,200000,100000,1]".into());
    let mut a = IntercityApi::new(&format!(r#"{{"zone_m":6000,"trip_rate":0.015,"sites":[{}]}}"#, sites.join(","))).unwrap();
    // metro stations 0-9 500 m apart, lines between neighbours; the inter-city line 10 -> 11, its
    // station 10 beyond every walk from the metro's homes
    let mut xy: Vec<f32> = (0..10).flat_map(|k| [500. * k as f32, 0.]).collect();
    xy.extend([2500., 5000., 100000., 0.]);
    let (mut n, mut stops, mut times) = (vec![], vec![], vec![]);
    for (s0, s1, sec) in (0..9).map(|k| (k, k + 1, 90f32)).chain([(10, 11, 1800.)]) {
        n.push(2u32);
        stops.extend([s0 as u32, s1 as u32]);
        let mut t = [0f32; 12];
        for lev in 0..3 {
            t[lev * 2] = sec;
            t[(3 + lev) * 2 + 1] = sec;
        }
        times.extend(t);
    }
    let tph = vec![12f32; 3 * n.len()];
    let cars = vec![6f32; n.len()];
    a.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let svc = a.svc.as_ref().unwrap();
    let h = svc.st_map[10] as usize;
    let g = &a.m.groups[0];
    let c = (0..g.city.len()).min_by(|&i, &j| (g.city.x[i].powi(2) + (g.city.y[i] - 200.).powi(2)).total_cmp(&(g.city.x[j].powi(2) + (g.city.y[j] - 200.).powi(2)))).unwrap();
    let z = a.m.cell_zone[g.lo + c];
    let list = &a.access[z];
    assert!(list.iter().any(|e| e.0 == h), "the inter-city station has a slot: {list:?}");
    a.solve(0);
    assert!(a.summary(0)[2] > 0., "rail between the zones");
}
/// T-1511: past the `ride_lines` cheapest lines, a place's rides add the cheapest station of each
/// track component those miss, at most `RIDE_TRACK_MAX`, never one without a train line; without
/// `track` they are the cheapest lines' alone.
#[test]
fn rides_reserve_track_components() {
    // one place at the origin; lines 0-11 start 1-12 km east of it and run 50 km north; lines 0-7
    // are taken as one track, lines 8-11 each its own, line 9 as without trains
    let mut a = IntercityApi::new(r#"{"zone_m":6000,"trip_rate":0.015,"sites":[[0,0,20000,10000,0],[100000,0,200000,100000,1]]}"#).unwrap();
    let mut xy: Vec<f32> = vec![];
    for k in 0..12 {
        xy.extend([1000. * (k + 1) as f32, 0., 1000. * (k + 1) as f32, 50000.]);
    }
    let n = vec![2u32; 12];
    let stops: Vec<u32> = (0..24).collect();
    let mut times = vec![];
    for _ in 0..12 {
        let mut t = [0f32; 12];
        for lev in 0..3 {
            t[lev * 2] = 60.;
            t[(3 + lev) * 2 + 1] = 60.;
        }
        times.extend(t);
    }
    let tph = vec![12f32; 36];
    let cars = vec![6f32; 12];
    a.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let svc = a.svc.as_ref().unwrap();
    let net = DemandApi::period_net(svc, 0);
    let first = |line: usize| svc.st_map[2 * line];
    let mut comp = vec![0u32; net.n_stations()];
    let mut on_train = vec![true; net.n_stations()];
    for line in 8..12 {
        comp[first(line) as usize] = line as u32;
    }
    on_train[first(9) as usize] = false;
    let p = a.base.clone();
    let g = &a.m.groups[0];
    let place = [(g.kz.cell_zone[0] as usize, 0f32, 0f32)];
    let st = |rides: &[Ride]| rides.iter().map(|r| r.0).collect::<Vec<_>>();
    let (_, plain) = access_options(&g.kz, &net, &p, &place, End::Home, None);
    assert_eq!(st(&plain), (0..p.ride_lines).map(first).collect::<Vec<_>>());
    let (_, with) = access_options(&g.kz, &net, &p, &place, End::Home, Some((&comp, &on_train)));
    assert_eq!(st(&with[..p.ride_lines]), st(&plain));
    assert_eq!(st(&with[p.ride_lines..]), vec![first(8), first(10)], "the next unlinked lines, the one without trains passed over");
}
