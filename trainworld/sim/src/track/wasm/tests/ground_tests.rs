use super::*;

/// A terrain file (`cost::ClassGrid`) of one class over the whole degree block round a point.
fn terrain_block(lon: f64, lat: f64, class: u8) -> Vec<u8> {
    let (c0, r0) = (((lon + 180.0).floor() as i32) * 240, ((lat + 90.0).floor() as i32) * 240);
    let mut b = crate::track::cost::TERRAIN_MAGIC.to_vec();
    for v in [240u32, 240, c0 as u32, r0 as u32, 240] {
        b.extend(v.to_le_bytes());
    }
    b.push(class);
    let mut n = 240 * 240usize;
    loop {
        let x = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 { b.push(x); break; }
        b.push(x | 0x80);
    }
    b
}

#[test]
fn terrain_reprices_blueprints_and_spares_bored_tunnels() {
    // T-1270: 3 km at ground and 3 km at -2, then a mountain grid lands under both
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let deep: Vec<f64> = [0.0, 0.0, 0.0, 2000.0, -2.0, 0.0, 0.0, 3000.0, 2000.0, -2.0].to_vec();
    assert_eq!(a.add_route(&deep, &[], 2, false), 0, "{}", a.issues());
    let before = a.blueprint_cost();
    assert!((before - (3.0 * 37.5 + 3.0 * 120.0)).abs() < 1e-6, "{before}");
    let ll = a.to_lonlat(1500.0, 1000.0);
    assert!(!a.add_terrain(b"TWT1"));
    assert!(a.add_terrain(&terrain_block(ll[0], ll[1], 3)));
    let after = a.blueprint_cost();
    assert!((after - (3.0 * 262.5 + 3.0 * 120.0)).abs() < 1e-6, "{after}");
    let rows = a.blueprint_cost_items();
    // T-1839: track rows carry the terrain class in the fifth field, the ground's part is no row
    let hilly: f64 = rows.chunks_exact(7).filter(|r| r[0] == 0.0 && r[4] == 3.0).map(|r| r[6]).sum();
    assert!((hilly - 3.0 * 262.5).abs() < 1e-6 &&!rows.chunks_exact(7).any(|r| r[0] == 6.0), "{rows:?}");
    assert!((rows.chunks_exact(7).map(|r| r[6]).sum::<f64>() - after).abs() < 1e-6);
    // the tool's preview prices the same and says how much ground it crosses in each class
    let more: Vec<f64> = free(0.0, -1000.0).iter().chain(free(2000.0, -1000.0).iter()).copied().collect();
    let out = a.preview_route(&more, &[], 2, false);
    assert_eq!(out[0], 1.0);
    assert!((out[1] - 2.0 * 262.5).abs() < 1e-6 && (out[4] - 3.5).abs() < 1e-9, "{:?}", &out[..5]);
    let g = a.preview_ground();
    // T-1887: then over water
    assert!(g[3] > 1999.0 && g[0] < 1.0 && g[4] == 0.0, "{g:?}");
    // a loaded network keeps the terrain in force
    let saved = a.save();
    assert!(a.load(&saved));
    assert!((a.blueprint_cost() - after).abs() < 1e-6);
}

#[test]
fn terrain_blocks_under_and_drop_keep_what_edges_stand_on() {
    // T-2901: a blueprint across two blocks reports both; a block under it cannot be dropped, and one
    // not under it can
    let mut a = api();
    let ll = a.to_lonlat(0.0, 0.0);
    let block = |lon: f64, lat: f64| crate::track::cost::block_key(lon, lat);
    let k0 = block(ll[0], ll[1]);
    // from the origin to the next block east: well over the 0.015 degrees left to 74 W's east side
    let east = a.w.net.frame().from_lonlat(ll[0].floor() + 1.2, ll[1]);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(east.0, east.1).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let k1 = block(ll[0].floor() + 1.2, ll[1]);
    assert!(k0 != k1);
    let under = a.terrain_blocks_under(true);
    assert!(under.contains(&k0) && under.contains(&k1), "{under:?}");
    assert!(a.add_terrain(&terrain_block(ll[0], ll[1], 3)));
    assert!(a.add_terrain(&terrain_block(ll[0], ll[1] + 5.0, 3)));
    assert!(!a.drop_terrain(k0), "an edge stands on it");
    assert!(a.drop_terrain(block(ll[0], ll[1] + 5.0)), "nothing stands on it");
    // what a blueprint wanted is reported when the index lists it, and nothing else; a built edge counts for neither list's blueprint
    let mut idx = b"TWI1".to_vec();
    idx.extend((-65i16).to_le_bytes());
    idx.extend(150u16.to_le_bytes());
    idx.extend((-180i16).to_le_bytes());
    idx.extend(360u16.to_le_bytes());
    idx.extend(1u32.to_le_bytes());
    let mut bits = vec![0u8; 6750];
    let bit = (k1.div_euclid(360) - 90 + 65) * 360 + k1.rem_euclid(360);
    bits[(bit >> 3) as usize] |= 1 << (bit & 7);
    idx.extend(bits);
    assert!(a.set_terrain_index(&idx));
    assert!(!a.set_terrain_index(b"TWI"));
    assert_eq!(a.take_terrain_wanted(), vec![k1], "k0 is installed, k1 is listed and not");
    assert_eq!(a.terrain_blocks_under(true), vec![k1], "k0 is not listed in this index");
}

/// Road tiles over the box of network points `(x0, y0)`-`(x1, y1)`: a road running north-south every
/// `every` tile units (about 0.46 m each at New York), lined up across tiles as real roads are.
fn add_road_grid(a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64, every: i32) {
    use crate::track::roads::{merc, EXTENT};
    let (w, s) = { let ll = a.to_lonlat(x0, y0); merc(ll[0], ll[1]) };
    let (e, n) = { let ll = a.to_lonlat(x1, y1); merc(ll[0], ll[1]) };
    let tile = |v: f64| (v / EXTENT).floor() as u32;
    for tx in tile(w.min(e))..=tile(w.max(e)) {
        for ty in tile(s.min(n))..=tile(s.max(n)) {
            let (mut pts, mut starts) = (vec![], vec![]);
            let ox = (tx as f64 * EXTENT) as i64;
            let mut gx = (every as i64 - ox.rem_euclid(every as i64)) as i32;
            while gx < EXTENT as i32 {
                starts.push((pts.len() / 2) as u32);
                pts.extend([gx, -64, gx, EXTENT as i32 + 64]);
                gx += every;
            }
            let weights = vec![1u8; starts.len()];
            assert!(a.add_road_tile(tx, ty, EXTENT, &pts, &starts, &weights));
        }
    }
}

#[test]
fn roads_charge_ground_track_and_stay_through_splits_and_loads() {
    // T-1929: 3 km at ground and 3 km at +1, then roads every ~92 m (200 units) land under both
    let mut a = api();
    let ground: Vec<f64> = free(0.0, 0.0).iter().chain(free(3000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ground, &[], 2, false), 0, "{}", a.issues());
    let high = [0.0, 0.0, 0.0, -1000.0, 1.0, 0.0, 0.0, 3000.0, -1000.0, 1.0];
    assert_eq!(a.add_route(&high, &[], 2, false), 0, "{}", a.issues());
    let before = a.blueprint_cost();
    assert!((before - 3.0 * (37.5 + 75.0)).abs() < 1e-6, "{before}");
    a.w.net.take_repriced();
    add_road_grid(&mut a, -500.0, -1500.0, 3500.0, 1500.0, 200);
    // T-2230: only edge prices changed, so no line's rows are read again; the crossings added tile
    // by tile are a whole pass's over every tile, to the bit; the viaduct keeps none
    assert!(!a.w.net.repriced());
    let net = &a.w.net;
    let frame = net.frame();
    for e in (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e)) {
        if crate::track::roads::off_ground(net.edge_vert(e)) {
            assert!(net.edge_roads(e).is_empty());
            continue;
        }
        let (pieces, mut whole) = (net.edge_pieces(e), vec![]);
        crate::track::roads::crossings(net.edge_len[e as usize], |s| { let (x, y, _) = crate::track::geom::pos_at(pieces, s); frame.to_lonlat(x, y) }, &net.roads, &mut whole);
        assert!(!whole.is_empty() && net.edge_roads(e) == &whole[..], "edge {e}: {:?} against {whole:?}", net.edge_roads(e));
    }
    let after = a.blueprint_cost();
    let crossings = (after - before) / ROAD_CROSSING_COST;
    assert!((crossings - crossings.round()).abs() < 1e-9 && (30.0..=36.0).contains(&crossings), "{crossings} crossings");
    // the quote has them as one row, its count and price, and the rows still add up to the total
    let rows = a.blueprint_cost_items();
    let road: Vec<&[f64]> = rows.chunks_exact(7).filter(|r| r[0] == 7.0).collect();
    assert!(road.len() == 1 && road[0][5] == crossings && (road[0][6] - crossings * ROAD_CROSSING_COST).abs() < 1e-9, "{rows:?}");
    assert!((rows.chunks_exact(7).map(|r| r[6]).sum::<f64>() - after).abs() < 1e-6);
    // the tool's preview counts them on the ground and not on a viaduct
    let more: Vec<f64> = free(0.0, 1000.0).iter().chain(free(2000.0, 1000.0).iter()).copied().collect();
    let out = a.preview_route(&more, &[], 2, false);
    let g = a.preview_ground();
    assert!(out[0] == 1.0 && (20.0..=24.0).contains(&g[5]) && (g[6] - g[5] * ROAD_CROSSING_COST).abs() < 1e-9, "{g:?}");
    assert!((out[1] - (2.0 * 37.5 + g[6])).abs() < 1e-6, "{:?}", &out[..5]);
    let up = [0.0, 0.0, 0.0, 1000.0, 1.0, 0.0, 0.0, 2000.0, 1000.0, 1.0];
    a.preview_route(&up, &[], 2, false);
    assert_eq!(a.preview_ground()[5], 0.0);
    // a station splitting the ground track keeps every crossing
    let e = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && a.w.net.node_level[a.w.net.edge_a[e as usize] as usize] == 0).unwrap();
    assert_eq!(a.add_station(2, e, 1530.0, 0.0, 200, "Mid".into()), 0, "{}", a.issues());
    let split: f64 = (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).map(|e| crate::track::roads::charge(a.w.net.edge_vert(e), a.w.net.edge_roads(e)).0).sum();
    assert_eq!(split, crossings);
    // constructing pays them; a save loaded keeps the tiles in force
    a.set_cash(1e6);
    let cash = a.cash();
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let paid = cash - a.cash();
    let costs = a.costs();
    assert!((costs[1] - costs[0]).abs() < 1e-6 && paid > crossings * ROAD_CROSSING_COST, "{paid} {costs:?}");
    let saved = a.save();
    assert!(a.load(&saved));
    assert!((a.costs()[1] - costs[1]).abs() < 1e-6, "{:?} {costs:?}", a.costs());
}

#[test]
fn upkeep_is_cheaper_for_track_that_stays_on_the_ground() {
    // T-1931, T-2619: US$35k a km a game day on the ground, 50k on a viaduct, 100k in cut and cover, 130k bored
    let mut up = vec![];
    for level in [0.0, 1.0, 3.0, -1.0, -2.0, -3.0] {
        let mut a = api();
        a.set_cash(1e6);
        let ends = [0.0, 0.0, 0.0, 0.0, level, 0.0, 0.0, 10_000.0, 0.0, level];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        up.push(a.upkeep()[1]);
    }
    for (got, want) in up.iter().zip([0.35, 0.5, 0.5, 1.0, 1.3, 1.3]) {
        assert!((got - want).abs() < 1e-3, "{up:?}");
    }
    let params = api().money_params();
    assert_eq!((params[7], params[15], params[16], params[17]), (350.0, 500.0, 1000.0, 1300.0));
    // an edge pays the dearest structure it touches
    use crate::track::cost::track_upkeep_km as rate;
    assert_eq!((rate(0, 0), rate(0, 2), rate(-1, 3), rate(-3, 1), rate(-2, -1), rate(-1, 0)), (350.0, 500.0, 1000.0, 1300.0, 1300.0, 1000.0));
    // a station below ground pays 200k, any other 50k (T-2619)
    use crate::track::cost::station_upkeep as st;
    assert_eq!((st(-3), st(-1), st(0), st(2), params[8], params[22]), (2000.0, 2000.0, 500.0, 500.0, 500.0, 2000.0));
}

#[test]
fn upkeep_of_a_ten_km_line_and_its_shares() {
    // T-867: 10 km of double track at -1 with a station every km: $100k a km (T-1931: cut and
    // cover) and $200k an underground station a day (T-2619)
    let mut a = api();
    a.set_cash(1e6);
    let mut st = vec![];
    for i in 0..10 {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * 1000.0, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, format!("S{i}")), 0, "{}", a.issues());
    }
    assert_eq!(a.upkeep(), vec![0.0; 10], "blueprint costs nothing");
    let l1 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l1, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, "L1".into()), 0, "{}", a.issues());
    // by headway, 6, 4 and 2 an hour
    assert_eq!(a.set_auto_headway(l1, 600, 900, 1800), 0, "{}", a.issues());
    assert_eq!(a.set_line_cars(l1, 6), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let up = a.upkeep();
    assert!((up[0] - 3.2).abs() < 1e-3 && (up[3] - 10.0).abs() < 1e-3 && up[4] == 11.0, "{up:?}");
    let hours = [8.0, 10.0, 6.0];
    let day: f64 = (0..3).map(|l| hours[l] * a.running_cost(l)).sum();
    let len = a.lines_info()[9] / 1000.0;
    assert!((day - 2000.0 * len / 10.0 * (4.0 + 2.0 * 6.0) * ECONOMY / 1e6).abs() < 1e-6, "{day} for {len} km");
    let s = a.line_upkeep(&hours);
    assert!(s.len() == 4 && s[0] == l1 as f64 && (s[1] - 3.2).abs() < 1e-3 && s[3].abs() < 1e-9, "{s:?}");
    // a second line as busy over the first half shares that half's track and stations
    let l2 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l2, &st[..6], 6, 4, 2, 30.0, 180.0, 0, 0x2d7dd2, "L2".into()), 0, "{}", a.issues());
    assert_eq!(a.set_auto_headway(l2, 600, 900, 1800), 0, "{}", a.issues());
    let s = a.line_upkeep(&hours);
    let share = |l: u32| s.chunks(2).find(|c| c[0] == l as f64).map_or(0.0, |c| c[1]);
    assert!((share(l1) - 2.35).abs() < 1e-3 && (share(l2) - 0.85).abs() < 1e-3, "{s:?}");
    // without trains the first line carries nothing; its own half goes to the rest
    assert_eq!(a.set_train_counts(l1, 0, 0, 0), 0, "{}", a.issues());
    let s = a.line_upkeep(&hours);
    assert!(!s.chunks(2).any(|c| c[0] == l1 as f64), "{s:?}");
    assert!((s[s.len() - 1] - 1.5).abs() < 1e-3 && (s.chunks(2).map(|c| c[1]).sum::<f64>() - 3.2).abs() < 1e-3, "{s:?}");
}

/// T-3474: a line running part of its plan shares the upkeep of that part only; the built track and
/// stations past an unbuilt station in its plan go to the rest.
#[test]
fn a_partly_built_line_pays_upkeep_on_the_part_it_runs() {
    let mut a = api();
    a.set_cash(1e6);
    let mut st = vec![];
    for i in 0..10 {
        let from = if i == 0 { [0.0, 0.0, 0.0, 0.0, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * 1000.0, 0.0, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            st.push(a.nodes_info()[0] as u32);
        }
        st.push(a.route_end());
    }
    for (i, &n) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 120, format!("S{i}")), 0, "{}", a.issues());
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, "L1".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    // S5 back to blueprint: S0-S4 and S6-S10 tie at 4 km and five stops, and the earlier runs
    let mut s5 = a.w.net.station_data(st[5]).unwrap();
    s5.built = false;
    a.w.apply(Op::Station { node: st[5], data: Some(s5) }).unwrap();
    assert_eq!(a.line_part(l), vec![0, 5, 0, 4]);
    assert_eq!(a.line_run_stops(l), st[..5].to_vec());
    // 10 km at $100k and 10 built underground stations at $200k a day; the line's part is 4 km and 5 stations
    let s = a.line_upkeep(&[8.0, 10.0, 6.0]);
    let share = |l: f64| s.chunks(2).find(|c| c[0] == l).map_or(0.0, |c| c[1]);
    assert!((share(l as f64) - 1.4).abs() < 1e-3 && (share(-1.0) - 1.6).abs() < 1e-3, "{s:?}");
}
