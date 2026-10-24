use super::*;
/// T-708: quad drawn through points, chains of random segments from the drawn pair's loose end
/// (half of them nearly straight, where a capped curve has tangents of a metre or two), in both
/// frames: no refusal is ever a curve that does not fit, which nothing the player drew sets.
#[test]
fn quad_chains_never_refuse_a_curve_that_does_not_fit() {
    let mut seed: u64 = 12345;
    let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64) / ((1u64 << 53) as f64) };
    let mut misfits = vec![];
    for world in [false, true] {
        for _chain in 0..40 {
            let mut a = TrackApi::new(-73.985, 40.758, true);
            if world { assert!(a.set_empty_frame(1)); }
            a.set_route_tangent(true);
            let o = if world { let p = a.from_lonlat(-73.985, 40.758); (p[0], p[1]) } else { (0.0, 0.0) };
            let ends = [0.0, 0.0, o.0, o.1, 0.0, 0.0, 0.0, o.0 + 600.0, o.1, 0.0];
            assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
            let mut end = a.route_end();
            let mut heading = 0.0f64;
            for _seg in 0..8 {
                let turn = (rnd() - 0.5) * 2.0 * if rnd() < 0.5 { 3.0 } else { 170.0 };
                let dist = 100.0 + rnd() * 2500.0;
                let lvl = if rnd() < 0.2 { (rnd() * 3.0) as i32 - 1 } else { a.w.net.node_level[end as usize] as i32 };
                let off = turn.to_radians();
                let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
                let (x, y) = (x0 + dist * (heading + off).cos(), y0 + dist * (heading + off).sin());
                let e = [1.0, end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, lvl as f64];
                let pv = a.preview_route(&e, &[], 4, false);
                if pv[0] == 0.0 {
                    if a.issue_kinds().contains("ArcDoesNotFit") {
                        misfits.push(format!("world {world} turn {turn:.2} dist {dist:.0}"));
                    }
                    continue;
                }
                assert_eq!(a.add_route(&e, &[], 4, false), 0);
                end = a.route_end();
                let p = &a.w.net.node_ports[end as usize];
                heading = if p.side[0] == 1 { a.w.net.node_heading[end as usize] + PI } else { a.w.net.node_heading[end as usize] };
            }
        }
    }
    assert!(misfits.is_empty(), "{misfits:#?}");
}

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn named(a: &TrackApi, name: &str) -> usize {
    (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).count()
}
fn route(a: &mut TrackApi, from: [f64; 5], x: f64, y: f64, tracks: u8) -> u32 {
    let ends: Vec<f64> = from.iter().copied().chain([0.0, 0.0, x, y, 0.0]).collect();
    assert_eq!(a.add_route(&ends, &[], tracks, false), 0, "{}", a.issues());
    a.route_end()
}
/// Every drawn point of `ex` lies `QUAD_SPACING` from the nearest of `local`, away from `taper`
/// metres of each end.
fn beside(a: &TrackApi, ex: &[u32], local: &[u32], taper: f64) {
    for &p in ex {
        let len = a.w.net.edge_len[p as usize];
        for q in a.edge_render(p).chunks(3) {
            let s = a.w.net.project(p, q[0], q[1]).0;
            if s < taper || s > len - taper {
                continue;
            }
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off at {s}");
        }
    }
}

/// T-708: doubling up. A double-track run of three pieces (a joint, then a station) gets an
/// express pair beside it, one piece each, joints and a twin station beside, loose ends at
/// both ends; one undo takes it all back. Track with a pair beside already, single track and
/// a crossover are refused. A branch from a junction swings in to meet at it. Constructed
/// track gets a blueprint pair, priced as the preview said.
#[test]
fn double_up_makes_quad() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    let j = route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 1000.0, 0.0, 2);
    let end = route(&mut a, [1.0, j as f64, 0.0, 0.0, 0.0], 1800.0, 300.0, 2);
    let second = *live(&a).last().unwrap();
    let (sx, sy, _) = geom::pos_at(a.w.net.edge_pieces(second), 450.0);
    assert_eq!(a.add_station(2, second, sx, sy, 160, "Mid".into()), 0, "{}", a.issues());
    let local = live(&a);
    assert_eq!(local.len(), 3);
    let price = a.preview_double_up(local[0]);
    assert!(price > 0.0, "{}", a.issues());
    let before = a.total_cost();
    let mut old = TrackApi::new(-73.985, 40.758, true);
    assert!(old.load(&a.save()));
    assert_eq!(a.double_up(local[2]), 0, "{}", a.issues());
    assert!((a.total_cost() - before - price).abs() < 1e-6);
    // T-733: a crossover past the station, splitting each pair twice
    assert_eq!(a.crossover_edges().len(), 2, "a crossover past the station");
    let on_old = |e: u32| {
        let r = a.edge_render(e);
        let k = r.len() / 6 * 3;
        local.iter().any(|&l| old.w.net.project(l, r[k], r[k + 1]).1 < 1.0)
    };
    let plain: Vec<u32> = live(&a).into_iter().filter(|&e| !a.w.net.edge_crossover[e as usize]).collect();
    let ex: Vec<u32> = plain.iter().copied().filter(|&e| !on_old(e)).collect();
    let now_local: Vec<u32> = plain.iter().copied().filter(|&e| on_old(e)).collect();
    assert_eq!((ex.len(), now_local.len()), (5, 5), "one express piece per piece, and the crossover's two splits on each pair");
    beside(&a, &ex, &now_local, 0.0);
    assert_eq!(named(&a, "Mid"), 2, "the station's twin");
    let ends = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 1).count();
    assert_eq!(ends, 4, "loose ends beside both ends");
    assert!(ex.iter().all(|&e| !a.w.net.edge_built[e as usize] && a.w.net.edge_tracks[e as usize] == 2));
    // quad now: no second time, from any piece of either pair
    for &e in local.iter().chain(&ex) {
        assert_ne!(a.double_up(e), 0);
        assert!(a.issue_kinds().contains("Alongside"), "{e}: {}", a.issue_kinds());
    }
    assert_eq!(a.undo(), 0);
    assert_eq!(live(&a), local);
    assert_eq!(named(&a, "Mid"), 1);
    // a branch off the middle of the first piece meets it at a junction: the pair swings in
    let (bx, by, _) = geom::pos_at(a.w.net.edge_pieces(local[0]), 500.0);
    let ends: Vec<f64> = [2.0, local[0] as f64, bx, by, 0.0, 0.0, 0.0, 900.0, -900.0, 0.0].to_vec();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let junction = (0..a.w.net.node_count() as u32).find(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 3).unwrap();
    let tip = a.route_end();
    let branch = live(&a).into_iter().find(|&e| a.w.net.edge_a[e as usize] == tip || a.w.net.edge_b[e as usize] == tip).unwrap();
    assert_eq!(a.double_up(branch), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[junction as usize].n, 4, "the express pair meets the junction");
    assert_eq!(a.undo(), 0);
    // single track is refused
    route(&mut a, [1.0, end as f64, 0.0, 0.0, 0.0], 2600.0, 900.0, 1);
    let single = live(&a).into_iter().find(|&e| a.w.net.edge_tracks[e as usize] == 1).unwrap();
    assert_ne!(a.double_up(single), 0);
    assert!(a.issue_kinds().contains("Tracks"));
    assert_eq!(a.undo(), 0);
    // constructed track: the pair is blueprint, what the preview priced
    a.cash = 1e6;
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let built = live(&a);
    let price = a.preview_double_up(local[1]);
    let cash = a.cash;
    assert_eq!(a.double_up(local[1]), 0, "{}", a.issues());
    // the crossover splits constructed track: re-summed shapes, rounding only (T-624)
    assert!((a.cash - cash).abs() < 1e-5, "blueprint costs nothing until constructed: {}", a.cash - cash);
    let ex: Vec<u32> = live(&a).into_iter().filter(|e| !built.contains(e)).collect();
    assert_eq!(ex.len(), 9, "five express pieces, two splits of the local pair and two diagonals");
    assert_eq!(ex.iter().filter(|&&e| !a.w.net.edge_built[e as usize]).count(), 7, "the express pieces and the crossover are blueprint");
    assert!(price > 0.0);
}

/// T-711: the drawing tag's cost multiplier (`preview_route`'s fifth number) is the level's on
/// plain ground for quad as for double (0.5x at ground, 1.0x at +1), and quad's price is what
/// the edit adds, twice double's.
#[test]
fn quad_preview_multiplier_and_price() {
    for level in [0.0, 1.0] {
        let ends = [0.0, 0.0, 0.0, 0.0, level, 0.0, 0.0, 1000.0, 0.0, level];
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        let double = a.preview_route(&ends, &[], 2, false);
        let quad = a.preview_route(&ends, &[], 4, false);
        assert_eq!((double[0], quad[0]), (1.0, 1.0), "{}", a.issues());
        let want = level_mult(level as i8);
        assert!((double[4] - want).abs() < 1e-6, "double {} at {level}", double[4]);
        assert!((quad[4] - want).abs() < 1e-6, "quad {} at {level}", quad[4]);
        assert!((quad[1] / double[1] - 2.0).abs() < 0.01, "quad {} against double {}", quad[1], double[1]);
        let before = a.total_cost();
        assert_eq!(a.add_route(&ends, &[], 4, false), 0);
        assert!((a.total_cost() - before - quad[1]).abs() < 1e-9, "the tag's price is what the edit adds");
    }
}

/// T-708: doubling up double track drawn on from a loose quad end carries on from the express
/// pair's loose end there, with no new end beside it.
#[test]
fn double_up_carries_on_from_quad() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    let q = route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 800.0, 0.0, 4);
    let quad = live(&a);
    route(&mut a, [1.0, q as f64, 0.0, 0.0, 0.0], 1600.0, 100.0, 2);
    let piece = *live(&a).last().unwrap();
    assert_eq!(a.double_up(piece), 0, "{}", a.issues());
    let ends = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 1).count();
    assert_eq!(ends, 4, "the two pairs' ends at each end, none left between");
    let ex: Vec<u32> = live(&a).into_iter().filter(|e| !quad.contains(e) && *e != piece).collect();
    beside(&a, &ex, &[piece], 0.0);
}

/// The plain track nearest (x, y): a station's spot.
fn track_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    live(a).into_iter().filter(|&e| !a.w.net.edge_crossover[e as usize]).min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap()
}
fn stations(a: &TrackApi, name: &str) -> Vec<u32> {
    (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).collect()
}

/// T-733: a station on quad track gets a crossover just past it, clear of the platform, at
/// the full price of two junctions, in the same undo step; a line changes pairs over it.
#[test]
fn quad_station_gets_a_crossover_past_it() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 3000.0, 0.0, 4);
    let edges0 = live(&a).len();
    assert_eq!(a.preview_station(2, track_at(&a, 800.0, 0.0), 800.0, 0.0, 200), 0, "{}", a.issues());
    assert_eq!(live(&a).len(), edges0, "a preview changes nothing");
    let cost0 = a.total_cost();
    assert_eq!(a.add_station(2, track_at(&a, 800.0, 0.0), 800.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(stations(&a, "West").len(), 2);
    let xs = a.crossover_edges();
    assert_eq!(xs.len(), 2, "one crossover");
    // its ends clear of the platform (100 m each way) by QUAD_CROSSOVER_CLEAR, and no farther than the crossover's length beyond
    for &x in &xs {
        for n in [a.w.net.edge_a[x as usize], a.w.net.edge_b[x as usize]] {
            let off = (a.w.net.node_x[n as usize] - 800.0).abs();
            assert!(off > 100.0 + QUAD_CROSSOVER_CLEAR - 1.0 && off < 100.0 + QUAD_CROSSOVER_CLEAR + CROSSOVER_LEN + 1.0, "end {off} m from the station");
        }
    }
    // full price: deleting it takes off two junctions
    let with = a.total_cost();
    assert_eq!(a.delete_edge(xs[0]), 0, "{}", a.issues());
    let two_junctions = 2.0 * crate::track::cost::junction_cost(0, false, false);
    assert!((with - a.total_cost() - two_junctions).abs() < 1e-6, "{} not {two_junctions}", with - a.total_cost());
    assert_eq!(a.undo(), 0);
    // one undo takes the stations and the crossover together
    assert_eq!(a.undo(), 0);
    assert!(stations(&a, "West").is_empty() && a.crossover_edges().is_empty());
    assert_eq!(live(&a).len(), edges0);
    assert!((a.total_cost() - cost0).abs() < 1e-6);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.crossover_edges().len(), 2);
    // a second station: a line from the local pair at one to the express pair at the other
    assert_eq!(a.add_station(2, track_at(&a, 2200.0, 0.0), 2200.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    assert_eq!(a.crossover_edges().len(), 4);
    let pick = |a: &TrackApi, name: &str, y: f64| *stations(a, name).iter().min_by(|&&p, &&q| (a.w.net.node_y[p as usize] - y).abs().total_cmp(&(a.w.net.node_y[q as usize] - y).abs())).unwrap();
    let (local, express) = (pick(&a, "West", 0.0), pick(&a, "East", QUAD_SPACING));
    assert!(a.w.net.node_y[express as usize].abs() > 1.0, "the express pair's station");
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[local, express], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "X".into()), 0, "{}", a.issues());
    let xs = a.crossover_edges();
    assert!(a.line_path(id).iter().any(|p| xs.contains(&(p >> 1))), "across a crossover");
}

/// T-733: where there is no room past a quad station (its track ends within reach both ways),
/// the stations go in without a crossover.
#[test]
fn quad_station_without_room_has_no_crossover() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 500.0, 0.0, 4);
    assert_eq!(a.add_station(2, track_at(&a, 250.0, 0.0), 250.0, 0.0, 200, "Short".into()), 0, "{}", a.issues());
    assert_eq!(stations(&a, "Short").len(), 2, "both stations");
    assert!(a.crossover_edges().is_empty());
    assert_eq!(a.undo(), 0);
    assert!(stations(&a, "Short").is_empty());
    // plain double track gets none either
    let mut b = TrackApi::new(-73.985, 40.758, true);
    route(&mut b, [0.0, 0.0, 0.0, 0.0, 0.0], 3000.0, 0.0, 2);
    assert_eq!(b.add_station(2, track_at(&b, 800.0, 0.0), 800.0, 0.0, 200, "Plain".into()), 0, "{}", b.issues());
    assert!(b.crossover_edges().is_empty());
}

/// T-646: the stations a line passes, by gap and in path order; on quad track only those on
/// the line's own pair; adding them is one set_stops edit.
#[test]
fn passed_stations_in_path_order() {
    for tracks in [2u8, 4] {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 6000.0, 0.0, tracks);
        let xs = [500.0, 1700.0, 2900.0, 4100.0, 5500.0];
        for (k, &x) in xs.iter().enumerate() {
            assert_eq!(a.add_station(2, track_at(&a, x, 0.0), x, 0.0, 120, format!("S{k}")), 0, "{}", a.issues());
        }
        let on_drawn = |a: &TrackApi, k: usize| *stations(a, &format!("S{k}")).iter().min_by(|&&p, &&q| a.w.net.node_y[p as usize].abs().total_cmp(&a.w.net.node_y[q as usize].abs())).unwrap();
        let s: Vec<u32> = (0..xs.len()).map(|k| on_drawn(&a, k)).collect();
        let id = a.new_line_id();
        assert_eq!(a.set_line(id, &[s[0], s[2], s[4]], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
        let passed: Vec<(u32, u32)> = a.passed_stations(id).chunks(2).map(|c| (c[0] as u32, c[1] as u32)).collect();
        assert_eq!(passed, vec![(0, s[1]), (1, s[3])], "{tracks} tracks");
        // and the other way round
        let back = a.new_line_id();
        assert_eq!(a.set_line(back, &[s[4], s[0]], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "B".into()), 0, "{}", a.issues());
        let passed: Vec<u32> = a.passed_stations(back).chunks(2).map(|c| c[1] as u32).collect();
        assert_eq!(passed, vec![s[3], s[2], s[1]]);
        // adding them all: one edit, one undo
        let all: Vec<u32> = s.iter().rev().copied().collect();
        assert_eq!(a.set_stops(back, &all), 0, "{}", a.issues());
        assert!(a.passed_stations(back).is_empty());
        assert_eq!(a.undo(), 0);
        assert_eq!(a.w.net.line_stops(back), &[s[4], s[0]]);
    }
}

/// T-733: a line's train length. 0 runs the most the shortest platform fits; a set length runs
/// as set, never past that cap; a platform shortened below it caps the trains while the
/// setting stays, so lengthening the platform again brings it back. Fleet cost follows.
#[test]
fn line_train_length_capped_by_platforms() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 4000.0, 0.0, 2);
    for (x, name) in [(500.0, "A"), (3500.0, "B")] {
        assert_eq!(a.add_station(2, track_at(&a, x, 0.0), x, 0.0, 200, name.into()), 0, "{}", a.issues());
    }
    let (sa, sb) = (stations(&a, "A")[0], stations(&a, "B")[0]);
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[sa, sb], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    let cars = |a: &TrackApi| a.w.svc.lines[id as usize].cars;
    assert_eq!(cars(&a), 10);
    assert_eq!(a.line_cars_cap(id), vec![10, sa]);
    let cost10 = a.line_train_cost(id);
    assert_eq!(a.set_line_cars(id, 4), 0, "{}", a.issues());
    assert_eq!(cars(&a), 4);
    let cost4 = a.line_train_cost(id);
    assert!(cost4 < cost10 * 0.5, "fleet cost {cost4} against {cost10}");
    assert_eq!(a.undo(), 0);
    assert_eq!(cars(&a), 10, "one undo step");
    assert_eq!(a.redo(), 0);
    // more than the platforms fit runs at the cap
    assert_eq!(a.set_line_cars(id, 30), 0);
    assert_eq!(cars(&a), 10);
    assert_eq!(a.set_line_cars(id, 6), 0);
    // a platform shortened below the setting caps it; the setting stays
    assert_eq!(a.set_station_props(sb, 80, "B".into()), 0, "{}", a.issues());
    assert_eq!(cars(&a), 4);
    assert_eq!(a.line_cars_cap(id), vec![4, sb]);
    assert_eq!(a.set_station_props(sb, 200, "B".into()), 0, "{}", a.issues());
    assert_eq!(cars(&a), 6);
    // saved with the line
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&a.save()));
    assert_eq!(b.w.svc.lines[id as usize].cars, 6);
}

/// T-733: the crossover tool and single track are gone from the game's Build tab, but saves
/// holding them still load and run: a line over single track and one over a hand-placed
/// crossover keep their routes and timetables.
#[test]
fn old_saves_with_crossovers_and_single_track_still_run() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    // single track with two stations
    route(&mut a, [0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 3000.0, 1);
    for (y, name) in [(500.0, "S1"), (2500.0, "S2")] {
        assert_eq!(a.add_station(2, track_at(&a, 0.0, y), 0.0, y, 160, name.into()), 0, "{}", a.issues());
    }
    let single = a.new_line_id();
    let ss = [stations(&a, "S1")[0], stations(&a, "S2")[0]];
    assert_eq!(a.set_line(single, &ss, 4.0, 2.0, 1.0, 30.0, 180.0, 0, 0, "Single".into()), 0, "{}", a.issues());
    // quad with stations, their automatic crossovers removed, and one placed by hand between
    route(&mut a, [0.0, 0.0, 2000.0, 0.0, 0.0], 6000.0, 0.0, 4);
    for (x, name) in [(2500.0, "Q1"), (5500.0, "Q2")] {
        assert_eq!(a.add_station(2, track_at(&a, x, 0.0), x, 0.0, 120, name.into()), 0, "{}", a.issues());
    }
    while let Some(&x) = a.crossover_edges().first() {
        assert_eq!(a.delete_edge(x), 0, "{}", a.issues());
    }
    assert_eq!(a.add_crossover(track_at(&a, 4000.0, 0.0), 4000.0, 0.0), 0, "{}", a.issues());
    let q1 = *stations(&a, "Q1").iter().min_by(|&&p, &&q| a.w.net.node_y[p as usize].abs().total_cmp(&a.w.net.node_y[q as usize].abs())).unwrap();
    let q2 = *stations(&a, "Q2").iter().max_by(|&&p, &&q| a.w.net.node_y[p as usize].abs().total_cmp(&a.w.net.node_y[q as usize].abs())).unwrap();
    let cross = a.new_line_id();
    assert_eq!(a.set_line(cross, &[q1, q2], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "Cross".into()), 0, "{}", a.issues());
    let bytes = a.save();
    let mut b = TrackApi::new(-73.985, 40.758, true);
    assert!(b.load(&bytes));
    assert_eq!(b.crossover_edges().len(), 2);
    assert!(live(&b).iter().any(|&e| b.w.net.edge_tracks[e as usize] == 1), "single track kept");
    assert!((b.total_cost() - a.total_cost()).abs() < 1e-6);
    let xs = b.crossover_edges();
    assert!(b.line_path(cross).iter().any(|p| xs.contains(&(p >> 1))), "still across the crossover");
    for id in [single, cross] {
        assert!(!b.w.net.line_broken[id as usize]);
        let s = b.line_summary(id);
        assert!(s[0] > 0.0 && s[1] > 0.0, "line {id} runs: {s:?}");
        assert_eq!(b.stop_departures(id, 0, 0).len(), 2);
    }
}
