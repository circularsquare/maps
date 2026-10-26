use super::*;
use crate::demand::network::{route_nodes, RealNet};
use crate::demand::{kernel, pack, synth};

/// A small pack (the synthetic city's inner 12 km, gravity solved and shipped) and two
/// crossing lines.
pub(super) fn small() -> (DemandApi, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut city = synth::synth_city(1).inner(12_000.0);
    let p = Params::default();
    let z = kernel::build_zones(&city, p.zone_m);
    let g = kernel::gravity(&z, &p, 100, 1e-5);
    city.gravity = Some(kernel::to_pack_gravity(&z, &g, &p));
    let (h, b) = pack::write(&city);
    let api = DemandApi::new(&h, &b).unwrap();
    // stations every 1 km along x and along y through the centre (shared at the origin)
    let mut xy = vec![];
    for i in -8..=8 {
        xy.extend([i as f32 * 1000.0, 0.0]);
    }
    let centre = 8u32;
    let mut line_b = vec![];
    for i in -8..=8i32 {
        if i == 0 {
            line_b.push(centre);
        } else {
            line_b.push((xy.len() / 2) as u32);
            xy.extend([0.0, i as f32 * 1000.0]);
        }
    }
    let line_a: Vec<u32> = (0..17).collect();
    let stops: Vec<u32> = line_a.iter().chain(&line_b).cloned().collect();
    // 90 s a hop in both directions at every level, 0 past the ends
    let mut times = vec![];
    for _ in 0..2 {
        for run in 0..2 {
            for _lev in 0..3 {
                for k in 0..17 {
                    let end = if run == 0 { k == 16 } else { k == 0 };
                    times.push(if end { 0.0 } else { 90.0 });
                }
            }
        }
    }
    let tph = vec![20.0, 10.0, 4.0, 12.0, 6.0, 0.0];
    let cars = vec![10.0, 8.0];
    (api, xy, vec![17, 17], stops, times, tph, cars)
}

/// T-842: a span of 1 everywhere changes nothing to the bit; line b with half its midday riders
/// in a gap loses riders, and with none due at midday nobody boards it then, while the morning
/// (its own span 1) is untouched. A new span forgets only the periods it moved.
#[test]
fn the_service_span_prices_stops_with_no_train_due() {
    let nr = 68;
    let run = |span: Option<Vec<f32>>| {
        let (mut api, xy, n, stops, times, tph, cars) = small();
        if let Some(s) = &span {
            api.set_span(s);
        }
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        assert!(api.solve(0) && api.solve(1));
        let b = |q: usize| -> f64 { api.board(q)[nr / 2..].iter().map(|&v| v as f64).sum() };
        (api.board(0), api.board(1), b(1), api.summary(1)[2])
    };
    let with_b = |phi: f32| (0..PERIODS * nr).map(|i| if i / nr == 1 && i % nr >= nr / 2 { phi } else { 1.0 }).collect::<Vec<f32>>();
    let base = run(None);
    let ones = run(Some(vec![1.0; PERIODS * nr]));
    assert!(same_bits(&base.0, &ones.0) && same_bits(&base.1, &ones.1));
    let half = run(Some(with_b(0.5)));
    let none = run(Some(with_b(0.0)));
    assert!(same_bits(&base.0, &half.0) && same_bits(&base.0, &none.0), "the morning keeps its riders");
    assert!(half.2 < 0.9 * base.2 && half.2 > 0.0, "b at midday: {} against {}", half.2, base.2);
    assert_eq!(none.2, 0.0);
    assert!(none.3 < half.3 && half.3 < base.3);
    // T-124: only the midday is forgotten when b's midday span moves
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    api.set_span(&with_b(0.5));
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert_eq!(api.same, (1 << PERIODS) - 1 - 2);
}

/// T-161: snapshot complexes carried to demand stations; a demand station in two complexes
/// joins them; a list of the wrong length means none.
#[test]
fn complexes_follow_merged_stations() {
    // snapshot stations 0..5 -> demand stations 0, 0, 1, 2, 3; complexes {0, 2}, {1, 3}, {4}
    let map = [0, 0, 1, 2, 3];
    assert_eq!(demand_groups(&map, 4, &[7, 9, 7, 9, 4]), vec![0, 0, 0, 3]);
    assert_eq!(demand_groups(&map, 4, &[7, 8, 7, 9, 4]), vec![0, 0, 2, 3]);
    assert!(demand_groups(&map, 4, &[1, 2]).is_empty());
    let (mut api, xy, n, stops, times, tph, cars) = small();
    // the two ends of line a, 16 km apart, as one complex
    let mut complex: Vec<u32> = (0..xy.len() as u32 / 2).collect();
    complex[16] = 0;
    api.set_complexes(&complex);
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert!(api.solve(0));
    let st = api.period(0).unwrap();
    let (a, b) = (api.station_map()[0], api.station_map()[16]);
    assert_eq!(st.net.st_group[a as usize], st.net.st_group[b as usize]);
    let g = &st.g;
    let walks = (0..g.adj_to.len()).filter(|&e| g.adj_seg[e] == u32::MAX && g.adj_to[e] < g.n_st as u32).count();
    api.set_complexes(&[]);
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert!(api.solve(0));
    let g = &api.period(0).unwrap().g;
    let plain = (0..g.adj_to.len()).filter(|&e| g.adj_seg[e] == u32::MAX && g.adj_to[e] < g.n_st as u32).count();
    // each end has one route node a run (the line's first or last stop), both runs: 4 walks
    assert_eq!(walks, plain + 4);
}

#[test]
fn shared_crowding_uses_complete_loads_without_overwriting_own_assignment() {
    let (mut own, xy, n, stops, times, tph, cars) = small();
    let (mut external, ..) = small();
    let (mut legacy, ..) = small();
    for api in [&mut own, &mut external, &mut legacy] {
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        assert!(api.solve(0));
    }
    let loads = own.seg(0);
    // segment loads, boardings and alightings (T-188), the population's own
    let own_all = [own.seg(0), own.board(0), own.alight(0)].concat();
    assert!(own.crowd_with_external(0, &own_all));
    assert!(legacy.crowd(0));
    assert_eq!(own.seg(0), legacy.seg(0), "own-only external input exactly preserves old solver");
    let shared: Vec<f32> = loads.iter().map(|v| v * 50.0).collect();
    assert!(external.crowd_with_external(0, &shared));
    assert!(external.summary(0)[2] < own.summary(0)[2], "other populations must reduce this population's rail choice when trains are full");
    assert!(external.seg(0).iter().sum::<f32>() < shared.iter().sum::<f32>(), "assignment remains this population, not the summed feedback vector");
    let before = external.seg(0);
    assert!(!external.crowd_with_external(0, &[f32::NAN]));
    let mut invalid = shared; invalid[0] = -1.0;
    assert!(!external.crowd_with_external(0, &invalid));
    assert_eq!(external.seg(0), before, "bad shared loads leave period intact");
    assert_eq!(external.summary(0)[0], 1.0);
}

/// T-119: strategies are searched only towards stations someone can alight at. A line out in the
/// country (no cell within the walk's bound of its stations) is not searched towards, and the
/// pass gives what a search towards every station gives, bit for bit.
#[test]
fn searching_only_the_egress_stations_changes_nothing() {
    let (mut api, mut xy, mut n, mut stops, mut times, mut tph, mut cars) = small();
    // a third line of two stations 60 km out, 4 minutes between them
    let far = (xy.len() / 2) as u32;
    xy.extend([60_000.0, 0.0, 62_000.0, 0.0]);
    n.push(2);
    stops.extend([far, far + 1]);
    for _ in 0..2 {
        times.extend([240.0, 0.0, 240.0, 0.0, 240.0, 0.0]);
    }
    // run 1 of a two-stop line: stop 1 to stop 0
    let at = times.len() - 6;
    times[at..].copy_from_slice(&[0.0, 240.0, 0.0, 240.0, 0.0, 240.0]);
    tph.extend([6.0, 6.0, 6.0]);
    cars.push(8.0);
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert!(api.want.iter().any(|&w| !w) && api.want.iter().any(|&w| w), "some stations searched, some not");
    assert!(!api.want[far as usize] && !api.want[far as usize + 1]);
    api.solve(0);
    api.crowd(0);
    let st = api.per[0].as_ref().unwrap();
    clear_skim_cache();
    let (rail, rail_walk, l, pairs, sub, _) = api.pass(&st.g, &st.p, 0, None);
    // the reference: every station searched, nothing cached
    let sz = api.sz.as_ref().unwrap();
    let sk = kernel::skims(&st.g, &st.p);
    let er = kernel::station_exp(&sk, &st.p);
    let mo = kernel::mode_choice(&api.z, &api.gr, sz, &er, &sk.walk, &sk.walk_max, st.g.n_st, &st.p, &api.nr);
    let s = st.g.n_st;
    let (w, h) = (TO_WORK[0], TO_HOME[0]);
    let (fw, fh) = (w / (w + h), h / (w + h));
    let mut od = mo.od.clone();
    for a in 0..s {
        for t in a + 1..s {
            let (x, y) = (od[a * s + t], od[t * s + a]);
            od[a * s + t] = fw * x + fh * y;
            od[t * s + a] = fw * y + fh * x;
        }
    }
    let want = kernel::assign(&st.g, &sk, &od);
    assert_eq!((rail, rail_walk, pairs), (mo.rail, mo.rail_walk, mo.pairs));
    assert_eq!(l.seg, want.seg);
    assert_eq!(l.board, want.board);
    assert_eq!(l.rn_board, want.rn_board);
    assert_eq!(l.rn_alight, want.rn_alight);
    assert_eq!(l.entry, want.entry);
    assert_eq!(sub.od, api.compact_od(&mo.od, s));
    assert!(rail > 0.0);
    clear_skim_cache();
}

/// T-119: when nothing is crowded a crowding round solves what the free-flow estimate did, so it
/// is skipped, and the skip changes nothing: the loads, the rail trips and the views' sums are
/// the free-flow estimate's to the last bit, round by round, and the round count still moves.
#[test]
fn uncrowded_rounds_repeat_the_estimate_exactly() {
    let (mut api, xy, n, stops, times, tph, _) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &[400.0, 400.0]).unwrap();
    api.solve(0);
    let (seg, rail, sums) = (api.seg(0), api.summary(0)[2], api.sub_sums());
    for r in 1..=3 {
        assert!(api.crowd(0));
        assert_eq!(api.summary(0)[0], r as f64);
        assert_eq!(api.seg(0), seg);
        assert_eq!(api.summary(0)[2], rail);
        assert_eq!(api.sub_sums(), sums);
    }
    // a skipped round is the pass it replaces: the same network crowded the legacy way gives the same
    let (mut crowded, xy, n, stops, times, tph, cars) = small();
    crowded.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    crowded.solve(0);
    let free = crowded.seg(0);
    crowded.crowd(0);
    assert_ne!(crowded.seg(0), free, "a crowded network moves in a round");
}

/// T-124: a change to one level's trains keeps the other levels' periods, and re-solving
/// the rest gives every figure a fresh solve of the new network gives, to the bit; a new
/// layout or fare keeps nothing.
#[test]
fn periods_an_edit_did_not_touch_are_kept_exactly() {
    let solve = |api: &mut DemandApi, mask: u32| {
        for q in (0..PERIODS).filter(|q| mask >> q & 1 == 1) {
            api.solve(q);
            for _ in 0..3 {
                assert!(api.crowd(q));
            }
        }
    };
    let figures = |api: &DemandApi| -> Vec<Vec<u64>> {
        let bits = |v: Vec<f32>| v.iter().map(|x| x.to_bits() as u64).collect::<Vec<_>>();
        let mut out = vec![bits(api.sub_sums())];
        for q in 0..PERIODS {
            let mut s = api.summary(q);
            s[4] = 0.0; // ms
            out.push(s.iter().map(|v| v.to_bits()).collect());
            out.extend([bits(api.seg(q)), bits(api.board(q)), bits(api.alight(q)), bits(api.load_of_crush(q))]);
        }
        out
    };
    let (mut api, xy, n, stops, times, tph, cars) = small();
    let km = |api: &DemandApi| vec![0.5f32; 2 * api.svc.as_ref().unwrap().stops.len()];
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let hop = km(&api);
    api.set_fares(2.0, 0.1, &hop).unwrap();
    solve(&mut api, 0b11111);
    let mut high = tph.clone();
    high[0] *= 0.5; // line a's high level
    api.set_network(&xy, &n, &stops, &times, &high, &cars).unwrap();
    api.set_fares(2.0, 0.1, &hop).unwrap();
    let kept = api.solved_mask();
    assert_eq!(kept, 0b11010, "the medium and low periods are kept");
    solve(&mut api, 0b11111 & !kept);
    let (mut fresh, ..) = small();
    fresh.set_network(&xy, &n, &stops, &times, &high, &cars).unwrap();
    fresh.set_fares(2.0, 0.1, &hop).unwrap();
    solve(&mut fresh, 0b11111);
    assert_eq!(figures(&api), figures(&fresh));
    // the same network again keeps everything; another fare, or cars, nothing
    api.set_network(&xy, &n, &stops, &times, &high, &cars).unwrap();
    api.set_fares(2.0, 0.1, &hop).unwrap();
    assert_eq!(api.solved_mask(), 0b11111);
    api.set_fares(2.5, 0.1, &hop).unwrap();
    assert_eq!(api.solved_mask(), 0);
    solve(&mut api, 0b11111);
    let mut more = cars.clone();
    more[1] += 1.0;
    api.set_network(&xy, &n, &stops, &times, &high, &more).unwrap();
    api.set_fares(2.5, 0.1, &hop).unwrap();
    assert_eq!(api.solved_mask(), 0);
}

/// T-188: a line cut to too few trains leaves riders behind, and with their extra wait priced
/// into the crowding rounds it loses riders (here to the car: the two lines cross, so they
/// are not alternatives, and line a loses the riders who changed from b); the extra wait is
/// a single line's, at its own stops; with room on every train nothing changes.
#[test]
fn a_line_with_too_few_trains_loses_riders_to_the_wait() {
    let run = |left_behind: bool, tph_b: f32, cars_b: f32| {
        let (mut api, xy, n, stops, times, mut tph, mut cars) = small();
        api.params_mut().left_behind = left_behind;
        // rail and the car judged on time and money alone
        api.params_mut().asc_rail = 0.0;
        tph[3] = tph_b;
        cars[1] = cars_b;
        // line a with room to spare (with rides to stations, T-371, 20 cars leave riders
        // behind mid-peak on this city)
        cars[0] = 40.0;
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        api.solve(0);
        for _ in 0..3 {
            assert!(api.crowd(0));
        }
        let st = api.period(0).unwrap();
        let b: f64 = api.board(0).iter().zip(&st.g.rn_line).filter(|p| *p.1 == 1).map(|p| *p.0 as f64).sum();
        let a: f64 = api.board(0).iter().zip(&st.g.rn_line).filter(|p| *p.1 == 0).map(|p| *p.0 as f64).sum();
        let left_a = st.g.rn_left.iter().zip(&st.g.rn_line).filter(|p| *p.1 == 0).map(|p| *p.0).fold(0.0f32, f32::max);
        let left_b = st.g.rn_left.iter().zip(&st.g.rn_line).filter(|p| *p.1 == 1).map(|p| *p.0).fold(0.0f32, f32::max);
        (b, a, api.summary(0)[2], left_a, left_b)
    };
    // line b with room: 12 trains an hour of 64 cars; nobody is left behind and the waits
    // change nothing
    let (b_full_off, _, rail_full_off, _, _) = run(false, 12.0, 64.0);
    let (b_full, _, rail_full, _, lb_full) = run(true, 12.0, 64.0);
    assert_eq!(lb_full, 0.0);
    assert_eq!((b_full, rail_full), (b_full_off, rail_full_off));
    // cut to 6 trains of 24 cars (under a fifth of the room): crowding alone takes some
    // riders off; the wait for full trains takes visibly more. (Cut harder, T-710's steep
    // penalty past 70% of crush empties line b in every crowded round by itself, and what is
    // left is the free-flow pass's quarter of the average, which no wait can move.)
    let (b_off, a_off, rail_off, _, lb_off) = run(false, 6.0, 24.0);
    let (b_on, a_on, rail_on, la_on, lb_on) = run(true, 6.0, 24.0);
    println!("line b full {b_full:.0}; cut, crowding only {b_off:.0}, with the wait {b_on:.0} riders (worst extra wait {lb_on:.1} min); line a {a_off:.0} -> {a_on:.0}; rail {rail_off:.0} -> {rail_on:.0}");
    assert_eq!(lb_off, 0.0, "no waits priced with the switch off");
    assert!(lb_on > 5.0, "line b's riders wait {lb_on} minutes more at worst");
    assert_eq!(la_on, 0.0, "line a has room: {la_on}");
    assert!(b_off < b_full, "crowding alone: {b_off:.0} of {b_full:.0}");
    // 7.7% fewer here
    assert!(b_on < 0.97 * b_off, "line b keeps {b_on:.0} of {b_off:.0} riders");
    // line a loses the riders who changed from b; the rest drive
    assert!(rail_on < rail_off, "and some drive: rail {rail_on:.0} against {rail_off:.0}");
}

/// T-1070, T-1255: after the rounds, riders are held to those the trains take within
/// `WAIT_FOR_ROOM_S`, so no segment carries more than every train full all period; the riders
/// turned away leave the rail trips, the entries and the loads alike; a network with room
/// everywhere is left exactly as it was; and riders who would come with room (the free-flow
/// pass's) crowd the platforms, so fewer of the rounds' riders fit.
#[test]
fn riders_are_held_to_the_trains_room() {
    let run = |tph_b: f32, cars_b: f32, come: f32| {
        let (mut api, xy, n, stops, times, mut tph, mut cars) = small();
        tph[3] = tph_b;
        cars[1] = cars_b;
        cars[0] = 40.0;
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        api.solve(0);
        for _ in 0..3 {
            assert!(api.crowd(0));
        }
        let before = (api.summary(0)[2], api.board(0).iter().map(|&v| v as f64).sum::<f64>(), api.entries(0).iter().map(|&v| v as f64).sum::<f64>());
        let mut total = [api.seg(0), api.board(0), api.alight(0)].concat();
        if come > 1.0 {
            total.extend(api.board(0).iter().map(|&v| v * come));
        }
        assert!(api.cap_with_external(0, &total));
        let after = (api.summary(0)[2], api.board(0).iter().map(|&v| v as f64).sum::<f64>(), api.entries(0).iter().map(|&v| v as f64).sum::<f64>());
        let worst = api.load_of_crush(0).iter().fold(0f32, |a, &b| a.max(b));
        (before, after, worst)
    };
    let (b, a, worst) = run(12.0, 64.0, 1.0);
    assert!(worst <= 1.0, "{worst}");
    assert_eq!(b, a);
    let (b, a, worst) = run(6.0, 24.0, 1.0);
    println!("rail {:.0} -> {:.0}, boardings {:.0} -> {:.0}, entries {:.0} -> {:.0}; worst load {worst:.3} of crush", b.0, a.0, b.1, a.1, b.2, a.2);
    // the busiest hour at crush is `peak_hour_factor` times the mean hour's; trains full all period
    // carry that much
    assert!(worst <= PEAK_HOUR_FACTOR[0] + 1e-3, "{worst}");
    assert!(a.1 < b.1 && a.0 < b.0, "{b:?} {a:?}");
    // rail trips and entries fall by the same share
    assert!((a.0 / b.0 - a.2 / b.2).abs() < 1e-4, "{b:?} {a:?}");
    let (_, a3, _) = run(6.0, 24.0, 3.0);
    println!("with three times the riders coming: boardings {:.0}", a3.1);
    assert!(a3.1 < a.1, "{a3:?} {a:?}");
}

/// T-188: `kernel::left_behind` on one line: riders within the trains' room wait nothing
/// extra; past it, the wait grows with the overload, falls on the stops where riders board
/// into full trains, and is capped.
#[test]
fn left_behind_waits_grow_with_the_overload() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    api.solve(0);
    let st = api.period(0).unwrap();
    let (g, net, p) = (&st.g, &st.net, &st.p);
    let nr = g.rn_line.len();
    // line a run 0 (route nodes 0..17): riders board at the first 4 stops, leave at the last
    let crush_hour = 20.0 * 10.0 * CRUSH_PER_CAR; // 32,000 an hour
    let waits = |per_hour: f32| {
        let mut board = vec![0.0f32; nr];
        let mut alight = vec![0.0f32; nr];
        for k in 0..4 {
            board[k] = per_hour / 4.0 * p.period_hours;
        }
        alight[16] = per_hour * p.period_hours;
        kernel::left_behind(g, net, p, &board, &alight)
    };
    let (w, s) = waits(0.5 * crush_hour);
    assert!(w.iter().all(|&v| v == 0.0) && s.left == 0.0, "half full: {s:?}");
    // the busiest hour is 1.4 times the mean: at 0.8 of crush on average the middle is over
    let (w08, s08) = waits(0.8 * crush_hour);
    let (w12, s12) = waits(1.2 * crush_hour);
    assert!(s08.left > 0.0 && s12.left > s08.left, "{s08:?} {s12:?}");
    assert!(w12[3] > w08[3] && w08[3] > 0.0, "{} {}", w08[3], w12[3]);
    // the first stops find empty trains; the fourth boards into trains three stops full
    assert!(w12[0] < w12[3], "{:?}", &w12[..4]);
    assert!(w12[4..].iter().all(|&v| v == 0.0));
    let (w5, _) = waits(5.0 * crush_hour);
    assert!(w5[3] == kernel::LEFT_WAIT_CAP_MIN, "{}", w5[3]);
    // the rider-minutes divided by the riders are the boardings' average (below the cap)
    assert!(w08.iter().all(|&v| v < kernel::LEFT_WAIT_CAP_MIN));
    let tot: f64 = (0..4).map(|k| w08[k] as f64 * (0.8 * crush_hour / 4.0 * p.period_hours) as f64).sum();
    assert!((tot - s08.rider_min).abs() < 1e-3 * s08.rider_min, "{tot} {}", s08.rider_min);
}

#[test]
fn fares_change_ridership_and_preserve_legacy_price() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let mut km = vec![];
    for &count in &n { for _ in 0..2 { for k in 0..count { km.push(if k + 1 == count { 0.0 } else { 1.0 }); } } }
    api.solve(0);
    let legacy = api.summary(0)[2];
    api.set_fares(2.0, 0.0, &km).unwrap();
    assert_eq!(api.solved_mask(), 0);
    api.solve(0);
    assert_eq!(legacy, api.summary(0)[2], "legacy $2 flat fare is exactly 6 minutes");
    api.set_fares(0.0, 0.0, &km).unwrap(); api.solve(0);
    let free = api.summary(0)[2];
    api.set_fares(1.5, 0.1, &km).unwrap(); api.solve(0);
    let default = api.summary(0)[2];
    api.set_fares(20.0, 0.0, &km).unwrap(); api.solve(0);
    let expensive = api.summary(0)[2];
    api.set_fares(0.0, 10.0, &km).unwrap(); api.solve(0);
    let distance = api.summary(0)[2];
    assert!(free > default && default > expensive && free > distance,
        "free {free}, default {default}, expensive {expensive}, distance {distance}");
    // Crowding stretches riding time, never the fare; transfers do not add another base.
    api.crowd(0);
    let st = api.per[0].as_ref().unwrap();
    for e in 0..st.g.adj_to.len() {
        let seg = st.g.adj_seg[e];
        if seg != u32::MAX { let r = seg as usize - st.g.n_st; assert_eq!(st.g.rn_fare[r], km[r] * 30.0); }
    }
    assert!(api.set_fares(f32::NAN, 0.0, &km).is_err());
    assert!(api.set_fares(0.0, -1.0, &km).is_err());
    assert!(api.set_fares(0.0, 0.0, &[]).is_err());
    println!("fare riders: legacy={legacy:.0} free={free:.0} default={default:.0} expensive={expensive:.0} distance={distance:.0}");
}

/// T-1300: the rail trips' fare sensitivity (`summary`'s last figure) is what a dearer base fare
/// does: the base fare moves no route, only the mode choice.
#[test]
fn the_fare_sensitivity_matches_a_dearer_base_fare() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let mut km = vec![];
    for &count in &n { for _ in 0..2 { for k in 0..count { km.push(if k + 1 == count { 0.0 } else { 1.0 }); } } }
    api.set_fares(1.5, 0.1, &km).unwrap();
    api.solve(0);
    let (r0, d) = (api.summary(0)[2], api.summary(0)[10]);
    api.set_fares(1.7, 0.1, &km).unwrap();
    api.solve(0);
    let r1 = api.summary(0)[2];
    let want = (r1 - r0) / (0.2 * api.base.fare_w as f64);
    assert!(d < 0. && (d - want).abs() < 0.03 * want.abs(), "sensitivity {d}, a dearer base fare gives {want}");
}

/// T-1143: a fare curve on the base and rate's own line changes nothing; one that steepens past
/// 3 km loses riders, and its extra is counted where they board, beyond the base and rate's fare.
#[test]
fn a_bent_fare_curve_loses_long_riders_and_counts_its_extra() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let mut km = vec![];
    for &count in &n { for _ in 0..2 { for k in 0..count { km.push(if k + 1 == count { 0.0 } else { 1.0 }); } } }
    api.set_fares(1.5, 0.1, &km).unwrap();
    api.solve(0);
    let line = api.summary(0);
    api.set_fare_curve(&[0.0, 1.5, 20.0, 3.5, 100.0, 11.5]).unwrap();
    api.solve(0);
    assert_eq!(api.summary(0)[2].to_bits(), line[2].to_bits());
    api.set_fare_curve(&[0.0, 1.5, 3.0, 1.8, 100.0, 1.8 + 97.0]).unwrap();
    assert_eq!(api.solved_mask(), 0);
    api.solve(0);
    let bent = api.summary(0)[2];
    let extra: f32 = api.fare_extra(0).iter().sum();
    assert!(bent < line[2] && extra > 0.0, "rail {bent} against {}, extra {extra}", line[2]);
    assert!(api.set_fare_curve(&[0.0, 2.0, 3.0, 1.0]).is_err());
    assert!(api.set_fare_curve(&[3.0, 1.0, 1.0, 2.0]).is_err());
    api.set_fare_curve(&[]).unwrap();
    api.solve(0);
    assert_eq!(api.summary(0)[2].to_bits(), line[2].to_bits());
    assert!(api.fare_extra(0).iter().all(|&x| x == 0.0));
}

#[test]
fn fare_distances_follow_each_runs_track_and_base_is_once() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let mut km = vec![0.0; 2 * stops.len()];
    // First line: 2 km forward segments and 3 km reverse segments.
    km[..16].fill(2.0);
    km[17..33].fill(3.0);
    api.set_fares(0.0, 0.0, &km).unwrap();
    api.solve(0);
    let st = api.per[0].as_ref().unwrap();
    let free = skims(&st.g, &st.p);
    api.set_fares(5.0, 0.5, &km).unwrap();
    api.solve(0);
    let st = api.per[0].as_ref().unwrap();
    let priced = skims(&st.g, &st.p);
    // Stations 0 and 1 are joined by the first forward/reverse segment respectively.
    assert!((priced.r[1] - free.r[1] - 3.0).abs() < 1e-5);
    assert!((priced.r[priced.s] - free.r[free.s] - 4.5).abs() < 1e-5);
    assert_eq!(st.p.fare_min, 15.0);
    // A two-line trip through the centre also sees no extra $5 in its route graph.
    let other = stops[17] as usize;
    assert!(priced.r[other].is_finite());
    let sk_before = priced.r.clone();
    api.set_fares(50.0, 0.5, &km).unwrap();
    api.solve(0);
    let st = api.per[0].as_ref().unwrap();
    assert_eq!(skims(&st.g, &st.p).r, sk_before);
    // Explicit multiplier check: fares stay unscaled even at double riding disutility.
    let mut graph = build_graph(&st.net, &st.p);
    graph.rn_fare = km.iter().map(|v| v * 1.5).collect();
    let factor = vec![2.0; graph.rn_ride.len()];
    set_crowding(&mut graph, &factor);
    for e in 0..graph.adj_to.len() {
        if graph.adj_seg[e] != u32::MAX {
            let r = graph.adj_seg[e] as usize - graph.n_st;
            assert_eq!(graph.adj_cost[e], 2.0 * graph.rn_ride[r] + km[r] * 1.5);
        }
    }
}

#[test]
/// T-558: a short commute walks with no rail; an overwhelmingly attractive train takes its
/// walkers (walking is in the choice), and the views count what it leaves.
fn short_commutes_walk_unless_the_train_is_better() {
    let city = pack::City { name: "short".into(), h3: vec![1, 2],
        x: vec![0.0, 300.0], y: vec![0.0, 0.0],
        pop: vec![100.0, 0.0], jobs: vec![0.0, 100.0], region: vec![], gravity: None };
    let (h, b) = pack::write(&city);
    let mut api = DemandApi::new(&h, &b).unwrap();
    let baseline = api.info()[3];
    assert!(baseline > 0.8, "same-zone trips overwhelmingly walk");
    api.params_mut().asc_rail = 20.0;
    let times = [10.0,0.0,10.0,0.0,10.0,0.0,0.0,10.0,0.0,10.0,0.0,10.0];
    api.set_network(&[0.0,0.0,300.0,0.0], &[2], &[0,1], &times, &[60.0;3], &[4.0]).unwrap();
    api.set_fares(0.0, 0.0, &[0.3,0.0,0.0,0.3]).unwrap();
    for q in 0..5 {
        api.solve(q);
        api.crowd(q);
        for v in [api.summary(q)] {
            let (rail, walk) = (v[2] / v[1], v[3] / v[1]);
            assert!(rail > 0.9 && walk > -1e-6 && walk < 0.1 && rail + walk <= 1.0 + 1e-5, "rail {rail} walk {walk}");
        }
    }
    let v = api.summary(1);
    let sums = api.flow_rail(1, &[1]);
    let flows = api.flows(1, &[1], &sums);
    let tot = (flows[0] + flows[1] + flows[2]) as f64;
    assert!((flows[1] as f64 / tot - v[3] / v[1]).abs() < 1e-3, "{} against {}", flows[1] as f64 / tot, v[3] / v[1]);
}

#[test]
fn periods_solve_and_add_up() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let info = api.info();
    assert!(info[3] > 0.0 && info[3] < 0.5, "walk share with no rail {}", info[3]);
    let mut day_trips = 0.0;
    for q in 0..PERIODS {
        assert!(api.solve(q));
        let s0 = api.summary(q);
        assert!(api.crowd(q));
        let s = api.summary(q);
        assert_eq!(s[0], 1.0);
        let (trips, rail, walk) = (s[1], s[2], s[3]);
        day_trips += trips;
        assert!(rail > 0.0 && rail < trips && walk > 0.0 && rail + walk < trips, "period {q}: {s:?}");
        assert!(s0[2] >= rail * 0.5);
        // every rail trip boards once and alights once (plus transfers, on both lines)
        let boards: f64 = api.board(q).iter().map(|&v| v as f64).sum();
        let alights: f64 = api.alight(q).iter().map(|&v| v as f64).sum();
        assert!(boards >= rail * 0.999 && (boards / alights - 1.0).abs() < 1e-3, "{boards} {alights} {rail}");
        assert_eq!(api.seg(q).len(), 68);
        // line B has no trains at night: nobody on it
        if q == 4 {
            assert!(api.board(q)[34..].iter().all(|&v| v == 0.0));
        }
    }
    assert!((day_trips / (2.0 * info[2]) - 1.0).abs() < 1e-6);
    // the trips home in the evening peak load the opposite direction to the morning's
    let lc = api.load_of_crush(0);
    assert!(lc.iter().any(|&v| v > 0.0));
}

#[test]
fn views_with_no_network() {
    // T-099: no running line means no rail, not no answer
    let (mut api, ..) = small();
    api.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
    let mut walk = 0.0;
    for q in 0..PERIODS {
        api.solve(q);
        walk += api.summary(q)[3];
    }
    let cm = api.cell_modes(&api.sub_sums());
    let nc = api.cell_xy().len() / 2;
    assert_eq!(cm.len(), 6 * nc);
    let tot = |b: usize| cm[b * nc..(b + 1) * nc].iter().map(|&v| v as f64).sum::<f64>();
    assert_eq!(tot(0), 0.0);
    assert!((tot(1) / (walk / 2.0) - 1.0).abs() < 2e-3, "walk {} vs {}", tot(1), walk / 2.0);
    let cell = (0..nc).find(|&c| cm[c] + cm[nc + c] + cm[2 * nc + c] > 0.0).unwrap() as u32;
    let rail = api.flow_rail(0, &[cell]);
    assert!(rail.is_empty());
    assert!(!api.flows(0, &[cell], &rail).is_empty());
}

/// The demand views (T-078): per-cell modes add up to the day's rail and walk, a station's
/// riders add up the same by cell and by zone at the far end, and stations far apart differ.
#[test]
fn views_add_up() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    let (mut rail, mut walk) = (0.0, 0.0);
    for q in 0..PERIODS {
        api.solve(q);
        api.crowd(q);
        let s = api.summary(q);
        rail += s[2];
        walk += s[3];
    }
    assert_eq!(api.solved_mask(), (1 << PERIODS) - 1);
    let cm = api.cell_modes(&api.sub_sums());
    let nc = api.cell_xy().len() / 2;
    assert_eq!(cm.len(), 6 * nc);
    let tot = |b: usize| cm[b * nc..(b + 1) * nc].iter().map(|&v| v as f64).sum::<f64>();
    let commuters = api.info()[2];
    for (end, b) in [("home", 0), ("work", 3)] {
        assert!((tot(b) / (rail / 2.0) - 1.0).abs() < 1e-3, "{end} rail {} vs {}", tot(b), rail / 2.0);
        assert!((tot(b + 1) / (walk / 2.0) - 1.0).abs() < 2e-3, "{end} walk {} vs {}", tot(b + 1), walk / 2.0);
        assert!(((tot(b) + tot(b + 1) + tot(b + 2)) / commuters - 1.0).abs() < 2e-3, "{end} all");
    }
    assert!(api.cell_modes(&[1.0]).is_empty());
    let nz = api.zone_xy().len() / 2;
    let riders = |st: u32| {
        let r = api.station_riders(st);
        let (nh, nw) = (r[0] as usize, r[1] as usize);
        let sum = |a: usize, k: usize| r[a..a + k].iter().map(|&v| v as f64).sum::<f64>();
        let (home, work) = (sum(4 + nh, nh), sum(4 + 2 * nh + nw, nw));
        let o = 4 + 2 * nh + 2 * nw;
        assert_eq!(r.len(), o + 2 * nz);
        assert!((sum(o, nz) / home - 1.0).abs() < 1e-3 && (sum(o + nz, nz) / work - 1.0).abs() < 1e-3, "station {st}");
        // T-374: the walked and rode riders are the home riders, split
        assert!(((r[2] + r[3]) as f64 / home - 1.0).abs() < 1e-3, "station {st}: walked {} rode {} of {home}", r[2], r[3]);
        (home, work, r[4] as usize)
    };
    // the line's end and the shared centre station
    let (h_end, _, c_end) = riders(0);
    let (h_mid, w_mid, _) = riders(8);
    assert!(h_end > 0.0 && h_mid > 0.0 && w_mid > 0.0);
    let (x, y) = (api.cell_xy()[2 * c_end], api.cell_xy()[2 * c_end + 1]);
    // home cells ride from up to `ride_max_m` from their group's centre (T-371)
    assert!((x - xy[0]).hypot(y - xy[1]) < api.base.ride_max_m + 9000.0, "a catchment cell beyond the ride's bound");
    assert!(api.station_riders(9999).is_empty());
}

/// Flows (T-098): a selection's totals match `cell_modes` over its cells, at both ends; the
/// far cells plus the cut add up to them; the cut is small.
#[test]
fn flows_add_up() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    for q in 0..PERIODS {
        api.solve(q);
        api.crowd(q);
    }
    let cm = api.cell_modes(&api.sub_sums());
    let cxy = api.cell_xy();
    let nc = cxy.len() / 2;
    let near = |x: f32, y: f32, r: f32| -> Vec<u32> { (0..nc as u32).filter(|&c| (cxy[2 * c as usize] - x).hypot(cxy[2 * c as usize + 1] - y) < r).collect() };
    // around the line's end, the shared centre (one cell, and a bubble), and far from both lines
    let one = (0..nc as u32).min_by(|&a, &b| cxy[2 * a as usize].hypot(cxy[2 * a as usize + 1]).partial_cmp(&cxy[2 * b as usize].hypot(cxy[2 * b as usize + 1])).unwrap()).unwrap();
    let sels = [near(xy[0], xy[1], 1500.0), vec![one, one], near(0.0, 0.0, 2000.0), near(-6500.0, 6500.0, 1000.0)];
    for (si, sel) in sels.iter().enumerate() {
        assert!(!sel.is_empty());
        for end in 0..2u32 {
            let f = api.flows(end, sel, &api.flow_rail(end, sel));
            let b = 3 * end as usize;
            // a cell named twice counts once
            let mut uniq = sel.clone();
            uniq.dedup();
            let want: Vec<f64> = (0..3).map(|m| uniq.iter().map(|&c| cm[(b + m) * nc + c as usize] as f64).sum()).collect();
            let total: f64 = want.iter().sum();
            let k = f[6] as usize;
            for m in 0..3 {
                let got = f[m] as f64;
                assert!((got - want[m]).abs() <= 1e-3 * total.max(1.0), "selection {si} end {end} mode {m}: {got} vs cell_modes {}", want[m]);
                let far: f64 = f[7 + (m + 1) * k..7 + (m + 2) * k].iter().map(|&v| v as f64).sum();
                assert!((far + f[3 + m] as f64 - got).abs() <= 1e-3 * total.max(1.0), "selection {si} end {end} mode {m}: far {far} + cut {} vs {got}", f[3 + m]);
                assert!(f[3 + m] as f64 >= -2e-3 * total.max(1.0), "selection {si} end {end} mode {m}: cut {}", f[3 + m]);
            }
            let cut: f64 = (3..6).map(|m| f[m] as f64).sum();
            assert!(cut <= 0.03 * total, "selection {si} end {end}: cut {cut} of {total}");
            if si == 0 {
                assert!(f[0] > 0.0, "the line's end has rail");
            }
            // far from the lines: homes ride to them (T-371), jobs have no walk from them
            if si == 3 && end == 0 {
                assert!(f[0] > 0.0, "homes far from the lines ride");
            }
            if si == 3 && end == 1 {
                assert_eq!(f[0], 0.0, "no rail to jobs far from the lines");
            }
        }
    }
    assert!(api.flows(0, &sels[0], &[1.0]).is_empty());
}

/// The real New York pack with T-005's hand-made network: prints the day's split and the
/// per-period timings. `cargo test --release --features demand nyc_day -- --ignored --nocapture`
#[test]
#[ignore]
fn nyc_day() {
    let (h, b) = pack::load_raw(concat!(env!("CARGO_MANIFEST_DIR"), "/../data/packs/nyc.json")).unwrap();
    let t0 = now_ms();
    let mut api = DemandApi::new(&h, &b).unwrap();
    // TW_LEFT_OFF=1: without the waits for full trains (T-188), to compare
    api.params_mut().left_behind = std::env::var("TW_LEFT_OFF").is_err();
    let info = api.info();
    println!("open {:.0} ms: {} cells, {} zones, {:.0} commutes a day, walk with no rail {:.2}%", now_ms() - t0, info[0], info[1], info[2], 100.0 * info[3]);
    let net = crate::demand::network::synth_network();
    let xy: Vec<f32> = (0..net.n_stations()).flat_map(|s| [net.st_x[s], net.st_y[s]]).collect();
    let n: Vec<u32> = (0..net.n_lines()).map(|l| net.stops(l).len() as u32).collect();
    let mut times = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        for run in 0..2 {
            for _ in 0..3 {
                for k in a..b {
                    times.push(60.0 * if run == 0 { net.hop_min[k] } else if k > a { net.hop_min[k - 1] } else { 0.0 });
                }
            }
        }
    }
    let tph: Vec<f32> = net.line_headway_min.iter().flat_map(|h| [60.0 / h[0], 60.0 / h[1], 60.0 / h[4]]).collect();
    // cars from the crush load (seats per car differ by kind in network.rs; here one car fits all)
    let cars: Vec<f32> = net.line_crush.iter().map(|c| (c / CRUSH_PER_CAR).round()).collect();
    let ms = api.set_network(&xy, &n, &net.line_stops, &times, &tph, &cars).unwrap();
    println!("set_network {ms:.0} ms ({} stations, {} lines)", net.n_stations(), net.n_lines());
    let (mut trips, mut rail, mut walk) = (0.0, 0.0, 0.0);
    for q in 0..PERIODS {
        api.solve(q);
        let s0 = api.summary(q);
        api.crowd(q);
        let s = api.summary(q);
        println!(
            "period {q}: free flow {:.0} ms, crowding round {:.0} ms; {:.0} pairs; trips {:.0}, rail {:.0} -> {:.0} ({:.1}%), walk {:.0}; worst {:.0}% of crush",
            s0[4], s[4], s[5], s[1], s0[2], s[2], 100.0 * s[2] / s[1], s[3], 100.0 * s[7]
        );
        trips += s[1];
        rail += s[2];
        walk += s[3];
    }
    println!("day: {:.0} trips, train {:.1}%, walk {:.1}%, drive {:.1}%", trips, 100.0 * rail / trips, 100.0 * walk / trips, 100.0 * (1.0 - (rail + walk) / trips));
}

/// T-119: the real combined Northeast network (nyc_real_track.json plus one Philadelphia line)
/// solved as the app's workers do (both city populations and the regional market, five periods,
/// free flow and three crowding rounds), with timings per stage. With `TW_DUMP=<file>` the
/// summed results are written; with `TW_COMPARE=<file>` they are compared to such a file.
/// `cargo test --release --features demand northeast_profile -- --ignored --nocapture`
#[test]
#[ignore]
fn northeast_profile() {
    use crate::demand::intercity::IntercityApi;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let nyc = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let phl = pack::load_raw(&format!("{root}/data/packs/phl.json")).unwrap();
    let ic_header = std::fs::read_to_string(format!("{root}/data/packs/intercity.json")).unwrap();
    let t0 = now_ms();
    let mut pops = [DemandApi::new(&nyc.0, &nyc.1).unwrap(), DemandApi::new(&phl.0, &phl.1).unwrap()];
    let mut ic = IntercityApi::new(&ic_header).unwrap();
    println!("open {:.0} ms", now_ms() - t0);
    let (RealNet { mut st_xy, mut line_n, mut stops, mut times, mut tph, mut cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    // one Philadelphia line: two stations 4 km apart under City Hall's world position
    const R: f64 = 6371008.8;
    let rad = std::f64::consts::PI / 180.0;
    let wx = R * (-75.1636 + 73.985) * rad * (40.758 * rad).cos();
    let wy = R * (39.9526 - 40.758) * rad;
    let a = (st_xy.len() / 2) as u32;
    st_xy.extend([wx as f32, (wy - 2000.0) as f32, wx as f32, (wy + 2000.0) as f32]);
    line_n.push(2);
    stops.extend([a, a + 1]);
    for run in 0..2 {
        for _ in 0..3 {
            times.extend(if run == 0 { [600.0, 0.0] } else { [0.0, 600.0] });
        }
    }
    // `times` is laid out per line as 6 blocks of n values; the NYC lines were pushed in that order
    tph.extend([4.0, 4.0, 2.0]);
    cars.push(8.0);
    let ns = st_xy.len() / 2;
    println!("{ns} stations, {} lines, {} stops", line_n.len(), stops.len());
    let hop_km = route_nodes(&st_xy, &line_n, &stops).2;
    // the game's start fare is 0.1 a km; TW_PERKM=0 shows the changes that must be exact
    let per_km: f32 = std::env::var("TW_PERKM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1);
    let origin = [(-73.985f64, 40.758f64), (-75.1636, 39.9526)];
    for (k, a) in pops.iter_mut().enumerate() {
        let (lon0, lat0) = origin[k];
        let mut xy = st_xy.clone();
        for i in 0..ns {
            let lon = -73.985 + xy[2 * i] as f64 / (R * (40.758 * rad).cos() * rad);
            let lat = 40.758 + xy[2 * i + 1] as f64 / (R * rad);
            xy[2 * i] = (R * (lon - lon0) * rad * (lat0 * rad).cos()) as f32;
            xy[2 * i + 1] = (R * (lat - lat0) * rad) as f32;
        }
        let ms = a.set_network(&xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, per_km, &hop_km).unwrap();
        println!("pop {k}: set_network {ms:.0} ms, {} subzones", a.n_subzones());
    }
    let ms = ic.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    ic.set_fares(2.9, per_km, &hop_km).unwrap();
    println!("intercity set_network {ms:.0} ms");
    // stage breakdown of the big population, period 0
    {
        let a = &mut pops[0];
        a.solve(0);
        let st = a.per[0].as_ref().unwrap();
        let t = now_ms();
        let g = kernel::build_graph(&st.net, &st.p);
        let t_graph = now_ms() - t;
        let t = now_ms();
        let sk = kernel::skims(&g, &st.p);
        let t_skims = now_ms() - t;
        let t = now_ms();
        let er = kernel::station_exp(&sk, &st.p);
        let t_er = now_ms() - t;
        let t = now_ms();
        let mo = kernel::mode_choice(&a.z, &a.gr, a.sz.as_ref().unwrap(), &er, &sk.walk, &sk.walk_max, g.n_st, &st.p, &a.nr);
        let t_mode = now_ms() - t;
        let t = now_ms();
        let _l = kernel::assign(&g, &sk, &mo.od);
        let t_assign = now_ms() - t;
        println!("NYC period 0 breakdown: graph {t_graph:.0}, skims {t_skims:.0}, station_exp {t_er:.0}, mode_choice {t_mode:.0} ({} pairs), assign {t_assign:.0} ms; S={} N={}", mo.pairs, g.n_st, g.n);
    }
    for a in pops.iter_mut() {
        a.per.iter_mut().for_each(|p| *p = None);
    }
    let mut result = serde_json::Map::new();
    crate::demand::alloc::reset_peak();
    let live0 = crate::demand::alloc::current();
    let tstart = now_ms();
    let mut stage_ms = vec![[0.0f64; 3]; 4];
    for round in 0..4 {
        let tr = now_ms();
        let mut per_period: Vec<f64> = vec![];
        for q in 0..PERIODS {
            let sum = |pops: &[DemandApi; 2], ic: &IntercityApi| -> Vec<f32> {
                let mut out = pops[0].seg(q);
                for (o, v) in out.iter_mut().zip(pops[1].seg(q)) {
                    *o += v;
                }
                for (o, v) in out.iter_mut().zip(ic.seg(q)) {
                    *o += v;
                }
                out
            };
            if round == 0 {
                for (k, a) in pops.iter_mut().enumerate() {
                    let t = now_ms();
                    a.solve(q);
                    stage_ms[round][k] += now_ms() - t;
                }
                let t = now_ms();
                ic.solve(q);
                stage_ms[round][2] += now_ms() - t;
            } else {
                let total = sum(&pops, &ic);
                for (k, a) in pops.iter_mut().enumerate() {
                    let t = now_ms();
                    assert!(a.crowd_with_external(q, &total));
                    stage_ms[round][k] += now_ms() - t;
                }
                let t = now_ms();
                assert!(ic.crowd_with_external(q, &total));
                stage_ms[round][2] += now_ms() - t;
            }
            release_skims();
            print!("[p{q} {:.0}] ", now_ms() - tr - per_period.iter().sum::<f64>());
            per_period.push(now_ms() - tr - per_period.iter().sum::<f64>());
        }
        println!();
        per_period.clear();
        println!("round {round}: all five periods one after another {:.0} ms (nyc {:.0}, phl {:.0}, intercity {:.0})", now_ms() - tr, stage_ms[round][0], stage_ms[round][1], stage_ms[round][2]);
    }
    println!("total {:.0} ms; heap {:.0} MB before, peak {:.0} MB", now_ms() - tstart, crate::demand::alloc::mb(live0), crate::demand::alloc::mb(crate::demand::alloc::peak()));
    for q in 0..PERIODS {
        let mut seg = pops[0].seg(q);
        let mut board = pops[0].board(q);
        let mut alight = pops[0].alight(q);
        for (o, v) in seg.iter_mut().zip(pops[1].seg(q).iter().zip(ic.seg(q)).map(|(a, b)| a + b)) {
            *o += v;
        }
        for (o, v) in board.iter_mut().zip(pops[1].board(q).iter().zip(ic.board(q)).map(|(a, b)| a + b)) {
            *o += v;
        }
        for (o, v) in alight.iter_mut().zip(pops[1].alight(q).iter().zip(ic.alight(q)).map(|(a, b)| a + b)) {
            *o += v;
        }
        result.insert(format!("seg{q}"), serde_json::json!(seg));
        result.insert(format!("board{q}"), serde_json::json!(board));
        result.insert(format!("alight{q}"), serde_json::json!(alight));
        result.insert(format!("icboard{q}"), serde_json::json!(ic.board(q)));
        result.insert(format!("entries{q}"), serde_json::json!(pops[0].entries(q)));
        result.insert(format!("sum_nyc{q}"), serde_json::json!(pops[0].summary(q)));
        result.insert(format!("sum_phl{q}"), serde_json::json!(pops[1].summary(q)));
        result.insert(format!("sum_ic{q}"), serde_json::json!(ic.summary(q)));
        result.insert(format!("cm_nyc{q}"), serde_json::json!(pops[0].sub_sums().iter().map(|&v| v as f64).sum::<f64>()));
    }
    let sums: Vec<f64> = (0..PERIODS).map(|q| pops[0].summary(q)[2] + pops[1].summary(q)[2] + ic.summary(q)[2]).collect();
    println!("rail trips by period (all populations): {sums:?}");
    let now = serde_json::Value::Object(result);
    if let Ok(path) = std::env::var("TW_DUMP") {
        std::fs::write(&path, serde_json::to_string(&now).unwrap()).unwrap();
        println!("dumped to {path}");
    }
    if let Ok(path) = std::env::var("TW_COMPARE") {
        let base: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let flat = |v: &serde_json::Value| -> Vec<f64> {
            match v {
                serde_json::Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect(),
                other => vec![other.as_f64().unwrap_or(0.0)],
            }
        };
        let mut worst = 0.0f64;
        for (k, v) in now.as_object().unwrap() {
            let (a, b) = (flat(v), flat(&base[k]));
            assert_eq!(a.len(), b.len(), "{k} length");
            let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-9);
            let mut kmax = 0.0f64;
            let mut rel_total = 0.0;
            let (sa, sb): (f64, f64) = (a.iter().sum(), b.iter().sum());
            for (x, y) in a.iter().zip(&b) {
                kmax = kmax.max((x - y).abs());
            }
            if sb.abs() > 0.0 {
                rel_total = (sa - sb).abs() / sb.abs();
            }
            if k.starts_with("sum_") {
                // ms (index 4) differs run to run
                let mut m = 0.0f64;
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    if i != 4 {
                        m = m.max((x - y).abs() / y.abs().max(1.0));
                    }
                }
                kmax = m * scale;
            }
            worst = worst.max(kmax / scale);
            println!("{k}: max abs diff {kmax:.4e} = {:.2e} of max {scale:.3e}; sum differs {rel_total:.2e}", kmax / scale);
        }
        println!("WORST relative to each array's max: {worst:.3e}");
    }
}

/// Two crossing lines whose crossing stations are `gap` metres apart: riders change trains by
/// walking (300 m by default) or, within 100 m, at one merged station.
#[test]
fn walking_transfers_and_merged_stations() {
    let run = |gap: f32, xfer: f32| {
        let (mut api, mut xy, n, stops, times, tph, cars) = small();
        api.params_mut().xfer_walk_m = xfer;
        // move line B (stations 17..) sideways and give it its own crossing station
        let b_cross = (xy.len() / 2) as u32;
        xy.extend([gap, 0.0]);
        let mut stops = stops;
        for s in stops[17..].iter_mut() {
            if *s == 8 {
                *s = b_cross;
            }
        }
        for i in 17..xy.len() / 2 - 1 {
            xy[2 * i] = gap;
        }
        api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
        api.solve(0);
        let entries: f64 = api.entries(0).iter().map(|&v| v as f64).sum();
        let boards: f64 = api.board(0).iter().map(|&v| v as f64).sum();
        let n_st = api.station_map().iter().max().unwrap() + 1;
        (boards / entries, n_st)
    };
    let (none, st_far) = run(200.0, 0.0);
    let (walk, _) = run(200.0, 300.0);
    let (merged, st_near) = run(50.0, 0.0);
    assert!((none - 1.0).abs() < 1e-3, "no transfers without walking: {none}");
    assert!(walk > 1.02, "walking transfers: boardings / entries {walk}");
    assert!(merged > 1.02, "merged station transfers: {merged}");
    assert_eq!(st_far, st_near + 1);
}

/// T-176: transfers on the real New York network (nyc_real_track.json), commutes and the
/// everyday purposes, periods 0 and 2, free flow and three crowding rounds, counted and not:
/// the loads must be the same to the bit; prints the stage times both ways, the transfer
/// arrays' size, and the biggest transfers at Penn Station and Grand Central.
/// `cargo test --release --features demand transfers_real_network -- --ignored --nocapture`
#[test]
#[ignore]
fn transfers_real_network() {
    use crate::demand::everyday::EverydayApi;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let mut e = EverydayApi::new(&h, &b).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, j) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let names: Vec<String> = j["stations"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
    let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
    // route node -> snapshot station and line, and the fares' km
    let (rn_st, rn_line, hop_km) = route_nodes(&st_xy, &line_n, &stops);
    let nr = rn_st.len();
    let periods = [0usize, 2];
    let mut runs = vec![];
    for off in [true, false] {
        kernel::XFER_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        e.set_fares(2.9, 0.1, &hop_km).unwrap();
        let mut ms = vec![];
        for &q in &periods {
            for round in 0..4 {
                let total: Vec<f32> = a.seg(q).iter().zip(e.seg(q)).map(|(x, y)| x + y).collect();
                let t = now_ms();
                if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                let tc = now_ms() - t;
                let t = now_ms();
                if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                ms.push((q, round, tc, now_ms() - t));
                release_skims();
            }
        }
        let loads: Vec<Vec<f32>> = periods.iter().flat_map(|&q| [a.seg(q), a.board(q), a.alight(q), e.seg(q), e.board(q), e.alight(q)]).collect();
        runs.push((ms, loads));
    }
    kernel::XFER_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(runs[0].1, runs[1].1, "loads with and without transfers counted");
    println!("stage ms (period, round): commute off -> on, everyday off -> on");
    let (mut sum_off, mut sum_on) = (0.0, 0.0);
    for (x, y) in runs[0].0.iter().zip(&runs[1].0) {
        println!("  p{} r{}: commute {:.0} -> {:.0}, everyday {:.0} -> {:.0}", x.0, x.1, x.2, y.2, x.3, y.3);
        sum_off += x.2 + x.3;
        sum_on += y.2 + y.3;
    }
    println!("  all stages {sum_off:.0} -> {sum_on:.0} ms ({:+.1}%)", 100.0 * (sum_on / sum_off - 1.0));
    let st = a.per[0].as_ref().unwrap();
    // the assignment alone, period 0's last graph and the commute's matrix, best of 7 each
    {
        let sk = kernel::skims_for(&st.g, &st.p, &a.want);
        let er = kernel::station_exp(&sk, &st.p);
        let mo = kernel::mode_choice(&a.z, &a.gr, a.sz.as_ref().unwrap(), &er, &sk.walk, &sk.walk_max, st.g.n_st, &st.p, &a.nr);
        let (mut plain, mut counted) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..7 {
            let t = now_ms();
            let l0 = kernel::assign(&st.g, &sk, &mo.od);
            plain = plain.min(now_ms() - t);
            let mut x = vec![0.0; st.xi.len];
            let t = now_ms();
            let l1 = kernel::assign_with_transfers(&st.g, &sk, &mo.od, &st.xi, &mut x);
            counted = counted.min(now_ms() - t);
            assert_eq!(l0.seg, l1.seg);
        }
        println!("commute assignment, period 0: {plain:.1} ms, counting transfers {counted:.1} ms");
    }
    // whole free-flow stages, alternating, best of 5 (the round-by-round times above are noisy)
    let mut best = [[f64::INFINITY; 2]; 2];
    for _ in 0..5 {
        for (i, off) in [true, false].into_iter().enumerate() {
            kernel::XFER_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
            let t = now_ms();
            a.solve(0);
            best[0][i] = best[0][i].min(now_ms() - t);
            let t = now_ms();
            e.solve(0);
            best[1][i] = best[1][i].min(now_ms() - t);
            release_skims();
        }
    }
    kernel::XFER_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
    println!("free-flow stage, period 0, best of 5: commute {:.0} -> {:.0} ms, everyday {:.0} -> {:.0} ms (strategy search counted in the commute's)", best[0][0], best[0][1], best[1][0], best[1][1]);
    let st = a.per[0].as_ref().unwrap();
    let nz = st.xfer.iter().filter(|&&v| v > 0.0).count();
    println!("transfer slots {} ({} kB a period and population, plus {} kB index), {} carry riders in period 0 (commute)", st.xi.len, st.xi.len * 4 / 1024, st.xi.at.len() * 4 / 1024, nz);
    // every boarding after the first is a transfer (commutes, both periods)
    let all = vec![1u8; nr];
    let pairs = a.transfers(&all);
    let xfer_sum: f64 = pairs.chunks(3).map(|c| c[2] as f64).sum();
    let walked: f64 = pairs.chunks(3).filter(|c| rn_st[c[0] as usize] != rn_st[c[1] as usize] && a.svc.as_ref().unwrap().st_map[rn_st[c[0] as usize]] != a.svc.as_ref().unwrap().st_map[rn_st[c[1] as usize]]).map(|c| c[2] as f64).sum();
    let mut want = 0.0f64;
    for &q in &periods {
        want += a.board(q).iter().map(|&v| v as f64).sum::<f64>() - a.entries(q).iter().map(|&v| v as f64).sum::<f64>();
    }
    println!("commute transfers {xfer_sum:.0} ({walked:.0} walking between demand stations), boardings after the first {want:.0}");
    assert!((xfer_sum / want - 1.0).abs() < 1e-3);
    // the biggest at two hubs, both populations, both periods
    for hub in ["Penn Station", "Grand Central"] {
        let sel: Vec<u8> = rn_st.iter().map(|&s| names[s].contains(hub) as u8).collect();
        let mut by: std::collections::HashMap<(usize, usize), f64> = Default::default();
        let mut total = 0.0;
        for p in [a.transfers(&sel), e.transfers(&sel)] {
            for c in p.chunks(3) {
                let (x, y) = (c[0] as usize, c[1] as usize);
                if sel[x] != 0 && sel[y] != 0 {
                    *by.entry((rn_line[x], rn_line[y])).or_default() += c[2] as f64;
                    total += c[2] as f64;
                }
            }
        }
        let mut v: Vec<_> = by.into_iter().collect();
        v.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap());
        println!("{hub}: {total:.0} transfers inside (periods 0 and 2), {} line pairs; biggest:", v.len());
        for ((x, y), r) in v.iter().take(8) {
            println!("  {} -> {}: {r:.0}", lines[*x], lines[*y]);
        }
    }
}

/// T-161: the real New York network (nyc_real_track.json) with and without its station
/// complexes, commutes and the everyday purposes, every period, free flow and three crowding
/// rounds. The complexes are T-007-real-nyc.save's as the game makes them
/// (`checks/t161-complexes.mjs` writes them to ../artifacts/trainworld/t161/). Prints rail trips,
/// boardings and transfers both ways, the transfers inside Penn Station, Grand Central and
/// Times Sq, and the lines whose riders move most.
/// `cargo test --release --features demand complexes_real_network -- --ignored --nocapture`
#[test]
#[ignore]
fn complexes_real_network() {
    use crate::demand::everyday::EverydayApi;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let mut e = EverydayApi::new(&h, &b).unwrap();
    let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/../artifacts/trainworld/t161/nyc_real_complex.json")).unwrap()).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, j) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let stations = j["stations"].as_array().unwrap();
    let names: Vec<String> = stations.iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
    let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
    // the save's stations to the network's, by position to the millimetre
    let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
    let mut of: std::collections::HashMap<(i64, i64), u32> = Default::default();
    for s in cj["stations"].as_array().unwrap() {
        of.insert(key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap()), s["complex"].as_u64().unwrap() as u32);
    }
    let ns = stations.len();
    let mut unmatched = 0;
    let complex: Vec<u32> = stations.iter().enumerate().map(|(i, s)| match of.get(&key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap())) {
        Some(&c) => c,
        None => { unmatched += 1; (ns + i) as u32 }
    }).collect();
    println!("{ns} stations, {unmatched} not found in the save; {} save stations", of.len());
    assert_eq!(unmatched, 0);
    let (rn_st, rn_line, hop_km) = route_nodes(&st_xy, &line_n, &stops);
    let nr = rn_st.len();
    // a hub is the biggest complex holding a station of that name
    let hub = |name: &str| -> Vec<u8> {
        let mut size: std::collections::HashMap<u32, usize> = Default::default();
        for (i, n) in names.iter().enumerate() {
            if n.contains(name) {
                *size.entry(complex[i]).or_default() += 0;
            }
        }
        for c in &complex {
            if let Some(v) = size.get_mut(c) { *v += 1; }
        }
        let best = size.into_iter().max_by_key(|&(c, n)| (n, std::cmp::Reverse(c))).unwrap().0;
        rn_st.iter().map(|&s| (complex[s] == best) as u8).collect()
    };
    let hubs: Vec<(&str, Vec<u8>)> = ["Penn Station", "Grand Central", "Times Sq"].into_iter().map(|n| (n, hub(n))).collect();
    // how many pairs of demand stations in one complex are beyond the walk limit: the walks
    // the complexes add
    {
        a.set_complexes(&complex);
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        let svc = a.svc.as_ref().unwrap();
        let lim = a.base.xfer_walk_m;
        let (mut pairs, mut beyond, mut far) = (0, 0, 0.0f32);
        for m in kernel::complex_members(&svc.group) {
            for (i, &p) in m.iter().enumerate() {
                for &q in &m[i + 1..] {
                    let d = (svc.st_x[p as usize] - svc.st_x[q as usize]).hypot(svc.st_y[p as usize] - svc.st_y[q as usize]);
                    pairs += 1;
                    far = far.max(d);
                    if d > lim { beyond += 1; }
                }
            }
        }
        println!("{} demand stations; {pairs} pairs of demand stations inside complexes, {beyond} beyond the {lim} m walk (farthest pair {far:.0} m)", svc.st_x.len());
        let ncx = complex.iter().collect::<std::collections::HashSet<_>>().len();
        println!("{ncx} complexes; complexes of more than one demand station: {}", kernel::complex_members(&svc.group).len());
        for (name, sel) in &hubs {
            let snap: std::collections::HashSet<usize> = rn_st.iter().zip(sel).filter(|p| *p.1 != 0).map(|p| *p.0).collect();
            let dem: std::collections::HashSet<u32> = snap.iter().map(|&s| svc.st_map[s]).collect();
            let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for &s in &snap { x0 = x0.min(st_xy[2 * s]); x1 = x1.max(st_xy[2 * s]); y0 = y0.min(st_xy[2 * s + 1]); y1 = y1.max(st_xy[2 * s + 1]); }
            println!("  {name}: {} stations with a line, {} demand station(s), centres spread {:.0} x {:.0} m", snap.len(), dem.len(), x1 - x0, y1 - y0);
        }
    }
    struct Run { rail: [f64; 2], board: f64, entries: f64, xfer: f64, walked: f64, hubs: Vec<(f64, Vec<((usize, usize), f64)>)>, line_riders: Vec<f64>, seg: Vec<f64>, max_crush: f64 }
    let mut runs: Vec<Run> = vec![];
    for with in [false, true] {
        let c: &[u32] = if with { &complex } else { &[] };
        a.set_complexes(c);
        e.set_complexes(c);
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        e.set_fares(2.9, 0.1, &hop_km).unwrap();
        let t0 = now_ms();
        for q in 0..PERIODS {
            for round in 0..4 {
                let total: Vec<f32> = a.seg(q).iter().zip(e.seg(q)).map(|(x, y)| x + y).collect();
                if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                release_skims();
            }
        }
        println!("{} complexes: a day in {:.1} s", if with { "with" } else { "without" }, (now_ms() - t0) / 1000.0);
        let svc = a.svc.as_ref().unwrap();
        let (mut rail, mut board, mut entries, mut seg, mut line_riders, mut max_crush) = ([0.0; 2], 0.0, 0.0, vec![0.0f64; nr], vec![0.0f64; line_n.len()], 0.0f64);
        for q in 0..PERIODS {
            rail[0] += a.summary(q)[2];
            rail[1] += e.summary(q)[2];
            for v in [a.board(q), e.board(q)] {
                for (r, &x) in v.iter().enumerate() {
                    board += x as f64;
                    line_riders[rn_line[r]] += x as f64;
                }
            }
            entries += a.entries(q).iter().map(|&v| v as f64).sum::<f64>();
            for v in [a.seg(q), e.seg(q)] {
                for (r, &x) in v.iter().enumerate() { seg[r] += x as f64; }
            }
            max_crush = max_crush.max(a.summary(q)[7]);
        }
        let all = vec![1u8; nr];
        let (mut xfer, mut walked) = (0.0, 0.0);
        for p in [a.transfers(&all), e.transfers(&all)] {
            for c in p.chunks(3) {
                let (x, y) = (c[0] as usize, c[1] as usize);
                xfer += c[2] as f64;
                if svc.st_map[rn_st[x]] != svc.st_map[rn_st[y]] { walked += c[2] as f64; }
            }
        }
        let mut hub_out = vec![];
        for (_, sel) in &hubs {
            let mut by: std::collections::HashMap<(usize, usize), f64> = Default::default();
            let mut total = 0.0;
            for p in [a.transfers(sel), e.transfers(sel)] {
                for c in p.chunks(3) {
                    let (x, y) = (c[0] as usize, c[1] as usize);
                    if sel[x] != 0 && sel[y] != 0 {
                        let k = (rn_line[x].min(rn_line[y]), rn_line[x].max(rn_line[y]));
                        *by.entry(k).or_default() += c[2] as f64;
                        total += c[2] as f64;
                    }
                }
            }
            let mut v: Vec<_> = by.into_iter().collect();
            v.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap());
            hub_out.push((total, v));
        }
        runs.push(Run { rail, board, entries, xfer, walked, hubs: hub_out, line_riders, seg, max_crush });
    }
    let (o, n) = (&runs[0], &runs[1]);
    let pct = |x: f64, y: f64| 100.0 * (y / x - 1.0);
    println!("rail trips a day: commutes {:.0} -> {:.0} ({:+.2}%), everyday {:.0} -> {:.0} ({:+.2}%)", o.rail[0], n.rail[0], pct(o.rail[0], n.rail[0]), o.rail[1], n.rail[1], pct(o.rail[1], n.rail[1]));
    println!("boardings a day (both): {:.0} -> {:.0} ({:+.2}%); commute first boardings {:.0} -> {:.0}", o.board, n.board, pct(o.board, n.board), o.entries, n.entries);
    println!("transfers a day (both): {:.0} -> {:.0} ({:+.2}%), walking between demand stations {:.0} -> {:.0}", o.xfer, n.xfer, pct(o.xfer, n.xfer), o.walked, n.walked);
    println!("worst load of crush (commutes): {:.2} -> {:.2}", o.max_crush, n.max_crush);
    for (k, (name, _)) in hubs.iter().enumerate() {
        let ((t0, v0), (t1, v1)) = (&o.hubs[k], &n.hubs[k]);
        println!("{name}: {t0:.0} -> {t1:.0} changes a day inside the complex ({:+.1}%); biggest with complexes:", pct(*t0, *t1));
        for ((x, y), r) in v1.iter().take(6) {
            let before = v0.iter().find(|p| p.0 == (*x, *y)).map_or(0.0, |p| p.1);
            println!("  {} with {}: {before:.0} -> {r:.0}", lines[*x], lines[*y]);
        }
    }
    let mut moved: Vec<(usize, f64, f64)> = (0..line_n.len()).map(|l| (l, o.line_riders[l], n.line_riders[l])).collect();
    moved.sort_by(|p, q| (q.2 - q.1).abs().partial_cmp(&(p.2 - p.1).abs()).unwrap());
    println!("lines whose boardings a day moved most:");
    for &(l, x, y) in moved.iter().take(10) {
        println!("  {}: {x:.0} -> {y:.0} ({:+.1}%)", lines[l], pct(x, y));
    }
    let mut rel: Vec<f64> = moved.iter().filter(|m| m.1 > 1000.0).map(|m| (m.2 / m.1 - 1.0).abs()).collect();
    rel.sort_by(|p, q| p.partial_cmp(q).unwrap());
    println!("lines over 1,000 boardings a day: median change {:.2}%, 90th percentile {:.2}%, largest {:.1}%", 100.0 * rel[rel.len() / 2], 100.0 * rel[rel.len() * 9 / 10], 100.0 * rel[rel.len() - 1]);
    let mut sd: Vec<(usize, f64, f64)> = (0..nr).map(|r| (r, o.seg[r], n.seg[r])).collect();
    sd.sort_by(|p, q| (q.2 - q.1).abs().partial_cmp(&(p.2 - p.1).abs()).unwrap());
    println!("segments whose riders a day moved most:");
    for &(r, x, y) in sd.iter().take(6) {
        println!("  {} leaving {}: {x:.0} -> {y:.0}", lines[rn_line[r]], names[rn_st[r]]);
    }
}

/// T-188: the real New York network as the save has it (nyc_real_track.json with the save's
/// complexes), commutes and the everyday purposes together as the workers solve them, every
/// period, free flow and three crowding rounds, with and without the waits for full trains.
/// Prints rail trips, riders left behind and their extra minutes (from the final averaged
/// loads, by `kernel::left_behind`), the busiest lines and the lines that move most.
/// `cargo test --release --features demand left_behind_real_network -- --ignored --nocapture`
#[test]
#[ignore]
fn left_behind_real_network() {
    use crate::demand::everyday::EverydayApi;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let mut e = EverydayApi::new(&h, &b).unwrap();
    let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/../artifacts/trainworld/t161/nyc_real_complex.json")).unwrap()).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, j) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let stations = j["stations"].as_array().unwrap();
    let names: Vec<String> = stations.iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
    let lines: Vec<String> = j["lines"].as_array().unwrap().iter().map(|l| format!("{} {}", l["feed"].as_str().unwrap(), l["name"].as_str().unwrap())).collect();
    let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
    let mut of: std::collections::HashMap<(i64, i64), u32> = Default::default();
    for s in cj["stations"].as_array().unwrap() {
        of.insert(key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap()), s["complex"].as_u64().unwrap() as u32);
    }
    let ns = stations.len();
    let complex: Vec<u32> = stations.iter().enumerate().map(|(i, s)| of.get(&key(s["x"].as_f64().unwrap(), s["y"].as_f64().unwrap())).copied().unwrap_or((ns + i) as u32)).collect();
    let (rn_st, rn_line, hop_km) = route_nodes(&st_xy, &line_n, &stops);
    struct Run { rail: [f64; 2], line: Vec<f64>, left: Vec<kernel::LeftStats>, worst: Vec<(f32, usize)>, ms: f64 }
    let mut runs = vec![];
    for on in [false, true] {
        a.params_mut().left_behind = on;
        e.params_mut().left_behind = on;
        a.set_complexes(&complex);
        e.set_complexes(&complex);
        a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        a.set_fares(2.9, 0.1, &hop_km).unwrap();
        e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
        e.set_fares(2.9, 0.1, &hop_km).unwrap();
        let t0 = now_ms();
        for q in 0..PERIODS {
            for round in 0..4 {
                let add = |x: Vec<f32>, y: Vec<f32>| -> Vec<f32> { x.iter().zip(y).map(|(p, q)| p + q).collect() };
                let total = [add(a.seg(q), e.seg(q)), add(a.board(q), e.board(q)), add(a.alight(q), e.alight(q))].concat();
                if round == 0 { a.solve(q); } else { assert!(a.crowd_with_external(q, &total)); }
                if round == 0 { e.solve(q); } else { assert!(e.crowd_with_external(q, &total)); }
                release_skims();
            }
        }
        let ms = now_ms() - t0;
        let (mut rail, mut line, mut left, mut worst) = ([0.0; 2], vec![0.0f64; line_n.len()], vec![], vec![]);
        for q in 0..PERIODS {
            rail[0] += a.summary(q)[2];
            rail[1] += e.summary(q)[2];
            let (board, alight): (Vec<f32>, Vec<f32>) = (a.board(q).iter().zip(e.board(q)).map(|(x, y)| x + y).collect(), a.alight(q).iter().zip(e.alight(q)).map(|(x, y)| x + y).collect());
            for (r, &x) in board.iter().enumerate() {
                line[rn_line[r]] += x as f64;
            }
            let st = a.period(q).unwrap();
            let (w, s) = kernel::left_behind(&st.g, &st.net, &st.p, &board, &alight);
            left.push(s);
            let (mut wv, mut wr) = (0.0f32, 0usize);
            for (r, &v) in w.iter().enumerate() {
                if v > wv { wv = v; wr = r; }
            }
            worst.push((wv, wr));
        }
        runs.push(Run { rail, line, left, worst, ms });
    }
    let (o, n) = (&runs[0], &runs[1]);
    let pct = |x: f64, y: f64| 100.0 * (y / x - 1.0);
    println!("a day: {:.1} s without, {:.1} s with the waits", o.ms / 1000.0, n.ms / 1000.0);
    println!("rail trips a day: commutes {:.0} -> {:.0} ({:+.3}%), everyday {:.0} -> {:.0} ({:+.3}%)", o.rail[0], n.rail[0], pct(o.rail[0], n.rail[0]), o.rail[1], n.rail[1], pct(o.rail[1], n.rail[1]));
    for q in 0..PERIODS {
        let (x, y) = (o.left[q], n.left[q]);
        println!(
            "period {q}: left behind {:.0} -> {:.0} riders, {:.0} -> {:.0} extra rider-minutes, at {} -> {} route nodes; worst average extra wait {:.1} min ({} leaving {}) -> {:.1} min ({} leaving {})",
            x.left, y.left, x.rider_min, y.rider_min, x.stops, y.stops,
            o.worst[q].0, lines[rn_line[o.worst[q].1]], names[rn_st[o.worst[q].1]], n.worst[q].0, lines[rn_line[n.worst[q].1]], names[rn_st[n.worst[q].1]]
        );
    }
    let mut busiest: Vec<usize> = (0..line_n.len()).collect();
    busiest.sort_by(|&x, &y| o.line[y].partial_cmp(&o.line[x]).unwrap());
    println!("busiest lines, boardings a day (both populations):");
    for &l in busiest.iter().take(12) {
        println!("  {}: {:.0} -> {:.0} ({:+.2}%)", lines[l], o.line[l], n.line[l], pct(o.line[l], n.line[l]));
    }
    let mut moved: Vec<usize> = (0..line_n.len()).filter(|&l| o.line[l] > 1000.0).collect();
    moved.sort_by(|&x, &y| (n.line[y] - o.line[y]).abs().partial_cmp(&(n.line[x] - o.line[x]).abs()).unwrap());
    println!("lines whose boardings moved most:");
    for &l in moved.iter().take(10) {
        println!("  {}: {:.0} -> {:.0} ({:+.2}%)", lines[l], o.line[l], n.line[l], pct(o.line[l], n.line[l]));
    }
    let by_feed = |r: &Run, f: &str| (0..line_n.len()).filter(|&l| lines[l].starts_with(f)).map(|l| r.line[l]).sum::<f64>();
    for f in ["subway", "lirr", "mnr", "njt", "path"] {
        println!("{f}: boardings {:.0} -> {:.0} ({:+.2}%)", by_feed(o, f), by_feed(n, f), pct(by_feed(o, f), by_feed(n, f)));
    }
}

#[test]
fn empty_network_is_all_road_and_walk() {
    let (mut api, ..) = small();
    assert!(api.warm_up() > 0.0);
    assert!(api.summary(0).is_empty() && !api.solve(0));
    api.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
    assert!(api.solve(0));
    let s = api.summary(0);
    assert_eq!(s[2], 0.0);
    assert!(s[3] > 0.0);
    assert!(api.set_network(&[0.0, 0.0], &[2], &[0, 1], &[0.0; 12], &[1.0, 1.0, 1.0], &[1.0]).is_err());
}
