//! T-3542 stage 2: the hours a line splits its day at (`params::Cuts`), rows crossing into another
//! period averaged into its slot.

use super::net::{Extra, Mix, Weekend};
use super::params::*;
use super::service::lattice_deps_local;
use super::slot_tests::{busy, kept, running};
use super::world::{Op, TrackWorld};

const DAY: f64 = 86400.0;

/// The default hours cut the day into the five periods; the rules for moved hours.
#[test]
fn t3542_hours_pieces_and_rules() {
    for n in ROWS_MIN..=ROWS_MAX {
        let (p, m) = day_pieces(CUTS_DEFAULT, n);
        assert_eq!(m, 5);
        let got: Vec<(u8, u8, u8)> = p[..m].iter().map(|p| (p.a, p.b, p.q)).collect();
        assert_eq!(got, vec![(0, 6, 4), (6, 10, 0), (10, 16, 1), (16, 20, 2), (20, 24, 3)]);
        assert!(p[..m].iter().all(|p| p.k as usize == PERIOD_PLACE[n - ROWS_MIN][p.q as usize]));
        assert_eq!(shortest_row_h(CUTS_DEFAULT, n), 4, "{n} rows: four-hour rows, the lead cap as before");
        assert!(Mix::of(CUTS_DEFAULT, n, PERIOD_SLOT_N[n - ROWS_MIN]).is_none());
    }
    assert!(cuts_ok(CUTS_DEFAULT) && cuts_ok([5, 10, 16, 20, 25]) && cuts_ok([6, 8, 10, 12, 14]) && cuts_ok([0, 2, 4, 6, 22]));
    assert!(!cuts_ok([6, 7, 16, 20, 24]), "a one-hour midday");
    assert!(!cuts_ok([6, 10, 16, 20, 29]), "a one-hour night");
    assert!(!cuts_ok([24, 26, 28, 30, 32]), "the morning starts in 0-23");
    // the cuts a line of three rows does not show (10 and 16 inside high) move nothing
    assert!(Mix::of([6, 12, 14, 20, 24], 3, PERIOD_SLOT_3).is_none());
    // the evening to 1 am: its hour past midnight is in the night's period
    let (p, m) = day_pieces([6, 10, 16, 20, 25], 5);
    assert_eq!((p[0].a, p[0].b, p[0].q, p[0].k), (0, 1, 4, 1));
    assert_eq!((p[1].a, p[1].b, p[1].k), (1, 6, 2));
    assert_eq!(m, 6);
    // the morning from 5: the night's slot has an hour of the morning in it
    let mix = Mix::of([5, 10, 16, 20, 24], 5, [0, 1, 2, 3, 4]).unwrap();
    assert!(mix.mixed[4] && !mix.mixed[0]);
    assert_eq!((mix.w[4][2], mix.w[4][0]), (5.0 / 6.0, 1.0 / 6.0));
    assert_eq!(mix.main[4], 2);
    // two-hour rows: departures lead a stretch by at most an hour
    assert_eq!(shortest_row_h([6, 8, 10, 12, 14], 5), 2);
    assert_eq!(shortest_row_h([6, 8, 10, 12, 14], 3), 2, "high 6-12, medium 12-14");
    // a row with no slot of its own (midday 12-14, inside the midday period)
    let mix = Mix::of([6, 12, 14, 20, 24], 5, [0, 1, 2, 3, 4]).unwrap();
    assert!(mix.mixed[1] && mix.w[1][3] > 0.0 && mix.w[1][0] > 0.0 && mix.w[1][4] > 0.0);
}

/// The line's departures in a weekday, local time.
fn deps(w: &TrackWorld, l: usize) -> Vec<(f64, u8, f64)> {
    let lat = w.svc.lattice(&w.net, l).expect("runs");
    let mut out = vec![];
    lattice_deps_local(&lat, 0, 3.0 * DAY, 4.0 * DAY, &mut out);
    out.into_iter().map(|(d, lev, hw, _)| (d, lev, hw)).collect()
}

/// The gaps between line `l`'s departures from `from_h` to `to_h` (local hours, each moved by the
/// pattern's lead, as departures lead their stretch).
fn gaps(w: &TrackWorld, l: usize, from_h: f64, to_h: f64) -> Vec<f64> {
    let lead = w.svc.lattice(&w.net, l).unwrap().lead;
    let (a, b) = (3.0 * DAY + from_h * 3600.0 - lead, 3.0 * DAY + to_h * 3600.0 - lead);
    let t: Vec<f64> = deps(w, l).iter().map(|x| x.0).filter(|&t| t >= a && t < b).collect();
    t.windows(2).map(|x| x[1] - x[0]).collect()
}

/// A corridor with a headway line (rush or morning every 5 minutes, evening 10, night 20) and a
/// count line on part of it, far from full, in a game of `rows` rows.
fn corridor(rows: u8) -> (TrackWorld, usize, usize) {
    let (mut w, ns) = super::tests::corridor(6);
    super::tests::terminals_at_dead_ends(&mut w);
    let h = super::tests::new_line(&mut w, &ns, [8, 4, 2]);
    w.apply(Op::AutoTrains { id: h, auto: Some([300, 600, 1200]), trains: Some([8, 4, 2]) }).unwrap();
    let c = super::tests::new_line(&mut w, &ns[1..4], [2, 1, 1]);
    if rows > 3 {
        let op = w.rows_ops(None, rows).unwrap();
        w.apply(op).unwrap();
    }
    (w, h as usize, c as usize)
}

/// The morning from 5 in a game of five rows. The headway line: its night's slot runs the
/// time-weighted rate, (5 x 3 + 12) / 6 = 4.5 trains an hour; its hour of the morning leaves every 5
/// minutes and the rest of the night every 20; demand reads the rate of the mean headway the night's
/// riders meet. The count line: the night's slot runs (5 x its night trains + its morning trains) / 6
/// over the night's round trip. The save keeps the hours to the bit; undo puts the default back.
#[test]
fn t3542_morning_from_five() {
    let (mut w, l, c) = corridor(5);
    let before = (w.net.line_tph[l], w.net.line_tph[c]);
    eprintln!("before: headway line {:?}, count line {:?}, count trains {:?}", before.0, before.1, w.net.trains_of(c));
    assert!((before.0[0] - 12.0).abs() < 1e-3 && (before.0[4] - 3.0).abs() < 1e-3, "the headway line runs its headways: {:?}", before.0);
    w.apply(Op::Hours { line: None, cuts: [5, 10, 16, 20, 24] }).unwrap();
    let (after, after_c) = (w.net.line_tph[l], w.net.line_tph[c]);
    eprintln!("after: headway line {after:?}, demand {:?}; count line {after_c:?}, demand {:?}", w.svc.demand_tph(&w.net, l), w.svc.demand_tph(&w.net, c));
    assert!((after[4] - 4.5).abs() < 1e-3, "night slot {} against (5 x 3 + 12) / 6 = 4.5", after[4]);
    assert!((after[0] - before.0[0]).abs() < 1e-4, "the morning's own slot keeps its rate");
    let (m, n) = (gaps(&w, l, 5.0, 6.0), gaps(&w, l, 1.0, 4.0));
    eprintln!("5-6 gaps {m:?}\n1-4 gaps {n:?}");
    assert!(!m.is_empty() && m.iter().all(|g| (g - 300.0).abs() < 0.05), "5-6 every 5 minutes");
    assert!(!n.is_empty() && n.iter().all(|g| (g - 1200.0).abs() < 0.05), "1-4 every 20 minutes");
    // the count line: its trains over the night's round trip, the morning's for a sixth of it
    // (its morning hour no more often than its morning runs: the borrowed hour's cap)
    let (t, trip) = (w.net.trains_of(c).unwrap(), w.svc.lines[c].round_trip[4]);
    let (night_r, morning_r) = (t[2] as f64 * 3600.0 / trip, (t[0] as f64 * 3600.0 / trip).min(after_c[0] as f64));
    let want = (5.0 * night_r + morning_r) / 6.0;
    eprintln!("count line night: trains {t:?}, round trip {trip} s, night {night_r}, morning hour min({} x 3600 / {trip}, {}) = {morning_r}: (5 x {night_r} + {morning_r}) / 6 = {want}", t[0], after_c[0]);
    assert!((after_c[4] as f64 - want).abs() < 1e-3, "count line night {} against {want}", after_c[4]);
    // demand: the night's riders by its shape, those 0-5 waiting 20 minutes, 5-6 five
    let cum = |h: f64| super::span::period_cum(4, h * 3600.0);
    let (c1, c2) = (cum(5.0) - cum(0.0), cum(6.0) - cum(5.0));
    let mean = (c1 * 1200.0 + c2 * 300.0) / (c1 + c2);
    let dem = w.svc.demand_tph(&w.net, l);
    eprintln!("demand night: riders 0-5 {c1:.4}, 5-6 {c2:.4}, mean headway {mean:.1} s, {} an hour", 3600.0 / mean);
    assert!((dem[4] as f64 - 3600.0 / mean).abs() < 1e-3, "demand's night {} against {}", dem[4], 3600.0 / mean);
    assert_eq!(dem[0], after[0]);
    // saved and loaded: the hours, every rate and every departure to the bit
    let back = TrackWorld::new(super::save::decode(&super::save::encode(&w.net)).unwrap());
    assert_eq!(back.net.hours_default, [5, 10, 16, 20, 24]);
    assert_eq!(kept(&back), kept(&w));
    assert_eq!(deps(&back, l), deps(&w, l));
    // undo: the default hours, the rates as before
    w.undo().unwrap().unwrap();
    assert_eq!(w.net.hours_default, CUTS_DEFAULT);
    assert!(w.net.mix(l).is_none());
    assert!((w.net.line_tph[l][4] - 3.0).abs() < 1e-4);
}

/// Two-hour rows at every count: the settle and the departures run, in day order with leads of an
/// hour at most, the evening's middle (12-14 at these hours) runs its own headway, and the save keeps
/// it all to the bit.
#[test]
fn t3542_two_hour_rows() {
    for n in [3u8, 4, 5] {
        let (mut w, l, c) = corridor(n);
        w.apply(Op::Hours { line: None, cuts: [6, 8, 10, 12, 14] }).unwrap();
        for k in [l, c] {
            let lat = w.svc.lattice(&w.net, k).unwrap();
            assert!(lat.lead <= 3600.0, "{n} rows: line {k} lead {}", lat.lead);
            let d = deps(&w, k);
            assert!(d.windows(2).all(|x| x[1].0 > x[0].0), "{n} rows: line {k} departures in order");
        }
        let g = gaps(&w, l, 12.25, 13.75);
        eprintln!("{n} rows: rates {:?}, 12:15-13:45 gaps {g:?}", w.net.line_tph[l]);
        assert!(!g.is_empty() && g.iter().all(|x| (x - 600.0).abs() < 0.05), "{n} rows: the evening every 10 minutes");
        let back = TrackWorld::new(super::save::decode(&super::save::encode(&w.net)).unwrap());
        assert_eq!(kept(&back), kept(&w), "{n} rows");
        assert_eq!(deps(&back, l), deps(&w, l), "{n} rows");
    }
}

/// A line's own hours: only it moves, the other line keeps its rates to the bit, and Same as game
/// (its rows back to the game's) drops them; a one-hour row is refused.
#[test]
fn t3542_line_hours_own() {
    let (mut w, l, c) = corridor(3);
    let op = w.rows_ops(Some(l as u32), 5).unwrap();
    w.apply(op).unwrap();
    w.apply(Op::Hours { line: Some(l as u32), cuts: [5, 10, 16, 21, 25] }).unwrap();
    assert_eq!(w.net.hours(l), [5, 10, 16, 21, 25]);
    assert_eq!(w.net.hours(c), CUTS_DEFAULT);
    assert!(w.net.mix(l).is_some() && w.net.mix(c).is_none());
    let g = gaps(&w, l, 20.25, 20.75);
    assert!(!g.is_empty() && g.iter().all(|x| (x - 300.0).abs() < 0.05), "the afternoon to 21: 20-21 every 5 minutes, {g:?}");
    let op = w.rows_ops(Some(l as u32), 0).unwrap();
    w.apply(op).unwrap();
    assert_eq!(w.net.line_x[l].hours, [0; 5]);
    assert!(w.net.mix(l).is_none());
    assert!(w.apply(Op::Hours { line: None, cuts: [6, 7, 16, 20, 24] }).is_err(), "a one-hour row is refused");
}

/// The 5-row figures through `busy` (shared, full track): moving the hours there and back gives
/// the game it was, to the bit.
#[test]
fn t3542_hours_there_and_back() {
    let mut a = busy(None);
    let op = a.rows_ops(None, 5).unwrap();
    a.apply(op).unwrap();
    let was = kept(&a);
    let l = running(&a)[0] as usize;
    a.apply(Op::Hours { line: None, cuts: [5, 11, 15, 21, 25] }).unwrap();
    assert!(a.net.mix(l).is_some());
    a.undo().unwrap().unwrap();
    assert!(a.net.mix(l).is_none());
    // (a settle from other rates ends within its tolerance, not to the bit)
    eprintln!("same after undo: {}", kept(&a) == was);
}

/// Line `l`'s trains per row of five, in the day's order (morning, midday, afternoon, evening,
/// night), on weekdays and set by hand on weekends, as the schedule window sets them.
fn five_rows(w: &mut TrackWorld, l: usize, week: [u32; 5], weekend: [u32; 5]) {
    let p = |r: [u32; 5]| unplace([r[0], r[3], r[4], r[1], r[2]]);
    let ((m, x), (wm, wx)) = (p(week), p(weekend));
    let d = w.net.line_data(l as u32).unwrap();
    w.apply(Op::Batch(vec![Op::TrainCounts { id: l as u32, trains: Some(m), tph: d.tph }, Op::Extra { id: l as u32, x: Extra { trains: x, ..d.x } }])).unwrap();
    let d = w.net.line_data(l as u32).unwrap();
    w.apply(Op::Batch(vec![Op::Weekend { id: l as u32, weekend: Weekend { counts: Some(wm), ..d.weekend } }, Op::Extra { id: l as u32, x: Extra { weekend_counts: wx, ..d.x } }])).unwrap();
}

/// Line `l`'s run 0 departures on local day `day` (day 6 a Saturday) from `from_h` to `to_h`,
/// each hour moved by the pattern's lead, as `gaps`.
fn deps_between(w: &TrackWorld, l: usize, day: f64, from_h: f64, to_h: f64) -> Vec<f64> {
    let lat = w.svc.lattice(&w.net, l).unwrap();
    let (a, b) = (day * DAY + from_h * 3600.0 - lat.lead, day * DAY + to_h * 3600.0 - lat.lead);
    let mut out = vec![];
    lattice_deps_local(&lat, 0, a - DAY, b + DAY, &mut out);
    out.into_iter().map(|x| x.0).filter(|&t| t >= a && t < b).collect()
}

/// T-4227: where moved hours share a period between two rows (midday 11-15 inside the 10-16
/// period), each row's weekend trains run at weekends: a row with none on weekdays runs its four
/// weekend trains as four weekday trains would run there, at the rate the schedule window shows,
/// and the row beside it no more often than its own two run (before, the midday ran nothing and the
/// morning's hour ran the period's every train).
#[test]
fn t4227_a_shared_period_runs_each_rows_weekend_trains() {
    let world = |week: [u32; 5], weekend: [u32; 5]| {
        let (mut w, _, c) = corridor(5);
        w.apply(Op::Hours { line: None, cuts: [6, 11, 15, 20, 24] }).unwrap();
        five_rows(&mut w, c, week, weekend);
        assert!(w.net.mix(c).is_some_and(|m| m.mixed[1]), "the midday period is shared");
        (w, c)
    };
    let gaps_in = |w: &TrackWorld, c: usize, day: f64, a: f64, b: f64| -> Vec<f64> { deps_between(w, c, day, a, b).windows(2).map(|x| x[1] - x[0]).collect() };
    let even = |g: &[f64], what: &str| -> f64 {
        assert!(!g.is_empty() && g.iter().all(|x| (x - g[0]).abs() < 1e-3 * g[0]), "{what}: gaps {g:?}");
        g[0]
    };
    // four weekday midday trains, the same every day: the reference
    let (a, ca) = world([2, 4, 2, 1, 1], [2, 4, 2, 1, 1]);
    let want = even(&gaps_in(&a, ca, 3.0, 11.25, 14.75), "Wednesday's midday, four weekday trains");
    let (w, c) = world([2, 0, 2, 1, 1], [2, 4, 2, 1, 1]);
    assert!(deps_between(&w, c, 3.0, 11.25, 14.75).is_empty(), "no weekday midday trains");
    let got = even(&gaps_in(&w, c, 6.0, 11.25, 14.75), "Saturday's midday");
    assert!((got - want).abs() < 0.03 * want, "Saturday's midday every {got:.1} s, four weekday trains every {want:.1} s");
    let trip = w.svc.lines[c].round_trip[1];
    let shown = w.svc.fleet_on(&w.net, c, WEEKEND).map(|f| w.net.place_tph_in(c, 1, 3, w.net.line_tph_on(c, WEEKEND, 1), trip, &f)).unwrap();
    assert!((3600.0 / shown as f64 - got).abs() < 1e-3 * got, "the window's weekend midday every {:.1} s, run every {got:.1} s", 3600.0 / shown as f64);
    let (own, borrowed) = (even(&gaps_in(&w, c, 6.0, 7.0, 9.0), "Saturday's morning"), even(&gaps_in(&w, c, 6.0, 10.0, 11.0), "Saturday's morning hour in the midday period"));
    assert!(borrowed >= own * (1.0 - 1e-3), "the morning's hour in the midday every {borrowed:.1} s, its own hours every {own:.1} s");
    // and the other way: weekday midday trains, none at weekends
    let (w, c) = world([2, 4, 2, 1, 1], [2, 0, 2, 1, 1]);
    assert!(deps_between(&w, c, 6.0, 11.25, 14.75).is_empty(), "no weekend midday trains");
    let got = even(&gaps_in(&w, c, 3.0, 11.25, 14.75), "Wednesday's midday");
    assert!((got - want).abs() < 1e-6 * want, "Wednesday's midday every {got:.1} s, {want:.1} s with the same weekend");
}
