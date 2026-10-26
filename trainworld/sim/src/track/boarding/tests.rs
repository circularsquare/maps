use super::*;
use crate::track::tests::{edge, line, node, station, world};

#[test]
fn ride_km_follows_the_alighting_shares() {
    // stops 0, 10, 20 and 120 km; half of whoever is aboard gets off at each middle stop
    let n = 4;
    let alpha: Vec<f64> = (0..PERIODS).flat_map(|_| [0.0, 0.5, 0.5, 1.0]).collect();
    let km = ride_km(&alpha, &[0.0, 10_000.0, 20_000.0, 120_000.0], n);
    let want = [10.0 + 0.5 * (10.0 + 0.5 * 100.0), 10.0 + 0.5 * 100.0, 100.0, 0.0];
    for p in 0..PERIODS {
        for k in 0..n {
            assert!((km[p * n + k] - want[k]).abs() < 1e-9, "{km:?}");
        }
    }
}

#[test]
fn shapes_keep_the_period_total_and_reach_the_busiest_hour() {
    for s in [Shape::Tent, Shape::Falling, Shape::Valley] {
        for hours in [4.0, 6.0] {
            let c = shape_strength(s, hours, 1.4);
            assert!((shape_cum(s, c, 1.0) - 1.0).abs() < 1e-12);
            // the busiest whole hour, against the mean hour
            let best = (0..(hours as usize))
                .map(|h| (shape_cum(s, c, (h as f64 + 1.0) / hours) - shape_cum(s, c, h as f64 / hours)) * hours)
                .fold(0.0f64, f64::max);
            assert!((best - 1.4).abs() < 1e-9, "{s:?} {hours}: {best}");
        }
    }
    // the rate is the derivative of the cumulative curve
    let a = Arrivals::new([1000.0, 600.0, 900.0, 300.0, 50.0], &[1.4, 1.15, 1.4, 1.3, 1.5]);
    for t in [7.2 * 3600.0, 13.0 * 3600.0, 21.5 * 3600.0, 2.0 * 3600.0, 5.9 * 3600.0] {
        let d = (a.cum(t + 1.0) - a.cum(t - 1.0)) / 2.0;
        assert!((d - a.rate(t)).abs() < 1e-6, "{t}: {d} {}", a.rate(t));
    }
    assert!((a.cum(DAY) - a.cum(0.0) - 2850.0).abs() < 1e-9);
}

/// T-842b: a peak-only stop's riders arrive only while a train is due within their tolerance,
/// each period's total kept; a stop served all day arrives as before to the bit.
#[test]
fn masked_arrivals_keep_each_period_and_skip_the_gaps() {
    let per = [1000.0, 600.0, 900.0, 300.0, 50.0];
    let peak = [1.4, 1.15, 1.4, 1.3, 1.5];
    let every = |from: f64, to: f64| -> Vec<f64> { (0..).map(|i| from + 900.0 * i as f64).take_while(|&t| t <= to).collect() };
    let all = Arrivals::with_mask(per, &peak, span::stop_mask(&every(0.0, DAY - 1.0)));
    let plain = Arrivals::new(per, &peak);
    for t in [3.0, 7.2, 13.0, 21.5] {
        assert_eq!(all.cum(t * 3600.0).to_bits(), plain.cum(t * 3600.0).to_bits());
    }
    let mut deps = every(6.0 * 3600.0, 10.0 * 3600.0);
    deps.extend(every(16.0 * 3600.0, 20.0 * 3600.0));
    let a = Arrivals::with_mask(per, &peak, span::stop_mask(&deps));
    assert!((a.cum(DAY) - a.cum(0.0) - 2850.0).abs() < 1e-6);
    for (p, &(h0, h1)) in PERIOD_HOURS.iter().enumerate() {
        let got = a.cum(h1 * 3600.0) - a.cum(h0 * 3600.0);
        // the evening has no train within 6 h: its riders arrive as if unmasked
        assert!((got - per[p]).abs() < 1e-6, "period {p}: {got}");
    }
    // nobody arrives at 10:30 (5.5 h to the next train is past most tolerances; a few)
    assert!(a.rate(10.5 * 3600.0) < 0.2 * plain.rate(10.5 * 3600.0));
    assert!(a.rate(10.1 * 3600.0) < 0.05 * plain.rate(10.1 * 3600.0), "5.9 h to the next train");
    assert!(a.rate(15.8 * 3600.0) > plain.rate(15.8 * 3600.0), "the midday's riders arrive near its end");
    // within the tolerance ramp the share steps once a minute: one second either side of a step
    for t in [7.2 * 3600.0, 13.0 * 3600.0 + 30.0, 15.9 * 3600.0, 2.0 * 3600.0 + 30.0] {
        let d = (a.cum(t + 1.0) - a.cum(t - 1.0)) / 2.0;
        assert!((d - a.rate(t)).abs() < 1e-6, "{t}: {d} {}", a.rate(t));
    }
}

/// A - B - C, a line every 6 minutes at high demand.
fn setup(tph: f32) -> (crate::track::world::TrackWorld, u32) {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let b = node(&mut w, 3000.0, 0.0, 0);
    let c = node(&mut w, 6000.0, 0.0, 0);
    edge(&mut w, a, b, &[], 0).unwrap();
    edge(&mut w, b, c, &[], 0).unwrap();
    for n in [a, b, c] {
        station(&mut w, n, 200);
    }
    let l = line(&mut w, &[a, b, c], [tph, tph / 2.0, tph / 4.0]);
    (w, l)
}

/// Riders boarding at A for C (run 0) and at C for A (run 1), `per_hour` an hour in every
/// period.
fn riders(l: u32, per_hour: [f32; PERIODS]) -> Vec<f32> {
    let mut v = vec![1.0; PERIODS]; // flat: the peak factor 1
    v.extend([l as f32, 3.0]);
    for p in 0..PERIODS {
        let (h0, h1) = PERIOD_HOURS[p];
        let x = per_hour[p] * (h1 - h0) as f32;
        v.extend([x, 0.0, 0.0, x, 0.0, 0.0]); // board: run 0 at A, run 1 at C
        v.extend([0.0, 0.0, x, 0.0, 0.0, x]); // alight: run 0 at C, run 1 at A
    }
    v
}

#[test]
fn a_weekend_hour_reads_the_weekend_account() {
    // T-260, T-262: game day 1 is a Monday, days 6 and 7 Saturday and Sunday
    assert_eq!(day_kind_at(86400.0 + 7.0 * 3600.0), 0);
    assert_eq!(day_kind_at(5.0 * 86400.0 + 23.0 * 3600.0), 0);
    assert_eq!(day_kind_at(6.0 * 86400.0), 1);
    assert_eq!(day_kind_at(7.0 * 86400.0 + 3600.0), 2);
    assert_eq!(day_kind_at(8.0 * 86400.0), 0);
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders_day(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]), 0);
    // before the weekend's riders, a weekend hour reads the weekday's
    let sat = 6.0 * 86400.0 + 8.0 * 3600.0;
    assert_eq!(b.kind_at(sat, l as usize), 0);
    b.set_riders_day(&riders(l, [100.0, 300.0, 200.0, 200.0, 50.0]), 1);
    assert_eq!(b.kind_at(sat, l as usize), 1);
    let mon = 8.0 * 86400.0 + 8.0 * 3600.0;
    let (h_sat, h_mon) = (b.hour(&w.net, &w.svc, sat), b.hour(&w.net, &w.svc, mon));
    // 8-9 in the morning peak: 500 an hour at each end on a weekday, 100 on Saturday
    assert!((arriving(&h_mon) / arriving(&h_sat) - 5.0).abs() < 0.01, "{} {}", arriving(&h_mon), arriving(&h_sat));
    assert_ne!(b.line_summary_day(&w.svc, l as usize, 1)[0], b.line_summary_day(&w.svc, l as usize, 0)[0]);
}

fn arriving(h: &[f32]) -> f32 {
    let (nt, nl, nq) = (h[0] as usize, h[1] as usize, h[2] as usize);
    let q0 = 5 + 2 * nt + nl;
    (0..nq).map(|k| h[q0 + k * QUEUE_STRIDE + 7]).sum::<f32>()
}

#[test]
fn a_holiday_hour_reads_the_sunday_account_on_its_line_only() {
    // T-872, T-262: Monday, game day 8, is a holiday for line `l` only, run as a Sunday
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders_day(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]), 0);
    b.set_riders_day(&riders(l, [100.0, 300.0, 200.0, 200.0, 50.0]), 1);
    b.set_riders_day(&riders(l, [50.0, 250.0, 150.0, 150.0, 40.0]), 2);
    let (sat, sun, mon, tue) = (6.0 * DAY + 8.0 * 3600.0, 7.0 * DAY + 8.0 * 3600.0, 8.0 * DAY + 8.0 * 3600.0, 9.0 * DAY + 8.0 * 3600.0);
    let (h_sat, h_sun) = (b.hour(&w.net, &w.svc, sat), b.hour(&w.net, &w.svc, sun));
    let (h_mon, h_tue) = (b.hour(&w.net, &w.svc, mon), b.hour(&w.net, &w.svc, tue));
    assert!((arriving(&h_mon) / arriving(&h_sat) - 5.0).abs() < 0.01);
    assert!((arriving(&h_mon) / arriving(&h_sun) - 10.0).abs() < 0.01);
    b.set_holidays(&[l as f64, 2.0, 8.0, 2.0, 20.0, 2.0]);
    assert_eq!(b.kind_at(mon, l as usize), 2);
    assert_eq!(b.kind_at(mon, l as usize + 1), 0);
    assert_eq!(b.kind_at(tue, l as usize), 0);
    assert_eq!(arriving(&b.hour(&w.net, &w.svc, mon)), arriving(&h_sun));
    assert_eq!(arriving(&b.hour(&w.net, &w.svc, tue)), arriving(&h_tue));
    // the day's account is Sunday's: today at Monday's end equals Sunday's
    assert_eq!(b.today(&w.svc, mon + 15.0 * 3600.0), b.today(&w.svc, sun + 15.0 * 3600.0));
    b.set_holidays(&[]);
    assert_eq!(arriving(&b.hour(&w.net, &w.svc, mon)), arriving(&h_mon));
    // a one-day weekend: Saturday listed as a weekday reads the weekday's account; Friday and
    // Saturday: Friday as a Sunday, Saturday as a Saturday
    b.set_holidays(&[l as f64, 1.0, 6.0, 0.0]);
    assert_eq!(b.kind_at(sat, l as usize), 0);
    assert_eq!(arriving(&b.hour(&w.net, &w.svc, sat)), arriving(&h_tue));
    let fri = 5.0 * DAY + 8.0 * 3600.0;
    b.set_holidays(&[l as f64, 2.0, 5.0, 2.0, 7.0, 0.0]);
    assert_eq!((b.kind_at(fri, l as usize), b.kind_at(sat, l as usize), b.kind_at(sun, l as usize)), (2, 1, 0));
}

#[test]
fn an_away_day_reads_its_own_account_or_the_sundays() {
    // T-264: a superholiday's away day is kind 3; without its riders it reads Sunday's
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders_day(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]), 0);
    b.set_riders_day(&riders(l, [50.0, 250.0, 150.0, 150.0, 40.0]), 2);
    let (sun, mon) = (7.0 * DAY + 8.0 * 3600.0, 8.0 * DAY + 8.0 * 3600.0);
    let h_sun = b.hour(&w.net, &w.svc, sun);
    b.set_holidays(&[l as f64, 1.0, 8.0, AWAY as f64]);
    assert_eq!(b.kind_at(mon, l as usize), 2);
    assert_eq!(arriving(&b.hour(&w.net, &w.svc, mon)), arriving(&h_sun));
    b.set_riders_day(&riders(l, [25.0, 300.0, 100.0, 100.0, 20.0]), AWAY);
    assert_eq!(b.kind_at(mon, l as usize), AWAY);
    assert!((arriving(&h_sun) / arriving(&b.hour(&w.net, &w.svc, mon)) - 2.0).abs() < 0.01);
    b.set_riders_day(&[], AWAY);
    assert_eq!(b.kind_at(mon, l as usize), 2);
}

/// T-798: on a circle 0-1-2-3-4-5-0, riders from 4 to 1 ride on through the first stop: the
/// train leaves it with them aboard, and they count as boarding there again.
#[test]
fn a_circle_carries_its_riders_round_through_the_first_stop() {
    let (mut w, ns) = crate::track::tests::ring_fixture();
    let mut stops = ns.clone();
    stops.push(ns[0]);
    let l = line(&mut w, &stops, [10.0, 5.0, 2.5]);
    let n = 7;
    let mut v = vec![1.0; PERIODS];
    v.extend([l as f32, n as f32]);
    for p in 0..PERIODS {
        let (h0, h1) = PERIOD_HOURS[p];
        let x = 100.0 * (h1 - h0) as f32;
        let mut board = vec![0.0; 2 * n];
        let mut alight = vec![0.0; 2 * n];
        board[4] = x;
        alight[1] = x;
        v.extend(board);
        v.extend(alight);
    }
    let mut b = Boarding::default();
    b.set_riders(&v);
    b.ensure(&w.net, &w.svc);
    let acc = b.accounts[0][l as usize].as_ref().unwrap();
    let ra = &acc.runs[0];
    let i = ra.deps.iter().position(|&d| d > 8.0 * 3600.0).unwrap();
    // ten an hour at 100 an hour: about ten aboard from 4 round to 1
    for k in [4, 5, 0] {
        assert!((ra.load[i * n + k] - 10.0).abs() < 1.5, "stop {k}: {}", ra.load[i * n + k]);
    }
    assert!(ra.load[i * n + 1] < 0.5 && ra.load[i * n + 2] < 0.5, "they leave at 1");
    assert_eq!(ra.load[i * n + 6], 0.0);
    assert!(ra.q_after.iter().all(|&q| q < 1e-3), "nobody is left at the first stop");
    assert!((ra.boarded - ra.alighted).abs() < 1e-6 * ra.boarded.max(1.0));
    // demand's riders board once at 4 and once more, staying aboard, at the first stop
    let day = 100.0 * 24.0;
    assert!((ra.boarded - 2.0 * day).abs() < 0.02 * day, "{}", ra.boarded);
}

#[test]
fn a_line_with_room_carries_demands_load_and_leaves_no_one() {
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
    b.ensure(&w.net, &w.svc);
    let acc = b.accounts[0][l as usize].as_ref().unwrap();
    for ra in &acc.runs {
        // every rider in the day boards and gets off
        assert!((ra.arrived - ra.boarded).abs() < 1e-6 * ra.arrived.max(1.0), "{} {}", ra.arrived, ra.boarded);
        assert!((ra.boarded - ra.alighted).abs() < 1e-6 * ra.boarded.max(1.0));
        assert!(ra.q_after.iter().all(|&q| q < 1e-3));
        // the day's arrivals are demand's
        assert!((ra.arrived - (500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0)).abs() < 1e-6);
    }
    // a morning-peak train carries a headway's riders: 500 an hour, 10 trains an hour
    let ra = &acc.runs[0];
    let i = ra.find(8.0 * 3600.0 + 60.0).or_else(|| ra.deps.iter().position(|&d| d > 8.0 * 3600.0)).unwrap();
    assert!((ra.load[i * 3] - 50.0).abs() < 1.0, "{}", ra.load[i * 3]);
    assert_eq!(ra.load[i * 3 + 1], ra.load[i * 3]);
    assert_eq!(ra.load[i * 3 + 2], 0.0);
}

#[test]
fn today_counts_up_to_the_days_riders() {
    // T-293: boardings since midnight grow through the day to the day's riders
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
    b.ensure(&w.net, &w.svc);
    let day = 500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0;
    let at = |h: f64| {
        let v = b.today(&w.svc, 2.0 * DAY + h * 3600.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], l as f64);
        v[3]
    };
    let (night, noon, end) = (at(0.0), at(12.0), at(23.999));
    // at midnight: at most the trains leaving at 00:00 (the night's 50 an hour at each end)
    assert!(night < 150.0, "{night}");
    assert!(noon > 0.3 * 2.0 * day && noon < 0.7 * 2.0 * day, "{noon}");
    // both directions, everyone who arrived by the last train has boarded
    assert!((end - 2.0 * day).abs() < 0.02 * day, "{end} {}", 2.0 * day);
    // stops: A and C board, B nobody
    let v = b.today(&w.svc, 2.0 * DAY + 23.999 * 3600.0);
    let spans = v[7] as usize;
    let stops = &v[8 + 2 * spans..8 + 2 * spans + 3];
    assert!(stops[1] < 1e-6 && (stops[0] - day).abs() < 0.02 * day, "{stops:?}");
}

#[test]
fn past_day_holds_a_whole_day_at_any_hour() {
    // T-446: the past 24 hours on weekdays are a whole weekday's riders, whatever the hour
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [500.0, 200.0, 500.0, 200.0, 50.0]));
    b.ensure(&w.net, &w.svc);
    let day = 2.0 * (500.0 * 8.0 + 200.0 * 10.0 + 50.0 * 6.0);
    for h in [0.0, 0.5, 7.25, 12.0, 18.9, 23.999] {
        let v = b.past_day(&w.svc, 2.0 * DAY + h * 3600.0);
        assert_eq!(v[0], 1.0);
        assert!((v[3] - day).abs() < 0.02 * day, "{h}: {} {day}", v[3]);
    }
}

#[test]
fn period_totals_match_demand_with_the_busiest_hour_shapes() {
    let (w, l) = setup(12.0);
    let mut b = Boarding::default();
    let per_hour = [900.0, 400.0, 1000.0, 300.0, 40.0];
    let mut packed = riders(l, per_hour);
    packed[..PERIODS].copy_from_slice(&[1.4, 1.15, 1.4, 1.3, 1.5]);
    b.set_riders(&packed);
    b.ensure(&w.net, &w.svc);
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let mut by_period = [0.0f64; PERIODS];
    for (i, &d) in ra.deps.iter().enumerate() {
        by_period[period_at(d)] += ra.load[i * 3] as f64;
    }
    for p in 0..PERIODS {
        let (h0, h1) = PERIOD_HOURS[p];
        let want = per_hour[p] as f64 * (h1 - h0);
        // riders arriving in the last headway of a period board the next period's first train
        let hw_riders = per_hour[p].max(per_hour[(p + PERIODS - 1) % PERIODS]) as f64 * 3600.0 / (12.0 / 4.0) / 3600.0;
        assert!((by_period[p] - want).abs() <= hw_riders, "period {p}: {} against {want}", by_period[p]);
    }
    // and the morning peak's busiest train carries about 1.4 times the mean train
    let mean = per_hour[0] as f64 / 12.0;
    let peak = ra.deps.iter().enumerate().filter(|(_, &d)| d >= 6.0 * 3600.0 && d < 10.0 * 3600.0).map(|(i, _)| ra.load[i * 3] as f64).fold(0.0, f64::max);
    assert!(peak > 1.45 * mean && peak < 1.85 * mean, "{peak} against a mean of {mean}");
}

#[test]
fn an_overloaded_line_leaves_people_behind_and_the_queue_clears() {
    // 10 trains an hour of crush 1,600 (10 cars) = 16,000 an hour; 20,000 an hour arrive in
    // the morning peak
    let (w, l) = setup(10.0);
    let cap = CRUSH_PER_CAR * w.svc.lines[l as usize].cars as f64;
    assert_eq!(cap, 1600.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
    b.ensure(&w.net, &w.svc);
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let m = ra.deps.len();
    // the queue at A, train by train through the day
    let q: Vec<(f64, f32, f32)> = (0..m).map(|j| (ra.order_t[j], ra.q_after[ra.order[j] as usize * 3], ra.load[ra.order[j] as usize * 3])).collect();
    let at = |h: f64| q.iter().filter(|e| e.0 < h * 3600.0).last().unwrap().1;
    assert!(at(7.0) > 0.0 && at(9.0) >= at(7.0), "{} {}", at(7.0), at(9.0));
    // T-1067: 400 more a train than it takes, until the platform holds WAITING_CAP as a train
    // comes (2,000 arriving a headway): 3,400 left as each leaves, from about 07:00 to 10:00
    assert!(q.iter().filter(|e| e.0 > 7.2 * 3600.0 && e.0 < 10.0 * 3600.0).all(|e| (e.1 as f64 - (WAITING_CAP - cap)).abs() < 1e-3), "{}", at(9.0));
    // every train in the peak leaves full
    assert!(q.iter().filter(|e| e.0 > 6.5 * 3600.0 && e.0 < 10.0 * 3600.0).all(|e| (e.2 as f64 - cap).abs() < 1e-3));
    // midday trains (5 an hour, 8,000 of room against 2,000 arriving) clear it within the hour
    assert!(at(10.25) > 0.0 && at(11.0) < 1.0, "{} {}", at(10.25), at(11.0));
    // nobody is lost: in = on + turned away (the day ends with nobody waiting): the peak's
    // 16,000 too many less the 3,400 still waiting at its end, who board at midday
    assert!((ra.gave_up - (16000.0 - (WAITING_CAP - cap))).abs() < 1000.0, "{}", ra.gave_up);
    assert!((ra.arrived - ra.boarded - ra.gave_up).abs() < 1e-6 * ra.arrived);
    // riders left behind and long waits are reported for the peak's hours
    assert!(ra.left_h[8] > 1000.0);
    assert!(ra.wait_h[9] > 900.0, "{}", ra.wait_h[9]);
    // once it has cleared, the longest wait is a headway (5 trains an hour at midday)
    assert!(ra.wait_h[15] < 730.0, "{}", ra.wait_h[15]);
    // the day's boardings in `today` leave out those turned away
    let v = b.today(&w.svc, 2.0 * DAY + 23.999 * 3600.0);
    let boarded: f64 = b.accounts[0][l as usize].as_ref().unwrap().runs.iter().map(|r| r.boarded).sum();
    assert!((v[3] - boarded).abs() < 0.01 * boarded, "{} {boarded}", v[3]);
}

/// T-1067, T-1255: riders whom full trains leave behind for GIVE_UP_S go another way, as do those
/// who find the platform holding as many as the trains of the next WAIT_FOR_ROOM_S take (at most
/// WAITING_CAP), so nobody waits past GIVE_UP_S.
#[test]
fn riders_left_behind_too_long_go_another_way() {
    // 2 trains an hour of 1,600 against 4,000 an hour from 06:00 to 10:00, then 1 an hour
    // against 2,400 to 16:00
    let (w, l) = setup(2.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [4000.0, 2400.0, 500.0, 500.0, 50.0]));
    b.ensure(&w.net, &w.svc);
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let m = ra.deps.len();
    // the queue grows 400 a train through the morning and fills the platform at 11:00
    let most = (0..m).map(|j| ra.q_after[j * 3]).fold(0.0f32, f32::max) as f64;
    assert!(most <= WAITING_CAP - 1600.0 + 1e-3, "{most}");
    let gave_by_noon: f64 = (0..m).filter(|&j| ra.order_t[j] < 12.5 * 3600.0).map(|j| ra.gone[ra.order[j] as usize * 3] as f64).sum();
    let gave_after: f64 = (0..m).filter(|&j| ra.order_t[j] > 12.5 * 3600.0).map(|j| ra.gone[ra.order[j] as usize * 3] as f64).sum();
    assert!(gave_by_noon > 0.0 && gave_after > 0.0, "{gave_by_noon} {gave_after}");
    // whoever is left waits at most GIVE_UP_S past their first train (T-1255: on a train every 30
    // minutes or rarer, a rider joins only for a train with room)
    let longest = ra.wait_h[6..16].iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
    assert!(longest > 1800.0 && longest <= GIVE_UP_S + 1.0, "{longest}");
    assert!(ra.gave_up > 1000.0, "{}", ra.gave_up);
    assert!((ra.arrived - ra.boarded - ra.gave_up).abs() < 1e-6 * ra.arrived);
    let gone: f64 = ra.gone.iter().map(|&g| g as f64).sum();
    assert!((gone - ra.gave_up).abs() < 1e-3 * ra.gave_up);
}

/// T-1255: riders come to the platform (`set_arrivals_day`, here three times the boardings) and
/// wait only while the riders ahead get on within WAIT_FOR_ROOM_S; the rest go another way and count
/// that wait in the line's crowding.
#[test]
fn riders_come_and_wait_only_for_a_train_with_room_soon() {
    // 10 trains an hour of 1,600: 16,000 an hour against 8,000 boarding and 24,000 coming
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    let per_hour = [8000.0, 1000.0, 1000.0, 1000.0, 50.0];
    b.set_riders(&riders(l, per_hour));
    b.ensure(&w.net, &w.svc);
    let calm = b.line_periods(&w.svc, l as usize, 0);
    let mut come = vec![l as f32, 3.0];
    for p in 0..PERIODS {
        let (h0, h1) = PERIOD_HOURS[p];
        let x = per_hour[p] * (h1 - h0) as f32 * if p == 0 { 3.0 } else { 1.0 };
        come.extend([x, 0.0, 0.0, x, 0.0, 0.0]);
    }
    b.set_arrivals_day(&come, 0);
    b.ensure(&w.net, &w.svc);
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let m = ra.deps.len();
    // every peak train leaves full and the platform holds at most the next 30 minutes' trains
    let peak: Vec<usize> = (0..m).filter(|&j| ra.order_t[j] > 7.0 * 3600.0 && ra.order_t[j] < 9.5 * 3600.0).map(|j| ra.order[j] as usize).collect();
    assert!(peak.iter().all(|&i| (ra.load[i * 3] as f64 - 1600.0).abs() < 1e-3));
    let most = (0..m).map(|i| ra.q_after[i * 3]).fold(0.0f32, f32::max) as f64;
    assert!(most <= 5.0 * 1600.0 + 1e-3, "{most}");
    // so nobody waits much beyond 30 minutes for a train with room (the head of the queue: a
    // headway more)
    let longest = ra.wait_h[6..11].iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
    assert!(longest > 1200.0 && longest <= WAIT_FOR_ROOM_S + 2.0 * 360.0 + 1.0, "{longest}");
    assert!(ra.gave_up > 10000.0 && (ra.arrived - ra.boarded - ra.gave_up).abs() < 1e-6 * ra.arrived, "{}", ra.gave_up);
    // the riders turned away count the wait in the period's crowding, and are counted
    let v = b.line_periods(&w.svc, l as usize, 0);
    let p = |v: &[f64], p: usize, i: usize| v[p * PERIOD_STRIDE + i];
    assert!(p(&v, 0, 8) > 10000.0 && p(&calm, 0, 8) == 0.0, "{} {}", p(&v, 0, 8), p(&calm, 0, 8));
    assert!(p(&v, 0, 4) / p(&v, 0, 2) > 10.0 * 60.0 && p(&calm, 0, 4) / p(&calm, 0, 2) < 60.0, "{} {}", p(&v, 0, 4) / p(&v, 0, 2), p(&calm, 0, 4) / p(&calm, 0, 2));
}

#[test]
fn the_line_summary_names_the_full_trains_and_when() {
    // T-230: 20,000 an hour against 16,000 in the morning peak only
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
    b.ensure(&w.net, &w.svc);
    let s = b.line_summary(&w.svc, l as usize);
    assert!((s[0] - 1.0).abs() < 1e-6, "fullest {}", s[0]);
    assert!(s[1] >= 60.0, "{} full trains", s[1]); // 10 an hour each way through the peak
    assert!(s[2] > 10000.0 && s[3] > 1800.0);
    // one span: from the peak's start until midday trains have cleared the queue (by 13:30)
    let spans: Vec<(f64, f64)> = (0..s[4] as usize).map(|k| (s[5 + 2 * k], s[6 + 2 * k])).collect();
    assert_eq!(spans.len(), 1, "{spans:?}");
    assert!(spans[0].0 >= 6.0 * 3600.0 && spans[0].0 < 7.0 * 3600.0 && spans[0].1 > 10.0 * 3600.0 && spans[0].1 < 13.5 * 3600.0, "{spans:?}");
    // a line with room: no full trains, nobody left
    let (w2, l2) = setup(10.0);
    let mut b2 = Boarding::default();
    b2.set_riders(&riders(l2, [500.0, 200.0, 500.0, 200.0, 50.0]));
    b2.ensure(&w2.net, &w2.svc);
    let s2 = b2.line_summary(&w2.svc, l2 as usize);
    assert!(s2[0] < 0.1 && s2[1] == 0.0 && s2[2] < 1e-3 && s2[4] == 0.0, "{s2:?}");
    // T-477, T-491: by period, the full trains, riders left behind and their wait fall in the
    // morning peak, spilling into midday while the queue clears; none at night
    let v = b.line_periods(&w.svc, l as usize, 0);
    let p = |p: usize, i: usize| v[p * PERIOD_STRIDE + i];
    assert!(p(0, 1) >= 60.0 && p(0, 3) > 10000.0, "{v:?}");
    assert!(p(4, 1) == 0.0 && p(4, 0) > 0.0 && p(4, 4) == 0.0, "{v:?}");
    // the average morning rider waits minutes for a train with room; everyone rides
    assert!(p(0, 4) / p(0, 2) > 300.0 && p(0, 5) > 0.0, "{v:?}");
    assert!(p(1, 4) > 0.0, "{v:?}");
    let v2 = b2.line_periods(&w2.svc, l2 as usize, 0);
    assert!(v2[..PERIODS * PERIOD_STRIDE].chunks(PERIOD_STRIDE).all(|c| c[1] == 0.0 && c[3] < 1e-3 && c[4] < 1e-3 && c[5] > 0.0), "{v2:?}");
    // T-818: every rider rides some km, so the km-weighted sums are positive and no larger than
    // the plain ones times the line's length; the stops' waits add up to the period's
    let len_km = (w.svc.lines[l as usize].runs[0].stop_s.last().unwrap() - w.svc.lines[l as usize].runs[0].stop_s[0]) / 1000.0;
    assert!(p(0, 6) > 0.0 && p(0, 6) <= p(0, 4) * len_km + 1e-6 && p(0, 7) > 0.0 && p(0, 7) <= p(0, 2) * len_km + 1e-6, "{v:?}");
    let n = w.svc.lines[l as usize].stops.len();
    let at: f64 = v[PERIODS * PERIOD_STRIDE..PERIODS * PERIOD_STRIDE + n].iter().sum();
    assert!(v.len() == PERIODS * (PERIOD_STRIDE + 2 * n) && (at - p(0, 4)).abs() < 1e-6 * p(0, 4).max(1.0), "{at} {}", p(0, 4));
    // T-472: at 08:00 the trains in service are about the line's peak trains, every one full
    let v = b.past_day(&w.svc, 2.0 * DAY + 8.0 * 3600.0);
    // T-491: the period now's wait, arrivals and ride end the block, then T-818's two (T-1255: the
    // wait and arrivals at the platforms where trains come full, here every one)
    let (wait, arrived) = (v[v.len() - 5], v[v.len() - 4]);
    assert!((wait - p(0, 9)).abs() < 1e-6 && (arrived - p(0, 10)).abs() < 1e-6 && wait > 0.0);
    let (on, packed, trains) = (v[v.len() - 8], v[v.len() - 7], v[v.len() - 6]);
    assert!(on > 0.0 && (on - w.svc.lines[l as usize].trains[HIGH] as f64).abs() <= 2.0 && packed == on, "{on} {packed}");
    // T-481: the full trains are counted out of every trip in the 24 hours
    assert!(trains >= v[4] && trains > 100.0, "{trains} trips, {} full", v[4]);
}

#[test]
fn a_queue_that_lasts_the_night_carries_round_the_day() {
    // the night (2.5 trains an hour, 4,000 of room) gets 6,000 an hour: still waiting at 03:00
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [2000.0, 2000.0, 2000.0, 2000.0, 6000.0]));
    b.ensure(&w.net, &w.svc);
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let quick = simulate(&w.net, &w.svc, l as usize, 0, b.riders[0][l as usize].as_ref().unwrap(), &b.peak, false);
    assert!(quick.leftover > 1000.0, "{}", quick.leftover);
    // the kept account is the warm one: the first train after 03:00 already finds a queue
    let i = ra.deps.iter().position(|&d| d >= START_HOUR * 3600.0).unwrap();
    assert!(ra.q_after[i * 3] > 1000.0, "{}", ra.q_after[i * 3]);
    let j = quick.deps.iter().position(|&d| d >= START_HOUR * 3600.0).unwrap();
    // (T-1255: a train's worth more, the platform holding only the next 30 minutes' trains)
    assert!(ra.q_after[i * 3] > quick.q_after[j * 3] + 500.0, "{} {}", ra.q_after[i * 3], quick.q_after[j * 3]);
    // and every departure of the day is in it once, in time order
    assert_eq!(ra.deps.len(), quick.deps.len());
    assert!(ra.deps.windows(2).all(|p| p[0] < p[1]));
}

#[test]
fn a_change_mid_peak_keeps_the_account_on_the_drawn_trains() {
    // T-206: 10 an hour cut to 4 at 08:20; trains already running keep their loads, and
    // every train drawn around the change has its own account
    let (mut w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [3000.0, 2000.0, 2000.0, 2000.0, 100.0]));
    let now = 2.0 * DAY + 8.0 * 3600.0 + 1200.0;
    b.ensure(&w.net, &w.svc);
    let before_trips = w.svc.trips(&w.net, now - 600.0, now);
    let loads_before: Vec<Option<Vec<f32>>> = before_trips.iter().map(|t| b.trip_account(t.line, t.run as usize, t.dep)).collect();
    let before = w.svc.before_edit(&w.net, now);
    w.apply(crate::track::world::Op::Schedule { id: l, tph: [4.0, 5.0, 2.5] }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    b.ensure(&w.net, &w.svc);
    for epoch in [now - 1200.0, now + 2400.0] {
        for t in w.svc.trips(&w.net, epoch, epoch + 3600.0) {
            assert!(b.trip_account(t.line, t.run as usize, t.dep).is_some(), "{t:?} has no account");
        }
    }
    // the trains that left before the change were already full or not: the same riders
    for (t, was) in before_trips.iter().zip(&loads_before) {
        if t.dep < now - 900.0 {
            assert_eq!(&b.trip_account(t.line, t.run as usize, t.dep), was, "{t:?}");
        }
    }
    // fewer trains: riders now wait, on the drawn departures
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    let late = ra.deps.iter().enumerate().filter(|(_, &d)| d > (now + 600.0).rem_euclid(DAY) && d < (now + 3000.0).rem_euclid(DAY)).count();
    assert!(late <= 5, "{late} departures in 40 minutes at 4 an hour");
}

/// T-309: a London line and a New York line in one Mercator world, on the registry's time
/// zones (New York's and Toronto's; London is in none, so its longitude's UTC+0). The same
/// schedule and riders run on each city's own hours.
#[test]
fn a_london_line_runs_on_londons_hours() {
    use crate::track::frame::FrameKind;
    use crate::track::localtime::TimeZones;
    use crate::track::net::Network;
    use crate::track::world::TrackWorld;
    let mut net = Network::new_in(FrameKind::Merc, -73.985, 40.758, true);
    net.time_zones = TimeZones::from_table(&[-5.0, 120_000.0, -73.985, 40.758, -5.0, -79.3832, 43.6534, -5.0]);
    let mut w = TrackWorld::new(net);
    let city = |w: &mut TrackWorld, lon: f64, lat: f64| {
        let k = 1.0 / (lat * std::f64::consts::PI / 180.0).cos();
        let (x, y) = w.net.frame().from_lonlat(lon, lat);
        let (x, y) = (quant(x), quant(y));
        let s: Vec<u32> = (0..3).map(|i| node(w, x + i as f64 * 3000.0 * k, y, 0)).collect();
        edge(w, s[0], s[1], &[], 0).unwrap();
        edge(w, s[1], s[2], &[], 0).unwrap();
        for &n in &s {
            station(w, n, 200);
        }
        line(w, &s, [10.0, 5.0, 2.5])
    };
    let nyc = city(&mut w, -73.99, 40.75);
    let lon = city(&mut w, -0.12, 51.51);
    assert_eq!(w.svc.lines[nyc as usize].shift, 0.0);
    assert_eq!(w.svc.lines[lon as usize].shift, 5.0 * 3600.0);
    // service periods: the morning peak's ten an hour start at 06:00 local, 01:00 game time
    let day = 3.0 * DAY;
    let levels = |l: u32, h: f64| -> Vec<u8> {
        // the hour's last 10 minutes left out: the next period leaves half a run early (T-835)
        let mut v: Vec<u8> = w.svc.trips(&w.net, day + h * 3600.0, day + (h + 1.0) * 3600.0).iter().filter(|t| t.line == l && t.draw >= day + h * 3600.0 && t.draw < day + (h + 1.0) * 3600.0 - 600.0).map(|t| t.level).collect();
        v.dedup();
        v
    };
    assert_eq!(levels(lon, 0.0), vec![LOW as u8]);
    assert_eq!(levels(lon, 1.0), vec![HIGH as u8]);
    assert_eq!(levels(nyc, 1.0), vec![LOW as u8]);
    assert_eq!(levels(nyc, 6.0), vec![HIGH as u8]);
    // every window agrees on the departures, across the change of period too
    let deps = |a: f64, b: f64| {
        let mut v = vec![];
        w.svc.run_deps(&w.net, lon as usize, 1, day + a * 3600.0, day + b * 3600.0, &mut v);
        v.iter().map(|t| t.dep).collect::<Vec<f64>>()
    };
    let mut two = deps(0.5, 1.0);
    two.extend(deps(1.0, 1.7));
    assert_eq!(deps(0.5, 1.7), two);
    // the boarding account: London at 03:00 game time is its 08:00, New York's at 08:00
    let mut b = Boarding::default();
    let mut wk = riders(nyc, [500.0, 200.0, 500.0, 200.0, 50.0]);
    wk.extend_from_slice(&riders(lon, [500.0, 200.0, 500.0, 200.0, 50.0])[PERIODS..]);
    b.set_riders_day(&wk, 0);
    let mut we = riders(nyc, [100.0, 300.0, 200.0, 200.0, 10.0]);
    we.extend_from_slice(&riders(lon, [100.0, 300.0, 200.0, 200.0, 10.0])[PERIODS..]);
    b.set_riders_day(&we, 1);
    let arriving = |h: &[f32], l: u32| {
        let (nt, nl, nq) = (h[0] as usize, h[1] as usize, h[2] as usize);
        let q0 = 5 + 2 * nt + nl;
        (0..nq).map(|k| &h[q0 + k * QUEUE_STRIDE..q0 + (k + 1) * QUEUE_STRIDE]).filter(|q| q[1] as u32 == l).map(|q| q[7]).sum::<f32>()
    };
    let at = |b: &mut Boarding, h: f64| b.hour(&w.net, &w.svc, day + h * 3600.0);
    let (h3, h8) = (at(&mut b, 3.0), at(&mut b, 8.0));
    // 500 an hour at each end in the morning peak, 50 at night
    assert!((arriving(&h3, lon) - 1000.0).abs() < 1.0, "{}", arriving(&h3, lon));
    assert!((arriving(&h3, nyc) - 100.0).abs() < 1.0, "{}", arriving(&h3, nyc));
    assert!((arriving(&h8, nyc) - 1000.0).abs() < 1.0, "{}", arriving(&h8, nyc));
    // every London trip of the hour has its load in the account
    let trips = w.svc.trips(&w.net, day + 3.0 * 3600.0, day + 4.0 * 3600.0);
    assert!(trips.iter().filter(|t| t.line == lon).all(|t| b.trip_account(t.line, t.run as usize, t.dep).is_some()));
    // weekends by the local day: game day 6 (Saturday) starts five hours early in London,
    // at 19:00 on Friday game time (day 5), where New York is still on Friday evening
    let fri = 5.0 * DAY + 20.0 * 3600.0 - day;
    let h = at(&mut b, fri / 3600.0);
    assert!((arriving(&h, lon) - 20.0).abs() < 0.5, "{}", arriving(&h, lon)); // Saturday 01:00
    assert!((arriving(&h, nyc) - 400.0).abs() < 1.0, "{}", arriving(&h, nyc)); // Friday 20:00
    // fares follow: today's boardings count from London's midnight
    let v = b.today(&w.svc, day + 3.0 * 3600.0);
    assert_eq!(v[0], 2.0);
}

/// T-309: the real New York save (trainworld/T-007-real-nyc.save, Anita's) runs exactly as
/// before local time with the registry's time zones: every line is on New York's time, and a
/// week of drawn trips, every hour's boarding slice, today's counts and the line summaries are
/// bit-identical to the network without a time zone table. Runs on request, on the save's
/// track bytes (gunzipped, after its JSON header): `T309_TRACK=<file> cargo test --manifest-path
/// sim/Cargo.toml --release t309 -- --ignored --nocapture`.
#[test]
#[ignore]
fn t309_real_save_runs_as_before() {
    use crate::track::localtime::TimeZones;
    use crate::track::world::TrackWorld;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let bytes = std::fs::read(std::env::var("T309_TRACK").expect("T309_TRACK: the save's track bytes")).unwrap();
    // the registry's table, as app/src/game/localTime.ts builds it
    let reg = std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap();
    let num = |s: &str| s.trim().trim_end_matches(',').trim().parse::<f64>().unwrap();
    let clock = num(reg.split("\"clockUtcOffset\":").nth(1).unwrap().split('\n').next().unwrap());
    let mut table = vec![clock, 120_000.0];
    for city in reg.split("\"center\": [").skip(1) {
        let mut it = city.split(']').next().unwrap().split(',');
        let (lon, lat) = (num(it.next().unwrap()), num(it.next().unwrap()));
        let off = num(city.split("\"utcOffset\":").nth(1).unwrap().split('\n').next().unwrap());
        table.extend([lon, lat, off]);
    }
    println!("time zone table: {} zones, clock UTC{clock}", (table.len() - 2) / 3);
    let a = TrackWorld::new(crate::track::save::decode(&bytes).unwrap());
    let mut net = crate::track::save::decode(&bytes).unwrap();
    net.time_zones = TimeZones::from_table(&table);
    let b = TrackWorld::new(net);
    let running: Vec<usize> = (0..b.svc.lines.len()).filter(|&l| b.svc.lines[l].running).collect();
    assert!(running.iter().all(|&l| b.svc.lines[l].shift == 0.0));
    assert!(b.net.time_zones.set);
    println!("{} running lines, all on New York's time", running.len());
    let day = 2.0 * DAY;
    let mut trips = 0;
    for h in 0..(7 * 24) {
        let (t0, t1) = (day + h as f64 * 3600.0, day + (h + 1) as f64 * 3600.0);
        let (x, y) = (a.svc.trips(&a.net, t0, t1), b.svc.trips(&b.net, t0, t1));
        assert_eq!(x, y, "hour {h}");
        trips += x.len();
    }
    // made-up riders on every running line (the account's arithmetic is what is compared)
    let packed = |scale: f32| {
        let mut v = vec![1.4, 1.15, 1.4, 1.3, 1.5];
        for &l in &running {
            let n = b.svc.lines[l].stops.len();
            v.extend([l as f32, n as f32]);
            for p in 0..PERIODS {
                let x = scale * (p as f32 + 1.0) * 37.0;
                v.extend((0..2 * n).map(|k| if k % n + 1 < n { x } else { 0.0 }));
                v.extend((0..2 * n).map(|k| if k % n > 0 { x } else { 0.0 }));
            }
        }
        v
    };
    let (mut ba, mut bb) = (Boarding::default(), Boarding::default());
    for bo in [&mut ba, &mut bb] {
        bo.set_riders_day(&packed(1.0), 0);
        bo.set_riders_day(&packed(0.6), 1);
    }
    for h in 0..(7 * 24) {
        let t = day + h as f64 * 3600.0;
        let (x, y) = (ba.hour(&a.net, &a.svc, t), bb.hour(&b.net, &b.svc, t));
        assert!(x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "hour {h}");
        let (x, y) = (ba.today(&a.svc, t + 1234.5), bb.today(&b.svc, t + 1234.5));
        assert!(x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "today {h}");
    }
    for &l in &running {
        for d in 0..DAY_KINDS {
            assert_eq!(ba.line_summary_day(&a.svc, l, d), bb.line_summary_day(&b.svc, l, d));
        }
    }
    println!("a week: {trips} trips, every hourly boarding slice and today's counts bit-identical");
}

/// T-554: riders changing from a train reach their next platform a transfer walk after it
/// leaves, the hour's arrivals stay the same, and the account itself does not move.
#[test]
fn riders_changing_lines_arrive_with_their_train() {
    // line 1 A - B - C every 6 minutes at the peak; line 2 E - F - G every 15, ending at G,
    // 60 m from B
    let (mut w, l1) = setup(10.0);
    let e = node(&mut w, 3000.0, 6060.0, 0);
    let f = node(&mut w, 3000.0, 3060.0, 0);
    let g = node(&mut w, 3000.0, 60.0, 0);
    edge(&mut w, e, f, &[], 0).unwrap();
    edge(&mut w, f, g, &[], 0).unwrap();
    for n in [e, f, g] {
        station(&mut w, n, 200);
    }
    let l2 = line(&mut w, &[e, f, g], [4.0, 2.0, 1.0]);
    let mut v = vec![1.0; PERIODS];
    let hours = |p: usize| (PERIOD_HOURS[p].1 - PERIOD_HOURS[p].0) as f32;
    v.extend([l1 as f32, 3.0]);
    for p in 0..PERIODS {
        let (x, y) = (300.0 * hours(p), 200.0 * hours(p));
        v.extend([x, y, 0.0, x, y, 0.0, 0.0, 0.0, x + y, 0.0, 0.0, x + y]);
    }
    v.extend([l2 as f32, 3.0]);
    for p in 0..PERIODS {
        let z = 200.0 * hours(p);
        v.extend([z, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * z, 0.0, 0.0, 0.0]);
    }
    let day_b: f32 = (0..PERIODS).map(|p| 200.0 * hours(p)).sum();
    let mut b = Boarding::default();
    b.set_riders(&v);
    let epoch = 3.0 * DAY + 8.0 * 3600.0;
    let before = b.hour(&w.net, &w.svc, epoch);
    // half of B's boarders each way come off line 2 at G
    b.set_transfers(&[l2 as f32, 0.0, 2.0, l1 as f32, 0.0, 1.0, day_b / 2.0, l2 as f32, 0.0, 2.0, l1 as f32, 1.0, 1.0, day_b / 2.0]);
    let after = b.hour(&w.net, &w.svc, epoch);
    let parts = |h: &[f32]| {
        let (nt, nl, nq, ne) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3] as usize);
        let q0 = 5 + 2 * nt + nl;
        let e0 = q0 + nq * QUEUE_STRIDE;
        (h[..q0].to_vec(), h[q0..e0].to_vec(), h[e0..e0 + 2 * ne].to_vec(), h[e0 + 2 * ne..e0 + 2 * ne + 2 * h[4] as usize].to_vec())
    };
    let (ta, qa, ea, pa) = parts(&before);
    let (tb, qb, eb, pb) = parts(&after);
    // T-1124: every pulse comes from G, a 65 s walk
    let (nq, ne, np) = (after[2] as usize, after[3] as usize, after[4] as usize);
    let s0 = 5 + 2 * after[0] as usize + after[1] as usize + nq * QUEUE_STRIDE + 2 * ne + 2 * np + ne;
    assert!(after[s0..s0 + 2 * np].chunks(2).all(|c| c[0] == g as f32 && (c[1] - 65.0).abs() < 0.01));
    // the trains' loads and every train leaving a platform are as before
    assert_eq!(ta[..4], tb[..4]);
    assert_eq!(ta[5..], tb[5..]);
    assert_eq!(ea, eb);
    assert!(pa.is_empty() && !pb.is_empty());
    let mut seen = 0;
    for (x, y) in qa.chunks(QUEUE_STRIDE).zip(qb.chunks(QUEUE_STRIDE)) {
        assert_eq!(x[..4], y[..4]);
        assert_eq!(x[5..10], y[5..10]); // left behind, waits, the hour's arrivals, events
        if x[1] as u32 != l1 || x[0] as u32 != w.svc.lines[l1 as usize].runs[0].nodes[w.svc.lines[l1 as usize].runs[0].stop_idx[1] as usize] {
            assert_eq!(x[4], y[4]);
            assert_eq!(y[11], 0.0);
            continue;
        }
        seen += 1;
        // at B: half arrive steadily, half in four pulses an hour, one per line 2 train
        assert!((y[4] - x[4] / 2.0).abs() < 1e-6, "{} {}", x[4], y[4]);
        let p = &pb[y[10] as usize * 2..(y[10] + y[11]) as usize * 2];
        let hour: Vec<&[f32]> = p.chunks(2).filter(|c| c[0] >= 0.0 && c[0] < 3600.0).collect();
        assert_eq!(hour.len(), 4);
        let sum: f32 = hour.iter().map(|c| c[1]).sum();
        assert!((sum - x[7] / 2.0).abs() < 0.01, "{sum} {}", x[7]);
        // before the hour, only after the last train to leave before it
        let since = eb[y[8] as usize * 2];
        assert!(p.chunks(2).all(|c| c[0] > since && c[0] < 3600.0));
        // each a 65 s walk (60 m x 1.3 at 1.2 m/s) after a line 2 train reaches G
        let g_t: Vec<f64> = {
            let ra = &b.accounts[0][l2 as usize].as_ref().unwrap().runs[0];
            let m = ra.deps.len();
            ra.order_t[2 * m..3 * m].to_vec()
        };
        let h0 = epoch.rem_euclid(DAY);
        for c in &hour {
            let t = c[0] as f64 - 65.0 + h0;
            assert!(g_t.iter().any(|&x| (x - t).abs() < 0.01), "{}", c[0]);
        }
    }
    assert_eq!(seen, 2);
}

#[test]
fn the_hour_lines_up_with_the_drawn_trips() {
    let (w, l) = setup(10.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [20000.0, 2000.0, 2000.0, 2000.0, 100.0]));
    let epoch = 3.0 * DAY + 8.0 * 3600.0;
    let out = b.hour(&w.net, &w.svc, epoch);
    let trips = w.svc.trips(&w.net, epoch, epoch + 3600.0);
    let (nt, nl, nq, ne, np) = (out[0] as usize, out[1] as usize, out[2] as usize, out[3] as usize, out[4] as usize);
    assert_eq!(nt, trips.len());
    assert_eq!(np, 0); // no transfers set
    let tail = 5 + 2 * nt + nl + nq * QUEUE_STRIDE + ne * 2;
    let nf = out[tail + ne] as usize;
    assert_eq!(out.len(), tail + ne + 1 + nf * 4);
    assert!(nf >= 1, "riders left behind at every train: a stretch at capacity (T-1125)");
    let shares = &out[tail + ne + 1 + nf * 2..tail + ne + 1 + nf * 3];
    assert!(shares.iter().all(|&s| s > 0.0 && s <= 1.0), "{shares:?} (T-1152)");
    let waits = &out[tail + ne + 1 + nf * 3..];
    assert!(waits.iter().all(|&s| s >= FULL_WAIT_MIN_S), "{waits:?} (T-1160)");
    // T-1123: each train stops before it leaves, never before the train ahead left
    let (ev, arr) = (&out[tail - ne * 2..tail], &out[tail..tail + ne]);
    for e in 0..ne {
        assert!(arr[e] <= ev[e * 2], "{e}: {} {}", arr[e], ev[e * 2]);
    }
    let offs: Vec<f32> = out[5..5 + 2 * nt].chunks(2).map(|c| c[0]).collect();
    assert!(offs.iter().all(|&o| o >= 0.0));
    let run_s = w.svc.lines[l as usize].runs[0].prof[HIGH].duration as f32;
    assert!(out[5..5 + 2 * nt].chunks(2).all(|c| c[1] == run_s), "{:?} {run_s}", &out[5..9]);
    let out = &out[nt + 1..]; // the rest as if one number per trip and a 4-number header
    let loads = &out[4 + nt..4 + nt + nl];
    for (t, o) in trips.iter().zip(offs.iter().copied()) {
        let own = b.trip_account(t.line, t.run as usize, t.dep).unwrap();
        let o = o as usize;
        if t.run == 0 {
            assert_eq!(&loads[o..o + 2], &own[0..2]);
        } else {
            assert_eq!(loads[o], own[1]);
            assert_eq!(loads[o + 1], own[0]);
        }
    }
    // platforms: A and C have riders (the line's two boarding stops), B none
    assert_eq!(nq, 2);
    let qs = &out[4 + nt + nl..4 + nt + nl + nq * QUEUE_STRIDE];
    let ev = &out[4 + nt + nl + nq * QUEUE_STRIDE..];
    for q in qs.chunks(QUEUE_STRIDE) {
        let (first, count) = (q[8] as usize, q[9] as usize);
        assert!(count >= 11, "{count}"); // 10 an hour, one before and one after
        let e = &ev[first * 2..(first + count) * 2];
        assert!(e[0] < 0.0 && e[(count - 1) * 2] >= 3600.0);
        assert!(e.chunks(2).all(|p| p[1] > 0.0)); // people left behind at every train
        assert!(q[5] > 0.0);
        assert!((q[7] - 20000.0 * 1.0).abs() < 1.0); // flat shape: 20,000 this hour
    }
}

#[test]
fn a_new_lines_platforms_fill_from_when_its_trains_start() {
    // T-680: trains set on a line at 08:20; its platforms are empty then and fill at the
    // arrival rate, not with everyone who would have arrived in the six hours before
    let (mut w, l) = setup(0.0);
    let mut b = Boarding::default();
    b.set_riders(&riders(l, [3000.0, 2000.0, 2000.0, 2000.0, 100.0]));
    let now = 2.0 * DAY + 8.0 * 3600.0 + 1200.0;
    let before = w.svc.before_edit(&w.net, now);
    w.apply(crate::track::world::Op::Schedule { id: l, tph: [10.0, 5.0, 2.5] }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    let epoch = now - 1200.0;
    let out = b.hour(&w.net, &w.svc, epoch);
    let (nt, nl, nq) = (out[0] as usize, out[1] as usize, out[2] as usize);
    let q0 = 5 + 2 * nt + nl;
    let ev = &out[q0 + nq * QUEUE_STRIDE..];
    assert_eq!(nq, 2);
    for q in out[q0..q0 + nq * QUEUE_STRIDE].chunks(QUEUE_STRIDE) {
        let (rate, first, count) = (q[4], q[8] as usize, q[9] as usize);
        let e = &ev[first * 2..(first + count) * 2];
        // as game/boarding.ts's crowdAt: the last train to leave at or before t, plus arrivals
        let crowd = |t: f32| {
            let j = e.chunks(2).rposition(|p| p[0] <= t).unwrap_or(0);
            e[j * 2 + 1] + (t - e[j * 2]).max(0.0) * rate
        };
        let mut t = 1200.0f32;
        while t <= 3600.0 {
            assert!(crowd(t) <= rate * (t - 1200.0) + 1.0, "{} waiting at {t} s of the hour, {} a second", crowd(t), rate);
            t += 30.0;
        }
        assert!(crowd(1199.0) < 1.0);
        // riders arriving this hour: from 08:20 only
        assert!((q[7] - 3000.0 * 40.0 / 60.0).abs() < 1.0, "{}", q[7]);
    }
    // the line's riders today: those since 08:20 at most
    let today = b.today(&w.svc, now + 1800.0);
    assert!(today[3] <= 2.0 * 3000.0 * 0.5 + 1.0, "{} riders today", today[3]);
    assert!(today[3] > 0.0);
}

#[test]
fn a_long_lines_opening_day_runs_as_full_as_the_next() {
    // T-848: 11 stops 10 km apart, opened at 07:20; each stop's riders ride one stop, half a
    // train's room a train at 4 an hour, so on a steady day nobody is left behind
    let mut w = world();
    let n = 11;
    let s: Vec<u32> = (0..n).map(|i| node(&mut w, i as f64 * 10_000.0, 0.0, 0)).collect();
    for i in 1..n {
        edge(&mut w, s[i - 1], s[i], &[], 0).unwrap();
    }
    for &x in &s {
        station(&mut w, x, 200);
    }
    let l = line(&mut w, &s, [0.0, 0.0, 0.0]);
    let rate = 0.5 * 4.0 * (CRUSH_PER_CAR * w.svc.lines[l as usize].cars.max(1) as f64) as f32;
    let mut v = vec![1.0; PERIODS];
    v.extend([l as f32, n as f32]);
    for p in 0..PERIODS {
        let x = rate * (PERIOD_HOURS[p].1 - PERIOD_HOURS[p].0) as f32;
        let board: Vec<f32> = (0..2 * n).map(|k| if k % n + 1 < n { x } else { 0.0 }).collect();
        let alight: Vec<f32> = (0..2 * n).map(|k| if k % n > 0 { x } else { 0.0 }).collect();
        v.extend(board);
        v.extend(alight);
    }
    let mut b = Boarding::default();
    b.set_riders(&v);
    let now = 2.0 * DAY + 7.0 * 3600.0 + 1200.0;
    let before = w.svc.before_edit(&w.net, now);
    w.apply(crate::track::world::Op::Schedule { id: l, tph: [4.0, 4.0, 4.0] }).unwrap();
    w.svc.note_changes(&w.net, &before, now);
    b.ensure(&w.net, &w.svc);
    let opening = b.line_periods(&w.svc, l as usize, 0);
    // the stop before the far end opens with its first train, about 50 minutes on
    let ra = &b.accounts[0][l as usize].as_ref().unwrap().runs[0];
    assert!(ra.stop_open[n - 2] > now + 2400.0 && ra.stop_open[n - 2] - ra.stop_open[0] > 2400.0, "{:?}", ra.stop_open);
    let today = b.today(&w.svc, now + 2.0 * 3600.0);
    let later = now + DAY + 3600.0;
    w.svc.prune_eras(&w.net, later);
    b.ensure(&w.net, &w.svc);
    let steady = b.line_periods(&w.svc, l as usize, 0);
    let row = |v: &[f64], p: usize| v[p * PERIOD_STRIDE..p * PERIOD_STRIDE + 5].to_vec();
    for p in 0..PERIODS {
        let (o, s) = (row(&opening, p), row(&steady, p));
        assert!(s[3] < 1.0, "steady period {p}: {s:?}");
        assert!(o[3] < 1.0 && o[4] < 60.0, "opening period {p} leaves riders behind: {o:?}");
        // the same riders; the opening day has no trains in the hours before 07:20
        assert_eq!(o[2], s[2]);
        assert!(o[0] <= s[0]);
    }
    assert!(row(&opening, 0)[0] < row(&steady, 0)[0]);
    // the first two hours: no one left behind, and riders counted only from each stop's opening
    assert!(today[5] < 1.0, "{} left behind", today[5]);
    assert!(today[3] > 0.0 && today[3] < 2.0 * 3600.0 * rate as f64 * 2.0 * (n - 1) as f64 / 3600.0, "{} riders", today[3]);
}
