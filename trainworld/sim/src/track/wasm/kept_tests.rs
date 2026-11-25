//! T-4299: the model's kept money (`wasm/kept.rs`) against the same worked out from nothing and against the whole sums it
//! stands for, after each kind of edit that moves it.

use super::*;
use crate::track::upgrades;

const HOURS: [f64; 5] = [4.0, 6.0, 4.0, 4.0, 6.0];

fn same(a: &mut TrackApi, what: &str) {
    let d = a.kept_check(&HOURS);
    assert!(d.is_empty(), "{what}: {d}");
}

/// A line along y = `y` with a station every `step` m up to `km` km, constructed or not; the line and its stations.
fn rail(a: &mut TrackApi, km: usize, step: f64, y: f64, built: bool) -> (u32, Vec<u32>) {
    let n = (km as f64 * 1000.0 / step).round() as usize;
    let mut st: Vec<u32> = vec![];
    for i in 0..n {
        let from = if i == 0 { [0.0, 0.0, 0.0, y, -1.0] } else { [1.0, st[i] as f64, 0.0, 0.0, 0.0] };
        let ends: Vec<f64> = from.iter().chain([0.0, 0.0, (i + 1) as f64 * step, y, -1.0].iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        if i == 0 {
            let ni = a.nodes_info();
            let first = ni.chunks_exact(10).find(|c| c[1].abs() < 1.0 && (c[2] - y).abs() < 1.0).unwrap()[0] as u32;
            st.push(first);
        }
        st.push(a.route_end());
    }
    for (i, &s) in st.iter().enumerate() {
        assert_eq!(a.add_station(1, s, 0.0, 0.0, 120, format!("S{y}-{i}")), 0, "{}", a.issues());
    }
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &st, 6, 4, 2, 30.0, 180.0, 0, 0xd7263d, format!("L{y}")), 0, "{}", a.issues());
    if built {
        assert_eq!(a.construct_line(l), 0, "{}", a.issues());
    }
    (l, st)
}

/// Every edit kind that moves the money, each followed by the kept money against the same from nothing.
#[test]
fn kept_money_is_what_a_whole_rebuild_gives_after_each_edit() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    same(&mut a, "a new model");
    let (la, sa) = rail(&mut a, 10, 2500.0, 0.0, true);
    same(&mut a, "a constructed line");
    let (lb, sb) = rail(&mut a, 6, 2000.0, 3000.0, false);
    same(&mut a, "a planned line");
    // a second line on the first's track: the shares split by trains a day
    let lc = a.new_line_id();
    assert_eq!(a.set_line_with_trains(lc, &sa[1..4], 4, 2, 1, 30.0, 180.0, 0, 0x2d7dd2, "C".into()), 0, "{}", a.issues());
    same(&mut a, "a line sharing track");
    assert_eq!(a.set_station_upgrade(&[sa[1]], upgrades::WAITING_ROOM, true), 0, "{}", a.issues());
    same(&mut a, "an upgrade");
    let _ = a.link_stations(sa[0], sb[0]);
    same(&mut a, "a passage asked for");
    let _ = a.add_building(&[-60.0, -40.0, 60.0, -40.0, 60.0, 40.0, -60.0, 40.0], "Hall".into());
    same(&mut a, "a building");
    let bus = a.new_line_id();
    assert_eq!(a.set_bus_line(bus, &[5000.0, 6100.0, 1.0, 5000.0, 100.0, 1.0], 480, 600, 1200, 30.0, 120.0, 0x2d7dd2, "B1".into()), 0, "{}", a.issues());
    assert_eq!(a.construct_line(bus), 0, "{}", a.issues());
    same(&mut a, "a constructed bus line");
    assert_eq!(a.set_line_one_way(lc, true), 0, "{}", a.issues());
    same(&mut a, "one way");
    assert_eq!(a.set_line_cars(la, 4), 0, "{}", a.issues());
    same(&mut a, "cars");
    // a weekend by headway reads the track limits
    assert_eq!(a.set_line_weekend_headways(la, 4, 3, 2, 900, 1200, 1800), 0, "{}", a.issues());
    same(&mut a, "a weekend by headway");
    let e = (0..a.w.net.edge_count() as u32).find(|&e| a.w.net.edge_ok(e) && !a.w.net.edge_built[e as usize]).unwrap();
    assert_eq!(a.delete_edge(e), 0, "{}", a.issues());
    same(&mut a, "planned track deleted");
    assert_eq!(a.undo(), 0);
    same(&mut a, "undone");
    assert_eq!(a.redo(), 0);
    same(&mut a, "redone");
    assert_eq!(a.construct_line(lb), 0, "{}", a.issues());
    same(&mut a, "the planned line constructed");
    assert_eq!(a.remove_station(sa[2]), 0, "{}", a.issues());
    same(&mut a, "a station removed");
    assert_eq!(a.undo(), 0);
    same(&mut a, "its removal undone");
    a.set_line_run_factors(&[la as f64, 0.5]);
    same(&mut a, "a line's prices");
    let saved = a.save();
    assert!(a.load(&saved));
    same(&mut a, "loaded");
    assert_eq!(a.set_line_cars(lb, 2), 0, "{}", a.issues());
    same(&mut a, "an edit after the load");
}

/// T-4299 review25: a line's price factor set after a state's first read (the clock worker sets them there, a load clears
/// them) moves the running cost the cash is drained by at once, not at the next state.
#[test]
fn a_price_factor_moves_the_running_cost_before_the_next_state() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (la, _) = rail(&mut a, 10, 2500.0, 0.0, true);
    let saved = a.save();
    assert!(a.load(&saved));
    // the load's state: its first read, then the factors
    a.take_changed_lines();
    let at_one = a.running_cost_period(0);
    a.set_line_run_factors(&[la as f64, 0.5]);
    let now = a.running_cost_period(0);
    let whole = super::money::running_cost_with(&a.w, &a.run_factor, 0, a.w.net.period_slot()[0], f64::NAN);
    assert!(at_one > 0.0 && (now - at_one * 0.5).abs() < 1e-9 * at_one, "running {now} at factor 0.5, {at_one} at 1");
    assert_eq!(now.to_bits(), whole.to_bits());
    same(&mut a, "the factors set after the state's first read");
    // the same list again names nothing; back to 1 names the line again
    a.set_line_run_factors(&[la as f64, 0.5]);
    same(&mut a, "the same factors");
    a.set_line_run_factors(&[]);
    assert_eq!(a.running_cost_period(0).to_bits(), at_one.to_bits());
    same(&mut a, "the world price again");
}

/// The check catches a line's money moved behind the kept money's back, through the ordinary read and the state's full one.
#[test]
fn kept_check_sees_line_money_nothing_named() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    let (la, _) = rail(&mut a, 6, 2000.0, 0.0, true);
    same(&mut a, "built");
    a.run_factor = vec![1.0; la as usize + 1];
    a.run_factor[la as usize] = 0.25;
    let d = a.kept_check(&HOURS);
    assert!(d.contains("running") && d.contains("nothing naming it"), "{d}");
    same(&mut a, "mended by the full read");
}

/// The check catches a record changed behind the kept money's back (no edit naming it).
#[test]
fn kept_check_sees_a_record_no_edit_named() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e6);
    rail(&mut a, 6, 2000.0, 0.0, true);
    same(&mut a, "built");
    let e = (0..a.w.net.edge_count()).find(|&e| a.w.net.edge_alive[e] && a.w.net.edge_built[e]).unwrap();
    a.w.net.edge_built[e] = false;
    assert!(!a.kept_check(&HOURS).is_empty(), "an edge unbuilt behind the kept money's back");
}

