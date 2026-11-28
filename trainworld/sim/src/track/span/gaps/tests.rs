use super::*;
use crate::track::tests::{edge, headway_line, node, station, terminals_at_dead_ends, world};
use crate::track::wasm::TrackApi;
use crate::track::world::TrackWorld;

fn pat(hw: f64, t: f64) -> Pat {
    Pat { hw, t, ride: 0.0 }
}

/// Her case (11-27): every 20 minutes each, B 2 minutes after A, then 18 minutes to the next A: 8.2 minutes, not 5.
/// Evenly spaced is half the combined headway, factor 1; every 10 and every 5 at their best 1.875 minutes, not 1.67.
#[test]
fn the_wait_is_the_real_gaps() {
    let (f, w) = real_wait(&[pat(1200.0, 0.0), pat(1200.0, 120.0)], 4.0 * 3600.0);
    assert!((w - 492.0).abs() < 1e-9 && (f - 1.64).abs() < 1e-12, "{f} {w}");
    let (f, w) = real_wait(&[pat(600.0, 86400.0 + 337.0), pat(600.0, 86400.0 + 37.0)], 4.0 * 3600.0);
    assert!(factor_sent(f) == 1.0 && (w - 150.0).abs() < 1e-9, "{f} {w}");
    let (f, w) = real_wait(&[pat(300.0, 0.0), pat(600.0, 150.0)], 4.0 * 3600.0);
    assert!((w - 112.5).abs() < 1.0 && (f - 1.125).abs() < 0.005, "{f} {w}");
    assert_eq!(real_wait(&[pat(600.0, 5.0)], 3600.0).0, 1.0);
    assert_eq!(real_wait(&[], 3600.0), (1.0, 0.0));
}

/// Two Headway lines on one track and its five stops, every 10 minutes from 6 to 20.
fn two_on_one_track() -> (TrackWorld, u32, u32) {
    let mut w = world();
    let n: Vec<u32> = (0..=4).map(|k| node(&mut w, k as f64 * 1500.0, 0.0, 0)).collect();
    for k in 0..4 {
        edge(&mut w, n[k], n[k + 1], &[], 0).unwrap();
    }
    for &x in &n {
        station(&mut w, x, 200);
    }
    terminals_at_dead_ends(&mut w);
    let a = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
    let b = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
    (w, a, b)
}

fn nudge(w: &mut TrackWorld, l: u32, by: f64) {
    let op = w.nudge_op(l, 0, by, false).unwrap().unwrap();
    w.apply(op).unwrap();
}

/// The spacing puts two lines at one headway half a headway apart: each hop of each run in each period is one group
/// of both, evenly spaced, factor 1 and nothing sent. Their peak row moved 2 minutes (3 and 7 minutes apart) waits 2.9
/// minutes, not 2.5: past 5 s from what was sent; 15 s (0.4 s more) is not.
#[test]
fn a_nudge_moves_the_wait_and_only_past_5_s_counts() {
    let (mut w, a, b) = two_on_one_track();
    let gs = groups(&w.net, &w.svc);
    assert_eq!(gs.len(), 8 * PERIODS, "{gs:?}");
    for g in &gs {
        let mut lines: Vec<u32> = g.members.iter().map(|m| m.line).collect();
        lines.sort_unstable();
        lines.dedup();
        assert_eq!(lines, vec![a, b]);
        assert_eq!(factor_sent(g.factor), 1.0, "{g:?}");
    }
    let peak = gs.iter().find(|g| g.q == 0).unwrap();
    assert!((peak.wait - 150.0).abs() < 0.5, "{peak:?}");
    let sent = Sent::of(gs);
    assert!(sent.triples(&w.svc).is_empty());
    nudge(&mut w, b, 15.0);
    assert!(sent.moved(&w.net, &w.svc, &[b]).is_empty());
    nudge(&mut w, b, 105.0);
    assert_eq!(sent.moved(&w.net, &w.svc, &[b]), vec![b]);
    assert_eq!(sent.moved(&w.net, &w.svc, &[a]), vec![a], "its partner's stretch moved too");
    let now = groups(&w.net, &w.svc);
    for g in &now {
        if g.q <= 2 {
            assert!((g.factor - 1.16).abs() < 0.01 && (g.wait - 174.0).abs() < 1.0, "the peak row: {g:?}");
        } else {
            assert_eq!(factor_sent(g.factor), 1.0, "the evening and night rows stay: {g:?}");
        }
    }
    let t = Sent::of(now).triples(&w.svc);
    assert_eq!(t.len(), 3 * 2 * 8 * 3, "both lines' 8 hops in the peak row's 3 periods");
    assert!(t.chunks(3).all(|x| x[2] == 1.16));
}

/// As the clock worker reads it: what `shared_gaps` sent is what a nudge's wait is measured from, and only a
/// line whose shared stretch moved past 5 s is demand's.
#[test]
fn dirty_demand_lines_take_a_nudged_shared_stretch() {
    let (mut a, l5) = two_lines_on_one_track(true);
    assert!(a.shared_gaps().is_empty(), "evenly spaced");
    assert_eq!(a.nudge_row(l5, 0, 15.0, false), 0);
    assert!(a.dirty_demand_lines().is_empty());
    assert_eq!(a.nudge_row(l5, 0, 105.0, false), 0);
    assert!(a.dirty_demand_lines().contains(&l5), "{:?}", a.dirty_demand_lines());
    let v = a.shared_gaps();
    assert!(!v.is_empty() && v.chunks(3).all(|x| x[2] > 1.0), "{v:?}");
    assert_eq!(a.nudge_row(l5, 0, 1.0, false), 0);
    assert!(a.dirty_demand_lines().is_empty(), "from what was sent last");
}

/// T-4543: with the switch off (as shipped) the same nudges send no factor and call for no calculation.
#[test]
fn real_gaps_are_off_until_switched_on() {
    let (mut a, l5) = two_lines_on_one_track(false);
    assert_eq!(a.nudge_row(l5, 0, 120.0, false), 0);
    assert!(a.shared_gaps().is_empty());
    assert!(a.dirty_demand_lines().is_empty());
    assert!(a.gaps_moved(&[l5]).is_empty());
}

fn two_lines_on_one_track(on: bool) -> (TrackApi, u32) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_real_gaps(on);
    a.set_cash(1e6);
    let free = |x: f64, y: f64| [0.0, 0.0, x, y, 0.0];
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0);
    let l4 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l4, &[n0, n1], 2, 2, 2, 30.0, 180.0, 0, 0x00933c, "4".into()), 0);
    let l5 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l5, &[n0, n1], 2, 2, 2, 30.0, 180.0, 0, 0x00933c, "5".into()), 0);
    assert_eq!(a.construct_line(l4), 0);
    (a, l5)
}

/// The real New York save (trainworld/T-007-real-nyc.save) has no shared stretch: every line has its own track and
/// station nodes (`bin/realsave.rs`), so demand gets no gaps and its figures stay to the bit. On request, given the
/// save's network bytes (`python trainworld/checks/save_net.py <save> <out>`) in `T4514_NET`:
/// `cargo test --manifest-path sim/Cargo.toml real_save_shared_stretches -- --ignored --nocapture`.
#[test]
#[ignore]
fn real_save_shared_stretches() {
    use crate::track::frame::FrameKind;
    let raw = std::fs::read(std::env::var("T4514_NET").expect("T4514_NET: the save's network bytes")).unwrap();
    let net = crate::track::save::decode(&crate::track::save::convert_frame(&raw, FrameKind::Merc).unwrap()).unwrap();
    let w = TrackWorld::new(net);
    let running = (0..w.net.line_count()).filter(|&l| w.svc.lattice(&w.net, l).is_some()).count();
    let t = std::time::Instant::now();
    let gs = groups(&w.net, &w.svc);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let past = gs.iter().filter(|g| factor_sent(g.factor) > 1.0).count();
    println!("{running} running lines; {} shared stretches by period, {past} past 1; {ms:.1} ms", gs.len());
    assert!(Sent::of(gs).triples(&w.svc).is_empty());
}
