use super::*;

/// T-4309: the track model's calls after a station added on one edge of a big save, as the worker makes them for a
/// state, and the whole limits pass beside them. On request, on the save's track bytes: `T4309_TRACK=<file>
/// T4309_EDGE=1482 cargo test --profile agent limits_on_a_station_edit -- --ignored --nocapture`.
#[test]
#[ignore]
fn limits_on_a_station_edit() {
    use std::time::Instant;
    let mut a = TrackApi::new(0.0, 0.0, true);
    assert!(a.load(&std::fs::read(std::env::var("T4309_TRACK").unwrap()).unwrap()));
    let e: u32 = std::env::var("T4309_EDGE").unwrap().parse().unwrap();
    let ids: Vec<u32> = (0..a.w.net.line_count() as u32).collect();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    // the state before the edits is read once, as after a load
    a.take_changed_lines();
    for (i, f) in [0.2, 0.4, 0.6, 0.8].into_iter().enumerate() {
        let (p, q) = (a.w.net.edge_a[e as usize] as usize, a.w.net.edge_b[e as usize] as usize);
        let (x, y) = (a.w.net.node_x[p] * (1.0 - f) + a.w.net.node_x[q] * f, a.w.net.node_y[p] * (1.0 - f) + a.w.net.node_y[q] * f);
        let t = Instant::now();
        let r = a.add_station(2, e, x, y, 160, format!("probe {i}"));
        let t_edit = ms(t);
        let t = Instant::now();
        a.peek_changed_lines();
        let t_peek = ms(t);
        let t = Instant::now();
        let ch = a.take_changed_lines().len();
        let t_take = ms(t);
        let t = Instant::now();
        a.line_weekends(&ids);
        let t_weekends = ms(t);
        let t = Instant::now();
        a.profile_meta();
        let t_meta = ms(t);
        let t = Instant::now();
        for p in 0..5 {
            a.markers_in(p);
            a.full_track_in(p);
        }
        let t_marks = ms(t);
        let t = Instant::now();
        a.turn_limits();
        let t_turn = ms(t);
        let t = Instant::now();
        let whole = a.w.svc.cap.line_limits_lone(&a.w.net, true);
        let t_whole = ms(t);
        let t = Instant::now();
        let n = a.w.net.line_count();
        a.w.svc.cap.ensure_index(n);
        let kept = a.w.svc.cap.limits_kept(&a.w.net, n);
        let t_kept = ms(t);
        assert!(kept == whole);
        eprintln!("click {i} (r={r}, {ch} lines changed, {n} lines, {} resources): edit {t_edit:.1}; state: peek {t_peek:.2} take {t_take:.2} weekends {t_weekends:.2} meta {t_meta:.2} markers+full {t_marks:.2} turns {t_turn:.2} ms; limits alone: whole {t_whole:.2} kept {t_kept:.3} ms", a.w.svc.cap.res.len());
    }
}
