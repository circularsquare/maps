//! T-4152: joining new track onto the end of other track.
use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn node_at(a: &TrackApi, x: f64, y: f64) -> f64 {
    a.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 0.01 && (c[2] - y).abs() < 0.01).map(|c| c[0]).expect("node")
}
fn line(a: &mut TrackApi, x0: f64, y0: f64, x1: f64, y1: f64) {
    let ends = [0.0, 0.0, x0, y0, 0.0, 0.0, 0.0, x1, y1, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
}

/// A track end that lies almost straight ahead of the other (within a lead-in's 0.002 rad) gets no
/// lead-in, while the other end, turned a little, does: that lead-in used to move the first end's
/// straight off its heading, refused as a kink at a junction. Both ways round, blueprint and built.
#[test]
fn near_straight_end_to_end_joins() {
    let mut refused = vec![];
    for built in [false, true] {
        for rev in [false, true] {
            for &rot in &[0.0f64, 0.2, 1.0, 2.0, 5.0, 20.0] {
                for &dy in &[0.0f64, 0.2, 1.0, 5.0] {
                    for &gap in &[100.0f64, 500.0, 2000.0] {
                        let mut a = api();
                        line(&mut a, 0.0, 0.0, 1000.0, 0.0);
                        let (bx, by) = (1000.0 + gap, dy);
                        let r = rot.to_radians();
                        line(&mut a, bx, by, bx + 1500.0 * r.cos(), by + 1500.0 * r.sin());
                        if built {
                            assert_eq!(a.construct_all(), 0, "{}", a.issues());
                        }
                        let (na, nb) = (node_at(&a, 1000.0, 0.0), node_at(&a, bx, by));
                        a.set_route_tangent(true);
                        let ends = if rev { [1.0, nb, 0.0, 0.0, 0.0, 1.0, na, 0.0, 0.0, 0.0] } else { [1.0, na, 0.0, 0.0, 0.0, 1.0, nb, 0.0, 0.0, 0.0] };
                        if a.add_route(&ends, &[], 2, false) != 0 {
                            refused.push(format!("built={built} rev={rev} rot={rot} dy={dy} gap={gap}: {}", a.issues()));
                        }
                    }
                }
            }
        }
    }
    assert!(refused.is_empty(), "{}", refused.join("\n"));
}
