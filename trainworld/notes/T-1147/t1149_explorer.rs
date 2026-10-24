//! T-1149 scratch: the trip explorer between London and Paris (`market_trip_between`) against the
//! solve's corridor figures, on a lone St Pancras - Gare du Nord line (`real_far_lines_share_rail_with_air`'s).
use super::*;

#[test]
#[ignore]
fn t1149_explorer_matches_solve() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let pairs: [(&str, &str, (f64, f64), (f64, f64), f32, f32); 3] = [
        ("lon", "par", (-0.1263, 51.5306), (2.3553, 48.8809), 492., 2.27),
        ("nyc", "wdc", (-73.9935, 40.7506), (-77.0064, 38.8973), 362., 2.75),
        ("tyo", "osa", (139.7671, 35.6812), (135.5003, 34.7334), 553., 2.5),
    ];
    for (a, b, sa, sb, km, h) in pairs {
        let mut api = IntercityApi::new(&super::tests::real_world(root, Some(&[a, b]))).unwrap();
        let (pa, pb) = (api.m.places.iter().position(|p| p.zone == a).unwrap(), api.m.places.iter().position(|p| p.zone == b).unwrap());
        let c = api.m.corridor(pa, pb);
        let (air, car) = (api.corridor_modes()[2 * c], api.corridor_modes()[2 * c + 1]);
        eprintln!("{a}-{b}: no rail, plane {:.1}% car {:.1}%", 100. * air, 100. * car);
        for (fb, fk) in [(1.5f32, 0.1f32), (1.5, 0.3), (1.5, 0.5)] {
            let s = h * 3600.;
            let times = [s, 0., s, 0., s, 0., 0., s, 0., s, 0., s];
            api.set_network_ll(&[sa.0, sa.1, sb.0, sb.1], &[2], &[0, 1], &times, &[2., 2., 2.], &[16.]).unwrap();
            api.set_fares(fb, fk, &[km, 0., 0., km]).unwrap();
            assert!(api.solve(1));
            let cs = api.corridor_summary(1);
            let (trips, rail) = (cs[4 * c + 2], cs[4 * c + 3]);
            let rest = trips - rail;
            let mut x = [0f64; 4];
            for dir in [0u32, 1] {
                let v = api.market_trip_between(1, 0, pa as u32, -1, dir, pb as u32, -1);
                for k in 0..4 {
                    x[k] += v[2 + k] / 2.;
                }
            }
            eprintln!(
                "  fare ${fb}+${fk}/km ({:.0}): solve trips {:.0} rail {:.1}% plane {:.1}% car {:.1}% | explorer trips {:.0} rail {:.1}% plane {:.1}% car {:.1}%",
                fb + fk * km, trips, 100. * rail / trips, 100. * rest * air as f64 / trips, 100. * rest * car as f64 / trips,
                x[0], 100. * x[1] / x[0], 100. * x[2] / x[0], 100. * x[3] / x[0]
            );
        }
    }
}
