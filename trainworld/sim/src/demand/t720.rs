//! T-720: access subzones split by walk bands, and the trip explorer's answer for the two cells'
//! own trip, on the synthetic city with one sparse line.

use super::api::DemandApi;
use super::kernel::*;
use super::trip::{HEAD, PAIR_LEN};
use super::{pack, synth};

/// The synthetic city's inner 12 km, one line of three stations 2 km apart along y = 0, solved
/// in the morning peak with walk bands of `band` minutes.
fn sparse(band: f32) -> DemandApi {
    let mut city = synth::synth_city(1).inner(12_000.0);
    let p = Params::default();
    let z = build_zones(&city, p.zone_m);
    let g = gravity(&z, &p, 100, 1e-5);
    city.gravity = Some(to_pack_gravity(&z, &g, &p));
    let (h, b) = pack::write(&city);
    let mut api = DemandApi::new(&h, &b).unwrap();
    api.params_mut().band_min = band;
    let n = 3;
    let xy = [-2000.0f32, 0.0, 0.0, 0.0, 2000.0, 0.0];
    let mut times = vec![0f32; 6 * n];
    for run in 0..2 {
        for lev in 0..3 {
            for k in 0..n {
                let end = if run == 0 { k + 1 == n } else { k == 0 };
                times[(run * 3 + lev) * n + k] = if end { 0.0 } else { 180.0 };
            }
        }
    }
    let km: Vec<f32> = (0..2).flat_map(|_| (0..n).map(move |k| if k + 1 == n { 0.0 } else { 2.0 })).collect();
    api.set_network(&xy, &[n as u32], &[0, 1, 2], &times, &[12.0, 8.0, 4.0], &[8.0]).unwrap();
    api.set_fares(2.9, 0.1, &km).unwrap();
    api.solve(0);
    api
}

/// Subzones whose cells' walks to their nearest station span more than one band.
fn mixed_subzones(api: &DemandApi, band: f32) -> usize {
    let sz = api.sz.as_ref().unwrap();
    let net = DemandApi::period_net(api.svc.as_ref().unwrap(), 0);
    let acc = build_access(&api.city, &net, &api.base);
    (0..sz.n)
        .filter(|&u| {
            let m = &sz.mem[sz.mem_off[u] as usize..sz.mem_off[u + 1] as usize];
            let b: Vec<u32> = m.iter().filter(|&&c| acc.n[c as usize] > 0).map(|&c| (acc.walk[c as usize * K] / band) as u32).collect();
            b.iter().any(|&x| x != b[0])
        })
        .count()
}

/// Every subzone's cells are within one 15-minute walk band of their station (Anita, T-720); a
/// zone's cells sharing a station were one subzone before, 1 to 30 minutes away.
#[test]
fn subzones_keep_to_one_walk_band() {
    assert_eq!(Params::default().band_min, 15.0);
    assert_eq!(mixed_subzones(&sparse(15.0), 15.0), 0);
    assert!(mixed_subzones(&sparse(0.0), 15.0) > 0, "without bands a subzone spans several");
}

/// The explorer answers for the two cells: its walk to the station is the home cell's own, and
/// with one subzone per cell it is the model's own answer (slots 37-39, the subzones', agree).
#[test]
fn the_explorer_walks_from_the_home_cell() {
    let api = sparse(15.0);
    let svc = api.svc.as_ref().unwrap();
    let p = &api.base;
    let mut walks = 0;
    for (ox, oy) in [(-2500.0f32, 1200.0f32), (-1800.0, -1600.0), (-600.0, 1900.0), (-2200.0, 300.0)] {
        let v = api.trip(0, ox, oy, 2100.0, 300.0);
        assert_eq!(v[0], 1.0);
        let o = v[2] as usize;
        for k in 0..v[24] as usize {
            let e = &v[HEAD + k * PAIR_LEN..HEAD + (k + 1) * PAIR_LEN];
            if e[12] == 0.0 {
                let a = e[0] as usize;
                let own = (api.city.x[o] - svc.st_x[a]).hypot(api.city.y[o] - svc.st_y[a]);
                assert!((e[10] - p.walk_min(own) as f64).abs() < 1e-3, "{} against the cell's own {}", e[10], p.walk_min(own));
                walks += 1;
            }
        }
    }
    assert!(walks > 0, "walks to check");
    let exact = sparse(-1.0);
    for (ox, oy) in [(-2500.0f32, 1200.0f32), (-1800.0, -1600.0), (-600.0, 1900.0)] {
        let v = exact.trip(0, ox, oy, 2100.0, 300.0);
        for (own, grouped) in [(17, 37), (18, 38), (19, 39)] {
            assert!((v[own] - v[grouped]).abs() < 1e-4, "slot {own}: {} against the subzones' {}", v[own], v[grouped]);
        }
    }
}
