//! T-2214: islands no road reaches (land parts, T-2213) in local demand, on the real packs
//! (notes/T-2214.md):
//!
//!   cargo test --release --features demand --lib t2214 -- --ignored --nocapture --test-threads=1
//!
//! `t2214_no_network`: the commutes between two land parts with no network against T-2207's
//! section 3 (Athens, Kagawa, Hong Kong within 20%). `t2214_sado_ferry`: Niigata with a test
//! ferry Ryotsu - Niigata port; Sado's commutes and everyday trips off the island all ferry or no
//! way, none by car or on foot.

use super::*;
use crate::demand::everyday::EverydayApi;
use crate::demand::network::route_nodes;
use crate::demand::pack;
use crate::track::frame::Frame;

fn open(id: &str) -> (String, Vec<u8>) {
    pack::load_raw(&format!("{}/../data/packs/{id}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
#[ignore]
fn t2214_no_network() {
    // T-2207 section 3: commuters across a day with no network
    for (id, want, gate) in [("ath", 22_200.0, true), ("kgw", 5_300.0, true), ("hon", 49_000.0, true), ("nig", 2_660.0, false), ("sea", 6_700.0, false), ("nyc", 1_370.0, false), ("tyo", 0.0, false)] {
        let (h, b) = open(id);
        let a = DemandApi::new(&h, &b).unwrap();
        let parts = a.base.parts.as_ref().map_or(1, |pp| pp.cell.iter().copied().max().unwrap_or(0) as usize + 1);
        println!("{id}: {:.0} commutes a day between two land parts of {:.0} ({:.2}%), T-2207 {want:.0}; {parts} parts, {} mixed zones, opened in {:.0} ms", a.unr_all, a.gr.total, 100.0 * a.unr_all / a.gr.total, a.base.parts.as_ref().map_or(0, |pp| pp.mixed.len()), a.open_ms);
        if gate {
            assert!((a.unr_all / want - 1.0).abs() <= 0.2, "{id}: {:.0} against {want}", a.unr_all);
        }
    }
}

#[test]
#[ignore]
fn t2214_sado_ferry() {
    let (h, b) = open("nig");
    let mut a = DemandApi::new(&h, &b).unwrap();
    let hv: serde_json::Value = serde_json::from_str(&h).unwrap();
    let f = Frame::equirect(hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
    // Sado Kisen's Ryotsu and Niigata terminals; a boat an hour, 65 minutes across
    let (rx, ry) = f.from_lonlat(138.4396, 38.0809);
    let (nx, ny) = f.from_lonlat(139.0610, 37.9290);
    let st_xy = vec![rx as f32, ry as f32, nx as f32, ny as f32];
    let (line_n, stops) = (vec![2u32], vec![0u32, 1]);
    let times = vec![3900.0, 0.0, 3900.0, 0.0, 3900.0, 0.0, 3900.0, 0.0, 3900.0, 0.0, 3900.0, 0.0];
    let (tph, cars) = (vec![1.0f32, 1.0, 1.0], vec![2.0f32]);
    let hop_km = route_nodes(&st_xy, &line_n, &stops).2;
    a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(1.5, 0.1, &hop_km).unwrap();
    let pp = a.base.parts.clone().expect("nig has land parts");
    let sz = a.sz.as_ref().unwrap();
    let sado = sz.st_part[0];
    assert_ne!(sado, sz.st_part[1], "the two ports are on one land part");
    assert_eq!(sado, pp.cell[a.nearest_cell(rx as f32, ry as f32).unwrap()]);
    for q in 0..PERIODS {
        assert!(a.solve(q));
    }
    let sums = a.sub_sums();
    let m = a.modes_cells(&sums, WEEKDAY as u32);
    let n = a.city.len();
    let on = |c: usize| pp.cell[c] == sado;
    let sum = |k: usize| (0..n).filter(|&c| on(c)).map(|c| m[k * n + c] as f64).sum::<f64>();
    let (rail, walk, drive, unr) = (sum(0), sum(1), sum(2), sum(6));
    let total: f64 = (0..n).filter(|&c| on(c)).map(|c| (a.city.pop[c] * a.home_k[a.z.cell_zone[c] as usize]) as f64).sum();
    // the day's commutes from Sado's homes off it, before any rail
    let across: f64 = (0..pp.s_part.len()).filter(|&s| pp.s_part[s] == sado).map(|s| a.unr_home[s]).sum();
    println!("Sado commutes from home a day: {total:.0}, {across:.0} off the island: ferry {rail:.1}, no way {unr:.1}; on it walk {walk:.0}, drive {drive:.0}");
    // the ferry is the only line, so every rail trip from Sado crosses
    assert!(((rail + unr) / across - 1.0).abs() < 1e-3, "ferry {rail} + no way {unr} against {across} across");
    assert!(((walk + drive) / (total - across) - 1.0).abs() < 1e-3, "walk {walk} + drive {drive} against {} on the island", total - across);
    assert!(rail > 0.0, "the ferry carries no commuter");
    let s = a.summary(0);
    println!("period 0: trips {:.0}, rail {:.1}, walk {:.0}, no way {:.1}", s[1], s[2], s[3], s[13]);

    // the trip explorer, Ryotsu to Niigata station: neither car nor walk
    let (sx, sy) = f.from_lonlat(139.0617, 37.9121);
    let t = a.trip(0, rx as f32, ry as f32 + 500.0, sx as f32, sy as f32);
    println!("trip Ryotsu - Niigata: apart {}, walk {:.3}, rail {:.3}, drive {:.3}", t[47], t[17], t[18], t[19]);
    assert!(t[47] == 1.0 && t[17] == 0.0 && t[19] == 0.0);

    // everyday trips: off the island all ferry or no way
    let mut e = EverydayApi::beside(&a, &h, &b).unwrap();
    e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    e.set_fares(1.5, 0.1, &hop_km).unwrap();
    for q in 0..PERIODS {
        assert!(e.solve(q));
    }
    let em = e.modes_all(None, WEEKDAY, None);
    let es = |k: usize| (0..n).filter(|&c| on(c)).map(|c| em[k * n + c] as f64).sum::<f64>();
    let (erail, ewalk, edrive, eunr) = (es(0), es(1), es(2), es(6));
    println!("Sado everyday trips from home a day: ferry {erail:.1}, no way {eunr:.1}; on the island walk {ewalk:.0}, drive {edrive:.0}");
    let unr_day: f64 = (0..PERIODS).map(|q| e.summary(q)[13]).sum();
    println!("everyday no way a day (both ends of the pack): {unr_day:.1}");
    assert!(eunr > 0.0);
}
