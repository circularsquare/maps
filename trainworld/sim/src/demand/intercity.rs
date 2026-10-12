//! T-117: a separate, tunable non-commute long-distance market on the common rail graph.
use super::{
    api::{merge_stations, DemandApi, Service, FARE_MIN_PER_USD, PEAK_HOUR_FACTOR},
    clock::now_ms,
    kernel::*,
    network::PERIODS,
    pack::City,
};
use crate::track::params::PERIOD_HOURS;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use wasm_bindgen::prelude::*;

pub const SHARES: [f32; PERIODS] = [0.20, 0.35, 0.25, 0.15, 0.05];
// Only the numeric runtime inputs are retained. Deserializing a generic Value builds
// an individually allocated JSON array for every site and keeps it alive while pairs
// are allocated; WASM cannot return that high-water memory to the browser.
#[derive(Deserialize)]
struct MarketPack {
    #[serde(default = "default_zone_m")]
    zone_m: f64,
    #[serde(default = "default_trip_rate")]
    trip_rate: f64,
    #[serde(default = "legacy_markets")]
    markets: Vec<String>,
    sites: Vec<[f64; 5]>,
}
fn legacy_markets() -> Vec<String> {
    ["New York", "Philadelphia", "Trenton / Mercer County"]
        .map(str::to_owned)
        .to_vec()
}
fn default_zone_m() -> f64 {
    6000.
}
fn default_trip_rate() -> f64 {
    0.015
}
struct Zone {
    x: f32,
    y: f32,
    pop: f64,
    attr: f64,
    market: usize,
}
struct Pair {
    a: usize,
    b: usize,
    trips: f64,
    drive: f32,
    corridor: usize,
}
struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    avg: Loads,
    rail: f64,
    trips: f64,
    corridors: Vec<f64>,
    /// Rail trips of each directed pair in this period (T-118), averaged over rounds like the loads.
    pair_rail: Vec<f32>,
    /// Per station, `2 * zones` numbers (T-118, T-120): rail trips starting here by destination
    /// zone, then ending here by origin zone. f32: it is one cell per station and zone.
    st: Vec<f32>,
    round: usize,
    ms: f64,
    /// As `api::Period`: the factors of the last pass, and whether every pass used them.
    fac: Vec<f32>,
    uniform: bool,
}
// Legacy fixtures retain the three-market layout. Runtime widths follow the pack.
#[cfg(test)]
const ST_W: usize = 18;
/// Side of the endpoint sites in metres (pipeline/intercity.py groups cells into 600 m squares).
const SITE_M: f32 = 600.;
#[cfg(test)]
const MARKETS: usize = 3;
/// What `pass` hands back besides the loads.
struct Pass {
    loads: Loads,
    rail: f64,
    cor: Vec<f64>,
    pair_rail: Vec<f32>,
    st: Vec<f32>,
}
#[wasm_bindgen]
pub struct IntercityApi {
    city: City,
    zones: Vec<Zone>,
    cell_zone: Vec<usize>,
    pairs: Vec<Pair>,
    base: Params,
    svc: Option<Service>,
    access: Vec<Vec<(usize, f32)>>,
    access_work: Vec<Vec<(usize, f32)>>,
    /// The stations a trip can end at: the only ones strategies are searched to (T-119).
    want: Vec<bool>,
    per: Vec<Option<Period>>,
    fare_km: Vec<f32>,
    fare_per_km: f32,
    corridors: Vec<(usize, usize)>,
    markets: usize,
    daily: f64,
    /// Site of a 600 m square of a market, to find the sites a selection of local cells holds.
    site_key: HashMap<(i32, i32, usize), usize>,
    /// Indices into `pairs` by origin zone and by destination zone.
    by_a: Vec<Vec<u32>>,
    by_b: Vec<Vec<u32>>,
}

impl IntercityApi {
    fn pass(&self, g: &Graph, p: &Params, q: usize) -> Pass {
        // the strategies are shared with the local populations on the same service (T-119)
        with_skims(g, p, &self.want, |sk| self.pass_on(g, p, q, sk))
    }
    fn pass_on(&self, g: &Graph, p: &Params, q: usize, sk: &Skims) -> Pass {
        let s = g.n_st;
        let mut od = vec![0.; s * s];
        let mut rail = 0.;
        let mut cor = vec![0.; self.corridors.len()];
        let mut pair_rail = vec![0f32; self.pairs.len()];
        let nz = self.zones.len();
        let w = 2 * nz;
        let mut st = vec![0f32; s * w];
        for (pi, pair) in self.pairs.iter().enumerate() {
            let trips = pair.trips * SHARES[q] as f64;
            let (a, b) = (&self.access[pair.a], &self.access_work[pair.b]);
            let mut terms = Vec::with_capacity(a.len() * b.len());
            let mut sum = 0f64;
            for &(sa, wa) in a {
                for &(sb, wb) in b {
                    if sa == sb {
                        continue;
                    } // A rail journey must actually board a train.
                    let cost = sk.r[sa * s + sb] + p.fare_min;
                    if !cost.is_finite() {
                        continue;
                    }
                    let v = wa as f64
                        * wb as f64
                        * ((-p.beta * (cost - pair.drive)).clamp(-80., 80.) as f64).exp();
                    sum += v;
                    terms.push((sa * s + sb, v));
                }
            }
            let fraction = sum / (1. + sum);
            let r = trips * fraction;
            rail += r;
            cor[pair.corridor] += r;
            pair_rail[pi] = r as f32;
            if sum > 0. {
                for (k, v) in terms {
                    let rr = r * v / sum;
                    od[k] += rr as f32;
                    let (sa, sb) = (k / s, k % s);
                    st[sa * w + pair.b] += rr as f32;
                    st[sb * w + nz + pair.a] += rr as f32;
                }
            }
        }
        Pass {
            loads: assign(g, sk, &od),
            rail,
            cor,
            pair_rail,
            st,
        }
    }
    fn per(&self, q: usize) -> Option<&Period> {
        self.per.get(q).and_then(|v| v.as_ref())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intercity_prunes_symmetric_pairs_after_averaging() {
        // Directed departures are .03 and .0015: only one passes the old threshold.
        let keep = IntercityApi::new(r#"{"sites":[[0,0,2,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
        assert_eq!(keep.pairs.len(), 2);
        assert_eq!(keep.pairs[0].trips, keep.pairs[1].trips);
        assert!((keep.pairs[0].trips - 0.01575).abs() < 1e-12);
        assert_eq!(
            (keep.pairs[0].a, keep.pairs[0].b),
            (keep.pairs[1].b, keep.pairs[1].a)
        );
        // .015 and .0015 average below .01: both directions must disappear.
        let drop = IntercityApi::new(r#"{"sites":[[0,0,1,100,0],[120000,0,0.1,0,1]]}"#).unwrap();
        assert!(drop.pairs.is_empty());
    }
    fn api() -> IntercityApi {
        let header = r#"{"zone_m":6000,"trip_rate":0.015,"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]]}"#;
        IntercityApi::new(header).unwrap()
    }
    fn network(api: &mut IntercityApi, freq: f32, slow: f32) {
        // Local feeder at each end, through intercity line, Mercer intermediate stop.
        let xy = [0., 0., 1000., 0., 60000., 0., 119000., 0., 120000., 0.];
        let n = [2, 3, 2];
        let stops = [0, 1, 1, 2, 3, 3, 4];
        let mut times = vec![0.; 42];
        for (a, len, sec) in [(0, 2, 120.), (2, 3, 1800. * slow), (5, 2, 120.)] {
            for lev in 0..3 {
                for k in 0..len - 1 {
                    times[6 * a + lev * len + k] = sec;
                    times[6 * a + (3 + lev) * len + k + 1] = sec;
                }
            }
        }
        api.set_network(
            &xy,
            &n,
            &stops,
            &times,
            &[12., 12., 12., freq, freq, freq, 12., 12., 12.],
            &[4., 4., 4.],
        )
        .unwrap();
        api.set_fares(
            0.,
            0.,
            &[1., 0., 0., 1., 59., 59., 0., 0., 59., 59., 1., 0., 0., 1.],
        )
        .unwrap();
    }
    #[test]
    fn intercity_market_is_symmetric_and_conserved() {
        let a = api();
        assert!((a.daily - 3600.).abs() < 1e-5);
        for p in &a.pairs {
            let reverse = a.pairs.iter().find(|q| q.a == p.b && q.b == p.a).unwrap();
            assert_eq!(p.trips, reverse.trips);
        }
        assert!((SHARES.iter().sum::<f32>() - 1.).abs() < 1e-6);
    }
    #[test]
    fn intercity_uses_feeders_fares_service_and_shared_crowding() {
        let mut a = api();
        network(&mut a, 6., 1.);
        a.solve(0);
        let good = a.summary(0)[2];
        assert!(good > 100.);
        let board = a.board(0);
        assert!(
            board[0] > 0. && board[5] > 0.,
            "local feeder boardings: {board:?}"
        );
        assert!((a.entries(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
        assert!((a.exits(0).iter().sum::<f32>() as f64 - good).abs() < 0.01);
        let sum = a
            .corridor_summary(0)
            .chunks_exact(4)
            .map(|v| v[3])
            .sum::<f64>();
        assert!((sum - good).abs() < 1e-6);
        let loads = vec![1e7; a.seg(0).len()];
        assert!(a.crowd_with_external(0, &loads));
        assert!(a.summary(0)[2] < good);
        a.set_fares(80., 0., &[0.; 14]).unwrap();
        a.solve(0);
        assert!(a.summary(0)[2] < good * 0.1);
        network(&mut a, 6., 3.);
        a.solve(0);
        assert!(a.summary(0)[2] < good);
        network(&mut a, 0., 1.);
        a.solve(0);
        assert_eq!(a.summary(0)[2], 0.);
    }
    #[test]
    fn intercity_access_respects_home_and_destination_weights() {
        let header = r#"{"sites":[[0,0,100000,0,0],[4000,0,0,100000,0],[100000,0,100000,0,1],[104000,0,0,100000,1]]}"#;
        let mut a = IntercityApi::new(header).unwrap();
        let mut times = vec![0.; 24];
        for lev in 0..3 {
            for k in 0..3 {
                let sec = if k == 1 { 2400. } else { 120. };
                times[lev * 4 + k] = sec;
                times[(3 + lev) * 4 + k + 1] = sec;
            }
        }
        a.set_network(
            &[0., 0., 4000., 0., 100000., 0., 104000., 0.],
            &[4],
            &[0, 1, 2, 3],
            &times,
            &[6., 6., 6.],
            &[4.],
        )
        .unwrap();
        a.solve(0);
        let e = a.entries(0);
        let x = a.exits(0);
        assert!(e[0] > e[1], "origins follow homes: {e:?}");
        assert!(x[1] > x[0], "destinations favour jobs: {x:?}");
    }
    fn solve_all(a: &mut IntercityApi) {
        for q in 0..PERIODS {
            assert!(a.solve(q));
        }
    }
    fn sum_markets(v: &[f64], off: usize) -> f64 {
        (0..MARKETS).map(|m| v[1 + 8 * m + off]).sum()
    }
    #[test]
    fn intercity_selection_places_conserve_and_rail_is_within_all() {
        let mut a = api();
        network(&mut a, 6., 1.);
        solve_all(&mut a);
        assert_eq!(a.solved_mask(), 31);
        // The whole of market 0 (one site at the origin): starts equal its zone's departures.
        let all = a.selection_flows(0, &[0., 0.], &[1.]);
        assert_eq!(all[0], 1.);
        let want: f64 = a
            .pairs
            .iter()
            .filter(|p| a.zones[p.a].market == 0)
            .map(|p| p.trips)
            .sum();
        assert!(want > 0.);
        assert!(
            (sum_markets(&all, 0) - want).abs() < 1e-6 * want.max(1.),
            "{all:?} vs {want}"
        );
        let want_in: f64 = a
            .pairs
            .iter()
            .filter(|p| a.zones[p.b].market == 0)
            .map(|p| p.trips)
            .sum();
        assert!((sum_markets(&all, 4) - want_in).abs() < 1e-6 * want_in.max(1.));
        for m in 0..MARKETS {
            for e in [0, 4] {
                let (trips, rail) = (all[1 + 8 * m + e], all[2 + 8 * m + e]);
                assert!(
                    rail <= trips + 1e-9,
                    "market {m}: rail {rail} above all {trips}"
                );
            }
        }
        assert!(
            sum_markets(&all, 1) > 0.,
            "service carries some of the selection"
        );
        // A part of the site is that part of everything; a point off every site matches nothing.
        let half = a.selection_flows(0, &[0., 0.], &[0.5]);
        for k in 1..1 + 8 * MARKETS {
            assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
        }
        let none = a.selection_flows(0, &[9e6, 9e6], &[1.]);
        assert_eq!(none[0], 0.);
        let head = 1 + 8 * MARKETS;
        let nz = a.zones.len();
        assert!(none[1..head + 4 * nz].iter().all(|v| *v == 0.));
        // per far zone (T-120): the sums equal the market sums, rail <= all, positions are the zones' own
        assert_eq!(all.len(), head + 7 * nz);
        for d in 0..2 {
            let (mut tot, mut rail, mut sx) = (0., 0., 0.);
            for z in 0..nz {
                let (t, r) = (all[head + 4 * z + 2 * d], all[head + 4 * z + 2 * d + 1]);
                assert!(r <= t + 1e-9, "zone {z}: rail {r} above all {t}");
                tot += t;
                rail += r;
                sx += t * a.zones[z].x as f64;
            }
            let (mt, mr) = (sum_markets(&all, 4 * d), sum_markets(&all, 4 * d + 1));
            assert!((tot - mt).abs() < 1e-9 * mt.max(1.) && (rail - mr).abs() < 1e-9 * mr.max(1.));
            assert!((sx - sum_markets(&all, 4 * d + 2)).abs() < 1e-6 * sx.abs().max(1.));
        }
        for z in 0..nz {
            assert_eq!(all[head + 4 * nz + 3 * z], a.zones[z].x as f64);
            assert_eq!(all[head + 4 * nz + 3 * z + 1], a.zones[z].y as f64);
            assert_eq!(all[head + 4 * nz + 3 * z + 2], a.zones[z].market as f64);
        }
        // a half share gives half of every zone too
        for k in head..head + 4 * nz {
            assert!((half[k] - 0.5 * all[k]).abs() < 1e-9 * all[k].abs().max(1.));
        }
        // The far centroid of New York's trips lies between the other two places.
        let (sx, tot) = (all[1 + 8 + 2], all[1 + 8]);
        assert!(
            tot > 0. && sx / tot > 100_000.,
            "Philadelphia sits 120 km east: {}",
            sx / tot
        );
    }
    #[test]
    fn intercity_station_places_equal_entries_and_exits() {
        let mut a = api();
        network(&mut a, 6., 1.);
        solve_all(&mut a);
        // the network's five stations: 0 and 1 are local feeders, 2 is Mercer, 3 and 4 Philadelphia's
        let map = a.station_map();
        let mut starts = 0.;
        let mut ends = 0.;
        for st in 0..5u32 {
            let row = a.station_flows(st);
            let nz = a.zones.len();
            assert_eq!(row.len(), ST_W + 5 * nz);
            // the zone rows add up to the market sums, and rail sits at the zones' own positions
            for d in 0..2 {
                let zsum: f64 = row[ST_W + d * nz..ST_W + (d + 1) * nz].iter().sum();
                let msum: f64 = row[d * MARKETS..(d + 1) * MARKETS].iter().sum();
                assert!((zsum - msum).abs() < 1e-9 * msum.max(1.));
            }
            for z in 0..nz {
                assert_eq!(row[ST_W + 2 * nz + 3 * z], a.zones[z].x as f64);
                assert_eq!(row[ST_W + 2 * nz + 3 * z + 1], a.zones[z].y as f64);
                assert_eq!(row[ST_W + 2 * nz + 3 * z + 2], a.zones[z].market as f64);
            }
            let (s, e) = (
                row[..MARKETS].iter().sum::<f64>(),
                row[MARKETS..2 * MARKETS].iter().sum::<f64>(),
            );
            let at = map[st as usize] as usize;
            let entries: f64 = (0..PERIODS).map(|q| a.entries(q)[at] as f64).sum();
            let exits: f64 = (0..PERIODS).map(|q| a.exits(q)[at] as f64).sum();
            assert!(
                (s - entries).abs() < 1e-3 * entries.max(1.),
                "station {st}: {s} vs {entries}"
            );
            assert!(
                (e - exits).abs() < 1e-3 * exits.max(1.),
                "station {st}: {e} vs {exits}"
            );
            starts += s;
            ends += e;
        }
        let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        assert!(
            rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail
        );
        assert!(a.station_flows(99).is_empty());
        // the same after crowding rounds (loads are averaged, not replaced)
        for q in 0..PERIODS {
            let loads = a.seg(q);
            assert!(a.crowd_with_external(q, &loads));
        }
        let rail2: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        let starts2: f64 = (0..5u32)
            .map(|s| a.station_flows(s)[..MARKETS].iter().sum::<f64>())
            .sum();
        assert!((starts2 - rail2).abs() < 1e-3 * rail2.max(1.));
    }
    #[test]
    #[ignore]
    fn intercity_real_pack_profile() {
        let header = std::fs::read_to_string("../data/packs/intercity.json").unwrap();
        let t = std::time::Instant::now();
        let mut a = IntercityApi::new(&header).unwrap();
        eprintln!(
            "intercity info {:?}, pairs {}, open {:?}",
            a.info(),
            a.pairs.len(),
            t.elapsed()
        );
        let x = [0., 0., -67000., -62000., -99000., -89000.];
        let mut times = vec![0.; 18];
        for lev in 0..3 {
            times[lev * 3] = 2400.;
            times[lev * 3 + 1] = 1200.;
            times[(3 + lev) * 3 + 1] = 2400.;
            times[(3 + lev) * 3 + 2] = 1200.;
        }
        a.set_network(&x, &[3], &[0, 1, 2], &times, &[6., 6., 6.], &[8.])
            .unwrap();
        let t = std::time::Instant::now();
        a.solve(0);
        eprintln!(
            "intercity solve {:?}, summary {:?}, corridors {:?}",
            t.elapsed(),
            a.summary(0),
            a.corridor_summary(0)
        );
        assert!(a.summary(0)[2] > 0.);
        assert!(a.summary(0).iter().all(|v| v.is_finite()));
    }
    #[test]
    fn intercity_no_stations_and_invalid_input_are_safe() {
        let mut a = api();
        a.set_network(&[], &[], &[], &[], &[], &[]).unwrap();
        assert!(a.solve(0));
        assert_eq!(a.summary(0)[2], 0.);
        assert!(a.seg(0).is_empty());
        assert!(!a.crowd_with_external(0, &[f32::NAN]));
        assert!(!a.solve(5));
        assert!(IntercityApi::new(r#"{"sites":[[0,0,-1,0,0]]}"#).is_err());
    }
    #[test]
    fn expanded_markets_route_boston_and_washington_and_keep_flow_layouts() {
        let mut a = IntercityApi::new(r#"{"markets":["New York","Philadelphia","Mercer","Boston","Washington DC"],"sites":[[0,0,100000,50000,0],[60000,0,100000,50000,1],[120000,0,100000,50000,2],[180000,0,100000,50000,3],[240000,0,100000,50000,4]]}"#).unwrap();
        assert_eq!(a.market_count(), 5);
        assert_eq!(a.corridors.len(), 15);
        let mut times = vec![0.; 30];
        for lev in 0..3 {
            for k in 0..4 {
                times[lev * 5 + k] = 1200.;
                times[(lev + 3) * 5 + k + 1] = 1200.;
            }
        }
        a.set_network(
            &[0., 0., 60000., 0., 120000., 0., 180000., 0., 240000., 0.],
            &[5],
            &[0, 1, 2, 3, 4],
            &times,
            &[6., 6., 6.],
            &[8.],
        )
        .unwrap();
        solve_all(&mut a);
        let nz = a.zones.len();
        let selected = a.selection_flows(3, &[180000., 0.], &[1.]);
        assert_eq!(selected.len(), 41 + 7 * nz);
        assert_eq!(selected[0], 1.);
        assert!(
            selected[1 + 8 * 4 + 1] > 0.,
            "Boston to Washington is served"
        );
        let washington = a.selection_flows(4, &[240000., 0.], &[1.]);
        assert_eq!(washington[0], 1.);
        assert!(
            washington[1 + 8 * 3 + 1] > 0.,
            "Washington to Boston is served"
        );
        let mut starts = 0.;
        let mut ends = 0.;
        for st in 0..5 {
            let row = a.station_flows(st);
            assert_eq!(row.len(), 30 + 5 * nz);
            for d in 0..2 {
                let by_market: f64 = row[d * 5..(d + 1) * 5].iter().sum();
                let by_zone: f64 = row[30 + d * nz..30 + (d + 1) * nz].iter().sum();
                assert!((by_market - by_zone).abs() < 1e-9 * by_market.max(1.));
            }
            starts += row[..5].iter().sum::<f64>();
            ends += row[5..10].iter().sum::<f64>();
        }
        let rail: f64 = (0..PERIODS).map(|q| a.summary(q)[2]).sum();
        assert!(
            rail > 0. && (starts - rail).abs() < 1e-3 * rail && (ends - rail).abs() < 1e-3 * rail
        );
        assert_eq!(
            api().market_count(),
            3,
            "legacy packs retain their output layout"
        );
    }
    #[test]
    fn compact_pack_accepts_metadata_and_rejects_malformed_sites() {
        let base = api();
        let reordered = IntercityApi::new(r#"{"sites":[[0,0,100000,50000,0],[120000,0,100000,50000,1],[60000,0,40000,20000,2]],"description":"unused","origin":{"lon":-73.985,"lat":40.758},"zone_m":6000,"trip_rate":0.015}"#).unwrap();
        assert_eq!(base.info(), reordered.info());
        for (a, b) in base.pairs.iter().zip(&reordered.pairs) {
            assert_eq!(
                (a.a, a.b, a.trips, a.drive, a.corridor),
                (b.a, b.b, b.trips, b.drive, b.corridor)
            );
        }
        for bad in [
            r#"{}"#,
            r#"{"sites":[[0,0,1,0]]}"#,
            r#"{"sites":[[0,0,1,0,0,0]]}"#,
            r#"{"sites":[[0,0,null,0,0]]}"#,
            r#"{"sites":[[0,0,"1",0,0]]}"#,
            r#"{"zone_m":0,"sites":[]}"#,
            r#"{"markets":[],"sites":[]}"#,
            r#"{"markets":["A"],"sites":[[0,0,1,0,1]]}"#,
            r#"{"sites":[[0,0,1,0,0.5]]}"#,
        ] {
            assert!(IntercityApi::new(bad).is_err(), "accepted {bad}");
        }
    }
}
#[wasm_bindgen]
impl IntercityApi {
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str) -> Result<IntercityApi, String> {
        let pack: MarketPack = serde_json::from_str(header).map_err(|e| e.to_string())?;
        let zm = pack.zone_m as f32;
        let rate = pack.trip_rate;
        let markets = pack.markets.len();
        if markets == 0 || !zm.is_finite() || zm <= 0. || !rate.is_finite() || rate < 0. {
            return Err("invalid market parameters".into());
        }
        let mut city = City {
            name: "Northeast intercity".into(),
            h3: vec![],
            x: vec![],
            y: vec![],
            pop: vec![],
            jobs: vec![],
            gravity: None,
        };
        let mut keys = BTreeMap::new();
        let mut zones: Vec<Zone> = vec![];
        let mut cell_zone = vec![];
        for nums in pack.sites {
            if nums.iter().any(|x| !x.is_finite())
                || nums[2] < 0.
                || nums[3] < 0.
                || nums[4] < 0.
                || nums[4].fract() != 0.
                || nums[4] >= markets as f64
            {
                return Err("invalid intercity weights".into());
            }
            let (x, y, pop, jobs, m) = (
                nums[0] as f32,
                nums[1] as f32,
                nums[2],
                nums[3],
                nums[4] as usize,
            );
            let key = ((x / zm).floor() as i32, (y / zm).floor() as i32, m);
            let z = *keys.entry(key).or_insert_with(|| {
                zones.push(Zone {
                    x: 0.,
                    y: 0.,
                    pop: 0.,
                    attr: 0.,
                    market: m,
                });
                zones.len() - 1
            });
            let attr = pop + 2. * jobs;
            let w = pop + attr;
            zones[z].x += x * w as f32;
            zones[z].y += y * w as f32;
            zones[z].pop += pop;
            zones[z].attr += attr;
            city.x.push(x);
            city.y.push(y);
            city.pop.push(pop as f32);
            city.jobs.push(attr as f32);
            cell_zone.push(z);
        }
        for z in &mut zones {
            let w = (z.pop + z.attr).max(1e-12) as f32;
            z.x /= w;
            z.y /= w;
        }
        let corridors: Vec<_> = (0..markets)
            .flat_map(|a| (a..markets).map(move |b| (a, b)))
            .collect();
        let mut pairs = vec![];
        // Non-commute purpose prevents duplicating local commuting. Smooth geographic onset,
        // including long trips within a metro, avoids a discontinuity at city boundaries.
        let decay = |a: usize, b: usize| {
            let d = (zones[a].x - zones[b].x).hypot(zones[a].y - zones[b].y) / 1000.;
            let onset = ((d - 40.) / 40.).clamp(0., 1.) as f64;
            (
                d,
                onset * (-d as f64 / 180.).exp() / (d as f64).max(40.).powf(1.2),
            )
        };
        // Store only row normalisers, not a full directed matrix or a reverse-pair map.
        let mut totals = vec![0.; zones.len()];
        for a in 0..zones.len() {
            for b in a + 1..zones.len() {
                let f = decay(a, b).1;
                totals[a] += zones[b].attr * f;
                totals[b] += zones[a].attr * f;
            }
        }
        for a in 0..zones.len() {
            for b in a + 1..zones.len() {
                let (d, f) = decay(a, b);
                let ab = if totals[a] > 0. {
                    rate * zones[a].pop * zones[b].attr * f / totals[a]
                } else {
                    0.
                };
                let ba = if totals[b] > 0. {
                    rate * zones[b].pop * zones[a].attr * f / totals[b]
                } else {
                    0.
                };
                let trips = (ab + ba) * 0.5;
                // Prune both directions together after averaging, including pairs whose
                // original flow was above the threshold in only one direction.
                if trips < 0.01 {
                    continue;
                }
                let ma = zones[a].market.min(zones[b].market);
                let mb = zones[a].market.max(zones[b].market);
                let corridor = corridors
                    .iter()
                    .position(|&(x, y)| x == ma && y == mb)
                    .unwrap();
                pairs.push(Pair {
                    a,
                    b,
                    trips,
                    drive: 15. + d * 1.25 / 85. * 60.,
                    corridor,
                });
                pairs.push(Pair {
                    a: b,
                    b: a,
                    trips,
                    drive: 15. + d * 1.25 / 85. * 60.,
                    corridor,
                });
            }
        }
        // The market is immutable after opening. Do not keep geometric-growth capacity
        // (over a million slots for the Northeast's ~668k pairs) in every worker.
        pairs.shrink_to_fit();
        let daily = pairs.iter().map(|x| x.trips).sum();
        let mut by_a = vec![vec![]; zones.len()];
        let mut by_b = vec![vec![]; zones.len()];
        for (i, p) in pairs.iter().enumerate() {
            by_a[p.a].push(i as u32);
            by_b[p.b].push(i as u32);
        }
        for indices in by_a.iter_mut().chain(&mut by_b) {
            indices.shrink_to_fit();
        }
        let mut site_key = HashMap::new();
        for c in 0..city.len() {
            let key = (
                (city.x[c] / SITE_M).floor() as i32,
                (city.y[c] / SITE_M).floor() as i32,
                zones[cell_zone[c]].market,
            );
            site_key.entry(key).or_insert(c);
        }
        let base = Params {
            beta: 0.025,
            asc_rail: 0.,
            fare_min: 0.,
            ..Params::default()
        };
        Ok(Self {
            city,
            zones,
            cell_zone,
            pairs,
            base,
            svc: None,
            access: vec![],
            access_work: vec![],
            want: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            fare_km: vec![],
            fare_per_km: 0.,
            corridors,
            markets,
            daily,
            site_key,
            by_a,
            by_b,
        })
    }
    /// Number of market slots in selection and station flow headers.
    pub fn market_count(&self) -> usize {
        self.markets
    }
    pub fn info(&self) -> Vec<f64> {
        vec![self.city.len() as f64, self.zones.len() as f64, self.daily]
    }
    pub fn set_network(
        &mut self,
        st_xy: &[f32],
        line_n: &[u32],
        stops: &[u32],
        times: &[f32],
        tph: &[f32],
        cars: &[f32],
    ) -> Result<f64, String> {
        let t = now_ms();
        let mut off = vec![0];
        for &n in line_n {
            off.push(off.last().unwrap() + n)
        }
        if st_xy.len() % 2 != 0
            || stops.len() != *off.last().unwrap() as usize
            || times.len() != 6 * stops.len()
            || tph.len() != 3 * cars.len()
            || cars.len() != line_n.len()
            || stops.iter().any(|&s| s as usize >= st_xy.len() / 2)
        {
            return Err("intercity network arrays disagree".into());
        }
        let (st_x, st_y, st_map) = merge_stations(st_xy, &off, stops, self.base.station_merge_m);
        let svc = Service {
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off: off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect(),
            cars: cars.to_vec(),
        };
        let net = DemandApi::period_net(&svc, 0);
        let acc = build_access(&self.city, &net, &self.base);
        let mut weights: Vec<BTreeMap<usize, f32>> =
            (0..self.zones.len()).map(|_| BTreeMap::new()).collect();
        let mut work = weights.clone();
        for c in 0..self.city.len() {
            let z = self.cell_zone[c];
            let w = self.city.pop[c] / self.zones[z].pop.max(1e-12) as f32;
            let jw = self.city.jobs[c] / self.zones[z].attr.max(1e-12) as f32;
            let n = acc.n[c] as usize;
            if n == 0 {
                continue;
            }
            for k in 0..n {
                let i = c * K + k;
                let walk = acc.walk[i];
                let cost = walk.min(self.base.walk_easy_min) * self.base.w_walk
                    + (walk - self.base.walk_easy_min).max(0.) * self.base.w_walk_far;
                let e = (-self.base.beta * cost).exp() / n as f32;
                *weights[z].entry(acc.st[i] as usize).or_default() += w * e;
                *work[z].entry(acc.st[i] as usize).or_default() += jw * e;
            }
        }
        self.access = weights
            .into_iter()
            .map(|m| {
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                v.truncate(8);
                v
            })
            .collect();
        self.access_work = work
            .into_iter()
            .map(|m| {
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                v.truncate(8);
                v
            })
            .collect();
        self.want = vec![false; svc.st_x.len()];
        for list in &self.access_work {
            for &(st, _) in list {
                self.want[st] = true;
            }
        }
        self.svc = Some(svc);
        self.per.iter_mut().for_each(|p| *p = None);
        Ok(now_ms() - t)
    }
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, km: &[f32]) -> Result<(), String> {
        if !base.is_finite()
            || base < 0.
            || !per_km.is_finite()
            || per_km < 0.
            || km.iter().any(|x| !x.is_finite() || *x < 0.)
            || km.len() != self.svc.as_ref().map_or(0, |s| 2 * s.stops.len())
        {
            return Err("invalid intercity fare".into());
        }
        self.base.fare_min = base * FARE_MIN_PER_USD;
        self.fare_per_km = per_km * FARE_MIN_PER_USD;
        self.fare_km = km.to_vec();
        self.per.iter_mut().for_each(|p| *p = None);
        Ok(())
    }
    pub fn solve(&mut self, q: usize) -> bool {
        let Some(svc) = &self.svc else { return false };
        if q >= PERIODS {
            return false;
        }
        let t = now_ms();
        let p = Params {
            period: q,
            period_share: SHARES[q],
            period_hours: (PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.base.clone()
        };
        let net = DemandApi::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        for (r, f) in g.rn_fare.iter_mut().enumerate() {
            *f = self.fare_km.get(r).copied().unwrap_or(0.) * self.fare_per_km;
        }
        // NOTE (T-119, TASKS): unlike the local populations this free-flow estimate routes on ride
        // time alone; the distance fare only enters from the first crowding round (`set_crowding`).
        // Left as it was: fixing it moves the market's rail trips by about 17%.
        let no_crowding = vec![1.0; g.rn_ride.len()];
        let fare_free = g.rn_fare.iter().all(|&f| f == 0.0);
        let pass = self.pass(&g, &p, q);
        self.per[q] = Some(Period {
            net,
            p,
            g,
            avg: pass.loads,
            rail: pass.rail,
            trips: self.daily * SHARES[q] as f64,
            corridors: pass.cor,
            pair_rail: pass.pair_rail,
            st: pass.st,
            round: 0,
            ms: now_ms() - t,
            fac: no_crowding,
            // a crowding round with no crowding only repeats this estimate if no fare was left out of it
            uniform: fare_free,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per(q) else { return false };
        if total.len() != st.avg.seg.len() || total.iter().any(|v| !v.is_finite() || *v < 0.) {
            return false;
        }
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        let mut loads = Loads::zero(st.g.n_st, total.len());
        loads.seg = total.to_vec();
        let (fac, _) = crowd_factors(&st.g, &st.net, &loads, &st.p);
        if st.uniform && fac == st.fac && !legacy() {
            // as `DemandApi::crowd_with_external`: nothing is crowded, the round changes nothing
            st.round += 1;
            st.ms = now_ms() - t;
            self.per[q] = Some(st);
            return true;
        }
        set_crowding(&mut st.g, &fac);
        st.uniform = false;
        st.fac = fac;
        let Pass {
            loads: l,
            rail,
            cor,
            pair_rail,
            st: st_new,
        } = self.pass(&st.g, &st.p, q);
        st.round += 1;
        let r = st.round;
        msa(&mut st.pair_rail, &pair_rail, r);
        for (a, b) in st.st.iter_mut().zip(st_new) {
            *a += (b - *a) / (r as f32 + 1.);
        }
        msa(&mut st.avg.seg, &l.seg, r);
        msa(&mut st.avg.board, &l.board, r);
        msa(&mut st.avg.alight, &l.alight, r);
        msa(&mut st.avg.rn_board, &l.rn_board, r);
        msa(&mut st.avg.rn_alight, &l.rn_alight, r);
        msa(&mut st.avg.entry, &l.entry, r);
        msa(&mut st.avg.exit, &l.exit, r);
        let w = 1. / (r as f64 + 1.);
        st.rail += (rail - st.rail) * w;
        for (a, b) in st.corridors.iter_mut().zip(cor) {
            *a += (b - *a) * w;
        }
        st.ms = now_ms() - t;
        self.per[q] = Some(st);
        true
    }
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let (_, c) = crowd_factors(&s.g, &s.net, &s.avg, &s.p);
            vec![
                s.round as f64,
                s.trips,
                s.rail,
                0.,
                s.ms,
                self.pairs.len() as f64,
                c.segs_over_seats as f64,
                c.max_load_of_crush as f64,
                c.p50 as f64,
                c.p90 as f64,
            ]
        })
    }
    pub fn corridor_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            self.corridors
                .iter()
                .enumerate()
                .flat_map(|(i, &(a, b))| {
                    [
                        a as f64,
                        b as f64,
                        self.pairs
                            .iter()
                            .filter(|p| p.corridor == i)
                            .map(|p| p.trips)
                            .sum::<f64>()
                            * SHARES[q] as f64,
                        s.corridors[i],
                    ]
                })
                .collect()
        })
    }
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.seg.clone())
    }
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_board.clone())
    }
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_alight.clone())
    }
    pub fn entries(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.entry.clone())
    }
    pub fn exits(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.exit.clone())
    }
    /// Bit q set for each period solved.
    pub fn solved_mask(&self) -> u32 {
        (0..PERIODS)
            .filter(|&q| self.per(q).is_some())
            .fold(0, |m, q| m | 1 << q)
    }
    /// Where the longer trips of a selection of local cells go and come from (T-118).
    ///
    /// The selection is points (`xy`: x, y per 600 m square, metres in the New York frame) of one
    /// `market` (the index in the pack's market list) with the `share` (0..1) of that square's people, or
    /// jobs, that are selected. Each point is the endpoint site of the same square (the pipeline
    /// built the sites from the same cells), and a site's share of its zone's departures is its
    /// population over the zone's, of its arrivals its attraction over the zone's. So a zone's
    /// trips are split to the sites by the same weights the market used to give them out, and a
    /// selection that is part of a square counts that part of it.
    ///
    /// Returns, over the periods solved (`solved_mask`): the matched share of the points, then
    /// 8 numbers for each of the markets at the far end: trips starting in the selection that
    /// end there (all modes, by rail, and the all-mode trips times far-zone x and y), then the
    /// same for trips ending in the selection that start there. Trips are a day's share for the
    /// periods solved, so rail is never above all.
    pub fn selection_flows(&self, market: u32, xy: &[f32], share: &[f32]) -> Vec<f64> {
        let nz = self.zones.len();
        let head = 1 + 8 * self.markets;
        let mut out = vec![0.; head + 7 * nz];
        self.write_zone_xy(&mut out[head + 4 * nz..]);
        if xy.len() != 2 * share.len() || market as usize >= self.markets {
            return out;
        }
        let (mut pop_w, mut attr_w) = (vec![0f64; nz], vec![0f64; nz]);
        let mut matched = 0.;
        for (i, &f) in share.iter().enumerate() {
            let key = (
                (xy[2 * i] / SITE_M).floor() as i32,
                (xy[2 * i + 1] / SITE_M).floor() as i32,
                market as usize,
            );
            let Some(&c) = self.site_key.get(&key) else {
                continue;
            };
            let f = (f as f64).clamp(0., 1.);
            matched += 1.;
            let z = self.cell_zone[c];
            pop_w[z] += f * self.city.pop[c] as f64 / self.zones[z].pop.max(1e-12);
            attr_w[z] += f * self.city.jobs[c] as f64 / self.zones[z].attr.max(1e-12);
        }
        out[0] = if share.is_empty() {
            1.
        } else {
            matched / share.len() as f64
        };
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let sh = SHARES[q] as f64;
            for (z, &w) in pop_w.iter().enumerate().filter(|(_, w)| **w > 0.) {
                for &pi in &self.by_a[z] {
                    let p = &self.pairs[pi as usize];
                    let (m, all) = (self.zones[p.b].market, w * p.trips * sh);
                    let o = 1 + 8 * m;
                    out[o] += all;
                    out[o + 1] += w * per.pair_rail[pi as usize] as f64;
                    out[o + 2] += all * self.zones[p.b].x as f64;
                    out[o + 3] += all * self.zones[p.b].y as f64;
                    out[head + 4 * p.b] += all;
                    out[head + 4 * p.b + 1] += w * per.pair_rail[pi as usize] as f64;
                }
            }
            for (z, &w) in attr_w.iter().enumerate().filter(|(_, w)| **w > 0.) {
                for &pi in &self.by_b[z] {
                    let p = &self.pairs[pi as usize];
                    let (m, all) = (self.zones[p.a].market, w * p.trips * sh);
                    let o = 1 + 8 * m + 4;
                    out[o] += all;
                    out[o + 1] += w * per.pair_rail[pi as usize] as f64;
                    out[o + 2] += all * self.zones[p.a].x as f64;
                    out[o + 3] += all * self.zones[p.a].y as f64;
                    out[head + 4 * p.a + 2] += all;
                    out[head + 4 * p.a + 3] += w * per.pair_rail[pi as usize] as f64;
                }
            }
        }
        out
    }
    /// A station's longer-trip rail riders over the periods solved (T-118; `station` indexes the
    /// network handed to `set_network`): per market, riders starting here bound for it, riders
    /// ending here from it, and those counts times the far zone's x and y (starts x, starts y,
    /// ends x, ends y; `market_count()` numbers each). Empty with no such station.
    pub fn station_flows(&self, station: u32) -> Vec<f64> {
        let Some(svc) = &self.svc else { return vec![] };
        let Some(&s) = svc.st_map.get(station as usize) else {
            return vec![];
        };
        let nz = self.zones.len();
        let head = 6 * self.markets;
        let mut out = vec![0.; head + 5 * nz];
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            if let Some(row) = per.st.get(s as usize * 2 * nz..(s as usize + 1) * 2 * nz) {
                for (z, &v) in row.iter().enumerate() {
                    out[head + z] += v as f64;
                }
            }
        }
        // the market sums, from the zones' sums: counts, then counts times the zone's x and y
        for d in 0..2 {
            for z in 0..nz {
                let (v, zone) = (out[head + d * nz + z], &self.zones[z]);
                out[d * self.markets + zone.market] += v;
                out[(2 + 2 * d) * self.markets + zone.market] += v * zone.x as f64;
                out[(3 + 2 * d) * self.markets + zone.market] += v * zone.y as f64;
            }
        }
        self.write_zone_xy(&mut out[head + 2 * nz..]);
        out
    }
    /// Every zone's position (x, y in the New York frame, metres) and exact market membership.
    fn write_zone_xy(&self, out: &mut [f64]) {
        for (z, zone) in self.zones.iter().enumerate() {
            out[3 * z] = zone.x as f64;
            out[3 * z + 1] = zone.y as f64;
            out[3 * z + 2] = zone.market as f64;
        }
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| {
            s.avg
                .seg
                .iter()
                .enumerate()
                .map(|(r, &v)| {
                    let l = s.g.rn_line[r] as usize;
                    let cap = 60. / s.net.line_headway_min[l][q] * s.net.line_crush[l];
                    if cap > 0. {
                        v / s.p.period_hours * s.p.peak_hour_factor / cap
                    } else {
                        0.
                    }
                })
                .collect()
        })
    }
}
