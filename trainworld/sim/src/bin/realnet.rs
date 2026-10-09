//! T-007: solve a day of demand on a network given as `DemandApi::set_network`'s arrays (the
//! real New York network from `pipeline/realnet.py net`, or any other test network), and write
//! what the comparison with measured ridership needs.
//!
//!   cargo run --release --features demand --bin realnet -- --net data/work/realnet/nyc_real.json
//!       --out data/work/realnet/result.json [--pack data/packs/nyc.json] [--rounds 1]
//!       [--counties data/work/realnet/cell_county.u8] [--asc 1.5] [--theta 0.2] [--dump-synth PATH]
//!       [--walk-m 3000] [--walk-mps 4] [--w-walk 1.5] [--xfer-m 300] [--set name=value,..]
//!
//! `--set road_kmh=28,walk_easy_min=99,walk_cutoff_m=2500,pair_walk_max=1e9` is the model before
//! T-090 (one road speed everywhere, a linear walk cost with a hard edge at 2.5 km). The log also
//! gives rail trips by access distance and by both walks together (how much sits near the bounds).
//!
//! The network JSON needs a `demand` object with `st_xy`, `line_n`, `stops`, `times`, `tph` and
//! `cars` (see `DemandApi::set_network`). Output: per period and round the summary
//! (`DemandApi::summary`), then per period the averaged station entries, exits, boardings and
//! alightings, segment loads and loads at the busiest hour as a share of crush, and with
//! `--counties` (one byte per pack cell, 255 = none) rail trips and all trips by home county.

use serde_json::{json, Value};
use trainworld_sim::demand::api::DemandApi;
use trainworld_sim::demand::clock::now_ms;
use trainworld_sim::demand::kernel::{DIST_BANDS_KM, N_BANDS, N_WALKSUM, WALKSUM_BAND_MIN};
use trainworld_sim::demand::network::PERIODS;
use trainworld_sim::demand::pack;

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap_or_else(|| panic!("network: no {k}")).iter().map(|x| x.as_u64().unwrap() as u32).collect()
}
fn round_vec(v: &[f32], scale: f32) -> Vec<f64> {
    v.iter().map(|&x| ((x * scale) as f64 * 10.0).round() / 10.0).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let pack_path = get("--pack").unwrap_or(format!("{root}/data/packs/nyc.json"));
    let rounds: usize = get("--rounds").map_or(1, |v| v.parse().unwrap());
    if let Some(p) = get("--dump-synth") {
        // T-005's hand-made network (the app's ?demandTest=1) in this tool's network format
        let net = trainworld_sim::demand::network::synth_network();
        let xy: Vec<f32> = (0..net.n_stations()).flat_map(|s| [net.st_x[s], net.st_y[s]]).collect();
        let n: Vec<u32> = (0..net.n_lines()).map(|l| net.stops(l).len() as u32).collect();
        let mut times = vec![];
        for l in 0..net.n_lines() {
            let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
            for run in 0..2 {
                for _ in 0..3 {
                    for k in a..b {
                        times.push(60.0 * if run == 0 { net.hop_min[k] } else if k > a { net.hop_min[k - 1] } else { 0.0 });
                    }
                }
            }
        }
        let tph: Vec<f32> = net.line_headway_min.iter().flat_map(|h| [60.0 / h[0], 60.0 / h[1], 60.0 / h[4]]).collect();
        let cars: Vec<f32> = net.line_crush.iter().map(|c| (c / 160.0).round()).collect();
        let out = json!({"demand": {"st_xy": xy, "line_n": n, "stops": net.line_stops, "times": times, "tph": tph, "cars": cars}});
        std::fs::write(&p, serde_json::to_string(&out).unwrap()).unwrap();
        eprintln!("-> {p}");
        return;
    }
    let net_path = get("--net").expect("--net <network json>");
    let out_path = get("--out").expect("--out <result json>");

    let t0 = now_ms();
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    {
        let p = api.params_mut();
        if let Some(v) = get("--asc") {
            p.asc_rail = v.parse().unwrap();
        }
        if let Some(v) = get("--theta") {
            p.station_theta = v.parse().unwrap();
        }
        if let Some(v) = get("--walk-m") {
            p.walk_cutoff_m = v.parse().unwrap();
        }
        if let Some(v) = get("--walk-mps") {
            p.walk_mps = v.parse().unwrap();
        }
        if let Some(v) = get("--w-walk") {
            p.w_walk = v.parse().unwrap();
        }
        if let Some(v) = get("--xfer-m") {
            p.xfer_walk_m = v.parse().unwrap();
        }
        if let Some(v) = get("--merge-m") {
            p.station_merge_m = v.parse().unwrap();
        }
        if let Some(v) = get("--alpha") {
            p.crowd_alpha = v.parse().unwrap();
        }
        // any other f32 parameter: --set name=value[,name=value]
        if let Some(v) = get("--set") {
            for kv in v.split(',') {
                let (k, x) = kv.split_once('=').expect("--set name=value");
                let x: f32 = x.parse().unwrap();
                match k {
                    "w_drive" => p.w_drive = x,
                    "park_min" => p.park_min = x,
                    "park_cost_cap_min" => p.park_cost_cap_min = x,
                    "park_jobs_per_km2_per_min" => p.park_jobs_per_km2_per_min = x,
                    "fare_min" => p.fare_min = x,
                    "beta" => p.beta = x,
                    "transfer_min" => p.transfer_min = x,
                    "walk_d0_km" => p.walk_d0_km = x,
                    "walk_s_km" => p.walk_s_km = x,

                    _ => p.set_extra(k, x).unwrap_or_else(|e| panic!("{e}")),
                }
            }
        }
        eprintln!("params: {:?}", p);
    }
    let info = api.info();
    eprintln!("open {:.0} ms: {} cells, {} zones, {:.0} commutes a day", now_ms() - t0, info[0], info[1], info[2]);

    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let d = &net["demand"];
    let ms = api
        .set_network(&arr_f32(d, "st_xy"), &arr_u32(d, "line_n"), &arr_u32(d, "stops"), &arr_f32(d, "times"), &arr_f32(d, "tph"), &arr_f32(d, "cars"))
        .unwrap();
    let n_st = arr_f32(d, "st_xy").len() / 2;
    eprintln!("set_network {ms:.0} ms: {n_st} stations, {} lines", arr_u32(d, "line_n").len());

    let counties: Option<Vec<u8>> = get("--counties").map(|p| std::fs::read(p).unwrap());
    let n_cty = counties.as_ref().map_or(0, |c| c.iter().filter(|&&v| v != 255).map(|&v| v as usize + 1).max().unwrap_or(0));

    let mut periods = vec![];
    let (mut day_trips, mut day_rail, mut day_walk) = (0.0, 0.0, 0.0);
    let (mut band_h, mut band_w) = (vec![0f64; N_BANDS], vec![0f64; N_BANDS]);
    let mut stage_ms = vec![];
    let mut walksum = vec![0f64; N_WALKSUM];
    for q in 0..PERIODS {
        // per round: the summary, plus how much the averaged segment loads moved (sum of absolute
        // changes over the sum of loads) and the ms the round took
        let mut rounds_out = vec![];
        api.solve(q);
        let mut s0 = api.summary(q);
        s0.push(f64::NAN);
        rounds_out.push(s0);
        let mut prev = api.seg(q);
        for _ in 0..rounds {
            api.crowd(q);
            let seg = api.seg(q);
            let moved: f64 = seg.iter().zip(&prev).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / seg.iter().map(|&a| a as f64).sum::<f64>().max(1.0);
            let mut s = api.summary(q);
            s.push(moved);
            rounds_out.push(s);
            prev = seg;
        }
        let s = rounds_out.last().unwrap().clone();
        eprintln!(
            "period {q}: trips {:.0}, rail {:.0} ({:.1}%), walk {:.0}; {:.1}M pairs; worst {:.0}% of crush; p50 {:.2} p90 {:.2}; rail by round {:?}",
            s[1], s[2], 100.0 * s[2] / s[1], s[3], s[5] / 1e6, 100.0 * s[7], s[8], s[9],
            rounds_out.iter().map(|r| (r[2] / 1000.0).round() as i64).collect::<Vec<_>>()
        );
        day_trips += s[1];
        day_rail += s[2];
        day_walk += s[3];
        stage_ms.extend(rounds_out.iter().map(|r| r[4]));
        let (bh, bw) = api.access_bands(q);
        for b in 0..bh.len() {
            band_h[b] += bh[b];
            band_w[b] += bw[b];
        }
        for (b, v) in api.walksum_bands(q).into_iter().enumerate() {
            walksum[b] += v;
        }
        let (board, alight) = api.station_board(q);
        let mut by_cty = Value::Null;
        if let Some(c) = &counties {
            let rail = api.rail_by_cell(q);
            let trips = api.trips_by_cell(q);
            let (mut r, mut t) = (vec![0f64; n_cty], vec![0f64; n_cty]);
            for (i, &k) in c.iter().enumerate() {
                if k != 255 && i < rail.len() {
                    r[k as usize] += rail[i] as f64;
                    t[k as usize] += trips[i] as f64;
                }
            }
            by_cty = json!({"rail": r, "trips": t});
        }
        periods.push(json!({
            "rounds": rounds_out,
            "entries": round_vec(&api.entries(q), 1.0),
            "exits": round_vec(&api.exits(q), 1.0),
            "board": round_vec(&board, 1.0),
            "alight": round_vec(&alight, 1.0),
            "seg": round_vec(&api.seg(q), 1.0),
            "load_of_crush": api.load_of_crush(q).iter().map(|&v| (v as f64 * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            "county": by_cty,
        }));
    }
    let (rn_station, rn_line) = api.route_nodes(0);
    eprintln!(
        "day: {:.0} trips, rail {:.0} ({:.1}%), walk {:.1}%; {} subzones; {:.0} s",
        day_trips,
        day_rail,
        100.0 * day_rail / day_trips,
        100.0 * day_walk / day_trips,
        api.n_subzones(),
        (now_ms() - t0) / 1000.0
    );
    // T-090: rail trips by distance from home (and work) to the station, % per band
    let (th, tw) = (band_h.iter().sum::<f64>().max(1e-9), band_w.iter().sum::<f64>().max(1e-9));
    let mut lo = 0.0;
    let mut line = String::new();
    for (b, &hi) in DIST_BANDS_KM.iter().enumerate() {
        line.push_str(&format!(" {lo}-{}: {:.1}/{:.1};", if hi.is_finite() { hi.to_string() } else { "".into() }, 100.0 * band_h[b] / th, 100.0 * band_w[b] / tw));
        lo = hi;
    }
    eprintln!("access km, % of rail at home/work end:{line}");
    let ws = walksum.iter().sum::<f64>().max(1e-9);
    let line: String = walksum.iter().enumerate().map(|(b, v)| format!(" {}: {:.2};", b as f32 * WALKSUM_BAND_MIN, 100.0 * v / ws)).collect();
    eprintln!("both walks, perceived min (band start), % of rail:{line}");
    stage_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("stage ms: min {:.0} median {:.0} max {:.0}; sum {:.0}", stage_ms[0], stage_ms[stage_ms.len() / 2], stage_ms[stage_ms.len() - 1], stage_ms.iter().sum::<f64>());
    // the demand views (T-078): per-cell modes for the day and the busiest station's riders
    let t1 = now_ms();
    let sums = api.sub_sums();
    let cm = api.cell_modes(&sums);
    let t2 = now_ms();
    let n = cm.len() / 6;
    let tot = |b: usize| cm[b * n..(b + 1) * n].iter().map(|&v| v as f64).sum::<f64>();
    let entries: Vec<f32> = (0..PERIODS).map(|q| api.entries(q)).fold(vec![], |a: Vec<f32>, e| if a.is_empty() { e } else { a.iter().zip(&e).map(|(x, y)| x + y).collect() });
    let st_map = api.station_map();
    let busiest = (0..st_map.len()).max_by(|&i, &j| entries[st_map[i] as usize].partial_cmp(&entries[st_map[j] as usize]).unwrap()).unwrap_or(0);
    let t3 = now_ms();
    let sr = api.station_riders(busiest as u32);
    let t4 = now_ms();
    let (nh, nw) = (sr[0] as usize, sr[1] as usize);
    let sum = |a: usize, k: usize| sr[a..a + k].iter().map(|&v| v as f64).sum::<f64>();
    eprintln!(
        "views: cell modes {:.0} ms ({n} cells; home rail {:.0} walk {:.0} drive {:.0}; work rail {:.0} walk {:.0} drive {:.0} commuters); \
         station {busiest} riders {:.1} ms: {nh} home cells {:.0}, {nw} work cells {:.0}, to zones {:.0}, from zones {:.0}",
        t2 - t1, tot(0), tot(1), tot(2), tot(3), tot(4), tot(5), t4 - t3,
        sum(2 + nh, nh), sum(2 + 2 * nh + nw, nw), sum(2 + 2 * nh + 2 * nw, api.info()[1] as usize), sum(2 + 2 * nh + 2 * nw + api.info()[1] as usize, api.info()[1] as usize)
    );
    let out = json!({
        "net": net_path,
        "params": format!("{:?}", api.params_mut()),
        "day": {"trips": day_trips, "rail": day_rail, "walk": day_walk, "subzones": api.n_subzones(), "band_home": band_h, "band_work": band_w, "walksum": walksum},
        "rn_station": rn_station,
        "rn_line": rn_line,
        "st_map": api.station_map(),
        "periods": periods,
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("-> {out_path}");
}
