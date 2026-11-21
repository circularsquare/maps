//! T-3887: the mode frontier on a made-up city (notes/T-3887.md). The game's own track model builds one straight
//! line of a mode at a stop spacing and its headways; the game's demand kernel (commutes and everyday trips, its
//! crowding rounds as the demand client runs them) calculates its trips; the trip explorer gives each mode's
//! share of trips of a length along the line. Driven by tools/frontier.py:
//!
//!   cargo run --release --features demand --bin frontier -- --spec spec.json --out result.json
//!
//! The spec: `city` (its size, people and jobs a km², the pack header's country and everyday rates), `fares` (what
//! tools/frontier_fares.ts prints: the game's fares for a new game in the city's country), `line_km`, `rounds`,
//! `trips` (`lengths_km`, `phases`, `offsets_m`) and `runs` (mode, spacing_m, level, cars, platform, tram, hsr, fast,
//! bus, headways s at high, medium and low demand). Each run is a fresh track model; the demand model is opened once
//! and given each run's network, as the game gives it a new one after an edit.
//!
//! The snapshot is packed as clock.worker.ts `buildSnapshot`, demandClient.ts `packService` and demand.worker.ts
//! `setUpPops` do for one city with no fare zones, links or blueprints; income as game/money.ts `incomeFor` takes it
//! for one line (the base on each trip, the rate times the line's multiple on each passenger-km, the curve's extra).

use serde_json::{json, Value};
use trainworld_sim::demand::api::DemandApi;
use trainworld_sim::demand::everyday::EverydayApi;
use trainworld_sim::demand::network::{BUS_CARS, PERIODS};
use trainworld_sim::track::params::{ECONOMY, PERIOD_HOURS};
use trainworld_sim::track::wasm::TrackApi;

/// A cell: the H3 resolution 9 mean area as a square (`demand::synth::CELL_M`).
const CELL_M: f64 = 324.0;
const ZONE_M: f64 = 2000.0;
/// A new train line's dwell and turnaround (app/src/game/newLine.ts `NEW_LINE.train`), and a bus's (`NEW_LINE.bus`).
const TRAIN_DWELL_S: f32 = 30.0;
const TRAIN_TURN_S: f32 = 180.0;
const BUS_DWELL_S: f32 = 30.0;
const BUS_TURN_S: f32 = 120.0;
/// Track kinds (`TrackKind`): tram track.
const KIND_TRAM: u8 = 3;

struct City {
    header: String,
    bin: Vec<u8>,
    // for the street grid
    pop: Vec<f32>,
    jobs: Vec<f32>,
    zone: Vec<u32>,
    gx: Vec<u16>,
    gy: Vec<u16>,
    cells9: Vec<u32>,
    zw: u32,
    zh: u32,
    x0: f64,
    y0: f64,
    lon: f64,
    lat: f64,
}

/// A uniform city of `w_km` by `h_km` about the origin: every cell the same people and jobs.
fn make_city(c: &Value) -> City {
    let f = |k: &str| c[k].as_f64().unwrap_or_else(|| panic!("city.{k}"));
    let (w_km, h_km, res, jobs_km2) = (f("width_km"), f("height_km"), f("residents_km2"), f("jobs_km2"));
    let (commute, errand, leisure) = (f("commuters_per_resident"), f("errand_jobs_per_resident"), f("leisure_jobs_per_resident"));
    let (lon, lat) = (c["origin"][0].as_f64().unwrap(), c["origin"][1].as_f64().unwrap());
    let area = (CELL_M / 1000.0).powi(2);
    let (nx, ny) = ((w_km * 1000.0 / CELL_M).round() as usize, (h_km * 1000.0 / CELL_M).round() as usize);
    let (x0, y0) = (-(nx as f64) * CELL_M / 2.0, -(ny as f64) * CELL_M / 2.0);
    let (zw, zh) = (((nx as f64 * CELL_M) / ZONE_M).ceil() as u32, ((ny as f64 * CELL_M) / ZONE_M).ceil() as u32);
    let n = nx * ny;
    let (mut h3, mut xs, mut ys, mut zone) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut zone_of = vec![u32::MAX; (zw * zh) as usize];
    let (mut gx, mut gy, mut cells9) = (vec![], vec![], vec![]);
    for iy in 0..ny {
        for ix in 0..nx {
            let (x, y) = (x0 + (ix as f64 + 0.5) * CELL_M, y0 + (iy as f64 + 0.5) * CELL_M);
            h3.push((iy * nx + ix) as u64);
            xs.push(x as f32);
            ys.push(y as f32);
            let (sx, sy) = (((x - x0) / ZONE_M) as u32, ((y - y0) / ZONE_M) as u32);
            let k = (sx * zh + sy) as usize;
            if zone_of[k] == u32::MAX {
                zone_of[k] = gx.len() as u32;
                gx.push(sx as u16);
                gy.push(sy as u16);
                cells9.push(0);
            }
            cells9[zone_of[k] as usize] += 1;
            zone.push(zone_of[k]);
        }
    }
    let per = |v: f64| vec![(v * area) as f32; n];
    let (pop, jobs) = (per(res), per(jobs_km2));
    let mut arrays: Vec<Value> = vec![];
    let mut bin: Vec<u8> = vec![];
    let mut push = |name: &str, dtype: &str, bytes: Vec<u8>| {
        arrays.push(json!({"name": name, "dtype": dtype, "per": "cell", "offset": bin.len(), "count": n}));
        bin.extend(bytes);
    };
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    push("h3", "u64", h3.iter().flat_map(|v| v.to_le_bytes()).collect());
    push("x_m", "f32", f32s(&xs));
    push("y_m", "f32", f32s(&ys));
    push("pop", "f32", f32s(&pop));
    push("jobs", "f32", f32s(&jobs));
    push("commute_home", "f32", f32s(&per(res * commute)));
    push("commute_work", "f32", f32s(&per(res * commute)));
    push("errand_jobs", "f32", f32s(&per(res * errand)));
    push("leisure_jobs", "f32", f32s(&per(res * leisure)));
    let header = json!({
        "format": 1, "city": "frontier", "built": "synthetic", "origin": {"lon": lon, "lat": lat}, "h3_res": 9, "cells": n,
        "arrays": arrays, "country": c["country"], "everyday": {"country": c["country"], "rate": c["everyday_rate"]},
        "totals": {"pop": res * area * n as f64, "jobs": jobs_km2 * area * n as f64},
    });
    City { header: header.to_string(), bin, pop, jobs, zone, gx, gy, cells9, zw, zh, x0, y0, lon, lat }
}

fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}

/// The line of `run` along y = 0, `line_km` long and centred on the origin, built, constructed and given its
/// headways. Returns the track model, the line and what building it and its fleet cost, US$M.
fn build(city: &City, run: &Value, line_km: f64) -> Result<(TrackApi, u32, f64), String> {
    let mut a = TrackApi::new(city.lon, city.lat, true);
    let ok = a.add_pack_streets(ZONE_M, city.x0, city.y0, city.zw, city.zh, &city.pop, &city.jobs, &city.zone, &city.gx, &city.gy, &[], &city.cells9, city.lon, city.lat);
    if !ok {
        return Err("street grid refused".into());
    }
    let cash0 = 1e9;
    a.set_cash(cash0);
    let half = line_km * 500.0;
    let hw: Vec<u32> = run["headways"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let line = a.new_line_id();
    if run["bus"].as_bool().unwrap_or(false) {
        // points every 100 m along the street, each stretch on the road, the ends clicked (as tools/frontier.js)
        let n = (line_km * 10.0).round() as usize;
        let mut pts = vec![];
        for k in 0..=n {
            let kind = (if k == 0 || k == n { 1.0 } else { 0.0 }) + if k > 0 { 2.0 } else { 0.0 };
            pts.extend([-half + 2.0 * half * k as f64 / n as f64, 0.0, kind]);
        }
        if a.set_bus_line(line, &pts, hw[0], hw[1], hw[2], BUS_DWELL_S, BUS_TURN_S, 0xc0392b, "Bus".into()) != 0 {
            return Err(format!("bus line: {}", a.issue_kinds()));
        }
        let sp = num(run, "spacing_m");
        if sp > 400.0 && a.set_bus_spacing(line, sp) != 0 {
            return Err(format!("bus spacing: {}", a.issue_kinds()));
        }
        if a.construct_line(line) != 0 {
            return Err(format!("construct: {}", a.issue_kinds()));
        }
    } else {
        let sp = num(run, "spacing_m");
        let gaps = ((line_km * 1000.0 / sp).round() as usize).max(1);
        let level = num(run, "level");
        let tram = run["tram"].as_bool().unwrap_or(false);
        a.set_route_kind(if tram { KIND_TRAM } else { 0 });
        a.set_route_hsr(run["hsr"].as_bool().unwrap_or(false));
        a.set_design_speed(if run["hsr"].as_bool().unwrap_or(false) { 320.0 } else { 160.0 });
        let x = |k: usize| -half + 2.0 * half * k as f64 / gaps as f64;
        let mut nodes = vec![];
        for k in 1..=gaps {
            let from = if k == 1 { [0.0, 0.0, x(0), 0.0, level] } else { [1.0, nodes[k - 1] as f64, 0.0, 0.0, 0.0] };
            let ends = [from[0], from[1], from[2], from[3], from[4], 0.0, 0.0, x(k), 0.0, level];
            if a.add_route(&ends, &[], 2, false) != 0 {
                return Err(format!("track {k}: {}", a.issue_kinds()));
            }
            if k == 1 {
                // the first route's start node: the live node nearest (x(0), 0)
                let ni = a.nodes_info();
                let first = ni.chunks_exact(10).min_by(|p, q| (p[1] - x(0)).hypot(p[2]).total_cmp(&(q[1] - x(0)).hypot(q[2]))).unwrap()[0] as u32;
                nodes.push(first);
            }
            nodes.push(a.route_end());
        }
        let platform = num(run, "platform") as u16;
        for (k, &nd) in nodes.iter().enumerate() {
            if a.add_station(1, nd, 0.0, 0.0, platform, format!("S{k}")) != 0 {
                return Err(format!("station {k}: {}", a.issue_kinds()));
            }
        }
        if a.construct_all() != 0 {
            return Err(format!("construct: {}", a.issue_kinds()));
        }
        let (dwell, turn) = if tram {
            let d = TrackApi::kind_defaults(KIND_TRAM);
            (d[0] as f32, d[1] as f32)
        } else {
            (TRAIN_DWELL_S, TRAIN_TURN_S)
        };
        // a run may set its own dwell (a what-if, as a player can per line)
        let dwell = run["dwell_s"].as_f64().map_or(dwell, |v| v as f32);
        let cars = num(run, "cars") as u8;
        if a.set_line_with_trains(line, &nodes, 0, 0, 0, dwell, turn, cars, 0x0a7d3b, "Line".into()) != 0 {
            return Err(format!("line: {}", a.issue_kinds()));
        }
        if run["fast"].as_bool().unwrap_or(false) && a.set_line_fast(line, true) != 0 {
            return Err(format!("fast: {}", a.issue_kinds()));
        }
    }
    if a.set_auto_headway(line, hw[0], hw[1], hw[2]) != 0 {
        return Err(format!("headways: {}", a.issue_kinds()));
    }
    let spent = cash0 - a.cash();
    Ok((a, line, spent))
}

/// What demand is given for the one line (buildSnapshot + packService): station x, y; stops; times (10 a stop);
/// trains an hour; cars; hop km per route node; grade flags per stop; bus catches.
struct Snap {
    xy: Vec<f32>,
    line_n: Vec<u32>,
    stops: Vec<u32>,
    times: Vec<f32>,
    tph: Vec<f32>,
    cars: Vec<f32>,
    hop_km: Vec<f32>,
    grade: Vec<u8>,
    catch_n: Vec<u32>,
    catch_xy: Vec<f32>,
    tram: bool,
    bus: bool,
    fast: bool,
}

fn snapshot(a: &TrackApi, id: u32) -> Snap {
    let ni = a.nodes_info();
    let (mut ids, mut xy, mut level) = (vec![], vec![], vec![]);
    for p in ni.chunks_exact(10) {
        if p[7] > 0.0 && p[8] == 1.0 {
            ids.push(p[0] as u32);
            xy.extend([p[1] as f32, p[2] as f32]);
            level.push(p[3]);
        }
    }
    let bus = a.is_bus(id);
    let (mut catch_n, mut catch_xy) = (vec![], vec![]);
    if bus {
        // its stops that are no station are stations of their own, at grade
        let st = a.bus_stops(id);
        for c in st.chunks_exact(8) {
            if c[5] < 0.0 && !ids.contains(&(c[0] as u32)) {
                ids.push(c[0] as u32);
                xy.extend([c[1] as f32, c[2] as f32]);
                level.push(0.0);
            }
        }
        let c = a.bus_catches(id);
        let mut k = 0;
        while k < c.len() {
            let n = c[k] as usize;
            catch_n.push(n as u32);
            catch_xy.extend(c[k + 1..k + 1 + 2 * n].iter().map(|&v| v as f32));
            k += 1 + 2 * n;
        }
    }
    let stops_id = a.line_run_stops(id);
    let n = stops_id.len();
    let stops: Vec<u32> = stops_id.iter().map(|s| ids.iter().position(|i| i == s).expect("a stop that is no station") as u32).collect();
    let mut times = vec![0f32; 2 * PERIODS * n];
    for q in 0..PERIODS {
        for run in 0..2 {
            let (arr, dep) = (a.stop_times_in(id, run, q), a.stop_departures_in(id, run, q));
            let at = (run * PERIODS + q) * n;
            for j in 0..n - 1 {
                let dwell = if j == 0 { 0.0 } else { dep[j] - arr[j] };
                let v = (arr[j + 1] - dep[j] + dwell) as f32;
                // run 1 travels the stops reversed: its j-th hop leaves stop n - 1 - j
                let k = if run == 0 { j } else { n - 1 - j };
                times[at + k] = v;
            }
        }
    }
    let mut hop_km = vec![0f32; 2 * n];
    for run in 0..2 {
        let s = a.line_stop_s(id, run);
        for k in 0..n - 1 {
            hop_km[run * n + k] = ((s[k + 1] - s[k]) / 1000.0) as f32;
        }
    }
    let li = a.lines_info();
    let row = li.chunks_exact(22).find(|r| r[0] as u32 == id).unwrap();
    let tram = !bus && a.line_tram(id);
    let fast = !bus && a.line_fast(id);
    let grade = stops.iter().map(|&s| (level[s as usize] == 0.0) as u8).collect();
    let cars = if bus { BUS_CARS } else { row[21] as f32 };
    if catch_n.len() != n && bus {
        panic!("bus catches: {} for {n} stops", catch_n.len());
    }
    Snap { xy, line_n: vec![n as u32], stops, times, tph: a.line_demand_tph(id), cars: vec![cars], hop_km, grade, catch_n, catch_xy, tram, bus, fast }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(get("--spec").expect("--spec")).unwrap()).unwrap();
    let out_path = get("--out").expect("--out");
    let t0 = std::time::Instant::now();
    let city = make_city(&spec["city"]);
    let mut api = DemandApi::new(&city.header, &city.bin).unwrap();
    let mut ev = EverydayApi::beside_with(&api, &city.header, &city.bin, &[]).unwrap();
    let info = api.info();
    eprintln!("city: {} cells, {} zones, {:.0} commutes a day, opened in {:.1} s", info[0], info[1], info[2], t0.elapsed().as_secs_f64());
    let fares = &spec["fares"];
    let (base, per_km) = (num(fares, "base") as f32, num(fares, "perKm") as f32);
    let knots: Vec<f32> = fares["knots"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let line_km = num(&spec, "line_km");
    let rounds = spec["rounds"].as_u64().unwrap_or(3) as usize;
    let trips = &spec["trips"];
    let lengths: Vec<f64> = trips["lengths_km"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let offsets: Vec<f64> = trips["offsets_m"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    // each offset's weight, the share of the people it stands for (default even)
    let weights: Vec<f64> = trips["offset_weights"].as_array().map_or(vec![1.0; offsets.len()], |a| a.iter().map(|v| v.as_f64().unwrap()).collect());
    let phases = trips["phases"].as_u64().unwrap_or(10) as usize;
    let mut results = vec![];
    for run in spec["runs"].as_array().unwrap() {
        let t1 = std::time::Instant::now();
        let name = format!("{} {} m {:?}", run["mode"], run["spacing_m"], run["headways"]);
        let (a, id, spent) = match build(&city, run, line_km) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{name}: {e}");
                results.push(json!({"run": run, "error": e}));
                continue;
            }
        };
        let s = snapshot(&a, id);
        let n = s.stops.len();
        let mul = if s.bus { num(fares, "bus") } else if s.fast { num(fares, "hsr") } else { 1.0 } as f32;
        let fare_mul = if mul != 1.0 { vec![mul; 2 * n] } else { vec![] };
        // setUpPops, in its order
        api.set_tram_lines(&[s.tram as u8]);
        ev.set_tram_lines(&[s.tram as u8]);
        api.set_stairs(&[], &s.grade);
        ev.set_stairs(&[], &s.grade);
        api.set_chunked(true);
        api.set_fare_weight(num(fares, "weight") as f32).unwrap();
        ev.set_fare_weight(num(fares, "weight") as f32).unwrap();
        api.set_bus(&s.catch_n, &s.catch_xy);
        ev.set_bus(&s.catch_n, &s.catch_xy);
        api.set_network(&s.xy, &s.line_n, &s.stops, &s.times, &s.tph, &s.cars).unwrap();
        ev.set_network(&s.xy, &s.line_n, &s.stops, &s.times, &s.tph, &s.cars).unwrap();
        api.set_fares(base, per_km, &s.hop_km).unwrap();
        ev.set_fares(base, per_km, &s.hop_km).unwrap();
        api.set_fare_mul(&fare_mul).unwrap();
        ev.set_fare_mul(&fare_mul).unwrap();
        api.set_fare_curve(&knots).unwrap();
        ev.set_fare_curve(&knots).unwrap();
        api.set_transfer_fare(num(fares, "transfer") as f32).unwrap();
        ev.set_transfer_fare(num(fares, "transfer") as f32).unwrap();

        // a weekday's periods, crowding rounds shared between the two layers, then held to the trains' room (calib.rs)
        let (mut board_day, mut pkm_day, mut extra_day, mut peak_hour) = (0f64, 0f64, 0f64, 0f64);
        let mut by_period = vec![];
        for q in 0..PERIODS {
            api.solve(q);
            ev.solve(q);
            let mut free: Vec<f32> = vec![];
            for round in 0..=rounds {
                let nr = api.seg(q).len();
                let mut total = vec![0f32; 3 * nr];
                for (v, at) in [(api.seg(q), 0), (api.board(q), 1), (api.alight(q), 2), (ev.seg(q), 0), (ev.board(q), 1), (ev.alight(q), 2)] {
                    for (t, x) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                        *t += x;
                    }
                }
                if round == 0 {
                    free = total[nr..2 * nr].to_vec();
                }
                if round == rounds {
                    if rounds > 0 {
                        let come: Vec<f32> = free.iter().zip(&total[nr..2 * nr]).map(|(a, b)| a.max(*b)).collect();
                        total.extend(come);
                        assert!(api.cap_with_external(q, &total));
                        assert!(ev.cap_with_external(q, &total));
                    }
                    break;
                }
                assert!(api.crowd_with_external(q, &total));
                assert!(ev.crowd_with_external(q, &total));
            }
            let sum = |v: Vec<f32>| v.iter().map(|&x| x as f64).sum::<f64>();
            let seg: Vec<f64> = api.seg(q).iter().zip(ev.seg(q)).map(|(a, b)| (*a + b) as f64).collect();
            let board = sum(api.board(q)) + sum(ev.board(q));
            let pkm: f64 = seg.iter().zip(&s.hop_km).map(|(l, k)| l * *k as f64).sum();
            let extra = sum(api.fare_extra(q)) + sum(ev.fare_extra(q));
            let hours = PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0;
            let busiest = seg.iter().cloned().fold(0.0, f64::max) / hours;
            peak_hour = peak_hour.max(busiest);
            board_day += board;
            pkm_day += pkm;
            extra_day += extra;
            by_period.push(json!({"board": board, "pkm": pkm, "busiest_per_hour": busiest, "commute": api.summary(q), "everyday": ev.summary(q)}));
        }
        // money a weekday, US$ at the game's economy (x100): fares as incomeFor (the base on each ride, one line so
        // every trip is one ride; the rate times the line's multiple on each passenger-km; the curve's extra)
        let fares_day = (base as f64 * board_day + mul as f64 * per_km as f64 * pkm_day + extra_day) * ECONOMY;
        let running_day: f64 = (0..PERIODS).map(|q| a.running_cost_period(q) * (PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0)).sum::<f64>() * 1e6;
        let upkeep_day = a.upkeep()[0] * 1e6;
        let li = a.lines_info();
        let row: Vec<f64> = li.chunks_exact(22).find(|r| r[0] as u32 == id).unwrap().to_vec();

        // the trip explorer, the morning peak's commutes: trips of each length along the line, both ends `offset` m to
        // one side of it (a row of cells), starts spread over one stop spacing about the line's middle
        let sp = num(run, "spacing_m").max(400.0);
        let mut shares = vec![];
        for &len in &lengths {
            let (mut w, mut t, mut d, mut cnt) = (0f64, 0f64, 0f64, 0f64);
            let mut by_off = vec![];
            for (&off, &wt) in offsets.iter().zip(&weights) {
                let (t_was, c_was) = (t, cnt);
                for k in 0..phases {
                    let ox = -len * 500.0 + sp * (k as f64 + 0.5) / phases as f64 - sp / 2.0;
                    let v = api.trip(0, ox as f32, off as f32, (ox + len * 1000.0) as f32, off as f32);
                    if v[0] != 1.0 {
                        continue;
                    }
                    w += wt * v[17];
                    t += wt * v[18];
                    d += wt * v[19];
                    cnt += wt;
                }
                by_off.push((t - t_was) / (cnt - c_was).max(1e-9));
            }
            let c = cnt.max(1e-9);
            shares.push(json!({"trip_km": len, "walk": w / c, "transit": t / c, "car": d / c, "n": cnt, "transit_by_offset": by_off}));
        }
        let el = t1.elapsed().as_secs_f64();
        eprintln!("{name}: {n} stops, {:.0} riders, {:.0} pkm, fares {:.2}M running {:.2}M upkeep {:.2}M built {:.0}M, transit share at {} km {:.3} ({el:.1} s)",
            board_day, pkm_day, fares_day / 1e6, running_day / 1e6, upkeep_day / 1e6, spent, lengths[1.min(lengths.len() - 1)], shares[1.min(shares.len() - 1)]["transit"].as_f64().unwrap_or(0.0));
        results.push(json!({
            "run": run, "stops": n, "line_km": row[9] / 1000.0, "cars": s.cars[0], "tph": s.tph, "round_trip_s": [row[10], row[11], row[12]],
            "trains_needed": [row[13], row[14], row[15]], "riders_day": board_day, "pkm_day": pkm_day, "busiest_per_hour": peak_hour,
            "fares_day": fares_day, "fare_extra_day": extra_day * ECONOMY, "running_day": running_day, "upkeep_day": upkeep_day, "built": spent * 1e6,
            "fare_mul": mul, "one_way_s": s.times[..n - 1].iter().sum::<f32>(), "periods": by_period, "shares": shares, "seconds": el,
        }));
        std::fs::write(&out_path, serde_json::to_string(&json!({"spec": spec, "fares": fares, "results": results})).unwrap()).unwrap();
    }
    eprintln!("-> {out_path} ({:.0} s)", t0.elapsed().as_secs_f64());
}
