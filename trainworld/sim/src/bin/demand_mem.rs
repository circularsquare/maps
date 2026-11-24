//! T-2496: what a solving demand worker's heap holds on the real New York network, stage by stage, natively
//! (the counting allocator). Opens the real save's zones fine with their everyday layers, the regional
//! market of the market packs around New York, gives every population the real network and solves the
//! five periods with crowding rounds, as `demand.worker.ts` does.
//!
//!   cargo run --release --features demand --bin demand_mem -- [--zones nyc,nhv,hfd,lck]
//!       [--market nyc,phl,pvd,nhv,spf,wor,hfd,lhg,lck] [--rounds 2] [--parts] [--net nyc] [--snapshot FILE]
//!
//! `--net tyo` reads `data/work/realnet/tyo_real_track.json` in Tokyo's frame (an empty `--market ""` is a phone's
//! solve, which has no market).
//!
//! `--parts` prints each population's structures (`mem_parts`) after the solve; `--stages` marks each population's
//! solve and each crowding round inside a period.
//!
//! `--app-order` solves as the game's worker does a cluster with no market (a phone's start): round by round over
//! the five periods, the skims released after each period's stage (period by period without it, which keeps
//! fewer periods held at a round's peak). Its rail trips differ from the default order's (it crowds on the
//! boardings too and holds the last round to the trains' room), so compare totals within one order.
//! T-3160: on the Tokyo real-network start (`--zones tyo,tcg,gnm,yns,szo --market "" --net tyo --app-order`)
//! the peak is a crowding round's period 4: every period held, the crowded graph's skims, one pass's working set.

use serde_json::Value;
use trainworld_sim::demand::alloc;
use trainworld_sim::demand::api::{release_skims, DemandApi};
use trainworld_sim::demand::clock::now_ms;
use trainworld_sim::demand::everyday::EverydayApi;
use trainworld_sim::demand::intercity::{IntercityApi, MarketPacks};
use trainworld_sim::demand::network::{RealNet, PERIODS};
use trainworld_sim::demand::pack;
use trainworld_sim::track::frame::Frame;

struct Stage {
    last: usize,
    t: f64,
}

impl Stage {
    fn mark(&mut self, what: &str) {
        let (now, peak) = (alloc::current(), alloc::peak());
        println!(
            "{what:<44} live {:>7.1} MB  {:>+7.1}  peak since last {:>7.1} MB  {:>6.0} ms",
            alloc::mb(now),
            alloc::mb(now) - alloc::mb(self.last),
            alloc::mb(peak),
            now_ms() - self.t
        );
        self.last = now;
        self.t = now_ms();
        alloc::reset_peak();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let zones: Vec<String> = get("--zones").unwrap_or("nyc,nhv,hfd,lck".into()).split(',').map(String::from).collect();
    let markets: Vec<String> = get("--market").unwrap_or("nyc,phl,pvd,nhv,spf,wor,hfd,lhg,lck".into()).split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let rounds: usize = get("--rounds").map_or(2, |v| v.parse().unwrap());
    let parts = args.iter().any(|a| a == "--parts");
    let mut st = Stage { last: alloc::current(), t: now_ms() };
    st.mark("start");

    let reg: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    let origin = |id: &str| {
        let c = reg["cities"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap();
        (c["origin"]["lon"].as_f64().unwrap(), c["origin"]["lat"].as_f64().unwrap())
    };
    let att: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/attractions.json")).unwrap()).unwrap();
    let att_rows: Vec<[f64; 3]> = att["rows"].as_array().unwrap().iter().map(|r| [r[2].as_f64().unwrap(), r[3].as_f64().unwrap(), r[4].as_f64().unwrap()]).collect();
    st.mark("registry, attractions read");

    // the zones, fine, each with its everyday layer
    let mut pops: Vec<(String, DemandApi, Option<EverydayApi>)> = vec![];
    for id in &zones {
        let (h, b) = pack::load_raw(&format!("{root}/data/packs/{id}.json")).unwrap();
        let a = DemandApi::at_level(&h, &b, 0).unwrap();
        st.mark(&format!("{id}: commutes (DemandApi)"));
        let (lon0, lat0) = origin(id);
        let f = Frame::equirect(lon0, lat0);
        let mut near = vec![];
        for r in &att_rows {
            let (x, y) = f.from_lonlat(r[0], r[1]);
            if x.abs() < 400_000.0 && y.abs() < 400_000.0 {
                near.extend([x as f32, y as f32, r[2] as f32]);
            }
        }
        let e = EverydayApi::beside_with(&a, &h, &b, &near).ok();
        drop((h, b));
        st.mark(&format!("{id}: everyday (EverydayApi)"));
        pops.push((id.clone(), a, e));
    }

    // the market: the world's rows, then the market packs, then the composition
    let mut ic: Option<IntercityApi> = None;
    if !markets.is_empty() {
        let by_id: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/market-rows.json")).unwrap()).unwrap();
        let m = &reg["market"];
        let rows: Vec<Value> = reg["cities"].as_array().unwrap().iter().filter_map(|c| by_id.get(c["id"].as_str()?).cloned()).collect();
        let world = serde_json::json!({"frame": m["frame"], "rule": m["rule"], "places": m["places"].as_array().unwrap().len(), "rows": rows}).to_string();
        drop((by_id, rows));
        let mut h = MarketPacks::new(&world).unwrap();
        drop(world);
        st.mark("market: world rows (MarketPacks)");
        for id in &markets {
            h.add(&std::fs::read_to_string(format!("{root}/data/packs/{id}.market.json")).unwrap()).unwrap();
        }
        st.mark(&format!("market: {} packs added", markets.len()));
        let mut a = h.market().unwrap();
        a.set_sampled(true);
        a.set_far_chunks(true);
        st.mark("market: composed (IntercityApi)");
        if parts { println!("\nmarket packs held:\n{}", h.mem_report()); }
        ic = Some(a);
        std::mem::forget(h);
    }

    // the network, in each pack's frame and (the market) in longitude and latitude
    let net = get("--net").unwrap_or("nyc".into());
    // T-3924: `--snapshot FILE`, a game's network as its demand workers got it, bus stops' stretches included
    // (`st_ll`, `line_n`, `stops`, `times`, `tph`, `cars`, `catch_n`, `catch_ll`; longitude and latitude)
    let (real, ll, catch_n, catch_ll) = match get("--snapshot") {
        Some(path) => {
            let j: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let f64s = |k: &str| j[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<f64>>();
            let f32s = |k: &str| f64s(k).into_iter().map(|v| v as f32).collect::<Vec<f32>>();
            let u32s = |k: &str| f64s(k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
            let ll = f64s("st_ll");
            let home = Frame::equirect(origin(&net).0, origin(&net).1);
            let st_xy = ll.chunks(2).flat_map(|p| { let (x, y) = home.from_lonlat(p[0], p[1]); [x as f32, y as f32] }).collect();
            (RealNet { st_xy, line_n: u32s("line_n"), stops: u32s("stops"), times: f32s("times"), tph: f32s("tph"), cars: f32s("cars") }, ll, u32s("catch_n"), f64s("catch_ll"))
        }
        None => {
            let (real, _) = RealNet::load(&format!("{root}/data/work/realnet/{net}_real_track.json"));
            let home = Frame::equirect(origin(&net).0, origin(&net).1);
            let ll: Vec<f64> = real.st_xy.chunks(2).flat_map(|p| { let (lon, lat) = home.to_lonlat(p[0] as f64, p[1] as f64); [lon, lat] }).collect();
            (real, ll, vec![], vec![])
        }
    };
    let hop_km = real.hop_km();
    println!("network: {} stations, {} lines, {} stops, {} bus stretch points", ll.len() / 2, real.line_n.len(), real.stops.len(), catch_ll.len() / 2);
    st.mark("network read");
    for (id, a, e) in pops.iter_mut() {
        let f = Frame::equirect(origin(id).0, origin(id).1);
        let xy: Vec<f32> = ll.chunks(2).flat_map(|p| { let (x, y) = f.from_lonlat(p[0], p[1]); [x as f32, y as f32] }).collect();
        let cxy: Vec<f32> = catch_ll.chunks(2).flat_map(|p| { let (x, y) = f.from_lonlat(p[0], p[1]); [x as f32, y as f32] }).collect();
        a.set_bus(&catch_n, &cxy);
        if let Some(e) = e {
            e.set_bus(&catch_n, &cxy);
        }
        a.set_chunked(true);
        a.set_network(&xy, &real.line_n, &real.stops, &real.times, &real.tph, &real.cars).unwrap();
        a.set_fares(1.5, 0.1, &hop_km).unwrap();
        st.mark(&format!("{id}: commutes set_network"));
        if let Some(e) = e {
            e.set_network(&xy, &real.line_n, &real.stops, &real.times, &real.tph, &real.cars).unwrap();
            e.set_fares(1.5, 0.1, &hop_km).unwrap();
            st.mark(&format!("{id}: everyday set_network"));
        }
    }
    if let Some(a) = ic.as_mut() {
        a.set_bus_ll(&catch_n, &catch_ll);
        a.set_network_ll(&ll,&real.line_n, &real.stops, &real.times, &real.tph, &real.cars).unwrap();
        a.set_fares(1.5, 0.1, &hop_km).unwrap();
        st.mark("market: set_network_ll");
    }

    // the periods: a free-flow solve, then the crowding rounds on every population's loads together
    let t_solve = now_ms();
    let mut peak_all = 0usize;
    let stages = args.iter().any(|a| a == "--stages");
    // `--app-order`: as `demand.worker.ts` `step` solves one cluster with no market: round by round over
    // the five periods, the skims released after each period's stage, the crowding priced on the
    // segment loads, boardings and alightings, and the last round's riders held to the trains' room
    let app_order = args.iter().any(|a| a == "--app-order");
    if app_order {
        let blocks = |pops: &Vec<(String, DemandApi, Option<EverydayApi>)>, q: usize| {
            let mut t: Vec<f32> = vec![];
            let mut put = |v: Vec<f32>, b: usize, n: usize| {
                if t.len() < 3 * n { t.resize(3 * n, 0.0); }
                for (k, x) in v.into_iter().enumerate() { t[b * n + k] += x; }
            };
            for (_, a, e) in pops.iter() {
                let n = a.seg(q).len();
                put(a.seg(q), 0, n); put(a.board(q), 1, n); put(a.alight(q), 2, n);
                if let Some(e) = e { put(e.seg(q), 0, n); put(e.board(q), 1, n); put(e.alight(q), 2, n); }
            }
            t
        };
        for round in 0..=rounds {
            for q in 0..PERIODS {
                let total = if round > 0 { Some(blocks(&pops, q)) } else { None };
                for (id, a, e) in pops.iter_mut() {
                    match &total { None => { a.solve(q); } Some(t) => assert!(a.crowd_with_external(q, t)) }
                    if stages { peak_all = peak_all.max(alloc::peak()); st.mark(&format!("  round {round} period {q}: {id} commutes")); }
                    if let Some(e) = e {
                        match &total { None => { e.solve(q); } Some(t) => assert!(e.crowd_with_external(q, t)) }
                        if stages { peak_all = peak_all.max(alloc::peak()); st.mark(&format!("  round {round} period {q}: {id} everyday")); }
                    }
                }
                if round == rounds && round > 0 {
                    let fin = blocks(&pops, q);
                    let n = fin.len() / 3;
                    let mut four = fin.clone();
                    four.extend_from_slice(&fin[n..2 * n]);
                    for (_, a, e) in pops.iter_mut() {
                        assert!(a.cap_with_external(q, &four));
                        if let Some(e) = e { assert!(e.cap_with_external(q, &four)); }
                    }
                }
                peak_all = peak_all.max(alloc::peak());
                release_skims();
                st.mark(&format!("round {round} period {q}: skims released"));
            }
        }
    }
    for q in 0..if app_order { 0 } else { PERIODS } {
        for (id, a, e) in pops.iter_mut() {
            a.solve(q);
            if stages { peak_all = peak_all.max(alloc::peak()); st.mark(&format!("  period {q}: {id} commutes solved")); }
            if let Some(e) = e {
                e.solve(q);
                if stages { peak_all = peak_all.max(alloc::peak()); st.mark(&format!("  period {q}: {id} everyday solved")); }
            }
        }
        if let Some(a) = ic.as_mut() {
            a.solve(q);
        }
        for _ in 0..rounds {
            let mut total: Vec<f32> = vec![];
            let mut put = |v: Vec<f32>| {
                if total.len() < v.len() { total.resize(v.len(), 0.0); }
                for (t, x) in total.iter_mut().zip(v) { *t += x; }
            };
            for (_, a, e) in pops.iter() {
                put(a.seg(q));
                if let Some(e) = e { put(e.seg(q)); }
            }
            if let Some(a) = ic.as_ref() { put(a.seg(q)); }
            for (_, a, e) in pops.iter_mut() {
                assert!(a.crowd_with_external(q, &total));
                if let Some(e) = e { assert!(e.crowd_with_external(q, &total)); }
            }
            if let Some(a) = ic.as_mut() { assert!(a.crowd_with_external(q, &total)); }
            if stages { peak_all = peak_all.max(alloc::peak()); st.mark(&format!("  period {q}: a crowding round")); }
        }
        peak_all = peak_all.max(alloc::peak());
        release_skims();
        st.mark(&format!("period {q}: solve + {rounds} rounds, skims released"));
    }
    println!("\nsolve {:.0} ms; highest heap during the periods {:.1} MB, at rest after {:.1} MB", now_ms() - t_solve, alloc::mb(peak_all.max(alloc::peak())), alloc::mb(alloc::current()));
    let mut rail = 0f64;
    for q in 0..PERIODS {
        for (_, a, e) in pops.iter() {
            rail += a.summary(q).get(2).copied().unwrap_or(0.0);
            if let Some(e) = e { rail += e.summary(q).get(2).copied().unwrap_or(0.0); }
        }
    }
    println!("rail trips (summary slot 2, every population and period): {rail:.0}");
    // every population's loads to the bit, so two builds can be told apart where the total rounds alike
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for q in 0..PERIODS {
        for (_, a, e) in pops.iter() {
            let mut vs = vec![a.seg(q), a.board(q), a.alight(q)];
            if let Some(e) = e { vs.extend([e.seg(q), e.board(q), e.alight(q)]); }
            for x in vs.iter().flatten() {
                h = (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    println!("loads hash {h:016x}, rail trips {rail:.4}");
    // the market's loads too (it reads the inter-city starts that `loads hash` does not see)
    if let Some(a) = ic.as_ref() {
        let mut hm: u64 = 0xcbf2_9ce4_8422_2325;
        for q in 0..PERIODS {
            for x in [a.seg(q), a.board(q), a.alight(q)].iter().flatten() {
                hm = (hm ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        println!("market loads hash {hm:016x}");
    }
    // `--lines FILE`: boardings by line over the five periods, line,commutes and everyday,market
    if let Some(path) = get("--lines") {
        let mut of = vec![];
        for (l, &k) in real.line_n.iter().enumerate() {
            of.extend(std::iter::repeat(l).take(2 * k as usize));
        }
        let mut per = vec![[0f64; 2]; real.line_n.len()];
        let mut add = |v: Vec<f32>, k: usize| {
            for (r, x) in v.iter().enumerate() {
                if let Some(&l) = of.get(r) { per[l][k] += *x as f64; }
            }
        };
        for q in 0..PERIODS {
            for (_, a, e) in pops.iter() {
                add(a.board(q), 0);
                if let Some(e) = e { add(e.board(q), 0); }
            }
            if let Some(a) = ic.as_ref() { add(a.board(q), 1); }
        }
        let s: String = per.iter().enumerate().map(|(l, x)| format!("{l},{:.3},{:.3}\n", x[0], x[1])).collect();
        std::fs::write(path, s).unwrap();
    }
    if parts {
        for (id, a, e) in pops.iter() {
            println!("\n{id} commutes:\n{}", a.mem_report());
            if let Some(e) = e { println!("{id} everyday:\n{}", e.mem_report()); }
        }
        if let Some(a) = ic.as_ref() { println!("\nmarket:\n{}", a.mem_report()); }
    }
}
