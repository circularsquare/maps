//! T-276: one commute gravity for every zone pack together (docs/zones.md 4).
//!
//!   cargo run --release --features demand --bin joint_gravity -- [--root ..] [--cut-km 150]
//!       [--tol 1e-8] [--threads 3] [--dry] [--check] [--report data/work/joint_report.json]
//!       [--dump factors.json] [--anderson 12] [--max-iters 500]
//!
//! Reads every zone pack of the registry (`app/src/game/registry.json`, cities with a market place)
//! and the regional market's pack (`data/packs/intercity.json`, its sites written by
//! pipeline/intercity.py), and solves the doubly constrained work gravity over all their 2 km zones
//! at once: every zone's commuters (the pack's home weights, scaled so the union's homes equal its
//! jobs) are shared over every job in reach, in their own pack or another.
//!
//! - A pair inside one pack uses the pack's own decay table and border factor, exactly as the pack
//!   does at runtime, so the factors written back rebuild the within-pack part to the bit.
//! - A pair across packs uses the straight-line distance between the two zones' squares in the
//!   world frame (a conformal conic, true to 0.4% within 5 degrees of latitude of New York), the home
//!   pack's curve, cut at `--cut-km`, and a border factor per region pair (`pipeline/borders.json`):
//!   the factor of the pack holding both regions, else the country's admin-1 default, and a national
//!   factor across a national border.
//!
//! Writes each pack's gravity (new balancing factors, `zone_home` and `zone_work`: the trips each
//! zone keeps in the pack, and a `joint` block), and the market pack's `commute_*` arrays: the
//! cross-pack part spread over the market's 6 km zones by each gravity zone's homes and jobs in
//! them. `--dry` writes nothing. Prints commuters between each pair of packs.
//!
//! The balance (T-297) runs per group of packs joined by any pair within the cut, each to `--tol`
//! on every column (rows exact), with Anderson mixing; `--check` re-measures both after the solve,
//! and `--dump` writes the factors and kept trips for comparing two solves.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use trainworld_sim::demand::{clock::now_ms, intercity, kernel, pack};
use trainworld_sim::track::frame::Frame;

struct Pk {
    id: String,
    header: String,
    bin: Vec<u8>,
    city: pack::City,
    z: kernel::Zones,
    ftab: Vec<f32>,
    th: i32,
    border: f32,
    decay_km: f32,
    decay_pow: f32,
    zone_m: f32,
    /// global region id per pack region
    regions: Vec<usize>,
    /// first global zone index
    off: usize,
    market: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let has = |k: &str| args.iter().any(|a| a == k);
    let root = get("--root").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/..").to_string());
    let cut_km: f64 = get("--cut-km").map_or(150.0, |v| v.parse().unwrap());
    let tol: f64 = get("--tol").map_or(1e-8, |v| v.parse().unwrap());
    let threads: usize = get("--threads").map_or(3, |v| v.parse().unwrap());
    let t0 = now_ms();

    let registry: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/app/src/game/registry.json")).unwrap()).unwrap();
    // Zone distances in the Northeast conic at the market's origin (T-125). The game's own frame
    // is Mercator since T-306 and its plane metres are not ground metres, so it is not used here.
    let wf = &registry["market"]["frame"];
    let world = Frame::lcc(wf["lon"].as_f64().unwrap(), wf["lat"].as_f64().unwrap());
    let borders: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/pipeline/borders.json")).unwrap()).unwrap();

    // ---- the packs and their zones
    let mut packs: Vec<Pk> = vec![];
    let mut region_key: Vec<(String, String)> = vec![]; // (country, region id)
    let mut off = 0usize;
    for c in registry["cities"].as_array().unwrap() {
        let market = c["market"].as_i64().unwrap_or(-1);
        if market < 0 {
            continue;
        }
        let id = c["id"].as_str().unwrap().to_string();
        let (header, bin) = pack::load_raw(&format!("{root}/data/packs/{id}.json")).unwrap_or_else(|e| panic!("{e}"));
        let city = pack::parse(&header, &bin).unwrap_or_else(|e| panic!("{id}: {e}"));
        let pg = city.gravity.clone().unwrap_or_else(|| panic!("{id} has no gravity"));
        let z = kernel::zones_from_pack(&city, &pg);
        let (ftab, _) = kernel::decay_table(z.w, z.h, z.zone_m, pg.decay_km, pg.decay_pow);
        let hv: Value = serde_json::from_str(&header).unwrap();
        let country = hv["country"].as_str().unwrap_or("US").to_uppercase();
        let ids: Vec<String> = hv["regions"]["ids"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect()).unwrap_or_else(|| vec!["".into()]);
        let regions = ids
            .iter()
            .map(|r| {
                let k = (country.clone(), r.clone());
                region_key.iter().position(|x| *x == k).unwrap_or_else(|| {
                    region_key.push(k);
                    region_key.len() - 1
                })
            })
            .collect();
        let n = z.n;
        packs.push(Pk { id, header, bin, th: 2 * z.h - 1, border: pg.border_factor, decay_km: pg.decay_km, decay_pow: pg.decay_pow, zone_m: pg.zone_m, city, z, ftab, regions, off, market: market as usize });
        off += n;
    }
    let nz = off;
    let np = packs.len();
    let nr = region_key.len();
    eprintln!("{} packs, {} zones, {} regions, loaded in {:.0} ms", np, nz, nr, now_ms() - t0);

    // ---- border factor per pair of global regions
    let mut kf = vec![f32::NAN; nr * nr];
    let mut kn = vec![0u32; nr * nr];
    for p in &packs {
        for (a, &ra) in p.regions.iter().enumerate() {
            for (b, &rb) in p.regions.iter().enumerate() {
                if a != b {
                    let i = ra * nr + rb;
                    kf[i] = if kn[i] == 0 { p.border } else { kf[i] + p.border };
                    kn[i] += 1;
                }
            }
        }
    }
    let mut table = vec![];
    for a in 0..nr {
        for b in 0..nr {
            let i = a * nr + b;
            let (ca, cb) = (&region_key[a].0, &region_key[b].0);
            kf[i] = if a == b {
                1.0
            } else if kn[i] > 0 {
                kf[i] / kn[i] as f32
            } else if ca == cb {
                borders["admin1_default"][ca.as_str()].as_f64().unwrap_or(1.0) as f32
            } else {
                let key = if ca < cb { format!("{ca}|{cb}") } else { format!("{cb}|{ca}") };
                borders["national"][key.as_str()].as_f64().or(borders["national_default"].as_f64()).unwrap_or(1.0) as f32
            };
            if a < b {
                table.push(json!([format!("{}-{}", region_key[a].0, region_key[a].1), format!("{}-{}", region_key[b].0, region_key[b].1), kf[i]]));
            }
        }
    }

    // ---- per global zone: pack, region, position in the world frame, weights
    let mut zp = vec![0u16; nz];
    let mut zr = vec![0u16; nz];
    let (mut px, mut py) = (vec![0f64; nz], vec![0f64; nz]);
    let (mut pop, mut jobs) = (vec![0f64; nz], vec![0f64; nz]);
    for (pi, p) in packs.iter().enumerate() {
        let hv: Value = serde_json::from_str(&p.header).unwrap();
        let fr = Frame::equirect(hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
        for i in 0..p.z.n {
            let g = p.off + i;
            zp[g] = pi as u16;
            zr[g] = p.regions[p.z.region[i] as usize] as u16;
            let x = p.z.x0 as f64 + (p.z.gx[i] as f64 + 0.5) * p.zone_m as f64;
            let y = p.z.y0 as f64 + (p.z.gy[i] as f64 + 0.5) * p.zone_m as f64;
            let (lon, lat) = fr.to_lonlat(x, y);
            let (wx, wy) = world.from_lonlat(lon, lat);
            px[g] = wx;
            py[g] = wy;
            pop[g] = p.z.pop[i];
            jobs[g] = p.z.jobs[i];
        }
    }
    // each pack's homes send as many commuters as its own jobs, as in the pack's own solve: scaled
    // to the union's total instead, every pack whose jobs and homes differ (net commuting from
    // outside the zones) would have to balance through its neighbours
    let tj: f64 = jobs.iter().sum();
    let mut o = vec![0f64; nz];
    for p in &packs {
        let r = p.off..p.off + p.z.n;
        let (tpp, tjp): (f64, f64) = (pop[r.clone()].iter().sum(), jobs[r.clone()].iter().sum());
        for g in r {
            o[g] = pop[g] * tjp / tpp.max(1e-9);
        }
    }

    // ---- cross-pack neighbours: per pack, its zones in buckets of 10 km of the world frame
    let bucket_m = 10_000.0;
    let reach_f = cut_km * 1000.0 / bucket_m;
    let reach = reach_f.ceil() as i32;
    let bucket = |g: usize| ((px[g] / bucket_m).floor() as i32, (py[g] / bucket_m).floor() as i32);
    let grids: Vec<Grid> = packs.iter().map(|p| Grid::new((p.off..p.off + p.z.n).map(|g| (g as u32, bucket(g))).collect())).collect();
    // per pack, the cross decay by distance in 10 m steps (its own curve)
    let step = 0.01f64;
    let nsteps = (cut_km / step) as usize + 2;
    let cross_tab: Vec<Vec<f32>> = packs
        .iter()
        .map(|p| {
            let dmin = 0.52 * p.zone_m as f64 / 1000.0;
            (0..nsteps)
                .map(|k| {
                    let d = (k as f64 * step).max(dmin);
                    (d.powf(-p.decay_pow as f64) * (-d / p.decay_km as f64).exp()) as f32
                })
                .collect()
        })
        .collect();
    let cut2 = (cut_km * 1000.0).powi(2);
    // for zone g, every cross-pack zone h within the cut, with w = f(d) x k (the home pack's curve
    // when `g_home`, else h's)
    let cross = |g: usize, g_home: bool, f: &mut dyn FnMut(usize, f64)| {
        let (bx, by) = bucket(g);
        for (q, grid) in grids.iter().enumerate() {
            if q == zp[g] as usize || bx < grid.x0 - reach || bx > grid.x1 + reach || by < grid.y0 - reach || by > grid.y1 + reach {
                continue;
            }
            for y in (by - reach).max(grid.y0)..=(by + reach).min(grid.y1) {
                // buckets whose nearest point can be within the cut
                let m = ((y - by).abs() - 1).max(0) as f64;
                let dxr = 1 + (reach_f * reach_f - m * m).max(0.0).sqrt().floor() as i32;
                for &h in grid.row(y, bx - dxr, bx + dxr) {
                    let h = h as usize;
                    let d2 = (px[h] - px[g]).powi(2) + (py[h] - py[g]).powi(2);
                    if d2 > cut2 {
                        continue;
                    }
                    let tab = &cross_tab[if g_home { zp[g] } else { zp[h] } as usize];
                    let w = tab[(d2.sqrt() / 1000.0 / step) as usize] * kf[zr[g] as usize * nr + zr[h] as usize];
                    f(h, w as f64);
                }
            }
        }
    };
    // within one pack: zone i's sum over the pack's zones of v_j x f x k (the runtime's table)
    let within = |p: &Pk, i: usize, v: &[f64]| -> f64 {
        let base = (p.z.w - 1 - p.z.gx[i]) * p.th + (p.z.h - 1) - p.z.gy[i];
        let ri = p.z.region[i];
        let mut s = 0.0;
        for j in 0..p.z.n {
            let vj = v[p.off + j];
            if vj == 0.0 {
                continue;
            }
            let k = if p.z.region[j] == ri { 1.0 } else { p.border };
            s += vj * (p.ftab[(base + p.z.gx[j] * p.th + p.z.gy[j]) as usize] * k) as f64;
        }
        s
    };
    // one half step over the zones `zs`: for each zone g, sum_h v_h w(g, h) over its pack and across.
    // Threads take blocks of zones as they free up: a New York zone costs ten of a small pack's.
    let sums = |v: &[f64], rows: bool, zs: &[usize]| -> Vec<f64> {
        const BLOCK: usize = 16;
        let next = AtomicUsize::new(0);
        let parts: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = vec![];
                        loop {
                            let lo = next.fetch_add(1, Ordering::Relaxed) * BLOCK;
                            if lo >= zs.len() {
                                break;
                            }
                            let vals: Vec<f64> = zs[lo..(lo + BLOCK).min(zs.len())]
                                .iter()
                                .map(|&g| {
                                    let p = &packs[zp[g] as usize];
                                    // the decay is symmetric within a pack; across packs the home end's curve
                                    let mut acc = within(p, g - p.off, v);
                                    cross(g, rows, &mut |h, w| acc += v[h] * w);
                                    acc
                                })
                                .collect();
                            mine.push((lo, vals));
                        }
                        mine
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut out = vec![0f64; nz];
        for (lo, vals) in parts.into_iter().flatten() {
            for (k, x) in vals.into_iter().enumerate() {
                out[zs[lo + k]] = x;
            }
        }
        out
    };

    // ---- groups of packs joined by a pair within the cut: they share no commuter, so each is
    // balanced alone and stops when it has converged
    let mut parent: Vec<usize> = (0..np).collect();
    fn root_of(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    let mut npairs = 0usize;
    for g in 0..nz {
        cross(g, true, &mut |h, _| {
            npairs += 1;
            let (a, b) = (root_of(&mut parent, zp[g] as usize), root_of(&mut parent, zp[h] as usize));
            if a != b {
                parent[a] = b;
            }
        });
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for pi in 0..np {
        let r = root_of(&mut parent, pi);
        let k = *group_of.entry(r).or_insert_with(|| {
            groups.push(vec![]);
            groups.len() - 1
        });
        groups[k].push(pi);
    }
    let within_pairs: usize = packs.iter().map(|p| p.z.n * p.z.n).sum();
    eprintln!("pairs: {within_pairs} within packs, {npairs} across");
    eprintln!("{} groups of packs: {}, {:.0} s", groups.len(), groups.iter().map(|g| g.iter().map(|&p| packs[p].id.as_str()).collect::<Vec<_>>().join("+")).collect::<Vec<_>>().join(", "), (now_ms() - t0) / 1000.0);

    // ---- balancing: Furness steps with Anderson mixing (T-297)
    let max_iters: usize = get("--max-iters").map_or(500, |v| v.parse().unwrap());
    let mix: usize = get("--anderson").map_or(12, |v| v.parse().unwrap());
    let mut row = vec![0f64; nz];
    let mut col = jobs.clone();
    let (mut iters, mut err) = (0, 0f64);
    for grp in &groups {
        let zs: Vec<usize> = grp.iter().flat_map(|&pi| packs[pi].off..packs[pi].off + packs[pi].z.n).collect();
        let name = grp.iter().map(|&p| packs[p].id.as_str()).collect::<Vec<_>>().join("+");
        let b = balance(&o, &jobs, &col, &zs, &|v, rows| sums(v, rows, &zs), tol, max_iters, mix, &mut |k, e| {
            if k % 10 == 0 {
                eprintln!("  {name}: step {k}, max column error {e:.2e} ({:.0} s)", (now_ms() - t0) / 1000.0);
            }
        });
        for &g in &zs {
            row[g] = b.row[g];
            col[g] = b.col[g];
        }
        eprintln!("{name}: {} Furness steps, max column error {:.2e} (rows exact), {:.0} s", b.steps, b.err, (now_ms() - t0) / 1000.0);
        iters = iters.max(b.steps);
        err = err.max(b.err);
    }
    if has("--check") {
        let all: Vec<usize> = (0..nz).collect();
        let sr = sums(&col, true, &all);
        let sc = sums(&row, false, &all);
        let (mut er, mut ec) = (0f64, 0f64);
        for g in 0..nz {
            if o[g] > 0.0 && sr[g] > 0.0 {
                er = er.max((row[g] * sr[g] / o[g] - 1.0).abs());
            }
            if jobs[g] > 0.0 && sc[g] > 0.0 {
                ec = ec.max((col[g] * sc[g] / jobs[g] - 1.0).abs());
            }
        }
        eprintln!("check: max row error {er:.2e}, max column error {ec:.2e}, {:.0} s", (now_ms() - t0) / 1000.0);
    }

    // ---- what each zone keeps in its pack, and the cross-pack flows
    let mut home = vec![0f64; nz];
    let mut work = vec![0f64; nz];
    for p in &packs {
        for i in 0..p.z.n {
            let g = p.off + i;
            home[g] = row[g] * within(p, i, &col);
            work[g] = col[g] * within(p, i, &row);
        }
    }
    if let Some(p) = get("--dump") {
        let d = json!({
            "packs": packs.iter().map(|p| json!([p.id, p.off, p.z.n])).collect::<Vec<_>>(),
            "o": o, "jobs": jobs, "row": row, "col": col, "home": home, "work": work, "iterations": iters,
        });
        std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
    }
    // market zones: each pack cell's 600 m square's site, and its zone
    let market_path = format!("{root}/data/packs/intercity.json");
    let market_text = std::fs::read_to_string(&market_path).unwrap();
    let (site_zone, zone_site, site_keys) = intercity::market_site_zones(&market_text).unwrap();
    let mv: Value = serde_json::from_str(&market_text).unwrap();
    let mframe = Frame::equirect(mv["frame"]["lon"].as_f64().unwrap(), mv["frame"]["lat"].as_f64().unwrap());
    let square: HashMap<(i32, i32, usize), u32> = site_keys.iter().enumerate().map(|(s, &k)| (k, s as u32)).collect();
    // per global zone: (market zone, share) of its homes and of its jobs
    let mut hsplit: Vec<Vec<(u32, f64)>> = vec![vec![]; nz];
    let mut wsplit: Vec<Vec<(u32, f64)>> = vec![vec![]; nz];
    let mut missed = (0f64, 0f64);
    for p in &packs {
        let hv: Value = serde_json::from_str(&p.header).unwrap();
        let fr = Frame::equirect(hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
        let mut acc: Vec<HashMap<u32, (f64, f64)>> = vec![HashMap::new(); p.z.n];
        for c in 0..p.city.len() {
            let (lon, lat) = fr.to_lonlat(p.city.x[c] as f64, p.city.y[c] as f64);
            let (mx, my) = mframe.from_lonlat(lon, lat);
            let key = ((mx / 600.0).floor() as i32, (my / 600.0).floor() as i32, p.market);
            let Some(&s) = square.get(&key) else {
                missed.0 += p.city.pop[c] as f64;
                missed.1 += p.city.jobs[c] as f64;
                continue;
            };
            let e = acc[p.z.cell_zone[c] as usize].entry(site_zone[s as usize]).or_default();
            e.0 += p.city.pop[c] as f64;
            e.1 += p.city.jobs[c] as f64;
        }
        for (i, m) in acc.into_iter().enumerate() {
            let (th, tw): (f64, f64) = m.values().fold((0., 0.), |a, v| (a.0 + v.0, a.1 + v.1));
            for (mz, (h, w)) in m {
                if th > 0.0 && h > 0.0 {
                    hsplit[p.off + i].push((mz, h / th));
                }
                if tw > 0.0 && w > 0.0 {
                    wsplit[p.off + i].push((mz, w / tw));
                }
            }
        }
    }
    eprintln!("cells with no market site: {:.0} home weight, {:.0} work weight", missed.0, missed.1);
    let mut flows: HashMap<(u32, u32), f64> = HashMap::new();
    let mut by_pack = vec![0f64; np * np];
    let mut cross_km = (0f64, 0f64);
    for g in 0..nz {
        if row[g] == 0.0 {
            continue;
        }
        cross(g, true, &mut |h, w| {
            let t = row[g] * col[h] * w;
            if t <= 0.0 {
                return;
            }
            by_pack[zp[g] as usize * np + zp[h] as usize] += t;
            cross_km.0 += t;
            cross_km.1 += t * ((px[h] - px[g]).powi(2) + (py[h] - py[g]).powi(2)).sqrt() / 1000.0;
            for &(mh, sh) in &hsplit[g] {
                for &(mw, sw) in &wsplit[h] {
                    *flows.entry((mh, mw)).or_default() += t * sh * sw;
                }
            }
        });
    }
    eprintln!("flows between packs, {:.0} s", (now_ms() - t0) / 1000.0);
    let total_within: f64 = home.iter().sum();
    let total_cross: f64 = by_pack.iter().sum();
    println!(
        "commuters: {:.0} in all, {:.0} within their pack, {:.0} ({:.2}%) to another pack, mean {:.1} km",
        tj,
        total_within,
        total_cross,
        100.0 * total_cross / tj,
        cross_km.1 / cross_km.0.max(1e-9)
    );
    println!("commuters between packs (home row, work column):");
    for a in 0..np {
        let line: Vec<String> = (0..np).filter(|&b| b != a && by_pack[a * np + b] >= 1.0).map(|b| format!("{} {:.0}", packs[b].id, by_pack[a * np + b])).collect();
        println!("  {}: {}", packs[a].id, line.join(", "));
    }
    // the commute list: pairs under `--min` commuters a day are dropped and their commuters given
    // to the home zone's kept pairs in proportion, so every zone keeps its commuters
    let min: f64 = get("--min").map_or(1.0, |v| v.parse().unwrap());
    for m in [0.02, 0.1, 0.2, 0.5, 1.0, 2.0] {
        let (n, t) = flows.values().filter(|t| **t >= m).fold((0, 0.0), |a, t| (a.0 + 1, a.1 + t));
        println!("  pairs of at least {m} a day: {n}, {:.1}% of the commuters", 100.0 * t / total_cross.max(1e-9));
    }
    let mut home_all: HashMap<u32, f64> = HashMap::new();
    let mut home_kept: HashMap<u32, f64> = HashMap::new();
    for (&(h, _), &t) in &flows {
        *home_all.entry(h).or_default() += t;
        if t >= min {
            *home_kept.entry(h).or_default() += t;
        }
    }
    let mut list: Vec<((u32, u32), f64)> = flows.into_iter().filter(|(_, t)| *t >= min).map(|(k, t)| (k, t * home_all[&k.0] / home_kept[&k.0])).collect();
    list.sort_by(|a, b| a.0.cmp(&b.0));
    let kept: f64 = list.iter().map(|x| x.1).sum();
    println!("market zone pairs: {} kept (at least {min} a day), {:.0} commuters after giving the dropped ones to each home zone's kept pairs ({:.2}% of the cross-pack ones; home zones with no kept pair lose theirs)", list.len(), kept, 100.0 * kept / total_cross.max(1e-9));

    let report = json!({
        "packs": packs.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "by_pack": by_pack,
        "commuters": tj, "within": total_within, "cross": total_cross,
        "border_pairs": table, "iterations": iters, "max_error": err, "cut_km": cut_km,
    });
    if let Some(p) = get("--report") {
        std::fs::write(&p, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    if has("--dry") {
        eprintln!("--dry: nothing written");
        return;
    }
    // ---- write the packs
    for p in &packs {
        let mut pg = p.city.gravity.clone().unwrap();
        let r = p.off..p.off + p.z.n;
        pg.row = row[r.clone()].to_vec();
        pg.col = col[r.clone()].to_vec();
        pg.home = home[r.clone()].to_vec();
        pg.work = work[r.clone()].to_vec();
        pg.trips = pg.home.iter().sum();
        pg.iters = iters;
        pg.max_err = err;
        let pi = zp[p.off] as usize;
        let out: f64 = (0..np).map(|b| by_pack[pi * np + b]).sum();
        let into: f64 = (0..np).map(|a| by_pack[a * np + pi]).sum();
        pg.joint = Some(json!({
            "packs": packs.iter().map(|q| q.id.clone()).collect::<Vec<_>>(),
            "cut_km": cut_km,
            "commuters_out": out,
            "commuters_in": into,
            "border_pairs": "pipeline/borders.json and each pack's own factor for the regions it holds",
        }));
        let (h2, b2) = pack::with_gravity(&p.header, &p.bin, &pg).unwrap_or_else(|e| panic!("{}: {e}", p.id));
        let back = pack::parse(&h2, &b2).unwrap();
        assert_eq!(back.gravity.as_ref().unwrap().home.len(), p.z.n);
        let stem = format!("{root}/data/packs/{}", p.id);
        std::fs::write(format!("{stem}.bin.part"), &b2).unwrap();
        std::fs::rename(format!("{stem}.bin.part"), format!("{stem}.bin")).unwrap();
        std::fs::write(format!("{stem}.json.part"), &h2).unwrap();
        std::fs::rename(format!("{stem}.json.part"), format!("{stem}.json")).unwrap();
        println!("  {}: {:.0} trips kept, {:.0} out, {:.0} in", p.id, pg.trips, out, into);
    }
    // ---- write the market's commutes
    let mut mv = mv;
    mv["commute_home"] = json!(list.iter().map(|x| zone_site[x.0 .0 as usize]).collect::<Vec<_>>());
    mv["commute_work"] = json!(list.iter().map(|x| zone_site[x.0 .1 as usize]).collect::<Vec<_>>());
    mv["commute_trips"] = json!(list.iter().map(|x| (x.1 * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    mv["commute_source"] = json!(format!("bin/joint_gravity.rs (T-276): the cross-pack part of one commute gravity over {} packs, cut at {cut_km} km; home and work sites name their 6 km zones", np));
    let text = serde_json::to_string(&mv).unwrap();
    intercity::IntercityApi::new(&text).unwrap_or_else(|e| panic!("the written market does not open: {e}"));
    std::fs::write(format!("{market_path}.part"), &text).unwrap();
    std::fs::rename(format!("{market_path}.part"), &market_path).unwrap();
    println!("wrote {} packs and {market_path} ({:.2} MB) in {:.0} s", np, text.len() as f64 / 1e6, (now_ms() - t0) / 1000.0);
}

/// One pack's zones by bucket, row by row, so a run of buckets in a row is one slice.
struct Grid {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    start: Vec<u32>,
    ids: Vec<u32>,
}

impl Grid {
    fn new(mut zs: Vec<(u32, (i32, i32))>) -> Grid {
        let x0 = zs.iter().map(|z| z.1 .0).min().unwrap_or(0);
        let x1 = zs.iter().map(|z| z.1 .0).max().unwrap_or(-1);
        let y0 = zs.iter().map(|z| z.1 .1).min().unwrap_or(0);
        let y1 = zs.iter().map(|z| z.1 .1).max().unwrap_or(-1);
        let w = (x1 - x0 + 1).max(0) as usize;
        let key = |b: (i32, i32)| (b.1 - y0) as usize * w + (b.0 - x0) as usize;
        zs.sort_by_key(|z| (key(z.1), z.0));
        let nb = w * (y1 - y0 + 1).max(0) as usize;
        let mut start = vec![0u32; nb + 1];
        for z in &zs {
            start[key(z.1) + 1] += 1;
        }
        for k in 0..nb {
            start[k + 1] += start[k];
        }
        Grid { x0, x1, y0, y1, start, ids: zs.into_iter().map(|z| z.0).collect() }
    }

    /// zones in buckets xa..=xb of row y (y within the grid)
    fn row(&self, y: i32, xa: i32, xb: i32) -> &[u32] {
        let (xa, xb) = (xa.max(self.x0), xb.min(self.x1));
        if xa > xb {
            return &[];
        }
        let w = (self.x1 - self.x0 + 1) as usize;
        let base = (y - self.y0) as usize * w;
        &self.ids[self.start[base + (xa - self.x0) as usize] as usize..self.start[base + (xb - self.x0) as usize + 1] as usize]
    }
}

struct Balanced {
    row: Vec<f64>,
    col: Vec<f64>,
    steps: usize,
    /// max over columns of |col (W^T row) / d - 1|; the rows are exact
    err: f64,
}

/// The doubly constrained balance over the zones `zs`: row = o / (W col), col = d / (W^T row), where
/// `sums(v, true)` is W v and `sums(v, false)` is W^T v. A Furness step maps log col to log col and the
/// solution is its fixed point. Plain Furness crawls along the few slow modes between weakly joined
/// packs and regions (500 steps to 2e-4 on the Northeast); Anderson mixing of the last `mix` steps
/// removes them (T-297). `mix` 0 is plain Furness.
#[allow(clippy::too_many_arguments)]
fn balance(o: &[f64], d: &[f64], c0: &[f64], zs: &[usize], sums: &dyn Fn(&[f64], bool) -> Vec<f64>, tol: f64, max_steps: usize, mix: usize, log: &mut dyn FnMut(usize, f64)) -> Balanced {
    let n = o.len();
    let act: Vec<usize> = zs.iter().copied().filter(|&g| d[g] > 0.0).collect();
    let mut x: Vec<f64> = act.iter().map(|&g| if c0[g] > 0.0 { c0[g] } else { d[g] }.ln()).collect();
    // a column no home reaches gets 0, as in plain Furness
    let mut dead = vec![false; act.len()];
    let mut hist: VecDeque<(Vec<f64>, Vec<f64>)> = VecDeque::new();
    let mut prev: Option<(Vec<f64>, Vec<f64>, f64)> = None;
    let mut best: Option<Balanced> = None;
    let mut c = vec![0f64; n];
    for k in 0..max_steps {
        for (i, &g) in act.iter().enumerate() {
            c[g] = if dead[i] { 0.0 } else { x[i].exp() };
        }
        let s = sums(&c, true);
        let mut row = vec![0f64; n];
        for &g in zs {
            if o[g] > 0.0 && s[g] > 0.0 {
                row[g] = o[g] / s[g];
            }
        }
        let s = sums(&row, false);
        let mut gx = vec![0f64; act.len()];
        let mut f = vec![0f64; act.len()];
        let mut err = 0f64;
        for (i, &g) in act.iter().enumerate() {
            if s[g] > 0.0 {
                gx[i] = (d[g] / s[g]).ln();
                f[i] = gx[i] - x[i];
                err = err.max((-f[i]).exp_m1().abs());
            } else {
                dead[i] = true;
                gx[i] = x[i];
            }
        }
        log(k, err);
        if best.as_ref().is_none_or(|b| err < b.err) {
            best = Some(Balanced { row, col: c.clone(), steps: k + 1, err });
        }
        if err < tol {
            break;
        }
        let fnorm = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        if let Some((fp, gp, fnp)) = prev.take() {
            if fnorm > 2.0 * fnp {
                hist.clear();
            } else if mix > 0 {
                hist.push_back((f.iter().zip(&fp).map(|(a, b)| a - b).collect(), gx.iter().zip(&gp).map(|(a, b)| a - b).collect()));
                if hist.len() > mix {
                    hist.pop_front();
                }
            }
        }
        let gamma = if hist.is_empty() { None } else { least_squares(&hist, &f) };
        if gamma.is_none() {
            hist.clear();
        }
        x = match &gamma {
            Some(gm) => {
                let mut xn = gx.clone();
                for (j, (_, dg)) in hist.iter().enumerate() {
                    for (v, dv) in xn.iter_mut().zip(dg) {
                        *v -= gm[j] * dv;
                    }
                }
                xn
            }
            None => gx.clone(),
        };
        prev = Some((f, gx, fnorm));
    }
    let mut b = best.unwrap();
    b.steps = b.steps.max(1);
    b
}

/// gamma minimising |f - sum_j gamma_j df_j| (normal equations, lightly regularised); None when singular
fn least_squares(hist: &VecDeque<(Vec<f64>, Vec<f64>)>, f: &[f64]) -> Option<Vec<f64>> {
    let m = hist.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut a = vec![vec![0f64; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            a[i][j] = dot(&hist[i].0, &hist[j].0);
        }
        a[i][m] = dot(&hist[i].0, f);
    }
    let tr: f64 = (0..m).map(|i| a[i][i]).sum::<f64>() / m as f64;
    for (i, r) in a.iter_mut().enumerate() {
        r[i] += 1e-12 * tr;
    }
    for p in 0..m {
        let piv = (p..m).max_by(|&i, &j| a[i][p].abs().total_cmp(&a[j][p].abs()))?;
        a.swap(p, piv);
        if a[p][p].abs() <= 1e-300 {
            return None;
        }
        for i in p + 1..m {
            let r = a[i][p] / a[p][p];
            for j in p..=m {
                a[i][j] -= r * a[p][j];
            }
        }
    }
    let mut x = vec![0f64; m];
    for i in (0..m).rev() {
        x[i] = (a[i][m] - (i + 1..m).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two blocks of 40 zones joined weakly, as packs are: Anderson reaches what plain Furness reaches
    /// after thousands of steps, in a small fraction of them.
    #[test]
    fn anderson_matches_long_furness() {
        let n = 80;
        let mut w = vec![0f64; n * n];
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let pos: Vec<f64> = (0..n).map(|i| i as f64 + rnd()).collect();
        for i in 0..n {
            for j in 0..n {
                let k = if (i < 40) == (j < 40) { 1.0 } else { 0.002 };
                w[i * n + j] = k * (-(pos[i] - pos[j]).abs() / 6.0).exp() * (1.0 + 0.5 * rnd());
            }
        }
        let mut o: Vec<f64> = (0..n).map(|_| 1.0 + 9.0 * rnd()).collect();
        let d: Vec<f64> = (0..n).map(|i| if i % 13 == 5 { 0.0 } else { 1.0 + 9.0 * rnd() }).collect();
        // each block's homes equal its jobs, as each pack's do
        for b in [0..40, 40..80] {
            let (so, sd): (f64, f64) = (o[b.clone()].iter().sum(), d[b.clone()].iter().sum());
            o[b.clone()].iter_mut().for_each(|v| *v *= sd / so);
        }
        let sums = |v: &[f64], rows: bool| -> Vec<f64> {
            (0..n).map(|g| (0..n).map(|h| v[h] * if rows { w[g * n + h] } else { w[h * n + g] }).sum()).collect()
        };
        let zs: Vec<usize> = (0..n).collect();
        let slow = balance(&o, &d, &d, &zs, &sums, 1e-10, 50000, 0, &mut |_, _| {});
        let fast = balance(&o, &d, &d, &zs, &sums, 1e-10, 50000, 8, &mut |_, _| {});
        eprintln!("furness {} steps, anderson {}", slow.steps, fast.steps);
        assert!(slow.err < 1e-10 && fast.err < 1e-10, "{} {}", slow.err, fast.err);
        assert!(fast.steps * 10 < slow.steps, "anderson {} steps, furness {}", fast.steps, slow.steps);
        // flows agree (the factors themselves are fixed only up to row x a, col / a); the weak link
        // fixes the blocks' relative scale only to about err / (the link's share), hence 1e-6
        for i in 0..n {
            for j in 0..n {
                let (a, b) = (slow.row[i] * slow.col[j], fast.row[i] * fast.col[j]);
                assert!((a - b).abs() <= 1e-6 * a.max(b).max(1e-300), "{i} {j} {a} {b}");
            }
        }
        for j in 0..n {
            assert_eq!(fast.col[j] == 0.0, d[j] == 0.0);
        }
    }
}
