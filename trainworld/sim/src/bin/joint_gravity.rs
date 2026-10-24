//! T-276: one commute gravity for every zone pack together (docs/zones.md 4).
//!
//!   cargo run --release --features demand --bin joint_gravity -- [--root ..] [--cut-km 150]
//!       [--tol 1e-8] [--threads 3] [--dry] [--check] [--report data/work/joint_report.json]
//!       [--dump factors.json] [--anderson 12] [--max-iters 500] [--packs a,b,.. --out FILE]
//!       [--admin1 F]  (F in place of borders.json's admin-1 default, for fitting it)
//!       [--fixed c,d [--init F] [--hold-report F [--new a,b]]]
//!
//! `--packs` (T-463) solves the named packs instead (a group outside the Northeast, Japan's), and
//! writes their cross-pack commutes to `--out` by 6 km squares of each pack's own frame, for
//! pipeline/market_zones.py; intercity.json is not read or written.
//!
//! `--fixed c,d` (T-786, with `--packs`): packs held at their factors (the pack's gravity block, or
//! `--init F`, a `--dump` of an earlier solve), a neighbouring country's: their zones take part in
//! the solved packs' sums across the cut but are not balanced or written. `--out` then holds every
//! commute with an end in a solved pack (listed under `solved`), from the fixed packs' homes too;
//! pipeline/market_zones.py takes each pair of packs from the newest file covering it.
//!
//! `--hold-report F [--new a,b]` (T-928, with `--fixed`): for each fixed pack, how far this solve
//! moved its zones' row and column sums from what its old neighbours gave them (the solved packs at
//! their old factors, the `--new` ones absent), and whether it has any pair with a solved pack: how
//! far holding a pack fixed leaves it off its own margins (notes/T-928.md).
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
//! T-1000: the balance's pull is capped per pack (`kernel::PULL_CAP`; `--pull-cap X`, 0 none), and
//! `LONG_SHARE` of every zone's workers (`--long-share`) commute to jobs `LONG_KM` away outside the
//! balance, listed with the cross-pack commutes, their own pack's included.
//!
//! T-1091: every solved pack's curve is held flat inside `kernel::flat_near_km` of it (none on the
//! US curve; `--flat-km D` for trials, 0 none), written to its gravity block; a fixed pack keeps its
//! own. So rerunning a join applies the rule to the packs it solves.
//!
//! The balance (T-297) runs per group of packs joined by any pair within the cut, each to `--tol`
//! on every column (rows exact), with Anderson mixing; `--check` re-measures both after the solve,
//! and `--dump` writes the factors and kept trips for comparing two solves.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use trainworld_sim::demand::{clock::now_ms, intercity, kernel, pack};
use trainworld_sim::track::frame::Frame;

/// T-1000 (docs/zones.md 4): the share of every zone's workers who commute to jobs between
/// `LONG_KM` km from home, shared by jobs times the home pack's curve and not balanced
/// (`--long-share` replaces it; 0 leaves it out). The balanced layer keeps the rest.
const LONG_SHARE: f64 = 0.01;
const LONG_KM: [f64; 2] = [50.0, 150.0];

struct Pk {
    id: String,
    header: String,
    bin: Vec<u8>,
    city: pack::City,
    z: kernel::Zones,
    ftab: Vec<f32>,
    /// zone-centre km by the same index as `ftab`
    dtab: Vec<f32>,
    /// the runtime's index into `ftab` (T-852: zones of two sizes)
    pk: kernel::PairKey,
    border: f32,
    curve: kernel::Curve,
    zone_m: f32,
    /// global region id per pack region
    regions: Vec<usize>,
    /// first global zone index
    off: usize,
    market: usize,
    fixed: bool,
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
    // `--packs a,b,..`: a group outside the Northeast (Japan, T-463), its commutes to `--out`
    let only: Option<Vec<String>> = get("--packs").map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect());
    let fixed_ids: Vec<String> = get("--fixed").map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default();
    assert!(fixed_ids.is_empty() || only.is_some(), "--fixed needs --packs");
    let cities = registry["cities"].as_array().unwrap();
    // Zone distances in a conic: the Northeast's at the market's origin (T-125), a group's at the
    // mean of its packs' origins (1.2% at 9 degrees of latitude off it). The game's own frame is
    // Mercator since T-306 and its plane metres are not ground metres, so it is not used here.
    let world = match &only {
        None => {
            let wf = &registry["market"]["frame"];
            Frame::lcc(wf["lon"].as_f64().unwrap(), wf["lat"].as_f64().unwrap())
        }
        Some(ids) => {
            assert_eq!(cities.iter().filter(|c| fixed_ids.iter().any(|i| c["id"].as_str() == Some(i))).count(), fixed_ids.len(), "--fixed names a city the registry lacks");
            // the frame at the mean of the solved and fixed packs' origins: a fixed pack's distances are the same as in its own solve to 1%
            let os: Vec<&Value> = cities.iter().filter(|c| ids.iter().chain(&fixed_ids).any(|i| c["id"].as_str() == Some(i))).map(|c| &c["origin"]).collect();
            assert_eq!(os.len(), ids.len() + fixed_ids.len(), "--packs names a city the registry lacks");
            let n = os.len() as f64;
            Frame::lcc(os.iter().map(|o| o["lon"].as_f64().unwrap()).sum::<f64>() / n, os.iter().map(|o| o["lat"].as_f64().unwrap()).sum::<f64>() / n)
        }
    };
    let borders: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/pipeline/borders.json")).unwrap()).unwrap();

    // ---- the packs and their zones
    let mut packs: Vec<Pk> = vec![];
    let mut region_key: Vec<(String, String)> = vec![]; // (country, region id)
    let mut off = 0usize;
    for c in cities {
        let market = c["market"].as_i64().unwrap_or(-1);
        let id = c["id"].as_str().unwrap().to_string();
        let fixed = fixed_ids.contains(&id);
        if match &only { Some(ids) => !ids.contains(&id) && !fixed, None => market < 0 } {
            continue;
        }
        let (header, bin) = pack::load_raw(&format!("{root}/data/packs/{id}.json")).unwrap_or_else(|e| panic!("{e}"));
        let city = pack::parse(&header, &bin).unwrap_or_else(|e| panic!("{id}: {e}"));
        let pg = city.gravity.clone().unwrap_or_else(|| panic!("{id} has no gravity"));
        let z = kernel::zones_from_pack(&city, &pg);
        // T-1091: a solved pack takes today's flat near end for its curve (`--flat-km` for
        // trials); a fixed one keeps its own, which its factors were solved with
        let flat_km = if fixed { pg.flat_km } else { get("--flat-km").map_or_else(|| kernel::flat_near_km(pg.decay_km, pg.decay_pow), |v| v.parse().unwrap()) };
        let curve = kernel::Curve { flat_km, ..pg.curve() };
        let (ftab, dtab) = kernel::decay_tables(&z, curve);
        let hv: Value = serde_json::from_str(&header).unwrap();
        // packs built before headers carried a country (tor, lon, par, sao) take the registry's (T-590)
        let country = hv["country"].as_str().or(c["country"].as_str()).unwrap_or("US").to_uppercase();
        let ids: Vec<String> = hv["regions"]["ids"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect()).unwrap_or_else(|| vec!["".into()]);
        // a tier 0 pack's one region ("00") is its own, not every other tier 0 pack's
        let single = ids.len() == 1 && ids[0] == "00";
        let regions = ids
            .iter()
            .map(|r| {
                let k = (country.clone(), if single { format!("pack {id}") } else { r.clone() });
                region_key.iter().position(|x| *x == k).unwrap_or_else(|| {
                    region_key.push(k);
                    region_key.len() - 1
                })
            })
            .collect();
        let n = z.n;
        packs.push(Pk { id, header, bin, pk: kernel::PairKey::of(&z), border: pg.border_factor, curve, zone_m: pg.zone_m, city, z, ftab, dtab, regions, off, market: market.max(0) as usize, fixed });
        off += n;
    }
    let nz = off;
    let np = packs.len();
    let nr = region_key.len();
    eprintln!("{} packs ({} fixed), {} zones, {} regions, loaded in {:.0} ms", np, packs.iter().filter(|p| p.fixed).count(), nz, nr, now_ms() - t0);
    // a fixed pack's factors: its gravity block's, or `--init`'s (a `--dump` of an earlier solve)
    let init: HashMap<String, (Vec<f64>, Vec<f64>)> = match get("--init") {
        None => HashMap::new(),
        Some(f) => {
            let d: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            let (rw, cl) = (d["row"].as_array().unwrap(), d["col"].as_array().unwrap());
            d["packs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let (o, n) = (p[1].as_u64().unwrap() as usize, p[2].as_u64().unwrap() as usize);
                    let v = |a: &Vec<Value>| a[o..o + n].iter().map(|x| x.as_f64().unwrap_or(0.0)).collect::<Vec<f64>>();
                    (p[0].as_str().unwrap().to_string(), (v(rw), v(cl)))
                })
                .collect()
        }
    };
    let mut base_row = vec![0f64; nz];
    let mut base_col = vec![0f64; nz];
    for p in packs.iter().filter(|p| p.fixed) {
        let (r, c) = match init.get(&p.id) {
            Some(rc) => rc.clone(),
            None => {
                let g = p.city.gravity.as_ref().unwrap();
                (g.row.clone(), g.col.clone())
            }
        };
        assert!(r.len() == p.z.n && c.len() == p.z.n, "{}: {} factors for {} zones", p.id, r.len(), p.z.n);
        base_row[p.off..p.off + p.z.n].copy_from_slice(&r);
        base_col[p.off..p.off + p.z.n].copy_from_slice(&c);
    }

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
                // `--admin1 F` overrides the country's default, for fitting it; a country without one
                // takes `pack_line_default` (T-584, T-1163: 0.5), not 1
                get("--admin1")
                    .map(|v| v.parse().unwrap())
                    .or(borders["admin1_default"][ca.as_str()].as_f64())
                    .or(borders["pack_line_default"].as_f64())
                    .unwrap_or(1.0) as f32
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
            let x = p.z.x0 as f64 + p.z.hx(i) as f64 * 0.5 * p.zone_m as f64;
            let y = p.z.y0 as f64 + p.z.hy(i) as f64 * 0.5 * p.zone_m as f64;
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
            let dmin = (0.52 * p.zone_m as f64 / 1000.0).max(p.curve.flat_km as f64);
            (0..nsteps)
                .map(|k| {
                    let d = (k as f64 * step).max(dmin);
                    (d.powf(-p.curve.pow as f64) * (-d / p.curve.km as f64).exp()) as f32
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
        let base = p.pk.base(&p.z, i);
        let ri = p.z.region[i];
        let mut s = 0.0;
        for j in 0..p.z.n {
            let vj = v[p.off + j];
            if vj == 0.0 {
                continue;
            }
            let k = if p.z.region[j] == ri { 1.0 } else { p.border };
            s += vj * (p.ftab[(base + p.pk.key(&p.z, j)) as usize] * k) as f64;
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
    // fixed packs join no group: their factors are given, so nothing is balanced through them
    let fixed_zone = |g: usize| packs[zp[g] as usize].fixed;
    for g in 0..nz {
        if fixed_zone(g) {
            continue;
        }
        cross(g, true, &mut |h, _| {
            npairs += 1;
            if fixed_zone(h) {
                return;
            }
            let (a, b) = (root_of(&mut parent, zp[g] as usize), root_of(&mut parent, zp[h] as usize));
            if a != b {
                parent[a] = b;
            }
        });
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for pi in 0..np {
        if packs[pi].fixed {
            continue;
        }
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

    // ---- T-1000: the add-back's ring. For zone g, every zone h between LONG_KM km from it, in its
    // pack (the pack's table) and across (as `cross`), with w = f(d) x k and d km
    let long_share: f64 = get("--long-share").map_or(LONG_SHARE, |v| v.parse().unwrap());
    let (r0, r1) = (LONG_KM[0], LONG_KM[1].min(cut_km));
    let ring = |g: usize, f: &mut dyn FnMut(usize, f64, f64)| {
        let p = &packs[zp[g] as usize];
        let i = g - p.off;
        let base = p.pk.base(&p.z, i);
        let ri = p.z.region[i];
        for j in 0..p.z.n {
            let key = (base + p.pk.key(&p.z, j)) as usize;
            let d = p.dtab[key] as f64;
            if d > r0 && d <= r1 {
                let k = if p.z.region[j] == ri { 1.0 } else { p.border };
                f(p.off + j, (p.ftab[key] * k) as f64, d);
            }
        }
        cross(g, true, &mut |h, w| {
            let d = ((px[h] - px[g]).powi(2) + (py[h] - py[g]).powi(2)).sqrt() / 1000.0;
            if d > r0 && d <= r1 {
                f(h, w, d);
            }
        });
    };
    // per home zone, share x homes over its ring's jobs x w (0: no ring); a fixed pack's zone only
    // where its ring reaches a solved pack, since only those pairs are listed
    let homes: Vec<usize> = if long_share > 0.0 { (0..nz).filter(|&g| o[g] > 0.0).collect() } else { vec![] };
    let ring_jobs = par_map(&homes, threads, &|g| {
        if fixed_zone(g) {
            let mut near = false;
            ring(g, &mut |h, _, _| near |= !fixed_zone(h));
            if !near {
                return 0.0;
            }
        }
        let mut s = 0.0;
        ring(g, &mut |h, w, _| s += w * jobs[h]);
        s
    });
    let mut long_w = vec![0f64; nz];
    // the balanced layer's homes and jobs: every zone with a ring keeps 1 - share of its workers,
    // and each pack's jobs are scaled to its homes' total, so the shares by pair do not move
    let mut ob = o.clone();
    for (&g, &s) in homes.iter().zip(&ring_jobs) {
        if s > 0.0 {
            long_w[g] = long_share * o[g] / s;
            if !fixed_zone(g) {
                ob[g] = o[g] * (1.0 - long_share);
            }
        }
    }
    let mut kappa = vec![1f64; np];
    let mut db = jobs.clone();
    for (pi, p) in packs.iter().enumerate().filter(|(_, p)| !p.fixed) {
        let r = p.off..p.off + p.z.n;
        let (so, sj): (f64, f64) = (ob[r.clone()].iter().sum(), jobs[r.clone()].iter().sum());
        kappa[pi] = if sj > 0.0 { so / sj } else { 1.0 };
        for g in r {
            db[g] = jobs[g] * kappa[pi];
        }
    }
    if long_share > 0.0 {
        eprintln!("add-back rings: {} home zones, {:.0} s", homes.len(), (now_ms() - t0) / 1000.0);
    }

    // ---- balancing: Furness steps with Anderson mixing (T-297), the pull capped (T-1000)
    let max_iters: usize = get("--max-iters").map_or(500, |v| v.parse().unwrap());
    let mix: usize = get("--anderson").map_or(12, |v| v.parse().unwrap());
    // `--pull-cap 0`: none
    let cap: f64 = get("--pull-cap").map_or(kernel::PULL_CAP, |v| v.parse().unwrap());
    let cap = if cap > 0.0 { cap } else { f64::INFINITY };
    let mut row = base_row.clone();
    let mut col: Vec<f64> = (0..nz).map(|g| if fixed_zone(g) { base_col[g] } else { db[g] }).collect();
    // per pack: the band `col / db` was held in, and the fill of the columns inside it
    let mut band = vec![(0f64, f64::INFINITY); np];
    let mut fill = vec![1f64; np];
    let mut edge = vec![false; nz];
    let (mut iters, mut err) = (0, 0f64);
    for grp in &groups {
        let zs: Vec<usize> = grp.iter().flat_map(|&pi| packs[pi].off..packs[pi].off + packs[pi].z.n).collect();
        let name = grp.iter().map(|&p| packs[p].id.as_str()).collect::<Vec<_>>().join("+");
        let b = balance(&ob, &db, &col, &base_row, &base_col, &zs, &zp, np, cap, &|v, rows| sums(v, rows, &zs), tol, max_iters, mix, &mut |k, e, g| {
            if k % 10 == 0 {
                let p = &packs[zp[g] as usize];
                eprintln!("  {name}: step {k}, max column change {e:.2e} (in {} zone {}), {:.0} s", p.id, g - p.off, (now_ms() - t0) / 1000.0);
            }
        });
        for &g in &zs {
            row[g] = b.row[g];
            col[g] = b.col[g];
            edge[g] = b.edge[g];
        }
        for &pi in grp {
            band[pi] = b.band[pi];
            fill[pi] = b.fill[pi];
        }
        eprintln!("{name}: {} Furness steps, max column change {:.2e} (rows exact), {:.0} s", b.steps, b.err, (now_ms() - t0) / 1000.0);
        iters = iters.max(b.steps);
        err = err.max(b.err);
    }
    // per pack, the share of its jobs in columns held at the band's edges
    let at_edge = |g: usize| edge[g];
    let mut held = vec![0f64; np];
    for (pi, p) in packs.iter().enumerate().filter(|(_, p)| !p.fixed) {
        let r = p.off..p.off + p.z.n;
        held[pi] = r.clone().filter(|&g| at_edge(g)).fold(0.0, |a, g| a + jobs[g]) / jobs[r].iter().sum::<f64>().max(1e-300);
    }
    if has("--check") {
        let all: Vec<usize> = (0..nz).collect();
        let sr = sums(&col, true, &all);
        let sc = sums(&row, false, &all);
        // fixed packs apart: their margins move when the solved packs' factors change (T-786);
        // a solved column inside the pull band is to its pack's fill of its jobs, one at an edge
        // anywhere (T-1000)
        let (mut er, mut ec, mut fr, mut fc) = (0f64, 0f64, 0f64, 0f64);
        let (mut fo, mut fo_off, mut fd, mut fd_off) = (0f64, 0f64, 0f64, 0f64);
        for g in 0..nz {
            let target = if fixed_zone(g) { db[g] } else { db[g] * fill[zp[g] as usize] };
            let (re, ce) = ((row[g] * sr[g] / ob[g].max(1e-300) - 1.0).abs(), (col[g] * sc[g] / target.max(1e-300) - 1.0).abs());
            let (rok, cok) = (ob[g] > 0.0 && sr[g] > 0.0, db[g] > 0.0 && sc[g] > 0.0);
            if fixed_zone(g) {
                if rok {
                    fr = fr.max(re);
                    fo += ob[g];
                    fo_off += (row[g] * sr[g] - ob[g]).abs();
                }
                if cok {
                    fc = fc.max(ce);
                    fd += db[g];
                    fd_off += (col[g] * sc[g] - db[g]).abs();
                }
                continue;
            }
            if rok {
                er = er.max(re);
            }
            if cok && !at_edge(g) {
                ec = ec.max(ce);
            }
        }
        let (hj, tj): (f64, f64) = (0..nz).filter(|&g| !fixed_zone(g)).fold((0., 0.), |a, g| (a.0 + if at_edge(g) { jobs[g] } else { 0. }, a.1 + jobs[g]));
        eprintln!("check: max row error {er:.2e}, max column error {ec:.2e} (inside the pull band; {:.2}% of the jobs at its edges), {:.0} s", 100.0 * hj / tj.max(1e-300), (now_ms() - t0) / 1000.0);
        if fo > 0.0 {
            eprintln!("check, fixed packs: max row error {fr:.2e} (homes off by {:.3e} of theirs in all), max column error {fc:.2e} (jobs off by {:.3e})", fo_off / fo, fd_off / fd.max(1e-300));
        }
        let mut worst: Vec<(f64, usize)> = (0..nz).filter(|&g| !fixed_zone(g) && !at_edge(g) && db[g] > 0.0 && sc[g] > 0.0).map(|g| ((col[g] * sc[g] / (db[g] * fill[zp[g] as usize]) - 1.0).abs(), g)).collect();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(e, g) in worst.iter().take(8) {
            let p = &packs[zp[g] as usize];
            let i = g - p.off;
            eprintln!("  worst column: {} zone {} ({}, {}): error {e:.2e}, jobs {:.1}, homes {:.1}, col {:.3e}", p.id, i, p.z.gx[i], p.z.gy[i], db[g], ob[g], col[g]);
        }
        let mut most: Vec<(f64, usize)> = packs.iter().enumerate().filter(|(_, p)| !p.fixed).map(|(pi, _)| (held[pi], pi)).collect();
        most.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(h, pi) in most.iter().take(8).filter(|x| x.0 > 0.0) {
            eprintln!("  held at the pull band: {} {:.2}% of its jobs, the rest filled to {:.4}", packs[pi].id, 100.0 * h, fill[pi] * kappa[pi]);
        }
    }

    // ---- T-928: how far the solve moved the fixed packs near it
    if let Some(path) = get("--hold-report") {
        let new_ids: Vec<String> = get("--new").map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default();
        // before: every pack's old factors, the new packs' zones taking no part
        let (mut row0, mut col0) = (base_row.clone(), base_col.clone());
        for p in packs.iter().filter(|p| !p.fixed && !new_ids.contains(&p.id)) {
            let (r, c) = match init.get(&p.id) {
                Some(rc) => rc.clone(),
                None => {
                    let g = p.city.gravity.as_ref().unwrap();
                    (g.row.clone(), g.col.clone())
                }
            };
            row0[p.off..p.off + p.z.n].copy_from_slice(&r);
            col0[p.off..p.off + p.z.n].copy_from_slice(&c);
        }
        let solved_grids: Vec<&Grid> = packs.iter().zip(&grids).filter(|(p, _)| !p.fixed).map(|(_, g)| g).collect();
        let near = |g: &Grid| solved_grids.iter().any(|s| g.x0 - reach <= s.x1 && s.x0 - reach <= g.x1 && g.y0 - reach <= s.y1 && s.y0 - reach <= g.y1);
        let zf: Vec<usize> = packs.iter().zip(&grids).filter(|(p, g)| p.fixed && near(g)).flat_map(|(p, _)| p.off..p.off + p.z.n).collect();
        let (ra, ca) = (sums(&col, true, &zf), sums(&row, false, &zf));
        let (rb, cb) = (sums(&col0, true, &zf), sums(&row0, false, &zf));
        let mut out = vec![];
        for p in packs.iter().filter(|p| p.fixed) {
            let (mut dr, mut dc, mut moved, mut base) = (0f64, 0f64, 0f64, 0f64);
            for g in p.off..p.off + p.z.n {
                // zones of under one commuter either end are left out: their ratios are noise
                if o[g] >= 1.0 && rb[g] > 0.0 {
                    dr = dr.max((ra[g] / rb[g] - 1.0).abs());
                    moved += o[g] * (ra[g] / rb[g] - 1.0).abs();
                    base += o[g];
                }
                if jobs[g] >= 1.0 && cb[g] > 0.0 {
                    dc = dc.max((ca[g] / cb[g] - 1.0).abs());
                }
            }
            let touched = (p.off..p.off + p.z.n).any(|g| ra[g] != rb[g] || ca[g] != cb[g]);
            out.push(json!({"id": p.id, "zones": p.z.n, "row_delta": dr, "col_delta": dc, "homes_moved": moved / base.max(1e-300), "touched": touched}));
        }
        let rep = json!({"task": "T-928", "solved": packs.iter().filter(|p| !p.fixed).map(|p| p.id.clone()).collect::<Vec<_>>(), "new": new_ids, "error": err, "fixed": out});
        std::fs::write(&path, serde_json::to_string_pretty(&rep).unwrap()).unwrap();
        eprintln!("hold report: {} fixed packs, {} near the solved ones -> {path}", packs.iter().filter(|p| p.fixed).count(), out.iter().filter(|v| v["touched"].as_bool() == Some(true)).count());
    }

    // ---- what each zone keeps in its pack, and the cross-pack flows
    let mut home = vec![0f64; nz];
    let mut work = vec![0f64; nz];
    for p in packs.iter().filter(|p| !p.fixed) {
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
    // market zones: in the Northeast, each pack cell's 600 m square's site in intercity.json, and
    // its zone; in a `--packs` group, the cell's 6 km square in its own pack's frame (the square
    // pipeline/market_zones.py's own sites fall in), listed in `group_zones`
    let market_path = format!("{root}/data/packs/intercity.json");
    let (site_zone, zone_site, site_keys, mv, mframe) = if only.is_none() {
        let market_text = std::fs::read_to_string(&market_path).unwrap();
        let (site_zone, zone_site, site_keys) = intercity::market_site_zones(&market_text).unwrap();
        let mv: Value = serde_json::from_str(&market_text).unwrap();
        let mframe = Frame::equirect(mv["frame"]["lon"].as_f64().unwrap(), mv["frame"]["lat"].as_f64().unwrap());
        (site_zone, zone_site, site_keys, mv, Some(mframe))
    } else {
        (vec![], vec![], vec![], Value::Null, None)
    };
    let square: HashMap<(i32, i32, usize), u32> = site_keys.iter().enumerate().map(|(s, &k)| (k, s as u32)).collect();
    let mut group_zones: Vec<(usize, i32, i32)> = vec![];
    let mut group_index: HashMap<(usize, i32, i32), u32> = HashMap::new();
    // per global zone: (market zone, share) of its homes and of its jobs
    let mut hsplit: Vec<Vec<(u32, f64)>> = vec![vec![]; nz];
    let mut wsplit: Vec<Vec<(u32, f64)>> = vec![vec![]; nz];
    let mut missed = (0f64, 0f64);
    for (pi, p) in packs.iter().enumerate() {
        let hv: Value = serde_json::from_str(&p.header).unwrap();
        let fr = Frame::equirect(hv["origin"]["lon"].as_f64().unwrap(), hv["origin"]["lat"].as_f64().unwrap());
        let mut acc: Vec<HashMap<u32, (f64, f64)>> = vec![HashMap::new(); p.z.n];
        for c in 0..p.city.len() {
            let mz = if let Some(mframe) = &mframe {
                let (lon, lat) = fr.to_lonlat(p.city.x[c] as f64, p.city.y[c] as f64);
                let (mx, my) = mframe.from_lonlat(lon, lat);
                let key = ((mx / 600.0).floor() as i32, (my / 600.0).floor() as i32, p.market);
                let Some(&s) = square.get(&key) else {
                    missed.0 += p.city.pop[c] as f64;
                    missed.1 += p.city.jobs[c] as f64;
                    continue;
                };
                site_zone[s as usize]
            } else {
                let key = (pi, (p.city.x[c] as f64 / 6000.0).floor() as i32, (p.city.y[c] as f64 / 6000.0).floor() as i32);
                *group_index.entry(key).or_insert_with(|| {
                    group_zones.push(key);
                    (group_zones.len() - 1) as u32
                })
            };
            let e = acc[p.z.cell_zone[c] as usize].entry(mz).or_default();
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
        let gf = fixed_zone(g);
        cross(g, true, &mut |h, w| {
            // a fixed pack's commutes into the solved packs are listed too (the newest solve covering a pair
            // gives it, T-786); between two fixed packs, nowhere
            if gf && fixed_zone(h) {
                return;
            }
            let t = row[g] * col[h] * w;
            if t <= 0.0 {
                return;
            }
            by_pack[zp[g] as usize * np + zp[h] as usize] += t;
            if !gf {
                cross_km.0 += t;
                cross_km.1 += t * ((px[h] - px[g]).powi(2) + (py[h] - py[g]).powi(2)).sqrt() / 1000.0;
            }
            for &(mh, sh) in &hsplit[g] {
                for &(mw, sw) in &wsplit[h] {
                    *flows.entry((mh, mw)).or_default() += t * sh * sw;
                }
            }
        });
    }
    eprintln!("flows between packs, {:.0} s", (now_ms() - t0) / 1000.0);
    // T-1000: the add-back's pairs, a pack's own included; pooled by market zone of work per home
    // zone before the market zones of home
    let nmz = if mframe.is_some() { zone_site.len() } else { group_zones.len() };
    let mut long_flows: HashMap<(u32, u32), f64> = HashMap::new();
    let mut by_work = vec![0f64; nmz];
    let mut touched: Vec<u32> = vec![];
    let mut long_out = vec![0f64; np];
    let mut long_km = (0f64, 0f64, 0f64);
    for g in (0..nz).filter(|&g| long_w[g] > 0.0) {
        let gf = fixed_zone(g);
        ring(g, &mut |h, w, d| {
            if gf && fixed_zone(h) {
                return;
            }
            let t = long_w[g] * w * jobs[h];
            if t <= 0.0 {
                return;
            }
            let (a, b) = (zp[g] as usize, zp[h] as usize);
            by_pack[a * np + b] += t;
            long_out[a] += t;
            if !gf {
                long_km.0 += t;
                long_km.1 += t * d;
                if a == b {
                    long_km.2 += t;
                }
            }
            for &(mw, sw) in &wsplit[h] {
                if by_work[mw as usize] == 0.0 {
                    touched.push(mw);
                }
                by_work[mw as usize] += t * sw;
            }
        });
        for &mw in &touched {
            for &(mh, sh) in &hsplit[g] {
                *long_flows.entry((mh, mw)).or_default() += by_work[mw as usize] * sh;
            }
            by_work[mw as usize] = 0.0;
        }
        touched.clear();
    }
    let total_within: f64 = home.iter().sum();
    let total_cross: f64 = (0..np * np).filter(|k| k / np != k % np).map(|k| by_pack[k]).sum();
    println!(
        "commuters: {:.0} in all, {:.0} within their pack, {:.0} ({:.2}%) to another pack, the balanced ones' mean {:.1} km",
        tj,
        total_within,
        total_cross,
        100.0 * total_cross / tj,
        cross_km.1 / cross_km.0.max(1e-9)
    );
    if long_share > 0.0 {
        println!(
            "add-back (T-1000): {:.0} commuters ({:.2}% of the solved packs' workers) to jobs {r0}-{r1} km away, mean {:.1} km, {:.0} of them inside their pack",
            long_km.0,
            100.0 * long_km.0 / packs.iter().filter(|p| !p.fixed).map(|p| o[p.off..p.off + p.z.n].iter().sum::<f64>()).sum::<f64>().max(1e-9),
            long_km.1 / long_km.0.max(1e-9),
            long_km.2
        );
    }
    println!("commuters between packs (home row, work column):");
    for a in 0..np {
        let line: Vec<String> = (0..np).filter(|&b| b != a && by_pack[a * np + b] >= 1.0).map(|b| format!("{} {:.0}", packs[b].id, by_pack[a * np + b])).collect();
        println!("  {}: {}", packs[a].id, line.join(", "));
    }
    // the commute list: pairs under `--min` commuters a day are dropped and their commuters given
    // to the home zone's kept pairs in proportion, so every zone keeps its commuters; the add-back
    // apart, so its dropped commuters stay in its own distances
    let min: f64 = get("--min").map_or(1.0, |v| v.parse().unwrap());
    let listed: f64 = flows.values().chain(long_flows.values()).sum();
    for m in [0.02, 0.1, 0.2, 0.5, 1.0, 2.0] {
        let (n, t) = flows.values().chain(long_flows.values()).filter(|t| **t >= m).fold((0, 0.0), |a, t| (a.0 + 1, a.1 + t));
        println!("  pairs of at least {m} a day: {n}, {:.1}% of the commuters", 100.0 * t / listed.max(1e-9));
    }
    let mut merged: HashMap<(u32, u32), f64> = HashMap::new();
    for (name, part) in [("balanced", flows), ("add-back", long_flows)] {
        let (before, mut after): (f64, f64) = (part.values().sum(), 0.0);
        for (k, t) in keep_pairs(part, min) {
            *merged.entry(k).or_default() += t;
            after += t;
        }
        println!("  {name} pairs: {after:.0} of {before:.0} commuters kept");
    }
    let mut list: Vec<((u32, u32), f64)> = merged.into_iter().collect();
    list.sort_by(|a, b| a.0.cmp(&b.0));
    let kept: f64 = list.iter().map(|x| x.1).sum();
    println!("market zone pairs: {} kept (at least {min} a day), {:.0} commuters after giving the dropped ones to each home zone's kept pairs ({:.2}% of those listed; home zones with no kept pair lose theirs)", list.len(), kept, 100.0 * kept / listed.max(1e-9));

    let solved: Vec<String> = packs.iter().filter(|p| !p.fixed).map(|p| p.id.clone()).collect();
    let report = json!({
        "packs": packs.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "solved": solved,
        "by_pack": by_pack,
        "commuters": tj, "within": total_within, "cross": total_cross,
        "border_pairs": table, "iterations": iters, "max_error": err, "cut_km": cut_km,
        "pull_cap": if cap.is_finite() { json!(cap) } else { Value::Null },
        "held": packs.iter().enumerate().filter(|(_, p)| !p.fixed).map(|(pi, p)| (p.id.clone(), json!(held[pi]))).collect::<serde_json::Map<_, _>>(),
        "long": {"share": long_share, "km": [r0, r1], "commuters": long_km.0, "mean_km": long_km.1 / long_km.0.max(1e-9), "within_pack": long_km.2},
    });
    if let Some(p) = get("--report") {
        std::fs::write(&p, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    if has("--dry") {
        eprintln!("--dry: nothing written");
        return;
    }
    // ---- write the packs
    for p in packs.iter().filter(|p| !p.fixed) {
        let mut pg = p.city.gravity.clone().unwrap();
        pg.flat_km = p.curve.flat_km;
        let r = p.off..p.off + p.z.n;
        pg.row = row[r.clone()].to_vec();
        pg.col = col[r.clone()].to_vec();
        pg.home = home[r.clone()].to_vec();
        pg.work = work[r.clone()].to_vec();
        pg.trips = pg.home.iter().sum();
        pg.iters = iters;
        pg.max_err = err;
        let pi = zp[p.off] as usize;
        let out: f64 = (0..np).filter(|&b| b != pi).map(|b| by_pack[pi * np + b]).sum();
        let into: f64 = (0..np).filter(|&a| a != pi).map(|a| by_pack[a * np + pi]).sum();
        // T-1000: the band and fill in `col / jobs`, as the pack's own solve writes them
        pg.pull = cap.is_finite().then(|| json!({"cap": cap, "band": [band[pi].0 * kappa[pi], band[pi].1 * kappa[pi]], "fill": fill[pi] * kappa[pi], "held": held[pi]}));
        pg.joint = Some(json!({
            "packs": packs.iter().map(|q| q.id.clone()).collect::<Vec<_>>(),
            "solved": solved,
            "cut_km": cut_km,
            "commuters_out": out,
            "commuters_in": into,
            "border_pairs": "pipeline/borders.json and each pack's own factor for the regions it holds",
            "long_share": long_share,
            "long_km": [r0, r1],
            "long_out": long_out[pi],
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
    if only.is_some() {
        let out = get("--out").expect("--packs needs --out");
        let zk = |m: u32| group_zones[m as usize];
        let g = json!({
            "task": "T-463", "packs": packs.iter().map(|p| p.id.clone()).collect::<Vec<_>>(), "zone_m": 6000,
            // the packs solved; a pair of packs with an end among them is this file's until a newer one covers it (T-786)
            "solved": solved,
            "solved_at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()),
            "frame": "each pack's own equirectangular frame at its origin; a zone is [floor(x / 6000), floor(y / 6000)]",
            "source": format!("bin/joint_gravity.rs (T-276, T-463, T-1000): the cross-pack part of one commute gravity over {np} packs, cut at {cut_km} km, and {long_share} of every zone's workers to jobs {r0}-{r1} km away"),
            "home_pack": list.iter().map(|x| zk(x.0 .0).0).collect::<Vec<_>>(),
            "home_zone": list.iter().map(|x| [zk(x.0 .0).1, zk(x.0 .0).2]).collect::<Vec<_>>(),
            "work_pack": list.iter().map(|x| zk(x.0 .1).0).collect::<Vec<_>>(),
            "work_zone": list.iter().map(|x| [zk(x.0 .1).1, zk(x.0 .1).2]).collect::<Vec<_>>(),
            "trips": list.iter().map(|x| (x.1 * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        });
        let text = serde_json::to_string(&g).unwrap();
        std::fs::write(format!("{out}.part"), &text).unwrap();
        std::fs::rename(format!("{out}.part"), &out).unwrap();
        println!("wrote {} packs and {out} ({:.2} MB) in {:.0} s", np, text.len() as f64 / 1e6, (now_ms() - t0) / 1000.0);
        return;
    }
    let mut mv = mv;
    mv["commute_home"] = json!(list.iter().map(|x| zone_site[x.0 .0 as usize]).collect::<Vec<_>>());
    mv["commute_work"] = json!(list.iter().map(|x| zone_site[x.0 .1 as usize]).collect::<Vec<_>>());
    mv["commute_trips"] = json!(list.iter().map(|x| (x.1 * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    mv["commute_source"] = json!(format!("bin/joint_gravity.rs (T-276): the cross-pack part of one commute gravity over {} packs, cut at {cut_km} km, and {long_share} of every zone's workers to jobs {r0}-{r1} km away (T-1000); home and work sites name their 6 km zones", np));
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
    /// max over columns of the step's change |col' / col - 1|: their error against their fill of
    /// d inside the pull band; the rows are exact
    err: f64,
    /// per pack (`part`), the band `col / d` was held in
    band: Vec<(f64, f64)>,
    /// per pack, the column sums over d of the columns inside the band (1 when none is held)
    fill: Vec<f64>,
    /// per zone: its column held at the band
    edge: Vec<bool>,
}

/// `f` over `zs`, on `threads` threads taking blocks of zones as they free up.
fn par_map(zs: &[usize], threads: usize, f: &(dyn Fn(usize) -> f64 + Sync)) -> Vec<f64> {
    const BLOCK: usize = 16;
    let next = AtomicUsize::new(0);
    let mut out = vec![0f64; zs.len()];
    let parts: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                s.spawn(|| {
                    let mut mine = vec![];
                    loop {
                        let lo = next.fetch_add(1, Ordering::Relaxed) * BLOCK;
                        if lo >= zs.len() {
                            break;
                        }
                        mine.push((lo, zs[lo..(lo + BLOCK).min(zs.len())].iter().map(|&g| f(g)).collect()));
                    }
                    mine
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (lo, vals) in parts.into_iter().flatten() {
        out[lo..lo + vals.len()].copy_from_slice(&vals);
    }
    out
}

/// Pairs of at least `min` a day; the dropped ones' commuters go to their home zone's kept pairs in
/// proportion (a home zone with none loses them).
fn keep_pairs(flows: HashMap<(u32, u32), f64>, min: f64) -> Vec<((u32, u32), f64)> {
    let mut home_all: HashMap<u32, f64> = HashMap::new();
    let mut home_kept: HashMap<u32, f64> = HashMap::new();
    for (&(h, _), &t) in &flows {
        *home_all.entry(h).or_default() += t;
        if t >= min {
            *home_kept.entry(h).or_default() += t;
        }
    }
    flows.into_iter().filter(|(_, t)| *t >= min).map(|(k, t)| (k, t * home_all[&k.0] / home_kept[&k.0])).collect()
}

/// The balance over the zones `zs`: row = o / (W col), col = d / (W^T row), where `sums(v, true)` is
/// W v and `sums(v, false)` is W^T v. A Furness step maps log col to log col and the solution is its
/// fixed point. Plain Furness crawls along the few slow modes between weakly joined packs and
/// regions (500 steps to 2e-4 on the Northeast); Anderson mixing of the last `mix` steps removes
/// them (T-297). `mix` 0 is plain Furness. Zones outside `zs` keep `base_row` and `base_col` (a fixed
/// pack's factors, T-786; zero otherwise).
///
/// T-1000: a column's pull 1 / (W^T row) is held within `cap` of its pack's (`part`, `np` packs)
/// d-weighted geometric mean pull, as `kernel::gravity` holds it. Each pack's columns then move by
/// one factor so that they still meet its d's total (exactly 1 when none is held): its columns
/// inside the band share what the held ones leave or take over, and its net exchange with every
/// other pack stays nil, as without the cap.
#[allow(clippy::too_many_arguments)]
fn balance(o: &[f64], d: &[f64], c0: &[f64], base_row: &[f64], base_col: &[f64], zs: &[usize], part: &[u16], np: usize, cap: f64, sums: &dyn Fn(&[f64], bool) -> Vec<f64>, tol: f64, max_steps: usize, mix: usize, log: &mut dyn FnMut(usize, f64, usize)) -> Balanced {
    let lcap = cap.ln();
    let act: Vec<usize> = zs.iter().copied().filter(|&g| d[g] > 0.0).collect();
    let mut x: Vec<f64> = act.iter().map(|&g| if c0[g] > 0.0 { c0[g] } else { d[g] }.ln()).collect();
    // a column no home reaches gets 0, as in plain Furness
    let mut dead = vec![false; act.len()];
    let mut hist: VecDeque<(Vec<f64>, Vec<f64>)> = VecDeque::new();
    let mut prev: Option<(Vec<f64>, Vec<f64>, f64)> = None;
    let mut best: Option<Balanced> = None;
    let mut c = base_col.to_vec();
    for k in 0..max_steps {
        for (i, &g) in act.iter().enumerate() {
            c[g] = if dead[i] { 0.0 } else { x[i].exp() };
        }
        let s = sums(&c, true);
        let mut row = base_row.to_vec();
        for &g in zs {
            if o[g] > 0.0 && s[g] > 0.0 {
                row[g] = o[g] / s[g];
            }
        }
        let s = sums(&row, false);
        // each pack's mean log pull
        let mut lp = vec![0f64; np];
        let mut wp = vec![0f64; np];
        for &g in &act {
            if s[g] > 0.0 {
                lp[part[g] as usize] -= d[g] * s[g].ln();
                wp[part[g] as usize] += d[g];
            }
        }
        for (l, w) in lp.iter_mut().zip(&wp) {
            if *w > 0.0 {
                *l /= w;
            }
        }
        let mut gx = vec![0f64; act.len()];
        let mut f = vec![0f64; act.len()];
        let mut edge = vec![false; d.len()];
        // per pack, d's total as the columns would meet it this step
        let mut met = vec![0f64; np];
        for (i, &g) in act.iter().enumerate() {
            if s[g] > 0.0 {
                let (lb, m) = (-s[g].ln(), lp[part[g] as usize]);
                if lb < m - lcap || lb > m + lcap {
                    let e = lb.clamp(m - lcap, m + lcap);
                    gx[i] = d[g].ln() + e;
                    met[part[g] as usize] += d[g] * (e - lb).exp();
                    edge[g] = true;
                } else {
                    gx[i] = (d[g] / s[g]).ln();
                    met[part[g] as usize] += d[g];
                }
            } else {
                dead[i] = true;
                gx[i] = x[i];
            }
        }
        let shift: Vec<f64> = met.iter().zip(&wp).map(|(&m, &w)| if w > 0.0 { (m / w).ln() } else { 0.0 }).collect();
        let (mut err, mut worst) = (0f64, 0usize);
        for (i, &dd) in dead.iter().enumerate() {
            if !dd {
                gx[i] -= shift[part[act[i]] as usize];
                f[i] = gx[i] - x[i];
                let e = (-f[i]).exp_m1().abs();
                if e > err {
                    (err, worst) = (e, act[i]);
                }
            }
        }
        log(k, err, worst);
        if best.as_ref().is_none_or(|b| err < b.err) {
            let band = lp.iter().zip(&shift).map(|(&m, &sh)| ((m - lcap - sh).exp(), (m + lcap - sh).exp())).collect();
            best = Some(Balanced { row, col: c.clone(), steps: k + 1, err, band, fill: shift.iter().map(|sh| (-sh).exp()).collect(), edge });
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
        let z = vec![0f64; n];
        let part = vec![0u16; n];
        let slow = balance(&o, &d, &d, &z, &z, &zs, &part, 1, f64::INFINITY, &sums, 1e-10, 50000, 0, &mut |_, _, _| {});
        let fast = balance(&o, &d, &d, &z, &z, &zs, &part, 1, f64::INFINITY, &sums, 1e-10, 50000, 8, &mut |_, _, _| {});
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
        // T-786: block A solved with block B held at the joint solution gives the joint flows back
        let a: Vec<usize> = (0..40).collect();
        let (mut br, mut bc) = (vec![0f64; n], vec![0f64; n]);
        for g in 40..n {
            br[g] = fast.row[g];
            bc[g] = fast.col[g];
        }
        let part = balance(&o, &d, &d, &br, &bc, &a, &part, 1, f64::INFINITY, &sums, 1e-10, 50000, 8, &mut |_, _, _| {});
        for i in 0..40 {
            for j in 0..n {
                let (x, y) = (fast.row[i] * fast.col[j], part.row[i] * part.col[j]);
                assert!((x - y).abs() <= 1e-6 * x.max(y).max(1e-300), "fixed: {i} {j} {x} {y}");
            }
        }
    }

    /// T-1000: a far cluster with five times the jobs of its homes. Its columns are held at the
    /// band, short of their jobs; the others share what they leave at one fill; rows exact.
    #[test]
    fn pull_cap_holds_far_surplus() {
        let pos: Vec<f64> = (0..40).map(|i| i as f64).chain((0..10).map(|i| 200.0 + i as f64)).collect();
        let n = pos.len();
        let w: Vec<f64> = (0..n * n).map(|k| (-(pos[k / n] - pos[k % n]).abs() / 6.0).exp()).collect();
        let mut o: Vec<f64> = (0..n).map(|i| if i < 40 { 10.0 } else { 1.0 }).collect();
        let d: Vec<f64> = (0..n).map(|i| if i < 40 { 10.0 } else { 5.0 }).collect();
        let (so, sd): (f64, f64) = (o.iter().sum(), d.iter().sum());
        o.iter_mut().for_each(|v| *v *= sd / so);
        let sums = |v: &[f64], rows: bool| -> Vec<f64> { (0..n).map(|g| (0..n).map(|h| v[h] * if rows { w[g * n + h] } else { w[h * n + g] }).sum()).collect() };
        let zs: Vec<usize> = (0..n).collect();
        let z = vec![0f64; n];
        let part = vec![0u16; n];
        let b = balance(&o, &d, &d, &z, &z, &zs, &part, 1, 20.0, &sums, 1e-11, 5000, 8, &mut |_, _, _| {});
        assert!(b.err < 1e-10, "err {} after {} steps", b.err, b.steps);
        let (sr, sc) = (sums(&b.col, true), sums(&b.row, false));
        for g in 0..n {
            assert!((b.row[g] * sr[g] / o[g] - 1.0).abs() < 1e-9, "row {g}");
            let r = b.col[g] * sc[g] / d[g];
            if b.edge[g] {
                assert!(g >= 40 && r < 0.5 && (b.col[g] / d[g] / b.band[0].1 - 1.0).abs() < 1e-8, "held {g}: fill {r}");
            } else {
                assert!((r / b.fill[0] - 1.0).abs() < 1e-8, "inside {g}: {r} against {}", b.fill[0]);
            }
        }
        assert!(b.edge[40..].iter().all(|&e| e) && b.fill[0] > 1.0);
        let total: f64 = (0..n).map(|g| b.col[g] * sc[g]).sum();
        assert!((total / sd - 1.0).abs() < 1e-9);
    }
}
