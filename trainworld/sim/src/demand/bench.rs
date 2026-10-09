//! Runs the kernel stage by stage with timings and heap figures. Same code natively and in wasm.

use super::alloc;
use super::clock::now_ms;
use super::kernel::*;
use super::network::{Network, PERIODS};
use super::pack::City;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct Opts {
    pub zone_m: f32,
    pub band_min: f32,
    /// Also run one subzone per cell (exact access) and compare.
    pub exact: bool,
    pub edit: bool,
    pub periods: bool,
    /// Extra runs over zone sizes and band widths.
    pub sweep: bool,
    pub prune_trips: f32,
    /// Number of crowding rounds after the free-flow one.
    pub crowd_rounds: usize,
    /// Per-city rail propensity (logit constant).
    pub asc_rail: f32,
    /// Solve the zone gravity even when the pack ships it (the pre-T-019 city open).
    pub solve_gravity: bool,
    /// Station-choice logit scale (T-020) and the fast walk's reach (T-078); None = kernel
    /// defaults.
    pub theta: Option<f32>,
    pub walk_cutoff_m: Option<f32>,
    /// Any other parameter by name (`Params::set_extra`).
    pub sets: Vec<(String, f32)>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            zone_m: 2000.0,
            band_min: 0.0,
            exact: false,
            edit: true,
            periods: true,
            sweep: false,
            prune_trips: 0.0,
            crowd_rounds: 1,
            asc_rail: 0.0,
            solve_gravity: false,
            theta: None,
            walk_cutoff_m: None,
            sets: vec![],
        }
    }
}

pub struct Rep(pub String);

impl Rep {
    fn line(&mut self, s: impl AsRef<str>) {
        self.0.push_str(s.as_ref());
        self.0.push('\n');
    }
    fn stage(&mut self, name: &str, ms: f64) {
        let _ = writeln!(
            self.0,
            "  {:<34} {:>9.1} ms   heap {:>7.1} MB  peak {:>7.1} MB",
            name,
            ms,
            alloc::mb(alloc::current()),
            alloc::mb(alloc::peak())
        );
    }
}

macro_rules! timed {
    ($rep:expr, $name:expr, $e:expr) => {{
        let t0 = now_ms();
        let v = $e;
        let ms = now_ms() - t0;
        $rep.stage($name, ms);
        (v, ms)
    }};
}

/// Result of one network-dependent recompute.
pub struct Outcome {
    pub ms_total: f64,
    pub ms_first: f64,
    pub od0: Vec<f32>,
    pub loads0: Loads,
    pub loads: Loads,
    pub rail0: f64,
    pub rail: f64,
    pub trips_total: f64,
    pub subzones: usize,
    pub pairs: u64,
}

/// Everything that depends on the network: access, subzones, search, mode choice, assignment,
/// crowding rounds. Gravity is passed in (network-independent).
pub fn recompute(city: &City, net: &Network, z: &Zones, gr: &Gravity, p: &Params, rounds: usize, rep: &mut Rep, verbose: bool) -> Outcome {
    let t_start = now_ms();
    let (acc, _) = timed!(rep, "access: 6 nearest stations/cell", build_access(city, net, p));
    let (sz, _) = timed!(rep, "access subzones", build_subzones(city, z, &acc, net, p));
    drop(acc);
    let (mut g, _) = timed!(rep, "network graph", build_graph(net, p));
    let (sk, _) = timed!(rep, "station-to-station search", skims(&g, p));
    let (er, _) = timed!(rep, "station-to-station exp table", station_exp(&sk, p));
    let (mo, _) = timed!(rep, "mode choice over subzone pairs", mode_choice(z, gr, &sz, &er, g.n_st, p));
    let (loads0, _) = timed!(rep, "assignment", assign(&g, &sk, &mo.od));
    let ms_first = now_ms() - t_start;
    if verbose {
        rep.line(format!(
            "    subzones {} (origins {}, destinations {}); zone pairs {}; subzone pairs {} ({:.1}M); pruned trips {:.0}",
            sz.n,
            sz.zo.len(),
            sz.zd.len(),
            mo.zone_pairs,
            mo.pairs,
            mo.pairs as f64 / 1e6,
            mo.pruned_trips
        ));
        rep.line(format!(
            "    trips in period {:.0}; with a station within reach at both ends {:.0} ({:.1}%)",
            mo.trips_total,
            mo.trips_access,
            100.0 * mo.trips_access / mo.trips_total,
        ));
        rep.line(format!("    rail {:.0} = {:.1}% of all trips", mo.rail, 100.0 * mo.rail / mo.trips_total));
        let mut edges = String::new();
        let mut lo = 0.0f32;
        for (b, &hi) in DIST_BANDS_KM.iter().enumerate() {
            let w = mo.rail_band[b];
            if w > 0.0 {
                let label = if hi.is_finite() { format!("{lo}-{hi}") } else { format!("{lo}+") };
                let _ = write!(edges, " {label} km {:.1}%;", 100.0 * w / mo.rail);
            }
            lo = hi;
        }
        rep.line(format!("    rail trips by home-to-boarding-station distance:{edges}"));
        rep.line(format!("    first estimate (free flow) ready after {:.1} ms", ms_first));
    }
    let od0 = mo.od.clone();
    let rail0 = mo.rail;
    let pairs = mo.pairs;
    let mut avg_loads = loads0.clone();
    let mut rail = mo.rail;
    drop(mo);
    drop(sk);
    drop(er);
    for round in 1..=rounds {
        let t0 = now_ms();
        let (fac, cs) = crowd_factors(&g, net, &avg_loads, p);
        if verbose {
            rep.line(format!(
                "    crowding before round {round}: {} of {} segments over seats at the peak hour, worst {:.0}% of crush, penalty p50 +{:.0}% p90 +{:.0}% (rider-minute weighted)",
                cs.segs_over_seats,
                cs.segs,
                100.0 * cs.max_load_of_crush,
                100.0 * cs.p50,
                100.0 * cs.p90
            ));
        }
        set_crowding(&mut g, &fac);
        let sk = skims(&g, p);
        let er = station_exp(&sk, p);
        let mo = mode_choice(z, gr, &sz, &er, g.n_st, p);
        let l = assign(&g, &sk, &mo.od);
        msa(&mut avg_loads.seg, &l.seg, round);
        msa(&mut avg_loads.board, &l.board, round);
        msa(&mut avg_loads.alight, &l.alight, round);
        rail += (mo.rail - rail) / (round as f64 + 1.0);
        rep.stage(&format!("crowding round {round} (search..assign, MSA)"), now_ms() - t0);
    }
    if verbose && rounds > 0 {
        let (_, cs) = crowd_factors(&g, net, &avg_loads, p);
        rep.line(format!(
            "    after averaging: {} segments over seats, worst {:.0}% of crush, penalty p50 +{:.0}% p90 +{:.0}%; rail {:.0}",
            cs.segs_over_seats,
            100.0 * cs.max_load_of_crush,
            100.0 * cs.p50,
            100.0 * cs.p90,
            rail
        ));
    }
    let ms_total = now_ms() - t_start;
    Outcome {
        ms_total,
        ms_first,
        od0,
        loads0,
        loads: avg_loads,
        rail0,
        rail,
        trips_total: gr.total * p.period_share as f64,
        subzones: sz.n,
        pairs,
    }
}

/// Weighted relative error sum|a-b| / sum|b|.
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum();
    let den: f64 = b.iter().map(|&y| (y as f64).abs()).sum();
    num / den.max(1e-9)
}

fn row_sums(od: &[f32], s: usize) -> Vec<f32> {
    (0..s).map(|i| od[i * s..(i + 1) * s].iter().sum()).collect()
}

fn col_sums(od: &[f32], s: usize) -> Vec<f32> {
    let mut c = vec![0f32; s];
    for i in 0..s {
        for j in 0..s {
            c[j] += od[i * s + j];
        }
    }
    c
}

fn compare(rep: &mut Rep, label: &str, a: &Outcome, exact: &Outcome, s: usize) {
    rep.line(format!(
        "    {label}: rail {:+.2}% vs exact; station entries off by {:.1}%, exits {:.1}%, segment loads {:.1}% (sum|diff|/sum)",
        100.0 * (a.rail0 / exact.rail0 - 1.0),
        100.0 * rel_err(&row_sums(&a.od0, s), &row_sums(&exact.od0, s)),
        100.0 * rel_err(&col_sums(&a.od0, s), &col_sums(&exact.od0, s)),
        100.0 * rel_err(&a.loads0.seg, &exact.loads0.seg)
    ));
}

pub fn run(city: &City, net0: &Network, o: &Opts) -> String {
    let mut rep = Rep(String::new());
    alloc::reset_peak();
    let heap0 = alloc::current();
    let (tp, tj) = city.totals();
    rep.line(format!(
        "city {}: {} cells, pop {:.2}M, jobs {:.2}M; heap before kernel {:.1} MB",
        city.name,
        city.len(),
        tp / 1e6,
        tj / 1e6,
        alloc::mb(heap0)
    ));
    let mut net = net0.clone();
    let g0 = build_graph(&net, &Params::default());
    rep.line(format!(
        "network: {} lines, {} stations, {} route nodes, {} edges",
        net.n_lines(),
        net.n_stations(),
        g0.rn_line.len(),
        g0.adj_to.len()
    ));
    drop(g0);
    let mut p = Params { zone_m: o.zone_m, band_min: o.band_min, prune_trips: o.prune_trips, asc_rail: o.asc_rail, ..Params::default() };
    if let Some(v) = o.theta {
        p.station_theta = v;
    }
    if let Some(v) = o.walk_cutoff_m {
        p.walk_cutoff_m = v;
    }
    for (k, v) in &o.sets {
        if let Err(e) = p.set_extra(k, *v) {
            rep.line(e);
        }
    }
    rep.line(format!(
        "params: zone {} m, walk band {} min, walk cutoff {} m, decay d^-{} exp(-d/{} km), rail constant {}, crowd rounds {}",
        p.zone_m, p.band_min, p.walk_cutoff_m, p.decay_pow, p.decay_km, p.asc_rail, o.crowd_rounds
    ));
    rep.line(format!(
        "        station logit {} per perceived min; access a fast walk at {} m/s x {} detour, weight {} for {} min then {}; both walks under {} perceived min",
        p.station_theta, p.walk_mps, p.walk_detour, p.w_walk, p.walk_easy_min, p.w_walk_far, p.pair_walk_max
    ));

    rep.line("\nonce per city (network-independent):");
    let solve = o.solve_gravity || city.gravity.is_none();
    let ((z, gr, from_pack), _) = timed!(
        rep,
        if solve { "city open: solve zone gravity" } else { "city open: zone gravity from pack" },
        if solve {
            let z = build_zones(city, p.zone_m);
            let gr = gravity(&z, &p, 100, 1e-3);
            (z, gr, false)
        } else {
            city_gravity(city, &p)
        }
    );
    rep.line(format!(
        "    {} zones ({}); Furness {} iterations, max row error {:.1e}; mean trip {:.1} km; {:.0} trips",
        z.n,
        if from_pack { "factors from the pack" } else { "solved here" },
        gr.iters,
        gr.max_err,
        gr.mean_km,
        gr.total
    ));

    rep.line("\nfull recompute, morning peak:");
    let base = recompute(city, &net, &z, &gr, &p, o.crowd_rounds, &mut rep, true);
    rep.line(format!("  TOTAL recompute {:.1} ms (first estimate {:.1} ms)", base.ms_total, base.ms_first));

    if o.edit {
        // move the Second Av-Bronx line's own stations 400 m east, 300 m south
        let l = net.line_name.iter().position(|n| n == "Second Av-Bronx").unwrap_or(0);
        let moved = net.move_line(l, 400.0, -300.0);
        rep.line(format!("\nedit: {} stations of line {} moved 400 m; same recompute (gravity reused):", moved, net.line_name[l]));
        let e = recompute(city, &net, &z, &gr, &p, o.crowd_rounds, &mut rep, false);
        rep.line(format!(
            "  TOTAL edit recompute {:.1} ms (first estimate {:.1} ms); rail {:.0} -> {:.0}",
            e.ms_total, e.ms_first, base.rail, e.rail
        ));
        net = net0.clone();
    }

    if o.periods {
        rep.line("\nfive periods, each with its own headways (access and subzones shared):");
        let t0 = now_ms();
        let shares = [0.45f32, 0.15, 0.40, 0.15, 0.05];
        let mut quiet = Rep(String::new());
        let mut per = vec![];
        for (k, &sh) in shares.iter().enumerate().take(PERIODS) {
            let pp = Params { period: k, period_share: sh, ..p.clone() };
            let t1 = now_ms();
            let r = recompute(city, &net, &z, &gr, &pp, o.crowd_rounds, &mut quiet, false);
            per.push(format!("{:.0}", now_ms() - t1));
            let _ = r;
        }
        rep.stage("five periods, run one by one", now_ms() - t0);
        rep.line(format!("    per period ms: {} (each repeats access + subzones, ~shared in a real build)", per.join(", ")));
    }

    if o.exact {
        rep.line("\naccuracy of subzones against exact per-cell access (free-flow round, same gravity):");
        let mut quiet = Rep(String::new());
        // the reference is the full model: one subzone per cell and no pruning
        let pe = Params { band_min: -1.0, prune_trips: 0.0, ..p.clone() };
        let t0 = now_ms();
        let ex = recompute(city, &net, &z, &gr, &pe, 0, &mut quiet, true);
        rep.stage("exact: one subzone per cell", now_ms() - t0);
        for l in quiet.0.lines().filter(|l| l.contains("subzones") || l.contains("mode choice") || l.contains("trips in") || l.contains("rail")) {
            rep.line(l);
        }
        let s = net.n_stations();
        compare(&mut rep, &format!("band {} min", p.band_min), &base, &ex, s);
        if o.sweep {
            for band in [0.0f32, 10.0, 2.5] {
                let pb = Params { band_min: band, ..p.clone() };
                let mut q = Rep(String::new());
                let t0 = now_ms();
                let r = recompute(city, &net, &z, &gr, &pb, 0, &mut q, false);
                rep.line(format!("    band {band} min: {} subzones, {:.1}M pairs, free-flow recompute {:.0} ms", r.subzones, r.pairs as f64 / 1e6, now_ms() - t0));
                compare(&mut rep, &format!("band {band} min"), &r, &ex, s);
            }
        }
    }

    if o.sweep {
        rep.line("\nzone size sweep (gravity once + free-flow recompute):");
        for zm in [1000.0f32, 3000.0, 4000.0] {
            p.zone_m = zm;
            let t0 = now_ms();
            let z2 = build_zones(city, zm);
            let g2 = gravity(&z2, &p, 100, 1e-3);
            let tg = now_ms() - t0;
            let mut q = Rep(String::new());
            let r = recompute(city, &net, &z2, &g2, &p, 0, &mut q, true);
            let mc = q.0.lines().find(|l| l.contains("mode choice")).unwrap_or("").trim().to_string();
            rep.line(format!(
                "    zone {zm} m: {} zones, gravity {:.0} ms ({} it), mean trip {:.1} km; {} subzones; free-flow recompute {:.0} ms [{}]; rail {:.0}",
                z2.n, tg, g2.iters, g2.mean_km, r.subzones, r.ms_first, mc, r.rail0
            ));
        }
    }

    rep.line(format!("\npeak heap during the run {:.1} MB (city data before it {:.1} MB)", alloc::mb(alloc::peak()), alloc::mb(heap0)));
    rep.0
}
