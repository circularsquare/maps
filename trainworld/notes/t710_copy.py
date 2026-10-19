"""T-710: a private copy of the sim crate with two parameters added and some readouts.

- `station_ref` (perceived minutes, default 0 = today's code): the station-to-station exp table
  is exp(-theta (r - ref)) and every logsum adds ref back, so a sharper theta keeps its range in f32.
- `crowd_start` (share of crush, default 0 = at the seats, today's): the crowding penalty starts at
  the larger of the seats and this share of crush.
- calib.rs: per period and round, how far the averaged and the raw (one round's own) segment
  loads moved; the AM peak's busiest segments.
- the trip test t710_trips.rs.

    python t710_copy.py <trainworld folder> <test file> <dest folder>
"""
import pathlib
import shutil
import sys


def patch(path, old, new, count=1):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) == count, f"{path.name}: {s.count(old)} matches for {old[:70]!r}"
    path.write_text(s.replace(old, new), encoding="utf-8")


def main():
    sim = pathlib.Path(sys.argv[1]) / "sim"
    test = pathlib.Path(sys.argv[2]).resolve()
    dest = pathlib.Path(sys.argv[3]) / "sim"
    if (dest / "src").exists():
        shutil.rmtree(dest / "src")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sim / "src", dest / "src")
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(sim / f, dest / f)
    d = dest / "src" / "demand"
    k = d / "kernel.rs"
    patch(k, "    pub station_theta: f32,\n", "    pub station_theta: f32,\n    pub station_ref: f32,\n    pub crowd_start: f32,\n")
    patch(k, "            station_theta: 0.2,\n", "            station_theta: 0.2,\n            station_ref: 0.0,\n            crowd_start: 0.0,\n")
    patch(k, '            "station_theta" => self.station_theta = v,\n',
          '            "station_theta" => self.station_theta = v,\n            "station_ref" => self.station_ref = v,\n            "crowd_start" => self.crowd_start = v,\n')
    patch(k, "fast_exp(-p.station_theta * r)", "fast_exp(-p.station_theta * (r - p.station_ref))")
    patch(k, "- ssum.ln() / F8::splat(th);", "- ssum.ln() / F8::splat(th) + F8::splat(p.station_ref);")
    patch(k, "- sn.ln() / F8::splat(th))", "- sn.ln() / F8::splat(th) + F8::splat(p.station_ref))")
    patch(k, "        let crush = tph * net.line_crush[l];\n        let hour",
          "        let crush = tph * net.line_crush[l];\n        let seats = seats.max(p.crowd_start * crush);\n        let hour")
    t = d / "trip.rs"
    patch(t, "fast_exp(-th * r)", "fast_exp(-th * (r - p.station_ref))", 2)
    patch(t, "fast_ln(ssum) / th;", "fast_ln(ssum) / th + p.station_ref;")
    patch(t, "fast_ln(sn) / th }", "fast_ln(sn) / th + p.station_ref }")
    patch(t, "fast_ln(sum) / th;", "fast_ln(sum) / th + p.station_ref;")
    patch(t, "fast_ln(sn) / th);", "fast_ln(sn) / th + p.station_ref);")
    # ---- the relative rule (Anita, 2026-10-19): theta = rel_gamma / T(d), T(d) = rel_t0 + rel_tkm x
    # km, by distance band; each band's exp table offset by its own T (also the f32 range fix)
    patch(k, "    pub crowd_start: f32,\n",
          "    pub crowd_start: f32,\n    pub rel_gamma: f32,\n    pub rel_t0: f32,\n    pub rel_tkm: f32,\n    pub rel_tmax: f32,\n    pub band_lo: f32,\n    pub band_hi: f32,\n")
    patch(k, "            crowd_start: 0.0,\n",
          "            crowd_start: 0.0,\n            rel_gamma: 0.0,\n            rel_t0: 25.0,\n            rel_tkm: 3.0,\n            rel_tmax: 60.0,\n            band_lo: 0.0,\n            band_hi: f32::INFINITY,\n")
    patch(k, '            "crowd_start" => self.crowd_start = v,\n',
          '            "crowd_start" => self.crowd_start = v,\n            "rel_gamma" => self.rel_gamma = v,\n            "rel_t0" => self.rel_t0 = v,\n            "rel_tkm" => self.rel_tkm = v,\n            "rel_tmax" => self.rel_tmax = v,\n')
    patch(k, "impl Params {\n    /// Set a parameter by name",
          # the r10/r20 runs used seven bands, [0, 2, 4, 7, 12, 20, 35, inf]
          "pub const REL_EDGES: [f32; 5] = [0.0, 3.0, 7.0, 12.0, f32::INFINITY];\n"
          "impl Params {\n"
          "    pub fn n_bands(&self) -> usize { if self.rel_gamma > 0.0 { REL_EDGES.len() - 1 } else { 1 } }\n"
          "    pub fn band_of(&self, d_km: f32) -> usize {\n"
          "        if self.rel_gamma <= 0.0 { return 0; }\n"
          "        REL_EDGES[1..].iter().position(|&e| d_km < e).unwrap_or(REL_EDGES.len() - 2)\n"
          "    }\n"
          "    pub fn band_params(&self, b: usize) -> Params {\n"
          "        if self.rel_gamma <= 0.0 { return self.clone(); }\n"
          "        let (lo, hi) = (REL_EDGES[b], REL_EDGES[b + 1]);\n"
          "        let c = if hi.is_finite() { 0.5 * (lo + hi) } else { 1.5 * lo };\n"
          "        let t = self.rel_t0 + self.rel_tkm * c;\n"
          "        Params { station_theta: self.rel_gamma / t.min(self.rel_tmax), station_ref: t, band_lo: lo, band_hi: hi, ..self.clone() }\n"
          "    }\n"
          "}\n\n"
          "impl ModeOut {\n"
          "    pub fn add(&mut self, b: ModeOut) {\n"
          "        let v = |a: &mut Vec<f32>, b: &[f32]| for (x, y) in a.iter_mut().zip(b) { *x += y; };\n"
          "        v(&mut self.od, &b.od);\n"
          "        v(&mut self.rail_by_sub, &b.rail_by_sub);\n"
          "        v(&mut self.rail_to_sub, &b.rail_to_sub);\n"
          "        v(&mut self.walk_lost_by_sub, &b.walk_lost_by_sub);\n"
          "        v(&mut self.walk_lost_to_sub, &b.walk_lost_to_sub);\n"
          "        v(&mut self.other_lost_by_sub, &b.other_lost_by_sub);\n"
          "        v(&mut self.acc_home, &b.acc_home);\n"
          "        v(&mut self.acc_work, &b.acc_work);\n"
          "        for (x, y) in self.rail_band.iter_mut().zip(b.rail_band) { *x += y; }\n"
          "        for (x, y) in self.rail_walksum.iter_mut().zip(b.rail_walksum) { *x += y; }\n"
          "        self.rail += b.rail;\n"
          "        self.trips_access += b.trips_access;\n"
          "        self.rail_walk += b.rail_walk;\n"
          "        self.zone_pairs += b.zone_pairs;\n"
          "        self.pairs += b.pairs;\n"
          "        self.pruned_trips += b.pruned_trips;\n"
          "        self.other_lost += b.other_lost;\n"
          "    }\n"
          "}\n\n"
          "impl Params {\n    /// Set a parameter by name")
    # commutes: only the destination zones of this band (zone-centre distance)
    patch(k, "                let by_ride = rides && long > 0.0;\n",
          "                let dkm = ((zj.gx - z.gx[i]) as f32).hypot((zj.gy - z.gy[i]) as f32) * p.zone_m * 0.001;\n"
          "                if dkm < p.band_lo || dkm >= p.band_hi {\n                    continue;\n                }\n"
          "                let by_ride = rides && long > 0.0;\n")
    a = d / "api.rs"
    patch(a, "            let er = station_exp(sk, p);\n            let mo = mode_choice(&self.z, &self.gr, sz, &er, g.n_st, p, &self.nr);\n            drop(er);\n",
          "            let mut mo: Option<ModeOut> = None;\n"
          "            for b in 0..p.n_bands() {\n"
          "                let pb = p.band_params(b);\n"
          "                let er = station_exp(sk, &pb);\n"
          "                let m = mode_choice(&self.z, &self.gr, sz, &er, g.n_st, &pb, &self.nr);\n"
          "                match mo.as_mut() {\n                    None => mo = Some(m),\n                    Some(x) => x.add(m),\n                }\n"
          "            }\n"
          "            let mo = mo.unwrap();\n")
    # the trip explorer: the band's theta for the pair
    head = "    let sz = api.sz.as_ref().expect(\"a network with subzones\");\n    let z = &api.z;\n    let th = p.station_theta;\n    let s = sk.s;\n    let (i, j) = (sz.zone[u] as usize, sz.zone[v] as usize);\n"
    s_ = t.read_text(encoding="utf-8")
    assert s_.count(head) == 2
    first = s_.index(head)
    s_ = s_[:first] + head + ("    let dkm = ((z.gx[j] - z.gx[i]) as f32).hypot((z.gy[j] - z.gy[i]) as f32) * p.zone_m * 0.001;\n"
                             "    let pb = p.band_params(p.band_of(dkm));\n    let p = &pb;\n    let th = p.station_theta;\n") + s_[first + len(head):]
    t.write_text(s_, encoding="utf-8")
    patch(t, "    let distance = ev.pair.map_or(cells_m, |x| x.0);\n",
          "    let distance = ev.pair.map_or(cells_m, |x| x.0);\n    let pb = p.band_params(p.band_of(distance / 1000.0));\n    let p = &pb;\n    let th = p.station_theta;\n")
    e = d / "everyday.rs"
    patch(e, "            let er = station_exp(sk, p);\n",
          "            let ers: Vec<Vec<f32>> = (0..p.n_bands()).map(|b| station_exp(sk, &p.band_params(b))).collect();\n")
    old_acc = """            let th = p.station_theta;
            // per subzone: the cheapest walk and each walk's weight at it; and for a home with
            // rides (T-389), the cheapest of walks and rides and every entry's weight at that
            let mut min = vec![0.0; sz.n];
            let mut min_r = vec![f32::INFINITY; sz.n];
            let mut acc = vec![0.0; sz.acc_st.len()];
            let mut acc_r = vec![0.0; sz.acc_st.len()];
            let lowest = |r: std::ops::Range<usize>| sz.acc_w[r].iter().copied().fold(f32::INFINITY, f32::min);
            for u in 0..sz.n {
                let r = sz.walk(u);
                min[u] = lowest(r.clone());
                for k in r {
                    acc[k] = fast_exp(-th * (sz.acc_w[k] - min[u]));
                }
                if !sz.ride(u).is_empty() {
                    let all = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
                    min_r[u] = lowest(all.clone());
                    for k in all {
                        acc_r[k] = fast_exp(-th * (sz.acc_w[k] - min_r[u]));
                    }
                }
            }
"""
    new_acc = """            let nb = p.n_bands();
            let bps: Vec<super::kernel::Params> = (0..nb).map(|b| p.band_params(b)).collect();
            let mut min = vec![0.0; sz.n];
            let mut min_r = vec![f32::INFINITY; sz.n];
            let mut accs = vec![vec![0.0f32; sz.acc_st.len()]; nb];
            let mut acc_rs = vec![vec![0.0f32; sz.acc_st.len()]; nb];
            let lowest = |r: std::ops::Range<usize>| sz.acc_w[r].iter().copied().fold(f32::INFINITY, f32::min);
            for u in 0..sz.n {
                let r = sz.walk(u);
                min[u] = lowest(r.clone());
                let all = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
                if !sz.ride(u).is_empty() {
                    min_r[u] = lowest(all.clone());
                }
                for b in 0..nb {
                    let th = bps[b].station_theta;
                    for k in r.clone() {
                        accs[b][k] = fast_exp(-th * (sz.acc_w[k] - min[u]));
                    }
                    if !sz.ride(u).is_empty() {
                        for k in all.clone() {
                            acc_rs[b][k] = fast_exp(-th * (sz.acc_w[k] - min_r[u]));
                        }
                    }
                }
            }
"""
    patch(e, old_acc, new_acc)
    patch(e, "            for (i, pair) in self.pairs.iter().enumerate() {\n",
          "            for (i, pair) in self.pairs.iter().enumerate() {\n"
          "                let bnd = p.band_of(pair.distance / 1000.0);\n"
          "                let (er, acc, acc_r, th, sref) = (&ers[bnd], &accs[bnd], &acc_rs[bnd], bps[bnd].station_theta, bps[bnd].station_ref);\n")
    patch(e, "fast_ln(sum) / th;", "fast_ln(sum) / th + sref;")
    patch(e, "fast_ln(sum_n) / th);", "fast_ln(sum_n) / th + sref);")
    with e.open("a", encoding="utf-8") as f:
        f.write(f'\n#[cfg(test)]\n#[path = "{test.as_posix()}"]\npub(crate) mod t710_trips;\n')
    c = dest / "src" / "bin" / "calib.rs"
    patch(c, "        for _ in 0..rounds {\n",
          "        let mut prev_avg: Vec<f32> = vec![];\n        let mut prev_raw: Vec<f32> = vec![];\n        for rr in 0..=rounds {\n")
    patch(c, "            assert!(api.crowd_with_external(q, &total));\n",
          "            // T-710: how far this round's priced loads moved (the settled test), and the raw loads\n"
          "            let segt = total[..nr].to_vec();\n"
          "            let base: f64 = segt.iter().map(|&x| x as f64).sum::<f64>().max(1e-9);\n"
          "            if !prev_avg.is_empty() {\n"
          "                let moved: f64 = segt.iter().zip(&prev_avg).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / base;\n"
          "                // the round's own loads: avg_r = avg_{r-1} + (raw_r - avg_{r-1}) / (r + 1)\n"
          "                let raw: Vec<f32> = segt.iter().zip(&prev_avg).map(|(a, b)| b + (a - b) * (rr as f32 + 1.0)).collect();\n"
          "                let flip = if prev_raw.is_empty() { -1.0 } else { raw.iter().zip(&prev_raw).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / base };\n"
          "                eprintln!(\"T710 q {q} before round {rr}: avg moved {:.4}, raw moved {:.4}\", moved, flip);\n"
          "                prev_raw = raw;\n"
          "            } else {\n"
          "                prev_raw = segt.clone();\n"
          "            }\n"
          "            prev_avg = segt;\n"
          "            if rr == rounds { break; }\n"
          "            assert!(api.crowd_with_external(q, &total));\n")
    patch(c, "        let s = api.summary(q);\n",
          "        let s = api.summary(q);\n"
          "        // T-710: the busiest segments of the period (every population), riders in the period\n"
          "        {\n"
          "            let mut segq = api.seg(q);\n"
          "            if let Some(e) = &ev { for (a, b) in segq.iter_mut().zip(e.seg(q)) { *a += b; } }\n"
          "            let mut ix: Vec<usize> = (0..segq.len()).collect();\n"
          "            ix.sort_by(|&a, &b| segq[b].partial_cmp(&segq[a]).unwrap());\n"
          "            let top: Vec<String> = ix.iter().take(12).map(|&r| format!(\"{r}:{:.0}\", segq[r])).collect();\n"
          "            eprintln!(\"T710 q {q} round {} max_crush {:.3} over_seats {} busiest {}\", s[0], s[7], s[6], top.join(\" \"));\n"
          "        }\n")
    print(f"copied to {dest}")


if __name__ == "__main__":
    main()
