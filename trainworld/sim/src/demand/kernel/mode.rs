use super::*;

// ---------------------------------------------------------------------------------------------
// 5. Mode choice

/// exp(x) to ~2e-4 relative: 2^int by exponent bits, 2^frac by a degree-5 polynomial. wasm has
/// no exp instruction, and libm's exp was the larger part of the mode choice loop there.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let t = x.clamp(-80.0, 80.0) * std::f32::consts::LOG2_E;
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355_8))));
    f32::from_bits(((i as i32 + 127) << 23) as u32) * p
}

/// ln(x) for normal x > 0, to 3e-6 absolute: exponent bits, then a degree-6 polynomial in
/// t = m - 1 with the mantissa m in [0.71, 1.41) (least-squares fit, notes/T-020.md). Same reason
/// as `fast_exp`: wasm has no log instruction. Branch- and division-free: shifting the bits by
/// those of 1/sqrt(2) folds the mantissa without a data-dependent branch, which in the pair loop
/// mispredicted half the time. In a logsum cost, 3e-6 is 1.5e-5 perceived minutes.
#[inline]
pub fn fast_ln(x: f32) -> f32 {
    const HALF_SQRT2: u32 = 0x3f35_04f3; // 0.70710677
    let b = x.to_bits().wrapping_sub(HALF_SQRT2);
    let e = (b as i32) >> 23;
    let t = f32::from_bits((b & 0x007f_ffff).wrapping_add(HALF_SQRT2)) - 1.0;
    let q = 1.000_004_68 + t * (-0.499_903_616 + t * (0.332_567_778 + t * (-0.253_978_318 + t * (0.220_702_996 + t * -0.143_383_126))));
    e as f32 * std::f32::consts::LN_2 + t * q
}

/// exp(-theta (station-to-station cost - offset)), S x S per band of `Params::band` one after
/// the other, with a zero diagonal: boarding and alighting at the same station is not a rail
/// trip. The access logsum per origin subzone is built from it.
pub fn station_exp(sk: &Skims, p: &Params) -> Vec<f32> {
    let s = sk.s;
    let mut er: Vec<f32> = Vec::with_capacity(p.n_bands() * s * s);
    for b in 0..p.n_bands() {
        let (th, off) = p.band(b);
        let at = er.len();
        er.extend(sk.r.iter().map(|&r| if r.is_finite() { fast_exp(-th * (r - off)) } else { 0.0 }));
        for a in 0..s {
            er[at + a * s + a] = 0.0;
        }
    }
    er
}

/// Distance bands (km, upper edges) for rail trips by how far home is from the boarding station.
pub const DIST_BANDS_KM: [f32; N_BANDS] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, f32::INFINITY];
pub const N_BANDS: usize = 13;
/// T-874: `Params::commute_chunk`'s value in the game (notes/T-874.md).
pub const COMMUTE_CHUNK: f32 = 0.1;
/// T-874: the side of the blocks of destination zones (16 km for 2 km zones) within which an
/// origin zone's small commute pairs are dealt their trips in chunks (`Params::commute_chunk`).
pub(super) const CHUNK_BLOCK: i32 = 8;

/// The band of `DIST_BANDS_KM` a straight-line distance in metres falls in.
pub fn dist_band(d_m: f32) -> usize {
    DIST_BANDS_KM.iter().position(|&e| d_m < e * 1000.0).unwrap_or(N_BANDS - 1)
}

/// T-1300: the step of the riders' fare sensitivity (`FareSense`), perceived minutes.
pub const FARE_D_MIN: f32 = 1.0;

/// T-1300: a population's fare sensitivity: rail trips per perceived minute dearer on every fare
/// (negative), the same with each trip weighed by its straight-line km, and its rail trips weighed so.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct FareSense {
    pub d: f64,
    pub d_km: f64,
    pub km: f64,
}

impl FareSense {
    /// Averaged in as a round's figures are (`w` the new round's weight).
    pub fn toward(&mut self, new: FareSense, w: f64) {
        self.d += (new.d - self.d) * w;
        self.d_km += (new.d_km - self.d_km) * w;
        self.km += (new.km - self.km) * w;
    }
    pub fn scale(&mut self, f: f64) {
        self.d *= f;
        self.d_km *= f;
        self.km *= f;
    }
}

pub struct ModeOut {
    /// Rail trips by (access station, egress station), S x S, in the home -> work direction.
    pub od: Vec<f32>,
    pub rail: f64,
    /// T-1300: how `rail` moves with the fare.
    pub sense: FareSense,
    /// Rail trips by straight-line distance from home (subzone centroid) to the boarding station,
    /// bands `DIST_BANDS_KM`.
    pub rail_band: [f64; N_BANDS],
    /// Trips between subzones offered rail: both ends within the walk's bound of a station, and
    /// at least one pair of the two zones under `Params::pair_walk_max`.
    pub trips_access: f64,
    /// All home->work trips in the period.
    pub trips_total: f64,
    /// Trips walking the whole way with no rail that rail takes (T-558): over the pairs offered
    /// rail, rail's share times walking's share with no rail (`Params::car_choice`). Walking's
    /// trips are those with no rail (`Gravity::no_rail_by_zone`) less these.
    pub rail_walk: f64,
    pub zone_pairs: u64,
    pub pairs: u64,
    pub pruned_trips: f64,
    /// Rail trips by home (origin) subzone and by work (destination) subzone.
    pub rail_by_sub: Vec<f32>,
    pub rail_to_sub: Vec<f32>,
    /// `rail_walk` by home subzone and by work subzone.
    pub walk_lost_by_sub: Vec<f32>,
    pub walk_lost_to_sub: Vec<f32>,
    /// T-2214: the rail trips between two land parts (`Subzones::apart`), which no other way
    /// serves: the trips no way serves are those with no rail (`Gravity::no_rail_by_slot`) less
    /// these. In all, by home subzone and by work subzone (empty with no parts).
    pub rail_unr: f64,
    pub unr_lost_by_sub: Vec<f32>,
    pub unr_lost_to_sub: Vec<f32>,
    /// Rail trips per access entry (`Subzones::acc_st`): at the home end, boarding at that
    /// station from that subzone; at the work end, leaving the train there for that subzone.
    pub acc_home: Vec<f32>,
    pub acc_work: Vec<f32>,
    /// T-2365: `rail`, `rail_walk` and `rail_unr` by trip length, each zone pair's spread as its
    /// trips are (`lengths::LenKernel`).
    pub len: super::super::lengths::LenOut,
}

/// Rail, the car and walking the whole way (T-558; the zone pair's walk) in one logit per
/// (origin subzone, destination subzone) pair; rail's cost is a logit over access and egress
/// stations (SPEC 4.3, 4.4; notes/T-020.md, T-078.md).
///
/// For a pair (u, v) the rail alternatives are every (access station a of u, egress station t of
/// v); each costs access_a + R[a, t] + egress_t. Their logsum separates:
///     S_uv = sum_t A_u[t] E_v[t],  A_u[t] = sum_a exp(-theta (access_a - m_u)) exp(-theta R[a, t]),
///     E_v[t] = exp(-theta (egress_t - m_v)),  rail cost = m_u + m_v + fare - ln(S_uv) / theta.
/// Theta is the pair's distance band's (T-710, `Params::band`), its exp table offset by the
/// band's offset, which the cost adds back; an origin builds its rows for the bands it uses.
/// A_u is one row of S per origin subzone, built once per subzone (access options x S
/// multiply-adds), so the pair loop costs one product per egress station plus one ln and one
/// exp. Rail trips are kept by (u, egress station) as Y_u[t] = sum_v x_uv E_v[t] / S_uv and split
/// back onto access stations once per subzone: od[a, t] += Y_u[t] w_a ER[a, t]. With rides
/// (T-371) an origin has a second row, B_u, over its walks and rides at the cheapest of either's
/// offset, used for the commutes `ride_offered`, with its own Y.
///
/// An origin first lists its pairs within the walk bound, then takes them eight at a time in
/// `lanes::F8` (T-036): every lane runs the scalar expressions in their order and every sum keeps
/// its order, so the results are those of one pair at a time to the bit (`trip::pair_choice`).
/// Anita's walking rule (T-1059, T-2265): station pair (a, t) is a rail option for a trip whose
/// ends are `direct` metres apart on foot when the way to a, the strategy's real minutes between
/// stations (`Skims::walk`, `between`) and the walk from t together take no longer, all in metres
/// at walking's pace (`Params::walk_eq_m`).
#[inline]
pub fn walk_ok(walk_a: f32, between: u16, walk_t: f32, direct: f32) -> bool {
    between != u16::MAX && walk_a + between as f32 + walk_t <= direct
}

/// Subzone `u`'s access entry `k` for the walking rule: a walk entry's mean metres, a drive's real
/// minutes as metres at walking's pace (T-2265).
#[inline]
pub fn walked(sz: &Subzones, p: &Params, u: usize, k: usize) -> f32 {
    if k < sz.acc_mid[u] as usize { sz.acc_d[k] } else { p.walk_eq_m(sz.acc_t[k]) }
}

/// The share of a trip's station-pair weight on pairs `walk_ok` keeps (all of it where the refused
/// are a sliver, `WALK_SLIVER`): over access and egress
/// entries (station, weight, metres walked, `Subzones` entry) and `er(a, t)`. `mode_choice`
/// reaches the same share from its lanes.
pub(crate) fn walk_kept(acc: &[(u32, f32, f32, u32)], eg: &[(u32, f32, f32, u32)], er: impl Fn(usize, usize) -> f32, walk_tab: &[u16], s: usize, direct: f32) -> f32 {
    let (mut tot, mut kept) = (0f32, 0f32);
    for &(a, w, wa, _) in acc {
        for &(t, e, wt, _) in eg {
            let (a, t) = (a as usize, t as usize);
            let c = w * er(a, t) * e;
            if c > 0.0 {
                tot += c;
                if walk_ok(wa, walk_tab[a * s + t], wt, direct) {
                    kept += c;
                }
            }
        }
    }
    // the kept part summed alone, so a trip that keeps a sliver keeps it exactly
    if tot > 0.0 && tot - kept > WALK_SLIVER * tot { (kept / tot).min(1.0) } else { 1.0 }
}

pub fn mode_choice(z: &Zones, gr: &Gravity, sz: &Subzones, er: &[f32], walk_tab: &[u16], walk_max: &[u16], s: usize, p: &Params, nr: &NoRail) -> ModeOut {
    mode_choice_part(z, gr, sz, er, walk_tab, walk_max, s, p, nr, (0, 1))
}

/// `mode_choice` over part `k` of `n` of the origin zones, dealt round-robin (T-1543): every
/// figure is a sum over origins, so the parts' figures add up to the whole's (`trips_total` is
/// the whole's in every part).
pub fn mode_choice_part(z: &Zones, gr: &Gravity, sz: &Subzones, er: &[f32], walk_tab: &[u16], walk_max: &[u16], s: usize, p: &Params, nr: &NoRail, (k, n): (u32, u32)) -> ModeOut {
    let share = p.period_share as f64;
    // T-1059: Anita's walking rule, with the strategies' real minutes between stations (`Skims::walk`)
    let rule = p.walk_rule > 0.0 && walk_tab.len() == s * s && walk_max.len() == s;
    // per access entry its way to the station, and that and the most its station takes on
    // (`Skims::walk_max`)
    let (mut acc_walk, mut acc_from) = (vec![0f32; sz.acc_st.len()], vec![0f32; sz.acc_st.len()]);
    if rule {
        for u in 0..sz.n {
            for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                acc_walk[k] = walked(sz, p, u, k);
                acc_from[k] = acc_walk[k] + walk_max[sz.acc_st[k] as usize] as f32;
            }
        }
    }
    // the refused station pairs of a block's lanes: access station, egress station, weight,
    // access entry, egress step
    let mut lane_bad: Vec<(u32, u32, f32, u32, u8)> = vec![];
    // T-2307: per station t, what each of an origin's access entries takes before alighting there
    // (its way to the station and the real minutes on to t, `walk_ok`'s first sum), its entries
    // side by side so a lane tests every one against an egress entry in one pass; a row is filled
    // when an origin's lane first reaches its station (`till_gen`)
    let till_w = if rule { (0..sz.n).map(|u| sz.acc_off[u + 1] - sz.acc_off[u]).max().unwrap_or(0) as usize } else { 0 };
    assert!(till_w <= 64, "the walking rule's masks hold 64 access entries");
    let mut till = vec![0f32; s * till_w];
    let mut till_at = vec![0u32; if rule { s } else { 0 }];
    let mut till_gen = 0u32;    // each access entry's distance band, for the refused pairs' take-off
    let acc_band: Vec<u8> = if rule { sz.acc_d.iter().map(|&d| dist_band(d) as u8).collect() } else { vec![] };
    // T-710: the station choice's scale and offset by distance band (`er` holds a table per band)
    let nb = p.n_bands();
    let bands: Vec<(f32, f32)> = (0..nb).map(|b| p.band(b)).collect();
    let (ss, na) = (s * s, sz.acc_st.len());
    // per subzone: its part of the zone's row / column factor
    let ou: Vec<f32> = (0..sz.n).map(|u| sz.origin_weight(u, z, gr, share)).collect();
    let dv: Vec<f32> = (0..sz.n)
        .map(|v| {
            let zj = sz.zone[v] as usize;
            if z.jobs[zj] > 0.0 { (gr.col[zj] * sz.jobs[v] as f64 / z.jobs[zj]) as f32 } else { 0.0 }
        })
        .collect();
    let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
    // per band, subzone and access entry: weights exp(-theta (cost - m)), a walk entry's with m
    // the cheapest walk (`acc_m`, the same at both ends; infinite with none), a ride entry's with
    // m the cheapest of walks and rides (`ride_m`)
    let mut acc_m = vec![f32::INFINITY; sz.n];
    let mut ride_m = vec![f32::INFINITY; sz.n];
    // the same as the access bound sees it (`Subzones::bound_min`)
    let mut ride_b = vec![f32::INFINITY; sz.n];
    let mut acc_e = vec![0f32; nb * na];
    let min_of = |r: std::ops::Range<usize>| sz.acc_w[r].iter().cloned().fold(f32::INFINITY, f32::min);
    for v in 0..sz.n {
        let m = min_of(sz.walk(v));
        let mr = m.min(min_of(sz.ride(v)));
        acc_m[v] = m;
        ride_m[v] = mr;
        ride_b[v] = sz.bound_min(p, v);
        for (b, &(th, _)) in bands.iter().enumerate() {
            for k in sz.walk(v) {
                acc_e[b * na + k] = fast_exp(-th * (sz.acc_w[k] - m));
            }
            for k in sz.ride(v) {
                acc_e[b * na + k] = fast_exp(-th * (sz.acc_w[k] - mr));
            }
        }
    }
    // per subzone: minutes a km on its local streets and main roads (by its zone's density)
    let dens = z.density();
    let local: Vec<(f32, f32)> = (0..sz.n).map(|v| p.place_min_per_km(dens[sz.zone[v] as usize])).collect();
    // T-2214: subzones on two land parts have neither car nor walk (`Subzones::apart`)
    let parted = !sz.part.is_empty();
    let mut out = ModeOut {
        od: vec![0.0; s * s],
        rail: 0.0,
        sense: FareSense::default(),
        rail_band: [0.0; N_BANDS],
        trips_access: 0.0,
        trips_total: gr.total * share,
        rail_walk: 0.0,
        zone_pairs: 0,
        pairs: 0,
        pruned_trips: 0.0,
        rail_by_sub: vec![0.0; sz.n],
        rail_to_sub: vec![0.0; sz.n],
        walk_lost_by_sub: vec![0.0; sz.n],
        walk_lost_to_sub: vec![0.0; sz.n],
        rail_unr: 0.0,
        unr_lost_by_sub: if parted { vec![0.0; sz.n] } else { vec![] },
        unr_lost_to_sub: if parted { vec![0.0; sz.n] } else { vec![] },
        acc_home: vec![0.0; sz.acc_st.len()],
        acc_work: vec![0.0; sz.acc_st.len()],
        len: super::super::lengths::LenOut::zero(),
    };
    let ozones: Vec<usize> = (0..z.n).filter(|&i| sz.zo_off[i + 1] > sz.zo_off[i]).collect();
    // Destination zones by their cheapest walk (their first subzone: `build_subzones` sorts
    // them), so an origin visits only the prefix within `pair_walk_max` of its own walk: in
    // 10-minute steps of that walk, in zone order within a step (zone order keeps the decay
    // table lookups near each other; sorting by the walk alone was 20% slower).
    let mut dzones: Vec<usize> = (0..z.n).filter(|&j| sz.zd_off[j + 1] > sz.zd_off[j]).collect();
    let zmin = |j: usize| acc_m[sz.zd[sz.zd_off[j] as usize] as usize];
    dzones.sort_by_key(|&j| ((zmin(j) * 0.1) as u32, j));
    let dstep: Vec<u32> = dzones.iter().map(|&j| (zmin(j) * 0.1) as u32).collect();
    // how many of them origin subzone u visits: past this, the cheapest walk there takes every
    // pair over `pair_walk_max` (u's own cheapest walk, or walk or ride when it has rides)
    let reach = |u: usize| {
        let lim = p.pair_walk_max - if sz.ride(u).is_empty() { acc_m[u] } else { ride_b[u] };
        dstep.partition_point(|&m| m as f32 * 10.0 <= lim)
    };
    // Everything the pair loop reads about a destination, packed in visiting order: per zone,
    // then per subzone, then its egress stations.
    struct DZone {
        key: i32,
        reg: u8,
        zid: u32,
        min: f32,
        max: f32,
        dv: f32,
        q0: u32,
        q1: u32,
        // its lower-left square, centre in half squares and drive supercell, for the zone
        // pair's factors without reading `z`
        gx: i32,
        gy: i32,
        hx: i32,
        hy: i32,
        sup: u16,
    }
    let mut dz: Vec<DZone> = Vec::with_capacity(dzones.len());
    // per destination: [x, y, local streets, main roads], [parking every driver pays, trips
    // factor, walk, a paid space], [the share parking free, -, -, -]
    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());
    // its subzone, and the walkers rail takes there (`walk_lost_to_sub`)
    let mut dd_sub: Vec<u32> = Vec::with_capacity(sz.zd.len());
    // and its land part (T-2214)
    let mut dd_part: Vec<u16> = vec![];
    // and its egress stations, their weights and `Subzones::acc_st` entries, padded to K with
    // station `s` (read as a dummy past the row's end, 0) at weight 0: a pad adds exactly 0 to
    // every sum it enters, so eight destinations go through the same steps
    let (mut eg_st, mut eg_e, mut eg_k, mut eg_n): (Vec<[u32; KE]>, Vec<[f32; KE]>, Vec<[u32; KE]>, Vec<u8>) = (vec![], vec![], vec![], vec![]);
    // and the longest of its egress walks (T-1059)
    let mut eg_far: Vec<f32> = vec![];
    for &j in &dzones {
        let q0 = dd.len() as u32;
        let mut dvz = 0.0f32;
        for &v in &sz.zd[sz.zd_off[j] as usize..sz.zd_off[j + 1] as usize] {
            let v = v as usize;
            let (mut st, mut kk) = ([s as u32; KE], [0u32; KE]);
            for (n, k) in sz.walk(v).enumerate() {
                (st[n], kk[n]) = (sz.acc_st[k], k as u32);
            }
            // its weights per band: destination q's band b at `eg_e[q * nb + b]`
            for b in 0..nb {
                let mut e = [0f32; KE];
                for (n, k) in sz.walk(v).enumerate() {
                    e[n] = acc_e[b * na + k];
                }
                eg_e.push(e);
            }
            eg_st.push(st);
            eg_k.push(kk);
            eg_n.push(sz.walk(v).len() as u8);
            eg_far.push(sz.walk(v).map(|k| sz.acc_d[k]).fold(0.0, f32::max));
            dvz += dv[v];
            let (fixed, paid, free) = park[j];
            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);
            dd_sub.push(v as u32);
            if parted {
                dd_part.push(sz.part[v]);
            }
        }
        let q1 = dd.len() as u32;
        dz.push(DZone { key: gr.key(z, j), reg: z.region[j], zid: j as u32, min: zmin(j), max: dd[q1 as usize - 1][1][2], dv: dvz, q0, q1, gx: z.gx[j], gy: z.gy[j], hx: z.hx(j), hy: z.hy(j), sup: p.routes.as_ref().map_or(0, |r| r.sup_of(j)) });
    }
    // T-874: destination zones by block of `CHUNK_BLOCK` square, in zone order within one: the
    // order an origin zone's small pairs are dealt their chunks in, so an edit re-deals only the
    // blocks whose zones' trips it changed
    let blk = |j: usize| (z.gx[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff) << 16 | (z.gy[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff);
    let mut dorder: Vec<(u32, u32)> = dz.iter().enumerate().map(|(k, zj)| (blk(zj.zid as usize), k as u32)).collect();
    dorder.sort_by_key(|&(b, k)| (b, dz[k as usize].zid));
    // each destination's factor on its trips from the origin zone: 1, or what the chunks dealt
    // it over its trips (0 for none)
    let mut fm = vec![1f32; dz.len()];
    let mut run: Vec<(usize, f64)> = vec![];
    let mut lost_q = vec![0f32; dd.len()];
    let mut unr_q = vec![0f32; if parted { dd.len() } else { 0 }];
    // the decay and the walk's cost by zone offset side by side: one lookup per zone pair
    let fw: Vec<[f32; 2]> = gr.ftab.iter().enumerate().map(|(k, &f)| [f, gr.walk_c(k)]).collect();
    // with a third: the road's length over the straight line between the zones (T-275); and a
    // fourth, 1 where the commute is long enough for the ride (T-371)
    let mut fz = vec![[0f32, 0.0, p.road_detour, 0.0]; dz.len()];
    // and the zone pair's walking with no rail (`NoRail::split`, T-558), and its distance band
    let mut fzw = vec![0f32; dz.len()];
    let mut fzb = vec![0u8; dz.len()];
    let pair_walk = gr.wroutes.is_some() || !gr.wf_off.is_empty();
    let drive = p.routes.as_deref().filter(|r| r.has_drive());
    let ride_min2 = (p.ride_min_km * 1000.0).powi(2);
    // the origin's two rows (walks; walks and rides) one after the other, each followed by
    // `PADS` dummies (0), and their rail trips by egress station the same way; a pair per band,
    // band b's at 2 b h
    let h = s + PADS;
    let mut ab_row = vec![0f32; 2 * nb * h];
    let mut ab_y = vec![0f32; 2 * nb * h];
    // rail trips reaching each destination's egress entries, by its egress step
    let mut acc_q = vec![[[0f32; 4]; 2]; dd.len()];
    // the origin's pairs within the walk bound, in visiting order: destination, and the zone
    // visit (`visit`: [trips at the origin x decay, walking's cost, road factor, origin walk]),
    // its top bit set for the walk-and-ride row
    let mut cand: Vec<[u32; 2]> = Vec::with_capacity(dd.len() + 7);
    let mut visit: Vec<[f32; 4]> = Vec::with_capacity(dz.len());
    let mut visit_w: Vec<f32> = Vec::with_capacity(dz.len());
    let mut visit_b: Vec<u8> = Vec::with_capacity(dz.len());
    // T-2365: each visit's destination zone, and per destination zone the origin zone's rail
    // trips, walkers taken and trips between land parts, spread by length once the zone is done
    let mut visit_k: Vec<u32> = Vec::with_capacity(dz.len());
    let mut len_z = vec![[0f64; 3]; dz.len()];
    let mut len_mark = vec![false; dz.len()];
    let mut len_touched: Vec<u32> = vec![];
    let lk = super::super::lengths::len_kernel(z.zone_m);
    for (oi, &i) in ozones.iter().enumerate() {
        if n > 1 && oi as u32 % n != k {
            continue;
        }
        let base = gr.base(z, i);
        let us = &sz.zo[sz.zo_off[i] as usize..sz.zo_off[i + 1] as usize];
        let ou_zone: f32 = us.iter().map(|&u| ou[u as usize]).sum();
        // the decay (times the region factor) and walk's cost to every destination zone one of
        // this zone's origins reaches, once for all of them
        let ri = z.region[i];
        let nzi = us.iter().map(|&u| reach(u as usize)).max().unwrap_or(0);
        let (gxi, gyi, hxi, hyi) = (z.gx[i], z.gy[i], z.hx(i), z.hy(i));
        for ((o, zb), zj) in fz[..nzi].iter_mut().zip(fzb.iter_mut()).zip(&dz) {
            let k = (base + zj.key) as usize;
            let [f, w] = fw[k];
            // `ride_offered`
            let (dx, dy) = (zj.gx - gxi, zj.gy - gyi);
            let (fx, fy) = ((zj.hx - hxi) as f32 * 0.5, (zj.hy - hyi) as f32 * 0.5);
            let d2 = (fx * fx + fy * fy) * z.zone_m * z.zone_m;
            let long = (d2 >= ride_min2) as u32 as f32;
            *zb = p.band_of_d2(d2) as u8;
            *o = [f, w, o[2], long];
            if zj.reg != ri {
                o[0] *= gr.border;
            }
            if pair_walk && w < NO_WALK {
                o[1] = gr.walk_cost_at(i, zj.zid as usize, k);
            }
            if let Some(r) = drive {
                // `Params::drive_factor`
                o[2] = r.drive_factor_by(i, dx, dy, zj.reg, zj.sup) * p.road_routed_scale;
            }
        }
        for (wz, (o, zj)) in fzw[..nzi].iter_mut().zip(fz.iter().zip(&dz)) {
            *wz = if o[1] < NO_WALK { nr.split(gr, p, i, zj.zid as usize, (base + zj.key) as usize) } else { 0.0 };
        }
        // T-874: per block, the pairs under `commute_chunk` trips are dealt their total in chunks
        // of about that size, as the market's (`intercity::solved_pairs`): systematic sampling by
        // trips from a start fixed by the zone and block, so the block's trips stay exact and a
        // network gives the same pairs every pass. Trips alone decide it, never rail's costs, so
        // crowding rounds and timetable edits deal the same chunks.
        fm[..nzi].fill(1.0);
        if p.commute_chunk > 0.0 {
            let c = p.commute_chunk as f64;
            let mut g = 0;
            while g < dorder.len() {
                let b = dorder[g].0;
                run.clear();
                while g < dorder.len() && dorder[g].0 == b {
                    let k = dorder[g].1 as usize;
                    g += 1;
                    if k < nzi {
                        let t = (ou_zone * dz[k].dv * fz[k][0]) as f64;
                        if t > 0.0 && t < c {
                            run.push((k, t));
                        }
                    }
                }
                let total: f64 = run.iter().map(|e| e.1).sum();
                let draws = (total / c).ceil();
                if draws >= run.len() as f64 {
                    continue;
                }
                let step = total / draws;
                let at = super::super::intercity::unit_hash(i as u32 ^ p.chunk_salt.wrapping_mul(0x27d4eb2f), b);
                // draws before trips x: those at (n + at) x step below x
                let before = |x: f64| (x / step - at).ceil().clamp(0., draws);
                let mut lo = 0.;
                for &(k, t) in &run {
                    let hi = lo + t;
                    fm[k] = ((before(hi) - before(lo)) * step / t) as f32;
                    lo = hi;
                }
            }
        }
        for (ui, &u) in us.iter().enumerate() {
            let u = u as usize;
            let (walk, ride) = (sz.walk(u), sz.ride(u));
            let (m_u, m_r) = (acc_m[u], ride_m[u]);
            let rides = !ride.is_empty();
            // the walk entries' weight at the rides' offset, per band
            let c_u: Vec<f32> = bands.iter().map(|&(th, _)| if walk.is_empty() { 0.0 } else { fast_exp(-th * (m_u - m_r)) }).collect();
            // T-1059: the most this origin's trips walk before their last station (the walk to a
            // station and the most its strategies walk between stations): a destination whose
            // egress walks keep within its distance on top of that keeps every station pair, the
            // common case, and only the rest are taken pair by pair
            let mw = if rule { acc_from[walk.start..ride.end].iter().copied().fold(0.0, f32::max) } else { 0.0 };
            // T-2307: this origin's rows of `till` are filled afresh as its lanes need them
            till_gen += 1;
            let n_all = ride.end - walk.start;
            debug_assert!(n_all <= till_w);
            let (ux8, uy8, lu08, lu18) = (F8::splat(sz.cx[u]), F8::splat(sz.cy[u]), F8::splat(local[u].0), F8::splat(local[u].1));
            let mut rail_sum = 0.0f32;
            let (mut d_sum, mut dk_sum, mut rk_sum) = (0.0f32, 0.0f32, 0.0f32);
            let mut walk_sum = 0.0f32;
            let mut unr_sum = 0.0f32;
            let pu = if parted { sz.part[u] } else { 0 };
            let mut all_sum = 0.0f32;
            let mut n_pairs = 0usize;
            // past this, a destination's walk takes the pair over `pair_walk_max` (by walk, and
            // by walk or ride)
            let wmax = p.pair_walk_max - m_u;
            let wmax_r = p.pair_walk_max - ride_b[u];
            let hp8 = F8::splat(sz.home_park[u]);
            let nz = reach(u);
            // the bands this origin's pairs fall in
            let mut used = 0u32;
            for (zk, ((((zj, &[f, cw, df, long]), &w0), &zb), &m)) in dz[..nz].iter().zip(&fz).zip(&fzw).zip(&fzb).zip(&fm).enumerate() {
                let by_ride = rides && long > 0.0;
                let (m_o, wmax) = if by_ride { (m_r, wmax_r) } else { (m_u, wmax) };
                if zj.min > wmax {
                    continue;
                }
                if p.prune_trips > 0.0 && ou_zone * zj.dv * f < p.prune_trips {
                    out.pruned_trips += (ou[u] * zj.dv * f) as f64;
                    continue;
                }
                if m == 0.0 {
                    continue;
                }
                if ui == 0 {
                    out.zone_pairs += 1;
                }
                let tu = ou[u] * f * m;
                all_sum += tu * zj.dv;
                // cheapest walk first: past the bound, every later one is too
                let nv = if zj.max <= wmax { zj.q1 - zj.q0 } else { dd[zj.q0 as usize..zj.q1 as usize].partition_point(|d| d[1][2] <= wmax) as u32 };
                n_pairs += nv as usize;
                let zv = visit.len() as u32 | (by_ride as u32) << 31;
                visit_k.push(zk as u32);
                visit.push([tu, cw, df, m_o]);
                visit_w.push(w0);
                visit_b.push(zb);
                used |= 1 << zb;
                cand.extend((zj.q0..zj.q0 + nv).map(|q| [q, zv]));
            }
            out.pairs += n_pairs as u64;
            // the origin's rows in each band its pairs use
            for (b, &c_ub) in c_u.iter().enumerate() {
                if used >> b & 1 == 0 {
                    continue;
                }
                let at = 2 * b * h;
                ab_y[at..at + 2 * h].fill(0.0);
                let er_b = &er[b * ss..(b + 1) * ss];
                let (a_row, b_row) = ab_row[at..at + 2 * h].split_at_mut(h);
                let (a_row, b_row) = (&mut a_row[..s], &mut b_row[..s]);
                a_row.fill(0.0);
                for k in walk.clone() {
                    let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                    for (av, &e) in a_row.iter_mut().zip(&er_b[a * s..(a + 1) * s]) {
                        *av += w * e;
                    }
                }
                // the row with the rides too, for the long commutes
                if rides {
                    for (bv, &av) in b_row.iter_mut().zip(a_row.iter()) {
                        *bv = c_ub * av;
                    }
                    for k in ride.clone() {
                        let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                        for (bv, &e) in b_row.iter_mut().zip(&er_b[a * s..(a + 1) * s]) {
                            *bv += w * e;
                        }
                    }
                }
            }
            // The pairs eight at a time, each lane the scalar expressions in their order
            // (`trip::pair_choice` gives the same bits one pair at a time), then each pair's rail
            // trips onto its egress stations, lane by lane in the pairs' order.
            let n_cand = cand.len();
            while cand.len() % 8 != 0 {
                cand.push(cand[0]);
            }
            for (ci, c) in cand.chunks_exact(8).enumerate() {
                let q: [usize; 8] = std::array::from_fn(|l| c[l][0] as usize);
                // T-2214: the lanes whose destination is on another land part than the origin
                let cross: u32 = if parted { (0..8).fold(0, |m, l| m | ((dd_part[q[l]] != pu) as u32) << l) } else { 0 };
                // each lane's band, and its row: the band's pair, walks or walks and rides
                let bl: [usize; 8] = std::array::from_fn(|l| visit_b[(c[l][1] & 0x7fff_ffff) as usize] as usize);
                let rr: [u32; 8] = std::array::from_fn(|l| if c[l][1] >> 31 != 0 { h as u32 } else { 0 });
                let ro: [u32; 8] = std::array::from_fn(|l| (2 * bl[l] * h) as u32 + rr[l]);
                let th8 = F8::new(std::array::from_fn(|l| bands[bl[l]].0));
                let off8 = F8::new(std::array::from_fn(|l| bands[bl[l]].1));
                let kn = q.iter().map(|&q| eg_n[q]).max().unwrap() as usize;
                let mut ix = [[0u32; 8]; KE];
                let mut ev = [F8::splat(0.0); KE];
                let mut rv = [F8::splat(0.0); KE];
                let mut ssum = F8::splat(0.0);
                for k in 0..kn {
                    // a pad goes to its own dummy (lane and step), so pads chain nothing
                    ix[k] = std::array::from_fn(|l| {
                        let t = eg_st[q[l]][k] as usize;
                        (if t == s { s + l * KE + k } else { t }) as u32 + ro[l]
                    });
                    ev[k] = F8::new(std::array::from_fn(|l| eg_e[q[l] * nb + bl[l]][k]));
                    rv[k] = F8::gather(&ab_row, &ix[k]);
                    ssum = ssum + rv[k] * ev[k];
                }
                let [x8, y8, l08, l18] = F8::transpose(std::array::from_fn(|l| dd[q[l]][0]));
                let (dx, dy) = (ux8 - x8, uy8 - y8);
                let d = (dx * dx + dy * dy).sqrt();
                // T-1059: a lane some of whose station pairs take longer than walking there keeps
                // only the other pairs' part of the logsum. Its station pairs are first passed by
                // the most their stations take on (`Skims::walk_max`) and by walking's reach; the
                // rest are tested against the origin's real minutes to each egress station
                // (`till`), and only the refused ones read the tables: kept to take off its loads
                // below, or where most are refused (`whole`), the kept ones to load alone.
                let (mut slow, mut whole) = (0u32, 0u32);
                let mut bad_at = [(0u32, 0u32); 8];
                // (no walk to compare with between two land parts, T-2214)
                let near = if rule {
                    d.lt_mask((F8::splat(mw) + F8::new(std::array::from_fn(|l| eg_far[q[l]]))).min(F8::splat(WALK_REACH_KM * 1000.0 + 1.0))) & ((1u32 << (n_cand - 8 * ci).min(8)) - 1) & !cross
                } else {
                    0
                };
                if near != 0 {
                    lane_bad.clear();
                    let (d_lane, s_lane) = (d.to_array(), ssum.to_array());
                    let mut keep = [1f32; 8];
                    for l in (0..8).filter(|&l| near >> l & 1 != 0) {
                        let qd = q[l];
                        let direct = d_lane[l] * gr.walk_factor(i, sz.zone[dd_sub[qd] as usize] as usize);
                        if !p.walk_rule_at(direct) {
                            continue;
                        }
                        // per egress entry, a bit per access entry whose station pair into it
                        // takes longer than walking (`walk_ok`; no route takes 65,535 m, past
                        // every walk the rule compares with)
                        let ne = eg_n[qd] as usize;
                        let nk = if rr[l] != 0 { n_all } else { walk.len() };
                        let mut slow_k = [0u64; KE];
                        let mut any = 0u64;
                        for kk in 0..ne {
                            let (t, wt) = (eg_st[qd][kk] as usize, sz.acc_d[eg_k[qd][kk] as usize]);
                            let row = &mut till[t * till_w..t * till_w + n_all];
                            if till_at[t] != till_gen {
                                till_at[t] = till_gen;
                                for (c, k) in row.iter_mut().zip(walk.start..ride.end) {
                                    *c = acc_walk[k] + walk_tab[sz.acc_st[k] as usize * s + t] as f32;
                                }
                            }
                            let m = row[..nk].iter().enumerate().fold(0u64, |m, (j, &c)| m | (!(c + wt <= direct) as u64) << j);
                            slow_k[kk] = m;
                            any |= m;
                        }
                        if any == 0 {
                            continue;
                        }
                        let (b, c_b) = (bl[l], if rr[l] != 0 { c_u[bl[l]] } else { 1.0 });
                        let er_b = &er[b * ss..(b + 1) * ss];
                        let lo = lane_bad.len() as u32;
                        let mut bad = 0f32;
                        for (j, k) in (walk.start..walk.start + nk).enumerate() {
                            if any >> j & 1 == 0 {
                                continue;
                            }
                            let a = sz.acc_st[k] as usize;
                            for kk in 0..ne {
                                if slow_k[kk] >> j & 1 == 0 {
                                    continue;
                                }
                                let t = eg_st[qd][kk] as usize;
                                let w = acc_e[b * na + k] * if k < walk.end { c_b } else { 1.0 };
                                let cw = w * er_b[a * s + t] * eg_e[qd * nb + b][kk];
                                if cw > 0.0 {
                                    bad += cw;
                                    lane_bad.push((a as u32, t as u32, cw, k as u32, kk as u8));
                                }
                            }
                        }
                        // T-2307: a sliver refused is let be (`WALK_SLIVER`)
                        if bad <= WALK_SLIVER * s_lane[l] {
                            lane_bad.truncate(lo as usize);
                            continue;
                        }
                        if bad > 0.5 * s_lane[l] {
                            // most of the lane refused: taking that off a spread at the kept rate
                            // would cancel too far for f32, so its kept pairs are loaded one by one
                            lane_bad.truncate(lo as usize);
                            let mut kept = 0f32;
                            for (j, k) in (walk.start..walk.start + nk).enumerate() {
                                let a = sz.acc_st[k] as usize;
                                let w = acc_e[b * na + k] * if k < walk.end { c_b } else { 1.0 };
                                for kk in 0..ne {
                                    let t = eg_st[qd][kk] as usize;
                                    let cw = w * er_b[a * s + t] * eg_e[qd * nb + b][kk];
                                    if cw > 0.0 && slow_k[kk] >> j & 1 == 0 {
                                        kept += cw;
                                        lane_bad.push((a as u32, t as u32, cw, k as u32, kk as u8));
                                    }
                                }
                            }
                            keep[l] = kept / s_lane[l];
                            whole |= 1 << l;
                        } else {
                            keep[l] = 1.0 - bad / s_lane[l];
                        }
                        bad_at[l] = (lo, lane_bad.len() as u32);
                        slow |= 1 << l;
                    }
                    if slow != 0 {
                        ssum = ssum * F8::new(keep);
                    }
                }
                // no rail route (or only the same station at both ends) where under 1e-30
                let ok = ((1u32 << (n_cand - 8 * ci).min(8)) - 1) & !ssum.lt_mask(F8::splat(1e-30));
                if ok == 0 {
                    continue;
                }
                let [park8, dv8, m8, paid8] = F8::transpose(std::array::from_fn(|l| dd[q[l]][1]));
                let [tu8, cw8, mut df8, mo8] = F8::transpose(std::array::from_fn(|l| visit[(c[l][1] & 0x7fff_ffff) as usize]));
                // T-2214: between two land parts, rail against a car on the bridge that is not
                // there (straight x `road_detour`)
                let pick = |m: u32, a: F8, b: F8| {
                    let (a, b) = (a.to_array(), b.to_array());
                    F8::new(std::array::from_fn(|l| if m >> l & 1 != 0 { b[l] } else { a[l] }))
                };
                if cross != 0 {
                    df8 = pick(cross, df8, F8::splat(p.road_detour));
                }
                // `Params::drive_min_f`
                let rd = d * F8::splat(0.001) * df8;
                let e = (F8::splat(0.5) * rd).min(F8::splat(p.road_local_km));
                let main = rd - F8::splat(2.0) * e;
                let k = (F8::splat(0.5) * main).min(F8::splat(p.road_end_km));
                let drive = e * (lu08 + l08) + k * (lu18 + l18) + (main - F8::splat(2.0) * k) * F8::splat(60.0) / F8::splat(p.road_open_kmh);
                // `Params::car_drive`, parking at the far end and at home (T-854)
                let road = drive * F8::splat(p.w_drive) + rd * F8::splat(p.car_own_km_min()) + park8 + hp8;
                let rail = mo8 + m8 + F8::splat(p.fare_min) - ssum.ln() / th8 + off8;
                // `Params::mode_beta`
                let bt = F8::splat(p.beta) * (F8::splat(p.damp_min) / road).min(F8::splat(1.0)).sqrt();
                // T-558: and walking the whole way, against the same driver
                let uw = bt * (road - cw8) + F8::splat(p.asc_walk());
                // T-380: rail's share among drivers parking free, and among those paying for a
                // space (`Params::car_choice`; where no lane pays, the second share equals the
                // first): walking's share of the three-way logit, then rail of the rest by
                // `Params::ratio_share` (T-918)
                let (zero, one) = (F8::splat(0.0), F8::splat(1.0));
                let (hi, kf) = (F8::splat(p.ratio_fade_hi), F8::splat(1.0 / (p.ratio_fade_hi - p.ratio_fade_lo)));
                let paying = paid8.lt_mask(F8::splat(1e-30)) != 0xff;
                let free8 = if paying { F8::transpose(std::array::from_fn(|l| dd[q[l]][2]))[0] } else { zero };
                let share = |rail: F8| {
                    let util = bt * (road - rail) + F8::splat(p.asc_rail);
                    let rail_c = rail.max(F8::splat(0.01));
                    let ratio = |car: F8| {
                        let qr = rail_c / car;
                        let s = one / (one + qr * qr * qr);
                        let fr = ((hi - qr) * kf).max(zero).min(one);
                        let fc = ((hi - one / qr) * kf).max(zero).min(one);
                        one - (one - s * fr) * fc
                    };
                    let e = (uw - util).exp();
                    let w0 = e / (one + util.neg().exp() + e);
                    let s0 = (one - w0) * ratio(road);
                    if !paying {
                        s0
                    } else {
                        let w1 = e / (one + (util + bt * paid8).neg().exp() + e);
                        let s1 = (one - w1) * ratio(road + paid8);
                        s0 + (one - free8) * (s1 - s0)
                    }
                };
                // T-2214: between two land parts rail is the only way, held by the ratio curve's
                // fade alone (`Params::bound_share`)
                let bound = |rail: F8| {
                    let rail_c = rail.max(F8::splat(0.01));
                    let f0 = ((hi - rail_c / road) * kf).max(zero).min(one);
                    if !paying {
                        f0
                    } else {
                        let f1 = ((hi - rail_c / (road + paid8)) * kf).max(zero).min(one);
                        f0 + (one - free8) * (f1 - f0)
                    }
                };
                let chosen = |rail: F8| {
                    let s = share(rail);
                    if cross != 0 { pick(cross, s, bound(rail)) } else { s }
                };
                let x = tu8 * dv8 * chosen(rail);
                // T-1300: and at a fare `FARE_D_MIN` dearer, for the riders' fare sensitivity
                let rd8 = rail + F8::splat(FARE_D_MIN);
                let xd = (tu8 * dv8 * chosen(rd8) - x).to_array();
                let dk = (d * F8::splat(0.001)).to_array();
                let g = x / ssum;
                // trips by egress station y, and y x the row: what reaches each egress entry
                let mut ya = [[0f32; 8]; KE];
                let mut yw = [F8::splat(0.0); 8];
                for k in 0..kn {
                    let y = g * ev[k];
                    yw[k] = y * rv[k];
                    ya[k] = y.to_array();
                }
                // y x the row by lane, egress steps 0-3 and 4-7 (zeros past each list's end)
                let yw_lo = F8::by_lane([yw[0], yw[1], yw[2], yw[3]]);
                let yw_hi = if kn > 4 { F8::by_lane([yw[4], yw[5], yw[6], yw[7]]) } else { yw_lo };
                let xa = x.to_array();
                let ssum_a = ssum.to_array();
                // and the walkers it takes: its trips times the zone pair's walking with no rail
                // (`rail_walk`), as the zones' sums have it
                let mut wl = (x * F8::new(std::array::from_fn(|l| visit_w[(c[l][1] & 0x7fff_ffff) as usize]))).to_array();
                for l in 0..8 {
                    if ok >> l & 1 == 0 {
                        continue;
                    }
                    rail_sum += xa[l];
                    d_sum += xd[l];
                    dk_sum += xd[l] * dk[l];
                    rk_sum += xa[l] * dk[l];
                    let lz = visit_k[(c[l][1] & 0x7fff_ffff) as usize] as usize;
                    if !len_mark[lz] {
                        len_mark[lz] = true;
                        len_touched.push(lz as u32);
                    }
                    if cross >> l & 1 != 0 {
                        // it takes trips no way served, never walkers
                        wl[l] = 0.0;
                        unr_sum += xa[l];
                        unr_q[q[l]] += xa[l];
                        len_z[lz][2] += xa[l] as f64;
                    }
                    walk_sum += wl[l];
                    lost_q[q[l]] += wl[l];
                    len_z[lz][0] += xa[l] as f64;
                    len_z[lz][1] += wl[l] as f64;
                    if whole >> l & 1 == 0 {
                        // pads add 0 to their own dummies: the block's one count predicts well
                        for k in 0..kn {
                            ab_y[ix[k][l] as usize] += ya[k][l];
                        }
                        let a = &mut acc_q[q[l]];
                        a[0] = (F4::new(a[0]) + yw_lo[l]).to_array();
                        if kn > 4 {
                            a[1] = (F4::new(a[1]) + yw_hi[l]).to_array();
                        }
                    }
                    if slow >> l & 1 != 0 {
                        // T-1059: the lane's loads went onto every station pair at the kept pairs'
                        // rate and the refused ones come off again (the station matrix is cleared
                        // of what rounding leaves below 0 at the end), or its kept pairs go on alone
                        let gl = xa[l] / ssum_a[l];
                        let sign = if whole >> l & 1 != 0 { 1.0 } else { -1.0 };
                        let (lo, hi) = bad_at[l];
                        for &(a, t, c, k, kk) in &lane_bad[lo as usize..hi as usize] {
                            let f = sign * gl * c;
                            out.od[a as usize * s + t as usize] += f;
                            acc_q[q[l]][kk as usize / 4][kk as usize % 4] += f;
                            out.acc_home[k as usize] += f;
                            out.rail_band[acc_band[k as usize] as usize] += f as f64;
                        }
                    }
                }
            }
            cand.clear();
            visit.clear();
            visit_k.clear();
            visit_w.clear();
            visit_b.clear();
            out.trips_access += all_sum as f64;
            if rail_sum == 0.0 {
                continue;
            }
            out.rail_walk += walk_sum as f64;
            out.walk_lost_by_sub[u] = walk_sum;
            if parted {
                out.rail_unr += unr_sum as f64;
                out.unr_lost_by_sub[u] = unr_sum;
            }
            out.rail += rail_sum as f64;
            out.sense.d += d_sum as f64 / FARE_D_MIN as f64;
            out.sense.d_km += dk_sum as f64 / FARE_D_MIN as f64;
            out.sense.km += rk_sum as f64;
            out.rail_by_sub[u] = rail_sum;
            // split this subzone's rail trips back onto its access stations, band by band: a walk
            // entry carries its share of both rows' trips, a ride entry of the long commutes' only
            for (b, &c_ub) in c_u.iter().enumerate() {
                if used >> b & 1 == 0 {
                    continue;
                }
                let at = 2 * b * h;
                let (y_row, yb_row) = (&ab_y[at..at + s], &ab_y[at + h..at + h + s]);
                for k in walk.start..ride.end {
                    let (a, w) = (sz.acc_st[k] as usize, acc_e[b * na + k]);
                    let row = &mut out.od[a * s..(a + 1) * s];
                    let mut tot = 0.0f32;
                    let er_a = &er[b * ss + a * s..b * ss + (a + 1) * s];
                    if k < walk.end {
                        if rides {
                            for (((o, &y), &yb), &e) in row.iter_mut().zip(y_row).zip(yb_row).zip(er_a) {
                                let f = (y + c_ub * yb) * w * e;
                                *o += f;
                                tot += f;
                            }
                        } else {
                            for ((o, &y), &e) in row.iter_mut().zip(y_row).zip(er_a) {
                                let f = y * w * e;
                                *o += f;
                                tot += f;
                            }
                        }
                    } else {
                        for ((o, &yb), &e) in row.iter_mut().zip(yb_row).zip(er_a) {
                            let f = yb * w * e;
                            *o += f;
                            tot += f;
                        }
                    }
                    out.acc_home[k] += tot;
                    out.rail_band[dist_band(sz.acc_d[k])] += tot as f64;
                }
            }
        }
        // T-2365: the zone's figures by length, each destination zone's spread as its trips are
        for &k in &len_touched {
            let (k, j) = (k as usize, dz[k as usize].zid as usize);
            let (hdx, hdy, sj) = (dz[k].hx - hxi, dz[k].hy - hyi, z.side(j));
            let [r, w, u] = std::mem::take(&mut len_z[k]);
            len_mark[k] = false;
            lk.add(hdx, hdy, z.side(i), sj, r, &mut out.len.rail);
            if w != 0.0 {
                lk.add(hdx, hdy, z.side(i), sj, w, &mut out.len.walk);
            }
            if u != 0.0 {
                lk.add(hdx, hdy, z.side(i), sj, u, &mut out.len.unr);
            }
        }
        len_touched.clear();
    }
    for (q, a) in acc_q.iter().enumerate() {
        for k in 0..eg_n[q] as usize {
            out.acc_work[eg_k[q][k] as usize] = a[k / 4][k % 4];
        }
    }
    // rail trips by work subzone: what its egress stations took
    for v in 0..sz.n {
        out.rail_to_sub[v] = out.acc_work[sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize].iter().sum();
    }
    for (q, &v) in dd_sub.iter().enumerate() {
        out.walk_lost_to_sub[v as usize] = lost_q[q];
        if parted {
            out.unr_lost_to_sub[v as usize] = unr_q[q];
        }
    }
    if rule {
        // T-1059: a refused station pair taken off its lane's loads can be left a rounding below 0
        for x in out.od.iter_mut().chain(out.acc_home.iter_mut()).chain(out.acc_work.iter_mut()) {
            *x = x.max(0.0);
        }
    }
    out
}
