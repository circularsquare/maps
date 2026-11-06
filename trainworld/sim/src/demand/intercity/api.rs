use super::*;

impl IntercityApi {
    /// `place_trips` on this market, with no second parse (`bin/market_pack.rs`).
    pub fn place_trips(&self, pairs: &[(usize, usize)], min_km: f64) -> Vec<(f64, f64)> {
        self.m.place_trips(pairs, min_km)
    }
    /// Number of places in selection and station flow headers.
    pub fn market_count(&self) -> usize {
        self.m.places.len()
    }
    /// T-377: the 6 km zones of the whole world (every place's, open or not): the length of the
    /// zone blocks of every answer.
    pub fn zone_count(&self) -> usize {
        self.m.n_gid
    }
    /// The places this market holds the packs of (ascending).
    pub fn open_places(&self) -> Vec<u32> {
        (0..self.m.places.len()).filter(|&p| self.m.places[p].open).map(|p| p as u32).collect()
    }
    /// The zone pairs the network's stations serve, and the place pairs with trips.
    pub fn pair_counts(&self) -> Vec<f64> {
        vec![self.pairs.len() as f64, self.m.cw.place_pairs as f64]
    }
    /// The network with stations in the pack's plane (an old single pack).
    pub fn set_network(&mut self, st_xy: &[f32], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        self.set_network_in(st_xy, line_n, stops, times, tph, cars)
    }
    /// T-276: per corridor, the commutes between zones in period `q`: legs and rail legs.
    pub fn commute_corridors(&self, q: usize) -> Vec<f64> {
        let legs = 0.5 * (TO_WORK[q] + TO_HOME[q]) as f64;
        let m = &self.m;
        self.per(q).filter(|_| m.cw.commuters > 0.).map_or(vec![], |s| {
            let mut out = vec![0f64; 2 * m.n_corridors()];
            for &(c, n) in &m.cw.commute {
                out[2 * c as usize] = n * legs;
            }
            for (k, c) in m.open_corridors() {
                out[2 * c + 1] = s.c_corridors.get(k);
            }
            out
        })
    }
    /// As `DemandApi::transfers` (T-176): this market's transfers over the periods held.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    /// Where the longer trips of a selection of local cells go and come from (T-118).
    ///
    /// The selection is points (`xy`: x, y per 600 m square, metres in that place's pack frame)
    /// of one `market` (place) with the `share` (0..1) of that square's people, or jobs, that are
    /// selected. A site's share of its zone's departures is its people over the zone's, of its
    /// arrivals its attraction over the zone's.
    ///
    /// Returns, over the periods solved: the matched share of the points, then 8 numbers per
    /// place at the far end (trips starting in the selection that end there: all modes, by rail,
    /// all-mode trips times far x and y; then the same for trips ending in the selection), then
    /// per global 6 km zone (`zone_count`) all and rail of starts and of ends, every zone's x, y
    /// and place, then the commutes' tail (4 per place, 4 per zone), then (T-378) the trips with
    /// places that have no zone, starting and ending. Places with no pack open here are counted by
    /// place only.
    pub fn selection_flows(&self, market: u32, xy: &[f32], share: &[f32]) -> Vec<f64> {
        self.selection_flows_day(market, xy, share, WEEKDAY as u32)
    }
    /// A station's longer-trip rail riders over the periods solved (T-118): per place, riders
    /// starting here bound for it, riders ending here from it, and those counts times the far
    /// zone's x and y (starts x, starts y, ends x, ends y); then per global 6 km zone the riders
    /// starting, then ending, then every zone's x, y and place. Empty with no such station.
    pub fn station_flows(&self, station: u32) -> Vec<f64> {
        self.station_flows_day(station, WEEKDAY as u32)
    }
}

/// T-373: the 2 km zones of a group's sites (`city`, whose `jobs` are attractions) with their
/// homes and jobs as the local layer's zones have them, and each zone's centre.
pub(super) fn ride_zones(city: &City) -> (Zones, Vec<(usize, f32, f32)>) {
    let jobs: Vec<f32> = city.pop.iter().zip(&city.jobs).map(|(&p, &a)| ((a - p) / 2.).max(0.)).collect();
    let sites = City { name: String::new(), h3: vec![], x: city.x.clone(), y: city.y.clone(), pop: city.pop.clone(), jobs, region: vec![], gravity: None };
    if sites.is_empty() {
        return (build_zones(&sites, 2000.), vec![]);
    }
    let mut z = build_zones(&sites, 2000.);
    // a site is a 600 m square, not an H3 cell: its land in cells for `Zones::density`
    let per_site = (SITE_M / 1000.).powi(2) / CELL_KM2;
    let mut c = vec![(0f64, 0f64, 0f64); z.n];
    for s in 0..sites.len() {
        let zi = z.cell_zone[s] as usize;
        let w = (sites.pop[s] + sites.jobs[s]) as f64 + 1e-9;
        c[zi].0 += w;
        c[zi].1 += w * sites.x[s] as f64;
        c[zi].2 += w * sites.y[s] as f64;
    }
    let mut places = Vec::with_capacity(z.n);
    for (zi, &(w, x, y)) in c.iter().enumerate() {
        z.ncell[zi] = ((z.ncell[zi] as f32 * per_site).round() as u32).max(1);
        places.push((zi, (x / w) as f32, (y / w) as f32));
    }
    (z, places)
}
/// T-373: a site's access weights at logit scale `beta`, by station, added into `out` times
/// `share`: its walks as the mean of their exp(-beta x cost), nested with its rides at the
/// station choice's scale `theta`.
pub(super) fn add_access(out: &mut BTreeMap<usize, f32>, walks: &[(usize, f32)], rides: &[Ride], beta: f32, theta: f32, share: f32) {
    let (beta, theta) = (beta as f64, theta as f64);
    let n = walks.len() as f64;
    let mean: f64 = walks.iter().map(|w| (-beta * w.1 as f64).exp() / n).sum();
    if rides.is_empty() {
        for &(st, c) in walks {
            *out.entry(st).or_default() += share * ((-beta * c as f64).exp() / n) as f32;
        }
        return;
    }
    let cw = if walks.is_empty() { f64::INFINITY } else { -mean.ln() / beta };
    let c0 = rides.iter().map(|r| r.1 as f64).fold(cw, f64::min);
    let aw = if walks.is_empty() { 0. } else { (-theta * (cw - c0)).exp() };
    let sum = aw + rides.iter().map(|r| (-theta * (r.1 as f64 - c0)).exp()).sum::<f64>();
    let total = (-beta * (c0 - sum.ln() / theta)).exp() / sum;
    for &(st, c) in walks {
        *out.entry(st).or_default() += share * (total * aw * (-beta * c as f64).exp() / n / mean) as f32;
    }
    for r in rides {
        *out.entry(r.0 as usize).or_default() += share * (total * (-theta * (r.1 as f64 - c0)).exp()) as f32;
    }
}
/// An access list's stations: its `ACCESS_TOP` best, then (T-1506) up to `ACCESS_MAX` in all, the
/// best station of each track component none of those is on (`on_train`: a station some train line
/// stops at). Without it a city of many metro lines filled the list with metro stations and an
/// inter-city station not linked to the metro was in no zone's list.
pub(crate) const ACCESS_TOP: usize = 8;
pub(crate) const ACCESS_MAX: usize = 10;
pub(super) fn access_list(weights: BTreeMap<usize, f32>, positive: bool, comp: &[u32], on_train: &[bool]) -> Vec<(usize, f32)> {
    let mut v: Vec<_> = weights.into_iter().filter(|e| !positive || e.1 > 0.).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    if v.len() <= ACCESS_TOP {
        return v;
    }
    let rest = v.split_off(ACCESS_TOP);
    let mut comps: Vec<u32> = v.iter().map(|e| comp[e.0]).collect();
    for e in rest {
        if v.len() >= ACCESS_MAX {
            break;
        }
        if on_train[e.0] && !comps.contains(&comp[e.0]) {
            comps.push(comp[e.0]);
            v.push(e);
        }
    }
    v
}

/// T-409: the pack's own data is shared (`Rc`) by every market `beside` it; the rest is one
/// network's.
#[wasm_bindgen]
pub struct IntercityApi {
    pub(super) m: Rc<Market>,
    /// The zone pairs between zones some station serves (T-174), two by two: (lower zone, higher
    /// zone), then the same reversed, in the order of (lower, higher) (`pair_at` searches it).
    pub(super) pairs: Vec<Pair>,
    /// The zones `pairs` was built for (ascending), and each zone's place in it (`u32::MAX` if not).
    pub(super) served: Vec<usize>,
    pub(super) served_at: Vec<u32>,
    pub(super) base: Params,
    pub(super) svc: Option<Service>,
    pub(super) access: Vec<Vec<(usize, f32)>>,
    pub(super) access_work: Vec<Vec<(usize, f32)>>,
    pub(super) access_c: Vec<Vec<(usize, f32)>>,
    pub(super) access_cw: Vec<Vec<(usize, f32)>>,
    pub(super) access_c_walk: Vec<Vec<(usize, f32)>>,
    pub(super) cpairs: Vec<CPair>,
    /// T-1175: the activity end's walks of the everyday pairs (attraction-weighted), and the
    /// everyday pairs some station serves
    pub(super) access_ew: Vec<Vec<(usize, f32)>>,
    pub(super) epairs: Vec<EPairOn>,
    pub(super) want: Vec<bool>,
    pub(super) per: Vec<Option<Period>>,
    pub(super) fare_km: Vec<f32>,
    /// T-1626: `DemandApi::set_fare_mul`'s multipliers
    pub(super) fare_mul: Vec<f32>,
    pub(super) fare_per_km: f32,
    /// `DemandApi::set_fare_curve`'s knots (T-883)
    pub(super) fare_curve: Vec<f32>,
    /// T-1286: per zone, its country's fare factor less 1 (`set_fare_factors`); empty while every
    /// factor is 1.
    pub(super) zone_fx: Vec<f32>,
    pub(super) complex_in: Vec<u32>,
    /// `DemandApi::set_upgrades` (T-1281).
    pub(super) ups_in: Vec<u8>,
    /// The passages of the next `set_network` (`set_passages`, T-1281).
    pub(super) passages_in: Vec<u32>,
    /// `DemandApi::set_span` (T-842).
    pub(super) span_in: Vec<f32>,
    /// set_bus (T-1464)
    pub(super) bus_in: (Vec<u32>, Vec<f32>),
    /// T-409: per served zone, the track components its stations lie in.
    pub(super) served_comp: Vec<Vec<u32>>,
    pub(super) st_pat: StPattern,
    /// T-780: trips a day each way a sampled zone pair carries (`solved_pairs`); 0 solves every pair.
    pub(super) chunk: f64,
    /// T-2641: with `chunk`, the chunk grows with distance and far place pairs are dealt over both
    /// places' zones (`solved_pairs`); false: T-780's chunk at every distance.
    pub(super) far_chunks: bool,
    /// T-1735: per zone, whether it is in the window (`set_window`); empty: no window, every pair
    pub(super) win: Vec<bool>,
    /// T-1735: the window the next network takes: stations (as `set_network` numbers them) whose
    /// zones are in it, and zones in it besides
    pub(super) win_in: Option<(Vec<u8>, Vec<u32>, f32, f32, u32)>,
}

/// T-761: the entries of the per-station far-zone table (`Period::st`) a pass can make non-zero,
/// row by station (CSR): row s's columns ascending in `col[off[s]..off[s + 1]]`, a column being a
/// served zone's place in `served` for trips starting at s, or that plus the served zones' count
/// for trips ending there. A station's riders go to and come from the zones its own zones trade
/// with (about a tenth of all on Japan's whole network), so the table holds those alone.
#[derive(Default)]
pub(super) struct StPattern {
    pub(super) off: Vec<u32>,
    pub(super) col: Cols,
}

/// Columns, as u16 while every one fits (Japan's whole network: 18,228 columns).
pub(super) enum Cols {
    Short(Vec<u16>),
    Long(Vec<u32>),
}
impl Default for Cols {
    fn default() -> Self {
        Cols::Short(vec![])
    }
}
impl Cols {
    pub(super) fn len(&self) -> usize {
        match self {
            Cols::Short(v) => v.len(),
            Cols::Long(v) => v.len(),
        }
    }
    pub(super) fn get(&self, i: usize) -> usize {
        match self {
            Cols::Short(v) => v[i] as usize,
            Cols::Long(v) => v[i] as usize,
        }
    }
}

impl StPattern {
    /// Where entry (station `s`, column `c`) sits, which must be one of the pattern's, searched
    /// from `cur` (a place in the row) outwards and left there: `demand` looks a row up in runs of
    /// one column, or of columns rising, so most lookups take a step or two.
    #[inline(always)]
    pub(super) fn at(&self, s: usize, c: usize, cur: &mut u32) -> usize {
        let (lo, hi) = (self.off[s] as usize, self.off[s + 1] as usize);
        lo + match &self.col {
            Cols::Short(v) => find_from(&v[lo..hi], c as u16, cur),
            Cols::Long(v) => find_from(&v[lo..hi], c as u32, cur),
        }
    }
    pub(super) fn row(&self, s: usize) -> std::ops::Range<usize> {
        self.off[s] as usize..self.off[s + 1] as usize
    }
}

/// `StPattern::at` in one row.
#[inline(always)]
pub(super) fn find_from<T: Copy + Ord>(row: &[T], c: T, cur: &mut u32) -> usize {
    {
        let i = (*cur as usize).min(row.len() - 1);
        let k = if row[i] == c {
            i
        } else if row[i] < c {
            let (mut a, mut step, mut b) = (i, 1, i + 1);
            while b < row.len() && row[b] < c {
                a = b;
                step *= 2;
                b = a + step;
            }
            a + 1 + row[a + 1..(b + 1).min(row.len())].partition_point(|&x| x < c)
        } else {
            let (mut b, mut step) = (i, 1);
            let a = loop {
                if b < step {
                    break 0;
                }
                let p = b - step;
                if row[p] <= c {
                    break p;
                }
                b = p;
                step *= 2;
            };
            a + row[a..b].partition_point(|&x| x < c)
        };
        assert!(row.get(k) == Some(&c), "a far-zone entry outside the pattern");
        *cur = k as u32;
        k
    }
}

/// Which pairs of the `served` zones (ascending) are solved, a bit each for (i, k > i) in that
/// order, and their trips a day each way, counted before `set_served` makes their table so it is
/// made once at its size (T-761: grown by doubling it peaked at twice that, 500 MB on Japan).
/// A pair needs a car or a plane to compete with (T-377, T-378: else its trips stay fixed), track
/// joining its zones, and `MIN_TRIPS`.
///
/// With `chunk` (T-780) each zone's pairs with the zones of one place after it are dealt that
/// place's trips in chunks of about `chunk`: systematic sampling over the place's zones in their
/// order, by trips, from a start fixed by the two zones' ids, so the same zones give the same pairs
/// on any network and each zone's trips with each later place are kept exactly. A pair of at least
/// a chunk is always solved; smaller ones are solved for a chunk or not at all. The chunks are dealt
/// over every zone, served or not, so a station serving another zone adds pairs without dealing
/// the rest again.
///
/// With `far` (T-2641) the chunk grows with the distance to the place's first zone, `chunk` times
/// that distance over `FAR_CHUNK_KM` from 1 to `FAR_CHUNK_MAX`, and place pairs at least
/// `FAR_DEAL_KM` apart are dealt once over both places' zones: each served origin zone's run of
/// the far place's zones follows the last one's, from a start fixed by the two places' ids, so an
/// origin zone with less than a chunk to a far place gets a whole chunk or nothing. A zone's trips
/// with a far place then stay whole only on average over the place's zones, and a far place's
/// zones get their riders in lumps. Each pair is still priced on its own zones' stations.
pub(super) fn solved_pairs(m: &Market, served: &[usize], joined: &impl Fn(usize, usize) -> bool, chunk: f64, far: bool) -> (Vec<u64>, Vec<f64>) {
    let min = m.rule.min_trips;
    let n = served.len();
    let mut any = vec![0u64; (n as u64 * n.saturating_sub(1) as u64 / 2).div_ceil(64) as usize];
    let mut each = vec![];
    let row = |i: usize| i * n.saturating_sub(1) - i * i.saturating_sub(1) / 2;
    let mut keep = |any: &mut [u64], i: usize, k: usize, trips: f64| {
        let bit = row(i) + k - i - 1;
        any[bit / 64] |= 1 << (bit % 64);
        each.push(trips);
    };
    if chunk <= 0. {
        for (i, &a) in served.iter().enumerate() {
            let za = &m.zones[a];
            for (k, &b) in served.iter().enumerate().skip(i + 1) {
                let zb = &m.zones[b];
                if !m.pr(za.place, zb.place).has_alt() || !joined(i, k) {
                    continue;
                }
                let trips = m.eachway_with(za, zb, m.curves(za, zb));
                if trips >= min {
                    keep(&mut any, i, k, trips);
                }
            }
        }
        return (any, each);
    }
    let mut at = vec![u32::MAX; m.zones.len()];
    let mut some = vec![false; m.places.len()];
    for (k, &z) in served.iter().enumerate() {
        at[z] = k as u32;
        some[m.zones[z].place] = true;
    }
    let mut run: Vec<(usize, f64)> = vec![];
    // T-2641: per place pair, the trips dealt so far
    let mut deal: HashMap<(u32, u32), f64> = HashMap::new();
    for (i, &a) in served.iter().enumerate() {
        let za = &m.zones[a];
        let mut b = a + 1;
        while b < m.zones.len() {
            // one place's zones after a: a run of the zones' order
            let place = m.zones[b].place;
            run.clear();
            if !some[place] || !m.pr(za.place, place).has_alt() {
                while b < m.zones.len() && m.zones[b].place == place {
                    b += 1;
                }
                continue;
            }
            // far places' zones are all at the places' distance
            let fixed = (!m.pr(za.place, place).near).then(|| m.curves(za, &m.zones[b]));
            while b < m.zones.len() && m.zones[b].place == place {
                let zb = &m.zones[b];
                let t = m.eachway_with(za, zb, fixed.unwrap_or_else(|| m.curves(za, zb)));
                if t >= min {
                    run.push((b, t));
                }
                b += 1;
            }
            let Some(&(first, _)) = run.first() else { continue };
            let total: f64 = run.iter().map(|e| e.1).sum();
            let km = m.zone_km(za, &m.zones[first]);
            let chunk = if far { chunk * (km / FAR_CHUNK_KM).clamp(1., FAR_CHUNK_MAX) } else { chunk };
            let draws = (total / chunk).ceil();
            let mut solve = |zb: usize, trips: f64| {
                let k = at[zb];
                if k != u32::MAX && joined(i, k as usize) {
                    keep(&mut any, i, k as usize, trips);
                }
            };
            if far && fixed.is_some() && km >= FAR_DEAL_KM {
                let e = deal.entry((za.place as u32, place as u32)).or_insert(0.);
                let lo0 = *e;
                *e += total;
                let u = unit_hash(za.place as u32, place as u32);
                // chunks before trips x: those at (j + u) chunk below x
                let before = |x: f64| (x / chunk - u).ceil().max(0.);
                let mut lo = lo0;
                for &(zb, t) in &run {
                    let hi = lo + t;
                    let got = before(hi) - before(lo);
                    lo = hi;
                    if got > 0. {
                        solve(zb, got * chunk);
                    }
                }
                continue;
            }
            if draws >= run.len() as f64 {
                for &(zb, t) in &run {
                    solve(zb, t);
                }
                continue;
            }
            let step = total / draws;
            let u = unit_hash(za.gid, m.zones[first].gid);
            // draws before trips x: those at (j + u) x step below x
            let before = |x: f64| (x / step - u).ceil().clamp(0., draws);
            let mut lo = 0.;
            for &(zb, t) in &run {
                let hi = lo + t;
                let got = before(hi) - before(lo);
                lo = hi;
                if got > 0. {
                    solve(zb, got * step);
                }
            }
        }
    }
    (any, each)
}
/// A number in [0, 1) fixed by two ids (splitmix64).
pub(crate) fn unit_hash(a: u32, b: u32) -> f64 {
    let mut z = ((a as u64) << 32 | b as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}
