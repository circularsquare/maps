//! Home-based outings. Job counts are attraction weights, never customer capacities.
//! Coarse destination gravity is integrated by deterministic stratified cell samples. Endpoint
//! distances, the 150 m exclusion, walking and regional overlap use the actual sampled cells.
use super::{
    api::{fare_x_usd, put_fare_curve, scale_triples, DemandApi, PEAK_HOUR_FACTOR},
    clock::now_ms,
    days::{day_rate, everyday, load_factor, share_out, share_ret, SATURDAY, SUNDAY, WEEKDAY},
    kernel::*,
    network::PERIODS,
    pack,
    trip::NEAR_CELL_M,
};
use crate::track::params::PERIOD_HOURS;
use wasm_bindgen::prelude::*;

const NONE: u32 = u32::MAX;
pub const MIN_DISTANCE_M: f32 = 150.0;
/// Purpose kinds, in this order everywhere (`info`, `part_loads`, `selection`; the app's
/// `EVERYDAY_KEYS`): errands, social, school, university, airport (T-142), medical (T-142),
/// attractions (T-263).
/// Kinds 0, 1 and 5 are outings per resident; 2 and 3 attendance per enrolled student (samples
/// drawn from the site); 4 the airport's own daily ground trips (`airport_trips`) and 6 each
/// attraction's visits (`beside_with`), their homes and visitors' ends drawn from each origin cell
/// but normalised per site.
/// The defaults of `Params::everyday_rate` and `everyday_decay_m`. A pack's `everyday` block (its
/// country's survey, T-509 stage 4) overrides the rates; the decay lengths are one worldwide set
/// (T-637), which no pack overrides. An attraction's rate multiplies its visits.
pub const RATE: [f32; KIND_COUNT] = [0.3, 0.2, 0.9, 0.6, 1.0, 0.04, 1.0];
/// Attractions: the chance halves every 10 km.
pub const DECAY: [f32; KIND_COUNT] = [1000.0, 1000.0, 1400.0, 3500.0, 20000.0, 2000.0, 14427.0];
pub const KIND_COUNT: usize = 7;
/// T-1092: this share of errand and social outings goes to central destinations: zones weighed
/// by the purpose's attraction times its attraction per km² of land, at this decay length, so the
/// long trips end in dense centres. The rest are drawn at `DECAY`.
pub const CENTRAL_SHARE: f32 = 0.2;
pub const CENTRAL_DECAY_M: f32 = 6000.0;
fn has_central(kind: usize) -> bool {
    kind < 2
}
/// The purposes' names in `set_extra` (`rate_errands`) and the pack's `everyday` block.
pub const KINDS: [&str; KIND_COUNT] = ["errands", "social", "school", "university", "airport", "medical", "attractions"];
pub const OUTWARD: [[f32; 5]; KIND_COUNT] = [
    [0.10, 0.55, 0.30, 0.04, 0.01],
    [0.05, 0.35, 0.42, 0.17, 0.01],
    [0.94, 0.04, 0.015, 0.004, 0.001],
    [0.65, 0.27, 0.07, 0.009, 0.001],
    // to the airport: early flights put some trips before 6
    [0.30, 0.35, 0.22, 0.06, 0.07],
    // to appointments, and visitors
    [0.30, 0.55, 0.12, 0.02, 0.01],
    // attractions: social's timing (notes/T-263.md)
    [0.05, 0.35, 0.42, 0.17, 0.01],
];
pub const RETURN: [[f32; 5]; KIND_COUNT] = [
    [0.06, 0.49, 0.36, 0.08, 0.01],
    [0.02, 0.25, 0.43, 0.27, 0.03],
    [0.01, 0.82, 0.15, 0.018, 0.002],
    [0.04, 0.42, 0.42, 0.11, 0.01],
    // from the airport: arrivals run late into the evening
    [0.12, 0.33, 0.28, 0.24, 0.03],
    [0.08, 0.60, 0.25, 0.05, 0.02],
    [0.02, 0.25, 0.43, 0.27, 0.03],
];
const AIRPORT: usize = 4;
pub const ATTRACTION: usize = 6;
/// The purposes with few samples a cell (T-142): airports and attractions are a handful of
/// destinations and nobody looks at one cell's medical trips alone (`Sampling::max_few`). Each
/// sample costs the same at opening and in `set_network`.
const FEW: [bool; KIND_COUNT] = [false, false, false, false, true, true, true];

/// A cell's destination samples for a purpose at the fine level (T-397: station boardings within
/// 5% of sixteen with 31% of the slots).
#[cfg(test)]
fn sample_count(kind: usize, trips: f32) -> usize {
    super::lod::Sampling::FINE.count(FEW[kind], trips)
}
/// No airport trip ends this close to a terminal (its own jobs are not its passengers' ends).
const AIRPORT_CLEAR_M: f32 = 1500.0;
/// T-196: the share of the social/leisure attraction that is parks (the pack's `park_draw`, OSM
/// parks by area and residents nearby), taken evenly from the population and leisure-jobs halves.
/// ATUS 2023: 7.6% of leisure outings (eating out, socialising, sport and recreation) end
/// outdoors away from home (notes/T-196.md). A pack without parks keeps 0.5 / 0.5 exactly.
const PARK_SHARE: f64 = 0.08;
/// School and university: samples are drawn from the activity site (homes), the site's
/// enrollment is the budget. Every other kind draws its samples from the home end.
fn site_anchored(kind: usize) -> bool {
    kind == 2 || kind == 3
}

fn unit(mut x: u32) -> f64 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x as f64 + 0.5) / 4294967296.0
}

/// What one purpose's destination sampling reads (T-172: the constructor's loops moved out of
/// line, so the browser optimises them; see `DecayTable::cdf`).
struct Sampler<'a> {
    kind: usize,
    /// `Sampling::salt`
    salt: u32,
    /// each cell's index in this purpose's `live` and `first`, NONE for a cell with no budget
    row: &'a [u32],
    /// this purpose's `first`, less `base` (where its slice starts in `destinations`)
    first: &'a [u32],
    base: usize,
    h3: &'a [u64],
    x: &'a [f32],
    y: &'a [f32],
    /// the zones with any attraction, their mass and each one's cells' running mass
    zones: &'a [usize],
    mass: &'a [f64],
    cell_cdf: &'a [Vec<f64>],
    members: &'a [Vec<usize>],
}

impl Sampler<'_> {
    /// Each cell's stratified destinations for one origin zone, from that zone's `cdf` over
    /// `zones`, into `out` (this purpose's slice of the destinations).
    #[inline(never)]
    fn sample(&self, cells: &[usize], cdf: &[f64], total: f64, out: &mut [u32]) {
        for &c in cells {
            let row = self.row[c];
            if row == NONE {
                continue;
            }
            let at = self.first[row as usize] as usize - self.base;
            let n_k = self.first[row as usize + 1] as usize - self.base - at;
            let seed = (self.h3[c] ^ (self.h3[c] >> 32)) as u32 ^ (self.kind as u32).wrapping_mul(0x9e3779b9) ^ self.salt.wrapping_mul(0x27d4eb2f);
            let offset = unit(seed);
            for k in 0..n_k {
                let target = (k as f64 + offset) / n_k as f64 * total;
                // a cdf with central draws runs over `zones` twice, local then central
                let j = cdf.partition_point(|&v| v <= target).min(cdf.len() - 1);
                let z = self.zones[j % self.zones.len()];
                let target = unit(seed.wrapping_add((k as u32 + 1).wrapping_mul(0x85ebca6b))) * self.mass[z];
                let j = self.cell_cdf[z].partition_point(|&v| v <= target).min(self.members[z].len() - 1);
                let d = self.members[z][j];
                let distance = ((self.x[c] - self.x[d]).powi(2) + (self.y[c] - self.y[d]).powi(2)).sqrt();
                // Excluded trips vanish, including from all-mode denominators. No replacement.
                if distance >= MIN_DISTANCE_M && local_share(distance) > 0.0 {
                    out[at + k] = d as u32;
                }
            }
        }
    }
}

/// `set_network`'s index of the pairs by purpose and subzones (T-398): open addressing over pair
/// numbers, each key read back from its pair, so a slot is 4 bytes; sized from the last network's
/// pair count. Pairs are numbered in first-seen order, so the table changes no result.
struct PairIndex {
    slots: Vec<u32>,
    len: usize,
}

impl PairIndex {
    fn with_capacity(n: usize) -> PairIndex {
        PairIndex { slots: vec![NONE; (n * 4 / 3 + 1).next_power_of_two().max(1 << 12)], len: 0 }
    }
    fn key(kind: u32, u: u32, v: u32) -> u64 {
        (kind as u64) << 58 | (u as u64) << 29 | v as u64
    }
    #[inline]
    fn start(&self, key: u64) -> usize {
        let h = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (h ^ h >> 32) as usize & (self.slots.len() - 1)
    }
    /// The number of the pair (kind, u, v), pushing it onto `pairs` if it is new.
    #[inline]
    fn find_or_push(&mut self, pairs: &mut Vec<Pair>, kind: u32, u: u32, v: u32) -> u32 {
        if 4 * (self.len + 1) > 3 * self.slots.len() {
            self.grow(pairs);
        }
        let key = Self::key(kind, u, v);
        let mask = self.slots.len() - 1;
        let mut i = self.start(key);
        loop {
            let at = self.slots[i];
            if at == NONE {
                pairs.push(Pair::new(kind, u, v));
                let n = (pairs.len() - 1) as u32;
                self.slots[i] = n;
                self.len += 1;
                return n;
            }
            let q = &pairs[at as usize];
            if q.uk == kind << 28 | u && q.v == v {
                return at;
            }
            i = (i + 1) & mask;
        }
    }
    fn grow(&mut self, pairs: &[Pair]) {
        self.slots = vec![NONE; 2 * self.slots.len()];
        let mask = self.slots.len() - 1;
        for (n, q) in pairs.iter().enumerate() {
            let mut i = self.start(Self::key(q.kind(), q.u(), q.v));
            while self.slots[i] != NONE {
                i = (i + 1) & mask;
            }
            self.slots[i] = n as u32;
        }
    }
}

/// exp(-d / decay) between two zone centres by their grid offset (T-172). Zones of one square
/// (the same zone, or zones a region border splits, T-162) are 0.52 of a zone apart. With blocks
/// (T-852) a table by `Gravity`'s layout (`kernel::pair_decay`, `kernel::PairKey`).
struct DecayTable {
    span: usize,
    v: Vec<f32>,
    layout: Option<PairKey>,
}

impl DecayTable {
    fn new(z: &Zones, decay: f32) -> DecayTable {
        if z.mixed() {
            return DecayTable { span: 0, v: pair_decay(z, |d_km| (-d_km * 1000.0 / decay).exp()), layout: Some(PairKey::of(z)) };
        }
        let (gx, gy, zone_m) = (&z.gx, &z.gy, z.zone_m);
        let range = |g: &[i32]| g.iter().max().zip(g.iter().min()).map_or(1, |(a, b)| (a - b) as usize + 1);
        let (wx, wy) = (range(gx), range(gy));
        let mut v = vec![0.0; wx * wy];
        for ax in 0..wx {
            for ay in 0..wy {
                let (dx, dy) = (ax as f32 * zone_m, ay as f32 * zone_m);
                let d = if ax == 0 && ay == 0 { 0.52 * zone_m } else { (dx * dx + dy * dy).sqrt() };
                v[ax * wy + ay] = (-d / decay).exp();
            }
        }
        DecayTable { span: wy, v, layout: None }
    }
    /// The listed zones' positions as `cdf` takes them.
    fn places(&self, z: &Zones, zones: &[usize]) -> (Vec<i32>, Vec<i32>) {
        match &self.layout {
            Some(g) => (zones.iter().map(|&k| g.key(z, k)).collect(), vec![]),
            None => (zones.iter().map(|&k| z.gx[k]).collect(), zones.iter().map(|&k| z.gy[k]).collect()),
        }
    }
    /// The running sum of `mass` x decay from zone `o` to each listed zone (`places`), into
    /// `cdf`; returns the total. Kept out of line: a browser optimises a WebAssembly function only
    /// between calls (no on-stack replacement), so this one is fast after the first few origins
    /// while one long loop inside the constructor would stay in the baseline compiler throughout.
    #[inline(never)]
    fn cdf(&self, z: &Zones, o: usize, zx: &[i32], zy: &[i32], mass: &[f64], cdf: &mut Vec<f64>) -> f64 {
        cdf.clear();
        let mut total = 0.0;
        if let Some(g) = &self.layout {
            let base = g.base(z, o);
            for k in 0..zx.len() {
                total += mass[k] * self.v[(base + zx[k]) as usize] as f64;
                cdf.push(total);
            }
            return total;
        }
        let (gx, gy) = (z.gx[o], z.gy[o]);
        for k in 0..zx.len() {
            let o = (zx[k] - gx).unsigned_abs() as usize * self.span + (zy[k] - gy).unsigned_abs() as usize;
            total += mass[k] * self.v[o] as f64;
            cdf.push(total);
        }
        total
    }
}

/// T-1092: `cdf` (the local draw, summing to `local`) becomes the mixture, a share `q` drawn
/// from `central` (summing to `tc`) appended after it, over the same zones; returns its total. A
/// part with nothing in reach leaves the other the whole draw.
fn mix_central(cdf: &mut Vec<f64>, local: f64, central: &[f64], tc: f64, q: f64) -> f64 {
    let (wl, wc) = match (local > 0.0, tc > 0.0) {
        (true, true) => ((1.0 - q) / local, q / tc),
        (true, false) => (1.0 / local, 0.0),
        (false, true) => (0.0, 1.0 / tc),
        (false, false) => return 0.0,
    };
    cdf.iter_mut().for_each(|v| *v *= wl);
    let base = local * wl;
    cdf.extend(central.iter().map(|&v| base + v * wc));
    base + tc * wc
}

/// T-142: each airport cell's daily trips divided by its draw, the decay-weighted origin weight
/// around it (sum over zones of their origin weight x decay to the airport's zone), as the mass
/// origin cells sample airports by. An origin cell then sends `origin x` the sum over airports of
/// mass x decay from its zone, and every airport receives exactly its trips (before the 150 m
/// exclusion and the 40-80 km taper, which apply to every purpose).
fn airport_draw(z: &Zones, members: &[Vec<usize>], origin: &[f32], trips: &[f32], decay_m: f32) -> Vec<f32> {
    let table = DecayTable::new(z, decay_m);
    let w: Vec<f64> = members.iter().map(|m| m.iter().map(|&c| origin[c] as f64).sum()).collect();
    let zones: Vec<usize> = (0..z.n).filter(|&k| w[k] > 0.0).collect();
    let (zx, zy) = table.places(z, &zones);
    let zw: Vec<f64> = zones.iter().map(|&k| w[k]).collect();
    let mut draw = vec![None; z.n];
    let mut cdf = Vec::with_capacity(zones.len());
    let mut mass = vec![0.0f32; trips.len()];
    for (c, &t) in trips.iter().enumerate() {
        if t <= 0.0 {
            continue;
        }
        let a = z.cell_zone[c] as usize;
        let d = *draw[a].get_or_insert_with(|| table.cdf(z, a, &zx, &zy, &zw, &mut cdf));
        if d > 0.0 {
            mass[c] = (t as f64 / d) as f32;
        }
    }
    mass
}

/// T-263: the attractions (x, y, visitors a year triples) that lie in the pack, each as its index,
/// the nearest cell within `NEAR_CELL_M` and its visits a weekday: a year's visitors over 365
/// days, set on weekdays so the weekend's factors (social's) bring the week back to them.
fn attraction_sites(commutes: &DemandApi, a: &[f32]) -> Vec<(u32, u32, f32)> {
    let city = &commutes.city;
    if a.len() < 3 || city.len() == 0 {
        return vec![];
    }
    let span = |v: &[f32]| v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    let ((x0, x1), (y0, y1)) = (span(&city.x), span(&city.y));
    let p = everyday(ATTRACTION);
    let week = 7.0 / (365.0 * (5.0 + day_rate(SATURDAY, p) + day_rate(SUNDAY, p)));
    let mut out = vec![];
    for (i, t) in a.chunks_exact(3).enumerate() {
        let (x, y, v) = (t[0], t[1], t[2]);
        if !(v > 0.0) || x < x0 - NEAR_CELL_M || x > x1 + NEAR_CELL_M || y < y0 - NEAR_CELL_M || y > y1 + NEAR_CELL_M {
            continue;
        }
        if let Some(c) = commutes.nearest_cell(x, y) {
            out.push((i as u32, c as u32, v * week));
        }
    }
    out
}

mod drives;
#[cfg(test)]
mod tests;
/// T-1222: the levels of detail against each other.
#[cfg(test)]
mod lod_tests;
/// 0..n in pieces of 16k, for loops over every destination sample (3.2M on New York): each piece
/// is one call of an out-of-line function, so the browser optimises it after the first few
/// pieces (T-172; WebAssembly has no on-stack replacement).
fn chunks(n: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    (0..n).step_by(1 << 14).map(move |a| a..(a + (1 << 14)).min(n))
}
pub(crate) fn local_share(d: f32) -> f32 {
    let x = ((d - 40000.0) / 40000.0).clamp(0.0, 1.0);
    1.0 - x * x * (3.0 - 2.0 * x)
}
#[derive(Clone, Copy)]
struct Pair {
    /// the purpose in the top 4 bits, the home subzone below (T-419: 20 bytes a pair)
    uk: u32,
    v: u32,
    trips: f32,
    walk: f32,
    distance: f32,
}
impl Pair {
    fn new(kind: u32, u: u32, v: u32) -> Pair {
        debug_assert!(kind < 16 && u < 1 << 28);
        Pair { uk: kind << 28 | u, v, trips: 0.0, walk: 0.0, distance: 0.0 }
    }
    fn u(&self) -> u32 {
        self.uk & 0x0fff_ffff
    }
    fn kind(&self) -> u32 {
        self.uk >> 28
    }
}
/// One purpose's share of a period's loads (T-158), per route node. The purposes' parts add up
/// to the period's `loads` (assignment is linear in the trips for fixed strategies); only the
/// display filter reads them, crowding and mode choice always use the total.
#[derive(Clone)]
struct Part {
    seg: Vec<f32>,
    rn_board: Vec<f32>,
    rn_alight: Vec<f32>,
}

impl Part {
    fn of(l: &Loads) -> Part {
        Part { seg: l.seg.clone(), rn_board: l.rn_board.clone(), rn_alight: l.rn_alight.clone() }
    }
}

/// `total += l`, every field.
fn add_loads(total: &mut Loads, l: &Loads) {
    for (a, b) in [
        (&mut total.seg, &l.seg),
        (&mut total.board, &l.board),
        (&mut total.alight, &l.alight),
        (&mut total.rn_board, &l.rn_board),
        (&mut total.rn_alight, &l.rn_alight),
        (&mut total.entry, &l.entry),
        (&mut total.exit, &l.exit),
    ] {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }
    total.pax_min += l.pax_min;
}

/// One lane per purpose kind in the everyday assignment (T-433); `RATE` has six kinds.
const LANES: usize = 8;
type Lanes = [f32; LANES];
const ZERO: Lanes = [0.0; LANES];

#[inline(always)]
fn add_lanes(a: &mut Lanes, b: &Lanes) {
    for k in 0..LANES {
        a[k] += b[k];
    }
}

/// Each purpose's station-pair trips by destination station (T-433): for kind k and destination
/// b, entries `off[k][b]..off[k][b + 1]` of `src` and `trips`, origins ascending. `off[k]` is
/// empty for a kind with no trips.
struct Columns {
    off: Vec<Vec<u32>>,
    src: Vec<u32>,
    trips: Vec<f32>,
    /// (destination, origin, trips) in origin order, while packing
    tmp: Vec<(u32, u32, f32)>,
    /// The origin rows of `od` written since the last `pack`.
    rows: Vec<bool>,
}

impl Columns {
    fn new(kinds: usize, s: usize) -> Columns {
        Columns { off: vec![vec![]; kinds], src: vec![], trips: vec![], tmp: vec![], rows: vec![false; s] }
    }
    /// Moves one kind's trips out of `od` (`od[a * s + b]`), leaving it zero.
    fn pack(&mut self, od: &mut [f32], s: usize, kind: usize) {
        debug_assert!(self.off[kind].is_empty(), "the pairs are stored purpose by purpose");
        let mut off = vec![0u32; s + 1];
        self.tmp.clear();
        for (a, row) in od.chunks_exact_mut(s).enumerate() {
            if !std::mem::take(&mut self.rows[a]) {
                continue;
            }
            for (b, v) in row.iter_mut().enumerate() {
                if *v != 0.0 {
                    self.tmp.push((b as u32, a as u32, *v));
                    off[b + 1] += 1;
                    *v = 0.0;
                }
            }
        }
        let base = self.src.len() as u32;
        off[0] = base;
        for b in 0..s {
            off[b + 1] += off[b];
        }
        self.src.resize(base as usize + self.tmp.len(), 0);
        self.trips.resize(self.src.len(), 0.0);
        let mut at = off.clone();
        for &(b, a, v) in &self.tmp {
            let j = at[b as usize] as usize;
            (self.src[j], self.trips[j]) = (a, v);
            at[b as usize] += 1;
        }
        self.off[kind] = off;
    }
}

/// `Loads` with a lane per purpose kind.
struct LaneLoads {
    seg: Vec<Lanes>,
    board: Vec<Lanes>,
    alight: Vec<Lanes>,
    rn_board: Vec<Lanes>,
    rn_alight: Vec<Lanes>,
    entry: Vec<Lanes>,
    exit: Vec<Lanes>,
    pax_min: [f64; LANES],
    /// `Loads::rn_fare_x` of every purpose together
    rn_fare_x: Vec<f32>,
}

impl LaneLoads {
    fn zero(s: usize, nr: usize) -> LaneLoads {
        LaneLoads { seg: vec![ZERO; nr], board: vec![ZERO; s], alight: vec![ZERO; s], rn_board: vec![ZERO; nr], rn_alight: vec![ZERO; nr], entry: vec![ZERO; s], exit: vec![ZERO; s], pax_min: [0.0; LANES], rn_fare_x: vec![0.0; nr] }
    }
    fn lane(&self, k: usize) -> Loads {
        let f = |v: &[Lanes]| v.iter().map(|x| x[k]).collect();
        Loads { seg: f(&self.seg), board: f(&self.board), alight: f(&self.alight), rn_board: f(&self.rn_board), rn_alight: f(&self.rn_alight), entry: f(&self.entry), exit: f(&self.exit), pax_min: self.pax_min[k], rn_fare_x: vec![0.0; self.seg.len()] }
    }
}

trait NodeIx: Copy {
    fn ix(self) -> usize;
}
impl NodeIx for u16 {
    #[inline(always)]
    fn ix(self) -> usize {
        self as usize
    }
}
impl NodeIx for u32 {
    #[inline(always)]
    fn ix(self) -> usize {
        self as usize
    }
}

/// `kernel::assign_with_transfers` for every purpose in one walk of each destination's strategy
/// (T-433): each lane does the kernel's `DestWalk` arithmetic in its order, so lane k is the
/// assignment of kind k's trips alone to the bit (`lanes_match_the_kernel_assignment` checks it;
/// a change to `DestWalk` is made here too). The transfers count every purpose's riders together,
/// their lanes summed before the split onto departures (DECISIONS.md, T-433).
fn assign_lanes(g: &Graph, sk: &Skims, cols: &Columns, ix: &XferIndex, xfer: &mut [f32]) -> LaneLoads {
    let (s, n) = (sk.s, sk.n);
    let mut l = LaneLoads::zero(s, n - s);
    let np = sk.most_change_nodes();
    let mut flow = vec![ZERO; n + np];
    let counting = !(cfg!(test) && XFER_OFF.load(std::sync::atomic::Ordering::Relaxed));
    let mut head = if counting { vec![NONE; s + np] } else { vec![] };
    let mut rec: Vec<(u32, f32, u32)> = vec![];
    let mut use_k: Vec<(u32, f32)> = vec![];
    let mut dep: Vec<(u32, f32)> = vec![];
    let mut km: Vec<f32> = vec![];
    let mut fxq: Vec<f32> = vec![];
    let bent = !g.fare_curve.is_line();
    let used: Vec<(usize, &[u32])> = cols.off.iter().enumerate().filter(|(_, o)| !o.is_empty()).map(|(k, o)| (k, &o[..])).collect();
    for dst in 0..s {
        let mut total = ZERO;
        for &(k, off) in &used {
            for j in off[dst] as usize..off[dst + 1] as usize {
                let (src, v) = (cols.src[j] as usize, cols.trips[j]);
                flow[src][k] = v;
                l.entry[src][k] += v;
                total[k] += v;
            }
        }
        add_lanes(&mut l.exit[dst], &total);
        let slot = sk.slot[dst];
        debug_assert!(total == ZERO || slot != NONE, "flow towards station {dst}, which was not searched");
        if total != ZERO && slot != NONE {
            let slot = slot as usize;
            if counting {
                head.fill(NONE);
                rec.clear();
            }
            if bent {
                let st = sk.strategy(slot);
                trip_km(g, st, dst, &mut km, &mut dep);
                board_extra(g, st, dst, |src| flow[src].iter().sum(), &km, &mut dep, &mut fxq, &mut l.rn_fare_x);
            }
            let mut w = LaneWalk { g, s, n, dst, bits: &sk.board[slot][..], cn: &sk.change[slot], flow: &mut flow, l: &mut l, head: &mut head, rec: &mut rec, use_k: &mut use_k, dep: &mut dep, xfer: if counting { Some((ix, &mut *xfer)) } else { None } };
            match (&sk.next[slot], &sk.order[slot]) {
                (NodeIds::Short(next), NodeIds::Short(order)) => w.run(next, order),
                (NodeIds::Long(next), NodeIds::Long(order)) => w.run(next, order),
                _ => unreachable!("a destination's ids in two widths"),
            }
        }
        // the walk leaves zero every node it took flow from; the rest set here
        flow[dst] = ZERO;
        for &(_, off) in &used {
            for j in off[dst] as usize..off[dst + 1] as usize {
                flow[cols.src[j] as usize] = ZERO;
            }
        }
        debug_assert!(flow.iter().all(|f| *f == ZERO));
    }
    l
}

struct LaneWalk<'a> {
    g: &'a Graph,
    s: usize,
    n: usize,
    dst: usize,
    bits: &'a [u64],
    cn: &'a ChangeNodes,
    flow: &'a mut [Lanes],
    l: &'a mut LaneLoads,
    head: &'a mut [u32],
    rec: &'a mut Vec<(u32, f32, u32)>,
    use_k: &'a mut Vec<(u32, f32)>,
    dep: &'a mut Vec<(u32, f32)>,
    xfer: Option<(&'a XferIndex, &'a mut [f32])>,
}

impl LaneWalk<'_> {
    #[inline(always)]
    fn run<I: NodeIx>(&mut self, next: &[I], order: &[I]) {
        let LaneWalk { g, s, n, dst, bits, cn, ref mut flow, ref mut l, ref mut head, ref mut rec, ref mut use_k, ref mut dep, ref mut xfer } = *self;
        let end = n + cn.len();
        for &v in order.iter().skip(1).rev() {
            let v = v.ix();
            let f = flow[v];
            if f == ZERO {
                continue;
            }
            flow[v] = ZERO;
            if v < s || v >= n {
                let st = departures(g, bits, cn, v, dep);
                let hx = if v < s { v } else { s + v - n };
                let sum: f32 = dep.iter().map(|x| x.1).sum();
                if sum == 0.0 {
                    continue;
                }
                add_lanes(&mut l.board[st], &f);
                for &(e, a) in dep.iter() {
                    let to = g.adj_to[e as usize] as usize;
                    let share: Lanes = std::array::from_fn(|k| f[k] * a / sum);
                    add_lanes(&mut l.rn_board[to - s], &share);
                    let nx = next[to].ix();
                    if nx >= s && nx < n {
                        add_lanes(&mut flow[to], &share);
                    } else {
                        // as the kernel's DestWalk: a boarding rides a stop first (T-1055)
                        let r = to - s;
                        add_lanes(&mut flow[g.rn_next[r] as usize], &share);
                        add_lanes(&mut l.seg[r], &share);
                        let ride = g.rn_ride[r];
                        for k in 0..LANES {
                            l.pax_min[k] += (share[k] * ride) as f64;
                        }
                    }
                }
                if head.get(hx).is_some_and(|&h| h != NONE) {
                    use_k.clear();
                    for &(e, b) in dep.iter() {
                        if b > 0.0 {
                            use_k.push((e - g.adj_off[st], b / sum));
                        }
                    }
                    let x = &mut xfer.as_mut().unwrap().1;
                    let mut i = head[hx];
                    while i != NONE {
                        let (at, fin, nx) = rec[i as usize];
                        for &(k, share) in use_k.iter() {
                            x[(at + k) as usize] += fin * share;
                        }
                        i = nx;
                    }
                }
            } else {
                let to = next[v].ix();
                if to >= end {
                    continue;
                }
                add_lanes(&mut flow[to], &f);
                if to >= s && to < n {
                    add_lanes(&mut l.seg[v - s], &f);
                    let ride = g.rn_ride[v - s];
                    for k in 0..LANES {
                        l.pax_min[k] += (f[k] * ride) as f64;
                    }
                } else {
                    add_lanes(&mut l.alight[g.rn_station[v - s] as usize], &f);
                    add_lanes(&mut l.rn_alight[v - s], &f);
                    let (st, hx) = if to < s { (to, to) } else { (cn.st[to - n] as usize, s + to - n) };
                    if st != dst {
                        if let Some((ix, _)) = xfer.as_ref() {
                            let e = (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).find(|&e| g.adj_to[e] as usize == st).unwrap();
                            rec.push((ix.at[e], f.iter().fold(0.0, |a, &b| a + b), head[hx]));
                            head[hx] = (rec.len() - 1) as u32;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch (T-433): every pass also assigns each purpose alone with the kernel and
    /// compares.
    static CHECK_LANES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Lane k of `lanes` against `kernel::assign` of kind k's trips alone, to the bit; the transfers
/// against the kernel's, purpose after purpose, to f32 rounding.
#[cfg(test)]
fn check_lanes(g: &Graph, sk: &Skims, cols: &Columns, ix: &XferIndex, lanes: &LaneLoads, xfer: &[f32]) {
    let s = sk.s;
    let mut want = vec![0.0f32; ix.len];
    for (k, off) in cols.off.iter().enumerate() {
        if off.is_empty() {
            continue;
        }
        let mut od = vec![0.0f32; s * s];
        for b in 0..s {
            for j in off[b] as usize..off[b + 1] as usize {
                od[cols.src[j] as usize * s + b] = cols.trips[j];
            }
        }
        let r = assign_with_transfers(g, sk, &od, ix, &mut want);
        let got = lanes.lane(k);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (name, a, b) in [("seg", &r.seg, &got.seg), ("board", &r.board, &got.board), ("alight", &r.alight, &got.alight), ("rn_board", &r.rn_board, &got.rn_board), ("rn_alight", &r.rn_alight, &got.rn_alight), ("entry", &r.entry, &got.entry), ("exit", &r.exit, &got.exit)] {
            assert_eq!(bits(a), bits(b), "kind {k}: {name}");
        }
        assert_eq!(r.pax_min.to_bits(), got.pax_min.to_bits(), "kind {k}: pax_min");
    }
    for (a, b) in want.iter().zip(xfer) {
        assert!((a - b).abs() <= 1e-5 * a.abs().max(1e-3), "transfers {a} against {b}");
    }
}

struct Period {
    net: super::network::Network,
    p: Params,
    g: Graph,
    loads: Loads,
    /// per purpose kind, as `totals`
    parts: Vec<Part>,
    probability: Vec<[f32; 2]>,
    round: usize,
    ms: f64,
    fac: Vec<f32>,
    uniform: bool,
    /// `kernel::settled` (T-434)
    seen: Vec<f32>,
    trips: f64,
    rail: f64,
    /// T-1300: `ModeOut::sense`, averaged like the shares
    rail_d: FareSense,
    walk: f64,
    /// And from walking the whole way, by purpose (T-509's walk shares).
    walk_lost: Vec<f64>,
    /// Transfers platform to platform, every purpose together (T-176), averaged like the loads.
    xi: XferIndex,
    xfer: Vec<f32>,
}

#[wasm_bindgen]
pub struct EverydayApi {
    engine: DemandApi,
    destinations: Vec<u32>,
    edge_pair: Vec<u32>,
    pairs: Vec<Pair>,
    /// Each subzone's part of its zone for walks round water (T-700): its busiest one.
    sz_part: Vec<u8>,
    per: Vec<Option<Period>>,
    totals: Vec<f64>,
    /// Trips walking the whole way with no rail, by purpose (T-558); the
    /// parameters as at opening (`ep`).
    walks: Vec<f64>,
    /// The everyday layer's parameters as at opening (`Params::everyday`), and per zone its
    /// streets' and main roads' minutes a km and its parking: the samples' choice with no rail.
    ep: Params,
    wplace: Vec<(f32, f32)>,
    wpark: Vec<(f32, f32, f32)>,
    /// trip kilometres by purpose, for `info` (summed once at opening, T-172)
    km: Vec<f64>,
    /// Optional enrollment budgets: school generation samples homes FROM each activity site.
    school_enrollment: Vec<f32>,
    university_enrollment: Vec<f32>,
    /// T-142, airports: each cell's share of the ground trips' home and visitor ends (sums to
    /// 1), and per origin zone the airports' normalised draw from there (`airport_draw`), so a
    /// cell's airport trips are `airport_origin[c] x airport_scale[zone]`. Empty without airports.
    airport_origin: Vec<f32>,
    airport_scale: Vec<f64>,
    /// T-263, attractions, as the airports': visitors' home and lodging ends, and the draw.
    attraction_origin: Vec<f32>,
    attraction_scale: Vec<f64>,
    /// Each attraction given to `beside_with` that lies in this pack: its index there, its cell
    /// and its visits a weekday.
    attractions: Vec<(u32, u32, f32)>,
    /// Where each purpose's samples start in `destinations` (one more entry than purposes),
    /// purpose by purpose.
    start: Vec<usize>,
    /// Per purpose, the cells with a budget for it (ascending), the only ones holding samples
    /// (T-395: about half of all cells x purposes have none).
    live: Vec<Vec<u32>>,
    /// Per purpose, where each `live` cell's samples start in `destinations`, and one more entry
    /// (`Sampling::count` of them, T-397).
    first: Vec<Vec<u32>>,
    /// The last `owner` answer: the sample loops go in order, so the next is that one or close.
    at: std::cell::Cell<(usize, usize)>,
    /// `Params::everyday_rate` as at opening: the samples and totals were drawn with it.
    rate: [f32; KIND_COUNT],
    /// T-1175: per purpose, each zone's share of its outings that stay in the pack (the pack's
    /// `everyday_kept_<purpose>`; the rest cross a zone line in the regional layer), by the zone
    /// of the sample's fixed end. Empty: all of them.
    kept: Vec<Vec<f32>>,
}

impl EverydayApi {
    /// The parameters every period starts from (checks; to change before `set_network`).
    pub fn params_mut(&mut self) -> &mut Params {
        self.engine.params_mut()
    }
    /// Rail trips starting at each demand station (first boarding) in period `q`, every purpose
    /// (checks: T-509's station entries).
    pub fn entries(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|v| v.as_ref()).map_or(vec![], |s| s.loads.entry.clone())
    }
    /// As `DemandApi::transfers` (T-176): every purpose's transfers over the periods held.
    pub fn transfers(&self, sel: &[u8]) -> Vec<f32> {
        let mut out = vec![];
        for st in self.per.iter().flatten() {
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
        }
        out
    }
    /// `cell_modes` for one purpose kind only (T-158's display filter).
    pub fn cell_modes_of(&self, kind: u32) -> Vec<f32> {
        self.modes_of(Some(kind as usize), WEEKDAY)
    }
}

/// What the trip explorer reads of one everyday trip in a solved period (T-586, trip.rs
/// `EverydayApi::trip`).
pub(crate) struct TripInputs<'a> {
    pub engine: &'a DemandApi,
    pub g: &'a Graph,
    pub p: &'a Params,
    pub round: usize,
    /// The purpose's subzone pair of the two cells, if any sample joins them: its mean distance
    /// (metres) and trips a day, both directions.
    pub pair: Option<(f32, f32)>,
    /// The two cells' share walking the whole way with no rail (`no_rail`).
    pub no_rail: f32,
    /// The walk around water between the subzones' zones (T-275), as the pass weighs it.
    pub walk_factor: f32,
    /// The same between the two cells (T-720: the explorer's own trip).
    pub walk_factor_cells: f32,
}

impl EverydayApi {
    pub(crate) fn engine(&self) -> &DemandApi {
        &self.engine
    }
    /// Purpose `kind` from home cell `c` to activity cell `d` in period `q`; None unless solved.
    pub(crate) fn trip_inputs(&self, kind: usize, q: usize, c: usize, d: usize) -> Option<TripInputs<'_>> {
        let st = self.per.get(q)?.as_ref()?;
        let sz = self.engine.sz.as_ref();
        let (u, v) = sz.map_or((NONE, NONE), |s| (s.cell[c], s.cell[d]));
        let pair = if u == NONE || v == NONE {
            None
        } else {
            self.pairs.iter().find(|x| x.kind() as usize == kind && x.u() == u && x.v == v).map(|x| (x.distance, x.trips))
        };
        let walk_factor = match sz {
            Some(s) if u != NONE && v != NONE => self.walk_factor_sz(s, u as usize, v as usize),
            _ => self.walk_factor(c, d),
        };
        let walk_factor_cells = self.walk_factor(c, d);
        Some(TripInputs { engine: &self.engine, g: &st.g, p: &st.p, round: st.round, pair, no_rail: self.no_rail(c, d, self.distance(c, d)), walk_factor, walk_factor_cells })
    }
    /// One sample of every subzone pair: its purpose, home and activity cells, and the pair's
    /// rail chance outward in period `q` (trip.rs's check).
    #[cfg(test)]
    pub(crate) fn pair_samples(&self, q: usize) -> Vec<(usize, usize, usize, f32)> {
        let Some(st) = self.per.get(q).and_then(|v| v.as_ref()) else { return vec![] };
        let mut seen = vec![false; self.pairs.len()];
        let mut out = vec![];
        for k in 0..self.edge_pair.len() {
            let pi = self.edge_pair[k];
            if pi == NONE || seen[pi as usize] {
                continue;
            }
            seen[pi as usize] = true;
            if let Some((kind, c, d, _, _)) = self.edge(k) {
                out.push((kind, c, d, st.probability[pi as usize][0]));
            }
        }
        out
    }
}

impl EverydayApi {
    /// Trips, walking trips and trip kilometres by purpose, over destination samples `r` (see
    /// `chunks`).
    #[inline(never)]
    fn add_totals(&mut self, r: std::ops::Range<usize>) {
        for k in r {
            let Some((kind, c, d, per)) = self.ends(k) else { continue };
            let distance = self.distance(c, d);
            let (w, walk) = self.weight(kind, per, c, d, distance);
            self.totals[kind] += w as f64;
            self.walks[kind] += (w * walk) as f64;
            self.km[kind] += distance as f64 * w as f64 / 1000.0;
        }
    }
    /// A sample's share walking the whole way with no rail (T-558, `Params::walk_no_rail`),
    /// against the car each way (parking at the far end): the two ways' mean.
    fn no_rail(&self, c: usize, d: usize, distance: f32) -> f32 {
        let p = &self.ep;
        let z = &self.engine.z;
        let (hz, az) = (z.cell_zone[c] as usize, z.cell_zone[d] as usize);
        let cw = p.walk_trip_cost(distance * self.walk_factor(c, d) / 1000.0);
        if cw >= NO_WALK {
            return 0.0;
        }
        let mut out = 0.0;
        // T-854: the household's car is parked at home, so both ways pay the home end's parking
        let hp = p.home_park_cell(c);
        for (u, v) in [(hz, az), (az, hz)] {
            let f = p.drive_factor(u, v);
            let (fixed, paid, free) = self.wpark[v];
            let road = p.car_drive(p.drive_min_f(distance, f, self.wplace[u], self.wplace[v]), distance * 0.001 * f) + fixed + hp;
            out += 0.5 * p.walk_no_rail(road, paid, free, cw);
        }
        out
    }
    /// T-275: the walk goes around water between cells c and d.
    fn walk_factor(&self, c: usize, d: usize) -> f32 {
        self.engine.gr.walk_factor_cells(&self.engine.z, c, d)
    }
    /// The same between subzones u and v, each at its busiest part of its zone (T-700).
    fn walk_factor_sz(&self, sz: &Subzones, u: usize, v: usize) -> f32 {
        let part = |s: usize| self.sz_part.get(s).copied().unwrap_or(0);
        self.engine.gr.wroutes.as_ref().map_or(1.0, |r| r.walk_factor_parts(sz.zone[u] as usize, part(u), sz.zone[v] as usize, part(v))).max(1.0)
    }
    /// `sz_part` for the present subzones.
    fn set_sz_part(&mut self) {
        let (Some(sz), Some(r)) = (self.engine.sz.as_ref(), self.engine.gr.wroutes.as_ref()) else {
            self.sz_part.clear();
            return;
        };
        let pop = self.engine.pop();
        let np = r.parts_max();
        let mut w = vec![0f32; sz.n * np];
        for (c, &s) in sz.cell.iter().enumerate() {
            if s != NONE {
                w[s as usize * np + r.part(c) as usize] += pop[c] + 1e-3;
            }
        }
        self.sz_part = w.chunks_exact(np).map(|v| (0..np).fold(0, |b, p| if v[p] > v[b] { p } else { b }) as u8).collect();
    }
    /// Join destination samples `r` to the subzone pairs they fall in (for `set_network`).
    #[inline(never)]
    fn index_edges(&mut self, r: std::ops::Range<usize>, index: &mut PairIndex) {
        for k in r {
            let Some((kind, c, d, per)) = self.ends(k) else {
                continue;
            };
            let Some(sz) = &self.engine.sz else { return };
            let (u, v) = (sz.cell[c], sz.cell[d]);
            // both ends in a subzone first: the weights cost an exp and a square root
            // the home end walks or rides (T-389), the activity end walks
            if u == NONE || v == NONE || sz.walk(v as usize).is_empty() {
                continue;
            }
            let distance = self.distance(c, d);
            let (w, pw) = self.weight(kind, per, c, d, distance);
            // a pair of weightless samples (a zone keeping none of a purpose, T-1257) would average
            // its distance and walking share as 0/0, NaN in every load it touches
            if !(w > 0.0) {
                continue;
            }
            let i = index.find_or_push(&mut self.pairs, kind as u32, u, v);
            if self.edge_pair.is_empty() {
                self.edge_pair.resize(self.destinations.len(), NONE);
            }
            self.edge_pair[k] = i;
            let pair = &mut self.pairs[i as usize];
            pair.trips += w;
            pair.walk += w * pw;
            pair.distance += w * distance;
        }
    }
    fn distance(&self, c: usize, d: usize) -> f32 {
        let city = &self.engine.city;
        ((city.x[c] - city.x[d]).powi(2) + (city.y[c] - city.y[d]).powi(2)).sqrt()
    }
    /// Destination sample `k`'s purpose, home cell, activity cell and its cell's sample count.
    fn ends(&self, k: usize) -> Option<(usize, usize, usize, usize)> {
        let d = self.destinations[k];
        if d == NONE {
            return None;
        }
        let (kind, i) = self.owner(k);
        let fixed = self.live[kind][i] as usize;
        let per = (self.first[kind][i + 1] - self.first[kind][i]) as usize;
        Some(if site_anchored(kind) { (kind, d as usize, fixed, per) } else { (kind, fixed, d as usize, per) })
    }
    /// Sample `k`'s purpose and its fixed cell's index in `live`.
    fn owner(&self, k: usize) -> (usize, usize) {
        let (mut kind, mut i) = self.at.get();
        if k < self.first[kind][i] as usize {
            (kind, i) = (0, 0);
        }
        while self.start[kind + 1] <= k {
            kind += 1;
            i = 0;
        }
        let f = &self.first[kind];
        if f[i + 1] as usize <= k {
            i += 1;
            if f[i + 1] as usize <= k {
                i = f.partition_point(|&x| x as usize <= k) - 1;
            }
        }
        self.at.set((kind, i));
        (kind, i)
    }
    /// A sample's trips and walking share with no rail, from its ends, their distance and its
    /// cell's samples.
    fn weight(&self, kind: usize, per: usize, c: usize, d: usize, distance: f32) -> (f32, f32) {
        let budget = match kind {
            2 => self.school_enrollment[d],
            3 => self.university_enrollment[d],
            AIRPORT => (self.airport_origin[c] as f64 * self.airport_scale[self.engine.z.cell_zone[c] as usize]) as f32,
            ATTRACTION => (self.attraction_origin[c] as f64 * self.attraction_scale[self.engine.z.cell_zone[c] as usize]) as f32,
            _ => self.engine.pop()[c],
        };
        let kept = self.kept.get(kind).filter(|k| !k.is_empty()).map_or(1.0, |k| k[self.engine.z.cell_zone[if site_anchored(kind) { d } else { c }] as usize]);
        let weight = budget * self.rate[kind] / per as f32 * local_share(distance) * kept;
        (weight, self.no_rail(c, d, distance))
    }
    fn edge(&self, k: usize) -> Option<(usize, usize, usize, f32, f32)> {
        let (kind, c, d, per) = self.ends(k)?;
        let (w, pw) = self.weight(kind, per, c, d, self.distance(c, d));
        Some((kind, c, d, w, pw))
    }
    /// Also returns the transfers, every purpose together, laid out by `ix` (T-176).
    /// Each pair's rail share both ways, the loads, the purposes' loads, the transfers, the walkers
    /// rail takes by purpose and (T-1300) the rail trips' fare sensitivity (`ModeOut::sense`).
    fn pass(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> (Vec<[f32; 2]>, Loads, Vec<Part>, Vec<f32>, Vec<f64>, FareSense) {
        let mut probability = vec![[0.0; 2]; self.pairs.len()];
        let s = g.n_st;
        let kinds = self.totals.len();
        let mut xfer = vec![0.0; ix.len];
        let mut walk_lost = vec![0.0f64; kinds];
        let mut rail_d = FareSense::default();
        let Some(sz) = &self.engine.sz else {
            let zero = Loads::zero(s, g.n - s);
            return (probability, zero.clone(), vec![Part::of(&zero); kinds], xfer, walk_lost, rail_d);
        };
        // Outward AND homeward destinations: every station either end of a pair can reach (T-142:
        // not every station, so a city with no pairs near the network, open in a worker beside
        // the one being built in, searches nothing; each destination's search is independent,
        // so this changes no result).
        let mut want = vec![false; s];
        // 1 an activity end, 2 a home end (walks and rides: the trip home alights there, T-389)
        let mut used = vec![0u8; sz.n];
        for pair in &self.pairs {
            used[pair.u() as usize] |= 2;
            used[pair.v as usize] |= 1;
        }
        for u in (0..sz.n).filter(|&u| used[u] != 0) {
            let r = if used[u] & 2 != 0 { sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize } else { sz.walk(u) };
            for k in r {
                want[sz.acc_st[k] as usize] = true;
            }
        }
        if !want.contains(&true) {
            let zero = Loads::zero(s, g.n - s);
            return (probability, zero.clone(), vec![Part::of(&zero); kinds], xfer, walk_lost, rail_d);
        }
        let (loads, parts) = with_skims(g, p, &want, |sk| {
            // a table per distance band (T-710)
            let ers = station_exp(sk, p);
            let ss = s * s;
            // One station-pair matrix, filled by one purpose's pairs (they are
            // stored purpose by purpose) and packed into its columns; then one walk of each
            // destination's strategy assigns every purpose, each kept apart for the display filter
            // (T-158, T-433). Their sum is the total.
            assert!(kinds <= LANES);
            let mut od = vec![0.0; s * s];
            let mut cols = Columns::new(kinds, s);
            let mut od_kind = usize::MAX;
            let mut od_used = false;
            // per band (T-710) and subzone: the cheapest walk and each walk's weight at it; and
            // for a home with rides (T-389), the cheapest of walks and rides and every entry's
            // weight at that
            let bands: Vec<(f32, f32)> = (0..p.n_bands()).map(|b| p.band(b)).collect();
            let na = sz.acc_st.len();
            let mut min = vec![0.0; sz.n];
            let mut min_r = vec![f32::INFINITY; sz.n];
            // `min_r` as the access bound sees it (`Subzones::bound_min`)
            let mut min_rb = vec![f32::INFINITY; sz.n];
            let mut accs = vec![0.0; bands.len() * na];
            let mut acc_rs = vec![0.0; bands.len() * na];
            let lowest = |r: std::ops::Range<usize>| sz.acc_w[r].iter().copied().fold(f32::INFINITY, f32::min);
            for u in 0..sz.n {
                let r = sz.walk(u);
                min[u] = lowest(r.clone());
                let all = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
                if !sz.ride(u).is_empty() {
                    min_r[u] = lowest(all.clone());
                    min_rb[u] = sz.bound_min(&p, u);
                }
                for (b, &(th, _)) in bands.iter().enumerate() {
                    for k in r.clone() {
                        accs[b * na + k] = fast_exp(-th * (sz.acc_w[k] - min[u]));
                    }
                    if !sz.ride(u).is_empty() {
                        for k in all.clone() {
                            acc_rs[b * na + k] = fast_exp(-th * (sz.acc_w[k] - min_r[u]));
                        }
                    }
                }
            }
            let ride_m = p.ride_min_km * 1000.0;
            let rule = p.walk_rule > 0.0 && sk.walk.len() == ss;
            // T-1059: per subzone, the most its trips walk before their last station, and the most
            // they walk from one: a pair within its distance on both counts keeps every pair
            // (and per access entry its walk, and that plus the most its station walks on)
            let (mut acc_walk, mut acc_from) = (vec![0f32; if rule { na } else { 0 }], vec![0f32; if rule { na } else { 0 }]);
            let (mut rule_from, mut rule_to) = (vec![0f32; if rule { sz.n } else { 0 }], vec![0f32; if rule { sz.n } else { 0 }]);
            if rule {
                for u in 0..sz.n {
                    for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                        acc_walk[k] = walked(sz, u, k);
                        acc_from[k] = acc_walk[k] + sk.walk_max[sz.acc_st[k] as usize] as f32;
                        rule_from[u] = rule_from[u].max(acc_from[k]);
                        rule_to[u] = rule_to[u].max(acc_walk[k]);
                    }
                }
            }
            let density = self.engine.z.density();
            let park: Vec<(f32, f32, f32)> = (0..self.engine.z.n).map(|j| p.park_parts(&self.engine.z, j)).collect();
            for (i, pair) in self.pairs.iter().enumerate() {
                if pair.kind() as usize != od_kind {
                    if od_used {
                        cols.pack(&mut od, s, od_kind);
                    }
                    od_kind = pair.kind() as usize;
                    od_used = false;
                }
                // the pair's distance band (T-710)
                let bd = p.band_of_d2(pair.distance * pair.distance);
                let (th, off) = bands[bd];
                let (acc, acc_r) = (&accs[bd * na..(bd + 1) * na], &acc_rs[bd * na..(bd + 1) * na]);
                let er = &ers[bd * ss..(bd + 1) * ss];
                // the home end: its walks, or on a trip of `ride_min_km` or more its walks and
                // rides (T-389, as the commute's `ride_offered` but on the pair's own distance);
                // the activity end walks
                let h = pair.u() as usize;
                let (hr, hm, hw, hb) = if pair.distance >= ride_m && min_r[h].is_finite() {
                    (sz.acc_off[h] as usize..sz.acc_off[h + 1] as usize, min_r[h], &acc_r, min_rb[h])
                } else {
                    (sz.walk(h), min[h], &acc, min[h])
                };
                let x = pair.v as usize;
                let (xr, xm) = (sz.walk(x), min[x]);
                // T-558: walking the whole way, over the pair's mean distance
                let walk_on = pair.distance * self.walk_factor_sz(sz, h, x);
                let cw = p.walk_trip_cost(walk_on / 1000.0);
                for direction in 0..2 {
                    let ((a, ma, wa), (b, mb, wb)) = if direction == 0 {
                        ((hr.clone(), hm, hw), (xr.clone(), xm, &acc))
                    } else {
                        ((xr.clone(), xm, &acc), (hr.clone(), hm, hw))
                    };
                    if hb + xm > p.pair_walk_max {
                        continue;
                    }
                    let (u, v) = if direction == 0 { (h, x) } else { (x, h) };
                    // T-1059: station pairs walking further than walking there are no option
                    let check = rule && rule_from[u] + rule_to[v] > walk_on;
                    let keep = |ka: usize, kb: usize| {
                        !check || acc_from[ka] + rule_to[v] <= walk_on || walk_ok(acc_walk[ka], sk.walk[sz.acc_st[ka] as usize * s + sz.acc_st[kb] as usize], acc_walk[kb], walk_on)
                    };
                    let mut sum = 0.0;
                    for ka in a.clone() {
                        for kb in b.clone() {
                            if keep(ka, kb) {
                                sum += wa[ka]
                                    * wb[kb]
                                    * er[sz.acc_st[ka] as usize * s + sz.acc_st[kb] as usize];
                            }
                        }
                    }
                    if sum < 1e-30 {
                        continue;
                    }
                    let (fixed, paid, free) = park[sz.zone[v] as usize];
                    let f = p.drive_factor(sz.zone[u] as usize, sz.zone[v] as usize);
                    let road = p.car_drive(
                        p.drive_min_f(pair.distance, f, p.place_min_per_km(density[sz.zone[u] as usize]), p.place_min_per_km(density[sz.zone[v] as usize])),
                        pair.distance * 0.001 * f,
                    ) + fixed
                        + sz.home_park[h];
                    let rail = ma + mb + p.fare_min - fast_ln(sum) / th + off;
                    let bt = p.mode_beta(road);
                    let (util, uw) = (bt * (road - rail) + p.asc_rail, bt * (road - cw) + p.asc_walk());
                    let pr = p.car_choice(util, uw, bt, paid, free, rail, road);
                    probability[i][direction] = pr;
                    let share = if direction == 0 {
                        OUTWARD[pair.kind() as usize][q]
                    } else {
                        RETURN[pair.kind() as usize][q]
                    };
                    let prd = p.car_choice(util - bt * FARE_D_MIN, uw, bt, paid, free, rail + FARE_D_MIN, road);
                    let (dd, km) = ((pair.trips * share * (prd - pr)) as f64 / FARE_D_MIN as f64, pair.distance as f64 / 1000.0);
                    rail_d.d += dd;
                    rail_d.d_km += dd * km;
                    rail_d.km += (pair.trips * share * pr) as f64 * km;
                    // T-558: rail takes its share of the pair's walking with no rail, as the
                    // samples' sums and views have it
                    walk_lost[pair.kind() as usize] += (pair.trips * share * pr * pair.walk) as f64;
                    od_used = true;
                    let trips = pair.trips * share * pr / sum;
                    for ka in a.clone() {
                        cols.rows[sz.acc_st[ka] as usize] = true;
                        for kb in b.clone() {
                            if !keep(ka, kb) {
                                continue;
                            }
                            let ia = sz.acc_st[ka] as usize;
                            let ib = sz.acc_st[kb] as usize;
                            od[ia * s + ib] += trips * wa[ka] * wb[kb] * er[ia * s + ib];
                        }
                    }
                }
            }
            if od_used {
                cols.pack(&mut od, s, od_kind);
            }
            drop(od);
            let lanes = assign_lanes(g, sk, &cols, ix, &mut xfer);
            #[cfg(test)]
            if CHECK_LANES.with(|c| c.get()) {
                check_lanes(g, sk, &cols, ix, &lanes, &xfer);
            }
            let parts: Vec<Loads> = (0..kinds).map(|k| lanes.lane(k)).collect();
            let mut total = Loads::zero(s, g.n - s);
            for l in &parts {
                add_loads(&mut total, l);
            }
            total.rn_fare_x = lanes.rn_fare_x;
            (total, parts.iter().map(Part::of).collect::<Vec<_>>())
        });
        (probability, loads, parts, xfer, walk_lost, rail_d)
    }
    /// Trips, rail trips and walking trips (less `walk_lost`, the walkers rail takes) in period `q`.
    fn mode_totals(&self, q: usize, probability: &[[f32; 2]], walk_lost: &[f64]) -> (f64, f64, f64) {
        let mut trips = 0.0;
        let mut walk = 0.0;
        let mut rail = 0.0;
        for kind in 0..self.totals.len() {
            let share = (OUTWARD[kind][q] + RETURN[kind][q]) as f64;
            trips += self.totals[kind] * share;
            walk += self.walks[kind] * share - walk_lost[kind];
        }
        for (i, pair) in self.pairs.iter().enumerate() {
            let r = pair.trips as f64
                * (OUTWARD[pair.kind() as usize][q] as f64 * probability[i][0] as f64
                    + RETURN[pair.kind() as usize][q] as f64 * probability[i][1] as f64);
            rail += r;
        }
        (trips, rail, walk.max(0.0))
    }
}

#[wasm_bindgen]
impl EverydayApi {
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str, bin: &[u8]) -> Result<EverydayApi, String> {
        Self::beside(&DemandApi::new(header, bin)?, header, bin)
    }
    /// The everyday layer of the city whose commutes `commutes` holds (opened from the same
    /// pack), sharing its cells, zones, gravity and routes (T-399).
    pub fn beside(commutes: &DemandApi, header: &str, bin: &[u8]) -> Result<EverydayApi, String> {
        Self::beside_with(commutes, header, bin, &[])
    }
    /// `beside` with attractions (T-263): (x, y, visitors a year) triples in the pack's frame,
    /// each taken where a cell of the pack lies within `NEAR_CELL_M` (`attractions` lists them).
    pub fn beside_with(commutes: &DemandApi, header: &str, bin: &[u8], attractions: &[f32]) -> Result<EverydayApi, String> {
        let n = commutes.city.len();
        let sites = attraction_sites(commutes, attractions);
        let mut attraction_visits = vec![0f32; n];
        for &(_, c, v) in &sites {
            attraction_visits[c as usize] += v;
        }
        let pop =
            pack::cell_weights(header, bin, "pop", n)?.ok_or("missing resident population")?;
        let retail = pack::cell_weights(header, bin, "errand_jobs", n)?
            .ok_or("missing errand attractors")?;
        let leisure = pack::cell_weights(header, bin, "leisure_jobs", n)?
            .ok_or("missing leisure attractors")?;
        let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
        let (tp, tl) = (sum(&pop), sum(&leisure));
        if tp <= 0.0 {
            return Err("empty everyday weights".into());
        }
        // T-582: a purpose with no sites in the pack (no retail, no leisure, no school or
        // university, no airport, no medical) is an empty slot, not an error.
        let nonzero = |v: Option<Vec<f32>>| v.filter(|v| sum(v) > 0.0);
        let pair = |a: Option<Vec<f32>>, b: Option<Vec<f32>>, what: &str| -> Result<(Option<Vec<f32>>, Option<Vec<f32>>), String> {
            if a.is_some() != b.is_some() {
                return Err(format!("incomplete {what} inputs"));
            }
            Ok(match (a, b) {
                (Some(a), Some(b)) if sum(&a) > 0.0 && sum(&b) > 0.0 => (Some(a), Some(b)),
                _ => (None, None),
            })
        };
        // T-196: parks, where the pack has them, take PARK_SHARE of the social attraction
        let parks = pack::cell_weights(header, bin, "park_draw", n)?.filter(|v| sum(v) > 0.0);
        let park_share = if parks.is_some() { PARK_SHARE } else { 0.0 };
        let (half_pop, half_leisure) = if tl > 0.0 { (0.5 * (1.0 - park_share), 0.5 * (1.0 - park_share)) } else { (1.0 - park_share, 0.0) };
        let tk = parks.as_ref().map_or(1.0, |p| sum(p));
        let social: Vec<f32> = (0..n)
            .map(|c| {
                let leisure_part = if tl > 0.0 { half_leisure * leisure[c] as f64 / tl } else { 0.0 };
                let park_part = parks.as_ref().map_or(0.0, |p| park_share * p[c] as f64 / tk);
                (half_pop * pop[c] as f64 / tp + leisure_part + park_part) as f32
            })
            .collect();
        let (children, school_enrollment) = pair(
            pack::cell_weights(header, bin, "school_children", n)?,
            pack::cell_weights(header, bin, "school_enrollment", n)?,
            "school",
        )?;
        let (adults, university_enrollment) = pair(
            pack::cell_weights(header, bin, "university_adults", n)?,
            pack::cell_weights(header, bin, "university_enrollment", n)?,
            "university",
        )?;
        // T-142: airport ground trips per terminal cell and medical visit sizes per cell. A pack
        // without them (or with no airport, Springfield) leaves the purpose empty.
        let airport_trips = nonzero(pack::cell_weights(header, bin, "airport_trips", n)?);
        let medical = nonzero(pack::cell_weights(header, bin, "medical_visits", n)?);
        // Access centroids and road density use real residents/employment, not commuter weights.
        let jobs = pack::cell_weights(header, bin, "jobs", n)?.ok_or("missing jobs")?;
        let engine = commutes.with_weights(pop, jobs);
        let sampling = engine.lod.sampling;
        let (rate, decay) = (engine.base.everyday_rate, engine.base.everyday_decay_m);
        let nz = engine.z.n;
        let mut members = vec![vec![]; nz];
        for c in 0..n {
            members[engine.z.cell_zone[c] as usize].push(c);
        }
        // Airports: the home and visitor ends are half residents, a quarter jobs (business
        // visitors) and a quarter leisure jobs (hotels, restaurants, attractions), each normalised.
        // Each airport keeps its own total: its trips are divided by its draw over every origin
        // (`airport_draw`), so an origin cell's sampled destinations pick airports by that
        // normalised mass times the decay, and the cell sends its weight times their sum.
        let (airport_origin, airport_mass) = match &airport_trips {
            Some(trips) => {
                let tj = sum(engine.jobs()).max(1e-9);
                let mut origin: Vec<f32> = (0..n)
                    .map(|c| (0.5 * engine.pop()[c] as f64 / tp + 0.25 * engine.jobs()[c] as f64 / tj + 0.25 * leisure[c] as f64 / tl.max(1e-9)) as f32)
                    .collect();
                // the airport's own jobs are not where its passengers come from (a terminal's
                // block holds tens of thousands of airport jobs): no ends within 1.5 km of one
                let sites: Vec<usize> = (0..n).filter(|&c| trips[c] > 0.0).collect();
                for (c, o) in origin.iter_mut().enumerate() {
                    let (x, y) = (engine.city.x[c], engine.city.y[c]);
                    if sites.iter().any(|&a| (engine.city.x[a] - x).powi(2) + (engine.city.y[a] - y).powi(2) < AIRPORT_CLEAR_M * AIRPORT_CLEAR_M) {
                        *o = 0.0;
                    }
                }
                let mass = airport_draw(&engine.z, &members, &origin, trips, decay[AIRPORT]);
                (origin, Some(mass))
            }
            None => (vec![], None),
        };
        // T-263, attractions: visitors' other ends half residents and half leisure jobs (where
        // they eat and stay), each normalised, every attraction keeping its own visits as an
        // airport does. No clearance as round a terminal: an attraction's own leisure jobs are
        // few, and with dozens of attractions in a centre it would take the centre's hotels out.
        let (attraction_origin, attraction_mass) = if sites.is_empty() {
            (vec![], None)
        } else {
            let origin: Vec<f32> = (0..n)
                .map(|c| {
                    let lodging = if tl > 0.0 { 0.5 * leisure[c] as f64 / tl } else { 0.0 };
                    ((if tl > 0.0 { 0.5 } else { 1.0 }) * engine.pop()[c] as f64 / tp + lodging) as f32
                })
                .collect();
            let mass = airport_draw(&engine.z, &members, &origin, &attraction_visits, decay[ATTRACTION]);
            (origin, Some(mass))
        };
        // Purpose indices stay stable: a missing purpose before a present one is an empty slot.
        let zeros = vec![0.0f32; n];
        let optional = [children.as_ref(), adults.as_ref(), airport_mass.as_ref(), medical.as_ref(), attraction_mass.as_ref()];
        let mut weights = vec![&retail, &social];
        if let Some(last) = optional.iter().rposition(|o| o.is_some()) {
            weights.extend(optional[..=last].iter().map(|o| o.unwrap_or(&zeros)));
        }
        let kinds = weights.len();
        let mut airport_scale = vec![0.0f64; if airport_mass.is_some() { nz } else { 0 }];
        let mut attraction_scale = vec![0.0f64; if attraction_mass.is_some() { nz } else { 0 }];
        // trips sent from each cell: residents, or a school's or university's enrollment
        let budget_of = |kind: usize| -> &[f32] {
            match kind {
                2 => school_enrollment.as_deref().unwrap_or(&[]),
                3 => university_enrollment.as_deref().unwrap_or(&[]),
                AIRPORT => &airport_origin,
                ATTRACTION => &attraction_origin,
                _ => engine.pop(),
            }
        };
        let mut start = vec![0usize];
        let mut live: Vec<Vec<u32>> = vec![];
        let mut first: Vec<Vec<u32>> = vec![];
        for kind in 0..kinds {
            let budget = budget_of(kind);
            live.push((0..n as u32).filter(|&c| budget.get(c as usize).is_some_and(|&b| !(b <= 0.0))).collect());
            let mut f = Vec::with_capacity(live[kind].len() + 1);
            let mut at = start[kind];
            f.push(at as u32);
            for &c in &live[kind] {
                at += sampling.count(FEW[kind], budget[c as usize] * rate[kind]);
                f.push(at as u32);
            }
            first.push(f);
            start.push(at);
        }
        let mut destinations = vec![NONE; start[kinds]];
        let mut row = vec![NONE; n];
        for (kind, weights) in weights.into_iter().enumerate() {
            let mut mass = vec![0.0; nz];
            let mut cell_cdf = vec![vec![]; nz];
            for z in 0..nz {
                for &c in &members[z] {
                    mass[z] += weights[c] as f64;
                    cell_cdf[z].push(mass[z]);
                }
            }
            let zones: Vec<usize> = (0..nz).filter(|&z| mass[z] > 0.0).collect();
            // T-172: zone centres sit on the zone grid, so the decay between two zones depends
            // only on their grid offset: one table per purpose instead of a square root and an
            // exp for every zone pair (5 s for New York's four purposes, now a fraction of that).
            let zmass: Vec<f64> = zones.iter().map(|&z| mass[z]).collect();
            let table = DecayTable::new(&engine.z, decay[kind]);
            let (zx, zy) = table.places(&engine.z, &zones);
            let q = if has_central(kind) { engine.base.everyday_central_share.clamp(0.0, 1.0) as f64 } else { 0.0 };
            let mut central = (q > 0.0).then(|| {
                let z = &engine.z;
                let land = |k: usize| (z.ncell[k].max(1) as f32 * CELL_KM2).min(z.square_km2(k)) as f64;
                let cmass: Vec<f64> = zones.iter().map(|&k| mass[k] * mass[k] / land(k)).collect();
                (DecayTable::new(z, engine.base.everyday_central_m), cmass, Vec::with_capacity(zones.len()))
            });
            let budget = budget_of(kind);
            row.fill(NONE);
            for (i, &c) in live[kind].iter().enumerate() {
                row[c as usize] = i as u32;
            }
            let s = Sampler {
                salt: sampling.salt,
                kind,
                row: &row,
                first: &first[kind],
                base: start[kind],
                h3: &engine.city.h3,
                x: &engine.city.x,
                y: &engine.city.y,
                zones: &zones,
                mass: &mass,
                cell_cdf: &cell_cdf,
                members: &members,
            };
            let out = &mut destinations[start[kind]..start[kind + 1]];
            let mut cdf = Vec::with_capacity(zones.len());
            for oz in 0..nz {
                // a zone with nothing to send needs no destinations
                if !members[oz].iter().any(|&c| budget.get(c).copied().unwrap_or(0.0) > 0.0) {
                    continue;
                }
                let mut total = table.cdf(&engine.z, oz, &zx, &zy, &zmass, &mut cdf);
                if let Some((ctable, cmass, ccdf)) = central.as_mut() {
                    let tc = ctable.cdf(&engine.z, oz, &zx, &zy, cmass, ccdf);
                    total = mix_central(&mut cdf, total, ccdf, tc, q);
                }
                match kind {
                    AIRPORT => airport_scale[oz] = total,
                    ATTRACTION => attraction_scale[oz] = total,
                    _ => {}
                }
                if total > 0.0 {
                    s.sample(&members[oz], &cdf, total, out);
                }
            }
        }
        let kept = (0..kinds)
            .map(|k| pack::zone_shares(header, bin, &format!("everyday_kept_{}", KINDS[k]), nz).map(Option::unwrap_or_default))
            .collect::<Result<Vec<_>, _>>()?;
        let ep = engine.base.everyday();
        let wplace = engine.z.density().iter().map(|&d| ep.place_min_per_km(d)).collect();
        let wpark = (0..engine.z.n).map(|j| ep.park_parts(&engine.z, j)).collect();
        let mut api = EverydayApi {
            engine,
            ep,
            wplace,
            wpark,
            destinations,
            edge_pair: vec![],
            pairs: vec![],
            sz_part: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            totals: vec![0.0; kinds],
            walks: vec![0.0; kinds],
            km: vec![0.0; kinds],
            school_enrollment: school_enrollment.unwrap_or_default(),
            university_enrollment: university_enrollment.unwrap_or_default(),
            airport_origin,
            airport_scale,
            attraction_origin,
            attraction_scale,
            attractions: sites,
            start,
            live,
            first,
            at: std::cell::Cell::new((0, 0)),
            rate,
            kept,
        };
        for r in chunks(api.destinations.len()) {
            api.add_totals(r);
        }
        Ok(api)
    }
    /// Outings retained, walking fraction, and mean distance by purpose.
    pub fn info(&self) -> Vec<f64> {
        let distance = &self.km;
        (0..self.totals.len())
            .flat_map(|k| {
                [
                    self.totals[k],
                    self.walks[k] / self.totals[k].max(1e-9),
                    distance[k] / self.totals[k].max(1e-9),
                ]
            })
            .collect()
    }
    pub fn purpose_count(&self) -> u32 {
        self.totals.len() as u32
    }
    /// The level its samples were drawn at: its commutes' (`DemandApi::at_level`, T-1222).
    pub fn level(&self) -> u32 {
        self.engine.lod.level
    }
    /// The attractions `beside_with` placed in the pack (T-263): per attraction its index in the
    /// list given, its cell and its visits a weekday.
    pub fn attractions(&self) -> Vec<f32> {
        self.attractions.iter().flat_map(|&(i, c, v)| [i as f32, c as f32, v]).collect()
    }
    /// The station complexes of the networks `set_network` is given from now on (T-161), as
    /// `DemandApi::set_complexes`.
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.engine.set_complexes(complex);
    }
    /// `DemandApi::set_upgrades` (T-1281).
    pub fn set_upgrades(&mut self, ups: &[u8]) {
        self.engine.set_upgrades(ups);
    }
    /// `DemandApi::set_passages` (T-1281).
    pub fn set_passages(&mut self, pairs: &[u32]) {
        self.engine.set_passages(pairs);
    }
    /// `DemandApi::set_span` (T-842).
    pub fn set_span(&mut self, span: &[f32]) {
        self.engine.set_span(span);
    }
    /// `DemandApi::set_bus` (T-1464).
    pub fn set_bus(&mut self, catch_n: &[u32], catch_xy: &[f32]) {
        self.engine.set_bus(catch_n, catch_xy);
    }
    pub fn set_network(
        &mut self,
        xy: &[f32],
        line_n: &[u32],
        stops: &[u32],
        times: &[f32],
        tph: &[f32],
        cars: &[f32],
    ) -> Result<f64, String> {
        let t = now_ms();
        self.engine
            .set_network(xy, line_n, stops, times, tph, cars)?;
        // T-124: as the engine, the periods whose inputs did not change are kept
        let svc = self.engine.svc.as_ref().unwrap();
        for (q, v) in self.per.iter_mut().enumerate() {
            match v {
                Some(st) if self.engine.same >> q & 1 == 1 => st.net = DemandApi::period_net(svc, q),
                _ => *v = None,
            }
        }
        if self.engine.layout_same {
            return Ok(now_ms() - t);
        }
        // T-398: the pair list keeps its room for the next network; the samples' pairs are
        // written in place, and only once a sample has a pair (none read as `NONE`)
        let last = self.pairs.len();
        self.pairs.clear();
        self.edge_pair.clear();
        if self.engine.sz.is_none() {
            return Ok(now_ms() - t);
        }
        self.set_sz_part();
        // about one pair in eight samples on New York's first network
        let mut index = PairIndex::with_capacity(if last > 0 { last } else { self.destinations.len() / 8 });
        for r in chunks(self.destinations.len()) {
            self.index_edges(r, &mut index);
        }
        for p in &mut self.pairs {
            p.walk = (p.walk / p.trips).clamp(0.0, 1.0);
            p.distance /= p.trips;
        }
        Ok(now_ms() - t)
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, hop_km: &[f32]) -> Result<(), String> {
        self.engine.set_fares(base, per_km, hop_km)?;
        if self.engine.same == 0 {
            self.per.iter_mut().for_each(|v| *v = None);
        }
        Ok(())
    }
    /// T-1277: `DemandApi::set_fare_weight`.
    pub fn set_fare_weight(&mut self, w: f32) -> Result<(), String> {
        self.engine.set_fare_weight(w)?;
        if self.engine.same == 0 {
            self.per.iter_mut().for_each(|v| *v = None);
        }
        Ok(())
    }
    pub fn set_fare_curve(&mut self, knots: &[f32]) -> Result<(), String> {
        self.engine.set_fare_curve(knots)?;
        if self.engine.same == 0 {
            self.per.iter_mut().for_each(|v| *v = None);
        }
        Ok(())
    }
    pub fn solve(&mut self, q: usize) -> bool {
        if q >= PERIODS {
            return false;
        }
        let Some(svc) = &self.engine.svc else {
            return false;
        };
        let t = now_ms();
        let (h0, h1) = PERIOD_HOURS[q];
        let p = Params {
            period: q,
            period_share: 1.0,
            period_hours: (h1 - h0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.engine.base.everyday()
        };
        let net = DemandApi::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        for (r, fare) in g.rn_fare.iter_mut().enumerate() {
            *fare = self.engine.fare_km.get(r).copied().unwrap_or(0.0) * self.engine.fare_per_km;
        }
        put_fare_curve(&mut g, &self.engine.fare_curve, &self.engine.fare_km, self.engine.base.fare_min, self.engine.fare_per_km, self.engine.base.fare_w);
        let fac = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &fac);
        let xi = xfer_index(&g);
        let (probability, loads, parts, xfer, walk_lost, rail_d) = self.pass(&g, &p, q, &xi);
        let (trips, rail, walk) = self.mode_totals(q, &probability, &walk_lost);
        self.per[q] = Some(Period {
            rail_d,
            walk_lost,
            net,
            p,
            g,
            loads,
            parts,
            probability,
            round: 0,
            ms: now_ms() - t,
            fac,
            uniform: true,
            seen: vec![],
            trips,
            rail,
            walk,
            xi,
            xfer,
        });
        true
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(Some(st)) = self.per.get(q) else {
            return false;
        };
        // the crowding and the waits for full trains (T-188) from every population's loads
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            return false;
        };
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        st.round += 1;
        if !settled(&mut st.seen, &total[..st.g.rn_line.len()]) && (!st.uniform || fac != st.fac || left != st.g.rn_left) {
            st.uniform = false;
            st.fac = fac;
            set_crowding(&mut st.g, &st.fac);
            st.g.rn_left = left;
            let (probability, loads, parts, xfer, walk_lost, rail_d) = self.pass(&st.g, &st.p, q, &st.xi);
            for (old, new) in st.walk_lost.iter_mut().zip(&walk_lost) {
                *old += (new - *old) / (st.round as f64 + 1.0);
            }
            st.rail_d.toward(rail_d, 1.0 / (st.round as f64 + 1.0));
            for (old, new) in st.probability.iter_mut().zip(&probability) {
                msa(old, new, st.round);
            }
            msa(&mut st.xfer, &xfer, st.round);
            msa(&mut st.loads.seg, &loads.seg, st.round);
            msa(&mut st.loads.rn_board, &loads.rn_board, st.round);
            msa(&mut st.loads.rn_alight, &loads.rn_alight, st.round);
            msa(&mut st.loads.entry, &loads.entry, st.round);
            msa(&mut st.loads.exit, &loads.exit, st.round);
            msa(&mut st.loads.rn_fare_x, &loads.rn_fare_x, st.round);
            // the same averaging, so the purposes still add up to the total
            for (old, new) in st.parts.iter_mut().zip(&parts) {
                msa(&mut old.seg, &new.seg, st.round);
                msa(&mut old.rn_board, &new.rn_board, st.round);
                msa(&mut old.rn_alight, &new.rn_alight, st.round);
            }
            let (trips, rail, walk) = self.mode_totals(q, &st.probability, &st.walk_lost);
            st.trips = trips;
            st.rail = rail;
            st.walk = walk;
        }
        st.ms = now_ms() - t;
        self.per[q] = Some(st);
        true
    }
    /// `DemandApi::cap_with_external` (T-1070): each pair's rail share falls by the share of the
    /// entries at the end it sets out from that still fit, those stations weighed as its access
    /// costs weigh them.
    pub fn cap_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(Some(st)) = self.per.get(q) else { return false };
        let Some(fit) = capacity_fit(&st.g, &st.net, &st.p, total) else { return false };
        if !fit.any || self.engine.sz.is_none() {
            return true;
        }
        let mut st = self.per[q].take().unwrap();
        let sz = self.engine.sz.as_ref().unwrap();
        let (keep, all) = cap_loads(&st.g, &fit, &st.xi, &mut st.loads, &mut st.xfer);
        for part in st.parts.iter_mut() {
            cap_part(&fit, &mut part.seg, &mut part.rn_board, &mut part.rn_alight);
        }
        let p = &st.p;
        let ride_m = p.ride_min_km * 1000.0;
        let kept_at = |r: std::ops::Range<usize>, th: f32| {
            let lo = sz.acc_w[r.clone()].iter().copied().fold(f32::INFINITY, f32::min);
            let (mut w, mut k) = (0f32, 0f32);
            for i in r {
                let e = fast_exp(-th * (sz.acc_w[i] - lo));
                w += e;
                k += e * keep[sz.acc_st[i] as usize];
            }
            if w > 0.0 { k / w } else { 1.0 }
        };
        // per band, subzone and its walks alone or with its rides, worked out once
        let mut memo = vec![f32::NAN; 2 * p.n_bands() * sz.n];
        let mut kept = |u: usize, rides: bool, b: usize| {
            let i = (2 * b + rides as usize) * sz.n + u;
            if memo[i].is_nan() {
                let r = if rides { sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize } else { sz.walk(u) };
                memo[i] = kept_at(r, p.band(b).0);
            }
            memo[i]
        };
        let kinds = self.totals.len();
        let (mut lost0, mut lost1) = (vec![0f64; kinds], vec![0f64; kinds]);
        for (i, pair) in self.pairs.iter().enumerate() {
            let pr = &mut st.probability[i];
            if pr[0] == 0.0 && pr[1] == 0.0 {
                continue;
            }
            let b = p.band_of_d2(pair.distance * pair.distance);
            let h = pair.u() as usize;
            let k = pair.kind() as usize;
            let before = OUTWARD[k][q] * pr[0] + RETURN[k][q] * pr[1];
            pr[0] *= kept(h, pair.distance >= ride_m && !sz.ride(h).is_empty(), b);
            pr[1] *= kept(pair.v as usize, false, b);
            let after = OUTWARD[k][q] * pr[0] + RETURN[k][q] * pr[1];
            lost0[k] += (pair.trips * before * pair.walk) as f64;
            lost1[k] += (pair.trips * after * pair.walk) as f64;
        }
        for k in 0..kinds {
            if lost0[k] > 0.0 {
                st.walk_lost[k] *= lost1[k] / lost0[k];
            }
        }
        // the pairs' shares by where they set out, their sum the entries kept
        let (_, rail, _) = self.mode_totals(q, &st.probability, &st.walk_lost);
        if rail > 0.0 {
            let f = (st.rail * all / rail) as f32;
            for pr in st.probability.iter_mut() {
                pr[0] = (pr[0] * f).min(1.0);
                pr[1] = (pr[1] * f).min(1.0);
            }
            st.walk_lost.iter_mut().for_each(|v| *v *= f as f64);
        }
        let (trips, rail, walk) = self.mode_totals(q, &st.probability, &st.walk_lost);
        if st.rail > 0.0 {
            st.rail_d.scale(rail / st.rail);
        }
        st.trips = trips;
        st.rail = rail;
        st.walk = walk;
        self.per[q] = Some(st);
        true
    }
    /// As `DemandApi::summary`, without the crowding figures (0).
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| {
                vec![
                    s.round as f64,
                    s.trips,
                    s.rail,
                    s.walk,
                    s.ms,
                    self.pairs.len() as f64,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    s.rail_d.d,
                    s.rail_d.d_km,
                    s.rail_d.km,
                ]
            })
    }
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.seg.clone())
    }
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.rn_board.clone())
    }
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per
            .get(q)
            .and_then(|v| v.as_ref())
            .map_or(vec![], |s| s.loads.rn_alight.clone())
    }
    /// As `DemandApi::fare_extra`.
    pub fn fare_extra(&self, q: usize) -> Vec<f32> {
        self.per.get(q).and_then(|v| v.as_ref()).map_or(vec![], |s| fare_x_usd(&s.loads.rn_fare_x, self.engine.base.fare_w))
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        (0..st.loads.seg.len())
            .map(|r| {
                let l = st.g.rn_line[r] as usize;
                let cap = 60.0 / st.net.line_headway_min[l][q] * st.net.line_crush[l];
                if cap > 0.0 {
                    st.loads.seg[r] / st.p.period_hours * st.p.peak_hour_factor / cap
                } else {
                    0.0
                }
            })
            .collect()
    }
    pub fn solved_mask(&self) -> u32 {
        (0..5)
            .filter(|&q| self.per[q].is_some())
            .fold(0, |m, q| m | 1 << q)
    }
    /// Each purpose's share of `seg`, `board` and `alight` (T-158): for each purpose kind in
    /// `info` order, three blocks of one value per route node (riders on the segment, boarding,
    /// alighting). They add up to the three totals. Empty before the period is solved.
    pub fn part_loads(&self, q: usize) -> Vec<f32> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        let mut out = Vec::with_capacity(st.parts.len() * 3 * st.loads.seg.len());
        for part in &st.parts {
            out.extend_from_slice(&part.seg);
            out.extend_from_slice(&part.rn_board);
            out.extend_from_slice(&part.rn_alight);
        }
        out
    }
    /// Six mode-major cell blocks, home and activity ends, trip legs over solved periods.
    pub fn cell_modes(&self) -> Vec<f32> {
        self.modes_of(None, WEEKDAY)
    }
    /// `cell_modes` on a kind of day (T-260, `days.rs`): that day's trips and period shares.
    pub fn cell_modes_day(&self, day: u32) -> Vec<f32> {
        self.modes_of(None, day as usize)
    }
    /// `cell_modes_of` on a kind of day.
    pub fn cell_modes_of_day(&self, kind: u32, day: u32) -> Vec<f32> {
        self.modes_of(Some(kind as usize), day as usize)
    }
    /// Each purpose kind's trips, rail trips and walked trips in period `q` (a weekday's), three
    /// numbers a kind in `info` order; the kinds add up to `summary`'s (T-260: the weekend's
    /// mode split is worked out from them). Empty before the period is solved.
    pub fn kind_totals(&self, q: usize) -> Vec<f64> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        let nk = self.totals.len();
        let mut out = vec![0.0; 3 * nk];
        for kind in 0..nk {
            let share = (OUTWARD[kind][q] + RETURN[kind][q]) as f64;
            out[3 * kind] = self.totals[kind] * share;
            out[3 * kind + 2] = (self.walks[kind] * share - st.walk_lost[kind]).max(0.0);
        }
        for (i, pair) in self.pairs.iter().enumerate() {
            let pr = st.probability[i];
            out[3 * pair.kind() as usize + 1] += pair.trips as f64
                * (OUTWARD[pair.kind() as usize][q] as f64 * pr[0] as f64 + RETURN[pair.kind() as usize][q] as f64 * pr[1] as f64);
        }
        out
    }
    /// Trips walked the whole way in period `q` by purpose (T-509: survey walk shares), as
    /// `kind_totals` has them.
    pub fn kind_walks_all(&self, q: usize) -> Vec<f64> {
        let Some(Some(st)) = self.per.get(q) else {
            return vec![];
        };
        (0..self.totals.len())
            .map(|k| (self.walks[k] * (OUTWARD[k][q] + RETURN[k][q]) as f64 - st.walk_lost[k]).max(0.0))
            .collect()
    }
    /// `transfers` on a kind of day (T-260). The purposes' transfers are kept together, so each
    /// period's are scaled by its rail riders' mean load factor that day.
    pub fn transfers_day(&self, sel: &[u8], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (q, st) in self.per.iter().enumerate() {
            let Some(st) = st else { continue };
            let at = out.len();
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
            if day as usize != WEEKDAY {
                let k = self.kind_totals(q);
                let (mut rail, mut weighted) = (0.0, 0.0);
                for kind in 0..k.len() / 3 {
                    rail += k[3 * kind + 1];
                    weighted += k[3 * kind + 1] * load_factor(day as usize, everyday(kind), q) as f64;
                }
                scale_triples(&mut out[at..], if rail > 0.0 { (weighted / rail) as f32 } else { 0.0 });
            }
        }
        out
    }
}

#[wasm_bindgen]
impl EverydayApi {
    /// `cell_modes_day` over the periods in `mask` (bit q = period q) whether solved or not, no
    /// rail where a period has no solve (T-325): a zone no station reaches, held by the
    /// auxiliary worker alone, answers for each solving worker's periods in turn, so its bubbles
    /// add up exactly as a solve's would.
    pub fn cell_modes_mask_day(&self, mask: u32, day: u32) -> Vec<f32> {
        self.modes_in(None, day as usize, Some(mask))
    }
    /// `cell_modes_of_day` over the periods in `mask`, as `cell_modes_mask_day`.
    pub fn cell_modes_of_mask_day(&self, kind: u32, mask: u32, day: u32) -> Vec<f32> {
        self.modes_in(Some(kind as usize), day as usize, Some(mask))
    }
}

impl EverydayApi {
    fn modes_of(&self, only: Option<usize>, day: usize) -> Vec<f32> {
        self.modes_in(only, day, None)
    }
    /// `mask` None: the solved periods, each with its rail. Some: the periods in the mask; a
    /// period's rail only where it has a solve.
    fn modes_in(&self, only: Option<usize>, day: usize, mask: Option<u32>) -> Vec<f32> {
        let n = self.engine.city.len();
        let mut out = vec![0.0; 6 * n];
        // edges are stored purpose by purpose (`edge`)
        let range = match only {
            Some(kind) if kind < self.totals.len() => self.start[kind]..self.start[kind + 1],
            Some(_) => 0..0,
            None => 0..self.destinations.len(),
        };
        for k in range {
            let Some((kind, c, d, w, pw)) = self.edge(k) else {
                continue;
            };
            let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
            let mut total = 0.0;
            let mut rail = 0.0;
            let p = everyday(kind);
            for q in 0..5 {
                let held = match mask {
                    Some(m) => m >> q & 1 == 1,
                    None => self.per[q].is_some(),
                };
                if held {
                    let (o, r) = (share_out(day, p, q), share_ret(day, p, q));
                    total += w * (o + r);
                    if let (true, Some(st)) = (pi != NONE, &self.per[q]) {
                        let pr = st.probability[pi as usize];
                        rail += w * (o * pr[0] + r * pr[1]);
                    }
                }
            }
            // rail takes from walking and the car in proportion (T-558)
            let walk = (total - rail) * pw;
            let drive = (total - walk - rail).max(0.0);
            for (mode, v) in [rail, walk, drive].into_iter().enumerate() {
                out[mode * n + c] += v;
                out[(mode + 3) * n + d] += v;
            }
        }
        out
    }
}

#[wasm_bindgen]
impl EverydayApi {
    /// Each enabled purpose: 30 timing values + 3 far-cell mode blocks. All legs; no cut.
    pub fn selection(&self, end: u32, cells: &[u32]) -> Vec<f32> {
        self.selection_day(end, cells, WEEKDAY as u32)
    }
    /// `selection` on a kind of day (T-260): that day's trips and period shares.
    pub fn selection_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        let n = self.engine.city.len();
        let stride = 30 + 3 * n;
        let mut out = vec![0.0; self.totals.len() * stride];
        for (kind, (timing, far, legs)) in self.selection_parts(end, cells, day, u32::MAX).into_iter().enumerate() {
            let base = kind * stride;
            out[base..base + 30].copy_from_slice(&timing);
            for (i, &c) in far.iter().enumerate() {
                for m in 0..3 {
                    out[base + 30 + m * n + c as usize] = legs[3 * i + m];
                }
            }
        }
        out
    }
    /// `selection_day` packed for the demand worker (T-285): only the far cells some sample of
    /// the selection reaches, the same numbers. Per purpose kind: 30 timing values, `m`, then `m`
    /// far cells (ascending), their rail, their walking and their driving legs.
    pub fn selection_sparse_day(&self, end: u32, cells: &[u32], day: u32) -> Vec<f32> {
        self.selection_sparse_mask_day(end, cells, u32::MAX, day)
    }
    /// `selection_sparse_day` over the solved periods in `mask` alone (bit q = period q; T-336),
    /// as `cell_modes_mask_day`: a zone held by the auxiliary worker alone answers for each
    /// solving worker's periods in turn.
    pub fn selection_sparse_mask_day(&self, end: u32, cells: &[u32], mask: u32, day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (timing, far, legs) in self.selection_parts(end, cells, day, mask) {
            out.extend_from_slice(&timing);
            out.push(far.len() as f32);
            out.extend(far.iter().map(|&c| c as f32));
            for m in 0..3 {
                out.extend((0..far.len()).map(|i| legs[3 * i + m]));
            }
        }
        out
    }
}

impl EverydayApi {
    /// Per purpose kind: the selection's 30 timing values, the far cells its samples reach
    /// (ascending) and their rail, walking and driving legs (3 per cell). T-285: only the
    /// selection's samples are visited and weighed. A purpose whose samples are laid out by the
    /// selected end (homes for outings, sites for schools) reads their ranges directly; at the
    /// other end the sample list is scanned for selected cells, and only those are weighed. The
    /// samples are taken in the order of the whole list, so every sum is the same as visiting all.
    fn selection_parts(&self, end: u32, cells: &[u32], day: u32, mask: u32) -> Vec<([f32; 30], Vec<u32>, Vec<f32>)> {
        let day = day as usize;
        let n = self.engine.city.len();
        let mut selected = vec![false; n];
        for &c in cells {
            if (c as usize) < n {
                selected[c as usize] = true;
            }
        }
        let sorted: Vec<usize> = (0..n).filter(|&c| selected[c]).collect();
        // a far cell's slot in this kind's lists
        let mut slot = vec![NONE; n];
        let kinds = self.totals.len();
        let mut parts = Vec::with_capacity(kinds);
        for kind in 0..kinds {
            let (a, b) = (self.start[kind], self.start[kind + 1]);
            let mut timing = [0f32; 30];
            let mut far_cells: Vec<u32> = vec![];
            let mut legs: Vec<f32> = vec![];
            let mut visit = |k: usize| {
                let Some((kind, c, d, w, pw)) = self.edge(k) else { return };
                let (near, far) = if end == 0 { (c, d) } else { (d, c) };
                if !selected[near] {
                    return;
                }
                let s = if slot[far] == NONE {
                    slot[far] = far_cells.len() as u32;
                    far_cells.push(far as u32);
                    legs.extend([0.0; 3]);
                    far_cells.len() - 1
                } else {
                    slot[far] as usize
                };
                let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
                for q in 0..5 {
                    if mask >> q & 1 == 0 {
                        continue;
                    }
                    if let Some(st) = &self.per[q] {
                        for direction in 0..2 {
                            let share = if direction == 0 { share_out(day, everyday(kind), q) } else { share_ret(day, everyday(kind), q) };
                            let pr = if pi == NONE { 0.0 } else { st.probability[pi as usize][direction] };
                            let rail = w * share * pr;
                            let walk = (w * share - rail) * pw;
                            let drive = (w * share - walk - rail).max(0.0);
                            for (m, v) in [rail, walk, drive].into_iter().enumerate() {
                                timing[direction * 15 + m * 5 + q] += v;
                                legs[3 * s + m] += v;
                            }
                        }
                    }
                }
            };
            // the samples' fixed end: the home for outings, the site for site-anchored purposes
            let fixed_is_near = (end == 0) != site_anchored(kind);
            if fixed_is_near {
                for &c in &sorted {
                    if let Ok(i) = self.live[kind].binary_search(&(c as u32)) {
                        for k in self.first[kind][i] as usize..self.first[kind][i + 1] as usize {
                            visit(k);
                        }
                    }
                }
            } else {
                for k in a..b {
                    let d = self.destinations[k];
                    if d != NONE && selected[d as usize] {
                        visit(k);
                    }
                }
            }
            // ascending far cells, as the dense layout reads them
            let mut order: Vec<usize> = (0..far_cells.len()).collect();
            order.sort_unstable_by_key(|&i| far_cells[i]);
            let cells_out: Vec<u32> = order.iter().map(|&i| far_cells[i]).collect();
            let legs_out: Vec<f32> = order.iter().flat_map(|&i| [legs[3 * i], legs[3 * i + 1], legs[3 * i + 2]]).collect();
            for &c in &far_cells {
                slot[c as usize] = NONE;
            }
            parts.push((timing, cells_out, legs_out));
        }
        parts
    }
}