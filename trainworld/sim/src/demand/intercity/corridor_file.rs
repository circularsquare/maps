//! T-4199: the market's corridors and corridor book, worked out once over the world's rows and drives
//! (`MarketPacks::corridor_file`, run in the demand WASM by `tools/corridor_file.mjs`, so the same bits as a
//! load's) and read at load (`MarketPacks::set_corridors`) in place of working them out in every worker.
//! The file is keyed by the world it was made from (rows, rule, frame, the fare figures `marketWorld` adds),
//! the drives and the sim's sources; a load checks that, and a sample of corridors' trips and modes, and works
//! everything out as before on any difference. Layout (little-endian): "TWC1", version, the three keys, the
//! place count, the sample count, `daily`, `place_pairs`, the book's row count and key, then `leaving_by`
//! (a f64 a place), the book's sums (`IntercityApi::book_rows`, five f64 a place) and the samples (places a
//! and b, the trips as `Table::trips_of` gives them, the shares flying and driving with no zone on two lands).

use super::*;
use std::cell::Cell;

const MAGIC: &[u8; 4] = b"TWC1";
const VERSION: u32 = 1;
/// the sim's sources (`build.rs`): a file made by another build is never read
const SRC_HASH: &str = env!("TW_SIM_SRC_HASH");
/// the corridors with trips whose trips and modes a load works out and checks against the file's
const SAMPLES: usize = 1024;

/// What became of the corridor file given (`MarketPacks::corridors_state`).
pub const FILE_NONE: u32 = 0;
pub const FILE_USED: u32 = 1;
/// made from other rows, rule or drives
pub const FILE_OTHER_WORLD: u32 = 2;
/// made by another build of the sim
pub const FILE_OTHER_BUILD: u32 = 3;
pub const FILE_UNREADABLE: u32 = 4;
/// a sample's trips or modes are not this build's
pub const FILE_DIFFERS: u32 = 5;

/// A key over bytes, a word at a time (FNV's step, rotated so every bit mixes): what the corridor file
/// is keyed by, not a checksum anything else reads.
pub(super) fn hash64(b: &[u8]) -> u64 {
    let step = |h: u64, w: u64| (h ^ w).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut words = b.chunks_exact(8);
    for w in &mut words {
        h = step(h, u64::from_le_bytes(w.try_into().unwrap()));
    }
    for &x in words.remainder() {
        h = step(h, x as u64);
    }
    step(h, b.len() as u64)
}
/// The key of a world as `MarketPacks::new` takes it. Its attractions (the last key, `game/cities.ts`
/// `marketWorld`) are left out: no corridor figure reads them, and a worker may compose before it has them.
pub(super) fn world_key(world: &str) -> u64 {
    hash64(world.rfind(",\"attractions\":").map_or(world, |i| &world[..i]).as_bytes())
}
pub(super) fn src_key() -> u64 {
    hash64(SRC_HASH.as_bytes())
}

/// What `Market::corridor_table` reads of each place, worked out once a place (and the income ratio's
/// power once a pair of incomes, a few hundred incomes against millions of pairs), so the table's loop and
/// a corridor worked out alone (`Market::trips_alone`) give the same bits.
pub(super) struct TripPrep {
    /// the border factor's general part, as `pair_trip_rule`'s
    general: Vec<f64>,
    /// by country (ids name one country each): in the open area; its residents see no pull
    open_area: Vec<bool>,
    off: Vec<bool>,
    /// T-3240: a place's pull as a destination from its own country and from abroad, and its unfaded local pull
    pull_home: Vec<f64>,
    pull_abroad: Vec<f64>,
    pull_near_home: Vec<f64>,
    inc_id: Vec<u32>,
    rels: Vec<f64>,
    ni: usize,
}
impl TripPrep {
    pub(super) fn of(m: &Market) -> TripPrep {
        let rule = &m.rule;
        let general: Vec<f64> = m.places.iter().map(|p| rule.border * p.inc.powf(rule.border_eps)).collect();
        let open_area: Vec<bool> = m.countries.iter().map(|c| OPEN_AREA.contains(&c.as_str())).collect();
        let home_gamma: Vec<f64> = m.countries.iter().map(|c| rule.pull_gamma_of(c, c)).collect();
        let near_gamma: Vec<f64> = m.countries.iter().map(|c| rule.pull_near_gamma_of(c, c)).collect();
        let off: Vec<bool> = m.countries.iter().map(|c| rule.pull_off.contains(c)).collect();
        let pull_of = |q: &Place, gamma: f64| if q.px > 0. { (gamma * q.px).exp_m1() } else { 0. };
        let pull_home: Vec<f64> = m.places.iter().map(|q| if q.has { pull_of(q, home_gamma[q.country as usize]) } else { 0. }).collect();
        let pull_abroad: Vec<f64> = m.places.iter().map(|q| if q.has { pull_of(q, rule.pull_gamma_x) } else { 0. }).collect();
        let pull_near_home: Vec<f64> = m.places.iter().map(|q| if q.has && near_gamma[q.country as usize] > 0. { pull_of(q, near_gamma[q.country as usize]) } else { 0. }).collect();
        let mut incs: Vec<f64> = m.places.iter().filter(|p| p.has).map(|p| p.inc).collect();
        incs.sort_by(f64::total_cmp);
        incs.dedup_by(|a, b| a.to_bits() == b.to_bits());
        let inc_id: Vec<u32> = m.places.iter().map(|p| if p.has { incs.binary_search_by(|x| x.total_cmp(&p.inc)).map_or(u32::MAX, |k| k as u32) } else { u32::MAX }).collect();
        let ni = incs.len();
        let rels: Vec<f64> = (0..ni * ni)
            .map(|k| {
                let (pi, qi) = (incs[k / ni], incs[k % ni]);
                // a ratio of 1 or more is held at 1, and 1 to any power is 1
                if pi > 0. {
                    let x = (qi / pi).min(1.);
                    if x == 1. { 1. } else { x.powf(rule.eps_t) }
                } else {
                    1.
                }
            })
            .collect();
        TripPrep { general, open_area, off, pull_home, pull_abroad, pull_near_home, inc_id, rels, ni }
    }
    /// `pair_trip_rule`'s factors from place i to place j: border, income ratio, pull, local pull.
    #[inline(always)]
    fn factors(&self, m: &Market, i: usize, j: usize) -> (f64, f64, f64, f64) {
        let (p, q) = (&m.places[i], &m.places[j]);
        let same = p.country == q.country;
        let b = if same {
            1.
        } else if self.open_area[p.country as usize] && self.open_area[q.country as usize] {
            OPEN_AREA_BORDER.max(self.general[i])
        } else {
            self.general[i]
        };
        let rel = self.rels[self.inc_id[i] as usize * self.ni + self.inc_id[j] as usize];
        let pull = if q.px > 0. && !self.off[p.country as usize] { if same { self.pull_home[j] } else { self.pull_abroad[j] } } else { 0. };
        let pull_near = if same { self.pull_near_home[j] } else { 0. };
        (b, rel, pull, pull_near)
    }
    /// Places i <= j (both with zones): their longer trips a day each way, as `place_eachway` gives them.
    #[inline(always)]
    pub(super) fn trip(&self, m: &Market, i: usize, j: usize) -> f64 {
        let (p, q, rule) = (&m.places[i], &m.places[j], &m.rule);
        let km = if i == j {
            0.
        } else if m.ground {
            chord_km(&p.u, &q.u)
        } else {
            ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
        };
        if km < rule.near_km {
            m.near.get(&(i as u32, j as u32)).copied().unwrap_or(0.)
        } else {
            let ((bi, ri, ui, ni), (bj, rj, uj, nj)) = (self.factors(m, i, j), self.factors(m, j, i));
            // the curves at the pair's distance once for both directions (`directed_by`'s, to the bit)
            let curves = (rule.ground(km), rule.tail(km), rule.pull_on(km));
            (Market::directed_with(bi, ri, ui, ni, p.g, p.t, q.a, q.ta, curves) + Market::directed_with(bj, rj, uj, nj, q.g, q.t, p.a, p.ta, curves)) * 0.5
        }
    }
}

/// The corridor book as the file holds it: `IntercityApi::table_summary`'s row count, its key before
/// the own rows' shares on two lands are added to it, and its sums (five a place).
#[derive(Clone)]
pub(super) struct Book {
    pub(super) n: usize,
    pub(super) key: u32,
    pub(super) sums: Vec<f64>,
}
struct Sample {
    a: u32,
    b: u32,
    trips: f64,
    mode: [f32; 2],
}
pub(super) struct CorridorFile {
    world: u64,
    drives: u64,
    src: u64,
    daily: f64,
    place_pairs: usize,
    leaving_by: Vec<f64>,
    book: Book,
    samples: Vec<Sample>,
}
impl CorridorFile {
    fn write(&self) -> Vec<u8> {
        let np = self.leaving_by.len();
        let mut o = Vec::with_capacity(72 + 48 * np + 24 * self.samples.len());
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&VERSION.to_le_bytes());
        for k in [self.world, self.drives, self.src] {
            o.extend_from_slice(&k.to_le_bytes());
        }
        o.extend_from_slice(&(np as u32).to_le_bytes());
        o.extend_from_slice(&(self.samples.len() as u32).to_le_bytes());
        o.extend_from_slice(&self.daily.to_le_bytes());
        o.extend_from_slice(&(self.place_pairs as u64).to_le_bytes());
        o.extend_from_slice(&(self.book.n as u64).to_le_bytes());
        o.extend_from_slice(&self.book.key.to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        for v in self.leaving_by.iter().chain(&self.book.sums) {
            o.extend_from_slice(&v.to_le_bytes());
        }
        for s in &self.samples {
            o.extend_from_slice(&s.a.to_le_bytes());
            o.extend_from_slice(&s.b.to_le_bytes());
            o.extend_from_slice(&s.trips.to_le_bytes());
            o.extend_from_slice(&s.mode[0].to_le_bytes());
            o.extend_from_slice(&s.mode[1].to_le_bytes());
        }
        o
    }
    pub(super) fn parse(b: &[u8]) -> Result<CorridorFile, String> {
        let u32_at = |k: usize| b.get(k..k + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
        let u64_at = |k: usize| b.get(k..k + 8).map(|x| u64::from_le_bytes(x.try_into().unwrap()));
        if b.get(..4) != Some(&MAGIC[..]) || u32_at(4) != Some(VERSION) {
            return Err("not a corridor file of this version".into());
        }
        let (np, ns) = (u32_at(32).unwrap_or(0) as usize, u32_at(36).unwrap_or(0) as usize);
        if b.len() != 72 + 48 * np + 24 * ns {
            return Err("corridor file of the wrong length".into());
        }
        let f64_at = |k: usize| f64::from_bits(u64_at(k).unwrap());
        let floats = |k: usize, n: usize| (0..n).map(|i| f64_at(k + 8 * i)).collect::<Vec<f64>>();
        let s0 = 72 + 48 * np;
        Ok(CorridorFile {
            world: u64_at(8).unwrap(),
            drives: u64_at(16).unwrap(),
            src: u64_at(24).unwrap(),
            daily: f64_at(40),
            place_pairs: u64_at(48).unwrap() as usize,
            book: Book { n: u64_at(56).unwrap() as usize, key: u32_at(64).unwrap(), sums: floats(72 + 8 * np, 5 * np) },
            leaving_by: floats(72, np),
            samples: (0..ns)
                .map(|i| {
                    let k = s0 + 24 * i;
                    let f32_at = |k: usize| f32::from_bits(u32_at(k).unwrap());
                    Sample { a: u32_at(k).unwrap(), b: u32_at(k + 4).unwrap(), trips: f64_at(k + 8), mode: [f32_at(k + 16), f32_at(k + 20)] }
                })
                .collect(),
        })
    }
}

impl Market {
    /// T-4199: the corridors from the file, the commutes from the rows (`commute_rows`) as
    /// `corridors_of` takes them; None when a sample's trips or modes are not this market's.
    pub(super) fn corridors_from(&self, f: &CorridorFile, commute_rows: &[Vec<(u32, f64)>]) -> Option<Corridors> {
        let np = self.places.len();
        if f.leaving_by.len() != np || f.book.sums.len() != 5 * np {
            return None;
        }
        let prep = TripPrep::of(self);
        let directs = Directs::of(&self.places);
        for s in &f.samples {
            let (a, b) = (s.a as usize, s.b as usize);
            if a > b || b >= np {
                return None;
            }
            let t = if self.places[a].has && self.places[b].has { prep.trip(self, a, b) } else { 0. };
            let t = if t > 0. { (2. * t) as f32 as f64 } else { 0. };
            let mode = self.corridor_mode_with(&directs, a, b, false);
            if t.to_bits() != s.trips.to_bits() || mode.map(f32::to_bits) != s.mode.map(f32::to_bits) {
                return None;
            }
        }
        let (commute, commuters) = self.corridor_commutes(Some(commute_rows));
        let leaving_by = f.leaving_by.clone();
        Some(Corridors { id: next_corridors_id(), daily: f.daily, leaving: leaving_by.iter().sum(), table: Default::default(), book: Some(f.book.clone()), prep: prep.into(), commute, commuters, leaving_by, place_pairs: f.place_pairs, rows: Default::default() })
    }
}

/// What `MarketPacks` keeps for its corridor file.
#[derive(Default)]
pub(super) struct FileHeld {
    /// the keys of the world and drives given
    pub(super) world: u64,
    pub(super) drives: u64,
    /// (kept once used, for a composition after `forget_corridors`; dropped once refused)
    pub(super) file: std::cell::RefCell<Option<Rc<CorridorFile>>>,
    pub(super) state: Cell<u32>,
}
impl FileHeld {
    /// The file, if it was made from this world, drives (with `drives`) and build; else none, the state saying
    /// which differs, and the file dropped.
    pub(super) fn usable(&self, drives: bool) -> Option<Rc<CorridorFile>> {
        let f = self.file.borrow().clone()?;
        let why = if f.src != src_key() {
            FILE_OTHER_BUILD
        } else if f.world != self.world || (drives && f.drives != self.drives) {
            FILE_OTHER_WORLD
        } else {
            return Some(f);
        };
        self.refuse(why);
        None
    }
    pub(super) fn refuse(&self, why: u32) {
        self.state.set(why);
        *self.file.borrow_mut() = None;
    }
}

#[wasm_bindgen]
impl MarketPacks {
    /// T-4199: the market's corridor file (`app/src/game/market-corridors.bin`), read by every composition
    /// after (none composed yet: the first). What became of it: `corridors_state`.
    pub fn set_corridors(&mut self, bytes: &[u8]) -> u32 {
        match CorridorFile::parse(bytes) {
            Ok(f) => {
                *self.held.file.borrow_mut() = Some(Rc::new(f));
                self.held.state.set(FILE_NONE);
                // (one of another world or build is not kept; the drives may yet be given, so they are checked
                // at the composition)
                self.held.usable(false);
            }
            Err(_) => self.held.refuse(FILE_UNREADABLE),
        }
        self.held.state.get()
    }
    /// T-4199: what became of the corridor file at the last composition: `FILE_USED`, `FILE_NONE` (none
    /// given, or the corridors were kept from an earlier composition), else why it was not used.
    pub fn corridors_state(&self) -> u32 {
        self.held.state.get()
    }
    /// T-4199: the corridor file of this world and drives, for this build: the corridors and the book
    /// worked out with no zone open, and checked: every corridor's trips worked out alone are the table's.
    pub fn corridor_file(&self) -> Result<Vec<u8>, String> {
        let w = self.world.clone();
        if w.rows.is_none() {
            return Err("a corridor file needs the world's rows".into());
        }
        let b = Built { zones: vec![], rule: w.rule, out: Some(w.frame), places: w.places, rows: w.rows, world: true, closed: false, legacy_commutes: vec![], drives: self.drives.clone(), attractions: vec![] };
        let api = IntercityApi::on(Rc::new(build_for(b, None, None, None)?));
        let m = &*api.m;
        if !m.apart().cross.is_empty() || !m.apart().own.is_empty() {
            return Err("zones on two lands with no zone open".into());
        }
        let np = m.places.len();
        let mut out = vec![0f64; 2 + 6 * np];
        let (n, key) = api.book_rows(np, &mut out);
        let table = m.table();
        for (c, (a, b)) in m.each_corridor().enumerate() {
            let (alone, held) = (m.trips_alone(a, b), table.trips_of(c));
            if alone.to_bits() != held.to_bits() {
                return Err(format!("corridor {c} ({a}, {b}): {alone} worked out alone, {held} in the table"));
            }
        }
        let directs = Directs::of(&m.places);
        let step = (table.trips.len() / SAMPLES).max(1);
        let samples = table
            .with_trips()
            .step_by(step)
            .take(SAMPLES)
            .map(|(c, t)| {
                let (a, b) = m.corridor_ends(c);
                Sample { a: a as u32, b: b as u32, trips: t, mode: m.corridor_mode_with(&directs, a, b, false) }
            })
            .collect();
        let cw = &m.cw;
        let f = CorridorFile { world: self.held.world, drives: self.held.drives, src: src_key(), daily: cw.daily, place_pairs: cw.place_pairs, leaving_by: cw.leaving_by.clone(), book: Book { n, key, sums: out[2..2 + 5 * np].to_vec() }, samples };
        Ok(f.write())
    }
}
