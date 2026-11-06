use super::*;

#[cfg(test)]
pub(super) const ST_W: usize = 18;
/// Side of the endpoint sites in metres.
pub(super) const SITE_M: f32 = 600.;
#[cfg(test)]
pub(super) const MARKETS: usize = 3;
/// The bytes of every destination's strategy past which a market's pass searches each destination
/// as its window needs it (`IntercityApi::pass_searched`, T-761, T-1545) instead of on the table
/// it shares with the cities.
pub(super) const MARKET_STRATEGY_BYTES: usize = 256 << 20;
/// T-780: the trips a day each way a sampled zone pair carries (`solved_pairs`).
pub(super) const MARKET_CHUNK: f64 = 1.;
/// T-2641: the chunk grows by this distance's multiples (`solved_pairs`), to `FAR_CHUNK_MAX` times
/// `MARKET_CHUNK`; place pairs from `FAR_DEAL_KM` apart are dealt over both places' zones.
pub(super) const FAR_CHUNK_KM: f64 = 50.;
pub(super) const FAR_CHUNK_MAX: f64 = 16.;
pub(super) const FAR_DEAL_KM: f64 = 250.;
/// A destination's strategy for `IntercityApi::assigned`: a table's, or searched for it alone.
pub(super) enum Held<'a> {
    Ref(Strategy<'a>),
    Own(DestStrategy),
}
impl Held<'_> {
    pub(super) fn view(&self) -> Strategy<'_> {
        match self {
            Held::Ref(s) => *s,
            Held::Own(d) => d.view(),
        }
    }
    /// The bytes it holds of its own.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Held::Ref(_) => 0,
            Held::Own(d) => d.bytes(),
        }
    }
}
pub(super) struct Pass {
    pub(super) loads: Loads,
    pub(super) c_loads: Loads,
    pub(super) r_loads: Option<Loads>,
    pub(super) rail: f64,
    pub(super) c_rail: f64,
    pub(super) c_frac: Vec<f32>,
    pub(super) cor: Vec<f64>,
    pub(super) c_cor: Vec<f64>,
    pub(super) pair_rail: Vec<f32>,
    pub(super) st: Vec<f32>,
    pub(super) xfer: Vec<f32>,
    /// T-1175: the everyday pairs' loads by purpose (`everyday` kind; none where it has no pair)
    pub(super) e_loads: Vec<Option<Loads>>,
    pub(super) e_rail: f64,
    pub(super) e_walk_lost: f64,
    pub(super) e_frac: Vec<f32>,
    pub(super) rail_d: FareSense,
}

thread_local! {
    static CORRIDORS_SEEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
pub(super) fn next_corridors_id() -> u32 {
    CORRIDORS_SEEN.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}
/// The index of pair (a, b), a <= b < n, among all such pairs numbered a then b.
#[inline(always)]
pub(super) fn tri(n: usize, a: usize, b: usize) -> usize {
    a * (2 * n - a + 1) / 2 + (b - a)
}
#[inline(always)]
pub(super) fn chord_km(ua: &[f64; 3], ub: &[f64; 3]) -> f64 {
    let c = ((ua[0] - ub[0]).powi(2) + (ua[1] - ub[1]).powi(2) + (ua[2] - ub[2]).powi(2)).sqrt();
    2. * EARTH_R / 1000. * (c / 2.).min(1.).asin()
}
pub(super) fn unit(lon: f64, lat: f64) -> [f64; 3] {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}
pub(super) fn lonlat_of(u: &[f64; 3]) -> (f64, f64) {
    (u[1].atan2(u[0]).to_degrees(), u[2].atan2(u[0].hypot(u[1])).to_degrees())
}
/// Driving minutes over `d` km.
pub(super) fn drive_min(d: f32) -> f32 {
    15. + d * ROAD_DETOUR / 85. * 60.
}
/// The road is this times the straight line.
pub(super) const ROAD_DETOUR: f32 = 1.25;
/// T-276: what a car costs to run a km, dollars (AAA 2024), where a place's row gives no country
/// figure (T-1521), and the people sharing it on a longer trip (NHTS: about two).
pub(super) const CAR_USD_PER_KM: f32 = 0.15;
pub(super) const LONG_OCCUPANCY: f32 = 2.;
/// T-276: a longer trip's car over `d` km to a zone where parking costs `park` perceived minutes,
/// at `FARE_MIN_PER_USD` (tests).
#[cfg(test)]
pub(super) fn car_min(d: f32, park: f32) -> f32 {
    routed_car(d, park, 1., 1., CAR_USD_PER_KM).at(FARE_MIN_PER_USD)
}
/// A longer trip's car over `d` km to parking `park`, with the rule's minutes and road km times
/// `rt` and `rk` (T-279, `Drive`), the car costing `usd_km` a km of road: parking and the running
/// cost each person's, shared by two.
pub(super) fn routed_car(d: f32, park: f32, rt: f32, rk: f32, usd_km: f32) -> Cost {
    Cost { min: drive_min(d) * rt + park / LONG_OCCUPANCY, usd: d * ROAD_DETOUR * rk * usd_km / LONG_OCCUPANCY }
}
/// A car carried over water by a shuttle or ferry where the landmass's fixed link takes no cars
/// at its own price (T-712): the fare per car in dollars and the minutes beyond the straight drive.
pub(super) struct Crossing {
    pub(super) usd: f32,
    pub(super) min: f32,
}
/// The Channel (Le Shuttle and the Dover ferries) between Great Britain and the continent.
pub(super) const CHANNEL: Crossing = Crossing { usd: 150., min: 60. };
/// The Tsugaru Strait: the Seikan Tunnel carries no cars, the Aomori-Hakodate ferry (3 h 40) does.
pub(super) const TSUGARU: Crossing = Crossing { usd: 170., min: 240. };
/// Great Britain: the UK less Northern Ireland, which is on Ireland's landmass (T-2423).
pub(super) fn great_britain(country: &str, lon: f64, lat: f64) -> bool {
    country == "GB" && !(lon < -5.4 && lat < 55.4)
}
/// The car's cost for the crossings between two places (country, longitude, latitude) on one
/// landmass; the fare is shared like the car's running cost.
pub(super) fn crossing_cost(a: (&str, f64, f64), b: (&str, f64, f64)) -> Cost {
    let hokkaido = |p: (&str, f64, f64)| p.0 == "JP" && p.2 > 41.4;
    let gb = |p: (&str, f64, f64)| great_britain(p.0, p.1, p.2);
    let mut m = Cost::default();
    for (on, c) in [(gb(a) != gb(b), &CHANNEL), (hokkaido(a) != hokkaido(b), &TSUGARU)] {
        if on {
            m = m.plus(Cost { min: c.min, usd: c.usd / LONG_OCCUPANCY });
        }
    }
    m
}
/// T-378 (notes/T-362.md "Modes"), T-712: air door to door, minutes at the airports, cruise speed
/// and the flight's fixed minutes on the great circle, the extra minutes with no direct flight, the
/// fare (a flat base: published average fares barely rise with distance, notes/T-712.md).
/// 90 before departure and 45 from landing to the kerb, both ends together (T-1564, notes/T-1546.md).
pub(super) const AIR_TERMINAL_MIN: f64 = 135.;
pub(super) const AIR_KMH: f64 = 750.;
pub(super) const AIR_FIXED_MIN: f64 = 30.;
pub(super) const AIR_NO_DIRECT_MIN: f64 = 60.;
/// T-1521: published average economy fares at market rates, one world fare (notes/T-1521.md).
pub(super) const AIR_FARE_USD: f64 = 100.;
pub(super) const AIR_USD_PER_KM: f64 = 0.05;
/// T-1430: the wait for a flight, whose frequency is not modelled: rail's planned first wait
/// (`kernel::planned_board_cost`) at a flight every 2 hours.
pub(super) const AIR_WAIT_MIN: f64 = 25.;
/// An airport's direct places sorted, so a pair's flight finds its place by halves (`pair_mode_rule`;
/// the rows and packs write them sorted already).
pub(super) fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}
/// T-1849: every place's airports' direct places as bits, for a pass over millions of pairs
/// (`corridor_rows_modes`), where searching the lists was most of the time; about 2.5 MB for the
/// world's 4,500 places, made for the pass and dropped.
pub(super) struct Directs {
    pub(super) words: usize,
    /// per place its first airport's row, then the rows' count
    pub(super) first: Vec<u32>,
    pub(super) bits: Vec<u64>,
}
impl Directs {
    pub(super) fn of(places: &[Place]) -> Directs {
        let (np, words) = (places.len(), places.len().div_ceil(64));
        let mut first = Vec::with_capacity(np + 1);
        let mut n = 0u32;
        for p in places {
            first.push(n);
            n += p.air.len() as u32;
        }
        first.push(n);
        let mut bits = vec![0u64; n as usize * words];
        for (i, p) in places.iter().enumerate() {
            for (k, (_, d)) in p.air.iter().enumerate() {
                let row = &mut bits[(first[i] as usize + k) * words..][..words];
                // (a place past the market's never matches a pair's)
                for &j in d.iter().filter(|&&j| (j as usize) < np) {
                    row[j as usize / 64] |= 1 << (j % 64);
                }
            }
        }
        Directs { words, first, bits }
    }
    /// Whether place `i`'s airport `k` flies directly to place `j`.
    #[inline]
    pub(super) fn has(&self, i: usize, k: usize, j: usize) -> bool {
        self.bits[(self.first[i] as usize + k) * self.words + j / 64] >> (j % 64) & 1 != 0
    }
}
/// A flight over `km` with `access` minutes of driving at the two ends: its minutes and its fare.
pub(super) fn air_cost(km: f64, access: f64, direct: bool) -> Cost {
    let min = access + AIR_TERMINAL_MIN + AIR_WAIT_MIN + km / AIR_KMH * 60. + AIR_FIXED_MIN + if direct { 0. } else { AIR_NO_DIRECT_MIN };
    Cost { min: min as f32, usd: (AIR_FARE_USD + AIR_USD_PER_KM * km) as f32 }
}

/// T-1373: how far a superholiday's rush leans a pair's longer trips: going out, those from the
/// larger place to the smaller up to `1 + LEAN` times, the other way down to `1 - LEAN` (70/30).
pub const LEAN: f32 = 0.4;
/// T-1373: the most memory the lean's station to station table may take (S x S floats beside the
/// longer trips' own); a larger network's rush days take the longer trips unleaned.
pub(super) const LEAN_OD_BYTES: usize = 256 << 20;
/// T-1374: the rush's further lean towards touristy places: trips into a fully touristy zone from
/// an ordinary one up to `1 + TOUR_LEAN` times going out (and the way back as much less), on top of
/// the places' lean, the two together at most `LEAN_MAX` either way (90/10).
pub const TOUR_LEAN: f32 = 0.5;
pub const LEAN_MAX: f32 = 0.8;
/// T-1374: attraction visits a year per resident at which a zone starts to count as touristy, and
/// at which it is fully so (`touristy`); between them by the logarithm.
pub(super) const TOUR_FROM: f64 = 3.;
pub(super) const TOUR_TO: f64 = 30.;
/// T-1374: how far from an attraction its zone's site may be, in 600 m squares either way.
pub(super) const TOUR_REACH: i32 = 2;

/// T-1374: how touristy a zone or place is, 0 to 1, from its attraction visits a year and people.
pub(super) fn touristy(visits: f64, pop: f64) -> f32 {
    if visits <= 0. {
        return 0.;
    }
    ((visits / pop.max(1.) / TOUR_FROM).ln() / (TOUR_TO / TOUR_FROM).ln()).clamp(0., 1.) as f32
}

impl Place {
    /// T-1373: a place's size as the rush's lean weighs it: its people times the square root of
    /// its people per 6 km zone, so a sparse countryside zone or a small town counts smaller than
    /// its people alone (Anita: home trips by population, with a boost towards rural areas, small
    /// towns and smaller cities).
    pub(super) fn lean_size(&self) -> f64 {
        if self.pop > 0. { self.pop * (self.pop / (self.zones.max(1) as f64)).sqrt() } else { 0. }
    }
}
