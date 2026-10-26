//! T-1435, where people drive: car trips a day between squares of a zone's grid, for the map's
//! drives view (docs/ui-demand-views.md "Where people drive"). The commutes' and the everyday
//! layer's car legs per pair of squares come in two parts, so a new demand result costs only the
//! second: `drive_base_day`, the legs with no rail and their train gain (no network in it, worked
//! out once per zone and day by the app), and `drive_rail_day`, the car legs rail takes over a
//! worker's periods. Display only: demand is untouched.
use super::*;

/// Squares hold at most this many: the pair tables are `n (n - 1) / 2` long.
const MAX_SQUARES: usize = 800;
/// Squares' sides in zones (2 km squares), the smallest that keeps under `MAX_SQUARES`.
const SIDES: [i32; 12] = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96];

/// The straight-line train the gain is measured against (DECISIONS 2026-10-26, T-1435): minutes
/// to reach a station, wait and leave the far one, and its speed over the straight line.
pub(crate) const TRAIN_ACCESS_MIN: f32 = 15.0;
pub(crate) const TRAIN_KMH: f32 = 100.0;

/// Minutes a car trip of `car_min` over `d_km` straight-line km would save by that train, never
/// below 0.
#[inline]
pub(crate) fn train_gain(car_min: f32, d_km: f32) -> f32 {
    (car_min - TRAIN_ACCESS_MIN - d_km * 60.0 / TRAIN_KMH).max(0.0)
}

/// The zone grid in squares of `side` zones: each zone's square, and per square its people's and
/// jobs' centre (pack metres) and its place in the coarser grid.
pub(crate) struct DriveSquares {
    pub side: i32,
    pub zone_m: f32,
    pub n: usize,
    pub of_zone: Vec<u32>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub sx: Vec<i32>,
    pub sy: Vec<i32>,
}

impl DriveSquares {
    pub fn of(z: &Zones, city: &pack::City) -> DriveSquares {
        let mut side = SIDES[SIDES.len() - 1];
        for s in SIDES {
            let mut seen = std::collections::HashSet::new();
            for i in 0..z.n {
                seen.insert((z.gx[i].div_euclid(s), z.gy[i].div_euclid(s)));
                if seen.len() > MAX_SQUARES {
                    break;
                }
            }
            if seen.len() <= MAX_SQUARES {
                side = s;
                break;
            }
        }
        let mut at = std::collections::HashMap::new();
        let (mut sx, mut sy) = (vec![], vec![]);
        let of_zone: Vec<u32> = (0..z.n)
            .map(|i| {
                let k = (z.gx[i].div_euclid(side), z.gy[i].div_euclid(side));
                *at.entry(k).or_insert_with(|| {
                    sx.push(k.0);
                    sy.push(k.1);
                    sx.len() as u32 - 1
                })
            })
            .collect();
        let n = sx.len();
        let (mut wx, mut wy, mut w) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
        for c in 0..city.len() {
            let s = of_zone[z.cell_zone[c] as usize] as usize;
            let v = (city.pop[c] + city.jobs[c]) as f64;
            wx[s] += v * city.x[c] as f64;
            wy[s] += v * city.y[c] as f64;
            w[s] += v;
        }
        let half = z.zone_m as f64 * side as f64 * 0.5;
        let (x, y) = (0..n)
            .map(|s| {
                if w[s] > 0.0 {
                    ((wx[s] / w[s]) as f32, (wy[s] / w[s]) as f32)
                } else {
                    ((z.x0 as f64 + sx[s] as f64 * 2.0 * half + half) as f32, (z.y0 as f64 + sy[s] as f64 * 2.0 * half + half) as f32)
                }
            })
            .unzip();
        DriveSquares { side, zone_m: z.zone_m, n, of_zone, x, y, sx, sy }
    }

    /// The unordered pair of squares a and b (a != b): its index in a table of `n (n - 1) / 2`.
    #[inline]
    pub fn pair(&self, a: u32, b: u32) -> usize {
        let (i, j) = if a < b { (a as usize, b as usize) } else { (b as usize, a as usize) };
        i * self.n - i * (i + 1) / 2 + (j - i - 1)
    }

    pub fn pairs(&self) -> usize {
        self.n * self.n.saturating_sub(1) / 2
    }

    /// `[side, zone metres, n, x.., y.., sx.., sy..]`, the app's view of the squares.
    pub fn packed(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(3 + 4 * self.n);
        out.extend([self.side as f32, self.zone_m, self.n as f32]);
        out.extend(&self.x);
        out.extend(&self.y);
        out.extend(self.sx.iter().map(|&v| v as f32));
        out.extend(self.sy.iter().map(|&v| v as f32));
        out
    }
}

/// A pair table's nonzero entries as `[m, index.., value..]` (indices exact in f32 below 2^24).
pub(crate) fn sparse(t: &[f32]) -> Vec<f32> {
    let at: Vec<usize> = (0..t.len()).filter(|&k| t[k] != 0.0).collect();
    let mut out = Vec::with_capacity(1 + 2 * at.len());
    out.push(at.len() as f32);
    out.extend(at.iter().map(|&k| k as f32));
    out.extend(at.iter().map(|&k| t[k]));
    out
}

#[wasm_bindgen]
impl DemandApi {
    /// T-1435: the squares the drive tables are by (`DriveSquares::packed`).
    pub fn drive_squares(&self) -> Vec<f32> {
        DriveSquares::of(&self.z, &self.city).packed()
    }

    /// T-1435: the commutes on `day` by unordered pair of squares, three blocks of
    /// `DriveSquares::pairs`: trip legs, those by car with no rail (the zone pair's walking with no
    /// rail left out), and those car legs times the minutes each would save by the straight-line
    /// train (`train_gain`, the drive priced as the model prices it between the zones' centres).
    /// Needs no network; pairs within a square are left out.
    pub fn drive_base_day(&self, day: u32) -> Vec<f32> {
        let sq = DriveSquares::of(&self.z, &self.city);
        let np = sq.pairs();
        let mut out = vec![0f64; 3 * np];
        let (z, gr, p) = (&*self.z, &*self.gr, &self.base);
        let rate = day_rate(day as usize, WORK) as f64;
        let place: Vec<(f32, f32)> = z.density().iter().map(|&d| p.place_min_per_km(d)).collect();
        for i in 0..z.n {
            let si = sq.of_zone[i];
            let (bi, ri) = (gr.base(z, i), gr.row[i] * rate);
            if ri <= 0.0 {
                continue;
            }
            for j in 0..z.n {
                let sj = sq.of_zone[j];
                if si == sj {
                    continue;
                }
                let idx = (bi + gr.key(z, j)) as usize;
                // both legs of each commuter
                let legs = 2.0 * ri * gr.col[j] * (gr.ftab[idx] * gr.k(z, i, j)) as f64;
                // a zone pair under a hundredth of a leg a day is left out: a few legs a day at most
                // over all the zone pairs of one pair of squares
                if !(legs > 0.01) {
                    continue;
                }
                let walk = if gr.walk_set && gr.walk_c(idx) < NO_WALK { self.nr.split(gr, p, i, j, idx) as f64 } else { 0.0 };
                let car = legs * (1.0 - walk);
                let d_km = gr.dist_km(idx);
                let car_min = p.drive_min_f(d_km * 1000.0, p.drive_factor(i, j), place[i], place[j]);
                let k = sq.pair(si, sj);
                out[k] += legs;
                out[np + k] += car;
                out[2 * np + k] += car * train_gain(car_min, d_km) as f64;
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    /// T-1435: the commutes' rail legs on `day` over this worker's solved periods by unordered pair
    /// of squares (`sparse`), through the station pairs as `flow_rail` reads them: each station's
    /// boarders spread over the squares they come from, its alighters over the squares they go to.
    /// Exact in total, blurred per pair. `[0]` with no network.
    pub fn drive_rail_day(&self, day: u32) -> Vec<f32> {
        let (Some(sz), Some(svc)) = (self.sz.as_ref(), self.svc.as_ref()) else { return vec![0.0] };
        let sq = DriveSquares::of(&self.z, &self.city);
        let np = sq.pairs();
        let mut r = vec![0f32; np];
        let s = svc.st_x.len();
        let k = self.od_st.len();
        let sub_sq: Vec<u32> = (0..sz.n).map(|u| sq.of_zone[sz.zone[u] as usize]).collect();
        let mut row = vec![0f32; sq.n];
        let mut touched: Vec<u32> = vec![];
        for q in 0..PERIODS {
            let Some(st) = self.period(q) else { continue };
            let sub = &st.sub;
            if sub.acc_home.len() != sz.acc_st.len() || sub.od.len() != k * k {
                continue;
            }
            let f = load_factor(day as usize, WORK, q);
            // per station, its boarders' squares (home end) and its alighters' (work end), shares
            let (mut dep, mut arr) = (vec![0f32; s], vec![0f32; s]);
            let mut home: Vec<Vec<(u32, f32)>> = vec![vec![]; s];
            let mut work: Vec<Vec<(u32, f32)>> = vec![vec![]; s];
            let add = |list: &mut Vec<(u32, f32)>, sq: u32, v: f32| match list.iter_mut().find(|e| e.0 == sq) {
                Some(e) => e.1 += v,
                None => list.push((sq, v)),
            };
            for u in 0..sz.n {
                for e in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                    let t = sz.acc_st[e] as usize;
                    dep[t] += sub.acc_home[e];
                    arr[t] += sub.acc_work[e];
                    if sub.acc_home[e] > 0.0 {
                        add(&mut home[t], sub_sq[u], sub.acc_home[e]);
                    }
                    if sub.acc_work[e] > 0.0 {
                        add(&mut work[t], sub_sq[u], sub.acc_work[e]);
                    }
                }
            }
            for ai in 0..k {
                let a = self.od_st[ai] as usize;
                if home[a].is_empty() || dep[a] <= 0.0 {
                    continue;
                }
                for (ti, &x) in sub.od[ai * k..(ai + 1) * k].iter().enumerate() {
                    let t = self.od_st[ti] as usize;
                    if x <= 0.0 || arr[t] <= 0.0 {
                        continue;
                    }
                    for &(j, w) in &work[t] {
                        if row[j as usize] == 0.0 {
                            touched.push(j);
                        }
                        row[j as usize] += x * w / arr[t];
                    }
                }
                for &(i, h) in &home[a] {
                    let h = h / dep[a] * f;
                    for &j in &touched {
                        if i != j {
                            r[sq.pair(i, j)] += h * row[j as usize];
                        }
                    }
                }
                for &j in &touched {
                    row[j as usize] = 0.0;
                }
                touched.clear();
            }
        }
        sparse(&r)
    }
}

/// The real New York pack and network: the tables' sizes and timings, the commutes' rail never
/// above their legs, and the biggest pairs by car legs times gain.
/// `cargo test --release --features demand drives_real_network -- --ignored --nocapture`
#[cfg(test)]
#[test]
#[ignore]
fn drives_real_network() {
    use crate::demand::everyday::EverydayApi;
    use crate::demand::network::{route_nodes, RealNet};
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let (h, b) = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let mut a = DemandApi::new(&h, &b).unwrap();
    let mut e = EverydayApi::beside(&a, &h, &b).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let (_, _, hop_km) = route_nodes(&st_xy, &line_n, &stops);
    a.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    a.set_fares(2.9, 0.1, &hop_km).unwrap();
    e.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    e.set_fares(2.9, 0.1, &hop_km).unwrap();
    a.solve(0);
    e.solve(0);
    let sq = DriveSquares::of(&a.z, &a.city);
    let np = sq.pairs();
    println!("squares of {} zones: {}, pairs {np}", sq.side, sq.n);
    let t = now_ms();
    let base = a.drive_base_day(0);
    let t1 = now_ms();
    let rail = a.drive_rail_day(0);
    let t2 = now_ms();
    let ebase = e.drive_base_day(0);
    let t3 = now_ms();
    let erail = e.drive_rail_day(0);
    let t4 = now_ms();
    println!("commutes base {:.0} ms, rail {:.0} ms ({} pairs); everyday base {:.0} ms, rail {:.0} ms ({} pairs)", t1 - t, t2 - t1, rail[0], t3 - t2, t4 - t3, erail[0]);
    let m = rail[0] as usize;
    let mut r = vec![0f32; np];
    for k in 0..m {
        r[rail[1 + k] as usize] = rail[1 + m + k];
    }
    let me = erail[0] as usize;
    let mut er = vec![0f32; np];
    for k in 0..me {
        er[erail[1 + k] as usize] = erail[1 + me + k];
    }
    let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
    println!("commute legs {:.0}, car {:.0}, rail (period 0) {:.0}; everyday car {:.0}, car rail takes (period 0) {:.0}", sum(&base[..np]), sum(&base[np..2 * np]), sum(&r), sum(&ebase[..np]), sum(&er));
    let mut top: Vec<(usize, f32, f32)> = (0..np)
        .map(|k| {
            let (tl, c0, cg) = (base[k], base[np + k], base[2 * np + k]);
            let car = (c0 - if tl > 0.0 { r[k] * c0 / tl } else { 0.0 }).max(0.0) + (ebase[k] - er[k]).max(0.0);
            let g = if c0 + ebase[k] > 0.0 { (cg + ebase[np + k]) / (c0 + ebase[k]) } else { 0.0 };
            (k, car, g)
        })
        .collect();
    top.sort_by(|x, y| (y.1 * y.2).total_cmp(&(x.1 * x.2)));
    for &(k, car, g) in top.iter().take(12) {
        let i = (0..sq.n).find(|&i| k < (i + 1) * sq.n - (i + 1) * (i + 2) / 2).unwrap();
        let j = k - (i * sq.n - i * (i + 1) / 2) + i + 1;
        let d = ((sq.x[i] - sq.x[j]).powi(2) + (sq.y[i] - sq.y[j]).powi(2)).sqrt() / 1000.0;
        println!("  ({:.1}, {:.1}) km - ({:.1}, {:.1}) km, {d:.1} km apart: {car:.0} car legs, {g:.1} min faster", sq.x[i] / 1000.0, sq.y[i] / 1000.0, sq.x[j] / 1000.0, sq.y[j] / 1000.0);
    }
}
