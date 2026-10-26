//! T-1435: the everyday layer's car legs by pair of squares, as `api/drives.rs` gives the
//! commutes': exact per sample, since each sample's pair keeps its own rail chance.
use super::*;
use crate::demand::api::drives::{sparse, train_gain, DriveSquares};

#[wasm_bindgen]
impl EverydayApi {
    /// T-1435: every purpose's legs on `day` by unordered pair of squares, two blocks of
    /// `DriveSquares::pairs`: car legs with no rail, and those times the minutes each would save by
    /// the straight-line train (`train_gain`, the drive between the sample's two cells as the layer
    /// prices it). Needs no network; pairs within a square are left out.
    pub fn drive_base_day(&self, day: u32) -> Vec<f32> {
        let (z, city) = (&*self.engine.z, &*self.engine.city);
        let sq = DriveSquares::of(z, city);
        let np = sq.pairs();
        let mut out = vec![0f64; 2 * np];
        let day = day as usize;
        for k in 0..self.destinations.len() {
            let Some((kind, c, d, w, pw)) = self.edge(k) else { continue };
            let (zc, zd) = (z.cell_zone[c] as usize, z.cell_zone[d] as usize);
            let (sc, sd) = (sq.of_zone[zc], sq.of_zone[zd]);
            if sc == sd {
                continue;
            }
            let p = everyday(kind);
            let shares: f32 = (0..PERIODS).map(|q| share_out(day, p, q) + share_ret(day, p, q)).sum();
            let car = (w * shares * (1.0 - pw)) as f64;
            if !(car > 0.0) {
                continue;
            }
            let dist = self.distance(c, d);
            let car_min = self.ep.drive_min_f(dist, self.ep.drive_factor(zc, zd), self.wplace[zc], self.wplace[zd]);
            let at = sq.pair(sc, sd);
            out[at] += car;
            out[np + at] += car * train_gain(car_min, dist * 0.001) as f64;
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    /// T-1435: the car legs rail takes on `day` over this worker's solved periods, by unordered pair
    /// of squares (`sparse`): each sample's rail legs times its share not walking with no rail (rail
    /// takes from walking and the car in proportion, T-558).
    pub fn drive_rail_day(&self, day: u32) -> Vec<f32> {
        let (z, city) = (&*self.engine.z, &*self.engine.city);
        let sq = DriveSquares::of(z, city);
        let mut r = vec![0f32; sq.pairs()];
        let day = day as usize;
        let held: Vec<(usize, &Period)> = self.per.iter().enumerate().filter_map(|(q, s)| s.as_ref().map(|s| (q, s))).collect();
        if held.is_empty() {
            return vec![0.0];
        }
        for k in 0..self.destinations.len() {
            let pi = self.edge_pair.get(k).copied().unwrap_or(NONE);
            if pi == NONE {
                continue;
            }
            let Some((kind, c, d, per)) = self.ends(k) else { continue };
            let (sc, sd) = (sq.of_zone[z.cell_zone[c] as usize], sq.of_zone[z.cell_zone[d] as usize]);
            if sc == sd {
                continue;
            }
            let p = everyday(kind);
            let mut chance = 0.0;
            for &(q, st) in &held {
                let pr = st.probability[pi as usize];
                chance += share_out(day, p, q) * pr[0] + share_ret(day, p, q) * pr[1];
            }
            // the sample's weight only where rail takes some of it (the weight is the dear part)
            if chance > 0.0 {
                let (w, pw) = self.weight(kind, per, c, d, self.distance(c, d));
                r[sq.pair(sc, sd)] += w * chance * (1.0 - pw);
            }
        }
        sparse(&r)
    }
}
