//! A synthetic New York-sized city, laid out on New York's own coordinates (km from the T-004
//! pack origin, Times Square) so the same hand-made network fits both it and the real pack.
//!
//! Cells are H3-res-9-sized squares (324 m, 0.105 km², the H3 res 9 average area; centre
//! spacing of real res-9 hexes is ~350 m). About 150k cells, 20M people, 10M jobs, a dense core,
//! sub-centres and job centres, decaying outward to 60-90 km. Water roughly where the Atlantic and
//! Long Island Sound are.

use super::pack::City;

/// splitmix64; deterministic and dependency-free.
pub struct Rng(pub u64);

impl Rng {
    pub fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn water(x: f64, y: f64) -> bool {
    // Atlantic and Lower Bay, south of Staten Island / Brooklyn / Long Island's south shore
    if x > -6.0 && y < -18.0 + 0.15 * x.max(0.0) {
        return true;
    }
    // Long Island Sound, between Long Island's north shore and the Westchester/Connecticut shore
    if x > 12.0 && y > 7.0 + 0.07 * (x - 12.0) && y < 16.0 + 0.45 * (x - 12.0) {
        return true;
    }
    x > 95.0
}

/// (x km, y km, peak density per km², e-folding radius km)
const POP_CENTRES: &[(f64, f64, f64, f64)] = &[
    (1.0, 2.0, 30000.0, 4.0),     // Manhattan
    (-0.5, -10.0, 16000.0, 6.0),  // Brooklyn
    (10.0, -3.0, 11000.0, 7.0),   // Queens
    (7.5, 11.0, 15000.0, 4.0),    // Bronx
    (-5.5, -2.0, 11000.0, 3.0),   // Jersey City / Hoboken
    (-15.1, -2.7, 7000.0, 4.0),   // Newark
    (-15.6, 17.7, 6000.0, 3.0),   // Paterson
    (7.2, 19.8, 5000.0, 3.0),     // Yonkers
    (17.7, 30.6, 3000.0, 3.0),    // White Plains
    (37.3, 32.1, 3000.0, 3.5),    // Stamford
    (28.0, -4.0, 4000.0, 7.0),    // Hempstead
    (40.0, 1.0, 3000.0, 7.0),     // Hicksville
    (56.0, -5.0, 2000.0, 9.0),    // Babylon
    (-38.8, -29.0, 3000.0, 4.0),  // New Brunswick
    (-19.4, -10.1, 6000.0, 3.0),  // Elizabeth
    (-12.0, -18.0, 4000.0, 5.0),  // Staten Island
    (-4.9, 14.2, 4000.0, 4.0),    // Hackensack
    (-41.7, 4.3, 1500.0, 6.0),    // Morristown
    (-2.0, -45.0, 1500.0, 8.0),   // Monmouth
];

/// (x km, y km, jobs, e-folding radius km); the rest of the jobs follow population.
const JOB_CENTRES: &[(f64, f64, f64, f64)] = &[
    (0.3, 0.0, 1_500_000.0, 1.1),  // Midtown
    (-2.0, -5.8, 450_000.0, 0.6),  // Lower Manhattan
    (-0.4, -7.3, 110_000.0, 0.7),  // Downtown Brooklyn
    (3.3, -1.2, 70_000.0, 0.8),    // Long Island City
    (-6.0, -3.2, 110_000.0, 0.9),  // Jersey City waterfront
    (-15.1, -2.7, 140_000.0, 1.4), // Newark
    (17.3, -12.5, 40_000.0, 1.0),  // JFK
    (14.8, -6.4, 50_000.0, 1.0),   // Jamaica
    (13.1, 0.1, 50_000.0, 1.0),    // Flushing
    (17.7, 30.6, 100_000.0, 2.0),  // White Plains
    (37.3, 32.1, 100_000.0, 2.0),  // Stamford
    (-4.9, 14.2, 100_000.0, 3.0),  // Hackensack
    (29.0, -1.9, 150_000.0, 4.0),  // Mineola / Garden City
    (65.0, 6.0, 80_000.0, 4.0),    // Hauppauge
    (-38.8, -29.0, 80_000.0, 3.0), // New Brunswick
    (-41.7, 4.3, 80_000.0, 3.0),   // Morristown
];

pub const CELL_M: f64 = 324.0;

pub fn synth_city(seed: u64) -> City {
    const RMAX_KM: f64 = 90.0;
    const POP_TOTAL: f64 = 20.0e6;
    const JOBS_TOTAL: f64 = 10.0e6;
    let mut rng = Rng(seed);
    let area_km2 = (CELL_M / 1000.0).powi(2);
    let n = (2.0 * RMAX_KM * 1000.0 / CELL_M) as i64;
    let (mut xs, mut ys, mut pops, mut jc, mut h3) = (vec![], vec![], vec![], vec![], vec![]);
    for iy in 0..n {
        for ix in 0..n {
            let x = -RMAX_KM + (ix as f64 + 0.5) * CELL_M / 1000.0;
            let y = -RMAX_KM + (iy as f64 + 0.5) * CELL_M / 1000.0;
            let r = (x * x + y * y).sqrt();
            if r > RMAX_KM || water(x, y) {
                continue;
            }
            // parks, cemeteries, airports, farmland: more holes farther out
            if rng.f64() > 0.99 - 0.15 * (r / RMAX_KM) {
                continue;
            }
            let mut dens = 22000.0 * (-r / 8.0).exp() + 250.0 * (-r / 35.0).exp();
            for &(cx, cy, amp, rr) in POP_CENTRES {
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                dens += amp * (-d / rr).exp();
            }
            let pop = dens * area_km2 * (0.5 * rng.normal()).exp();
            let mut jobs_c = 0.0;
            for &(cx, cy, count, rr) in JOB_CENTRES {
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                jobs_c += count / (2.0 * std::f64::consts::PI * rr * rr) * (-d / rr).exp() * area_km2;
            }
            xs.push((x * 1000.0) as f32);
            ys.push((y * 1000.0) as f32);
            pops.push(pop);
            jc.push(jobs_c * (0.4 * rng.normal()).exp());
            h3.push((iy * n + ix) as u64); // dummy ids, sorted like the real pack
        }
    }
    let psum: f64 = pops.iter().sum();
    let pop: Vec<f32> = pops.iter().map(|&p| (p * POP_TOTAL / psum) as f32).collect();
    // local jobs (shops, schools, services) follow population, the centres sit on top
    let centre_sum: f64 = jc.iter().sum();
    let local: Vec<f64> = pop.iter().map(|&p| (p as f64).powf(0.9) * (0.6 * rng.normal()).exp()).collect();
    let lsum: f64 = local.iter().sum();
    let lscale = (JOBS_TOTAL - centre_sum) / lsum;
    let jobs: Vec<f32> = jc.iter().zip(&local).map(|(&c, &l)| (c + l * lscale) as f32).collect();
    City { name: "synthetic-ny".into(), h3, x: xs, y: ys, pop, jobs, gravity: None }
}
