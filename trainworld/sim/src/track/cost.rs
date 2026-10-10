//! Build costs (SPEC 6.4) and the water mask interface (T-030 will supply the real mask).

use super::params::*;

/// Where the water is. T-030 ships a mask in the city pack; anything that can answer "is this
/// point water?" in local metres works here.
pub trait WaterMask {
    fn is_water(&self, x: f64, y: f64) -> bool;
    /// True when there is no water anywhere (lets the model skip sampling).
    fn is_empty(&self) -> bool {
        false
    }
}

/// No water at all.
pub struct NoWater;
impl WaterMask for NoWater {
    fn is_water(&self, _: f64, _: f64) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        true
    }
}

impl<F: Fn(f64, f64) -> bool> WaterMask for F {
    fn is_water(&self, x: f64, y: f64) -> bool {
        self(x, y)
    }
}

/// The city pack's water mask (notes/T-004.md "Water mask"): a grid of `cell` metre pixels from
/// the south-west corner (`x0`, `y0`), `w` x `h`, stored as row runs. Row r's entries are
/// `xs[row[r] .. row[r+1]]`, pairs (start, end): columns start..end-1 are water. Outside the grid
/// is land.
pub struct RunMask {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub w: u32,
    pub h: u32,
    pub row: Vec<u32>,
    pub xs: Vec<u16>,
}

impl RunMask {
    /// From the pack header's `water` block and the two arrays' bytes (little-endian).
    pub fn from_pack(cell: f64, origin: [f64; 2], size: [u32; 2], row_bytes: &[u8], x_bytes: &[u8]) -> RunMask {
        let row = row_bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let xs = x_bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        RunMask { x0: origin[0], y0: origin[1], cell, w: size[0], h: size[1], row, xs }
    }
}

impl WaterMask for RunMask {
    fn is_water(&self, x: f64, y: f64) -> bool {
        let c = ((x - self.x0) / self.cell).floor();
        let r = ((y - self.y0) / self.cell).floor();
        if c < 0.0 || r < 0.0 || c >= self.w as f64 || r >= self.h as f64 {
            return false;
        }
        let r = r as usize;
        let row = &self.xs[self.row[r] as usize..self.row[r + 1] as usize];
        row.partition_point(|&v| (v as f64) <= c) % 2 == 1
    }
}

/// Chainage spans `[s0, s1]` over water along an alignment, sampled every `WATER_STEP` m.
pub fn water_spans(len: f64, at: impl Fn(f64) -> (f64, f64), mask: &dyn WaterMask, out: &mut Vec<[f64; 2]>) {
    if mask.is_empty() || len <= 0.0 {
        return;
    }
    let n = (len / WATER_STEP).ceil().max(1.0) as usize;
    let step = len / n as f64;
    let mut open: Option<f64> = None;
    for j in 0..n {
        let (x, y) = at((j as f64 + 0.5) * step);
        let w = mask.is_water(x, y);
        match (w, open) {
            (true, None) => open = Some(j as f64 * step),
            (false, Some(s0)) => {
                out.push([s0, j as f64 * step]);
                open = None;
            }
            _ => {}
        }
    }
    if let Some(s0) = open {
        out.push([s0, len]);
    }
}

/// Level of a height, rounded.
pub fn level_of(z: f64) -> i8 {
    (z / LEVEL_H).round() as i8
}

/// Cost of one edge's track, US$M: per sub-interval, the level's multiplier (a ramp at the dearer
/// of its two levels), 2x over water, 0.6x for single track.
pub fn track_cost(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8) -> f64 {
    let mut c = 0.0;
    track_cost_parts(len, vert, water, tracks, |_, _, _, _, cost| c += cost);
    c
}

/// Visit each priced interval: charged level, water, ramp, metres, US$M.
/// Shared by the price and its explanation, including the dearer end of a ramp.
pub fn track_cost_parts(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8, mut part: impl FnMut(i8, bool, bool, f64, f64)) {
    let mut cuts: Vec<f64> = Vec::with_capacity(vert.len() + 2 * water.len() + 2);
    cuts.push(0.0);
    cuts.push(len);
    cuts.extend(vert.iter().map(|p| p[0]));
    for w in water {
        cuts.push(w[0]);
        cuts.push(w[1]);
    }
    cuts.sort_by(|a, b| a.total_cmp(b));
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let track_f = if tracks == 1 { SINGLE_TRACK_COST } else { 1.0 };
    for w in cuts.windows(2) {
        let (u, v) = (w[0].max(0.0), w[1].min(len));
        if v <= u {
            continue;
        }
        let m = 0.5 * (u + v);
        let i = vert.partition_point(|p| p[0] <= m).clamp(1, vert.len().max(2) - 1);
        let (level, ramp) = if vert.len() < 2 {
            (level_of(vert.first().map_or(0.0, |p| p[1])), false)
        } else {
            let a = level_of(vert[i - 1][1]);
            let b = level_of(vert[i][1]);
            (if level_mult(a) >= level_mult(b) { a } else { b }, (vert[i - 1][1] - vert[i][1]).abs() > 1e-9)
        };
        let wet = water.iter().any(|s| m >= s[0] && m < s[1]);
        let c = (v - u) / 1000.0 * BASE_COST_PER_KM * level_mult(level) * if wet { WATER_COST } else { 1.0 } * track_f;
        part(level, wet, ramp, v - u, c);
    }
}

/// Station cost, US$M: the 200 m price for its level scaled by `0.4 + 0.6 x length / 200`.
pub fn station_cost(level: i8, platform: u16, tracks: u8, wet: bool) -> f64 {
    let base = STATION_COST_200[(level.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize];
    base * (0.4 + 0.6 * platform as f64 / 200.0)
        * if tracks == 1 { SINGLE_TRACK_STATION } else { 1.0 }
        * if wet { WATER_COST } else { 1.0 }
}

/// Junction switches: 0.25 km of track at its level; a flying junction adds 0.6 km one level away
/// from the ground (toward it at +3 and -3).
pub fn junction_cost(level: i8, flying: bool, wet: bool) -> f64 {
    let w = if wet { WATER_COST } else { 1.0 };
    let mut c = JUNCTION_KM * BASE_COST_PER_KM * level_mult(level);
    if flying {
        let away = match level {
            MAX_LEVEL => MAX_LEVEL - 1,
            MIN_LEVEL => MIN_LEVEL + 1,
            l if l >= 0 => l + 1,
            l => l - 1,
        };
        c += FLYOVER_KM * BASE_COST_PER_KM * level_mult(away);
    }
    c * w
}

/// A flat crossing: 0.1 km of track at its level.
pub fn crossing_cost(level: i8, wet: bool) -> f64 {
    FLAT_CROSSING_KM * BASE_COST_PER_KM * level_mult(level) * if wet { WATER_COST } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_worked_example_20km_at_minus2_with_18_stations() {
        // SPEC 6.4: a 20 km line at -2 with 18 stations is about 3.8B.
        let vert = [[0.0, -16.0], [20000.0, -16.0]];
        let track = track_cost(20000.0, &vert, &[], 2);
        assert!((track - 1980.0).abs() < 1e-9);
        let stations = 18.0 * station_cost(-2, 200, 2, false);
        assert!((stations - 1800.0).abs() < 1e-9);
        let total = track + stations;
        assert!((total - 3780.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn multipliers_ramps_water_single() {
        let flat0 = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat0, &[], 2) - 27.0).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat0, &[], 1) - 16.2).abs() < 1e-9);
        // +1 viaduct over 500 m of water: 72 x 0.5 + 144 x 0.5
        let v1 = [[0.0, 8.0], [1000.0, 8.0]];
        assert!((track_cost(1000.0, &v1, &[[250.0, 750.0]], 2) - 108.0).abs() < 1e-9);
        // 0 -> -2 ramp over 400 m in a 1000 m edge: 300 m at 0.3, 400 m at 1.1 (dearer), 300 m at 1.1.
        let ramp = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let want = 90.0 * (0.3 * 0.3 + 0.4 * 1.1 + 0.3 * 1.1);
        assert!((track_cost(1000.0, &ramp, &[], 2) - want).abs() < 1e-9);
        // Ramp +1 -> -1 is priced at the dearer end level (-1: 1.0), not at 0.
        let r2 = [[0.0, 8.0], [400.0, -8.0]];
        assert!((track_cost(400.0, &r2, &[], 2) - 36.0).abs() < 1e-9);
    }

    #[test]
    fn stations_junctions_crossings() {
        assert!((station_cost(0, 200, 2, false) - 10.0).abs() < 1e-12);
        assert!((station_cost(-2, 100, 2, false) - 70.0).abs() < 1e-9);
        assert!((station_cost(-2, 400, 2, false) - 160.0).abs() < 1e-9);
        assert!((junction_cost(0, false, false) - 0.25 * 27.0).abs() < 1e-9);
        assert!((junction_cost(0, true, false) - (6.75 + 0.6 * 72.0)).abs() < 1e-9);
        assert!((junction_cost(-2, true, false) - (0.25 * 99.0 + 0.6 * 108.0)).abs() < 1e-9);
        assert!((crossing_cost(0, false) - 2.7).abs() < 1e-9);
    }

    #[test]
    fn breakdown_splits_water_ramps_and_single_track() {
        let vert = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let mut rows = vec![];
        track_cost_parts(1000.0, &vert, &[[500.0, 900.0]], 1, |l, wet, ramp, m, c| rows.push((l, wet, ramp, m, c)));
        assert_eq!(rows.iter().map(|r| r.3).sum::<f64>(), 1000.0);
        assert_eq!(rows.iter().filter(|r| r.1).map(|r| r.3).sum::<f64>(), 400.0);
        assert_eq!(rows.iter().filter(|r| r.2).map(|r| r.3).sum::<f64>(), 400.0);
        assert!(rows.iter().filter(|r| r.2).all(|r| r.0 == -2));
        assert!((rows.iter().map(|r| r.4).sum::<f64>() - 70.2).abs() < 1e-9);
    }

    #[test]
    fn water_spans_sampled() {
        let mask = |x: f64, _y: f64| (400.0..600.0).contains(&x);
        let mut out = vec![];
        water_spans(1000.0, |s| (s, 0.0), &mask, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0][0] - 400.0).abs() <= 25.0 && (out[0][1] - 600.0).abs() <= 25.0);
        // Row 0: columns 1..3 water; row 1: none.
        let r = RunMask { x0: 0.0, y0: 0.0, cell: 50.0, w: 4, h: 2, row: vec![0, 2, 2], xs: vec![1, 3] };
        assert!(!r.is_water(10.0, 10.0) && r.is_water(60.0, 10.0) && r.is_water(149.0, 10.0));
        assert!(!r.is_water(151.0, 10.0) && !r.is_water(60.0, 60.0) && !r.is_water(-1.0, 0.0));
    }
}
