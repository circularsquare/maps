//! T-3553: what the water rule does to the real New York save. Needs the real save and the nyc pack, so it
//! runs only on request, on the save's track bytes (gunzipped, after its JSON header; notes/T-3553.md):
//! `T3553_TRACK=<file> cargo test --release --manifest-path sim/Cargo.toml t3553_real_save -- --ignored --nocapture`
use super::*;

#[test]
#[ignore]
fn t3553_real_save_water_cost() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let bin = std::fs::read(format!("{root}/data/packs/nyc.bin")).unwrap();
    // offsets from data/packs/nyc.json: water_row u32 x 11648 at 2224584, water_x u16 x 532802 at 2271176
    let rows = &bin[2224584..2224584 + 11648 * 4];
    let xs = &bin[2271176..2271176 + 532802 * 2];
    let mask = Arc::new(RunMask::from_pack(25.0, [-103925.0, -142050.0], [11410, 11647], rows, xs));
    let bytes = std::fs::read(std::env::var("T3553_TRACK").expect("T3553_TRACK: the save's track bytes")).unwrap();
    let mut net = save::decode(&bytes).unwrap();
    let (seen, b) = pack_mask_in(net.frame(), &mask, -73.985, 40.758);
    net.set_water_mask(Box::new(MaskSet { rasters: vec![(seen, b)], world: None }));
    let w = TrackWorld::new(net);
    let (mut wet, mut n, mut edges) = (0.0, 0, 0);
    for e in 0..w.net.edge_count() {
        if !w.net.edge_ok(e as u32) { continue; }
        edges += 1;
        let s = w.net.edge_water(e as u32);
        let m: f64 = s.iter().map(|p| p[1] - p[0]).sum();
        if m > 0.0 { n += 1; }
        wet += m;
    }
    println!("T-3553 real save: {edges} edges, {n} with water, wet {wet:.0} m, built cost {:.0}", w.net.built_cost());
}

/// A pack's mask read at points from a file, for comparing `RunMask::is_water` with another implementation
/// (notes/T-3553.md). The file: a line `bin cell x0 y0 w h row_offset row_count x_offset x_count`, then a line `x y`
/// (pack metres) per point. Writes `<file>.out`, a 0/1 per point.
#[test]
#[ignore]
fn t3553_mask_at_points() {
    let path = std::env::var("T3553_POINTS").expect("T3553_POINTS: the points file");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let h: Vec<&str> = lines.next().unwrap().split(' ').collect();
    let n = |i: usize| h[i].parse::<f64>().unwrap();
    let bin = std::fs::read(h[0]).unwrap();
    let (ro, rc, xo, xc) = (n(6) as usize, n(7) as usize, n(8) as usize, n(9) as usize);
    let mask = RunMask::from_pack(n(1), [n(2), n(3)], [n(4) as u32, n(5) as u32], &bin[ro..ro + rc * 4], &bin[xo..xo + xc * 2]);
    let out: String = lines.map(|l| {
        let mut it = l.split(' ').map(|v| v.parse::<f64>().unwrap());
        let (x, y) = (it.next().unwrap(), it.next().unwrap());
        if mask.is_water(x, y) { "1\n" } else { "0\n" }
    }).collect();
    std::fs::write(format!("{path}.out"), out).unwrap();
}
