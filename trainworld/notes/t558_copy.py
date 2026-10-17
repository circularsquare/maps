"""T-558: a private copy of the sim crate with the walking test (notes/T-558.md).

Like notes/t551_copy.py: the model's code is copied unchanged except for read-only test hooks:
- everyday.rs declares the test module t558.rs and reports each pair's rail costs from `pass`;
- kernel.rs's commute pair loop reports each lane's costs to that module.

    venv\\Scripts\\python.exe trainworld/notes/t558_copy.py <dest folder>

Then in <dest>/sim, with T558_DATA (the tree's data folder), T558_OUT and T558_XFER (the change
penalty, 0 for T-557) set:
    cargo test --release --features demand --lib -- --ignored --nocapture --exact demand::everyday::t558::t558_fit
    cargo test --release --features demand --lib -- --ignored --nocapture --exact demand::everyday::t558::t558
the second with T558_VARS, a JSON list of forms ({"l", "w2", "t0", "ac", "ae"}); then
t558_fit_read.py and t558_tables.py on the outputs. About 1 and 2 minutes on one core.
"""
import pathlib
import shutil
import sys

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim"
NOTE = pathlib.Path(__file__).resolve().parent / "t558.rs"


def patch(path, old, new):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) == 1, f"{path.name}: {s.count(old)} matches for {old[:60]!r}"
    path.write_text(s.replace(old, new), encoding="utf-8")


def main():
    dest = pathlib.Path(sys.argv[1]) / "sim"
    if (dest / "src").exists():
        shutil.rmtree(dest / "src")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SIM / "src", dest / "src")
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(SIM / f, dest / f)
    ev = dest / "src" / "demand" / "everyday.rs"
    with ev.open("a", encoding="utf-8") as f:
        f.write(f'\n#[cfg(test)]\n#[path = "{NOTE.as_posix()}"]\npub(crate) mod t558;\n')
    # everyday pass: each pair's rail cost (car owners) and rail cost of those without a car
    patch(ev, "                    let (mut r, mut r_n, mut lost_q, mut lost_w) = (r_car, 0.0f32, 0.0f32, 0.0f32);\n",
          "                    let (mut r, mut r_n, mut lost_q, mut lost_w) = (r_car, 0.0f32, 0.0f32, 0.0f32);\n                    #[cfg(test)]\n                    let mut t558_rn = Some(rail);\n")
    patch(ev, "                        // T-513: and other (`Params::nocar_choice`)\n",
          "                        #[cfg(test)]\n                        {\n                            t558_rn = rail_n;\n                        }\n                        // T-513: and other (`Params::nocar_choice`)\n")
    patch(ev, "                    let pr = (1.0 - pair.walk) * r;\n",
          "                    let pr = (1.0 - pair.walk) * r;\n                    #[cfg(test)]\n                    super::everyday::t558::hook_ev(i, direction, rail, t558_rn);\n")
    patch(ev, "        let mut walk_lost = vec![0.0f64; kinds];\n",
          "        let mut walk_lost = vec![0.0f64; kinds];\n        #[cfg(test)]\n        super::everyday::t558::hook_ev_start(self.pairs.len());\n")
    k = dest / "src" / "demand" / "kernel.rs"
    patch(k, "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n",
          "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n    #[cfg(test)]\n    let mut dd_zone: Vec<u32> = vec![];\n")
    patch(k, "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n",
          "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n            #[cfg(test)]\n            dd_zone.push(j as u32);\n")
    patch(k, "                let r = if cu > 0.0 {\n                    let mut rail_n = rail;\n",
          "                #[cfg(test)]\n                let mut t558_rn = rail.to_array();\n                let r = if cu > 0.0 {\n                    let mut rail_n = rail;\n")
    patch(k, "                    // `Params::nocar_choice`: walking and other along the drive's road\n",
          "                    #[cfg(test)]\n                    {\n                        t558_rn = rail_n.to_array();\n                    }\n                    // `Params::nocar_choice`: walking and other along the drive's road\n")
    patch(k, "                let bw = ((mo8 + m8) * F8::splat(1.0 / WALKSUM_BAND_MIN)).to_array();\n",
          "                let bw = ((mo8 + m8) * F8::splat(1.0 / WALKSUM_BAND_MIN)).to_array();\n"
          "                #[cfg(test)]\n"
          "                let t558 = ((tu8 * dv8).to_array(), dpw8.to_array(), road.to_array(), paid8.to_array(), rail.to_array(), rd.to_array(), bt.to_array());\n")
    patch(k, "                    rail_sum += xa[l];\n",
          "                    rail_sum += xa[l];\n"
          "                    #[cfg(test)]\n"
          "                    super::everyday::t558::hook_c(i, dd_zone[q[l]] as usize, [t558.0[l], 1.0 - t558.1[l], t558.2[l], t558.3[l], dd[q[l]][2][0], t558.4[l], if okn >> l & 1 != 0 { t558_rn[l] } else { f32::INFINITY }, t558.5[l], t558.6[l], cu, xa[l]]);\n")
    print(f"copied to {dest}")


if __name__ == "__main__":
    main()
