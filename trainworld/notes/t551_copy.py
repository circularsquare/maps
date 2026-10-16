"""T-551: a private copy of the sim crate with the ceiling test added (notes/T-551.md).

The model's own code is copied unchanged except for two read-only additions, so the shared crate
is never touched:
- everyday.rs declares the test module `notes/t551_ceiling.rs` (test builds only);
- kernel.rs's commute pair loop reports each lane's rail trips by (home zone, work zone) to that
  module's hook (test builds only), so the real network's result can be split by distance.

    venv\\Scripts\\python.exe trainworld/notes/t551_copy.py <dest folder>
"""
import pathlib
import shutil
import sys

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim"
NOTE = pathlib.Path(__file__).resolve().parent / "t551_ceiling.rs"


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
        f.write(f'\n#[cfg(test)]\n#[path = "{NOTE.as_posix()}"]\npub(crate) mod t551_ceiling;\n')
    k = dest / "src" / "demand" / "kernel.rs"
    # each destination subzone's zone, in the pair loop's order
    patch(k, "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n",
          "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n    #[cfg(test)]\n    let mut dd_zone: Vec<u32> = vec![];\n")
    patch(k, "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n",
          "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n            #[cfg(test)]\n            dd_zone.push(j as u32);\n")
    patch(k, "                    rail_sum += xa[l];\n",
          "                    rail_sum += xa[l];\n                    #[cfg(test)]\n                    super::everyday::t551_ceiling::hook(i, dd_zone[q[l]] as usize, xa[l]);\n")
    print(f"copied to {dest}")


if __name__ == "__main__":
    main()
