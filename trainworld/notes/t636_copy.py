"""T-636: a private copy of the sim crate with the ceiling test added (notes/T-636.md).

Same two read-only additions as T-551's copy: everyday.rs declares the test module, and kernel.rs's
commute pair loop reports each lane's rail trips by (home zone, work zone) to its hook.

    venv\\Scripts\\python.exe t636_copy.py <trainworld folder> <test file> <dest folder>
"""
import pathlib
import shutil
import sys


def patch(path, old, new):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) == 1, f"{path.name}: {s.count(old)} matches for {old[:60]!r}"
    path.write_text(s.replace(old, new), encoding="utf-8")


def main():
    sim = pathlib.Path(sys.argv[1]) / "sim"
    note = pathlib.Path(sys.argv[2]).resolve()
    dest = pathlib.Path(sys.argv[3]) / "sim"
    if (dest / "src").exists():
        shutil.rmtree(dest / "src")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sim / "src", dest / "src")
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(sim / f, dest / f)
    ev = dest / "src" / "demand" / "everyday.rs"
    with ev.open("a", encoding="utf-8") as f:
        f.write(f'\n#[cfg(test)]\n#[path = "{note.as_posix()}"]\npub(crate) mod t636_ceiling;\n')
    k = dest / "src" / "demand" / "kernel.rs"
    patch(k, "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n",
          "    let mut dd: Vec<[[f32; 4]; 3]> = Vec::with_capacity(sz.zd.len());\n    #[cfg(test)]\n    let mut dd_zone: Vec<u32> = vec![];\n")
    patch(k, "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n",
          "            dd.push([[sz.cx[v], sz.cy[v], local[v].0, local[v].1], [fixed, dv[v], acc_m[v], paid], [free, 0.0, 0.0, 0.0]]);\n            #[cfg(test)]\n            dd_zone.push(j as u32);\n")
    patch(k, "                    rail_sum += xa[l];\n",
          "                    rail_sum += xa[l];\n                    #[cfg(test)]\n                    super::everyday::t636_ceiling::hook(i, dd_zone[q[l]] as usize, xa[l]);\n")
    print(f"copied to {dest}")


if __name__ == "__main__":
    main()
