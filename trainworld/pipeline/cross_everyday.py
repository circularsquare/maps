"""T-1163 (T-1175): everyday outings across zone lines, per joint solve unit (`bin/cross_everyday.rs`).

    python pipeline/cross_everyday.py --group auto [ids]   every solve unit (intercity.solve_units), or
                                                            those holding the ids
    python pipeline/cross_everyday.py --solve a,b [--fixed c,d] --name n
    ... --root DIR    a copy's registry and packs (and outputs in its data/work)
    ... --dry         the unit files only (<name>.cross_everyday.dry.json), no pack written

intercity.py runs it after each unit's joint gravity, so a join makes both. Each unit's members are
drawn from (their zones' kept shares and the pairs their draws make), its fixed packs only drawn on.
Writes data/work/<name>.cross_everyday.json (market_zones.py takes each pack's pairs from the newest
file that drew from it) and each member's kept shares into its pack and its res-9 copy as per-zone
arrays `everyday_kept_<purpose>` (f32; the everyday layer multiplies a zone's budget by them).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import packio
import simbin

ROOT = Path(__file__).resolve().parents[1]
KEPT = "everyday_kept_"


def run_unit(members, fixed, name, root=ROOT, work=None, dry=False):
    """One solve unit's draw, then its members' kept shares into their packs (not with `dry`). A
    file whose drawn packs a newer one draws from too is removed."""
    work = Path(work or root / "data/work")
    # a dry run's file is not one market_zones.py reads
    out = work / f"{name}.cross_everyday{'.dry' if dry else ''}.json"
    simbin.run(ROOT, "cross_everyday", ["--root", str(root), "--packs-dir", str(root / "data/packs"),
               "--threads", os.environ.get("OMP_NUM_THREADS", "2"), "--packs", ",".join(members),
               "--fixed", ",".join(fixed), "--out", str(out)])
    if dry:
        return out
    write_kept(out, root)
    files = unit_files(work)
    for p, f in files.items():
        if any(q != p and g["t"] > f["t"] and f["solved"] <= g["solved"] for q, g in files.items()):
            print(f"removed {p.name}: a newer draw covers every pack it drew from")
            p.unlink()
    return out


def unit_files(work):
    """Every unit file as {path: {solved, t}}, oldest first."""
    files = {}
    for p in sorted(Path(work).glob("*.cross_everyday.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        files[p] = {"solved": set(g["solved"]), "t": float(g.get("solved_at") or p.stat().st_mtime)}
    return dict(sorted(files.items(), key=lambda kv: kv[1]["t"]))


def kept_arrays(k, kinds):
    """A pack's kept shares from a unit file's entry: {array name: one value per zone}."""
    nz = int(k["zones"])
    out = {KEPT + kind: np.ones(nz, np.float32) for kind in kinds}
    zone = np.asarray(k["zone"], np.int64)
    if len(zone):
        v = np.asarray(k["kept"], np.float32)
        for i, kind in enumerate(kinds):
            out[KEPT + kind][zone] = v[:, i]
    return out


def write_kept(path, root=ROOT):
    """Every drawn pack's kept shares from unit file `path` into the pack, and its res-9 copy where
    there is one (as everyday_rates.py writes its block), keeping merge_cells' stamp in step."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    packs, res9 = root / "data/packs", root / "data/work/res9"
    import merge_cells as mc
    for cid, k in g["kept"].items():
        arrays = kept_arrays(k, g["kinds"])
        stamped = root == ROOT and mc.written(cid) == mc.stamp(str(packs), cid)
        for d in (packs, res9):
            stem = str(d / cid)
            if not os.path.exists(stem + ".json"):
                continue
            with open(stem + ".json", encoding="utf-8") as f:
                h = json.load(f)
            if h.get("zones") != k["zones"]:
                print(f"{cid}: {d.name} holds {h.get('zones')} zones, the draw {k['zones']}: not written")
                continue
            with open(stem + ".bin", "rb") as f:
                buf = f.read()
            # a pack keeping everything carries none (the everyday layer's default)
            keep = {n: v for n, v in arrays.items() if (v < 1).any()}
            gone = [a["name"] for a in h["arrays"] if a["name"].startswith(KEPT) and a["name"] not in keep]
            if not keep and not gone:
                continue
            h["arrays"] = [a for a in h["arrays"] if a["name"] not in gone]
            out = packio.replace_cell_arrays(h, buf, keep, per="zone")
            packio.write(stem, h, bytes(out))
        if stamped:
            with open(res9 / f"{cid}.merged.json", "w") as f:
                json.dump(mc.stamp(str(packs), cid), f)
        print(f"{cid}: kept shares for {len(k['zone'])} of {k['zones']} zones", flush=True)


def main(argv):
    root = Path(argv[argv.index("--root") + 1]).resolve() if "--root" in argv else ROOT
    dry = "--dry" in argv
    arg = lambda k: argv[argv.index(k) + 1] if k in argv else ""  # noqa: E731
    if "--solve" in argv:
        run_unit([i for i in arg("--solve").split(",") if i], [i for i in arg("--fixed").split(",") if i],
                 arg("--name"), root, dry=dry)
        return
    if "--group" in argv:
        import intercity
        rest = argv[argv.index("--group") + 2:]
        only = [a for k, a in enumerate(rest) if not a.startswith("--") and (k == 0 or rest[k - 1] != "--root")]
        reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
        for u in intercity.solve_units(reg, root, only):
            run_unit(u["members"], u["fixed"], u["name"], root, dry=dry)
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
