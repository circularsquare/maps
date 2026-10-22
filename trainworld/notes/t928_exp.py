"""T-928's check of the incremental join: the full joint solve against the incremental one, as dry
runs of bin/joint_gravity (my tree's build) on copies of the packs; nothing in the main tree is written.

    python exp.py hel     the hel-fi join of batch 20261022-114119 (poj new), from its backup
    python exp.py osl     the osl-no join of batch 20261022-142205 (ves new), from its backup
    python exp.py cn X    the Jiangsu unit (39 packs) with X taken out as the "before", then added
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCR = Path(r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\2e31d80c-814f-4eb1-abc9-7e758da09685\scratchpad\t928")
MAPS = Path(r"C:\Users\anita\projects\maps")
MAIN = MAPS / "trainworld"
TREE = MAPS / ".tw-trees" / "t928" / "trainworld"
WR = MAPS / "artifacts" / "trainworld" / "world_run"
BIN = SCR / "target" / "release" / "joint_gravity.exe"
sys.path.insert(0, str(TREE / "pipeline"))
import intercity  # noqa: E402
from read_pack import load_pack  # noqa: E402

HOLD_TOL = float(os.environ.get("HOLD_TOL", "1e-4"))


def setup(case, sources):
    """A scratch root: the main registry, borders and plan, and copies of the packs in `sources`
    ({id: folder holding id.json and id.bin}); every boundary from the main packs."""
    root = SCR / "exp" / case
    for d in ("app/src/game", "pipeline", "data/packs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAIN / "app/src/game/registry.json", root / "app/src/game/registry.json")
    shutil.copy2(MAIN / "pipeline/borders.json", root / "pipeline/borders.json")
    shutil.copy2(MAIN / "pipeline/world_plan.json", root / "pipeline/world_plan.json")
    for i, src in sources.items():
        for ext in (".json", ".bin"):
            dst = root / "data/packs" / f"{i}{ext}"
            if not dst.exists() or dst.stat().st_size != (Path(src) / f"{i}{ext}").stat().st_size:
                shutil.copy2(Path(src) / f"{i}{ext}", dst)
        b = root / "data/packs" / f"{i}.boundary.geojson"
        if not b.exists():
            shutil.copy2(MAIN / "data/packs" / f"{i}.boundary.geojson", b)
    return root


def solve(root, packs, fixed=(), new=(), init=None, tag="x"):
    args = [str(BIN), "--root", str(root), "--threads", "4", "--max-iters", "150", "--packs", ",".join(packs),
            "--dry", "--dump", str(root / f"{tag}.dump.json"), "--report", str(root / f"{tag}.report.json")]
    if fixed:
        args += ["--fixed", ",".join(fixed), "--hold-report", str(root / f"{tag}.hold.json")]
        if new:
            args += ["--new", ",".join(new)]
    if init:
        args += ["--init", str(init)]
    t0 = time.time()
    key = root / f"{tag}.args"
    if key.exists() and key.read_text() == " ".join(args) and (root / f"{tag}.dump.json").exists():
        secs = float((root / f"{tag}.secs").read_text())  # the same run as before: its outputs
    else:
        with open(root / f"{tag}.log", "w", encoding="utf-8") as log:
            p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="4"))
        secs = time.time() - t0
        if p.returncode:
            raise SystemExit(f"{tag}: joint_gravity failed, see {root / (tag + '.log')}")
        key.write_text(" ".join(args))
        (root / f"{tag}.secs").write_text(str(secs))
    d = json.loads((root / f"{tag}.dump.json").read_text())
    hold = json.loads((root / f"{tag}.hold.json").read_text()) if fixed else None
    rep = json.loads((root / f"{tag}.report.json").read_text())
    print(f"  {tag}: {len(packs)} solved, {len(fixed)} fixed, {d['iterations']} steps, error {rep['max_error']:.1e}, {secs:.0f} s")
    return {"dump": d, "hold": hold, "report": rep, "secs": secs}


def per_pack(dump, key):
    """{id: that array's slice} for the packs of a dump."""
    a = dump[key]
    return {p[0]: np.array(a[p[1]:p[1] + p[2]], np.float64) for p in dump["packs"]}


def rel(a, b, floor=1.0):
    """Max relative difference over zones where either is at least `floor`, and |a - b| summed over b summed."""
    m = (np.abs(a) >= floor) | (np.abs(b) >= floor)
    if not m.any():
        return 0.0, 0.0
    d = np.abs(a - b)[m] / np.maximum(np.maximum(np.abs(a), np.abs(b))[m], 1e-300)
    return float(d.max()), float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300))


def pack_flows(folder, i):
    h, a = load_pack(i, str(folder))
    return {"home": np.asarray(a["zone_home"], np.float64), "work": np.asarray(a["zone_work"], np.float64),
            "row": np.asarray(a["zone_row"], np.float64), "col": np.asarray(a["zone_col"], np.float64)}


def by_pack(rep):
    ids, m = rep["packs"], np.array(rep["by_pack"], np.float64)
    n = len(ids)
    m = m.reshape(n, n)
    return {(ids[a], ids[b]): m[a, b] for a in range(n) for b in range(n) if a != b}


def reach_split(unit, new, root, km=intercity.CUT_KM):
    """The tried split: the new packs and every member within km of one solved, the rest held."""
    members = unit["members"]
    shapes = {i: intercity.outline(root / f"data/packs/{i}.boundary.geojson") for i in members}
    solved = [i for i in members if i in new or intercity.near_any(shapes[i], [shapes[j] for j in new if j in shapes], km)]
    return solved, [i for i in members if i not in solved]


def incremental(root, members, fixed, new, init=None, tag="inc", full=None, before=None, prev_err=0.0):
    """The incremental solve as the runner does it: reach_split, then widen by the hold report.
    With `full` and `before`, each round is compared with the full solve."""
    unit = {"members": members, "fixed": fixed}
    solved, held = reach_split(unit, new, root)
    rounds = []
    for k in range(4):
        r = solve(root, solved, held + list(fixed), new, init, tag=f"{tag}{k}")
        if full is not None:
            compare(full, r, solved, held, before)
        tol = max(HOLD_TOL, prev_err)
        over = [f for f in r["hold"]["fixed"] if f["id"] in held and max(f["row_delta"], f["col_delta"]) > tol]
        rounds.append({"solved": list(solved), "held": list(held), "over": [(f["id"], f["row_delta"], f["col_delta"]) for f in over], "secs": r["secs"]})
        print(f"    round {k}: solved {' '.join(solved)}; held {' '.join(held)}; over {tol:.1e}: "
              + (", ".join(f"{f['id']} {max(f['row_delta'], f['col_delta']):.1e}" for f in over) or "none"))
        if not over:
            break
        solved = [i for i in members if i in solved or i in {f["id"] for f in over}]
        held = [i for i in members if i not in solved]
    return r, solved, held, rounds


def compare(full, inc, solved, held, before):
    """Solved packs: incremental against full. Held packs: their old flows (what the incremental
    join leaves) against the full solve's."""
    out = {"solved": {}, "held": {}}
    fh, fw = per_pack(full["dump"], "home"), per_pack(full["dump"], "work")
    ih, iw = per_pack(inc["dump"], "home"), per_pack(inc["dump"], "work")
    for i in solved:
        out["solved"][i] = {"home": rel(ih[i], fh[i]), "work": rel(iw[i], fw[i])}
    for i in held:
        out["held"][i] = {"home": rel(before[i]["home"], fh[i]), "work": rel(before[i]["work"], fw[i])}
    for k in ("solved", "held"):
        for i, v in out[k].items():
            print(f"    {k:6} {i}: home max {v['home'][0]:.2e} sum {v['home'][1]:.2e}; work max {v['work'][0]:.2e} sum {v['work'][1]:.2e}")
    # commuters between packs: pairs with an end solved come from the incremental file, the rest
    # from the held packs' earlier solve (the before)
    fb, ib = by_pack(full["report"]), by_pack(inc["report"])
    worst = []
    for (a, b), v in fb.items():
        if a in solved or b in solved:
            w = ib.get((a, b), 0.0)
            if max(v, w) >= 1.0:
                worst.append((abs(v - w) / max(v, w), a, b, v, w))
    worst.sort(reverse=True)
    out["pairs"] = worst[:5]
    for e in worst[:5]:
        print(f"    pair {e[1]}->{e[2]}: full {e[3]:.1f}, incremental {e[4]:.1f} ({e[0]:.2e})")
    return out


def real_case(case, batch, group, members, fixed, new):
    bak = WR / batch / f"join.group-{group}-backup"
    src = {i: bak / "data/work/res9" for i in members}  # the join solved the res-9 packs
    src.update({i: MAIN / "data/packs" for i in fixed})
    root = setup(case, src)
    print(f"{case}: members {' '.join(members)} (new {' '.join(new)}), fixed {' '.join(fixed)}")
    full = solve(root, members, fixed, new, tag="full")
    # sanity: the dry full solve gives what the real join wrote (the merged packs carry the res-9 gravity)
    now = {i: pack_flows(MAIN / "data/packs", i) for i in members}
    fh = per_pack(full["dump"], "home")
    for i in members:
        print(f"    full dry run against the real join's {i}: home max {rel(fh[i], now[i]['home'])[0]:.1e}")
    before = {i: pack_flows(bak / "data/work/res9", i) for i in members if i not in new}
    prev = max(json.loads((bak / "data/work/res9" / f"{i}.json").read_text(encoding="utf-8"))["gravity"].get("max_error", 0) for i in before)
    print(f"    the unit's last solve: error {prev:.1e}")
    inc, solved, held, rounds = incremental(root, members, fixed, new, full=full, before=before, prev_err=prev)
    res = compare(full, inc, solved, held, before)
    res.update({"rounds": rounds, "full_secs": full["secs"]})
    (SCR / "exp" / f"{case}.result.json").write_text(json.dumps(res, indent=1, default=str))


def cn_case(x):
    rep = json.loads((MAIN / "data/work/sha.joint_report.json").read_text())
    members = rep["solved"]
    root = setup(f"cn-{x}", {i: MAIN / "data/work/res9" if (MAIN / "data/work/res9" / f"{i}.json").exists() else MAIN / "data/packs" for i in members})
    print(f"cn: {len(members)} members, {x} as the new one")
    rest = [i for i in members if i != x]
    before = solve(root, rest, tag="before")
    full = solve(root, members, tag="full")
    bh, bw = per_pack(before["dump"], "home"), per_pack(before["dump"], "work")
    bef = {i: {"home": bh[i], "work": bw[i]} for i in rest}
    inc, solved, held, rounds = incremental(root, members, [], [x], init=root / "before.dump.json", full=full, before=bef, prev_err=before["report"]["max_error"])
    res = compare(full, inc, solved, held, bef)
    res.update({"rounds": rounds, "full_secs": full["secs"]})
    (SCR / "exp" / f"cn-{x}.result.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "hel":
        real_case("hel", "20261022-114119", "hel-fi", ["hel", "vri", "aan", "pik", "poj"], ["sai", "sto", "pos", "kid", "ups"], ["poj"])
    elif which == "osl":
        real_case("osl", "20261022-142205", "osl-no", ["osl", "rog", "stf", "vef", "ves"], ["noj", "vag", "osx"], ["ves"])
    elif which == "gb":
        u = json.loads((WR / "20261022-144605/join.groups.log").read_text(encoding="utf-8").split("RESULT ", 1)[1])[0]
        real_case("gb", "20261022-144605", "lon-gb", u["members"], u["fixed"], ["hih"])
    elif which == "cn":
        cn_case(sys.argv[2])
