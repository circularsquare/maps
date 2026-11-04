"""T-929's check: market_zones' rows worked out only where a change reaches, against every row worked
out again (`--full`), on scratch copies through world_run's `--root` redirect (nothing in the main
tree is written). Paths point at T-929's scratchpad: `base` there is a copy of the main tree's
registry, market rows and market packs.

    python -I t929_check.py seed | fresh <ids> | add | remove | register | newer <state>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path(r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\2e31d80c-814f-4eb1-abc9-7e758da09685\scratchpad\t929")
TREE = Path(r"C:\Users\anita\projects\maps\.tw-trees\t929\trainworld")
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
GAME = "app/src/game"


def clone(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run(root, *args):
    env = dict(os.environ, TW_RUN_ROOT=str(root), TW_SIM_DIR=str(TREE / "sim"),
               TW_CARGO_TARGET_DIR=str(TREE / "sim/target"), OMP_NUM_THREADS="3", CARGO_BUILD_JOBS="3",
               PYTHONIOENCODING="utf-8")
    t = time.time()
    log = root.parent / f"{root.name}.{'_'.join(a.strip('-') for a in args)}.log"
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([PY, "-I", str(TREE / "pipeline/world_run.py"), "step", "market_zones", *args],
                           env=env, stdout=f, stderr=subprocess.STDOUT)
    s = time.time() - t
    tail = [l for l in log.read_text(encoding="utf-8").splitlines() if l.startswith("market rows:")]
    print(f"  {root.name} {' '.join(args)}: exit {p.returncode}, {s:.1f} s {tail}", flush=True)
    if p.returncode:
        print(log.read_text(encoding="utf-8")[-3000:])
        sys.exit(1)
    return s


def same(a, b):
    out = True
    for f in ("market-rows.json", "registry.json"):
        x, y = (a / GAME / f).read_bytes(), (b / GAME / f).read_bytes()
        if x != y:
            out = False
            print(f"  DIFFER {f}: {a.name} {len(x)} bytes, {b.name} {len(y)} bytes")
            if f == "market-rows.json":
                rx, ry = json.loads(x), json.loads(y)
                bad = [k for k in set(rx) | set(ry) if rx.get(k) != ry.get(k)]
                print(f"    {len(bad)} rows differ: {sorted(bad)[:20]}")
    print(f"  {a.name} vs {b.name}: {'IDENTICAL' if out else 'DIFFERENT'}", flush=True)
    return out


def drop(root, ids):
    """As world_run's unregister: the entries, their rows and market packs go."""
    reg = json.loads((root / GAME / "registry.json").read_text(encoding="utf-8"))
    gone = [c for c in reg["cities"] if c["id"] in ids]
    reg["cities"] = [c for c in reg["cities"] if c["id"] not in ids]
    (root / GAME / "registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.path.insert(0, str(TREE / "pipeline"))
    import market_rows
    market_rows.save(reg, market_rows.load(root), root)
    keep = S / f"dropped"
    keep.mkdir(exist_ok=True)
    for i in ids:
        shutil.move(str(root / f"data/packs/{i}.market.json"), str(keep / f"{i}.market.json"))
    (keep / "entries.json").write_text(json.dumps(gone), encoding="utf-8")


def strip(root, ids):
    """Every commute into or out of `ids` taken out of the market packs, as if they were never
    joined (a zone the gate registers has none yet). Returns the packs rewritten."""
    reg = json.loads((root / GAME / "registry.json").read_text(encoding="utf-8"))
    places = {c["place"] for c in reg["cities"] if c["id"] in ids}
    n = 0
    for c in reg["cities"]:
        p = root / f"data/packs/{c['id']}.market.json"
        z = json.loads(p.read_text(encoding="utf-8"))
        if "commute_place" not in z:
            continue
        keep = [k for k, pl in enumerate(z["commute_place"]) if pl not in places and c["id"] not in ids]
        if len(keep) == len(z["commute_place"]):
            continue
        for f in ("commute_home", "commute_place", "commute_work", "commute_trips"):
            z[f] = [z[f][k] for k in keep]
        p.write_text(json.dumps(z, separators=(",", ":")), encoding="utf-8")
        n += 1
    print(f"  commutes with {' '.join(ids)} taken out of {n} packs")


def restore(root):
    """The dropped entries back at the registry's end (register's order), and their market packs."""
    keep = S / "dropped"
    reg = json.loads((root / GAME / "registry.json").read_text(encoding="utf-8"))
    for c in json.loads((keep / "entries.json").read_text(encoding="utf-8")):
        reg["cities"].append(c)
        shutil.copy2(keep / f"{c['id']}.market.json", root / f"data/packs/{c['id']}.market.json")
    (root / GAME / "registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pair(state, name, *args):
    """state's copy run incrementally and with --full; the two compared."""
    inc, full = clone(state, S / f"{name}_inc"), clone(state, S / f"{name}_full")
    args = args or ("--rows",)
    ti = run(inc, *args)
    tf = run(full, *args, "--full")
    ok = same(inc, full)
    shutil.rmtree(full)
    return inc, ok, ti, tf


if __name__ == "__main__":
    case = sys.argv[1]
    if case == "seed":  # every row worked out with this tree's code, and the record
        s0 = clone(S / "base", S / "s0")
        run(s0, "--rows")
        run(s0, "--rows")  # nothing changed
    elif case == "fresh":  # zones out, and every commute with them (their neighbours' packs rewritten)
        ids = sys.argv[2:]
        st = clone(S / "s0", S / "state")
        strip(st, ids)
        drop(st, ids)
        (S / "fresh_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        pair(st, "fresh")
    elif case == "add":  # the same zones registered as new ones (a gate's register --market)
        st = clone(S / "fresh_inc", S / "state")
        restore(st)
        pair(st, "added")
    elif case == "remove":  # and parked again (the gate's unregister, then the join's --rows)
        st = clone(S / "added_inc", S / "state")
        drop(st, json.loads((S / "fresh_ids.json").read_text(encoding="utf-8")))
        pair(st, "removed")
    elif case == "register":  # the zones registered and their market packs built (register --market)
        st = clone(S / "fresh_inc", S / "state")
        main = Path(r"C:\Users\anita\projects\maps\trainworld")
        for f in ["world_field.npz", "world_field.countries.json"] + [p.name for p in (main / "data/work").glob("*.joint_commutes.json")]:
            shutil.copy2(main / "data/work" / f, st / "data/work" / f)
        ents = json.loads((S / "dropped/entries.json").read_text(encoding="utf-8"))
        rows = json.loads((st / GAME / "market-rows.json").read_text(encoding="utf-8"))
        near = [i for i, r in rows.items() if r["country"] in ("CN", "GB")]  # the joint groups' other packs
        for f in ["intercity.json"] + [f"{i}{x}" for i in [c["id"] for c in ents] + near for x in (".json", ".bin")]:
            shutil.copy2(main / "data/packs" / f, st / "data/packs" / f)
        reg = json.loads((st / GAME / "registry.json").read_text(encoding="utf-8"))
        reg["cities"] += ents
        (st / GAME / "registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pair(st, "registered", *[c["id"] for c in ents])
    elif case == "newer":  # every pack and row as the main tree has them now (a batch's gates and join)
        st = clone(S / sys.argv[2], S / "state")
        main = Path(r"C:\Users\anita\projects\maps\trainworld")
        shutil.copy2(main / GAME / "registry.json", st / GAME / "registry.json")
        n = 0
        reg = json.loads((st / GAME / "registry.json").read_text(encoding="utf-8"))
        new = []
        for c in reg["cities"]:
            src, dst = main / f"data/packs/{c['id']}.market.json", st / f"data/packs/{c['id']}.market.json"
            if src.exists() and (not dst.exists() or src.read_bytes() != dst.read_bytes()):
                shutil.copy2(src, dst)
                new.append(c["id"])
        print(f"  {len(new)} market packs new or changed in the main tree: {' '.join(new)}")
        pair(st, "newer")
