"""T-2554: the data/work and data/frozen deletions that wait for a quiet tree (no world_run live).

    venv\\Scripts\\python.exe -I trainworld\\notes\\t2554\\after_join.py          list what would go
    venv\\Scripts\\python.exe -I trainworld\\notes\\t2554\\after_join.py --go     delete it, print what each kind freed

Refuses while any world_run batch lock is held. What goes (notes/T-2554.md has why each is unread):
per-zone `drive_near_f.npy`, `cells.bin`/`cells.json`, `walk_zone.u8`/`.u8.part`; the per-zone files of
ids in neither the registry, the plan nor the Europe backup; `data/work/retired`, `target-wasm`,
`target/{debug,wasm,wasm32-unknown-unknown}`, `t275`, `t287`, `t537`, `t539`; freezes 20261104-131517
and -182622 once a freeze newer than 20261104-222232 exists. Never pre-T2302 or res9.
"""
import json
import os
import shutil
import sys
from pathlib import Path

TW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TW / "pipeline"))
import freeze  # noqa: E402

WORK = TW / "data" / "work"
FROZEN = TW / "data" / "frozen"
DEAD_SUFFIXES = (".drive_near_f.npy", ".cells.bin", ".cells.json", ".walk_zone.u8", ".walk_zone.u8.part")
ZONE_SUFFIXES = (".tier0_signals.npz", ".units.geojson", ".outline.geojson", ".parks.audit.json",
                 ".medical.audit.json", ".universities.audit.json", ".blocks.csv.gz", ".usa_sectors.npz",
                 ".counties.geojson")
FOLDERS = ("retired", "target-wasm", "target/debug", "target/wasm", "target/wasm32-unknown-unknown",
           "t275", "t287", "t537", "t539")
OLD_FREEZES = ("20261104-131517", "20261104-182622")
NEWEST_BEFORE_EUROPE = "20261104-222232"


def ids_of(registry, plan):
    r = json.loads(Path(registry).read_text(encoding="utf-8"))
    c = r.get("cities") if isinstance(r, dict) and "cities" in r else r
    out = set(c.keys()) if isinstance(c, dict) else {x.get("id") for x in c}
    p = json.loads(Path(plan).read_text(encoding="utf-8"))
    for k in ("metros", "zones"):
        out |= {x["id"] for x in p.get(k, []) if isinstance(x, dict) and x.get("id")}
    return out


def size(p):
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main(go):
    live = freeze.live_runs(TW)
    if live:
        sys.exit(f"not now: {', '.join(live)} live")
    keep = ids_of(TW / "app/src/game/registry.json", TW / "pipeline/world_plan.json")
    bk = WORK / "pre-T2302" / "pre"
    if bk.is_dir():
        keep |= ids_of(bk / "app/src/game/registry.json", bk / "pipeline/world_plan.json")
    todo = {}
    for e in os.scandir(WORK):
        if not e.is_file():
            continue
        if e.name.endswith(DEAD_SUFFIXES):
            todo.setdefault(next(s for s in DEAD_SUFFIXES if e.name.endswith(s)), []).append(Path(e.path))
        elif e.name.endswith(ZONE_SUFFIXES) and e.name.split(".")[0] not in keep:
            todo.setdefault("files of dead ids", []).append(Path(e.path))
    for d in FOLDERS:
        if (WORK / d).exists():
            todo.setdefault("folders", []).append(WORK / d)
    have = sorted(p.name for p in FROZEN.iterdir() if p.is_dir()) if FROZEN.is_dir() else []
    if any(n > NEWEST_BEFORE_EUROPE for n in have):
        todo["old freezes"] = [FROZEN / n for n in OLD_FREEZES if n in have]
    elif any(n in have for n in OLD_FREEZES):
        print("old freezes kept: no freeze newer than", NEWEST_BEFORE_EUROPE, "yet")
    total = 0
    for kind, paths in todo.items():
        s = sum(size(p) for p in paths)
        total += s
        print(f"{s / 1e9:7.3f} GB  {len(paths):5d}  {kind}")
        if go:
            for p in paths:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"{total / 1e9:7.3f} GB {'freed' if go else 'would go'} (frozen sizes are apparent: hard links to packs free nothing)")


if __name__ == "__main__":
    main("--go" in sys.argv)
