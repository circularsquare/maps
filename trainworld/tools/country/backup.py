"""Back up the shared files a run or join rewrites, before it: the state `restore.py` puts back.

    backup.py <task> <stage> <CC,CC,...|all> [--dry] [--tw <trainworld folder>]

    backup.py T1990 pre  VN,TH,SG,MY       before a run
    backup.py T1990 prejoin VN,TH,SG,MY    before the join

Copies to data/work/pre-<task>/<stage>/, keeping paths relative to trainworld/: the registry, market
rows, zone names and aliases, the world plan, realms and outlines, every data/work/*.joint_commutes.json,
and for each registered pack whose registry `country` is in the list (ISO 3166 alpha-2; list the
neighbours the join also touches): its .json, .bin, .market.json, .boundary.geojson and
data/work/res9/<id>.* files. `all` takes every registered pack. --dry counts and writes nothing.
Refuses a stage folder that exists. Delete the backup once a later state is checked clean
(CLAUDE.md "Pack rebuilds"). --tw points at another tree of the same layout (a scratch tree).
"""
import argparse
import glob
import json
import os
import shutil
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHARED = ["app/src/game/registry.json", "app/src/game/market-rows.json", "app/src/game/zone-aliases.json",
          "pipeline/world_plan.json", "data/work/market_rows.json",
          "data/packs/realms.geojson", "data/packs/outlines.json", "app/src/game/zone-names.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="task id for the folder name, e.g. T1990")
    ap.add_argument("stage", help="folder under it, e.g. pre or prejoin")
    ap.add_argument("countries", help="comma list of ISO alpha-2 codes, or all")
    ap.add_argument("--dry", action="store_true", help="count the files, copy nothing")
    ap.add_argument("--tw", default=TW, help="trainworld folder (default: this tree's)")
    a = ap.parse_args()
    tw = os.path.abspath(a.tw)
    countries = None if a.countries == "all" else set(a.countries.split(","))
    dest = os.path.join(tw, "data", "work", f"pre-{a.task}", a.stage)
    if os.path.exists(dest):
        sys.exit(f"{dest} exists")

    files = list(SHARED)
    files += [os.path.relpath(p, tw) for p in glob.glob(os.path.join(tw, "data/work/*.joint_commutes.json"))]
    with open(os.path.join(tw, "app/src/game/registry.json"), encoding="utf8") as f:
        reg = json.load(f)
    ids = [c["id"] for c in reg["cities"] if countries is None or c.get("country") in countries]
    if not ids:
        sys.exit(f"no registered pack has country in {a.countries}")
    for i in ids:
        for pat in (f"data/packs/{i}.json", f"data/packs/{i}.bin", f"data/packs/{i}.market.json",
                    f"data/packs/{i}.boundary.geojson", f"data/work/res9/{i}.*"):
            files += [os.path.relpath(p, tw) for p in glob.glob(os.path.join(tw, pat))]
    n = 0
    size = 0
    for rel in sorted(set(files)):
        src = os.path.join(tw, rel)
        if not os.path.exists(src):
            continue
        if not a.dry:
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(src, out)
        n += 1
        size += os.path.getsize(src)
    print(f"{len(ids)} packs, {n} files, {size / 1e6:.0f} MB -> {dest}" + (" (dry: nothing copied)" if a.dry else ""))


if __name__ == "__main__":
    main()
