"""Backup before a run or join: backup.py <task> <stage> <CC,CC,...>   (e.g. backup.py T1017 pilot US)
Copies the registry, market rows, zone names and aliases, world plan, realms and outlines, every
joint_commutes file, and the packs (json, bin, market.json, boundary) and res9 files of registered
packs in the given countries, to data/work/pre-<task>/<stage>/ keeping relative paths under trainworld/."""
import glob, json, os, shutil, sys

TW = r"C:\Users\anita\projects\maps\trainworld"
task, stage, countries = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
dest = os.path.join(TW, "data", "work", f"pre-{task}", stage)
if os.path.exists(dest):
    sys.exit(f"{dest} exists")

files = ["app/src/game/registry.json", "app/src/game/market-rows.json", "app/src/game/zone-aliases.json",
         "pipeline/world_plan.json", "data/work/market_rows.json",
         "data/packs/realms.geojson", "data/packs/outlines.json", "app/src/game/zone-names.json"]
files += [os.path.relpath(p, TW) for p in glob.glob(os.path.join(TW, "data/work/*.joint_commutes.json"))]
reg = json.load(open(os.path.join(TW, "app/src/game/registry.json"), encoding="utf8"))
ids = [c["id"] for c in reg["cities"] if c.get("country") in countries]
for i in ids:
    for pat in (f"data/packs/{i}.json", f"data/packs/{i}.bin", f"data/packs/{i}.market.json",
                f"data/packs/{i}.boundary.geojson", f"data/work/res9/{i}.*"):
        files += [os.path.relpath(p, TW) for p in glob.glob(os.path.join(TW, pat))]
n = 0
size = 0
for rel in sorted(set(files)):
    src = os.path.join(TW, rel)
    if not os.path.exists(src):
        continue
    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy2(src, out)
    n += 1
    size += os.path.getsize(src)
print(f"{len(ids)} packs, {n} files, {size/1e6:.0f} MB -> {dest}")
