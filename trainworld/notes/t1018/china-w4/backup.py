"""Backup before a T-897 join: backup.py <wave> <CC,CC,...>
Copies the registry, market rows, world plan, every joint_commutes file, and the packs
(json, bin, market.json, boundary) and res9 files of registered packs in the given countries,
to data/work/pre-T897/<wave>/ keeping relative paths under trainworld/."""
import glob, json, os, shutil, sys

TW = r"C:\Users\anita\projects\maps\trainworld"
wave, countries = sys.argv[1], set(sys.argv[2].split(","))
dest = os.path.join(TW, "data", "work", "pre-T897", wave)
if os.path.exists(dest):
    sys.exit(f"{dest} exists")

files = ["app/src/game/registry.json", "app/src/game/market-rows.json",
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
