"""Fold a fix-up batch into the batches its regions belong to, then delete it.

    python asia1m/naming/merge.py <fix batch> [--drop <batch>:<n> ...]

Each region goes to the batch (in batches/index.json) whose province and prefecture list holds the
region's main province and prefecture. It is appended with the next free number, so numbers
already used in that batch (and in feedback) stay put. Facts, candidates, pins, overlay and the
thumbnail move with it. --drop removes stale entries (e.g. a region that was split) from a batch.
"""
import os, sys, json, shutil, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "batches")
ap = argparse.ArgumentParser(); ap.add_argument("fix"); ap.add_argument("--drop", nargs="*", default=[])
a = ap.parse_args()

def load(b, f, default=None):
    p = os.path.join(B, b, f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default
def save(b, f, obj):
    json.dump(obj, open(os.path.join(B, b, f), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

index = json.load(open(os.path.join(B, "index.json"), encoding="utf-8"))
def top(r, level):
    ps = [p for p in r["parents"] if p["level"] == level]
    return ps[0]["name"] if ps else None
def home(r):
    for b in index:
        if b["name"] == a.fix or not b.get("province"): continue
        if top(r, "province") == b["province"] and (not b.get("prefectures") or top(r, "prefecture") in b["prefectures"]):
            return b["name"]
    return None

for spec in a.drop:
    b, n = spec.rsplit(":", 1); n = int(n)
    for f, kf in [("facts.json", "regions"), ("candidates.json", "regions")]:
        obj = load(b, f)
        if obj: obj[kf] = [r for r in obj[kf] if r["n"] != n]; save(b, f, obj)
    for f in ["pins.json", "overlay.json"]:
        obj = load(b, f)
        if obj is not None: obj.pop(str(n), None); save(b, f, obj)
    print(f"dropped {b} {n}")

fx = load(a.fix, "facts.json"); fc = load(a.fix, "candidates.json", {"regions": []})
fp = load(a.fix, "pins.json", {}); fo = load(a.fix, "overlay.json", {})
for r in fx["regions"]:
    h = home(r)
    if not h: print(f"no home batch for {r['key']}; left in {a.fix}"); continue
    hf = load(h, "facts.json"); hc = load(h, "candidates.json", {"batch": h, "regions": []})
    hp = load(h, "pins.json", {}); ho = load(h, "overlay.json", {})
    old, new = r["n"], max([x["n"] for x in hf["regions"]] + [x["n"] for x in hc["regions"]] + [0]) + 1
    hf["regions"].append({**r, "n": new})
    c = next((x for x in fc["regions"] if x["n"] == old), None)
    if c: hc["regions"].append({**c, "n": new})
    if str(old) in fp: hp[str(new)] = fp[str(old)]
    if str(old) in fo: ho[str(new)] = fo[str(old)]
    shutil.copyfile(os.path.join(B, a.fix, "thumbs", f"{old}.png"), os.path.join(B, h, "thumbs", f"{new}.png"))
    save(h, "facts.json", hf); save(h, "candidates.json", hc); save(h, "pins.json", hp); save(h, "overlay.json", ho)
    print(f"{a.fix} {old} ({r['key']}) -> {h} {new}")
    fx["regions"] = [x for x in fx["regions"] if x is not r]

if not fx["regions"]:
    shutil.rmtree(os.path.join(B, a.fix))
    index = [b for b in index if b["name"] != a.fix]
    print(f"removed batch {a.fix}")
for b in index:
    f = load(b["name"], "facts.json")
    if f: b["count"] = len(f["regions"])
json.dump(index, open(os.path.join(B, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
