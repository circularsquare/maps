import json, os, collections
here = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(r"C:/Users/anita/projects/maps/artifacts/trainworld/t1000-pullscan.json"))
by = collections.defaultdict(list)
for r in rows:
    by[r["country"] or "?"].append(r)
print("country packs | cap-20 touches any job | >1% of jobs | mean>25 km | cross-pack commuters out / workers (joined)")
tot = [0, 0, 0, 0]
for c, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    a = sum(1 for r in rs if r["jobs_past_20"] > 0)
    b = sum(1 for r in rs if r["jobs_past_20"] > 0.01)
    m = sum(1 for r in rs if r["mean_km"] > 25)
    js = [r for r in rs if r.get("out") is not None]
    rate = sum(r["out"] for r in js) / max(sum(r["workers"] for r in js), 1)
    tot = [tot[0] + len(rs), tot[1] + a, tot[2] + b, tot[3] + m]
    if a or m or len(rs) >= 20:
        print(f"{c:4s} {len(rs):5d} {a:5d} {b:5d} {m:5d}   {rate:6.1%} over {len(js)}")
print("all", tot)
print()
print("China by province: packs, cap-20 >1% of jobs, mean>25 (ids with mean km)")
cn = collections.defaultdict(list)
for r in by["CN"]:
    cn[r["region"]].append(r)
for reg, rs in sorted(cn.items(), key=lambda kv: str(kv[0])):
    b = sum(1 for r in rs if r["jobs_past_20"] > 0.01)
    long = sorted([r for r in rs if r["mean_km"] > 25], key=lambda r: -r["mean_km"])
    print(f"{str(reg):14s} {len(rs):4d} {b:4d} {len(long):4d}  " + ", ".join(f"{r['id']} {r['mean_km']:.0f}" for r in long))
print()
print("outside China, mean > 25 km or cap-20 touches >1% of jobs:")
for r in sorted(rows, key=lambda r: -r["mean_km"]):
    if r["country"] != "CN" and (r["mean_km"] > 25 or r["jobs_past_20"] > 0.01):
        print(f"  {r['id']} {r['country']} {r['name']}: mean {r['mean_km']:.1f} km, jobs past 20x {r['jobs_past_20']:.1%}, pull max {r['pull_max']:.3g}")
print()
cn_out = sorted([r for r in by["CN"] if r.get("out")], key=lambda r: -r["out"] / r["workers"])
print("CN cross-pack out share: median", sorted(r["out"] / r["workers"] for r in cn_out)[len(cn_out) // 2], "top:", ", ".join(f"{r['id']} {r['out'] / r['workers']:.0%}" for r in cn_out[:8]))
for c in ("US", "JP", "GB", "DE", "FR"):
    rs = [r for r in by.get(c, []) if r.get("out")]
    if rs:
        print(c, "cross-pack out share", f"{sum(r['out'] for r in rs) / sum(r['workers'] for r in rs):.1%}")
