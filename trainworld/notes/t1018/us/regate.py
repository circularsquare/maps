"""Mean commute and verdict per pack from a gate folder: regate.py <gate dir>"""
import glob, json, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    g = json.load(open(p, encoding="utf-8"))
    cs = {c["name"]: c for c in g.get("checks") or []}
    m = cs.get("mean_commute_km", {})
    bad = [f"{c['status']} {c['name']}" for c in cs.values() if c.get("status") in ("FAIL", "WARN")]
    print(os.path.basename(p)[:-5], g.get("verdict"), round(m.get("value") or 0, 2), m.get("status"), "|", ", ".join(bad))
