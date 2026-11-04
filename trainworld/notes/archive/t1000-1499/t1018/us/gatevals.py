"""Key gate figures per row, from every gate json in the given batch dirs:
gatevals.py <batch dir>[,<batch dir>...] [ids]   (FAIL and WARN checks listed after)"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
KEYS = ["pop_vs_plan", "pop_vs_fua", "pop_vs_ghsl_other", "jobs_per_resident", "mean_commute_km", "drive_factor_median",
        "top_zone_jobs_share", "area_km2"]
want = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
print("id verdict " + " ".join(KEYS))
for d in sys.argv[1].split(","):
    for p in sorted(glob.glob(os.path.join(d, "**", "gate", "*.json"), recursive=True) + glob.glob(os.path.join(d, "gate", "*.json"))):
        i = os.path.basename(p)[:-5]
        if want and i not in want:
            continue
        g = json.load(open(p, encoding="utf-8"))
        cs = {c["name"]: c for c in g.get("checks") or []}
        def v(k):
            x = cs.get(k, {}).get("value")
            return f"{x:.2f}" if isinstance(x, (int, float)) else "-"
        bad = [f"{c['status']} {c['name']} {str(c.get('value'))[:40]}" for c in cs.values() if c.get("status") in ("FAIL", "WARN")]
        print(i, g.get("verdict"), " ".join(v(k) for k in KEYS), "|", "; ".join(bad)[:400])
