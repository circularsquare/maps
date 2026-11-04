"""A dry plan's rows and dead space for countries. Usage: ua_look.py <plan.json> UKR,RUS [--rows]"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
doc = json.load(open(sys.argv[1], encoding="utf-8"))
isos = sys.argv[2].split(",")
for iso in isos:
    z = doc["zones"].get(iso, {})
    dead = z.get("dead", [])
    print(iso, "dead units", len(dead), "people", round(sum(d["people"] for d in dead) / 1e6, 2), "M;",
          "left out", len(z.get("left_out", {}).get("units", [])),
          round(sum(u["people"] for u in z.get("left_out", {}).get("units", [])) / 1e6, 2), "M")
    big = sorted(dead, key=lambda d: -d["people"])[:15]
    print("  biggest dead:", [(d["name"], d["people"], d.get("group"), d.get("per_km2")) for d in big])
    rows = [r for r in doc["metros"] if iso in (r.get("units") or {}) and r.get("status") in ("planned", "held")]
    tot = 0
    for r in rows:
        tot += r.get("people", 0)
        s = r.get("source", {})
        print(f"  {r['id']} {r['name']} {r.get('kind', 'metro')} {r.get('status')} {r['people'] / 1e6:.2f}M origin {r['origin']}"
              f" groups {s.get('groups', [])[:4]} units {s.get('unit_names', [])[:3]} extracts {r.get('extracts')} tz {r.get('tz')}")
    print(f"  {len(rows)} rows, {tot / 1e6:.2f}M")
