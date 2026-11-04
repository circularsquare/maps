"""Summary of a dry-run plan's zone rows for one country: count, people, area, units; plus merge kinds."""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
plan, iso, log = sys.argv[1], sys.argv[2], sys.argv[3]
doc = json.load(open(plan, encoding="utf-8"))
rows = [r for r in doc["metros"] if r.get("kind") == "zone" and iso in r.get("units", {})]
ext = {}
print(f"{iso}: {len(rows)} zones, {sum(r['people'] < 500_000 for r in rows)} under 0.5M")
for r in sorted(rows, key=lambda r: -r["people"]):
    w, s, e, n = r["bbox"]
    print(f"  {r['id']} {r['name'][:30]:30} {r['people'] / 1e6:5.2f}M {len(r['units'][iso]):4} units  bbox {e - w:5.1f} x {n - s:4.1f} deg  origin {r['origin']}")
txt = open(log, encoding="utf-8").read()
print("joins: partner", len(re.findall(r"joins .*partner \d", txt)), "compact", len(re.findall(r"joins .*compact \d", txt)),
      "small unit", txt.count("biggest partner, shared"), "refused (stays)", txt.count("no neighbour is a partner"),
      "another region", txt.count("another region"))
