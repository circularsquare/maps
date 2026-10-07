"""The log lines (rejected / untraceable / hole) for lines in pieces of one country.
    python why.py <base> <cc> [n]"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from shipped import pieces
base, cc = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
L = json.loads((base / cc / "lines.json").read_text(encoding="utf-8"))["lines"]
log = (base / f"{cc}.log").read_text(encoding="utf-8", errors="replace").splitlines()
rows = []
for l in L:
    if l.get("src", "osm") == "osm" or l.get("service"):
        continue
    ps = pieces(l["sections"])
    if len(ps) > 1:
        rows.append((sum(ps[1:]), l))
for _k, l in sorted(rows, key=lambda r: -r[0])[:n]:
    ids = l.get("rinf_ids") or []
    key = "/".join(ids)[:24]
    print(f"== {l['name']} {ids} {[round(p, 1) for p in pieces(l['sections'])]}")
    for x in log:
        if ("rejected:" in x and x.split("rejected:")[1].strip().startswith(key)) or \
           ("untraceable:" in x and x.split("untraceable:")[1].split()[0] in ids) or \
           ("hole " in x and key.strip() in x):
            print("   ", x.strip()[:200])
