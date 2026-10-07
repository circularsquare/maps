import json, sys
sys.stdout.reconfigure(encoding="utf-8")
tot = {}
print(f"{'cc':3} {'osm/osm':>8} {'osm|none/reg':>12} {'none/osm':>8} {'reg/reg':>8} | {'shore':>6} {'water':>6} {'foreign':>7} {'homeNE':>6}")
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    if "error" in d:
        print(d["cc"], "ERROR", d["error"][-200:])
        continue
    df, ab = d["diff"], d["abroad"]
    row = [df.get("osm/osm(diff)", 0), df.get("osm/reg", 0) + df.get("none/reg", 0), df.get("none/osm", 0),
           df.get("reg/reg(diff)", 0), ab.get("home full outline", 0), ab.get("water", 0), ab.get("foreign", 0), ab.get("home in NE", 0)]
    for i, v in enumerate(row):
        tot[i] = tot.get(i, 0) + v
    print(f"{d['cc']:3} " + " ".join(f"{v:8.1f}" for v in row) + "   " + "; ".join(f"{n[:30]} {k}" for n, k in d["diff_lines"][:3]))
print("all " + " ".join(f"{tot[i]:8.1f}" for i in range(8)))
