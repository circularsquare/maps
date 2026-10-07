"""Register lines in pieces in a lines.json folder: per country counts and the worst lines."""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
RINF = ("at be nl cz pl hu pt si sk bg ro fi lt lv ee hr gr lu de it es se dk ie ru ua by md "
        "in id nz za rs ba me mk al xk kz uz kg tj tm ge am az xa ma dz tn eg").split()


def pieces(secs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s in secs:
        parent[find(s[0])] = find(s[1])
    by = defaultdict(float)
    for s in secs:
        by[find(s[0])] += s[2]
    return sorted(by.values(), reverse=True)


def run(base, ccs):
    worst, tot = [], [0, 0, 0, 0.0, 0.0]
    for cc in ccs:
        f = base / cc / "lines.json"
        if not f.exists():
            continue
        ls = json.loads(f.read_text(encoding="utf-8"))["lines"]
        reg = [l for l in ls if l.get("src", "osm") != "osm" and not l.get("service")]
        n = km_out = km_on = 0
        for l in reg:
            ps = pieces(l["sections"])
            if len(ps) > 1:
                n += 1
                km_out += sum(ps[1:])
                km_on += sum(ps)
                worst.append((sum(ps[1:]), cc, l["name"], l.get("ref"), [round(p, 1) for p in ps]))
        print(f"{cc}: {len(reg)} register lines, {n} in pieces, {km_on:,.0f} km on them, "
              f"{km_out:,.0f} km outside the biggest piece")
        tot[0] += len(reg); tot[1] += n; tot[3] += km_out; tot[4] += km_on
    print(f"ALL: {tot[0]} register lines, {tot[1]} in pieces, {tot[4]:,.0f} km on them, "
          f"{tot[3]:,.0f} km outside")
    for w in sorted(worst, reverse=True)[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
        print(f"  {w[0]:7.1f}  {w[1]} {w[2]} [{w[3]}] {w[4]}")


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "dist" else ROOT / "dist" / "data"
    run(base, RINF)
