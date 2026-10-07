"""tools/ab.py's comparison between two trial folders (baseline and change), per country.
    python abdirs.py <base> <dev> cc ... [--rows N]"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from shipped import pieces  # noqa: E402


def load(d, n):
    return json.loads((d / n).read_text(encoding="utf-8"))


def compare(a, b, nrows):
    la = {l["id"]: l for l in load(a, "lines.json")["lines"]}
    lb = {l["id"]: l for l in load(b, "lines.json")["lines"]}
    rows = []
    reg_km = [0.0, 0.0]
    inp = [0, 0]
    for k in sorted(set(la) | set(lb)):
        x, y = la.get(k), lb.get(k)
        for i, z in enumerate((x, y)):
            if z and z.get("src", "osm") != "osm" and not z.get("service"):
                reg_km[i] += z["km"]
                inp[i] += len(pieces(z["sections"])) > 1
        if x is None or y is None:
            z = x or y
            rows.append(f"{'gone' if y is None else 'new '} {z.get('src')} {z['name']} {z['km']:.2f} km")
            continue
        d = []
        if round(x["km"], 2) != round(y["km"], 2):
            d.append(f"km {x['km']:.2f}->{y['km']:.2f}")
        if sorted(tuple(s[:2]) for s in x["sections"]) != sorted(tuple(s[:2]) for s in y["sections"]):
            px, py = pieces(x["sections"]), pieces(y["sections"])
            d.append(f"sections {len(x['sections'])}->{len(y['sections'])}, pieces {len(px)}->{len(py)}")
        for f in ("name", "name_en", "src", "service"):
            if x.get(f) != y.get(f):
                d.append(f)
        s = lambda v: sorted(v) if isinstance(v, list) else v
        if s(x.get("closed")) != s(y.get("closed")):
            d.append("closed")
        if d:
            rows.append(f"chg  {x.get('src')} {x['name']}: {', '.join(d)}")
    sa, sb = load(a, "stations.json")["stations"], load(b, "stations.json")["stations"]
    same = {n: (a / n).read_bytes() == (b / n).read_bytes() for n in ("foot.json", "ways.json")}
    reg = sum(1 for r in rows if " osm " not in f" {r} ")
    print(f"{a.name}: {len(rows)} lines differ ({reg} not OSM); register km {reg_km[0]:,.1f} -> "
          f"{reg_km[1]:,.1f}; register lines in pieces {inp[0]} -> {inp[1]}; stations "
          f"{len(sa)}->{len(sb)} ({len(set(sa) - set(sb))} gone, {len(set(sb) - set(sa))} new); "
          + ", ".join(f"{n} {'same' if v else 'DIFFERS'}" for n, v in same.items()))
    for r in rows[:nrows]:
        print("    " + r)
    if len(rows) > nrows:
        print(f"    ... {len(rows) - nrows} more")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 25
    if "--rows" in args:
        i = args.index("--rows")
        n = int(args[i + 1])
        del args[i:i + 2]
    base, dev = Path(args[0]), Path(args[1])
    for cc in args[2:]:
        if (base / cc / "lines.json").exists() and (dev / cc / "lines.json").exists():
            compare(base / cc, dev / cc, n)
        else:
            print(f"{cc}: missing")
