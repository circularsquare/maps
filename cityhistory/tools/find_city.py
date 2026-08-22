"""Look up what cities.json actually knows about a place, for writing history notes.

A note is only worth having if the map does something at that year, and the anchor has to be a
real coordinate. This answers both without loading 4.8MB of JSON by hand.

    python tools/find_city.py hampi vijayanagara     # search by name (substrings, any number)
    python tools/find_city.py --near 33.31 44.36     # what is within ~300km of a coordinate
    python tools/find_city.py --near 33.31 44.36 --year 1258
"""
import json, math, sys, pathlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # city names are not cp1252

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITIES = ROOT / "data" / "cities.json"


def haversine(la1, lo1, la2, lo2):
    p = math.pi / 180
    a = (0.5 - math.cos((la2 - la1) * p) / 2
         + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2)
    return 12742 * math.asin(math.sqrt(max(0.0, a)))


def fmt_year(y):
    return f"{-y}BC" if y < 0 else str(int(y))


def fmt_pop(p):
    if p >= 1e6:
        return f"{p / 1e6:.1f}M"
    if p >= 1e3:
        return f"{p / 1e3:.0f}k"
    return str(int(p))


def show(c, extra=""):
    p = c["p"]
    print(f"\n{c['n']}   {c['la']:.4f}, {c['lo']:.4f}   "
          f"{fmt_year(p[0][0])}..{fmt_year(p[-1][0])}   {len(p)} points{extra}")
    # every control point, wrapped -- the shape of the curve is the whole reason to look
    line, out = "  ", []
    for y, v in p:
        cell = f"{fmt_year(y)}:{fmt_pop(v)}"
        if len(line) + len(cell) > 96:
            out.append(line)
            line = "  "
        line += cell + "  "
    out.append(line)
    print("\n".join(out))


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    cities = json.loads(CITIES.read_text(encoding="utf-8"))["cities"]

    if argv[0] == "--near":
        la, lo = float(argv[1]), float(argv[2])
        year = None
        if "--year" in argv:
            year = float(argv[argv.index("--year") + 1])
        near = sorted(((haversine(la, lo, c["la"], c["lo"]), c) for c in cities),
                      key=lambda t: t[0])[:12]
        for d, c in near:
            if d > 400:
                break
            note = f"   {d:.0f}km"
            if year is not None:
                p = c["p"]
                live = p[0][0] <= year <= p[-1][0]
                note += "   ON the map in " + fmt_year(year) if live else \
                        f"   NOT on the map in {fmt_year(year)}"
            show(c, note)
        return 0

    terms = [a.lower() for a in argv]
    hits = [c for c in cities if any(t in c["n"].lower() for t in terms)]
    if not hits:
        print(f"no city name contains any of {terms}. Names are the source's own spelling -- try "
              "a shorter substring, or --near with a coordinate.")
        return 1
    for c in sorted(hits, key=lambda c: -max(v for _, v in c["p"]))[:15]:
        show(c)
    if len(hits) > 15:
        print(f"\n...and {len(hits) - 15} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
