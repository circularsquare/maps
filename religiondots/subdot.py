"""
The groups too small to reach one dot, with their people — spec §17.

WHAT THIS IS FOR. §4.1a's carry means the only people the dots leave out are a node with fewer
people in a whole country than one dot: under 1,000, or 10,000 in the coarse edition. scatter.py
already places each such node ONCE, at its largest concentration, as a ring (§4.3) — but a ring
carries no count (`n`, `c`, `why`, and the geojson adds `congregations`), and counts.json's `rings`
tally counts rings, one each. So nothing in the archive says how many people a ring stands for,
and the viewer could only draw them as a symbol that claims no size.

This writes the missing number. For every country, for each edition, every ring with the node's
NATIONAL people from that country's own `counts()`, at the ring's position:

    data/processed/subdot.json   {cc: {"1000": [[node, lon, lat, people], ...], "10000": [...]}}

`tiles.py --refresh-meta` carries each country's entry into counts.json as `subdot`, so no retile.
The viewer's pies mode draws each as part of the pie at that place, at its real size (§17).

A ring whose national count is not under the edition's dot value is printed and still written:
it would mean counts() and the scatter disagree about that node, which is worth seeing rather
than hiding.

Run:  python subdot.py                  every country -> data/processed/subdot.json
      python subdot.py --countries in,us   just these, merged into what is there
      python subdot.py --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROC = HERE / "data" / "processed"
OUT = PROC / "subdot.json"
EDITIONS = [("", 1000), ("_10k", 10000)]


def rings_of(cc, suffix):
    """[(node, lon, lat)] from scatter.py's rings file for one country and edition."""
    p = PROC / f"rings_{cc}{suffix}.geojson"
    if not p.exists():
        return []
    gj = json.loads(p.read_text(encoding="utf-8"))
    return [(f["properties"]["n"], *f["geometry"]["coordinates"]) for f in gj["features"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", default="", help="comma-separated; default every country")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    import countries as C

    want = [c.strip() for c in args.countries.split(",") if c.strip()]
    ccs = [cc for cc in sorted(C.COUNTRIES) if not want or cc in want]
    out = json.loads(OUT.read_text(encoding="utf-8")) if (OUT.exists() and want) else {}

    n_rings = n_people = 0
    odd = []
    for cc in ccs:
        rings = {suffix: rings_of(cc, suffix) for suffix, _ in EDITIONS}
        if not any(rings.values()):
            out.pop(cc, None)
            continue
        try:
            df = C.COUNTRIES[cc]["counts"]()
        except Exception as e:                                      # noqa: BLE001
            print(f"{cc:<4} !! counts() failed: {str(e)[:70]}")
            continue
        people = df.groupby("node")["count"].sum().to_dict()
        entry = {}
        for suffix, dv in EDITIONS:
            rows = []
            for node, lon, lat in rings[suffix]:
                p = float(people.get(node, 0) or 0)
                if p <= 0:
                    odd.append((cc, dv, node, p))
                    continue
                if p >= dv:
                    odd.append((cc, dv, node, p))
                rows.append([node, round(lon, 4), round(lat, 4), round(p, 1)])
                n_rings += 1
                n_people += p
            if rows:
                entry[str(dv)] = rows
        if entry:
            out[cc] = entry
            print(f"{cc:<4} " + "  ".join(f"1:{dv} {len(entry.get(str(dv), []))} groups, "
                                          f"{sum(r[3] for r in entry.get(str(dv), [])):,.0f} people"
                                          for _, dv in EDITIONS))
        else:
            out.pop(cc, None)

    print(f"\n{len(out)} countries, {n_rings:,} groups, {n_people:,.0f} people over both editions")
    if odd:
        print(f"\n{len(odd)} ring(s) whose national count is zero (left out) or not under the dot "
              "value (kept):")
        for cc, dv, node, p in odd[:60]:
            print(f"  {cc} 1:{dv} {node:<40} {p:,.1f}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    # through a temp file, as rollup.py: tiles.py reads this and must never see half of it
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(OUT)
    print(f"\nwrote {OUT}")
    print("Now: python tiles.py --refresh-meta   (carries it into counts.json; no retile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
