"""T-3280: resort towns' visitors for the long-trip destination weight, pipeline/market_tourism.json.

    python pipeline/market_tourism.py           the towns from market_tourism.csv into market_tourism.json
    python pipeline/market_tourism.py --dry     printed, not written

`market_tourism.csv` is the hand table: one row per municipality (its JIS code), its visitors a year from its
prefecture's tourism report (or the town's own where the prefecture has no municipal table), with the year, what
the count is, the URL and the page or table. This script adds each town's outline (KSJ N03 2025, as cached by
pipeline/japan.py under data/raw/tier0/japan/, simplified to about 200 m) and a point inside it, and keeps the
config keys of market_tourism.json (`on`, `beta`, the notes) as they are. market_zones.py reads the file: each
town's visits a day times `beta` join the long-trip attraction of the pack's sites inside its outline, by their
jobs (docs/regional.md 5.1 "Resort towns").
"""
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = Path(__file__).parent / "market_tourism.csv"
OUT = Path(__file__).parent / "market_tourism.json"
MUNI = ROOT / "data/raw/tier0/japan"
SIMPLIFY_DEG = 0.002


def outlines():
    """{JIS code: shapely geometry} over every cached prefecture file."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    parts = {}
    for p in glob.glob(str(MUNI / "municipalities_*.geojson")):
        for f in json.loads(Path(p).read_text(encoding="utf-8"))["features"]:
            parts.setdefault(f["properties"]["code"], {}).setdefault(p, []).append(shape(f["geometry"]))
    # the same municipality sits in several cached sets; one set's features of each
    return {c: unary_union(next(iter(v.values()))).buffer(0) for c, v in parts.items()}


def rings(g):
    """A (multi)polygon as [[[lon, lat], ...] outer rings with their holes], rounded to 4 places."""
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    r = lambda cs: [[round(x, 4), round(y, 4)] for x, y in cs]
    return [[r(p.exterior.coords)] + [r(h.coords) for h in p.interiors] for p in polys]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"task": "T-3280", "on": False, "beta": 0.0}
    shapes = outlines()
    towns, total = [], 0.0
    with open(TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = shapes.get(r["code"])
            if g is None:
                raise SystemExit(f"{r['code']} {r['name']}: no N03 outline cached")
            s = g.simplify(SIMPLIFY_DEG, preserve_topology=True)
            pt = s.representative_point()
            towns.append({"code": r["code"], "name": r["name_en"], "visitors": int(r["visitors"]), "year": r["year"],
                          "lon": round(pt.x, 4), "lat": round(pt.y, 4), "shape": rings(s)})
            total += int(r["visitors"])
    cfg["towns"] = towns
    print(f"{len(towns)} towns, {total / 1e6:.1f}M visitors a year, beta {cfg.get('beta')}, on {cfg.get('on')}")
    if "--dry" not in sys.argv:
        part = OUT.with_suffix(".json.part")
        part.write_text(json.dumps(cfg, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        part.replace(OUT)
        print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
