"""Write dist/regions.json: every built country's name, box, opening view and outline.

    python tools/build_regions.py

WHAT IT IS FOR. The app loads a country only when the map moves over it, so it has to know
where each country is BEFORE loading anything of it. This is that index: a few KB, fetched
once at startup, one entry per country that has a build under dist/data/<cc>/.

Run it after a new country's first build. It changes nothing else, and the app still works
without it for countries already listed.

The outlines are religiondots' `country_shapes.geojson` (194 countries, cleaned and
simplified to ~200 m there), simplified again here to ~2 km: the only question asked of them
is "is this country on screen, and is the middle of the screen in it", and at 2 km Japan is
a few hundred points rather than tens of thousands. Names are Natural Earth's.
"""
import json
import pathlib
import sys

from shapely.geometry import shape, mapping

HERE = pathlib.Path(__file__).resolve().parent.parent
DIST = HERE / "dist"
MAPS = HERE.parent
SHAPES = MAPS / "religiondots" / "data" / "processed" / "country_shapes.geojson"
NAMES = MAPS / "religiondots" / "data" / "geo" / "ne_110m_admin_0_countries.geojson"

# Degrees. About 2 km at Japan's latitude; see the docstring for why that is enough.
TOLERANCE = 0.02
# Parts smaller than this (square degrees, ~25 km2) are dropped: a rock offshore does not
# decide whether a country is on screen, and there are thousands of them.
MIN_PART = 0.002

# religiondots' codes where they differ from ISO 3166.
ISO = {"uk": "GB"}


def main():
    built = sorted(p.name for p in (DIST / "data").iterdir()
                   if p.is_dir() and (p / "lines.json").exists()
                   and (DIST / "data" / f"{p.name}.pmtiles").exists())
    if not built:
        sys.exit("no built country under dist/data/")

    names = {}
    for f in json.loads(NAMES.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        names[q.get("ISO_A2_EH")] = q.get("NAME")

    by_cc = {}
    for f in json.loads(SHAPES.read_text(encoding="utf-8"))["features"]:
        by_cc.setdefault(f["properties"]["cc"], []).append(shape(f["geometry"]))

    out = {}
    for cc in built:
        if cc not in by_cc:
            print(f"  {cc}: no outline in {SHAPES.name}; left out, so it will not load by panning")
            continue
        geom = by_cc[cc][0]
        for g in by_cc[cc][1:]:
            geom = geom.union(g)
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        parts = []
        for p in polys:
            p = p.simplify(TOLERANCE, preserve_topology=True)
            if p.is_empty or p.area < MIN_PART:
                continue
            parts.append([[round(x, 3), round(y, 3)] for x, y in p.exterior.coords])
        w, s, e, n = geom.bounds
        # The opening view leaves out outlying islands: Japan's full box reaches Minami-Torishima
        # at 154 E and would open on the Pacific. Parts under 5% of the largest do not count.
        big = max(p.area for p in polys)
        core = [p for p in polys if p.area >= 0.05 * big]
        vw = min(p.bounds[0] for p in core); vs = min(p.bounds[1] for p in core)
        ve = max(p.bounds[2] for p in core); vn = max(p.bounds[3] for p in core)
        out[cc] = {
            "name": names.get(ISO.get(cc, cc.upper()), cc.upper()),
            "bbox": [round(w, 3), round(s, 3), round(e, 3), round(n, 3)],
            "view": [round(vw, 3), round(vs, 3), round(ve, 3), round(vn, 3)],
            "parts": parts,
        }
        print(f"  {cc}: {out[cc]['name']}, {len(parts)} parts, "
              f"{sum(len(p) for p in parts)} points")

    path = DIST / "regions.json"
    path.write_text(json.dumps({"regions": out}, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"wrote {path}, {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
