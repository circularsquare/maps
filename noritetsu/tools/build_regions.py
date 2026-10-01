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
# 1:10m, not 1:110m: the small-scale file has no Singapore or Hong Kong at all.
NAMES = MAPS / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"

# Degrees. About 2 km at Japan's latitude; see the docstring for why that is enough.
TOLERANCE = 0.02
# Parts smaller than this (square degrees, ~25 km2) are dropped: a rock offshore does not
# decide whether a country is on screen, and there are thousands of them.
MIN_PART = 0.002
# Degrees from the largest part beyond which a part is overseas, for the opening view only.
FAR_DEG = 15

# religiondots' codes where they differ from ISO 3166.
ISO = {"uk": "GB"}


def main():
    built = sorted(p.name for p in (DIST / "data").iterdir()
                   if p.is_dir() and (p / "lines.json").exists()
                   and (DIST / "data" / f"{p.name}.pmtiles").exists())
    if not built:
        sys.exit("no built country under dist/data/")

    # Several features can share a code (France's is also on Clipperton Island), so the most
    # populous one names the country.
    names, pop, ne_geom = {}, {}, {}
    for f in json.loads(NAMES.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        cc, p = q.get("ISO_A2_EH"), q.get("POP_EST") or 0
        if cc not in names or p > pop[cc]:
            names[cc], pop[cc], ne_geom[cc] = q.get("NAME"), p, f["geometry"]

    by_cc = {}
    for f in json.loads(SHAPES.read_text(encoding="utf-8"))["features"]:
        by_cc.setdefault(f["properties"]["cc"], []).append(shape(f["geometry"]))

    out = {}
    for cc in built:
        if cc not in by_cc:
            # religiondots leaves out a few small countries (Luxembourg); Natural Earth's
            # 1:10m outline does as well for "is it on screen".
            g = ne_geom.get(ISO.get(cc, cc.upper()))
            if g is None:
                print(f"  {cc}: no outline in {SHAPES.name} or {NAMES.name}; left out, so it "
                      f"will not load by panning")
                continue
            print(f"  {cc}: outline from {NAMES.name}")
            by_cc[cc] = [shape(g)]
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
        # Nor do parts far from the largest: French Guiana is big enough to pass the 5% test
        # and would open France on South America.
        main = max(polys, key=lambda p: p.area)
        mx, my = main.centroid.x, main.centroid.y
        core = [p for p in polys if p.area >= 0.05 * main.area
                and abs(p.centroid.x - mx) < FAR_DEG and abs(p.centroid.y - my) < FAR_DEG]
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

    # A line that crosses a border is built in every country it touches under one id (the OSM
    # route_master's). The app merges them into one line, and needs to know which countries to
    # load for it before loading any of them.
    seen = {}
    for cc in out:
        for line in json.loads((DIST / "data" / cc / "lines.json").read_text(
                encoding="utf-8"))["lines"]:
            seen.setdefault(line["id"], []).append(cc)
    shared = {lid: sorted(ccs) for lid, ccs in sorted(seen.items()) if len(ccs) > 1}
    print(f"  {len(shared)} line ids shared between countries")

    path = DIST / "regions.json"
    path.write_text(json.dumps({"regions": out, "shared_lines": shared}, ensure_ascii=False,
                               separators=(",", ":")),
                    encoding="utf-8")
    print(f"wrote {path}, {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
