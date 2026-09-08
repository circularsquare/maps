"""Slovakia — the 2,927 obce, and the census's own 1 km grid to place dots inside them.

Writes:
    data/geo/sk/sk_obce.gpkg        2,927 polygons keyed as sk.py keys them   (`units`)
    data/geo/sk/sk_grid_1km.gpkg    1 km cells with `unit` and `pop`          (`place`)

Usage:
    python sources/sk_geo.py --fetch   # two paged ArcGIS queries, ~25 MB of GeoJSON
    python sources/sk_geo.py           # build both layers

**THERE IS NO JOIN, WHICH IS NEW HERE.** Every other country on this map pairs a counts table
from one publisher with polygons from another, and §12's "shapes of failure" 1 and 2 are both
about that pairing. Slovakia's religion counts and its municipal polygons are **fields and
geometry on the same ArcGIS feature layer** — `obyv_ekchar_nabo_vekskup/FeatureServer/4` — so
the unit a number belongs to is not asserted by this build at all. The `uzemie` codes are
still checked against `data/normalized/sk.csv` for equality as sets, but that check cannot
fail unless one of the two fetches is broken.

**THE PLACEMENT GRID IS THE CENSUS ITSELF, AND THAT CUTS BOTH WAYS.** `obyv_grid_1km` is SODB
2021 redistributed to 49,969 1 km cells, and `sum(obyv_tp_all)` over the whole layer is
**5,449,270 — the census total to the person**. So Slovakia joins Germany (§9g) as a country
whose placement is *measured* rather than modelled, and no Kontur extract is needed.

> **BUT IT IS NOT AN INDEPENDENT CHECK, AND §9av's RULE IS WHY.** The Central African Republic
> taught that a population grid can be downstream of the census it is being checked against;
> here it is not merely downstream, it *is* the same enumeration. So the grid-vs-census ratio
> band that catches a scrambled name join everywhere else **cannot say anything about the
> counts** in Slovakia. What it still does is validate the one join this build really makes —
> cell to obec — and that is the only thing its output should be read as.

**EVERY OBEC MUST APPEAR IN THE PLACE LAYER OR ITS PEOPLE ARE SILENTLY DROPPED.** §9af's
Mauritius lesson: `place_weight`'s equal-share fallback cannot fire for a unit that has no
geometry to fall back into, so a unit missing from the grid loses its dots rather than
spreading them. Slovakia's smallest obce are well under 1 km², so cells are assigned by
centroid and any obec left with none is then given its own polygon as a one-cell placement
area. `check()` asserts the final layer covers all 2,927.
"""

import json
import os
import sys
import urllib.parse

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sk")
GEO = os.path.join(ROOT, "data", "geo", "sk")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "sk.csv")

SERVICE = "https://gis.scitanie.sk/server/rest/services/Hosted"
OBEC = (f"{SERVICE}/obyv_ekchar_nabo_vekskup/FeatureServer/4", ["uzemie", "nazov", "kraj", "okres", "spolu"])
GRID = (f"{SERVICE}/obyv_grid_1km/FeatureServer/2", ["cellcode", "obyv_tp_all"])

OBEC_GEOJSON = os.path.join(RAW, "sk_obce.geojson")
GRID_GEOJSON = os.path.join(RAW, "sk_grid_1km.geojson")
UNITS_GPKG = os.path.join(GEO, "sk_obce.gpkg")
PLACE_GPKG = os.path.join(GEO, "sk_grid_1km.gpkg")

PAGE = 2000
EXPECTED_OBCE = 2_927
EXPECTED_CELLS = 49_969
NATIONAL = 5_449_270

# EPSG:3035 (LAEA Europe) for anything measured in metres; the service serves 4326.
METRIC = 3035
WGS84 = 4326


def _ctx():
    """See sources/sk.py — certifi verifies this host, the Windows store does not."""
    import ssl

    import certifi
    return ssl.create_default_context(cafile=certifi.where())


def _get(url, timeout=300):
    import urllib.request

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        body = r.read()
    if body[:1] != b"{":
        raise SystemExit(f"not JSON from {url}\n  first bytes: {body[:120]!r}")
    d = json.loads(body)
    if "error" in d:
        raise SystemExit(f"ArcGIS error from {url}: {d['error']}")
    return d


def _fetch(layer_url, fields, expected, path):
    n = _get(f"{layer_url}/query?where=1%3D1&returnCountOnly=true&f=json")["count"]
    if n != expected:
        raise SystemExit(f"{layer_url}: server says {n} features, expected {expected}")
    feats, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1", "outFields": ",".join(fields), "returnGeometry": "true",
            "outSR": WGS84, "orderByFields": "objectid",
            "resultOffset": offset, "resultRecordCount": PAGE, "f": "geojson"})
        d = _get(f"{layer_url}/query?{q}")
        got = d.get("features", [])
        feats += got
        if offset % 10000 == 0 or len(got) < PAGE:
            print(f"    offset {offset:>6d}: {len(got):>5d}  (total {len(feats):,})")
        if len(got) < PAGE:
            break
        offset += len(got)
    # §5a in its ArcGIS disguise: maxRecordCount silently truncates and sets a flag rather
    # than erroring. Assert against the server's own count, never against one response.
    if len(feats) != expected:
        raise SystemExit(f"{layer_url}: paged {len(feats)}, expected {expected}")
    os.makedirs(RAW, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, ensure_ascii=False)
    print(f"  wrote {path} ({os.path.getsize(path):,} bytes)")


def fetch():
    print("GET obce (polygons + counts)")
    _fetch(OBEC[0], OBEC[1], EXPECTED_OBCE, OBEC_GEOJSON)
    print("GET 1 km population grid")
    _fetch(GRID[0], GRID[1], EXPECTED_CELLS, GRID_GEOJSON)


def build():
    import geopandas as gpd
    import pandas as pd

    for p in (OBEC_GEOJSON, GRID_GEOJSON):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run with --fetch first")

    obce = gpd.read_file(OBEC_GEOJSON).set_crs(WGS84, allow_override=True)
    obce["uzemie"] = obce["uzemie"].astype(str).str.strip()
    obce["nazov"] = obce["nazov"].astype(str).str.strip()
    if len(obce) != EXPECTED_OBCE:
        raise SystemExit(f"{len(obce)} obce polygons, expected {EXPECTED_OBCE}")
    if obce["uzemie"].duplicated().any():
        raise SystemExit("duplicate uzemie in the polygon layer")

    # the units layer, keyed exactly as sk.py keys the counts
    units = obce[["uzemie", "nazov", "kraj", "okres", "spolu", "geometry"]].copy()

    grid = gpd.read_file(GRID_GEOJSON).set_crs(WGS84, allow_override=True)
    grid["pop"] = pd.to_numeric(grid["obyv_tp_all"], errors="coerce").fillna(0.0)
    if len(grid) != EXPECTED_CELLS:
        raise SystemExit(f"{len(grid)} grid cells, expected {EXPECTED_CELLS}")

    # ---- CELLS ARE SPLIT BY AREA, NOT ASSIGNED BY CENTRE -------------------------------
    # me_geo.py's centroid rule is wrong here and the failure is loud: Slovakia has obce far
    # smaller than a 1 km cell, so a cell whose centre happens to land in a tiny village
    # credits that village with the whole cell's people. Záborie (170 people) came out with
    # 1,409 and Mošurov (180) with 1,338 — an 8x over-weight — which would pull most of a
    # neighbouring town's dots into a hamlet. Intersecting instead and splitting each cell's
    # population by the share of its area in each obec is both correct and cheap here.
    m_grid = grid.to_crs(METRIC)
    m_obce = units.to_crs(METRIC)
    m_grid["cell_area"] = m_grid.geometry.area

    pieces = gpd.overlay(m_grid[["cellcode", "pop", "cell_area", "geometry"]],
                         m_obce[["uzemie", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces = pieces[pieces.geometry.notna() & ~pieces.geometry.is_empty].copy()
    pieces["piece_area"] = pieces.geometry.area
    # A CELL'S PEOPLE ARE ALL IN SLOVAKIA EVEN WHERE ITS SQUARE IS NOT. Dividing by the full
    # cell area throws away the share of a border cell that lies over Austria, Hungary or
    # Poland, and with it 5,762 real Slovaks (0.106%). Normalise against the area actually
    # inside the country instead, so every cell's population is fully distributed among the
    # obce it genuinely overlaps.
    inside = pieces.groupby("cellcode")["piece_area"].transform("sum")
    pieces["pop"] = pieces["pop"] * (pieces["piece_area"] / inside)
    pieces = pieces.rename(columns={"uzemie": "unit"})
    # a cell split across two obce yields two pieces, so the id must carry both
    pieces["cellcode"] = pieces["cellcode"].astype(str) + "|" + pieces["unit"].astype(str)

    placed = float(pieces["pop"].sum())
    lost = NATIONAL - placed
    print(f"  {len(pieces):,} cell/obec pieces; {placed:,.0f} of {NATIONAL:,} people fall "
          f"inside an obec ({lost:,.0f} outside, {100.0 * lost / NATIONAL:.3f}%)")

    # ---- every obec must be represented (§9af) ---------------------------------------
    missing = sorted(set(units["uzemie"]) - set(pieces["unit"]))
    if missing:
        print(f"  {len(missing)} obec(s) intersect no grid cell; adding their own polygon "
              f"as a placement area")
        patch = m_obce[m_obce["uzemie"].isin(missing)][["uzemie", "geometry"]].copy()
        patch = patch.rename(columns={"uzemie": "unit"})
        patch["cellcode"] = "obec:" + patch["unit"]
        # nominal: it is the unit's only cell, so its share is 1 whatever the number
        patch["pop"] = 1.0
        pieces = gpd.GeoDataFrame(pd.concat([pieces, patch], ignore_index=True),
                                  geometry="geometry", crs=m_obce.crs)

    place = pieces[["cellcode", "unit", "pop", "geometry"]].to_crs(WGS84)
    units = units.to_crs(WGS84)

    os.makedirs(GEO, exist_ok=True)
    units.to_file(UNITS_GPKG, layer="obce", driver="GPKG")
    place.to_file(PLACE_GPKG, layer="grid", driver="GPKG")
    print(f"  wrote {UNITS_GPKG} ({len(units):,} polygons)")
    print(f"  wrote {PLACE_GPKG} ({len(place):,} cells)")
    return units, place, lost


def check(units, place, lost):
    import pandas as pd

    ok = True
    counts = pd.read_csv(NORMALIZED, dtype={"geo_id": str})
    obec_codes = set(counts.loc[counts["geo_level"] == "obec", "geo_id"])

    good = set(units["uzemie"]) == obec_codes
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the polygon layer's {len(units):,} codes are exactly "
          f"the {len(obec_codes):,} in sk.csv")

    covered = set(place["unit"])
    good = covered == set(units["uzemie"])
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every obec has somewhere to put its dots "
          f"({len(covered):,} of {len(units):,})"
          + ("" if good else f" — missing {sorted(set(units['uzemie']) - covered)[:5]}"))

    good = abs(lost) / NATIONAL < 0.001
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the grid population falling outside every obec is "
          f"{lost:,.0f} ({100.0 * lost / NATIONAL:.3f}%, want <0.1%)")

    # THE RATIO BAND, AND WHAT IT CAN AND CANNOT SAY. §9av: this grid is the same
    # enumeration as the counts, so agreement is not evidence about the census. It IS
    # evidence about the cell-to-obec join, which is the only join here.
    per = place.groupby("unit")["pop"].sum()
    cen = counts[(counts["geo_level"] == "obec") & (counts["source_category"] == "spolu")]
    cen = cen.set_index("geo_id")["count"]
    both = pd.concat([per.rename("grid"), cen.rename("census")], axis=1).dropna()
    both = both[both["census"] > 0]
    both["ratio"] = both["grid"] / both["census"]
    inband = both["ratio"].between(0.5, 2.0)
    good = float(inband.mean()) > 0.97
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} grid/census ratio within 0.5-2.0 for "
          f"{100.0 * inband.mean():.2f}% of obce (median {both['ratio'].median():.3f})")
    print("      NOT an independent check on the counts (§9av) — the grid IS SODB 2021. It "
          "checks the\n      cell-to-obec assignment and nothing else.")
    worst = both.reindex(both["ratio"].sub(1).abs().sort_values(ascending=False).index).head(5)
    for code, r in worst.iterrows():
        name = units.loc[units["uzemie"] == code, "nazov"]
        print(f"      {code} {str(name.iloc[0])[:26]:26s} grid {r['grid']:>8,.0f} "
              f"census {r['census']:>8,.0f}  ratio {r['ratio']:.2f}")

    if not ok:
        raise SystemExit("geometry checks FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units, place, lost = build()
    check(units, place, lost)


if __name__ == "__main__":
    main()
