"""Cyprus — the placement layer: Kontur 400 m population hexagons, keyed to community.

Writes data/geo/cy/cy_hexes.gpkg. `countries.py` uses it to weight where a community's dots
land inside its own polygon, never to change how many there are.

**THE UNITS ARE SMALL AND THE GRID STILL EARNS ITS PLACE, FOR ONE REASON.** 5,846 km² over 396
communities is **14.8 km² per unit**, comfortably inside [[reference_kontur_resolution_floor]]:
a 400 m hexagon is about 0.14 km², a hundredth of the counting tier, so the grid is finer than
the thing it is placing and not the other way round. What it buys is not the cities, which are
built up all through their polygons, but the **Troodos and the Pafos hinterland**, where a
community is a mountain valley of 10 to 40 km² with everybody in one village at the bottom of
it. Cyprus draws only about 920 dots at 1:1,000, so a single dot landing on a ridge instead of
a village is a visible error rather than a rounding one.

**THE JOIN IS SPATIAL, ON HEX CENTROIDS.** A hexagon on a community line belongs wholly to one
side, so nothing is split and nobody is counted twice.

**KONTUR'S CY EXTRACT IS THE WHOLE ISLAND AND MOST OF WHAT IS DROPPED IS NOT AN ERROR.** The
north is in the file and is not in the census, so its hexes fall outside every one of the 396
polygons and go. That is a large share to discard and it is expected; the check is therefore
not "almost nothing was lost" but that what is KEPT is close to the census's own 923,381.
Kontur is modelled from GHSL, HRSL and building footprints and is not a census, so it is never
asserted equal to one (§12, North Macedonia) and is used only as a within-community weight.

Usage:
    python sources/cy_grid.py --fetch    one ~2 MB gzipped gpkg from Kontur
    python sources/cy_grid.py            rebuild from data/raw/cy/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cy")
GEO = os.path.join(ROOT, "data", "geo", "cy")
UNITS = os.path.join(GEO, "cy_lau.gpkg")
OUT = os.path.join(GEO, "cy_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CY_20231101.gpkg"

EXPECTED_UNITS = 396

# The census's own enumerated population, which is what should survive the clip. Kontur's
# vintage is 2023-11 against a 2021 census, and Cyprus grew, so it should read slightly HIGH.
CENSUS_POPULATION = 923_381
KONTUR_TOLERANCE = 0.30

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def _census_population():
    """community code -> the 2021 census population, from the same cube sources/cy.py reads."""
    import itertools
    import json

    p = os.path.join(RAW, "cit_comm.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run `python sources/cy.py --fetch` first")
    with open(p, encoding="utf-8") as fh:
        js = json.load(fh)
    ids = js["id"]
    dims = []
    for d in ids:
        cat = js["dimension"][d]["category"]
        idx = cat["index"]
        keys = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
        dims.append([(k, cat["label"][k]) for k in keys])
    out = {}
    for i, combo in enumerate(itertools.product(*dims)):
        key = dict(zip(ids, combo))
        code = key["DISTRICT, MUNICIPALITY/COMMUNITY"][0]
        if (len(code) == 4 and code.isdigit()
                and key["CITIZENSHIP GROUP"][1] == "Total"
                and key["SEX"][1] == "Total"):
            out[code] = js["value"][i] or 0
    return out


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/cy_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f} (the whole island)")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    total = float(pts[popcol].sum())
    print(f"\n  hexes outside every enumerated community: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / total:.1f}%)")
    print("     almost all of it is the north, which the census does not cover; dropped.")

    keep = ~outside
    kept_pop = float(pts.loc[keep, popcol].sum())
    ratio = kept_pop / CENSUS_POPULATION
    print(f"  kept {kept_pop:,.0f} against the census's {CENSUS_POPULATION:,} "
          f"({ratio:.3f}x)")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"!! Kontur inside the drawn area is {ratio:.3f}x the census, "
                         f"outside the {KONTUR_TOLERANCE:.0%} band -- the clip is wrong, "
                         "not the model")

    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
        crs=units.crs)
    out = out[out["pop"] > 0].reset_index(drop=True)

    covered = out["unit"].nunique()
    print(f"  {len(out):,} hexes over {covered} of {EXPECTED_UNITS} communities")

    # ANY ENUMERATED COMMUNITY WITH NO POPULATED HEXAGON GETS ITS OWN POLYGON BACK.
    # `countries.py` points `place` at THIS layer, so a community missing from it is not
    # placed on its polygon instead -- it is not placed at all, and scatter.py carries its
    # people into other units of the same node, which puts them in the wrong village rather
    # than in none. On the 2023-11 extract this is Akrotiri (5200, 783 people), inside the
    # Western Sovereign Base Area, where Kontur models nobody. The polygon is real and the
    # census count is real, so the fallback invents no geometry and no population.
    empty = sorted(set(units["unit"]) - set(out["unit"]))
    if empty:
        pop_by_unit = _census_population()
        print(f"     {len(empty)} enumerated communities have no populated hexagon: "
              + ", ".join(f"{u} ({pop_by_unit.get(u, 0):,})" for u in empty[:8]))
        fill = units[units["unit"].isin(empty)].copy()
        fill["pop"] = fill["unit"].map(pop_by_unit).astype(float)
        if fill["pop"].isna().any() or (fill["pop"] <= 0).any():
            raise SystemExit("!! a community with no hexagon has no census population "
                             "either, which should be impossible")
        out = gpd.GeoDataFrame(
            __import__("pandas").concat([out, fill[["unit", "pop", "geometry"]]],
                                        ignore_index=True),
            geometry="geometry", crs=units.crs)
        print("     added their LAU polygons at their census population instead")
    if out["unit"].nunique() != EXPECTED_UNITS:
        raise SystemExit(f"!! {out['unit'].nunique()} communities in the placement layer, "
                         f"expected {EXPECTED_UNITS}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="cy_hexes")
    print(f"\nwrote {OUT}  ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
