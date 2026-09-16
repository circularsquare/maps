"""Somalia: the placement layer, Kontur 400 m population hexagons keyed to region.

Writes data/geo/so/so_hexes.gpkg. Copied in shape from `sources/sd_grid.py`; `sources/so.md` §5 is
the record.

Bari, Sanaag, Sool, Nugaal and Mudug are the arid north-east, 245,000 km2 between them. Spread flat,
their dots would sit on empty plateau; with Kontur an empty hex takes no dots.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every region is snapped to the nearest
region within `SNAP_KM` when it holds people, and dropped beyond it. Every hex outside is also
classed by the Natural Earth country it falls in, so the dropped people can be told apart: Kontur's
`SO` extract runs over the Ethiopian, Kenyan and Djiboutian borders, and those people belong to
their own countries.

THREE CHECKS AGAINST THE 2026 PLANNING ESTIMATE:

  * **the national ratio**, Kontur over the estimate, inside `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **the rank witness for the join**: Kontur people per region against the estimate, Spearman,
    against `N_PERM` shuffles (at most `PERM_P` of them may reach it);
  * **no town lost**: every GeoNames seat of `SEAT_MIN_POP` or more, Kontur within 5 and 10 km,
    a hole counted only in a region under `LOW_RATIO` of its share (Morocco's gate).

Usage:
    python sources/so_grid.py --fetch    Kontur SO (8 MB gzipped) and GeoNames SO.zip
    python sources/so_grid.py            rebuild from data/raw/so/
"""

import gzip
import json
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "so")
GEO = os.path.join(ROOT, "data", "geo", "so")
UNITS = os.path.join(GEO, "so_regions.gpkg")
OUT = os.path.join(GEO, "so_hexes.gpkg")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SO_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/SO.zip"
GEONAMES = os.path.join(RAW, "geonames_SO.zip")

EXPECTED_UNITS = 18
EXPECTED_RATIO = 1.0
TOLERANCE = 0.25
N_PERM = 20_000
PERM_P = 0.001
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

SEAT_MIN_POP = 50_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32638"

# OUTSIDE COD-AB. Measured 2026-09-15 (sources/so.md §5): COD-AB's 1984 line leaves Cabudwaaq, a
# Galmudug town on the Ethiopian border, 2 to 7 km outside Galgaduud, and neither Ethiopia's nor
# Kenya's place layer holds those hexes, so dropping them would draw about 115,000 people nowhere.
# The opposite case is a border hex a neighbour already places people on (Wajaale on the Ethiopian
# line, Beled Hawo and Mandera on the Kenyan one): COD-AB puts it outside Somalia and the
# neighbour's boundary file puts it inside, so it is left to the neighbour.
FAR_SNAP_KM = 10.0
OWN_NE = ("SOM", "SOL", "sea")            # Natural Earth's Somalia and Somaliland, and the coast
NEIGHBOUR_LAYERS = ("et", "ke")           # drawn neighbours whose Kontur place layers touch Somalia
MATCH_DEG = 1e-4                          # the same hex: centroids within about 10 m
OUTSIDE_PINNED = {"drop": 8424, "drop: in et_hexes": 84082, "drop: in ke_hexes": 28345,
                  "snap": 370852, "snap far": 162981}   # people per rule, measured 2026-09-15
TOWN_WITNESS = {"Cabudwaaq": 50_000}      # Kontur people within 5 km that the snap must place


def neighbour_held(pts, crs):
    """'et', 'ke' or 'none' for each point: whether that neighbour's place layer has the same hex."""
    import geopandas as gpd
    from scipy.spatial import cKDTree

    p = pts.to_crs(4326)
    xy = np.column_stack([p.geometry.x, p.geometry.y])
    held = pd.Series("none", index=pts.index)
    for cc in NEIGHBOUR_LAYERS:
        path = os.path.join(ROOT, "data", "geo", cc, f"{cc}_hexes.gpkg")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: the border rule needs {cc}'s place layer")
        g = gpd.read_file(path, engine="pyogrio")
        c = g.geometry.to_crs(3857).centroid.to_crs(4326)
        dist, _ = cKDTree(np.column_stack([c.x, c.y])).query(xy)
        hit = (dist < MATCH_DEG) & (held.to_numpy() == "none")
        held[hit] = cc
        print(f"  {cc}_hexes.gpkg: {len(g):,} cells, {int(hit.sum()):,} of these outside hexes among them")
    return held


def town_witness(out):
    """Kontur people within 5 km of each GeoNames town in TOWN_WITNESS, in the layer as written."""
    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("SO.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    for name, floor in TOWN_WITNESS.items():
        m = t[(t["fclass"] == "P") & (t["asciiname"] == name)]
        if len(m) != 1:
            raise SystemExit(f"GeoNames has {len(m)} populated places named {name}")
        lat, lon = float(m["lat"].iloc[0]), float(m["lon"].iloc[0])
        d = km(lat, lon, out["lat"].to_numpy(), out["lon"].to_numpy())
        got = float(out.loc[d <= HOLE_KM, "pop"].sum())
        print(f"  town witness: {name} ({lat:.3f} N, {lon:.3f} E), Kontur within {HOLE_KM:g} km "
              f"{got:,.0f}, floor {floor:,}, in {sorted(set(out.loc[d <= HOLE_KM, 'unit']))}")
        if got < floor:
            raise SystemExit(f"{name} holds {got:,.0f} within {HOLE_KM:g} km; the border snap did not place it")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES, "wb") as fh:
            fh.write(data)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
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
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def ne_countries(crs):
    import geopandas as gpd

    d = json.load(open(NE, encoding="utf-8"))
    rows = [(f["properties"]["ADM0_A3"], f["properties"]["ADMIN"], f["geometry"]) for f in d["features"]
            if f["properties"]["ADM0_A3"] in ("SOM", "SOL", "ETH", "KEN", "DJI", "YEM")]
    return gpd.GeoDataFrame({"a3": [r[0] for r in rows], "admin": [r[1] for r in rows]},
                            geometry=gpd.GeoSeries.from_wkt(
                                [__import__("shapely.geometry", fromlist=["shape"]).shape(r[2]).wkt
                                 for r in rows]), crs=4326).to_crs(crs)


def seat_check(out, units, names, rel):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("SO.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLC places, {j['unit'].nunique()} of {len(units)} "
          "regions hold one")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "pop"].sum()), float(h.loc[d <= WIDE_KM, "pop"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["region"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"  Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of each seat of {SEAT_MIN_POP:,}+, "
          f"all {len(c)}, with the region's own Kontur/estimate ratio:")
    for _i, r in c.iterrows():
        print(f"      {names[r['unit']]:<16} {r['seat']:<16} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   region {r['region']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["region"] < LOW_RATIO), "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a "
                         f"region under {LOW_RATIO} of its share: {sorted(holes)}, not "
                         f"{sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/so_geo.py first")

    hexes = read_layer(gpkg, "Kontur SO")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} regions, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")].reindex(pts.index[outside])
        print(f"\n  hexes whose centroid is outside every region: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance to a region:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        ne = ne_countries(units.crs)
        where = gpd.sjoin(pts.loc[outside], ne, how="left", predicate="within")
        where = where[~where.index.duplicated(keep="first")].reindex(pts.index[outside])
        a3 = where["a3"].fillna("sea")
        held = neighbour_held(pts.loc[outside], units.crs)
        d_km = near["d"] / 1000.0
        rule = pd.Series("drop", index=near.index)
        rule[(held == "none") & (d_km <= SNAP_KM)] = "snap"
        rule[(held == "none") & (d_km > SNAP_KM) & (d_km <= FAR_SNAP_KM) & a3.isin(OWN_NE)] = "snap far"
        rule[held != "none"] = "drop: in " + held[held != "none"] + "_hexes"
        tab = pd.DataFrame({"rule": rule, "ne": a3, "pop": pts.loc[outside, popcol]})
        print(f"  the same hexes by rule and Natural Earth country (snap: within {SNAP_KM:g} km; "
              f"snap far: within {FAR_SNAP_KM:g} km and inside Natural Earth's {'/'.join(OWN_NE)}; "
              "drop: in a neighbour's place layer, or neither):")
        for (r_, n_), r in tab.groupby(["rule", "ne"])["pop"].agg(["size", "sum"]).iterrows():
            print(f"      {r_:<18} {n_:<5} {int(r['size']):>6,} hexes  {r['sum']:>10,.0f} people")
        got = {k: round(float(v)) for k, v in tab.groupby("rule")["pop"].sum().items()}
        if OUTSIDE_PINNED and got != OUTSIDE_PINNED:
            raise SystemExit(f"people per snap rule changed: {got}, pinned {OUTSIDE_PINNED}")
        snapped = rule.str.startswith("snap")
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
    dropped = joined["unit"].isna()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing or (per["sum"] <= 0).any():
        raise SystemExit(f"regions with no populated hex: {missing}")
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["pop"].sum())
    ratio = tot / sum(est.values())
    print(f"\n  Kontur {tot:,.0f} vs the 2026 estimate {sum(est.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the 2026 estimate disagree beyond the band")

    u = sorted(est)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([est[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, estimate) over {len(u)} regions = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten > PERM_P * N_PERM:
        raise SystemExit("the rank witness fails; the region join may be permuted")

    print("\n  per-region Kontur 2023 / 2026 estimate, over the national ratio:")
    rel = {x: (per.loc[x, "sum"] / est[x]) / ratio for x in u}
    for x in sorted(rel, key=rel.get):
        print(f"      {names[x]:<16} {int(per.loc[x, 'size']):>7,} hexes   {rel[x]:5.2f}")

    seat_check(out, units, names, rel)
    town_witness(out)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
