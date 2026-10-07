"""Eritrea: the placement layer, Meta's 2020 high-resolution population grid binned into cells of
about 0.75 km2, keyed to zoba and scaled to each zoba's population.

Writes data/geo/er/er_cells.gpkg. `sources/er.md` §5 is the record.

WHY NOT KONTUR. Kontur `ER` (2023-11-01, 3,877,037 people) was read first and rejected: per zoba
it holds 0.45 (Maekel) to 2.68 (Semenawi Keih Bahri) of the survey's share, it puts most of each
secondary town's people in blocks at its 46,200/km2 cap (Massawa 268,920 in 12 hexes, Ghinda
140,872 in 6, Karora 94,801 in 3), and within 5 km of the towns it holds 3 to 13 times GeoNames'
figures (Keren 290,550, Barentu 114,575, Massawa 308,810). Scaling per zoba would still have drawn
Massawa near 100,000 and Karora, a border village, near 30,000.

THE GRID. Data for Good at Meta and CIESIN, *High Resolution Population Density Maps*, Eritrea,
`eri_general_2020.csv` (HDX `highresolutionpopulationdensitymaps-eri`, CC BY 4.0): about 30 m
points, 3,546,847 people inside the zobas. Its split between zobas is no better than Kontur's
(Semenawi Keih Bahri again over twice the survey's share), since neither grid had a zoba count to
fit, but inside a zoba it is closer to the towns' sizes, so after scaling it is used for
placement only:

  * points are binned into `CELL_DEG` squares, each cut to the zoba it falls in, unless the cut
    piece is under `SLIVER` of the square (the coast), where the square is kept;
  * a point outside every zoba is snapped to the nearest within `SNAP_KM` (the coast and islands)
    unless it lies inside a Natural Earth neighbour, and dropped otherwise (the CSV runs on into
    Ethiopia and Sudan; 54,330 people within 2 km are inside Natural Earth's Ethiopia);
  * every cell is scaled to its zoba's population (`sources/er_geo.py`);
  * the town witness: people within 5 km of each GeoNames place of 5,000+, after scaling,
    printed against GeoNames' figure, with `TOWN_LOW` pinned.

Usage:
    python sources/er_grid.py --fetch    Meta's CSV and GeoNames ER.zip into data/raw/er/
    python sources/er_grid.py            rebuild from data/raw/er/
"""

import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "er")
GEO = os.path.join(ROOT, "data", "geo", "er")
UNITS = os.path.join(GEO, "er_zobas.gpkg")
OUT = os.path.join(GEO, "er_cells.gpkg")

META_URL = ("https://data.humdata.org/dataset/5dcd3716-c351-453c-8946-3492f2e81bbd/resource/"
            "253f22d9-b236-46af-ba09-1f5e89c2c9b4/download/eri_general_2020_csv.zip")
META = os.path.join(RAW, "eri_general_2020_csv.zip")
GEONAMES_URL = "https://download.geonames.org/export/dump/ER.zip"
GEONAMES = os.path.join(RAW, "geonames_ER.zip")
NE_COUNTRIES = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

EXPECTED_UNITS = 6
CELL_DEG = 0.008                # about 0.86 x 0.83 km at 15 N, near a Kontur r8 hex
SNAP_KM = 2.0
SLIVER = 0.25                   # a cut cell under this share of its square keeps the square
META_IN_ZOBAS = 3_546_847       # measured 2026-10-03; asserted within 0.5%
SEAT_MIN_POP = 5_000
TOWN_KM = 5.0
# GeoNames places that hold under a tenth of their GeoNames figure within 5 km after scaling.
# Edd (Debubawi Keih Bahri, 11,259 in GeoNames): Meta has 1,358 people within 5 km and Kontur 957;
# both grids agree it is a village, and GeoNames' figure is old. Left.
# Himora (46,100; GeoNames 334717 at 14.304 N, 36.606 E) is Humera, the Ethiopian town across the
# Tekezé from Om Hajer: its point is inside COD-AB's Gash-Barka by a few hundred metres, and the
# Meta people around it are inside Natural Earth's Ethiopia and dropped. Left.
TOWN_LOW = {"Edd", "Himora"}
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32637"


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, path in ((META_URL, META), (GEONAMES_URL, GEONAMES)):
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=900) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{url} is not a zip")
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
        print(f"  fetched {url} ({len(data):,} bytes)")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def town_witness(pts, units, names):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("ER.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fclass"] == "P"].copy()
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    s = s[s["population"] >= SEAT_MIN_POP]
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    g = gpd.sjoin(g, units[["unit", "geometry"]], how="left", predicate="within")
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    raw, sc = pts["meta"].to_numpy(), pts["pop"].to_numpy()
    print(f"\n  people within {TOWN_KM:g} km of each GeoNames place of {SEAT_MIN_POP:,}+ "
          "(GeoNames' figures are mostly old estimates):")
    low = set()
    rows = []
    for _i, r in g.iterrows():
        d = km(r["lat"], r["lon"], lat, lon) <= TOWN_KM
        rows.append((names.get(r["unit"], "?"), r["name"], int(r["population"]),
                     raw[d].sum(), sc[d].sum()))
    for z, n, gp, a, b in sorted(rows, key=lambda x: -x[2]):
        print(f"      {z:<20} {n:<14} GeoNames {gp:>8,}   Meta {a:>9,.0f}   drawn {b:>9,.0f} ({b / gp:.2f})")
        if b < 0.10 * gp:
            low.add(n)
    if low != TOWN_LOW:
        raise SystemExit(f"places drawn under a tenth of GeoNames: {sorted(low)}, pinned {sorted(TOWN_LOW)}")


def main():
    import geopandas as gpd
    from shapely.geometry import box

    if "--fetch" in sys.argv or not (os.path.exists(META) and os.path.exists(GEONAMES)):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/er_geo.py first")
    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} zobas, expected {EXPECTED_UNITS}")
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))

    with zipfile.ZipFile(META) as z:
        t = pd.read_csv(z.open("eri_general_2020.csv"))
    if list(t.columns) != ["longitude", "latitude", "eri_general_2020"]:
        raise SystemExit(f"unexpected columns {list(t.columns)}")
    minx, miny, maxx, maxy = units.total_bounds
    t = t[(t["longitude"] >= minx - 0.1) & (t["longitude"] <= maxx + 0.1)
          & (t["latitude"] >= miny - 0.1) & (t["latitude"] <= maxy + 0.1)]
    pts = gpd.GeoDataFrame({"meta": t["eri_general_2020"].to_numpy(dtype=float),
                            "lon": t["longitude"].to_numpy(), "lat": t["latitude"].to_numpy()},
                           geometry=gpd.points_from_xy(t["longitude"], t["latitude"]), crs=4326)
    u = units[["unit", "geometry"]]
    j = gpd.sjoin(pts, u, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    inside = float(pts.loc[j["unit"].notna(), "meta"].sum())
    print(f"Meta 2020 points in the zobas: {int(j['unit'].notna().sum()):,}, {inside:,.0f} people")
    if abs(inside / META_IN_ZOBAS - 1) > 0.005:
        raise SystemExit(f"expected about {META_IN_ZOBAS:,}")
    out_mask = j["unit"].isna().to_numpy()
    # only points within SNAP_KM of the country can snap; test that against one buffered outline
    # first, since the file's box holds a million points in Ethiopia and Sudan
    ring = gpd.GeoDataFrame(geometry=[u.to_crs(METRIC).union_all().buffer(SNAP_KM * 1000)],
                            crs=METRIC).to_crs(4326)
    cand = gpd.sjoin(pts.loc[out_mask], ring, how="inner", predicate="within").index.unique()
    near = gpd.sjoin_nearest(pts.loc[cand].to_crs(METRIC), u.to_crs(METRIC), how="left",
                             distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    # across a land border the point is the neighbour's town: drop it (playbooks/geography.md,
    # Bhutan and Namibia); only points in no Natural Earth country (the sea, the islands' edges,
    # or where COD-AB's line runs short of Natural Earth's Eritrea) are snapped
    ne = gpd.read_file(NE_COUNTRIES)
    others = ne[ne["ADM0_A3"] != "ERI"][["ADM0_A3", "geometry"]].to_crs(4326)
    hit = gpd.sjoin(pts.loc[near.index], others, how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")].reindex(near.index)
    foreign = hit["ADM0_A3"].notna().to_numpy()
    by_country = hit.loc[foreign].groupby("ADM0_A3")["meta"].sum().round()
    print(f"  outside every zoba, within {SNAP_KM:g} km but inside a neighbour, dropped: "
          + ", ".join(f"{k} {v:,.0f}" for k, v in by_country.items()))
    snap = (near["d"] <= SNAP_KM * 1000).to_numpy() & ~foreign
    j.loc[near.index[snap], "unit"] = near.loc[snap, "unit"].to_numpy()
    print(f"  outside every zoba within {SNAP_KM:g} km, snapped: {pts.loc[near.index[snap], 'meta'].sum():,.0f} "
          f"people; beyond it, dropped (Ethiopia and Sudan in the same file): "
          f"{pts.loc[out_mask, 'meta'].sum() - pts.loc[near.index[snap], 'meta'].sum():,.0f}")
    pts["unit"] = j["unit"]
    pts = pts[pts["unit"].notna()].copy()

    per = pts.groupby("unit")["meta"].sum()
    tot = float(per.sum())
    print(f"\n  per zoba, Meta 2020 share against the drawn share (sources/er_geo.py):")
    for x in sorted(est, key=lambda x: (per[x] / tot) / (est[x] / sum(est.values()))):
        r = (per[x] / tot) / (est[x] / sum(est.values()))
        print(f"      {names[x]:<20} Meta {per[x]:>10,.0f}   drawn {est[x]:>10,}   {r:5.2f}")
    pts["pop"] = pts["meta"] * pts["unit"].map(lambda x: est[x] / per[x]).to_numpy()

    town_witness(pts, units, names)

    # ---- bin into cells, one per (cell, zoba), each cut to its zoba ----
    pts["cx"] = np.floor(pts["lon"].to_numpy() / CELL_DEG).astype(np.int64)
    pts["cy"] = np.floor(pts["lat"].to_numpy() / CELL_DEG).astype(np.int64)
    cells = pts.groupby(["unit", "cx", "cy"], as_index=False)["pop"].sum()
    cells = cells[cells["pop"] > 0]
    geom = [box(cx * CELL_DEG, cy * CELL_DEG, (cx + 1) * CELL_DEG, (cy + 1) * CELL_DEG)
            for cx, cy in zip(cells["cx"], cells["cy"])]
    cells = gpd.GeoDataFrame(cells[["unit", "pop"]], geometry=geom, crs=4326)
    shapes = dict(zip(units["unit"], units.geometry))
    abroad = others.union_all()
    cut = []
    for x, grp in cells.groupby("unit"):
        poly = shapes[x]
        inner = grp.geometry.within(poly)
        g2 = grp.copy()
        g2.loc[~inner, "geometry"] = g2.loc[~inner].geometry.intersection(poly)
        # a snapped coastal point's cell can lie wholly or mostly outside its zoba, which would
        # pile its people on a sliver: keep the square there, less any neighbour's land in it
        # (water.py clips the sea; a border cell keeps only Natural Earth's Eritrean side)
        sliver = g2.geometry.is_empty | (g2.geometry.area < SLIVER * grp.geometry.area)
        g2.loc[sliver, "geometry"] = grp.loc[sliver].geometry.difference(abroad)
        g2.loc[sliver & g2.geometry.is_empty, "geometry"] = grp.loc[sliver & g2.geometry.is_empty].geometry.intersection(poly)
        if g2.geometry.is_empty.any():
            raise SystemExit(f"{int(g2.geometry.is_empty.sum())} cells in {x} have no ground left")
        cut.append(g2)
    cells = pd.concat(cut, ignore_index=True)
    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs=4326)
    chk = cells.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - est[x]) for x in est)
    if worst > 0.5:
        raise SystemExit(f"cells leave a zoba {worst:.2f} people off its population")
    area = cells.to_crs(6933).geometry.area.to_numpy() / 1e6
    dens = cells["pop"].to_numpy() / np.maximum(area, 1e-6)
    print(f"\n  {len(cells):,} cells; median {np.median(area):.2f} km2; densest {dens.max():,.0f}/km2; "
          f"every zoba sums to its population (worst {worst:.3f})")
    os.makedirs(GEO, exist_ok=True)
    cells[["unit", "pop", "geometry"]].to_file(OUT, layer="cells", driver="GPKG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
