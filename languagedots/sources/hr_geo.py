"""Croatia: the placement layer, Kontur 400 m hexes keyed to the 556 towns and municipalities.

    python sources/hr_geo.py      -> data/geo/hr/hr_hexes.gpkg

Units are religiondots' data/geo/hr/hr_opcine.gpkg (GISCO LAU 2021, 556 polygons keyed by the
LAU code `kod`), read only. religiondots' hr_lookup.csv (also read only) routes every one of the
census's 572 rows (555 municipalities and the 17 city districts of Zagreb) to a `kod`; Zagreb's
districts all go to 01333, because no boundary file for them was found (religiondots'
sources/hr_geo.md section 4). The census totals per `kod` come from data/normalized/hr.csv.

religiondots places Croatia on the bare polygons; this map needs a population weight inside
them, so the hexes are new here. Kontur HR is not in religiondots' kontur folder, so _grid.py
downloads it into languagedots' data/geo/kontur/.

HEXES OUTSIDE EVERY UNIT (2,175 hexes, 135,973 people, 3.5% of Kontur's Croatia). GISCO's
coastline is generalised, so coastal cities (Rijeka, Split, Zadar, Dubrovnik) lose hexes to the
sea; and Kontur's extract runs over the land border, where a hex outside the units is the
neighbour's town (playbook, "Across a land border, a hex outside the units is the neighbour's
town": Gradiska, Kozarska Dubica, Novi Grad, Brcko and Orasje in Bosnia sit on the Sava and Una
opposite Croatian towns, Barcs in Hungary on the Drava). Each outside hex is classed by its
centroid against Natural Earth 10m (religiondots' copy, read only):
  in the sea (no country)          -> snapped to the nearest unit within SNAP_M (75,409 people)
  in a neighbour country           -> dropped (35,713)
  in Natural Earth's Croatia       -> dropped within NB_M of a neighbour country (Natural
                                      Earth's border is too coarse to trust there), else
                                      snapped within SNAP_M (the GISCO coast stopping short)
Hexes whose centroid is inside a unit are kept as joined: GISCO's border is finer than Natural
Earth's, which puts most of Metkovic in Bosnia.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402
from _grid import kontur_path  # noqa: E402

UNITS = RD_GEO / "hr" / "hr_opcine.gpkg"
LOOKUP = RD_GEO / "hr" / "hr_lookup.csv"
NORM = ROOT / "data" / "normalized" / "hr.csv"
OUT = ROOT / "data" / "geo" / "hr" / "hr_hexes.gpkg"
N_UNITS = 556
NE = RD_GEO / "ne_10m_admin_0_countries.geojson"
NEIGHBOURS = ["SVN", "HUN", "SRB", "BIH", "MNE", "ITA"]
SNAP_M = 1000
NB_M = 2000
CRS_M = 3765                     # HTRS96 / Croatia TM


def census_by_kod():
    import pandas as pd
    df = pd.read_csv(NORM)
    df = df[df["geo_level"].isin(["municipality", "city_district"])
            & (df["source_category"] == "Total")]
    lut = pd.read_csv(LOOKUP, dtype=str)
    m = dict(zip(lut["geo_id"], lut["kod"]))
    df["kod"] = df["geo_id"].map(m)
    if df["kod"].isna().any():
        raise SystemExit(f"{int(df['kod'].isna().sum())} census rows have no LAU code in "
                         "religiondots' hr_lookup.csv")
    out = df.groupby("kod")["count"].sum()
    if len(out) != N_UNITS:
        raise SystemExit(f"census covers {len(out)} LAU codes, expected {N_UNITS}")
    return out


def main():
    import math
    import random

    import geopandas as gpd

    units = gpd.read_file(UNITS)[["kod", "name", "geometry"]].rename(columns={"kod": "unit"})
    units["unit"] = units["unit"].astype(str)
    if len(units) != N_UNITS or units["unit"].nunique() != N_UNITS:
        raise SystemExit(f"{UNITS}: {len(units)} polygons, {units['unit'].nunique()} codes")
    census = census_by_kod()
    miss = sorted(set(census.index) ^ set(units["unit"]))
    if miss:
        raise SystemExit(f"census and polygons disagree on {len(miss)} codes: {miss[:8]}")
    print(f"  {N_UNITS} polygons and {N_UNITS} census units, matched both ways by LAU code")

    hexes = gpd.read_file(kontur_path("hr"))
    if len(hexes) == 0:
        raise SystemExit("Kontur HR extract has ZERO features")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    um = units.to_crs(CRS_M)
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(CRS_M)
    j = gpd.sjoin(pts, um[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    outside = j["unit"].isna()
    print(f"  Kontur HR: {len(hexes):,} hexes, {pts['pop'].sum():,.0f} people; "
          f"{int(outside.sum()):,} hexes ({pts.loc[outside, 'pop'].sum():,.0f} people) outside "
          "every unit")

    # class the outside hexes by Natural Earth: sea, a neighbour, or Croatia (see the docstring)
    o = pts[outside]
    ne = gpd.read_file(NE)
    ne = ne[ne["ADM0_A3"].isin(["HRV"] + NEIGHBOURS)][["ADM0_A3", "geometry"]].to_crs(CRS_M)
    oc = gpd.sjoin(o, ne, how="left", predicate="within")
    oc = oc[~oc.index.duplicated(keep="first")].reindex(o.index)
    where = oc["ADM0_A3"].fillna("sea")
    nb = ne[ne["ADM0_A3"] != "HRV"].geometry.union_all()
    d_nb = o.geometry.distance(nb)
    cls = where.where(where.isin(["sea", "HRV"]), "neighbour")
    cls[(cls == "HRV") & (d_nb < NB_M)] = "HRV near a neighbour"
    for k in ("sea", "HRV", "HRV near a neighbour", "neighbour"):
        s = cls == k
        print(f"    outside, {k:<21} {int(s.sum()):>5,} hexes {o.loc[s, 'pop'].sum():>9,.0f} people")
    cand = o[cls.isin(["sea", "HRV"])]
    near = gpd.sjoin_nearest(cand, um[["unit", "geometry"]], how="left", max_distance=SNAP_M,
                             distance_col="dist")
    near = near[~near.index.duplicated(keep="first")]
    snapped = near["unit"].notna()
    j.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
    dropped = o["pop"].sum() - near.loc[snapped, "pop"].sum()
    print(f"  snapped {int(snapped.sum()):,} hexes ({near.loc[snapped, 'pop'].sum():,.0f} people) "
          f"within {SNAP_M} m; dropped the other {len(o) - int(snapped.sum()):,} "
          f"({dropped:,.0f} people)")
    if not (60_000 < near.loc[snapped, "pop"].sum() < 100_000):
        raise SystemExit("the snapped population moved far from the 2026-10-05 build's 89,973; "
                         "look at the classes above before trusting it")
    by_dist = near.loc[snapped].sort_values("pop", ascending=False).head(8)
    for i, r in by_dist.iterrows():
        c = gpd.GeoSeries([r.geometry], crs=CRS_M).to_crs(4326).iloc[0]
        print(f"      {r['pop']:>8,.0f} people  {r['dist']:>5.0f} m  -> {r['unit']}  "
              f"({c.y:.4f}, {c.x:.4f})")

    keep = j["unit"].notna().to_numpy()
    layer = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                              "pop": pts.loc[keep, "pop"].to_numpy()},
                             geometry=hexes.geometry[keep].to_numpy(), crs=hexes.crs).to_crs(4326)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"]) - set(per.index[per > 0]))
    print(f"  {len(empty)} units with no populated hex: {empty[:8]}")
    if empty:
        raise SystemExit("a unit with no populated hex draws nothing; give it its polygon")

    rows = [(u, float(census[u]), float(per.get(u, 0.0))) for u in census.index if census[u] > 0]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    norm = sorted(((k / c / ratio), u) for u, c, k in rows)
    names = dict(zip(units["unit"], units["name"]))
    print(f"  Kontur / census nationally {ratio:.3f}; per unit, normalised: "
          f"p10 {norm[len(norm) // 10][0]:.2f}  median {norm[len(norm) // 2][0]:.2f}  "
          f"p90 {norm[9 * len(norm) // 10][0]:.2f}")
    print("  lowest: " + ", ".join(f"{names[u]} {r:.2f}" for r, u in norm[:6]))
    print("  highest: " + ", ".join(f"{names[u]} {r:.2f}" for r, u in norm[-6:]))
    band = 3.0
    out_band = [(r, u) for r, u in norm if not (1 / band <= r <= band)]
    print(f"  {len(out_band)} of {len(norm)} units outside a factor of {band:g}: "
          + ", ".join(f"{names[u]} {r:.2f}" for r, u in out_band[:10]))
    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(max(k, 1.0)) for _, _, k in rows]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("the join is not carrying information")

    area = um.geometry.area / 1e6
    print(f"  median unit {area.median():.1f} km2, {area.median() / 0.74:.0f} hexes; "
          f"median hexes per unit in the layer {layer.groupby('unit').size().median():.0f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(layer):,} hexes, {layer['pop'].sum():,.0f} people)")


if __name__ == "__main__":
    main()
