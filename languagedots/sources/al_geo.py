"""Albania placement layer: religiondots' Kontur hexes (12 qarqe), each tagged with its 2011
administrative unit.

    python sources/al_geo.py      -> data/geo/al/al_hexes.gpkg (unit = qark code, au, pop)

religiondots/data/geo/al/al_hexes.gpkg (read only) is 22,289 Kontur 400 m hexes keyed by INSTAT's
qark code, joined and checked there. This adds `au`, the ID_ADMUNIT of the 2011 commune or
municipality the hex's centroid falls in, from INSTAT's own administrativeunit_p_distrib_2011_view
polygons (fetched by sources/al_census.py). countries/al.py uses it to place Albanian, Greek and
Macedonian inside each qark by the census's own unit counts.

CHECKS: every one of the 373 units gets at least one hex or is reported; a hex whose unit lies in a
different qark than the hex's own (a boundary hex, the two layers being drawn apart) is counted
and keeps its own qark; such hexes get the nearest unit of their OWN qark instead, so a qark's
hexes only ever borrow weights from that qark's units.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from rdlink import RD_GEO  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "al", "administrativeunit_p_distrib_2011_view.geojson")
OUT = os.path.join(ROOT, "data", "geo", "al", "al_hexes.gpkg")


def main():
    import geopandas as gpd
    import numpy as np

    hexes = gpd.read_file(RD_GEO / "al" / "al_hexes.gpkg")
    if len(hexes) != 22_289 or hexes["unit"].nunique() != 12:
        raise SystemExit(f"religiondots' al_hexes: {len(hexes)} hexes, "
                         f"{hexes['unit'].nunique()} units")
    aus = gpd.read_file(RAW)[["ID_ADMUNIT", "CODE_PREFECTURE", "NAME_ADMINUNIT", "P_DISTRIB",
                               "geometry"]]
    if len(aus) != 373 or aus["ID_ADMUNIT"].nunique() != 373:
        raise SystemExit(f"{len(aus)} administrative units")
    m = 32634  # UTM 34N
    hx = hexes.to_crs(m)
    au = aus.to_crs(m)
    pts = gpd.GeoDataFrame(geometry=hx.geometry.centroid, crs=m)

    out_au = np.empty(len(hx), dtype=object)
    moved = 0
    for q, idx in hx.groupby("unit").groups.items():
        mine = au[au["CODE_PREFECTURE"] == q]
        p = pts.loc[idx]
        j = gpd.sjoin(p, mine[["ID_ADMUNIT", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")].reindex(p.index)
        miss = j["ID_ADMUNIT"].isna()
        if miss.any():
            near = gpd.sjoin_nearest(p[miss], mine[["ID_ADMUNIT", "geometry"]], how="left")
            near = near[~near.index.duplicated(keep="first")]
            j.loc[miss, "ID_ADMUNIT"] = near["ID_ADMUNIT"]
            moved += int(miss.sum())
        out_au[hx.index.get_indexer(idx)] = j["ID_ADMUNIT"].to_numpy()
    hexes["au"] = out_au
    if hexes["au"].isna().any():
        raise SystemExit("hexes without a unit")
    print(f"  {len(hexes):,} hexes; {moved:,} centroids outside every unit of their own qark "
          "(coast, border, or the two layers' lines apart) took the nearest unit of that qark")

    per = hexes.groupby("au")["pop"].agg(["sum", "size"])
    none = sorted(set(aus["ID_ADMUNIT"]) - set(per.index))
    print(f"  {len(per)} of 373 units hold a hex; without one: "
          + ", ".join(aus.set_index("ID_ADMUNIT").loc[none, "NAME_ADMINUNIT"]) if none else
          "  every one of the 373 units holds at least one hex")
    zero = per[per["sum"] <= 0]
    if len(zero):
        print(f"  {len(zero)} units whose hexes hold no Kontur population")
    # Kontur 2023-ish against census 2011 per unit: a sanity look, not a gate
    c = aus.set_index("ID_ADMUNIT")["P_DISTRIB"]
    r = (per["sum"] / c.reindex(per.index)).replace([np.inf], np.nan).dropna()
    print(f"  Kontur / census 2011 per unit: median {r.median():.2f}, "
          f"10th pct {r.quantile(.1):.2f}, 90th {r.quantile(.9):.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    hexes[["unit", "au", "pop", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
