"""Nigeria placement layer: religiondots' ng_hexes (37 states) with each hex's LGA added.

    python sources/ng_place.py      -> data/geo/ng/ng_hexes.gpkg  (unit, pop, lga)

religiondots' layer (read-only) is COD-AB ADM1 over Kontur 2023 400 m hexes, `unit` = the
state's pcode (NG001..NG037). Each hex gets the COD-AB ADM2 pcode (`lga`, NG001001...) of the
LGA its centroid falls in. countries/ng.py weights each language's dots inside a state by the
Afrobarometer's own respondents per LGA (data/normalized/ng_lga.csv). The counts stay the
state's.

CHECKS: every hex gets an LGA of its own state; all 774 LGAs are hit, or the misses are listed;
the population is religiondots' to the person.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "ng" / "shp" / "nga_admin2.shp"
OUT = HERE / "data" / "geo" / "ng" / "ng_hexes.gpkg"


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "ng" / "ng_hexes.gpkg", engine="pyogrio")
    say(hexes["unit"].nunique() == 37, f"{len(hexes):,} hexes in 37 states")
    adm2 = gpd.read_file(ADM2, engine="pyogrio")[["adm2_pcode", "adm1_pcode", "geometry"]]
    adm2 = adm2.to_crs(hexes.crs)
    say(len(adm2) == 774, "774 LGAs")

    pts = hexes[["unit", "geometry"]].copy()
    pts["geometry"] = hexes.to_crs(3857).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, adm2, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    lga = j["adm2_pcode"].reindex(hexes.index)
    bad = lga.isna() | (j["adm1_pcode"].reindex(hexes.index) != hexes["unit"])
    # centroids off the ADM2 polygons (coast, border) or across a state line: the nearest LGA
    # of the hex's own state
    if bad.any():
        p3857 = pts.loc[bad].to_crs(3857)
        a3857 = adm2.to_crs(3857)
        for st, grp in p3857.groupby("unit"):
            cand = a3857[a3857["adm1_pcode"] == st]
            near = gpd.sjoin_nearest(grp[["geometry"]], cand, how="left")
            near = near[~near.index.duplicated(keep="first")]
            lga.loc[near.index] = near["adm2_pcode"]
    print(f"  {int(bad.sum()):,} of {len(hexes):,} hexes put on the nearest LGA of their own "
          "state (centroid off ADM2 or over a state line)")
    hexes["lga"] = lga.astype(str)
    pa = dict(zip(adm2["adm2_pcode"], adm2["adm1_pcode"]))
    say(hexes["lga"].map(pa).eq(hexes["unit"]).all(), "every hex's LGA lies in its state")
    hit = hexes["lga"].nunique()
    missing = sorted(set(adm2["adm2_pcode"]) - set(hexes["lga"]))
    say(hit >= 770, f"{hit} of 774 LGAs hit (missing: {missing})")
    rd = gpd.read_file(RD_GEO / "ng" / "ng_hexes.gpkg", engine="pyogrio", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1, f"population {hexes['pop'].sum():,.0f} "
        "= religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "lga", "geometry"]].to_file(OUT, driver="GPKG", engine="pyogrio")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
