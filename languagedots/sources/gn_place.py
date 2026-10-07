"""Guinea placement layer: religiondots' gn_hexes (8 régions) with each hex's prefecture added.

    python sources/gn_place.py      -> data/geo/gn/gn_hexes.gpkg  (unit, pop, pref)

religiondots' layer (read-only) is COD-AB ADM1 over Kontur 2023 400 m hexes, `unit` = the
région as the census spells it. Each hex gets the COD-AB ADM2 pcode (`pref`, GN001001...) of the
prefecture its centroid falls in, the same pcodes CLEAR Global's prefecture shares use
(data/normalized/gn_clear.csv). countries/gn.py uses `pref` to weight each language's dots
inside a région by CLEAR's share of that language in each prefecture. The counts stay INS's.

CHECKS: every hex gets a prefecture; every prefecture lies in its hex's région (ADM2's adm1 pcode
= the région's, via religiondots' gn_lookup.csv); all 34 prefectures are hit; the population is
religiondots' to the person.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "gn" / "shp" / "gin_admin2.shp"
OUT = HERE / "data" / "geo" / "gn" / "gn_hexes.gpkg"


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "gn" / "gn_hexes.gpkg")
    lut = pd.read_csv(RD_GEO / "gn" / "gn_lookup.csv", dtype=str)
    reg_pc = dict(zip(lut["unit"], lut["adm1_pcode"]))
    adm2 = gpd.read_file(ADM2, engine="fiona")[["adm2_pcode", "adm1_pcode", "geometry"]]
    adm2 = adm2.to_crs(hexes.crs)

    pts = hexes.copy()
    pts["geometry"] = hexes.to_crs(3857).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, adm2, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    pref = j["adm2_pcode"].reindex(hexes.index)
    own = hexes["unit"].map(reg_pc)
    bad = pref.isna() | (j["adm1_pcode"].reindex(hexes.index) != own)
    # centroids off the ADM2 polygons (coast) or across a région line: nearest prefecture of
    # the hex's own région
    if bad.any():
        p3857 = pts.loc[bad].to_crs(3857)
        a3857 = adm2.to_crs(3857)
        for i, g in p3857.geometry.items():
            cand = a3857[a3857["adm1_pcode"] == own[i]]
            pref[i] = cand.loc[cand.distance(g).idxmin(), "adm2_pcode"]
    print(f"  {int(bad.sum()):,} of {len(hexes):,} hexes put on the nearest prefecture of their "
          "own région (centroid off ADM2 or over a région line)")
    hexes["pref"] = pref.astype(str)
    pa = dict(zip(adm2["adm2_pcode"], adm2["adm1_pcode"]))
    say(hexes["pref"].map(pa).eq(own).all(), "every hex's prefecture lies in its région")
    say(hexes["pref"].nunique() == 34, f"all 34 prefectures hit ({hexes['pref'].nunique()})")
    rd = gpd.read_file(RD_GEO / "gn" / "gn_hexes.gpkg", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1, f"population {hexes['pop'].sum():,.0f} "
        "= religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "pref", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"wrote {OUT}")
    print(hexes.groupby("pref")["pop"].sum().round().astype(int).to_string())


if __name__ == "__main__":
    main()
