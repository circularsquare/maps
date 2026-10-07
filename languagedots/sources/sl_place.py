"""Sierra Leone placement layer: religiondots' sl_hexes (the 14 districts of 2015) with each
hex's 2017 district added.

    python sources/sl_place.py      -> data/geo/sl/sl_hexes.gpkg  (unit, pop, pcode17)

religiondots' layer (read-only) is COD-AB chiefdoms regrouped to the 2015 districts over Kontur
2023 400 m hexes (religiondots/sources/sl_geo.py). Each hex gets the COD-AB ADM2 pcode
(`pcode17`, SL0101...) of the 2017 district its centroid falls in: the pcodes CLEAR Global's
district shares use. Only three 2015 districts hold more than one: Bombali and Port Loko
(Karene, made from both in 2017) and Koinadugu (Falaba). sources/sl_census.py mixes CLEAR's
shares by these pieces; countries/sl.py places each language's dots by them.

CHECKS: every hex gets a 2017 district; each 2015 district holds exactly the 2017 districts it
should (the twelve unchanged ones themselves; Bombali {Bombali, Karene}, Port Loko {Port Loko,
Karene}, Koinadugu {Koinadugu, Falaba}); all 16 are hit; the population is religiondots' to
the person.
"""
import sys
from pathlib import Path

import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "sl" / "shp" / "sle_admin2.shp"
OUT = HERE / "data" / "geo" / "sl" / "sl_hexes.gpkg"

SPLIT = {"Bombali": {"SL0201", "SL0502"}, "Port Loko": {"SL0503", "SL0502"},
         "Koinadugu": {"SL0203", "SL0206"}}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "sl" / "sl_hexes.gpkg")
    adm2 = gpd.read_file(ADM2, engine="fiona")[["adm2_pcode", "adm2_name", "geometry"]]
    adm2 = adm2.to_crs(hexes.crs)
    allowed = {u: SPLIT.get(u) for u in hexes["unit"].unique()}
    by_name = dict(zip(adm2["adm2_name"], adm2["adm2_pcode"]))
    for u in allowed:
        if allowed[u] is None:
            allowed[u] = {by_name[u]}

    pts = hexes.copy()
    pts["geometry"] = hexes.to_crs(3857).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, adm2, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    pc = j["adm2_pcode"].reindex(hexes.index)
    bad = [i for i in hexes.index
           if not isinstance(pc[i], str) or pc[i] not in allowed[hexes.at[i, "unit"]]]
    # centroids offshore or across a line: nearest allowed 2017 district
    if bad:
        a3857 = adm2.to_crs(3857)
        p3857 = pts.loc[bad].to_crs(3857)
        for i, g in p3857.geometry.items():
            cand = a3857[a3857["adm2_pcode"].isin(allowed[hexes.at[i, "unit"]])]
            pc[i] = cand.loc[cand.distance(g).idxmin(), "adm2_pcode"]
    print(f"  {len(bad):,} of {len(hexes):,} hexes put on the nearest allowed 2017 district")
    hexes["pcode17"] = pc.astype(str)
    got = hexes.groupby("unit")["pcode17"].agg(set).to_dict()
    say(got == allowed, "each 2015 district holds exactly its 2017 districts")
    say(hexes["pcode17"].nunique() == 16, "all 16 2017 districts hit")
    rd = gpd.read_file(RD_GEO / "sl" / "sl_hexes.gpkg", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1,
        f"population {hexes['pop'].sum():,.0f} = religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "pcode17", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"wrote {OUT}")
    print(hexes.groupby(["unit", "pcode17"])["pop"].sum().round().astype(int).to_string())


if __name__ == "__main__":
    main()
