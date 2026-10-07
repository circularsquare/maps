"""Karabakh as it is now: the Azerbaijanis resettled since 2022, as extra units of `countries/az.py`.

    python sources/az_karabakh.py   -> data/normalized/az_karabakh.csv,
                                       data/geo/az/az_plus_hexes.gpkg

WHY. Azerbaijan's 2019 census did not enumerate the ground outside the government's control
(1994-2020), and religiondots and Natural Earth hatch it (`Artsakh`, the 1994-2020 line). Nearly
all of its Armenian population, more than 100,000 people by UNHCR's count, left for Armenia in
September 2023; Armenia's 2022 census predates that. What lives there now is the Azerbaijani
government's resettlement of former internally displaced people ("Great Return"), so that is what
is drawn. The pre-2023 Armenian population is not drawn as current.

THE FIGURES. No census or register publishes the resettled population by settlement. Three
official statements are used:
  * 40,000: "more than 40,000 former internally displaced persons have already returned",
    Hikmet Hajiyev, Assistant to the President, 11 September 2026 (Trend,
    trend.az/azerbaijan/politics/4222940.html). The total drawn.
  * 23,000 of them in the Khankendi, Aghdara and Khojaly districts: "more than 23,000 residents
    have been settled in the districts of Khankendi, Aghdara and Khojaly", Sabuhi Gahramanov,
    deputy Special Representative of the President there, 24 November 2025 (APA,
    en.apa.az/social/...-484490). This may include state employees, not only returnees; it is
    the only zone figure published.
  * the rest, 17,000, in the other districts, spread over their resettled settlements in the
    proportions of the settlement counts compiled by Researching Internal Displacement from
    newspaper reports to January 2025 (Fuzuli city 3,132, Lachin city 2,090, Shusha 1,386,
    Jabrayil 1,346, Aghali 871, Zabukh 823, Sus 215), with Zangilan city, settled in 2025, at 300.
    Shukurbeyli (Jabrayil, 2025-26) is left in Jabrayil's weight: its disc would touch the
    Jabrayil hexes the 2019 census populated.
  Inside the Khankendi-Aghdara-Khojaly zone, Khankendi takes half and the six resettled villages
  and towns named in the convoy reports (Khojaly, Ballija, Aghdara, Talish, Hasanriz,
  Sugovushan) a twelfth each. That split is an assumption; nothing published gives it.
Every row is `modelled`, one language (Azerbaijani): the returnees are former IDPs from these
districts, overwhelmingly Azerbaijani. They were counted by the 2019 census where they lived then
(Baku, Sumgayit, Barda and elsewhere), so these 40,000 are also inside az's 2019 figures; the
double count is 0.4% of Azerbaijan and is said in note_public.

The few dozen Armenians who stayed in Khankendi after 2023 (ICRC reports) are under one dot and
are not drawn.

PLACEMENT: each settlement is a unit of its own, a disc around OpenStreetMap's place point
(found with Nominatim), radius RADIUS_M; nothing finer exists, and Kontur's 2023 grid shows the
pre-2023 population, not this one. The discs are asserted to touch none of religiondots' AZ hexes
(the 2019-populated ground), so no ground is drawn twice. az_plus_hexes.gpkg = those hexes + the
discs; countries/az.py reads it.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

NORM = ROOT / "data" / "normalized" / "az_karabakh.csv"
OUT = ROOT / "data" / "geo" / "az" / "az_plus_hexes.gpkg"
TOTAL, KAK = 40_000, 23_000
RADIUS_M = {"KB-KHANKENDI": 3000, "KB-FUZULI": 2000, "KB-LACHIN": 2000, "KB-SHUSHA": 2000}
R_DEFAULT = 1200

# unit: (place, lat, lon, zone weight); OSM place points via Nominatim, 2026-10-06
KAK_ZONE = {
    "KB-KHANKENDI": ("Khankendi", 39.8182, 46.7511, 6),
    "KB-KHOJALY": ("Khojaly", 39.9097, 46.7944, 1),
    "KB-BALLIJA": ("Ballija", 39.8700, 46.7271, 1),
    "KB-AGHDARA": ("Aghdara", 40.2097, 46.8225, 1),
    "KB-TALISH": ("Talish", 40.3776, 46.7408, 1),
    "KB-HASANRIZ": ("Hasanriz", 40.1615, 46.5151, 1),
    "KB-SUGOVUSHAN": ("Sugovushan", 40.3248, 46.7470, 1),
}
REST = {
    "KB-FUZULI": ("Fuzuli", 39.6007, 47.1486, 3132),
    "KB-LACHIN": ("Lachin", 39.6402, 46.5488, 2090),
    "KB-SHUSHA": ("Shusha", 39.7633, 46.7512, 1386),
    "KB-JABRAYIL": ("Jabrayil", 39.3959, 47.0286, 1346),
    "KB-AGHALI": ("Aghali", 39.1756, 46.7645, 871),
    "KB-ZABUKH": ("Zabukh", 39.5911, 46.5430, 823),
    "KB-SUS": ("Sus", 39.6268, 46.5138, 215),
    "KB-ZANGILAN": ("Zangilan", 39.0864, 46.6564, 300),
}


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    rows = []
    for zone, total in ((KAK_ZONE, KAK), (REST, TOTAL - KAK)):
        w = sum(v[3] for v in zone.values())
        for unit, (name, lat, lon, wt) in zone.items():
            rows.append(dict(geo_level="unit", geo_id=unit, name=name, lat=lat, lon=lon,
                             source_category="Azerbaijani", count=total * wt / w, tier="modelled"))
    df = pd.DataFrame(rows)
    assert abs(df["count"].sum() - TOTAL) < 1e-6
    df.to_csv(NORM, index=False)
    print(df[["geo_id", "name", "count"]].round(0).to_string(index=False))

    pts = gpd.GeoDataFrame(df[["geo_id"]], geometry=[Point(x, y) for x, y in zip(df["lon"], df["lat"])],
                           crs=4326).to_crs(32638)
    pts["geometry"] = [p.buffer(RADIUS_M.get(u, R_DEFAULT)) for p, u in zip(pts.geometry, pts["geo_id"])]
    discs = gpd.GeoDataFrame({"unit": df["geo_id"], "pop": 1.0}, geometry=pts.to_crs(4326).geometry,
                             crs=4326)
    az = gpd.read_file(RD_GEO / "az" / "az_hexes.gpkg")
    hit = gpd.sjoin(discs, az[["unit", "geometry"]].rename(columns={"unit": "az_unit"}),
                    predicate="intersects")
    if len(hit):
        raise SystemExit(f"discs touch 2019-populated hexes: {hit[['unit', 'az_unit']].values.tolist()}")
    print(f"  {len(discs)} settlement discs, none touching the {len(az):,} hexes of az_hexes.gpkg")
    plus = gpd.GeoDataFrame(pd.concat([az[["unit", "pop", "geometry"]], discs], ignore_index=True),
                            crs=4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plus.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {NORM.name} ({TOTAL:,}) and {OUT.name} ({len(plus):,})")


if __name__ == "__main__":
    main()
