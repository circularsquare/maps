"""Build Turkmenistan's two boundary levels from the Geofabrik OSM extract.

Reads  helper1m/data/turkmenistan/raw/turkmenistan-261005.osm.pbf  (OSM data to 2026-10-05)
Writes helper1m/data/turkmenistan/boundaries/adm2.gpkg  (etraps and cities, 53 units)
       helper1m/data/turkmenistan/boundaries/adm1.gpkg  (velayats + Ashgabat + Arkadag, 7)

OSM tags Turkmenistan's etraps and cities of velayat subordination at
admin_level 5 (49 relations, one of them Arkadag) and the five velayats and
Ashgabat at level 4. Ashgabat's four etraps are level 7. The level-5 map is the
current one: it has the etraps created after the 2022 census (Altyn asyr,
Döwletli, Farap, Garabekewül, Oguzhan).

adm2 = the level-5 units plus Ashgabat's four etraps. adm1 is dissolved from
adm2, so the levels nest exactly, with Arkadag as its own top-level unit (a
city with velayat status since 2023).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ["OSM_USE_CUSTOM_INDEXING"] = "NO"

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "turkmenistan" / "raw"
PBF = RAW / "turkmenistan-261005.osm.pbf"
OUT = HELPER / "data" / "turkmenistan" / "boundaries"

# Top-level units: code (ISO 3166-2 where one exists), English name, OSM name.
REGIONS = [
    ("TM-S", "Ashgabat", "Aşgabat"),
    ("TM-AR", "Arkadag", "Arkadag"),
    ("TM-A", "Ahal", "Ahal welaýaty"),
    ("TM-B", "Balkan", "Balkan welaýaty"),
    ("TM-D", "Dashoguz", "Daşoguz welaýaty"),
    ("TM-L", "Lebap", "Lebap welaýaty"),
    ("TM-M", "Mary", "Mary welaýaty"),
]

TK2EN = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ý": "y", "ň": "n", "w": "v",
                       "Ä": "A", "Ö": "O", "Ü": "U", "Ý": "Y", "Ň": "N", "W": "V"})


def english(tk):
    """Turkmen Latin -> the census's English spelling (Wekilbazar -> Vekilbazar)."""
    s = tk.replace("ş", "sh").replace("Ş", "Sh").replace("ç", "ch").replace("Ç", "Ch")
    s = s.replace("ž", "zh").replace("Ž", "Zh")
    s = s.translate(TK2EN)
    s = re.sub(r"\s+etraby$", " etrap", s)
    s = re.sub(r"\s+shaheri$", " city", s)
    return s


def tag(other, key):
    m = re.search(rf'"{re.escape(key)}"=>"([^"]*)"', other or "")
    return m.group(1) if m else None


def main():
    g = pyogrio.read_dataframe(PBF, layer="multipolygons",
                               where="boundary='administrative' AND admin_level IN ('2','4','5','7')")
    g = g[g.osm_id.notna()].to_crs(4326)
    g["geometry"] = g.geometry.make_valid()
    nat = g[(g.admin_level == "2") & (g.name == "Türkmenistan")]
    assert len(nat) == 1, nat[["name"]]
    natg = nat.geometry.iloc[0]
    eqa = lambda geom: gpd.GeoSeries([geom], crs=4326).to_crs("ESRI:54009").area.iloc[0] / 1e6

    vel = g[g.admin_level == "4"].set_index("name")
    l5 = g[g.admin_level == "5"].copy()
    # drop Afghanistan's Ghormach and anything else outside the country
    l5 = l5[l5.geometry.apply(lambda x: eqa(x.intersection(natg)) > 0.5 * eqa(x))]
    l7 = g[g.admin_level == "7"].copy()
    ash = vel.loc["Aşgabat"].geometry
    l7 = l7[l7.geometry.apply(lambda x: eqa(x.intersection(ash)) > 0.5 * eqa(x))]
    assert len(l7) == 4, l7.name.tolist()

    rows = []
    for r in l5.itertuples():
        if r.name == "Arkadag":
            reg = "TM-AR"
        else:
            best = max(((code, eqa(r.geometry.intersection(vel.loc[tkn].geometry)))
                        for code, _, tkn in REGIONS if code not in ("TM-AR", "TM-S")),
                       key=lambda t: t[1])
            reg = best[0]
        rows.append({"osm_id": int(r.osm_id), "name_tk": r.name, "reg": reg, "geometry": r.geometry})
    for r in l7.itertuples():
        rows.append({"osm_id": int(r.osm_id), "name_tk": r.name, "reg": "TM-S", "geometry": r.geometry})
    adm2 = gpd.GeoDataFrame(rows, crs=4326)
    adm2["name"] = adm2.name_tk.map(english)
    # Arkadag sits inside Ahal's level-4 polygon in OSM; clip nothing, just check overlap.
    adm2 = adm2.sort_values(["reg", "name"]).reset_index(drop=True)
    adm2["code"] = adm2.groupby("reg").cumcount().add(1).map("{:02d}".format)
    adm2["code"] = adm2.reg + "-" + adm2.code
    adm2.loc[adm2.reg == "TM-AR", "code"] = "TM-AR"
    adm2["parent"] = adm2.reg
    adm2["group"] = adm2.reg
    adm2["name_cn"] = adm2.name_tk          # extra_cols slot: the Turkmen name

    # gaps and overlaps against the country outline
    eq = adm2.to_crs("ESRI:54009")
    tot = eq.area.sum() / 1e6
    uni = shapely.union_all(eq.geometry.values)
    natarea = eqa(natg)
    print(f"adm2: {len(adm2)} units, sum {tot:,.0f} km2, union {uni.area / 1e6:,.0f} km2, "
          f"country {natarea:,.0f} km2 (overlap {tot - uni.area / 1e6:,.1f}, "
          f"gap {natarea - uni.area / 1e6:,.1f})")
    for code, _, tkn in REGIONS:
        if code == "TM-AR":
            continue
        u = adm2[adm2.reg == code]
        if code == "TM-A":
            u = adm2[adm2.reg.isin(["TM-A", "TM-AR"])]
        a = eqa(shapely.union_all(u.geometry.values))
        print(f"  {code}: units {a:,.0f} km2 vs OSM level-4 {eqa(vel.loc[tkn].geometry):,.0f} km2")

    adm2["km2"] = (eq.area / 1e6).round(1)
    reg = {r[0]: r for r in REGIONS}
    adm1 = adm2.dissolve(by="group", as_index=False)[["group", "geometry"]]
    adm1["code"] = adm1["group"]
    adm1["name"] = adm1.code.map(lambda c: reg[c][1])
    adm1["name_cn"] = adm1.code.map(lambda c: reg[c][2])
    adm1 = adm1[["code", "name", "name_cn", "group", "geometry"]]

    OUT.mkdir(parents=True, exist_ok=True)
    adm2[["code", "name", "name_cn", "parent", "group", "osm_id", "km2", "geometry"]].to_file(
        OUT / "adm2.gpkg", driver="GPKG")
    adm1.to_file(OUT / "adm1.gpkg", driver="GPKG")
    pd.set_option("display.width", 200)
    print(adm2[["code", "name", "name_tk", "osm_id", "km2"]].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
