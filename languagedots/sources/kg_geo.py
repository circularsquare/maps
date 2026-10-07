"""Kyrgyzstan: the 2022 census's rayons and cities as polygons, and a Kontur placement layer.

    python sources/kg_geo.py      -> data/geo/kg/kg_units.gpkg, kg_lookup.csv, kg_hexes.gpkg

UNITS. COD-AB Kyrgyzstan (religiondots/data/raw/kg/shp, read-only): admin 2 has 53 rayons and
cities of oblast significance; admin 1 has Bishkek and Osh city, which have no admin 2. The census
(sources/kg_census.py) has 52 units in the seven oblasts plus the two cities:
  * COD's Kok-Zhangak (city), inside Suzak rayon on the ground, has no row of its own in table 3.4,
    whose Jalal-Abad cities are Jalal-Abad, Kara-Kul, Mailuu-Suu and Tash-Kumyr: it is dissolved
    into Suzak.
  * Talas's "Aitmatovskiy rayon" is COD's Kara-Buura (renamed in 2022, after COD's vintage).
  * Bishkek's four districts are in the table but not in COD; Bishkek is drawn as one unit (see
    sources/kg.md for what was tried).
  * Kara-Kul city (census 2022: its own row) is a 1.2 km2 polygon in COD, the dam town's core:
    Kontur put 0.02 of its census people in it (normalised), against 0.24-0.30 for the next lowest
    cities, so all its dots would have stacked on one or two hexes. It is drawn together with
    Toktogul rayon, which surrounds it, on the two polygons dissolved (both census rows go to
    Toktogul's pcode in the lookup).
So 53 placement units for 55 census rows. THE JOIN is by name inside each oblast (the census
volume is the oblast), on COD's Russian `adm2_name1`, with the city prefix and the word "rayon"
normalised away; base names are unique inside every oblast. Asserted 1:1 both ways but for the
Kara-Kul + Toktogul pair, which is asserted to be the only pair. Unit ids are COD pcodes.

WITNESS (a second key neither name determines): COD's own areas against the census's rayon sizes
are not printed, so the witness is Kontur: per-unit Kontur over census, normalised nationally,
with the shuffled-join control `_grid.hex_layer` prints.

PLACEMENT. Kontur 2023 r8 hexes keyed by centroid to the 53 units (`sources/_grid.py`). The Kontur
extract is religiondots' (data/raw/kg/), copied into languagedots/data/geo/kontur/ once.
"""
import os
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

SHP = RD / "data" / "raw" / "kg" / "shp"
RD_KONTUR_GZ = RD / "data" / "raw" / "kg" / "kontur_population_KG_20231101.gpkg.gz"
OUR_KONTUR = ROOT / "data" / "geo" / "kontur"
GEO = ROOT / "data" / "geo" / "kg"
NORM = ROOT / "data" / "normalized" / "kg.csv"

BOOK_OBLAST = {"batken": "KG05000000000", "jalalabad": "KG03000000000",
               "issykkul": "KG02000000000", "naryn": "KG04000000000",
               "osh_oblast": "KG06000000000", "talas": "KG07000000000",
               "chui": "KG08000000000"}
CITIES = {"bishkek:г.Бишкек": "KG11000000000", "osh_city:г.Ош": "KG21000000000"}
ALIAS = {"айтматовский": "кара-бууринский"}     # renamed 2022
MERGE = {"KG03220400010": "KG03220000000",      # Kok-Zhangak city -> Suzak rayon
         "KG03440000010": "KG03225000000"}      # Kara-Kul city -> Toktogul rayon, see above
EXPECTED = 53


def key(name):
    s = name.lower().replace("ё", "е")
    city = bool(re.match(r"^\s*г\.", s))
    s = re.sub(r"^\s*г\.\s*", "", s)
    s = re.sub(r"\s+район$", "", s).strip()
    s = ALIAS.get(s, s)
    return ("city:" if city else "") + s


def main():
    import geopandas as gpd

    a1 = gpd.read_file(SHP / "kgz_admin1.shp")
    a2 = gpd.read_file(SHP / "kgz_admin2.shp")
    if len(a1) != 9 or len(a2) != 53:
        raise SystemExit(f"COD-AB: {len(a1)} admin 1, {len(a2)} admin 2; expected 9 and 53")

    a2["unit"] = a2["adm2_pcode"].replace(MERGE)
    units = a2.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    names = a2.set_index("adm2_pcode")
    cap = a1[a1["adm1_pcode"].isin(CITIES.values())][["adm1_pcode", "geometry"]].rename(
        columns={"adm1_pcode": "unit"})
    units = pd.concat([units, cap], ignore_index=True)
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=a2.crs)
    if len(units) != EXPECTED:
        raise SystemExit(f"{len(units)} units, expected {EXPECTED}")

    # the join: census names per oblast -> COD's Russian names in the same oblast
    df = pd.read_csv(NORM)
    census = df[df["geo_level"].isin(["unit", "city"])].groupby(
        ["geo_id", "book", "geo_level"], as_index=False)["count"].sum()
    census = census[~((census["book"] == "bishkek") & (census["geo_level"] == "district"))]
    rows = []
    for _i, r in census.iterrows():
        if r["geo_id"] in CITIES:
            rows.append((r["geo_id"], CITIES[r["geo_id"]], r["count"]))
            continue
        cod = names[names["adm1_pcode"] == BOOK_OBLAST[r["book"]]]
        # on the base name (COD writes Kara-Kul without the city prefix); base names are
        # unique inside every oblast (Batken city "баткен" against Batken rayon "баткенский")
        k = key(r["geo_id"].split(":", 1)[1]).replace("city:", "")
        hit = [p for p, n in cod["adm2_name1"].items() if key(n).replace("city:", "") == k]
        if len(hit) != 1:
            raise SystemExit(f"{r['geo_id']} ({k}): {len(hit)} COD matches {hit}")
        rows.append((r["geo_id"], MERGE.get(hit[0], hit[0]), r["count"]))
    lut = pd.DataFrame(rows, columns=["geo_id", "unit", "pop"])
    dup = lut[lut["unit"].duplicated(keep=False)]
    if set(dup["geo_id"]) != {"jalalabad:г. Кара-Куль", "jalalabad:Токтогульский район"}:
        raise SystemExit(f"two census units on one polygon: {dup.to_string()}")
    if set(lut["unit"]) != set(units["unit"]):
        raise SystemExit(f"census without polygon {set(lut['unit']) - set(units['unit'])}, "
                         f"polygon without census {set(units['unit']) - set(lut['unit'])}")
    if lut["pop"].sum() != 6_936_156:
        raise SystemExit(f"the joined units hold {lut['pop'].sum():,}")
    # print the pairs whose names are not identical after normalising, for the record
    for _i, r in lut.iterrows():
        if r["unit"] in names.index:
            cn, nn = r["geo_id"].split(":", 1)[1], names.loc[r["unit"], "adm2_name1"]
            if re.sub(r"(г\.|район|\s)", "", cn) != re.sub(r"(г\.|район|\s)", "", nn):
                print(f"  joined on a different name: {cn} -> {nn} ({r['unit']})")
    print(f"join: {len(lut)} census units <-> {len(units)} polygons, 1:1 but Kara-Kul + "
          f"Toktogul, {lut['pop'].sum():,} people")

    GEO.mkdir(parents=True, exist_ok=True)
    lut.to_csv(GEO / "kg_lookup.csv", index=False, encoding="utf-8")
    units.to_file(GEO / "kg_units.gpkg", driver="GPKG")
    a = units.to_crs(6933)
    a["km2"] = a.area / 1e6
    small = a.sort_values("km2").head(6)
    print("  smallest units, km2: " + ", ".join(f"{u} {k:.1f}" for u, k in
                                                zip(small["unit"], small["km2"])))

    OUR_KONTUR.mkdir(parents=True, exist_ok=True)
    gz = OUR_KONTUR / RD_KONTUR_GZ.name
    if not gz.exists() and not (OUR_KONTUR / RD_KONTUR_GZ.name[:-3]).exists():
        shutil.copyfile(RD_KONTUR_GZ, gz)
    from _grid import hex_layer
    hex_layer("kg", units, census=lut.groupby("unit")["pop"].sum().to_dict())


if __name__ == "__main__":
    main()
