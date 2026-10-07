"""Turks and Caicos Islands: first language from the 2012 census's country of citizenship,
national, placed by island -> data/normalized/tc.csv and data/geo/tc/tc_hexes.gpkg.

    python sources/tc_census.py [--fetch]

NO CENSUS LANGUAGE QUESTION (2012). Built as the Bahamas (sources/bs.md): citizens on the
national creole, foreign nationals on their country's language. Every row `derived`.

TABLES (Department of Statistics, gov.tc/stats/statistics/social/5-population; each table is a
public Google Sheet, fetched as CSV to data/raw/tc/):
- "Population by Country of Citizenship 1990-2012": 2012 total 31,458; TCI 12,239, Haiti 10,928,
  Dominican Republic 1,541, USA 874, Bahamas 551, Canada 423, England 384, Other 4,518.
- "Population by Island 1960-2012": Providenciales 23,769, Grand Turk 4,831, North Caicos 1,312,
  South Caicos 1,139, Middle Caicos 168, Parrot Cay 131, Salt Cay 108.
Citizenship is national only, so every island takes the national mix (one unit per island keeps
each island's population right; the mix inside is the same everywhere).

GEOGRAPHY: no religiondots layer and no geoBoundaries ADM1 for TCA. Kontur hexes (sources/_grid.py)
keyed to islands by the nearest island point (a Voronoi split of the islands' coordinates; the
islands are separate land masses except North and Middle Caicos, joined by a causeway).
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))

RAW = HERE / "data" / "raw" / "tc"
OUT = HERE / "data" / "normalized" / "tc.csv"
SHEETS = {"tc_citizenship_1990_2012.csv": "1j7bre6UenqqP5bSY92MSsTpdqiwEvy7HDOk_khxZlyo",
          "tc_island_1960_2012.csv": "17c42SlkObjMqAx8L8Q_226JqPZ4o4q_AA8ql01tGAN8"}
TOTAL = 31_458
# island -> (unit, (lon, lat)) of its main settlement
ISLANDS = {"Providenciales": ("TC-PR", (-72.25, 21.78)), "Grand Turk": ("TC-GT", (-71.14, 21.47)),
           "North Caicos": ("TC-NC", (-71.97, 21.88)), "Middle Caicos": ("TC-MC", (-71.73, 21.82)),
           "South Caicos": ("TC-SC", (-71.52, 21.51)), "Parrot Cay": ("TC-PC", (-72.06, 21.93)),
           "Salt Cay": ("TC-SL", (-71.20, 21.33))}
CR = "creole.english_based."
NODE = {"TCI": CR + "turks_caicos", "Haiti": "creole.french_based.haitian",
        "Domincan Republic": "indoeuropean.romance.spanish", "Bahamas": CR + "bahamian",
        # as Barbados, Antigua and Bermuda: English for US, Canadian and British nationals
        "USA": "indoeuropean.germanic.english", "Canada": "indoeuropean.germanic.english",
        "England": "indoeuropean.germanic.english",
        "Other": "other"}   # unnamed pooled countries


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for fn, key in SHEETS.items():
        url = f"https://docs.google.com/spreadsheets/d/{key}/export?format=csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        (RAW / fn).write_bytes(urllib.request.urlopen(req, timeout=60).read())


def num(s):
    return int(str(s).replace(",", ""))


def main():
    if "--fetch" in sys.argv:
        fetch()
    c = pd.read_csv(RAW / "tc_citizenship_1990_2012.csv")
    c = dict(zip(c.iloc[:, 0].str.strip(), c["NUMBER 2012"].map(num)))
    assert c.pop("Total") == TOTAL and sum(c.values()) == TOTAL, c
    assert set(c) == set(NODE), set(c) ^ set(NODE)

    d = pd.read_csv(RAW / "tc_island_1960_2012.csv")
    d.columns = [str(x).strip() for x in d.columns]
    col = next(x for x in d.columns if "2012" in x)
    # the sheet spells "MIddle Caicos"
    isl = {str(k).strip().title(): num(v) for k, v in zip(d.iloc[:, 0], d[col]) if pd.notna(v)}
    assert isl["Total"] == TOTAL
    pop = {k: isl[k] for k in ISLANDS}
    assert sum(pop.values()) == TOTAL, (pop, sum(pop.values()))

    rows = []
    for k, n in pop.items():
        for lab, m in c.items():
            rows.append(dict(geo_id=ISLANDS[k][0], geo_name=k, source_category=lab,
                             count=n * m / TOTAL))
    df = pd.DataFrame(rows)
    df["geo_level"] = "island"
    df["tier"] = "derived"
    df["year"] = 2012
    assert abs(df["count"].sum() - TOTAL) < 1e-6
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people")

    import geopandas as gpd
    from shapely.geometry import MultiPoint, box
    from shapely.ops import voronoi_diagram
    from _grid import hex_layer
    pts = {u: xy for u, xy in ISLANDS.values()}
    frame = box(-72.8, 21.0, -70.8, 22.3)
    cells = voronoi_diagram(MultiPoint(list(pts.values())), envelope=frame)
    units = []
    for poly in cells.geoms:
        u = next(u for u, xy in pts.items() if poly.contains(gpd.points_from_xy([xy[0]], [xy[1]])[0]))
        units.append(dict(unit=u, geometry=poly.intersection(frame)))
    units = gpd.GeoDataFrame(units, crs=4326)
    assert sorted(units["unit"]) == sorted(pts)
    hex_layer("tc", units, census={ISLANDS[k][0]: n for k, n in pop.items()})


if __name__ == "__main__":
    main()
