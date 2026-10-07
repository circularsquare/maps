"""Tajikistan: the five first-level units, geoBoundaries polygons with the 2020 census's permanent
population in each.

Writes data/geo/tj/tj_units.gpkg and data/geo/tj/tj_lookup.csv. `sources/tj.md` §7 is the record.

  * **boundaries**: geoBoundaries gbOpen TJK ADM1 (commit 9469f09; OpenStreetMap via Wambacher,
    ODbL 1.0, boundary year 2017): Sughd (with the Vorukh and Western Qal'acha exclaves in
    Kyrgyzstan), Gorno-Badakhshan, Khatlon, Dushanbe and the Districts of Republican Subordination.
    COD-AB `cod-ab-tjk` on HDX carries only the p-code workbook and no polygons (checked
    2026-10-03).
  * **population**: 2020 Population and Housing Census, Volume I, *Численность постоянного
    населения по областям, районам, городским поселениям ...* (Agency on Statistics, 2022; on
    `stat.tj` as `jadvali-1 ... 5-hazor-nafar-va-ziyod.pdf`), permanent population, 9,657,005. The
    same table prints the 2010 census column, which is the witness for the 2010 nationality volume
    `sources/tj.py` reads.

The permanent population includes people temporarily away. The census's migration volume (table 6,
`census2020_migration_table6.pdf`) counts 470,973 temporarily absent, 352,681 of them outside the
country (246,133 gone to work). They are Tajik residents who are counted where they live and are
drawn there; the note says so.

DUSHANBE IS REDRAWN FROM OPENSTREETMAP. geoBoundaries' Dushanbe (OSM as of 2017) is 366 km2; the city
gives its own area as 203.18 km2 on 1 July 2019 (`dushanbe.tj/public/ru/pasport-goroda`, read
2026-10-03), and OSM's current relation 7328360 (`polygons.openstreetmap.fr`, saved as
`osm_dushanbe_7328360.geojson`) is 197 km2, 186 km2 of it inside the old polygon. The 180 km2 the old
polygon holds beyond today's line is suburb of Rudaki, Hisor and Varzob, which the census counts in
the Districts of Republican Subordination; Kontur puts 260,000 people there. So Dushanbe is today's
OSM line and the rest of the old polygon goes to the Districts (`DU_KM2` pins both areas).

The join is by name, five rows, with the census's 2010 column as a second witness (it must equal
the 2010 nationality volume's region totals exactly).

Usage:
    python sources/tj_geo.py --fetch    geoBoundaries and the census table into data/raw/tj/
    python sources/tj_geo.py            rebuild from data/raw/tj/
"""

import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tj")
GEO = os.path.join(ROOT, "data", "geo", "tj")
OUT = os.path.join(GEO, "tj_units.gpkg")
LOOKUP = os.path.join(GEO, "tj_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/TJK/ADM1/"
          "geoBoundaries-TJK-ADM1.geojson")
GB = os.path.join(RAW, "geoBoundaries-TJK-ADM1.geojson")
T1_URL = ("https://www.stat.tj/wp-content/uploads/2024/03/jadvali-1.-shumorai-aholii-doimi-dar-"
          "viloyatho-nohiyaho-mahalhoi-shahri-markazi-nohiya-va-mahalhoi-aholinishin-bo-aholii-5-"
          "hazor-nafar-va-ziyod.pdf")
T1 = os.path.join(RAW, "census2020_vol1_table1.pdf")

# unit code -> (geoBoundaries shapeISO, the label that opens the unit's row in Volume I table 1)
UNITS = {
    "TJ-GB": ("Gorno-Badakhshan", r"Вилояти Мухтори\s+Кӯҳистони Бадахшон"),
    "TJ-SU": ("Sughd", r"Вилояти Суғд\s+Согдийская область"),
    "TJ-KT": ("Khatlon", r"Вилояти Хатлон\s+Хатлонская область"),
    "TJ-DU": ("Dushanbe", r"шаҳри Душанбе\s+город Душанбе"),
    "TJ-RA": ("Districts of Republican Subordination",
              r"Шаҳру ноҳияҳои тобеи\s+ҷумҳурӣ\s+Города и районы\s+республиканского\s+подчинения"),
}
TOTAL_2010, TOTAL_2020 = 7_564_502, 9_657_005
METRIC_AREA = "ESRI:54034"
OSM_DU_URL = "https://polygons.openstreetmap.fr/get_geojson.py?id=7328360&params=0"
OSM_DU = os.path.join(RAW, "osm_dushanbe_7328360.geojson")
DU_KM2 = (190, 210)              # OSM relation 7328360, measured 197.4; the city's own 203.18
OLD_DU_KM2 = (355, 375)          # geoBoundaries' Dushanbe, measured 366.1


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst, magic in ((GB_URL, GB, b"{"), (T1_URL, T1, b"%PDF"), (OSM_DU_URL, OSM_DU, b"{")):
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if not data.lstrip().startswith(magic):
            raise SystemExit(f"{url} did not return the expected file (starts {data[:8]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def census_2020():
    """{unit: (people 2010, people 2020)} from Volume I table 1, both sexes, asserted to sum."""
    import fitz

    doc = fitz.open(T1)
    if doc.page_count == 0:
        raise SystemExit(f"{T1} opened with zero pages")
    text = "\n".join(p.get_text() for p in doc)
    text = re.sub(r"[ \t]+", " ", text)
    out = {}
    for unit, (_name, label) in UNITS.items():
        m = re.search(label + r"[^\d]*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
        if not m:
            raise SystemExit(f"Volume I table 1: no row for {unit} ({label!r})")
        b10, m10, f10, b20, m20, f20 = (int(x) for x in m.groups())
        if b10 != m10 + f10 or b20 != m20 + f20:
            raise SystemExit(f"{unit}: sexes do not sum ({m.groups()})")
        out[unit] = (b10, b20)
    if sum(v[0] for v in out.values()) != TOTAL_2010 or sum(v[1] for v in out.values()) != TOTAL_2020:
        raise SystemExit(f"region totals sum to {sum(v[0] for v in out.values()):,} (2010) and "
                         f"{sum(v[1] for v in out.values()):,} (2020)")
    if not re.search(rf"Республика\s+Таджикистан\s+{TOTAL_2010}\s+\d+\s+\d+\s+{TOTAL_2020}", text):
        raise SystemExit("Volume I table 1's national row is not 7,564,502 / 9,657,005")
    return out


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not (os.path.exists(GB) and os.path.exists(T1)):
        fetch()
    g = gpd.read_file(GB)
    if len(g) != 5 or set(g["shapeISO"]) != set(UNITS):
        raise SystemExit(f"geoBoundaries TJK ADM1 is not the five units: {list(g['shapeISO'])}")
    for _i, r in g.iterrows():
        if not r["shapeName"].startswith(UNITS[r["shapeISO"]][0]):
            raise SystemExit(f"{r['shapeISO']} is named {r['shapeName']!r}")
    pop = census_2020()

    # Dushanbe: today's OSM line; the rest of geoBoundaries' polygon to the Districts
    import json
    from shapely.geometry import shape
    osm = gpd.GeoSeries([shape(json.load(open(OSM_DU, encoding="utf-8")))], crs=4326).to_crs(g.crs).iloc[0]
    km2 = lambda geom: float(gpd.GeoSeries([geom], crs=g.crs).to_crs(METRIC_AREA).area.iloc[0] / 1e6)
    i_du = g.index[g["shapeISO"] == "TJ-DU"][0]
    i_ra = g.index[g["shapeISO"] == "TJ-RA"][0]
    old = g.loc[i_du, "geometry"]
    if not (DU_KM2[0] <= km2(osm) <= DU_KM2[1] and OLD_DU_KM2[0] <= km2(old) <= OLD_DU_KM2[1]):
        raise SystemExit(f"Dushanbe areas moved: OSM {km2(osm):.1f}, geoBoundaries {km2(old):.1f} km2")
    others = g.drop(index=[i_du, i_ra]).geometry.union_all()
    if km2(osm.intersection(others)) > 1.0:
        raise SystemExit(f"OSM's Dushanbe overlaps another region by {km2(osm.intersection(others)):.1f} km2")
    g.loc[i_ra, "geometry"] = g.loc[i_ra, "geometry"].union(old).difference(osm)
    g.loc[i_du, "geometry"] = osm
    print(f"  Dushanbe redrawn: geoBoundaries {km2(old):.1f} km2 -> OSM 7328360 {km2(osm):.1f} km2; "
          f"{km2(old.difference(osm)):.1f} km2 to the Districts, {km2(osm.difference(old)):.1f} km2 from them")

    g["unit"] = g["shapeISO"]
    g["name"] = g["unit"].map(lambda u: UNITS[u][0])
    g["pop"] = g["unit"].map(lambda u: pop[u][1]).astype(int)
    g["pop2010"] = g["unit"].map(lambda u: pop[u][0]).astype(int)
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("  unit, 2010 and 2020 census permanent population, area km2 (geoBoundaries):")
    for (_i, r), a in zip(g.iterrows(), area):
        print(f"      {r['unit']}  {r['name']:<38} {r['pop2010']:>9,} {r['pop']:>10,}  {a:>9,.0f}")
    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "pop2010", "geometry"]].to_file(OUT, layer="units", driver="GPKG")
    lut = g[["unit", "name", "pop", "pop2010"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (5 units, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
