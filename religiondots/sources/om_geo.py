"""Oman: boundaries and end-2024 register populations for the 61 wilayat drawn.

Writes data/geo/om/om_units.gpkg and data/geo/om/om_lookup.csv.

  * **boundaries**: geoBoundaries gbOpen `OMN` ADM2 (build 12 December 2023), 61 wilayat traced
    from NCSI's own *Oman National Geographic Structure* (`nsdig2gapps.ncsi.gov.om`), 2020. Licence
    as geoBoundaries records it: "Other - Direct Permission" (WMgeoLab, April 2020), with citation of
    geoBoundaries required.
  * **populations**: NCSI, *Statistical Year Book 2025* (Issue 53), Table 7-2, *Total Population
    Registered in the Sultanate by Nationality, Governorates & Wilayats*, end of December 2024:
    Omani and expatriate per wilayat (`WILAYAT_2024`). `sources/om.py` finds every figure on the
    yearbook's pages 32-33 and holds its 2023 column against GLMM's copy of the same table.

## NOT COD-AB

COD-AB `cod-ab-omn` (GADM lineage) has 11 governorates and 49 districts, and its governorate lines
are not the 2011 governorates: it puts Al Kamil wa al Wafi in Ash Sharqiyah North, draws Ash
Sharqiyah South at 6,376 km2 and Al Buraymi at 4,810 km2, where the register's population over
NCSI's own density map (yearbook p.31: 30.7 and 16.8 persons/km2) gives about 12,000 and 8,100.
`main` prints the COD governorate each wilaya mostly falls in, for the record; nothing uses it.

## TWO WILAYAT NEWER THAN THE BOUNDARIES

The register counts 63 wilayat, the 2020 layer 61. Sinaw (Ash Sharqiyah North) and Al Jabal al
Akhdar (Ad Dakhliyah) have no polygon, and each is merged into the polygon that holds its seat,
taken from GeoNames (`MERGED`, asserted in `main`).

## CHECKS

  1. 61 features in EPSG:4326, names as pinned (`UNITS`), one register row per polygon after the
     two merges;
  2. every governorate's wilayat sum to its row in Table 7-2, and the governorates to the yearbook's
     national totals (2,984,793 Omanis, 2,283,279 expatriates);
  3. the seats of Sinaw and Al Jabal al Akhdar lie in the polygons they are merged into.

**Not checked:** area against NCSI's own figure per wilaya (none was found as a table). The
Kontur rank witness for the name join is in `sources/om_grid.py`.

Usage:
    python sources/om_geo.py --fetch    geoBoundaries geojson (2.3 MB), GeoNames OM.zip, COD-AB zip
    python sources/om_geo.py            rebuild from data/raw/om/
"""

import io
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "om")
OUT_DIR = os.path.join(ROOT, "data", "geo", "om")
OUT_UNITS = os.path.join(OUT_DIR, "om_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "om_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/OMN/ADM2/"
          "geoBoundaries-OMN-ADM2.geojson")
GB = os.path.join(RAW, "geoBoundaries-OMN-ADM2.geojson")
GEONAMES_URL = "https://download.geonames.org/export/dump/OM.zip"
GEONAMES = os.path.join(RAW, "geonames_OM.zip")
COD_URL = ("https://data.humdata.org/dataset/da87f54e-64bd-4cf4-bd31-3fc520f94609/resource/"
           "562549de-c1db-48dd-b50a-d3c27c54c963/download/omn_admin_boundaries.geojson.zip")
COD = os.path.join(RAW, "omn_admin_boundaries.geojson.zip")
EQUAL_AREA = "EPSG:6933"

OMANIS, EXPATRIATES = 2_984_793, 2_283_279

# governorate -> (expatriates, Omanis) at end-2024, Table 7-2's governorate rows
GOVERNORATE_2024 = {
    "Muscat": (914_679, 583_842),
    "Dhofar": (290_763, 238_811),
    "Musandam": (19_087, 36_063),
    "Al Buraymi": (58_581, 77_240),
    "Ad Dakhliyah": (155_665, 404_818),
    "Al Batinah North": (327_867, 597_296),
    "Al Batinah South": (184_893, 381_878),
    "Ash Sharqiyah South": (122_735, 246_793),
    "Ash Sharqiyah North": (108_106, 213_356),
    "Adh Dhahirah": (66_649, 177_724),
    "Al Wusta": (34_254, 26_972),
}

# (governorate, the yearbook's wilaya label) -> (expatriates, Omanis), end-2024, Table 7-2.
# Labels are spelt as the yearbook prints them; om.py finds each one on the page.
WILAYAT_2024 = {
    ("Muscat", "Muscat"): (14_265, 22_007),
    ("Muscat", "Mutrah"): (185_036, 42_697),
    ("Muscat", "Al Amrat"): (58_395, 95_932),
    ("Muscat", "Bawshar"): (346_359, 94_540),
    ("Muscat", "As Seeb"): (296_597, 277_500),
    ("Muscat", "Qurayyat"): (14_027, 51_166),
    ("Dhofar", "Salalah"): (264_938, 165_425),
    ("Dhofar", "Taqah"): (4_545, 20_621),
    ("Dhofar", "Mirbat"): (4_373, 14_621),
    ("Dhofar", "Rakhyut"): (565, 4_673),
    ("Dhofar", "Thumrayt"): (8_513, 10_824),
    ("Dhofar", "Dalkut"): (377, 3_079),
    ("Dhofar", "Al Mazuynah"): (2_143, 8_974),
    ("Dhofar", "Muqshin"): (221, 655),
    ("Dhofar", "Shalim wa juzor al Hallniyat"): (4_324, 4_417),
    ("Dhofar", "Sadah"): (764, 5_522),
    ("Musandam", "Khasab"): (13_416, 21_910),
    ("Musandam", "Daba"): (4_035, 7_177),
    ("Musandam", "Bukha"): (1_029, 3_339),
    ("Musandam", "Madaha"): (607, 3_637),
    ("Al Buraymi", "Al Buraymi"): (55_783, 68_014),
    ("Al Buraymi", "Mahdah"): (2_170, 8_601),
    ("Al Buraymi", "As Sunaynah"): (628, 625),
    ("Ad Dakhliyah", "Nizwa"): (56_755, 90_430),
    ("Ad Dakhliyah", "Bahla"): (22_988, 78_555),
    ("Ad Dakhliyah", "Manah"): (6_138, 20_267),
    ("Ad Dakhliyah", "Al Hamra"): (4_807, 27_123),
    ("Ad Dakhliyah", "Adam"): (11_180, 21_181),
    ("Ad Dakhliyah", "Izki"): (17_930, 52_342),
    ("Ad Dakhliyah", "Samail"): (22_618, 70_773),
    ("Ad Dakhliyah", "BidBid"): (11_831, 32_440),
    ("Ad Dakhliyah", "AL Jabal Alakhdar"): (1_418, 11_707),
    ("Al Batinah North", "Sohar"): (144_519, 142_575),
    ("Al Batinah North", "Shinas"): (21_434, 71_637),
    ("Al Batinah North", "Liwa"): (24_561, 39_869),
    ("Al Batinah North", "Saham"): (51_301, 125_265),
    ("Al Batinah North", "Al khaburah"): (22_901, 70_592),
    ("Al Batinah North", "As Suwayq"): (63_151, 147_358),
    ("Al Batinah South", "Al Rustaq"): (32_339, 106_202),
    ("Al Batinah South", "Al Awabi"): (2_908, 17_759),
    ("Al Batinah South", "Nakhal"): (7_214, 25_519),
    ("Al Batinah South", "Wadi Al maawil"): (4_453, 17_954),
    ("Al Batinah South", "Barka"): (107_492, 130_354),
    ("Al Batinah South", "Al Musanaah"): (30_487, 84_090),
    ("Ash Sharqiyah South", "Sur"): (48_370, 80_390),
    ("Ash Sharqiyah South", "Al Kamil wa al Wafi"): (16_698, 29_451),
    ("Ash Sharqiyah South", "Jaalan bani bu Hasan"): (15_586, 38_846),
    ("Ash Sharqiyah South", "Jaalan bani bu Ali"): (34_668, 88_531),
    ("Ash Sharqiyah South", "Masirah"): (7_413, 9_575),
    ("Ash Sharqiyah North", "Ibra"): (24_372, 32_379),
    ("Ash Sharqiyah North", "Al Mudaybi"): (29_497, 70_182),
    ("Ash Sharqiyah North", "Bidiyah"): (25_411, 28_869),
    ("Ash Sharqiyah North", "Al qabil"): (8_239, 20_992),
    ("Ash Sharqiyah North", "Wadi bani Khalid"): (2_036, 12_459),
    ("Ash Sharqiyah North", "Dima wa at Taiyin"): (4_545, 26_796),
    ("Ash Sharqiyah North", "Sinaw"): (14_006, 21_679),
    ("Adh Dhahirah", "Ibri"): (53_436, 134_286),
    ("Adh Dhahirah", "Yanqul"): (6_792, 22_992),
    ("Adh Dhahirah", "Dank"): (6_421, 20_446),
    ("Al Wusta", "Hayma"): (8_627, 3_708),
    ("Al Wusta", "Muhut"): (5_489, 13_952),
    ("Al Wusta", "Ad Duqm"): (17_960, 5_664),
    ("Al Wusta", "Al Jazir"): (2_178, 3_648),
}

# the yearbook's wilaya label -> geoBoundaries shapeName (every one of the 61, by hand)
UNITS = {
    "Muscat": "WILAYAT MUSCAT", "Mutrah": "WILAYAT MUTRAH", "Al Amrat": "WILAYAT AL AMRAT",
    "Bawshar": "WILAYAT BAWSHAR", "As Seeb": "WILAYAT AS SEEB", "Qurayyat": "WILAYAT QURAYYAT",
    "Salalah": "WILAYAT SALALAH", "Taqah": "WILAYAT TAQAH", "Mirbat": "WILAYAT MIRBAT",
    "Rakhyut": "WILAYAT RAKHYUT", "Thumrayt": "WILAYAT THUMRAYT", "Dalkut": "WILAYAT DALKUT",
    "Al Mazuynah": "WILAYAT AL MAZYUNAH", "Muqshin": "WILAYAT MUQSHIN",
    "Shalim wa juzor al Hallniyat": "WILAYAT SHALIM WA JUZOR AL HALLANIYAT",
    "Sadah": "WILAYAT SADAH", "Khasab": "WILAYAT KHASAB", "Daba": "WILAYAT DABA",
    "Bukha": "WILAYAT BUKHA", "Madaha": "WILAYAT MADHA", "Al Buraymi": "WILAYAT AL BURAYMI",
    "Mahdah": "WILAYAT MAHADAH", "As Sunaynah": "WILAYAT AS SUNAYNAH", "Nizwa": "WILAYAT NIZWA",
    "Bahla": "WILAYAT BAHLA", "Manah": "WILAYAT MANAH", "Al Hamra": "WILAYAT AL HAMRA",
    "Adam": "WILAYAT ADAM", "Izki": "WILAYAT IZKI", "Samail": "WILAYAT SAMAIL",
    "BidBid": "WILAYAT BIDBID", "Sohar": "WILAYAT SOHAR", "Shinas": "WILAYAT SHINAS",
    "Liwa": "WILAYAT LIWA", "Saham": "WILAYAT SAHAM", "Al khaburah": "WILAYAT AL KHABURAH",
    "As Suwayq": "WILAYAT AS SUWAYQ", "Al Rustaq": "WILAYAT AR RUSTAQ",
    "Al Awabi": "WILAYAT AL AWABI", "Nakhal": "WILAYAT NAKHAL",
    "Wadi Al maawil": "WILAYAT WADI AL MAAWIL", "Barka": "WILAYAT BARKA",
    "Al Musanaah": "WILAYAT AL MUSANAAH", "Sur": "WILAYAT SUR",
    "Al Kamil wa al Wafi": "WILAYAT AL KAMIL WA AL WAFI",
    "Jaalan bani bu Hasan": "WILAYAT JAALAN BANI BU HASAN",
    "Jaalan bani bu Ali": "WILAYAT JAALAN BANI BU ALI", "Masirah": "WILAYAT MASIRAH",
    "Ibra": "WILAYAT IBRA", "Al Mudaybi": "WILAYAT AL MUDAYBI", "Bidiyah": "WILAYAT BIDIYAH",
    "Al qabil": "WILAYAT AL QABIL", "Wadi bani Khalid": "WILAYAT WADI BANI KHALID",
    "Dima wa at Taiyin": "WILAYAT DAMA WA AT TAIYIN", "Ibri": "WILAYAT IBRI",
    "Yanqul": "WILAYAT YANQUL", "Dank": "WILAYAT DANK", "Hayma": "WILAYAT HAYMA",
    "Muhut": "WILAYAT MAHAWT", "Ad Duqm": "WILAYAT AD DUQM", "Al Jazir": "WILAYAT AL JAZIR",
}
# wilaya with no polygon -> (the wilaya whose polygon holds its seat, GeoNames names for the seat)
MERGED = {
    "Sinaw": ("Al Mudaybi", ("Sinaw", "Sināw")),
    # Not by `Sayq`, the town on the plateau: GeoNames has two, and the other is a village in Al
    # Kamil wa al Wafi, 250 km away (the wrong twin, found on the first run).
    "AL Jabal Alakhdar": ("Nizwa", ("Al Jabal al Akhdar", "Al Jabal al Akhḑar")),
}

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst, ok in ((GB_URL, GB, lambda d: d.lstrip()[:1] == b"{"),
                         (GEONAMES_URL, GEONAMES, lambda d: d[:2] == b"PK"),
                         (COD_URL, COD, lambda d: d[:2] == b"PK")):
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            continue
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
            data = r.read()
        if not ok(data):
            raise SystemExit(f"{url} did not return the expected file")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)


def unit_key(label):
    return UNITS[label].removeprefix("WILAYAT ").strip().title()


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (GB, GEONAMES, COD)):
        fetch()

    # ---- 2: the register's sums ----
    for gov, (exp, om) in GOVERNORATE_2024.items():
        rows = [v for (g, _w), v in WILAYAT_2024.items() if g == gov]
        if (sum(e for e, _o in rows), sum(o for _e, o in rows)) != (exp, om):
            raise SystemExit(f"Table 7-2 {gov}: wilayat sum to "
                             f"{(sum(e for e, _o in rows), sum(o for _e, o in rows))}, row {(exp, om)}")
    if (sum(e for e, _o in GOVERNORATE_2024.values()), sum(o for _e, o in GOVERNORATE_2024.values())) \
            != (EXPATRIATES, OMANIS):
        raise SystemExit("Table 7-2's governorates do not sum to the national totals")
    if len(WILAYAT_2024) != 63:
        raise SystemExit(f"{len(WILAYAT_2024)} wilayat, expected 63")
    print(f"  Table 7-2: 63 wilayat in 11 governorates sum to {OMANIS:,} Omanis and {EXPATRIATES:,} "
          "expatriates")

    # ---- 1: the polygons ----
    g = gpd.read_file(GB)
    if len(g) != 61 or g.crs.to_epsg() != 4326:
        raise SystemExit(f"geoBoundaries: {len(g)} features, crs {g.crs}; expected 61 in 4326")
    labels = [w for (_g, w) in WILAYAT_2024 if w not in MERGED]
    if len(labels) != 61 or set(UNITS) != set(labels):
        raise SystemExit(f"UNITS does not cover the 61 register wilayat with polygons: "
                         f"{sorted(set(labels) ^ set(UNITS))}")
    if set(g["shapeName"]) != set(UNITS.values()) or g["shapeName"].duplicated().any():
        raise SystemExit(f"geoBoundaries names are not the pinned 61: "
                         f"{sorted(set(g['shapeName']) ^ set(UNITS.values()))}")
    print("  geoBoundaries: 61 wilayat, one register row per polygon by pinned name")

    # ---- 3: the two merges, by seat ----
    with zipfile.ZipFile(GEONAMES) as zf:
        gn = pd.read_csv(zf.open("OM.txt"), sep="\t", header=None, names=GEONAMES_COLS, quoting=3,
                         dtype=str, keep_default_na=False)
    gn = gn[gn["fclass"] == "P"].copy()
    gn["lat"], gn["lon"] = gn["lat"].astype(float), gn["lon"].astype(float)
    by_name = g.set_index("shapeName")
    target_of = {}
    for new, (into, seats) in MERGED.items():
        hit = gn[gn["name"].isin(seats) | gn["asciiname"].isin(seats)]
        if hit.empty:
            raise SystemExit(f"GeoNames has no populated place named {seats} for {new}")
        pts = gpd.GeoSeries(gpd.points_from_xy(hit["lon"], hit["lat"]), crs=4326)
        inside = []
        for p, nm in zip(pts, hit["name"]):
            poly = [s for s, geom in by_name.geometry.items() if geom.contains(p)]
            inside.append((nm, poly[0] if poly else None))
        print(f"  {new}: GeoNames seats {inside}")
        if any(pg != UNITS[into] for _nm, pg in inside):
            raise SystemExit(f"{new}'s seat is not inside {UNITS[into]}")
        target_of[new] = into

    # ---- the lookup: one row per register wilaya, unit = polygon ----
    rows = []
    for (gov, w), (exp, om) in WILAYAT_2024.items():
        poly_label = target_of.get(w, w)
        rows.append(dict(geo_id=f"{gov}|{w}", governorate=gov, wilaya=w, unit=unit_key(poly_label),
                         shape=UNITS[poly_label], expatriates=exp, omanis=om, pop=exp + om))
    lut = pd.DataFrame(rows)
    per_unit = lut.groupby("unit").agg(pop=("pop", "sum"), governorate=("governorate", "first"),
                                       n_gov=("governorate", "nunique"))
    if (per_unit["n_gov"] != 1).any() or len(per_unit) != 61:
        raise SystemExit("a merge crosses a governorate line, or the units are not 61")

    # for the record: COD-AB's governorate for each wilaya (largest overlap)
    with zipfile.ZipFile(COD) as zf:
        cod = gpd.read_file(io.BytesIO(zf.read("omn_admin1.geojson")))
    cod_names = {"Ad Dakhliyah": "Ad Dakhliyah", "Al Batinah North": "Al Batinah North",
                 "Al Batinah South": "Al Batinah South", "Al Buraymi": "Al Buraymi",
                 "Al Dhahira": "Adh Dhahirah", "Al Wusta": "Al Wusta",
                 "Ash Sharqiyah North": "Ash Sharqiyah North",
                 "Ash Sharqiyah South": "Ash Sharqiyah South", "Dhofar": "Dhofar",
                 "Musandam": "Musandam", "Muscat": "Muscat"}
    cod["gov"] = cod["adm1_name"].map(cod_names)
    ga = g.to_crs(EQUAL_AREA)
    ca = cod.to_crs(EQUAL_AREA)
    ov = gpd.overlay(ga[["shapeName", "geometry"]], ca[["gov", "geometry"]], how="intersection",
                     keep_geom_type=True)
    ov["a"] = ov.geometry.area
    best = ov.sort_values("a").groupby("shapeName").tail(1).set_index("shapeName")
    share = ov.groupby("shapeName")["a"].sum()
    gov_of_shape = {UNITS[w]: gv for (gv, w) in WILAYAT_2024 if w not in MERGED}
    off = [(s, gov_of_shape[s], best.loc[s, "gov"], round(float(best.loc[s, "a"] / share[s]), 2))
           for s in best.index if best.loc[s, "gov"] != gov_of_shape[s]]
    print(f"  COD-AB's governorate differs from the register's for {len(off)} of 61 wilayat "
          "(shape, register, COD, COD's share of the wilaya):")
    for o in off:
        print("      ", o)
    gov_area = ga.assign(gov=ga["shapeName"].map(gov_of_shape)).dissolve("gov").geometry.area / 1e6
    cod_area = ca.set_index("gov").geometry.area / 1e6
    print("  governorate area, km2: this layer / COD-AB")
    for gv in GOVERNORATE_2024:
        print(f"      {gv:<20} {gov_area[gv]:>10,.0f} {cod_area[gv]:>10,.0f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    units = g.rename(columns={"shapeName": "shape"})[["shape", "geometry"]]
    units["unit"] = units["shape"].str.removeprefix("WILAYAT ").str.strip().str.title()
    units = units.merge(per_unit[["pop", "governorate"]], left_on="unit", right_index=True,
                        how="inner", validate="1:1")
    if len(units) != 61:
        raise SystemExit("units lost in the merge with the register")
    units[["unit", "governorate", "pop", "geometry"]].to_file(OUT_UNITS, layer="units", driver="GPKG")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS} (61) and {LOOKUP} (63 rows, {int(lut['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
