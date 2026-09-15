"""Sierra Leone — the fourteen 2015 districts and the placement grid.

Writes:
    data/geo/sl/sl_districts.gpkg   the 14 counted units (`units`)
    data/geo/sl/sl_hexes.gpkg       Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/sl/sl_lookup.csv       unit -> census population, area, Kontur population

Usage:
    python sources/sl_geo.py --fetch    COD-AB shapefile zip (~10 MB), geoBoundaries ADM2 (~1 MB),
                                        Kontur (~3.5 MB)
    python sources/sl_geo.py            rebuild from data/raw/sl/

## THE CENSUS HAS 14 DISTRICTS AND COD-AB HAS 16

The 2015 census was taken on 14 districts. In 2017 Karene was made out of Bombali and Port Loko,
and Falaba out of Koinadugu, and OCHA COD-AB Sierra Leone (`cod-ab-sle`, version 02, reviewed 30
October 2025) draws the 16. Its ADM3 is still the pre-2017 chiefdom set (167 units: the 149 rural
chiefdoms, the city councils and the Western Area wards), each a whole chiefdom, so the 2015
districts are rebuilt by giving Karene's and Falaba's chiefdoms back to their 2015 parents:

    Karene   Buya Romende, Dibia, Sanda Magbolontor                      -> Port Loko
             Libeisaygahun, Sanda Loko, Sanda Tendaran, Sella Limba,
             Tambakha                                                    -> Bombali
    Falaba   Dembelia Sinkunia, Folosaba Dembelia, Mongo, Neya, Sulima   -> Koinadugu

Every one of those thirteen is listed under its 2015 district in the census's own Table 3.3b
(Bombali: `Libeisaygahun`, `Sando loko`, `Sabda Tendaren`, `Sella limba`, `Tambakka`; Port Loko:
`Buya Romende`, `Bibia`, `Sanda Magbo-lotor`; Koinadugu: `Dembelia Sinknia`, `Follosaba
Dembelia`, `Mongo`, `Neya`, `Sulima`), and the number of COD chiefdoms that lands in each 2015
district is asserted against Table 3.3's row count for it. The census report's
16-district re-cut (`sierraleone_-2015_population_census_data_for_16_districts_5_regions.pdf`)
prints Falaba 205,353 and Koinadugu 204,019, which sum to 2015 Koinadugu's 409,372 exactly.

## AN INDEPENDENT WITNESS: geoBoundaries' 14 DISTRICTS

geoBoundaries gbOpen SLE ADM2 (commit 9469f09) is 14 districts from the Government of Sierra
Leone and OCHA ROWCA's older HDX release, drawn on different linework. Each rebuilt district is
asserted to overlap its geoBoundaries namesake (IoU) and no other district, which neither the
name join nor the chiefdom move can make true by construction.

## KONTUR IS A WEIGHT INSIDE A DISTRICT

Koinadugu is 12,000 km2 of savannah and hills with its people along a few roads; Western Area
Urban is Freetown, 80 km2 holding a million people. Hexes join on their centroid; a centroid
that falls just offshore of a coastal district (the Sherbro and Turtle islands, the Freetown
peninsula) is snapped to the nearest district within SNAP_M rather than dropped.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "sl")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "sl")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "sl.csv")

OUT_UNITS = os.path.join(GEO, "sl_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "sl_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "sl_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    os.path.join(RAW, "sle_admin_boundaries.shp.zip"): (
        "https://data.humdata.org/dataset/a4816317-a913-4619-b1e9-d89e21c056b4/resource/"
        "deacde79-fe3e-4113-9b26-87c4b6c17d35/download/sle_admin_boundaries.shp.zip",
        b"PK", 5_000_000),
    os.path.join(RAW, "geoBoundaries-SLE-ADM2.geojson"): (
        "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/SLE/ADM2/"
        "geoBoundaries-SLE-ADM2.geojson",
        b"{", 500_000),
    os.path.join(KONTUR, "kontur_population_SL_20231101.gpkg.gz"): (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_SL_20231101.gpkg.gz",
        b"\x1f\x8b", 1_000_000),
}
KONTUR_GPKG = os.path.join(KONTUR, "kontur_population_SL_20231101.gpkg")

UNITS = 14
COD_ADM2 = 16
COD_ADM3 = 167

# COD ADM3 pcode -> (COD's name, the 2015 district it belonged to). Karene and Falaba only.
MOVE = {
    "SL050201": ("Buya Romende", "Port Loko"),
    "SL050202": ("Dibia", "Port Loko"),
    "SL050203": ("Sanda Magbolont", "Port Loko"),
    "SL050206": ("Libeisaygahun", "Bombali"),
    "SL050207": ("Sanda Loko", "Bombali"),
    "SL050208": ("Sanda Tendaran", "Bombali"),
    "SL050209": ("Sella Limba", "Bombali"),
    "SL050210": ("Tambakha", "Bombali"),
    "SL020601": ("Dembelia - Sink", "Koinadugu"),
    "SL020602": ("Folosaba Dembel", "Koinadugu"),
    "SL020603": ("Mongo", "Koinadugu"),
    "SL020604": ("Neya", "Koinadugu"),
    "SL020606": ("Sulima", "Koinadugu"),
}
NEW_DISTRICTS = {"SL0502": "Karene", "SL0206": "Fabala"}      # COD spells Falaba `Fabala`

# Chiefdoms (with the city councils) per 2015 district in the census's Table 3.3a-c, which
# does not cover the Western Area; those two are COD's own ward counts, asserted unchanged.
EXPECTED_CHIEFDOMS = {
    "Kailahun": 14, "Kenema": 17, "Kono": 15,
    "Bombali": 14, "Kambia": 7, "Koinadugu": 11, "Port Loko": 11, "Tonkolili": 11,
    "Bo": 16, "Bonthe": 12, "Moyamba": 14, "Pujehun": 12,
    "Western Area Rural": 4, "Western Area Urban": 9,
}

IOU_MIN = 0.90          # a rebuilt district against its geoBoundaries namesake
IOU_OTHER_MAX = 0.02    # and against every other geoBoundaries district
# COD draws Tasso Island, in the Sierra Leone River estuary, as a ninth Western Area Urban ward
# (8 km2). geoBoundaries' Western Area Urban is 74 km2 and leaves the island out, which on an
# 82 km2 unit is enough to take the IoU to 0.878 (first build, 2026-09-15). The island stays in
# the drawn district: the census's own 16-district re-cut prints eight Western Area Urban wards
# summing to the district's 1,055,964, so whoever lives on Tasso is counted inside one of them.
# The witness compares the district without it, and the build asserts the island is that ward.
WITNESS_EXCLUDE = {"Western Area Urban": ("SL040209", "Tasso Island")}
AREA_TOL = 0.001        # the dissolved districts against COD's own national polygon
SNAP_M = 500

# Census December 2015 against Kontur November 2023. The census's own 2004-2015 growth is 3.2% a
# year, which is about 1.28x over eight years. Set wide on the first build and printed per unit.
KONTUR_RATIO = (0.90, 1.80)
KONTUR_UNIT_BAND = (0.60, 2.50)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, magic, min_size) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        if not r.content.startswith(magic):                    # §5a: a 200 is not a download
            raise SystemExit(f"{os.path.basename(dst)} starts {r.content[:16]!r}, not {magic!r}")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(r.content):,} bytes)")


def unpack():
    z = os.path.join(RAW, "sle_admin_boundaries.shp.zip")
    if not os.path.exists(os.path.join(SHP_DIR, "sle_admin3.shp")):
        if not os.path.exists(z):
            raise SystemExit(f"missing {z}; run with --fetch first")
        with zipfile.ZipFile(z) as zz:
            zz.extractall(SHP_DIR)
    gz = KONTUR_GPKG + ".gz"
    if not os.path.exists(KONTUR_GPKG):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(KONTUR_GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR_GPKG + ".part", KONTUR_GPKG)
    with open(KONTUR_GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{KONTUR_GPKG} is not a GeoPackage")


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/sl.py first")
    census = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby(
        "geo_id")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} districts in sl.csv, expected {UNITS}")

    # ---- 1. COD-AB, chiefdoms regrouped to the 2015 districts
    a0 = gpd.read_file(os.path.join(SHP_DIR, "sle_admin0.shp"), engine="fiona")
    a2 = gpd.read_file(os.path.join(SHP_DIR, "sle_admin2.shp"), engine="fiona")
    a3 = gpd.read_file(os.path.join(SHP_DIR, "sle_admin3.shp"), engine="fiona")
    if len(a2) != COD_ADM2 or len(a3) != COD_ADM3:
        raise SystemExit(f"COD-AB has {len(a2)} districts and {len(a3)} chiefdoms, expected "
                         f"{COD_ADM2} and {COD_ADM3}")
    print(f"COD-AB Sierra Leone: {len(a2)} districts, {len(a3)} chiefdoms, version "
          f"{a3['version'].iloc[0]}, valid_on {a3['valid_on'].iloc[0]}, crs={a3.crs}")

    got_new = {r.adm2_pcode: r.adm2_name for r in a2.itertuples()
               if r.adm2_pcode in NEW_DISTRICTS}
    if got_new != NEW_DISTRICTS:
        raise SystemExit(f"COD's 2017 districts are {got_new}, expected {NEW_DISTRICTS}")
    in_new = set(a3.loc[a3["adm2_pcode"].isin(NEW_DISTRICTS), "adm3_pcode"])
    if in_new != set(MOVE):
        raise SystemExit(f"Karene and Falaba hold {sorted(in_new)}, MOVE lists {sorted(MOVE)}")
    names = dict(zip(a3["adm3_pcode"], a3["adm3_name"]))
    bad = {pc: (names.get(pc), want) for pc, (want, _d) in MOVE.items()
           if fold(names.get(pc)) != fold(want)}
    if bad:
        raise SystemExit(f"COD chiefdom names moved under their pcodes: {bad}")

    def district(r):
        if r.adm3_pcode in MOVE:
            return MOVE[r.adm3_pcode][1]
        return r.adm2_name

    a3["unit"] = [district(r) for r in a3.itertuples()]
    got = a3.groupby("unit").size().to_dict()
    if got != EXPECTED_CHIEFDOMS:
        raise SystemExit(f"chiefdoms per 2015 district {got}, census Table 3.3 {EXPECTED_CHIEFDOMS}")
    print(f"  {len(MOVE)} Karene and Falaba chiefdoms returned to their 2015 districts; every "
          "district's chiefdom count equals the census's Table 3.3")
    if set(got) != set(census.index):
        raise SystemExit(f"district names differ from sl.csv: {sorted(set(got) ^ set(census.index))}")

    units = a3[["unit", "geometry"]].dissolve(by="unit").reset_index()
    eq = "EPSG:6933"
    units["area_km2"] = units.to_crs(eq).area / 1e6
    nat = float(a0.to_crs(eq).area.sum() / 1e6)
    if abs(units["area_km2"].sum() / nat - 1) > AREA_TOL:
        raise SystemExit(f"districts sum to {units['area_km2'].sum():,.0f} km2 against COD's "
                         f"national {nat:,.0f}")
    units["census_pop"] = units["unit"].map(census).astype("int64")
    print(f"  {UNITS} districts dissolved, {units['area_km2'].sum():,.0f} km2 against COD's "
          f"national polygon {nat:,.0f}")

    # ---- 2. the witness: geoBoundaries' own 14 districts
    gb = geo_checks.read_layer(os.path.join(RAW, "geoBoundaries-SLE-ADM2.geojson"),
                               "geoBoundaries SLE ADM2")
    if len(gb) != UNITS:
        raise SystemExit(f"geoBoundaries SLE ADM2 has {len(gb)} features, expected {UNITS}")
    gb["key"] = gb["shapeName"].map(fold)
    lut = {}
    for u in units["unit"]:
        hits = gb.index[gb["key"] == fold(u)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"district {u!r} matched {len(hits)} geoBoundaries features: "
                             f"{sorted(gb['shapeName'])}")
        lut[u] = hits[0]
    from shapely.ops import unary_union

    ue = units.set_index("unit").to_crs(eq)
    ge = gb.to_crs(eq)
    a3e = a3.to_crs(eq)
    print("\n  rebuilt district against geoBoundaries (IoU with its namesake; largest IoU with "
          "any other):")
    fails = []
    for u in units["unit"]:
        a = ue.geometry[u]
        what = ""
        if u in WITNESS_EXCLUDE:
            pc, nm = WITNESS_EXCLUDE[u]
            ward = a3e[a3e["adm3_pcode"] == pc]
            if len(ward) != 1 or fold(ward["adm3_name"].iloc[0]) != fold(nm) \
                    or ward["unit"].iloc[0] != u:
                raise SystemExit(f"COD {pc} is not {nm} in {u}")
            a = a.difference(unary_union(list(ward.geometry)))
            what = f"  (without {nm}, {ward.geometry.area.sum() / 1e6:.1f} km2)"
        ious = {}
        for j, g in ge.geometry.items():
            inter = a.intersection(g).area
            if inter > 0:
                ious[j] = inter / a.union(g).area
        own = ious.get(lut[u], 0.0)
        other = max([v for j, v in ious.items() if j != lut[u]], default=0.0)
        print(f"    {u:<20} IoU {own:.3f}   other {other:.3f}   {ue.loc[u, 'area_km2']:>8,.0f} km2 "
              f"(geoBoundaries {ge.geometry[lut[u]].area / 1e6:>8,.0f}){what}")
        if own < IOU_MIN or other > IOU_OTHER_MAX:
            fails.append(u)
    if fails:
        raise SystemExit(f"districts that do not match geoBoundaries' 2015 layout: {fails}")

    # ---- 3. Kontur
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur SL")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit_of = joined["unit"].copy()
    outside = unit_of.isna()
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['pop'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids in no district "
          f"({pts.loc[outside, 'pop'].sum():,.0f} people)")
    if outside.any():
        utm = "EPSG:32629"
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(utm),
                                 units[["unit", "geometry"]].to_crs(utm),
                                 how="left", max_distance=SNAP_M, distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")]
        snapped = near["unit"].dropna()
        unit_of.loc[snapped.index] = snapped
        print(f"  snapped {len(snapped):,} of them within {SNAP_M} m "
              f"({pts.loc[snapped.index, 'pop'].sum():,.0f} people)")
    outside = unit_of.isna()
    lost = float(pts.loc[outside, "pop"].sum())
    print(f"  dropped {int(outside.sum()):,} further out ({lost:,.0f} people, "
          f"{100 * lost / pts['pop'].sum():.3f}%)")

    keep = ~outside
    out = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    ratio = tot / float(units["census_pop"].sum())
    print(f"  Kontur 2023 {tot:,.0f} against the drawn 2015 household population "
          f"{int(units['census_pop'].sum()):,}: ratio {ratio:.3f}")
    if not KONTUR_RATIO[0] <= ratio <= KONTUR_RATIO[1]:
        raise SystemExit(f"ratio outside {KONTUR_RATIO}; check the download")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print("\n  per district, census 2015 against Kontur 2023 (a witness; Kontur counts nothing):")
    for r in units.sort_values("census_pop", ascending=False).itertuples():
        print(f"    {r.unit:<20}{r.census_pop:>11,}  Kontur {r.kontur_pop:>11,}  "
              f"{r.kontur_pop / r.census_pop:5.2f}x  {r.hexes:>7,} hexes  {r.area_km2:>8,.0f} km2")
    geo_checks.ratio_band(dict(zip(units["unit"], units["census_pop"])),
                          dict(zip(units["unit"], units["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="district")

    # ---- write
    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "census_pop", "kontur_pop", "hexes", "area_km2", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="districts", driver="GPKG")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "census_hh_pop_2015", "kontur_pop_2023", "hexes", "area_km2",
                    "kontur_over_census"])
        for r in units.sort_values("unit").itertuples():
            w.writerow([r.unit, r.census_pop, r.kontur_pop, r.hexes, round(r.area_km2, 1),
                        round(r.kontur_pop / r.census_pop, 3)])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES} ({len(out):,} hexes)\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
