"""Chad — the 22 régions of the 2009 census, and the placement grid.

Writes:
    data/geo/td/td_regions.gpkg     the 22 counted units (`units`)
    data/geo/td/td_hexes.gpkg       Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/td/td_lookup.csv       unit -> census population, Kontur population

Usage:
    python sources/td_geo.py --fetch    COD-AB shapefile zip (~150 KB) + Kontur (~8.1 MB)
    python sources/td_geo.py            rebuild from data/raw/td/

THE BOUNDARIES ARE OCHA COD-AB CHAD (`cod-ab-tcd`, v01, valid_on 2025-02-12), the shapefile
bundle, read with `engine="fiona"`. ADM1 has 23 provinces; the census has 22 régions.

THE 2009 RÉGIONS ARE REBUILT FROM COD'S 70 DÉPARTEMENTS, BECAUSE TWO CHANGES HAPPENED SINCE.

1. **Ennedi was split in 2012** into Ennedi Est (Amdjarass) and Ennedi Ouest (Fada), as the
   structure volume's own footnote says (printed p32). All four of their COD départements go
   back into one `Ennedi`.
2. **Djourf Al Ahmar and Abdi have changed province**, which nothing in the census volume
   mentions and the 2009 areas give away. On COD, Djourf Al Ahmar (TCD1402, 14,721 km2) is in
   Ouaddaï and Abdi (TCD2102, 3,829 km2) in Sila. In 2009 Sila was Kimiti plus **Djourouf Al
   Amar** (Am-Dam, Haouich, Magrane) and Ouaddaï was Ouara, **Abdi** (Abdi, Abker-Djombo,
   Biyéré) and Assoungha, 721,166, read off INSEED's *Résultats définitifs par
   sous-préfecture*, Tableau 01 (printed pp13 and 15) and Tableau 02 (p16) (NADA catalog 26,
   download 154, a scan with no text layer). With COD's provinces as they are, Tableau 2.13's population over density puts
   2009 Ouaddaï at 30,049 km2 against COD's 40,663 and 2009 Sila at 35,876 against 24,835; with
   the two départements moved back they come out at 29,770 and 35,727. `AREA_2009` asserts it.

The 2018 constitution renamed the régions provinces; the other names are the census's, one
spelling apart (`Barh El Gazal` in the census, `BARH EL GAZEL` in COD).

THE JOIN IS BY NAME, folded, through one alias, and asserted to be a bijection onto 22 units
after the rebuild. Any unmatched name raises.

KONTUR IS NEEDED BECAUSE THE UNITS ARE VERY UNEQUAL. N'Djaména is 436 km2 holding 951,418
people in 2009, Tibesti is 213,590 km2 holding 21,303, and the Sahelian régions have their
people along wadis and around the lake. Kontur 2023-11 is used only as a WITHIN-région weight.
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
RAW = os.path.join(ROOT, "data", "raw", "td")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "td")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "td.csv")

ZIP_NAME = "tcd_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/f01c8222-b67d-41ff-954a-731aafd4466b/resource/"
           "5307927e-9d4f-4b5f-9996-e55c3750a2ab/download/tcd_admin_boundaries.shp.zip")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TD_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TD_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TD_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "td_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "td_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "td_lookup.csv")

UNITS = 22
COD_FEATURES = 23
COD_ADM2 = 70

# COD ADM1 names dissolved into one census région.
DISSOLVE = {"Ennedi": ("ENNEDI EST", "ENNEDI OUEST")}
# COD ADM2 pcode -> (its COD name, the COD ADM1 name of the région it belonged to in 2009).
MOVED = {"TCD1402": ("Djourf Al Ahmar", "SILA"),
         "TCD2102": ("Abdi", "OUADDAÏ")}

# Tableau 2.13 (structure volume, printed p54): total population and density, one decimal.
# Area = population / density; the density's rounding sets the interval. Sila and Tibesti's
# totals include the 98,191 estimated, the other twenty equal their censused populations.
T213_TOTAL = {"Sila": 387_461, "Tibesti": 25_483}
T213_DENSITY = {
    "Batha": 5.3, "Borkou": 0.5, "Chari Baguirmi": 12.2, "Guéra": 8.8, "Hadjer Lamis": 19.3,
    "Kanem": 4.6, "Lac": 19.8, "Logone Occidental": 77.3, "Logone Oriental": 32.7,
    "Mandoul": 36.0, "Mayo Kebbi Est": 42.2, "Mayo Kebbi Ouest": 43.6, "Moyen Chari": 14.6,
    "Ouaddaï": 24.0, "Salamat": 4.4, "Tandjilé": 37.5, "Wadi Fira": 9.8, "N'Djaména": 1902.8,
    "Barh El Gazal": 5.1, "Ennedi": 0.8, "Sila": 10.8, "Tibesti": 0.1,
}
# The rebuilt units must land inside their 2009 interval, widened by this much for digitising.
AREA_SLACK = 0.03
# census spelling -> COD adm1_name, where folding alone does not meet.
ALIAS = {"Barh El Gazal": "BARH EL GAZEL"}

# Census June 2009 against Kontur 2023-11: fourteen and a half years at the census's own 3.6% a
# year intercensal growth (Tableau 2.08) is about 1.67x, before the Sudanese refugees who
# arrived in the east from 2023. The band is wide around it.
KONTUR_RATIO_MIN = 1.10
KONTUR_RATIO_MAX = 2.60

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _get(url, dst, magic, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(len(magic))
    # §5a: a 200 is not a download.
    if head != magic:
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    z = os.path.join(RAW, ZIP_NAME)
    _get(ZIP_URL, z, b"PK", 100_000)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SHP_DIR)

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 5_000_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    shp = os.path.join(SHP_DIR, "tcd_admin1.shp")
    shp2 = os.path.join(SHP_DIR, "tcd_admin2.shp")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    for p in (shp, shp2, gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/td.py --fetch and "
                             "sources/td_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df.groupby("geo_name")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in td.csv, expected {UNITS}")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    a1 = gpd.read_file(shp, engine="fiona")
    g = gpd.read_file(shp2, engine="fiona")
    print(f"COD-AB Chad: admin1 {len(a1)} features, admin2 {len(g)}, crs={g.crs}, valid_on "
          f"{sorted(set(map(str, g['valid_on'])))}, version "
          f"{sorted(set(map(str, g['version'])))}")
    if len(a1) != COD_FEATURES or len(g) != COD_ADM2:
        raise SystemExit(f"expected {COD_FEATURES} provinces and {COD_ADM2} départements")
    if set(g["adm1_name"]) != set(a1["adm1_name"]):
        raise SystemExit("admin2's province names are not admin1's")

    # ---- 2009's régions, rebuilt from 2025's départements
    g["unit_cod"] = g["adm1_name"]
    for pcode, (name, was) in MOVED.items():
        hit = g["adm2_pcode"] == pcode
        if int(hit.sum()) != 1 or g.loc[hit, "adm2_name"].iloc[0] != name:
            raise SystemExit(f"{pcode}: expected COD département {name!r}")
        if was not in set(a1["adm1_name"]):
            raise SystemExit(f"{was!r} is not a COD province")
        print(f"  {name} ({pcode}) is in {g.loc[hit, 'adm1_name'].iloc[0]} on COD and was in "
              f"{was} in 2009")
        g.loc[hit, "unit_cod"] = was
    for unit, parts in DISSOLVE.items():
        hit = g["adm1_name"].isin(parts)
        if set(g.loc[hit, "adm1_name"]) != set(parts):
            raise SystemExit(f"{unit}: found {sorted(set(g.loc[hit, 'adm1_name']))}, "
                             f"expected {parts}")
        g.loc[hit, "unit_cod"] = unit
    g = g.dissolve(by="unit_cod",
                   aggfunc={"adm1_pcode": lambda s: "+".join(sorted(set(s)))},
                   as_index=False)
    print(f"  {COD_ADM2} départements dissolved into {len(g)} units (Ennedi Est + Ennedi Ouest, "
          f"and {len(MOVED)} départements moved back)")

    g["key"] = g["unit_cod"].map(norm)
    lut = {}
    for name in census.index:
        want = norm(ALIAS.get(name, name))
        hits = g.index[g["key"] == want].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census région {name!r} matched {len(hits)} COD polygons: "
                             f"{sorted(g['unit_cod'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS or len(g) != UNITS:
        raise SystemExit("the join is not a bijection onto the 22 units")
    print(f"  name join: {UNITS}/{UNITS} census régions matched one polygon each, "
          "0 polygons unused")
    for name, i in sorted(lut.items()):
        if norm(g.loc[i, "unit_cod"]) != norm(name):
            print(f"    census {name!r} <-> COD {g.loc[i, 'unit_cod']!r} "
                  f"({g.loc[i, 'adm1_pcode']})")

    units = g.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units = units.rename(columns={"unit_cod": "cod_name"})
    units = units[["unit", "adm1_pcode", "cod_name", "census_pop", "geometry"]]
    units.to_file(OUT_UNITS, layer="regions", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} régions)")

    eq = units.to_crs(6933)
    km2 = (eq.geometry.area / 1e6).to_numpy()
    for (_i, r), a in sorted(zip(units.iterrows(), km2), key=lambda t: -t[1]):
        print(f"    {r['unit']:<18} {a:>9,.0f} km²  {int(r['census_pop']):>10,} people  "
              f"{r['census_pop'] / a:>8,.1f}/km²")

    # ---- the 2009 areas, Tableau 2.13's population over its one-decimal density
    print("\n  COD area against the 2009 area (Tableau 2.13 population / density):")
    rebuilt = {was for _n, was in MOVED.values()}
    bad = []
    for (_i, r), a in zip(units.iterrows(), km2):
        u = r["unit"]
        pop = T213_TOTAL.get(u, int(r["census_pop"]))
        d = T213_DENSITY[u]
        lo, hi = pop / (d + 0.05), pop / max(d - 0.05, 1e-9)
        inside = lo * (1 - AREA_SLACK) <= a <= hi * (1 + AREA_SLACK)
        touched = norm(u) in {norm(x) for x in rebuilt}
        if touched and not inside:
            bad.append(u)
        print(f"    {u:<18} COD {a:>9,.0f}  2009 {pop / d:>9,.0f} [{lo:,.0f}-{hi:,.0f}]"
              f"  {a / (pop / d):5.2f}x{'' if inside else '  outside'}"
              f"{'  (rebuilt)' if touched else ''}")
    if bad:
        raise SystemExit(f"rebuilt régions outside their 2009 area: {bad}")

    # ---- 2. Kontur, joined on hex CENTROIDS
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Take the centroid in the CRS the hexes were tiled in, then reproject the POINTS.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no région: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%), dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"régions with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"régions whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} régions has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2009 drawn population {census_total:,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
                         f"{KONTUR_RATIO_MAX}], check the download")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units[["unit", "adm1_pcode", "census_pop"]].merge(
        per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_pop_2009", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.adm1_pcode, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    # A région far off the national ratio is the witness for a boundary that moved between 2009
    # and 2025 (or for refugee camps Kontur counts and the census placed elsewhere): read it.
    print(f"\n  per-région Kontur/census ratio (national {ratio:.2f}x):")
    for r in lk.sort_values("kontur_over_census").itertuples(index=False):
        flag = "  <-- read" if not 0.6 * ratio <= r.kontur_over_census <= 1.6 * ratio else ""
        print(f"    {r.unit:<18} {r.kontur_over_census:5.2f}x  {int(r.hexes):>7,} hexes{flag}")


if __name__ == "__main__":
    main()
