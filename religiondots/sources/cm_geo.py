"""Cameroon — 12 units (the 10 regions, with Yaoundé and Douala apart), populations, and the grid.

Writes:
    data/geo/cm/cm_regions.gpkg      the 12 units (`units`)
    data/geo/cm/cm_hexes.gpkg        Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/cm/cm_lookup.csv        unit -> name, COD-PS 2025 population, Kontur population
    data/geo/cm/cm_departments.csv   the 58 departments and their unit, for `sources/cm.py`'s
                                     check of the survey's district column against its region

Usage:
    python sources/cm_geo.py --fetch    COD-AB shapefile zip (~12 MB), COD-PS workbook, Kontur (~8.5 MB)
    python sources/cm_geo.py            rebuild from data/raw/cm/

## WHY 12 UNITS AND NOT 10

Every Afrobarometer round that asked Cameroonians their religion (5 to 9) samples Yaoundé and
Douala as strata of their own, under labels that change every round (`Yaounde`, `Centre-Yaoundé`,
`Mfoundi`; `Douala`, `Littoral-Douala`, `Wouri`). The district column in rounds 6, 7 and 9 puts
every one of those respondents in Mfoundi or Wouri department and none of the `Centre` or
`Littoral` respondents there. So the survey measures the two cities apart from the rest of their
regions, and drawing Centre as one unit would spread Yaoundé's mix over the forest villages of
the Lekié and the Nyong and pile the village mix into the city, where Kontur puts most of the
dots. Mfoundi (289 km2) and Wouri (976 km2) are cut out of their regions as whole departments.

## BOUNDARIES: OCHA COD-AB CAMEROON (`cod-ab-cmr`), VERSION 01

Institut National de Cartographie; 10 regions, 58 departments, 360 arrondissements; reviewed
30 October 2025. The units are built by dissolving the 58 department polygons, so the two cities
and the rest of their regions share edges exactly; each dissolved region is asserted against
COD's own region polygon by area.

## POPULATION: COD-PS 2025, WHICH IS BUCREP'S OWN PROJECTION

COD-PS Cameroon is the Bureau Central des Recensements et des Etudes de Population's projection
to 2025 from the 2005 census (cohort component, zero internal migration, calibrated to the
national projection), with an extra table for the two metropolises, `Ville de Douala` on Wouri's
pcode and `Ville de Yaoundé` on Mfoundi's. There is nothing newer to prefer: the 4th census was
enumerated from 24 April 2026 and nothing from it is published. The explanatory note says the zero
internal migration assumption may distort the regional counts; Kontur is printed beside it per
unit as a witness, never used to count.

## KONTUR IS A WEIGHT INSIDE A UNIT

Est is 110,000 km2 and Mfoundi 289; most regions hold their people along roads and in a few towns.
Hexes join on their centroid, so a hex on a line belongs wholly to one side.
"""

import os
import re
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cm")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "cm")
OUT_UNITS = os.path.join(GEO, "cm_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "cm_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "cm_lookup.csv")
OUT_DEPTS = os.path.join(GEO, "cm_departments.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

_AB = "https://data.humdata.org/dataset/b13f08ef-92ee-4446-9b0a-e219f5c25415/resource/"
_PS = "https://data.humdata.org/dataset/e8db7923-6cd2-4974-ad57-14a157978cfd/resource/"
DOWNLOADS = {
    "cmr_admin_boundaries.shp.zip": (
        _AB + "918153c4-4474-4734-9cf5-1a4eb74a9c84/download/cmr_admin_boundaries.shp.zip",
        b"PK", 5_000_000),
    "cmr_admpop_2025.xlsx": (
        _PS + "845f2655-6348-4554-8570-04d9556cded6/download/copy-of-cmr_admpop_2025.xlsx",
        b"PK", 10_000),
    "kontur_population_CM_20231101.gpkg.gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_CM_20231101.gpkg.gz", b"\x1f\x8b", 1_000_000),
}
KONTUR_GPKG = "kontur_population_CM_20231101.gpkg"

N_REGIONS = 10
N_DEPARTMENTS = 58
N_UNITS = 12
CODPS_2025 = 29_442_318

# Departments drawn as units of their own; the rest of their region keeps the region's pcode.
CITY = {"CM002007": ("Mfoundi", "CM002"), "CM005004": ("Wouri", "CM005")}
UNIT_NAME = {
    "CM001": "Adamaoua", "CM002": "Centre", "CM002007": "Mfoundi (Yaoundé)", "CM003": "Est",
    "CM004": "Extrême-Nord", "CM005": "Littoral", "CM005004": "Wouri (Douala)", "CM006": "Nord",
    "CM007": "Nord-Ouest", "CM008": "Ouest", "CM009": "Sud", "CM010": "Sud-Ouest",
}
METROPOLIS = {"CM002007": "Ville de Yaoundé", "CM005004": "Ville de Douala"}

# A dissolved region against COD's own region polygon, in km2 (equal-area).
AREA_TOL = 0.005
# Kontur 2023-11 against COD-PS 2025, nationally and per unit. Kontur is modelled from building
# footprints and COD-PS assumes nobody moved between regions since 2005, yet on the first build
# (2026-09-14) every unit came out between 0.87x (Extrême-Nord, Est) and 1.13x (Sud), nationally
# 0.973. The band sits a little outside that, so a join that moved a city fails here.
KONTUR_TOL = 0.25
KONTUR_UNIT_BAND = (0.75, 1.30)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, magic, min_size) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=900)
        r.raise_for_status()
        if not r.content.startswith(magic):                    # §5a: a 200 is not a download
            raise SystemExit(f"{name} starts {r.content[:16]!r}, not {magic!r}")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def unpack():
    import gzip
    import shutil

    z = os.path.join(RAW, "cmr_admin_boundaries.shp.zip")
    if not os.path.exists(z):
        raise SystemExit(f"missing {z}; run with --fetch first")
    if not os.path.exists(os.path.join(SHP_DIR, "cmr_admin2.shp")):
        with zipfile.ZipFile(z) as zz:
            zz.extractall(SHP_DIR)
    gz = os.path.join(RAW, KONTUR_GPKG + ".gz")
    gpkg = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(gpkg):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg


def read_codps():
    import pandas as pd

    x = pd.ExcelFile(os.path.join(RAW, "cmr_admpop_2025.xlsx"))
    meta = x.parse("Metadata").set_index("Item")["Metadata"]
    if int(meta["Reference year of this COD-PS"]) != 2025 or int(meta["Year of the Baseline Population"]) != 2005:
        raise SystemExit("COD-PS is no longer the 2025 projection off the 2005 census; re-read it")
    adm0 = int(x.parse("cmr_admpop_adm0_2025")["T_TL"].iloc[0])
    adm1 = x.parse("cmr_admpop_adm1_2025").set_index("ADM1_PCODE")
    met = x.parse("cmr_admpop_met_2025").set_index("ADM2_PCODE")
    if adm0 != CODPS_2025:
        raise SystemExit(f"COD-PS adm0 is {adm0:,}, this was written against {CODPS_2025:,}")
    if len(adm1) != N_REGIONS or int(adm1["T_TL"].sum()) != adm0:
        raise SystemExit(f"COD-PS adm1 has {len(adm1)} rows summing to {int(adm1['T_TL'].sum()):,}")
    if sorted(met.index) != sorted(CITY):
        raise SystemExit(f"COD-PS metropolis rows are {sorted(met.index)}, expected {sorted(CITY)}")
    for pc, m in METROPOLIS.items():
        if met.at[pc, "Metropolis"] != m or met.at[pc, "ADM1_PCODE"] != CITY[pc][1]:
            raise SystemExit(f"COD-PS metropolis row {pc} is {met.at[pc, 'Metropolis']!r} in "
                             f"{met.at[pc, 'ADM1_PCODE']}, expected {m!r} in {CITY[pc][1]}")
    pop = {pc: int(v) for pc, v in adm1["T_TL"].items()}
    for pc, (_name, parent) in CITY.items():
        pop[pc] = int(met.at[pc, "T_TL"])
        pop[parent] -= pop[pc]
        if pop[parent] <= 0:
            raise SystemExit(f"{parent} has no people left once {pc} is taken out")
    if sum(pop.values()) != CODPS_2025:
        raise SystemExit("the 12 units do not sum to COD-PS's national total")
    return pop, adm1


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    pop, adm1_ps = read_codps()

    a1 = gpd.read_file(os.path.join(SHP_DIR, "cmr_admin1.shp"), engine="fiona")
    a2 = gpd.read_file(os.path.join(SHP_DIR, "cmr_admin2.shp"), engine="fiona")
    if len(a1) != N_REGIONS or len(a2) != N_DEPARTMENTS:
        raise SystemExit(f"COD-AB has {len(a1)} regions and {len(a2)} departments, expected "
                         f"{N_REGIONS} and {N_DEPARTMENTS}")
    print(f"COD-AB Cameroon: {len(a1)} regions, {len(a2)} departments, version "
          f"{a1['version'].iloc[0]}, valid_on {a1['valid_on'].iloc[0]}, crs={a1.crs}")

    # The join to COD-PS is on pcode; the French names are asserted too, so a renumbering fails.
    ps_names = {pc: fold(n) for pc, n in adm1_ps["ADM1_FR"].items()}
    ab_names = {r.adm1_pcode: fold(r.adm1_name1) for r in a1.itertuples()}
    if ps_names != ab_names:
        raise SystemExit(f"COD-AB and COD-PS disagree on region pcodes or names:\n  {ab_names}\n  {ps_names}")
    for pc, (name, parent) in CITY.items():
        row = a2[a2["adm2_pcode"] == pc]
        if len(row) != 1 or fold(row["adm2_name1"].iloc[0]) != fold(name) or row["adm1_pcode"].iloc[0] != parent:
            raise SystemExit(f"COD-AB department {pc} is not {name} in {parent}")

    a2["unit"] = a2.apply(lambda r: r["adm2_pcode"] if r["adm2_pcode"] in CITY else r["adm1_pcode"], axis=1)
    units = a2[["unit", "geometry"]].dissolve(by="unit").reset_index()
    if len(units) != N_UNITS:
        raise SystemExit(f"{len(units)} units after the dissolve, expected {N_UNITS}")
    eq = "EPSG:6933"
    units["area_sqkm"] = units.to_crs(eq).area / 1e6
    reg_area = a1.set_index("adm1_pcode").to_crs(eq).area / 1e6
    back = units.assign(region=units["unit"].str[:5]).groupby("region")["area_sqkm"].sum()
    bad = {pc: (back[pc], reg_area[pc]) for pc in reg_area.index
           if abs(back[pc] / reg_area[pc] - 1) > AREA_TOL}
    if bad:
        raise SystemExit(f"dissolved departments do not rebuild COD's regions by area: {bad}")
    print(f"  the 58 departments dissolve into {N_UNITS} units and rebuild every region's area "
          f"within {AREA_TOL:.1%}")

    units["geo_id"] = units["unit"]
    units["name"] = units["unit"].map(UNIT_NAME)
    units["pop"] = units["unit"].map(pop).astype("int64")
    if units["name"].isna().any() or units["pop"].isna().any():
        raise SystemExit("a unit has no name or no population")

    # ---- Kontur ----
    hexes = geo_checks.read_layer(gpkg, "Kontur CM")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, "pop"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['pop'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every unit ({lost:,.0f} people, "
          f"{100 * lost / pts['pop'].sum():.3f}%) dropped")
    keep = ~outside
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    print(f"  Kontur {tot:,.0f} against COD-PS 2025 {CODPS_2025:,}: ratio {tot / CODPS_2025:.3f}")
    if abs(tot / CODPS_2025 - 1) > KONTUR_TOL:
        raise SystemExit("Kontur and COD-PS disagree nationally by more than the tolerance")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print("\n  per unit, COD-PS 2025 against Kontur (the witness; Kontur counts nothing here):")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.unit:<9} {r.name:<20}{r.pop:>11,}  Kontur {r.kontur_pop:>11,}  "
              f"{r.kontur_pop / r.pop:5.2f}x  {r.hexes:>7,} hexes  {r.area_sqkm:>9,.0f} km2")
    geo_checks.ratio_band(dict(zip(units["unit"], units["pop"])),
                          dict(zip(units["unit"], units["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="unit")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    keep_cols = ["geo_id", "unit", "name", "pop", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[keep_cols].to_file(OUT_UNITS, layer="regions", driver="GPKG")
    pd.DataFrame(units[keep_cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False,
                                                                  encoding="utf-8")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    d = a2[["adm2_name1", "adm2_pcode", "adm1_pcode", "unit"]].rename(
        columns={"adm2_name1": "department"}).sort_values("adm2_pcode")
    d.to_csv(OUT_DEPTS, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_HEXES} ({len(out):,} hexes)\n"
          f"wrote {OUT_DEPTS}")


if __name__ == "__main__":
    main()
