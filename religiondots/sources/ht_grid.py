"""Haiti — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/ht/ht_hexes.gpkg.

Ten departments over 27,750 km² is **2,775 km² per unit**, which is nearly twice the
Dominican Republic's ratio on the same island and well inside the range where §8.2's
placement layer pays ([[reference_kontur_resolution_floor]] puts the floor at about 1 km²
per unit and this is far above it).

**Ouest is the reason.** It is 3.98 million people, a third of the country, in a polygon that
runs from Port-au-Prince across the Chaîne des Matheux and up onto the Plateau, and the
capital's own agglomeration is a strip along the bay. Spread its dots evenly over the polygon
and a third of Haiti's colour lands on mountains. Grand'Anse and Nippes have the reverse
problem: the southern peninsula is a spine of mountain with the people on the coastal shelf
either side of it.

There is a third reason that is specific to Haiti. **The country's own population base is a
2003 census carried forward by projection** (`sources/ht_geo.py`). Kontur is built from
buildings and settlement imagery, so it is the only thing in this build that has seen Haiti
since the 2010 earthquake and the 2021 displacement out of Port-au-Prince. It sets no levels
here, only the within-department placement.

**CORRECTED 2026-09-14 (session `f95259a4-clht`): the projection does have sub-departmental
detail.** This docstring said it had none. COD-PS 2024 prints all 140 communes
(`hti_admpop_adm2_2024.csv`), summing exactly to the department figures. So each hex now also
carries its COD-AB commune (`commune`) and that commune's COD-PS population (`commune_pop`),
and `countries.py::_ht_place_weight` uses them for the two communes where every grid is wrong
(Anse-à-Galets on La Gonâve, and Petit-Goâve since the same evening, `_HT_COMMUNE_LEVEL`).
They are NOT used to rescale every commune,
because the projection is wrong in the other direction wherever Haiti has settled since 2003:
Kontur and both WorldPop rasters put Croix-des-Bouquets at 2.2x its COD-PS figure, and that is
Canaan. `sources/ht.md` §12 has the comparison.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a department
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every department are dropped and reported; Kontur's HT
extract is cut at the international border, so what is lost is coastline rather than the
Dominican Republic. The commune join is the same centroid, and is asserted to land in a
commune of the department the hex was given.

Usage:
    python sources/ht_grid.py --fetch    one ~2 MB gzipped gpkg from Kontur, one CSV from HDX
    python sources/ht_grid.py            rebuild from data/raw/ht/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ht")
GEO = os.path.join(ROOT, "data", "geo", "ht")
DEPARTMENTS = os.path.join(GEO, "ht_departements.gpkg")
OUT = os.path.join(GEO, "ht_hexes.gpkg")
# The COD-AB bundle sources/ht_geo.py already fetched; its admin2 layer is the 140 communes.
BOUNDARIES = os.path.join(RAW, "hti_admin_boundaries.shp.zip")
COMMUNE_LAYER = "hti_admin2.shp"
COMMUNE_POP = os.path.join(RAW, "hti_admpop_adm2_2024.csv")
COMMUNE_POP_URL = ("https://data.humdata.org/dataset/95ba5281-fd76-4d3a-ae29-247e1ff26447/"
                   "resource/fc2ae030-c37e-4070-9f81-0f5f1c3d446c/download/"
                   "hti_admpop_adm2_2024.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_HT_20231101.gpkg.gz")
GZ_NAME = "kontur_population_HT_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_HT_20231101.gpkg"

EXPECTED_DEPARTMENTS = 10
EXPECTED_COMMUNES = 140

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-department weight,
# so all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule), and it has to be wide. The reference
# is a 2024 COD-PS projection off a 2003 census, and Haiti is one of the hardest countries in
# the hemisphere to model from buildings: dense unplanned settlement in and around
# Port-au-Prince, dispersed rural *lakou*, and a population that moved twice in fifteen years.
CODPS_POPULATION = 11_899_555
KONTUR_TOLERANCE = 0.35

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(COMMUNE_POP):
        print("GET", COMMUNE_POP_URL)
        r = requests.get(COMMUNE_POP_URL, timeout=300, headers={"User-Agent": UA})
        r.raise_for_status()
        if not r.content.startswith(b"Year,ISO3"):
            raise SystemExit(f"{COMMUNE_POP_URL} is not the COD-PS CSV -- starts "
                             f"{r.content[:40]!r}")
        with open(COMMUNE_POP, "wb") as fh:
            fh.write(r.content)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def attach_communes(out, pts_kept, deps):
    """Give each kept hex its COD-AB commune and that commune's COD-PS 2024 population.

    Asserted: every hex lands in a commune, the commune belongs to the hex's department, all
    140 communes have a hex and a COD-PS row, and the communes sum to each department's own
    COD-PS figure exactly.
    """
    import geopandas as gpd
    import pandas as pd

    com = gpd.read_file(f"zip://{BOUNDARIES}!{COMMUNE_LAYER}")
    if len(com) != EXPECTED_COMMUNES:
        raise SystemExit(f"{COMMUNE_LAYER} has {len(com)} communes, expected {EXPECTED_COMMUNES}")
    com = com[["adm2_pcode", "adm1_pcode", "geometry"]].to_crs(deps.crs)

    j = gpd.sjoin(pts_kept[["geometry"]], com, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts_kept.index)
    if j["adm2_pcode"].isna().any():
        raise SystemExit(f"{int(j['adm2_pcode'].isna().sum())} hexes are in a department but "
                         "in no commune")
    commune = j["adm2_pcode"].astype(str).to_numpy()
    wrong = (j["adm1_pcode"].astype(str).to_numpy() != out["unit"].astype(str).to_numpy())
    if wrong.any():
        raise SystemExit(f"{int(wrong.sum())} hexes have a commune in another department")

    cp = pd.read_csv(COMMUNE_POP)
    cp = cp[cp["ADM2_PCODE"].astype(str).str.fullmatch(r"HT\d{4}")]
    if len(cp) != EXPECTED_COMMUNES or cp["T_TL"].sum() != CODPS_POPULATION:
        raise SystemExit(f"{COMMUNE_POP}: {len(cp)} communes summing to {cp['T_TL'].sum():,}, "
                         f"expected {EXPECTED_COMMUNES} summing to {CODPS_POPULATION:,}")
    by_dep = cp.groupby("ADM1_PCODE")["T_TL"].sum()
    dep_pop = dict(zip(deps["unit"], deps["pop"]))
    off = {d: (int(by_dep.get(d, 0)), int(p)) for d, p in dep_pop.items()
           if int(by_dep.get(d, 0)) != int(p)}
    if off:
        raise SystemExit(f"COD-PS communes do not sum to the department figures: {off}")
    missing = sorted(set(cp["ADM2_PCODE"]) - set(commune))
    if missing:
        raise SystemExit(f"communes with no hex centroid: {missing}")

    out["commune"] = commune
    out["commune_pop"] = pd.Series(commune).map(dict(zip(cp["ADM2_PCODE"], cp["T_TL"]))
                                                ).to_numpy(dtype=float)
    print(f"\n  every hex has a commune of its own department; all {EXPECTED_COMMUNES} communes "
          "have hexes and sum to\n  the department figures exactly")

    # The record for sources/ht.md §12: where Kontur and the projection disagree inside a
    # department. Nothing here is asserted; countries.py acts on one named commune only.
    k = out.groupby("commune")["pop"].sum()
    t = cp.set_index("ADM2_PCODE")
    t["kontur"] = k
    t["norm"] = ((t["kontur"] / t.groupby("ADM1_PCODE")["kontur"].transform("sum"))
                 / (t["T_TL"] / t.groupby("ADM1_PCODE")["T_TL"].transform("sum")))
    t = t.sort_values("norm")
    print("  Kontur's within-department share against COD-PS's (raw Kontur, before "
          "kontur_cap.py):")
    for code, r in list(t.iterrows())[:3] + list(t.iterrows())[-6:]:
        print(f"      {code}  {r['ADM2_FR'][:22]:<22} COD-PS {int(r['T_TL']):>8,}  Kontur "
              f"{r['kontur']:>9,.0f}  {r['norm']:5.2f}x")
    return out


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(DEPARTMENTS):
        raise SystemExit(f"missing {DEPARTMENTS} — run sources/ht_geo.py first")
    for p in (BOUNDARIES, COMMUNE_POP):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run with --fetch (and sources/ht_geo.py --fetch)")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    deps = gpd.read_file(DEPARTMENTS)
    if len(deps) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{DEPARTMENTS} has {len(deps)} departments, "
                         f"expected {EXPECTED_DEPARTMENTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(deps.crs)
    hexes = hexes.to_crs(deps.crs)

    joined = gpd.sjoin(pts, deps[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     the HT extract is cut at the Dominican border, so this is coastline; "
          "dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=deps.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(deps["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"departments with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"departments whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DEPARTMENTS} departments has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CODPS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs COD-PS 2024 {CODPS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and COD-PS disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-department weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-department Kontur/COD-PS ratio (the shape check):")
    codps = dict(zip(deps["unit"], deps["pop"]))
    rows = [(deps.loc[deps["unit"] == u, "name"].iloc[0], int(r["size"]),
             r["sum"] / codps[u]) for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<20} {n:>7,} hexes   {r:5.2f}x")

    out = attach_communes(out, pts[keep.to_numpy()].reset_index(drop=True), deps)

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
