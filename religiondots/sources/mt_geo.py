"""Malta: the 68 localities as polygons, and the Kontur placement layer inside them.

Writes
    data/geo/mt/mt_units.gpkg       68 localities: `unit` (the census name), `lau_id`, `district`
    data/geo/mt/mt_hexes.gpkg       Kontur 400 m hexagons cut to the localities: `unit`, `pop`
    data/geo/mt/mt_lookup.csv       census name -> LAU code, district

## THE UNITS ARE EUROSTAT'S LAU 2021, WHICH IS THE CENSUS'S OWN LIST

Volume 1 groups "all Maltese localities ... according to the Local Administrative Unit (LAU)
classification" (p.172), and GISCO's LAU 2021 layer, already on disk for Cyprus, carries all 68
under `CNTR_CODE='MT'`. The census prints names and no codes, so the join is on the name
(hyphens and apostrophes folded) and is checked three ways that the name does not decide:
  1. **District from the code.** A Maltese LAU code's first five characters are the district
     (MT011 Southern Harbour ... MT026 Gozo and Comino), and every joined locality must sit in
     the district Table 5.3 prints it under. The twins that could swap, Ir-Rabat with Ir-Rabat,
     Għawdex and Ħaż-Żebbuġ with Iż-Żebbuġ, are in different districts.
  2. **Area.** Table 1.10 (pp.106-107) prints each locality's land area; GISCO's `AREA_KM2`
     must agree within 7% for every locality. Rabat's 26.60 km2 against Rabat, Gozo's 2.90,
     and Żebbuġ's 8.66 against 7.56, both fall outside that.
  3. **Population.** Eurostat's own LAU population (the workbook's `POPULATION`, a register
     figure of another date) against Table 1.2, in a wide band.

## THE PLACEMENT LAYER, AND WHY IT IS CUT RATHER THAN JOINED ON CENTROIDS

The median locality is under 3 km2, about four Kontur hexes, which is below spec §8.2e's floor
for a centroid join: small harbour towns (L-Isla 0.16 km2, Ta' Xbiex 0.29) would get no hex at
all. So each hex is intersected with the localities and its people shared over its area inside
them, which is its land because every part of Malta is in a locality. That gives every locality at
least one piece, and the large rural ones (Ir-Rabat, Il-Mellieħa,
Is-Siġġiewi, L-Imġarr) the dozens of pieces that keep dots off the Dingli cliffs and the
Majjistral park. Whether the grid is kept at all is decided on its per-locality agreement with
Table 1.2, which main() prints; sources/mt.md §5 records the numbers and the decision.

Usage:
    python sources/mt_geo.py --fetch    Kontur MT 2023-11-01 (about 33 KB gzipped)
    python sources/mt_geo.py            build from data/raw/mt/ and the shared LAU bundle
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import mt                                                       # noqa: E402  the census tables
from geo_checks import ratio_band, read_layer                   # noqa: E402  shared, not copied

RAW = os.path.join(ROOT, "data", "raw", "mt")
GEO = os.path.join(ROOT, "data", "geo", "mt")
UNITS = os.path.join(GEO, "mt_units.gpkg")
HEXES = os.path.join(GEO, "mt_hexes.gpkg")
LOOKUP = os.path.join(GEO, "mt_lookup.csv")

LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326", "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021", "EU-27-LAU-2021-NUTS-2021.xlsx")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MT_20231101.gpkg.gz")
GZ = os.path.join(RAW, "kontur_population_MT_20231101.gpkg.gz")
GPKG = GZ[:-3]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

PAGES_T110 = (105, 106)        # pp.106-107
NATIONAL_KM2 = 315.15
AREA_BAND = (0.93, 1.07)       # GISCO AREA_KM2 / Table 1.10, every locality
POP_BAND = (0.70, 1.45)        # Eurostat LAU POPULATION / Table 1.2, every locality
KONTUR_NATIONAL = (0.85, 1.30)  # Kontur 2023-11 inside the localities / census November 2021
EQUAL_AREA = "EPSG:3035"

CODE_DISTRICT = {"MT011": "Southern Harbour", "MT012": "Northern Harbour",
                 "MT013": "South Eastern", "MT014": "Western", "MT015": "Northern",
                 "MT026": "Gozo and Comino"}
AREA = re.compile(r"\d{1,3}\.\d{2}")


def key(s):
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("‐", "-").replace("‑", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().casefold()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(GZ) and os.path.getsize(GZ) > 1000:
        print("already have", GZ)
        return
    print("GET", GZ_URL)
    r = requests.get(GZ_URL, timeout=600, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(GZ + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(GZ + ".part", GZ)
    print(f"  {os.path.getsize(GZ):,} bytes")


def unpack():
    if os.path.exists(GPKG) and os.path.getsize(GPKG) > 0:
        return
    if not os.path.exists(GZ):
        raise SystemExit(f"missing {GZ}; run with --fetch")
    with gzip.open(GZ, "rb") as src, open(GPKG + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(GPKG + ".part", GPKG)
    with open(GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{GPKG} is not a GeoPackage")


def read_areas(doc):
    """Table 1.10 -> {locality: (area km2, population)}, rows asserted in mt.SEQUENCE order."""
    rows = []
    for pno in PAGES_T110:
        ls = mt.lines_of(doc, pno)
        if "TABLE 1.10. Population density in Malta by locality" not in ls[0]:
            raise SystemExit(f"p.{pno + 1} is not Table 1.10: {ls[0]!r}")
        start = ls.index("per km2") + 1
        body = mt._strip_page_number(ls[start:], pno, f"Table 1.10 p.{pno + 1}")
        rows += mt.read_rows(body, 3, lambda t: bool(mt.COUNT.fullmatch(t) or AREA.fullmatch(t)),
                             f"Table 1.10 p.{pno + 1}")
    if [r[0] for r in rows] != mt.SEQUENCE:
        raise SystemExit("Table 1.10: rows are not in the census's locality order")
    return [(lab, float(v[0]), mt.num(v[1])) for lab, v in rows]


def main():
    import fitz
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    unpack()

    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Malta: localities and placement\n")
    doc = fitz.open(mt.PDF)
    pop_rows = mt.read_population(doc)
    census_pop = {lab: t for lab, (_m, _f, t) in pop_rows if lab in dict(mt.LOCALITIES).values()
                  or lab in {loc for _, loc in mt.LOCALITIES}}
    census_pop = {loc: dict((lab, t) for lab, (_m, _f, t) in pop_rows)[loc]
                  for _, loc in mt.LOCALITIES}
    area_rows = read_areas(doc)
    say([p for _, _, p in area_rows] == [t for _, (_m, _f, t) in pop_rows],
        "Table 1.10's population column equals Table 1.2's in all 77 rows")
    census_km2 = {lab: a for lab, a, _ in area_rows if lab in census_pop}
    say(len(census_km2) == mt.N_LOCALITIES and area_rows[0][1] == NATIONAL_KM2,
        f"Table 1.10 gives an area for all 68 localities; Malta {NATIONAL_KM2} km2")
    dist_km2 = {}
    for lab, a, _ in area_rows:
        dist_km2.setdefault(lab, []).append(a)
    say(all(abs(sum(census_km2[loc] for loc in locs) - dist_km2[d][-1]) <= 0.01 * len(locs)
            for d, locs in mt.DISTRICTS.items()),
        "the locality areas sum to their district's within rounding")

    # GISCO LAU 2021
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; the GISCO LAU 2021 bundle is a shared asset")
    g = read_layer(LAU_SHP, "GISCO LAU 2021 MT", where="CNTR_CODE='MT'")
    g["lau_id"] = g["LAU_ID"].astype(str).str.strip()
    say(len(g) == mt.N_LOCALITIES and not g["lau_id"].duplicated().any(),
        f"GISCO has {len(g)} Maltese LAUs with distinct codes (expected {mt.N_LOCALITIES})")

    by_key = {}
    for i, name in zip(g.index, g["LAU_NAME"]):
        by_key.setdefault(key(name), []).append(i)
    dup = {k: v for k, v in by_key.items() if len(v) > 1}
    district = {loc: d for d, loc in mt.LOCALITIES}
    match = {loc: by_key.get(key(loc), []) for _, loc in mt.LOCALITIES}
    unmatched = [loc for loc, v in match.items() if len(v) != 1]
    spare = sorted(set(g.index) - {v[0] for v in match.values() if len(v) == 1})
    say(not dup and not unmatched and not spare,
        "the name join is one-to-one: 68 census localities, 68 GISCO polygons"
        + (f"; duplicate keys {list(dup)[:4]}, unmatched {unmatched[:6]}, spare "
           f"{list(g.loc[spare, 'LAU_NAME'])[:6]}" if dup or unmatched or spare else ""))
    if not ok:
        raise SystemExit("join FAILED")

    g["unit"] = None
    for loc, v in match.items():
        g.loc[v[0], "unit"] = loc
    g["district"] = g["unit"].map(district)

    # witness 1: the code's district
    code_d = g["lau_id"].str[:5].map(CODE_DISTRICT)
    wrong = g[code_d != g["district"]]
    say(len(wrong) == 0, "every locality's LAU code is in the district Table 5.3 prints it under"
        + (f"; wrong: {list(zip(wrong['unit'], wrong['lau_id']))[:6]}" if len(wrong) else ""))

    # witness 2: area against Table 1.10
    xl = pd.read_excel(LAU_XLSX, sheet_name="MT")
    xl["lau_id"] = xl["LAU CODE"].astype(str).str.strip()
    g = g.merge(xl[["lau_id", "POPULATION", "LAU NAME NATIONAL"]], on="lau_id", how="left")
    say(g["POPULATION"].notna().all() and (g["LAU NAME NATIONAL"] == g["LAU_NAME"]).all(),
        "the LAU workbook has every code, and its national name is the layer's")
    ratios = g.set_index("unit")["AREA_KM2"] / pd.Series(census_km2)
    lo, hi = ratios.min(), ratios.max()
    say(AREA_BAND[0] <= lo and hi <= AREA_BAND[1],
        f"GISCO area / Table 1.10 area runs {lo:.3f} ({ratios.idxmin()}) to {hi:.3f} "
        f"({ratios.idxmax()}), inside {AREA_BAND}; total {g['AREA_KM2'].sum():.2f} km2")
    swapped = [(a, b, census_km2[b] / census_km2[a])
               for a, b in (("Ir-Rabat", "Ir-Rabat, Għawdex"), ("Ħaż-Żebbuġ", "Iż-Żebbuġ"))]
    say(all(not AREA_BAND[0] <= r <= AREA_BAND[1] for _, _, r in swapped),
        "the band would catch the two same-name swaps: "
        + ", ".join(f"{a} as {b} {r:.3f}" for a, b, r in swapped))

    # witness 3: Eurostat's LAU population against Table 1.2
    band = ratio_band(census_pop, g.set_index("unit")["POPULATION"].astype(float).to_dict(),
                      *POP_BAND, what="locality")
    say(len(band) == mt.N_LOCALITIES,
        f"Eurostat LAU population / census runs {band['ratio'].min():.2f} to "
        f"{band['ratio'].max():.2f}, inside {POP_BAND} (a register figure of another date)")
    if not ok:
        raise SystemExit("unit checks FAILED")

    os.makedirs(GEO, exist_ok=True)
    units = g[["unit", "lau_id", "district", "geometry"]].copy()
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=g.crs).to_crs("EPSG:4326")
    units.to_file(UNITS + ".part.gpkg", driver="GPKG", layer="mt_units")
    os.replace(UNITS + ".part.gpkg", UNITS)
    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["unit", "lau_id", "district"])
        wr.writerows(units[["unit", "lau_id", "district"]].itertuples(index=False))
    print(f"\n  wrote {UNITS} and {LOOKUP}")

    # ---- Kontur
    hexes = read_layer(GPKG, "Kontur MT")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes[hexes[popcol] > 0].to_crs(EQUAL_AREA).reset_index(drop=True)
    hexes["hid"] = np.arange(len(hexes))
    hexes["hex_km2"] = hexes.area / 1e6
    k_all = float(hexes[popcol].sum())
    print(f"\n  Kontur MT: {len(hexes):,} populated hexes, {k_all:,.0f} people, median hex "
          f"{hexes['hex_km2'].median():.3f} km2")
    u = units.to_crs(EQUAL_AREA)
    ukm2 = u.set_index("unit").area / 1e6
    print(f"  localities: median {ukm2.median():.2f} km2, {ukm2.median() / 0.74:.1f} hexes "
          f"(spec §8.2e); {int((ukm2 < 0.74).sum())} smaller than one hex")

    # the centroid join spec §8.2e measures
    cent = gpd.GeoDataFrame({"hid": hexes["hid"]}, geometry=hexes.centroid, crs=EQUAL_AREA)
    cj = gpd.sjoin(cent, u[["unit", "geometry"]], how="inner", predicate="within")
    per = cj.groupby("unit").size().reindex(u["unit"]).fillna(0)
    print(f"  centroid join: median {per.median():.0f} hexes per locality, "
          f"{int((per == 0).sum())} localities with none: "
          f"{sorted(per[per == 0].index)[:10]}")

    # the layer actually built: hexes cut to localities, each hex's people shared over its area
    # INSIDE the localities. Every part of Malta's land is in a locality, so that is the hex's land
    # area. Dividing by the whole hex (`hex_km2`) gave the sea part of the 127 coastal hexes its
    # share of the people and dropped it: 29,261 of Kontur's 535,078, and weight tilted inland in
    # every harbour town (sources/mt.md, review of 2026-09-15).
    pieces = gpd.overlay(hexes[["hid", popcol, "hex_km2", "geometry"]],
                         u[["unit", "geometry"]], how="intersection", keep_geom_type=True)
    land_km2 = pieces.area.groupby(pieces["hid"]).transform("sum") / 1e6
    pieces["pop"] = pieces[popcol] * (pieces.area / 1e6) / land_km2
    pieces = pieces[pieces["pop"] > 0].reset_index(drop=True)
    kept = float(pieces["pop"].sum())
    print(f"  cut layer: {len(pieces):,} pieces; {kept:,.0f} of Kontur's {k_all:,.0f} people are "
          f"in hexes touching a locality ({100 * kept / k_all:.1f}%; the rest are wholly at sea)")
    national = kept / mt.RESIDENTS
    say(KONTUR_NATIONAL[0] <= national <= KONTUR_NATIONAL[1],
        f"Kontur inside the localities is {national:.3f}x the census's {mt.RESIDENTS:,} "
        f"(2023-11 against November 2021), inside {KONTUR_NATIONAL}")
    kpu = pieces.groupby("unit")["pop"].sum().reindex(u["unit"]).fillna(0)
    rel = (kpu / pd.Series(census_pop)) / national
    q = rel.quantile([0.1, 0.5, 0.9])
    print(f"  per locality, Kontur share / census share: p10 {q[0.1]:.2f}, median {q[0.5]:.2f}, "
          f"p90 {q[0.9]:.2f}; Spearman {kpu.rank().corr(pd.Series(census_pop).rank()):.3f}")
    for name, s in (("lowest", rel.nsmallest(5)), ("highest", rel.nlargest(5))):
        print(f"     {name}: " + ", ".join(f"{loc} {v:.2f} ({census_pop[loc]:,})"
                                          for loc, v in s.items()))
    npieces = pieces.groupby("unit").size().reindex(u["unit"]).fillna(0)
    print(f"  pieces per locality: median {npieces.median():.0f}, min {npieces.min():.0f}, "
          f"{int((npieces < 3).sum())} with fewer than three")

    empty = sorted(set(u["unit"]) - set(pieces["unit"]))
    out = pieces[["unit", "pop", "geometry"]]
    if empty:
        fill = u[u["unit"].isin(empty)][["unit", "geometry"]].copy()
        fill["pop"] = fill["unit"].map(census_pop).astype(float)
        out = pd.concat([out, fill[["unit", "pop", "geometry"]]], ignore_index=True)
        print(f"  {len(empty)} localities with no Kontur piece get their own polygon at census "
              f"population: {empty}")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=EQUAL_AREA).to_crs("EPSG:4326")
    say(out["unit"].nunique() == mt.N_LOCALITIES,
        f"the placement layer covers all {mt.N_LOCALITIES} localities")
    w, s, e, n = out.total_bounds
    say(14.15 < w < e < 14.60 and 35.78 < s < n < 36.10,
        f"bbox {w:.3f} {s:.3f} {e:.3f} {n:.3f} is Malta")
    if not ok:
        raise SystemExit("placement checks FAILED")
    out.to_file(HEXES + ".part.gpkg", driver="GPKG", layer="mt_hexes")
    os.replace(HEXES + ".part.gpkg", HEXES)
    print(f"\n  wrote {HEXES} ({len(out):,} pieces)")


if __name__ == "__main__":
    main()
