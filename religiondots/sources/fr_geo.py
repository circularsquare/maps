"""France — placement polygons: 34,966 communes, from the GISCO LAU file already on disk.

Writes data/geo/fr/fr_lau.gpkg with, per commune:
    lau      5-character INSEE commune code
    nuts3    its département, from the GISCO correspondence workbook
    unit     the same thing — **THE COUNTING UNIT, 99 of them**, since 2026-09-08
    nuts2    its ancienne région, the first four characters of that
    pop      the workbook's POPULATION, the fallback placement weight
    french   French nationals resident there    ] INSEE RP 2021 TD_NAT1, and the
    foreign  foreign nationals resident there   ] reason this file has a --fetch
    name     LAU NAME LATIN

Usage:
    python sources/fr_geo.py --fetch   # INSEE TD_NAT1, commune x nationality (~4.5 MB)
    python sources/fr_geo.py

TWO FILES THAT WERE ALREADY HERE, AND ONE JOIN — Greece's §9z arrangement exactly. The GISCO
LAU 2021 bundle ships `LAU_RG_01M_2021_4326.shp` and `EU-27-LAU-2021-NUTS-2021.xlsx` together,
the workbook's `FR` sheet maps every commune to its NUTS 3, and NUTS 3 to NUTS 2 *is* a prefix.
34,966 rows on both sides and 34,966 in the intersection.

THE COUNTING GEOGRAPHY IS NUTS 3 SINCE 2026-09-08 — 94 départements plus the five overseas
régions, 99 units at about 680,000 people each. **This paragraph used to say the opposite**,
and the reason it changed is worth keeping: it argued that a Muslim dot in Île-de-France sits
where Île-de-France's people are, *"which means Paris intra-muros and not Seine-Saint-Denis.
That is the single biggest thing wrong with this country and it is a property of the counting
geography, not of the placement."* Both halves of that were right, and the second half is why
the fix was a counting change rather than a placement one.

The foreign half was always available at this level and was declined on Greece's
never-mix-resolutions rule, which Italy (§9as) showed has an unstated premise. Île-de-France
now draws as eight départements — **Seine-Saint-Denis 21.5% Muslim against Seine-et-Marne's
13.9%**, where all eight used to show 16.3%. The citizen half still has only the région, so
`nuts2` stays on this layer and every citizen row says which région its composition came from.

AND THE ZERO-PADDING TRAP DOES NOT BITE HERE, FOR A REASON WORTH KNOWING. Greece lost 644 LAU
codes to Excel stripping their leading zeros (§9z). France has the same exposure — 01001 Abergement-
Clémenciat and every commune in départements 01-09 — and escapes it because **Corsica's codes are
`2A001` and `2B033`.** One alphanumeric value anywhere in the column forces pandas to read the
whole column as strings, so every leading zero survives. The guard below is kept anyway: a future
vintage that drops Corsica, or a re-export that splits the sheet, would silently re-introduce it.
"""

import argparse
import io
import os
import ssl
import sys
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021",
                        "EU-27-LAU-2021-NUTS-2021.xlsx")
OUT_DIR = os.path.join(ROOT, "data", "geo", "fr")
OUT = os.path.join(OUT_DIR, "fr_lau.gpkg")

# --- the second placement weight, added 2026-09-08 -----------------------------------------
# §9as's finding: a country with a citizen half and a nationality-derived foreign half was
# placing BOTH by total population, so the foreign half was scattered in proportion to where
# CITIZENS live. France is the worst case of it on the map, because its counting units are 26
# régions of 2.6M people — the coarsest here — so the placement weight is doing more of the
# work of making the country look like a country than anywhere else (see `_fr_place_weight`).
#
# INSEE's RP 2021 detailed table TD_NAT1 is `population par sexe, âge et nationalité`, at
# CODGEO — the same five-character commune code GISCO carries as LAU_ID, so there is no
# crosswalk. `INATC` is the condensed nationality indicator: 1 Français, 2 Étrangers.
# Placement only, never a magnitude (spec §8.2).
RAW = os.path.join(ROOT, "data", "raw", "fr")
NAT = os.path.join(RAW, "TD_NAT1_2021.csv")
NAT_URL = ("https://www.insee.fr/fr/statistiques/fichier/8202752/"
           "TD_NAT1_2021_csv.zip")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}


def fetch():
    import certifi
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(NAT):
        print(f"  already on disk ({os.path.getsize(NAT):,} bytes)")
        return
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(NAT_URL, headers=UA),
                                timeout=900, context=ctx) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with open(NAT, "wb") as f:
            f.write(z.read(name))
    print(f"  {len(blob):,} bytes zipped -> {os.path.basename(NAT)} "
          f"({os.path.getsize(NAT):,} bytes)")


def _nationality():
    """commune -> (French nationals, foreign nationals), from INSEE TD_NAT1 2021."""
    if not os.path.exists(NAT):
        sys.exit(f"missing {NAT} — run: python sources/fr_geo.py --fetch")
    df = pd.read_csv(NAT, sep=";", encoding="latin-1",
                     dtype={"CODGEO": str, "INATC": str, "NIVGEO": str})
    # NIVGEO is COM or ARM — ARM is the municipal arrondissements of Paris, Lyon and
    # Marseille, which are ALSO counted inside their commune's COM row. Summing both
    # double-counts 2.7M people in exactly the three cities the map most wants right.
    lvl = df["NIVGEO"].value_counts().to_dict()
    df = df[df["NIVGEO"] == "COM"].copy()
    print(f"  NIVGEO in the file: {lvl} -> keeping COM only")
    df["n"] = pd.to_numeric(df["NB"], errors="coerce").fillna(0.0)
    fr_ = df[df["INATC"] == "1"].groupby("CODGEO")["n"].sum()
    fo = df[df["INATC"] == "2"].groupby("CODGEO")["n"].sum()
    total = fr_.sum() + fo.sum()
    print(f"  {len(df):,} rows, {df['CODGEO'].nunique():,} communes")
    print(f"  {fr_.sum():,.0f} French + {fo.sum():,.0f} foreign = {total:,.0f} "
          f"({100 * fo.sum() / total:.2f}% foreign)")
    if not 62_000_000 < total < 72_000_000:
        sys.exit(f"!! TD_NAT1 sums to {total:,.0f}, which is not France — check NIVGEO "
                 f"and INATC")
    return fr_, fo

from fr import NUTS2, DOM, NOT_DRAWN     # noqa: E402  — one definition of the unit list

# Every unit sources/fr.py emits counts for: 21 metropolitan régions from ESS plus the five
# overseas régions from Pew. Corsica is in neither and its communes are dropped below.
DRAWN = {**NUTS2, **{u: n for u, (n, _) in DOM.items()}}

# 94 metropolitan departements + the five overseas regions, each of which is one NUTS 3 unit
# of its own. Asserted rather than counted, so a NUTS revision fails loudly.
N_UNITS = 99


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (FR only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='FR'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip()
    print(f"  {len(g):,} French communes, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="FR")
    x["lau"] = x["LAU CODE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # See the docstring: Corsica keeps this column textual, so no code should need padding.
    # If one does, Excel has re-typed the column and the join is about to fail one-sidedly.
    short = x[x["lau"].str.len() < 5]
    if len(short):
        sys.exit(f"!! {len(short)} LAU codes are under five characters "
                 f"({short['lau'].head(5).tolist()}) — Excel has stripped leading zeros; "
                 f"zfill(5) is the fix, and check whether Corsica left the sheet")
    x["nuts3"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME LATIN"].astype(str)
    print(f"  {len(x):,} rows, {x['nuts3'].nunique()} NUTS 3, "
          f"population {x['pop'].sum():,.0f}")

    # spec §8.1, both directions. A one-sided check passes on a file that has silently lost
    # half its rows on the other side.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! LAU codes do not agree: {only_shp[:5]} / {only_xls[:5]}")

    g = g.merge(x[["lau", "nuts3", "pop", "name"]], on="lau", how="left")
    # THE COUNTING UNIT IS NUTS 3 SINCE 2026-09-08 — 94 départements plus the five overseas
    # régions, which are each one NUTS 3 unit of their own. `nuts2` is kept because every
    # check below, and the citizen half's composition, is still about the région.
    g["nuts2"] = g["nuts3"].str[:4]
    g["unit"] = g["nuts3"]

    # THE DRAWN/NOT-DRAWN DECISION IS STILL THE RÉGION'S, so it is checked on `nuts2` even
    # though the counting unit is now `nuts3`. Corsica is dropped as a région, not as two
    # départements, which is the level at which fr.py made the call.
    seen = sorted(g["nuts2"].unique())
    stray = sorted(set(seen) - set(DRAWN) - set(NOT_DRAWN))
    if stray:
        sys.exit(f"!! NUTS 2 codes this build has never heard of: {stray}")

    # DROP WHAT IS NOT COUNTED. sources/fr.py deliberately does not borrow a composition for
    # Corsica, so its communes must not be in the placement layer either — a polygon with no
    # counted unit behind it is either an empty region on the map or a crash, depending on
    # which loop reaches it first.
    drop = g[~g["nuts2"].isin(DRAWN)]
    if len(drop):
        print(f"  dropping {len(drop):,} communes in régions fr.py does not draw:")
        for u, n in drop.groupby("nuts2").size().sort_index().items():
            print(f"    {u}  {NOT_DRAWN.get(u, '?'):<14} {n:>6,} communes, "
                  f"{drop[drop['nuts2'] == u]['pop'].sum():>10,.0f} people")
        g = g[g["nuts2"].isin(DRAWN)]

    regions = sorted(g["nuts2"].unique())
    if set(regions) != set(DRAWN):
        sys.exit(f"!! NUTS 2 set is not the expected {len(DRAWN)}: "
                 f"{sorted(set(regions) ^ set(DRAWN))}")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate LAU codes")

    units = sorted(g["unit"].unique())
    if len(units) != N_UNITS:
        sys.exit(f"!! expected {N_UNITS} NUTS 3 counting units, got {len(units)}")
    per_reg = g.groupby("nuts2").agg(pop=("pop", "sum"), n=("lau", "size"),
                                     d=("unit", "nunique"))
    print(f"  {len(g):,} communes over {len(units)} counted units (NUTS 3), "
          f"{len(g) / len(units):,.0f} placement polygons each; by région:")
    for u in regions:
        print(f"    {u}  {DRAWN[u]:<26} {per_reg.loc[u, 'pop']:>10,.0f} people, "
              f"{per_reg.loc[u, 'd']:>3} départements, {per_reg.loc[u, 'n']:>5,} communes")

    print("reading INSEE nationality by commune…")
    fr_, fo = _nationality()
    only_lau = sorted(set(g["lau"]) - set(fr_.index))
    only_ins = sorted(set(fr_.index) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(fr_.index)):,} matched, "
          f"{len(only_lau)} LAU-only, {len(only_ins)} INSEE-only")
    g["french"] = g["lau"].map(fr_).fillna(0.0)
    g["foreign"] = g["lau"].map(fo).fillna(0.0)
    if only_lau:
        # THE OVERSEAS RÉGIONS ARE THE EXPECTED CASE HERE and they are fine. TD_NAT1 is
        # metropolitan; the DOM are drawn from Pew as whole units (§9ag) and their people
        # are overwhelmingly French nationals anyway, so a DOM commune keeps the population
        # weight and nothing about it changes.
        miss = g[g["lau"].isin(only_lau)]
        print(f"  !! {len(only_lau)} communes have no INSEE row and keep the population "
              f"weight, over units {sorted(miss['unit'].unique())}")
    share = 100 * g["foreign"].sum() / max(g["french"].sum() + g["foreign"].sum(), 1)
    print(f"  foreign share of the joined communes: {share:.2f}%")
    if not 4.0 < share < 12.0:
        sys.exit(f"!! foreign share {share:.2f}% is outside the plausible band")
    top = g.sort_values("foreign", ascending=False).head(5)
    print("  most foreign residents:")
    for _, r in top.iterrows():
        print(f"    {r['name'][:28]:<30} {r['foreign']:>9,.0f}  "
              f"{100 * r['foreign'] / max(r['french'] + r['foreign'], 1):5.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "nuts3", "nuts2", "unit", "pop", "french", "foreign", "name",
             "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} communes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        main()
