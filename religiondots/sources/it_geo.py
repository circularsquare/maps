"""Italy — placement polygons: the 7,903 comuni, from the GISCO LAU file already on disk.

Writes data/geo/it/it_lau.gpkg with, per comune:
    lau      6-digit ISTAT comune code
    unit     its NUTS 3 provincia — THE COUNTING UNIT, 107 of them
    nuts2    its regione, the first four characters of `unit`
    nuts1    its ripartizione, the first three
    pop      the workbook's POPULATION, used as the fallback placement weight
    ital     Italian citizens resident in the comune       ] ISTAT RCS, and the reason
    foreign  foreign citizens resident in the comune       ] this file has a --fetch
    name     LAU NAME LATIN

Usage:
    python sources/it_geo.py --fetch   # ISTAT comune x citizenship  (~2 MB)
    python sources/it_geo.py

TWO PLACEMENT WEIGHTS, NOT ONE, AND IT IS THE ONLY REASON THIS FILE DOWNLOADS ANYTHING.
Anita, looking at the finished map: *"it does look a bit weird to see hinduism spread like
population-proportionally throughout the italian countryside when in reality i imagine it's
much more urbanized."* She is right, and the diagnosis is not the one the question implies —
Italy's Hindus are **166,000 people and essentially all of them are in the foreign half**, so
no religion-by-settlement statistic was ever going to fix it. ESS does carry `domicil` (big
city / suburbs / town / village / farm) and it was checked: among citizens it gives Islam
**no urban gradient at all** (1.10% village against 1.10% big city) and "Eastern religions" a
4.4x one on about twenty respondents. Nothing there is strong enough to move a dot with.

What was wrong instead is that **every dot in the country was placed by TOTAL comune
population**, so a province's foreign residents were smeared across its countryside in
proportion to where Italians live. ISTAT publishes resident population by comune AND
citizenship, so the fix is to place each half by its own population:

    citizen-half nodes   ->  `ital`     Italian citizens per comune
    foreign-half nodes   ->  `foreign`  foreign citizens per comune

**This changes placement only and never a magnitude** — every province's totals stay
Eurostat's — which is exactly what spec §8.2 asks for: *"where the placement layer ships its
own population, use it rather than approximating."* countries.py's `_ItWeighter` does the
blending, because a node like `islam.sunni` draws from both halves and has to be split
between the two weights in the proportion that province's own arithmetic gives it.

WHAT IT DOES NOT DO, STATED BECAUSE THE FILE SUPPORTS IT AND THIS DOES NOT USE IT. RCS is
comune x INDIVIDUAL citizenship — it would place Indians in the Agro Pontino and Chinese in
Prato specifically, rather than "foreigners" generally. That needs ISTAT's numeric country
code mapped to the ISO-2 codes Eurostat uses, and **ISTAT's own code list is a dead link**:
`Elenco-codici-e-denominazioni-unita-territoriali-estere.zip` is listed on the classification
page and 404s. It is worth less than it sounds, because at 107 province the composition
already carries most of the nationality geography — Prato is its own provincia, and so is
Latina. Recorded per §2.4.

THE VINTAGE IS DELIBERATELY NOT THE CENSUS'S. RCS is 1 January 2025 and the magnitudes are
the 2021 census. That is fine and is the point: a placement weight is a statement about where
people are now, not a count, and nothing downstream sums it.

THREE LEVELS ARE CARRIED, WHICH NO OTHER COUNTRY HERE NEEDS, and the reason is that Italy is
the first country whose two halves are known at different resolutions and whose citizen half
is known at two resolutions by itself. §9z's Greece had the same asymmetry and resolved it by
throwing the finer level away — the foreign half was aggregated up to NUTS 2 because "that
extra level is used for placement rather than for counting, which is the only honest thing to
do with a resolution that only half the data has". **Italy does the opposite, on Anita's
call**, and `countries.py` disaggregates the coarse rows down instead. sources/it.md §3 is
the argument; the short version is that Greece's foreign half is 7.2% of the country and
Italy's is 8.5% holding four fifths of its religious minorities, so the level that only half
the data has is the level that carries most of what the map is for.

The three columns exist because that disaggregation needs a nesting, and NUTS gives one for
free: `ITC11` is in `ITC1` is in `ITC`, by prefix, with no lookup table.

THE COUNTING GEOGRAPHY IS NUTS 3, WHICH IS 107 PROVINCE at about 553,000 people each — finer
per person than every other European country on this map, and about half Kenya's counties
(§9o). None of that is a claim about the citizen half, which is drawn at 20 regioni and 5
ripartizioni and says so.

THE ZERO-PADDING TRAP, FOR THE SECOND TIME AND WIDER THAN GREECE'S. Excel stores the comune
code as a number, so the workbook writes Agliè's `001001` as `1001`. The workbook's codes come
out at THREE different lengths (4, 5 and 6) against the shapefile's uniform 6, and `zfill(6)`
takes the join from 370 matched to 7,903. Greece lost 644 rows to this and the failure looked
like a workbook missing rows; Italy would lose 7,533 and it would look like the wrong country.
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
LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021",
                        "EU-27-LAU-2021-NUTS-2021.xlsx")
RAW = os.path.join(ROOT, "data", "raw", "it")
RCS_ZIP = os.path.join(RAW, "Dati_RCS_cittadinanza_2025.zip")
OUT_DIR = os.path.join(ROOT, "data", "geo", "it")
OUT = os.path.join(OUT_DIR, "it_lau.gpkg")

# demo.istat.it, "Popolazione residente per cittadinanza o paese di nascita", 1 Jan 2025.
# Comune x individual citizenship; only the Italy/not-Italy split is used — see the docstring.
RCS_URL = "https://demo.istat.it/data/rcs/Dati_RCS_cittadinanza_2025.zip"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}
ITALY_CITIZENSHIP = "100"      # `Codice stato di cittadinanza` for Italia

N_LAU = 7903
N_NUTS3 = 107
N_NUTS2 = 21          # 19 regioni + Bolzano and Trento, which NUTS splits
N_NUTS1 = 5


def fetch():
    """demo.istat.it needs certifi's CURRENT bundle, and curl fails where python succeeds.

    `curl` exits 60 (certificate problem) on this host while urllib with `certifi.where()`
    is fine — the same shape sources/gr.py records for Eurostat, and it is a stale local
    trust store rather than a server fault. `pip install -U certifi` if this ever fails.
    """
    import certifi
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(RCS_ZIP):
        print(f"  already on disk ({os.path.getsize(RCS_ZIP):,} bytes)")
        return
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(RCS_URL, headers=UA),
                                timeout=600, context=ctx) as r, \
            open(RCS_ZIP, "wb") as f:
        f.write(r.read())
    print(f"  {os.path.getsize(RCS_ZIP):,} bytes -> {os.path.basename(RCS_ZIP)}")


def _citizenship():
    """comune -> (Italian citizens, foreign citizens), from ISTAT RCS."""
    if not os.path.exists(RCS_ZIP):
        sys.exit(f"missing {RCS_ZIP} — run: python sources/it_geo.py --fetch")
    with zipfile.ZipFile(RCS_ZIP) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1",
                     dtype={"Codice Istat": str, "Codice stato di cittadinanza": str})
    df["code"] = df["Codice Istat"].str.strip()

    # THE FILE STACKS FOUR TERRITORIAL LEVELS IN ONE COLUMN AND MARKS THEM ONLY BY CODE
    # WIDTH. `1` is Nord-ovest, `01` Piemonte, `001` Torino, `001001` Aglie — and summing
    # the column blind gives 294.7M people, five times Italy, which is how this was caught.
    # Worse, `zfill(6)` FIRST turns the aggregates into plausible comune codes (`000001`)
    # and they then fail the join quietly as 117 unmatched rows rather than loudly as a
    # wrong total. Filter on the RAW width, before any padding.
    lvl = df["code"].str.len().value_counts().to_dict()
    df = df[df["code"].str.len() == 6].copy()
    print(f"  code widths in the file: {dict(sorted(lvl.items()))} "
          f"-> keeping the {len(df):,} comune rows")

    df["lau"] = df["code"]
    df["n"] = pd.to_numeric(df["Totale"], errors="coerce").fillna(0.0)
    ital = df[df["Codice stato di cittadinanza"] == ITALY_CITIZENSHIP]
    ital = ital.groupby("lau")["n"].sum()
    forn = df[df["Codice stato di cittadinanza"] != ITALY_CITIZENSHIP]
    forn = forn.groupby("lau")["n"].sum()
    print(f"  RCS: {len(df):,} rows, {df['lau'].nunique():,} comuni, "
          f"{df['Codice stato di cittadinanza'].nunique()} citizenships")
    total = ital.sum() + forn.sum()
    print(f"  {ital.sum():,.0f} Italian citizens + {forn.sum():,.0f} foreign "
          f"= {total:,.0f}")
    # The check that would have caught the stacked levels on the first run.
    if not 55_000_000 < total < 62_000_000:
        sys.exit(f"!! RCS comune rows sum to {total:,.0f}, which is not Italy — the level "
                 f"filter in _citizenship() is wrong")
    return ital, forn


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (IT only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='IT'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip()
    print(f"  {len(g):,} Italian comuni, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="IT")
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(6))
    x["unit"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME LATIN"].astype(str)
    print(f"  {len(x):,} rows, {x['unit'].nunique()} NUTS 3, "
          f"population {x['pop'].sum():,.0f}")

    # spec §8.1, both directions. A one-sided check passes on a file that has silently lost
    # half its rows on the other side.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! comune codes do not agree: {only_shp[:5]} / {only_xls[:5]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")
    g["nuts2"] = g["unit"].str[:4]
    g["nuts1"] = g["unit"].str[:3]

    for lvl, want in (("unit", N_NUTS3), ("nuts2", N_NUTS2), ("nuts1", N_NUTS1)):
        got = g[lvl].nunique()
        print(f"  {lvl}: {got} units")
        if got != want:
            sys.exit(f"!! expected {want} {lvl} units, got {got}")
    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} comuni, got {len(g)}")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate comune codes")
    if (g["pop"] <= 0).any():
        n = int((g["pop"] <= 0).sum())
        print(f"  !! {n} comuni have no population and will attract no dots")

    per1 = g.groupby("nuts1")["pop"].sum()
    print("  population per ripartizione:")
    for u in sorted(per1.index):
        n2 = g[g["nuts1"] == u]["nuts2"].nunique()
        n3 = g[g["nuts1"] == u]["unit"].nunique()
        print(f"    {u}  {per1[u]:>11,.0f}   {n2:>2} regioni, {n3:>3} province")

    print("reading ISTAT citizenship by comune…")
    ital, forn = _citizenship()
    # BOTH DIRECTIONS, because RCS is a different vintage AND a different agency's comune
    # list: 2025 against the LAU file's 2021. Comuni merge in Italy every year, so a
    # one-sided check would pass on a file that had lost a hundred of them.
    only_lau = sorted(set(g["lau"]) - set(ital.index))
    only_rcs = sorted(set(ital.index) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(ital.index)):,} matched, "
          f"{len(only_lau)} LAU-only, {len(only_rcs)} RCS-only")
    g["ital"] = g["lau"].map(ital).fillna(0.0)
    g["foreign"] = g["lau"].map(forn).fillna(0.0)
    if only_lau:
        # Not fatal: a comune RCS does not carry falls back to `pop` in the weighter, which
        # is exactly the old behaviour for that one comune. Named rather than silently zeroed.
        print(f"  !! {len(only_lau)} comuni have no RCS row and keep the population "
              f"weight: {only_lau[:6]}")
    share = 100 * g["foreign"].sum() / (g["ital"].sum() + g["foreign"].sum())
    print(f"  foreign share of the joined comuni: {share:.2f}%")
    if not 5.0 < share < 12.0:
        sys.exit(f"!! foreign share {share:.2f}% is outside the plausible band — "
                 f"the Italy/not-Italy split is probably reading the wrong column")
    top = g.sort_values("foreign", ascending=False).head(5)
    print("  most foreign residents:")
    for _, r in top.iterrows():
        print(f"    {r['name'][:28]:<30} {r['foreign']:>9,.0f}  "
              f"{100 * r['foreign'] / max(r['ital'] + r['foreign'], 1):5.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "unit", "nuts2", "nuts1", "pop", "ital", "foreign", "name",
             "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} comuni)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        main()
