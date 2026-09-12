"""Panama — boundaries for the provinces and comarcas LAPOP can be drawn on.

Writes data/geo/pa/pa_provincias.gpkg and data/geo/pa/pa_lookup.csv.

OCHA COD-AB Panama (`cod-ab-pan`), the **shapefile** bundle rather than the geodatabase on
§12's Chile rule, read with `engine="fiona"`. ADM1 is 13 features: ten provinces and three
comarcas with the rank of a province.

## THE JOIN IS ON NAME, FROM LAPOP'S OWN VALUE LABELS, AND THE CODE JOIN IS A TRAP

Guatemala joins on the code because LAPOP's `prov` is 200 plus the official department number
and COD's pcode is the same number. **Panama looks identical and is not**, in exactly El
Salvador's way: LAPOP's `prov` is 700 plus the official province number, and **COD's `PA`
pcodes are alphabetical by name**. Two of the ten coincide, Bocas del Toro at 01 and Darien at
05, which is enough for a spot check to pass; the other eight are wrong, and the worst pairs
`prov=708` Panama, 2.09 million people, onto `PA08` Kuna Yala, which has 32,016.

There is no guesswork in the names. The Grand Merge's `prov` value-label set spells all ten
out, so this file reads them rather than inferring them from a numbering:

    701 Bocas del Toro   702 Cocle    703 Colon     704 Chiriqui   705 Darien
    706 Herrera          707 Los Santos            708 Panama     709 Veraguas
    712 Comarca Ngabe Bugle

One alias is needed: LAPOP's `Comarca Ngabe Bugle` against COD's `Ngabe Bugle`.

## PANAMA AND PANAMA OESTE ARE ONE UNIT, BECAUSE THE SURVEY NEVER SPLIT THEM

Panama Oeste became a province in 2014 out of the western districts of Panama. **LAPOP never
adopted it**: there is no `713` in any wave, and the 2023 round still codes 789 respondents to
`708`. So the two COD polygons are dissolved into one drawn unit carrying both populations,
2,092,950 people. That is a real loss of resolution and it is stated in `grain=`; drawing them
as two units on one measurement would claim a comparison nobody made.

The dissolve is also the strongest witness that `708` is the pre-2014 province rather than the
post-2014 one: `708` is 50.9% of the pooled sample against 51.5% of the population, and
Panama alone is 35.4%.

## KUNA YALA AND EMBERA KEEP THEIR POLYGONS AND ARE NOT DRAWN

LAPOP has no code for either, in any wave. They stay in this file and in the hex grid, and
`sources/pa.py` leaves them out of `pa.csv` — Ecuador's Galapagos, treated the same way and
for the same reason (§9bn): the line is whether anything measured the place, not how thin the
sample is. Together they are 44,374 people, 1.09% of Panama, and they are in `gap=`.

Usage:
    python sources/pa_geo.py --fetch    one ~6.9 MB zip from HDX, plus a 5 KB CSV
    python sources/pa_geo.py            rebuild from data/raw/pa/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pa")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "pa")
OUT = os.path.join(OUT_DIR, "pa_provincias.gpkg")
LOOKUP = os.path.join(OUT_DIR, "pa_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "pan_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/19b220b3-2802-432b-b93d-b1a2f2ee347a/resource/"
        "f31f539a-45fb-47fc-b7a6-d899085713b8/download/pan_admin_boundaries.shp.zip",
    "pan_admpop_adm1_2023.csv":
        "https://data.humdata.org/dataset/e0ceea6c-f35f-4883-b43e-8239951992d9/resource/"
        "abcef221-8bc3-4d5e-80f4-c87dbadbcf2d/download/pan_admpop_adm1_2023.csv",
}

# Verbatim from the Grand Merge's own `prov` value-label set, read with pyreadstat's
# `metadataonly=True`. Not inferred from a numbering, which is the whole point.
LAPOP_PROVINCES = {
    701: "Bocas del Toro", 702: "Coclé",     703: "Colón",    704: "Chiriquí",
    705: "Darién",         706: "Herrera",   707: "Los Santos",
    708: "Panamá",         709: "Veraguas",  712: "Comarca Ngäbe Buglé",
}

# LAPOP's name -> COD's name, where the two differ. One entry, and it is two differences at
# once: the `Comarca` prefix, and the vowel. LAPOP and INEC write **Ngäbe**, COD-AB writes
# **Ngöbe**, and folding accents away does not close that gap because a is not o. The comarca
# is the third largest unit in the country at 212,084 people, so the alias is load-bearing.
ALIASES = {"Comarca Ngäbe Buglé": "Ngöbe Buglé"}

# Panama Oeste is inside LAPOP's `708`, so its polygon is dissolved into Panama's.
MERGE_INTO = {"PA11": "PA12"}
MERGED_NAME = "Panamá y Panamá Oeste"

# The two comarcas LAPOP never sampled. They keep polygons and hexes and draw no religion.
NOT_SAMPLED = {"PA06": "Emberá", "PA08": "Kuna Yala"}

# How many of the ten the naive `prov - 700` -> `PAnn` join happens to get right: Bocas del
# Toro at 01 and Darien at 05. If this changes, OCHA has re-cut the pcodes.
CODE_JOIN_CORRECT = 2

N_COD_ADM1 = 13          # what COD-AB ships
N_UNITS = 12             # after the Panama / Panama Oeste dissolve
N_DRAWN = 10             # units LAPOP measures


def fold(s):
    """Accent- and case-insensitive key for a province name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_code_join(by_name):
    """Report what a `prov - 700` -> `PAnn` join would have done, and refuse to let it pass.

    Kept as an assertion rather than a comment because the failure it guards is the one that
    survives every reconciliation: a permutation of ten units preserves the national total.
    """
    right, wrong = [], []
    for code, lname in sorted(LAPOP_PROVINCES.items()):
        naive = f"PA{code - 700:02d}"
        actual = by_name[fold(ALIASES.get(lname, lname))]
        (right if naive == actual else wrong).append((code, lname, naive, actual))
    print(f"\n  witness 2 — the code join is NOT used. `prov - 700` -> PAnn would pair "
          f"{len(right)} of {len(LAPOP_PROVINCES)} correctly and MISPAIR {len(wrong)}:")
    for code, lname, naive, actual in wrong:
        other = next(n for n, p in by_name.items() if p == naive)
        print(f"      LAPOP {code} {lname:<22} -> {naive}, which COD says is {other!r} "
              f"(the real one is {actual})")
    if len(right) != CODE_JOIN_CORRECT:
        raise SystemExit(
            f"the code join now gets {len(right)} of {len(LAPOP_PROVINCES)} right, not "
            f"{CODE_JOIN_CORRECT}. OCHA has re-cut Panama's pcodes — STOP and decide "
            "deliberately whether this file should switch to them. Do not delete this check.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "pan_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "pan_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_COD_ADM1:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_COD_ADM1} — COD has re-cut "
                         "Panama")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} provinces and comarcas, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    by_name = dict(zip(g["adm1_name"].map(fold), g["pcode"]))
    if len(by_name) != N_COD_ADM1:
        raise SystemExit("COD's province names are not unique — the name join is unsafe")

    # ---- witness 1: every LAPOP name is a COD name, with one alias ----
    missing = [n for n in LAPOP_PROVINCES.values() if fold(ALIASES.get(n, n)) not in by_name]
    if missing:
        print(f"    LAPOP names with no polygon: {missing}")
        print(f"    COD names: {sorted(g['adm1_name'])}")
        raise SystemExit("the name join FAILED")
    spare = sorted(n for n in g["adm1_name"]
                   if fold(n) not in {fold(ALIASES.get(v, v))
                                      for v in LAPOP_PROVINCES.values()})
    print(f"  witness 1 — all {len(LAPOP_PROVINCES)} LAPOP names match a COD name, "
          f"{len(ALIASES)} alias needed ({', '.join(f'{k!r} -> {v!r}' for k, v in ALIASES.items())})")
    print(f"    COD units LAPOP never names: {spare}")
    expected_spare = sorted(list(NOT_SAMPLED.values()) + ["Panamá Oeste"])
    if sorted(spare) != expected_spare:
        raise SystemExit(f"the unmatched COD units are {sorted(spare)}, expected "
                         f"{expected_spare} — the geography has changed")

    check_code_join(by_name)

    # ---- populations, joined on the pcode, which is COD's own key on both sides ----
    ppath = os.path.join(RAW, "pan_admpop_adm1_2023.csv")
    pop = pd.read_csv(ppath, encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != N_COD_ADM1 or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same pcodes as COD-AB")
    g = g.merge(pop[["pcode", "T_TL"]], on="pcode", how="left")
    if g["T_TL"].isna().any():
        raise SystemExit("a province came out of the population join with no total")
    g["pop"] = g["T_TL"].astype("int64")
    print(f"\n  witness 3 — COD-PS 2023 joins on the pcode: {g['pop'].sum():,} people")

    # COD-PS's 2023 table is the census year itself, and it lands on INEC's own published
    # 4,064,780 to within 335 people. That is why this country is drawn on COD-PS where
    # Ecuador had to be drawn on its office's census (§9bn): here they are the same count.
    print(f"    against INEC's published 2023 census total of 4,064,780: "
          f"{g['pop'].sum() - 4_064_780:+,}")

    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "adm1_name"]
    hi = g.loc[g["density"].idxmax(), "adm1_name"]
    print(f"    sparsest {lo!r}, densest {hi!r}")
    if fold(hi) != fold("Panamá Oeste"):
        raise SystemExit("Panama Oeste is not the densest unit — the population join is "
                         "permuted")

    # ---- the dissolve: Panama Oeste folds into Panama ----
    g["unit"] = g["pcode"].map(lambda p: MERGE_INTO.get(p, p))
    merged = g.dissolve(by="unit", aggfunc={"pop": "sum"}).reset_index()
    if len(merged) != N_UNITS:
        raise SystemExit(f"{len(merged)} units after the dissolve, expected {N_UNITS}")
    names = {p: n for p, n in zip(g["pcode"], g["adm1_name"])}
    merged["name"] = merged["unit"].map(lambda u: MERGED_NAME if u in MERGE_INTO.values()
                                        else names[u])
    merged["geo_id"] = merged["unit"]
    merged["pcode"] = merged["unit"]
    pan = int(merged.loc[merged["unit"] == "PA12", "pop"].iloc[0])
    print(f"\n  dissolved {sorted(MERGE_INTO)} into {sorted(set(MERGE_INTO.values()))}: "
          f"{MERGED_NAME} is {pan:,} people, {pan / merged['pop'].sum():.1%} of Panama")
    if merged["pop"].sum() != g["pop"].sum():
        raise SystemExit("the dissolve lost or gained people")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = merged[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="provincias", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    prov_of = {}
    for code, lname in LAPOP_PROVINCES.items():
        pcode = by_name[fold(ALIASES.get(lname, lname))]
        prov_of[MERGE_INTO.get(pcode, pcode)] = code
    if len(prov_of) != N_DRAWN:
        raise SystemExit(f"{len(prov_of)} units carry a LAPOP code, expected {N_DRAWN}")

    lut = pd.DataFrame({
        "geo_id": sorted(merged["unit"]),
        "unit": sorted(merged["unit"]),
        "name": [merged.loc[merged["unit"] == u, "name"].iloc[0]
                 for u in sorted(merged["unit"])],
        "lapop_prov": [prov_of.get(u, "") for u in sorted(merged["unit"])],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    drawn = lut[lut["lapop_prov"] != ""]
    print(f"wrote {LOOKUP} ({len(lut)} rows, {len(drawn)} with a LAPOP prov code)")
    print(f"  not sampled and not drawn: "
          f"{', '.join(sorted(NOT_SAMPLED.values()))} — "
          f"{int(merged.loc[merged['unit'].isin(NOT_SAMPLED), 'pop'].sum()):,} people, "
          f"{merged.loc[merged['unit'].isin(NOT_SAMPLED), 'pop'].sum() / merged['pop'].sum():.2%}")


if __name__ == "__main__":
    main()
