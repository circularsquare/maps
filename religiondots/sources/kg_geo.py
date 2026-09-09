"""Kyrgyzstan — boundaries and populations for the seven oblasts and two republican cities.

Writes data/geo/kg/kg_oblasts.gpkg and data/geo/kg/kg_lookup.csv.

OCHA COD-AB Kyrgyzstan (`cod-ab-kgz`), the **shapefile** bundle rather than the geodatabase on
§12's Chile rule. Nine ADM1 features: Batken, Chui, Issyk-Kul, Jalal-Abad, Naryn, Osh, Talas,
and the cities of Bishkek and Osh, which are republican-subordinate and are ADM1 in COD, in
the national statistics committee's own tables, and in the LiTS sample frame alike.

## THE POPULATION IS THE OFFICE'S OWN AND NOT COD-PS, and the margin is 21%

COD-PS for Kyrgyzstan is `kyrgyzstan_population_statistics_2018_adm1_v2.csv`, dated **2018**
and totalling 6,140,200. The National Statistical Committee's own resident-population file
puts the country at **7,404,329 at 1 January 2026**. That is a 20.6% gap, and it is not a
projection disagreement: it is eight years of one of the fastest-growing populations in the
former Soviet Union. Ecuador (§9bn) took the office's own over a COD error of 3.4%; this one
is six times larger and the same call is not close.

    stat.gov.kg -> /ru/statistics/naselenie/ -> download/operational/825/
    "Численность постоянного населения областей, районов, городов, айылных аймаков и
     айылов (сел) Кыргызской Республики" — an .xls of eight sheets, one per oblast plus a
     national sheet carrying Bishkek and Osh city, down to the individual village.

COD-PS is still read, for one thing only: its 5-year age bands by sex are what `sources/kg.py`
uses to measure how the adult-only survey leans against a population a third of which is
under 18. That is a ratio within each oblast, so its 2018 vintage does not matter there.

## THREE WITNESSES ON TWO DIFFERENT JOINS, AND THE THIRD IS THE ONE THE SURVEY NEEDS

The office keys its rows on the 14-digit SOATE territory code and COD keys its polygons on a
pcode, and the two agree by construction: SOATE `417NN000000000` is COD `KGNN000000000`, for
all nine. That is witness one, and it is arithmetic rather than a name match.

Witness two is the Russian name, which COD carries in `adm1_name1` and the office in its own
column, and they agree exactly for all nine once `г.Бишкек`/`г.Ош` are folded.

**Both of those pin the OFFICE to COD, and the survey is joined by a third thing entirely.**
`LITS_REGION` below is a hand-written decode of LiTS III's abbreviated `region_name` strings,
and no arithmetic reaches it. The population correlation in `lits.held_out` is not a witness
on it either: `sources/kg.md` §9.5 swapped `И-КУЛЬСКАЯ` and `БАТКЕНСКАЯ` and the wrong join
scored **better** than the right one. So witness three is `lits_decode_witness`, which reads
the abbreviations the way the person who wrote the dict read them, and requires each label to
abbreviate exactly one of the nine Russian names.

**The trap this guards is the one COD-PS would have walked into.** COD-PS ships the same
pcodes in a CSV column typed as a float, so `41702000000000` reaches pandas as `4.1702E+13`
and every pcode in the file is a *different* rounded float. Reading that column as a key
silently pairs nothing, and reading it as a string pairs nothing either.

## AND ONE PARSER TRAP IN THE OFFICE'S OWN FILE

Talas oblast's territory code is written `41707 000 000 00 0`, with spaces, on a sheet where
the other eight are written without. Digits are extracted rather than the string compared.

Usage:
    python sources/kg_geo.py --fetch    one ~4.9 MB zip from HDX, one ~0.5 MB xls from
                                        stat.gov.kg, one 7 KB CSV from HDX
    python sources/kg_geo.py            rebuild from data/raw/kg/
"""

import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kg")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "kg")
OUT = os.path.join(OUT_DIR, "kg_oblasts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "kg_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "kgz_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/693aca77-1252-4404-9045-785e43bb6846/resource/"
        "10fa9f94-756c-4cb7-b61d-d7e5b434069b/download/kgz_admin_boundaries.shp.zip",
    # COD-PS 2018, read ONLY for its age bands — see the module docstring.
    "kgz_codps_adm1_2018.csv":
        "https://data.humdata.org/dataset/7c38a699-d5e5-4972-8090-c676d299d4ea/resource/"
        "f1c48abd-e89d-4aac-944f-0d32ed11a23e/download/"
        "kyrgyzstan_population_statistics_2018_adm1_v2.csv",
    # The office's own resident population, 1 January 2026.
    "kg_nsc_population_2026.xls":
        "https://stat.gov.kg/ru/statistics/download/operational/825/",
}

N_UNITS = 9
NSC_TOTAL = 7_404_329          # Кыргызская Республика, SOATE 41700000000000, 1 Jan 2026
NSC_DATE = "1 January 2026"

# The LiTS III `region_name` strings, verbatim, against the COD pcode. LiTS abbreviates two of
# the oblast names (`Д-АБАДСКАЯ`, `И-КУЛЬСКАЯ`) and prefixes the two cities with `горкенеш`,
# the Kyrgyz word for the city council, so this cannot be a string match and is written out.
LITS_REGION = {
    "БАТКЕНСКАЯ":        "KG05000000000",
    "И-КУЛЬСКАЯ":        "KG02000000000",
    "Д-АБАДСКАЯ":        "KG03000000000",
    "НАРЫНСКАЯ":         "KG04000000000",
    "ОШСКАЯ":            "KG06000000000",
    "ТАЛАССКАЯ":         "KG07000000000",
    "ЧУЙСКАЯ":           "KG08000000000",
    "горкенеш г.БИШКЕК": "KG11000000000",
    "горкенеш г.ОШ":     "KG21000000000",
}

# COD's own Russian name for each, for the second witness. Folded case-insensitively and with
# spaces stripped, because COD writes `г.Бишкек` and the office writes `г.Бишкек` on one sheet
# and `г. Бишкек` on another.
EN_NAME = {
    "KG02000000000": "Issyk-Kul", "KG03000000000": "Jalal-Abad", "KG04000000000": "Naryn",
    "KG05000000000": "Batken",    "KG06000000000": "Osh",        "KG07000000000": "Talas",
    "KG08000000000": "Chui",      "KG11000000000": "Bishkek",    "KG21000000000": "Osh city",
}

# COD-PS's own `admin1Name_en` spellings, which are NOT COD-AB's: it writes `Bishkek (city)`
# and `Osh (city)` where COD-AB writes the same, but the oblast is plain `Osh` in both, so a
# substring or a `.split("(")` rule pairs Osh oblast with Osh city. Written out instead.
CODPS_NAME = {
    "Issyk-Kul": "KG02000000000", "Jalal-Abad": "KG03000000000", "Naryn": "KG04000000000",
    "Batken": "KG05000000000",    "Osh": "KG06000000000",        "Talas": "KG07000000000",
    "Chui": "KG08000000000",      "Bishkek (city)": "KG11000000000",
    "Osh (city)": "KG21000000000",
}


def fold(s):
    return re.sub(r"[\s.]+", "", str(s)).lower()


# Words that are not part of a name and are dropped before the abbreviation is compared token
# by token. `горкенеш` is the Kyrgyz for city council and LiTS prefixes both cities with it;
# `область` is on the office's own spelling of an oblast and not on COD's, and this witness
# runs against COD's, so it is only here so that a re-issued COD file cannot fail it spuriously.
DECODE_DROP = {"горкенеш", "область", "обл"}


def _tokens(s):
    """`И-КУЛЬСКАЯ` -> ('и', 'кульская'); `горкенеш г.БИШКЕК` -> ('г', 'бишкек')."""
    return tuple(p for p in re.split(r"[-.\s]+", str(s).lower())
                 if p and p not in DECODE_DROP)


def _abbreviates(lits, name_ru):
    """Does this LiTS label abbreviate this Russian name, token for token?

    The same number of tokens, and every LiTS token either equal to or an initial abbreviation
    of the name's token in the same position: `и` for `иссык`, `кульская` for `кульская`,
    `баткенская` for `баткенская`. `и-кульская` against `баткенская` fails on the token count
    before it gets as far as the letters.
    """
    a, b = _tokens(lits), _tokens(name_ru)
    return len(a) == len(b) and all(y.startswith(x) for x, y in zip(a, b))


def lits_decode_witness(name_ru):
    """WITNESS 3: EACH LiTS LABEL ABBREVIATES EXACTLY ONE RUSSIAN NAME, AND IT IS THIS ONE.

    ## Witnesses 1 and 2 pin the office to COD. Neither of them touches this dict.

    `LITS_REGION` is hand-written, and the held-out population check in `lits.held_out` is the
    only thing that runs on it — which is exactly the check that has no power over Issyk-Kul
    and Batken. The second read (`sources/kg.md` §9.5) demonstrated the gap rather than arguing
    it: swap those two labels in `LITS_REGION` and the WRONG join scores r=+0.9870 with none of
    the 362,879 other orderings reaching it, against the true join's +0.9866 with one. The
    wrong join passes more cleanly than the right one. The old witness 3, a set comparison of
    `LITS_REGION.values()` against the pcodes, is a coverage test that every permutation of the
    dict passes identically.

    So the decode is closed here on the thing a human was reading when they wrote the dict
    down: `И-КУЛЬСКАЯ` cannot abbreviate `Баткенская`. Strip the noise words, split both the
    LiTS label and COD's own Russian name on `-`, `.` and space, and require each LiTS token to
    equal or to begin the name's token in the same position. Then require the match to be
    **unique** across all nine names, so this asserts not merely that the written pairing is
    consistent but that no other pairing of the nine is. `[[reference_name_join_wrong_neighbour]]`.

    No correlation and no population figure is involved, which is the point: the two units the
    held-out check cannot separate are separated here by their names.
    """
    bad = []
    for lits, pcode in LITS_REGION.items():
        hits = sorted(p for p in name_ru if _abbreviates(lits, name_ru[p]))
        if hits != [pcode]:
            bad.append((lits, pcode, hits))
    if bad:
        for lits, pcode, hits in bad:
            got = ", ".join(f"{p} {name_ru[p]}" for p in hits) or "nothing"
            print(f"      {lits!r} is written against {pcode} "
                  f"({name_ru.get(pcode, '?')}) but abbreviates {got}")
        raise SystemExit(
            f"{len(bad)} of {len(LITS_REGION)} LiTS III region labels do not abbreviate the "
            "Russian name of the oblast they are written against, or abbreviate more than one "
            "of them. LITS_REGION is hand-written and the population check cannot separate "
            "Issyk-Kul from Batken, so this is the assertion that pins the decode: read the "
            "names above before changing either side.")
    print(f"  witness 3 — each of the {N_UNITS} LiTS III `region_name` labels abbreviates "
          "exactly one\n              of COD's nine Russian oblast names, token for token, and "
          "it is the one it is\n              written against; no other pairing of the nine "
          "satisfies it")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def nsc_population():
    """Oblast totals out of the office's own .xls, keyed on the SOATE code."""
    path = os.path.join(RAW, "kg_nsc_population_2026.xls")
    xl = pd.ExcelFile(path)
    rows = {}
    national = None
    for sheet in xl.sheet_names:
        d = xl.parse(sheet, header=None)
        for _, r in d.iterrows():
            code = re.sub(r"\D", "", str(r.iloc[0]))       # Talas is `41707 000 000 00 0`
            if not re.fullmatch(r"417\d{11}", code):
                continue
            if code == "41700000000000":
                national = int(r.iloc[2])
                continue
            if not code.endswith("0" * 9):
                continue
            pcode = "KG" + code[3:5] + "0" * 9
            n = int(r.iloc[2])
            if pcode in rows and rows[pcode] != n:
                raise SystemExit(f"{pcode} appears twice with {rows[pcode]} and {n}")
            rows[pcode] = n
    if national != NSC_TOTAL:
        raise SystemExit(f"the office's national total is now {national:,}, not "
                         f"{NSC_TOTAL:,}. The file has been reissued — update NSC_TOTAL and "
                         "NSC_DATE deliberately, and re-read the header for the new date.")
    if len(rows) != N_UNITS:
        raise SystemExit(f"{len(rows)} oblast rows, expected {N_UNITS}: {sorted(rows)}")
    if sum(rows.values()) != national:
        raise SystemExit(f"the nine oblasts sum to {sum(rows.values()):,} against the "
                         f"office's own national {national:,}")
    print(f"  the office's own resident population, {NSC_DATE}: {national:,} across "
          f"{len(rows)} units, and the nine sum to it exactly")
    return pd.Series(rows, name="pop")


def codps_ages():
    """COD-PS 2018's 5-year bands by sex, per pcode — used ONLY for the §3.5 age lean.

    Its pcode column is typed as a float in the delivered CSV, so it arrives as `4.1702E+13`
    and is unusable as a key; the ADM1 English name is the key instead, and the row count and
    the name set are both asserted.
    """
    path = os.path.join(RAW, "kgz_codps_adm1_2018.csv")
    # cp1251, not UTF-8: the Russian and Kyrgyz name columns are Windows-Cyrillic and a
    # utf-8 read raises on the first oblast name. Only the English column is used.
    d = pd.read_csv(path, encoding="cp1251", skiprows=[1], thousands=",")
    d = d[d["admin1Name_en"].notna()]
    if len(d) != N_UNITS:
        raise SystemExit(f"COD-PS has {len(d)} ADM1 rows, expected {N_UNITS}")
    d["pcode"] = d["admin1Name_en"].astype(str).str.strip().map(CODPS_NAME)
    if d["pcode"].isna().any() or d["pcode"].nunique() != N_UNITS:
        raise SystemExit(f"COD-PS ADM1 names did not resolve: "
                         f"{d[['admin1Name_en', 'pcode']].to_dict('records')}")
    male = [c for c in d.columns if c.startswith("Number_of_male")]
    female = [c for c in d.columns if c.startswith("Number_of_female")]
    if len(male) != 21 or len(female) != 21:
        raise SystemExit(f"COD-PS has {len(male)}/{len(female)} age columns, expected 21 each")
    out = d.set_index("pcode")[male + female].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        raise SystemExit("a COD-PS age cell did not parse as a number")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "kgz_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "kgz_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_UNITS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_UNITS} — COD has re-cut "
                         "Kyrgyzstan")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    print(f"read {shp}: {len(g)} oblasts and republican cities, {g.crs}")

    if set(g["pcode"]) != set(EN_NAME):
        raise SystemExit(f"COD's pcodes have changed: {sorted(set(g['pcode']) ^ set(EN_NAME))}")

    # ---- witness 1: the office's SOATE code IS COD's pcode, 417NN -> KGNN ----
    pop = nsc_population()
    missing = sorted(set(g["pcode"]) - set(pop.index))
    if missing:
        raise SystemExit(f"no office population for {missing}")
    print(f"  witness 1 — the office's SOATE `417NN` and COD's `KGNN` pair all "
          f"{N_UNITS} units by arithmetic, with no name match involved")

    # ---- witness 2: the Russian names agree, independently of the codes ----
    xl = pd.ExcelFile(os.path.join(RAW, "kg_nsc_population_2026.xls"))
    office_name = {}
    for sheet in xl.sheet_names:
        d = xl.parse(sheet, header=None)
        for _, r in d.iterrows():
            code = re.sub(r"\D", "", str(r.iloc[0]))
            if re.fullmatch(r"417\d\d0{9}", code) and code != "41700000000000":
                office_name["KG" + code[3:5] + "0" * 9] = str(r.iloc[1]).strip()
    cod_ru = dict(zip(g["pcode"], g["adm1_name1"].astype(str)))
    bad = []
    for pcode, nm in office_name.items():
        # `Баткенская область` against COD's `Баткенская`; `г.Бишкек` against `г.Бишкек`
        a = fold(nm).replace("область", "")
        b = fold(cod_ru[pcode])
        if a != b:
            bad.append((pcode, nm, cod_ru[pcode]))
    if bad:
        for row in bad:
            print(f"      {row}")
        raise SystemExit("the office's Russian oblast names do not match COD's")
    print(f"  witness 2 — all {N_UNITS} Russian names agree between the office and COD, "
          "which is independent of the code join")

    # ---- witness 3: each LiTS label abbreviates the Russian name it is written against ----
    # Coverage first, which is only the old form of this witness: it says nothing about WHICH
    # pcode a label goes to, because every permutation of the dict passes it identically.
    if set(LITS_REGION.values()) != set(g["pcode"]):
        raise SystemExit("LITS_REGION does not cover the same nine pcodes as COD")
    lits_decode_witness(cod_ru)

    g = g.merge(pop.rename("pop").reset_index().rename(columns={"index": "pcode"}),
                on="pcode", how="left")
    g["pop"] = g["pop"].astype("int64")

    # The two republican cities are the two densest units by three orders of magnitude, and
    # Naryn -- 45,000 km2 of high pasture -- is the sparsest. A permuted population join is
    # what this catches. Which of the two cities is denser is NOT asserted: COD draws Osh city
    # tight around the built-up area and Bishkek with its ring of suburbs, so Osh comes out
    # ahead, and that is a fact about the polygons rather than about the join.
    g["density"] = g["pop"] / g["area_sqkm"]
    order = g.sort_values("density", ascending=False)["pcode"].map(EN_NAME).tolist()
    print(f"    densest to sparsest: {', '.join(order)}")
    if set(order[:2]) != {"Bishkek", "Osh city"} or order[-1] != "Naryn":
        raise SystemExit(f"density order is {order}, expected the two cities on top and "
                         "Naryn at the bottom; the population join is permuted")

    ages = codps_ages()
    print(f"  COD-PS 2018 age bands read for the §3.5 lean: {ages.shape[1]} columns, "
          f"{int(ages.to_numpy().sum()):,} people (used as a RATIO only)")

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(EN_NAME)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="oblasts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    by_pcode = {v: k for k, v in LITS_REGION.items()}
    lut = pd.DataFrame({
        "geo_id": sorted(g["pcode"]),
        "unit": sorted(g["pcode"]),
        "name": [EN_NAME[p] for p in sorted(g["pcode"])],
        "name_ru": [cod_ru[p] for p in sorted(g["pcode"])],
        "lits_region": [by_pcode[p] for p in sorted(g["pcode"])],
        "pop": [int(pop[p]) for p in sorted(g["pcode"])],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, with the LiTS region string alongside)")
    print(lut[["geo_id", "name", "lits_region", "pop"]].to_string(index=False))

    ages.to_csv(os.path.join(OUT_DIR, "kg_codps_ages.csv"), encoding="utf-8")
    print(f"wrote {os.path.join(OUT_DIR, 'kg_codps_ages.csv')}")


if __name__ == "__main__":
    main()
