"""Eritrea: no census has ever been held, so the national shares of the 2010 Eritrea Population
and Health Survey are drawn in every zoba.

Reads data/raw/er/ephs2010_final_report_v4.pdf (*Eritrea Population and Health Survey 2010*,
National Statistics Office and Fafo Institute for Applied International Studies, 2013; open on the
WHO Africa site), data/raw/er/dhs_fr137.pdf (*Eritrea Demographic and Health Survey 2002*, DHS
FR137, a witness), data/geo/er/er_lookup.csv (`sources/er_geo.py`) and data/raw/estimates/pew.zip;
writes data/normalized/er.csv. `sources/er.md` is the record; `ask/RULINGS.md` 2026-09-15 and
2026-09-16 (draw a country with no published religion table on the best survey or compiler figure,
with the method said) the rulings.

## WHAT IS PUBLISHED

Religion is asked in all three national surveys (EDHS 1995, EDHS 2002, EPHS 2010), each time of
women 15-49 and, in 2002 and 2010, of men, and each report prints it once, nationally, in its
Table 3.1/3-1 beside a zoba block it is never crossed with (every page with `religi` read in both
reports, 2026-10-03). The DHS Program lists no Eritrean microdata at all.

## THE CONSTRUCTION

  * EPHS 2010 Table 3-1 (printed p.42, PDF p.70): `Women ALL` (30,224 women 15-49, the core and the
    maternal mortality questionnaires together) and `Men` (4,299 men 15-49), weighted, five answers.
  * Combined at the sex split of the survey's own de facto household population aged 15-49
    (Table 2-1: 36.7% men), since that is the population the two samples stand for. Men are scarce
    in it (national service and emigration); at an even split Muslims would be 0.6 points lower.
  * Every zoba takes the same mix: nothing published places any religion below the nation.

## THE WITNESSES, NOT DRAWN

  * EDHS 2002 Table 3.1, women 15-49 (8,754): Muslim 36.5%, within `DHS2002_GAP` of 2010's women.
  * Pew Research Center 2020: Muslim 51.7%, Christian 46.7%. Pew's own appendix of sources (2025)
    gives the World Religion Database as its only source for Eritrea in both 2010 and 2020, which
    ascribes religion by ethnic group; it is not a measurement and is not drawn.

Usage:
    python sources/er.py            rebuild data/normalized/er.csv and print the checks
"""

import io
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows

EPHS = os.path.join(ROOT, "data", "raw", "er", "ephs2010_final_report_v4.pdf")
EPHS_URL = "https://www.afro.who.int/sites/default/files/2017-05/ephs2010_final_report_v4.pdf"
EPHS_TABLE_PAGE = 69            # 0-based; Table 3-1
EPHS_POP_PAGE = 39              # 0-based; Table 2-1
DHS = os.path.join(ROOT, "data", "raw", "er", "dhs_fr137.pdf")
DHS_URL = "https://dhsprogram.com/pubs/pdf/fr137/fr137.pdf"
DHS_TABLE_PAGE = 56             # 0-based; Table 3.1
LOOKUP = os.path.join(ROOT, "data", "geo", "er", "er_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "er.csv")

# EPHS 2010 Table 3-1, religion block: (weighted %, weighted n, unweighted n) for Women CORE,
# Women ALL and Men 15-49. Transcribed 2026-10-03 and asserted against the text layer every run.
RELIGION = {
    "Orthodox": ((55.4, 5671, 5147), (55.8, 16851, 15768), (60.0, 2581, 2336)),
    "Catholic": ((4.3, 445, 423), (4.2, 1264, 1254), (4.7, 200, 193)),
    "Protestant": ((0.8, 85, 83), (0.8, 238, 243), (1.0, 43, 42)),
    "Muslim": ((39.1, 4005, 4556), (39.0, 11795, 12886), (33.9, 1458, 1662)),
    "Traditional believer": ((0.3, 29, 27), (0.2, 69, 65), (0.2, 7, 7)),
}
TOTALS = (10_238, 30_224, 4_299)            # weighted: Women CORE, Women ALL, Men 15-49
# Table 2-1, de facto household population: number by sex and the 15-49 bands' percentages.
HH_MEN, HH_WOMEN = 60_320, 73_065
BANDS_15_49 = {"male": (11.2, 4.5, 3.2, 2.8, 3.1, 2.8, 2.9),
               "female": (9.9, 7.2, 7.1, 5.3, 6.0, 3.9, 4.0)}
# EDHS 2002 Table 3.1, all women: weighted percent.
DHS2002 = {"Orthodox": 57.7, "Catholic": 4.6, "Protestant": 0.7, "Muslim": 36.5,
           "Traditional believer": 0.4, "Other": 0.1, "Missing": 0.1}
DHS2002_GAP = 3.5               # points, Muslim share of women, 2002 against 2010
YEAR = 2010
SOURCE_ID = "er_ephs2010_national_mix"
# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(orthodox=57.39, catholic=4.36, protestant=0.87, muslim=37.18, traditional=0.20,
            men_share=36.72, women_muslim=39.0, men_muslim=33.9, pew_christian=46.7,
            pew_muslim=51.7)


def _lines(path, page):
    import fitz

    return [x.strip() for x in fitz.open(path)[page].get_text().split("\n") if x.strip()]


def read_table():
    if not os.path.exists(EPHS):
        raise SystemExit(f"missing {EPHS}; run sources/er_geo.py --fetch ({EPHS_URL})")
    lines = _lines(EPHS, EPHS_TABLE_PAGE)
    if "Table 3-1" not in " ".join(lines) or "Religion" not in lines:
        raise SystemExit("EPHS 2010 Table 3-1's religion block is not on the pinned page")
    i = lines.index("Religion")
    num = lambda s: float(s.replace(",", ""))      # noqa: E731
    got = {}
    for label in RELIGION:
        j = lines.index(label, i)
        v = [num(x) for x in lines[j + 1:j + 10]]
        got[label] = tuple((v[k], int(v[k + 1]), int(v[k + 2])) for k in (0, 3, 6))
    if got != RELIGION:
        raise SystemExit(f"Table 3-1 reads {got}, pinned {RELIGION}")
    for k in range(3):
        s = sum(v[k][1] for v in RELIGION.values())
        if abs(s - TOTALS[k]) > 10:      # CORE is 3 short and ALL 7: the table prints no missing row
            raise SystemExit(f"Table 3-1 column {k}: weighted rows sum to {s}, total {TOTALS[k]}")
    print(f"  EPHS 2010 Table 3-1 read back: religion of {TOTALS[1]:,} women and {TOTALS[2]:,} men "
          "15-49, weighted")

    flat = " ".join(_lines(EPHS, EPHS_POP_PAGE))
    if "Table 2-1" not in flat or f"{HH_MEN:,} | {HH_WOMEN:,}".replace(" | ", " ") not in flat:
        raise SystemExit("EPHS 2010 Table 2-1's household numbers are not on the pinned page")
    return RELIGION


def dhs_witness(mix_women):
    if not os.path.exists(DHS):
        print(f"  (EDHS 2002 witness skipped: {DHS} missing; {DHS_URL})")
        return
    flat = " ".join(_lines(DHS, DHS_TABLE_PAGE))
    for label, pct in DHS2002.items():
        lab = "Traditional believer" if label == "Traditional believer" else label
        if f"{lab} {pct}" not in flat:
            raise SystemExit(f"EDHS 2002 Table 3.1: `{lab} {pct}` not on the pinned page")
    gap = abs(DHS2002["Muslim"] - mix_women)
    print(f"  EDHS 2002 Table 3.1, women: Muslim {DHS2002['Muslim']}% against 2010's {mix_women}% "
          f"(gap {gap:.1f}, bar {DHS2002_GAP})")
    if gap > DHS2002_GAP:
        raise SystemExit("the 2002 and 2010 surveys disagree beyond the bar")


def pew_row():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)))
    r = t[(t["Country"] == "Eritrea") & (t["Year"] == 2020)].iloc[0]
    print(f"  Pew 2020 (World Religion Database): Christian {float(r['Christians']):.2f}%, Muslim "
          f"{float(r['Muslims']):.2f}%, unaffiliated {float(r['Religiously_unaffiliated']):.2f}%, "
          f"other {float(r['Other_religions']):.2f}%")
    return round(float(r["Christians"]), 1), round(float(r["Muslims"]), 1)


def national_mix(tab):
    men = HH_MEN * sum(BANDS_15_49["male"]) / 100
    women = HH_WOMEN * sum(BANDS_15_49["female"]) / 100
    fm = men / (men + women)
    sexes = {}
    for k, sex in ((1, "women"), (2, "men")):
        n = {c: v[k][1] for c, v in tab.items()}
        tot = sum(n.values())
        sexes[sex] = {c: x / tot for c, x in n.items()}
    mix = {c: fm * sexes["men"][c] + (1 - fm) * sexes["women"][c] for c in tab}
    even = 0.5 * sexes["men"]["Muslim"] + 0.5 * sexes["women"]["Muslim"]
    print(f"  household population 15-49 (Table 2-1): {men:,.0f} men, {women:,.0f} women, {fm:.2%} men")
    for c in mix:
        print(f"      {c:<22} women {100 * sexes['women'][c]:6.2f}%   men {100 * sexes['men'][c]:6.2f}%"
              f"   drawn {100 * mix[c]:6.2f}%")
    print(f"  at an even sex split Muslims would be {100 * even:.2f}%")
    return mix, fm


def main():
    tab = read_table()
    mix, fm = national_mix(tab)
    dhs_witness(tab["Muslim"][1][0])
    pew = pew_row()

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 6:
        raise SystemExit(f"{LOOKUP}: {len(lut)} zobas; re-run sources/er_geo.py")
    pop = lut.set_index("geo_id")["pop"]
    m = pd.DataFrame({c: pop * s for c, s in mix.items()})
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a zoba's rounded counts do not sum to its population")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "zoba"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("EPHS 2010 national shares (women and men 15-49, combined at the survey's household "
                   "sex split), the same mix in every zoba, on UN WPP 2024's 2020 total split by the "
                   "survey's own zoba shares")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people, 6 zobas)")
    for c, n in out.groupby("source_category")["count"].sum().sort_values(ascending=False).items():
        print(f"      {c:<24} {n:>10,}")

    r2 = lambda x: round(100 * float(x), 2)       # noqa: E731
    got = dict(orthodox=r2(mix["Orthodox"]), catholic=r2(mix["Catholic"]),
               protestant=r2(mix["Protestant"]), muslim=r2(mix["Muslim"]),
               traditional=r2(mix["Traditional believer"]), men_share=r2(fm),
               women_muslim=tab["Muslim"][1][0], men_muslim=tab["Muslim"][2][0],
               pew_christian=pew[0], pew_muslim=pew[1])
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
