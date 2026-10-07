"""Equatorial Guinea: the 2015 census asked religion and published no table of it, so the national
shares of the 2011 Demographic and Health Survey are drawn in every province, with Christians
split between Catholic and Protestant at the government's 2015 estimate.

Reads data/raw/gq/dhs_fr271.pdf (EDSGE-I 2011 final report, *Encuesta Demográfica y de Salud
Guinea Ecuatorial 2011*, Ministerio de Sanidad y Bienestar Social and Ministerio de Economía,
Comercio y Promoción Empresarial with ICF International, 2012; DHS FR271, open on
dhsprogram.com), data/geo/gq/gq_lookup.csv (`sources/gq_geo.py`, the 2015 census's definitive
count per province) and data/raw/estimates/pew.zip; writes data/normalized/gq.csv.
`sources/gq.md` is the record; `ask/RULINGS.md` 2026-09-15 and 2026-09-16 (draw a country with no
published religion table on the best survey or compiler figure, with the method said) the rulings.

## WHAT IS PUBLISHED

The 2015 census form asks religion (Bloque V, question 9: sin religión, católica, protestante,
Islam, otra, no sabe), but none of its three published volumes tabulates it: the *Síntesis* (3
pages) and *Resultados preliminares* (16 pages) read 2026-10-03, the *Resultados definitivos* (72
pages) read 2026-09-15. The DHS final report prints religion once, in Cuadro 3.1 (printed p.30,
PDF p.60), nationally, for women and men aged 15 to 49, weighted, beside a region block it is
never crossed with. Its card has one Christian box (questionnaires, PDF pp.365 and 437), so it
does not split Catholics from Protestants. The microdata would give the survey's four domains
(Malabo urban, the rest of Bioko, Bata urban, the rest of the mainland) and is behind a DHS
registration, which is Anita's (`ask/`).

## THE CONSTRUCTION

  * Women and men are each taken as their weighted shares with `Sin información` left out of the
    base, and combined at the census's own sex split (preliminary Tabla 3.1: 651,820 men,
    570,622 women). Men are weighted up because two thirds of foreign residents are men and
    Muslims are more common among them (5.4% of men, 1.9% of women).
  * `Cristiano` is split at the government estimate for 2015 that the US State Department's 2023
    report on religious freedom quotes: "88 percent of the population is Roman Catholic, 5 percent
    Protestant, and 2 percent Muslim", the remaining 5 percent animism, the Baha'i Faith, Judaism
    and other beliefs. So 88/93 of the survey's Christians are drawn Catholic and 5/93 Protestant.
    The estimate's own Christian total (93%) is beside the survey's, and the build stops if they
    drift more than `CHRISTIAN_GAP` apart.
  * Every province takes the same mix: nothing published places any religion below the nation.

Usage:
    python sources/gq.py            rebuild data/normalized/gq.csv and print the checks
"""

import io
import os
import re
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

REPORT = os.path.join(ROOT, "data", "raw", "gq", "dhs_fr271.pdf")
REPORT_URL = "https://dhsprogram.com/pubs/pdf/fr271/fr271.pdf"
TABLE_PAGE = 59                 # 0-based; printed p.30, Cuadro 3.1
LOOKUP = os.path.join(ROOT, "data", "geo", "gq", "gq_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "gq.csv")

# Cuadro 3.1's religion block: (weighted %, weighted n, unweighted n) for women, then men, 15-49.
# Transcribed 2026-10-03 and asserted against the text layer on every run.
RELIGION = {
    "Cristiano": ((96.3, 3442, 3449), (91.9, 1432, 1469)),
    "Musulmana": ((1.9, 68, 68), (5.4, 84, 103)),
    "Animista": ((1.0, 34, 24), (0.6, 10, 5)),
    "Sin religión": ((0.0, 2, 1), (0.4, 6, 8)),
    "Otro": ((0.2, 7, 9), (0.4, 6, 8)),
    "Sin información": ((0.6, 22, 24), (1.3, 20, 19)),
}
DROPPED = "Sin información"
TOTALS = ((3575, 3575), (1557, 1612))      # weighted, unweighted: women, men 15-49
MEN_2015, WOMEN_2015 = 651_820, 570_622    # census 2015, preliminary Tabla 3.1
# US State Department, 2023 Report on International Religious Freedom: Equatorial Guinea, Section I
# ("According to a government estimate from 2015 ..."), read 2026-10-03.
GOV_2015 = {"Catholic": 88, "Protestant": 5, "Muslim": 2, "other": 5}
CHRISTIAN_GAP = 0.04
NATIONAL_2015 = 1_225_377
YEAR = 2011
SOURCE_ID = "gq_dhs2011_national_mix"
# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(christian=94.87, catholic=89.77, protestant=5.10, muslim=3.81, animist=0.79,
            none=0.23, other=0.30, women_muslim=1.9, men_muslim=5.4, pew_christian=88.7,
            pew_muslim=4.0, pew_unaff=5.0)


def read_table():
    """Cuadro 3.1's religion block from the report's text layer, checked against RELIGION."""
    import fitz

    if not os.path.exists(REPORT):
        raise SystemExit(f"missing {REPORT}; fetch {REPORT_URL} (curl with a browser user agent; "
                         "WebFetch gets 403)")
    page = fitz.open(REPORT)[TABLE_PAGE].get_text()
    if "Cuadro 3.1" not in page or "Religión" not in page:
        raise SystemExit("Cuadro 3.1's religion block is not on the pinned page")
    lines = [x.strip() for x in page.split("\n") if x.strip()]
    i = lines.index("Religión")
    num = lambda s: float(s.replace(".", "").replace(",", "."))      # noqa: E731
    got = {}
    for label in RELIGION:
        j = lines.index(label, i)
        v = [num(x) for x in lines[j + 1:j + 7]]
        got[label] = ((v[0], int(v[1]), int(v[2])), (v[3], int(v[4]), int(v[5])))
    if got != RELIGION:
        raise SystemExit(f"Cuadro 3.1 reads {got}, pinned {RELIGION}")
    w_sum = sum(v[0][1] for v in RELIGION.values())
    m_sum = sum(v[1][1] for v in RELIGION.values())
    if w_sum != TOTALS[0][0] or abs(m_sum - TOTALS[1][0]) > 1:
        raise SystemExit(f"weighted rows sum to {w_sum} and {m_sum}, the table's totals "
                         f"{TOTALS[0][0]} and {TOTALS[1][0]}")
    print(f"  DHS EDSGE-I 2011, Cuadro 3.1: religion of {TOTALS[0][1]:,} women and {TOTALS[1][1]:,} "
          f"men 15-49 read back from the report; men's weighted rows sum to {m_sum} against "
          f"{TOTALS[1][0]} (rounding)")
    return RELIGION


def pew_row():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)))
    r = t[(t["Country"] == "Equatorial Guinea") & (t["Year"] == 2020)].iloc[0]
    print("  Pew Research Center 2020, everyone living in Equatorial Guinea: " + ", ".join(
        f"{c} {float(r[c]):.2f}%" for c in ("Christians", "Muslims", "Religiously_unaffiliated",
                                            "Other_religions", "Jews", "Hindus", "Buddhists")))
    return (round(float(r["Christians"]), 1), round(float(r["Muslims"]), 1),
            round(float(r["Religiously_unaffiliated"]), 1))


def national_mix(tab):
    """{source category: share} summing to 1."""
    sexes = {}
    for k, sex in ((0, "women"), (1, "men")):
        n = {c: v[k][1] for c, v in tab.items() if c != DROPPED}
        tot = sum(n.values())
        sexes[sex] = {c: x / tot for c, x in n.items()}
    fm = MEN_2015 / (MEN_2015 + WOMEN_2015)
    mix = {c: fm * sexes["men"][c] + (1 - fm) * sexes["women"][c] for c in sexes["men"]}
    print(f"  shares without `{DROPPED}`, combined at the census's {fm:.1%} men:")
    for c in mix:
        print(f"      {c:<14} women {100 * sexes['women'][c]:6.2f}%   men {100 * sexes['men'][c]:6.2f}%"
              f"   drawn {100 * mix[c]:6.2f}%")
    gov_christ = (GOV_2015["Catholic"] + GOV_2015["Protestant"]) / 100
    if abs(mix["Cristiano"] - gov_christ) > CHRISTIAN_GAP:
        raise SystemExit(f"the survey's Christian share {mix['Cristiano']:.3f} is more than "
                         f"{CHRISTIAN_GAP} from the 2015 estimate's {gov_christ}")
    christ = mix.pop("Cristiano")
    split = GOV_2015["Catholic"] + GOV_2015["Protestant"]
    out = {"Cristiano, católico": christ * GOV_2015["Catholic"] / split,
           "Cristiano, protestante": christ * GOV_2015["Protestant"] / split}
    out.update(mix)
    print(f"  Christians {100 * christ:.2f}% (the 2015 estimate: {100 * gov_christ:.0f}%), split "
          f"{GOV_2015['Catholic']}:{GOV_2015['Protestant']}: Catholic {100 * out['Cristiano, católico']:.2f}%, "
          f"Protestant {100 * out['Cristiano, protestante']:.2f}%; Muslim {100 * out['Musulmana']:.2f}% "
          f"against the estimate's {GOV_2015['Muslim']}%")
    return out, christ


def main():
    tab = read_table()
    mix, christ = national_mix(tab)
    pew = pew_row()

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 7 or int(lut["pop"].sum()) != NATIONAL_2015:
        raise SystemExit(f"{LOOKUP}: {len(lut)} provinces, {int(lut['pop'].sum()):,}; re-run gq_geo.py")
    pop = lut.set_index("geo_id")["pop"]
    m = pd.DataFrame({c: pop * s for c, s in mix.items()})
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a province's rounded counts do not sum to its census count")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("DHS 2011 national shares (women and men 15-49, combined at the census sex split), "
                   "Christians split 88:5 at the 2015 government estimate, the same mix in every "
                   "province, on the 2015 census count")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people, 7 provinces)")
    for c, n in out.groupby("source_category")["count"].sum().sort_values(ascending=False).items():
        print(f"      {c:<24} {n:>10,}")

    r2 = lambda x: round(100 * float(x), 2)       # noqa: E731
    got = dict(christian=r2(christ), catholic=r2(mix["Cristiano, católico"]),
               protestant=r2(mix["Cristiano, protestante"]), muslim=r2(mix["Musulmana"]),
               animist=r2(mix["Animista"]), none=r2(mix["Sin religión"]), other=r2(mix["Otro"]),
               women_muslim=tab["Musulmana"][0][0], men_muslim=tab["Musulmana"][1][0],
               pew_christian=pew[0], pew_muslim=pew[1], pew_unaff=pew[2])
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
