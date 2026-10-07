"""Comoros: no census or survey publishes religion below the nation, so the national shares of
Afrobarometer's round 10 (2025) are drawn on each island's 2017 census population.

Reads, from data/raw/km/ (`--fetch` downloads them):

  * Afrobarometer, *Comores Round 10 : Résumé des résultats* (fieldwork 13 May to 5 June 2025, 1,200
    Comorian citizens 18 and over; published 30 September 2025, revised 1 November 2025): the
    weighted religion line of the sample description (p.5) and Q97 in full (p.7);
  * Afrobarometer, *AB_R10.Comoros_Codebook_25Aug25* (unweighted Q97 counts, a witness);
  * INSEED, *Cartographie de vulnérabilité au COVID-19 en Union des Comores* (9 July 2020), annex
    Tableau 1, *Population par préfecture et commune*, from the 2017 census (RGPH 2017): the three
    islands, 18 préfectures and 54 communes, by sex;
  * DHS Program FR278, *Enquête Démographique et de Santé et à Indicateurs Multiples (EDSC-MICS II)
    2012*, Tableau 3.1 (religion of women and men 15-49), a witness;
  * data/raw/estimates/pew.zip (Pew Research Center 2020), a witness.

Writes data/normalized/km.csv. `sources/km.md` is the record.

## WHY THIS ROUTE

The 2017 census asked no religion (INSEED's NADA, `DDI-COM-RPGH-2017`, 351 variables, read
2026-10-03; nationality is B10). Neither did the 2003 census (556 variables, §11aq). MICS 2022
asks only whether a woman felt discriminated against for her religion. The 2012 DHS asked and
prints it nationally only (Tableau 3.1); DHS microdata is off the table (ask 047/048). EHCVM 2020
asks (`s01q12`) and is licensed. Afrobarometer surveyed Comoros for the first time in round 10;
the data set is not released yet (the data-sets page filtered to Comoros lists nothing on
2026-10-03, against 13 for Namibia), so its island split cannot be read.

## THE CONSTRUCTION

  * The sample description's weighted religion line: Christians 0.3, Muslims 99.6, `Autre` 0.1,
    refused 0.0. Q97 shows what `Autre` is: its only answer outside Christianity, Islam and refusal
    is `Aucune` (0.1), so `Autre` is no religion. Refused (0.0) is dropped.
  * The same three shares on each island's census count, rounded inside each island.

Usage:
    python sources/km.py --fetch    download the four PDFs (about 10 MB)
    python sources/km.py            rebuild data/normalized/km.csv and print the checks
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

RAW = os.path.join(ROOT, "data", "raw", "km")
SOR = os.path.join(RAW, "COM_R10-Resume-des-resultats-ka-bh-23aout25-rev-1nov25.pdf")
SOR_URL = ("https://www.afrobarometer.org/wp-content/uploads/2025/09/"
           "COM_R10-Resume-des-resultats-ka-bh-23aout25-rev-1nov25.pdf")
SOR_SAMPLE_PAGE = 4             # 0-based; printed page 5, the sample description
SOR_Q97_PAGE = 6                # 0-based; printed page 7, Q97
CODEBOOK = os.path.join(RAW, "AB_R10.Comoros_Codebook_25Aug25.pdf")
CODEBOOK_URL = ("https://www.afrobarometer.org/wp-content/uploads/2025/10/"
                "AB_R10.Comoros_Codebook_25Aug25.pdf")
CODEBOOK_Q97_PAGES = (179, 180, 181)
INSEED = os.path.join(RAW, "inseed_cartographie_vulnerabilite_covid19.pdf")
INSEED_URL = ("https://www.inseed-comores.org/backend/uploads/"
              "Rapport_cartographie_sur_la_vulnerabilite_au_COVID_19_UNION_COMORES_cea7f383b1.pdf")
INSEED_TABLE_PAGES = (26, 27)   # 0-based; annex Tableau 1
DHS = os.path.join(RAW, "dhs_fr278.pdf")
DHS_URL = "https://dhsprogram.com/pubs/pdf/FR278/FR278.pdf"
DHS_TABLE_PAGE = 47             # 0-based; printed page 26, Tableau 3.1
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "km.csv")

# Sample description, `Religion`: (unweighted %, weighted %). Asserted against the text layer.
SAMPLE = {"Chrétiens": (0.3, 0.3), "Musulmans": (99.5, 99.6), "Autre": (0.2, 0.1),
          "Refus": (0.1, 0.0)}
DRAWN = ("Musulmans", "Chrétiens", "Autre")
# Q97, Total column (weighted %), every row the summary prints.
Q97 = {"Aucune": 0.1, "Chrétien seulement": 0.0, "Mormon/saints des derniers jours": 0.2,
       "Musulman seulement": 96.2, "Sunnite seulement": 1.0, "Ismaélite": 2.4, "Refus": 0.0}
# Codebook (25 August 2025), Q97 unweighted counts of the non-zero codes.
CODEBOOK_Q97 = {"Aucune": 1, "Mormon": 3, "Musulman seulement": 1154, "Sunnite seulement": 12,
                "Ismaélite": 29, "A refusé": 1}
N = 1200
# INSEED Tableau 1, island rows: total, male, female (RGPH 2017).
ISLANDS = {"Mwali": (51_567, 26_620, 24_946), "Ndzuwani": (327_382, 165_110, 162_272),
           "Ngazidja": (379_367, 190_082, 189_285)}
NATIONAL = (758_316, 381_812, 376_503)
PREFECTURES = {"Mwali": 3, "Ndzuwani": 6, "Ngazidja": 9}
# DHS 2012 Tableau 3.1: weighted % (women, men).
DHS2012 = {"Musulmane": (99.0, 99.3), "Catholique/Protestante": (0.3, 0.3),
           "Manquant": (0.6, 0.4)}
DHS_GAP = 0.5                   # points, Christian share, 2012 against 2025
YEAR = 2025
SOURCE_ID = "km_ab_r10_national_mix"
# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(muslim=99.6, christian=0.3, none=0.1, ismaili=2.4, pew_muslim=98.3,
            pew_christian=0.51, pew_other=1.06, christians_drawn=2275, census=758_316)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((SOR_URL, SOR), (CODEBOOK_URL, CODEBOOK), (INSEED_URL, INSEED),
                     (DHS_URL, DHS)):
        if os.path.exists(dst) and os.path.getsize(dst) > 500_000:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise SystemExit(f"{url} did not return a PDF")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)


def _lines(path, page):
    import fitz

    return [x.strip() for x in fitz.open(path)[page].get_text().split("\n") if x.strip()]


def _num(s):
    s = s.replace(" ", "").replace("\xa0", "").replace(" ", "")
    return float(s.replace(",", ".")) if s.startswith(("0", "1", "2", "3", "4", "5", "6", "7",
                                                       "8", "9", ",")) else None


def read_survey():
    lines = _lines(SOR, SOR_SAMPLE_PAGE)
    i = lines.index("Religion")
    got = {}
    for label in SAMPLE:
        j = lines.index(label, i)
        got[label] = (_num(lines[j + 1]), _num(lines[j + 2]))
    if got != SAMPLE:
        raise SystemExit(f"summary p.5 religion line reads {got}, pinned {SAMPLE}")
    if round(sum(v[1] for v in SAMPLE.values()), 1) != 100.0:
        raise SystemExit("the weighted religion line does not sum to 100")

    flat = _lines(SOR, SOR_Q97_PAGE)
    if not any(x.startswith("Q97.") for x in flat):
        raise SystemExit("Q97 is not on the pinned page of the summary")
    # Each row's five figures (urban, rural, men, women, total) follow its label; a blank cell
    # prints nothing, so the Total is the last number before the next label.
    q = next(k for k, x in enumerate(flat) if x.startswith("Q97."))
    end = next((k for k, x in enumerate(flat) if k > q and x.startswith("Q90A.")), len(flat))
    flat = flat[q:end]
    pos = {}
    for k in Q97:
        hit = [m for m, x in enumerate(flat) if x == k or x.startswith(k + " ")]
        if len(hit) != 1:
            raise SystemExit(f"Q97 row `{k}` found {len(hit)} times on the pinned page")
        pos[k] = hit[0]
    order = sorted(pos, key=pos.get)
    q97 = {}
    for a, b in zip(order, order[1:] + [None]):
        stop = pos[b] if b else len(flat)
        nums = [_num(x) for x in flat[pos[a] + 1:stop]]
        nums = [x for x in nums if x is not None]
        q97[a] = nums[-1]
    if q97 != Q97:
        raise SystemExit(f"summary p.7 Q97 Total column reads {q97}, pinned {Q97}")
    muslim = round(Q97["Musulman seulement"] + Q97["Sunnite seulement"] + Q97["Ismaélite"], 1)
    if muslim != SAMPLE["Musulmans"][1]:
        raise SystemExit(f"Q97's Muslim rows sum to {muslim}, the sample line says "
                         f"{SAMPLE['Musulmans'][1]}")
    if Q97["Aucune"] != SAMPLE["Autre"][1]:
        raise SystemExit("`Autre` in the sample line is not Q97's `Aucune`")
    print(f"  Afrobarometer R10 summary read back: Muslim {SAMPLE['Musulmans'][1]}%, Christian "
          f"{SAMPLE['Chrétiens'][1]}%, other (= Q97 `Aucune`) {SAMPLE['Autre'][1]}%, refused "
          f"{SAMPLE['Refus'][1]}%, weighted")
    print(f"      Q97 Muslim rows: only {Q97['Musulman seulement']}, Sunni {Q97['Sunnite seulement']}, "
          f"Ismaili {Q97['Ismaélite']}; Christian rows: only {Q97['Chrétien seulement']}, Mormon "
          f"{Q97['Mormon/saints des derniers jours']}")


def codebook_witness():
    if not os.path.exists(CODEBOOK):
        print(f"  (codebook witness skipped: {CODEBOOK} missing)")
        return
    lines = []
    for p in CODEBOOK_Q97_PAGES:
        lines += _lines(CODEBOOK, p)
    i = next(k for k, x in enumerate(lines) if x.startswith("Q97. Religion"))
    import re

    got, label = {}, None
    for x in lines[i + 1:]:
        if x.startswith("Q97Other"):
            break
        if re.match(r"^-?\d+ \S", x):          # a code and the start of its label
            label = x
        elif x.isdigit() and label is not None:  # the count that closes a (wrapped) label
            if int(x) > 0:
                got[label] = int(x)
            label = None
    short = {}
    for lab, n in got.items():
        for key in CODEBOOK_Q97:
            if key in lab:
                short[key] = n
    if short != CODEBOOK_Q97 or sum(short.values()) != N:
        raise SystemExit(f"codebook Q97 reads {got}, pinned {CODEBOOK_Q97}")
    print(f"  codebook (25 August 2025), Q97 unweighted: {CODEBOOK_Q97}, {N:,} respondents")


def read_population():
    lines = []
    for p in INSEED_TABLE_PAGES:
        lines += _lines(INSEED, p)
    if "Tableau 1 : Population par préfecture et commune" not in lines:
        raise SystemExit("INSEED's annex Tableau 1 is not on the pinned pages")
    n = lambda s: int(s.replace(" ", "").replace("\xa0", ""))     # noqa: E731
    k = lines.index("Ensemble des trois îles")
    if tuple(n(x) for x in lines[k + 1:k + 4]) != NATIONAL:
        raise SystemExit(f"Tableau 1 national row: {lines[k + 1:k + 4]}")
    pop, idx = {}, {}
    for isl in ISLANDS:
        j = lines.index(f"Ile de {isl.upper()}")
        idx[isl] = j
        pop[isl] = tuple(n(x) for x in lines[j + 1:j + 4])
    if pop != ISLANDS:
        raise SystemExit(f"Tableau 1 island rows: {pop}, pinned {ISLANDS}")
    if tuple(sum(v[c] for v in ISLANDS.values()) for c in range(3)) != NATIONAL:
        raise SystemExit("the island rows do not sum to the national row")
    # every préfecture under its island, and the préfectures sum to the island
    order = sorted(idx, key=idx.get)
    for isl, nxt in zip(order, order[1:] + [None]):
        block = lines[idx[isl]:idx[nxt] if nxt else len(lines)]
        pref = [m for m, x in enumerate(block) if x.startswith("Préfecture de ")]
        tot = sum(n(block[m + 1]) for m in pref)
        if len(pref) != PREFECTURES[isl] or tot != ISLANDS[isl][0]:
            raise SystemExit(f"{isl}: {len(pref)} préfectures summing to {tot:,}")
    print(f"  INSEED Tableau 1 (RGPH 2017) read back: {NATIONAL[0]:,} people; "
          + ", ".join(f"{k} {v[0]:,}" for k, v in ISLANDS.items())
          + "; each island's préfectures sum to it")
    return {k: v[0] for k, v in ISLANDS.items()}


def dhs_witness():
    if not os.path.exists(DHS):
        print(f"  (DHS 2012 witness skipped: {DHS} missing; {DHS_URL})")
        return
    lines = _lines(DHS, DHS_TABLE_PAGE)
    if "Tableau 3.1  Caractéristiques sociodémographiques des enquêtés" not in lines:
        raise SystemExit("DHS 2012 Tableau 3.1 is not on the pinned page")
    i = lines.index("Religion")
    for label, (w, m) in DHS2012.items():
        j = lines.index(label, i)
        if (_num(lines[j + 1]), _num(lines[j + 4])) != (w, m):
            raise SystemExit(f"DHS 2012 Tableau 3.1 `{label}`: {lines[j + 1:j + 7]}")
    chr_w = 100 * DHS2012["Catholique/Protestante"][0] / (100 - DHS2012["Manquant"][0])
    gap = abs(chr_w - SAMPLE["Chrétiens"][1])
    print(f"  DHS 2012 Tableau 3.1, women 15-49: Christian {chr_w:.2f}% of those answering, "
          f"against 2025's {SAMPLE['Chrétiens'][1]}% (gap {gap:.2f}, bar {DHS_GAP})")
    if gap > DHS_GAP:
        raise SystemExit("the 2012 and 2025 surveys disagree beyond the bar")


def pew_row():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)))
    r = t[(t["Country"] == "Comoros") & (t["Year"] == 2020)].iloc[0]
    print(f"  Pew 2020: Muslim {float(r['Muslims']):.2f}%, Christian {float(r['Christians']):.2f}%, "
          f"unaffiliated {float(r['Religiously_unaffiliated']):.2f}%, other "
          f"{float(r['Other_religions']):.2f}%")
    return (round(float(r["Muslims"]), 1), round(float(r["Christians"]), 2),
            round(float(r["Other_religions"]), 2))


def main():
    if "--fetch" in sys.argv or not (os.path.exists(SOR) and os.path.exists(INSEED)):
        fetch()
    read_survey()
    codebook_witness()
    pop = read_population()
    dhs_witness()
    pew = pew_row()

    total = sum(SAMPLE[c][1] for c in DRAWN)
    shares = {c: SAMPLE[c][1] / total for c in DRAWN}
    pop = pd.Series(pop)
    m = pd.DataFrame({c: pop * s for c, s in shares.items()})
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("an island's rounded counts do not sum to its population")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "island"
    out["geo_name"] = out["geo_id"]
    out["basis"] = "self_id"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("Afrobarometer R10 (2025) national weighted shares, the same mix on every "
                   "island, on the 2017 census's island counts")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people, 3 islands)")
    by = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for c, k in by.items():
        print(f"      {c:<12} {k:>10,}")

    got = dict(muslim=SAMPLE["Musulmans"][1], christian=SAMPLE["Chrétiens"][1],
               none=SAMPLE["Autre"][1], ismaili=Q97["Ismaélite"], pew_muslim=pew[0],
               pew_christian=pew[1], pew_other=pew[2], christians_drawn=int(by["Chrétiens"]),
               census=int(out["count"].sum()))
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
