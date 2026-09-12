"""Haiti — religion by department, from IHSI's own ECVMAS 2012 household survey.

Reads data/raw/ht/ecvmas/*.sav and writes data/normalized/ht.csv.
`sources/ht.md` is this country's record and `sources.md` §9cg is the write-up.

## THE QUEUE REACHED HAITI THROUGH LAPOP, AND THE OFFICE HAS BETTER

`queue.md` §11ad offered Haiti as **7,252 AmericasBarometer respondents pooled over six
waves**, all ten departments, with a note from Anita that the 4.4% Vodou figure is a
self-identification floor and wants a second source before it is drawn. Asking what IHSI
publishes instead turns up two of its own instruments and one census, none of which anybody
here had looked at:

    LAPOP AmericasBarometer 2010-2023 pooled      7,252 respondents, 10 departments, 18+
    IHSI ECVMAS 2012                            17,977 respondents, 10 departments, 10+
    IHSI ECVH 2001                              32,840 respondents,  9 departments, all ages
    IHSI 4eme RGPH 2003 (census)             8,373,750 people,   NATIONAL ONLY, all ages

**The build is ECVMAS 2012** and the other three are cross-checks, which is the useful shape:
the census fixes the national level on a complete count, ECVH 2001 replicates the departmental
ORDERING with a different questionnaire eleven years earlier, and LAPOP is a third instrument
with a different sponsor.

## WHERE THE FILE CAME FROM, BECAUSE THE HOST IS GONE

IHSI published the whole ECVMAS 2012 database openly, no account and no request form, at
`www.ihsi.ht/pdf/ecvmas/ecvmas_base_donnees/2_ECVMAS_BASE DE DONNEES.zip`. **That domain is
now squatted**: `ihsi.ht` answers 301 and lands on an unrelated commercial page, and
`rgph-haiti.ht`, the fifth census's own site, answers 200 with a betting site. The office
itself moved to `ihsi.gouv.ht` and did not bring the microdata. So the zip is fetched from the
Wayback Machine's byte-identical copy, the way `sources/do.py` fetches ONE's files, and the
2003 census tables still come from the live `ihsi.gouv.ht`.

## THE QUESTION IS ASKED OF EVERYONE AGED TEN AND OVER

`I_H04`, *"Quelle est votre religion ?"*, is in ECVMAS's individual module, and the module's
universe is **10 and over**: of 23,775 people in the household rosters, 5,792 have no answer
and 5,714 of those are children under ten. So this map is drawn from the religion of Haitians
aged 10+, applied to each department's whole population, which is what `sources/lapop.py` does
with an 18+ survey everywhere else on this map.

**The 2003 census says what that costs, because it tabulated religion BY AGE GROUP.** Recompute
its national table over the 10-and-overs only and no category moves as much as a point: the
largest movement is `Aucune religion`, from 10.22% of everyone to 9.40% of the 10-and-overs,
and Catholicism goes the other way by a third of a point, 54.68% to 55.02%.
`age_universe_check()` prints the whole comparison. Children are recorded as having no
religion rather more often than adults, which is the only systematic thing in it.

## THE DEPARTMENT JOIN IS THE DANGEROUS PART AND IT IS NOT A CODE JOIN

ECVMAS's `DEPT` is 1..10 in French alphabetical order; COD's pcodes are `HT01`..`HT10` in
Haiti's traditional order. Both are dense 1..10 sequences over the same ten units, so the
obvious pairing runs clean and is wrong for **all ten**. `sources/ht_geo.py` asserts that and
does the join on the French name; `DEPT` never leaves this file.

## WHAT CARRIES ITS OWN GEOGRAPHY, ON TEN UNITS

The split-half is run on the parity of ECVMAS's 500 sampling clusters, and on ten departments
the bar is **+0.65** — a Spearman over `n` units has SE about 1/sqrt(n-1), so a country with
few units needs a much higher correlation to be distinguishable from zero. **A single parity
split is one draw and on ten units it is a noisy one**, so this file reports the parity split
AND the median over 400 random cluster half-splits, which estimates the same quantity with
less draw-to-draw variance. The bar is not moved.

Even so the internal test is weak here, and the reason to believe the Catholic geography is
not internal at all: **ECVH 2001 ranked the same departments with a different questionnaire
eleven years earlier, and the orderings agree.** `ecvh_check()` reports it. Two overrides are
taken and both are printed on every run; see `OVERRIDE`.

Usage:
    python sources/ht.py --fetch    a ~10 MB zip from the Wayback Machine, a 6 MB PDF
    python sources/ht.py            rebuild data/normalized/ht.csv
"""

import os
import re
import sys
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ht")
ECVMAS_DIR = os.path.join(RAW, "ecvmas")
ZIP = os.path.join(RAW, "ecvmas_base_de_donnees.zip")
CENSUS_PDF = os.path.join(RAW, "rgph2003_population_tables.pdf")
LOOKUP = os.path.join(ROOT, "data", "geo", "ht", "ht_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ht.csv")

SOURCE_ID = "ht_ihsi_ecvmas_2012"
N_DEPARTMENTS = 10

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    # IHSI's own publication of the ECVMAS 2012 database. `www.ihsi.ht` is a squatted domain
    # now, so this is the Wayback Machine's copy of the file the office put up.
    ZIP: "https://web.archive.org/web/2018id_/http://ihsi.ht/pdf/ecvmas/"
         "ecvmas_base_donnees/2_ECVMAS_BASE%20DE%20DONNEES.zip",
    # The 4eme RGPH 2003 population tables, still live on the office's current host. Tableau
    # 206 is religion by age group and sex, national, and it has a text layer.
    CENSUS_PDF: "https://ihsi.gouv.ht/public/tableau/POPULATION.pdf",
}

# The two files this build needs out of the 32 in the zip.
IND_SAV = "2_ECVMAS_Individus_e h i j k n p_ok.sav"
ECH_SAV = "0_ECVMAS_ECHANTILLON_ok.sav"

# `I_H04`'s value labels, verbatim from the .sav's own label set (spec §2.4 keeps the
# source's own words). Asserted against the file on every run by check_labels().
CATEGORY = {
    1: "Catholique",
    2: "Baptiste",
    3: "Adventiste",
    4: "Pentecôtiste",
    5: "Méthodiste",
    6: "Episcopale",
    7: "Témoin de Jéhovah",
    8: "Autre protestant",
    9: "Vaudou",
    10: "Musulman",
    11: "Autre",
    12: "Aucune",
}

# -8 "Ne sait pas" and -9 "Refus de réponse". Six people between them; they are the `gap`.
NONRESPONSE = {-8: "Ne sait pas", -9: "Refus de réponse"}

# The split-half bar, 1.96/sqrt(n-1) on ten departments. NEVER MOVED — see OVERRIDE.
STABILITY_BAR = 1.96 / np.sqrt(N_DEPARTMENTS - 1)

# Categories drawn on their own department shares. Asserted against `stability()` on every
# run, so a change in the data stops the build rather than quietly redrawing the country.
# Three of these pass the split-half as written; four are OVERRIDE entries below.
CARRIES = [1, 2, 3, 4, 5, 8, 9]

# THE BAR IS NOT MOVED. These are four named decisions with their reasons, printed on every
# run (§9bi's rule), and every one of them rests on ECVH 2001 replicating the ORDERING with
# a different questionnaire, a different sample and eleven years in between — which is what
# the internal split-half is a cheap proxy for. On ten departments the internal test has
# almost no power: `Catholique` has a 26-point spread and a chi-square of 1e-117 and still
# lands within 0.003 of the bar. `ecvh_check()` prints the numbers quoted here.
OVERRIDE = {
    1: "ECVH 2001 and ECVMAS 2012 rank the nine departments that existed in 2001 the same "
       "way, Spearman +0.85, which 29 of 20,000 random pairings reach. Two IHSI household "
       "surveys eleven years apart agreeing on the order is stronger evidence than one "
       "survey agreeing with itself, and the internal test on ten units cannot separate "
       "+0.65 from zero.",
    2: "Same ECVH 2001 replication, +0.78, reached by 35 of 20,000 pairings. Baptists are "
       "17.5% of Haiti and run 7.9% of Grand'Anse to 25.8% of Nord; drawn flat the map "
       "would be asserting those are the same.",
    4: "Pentecostals cannot be compared category-for-category across the two cards, because "
       "ECVH offered `Eglise de Dieu` and `Eglise wesleyenne` and ECVMAS offers neither; "
       "the same people answer `Pentecôtiste` or `Autre protestant` in 2012. Compared as "
       "the bloc those cards can both express, the replication is +0.90 and 8 of 20,000 "
       "random pairings reach it. `Autre protestant`, the other half of that bloc, passes "
       "the internal split-half on its own.",
    9: "ECVH 2001 and ECVMAS 2012 both put Artibonite far ahead of every other department "
       "(7.3% then, 5.7% now, against 2.0% and 1.2% for the next highest) and both put "
       "Nord-Est at the bottom. The whole-ordering Spearman is only +0.47 because the "
       "middle seven shuffle, which is §9bi's Guatemala case: the ends replicate and the "
       "middle does not, so `note_public` says to read the ends and not the ordering.",
}

# The Protestant bloc as ECVMAS splits it, used only for printing and for the cross-checks.
PROTESTANT = [2, 3, 4, 5, 6, 7, 8]

# IHSI, ECVH 2001, Tableau 2.2.4.2, "Distribution en pourcentage (%) de la population selon
# la religion déclarée par département géographique", from ecvh_volume_I_(juillet2003).pdf
# pages 76-77. Transcribed as published, in per cent, `-` read as 0.0. NINE departments:
# Nippes was created out of Grand'Anse in September 2003, so ECVH's `Grande-Anse` covers
# both of today's units and the comparison below is run on nine.
#
# THE CARD IS NOT ECVMAS's. ECVH offered `Eglise de Dieu` and `Eglise wesleyenne` and offered
# NO "no religion" answer at all; ECVMAS offers `Autre protestant`, `Méthodiste` and
# `Aucune` and offers no Church of God. Only the seven answers on BOTH cards are compared.
ECVH_2001 = {
    #                    Ouest SudEst  Nord  NordE  Artib Centre   Sud   GrAnse NordO
    "Catholique":        [55.9, 60.8, 62.3, 64.7, 53.7, 51.8, 61.5, 68.5, 59.2],
    "Baptiste":          [13.3, 16.1, 20.3, 26.4, 14.9, 22.8, 14.5,  7.9, 23.6],
    "Adventiste":        [3.3,   1.4,  5.5,  2.9,  2.2,  1.8,  1.3,  0.9,  6.1],
    "Episcopale":        [0.7,   0.6,  0.1,  0.1,  0.7,  1.4,  0.2,  0.1,  0.3],
    "Pentecôtiste":      [8.4,   0.9,  3.8,  0.8,  2.7,  3.4,  5.1,  1.6,  1.1],
    "Eglise de Dieu":    [9.8,  13.4,  2.0,  0.9, 11.5, 14.2, 10.6, 10.9,  6.6],
    "Eglise wesleyenne": [1.2,   1.3,  2.7,  0.2,  0.2,  0.6,  0.3,  0.5,  0.0],
    "Témoin de Jéhovah": [0.5,   0.0,  0.1,  0.1,  0.8,  0.0,  0.8,  0.4,  0.4],
    "Vodouisant":        [1.3,   0.4,  1.9,  0.0,  7.3,  0.7,  0.9,  2.0,  1.5],
    "Autres":            [5.3,   5.1,  1.3,  4.0,  6.0,  3.4,  4.6,  6.9,  1.3],
    "NSP":               [0.3,   0.1,  0.1,  0.0,  0.1,  0.0,  0.1,  0.4,  0.0],
}
# The pcodes ECVH's nine columns are in, in order. HT08+HT10 is the undivided Grand'Anse.
ECVH_UNITS = ["HT01", "HT02", "HT03", "HT04", "HT05", "HT06", "HT07",
              ("HT08", "HT10"), "HT09"]
ECVH_N = [8355, 2576, 3737, 1936, 3923, 2778, 3192, 3335, 3008]
ECVH_TOTAL = 32_840
# ECVH label -> the ECVMAS code that asks the same question. The four ECVH answers with no
# ECVMAS counterpart, and the four ECVMAS answers with no ECVH counterpart, are left out.
ECVH_PAIRS = {
    "Catholique": 1, "Baptiste": 2, "Adventiste": 3, "Pentecôtiste": 4,
    "Episcopale": 6, "Témoin de Jéhovah": 7, "Vodouisant": 9,
}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for dst, url in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=1200) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def unpack():
    """Pull the two .sav this build needs out of IHSI's zip.

    The member names carry accented Latin-1 bytes that Python's zipfile decodes as cp437, so
    they are matched on a prefix rather than compared whole.
    """
    if not os.path.exists(ZIP):
        raise SystemExit(f"{ZIP} missing — run with --fetch")
    os.makedirs(ECVMAS_DIR, exist_ok=True)
    got = {}
    with zipfile.ZipFile(ZIP) as z:
        for want, prefix in (("ind", "2_ECVMAS_Individus"), ("ech", "0_ECVMAS_ECHANTILLON")):
            hits = [n for n in z.namelist() if os.path.basename(n).startswith(prefix)]
            if len(hits) != 1:
                raise SystemExit(f"{len(hits)} members match {prefix!r} in the ECVMAS zip: "
                                 f"{hits}")
            dst = os.path.join(ECVMAS_DIR, prefix + ".sav")
            if not os.path.exists(dst):
                with open(dst + ".part", "wb") as f:
                    f.write(z.read(hits[0]))
                os.replace(dst + ".part", dst)
            got[want] = dst
    return got["ind"], got["ech"]


def check_labels(ind_path):
    """Assert CATEGORY and the department labels against the .sav's own label sets."""
    import pyreadstat

    _, meta = pyreadstat.read_sav(ind_path, metadataonly=True)
    got = {int(k): v for k, v in meta.variable_value_labels.get("I_H04", {}).items()}
    want = dict(CATEGORY)
    want.update(NONRESPONSE)
    if got != want:
        raise SystemExit(f"I_H04's value labels have changed: {got} against {want} — the "
                         "answer card is not the one this file was written against")
    print(f"  I_H04: {meta.column_names_to_labels.get('I_H04', '')}")
    print(f"    {len(CATEGORY)} answers plus {len(NONRESPONSE)} non-response codes, "
          "matching the .sav's label set exactly")


def load(ind_path, ech_path):
    """The person file joined to the sample file for its weight, with the universe asserted."""
    import pyreadstat

    ind, _ = pyreadstat.read_sav(ind_path)
    ech, _ = pyreadstat.read_sav(ech_path)
    print(f"Haiti: {len(ind):,} people in {ind.groupby(['i_id1new', 'camp', 'i_id2new']).ngroups:,} "
          f"households in {ind['i_id1new'].nunique():,} ECVMAS clusters")

    m = ind.merge(ech, left_on=["i_id1new", "camp", "i_id2new"],
                  right_on=["hh_id1new", "camp", "hh_id2new"], how="left",
                  suffixes=("", "_e"), validate="many_to_one")
    if m["Poids_Finaux"].isna().any():
        raise SystemExit(f"{int(m['Poids_Finaux'].isna().sum()):,} people are in the person "
                         "file with no row in ECHANTILLON — the household key is not what it "
                         "looks like")
    # THE SAMPLE FILE CARRIES ITS OWN DEPT AND THE PERSON FILE CARRIES ANOTHER. If the
    # (cluster, camp, household) key were wrong the two would disagree, and every national
    # total would still reconcile. This is the join witness.
    if int((m["DEPT"] != m["DEPT_e"]).sum()):
        raise SystemExit("the person file's DEPT disagrees with ECHANTILLON's — the "
                         "household join is wrong")
    print("  ECHANTILLON's own DEPT agrees with the person file's on every row, so the "
          "(cluster, camp, household) join holds")

    unknown = sorted(set(m.loc[m["I_H04"].notna(), "I_H04"].astype(int))
                     - set(CATEGORY) - set(NONRESPONSE))
    if unknown:
        raise SystemExit(f"I_H04 codes with no label: {unknown} — the card has changed")

    # ---- the universe: the individual module is asked of 10 and over ----
    blank = m["I_H04"].isna()
    kids = blank & (m["HH_E06B"] < 10)
    print(f"  {int(blank.sum()):,} people have no I_H04, and {int(kids.sum()):,} of them are "
          f"under ten; the individual module's universe is 10 and over")
    if m.loc[m["I_H04"].notna(), "HH_E06B"].min() < 10:
        raise SystemExit("someone under ten answered I_H04 — the universe is not 10+")
    if int((blank & (m["HH_E06B"] >= 10)).sum()) > 200:
        raise SystemExit(f"{int((blank & (m['HH_E06B'] >= 10)).sum()):,} people aged 10+ have "
                         "no religion answer, which is more attrition than this file expects")

    refused = m["I_H04"].isin(list(NONRESPONSE))
    print(f"  {int(refused.sum())} of the 10-and-overs said "
          f"{' or '.join(NONRESPONSE.values())}")

    df = m[m["I_H04"].notna() & ~refused].copy()
    df["code"] = df["I_H04"].astype(int)
    df["w"] = df["Poids_Finaux"].astype(float)
    df["dept"] = df["DEPT"].astype(int)
    df["clu"] = df["i_id1new"].astype(int)
    bad = sorted(set(df["dept"]) - set(range(1, N_DEPARTMENTS + 1)))
    if bad:
        raise SystemExit(f"department codes outside 1-{N_DEPARTMENTS}: {bad}")
    if df["dept"].nunique() != N_DEPARTMENTS:
        raise SystemExit(f"{df['dept'].nunique()} departments have respondents, expected "
                         f"{N_DEPARTMENTS}")
    print(f"  {len(df):,} answers in all {N_DEPARTMENTS} departments, weighting to "
          f"{df['w'].sum():,.0f} people aged 10 and over")
    # 1,878 of the individuals are in the 30 displaced-persons camp enumeration areas the
    # survey sampled separately two and a half years after the earthquake. They are people
    # living in Haiti in 2012 and are kept; `camp` is reported so the share is on the record.
    print(f"    {int((df['camp'] == 1).sum()):,} of them are in the 30 displaced-persons "
          f"camp enumeration areas, {(df.loc[df['camp'] == 1, 'w'].sum() / df['w'].sum()):.2%} "
          "by weight")
    return df


def national(df):
    return df.groupby("code")["w"].sum() / df["w"].sum()


def census_table():
    """The 2003 census's Tableau 206, national, by age group — read out of IHSI's own PDF.

    The table is a fixed-width text layer inside `POPULATION.pdf`, so this is a reading of
    the census rather than a transcription of it. Returns a DataFrame indexed by age band
    with one column per census religion label.
    """
    if not os.path.exists(CENSUS_PDF):
        raise SystemExit(f"{CENSUS_PDF} missing — run with --fetch")
    import fitz

    cols = ["Total", "Aucune religion", "Catholique", "Adventiste", "Témoin de Jéhovah",
            "Baptiste", "Méthodiste", "Episcopale", "Pentecôtiste", "Vaudouïsant",
            "Musulman", "Mormon", "Autre religion"]
    doc = fitz.open(CENSUS_PDF)
    page = None
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if "Tableau 206" in t and "ENSEMBLE DU PAYS" in t and "Deux sexes" in t:
            page = t
            break
    if page is None:
        raise SystemExit("Tableau 206's 'Deux sexes' page is not in POPULATION.pdf — the "
                         "office has republished it")
    # The page carries three blocks — Deux sexes, Masculin, Féminin — each with its own
    # `Total` row. Cut to the first, or the parse silently returns the male half.
    body = page.split("Deux sexes", 1)[1].split("Masculin", 1)[0]
    rows = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("│")]
        if len(cells) < len(cols) + 1:
            continue
        label = cells[0].strip()
        nums = []
        for c in cells[1:len(cols) + 1]:
            c = c.replace(",", "").strip()
            if not re.fullmatch(r"\d+", c):
                nums = None
                break
            nums.append(int(c))
        if nums and label:
            rows[label] = nums
    if "Total" not in rows:
        raise SystemExit(f"no Total row parsed out of Tableau 206; got {list(rows)}")
    t = pd.DataFrame(rows, index=cols).T
    if int(t.loc["Total", "Total"]) != 8_373_750:
        raise SystemExit(f"Tableau 206's total is {int(t.loc['Total', 'Total']):,}, not the "
                         "census's 8,373,750 — the parse is wrong")
    if int(t.loc["Total", cols[1:]].sum()) != int(t.loc["Total", "Total"]):
        raise SystemExit("Tableau 206's religion columns do not sum to its own total")
    return t


def age_universe_check(cen):
    """What excluding the under-tens costs, measured on the census that included them."""
    bands = [i for i in cen.index if i != "Total"]
    young = [b for b in bands if b.startswith("0 -") or b.startswith("5 -")]
    older = [b for b in bands if b not in young]
    if len(young) != 2:
        raise SystemExit(f"expected two under-ten bands in Tableau 206, got {young}")
    cats = [c for c in cen.columns if c != "Total"]
    allp = cen.loc["Total", cats] / cen.loc["Total", "Total"]
    tenp = cen.loc[older, cats].sum() / cen.loc[older, "Total"].sum()
    print(f"\n  the age universe, measured on the 2003 census (which asked everyone): "
          f"{cen.loc[young, 'Total'].sum() / cen.loc['Total', 'Total']:.1%} of Haiti was "
          f"under ten")
    print(f"    {'census category':<22}{'all ages':>10}{'aged 10+':>10}{'shift':>9}")
    for c in sorted(cats, key=lambda k: -allp[k]):
        print(f"    {c:<22}{allp[c] * 100:9.2f}%{tenp[c] * 100:9.2f}%"
              f"{(tenp[c] - allp[c]) * 100:+9.2f}")
    worst = max(cats, key=lambda c: abs(tenp[c] - allp[c]))
    print(f"    largest movement {worst} at {(tenp[worst] - allp[worst]) * 100:+.2f} points, "
          "so drawing 10+ shares over the whole population is under a point out on every "
          "category")


def census_check(nat, cen):
    """The census as an outside witness on the national level. Reported, never asserted.

    Nine years apart and on different answer cards: the census offered `Mormon` and no
    `Autre protestant`, so its `Autre religion` holds people ECVMAS puts in a Protestant box.
    """
    pairs = [("Catholique", "Catholique", 1), ("Baptiste", "Baptiste", 2),
             ("Pentecôtiste", "Pentecôtiste", 4), ("Adventiste", "Adventiste", 3),
             ("Méthodiste", "Méthodiste", 5), ("Episcopale", "Episcopale", 6),
             ("Témoin de Jéhovah", "Témoin de Jéhovah", 7), ("Vaudou", "Vaudouïsant", 9),
             ("Musulman", "Musulman", 10), ("Aucune", "Aucune religion", 12)]
    print("\n  cross-check 1: the 4eme RGPH 2003, a complete count nine years earlier, "
          "national only")
    print(f"    {'answer':<20}{'census 2003':>13}{'ECVMAS 2012':>13}{'change':>10}")
    for label, ccol, code in pairs:
        c = cen.loc["Total", ccol] / cen.loc["Total", "Total"]
        e = float(nat.get(code, 0.0))
        print(f"    {label:<20}{c * 100:12.2f}%{e * 100:12.2f}%{(e - c) * 100:+10.2f}")
    other_c = (cen.loc["Total", "Autre religion"] + cen.loc["Total", "Mormon"]) \
        / cen.loc["Total", "Total"]
    other_e = float(nat.get(8, 0.0) + nat.get(11, 0.0))
    print(f"    {'other + Mormon':<20}{other_c * 100:12.2f}%{other_e * 100:12.2f}%"
          f"{(other_e - other_c) * 100:+10.2f}   different cards, see the docstring")
    prot_c = sum(cen.loc["Total", c] for c in
                 ["Adventiste", "Témoin de Jéhovah", "Baptiste", "Méthodiste", "Episcopale",
                  "Pentecôtiste", "Mormon", "Autre religion"]) / cen.loc["Total", "Total"]
    prot_e = float(sum(nat.get(c, 0.0) for c in PROTESTANT) + nat.get(11, 0.0))
    print(f"    everything non-Catholic Christian, taken together, to get past the card "
          f"difference: {prot_c * 100:.2f}% -> {prot_e * 100:.2f}%")


def ecvh_check(df, n_perm=20000, seed=0):
    """ECVH 2001 as an outside witness on the departmental ORDERING.

    This is the evidence the `OVERRIDE` entries rest on, so it is worth being clear about
    what it is and is not. It is NOT independent of Haiti's real geography, which is the
    point: two IHSI household surveys eleven years apart, different questionnaires,
    different samples, different answer cards. If the ordering they produce agrees, the
    ordering is a property of the country rather than of one sample.

    Run on ECVH's nine departments, so ECVMAS's Grand'Anse and Nippes are pooled back into
    the undivided Grand'Anse the 2001 survey measured.
    """
    print(f"\n  cross-check 2: IHSI's ECVH 2001, n={ECVH_TOTAL:,}, nine departments "
          "(Nippes not yet split off), a different questionnaire eleven years earlier")
    by = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    tot = by.sum(axis=1)

    def pooled(col_values):
        out = []
        for u in ECVH_UNITS:
            keys = u if isinstance(u, tuple) else (u,)
            out.append(sum(col_values.get(k, 0.0) for k in keys))
        return np.array(out)

    rng = np.random.default_rng(seed)
    results = {}
    print(f"    {'answer':<20}{'ECVH 01':>9}{'ECVMAS 12':>11}{'spearman':>10}"
          f"{'of 20k perms':>14}")
    for label, code in ECVH_PAIRS.items():
        if code not in by.columns:
            continue
        num = pooled(dict(by[code]))
        den = pooled(dict(tot))
        a = np.array(ECVH_2001[label], dtype=float) / 100.0
        b = num / den
        sp = pd.Series(a).corr(pd.Series(b), method="spearman")
        perm = np.array([pd.Series(a).corr(pd.Series(rng.permutation(b)), method="spearman")
                         for _ in range(n_perm)])
        beaten = int((perm >= sp).sum())
        nat_a = float((a * np.array(ECVH_N)).sum() / ECVH_TOTAL)
        nat_b = float(num.sum() / den.sum())
        results[code] = sp
        print(f"    {label:<20}{nat_a * 100:8.2f}%{nat_b * 100:10.2f}%{sp:+10.2f}"
              f"{beaten:>9,} reach it")

    # THE PENTECOSTAL BLOC, because the two cards cut it differently and the category-level
    # comparison above is meaningless for it. ECVH's `Eglise de Dieu` is the Church of God,
    # a Pentecostal body ECVMAS has no box for; ECVMAS's `Autre protestant` is a box ECVH
    # does not have. The bloc is the largest object both cards can express.
    ecvh_bloc = np.array([sum(ECVH_2001[k][i] for k in
                              ("Pentecôtiste", "Eglise de Dieu", "Eglise wesleyenne"))
                          for i in range(len(ECVH_N))]) / 100.0
    num = pooled(dict(by[[4, 8]].sum(axis=1)))
    den = pooled(dict(tot))
    b = num / den
    sp = pd.Series(ecvh_bloc).corr(pd.Series(b), method="spearman")
    perm = np.array([pd.Series(ecvh_bloc).corr(pd.Series(rng.permutation(b)),
                                               method="spearman")
                     for _ in range(n_perm)])
    nat_a = float((ecvh_bloc * np.array(ECVH_N)).sum() / ECVH_TOTAL)
    print(f"    {'Pentecostal bloc':<20}{nat_a * 100:8.2f}%{num.sum() / den.sum() * 100:10.2f}%"
          f"{sp:+10.2f}{int((perm >= sp).sum()):>9,} reach it")
    print("      ECVH's Pentecôtiste + Eglise de Dieu + Eglise wesleyenne against ECVMAS's "
          "Pentecôtiste + Autre protestant, which is the largest object both cards can say")
    results["bloc"] = sp
    return results


def stability(df, nat, n_splits=400, seed=0):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY, split on ECVMAS's sampling cluster.

    Split on the CLUSTER and not on the person, because the cluster is the sampling unit:
    two halves drawn person-by-person out of the same 500 neighbourhoods would agree far
    better than two real samples of Haiti would, and the test would pass everything.

    **On ten units a single split is a noisy estimate**, so the parity split is reported
    alongside the median over `n_splits` random cluster half-splits. The verdict is taken
    off the median, which estimates the same quantity with less draw-to-draw variance; the
    bar is untouched at 1.96/sqrt(9).
    """
    from scipy import stats

    codes = sorted(nat.index, key=lambda k: -nat[k])
    clusters = np.array(sorted(df["clu"].unique()))
    cidx = {c: i for i, c in enumerate(clusters)}
    uidx = {u: i for i, u in enumerate(sorted(df["geo_id"].unique()))}
    kidx = {c: i for i, c in enumerate(codes)}

    # W[cluster, unit, code] as a flat (cluster, unit*code) matrix so a split is a matmul.
    w = np.zeros((len(clusters), len(uidx), len(codes)))
    np.add.at(w,
              (df["clu"].map(cidx).to_numpy(),
               df["geo_id"].map(uidx).to_numpy(),
               df["code"].map(kidx).to_numpy()),
              df["w"].to_numpy())
    each_cluster_unit = df.groupby("clu")["geo_id"].nunique()
    if int(each_cluster_unit.max()) != 1:
        raise SystemExit("a sampling cluster spans two departments — the split-half assumes "
                         "it does not")

    def spearmans(mask):
        a = w[mask].sum(axis=0)
        b = w[~mask].sum(axis=0)
        sa = a / a.sum(axis=1, keepdims=True)
        sb = b / b.sum(axis=1, keepdims=True)
        out = []
        for k in range(len(codes)):
            out.append(pd.Series(sa[:, k]).corr(pd.Series(sb[:, k]), method="spearman"))
        return np.array(out, dtype=float)

    parity = spearmans((clusters % 2) == 0)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_splits, len(codes)))
    for i in range(n_splits):
        m = np.zeros(len(clusters), dtype=bool)
        m[rng.permutation(len(clusters))[:len(clusters) // 2]] = True
        draws[i] = spearmans(m)
    med = np.nanmedian(draws, axis=0)

    n_unit = df.groupby("geo_id").size()
    print(f"\n  split-half on cluster parity, bar = +{STABILITY_BAR:.2f} on "
          f"{N_DEPARTMENTS} departments ({len(clusters)} clusters, and the median over "
          f"{n_splits} random half-splits beside it):")
    print(f"    {'category':<20}{'national':>10}{'parity':>8}{'median':>8}{'chi-sq p':>11}"
          f"  verdict")
    carries = []
    for i, c in enumerate(codes):
        hit = df[df["code"] == c].groupby("geo_id").size().reindex(n_unit.index).fillna(0)
        p = (stats.chi2_contingency(np.vstack([hit, n_unit - hit]))[1]
             if hit.sum() > 0 else float("nan"))
        passed = bool(np.isfinite(med[i])) and med[i] >= STABILITY_BAR
        if not passed and c in OVERRIDE:
            passed = True
            verdict = "own geography — OVERRIDE"
        elif passed:
            verdict = "own geography"
        else:
            verdict = "national rate inside the department's residual"
        print(f"    {CATEGORY[c]:<20}{nat[c] * 100:9.2f}%{parity[i]:+8.2f}{med[i]:+8.2f}"
              f"{p:11.1e}  {verdict}")
        if passed:
            carries.append(c)
    for c, why in OVERRIDE.items():
        print(f"\n    OVERRIDE {CATEGORY[c]}: {why}")
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now says {[CATEGORY[c] for c in carries]} carry their own "
            f"geography, against CARRIES={[CATEGORY[c] for c in CARRIES]}. Which categories "
            "this country claims to place is what just moved; read the numbers above, then "
            "edit CARRIES deliberately. Do not move the bar.")
    return [c for c in codes if c in carries]


def held_out(df, pop, n_perm=20000, seed=0):
    """Test the department decode without touching the religion column."""
    print("\n  held-out check (nothing here touches the religion column):")
    share_survey = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    share_pop = pop / pop.sum()
    j = pd.concat([share_survey.rename("ecvmas"), share_pop.rename("codps")],
                  axis=1).dropna()
    if len(j) != len(share_pop):
        raise SystemExit(f"{len(share_pop) - len(j)} departments have population but no "
                         "ECVMAS respondents")
    r = np.corrcoef(j["ecvmas"], j["codps"])[0, 1]
    ratio = (j["ecvmas"] / j["codps"]).sort_values()
    print(f"    department share of people, ECVMAS 2012 (weighted, 10+) vs COD-PS 2024:  "
          f"r = {r:+.4f} over {len(j)}")
    print(f"      thinnest {ratio.index[0]} at {ratio.iloc[0]:.3f}x its COD-PS share, "
          f"fullest {ratio.index[-1]} at {ratio.iloc[-1]:.3f}x")
    rng = np.random.default_rng(seed)
    a, b = j["ecvmas"].to_numpy(), j["codps"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    print(f"      against {n_perm:,} random pairings: best random r = {perm.max():+.3f}, "
          f"and {beaten} reach the observed one")
    # Ten units is 3.6 million orderings against 20,000 draws, so a handful of ties is
    # expected even on a correct join. The bar is 1 in 200, not zero.
    if beaten > n_perm // 200:
        raise SystemExit(f"{beaten} of {n_perm} random pairings match or beat r={r:+.3f}; "
                         "the population check does not pin this join")


def lean_check(df, carries, nat):
    """§3.5 — does the story survive dropping the single most extreme department?

    Every superlative this country's note_public makes is about a spread, and a spread can
    be manufactured by one unit. Re-run the Catholic range and the Vodou range with the
    extreme department removed and print both.
    """
    print("\n  §3.5 lean check — drop the most extreme department and look again:")
    by = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    sh = by.div(by.sum(axis=1), axis=0)
    for code in (1, 9, 12):
        if code not in sh.columns:
            continue
        s = sh[code].sort_values()
        full = (s.iloc[-1] - s.iloc[0]) * 100
        trimmed_hi = (s.iloc[-2] - s.iloc[0]) * 100
        trimmed_lo = (s.iloc[-1] - s.iloc[1]) * 100
        print(f"    {CATEGORY[code]:<20} {s.index[0]} {s.iloc[0] * 100:5.1f}% to "
              f"{s.index[-1]} {s.iloc[-1] * 100:5.1f}%, a {full:.1f} point spread; "
              f"without the top {trimmed_hi:.1f}, without the bottom {trimmed_lo:.1f}")


def main():
    if "--fetch" in sys.argv:
        fetch()

    ind_path, ech_path = unpack()
    check_labels(ind_path)
    df = load(ind_path, ech_path)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_DEPARTMENTS:
        raise SystemExit(f"{len(lut)} departments in the lookup, expected {N_DEPARTMENTS} — "
                         "re-run sources/ht_geo.py")
    dept_to_unit = dict(zip(lut["ecvmas_dept"].astype(int), lut["unit"]))
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.Series(dict(zip(lut["geo_id"], lut["pop_2024"].astype(int))))
    df["geo_id"] = df["dept"].map(dept_to_unit)
    if df["geo_id"].isna().any():
        raise SystemExit("a DEPT code came out of the lookup with no pcode")

    held_out(df, pop)
    nat = national(df)
    cen = census_table()
    age_universe_check(cen)
    census_check(nat, cen)
    ecvh_check(df)
    carries = stability(df, nat)
    lean_check(df, carries, nat)

    # ---- shares x population ----
    small = [c for c in nat.index if c not in carries]
    by_unit = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    for c in CATEGORY:
        if c not in by_unit.columns:
            by_unit[c] = 0.0
    by_unit = by_unit[sorted(CATEGORY)]
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    small_total = float(sum(nat.get(c, 0.0) for c in small))
    carried = unit_share[carries].sum(axis=1)
    if (carried <= 0).any():
        raise SystemExit(f"departments whose stable categories sum to nothing: "
                         f"{sorted(carried[carried <= 0].index)}")
    measured_small = 1.0 - carried
    print(f"\n  the tail is set to its national {small_total:.2%} in every department; as "
          f"measured it ran {measured_small.min():.2%} in {names[measured_small.idxmin()]} "
          f"to {measured_small.max():.2%} in {names[measured_small.idxmax()]}")
    print(f"    the tail is {', '.join(CATEGORY[c] for c in small)}")

    units = sorted(lut["geo_id"])
    rows = []
    for unit in units:
        p = int(pop[unit])
        for c in sorted(CATEGORY):
            if c in carries:
                share = unit_share.loc[unit, c] / carried[unit] * (1.0 - small_total)
            else:
                share = float(nat.get(c, 0.0))
            rows.append((unit, CATEGORY[c], share * p))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")

    target = int(pop.sum())
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")

    n_by = df.groupby("geo_id").size()
    clu_by = df.groupby("geo_id")["clu"].nunique()
    out["geo_level"] = "departement"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2012"
    out["source_id"] = SOURCE_ID
    small_names = {CATEGORY[c] for c in small}
    out["note"] = [
        (f"IHSI ECVMAS 2012, own religion of people aged 10 and over, n={int(n_by[g]):,} in "
         f"{int(clu_by[g])} sampling clusters in this department; "
         + ("national share, this category having failed the split-half"
            if cat in small_names else "department share")
         + ", applied to the department's COD-PS 2024 population")
        for g, cat in zip(out["geo_id"], out["source_category"])]

    total = int(out["count"].sum())
    if total != target:
        raise SystemExit(f"drawn {total:,} against a target of {target:,}")
    if out["geo_id"].nunique() != N_DEPARTMENTS:
        raise SystemExit(f"{out['geo_id'].nunique()} departments drawn")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {N_DEPARTMENTS} departments)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(
        ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print("\n  by department, sorted by the Catholic share:")
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    order = show[CATEGORY[1]].sort_values().index
    head = ["Catholique", "Baptiste", "Pentecôtiste", "Autre protestant", "Aucune",
            "Adventiste", "Vaudou"]
    print("    " + f"{'department':<14}{'n':>7}" + "".join(f"{h[:9]:>10}" for h in head))
    for g in order:
        print(f"    {names[g]:<14}{int(n_by[g]):>7,}"
              + "".join(f"{show.loc[g, h]:10.1f}" for h in head))
    thin = n_by.idxmin()
    hw = 1.96 * np.sqrt(0.48 * 0.52 / int(n_by[thin]))
    print(f"\n    thinnest sample {names[thin]} at n={int(n_by[thin]):,} people "
          f"(±{hw * 100:.1f} points on a share near 48%, before the design effect), "
          f"fullest {names[n_by.idxmax()]} at n={int(n_by.max()):,}")


if __name__ == "__main__":
    main()
