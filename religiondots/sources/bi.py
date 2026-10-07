"""Burundi — the 2008 census's national religion table, given a 17-province geography by the
Afrobarometer and the census's own urban and rural split.

Reads data/raw/bi/rgph2008_chapitre1.xlsx (Tableau 1.13), data/raw/afrobarometer/*.sav,
data/geo/bi/bi_lookup.csv and bi_communes.csv; writes data/normalized/bi.csv. `sources/bi.md` is
the record; `sources/lr.py` and `sources/tg.py` are the construction this follows.

## BURUNDI ASKED, AND PUBLISHED RELIGION ONLY FOR THE NATION, URBAN AND RURAL

RGPH 2008 asked every household member's religion (P14, eight answers). ISTEEBU's tables cut it by
sex and by urban against rural, and never by province or commune: chapter 1 Tableau 1.13, the
thematic volume *État et structure de la population* Tableau 4.4 (a copy USAID hosted, read whole
2026-10-03), and the marriage and fertility volumes all stop at the nation (`sources/bi.md` §3-§4).

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

A province x urban/rural x religion table is fitted (`ipf3`) to three sets of census totals, all
EXACT and all over the same 8,053,574 residents:

    province x urban/rural      residents                     Tableau 1.5
    urban/rural x religion      the 9 religion rows           Tableau 1.13 (ordinary households)
                                plus collective households    Tableau 1.2 (43,653 / 45,843)
    province, collective only   collective households         Tableau 1.4

and summed over urban and rural. The survey supplies only the seed, so it decides how a province's
people divide between religions and nothing else. Every row is `modelled` (§7b: nobody counted the
cell). The 89,496 people in collective households, outside the religion table's universe, come out
at their exact Tableau 1.4 count per province and are an excluded column.

This is Liberia's and Togo's construction (`lr.ipf`, two margins) with the census's urban and rural
split as a third dimension, because the census measured it and it is large: Muslims are 14.3% of
urban Burundi and 1.3% of rural, Catholics 51.6% and 63.2%.

## THE SEED, AND WHY IT IS SPLIT BY THE SURVEY'S OWN URBAN CONTRAST

`SEED_FROM` pairs each census row with a survey group. A group that passes `cab.stability` at the
17 provinces over the two rounds and is at least 1% of respondents gives its row its own pattern
(`CARRIES`: Catholic, Protestant, Muslim). Every other row is seeded flat, so the fit gives it the
census's urban and rural shares on each province's urban and rural people and nothing else.

A carried row's province share is split between the province's town and countryside by the
SURVEY's own national urban and rural multiples for that group (Muslim 3.26x urban, 0.59x rural),
so the seed for each province still sums to the survey's share there; the fit then moves the
contrast to the census's (5.67x for Muslims). Seeding a carried row flat across urban and rural was
tried first and double counts: the survey's provincial Muslim shares already contain their towns,
and the fit, which has to find the census's 109,748 urban Muslims, put 95,581 of them in the Mairie
(20.0% of its people, against the survey's 12.6% there, n=175) and left the other towns at 4.9%.
Split by the survey's contrast, the Mairie is 14.4% Muslim and the other towns 14.1%, beside the
census's 14.3% for all of urban Burundi (`TOWN_BAND` asserts it; `python sources/bi.py
--flat-milieu` reruns the rejected seed and stops on it). The plain two-margin fit with no urban
dimension drew the Mairie at 9.3% and left about 21% for the other towns.

## `Christian only` IS SPREAD OVER THE NAMED CHURCHES OF ITS OWN UNIT

As Togo (`sources/tg.py`): a carried Christian group's share is divided by one minus its
province's unnamed share.

Usage:
    python sources/bi.py --fetch    nothing of its own; the Afrobarometer files are shared
                                    (`python sources/afrobarometer.py --fetch`), the census
                                    workbook through `python sources/bi_geo.py --fetch`
    python sources/bi.py            rebuild data/normalized/bi.csv
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, "tools"))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab
from bi_geo import CHAP1, RAW, fold
from cm import gkey, key
from lr import round_within_rows

LOOKUP = os.path.join(ROOT, "data", "geo", "bi", "bi_lookup.csv")
COMMUNES = os.path.join(ROOT, "data", "geo", "bi", "bi_communes.csv")
OUT = os.path.join(ROOT, "data", "normalized", "bi.csv")

COUNTRY = "Burundi"
ROUNDS = [5, 6]
SOURCE_ID = "bi_census2008_afrobarometer_2012_2014"
YEARS = "2008"
N_UNITS = 17
CENSUS_2008 = 8_053_574
ORDINARY_2008 = 7_964_078

# Tableau 1.13, ordinary households, both sexes: (urban, rural, total). Re-read from the workbook.
CENSUS_RELIGION = {
    "Aucune religion": (28_486, 462_612, 491_098),
    "Catholique": (396_334, 4_545_499, 4_941_833),
    "Protestante": (180_951, 1_541_088, 1_722_039),
    "Musulmane": (109_748, 90_761, 200_509),
    "Adventiste": (11_012, 174_349, 185_361),
    "Témoin de Jéhovah": (5_253, 20_201, 25_454),
    "Traditionnelle": (256, 2_491, 2_747),
    "Autre religion": (14_910, 246_171, 261_081),
    "ND": (21_263, 112_693, 133_956),
}
COLLECTIVE = "Ménages collectifs"       # not in the religion table; excluded, exact per province
COLLECTIVE_UR = (43_653, 45_843)        # Tableau 1.2: collective households, urban and rural

# Every Burundian answer over the two rounds -> the survey group, keyed through `cm.key`.
GROUP = {
    "roman catholic": "Catholic",
    "pentecostal": "Protestant", "anglican": "Protestant", "methodist": "Protestant",
    "baptist": "Protestant", "evangelical": "Protestant", "lutheran": "Protestant",
    "presbyterian": "Protestant", "church of christ": "Protestant", "dutch reformed": "Protestant",
    "seventh day adventist": "Adventist",
    "jehovah's witness": "Jehovah's Witness",
    "muslim only": "Muslim", "sunni only": "Muslim",
    "none": "None", "atheist": "None",
    "christian only": "Christian only",
    "other": "Other", "orthodox": "Other", "coptic": "Other", "independent": "Other",
}
GROUPS = ["Catholic", "Protestant", "Christian only", "Adventist", "Muslim", "Other",
          "Jehovah's Witness", "None"]
CHRISTIAN = ["Catholic", "Protestant", "Christian only", "Adventist", "Jehovah's Witness"]

# census row -> the survey group that gives it a pattern (None: the census urban/rural seed)
SEED_FROM = {
    "Catholique": "Catholic",
    "Protestante": "Protestant",
    "Musulmane": "Muslim",
    "Adventiste": "Adventist",
    "Témoin de Jéhovah": "Jehovah's Witness",
    "Aucune religion": "None",
    "Autre religion": "Other",
    "Traditionnelle": None,
    "ND": None,
}

# REGION label (through `cm.gkey`) -> unit. `Bujumbura` is Bujumbura Rural (R6's commune column
# puts its respondents in Mutimbuzi, Kabezi, Isale and the rest; asserted in `check_locations`).
NORM = {"bubanza": "BI01", "bujumbura": "BI02", "bururi": "BI03", "cankuzo": "BI04",
        "cankuza": "BI04", "cibitoke": "BI05", "gitega": "BI06", "karusi": "BI07",
        "karuzi": "BI07", "kayanza": "BI08", "kirundo": "BI09", "makamba": "BI10",
        "muramvya": "BI11", "muyinga": "BI12", "mwaro": "BI13", "ngozi": "BI14", "rutana": "BI15",
        "ruyigi": "BI16", "ruyiga": "BI16", "bujumburamairie": "BI17", "bujumburamarie": "BI17"}
# R6's commune spellings the census spells otherwise (folded), and the Mairie's quartiers.
COMMUNE_ALIAS = {"buhiga": "buhuga", "isare": "isale", "kanyosharural": "kanyosha",
                 "mpingakayove": "mpinga"}
MAIRIE_QUARTIERS = {"buterere", "buyenzi", "bwiza", "cibitoke", "gihosha", "kamenge", "kanyosha",
                    "kinama", "kinindo", "musaga", "ngagara", "nyakabiga", "rohero"}
# R6 (REGION unit, commune key) pairs that disagree, set from the first run's output.
EXPECTED_DISAGREE = {}

# What carries its own pattern, asserted against the split-half; set from its output 2026-10-03
# (R5 against R6, the one halving two rounds allow): Catholic +0.639, Protestant +0.659, Muslim
# +0.467 against nulls of +0.42-0.43. Jehovah's Witness passes too (+0.472, 18 answers) and is
# under the 1% floor; None passes the rank test and fails the chi-square.
CARRIES = ["Catholic", "Protestant", "Muslim"]
# `Christian only` is 0.0-1.7% of Christians in 12 provinces and 10-16% in Bujumbura Rural, Gitega,
# Mwaro and Karuzi; its order across provinces does not repeat between the rounds (+0.028), so it
# is the fieldwork, not a place. Togo's bar was 0.08. It is raised here because both carried
# Christian rows take the same factor in a province, so the Catholic/Protestant balance inside a
# province is untouched; the factor only says those people are Christians rather than members of
# the rows the census seeds. Measured 2026-10-03: range 15.9 points.
UNNAMED_RANGE_MAX = 0.17
SPLIT_SEED_BY_SURVEY = "--flat-milieu" not in sys.argv
# Muslim share of the Mairie and of the other towns, each as a multiple of urban Burundi's 14.3%.
# Built 2026-10-03: 14.4% and 14.1% (1.01x, 0.99x); the rejected flat seed gives 20.0% and 4.9%.
TOWN_BAND = (0.5, 1.5)


def check_census():
    """Tableau 1.13 re-read from the workbook must equal the transcription and close."""
    import openpyxl

    wb = openpyxl.load_workbook(os.path.join(RAW, CHAP1), data_only=True, read_only=True)
    rows = [r for r in wb["ETA13"].iter_rows(values_only=True) if any(c is not None for c in r)]
    if not str(rows[0][0]).startswith("Tableau 1.13"):
        raise SystemExit(f"ETA13 is not Tableau 1.13: {rows[0][0]!r}")
    got = {str(r[0]).strip(): (int(r[3]), int(r[6]), int(r[9])) for r in rows[3:]}
    total = got.pop("Total")
    if got != CENSUS_RELIGION:
        raise SystemExit(f"Tableau 1.13 differs from the transcription: "
                         f"{sorted(set(got.items()) ^ set(CENSUS_RELIGION.items()))}")
    sums = tuple(sum(v[i] for v in got.values()) for i in range(3))
    if sums != total or total[2] != ORDINARY_2008:
        raise SystemExit(f"Tableau 1.13's rows sum to {sums}, its Total row is {total}")
    if any(u + r != t for u, r, t in got.values()):
        raise SystemExit("an urban plus rural cell misses its total")
    print(f"  Tableau 1.13 re-read: 9 rows equal the transcription and sum to its Total row, "
          f"{ORDINARY_2008:,} people in ordinary households ({total[0]:,} urban)")
    return {k: v[2] for k, v in CENSUS_RELIGION.items()}


def check_locations(df, lut):
    """Round 6 carries the commune (`LOCATION.LEVEL.1`); it must sit in REGION's province."""
    c = pd.read_csv(COMMUNES, dtype=str)
    in_unit = {}
    for u, k in zip(c["unit"], c["commune"]):
        in_unit.setdefault(u, set()).add(k)
    in_unit["BI17"] = set(MAIRIE_QUARTIERS)
    sub = df[df["round"] == 6]
    loc = sub["LOCATION.LEVEL.1"].map(fold).map(lambda s: COMMUNE_ALIAS.get(s, s))
    known = set().union(*in_unit.values())
    unknown = sorted(set(loc) - known)
    if unknown:
        raise SystemExit(f"R6 communes the census does not have: {unknown}; add to COMMUNE_ALIAS")
    bad = [not (k in in_unit[u]) for u, k in zip(sub["geo_id"], loc)]
    seen = {}
    for (u, k), n in sub[bad].assign(k=loc[bad]).groupby(["geo_id", "k"]).size().items():
        seen[(u, k)] = int(n)
    print(f"\n  R6: {len(sub):,} respondents in {loc.nunique()} communes; {sum(bad)} sit outside "
          f"their REGION province: {seen or 'none'}")
    if seen != EXPECTED_DISAGREE:
        raise SystemExit(f"REGION and commune disagree differently from EXPECTED_DISAGREE: {seen}")


def unnamed_scale(df, nm):
    """1 / (1 - the unit's share of Christians answering `Christian only`); `sources/tg.py`'s."""
    c = df[df["category"].isin(CHRISTIAN)]
    tot = c.groupby("geo_id")["w"].sum()
    un = c[c["category"] == "Christian only"].groupby("geo_id")["w"].sum().reindex(tot.index, fill_value=0)
    frac = un / tot
    print("\n  `Christian only` as a share of Christians: national "
          f"{100 * un.sum() / tot.sum():.1f}%; " + ", ".join(f"{nm[u]} {100 * v:.1f}%" for u, v in frac.sort_values().items()))
    if frac.max() - frac.min() > UNNAMED_RANGE_MAX:
        raise SystemExit(f"the unnamed share ranges {100 * (frac.max() - frac.min()):.1f} points "
                         "across units; decide again")
    return 1.0 / (1.0 - frac)


def urban_witness(df):
    """Tableau 1.13's urban share per row against the survey's, each as a multiple of its source's
    overall urban share. Islam must be the most urban row in both."""
    urb = df["URBRUR"].astype(str).str.casefold().str.startswith("urban")
    s_all = float((df["w"] * urb).sum() / df["w"].sum())
    s = df.assign(u=urb).groupby("category").apply(lambda d: (d["w"] * d["u"]).sum() / d["w"].sum(),
                                                   include_groups=False) / s_all
    n = df.groupby("category").size()
    c_all = sum(v[0] for v in CENSUS_RELIGION.values()) / ORDINARY_2008
    c = pd.Series({k: v[0] / v[2] / c_all for k, v in CENSUS_RELIGION.items()})
    print("\n  urban share as a multiple of the whole (census rows from Tableau 1.13; survey pooled):")
    for row, grp in SEED_FROM.items():
        right = f"{grp:<20}{s[grp]:5.2f}x  n={int(n[grp])}" if grp else "(no survey group)"
        print(f"    {row:<20}{c[row]:5.2f}x    {right}")
    big = s[n[n >= 50].index]
    if c.idxmax() != "Musulmane" or big.idxmax() != "Muslim":
        raise SystemExit(f"the most urban census row is {c.idxmax()} and survey group "
                         f"{big.idxmax()}, not Musulmane and Muslim")
    print("    asserted: Islam is the most urban row in the census and in the survey")


def ipf3(seed, um, mr, u_coll, coll_col, tol=1e-9, max_iter=2000):
    """Fit `seed[unit, milieu, religion]` to three sets of census totals at once.

    `um[unit, milieu]`: residents by province, urban and rural (Tableau 1.5).
    `mr[milieu, religion]`: Tableau 1.13's rows by urban and rural, plus the collective households
        as one more column (Tableau 1.2: 43,653 urban, 45,843 rural).
    `u_coll[unit]`: each province's collective households (Tableau 1.4), the one column whose
        province totals are also known.
    All three are counts of the same 8,053,574 people, so the fit exists. A zero seed cell stays
    zero, and the Mairie's rural margin is zero."""
    import numpy as np

    m = seed.astype(float).copy()
    nu, nm_, nr = m.shape
    for i in range(max_iter):
        s = m.sum(axis=2)
        m *= np.where(s > 0, um / np.where(s > 0, s, 1), 0)[:, :, None]
        s = m.sum(axis=0)
        m *= np.where(s > 0, mr / np.where(s > 0, s, 1), 0)[None, :, :]
        s = m[:, :, coll_col].sum(axis=1)
        m[:, :, coll_col] *= np.where(s > 0, u_coll / np.where(s > 0, s, 1), 0)[:, None]
        err = max(np.abs(m.sum(axis=2) - um).max(), np.abs(m.sum(axis=0) - mr).max(),
                  np.abs(m[:, :, coll_col].sum(axis=1) - u_coll).max())
        if err < tol * um.sum():
            break
    else:
        raise SystemExit(f"the three-way fit did not converge in {max_iter} passes (worst {err:,.3f})")
    print(f"    three-way IPF converged in {i + 1} passes; worst margin error {err:.3g} people")
    return m


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    print("=== Burundi: the 2008 census's religion table on an Afrobarometer pattern ===")
    census = check_census()

    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True, extra=["LOCATION.LEVEL.1", "URBRUR"])
    raw["k"] = raw["category"].map(key)
    ct = pd.crosstab(raw["k"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print(f"\n  pooled: {len(raw):,} respondents; every answer as it arrives, keyed:")
    print(ct.sort_values("all", ascending=False).to_string())
    unmapped = sorted(set(raw["k"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no group: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = raw["k"].map(GROUP)
    ab.assert_one_wording(df, COUNTRY)

    tot = df.groupby("round")["w"].sum()
    byr = df.groupby(["category", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    print("\n  weighted share of all respondents by round (%):")
    print((100 * byr).round(1).reindex(GROUPS).to_string())

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run bi_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    lut = lut.set_index("geo_id")
    if int(lut["pop"].sum()) != CENSUS_2008 or int(lut["ordinary"].sum()) != ORDINARY_2008:
        raise SystemExit("bi_lookup.csv does not sum to the census")
    units = sorted(lut.index)
    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    if (per_round != N_UNITS).any():
        raise SystemExit(f"a pooled round does not sample all 17 units: {per_round.to_dict()}")
    if (df.groupby(["round", "geo_raw"])["geo_id"].nunique() > 1).any():
        raise SystemExit("one REGION label names two units in a round")
    check_locations(df, lut)
    share_s = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    share_p = lut["pop"] / lut["pop"].sum()
    print("\n  weighted share of respondents against the 2008 census share (printed, not asserted):")
    for u in units:
        print(f"    {nm[u]:<20}{100 * share_s[u]:6.1f}%{100 * share_p[u]:7.1f}%  "
              f"{share_s[u] / share_p[u]:5.2f}x")
    ab.held_out(df, lut["pop"], COUNTRY)

    # ---- the survey against the census, nationally ----
    nat = ab.national(df).reindex(GROUPS).fillna(0.0)
    shown = sum(v for k, v in census.items() if SEED_FROM[k])
    print("\n  national: survey group against the census row it seeds (% of those with a group):")
    for row, grp in SEED_FROM.items():
        if grp:
            cs = census[row] / shown
            print(f"    {row:<20}{100 * cs:6.2f}%   {grp:<20}{100 * nat[grp]:6.2f}%   {nat[grp] / cs:5.2f}x")
    print(f"    {'':<20}{'':>7}   {'Christian only':<20}{100 * nat['Christian only']:6.2f}%   (spread over its unit's churches)")
    urban_witness(df)

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, _table = cab.stability(dfw, GROUPS, units, f"{N_UNITS} provinces")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR and c != "Christian only"]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. Read the "
                         "table above, then edit CARRIES and the docstring deliberately.")

    # ---- seed, then fit to both census margins ----
    scale = unnamed_scale(df, nm)
    by = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    by = by.reindex(index=units, columns=GROUPS, fill_value=0.0)
    own = by.div(by.sum(axis=1), axis=0)
    cols = list(SEED_FROM)
    allc = cols + [COLLECTIVE]
    seed = np.ones((len(units), 2, len(allc)))
    basis = {}
    # the survey's own national urban and rural contrast per group, as a multiple of its share
    urb = df["URBRUR"].astype(str).str.casefold().str.startswith("urban")
    sh_u = df[urb].groupby("category")["w"].sum() / df.loc[urb, "w"].sum()
    sh_r = df[~urb].groupby("category")["w"].sum() / df.loc[~urb, "w"].sum()
    sh = df.groupby("category")["w"].sum() / df["w"].sum()
    k_u, k_r = (sh_u / sh).reindex(GROUPS).fillna(1.0), (sh_r / sh).reindex(GROUPS).fillna(1.0)
    us = (lut["urban"] / lut["pop"]).reindex(units).to_numpy()
    print("\n  the survey's urban and rural multiples for the carried groups: " + ", ".join(
        f"{g} {k_u[g]:.2f}/{k_r[g]:.2f}" for g in carries))
    for j, row in enumerate(cols):
        grp = SEED_FROM[row]
        if grp in carries:
            s = (own[grp] * (scale if grp in CHRISTIAN else 1.0)).reindex(units).to_numpy()
            if SPLIT_SEED_BY_SURVEY:
                d = us * k_u[grp] + (1 - us) * k_r[grp]
                seed[:, 0, j] = s * k_u[grp] / d
                seed[:, 1, j] = s * k_r[grp] / d
            else:
                seed[:, :, j] = s[:, None]
            basis[row] = f"the province's own pattern from the survey's {grp} answers"
        else:
            basis[row] = ("not placed by the survey; the census's urban and rural shares on the "
                          "province's urban and rural people")
    if not SPLIT_SEED_BY_SURVEY:
        print("  --flat-milieu: the rejected seed (carried rows flat across urban and rural)")
    basis[COLLECTIVE] = "people in collective households, Tableau 1.4; outside the religion table"
    zero = [(nm[u], r) for i, u in enumerate(units) for j, r in enumerate(allc) if seed[i, 0, j] <= 0]
    print(f"\n  seed cells at zero (they stay zero): {zero or 'none'}")
    um = lut[["urban", "rural"]].reindex(units).to_numpy(dtype=float)
    mr = np.array([[CENSUS_RELIGION[r][0] for r in cols] + [COLLECTIVE_UR[0]],
                   [CENSUS_RELIGION[r][1] for r in cols] + [COLLECTIVE_UR[1]]], dtype=float)
    u_coll = lut["collective"].reindex(units).to_numpy(dtype=float)
    print("  fitting province x urban/rural x religion to the census's three tables:")
    fit = ipf3(seed, um, mr, u_coll, len(cols))
    fitted = pd.DataFrame(fit.sum(axis=1), index=units, columns=allc)
    ordinary = lut["ordinary"].reindex(units)
    if (abs(fitted[cols].sum(axis=1) - ordinary) > 0.01).any():
        raise SystemExit("the fit's religion rows do not sum to each province's ordinary households")
    counts = round_within_rows(fitted[cols])
    if not (counts.sum(axis=1) == ordinary).all():
        raise SystemExit("a unit's drawn total is not its census population in ordinary households")
    drift = counts.sum(axis=0) - pd.Series(census).reindex(counts.columns)
    print("    rounding drift against the census rows: "
          + (", ".join(f"{c} {int(d):+d}" for c, d in drift.items() if d) or "none"))
    if drift.abs().max() > N_UNITS:
        raise SystemExit(f"rounding drift {drift.to_dict()} exceeds one person per unit")
    counts[COLLECTIVE] = lut["collective"].reindex(units).astype("int64")
    # the urban Muslims the fit puts in each province, against the census's national figure
    mus, mi = cols.index("Musulmane"), units.index("BI17")
    m_mairie = fit[mi, 0, mus] / fit[mi, 0, :len(cols)].sum()
    rest_pop = fit[:, 0, :len(cols)].sum() - fit[mi, 0, :len(cols)].sum()
    m_rest = (fit[:, 0, mus].sum() - fit[mi, 0, mus]) / rest_pop
    m_all = CENSUS_RELIGION["Musulmane"][0] / sum(v[0] for v in CENSUS_RELIGION.values())
    print(f"    Muslim share of the Mairie {100 * m_mairie:.1f}% and of the other towns "
          f"{100 * m_rest:.1f}%, against {100 * m_all:.1f}% of all urban Burundi (Tableau 1.13)")
    if not all(TOWN_BAND[0] * m_all <= x <= TOWN_BAND[1] * m_all for x in (m_mairie, m_rest)):
        raise SystemExit("the fit has put the census's urban Muslims mostly in one kind of town; "
                         "read the docstring's seed section")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("2008 census province population by urban and rural and national religion rows "
                   "by urban and rural, fitted with the pattern of Afrobarometer rounds 5-6 pooled "
                   f"(n={int(n_by[r.geo_id])} here); {basis[r.source_category]}"), axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2008:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2008:,}")
    keep = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    if not SPLIT_SEED_BY_SURVEY:
        print("\n  --flat-milieu: not writing; this seed is the rejected one")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[keep].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, {out['source_category'].nunique()} "
          f"categories, {out['geo_id'].nunique()} units)")

    # ---- what the file says, for note_public ----
    stated = [c for c in cols if c != "ND"]
    st = counts[stated].sum(axis=1)
    show = ["Catholique", "Protestante", "Musulmane", "Adventiste", "Aucune religion", "Autre religion",
            "Témoin de Jéhovah"]
    print("\n  as drawn, share of people with a stated religion (%), most Muslim first; survey's own "
          "share of all respondents in brackets:")
    print(f"    {'':<20}" + "".join(f"{r[:10]:>18}" for r in show))
    for u in (counts["Musulmane"] / st).sort_values(ascending=False).index:
        cells = []
        for r in show:
            g = SEED_FROM[r]
            cells.append(f"{100 * counts.loc[u, r] / st[u]:6.1f}({100 * own.loc[u, g]:5.1f})    ")
        print(f"    {nm[u]:<20}" + "".join(f"{c:>18}" for c in cells) + f"  n={int(n_by[u])}")
    print("  national, stated religion only: " + ", ".join(
        f"{r} {100 * counts[r].sum() / st.sum():.2f}" for r in show))


if __name__ == "__main__":
    main()
