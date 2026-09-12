"""Uruguay — religion by departamento, from INE's own Encuesta Nacional de Hogares Ampliada.

Reads data/raw/uy/P_2006_TERCEROS.sav and writes data/normalized/uy.csv.
`sources/uy.md` is this country's record and `sources.md` §9ce is the write-up.

## URUGUAY'S CENSUS HAS NOT ASKED ABOUT RELIGION SINCE 1908, AND ITS HOUSEHOLD SURVEY HAS

The queue priced this country from the AmericasBarometer, 4,318 respondents pooled over three
waves. **It is not the best source available and it is not even close.** INE put the religion
question on the *Encuesta Nacional de Hogares Ampliada 2006* — the year the continuous
household survey was expanded to cover the whole country including rural areas — and asked it
of every household member over six. That is **230,898 people with an answer**, fifty-three
times LAPOP's pool, on the same nineteen departments, collected by the national statistics
office rather than by a political-attitudes survey.

The census half of the question is genuinely closed and this file says so rather than
implying it: Uruguay is ABSENT from the UNSD oracle, and the last Uruguayan census to ask
about religion was **1908**, nine years before the 1917 constitution separated church and
state. The 2011 and 2023 censuses do not carry the question. So the ENHA is not a survey
standing in front of a census that exists; it is the only thing there is, and it happens to
be very good.

    variable   e29_1, `Religion`, file p2006
    question   PARA MAYORES DE 6 AÑOS ¿como se definiria desde el punto de vista religioso?
    universe   persons over 6 in private households, national territory
    answers    Catolico, Cristiano no catolico, Judio, Umbandista/afroamericano,
               Creyente sin confesion, Ateo/agnostico, Otra

## THE UNIVERSE IS AGES 7 AND OVER, WHICH THE METADATA GETS WRONG

The ANDA data dictionary says *"Personas mayores de 6 años de edad"* and then glosses it as
*"solo se preguntará a personas de 6 años y más"*, which are two different universes. **The
data says the questionnaire is right and the gloss is wrong**: the not-applicable code 0
covers ages 0 to 6 exactly and completely — all 3,136 zero-year-olds through all 4,225
six-year-olds — and nobody aged 7 or over carries it. Asserted below, because the difference
is 4,225 respondents and a percentage point of the country's population base.

So 8.99% of Uruguay is not drawn, and it is children rather than a non-response cell. There
is no non-response cell at all: every person in the universe has one of the seven answers.

## WHAT THE COUNTRY LOOKS LIKE, AND WHY IT IS THE STANDOUT IN THE AMERICAS

    45.97%  Catolico
    26.87%  Creyente sin confesion
    15.72%  Ateo/agnostico
    10.07%  Cristiano no catolico
     0.64%  Umbandista/afroamericano
     0.41%  Judio
     0.31%  Otra

Those are the SURVEY's own weighted shares, which is what this
module measures. As drawn they move by up to a quarter of a point, because each
department carries its 2023 population rather than its 2006 one; anything quoted to
a reader has to come off `data/normalized/uy.csv` instead, where Catholic is 46.19%
and the two no-religion cells are 42.47% together.

**42.6% of Uruguayans over six claim no religious affiliation**, and two thirds of those still
say they believe in God. Nothing else this map has drawn in the Americas is close. It is also
not evenly spread: the unaffiliated share runs from **20.5% of Paysandú to 58.1% of Rocha**,
a spread of thirty-eight points across nineteen departments of one small country.

## TWO INDEPENDENT CHECKS, AND ONE OF THEM IS A SECOND SURVEY

`held_out()` is the population check every survey-built country here runs, and it is written
in this file rather than borrowed from `sources/lapop.py` for one reason: that module's
printout says the word LAPOP, and a line of output naming the wrong instrument is a false
provenance on screen (the argument `sources/ec.py` made about `pop_source`). Nothing else
about it differs.

`cross_check()` is the one Uruguay can do that no LAPOP country can. **The AmericasBarometer
also measured Uruguay, four to eight years later, with a different instrument, a different
sample and a different sponsor**, so its departmental ordering is an outside witness on this
one. It agrees strongly where the two cards mean the same thing:

    non-Catholic Christian    r = +0.86, and none of 20,000 random pairings reaches it
    atheist / agnostic        r = +0.73, 3 of 20,000
    no religion, both cells   r = +0.56, 118 of 20,000
    Catholic                  r = +0.34, 1,574 of 20,000 — NOT significant, see below

**The Catholic row failing is a fact about the pair of instruments and not about either
one.** LAPOP's card offers *Ninguna (cree en un Ser Superior pero no pertenece a ninguna
religión)* and INE's offers *Creyente sin confesión*, which are the same idea worded
differently, and the boundary between that answer and *Católico* is exactly where wording
moves people: LAPOP finds 37.0% Catholic and 32.4% in that cell where INE finds 46.0% and
26.9%. On top of that LAPOP has 45 respondents in Rivera against INE's 9,322, so most of the
scatter in the Catholic comparison is LAPOP's sampling error and not Uruguay's geography.
The two agree about which departments are Protestant and which are atheist, which is the part
where neither card is ambiguous.

**Reviewed 2026-09-08 and one sentence above is wrong; `sources/uy.md` §11.2 has the working.**
The boundary explanation holds and now has two direct measurements: the two cells either side
of the line agree badly alone (Catholic +0.34, `Creyente sin confesión` against LAPOP's cell
+0.25) and well summed (+0.69, 12 of 20,000), and their per-department errors run r = -0.84.
But **the sampling-error clause does not hold** — correcting each pair for LAPOP's own binomial
noise moves the two failing pairs LEAST (+0.34 to +0.46 and +0.25 to +0.32, against +0.86 to
+0.93 and +0.73 to +0.79), because Catholic has the largest between-department spread of the
four. Noise cannot be why one cell uniquely fails. The weak cell is `Ninguna (cree en un Ser
Superior)`, not Católico, which is the half that matters for `gt`, `sv` and `ec`.

## THE SPLIT-HALF RUNS ACROSS THE YEAR, AND FIVE OF THE SEVEN PASS OUTRIGHT

§14.16's test needs the country ranked twice. A single-year survey has no waves, but the ENHA
is a monthly panel with its own semester weights, so the two halves of 2006 are two samples
of the same country, 115,718 respondents against 115,180.

    Catolico                  45.97%   +0.92
    Creyente sin confesion    26.87%   +0.88
    Ateo/agnostico            15.72%   +0.88
    Cristiano no catolico     10.07%   +0.85
    Umbandista/afroamericano   0.64%   +0.78
    ------------------------------------------------- bar +0.46 on 19 departments
    Judio                      0.41%   +0.26   drawn anyway, see UNDER_BAR
    Otra                       0.31%   +0.26   drawn anyway, see UNDER_BAR

**The two that fail are the two that are near zero in eighteen of nineteen departments**, and
what the rank test measures for those is which department happened to catch one respondent in
a half-year. It is not measuring the thing the map draws. `Judio` is 0.92% of Montevideo
against 0.06% of the rest of the country, on 80,196 Montevideo respondents, and the two halves
of 2006 agree on the SHARES at +0.93 while disagreeing on the ranking of the near-zeros.
Drawing that cell at the national rate would put two thirds of Uruguay's Jews outside
Montevideo, which is a stronger claim than the one the rank test declines to license and a
false one. Both exceptions are named in `UNDER_BAR` with their reasons and their chi-squares,
the bar is not moved, and everything else clears it by a distance.

So **nothing here is drawn at a national rate**; every department carries its own measured
composition, which is not true of any of the three AmericasBarometer countries.

Usage:
    python sources/uy.py --fetch    check the ENHA microdata is unpacked, and say where from
    python sources/uy.py            rebuild data/normalized/uy.csv
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uy")
SAV = os.path.join(RAW, "P_2006_TERCEROS.sav")
RAR = os.path.join(RAW, "2006_SAV.rar")
POP = os.path.join(ROOT, "data", "geo", "uy", "uy_pop_2023.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "uy", "uy_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "uy.csv")

SOURCE_ID = "uy_ine_enha_2006"
N_DEPARTMENTS = 19

# The value labels of `e29_1`, verbatim from the .sav's own label set. INE writes them
# without accents there; spec §2.4 keeps the source's own words, so these strings are what
# reaches `source_category` and what `taxonomy/uy2006.py` keys on. Asserted against the file
# on every run, so a re-labelled release fails here rather than dropping a category silently.
CATEGORY = {
    1: "Catolico",
    2: "Cristiano no catolico",
    3: "Judio",
    4: "Umbandista/afroamericano",
    5: "Creyente sin confesion",
    6: "Ateo/agnostico",
    7: "Otra",
}

# Code 0 is the not-applicable, and it is the children. Asserted as a closed age range in
# both directions: nobody 7 or over carries it, and nobody 0 to 6 carries anything else.
NOT_ASKED = 0
NOT_ASKED_MAX_AGE = 6

# The split-half bar, 1.96/sqrt(n-1) on 19 units (§14.16).
STABILITY_BAR = 1.96 / np.sqrt(N_DEPARTMENTS - 1)

# THE TWO CATEGORIES DRAWN ON THEIR OWN SHARES WITH A RANK CORRELATION UNDER THE BAR, each
# with the reason, printed on every run so it cannot become invisible. This is `sources/gt.py`'s
# `OVERRIDE` in the shape AGENT_BRIEF §2 allows an agent to use: the bar is not moved, the
# exception is named, and the argument is here to be disagreed with.
#
# **The precondition is the chi-square, not the size** (`lapop.stability`'s docstring): a
# category whose departments do NOT differ significantly has nothing to draw and gets no
# exception. Both of these differ at p < 1e-12 on 230,898 respondents.
UNDER_BAR = {
    3: "Judio: the rank test is measuring eighteen departments whose true share is near "
       "zero, and their ordering is which of them happened to catch one respondent in a "
       "half-year. What it is NOT measuring is the fact the map draws, which is that "
       "Montevideo is 0.92% Jewish against 0.06% across the rest of the country: that share "
       "rests on 80,196 respondents and a 95% half-width of 0.07 points, and the two halves "
       "of 2006 agree on the SHARES at +0.93 even as they disagree on the ranking of the "
       "near-zeros. Uruguay's Jewish community is a Montevideo community; drawing this cell "
       "at the national rate would move two thirds of it out of the city, which is a "
       "stronger claim than the one the rank test declines to license, and a false one.",
    7: "Otra: 0.31% of the country, ranging 0.04% in Tacuarembo to 0.60% in Canelones. The "
       "chi-square says the departments differ and the split-half rank does not replicate "
       "(+0.26, shares +0.45). Drawn as measured, with the caveat that belongs in "
       "`taxonomy/uy2006.py`: this box collects both real minority religions and the "
       "interviewer's give-up answer (`NO SABE`, `SIN DEFINICION` and `NO SE DEFINE` are "
       "together a fifth of its free text), so how often it is reached for is partly a "
       "property of the fieldwork team and those are regional. At three people in a "
       "thousand the choice between measured and national moves nothing a reader can see; "
       "it is written down because it is a real doubt and not because it matters here.",
}


def fetch():
    """The .sav is inside a RAR from INE's ANDA catalogue; this only says where.

    `www4.ine.gub.uy/Anda5/index.php/catalog/48` -> *Obtener Microdatos* -> the terms page
    -> `Aceptar`, which is a POST of `accept=Aceptar` with a session cookie and no login,
    then `.../catalog/48/download/1166` is `2006_SAV.rar` (31.8 MB). It unpacks to four
    files; only `P_2006_TERCEROS.sav` (234 MB, the person file) is needed.

    **INE's terms are not a bar to what this map does and are worth reading anyway**: no
    redistribution without written consent, scientific and statistical use only, presentation
    of AGGREGATED information only, no re-identification, and a request that publications
    based on the data be sent to INE. Nothing here redistributes the microdata (`data/` is
    gitignored) and the output is a departmental aggregate. `sources/uy.md` §2 carries the
    text and the deposit request, which is the one clause somebody has to act on if this map
    is ever published as a paper.

    `bsdtar -xf 2006_SAV.rar P_2006_TERCEROS.sav` unpacks it; libarchive reads RAR3 and there
    is no unrar on this machine.
    """
    if os.path.exists(SAV):
        print(f"  have {SAV} ({os.path.getsize(SAV):,} bytes)")
        return
    raise SystemExit(
        f"{SAV} missing.\n"
        f"  1. www4.ine.gub.uy/Anda5/index.php/catalog/48 -> Obtener Microdatos -> Aceptar\n"
        f"  2. download/1166 is 2006_SAV.rar -> {RAR}\n"
        f"  3. bsdtar -xf 2006_SAV.rar P_2006_TERCEROS.sav\n"
        "  See this module's `fetch` docstring and sources/uy.md for the terms.")


def load():
    """The five columns this build needs out of 527, with the universe asserted."""
    import pyreadstat

    if not os.path.exists(SAV):
        raise SystemExit(f"{SAV} missing — run with --fetch for the walk-through")
    df, meta = pyreadstat.read_sav(
        SAV, usecols=["dpto", "e29_1", "e29_2", "pesoano", "e27", "mes"])
    df = df.rename(columns={"e27": "age", "e29_1": "code", "e29_2": "other_text",
                            "pesoano": "w"})

    labels = meta.value_labels.get(meta.variable_to_label.get("e29_1"), {})
    got = {int(k): v for k, v in labels.items() if int(k) in CATEGORY}
    if got != CATEGORY:
        raise SystemExit(f"the e29_1 value labels have changed: {got} against {CATEGORY} — "
                         "the answer card is not the one this file was written against")

    seen = sorted(int(c) for c in df["code"].dropna().unique())
    if seen != [NOT_ASKED] + sorted(CATEGORY):
        raise SystemExit(f"e29_1 holds {seen}, expected {[NOT_ASKED] + sorted(CATEGORY)}")
    if df["code"].isna().any():
        raise SystemExit(f"{int(df['code'].isna().sum())} rows have no e29_1 at all — this "
                         "file assumes the question has no missing cell")

    # THE UNIVERSE, BOTH WAYS ROUND. The data dictionary and the questionnaire disagree about
    # whether six-year-olds were asked; this settles it and stops the build if it changes.
    kids = df[df["code"] == NOT_ASKED]
    if kids["age"].max() != NOT_ASKED_MAX_AGE or kids["age"].min() != 0:
        raise SystemExit(f"the not-asked code covers ages {kids['age'].min()} to "
                         f"{kids['age'].max()}, not 0 to {NOT_ASKED_MAX_AGE}")
    stray = int((df["age"] <= NOT_ASKED_MAX_AGE).sum()) - len(kids)
    if stray:
        raise SystemExit(f"{stray} people aged {NOT_ASKED_MAX_AGE} or under carry a religion "
                         "answer — the universe is not a clean age cut")
    print(f"Uruguay: {len(df):,} people in the ENHA 2006 person file, of whom {len(kids):,} "
          f"are aged 0 to {NOT_ASKED_MAX_AGE} and were not asked")

    df = df[df["code"] != NOT_ASKED].copy()
    df["code"] = df["code"].astype(int)
    df["dpto"] = df["dpto"].astype(int)
    bad = sorted(set(df["dpto"]) - set(range(1, N_DEPARTMENTS + 1)))
    if bad:
        raise SystemExit(f"department codes outside 1-{N_DEPARTMENTS}: {bad}")
    print(f"  {len(df):,} respondents over {NOT_ASKED_MAX_AGE} with an answer, "
          f"{df['dpto'].nunique()} of {N_DEPARTMENTS} departments, "
          f"weighting to {df['w'].sum():,.0f} people in 2006")
    return df


def national(df):
    return df.groupby("code")["w"].sum() / df["w"].sum()


def held_out(df, pop, unit_col="geo_id", n_perm=20000, seed=0):
    """Test the department decode without touching the religion column.

    The ENHA's own weighted population per department against INE's 2006 estimate, ranked
    against 20,000 shuffles of the unit labels. **The comparison year is 2006 and not 2023**:
    Uruguay's people have moved since (Montevideo is down 64,657 and Canelones up 96,535)
    so a 2023 population would test the survey against seventeen years of internal migration
    rather than against the join.

    19! is 1.2e17 against 20,000 draws, so `sources/lapop.py`'s small-country failure mode
    (its docstring puts it at roughly seven to ten units) is not in reach here.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_survey = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_pop = pop["pop_2006"] / pop["pop_2006"].sum()
    j = pd.concat([share_survey.rename("enha"), share_pop.rename("ine")], axis=1).dropna()
    if len(j) != len(share_pop):
        raise SystemExit(f"{len(share_pop) - len(j)} departments have population but no "
                         "ENHA respondents")
    r = np.corrcoef(j["enha"], j["ine"])[0, 1]
    ratio = (j["enha"] / j["ine"]).sort_values()
    print(f"    department share of people, ENHA 2006 vs INE 2006:  r = {r:+.4f} "
          f"over {len(j)}")
    print(f"      thinnest {ratio.index[0]} at {ratio.iloc[0]:.3f}x its population share, "
          f"fullest {ratio.index[-1]} at {ratio.iloc[-1]:.3f}x")

    rng = np.random.default_rng(seed)
    a, b = j["enha"].to_numpy(), j["ine"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    print(f"      against {n_perm:,} random pairings: best random r = {perm.max():+.3f}, "
          f"and {beaten} reach the observed one")
    if beaten:
        raise SystemExit(f"{beaten} of {n_perm} random pairings match or beat r={r:+.3f}; "
                         "the population check does not pin this join")


def stability(df, nat, names, unit_col="geo_id"):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — §14.16, split across the survey year.

    A one-year survey has no waves to split, so the split is January-June against July-
    December. That is the same design twice, unlike a split across LAPOP rounds, so it tests
    sampling stability rather than instrument drift; with a hundred and fifteen thousand
    respondents a half it is a strong test of the first and no test of the second.

    Five of the seven clear the bar outright. The two that do not are `Judio` and `Otra`,
    both under half a percent of the country, and both are in `UNDER_BAR` with the argument.
    The chi-square beside each is the precondition for that: it asks whether the departments
    differ AT ALL, which is the question the ranking cannot answer for a category that is
    near zero in eighteen of nineteen units.
    """
    from scipy import stats

    early = df[df["mes"] <= 6]
    late = df[df["mes"] > 6]
    print(f"\n  split-half across 2006 (§14.16), bar = +{STABILITY_BAR:.2f} on "
          f"{N_DEPARTMENTS} departments:")
    print(f"    Jan-Jun n={len(early):,}   Jul-Dec n={len(late):,}")
    print(f"    {'category':<28}{'national':>10}{'rank':>7}{'shares':>8}"
          f"{'chi-sq p':>11}  verdict")

    n_unit = df.groupby(unit_col).size()
    surprises = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        def share(d):
            return d.groupby(unit_col).apply(
                lambda x: x.loc[x["code"] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        sp = j["e"].corr(j["l"], method="spearman")
        pe = j["e"].corr(j["l"])
        hit = df[df["code"] == c].groupby(unit_col).size().reindex(n_unit.index).fillna(0)
        p = stats.chi2_contingency(np.vstack([hit, n_unit - hit]))[1]
        passed = bool(np.isfinite(sp)) and sp >= STABILITY_BAR
        if passed:
            verdict = "own geography"
        elif c in UNDER_BAR:
            verdict = "own geography — UNDER THE BAR, see UNDER_BAR"
        else:
            verdict = "NOT distinguishable from zero"
        print(f"    {CATEGORY[c]:<28}{nat[c] * 100:9.2f}%{sp:+7.2f}{pe:+8.2f}"
              f"{p:11.1e}  {verdict}")
        if not passed and c in UNDER_BAR:
            if p > 1e-12:
                raise SystemExit(
                    f"{CATEGORY[c]} is in UNDER_BAR but its departments now differ at only "
                    f"p={p:.1e}. The exception is not available without the chi-square; "
                    "draw it at the national rate or find out what changed.")
            print(f"        reason: {UNDER_BAR[c]}")
        if passed != (c not in UNDER_BAR):
            surprises.append((CATEGORY[c], passed))
    if surprises:
        raise SystemExit(
            f"the split-half has changed its mind about {surprises}. Which categories this "
            "country claims to place is what just moved; read the numbers above, then edit "
            "UNDER_BAR deliberately. Do not move the bar.")
    print(f"    -> all {len(nat)} categories carry their own department shares, "
          f"{len(UNDER_BAR)} of them by the exception above")


def cross_check(df, names, n_perm=20000, seed=0):
    """The AmericasBarometer as an outside witness on the same nineteen departments.

    This is the check no country built from LAPOP alone can run, and it is the reason
    Uruguay is drawn from INE rather than from LAPOP: there are two independent measurements
    of the same thing and they can be put side by side.

    Reported, never asserted, and the module docstring says why the Catholic pair is weak. It
    imports `sources/lapop.py` read-only and changes nothing about how any other country is
    drawn.
    """
    import lapop

    if not os.path.exists(lapop.SLIM):
        print("\n  cross-check against the AmericasBarometer SKIPPED — "
              f"{lapop.SLIM} is not on disk")
        return
    lp = lapop.load(14, [2010, 2012, 2014])
    lp["dpto"] = lp["prov_code"] - 1400
    ltot = lp.groupby("dpto")["w"].sum()
    etot = df.groupby("dpto")["w"].sum()

    pairs = [
        ("Catolico", [1], [1]),
        ("Cristiano no catolico", [2], [2, 5]),
        ("Ateo/agnostico", [6], [11]),
        ("no religion, both cells", [5, 6], [4, 11]),
    ]
    print(f"\n  cross-check: LAPOP 2010-2014 (n={len(lp):,}) on the same {N_DEPARTMENTS} "
          "departments, an independent instrument four to eight years later")
    rng = np.random.default_rng(seed)
    for label, ec, lc in pairs:
        a = df[df["code"].isin(ec)].groupby("dpto")["w"].sum().reindex(etot.index).fillna(0) / etot
        b = lp[lp["code"].isin(lc)].groupby("dpto")["w"].sum().reindex(ltot.index).fillna(0) / ltot
        j = pd.concat([a.rename("ine"), b.rename("lapop")], axis=1).dropna()
        r = np.corrcoef(j["ine"], j["lapop"])[0, 1]
        x, y = j["ine"].to_numpy(), j["lapop"].to_numpy()
        perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
        print(f"    {label:<26} r = {r:+.2f}   {int((perm >= r).sum()):>5} of {n_perm:,} "
              f"random pairings reach it")
        print(f"      INE {a.min() * 100:5.1f}-{a.max() * 100:5.1f}%   "
              f"LAPOP {b.min() * 100:5.1f}-{b.max() * 100:5.1f}%")


def report_other(df):
    """What is inside `Otra`, from the free-text follow-up `e29_2`.

    0.31% of the country and not split — see `taxonomy/uy2006.py` — but the ENHA is the only
    source on this map that can say what is in an `other` cell at all, so it is printed and
    written into `branches.py`'s note for `other.uy` rather than left as a shrug.
    """
    o = df[df["code"] == 7].copy()
    o["t"] = o["other_text"].fillna("").astype(str).str.strip().str.upper()
    w = o.groupby("t")["w"].sum().sort_values(ascending=False)
    tot = w.sum()
    print(f"\n  inside `Otra` ({len(o):,} respondents, {tot:,.0f} people weighted), the "
          "twelve commonest free-text answers:")
    for t, v in w.head(12).items():
        print(f"      {v / tot:6.1%}  {t if t else '(blank)'}")


def main():
    if "--fetch" in sys.argv:
        fetch()

    df = load()

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_DEPARTMENTS:
        raise SystemExit(f"{len(lut)} departments in the lookup, expected {N_DEPARTMENTS} — "
                         "re-run sources/uy_geo.py")
    dpto_to_unit = dict(zip(lut["ine_dpto"].astype(int), lut["unit"]))
    names = dict(zip(lut["geo_id"], lut["name"]))
    df["geo_id"] = df["dpto"].map(dpto_to_unit)
    if df["geo_id"].isna().any():
        raise SystemExit("a department code came out of the lookup with no pcode")

    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = pop["geo_id"].astype(str).str.strip()
    pop = pop.set_index("geo_id")
    if sorted(pop.index) != sorted(lut["geo_id"]):
        raise SystemExit("the population file and the lookup cover different departments")
    if df["geo_id"].nunique() != N_DEPARTMENTS:
        raise SystemExit(f"{df['geo_id'].nunique()} departments have respondents, expected "
                         f"{N_DEPARTMENTS}")

    held_out(df, pop)

    nat = national(df)
    stability(df, nat, names)
    cross_check(df, names)
    report_other(df)

    # ---- shares x population. Every department carries its own measured composition, so
    # there is no tail to spread and no national rate anywhere in this country.
    by_unit = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    for c in CATEGORY:
        if c not in by_unit.columns:
            raise SystemExit(f"{CATEGORY[c]} is absent from every department")
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    units = sorted(lut["geo_id"])
    rows = []
    for unit in units:
        p = int(round(pop.loc[unit, "pop7_2023"]))
        for c in sorted(CATEGORY):
            rows.append((unit, CATEGORY[c], unit_share.loc[unit, c] * p))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")

    target = int(sum(int(round(pop.loc[u, "pop7_2023"])) for u in units))
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"\n  rounding drift {drift:+d} people, absorbed into the largest cell")

    n_by = df.groupby("geo_id").size()
    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2006"
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda g: (f"INE Encuesta Nacional de Hogares Ampliada 2006, n={int(n_by[g]):,} in "
                   "this department; department share applied to INE's estimated population "
                   "aged 7 and over in 2023"))

    total = int(out["count"].sum())
    if total != target:
        raise SystemExit(f"drawn {total:,} against a target of {target:,}")
    if out["geo_id"].nunique() != N_DEPARTMENTS:
        raise SystemExit(f"{out['geo_id'].nunique()} departments drawn")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {N_DEPARTMENTS} departments)")
    under7 = int(round(pop["pop_2023"].sum())) - target
    print(f"  {under7:,} children under 7 are NOT in this file and belong in countries.py's "
          f"`gap=` ({under7 / pop['pop_2023'].sum():.2%} of Uruguay)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print("\n  by department, sorted by the unaffiliated share:")
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    none = show["Creyente sin confesion"] + show["Ateo/agnostico"]
    head = f"{'n':>8}{'Cath':>8}{'nonCath':>9}{'believer':>10}{'atheist':>9}{'none':>8}"
    print(f"    {'department':<18}{head}")
    for g in none.sort_values().index:
        print(f"    {names[g]:<18}{int(n_by[g]):>8,}{show.loc[g, 'Catolico']:8.1f}"
              f"{show.loc[g, 'Cristiano no catolico']:9.1f}"
              f"{show.loc[g, 'Creyente sin confesion']:10.1f}"
              f"{show.loc[g, 'Ateo/agnostico']:9.1f}{none[g]:8.1f}")
    thin = n_by.idxmin()
    hw = 1.96 * np.sqrt(0.46 * 0.54 / int(n_by[thin]))
    print(f"\n    thinnest sample {names[thin]} at n={int(n_by[thin]):,} "
          f"(±{hw * 100:.1f} points on a share near 46%), fullest "
          f"{names[n_by.idxmax()]} at n={int(n_by.max()):,}")


if __name__ == "__main__":
    main()
