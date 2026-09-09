"""Egypt — religion by governorate, from the pooled Arab Barometer.

Reads data/raw/arabbarometer/*.sav and writes data/normalized/eg.csv.
`sources/eg.md` has the acquisition route, the terms, the §14 history and the checks in prose.
`sources/arabbarometer.py` holds the construction, which is `sources/lapop.py`'s pointed at a
different survey. `sources.md` §11af assesses the source across the whole Arab world.

**EGYPT HAS COLLECTED RELIGION IN FOUR CENSUSES AND PUBLISHED IT IN NONE SINCE 1986.** That is
the whole reason this file exists, and it is also why the country was closed twice before being
opened. CAPMAS asked the question in 1986, 1996, 2006 and 2017; the last national figure it
published is 1986's, and the microdata extract it deposited for the 2017 census has thirteen
variables (governorate, station, marital status, sex, age, work status, education) and religion
is not among them, so there is no state route to a Coptic map at any geography. §11af has the
catalogue record it was read off.

**DRAWING IT AT GOVERNORATE IS ANITA'S CALL, 2026-09-08**, on `ask/answered/001-eg-*`, which
was filed because spec §14.4 rule 2 is about exactly this: a persecuted minority, mapped finer
than its own state publishes. Her words were that governorates are pretty big. She also asked
whether anything finer was possible and it is not: the Arab Barometer cuts by `Q1` Governorate
and carries no finer geography, so **governorate is simultaneously the ruling and the ceiling
of the instrument**. Nothing in this file re-opens either question.

## What is drawn

    93.3%  Muslim      -> governorate share   (split-half +0.495)
     6.7%  Christian   -> governorate share   (split-half +0.495)

Two answers, both above the 1% eligibility floor, both clearing the split-half bar, so there is
no tail: the shares are a closed partition of every governorate and nothing is spread at a
national rate. That is unusual here and it is what a two-box answer card looks like drawn
honestly.

## THE TWO ATHEISTS ARE NOT DRAWN, AND THE REASON IS THE CARD RATHER THAN THE COUNT

Pooled, `Q1012` returns Muslim, Christian and **two respondents in wave V who answered
`Atheist`**. They are dropped from the universe here rather than spread at a national rate,
which is what `sources/gt.py` does with its one Guatemalan Jew, and the difference is not
squeamishness about the number:

**the option was on one of the four answer cards.** Wave V offers `Atheist`; wave VII offers
`No religion`, which is a different answer; waves III and IV offer neither. A share pooled
across all four for a box that existed on one of them is measuring which questionnaire was
used, not what Egyptians answered, and applying it to 108 million people would draw about
thirty thousand irreligious Egyptians on the strength of two interviews in 2019. LAPOP's card
is stable across its waves, so Guatemala's case is not this case.

**And a floor is not a magnitude.** Egypt is a country where saying this to a stranger with a
clipboard carries a real risk, so 2 in 6,840 is a lower bound of unknown depth; §14.4 rule 1
forbids inventing the correction and nothing published supplies one. `note_public` says so
rather than the map implying it.

## The check that made this country drawable, and it comes from outside the survey

    weighted national Christian share   5.93%    1986 census, the last published: 5.7-5.8%
    Cairo governorate                   8.50%    census Cairo: 9.3% (1986), 8.57% (1996)

§11ad's warning about survey instruments is Suriname, where LAPOP reads **0.21x a census** on
exactly the kind of minority cell this map is for. Egypt does not do that. Both checks are
re-run on every build below rather than quoted.

## THE GOVERNORATE LABELS ARE `[[reference_pooled_survey_labels]]`, AND HARMONISING IS NOT TIDYING

The pooled file carries 45 distinct `Q1` labels for a country with 27 governorates, because
each wave brings its own label set. Two of them prove it is a mechanism and not untidiness:

  * **`The Lake` is Beheira.** al-Buhayra means "the lake", as al-Sharqiyya means "the eastern"
    (which appears as `Eastern`) and al-Gharbiyya "the western" (as `Western`). A
    transliteration join misses all three; a translation join finds them.
  * **`The West Bank` appears in the Egyptian rows of wave III**, n=60, and is dropped. See
    `WEST_BANK_IS_PROBABLY_GHARBIA` below, which is the argument for what it really is and the
    argument for not acting on it.

**Run against the raw labels the split-half FAILS**, at +0.416 against a +0.566 bar, because
only 13 of 27 units appear in both halves and the test is being run on mangled units.
Harmonised first it is +0.495 against +0.418 and passes. The lesson is not that Egypt passes;
it is that a stability test run before the units are harmonised reports noise as a negative,
and a negative is what this project treats as evidence.

Usage:
    python sources/eg.py --fetch    download the Arab Barometer waves (~46 MB of zips)
    python sources/eg.py            rebuild data/normalized/eg.csv
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import arabbarometer as ab

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "eg", "eg_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "eg.csv")

COUNTRY = "Egypt"

# The pool, PINNED rather than derived, and the pin is load-bearing since 2026-09-08.
#
# Egypt is fielded in III, IV, V and VII; VI and VIII do not include it. **Wave II DOES**, with
# 1,219 Egyptian respondents, `q1012 Religion`, `q1 Province/Governorate/State` over 21
# governorates and a weight — and it was invisible to `ab.load()` until `ab.country_key` was
# written, because wave II alone spells its country labels `5. Egypt`. See that function.
#
# **It is left out here, and leaving it out is what keeps this country's published numbers
# still.** Egypt was drawn on 6,840 respondents in four waves; adding a fifth would move every
# governorate cell, the split-half and the 2013-2022 vintage on a map that is already up.
# Whether to rebuild Egypt on the deeper pool changes an already-drawn country and is
# therefore Anita's (AGENT_BRIEF §3), not a side effect of a shared-module fix. `sources/jo.md`
# has the finding written up; wave II reads 71 Christians in 1,219, which is 5.83% against this
# pool's 5.93%, so nothing about it looks like a correction waiting to happen.
WAVES = ["III", "IV", "V", "VII"]
SOURCE_ID = "eg_arabbarometer_2013_2022"
YEARS = "2013-2022"

# The units the split-half is run on: 24 governorates are sampled, and 23 of them have
# respondents in BOTH wave halves. Matrouh appears only in wave V, so it is in the late half
# and not the early one. The bar is 1.96/sqrt(23-1) = +0.418 and `ab.stability` asserts that
# the overlap really is 23 before believing any correlation.
N_UNITS_BOTH_HALVES = 23

# What is drawn on its own governorate shares, asserted so a change in the data is a failure
# here rather than a silent re-drawing of the country.
CARRIES = ["Christian", "Muslim"]

# Answers that leave the universe rather than being drawn. See the module docstring: this is a
# statement about which answer card was in the field, not about the size of the count.
DROPPED = {
    "Atheist": ("2 respondents, wave V only. `Atheist` is on wave V's card, `No religion` on "
                "wave VII's, and neither is on III's or IV's, so a share pooled over the four "
                "measures the questionnaire rather than the country."),
}

# Every `Q1` label the four Egyptian waves use -> the governorate. Keys are the label folded to
# lower case with everything but letters and spaces removed, so `Kafir el-Sheikh` and
# `Kafr El Sheik` are looked up as `kafir elsheikh` and `kafr el sheik`.
NORM = {
    "cairo": "Cairo",
    "alexander": "Alexandria", "alexandria": "Alexandria",
    "port said": "Port Said",
    "suez": "Suez",
    "damietta": "Damietta",
    "dakahlia": "Dakahlia",
    "el sharqiya": "Sharqia", "sharqia": "Sharqia", "eastern": "Sharqia",
    "qalyubia": "Qalyubia", "qalioubia": "Qalyubia", "kaliobeya": "Qalyubia",
    "kafir elsheikh": "Kafr El Sheikh", "kafr al shiekh": "Kafr El Sheikh",
    "kafr el sheik": "Kafr El Sheikh",
    "gharbia": "Gharbia", "western": "Gharbia",
    "monufia": "Monufia", "mnoufia": "Monufia", "menoufia": "Monufia",
    "beheira": "Beheira", "al bouhira": "Beheira", "the lake": "Beheira",
    "ismailia": "Ismailia", "ismalilia": "Ismailia",
    "giza": "Giza",
    "beni suef": "Beni Suef", "bani swif": "Beni Suef",
    "faiyum": "Faiyum", "al fayoum": "Faiyum", "fayoum": "Faiyum",
    "minya": "Minya", "menia": "Minya",
    "asyut": "Asyut", "assiut": "Asyut",
    "sohag": "Sohag", "souhag": "Sohag",
    "qena": "Qena",
    "aswan": "Aswan",
    "luxor": "Luxor",
    "the red sea": "Red Sea", "red sea": "Red Sea",
    "matrouh": "Matrouh",
}

# A label that cannot be an Egyptian governorate. Dropped, reported, never silently mapped.
#
# WEST_BANK_IS_PROBABLY_GHARBIA, AND IT IS STILL NOT MAPPED.
# Wave III's Egyptian rows carry `The West Bank` on code 2010, n=60, and the code appears on no
# non-Egyptian row, so it is not a shared label set with overlapping country ranges. Three
# things say it is Gharbia:
#   1. wave III's codes 2001..2023 reproduce Egypt's own governorate numbering in order, and
#      position 10 in that numbering is Gharbia;
#   2. Gharbia, which is Egypt's fifth-largest governorate, is otherwise absent from wave III
#      and present in every other wave;
#   3. the mechanism is already visible in the same file. al-Gharbiyya is "the western", and a
#      translator handed it alone will offer "the West Bank" for al-Diffa al-Gharbiyya; the
#      same file translates al-Buhayra as `The Lake`.
# That is a good argument and it buys 60 respondents, 0.9% of the pool, in a governorate that
# has 287 of them anyway from the other three waves. It is not taken, because being wrong about
# it is `[[reference_name_join_wrong_neighbour]]` exactly: 60 people assigned to a governorate
# they are not in, preserving every total, invisible to every arithmetic check the build runs.
# A small gain is not worth the project's own worst failure mode. Recorded so nobody re-derives
# it and so a later source can settle it.
BOGUS = {"the west bank"}

# Governorates the survey never sampled. Named here so they are a stated hole rather than three
# units that quietly get no dots: they are 0.8% of Egypt and 48% of its land.
UNSAMPLED = ["New Valley", "North Sinai", "South Sinai"]

UPPER_EGYPT = ["Beni Suef", "Faiyum", "Minya", "Asyut", "Sohag", "Qena", "Luxor", "Aswan"]


def key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", str(s).strip().lower())).strip()


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    ab.unzip()

    print("=== Arab Barometer, Egypt ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES)
    print(f"\n  pooled: {len(df):,} respondents with a religion answer")
    print("  answers, by wave (the CARD is not the same card twice):")
    print(pd.crosstab(df["category"], df["wave"]).to_string())

    # ---- the published-census cross-check, on the pool as it arrives ----
    # Run before anything is dropped or joined, because that is the state §11af measured and
    # the state a reader can reproduce from the .sav files alone.
    chr_all = df["category"].astype(str).str.strip().str.lower().eq("christian")
    w_share = df.loc[chr_all, "w"].sum() / df["w"].sum()
    print(f"\n  pooled Christian share: {chr_all.mean() * 100:.2f}% unweighted, "
          f"{w_share * 100:.2f}% weighted")
    print("    against the 1986 census, the last Egyptian census to publish one: 5.7-5.8%")
    if not 0.050 <= w_share <= 0.070:
        raise SystemExit(
            f"the weighted national Christian share is {w_share:.2%}, outside 5.0-7.0%. The "
            "1986 census published 5.7-5.8% and §11af measured 5.93% on this pool; a reading "
            "outside that band means the instrument or the pool has changed and the country "
            "should not be drawn from it without re-checking §11af's argument.")

    # ---- drop the answers that leave the universe ----
    for cat, why in DROPPED.items():
        n = int((df["category"] == cat).sum())
        if not n:
            raise SystemExit(f"DROPPED names {cat!r}, which this pool no longer contains")
        print(f"\n  dropping {n} respondent(s) who answered {cat!r}: {why}")
        df = df[df["category"] != cat]

    # ---- harmonise the governorate labels BEFORE any test is run on them ----
    g = df[df["geo_raw"].notna()].copy()
    g["raw"] = g["geo_raw"].astype(str).str.strip()
    k = g["raw"].map(key)
    g["gov"] = k.map(NORM)
    print(f"\n  {g['raw'].nunique()} distinct Q1 labels over the four waves")
    bogus = g[k.isin(BOGUS)]
    if len(bogus):
        print(f"  dropped as impossible-in-Egypt labels: {len(bogus)} "
              f"({sorted(bogus['raw'].unique())}) — see BOGUS for what they probably are")
    unmapped = sorted(g.loc[g["gov"].isna() & ~k.isin(BOGUS), "raw"].unique())
    if unmapped:
        raise SystemExit(f"Q1 labels with no governorate: {unmapped}. A label that falls "
                         "through here is silently dropped and the split-half is then run on "
                         "fewer units than its bar assumes — add it to NORM deliberately.")
    g = g[g["gov"].notna()].copy()

    # Two raw labels in the SAME wave collapsing to one governorate would mean the label set is
    # not what it looks like, and the sum would be right while the units were wrong.
    dup = (g.groupby(["wave", "gov"])["raw"].nunique())
    if (dup > 1).any():
        bad = dup[dup > 1]
        raise SystemExit(f"one wave uses two labels for the same governorate: {bad.to_dict()}")
    print(f"  usable: {len(g):,} respondents over {g['gov'].nunique()} governorates")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    names = dict(zip(lut["name"], lut["geo_id"]))
    if sorted(set(g["gov"]) | set(UNSAMPLED)) != sorted(lut["name"]):
        raise SystemExit("the harmonised governorates plus UNSAMPLED are not the 27 in "
                         f"eg_lookup.csv: {sorted(set(g['gov']) ^ set(lut['name']))}")
    g["geo_id"] = g["gov"].map(names)
    sampled = sorted(g["geo_id"].unique())
    print(f"  {len(sampled)} of {len(lut)} governorates sampled; not sampled: "
          f"{', '.join(UNSAMPLED)}")

    pop = lut.set_index("geo_id")["pop"]
    drawn_pop = int(pop[sampled].sum())
    print(f"  CAPMAS 2026-01-01: {int(pop.sum()):,} people, of whom {drawn_pop:,} "
          f"({drawn_pop / pop.sum():.2%}) are in a sampled governorate")

    # ---- the decode, tested without touching the religion column ----
    ab.held_out(g, pop[sampled], COUNTRY, pop_source="CAPMAS 2026-01-01")

    # ---- do the governorates differ at all, and which categories carry a geography ----
    tab = g.assign(chr=g["category"].eq("Christian")).groupby("geo_id")["chr"].agg(
        n="size", chr="sum")
    tab["pct"] = tab["chr"] / tab["n"] * 100
    tab["name"] = [lut.set_index("geo_id")["name"][i] for i in tab.index]
    chi2, p, dof, _ = stats.chi2_contingency(
        np.array([tab["chr"].values, (tab["n"] - tab["chr"]).values]))
    print(f"\n  Christian share by governorate, pooled (chi-square across the "
          f"{len(tab)}: chi2={chi2:.1f} dof={dof} p={p:.3g}):")
    for _i, r in tab.sort_values("pct", ascending=False).iterrows():
        print(f"    {r['name']:<16}{r['pct']:6.1f}%   n={int(r['n']):>4}  "
              f"{int(r['chr']):>3} Christians")
    if p > 1e-6:
        raise SystemExit(f"the governorates no longer differ (p={p:.3g}); nothing here is "
                         "drawable on its own unit shares")

    up = g["gov"].isin(UPPER_EGYPT)
    t2 = g.assign(chr=g["category"].eq("Christian")).groupby(up)["chr"].agg(["mean", "size"])
    chi_up = stats.chi2_contingency(
        pd.crosstab(up, g["category"].eq("Christian")).values)
    print(f"  Upper Egypt {t2.loc[True, 'mean'] * 100:.1f}% Christian on n="
          f"{int(t2.loc[True, 'size']):,}, the rest {t2.loc[False, 'mean'] * 100:.1f}% on n="
          f"{int(t2.loc[False, 'size']):,}  (p={chi_up[1]:.3g})")

    nat = ab.national(g)
    large = ab.stability(g, nat, N_UNITS_BOTH_HALVES)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")

    # ---- Cairo against the only governorate figure an Egyptian census ever published ----
    cairo = tab.loc[names["Cairo"], "pct"]
    print(f"\n  Cairo: {cairo:.2f}% Christian on n={int(tab.loc[names['Cairo'], 'n'])}")
    print("    against the census: 9.3% in 1986 and 8.57% in 1996 (CAPMAS, via the academic "
          "literature; the office has never released the table itself)")
    if not 6.0 <= cairo <= 11.0:
        raise SystemExit(f"Cairo reads {cairo:.2f}%, outside 6-11%. The published census "
                         "figures either side of this survey are 8.57% and 9.3%; a reading "
                         "outside that band is the one external check this country has "
                         "failing, and it should stop the build.")

    # ---- build ----
    out = ab.build(g, nat, large, small, pop, sampled, unit_noun="governorate")
    n_by = g.groupby("geo_id").size()
    out["geo_level"] = "governorate"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["n_gov"] = out["geo_id"].map(n_by)
    out["note"] = out.apply(
        lambda r: (f"Arab Barometer waves III, IV, V and VII pooled, n={r.n_gov} in this "
                   f"governorate; {r.basis_note} applied to CAPMAS's 2026-01-01 governorate "
                   "population estimate"),
        axis=1)

    total = int(out["count"].sum())
    if total != drawn_pop:
        raise SystemExit(f"drawn {total:,} against CAPMAS {drawn_pop:,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    # ---- what the file now says, which is what note_public has to reproduce ----
    drawn = out.groupby("source_category")["count"].sum()
    print("\n  national, as drawn (population-weighted, so not the survey's own weighting):")
    for cat, n in drawn.sort_values(ascending=False).items():
        print(f"    {n / total * 100:6.2f}%  {cat}  ({n:,})")
    ch = out[out["source_category"] == "Christian"].set_index("geo_id")["count"]
    tot = out.groupby("geo_id")["count"].sum()
    share = (ch / tot * 100).sort_values(ascending=False)
    nm = dict(zip(lut["geo_id"], lut["name"]))
    print("\n  Christian share as drawn, with the sample behind each:")
    for gid, s in share.items():
        n = int(n_by[gid])
        phat = s / 100
        ci = 1.96 * np.sqrt(max(phat * (1 - phat), 1e-9) / n)
        print(f"    {nm[gid]:<16}{s:6.2f}%  n={n:>5}  +/-{ci * 100:4.1f}pp   "
              f"{int(ch[gid]):>9,} people")
    thin = n_by.idxmin()
    print(f"\n  thinnest governorate {nm[thin]} at n={int(n_by[thin])}; "
          f"median n={int(n_by.median())}")
    print(f"  largest Christian count: {nm[ch.idxmax()]} at {int(ch.max()):,}; "
          f"largest share: {nm[share.idxmax()]} at {share.max():.2f}%")
    gap = int(pop.sum()) - drawn_pop
    print(f"  not drawn: {', '.join(UNSAMPLED)} = {gap:,} people, "
          f"{gap / pop.sum() * 100:.2f}% of Egypt")


if __name__ == "__main__":
    main()
