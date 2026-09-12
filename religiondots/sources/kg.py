"""Kyrgyzstan — religion by oblast, from the EBRD Life in Transition Survey III.

Reads data/raw/lits/lits_iii.dta and writes data/normalized/kg.csv. `sources/lits.py` holds
the construction, the split-half and the permutation nulls; `sources/kg.md` has the source
hunt and the checks in prose; `sources.md` §9co is the write-up.

**NO KYRGYZ CENSUS HAS EVER ASKED ABOUT RELIGION AND THIS IS THE ONLY ROUTE.** That was
established against the questionnaire rather than against the publications, which is §9cd's
rule: Book I of the 2022 census prints Form 2 in full as an annex and it runs questions 1 to
17 — relationship, sex, residence, date of birth, birth certificate, marital status,
nationality, country of birth, citizenship, languages, migration, education, functional
limitations, sources of livelihood, employment, communications, fertility. There is no
religion item, there was none in 2009, and the office's own catalogue of 806 published
statistical tables across its twenty-two subject pages contains no religion table of any kind.
`sources/kg.md` has the full record. Every row this file writes is `modelled` in §7's sense.

## Two categories carry their own oblast geography, and six do not

    89.23%  MUSLIM                                  -> oblast share   (median rho +0.55, p=0.020)
     6.76%  ORTHODOX CHRISTIAN                      -> oblast share   (median rho +0.75, p=0.008)
    ---------------------------------------------------------------------------------------
     1.94%  ATHEISTIC / AGNOSTIC / NONE             -> national rate  (+0.19, p=0.34)
     0.91%  BUDDHIST                                -> national rate  (-0.15, p=0.76)
     0.78%  OTHER CHRISTIAN, INCLUDING PROTESTANT   -> national rate  (+0.39, p=0.17)
     0.19%  OTHER                                   -> national rate  (+0.54, p=0.34)
     0.14%  JEWISH                                  -> national rate  (-0.13, p=0.97)
     0.03%  CATHOLIC                                -> national rate  (one respondent)
     0.02%  Refusal                                 -> EXCLUDED, one respondent

**THE BAR IS A PERMUTATION AND NOT `1.96/sqrt(n-1)`, AND THAT CHANGES THE ANSWER.** On nine
units the fixed bar is +0.693; Islam's median split-half is +0.548 and would have failed it.
The permutation null — the same statistic with the PSU-to-region labels shuffled 400 times —
puts its own 95th percentile at +0.450, so the fixed bar was rejecting a category the data can
distinguish from chance at p=0.020. `sources/lits.py` has the argument, and the general form
is `[[reference_check_needs_power]]`: measure the null before believing a bar.

**And it is seven bars rather than one, two of them stricter than the +0.693 they replaced.**
Read the `null 95th` column the run prints: MUSLIM +0.450, ORTHODOX +0.468, BUDDHIST +0.460,
ATHEISTIC/AGNOSTIC/NONE +0.524, OTHER CHRISTIAN +0.550, then OTHER at +0.750 and JEWISH at
+1.000. A permutation null is looser exactly where a category is dense enough for a
median-of-400 to be stable and stricter exactly where it is not, which is what separates a
re-calibration from a loosening; OTHER fails at p=0.34 because its own null sits at +0.750.
The level was never touched either: `alpha` is `lits.stability`'s inherited 0.05 default, the
same 95% `1.96/sqrt(n-1)` already encoded. `sources/kg.md` §4.1 and §9.1 have the calibration
that shows the test is the size it claims to be.

## THE BUDDHIST CELL IS AN ARTEFACT AND THE TEST CAUGHT IT

Sixteen respondents answer `BUDDHIST`, 0.91% of the weighted sample, and **six of them are in
Osh oblast and four in Batken** — the two most rural, most observant, most overwhelmingly
Muslim oblasts in the country, where Bishkek, which has whatever Buddhist community
Kyrgyzstan has, returns one. There is no reading of Kyrgyzstan in which that is a geography.
Its split-half median is **-0.15 with 68% of halves negative**, the worst of any category
here, so it is drawn at the national rate rather than where the survey put it, and `kg.md`
says the national figure is a ceiling. This is what a keying error looks like from the inside
of a thin survey, and it is the reason the stability test is run on every category rather than
only on the ones that look wrong.

## What drawing only two categories on their own shares actually does

Islam and Orthodoxy between them are 96.0% of the country, so the residual construction in
`lits.build` gives the remaining six the same *relative* proportions everywhere while their
combined weight still varies: Bishkek's tail is 7.9% of the city against Osh oblast's 2.3%.
The Muslim-against-Orthodox line, which is the only thing this survey is really shaped to
measure, is drawn where it was measured.

Usage:
    python sources/kg.py            rebuild data/normalized/kg.csv
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "kg")
LOOKUP = os.path.join(GEO, "kg_lookup.csv")
AGES = os.path.join(GEO, "kg_codps_ages.csv")
OUT = os.path.join(ROOT, "data", "normalized", "kg.csv")

COUNTRY = "Kyrgyz"                 # LiTS labels it `Kyrgyz Rep.`
SOURCE_ID = "kg_lits3_2016"
N_UNITS = 9

# What is drawn on its own oblast shares, asserted so that a change in the data or in the test
# is a failure here rather than a silent re-drawing of the country.
CARRIES = ["MUSLIM", "ORTHODOX CHRISTIAN"]

# One respondent refused. It is excluded from the partition rather than drawn, and `lits.lean`
# reports what excluding it does; with a single non-zero unit there is nothing to measure and
# the check says so instead of producing a number.
EXCLUDE = [lits.REFUSAL]


def age_lean(df, lut, ages):
    """§3.5's OTHER hole, and for Kyrgyzstan it is much larger than the refusals.

    LiTS III interviews adults. Kyrgyzstan is one of the youngest countries in the former
    Soviet Union and the under-18 share is not flat across it, so applying an adult
    composition to the whole resident population is not neutral and the direction is
    measurable from COD-PS's own age bands. Reported, never corrected: §14.4 forbids inventing
    the magnitude of a correction and nothing published says what it would be.
    """
    child = [c for c in ages.columns
             if any(b in c for b in ("from_0to4", "from_5to9", "from_10to14"))]
    teen = [c for c in ages.columns if "from_15to19" in c]
    if len(child) != 6 or len(teen) != 2:
        raise SystemExit(f"age band columns did not resolve: {len(child)} child, {len(teen)}")
    total = ages.sum(axis=1)
    # 15-19 is carried at 60% under-18, the convention lapop.py uses in the other direction.
    under18 = (ages[child].sum(axis=1) + 0.6 * ages[teen].sum(axis=1)) / total
    print("\n  §3.5, the age hole, which is the large one here:")
    print(f"    LiTS III interviews adults; COD-PS 2018 puts {under18.mean():.1%} of the "
          f"average oblast under 18,\n    and the spread across the nine is "
          f"{under18.min():.1%} ({lut.loc[under18.idxmin(), 'name']}) to "
          f"{under18.max():.1%} ({lut.loc[under18.idxmax(), 'name']}).")
    return under18


def main():
    lut = pd.read_csv(LOOKUP).set_index("geo_id")
    pop = lut["pop"].astype("int64")
    units = sorted(lut.index)

    df = lits.load(COUNTRY)
    to_geo = {v: k for k, v in lut["lits_region"].items()}
    unknown = sorted(set(df["region"]) - set(to_geo))
    if unknown:
        raise SystemExit(f"LiTS region strings with no oblast: {unknown}")
    df["geo_id"] = df["region"].map(to_geo)

    missing = lits.coverage(df, pop)
    if missing:
        raise SystemExit(f"LiTS III does not reach {missing}; Kyrgyzstan was taken because it "
                         "reaches all nine, so this is a change in the file, not in the code")

    lits.held_out(df, pop, "the Kyrgyz Republic",
                  pop_source="the office's 1 January 2026 register",
                  names=[lut.loc[u, "name"] for u in units])

    kept = df[~df["code"].isin(EXCLUDE)]
    print(f"\n  {len(df) - len(kept)} of {len(df):,} respondents excluded from the partition "
          f"({', '.join(EXCLUDE)})")
    nat = lits.national(kept)

    large = lits.stability(kept, nat, units)
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    small = [c for c in nat.index if c not in large]
    print(f"    -> {len(large)} categories drawn on their own oblast shares, "
          f"{len(small)} spread at the national rate")

    out = lits.build(kept, nat, large, small, pop, units, unit_noun="oblast")

    out["geo_level"] = "oblast"
    out["geo_name"] = out["geo_id"].map(lut["name"])
    out["basis"] = "self_id"
    out["year"] = "2015-2016"
    out["source_id"] = SOURCE_ID
    n_obl = kept.groupby("geo_id").size()
    out["n_oblast"] = out["geo_id"].map(n_obl)
    out["note"] = out.apply(
        lambda r: (f"EBRD Life in Transition Survey III, n={r.n_oblast} in this oblast; "
                   f"{r.basis_note} applied to the National Statistical Committee's "
                   "1 January 2026 resident population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against the office's {int(pop.sum()):,}")

    lits.lean(df, out, EXCLUDE, units)
    ages = pd.read_csv(AGES).set_index("pcode")
    under18 = age_lean(df, lut, ages)
    drawn = out.pivot_table(index="geo_id", columns="source_category", values="count",
                            aggfunc="sum")
    drawn = drawn.div(drawn.sum(axis=1), axis=0).reindex(units)
    for c in CARRIES:
        r = float(np.corrcoef(under18.reindex(units), drawn[c])[0, 1])
        loo = [float(np.corrcoef(under18.reindex(units).drop(u), drawn[c].drop(u))[0, 1])
               for u in units]
        print(f"    under-18 share vs the drawn {c[:22]:<24} r = {r:+.3f}  "
              f"leave-one-out {min(loo):+.3f} to {max(loo):+.3f}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    share = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in share.items():
        print(f"    {sh * 100:7.3f}%  {cat}")

    print("\n  the two categories drawn on their own oblast shares, with the sample behind "
          "each:")
    print(f"    {'oblast':<12}{'n':>5}{'Muslim':>10}{'Orthodox':>10}"
          f"{'  95% CI on Orthodox':>22}")
    for geo_id in drawn["ORTHODOX CHRISTIAN"].sort_values(ascending=False).index:
        n = int(n_obl[geo_id])
        p = drawn.loc[geo_id, "ORTHODOX CHRISTIAN"]
        ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
        print(f"    {lut.loc[geo_id, 'name']:<12}{n:>5}"
              f"{drawn.loc[geo_id, 'MUSLIM'] * 100:9.1f}%{p * 100:9.1f}%"
              f"{'':>8}+/-{ci * 100:4.1f}pp")
    thin = n_obl.idxmin()
    print(f"\n    the thinnest oblast is {lut.loc[thin, 'name']} at n={int(n_obl[thin])}, "
          "which countries.py names in note_public")
    zero = [lut.loc[g, "name"] for g in units
            if kept[(kept["geo_id"] == g) & (kept["code"] == "ORTHODOX CHRISTIAN")].empty]
    print(f"    oblasts where NO respondent answered Orthodox, and which are therefore drawn "
          f"with none: {zero}")


if __name__ == "__main__":
    main()
