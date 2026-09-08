"""Guatemala — religion by departamento, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/gt.csv.
`sources/gt.md` has the acquisition route, the terms, and the checks in prose.
`sources/lapop.py` holds the construction, the split-half and the category list, which are
the same in every country drawn from this survey. `sources.md` §11ad assesses the source
across all nine countries it could serve; read that before extending this to another one.

**GUATEMALA'S CENSUS HAS NEVER ASKED ABOUT RELIGION AND THIS IS THE ONLY ROUTE.** sources.md
§11x closed the country on two witnesses — the UNSD oracle reports 1964 as the only Guatemalan
religion tabulation ever forwarded, and IPUMS's `RELIGION` variable says the same — after
finding INE's own site behind a Radware captcha and `censopoblacion.gt` a parked domain. None
of that has changed and this file does not reopen it. It is a survey standing where a census
would be, and every row it writes is `modelled` in §7's sense.

## The three categories that carry their own geography, and the eight that do not

    51.97%  Católico                                 -> department share   (+0.57)
    34.52%  Evangélica y Pentecostal                 -> department share   (+0.50)
     4.85%  Ninguna (cree en un Ser Superior)        -> department share   (+0.21, OVERRIDE)
    ------------------------------------------------------------------------
     5.43%  Protestante / Protestante Tradicional    -> national rate      (-0.04)
     1.14%  Otro                                     -> national rate
     0.64%  Agnóstico o ateo                         -> national rate
     0.49%  Testigos de Jehová                       -> national rate
     0.42%  Mormones                                 -> national rate
     0.31%  Religiones Orientales no Cristianas      -> national rate
     0.22%  Religiones Tradicionales                 -> national rate
     0.01%  Judío                                    -> national rate

**THE CUT IS THE SPLIT-HALF AND NOT THE SIZE**, and the two disagree here: `Protestante
Tradicional` is 5.43% of the country and its rank correlation across the wave halves is
**-0.04**. `lapop.stability()` has the argument.

## `Ninguna (creyente)` IS DRAWN UNDER THE BAR — Anita's call, 2026-09-08

**It is the case where the two halves of the evidence disagree**, and the reason it goes in
`OVERRIDE` rather than being handled by moving the bar is that only one named category is
being decided, by a person, with the reason on screen every run.

The split-half asks whether the ORDERING replicates and returns **+0.21** against a +0.43 bar.
But it never asks whether the departments differ at all, and they do, overwhelmingly:
**chi-square p = 3.6e-16** across the 22. So the +0.21 says the ordering is not pinned, not
that the variation is fake. That is spec §14.16's China exactly — Protestantism at +0.17 rank
stability with p=1.3e-84 spatial significance, drawn on Anita's call with the weakness named —
and drawing this one keeps the two countries consistent.

**And the part that is stable is the part a reader takes off the map.** The capital is top in
both wave halves (8.2% then 8.6%, on 188 respondents) and four of the five Maya highland
departments sit stably at the bottom (Quiché 0.9→1.3, Alta Verapaz 1.5→1.8, Huehuetenango
1.5→2.4, Sololá 2.2→2.1). It is the middle eighteen that shuffle — Chiquimula 8.9→2.3,
Sacatepéquez 0.0→7.9 — and `note_public` says so in those terms.

**What the alternative claimed.** Drawn flat, every department got 4.85%, which asserts that
Quiché and Guatemala City have identical shares. That is not a neutral default; it is a
different wrong claim, and a louder one.

## What the survey cannot see at all, and it is not a small thing

`Religiones Tradicionales` is 0.22% here, in a country the 2018 census found to be 43.6%
indigenous. §11ad measured what this instrument does to folk practice in the one place a
census could check it: in Suriname LAPOP's traditional-religion share is **0.21x the census's**
and the missing people come back as Christians (that cell reads 1.50x census). The card has no
Maya-spirituality option and `costumbre` is not a word on it. **So 0.22% is a floor and the
country note says so.** It is not corrected here, because §14.4 forbids inventing a magnitude
and nothing published says what the right one is.

Usage:
    python sources/gt.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/gt.py            rebuild data/normalized/gt.csv from the slim extract
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lapop

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POP = os.path.join(ROOT, "data", "raw", "gt", "gtm_admpop_adm1_2024.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "gt", "gt_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "gt.csv")

PAIS = 2                       # LAPOP's country code for Guatemala
WAVES = [2010, 2012, 2014, 2016, 2018, 2023]
SOURCE_ID = "gt_lapop_2010_2023"
N_UNITS = 22

# What is drawn on its own department shares, asserted so a change in the data is a failure
# here rather than a silent re-drawing of the country.
#   1 = Católico (+0.57), 5 = Evangélica y Pentecostal (+0.50), both clearing the +0.43 bar
#   4 = Ninguna, creyente (+0.21) — UNDER the bar, drawn on Anita's call, see OVERRIDE
CARRIES = [1, 4, 5]

# ANITA'S CALL, 2026-09-08. The split-half is not the only evidence and this is the case
# where the two halves of the evidence disagree. See the module docstring for the argument.
OVERRIDE = {
    4: ("chi-square across the 22 departments is p=3.6e-16, so the departments genuinely "
        "differ; the split-half's +0.21 says the ORDERING is not pinned, not that the "
        "variation is fake. Spec §14.16's China precedent is the same shape at +0.17 and was "
        "drawn. The stable part is also the readable part: the capital is top in both wave "
        "halves (8.2% then 8.6% on 188 respondents) and four of the five Maya highland "
        "departments are stably at the bottom; it is the middle eighteen that shuffle. "
        "note_public says the middle ranking is not to be trusted."),
}


def main():
    if "--fetch" in sys.argv:
        lapop.fetch()

    df = lapop.load(PAIS, WAVES)

    # THE DECODE. LAPOP's `prov` for Guatemala is 200 plus the official department number,
    # and COD's pcode is `GT` plus the same number — sources/gt_geo.py asserts the other
    # half of that identity, name by name, and refuses to run if it stops holding.
    # It is NOT like this in every country: El Salvador's COD pcodes are alphabetical and
    # the same arithmetic would mispair twelve of fourteen (sources/sv_geo.py).
    bad = sorted(set(df["prov_code"]) - set(range(201, 223)))
    if bad:
        raise SystemExit(f"prov codes outside 201..222: {bad}")
    df["geo_id"] = (df["prov_code"] - 200).astype(str).str.zfill(2)

    print(f"Guatemala: {len(df):,} respondents with a religion answer and a department, "
          f"waves {WAVES[0]}-{WAVES[-1]}")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = (pop["ADM1_PCODE"].astype(str)
                     .str.replace("^GT", "", regex=True).str.zfill(2))
    pop = pop.set_index("geo_id").rename(columns={"T_TL": "pop"})
    missing = sorted(set(lut["geo_id"]) - set(pop.index))
    if missing:
        raise SystemExit(f"no COD-PS population for {missing}")
    if sorted(df["geo_id"].unique()) != sorted(lut["geo_id"]):
        raise SystemExit("LAPOP's departments and the lookup's do not agree: "
                         f"{sorted(set(lut['geo_id']) ^ set(df['geo_id'].unique()))}")

    lapop.held_out(df, pop, "Guatemala")

    nat = lapop.national(df)
    large = lapop.stability(df, nat, N_UNITS, override=OVERRIDE)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    print(f"    -> {len(large)} categories drawn on their own department shares, "
          f"{len(small)} spread at the national rate")

    units = sorted(lut["geo_id"])
    out = lapop.build(df, nat, large, small, pop["pop"], units, unit_noun="department")

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    out["n_dept"] = out["geo_id"].map(df.groupby("geo_id").size())
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_dept} in this "
                   f"department; {r.basis_note} applied to the COD-PS 2024 population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop["pop"].sum()):
        raise SystemExit(f"drawn {total:,} against COD-PS {int(pop['pop'].sum()):,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print(f"\n  the {len(large)} categories drawn on their own department shares, with the "
          "sample behind each:")
    n_by = df.groupby("geo_id").size()
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    cath, ev = lapop.CATEGORY[1], lapop.CATEGORY[5]
    names = dict(zip(lut["geo_id"], lut["name"]))
    print(f"    {'dept':<16}{'n':>6}{'Catholic':>10}{'Evang':>8}{'  95% CI on Catholic'}")
    for geo_id in show[ev].sort_values(ascending=False).index:
        n = int(n_by[geo_id])
        p = show.loc[geo_id, cath]
        ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
        print(f"    {names[geo_id]:<16}{n:>6}{p * 100:9.1f}%{show.loc[geo_id, ev] * 100:7.1f}%"
              f"   +/-{ci * 100:4.1f}pp")
    thin = n_by.idxmin()
    print(f"\n    the thinnest department is {names[thin]} at n={int(n_by[thin])}, which "
          "countries.py names in note_public")


if __name__ == "__main__":
    main()
