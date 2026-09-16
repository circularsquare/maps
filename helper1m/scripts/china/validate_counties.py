"""How close the built county populations are to the published county census.

validate_provinces.py can only ever look good, because fetch.py scales each
province onto its published total. This is the test that is not rigged: county
figures from the census panel were never used to set the 2020 numbers, only to
set the 2010 ratio, so comparing them measures what the ASPECT recovery is
actually worth at the level below.

Also prints the township size distribution, which is what says whether the
township level is the practical one to assemble 1M-person regions from.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data/china"


def main():
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    # The RAW zonal sums, not the published population.csv. fetch.py now anchors
    # every matched county onto its census figure, so scoring the output against
    # the census would only prove the anchoring ran. What is worth measuring is
    # the grid recovery underneath it — that is what decides the counties
    # anchoring cannot reach, and what would show the recovery getting worse.
    twn = pd.read_csv(DATA / "township_pop2020.csv", dtype={"code": str})
    built = twn.groupby(twn["code"].str[:6])["pop_2020"].sum()
    cnty = pd.read_csv(DATA / "county_ratios.csv", dtype={"code": str})
    cnty["built_2020"] = cnty["code"].map(built)

    m = cnty[cnty["panel_2020"].notna()].copy()
    # A merged panel row — 蜀山区+高新区+经开区 — covers several of our counties at
    # once, so compare the group against the row rather than each county against a
    # total it is only one part of. Grouping is what makes this a test of whether
    # our polygons hold the people the census puts on that ground.
    grp = m.groupby("panel_row", dropna=False).agg(
        built=("built_2020", "sum"),
        panel=("panel_2020", "first"),
        n=("code", "size"),
        prov=("prov", "first"),
        pref=("pref", "first"),
        name=("cnty", lambda s: "+".join(sorted(s))),
    )
    grp["rel"] = (grp["built"] - grp["panel"]) / grp["panel"]
    err = grp["rel"].abs()

    multi = int((grp["n"] > 1).sum())
    print(f"counties matched to the census panel: {len(m)} of {len(cnty)}")
    print(f"compared as {len(grp)} groups, {multi} of them a merged census row "
          f"covering more than one of our counties")
    print("absolute error against the published county census:")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"   p{int(q * 100):<3} {err.quantile(q):6.1%}")
    print(f"   mean  {err.mean():6.1%}")
    print(f"   within 5%:  {(err <= 0.05).mean():5.1%}")
    print(f"   within 10%: {(err <= 0.10).mean():5.1%}")
    print()
    print("worst, which are mostly districts that annexed fringe after 2018:")
    for r in grp.reindex(err.sort_values(ascending=False).index).head(10).itertuples():
        print(f"   {r.prov} {r.pref} {r.name:<22.22} built {r.built:>10,.0f}  "
              f"census {r.panel:>10,.0f}  {r.rel:+7.1%}")

    # The blind spot, stated rather than left implicit. Counties with no census
    # figure are not in any number above, and they are disproportionately the
    # renamed and merged urban districts — the same group most likely to be wrong,
    # so leaving them unsaid flatters the result.
    un = cnty[cnty["panel_2020"].isna()]
    share = un["built_2020"].sum() / cnty["built_2020"].sum()
    print()
    print(f"NOT measured above: {len(un)} counties have no census figure to "
          f"check against, holding {un['built_2020'].sum():,.0f} people "
          f"({share:.1%} of the country).")
    for r in un.nlargest(8, "built_2020").itertuples():
        print(f"   {r.prov} {r.pref} {r.cnty:<14.14} built {r.built_2020:>10,.0f}")

    print()
    towns = pop[(pop.level == 4) & (pop.year == 2020)]["pop"]
    print("township size, 2020:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"   p{int(q * 100):<3} {int(towns.quantile(q)):>9,}")
    print(f"   max  {int(towns.max()):>9,}")
    print(f"   zero {int((towns == 0).sum()):>9,}")
    print()
    counties = pop[(pop.level == 3) & (pop.year == 2020)]["pop"]
    print(f"a 1M region is about {1e6 / towns.median():.0f} townships "
          f"or {1e6 / counties.median():.1f} counties")


if __name__ == "__main__":
    main()
