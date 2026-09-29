"""How the built county populations compare with the county census.

fetch.py carries the census onto our boundaries through the grid, so three things
are worth measuring, and they are kept apart:

1. The grid itself, summed over the census's own county polygons and set against
   the census figure for that polygon. This is what ASPECT is worth at county
   level, before we touch it.
2. The published figures, where one of our counties and one census county are
   the same ground (each holds 95% or more of the other's people). There the
   census figure is the right answer outright.
3. The counties the rule deliberately left away from their census-implied figure
   (the census rows split by grid share), with the reason, so a large gap is
   never silent: either the prefecture balanced and the grid kept where people
   are, or people the census books to no county were handed back.

Also prints the township size distribution, which is what says whether the
township level is the practical one to assemble 1M-person regions from.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data/china"


def dist(err, label):
    print(f"{label}")
    for q in (0.5, 0.9, 0.95, 0.99):
        print(f"   p{int(q * 100):<3} {err.quantile(q):6.1%}")
    print(f"   within 5%:  {(err <= 0.05).mean():5.1%}   within 10%: {(err <= 0.10).mean():5.1%}")


def main():
    rows = pd.read_csv(DATA / "panel_rows.csv", dtype={"county_code": str})
    cty = pd.read_csv(DATA / "counties.csv", dtype={"code": str, "top_code": str})
    pieces = pd.read_csv(DATA / "township_panel_pop2020.csv",
                         dtype={"code": str, "panel_code": str}, keep_default_na=False)
    pieces = pieces[(pieces["code"] != "") & (pieces["panel_code"] != "")]

    # 1. The grid on the census's own ground.
    r = rows[rows["pooled"]].copy()
    r["rel"] = r["grid"] / r["popu_2020"] - 1
    dist(r["rel"].abs(), f"1. grid summed over each census county's own polygon, "
                         f"{len(r)} census counties:")
    print("   worst:")
    for x in r.reindex(r["rel"].abs().sort_values(ascending=False).index).head(10).itertuples():
        print(f"     {x.province} {x.city} {str(x.county)[:26]:<26} grid {x.grid:>10,.0f}  "
              f"census {x.popu_2020:>10,.0f}  {x.rel:+7.1%}")

    # 2. One of ours = one census county.
    pieces["county"] = pieces["code"].str[:6]
    pair = pieces.groupby(["county", "panel_code"])["pop"].sum().reset_index()
    pair["of_ours"] = pair["pop"] / pair.groupby("county")["pop"].transform("sum")
    pair["of_theirs"] = pair["pop"] / pair["panel_code"].map(rows.set_index("county_code")["grid"])
    same = pair[(pair["of_ours"] >= 0.95) & (pair["of_theirs"] >= 0.95)]
    same = same.merge(rows[["county_code", "popu_2020", "pooled"]], left_on="panel_code",
                      right_on="county_code")
    same = same[same["pooled"]]
    same["pub"] = same["county"].map(cty.set_index("code")["pub_2020"])
    same["raw"] = same["county"].map(cty.set_index("code")["raw_2020"])
    print()
    print(f"2. {len(same)} of our {len(cty)} counties are the same ground as one census county")
    dist((same["raw"] / same["popu_2020"] - 1).abs(), "   the raw grid sum against the census:")
    dist((same["pub"] / same["popu_2020"] - 1).abs(), "   the published figure against the census:")

    # 3. Left away from the census-implied figure, on purpose.
    c = cty[cty["covered"] >= 0.95].copy()
    c["rel"] = c["pub_2020"] / c["implied_2020"] - 1
    returned = rows.set_index("county_code")["returned"]
    off = c[c["rel"].abs() > 0.15].sort_values("rel", key=abs, ascending=False)
    print()
    print(f"3. {len(off)} counties sit more than 15% from their census-implied figure, "
          f"holding {off['pub_2020'].sum():,.0f} people:")
    for x in off.head(20).itertuples():
        why = ("people the census books to no county, handed back"
               if returned.get(x.top_code, 0) > 0
               else "prefecture balances; the grid decides where")
        print(f"     {x.prov} {x.pref} {x.cnty:<10} pub {x.pub_2020:>10,.0f}  implied "
              f"{x.implied_2020:>10,.0f}  {x.rel:+7.1%}  ({why})")

    un = cty[cty["covered"] < 0.95]
    print()
    print(f"NOT checkable: {len(un)} counties have under 95% of their people on a census "
          f"county with a figure, holding {un['raw_2020'].sum():,.0f} people.")
    for x in un.nlargest(6, "raw_2020").itertuples():
        print(f"   {x.prov} {x.pref} {x.cnty:<12} {x.raw_2020:>10,.0f}  covered {x.covered:.0%}")

    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    towns = pop[(pop.level == 4) & (pop.year == 2020)]["pop"]
    print()
    print("township size, 2020:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"   p{int(q * 100):<3} {int(towns.quantile(q)):>9,}")
    print(f"   max  {int(towns.max()):>9,}")
    print(f"   zero {int((towns == 0).sum()):>9,}")
    counties = pop[(pop.level == 3) & (pop.year == 2020)]["pop"]
    print(f"a 1M region is about {1e6 / towns.median():.0f} townships "
          f"or {1e6 / counties.median():.1f} counties")


if __name__ == "__main__":
    main()
