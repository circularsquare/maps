"""Check the built populations against the published census, province by province.

2020 is the real test: those counts are recovered from the ASPECT grid and
nothing else pins them, so a province that drifts is a place where our
2018-vintage boundaries disagree with the 2020 census geography.

2010 is a weaker test by construction — fetch.py normalises each province onto
its published 2010/2020 growth — so it mainly confirms that normalisation
landed, and that the 2020 bias carries through rather than compounding.

Note the census's own national figures are above the sum of the 31 provinces:
by 2,000,000 in 2020 and by about 6,900,000 in 2010, being servicemen and, in
2010, people whose usual residence could not be determined. Both are counted
nationally only, so the comparison here is against the provincial sum.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fiona
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/china"
XIANGZHEN = HERE.parents[2] / "data/asia1m/china/xiangzhen.shp"


def province_codes():
    """Chinese province name -> code, rebuilt the way assign_codes ranks them."""
    names = set()
    with fiona.open(XIANGZHEN) as src:
        for feat in src:
            names.add(feat["properties"]["省"])
    return {n: str(i + 1).zfill(2) for i, n in enumerate(sorted(names))}


def main():
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    ours = pop[pop["level"] == 1].pivot(index="code", columns="year", values="pop")
    ours.columns = [f"ours_{y}" for y in ours.columns]

    codes = province_codes()
    ref = pd.read_csv(HERE / "census2010_provinces.csv").merge(
        pd.read_csv(HERE / "census2020_provinces.csv")[["name_cn", "pop_2020"]],
        on="name_cn")
    # The 2024 reference is the yearbook, not a census — a different series, so
    # a gap here is the base year differing, not an error.
    yb = pd.read_csv(HERE / "yearbook_provinces.csv", comment="#")
    ref = ref.merge(yb[["name_cn", "pop_2024"]], on="name_cn")
    ref["code"] = ref["name_cn"].map(codes)
    tbl = ref.merge(ours, on="code", how="left")

    years = [y for y in (2010, 2020, 2024) if f"ours_{y}" in tbl.columns]
    for y in years:
        tbl[f"d{y}"] = 100 * (tbl[f"ours_{y}"] - tbl[f"pop_{y}"]) / tbl[f"pop_{y}"]

    head = f"{'province':<36}" + "".join(f"{y:>10}" for y in years)
    print(head + "   (built minus published, %)")
    for r in tbl.sort_values(f"d{years[-1]}").itertuples():
        line = f"{r.name:<36}"
        for y in years:
            line += f"{getattr(r, f'd{y}'):>9.2f}%"
        print(line)

    print()
    for y in years:
        o, c = int(tbl[f"ours_{y}"].sum()), int(tbl[f"pop_{y}"].sum())
        print(f"{'TOTAL ' + str(y):<36} {o:>15,} {c:>15,} "
              f"{100 * (o - c) / c:>6.2f}%")


if __name__ == "__main__":
    main()
