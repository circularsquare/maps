"""Check the zonal-recovered township populations against the published 2020 census.

A township count recovered from a grid is only worth using if it reconciles with
the printed provincial totals, so this sums township_pop2020.csv up to province
level and prints the gap. Provinces that drift are where our 2018-vintage
boundaries disagree with the 2020 census geography.

Note the census's own national figure (1,411,778,724) is 2,000,000 above the sum
of the 31 provinces: active service personnel are counted nationally only. The
comparison here is against the provincial sum.
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
    """Province code -> Chinese name, rebuilt the way assign_codes ranks them."""
    names = set()
    with fiona.open(XIANGZHEN) as src:
        for feat in src:
            names.add(feat["properties"]["省"])
    return {n: str(i + 1).zfill(2) for i, n in enumerate(sorted(names))}


def main():
    pops = pd.read_csv(DATA / "township_pop2020.csv", dtype={"code": str})
    pops["prov"] = pops["code"].str[:2]
    recovered = pops.groupby("prov")["pop_2020"].sum().rename("recovered")

    ref = pd.read_csv(HERE / "census2020_provinces.csv")
    ref["prov"] = ref["name_cn"].map(province_codes())
    missing = ref[ref["prov"].isna()]
    if len(missing):
        print("no province code for:", list(missing["name_cn"]))

    tbl = ref.merge(recovered, on="prov", how="left")
    tbl["diff_pct"] = 100 * (tbl["recovered"] - tbl["pop_2020"]) / tbl["pop_2020"]

    print(f"{'province':<36} {'recovered':>13} {'census 2020':>13} {'diff':>8}")
    for r in tbl.sort_values("diff_pct").itertuples():
        print(f"{r.name:<36} {int(r.recovered):>13,} {r.pop_2020:>13,} "
              f"{r.diff_pct:>7.2f}%")
    tot_r, tot_c = int(tbl.recovered.sum()), int(tbl.pop_2020.sum())
    print()
    print(f"{'TOTAL':<36} {tot_r:>13,} {tot_c:>13,} "
          f"{100 * (tot_r - tot_c) / tot_c:>7.2f}%")


if __name__ == "__main__":
    main()
