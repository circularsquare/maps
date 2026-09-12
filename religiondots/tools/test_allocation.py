"""
How wrong is it to reconstruct fine categories at fine geography?

Every source acquired so far publishes fine CATEGORIES at coarse geography, or coarse
categories at fine GEOGRAPHY, never both (spec §3.9). The obvious repair is to combine them:
take a fine unit's coarse-branch total and split it by the coarse unit's fine-category shares
within that branch. Because our categories nest inside branches, that is what iterative
proportional fitting reduces to.

It rests on an assumption that is false in the interesting cases — that a branch's internal
composition is the same in every fine unit. This script measures the cost of that assumption
using the one dataset that has both: the US Religion Census, 372 bodies BY COUNTY.

Method:
  truth   = county x body                       (what ASARB actually publishes)
  input A = county x branch    (coarse categories, fine geography)   <- derived from truth
  input B = state  x body      (fine categories, coarse geography)   <- derived from truth
  est[county, body] = A[county, branch(body)] * B[state, body] / B[state, branch(body)]

Deriving both inputs from the truth is deliberate: it isolates the method's own error from
any rounding or perturbation the real coarse tables would add.

Run: python tools/test_allocation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent
RAW = HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx"
PATHS = HERE / "taxonomy" / "usrc_groups.csv"

BRANCH_DEPTH = 2   # what a "coarse category" source gives, e.g. Australia's 34 at SA2


def main():
    paths = pd.read_csv(PATHS, dtype={"Group Code": str})
    path_of = dict(zip(paths["Group Code"], paths["path"]))

    df = pd.read_excel(RAW, sheet_name="2020 Group by County",
                       dtype={"FIPS": str, "Group Code": str})
    df["FIPS"] = df["FIPS"].str.strip().str.zfill(5)
    df["Group Code"] = df["Group Code"].str.strip().str.zfill(3)
    df["path"] = df["Group Code"].map(path_of)
    df = df[df["path"].notna() & (df["path"] != "UNMAPPED") & df["Adherents"].notna()]
    df = df[df["Adherents"] > 0].copy()
    df["branch"] = df["path"].str.split(".").str[:BRANCH_DEPTH].str.join(".")
    df["state"] = df["FIPS"].str[:2]
    df = df.rename(columns={"Adherents": "truth"})

    total = df["truth"].sum()
    print(f"truth: {len(df):,} (county, body) cells · {df['path'].nunique()} bodies · "
          f"{df['FIPS'].nunique():,} counties · {df['branch'].nunique()} branches · "
          f"{total:,.0f} adherents\n")

    # the two inputs a real pair of sources would give us
    A = df.groupby(["FIPS", "branch"])["truth"].sum().rename("a")          # fine geo, coarse cat
    B = df.groupby(["state", "path"])["truth"].sum().rename("b")           # coarse geo, fine cat
    Bb = df.groupby(["state", "branch"])["truth"].sum().rename("bb")

    e = df.join(A, on=["FIPS", "branch"]).join(B, on=["state", "path"]).join(
        Bb, on=["state", "branch"])
    e["est"] = e["a"] * e["b"] / e["bb"]

    # A cell the method can never invent: a body present in the county but absent from the
    # state margin is impossible, so only false-positive smearing is at issue here.
    e["err"] = e["est"] - e["truth"]
    tvd = e["err"].abs().sum() / 2 / total
    print(f"total variation distance: {tvd:.4%} of all adherents land in the wrong body")
    print(f"mean |error| per cell: {e['err'].abs().mean():,.0f} people\n")

    # how does it do per body, weighted by that body's size?
    per = e.groupby("path").apply(
        lambda g: pd.Series({
            "adherents": g["truth"].sum(),
            "counties": len(g),
            "misallocated": g["err"].abs().sum() / 2,
        }))
    per["rate"] = per["misallocated"] / per["adherents"]

    print("error by body size — the question is whether small bodies fare worse:")
    for lo, hi, lbl in [(0, 1e4, "     <10k"), (1e4, 1e5, "  10k-100k"),
                        (1e5, 1e6, " 100k-1M"), (1e6, 1e9, "     >1M")]:
        s = per[(per["adherents"] >= lo) & (per["adherents"] < hi)]
        if len(s):
            print(f"  {lbl}: {len(s):>3} bodies, "
                  f"median misallocation {s['rate'].median():>7.2%}, "
                  f"worst {s['rate'].max():>7.2%}")
    print()

    print("worst-served bodies with over 50k adherents (the clustered ones):")
    big = per[per["adherents"] > 50_000].nlargest(12, "rate")
    print(big.assign(
        adherents=lambda d: d["adherents"].map("{:,.0f}".format),
        rate=lambda d: d["rate"].map("{:.1%}".format),
    )[["adherents", "counties", "rate"]].to_string())
    print()

    print("best-served bodies with over 50k adherents:")
    small = per[per["adherents"] > 50_000].nsmallest(8, "rate")
    print(small.assign(
        adherents=lambda d: d["adherents"].map("{:,.0f}".format),
        rate=lambda d: d["rate"].map("{:.1%}".format),
    )[["adherents", "counties", "rate"]].to_string())
    print()

    # Does concentration predict failure? A body in few counties per state should smear worst.
    per["counties_per_state"] = e.groupby("path").apply(
        lambda g: g.groupby("state")["FIPS"].nunique().mean())
    ok = per[per["adherents"] > 20_000]
    r = np.corrcoef(np.log(ok["counties_per_state"]), ok["rate"])[0, 1]
    print(f"correlation between counties-per-state and misallocation rate: r = {r:.3f}")
    print("  (negative = the more spread out a body is, the better the method does)")


if __name__ == "__main__":
    main()
