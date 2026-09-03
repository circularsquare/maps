"""
Reunite fine categories with fine geography — spec.md §3.10.

Every source acquired so far publishes fine categories at coarse geography OR coarse
categories at fine geography, never both (§3.9). This splits the fine-geography column by the
coarse-geography shares within the same branch of the SOURCE'S OWN classification, so it runs
before any taxonomy mapping and needs none.

    est[fine unit, leaf] = fine[fine unit, home(leaf)]
                           x coarse[leaf] / sum(coarse[l] for l with home(l) == home(leaf))

Measured cost of the assumption, against US ground truth (tools/test_allocation.py): 5.84% of
all adherents land in the wrong unit, but ~42% for bodies under 10,000 — the ones this project
is about. Output is therefore `derived` (§7), draws desaturated, and MUST NOT feed a ring:
allocation cannot establish presence, only spread a total.

## The mapping is validated by arithmetic, never trusted from codes

Australia looks like a clean prefix hierarchy (ASCRG `2233` Greek Orthodox nests under `223`
Eastern Orthodox) and is not: the SA2 column `603 Other Religious Groups` carries its own
prefix children AND every other narrow group in broad group 6 — Paganism, Mandaean, Wiccan,
Taoism, Jainism, Druse, Yezidi, Zoroastrianism. A pure prefix join drops 30 leaves and 92,331
people silently. So every fine column's children are summed and checked against that column's
own coarse total, and a column that does not reconcile is reported rather than allocated.

## Two hierarchy styles, because sources differ

Each source encodes its own classification differently and there is no point pretending
otherwise:

  --hierarchy prefix   the code is a nested string, so `2233` sits under `223` (ABS/ASCRG).
  --hierarchy parent   each row names its parent category (StatCan), so walk up the chain
                       until reaching a category the fine geography actually publishes.

In parent mode the fine "leaves" are derived rather than declared: a fine category that is an
ancestor of another fine category is an aggregate, not a leaf, and is dropped. Canada's CSD
list contains both `Christian` and `Anglican`, and summing them would count several million
people twice.

Usage:
    python allocate.py --source au --fine sa2 --coarse nation --hierarchy prefix
    python allocate.py --source ca --fine csd --coarse province --hierarchy parent
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NORM = HERE / "data" / "normalized"

# Per-column reconciliation.
#
# This is a check that the MAPPING is right, not that the totals agree — and the distinction
# sets the threshold. Shares are normalised within each column (`share = count / sum over the
# column's children`), so a discrepancy between the children's sum and the column total does
# not affect the answer at all; only the relative composition does. What the check has to
# catch is a structurally wrong mapping, and those are enormous: routing a coarse tree's
# aggregates into the same column as its leaves gives children ≈ 2x the column (+100%), which
# is what Canada did before `coarse_leaves` was added.
#
# So the band is wide. StatCan's province and CSD figures come from different products that
# disagree by 2.5–4.4% on the same category, which is a fact about StatCan and not a mapping
# error; rejecting those would silently drop Anabaptist — and with it every Old Order Mennonite
# group, exactly the granularity this project exists for. The absolute floor covers base-5
# rounding on small columns (spec §3.8), the same lesson as §3.6's residual threshold.
TOL_REL = 0.10
TOL_ABS = 100


def tag(series, key):
    return series.str.extract(rf"{key}=(\w+)", expand=False)


def leaves_of(fine_cats, parent_of):
    """Fine categories that are not an ancestor of another fine category."""
    fine = set(fine_cats)
    ancestors = set()
    for c in fine:
        p = parent_of.get(c)
        seen = 0
        # A category that is its own parent is an identity row in a hand-written hierarchy
        # file, not an aggregate — it means "this coarse category IS this fine column".
        # Treating it as an ancestor silently deletes the column.
        while p and p != c and seen < 20:
            if p in fine:
                ancestors.add(p)
            c, p = p, parent_of.get(p)
            seen += 1
    return fine - ancestors


def climb(cat, fine_leaves, parent_of):
    """Nearest ancestor-or-self of `cat` that the fine geography publishes as a leaf."""
    c, seen = cat, 0
    while c is not None and seen < 20:
        if c in fine_leaves:
            return c
        c = parent_of.get(c)
        seen += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="two-letter source file stem, e.g. au")
    ap.add_argument("--fine", required=True, help="geo_level with fine geography")
    ap.add_argument("--coarse", required=True, help="geo_level with fine categories")
    ap.add_argument("--code", default="ascrg", help="note= key holding the source's own code")
    ap.add_argument("--hierarchy", default="prefix", choices=["prefix", "parent"])
    ap.add_argument("--drop", default="", help="regex of categories to exclude as totals")
    ap.add_argument("--parent-file", help="csv of source_category,parent — for sources that "
                                          "do not encode their own hierarchy (see "
                                          "taxonomy/hierarchy/)")
    args = ap.parse_args()

    src = NORM / f"{args.source}.csv"
    df = pd.read_csv(src, dtype={"geo_id": str}, low_memory=False)

    if args.hierarchy == "prefix":
        df["level"] = tag(df["note"], "level")
        df["code"] = tag(df["note"], args.code)
        df = df[df["code"].notna() & (df["level"] == "leaf")]
    else:
        df["code"] = df["source_category"]
        df["parent"] = df["note"].str.extract(r"parent=([^;]*)")
    if args.drop:
        df = df[~df["source_category"].str.contains(args.drop, regex=True, na=False)]

    fine = df[df["geo_level"] == args.fine]
    coarse = df[df["geo_level"] == args.coarse]
    if fine.empty or coarse.empty:
        raise SystemExit(f"need rows at both {args.fine} and {args.coarse}")

    if args.hierarchy == "parent":
        if args.parent_file:
            pf = pd.read_csv(HERE / args.parent_file, comment="#")
            parent_of = dict(zip(pf["source_category"], pf["parent"]))
            print(f"  hierarchy from {args.parent_file}: {len(parent_of)} categories")
        else:
            parent_of = (coarse.dropna(subset=["parent"])
                         .drop_duplicates("source_category")
                         .set_index("source_category")["parent"].to_dict())
        parent_of = {k: (v if isinstance(v, str) and v else None)
                     for k, v in parent_of.items()}

        # The coarse table holds the whole tree — leaves AND every aggregate above them.
        # Summing all of a column's descendants therefore counts the intermediate totals as
        # well, which shows up as children ≈ 2x the column. Keep only categories that are
        # nobody's parent; those partition the population exactly once.
        coarse_cats = set(coarse["code"].unique())
        is_parent = {p for c, p in parent_of.items() if p in coarse_cats and p != c}
        coarse_leaves = coarse_cats - is_parent
        print(f"  {args.coarse}: {len(coarse_cats)} categories, "
              f"{len(coarse_leaves)} are leaves ({len(is_parent)} are aggregates)")
        coarse = coarse[coarse["code"].isin(coarse_leaves)]

        fine_leaves = leaves_of(fine["code"].unique(), parent_of)
        dropped = set(fine["code"].unique()) - fine_leaves
        if dropped:
            print(f"  {args.fine} aggregates dropped (ancestors of other {args.fine} "
                  f"categories): {sorted(dropped)}")
        fine = fine[fine["code"].isin(fine_leaves)]
    fine_tot = fine.groupby("code")["count"].sum()
    codes = set(fine_tot.index)
    print(f"{args.source}: {len(codes)} categories at {args.fine}, "
          f"{coarse['code'].nunique()} at {args.coarse}")

    cat = coarse.groupby(["code", "source_category"])["count"].sum().reset_index()

    if args.hierarchy == "parent":
        cat["home"] = cat["code"].map(lambda c: climb(c, codes, parent_of))
        lost = cat[cat["home"].isna()]
        if len(lost):
            print(f"  !! {len(lost)} categories have no {args.fine} home "
                  f"({lost['count'].sum():,.0f} people): "
                  f"{sorted(lost['code'])[:6]}")
    else:
        def prefix_home(c):
            cands = [x for x in codes if c.startswith(x)]
            return max(cands, key=len) if cands else None

        cat["home"] = cat["code"].map(prefix_home)

        # orphans fall back to the residual bucket of their broad group, identified as the
        # fine column in that group whose children under-account for it by the most.
        orphans = cat[cat["home"].isna()]
        if len(orphans):
            matched = cat[cat["home"].notna()].groupby("home")["count"].sum()
            shortfall = (fine_tot - matched.reindex(fine_tot.index).fillna(0))
            for grp in sorted(orphans["code"].str[0].unique()):
                in_grp = [c for c in codes if c.startswith(grp)]
                if not in_grp:
                    print(f"  !! broad group {grp}: no {args.fine} column at all — "
                          f"{orphans[orphans['code'].str.startswith(grp)]['count'].sum():,.0f} "
                          f"people cannot be placed")
                    continue
                bucket = shortfall.reindex(in_grp).idxmax()
                sel = cat["home"].isna() & cat["code"].str.startswith(grp)
                cat.loc[sel, "home"] = bucket
                print(f"  broad group {grp}: {sel.sum()} orphaned categories -> "
                      f"residual column {bucket}")

    cat = cat[cat["home"].notna()]

    # 3. validate every column by arithmetic before trusting it.
    #
    # Compare within ONE geography wherever possible. The fine geography may cover only part
    # of the country — Canadian census tracts are 75.8% of the population — so testing the
    # coarse table's children against the fine table's column measures coverage, not mapping,
    # and threw away 17 of 23 Canadian columns for a reason that had nothing to do with the
    # mapping being wrong. Where the coarse table also publishes the fine column's own
    # category (Canada's province table carries both the aggregates and the leaves), check
    # against that instead: same geography, so only the mapping can move it.
    child_sum = cat.groupby("home")["count"].sum()
    coarse_all = df[df["geo_level"] == args.coarse].groupby("code")["count"].sum()
    same_geo = coarse_all.reindex(child_sum.index)

    # Per column, not either/or: validate against the coarse table's own aggregate where it
    # publishes one, and fall back to the fine column otherwise. An earlier version chose one
    # mode for the whole run and so silently allocated only the columns the chosen mode could
    # see — Ireland lost 22 of 24 categories that way, with every check passing.
    chk = pd.DataFrame({
        "children": child_sum,
        "column": same_geo.fillna(fine_tot.reindex(child_sum.index)),
    }).dropna()
    chk["how"] = np.where(same_geo.reindex(chk.index).notna(), "same-geo", "cross-geo")
    n_same = int((chk["how"] == "same-geo").sum())
    print(f"  validating {n_same} columns against the {args.coarse} table's own aggregates, "
          f"{len(chk) - n_same} across geographies")
    chk["diff"] = chk["children"] - chk["column"]
    chk["rel"] = chk["diff"] / chk["column"]
    bad = chk[(chk["rel"].abs() > TOL_REL) & (chk["diff"].abs() > TOL_ABS)]
    print(f"\n  columns reconciling within {TOL_REL:.0%} or {TOL_ABS} people: "
          f"{len(chk) - len(bad)}/{len(chk)}")
    for code, row in bad.iterrows():
        name = fine[fine["code"] == code]["source_category"].iloc[0]
        print(f"  !! {code} {name[:34]:<36} children {row['children']:>10,.0f} "
              f"vs column {row['column']:>10,.0f} ({row['rel']:+.1%}) — NOT allocated")
    ok = set(chk.index) - set(bad.index)
    cat = cat[cat["home"].isin(ok)]

    # a column with one child needs no allocation and stays exact
    nkids = cat.groupby("home")["code"].transform("size")
    cat["exact"] = nkids == 1
    print(f"  {int((nkids == 1).sum())} columns have a single category — exact, not allocated")

    # 4. allocate
    share = cat["count"] / cat.groupby("home")["count"].transform("sum")
    cat = cat.assign(share=share)

    f = fine[fine["code"].isin(ok)][
        ["geo_id", "geo_level", "geo_name", "code", "count", "basis", "year", "source_id"]]
    out = f.merge(cat[["code", "source_category", "home", "share", "exact"]],
                  left_on="code", right_on="home", suffixes=("_fine", ""))
    out["count"] = out["count"] * out["share"]

    codekey = args.code if args.hierarchy == "prefix" else "cat"
    out["note"] = out.apply(
        lambda r: (f"level=leaf; {codekey}={r['code']}; "
                   + ("derivation=exact_single_child; " if r["exact"] else
                      f"derivation=allocated; structure_geo={args.coarse}; "
                      f"structure_share={r['share']:.6f}; ")
                   + f"parent_column={r['home']}"), axis=1)
    out["tier"] = out["exact"].map({True: "measured", False: "derived"})

    keep = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "tier", "note"]
    out = out[keep].sort_values(["geo_id", "source_category"])

    dest = NORM / f"{args.source}_{args.fine}_allocated.csv"
    out.to_csv(dest, index=False)

    # Every bug in this step so far has lost data while reporting success — a self-parent
    # treated as an aggregate, a whole-run validation mode that could only see some columns,
    # an output name that let one run overwrite another. Comparing in-to-out cannot catch any
    # of them, because a dropped column leaves both sides. So the guard is against the SOURCE:
    # what fraction of the fine geography's people came out the other end.
    tot_in = fine[fine["code"].isin(ok)]["count"].sum()
    tot_out = out["count"].sum()
    source_total = fine["count"].sum()
    kept = tot_out / source_total if source_total else 0
    if kept < 0.999:
        lost = sorted(set(fine_tot.index) - ok)
        print(f"\n  !! only {kept:.1%} of the {args.fine} population survived "
              f"({source_total - tot_out:,.0f} people in {len(lost)} unallocated columns)")
        for c in lost[:8]:
            print(f"       {c}: {fine_tot[c]:,.0f}")
    print(f"\n  wrote {len(out):,} rows -> {dest.name}")
    print(f"  {kept:.1%} of the {args.fine} population allocated")
    print(f"  {out['source_category'].nunique()} categories at {args.fine} "
          f"(was {len(ok)})")
    print(f"  people in : {tot_in:,.0f}")
    print(f"  people out: {tot_out:,.0f}   (diff {tot_out - tot_in:+,.0f})")
    print(f"  derived rows: {(out['tier'] == 'derived').sum():,} "
          f"({out.loc[out['tier'] == 'derived', 'count'].sum():,.0f} people)")


if __name__ == "__main__":
    main()
