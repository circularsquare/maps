"""Check a country's taxonomy mapping against its normalised data and against branches.py.

    python tools/check_mapping.py pl

Answers the three questions that go wrong quietly:

  1. Does every source category in data/normalized/<cc>.csv resolve, or is it EXCLUDED?
     An unmapped category is not an error anywhere downstream — countries.py drops rows
     whose node is NaN — so it silently removes people from the map.
  2. Does every node the mapping points at actually exist in branches.py?  A typo in a
     path produces a node the viewer has never heard of, which greys out.
  3. How many people land on each node, so the result can be eyeballed.

build_tree.py validates usrc2020.py this way; every other country's mapping had nothing.
"""

import argparse
import importlib
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

MODULES = {"pl": "pl2021", "ro": "ro2021", "ee": "ee2021", "hr": "hr2021", "cz": "cz2021", "au": "au2021", "br": "br2010",
           "ie": "ie2022", "mx": "mx2020", "nz": "nz2023", "uk": "uk2021",
           "ca": "ca2021", "in": "in2011", "de": "de2022", "hu": "hu2022"}


def main():
    import pandas as pd
    from branches import BRANCHES

    ap = argparse.ArgumentParser()
    ap.add_argument("cc")
    ap.add_argument("--level", default=None,
                    help="geo_level to tally over (default: the one with most units)")
    args = ap.parse_args()

    mod = importlib.import_module(MODULES.get(args.cc, f"{args.cc}2021"))
    resolve = mod.resolve
    excluded = set(getattr(mod, "EXCLUDED", {}))
    # A mapping may normalise labels before lookup (Poland strips GUS's trailing "w tym:"),
    # so EXCLUDED has to be tested against the same key the module resolves on -- otherwise
    # a deliberately excluded universe row is reported as an unmapped category.
    key = getattr(mod, "_key", lambda c: " ".join(str(c).split()))

    df = pd.read_csv(os.path.join(ROOT, "data", "normalized", f"{args.cc}.csv"),
                     dtype={"geo_id": str}, low_memory=False)

    level = args.level
    if level is None:
        level = df.groupby("geo_level")["geo_id"].nunique().idxmax()
    sub = df[df["geo_level"] == level]
    print(f"{args.cc}: level {level!r}, {sub['geo_id'].nunique():,} units, "
          f"{len(sub):,} rows")

    # ---- 1. coverage
    cats = sorted(df["source_category"].unique())
    unmapped = [c for c in cats if resolve(c) is None and key(c) not in excluded]
    print(f"\n  {len(cats)} distinct source categories")
    print(f"  {'OK ' if not unmapped else 'BAD'} unmapped and not EXCLUDED: {len(unmapped)}")
    for c in unmapped:
        n = df[(df["source_category"] == c) & (df["geo_level"] == level)]["count"].sum()
        print(f"      {n:>10,}  {c}")

    # ---- 2. every target node exists
    branch_ids = {b[0] for b in BRANCHES}
    targets = {resolve(c) for c in cats} - {None}
    unknown = sorted(t for t in targets if t not in branch_ids)
    print(f"\n  {len(targets)} distinct target nodes")
    print(f"  {'OK ' if not unknown else 'BAD'} targets not declared in branches.py: "
          f"{len(unknown)}")
    for t in unknown:
        srcs = [c for c in cats if resolve(c) == t]
        print(f"      {t}   <- {len(srcs)} categories, e.g. {srcs[0]!r}")

    # ---- 3. where the people land
    sub = sub.copy()
    sub["node"] = sub["source_category"].map(resolve)
    drawn = sub[sub["node"].notna()]
    by_node = drawn.groupby("node")["count"].sum().sort_values(ascending=False)
    print(f"\n  {drawn['count'].sum():,} people on {len(by_node)} nodes:")
    for node, n in by_node.items():
        print(f"    {n:>11,}  {node}")

    # NOT a population: the excluded rows are nested universes (Ogółem contains
    # Udzielający contains należący), so this sum counts the same people several times.
    # It is here to show WHICH categories are dropped, not how many people they are.
    undrawn = sub[sub["node"].isna()].groupby("source_category")["count"].sum()
    print(f"\n  {len(undrawn)} categories resolve to nothing — nested universes and "
          f"refusals, so these overlap and must not be summed:")
    for cat, n in undrawn.sort_values(ascending=False).items():
        print(f"    {n:>11,}  {cat}")

    if unmapped or unknown:
        raise SystemExit("mapping check FAILED")


if __name__ == "__main__":
    main()
