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

# cc -> mapping module. **DISCOVERED, not listed** — taxonomy/registry.py walks
# taxonomy/<cc><YYYY>.py. This was a hand-maintained dict, duplicated in coverage.py, and
# keeping the two in step was forgotten three times on 2026-09-06 alone.
# drawn_only=False so a mapping can be checked BEFORE its country is registered in
# countries.py, which is the order the two usually get written in.
def _module_for(cc):
    from registry import SPECIAL, discover
    reg = discover(drawn_only=False)
    if cc in SPECIAL:
        raise SystemExit(
            f"{cc!r} is handled specially and this tool does not cover it — "
            f"{SPECIAL[cc]}  (taxonomy/registry.py SPECIAL). For the US, build_tree.py "
            f"validates usrc2020.py; for Canada, nothing does yet.")
    if cc not in reg:
        raise SystemExit(
            f"no taxonomy module for {cc!r}: expected taxonomy/{cc}<YYYY>.py. "
            f"Known: {', '.join(sorted(reg))}")
    return reg[cc]

# A country whose drawn tier is more than one geo_level, so the default "the level with the
# most units" would silently tally only part of it. Ghana's 272 drawn units are 255 plain
# districts plus the 17 sub-metros that REPLACE six metropolitan parents; picking `district`
# alone reports 255 units and 1.7M too few people, and nothing about that looks wrong.
DEFAULT_LEVELS = {# Austria's drawn tier is 2,358 Gemeinden PLUS Vienna's 23 Gemeindebezirke.
                  # Vienna is a single Gemeinde, so `gemeinde` alone reports 2,358 units and
                  # 1,550,123 too few people -- the whole capital, and the most distinctive
                  # fifth of the country, missing with nothing looking wrong.
                  "at": ["gemeinde", "gemeindebezirk"],
                  "gh": ["district", "submetro"],
                  # Armenia is the other failure mode: the drawn tier is the ELEVEN marzes,
                  # and am.csv also carries the national religion-by-ethnicity block, which
                  # is twelve ethnicities and therefore wins "the level with the most
                  # units". Tallying it reports the country correctly by accident and the
                  # geography not at all.
                  "am": ["marz"],
                  # Indonesia's drawn tier is decided per unit, not by rule: a regency's
                  # kecamatan REPLACE it where they sum to it exactly in EVERY
                  # category (403 of 492) and the regency is drawn where they
                  # do not (89). `regency_covered` and
                  # `kecamatan_partial` are the two record-only halves of that split and
                  # must not be tallied -- either one would double-count.
                  "id": ["kecamatan", "regency"],
                  # Pakistan's drawn tier is NOT the finest one in its file. pk.csv carries
                  # 585 tehsils and 155 districts, and countries.py draws districts because
                  # PBS publishes religion at district and not below (spec §14.4). The
                  # default "level with the most units" would tally the tehsils, which are
                  # a tier this map deliberately does not draw.
                  "pk": ["district"],
                  # Israel's drawn tier is two levels by construction: CBS splits 142
                  # localities into statistical areas and publishes the other 1,043 whole,
                  # so a locality that HAS statistical areas is not drawn itself (drawing
                  # both would double the country). The default "level with the most units"
                  # would take `statarea` alone and lose every rural locality in the state.
                  "il": ["statarea", "locality"]}


def main():
    import pandas as pd
    from branches import BRANCHES

    ap = argparse.ArgumentParser()
    ap.add_argument("cc")
    ap.add_argument("--level", default=None,
                    help="geo_level(s) to tally over, comma-separated (default: the "
                         "country's drawn tier, else the level with most units)")
    args = ap.parse_args()

    mod = importlib.import_module(_module_for(args.cc))
    resolve = mod.resolve
    excluded = set(getattr(mod, "EXCLUDED", {}))
    # A mapping may normalise labels before lookup (Poland strips GUS's trailing "w tym:"),
    # so EXCLUDED has to be tested against the same key the module resolves on -- otherwise
    # a deliberately excluded universe row is reported as an unmapped category.
    key = getattr(mod, "_key", lambda c: " ".join(str(c).split()))

    # keep_default_na=False because a source category can BE one of pandas' NA strings.
    # The Philippines has a category literally called "None" -- 43,931 people reporting no
    # religion -- and with default parsing it arrives as NaN, fails to resolve, and is
    # dropped by countries.py without a word. Read the file the way the data is written.
    df = pd.read_csv(os.path.join(ROOT, "data", "normalized", f"{args.cc}.csv"),
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    if args.level is not None:
        levels = [s.strip() for s in args.level.split(",") if s.strip()]
    elif args.cc in DEFAULT_LEVELS:
        levels = DEFAULT_LEVELS[args.cc]
    else:
        levels = [df.groupby("geo_level")["geo_id"].nunique().idxmax()]
    unknown_levels = sorted(set(levels) - set(df["geo_level"].unique()))
    if unknown_levels:
        raise SystemExit(f"{args.cc}.csv has no geo_level {unknown_levels} -- it has "
                         f"{sorted(df['geo_level'].unique())}")
    sub = df[df["geo_level"].isin(levels)]
    print(f"{args.cc}: level {'+'.join(levels)}, {sub['geo_id'].nunique():,} units, "
          f"{len(sub):,} rows")

    # ---- 1. coverage
    cats = sorted(df["source_category"].unique())
    unmapped = [c for c in cats if resolve(c) is None and key(c) not in excluded]
    print(f"\n  {len(cats)} distinct source categories")
    print(f"  {'OK ' if not unmapped else 'BAD'} unmapped and not EXCLUDED: {len(unmapped)}")
    for c in unmapped:
        n = df[(df["source_category"] == c) & df["geo_level"].isin(levels)]["count"].sum()
        print(f"      {n:>10,}  {c}")

    # ---- 2. every target node exists
    # branches.py alone is NOT the set of nodes. Leaves are contributed by source maps --
    # usrc2020.py creates christianity.oriental.ethiopian for a US diaspora of 66,000 --
    # and another country may legitimately map onto one: et2007.py sends Ethiopia's 32.1M
    # Orthodox to exactly that leaf. Checking against BRANCHES rejected a correct mapping,
    # so the authority is the built tree, with branches.py as the fallback before it exists.
    node_ids = {b[0] for b in BRANCHES}
    tree = os.path.join(ROOT, "taxonomy", "religions.json")
    if os.path.exists(tree):
        import json
        with open(tree, encoding="utf-8") as fh:
            node_ids |= {n["id"] for n in json.load(fh)["nodes"]}
    else:
        print("  note: religions.json not built, checking targets against branches.py only")
    targets = {resolve(c) for c in cats} - {None}
    unknown = sorted(t for t in targets if t not in node_ids)
    print(f"\n  {len(targets)} distinct target nodes")
    print(f"  {'OK ' if not unknown else 'BAD'} targets that are not nodes of the tree: "
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
