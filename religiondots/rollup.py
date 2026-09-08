"""
Where each country's inferred dots fall back to — spec §7a-i-1.

WHAT THIS IS FOR. `inferred dots: not shown` does not delete a derived dot; it redraws it at
the nearest level its own country MEASURED (§7a-i). The first build worked that out by
walking the religion tree for an ancestor the country has a measured row on, and that is the
wrong question. **The thing a source measured is its own COLUMN**, and a column's node is not
always an ancestor of the leaves split out of it:

    Hungary   `christianity.baptist` came out of a settlement column called
              `Other Christian denomination`. The tree walk looks for a Baptist ancestor and
              finds nothing; the honest answer, `christianity`, is sideways from it.
    Ireland   126,658 Anglicans came out of a Small Area column called `Other religion`.
              `christianity.anglican` is nowhere under `other.ie`.
    UK        22.1 million people answered `No religion` at their own Output Area. The node
              is a ROOT: there is no ancestor at all, so every one of them left the map.

That last line is why this file exists. Before it, 185.9 million people vanished when a
reader asked to see what was counted, and **most of them had been counted** — the census put
them in a cell at the drawn unit and only the sub-category came from somewhere coarser.

THE ARITHMETIC IS EXACT, which is the argument for doing it this way rather than by
judgement. `allocate.py` normalises its shares within a column, so a column's leaves sum
back to the column's own measured total. Rolling every leaf to the column's node therefore
reconstructs a number the source published, not an estimate of one.

THE RULE THAT KEEPS THE HONEST EMPTINESS HONEST: **the column must have been measured at the
SAME UNIT.** True by construction for `allocate.py` (the fine table's column is at the fine
unit) and for `br_rescale.py` (a 2022 município total). False for Switzerland, whose measured
number is a CANTON total spread over communes; false for Israel's lumped localities, split
from district composition; false for China, where nothing was counted at any level. Those
three keep vanishing, and they should — `ch`, `cn` and `il` are the map saying so.

WHY A TABLE AND NOT A PER-DOT FIELD. The relation turns out to be single-valued for all but a
handful of (country, node) pairs, so it fits in `counts.json` beside `covers` — no new tile
attribute, no new buffer attribute, and `tiles.py --refresh-meta` can ship a mapping fix
without re-encoding 36,000 tiles. Where a node IS fed by two columns with different targets
the majority of the people wins and the split is printed below, because a silent tie-break on
a claim about what was counted is exactly the kind of thing that should be visible.

Run:  python rollup.py              every drawn country -> data/processed/rollup.json
      python rollup.py --countries br,uk,ca      just these, merged into what is there
      python rollup.py --dry-run    print the table and write nothing
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = HERE / "data" / "processed" / "rollup.json"


# An adapter writes this into `roll` to mean "this country measured NO ancestor of this node
# at this unit, so do not walk the tree for one". Distinct from an absent roll, which means
# "I did not record a column; walk". Added 2026-09-08 with Angola; see `table_for`.
NOWHERE = "__nowhere__"


def parent_of(node):
    i = node.rfind(".")
    return node[:i] if i > 0 else None


def table_for(cc, df):
    """node -> the node its derived dots roll up to, for one country.

    Two sources, in order. A `roll` the adapter recorded is the source's own column and
    wins. Everything else falls back to the ancestor walk §7a-i shipped with, which is still
    right wherever a country measured a coarser level of the same branch — Israel's Haredim
    under Judaism, which is the case that prompted the whole control.

    **AND `roll == NOWHERE` MEANS THE WALK MUST NOT RUN**, which nothing could say before
    2026-09-08. `measured` is a set for the WHOLE COUNTRY, so a country that counted a
    branch in most of its territory and not in the rest gets the wrong answer from the walk
    in the part that did not: Angola's Uíge and Moxico Leste have their Assembleias de Deus
    filled in from a province total (sources/ao.py `fill()`), the walk finds
    `christianity.pentecostal` measured — in the nineteen OTHER provinces, from a body those
    two never printed — and 116,174 people who should disappear under `inferred dots: not
    shown` would draw as Pentecostals instead. That is exactly the failure §7a-i-1 added the
    column control for, one level up. An adapter that knows a derived row has no measured
    ancestor AT ITS OWN UNIT says so with this sentinel and the walk is skipped.
    """
    if "tier" not in df.columns:
        return {}, [], 0.0, 0.0
    df = df.assign(tier=df["tier"].fillna("measured"))
    measured = set(df.loc[df["tier"] == "measured", "node"])
    der = df[df["tier"] == "derived"]
    if der.empty:
        return {}, [], 0.0, 0.0

    # People per (node, target). `roll` may be absent entirely for a country whose adapter
    # does not record it, which is not an error — it means "walk the tree".
    votes = defaultdict(lambda: defaultdict(float))
    nowhere = set()
    if "roll" in der.columns:
        for node, roll, n in zip(der["node"], der["roll"], der["count"]):
            if roll == NOWHERE:
                nowhere.add(node)
            elif isinstance(roll, str) and roll:
                votes[node][roll] += float(n)

    out, conflicts = {}, []
    for node in sorted(set(der["node"])):
        if node in nowhere and node not in votes:
            continue                     # the adapter says there is no measured ancestor
        cand = votes.get(node)
        if cand:
            ranked = sorted(cand.items(), key=lambda kv: -kv[1])
            out[node] = ranked[0][0]
            if len(ranked) > 1:
                conflicts.append((node, ranked))
            continue
        # no column recorded: the ancestor walk, unchanged from §7a-i
        a = parent_of(node)
        while a and a not in measured:
            a = parent_of(a)
        if a:
            out[node] = a

    # A target that is itself the node changes nothing about where the dot draws — it only
    # says the dot was counted there. Kept in the table anyway: the viewer needs to tell
    # "rolls up to itself, so it stays" from "no target, so it goes", and dropping the
    # entry would collapse the two.
    rolled = der[der["node"].isin(out)]["count"].sum()
    lost = der["count"].sum() - rolled
    return out, conflicts, float(rolled), float(lost)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", default="",
                    help="comma-separated; default every country in countries.py. Named "
                         "countries are merged into the existing file rather than "
                         "replacing it, so a one-country mapping fix costs one country.")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    import countries as C

    want = [c.strip() for c in args.countries.split(",") if c.strip()]
    ccs = [cc for cc in sorted(C.COUNTRIES) if not want or cc in want]

    existing = {}
    if OUT.exists() and want:
        existing = json.loads(OUT.read_text(encoding="utf-8"))

    W = 14
    print(f"{'cc':<4}{'derived':>{W}}{'rolls up':>{W}}{'still gone':>{W}}  targets")
    print("-" * 96)
    tables, all_conflicts = dict(existing), []
    tot_roll = tot_lost = 0.0
    for cc in ccs:
        try:
            df = C.COUNTRIES[cc]["counts"]()
        except Exception as e:                                      # noqa: BLE001
            print(f"{cc:<4}  !! counts() failed: {str(e)[:70]}")
            continue
        t, conflicts, rolled, lost = table_for(cc, df)
        if t:
            tables[cc] = t
        elif cc in tables:
            del tables[cc]
        all_conflicts += [(cc, *c) for c in conflicts]
        tot_roll += rolled
        tot_lost += lost
        if rolled or lost:
            top = sorted({v for v in t.values()})
            print(f"{cc:<4}{rolled + lost:>{W},.0f}{rolled:>{W},.0f}{lost:>{W},.0f}  "
                  f"{len(t)} nodes -> {len(top)} targets")
    print("-" * 96)
    print(f"{'ALL':<4}{tot_roll + tot_lost:>{W},.0f}{tot_roll:>{W},.0f}{tot_lost:>{W},.0f}")

    if all_conflicts:
        print(f"\n{len(all_conflicts)} node(s) fed by more than one column — the majority of "
              "the people wins, and here is what it beat:")
        for cc, node, ranked in all_conflicts:
            parts = ", ".join(f"{t} {v:,.0f}" for t, v in ranked)
            print(f"  {cc} {node:<28} -> {ranked[0][0]}   ({parts})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tables), encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(tables)} countries, "
          f"{sum(len(v) for v in tables.values())} nodes)")
    print("Now: python tiles.py --refresh-meta   (carries it into counts.json; no retile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
