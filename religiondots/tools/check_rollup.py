"""Which countries gain from §7a-i's roll-up, and which still empty when it is applied.

WHAT THIS IS FOR. `inferred dots: not shown` no longer deletes a non-measured dot; it redraws
it at the nearest level its own country MEASURED, and deletes it only where nothing above it
was measured (spec §7a-i). Whether a country benefits is therefore a property of its own
mapping — of whether its derived rows sit UNDER something it also counted — and that is not
something you can tell by reading `countries.py`. This reports it per country.

**MOSTLY FIXED 2026-09-07 — READ THIS BEFORE THE THREE CASES BELOW.** The scan that produced
them was right about the sizes and wrong about the cause. Case C is not about PLACEMENT:
`allocate.py` never moves anybody between units, and the UK's 22.1 million `No religion`
answers were counted at their own Output Area. What was inferred is the sub-category, and the
reason they vanished is that **the roll-up walked the religion tree while the thing the source
measured was its own COLUMN** — and a column's node need not be an ancestor of what was split
out of it. `unaffiliated` is a root with no ancestor at all; Hungary's Baptists came out of a
column called `Other Christian denomination`, which is sideways from them.

So the fix was neither in the viewer alone nor a change to every adapter: each mapping module
names its fine columns in a `COLUMNS` dict, `countries.py` attaches a `roll` per row, and
`rollup.py` builds the per-country table the viewer now reads first. **185.9M orphaned became
39.3M**, and what remains is all of one kind: China, Switzerland, Israel's lumped localities
and Kosovo, none of which measured anything at the unit their dots are drawn on. The single
exception is New Zealand's `Māori Religions` cell, 65,151, left out on purpose because the
tree has no node for it — see `nz2023.COLUMNS`. This file now reports against that table, so
what it prints is what the map does.

THREE KINDS OF `derived` HIDE IN ONE WORD, and telling them apart is the whole job:

  A. **Derived by BRANCH, under something the country measured.** Israel counted Jews and
     split them by observance. The roll-up already handles it: the dot falls back to Judaism.
     Shows up in the `rolls up` column. Nothing to do.

  B. **Derived by BRANCH, with nothing measured above it.** China derives every row from
     ethnicity and counted nobody, so there is no level to fall back to and the country
     correctly empties. Its emptiness IS the honest answer. Shows up as `ORPHANED`.

  C. **Derived by SUB-CATEGORY, under a column that is not an ancestor.** The node is what
     the source measured or very close to it; what was derived is which finer answer inside a
     cell the source published AT THIS UNIT. The UK's 22.1M and Canada's 12.4M `unaffiliated`
     are census counts of people who ticked "no religion" in their own Output Area or CSD, and
     `allocate.py` only decided which of them are Agnostic or Humanist. Since 2026-09-07 they
     roll to the column's own node and stay. **This was 186 million people and is now the
     `roll` column's job**; anything still here in this shape means a `COLUMNS` entry is
     missing, or was left out on purpose with a comment saying why.

  **The tell between B and C is whether the source published a cell holding these people at
  the unit they are drawn on.** If it did, that cell is the answer and belongs in `COLUMNS`.
  If it did not — China derives every row from an ethnicity table and counted nobody's
  religion anywhere — there is nothing to fall back to and the emptiness is the finding.

  **`modelled` never rolls up and that is deliberate** (§7b): nobody was counted at any level,
  so there is nothing to fall back to. Reported separately; not a defect.

**What is left, and why each one is left.** Switzerland spreads a CANTON total over communes,
so its column is not measured at the drawn unit and the rule does not reach it. Israel's
lumped localities are split from district composition, the same shape. China measured nothing.
Mexico's `sin religión o sin adscripción religiosa` and New Zealand's two odd Stats NZ groups
have no honest node to roll to and are left out on purpose — see the comments in `mx2020.py`
and `nz2023.py`, which are where those decisions live.

The check is on the NORMALISED files plus each country's taxonomy mapping, so it needs no
buffers and no viewer — the same relation the viewer computes at run time from the tiles.

Run:  python tools/check_rollup.py            every drawn country
      python tools/check_rollup.py il cn us   just these
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rollup import NOWHERE                                        # noqa: E402


def parent_of(node):
    i = node.rfind(".")
    return node[:i] if i > 0 else None


def main():
    import pandas as pd
    import countries as C
    from registry import discover

    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    reg = discover()
    ccs = [cc for cc in sorted(C.COUNTRIES) if not want or cc in want]

    W = 14      # the US is 326,813,748 and the columns must not run together
    print(f"{'cc':<4}{'people':>{W}}{'measured':>{W}}{'derived':>{W}}{'modelled':>{W}}"
          f"{'rolls up':>{W}}{'ORPHANED':>{W}}  nodes with no measured ancestor")
    print("-" * 130)

    totals = defaultdict(float)
    flagged = []
    for cc in ccs:
        try:
            df = C.COUNTRIES[cc]["counts"]()
        except Exception as e:                          # noqa: BLE001
            print(f"{cc:<4}  !! counts() failed: {str(e)[:80]}")
            continue
        if "tier" not in df.columns:
            df = df.assign(tier="measured")
        df["tier"] = df["tier"].fillna("measured")

        by = df.groupby(["node", "tier"])["count"].sum()
        measured_nodes = {n for (n, t) in by.index if t == "measured"}

        # The adapter's own record of which column each derived row came out of, which is
        # what the viewer reads first. A country whose module has no COLUMNS contributes
        # nothing here and falls through to the ancestor walk below, exactly as the viewer
        # does — the two have to agree or this file is checking something else.
        # `rollup.NOWHERE` is an adapter saying there is no measured ancestor AT THIS UNIT,
        # so the walk below must not run for that node. Excluded here rather than treated
        # as a target, because a country-level `measured_nodes` cannot see a gap that
        # covers only part of a country — Angola's Uíge and Moxico Leste, whose filled
        # bodies would otherwise find `christianity.pentecostal` measured in the nineteen
        # provinces that printed it. Such nodes are ORPHANED, which is the truthful column.
        col_target = {}
        nowhere = set()
        if "roll" in df.columns:
            d = df[(df["tier"] == "derived") & df["roll"].notna()]
            nowhere = set(d.loc[d["roll"] == NOWHERE, "node"])
            d = d[d["roll"] != NOWHERE]
            if not d.empty:
                v = d.groupby(["node", "roll"])["count"].sum()
                for (node, target), n in v.items():
                    if node not in col_target or n > col_target[node][1]:
                        col_target[node] = (target, n)

        per = defaultdict(float)
        orphan_nodes = defaultdict(float)
        rolls = 0.0
        for (node, tier), n in by.items():
            per[tier] += n
            if tier != "derived":
                continue
            if node in col_target:
                rolls += n
                continue
            if node in nowhere:
                orphan_nodes[node] += n
                continue
            anc = parent_of(node)
            while anc and anc not in measured_nodes:
                anc = parent_of(anc)
            if anc:
                rolls += n
            else:
                orphan_nodes[node] += n

        tot = float(df["count"].sum())
        orph = sum(orphan_nodes.values())
        for k, v in per.items():
            totals[k] += v
        totals["all"] += tot
        totals["rolls"] += rolls
        totals["orphan"] += orph

        names = ", ".join(f"{n} ({v:,.0f})" for n, v in
                          sorted(orphan_nodes.items(), key=lambda kv: -kv[1])[:3])
        print(f"{cc:<4}{tot:>{W},.0f}{per['measured']:>{W},.0f}{per['derived']:>{W},.0f}"
              f"{per['modelled']:>{W},.0f}{rolls:>{W},.0f}{orph:>{W},.0f}  {names[:40]}")
        if orph > 0.02 * tot:
            flagged.append((cc, orph, tot, dict(orphan_nodes)))

    print("-" * 130)
    print(f"{'ALL':<4}{totals['all']:>{W},.0f}{totals['measured']:>{W},.0f}"
          f"{totals['derived']:>{W},.0f}{totals['modelled']:>{W},.0f}"
          f"{totals['rolls']:>{W},.0f}{totals['orphan']:>{W},.0f}")

    print(f"\n{len(flagged)} country(ies) where derived-and-orphaned exceeds 2% of the "
          "country — these are the ones to look at:")
    for cc, orph, tot, nodes in sorted(flagged, key=lambda t: -t[1] / t[2]):
        print(f"\n  {cc}: {orph:,.0f} of {tot:,.0f} ({100 * orph / tot:.1f}%) disappear when "
              "inferred divisions are not shown")
        for n, v in sorted(nodes.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      {n:<34} {v:>12,.0f}")
        roots = [n for n in nodes if "." not in n]
        if roots:
            print(f"      NOTE: {len(roots)} of these are ROOT nodes "
                  f"({', '.join(sorted(roots)[:5])}). A root has no ancestor to walk to, so "
                  "the tree can never save it — if these people were counted, the only thing "
                  "that can say so is a COLUMNS entry naming the cell they were counted in.")
        print("      ASK: did the source publish a cell holding these people AT THE UNIT they "
              "are drawn on?\n      If yes, that cell's node belongs in the mapping module's "
              "COLUMNS and this stops being a finding. If no — the number was measured "
              "somewhere coarser, or not at all — the emptiness is the honest answer and "
              "should be left, with a comment saying so.")
    if not flagged:
        print("  none — every derived row in every country has somewhere to fall back to.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
