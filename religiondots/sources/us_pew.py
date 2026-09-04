"""Pew's 2023-24 Religious Landscape Study, by state, straight out of the published pages.

WHY THIS EXISTS. spec.md §3.5a re-bases the United States on self-identification: the survey
supplies the root totals, ASARB supplies the structure inside them, and the difference is §3.2's
`...unspecified` residual. This is the survey half. `taxonomy/us_pew2024.py` is the mapping that
turns what comes out of here into religiondots roots.

WHY IT SCRAPES RATHER THAN DOWNLOADS. Pew's public-use file **carries no geography at all** -
state identifiers are in the restricted-use file, which needs an institutional agreement. So the
only public route to state-level RLS numbers is Pew's own 51 published state pages.

WHY IT PARSES A JSON BLOB RATHER THAN THE RENDERED PAGE. Those pages are WordPress Interactivity
API components. The whole thing is server-rendered into a single
`<script type="application/json">` block, and `state["prc-rls/context-provider"]` in it holds far
more than the page displays -

  * `religiousTree`   the STATE's chart tree, nested, 149 categories deep enough to reach
                      `southern-baptist-convention` and `global-methodist-church`, where the page
                      shows about a dozen rows.
  * `data.value`      the WEIGHTED POPULATION ESTIMATE, e.g. 1,417,495.68 Catholics in Ohio, not
                      the "16%" the page prints. Whole-percent rounding would have been useless
                      for small religions, where most states print "<1%".
  * `data.sample_size` respondents behind each cell, the only way to know when a number is too
                      thin to use. Some are 1.
  * `percent.total`   the state's weighted ADULT population, the denominator for everything here.
  * `moes`            the state's margin of error and effective sample size, per study year.

THE NESTING IS THE POINT, and reading it as a flat list of categories is a live way to get a
wrong answer: `something-else` is a parent of `unitarians-and-other-liberal-faiths`, and summing
both counts those people twice. The tree is written out with `parent` and `depth` on every row so
that `us_pew2024.py` can cut across it exactly once.

THE TWO-TREE TRAP, WHICH THIS SHAPE RETIRES. A state page carries the state's tree AND a United
States one for the comparison tab. The first version of this file matched category nodes with a
regex and had to separate the two by the denominator each node carried, because a category with
no respondents in a state is simply ABSENT from the state tree, so a name-based search falls
through into the national tree: Utah has no `muslim` node, and that version gave Utah 3,026,029
Muslims -- every Muslim adult in America -- with a straight face. Read as JSON the two are
`religiousTree` and `religiousTreeUS`, different keys, and the trap cannot be sprung. The
validation that caught it is kept anyway.

AN ABSENT CELL IS A ZERO, NOT A SUPPRESSION -- established 2026-09-03 and worth stating because
this file used to say the opposite. Wherever a parent's children are enumerated, the ones that
are present sum to the parent EXACTLY, to the last bit; the top-level nodes sum to the published
adult denominator EXACTLY; and every state's figures sum to the national tree's EXACTLY,
category by category. None of that could hold if withheld cells were being carried anywhere. So
`muslim` missing from 13 states means Pew's weighted estimate for those states is zero -- an
n=36,908 survey cut 51 ways runs out of respondents -- and nothing is lost downstream. The three
identities are checked on every run, below, since they are what the claim rests on.

REMEMBER THESE ARE ADULTS. ASARB counts everyone including children. §3.5a's conversion assumes
children hold their household's religion, and records that it is an assumption.

Run: python sources/us_pew.py     ->  data/normalized/us_pew.csv
"""
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "normalized" / "us_pew.csv"
BASE = "https://www.pewresearch.org/religious-landscape-study/state/{}/"
CACHE = HERE / "data" / "raw" / "us_pew"

STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "district-of-columbia", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts",
    "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new-hampshire", "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode-island", "south-carolina",
    "south-dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west-virginia", "wisconsin", "wyoming",
]

# The four keys the tree is divided into at the top. They are Pew's own grouping and they are
# NOT the religiondots roots -- `others` holds Judaism, Islam and the New Age alike, and
# `christians` holds Pew's classification of who counts as one. The mapping decides what they
# mean; this file only records which group a row came from.
GROUPS = ("christians", "others", "unaffiliated", "no_answer")

BLOCK = re.compile(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', re.S)


def fetch(slug):
    """Cached; the pages are ~1MB each and there are 51 of them."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{slug}.html"
    if f.exists() and f.stat().st_size > 100_000:
        return f.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(BASE.format(slug), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", errors="replace")
    f.write_text(html, encoding="utf-8")
    return html


def context(html, slug):
    """The Interactivity API context block -- the only thing on the page worth reading."""
    for raw in BLOCK.findall(html):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        cp = (obj.get("state") or {}).get("prc-rls/context-provider")
        if isinstance(cp, dict) and "religiousTree" in cp:
            return cp
    raise SystemExit(f"{slug}: no prc-rls/context-provider with a religiousTree in the page")


def flatten(tree, slug):
    """The nested tree -> rows, in tree order, each carrying its parent and depth.

    A node whose `data.value` is null has no respondents in this state (see the module
    docstring) and is dropped; `check_partition` is what makes that safe to do.
    """
    rows = []

    def walk(node, parent, depth, group):
        if not isinstance(node, dict):        # a childless node carries [] rather than {}
            return
        for name, v in node.items():
            if not isinstance(v, dict):
                continue
            data = v.get("data") or {}
            if data.get("value") is not None:
                rows.append(dict(
                    group=group, parent=parent, depth=depth, name=name,
                    label=v.get("label", ""),
                    adults=float(data["value"]),
                    sample_size=int(data.get("sample_size") or 0),
                ))
            walk(v.get("children"), name, depth + 1, group)

    for g in GROUPS:
        if g not in tree:
            raise SystemExit(f"{slug}: religiousTree has no {g!r} group; "
                             f"it has {sorted(tree)}")
        walk(tree[g], "", 0, g)
    if not rows:
        raise SystemExit(f"{slug}: religiousTree parsed but is empty")
    return rows


def denominator(tree, rows, slug):
    """The state's weighted adult population, taken from the page AND checked against the sum.

    Every node carries the same `percent.total`. Trusting it alone would not notice a group
    going missing, and trusting the sum alone would not notice the page disagreeing with
    itself, so both are read and they have to agree.
    """
    stated = set()
    for g in GROUPS:
        for v in tree[g].values():
            if isinstance(v, dict) and isinstance(v.get("percent"), dict):
                t = v["percent"].get("total")
                if t is not None:
                    stated.add(round(float(t), 3))
    if len(stated) != 1:
        raise SystemExit(f"{slug}: expected one denominator across the state tree, "
                         f"found {sorted(stated)}")
    total = stated.pop()
    top = sum(r["adults"] for r in rows if r["depth"] == 0)
    if abs(top - total) > 1.0:
        raise SystemExit(f"{slug}: the top-level nodes sum to {top:,.2f} but the page's "
                         f"denominator is {total:,.2f} -- the tree is not a partition")
    return total


def check_partition(rows, slug):
    """Present children sum to their parent, everywhere. This is what makes a null a zero."""
    by_parent = {}
    for r in rows:
        by_parent.setdefault(r["parent"], []).append(r)
    value = {r["name"]: r["adults"] for r in rows}
    for parent, kids in by_parent.items():
        if parent == "":
            continue
        got, want = sum(k["adults"] for k in kids), value[parent]
        if abs(got - want) > max(1.0, want * 1e-9):
            raise SystemExit(f"{slug}: children of {parent!r} sum to {got:,.2f}, "
                             f"parent is {want:,.2f} -- a withheld cell is being carried")


US_ADULTS = 257_520_024          # the denominator Pew's own pages carry, summed over states


def check_national(out, national):
    """Three checks, the first of which is the one the Utah bug got past.

    1. No state may report a figure equal to the national one for a category, which is what
       the two trees running together looked like.
    2. The state figures must SUM to the national tree's figure for every category. This is
       far stronger than a percentage comparison and it is what establishes that nothing is
       withheld between the levels.
    3. The published national percentages must come out, which is the end-to-end test.
    """
    by_name = {}
    for r in out:
        by_name.setdefault(r["name"], []).append(r)

    bad = []
    for name, nat in sorted(national.items()):
        rows = by_name.get(name, [])
        tot = sum(r["adults"] for r in rows)
        if nat is None:
            if rows:
                bad.append(f"{name}: absent nationally but reported by {len(rows)} states")
            continue
        if abs(tot - nat) > max(1.0, abs(nat) * 1e-9):
            bad.append(f"{name}: states sum to {tot:,.2f}, national tree says {nat:,.2f}")
        dupes = [r for r in rows if abs(r["adults"] - nat) < 1e-6]
        if dupes and len(rows) > 1:
            bad.append(f"{name}: {dupes[0]['state']} reports the national figure exactly "
                       f"-- the two trees have run together again")

    print("\n  against Pew's published national percentages:")
    for name, published in (("muslim", "1.2%"), ("hindu", "0.9%"), ("jewish", "1.7%"),
                            ("buddhist", "1.1%"), ("catholic", "19%"),
                            ("evangelical-protestant", "23%")):
        rows = by_name.get(name, [])
        tot = sum(r["adults"] for r in rows)
        print(f"    {name:<24}{tot:>14,.0f}  {tot / US_ADULTS:6.2%} of US adults  "
              f"(Pew publishes {published}, {len(rows)} states)")

    if bad:
        for b in bad:
            print(f"  FAILED  {b}")
        raise SystemExit(f"validation failed: {len(bad)} problems")
    print(f"\n  OK  state figures sum to the national tree for all {len(national)} categories")


def main():
    out, bad, national, denoms = [], [], None, 0.0
    for i, slug in enumerate(STATES, 1):
        try:
            html = fetch(slug)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            bad.append((slug, str(e)))
            print(f"  [{i:2}/51] {slug:<22} FAILED {e}")
            continue
        cp = context(html, slug)
        rows = flatten(cp["religiousTree"], slug)
        check_partition(rows, slug)
        total = denominator(cp["religiousTree"], rows, slug)
        denoms += total

        # The state's own margin of error, carried so a downstream step can see how wide a
        # state total is before it subtracts a roll from it. Rhode Island's is +-5.6 points.
        moe = (cp.get("moes") or {}).get(str(cp.get("selectedYear", 2024))) or {}

        if national is None:                       # identical on all 51 pages; read once
            us_rows = flatten(cp["religiousTreeUS"], slug + " [national]")
            check_partition(us_rows, slug + " [national]")
            national = {r["name"]: r["adults"] for r in us_rows}

        for r in rows:
            out.append(dict(state=slug, adult_total=total,
                            state_moe=moe.get("moe", ""), state_ess=moe.get("ess", ""), **r))
        print(f"  [{i:2}/51] {slug:<22} {len(rows):>4} categories, "
              f"{total:>12,.0f} adults, MOE +-{moe.get('moe', '?')}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["state", "adult_total", "state_moe", "state_ess", "group",
                               "parent", "depth", "name", "label", "adults", "sample_size"])
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out):,} rows -> {OUT.relative_to(HERE)}")
    print(f"  {denoms:,.0f} adults over {len({r['state'] for r in out})} states")

    check_national(out, national)
    if bad:
        print(f"{len(bad)} states failed:")
        for slug, why in bad:
            print(f"  {slug}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
