"""
Which religions a country's source can SEE, as against which it happens to report.

THE DISTINCTION THIS EXISTS FOR. Select Sikhism and Poland draws no dots. That could mean
two completely different things — GUS asked and almost nobody said Sikh, or GUS never
offered the category and Poland's Sikhs are inside a residual — and the map has no way to
say which. §3.5 already insists that undercounting is marked rather than filled; this is the
same rule one level up, about the QUESTION rather than the answer. An absence of dots is
evidence about a country only if the country was asked.

So each country gets the set of taxonomy nodes its classification can express, and the
viewer lights the countries that can answer whatever is selected. Three readable states
instead of two:

    lit, with dots     the source asked, and these people are there
    lit, no dots       the source asked, and essentially nobody said it
    unlit              the source never offered the category — this is not evidence

COVERAGE IS THE MAPPING'S TARGETS, NOT THE DATA'S CONTENTS, and the difference is the whole
point. A category that exists on the census form and scores zero nationally still counts as
covered, because the question was put. Reading coverage off the counts instead would
collapse exactly the two cases this is here to separate.

FOUR SHAPES, because the mapping modules are not uniform and pretending otherwise would
rot. Most expose `MAP`; Canada resolves up an ancestor chain and keeps `NODE` + `LEAF`; Pew
maps one category to SEVERAL paths and keeps them in `CUT`; the US is two sources at once
and is the union of both. Each is named below rather than sniffed for, so a module that
changes shape breaks loudly here instead of silently returning a short set.

THE CHECK THAT KEEPS IT HONEST: every node that actually draws a dot must be in its
country's coverage. If a mapping grows a target this file does not know about, `verify()`
says so — otherwise a country would go dark for a religion it demonstrably contains.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))

# cc -> the module whose mapping countries.py actually uses. **DISCOVERED, not listed** —
# taxonomy/registry.py walks taxonomy/<cc><YYYY>.py and intersects with COUNTRIES. This was
# a hand-maintained dict until 2026-09-06, and adding a country meant remembering to edit it
# here AND in tools/check_mapping.py; that was forgotten three times in one day (Indonesia,
# South Korea, Bangladesh), each time shipping a country that would go dark for every
# religion it contains. The check caught it every time and a human still had to fix it.
#
# Called lazily rather than at import: registry.discover() imports countries.py, and
# tiles.py imports THIS module, so evaluating at import time would drag countries.py into
# every consumer whether or not it needed it.
def _simple():
    from registry import discover
    return discover()

# CHINA IS THE STRONGEST CASE THIS FILE HAS, and it is worth saying why. Its source counts
# NATIONALITY, so the religions it can express at all are only the ones some ethnic category
# implies: spec §14.5's three religio-ethnic groups, §14.9's fractional Protestant share over
# six Yunnan border peoples, and §14.7's `unknown` for everyone else. Five nodes out of 572.
# **So selecting Judaism, or Hinduism, or Daoism leaves China unlit, and that is exactly
# right** — an unlit China says "not asked", where a lit one with no dots would have said
# "asked, and nobody is there", which would be a lie about a very large number of people.
# China has tens of millions of Christians beyond the six peoples drawn and this map cannot
# see one of them.
#
# **It needs its own branch below because `cn2000.MAP` stopped being the whole answer on
# 2026-09-07** (spec §14.13): the module now resolves through `shares()`, which returns a
# list of (node, share, tier) and carries `christianity.protestant` and `unknown` in tables
# `MAP` knows nothing about. Reading `MAP` alone silently dropped 97.6% of the country's
# nodes from its coverage and this file caught it on the next run, which is the argument for
# `verify()` existing at all.


def _clean(paths):
    return {p for p in paths if isinstance(p, str) and p}


def coverage():
    """cc -> set of node ids the country's classification can express."""
    import importlib

    out = {}
    for cc, name in _simple().items():
        mod = importlib.import_module(name)
        if not hasattr(mod, "MAP"):
            raise SystemExit(f"coverage: {name} has no MAP — its shape changed, "
                             "add it to coverage.py explicitly")
        out[cc] = _clean(mod.MAP.values())

    # Canada: resolve() walks StatCan's own parent chain, so the targets are spread over two
    # dicts rather than one. NODE carries `None` for the universe row; _clean drops it.
    ca = importlib.import_module("ca2021")
    out["ca"] = _clean(ca.NODE.values()) | _clean(ca.LEAF.values())

    # China: `shares()` is the resolver, not `MAP` — see the note above. Asking it what every
    # source category returns is the only way to get the whole target set, and it stays right
    # if a nationality moves between the tables.
    cn = importlib.import_module("cn2000")
    cats = set(cn.MAP) | set(cn.MISSION) | set(cn.NOT_ASSERTED) | set(cn.EXCLUDED)
    out["cn"] = _clean(node for c in cats for node, _s, _t in cn.shares(c))

    # The United States is two instruments (§3.5a): ASARB's 372 bodies, plus Pew for the
    # self-identification re-basing. A Pew category maps to a TUPLE of paths, not one.
    usrc = importlib.import_module("usrc2020")
    pew = importlib.import_module("us_pew2024")
    us = _clean(usrc.MAP.values())
    for paths in pew.CUT.values():
        us |= _clean(paths)
    out["us"] = us

    # A COUNTRY WITH A FOREIGN HALF IS ASKED TWICE, and its coverage is the union.
    # Spain's CIS and Greece's ESS each offer a handful of religion categories to residents
    # they can reach; the other half of each country is drawn from the state's own count of
    # foreign residents by citizenship, crossed with taxonomy/origin_religion.py. That second
    # instrument can express about forty nodes, and every one of them is a question that was
    # put — so a country that draws no Sikh dots because no Sikh-majority nationality lives
    # there is "asked, and essentially nobody", not "never asked". Reading coverage off the
    # MAP alone left Spain dark for twenty-three religions it demonstrably contains.
    origin = importlib.import_module("origin_religion")
    # `it` was missing here from the day Italy landed and the omission was INVISIBLE for as
    # long as no oriental-Orthodox dot happened to survive rounding: coverage checks the
    # nodes that DRAW, so a node the model can emit but which draws zero dots looks the same
    # as one the model cannot emit at all. Re-placing the foreign half (§9as) moved enough
    # weight into Italy's cities for `christianity.oriental` and its Armenian child to reach
    # one dot each, and the gap surfaced as two failures in a country that had passed.
    # **A coverage list keyed to a country's SOURCES cannot be maintained by watching the
    # map** — anything that changes rounding can expose or hide an entry.
    for cc, other in (("es", "other.es"), ("gr", "other.gr"), ("fr", "other.fr"),
                      ("it", "other.it")):
        if cc in out:
            out[cc] |= origin.nodes(other)
    return out


# A country whose CLASSIFICATION CHANGES INSIDE ITS OWN BORDERS, and so cannot be lit or
# unlit as one shape. The UK is the only one on the map and is an extreme case: three
# censuses, three agencies, three category lists (sources/uk.md).
#
# Why it had to be split. England and Wales publish no Christian denomination at all, for
# 27.5 million people; Scotland names the Church of Scotland and the Roman Catholics;
# Northern Ireland names twenty-two bodies including four kinds of Presbyterian. So
# selecting Latin Catholic lit the whole United Kingdom while dots appeared only around
# Glasgow and Belfast — and the empty half of that picture was England reading as "asked,
# and nobody said it" when the truth is that England was never asked. That is precisely
# the confusion §6.12 exists to remove, committed by the layer meant to remove it.
#
# The three keys mirror _uk_counts exactly, including its exclusion of NISRA's second
# question (religion brought up in), which is a different variable and is not on the map.
UK_REGIONS = {
    "uk-ew": ("uk_ew_census_2021", "England and Wales"),
    "uk-sc": ("uk_sc_census_2022", "Scotland"),
    "uk-ni": ("uk_ni_census_2021", "Northern Ireland"),
}


def regions():
    """region id -> {cc, name, covers}. Most countries are one region equal to themselves.

    The wash draws THIS, not `coverage()`, because a shape that is lit or unlit as a whole
    can only tell the truth about a country whose source is uniform across it.
    """
    import importlib
    import pandas as pd

    out = {}
    cov = coverage()
    for cc, covers in cov.items():
        if cc != "uk":
            out[cc] = {"cc": cc, "name": None, "covers": sorted(covers)}

    uk = importlib.import_module("uk2021")
    src = HERE / "data" / "normalized" / "uk.csv"
    if not src.exists():
        # No uk.csv on a fresh checkout: fall back to one undivided UK rather than dropping
        # it from the wash entirely, and say so.
        print("  !! no uk.csv — the UK stays one region and England will over-claim "
              "(coverage.py UK_REGIONS)")
        out["uk"] = {"cc": "uk", "name": None, "covers": sorted(cov["uk"])}
        return out

    cats = pd.read_csv(src, usecols=["source_id", "source_category"],
                       low_memory=False).drop_duplicates()
    for rg, (source_id, name) in UK_REGIONS.items():
        here = cats.loc[cats["source_id"] == source_id, "source_category"]
        got = _clean(uk.resolve(c) for c in here)
        if not got:
            raise SystemExit(f"coverage: {rg} resolved to nothing — uk.csv no longer "
                             f"carries source_id {source_id!r}?")
        out[rg] = {"cc": "uk", "name": name, "covers": sorted(got)}
    return out


def verify(counts_path=None, verbose=True):
    """Every node that draws a dot must be covered. Returns the list of violations."""
    import json

    counts_path = counts_path or HERE / "data" / "processed" / "counts.json"
    if not os.path.exists(counts_path):
        if verbose:
            print("coverage: no counts.json yet, nothing to verify against")
        return []
    cov = coverage()
    counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
    bad = []
    for cc, meta in counts.get("countries", {}).items():
        have = cov.get(cc)
        if have is None:
            bad.append((cc, "*", "country has no coverage entry in coverage.py"))
            continue
        for nid in meta.get("dots", {}):
            if nid not in have:
                bad.append((cc, nid, "draws dots but is not in the country's coverage"))
    if verbose:
        if bad:
            print(f"coverage: {len(bad)} PROBLEM(S) — a country would go dark for a religion "
                  "it demonstrably contains:")
            for cc, nid, why in bad[:40]:
                print(f"    {cc}  {nid}  — {why}")
        else:
            n = sum(len(v) for v in cov.values())
            print(f"coverage: ok — {len(cov)} countries, {n:,} (country, node) pairs, "
                  "every drawn node covered")
    return bad


if __name__ == "__main__":
    sys.exit(1 if verify() else 0)
