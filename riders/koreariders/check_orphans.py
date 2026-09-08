# -*- coding: utf-8 -*-
"""Yearbook riders that reach no line's cumulation.

The one check here that looks for passengers the map has *lost* rather than for
passengers it has put in the wrong place. A station name the chains do not carry
is silently worth zero -- a missing key reads as no traffic, nothing errors, and
the map is simply light. That is how 김천구미 lost 1.2M a year, 서대구 184k, and
신경주 1.99M, each found by accident years apart.

Two ways to miss, and they want different fixes:

  **on no chain at all** -- the name does not appear in any line's stop list.
    Either it is spelled differently from OSM (`lines.STATION_ALIAS`), or the
    chain genuinely does not reach it.

  **on a chain whose line does not run that train type** -- 용산 is on 경부선's
    and 경원선's chains and neither declares KTX, so 10.75M KTX a year are
    dropped at the busiest station in the country. `lines.TYPES` and
    `lines.PART_TYPES` are the levers, and both have to be used carefully:
    where a parallel high-speed line already draws the same trains, granting
    the type twice double-counts them. See README.md, "What a railfan would
    catch".

**Clearing a station off this list is not the same as drawing it right**, and
신경주 is the case that proves it. Aliased onto 경주 it vanishes from here at
once, while its 1.99M SRT go to 중앙선, which runs none -- this report only asks
whether *some* line claims the traffic, never whether the right one did. After a
fix here, diff the segments and look at which line actually gained.
`lines.TYPE_HOME` is the lever for that half.

Do **not** test this against `data/segments.geojson`. `merge_unserved` joins
across a station with no flow of the line's own types, so exactly the stations
this is looking for are the ones missing from the drawn output -- 행신 reads as
absent when it is on 경의선's chain. Build the chains and test those.

Costs the usual setup minute and no fit, since it never solves.

    python check_orphans.py
    KOREARIDERS_YEARBOOK=data/korail_yearbook_2022_excel.zip python check_orphans.py
"""

import collections
import sys

import lines as LN
import membership as M
import build as B
import solve as S


def sweep():
    """(no_chain, wrong_type) as lists of (riders, station, note)."""
    table, _ = LN.resolve()
    serves = M.serves(LN)
    named = M.load_stations(LN)
    g = B.load_network()
    chains = S.build_chains(table, serves, g, named, {})

    on_chain = collections.defaultdict(set)
    for L, stops in chains.items():
        for _, nm in stops:
            on_chain[nm].add(L)

    flows = LN.station_flows()
    fk = LN.station_flows_by_type()

    no_chain, wrong_type = [], []
    for st, v in flows.items():
        n = sum(v)
        if n <= 0:
            continue
        ls = on_chain.get(st)
        if not ls:
            no_chain.append((n, st, ""))
            continue
        miss, kinds = 0, []
        for k in LN.ALL_TYPES:
            kn = sum(fk.get(k, {}).get(st, (0,) * 4))
            if not kn:
                continue
            # Claimed if any line calling here runs the type at all. This is
            # deliberately generous -- PART_TYPES restricts a type to a span,
            # and treating a station outside the span as a miss would flag
            # every 호남선 stop north of 광주송정 as lost when the traffic is
            # 호남고속선's and correctly drawn there.
            if not any(k in table[L]["types"] for L in ls):
                miss += kn
                kinds.append("%s=%d" % (k, kn))
        if miss:
            wrong_type.append((miss, st, "%s | on %s"
                               % (" ".join(kinds), ", ".join(sorted(ls)))))
    return sorted(no_chain, reverse=True), sorted(wrong_type, reverse=True)


def main():
    no_chain, wrong_type = sweep()
    flows = LN.station_flows()
    total = sum(sum(v) for v in flows.values())
    lost = sum(r[0] for r in no_chain) + sum(r[0] for r in wrong_type)

    print("riders the %s yearbook counts that reach no line: %.2fM of %.2fM, "
          "%.1f %%" % (LN.year(), lost / 1e6, total / 1e6, 100.0 * lost / total))

    print("\non no chain at all -- a name nothing carries")
    print("   %-14s %12s" % ("station", "riders/yr"))
    for n, st, _ in no_chain:
        print("   %-14s %12d" % (st, n))
    print("   %.2fM" % (sum(r[0] for r in no_chain) / 1e6))

    print("\non a chain, but no line there runs their train type")
    print("   %-14s %12s  %s" % ("station", "riders/yr", "types | chains"))
    for n, st, why in wrong_type:
        print("   %-14s %12d  %s" % (st, n, why))
    print("   %.2fM over %d stations"
          % (sum(r[0] for r in wrong_type) / 1e6, len(wrong_type)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
