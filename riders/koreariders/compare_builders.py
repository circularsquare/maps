# -*- coding: utf-8 -*-
"""Where the network fit and the single-line build disagree, line by line.

The published 인거리 says the whole single-line build reaches 92.8 % of the
network's real passenger-km and the network fit 85.4 %, so the fit is losing
about 1.9 billion passenger-km somewhere beyond the undercount they share. A
network total cannot say where. This can: both builders write the same segment
schema, so their passenger-km can be differenced per line.

Read the columns in this order. `pkm` is the line's own passenger-km, which is
what the published total is made of, so a line's `diff` is its contribution to
the 1.9 billion. `밀도` is passenger-km over line length, which is what the map
draws, so a line can move the total hardly at all and still look completely
different on screen.

    python compare_builders.py
    python compare_builders.py --sort density
"""

import argparse
import collections
import io
import json
import os
import sys

import yearbook_extra as YX

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

FIT = os.path.join(D, "segments.geojson")
SINGLE = os.path.join(D, "segments_singleline.geojson")


def by_line(path):
    """{line: (passenger-km per year, drawn km, segment count)}."""
    with io.open(path, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    pkm, km, n = (collections.defaultdict(float), collections.defaultdict(float),
                  collections.Counter())
    for ft in feats:
        p = ft["properties"]
        pkm[p["line"]] += p["km"] * p["daily"] * 365.0
        km[p["line"]] += p["km"]
        n[p["line"]] += 1
    return {ln: (pkm[ln], km[ln], n[ln]) for ln in pkm}


def segments(path, line):
    with io.open(path, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    return [ft["properties"] for ft in feats
            if ft["properties"]["line"] == line]


def per_segment(line):
    """Both builders' profiles for one line, side by side.

    A line's passenger-km can differ because the whole profile is scaled or
    because one stretch of it collapsed, and those are different faults. 경원선
    looks like the second and is really the first: the fit holds 31 of its 38
    segments near zero on purpose, because no 일반열차 runs there.
    """
    # A segment split at a junction with no platform shares its name with the
    # other half, so key on the pair *and* its position in the chain.
    def keyed(rows):
        seen, out = {}, {}
        for p in rows:
            k = (p["from"], p["to"])
            seen[k] = seen.get(k, 0) + 1
            out[k + (seen[k],)] = p
        return out

    a, b = keyed(segments(FIT, line)), keyed(segments(SINGLE, line))
    order = list(a)
    order += [k for k in b if k not in a]
    print("%s -- network fit against single-line build\n" % line)
    print("%-19s %7s %10s %10s %8s" % ("segment", "km", "fit", "single",
                                       "ratio"))
    print("-" * 58)
    for k in order:
        pa, pb = a.get(k), b.get(k)
        nth = k[2]
        fa = pa["daily"] if pa else None
        fb = pb["daily"] if pb else None
        km = (pa or pb)["km"]
        ratio = ("%.2f" % (fa / fb)) if (fa is not None and fb) else "-"
        print("%-19s %7.1f %10s %10s %8s"
              % (("%s-%s%s" % (k[0], k[1], "" if nth == 1 else " (%d)" % nth))[:19], km,
                 "%d" % fa if fa is not None else "--",
                 "%d" % fb if fb is not None else "--", ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", choices=("diff", "density", "line"),
                    default="diff")
    ap.add_argument("--line", help="segment-by-segment for one line")
    args = ap.parse_args()

    if args.line:
        return per_segment(args.line)

    fit, single = by_line(FIT), by_line(SINGLE)
    _, pub_pkm = YX.intercity_pkm()

    rows = []
    for ln in sorted(set(fit) | set(single)):
        f_pkm, f_km, f_n = fit.get(ln, (0.0, 0.0, 0))
        s_pkm, s_km, s_n = single.get(ln, (0.0, 0.0, 0))
        rows.append({
            "line": ln,
            "fit_pkm": f_pkm, "single_pkm": s_pkm, "diff": f_pkm - s_pkm,
            "fit_den": f_pkm / (f_km * 365.0) if f_km else 0.0,
            "single_den": s_pkm / (s_km * 365.0) if s_km else 0.0,
            "ratio": (f_pkm / s_pkm) if s_pkm else float("nan"),
            "segs": (f_n, s_n),
        })

    key = {"diff": lambda r: r["diff"],
           "density": lambda r: -r["fit_den"],
           "line": lambda r: r["line"]}[args.sort]
    rows.sort(key=key)

    print("passenger-km per year, and 수송밀도, per line\n")
    print("%-11s %10s %10s %10s %7s %10s %10s %8s"
          % ("line", "fit bn", "single bn", "diff bn", "ratio",
             "fit 밀도", "single 밀도", "segs"))
    print("-" * 84)
    for r in rows:
        print("%-11s %10.3f %10.3f %+10.3f %7s %10.0f %10.0f %4d/%-3d"
              % (r["line"], r["fit_pkm"] / 1e9, r["single_pkm"] / 1e9,
                 r["diff"] / 1e9,
                 "%.2f" % r["ratio"] if r["ratio"] == r["ratio"] else "-",
                 r["fit_den"], r["single_den"],
                 r["segs"][0], r["segs"][1]))

    f_tot = sum(r["fit_pkm"] for r in rows)
    s_tot = sum(r["single_pkm"] for r in rows)
    print("-" * 84)
    print("%-11s %10.3f %10.3f %+10.3f" % ("total", f_tot / 1e9, s_tot / 1e9,
                                           (f_tot - s_tot) / 1e9))
    print("%-11s %10.1f%% %9.1f%%   of the published %.2f bn"
          % ("of published", 100 * f_tot / pub_pkm, 100 * s_tot / pub_pkm,
             pub_pkm / 1e9))

    losers = [r for r in rows if r["diff"] < 0]
    gap = sum(r["diff"] for r in losers)
    print("\nthe fit's shortfall is %.2f bn, concentrated in:" % (-gap / 1e9))
    for r in sorted(losers, key=lambda r: r["diff"])[:6]:
        print("   %-11s %+7.3f bn  (%.0f %% of the gap)"
              % (r["line"], r["diff"] / 1e9, 100 * r["diff"] / gap))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
