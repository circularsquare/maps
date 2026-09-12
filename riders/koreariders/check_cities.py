# -*- coding: utf-8 -*-
"""Check the city models against the yearbook, without rebuilding them.

The five intracity models had no check on their output at all. Their gate
residuals are an accounting identity -- they say the IPF converged, not that
the OD is right -- and the scenario spans are a sensitivity sweep over an
assumption, not error bars. So a model could put Busan's busiest stretch in
the wrong place and nothing would say so.

The yearbook's 도시철도 sheet 13 says. It names the busiest segment on every
city line, every year, with the half-hour it applies to. That is a genuine
falsification test: peak crowding is a peak *rate* and our loads are a daily
total, so the two need not agree exactly -- a line can have its heaviest day
and its heaviest half-hour in different places -- but they should not be far
apart, and a model naming the opposite end of a line is wrong.

Sheet 10 gives the other half: the published mean trip length that
build_small_cities.py and build_daegu_busan.py now fit their decay to. Printing
what came out beside what was asked for is how a fit pinned at the network's
ceiling stays visible.

    python check_cities.py
"""

import io
import json
import os
import sys
import unicodedata

import yearbook_extra as YX

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

FILES = ("daegu_busan_segments.geojson", "small_city_segments.geojson",
         "busan_segments.geojson")

# Which yearbook 혼잡도 row belongs to which drawn line. The sheet is keyed by
# operator and the operator's own line number; the GeoJSON carries this
# project's line names.
PEAK_ROW = {
    "대구 1호선": ("대구교통공사", "1"),
    "대구 2호선": ("대구교통공사", "2"),
    "대구 3호선": ("대구교통공사", "3"),
    "부산김해경전철": ("부산-김해 경전철(주)", "1"),
    "부산 1호선": ("부산교통공사", "1"),
    "부산 2호선": ("부산교통공사", "2"),
    "부산 3호선": ("부산교통공사", "3"),
    "부산 4호선": ("부산교통공사", "4"),
    "대전 1호선": ("대전교통공사", "1"),
    "광주 1호선": ("광주광역시", "1"),
}


def norm(name):
    """Station names for comparison: no spaces, no bracketed suffix."""
    s = unicodedata.normalize("NFC", str(name)).strip()
    for cut in ("(", "（"):
        if cut in s:
            s = s.split(cut)[0]
    return s.replace(" ", "")


def segment_order(props):
    """Chain a line's segments into running order, so peaks can be compared.

    "Do the two name the same segment?" is the wrong question -- it scores a
    model that missed by one stop the same as one that put the peak at the far
    end of the line. What matters is how far apart they are, and for that the
    segments have to be in order. They are written in order by both city
    builders, but chaining on the station names rather than trusting that is
    cheap and survives a builder changing.
    """
    nxt = {}
    for p in props:
        nxt.setdefault(norm(p["from"]), []).append(p)
    starts = set(nxt) - {norm(p["to"]) for p in props}
    head = sorted(starts)[0] if starts else norm(props[0]["from"])
    out, seen = [], set()
    while head in nxt:
        p = next((q for q in nxt[head] if id(q) not in seen), None)
        if p is None:
            break
        seen.add(id(p))
        out.append(p)
        head = norm(p["to"])
    # A ring or a builder that did not write in order leaves stragglers.
    out.extend(p for p in props if id(p) not in seen)
    return out


def published_pair(segment):
    """Pull the two station names out of a 혼잡도 cell like '사당→방배 (08:30~09:00)'."""
    body = segment.split("(")[0]
    for arrow in ("→", "~", "-"):
        if arrow in body:
            a, b = body.split(arrow, 1)
            return norm(a), norm(b)
    return None


def resolve_pair(pair, chain):
    """Match abbreviated 혼잡도 station names to the ones on the chain.

    The sheet shortens names the way the KRIC portal does -- Busan-Gimhae's
    괘법르네시떼 and 서부산유통지구 are written 괘법 and 서부산 -- so an exact
    comparison silently reports "not on chain" and the check quietly stops
    checking. A published name that is a unique prefix of a chain station is
    that station; anything ambiguous is left alone rather than guessed.
    """
    stations = {norm(p[k]) for p in chain for k in ("from", "to")}
    out = []
    for name in pair:
        if name in stations:
            out.append(name)
            continue
        hits = [s for s in stations if s.startswith(name)]
        if len(hits) != 1:
            return None
        out.append(hits[0])
    return tuple(out)


def main():
    peaks = YX.urban_peak_segments()
    # Key the published rows the way PEAK_ROW names them.
    pub = {}
    for (op, line), value in peaks.items():
        pub[(op, str(line).replace("호선", "").strip())] = value

    print("city models against 철도통계연보 2022 도시철도 sheets 10 and 13\n")

    for fn in FILES:
        path = os.path.join(D, fn)
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8") as f:
            doc = json.load(f)
        print("=" * 78)
        print(fn)

        # --- trip length, where the build fitted one --------------------
        rep = doc.get("model_report", {})
        systems = (rep if any(isinstance(v, dict) and "scenarios" in v
                              for v in rep.values()) else {"": rep})
        for key, r in sorted(systems.items()):
            if not isinstance(r, dict) or "published_mean_trip_km" not in r:
                continue
            got = r["scenario_reached_km"][1]
            print("  %-14s mean trip: published %.2f km, model %.2f km  %s"
                  % (key, r["published_mean_trip_km"], got,
                     "OK" if abs(got - r["published_mean_trip_km"]) <= 0.01
                     else "SHORT by %.2f km -- at the network ceiling"
                          % (r["published_mean_trip_km"] - got)))
            if r.get("published_pkm_dropped_beyond_network"):
                print("  %-14s %.1f %% of the published 인거리 sits in distance "
                      "bands longer than\n  %-14s the network itself, and was "
                      "dropped before fitting."
                      % ("", 100 * r["published_pkm_dropped_beyond_network"], ""))

        # --- where the model puts each line's heaviest segment ----------
        by_line = {}
        for ft in doc["features"]:
            p = ft["properties"]
            by_line.setdefault(p["line"], []).append(p)
        print("")
        print("  %-14s %-22s %-22s %7s %s"
              % ("line", "model's busiest", "published busiest", "혼잡도",
                 "apart"))
        print("  " + "-" * 76)
        gaps = []
        for line, props in sorted(by_line.items()):
            chain = segment_order(props)
            top = max(chain, key=lambda p: p["daily"])
            model = "%s-%s" % (top["from"], top["to"])
            row = PEAK_ROW.get(line)
            rec = pub.get(row) if row else None
            if not rec:
                print("  %-14s %-22s %-22s" % (line, model, "(not published)"))
                continue
            seg, pct = rec
            pair = published_pair(seg)
            # How many segments along the line separate the two, which is the
            # only comparison that distinguishes "one stop out" from "wrong
            # end of the railway".
            gap = None
            if pair:
                pair = resolve_pair(pair, chain)
            if pair:
                mi = chain.index(top)
                pj = [i for i, p in enumerate(chain)
                      if {norm(p["from"]), norm(p["to"])} == set(pair)]
                if pj:
                    gap = min(abs(mi - j) for j in pj)
                    gaps.append((line, gap, len(chain)))
            print("  %-14s %-22s %-22s %6.1f%% %s"
                  % (line, model, seg.split("(")[0].strip()[:22], pct,
                     ("%d of %d" % (gap, len(chain))) if gap is not None
                     else "not on chain"))
        if gaps:
            worst = max(gaps, key=lambda g: g[1])
            print("\n  median %.1f segments apart, worst %s at %d of %d"
                  % (sorted(g[1] for g in gaps)[len(gaps) // 2],
                     worst[0], worst[1], worst[2]))
        print("")

    print("These are not the same quantity and are not expected to match "
          "exactly. Published\n혼잡도 is a rate in one half-hour on the peak "
          "approach; these loads are a daily\ntotal, which peaks at the "
          "busiest interchange instead. A model sitting one or two\nsegments "
          "centre-ward of the published peak is behaving correctly. What this "
          "catches\nis a model that puts the peak at the wrong end of a line.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
