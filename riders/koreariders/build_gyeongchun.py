# -*- coding: utf-8 -*-
"""ITX-청춘, 용산-춘천, from Korail's published station counts.

경춘선 is the one line on the intercity map drawn grey with no rider figure at
all. That is honest — the 철도통계연보 credits the whole line with **1,208
passengers a year** against 26 ITX-청춘 a day each way, which is plainly wrong —
but a reader still sees a blank where a busy railway is, and README.md's railfan
list says so. This draws the service the yearbook misses.

**Where the numbers come from.** `info.korail.com` board 425 files ITX-청춘 as a
line of its own, with per-station 승차/하차, exactly like a 광역철도 line. 2025
is used rather than 2023 because **only the 2025 and later sheets carry a
노선명 column**: on 2023's flat list every 경춘선 station has two rows, one for
the ITX and one for the 전동차, distinguishable by nothing but their position in
the file. That is the kind of positional assumption that breaks silently, so it
is not made.

**Its route is not 경춘선's**, which is why this is a separate builder from
`build_donghae.py` rather than another entry in it. The train runs
용산-옥수-왕십리-청량리 over 경의중앙선 and only then turns up 경춘선, so five of
its sixteen stops are off the intercity 경춘선 chain entirely. The corridor is
assembled instead from the Seoul layer, which already draws both lines with
geometry: 경의중앙선 for 용산-청량리 and 경춘선 for 청량리-춘천.

**The trip length is published and unusually well attested.** The yearbook's
광역철도 volume files ITX-청춘 as its own line in `3. 수송실적` sheets 1 and 3,
and the ratio barely moves across five years — 72.4 km in 2019, 70.2 in 2020,
71.0 in 2021, 72.8 in 2022, and KRIC's 차종별 여객수송실적 gives 73.3 for 2023.
So 73 km is not one reading, it is a stable series, and pairing it with 2025
counts costs little.

That mean trip is most of the route, which is the point: this is a limited-stop
service whose riders overwhelmingly go end to end, and the fit has to reproduce
that rather than the short hops a commuter line makes.

    python build_gyeongchun.py              # -> data/gyeongchun_segments.geojson
    python build_gyeongchun.py --trip-km 65 # sensitivity on the decay
"""

import argparse
import json
import math
import os
import sys

import numpy as np

import commuter as C

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

LINE = "ITX-청춘"
YEAR = "2025"
SHEET = os.path.join(D, "gyeongchun", "gwangyeok_2025.xlsx")
METRO = os.path.join(D, "metro_segments.geojson")

# 360 백만인-km / 4,947 천명, 2022; 72.4 / 70.2 / 71.0 either side of it and
# 73.3 from KRIC for 2023. See the module docstring.
TRIP_KM = 73.0

# The sheet truncates 백양리 to 백 on its 하차 row -- one station, two spellings,
# the 서대구 fault again. Merged rather than dropped: it is 3,452 alightings.
ALIAS = {"백": "백양리"}

# The two Seoul-layer lines the route is assembled from, in running order, and
# the station each leg runs between.
LEGS = [("수도권 경의중앙선", "용산", "청량리"),
        ("수도권 경춘선", "청량리", "춘천")]

EN = {
    "용산": "Yongsan", "옥수": "Oksu", "왕십리": "Wangsimni",
    "청량리": "Cheongnyangni", "상봉": "Sangbong", "퇴계원": "Toegyewon",
    "사릉": "Sareung", "평내호평": "Pyeongnae-Hopyeong", "마석": "Maseok",
    "청평": "Cheongpyeong", "가평": "Gapyeong", "강촌": "Gangchon",
    "백양리": "Baegyangni", "남춘천": "Namchuncheon", "춘천": "Chuncheon",
}


def haversine(a, b):
    """km between two (lon, lat) points."""
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def assemble():
    """The 용산 -> 춘천 corridor, and where each station sits along it.

    Each leg is a *path search*, not a walk. 용산 is mid-line on 경의중앙선 --
    the railway carries on west to 문산 -- so following segments greedily out of
    it goes the wrong way and dies at 홍대입구. A breadth-first search over the
    line's own from/to pairs takes the leg that actually reaches the target.

    Returns (polyline, cumulative km, {station: km along it}).
    """
    gj = json.load(open(METRO, encoding="utf-8"))
    poly, at = [], {}
    for line, first, last in LEGS:
        adj = {}
        for f in gj["features"]:
            p = f["properties"]
            if p["line"] != line or not f["geometry"]:
                continue
            co = f["geometry"]["coordinates"]
            adj.setdefault(p["from"], []).append((p["to"], co))
            adj.setdefault(p["to"], []).append((p["from"], co[::-1]))

        prev, queue = {first: None}, [first]
        while queue and last not in prev:
            cur = queue.pop(0)
            for nxt, co in adj.get(cur, []):
                if nxt not in prev:
                    prev[nxt] = (cur, co)
                    queue.append(nxt)
        if last not in prev:
            raise SystemExit("%s: no path from %s to %s" % (line, first, last))

        hops = []
        cur = last
        while prev[cur] is not None:
            back, co = prev[cur]
            hops.append((back, cur, co))
            cur = back
        hops.reverse()

        for frm, to, co in hops:
            if poly and haversine(poly[-1], co[0]) < 0.05:
                co = co[1:]
            if frm not in at:
                at[frm] = max(len(poly) - 1, 0)
            poly.extend(co)
            at[to] = len(poly) - 1

    cum = [0.0]
    for a, b in zip(poly, poly[1:]):
        cum.append(cum[-1] + haversine(a, b))
    return poly, cum, {nm: cum[i] for nm, i in at.items()}


def slice_poly(poly, cum, a, b):
    """The corridor between two chainages, ends included, running a -> b."""
    lo, hi = (a, b) if a <= b else (b, a)
    pts = [p for p, c in zip(poly, cum) if lo <= c <= hi]
    if len(pts) < 2:
        return []
    return pts if a <= b else pts[::-1]


def build(trip_km):
    counts = C.read_counts(SHEET, line=LINE, alias=ALIAS)
    poly, cum, at = assemble()
    print("   corridor assembled: %d points, %.1f km" % (len(poly), cum[-1]))

    stops = sorted((nm for nm in counts if nm in at), key=lambda n: at[n])
    missing = [nm for nm in counts if nm not in at]
    if missing:
        print("   counted but not on the corridor, dropped: %s"
              % ", ".join(missing))
    if len(stops) < 3:
        raise SystemExit("only %d stations matched the corridor" % len(stops))

    km = np.array([at[nm] for nm in stops], dtype=float)
    board = np.array([counts[nm][0] for nm in stops], dtype=float)
    alight = np.array([counts[nm][1] for nm in stops], dtype=float)
    cost = np.abs(km[:, None] - km[None, :])

    beta, od = C.fit_beta(cost, board, alight, trip_km)
    got = float((od * cost).sum() / od.sum())
    print("   %d stations over %.1f km" % (len(stops), km[-1] - km[0]))
    print("   beta %+.5f /km -> mean trip %.2f km (published %.1f)"
          % (beta, got, trip_km))
    print("   %.2fM boardings a year, %.0f a day"
          % (board.sum() / 1e6, board.sum() / 365.0))

    feats = []
    for k in range(len(stops) - 1):
        down = float(od[:k + 1, k + 1:].sum())
        up = float(od[k + 1:, :k + 1].sum())
        geom = slice_poly(poly, cum, at[stops[k]], at[stops[k + 1]])
        feats.append({
            "type": "Feature",
            "geometry": ({"type": "LineString", "coordinates": geom}
                         if len(geom) >= 2 else None),
            "properties": {
                "line": LINE,
                "from": stops[k], "to": stops[k + 1],
                "daily_down": round(down / 365.0, 1),
                "daily_up": round(up / 365.0, 1),
                "daily": round((down + up) / 365.0, 1),
                "km": round(km[k + 1] - km[k], 1),
                "source": "korail_gwangyeok_gravity",
                "period": YEAR,
                "geometry_source": "seoul_layer",
                "estimated": True,
            },
        })

    report = {
        "line": LINE, "year": YEAR, "stations": len(stops),
        "length_km": round(km[-1] - km[0], 2),
        "boardings_year": int(board.sum()),
        "alightings_year": int(alight.sum()),
        "trip_km_published": trip_km,
        "trip_km_fitted": round(got, 3),
        "beta_per_km": round(beta, 5),
        "busiest": max(({"from": f["properties"]["from"],
                         "to": f["properties"]["to"],
                         "daily": f["properties"]["daily"]} for f in feats),
                       key=lambda x: x["daily"]),
    }
    return {
        "type": "FeatureCollection",
        "features": feats,
        "line_meta": {LINE: {
            # Violet deliberately: 중앙선 is #ff6fb5 and runs out of 청량리
            # beside this train, 경춘선 and 경의선 are both teal, and a pink or
            # a green here is unreadable against them.
            "en": "ITX-Cheongchun (est.)", "color": "#b14aed",
            "cls": "conv", "estimated": True, "period": YEAR, "city": None,
        }},
        "station_names": {nm: EN[nm] for nm in stops if nm in EN},
        "model_report": report,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trip-km", type=float, default=TRIP_KM)
    ap.add_argument("--out",
                    default=os.path.join(D, "gyeongchun_segments.geojson"))
    args = ap.parse_args()
    result = build(args.trip_km)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, args.out)
    print("\nwrote %s (%d segments)" % (args.out, len(result["features"])))
    print(json.dumps(result["model_report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
