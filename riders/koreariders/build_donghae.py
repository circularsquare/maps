# -*- coding: utf-8 -*-
"""동해선 광역전철, 부전-태화강, from Korail's published station counts.

Sixty-five km of electrified commuter railway through Busan and up to Ulsan.
Until now no layer drew it: the city models are 부산 1-4, 대구 1-3, 대전, 광주
and 부산김해경전철, and this line is none of those -- it is Korail's, running on
the national 동해선 metals the intercity map already draws. README.md's railfan
list has it as the whole railway a 부산 reader would object to first.

**Where the numbers come from.** `info.korail.com` board 425 (광역수송 수송통계)
publishes per-station monthly 승차/하차 for every 광역철도 line, free and
without a login, 2017 to the current year. `data/donghae/gwangyeok_2023.xlsx` is
the 2023 sheet, chosen over the newer ones so the commuter figures sit on the
same year as the intercity 동해선 they are drawn beside. The older
`data/donghae/counts.csv` is the 2021-only data.go.kr file this replaces, kept
because it is what the README's complaint refers to.

Note the 2023 sheet has **no 노선명 column** -- it is a flat list of every
광역철도 station in the country -- so the line's membership comes from the
동해선 chain `solve.py` already builds, truncated at 태화강. The newer sheets do
carry 노선명 and agree with that list station for station.

**The model is the city models' one**, and deliberately so: published gate
counts, no OD, so a doubly-constrained gravity fit (`build_busan.fit_od`) with
an exponential decay tuned to a published mean trip length. What is different is
that this line is a single corridor with no branches and no transfers, so the
cost between two stations is just the distance along it -- no shortest path, no
transfer penalty.

**The trip length is published rather than assumed**, which is what 부산 1-4
still lack. The 2022 yearbook's 광역철도 volume, `3. 수송실적` sheets 1 and 3,
gives 동해선(부산) 16,088 천명 boarding and 293 백만인-km, so **18.2 km a trip**.
2022 rather than 2023 because that volume is `.xlsb` in the 2023 bundle and
openpyxl cannot open it, and because 2022 is the first full year with the
태화강 extension -- 2021 reads 11.7 km on a line that then ended at 일광.

Two things about that sheet worth not rediscovering: the row is labelled
**동해선(부산)**, not 동해선, and it sits in a different column from the year
header, which is why a naive search finds nothing. And 장항선's 인거리 jumps
86.6 to 1590 between 2019 and 2020 under an unchanged unit header, so the sheet
is not uniformly trustworthy -- 동해선's own series is smooth (111, 116, 127,
108, 124, 293) and 경춘선's is too, so this reading stands, but check any other
line's row before using it.

    python build_donghae.py                 # -> data/donghae_segments.geojson
    python build_donghae.py --trip-km 15    # sensitivity on the decay
"""

import argparse
import json
import os
import sys

import numpy as np

import build as B
import commuter as C
import lines as LN
import membership as M
import solve as S

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

LINE = "동해선 광역전철"
HOST = "동해선"                 # whose chain and corridor this borrows
LAST = "태화강"                 # the commuter service ends here; the line does not
YEAR = "2023"
SHEET = os.path.join(D, "donghae", "gwangyeok_2023.xlsx")

# 293 백만인-km / 16,088 천명, 2022 -- see the module docstring.
TRIP_KM = 293.0 / 16.088

# The counts spell two stations with the disambiguator the chain leaves off.
# Both are real collisions elsewhere in the country -- 교대 is also a 대구 metro
# station and 송정 a 광주 one -- so the sheet is right to qualify them and the
# chain, which only ever holds Korail's, is right not to.
ALIAS = {"교대(부산)": "교대", "송정(부산)": "송정"}

EN = {
    "부전": "Bujeon", "거제해맞이": "Geoje Haemaji", "거제": "Geoje",
    "교대": "Gyodae", "동래": "Dongnae", "안락": "Allak",
    "부산원동": "Busan Wondong", "재송": "Jaesong", "센텀": "Centum",
    "벡스코": "BEXCO", "신해운대": "Sin-Haeundae", "송정": "Songjeong",
    "오시리아": "Osiria", "기장": "Gijang", "일광": "Ilgwang",
    "좌천": "Jwacheon", "월내": "Wollae", "서생": "Seosaeng",
    "남창": "Namchang", "망양": "Mangyang", "덕하": "Deokha",
    "개운포": "Gaeunpo", "태화강": "Taehwagang",
}


def build(trip_km):
    table, _ = LN.resolve()
    serves = M.serves(LN)
    named = M.load_stations(LN)
    print("loading the national track graph ...")
    g = B.load_network()
    corridors = {}
    chains = S.build_chains(table, serves, g, named, corridors)
    if HOST not in chains:
        raise SystemExit("%s has no chain to borrow" % HOST)

    order = [nm for _, nm in chains[HOST]]
    if LAST not in order:
        raise SystemExit("%s is not on %s's chain" % (LAST, HOST))
    order = order[:order.index(LAST) + 1]

    counts = C.read_counts(SHEET, alias=ALIAS)
    shape = corridors[HOST]
    raw = shape["raw"]

    stops, missing = [], []
    for nm in order:
        if nm in counts and nm in raw:
            stops.append(nm)
        else:
            missing.append(nm)
    if missing:
        print("   not counted or not on the corridor, dropped: %s"
              % ", ".join(missing))
    if len(stops) < 3:
        raise SystemExit("only %d stations matched; nothing to draw"
                         % len(stops))

    km = np.array([raw[nm] for nm in stops], dtype=float)
    board = np.array([counts[nm][0] for nm in stops], dtype=float)
    alight = np.array([counts[nm][1] for nm in stops], dtype=float)
    cost = np.abs(km[:, None] - km[None, :])

    decay, od = C.fit_decay(cost, board, alight, trip_km)
    got = float((od * cost).sum() / od.sum())
    print("   %d stations, %.1f km of railway" % (len(stops),
                                                  abs(km[-1] - km[0])))
    print("   decay %.3f km -> mean trip %.2f km (published %.2f)"
          % (decay, got, trip_km))
    print("   %.2fM boardings a year, %.0f a day"
          % (board.sum() / 1e6, board.sum() / 365.0))

    # A segment carries every trip that starts on one side of it and ends on
    # the other. Split by direction so the page can show both, 하행 being the
    # way the line's own chainage runs, 부전 -> 태화강.
    feats = []
    poly, cum = shape["poly"], shape["cum"]
    for k in range(len(stops) - 1):
        down = float(od[:k + 1, k + 1:].sum())
        up = float(od[k + 1:, :k + 1].sum())
        geom = B.slice_corridor(poly, cum, raw[stops[k]], raw[stops[k + 1]])
        feats.append({
            "type": "Feature",
            "geometry": ({"type": "LineString", "coordinates":
                          [[p[1], p[0]] for p in geom]} if len(geom) >= 2
                         else None),
            "properties": {
                "line": LINE,
                "from": stops[k], "to": stops[k + 1],
                "daily_down": round(down / 365.0, 1),
                "daily_up": round(up / 365.0, 1),
                "daily": round((down + up) / 365.0, 1),
                "km": round(abs(km[k + 1] - km[k]), 1),
                "source": "korail_gwangyeok_gravity",
                "period": YEAR,
                "geometry_source": "osm",
                "estimated": True,
            },
        })

    report = {
        "line": LINE,
        "year": YEAR,
        "stations": len(stops),
        "length_km": round(abs(km[-1] - km[0]), 2),
        "boardings_year": int(board.sum()),
        "alightings_year": int(alight.sum()),
        "trip_km_published": round(trip_km, 3),
        "trip_km_fitted": round(got, 3),
        "decay_km": round(decay, 4),
        "busiest": max(
            ({"from": f["properties"]["from"], "to": f["properties"]["to"],
              "daily": f["properties"]["daily"]} for f in feats),
            key=lambda x: x["daily"]),
    }
    return {
        "type": "FeatureCollection",
        "features": feats,
        "line_meta": {LINE: {
            "en": "Donghae Line commuter rail (est.)",
            "color": "#00a5a8", "cls": "metro", "estimated": True,
            "period": YEAR, "city": "부산",
        }},
        "station_names": {nm: EN[nm] for nm in stops if nm in EN},
        "model_report": report,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trip-km", type=float, default=TRIP_KM,
                    help="mean trip length to fit the decay to")
    ap.add_argument("--out", default=os.path.join(D,
                                                  "donghae_segments.geojson"))
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
