# -*- coding: utf-8 -*-
"""Check the two Korail commuter layers, without rebuilding them.

`check_cities.py` tests the five intracity models against the yearbook's
도시철도 volume. Neither of these lines is in it: 동해선 광역전철 and ITX-청춘
are Korail's 광역철도, so their falsification test lives in a different volume
and their geometry comes from somewhere else again. This is that check.

**What it can test.**

- *Internal.* Whether consecutive segments meet, whether anything is negative,
  whether the drawn kilometres match the table, and whether the drawn profile
  still implies the mean trip length the fit was given. That last one is not
  circular: the fit sets the OD, the geojson is written from it segment by
  segment, and a slicing or ordering bug shows up here as a mean trip that no
  longer closes.
- *External, and this is the real one.* The 2022 yearbook's 광역철도 volume
  publishes 승차인원 and 인거리 per line, and **both of these lines are rows in
  it** — 동해선(부산) and itx-청춘. So the station counts the layers are built
  from can be summed and held against a published line total, and the trip
  length the decay was fitted to can be recomputed from published figures
  rather than taken on trust.

**What it cannot test.** Neither layer has a published per-segment figure to
check the *shape* against — that is the whole reason they are modelled. The
city models at least have 혼잡도 naming their busiest segment; nothing does that
for these. Treat the profile as reasoned, not verified.

The source check is on 2022 for both lines because that is the last edition of
the 광역철도 volume openpyxl can open — 2023's is `.xlsb`. The layers themselves
are built on 2023 (동해선) and 2025 (ITX-청춘), so a source check that passes
says the sheet series is sound, not that a particular year's file is.

    python check_commuter.py
"""

import io
import json
import math
import os
import sys
import zipfile

import commuter as C

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

GY_VOL = "3.광역철도/3. 수송실적_완.xlsx"
YEARBOOK_2022 = os.path.join(D, "korail_yearbook_2022_excel.zip")
BOARD_2022 = os.path.join(D, "donghae", "gwangyeok_2022.xlsx")

# layer file, the line as the 광역철도 volume names it, the alias its stations
# need in the 2022 board sheet, the line's published length, and whether that
# sheet's station rows can be attributed to this line at all.
#
# They can for 동해선, whose stations serve no other 광역철도 service. They
# cannot for ITX-청춘: the 2022 sheet carries no 노선명 column, and every one of
# its stops is shared -- with the 경춘선 전동차 up the line and with 경의중앙선
# at 용산, 옥수, 왕십리 and 청량리, where 용산 alone is 13M a year. Summing them
# gives 41M against a published 4.9M. That is the check being impossible, not
# the layer being wrong, so it is not run.
LAYERS = [
    ("donghae_segments.geojson", "동해선(부산)",
     {"교대(부산)": "교대", "송정(부산)": "송정"}, 65.7, True),
    ("gyeongchun_segments.geojson", "itx-청춘", {"백": "백양리"}, None, False),
]


def hav(a, b):
    """km between two [lon, lat] points."""
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def published_2022():
    """{line: (승차인원 명, 인거리 km)} from the 광역철도 volume, keyed casefolded.

    Three things about these sheets, all of which have cost time:

    - The row labels sit in a different column from the year header, so nothing
      can be read by position; both are matched on the row's own text.
    - 동해선 is written **동해선(부산)**, which a plain search for 동해선 misses.
    - **The same line is spelled two ways in the same workbook**: sheet 1 has
      `itx-청춘` and sheet 3 has `ITX-청춘`. Keyed as written, the line comes out
      with boardings and no 인거리 and a second entry with the reverse, and the
      check reports it missing. Hence the casefold.
    """
    import openpyxl

    with zipfile.ZipFile(YEARBOOK_2022) as z:
        member = None
        for i in z.infolist():
            n = i.filename
            if not (i.flag_bits & 0x800):
                n = n.encode("cp437").decode("cp949")
            if n == GY_VOL:
                member = i.filename
        if member is None:
            raise SystemExit("%s not in the 2022 bundle" % GY_VOL)
        data = z.read(member)

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True,
                                data_only=True)
    out = {}
    for sheet, scale, slot in (("1", 1000.0, 0), ("3", 1e6, 1)):
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        year_col = None
        for r in rows:
            for j, v in enumerate(r):
                if str(v).strip() == "2022":
                    year_col = j
            if year_col is not None:
                break
        for r in rows:
            label = next((str(v).strip() for v in r[:4]
                          if v is not None and str(v).strip()), "")
            if not label:
                continue
            v = r[year_col] if year_col is not None and len(r) > year_col \
                else None
            if not isinstance(v, (int, float)):
                continue
            rec = out.setdefault(label.casefold(), [None, None])
            rec[slot] = v * scale
    wb.close()
    return out


def main():
    pub = published_2022()
    ok = True

    for fname, pubname, alias, legal_km, sum_2022 in LAYERS:
        path = os.path.join(D, fname)
        if not os.path.exists(path):
            print("%s: not built" % fname)
            ok = False
            continue
        gj = json.load(open(path, encoding="utf-8"))
        fs = gj["features"]
        rep = gj.get("model_report", {})
        line = rep.get("line", "?")
        print("\n" + "=" * 70)
        print("%s  (%s, %s)" % (line, fname, rep.get("year")))

        nulls = sum(1 for f in fs if not f["geometry"])
        neg = [f["properties"] for f in fs if f["properties"]["daily"] < 0]
        worst, prev = 0.0, None
        tab = drawn = 0.0
        for f in fs:
            p = f["properties"]
            tab += p["km"]
            co = f["geometry"]["coordinates"] if f["geometry"] else []
            if len(co) > 1:
                drawn += sum(hav(a, b) for a, b in zip(co, co[1:]))
            if prev and co:
                worst = max(worst, hav(prev, co[0]) * 1000.0)
            if co:
                prev = co[-1]
        print("   %d segments, %d without geometry, %d negative"
              % (len(fs), nulls, len(neg)))
        print("   worst join between consecutive segments: %.0f m" % worst)
        print("   table %.1f km, drawn %.1f km%s"
              % (tab, drawn,
                 (", published %.1f" % legal_km) if legal_km else ""))
        if nulls or neg or worst > 50.0:
            ok = False
            print("   ** geometry or loads are wrong")

        board_day = rep.get("boardings_year", 0) / 365.0
        pkm = sum(f["properties"]["daily"] * f["properties"]["km"] for f in fs)
        want = rep.get("trip_km_published")
        if board_day and want:
            got = pkm / board_day
            bad = abs(got - want) > 0.5
            print("   implied mean trip %.2f km against the %.2f it was fitted "
                  "to%s" % (got, want, "   ** does not close" if bad else ""))
            if bad:
                ok = False
        peak = max(f["properties"]["daily"] for f in fs)
        print("   %.0f boardings a day, peak segment %.0f (%.0f %% of them)"
              % (board_day, peak, 100.0 * peak / board_day if board_day else 0))

        # --- the external half
        got = pub.get(pubname.casefold())
        if not got or not all(got):
            print("   no 2022 row named %r in the 광역철도 volume" % pubname)
            ok = False
            continue
        p_board, p_km = got
        print("   published 2022: %s 승차, %.1f km a trip"
              % ("{:,.0f}".format(p_board), p_km / p_board))
        if not sum_2022:
            print("   station rows cannot be attributed to this line in 2022 "
                  "-- every stop is shared and the sheet has no 노선명; see "
                  "LAYERS")
            continue
        stops = {f["properties"]["from"] for f in fs}
        stops |= {f["properties"]["to"] for f in fs}
        counts = C.read_counts(BOARD_2022, alias=alias)
        have = sum(counts[s][0] for s in stops if s in counts)
        miss = sorted(s for s in stops if s not in counts)
        d = 100.0 * (have - p_board) / p_board
        print("   the same stations in the 2022 board sheet: %s  (%+.2f %%)"
              % ("{:,.0f}".format(have), d))
        if miss:
            print("   not in the 2022 sheet: %s" % ", ".join(miss))
        if abs(d) > 1.0:
            ok = False
            print("   ** the source does not agree with the yearbook")

    print("\n" + ("all checks pass" if ok else "SOMETHING IS WRONG, see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
