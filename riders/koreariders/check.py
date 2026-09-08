# -*- coding: utf-8 -*-
"""Check data/segments.geojson against the yearbook, without re-solving.

The reconstruction has one external check on its numbers (통과인원) and, since
the segments started carrying a drawn shape, one on its geometry: the length of
what was drawn can be compared with the 영업거리 the chainage was rescaled to.
The rescale is what hides the error -- every station still lands in the right
*proportion* of the line, so a corridor routed 150 km the wrong way looks
perfectly ordered in the segment table and only the drawn kilometres give it
away. That check caught `own_component` regressing 호남고속선 from -0.5 % to
-50 % and 태백선 to -37 % in the same run that fixed 영동선, and nothing else in
the pipeline would have noticed.

Since 2026-09-07 it also checks that two lines calling at the same station are
drawn in the same *place*. Nothing did before — each line is routed on its own
metals and only its own segments were ever tested for meeting each other — and
the first run found two stations that are two stations, sharing a name 189 km
and 26 km apart. `solve.py` pairs lines at a junction by station name, so a
collision like that is a junction constraint between lines that never meet.

It is a check on the *output file*, so it runs in a couple of seconds against
whatever is on disk rather than behind a solve.

    python check.py                     # data/segments.geojson
    python check.py --line 경원선        # that line's segments, from the file
    python check.py --file data/segments_singleline.geojson
"""

import argparse
import collections
import io
import json
import math
import os
import sys

import frequency as FQ
import lines as LN
import yearbook_extra as YX

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

TOL_PCT = 3.0           # how far a drawn line may sit from its 영업거리
# A line whose ends lines.ENDS moved is not measured by its 영업거리 at all: the
# difference between them is the stub ENDS added or removed, which is a real
# length rather than a proportion -- 대구선's +6.2 km is the 동대구-가천 approach
# and reads as +24 % only because the line is short. So allow those in km. The
# largest legitimate stub is under ten km (태백-백산, 경주-모량), while the
# misroutes this check exists to catch were 38 and 92 km.
#
# 경부고속선 is now the largest at 18.7 km: lines.OVER extends it up 경부선 to
# reach 서울역, so it draws 417.3 km against a 398.2 km 영업거리 that measures the
# high-speed metals alone. 25 km still catches the 38 and the 92.
MOVED_TOL_KM = 25.0
GAP_M = 50.0            # how far consecutive segments may fail to meet

# Two lines calling at the same station should be drawn touching there, and a
# line's own segments meeting end to end does not test that at all -- each line
# is routed on its own metals, so nothing has ever checked that the metals meet.
# On the map a junction that does not join reads as two railways passing.
#
# The threshold is generous on purpose. A shared station is one *name*, and the
# two lines are drawn on their own tracks through it, which at a big station is
# genuinely a few hundred metres apart -- 서울 has 경부선 and 경부고속선 on
# different platform faces. What this is looking for is the case where a line
# was routed somewhere else entirely and the two are kilometres apart.
JUNCTION_M = 600.0

# Seats on the largest train of each yearbook passenger type, 2022 stock. The
# occupancy check takes the *largest* type a line runs, so the ceiling assumes
# every train on the section is the roomiest one available and completely full.
# A segment over that has more passengers than the railway can physically move,
# whatever the 승하차 say.
#
# KTX is the wide one: a KTX-1 seats 935 over eighteen cars and a KTX-산천 363,
# so a Gyeongbu section gets the benefit of the doubt a Honam one does not
# really deserve. That is deliberate -- this check should only ever fire on
# something indefensible.
SEATS = {
    "KTX": 935,          # KTX-1; KTX-산천 is 363
    "SRT": 410,
    "새마을": 376,        # ITX-새마을, 6 cars
    "ITX-새마을": 376,
    "무궁화": 432,        # 6 cars at 72
    "통근": 400,
}

# Standing tickets are 1.44M of 66.8M KTX journeys, 2.2 %, so a whole-day mean
# meaningfully above the seat count is not a crowded train -- it is a wrong
# number. The margin is for consist variation rather than for standing.
OCCUPANCY_MARGIN = 1.15


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def polyline_km(coords):
    """Length of a GeoJSON LineString's [lon, lat] coordinate list."""
    return sum(haversine((a[1], a[0]), (b[1], b[0]))
               for a, b in zip(coords, coords[1:]))


def junctions(by_line):
    """Stations two lines both call at: how far apart are they drawn?

    Returns [(station, metres, [(line, point), ...]), ...] worst first.

    Where a line puts a station is not in the file as a point, but it is in the
    geometry: a segment runs from its `from` stop to its `to` stop, so the first
    coordinate of the segment leaving a station and the last of the one arriving
    are both that line's idea of where the station is.
    """
    at = collections.defaultdict(dict)      # station -> {line: point}
    for ln, fts in by_line.items():
        for ft in fts:
            g = ft.get("geometry")
            cs = (g or {}).get("coordinates")
            if not cs:
                continue
            p = ft["properties"]
            at[p["from"]].setdefault(ln, (cs[0][1], cs[0][0]))
            at[p["to"]].setdefault(ln, (cs[-1][1], cs[-1][0]))

    out = []
    for st, byline in at.items():
        if len(byline) < 2:
            continue
        pairs = sorted(byline.items())
        worst = max(haversine(a[1], b[1]) * 1000.0
                    for i, a in enumerate(pairs) for b in pairs[i + 1:])
        out.append((st, worst, pairs))
    return sorted(out, key=lambda z: -z[1])


def mirror(by_line):
    """하행 against 상행 per line, the same figure solve.py reports.

    The two profiles are built from disjoint columns of the source, so their
    disagreement tests the method rather than the data and nothing but a broken
    cumulation can widen it. It is the project's headline quality number and it
    used to be visible only at the end of an eight-minute solve -- but the
    segments carry `down` and `up`, so it can be read off the file instead.

    Weighted by the traffic it applies to, because the worst single segment is
    whichever one carries almost nobody: 충북선's 조치원-오송 stub is 0 against
    846, two people a day, while its other fifteen segments agree to 3 %.
    """
    rows = []
    for ln, fts in by_line.items():
        d = [f["properties"]["down"] for f in fts]
        u = [f["properties"]["up"] for f in fts]
        gap = sum(abs(a - b) for a, b in zip(d, u))
        held = sum(max(abs(a), abs(b)) for a, b in zip(d, u))
        worst = max((abs(a - b) / max(abs(a), abs(b), 1)
                     for a, b in zip(d, u)), default=0.0)
        rows.append((gap / held if held else 0.0, worst, ln,
                     sum(f["properties"]["daily"] for f in fts)))
    rows.sort(reverse=True)
    return rows


def occupancy(by_line, table):
    """Passengers per train on each segment, against what a train holds.

    Every other check here compares one derived number with another. This one
    compares a derived number with the railway: `6. 운전` publishes trains a day
    on each section, so a segment's load divided by its trains is the average
    number of people aboard, and that cannot exceed what the train has room
    for. Nothing about the reconstruction enters it.

    It is worth having because an aggregate can be right for the wrong reason.
    The published 인거리 puts the single-line build closer to the network total
    than the fit, which reads as a point in its favour -- and per segment it
    turns out to need 889 people on each 무궁화 through 지천-신동, which seats
    about 432. Its total is closer because an impossible overcount on 경부선
    happens to fill the gap left by the undercount both builders share. Two
    errors cancelling look exactly like accuracy until you check per segment.

    **The counts are per direction**, so a segment's capacity is trains x seats
    x 2 against a load that is both directions. 수서고속선 settles it without
    involving any model at all: SRT carried 19.56M journeys in 2022, every one
    of which crosses that line since every SRT train starts or ends at 수서,
    and the sheet gives it 60 SRT a day. Read as both directions that is 893
    people on a 410-seat train, which is impossible; read as one it is 447, or
    109 % of seats, which is an ordinary busy service. The service levels agree
    -- SR ran about 40 Gyeongbu and 20 Honam round trips a day in 2022, which
    is the 40 and 20 the sheet gives those sections. So does the network: the
    published 인거리 over twice these counts is about 280 a train against a
    fleet averaging some 500 seats, a load factor near half, where the
    both-directions reading would put the whole network above 100 % all day.

    The comparison between two outputs does not depend on the reading at all,
    since both are divided by the same counts. The mean printed here is an
    upper bound either way: train-km is summed over the 22 modelled lines only,
    so the denominator misses track the published 인거리 counts traffic on.

    The low side is not a check, because a near-empty train is possible and an
    overfull one is not. It is still worth printing: a line whose trains would
    run far emptier than the network's says either that the railway really is
    that quiet or that some of its riders have been booked elsewhere, and both
    are things to know. Two ways the denominator misleads, and both are
    ordinary rather than faults:

    - **A high-speed train on a conventional line's metals.** 호남선's
      익산-광주송정 runs 20 SRT and 2 KTX a day each way against 15 conventional
      trains, and those high-speed passengers are credited to 호남고속선. So
      호남선 reads 46 a train over 장성-광주송정 where the conventional service
      alone carries 114 -- an ordinary 무궁화 load, and the reason that stretch
      looks thin on the map.
    - **The source not counting the service at all**, which is 경춘선 at 0.2 a
      train. That one the map now says outright; see `service` in solve.py.

    Returns (rows, mean_occupancy, per_line) with rows sorted worst first and
    per_line as (passengers per train, line) ascending.
    """
    rows, pkm, train_km = [], 0.0, 0.0
    line_pkm = collections.defaultdict(lambda: [0.0, 0.0])
    for ln, fts in by_line.items():
        spec = table.get(ln)
        if not spec:
            continue
        # A run-on stub is an extra piece of drawing past the last platform, not
        # a link in the chain -- its `to` is a line rather than a station, and
        # leaving it in shifts every section count by one.
        props = [f["properties"] for f in fts if not f["properties"].get("stub")]
        if not props:
            continue
        stops = [props[0]["from"]] + [p["to"] for p in props]
        # runs_along wants the line's own 기점 -> 종점 order, which is the chain
        # reversed where resolve() swapped the ends to put the anchor last.
        order = stops[::-1] if spec.get("reversed") else stops
        runs = FQ.runs_along(ln, spec["types"], order)
        if runs is None or len(runs) != len(props):
            continue
        if spec.get("reversed"):
            runs = runs[::-1]
        seats = max((SEATS.get(t, 0) for t in spec["types"]), default=0)
        if not seats:
            continue
        for p, n in zip(props, runs):
            if n <= 0:
                continue
            # `daily` is both directions and the published count is one, so the
            # trains carrying that load are 2n.
            both = 2 * n
            pkm += p["km"] * p["daily"]
            train_km += both * p["km"]
            line_pkm[ln][0] += p["km"] * p["daily"]
            line_pkm[ln][1] += both * p["km"]
            per = p["daily"] / both
            if per > seats * OCCUPANCY_MARGIN:
                rows.append((per / seats, ln, p["from"], p["to"], per, both,
                             p["daily"], seats))
    rows.sort(reverse=True)
    per_line = sorted((v[0] / v[1], ln) for ln, v in line_pkm.items() if v[1])
    return rows, (pkm / train_km if train_km else 0.0), per_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(D, "segments.geojson"))
    ap.add_argument("--line")
    args = ap.parse_args()

    path = args.file if os.path.isabs(args.file) else os.path.join(HERE, args.file)
    with io.open(path, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    table, _ = LN.resolve()

    # Features are written in chain order within each line, so consecutive
    # features of one line are consecutive segments of it.
    by_line = collections.OrderedDict()
    for ft in feats:
        by_line.setdefault(ft["properties"]["line"], []).append(ft)

    print("%s: %d segments over %d lines\n"
          % (os.path.relpath(path, HERE), len(feats), len(by_line)))

    # One line's table, read out of the file. solve.py prints the same thing but
    # only after an eight-minute fit, and most questions about a suspect line are
    # about the answer it already wrote.
    if args.line:
        fts = by_line.get(args.line)
        if not fts:
            raise SystemExit("no line named %s in %s" % (args.line, path))
        print("%-17s %7s %10s %10s %10s %9s"
              % ("segment", "km", "하행", "상행", "명/일", "drawn km"))
        print("-" * 68)
        for ft in fts:
            p = ft["properties"]
            g = ft.get("geometry")
            print("%-17s %7.1f %10d %10d %10d %9.1f"
                  % ((p["from"] + "-" + p["to"])[:17], p["km"], p["down"],
                     p["up"], p["daily"],
                     polyline_km(g["coordinates"]) if g else 0.0))
        print("")

    print("%-11s %9s %9s %9s %8s %7s %8s %9s"
          % ("line", "drawn km", "table km", "영업거리", "diff", "no geom",
             "worst gap", "min load"))
    print("-" * 80)
    bad_len, no_geom, gaps = [], 0, []
    for ln, fts in sorted(by_line.items(),
                          key=lambda kv: -sum(f["properties"]["km"]
                                              for f in kv[1])):
        drawn, missing, worst_gap, prev_end = 0.0, 0, 0.0, None
        for ft in fts:
            g = ft.get("geometry")
            if not g or not g.get("coordinates"):
                missing += 1
                prev_end = None
                continue
            cs = g["coordinates"]
            drawn += polyline_km(cs)
            # slice_corridor runs stop i -> stop i+1, so the segments of a line
            # are written head to tail and one's last point is the next one's
            # first. Anything else is a hole in the drawn line.
            a, b = (cs[0][1], cs[0][0]), (cs[-1][1], cs[-1][0])
            if prev_end is not None:
                worst_gap = max(worst_gap, haversine(prev_end, a) * 1000.0)
            prev_end = b
        spec = table.get(ln, {})
        pub = spec.get("length_km", 0.0)
        moved = not spec.get("scaled", True)
        # What the segment table itself claims, which is the drawn chainage
        # rescaled to 영업거리 -- or not rescaled at all where the ends moved.
        # Printing both is how a rescale onto the wrong extent shows up.
        tab = sum(f["properties"]["km"] for f in fts)
        diff = 100.0 * (drawn - pub) / pub if pub else 0.0
        loads = [f["properties"]["daily"] for f in fts]
        print("%-11s %9.1f %9.1f %9.1f %+7.1f%%%s %6d %7.0f m %9d"
              % (ln, drawn, tab, pub, diff, "*" if moved else " ", missing,
                 worst_gap, min(loads)))
        no_geom += missing
        off = (abs(drawn - pub) > MOVED_TOL_KM if moved
               else abs(diff) > TOL_PCT)
        if pub and off and missing == 0:
            bad_len.append((ln, drawn, pub, diff))
        if worst_gap > GAP_M:
            gaps.append((ln, worst_gap))

    print("\n%d segments with no geometry" % no_geom)
    print("* the ends of this line were moved by lines.ENDS, so its 영업거리 "
          "covers different\n  track: the gap is the stub that was added or "
          "removed, and it is allowed up to\n  %.0f km rather than a percentage."
          " Those lines are not rescaled, so their drawn\n  and table km agree "
          "by construction." % MOVED_TOL_KM)
    if bad_len:
        print("\n%d lines drawn too far from their 영업거리 -- the corridor is "
              "routed wrong,\nnot the chainage, since the rescale hides exactly "
              "this:" % len(bad_len))
        for ln, drawn, pub, diff in sorted(bad_len, key=lambda z: -abs(z[3])):
            print("   %-11s drew %.1f km for a %.1f km line (%+.0f %%)"
                  % (ln, drawn, pub, diff))
    else:
        print("\nevery line drawn within tolerance of its 영업거리")
    if gaps:
        print("\n%d lines with consecutive segments failing to meet:" % len(gaps))
        for ln, g in sorted(gaps, key=lambda z: -z[1]):
            print("   %-11s %.0f m" % (ln, g))
    else:
        print("consecutive segments meet everywhere (worst gap under %.0f m)"
              % GAP_M)

    # --- and where two lines share a station, do they touch? ---------------
    jn = junctions(by_line)
    apart = [j for j in jn if j[1] > JUNCTION_M]
    print("\n junctions -- %d stations are called at by more than one line"
          % len(jn))
    if apart:
        print(" %d of them are drawn more than %.0f m apart, so the lines do "
              "not visually\n meet there:" % (len(apart), JUNCTION_M))
        for st, m, pairs in apart:
            print("   %-10s %7.0f m   %s"
                  % (st, m, ", ".join(ln for ln, _ in pairs)))
    else:
        print(" every one of them is drawn within %.0f m on every line that "
              "calls there" % JUNCTION_M)
    rest = jn[len(apart):]
    if rest:
        st, m, pairs = rest[0]
        print(" widest that passes: %s at %.0f m (%s)"
              % (st, m, ", ".join(ln for ln, _ in pairs)))

    # --- the level, against the published 인거리 --------------------------
    #
    # Every other check here is on the *shape* -- the mirror, the geometry,
    # positivity. The level had nothing: build.py put 경부선 at 33,018 and
    # solve.py at 20,051, both with clean mirrors, and README.md recorded that
    # nothing said which was right.
    #
    # Sheet 14 says. It publishes 인거리 by distance band and train type, and
    # that total is a plain figure for the whole intercity network rather than
    # the per-line 인거리 probe_ingeori.py rejected for not being attributed to
    # track. Summing load x length over our own segments is the same quantity,
    # so the two compare directly.
    #
    # Expect to land *under* it and not over. The rebuild misses traffic it
    # knows about -- 10.4M boardings a year on no chain, the high-speed lines
    # seeing 0.54-0.61 of their 통과인원 -- and none of that inflates the total.
    # Coming in over the published figure would mean inventing passenger-km.
    pub_pax, pub_pkm = YX.intercity_pkm()
    model_pkm = sum(f["properties"]["km"] * f["properties"]["daily"]
                    for f in feats) * 365.0
    print("\n인거리 -- the model's level against the published total")
    print("   published (yearbook sheet 14, 2022)  %8.2f bn passenger-km"
          % (pub_pkm / 1e9))
    print("   this file                            %8.2f bn   (%.1f %% of it)"
          % (model_pkm / 1e9, 100.0 * model_pkm / pub_pkm))
    if model_pkm > pub_pkm:
        print("   OVER the published total, which the reconstruction cannot "
              "legitimately be:\n   it sums a subset of the traffic. Something "
              "is being double-counted.")

    # --- 하행 against 상행, without re-solving ------------------------------
    mir = mirror(by_line)
    # np.median, which is what solve.py's own report uses: the mean of the two
    # middle values on an even count, so the two agree to the last tenth.
    vals = sorted(r[0] for r in mir)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    print("\nmirror -- 하행 against 상행, weighted by the traffic it applies to")
    print("   median line %.1f %%, worst %.1f %% (%s)"
          % (100 * med, 100 * mir[0][0], mir[0][2]))
    # A line the map draws no rider figure on cannot have a meaningful mirror:
    # both directions are the fit's residual noise around a published total that
    # is not describing the railway. Say so beside the percentage rather than
    # leaving 경춘선's 113 % looking like a modelling failure to chase.
    greyed = set(f["properties"]["line"] for f in feats
                 if f["properties"].get("service") == "unrecorded")
    rough = [r for r in mir if r[0] > 0.05]
    if rough:
        print("   %-11s %9s %9s" % ("line", "weighted", "worst seg"))
        for w, worst, ln, _ in rough:
            print("   %-11s %8.1f%% %8.1f%%%s"
                  % (ln, 100 * w, 100 * worst,
                     "   (drawn without a rider figure)" if ln in greyed else ""))

    # --- passengers per train, against what a train holds ------------------
    over, mean_occ, per_line = occupancy(by_line, table)
    print("\n승차율 -- passengers per train against the published 운행횟수")
    print("   mean occupancy over sections with a count: %.0f per train "
          "(an upper bound)" % mean_occ)
    if over:
        print("   **%d segments over capacity, worst %.1fx** -- compare this "
              "against the\n   other builder rather than reading the count on "
              "its own." % (len(over), over[0][0]))
        print("   %-11s %-19s %8s %7s %9s %6s"
              % ("line", "segment", "명/일", "trains", "per train", "seats"))
        for ratio, ln, a, b, per, n, daily, seats in over[:12]:
            print("   %-11s %-19s %8d %7d %9.0f %6d  %.1fx"
                  % (ln, ("%s-%s" % (a, b))[:19], daily, n, per, seats, ratio))
        if len(over) > 12:
            print("   ... and %d more" % (len(over) - 12))
        print("   Trains shown are the published per-direction count doubled. "
              "경부고속선\n   천안아산-오송 is a known false positive -- its "
              "count changes at SR분기,\n   which has no platform, so "
              "runs_along reads 117 where 177 run.")
    else:
        print("   every segment fits inside its trains")
    if per_line:
        print("   emptiest lines (not a failure -- read the docstring):")
        for v, ln in per_line[:5]:
            print("      %-11s %6.1f per train%s"
                  % (ln, v, "   (drawn without a rider figure)"
                     if ln in greyed else ""))

    neg = [f for f in feats if f["properties"]["daily"] < 0]
    if neg:
        print("\n%d segments carry a negative load:" % len(neg))
        for f in sorted(neg, key=lambda z: z["properties"]["daily"])[:10]:
            p = f["properties"]
            print("   %-11s %-8s -> %-8s %9d 명/일"
                  % (p["line"], p["from"], p["to"], p["daily"]))
    else:
        print("no segment carries a negative load")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
