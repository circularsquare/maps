# -*- coding: utf-8 -*-
"""Resolve the station universe and write data/stations.json.

Model, following ../nycriders/: a rider boards at a physical *complex*, and
which platform they use is a routing decision, not data. So ridership attaches
to complexes while routing runs over (line, code) platforms.

  identity + order   timetable (역사코드, and the stop sequence of the trains)
                     -- 서울교통공사's file for lines 1-9, and kric.py's
                     timetable_extra.csv, in the same columns, for the rest
  coordinates        OSM, disambiguated by proximity to that line's own track
  measured hours     서울교통공사 2023 daily x hourly file
  existence in 2023  the OD file for the night we are drawing

Three wrinkles the sources force on us, all handled below:

  * The OD sometimes files a whole complex under a line we do not carry --
    에버라인 is the only one left now that the other 15 are in. Absorb those
    only when the complex has no rows on a line we do carry, so that
    genuinely distinct same-name stations (양평 on 5 vs 양평 on 경의중앙) do
    not swallow each other.
  * 당고개 was renamed 불암산 in 2024. The timetable is 2026, the ridership
    2023.
  * The timetable is 2026 and includes stations that did not exist on
    2023-12-31 -- the 8호선 별내 extension above all.

    python build_stations.py
"""

import collections
import csv
import io
import json
import math
import os
import re
import sys

import build_shapes as BS
import lines as LR

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(D, "stations.json")

OD_DATE = "2023-12-31"
LINES = LR.ALL_IDS

# timetable (2026) name -> ridership (2023) name
RENAMES = {
    "불암산": "당고개",
}

# Platforms the 2026 timetable carries that we deliberately leave out.
EXCLUDE = {
    # The 8호선 별내 extension opened August 2024. 구리 and 별내 sit at
    # complexes that did exist in 2023 on 경의중앙선 and 경춘선, so they pass
    # the OD presence test and have to be named explicitly.
    ("8", "암사역사공원"): "8호선 별내 extension, opened Aug 2024",
    ("8", "장자호수공원"): "8호선 별내 extension, opened Aug 2024",
    ("8", "구리"): "8호선 별내 extension, opened Aug 2024",
    ("8", "동구릉"): "8호선 별내 extension, opened Aug 2024",
    ("8", "다산"): "8호선 별내 extension, opened Aug 2024",
    ("8", "별내"): "8호선 별내 extension, opened Aug 2024",
    # The 연천 extension opened 2023-12-16, a fortnight before our date. 청산
    # has no OD rows at all and 연천 carries 545 trips, 0.016% of the day.
    # Neither is in OSM, and 연천 is a terminus so there is nothing to
    # interpolate between. Not worth inventing a coordinate for.
    ("1", "연천"): "연천 extension, 0.016% of trips and absent from OSM",
}

# how far a candidate OSM node may sit from the line's own track, in metres,
# before we call the match doubtful
COORD_TOL_M = 600.0

# Platforms grouped under one name but further apart than this are not one
# station. Seoul reuses names across distant places -- 양평 on line 5 is in
# 영등포구, 양평 on 경의중앙선 is 27 km east in 양평군 -- and averaging their
# coordinates puts a station in the river between them. Real interchanges stay
# well under this: even 서울역, with four lines, spreads about 400 m.
SPLIT_M = 1500.0

# Below this share of within-line trips, the OD is not describing the line's
# own ridership and the page says so. Seoul's nine lines run 36-55%; every
# other operator runs 0.3-12%, so anything in between would be a new case
# worth looking at rather than a boundary to argue over.
PARTIAL_BELOW = 0.20


def read_cp949(name):
    with io.open(os.path.join(D, name), encoding="cp949", errors="replace",
                 newline="") as f:
        return list(csv.DictReader(f))


def read_timetables():
    """The 서울교통공사 file plus kric.py's, which share their columns."""
    rows = read_cp949("timetable_raw.csv")
    extra = os.path.join(D, "timetable_extra.csv")
    if os.path.exists(extra):
        more = read_cp949("timetable_extra.csv")
        print("   +%s rows from timetable_extra.csv" % format(len(more), ","))
        rows += more
    else:
        print("   timetable_extra.csv not found -- lines 1-9 only. "
              "Run kric.py to add the rest.")
    return rows


def read_json(name):
    with io.open(os.path.join(D, name), encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    return re.sub(r"\s+", "", (s or "").strip())


def base(s):
    """잠실(송파구청) -> 잠실. The three sources disagree on the suffix."""
    return re.sub(r"\(.*?\)$", "", norm(s))


def clean_en(s):
    """Tidy an OSM name:en into what the page should print.

    Only a *trailing* bracket is a 부역명 to drop -- 'Jongno 3(sam)-ga' carries
    its bracket in the middle and is the whole name. 'Seoul (Station)' is the
    one node that brackets a word it means to keep.
    """
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"\s*\(Station\)$", " Station", s)
    s = re.sub(r"\s*\([^()]*\)$", "", s).strip()
    return s


# The five complexes with no OSM node of their own name, so no name:en either.
# Revised romanisation, as the operators' own signage spells them.
EN_FALLBACK = {
    "박촌": "Bakchon",
    "임학": "Imhak",
    "화전": "Hwajeon",
    "부천시청": "Bucheon City Hall",
    "서구청": "Seo-gu Office",
}


def line_label(s):
    """An OD 호선 label -> our line id, or None if we do not carry that line."""
    return LR.OD_TO_ID.get(norm(s))


def hhmmss(s):
    s = (s or "").strip()
    if not s or s.count(":") != 2:
        return None
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def cluster(pts, radius):
    """Single-linkage grouping of points, returning lists of indices."""
    groups = []
    for i, p in enumerate(pts):
        hit = [g for g in groups
               if any(metres(p, pts[j]) <= radius for j in g)]
        if not hit:
            groups.append([i])
            continue
        merged = [i]
        for g in hit:
            merged += g
            groups.remove(g)
        groups.append(sorted(merged))
    return sorted(groups, key=lambda g: g[0])


def split_far_apart(complexes, report):
    """One name, two places -> two complexes. See SPLIT_M."""
    out = collections.OrderedDict()
    for name, c in complexes.items():
        groups = cluster(c["pts"], SPLIT_M)
        if len(groups) == 1:
            out[name] = c
            continue
        for k, g in enumerate(groups):
            key = "%s#%d" % (name, k + 1)
            out[key] = {
                "id": key, "name": name, "display": c["display"],
                "name_en": c.get("name_en", ""),
                "platforms": [c["platforms"][i] for i in g],
                "pts": [c["pts"][i] for i in g],
            }
        report.append((name, [",".join(sorted(set(
            c["platforms"][i]["line"] for i in g))) for g in groups]))
    return out


def numeric_code(code):
    """역사코드 as the hourly file writes it, or '' for our synthetic codes."""
    try:
        return str(int(code))
    except (TypeError, ValueError):
        return ""


def metres(a, b):
    """Rough planar distance; fine at Seoul's latitude over a few km."""
    dy = (a[0] - b[0]) * 111320.0
    dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def along(pts, frac):
    """The point `frac` of the way along a polyline, by distance."""
    seg = [metres(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    if total <= 0:
        return pts[0]
    want, run = total * frac, 0.0
    for i, s in enumerate(seg):
        if run + s >= want:
            f = (want - run) / s if s > 0 else 0.0
            return (pts[i][0] + f * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + f * (pts[i + 1][1] - pts[i][1]))
        run += s
    return pts[-1]


def fill_gaps(plats, graph):
    """Place the stations OSM has no node for, along their line's own track.

    OSM misses five of our 626 complexes outright. Dropping them on the chord
    between their placed neighbours puts them wherever the track does not go --
    박촌 and 임학 on 인천 1호선 ended up 800 m and 650 m out into 계양구, well
    off the drawn line and visibly wrong. So walk the track between the
    neighbours instead and space the missing stations along *that*, which is at
    worst wrong about where on the line they sit rather than wrong about being
    on it. Falls back to the chord when there is no track to walk.

    Returns a list of (name, how) for the report.
    """
    done = []
    i = 0
    while i < len(plats):
        if plats[i]["pt"] is not None:
            i += 1
            continue
        j = i
        while j < len(plats) and plats[j]["pt"] is None:
            j += 1
        before = plats[i - 1]["pt"] if i > 0 else None
        after = plats[j]["pt"] if j < len(plats) else None
        if not before or not after:
            i = j
            continue

        gap = j - i
        path = None
        if graph is not None:
            adj, nodes = graph
            got = BS.path_between(adj, nodes,
                                  BS.candidates(nodes, before),
                                  BS.candidates(nodes, after),
                                  max(3000.0, metres(before, after) * 4.0))
            if got and len(got[0]) > 1:
                path = [before] + list(got[0]) + [after]
        for k in range(gap):
            frac = (k + 1) / float(gap + 1)
            if path:
                plats[i + k]["pt"] = along(path, frac)
                how = "along the track between neighbours"
            else:
                plats[i + k]["pt"] = (
                    before[0] + frac * (after[0] - before[0]),
                    before[1] + frac * (after[1] - before[1]))
                how = "between neighbours (no track to follow)"
            done.append((plats[i + k]["name"], how))
        i = j
    return done


# --------------------------------------------------------------------------
# trains and running order
# --------------------------------------------------------------------------

def load_runs(tt):
    runs = collections.defaultdict(list)
    for r in tt:
        line = norm(r["호선"])
        if line not in LINES:
            continue
        arr, dep = hhmmss(r["열차도착시간"]), hhmmss(r["열차출발시간"])
        t = dep if dep is not None else arr
        if t is None:
            continue
        key = (line, norm(r["주중주말"]), norm(r["방향"]), norm(r["열차코드"]))
        runs[key].append({"code": norm(r["역사코드"]), "t": t})
    for k in runs:
        runs[k].sort(key=lambda s: s["t"])
    return runs


def line_order(runs, line):
    """Merge train stop sequences into one running order for the line."""
    seqs = [[s["code"] for s in v] for k, v in runs.items() if k[0] == line]
    seqs = [list(dict.fromkeys(s)) for s in seqs if len(s) > 1]
    if not seqs:
        return []
    seqs.sort(key=len, reverse=True)
    order = list(seqs[0])
    known = set(order)
    for seq in seqs[1:]:
        for i, code in enumerate(seq):
            if code in known:
                continue
            before = next((seq[j] for j in range(i - 1, -1, -1)
                           if seq[j] in known), None)
            after = next((seq[j] for j in range(i + 1, len(seq))
                          if seq[j] in known), None)
            if before is not None:
                order.insert(order.index(before) + 1, code)
            elif after is not None:
                order.insert(order.index(after), code)
            else:
                order.append(code)
            known.add(code)
    return order


# --------------------------------------------------------------------------
# OSM
# --------------------------------------------------------------------------

def load_osm(routes, nodes):
    """Per-line track points and geometry, a name index, and the English names.

    `out geom` gives relation members coordinates but not tags, so the
    relations cannot name their stops. They can still *place* the line, which
    is all we need: pick, among the nodes carrying a station's name, the one
    nearest that line's own track.

    The same node pull carries `name:en` on 811 of its 813 stations, which is
    where the page's English gets its station names -- so the language toggle
    costs no extra download and no romanisation of our own. The five complexes
    with no OSM node at all fall back to EN_FALLBACK.
    """
    track_pts = collections.defaultdict(list)
    geometry = collections.defaultdict(list)
    # Line 1 alone has ~50 route relations, one per through-running pattern,
    # and they share almost all their track. Keep each way once per line.
    seen_ways = collections.defaultdict(set)
    for rel in routes["elements"]:
        line = LR.osm_line_of(rel.get("tags", {}).get("name", ""))
        if line is None:
            continue
        for m in rel.get("members", []):
            if m["type"] == "node" and "lat" in m:
                track_pts[line].append((m["lat"], m["lon"]))
            elif m["type"] == "way" and "geometry" in m:
                if m["ref"] in seen_ways[line]:
                    continue
                seen_ways[line].add(m["ref"])
                pts = [(p["lat"], p["lon"]) for p in m["geometry"]]
                if len(pts) > 1:
                    geometry[line].append(pts)
                    track_pts[line].extend(pts[::10])

    byname = collections.defaultdict(list)
    votes = collections.defaultdict(collections.Counter)
    for e in nodes["elements"]:
        t = e.get("tags", {})
        pt = (e["lat"], e["lon"])
        en = clean_en(t.get("name:en", ""))
        seen = set()
        for key in ("name", "name:ko"):
            nm = base(t.get(key, ""))
            for cand in (nm, nm[:-1] if nm.endswith("역") and len(nm) > 1 else nm):
                if cand and cand not in seen:
                    seen.add(cand)
                    byname[cand].append(pt)
                    if en:
                        votes[cand][en] += 1
    # 서울역 has one node per line and they do not all spell it the same way,
    # so take the spelling most of them agree on rather than whichever came
    # first out of the file.
    english = dict((nm, max(c.items(), key=lambda kv: (kv[1], -len(kv[0])))[0])
                   for nm, c in votes.items())
    return track_pts, geometry, byname, english


def place(name, line, byname, track_pts):
    """Nearest node bearing this name to this line's track."""
    cands = byname.get(name)
    if not cands:
        return None, None
    pts = track_pts.get(line)
    if not pts:
        return cands[0], None
    best, bestd = None, None
    for c in cands:
        d = min(metres(c, p) for p in pts)
        if bestd is None or d < bestd:
            best, bestd = c, d
    return best, bestd


# --------------------------------------------------------------------------

def main():
    print("reading timetables ...")
    tt = read_timetables()
    runs = load_runs(tt)
    print("   %d train runs across %d lines" % (len(runs), len(LINES)))

    code_name = {}
    for r in tt:
        line = norm(r["호선"])
        if line in LINES:
            code_name[norm(r["역사코드"])] = (line, base(r["역사명"]))

    print("reading hourly counts (2023) ...")
    hr = read_cp949("hourly_2023_raw.csv")
    measured_codes = set(str(int(norm(r["역번호"]))) for r in hr)
    measured_names = set((norm(r["호선"]).replace("호선", ""), base(r["역명"]))
                         for r in hr)

    print("reading OD (%s) ..." % OD_DATE)
    od = read_cp949("od_2023-12-31.csv")

    # name -> {line label -> (boardings, alightings)}
    od_by_name = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0]))
    for r in od:
        n = int(r["총_승객수"])
        od_by_name[base(r["승차_역"])][norm(r["승차_호선"])][0] += n
        od_by_name[base(r["하차_역"])][norm(r["하차_호선"])][1] += n
    print("   %d station names appear in the OD" % len(od_by_name))

    # How much of each line's own traffic the OD contains at all. A line
    # whose riders mostly travel within it should show a high within-line
    # share -- 2호선 is 55%. The lines outside 서울교통공사 sit near zero, because
    # the OD only holds trips touching Seoul's network, so what we draw for
    # them is their traffic to and from Seoul rather than their ridership.
    # See README, "What the OD actually contains".
    od_board = collections.Counter()
    od_within = collections.Counter()
    for r in od:
        n = int(r["총_승객수"])
        o = LR.OD_TO_ID.get(norm(r["승차_호선"]))
        d = LR.OD_TO_ID.get(norm(r["하차_호선"]))
        if o:
            od_board[o] += n
            if o == d:
                od_within[o] += n
    coverage = {}
    for l in LR.LINES:
        tot = od_board.get(l.id, 0)
        coverage[l.id] = round(od_within.get(l.id, 0) / tot, 4) if tot else 0.0
    print("   within-line trip share, by line (low = the OD holds only this")
    print("   line's trips to and from Seoul):")
    for lid in sorted(coverage, key=LR.order_key):
        print("      %-4s %-14s %5.1f%%%s"
              % (lid, LR.DISPLAY[lid], 100 * coverage[lid],
                 "   partial" if coverage[lid] < PARTIAL_BELOW else ""))

    print("reading OSM ...")
    track_pts, geometry, byname, english = load_osm(
        read_json("osm_routes.json"), read_json("osm_stations.json"))
    print("   name:en for %d of the names OSM knows" % len(english))

    missing_en = []

    def english_name(tt_name, rname):
        """The page's English label for a station, or '' if we have none."""
        for nm in (tt_name, rname):
            en = english.get(nm) or EN_FALLBACK.get(nm)
            if not en:
                continue
            # 역 is part of 서울역's name and the English keeps it, but some
            # mappers append "Station" to names that do not have it -- and
            # "Dongdaemun History & Culture Park Station" is long enough to
            # break the tooltip on its own.
            if en.endswith(" Station") and not nm.endswith("역"):
                en = en[:-len(" Station")]
            return en
        missing_en.append(rname)
        return ""

    print("   track points for lines: %s"
          % ",".join(sorted(track_pts, key=LR.order_key)))
    no_track = [l for l in LINES if l not in track_pts]
    if no_track:
        print("   NO OSM TRACK for: %s   (run fetch_osm.py)"
              % ",".join(sorted(no_track, key=LR.order_key)))

    # ----------------------------------------------------------------------
    # platforms, grouped into complexes by ridership name
    # ----------------------------------------------------------------------
    complexes = collections.OrderedDict()
    doubtful, absent, interpolated, unplaced = [], [], [], []

    # Only the handful of lines with a station OSM misses ever need one, so
    # build the routing graph on demand rather than for all 22.
    graph_cache = {}

    def track_graph(line):
        if line not in graph_cache:
            ways = geometry.get(line)
            graph_cache[line] = BS.build_graph(ways) if ways else None
        return graph_cache[line]

    for line in LINES:
        plats = []
        for seq, code in enumerate(line_order(runs, line)):
            if code not in code_name:
                continue
            _, tt_name = code_name[code]
            rname = RENAMES.get(tt_name, tt_name)

            if (line, tt_name) in EXCLUDE:
                absent.append((line, tt_name, EXCLUDE[(line, tt_name)]))
                continue
            if rname not in od_by_name:
                absent.append((line, tt_name, "no rows in the OD"))
                continue

            pt, dist = place(tt_name, line, byname, track_pts)
            if pt is not None and dist is not None and dist > COORD_TOL_M:
                doubtful.append((line, tt_name, int(dist)))
            plats.append({"seq": seq, "code": code, "name": tt_name,
                          "rname": rname, "pt": pt})

        # OSM misses a few stations outright. Place them along the track
        # between whichever neighbours on the line we did manage to locate,
        # rather than inventing a coordinate.
        for name, how in fill_gaps(plats, track_graph(line)):
            interpolated.append((line, name, how))
        for pl in plats:
            if pl["pt"] is None:
                unplaced.append((line, pl["name"]))

        for pl in plats:
            if pl["pt"] is None:
                continue
            c = complexes.setdefault(pl["rname"], {
                "id": pl["rname"], "name": pl["rname"], "display": pl["name"],
                "name_en": english_name(pl["name"], pl["rname"]),
                "platforms": [], "pts": [],
            })
            c["platforms"].append({"line": line, "code": pl["code"],
                                   "seq": pl["seq"]})
            c["pts"].append(pl["pt"])

    splits = []
    complexes = split_far_apart(complexes, splits)

    # ----------------------------------------------------------------------
    # attach ridership and measurement status
    # ----------------------------------------------------------------------
    absorbed = []
    for key, c in complexes.items():
        lines_here = set(p["line"] for p in c["platforms"])
        rows = od_by_name[c["name"]]

        own = dict((lbl, v) for lbl, v in rows.items()
                   if line_label(lbl) in lines_here)
        if own:
            used = own
        else:
            # the OD filed this whole complex under a line we do not carry
            used = rows
            absorbed.append((c["name"], ",".join(sorted(rows))))

        # Which OD 호선 labels belong to this complex rather than to another
        # station of the same name. build_od.py looks trips up on this.
        c["od_labels"] = sorted(used)
        c["od_boardings"] = sum(v[0] for v in used.values())
        c["od_alightings"] = sum(v[1] for v in used.values())

        # The hourly file covers 서울교통공사's own lines only, so every
        # platform on a line it does not report is unmeasured by definition.
        meas = [(numeric_code(p["code"]) in measured_codes
                 or (p["line"], c["display"]) in measured_names)
                for p in c["platforms"]]
        c["measured"] = all(meas)
        c["partly_measured"] = any(meas) and not all(meas)

        lat = sum(p[0] for p in c["pts"]) / len(c["pts"])
        lon = sum(p[1] for p in c["pts"]) / len(c["pts"])
        c["lat"], c["lon"] = round(lat, 6), round(lon, 6)
        c["spread_m"] = int(max(metres(p, (lat, lon)) for p in c["pts"]))
        del c["pts"]

    # ----------------------------------------------------------------------
    out_complexes = list(complexes.values())
    n_plat = sum(len(c["platforms"]) for c in out_complexes)
    n_meas = sum(1 for c in out_complexes if c["measured"])
    n_part = sum(1 for c in out_complexes if c["partly_measured"])

    print("\ncomplexes: %d   platforms: %d" % (len(out_complexes), n_plat))
    print("   fully measured hourly:  %d" % n_meas)
    print("   partly measured:        %d" % n_part)
    print("   hours to be inferred:   %d"
          % (len(out_complexes) - n_meas - n_part))

    print("\ndropped: %d" % len(absent))
    for line, name, why in absent:
        print("   line %-3s %-14s %s" % (line, name, why))

    if splits:
        print("\nsame name, different place -- split into separate complexes: %d"
              % len(splits))
        for name, groups in splits:
            print("   %-14s %s" % (name, "  |  ".join(groups)))

    if absorbed:
        print("\nfiled by the OD under a line we do not carry, absorbed: %d"
              % len(absorbed))
        for name, lbls in absorbed:
            print("   %-14s <- %s" % (name, lbls))

    if interpolated:
        print("\nnot in OSM, positioned between neighbours: %d"
              % len(interpolated))
        for line, name, how in interpolated:
            print("   line %-3s %-14s %s" % (line, name, how))

    if unplaced:
        print("\nCOULD NOT PLACE AT ALL: %d" % len(unplaced))
        for line, name in unplaced:
            print("   line %-3s %s" % (line, name))

    if missing_en:
        # The page falls back to the Korean, so this is a gap in the label
        # rather than a broken build -- but it should stay at zero.
        print("\nNO ENGLISH NAME (add to EN_FALLBACK): %d"
              % len(set(missing_en)))
        for name in sorted(set(missing_en)):
            print("   %s" % name)

    if doubtful:
        print("\ncoordinate more than %dm from the line's track: %d"
              % (COORD_TOL_M, len(doubtful)))
        for line, name, d in doubtful:
            print("   line %-3s %-14s %6dm" % (line, name, d))

    wide = [c for c in out_complexes if c["spread_m"] > 1500]
    if wide:
        print("\ncomplexes whose platforms sit far apart (suspect merge): %d"
              % len(wide))
        for c in wide:
            print("   %-14s %5dm  %s" % (c["name"], c["spread_m"],
                  ",".join(p["line"] for p in c["platforms"])))

    per_line = collections.Counter(p["line"] for c in out_complexes
                                   for p in c["platforms"])
    print("\nplatforms per line:")
    for lid in sorted(per_line, key=LR.order_key):
        print("   %-4s %-14s %4d" % (lid, LR.DISPLAY.get(lid, lid),
                                     per_line[lid]))
    empty = [l for l in LINES if l not in per_line]
    if empty:
        print("   NO PLATFORMS AT ALL: %s"
              % ",".join(sorted(empty, key=LR.order_key)))

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"date": OD_DATE, "lines": LINES,
                   "line_meta": dict(
                       (l.id, {"display": l.display,
                               "display_en": l.display_en,
                               "color": l.color,
                               "capacity": l.capacity,
                               "within_share": coverage[l.id],
                               "partial": coverage[l.id] < PARTIAL_BELOW})
                       for l in LR.LINES),
                   "complexes": out_complexes,
                   "geometry": dict(geometry)}, f, ensure_ascii=False)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
