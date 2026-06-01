"""
Build geometry.json from cached TfL Unified API responses + NUMBAT definitions.

Outputs geometry.json with:
  stations: { mnlc: { name, lat, lon, naptan_id, asc, lines: [...] } }
  branches: [ { line, dir, branch_id, mnlcs: [...], naptan_ids: [...], distances_m: [...] } ]
  lines:    [ { line, color, coords: [[lon, lat], ...] } ]   (curved polylines from API)
  line_colors: { line_code: hex }
  naptan_to_mnlc: { naptan_id: mnlc, ... }   (used by build.py)
  numbat_line_to_tfl: { code: tfl_id, ... }   (used by build.py)

Key challenge: TfL `icsCode` (e.g. '1000011') is unrelated to NUMBAT `MasterNLC`
(e.g. 511). We map by normalized station name with a small override dict.
"""
import heapq
import json
import math
import os
import re
from collections import defaultdict

import openpyxl

CACHE_DIR = "data/api_cache"
DEF_XLSX = "data/numbat_definitions.xlsx"
OSM_JSON = "data/api_cache/osm_routes.json"
OUT_JSON = "geometry.json"

# Douglas-Peucker tolerance for OSM track polylines (metres). Light simplification
# to follow real curvature closely without keeping every survey vertex.
SIMPLIFY_TOL_M = 10.0

# Official TfL line colors. Same palette as before — keep NUMBAT codes as keys so
# the existing index.html legend works. New Overground branches each get the
# canonical orange family (we can refine later).
LINE_COLORS = {
    "BAK": "#B36305",  # Bakerloo
    "CEN": "#E32017",  # Central
    "CIR": "#FFD300",  # Circle
    "DIS": "#00782A",  # District
    "HAM": "#F3A9BB",  # Hammersmith & City
    "JUB": "#A0A5A9",  # Jubilee
    "MET": "#9B0056",  # Metropolitan
    "NOR": "#000000",  # Northern
    "PIC": "#003688",  # Piccadilly
    "VIC": "#0098D4",  # Victoria
    "WAC": "#95CDBA",  # Waterloo & City
    "DLR": "#00A4A7",  # DLR
    "EZL": "#6950A1",  # Elizabeth line
    "NLL": "#FA7B05",  # Mildmay (North London)
    "ELL": "#FA7B05",  # Windrush (East London)
    "WEL": "#FA7B05",  # Lioness (Watford DC)
    "GOB": "#FA7B05",  # Suffragette (Gospel Oak - Barking)
    "URL": "#FA7B05",  # Liberty (Romford)
    "WAG": "#FA7B05",  # Weaver (West Anglia)
    "TRM": "#84B817",  # Trams (no API coverage)
}
DEFAULT_COLOR = "#808183"

# NUMBAT line code -> TfL line id used by the Unified API.
# Some Overground branches in NUMBAT cover multiple TfL lines after the 2024
# rename; we list the primary TfL id here.
NUMBAT_TO_TFL = {
    "BAK": "bakerloo",
    "CEN": "central",
    "CIR": "circle",
    "DIS": "district",
    "DLR": "dlr",
    "EZL": "elizabeth",
    "HAM": "hammersmith-city",
    "JUB": "jubilee",
    "MET": "metropolitan",
    "NOR": "northern",
    "PIC": "piccadilly",
    "VIC": "victoria",
    "WAC": "waterloo-city",
    "ELL": "windrush",
    "GOB": "suffragette",
    "NLL": "mildmay",
    "URL": "liberty",
    "WAG": "weaver",
    "WEL": "lioness",
}


# OSM relation `ref` tag -> NUMBAT line code. Covers the 11 LU lines and the
# 6 2024-renamed Overground "lines". DLR and Elizabeth aren't ref-tagged per
# branch, so they're matched on `network` instead (see OSM_NETWORK_TO_LINE).
OSM_REF_TO_LINE = {
    "Bakerloo": "BAK",
    "Central": "CEN",
    "Circle": "CIR",
    "District": "DIS",
    "Hammersmith & City": "HAM",
    "Jubilee": "JUB",
    "Metropolitan": "MET",
    "Northern": "NOR",
    "Piccadilly": "PIC",
    "Victoria": "VIC",
    "Waterloo & City": "WAC",
    "Mildmay": "NLL",
    "Lioness": "WEL",
    "Windrush": "ELL",
    "Suffragette": "GOB",
    "Liberty": "URL",
    "Weaver": "WAG",
    "Elizabeth": "EZL",
}
OSM_NETWORK_TO_LINE = {
    "Docklands Light Railway": "DLR",
}

# NUMBAT data fixes: stations whose published lat/lon are wrong.
NUMBAT_COORD_FIXES = {
    831: (51.4805, -0.1283),   # Nine Elms (NUMBAT had positive longitude)
    832: (51.4795, -0.1418),   # Battersea Power Station (same)
}


# Map { normalized TfL name -> normalized NUMBAT name } for cases where
# normalization can't bridge the gap (different physical stations, custom
# disambiguators, paren tags that don't line up).
NAME_OVERRIDES = {
    # Bank & Monument is one NUMBAT entry (513) shared by both Bank and Monument.
    "bank": "bank and monument",
    "monument": "bank and monument",
    # Edgware Road: TfL has 2 naptans (Bakerloo + Circle/Dist/H&C/Met).
    # NUMBAT has 2 MNLCs: 774 "(Bak)" and 569 "(DIS)".
    "edgware road (bakerloo)": "edgware road (bak)",
    "edgware road (circle line)": "edgware road (dis)",
    # Hammersmith: TfL has 2 naptans, NUMBAT splits as (DIS) and (H&C).
    "hammersmith (h and c line)": "hammersmith (handc)",
    "hammersmith (handc line)": "hammersmith (handc)",
    # The District/Piccadilly Hammersmith normalizes to "(distandpicc line)"
    # since we replace & with "and" without surrounding spaces.
    "hammersmith (distandpicc line)": "hammersmith (dis)",
    "hammersmith (dist and picc line)": "hammersmith (dis)",
    # Shepherd's Bush Central is NUMBAT "Shepherd's Bush LU" (700);
    # Shepherd's Bush (Overground/H&C-Market) is "Shepherd's Bush Market".
    "shepherd's bush (central)": "shepherd's bush",
    # Paddington has multiple flavors in NUMBAT (TfL/HEx/NR). LU stop
    # collapses to "paddington tfl" (670).
    "paddington": "paddington tfl",
    # Heathrow Terminals 2 & 3: NUMBAT name has no spaces in '123' (mnlc 780).
    "heathrow terminals 2 and 3": "heathrow terminals 123",
    # DLR-station-name-with-extra-detail vs NUMBAT bare-name pairs.
    # NUMBAT names "Cutty Sark" / "Custom House DLR"; after normalize-strip
    # the latter becomes "custom house".
    "cutty sark (for maritime greenwich)": "cutty sark",
    "custom house (for excel)": "custom house",
    # Mainline terminus stations: TfL prefixes "London" on the rail-station
    # naming; NUMBAT just uses the bare name. Without these overrides the
    # final Overground/Elizabeth segments into these termini get filtered
    # out of branches and the renderer falls back to a straight line.
    "london euston": "euston",
    "london liverpool street": "liverpool street",
    "london paddington": "paddington",
    # Windrush: New Cross terminus uses an ELL-suffixed naptan name in TfL
    # data; Norwood Junction is "Jn" in NUMBAT. Without these the Windrush
    # branch drops the New Cross terminus and the West Croydon ↔ Anerley
    # segment skips Norwood Junction in between.
    "new cross ell": "new cross",
    "norwood junction": "norwood jn",
}


# NUMBAT suffixes that identify the operator at an interchange site.
# Stripping these collapses "Marylebone LU"/"Marylebone NR" to "marylebone"
# so they match the bare TfL name; we then disambiguate by lat/lon proximity.
NUMBAT_OPERATOR_SUFFIXES = (" lu", " nr", " lo", " hex", " tfl", " dlr", " el")

# TfL suffixes that identify the operator/mode and don't carry
# disambiguating information.
TFL_STATION_SUFFIXES = (
    " underground station", " dlr station", " rail station",
    " overground station", " elizabeth line station",
    " station", " (london)",
)


def normalize_name(s):
    """Lowercase, strip operator suffixes, normalize ampersands."""
    if not s:
        return ""
    n = s.lower().strip()
    n = n.replace(".", "")  # St. Paul's -> St Paul's
    # Apostrophes are inconsistent: TfL returns "Shepherds Bush Rail Station"
    # (none) but NUMBAT has "Shepherd's Bush LU" (straight) and other sources
    # use the curly variant. Stripping all forms keeps both sides comparable.
    n = n.replace("'", "").replace("’", "")
    for suffix in TFL_STATION_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    for suffix in NUMBAT_OPERATOR_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    n = n.replace(" & ", " and ")
    n = n.replace("&", "and")
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_numbat_stations():
    print(f"Loading NUMBAT stations...")
    wb = openpyxl.load_workbook(DEF_XLSX, read_only=True, data_only=True)
    ws = wb["Stations"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    idx = {h: i for i, h in enumerate(header)}
    stations = {}
    for row in rows:
        mnlc = row[idx["MasterNLC"]]
        if mnlc is None:
            continue
        if not row[idx["Active"]]:
            continue
        lat = row[idx["Latitude"]]
        lon = row[idx["Longitude"]]
        if lat is None or lon is None:
            continue
        mnlc = int(mnlc)
        lat, lon = float(lat), float(lon)
        # NUMBAT bug: a few stations have a sign-flipped longitude that places
        # them in southeast London instead of central. Override with the real
        # coords so they don't sit miles from the rest of their line.
        if mnlc in NUMBAT_COORD_FIXES:
            lat, lon = NUMBAT_COORD_FIXES[mnlc]
        stations[mnlc] = {
            "name": row[idx["UniqueStationName"]],
            "asc": row[idx["MasterASC"]],
            "lat": lat,
            "lon": lon,
        }
    print(f"  {len(stations)} active NUMBAT stations")
    return stations


def apply_tfl_coord_overrides(numbat_stations, all_stops, naptan_to_mnlc,
                              warn_threshold_m=80):
    """Replace each NUMBAT station's lat/lon with the average of TfL's coords
    for the naptans mapped to that mnlc. NUMBAT often places stations 50-300m
    off the actual platform; TfL's stops API is consistently within a few
    metres of the track. Per-naptan granularity is kept separately in
    naptan_coords for train animation; this average is just for the static
    mnlc-keyed station marker."""
    coords_by_mnlc = defaultdict(list)
    for s in all_stops:
        nid = s.get("naptan_id")
        m = naptan_to_mnlc.get(nid)
        if m is None or s.get("lat") is None or s.get("lon") is None:
            continue
        coords_by_mnlc[m].append((s["lat"], s["lon"]))

    overridden = 0
    big_moves = 0
    for mnlc, coords in coords_by_mnlc.items():
        if mnlc not in numbat_stations:
            continue
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)
        old = numbat_stations[mnlc]
        delta = haversine_m(old["lat"], old["lon"], avg_lat, avg_lon)
        old["lat"] = avg_lat
        old["lon"] = avg_lon
        overridden += 1
        if delta > warn_threshold_m:
            big_moves += 1
    print(f"  TfL coords override: {overridden} stations updated"
          f" ({big_moves} moved >{warn_threshold_m}m)")


def load_tfl_lines():
    with open(f"{CACHE_DIR}/lines.json", encoding="utf-8") as f:
        return json.load(f)


def load_tfl_stops_per_line(tfl_id):
    """Return list of stop dicts {naptan_id, name, lat, lon, ics_code} for one line."""
    path = f"{CACHE_DIR}/stops_{tfl_id}.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        sps = json.load(f)
    out = []
    for s in sps:
        nid = s.get("naptanId")
        if not nid:
            continue
        out.append({
            "naptan_id": nid,
            "name": s.get("commonName", ""),
            "lat": float(s["lat"]) if s.get("lat") is not None else None,
            "lon": float(s["lon"]) if s.get("lon") is not None else None,
            "ics_code": s.get("icsCode"),
        })
    return out


# TfL line mode -> the NUMBAT operator suffix that names that mode's variant
# of an interchange. Bethnal Green has both an LU entry (Central) and an LO
# entry (Suffragette/Overground) — when matching the LO naptan we want the LO
# MNLC, and vice versa. This bias prevents collocated Tube/Overground sites
# from collapsing to one MNLC.
TFL_MODE_TO_NUMBAT_OPERATOR = {
    # Tuple of preferred NUMBAT operator suffixes, tried in order. The matcher
    # picks the first one that has a candidate at the same name. Falls back to
    # all candidates only if none of the preferences match.
    #
    # Most modes have a single suffix, but Overground and Elizabeth often map
    # to NR-tagged entries in NUMBAT (e.g. mnlc 9587 "Shepherd's Bush NR" is
    # the Overground platform; mnlc 6965 "Liverpool Street NR" is the Elizabeth
    # platform). Without the NR fallback those MNLCs go unresolved and ~35%
    # of Elizabeth riders & a third of Mildmay riders end up unassigned.
    "tube": ("lu",),
    "overground": ("lo", "nr"),
    "dlr": ("dlr",),
    "elizabeth-line": ("el", "nr"),
    "national-rail": ("nr",),
}


def build_per_line_mnlc_map(numbat_stations, tfl_lines):
    """Build (line_id, naptan_id) -> mnlc.

    Walks per TfL line so we can prefer a NUMBAT entry whose operator suffix
    matches the line's mode (LO for Overground, LU for Tube, etc).
    Also returns the flat naptan_id -> mnlc fallback (first MNLC seen for each
    naptan, used by trip-flatten which doesn't know the line of the trip's
    intermediate stops).
    """
    print("Mapping (line, naptan) -> NUMBAT mnlc...")

    # Index NUMBAT stations by normalized base name AND by the operator suffix
    # we stripped (so we can prefer matching operators).
    by_norm = defaultdict(list)   # norm_name -> [(mnlc, ns, op_suffix), ...]
    for mnlc, s in numbat_stations.items():
        original = (s["name"] or "").lower().strip().replace(".", "")
        op_suffix = ""
        for sfx in NUMBAT_OPERATOR_SUFFIXES:
            if original.endswith(sfx):
                op_suffix = sfx.strip()
                break
        norm = normalize_name(s["name"])
        by_norm[norm].append((mnlc, s, op_suffix))

    pair_map = {}            # (line_id, naptan_id) -> mnlc
    flat_map = {}            # naptan_id -> mnlc (first match wins)
    unmatched = []           # (line_id, naptan_id, name)

    for ln in tfl_lines:
        line_id = ln["id"]
        # Cached lines.json uses the raw TfL API key 'modeName' (not 'mode').
        # If preferred_ops is empty, the operator-preference filter never fires
        # and lat/lon disambiguation picks the wrong NUMBAT entry at sites with
        # multiple operators (e.g. Shadwell LU vs Shadwell DLR vs Shadwell LO).
        mode = ln.get("modeName") or ln.get("mode") or ""
        preferred_ops = TFL_MODE_TO_NUMBAT_OPERATOR.get(mode, ())
        for stop in load_tfl_stops_per_line(line_id):
            nid = stop["naptan_id"]
            norm = normalize_name(stop["name"])
            # Direct match first; otherwise try the override table.
            cands = by_norm.get(norm)
            if not cands:
                target = NAME_OVERRIDES.get(norm)
                if target:
                    cands = by_norm.get(target)
            if not cands:
                unmatched.append((line_id, nid, stop["name"]))
                continue

            # Try each preferred operator suffix in order. First non-empty
            # match wins; otherwise fall back to all candidates and let
            # lat/lon disambiguation pick the closest.
            pool = None
            for pop in preferred_ops:
                same_op = [c for c in cands if c[2] == pop]
                if same_op:
                    pool = same_op
                    break
            if pool is None:
                pool = cands
            if len(pool) == 1:
                mnlc = pool[0][0]
            else:
                best = pool[0][0]
                best_d = float("inf")
                for mnlc, ns, _ in pool:
                    if stop["lat"] is None:
                        best = mnlc
                        break
                    d = (ns["lat"] - stop["lat"]) ** 2 + (ns["lon"] - stop["lon"]) ** 2
                    if d < best_d:
                        best_d = d
                        best = mnlc
                mnlc = best

            pair_map[(line_id, nid)] = mnlc
            flat_map.setdefault(nid, mnlc)

    print(f"  matched {len(pair_map)} (line, naptan) pairs -> {len(flat_map)} unique naptans")
    if unmatched:
        # Group by name for compactness
        by_name = defaultdict(list)
        for line_id, nid, name in unmatched:
            by_name[name].append((line_id, nid))
        print(f"  {len(unmatched)} unmatched across {len(by_name)} unique names (first 15):")
        for name, refs in list(by_name.items())[:15]:
            tag = ", ".join(sorted({lid for lid, _ in refs}))
            print(f"    {name!r}  on lines: {tag}")
    return pair_map, flat_map


def perp_distance_m(p, a, b):
    """Perpendicular distance from point p to segment a-b, in metres.
    Inputs are [lon, lat]. Local equirectangular projection — fine at this scale."""
    R = 6_371_000
    cos_lat = math.cos(math.radians(a[1]))
    bx = math.radians(b[0] - a[0]) * cos_lat * R
    by = math.radians(b[1] - a[1]) * R
    px = math.radians(p[0] - a[0]) * cos_lat * R
    py = math.radians(p[1] - a[1]) * R
    L2 = bx * bx + by * by
    if L2 == 0:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * bx + py * by) / L2))
    cx, cy = t * bx, t * by
    return math.hypot(px - cx, py - cy)


def simplify_dp(coords, tol_m):
    """Iterative Douglas-Peucker. coords are [lon, lat]; tolerance in metres."""
    n = len(coords)
    if n < 3 or tol_m <= 0:
        return list(coords)
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        a, b = coords[i0], coords[i1]
        max_d = 0.0
        max_idx = -1
        for k in range(i0 + 1, i1):
            d = perp_distance_m(coords[k], a, b)
            if d > max_d:
                max_d = d
                max_idx = k
        if max_idx >= 0 and max_d > tol_m:
            keep[max_idx] = True
            stack.append((i0, max_idx))
            stack.append((max_idx, i1))
    return [coords[i] for i in range(n) if keep[i]]


def relation_line_code(rel):
    tags = rel.get("tags") or {}
    ref = tags.get("ref")
    if ref and ref in OSM_REF_TO_LINE:
        return OSM_REF_TO_LINE[ref]
    network = tags.get("network")
    if network and network in OSM_NETWORK_TO_LINE:
        return OSM_NETWORK_TO_LINE[network]
    return None


def build_lines_from_osm(osm_path, simplify_tol_m=SIMPLIFY_TOL_M):
    """Read raw Overpass JSON, return per-line polyline features.

    Output shape matches the existing `lines_out` schema:
      { line, branch_id, branch_name, color, coords }
    Each member way of each route relation becomes one feature; ways are
    deduped per line code so the same physical track isn't drawn twice.
    """
    if not os.path.exists(osm_path):
        print(f"  WARNING: {osm_path} missing; run fetch_track_shapes.py first")
        return []
    print(f"Loading OSM track shapes from {osm_path}...")
    with open(osm_path, encoding="utf-8") as f:
        osm = json.load(f)

    nodes = {}      # id -> [lon, lat]
    ways = {}       # id -> [node_id, ...]
    relations = []
    for e in osm.get("elements", []):
        t = e.get("type")
        if t == "node":
            nodes[e["id"]] = [e["lon"], e["lat"]]
        elif t == "way":
            ways[e["id"]] = e.get("nodes", []) or []
        elif t == "relation":
            relations.append(e)
    print(f"  {len(nodes)} nodes, {len(ways)} ways, {len(relations)} relations")

    seen_per_line = defaultdict(set)
    branch_counters = defaultdict(int)
    lines_out = []
    skipped = 0

    for rel in relations:
        code = relation_line_code(rel)
        if code is None:
            skipped += 1
            continue
        rel_name = (rel.get("tags") or {}).get("name") or f"{code} relation"
        for m in rel.get("members", []) or []:
            if m.get("type") != "way":
                continue
            wid = m.get("ref")
            if wid is None or wid in seen_per_line[code]:
                continue
            seen_per_line[code].add(wid)
            node_ids = ways.get(wid)
            if not node_ids or len(node_ids) < 2:
                continue
            coords = [nodes[nid] for nid in node_ids if nid in nodes]
            if len(coords) < 2:
                continue
            simplified = simplify_dp(coords, simplify_tol_m)
            simplified = [[round(p[0], 5), round(p[1], 5)] for p in simplified]
            lines_out.append({
                "line": code,
                "branch_id": branch_counters[code],
                "branch_name": rel_name,
                "color": LINE_COLORS.get(code, DEFAULT_COLOR),
                "coords": simplified,
            })
            branch_counters[code] += 1

    by_line = defaultdict(int)
    pts_by_line = defaultdict(int)
    for L in lines_out:
        by_line[L["line"]] += 1
        pts_by_line[L["line"]] += len(L["coords"])
    print(f"  {len(lines_out)} polylines across {len(by_line)} lines"
          f" ({sum(pts_by_line.values())} pts after {simplify_tol_m:g}m simplify)")
    for k in sorted(by_line):
        print(f"    {k}: {by_line[k]} ways, {pts_by_line[k]} pts")
    if skipped:
        print(f"  ({skipped} relations skipped — no line code match)")
    return lines_out


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_per_line_graphs(osm_path):
    """Per-line OSM-track adjacency graphs for Dijkstra-based station routing.

    Returns: { line_code: {'adj': {nid: [(neighbor, dist_m), ...]},
                           'coords': {nid: [lon, lat]}} }
    """
    if not os.path.exists(osm_path):
        return {}
    print(f"Building per-line OSM graphs from {osm_path}...")
    with open(osm_path, encoding="utf-8") as f:
        osm = json.load(f)

    nodes_lat_lon = {}
    ways = {}
    relations = []
    for e in osm.get("elements", []):
        t = e.get("type")
        if t == "node":
            nodes_lat_lon[e["id"]] = (e["lat"], e["lon"])
        elif t == "way":
            ways[e["id"]] = e.get("nodes") or []
        elif t == "relation":
            relations.append(e)

    ways_per_line = defaultdict(set)
    for rel in relations:
        code = relation_line_code(rel)
        if code is None:
            continue
        for m in rel.get("members", []) or []:
            if m.get("type") == "way":
                wid = m.get("ref")
                if wid in ways:
                    ways_per_line[code].add(wid)

    out = {}
    stitched_total = 0
    for code, way_ids in ways_per_line.items():
        adj = defaultdict(list)
        used = set()
        for wid in way_ids:
            ns = ways[wid]
            for i in range(len(ns) - 1):
                a, b = ns[i], ns[i + 1]
                if a not in nodes_lat_lon or b not in nodes_lat_lon:
                    continue
                la, lo_a = nodes_lat_lon[a]
                lb, lo_b = nodes_lat_lon[b]
                d = haversine_m(la, lo_a, lb, lo_b)
                adj[a].append((b, d))
                adj[b].append((a, d))
                used.add(a)
                used.add(b)
        coords = {n: [nodes_lat_lon[n][1], nodes_lat_lon[n][0]] for n in used}
        # OSM commonly tags up and down tracks as separate ways without sharing
        # nodes, splitting the graph into disconnected components. Stitch any
        # node-pairs that sit within ~30m of each other so Dijkstra can route
        # across parallel tracks.
        stitched_total += stitch_nearby_nodes(adj, coords, max_dist_m=30)
        out[code] = {"adj": dict(adj), "coords": coords}

    print(f"  graphs for {len(out)} lines"
          f" ({sum(len(g['coords']) for g in out.values())} total nodes,"
          f" {stitched_total} cross-track stitches)")
    return out


def stitch_nearby_nodes(adj, coords, max_dist_m=30):
    """Add edges between graph nodes that are physically close but not yet
    directly connected. Lets Dijkstra route across parallel tracks/platforms."""
    grid_deg = max_dist_m / 111_000  # 1 deg ≈ 111km
    grid = defaultdict(list)
    for nid, (lon, lat) in coords.items():
        grid[(int(lat / grid_deg), int(lon / grid_deg))].append(nid)
    added = 0
    for nid, (lon, lat) in coords.items():
        gx = int(lat / grid_deg)
        gy = int(lon / grid_deg)
        existing = {v for v, _ in adj.get(nid, [])}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((gx + dx, gy + dy), []):
                    if other <= nid or other in existing:
                        continue
                    olon, olat = coords[other]
                    d = haversine_m(lat, lon, olat, olon)
                    if d <= max_dist_m:
                        adj[nid].append((other, d))
                        adj[other].append((nid, d))
                        existing.add(other)
                        added += 1
    return added


def snap_to_graph(lat, lon, coords):
    """Find graph node closest to (lat, lon). coords is {node_id: [lon, lat]}.
    Returns (node_id, distance_m). Brute force; line graphs are small enough."""
    best = None
    best_d2 = float("inf")
    for nid, (clon, clat) in coords.items():
        d2 = (clat - lat) ** 2 + (clon - lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = nid
    if best is None:
        return None, float("inf")
    nlon, nlat = coords[best]
    return best, haversine_m(lat, lon, nlat, nlon)


def k_nearest_in_graph(lat, lon, coords, k=6, max_dist_m=400):
    """Top-k nearest graph nodes within max_dist_m. Used so we can try several
    candidate snaps per station — major interchanges like Stratford have many
    platform tracks, and a single nearest-node snap can land on one that's only
    reachable by absurd detour."""
    cands = []
    for nid, (clon, clat) in coords.items():
        d2 = (clat - lat) ** 2 + (clon - lon) ** 2
        cands.append((d2, nid))
    cands.sort(key=lambda x: x[0])
    out = []
    for _, nid in cands[: k * 2]:  # over-fetch then filter by real distance
        nlon, nlat = coords[nid]
        dm = haversine_m(lat, lon, nlat, nlon)
        if dm > max_dist_m:
            continue
        out.append((nid, dm))
        if len(out) >= k:
            break
    return out


def dijkstra_path(adj, src, dst, max_dist_m=20_000):
    """Shortest path in metres from src to dst. Returns list of node_ids, or None."""
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    pq = [(0.0, src)]
    dist_to = {src: 0.0}
    came_from = {}
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            path = [u]
            while u in came_from:
                u = came_from[u]
                path.append(u)
            return path[::-1]
        if d > dist_to.get(u, float("inf")) or d > max_dist_m:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist_to.get(v, float("inf")):
                dist_to[v] = nd
                came_from[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def _polyline_len_m(poly):
    L = 0.0
    for i in range(1, len(poly)):
        L += haversine_m(poly[i - 1][1], poly[i - 1][0], poly[i][1], poly[i][0])
    return L


def load_osm_stop_positions(osm_path):
    """Per-line stop_position node coords from OSM relations. Each route
    relation has node members tagged role=stop (or stop_entry_only /
    stop_exit_only) at the exact platform stop position. Returns:
      { line_code: [(lat, lon), ...] }
    """
    if not os.path.exists(osm_path):
        return {}
    with open(osm_path, encoding="utf-8") as f:
        osm = json.load(f)
    nodes_lat_lon = {}
    relations = []
    for e in osm.get("elements", []):
        t = e.get("type")
        if t == "node":
            nodes_lat_lon[e["id"]] = (e["lat"], e["lon"])
        elif t == "relation":
            relations.append(e)
    out = defaultdict(list)
    seen_per_line = defaultdict(set)
    for rel in relations:
        code = relation_line_code(rel)
        if code is None:
            continue
        for m in rel.get("members", []) or []:
            if m.get("type") != "node":
                continue
            role = (m.get("role") or "").lower()
            if "stop" not in role:
                continue
            nid = m.get("ref")
            if nid in seen_per_line[code]:
                continue
            if nid in nodes_lat_lon:
                seen_per_line[code].add(nid)
                lat, lon = nodes_lat_lon[nid]
                out[code].append((lat, lon))
    print(f"  OSM stop_position nodes: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(out.items())))
    return dict(out)


def build_per_line_station_coords(graphs, branches, naptan_coords, osm_path,
                                  max_dist_m=200):
    """Per-(line, naptan) coord. Prefers OSM stop_position nodes (role=stop
    members of route relations) since OSM contributors place these on the
    actual platform face. Falls back to nearest track-node snap if no
    stop_position lies within range — useful when OSM coverage is patchy.
    Lets single-naptan complexes (Embankment, Charing Cross) split per line."""
    stops_per_line = load_osm_stop_positions(osm_path)

    out = {}
    n_stop = n_track = n_skip = 0
    seen = set()
    for br in branches:
        line = br["line"]
        line_out = out.setdefault(line, {})
        stops = stops_per_line.get(line, [])
        graph_coords = graphs.get(line, {}).get("coords") if graphs else None
        for n in br["naptan_ids"]:
            key = (line, n)
            if key in seen:
                continue
            seen.add(key)
            base = naptan_coords.get(n)
            if not base:
                n_skip += 1
                continue
            # 1) prefer nearest stop_position, then snap THAT to a track node.
            # We always snap to a real graph node — link_shape endpoints are
            # graph-node coords too, so storing the same canonical node here
            # means line-end and station-coord agree exactly (no offset, no
            # backtrack zigzag from snap to platform).
            best_d = float("inf")
            best = None
            for lat, lon in stops:
                d = (lat - base[0]) ** 2 + (lon - base[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = (lat, lon)
            if best is not None:
                dist = haversine_m(base[0], base[1], best[0], best[1])
                if dist <= max_dist_m:
                    if graph_coords:
                        nid, _ = snap_to_graph(best[0], best[1], graph_coords)
                        if nid is not None:
                            nlon, nlat = graph_coords[nid]
                            line_out[n] = [round(nlat, 6), round(nlon, 6)]
                            n_stop += 1
                            continue
                    line_out[n] = [round(best[0], 6), round(best[1], 6)]
                    n_stop += 1
                    continue
            # 2) fallback to nearest track-node
            if graph_coords:
                nid, dist = snap_to_graph(base[0], base[1], graph_coords)
                if nid is not None and dist <= max_dist_m:
                    nlon, nlat = graph_coords[nid]
                    line_out[n] = [round(nlat, 6), round(nlon, 6)]
                    n_track += 1
                    continue
            n_skip += 1
    print(f"  per-line station coords: {n_stop} via stop_position,"
          f" {n_track} via track-node fallback, {n_skip} skipped")
    return out


def build_link_shapes(graphs, branches, numbat_stations, naptan_coords,
                      per_line_station_coords=None,
                      simplify_tol_m=SIMPLIFY_TOL_M, max_snap_m=400,
                      k_snap=6, max_detour_ratio=2.5):
    """For every (line, naptan_a, naptan_b) consecutive pair on a branch, find
    the shortest OSM-track path. Returns { 'LINE:NA:NB': [[lon, lat], ...] }.

    Endpoint selection: pre-pick ONE canonical snap node per (line, naptan) —
    the graph node nearest to the line's canonical station coord. Dijkstra
    runs between these canonical nodes, so adjacent shapes A→B and B→C share
    the exact same B endpoint. No pin-to-station-coord step is needed, which
    avoids the up-down backtrack zigzag that pinning produced when the best
    Dijkstra snap landed far from the platform coord.

    If canonical-to-canonical Dijkstra fails or detours too far, falls back
    to best-of-`k_snap`×`k_snap` candidate pairs for robustness. If even that
    fails, emits a 2-point straight-chord shape between station coords so the
    rendered line stays continuous instead of leaving a missing arc."""
    print("Building per-link shapes (Dijkstra on OSM graphs)...")
    canon_cache = {}  # (line, naptan) -> node_id or None
    snap_cache = {}   # (line, naptan) -> [(node_id, dist_m), ...]

    def coord_for(naptan, mnlc):
        # Prefer per-naptan TfL coord; fall back to mnlc-averaged.
        c = naptan_coords.get(naptan)
        if c:
            return c[0], c[1]
        s = numbat_stations.get(mnlc)
        return (s["lat"], s["lon"]) if s else None

    def station_ref(line, naptan, mnlc):
        # (lat, lon) reference for snapping/pinning. Prefer the line's
        # per_line_station_coords (OSM stop_position-derived); fall back
        # to the TfL naptan coord.
        if per_line_station_coords:
            c = per_line_station_coords.get(line, {}).get(naptan)
            if c:
                return (c[0], c[1])
        return coord_for(naptan, mnlc)

    def canonical(line, naptan, mnlc):
        # The single best snap node for this (line, naptan): the graph node
        # nearest to the canonical station coord. Cached per pair.
        key = (line, naptan)
        if key in canon_cache:
            return canon_cache[key]
        ref = station_ref(line, naptan, mnlc)
        g = graphs.get(line)
        if not ref or not g:
            canon_cache[key] = None
            return None
        nid, dist = snap_to_graph(ref[0], ref[1], g["coords"])
        if nid is None or dist > max_snap_m:
            canon_cache[key] = None
            return None
        canon_cache[key] = nid
        return nid

    def snaps(line, naptan, mnlc):
        # k nearest snap candidates — used only as a fallback when canonical
        # Dijkstra can't find a path.
        key = (line, naptan)
        if key in snap_cache:
            return snap_cache[key]
        ref = station_ref(line, naptan, mnlc)
        g = graphs.get(line)
        if not ref or not g:
            snap_cache[key] = []
            return []
        cands = k_nearest_in_graph(ref[0], ref[1], g["coords"],
                                   k=k_snap, max_dist_m=max_snap_m)
        snap_cache[key] = cands
        return cands

    link_shapes = {}
    hits = hits_canonical = hits_fallback = 0
    miss_no_snap = miss_no_path = miss_detour = chord_fallbacks = 0
    for br in branches:
        line = br["line"]
        if line not in graphs:
            continue
        adj = graphs[line]["adj"]
        coords = graphs[line]["coords"]
        for i in range(len(br["naptan_ids"]) - 1):
            n_a, n_b = br["naptan_ids"][i], br["naptan_ids"][i + 1]
            key = f"{line}:{n_a}:{n_b}"
            if key in link_shapes:
                continue
            ra = station_ref(line, n_a, br["mnlcs"][i])
            rb = station_ref(line, n_b, br["mnlcs"][i + 1])
            sa = [round(ra[1], 5), round(ra[0], 5)] if ra else None
            sb = [round(rb[1], 5), round(rb[0], 5)] if rb else None

            def emit_chord_fallback():
                nonlocal chord_fallbacks
                if sa and sb:
                    link_shapes[key] = [sa, sb]
                    chord_fallbacks += 1

            direct = (haversine_m(ra[0], ra[1], rb[0], rb[1])
                      if ra and rb else 0.0)
            best_poly = None
            best_len = float("inf")
            from_canonical = False

            # 1) Try the canonical-to-canonical path first. Guarantees adjacent
            # shapes share endpoints when this succeeds.
            ca_nid = canonical(line, n_a, br["mnlcs"][i])
            cb_nid = canonical(line, n_b, br["mnlcs"][i + 1])
            if ca_nid is not None and cb_nid is not None:
                path = dijkstra_path(adj, ca_nid, cb_nid)
                if path and len(path) >= 2:
                    poly = [coords[n] for n in path]
                    L = _polyline_len_m(poly)
                    if not (direct > 50 and L > direct * max_detour_ratio):
                        best_poly = poly
                        best_len = L
                        from_canonical = True

            # 2) Fallback to best-of-k×k if canonical didn't yield a good path.
            if best_poly is None:
                cands_a = snaps(line, n_a, br["mnlcs"][i])
                cands_b = snaps(line, n_b, br["mnlcs"][i + 1])
                if not cands_a or not cands_b:
                    miss_no_snap += 1
                    emit_chord_fallback()
                    continue
                for nid_a, _ in cands_a:
                    for nid_b, _ in cands_b:
                        path = dijkstra_path(adj, nid_a, nid_b)
                        if not path or len(path) < 2:
                            continue
                        poly = [coords[n] for n in path]
                        L = _polyline_len_m(poly)
                        if L < best_len:
                            best_len = L
                            best_poly = poly
                if best_poly is None:
                    miss_no_path += 1
                    emit_chord_fallback()
                    continue
                if direct > 50 and best_len > direct * max_detour_ratio:
                    miss_detour += 1
                    emit_chord_fallback()
                    continue

            poly = best_poly
            if simplify_tol_m > 0 and len(poly) > 2:
                poly = simplify_dp(poly, simplify_tol_m)
            poly = [[round(p[0], 5), round(p[1], 5)] for p in poly]
            link_shapes[key] = poly
            hits += 1
            if from_canonical:
                hits_canonical += 1
            else:
                hits_fallback += 1
    print(f"  built {hits} link shapes"
          f" ({hits_canonical} canonical, {hits_fallback} best-of-k fallback;"
          f" missed: {miss_no_snap} no-snap, {miss_no_path} no-path,"
          f" {miss_detour} >{max_detour_ratio}x detour;"
          f" {chord_fallbacks} chord fallbacks emitted)")
    return link_shapes


def build_branches_from_seq(line_code, tfl_id, naptan_to_mnlc, naptan_coords):
    """Read seq_{tfl_id}.json's stopPointSequences -> branch dicts."""
    path = f"{CACHE_DIR}/seq_{tfl_id}.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        seq = json.load(f)
    out = []
    branch_id_counter = 0
    for sps in seq.get("stopPointSequences", []) or []:
        direction = sps.get("direction") or "I"
        # one-letter dir code consistent with NUMBAT branches
        d1 = direction[0].upper()
        sp = sps.get("stopPoint", []) or []
        if len(sp) < 2:
            continue
        naptans = [s.get("id") or s.get("stationId") for s in sp]
        naptans = [n for n in naptans if n]
        mnlcs = []
        for n in naptans:
            m = naptan_to_mnlc.get(n)
            if m is not None:
                mnlcs.append(m)
            else:
                mnlcs.append(None)  # gap; we'll filter below
        # Filter to consecutive non-None runs (drop unmapped stops)
        run_naptans = []
        run_mnlcs = []
        for n, m in zip(naptans, mnlcs):
            if m is None:
                continue
            run_naptans.append(n)
            run_mnlcs.append(m)
        if len(run_mnlcs) < 2:
            continue
        # Compute inter-station distances using stop coords
        dists = []
        for i in range(len(run_naptans) - 1):
            a = naptan_coords.get(run_naptans[i])
            b = naptan_coords.get(run_naptans[i + 1])
            if a and b:
                dists.append(round(haversine_m(a[0], a[1], b[0], b[1]), 1))
            else:
                dists.append(0.0)
        out.append({
            "line": line_code,
            "dir": d1,
            "branch_id": int(sps.get("branchId", branch_id_counter)),
            "branch_name": sps.get("name") or f"branch {branch_id_counter}",
            "mode": "u",  # cosmetic; speed no longer used
            "mnlcs": run_mnlcs,
            "naptan_ids": run_naptans,
            "distances_m": dists,
        })
        branch_id_counter += 1
    return out


def main():
    numbat_stations = load_numbat_stations()
    tfl_lines = load_tfl_lines()
    print(f"  {len(tfl_lines)} TfL lines from cache")

    # Collect all TfL stops (deduped) so we can do one big naptan->mnlc match
    all_stops = []
    seen = set()
    for ln in tfl_lines:
        for s in load_tfl_stops_per_line(ln["id"]):
            if s["naptan_id"] in seen:
                continue
            seen.add(s["naptan_id"])
            all_stops.append(s)
    print(f"  {len(all_stops)} unique TfL rail stops")

    naptan_coords = {s["naptan_id"]: (s["lat"], s["lon"]) for s in all_stops if s["lat"] is not None}

    pair_map, naptan_to_mnlc = build_per_line_mnlc_map(numbat_stations, tfl_lines)

    # NUMBAT publishes station coords that are sometimes 100-300m off (London
    # Bridge is 220m wrong) — they put the marker on the surface building
    # rather than the platform centre. Override with TfL Unified API coords,
    # which sit much closer to where OSM has the tracks.
    apply_tfl_coord_overrides(numbat_stations, all_stops, naptan_to_mnlc)

    # Build per-NUMBAT-line branches by walking each TfL line's stopPointSequences.
    # Use pair_map[(line_id, naptan)] when available so Overground branches see
    # the LO mnlc instead of the colocated LU one.
    print("Building branches from TfL Route/Sequence...")
    tfl_to_numbat = {v: k for k, v in NUMBAT_TO_TFL.items()}
    branches = []
    for ln in tfl_lines:
        tfl_id = ln["id"]
        line_code = tfl_to_numbat.get(tfl_id)
        if not line_code:
            continue
        per_line_naptan = {nid: pair_map[(tfl_id, nid)]
                           for (lid, nid) in pair_map.keys()
                           if lid == tfl_id}
        # Fall back to flat naptan_to_mnlc for stops that didn't match via pair_map
        # (e.g. interchange-only naptans tagged on this line but not actually served)
        merged = {**naptan_to_mnlc, **per_line_naptan}
        bs = build_branches_from_seq(line_code, tfl_id, merged, naptan_coords)
        branches.extend(bs)
    print(f"  {len(branches)} branches built")

    # Build line geometries from OSM track centrelines (real curvature).
    lines_out = build_lines_from_osm(OSM_JSON)

    # Per-link sub-polylines for path-following train animation. The renderer
    # expands each station-to-station leg using these. Use a tighter DP
    # tolerance than the background lines (2m instead of SIMPLIFY_TOL_M=10m):
    # per-link polylines are short and need to track real track curvature so
    # trains don't cut visible chords across slightly-bent track (Elm Park ↔
    # Dagenham East, Harold Wood ↔ Gidea Park, etc.). At the default 10m
    # tolerance these collapse to 2pt; 2m gives ~8pt for typical 2km segments.
    line_graphs = build_per_line_graphs(OSM_JSON)
    # Build per_line_station_coords FIRST so link_shapes can pin both endpoints
    # to the canonical per-station coord. Adjacent link_shapes A→B and B→C
    # then share the exact same B coord and meet cleanly at the station,
    # closing the visual gap caused by Dijkstra picking different snap
    # candidates between adjacent hops.
    per_line_station_coords = build_per_line_station_coords(
        line_graphs, branches, naptan_coords, OSM_JSON)
    link_shapes = build_link_shapes(line_graphs, branches, numbat_stations,
                                    naptan_coords,
                                    per_line_station_coords=per_line_station_coords,
                                    simplify_tol_m=2.0)

    # Stations: only those that appear on at least one branch
    used_mnlcs = set()
    for br in branches:
        used_mnlcs.update(br["mnlcs"])
    print(f"  {len(used_mnlcs)} stations used by branches")

    # Map mnlc -> naptan_id (first one found, used by build.py for fallback)
    mnlc_to_naptans = defaultdict(set)
    for naptan, mnlc in naptan_to_mnlc.items():
        mnlc_to_naptans[mnlc].add(naptan)

    # Per-NUMBAT-line: mnlc -> set of naptan ids serving that mnlc on that line.
    # Built from pair_map so an Overground OD looks up the LO-side naptan first
    # rather than the LU-side one. Keyed by NUMBAT line code.
    mnlc_to_naptans_per_line = defaultdict(lambda: defaultdict(set))
    for (tfl_id, nid), mnlc in pair_map.items():
        line_code = tfl_to_numbat.get(tfl_id)
        if line_code:
            mnlc_to_naptans_per_line[line_code][mnlc].add(nid)

    # Per-station: which lines (NUMBAT codes) serve it
    station_routes = defaultdict(set)
    for br in branches:
        for m in br["mnlcs"]:
            station_routes[m].add(br["line"])

    stations_out = {}
    for mnlc in used_mnlcs:
        s = numbat_stations.get(mnlc)
        if not s:
            continue
        naptans = sorted(mnlc_to_naptans.get(mnlc, []))
        stations_out[str(mnlc)] = {
            "name": s["name"],
            "asc": s["asc"],
            "lat": round(s["lat"], 6),
            "lon": round(s["lon"], 6),
            "naptan_ids": naptans,
            "lines": sorted(station_routes.get(mnlc, [])),
        }

    out = {
        "stations": stations_out,
        "branches": [
            {
                "line": b["line"],
                "dir": b["dir"],
                "branch_id": b["branch_id"],
                "branch_name": b["branch_name"],
                "mode": b["mode"],
                "mnlcs": b["mnlcs"],
                "naptan_ids": b["naptan_ids"],
                "distances_m": b["distances_m"],
            }
            for b in branches
        ],
        "lines": lines_out,
        "link_shapes": link_shapes,
        "line_colors": LINE_COLORS,
        # Per-naptan TfL coords. Trains use these so DLR/LU/NR platforms at
        # an interchange (Bank/Monument 390m apart, Seven Sisters 220m, etc.)
        # render at distinct points instead of one averaged complex centre.
        "naptan_coords": {
            n: [round(c[0], 6), round(c[1], 6)]
            for n, c in naptan_coords.items()
            if n in naptan_to_mnlc and naptan_to_mnlc[n] in used_mnlcs
        },
        # Per-line snap of each naptan to its line's OSM track. Distinguishes
        # platforms at single-naptan complexes (Embankment etc.) where
        # naptan_coords would collapse all lines to one point.
        "per_line_station_coords": per_line_station_coords,
        "naptan_to_mnlc": {n: m for n, m in naptan_to_mnlc.items() if m in used_mnlcs},
        "mnlc_to_naptans": {str(m): sorted(v) for m, v in mnlc_to_naptans.items() if m in used_mnlcs},
        "mnlc_to_naptans_per_line": {
            line: {str(m): sorted(v) for m, v in by_mnlc.items() if m in used_mnlcs}
            for line, by_mnlc in mnlc_to_naptans_per_line.items()
        },
        "numbat_line_to_tfl": NUMBAT_TO_TFL,
    }

    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
