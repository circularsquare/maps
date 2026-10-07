"""Ireland: RINF has every Iarnród Éireann line, but one id per SECTION (274 ids, "SOL1039"
Greystones - Kilcoole), so the lines are put together here, as Denmark's are. ie_sources.md has
the sources, the checks and what is still off.

    python rinf.py --fetch ie                  # RINF and Wikidata, a few seconds
    python extract.py --region ie --pbf data/raw/ireland-and-northern-ireland-latest.osm.pbf
    python -m rinf_countries.ie --clip         # after every extract: Northern Ireland out
    python build_model.py --region ie --register rinf:data/raw/rinf/ie

THE IDS.  "SOL1001" ... "SOL1274", numbered roughly along each route, with no line number in
them at all. `fix` files each section under a named line by its number (LINE_OF), the names
being en.wikipedia's article titles for Iarnród Éireann's routes (ie_sources.md has the
list and the few that are ours). Freight-only sections (Tara Mines, Foynes, Belview, Dublin
Port and its yard links) and the stub of the closed Waterford - Rosslare line are left out.

POINTS.  Every one of RINF's 267 Irish points is typed 30 (passenger terminal), the TD
(train describer) boundaries and junctions too. `fix` retypes a point that is a TD, a
junction, the border or a freight terminal as a junction (80), and gives the stations whose
RINF name OSM does not share a second name to match by ("Connolly Station | Dublin Connolly").
uopids are "IE+OP42"; the "+" is dropped, so section ends are "eIEOP42".

RINF'S SHAPE, PUT RIGHT.  Clonsilla, Manulla Junction and Athenry each have a station and a
junction point at one coordinate, joined by a 0.15-0.63 km section, so the station hung off
the line on a stub of no length: the junction point is merged into the station (Athenry: the
Western Railway Corridor's end, "Athenry Junction", 480 m west of the station, met the Galway
line nowhere, and "Athenry West Junction" sat on the station; the latter is moved to the
former and the two are one). Park West and Cherry Orchard are one station, two RINF points 47
m apart, with the TD points between them that lie 1.5 km nearer Dublin: the section runs
Islandbridge Junction - TD 83 - TD 82 - Park West and Cherry Orchard instead.

LENGTHS.  Most RINF sections are right to within a few hundred metres, but some 20 are not
(Mallow - Killarney Junction is 59.1 km for 1.1 km of track, Tralee Junction - Killarney
27.1 for 0.4; ie_sources.md lists them). They are left to rinf.py's "traces agree" rule and
logged as "length off", and no km_official is shipped (`no_chain`): Iarnród Éireann's
published line lengths in check_model.REGISTER are the check. Three RINF lengths are too short
for the trace to reach the far end at all; `LENGTH` replaces them.

ENNIS JUNCTION is placed 7 km west of Limerick, nowhere near track; it is 0.45 km from Foynes
Junction in RINF and is merged into it.

THE BORDER.  RINF's "Border" point (IE+OP42) is moved onto the track where OSM's boundary of
the Republic crosses it (ways 31695909 / 395764463 end there), and borders.EXTRA carries the
same id and point for gb.
"""
import csv
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_IE = ROOT / "data" / "raw" / "ie"

BEL = "Great Northern Railway Main Line"     # gb's name for its half (OSM's on both sides)
HOWTH = "Howth branch"
ROSS = "Dublin–Rosslare line"
SLIGO = "Dublin–Sligo line"
NAVAN = "Dublin–Navan line"
PPT = "Phoenix Park Tunnel line"
CORK = "Dublin–Cork line"
GALWAY = "Dublin–Galway line"
WESTPORT = "Dublin–Westport line"
BALLINA = "Ballina branch"
WATERFORD = "Dublin–Waterford line"
LIMROSS = "Limerick–Rosslare line"
LIMBAL = "Limerick–Ballybrophy line"
WRC = "Western Railway Corridor"
TRALEE = "Mallow–Tralee line"
COBH = "Cork–Cobh line"
MIDLETON = "Glounthaune–Midleton line"
OUT = None                                   # freight only, or the closed Rosslare stub


def _ranges(*spans):
    out = []
    for s in spans:
        a, _, b = str(s).partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# RINF section number -> line. Every one of SOL1001 - SOL1274 is listed.
_TABLE = [
    (BEL, _ranges("1001-1017", 1239, "1244-1245", 1252, 1274)),
    (HOWTH, _ranges("1240-1242")),
    (ROSS, _ranges("1018-1047")),
    (SLIGO, _ranges(1048, "1053-1086", 1246, 1251, "1257-1259", "1267-1273")),
    (NAVAN, _ranges("1049-1052")),
    (PPT, _ranges(1247, "1253-1256", "1260-1261")),
    (CORK, _ranges("1087-1098", "1120-1123", "1155-1162", "1165-1170", "1176-1184")),
    (LIMBAL, _ranges("1163-1164", "1220-1225")),
    (GALWAY, _ranges("1124-1130", "1142-1154")),
    (WESTPORT, _ranges("1131-1137", "1140-1141")),
    (BALLINA, _ranges("1138-1139")),
    (WATERFORD, _ranges("1099-1119", 1199)),
    (LIMROSS, _ranges("1171-1175", "1194-1197", "1226-1227", 1229, "1237-1238")),
    (WRC, _ranges(1228, "1230-1236")),
    (TRALEE, _ranges("1201-1219")),
    (COBH, _ranges("1185-1191")),
    (MIDLETON, _ranges("1192-1193")),
    (OUT, _ranges(1198, 1200, 1243, "1248-1250", "1262-1266")),
]
LINE_OF = {}
for _name, _nums in _TABLE:
    for _n in _nums:
        assert f"SOL{_n}" not in LINE_OF, _n
        LINE_OF[f"SOL{_n}"] = _name
assert len(LINE_OF) == 274
NAMES = {n for n, _ in _TABLE if n}

# Points that are not stops, whatever RINF types them.
NOT_STOP = re.compile(r"^TD \d+$|Junction$|Junction \(|^Border$|Viaduct")
FREIGHT = {"Dublin Port", "Port of Foynes", "Bellview Port Station", "Tara Mines Station",
           "East Wall Yard Station"}
# RINF name -> what it is called (OSM's and the timetable's name), offered as a second name to
# match OSM's stations by, and a typo or two.
NAME = {
    "Connolly Station": "Dublin Connolly", "Heuston": "Dublin Heuston",
    "Pearse Station": "Dublin Pearse", "Drohgeda McBride": "Drogheda MacBride",
    "Kilkenny Station": "Kilkenny MacDonagh", "Wexford Station": "Wexford O'Hanrahan",
    "Waterford Station": "Waterford Plunkett", "Tralee Station": "Tralee Casement",
    "Sligo MacDiarmada Station": "Sligo Mac Diarmada",
    "Hazelhatch Station": "Hazelhatch and Celbridge", "Sallins Station": "Sallins and Naas",
    "Rush and Lusk Station": "Rush & Lusk", "Little Island Station": "Littleisland",
    "Park West Station": "Park West and Cherry Orchard",
    "Clondalkin Fonthill Station": "Clondalkin & Fonthill",
    "Salthill & Monkstown Station": "Salthill and Monkstown",
    "Sandycove & Glasthule Station": "Sandycove and Glasthule",
    "Dun Laoghaire Mallin Station": "Dún Laoghaire Mallin",
    "Howth Junction & Donaghmede Station": "Howth Junction and Donaghmede",
    "Leixlip (Confey) Station": "Leixlip Confey",
    "Leixlip (Louisa Bridge) Station": "Leixlip Louisa Bridge",
    "Muine Bheag (Bagenalstown) Station": "Muine Bheag",
    "Carrick on Shannon Station": "Carrick-on-Shannon",
    "Carrick on Suir Station": "Carrick-on-Suir",
    "Foyens Junction": "Foynes Junction",
    "Dunkitt Junction (No turnout here)": "Dunkitt Junction",
    "Ossary Road Junction": "Ossory Road Junction",
}
# Limerick Junction has no station point in RINF, only five junctions around it; the
# "Station Junction" at the platforms' south end (121 m from OSM's station) stands for it.
# Sections there are too short for `osm_stops` to cut one at the station (400 m from an end).
STOP = {"IE+OP198": "Limerick Junction"}
# (point kept, point merged into it): one place, two RINF points (docstring).
MERGE = [("IE+OP74", "IE+OP75"),      # Clonsilla Station <- Clonsilla Junction
         ("IE+OP160", "IE+OP161"),    # Manulla Junction Station <- Manulla Junction Junction
         ("IE+OP114", "IE+OP117"),    # Park West <- Cherry Orchard (one station)
         ("IE+OP174", "IE+OP262"),    # Athenry West Junction <- Athenry Junction (moved there)
         ("IE+OP250", "IE+OP249")]    # Foynes Junction <- Ennis Junction (placed 7 km off)
MOVE_TO = {"IE+OP174": "IE+OP262"}    # take the other point's coordinate first
# Sections that are one point to itself once merged, or that the new shape replaces.
DROP = {"SOL1048", "SOL1137"}
# Park West: Islandbridge Junction - TD 83 - TD 82 - Park West and Cherry Orchard.
REEND = {"SOL1087": ("IE+OP23", "IE+OP115"),
         "SOL1088": ("IE+OP115", "IE+OP116"),
         "SOL1089": ("IE+OP116", "IE+OP114"),
         "SOL1090": None}
# The border, where OSM's boundary of the Republic crosses the two tracks (ways 31695909 and
# 395764463 end at -6.379110, 54.069069 and -6.379074, 54.069092): their midpoint, 20 m from
# RINF's own point. borders.EXTRA has the same id ("e" + the uopid as fix leaves it) and point.
BORDER_UOPID = "IE+OP42"
BORDER_AT = (-6.379092, 54.069080)
# Section lengths (km) put in for RINF's where RINF's is far too short for the trace to reach
# the other end at all. No km_official is shipped (`no_chain`), so these only bound the trace.
LENGTH = {
    # Rosslare Strand - Rosslare Europort: RINF 0.358 for 4.2 km as the crow flies; the
    # Network Statement's 3 1/4 miles (en.wikipedia, Dublin-Rosslare line, its mileage note)
    "SOL1047": 5.23,
    # TD 66 (Rathmore) - TD 67 (Killarney): RINF 0.81 for 17.2 km as the crow flies; RINF's
    # 27.1 km for Tralee Junction - Killarney (0.4 km of track) is this stretch's, misfiled
    "SOL1210": 19.0,
    # Ennis Junction (merged into Foynes Junction) - Sixmilebridge: RINF 14.9 plus 0.45 for
    # 13.3 km as the crow flies on a line that swings west by Cratloe
    "SOL1230": 19.5,
}


def ie_fix(secs, points):
    out = []
    by_uop = {p.get("uopid"): op for op, p in points.items()}
    # --- the border point on the track
    bp = points[by_uop[BORDER_UOPID]]
    bp["lon"], bp["lat"] = BORDER_AT
    out.append(f"{BORDER_UOPID} moved onto the track at the border {BORDER_AT}")
    # --- one point per place
    for keep, gone in MOVE_TO.items():
        points[by_uop[keep]]["lon"] = points[by_uop[gone]]["lon"]
        points[by_uop[keep]]["lat"] = points[by_uop[gone]]["lat"]
    ren = {by_uop[g]: by_uop[k] for k, g in MERGE}
    kept = []
    for s in secs:
        base = s["line"]
        if base in DROP:
            continue
        if base in REEND:
            if REEND[base] is None:
                continue
            a, b = REEND[base]
            s["a"], s["b"] = by_uop[a], by_uop[b]
        s["a"], s["b"] = ren.get(s["a"], s["a"]), ren.get(s["b"], s["b"])
        if s["a"] == s["b"]:
            continue
        if base in LENGTH:
            out.append(f"{base} {s['label'][15:70]}: RINF {s['km']} km -> {LENGTH[base]}")
            s["km"] = LENGTH[base]
        kept.append(s)
    out.append(f"{len(secs) - len(kept)} sections dropped by merging points "
               f"({', '.join(sorted(points[g]['name'] for g in ren))})")
    secs[:] = kept
    # --- what each point is
    n_j = n_name = 0
    for op, p in points.items():
        nm = p.get("name") or ""
        if p.get("uopid") in STOP:
            p["type"] = "30"
            p["name"] = f"{nm} | {STOP[p['uopid']]}"
            n_name += 1
        elif NOT_STOP.search(nm) or nm in FREIGHT:
            p["type"] = "80"
            n_j += 1
            if nm in NAME:                     # a junction's name is shown: put it right
                p["name"] = NAME[nm]
        elif nm in NAME:
            p["name"] = f"{nm} | {NAME[nm]}"
            n_name += 1
        if p.get("uopid"):
            p["uopid"] = p["uopid"].replace("+", "")
    out.append(f"{n_j} points typed junction (TD boundaries, junctions, the border, freight "
               f"terminals); {n_name} given a second name to match OSM by")
    # --- lines
    n_left, left = 0, defaultdict(float)
    for s in secs:
        g = LINE_OF.get(s["line"])
        if g is None:
            n_left += 1
            left[s["label"].replace("Section of Line ", "")] += s["km"] or 0.0
            g = f"(left out {s['line']})"
        s["line"] = s["base"] = g
    out.append(f"{len(secs) - n_left} sections filed under {len(NAMES)} named lines; "
               f"{n_left} left out (freight): " + ", ".join(
                   f"{k} {v:.2f} km" for k, v in sorted(left.items())))
    # pieces of one name that do not touch, numbered as rinf.py numbers an id's pieces
    by = defaultdict(list)
    for s in secs:
        by[s["base"]].append(s)
    for name, ss in by.items():
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for s in ss:
            parent[find(s["a"])] = find(s["b"])
        comps = defaultdict(list)
        for s in ss:
            comps[find(s["a"])].append(s)
        if len(comps) < 2:
            continue
        order = sorted(comps.values(), key=lambda c: -sum(s["km"] or 0 for s in c))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = name if k == 0 else f"{name}#{k + 1}"
        out.append(f"{name} is {len(comps)} pieces: " + " | ".join(
            f"{sum(s['km'] or 0 for s in c):.1f} km" for c in order))
    return out


def ie_id_name(lid, _uop=None):
    n = lid.split("#")[0]
    return n if n in NAMES else None


FEED_DIR = ROOT / "data" / "raw" / "gtfs" / "ie"
_FEED_KEYS = None


def _station_key(name):
    n = re.sub(r"\s*\(.*?\)", "", (name or "")).strip()
    n = re.sub(r"\s+(?:stn|station)\.?$", "", n, flags=re.I)
    return re.sub(r"[^0-9a-z]", "", n.casefold().replace("&", "and"))


def ie_stop_extra(name):
    """An OSM station no route lists as a stop, which Iarnród Éireann's timetable calls at.
    No feed on disk: nothing."""
    global _FEED_KEYS
    if _FEED_KEYS is None:
        _FEED_KEYS = set()
        for z in sorted(FEED_DIR.glob("*.zip")):
            with zipfile.ZipFile(z) as zf, zf.open("stops.txt") as f:
                for r in csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")):
                    _FEED_KEYS.add(_station_key(r.get("stop_name")))
    return _station_key(name) in _FEED_KEYS


COUNTRY = {
    "iso3": "IRL", "wikidata": "Q27", "langs": ["en", "ga"],
    "fix": ie_fix, "id_name": ie_id_name,
    "skip_line": lambda lid: lid.startswith("(left out"),
    "osm_rel": lambda _tags: None,
    "osm_stops": True,
    "osm_stop_extra": ie_stop_extra,
    "cut_at_junctions": True,
    "no_chain": True,
    # every section carries its own manager code (1001_IM ... 1274_IM); all are IÉ's
    "im_of": lambda _s: "Iarnród Éireann",
}


def clip(log=print):
    """Rewrite data/proc/ie without Northern Ireland, which is built in gb: Geofabrik's
    ireland-and-northern-ireland extract holds the whole island. OSM's own boundary of the
    Republic (relation 62273, data/raw/ie/ie_boundary.geojson from polygons.openstreetmap.fr)
    decides: a way stays if any of its nodes is inside (so the one way over the border near
    Killeen stays whole, and the border point lies on it), a stop if it is inside, a route
    relation if a member stayed, an infrastructure relation if one of its track ways stayed.
    Idempotent; run after every extract."""
    import pickle
    import numpy as np
    import shapely
    from shapely.geometry import shape
    d = ROOT / "data" / "proc" / "ie"
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(d / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    ie = shape(json.loads((RAW_IE / "ie_boundary.geojson").read_text(encoding="utf-8")))
    shapely.prepare(ie)
    inside = set(cid[shapely.contains_xy(ie, cx / 1e7, cy / 1e7)].tolist())
    keep_w, cut, cross = {}, Counter(), []
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes]
        k = sum(n in inside for n in ns)
        if k:
            keep_w[wid] = (tags, nodes)
            if k < len(ns):
                cross.append((wid, tags.get("railway"), tags.get("name") or ""))
        else:
            cut[(tags.get("railway"), tags.get("name") or "(unnamed)")] += 1
    keep_s = {k: v for k, v in stops.items() if shapely.contains_xy(ie, v[1], v[2])}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    keep_i = {k: v for k, v in infra.items()
              if any(t == "w" and r in keep_w for t, r, _ in v[1])
              or not any(t == "w" and r in ways for t, r, _ in v[1])}
    for (rw, name), v in sorted(cut.items(), key=lambda kv: -kv[1])[:40]:
        log(f"  cut {v:4d}  {rw} {name}")
    for wid, rw, name in cross:
        log(f"  over the border, kept whole: way {wid} {rw} {name}")
    log(f"IE clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--clip" in sys.argv:
        clip()
    else:
        print(__doc__)
