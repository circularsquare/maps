"""Norway: register lines from Bane NOR's Banenettverk, stops and running track from Entur.

    python no_register.py --fetch     # data/raw/no: Banenettverk GML (11 MB zip) + Entur's rail lines
    python no_register.py --dry       # the register half alone, no OSM: lines, km, checks
    python build_model.py --region no --register no_register:data/raw/no

no_sources.md is the record: numbers, checks, what is off.

SOURCES.
- Banenettverk (Bane NOR SF via Geonorge, NLOD 1.0; "Jernbane - Banenettverk"): the reference
  line of every banestrekning, as `Banelenke` centre lines carrying the line's name (banenavn,
  "Dovrebanen"), short code (banekortnavn, "DOVB"), status (banestatus: I in use, N closed, M
  museum, P planned, F removed), purpose (baneformål: B railway, P passenger, G freight siding,
  M museum railway) and Bane NOR's chainage at both ends (startposisjon/sluttposisjon, km);
  and `Stasjonsnode`s (station name, type S station / I halt, validity). ONE CENTRE LINE PER
  LINE: Oslo - Lillestrøm is two lines (Hovedbanen, Gardermobanen), each one centre line, not
  four tracks. The national GML (EPSG:4258) lists every Banelenke twice under two ids, same
  geometry; `read` keeps one.
- Entur's journey planner (api.entur.io/journey-planner/v3, open, ET-Client-Name header;
  NLOD): every rail line in Norway's national timetable (Vy, SJ Norge/SJ Nord, Go-Ahead,
  Flytoget, Flåmsbanen, the cross-border Swedish trains) with each journey pattern's stop
  places in order and their coordinates. `entur_rail_lines.json`.

WHY NOT RINF (rinf_countries/no.py exists only so `rinf.py --fetch no` works): Bane NOR's RINF
has no line ids (each section's nationalLine is labelled "B05-Nordlandsbanen" instead), and its
375 sections are 3,234 km against Banenettverk's ~4,000 km in use: Støren is in it with no
section to it (Dovrebanen and Rørosbanen end at Soknedal, Hovin and Singsås), Drammen -
Galleberg and Barkåker - Sem (Vestfoldbanen), Magnor - border, Kopperå - border, Bjørnfjell -
border and half of Ofotbanen are missing. no_sources.md has the per-line comparison.

THE LINE UNIT is Bane NOR's banestrekning (`group`): Dovrebanen, Bergensbanen, Nordlandsbanen...
Short connector strekninger whose code joins two lines' codes are folded in (`fold`): "DOVB_RAUB
Dovrebanen spor mot Raumabanen" and "HVDB_GARB Oslo S mot Gardermobanen" into the second line,
which starts there; "OFTB_KAT Ofotbanen Katterat" into the first. Track in use (status I) for
railway or passenger traffic (formål B, P) is read; freight sidings (G), museum railways (M),
closed (N), planned (P) and removed (F) track is not (`--dry` lists what that leaves out).
`snap` joins links whose ends miss each other by up to a metre (or bridge up to 100 m), and
`fill_gaps` joins a line's pieces over other lines' links where Bane NOR files a stretch of
it under another code (Drammenbanen through Asker).

STOPS are Entur's rail stop places that some journey pattern calls at, within ATTACH_M of the
network, put on the lines their trains' paths run over near them (`place_stops`; halts, Bane
NOR's type I, only on their own line: `halts_on_own_line`). A stop no train calls at
(Brennhaug, Agle, Solørbanen's stations) is no stop, as Bane NOR's own station list also holds
crossing loops. Names are Entur's without " stasjon" ("Skien", "Oslo S", "Oslo lufthavn").

SECTIONS are n02.py's: the absorbing search between neighbouring stops along a line's own
centre line, with two kinds of extra section end: border points (the four RINF points to
Sweden, eEU00234-37, which borders.py and Sweden's register already use) and junctions where
two lines trains run over meet away from a stop ("j..." ids, `junction`; a line that ends
within STOP_JUNCTION_M of a stop on the line it leaves starts at that stop instead:
Arendalsbanen at Nelaug). Sections lying along others of their own line are dropped
(`redundant`), and junctions only one line still reaches are fused away (`tidy_junctions`).

WHICH SECTIONS ARE RUN OVER (`served`): every pair of consecutive calls in Entur's journey
patterns credits the shortest track between the two stops over the whole network (no longer
than DETOUR x the crow-fly distance + DETOUR_KM); a pair whose second stop lies abroad credits
the section from the first stop to the border point between them. A section is run over when
SERVED_SHARE of it lies on credited track. Sections no train runs over are left out and
listed: junction-ended ones (curves, freight approaches) and stop-to-stop ones too (Roa -
Hønefoss, freight only), since Entur's timetable is every passenger train in the country.
build_model then drops junction-ended sections OSM's routes do not run over, as for every
register; a section Entur's trains use but OSM has no route on is lost there (no_sources.md).

GEOMETRY is Banenettverk's own centre line, as n02.py uses N02's: build_model ties the register
lines to OSM ways afterwards (register_way_lines), which is where OSM's track takes over.

The `path` argument is data/raw/no; the OSM half is read from data/proc/no (extract.py), and
only by build_model, not here.
"""
import hashlib
import heapq
import json
import math
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import n02

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "no"
GML = "banenettverk/Samferdsel_0000_Norge_4258_Banenettverk_GML.gml"
GML_ZIP_URL = ("https://nedlasting.geonorge.no/geonorge/Samferdsel/Banenettverk/GML/"
               "Samferdsel_0000_Norge_4258_Banenettverk_GML.zip")
ENTUR = "entur_rail_lines.json"
ENTUR_URL = "https://api.entur.io/journey-planner/v3/graphql"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
CLIENT_NAME = "noritetsu-hobby-railmap"

A = "{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Banenettverk/1.0}"
G = "{http://www.opengis.net/gml/3.2}"
INF = float("inf")

STATUS_KEEP = {"I"}             # in use
PURPOSE_KEEP = {"B", "P"}       # railway, passenger (not G freight siding, M museum)

def fold(link):
    """The line a link belongs to. A connector strekning's code joins two lines' codes:
    "<A> spor mot <B>" (Nordlandsbanen spor mot Meråkerbanen, Sørlandsbanen spor mot
    Arendalsbanen) and "Oslo S mot <B>" are the start of line B, which leaves A there (from
    Hell, Nelaug, Oslo S); anything else ("Ofotbanen Katterat", "Gardermobanen Eidsvoll",
    "Østfoldbanen Hafslundsløyfa", "Gardermobanen Eidsvoll spor fra Hovedbanen") is A's."""
    parts = link["code"].split("_")
    if len(parts) == 2 and re.search(r"\bs?mot\b", link["name"]):
        return parts[1]
    return parts[0]

# English names: the en.wikipedia article for each line.
NAME_EN = {
    "Dovrebanen": "Dovre Line", "Bergensbanen": "Bergen Line",
    "Nordlandsbanen": "Nordland Line", "Ofotbanen": "Ofoten Line", "Raumabanen": "Rauma Line",
    "Rørosbanen": "Røros Line", "Sørlandsbanen": "Sørlandet Line",
    "Østfoldbanen vestre linje": "Østfold Line (Western Line)",
    "Østfoldbanen østre linje": "Østfold Line (Eastern Line)",
    "Vestfoldbanen": "Vestfold Line", "Gjøvikbanen": "Gjøvik Line",
    "Kongsvingerbanen": "Kongsvinger Line", "Meråkerbanen": "Meråker Line",
    "Randsfjordbanen": "Randsfjord Line", "Drammenbanen": "Drammen Line",
    "Gardermobanen": "Gardermoen Line", "Hovedbanen": "Trunk Line", "Flåmsbana": "Flåm Line",
    "Spikkestadbanen": "Spikkestad Line", "Arendalsbanen": "Arendal Line",
    "Bratsbergbanen": "Bratsberg Line", "Askerbanen": "Asker Line", "Follobanen": "Follo Line",
    "Roa-Hønefossbanen": "Roa–Hønefoss Line", "Tinnosbanen": "Tinnoset Line",
    "Solørbanen": "Solør Line", "Brevikbanen": "Brevik Line",
    "Stavne-Leangenbanen": "Stavne–Leangen Line", "Numedalsbanen": "Numedal Line",
}

ATTACH_M = 400        # a stop is on a line whose centre line passes this close...
NEAR_EXTRA_M = 100    # ...and no more than this further off than the nearest line's
BORDER_M = 1500       # a border point this close to a line's end is that end
STOP_JUNCTION_M = 1500  # a junction this close to a stop on all its lines is no section end
DETOUR, DETOUR_KM = 1.6, 4.0
SERVED_SHARE = 0.5
ANCHOR_EXTRA_M = 60   # a stop's anchors in the network: vertices this much beyond its nearest

# RINF's border points to Sweden (border_points.json): id, lon, lat.
BORDERS = {
    "eEU00234": (11.66987, 58.93476),   # Kornsjø, Østfoldbanen - Norge/Vänerbanan
    "eEU00235": (12.24341, 59.93187),   # Magnor - Charlottenberg, Kongsvingerbanen - Värmlandsbanan
    "eEU00236": (12.0605, 63.337),      # Kopperå - Storlien, Meråkerbanen - Mittbanan
    "eEU00237": (18.10572, 68.43004),   # Bjørnfjell - Riksgränsen, Ofotbanen - Malmbanan
}


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def xyz(lonlat):
    """Points on a sphere in metres, so a KD-tree measures true distance from 58 to 71 N."""
    a = np.radians(np.asarray(lonlat, dtype=np.float64).reshape(-1, 2))
    c = np.cos(a[:, 1])
    return 6371000.0 * np.column_stack([c * np.cos(a[:, 0]), c * np.sin(a[:, 0]),
                                        np.sin(a[:, 1])])


def line_id(code):
    h = hashlib.blake2b(f"no|{code}".encode("utf-8"), digest_size=5)
    return "n" + h.hexdigest()


# ================================================================ fetching

ENTUR_QUERY = """{ lines(transportModes: [rail]) {
  id publicCode name transportSubmode
  authority { name } operator { name }
  presentation { colour }
  journeyPatterns { id name directionType
    quays { id name stopPlace { id name latitude longitude transportMode } } }
  serviceJourneys { id }
} }"""


def fetch(entur_only=False):
    import zipfile
    RAW.mkdir(parents=True, exist_ok=True)
    if not entur_only:
        z = RAW / "Banenettverk_4258_GML.zip"
        req = urllib.request.Request(GML_ZIP_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as r:
            z.write_bytes(r.read())
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW / "banenettverk")
        print(f"wrote {z} ({z.stat().st_size:,} bytes) and unpacked it")
    req = urllib.request.Request(ENTUR_URL, data=json.dumps({"query": ENTUR_QUERY}).encode(),
                                 headers={"User-Agent": USER_AGENT, "ET-Client-Name": CLIENT_NAME,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    if d.get("errors"):
        raise SystemExit(f"Entur: {d['errors']}")
    (RAW / ENTUR).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {RAW / ENTUR}: {len(d['data']['lines'])} rail lines")


# ================================================================ Banenettverk

def read(path, log):
    """Banelenke and Stasjonsnode records. Each Banelenke is listed twice in the national file
    (two lokalIds, one geometry); one is kept."""
    links, nodes, seen = [], [], set()
    n_dup = 0
    for _ev, el in ET.iterparse(str(path)):
        if el.tag == A + "Banelenke":
            ji = el.find(f"{A}jernbaneinformasjon/{A}Jernbaneinformasjon")
            info = {c.tag[len(A):]: (c.text or "") for c in ji}
            pl = el.find(f".//{G}posList").text.split()
            # GML 3.2 in EPSG:4258 is latitude first
            pts = [(float(pl[i + 1]), float(pl[i])) for i in range(0, len(pl), 2)]
            key = (info.get("banekortnavn"), tuple(round(v, 6) for v in pts[0]),
                   tuple(round(v, 6) for v in pts[-1]), len(pts))
            if key in seen:
                n_dup += 1
            else:
                seen.add(key)
                med = el.find(A + "medium")
                links.append({
                    "code": info.get("banekortnavn", ""), "name": info.get("banenavn", ""),
                    "status": info.get("banestatus", ""), "purpose": info.get("baneformål", ""),
                    "owner": info.get("anleggseier", ""),
                    "medium": med.text if med is not None else "",
                    "km0": float(el.find(A + "startposisjon").text),
                    "km1": float(el.find(A + "sluttposisjon").text),
                    "pts": pts})
            el.clear()
        elif el.tag == A + "Stasjonsnode":
            ji = el.find(f"{A}jernbaneinformasjon/{A}Jernbaneinformasjon")
            info = {c.tag[len(A):]: (c.text or "") for c in ji}
            lat, lon = map(float, el.find(f".//{G}pos").text.split())
            rec = {"name": (el.findtext(A + "stasjonsnavn") or "").strip(),
                   "type": el.findtext(A + "stasjonstype") or "",
                   "until": el.findtext(A + "gyldigTil"), "code": info.get("banekortnavn", ""),
                   "line": info.get("banenavn", ""), "km": info.get("sporkilometer"),
                   "lon": lon, "lat": lat}
            nodes.append(rec)
            el.clear()
    log(f"Banenettverk: {len(links)} centre-line links ({n_dup} listed twice, dropped), "
        f"{len(nodes)} station nodes ({sum(1 for n in nodes if not n['until'])} current)")
    return links, nodes


def coord_lists(ls):
    """A line's links as coordinate lists, with the bridges `snap` made."""
    return [l["pts"] for l in ls] + [b for l in ls for b in l.get("bridges", ())]


def link_km(l):
    return sum(dist_m(*a, *b) for a, b in zip(l["pts"][:-1], l["pts"][1:])) / 1000


def group(links, log):
    """Kept links by line (banekortnavn, connectors folded in), and what is left out."""
    by, left = defaultdict(list), defaultdict(float)
    names = {}
    for l in links:
        if l["status"] not in STATUS_KEEP or l["purpose"] not in PURPOSE_KEEP:
            left[(l["name"], l["status"], l["purpose"])] += link_km(l)
            continue
        code = fold(l)
        by[code].append(l)
        if l["code"] == code:
            names[code] = l["name"]
    for code in by:
        names.setdefault(code, by[code][0]["name"])
    log(f"Banenettverk: {len(by)} lines in use, {sum(link_km(l) for ls in by.values() for l in ls):,.0f}"
        f" km of centre line; left out {sum(left.values()):,.0f} km "
        f"(status not I or purpose not B/P)")
    snap(by, log)
    return by, names, left


SNAP_M = 1.0        # vertices of the kept links this close are one point
BRIDGE_M = 100.0    # a link end touching nothing is joined to the nearest vertex this close
BRIDGE_OWN_M = 40.0 # ...or to its own line's nearest vertex this close, whatever it touches


def snap(by, log):
    """Join the links into one network, in place.

    Banenettverk's links meet end to end or end on another link's vertex, but not always on
    exactly the same coordinate: 7,434 of 7,558 link ends have another link's vertex within
    1 m and only 3,595 share it exactly, so a graph joined on rounded coordinates fell apart
    (Bergensbanen in 6 pieces, Sørlandsbanen in 7). Vertices within SNAP_M become one point;
    a link end still touching nothing gets a short bridge link to the nearest vertex of
    another link within BRIDGE_M (`l["bridge"]`)."""
    from scipy.spatial import cKDTree
    allv, where, end_idx = [], [], []
    for code, ls in by.items():
        for li, l in enumerate(ls):
            for vi, p in enumerate(l["pts"]):
                if vi == 0 or vi == len(l["pts"]) - 1:
                    end_idx.append(len(allv))
                allv.append(p)
                where.append((code, li, vi))
    P = xyz(allv)
    tree = cKDTree(P)
    parent = list(range(len(allv)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in tree.query_pairs(SNAP_M, output_type="ndarray"):
        i, j = int(i), int(j)
        if where[i][:2] == where[j][:2]:
            continue          # never along one link: its own vertices can be under 1 m apart
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    n_moved = 0
    for k, (code, li, vi) in enumerate(where):
        r = find(k)
        if r != k:
            by[code][li]["pts"][vi] = allv[r]
            n_moved += 1
    # link ends that touch no other link's vertex
    n_bridge = n_loose = 0
    for k in end_idx:
        code, li, _vi = where[k]
        hits = tree.query_ball_point(P[k], BRIDGE_M)
        others = [h for h in hits if where[h][:2] != (code, li)]
        if not others:
            n_loose += 1
            continue
        # Its own line first: each line's graph is built from its own links alone, so a
        # Drammenbanen link ending on Askerbanen's track at Asker, 5 m short of the next
        # Drammenbanen link, cut Drammenbanen in two (Asker - Lier was lost).
        own = [h for h in others if where[h][0] == code
               and float(np.sum((P[h] - P[k]) ** 2)) <= BRIDGE_OWN_M ** 2]
        if any(find(h) == find(k) for h in own):
            continue
        if not own and any(find(h) == find(k) for h in others):
            continue
        h = min(own or others, key=lambda h: float(np.sum((P[h] - P[k]) ** 2)))
        by[code][li].setdefault("bridges", []).append([allv[find(k)], allv[find(h)]])
        n_bridge += 1
    log(f"Banenettverk: {n_moved:,} vertices snapped to a neighbour within {SNAP_M} m; "
        f"{n_bridge} link ends bridged to another link within {BRIDGE_M} m; "
        f"{n_loose} link ends touch nothing (line ends)")


# ================================================================ Entur

def stop_name(n):
    n = re.sub(r"\s+(stasjon|station|Centralstation)$", "", n or "").strip()
    return n


def read_entur(path, log):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    stops = {}
    patterns = []
    n_bus = Counter()
    for l in d["data"]["lines"]:
        for p in l["journeyPatterns"]:
            seq = []
            for q in p["quays"]:
                sp = q["stopPlace"]
                # rail-replacement bus stops ("Asker stasjon Lensmannslia", "Oslo S
                # Trelastgata") are in some rail lines' patterns
                if "rail" not in (sp.get("transportMode") or ["rail"]):
                    n_bus[sp["name"]] += 1
                    continue
                sid = "o" + sp["id"].rsplit(":", 1)[-1]
                stops.setdefault(sid, {"id": sid, "nsr": sp["id"], "name": stop_name(sp["name"]),
                                       "raw": sp["name"], "lon": sp["longitude"],
                                       "lat": sp["latitude"], "lines": set()})
                stops[sid]["lines"].add(l["publicCode"] or l["name"])
                if not seq or seq[-1] != sid:
                    seq.append(sid)
            patterns.append((l, seq))
    log(f"Entur: {len(d['data']['lines'])} rail lines, {len(patterns)} journey patterns, "
        f"{len(stops)} rail stop places called at; {len(n_bus)} stop places of other modes "
        f"in rail patterns left out")
    return stops, patterns, d["data"]["lines"]


# ================================================================ the network graph

class Net:
    """Every kept link's vertices, joined where they share a coordinate (Banenettverk is
    topological: links of one line and of the lines meeting it end on one point)."""

    def __init__(self, by):
        self.num, self.xy = {}, []
        self.adj = defaultdict(list)
        self.line_of = defaultdict(set)
        for code, ls in by.items():
            for pts in coord_lists(ls):
                prev = None
                for x, y in pts:
                    k = self.vid(x, y)
                    self.line_of[k].add(code)
                    if prev is not None and prev != k:
                        w = dist_m(*self.xy[prev], *self.xy[k]) / 1000
                        self.adj[prev].append((k, w))
                        self.adj[k].append((prev, w))
                    prev = k
        from scipy.spatial import cKDTree
        self.tree = cKDTree(xyz(self.xy))

    def vid(self, x, y):
        r = (round(x, 5), round(y, 5))
        k = self.num.get(r)
        if k is None:
            k = self.num[r] = len(self.xy)
            self.xy.append(r)
        return k

    def near(self, lon, lat, r):
        q = xyz([lon, lat])[0]
        d, _j = self.tree.query(q)
        if d > r:
            return d, []
        return d, self.tree.query_ball_point(q, d + ANCHOR_EXTRA_M)

    def path(self, src, dst, cap):
        """Shortest path (km) from any vertex of src to any of dst, no longer than cap."""
        dist = {v: 0.0 for v in src}
        prev = {}
        heap = [(0.0, v) for v in src]
        heapq.heapify(heap)
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if d > cap:
                return None
            if u in done:
                continue
            done.add(u)
            if u in dst:
                out = [u]
                while out[-1] in prev:
                    out.append(prev[out[-1]])
                return out[::-1], d
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None


# ================================================================ build

def stop_lines(by, stops, log):
    """Each stop's distance to every line passing within ATTACH_M: {sid: {code: m}}, and
    {sid: nearest distance or None} for stops on no line (abroad, or off the network)."""
    from scipy.spatial import cKDTree
    trees = {code: cKDTree(xyz([p for l in ls for p in n02_dense(l["pts"])]))
             for code, ls in by.items()}
    cand, off = {}, {}
    for sid, s in stops.items():
        q = xyz([s["lon"], s["lat"]])[0]
        ds = {}
        for code, t in trees.items():
            d, _j = t.query(q)
            if d <= 3000:
                ds[code] = float(d)
        if not ds or min(ds.values()) > ATTACH_M:
            off[sid] = min(ds.values()) if ds else None
            continue
        cand[sid] = {c: d for c, d in ds.items() if d <= ATTACH_M}
    log(f"stops: {len(cand)} of {len(stops)} Entur rail stops lie within {ATTACH_M} m of the "
        f"network")
    return cand, off


HALT_M = 500


def halts_on_own_line(cand, stops, nodes, log):
    """A halt is on its own line only. Bane NOR's station nodes say which line each stop is
    on, but name a junction station under one line alone (Lillestrøm under Gardermobanen), so
    only halts (type I: no points, so no junction) are held to it: Leirsund, a Hovedbanen
    halt 13 m from Gardermobanen's track, took Lillestrøm - Leirsund's trains onto
    Gardermobanen and left Hovedbanen's section with none. Holding stations too (unless
    another line ends within 1 km) was tried and dropped: Banenettverk's lines do not end at
    a shared point at Lillestrøm or Oslo S, so both were taken off Hovedbanen."""
    n = 0
    for sid, ds in cand.items():
        s = stops[sid]
        hit = [x for x in nodes if not x["until"]
               and x["name"] == s["name"] and fold({"code": x["code"], "name": ""}) in ds
               and dist_m(s["lon"], s["lat"], x["lon"], x["lat"]) <= HALT_M]
        if hit and hit[0]["type"] != "I":
            continue
        if hit and len(ds) > 1:
            code = fold({"code": hit[0]["code"], "name": ""})
            cand[sid] = {code: ds[code]}
            n += 1
    log(f"stops: {n} halts near another line held to their own (Bane NOR's station nodes)")


def place_stops(cand, evidence, stops, log):
    """Which stops go on which line. A stop is on a line its trains' paths run over within
    ATTACH_M of it (`evidence`, from the call pairs): Hønefoss is on Randsfjordbanen and
    Bergensbanen, Oslo S and Ski on Follobanen, while Stabekk, beside Askerbanen's tunnel, is
    on Drammenbanen only. A stop no path reaches goes on its nearest line, and on any other
    within NEAR_EXTRA_M of that. Returns {code: [(sid, lon, lat, d)]}."""
    on = defaultdict(list)
    n_ev = n_near = 0
    for sid, ds in cand.items():
        s = stops[sid]
        ev = evidence.get(sid, set()) & set(ds)
        if ev:
            n_ev += 1
            codes = ev
        else:
            n_near += 1
            dmin = min(ds.values())
            codes = {c for c, d in ds.items() if d <= dmin + NEAR_EXTRA_M}
        for c in codes:
            on[c].append((sid, s["lon"], s["lat"], ds[c]))
    log(f"stops: {n_ev} placed on the lines their trains run over, {n_near} (no path reaches "
        f"them) on the nearest line; {sum(len(v) for v in on.values())} stop-line pairs")
    return on


def n02_dense(pts, step=40.0):
    out = [pts[0]]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        n = max(1, int(dist_m(x1, y1, x2, y2) // step))
        out += [(x1 + (x2 - x1) * t / n, y1 + (y2 - y1) * t / n) for t in range(1, n + 1)]
    return out


def junctions(by, on, stops, net, credited, nodes, log):
    """Points where two lines trains run over meet away from any stop, and the border points
    at line ends. A line is live if it has a stop or LIVE_KM of track a call pair's path
    runs over (Follobanen has no stop of its own: it leaves Østfoldbanen's tracks outside
    Oslo S and joins them again north of Ski). Returns {code: [(id, lon, lat)]}, {id: record}
    and the live lines."""
    run_km = Counter()
    for u, v in credited:
        if u < v:
            for c in net.line_of[u] & net.line_of[v]:
                run_km[c] += dist_m(*net.xy[u], *net.xy[v]) / 1000
    live = {c for c in by if on.get(c) or run_km[c] >= LIVE_KM}
    extra, recs = defaultdict(list), {}
    # every line end, and where a bridge from it lands on another line
    ends = set()
    for code in live:
        for l in by[code]:
            for p in (l["pts"][0], l["pts"][-1]):
                ends.add((round(p[0], 5), round(p[1], 5)))
            for _a, far in l.get("bridges", ()):
                ends.add((round(far[0], 5), round(far[1], 5)))
    # how many link ends of each line lie on a point: one means the line ends there
    # (the line's own links only: a folded connector's end, as Gardermobanen's spurs onto
    # Hovedbanen at Sagdalen and Jessheim, is no end of the line)
    n_ends = Counter()
    for code in live:
        for l in by[code]:
            for p in (l["pts"][0], l["pts"][-1]):
                n_ends[(code, (round(p[0], 5), round(p[1], 5)))] += 1 if l["code"] == code else 2
    n_j = n_at_stop = 0
    at_stop = []
    for p in sorted(ends):
        k = net.num.get(p)
        codes = (net.line_of.get(k, set()) if k is not None else set()) & live
        if len(codes) < 2:
            continue
        near = {c: {s[0] for s in on[c] if dist_m(p[0], p[1], s[1], s[2]) <= STOP_JUNCTION_M}
                for c in codes}
        # a stop near enough on every line: that stop joins the lines already (Eidsvoll,
        # where Gardermobanen meets Hovedbanen 1 km south of the station both reach)
        if set.intersection(*near.values()):
            continue
        # a line that ends here, a stop near on another line: the line starts at that stop
        # (Arendalsbanen leaves Sørlandsbanen 0.5 km out of Nelaug, where its trains start)
        others = {s for c in codes for s in near[c]}
        if others:
            # the stop Bane NOR's own station node there is named for (Meråkerbanen starts at
            # Hell, not at Trondheim lufthavn 1 km off), else the nearest
            bn = min(((dist_m(p[0], p[1], n["lon"], n["lat"]), n["name"]) for n in nodes
                      if not n["until"]), default=(INF, ""))
            named = [s for s in others if bn[0] <= STOP_JUNCTION_M
                     and stops[s]["name"] == bn[1]]
            sid = named[0] if named else min(
                others, key=lambda s: dist_m(p[0], p[1], stops[s]["lon"], stops[s]["lat"]))
            took = [c for c in codes if sid not in near[c] and n_ends[(c, p)] == 1
                    and all(x[0] != sid for x in extra[c])]
            if took and all(sid in near[c] or c in took for c in codes):
                for c in took:
                    extra[c].append((sid, p[0], p[1]))
                at_stop.append(f"{'/'.join(sorted(took))} at {stops[sid]['name']}")
                n_at_stop += 1
                continue
        jid = "j" + hashlib.blake2b(f"{p[0]:.5f},{p[1]:.5f}".encode(), digest_size=4).hexdigest()
        if jid in recs:
            continue
        recs[jid] = {"id": jid, "lon": p[0], "lat": p[1], "codes": codes}
        for c in codes:
            extra[c].append((jid, p[0], p[1]))
        n_j += 1
    # border points: at a line end within BORDER_M
    for bid, (lon, lat) in BORDERS.items():
        best = None
        for code in live:
            for l in by[code]:
                for p in (l["pts"][0], l["pts"][-1]):
                    d = dist_m(lon, lat, *p)
                    if d <= BORDER_M and (best is None or d < best[0]):
                        best = (d, code, p)
        if best is None:
            log(f"border {bid}: no line end within {BORDER_M} m")
            continue
        d, code, p = best
        recs[bid] = {"id": bid, "lon": p[0], "lat": p[1], "codes": {code}, "border": True,
                     "d": d}
        extra[code].append((bid, p[0], p[1]))
        log(f"border {bid}: {code} ends {d:.0f} m from it")
    log(f"junctions: {n_j} where two live lines meet away from a stop; {n_at_stop} line ends "
        f"near a stop on the other line start at that stop ({', '.join(at_stop)}); live lines "
        f"with no stop: {', '.join(sorted(c for c in live if not on.get(c))) or 'none'}")
    return extra, recs, live


GAP_KM = 2.0


def path_km(pts):
    return sum(dist_m(*a, *b) for a, b in zip(pts[:-1], pts[1:])) / 1000


def fill_gaps(coord_lists_, net):
    """Paths over the whole network joining the pieces of one line's own centre line, each
    no longer than GAP_KM. Bane NOR files some of a line's own track under another
    strekning's code: Drammenbanen through Asker station runs over links coded Askerbanen
    and "Askerbanen spor mot Spikkestadbanen", which left Asker - Lier out of Drammenbanen."""
    fills = []
    for _round in range(40):
        g, xy = n02.build_graph(coord_lists_ + fills)
        seen, comps = set(), []
        for v in g:
            if v in seen or v < 0:
                continue
            stack, c = [v], set()
            seen.add(v)
            while stack:
                u = stack.pop()
                if u >= 0:
                    c.add(u)
                for w, _ in g[u]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            comps.append(c)
        if len(comps) < 2:
            break
        comps.sort(key=len, reverse=True)
        ids = [{net.num[xy[u]] for u in c if xy[u] in net.num} for c in comps]
        got = None
        for i, src in enumerate(ids):
            rest = set().union(*(ids[j] for j in range(len(ids)) if j != i))
            got = net.path(src, rest, GAP_KM)
            if got:
                break
        if not got:
            break
        fills.append([net.xy[v] for v in got[0]])
    return fills


ALONG_M, ALONG_SHARE = 40.0, 0.85
LIVE_KM = 1.0


def redundant(sections, force=None):
    """Sections of one line lying along its other sections: a stop on a loop or side track
    beside the through track (Katterat on Ofotbanen) gives Rombak - Søsterbekk past it as
    well as Rombak - Katterat - Søsterbekk. Longest first, a section with ALONG_SHARE of its
    length within ALONG_M of the line's other sections goes, if its ends stay joined.
    `force`: judge that one section on its length alone, joined ends or not."""
    from scipy.spatial import cKDTree
    gone = []
    for key in ([force] if force else sorted(sections, key=lambda k: -sections[k]["km"])):
        others = [k for k in sections if k != key and k not in gone]
        if not others:
            continue
        pts = xyz(n02_dense(sections[key]["geom"], 25.0))
        tree = cKDTree(xyz([p for k in others for p in n02_dense(sections[k]["geom"], 25.0)]))
        d, _ = tree.query(pts)
        if (d <= ALONG_M).mean() < ALONG_SHARE:
            continue
        if force:
            gone.append(key)
            continue
        adj = defaultdict(set)
        for x, y in others:
            adj[x].add(y)
            adj[y].add(x)
        a, b = key
        seen, stack = {a}, [a]
        while stack:
            for v in adj[stack.pop()] - seen:
                seen.add(v)
                stack.append(v)
        if b in seen:
            gone.append(key)
    return gone


def tidy_junctions(lines):
    """A junction (not a border point) only one line's kept sections reach, once the sections
    no train runs over are gone, is no longer where lines meet: between two of that line's
    sections it is fused away (Sarpsborg's Hafslundsløyfa junction on Østfoldbanen), and a
    dead-end section to it lying along the line's other sections is dropped (Østfoldbanen
    vestre linje's section from where østre linje joins, beside Sarpsborg - Halden). In place
    on each line's "_keep"; returns (fused, dropped)."""
    users = defaultdict(set)
    for l in lines:
        for key in l["_keep"]:
            for x in key:
                if x.startswith("j"):
                    users[x].add(l["id"])
    n_fused = n_spur = 0
    for l in lines:
        keep = l["_keep"]
        for j in sorted(x for x, u in users.items() if u == {l["id"]}):
            at = [k for k in keep if j in k]
            if len(at) == 2:
                (k1, v1), (k2, v2) = [(k, keep[k]) for k in at]
                a = k1[0] if k1[1] == j else k1[1]
                b = k2[0] if k2[1] == j else k2[1]
                if a == b or (a, b) in keep or (b, a) in keep:
                    continue
                g1 = v1["geom"] if k1[1] == j else v1["geom"][::-1]       # a ... j
                g2 = v2["geom"] if k2[0] == j else v2["geom"][::-1]       # j ... b
                v1o = v1["verts"] if k1[1] == j else v1["verts"][::-1]
                v2o = v2["verts"] if k2[0] == j else v2["verts"][::-1]
                del keep[k1], keep[k2]
                keep[(a, b)] = {"km": v1["km"] + v2["km"], "geom": g1 + g2[1:],
                                "verts": v1o + v2o, "reverses": False}
                n_fused += 1
        for j in sorted(x for x, u in users.items() if u == {l["id"]}):
            at = [k for k in keep if j in k]
            if len(at) == 1 and len(keep) > 1 and at[0] in redundant(
                    {k: keep[k] for k in keep}, force=at[0]):
                del keep[at[0]]
                n_spur += 1
    return n_fused, n_spur


def run_paths(net, stops, cand, patterns, log):
    """Every distinct pair of consecutive calls, as the shortest track between the two stops
    (module docstring). Returns the credited edges (both ways), {stop: stops abroad called
    next to it}, and {stop: lines its paths run over within ATTACH_M of it}."""
    anchors = {}
    for sid in cand:
        s = stops[sid]
        _d, hits = net.near(s["lon"], s["lat"], ATTACH_M)
        hits = [h for h in hits if net.line_of[h] & set(cand[sid])]
        if hits:
            anchors[sid] = set(hits)
    credited = set()
    evidence = defaultdict(set)
    n_pairs = n_nopath = n_abroad = 0
    pair_seen = set()
    abroad_from = defaultdict(set)
    nopath = []
    t0 = time.time()

    def near_lines(sid, vs):
        s = stops[sid]
        out = set()
        for u, v in zip(vs[:-1], vs[1:]):
            if dist_m(s["lon"], s["lat"], *net.xy[u]) > ATTACH_M:
                break
            out |= net.line_of[u] & net.line_of[v]
        return out

    for _l, seq in patterns:
        for a, b in zip(seq[:-1], seq[1:]):
            key = (a, b) if a <= b else (b, a)
            if key in pair_seen:
                continue
            pair_seen.add(key)
            ia, ib = a in anchors, b in anchors
            if ia != ib:
                (abroad_from[a] if ia else abroad_from[b]).add(b if ia else a)
                n_abroad += 1
                continue
            if not ia:
                continue
            n_pairs += 1
            sa, sb = stops[a], stops[b]
            cap = DETOUR * dist_m(sa["lon"], sa["lat"], sb["lon"], sb["lat"]) / 1000 + DETOUR_KM
            got = net.path(anchors[a], anchors[b], cap)
            if got is None:
                n_nopath += 1
                nopath.append(f"{sa['name']} - {sb['name']}")
                continue
            vs = got[0]
            credited.update(zip(vs[:-1], vs[1:]))
            credited.update(zip(vs[1:], vs[:-1]))
            evidence[a] |= near_lines(a, vs)
            evidence[b] |= near_lines(b, vs[::-1])
    log(f"served: {n_pairs} distinct call pairs in Norway, {n_nopath} with no path within "
        f"{DETOUR} x crow-fly + {DETOUR_KM:.0f} km; {n_abroad} pairs to a stop abroad or off "
        f"the network; {time.time() - t0:.0f} s")
    if nopath:
        log(f"    no path: {', '.join(nopath)}")
    return credited, abroad_from, evidence


def build(path, log):
    path = Path(path)
    raw = path if path.is_dir() else path.parent
    links, nodes = read(raw / GML, log)
    by, names, left = group(links, log)
    stops, patterns, elines = read_entur(raw / ENTUR, log)
    cand, off = stop_lines(by, stops, log)
    halts_on_own_line(cand, stops, nodes, log)
    net = Net(by)
    credited, abroad_from, evidence = run_paths(net, stops, cand, patterns, log)
    on = place_stops(cand, evidence, stops, log)
    extra, jrec, live = junctions(by, on, stops, net, credited, nodes, log)

    # --- sections per line, n02's way
    lines, geoms, out_st = [], {}, {}
    sec_pts = {}             # (line id, a, b) -> list of network vertex ids on it (originals)
    n_rev = n_red = 0
    km_red = 0.0
    filled, red = [], []
    for code, ls in sorted(by.items()):
        pts = [(sid, x, y, [(x, y)]) for sid, x, y, _d in on.get(code, ())]
        pts += [(jid, x, y, [(x, y)]) for jid, x, y in extra.get(code, ())]
        if code not in live or len(pts) < 2:
            continue
        fills = fill_gaps(coord_lists(ls), net)
        if fills:
            filled.append(f"{code} {len(fills)} ({sum(path_km(f) for f in fills):.1f} km)")
        g, xy = n02.build_graph(coord_lists(ls) + fills)
        foot, centre, offsets = n02.footprints(xy, pts, n02.FOOT_M)
        if len(centre) < 2:
            continue
        sections = {}
        for a, b in sorted(n02.neighbours(g, foot)):
            got = n02.between(g, foot, a, b, offsets[a], offsets[b])
            if got is None:
                continue
            nodes_, km = got
            sections[(a, b)] = {"km": km, "geom": [centre[a]] + [
                xy[n] for i, n in enumerate(nodes_)
                if n >= 0 or i == 0 or i == len(nodes_) - 1] + [centre[b]],
                "verts": [xy[n] for n in nodes_ if n >= 0],
                "reverses": n02.reverses([xy[n] for n in nodes_])}
        n_rev += len(n02.drop_reversing(sections))
        for key in redundant(sections):
            n_red += 1
            km_red += sections[key]["km"]
            red.append(f"{names[code]} {'-'.join(_nm(x, stops, jrec) for x in key)} "
                       f"{sections.pop(key)['km']:.1f} km")
        if not sections:
            continue
        lid = line_id(code)
        name = names[code]
        lines.append({
            "id": lid, "src": "banenor", "service": False,
            "name": name, "name_en": NAME_EN.get(name, ""), "ref": "", "colour": "",
            "operator": "Bane NOR", "operator_en": "", "network": "", "kind": "rail",
            "km": 0.0, "variants": 1, "straight_sections": 0, "display": [],
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sorted(sections.items())],
            "_code": code, "_sec": sections, "_centre": centre})
    log(f"lines: {len(lines)} with sections ({n_rev} sections dropped for doubling back, "
        f"{n_red} ({km_red:.1f} km) for lying along others of their line: {'; '.join(red)})")
    log(f"lines: gaps in a line's own centre line crossed over other lines' links: "
        f"{', '.join(filled) or 'none'}")

    # --- which sections trains run over (docstring; the paths are `run_paths`')
    def share(verts):
        ids = [net.num.get(p) for p in verts]
        tot = on_ = 0.0
        for u, v in zip(ids[:-1], ids[1:]):
            if u is None or v is None or u == v:
                continue
            w = dist_m(*net.xy[u], *net.xy[v])
            tot += w
            if (u, v) in credited:
                on_ += w
        return on_ / tot if tot else 0.0

    kept_lines = []
    unserved_stop, dropped_j = [], []
    stop_names = {s["name"] for s in stops.values()}
    for l in lines:
        secs = l.pop("_sec")
        centre = l.pop("_centre")
        code = l.pop("_code")
        keep = {}
        for (a, b), v in secs.items():
            sh = share(v["verts"])
            ends_j = [x for x in (a, b) if not x.startswith("o")]
            if ends_j:
                # a border section: run over if a train calls at its stop and then abroad
                bid = next((x for x in ends_j if x in BORDERS), None)
                other = b if a == bid else a
                if bid and other.startswith("o") and abroad_from.get(other):
                    bx, by_ = BORDERS[bid]
                    so = stops[other]
                    if any(dist_m(stops[f]["lon"], stops[f]["lat"], bx, by_)
                           < dist_m(stops[f]["lon"], stops[f]["lat"], so["lon"], so["lat"])
                           for f in abroad_from[other]):
                        sh = 1.0
                if sh < SERVED_SHARE:
                    dropped_j.append((v["km"], l["name"], a, b, sh))
                    continue
            elif sh < SERVED_SHARE:
                # Entur's timetable is every train in the country: two stops no call pair's
                # path joins over this track have no train between them over it (Roa -
                # Hønefoss, freight only, once Roa-Hønefossbanen's ends took the stops there)
                unserved_stop.append((v["km"], l["name"], a, b, sh))
                continue
            keep[(a, b)] = v
        if keep:
            l["_keep"] = keep
            l["_code"] = code
            kept_lines.append(l)
    n_fused, n_spur = tidy_junctions(kept_lines)
    log(f"junctions only one line's sections reach: {n_fused} fused away between two of its "
        f"sections, {n_spur} dead-end sections to one dropped as lying along the line's others")
    for l in kept_lines:
        keep = l.pop("_keep")
        code = l.pop("_code")
        l["sections"] = [[a, b, round(v["km"], 3)] for (a, b), v in sorted(keep.items())]
        l["km"] = round(sum(v["km"] for v in keep.values()), 3)
        l["display"] = n02.walk_order(keep.keys())
        geoms[l["id"]] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                          for (a, b), v in keep.items()}
        for (a, b) in keep:
            for sid in (a, b):
                if sid not in out_st:
                    if sid.startswith("o"):
                        s = stops[sid]
                        out_st[sid] = {"id": sid, "name": s["name"], "name_en": "",
                                       "lon": s["lon"], "lat": s["lat"], "lines": set()}
                    else:
                        r = jrec[sid]
                        out_st[sid] = {"id": sid, "name": junction_name(r, nodes, names, stop_names),
                                       "name_en": "", "lon": r["lon"], "lat": r["lat"],
                                       "lines": set(), "junction": True}
                out_st[sid]["lines"].add(l["id"])
        l["_code"] = code
    log(f"junction-ended sections no train runs over, left out: {len(dropped_j)} "
        f"({sum(x[0] for x in dropped_j):,.1f} km)")
    for km, name, a, b, sh in sorted(dropped_j, reverse=True)[:40]:
        log(f"    {km:7.2f} km  {name}  {_nm(a, stops, jrec)} - {_nm(b, stops, jrec)}  ({sh:.0%})")
    log(f"stop-to-stop sections no call pair credited, left out: {len(unserved_stop)} "
        f"({sum(x[0] for x in unserved_stop):,.1f} km)")
    for km, name, a, b, sh in sorted(unserved_stop, reverse=True)[:40]:
        log(f"    {km:7.2f} km  {name}  {_nm(a, stops, jrec)} - {_nm(b, stops, jrec)}  ({sh:.0%})")
    for l in kept_lines:
        l.pop("_code")
    total = sum(l["km"] for l in kept_lines)
    log(f"NO: {len(kept_lines)} register lines, {total:,.0f} km, {len(out_st)} stations "
        f"({sum(1 for s in out_st.values() if s.get('junction'))} junctions or border points)")
    off_named = sorted(stops[s]["raw"] for s, d in off.items() if d is not None and d < 3000)
    if off_named:
        log(f"Entur stops 0.4-3 km from the network, on no line: {', '.join(off_named)}")
    return kept_lines, out_st, geoms


def _nm(sid, stops, jrec):
    if sid in stops:
        return stops[sid]["name"]
    return sid


def junction_name(r, nodes, names, stop_names=frozenset()):
    """A junction's name: Bane NOR's station node nearest it (within 1.5 km), else the lines.
    A name a stop also has gets " (junction)": build_model matches OSM stations onto register
    stations by name, and a junction called "Eidsvoll" took Eidsvoll station's OSM record."""
    name = _junction_name(r, nodes, names)
    return f"{name} (junction)" if name in stop_names else name


def _junction_name(r, nodes, names):
    best = None
    for n in nodes:
        if n["until"]:
            continue
        d = dist_m(r["lon"], r["lat"], n["lon"], n["lat"])
        if d <= 1500 and (best is None or d < best[0]):
            best = (d, n["name"])
    if r.get("border"):
        return "Norway – Sweden border"
    if best:
        return best[1]
    return " / ".join(sorted(names.get(c, c) for c in r["codes"]))


# ================================================================ --dry

def dry():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    lines, st, geoms = build(RAW, log)
    links, _nodes = read(RAW / GML, lambda m: None)
    by, names, left = group(links, lambda m: None)
    span = {}
    for code, ls in by.items():
        ks = [k for l in ls if l["code"] == code for k in (l["km0"], l["km1"])]
        if ks:
            span[names[code]] = (min(ks), max(ks), sum(link_km(l) for l in ls))
    print(f"\n{'built':>8} {'secs':>4}  {'Bane NOR km span':>18} {'centre line':>11}  line")
    for l in sorted(lines, key=lambda l: -l["km"]):
        s = span.get(l["name"])
        sp = f"{s[0]:7.1f}-{s[1]:7.1f}" if s else ""
        cl = f"{s[2]:9.1f}" if s else ""
        print(f"{l['km']:8.1f} {len(l['sections']):4d}  {sp:>18} {cl:>11}  {l['name']}")
    print("\nleft out of the register (km of centre line by line, status, purpose):")
    for (name, stt, pur), km in sorted(left.items(), key=lambda x: -x[1]):
        print(f"  {km:7.1f}  {stt} {pur}  {name}")
    built = {l["name"] for l in lines}
    print("\nlines in use with no section built: "
          + ", ".join(sorted(n for n in names.values() if n not in built)))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    elif "--fetch-entur" in sys.argv:
        fetch(entur_only=True)
    elif "--dry" in sys.argv:
        dry()
    else:
        sys.exit(__doc__)
