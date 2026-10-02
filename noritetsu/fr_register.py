"""Lines, stations and sections for France, from SNCF Réseau's register of the national network.

    python fr_register.py --fetch             # download the register files into data/raw/fr
    python build_model.py --region fr --register fr_register:data/raw/fr

Files, all open data (ODbL, SNCF Réseau; Wikidata CC0), in data/raw/fr, fetched by --fetch:

  lignes-par-statut.geojson             every RFN line by its six-digit code (830000 is Paris-Lyon
      to Marseille-Saint-Charles), cut into portions of one status (Exploitée, Neutralisée,
      Fermée...), each with its PK at both ends and its geometry (decametric, 1:50,000)
  lignes-par-type.geojson               whether a code is a line, a connecting curve (Rac), a
      private siding (Vmère) or port track (Vport)
  lignes-lgv-et-par-ecartement.geojson  which portions are high-speed (LGV) or narrow gauge
  liste-des-gares.geojson               every station with its UIC code, its line, its PK on
      that line and a `voyageurs` flag (O/N) saying whether it is a passenger station
  wikidata_lines.json                   line names: Wikidata items with P1671 (route number)
      in France. SNCF's own files carry the code where the name used to be.
  voies-de-ligne.geojson                every line track (type_voie VPL) of the per-track file,
      for the lines the files above lack and for coarse geometry (see below); optional

WHAT THE REGISTER IS.  About 1,590 line codes, of which some 780 portions are exploited. A line
is a chainage axis: its PK (point kilométrique) runs from its origin, which need not be where the
line physically starts (Meaux is PK 44 on line 070000, which begins at Noisy-le-Sec at PK 8.9).
A line can be several tronçons (rg_troncon), each with its own continuous PK. The unit counted
here is the line code, as the km-line is in Switzerland (schienennetz.py), and this reader has
the same shape: every end of a line is a section end, flagged `junction` unless it is a stop,
and build_model keeps a junction-ended section only where OSM passenger routes run over it.

HOW IT DIFFERS FROM SWITZERLAND:

- THERE IS NO NODE TOPOLOGY. A portion is a polyline with a PK at each end; the export also
  splits one portion into several parts (the MultiLineString exploded), sometimes out of order,
  and the Breil-Tende line's spiral tunnel is a part that starts where it ends. Parts are
  chained end to end (JOIN_M), and portions of one code meeting at a point become one node.
- STATIONS ARE PLACED BY THEIR PK AND THEIR PROJECTION. Every passenger station of a line is
  projected onto that line's polyline; the stations and the portion ends are the anchors of the
  chainage, so every section's `chain` is an exact PK difference.
- THE PASSENGER FLAG IS PER STATION, NOT PER LINE. SNCF projects a station onto every line near
  it ("lorsqu'un élément se trouve à proximité de plusieurs lignes, il est projeté sur les
  différentes lignes"), so a freight line that passes two passenger stations would come out as a
  stop-to-stop section, which build_model never questions: the Chartres-Orléans line (556000)
  has Voves as a passenger station because the Paris-Tours line crosses it there. So this
  reader asks OSM itself, with build_model's own test (register_way_lines): a stop-to-stop
  section that no passenger route runs over (under STOP_SHARE of its track) is dropped here.
- ONLY THE LINES. Connecting curves (Rac), sidings and port track are left out: 820 curves,
  mostly under 3 km and unnamed, would each be a line of their own in a rider's list. A curve's
  stations are all on a line too (Rungis on 985000, St-Quentin-en-Yvelines on 420000).
- NAMES come from Wikidata's route numbers, then OSM's route=railway relations whose ref is the
  code, then "Ligne 830 000". Station names are OSM's where an OSM station of a compatible
  name is near (SNCF writes "St-Nom-la-Bretêche-Forêt-de-Marly", OSM "Saint-Nom-la-Bretèche -
  Forêt de Marly"), so build_model's merge finds them; SNCF stations that land on one OSM
  station are one station (Paris-Gare-de-Lyon and its RER platforms, Paris-Gare-de-Lyon-
  Souterrain).
- SNCF'S LIST MISSES A FEW STOPS: Paris-Montparnasse's main halls, Versailles-Rive-Droite,
  and the RATP-owned Châtelet - Les Halles on the RER D tunnel (981000). An OSM station on a
  line joins it where an OSM route calls at it and at both its neighbours on the line (a T4
  stop beside 070000 at Bondy does not), and a line that ends within END_SNAP_M of an OSM
  station ends there. The build log lists both kinds, and the ones refused.
- HIGH SPEED: LGV lines (lignes-lgv-et-par-ecartement) are `highspeed`. A classic line's
  sections are flagged not high-speed per section (`highspeed_sections`), except where OSM
  tags its track highspeed=yes (200-220 km/h stretches), which are left unknown so the
  trains OSM puts there still credit it.
- TRAM-TRAINS: a line most of whose ridden length is OSM light rail track (T11 on 960000,
  Esbly-Crécy 071000) takes kind light_rail. Tram-trains on railway=rail track (Nantes -
  Châteaubriant 519000, Lyon - Montbrison 782000) leave their line rail. The Grande Ceinture
  (990000) is mixed: RER C and T12 run on rail track, T13 on light rail track, and it stays
  rail, so T13 rides do not credit it.
- NATIONAL BORDERS: a line's dead end near a RINF border point (borders.py) ends AT it, under
  its id, so the neighbour's half joins there (`snap_borders`): cut where the point lies on
  the track short of the end (Basel 900 m, Portbou 860 m, Le Locle 710 m, Geneva 15 km), or
  drawn on where the end stops up to BORDER_GAP_M short of it (Jeumont 89 m).
- LINES SNCF'S LINE FILES LACK. lignes-par-statut has no rows at all for some exploited lines
  that the per-track file (voies-de-ligne.geojson, "Fichier de formes des voies du réseau
  ferré national", line tracks only) has: the LGV Interconnexion Est (226310), Douai -
  Blanc-Misseron (262000), Lamothe - Arcachon (657000), the LGV branch Pasilly - Aisy
  (768300), Bondy - Aulnay (958000, T4). Those are read from their first track, and take the
  passenger stations SNCF lists under other lines on their track plus OSM's (`added`). A
  portion whose geometry has a vertex only every COARSE_M or more (LGV Est from Baudrecourt,
  one per 1.5 km, 46 of its 124 km more than 40 m off the rails) takes its track's shape.
  Without the file both are skipped and the rest builds as before.
- THE REGISTER IS CLIPPED TO THE EXTRACT. A section is kept only where the OSM extract has rail
  around nearly all of it (COVER_*), so a regional extract (Ile-de-France, for development)
  gets the register lines it can check, and the whole-country extract loses nothing.

The `path` argument is data/raw/fr; the OSM half is read from data/proc/<name of path>.
"""
import hashlib
import heapq
import json
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

SNCF_URL = ("https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/"
            "{}/exports/geojson")
SNCF_FILES = ("lignes-par-statut", "lignes-par-type", "lignes-lgv-et-par-ecartement",
              "liste-des-gares")
# The per-track file, line tracks only (type_voie VPL): see "LINES SNCF'S LINE FILES LACK".
VOIES_DS = "fichier-de-formes-des-voies-du-reseau-ferre-national"
VOIES_FILE = "voies-de-ligne.geojson"
VOIES_WHERE = 'type_voie="VPL"'
WIKIDATA = "wikidata_lines.json"
WIKIDATA_QUERY = """
SELECT ?item ?num ?label ?len ?enlabel WHERE {
  ?item wdt:P1671 ?num ; wdt:P17 wd:Q142 .
  OPTIONAL { ?item rdfs:label ?label FILTER(lang(?label) = "fr") }
  OPTIONAL { ?item rdfs:label ?enlabel FILTER(lang(?enlabel) = "en") }
  OPTIONAL { ?item wdt:P2043 ?len }
}"""

EXPLOITED = "Exploitée"
OPERATOR = "SNCF Réseau"

JOIN_M = 50          # part and portion ends this close are one point
ON_LINE_M = 150      # a station's projection may be this far from our polyline of its line
PK_SLACK_KM = 0.3    # a station's PK may be this far outside its portion's PK range
END_SNAP_M = 250     # a line end this close to a passenger station ends at that station
STUB_M = 400         # a junction-ended section shorter than this beside a stop is tail track
OSM_NAME_M = 1000    # an OSM station of the same name may be this far from SNCF's point
OSM_SUBSET_M = 600   # ... of a name one word-set inside the other
OSM_BLIND_M = 80     # ... of any name at all
COVER_CELL = 0.02    # degrees; the extract's coverage grid
COVER_SHARE = 0.9    # a section is kept when this share of its vertices is covered
STOP_SHARE = 0.5     # a stop-to-stop section needs this share of its track on OSM routes,
STOP_SURE = 0.75     # ... and below this, one OSM route that calls at both its stations
JUNCTION_NAME_M = 1500   # a junction takes the name of a point SNCF lists this close
RIDE_M = 40          # a point of a section is ridden with a passenger route's way this close
RIDE_STEP_M = 50     # ... tested this often along it
RIDE_KINDS = {"train", "tram", "light_rail"}
SAME_NAME_M = 500    # two stations of one name this close are one station
OSM_ON_LINE_M = 120  # an OSM station SNCF does not list may be this far from a line's track
EXTRA_APART_M = 150  # ... and must be this far from every station SNCF does list
CALL_M = 600         # a route's stop node belongs to a station this close (and of its name)
HS_M = 25            # a classic section is beside OSM highspeed=yes track this close ...
HS_SHARE = 0.5       # ... along this share of it
BORDER_ON_M = 100    # a border point this close to a line's track lies on it
BORDER_REACH_M = 2000  # ... and is its border when this far along from a dead end, or less,
BORDER_FAR_M = 30000   # ... or this far, when what lies past it is outside the country
BORDER_GAP_M = 100   # a dead end stopping this short of a border point is drawn on to it
COARSE_M = 1000      # a portion with a vertex only every this many metres takes the track's shape
BORROW_M = 100       # a line SNCF lists no stations for takes passenger stations this close

# Words that say nothing about which station a name is.
STOPWORDS = {"gare", "de", "du", "des", "la", "le", "les", "l", "d", "sur", "en", "et", "a",
             "aux", "au", "tt", "souterrain", "souterraine", "sncf", "rer", "ville", "station"}


def log_default(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- small helpers

def pk_km(s):
    """'049+056' -> 49.056, '000-771' -> -0.771. A few PKs are lettered ('D+841'): None."""
    s = (s or "").strip()
    m = re.fullmatch(r"(\d+)([+-])(\d+)", s)
    if not m:
        return None
    return int(m.group(1)) + (1 if m.group(2) == "+" else -1) * int(m.group(3)) / 1000.0


def pk_str(km):
    """49.056 -> '049+056', -0.771 -> '000-771': pk_km's inverse."""
    m = int(round(abs(km) * 1000))
    return f"{m // 1000:03d}+{m % 1000:03d}" if km >= 0 else f"000-{m:03d}"


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def line_id(code):
    h = hashlib.blake2b(f"fr|{code}".encode("utf-8"), digest_size=5)
    return "f" + h.hexdigest()


def code_ref(code):
    return f"{code[:3]} {code[3:]}"


def fold(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return s.casefold()


def name_words(name):
    s = fold(name)
    s = re.sub(r"\bste\b", "sainte", s)
    s = re.sub(r"\bst\b", "saint", s)
    return frozenset(w for w in re.split(r"[^a-z0-9]+", s) if w and w not in STOPWORDS)


def pretty_station(name):
    """SNCF's abbreviations written out: St-Cloud -> Saint-Cloud, Ste-Colombe -> Sainte-."""
    s = re.sub(r"(^|[\s\-'(])St-", r"\1Saint-", name or "")
    s = re.sub(r"(^|[\s\-'(])Ste-", r"\1Sainte-", s)
    s = re.sub(r"(^|[\s\-'(])Sts-", r"\1Saints-", s)
    return s


class Poly:
    """A polyline in lon/lat with true lengths, and projection onto it."""

    def __init__(self, pts):
        self.p = np.asarray(pts, dtype=np.float64)
        a, b = self.p[:-1], self.p[1:]
        lat = np.radians((a[:, 1] + b[:, 1]) / 2)
        self.seg = np.hypot((b[:, 0] - a[:, 0]) * np.cos(lat) * 111320,
                            (b[:, 1] - a[:, 1]) * 110570)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.length = float(self.cum[-1])

    def project(self, lon, lat):
        """(measure in metres along the line, distance in metres off it)."""
        k = math.cos(math.radians(lat))
        a, b = self.p[:-1], self.p[1:]
        ax, ay = (a[:, 0] - lon) * k * 111320, (a[:, 1] - lat) * 110570
        bx, by = (b[:, 0] - lon) * k * 111320, (b[:, 1] - lat) * 110570
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.where(L2 > 0, -(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1), 0.0)
        t = np.clip(t, 0, 1)
        px, py = ax + t * dx, ay + t * dy
        d = np.hypot(px, py)
        i = int(np.argmin(d))
        return float(self.cum[i] + t[i] * self.seg[i]), float(d[i])

    def point(self, m):
        m = min(max(m, 0.0), self.length)
        i = int(np.searchsorted(self.cum, m, side="right") - 1)
        i = min(max(i, 0), len(self.seg) - 1)
        t = (m - self.cum[i]) / self.seg[i] if self.seg[i] > 0 else 0.0
        return tuple(self.p[i] + t * (self.p[i + 1] - self.p[i]))

    def cut(self, m1, m2):
        lo, hi = sorted((m1, m2))
        i = np.searchsorted(self.cum, lo, side="right")
        j = np.searchsorted(self.cum, hi, side="left")
        pts = [self.point(lo)] + [tuple(q) for q in self.p[i:j]] + [self.point(hi)]
        return pts if m1 <= m2 else pts[::-1]


# ---------------------------------------------------------------- reading

def fetch(raw):
    import urllib.parse
    import urllib.request
    raw.mkdir(parents=True, exist_ok=True)
    for ds in SNCF_FILES:
        dst = raw / f"{ds}.geojson"
        urllib.request.urlretrieve(SNCF_URL.format(ds), dst)
        print(f"{dst.name}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    dst = raw / VOIES_FILE
    urllib.request.urlretrieve(SNCF_URL.format(VOIES_DS) + "?" + urllib.parse.urlencode(
        {"where": VOIES_WHERE}), dst)
    print(f"{dst.name}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": WIKIDATA_QUERY, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "noritetsu-map/0.1"})
    rows = json.load(urllib.request.urlopen(req, timeout=180))["results"]["bindings"]
    out = [{k: v["value"] for k, v in r.items()} for r in rows]
    (raw / WIKIDATA).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{WIKIDATA}: {len(out)} items", flush=True)


def features(raw, ds):
    f = raw / f"{ds}.geojson"
    if not f.exists():
        raise SystemExit(f"{f} is missing; run python fr_register.py --fetch")
    return json.loads(f.read_text(encoding="utf-8"))["features"]


def main_tracks(raw, log):
    """{code: [(rg_troncon, pkd, pkf, coords)]} from the per-track file: for each tronçon, its
    first track ("V1", "V1B", "UNIQUE"...), and a second or other track only over PK the first
    does not cover (262 000 is double track to PK 247, single track on). {} without the file,
    and the reader then works as it did before the file was used."""
    f = raw / VOIES_FILE
    if not f.exists():
        log(f"FR: no {VOIES_FILE} (python fr_register.py --fetch); lines SNCF's line files "
            f"lack are left out and coarse geometry is kept")
        return {}
    recs = defaultdict(list)
    for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
        p, g = x["properties"], x.get("geometry")
        a, b = pk_km(p.get("pk_debut_r")), pk_km(p.get("pk_fin_r"))
        if not g or g["type"] != "LineString" or a is None or b is None or b <= a:
            continue
        name = (p.get("nom_voie") or "").upper()
        cls = (0 if re.fullmatch(r"V1\w*|UNIQUE|U|VU|1", name) else
               1 if re.fullmatch(r"V2\w*|2", name) else 2)
        recs[(p["code_ligne"], p["rg_troncon"])].append((cls, -(b - a), a, b, g["coordinates"]))
    out = defaultdict(list)
    for (code, rg), rs in recs.items():
        chosen = []
        for cls, _neg, a, b, coords in sorted(rs, key=lambda r: r[:2]):
            have = sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi, _c in chosen)
            if have < 0.5 * (b - a):
                chosen.append((a, b, coords))
        for a, b, coords in sorted(chosen, key=lambda c: c[0]):
            out[code].append((rg, a, b, coords))
    log(f"FR: {VOIES_FILE}: main tracks for {len(out)} line codes")
    return out


def track_between(tracks, rg, a, b, ends=None):
    """The main track of one tronçon between PK a and b, as parts in PK order, each cut by PK
    in proportion to its length; where `ends` (the portion's own first and last point) lies
    within JOIN_M * 4 of the track, it is cut there instead, so the portion still meets its
    neighbours (LGV Est's track is 950 m short of its PK range, which put Baudrecourt 260 m
    off). None when the tracks cover under 90% of a..b."""
    pieces, got = [], 0.0
    for trg, ta, tb, coords in tracks:
        lo, hi = max(a, ta), min(b, tb)
        if trg != rg or hi <= lo:
            continue
        poly = Poly(coords)
        m = lambda pk: (pk - ta) / (tb - ta) * poly.length
        pieces.append([poly, m(lo), m(hi)])
        got += hi - lo
    if not pieces or got < 0.9 * (b - a):
        return None
    tie = [None, None]
    if ends:
        for j, (piece, pt, k) in enumerate(((pieces[0], ends[0], 1), (pieces[-1], ends[1], 2))):
            mm, off = piece[0].project(*pt)
            if off <= JOIN_M * 4:
                piece[k] = mm
                tie[j] = tuple(pt)
    out = [poly.cut(m1, m2) for poly, m1, m2 in pieces if m2 > m1]
    # ... and starts and ends exactly where the portion did, so its joints stay joints.
    if out and tie[0]:
        out[0] = [tie[0]] + out[0]
    if out and tie[1]:
        out[-1] = out[-1] + [tie[1]]
    return out


def line_names(raw, region, log):
    """code -> (name, English name). Wikidata first, then OSM's infrastructure relations."""
    names = {}
    wd = raw / WIKIDATA
    if wd.exists():
        rows = json.loads(wd.read_text(encoding="utf-8"))
        # One item per code where there are several (752000 is both "ligne de Combs-la-Ville à
        # Saint-Louis (LGV)" and "LGV Sud-Est"): the official "ligne de A à B" first, then an
        # LGV name, then the rest ("ligne 167" is a Belgian number); among those, the LONGEST
        # item, because a code shared by the whole line and a historic piece of it names the
        # whole line (750000 is Moret - Lyon-Perrache, 490 km, not the 58 km "ligne de
        # Saint-Étienne à Lyon"; 905000 is Lyon - Marseille via Grenoble, not "Lyon -
        # Grenoble"); then the shortest item id, so that builds agree.
        def rank(r):
            lab = (r.get("label") or "").lower()
            try:
                km = float(r.get("len") or 0)
            except ValueError:
                km = 0.0
            return (0 if re.match(r"ligne (de |d'|des |du )", lab) else
                    1 if lab.startswith("lgv") else 2, -km, len(r["item"]), r["item"])
        for r in sorted(rows, key=rank):
            digits = re.sub(r"\D", "", r.get("num", ""))
            if len(digits) not in (5, 6) or not r.get("label"):
                continue
            code = digits.zfill(6)
            if code in names:
                continue
            lab = r["label"]
            names[code] = (lab[:1].upper() + lab[1:], english_name(r.get("enlabel"), lab))
    n_wd = len(names)
    infra = ROOT / "data" / "proc" / region / "infra.pkl"
    n_osm = 0
    if infra.exists():
        with open(infra, "rb") as fh:
            rels = pickle.load(fh)
        cand = defaultdict(list)
        for tags, _members in rels.values():
            digits = re.sub(r"\D", "", tags.get("ref") or "")
            name = (tags.get("name") or "").strip()
            if len(digits) not in (4, 5, 6) or not name:
                continue
            # "Voie 1 de la ligne de Paris-Est à Mulhouse-Ville": one track of the line.
            name = re.sub(r"^Voie \w+ de (la )?", "", name)
            cand[digits.zfill(6)].append(name[:1].upper() + name[1:])
        for code, ns in cand.items():
            if code not in names:
                best = sorted(ns, key=lambda n: (not n.lower().startswith("ligne"), len(n)))[0]
                names[code] = (best, "")
                n_osm += 1
    log(f"FR: line names for {n_wd} codes from Wikidata, {n_osm} more from OSM relations")
    return names


EN_WORDS = {"railway", "line", "lines", "from", "to", "the", "of", "via", "and", "junction",
            "high", "speed", "railroad", "branch"}


def english_name(en, fr):
    """Wikidata's English label, where it is an English name for the same places. Some are the
    French label again ("Ligne de Gretz-Armainvilliers à Sézanne") and some machine-translated
    the place names ("line from Choisy-le-Roi to Massy - Canopies", for Verrières); those
    are dropped, and the app shows the French name, which is the line's real name anyway."""
    if not en or en == fr or fold(en).startswith("ligne"):
        return ""
    words = set(re.split(r"[^a-z0-9]+", fold(en))) - {""}
    fr_words = set(re.split(r"[^a-z0-9]+", fold(fr))) - {""}
    return en if words - EN_WORDS <= fr_words else ""


def osm_stations(region):
    """OSM's rail stations: (lon, lat, name, name:en, word set)."""
    f = ROOT / "data" / "proc" / region / "stops.pkl"
    if not f.exists():
        return []
    with open(f, "rb") as fh:
        stops = pickle.load(fh)
    out = []
    for nid, (tags, lon, lat) in stops.items():
        name = tags.get("name")
        if not name:
            continue
        rw = tags.get("railway")
        rail = rw in ("station", "halt") or (
            tags.get("public_transport") == "station"
            and (rw or tags.get("train") == "yes" or tags.get("light_rail") == "yes"))
        if not rail or rw == "tram_stop":
            continue
        # A metro station is never an SNCF station, even where they share a name.
        if (tags.get("station") == "subway" or tags.get("subway") == "yes") \
                and tags.get("train") != "yes":
            continue
        out.append((lon, lat, name, tags.get("name:en") or "", name_words(name), nid))
    return out


def coverage(region):
    """The set of COVER_CELL grid cells the extract has any node in, grown by one cell."""
    f = ROOT / "data" / "proc" / region / "coords.npz"
    if not f.exists():
        return None
    c = np.load(f)
    cells = set(zip((c["x"] / 1e7 // COVER_CELL).astype(int).tolist(),
                    (c["y"] / 1e7 // COVER_CELL).astype(int).tolist()))
    grown = set()
    for x, y in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grown.add((x + dx, y + dy))
    return grown


def covered(pts, cells):
    if cells is None:
        return True
    ok = sum(1 for x, y in pts if (int(x // COVER_CELL), int(y // COVER_CELL)) in cells)
    return ok >= COVER_SHARE * len(pts)


# ---------------------------------------------------------------- geometry of a line

def chain_parts(parts):
    """Order a portion's exploded parts into runs, end to end, in the direction of its PK.

    The parts come in no particular order (Fives-Abbeville, 289000, arrives as its middle,
    then its start, then the piece between), but each runs in the PK direction, so the run
    starts at the part whose start no other part ends at. Returns a list of runs, each a list
    of (lon, lat); more than one where the geometry has a gap. A part that starts where it
    ends (a spiral tunnel) is walked the moment the run reaches its point."""
    left = [list(map(tuple, p)) for p in parts if len(p) >= 2]
    runs = []
    cur = None
    while left:
        if cur is None:
            def lead(k):
                """How far this part's start is from every other part's end."""
                p = left[k]
                if dist_m(*p[0], *p[-1]) <= JOIN_M:
                    return -1.0                      # a loop never starts a run
                return min((dist_m(*q[-1], *p[0]) for j, q in enumerate(left) if j != k),
                           default=math.inf)
            # A head is a part nothing leads into; after a gap, the one furthest from any
            # other part's end is where the geometry resumes.
            i = max(range(len(left)), key=lead)
            cur = list(left.pop(i))
            runs.append(cur)
            continue
        end = cur[-1]
        best = None
        for k, p in enumerate(left):
            for rev in (False, True):
                q = p[::-1] if rev else p
                d = dist_m(*q[0], *end)
                if d <= JOIN_M:
                    loop = dist_m(*q[-1], *q[0]) <= JOIN_M
                    key = (not loop, d)
                    if best is None or key < best[0]:
                        best = (key, k, q)
        if best is None:
            cur = None
            continue
        _key, k, q = best
        left.pop(k)
        cur.extend(q[1:])
    return runs


def country_outline(region):
    """The country's outline from dist/regions.json (lon/lat, prepared), or None before the
    country's first build."""
    try:
        reg = json.loads((ROOT / "dist" / "regions.json").read_text(
            encoding="utf-8"))["regions"][region]
    except (OSError, ValueError, KeyError):
        return None
    import shapely
    from shapely.geometry import Polygon
    polys = []
    for ring in reg.get("parts") or []:
        r = ring[0] if ring and isinstance(ring[0][0], list) else ring
        if len(r) >= 3:
            polys.append(Polygon(r).buffer(0))
    if not polys:
        return None
    g = shapely.union_all(polys)
    shapely.prepare(g)
    return g


def snap_borders(code, runs, node_of, deg, bidx, border_pts, outline=None):
    """End a line at the national border where it reaches one: at a dead end of a run, the
    border point (borders.py) on its track within BORDER_REACH_M, with no station between,
    cuts the run there (what lies past it is the neighbour's: 677 000 ran 860 m into Portbou,
    890 000 15 km into Geneva); one the run stops short of by up to BORDER_GAP_M, straight on
    ahead, is drawn on to (Jeumont, 89 m). Not further: Modane's line stops 430 m short, in
    the Fréjus tunnel, which OSM maps as long ways the register line cannot own, so a straight
    line drawn on to the point was a stretch nothing could credit. The end node becomes the
    point's own id, the one every country's build and every RINF register gives it, so the
    neighbour's half of the line joins this one there. Mutates runs, node_of and deg; returns
    [(point id, what was done)]."""
    from shapely.geometry import Point
    done, used = [], set()
    for i, r in enumerate(runs):
        for e in (0, 1):
            node = node_of[(i, e)]
            if deg.get(node, 0) != 1 or node in border_pts:
                continue
            poly = r["poly"]
            L = poly.length
            end_m = 0.0 if e == 0 else L
            end_pt = poly.point(end_m)
            best = None
            reach = BORDER_FAR_M if outline is not None else BORDER_REACH_M
            for p in bidx.pts:
                if p["id"] in used or dist_m(*end_pt, p["lon"], p["lat"]) > reach:
                    continue
                m, off = poly.project(p["lon"], p["lat"])
                along = abs(m - end_m)
                if along > 1.0:
                    # On the track, no station in between, and nearer this end than the
                    # other where that is a dead end too (202 100 is 650 m long, with a
                    # border point 16 m from one end).
                    other = node_of[(i, 1 - e)]
                    if off > BORDER_ON_M or (deg.get(other, 0) == 1
                                             and along >= abs(m - (L - end_m))):
                        continue
                    lo, hi = sorted((m, end_m))
                    if any(lo < em < hi for em, _s, _pk in r["events"]):
                        continue
                    # Further than BORDER_REACH_M only when the stretch cut off lies abroad.
                    if along > BORDER_REACH_M and (along > BORDER_FAR_M or outline.contains(
                            Point(*poly.point((m + end_m) / 2)))):
                        continue
                    cand = (along, "cut", p, m)
                else:
                    # Ahead of the end: within reach, and straight on (or practically there).
                    if off > BORDER_GAP_M:
                        continue
                    if off > 30:
                        back = poly.point(min(L, 200.0) if e == 0 else max(0.0, L - 200.0))
                        k = math.cos(math.radians(end_pt[1]))
                        ux, uy = (end_pt[0] - back[0]) * k, end_pt[1] - back[1]
                        vx, vy = (p["lon"] - end_pt[0]) * k, p["lat"] - end_pt[1]
                        nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
                        if not nu or not nv or (ux * vx + uy * vy) / (nu * nv) < 0.7:
                            continue
                    cand = (off, "extend", p, m)
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is None:
                continue
            dist, how, p, m = best
            pt = (p["lon"], p["lat"])
            anchors = [a for a in r["anchors"] if a[1] is not None]

            def pk_at(x):
                if len(anchors) < 2:
                    return None
                a = sorted(anchors)
                return float(np.interp(x, [q[0] for q in a], [q[1] for q in a]))
            if how == "cut" and min(m, L - m) <= 1.0:
                # The whole run lies abroad (890 000's last portion runs from the border to
                # Geneva): drop it, and the joint at its other end is the border point.
                joint = node_of[(i, 1 - e)]
                for key, n in list(node_of.items()):
                    if n == joint:
                        node_of[key] = p["id"]
                node_of[(i, 0)], node_of[(i, 1)] = f"x{i}a", f"x{i}b"
                r["drop"] = True
                deg[p["id"]] = deg.get(joint, 1) - 1
                border_pts[p["id"]] = p
                used.add(p["id"])
                done.append((p["id"], f"cut {dist:.0f} m short of its end (a run wholly abroad "
                                      f"dropped), at"))
                break
            if how == "cut":
                pk = pk_at(m)
                # The track ends at the foot of the point, as border_tails cuts OSM's.
                if e == 1:
                    r["poly"] = Poly(poly.cut(0.0, m))
                    r["anchors"] = [a for a in r["anchors"] if a[0] < m] + [
                        (r["poly"].length, pk)]
                    r["events"] = [ev for ev in r["events"] if ev[0] <= m]
                else:
                    r["poly"] = Poly(poly.cut(m, L))
                    r["anchors"] = [(0.0, pk)] + [(a[0] - m, a[1]) for a in r["anchors"]
                                                  if a[0] > m]
                    r["events"] = [(em - m, s, k) for em, s, k in r["events"] if em >= m]
                how = f"cut {dist:.0f} m short of its end, at"
            else:
                pts = [tuple(q) for q in poly.p]
                pa, pb = (r["anchors"][0][1], r["anchors"][-1][1])
                sign = (1 if pb >= pa else -1) if pa is not None and pb is not None else 0
                if e == 1:
                    r["poly"] = Poly(pts + [pt])
                    grow = r["poly"].length - L
                    r["anchors"] = r["anchors"] + [
                        (r["poly"].length, pb + sign * grow / 1000 if sign else None)]
                else:
                    r["poly"] = Poly([pt] + pts)
                    grow = r["poly"].length - L
                    r["anchors"] = [(0.0, pa - sign * grow / 1000 if sign else None)] + [
                        (a[0] + grow, a[1]) for a in r["anchors"]]
                    r["events"] = [(em + grow, s, k) for em, s, k in r["events"]]
                how = f"drawn on {grow:.0f} m to"
            node_of[(i, e)] = p["id"]
            deg[p["id"]] = 1
            border_pts[p["id"]] = p
            used.add(p["id"])
            done.append((p["id"], how))
    return done


def build(path, log=log_default):
    raw = Path(path)
    region = raw.name
    statut = features(raw, "lignes-par-statut")
    types = {f["properties"]["code_ligne"]: f["properties"]["type_ligne"]
             for f in features(raw, "lignes-par-type")}
    cat = defaultdict(Counter)
    for f in features(raw, "lignes-lgv-et-par-ecartement"):
        p = f["properties"]
        a, b = pk_km(p.get("pkd")), pk_km(p.get("pkf"))
        if a is not None and b is not None:
            cat[p["code_ligne"]][p["catlig"]] += abs(b - a)
    gares = features(raw, "liste-des-gares")
    names = line_names(raw, region, log)

    # --- portions: the exploded parts of one (code, tronçon, pkd, pkf) put back together
    portions = defaultdict(list)
    for f in statut:
        p = f["properties"]
        if p["statut"] != EXPLOITED or not f.get("geometry"):
            continue
        key = (p["code_ligne"], p["rg_troncon"], p["pkd"], p["pkf"])
        portions[key].append((f["geometry"]["coordinates"],
                              (p["x_d_wgs84"], p["y_d_wgs84"])))
    # --- the per-track file: coarse portions take its shape, and the lines the line files lack
    tracks = main_tracks(raw, log)
    n_coarse = 0
    for key, parts in list(portions.items()):
        code, rg, pkd, pkf = key
        a, b = pk_km(pkd), pk_km(pkf)
        if code not in tracks or a is None or b is None:
            continue
        polys = [Poly(pp) for pp, _s in parts if len(pp) >= 2]
        nseg = sum(len(q.p) - 1 for q in polys)
        if not nseg or sum(q.length for q in polys) / nseg <= COARSE_M:
            continue
        runs0 = chain_parts([pp for pp, _s in parts])
        ends = (runs0[0][0], runs0[-1][-1]) if runs0 else None
        got = track_between(tracks[code], rg, min(a, b), max(a, b), ends)
        if got:
            portions[key] = [(pp, None) for pp in got]
            n_coarse += 1
            log(f"    coarse geometry of {code_ref(code)} PK {pkd}-{pkf} "
                f"({sum(q.length for q in polys) / 1000:.1f} km, a vertex every "
                f"{sum(q.length for q in polys) / nseg:.0f} m) replaced by its track")
    in_statut = {f["properties"]["code_ligne"] for f in statut}
    added = set()
    for code, ts in sorted(tracks.items()):
        if code in in_statut or types.get(code, "Ligne") != "Ligne":
            continue
        for rg, a, b, coords in ts:
            portions[(code, rg, pk_str(a), pk_str(b))].append((coords, None))
        added.add(code)
    if tracks:
        log(f"FR: {n_coarse} coarse portions took their track's shape; {len(added)} line codes "
            f"the line files lack, taken from the per-track file: "
            f"{', '.join(code_ref(c) for c in sorted(added))}")
    kept_types = Counter()
    by_code = defaultdict(list)
    for key, parts in portions.items():
        t = types.get(key[0], "Ligne")
        kept_types[t] += 1
        if t != "Ligne":
            continue
        by_code[key[0]].append((key, parts))
    log(f"FR: {len(portions)} exploited portions; kept {kept_types['Ligne']} of lines, left "
        f"out {sum(v for k, v in kept_types.items() if k != 'Ligne')} of curves, sidings and "
        f"port track ({dict(kept_types)})")

    # --- stations: passenger stations only, one record per UIC, each line's PK kept
    st_recs = defaultdict(list)                     # uic -> records
    for f in gares:
        p = f["properties"]
        if p.get("voyageurs") != "O":
            continue
        st_recs[p["code_uic"]].append(p)
    osm = osm_stations(region)
    opos = np.array([[o[0], o[1]] for o in osm]) if osm else np.zeros((0, 2))
    sid_of, stations = {}, {}
    by_osm = {}
    n_exact = n_subset = n_blind = 0
    for uic, recs in sorted(st_recs.items()):
        lon = float(np.mean([r["x_wgs84"] for r in recs]))
        lat = float(np.mean([r["y_wgs84"] for r in recs]))
        sname = recs[0]["libelle"]
        hit = None
        if len(opos):
            dd = np.hypot((opos[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                          (opos[:, 1] - lat) * 110570)
            w = name_words(sname)
            best = None
            for j in np.argsort(dd)[:40]:
                if dd[j] > OSM_NAME_M:
                    break
                ow = osm[j][4]
                if w and ow == w:
                    rank = 0
                elif w and ow and dd[j] <= OSM_SUBSET_M and (
                        ow < w or w < ow or len(w & ow) >= 0.75 * min(len(w), len(ow))):
                    # One inside the other, or nearly: "Roissy-Aéroport-Charles-de-Gaulle 1"
                    # is OSM's "Aéroport Charles de Gaulle 1 (Terminal 3)".
                    rank = 1
                elif dd[j] <= OSM_BLIND_M:
                    rank = 2
                else:
                    continue
                if best is None or (rank, dd[j]) < best[0]:
                    best = ((rank, dd[j]), j)
            if best is not None:
                hit = best[1]
                if best[0][0] == 0:
                    n_exact += 1
                elif best[0][0] == 1:
                    n_subset += 1
                else:
                    n_blind += 1
        if hit is not None and osm[hit][5] in by_osm:
            sid = by_osm[osm[hit][5]]                # one OSM station, one register station
        else:
            sid = f"fr{uic}"
            if hit is not None:
                o = osm[hit]
                stations[sid] = {"id": sid, "name": o[2], "name_en": o[3],
                                 "lon": o[0], "lat": o[1], "lines": set(), "_sncf": sname}
                by_osm[o[5]] = sid
            else:
                stations[sid] = {"id": sid, "name": pretty_station(sname), "name_en": "",
                                 "lon": lon, "lat": lat, "lines": set(), "_sncf": sname}
        sid_of[uic] = sid
    # One name, close together: one station. Paris-Gare-de-Lyon and its RER platforms
    # (Paris-Gare-de-Lyon-Souterrain) land on two OSM nodes both called "Gare de Lyon".
    by_name = defaultdict(list)
    for sid, s in stations.items():
        by_name[fold(s["name"]).strip()].append(sid)
    merged = {}
    for ids in by_name.values():
        ids.sort()
        for i, a in enumerate(ids):
            if a in merged:
                continue
            for b in ids[i + 1:]:
                if b not in merged and dist_m(stations[a]["lon"], stations[a]["lat"],
                                              stations[b]["lon"], stations[b]["lat"]) <= SAME_NAME_M:
                    merged[b] = a
    for b in merged:
        del stations[b]
    sid_of = {u: merged.get(s, s) for u, s in sid_of.items()}
    log(f"FR: {len(st_recs)} passenger stations in the list; matched to an OSM station by "
        f"the same name {n_exact}, a name inside the other {n_subset}, position alone "
        f"{n_blind}; {len(stations)} register stations after merging those on one OSM station")

    # Passenger stations by position, for line ends that stop at a station not on the line.
    all_pos = [(sid, s["lon"], s["lat"]) for sid, s in stations.items()]
    apos = np.array([[p[1], p[2]] for p in all_pos])

    # OSM stations SNCF does not list: the RATP-owned stations on SNCF track (Châtelet - Les
    # Halles on the RER D tunnel, 981000) and a few the list lacks (Paris-Montparnasse,
    # Versailles-Rive-Droite). Candidates for stops on a line; see `extra_stops`.
    ridden = ridden_test(region) if (ROOT / "data" / "proc" / region / "ways.pkl").exists() \
        else None
    free = [o for o in osm if o[5] not in by_osm]          # for line ends
    fpos = np.array([[o[0], o[1]] for o in free]) if free else np.zeros((0, 2))
    extra = []
    # Keyed by the name's words, so OSM's "Arcy sur Cure" and "Arcy-sur-Cure", or
    # "Marne-la-Vallée - Chessy" and "Marne-la-Vallée Chessy", 60 m apart, are one candidate.
    seen_names = defaultdict(list)
    skey = lambda n: name_words(n) or fold(n).strip()
    for s in stations.values():
        seen_names[skey(s["name"])].append((s["lon"], s["lat"]))
    for o in osm:
        if o[5] in by_osm:
            continue
        if len(apos) and np.min(np.hypot((apos[:, 0] - o[0]) * math.cos(math.radians(o[1]))
                                         * 111320, (apos[:, 1] - o[1]) * 110570)) <= EXTRA_APART_M:
            continue
        key = skey(o[2])
        if any(dist_m(o[0], o[1], x, y) <= SAME_NAME_M for x, y in seen_names[key]):
            continue
        # Another record of a listed station: "Gare d'Austerlitz" beside "Paris Austerlitz".
        near = np.nonzero(np.hypot((apos[:, 0] - o[0]) * math.cos(math.radians(o[1])) * 111320,
                                   (apos[:, 1] - o[1]) * 110570) <= OSM_NAME_M)[0] \
            if len(apos) else []
        if any(o[4] and (o[4] <= name_words(stations[all_pos[k][0]]["name"])
                         or name_words(stations[all_pos[k][0]]["name"]) <= o[4])
               for k in near):
            continue
        seen_names[key].append((o[0], o[1]))
        extra.append(o)
    epos = np.array([[o[0], o[1]] for o in extra]) if extra else np.zeros((0, 2))
    n_extra, not_added = 0, []

    # Stations per line: {code: [(sid, rg, pk, lon, lat)]}, one per (code, uic).
    on_line = defaultdict(dict)
    for uic, recs in st_recs.items():
        by_c = defaultdict(list)
        for r in recs:
            by_c[r["code_ligne"]].append(r)
        for code, rs in by_c.items():
            rs.sort(key=lambda r: pk_km(r["pk"]) or 0)
            r = rs[len(rs) // 2]                       # SNCF lists Martigues four times
            on_line[code][uic] = (sid_of[uic], r["rg_troncon"], pk_km(r["pk"]),
                                  r["x_wgs84"], r["y_wgs84"])

    def end_stop(pt, code):
        """The station a line's dead end at pt ends at, or None. A line that ends at a
        passenger station of another line ends there, or at an OSM station SNCF does not list:
        Paris-Montparnasse, whose main halls are not in the list (only Montparnasse 3 -
        Vaugirard, 400 m off)."""
        nonlocal n_extra
        k = math.cos(math.radians(pt[1])) * 111320
        if len(apos):
            dd = np.hypot((apos[:, 0] - pt[0]) * k, (apos[:, 1] - pt[1]) * 110570)
            j = int(np.argmin(dd))
            if dd[j] <= END_SNAP_M:
                return all_pos[j][0]
        if len(fpos):
            dd = np.hypot((fpos[:, 0] - pt[0]) * k, (fpos[:, 1] - pt[1]) * 110570)
            j = int(np.argmin(dd))
            if dd[j] <= END_SNAP_M:
                o = free[j]
                sid = f"fro{o[5]}"
                if sid not in stations:
                    stations[sid] = {"id": sid, "name": o[2], "name_en": o[3],
                                     "lon": o[0], "lat": o[1], "lines": set()}
                    n_extra += 1
                    log(f"    OSM station added to {code_ref(code)}, at its end: {o[2]}")
                return sid
        return None

    try:
        import borders
        bidx = borders.Index(borders.load())
    except (ImportError, OSError, ValueError) as err:
        log(f"FR: no border points ({err}); line ends at borders are left as they are")
        bidx = None
    border_pts, snaps = {}, []
    outline = country_outline(region) if bidx is not None else None

    cells = coverage(region)
    lines, geoms, n_off, n_pkbad, n_gap = [], {}, 0, 0, 0
    clipped_km, clipped_lines = 0.0, 0
    junctions = {}
    for code, plist in sorted(by_code.items()):
        # --- runs and their anchors
        runs = []          # (Poly, rg, [(m, pk)], start pt)
        for (c, rg, pkd, pkf), parts in plist:
            rs = chain_parts([pp for pp, _s in parts])
            if len(rs) > 1:
                n_gap += len(rs) - 1
            polys = [Poly(r) for r in rs if len(r) >= 2]
            polys = [p for p in polys if p.length > 0]
            if not polys:
                continue
            total = sum(p.length for p in polys)
            a, b = pk_km(pkd), pk_km(pkf)
            off = 0.0
            for p in polys:
                # PK at the run's ends, by its share of the portion; exact when unbroken.
                pa = a + (b - a) * off / total if a is not None and b is not None else None
                pb = a + (b - a) * (off + p.length) / total if pa is not None else None
                runs.append({"poly": p, "rg": rg, "anchors": [(0.0, pa), (p.length, pb)],
                             "events": []})
                off += p.length

        # --- stations onto runs
        for uic, (sid, rg, pk, slon, slat) in on_line.get(code, {}).items():
            best = None
            for r in runs:
                if r["rg"] != rg:
                    continue
                lo_hi = [x for _m, x in r["anchors"] if x is not None]
                if pk is not None and len(lo_hi) == 2 and not (
                        min(lo_hi) - PK_SLACK_KM <= pk <= max(lo_hi) + PK_SLACK_KM):
                    continue
                m, d = r["poly"].project(slon, slat)
                if d <= ON_LINE_M and (best is None or d < best[0]):
                    best = (d, r, m)
            if best is None:
                n_off += 1
                continue
            _d, r, m = best
            r["events"].append((m, sid, pk))
        # A line the line files lack has no stations in SNCF's list either: it takes the
        # passenger stations SNCF lists under other lines that lie on its track and that an
        # OSM route calls at (Somain on 262 000; not Roissy-CDGX 2 of the unopened CDG Express
        # beside 226 310).
        if code in added and len(apos) and ridden is not None:
            for r in runs:
                p = r["poly"].p
                pad = 0.002
                box = np.nonzero((apos[:, 0] >= p[:, 0].min() - pad)
                                 & (apos[:, 0] <= p[:, 0].max() + pad)
                                 & (apos[:, 1] >= p[:, 1].min() - pad)
                                 & (apos[:, 1] <= p[:, 1].max() + pad))[0]
                for j in box:
                    m, d = r["poly"].project(apos[j, 0], apos[j, 1])
                    sid = all_pos[j][0]
                    if (d <= BORROW_M and not any(e[1] == sid for e in r["events"])
                            and called_at(ridden.calls, stations[sid])):
                        r["events"].append((m, sid, None))

        # --- nodes: run ends (joined across runs within JOIN_M), and stations
        end_pts = []
        for i, r in enumerate(runs):
            end_pts.append((i, 0, r["poly"].point(0)))
            end_pts.append((i, 1, r["poly"].point(r["poly"].length)))
        node_of = {}
        reps = []
        for i, e, pt in end_pts:
            for k, q in enumerate(reps):
                if dist_m(*pt, *q) <= JOIN_M:
                    node_of[(i, e)] = f"e{k}"
                    break
            else:
                node_of[(i, e)] = f"e{len(reps)}"
                reps.append(pt)
        deg = Counter(node_of.values())
        # --- national borders: a dead end near a border point ends AT it (see the docstring)
        if bidx is not None:
            for bid, how in snap_borders(code, runs, node_of, deg, bidx, border_pts, outline):
                snaps.append(f"{code_ref(code)} {how} {bid} ({border_pts[bid]['name']})")

        def across(i, end):
            """The nearest station past run i's `end` (0 or 1), on the runs joined there."""
            node = node_of[(i, end)]
            for (k, e2), n2 in node_of.items():
                if n2 != node or k == i:
                    continue
                ev2 = sorted(runs[k]["events"])
                if ev2:
                    return ev2[0] if e2 == 0 else ev2[-1]
            return None

        def line_end(i, end):
            """The station run i's `end` stops at, when it is a dead end at a station."""
            node = node_of[(i, end)]
            if node in border_pts or deg.get(node, 0) != 1:
                return None
            sid = end_stop(reps[int(node[1:])], code)
            return None if sid is None else (0.0, sid, None)

        # --- OSM stations on the line that SNCF does not list: one within OSM_ON_LINE_M of
        # the track, which an OSM route calls at together with its neighbouring stations on
        # this line. The second test keeps out a station of another line that merely passes.
        if ridden is not None and len(epos):
            for i, r in enumerate(runs):
                ev = sorted(r["events"])
                if r.get("drop") or (not ev and code not in added):
                    continue
                p = r["poly"].p
                pad = 0.003
                box = ((epos[:, 0] >= p[:, 0].min() - pad) & (epos[:, 0] <= p[:, 0].max() + pad)
                       & (epos[:, 1] >= p[:, 1].min() - pad) & (epos[:, 1] <= p[:, 1].max() + pad))
                for j in np.nonzero(box)[0]:
                    o = extra[j]
                    m, d = r["poly"].project(o[0], o[1])
                    if d > OSM_ON_LINE_M:
                        continue
                    before = [e for e in ev if e[0] <= m]
                    after = [e for e in ev if e[0] > m]
                    # Where the run ends at a joint with another run of the line (the RER D
                    # tunnel is two runs meeting at Châtelet), the neighbour is on that run.
                    b = before[-1] if before else across(i, 0)
                    a = after[0] if after else across(i, 1)
                    # A line SNCF lists no stations for may have only one of its own: its
                    # station-ended dead end is the other neighbour (Arcachon on 657 000).
                    # It may also have no station at one end (657 000 leaves 655 000 at a
                    # junction): there the one neighbour it has must do.
                    if code in added:
                        b = b if b is not None else line_end(i, 0)
                        a = a if a is not None else line_end(i, 1)
                        nb = [x for x in (b, a) if x is not None]
                    elif b is None or a is None:
                        continue            # a line's end: see the dead-end snapping below
                    else:
                        nb = [b, a]
                    if not nb:
                        continue
                    # Both neighbours: T4 calls at Remise à Jorelle and at Bondy, but not at
                    # the next station along 070000; RER B calls at Les Baconnets and at
                    # Massy-Verrières, where 985000 ends.
                    cand = {"name": o[2], "lon": o[0], "lat": o[1]}
                    if not all(served_together(ridden.calls, cand, stations[e[1]])
                               for e in nb):
                        not_added.append(f"{code_ref(code)}: {o[2]} (between "
                                         f"{' and '.join(stations[e[1]]['name'] for e in nb)})")
                        continue
                    sid = f"fro{o[5]}"
                    if sid not in stations:
                        stations[sid] = {"id": sid, "name": o[2], "name_en": o[3],
                                         "lon": o[0], "lat": o[1], "lines": set()}
                        n_extra += 1
                        log(f"    OSM station added to {code_ref(code)}: {o[2]}")
                    if any(e[1] == sid for e in r["events"]):
                        continue
                    r["events"].append((m, sid, None))

        # Edges along each run between consecutive events.
        adj = defaultdict(list)
        edges = []
        for i, r in enumerate(runs):
            if r.get("drop"):
                continue
            poly = r["poly"]
            ev = sorted(r["events"])
            # PK anchors: the run's ends and every station whose PK sits in order.
            anchors = [a for a in r["anchors"] if a[1] is not None]
            for m, _sid, pk in ev:
                if pk is not None:
                    anchors.append((m, pk))
            anchors.sort()
            # A station whose PK is out of order with its neighbours is not an anchor.
            ok = []
            for j, (m, pk) in enumerate(anchors):
                prev = ok[-1][1] if ok else None
                nxt = anchors[j + 1][1] if j + 1 < len(anchors) else None
                if prev is not None and nxt is not None and not (
                        min(prev, nxt) - PK_SLACK_KM <= pk <= max(prev, nxt) + PK_SLACK_KM):
                    n_pkbad += 1
                    continue
                ok.append((m, pk))
            am = np.array([a[0] for a in ok]) if ok else None
            ap = np.array([a[1] for a in ok]) if ok else None

            def pk_at(m):
                if am is None or len(am) < 2:
                    return None
                return float(np.interp(m, am, ap))

            seq = [(0.0, node_of[(i, 0)])] + [(m, ("s", sid)) for m, sid, _pk in ev] \
                + [(poly.length, node_of[(i, 1)])]
            for (m1, u), (m2, v) in zip(seq[:-1], seq[1:]):
                if u == v or m2 - m1 <= 0:
                    continue
                p1, p2 = pk_at(m1), pk_at(m2)
                chain = abs(p2 - p1) if p1 is not None and p2 is not None else None
                e = {"u": u, "v": v, "km": (m2 - m1) / 1000.0, "chain": chain,
                     "pts": poly.cut(m1, m2)}
                edges.append(e)
                adj[u].append((v, e, False))
                adj[v].append((u, e, True))

        # --- section ends: stations, and dead ends that are not stops
        at = {}
        for n in adj:
            if isinstance(n, tuple):
                at[n] = n[1]
        for n in adj:
            if n in border_pts:          # a section end even where two runs meet there
                at[n] = n
                if n not in stations:
                    p = border_pts[n]
                    stations[n] = {"id": n, "name": p["name"], "name_en": "",
                                   "lon": p["lon"], "lat": p["lat"], "lines": set(),
                                   "junction": True}
                continue
            if isinstance(n, tuple) or deg.get(n, 0) != 1:
                continue
            pt = reps[int(n[1:])]
            sid = end_stop(pt, code)
            if sid is not None:
                at[n] = sid
                continue
            jid = f"fj{code}_{pt[0]:.4f}_{pt[1]:.4f}"
            at[n] = jid
            junctions[jid] = pt
        if len(set(at.values())) < 2:
            continue

        # --- absorbing Dijkstra from each section end, as in schienennetz.py
        sections = {}
        for src, sid in at.items():
            dist, prev, seen = {src: 0.0}, {}, set()
            heap = [(0.0, 0, src)]
            tie = 0
            while heap:
                d, _t, u = heapq.heappop(heap)
                if u in seen:
                    continue
                seen.add(u)
                other = at.get(u)
                if u != src and other is not None:
                    if other != sid:
                        key = (sid, other) if sid <= other else (other, sid)
                        if key not in sections or d < sections[key]["km"]:
                            path, cur = [], u
                            while cur != src:
                                pnode, e, rev = prev[cur]
                                path.append((e, rev))
                                cur = pnode
                            path.reverse()
                            pts = []
                            for e, rev in path:
                                ep = e["pts"][::-1] if rev else e["pts"]
                                pts.extend(ep if not pts else ep[1:])
                            if sid > other:
                                pts.reverse()
                            ch = [e["chain"] for e, _r in path]
                            sections[key] = {
                                "km": d, "pts": pts,
                                "chain": sum(ch) if all(c is not None for c in ch) else None}
                    continue
                for v, e, rev in adj[u]:
                    nd = d + e["km"]
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        prev[v] = (u, e, rev)
                        tie += 1
                        heapq.heappush(heap, (nd, tie, v))

        # Tail track: a short run from a stop to the buffer stops is not a section.
        for key in list(sections):
            a, b = key
            if (a in junctions) != (b in junctions) and sections[key]["km"] * 1000 < STUB_M:
                del sections[key]
        # Clip to the extract.
        for key in list(sections):
            if not covered(sections[key]["pts"], cells):
                clipped_km += sections[key]["km"]
                del sections[key]
        if not sections:
            clipped_lines += 1
            continue

        lid = line_id(code)
        name, name_en = names.get(code, (f"Ligne {code_ref(code)}", ""))
        c = cat.get(code, Counter())
        catlig = c.most_common(1)[0][0] if c else ""
        kind = "narrow_gauge" if "étroite" in catlig else "rail"
        lines.append({
            "id": lid, "src": "rfn", "service": False,
            "name": name, "name_en": name_en, "ref": code_ref(code), "colour": "",
            "operator": OPERATOR, "operator_en": "", "network": "",
            "kind": kind,
            "highspeed": catlig == "Ligne à grande vitesse",
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "km_official": round(sum(v["chain"] or v["km"] for v in sections.values()), 3),
            "chain": {f"{a}|{b}": round(v["chain"] if v["chain"] is not None else v["km"], 3)
                      for (a, b), v in sections.items()},
            "variants": len(plist), "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        })
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["pts"]]
                      for (a, b), v in sections.items()}

    log(f"FR: {n_extra} OSM stations SNCF does not list became stops on a line; "
        f"{len(not_added)} beside a line were not, no OSM route calling at them and at both "
        f"their neighbours on it:")
    for s in not_added[:40]:
        log(f"    {s}")
    log(f"FR: {n_off} station-line pairs found no exploited track of their line within "
        f"{ON_LINE_M} m (closed portions, curves), {n_pkbad} station PKs out of order, "
        f"{n_gap} gaps inside a portion's geometry")
    if cells is not None:
        log(f"FR: clipped to the extract: {clipped_km:,.0f} km of sections outside it, "
            f"{clipped_lines} lines with nothing inside")
    log(f"FR: {len(snaps)} line ends at a national border now end at its border point:")
    for s in snaps:
        log(f"    {s}")

    # --- junctions, named after the nearest point SNCF lists
    gpos = np.array([[f["properties"]["x_wgs84"], f["properties"]["y_wgs84"]] for f in gares])
    for jid, (lon, lat) in junctions.items():
        dd = np.hypot((gpos[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                      (gpos[:, 1] - lat) * 110570)
        j = int(np.argmin(dd))
        code = jid[2:8]
        name = (f"Bif. {pretty_station(gares[j]['properties']['libelle'])}"
                if dd[j] <= JUNCTION_NAME_M else f"Ligne {code_ref(code)}")
        stations[jid] = {"id": jid, "name": name, "name_en": "", "lon": lon, "lat": lat,
                         "lines": set(), "junction": True}

    # --- stop-to-stop sections that no passenger train runs over (see the docstring)
    lines = question_stop_sections(ridden, lines, stations, geoms, log)

    used = {s for l in lines for sec in l["sections"] for s in sec[:2]}
    stations = {k: v for k, v in stations.items() if k in used}
    for s in stations.values():
        s.pop("_sncf", None)
        s["lines"] = set()
    for l in lines:
        for sec in l["sections"]:
            for s in sec[:2]:
                stations[s]["lines"].add(l["id"])
    n_j = sum(1 for s in stations.values() if s.get("junction"))
    total = sum(l["km"] for l in lines)
    log(f"FR: {len(lines)} register lines, {total:,.0f} km, {len(stations)} stations of which "
        f"{n_j} are line ends that are not stops; {sum(1 for l in lines if l.get('highspeed'))} "
        f"LGV")
    return lines, stations, geoms


def route_ways(region):
    """OSM ways a passenger route runs over, in Lambert-93 metres, with the route kinds."""
    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as fh:
        ways = pickle.load(fh)
    with open(d / "rels.pkl", "rb") as fh:
        rels = pickle.load(fh)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    with open(d / "stops.pkl", "rb") as fh:
        stops = pickle.load(fh)
    kinds = defaultdict(set)
    calls = []                  # per route, the word sets of the stops it calls at
    for tags, members in rels.values():
        if tags.get("type") != "route" or tags.get("route") not in RIDE_KINDS:
            continue
        ws = []
        for ty, ref, role in members:
            if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                kinds[ref].add(tags["route"])
            elif ty == "n" and role.startswith("stop") and ref in stops:
                stags, lon, lat = stops[ref]
                w = name_words(stags.get("name"))
                if w:
                    ws.append((w, lon, lat))
        calls.append(ws)
    from pyproj import Transformer
    to_l93 = Transformer.from_crs(4326, 2154, always_xy=True)
    def xy(wid):
        nodes = ways[wid][1]
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            return None
        x, y = to_l93.transform(cx[pos] / 1e7, cy[pos] / 1e7)
        return np.column_stack([x, y])

    geo, knd = [], []
    for wid, ks in kinds.items():
        g = xy(wid) if wid in ways else None
        if g is not None:
            geo.append(g)
            # What the TRACK is, not which routes run on it (question_stop_sections).
            knd.append("light" if ways[wid][0].get("railway") in ("light_rail", "tram")
                       else "train")
    hs = [g for g in (xy(w) for w, (t, _n) in ways.items() if t.get("highspeed") == "yes")
          if g is not None]
    return geo, knd, to_l93, calls, hs


def served_together(calls, a, b):
    """Does some OSM route call at both stations? A stop is a station's when it is within
    CALL_M of it and one name's words are inside the other's ("Paris Gare du Nord – Voie 44"
    is Paris Gare du Nord). Both, because RER B's "Antony" is in "Chemin d'Antony" by name."""
    wa, wb = name_words(a["name"]), name_words(b["name"])
    if not wa or not wb:
        return False

    def at(ws, w, s):
        return any((x == w or x < w or w < x)
                   and dist_m(lon, lat, s["lon"], s["lat"]) <= CALL_M for x, lon, lat in ws)
    return any(at(ws, wa, a) and at(ws, wb, b) for ws in calls)


def called_at(calls, s):
    """Does some OSM route call at this station (by served_together's test)?"""
    w = name_words(s["name"])
    return bool(w) and any(
        any((x == w or x < w or w < x) and dist_m(lon, lat, s["lon"], s["lat"]) <= CALL_M
            for x, lon, lat in ws) for ws in calls)


def question_stop_sections(ridden, lines, stations, geoms, log):
    """Ask OSM which register sections passenger trains run over.

    A point of a section is ridden when a way some OSM passenger route (train, tram, light
    rail) runs over lies within RIDE_M of it; any kind counts, since the tram-trains (T11 on
    960000, T13 on the Grande Ceinture) are route=tram on light_rail track. Two uses:

    - a STOP-TO-STOP section is dropped unless STOP_SURE of it is ridden, or STOP_SHARE is and
      some OSM route calls at both its stations: a freight line crossing two passenger
      stations (see the module docstring) is ridden only near its ends, and the Grande
      Ceinture from Athis-Mons to Les Saules is 54% "ridden" by the RER C track beside it,
      though no route calls at both. Junction-ended sections are left to build_model's own
      test (drop_unridden_sections).
    - a line most of whose ridden length is light rail TRACK (railway=light_rail or tram, as
      T11's on 960 000) takes kind light_rail, so that build_model pairs it with that track
      and not with the freight tracks of the Grande Ceinture beside it. The track's tag, not
      the route's: Nantes - Châteaubriant (519 000) and Lyon-Saint-Paul - Montbrison
      (782 000) run tram-trains on railway=rail track, and as light_rail they owned none of
      it (2.6% and 0% creditable until 2026-10-02)."""
    if ridden is None:
        log("FR: no OSM extract; stop-to-stop sections not questioned")
        return lines
    n_drop, km_drop, out, hit, rekinded = 0, 0.0, [], [], []
    hs_open = []
    junction = {s for s, v in stations.items() if v.get("junction")}
    for l in lines:
        keep = []
        ride_km = light_km = 0.0
        hs_flags = {}
        for sec in l["sections"]:
            a, b, km = sec
            share, light, hs = ridden(geoms[l["id"]][f"{a}|{b}"])
            ride_km += share * km
            light_km += share * light * km
            # Not high-speed, unless OSM tags this classic line's track highspeed=yes (a
            # 200-220 km/h stretch): then left unknown, so the trains OSM puts on it credit it.
            if hs < HS_SHARE:
                hs_flags[f"{a}|{b}"] = False
            elif not l["highspeed"]:
                hs_open.append((l["ref"], stations[a]["name"], stations[b]["name"]))
            if (a in junction or b in junction or share >= STOP_SURE
                    or (share >= STOP_SHARE and served_together(
                        ridden.calls, stations[a], stations[b]))):
                keep.append(sec)
            else:
                n_drop += 1
                km_drop += km
                hit.append((l["ref"], l["name"], stations[a]["name"], stations[b]["name"], km,
                            share))
                geoms[l["id"]].pop(f"{a}|{b}", None)
        if not keep:
            geoms.pop(l["id"], None)
            continue
        if len(keep) != len(l["sections"]):
            l["sections"] = keep
            l["km"] = round(sum(s[2] for s in keep), 3)
            l["chain"] = {k: v for k, v in l["chain"].items()
                          if k in {f"{s[0]}|{s[1]}" for s in keep}}
            l["km_official"] = round(sum(l["chain"].values()), 3)
            ends = {s for sec in keep for s in sec[:2]}
            l["display"] = [s for s in l["display"] if s in ends]
        if l["kind"] == "rail" and light_km > 0.5 * ride_km:
            l["kind"] = "light_rail"
            rekinded.append(f"{l['ref']} {l['name']}")
        if not l["highspeed"]:
            del l["highspeed"]
            kept = {f"{s[0]}|{s[1]}" for s in keep}
            l["highspeed_sections"] = {k: v for k, v in hs_flags.items() if k in kept}
        out.append(l)
    log(f"FR: dropped {n_drop} stop-to-stop sections ({km_drop:,.0f} km) no OSM passenger "
        f"route runs over; {len(lines) - len(out)} lines left with nothing")
    for ref, name, a, b, km, share in sorted(hit)[:80]:
        log(f"    {ref}  {a} - {b}  {km:.1f} km, {share:.0%} ridden  ({name})")
    log(f"FR: {len(rekinded)} lines mostly on light rail track, now light_rail: "
        f"{'; '.join(rekinded)}")
    log(f"FR: {len(hs_open)} sections of classic lines lie on OSM highspeed=yes track and are "
        f"left unflagged: {'; '.join(f'{r} {a} - {b}' for r, a, b in hs_open[:30])}")
    return out


def ridden_test(region):
    """A function of a section's points: (share of it ridden, share of that on tram-train,
    share of it beside OSM highspeed=yes track)."""
    from shapely import STRtree, line_interpolate_point
    from shapely.geometry import LineString
    geo, knd, to_l93, calls, hs = route_ways(region)
    tree = STRtree([LineString(g) for g in geo])
    hs_tree = STRtree([LineString(g) for g in hs]) if hs else None
    knd = np.array(knd)

    def ridden(pts):
        a = np.asarray(pts, dtype=np.float64)
        x, y = to_l93.transform(a[:, 0], a[:, 1])
        line = LineString(np.column_stack([x, y]))
        if line.length <= 0:
            return 0.0, 0.0, 0.0
        n = max(2, int(line.length / RIDE_STEP_M))
        probe = line_interpolate_point(line, (np.arange(n) + 0.5) / n, normalized=True)
        pi, wi = tree.query(probe, predicate="dwithin", distance=RIDE_M)
        hit_train, hit_any = set(), set()
        for p, w in zip(pi.tolist(), wi.tolist()):
            hit_any.add(p)
            if knd[w] == "train":
                hit_train.add(p)
        light = len(hit_any - hit_train)
        hs_share = 0.0
        if hs_tree is not None:
            hp, _hw = hs_tree.query(probe, predicate="dwithin", distance=HS_M)
            hs_share = len(set(hp.tolist())) / n
        return (len(hit_any) / n, (light / len(hit_any) if hit_any else 0.0), hs_share)

    ridden.calls = calls
    return ridden


def walk_order(keys):
    from n02 import walk_order as w
    return w(keys)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--fetch" in sys.argv:
        fetch(ROOT / "data" / "raw" / "fr")
    else:
        print(__doc__)
