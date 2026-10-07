"""Denmark: RINF has every Banedanmark line plus Lokaltog's and Nordjyske Jernbaner's, but one id
per SECTION (340 ids, "016080" Sorø - Slagelse), so the lines are put together here. dk_sources.md
has the sources, the checks and what is still off.

THE IDS.  A six-digit id is Banedanmark's line ("strækning") in its first two digits and a
sub-line in the third, then a kilometre: 012/014/016 are Vestbanen (København H - Roskilde -
Ringsted - Korsør - Nyborg), 222/224 Den fynske hovedbane, 8xx the S-bane, 6xx the private
lines. A twelve-digit id is two six-digit ones joined: a link inside a station (København H's
S-bane and main-line points) or a spur to a freight terminal. So `fix` files each section under a
line by the first three digits of its id (PREFIX), with ID for the few sub-lines that hold two
named lines (244: Langå - Randers is Aarhus-Randers-banen, Randers - Hobro Randers-Aalborg
Jernbane; 264: Lunderskov - Vamdrup and Vamdrup - Vojens; 016: Korsør - Nyborg is the Great
Belt link). Twelve-digit ids are left out unless ID names them (Padborg - the border point,
Hundested - Hundested Havn). A line's name is its da.wikipedia article's title.

LENGTHS.  RINF's Danish section lengths leave out the station areas: Sorø - Slagelse is 12.3 km
for 13.7 km as the crow flies, København H - Korsør sums to 78 km against 111, and short
sections in Copenhagen are a third of their track. So `km_floor` (rinf.py): a trace is judged
against that floor and the crow-fly distance, and no km_official is shipped.

TRACK.  OSM Denmark maps the S-bane as railway=light_rail (`light_rail_track`, as Germany's
S-Bahn), and names each track way for its line ("Vestbanen" ref 1 beside "Høje Taastrup-banen"
ref 810 between Valby and Høje Taastrup): `way_line` gives those ways to their line, so pass 2
keeps a main line off the S-bane track beside it.

STATIONS.  RINF lists Banedanmark's stations, not every halt (Jægersborg, Bernstorffsvej, Viby
J, Støvring are not in it): `osm_stops`, with the S-tog's route=light_rail relations
(`osm_stop_route`) and the timetable's rail stations no OSM route lists (`osm_stop_extra`:
Brejning, Gelsted, Jerne...). Four names are written out in RINF and abbreviated in the
timetable (Kirke/Kr., Store/St., Nørre/Nr.); `fix` adds the short form.

THE ØRESUND BRIDGE.  RINF's border point EU00141 ("Peberholm grænse") is the managers' boundary
at Peberholm's west end, 5.4 km inside Denmark; the national border is on the bridge at
12.80896, 55.57924, where the Swedish extract has track and which Anita's rule makes the cut
("border track counts in the country it lies in"). `fix` moves the point there and adds the
5.2 km of track to Københavns Lufthavn - Peberholm grænse (a floor: da.wikipedia gives about
5.5). borders.py and Sweden's se.py need the same move (dk_sources.md has both diffs).

GEDSERBANEN (Nykøbing Falster - Gedser) has had no scheduled train since 2009, only museum runs
(da.wikipedia, "Sydbanen"): drawn greyed (`suspended`), as Finland's Porvoo is.
"""
import csv
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

KKR = "København-Køge-Ringsted-banen"
FV = "Fredericia-Vamdrup-banen"
VP = "Vamdrup-Padborg-banen"

# Banedanmark's line and sub-line (the id's first three digits) -> the named line.
PREFIX = {
    "012": "Vestbanen", "014": "Vestbanen", "016": "Vestbanen",
    # the new line from Vigerslev, and its way in from København H past the goods yard (070,
    # 072: København H - København G - Vigerslev, and the curve Vigerslev - Hvidovre Fjern
    # that RE 50 takes from København Syd to Høje Taastrup)
    "020": KKR, "021": KKR, "070": KKR, "072": KKR,
    "022": "Sydbanen", "024": "Sydbanen", "026": "Sydbanen", "028": "Sydbanen",
    "032": "Gedserbanen",
    "042": "Lille Syd",
    "052": "Nordvestbanen",
    "092": "Lille Nord",
    "102": "Kystbanen", "104": "Kystbanen", "106": "Kystbanen",
    # 129 Vigerslev - Kalvebod: RE 50's way from the airport to København Syd
    "122": "Øresundsbanen", "124": "Øresundsbanen", "126": "Øresundsbanen",
    "129": "Øresundsbanen",
    "212": "Svendborgbanen",
    "222": "Den fynske hovedbane", "224": "Den fynske hovedbane",
    "232": "Fredericia-Aarhus-banen", "234": "Fredericia-Aarhus-banen",
    "236": "Fredericia-Aarhus-banen",
    "242": "Aarhus-Randers-banen", "244": "Aarhus-Randers-banen",
    "246": "Randers-Aalborg Jernbane",
    "252": "Vendsysselbanen", "254": "Vendsysselbanen",
    "262": FV, "264": FV, "266": VP, "268": VP,
    "271": "Snoghøj-Taulov-banen",
    "282": "Sønderborgbanen",
    "292": "Lunderskov-Esbjerg-banen", "294": "Lunderskov-Esbjerg-banen",
    "302": "Bramming-Tønder-banen",
    "312": "Den vestjyske længdebane", "314": "Den vestjyske længdebane",
    "316": "Den vestjyske længdebane",
    "322": "Langå-Struer-banen",
    "332": "Vejle-Holstebro-banen", "334": "Vejle-Holstebro-banen",
    "342": "Thybanen",
    "352": "Skanderborg-Skjern-banen", "354": "Skanderborg-Skjern-banen",
    # Lindholm - Aalborg Lufthavn (2020; RE 69). No article of its own: the name is ours.
    "521": "Lindholm-Aalborg Lufthavn",
    "602": "Nærumbanen", "604": "Frederiksværkbanen", "605": "Hornbækbanen",
    "606": "Gribskovbanen", "608": "Gribskovbanen",
    "609": "Odsherredsbanen",
    "615": "Tølløsebanen", "616": "Tølløsebanen",
    "618": "Østbanen", "619": "Østbanen",
    "632": "Lollandsbanen",
    "662": "Varde-Nørre Nebel Jernbane", "668": "Hirtshalsbanen", "669": "Skagensbanen",
    # S-bane. København H - Østerport - Hellerup is Nordbanen's (da.wikipedia: København -
    # Hillerød, 36.5 km); the main-line tracks beside it are Kystbanen's (102).
    "812": "Høje Taastrup-banen", "814": "Høje Taastrup-banen", "816": "Høje Taastrup-banen",
    "818": "Høje Taastrup-banen",
    "822": "Nordbanen", "824": "Nordbanen", "826": "Nordbanen",
    "832": "Frederikssundbanen", "834": "Frederikssundbanen", "836": "Frederikssundbanen",
    "842": "Hareskovbanen",
    "852": "Køge Bugt-banen", "854": "Køge Bugt-banen",
    "862": "Klampenborgbanen",
    "882": "Ringbanen", "884": "Ringbanen",
    # 076 Lersøen - Svanemøllen - Østerport: a main-line connection between junctions, no
    # article; named "first - last" by rinf.py and kept only where trains run.
    "076": None,
}
# Whole ids filed apart from their sub-line, and the twelve-digit links that are kept.
ID = {
    "016114": "Storebæltsforbindelsen", "016124": "Storebæltsforbindelsen",
    "244170": "Randers-Aalborg Jernbane", "244184": "Randers-Aalborg Jernbane",
    "264040": VP, "264046": VP, "264054": VP,
    "269000269700": VP,                          # Padborg - Padborg grænse (EU00059)
    "604039604041": "Frederiksværkbanen",        # Hundested - Hundested Havn
}
NAMES = {n for n in list(PREFIX.values()) + list(ID.values()) if n}

# not running: no scheduled train since 2009 (museum runs only)
SUSPENDED = {"Gedserbanen"}

# Point names RINF writes out where the timetable (and likely OSM) abbreviates.
SHORT_NAME = {"Kirke Eskilstrup": "Kr. Eskilstrup", "Store Heddinge": "St. Heddinge",
              "Store Merløse": "St. Merløse", "Nørre Asmindrup": "Nr. Asmindrup",
              "Kavslunde": "Kauslunde"}

# The Øresund crossing (see the docstring): where the bridge's track crosses the border in OSM
# (ways 1185526677/8 against boundary way 71417261), and the track from RINF's point to it.
ORESUND_UOPID = "EU00141"
ORESUND_BORDER = (12.808962, 55.579239)
ORESUND_EXTRA_KM = 5.2


def _key(base):
    return base if len(base) != 6 else base[:3]


def group(base):
    """The line a RINF id is filed under: a name, a bare prefix for an unnamed group, or None
    to leave the section out."""
    if base in ID:
        return ID[base]
    if len(base) != 6 or base[:3] not in PREFIX:
        return None
    return PREFIX[base[:3]] or base[:3]


def dk_fix(secs, points):
    out = []
    # the Øresund border point, moved to the border
    for op, p in points.items():
        if p.get("uopid") == ORESUND_UOPID:
            p["lon"], p["lat"] = ORESUND_BORDER
            for s in secs:
                if op in (s["a"], s["b"]):
                    s["km"] = (s["km"] or 0.0) + ORESUND_EXTRA_KM
                    out.append(f"{ORESUND_UOPID} moved to the national border on the bridge; "
                               f"{s['label'][:60]} now {s['km']:.3f} km")
    n_left, left = 0, defaultdict(float)
    for s in secs:
        g = group(s["base"])
        if g is None:
            n_left += 1
            left[s["label"].replace("Section of Line ", "").split(" (")[0]] += s["km"] or 0.0
            g = f"(left out {s['base']})"
        s["line"] = s["base"] = g
    out.append(f"{len(secs) - n_left} sections filed under {len(NAMES)} named lines and "
               f"unnamed groups; {n_left} left out (station links, freight spurs): "
               + ", ".join(f"{k} {v:.2f} km" for k, v in sorted(left.items())))
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
            f"{sum(s['km'] or 0 for s in c):.1f} km "
            + "/".join(sorted({points[op].get('name', '?') for s in c
                               for op in (s['a'], s['b'])}))[:80] for c in order))
    n_alias = 0
    for p in points.values():
        short = SHORT_NAME.get(p.get("name"))
        if short:
            p["name"] = f"{p['name']} | {short}"
            n_alias += 1
    out.append(f"{n_alias} station names given the timetable's short form as a second name")
    return out


def dk_id_name(lid, _uop=None):
    n = lid.split("#")[0]
    return n if n in NAMES else None


def _fold(s):
    return re.sub(r"[^0-9a-zæøå]", "", (s or "").casefold())


_BY_FOLD = {_fold(n): n for n in NAMES}
# OSM way names that are another name for one of our lines (probe_kr_ways, 2026-10-03):
# Ringsted - Nykøbing F is being rebuilt as part of the Ringsted - Femern line.
WAY_ALIAS = {"Ringsted-Femern Banen": "Sydbanen", "Öresundsbanan": "Øresundsbanen"}
_BY_FOLD.update({_fold(k): v for k, v in WAY_ALIAS.items()})


def dk_way_line(tags):
    """The register line a track way's `name` is, if it is one of ours exactly (spaces, case
    and dashes aside). OSM's "Østjyske Længdebane" (Padborg - Frederikshavn) spans five of our
    lines and names none of them."""
    return _BY_FOLD.get(_fold(tags.get("name")))


def dk_stop_route(tags):
    """The S-tog lines (route=light_rail, "S-tog A: Hillerød => Køge") and Lokaltog's Nærumbanen
    (910), whose stops are halts on register lines. Not the light rails (Hovedstadens Letbane,
    Aarhus L1/L2), whose stops beside S-bane track are no S-bane halt."""
    return tags.get("route") == "light_rail" and (
        (tags.get("name") or "").startswith(("S-tog ", "Lokaltog ")))


FEED_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "gtfs" / "dk"
_FEED_NAMES = None


def _station_key(name):
    n = re.sub(r"\s*\(.*?\)", "", (name or "")).strip()
    n = re.sub(r"\s+st\.?$", "", n, flags=re.I)
    return _fold(n)


def dk_stop_extra(name):
    """An OSM station no route lists as a stop, which the timetable has as a rail station
    (stop_id 0000086xxxxx, a Danish UIC code). OSM's routes leave out Brejning, Gelsted, Jerne,
    Dyreby, Høvelte and Hjørring Øst; with every OSM station instead, Kongsvang, Rosenhøj, Tvis,
    Teglværket and Roskilde Festivalplads came in too, none of which has a train in the feed.
    No feed on disk: nothing."""
    global _FEED_NAMES
    if _FEED_NAMES is None:
        _FEED_NAMES = set()
        for z in sorted(FEED_DIR.glob("*.zip")):
            with zipfile.ZipFile(z) as zf, zf.open("stops.txt") as f:
                for r in csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")):
                    if r.get("stop_id", "").startswith("0000086"):
                        _FEED_NAMES.add(_station_key(r.get("stop_name")))
    return _station_key(name) in _FEED_NAMES


COUNTRY = {
    "iso3": "DNK", "wikidata": "Q35", "langs": ["da", "en"],
    "fix": dk_fix, "id_name": dk_id_name,
    "skip_line": lambda lid: lid.startswith("(left out"),
    "osm_rel": lambda _tags: None,
    "osm_stops": True,
    "osm_stop_route": dk_stop_route,
    "osm_stop_extra": dk_stop_extra,
    "cut_at_junctions": True,
    "light_rail_track": True,
    "km_floor": True,
    "way_line": dk_way_line,
    "suspended": lambda _ref, lids: any(l.split("#")[0] in SUSPENDED for l in lids),
    "im": {"8601_IM": "Banedanmark", "8604_IM": "Lokaltog", "8607_IM": "Lokaltog",
           "8606_IM": "Nordjyske Jernbaner"},
}
