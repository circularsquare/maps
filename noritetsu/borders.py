"""Where a railway crosses a national border: the points every country's build cuts at.

    python borders.py --fetch        # ERA RINF: every border point (op-type 90), two SPARQLs
    python borders.py                # how many points, by country pair

WHY A SHARED TABLE.  A route over a border is built in each country from its own extract, and
neither extract has the station on the far side, so the section between the last station on
one side and the first on the other belonged to neither build (TER K80: Mouscron and Tourcoing,
no line between). Each country now keeps its own side of that section, from its last station
to the border point (build_model.border_tails), and the two halves join in the app because
both builds name the border point with ONE id. That only works if both builds cut at the same
point, so the point comes from a table both read, not from geometry each computes.

RINF's border points are that table for Europe: one per line crossing, the same uopid in both
countries' registers (Mouscron-Frontière is EU00084 in Belgium's RINF and France's), and on
the track (median 2 m from OSM's rails, 90% within 9 m, over 415 measured crossings). A RINF
register's junction there is already "e" + uopid, so the OSM half ends at the register's own
junction station. Country outlines were tried and are far too coarse to cut at: Natural Earth
10m and religiondots' shapes put the border a median 600 m from RINF's points, up to 5.5 km.

NAMES.  A border point is shown under one neutral name, the same in every country's
stations.json: "Belgium – France border", the two countries' Natural Earth names (as
tools/build_regions.py names countries) in alphabetical order, joined by an en dash. Not
either register's own name: RINF calls the same point "Mouscron-Frontière" in Belgium and
"Frontière FR - BE (Tourcoing - Mouscron)" in France. `countries` comes from which countries'
sections of line end at the point; where only one country files it (83 of 227), the other is
the nearest other country in Natural Earth within NEIGHBOUR_KM, which is coarse but only has
to say which country, not where the border is.

EXTRA holds crossings RINF lacks (Anita, 2026-10-01: add them by hand when the country on the
other side is built, not before). Ids there are "x" + a short name, except where one country's
RINF register already has its own point at the crossing under another type (DB InfraGO files
the Swiss crossings as type 120, "Weil am Rhein BW/CH"): then the id is that point's, "e" +
its uopid, so the register's line ends at the shared point with nothing else to do, and
gtfs_served, which counts only "e" ids as borders, sees it. A route end that meets no point is
logged by build_model ("no border point") and not drawn.

SAME names points that are one crossing although further apart than SAME_POINT_M, or not in
the table at all (a register's own junction there): canon() maps them to the first.

The table is border_points.json beside this file, tracked:
{"points": [{id, name, lon, lat, countries: [cc, cc]}], ...}.
"""
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLE = ROOT / "border_points.json"
NAMES = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
ENDPOINT = "https://graph.data.era.europa.eu/repositories/rinf-plus"
USER_AGENT = "noritetsu-rail-map/0.1 (personal rail map research; python-urllib)"

ISO3 = {"AUT": "at", "BEL": "be", "BGR": "bg", "CHE": "ch", "CZE": "cz", "DEU": "de",
        "DNK": "dk", "EST": "ee", "ESP": "es", "FIN": "fi", "FRA": "fr", "GRC": "gr",
        "HRV": "hr", "HUN": "hu", "IRL": "ie", "ITA": "it", "LIE": "li", "LTU": "lt",
        "LUX": "lu", "LVA": "lv", "NLD": "nl", "NOR": "no", "POL": "pl", "PRT": "pt",
        "ROU": "ro", "SWE": "se", "SVN": "si", "SVK": "sk"}

# (id, lon, lat, [cc, cc]): crossings RINF has no point for. Named like the rest.
EXTRA = [
    # Germany - Switzerland, DB InfraGO's lines on Swiss soil (Basel Badischer Bahnhof and
    # Riehen; the Hochrhein line Trasadingen - Schaffhausen - Thayngen). DB's RINF point there,
    # whose id this is; at the Swiss register's own "Landesgrenze" node (schienennetz.py
    # snaps a node within 30 m), 0-19 m from DB's point (Thayngen 57 m). rinf_countries/de.py
    # cuts DB's lines here (SWISS_SOIL).
    ("eDE00RQW", 7.60471, 47.58397, ["ch", "de"]),    # Weil am Rhein - Basel Bad Bf (4000)
    ("eDE00RQB", 7.63598, 47.56374, ["ch", "de"]),    # Basel Bad Bf - Grenzach (4000)
    ("eDE0RQRI", 7.65703, 47.59490, ["ch", "de"]),    # Riehen - Lörrach-Stetten (4400)
    ("eDE0RQER", 8.43249, 47.66195, ["ch", "de"]),    # Erzingen (Baden) - Trasadingen (4000)
    ("eDE0RQTG", 8.72804, 47.74578, ["ch", "de"]),    # Thayngen - Bietingen (4000)
    # Strasbourg's tram D over the Beatus-Rhenanus bridge to Kehl: the Rhine's middle on the
    # tram track, abreast of the rail bridge's EU00016 (160 m north).
    ("xKehlTram", 7.80167, 48.57419, ["de", "fr"]),
    # Niebüll - Tønder (NEG): where OSM's track (way 223519292) crosses the boundary (way
    # 1052245273). Germany's register ends at "Niebüll DB-Grenze", Denmark's at Tønder
    # (Denmark agent, 2026-10-03).
    ("xTonder", 8.872908, 54.899385, ["de", "dk"]),
    # Dundalk - Newry: where OSM's boundary of the Republic crosses the two tracks (ways
    # 31695909 / 395764463 end there); Ireland's RINF "Border" point (IE+OP42) moved onto it.
    ("eIEOP42", -6.379092, 54.069080, ["gb", "ie"]),
    # Thailand: the rail/boundary intersection in OSM (th_sources.md "Borders")
    ("xPadangBesar", 100.322477, 6.665252, ["my", "th"]),          # OSM node 11494429960
    # Malaysia - Singapore on the Johor Causeway: KTM's track (way 925109455) crosses the
    # boundary (way 1455785333), 1.27 km from JB Sentral (Shuttle Tebrau to Woodlands).
    ("xWoodlands", 103.769336, 1.452652, ["my", "sg"]),
    # Vietnam - China at Hữu Nghị Quan (Đồng Đăng - Pingxiang): where OSM's track (way
    # 482593999) crosses OSM's boundary of Vietnam (relation 49915). MR1/MR2 Gia Lâm -
    # Nanning, daily since 25 May 2025 (vn_sources.md "Borders").
    ("xDongDang", 106.715575, 21.972213, ["cn", "vn"]),
    # China - Hong Kong on the high-speed line (Futian - West Kowloon), in the tunnel under the
    # Shenzhen River: OSM splits both tracks on its boundary of Hong Kong (relation 913110),
    # at nodes 11 m apart (114.05339, 22.50288 and 114.05349, 22.50292); this is between them.
    # cn_register runs 广深港高速线 on from 福田 to it, hk_register West Kowloon to it.
    ("xFutian", 114.053440, 22.502902, ["cn", "hk"]),
    # Sadakhlo - Ayrum: the OSM node on the Debed bridge where ways 48754465/1103672429 cross
    # the boundary (caucasus_register.BORDER_XY). Yerevan - Tbilisi, Yerevan - Batumi.
    ("eXAMGE1", 44.898762, 41.211857, ["am", "ge"]),
    # Gardabani - Böyük Kəsik: ways 453284165 / 1186547878 over the boundary. Baku - Tbilisi.
    ("eXAZGE1", 45.165160, 41.405885, ["az", "ge"]),
    # Souk Ahras - Ghardimaou (Annaba - Tunis, 3 a week each way since 2026-09-15): where the
    # line from OSM's last Algerian track (8.355 E) meets the outline. OSM has no track for the
    # Tunisian side's last ~3 km, so only dz reaches it (nafrica_sources.md).
    ("eXDZTN1", 8.35785, 36.40970, ["dz", "tn"]),
    # Belarus - Russia and Moldova - Ukraine (by-md agent, 2026-10-03): where OSM's track
    # crosses OSM's boundary (relations 59065 / 58974); passenger trains over each.
    ("eBYRUOSINOVKA", 30.987625, 54.682434, ["by", "ru"]),    # Osinovka - Krasnoye (Minsk - Moscow)
    ("eBYRUZAOLSHA", 30.955662, 54.979000, ["by", "ru"]),     # Zaolsha - Rudnya (Vitebsk - Smolensk)
    ("eBYRUEZERISHCHE", 29.962024, 55.856790, ["by", "ru"]),  # Ezerishche - Nevel (St Petersburg)
    ("eBYRUALESHA", 29.379967, 55.754028, ["by", "ru"]),      # Alesha - Klyastitsa (Pskov region)
    ("eBYRUZAKOPYTYE", 31.592222, 52.453851, ["by", "ru"]),   # Zakopytye - Zlynka (Minsk - Adler)
    ("eMDUAVALCINET", 27.779499, 48.448785, ["md", "ua"]),    # Otaci - Mohyliv-Podilskyi (Kyiv)
    # Central Asia (casia agent, 2026-10-03): where OSM's track crosses the boundary, at the
    # crossings with passenger trains; ids from casia_register --borders.
    ("eXKGKZ01", 71.178156, 42.672092, ["kg", "kz"]),   # Almaty - Shymkent line into Kyrgyzstan at Kurkureu-su
    ("eXKGKZ02", 71.268025, 42.770566, ["kg", "kz"]),   # ... and back out (Zhuantobe)
    ("eXKGKZ03", 73.510054, 42.843153, ["kg", "kz"]),   # Merke - Kaindy (Bishkek trains)
    ("eXKZRU01", 46.739593, 48.357746, ["kz", "ru"]),   # Krasny Kut - Verkhny Baskunchak at Shunguli
    ("eXKZRU02", 46.790729, 48.951905, ["kz", "ru"]),   # ... at Saykhin
    ("eXKZRU03", 46.835359, 49.565688, ["kz", "ru"]),   # ... Dzhanybek - Kommunistichesky
    ("eXKZRU04", 46.840088, 49.318223, ["kz", "ru"]),   # ... Dzhanybek - Ingelovsky
    ("eXKZRU05", 48.489663, 46.674261, ["kz", "ru"]),   # Ganyushkino - Kigash (Atyrau - Astrakhan)
    ("eXKZRU06", 49.921064, 51.205373, ["kz", "ru"]),   # Semiglavy Mar - Ozinki (Oral - Saratov)
    ("eXKZRU07", 54.149785, 51.077942, ["kz", "ru"]),   # Shyngyrlau - Iletsk
    ("eXKZRU08", 56.145949, 50.855533, ["kz", "ru"]),   # Zhaysan - Iletsk (Aktobe - Orenburg)
    ("eXKZRU15", 68.217874, 55.004599, ["kz", "ru"]),   # Petropavl - Petukhovo
    ("eXKZRU16", 70.969414, 54.902734, ["kz", "ru"]),   # Bulayevo - Isilkul
    ("eXKZRU17", 75.202960, 53.869223, ["kz", "ru"]),   # Urlyutyub - Cherlak
    ("eXKZRU18", 77.016583, 53.756348, ["kz", "ru"]),   # Kyzyltuz - Terengul (Kulunda line)
    ("eXKZRU20", 81.082986, 51.180500, ["kz", "ru"]),   # Aul - Lokot (Semey - Barnaul)
    ("eXKZUZ01", 55.999280, 44.880861, ["kz", "uz"]),   # Oasis - Karakalpakstan
    ("eXKZUZ03", 69.179081, 41.441096, ["kz", "uz"]),   # Saryagash - Keles
    ("eXTJUZ02", 68.064605, 38.407029, ["tj", "uz"]),   # Kudukli - Pakhtaabad
    ("eXTJUZ03", 69.264341, 40.200681, ["tj", "uz"]),   # Bekabad - Spitamen
    # Vrbnica - Bijelo Polje (Belgrade - Bar): IŽS 287.438 = ŽICG 287+438.70; measured 2.245 km
    # along OSM's track from Vrbnica (balkans_register.BORDERS). Belgrade - Bar trains.
    ("eXMERS1", 19.777497, 43.136296, ["me", "rs"]),
    # Čapljina - Metković: HŽI's own RINF point "Metković DG" (type 90, missing from the table).
    # The seasonal Sarajevo - Ploče train; ba's Čapljina - border 7.3 km is kept by the feed.
    ("eEU00222", 17.65746, 43.05886, ["ba", "hr"]),
    ("xNongKhaiThanaleng", 102.715092, 17.880451, ["la", "th"]),    # mid-Mekong, Friendship Bridge
    ("xAranyaprathetPoipet", 102.550138, 13.661698, ["kh", "th"]),
    # Euskotren's E2 over the Bidasoa, on the metre-gauge track abreast of Adif/SNCF's EU00119.
    # France builds Hendaia - Irun Ficoba whole, an OSM section with no register station at
    # either end: build_model.split_at_borders sides those by the outline (2026-10-04).
    ("xIrunHendaia", -1.78530, 43.35024, ["es", "fr"]),
    # Abkhazia - Russia at the Psou bridge: where way 124797554, the only track over the river,
    # crosses OSM's boundary of Abkhazia (relation 1152720). Moscow, St Petersburg and Dioskuria
    # trains to Sukhum (caucasus_register.BORDER_XY, ru_register.BORDER).
    ("eXARUPSOU", 40.008443, 43.393733, ["ru", "xa"]),
]

# id -> (lon, lat): a fetched point RINF places off the track, put on it. Tracing finds a point
# only within NEAR_M of a route's rails.
MOVE = {
    # Grambow - Szczecin Gumieńce: the table's coordinate (Poland's) is 178 m off the rails in
    # both extracts, Germany's own RINF coordinate for the point 3 m.
    "eEU00056": (14.373356, 53.41592),
    # Görlitz - Zgorzelec over the Neisse viaduct: 121 m off (Poland's), Germany's 9 m.
    "eEU00049": (14.989824, 51.143241),
    # Klingenthal - Kraslice: 62 m off, Germany's 2 m.
    "eEU00040": (12.467407, 50.354166),
    # Bad Brambach - Plesná: the table has Germany's "Bad Brambach Grenze 3" under this id,
    # 690 m from the crossing; Czechia files EU00039 where Germany files its own "Bad Brambach
    # Grenze" (DE00DXB, 11 m apart), and both registers' lines end there
    # (rinf_countries/de.py swaps the two German ids).
    "eEU00039": (12.327243, 50.213682),
    # Øresund: "Peberholm grænse" is the managers' boundary at Peberholm's west end, 5.4 km
    # inside Denmark; the bridge's rail ways (1185526677/8) cross the boundary (way 71417261)
    # here. rinf_countries/dk.py and se.py (_oresund) move it the same way (Denmark agent).
    "eEU00141": (12.808962, 55.579239),
    # Röszke's point is at Röszke station, 7 km inside Hungary; this is where the Szeged -
    # Subotica track crosses (balkans agent, 2026-10-03).
    "eEU00200": (19.968472, 46.160999),
    # Jimbolia's is 6 km off the Kikinda line (RINF "Jimbolia FR").
    "eEU00247": (20.641535, 45.806910),
    # The Channel Tunnel: RINF's point lies 450 m off OSM's two bores. OSM cuts both at the
    # boundary, nodes 1567644045 (north bore, 1.4962632, 51.0150141) and 329586625 (south,
    # 1.4959543, 51.0146143), 49 m apart; this is midway, 25 m from each (within NEAR_M), so
    # Eurostar's and Le Shuttle's routes in gb and fr reach it (gb/fr agent, 2026-10-05).
    "eEU00228": (1.496109, 51.014814),
}

# (the id that stands for the crossing, [the other ids of it]).
SAME = [
    # Konstanz - Kreuzlingen: RINF files the two tracks over the border 22 m apart (4000 to
    # Kreuzlingen, 4322 to Kreuzlingen Hafen) and DB's register ends both at its own points,
    # "Konstanz Grenze" 64 m off and "Konstanz Grenze Romanshorn" 207 m off. Two table
    # points 22 m apart both lie on a section to Kreuzlingen, and build_model.split_at_borders
    # cuts only at one, so Switzerland's Konstanz - Kreuzlingen stayed whole.
    ("eEU00027", ["eEU00026", "eDE0RXKZ", "eDE0RXKR"]),
    # Selb - Aš: DB's register ends at its "Selb-Plößberg Grenze", 7 m from Czechia's EU00230.
    ("eEU00230", ["eDE0NXSB"]),
]

# A border point this close to a route's track is where that track crosses.
NEAR_M = 60
# A point only one country files: the other country is the nearest one within this.
NEIGHBOUR_KM = 50

# BUILT COUNTRIES' OUTLINES, read here and by tools/build_regions.py: religiondots'
# country_shapes.geojson (~200 m), Natural Earth 1:10m for a country it lacks (Luxembourg),
# and OUTLINE where religiondots' country is not the one built. Too coarse to cut at (above),
# but fine for saying which of a section's two ends lies deeper inside a country
# (build_model.split_at_borders).
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"
# religiondots' codes where they differ from ISO 3166.
ISO = {"uk": "GB"}
# Regions Natural Earth's admin-0 file has no country for (user-assigned codes): their name,
# for border points' names and tools/build_regions.py.
NAME = {"xa": "Abkhazia"}
# Ukraine: religiondots' `ua` holds Crimea (built with Russia) and the 2022-annexed area (given
# to no country); ua_register.py writes OSM's Ukraine less both (2026-10-03).
OUTLINE = {"ua": ROOT / "data" / "raw" / "ua" / "outline.geojson",
           # Georgia less Abkhazia (its own region, xa) and South Ossetia (given to no
           # region) (caucasus_register.py, 2026-10-04).
           "ge": ROOT / "data" / "raw" / "ge" / "outline.geojson",
           # Abkhazia, OSM relation 1152720 (caucasus_register.py, 2026-10-04); in neither
           # religiondots' shapes nor Natural Earth.
           "xa": ROOT / "data" / "raw" / "xa" / "outline.geojson",
           # Moldova, Transnistria included again and greyed (bymd_register.py, 2026-10-04).
           "md": ROOT / "data" / "raw" / "md" / "outline.geojson"}
_OUTLINES = {}


def outline(cc):
    """cc's outline as one shapely geometry, or None. Cached."""
    if cc in _OUTLINES:
        return _OUTLINES[cc]
    from shapely.geometry import shape
    from shapely.ops import unary_union
    geoms = []
    if cc in OUTLINE and OUTLINE[cc].exists():
        feats = json.loads(OUTLINE[cc].read_text(encoding="utf-8"))
        geoms = [shape(f.get("geometry", f)) for f in feats.get("features", [feats])]
    elif SHAPES.exists():
        if "_shapes" not in _OUTLINES:
            by = defaultdict(list)
            for f in json.loads(SHAPES.read_text(encoding="utf-8"))["features"]:
                by[f["properties"]["cc"]].append(f["geometry"])
            _OUTLINES["_shapes"] = by
        geoms = [shape(g) for g in _OUTLINES["_shapes"].get(cc, [])]
    if not geoms and NAMES.exists():
        want, best = ISO.get(cc, cc.upper()), None
        for f in json.loads(NAMES.read_text(encoding="utf-8"))["features"]:
            q = f["properties"]
            if q.get("ISO_A2_EH") == want and (best is None
                                               or (q.get("POP_EST") or 0) > best[0]):
                best = (q.get("POP_EST") or 0, f["geometry"])
        geoms = [shape(best[1])] if best else []
    _OUTLINES[cc] = unary_union(geoms) if geoms else None
    return _OUTLINES[cc]


def depth_m(cc, lon, lat):
    """How far (lon, lat) is inside cc's outline, in metres, negative outside; None with no
    outline. Approximate (degrees times 111 km); for comparing two points a few km apart."""
    g = outline(cc)
    if g is None:
        return None
    from shapely.geometry import Point
    p = Point(lon, lat)
    d = g.boundary.distance(p) * 111320
    return d if g.contains(p) else -d

Q_POINTS = """
PREFIX era: <http://data.europa.eu/949/>
PREFIX wgs: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
SELECT DISTINCT ?op ?uopid ?lat ?lon ?wkt ?country WHERE {
  ?op a era:OperationalPoint ; era:opType <http://data.europa.eu/949/concepts/op-types/90> .
  OPTIONAL { ?op era:uopid ?uopid }
  OPTIONAL { ?op era:inCountry ?country }
  OPTIONAL { ?op era:netReference ?nr . ?nr wgs:lat ?lat ; wgs:long ?lon }
  OPTIONAL { ?op geo:hasGeometry ?g . ?g geo:asWKT ?wkt }
} ORDER BY ?op
"""
Q_PAIRS = """
PREFIX era: <http://data.europa.eu/949/>
SELECT DISTINCT ?uopid ?country WHERE {
  ?op a era:OperationalPoint ; era:opType <http://data.europa.eu/949/concepts/op-types/90> ;
      era:uopid ?uopid .
  ?sol a era:SectionOfLine ; era:inCountry ?country .
  { ?sol era:opStart ?op } UNION { ?sol era:opEnd ?op }
} ORDER BY ?uopid ?country
"""


def sparql(query):
    body = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return [{v: b[v]["value"] for v in b} for b in d["results"]["bindings"]]


def cc_of(iri):
    t = (iri or "").rsplit("/", 1)[-1]
    return ISO3.get(t, t.lower())


def countries():
    """cc -> (English name, outline), Natural Earth 10m; the most populous feature names a
    code, as in tools/build_regions.py."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    names, pop, geo = {}, {}, defaultdict(list)
    for f in json.loads(NAMES.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        cc = (q.get("ISO_A2_EH") or "").lower()
        if not cc or cc == "-99":
            continue
        p = q.get("POP_EST") or 0
        if cc not in names or p > pop[cc]:
            names[cc], pop[cc] = q.get("NAME"), p
        geo[cc].append(shape(f["geometry"]))
    return {cc: (names[cc], unary_union(g)) for cc, g in geo.items()}


def border_name(ccs, ne):
    return " – ".join(sorted(ne[c][0] if c in ne else NAME.get(c, c.upper())
                             for c in ccs)) + " border"


def complete(points, ne):
    """Give each point two countries (the nearest other one where only one is known) and its
    neutral name."""
    from shapely.geometry import Point
    lone = 0
    for p in points:
        if len(p["countries"]) < 2:
            pt = Point(p["lon"], p["lat"])
            near = sorted((g.distance(pt), cc) for cc, (_n, g) in ne.items()
                          if cc not in p["countries"])
            # Not always at the border itself: Röszke's point is 7 km short of Serbia, the
            # Channel Tunnel's at the French portal, 35 km from England.
            km = near[0][0] * 111.32 * math.cos(math.radians(p["lat"])) if near else None
            if km is not None and km <= NEIGHBOUR_KM:
                p["countries"].add(near[0][1])
            else:
                lone += 1
        p["name"] = border_name(p["countries"], ne)
    return lone


def fetch():
    import re
    pairs = defaultdict(set)
    for r in sparql(Q_PAIRS):
        pairs[r["uopid"]].add(cc_of(r["country"]))
    out = {}
    for r in sparql(Q_POINTS):
        u = r.get("uopid")
        if not u:
            continue
        try:
            lon, lat = float(r["lon"]), float(r["lat"])
        except (KeyError, ValueError):
            m = re.search(r"POINT\s*\(\s*([-+\d.]+)\s+([-+\d.]+)", r.get("wkt", ""))
            if not m:
                continue
            lon, lat = float(m.group(1)), float(m.group(2))
        p = out.setdefault(u, {"id": f"e{u}", "lon": round(lon, 6), "lat": round(lat, 6),
                               "countries": set(pairs.get(u, ()))})
        if r.get("country"):
            p["countries"].add(cc_of(r["country"]))
    pts = [p for _u, p in sorted(out.items())]
    lone = complete(pts, countries())
    rows = [{"id": p["id"], "name": p["name"], "lon": p["lon"], "lat": p["lat"],
             "countries": sorted(p["countries"])} for p in pts]
    TABLE.write_text(json.dumps({"source": ENDPOINT, "fetched": date.today().isoformat(),
                                 "points": rows}, ensure_ascii=False, indent=0),
                     encoding="utf-8")
    print(f"wrote {len(rows)} border points to {TABLE}; {lone} with one country only")


def load(canonical_only=False):
    """[{id, name, lon, lat, countries}], id being the station id every build gives it.
    `canonical_only`: leave out points that are a co-located duplicate of another (`canon`),
    for tracing route ends to a border; naming still wants every id."""
    rows = json.loads(TABLE.read_text(encoding="utf-8"))["points"]
    pts = [{**r, "countries": set(r["countries"])} for r in rows]
    for p in pts:
        if p["id"] in MOVE:
            p["lon"], p["lat"] = MOVE[p["id"]]
    if EXTRA:
        ne = countries()
        extra = [{"id": pid, "lon": lon, "lat": lat, "countries": set(ccs)}
                 for pid, lon, lat, ccs in EXTRA]
        complete(extra, ne)
        pts += extra
    if canonical_only:
        dup = canon(pts)
        pts = [p for p in pts if p["id"] not in dup]
    return pts


# RINF files some crossings twice, a point per country at the same spot: Bulgaria's register
# ends at EU00208 and Romania's at EU00209 (0 m apart), Czechia and Poland carry both EU00072
# and EU00073 (6 m), Austria and Switzerland both EU00118 and CH15472 (0 m). Two ids never
# join in the app, so each such group is one id: the eEU one, else the lowest.
SAME_POINT_M = 15


def canon(pts=None):
    """{duplicate id: the id that stands for it}, for points within SAME_POINT_M."""
    if pts is None:
        pts = load()
    out = {}

    def m(a, b):
        dx = (a["lon"] - b["lon"]) * math.cos(math.radians(a["lat"])) * 111320
        return math.hypot(dx, (a["lat"] - b["lat"]) * 110570)

    same = {o: keep for keep, others in SAME for o in others}
    order = sorted(pts, key=lambda p: (not p["id"].startswith("eEU"), p["id"]))
    for i, a in enumerate(order):
        if a["id"] in out or a["id"] in same:
            continue
        for b in order[i + 1:]:
            if b["id"] not in out and b["id"] not in same and m(a, b) <= SAME_POINT_M:
                out[b["id"]] = a["id"]
    out.update(same)
    return out


class Index:
    """Border points by 0.02-degree cell, for "which points lie on this polyline"."""

    def __init__(self, points):
        self.pts = points
        self.by_id = {p["id"]: p for p in points}
        self.cell = defaultdict(list)
        for i, p in enumerate(points):
            self.cell[(int(p["lon"] * 50), int(p["lat"] * 50))].append(i)

    def along(self, xy, near_m=NEAR_M):
        """Points within near_m of the polyline xy (rows of lon, lat), nearest-along first:
        (metres along xy to the foot of the perpendicular, point, segment index, foot lon,
        foot lat, metres off the track)."""
        if len(xy) < 2:
            return []
        cand = set()
        for x, y in xy:
            cx, cy = int(x * 50), int(y * 50)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.update(self.cell.get((cx + dx, cy + dy), ()))
        if not cand:
            return []
        kx = math.cos(math.radians(float(xy[0][1]))) * 111320
        ky = 110570
        cum = [0.0]
        for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
            cum.append(cum[-1] + math.hypot((x2 - x1) * kx, (y2 - y1) * ky))
        out = []
        for i in cand:
            p = self.pts[i]
            best = None
            for k, ((x1, y1), (x2, y2)) in enumerate(zip(xy[:-1], xy[1:])):
                ax, ay = (x2 - x1) * kx, (y2 - y1) * ky
                px, py = (p["lon"] - x1) * kx, (p["lat"] - y1) * ky
                L2 = ax * ax + ay * ay
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, (px * ax + py * ay) / L2))
                d = math.hypot(px - t * ax, py - t * ay)
                if best is None or d < best[0]:
                    best = (d, k, t)
            d, k, t = best
            if d <= near_m:
                (x1, y1), (x2, y2) = xy[k], xy[k + 1]
                out.append((cum[k] + t * (cum[k + 1] - cum[k]), p, k,
                            x1 + t * (x2 - x1), y1 + t * (y2 - y1), d))
        return sorted(out, key=lambda o: o[0])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--fetch" in sys.argv:
        fetch()
    pts = load()
    by = defaultdict(int)
    for p in pts:
        by[p["name"]] += 1
    print(f"{len(pts)} border points: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
