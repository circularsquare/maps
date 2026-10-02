"""Germany: RINF holds DB InfraGO's network alone (`0080_IM`, 1,496 line ids, about 33,400 km),
no NE-Bahnen; those, the S-Bahn services, U-Bahn, Stadtbahn and trams stay OSM lines.

LINE NUMBERS. RINF's line id IS DB's VzG Streckennummer ("1700"), the number DB InfraGO, the
Infrastrukturregister, OSM's route=tracks relations and Wikidata's P1671 all use, so the rule
is certain and OSM's relations only steer the second-pass trace onto their own ways.

NAMES. DB InfraGO's own Streckenkurzname for the number, from its open "Infrastrukturdaten"
(Mobilithek; `data/raw/de/db/M1 Streckennetz.csv`): "1700 Hannover – Hamm (Westf)",
"6100 Berlin-Spandau – Hamburg-Altona". Wikidata is not used for names ("wikidata": None, and
its route-number pull lives in data/raw/de/wikidata_lines.json, outside rinf/de, so rinf.py
does not read it): its items are de.wikipedia's articles, which describe historic railways
over other extents (VzG 1700 carries "Bahnstrecke Hannover–Minden", "Hamm–Minden" and
"Hannover–Hamm"; 1720 carries four). OSM's route=tracks names are dropped for the same reason
(`de_rel` returns no name), so `id_name` names every line. DB abbreviates ("Bln-Spandau", "Frankf. Süd",
"Flensb. Gr"); `expand` writes those out, a dotted word from the words of the line's own RINF
points first ("Kaiserbr." -> "Kaiserbrücke"), a city code from a fixed list ("Bln-", "HH",
"DO-"), a one-letter code only where the line's own points carry the full name ("M-Gaschwitz"
is Markkleeberg-Gaschwitz, not München). DB writes a via station between double hyphens
("Mannheim --Basel-- - Konstanz"); it becomes a middle stop in the name.

VERSIONS. RINF carries every German section twice, "(from 2026-01-01 until 2026-12-31)" and
"(from 2027-01-01)", with different point URIs but the same uopids. rinf.py's version step
pairs most of them on (line, uopid, uopid), but where the 2027 network differs (Stuttgart 21's
new lines, a halt added mid-section: Seelze Mitte - Seelze Ost - Ahlem against Seelze Mitte -
Ahlem) the 2027-only sections have no 2026 partner and would be kept as "future, nothing
else". `fix` keeps only the sections whose label says they are valid today, so the 2026
network is built now, and the 2027 one from 2027-01-01 without any change here.

STATIONS. RINF places a big station at one point of its whole area, up to 2.3 km from the
platforms; `fix` also moves those onto their OSM station by exact name (`relocate`).

S-BAHN. OSM maps the Berlin and Hamburg S-Bahn as railway=light_rail, so `light_rail_track`
(rinf.py) adds that track and its stations; lines traced mostly on it are kind light_rail.
"""
import csv
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_CSV = ROOT / "data" / "raw" / "de" / "db" / "M1 Streckennetz.csv"
RINF_DIR = ROOT / "data" / "raw" / "rinf" / "de"

VALIDITY = re.compile(r"\(from (\d{4}-\d{2}-\d{2})(?: until (\d{4}-\d{2}-\d{2}))?\)\s*$")


def de_fix(secs, points):
    """Keep the sections whose label says they are valid today (module docstring), and move
    stations RINF places far from their platforms onto them (`relocate`)."""
    today = date.today().isoformat()
    keep, dropped = [], Counter()
    for s in secs:
        m = VALIDITY.search(s.get("label") or "")
        if m and (m.group(1) > today or (m.group(2) and m.group(2) < today)):
            dropped[m.group(1)[:4]] += 1
            continue
        keep.append(s)
    secs[:] = keep
    yield (f"{sum(dropped.values())} sections not valid on {today} left out "
           f"(by first year: {dict(dropped)}), {len(keep)} kept")
    yield from relocate(points)


PROC = ROOT / "data" / "proc" / "de"
FAR_M = 3000
CITY_FAR_M = 1600         # by the name without its city, only this far: "Hamburg-Eidelstedt"
                          # (2.8 km) and "Hamburg-Wilhelmsburg" (2.1 km) are yards, not the
                          # S-Bahn halts Eidelstedt and Wilhelmsburg
# Not a stop by its name, or not this one: "Hamburg-Altona (neu)" is the Diebsteich station
# still being built, 1.4 km from today's Altona.
NOT_A_STOP = re.compile(r"\((?:Üst|Awanst|Anst|Abzw|Bk|neu)\)")
# A two-part name is looked up by its first part only: "Berlin Hauptbahnhof - Lehrter Bahnhof"
# is Berlin Hauptbahnhof, but "Mannheim-Waldhof - Lampertheim" (a block post between the two)
# had been moved 2.5 km onto Lampertheim by its second part.
PARTS = re.compile(r"\s[-/|]\s")


def relocate(points):
    """DB InfraGO's RINF coordinate for a big station is one point of its whole area, often a
    yard throat: Göttingen 1,955 m from its platforms, Cuxhaven 2,327, Saarbrücken Hbf 2,303,
    Wiesbaden Hbf 1,630, Berlin Südkreuz 1,155. rinf.py looks for the OSM station within
    NAME_M (1,000 m) of the point, so these became junctions and their sections were traced
    from a yard. A passenger-typed point with no OSM station of a matching name within NAME_M
    (nor any within BLIND_M) is moved to the OSM station of EXACTLY its name within FAR_M, if
    there is one place of that name; failing that, of its name without the city in front within
    CITY_FAR_M, which is how OSM names Hamburg's and Berlin's S-Bahn stations ("Hamburg-Barmbek"
    is "Barmbek").
    Points that are not stops by their name (an "(Üst)", an "(Awanst)", "Hamburg-Altona (neu)")
    are left alone, and a two-part name is looked up by its first part (PARTS)."""
    import pickle
    import rinf
    try:
        with open(PROC / "stops.pkl", "rb") as f:
            stops = pickle.load(f)
    except FileNotFoundError:
        return
    try:
        ost = rinf.osm_stations(stops, light_rail=True)    # S-Bahn stations (light_rail_track)
    except TypeError:                                      # a rinf.py without the hook
        ost = rinf.osm_stations(stops)
    idx = rinf.StationIndex(ost)
    n, by = 0, Counter()
    for p in points.values():
        if p.get("type") not in rinf.PASSENGER_TYPES or "lon" not in p:
            continue
        name = re.sub(r"\s+", " ", rinf.NAME_TAG.sub("", p.get("name") or "")).strip()
        if not name or NOT_A_STOP.search(name):
            continue
        near = idx.within(p["lon"], p["lat"], rinf.NAME_M)
        if any(d <= rinf.BLIND_M or rinf.names_match(rinf.name_variants(name), ost[s]["keys"])
               for d, s in near):
            continue
        name = PARTS.split(name)[0].strip()
        keys = rinf.name_variants(name)
        far = sorted(idx.within(p["lon"], p["lat"], FAR_M))
        hits, how = [(d, s) for d, s in far if keys & ost[s]["keys"]], "exact name"
        m = re.match(r"^[A-Za-zÄÖÜäöüß.]+[-\s]+(.{5,})$", name)
        if not hits and m:
            alt = rinf.name_variants(m.group(1))
            hits, how = [(d, s) for d, s in far
                         if d <= CITY_FAR_M and alt & ost[s]["keys"]], "city dropped"
        if not hits:
            continue
        s0 = ost[hits[0][1]]
        if any(rinf.dist_m(s0["lon"], s0["lat"], ost[s]["lon"], ost[s]["lat"]) > 300
               for _d, s in hits):
            continue                                   # two places of that name: leave it
        p["lon"], p["lat"] = s0["lon"], s0["lat"]
        n += 1
        by[how] += 1
    yield (f"{n} passenger points more than {rinf.NAME_M} m from their OSM station moved onto "
           f"it ({dict(by)})")


def de_rel(tags):
    """OSM route=tracks/route=railway relations: the VzG number only (4 digits), no name."""
    ref = (tags.get("ref") or "").strip()
    m = re.fullmatch(r"(?:DB\s*|Strecke\s*|VzG\s*)?(\d{4})", ref)
    if not m:
        return None
    return m.group(1), None


# ---------------------------------------------------------------- names

_db = None


def db_names():
    """{VzG number: Streckenkurzname} from DB InfraGO's Streckennetz CSV."""
    global _db
    if _db is None:
        _db = {}
        csv.field_size_limit(1 << 30)
        if DB_CSV.exists():
            with open(DB_CSV, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    _db.setdefault(r["Streckennummer"], r["Streckenkurzname"].strip())
    return _db


_words = None
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def rinf_words():
    """Words of every RINF point name, per line and overall, for writing abbreviations out."""
    global _words
    if _words is None:
        import json
        per, every, names = defaultdict(Counter), Counter(), defaultdict(set)
        try:
            secs = json.loads((RINF_DIR / "sections.json").read_text(encoding="utf-8"))["rows"]
            pts = json.loads((RINF_DIR / "points.json").read_text(encoding="utf-8"))["rows"]
        except FileNotFoundError:
            secs, pts = [], []
        pname = {r["op"]: re.sub(r"\s+", " ", r.get("name") or "").strip() for r in pts}
        for r in secs:
            for op in (r["a"], r["b"]):
                n = pname.get(op, "")
                names[r.get("line") or ""].add(n)
                for w in WORD.findall(n):
                    # not RINF's bookkeeping names ("StrUeb4700_4813_20001961")
                    if len(w) >= 3 and w != "StrUeb":
                        per[r.get("line") or ""][w] += 1
                        every[w] += 1
        _words = (per, every, names)
    return _words


CITY = {"Bln": "Berlin", "Hmb": "Hamburg", "HH": "Hamburg", "HL": "Lübeck", "HB": "Bremen",
        "Stg": "Stuttgart", "Mü": "München", "Nür": "Nürnberg", "Ffm": "Frankfurt",
        "Lpz": "Leipzig", "Dre": "Dresden", "Drsd": "Dresden", "Mgd": "Magdeburg",
        "Bhv": "Bremerhaven", "Mz": "Mainz", "MZ": "Mainz", "Hal": "Halle", "Dü": "Düsseldorf",
        "DO": "Dortmund", "BO": "Bochum", "DU": "Duisburg", "GE": "Gelsenkirchen",
        "OB": "Oberhausen", "KO": "Koblenz", "WÜ": "Würzburg", "DA": "Darmstadt",
        "WI": "Wiesbaden", "KR": "Krefeld", "GI": "Gießen", "Han": "Hannover"}
# One-letter codes, used only where the line's own RINF points carry the city's name:
# "M-Gaschwitz" is Markkleeberg, "M-Pasing" München, "K M/Deutz" Köln.
CITY1 = {"D": ["Düsseldorf"], "F": ["Frankfurt"], "E": ["Essen"], "K": ["Köln"],
         "W": ["Wuppertal"], "H": ["Hannover"], "N": ["Nürnberg"],
         "M": ["München", "Markkleeberg"]}


def _sub_seq(a, b):
    a, b = a.lower(), b.lower()
    if not a or not b or a[0] != b[0] or len(b) <= len(a):
        return False
    i = 0
    for ch in b:
        if i < len(a) and ch == a[i]:
            i += 1
    return i == len(a)


def _full_word(ab, lid):
    per, every, _n = rinf_words()
    for pool, own in ((per.get(lid, Counter()), True), (every, False)):
        fits = Counter({w: c for w, c in pool.items()
                        if w.lower().startswith(ab.lower()) and len(w) > len(ab)})
        if not fits:
            fits = Counter({w: c for w, c in pool.items() if _sub_seq(ab, w)})
        if fits:
            top = fits.most_common(2)
            # Anywhere in the country, only a clear favourite: "Potsd." is Potsdam far more
            # often than Potsdamer.
            if own or len(top) == 1 or top[0][1] >= 2 * top[1][1]:
                return top[0][0]
    return None


def _expand_end(e, lid):
    per, _every, _n = rinf_words()
    e = re.sub(r"DB-Gr\b\.?", "DB-Grenze", e)
    e = re.sub(r"\bam M\.", "am Main", e)
    e = re.sub(r"\bM/Deutz\b", "Messe/Deutz", e)
    e = re.sub(r"\bGr$", "Grenze", e)
    m = re.match(r"^(Abzw\s+)?([A-Za-zÄÖÜäöü]{1,4})([-\s])", e)
    if m:
        code, sep = m.group(2), m.group(3)
        city = CITY.get(code)
        if not city and code in CITY1:
            city = next((c for c in CITY1[code] if per.get(lid, {}).get(c)), None)
        if city:
            e = (m.group(1) or "") + city + sep + e[m.end():]

    def rep(mm):
        ab = mm.group(1)
        if ab in ("St", "Gr"):                     # Sankt; Grenze is handled above
            return mm.group(0)
        if ab.lower().endswith("str"):
            return ab[:-3] + ("Straße" if len(ab) == 3 else "straße")
        w = _full_word(ab, lid)
        if w:
            return w
        if ab.endswith("br") and len(ab) > 3:
            return ab[:-2] + "brücke"
        return mm.group(0)
    # A dotted word at a word's end only: "Mannheim-Fr.feld" stays as DB writes it.
    e = re.sub(r"([A-Za-zÄÖÜäöüß]{2,})\.(?![A-Za-zÄÖÜäöüß])", rep, e)

    # DB also cuts words short with no dot to fit its column: "Johanngeorgenst - Schwarzenbg",
    # "Annaberg-Buchh". A word that is no word of the line's own RINF points, but begins one
    # of them (or, if only one fits, is spelt within it from its first letter), is that word.
    own = per.get(lid, Counter())
    every = _every

    def rep2(mm):
        w = mm.group(1)
        # a word some RINF point has whole is a word ("Nord", "Burbach"), never a stub
        if w in every or len(w) < 4:
            return w
        fits = [x for x in own if x.startswith(w) and len(x) > len(w)]
        if not fits:
            fits = [x for x in own if _sub_seq(w, x)]
            if len(fits) != 1:
                return w
        return max(fits, key=lambda x: own[x])
    return re.sub(r"(?<![A-Za-zÄÖÜäöüß.])([A-Za-zÄÖÜäöüß]+)(?![A-Za-zÄÖÜäöüß.])", rep2, e)


def expand(lid, raw):
    """DB's Streckenkurzname with its abbreviations written out, ends joined by en dashes."""
    via = [v.strip() for v in re.findall(r"--([^-]+)--", raw)]
    base = re.sub(r"\s*--[^-]+--\s*", " ", raw).strip()
    ends = [_expand_end(x.strip(), lid) for x in re.split(r"\s+-\s+", base) if x.strip()]
    via = [_expand_end(v, lid) for v in via]
    if via and len(ends) == 2:
        ends = [ends[0]] + via + [ends[1]]
    return " – ".join(ends)


# Abbreviated without a dot, so `expand` cannot see them; main lines only.
NAME_FIX = {"6081": "Berlin-Gesundbrunnen – Eberswalde – Stralsund",
            "6088": "Berlin-Gesundbrunnen – Neubrandenburg – Stralsund",
            # Usedom (UBB), not in DB's list; ends read off RINF's sections
            "6773": "Wolgaster Fähre – Seebad Heringsdorf",
            "6774": "Zinnowitz – Peenemünde"}


def de_name(lid, _uop_name=None):
    """"1700 Hannover – Hamm (Westf)", or None for a number DB's list lacks."""
    base = (lid or "").split("#")[0]
    if base in NAME_FIX:
        return f"{base} {NAME_FIX[base]}"
    raw = db_names().get(base)
    if raw:
        return f"{base} {expand(base, raw)}"
    ends = rinf_ends(base)
    return f"{base} {' – '.join(ends)}" if ends else None


def rinf_ends(base):
    """For the few numbers DB's list lacks (Usedom's 6768, 6773, 6774; Neuruppin's 6946): the
    names of the line's two ends in RINF, today's version, switch suffixes dropped."""
    import json
    try:
        secs = json.loads((RINF_DIR / "sections.json").read_text(encoding="utf-8"))["rows"]
        pts = json.loads((RINF_DIR / "points.json").read_text(encoding="utf-8"))["rows"]
    except FileNotFoundError:
        return None
    today = date.today().isoformat()
    pname = {r["op"]: r.get("name") or "" for r in pts}
    deg = Counter()
    for r in secs:
        m = VALIDITY.search(r.get("label") or "")
        if r.get("line") != base or (m and (m.group(1) > today
                                            or (m.group(2) and m.group(2) < today))):
            continue
        a, b = (re.sub(r"\s+", " ", re.sub(r",\s*\d*W\s*\d+.*$", "", pname.get(x, ""))).strip()
                for x in (r["a"], r["b"]))
        if a != b:
            deg[a] += 1
            deg[b] += 1
    ends = sorted(n for n, k in deg.items() if k == 1 and not n.startswith("StrUeb"))
    return ends if len(ends) == 2 else None


COUNTRY = {
    "iso3": "DEU", "wikidata": None, "langs": ["de"],
    "ref": lambda lid: lid if re.fullmatch(r"\d{4}", lid or "") else None,
    "rule_certain": True,
    "osm_rel": de_rel,
    "id_name": de_name,
    "fix": de_fix,
    # OSM maps the Berlin and Hamburg S-Bahn (DB InfraGO lines 6020, 6030, 1241, ...) as
    # railway=light_rail with light-rail stations; without this they had no track to trace on.
    "light_rail_track": True,
    "im": {"0080_IM": "DB InfraGO"},
}
