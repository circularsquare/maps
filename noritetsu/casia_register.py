"""Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan and Turkmenistan: Tariff Guide No. 4 (the
CIS tariff guide, Тарифное руководство № 4) written into rinf.py's input format, plus a
timetable feed made from KTZ's ticket site. The same recipe as Ukraine's (ua_register.py) and
Russia's (ru_register.py), whose parsers this reuses. casia_sources.md has the research and the
numbers; this docstring is how it works.

    python casia_register.py --esr kz data/raw/kazakhstan-latest.osm.pbf   # ESR codes, station names
    (likewise uz uzbekistan, kg kyrgyzstan, tj tajikistan, tm turkmenistan; before the .pbf goes)
    python casia_register.py --clip kz       # after every extract (data/proc/kz/full kept)
    python casia_register.py --borders       # data/raw/kz/casia_borders.json: every crossing
    python casia_register.py --place         # placement alone, with its log (cached in data/raw/kz/placed.pkl)
    python casia_register.py --crawl         # KTZ's station schedules and train routes (~90 min, 1 s apart)
    python casia_register.py --names         # each route call's Express code, KTZ's station search (~20 min)
    python casia_register.py --timetable     # -> data/raw/gtfs/<cc>/<cc>_casia.gtfs.zip (all five)
    python casia_register.py --convert kz --colours kz   # -> data/raw/rinf/kz/*.json, colours/kz.csv
    python build_model.py --region kz --register casia_register:data/raw/rinf/kz   # converts, then builds

THE REGISTER.  Book 1 of the tariff guide (data/raw/ru/tr4_kniga1_*.xls, the file Russia's
build reads) has one sheet per railway administration: Кзх (Kazakhstan, road 68), Узбк (73),
Кирг (70), Тадж (74), Трк (75). Each lists every tariff section with every station, halt and
post in order, its ESR code and integer tariff km. One tariff section is one register line, as
in Russia and Ukraine. A sheet is an administration, not a territory: the Kyrgyz sheet's first
section starts at Lugovaya in Kazakhstan, the Turkmen sheet's Kerki line runs through
Uzbekistan at Talimarjan, Russia's South Urals sheet runs through Petropavl. So every sheet
that touches the five (theirs and Russia's) is read, and a pair of consecutive points goes to
the country both points lie in (OSM's boundary of that country). The operator shown is the
administration whose sheet lists the section.

POINTS are placed at OSM's node with their ESR code (`--esr`, all five extracts and Russia's),
else at osm.sbin.ru's 2021 node for the code, else at an OSM station of the same name near the
point's placed neighbours (`lkey`: one Latin spelling for Russian, Kazakh, Uzbek and Turkmen
names, so "Ашгабат" meets "Aşgabat"), else at Wikidata's item with the code (P2815). A
placement that disagrees with the tariff km to its placed neighbours is taken out again
(`drop_outliers`), and --convert puts what is left between two placed points on OSM's track at
its share of the tariff km (`interpolate`). Names shown are OSM's `name` (the national
language), else Wikidata's label in it, else Book 1's own Russian.

STOPS (ua_register's rule).  A point is a stop when Book 2 gives it a passenger operation and
a train calls there (the KTZ timetable, or an OSM train route stop within SERVED_M) or it is a
halt ("ОП" in Book 1, or an OSM railway=halt within HALT_M) on a stretch that has halts. A
stretch between two called points with no halt, STRETCH_KM or longer, is freight track unless
some train calls at both ends one after the other: its ends are cloned as junctions, so it
answers to the timetable like any junction-ended section.

BORDERS.  A consecutive pair of points in two countries is a crossing. `--borders` finds where
OSM's track between them crosses the boundary and gives it an id ("XKZUZ01", border point
"eXKZUZ01" in the build), the same in both countries' builds: each country's line runs from
its last point to that border point, with the pair's tariff km split by crow-fly distance.
borders.EXTRA needs these (casia_sources.md lists them).

THE TIMETABLE.  No feed is published. KTZ's ticket site (bilet.railways.kz; robots.txt allows
everything) sells every train in the CIS Express system, Uzbek, Kyrgyz and Tajik trains
included, and shows each station's departures for a date and each train's calls. `--crawl`
fetches the schedules of every station with an Express code (osm.sbin.ru's ESR list) for
CRAWL_DATES, then one route per train; `--names` looks each call's name up in the site's
station search for its Express code. `--timetable` places every call (the Express code's ESR
code where its Book 1 name agrees, else the name, `Places`), picks one place per call by the
shortest path with the call times as a speed limit (`match_calls`), adds the trains no KTZ
page shows (HAND and railway.gov.tm's page, `hand_trains`), leaves out trains running less
than weekly, and writes a GTFS feed per country that gtfs_served reads as it reads every
national feed, plus the called places --convert reads for stops.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "data" / "raw" / "kz"          # files the five share live with Kazakhstan's
RU_RAW = ROOT / "data" / "raw" / "ru"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
CCS = ("kz", "uz", "kg", "tj", "tm")
ISO3 = {"kz": "KAZ", "uz": "UZB", "kg": "KGZ", "tj": "TJK", "tm": "TKM"}
LANG = {"kz": "kk", "uz": "uz", "kg": "ky", "tj": "tg", "tm": "tk"}
NAME_EN = {"kz": "Kazakhstan", "uz": "Uzbekistan", "kg": "Kyrgyzstan", "tj": "Tajikistan",
           "tm": "Turkmenistan", "ru": "Russia", "cn": "China", "af": "Afghanistan",
           "ir": "Iran"}

# Book 1's sheets of the five administrations: sheet -> (road code, operator as shown).
ROADS = {
    "Кзх": ("68", "Қазақстан темір жолы"),
    "Узбк": ("73", "Oʻzbekiston temir yoʻllari"),
    "Кирг": ("70", "Кыргыз темир жолу"),
    "Тадж": ("74", "Роҳи оҳани Тоҷикистон"),
    "Трк": ("75", "Türkmendemirýollary"),
}


def log(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def raw(cc):
    return ROOT / "data" / "raw" / cc


def proc(cc):
    return ROOT / "data" / "proc" / cc


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


# ================================================================ the OSM side

NAME_TAGS = ("name", "name:ru", "name:kk", "name:uz", "name:uz-Cyrl", "name:ky", "name:tg",
             "name:tk", "name:en", "old_name", "official_name")


def esr_pass(cc, pbf):
    """From the .pbf (extract.py keeps neither esr:user nor the name:xx tags): every node with
    an ESR code, and every rail station or halt node with its names. data/raw/<cc>/osm_esr.json
    and osm_stations.json."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    esr, st = defaultdict(list), []
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("esr:user", "esr", "railway:esr", "railway",
                                               "public_transport")))
    for obj in fp:
        t = obj.tags
        code = t.get("esr:user") or t.get("esr") or t.get("railway:esr")
        rw = t.get("railway")
        if not code and rw not in ("station", "halt") and not (
                t.get("public_transport") == "station" and t.get("train") == "yes"):
            continue
        lon, lat = round(obj.location.lon, 6), round(obj.location.lat, 6)
        names = {k: t.get(k) for k in NAME_TAGS if t.get(k)}
        row = {"id": obj.id, "lon": lon, "lat": lat, "names": names, "railway": rw,
               "pt": t.get("public_transport"), "train": t.get("train"),
               "usage": t.get("usage") or t.get("disused:railway"), "esr": code or "",
               "express": t.get("express:user") or t.get("railway:ref:express") or ""}
        if code:
            for c in code.replace(",", ";").split(";"):
                c = c.strip()
                if c.isdigit() and len(c) == 6:
                    esr[c].append(row)
        if rw in ("station", "halt") or t.get("public_transport") == "station":
            st.append(row)
    d = raw(cc)
    d.mkdir(parents=True, exist_ok=True)
    (d / "osm_esr.json").write_text(json.dumps(esr, ensure_ascii=False), "utf-8")
    (d / "osm_stations.json").write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    log(f"--esr {cc}: {sum(len(v) for v in esr.values())} nodes, {len(esr)} codes; "
        f"{len(st)} station nodes -> {d}")


# ================================================================ outlines and the clip

REACH_DEG = 0.03          # the clip reaches this far over the border, so crossings keep their track


_SHAPES = {}


def boundary(cc):
    """OSM's boundary of the country (polygons.openstreetmap.fr), as shapely; Russia's from
    Russia's build."""
    if cc not in _SHAPES:
        from shapely.geometry import shape
        p = (RU_RAW / "ru_boundary.geojson") if cc == "ru" else raw(cc) / f"{cc}_boundary.geojson"
        g = json.loads(p.read_text("utf-8"))
        if g.get("type") == "FeatureCollection":
            from shapely.ops import unary_union
            g = unary_union([shape(f["geometry"]) for f in g["features"]])
        else:
            g = shape(g)
        import shapely
        _SHAPES[cc] = g.buffer(0)
        shapely.prepare(_SHAPES[cc])
    return _SHAPES[cc]


def outline(cc, reach=True):
    import shapely
    g = boundary(cc)
    if reach:
        g = g.buffer(REACH_DEG)
    shapely.prepare(g)
    return g


def write_outline(cc):
    from shapely.geometry import mapping
    shp = boundary(cc)
    p = raw(cc) / "outline.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"what": f"{NAME_EN[cc]}: OSM's boundary relation"},
         "geometry": mapping(shp)}]}), "utf-8")
    log(f"--outline {cc}: {shp.area:.3f} sq deg -> {p}")


def clip(cc):
    """data/proc/<cc> from data/proc/<cc>/full (the extract as extract.py wrote it; moved
    there on the first run, so this can be rerun): what lies inside the country's boundary
    plus REACH_DEG kept. A way goes if at least half its nodes are outside, a stop if it is,
    a relation if no member is left (ua_register.clip's rules). Geofabrik's extracts reach
    some km over each border; without this the track and routes there were built twice."""
    import numpy as np
    import shapely
    pr = proc(cc)
    full = pr / "full"
    names = ("ways.pkl", "rels.pkl", "stops.pkl", "infra.pkl", "coords.npz")
    stamp = pr / "clip_stamp.json"
    fresh = (pr / "ways.pkl").exists() and (
        not stamp.exists() or (pr / "ways.pkl").stat().st_mtime > stamp.stat().st_mtime + 5)
    if fresh:
        full.mkdir(exist_ok=True)
        for fn in names:
            os.replace(pr / fn, full / fn)
        log(f"clip: the extract in {pr} moved to {full}")

    def rd(fn):
        with open(full / fn, "rb") as f:
            return pickle.load(f)
    ways, rels, stops, infra = (rd(f) for f in names[:4])
    with np.load(full / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    shp = outline(cc)
    out = ~shapely.contains_xy(shp, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut[tags.get("name") or "(unnamed)"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items() if shapely.contains_xy(shp, v[1], v[2])}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    for name, v in cut.most_common(15):
        log(f"  cut {v:5d}  {name}")
    log(f"clip {cc}: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = pr / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, pr / fn)
    import shutil
    shutil.copyfile(full / "coords.npz", pr / "coords.npz")
    stamp.write_text(json.dumps({"clipped_from": str(full), "ways": len(keep_w)}), "utf-8")


# ================================================================ the register: Book 1

def roads():
    """Sheet -> (road code, operator): the five administrations and Russia's railways (whose
    sections run through Kazakhstan at Petropavl, Kulunda, Saykhin...). The 2022-annexed
    sheets are left out."""
    import ru_register as rr
    out = dict(ROADS)
    for sh, v in rr.ROADS.items():
        if sh not in rr.ANNEXED:
            out[sh] = v
    return out


def book1():
    """Every tariff section of the sheets in roads() that has a point in one of the five
    (by osm.sbin.ru's country of its ESR code, or its sheet), points in order."""
    import ru_register as rr
    saved = rr.ROADS
    rr.ROADS = roads()
    try:
        secs, name = rr.book1()
    finally:
        rr.ROADS = saved
    iso = sbin_iso()
    out = [s for s in secs if s["sheet"] in ROADS
           or any(iso.get(p["esr"]) in ("KZ", "UZ", "KG", "TJ", "TM") for p in s["points"])]
    return out, name


_SBIN = {}


def sbin_iso():
    """ESR code -> its country (ISO 3166 alpha-2) in osm.sbin.ru's ESR list."""
    if not _SBIN:
        import csv
        with open(SHARED / "sbin_esr.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                _SBIN[r["esr"]] = (r["iso3166"] or "")[:2]
    return _SBIN


def sbin_osm():
    """ESR code -> (lon, lat, name) from osm.sbin.ru's osm2esr table (a 2021 snapshot of OSM's
    esr:user nodes): codes today's OSM no longer carries."""
    import csv
    out = {}
    p = SHARED / "sbin_osm2esr.csv"
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                out.setdefault(r["esr"], (float(r["lon"]), float(r["lat"]), r["name"],
                                          r.get("railway") or ""))
            except ValueError:
                continue
    return out


def load_osm_stations(ccs=CCS):
    """OSM rail station and halt nodes of the five extracts (from --esr), every name kept."""
    out = []
    for cc in ccs:
        p = raw(cc) / "osm_stations.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text("utf-8")):
            if r["usage"] in ("disused", "abandoned") or r["railway"] in ("disused", "abandoned"):
                continue
            if r["railway"] not in ("station", "halt") and not (
                    r["pt"] == "station" and r["train"] == "yes"):
                continue
            nm = r["names"]
            out.append({"id": r["id"], "lon": r["lon"], "lat": r["lat"],
                        "name": nm.get("name") or nm.get("name:ru") or "",
                        "ru": nm.get("name:ru") or "",
                        "names": list(nm.values()),
                        "en": nm.get("name:en") or "", "rw": r["railway"], "esr": r["esr"],
                        "cc": cc})
    return out


def osm_esr():
    """ESR code -> OSM nodes carrying it: the five extracts' and Russia's (data/raw/ru, read
    only; its rows are lists)."""
    out = defaultdict(list)
    for cc in CCS:
        p = raw(cc) / "osm_esr.json"
        if p.exists():
            for c, rows in json.loads(p.read_text("utf-8")).items():
                out[c] += rows
    p = RU_RAW / "osm_esr.json"
    if p.exists():
        for c, rows in json.loads(p.read_text("utf-8")).items():
            for r in rows:
                if c not in out:
                    out[c].append({"id": r[1], "lon": r[2], "lat": r[3],
                                   "names": {"name": r[4]} if r[4] else {}, "railway": r[5],
                                   "pt": r[6], "train": r[7], "ru_extract": True})
    return out


def wd_stations():
    """ESR code -> {lon, lat, ru, nat, en, cc}: Wikidata's items with P2815 in the five. A
    code on two items that disagree is left out."""
    got = defaultdict(list)
    for cc in CCS:
        p = raw(cc) / "wd_stations.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text("utf-8"))["rows"]:
            m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r.get("coord", ""))
            en = r.get("en") or ""
            if not en and r.get("enwiki"):
                import urllib.parse
                en = urllib.parse.unquote(r["enwiki"].rsplit("/", 1)[-1]).replace("_", " ")
            got[r["esr"]].append({"lon": float(m.group(1)) if m else None,
                                  "lat": float(m.group(2)) if m else None,
                                  "ru": clean_label(r.get("ru") or ""),
                                  "nat": clean_label(r.get("nat") or ""), "en": en,
                                  "q": r["s"], "cc": cc})
    out = {}
    for c, v in got.items():
        if len({(x["ru"], x["lon"]) for x in v}) == 1:
            out[c] = v[0]
    return out


def clean_label(s):
    """A Wikidata label as a station name: "Шу (станция)" -> "Шу"."""
    s = re.sub(r"\s*\((?:станция|станциясы|бекеті|stansiya|bekat|вокзал|остановочный пункт|"
               r"платформа|разъезд|железнодорожная станция|station)[^)]*\)", "", s or "",
               flags=re.I).strip()
    s = re.sub(r"^(?:станция|железнодорожная станция|темір жол бекеті)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(?:railway station|station|temir yo'l bekati|темір жол бекеті|бекеті)$",
               "", s, flags=re.I)
    return s


def country_at(lon, lat):
    """Which of the five (or ru) a place lies in, by OSM's boundaries; "" for none."""
    import shapely
    for cc in CCS + ("ru",):
        if shapely.contains_xy(boundary(cc), lon, lat):
            return cc
    return ""


# One Latin spelling for a name in Russian, Kazakh or Kyrgyz Cyrillic, or Uzbek or Turkmen
# Latin, so that Book 1's "Ашгабат" meets OSM's "Aşgabat" and "Бухоро 1" meets "Buxoro 1".
_CYR = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
        "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
        "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
        "я": "ya", "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
        "һ": "h", "і": "i", "ӯ": "u", "ҳ": "h", "ҷ": "j", "ӣ": "i", "ў": "u"}
_LAT = {"ş": "sh", "ç": "ch", "ž": "zh", "ý": "y", "ä": "a", "ö": "o", "ü": "u", "ň": "n",
        "ı": "i", "x": "kh", "q": "k", "ğ": "g", "ʻ": "", "ʼ": "", "'": "", "’": "", "`": ""}
_GENERIC_W = {"ост", "пункт", "остановочный", "оп", "о.п", "платформа", "пл", "станция", "ст",
              "разъезд", "рзд", "пост", "блок", "блокпост", "путевой", "пут", "обг", "обгонный",
              "бп", "пп", "обп", "вокзал", "бекеті", "бекет", "станциясы", "stansiya",
              "stansiyasi", "bekati", "bekat", "temir", "yo'l", "yol", "station", "halt",
              "railway", "платформасы", "темір", "жол", "эксп", "перев", "пасс",
              "пассажирский", "пассажирская", "№", "no"}


def lkey(name):
    """(key, skeleton) of a name: Latin letters and digits, generic words ("Ост. пункт",
    "бекеті", "(рзд)") out; the skeleton also drops vowels, y, h and doubled letters."""
    import unicodedata
    s = unicodedata.normalize("NFC", (name or "").lower())
    s = re.sub(r"\(([^)]*)\)", r" \1 ", s)
    words = [w for w in re.split(r"[\s\-–—.,/№()]+", s) if w and w not in _GENERIC_W]
    out = []
    for w in words:
        t = "".join(_CYR.get(ch, _LAT.get(ch, ch)) for ch in w)
        t = re.sub(r"[^a-z0-9]", "", t)
        t = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}.get(t, t) if len(words) > 1 else t
        out.append(t)
    k = "".join(out)
    # Turkmen and Uzbek Latin j is Russian дж / ж, w is в
    sk = k.replace("dzh", "j").replace("zh", "j").replace("w", "v").replace("kh", "k")
    sk = re.sub(r"[aeiouyh]", "", sk)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return k, sk


def words_of(name):
    """A name's words as lkey spells them, generic words left in ("Пасс" stays)."""
    import unicodedata
    s = unicodedata.normalize("NFC", (name or "").lower())
    out = []
    for w in re.split(r"[\s\-–—.,/№()']+", s):
        t = re.sub(r"[^a-z0-9]", "", "".join(_CYR.get(ch, _LAT.get(ch, ch)) for ch in w))
        if t:
            out.append(t)
    return out


class LIndex:
    """Places by lkey; find() returns the strongest tier that finds anything."""

    def __init__(self):
        self.by = defaultdict(list)

    def add(self, name, lon, lat, ref):
        k, sk = lkey(name)
        if k:
            self.by[("k", k)].append((lon, lat, ref))
        if len(sk) >= 2 and any(c.isalpha() for c in sk):
            self.by[("s", sk)].append((lon, lat, ref))

    def find(self, name):
        k, sk = lkey(name)
        got = self.by.get(("k", k), []) if k else []
        if not got and len(sk) >= 2:
            got = self.by.get(("s", sk), [])
        seen, out = set(), []
        for x in got:
            if (x[0], x[1]) not in seen:
                seen.add((x[0], x[1]))
                out.append(x)
        return out


NAME_REACH_KM = 10        # a name match may lie this much further than the tariff km say


def _fits(q1, q2, km):
    """Two placed points agree with the tariff km between them: no further apart than the
    km allow, and not on top of each other when the km say they are far apart."""
    d = dist_m(*q1, *q2) / 1000
    return km * 0.25 - 3 <= d <= km * 1.3 + 3


def drop_outliers(secs, pos, how, node_of):
    """Take out placements that disagree with the line: see placement(). Returns how many."""
    gone = 0
    for _round in range(5):
        bad = set()
        for s in secs:
            pl_ = [p for p in s["pts"] if p["esr"] in pos]
            for k, p in enumerate(pl_):
                q = pos[p["esr"]]
                nb = []
                if k > 0:
                    nb.append(pl_[k - 1])
                if k + 1 < len(pl_):
                    nb.append(pl_[k + 1])
                fit = [_fits(q, pos[n["esr"]], abs((n["km0"] or 0) - (p["km0"] or 0))) for n in nb]
                if len(nb) == 2:
                    a, b = nb
                    qa, qb = pos[a["esr"]], pos[b["esr"]]
                    kab = abs((b["km0"] or 0) - (a["km0"] or 0))
                    if dist_m(*qa, *qb) / 1000 > kab * 1.5 + 5:
                        continue                 # the neighbours disagree: nothing to judge by
                    if not any(fit):
                        bad.add(p["esr"])        # Когон's node is in the Fergana valley
                        continue
                    # on the way, but at the wrong place along it: Акча's 2021 node lies 6 km
                    # on from Ост. пункт 85 км, the tariff says 1
                    da, db = dist_m(*qa, *q), dist_m(*q, *qb)
                    if kab > 0 and da + db > 0:
                        f_exp = abs((p["km0"] or 0) - (a["km0"] or 0)) / kab
                        f_act = da / (da + db)
                        off = abs(f_exp - f_act) * (da + db) / 1000
                        if off > max(3.0, 0.3 * (da + db) / 1000):
                            bad.add(p["esr"])
                    continue
                if not fit or any(fit):
                    continue
                # a line end: its one neighbour must agree with the next one in
                n = nb[0]
                j = pl_.index(n)
                m = pl_[j + 1] if j + 1 < len(pl_) and pl_[j + 1] is not p else (
                    pl_[j - 1] if j > 0 and pl_[j - 1] is not p else None)
                if m is not None and _fits(pos[n["esr"]], pos[m["esr"]],
                                           abs((n["km0"] or 0) - (m["km0"] or 0))):
                    bad.add(p["esr"])
        if not bad:
            break
        for c in bad:
            pos.pop(c, None)
            how.pop(c, None)
            node_of.pop(c, None)
        gone += len(bad)
    return gone
EN_MIN = 0.75             # an English name must read this well as a romanisation (ru_register.en_score)


def placement(log=log):
    """Book 1's sections with their points cleaned (ru_register's rules), every point placed
    where a source allows, named, and given its country. Cached in data/raw/kz/placed.pkl
    (keyed on the inputs' times), since --borders, --convert and --timetable all need it."""
    import ru_register as rr
    from ua_register import name_keys
    srcs = [rr.newest("tr4_kniga1_*.xls"), SHARED / "sbin_esr.csv", SHARED / "sbin_osm2esr.csv",
            RU_RAW / "osm_esr.json", Path(__file__)]
    for cc in CCS:
        srcs += [raw(cc) / "osm_esr.json", raw(cc) / "osm_stations.json",
                 raw(cc) / "wd_stations.json", raw(cc) / f"{cc}_boundary.geojson"]
    key = [(str(p), int(p.stat().st_mtime), p.stat().st_size) for p in srcs if p.exists()]
    cache = SHARED / "placed.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            got = pickle.load(f)
        if got["key"] == key:
            return got
    secs, b1name = book1()
    stat = Counter()
    for s in secs:
        pts = []
        for p in s["points"]:
            p = dict(p, name=rr.clean(p["name"]), raw=p["name"], km0=p["km"][0])
            if pts and p["km0"] == pts[-1]["km0"] and rr.EXTRA_CODE.search(p["name"]):
                stat["extra code dropped"] += 1
                continue
            if pts and p["km0"] == pts[-1]["km0"] and rr.EXTRA_CODE.search(pts[-1]["name"]):
                stat["extra code dropped"] += 1
                pts[-1] = p
                continue
            pts.append(p)
        s["pts"] = pts
        s["km"] = (pts[-1]["km0"] or 0) - (pts[0]["km0"] or 0) if pts else 0
    # EXPORT CODES ("Сарыагаш (эксп.)") that stay after that: where the section also lists
    # the station itself (4 km before, at Saryagash), the export point is the border and is
    # never placed (find_borders and convert end the line at the border instead); where it
    # does not (the Kyrgyz sheet starts at "Турксиб (эксп.)"), it stands for the station
    # under its plain name, if another section lists that.
    plain = {}
    for s in secs:
        for p in s["pts"]:
            if not rr.EXTRA_CODE.search(p["name"]):
                plain.setdefault(rr.nkey(p["name"]), p["esr"])
    border_codes = set()
    for s in secs:
        here = {rr.nkey(p["name"]) for p in s["pts"] if not rr.EXTRA_CODE.search(p["name"])}
        for p in s["pts"]:
            if not rr.EXTRA_CODE.search(p["name"]):
                continue
            base = rr.nkey(rr.EXTRA_CODE.sub("", p["name"]))
            if base in here:
                border_codes.add(p["esr"])
                stat["export points at a border"] += 1
            elif base in plain:
                p["esr"] = plain[base]
                p["name"] = rr.EXTRA_CODE.sub("", p["name"]).strip()
                stat["export points read as their station"] += 1
    raw_of = {}
    for s in secs:
        for p in s["pts"]:
            raw_of.setdefault(p["esr"], p)
    esr = osm_esr()
    old = sbin_osm()
    wd = wd_stations()
    ost = load_osm_stations()
    nidx = LIndex()
    for i, s in enumerate(ost):
        for nm in s["names"]:
            nidx.add(nm, s["lon"], s["lat"], i)
    iso = sbin_iso()

    pos, how, node_of = {}, {}, {}
    for c in raw_of:
        if c in border_codes:
            continue
        rows = esr.get(c) or []
        best = ([r for r in rows if r.get("railway") in ("station", "halt")]
                or [r for r in rows if r.get("pt") == "station"] or rows)
        if best:
            pos[c], how[c], node_of[c] = (best[0]["lon"], best[0]["lat"]), "esr", best[0]
        elif c in old:
            pos[c], how[c] = old[c][:2], "osm2021"
    # A code on the wrong node (osm.sbin.ru's 2021 table puts Жаксыбулак of the Aktogay -
    # Sayak line near Kostanay): a point further from both its placed neighbours than the
    # tariff km allow, while those two agree with each other, is taken out again.
    stat["placements taken out as far off the line"] += drop_outliers(secs, pos, how, node_of)
    for _round in range(4):
        for s in secs:
            pts = s["pts"]
            for i, p in enumerate(pts):
                c = p["esr"]
                if c in pos or c in border_codes:
                    continue
                anchors = []
                for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(pts))):
                    q = pts[j]
                    if q["esr"] in pos:
                        anchors.append((pos[q["esr"]], abs((q["km0"] or 0) - (p["km0"] or 0))))
                        if len(anchors) >= 2:
                            break
                if not anchors and _round < 3:
                    continue

                def fits(lon, lat):
                    return all(dist_m(lon, lat, *a) / 1000 <= km + NAME_REACH_KM
                               for a, km in anchors)
                # An OSM station of the name first: Wikidata's coordinates are rougher
                # (Бюзмейин's is 12 km west, by Gökdepe).
                found = nidx.find(p["name"])
                cands = [x for x in found if fits(x[0], x[1])]
                w = wd.get(c)
                if cands and anchors:
                    lon, lat, ref = min(cands, key=lambda x: min(
                        dist_m(x[0], x[1], *a) for a, _k in anchors))
                    pos[c], how[c], node_of[c] = (lon, lat), "name", ost[ref]
                elif w and w["lon"] is not None and (fits(w["lon"], w["lat"]) or not anchors):
                    pos[c], how[c] = (w["lon"], w["lat"]), "wikidata"
                elif _round == 3 and not anchors:
                    # no placed neighbour at all: a name found once in the point's own
                    # country (osm.sbin.ru's) is taken
                    own = [x for x in found if ost[x[2]]["cc"].upper() == iso.get(c)]
                    if len(own) == 1:
                        lon, lat, ref = own[0]
                        pos[c], how[c], node_of[c] = (lon, lat), "name, alone", ost[ref]
    stat["name placements taken out as far off the line"] += drop_outliers(secs, pos, how, node_of)
    log(f"Book 1: {len(secs)} sections; {dict(stat)}")
    log(f"points: {len(raw_of)}; placed by " + ", ".join(
        f"{k} {v}" for k, v in Counter(how.values()).most_common()) +
        f", unplaced {sum(1 for c in raw_of if c not in pos)}")

    # names: OSM's (the national language), else Wikidata's label in it, else an OSM station
    # of a matching name within 300 m, else Book 1's own (Russian)
    import numpy as np
    from scipy.spatial import cKDTree
    near_tree = cKDTree(np.array([[s["lon"] * 0.7, s["lat"]] for s in ost]))
    shown, en_of, name_how = {}, {}, Counter()
    for c, p in raw_of.items():
        n = node_of.get(c)
        nm, en = "", ""
        if n is not None and "names" in n and isinstance(n["names"], dict) and (
                n.get("railway") in ("station", "halt") or n.get("pt") == "station") and \
                not n.get("ru_extract") and n["names"].get("name"):
            nm, en = n["names"]["name"], n["names"].get("name:en") or ""
            name_how["OSM by ESR"] += 1
        elif n is not None and "rw" in n:
            nm, en = n["name"], n["en"]
            name_how["OSM by name"] += 1
        elif wd.get(c, {}).get("nat") or wd.get(c, {}).get("ru"):
            nm = wd[c]["nat"] or wd[c]["ru"]
            name_how["Wikidata"] += 1
        elif c in pos:
            ks = set(name_keys(p["name"]))
            for j in near_tree.query_ball_point([pos[c][0] * 0.7, pos[c][1]], 0.003):
                if any(set(name_keys(x)) & ks for x in ost[j]["names"]):
                    nm, en = ost[j]["name"], ost[j]["en"]
                    name_how["OSM nearby"] += 1
                    break
        if not nm:
            nm = p["name"]
            name_how["Book 1 (Russian)"] += 1
        shown[c] = nm
        if not en and wd.get(c, {}).get("en"):
            en = rr.clean_en(wd[c]["en"])
        if en and rr.en_score(p["name"], en) >= EN_MIN:
            en_of[c] = en
    log(f"names: {dict(name_how)}; English {len(en_of)}")
    country = {c: country_at(*q) for c, q in pos.items()}
    out = {"key": key, "secs": secs, "raw_of": raw_of, "pos": pos, "how": how,
           "shown": shown, "en": en_of, "country": country, "b1name": b1name,
           "border_codes": border_codes,
           "osm_station": {c: (n.get("id"), n.get("lon"), n.get("lat")) for c, n in node_of.items()}}
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out


# ================================================================ border crossings

TRACK_KINDS = {"rail", "narrow_gauge", "preserved", "disused", "construction"}
SAME_CROSSING_M = 2500     # crossings this close are one (two sections over one bridge)


def rail_lines(cc):
    """The extract's rail ways (unclipped, data/proc/<cc>/full if there) as shapely lines."""
    import numpy as np
    import shapely
    d = proc(cc) / "full" if (proc(cc) / "full" / "ways.pkl").exists() else proc(cc)
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with np.load(d / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    order = np.argsort(cid)
    cid, cx, cy = cid[order], cx[order] / 1e7, cy[order] / 1e7
    lines, wids = [], []
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in TRACK_KINDS:
            continue
        nodes = np.asarray(nodes, dtype=np.int64)
        i = np.searchsorted(cid, nodes)
        i = np.clip(i, 0, len(cid) - 1)
        ok = cid[i] == nodes
        if ok.sum() < 2:
            continue
        lines.append(shapely.linestrings(np.c_[cx[i[ok]], cy[i[ok]]]))
        wids.append(wid)
    return lines, wids


def find_borders(log=log):
    """data/raw/kz/casia_borders.json: every pair of consecutive Book 1 points in two
    countries (one of them one of the five), with where OSM's track between them crosses the
    boundary, and an id shared by both countries' builds."""
    import shapely
    pl = placement(log)
    pos, country, raw_of = pl["pos"], pl["country"], pl["raw_of"]
    # the boundary crossings of OSM's track, per country of the five
    hits = {}
    for x in CCS:
        lines, _w = rail_lines(x)
        ring = boundary(x).boundary
        tree = shapely.STRtree(lines)
        hh = []
        for i in tree.query(ring, predicate="intersects"):
            g = lines[i].intersection(ring)
            hh += [(p.x, p.y) for p in shapely.get_parts(g) if p.geom_type == "Point"]
        hits[x] = hh
        log(f"  {x}: {len(hh)} places where rail track crosses the boundary")
    # EXITS: where a section leaves one of the five. The last point inside, and how far the
    # list runs on beyond it: to a placed point in another country (`pair`), or over points
    # nobody could place to the section's end, an export code at the border (`tail`).
    exits = {}
    for s in pl["secs"]:
        pts = s["pts"]
        for direction in (1, -1):
            seq = pts if direction == 1 else pts[::-1]
            for i, p in enumerate(seq):
                x = country.get(p["esr"]) if p["esr"] in pos else None
                if x not in CCS:
                    continue
                nxt = next((j for j in range(i + 1, len(seq)) if seq[j]["esr"] in pos), None)
                if nxt is not None and country.get(seq[nxt]["esr"]) == x:
                    continue
                if nxt is None and i == len(seq) - 1:
                    continue
                q = seq[nxt] if nxt is not None else seq[-1]
                rem = abs((q["km0"] or 0) - (p["km0"] or 0))
                other = country.get(q["esr"], "") if nxt is not None else ""
                if nxt is None and rem > 25:
                    continue           # a long unplaced tail is a gap in OSM, not a border
                key = (p["esr"], q["esr"])
                exits.setdefault(key, {"from": p["esr"], "to": q["esr"], "cc": x,
                                       "other": other, "km": rem,
                                       "kind": "pair" if nxt is not None else "tail",
                                       "sections": []})
                exits[key]["sections"].append(s["id"])
    found = []
    for key, e in sorted(exits.items()):
        lo1, la1 = pos[e["from"]]
        best = None
        far = pos.get(e["to"]) if e["kind"] == "pair" else None
        for hx, hy in hits[e["cc"]]:
            d1 = dist_m(lo1, la1, hx, hy) / 1000
            if far:
                sm = d1 + dist_m(hx, hy, *far) / 1000
                crow = dist_m(lo1, la1, *far) / 1000
                if sm > max(1.25 * crow + 3, e["km"] * 1.15 + 3):
                    continue
            else:
                sm = d1
                if d1 > e["km"] * 1.15 + 3:
                    continue
            if best is None or sm < best[0]:
                best = (sm, hx, hy)
        if best is None:
            log(f"  no crossing for {e['kind']} {raw_of[e['from']]['name']} ({e['cc']}) -> "
                f"{raw_of[e['to']]['name']} ({e['other'] or '?'}), {e['km']} km "
                f"[{', '.join(sorted(set(e['sections'])))}]")
            continue
        e["at"] = (round(best[1], 6), round(best[2], 6))
        found.append(e)
    # one id per place: exits whose crossings are within SAME_CROSSING_M share it, from
    # either side
    groups = []
    for e in sorted(found, key=lambda e: e["at"]):
        for g in groups:
            if dist_m(*g["at"], *e["at"]) <= SAME_CROSSING_M:
                g["exits"].append(e)
                break
        else:
            groups.append({"at": e["at"], "exits": [e]})
    n = Counter()
    out = []
    ne = None
    for g in groups:
        ccs = {e["cc"] for e in g["exits"]} | {e["other"] for e in g["exits"] if e["other"]}
        if len(ccs) < 2:
            # the far side is no country we read: the nearest other country in Natural
            # Earth (as borders.complete does)
            lon, lat = g["at"]
            if ne is None:
                import borders as _b
                ne = _b.countries()
            near = sorted((geom.distance(shapely.Point(lon, lat)), c)
                          for c, (_n, geom) in ne.items() if c not in ccs)
            ccs.add(near[0][1])
        g["countries"] = sorted(ccs)[:2]
    for g in sorted(groups, key=lambda g: (g["countries"], g["at"][0])):
        a, b = g["countries"]
        n[(a, b)] += 1
        gid = f"X{a.upper()}{b.upper()}{n[(a, b)]:02d}"
        names = sorted({raw_of[e["from"]]["name"] + " - " + raw_of[e["to"]]["name"]
                        for e in g["exits"]})
        out.append({"id": gid, "lon": g["at"][0], "lat": g["at"][1], "countries": [a, b],
                    "name": " – ".join(sorted(NAME_EN.get(c, c) for c in (a, b))) + " border",
                    "exits": [{k: e[k] for k in ("from", "to", "cc", "kind", "km", "sections")}
                              for e in g["exits"]],
                    "label": "; ".join(names)})
        log(f"  {gid} {g['at']} {'; '.join(names)}")
    (SHARED / "casia_borders.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               "utf-8")
    log(f"borders: {len(out)} crossings -> {SHARED / 'casia_borders.json'}")


# ================================================================ the conversion

OUT = ROOT / "data" / "raw" / "rinf"
SERVED_M = 400            # an OSM train route stop or a timetable call this close serves a point
STRETCH_KM = 8            # an unserved stop-to-stop stretch at least this long with no halt is freight
HALT_M = 400              # an OSM railway=halt this close to a point marks it a passenger halt
INTERP_TOL = 0.25         # interpolating along a trace: it must be within this share (+2 km) of the tariff km


def road_names():
    return {code: name for code, name in roads().values()}


def interpolate(cc, pl, log):
    """Points no source placed, between two placed points of one section inside the
    country: put on OSM's track between them at their share of the tariff km (most are
    "Ост. пункт 1174 км" halts, which OSM maps under other names or not at all). Returns
    {code: (lon, lat)}."""
    import build_model as bm
    import rinf
    ways, _rels, _stops, cid, cx, cy = bm.load(cc, log)
    track = rinf.Track(ways, bm.Coords(cid, cx, cy), log)
    pos = pl["pos"]
    out, stat = {}, Counter()
    for s in pl["secs"]:
        pts = s["pts"]
        placed = [i for i, p in enumerate(pts) if p["esr"] in pos or p["esr"] in out]
        for i, j in zip(placed, placed[1:]):
            if j - i < 2:
                continue
            a, b = pts[i], pts[j]
            qa = pos.get(a["esr"]) or out.get(a["esr"])
            qb = pos.get(b["esr"]) or out.get(b["esr"])
            if country_at(*qa) != cc or country_at(*qb) != cc:
                continue
            inner = [p for p in pts[i + 1:j] if p["esr"] not in pos and p["esr"] not in out
                     and p["esr"] not in pl["border_codes"]]
            if not inner:
                continue
            km = abs((b["km0"] or 0) - (a["km0"] or 0))
            sa, sb = track.snap(*qa), track.snap(*qb)
            got = track.trace(sa, sb, km) if sa is not None and sb is not None else None
            if not got or km <= 0 or abs(got[1] - km) > INTERP_TOL * km + 2:
                stat["runs not traced"] += 1
                continue
            path = got[0]
            cum = [0.0]
            for u, v in zip(path[:-1], path[1:]):
                cum.append(cum[-1] + dist_m(*u, *v))
            for p in inner:
                f = abs((p["km0"] or 0) - (a["km0"] or 0)) / km
                target = min(max(f, 0.0), 1.0) * cum[-1]
                for k in range(len(cum) - 1):
                    if cum[k + 1] >= target:
                        t = (target - cum[k]) / (cum[k + 1] - cum[k]) if cum[k + 1] > cum[k] else 0
                        u, v = path[k], path[k + 1]
                        out[p["esr"]] = (round(u[0] + t * (v[0] - u[0]), 6),
                                         round(u[1] + t * (v[1] - u[1]), 6))
                        break
            stat["points interpolated"] += len(inner)
    log(f"interpolate {cc}: {dict(stat)}")
    return out


def convert(cc, log=log):
    import numpy as np
    from scipy.spatial import cKDTree
    import ru_register as rr
    from ua_register import section_ends
    pl = placement(log)
    secs, raw_of = pl["secs"], pl["raw_of"]
    pos = dict(pl["pos"])
    how = dict(pl["how"])
    for c, q in interpolate(cc, pl, log).items():
        pos[c], how[c] = q, "interpolated"
    country = {c: (pl["country"].get(c) if c in pl["pos"] else country_at(*q))
               for c, q in pos.items()}
    ops, b2name = rr.book2()
    shown, en_of = pl["shown"], pl["en"]
    stat = Counter()
    km_kept = Counter()
    ost = load_osm_stations((cc,))

    def inside(c):
        return c in pos and country.get(c) == cc

    # --- which points are stops
    kx = 111.32 * math.cos(math.radians(43))
    with open(proc(cc) / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(proc(cc) / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    served_xy = []
    for tags, members in rels.values():
        if tags.get("type") == "route" and tags.get("route") == "train":
            for ty, ref, role in members:
                if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                    served_xy.append((stops[ref][1], stops[ref][2]))
    n_osm = len(set(served_xy))
    tp = raw(cc) / "timetable_calls.json"
    tt = json.loads(tp.read_text("utf-8")) if tp.exists() else []
    served_xy += [(x, y) for x, y in tt]
    log(f"served places: {n_osm} OSM train route stops, {len(tt)} timetable stations")
    stree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in set(served_xy)])) \
        if served_xy else None
    halts = [(s["lon"], s["lat"]) for s in ost if s["rw"] == "halt"]
    htree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in halts])) if halts else None
    ttree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in tt])) if tt else None

    def near(tree, c, m):
        q = pos.get(c)
        return bool(q and tree is not None and tree.query_ball_point(
            [q[0] * kx, q[1] * 110.57], m / 1000))
    tt_pairs = set()
    pp = raw(cc) / "timetable_pairs.json"
    if pp.exists() and ttree is not None:
        tt_pairs = {tuple(p) for p in json.loads(pp.read_text("utf-8"))}

    def call_ids(c):
        q = pos.get(c)
        if not q or ttree is None:
            return set()
        return set(ttree.query_ball_point([q[0] * kx, q[1] * 110.57], SERVED_M / 1000))

    def consecutive(c1, c2):
        a, b = call_ids(c1), call_ids(c2)
        return any((min(x, y), max(x, y)) in tt_pairs for x in a for y in b)
    kind = {}
    for c, p in raw_of.items():
        if near(ttree, c, SERVED_M):
            # a timetable call is the evidence: Book 2 gives Вахш, Сангтуда and Дангара no
            # passenger operation, and Dushanbe - Kulyab trains call there
            kind[c] = "served"
        elif not rr.passenger_op(ops.get(c)):
            kind[c] = "none"
        elif near(stree, c, SERVED_M):
            kind[c] = "served"
        elif re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) or near(htree, c, HALT_M):
            kind[c] = "halt"
        else:
            kind[c] = "flag"
    log(f"point kinds (all sheets): {dict(Counter(kind.values()))}")

    # --- sections: owners, pieces inside the country, unserved stretches
    owner = {}
    order = sorted(secs, key=lambda s: (rr.TYPE_RANK.get(s["type"], 9), s["id"]))
    for s in order:
        for a, b in zip(s["pts"], s["pts"][1:]):
            owner.setdefault(frozenset((a["esr"], b["esr"])), s["id"])
    where = defaultdict(list)
    for s in secs:
        for i, p in enumerate(s["pts"]):
            where[p["esr"]].append((s["id"], i, p["km0"] or 0))
    coarse = set()
    for s in secs:
        for a, b in zip(s["pts"], s["pts"][1:]):
            km = (b["km0"] or 0) - (a["km0"] or 0)
            at_b = {sid: (i, k) for sid, i, k in where[b["esr"]]}
            for sid, i, k in where[a["esr"]]:
                if sid == s["id"] or sid not in at_b:
                    continue
                j, kb = at_b[sid]
                if abs(j - i) >= 2 and abs(abs(kb - k) - km) <= 0.15 * km + 2:
                    coarse.add((s["id"], frozenset((a["esr"], b["esr"]))))
                    break
    rnames = road_names()
    sheet_road = {sh: code for sh, (code, _n) in roads().items()}
    rows, names, used, stop_codes = [], {}, set(), set()
    for s in secs:
        if s["km"] == 0:
            continue
        # An unplaced point counts as inside when its placed neighbours both are.
        pts = s["pts"]
        ins = {}
        for i, p in enumerate(pts):
            c = p["esr"]
            if c in pos:
                ins[c] = inside(c)
                continue
            if c in pl["border_codes"]:
                ins[c] = False
                continue
            prev = next((pts[j]["esr"] for j in range(i - 1, -1, -1) if pts[j]["esr"] in pos), None)
            nxt = next((pts[j]["esr"] for j in range(i + 1, len(pts)) if pts[j]["esr"] in pos), None)
            ins[c] = bool(prev and nxt and inside(prev) and inside(nxt))
        pairs = []
        for a, b in zip(pts, pts[1:]):
            key = frozenset((a["esr"], b["esr"]))
            km = (b["km0"] or 0) - (a["km0"] or 0)
            if a["esr"] == b["esr"]:
                continue
            if not (ins[a["esr"]] and ins[b["esr"]]):
                if ins[a["esr"]] or ins[b["esr"]]:
                    km_kept["crossing a border (border rows)"] += abs(km)
                continue
            if owner[key] != s["id"]:
                stat["pairs on another section"] += 1
                km_kept["on another section"] += abs(km)
                continue
            if (s["id"], key) in coarse:
                stat["pairs another section lists finer"] += 1
                km_kept["listed finer elsewhere"] += abs(km)
                continue
            pairs.append((a, b, km))
        if not pairs:
            continue
        road = sheet_road[s["sheet"]]
        is_stop = {}
        runs, cur = [], []
        for a, b, km in pairs:
            if cur and a["esr"] != cur[-1][1]["esr"]:
                runs.append(cur)
                cur = []
            cur.append((a, b, km))
        runs.append(cur)
        clone, stretch_idx = set(), set()
        idx0 = 0
        for run in runs:
            pts_run = [run[0][0]] + [b for _a, b, _k in run]
            stop_at = [i for i, p in enumerate(pts_run) if kind[p["esr"]] == "served"]
            cuts = sorted(set([0, len(pts_run) - 1] + stop_at))
            for u, v in zip(cuts, cuts[1:]):
                inner = pts_run[u:v + 1]
                km = sum(abs(k) for _a, _b, k in run[u:v])
                has_halt = any(kind[p["esr"]] == "halt" for p in inner)
                for p in inner:
                    if kind[p["esr"]] == "served" or (has_halt and kind[p["esr"]] in ("halt", "flag")):
                        is_stop[p["esr"]] = True
                ends_stop = (kind[inner[0]["esr"]] == "served" and kind[inner[-1]["esr"]] == "served")
                if (not has_halt and ends_stop and km >= STRETCH_KM and v - u >= 1
                        and not consecutive(inner[0]["esr"], inner[-1]["esr"])):
                    clone |= {inner[0]["esr"], inner[-1]["esr"]}
                    stretch_idx |= set(range(idx0 + u, idx0 + v))
                    stat["unserved stretches with no halt left to the timetable"] += 1
                    km_kept["stretches left to the timetable"] += km
            idx0 += len(run)
        flat = [x for run in runs for x in run]
        for k, (a, b, km) in enumerate(flat, 1):
            ca, cb = a["esr"], b["esr"]
            if k - 1 in stretch_idx:
                ca = f"{ca}@{s['id']}" if ca in clone else ca
                cb = f"{cb}@{s['id']}" if cb in clone else cb
            rows.append({"sol": f"{s['id']}:{k}", "line": s["id"], "a": f"esr:{ca}",
                         "b": f"esr:{cb}", "len": str(abs(km)), "im": road,
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
            km_kept["kept"] += abs(km)
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": road, "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": road, "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        stop_codes |= {c for c, v in is_stop.items() if v}
        by_key = {}
        for p in s["pts"]:
            by_key.setdefault(rr.nkey(p["name"]), p)
            by_key.setdefault(rr.nkey(rr.EXTRA_CODE.sub("", p["name"])), p)
        pa, pb, vias = section_ends(s, by_key)

        def nm_(p):          # a section may end at a border's export code: "Сарыагаш (эксп.)"
            return rr.EXTRA_CODE.sub("", shown.get(p["esr"], p["name"])).strip()
        name = f"{nm_(pa)} — {nm_(pb)}"
        ea, eb = en_of.get(pa["esr"], ""), en_of.get(pb["esr"], "")
        name_en = f"{ea} — {eb}" if ea and eb else ""
        if vias:
            if all(vias):
                name += f" (через {', '.join(nm_(v) for v in vias)})"
                ven = [en_of.get(v["esr"], "") for v in vias]
                name_en = f"{name_en} (via {', '.join(ven)})" if name_en and all(ven) else ""
            else:
                name += " (2)"
                name_en = f"{name_en} (2)" if name_en else ""
        names[s["id"]] = {"name": name, "name_en": name_en, "type": s["type"],
                          "sheet": s["sheet"], "road": road, "operator": rnames[road],
                          "tariff_km": s["km"], "header": s["name"]}
    # --- border rows: from the last point inside to the crossing (find_borders)
    border_rows = []
    bp = SHARED / "casia_borders.json"
    crossings = json.loads(bp.read_text("utf-8")) if bp.exists() else []
    for x in crossings:
        if cc not in x["countries"]:
            continue
        for e in x["exits"]:
            if e["cc"] != cc or e["from"] not in used:
                continue
            sid = next((t for t in e["sections"] if t in names), None)
            if sid is None:
                continue
            q = pos[e["from"]]
            d1 = dist_m(*q, x["lon"], x["lat"]) / 1000
            if e["kind"] == "pair" and e["to"] in pos:
                d2 = dist_m(x["lon"], x["lat"], *pos[e["to"]]) / 1000
                km = e["km"] * d1 / (d1 + d2) if d1 + d2 > 0 else d1
            else:
                km = e["km"]
            km = max(km, d1 * 1.05, 0.3)
            rows.append({"sol": f"{sid}:{x['id']}:{e['from']}", "line": sid,
                         "a": f"esr:{e['from']}", "b": f"x:{x['id']}", "len": f"{km:.2f}",
                         "im": names[sid]["road"],
                         "label": f"{shown.get(e['from'], '')} - {x['name']}"})
            if not any(r["op"] == f"x:{x['id']}" for r in border_rows):
                border_rows.append({"op": f"x:{x['id']}", "uopid": x["id"], "name": x["name"],
                                    "type": "90", "lon": x["lon"], "lat": x["lat"]})
            stat["border rows"] += 1
    cnt = Counter(e["name"] for e in names.values())
    for sid, e in names.items():
        if cnt[e["name"]] > 1:
            e["name"] += f" ({sid})"
            if e["name_en"]:
                e["name_en"] += f" ({sid})"
    log(f"{cc}: sections {dict(stat)}")
    log(f"{cc}: tariff km { {k: round(v) for k, v in km_kept.items()} }")
    pts_out = []
    for c in sorted(used):
        base = c.split("@")[0]
        p = raw_of[base]
        q = pos.get(base)
        typ = ("70" if re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) else "10") \
            if base in stop_codes and base == c else "80"
        r = {"op": f"esr:{c}", "uopid": f"{cc.upper()}{c}", "name": shown.get(base, p["name"]),
             "type": typ, "placed_by": how.get(base, "")}
        if en_of.get(base) and not c.endswith("x"):
            r["name_en"] = en_of[base]
        if c.endswith("x"):
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    pts_out += border_rows
    d = OUT / cc
    d.mkdir(parents=True, exist_ok=True)
    stampd = {"endpoint": f"Тарифное руководство № 4, {pl['b1name']}, {b2name}",
              "fetched": date.today().isoformat()}
    (d / "sections.json").write_text(json.dumps({**stampd, "rows": rows}, ensure_ascii=False),
                                     "utf-8")
    (d / "points.json").write_text(json.dumps({**stampd, "rows": pts_out}, ensure_ascii=False),
                                   "utf-8")
    (d / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    log(f"{cc}: wrote {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
        f"({sum(1 for r in pts_out if r['type'] in ('10', '70'))} stops, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced) -> {d}")


# No operator colours its tariff sections, nor does a widely used map: one picked colour per
# administration (as Russia's and Ukraine's), no two neighbouring countries alike; Russia's
# railways' sections in Kazakhstan keep the colour Russia's build gives that railway.
ROAD_COLOUR = {
    "68": ("2F7FD8", "blue"),       # Қазақстан темір жолы
    "73": ("3FA34D", "green"),      # Oʻzbekiston temir yoʻllari
    "70": ("D7263D", "red"),        # Кыргыз темир жолу
    "74": ("E08E0B", "orange"),     # Роҳи оҳани Тоҷикистон
    "75": ("1E9E8F", "teal"),       # Türkmendemirýollary
}


def colours(cc):
    """colours/<cc>.csv: one row per register line in names.json, its administration's
    colour (`picked`)."""
    import csv
    import ru_register as rr
    names = json.loads((OUT / cc / "names.json").read_text("utf-8"))
    rows = []
    for _sid, e in sorted(names.items()):
        col, hue = ROAD_COLOUR.get(e["road"]) or rr.ROAD_COLOUR.get(e["road"], ("888888", "grey"))
        rows.append({"line": e["name"], "operator": e["operator"], "colour": "#" + col,
                     "source": "picked", "url": "",
                     "note": f"{e['operator']}: one {hue} per railway administration"})
    path = ROOT / "colours" / f"{cc}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["line", "operator", "colour", "source", "url", "note"])
        w.writeheader()
        w.writerows(rows)
    log(f"--colours {cc}: {len(rows)} register lines -> {path}")


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(Path(path).name, log)
    import rinf
    return rinf.build(path, log)


# ================================================================ rinf.py settings

# Lines to draw greyed, as not running, that gtfs_served cannot close itself (none so far: the
# Turkmen lines to Iran and Afghanistan with no train are junction-ended and dropped anyway).
SUSPENDED = {}

def country_conf(cc):
    names_file = OUT / cc / "names.json"
    cache = {}

    def _names():
        if "n" not in cache:
            cache["n"] = json.loads(names_file.read_text("utf-8")) if names_file.exists() else {}
        return cache["n"]

    def id_name(lid, _uop=None):
        e = _names().get(lid)
        return (e["name"], e.get("name_en") or "") if e else None

    def station_en(point, _name):
        if "en" not in cache:
            pf = OUT / cc / "points.json"
            cache["en"] = {r["uopid"]: r["name_en"] for r in
                           (json.loads(pf.read_text("utf-8"))["rows"] if pf.exists() else [])
                           if r.get("name_en")}
        return cache["en"].get(point.get("uopid"), "")

    def stop_name(point):
        if point.get("type") in ("10", "70"):
            return point.get("name") or None
        return None

    def suspended(_ref, lids):
        return any(l.split("#")[0] in SUSPENDED.get(cc, ()) for l in lids)
    return {
        "suspended": suspended,
        "stop_name": stop_name,
        "iso3": ISO3[cc], "langs": [LANG[cc], "ru", "en"],
        "osm_rel": lambda _tags: None,
        "id_name": id_name,
        "tol_abs": 1.5,
        "direct_near_m": 1500,
        # End sections where other lines meet too: a long freight-only stretch between two
        # stops would otherwise be one section, and one piece OSM lacks (Ashgabat's northern
        # bypass, Бюзмейин - Рзд № 0001) left out Gypjak - Ashgabat with it, which cut the
        # Karakum line to Dashoguz off from Ashgabat.
        "cut_at_junctions": True,
        "im": road_names(),
        "station_en": station_en,
    }


# ================================================================ the timetable crawl (KTZ)

KTZ = "https://bilet.railways.kz"
KTZ_DIR = SHARED / "ktz"
CRAWL_DELAY = 1.0          # seconds between requests
# Wednesday and Saturday: odd and even days, a weekday and a weekend day. A train running on
# other days only still shows at the stations it reaches on these (a long run spans days).
CRAWL_DATES = ("2026-10-07", "2026-10-10")


class Ktz:
    """bilet.railways.kz with a session cookie (its station search answers only inside one)."""

    def __init__(self):
        import http.cookiejar
        import urllib.request
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.last = 0.0
        self.get("/sale/default/station/schedule?_locale=ru")

    def _req(self, path, data=None, xhr=False, tries=4):
        import urllib.parse
        import urllib.request
        for k in range(tries):
            wait = CRAWL_DELAY - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            h = {"User-Agent": USER_AGENT}
            if xhr:
                h["X-Requested-With"] = "XMLHttpRequest"
                h["Referer"] = KTZ + "/sale/default/route/search"
            body = urllib.parse.urlencode(data).encode() if data is not None else None
            try:
                with self.op.open(urllib.request.Request(KTZ + path, data=body, headers=h),
                                  timeout=90) as r:
                    return r.read().decode("utf-8", errors="replace")
            except Exception as e:                                   # noqa: BLE001
                if getattr(e, "code", None) in (404, 500):
                    return None
                log(f"  retry {k}: {path[:90]} {type(e).__name__} {str(e)[:80]}")
                time.sleep(10 * (k + 1))
        return None

    def get(self, path):
        return self._req(path)

    def post(self, path, data):
        return self._req(path, data, xhr=True)

    def search(self, q):
        import urllib.parse
        t = self._req("/api/v1/ktj/station/search?q=" + urllib.parse.quote(q), xhr=True)
        try:
            return json.loads(t)["results"] if t else []
        except ValueError:
            return []


def _cells(tr):
    import html as _h
    out = []
    for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S):
        spans = re.findall(r"<span[^>]*>(.*?)</span>", td, re.S)
        val = spans[-1] if spans else td
        out.append(re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", val))).strip())
    return out


def parse_schedule(t):
    """A station schedule page -> [{kind, dep, run, arr, num, from, to, reg}]: kind is the
    page's group (Отправляющиеся / Транзитные / Прибывающие поезда); dep is the departure from
    this station (from the origin for an arriving train), run the time on to the end."""
    rows, kind = [], ""
    i0 = t.find("station_schedule_form")
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t[i0:], re.S):
        m = re.search(r'ribbon label">\s*([^<]+?)\s*<', tr)
        if m:
            kind = m.group(1)
            continue
        c = _cells(tr)
        if len(c) == 7 and re.fullmatch(r"\d{2}:\d{2}:\d{2}", c[0] or "") and c[3]:
            rows.append({"kind": kind, "dep": c[0], "run": c[1], "arr": c[2], "num": c[3],
                         "from": c[4], "to": c[5], "reg": c[6]})
    return rows


def parse_route(t):
    """A train's route popup -> [(station name, arrival, departure)], main route only."""
    import html as _h
    i1 = t.find('<div class="mobile only">')
    body = t[:i1 if i1 > 0 else len(t)]
    calls = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) != 4 or "<strong>Станция</strong>" in tr:
            continue
        nm = re.sub(r"<strong[^>]*>.*?</strong>", "", tds[0], flags=re.S)
        nm = re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", nm))).strip()
        vals = [re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", x)) for x in tds[1:]]
        calls.append((nm, vals[0], vals[2]))
    return calls


def express_codes():
    """Express code -> (ESR code, name, iso) for every station of the five (and the Russian
    railways' stations in Kazakhstan) in osm.sbin.ru's ESR list."""
    import csv
    out = {}
    with open(SHARED / "sbin_esr.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            iso = (r["iso3166"] or "")[:2]
            if iso in ("KZ", "UZ", "KG", "TJ", "TM") and r["express"]:
                out[r["express"]] = (r["esr"], r["name"], iso)
    return out


def crawl(limit=None):
    """Every station's schedule for CRAWL_DATES (data/raw/kz/ktz/sched/<code>_<date>.json),
    then one route per train (number, origin, destination) (data/raw/kz/ktz/routes.json).
    Files already there are not fetched again."""
    k = Ktz()
    sd = KTZ_DIR / "sched"
    sd.mkdir(parents=True, exist_ok=True)
    codes = sorted(express_codes())
    n = 0
    for di, d in enumerate(CRAWL_DATES):
        dd = date.fromisoformat(d).strftime("%d-%m-%Y")
        for i, c in enumerate(codes):
            p = sd / f"{c}_{d}.json"
            if p.exists():
                continue
            if limit and n >= limit:
                break
            n += 1
            t = k.get(f"/sale/default/station/schedule?_locale=ru&station_schedule_form%5B"
                      f"station%5D={c}&station_schedule_form%5Bdate%5D={dd}")
            if t is None:
                continue
            p.write_text(json.dumps(parse_schedule(t), ensure_ascii=False), "utf-8")
            if n % 100 == 0:
                log(f"  schedules: {n} fetched ({d}, {i + 1}/{len(codes)})")
    log(f"schedules: {n} fetched")
    # one route per train: from a station it departs from, on a crawled date
    rp = KTZ_DIR / "routes.json"
    routes = json.loads(rp.read_text("utf-8")) if rp.exists() else {}
    want = {}
    for p in sorted(sd.glob("*.json")):
        c, d = p.stem.split("_")
        for r in json.loads(p.read_text("utf-8")):
            if r["kind"].startswith("Прибыва"):
                continue
            key = f"{r['num']}|{r['from']}|{r['to']}"
            want.setdefault(key, []).append((r["num"], f"{d} {r['dep']}", c))
    got = 0
    for key, tries in sorted(want.items()):
        if key in routes and routes[key]["calls"]:
            continue
        # The popup answers 500 for some (station, time) a schedule row gives: try others.
        for num, when, c in tries[:5]:
            t = k.post("/api/v1/ktj/train/route/html", {"trainNumber": num, "date": when,
                                                         "station": c})
            routes[key] = {"num": num, "when": when, "station": c,
                           "calls": parse_route(t) if t else []}
            if routes[key]["calls"]:
                break
        got += 1
        if got % 50 == 0:
            rp.write_text(json.dumps(routes, ensure_ascii=False), "utf-8")
            log(f"  routes: {got} fetched of {len(want)}")
    rp.write_text(json.dumps(routes, ensure_ascii=False), "utf-8")
    log(f"routes: {got} fetched, {len(routes)} trains, "
        f"{sum(1 for v in routes.values() if not v['calls'])} with no calls")


def crawl_names():
    """Every station name a crawled route calls at, looked up in KTZ's station search for its
    Express code (data/raw/kz/ktz/names.json): the route popups give names only."""
    rp = KTZ_DIR / "routes.json"
    routes = json.loads(rp.read_text("utf-8"))
    np_ = KTZ_DIR / "names.json"
    got = json.loads(np_.read_text("utf-8")) if np_.exists() else {}
    want = sorted({c[0] for r in routes.values() for c in r["calls"]} - set(got))
    k = Ktz()
    for i, nm in enumerate(want):
        res = k.search(nm)
        got[nm] = [r["value"] for r in res if r.get("name", "").strip().upper() == nm.upper()]
        if not got[nm]:
            # the popup cuts long names ("САРЫАГАШ(ГР"): a result that starts with it
            got[nm] = [r["value"] for r in res
                       if r.get("name", "").strip().upper().startswith(nm.upper())][:3]
        if i % 100 == 99:
            np_.write_text(json.dumps(got, ensure_ascii=False), "utf-8")
            log(f"  names: {i + 1} of {len(want)}")
    np_.write_text(json.dumps(got, ensure_ascii=False), "utf-8")
    log(f"names: {len(want)} looked up, {sum(1 for v in got.values() if v)} found")


# ================================================================ the timetable feed

TT_FROM, TT_TO = date(2026, 10, 5), date(2026, 11, 29)    # the feed's eight weeks
MIN_DAYS = 7               # a train running fewer days than this in the eight weeks is left out
SKIP_KM = 150.0            # the DP: leaving a call unmatched costs this many km of path ...
HOP_WEIGHT = 0.1           # ... where a km between calls weighs this (Talgo calls 300 km apart)
MAX_HOP_KM = 1200.0

WEEKDAY = {"ПН": 1, "ВТ": 2, "СР": 3, "ЧТ": 4, "ПТ": 5, "СБ": 6, "ВС": 7}


def running_days(regs):
    """The days in TT_FROM..TT_TO a train runs, from KTZ's "Регулярность курсирования"
    strings (several, from different stations and dates): "" daily, ЧЕТ / НЕЧ even / odd
    dates, "ПО 135" weekdays (1 Monday), "ДАТЫ:06.10,..." those dates (a train listed by
    dates every day of the crawl window runs daily), "ПО 24.10 ЕЖ" daily until then."""
    days = set()
    d0, n = TT_FROM, (TT_TO - TT_FROM).days + 1
    every = [d0 + timedelta(days=i) for i in range(n)]
    for reg in regs:
        r = (reg or "").strip().upper()
        if not r or r.startswith("ЕЖ"):
            days |= set(every)
        elif r.startswith("ЧЕТ"):
            days |= {d for d in every if d.day % 2 == 0}
        elif r.startswith("НЕЧ"):
            days |= {d for d in every if d.day % 2 == 1}
        elif re.match(r"ПО\s+\d\d\.\d\d", r):
            m = re.match(r"ПО\s+(\d\d)\.(\d\d)", r)
            until = date(2026, int(m.group(2)), int(m.group(1)))
            days |= {d for d in every if d <= until}
        elif re.match(r"ПО\s+[1-7]+", r):
            wd = {int(c) for c in re.match(r"ПО\s+([1-7]+)", r).group(1)}
            days |= {d for d in every if d.isoweekday() in wd}
        elif r.startswith("ДАТЫ"):
            ds = []
            for dd, mm in re.findall(r"(\d\d)\.(\d\d)", r):
                ds.append(date(2026, int(mm), int(dd)))
            span = (max(ds) - min(ds)).days + 1 if ds else 0
            if ds and len(ds) == span:
                days |= set(every)            # listed for every day shown: daily
            elif ds and span == 2 * len(ds) - 1:
                par = ds[0].day % 2
                days |= {d for d in every if d.day % 2 == par}
            elif ds:
                # The page lists the next three dates of an irregular pattern: run it on at
                # the same rate (07.10, 09.10, 13.10 is about every other day).
                step = max(1, round(span / len(ds)))
                days |= set(every[::step])
        else:
            days |= set(every)
    return days


def same_place_name(a, b):
    """Do two spellings of a station name agree (KTZ's cut capitals against Book 1's)? The
    first word of one starts the other's, or the skeletons agree."""
    wa, wb = words_of(a), words_of(b)
    if not wa or not wb:
        return False
    x, y = wa[0][:6], wb[0][:6]
    return x.startswith(y[:4]) or y.startswith(x[:4]) or lkey(a)[1] == lkey(b)[1]


ABBR_GENERIC = {"vokz", "vokzal", "pass", "pas", "p", "gl", "gr", "st", "ost", "rzd"}


class Places:
    """Every place a call can be: placed register points (by Book 1's and the shown name)
    and OSM stations of the five, found by lkey, else by an lkey that starts with the call's
    (KTZ's names are cut at about 12 letters)."""

    def __init__(self, pl, pos):
        self.items = []          # (lon, lat, name, esr or None)
        self.idx = LIndex()
        self.words, self.cache = [], {}
        for c, q in pos.items():
            p = pl["raw_of"][c]
            self.items.append((q[0], q[1], pl["shown"].get(c, p["name"]), c))
            i = len(self.items) - 1
            nms = {p["name"], pl["shown"].get(c, "")} - {""}
            for nm in nms:
                self.idx.add(nm, q[0], q[1], i)
            self.words.append((i, [words_of(nm) for nm in nms]))
        for s in load_osm_stations():
            self.items.append((s["lon"], s["lat"], s["name"], None))
            i = len(self.items) - 1
            for nm in s["names"]:
                self.idx.add(nm, s["lon"], s["lat"], i)
            self.words.append((i, [words_of(nm) for nm in s["names"]]))
        self.keys = sorted(k for t, k in self.idx.by if t == "k")

    def find(self, name):
        if name not in self.cache:
            self.cache[name] = self._find(name)
        return self.cache[name]

    def _find(self, name):
        got = [r for _x, _y, r in self.idx.find(name)]
        if got:
            return got
        import bisect
        k, _sk = lkey(name)
        if len(k) < 5:
            return []
        out = []
        i = bisect.bisect_left(self.keys, k)
        while i < len(self.keys) and self.keys[i].startswith(k):
            out += [r for _x, _y, r in self.idx.by[("k", self.keys[i])]]
            i += 1
        if not out and len(k) >= 6:
            # "НУРЛЫ ЖОЛ" is OSM's "Астана Нұрлы Жол"
            for key in self.keys:
                if k in key:
                    out += [r for _x, _y, r in self.idx.by[("k", key)]]
        if not out:
            # word by word, each an abbreviation: "КАРАГАНД П" is "Караганда-Пассажирская"
            cw = words_of(name)
            if cw and len(cw[0]) >= 4:
                for i, wl in self.words:
                    if any(len(iw) >= len(cw) and all(b.startswith(a) for a, b in zip(cw, iw))
                           for iw in wl):
                        out.append(i)
            short = [w for w in cw if w not in ABBR_GENERIC]
            if not out and short and short != cw:
                # "АНГРЕН ВОКЗ", "ОАЗИС (ГР)" (the border), "ИСИЛЬКУЛЬ ПАС"
                return self._find(" ".join(short))
        return out[:40]


# Express code prefixes of the five networks (Kazakhstan 27, Uzbekistan 29, Kyrgyzstan 59,
# Tajikistan 66, Turkmenistan 67)
OWN_EXPRESS = {"27", "29", "59", "66", "67"}
MAX_KMH = 200.0           # no train goes faster than this between two calls (crow-fly)


def call_minutes(times):
    """[(arr, dep)] "HH:MM:SS" per call -> minutes since the first call, days rolled over
    where a time goes backwards; None where a call has no time."""
    out, prev, add = [], None, 0
    for arr, dep in times:
        t = arr or dep
        if not t:
            out.append(None)
            continue
        h, m = int(t[:2]), int(t[3:5])
        v = h * 60 + m + add
        if prev is not None and v < prev:
            add += 1440
            v += 1440
        out.append(v)
        prev = v
        if dep and arr and dep != arr:
            h2, m2 = int(dep[:2]), int(dep[3:5])
            v2 = h2 * 60 + m2 + add
            if v2 < v:
                add += 1440
                v2 += 1440
            prev = v2
    return out


def match_calls(calls, cands, places, mins=None):
    """One place per call (or none) so that the train's path is shortest (ua_register's DP):
    calls [(name, ...)], cands[i] = candidate item indexes. With `mins` (call times), two
    calls can be no further apart than MAX_KMH allows: a Russian "Тайга" or "Курган" in the
    route is not the Kazakh place of that name. Returns {call index: item}."""
    n = len(calls)
    best = [dict() for _ in range(n)]
    for i in range(n):
        for r in cands[i]:
            bc, bp = SKIP_KM * i, None
            for j in range(max(0, i - 6), i):
                lim = MAX_HOP_KM
                if mins and mins[i] is not None and mins[j] is not None:
                    lim = min(lim, (mins[i] - mins[j]) / 60 * MAX_KMH + 20)
                for r2, (c2, _p) in best[j].items():
                    d = dist_m(*places.items[r][:2], *places.items[r2][:2]) / 1000
                    if d > lim:
                        continue
                    v = c2 + d * HOP_WEIGHT + SKIP_KM * (i - j - 1)
                    if v < bc:
                        bc, bp = v, (j, r2)
            best[i][r] = (bc, bp)
    end, ec = None, SKIP_KM * n
    for i in range(n):
        for r, (c, _p) in best[i].items():
            v = c + SKIP_KM * (n - 1 - i)
            if v < ec:
                end, ec = (i, r), v
    chosen = {}
    while end is not None:
        i, r = end
        chosen[i] = r
        end = best[i][r][1]
    if mins:
        # a lone match among unmatched calls (097С's Russian "Курган" as a Kyrgyz station)
        # is dropped where it is too far, in the time, from the next matched call
        changed = True
        while changed:
            changed = False
            ks = sorted(chosen)
            for i, j in zip(ks, ks[1:]):
                if mins[i] is None or mins[j] is None:
                    continue
                d = dist_m(*places.items[chosen[i]][:2], *places.items[chosen[j]][:2]) / 1000
                if d <= (mins[j] - mins[i]) / 60 * MAX_KMH + 20:
                    continue
                lone = [k for k in (i, j) if (k - 1 not in chosen) and (k + 1 not in chosen)]
                k = lone[0] if lone else (i if len(ks) and i == ks[0] else j)
                del chosen[k]
                changed = True
                break
    return chosen


def hand_trains():
    """Trains no KTZ page shows, from the operators' own timetables (casia_sources.md):
    [{num, title, calls: [names], days: set, src}]. Turkmenistan: railway.gov.tm's schedule
    page (every route with its stops and days); Tajikistan, Kyrgyzstan, Uzbekistan's suburban
    trains: written out below from railway.tj, the Kyrgyz press and tashtrans.uz."""
    import html as _h
    out = []
    every = {TT_FROM + timedelta(days=i) for i in range((TT_TO - TT_FROM).days + 1)}
    p = raw("tm") / "railway_gov_tm_schedule.html"
    if p.exists():
        t = p.read_text("utf-8", errors="replace")
        t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
        s = re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", t)))
        s = s[:s.find("Train Route")] if "Train Route" in s else s
        days_of = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
        heads = list(re.finditer(r"(\S+) → (\S+) (\d+) (?:Tiz otly|Ýolagçy otly)", s))
        for m, nx in zip(heads, heads[1:] + [None]):
            body = s[m.end():nx.start() if nx else len(s)].strip()
            wd = {days_of[d] for d, rest in re.findall(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \w+ "
                                                       r"(\d\d:\d\d|– No service)", body)
                  if rest != "– No service"}
            body = body[body.find(" stop"):] if " stop" in body else body
            body = re.sub(r"^ stops? ", "", body)
            stops = re.findall(r"([\d,]+) km (.+?)(?= \d+ min stop| [\d,]+ km |$)", body)
            names = [nm.strip() for _k, nm in stops]
            if len(names) >= 2:
                out.append({"num": m.group(3), "title": f"{m.group(1)} → {m.group(2)}",
                            "calls": names, "days": {d for d in every if d.isoweekday() in wd},
                            "src": "railway.gov.tm", "cc": "tm"})
    for num, title, calls, wd in HAND:
        days = {d for d in every if d.isoweekday() in wd} if isinstance(wd, set) else set(wd)
        out.append({"num": num, "title": title, "calls": calls, "days": days, "src": "hand",
                    "cc": ""})
    return out


ALL_DAYS = {1, 2, 3, 4, 5, 6, 7}
SUMMER_2026 = [date(2026, 6, 26) + timedelta(days=i) for i in range(80)]   # 26 June - 13 Sept
# (number, title, calls in order, weekdays or a list of dates). Sources in casia_sources.md.
HAND = [
    # Tajikistan, railway.tj "Расписание движения пассажирских поездов на 2026 г."
    ("6373", "Душанбе – Пахтаабад", ["Душанбе I", "Душанбе II", "Айни", "Ханака", "Чептура",
                                      "Регар", "Пахтаабад"], ALL_DAYS),
    ("6374", "Пахтаабад – Душанбе", ["Пахтаабад", "Регар", "Чептура", "Ханака", "Айни",
                                      "Душанбе II", "Душанбе I"], ALL_DAYS),
    ("602", "Душанбе – Куляб", ["Душанбе I", "Рохаты", "Вахдат", "Бостон", "Яван", "Вахш",
                                "Хатлон", "Сангтуда", "Дангара", "Хулбук", "Куляб"], {6}),
    ("601", "Куляб – Душанбе", ["Куляб", "Хулбук", "Дангара", "Сангтуда", "Хатлон", "Вахш",
                                "Яван", "Бостон", "Вахдат", "Рохаты", "Душанбе I"], {7}),
    # Dushanbe - Kanibadam, weekly, through Uzbekistan and in at Spitamen, as the old Tashkent -
    # Fergana line ran (Khavast - Bekabad - Khujand - Kanibadam): its Tajik part only
    ("367", "Душанбе – Канибадам", ["Спитамен", "Худжанд", "Канибадам"], {4}),
    ("368", "Канибадам – Душанбе", ["Канибадам", "Худжанд", "Спитамен"], {3}),
    # Kyrgyzstan: the two daily suburban trains (Kyrgyz Temir Zholu, vb.kg / kaktus.media)
    # and the Issyk-Kul summer train 608/609 (economist.kg, 2026: daily 26 June - 13 September)
    ("6063", "Бишкек-2 – Каинды", ["Бишкек II", "Сокулук", "Шопоков", "Беловодская",
                                   "Кара-Балта", "Каинды"], ALL_DAYS),
    ("6064", "Каинды – Бишкек-2", ["Каинды", "Кара-Балта", "Беловодская", "Шопоков", "Сокулук",
                                   "Бишкек II"], ALL_DAYS),
    ("6050", "Бишкек-1 – Токмок", ["Бишкек I", "Бишкек II", "Аламедин", "Кант", "Ивановка",
                                   "Токмок"], ALL_DAYS),
    ("6051", "Токмок – Бишкек-1", ["Токмок", "Ивановка", "Кант", "Аламедин", "Бишкек II",
                                   "Бишкек I"], ALL_DAYS),
    ("608", "Бишкек-2 – Балыкчы", ["Бишкек II", "Аламедин", "Кант", "Токмок", "Кемин",
                                   "Балыкчы"], SUMMER_2026),
    ("609", "Балыкчы – Бишкек-2", ["Балыкчы", "Кемин", "Токмок", "Кант", "Аламедин",
                                   "Бишкек II"], SUMMER_2026),
    # Uzbekistan's suburban electric trains from Tashkent (tashtrans.uz, updated 2026-09-08)
    ("", "Ташкент – Ходжикент", ["Тошкент-Йуловчи", "Тошкент", "Салор", "Кибрай", "Чирчик",
                                "Хужакент"], ALL_DAYS),
    ("", "Ташкент – Хаваст", ["Тошкент-Йуловчи", "Тошкент Жанубий", "Янгийул", "Чиноз",
                             "Сирдаре", "Гулистон", "Янгиер", "Ховос"], ALL_DAYS),
    ("", "Ташкент – Ангрен", ["Тошкент-Йуловчи", "Тукимачи", "Ангрен"], ALL_DAYS),
]


def timetable(log=log):
    """The GTFS feed per country (data/raw/gtfs/<cc>/<cc>_ktz.gtfs.zip), and
    data/raw/<cc>/timetable_calls.json / timetable_pairs.json, which --convert reads."""
    import csv
    import io
    import zipfile
    pl = placement(log)
    pos = dict(pl["pos"])
    for cc in CCS:
        for c, q in interpolate(cc, pl, log).items():
            pos.setdefault(c, q)
    places = Places(pl, pos)
    ex = express_codes()
    names = json.loads((KTZ_DIR / "names.json").read_text("utf-8")) \
        if (KTZ_DIR / "names.json").exists() else {}
    routes = json.loads((KTZ_DIR / "routes.json").read_text("utf-8"))
    # regularity strings per train, from every schedule row
    regs = defaultdict(set)
    for p in sorted((KTZ_DIR / "sched").glob("*.json")):
        for r in json.loads(p.read_text("utf-8")):
            regs[f"{r['num']}|{r['from']}|{r['to']}"].add(r["reg"])
    by_esr = {it[3]: i for i, it in enumerate(places.items) if it[3]}
    trains = []
    stat = Counter()
    for key, r in routes.items():
        calls = [c[0] for c in r["calls"]]
        if len(calls) < 2:
            stat["routes with no calls"] += 1
            continue
        cands = []
        for nm in calls:
            cc = []
            for code in names.get(nm, []):
                e = ex.get(code)
                # osm.sbin.ru's ESR for an Express code is from 2021 and Kazakhstan has
                # renumbered since (its 667909 "Актобе" is Book 1's Кардон): kept only where
                # the names agree
                if e and e[0] in by_esr and same_place_name(nm, pl["raw_of"][e[0]]["name"]):
                    cc.append(by_esr[e[0]])
            codes = names.get(nm) or []
            if not cc and codes and not any(c[:2] in OWN_EXPRESS for c in codes):
                pass       # an Express code of another network (Russia's 20...): abroad
            elif not cc:
                cc = places.find(nm)
            cands.append(cc[:40])
        trains.append({"num": r["num"], "title": f"{calls[0]} – {calls[-1]}", "calls": calls,
                       "cands": cands, "days": running_days(regs.get(key) or {""}),
                       "mins": call_minutes([(c[1], c[2]) for c in r["calls"]]),
                       "src": "ktz"})
    def plain_num(n):
        return n.lstrip("0").rstrip("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
    known = defaultdict(set)          # number -> skeletons of its calls
    for t in trains:
        known[plain_num(t["num"])] |= {lkey(c)[1] for c in t["calls"]}
    for h in hand_trains():
        sk = known.get(h["num"])
        if h["num"] and sk and lkey(h["calls"][0])[1] in sk and lkey(h["calls"][-1])[1] in sk:
            stat["hand trains already in KTZ's pages"] += 1
            continue
        h["cands"] = [places.find(nm)[:40] for nm in h["calls"]]
        trains.append(h)
    out = {cc: [] for cc in CCS}
    unmatched = Counter()
    for t in trains:
        ch = match_calls(t["calls"], t["cands"], places, t.get("mins"))
        stat["calls"] += len(t["calls"])
        stat["matched"] += len(ch)
        for i, nm in enumerate(t["calls"]):
            if i not in ch:
                unmatched[nm] += 1
        seq = [(i, ch[i]) for i in sorted(ch)]
        if len(seq) < 2:
            stat["trains with fewer than 2 calls placed"] += 1
            continue
        if len(t["days"]) < MIN_DAYS:
            stat["trains running less than weekly, left out"] += 1
            continue
        where = [country_at(*places.items[r][:2]) for _i, r in seq]
        for cc in CCS:
            if cc in where:
                out[cc].append((t, seq, where))
    log(f"timetable: {len(trains)} trains; {dict(stat)}")
    dbg = []
    for t in trains:
        ch = match_calls(t["calls"], t["cands"], places, t.get("mins"))
        dbg.append({"num": t["num"], "title": t["title"], "src": t["src"], "days": len(t["days"]),
                    "calls": [[nm, places.items[ch[i]][2] if i in ch else None,
                               country_at(*places.items[ch[i]][:2]) if i in ch else None]
                              for i, nm in enumerate(t["calls"])]})
    (KTZ_DIR / "matched.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=0), "utf-8")
    log("  most frequent unmatched calls: " + ", ".join(
        f"{k} ({v})" for k, v in unmatched.most_common(40)))

    def tbl(header, rows):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return b.getvalue()
    for cc in CCS:
        ts = [x for x in out[cc] if sum(1 for w in x[2] if w == cc) >= 1]
        used = sorted({r for _t, seq, _w in ts for _i, r in seq})
        calls_xy = sorted({(round(places.items[r][0], 6), round(places.items[r][1], 6))
                           for _t, seq, w in ts for (_i, r), ww in zip(seq, w) if ww == cc})
        at = {xy: i for i, xy in enumerate(calls_xy)}
        pairs = set()
        for _t, seq, w in ts:
            for ((_i, r1), w1), ((_j, r2), w2) in zip(zip(seq, w), zip(seq[1:], w[1:])):
                a = at.get((round(places.items[r1][0], 6), round(places.items[r1][1], 6)))
                b = at.get((round(places.items[r2][0], 6), round(places.items[r2][1], 6)))
                if a is not None and b is not None and a != b:
                    pairs.add((min(a, b), max(a, b)))
        raw(cc).mkdir(parents=True, exist_ok=True)
        (raw(cc) / "timetable_calls.json").write_text(json.dumps(calls_xy), "utf-8")
        (raw(cc) / "timetable_pairs.json").write_text(json.dumps(sorted(pairs)), "utf-8")
        stops_rows = [(f"p{r}", places.items[r][2], f"{places.items[r][1]:.6f}",
                       f"{places.items[r][0]:.6f}") for r in used]
        routes_rows, trips, st, cal = [], [], [], []
        for k, (t, seq, _w) in enumerate(ts):
            rid = f"{t['src']}{k}_{t['num']}"
            num = int(re.match(r"\d+", t["num"]).group()) if re.match(r"\d+", t["num"]) else 0
            rtype = "109" if 6000 <= num <= 7999 else "102"
            routes_rows.append((rid, t["src"], t["num"], t["title"][:120], rtype))
            trips.append((rid, rid, rid))
            for sq, (_i, r) in enumerate(seq, 1):
                st.append((rid, "", "", f"p{r}", sq))
            for d in sorted(t["days"]):
                cal.append((rid, d.strftime("%Y%m%d"), 1))
        gd = ROOT / "data" / "raw" / "gtfs" / cc
        gd.mkdir(parents=True, exist_ok=True)
        zp = gd / f"{cc}_casia.gtfs.zip"
        tmp = gd / f"{cc}_casia.gtfs.zip.tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("agency.txt", tbl(("agency_id", "agency_name", "agency_url",
                                          "agency_timezone"),
                                         [("ktz", AGENCY[cc], "https://bilet.railways.kz/",
                                           "Asia/Almaty"),
                                          ("railway.gov.tm", AGENCY[cc], "https://railway.gov.tm/",
                                           "Asia/Ashgabat"),
                                          ("hand", AGENCY[cc], "", "Asia/Almaty")]))
            z.writestr("stops.txt", tbl(("stop_id", "stop_name", "stop_lat", "stop_lon"),
                                        stops_rows))
            z.writestr("routes.txt", tbl(("route_id", "agency_id", "route_short_name",
                                          "route_long_name", "route_type"), routes_rows))
            z.writestr("trips.txt", tbl(("route_id", "service_id", "trip_id"), trips))
            z.writestr("stop_times.txt", tbl(("trip_id", "arrival_time", "departure_time",
                                              "stop_id", "stop_sequence"), st))
            z.writestr("calendar_dates.txt", tbl(("service_id", "date", "exception_type"), cal))
            z.writestr("feed_info.txt", tbl(
                ("feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_version"),
                [("noritetsu, from KTZ's ticket site and the operators' timetables",
                  "https://bilet.railways.kz/", "ru", date.today().isoformat())]))
        os.replace(tmp, zp)
        log(f"timetable {cc}: {len(ts)} trains, {len(used)} stations, {len(calls_xy)} called "
            f"here, {len(pairs)} consecutive pairs -> {zp}")


# The feed's agency, named with every administration whose lines the country's register has:
# gtfs_served treats a line whose manager no agency names as run by an operator the feed may
# lack, and leaves its sections "unknown" instead of closing them.
AGENCY = {
    "kz": "Қазақстан темір жолы; Кыргыз темир жолу; Южно-Уральская, Западно-Сибирская, "
          "Приволжская железная дорога",
    "uz": "Oʻzbekiston temir yoʻllari; Türkmendemirýollary",
    "kg": "Кыргыз темир жолу; Oʻzbekiston temir yoʻllari",
    "tj": "Роҳи оҳани Тоҷикистон; Oʻzbekiston temir yoʻllari",
    "tm": "Türkmendemirýollary; Oʻzbekiston temir yoʻllari",
}


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esr", nargs=2, metavar=("CC", "PBF"))
    ap.add_argument("--outline", metavar="CC")
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--place", action="store_true", help="placement alone, with its log")
    ap.add_argument("--borders", action="store_true")
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--timetable", action="store_true")
    ap.add_argument("--colours", metavar="CC")
    a = ap.parse_args()
    if a.esr:
        p = Path(a.esr[1])
        esr_pass(a.esr[0], p if p.is_absolute() else ROOT / p)
    if a.outline:
        write_outline(a.outline)
    if a.clip:
        clip(a.clip)
    if a.place:
        pl = placement()
        cnt = Counter()
        for c, p in pl["raw_of"].items():
            cnt[(pl["country"].get(c, "?") if c in pl["pos"] else "unplaced")] += 1
        log(f"points by country: {dict(cnt)}")
    if a.borders:
        find_borders()
    if a.crawl:
        crawl(a.limit)
    if a.names:
        crawl_names()
    if a.timetable:
        timetable()
    if a.convert:
        convert(a.convert)
    if a.colours:
        colours(a.colours)
