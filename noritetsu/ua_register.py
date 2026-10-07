"""Ukraine: Tariff Guide No. 4 (Тарифное руководство № 4), Ukrzaliznytsia's six railways,
written into rinf.py's input format, plus a timetable feed made from poizdato.net. The same
recipe as Russia's (ru_register.py), whose parser this reuses. ua_sources.md has the research
and the numbers; this docstring is how it works.

    python ua_register.py --esr data/raw/ukraine-latest.osm.pbf      # ESR codes + every station name
    python ua_register.py --disused data/raw/ukraine-latest.osm.pbf  # disused main-line ways (see below)
    #   (both before the .pbf is deleted; they write data/raw/ua/osm_*.json/.pkl)
    python ua_register.py --outline      # data/raw/ua/outline.geojson
    python ua_register.py --clip         # after every extract: data/proc/ua (kept as data/proc/ua/full)
    python ua_register.py --crawl-trains # poizdato.net's train pages -> data/raw/ua/poizdato/ (~1 h)
    python ua_register.py --timetable    # -> data/raw/gtfs/ua/ua_poizdato.gtfs.zip, timetable_calls.json
    python ua_register.py --convert      # -> data/raw/rinf/ua/{sections,points,names}.json
    python ua_register.py --colours      # -> colours/ua.csv
    python build_model.py --region ua --register rinf:data/raw/rinf/ua

THE REGISTER.  Book 1 of the tariff guide (the CIS rail council's, data/raw/ru/tr4_kniga1_*.xls,
the same file Russia's build reads) has six sheets for Ukrzaliznytsia: Ю-Зап, Льв, Од, Южн,
Придн, Дон (У). 499 tariff sections, every station, halt and post on each in order with its ESR
code and integer tariff km. One tariff section is one register line, as in Russia. Points are
placed at OSM's node with their ESR code, else at Wikidata's item with that code (P2815), else
at an OSM station of the same name (Russian or Ukrainian spelling: `name_keys`, whose
consonant skeleton joins "Здолбунов" and "Здолбунів") near the point's placed neighbours.
Names shown are Ukrainian: OSM's, else Wikidata's uk label, else Book 1's own Russian one.

TERRITORY.  The extract is clipped (`--clip`) to OSM's boundary of Ukraine less OSM's boundary
of Russia (which holds Crimea) less data/raw/ru/annex.geojson (the 2022-annexed oblasts where
Russia holds them), reaching REACH_DEG into the EU neighbours and Moldova so border crossings
keep their track. Pairs of points outside that (without the reach) are left out.

DISUSED TRACK.  OSM's mappers retag the frontline railways railway=disused (Kramatorsk,
Kupiansk, Pokrovsk, the Kherson line, which trains run on again). `--disused` reads those ways
of main-line kind and `--clip` puts them back as track, so tariff sections there trace and the
timetable decides (gtfs_served) whether they are drawn running or greyed.

STOPS.  A point is a stop when Book 2 gives it a passenger operation and either a train calls
there (the timetable feed, or an OSM train route, within SERVED_M) or it is a halt (an "ОП" in
Book 1, or an OSM railway=halt within HALT_M), the halt being on a stretch, between called
points, that has halts: such a stretch is a passenger line, running or not, and gtfs_served
greys it if no train runs. A stretch between two called points with no halt, STRETCH_KM or
longer, is freight track unless trains run over it: its end stops are cloned as junctions on
that line (ru_register's clones), so it answers to the timetable like any junction-ended
section and is dropped where no train runs.

THE TIMETABLE.  Ukrzaliznytsia publishes no feed, and its own sites do not answer from abroad.
poizdato.net lists every suburban and long-distance train with its calls and running days;
`--timetable` matches each call to an OSM station (or Wikidata's) by name along the train's
path and writes a GTFS feed that gtfs_served reads as it reads every country's. The
international trains come from Transitous' Ukrzaliznytsia feed (jbb.ghsq.de), kept beside it.

BORDERS.  BORDER lists the crossings with built neighbours; each gets a piece from the last
Ukrainian point to the neighbour's RINF border point, so both builds end at one id.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ua"


def esr_pass(pbf):
    """From the .pbf (extract.py keeps neither esr:user nor name:ru/name:uk): every node with
    an ESR code, and every rail station or halt node with its names. Ways and areas tagged as
    stations are left out (a station is a node in OSM's Ukraine nearly everywhere)."""
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
        row = [obj.id, lon, lat, t.get("name"), t.get("name:uk"), t.get("name:ru"),
               t.get("name:en"), rw, t.get("public_transport"), t.get("train"),
               t.get("old_name"), t.get("usage") or t.get("disused:railway")]
        if code:
            for c in code.replace(",", ";").split(";"):
                c = c.strip()
                if c.isdigit() and len(c) == 6:
                    esr[c].append(row)
        if rw in ("station", "halt") or t.get("public_transport") == "station":
            st.append(row + [code or ""])
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "osm_esr.json").write_text(json.dumps(esr, ensure_ascii=False), "utf-8")
    (RAW / "osm_stations.json").write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    print(f"--esr: {sum(len(v) for v in esr.values())} nodes, {len(esr)} codes; "
          f"{len(st)} station nodes -> {RAW}")


DISUSED_SERVICE = {"spur", "yard", "siding", "crossover"}


def disused_pass(pbf):
    """data/raw/ua/osm_disused.pkl: OSM's railway=disused ways of main-line kind (no service
    tag of spur, yard, siding or crossover; not tram gauge), with their nodes' coordinates.
    Since 2022 OSM's mappers have retagged the frontline railways disused (Kramatorsk,
    Kupiansk, Pokrovsk, Kherson - Snihurivka, the Nikopol bank of the Dnipro): extract.py
    keeps no disused track, so the tariff sections there had no rails to trace and were left
    out altogether. --clip puts these ways back as track (`railway=rail`, tagged
    `noritetsu:osm_railway=disused`), so the register traces them and the timetable decides,
    as everywhere else, whether a train runs: stop-to-stop sections no train calls over are
    drawn greyed as not running, and the Kherson line, which trains do run on, is drawn."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import pickle
    import osmium
    out = {}
    fp = (osmium.FileProcessor(str(pbf)).with_locations()
          .with_filter(osmium.filter.KeyFilter("railway")))
    for o in fp:
        if not o.is_way() or o.tags.get("railway") != "disused":
            continue
        t = dict(o.tags)
        if t.get("service") in DISUSED_SERVICE or t.get("disused:railway") in (
                "tram", "light_rail", "subway") or t.get("gauge") in ("1524", "750", "1000"):
            continue
        try:
            nds = [(n.ref, round(n.lon, 7), round(n.lat, 7)) for n in o.nodes]
        except osmium.InvalidLocationError:
            continue
        out[o.id] = (t, nds)
    with open(RAW / "osm_disused.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"--disused: {len(out)} disused main-line ways -> {RAW / 'osm_disused.pkl'}")


# ================================================================ the outline and the clip

PROC = ROOT / "data" / "proc" / "ua"
RU_RAW = ROOT / "data" / "raw" / "ru"
NE = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
# The outline reaches this far into the neighbours whose trains cross (and Moldova, whose
# track Ukrainian lines pass through), so a border crossing keeps its track up to RINF's
# border point. Never into Russia, Belarus, Crimea or the annexed area.
REACH_DEG = 0.03
REACH_INTO = ("pl", "sk", "hu", "ro", "md")


def outline(reach=True):
    """Ukraine as this build draws it: OSM's boundary of Ukraine (relation 60199, from
    polygons.openstreetmap.fr: data/raw/ua/ua_boundary.geojson) less OSM's boundary of Russia
    (data/raw/ru/ru_boundary.geojson, relation 60189, which holds Crimea) less
    data/raw/ru/annex.geojson (the 2022-annexed oblasts where Russia holds them). With `reach`,
    plus REACH_DEG into the EU neighbours and Moldova (Natural Earth 10m)."""
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    ua = shape(json.loads((RAW / "ua_boundary.geojson").read_text("utf-8")))
    ru = shape(json.loads((RU_RAW / "ru_boundary.geojson").read_text("utf-8")))
    ann = unary_union([shape(f["geometry"]) for f in json.loads(
        (RU_RAW / "annex.geojson").read_text("utf-8"))["features"]])
    out = ua.difference(ru).difference(ann)
    if reach:
        nb = []
        for f in json.loads(NE.read_text("utf-8"))["features"]:
            if (f["properties"].get("ISO_A2_EH") or "").lower() in REACH_INTO:
                nb.append(shape(f["geometry"]))
        # Natural Earth's borders are coarse (the Tisza at Chop - Záhony fell in neither
        # country and the bridge was cut), so the neighbours are widened a little first.
        near = (ua.buffer(REACH_DEG).intersection(unary_union(nb).buffer(0.02))
                .difference(ru).difference(ann))
        out = out.union(near)
    polys = [g for g in shapely.get_parts(out) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.multipolygons(shapely.get_parts(shapely.geometrycollections(polys)))


def write_outline():
    """data/raw/ua/outline.geojson: the outline without the reach, for the app's region
    outline (tools/build_regions.py) and for checks."""
    from shapely.geometry import mapping
    shp = outline(reach=False)
    (RAW / "outline.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {
            "what": "Ukraine as noritetsu draws it: OSM relation 60199 less OSM relation 60189 "
                    "(Russia, with Crimea) less data/raw/ru/annex.geojson"},
         "geometry": mapping(shp)}]}), "utf-8")
    print(f"--outline: {shp.area:.3f} sq deg -> {RAW / 'outline.geojson'}")


def clip():
    """data/proc/ua from data/proc/ua/full (the extract as extract.py wrote it; moved there
    on the first run, so this can be rerun after changing the outline): what lies in
    `outline()` kept. A way goes if at least half its nodes are outside, a stop if it is, a
    relation if no member is left (ru_register.clip's rules). Run after every extract."""
    import pickle
    import numpy as np
    import shapely
    full = PROC / "full"
    names = ("ways.pkl", "rels.pkl", "stops.pkl", "infra.pkl", "coords.npz")
    stamp = PROC / "clip_stamp.json"
    # A new extract writes data/proc/ua/*.pkl after the last clip's stamp: it becomes `full`.
    fresh = (PROC / "ways.pkl").exists() and (
        not stamp.exists() or (PROC / "ways.pkl").stat().st_mtime > stamp.stat().st_mtime + 5)
    if fresh:
        full.mkdir(exist_ok=True)
        for fn in names:
            os.replace(PROC / fn, full / fn)
        print("clip: the extract in data/proc/ua moved to data/proc/ua/full")

    def rd(fn):
        with open(full / fn, "rb") as f:
            return pickle.load(f)
    ways, rels, stops, infra = (rd(f) for f in names[:4])
    with np.load(full / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    shp = outline()
    shapely.prepare(shp)
    # The disused main-line ways (disused_pass), as track, inside Ukraine as drawn only.
    dp = RAW / "osm_disused.pkl"
    if dp.exists():
        with open(dp, "rb") as f:
            dis = pickle.load(f)
        strict = outline(reach=False)
        shapely.prepare(strict)
        add_id, add_x, add_y, n_add = [], [], [], 0
        for wid, (tags, nds) in dis.items():
            if wid in ways or len(nds) < 2:
                continue
            xs = np.array([x for _n, x, _y in nds])
            ys = np.array([y for _n, _x, y in nds])
            if 2 * int(shapely.contains_xy(strict, xs, ys).sum()) < len(nds):
                continue
            t = {k: v for k, v in tags.items() if k != "railway"}
            t["railway"] = "rail"
            t["noritetsu:osm_railway"] = "disused"
            ways[wid] = (t, np.array([n for n, _x, _y in nds], dtype=np.int64))
            add_id += [n for n, _x, _y in nds]
            add_x += [int(round(x * 1e7)) for _n, x, _y in nds]
            add_y += [int(round(y * 1e7)) for _n, _x, y in nds]
            n_add += 1
        allid = np.concatenate([cid, np.array(add_id, dtype=cid.dtype)])
        allx = np.concatenate([cx, np.array(add_x, dtype=cx.dtype)])
        ally = np.concatenate([cy, np.array(add_y, dtype=cy.dtype)])
        cid, first = np.unique(allid, return_index=True)
        cx, cy = allx[first], ally[first]
        print(f"clip: {n_add} disused main-line ways put back as track")
    out = ~shapely.contains_xy(shp, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, defaultdict(int)
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
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  cut {v:5d}  {name}")
    print(f"clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
          f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = PROC / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, PROC / fn)
    tmp = PROC / "coords.tmp.npz"
    np.savez_compressed(tmp, id=cid, x=cx, y=cy)
    os.replace(tmp, PROC / "coords.npz")
    stamp.write_text(json.dumps({"clipped_from": str(full), "ways": len(keep_w)}), "utf-8")


# ================================================================ the timetable (poizdato.net)

POIZDATO = "https://poizdato.net"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
CRAWL_DELAY = 1.2         # seconds between requests


def _get(url, tries=3):
    import time
    import urllib.request
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:                                   # noqa: BLE001
            if getattr(e, "code", None) == 404:
                return None
            print(f"  retry {i}: {url} {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(10 * (i + 1))
    return None


def crawl(kinds, limit=None):
    """Every train page poizdato.net's sitemap lists (data/raw/ua/poizdato_sitemap.json),
    kept as HTML in data/raw/ua/poizdato/<kind>/<slug>.html; pages already there are not
    fetched again. Station pages (rozklad-po-stantsii) list every train calling there,
    including some the sitemap leaves out; `--crawl-stations` fetches those and then the
    train pages they name."""
    import re
    import time
    import urllib.parse
    sm = json.loads((RAW / "poizdato_sitemap.json").read_text("utf-8"))
    todo = []
    for kind in kinds:
        for u in sm.get(kind, []):
            todo.append((kind, u))
    # train pages named on station pages fetched earlier
    st_dir = RAW / "poizdato" / "rozklad-po-stantsii"
    if "rozklad-elektrychky" in kinds and st_dir.exists():
        known = {u for _k, u in todo}
        for f in st_dir.glob("*.html"):
            for href in re.findall(r'href="(/rozklad-(?:elektrychky|poizda)/[^"]+/)"',
                                   f.read_text("utf-8", errors="replace")):
                u = POIZDATO + href
                if u not in known and urllib.parse.unquote(u) not in known:
                    known.add(u)
                    todo.append((href.split("/")[1], u))
    n = got = 0
    for kind, u in todo:
        slug = urllib.parse.unquote(u.rstrip("/").split("/")[-1])
        slug = re.sub(r'[\\/:*?"<>|]', "_", slug)
        p = RAW / "poizdato" / kind / f"{slug}.html"
        if p.exists():
            continue
        if limit and n >= limit:
            break
        n += 1
        t = _get(u)
        if t:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t, "utf-8")
            got += 1
        if n % 50 == 0:
            print(f"  {n} fetched ({got} ok) of {len(todo)} listed", flush=True)
        time.sleep(CRAWL_DELAY)
    print(f"crawl {kinds}: {n} fetched, {got} ok, {len(todo)} listed", flush=True)


# ================================================================ names

GENERIC = {"пасажирський", "пасажирська", "пасажирське", "пас", "пасс", "головний", "головна",
           "головне", "пассажирский", "пассажирская", "пассажирское", "главный", "главная",
           "оп", "зп", "о", "п", "з", "платформа", "пл", "ост", "пункт", "зупинний",
           "зупинка", "остановочный", "станція", "станция", "ст", "рзд", "роз", "разъезд",
           "роз'їзд", "блокпост", "бп", "обп", "пп", "вокзал", "залізничний", "зал", "station",
           "halt", "railway"}
ROMAN = {"i": "1", "і": "1", "ii": "2", "іі": "2", "iii": "3", "ііі": "3", "iv": "4", "іv": "4"}
VOWELS = set("аеєиіїйоуюяыэёьъ'")


def _tokens(name, brackets=False):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKC", name or "").lower()
    for a, b in (("ё", "е"), ("ґ", "г"), ("’", "'"), ("ʼ", "'"), ("`", "'"), ("«", ""),
                 ("»", ""), ("\"", "")):
        s = s.replace(a, b)
    s = re.sub(r"\(([^)]*)\)", r" \1 " if brackets else " ", s)
    toks = [t for t in re.split(r"[\s\-–—.,/№]+", s) if t]
    out = []
    for t in toks:
        t = ROMAN.get(t, t)
        t = {"ім": "імені", "им": "имени"}.get(t, t)
        if t in GENERIC:
            continue
        out.append(t)
    return out


def name_keys(name):
    """Keys for one station name, strongest first: the whole name folded to Latin
    (rinf.norm), the name less generic words and brackets ("Київ-Пас.(Північна)" ->
    "kiyiv"), and that base's consonant skeleton, which a Russian and a Ukrainian spelling of
    one place usually share ("Здолбунов" / "Здолбунів" -> здлбнв)."""
    from rinf import norm
    full = norm(name)
    toks = _tokens(name)
    base = norm("".join(toks))
    sk = "".join(c for c in "".join(toks) if c not in VOWELS and (c.isalnum()))
    out = []
    for k in (full, base, "~" + sk if len(sk) >= 3 else ""):
        if k and k not in out:
            out.append(k)
    return out


def clean_uk(label):
    """A Wikidata Ukrainian label as a station name: 'Керч (станція)' -> 'Керч'."""
    import re
    s = re.sub(r"\s*\((?:станція|зупинний пункт|платформа|залізнична станція|"
               r"роз'їзд|пасажирська платформа)[^)]*\)", "", label or "").strip()
    s = re.sub(r"^(?:залізнична станція|станція|зупинний пункт|платформа)\s+", "", s)
    return s


UK_LATIN = dict(zip("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'",
                    ["a", "b", "v", "h", "g", "d", "e", "ie", "zh", "z", "y", "i", "i", "i", "k",
                     "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh",
                     "shch", "", "iu", "ia", ""]))


def en_ok(uk, en):
    """Does an English name read as a romanisation of the Ukrainian (or Russian) one? 0..1
    against the national transliteration; a check, never a source of names."""
    import re
    from difflib import SequenceMatcher
    if not en or any(ord(c) > 127 for c in en):
        return 0.0
    a = "".join(UK_LATIN.get(c, c) for c in "".join(_tokens(uk)))
    b = re.sub(r"[^a-z0-9]", "", "".join(_tokens(en)))
    sq = [("shch", "sh"), ("kh", "h"), ("j", "i"), ("y", "i"), ("w", "v"), ("ii", "i"),
          ("ie", "e"), ("ia", "a"), ("iu", "u"), ("yu", "u"), ("ya", "a")]
    for x, y in sq:
        a, b = a.replace(x, y), b.replace(x, y)
    if sorted(re.findall(r"\d+", a)) != sorted(re.findall(r"\d+", b)):
        return 0.0
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


# ================================================================ the timetable

def parse_train(path):
    """One poizdato train page: number, kind, its calls (station name, station slug, time) in
    order, and the dates the page's running calendar marks (day-of-year cells)."""
    import html
    import re
    from datetime import date, timedelta
    t = path.read_text("utf-8", errors="replace")
    title = re.search(r"<title>(.*?)</title>", t, re.S)
    title = html.unescape(title.group(1)).strip() if title else ""
    num = title.split(" ")[0] if title else ""
    i0 = t.find("<th>Станція</th>")
    if i0 < 0:
        i0 = t.find("Станція")
    i1 = t.find("Графік руху", i0)
    body = t[i0:i1 if i1 > 0 else len(t)]
    calls = []
    for m in re.finditer(r'<tr[^>]*>\s*<td>\s*(?:<a href="/rozklad-po-stantsii/([^"]+)/">)?'
                         r'\s*([^<]+?)\s*(?:</a>)?\s*</td>(.*?)</tr>', body, re.S):
        slug, name, rest = m.group(1) or "", html.unescape(m.group(2)).strip(), m.group(3)
        times = re.findall(r"(\d{1,2})\.(\d{2})", rest)
        calls.append({"name": name, "slug": slug, "times": [f"{h}:{mm}" for h, mm in times]})
    days = []
    y0 = date(2026, 1, 1)
    for m in re.finditer(r"<td id='i_(\d+)' class=\"[^\"]*\"><span class=\"([^\"]*)\">\d+</span>", t):
        if "ui-state-active" in m.group(2):
            days.append((y0 + timedelta(days=int(m.group(1)))).isoformat())
    every = re.search(r"курсує\s+([^.<]*)", t)
    return {"num": num, "title": title, "kind": path.parent.name, "calls": calls,
            "days": sorted(set(days)), "runs": every.group(1).strip() if every else ""}


# ================================================================ the register

OUT = ROOT / "data" / "raw" / "rinf" / "ua"
GTFS_DIR = ROOT / "data" / "raw" / "gtfs" / "ua"
# Book 1's sheets of Ukrzaliznytsia's six regional railways: sheet -> (road code, name).
UA_ROADS = {
    "Ю-Зап (У)": ("32", "Південно-Західна залізниця"),
    "Льв (У)": ("35", "Львівська залізниця"),
    "Од (У)": ("40", "Одеська залізниця"),
    "Южн (У)": ("43", "Південна залізниця"),
    "Придн (У)": ("45", "Придніпровська залізниця"),
    "Дон (У)": ("48", "Донецька залізниця"),
}
SERVED_M = 400            # an OSM train route stop or a timetable call this close serves a point
NAME_REACH_KM = 10        # a name match may lie this much further than the tariff km say
STRETCH_KM = 8            # an unserved stop-to-stop stretch at least this long with no halt is freight
HALT_M = 400              # an OSM railway=halt this close to a point marks it a passenger halt


# Crossings with a built neighbour where passenger trains run or ran: the border point's uopid
# (borders.load(): ERA RINF's border_points.json, or borders.EXTRA where RINF has none) and the
# Book 1 point the Ukrainian side reaches it from (the nearest, a station or the line's end).
# Each gets a piece from that point to the border point, whose id ("eEU...", "eMDUA...") is the
# one both countries' builds end at (borders.py). Its length is the crow-fly distance x 1.2
# (Book 1 stops at the border station); the trace is checked against that with tol_abs. Left
# out: Uzhhorod - Maťovce (EU00161) and Esen - Eperjeske (EU00192), broad-gauge freight; Russia
# and Belarus (no passenger trains).
BORDER = [
    ("EU00173", "373502"),     # Мостиська ІІ - Przemyśl
    ("EU00174", "372500"),     # Рава-Руська - Werchrata
    ("EU00175", "372500"),     # Рава-Руська - Hrebenne
    ("EU00178", "351306"),     # Ягодин - Dorohusk
    ("EU00193", "380101"),     # Чоп - Záhony
    ("EU00162", "380101"),     # Чоп - Čierna nad Tisou (EU00163 is the same spot)
    ("EU00240", "384812"),     # Дяково - Halmeu
    ("EU00241", "385603"),     # Тересва - Câmpulung la Tisa
    ("EU00242", "386220"),     # Ділове - Valea Vișeului
    ("EU00243", "368612"),     # Багринівка (Вадул-Сірет) - Vicșani
    # Могилів-Подільський - Otaci (Moldova), borders.EXTRA's point on the Dniester bridge
    # (by-md agent, 2026-10-03): Kyiv - Chișinău trains.
    ("MDUAVALCINET", "331904"),
]


def book1():
    import ru_register as rr
    saved = rr.ROADS
    rr.ROADS = UA_ROADS
    try:
        return rr.book1()
    finally:
        rr.ROADS = saved


def dist_m(lon1, lat1, lon2, lat2):
    import math
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def load_osm_stations():
    """OSM rail station and halt nodes (from --esr), with every name a node carries."""
    rows = json.loads((RAW / "osm_stations.json").read_text("utf-8"))
    out = []
    for r in rows:
        nid, lon, lat, nm, nm_uk, nm_ru, nm_en, rw, pt, train, old, usage = r[:12]
        if usage in ("disused", "abandoned") or rw in ("disused", "abandoned"):
            continue
        if rw not in ("station", "halt") and not (pt == "station" and train == "yes"):
            continue
        out.append({"id": nid, "lon": lon, "lat": lat, "name": nm_uk or nm or "",
                    "names": [x for x in (nm, nm_uk, nm_ru, old) if x], "en": nm_en or "",
                    "rw": rw, "esr": r[12] if len(r) > 12 else ""})
    return out


def wd_stations():
    """ESR code -> {lon, lat, uk, en}, Wikidata's items with P2815 in Ukraine."""
    import re
    p = RAW / "wd_stations.json"
    if not p.exists():
        return {}
    got = defaultdict(list)
    for r in json.loads(p.read_text("utf-8"))["rows"]:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r.get("coord", ""))
        en = r.get("en") or ""
        if not en and r.get("enwiki"):
            import urllib.parse
            en = urllib.parse.unquote(r["enwiki"].rsplit("/", 1)[-1]).replace("_", " ")
        got[r["esr"]].append({"lon": float(m.group(1)) if m else None,
                              "lat": float(m.group(2)) if m else None,
                              "uk": clean_uk(r.get("uk") or ""), "en": en, "q": r["s"]})
    out = {}
    for c, v in got.items():
        if len({(x["uk"], x["lon"]) for x in v}) == 1:
            out[c] = v[0]
    return out


class NameIndex:
    """Places by name key, for finding a named station near a known spot."""

    def __init__(self):
        self.by = defaultdict(list)

    def add(self, name, lon, lat, ref):
        for rank, k in enumerate(name_keys(name)):
            self.by[k].append((lon, lat, ref, rank))

    def find(self, name):
        """[(lon, lat, ref, rank)], rank 0 an exact name, 1 the base, 2 the skeleton; the
        strongest rank any candidate reaches, only."""
        out = []
        for qr, k in enumerate(name_keys(name)):
            # a match is as weak as the weaker of its two keys: "Київ-Пас." meets
            # "Київ-Пасажирський" through both bases, not a station called plain "Київ"
            out += [(x, y, ref, max(qr, r)) for x, y, ref, r in self.by.get(k, [])]
        if not out:
            return []
        best = min(r for *_x, r in out)
        seen, res = set(), []
        for x in out:
            if x[3] <= best and (x[0], x[1]) not in seen:
                seen.add((x[0], x[1]))
                res.append(x)
        return res


def section_ends(s, by_key):
    """The two end points a section header names (Book 1 lists can run past the node under
    extra codes), else the first and last point; plus the header's via names."""
    import re
    import ru_register as rr
    hdr = rr.VIA.sub("", s["name"])
    hdr = re.sub(r"\((?:[^()]*ж\.\s*д\.?|[^()]*(?:ПАССАЖИР|пассажир)[^()]*)\)", "", hdr)
    ends = [x.strip() for x in re.split(r"\s+-\s+", hdr)]
    pa, pb = s["pts"][0], s["pts"][-1]
    if len(ends) == 2 and all(ends):
        ga = by_key.get(rr.nkey(ends[0])) or by_key.get(rr.nkey(rr.EXTRA_CODE.sub("", ends[0])))
        gb = by_key.get(rr.nkey(ends[1])) or by_key.get(rr.nkey(rr.EXTRA_CODE.sub("", ends[1])))
        pa, pb = ga or pa, gb or pb
    vias = []
    m = rr.VIA.search(s["name"])
    if m:
        via = re.split(r"\s*\(", m.group(1))[0].strip()
        for v in re.split(r",|\s+и\s+", via):
            v = re.sub(r"^(?:ст\.?|оп|о\.п\.)\s+", "", v.strip(), flags=re.I)
            vias.append(by_key.get(rr.nkey(v)))
    return pa, pb, vias


def convert():
    import math
    import pickle
    import re
    from datetime import date
    import numpy as np
    import shapely
    from scipy.spatial import cKDTree
    import ru_register as rr
    secs, b1name = book1()
    ops, b2name = rr.book2()
    print(f"Book 1 ({b1name}): {len(secs)} Ukrainian sections; Book 2 ({b2name}): "
          f"{len(ops)} points")
    esr = json.loads((RAW / "osm_esr.json").read_text("utf-8"))
    wd = wd_stations()
    ost = load_osm_stations()
    nidx = NameIndex()
    for i, s in enumerate(ost):
        for nm in s["names"]:
            nidx.add(nm, s["lon"], s["lat"], i)
    stat = defaultdict(int)

    # --- 1. each section's points, extra codes at the same km dropped (as ru_register)
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
    raw_of = {}
    for s in secs:
        for p in s["pts"]:
            raw_of.setdefault(p["esr"], p)

    # --- 2. place the points: OSM's ESR node, else Wikidata's item, else an OSM station of
    # the same name (Russian or Ukrainian spelling) near the point's placed neighbours.
    pos, how, node_of = {}, {}, {}
    for c in raw_of:
        rows = esr.get(c) or []
        best = ([r for r in rows if r[7] in ("station", "halt")]
                or [r for r in rows if r[8] == "station"] or rows)
        if best:
            pos[c], how[c], node_of[c] = (best[0][1], best[0][2]), "esr", best[0]
    for _round in range(4):
        for s in secs:
            pts = s["pts"]
            for i, p in enumerate(pts):
                c = p["esr"]
                if c in pos:
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
                w = wd.get(c)
                if w and w["lon"] is not None and fits(w["lon"], w["lat"]):
                    pos[c], how[c] = (w["lon"], w["lat"]), "wikidata"
                    continue
                cands = [x for x in nidx.find(p["name"]) if fits(x[0], x[1])]
                if cands and anchors:
                    lon, lat, ref, _r = min(cands, key=lambda x: min(
                        dist_m(x[0], x[1], *a) for a, _k in anchors))
                    pos[c], how[c], node_of[c] = (lon, lat), "name", ost[ref]
    codes = set(raw_of)
    print(f"points: {len(codes)}; placed by ESR {sum(1 for c in codes if how.get(c) == 'esr')}, "
          f"Wikidata {sum(1 for c in codes if how.get(c) == 'wikidata')}, by name "
          f"{sum(1 for c in codes if how.get(c) == 'name')}, unplaced "
          f"{sum(1 for c in codes if c not in pos)}")

    # --- names: OSM's (Ukrainian) where the point is an OSM station, else Wikidata's
    # Ukrainian label, else an OSM station of a matching name within 300 m, else Book 1's own
    # (Russian) spelling, counted.
    uk, en_of, name_how = {}, {}, defaultdict(int)
    near_tree = cKDTree(np.array([[s["lon"] * 0.67, s["lat"]] for s in ost]))
    for c, p in raw_of.items():
        n = node_of.get(c)
        nm, en = "", ""
        if isinstance(n, list) and n[7] in ("station", "halt") and (n[4] or n[3]):
            nm, en = n[4] or n[3], n[6] or ""
            name_how["OSM by ESR"] += 1
        elif isinstance(n, dict):
            nm, en = n["name"], n["en"]
            name_how["OSM by name"] += 1
        elif wd.get(c, {}).get("uk"):
            nm = wd[c]["uk"]
            name_how["Wikidata"] += 1
        elif c in pos:
            ks = set(name_keys(p["name"]))
            for j in near_tree.query_ball_point([pos[c][0] * 0.67, pos[c][1]], 0.003):
                if any(set(name_keys(x)) & ks for x in ost[j]["names"]):
                    nm, en = ost[j]["name"], ost[j]["en"]
                    name_how["OSM nearby"] += 1
                    break
        if not nm:
            nm = p["name"]
            name_how["Book 1 (Russian)"] += 1
        uk[c] = nm
        if not en and wd.get(c, {}).get("en"):
            en = rr.clean_en(wd[c]["en"])
        if en and en_ok(nm, en) >= 0.75:
            en_of[c] = en
    print(f"names: {dict(name_how)}; English {len(en_of)}")

    # --- 3. inside Ukraine as drawn (no Crimea, no annexed area, nothing abroad)
    shp = outline(reach=False)
    shapely.prepare(shp)
    placed = sorted(c for c in codes if c in pos)
    xy = np.array([pos[c] for c in placed])
    in_out = dict(zip(placed, shapely.contains_xy(shp, xy[:, 0], xy[:, 1]).tolist()))
    for s in secs:
        pts = s["pts"]
        for i, p in enumerate(pts):
            c = p["esr"]
            if c in pos:
                continue
            prev = next((pts[j]["esr"] for j in range(i - 1, -1, -1) if pts[j]["esr"] in pos), None)
            nxt = next((pts[j]["esr"] for j in range(i + 1, len(pts)) if pts[j]["esr"] in pos), None)
            nb = [in_out[q] for q in (prev, nxt) if q is not None]
            in_out[c] = in_out.get(c, True) and (all(nb) if nb else False)

    def inside(c):
        return in_out.get(c, False)
    from shapely.geometry import shape as _shape
    _ru = _shape(json.loads((RU_RAW / "ru_boundary.geojson").read_text("utf-8")))
    _ann = shapely.union_all([_shape(f["geometry"]) for f in json.loads(
        (RU_RAW / "annex.geojson").read_text("utf-8"))["features"]])
    _ua = _shape(json.loads((RAW / "ua_boundary.geojson").read_text("utf-8")))
    for g in (_ru, _ann, _ua):
        shapely.prepare(g)

    def where_is(q):
        if q is None:
            return "unplaced, between points outside"
        if shapely.contains_xy(_ann, *q):
            return "annexed area"
        if shapely.contains_xy(_ru, *q):
            return "Crimea" if q[1] < 46.3 else "Russia"
        if not shapely.contains_xy(_ua, *q):
            return "abroad"
        return "other"

    # --- 4. which points are stops
    kx = 111.32 * math.cos(math.radians(48.5))
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(PROC / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    served_xy = []
    for tags, members in rels.values():
        if tags.get("type") == "route" and tags.get("route") == "train":
            for ty, ref, role in members:
                if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                    served_xy.append((stops[ref][1], stops[ref][2]))
    n_osm = len(set(served_xy))
    tt = json.loads((RAW / "timetable_calls.json").read_text("utf-8")) \
        if (RAW / "timetable_calls.json").exists() else []
    served_xy += [(x, y) for x, y in tt]
    print(f"served places: {n_osm} OSM train route stops, {len(tt)} timetable stations")
    stree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in set(served_xy)]))
    halts = [(s["lon"], s["lat"]) for s in ost if s["rw"] == "halt"]
    htree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in halts]))
    ttree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in tt])) if tt else None

    def near(tree, c, m):
        q = pos.get(c)
        return bool(q and tree is not None and tree.query_ball_point(
            [q[0] * kx, q[1] * 110.57], m / 1000))
    # Consecutive calls of a running train (timetable_pairs.json): a stretch between two such
    # calls is run over, whatever it has on it, and is never treated as freight track.
    tt_pairs = set()
    if (RAW / "timetable_pairs.json").exists() and ttree is not None:
        tt_pairs = {tuple(p) for p in json.loads((RAW / "timetable_pairs.json").read_text("utf-8"))}

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
        flag = rr.passenger_op(ops.get(c))
        if not flag:
            kind[c] = "none"
        elif near(stree, c, SERVED_M):
            kind[c] = "served"
        elif re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) or near(htree, c, HALT_M):
            kind[c] = "halt"
        else:
            kind[c] = "flag"
    print(f"point kinds: {dict(Counter_(kind.values()))}; of the served, "
          f"{sum(1 for c in kind if kind[c] == 'served' and near(ttree, c, SERVED_M))} by the "
          f"timetable")

    # --- 5. sections: owners, cut at the outline, unserved stretches
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
    rows, names, used, stop_codes = [], {}, set(), set()
    border_rows = []
    km_kept = defaultdict(float)
    for s in secs:
        if s["km"] == 0:
            stat["node-list sections dropped (all 0 km)"] += 1
            continue
        pairs = []
        for a, b in zip(s["pts"], s["pts"][1:]):
            key = frozenset((a["esr"], b["esr"]))
            km = (b["km0"] or 0) - (a["km0"] or 0)
            if a["esr"] == b["esr"]:
                continue
            if owner[key] != s["id"]:
                stat["pairs on another section"] += 1
                km_kept["on another section"] += km
                continue
            if (s["id"], key) in coarse:
                stat["pairs another section lists finer"] += 1
                km_kept["listed finer elsewhere"] += km
                continue
            if not (inside(a["esr"]) and inside(b["esr"])):
                stat["pairs outside Ukraine as drawn"] += 1
                c = a["esr"] if not inside(a["esr"]) else b["esr"]
                km_kept["outside: " + where_is(pos.get(c))] += km
                continue
            pairs.append((a, b, km))
        if not pairs:
            continue
        # Stops. A served point with a passenger operation is one. A stretch between two of
        # them (or a line end) with a halt on it is a passenger line, running or not: its
        # flagged points are stops too, so a closed line keeps its stations and gtfs_served
        # greys what no train runs over. A stretch with no halt at all of STRETCH_KM or more
        # is freight track unless trains run over it: its end stops are cloned as junctions on
        # this line ("<code>@<section>"), so it answers to the timetable like any
        # junction-ended section (ru_register's clones).
        is_stop = {}
        seq = [pairs[0][0]] + [b for _a, b, _k in pairs]
        brk = [0]
        for i in range(1, len(pairs)):
            if pairs[i][0]["esr"] != pairs[i - 1][1]["esr"]:
                brk.append(i)
        runs, cur = [], []
        for i, (a, b, km) in enumerate(pairs):
            if i in brk and cur:
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
                km = sum(k for _a, _b, k in run[u:v])
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
                         "b": f"esr:{cb}", "len": str(km), "im": UA_ROADS[s["sheet"]][0],
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
            km_kept["kept"] += km
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": UA_ROADS[s["sheet"]][0],
                         "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": UA_ROADS[s["sheet"]][0],
                         "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        stop_codes |= {c for c, v in is_stop.items() if v}
        # the line's name: its two ends' names as shown (Ukrainian), with the header's via
        by_key = {}
        for p in s["pts"]:
            by_key.setdefault(rr.nkey(p["name"]), p)
            by_key.setdefault(rr.nkey(rr.EXTRA_CODE.sub("", p["name"])), p)
        pa, pb, vias = section_ends(s, by_key)
        name = f"{uk[pa['esr']]} — {uk[pb['esr']]}"
        ea, eb = en_of.get(pa["esr"], ""), en_of.get(pb["esr"], "")
        name_en = f"{ea} — {eb}" if ea and eb else ""
        if vias:
            if all(vias):
                name += f" (через {', '.join(uk[v['esr']] for v in vias)})"
                ven = [en_of.get(v["esr"], "") for v in vias]
                name_en = f"{name_en} (via {', '.join(ven)})" if name_en and all(ven) else ""
            else:
                name += " (2)"
                name_en = f"{name_en} (2)" if name_en else ""
        names[s["id"]] = {"name": name, "name_en": name_en, "type": s["type"],
                          "sheet": s["sheet"], "road": UA_ROADS[s["sheet"]][0],
                          "tariff_km": s["km"], "header": s["name"]}
    # --- border pieces: from the last Ukrainian point to the neighbour's border point
    import borders
    bpts = {p["id"][1:]: p for p in borders.load()}
    n_border = 0
    for uop, code in BORDER:
        b = bpts.get(uop)
        if b is None or code not in used or code not in pos:
            stat["border pieces with no point"] += 1
            continue
        own = sorted((s for s in secs if s["pts"] and code in (s["pts"][0]["esr"],
                                                               s["pts"][-1]["esr"])
                      and s["id"] in names),
                     key=lambda s: (rr.TYPE_RANK.get(s["type"], 9), s["id"]))
        if not own:
            own = sorted((s for s in secs if s["id"] in names
                          and any(p["esr"] == code for p in s["pts"])), key=lambda s: s["id"])
        if not own:
            continue
        crow = dist_m(*pos[code], b["lon"], b["lat"]) / 1000
        rows.append({"sol": f"{own[0]['id']}:{uop}", "line": own[0]["id"], "a": f"esr:{code}",
                     "b": f"eu:{uop}", "len": f"{max(0.5, crow * 1.2):.1f}",
                     "im": UA_ROADS[own[0]["sheet"]][0], "label": f"{uk[code]} - {b['name']}"})
        border_rows.append({"op": f"eu:{uop}", "uopid": uop, "name": b["name"], "type": "90",
                            "lon": b["lon"], "lat": b["lat"]})
        n_border += 1
    stat["border pieces"] = n_border
    # Two sections named alike (two ways between the same nodes) are told apart by id.
    cnt = defaultdict(int)
    for e in names.values():
        cnt[e["name"]] += 1
    for sid, e in names.items():
        if cnt[e["name"]] > 1:
            e["name"] += f" ({sid})"
            if e["name_en"]:
                e["name_en"] += f" ({sid})"
    print(f"sections: {dict(stat)}")
    print(f"tariff km: { {k: round(v) for k, v in km_kept.items()} }")
    pts_out = []
    for c in sorted(used):
        base = c.split("@")[0]
        p = raw_of[base]
        q = pos.get(base)
        typ = ("70" if re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) else "10") \
            if base in stop_codes and base == c else "80"
        r = {"op": f"esr:{c}", "uopid": f"UA{c}", "name": uk[base], "type": typ}
        if en_of.get(base) and not c.endswith("x"):
            r["name_en"] = en_of[base]
        if c.endswith("x"):
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    pts_out += border_rows
    OUT.mkdir(parents=True, exist_ok=True)
    stampd = {"endpoint": f"Тарифное руководство № 4, {b1name}, {b2name}",
              "fetched": date.today().isoformat()}
    (OUT / "sections.json").write_text(json.dumps({**stampd, "rows": rows}, ensure_ascii=False),
                                       "utf-8")
    (OUT / "points.json").write_text(json.dumps({**stampd, "rows": pts_out}, ensure_ascii=False),
                                     "utf-8")
    (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    print(f"wrote {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
          f"({sum(1 for r in pts_out if r['type'] != '80')} stops, "
          f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced) -> {OUT}")


# The feed's one agency, named with the six regional railways that run the suburban trains:
# gtfs_served treats a line whose manager no agency names as run by an operator the feed may
# lack, and leaves its sections with no train "unknown" instead of closing them.
AGENCY = ("Укрзалізниця (Південно-Західна, Львівська, Одеська, Південна, Придніпровська, "
          "Донецька залізниці)")
# The timetable match: what leaving a call unmatched costs, in km of path. A place on the way
# adds next to nothing to the path, so it is always taken; a same-named place off the route
# adds twice its distance from it, so one more than about SKIP_KM / 2 off is left out. (At
# 25 km a long-distance train's whole path cost more than skipping its calls.)
SKIP_KM = 150.0
MAX_HOP_KM = 900.0        # consecutive matched calls further apart than this are not joined


def timetable():
    """poizdato.net's train pages as a GTFS feed, data/raw/gtfs/ua/ua_poizdato.gtfs.zip, which
    gtfs_served reads like any national feed; and data/raw/ua/timetable_calls.json, the
    places a train calls at, which --convert reads to tell served points.

    Each call's station name is matched to OSM's rail stations (every name a node carries:
    name, name:uk, name:ru, old_name) and to Wikidata's stations with an ESR code, by
    name_keys, the strongest key that finds anything; of the places a name finds, the train's
    route decides: a dynamic programme picks one place per call (or none, at SKIP_KM) so that
    the train's path from call to call is shortest. Calls abroad (Przemyśl, Chełm, Záhony,
    Košice) find nothing in Ukraine and are left out; the international trains come into the
    feed folder separately (data/raw/ua/uz_jbb.gtfs.zip, copied by --timetable)."""
    import csv
    import io
    import shutil
    import zipfile
    from collections import Counter
    from datetime import date
    files = sorted(f for f in (RAW / "poizdato").glob("rozklad-*/*.html")
                   if f.parent.name in ("rozklad-elektrychky", "rozklad-poizda"))
    trains = [parse_train(f) for f in files]
    for t, f in zip(trains, files):
        t["slug"] = f.stem
    ost = load_osm_stations()
    places = [(s["lon"], s["lat"], s["name"]) for s in ost]
    nidx = NameIndex()
    for i, s in enumerate(ost):
        for nm in s["names"]:
            nidx.add(nm, s["lon"], s["lat"], i)
    for c, w in wd_stations().items():
        if w["lon"] is not None and w["uk"]:
            places.append((w["lon"], w["lat"], w["uk"]))
            nidx.add(w["uk"], w["lon"], w["lat"], len(places) - 1)
    stat = Counter()
    out_trips = []
    for t in trains:
        calls = t["calls"]
        cands = []
        for c in calls:
            got = nidx.find(c["name"])
            # several records of one place (an OSM node and its Wikidata item): one each
            seen, cc = [], []
            for lon, lat, ref, _r in got:
                if all(dist_m(lon, lat, x, y) > 300 for x, y in seen):
                    seen.append((lon, lat))
                    cc.append(ref)
            cands.append(cc[:80])
        n = len(calls)
        INF = float("inf")
        best = [dict() for _ in range(n)]          # i -> {ref: (cost, prev (j, ref))}
        for i in range(n):
            for r in cands[i]:
                bc, bp = SKIP_KM * i, None
                for j in range(max(0, i - 5), i):
                    for r2, (c2, _p) in best[j].items():
                        d = dist_m(*places[r][:2], *places[r2][:2]) / 1000
                        if d > MAX_HOP_KM:
                            continue
                        v = c2 + d + SKIP_KM * (i - j - 1)
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
        stat["calls"] += n
        stat["matched"] += len(chosen)
        stat["no candidate"] += sum(1 for c in cands if not c)
        if len(chosen) < 2:
            stat["trains with fewer than 2 calls matched"] += 1
            continue
        out_trips.append((t, [(i, chosen[i]) for i in sorted(chosen)]))
    print(f"timetable: {len(trains)} train pages; {dict(stat)}")
    unmatched = Counter(c["name"] for t in trains for i, c in enumerate(t["calls"])
                        if not any(i == j for tt, ch in out_trips if tt is t for j, _r in ch))
    print("  most frequent unmatched calls: " + ", ".join(
        f"{k} ({v})" for k, v in unmatched.most_common(40)))
    days_n = Counter(min(len(t["days"]) // 10 * 10, 60) for t, _c in out_trips)
    print(f"  running days shown (Oct-Nov 2026), trains by tens: {sorted(days_n.items())}")

    # --- the feed
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    used = sorted({r for _t, ch in out_trips for _i, r in ch})
    calls_xy = sorted({(round(places[r][0], 6), round(places[r][1], 6)) for r in used})
    (RAW / "timetable_calls.json").write_text(json.dumps(calls_xy), "utf-8")
    # consecutive calls of some train that runs, as index pairs into calls_xy
    at = {xy: i for i, xy in enumerate(calls_xy)}
    pairs = set()
    for t, ch in out_trips:
        if not t["days"]:
            continue
        for (_i, r1), (_j, r2) in zip(ch, ch[1:]):
            a = at[(round(places[r1][0], 6), round(places[r1][1], 6))]
            b = at[(round(places[r2][0], 6), round(places[r2][1], 6))]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    (RAW / "timetable_pairs.json").write_text(json.dumps(sorted(pairs)), "utf-8")

    def tbl(header, rows):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return b.getvalue()
    stops_rows = [(f"p{r}", places[r][2], f"{places[r][1]:.6f}", f"{places[r][0]:.6f}")
                  for r in used]
    routes, trips, st, cal = [], [], [], []
    for t, ch in out_trips:
        rid = t["slug"]
        rtype = "109" if t["kind"] == "rozklad-elektrychky" else "102"
        routes.append((rid, "uz", t["num"], t["title"][:120], rtype))
        trips.append((rid, rid, rid))
        prev = -1
        for seq, (i, r) in enumerate(ch, 1):
            tm = t["calls"][i]["times"]
            arr, dep = (tm[0], tm[-1]) if tm else ("", "")

            def secs(x):
                h, m = x.split(":")
                return int(h) * 3600 + int(m) * 60
            if arr:
                a, d = secs(arr), secs(dep)
                while a < prev:
                    a += 86400
                while d < a:
                    d += 86400
                prev = d
                arr = f"{a // 3600:02d}:{a % 3600 // 60:02d}:00"
                dep = f"{d // 3600:02d}:{d % 3600 // 60:02d}:00"
            st.append((rid, arr, dep, f"p{r}", seq))
        for d in t["days"]:
            cal.append((rid, d.replace("-", ""), 1))
    zp = GTFS_DIR / "ua_poizdato.gtfs.zip"
    tmp = GTFS_DIR / "ua_poizdato.gtfs.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agency.txt", tbl(("agency_id", "agency_name", "agency_url", "agency_timezone"),
                                     [("uz", AGENCY, "https://uz.gov.ua/", "Europe/Kyiv")]))
        z.writestr("stops.txt", tbl(("stop_id", "stop_name", "stop_lat", "stop_lon"), stops_rows))
        z.writestr("routes.txt", tbl(("route_id", "agency_id", "route_short_name",
                                      "route_long_name", "route_type"), routes))
        z.writestr("trips.txt", tbl(("route_id", "service_id", "trip_id"), trips))
        z.writestr("stop_times.txt", tbl(("trip_id", "arrival_time", "departure_time", "stop_id",
                                          "stop_sequence"), st))
        z.writestr("calendar_dates.txt", tbl(("service_id", "date", "exception_type"), cal))
        z.writestr("feed_info.txt", tbl(
            ("feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_version"),
            [("noritetsu, from poizdato.net's train pages", "https://poizdato.net/", "uk",
              date.today().isoformat())]))
    os.replace(tmp, zp)
    jbb = RAW / "uz_jbb.gtfs.zip"
    if jbb.exists():
        shutil.copyfile(jbb, GTFS_DIR / "ua_ukrzaliznytsya.gtfs.zip")
    print(f"timetable: {len(out_trips)} trains, {len(used)} stations, {len(cal)} train-days "
          f"-> {zp}" + ("; + the international feed" if jbb.exists() else ""))


# Tariff sections have no colours of their own, nor does any widely used map colour
# Ukrzaliznytsia's railways one by one, so each register line takes its regional railway's
# colour, picked here (as Russia's are), no two neighbouring railways alike. `picked` in
# colours/ua.csv.
ROAD_COLOUR = {
    "32": ("2F7FD8", "blue"),         # Південно-Західна
    "35": ("D7263D", "red"),          # Львівська
    "40": ("1E9E8F", "teal"),         # Одеська
    "43": ("E08E0B", "orange"),       # Південна
    "45": ("8A5CC2", "purple"),       # Придніпровська
    "48": ("3FA34D", "green"),        # Донецька
}


def colours():
    """colours/ua.csv: one row per register line in names.json, in its railway's colour."""
    import csv
    names = json.loads((OUT / "names.json").read_text("utf-8"))
    rows = []
    for sid, e in sorted(names.items()):
        road = e["road"]
        col, hue = ROAD_COLOUR[road]
        op = {c: n for c, n in UA_ROADS.values()}[road]
        rows.append({"line": e["name"], "operator": op, "colour": "#" + col, "source": "picked",
                     "url": "", "note": f"{op}: one {hue} per regional railway"})
    path = ROOT / "colours" / "ua.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["line", "operator", "colour", "source", "url", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"--colours: {len(rows)} register lines -> {path}")


def Counter_(it):
    from collections import Counter
    return Counter(it)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esr", metavar="PBF")
    ap.add_argument("--disused", metavar="PBF")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--crawl-trains", action="store_true")
    ap.add_argument("--crawl-stations", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--timetable", action="store_true")
    ap.add_argument("--colours", action="store_true")
    args = ap.parse_args()
    if args.esr:
        p = Path(args.esr)
        esr_pass(p if p.is_absolute() else ROOT / p)
    if args.disused:
        p = Path(args.disused)
        disused_pass(p if p.is_absolute() else ROOT / p)
    if args.outline:
        write_outline()
    if args.clip:
        clip()
    if args.crawl_stations:
        crawl(["rozklad-po-stantsii"], args.limit)
    if args.crawl_trains:
        crawl(["rozklad-elektrychky", "rozklad-poizda"], args.limit)
    if args.timetable:
        timetable()
    if args.convert:
        convert()
    if args.colours:
        colours()
