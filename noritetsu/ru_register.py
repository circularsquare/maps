"""Russia: Tariff Guide No. 4 (Тарифное руководство № 4) written into rinf.py's input format.

    $env:OSMIUM_POOL_THREADS=2
    python extract.py --region ru/full --pbf data/raw/ru/russia-YYMMDD.osm.pbf
    python extract.py --region ru/ua --pbf data/raw/ru/ukraine-YYMMDD.osm.pbf --bbox 32.0,45.9,40.3,50.2
    python ru_register.py --esr data/raw/ru/russia-YYMMDD.osm.pbf        # esr:user codes, before deleting the .pbf
    python ru_register.py --esr data/raw/ru/ukraine-YYMMDD.osm.pbf --ua  # same for the 2022-annexed areas
    python ru_register.py --annex        # data/raw/ru/annex.geojson: the annexed area under Russian control
    python ru_register.py --clip         # after every extract: ru/full + ru/ua -> data/proc/ru, clipped
    python ru_register.py --wikidata     # line and station labels, lengths (cached in data/raw/ru/wdx_*.json)
    python ru_register.py --convert      # data/raw/rinf/ru/{sections,points,names}.json
    python build_model.py --region ru --register rinf:data/raw/rinf/ru

ru_sources.md has the research, the sources and the numbers; this docstring is how it works.

THE REGISTER.  Book 1 of the tariff guide lists, per railway, every tariff section (участок)
between two tariff nodes with every station, passing loop, post and halt on it in order, each
with its six-digit ESR code and integer tariff km. One tariff section is one register line
(Anita, 2026-10-01): "01-011", named after its two ends, "Обухово — Чудово-Московское". Each
consecutive pair of points is one rinf section row with the difference of their tariff km as
its length, and rinf.py does the rest (tracing over OSM track with a length check, stops
matched to OSM stations, junction-ended sections left to OSM's routes). rinf_countries/ru.py
holds the settings.

POINTS.  Placed at the OSM node carrying their ESR code (`esr:user`, read straight from the
.pbf by --esr, since extract.py does not keep the tag); else at an OSM station of the same
name near the point's placed neighbours on the section; else left for rinf.py's own name
placement. A point is a passenger stop (rinf type 10 or 70) when Book 2 lists a passenger
operation for it (П, Б or О) AND an OSM train route stops within SERVED_M of it. Book 2's
letters are permissions, not service: 93% of all points carry one, freight branches included,
so without OSM's routes every freight branch would be a stop-to-stop section that build_model
never questions. Everything else is a junction (type 80), which rinf merges away inside a line
and build_model keeps at a line's end only where OSM passenger routes run over it.

LEFT OUT OR CHANGED.
- Sections whose km are all 0: node lists (17-150 is the Moscow Central Circle's stations, all
  at 0 km), which stay OSM lines.
- A point that repeats its neighbour under an extra code, "Левшино (эксп.)" beside "Левшино" at
  the same km (export, transshipment and border-junction codes of one station).
- A pair of points listed on two sections (a passenger-only variant beside its main section,
  Kaliningrad's five sections leaving Kaliningrad-Passazhirsky over the same first kilometres)
  stays on one: main sections first, then the lower id. Otherwise riding it would count twice.
- Pairs with a point outside Russia's outline (`outline()`: OSM's boundary of Russia plus the
  annex polygon below): Trans-Siberian sections through Kazakhstan, the border stubs. rinf.py
  splits what remains into pieces.
- A stop-to-stop stretch of UNSERVED_KM past an unserved passenger point, or BARE_KM with
  none, has its end stops cloned as junctions on that line, so it answers to OSM's routes like
  any junction-ended section (see convert()).

THE 2022-ANNEXED RAILWAYS (Донецкая, Луганская, Мелитопольская-Херсонская; Anita, 2026-10-01:
with Russia, de facto). Book 1 lists the whole pre-war Donetsk railway, Kramatorsk and
Sloviansk included, which Ukraine holds and Ukrzaliznytsia serves. So the clip polygon there is
the area under Russian control (`--annex`): the four oblasts (OCHA COD-AB admin 1) cut to the
Voronoi cells of the settlements en.wikipedia's war maps mark as Russian-held
(Module:Russo-Ukrainian war overview map and ... detailed map, CC BY-SA; contested places count
as not held). OSM has NO passenger route relations there (checked 2026-10-01: only
Ukrzaliznytsia's routes on the Ukrainian side and Crimea's Armyansk trains), so the OSM test
above would drop every section. Points there are stops on Book 2's letters alone, and
rinf_countries/ru.py's `suspended` greys the lines (ANNEX_RUNNING switches that off). OSM
station names there are mostly Ukrainian; --clip puts their name:ru in `name` (from --esr --ua)
so the register's Russian names match and show.

ESR codes in the annexed railways are Russia's new ones (89xxxx, 84xxxx, 82xxxx); OSM still
carries Ukrzaliznytsia's (48xxxx), so those points are placed by name (name:ru).
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ru"
PROC = ROOT / "data" / "proc" / "ru"
OUT = ROOT / "data" / "raw" / "rinf" / "ru"
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"
UA_ADMIN1 = ROOT.parent / "data" / "asia1m" / "ukraine" / "ukr_admin1.shp"
USER_AGENT = "noritetsu-rail-map/1.0"

# Book 1's sheets of the Russian administration: sheet -> (road code, the railway's name).
ROADS = {
    "Окт (Р)": ("01", "Октябрьская железная дорога"),
    "Моск (Р)": ("17", "Московская железная дорога"),
    "Горьк (Р)": ("24", "Горьковская железная дорога"),
    "Сев (Р)": ("28", "Северная железная дорога"),
    "С-Кав (Р)": ("51", "Северо-Кавказская железная дорога"),
    "Ю-Вост (Р)": ("58", "Юго-Восточная железная дорога"),
    "Прив (Р)": ("61", "Приволжская железная дорога"),
    "Кбш (Р)": ("63", "Куйбышевская железная дорога"),
    "Сверд (Р)": ("76", "Свердловская железная дорога"),
    "Ю-Ур (Р)": ("80", "Южно-Уральская железная дорога"),
    "З-Сиб (Р)": ("83", "Западно-Сибирская железная дорога"),
    "Крас (Р)": ("88", "Красноярская железная дорога"),
    "В-Сиб (Р)": ("92", "Восточно-Сибирская железная дорога"),
    "Заб (Р)": ("94", "Забайкальская железная дорога"),
    "Д-Вост (Р)": ("96", "Дальневосточная железная дорога"),
    "Клг (Р)": ("10", "Калининградская железная дорога"),
    "Якут (Р)": ("91", "Железные дороги Якутии"),
    "ИФР-1 (Р)": ("97", "ИФР-1"),
    "Крым (Р)": ("85", "Крымская железная дорога"),
    "Донец (Р)": ("89", "Донецкая железная дорога"),
    "ЛУГАН (Р)": ("84", "Луганская железная дорога"),
    "МЕЛИТ (Р)": ("82", "Мелитопольская-Херсонская железная дорога"),
}
ANNEXED = {"Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)"}
# Oblasts annexed in 2022 (COD-AB pcodes): Donetsk, Luhansk, Zaporizhzhia, Kherson.
ANNEX_OBLASTS = {"UA14", "UA44", "UA23", "UA65"}
ANNEX_BBOX = (32.0, 45.9, 40.3, 50.2)          # what extract.py --bbox takes from Ukraine's .pbf

SERVED_M = 400            # an OSM train route stopping this close makes a flagged point a stop
UNSERVED_KM = 10          # a stop-to-stop stretch this long past a flagged, unserved point ...
BARE_KM = 25              # ... or this long with no passenger point at all answers to OSM routes
NAME_REACH_KM = 10        # a name match may lie this much further than the tariff km say
TYPE_RANK = {"Основной тарифный участок": 0, "Линии Московского узла": 1,
             "Линии Санкт-Петербургского узла": 1, "Малодеятельный": 2,
             "Соединительная линия": 3, "Для местного сообщения": 4,
             "Для пассажирского движения": 5, "Строящиеся линии": 6}

HEAD = re.compile(r'^\s*\d+\)\s*участок\s+(\d+-\d+)\s+"(.*)"\s*(?:\((.*)\))?\s*$')
KM = re.compile(r"^\s*(-?\d+)\s*км")
EXTRA_CODE = re.compile(r"\((?:эксп|перев|стык)[^)]*\)", re.I)


def log(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def newest(pattern):
    files = sorted(RAW.glob(pattern))
    if not files:
        sys.exit(f"no {pattern} in {RAW}")
    return files[-1]


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


# ================================================================ tariff guide

def book1():
    """Every tariff section on the Russian-administration sheets, points in order."""
    import xlrd
    path = newest("tr4_kniga1_*.xls")
    wb = xlrd.open_workbook(str(path))
    out = []
    for sh in wb.sheets():
        if sh.name not in ROADS:
            continue
        cur = None
        for r in range(sh.nrows):
            row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
            m = HEAD.match(row[0])
            if m:
                cur = {"sheet": sh.name, "id": m.group(1), "name": m.group(2),
                       "type": (m.group(3) or "").strip(), "points": []}
                out.append(cur)
                continue
            if cur is None or row[0].startswith("№"):
                continue
            code = row[1].replace(".", "").strip()
            if re.fullmatch(r"\d{6}", code):
                kms = [KM.match(v) for v in row[3:]]
                kms = [int(k.group(1)) if k else None for k in kms]
                cur["points"].append({"esr": code, "name": row[2], "km": kms})
            elif not row[0] and not row[1] and row[2] and cur["points"]:
                cur["points"][-1]["name"] += " " + row[2]      # a wrapped name
    return out, path.name


def book2():
    """ESR code -> its operations, from part 1 (separation points) and part 2 (halts)."""
    import xlrd
    path = newest("tr4_kniga2_*.xls")
    wb = xlrd.open_workbook(str(path))
    ops = {}
    sh = wb.sheet_by_name("РП")
    last = None
    for r in range(6, sh.nrows):
        row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
        if row[0]:
            last = row[5]
            ops[last] = row[2]
        elif last and row[2]:
            ops[last] += row[2]                           # wrapped operations
    sh = wb.sheet_by_name("ОП")
    for r in range(6, sh.nrows):
        row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
        if row[0] and row[4]:
            ops.setdefault(row[4], row[2])
    return ops, path.name


def passenger_op(ops_text):
    return any(x in ("П", "Б", "О") for x in re.split(r"[\s,]", ops_text or ""))


# ================================================================ names

def clean(name):
    """A point's name as shown: TR-4's own, less the "ОП" (halt) prefix and doubled spaces."""
    n = re.sub(r"\s+", " ", name or "").strip()
    n = re.sub(r"^(?:ОП|О\.П\.)\s+", "", n)
    return n


def nkey(s):
    """A name as a matching key: case, ё, halt prefixes and point-type brackets folded."""
    s = unicodedata.normalize("NFKC", s or "").lower().replace("ё", "е")
    s = re.sub(r"^(оп|о\.п\.|ост\.?\s*пункт|остановочный пункт|платформа|станция)\s+", "", s)
    s = re.sub(r"\((рзд|бп|пп|п|обп|эксп\.?|перев\.?|стык|узк\.?)\)", "", s)
    s = re.sub(r"\bразъезд\b|\bрзд\b", "", s)
    s = re.sub(r"[\s\-–—.,]+", " ", s).strip()
    return s


def base_name(n):
    """'Санкт-Петербург-Главный' -> 'санкт-петербург'; 'Волховстрой I' -> 'волховстрой'. For
    finding a Wikidata line item named after a section's two ends (probe_ru_wikidata.base)."""
    n = n.lower().replace("ё", "е")
    n = re.sub(r"^(оп|о\.п\.)\s+", "", n.strip())
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"\s+(i{1,3}|iv|[1-4])$", "", n.strip())
    n = re.sub(r"-(пассажирский|пассажирская|главный|московский|московское|сортировочная|"
               r"сортировочный|товарный|товарная|город|балтийский|витебский|финляндский|"
               r"ладожский|белорусский|курский|ярославский|казанский|киевский|рижский|"
               r"павелецкий|савеловский|ленинградский|восточный|западный|северный|южный)$",
               "", n.strip())
    return n.strip(" -")


VIA = re.compile(r"\((?:ЧЕРЕЗ|через)\s+((?:[^()]|\([^()]*\))*)\)?")
SMALL = {"через", "ст.", "ст", "и", "оп", "о.п.", "км", "рзд"}


def tidy_caps(s):
    """'СТ. ПРИМОРСК-НОВЫЙ' -> 'ст. Приморск-Новый': title case for a header's capitals."""
    out = []
    for w in s.split():
        lw = w.lower()
        if re.fullmatch(r"[IVX]+,?", w):
            out.append(w)
        elif lw in SMALL or lw.rstrip(",") in SMALL:
            out.append(lw)
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in lw.split("-")))
    return " ".join(out)


def section_name(s):
    """"Обухово — Чудово-Московское": the section's two end points by TR-4's own spelling,
    with the via note of the header where it has one, since some pairs of nodes have two
    sections ("Царевщина — Красная Глинка (через ст. Безымянка, Жигулёвское Море)")."""
    a, b = clean(s["points"][0]["name"]), clean(s["points"][-1]["name"])
    # The header's two ends, spelled as the point list spells them: lists can run on past
    # the node under extra codes (10-002 ends "Калининград-Пассажирский", then
    # "Калининград-Сортировочный (эксп.)" and "(перев.)" at the same km).
    hdr = VIA.sub("", s["name"])
    hdr = re.sub(r"\((?:[^()]*ж\.\s*д\.?|[^()]*(?:ПАССАЖИР|пассажир)[^()]*)\)", "", hdr)
    ends = [x.strip() for x in re.split(r"\s+-\s+", hdr)]
    if len(ends) == 2 and all(ends):
        by_key = {}
        for p in s["points"]:
            by_key.setdefault(nkey(p["name"]), clean(p["name"]))
            by_key.setdefault(nkey(EXTRA_CODE.sub("", p["name"])), clean(p["name"]))
        a, b = (by_key.get(nkey(e)) or by_key.get(nkey(EXTRA_CODE.sub("", e))) or tidy_caps(e)
                for e in ends)
    name = f"{a} — {b}"
    m = VIA.search(s["name"])
    if m:
        via = re.split(r"\s*\(", m.group(1))[0].strip()
        via = tidy_caps(via) if via.upper() == via else via
        name += f" (через {via[0].lower() + via[1:] if via.startswith('Ст') else via})"
    return name


# ================================================================ OSM side

def esr_pass(pbf, ua=False):
    """esr:user codes from a .pbf (extract.py does not keep the tag), and for the Ukraine
    extract also every rail station node in ANNEX_BBOX with its name:ru."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    out, names = defaultdict(list), []
    w, s, e, n = ANNEX_BBOX
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("esr:user", "esr", "railway:esr", "railway",
                                               "public_transport")))
    for obj in fp:
        t = obj.tags
        code = t.get("esr:user") or t.get("esr") or t.get("railway:esr")
        lon, lat = round(obj.location.lon, 6), round(obj.location.lat, 6)
        if code:
            for c in code.replace(",", ";").split(";"):
                c = c.strip()
                if c.isdigit() and len(c) == 6:
                    out[c].append(["n", obj.id, lon, lat, t.get("name"), t.get("railway"),
                                   t.get("public_transport"), t.get("train")])
        if ua and w <= lon <= e and s <= lat <= n and (
                t.get("railway") in ("station", "halt", "stop")
                or t.get("public_transport") in ("station", "stop_position")):
            names.append([obj.id, lon, lat, t.get("name"), t.get("name:ru"), t.get("name:uk"),
                          t.get("railway")])
    path = RAW / ("osm_esr_ua.json" if ua else "osm_esr.json")
    path.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    log(f"--esr: {sum(len(v) for v in out.values())} nodes, {len(out)} codes -> {path.name}")
    if ua:
        (RAW / "osm_names_ua.json").write_text(json.dumps(names, ensure_ascii=False), "utf-8")
        log(f"--esr: {len(names)} stop nodes in {ANNEX_BBOX}, "
            f"{sum(1 for x in names if x[4])} with name:ru -> osm_names_ua.json")


# ================================================================ the outline

WP_MODULES = {"wp_overview_map.lua": "Module:Russo-Ukrainian_war_overview_map",
              "wp_detailed_map.lua": "Module:Russo-Ukrainian_war_detailed_map"}
MARK = re.compile(r'^\s*\{\s*lat\s*=\s*"([-\d.]+)",\s*long\s*=\s*"([-\d.]+)",\s*mark\s*=\s*'
                  r'(mk\.\w+|"[^"]*")(.*)$')
MARK_SIDE = {"mk.rus": "R", '"Location dot red.svg"': "R",
             "mk.ukr": "U", '"Location dot blue.svg"': "U",
             "mk.con": "C", "mk.shr": "C", '"80x80-red-blue-anim.gif"': "C",
             '"Map-ctl2-red+blue.svg"': "C"}


def wp_marks():
    """Settlements en.wikipedia's war maps mark as held by one side, (lon, lat, side, label).
    Fetched once and kept in data/raw/ru/ with the date; delete them to refresh."""
    marks = []
    for fn, title in WP_MODULES.items():
        p = RAW / fn
        if not p.exists():
            url = ("https://en.wikipedia.org/w/index.php?" +
                   urllib.parse.urlencode({"title": title, "action": "raw"}))
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = r.read().decode("utf-8")
            p.write_text(f"-- fetched {date.today().isoformat()} from {url}\n" + txt, "utf-8")
        for line in p.read_text("utf-8").splitlines():
            m = MARK.match(line)
            if not m:
                continue
            side = MARK_SIDE.get(m.group(3))
            if side:
                lab = re.search(r'label\s*=\s*"\[\[([^|\]]+)', m.group(4))
                marks.append((float(m.group(2)), float(m.group(1)), side,
                              lab.group(1) if lab else ""))
    return marks


def annex():
    """data/raw/ru/annex.geojson: the 2022-annexed oblasts where Russia holds them (see the
    docstring). Voronoi cells in a frame scaled by cos(48°), so they are cut at true
    perpendicular bisectors."""
    import shapefile
    import shapely
    from shapely.geometry import MultiPoint, Point, mapping, shape
    from shapely.ops import unary_union
    rd = shapefile.Reader(str(UA_ADMIN1))
    fields = [f[0] for f in rd.fields[1:]]
    obl = []
    for sr in rd.shapeRecords():
        rec = dict(zip(fields, sr.record))
        if rec["adm1_pcode"] in ANNEX_OBLASTS:
            obl.append(shape(sr.shape.__geo_interface__))
    obl = unary_union(obl)
    marks = wp_marks()
    k = math.cos(math.radians(48.0))
    x0, y0, x1, y1 = obl.buffer(1.0).bounds
    near = [(x, y, sd, lb) for x, y, sd, lb in marks if x0 <= x <= x1 and y0 <= y <= y1]
    # One mark per place (both modules mark some): the detailed module's, which comes second.
    seen = {}
    for x, y, sd, lb in near:
        seen[(round(x, 3), round(y, 3))] = (x, y, sd, lb)
    near = list(seen.values())
    pts = MultiPoint([(x * k, y) for x, y, _s, _l in near])
    env = shapely.box(x0 * k, y0, x1 * k, y1)
    cells = shapely.voronoi_polygons(pts, extend_to=env)
    held = []
    for cell in cells.geoms:
        for x, y, sd, _lb in near:
            if sd == "R" and cell.contains(Point(x * k, y)):
                held.append(cell)
                break
    held = shapely.transform(unary_union(held), lambda c: c / np.array([k, 1.0]))
    area = held.intersection(obl).buffer(0)
    cnt = Counter(sd for _x, _y, sd, _l in near)
    log(f"--annex: {len(near)} marked places in and around the four oblasts ({dict(cnt)}); "
        f"held area {area.area / obl.area:.0%} of the oblasts")
    feat = {"type": "Feature", "properties": {
        "what": "2022-annexed oblasts (Donetsk, Luhansk, Zaporizhzhia, Kherson) where Russia "
                "holds them: Voronoi cells of en.wikipedia war-map marks (CC BY-SA), cut to "
                "OCHA COD-AB admin 1", "built": date.today().isoformat(),
        "marks": dict(cnt)}, "geometry": mapping(area)}
    (RAW / "annex.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [feat]}), "utf-8")
    return area


COAST_DEG = 0.05          # the outline reaches this far out to sea (about 3-5 km)


def outline():
    """Russia as the build sees it: OSM's own boundary of Russia (relation 60189, from
    polygons.openstreetmap.fr: data/raw/ru/ru_boundary.geojson), which holds Crimea and the
    territorial sea, plus annex.geojson. religiondots' and Natural Earth's outlines are too
    generalised at land borders: both put Bagrationovsk station, 2 km from the Polish border,
    in Poland. Without the OSM file: religiondots' `ru` reaching COAST_DEG out to sea."""
    import shapely
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    bp = RAW / "ru_boundary.geojson"
    if bp.exists():
        parts = [shape(json.loads(bp.read_text("utf-8")))]
        ap = RAW / "annex.geojson"
        if ap.exists():
            parts += [shape(f["geometry"]) for f in
                      json.loads(ap.read_text("utf-8"))["features"]]
        got = unary_union(parts)
        polys = [g for g in shapely.get_parts(got) if g.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.multipolygons(shapely.get_parts(shapely.geometrycollections(polys)))
    feats = json.loads(SHAPES.read_text(encoding="utf-8"))["features"]
    parts = [shape(f["geometry"]) for f in feats if f["properties"].get("cc") == "ru"]
    ap = RAW / "annex.geojson"
    if ap.exists():
        parts += [shape(f["geometry"]) for f in
                  json.loads(ap.read_text("utf-8"))["features"]]
    ru = unary_union(parts)
    reach = ru.buffer(COAST_DEG)
    others = [shape(f["geometry"]) for f in feats if f["properties"].get("cc") != "ru"]
    tree = shapely.STRtree(others)
    near = [others[i] for i in tree.query(reach)]
    others = unary_union([g.intersection(reach) for g in near]).difference(ru)
    got = ru.union(reach.difference(others))
    # Polygons only: the overlay leaves slivers and lines, and contains_xy on a
    # GeometryCollection is not prepared (13,000 points took 4 minutes).
    polys = [g for g in shapely.get_parts(got) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.multipolygons(shapely.get_parts(shapely.geometrycollections(polys)))


def clip():
    """data/proc/ru from the two extracts, data/proc/ru/full (Russia's .pbf) and
    data/proc/ru/ua (Ukraine's, ANNEX_BBOX): merged, then what lies in `outline()` kept. A way
    goes if at least half its nodes are outside, a stop if it is, a relation if no member is
    left. Station names inside the annex polygon take their name:ru. The two sources are left
    as they are, so this can be rerun after changing the outline; run it after every extract."""
    import shapely
    from shapely.geometry import shape

    def rd(d, fn):
        with open(d / fn, "rb") as f:
            return pickle.load(f)
    full = PROC / "full"
    ways, rels, stops, infra = (rd(full, f) for f in
                                ("ways.pkl", "rels.pkl", "stops.pkl", "infra.pkl"))
    with np.load(full / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    ua = PROC / "ua"
    if (ua / "ways.pkl").exists():
        n0 = (len(ways), len(rels), len(stops))
        ways.update(rd(ua, "ways.pkl"))
        rels.update(rd(ua, "rels.pkl"))
        stops.update(rd(ua, "stops.pkl"))
        infra.update(rd(ua, "infra.pkl"))
        with np.load(ua / "coords.npz") as u:
            allid = np.concatenate([cid, u["id"]])
            allx = np.concatenate([cx, u["x"]])
            ally = np.concatenate([cy, u["y"]])
        allid, first = np.unique(allid, return_index=True)
        cid, cx, cy = allid, allx[first], ally[first]
        log(f"clip: merged data/proc/ru/ua: ways {n0[0]} -> {len(ways)}, relations {n0[1]} -> "
            f"{len(rels)}, stops {n0[2]} -> {len(stops)}, coords {cid.size}")
    shp = outline()
    shapely.prepare(shp)
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
    # Russian names in the annexed area (docstring).
    n_ru = 0
    ap = RAW / "annex.geojson"
    nu = RAW / "osm_names_ua.json"
    if ap.exists() and nu.exists():
        ann = shapely.union_all([shape(f["geometry"]) for f in
                                 json.loads(ap.read_text("utf-8"))["features"]])
        shapely.prepare(ann)
        ru_name = {x[0]: x[4] for x in json.loads(nu.read_text("utf-8")) if x[4]}
        for k, (tags, lon, lat) in keep_s.items():
            nm = ru_name.get(k)
            if nm and nm != tags.get("name") and shapely.contains_xy(ann, lon, lat):
                tags = dict(tags)
                tags.setdefault("name:uk", tags.get("name"))
                tags["name"] = nm
                keep_s[k] = (tags, lon, lat)
                n_ru += 1
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1])[:20]:
        log(f"  cut {v:5d}  {name}")
    log(f"clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations; "
        f"{n_ru} stop names in the annexed area set from name:ru")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = PROC / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, PROC / fn)
    tmp = PROC / "coords.tmp.npz"
    np.savez_compressed(tmp, id=cid, x=cx, y=cy)
    os.replace(tmp, PROC / "coords.npz")


# ================================================================ Wikidata

WIKIDATA = "https://query.wikidata.org/sparql"
Q_LINES = """SELECT ?x ?ru ?en ?len ?unit ?osm WHERE {
  ?x wdt:P17 ?c ; wdt:P31 ?cls . VALUES ?c { wd:Q159 wd:Q212 } ?cls wdt:P279* wd:Q728937 .
  OPTIONAL { ?x rdfs:label ?ru FILTER(LANG(?ru) = "ru") }
  OPTIONAL { ?x rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?x p:P2043/psv:P2043 [ wikibase:quantityAmount ?len ; wikibase:quantityUnit ?unit ] }
  OPTIONAL { ?x wdt:P402 ?osm } }"""
Q_STATIONS = """SELECT ?s ?esr ?ru ?en WHERE {
  ?s wdt:P2815 ?esr .
  OPTIONAL { ?s rdfs:label ?ru FILTER(LANG(?ru) = "ru") }
  OPTIONAL { ?s rdfs:label ?en FILTER(LANG(?en) = "en") } }"""


def sparql(q, tries=3):
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode()
    for i in range(tries):
        req = urllib.request.Request(WIKIDATA, data=data, headers={
            "User-Agent": USER_AGENT, "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                rows = json.load(r)["results"]["bindings"]
            return [{k: v["value"].rsplit("/", 1)[-1] if v["value"].startswith(
                "http://www.wikidata.org/entity/") else v["value"] for k, v in b.items()}
                for b in rows]
        except Exception as e:                                   # noqa: BLE001
            log(f"  Wikidata retry {i}: {type(e).__name__} {str(e)[:100]}")
            time.sleep(15 * (i + 1))
    raise SystemExit("Wikidata query failed")


def wikidata():
    for key, q in (("lines", Q_LINES), ("stations", Q_STATIONS)):
        rows = sparql(q)
        (RAW / f"wdx_{key}.json").write_text(json.dumps(
            {"fetched": date.today().isoformat(), "rows": rows}, ensure_ascii=False), "utf-8")
        log(f"--wikidata: {len(rows)} {key} rows")
        time.sleep(3)


def wd_lines():
    """{frozenset of two base names: [(qid, ru label, en label, km)]} for 'A — B' items."""
    p = RAW / "wdx_lines.json"
    if not p.exists():
        return {}
    items = defaultdict(dict)
    for r in json.loads(p.read_text("utf-8"))["rows"]:
        e = items[r["x"]]
        e.setdefault("ru", r.get("ru"))
        e.setdefault("en", r.get("en"))
        if r.get("len") and r.get("unit") in ("Q828224", "Q11573"):
            e.setdefault("km", float(r["len"]) * (1.0 if r["unit"] == "Q828224" else 0.001))
    out = defaultdict(list)
    for q, e in items.items():
        parts = re.split(r"\s+[—–-]\s+", e.get("ru") or "")
        if len(parts) == 2:
            out[frozenset((base_name(parts[0]), base_name(parts[1])))].append(
                (q, e.get("ru"), e.get("en"), e.get("km")))
    return out


# ================================================================ convert

def osm_side():
    """What the conversion needs from the (clipped) extract: OSM stations by name key, the
    positions where train routes stop, and the outline test."""
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(PROC / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    by_name = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if (tags.get("railway") in ("station", "halt")
                or (tags.get("public_transport") == "station" and tags.get("train") == "yes")):
            for nm in (tags.get("name"), tags.get("name:uk")):
                if nm:
                    by_name[nkey(nm)].append((lon, lat, nid))
    served = []
    for tags, members in rels.values():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        for ty, ref, role in members:
            if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                served.append((stops[ref][1], stops[ref][2]))
    served = np.array(sorted(set(served))) if served else np.zeros((0, 2))
    return by_name, served


def convert():
    from scipy.spatial import cKDTree
    import shapely
    secs, b1name = book1()
    ops, b2name = book2()
    log(f"Book 1 ({b1name}): {len(secs)} sections; Book 2 ({b2name}): {len(ops)} points")
    esr = json.loads((RAW / "osm_esr.json").read_text("utf-8"))
    if (RAW / "osm_esr_ua.json").exists():
        for k, v in json.loads((RAW / "osm_esr_ua.json").read_text("utf-8")).items():
            esr.setdefault(k, []).extend(v)
    by_name, served = osm_side()
    lat0 = 55.0
    kx = 111.32 * math.cos(math.radians(lat0))
    stree = cKDTree(np.c_[served[:, 0] * kx, served[:, 1] * 110.57]) if len(served) else None
    shp = outline()
    shapely.prepare(shp)
    ann = None
    if (RAW / "annex.geojson").exists():
        from shapely.geometry import shape
        ann = shapely.union_all([shape(f["geometry"]) for f in json.loads(
            (RAW / "annex.geojson").read_text("utf-8"))["features"]])
        shapely.prepare(ann)
    wdl = wd_lines()

    def esr_pos(code):
        rows = esr.get(code) or []
        best = ([r for r in rows if r[5] in ("station", "halt")]
                or [r for r in rows if r[6] == "station"] or rows)
        return (best[0][2], best[0][3]) if best else None

    stat = Counter()
    # --- 1. clean each section's point list
    for s in secs:
        pts = []
        for p in s["points"]:
            p = dict(p, name=clean(p["name"]), km0=p["km"][0])
            # An extra code at the same km as its neighbour is that place again (10-002 lists
            # Калининград-Пассажирский, then Калининград-Сортировочный (эксп.) and (перев.),
            # all at 124 km).
            if pts and p["km0"] == pts[-1]["km0"] and EXTRA_CODE.search(p["name"]):
                stat["extra code dropped"] += 1
                continue
            if pts and p["km0"] == pts[-1]["km0"] and EXTRA_CODE.search(pts[-1]["name"]):
                stat["extra code dropped"] += 1
                pts[-1] = p
                continue
            pts.append(p)
        s["pts"] = pts
        # Last less first: some sections count from a node further back (96-029 Ружино -
        # Кабарга starts at 409 km).
        s["km"] = (pts[-1]["km0"] or 0) - (pts[0]["km0"] or 0) if pts else 0

    # --- 2. place the points
    pos, how = {}, {}
    for s in secs:
        for p in s["pts"]:
            if p["esr"] not in pos:
                q = esr_pos(p["esr"])
                if q:
                    pos[p["esr"]], how[p["esr"]] = q, "esr"
    # The annexed railways' codes are not in OSM at all, so they have no anchors to start
    # from: a station whose name only one place in the annexed area carries is placed there.
    # Ukraine's whole ANNEX_BBOX is searched (osm_names_ua.json, from --esr --ua), so points
    # on the Ukrainian side are placed too, and then fall outside the outline.
    ua_names = defaultdict(list)
    nu = RAW / "osm_names_ua.json"
    if nu.exists():
        for nid, lon, lat, nm, nm_ru, nm_uk, rw in json.loads(nu.read_text("utf-8")):
            if rw in ("station", "halt"):
                for k in {nkey(x) for x in (nm, nm_ru, nm_uk) if x}:
                    ua_names[k].append((lon, lat, nid))
    for s in secs:
        if s["sheet"] not in ANNEXED:
            continue
        for p in s["pts"]:
            if p["esr"] in pos:
                continue
            cand = ua_names.get(nkey(p["name"]), ())
            if cand and all(dist_m(*cand[0][:2], *q[:2]) <= 2000 for q in cand):
                pos[p["esr"]], how[p["esr"]] = cand[0][:2], "name"
    for k, v in ua_names.items():
        by_name.setdefault(k, v)
    for _round in range(3):
        for s in secs:
            pts = s["pts"]
            for i, p in enumerate(pts):
                if p["esr"] in pos:
                    continue
                cand = by_name.get(nkey(p["name"])) or by_name.get(
                    nkey(EXTRA_CODE.sub("", p["name"]))) or []
                if not cand:
                    continue
                anchors = []
                for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(pts))):
                    q = pts[j]
                    if q["esr"] in pos:
                        anchors.append((pos[q["esr"]], abs((q["km0"] or 0) - (p["km0"] or 0))))
                        if len(anchors) >= 2:
                            break
                if not anchors:
                    continue
                best = None
                for lon, lat, _nid in cand:
                    ok = all(dist_m(lon, lat, *a) / 1000 <= km + NAME_REACH_KM
                             for a, km in anchors)
                    d = min(dist_m(lon, lat, *a) for a, _km in anchors)
                    if ok and (best is None or d < best[0]):
                        best = (d, (lon, lat))
                if best:
                    pos[p["esr"]], how[p["esr"]] = best[1], "name"
    codes = {p["esr"] for s in secs for p in s["pts"]}
    log(f"points: {len(codes)} distinct; placed by ESR {sum(1 for c in codes if how.get(c) == 'esr')}, "
        f"by name {sum(1 for c in codes if how.get(c) == 'name')}, "
        f"unplaced {sum(1 for c in codes if c not in pos)}")
    acodes = {p["esr"] for s in secs if s["sheet"] in ANNEXED for p in s["pts"]}
    log(f"  of them on the annexed railways {len(acodes)}: by ESR "
        f"{sum(1 for c in acodes if how.get(c) == 'esr')}, by name "
        f"{sum(1 for c in acodes if how.get(c) == 'name')}, unplaced "
        f"{sum(1 for c in acodes if c not in pos)}")

    # --- 3. which points are inside, which are stops
    placed = sorted(c for c in codes if c in pos)
    xy = np.array([pos[c] for c in placed])
    in_out = dict(zip(placed, shapely.contains_xy(shp, xy[:, 0], xy[:, 1]).tolist()))
    in_ann = (dict(zip(placed, shapely.contains_xy(ann, xy[:, 0], xy[:, 1]).tolist()))
              if ann is not None else {})

    # An unplaced point is inside when its nearest placed neighbours on each of its sections
    # are: Kazakhstan's points on Trans-Siberian sections and Kramatorsk's on the Donetsk sheet
    # have no OSM node in the extracts, and would otherwise keep their whole stretch.
    for s in secs:
        pts = s["pts"]
        for i, p in enumerate(pts):
            c = p["esr"]
            if c in in_out and c in pos:
                continue
            prev = next((pts[j]["esr"] for j in range(i - 1, -1, -1) if pts[j]["esr"] in pos), None)
            nxt = next((pts[j]["esr"] for j in range(i + 1, len(pts)) if pts[j]["esr"] in pos), None)
            nb = [in_out[q] for q in (prev, nxt) if q is not None]
            ok = all(nb) if nb else s["sheet"] not in ANNEXED
            in_out[c] = in_out.get(c, True) and ok

    def inside(code):
        return in_out.get(code, True)

    def in_annex(code):
        return in_ann.get(code, False)

    def is_served(code):
        q = pos.get(code)
        if q is None or stree is None:
            return False
        return bool(stree.query_ball_point([q[0] * kx, q[1] * 110.57], SERVED_M / 1000))

    sheet_of = {}
    for s in secs:
        for p in s["pts"]:
            sheet_of.setdefault(p["esr"], s["sheet"])
    kind = {}
    for c in codes:
        flag = passenger_op(ops.get(c))
        annexed = sheet_of[c] in ANNEXED
        if flag and (is_served(c) or (annexed and in_annex(c))):
            kind[c] = "stop"
        else:
            kind[c] = "flag, not served" if flag else "no passenger op"
    log(f"point kinds: {dict(Counter(kind.values()))}")

    # --- 4. sections: drop node lists, cut at the outline, give each shared pair one owner
    owner = {}
    order = sorted(secs, key=lambda s: (TYPE_RANK.get(s["type"], 9), s["id"]))
    for s in order:
        for a, b in zip(s["pts"], s["pts"][1:]):
            owner.setdefault(frozenset((a["esr"], b["esr"])), s["id"])
    # A pair whose two points both lie on another section, with points between them there and
    # about the same km, is that section's track listed coarsely: 61-004 Сенная — Трофимовский I
    # is 61-005 and 61-002 end to end, with only their nodes. The finer listing keeps it.
    where = defaultdict(list)                         # code -> [(section id, index, km)]
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
    rows, names, used = [], {}, set()
    km_kept = Counter()
    for s in secs:
        sheet = s["sheet"]
        if s["km"] == 0:
            stat["node-list sections dropped (all 0 km)"] += 1
            continue
        if sheet in ANNEXED and ann is None:
            stat["annexed sections without annex.geojson"] += 1
            continue
        k = 0
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
                stat["pairs outside the outline"] += 1
                km_kept["outside"] += km
                continue
            qa, qb = pos.get(a["esr"]), pos.get(b["esr"])
            if km == 0 and qa and qb and dist_m(*qa, *qb) > 2000:
                stat["0 km pairs over 2 km apart"] += 1
            k += 1
            km_kept["kept"] += km
            pairs.append((a, b, km))
        if not k:
            continue
        # A long stop-to-stop stretch with no served stop on it answers to OSM's routes too:
        # its end stops become junctions on this line only ("esr:<code>@<section>"), so
        # build_model keeps it only where passenger routes run over it. Otherwise a freight
        # branch or bypass between two served stations (Безенчук — Кинель's southern bypass,
        # Сенная — Аткарск) is a section between two stops, which nothing ever questions.
        clone, stretch = set(), set()
        if sheet not in ANNEXED:
            run, run_km, flagged, idx = [], 0, False, []
            for i, (a, b, km) in enumerate(pairs):
                if not run:
                    run = [a["esr"]]
                    run_km, flagged, idx = 0, False, []
                run.append(b["esr"])
                idx.append(i)
                run_km += km
                at_end = kind[b["esr"]] == "stop" or i == len(pairs) - 1 or \
                    pairs[i + 1][0]["esr"] != b["esr"]
                if not at_end:
                    flagged |= kind[b["esr"]] == "flag, not served"
                    continue
                if (kind[run[0]] == "stop" and kind[run[-1]] == "stop"
                        and ((flagged and run_km >= UNSERVED_KM) or run_km >= BARE_KM)):
                    clone |= {run[0], run[-1]}
                    stretch |= set(idx)
                    stat["stop-to-stop stretches left to OSM's routes"] += 1
                    km_kept["stretches left to OSM's routes"] += run_km
                run = []
        for k, (a, b, km) in enumerate(pairs, 1):
            ca, cb = a["esr"], b["esr"]
            if k - 1 in stretch:
                ca = f"{ca}@{s['id']}" if ca in clone else ca
                cb = f"{cb}@{s['id']}" if cb in clone else cb
            rows.append({"sol": f"{s['id']}:{k}", "line": s["id"], "a": f"esr:{ca}",
                         "b": f"esr:{cb}", "len": str(km), "im": ROADS[sheet][0],
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
        # Each clone joined to its stop by a 0 km piece, so the line stays one connected
        # piece (rinf.py would split it into separately named lines) and keeps the stop for
        # its served stretches; and given a 0 km stub, so it has three neighbours and rinf.py
        # ends a section there instead of merging it away. build_model drops both 0 km
        # pieces, which no route runs over.
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": ROADS[sheet][0], "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": ROADS[sheet][0], "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        name = section_name(s)
        wd = wdl.get(frozenset((base_name(s["pts"][0]["name"]), base_name(s["pts"][-1]["name"]))))
        e = {"name": name, "name_en": "", "type": s["type"], "sheet": sheet,
             "tariff_km": s["km"], "header": s["name"], "annexed": sheet in ANNEXED}
        if wd:
            q, _ru, en, wkm = sorted(wd, key=lambda x: abs((x[3] or s["km"]) - s["km"]))[0]
            e.update({"wikidata": q, "wikidata_km": wkm})
            # Only labels that read as English names: many of these items carry another
            # language's transliteration as their English label ("Tinda — Bestoezjevo").
            if en and re.search(r"\b(?:railway|line|railroad)\b", en, re.I):
                e["name_en"] = en
        names[s["id"]] = e
    log(f"sections: {dict(stat)}")
    log(f"tariff km: {dict(km_kept)}")
    raw_of = {}
    for s in secs:
        for p in s["points"]:
            raw_of.setdefault(p["esr"], p["name"])
    pts_out = []
    for c in sorted(used):
        base = c.split("@")[0]
        q = pos.get(base)
        raw = raw_of[base]
        typ = ("70" if re.match(r"(?:ОП|О\.П\.)\s", raw) else "10") \
            if kind[base] == "stop" and base == c else "80"
        r = {"op": f"esr:{c}", "uopid": f"RU{c}", "name": clean(raw), "type": typ}
        if c.endswith("x"):
            # A clone's stub end: no place and no name, so rinf.py cannot trace the stub and
            # leaves it out (logged as a rejected 0 km section), after it has served to end a
            # section at the clone. Placed, a 0 km stub survived into the lines.
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = {"endpoint": f"Тарифное руководство № 4, {b1name}, {b2name}",
             "fetched": date.today().isoformat()}
    (OUT / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                       "utf-8")
    (OUT / "points.json").write_text(json.dumps({**stamp, "rows": pts_out}, ensure_ascii=False),
                                     "utf-8")
    (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    log(f"wrote {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
        f"({sum(1 for r in pts_out if r['type'] != '80')} stops, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced), "
        f"{sum(1 for e in names.values() if e.get('wikidata'))} lines with a Wikidata item, "
        f"{sum(1 for e in names.values() if e['name_en'])} with an English name -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esr", metavar="PBF")
    ap.add_argument("--ua", action="store_true", help="with --esr: the Ukraine extract")
    ap.add_argument("--annex", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--wikidata", action="store_true")
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args()
    if args.esr:
        p = Path(args.esr)
        esr_pass(p if p.is_absolute() else ROOT / p, args.ua)
    if args.annex:
        annex()
    if args.clip:
        clip()
    if args.wikidata:
        wikidata()
    if args.convert:
        convert()
