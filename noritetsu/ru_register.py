"""Russia: Tariff Guide No. 4 (Тарифное руководство № 4) written into rinf.py's input format.

    $env:OSMIUM_POOL_THREADS=2
    python extract.py --region ru/full --pbf data/raw/ru/russia-YYMMDD.osm.pbf
    python extract.py --region ru/ua --pbf data/raw/ru/ukraine-YYMMDD.osm.pbf --bbox 32.0,45.9,40.3,50.2
    python ru_register.py --esr data/raw/ru/russia-YYMMDD.osm.pbf        # esr:user codes, before deleting the .pbf
    python ru_register.py --esr data/raw/ru/ukraine-YYMMDD.osm.pbf --ua  # same for the 2022-annexed areas
    python ru_register.py --annex        # data/raw/ru/annex.geojson: the annexed area under Russian control
    python ru_register.py --annex-trains # data/raw/ru/annex_trains.json: the trains that run there
    python ru_register.py --clip         # after every extract: ru/full + ru/ua -> data/proc/ru, clipped
    python ru_register.py --wikidata     # line and station labels, lengths (cached in data/raw/ru/wdx_*.json)
    python ru_register.py --wikidata-stations   # the station labels alone
    python ru_register.py --convert      # data/raw/rinf/ru/{sections,points,names}.json
    python ru_register.py --colours      # colours/ru.csv from the last build's register lines
    python build_model.py --region ru --register rinf:data/raw/rinf/ru

ENGLISH NAMES (see "English names" below): stations take Wikidata's English label by ESR code
where it reads as a romanisation of their Russian name; lines are named from their two ends'
English names. Nothing is transliterated here.

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

BORDERS (2026-10-03).  BORDER gives each crossing with a built neighbour that has a point in
borders.EXTRA (Belarus, Kazakhstan) a piece from the last Russian point to that point; FOREIGN
reads Kazakhstan's sections on Russian soil around Iletsk; ALIAS makes an export code another
station; --clip ends OSM's track over each crossing at the point. ru_sources.md "Borders".

THE 2022-ANNEXED RAILWAYS (Донецкая, Луганская, Мелитопольская-Херсонская; Anita, 2026-10-01:
with Russia, de facto). Book 1 lists the whole pre-war Donetsk railway, Kramatorsk and
Sloviansk included, which Ukraine holds and Ukrzaliznytsia serves. So the clip polygon there is
the area under Russian control (`--annex`): the four oblasts (OCHA COD-AB admin 1) cut to the
Voronoi cells of the settlements en.wikipedia's war maps mark as Russian-held
(Module:Russo-Ukrainian war overview map and ... detailed map, CC BY-SA; contested places count
as not held). OSM has NO passenger route relations there (checked 2026-10-01: only
Ukrzaliznytsia's routes on the Ukrainian side and Crimea's Armyansk trains), so the OSM test
above would drop every section. What runs comes from poizdato.net's pages for the Russian-run
suburban trains there instead (--annex-trains, annex_evidence; Anita 2026-10-04): the pairs
they run over are running, a section they run over in part is split into its id and "<id>~",
and rinf_countries/ru.py's `suspended` greys every line no train runs over. Stops there are the
points trains call at, and off their track Book 2's passenger points. OSM station names there
are mostly Ukrainian; --clip puts their name:ru in `name` (from --esr --ua) so the register's
Russian names match and show, and annex_place places points OSM knows only in Ukrainian.

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

# Another administration's sections on Russian soil. Kazakhstan's railway (KTZ) runs the
# Orenburg - Aktobe line from Iletsk I to the border at Zhaysan, and Iletsk I - Uyutny towards
# Oral, and Book 1 lists them on its sheet only: without these, 150 km of Russian track was in
# no country's build and the two crossings could not join. Only the pairs with both points
# placed inside Russia are kept (the rest is casia_register's). Lokot - Tretyakovo (68-067, no
# train over the border) is left out.
FOREIGN = {"Кзх": ("68", "Қазақстан темір жолы")}
FOREIGN_SECTIONS = {"68-001", "68-003"}
# An export code that is another listed station: RZD's 80-030 Orenburg - Kanisai ends at
# "Канисай (рзд) (эксп.)", 8 tariff km past Kanisai, where KTZ's sheet starts at Iletsk I (7 km
# from Kanisai as the crow flies). Placed by name it sat at Kanisai, so the line stopped short.
ALIAS = {"813008": ("666906", "Илецк I")}
ALIAS_TO = {code for code, _name in ALIAS.values()}

# Crossings with a built neighbour (borders.EXTRA's points): (border point uopid, the Russian
# point the line reaches it from, the Book 1 point beyond it on the same section or None, how
# long the piece is). The piece runs from the Russian point to the border point, whose id
# ("e" + uopid) the neighbour's build ends at too, on the line of the section the pair is on
# (with no next point: the section that ends at the Russian point). How long:
#   "at"     the next point is the border: an export code ("Красное (эксп.)", the tariff
#            handover), the list's last; the pair's tariff km, and the piece replaces the
#            pair. Those codes are placed by name at their station, so the pair to them was
#            traced as a loop (Рудня, 10.5 km of track for 11 km) or rejected (Красное).
#   "split"  the pair crosses the border: its tariff km shared by crow-fly distance to the two
#            points, as casia_register shares it on the far side (crow-fly x 1.2 when the far
#            point is not placed here). The pair stays as it was (see convert()).
#   a number the tariff km from the border to the Russian point, from the section's own km;
#   None     crow-fly x 1.2 (ua_register's rule).
# Every piece is junction-ended, so build_model keeps it only where OSM's passenger routes run
# over it (drop_unridden_sections). casia_borders.json says which section each Kazakh crossing
# is on; the by-md agent named the Belarusian ones' sections (2026-10-03). --clip keeps the
# OSM track up to each of these points (ways mostly abroad were cut whole before).
BORDER = [
    ("BYRUOSINOVKA", "171346", "171401", "at"),     # Красное (17-066): Minsk - Moscow
    ("BYRUZAOLSHA", "171914", "172008", "at"),      # ОП 462 км, past Рудня (17-067): Smolensk - Vitebsk
    ("BYRUEZERISHCHE", "067119", "067405", "at"),   # Завережье (01-072): St Petersburg - Vitebsk
    ("BYRUALESHA", "066830", "066900", "at"),       # Клястица (01-073): Velikiye Luki - Alesha
    ("BYRUZAKOPYTYE", "202309", "201202", "at"),    # Злынка (17-052): Minsk - Adler
    # Abkhazia (caucasus_register.BORDER_XY, 2026-10-04): 51-032 ends at "Веселое (эксп.)", 2
    # km past Веселое, the Psou bridge. Moscow, St Petersburg and Dioskuria trains to Sukhum.
    ("XARUPSOU", "532701", "532608", "at"),         # Веселое (51-032): Adler - Sukhum
    # 61-017 Krasny Kut - Verkhny Baskunchak crosses Kazakhstan twice (Astrakhan trains)
    ("XKZRU01", "618226", "618230", "split"),       # Шунгули | Молодость
    ("XKZRU02", "618616", "618601", "split"),       # Полынный | Сайхин
    ("XKZRU04", "618724", "618809", "split"),       # Ингеловский | Джаныбек
    ("XKZRU03", "618813", "618809", "split"),       # Коммунистический | Джаныбек
    ("XKZRU05", "617011", "618404", "at"),          # Кигаш (61-074): Astrakhan - Atyrau
    ("XKZRU06", "628406", None, 13),                # Озинки (61-072, its end 13 km on): Saratov - Oral
    ("XKZRU07", "666713", "666709", "split"),       # Уютный | Шынгырлау (KTZ's 68-001): Iletsk - Oral
    ("XKZRU08", "667425", "667504", "split"),       # Кос-Арал | Жайсан (KTZ's 68-003): Orenburg - Aktobe
    ("XKZRU15", "826332", "826224", "split"),       # Горбуново | ОП 2564 км (80-011): Kurgan - Petropavl
    # 83-002 starts at Пут. Пост 2742 км, the road boundary at the border (casia's crossing is
    # 3.4 km from 2739 km, the post 3 km on), 18 tariff km before Исилькуль; its first points
    # are unplaced or misplaced (ОП Охровка at 0 km lies 75 km east), so the piece runs from
    # Исилькуль, the first point placed.
    ("XKZRU16", "832009", "831985", 18),            # Исилькуль | ОП 2758 км (83-002): Petropavl - Omsk
    ("XKZRU17", "835308", "835312", "split"),       # Черлак | Урлютюб (83-018): the Kulunda line
    ("XKZRU18", "835524", "835510", "split"),       # Теренгуль | Кызылтуз (83-018)
    ("XKZRU20", "843304", None, None),              # Локоть (83-014's end): Barnaul - Semey
]

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

def book1(sheets=None):
    """Every tariff section on the Russian-administration sheets (or on `sheets`, a dict keyed
    by sheet name; default ROADS as it stands, which ua_register and the others swap for
    their own), points in order."""
    import xlrd
    sheets = ROADS if sheets is None else sheets
    path = newest("tr4_kniga1_*.xls")
    wb = xlrd.open_workbook(str(path))
    out = []
    for sh in wb.sheets():
        if sh.name not in sheets:
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


def section_name(s, en=None):
    """"Обухово — Чудово-Московское": the section's two end points by TR-4's own spelling,
    with the via note of the header where it has one, since some pairs of nodes have two
    sections ("Царевщина — Красная Глинка (через ст. Безымянка, Жигулёвское Море)").

    With `en` (ESR code -> English name) it returns (name, English name): the English name is
    built the same way from the same points' English names ("Obukhovo — Chudovo-Moskovskoye",
    "... (via Bezymyanka, Zhigulyovskoye More)"), and is "" unless every end and every via
    station has one."""
    a, b = clean(s["points"][0]["name"]), clean(s["points"][-1]["name"])
    pa, pb = s["points"][0], s["points"][-1]
    # The header's two ends, spelled as the point list spells them: lists can run on past
    # the node under extra codes (10-002 ends "Калининград-Пассажирский", then
    # "Калининград-Сортировочный (эксп.)" and "(перев.)" at the same km).
    hdr = VIA.sub("", s["name"])
    hdr = re.sub(r"\((?:[^()]*ж\.\s*д\.?|[^()]*(?:ПАССАЖИР|пассажир)[^()]*)\)", "", hdr)
    ends = [x.strip() for x in re.split(r"\s+-\s+", hdr)]
    by_key = {}
    # On another administration's sheet, the station before its export code where both are
    # listed (68-003 starts "Илецк I (эксп.)", then "Илецк I", at the same km), so the names
    # are casia_register's. Russia's own sheets keep the first listed (about 40 names carry
    # "(эксп.)" or "(перев.)" that way: "Угольная (эксп.) — Владивосток"; colours/ru.csv is
    # keyed by these names).
    for p in sorted(s["points"], key=lambda p: s["sheet"] in FOREIGN
                    and bool(EXTRA_CODE.search(p["name"]))):
        by_key.setdefault(nkey(p["name"]), p)
        by_key.setdefault(nkey(EXTRA_CODE.sub("", p["name"])), p)
    if len(ends) == 2 and all(ends):
        pa, pb = (by_key.get(nkey(e)) or by_key.get(nkey(EXTRA_CODE.sub("", e))) for e in ends)
        a, b = (clean(p["name"]) if p else tidy_caps(e) for p, e in zip((pa, pb), ends))
    name = f"{a} — {b}"
    ea = en.get(pa["esr"], "") if en is not None and pa else ""
    eb = en.get(pb["esr"], "") if en is not None and pb else ""
    name_en = f"{ea} — {eb}" if ea and eb else ""
    m = VIA.search(s["name"])
    if m:
        via = re.split(r"\s*\(", m.group(1))[0].strip()
        via = tidy_caps(via) if via.upper() == via else via
        name += f" (через {via[0].lower() + via[1:] if via.startswith('Ст') else via})"
        # The via stations are on the section's own list: "ст. Козелковская, Смышляевка".
        vias = [re.sub(r"^(?:ст\.?|оп|о\.п\.)\s+", "", v.strip(), flags=re.I)
                for v in re.split(r",|\s+и\s+", via)]
        vp = [by_key.get(nkey(v)) for v in vias]
        ven = [en.get(p["esr"], "") if en is not None and p else "" for p in vp]
        name_en = f"{name_en} (via {', '.join(ven)})" if name_en and all(ven) else ""
    return (name, name_en) if en is not None else name


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


# ================================================================ trains in the annexed area

# The ua build's crawl of poizdato.net (ua_register.py --crawl-trains), which lists the
# Russian-run suburban trains of the annexed area beside Ukrzaliznytsia's (ru_sources.md "The
# annexed railways: what runs"). --annex-trains copies those trains' calls into
# data/raw/ru/annex_trains.json, so this build does not change when the ua build refreshes or
# deletes its pages.
POIZDATO = ROOT / "data" / "raw" / "ua" / "poizdato"
ANNEX_TRAINS = RAW / "annex_trains.json"
ANNEX_MIN_DAYS = 9        # running days in the pages' two-month calendar: more than about weekly
ANNEX_MIN_CALLS = 2       # calls at stations inside annex.geojson that make a train the area's


def ukey(s):
    """A Ukrainian station name as a matching key: case, "Пас." and separators folded."""
    s = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'").replace("ʼ", "'")
    s = re.sub(r"\bпас\.?(?=\s|$)", "пасажирська", s)
    return re.sub(r"[\s\-–—.,]+", " ", s).strip()


UK_RU_VOWELS = set("аеёиоуыэюяіїєйьъ'")


def skel(s):
    """A name's consonants in order, doubled ones once: one key for the Ukrainian and Russian
    spellings of a place ("Старобільськ", "Старобельск" -> "стрблск"; "Нижньокринка",
    "Нижнекрынка" -> "нжнкрнк"). Only ever compared between places a few km apart."""
    s = nkey(s).replace("ґ", "г")
    s = re.sub(r"\b(ii|іі)\b", "2", s)
    out = []
    for ch in s:
        if ch in UK_RU_VOWELS or not ch.isalnum():
            continue
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def annex_trains():
    """data/raw/ru/annex_trains.json: every train page of the ua build's poizdato crawl with at
    least ANNEX_MIN_CALLS calls, and at least half its calls, at stations inside annex.geojson,
    and ANNEX_MIN_DAYS running days; its calls in order (station names as the page spells them,
    in Ukrainian)."""
    import shapely
    from shapely.geometry import shape
    from ua_register import parse_train
    ann = shapely.union_all([shape(f["geometry"]) for f in json.loads(
        (RAW / "annex.geojson").read_text("utf-8"))["features"]])
    shapely.prepare(ann)
    # A name counts as inside when every OSM station of that name in Ukraine's extract is
    # ("1100 км" or "Пост" are everywhere).
    hits = defaultdict(set)
    for _nid, lon, lat, nm, _ru, uk, rw in json.loads(
            (RAW / "osm_names_ua.json").read_text("utf-8")):
        if rw in ("station", "halt"):
            hit = bool(shapely.contains_xy(ann, lon, lat))
            for x in {nm, uk} - {None, ""}:
                hits[ukey(x)].add(hit)
    inside = defaultdict(bool, {k: v == {True} for k, v in hits.items()})
    files = sorted(POIZDATO.glob("rozklad-*/*.html"))
    files = [f for f in files if f.parent.name != "rozklad-po-stantsii"]
    trains, few = [], []
    for f in files:
        t = parse_train(f)
        n_in = sum(1 for c in t["calls"] if inside[ukey(c["name"])])
        # Most of its calls, too: a train elsewhere in Ukraine can call at a halt whose name
        # only the annexed area otherwise has (Podilsk - Pomichna's did).
        if n_in < ANNEX_MIN_CALLS or 2 * n_in < len(t["calls"]):
            continue
        rec = {"page": f"{f.parent.name}/{f.name}", "title": t["title"].split(",")[0],
               "runs": t["runs"], "days": len(t["days"]),
               "first": t["days"][0] if t["days"] else "", "last": t["days"][-1] if t["days"] else "",
               "calls": [c["name"] for c in t["calls"]]}
        (trains if rec["days"] >= ANNEX_MIN_DAYS else few).append(rec)
    ANNEX_TRAINS.write_text(json.dumps({
        "source": "poizdato.net train pages, as crawled by ua_register.py --crawl-trains "
                  "(data/raw/ua/poizdato/)", "built": date.today().isoformat(),
        "trains": trains, "too_few_days": few}, ensure_ascii=False, indent=1), "utf-8")
    log(f"--annex-trains: {len(files)} pages, {len(trains)} trains in the annexed area "
        f"({len(few)} more running under {ANNEX_MIN_DAYS} days) -> {ANNEX_TRAINS.name}")
    for r in trains:
        log(f"    {r['title'][:60]:60s} {r['days']:3d} days, {len(r['calls'])} calls")


COAST_DEG = 0.05          # the outline reaches this far out to sea (about 3-5 km)
SYNTH_NODE = 10 ** 15     # --clip's nodes where a way meets a border point (OSM ids are ~1e10)


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
    left. Station names inside the annex polygon take their name:ru, and OSM's disused
    frontline track there comes back from the ua build's osm_disused.pkl. The two sources are
    left as they are, so this can be rerun after changing the outline; run it after every
    extract."""
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
    # OSM's mappers have retagged the frontline railways railway=disused since 2022, and
    # extract.py keeps no disused track, so the annexed sections there (Lysychansk - Svatove,
    # Popasna, Bakhmut, Avdiivka) had no rails to trace and were not drawn even greyed. The ua
    # build reads those ways from Ukraine's .pbf (ua_register.py --disused,
    # data/raw/ua/osm_disused.pkl, main-line kinds only); the ones mostly inside annex.geojson
    # go back in here as track, tagged as ua's are, so they trace and are greyed with the rest.
    dp = ROOT / "data" / "raw" / "ua" / "osm_disused.pkl"
    ap = RAW / "annex.geojson"
    if dp.exists() and ap.exists():
        ann = shapely.union_all([shape(f["geometry"]) for f in
                                 json.loads(ap.read_text("utf-8"))["features"]])
        shapely.prepare(ann)
        with open(dp, "rb") as f:
            dis = pickle.load(f)
        add_id, add_x, add_y, n_add = [], [], [], 0
        for wid, (tags, nds) in dis.items():
            if wid in ways or len(nds) < 2:
                continue
            xs = np.array([x for _n, x, _y in nds])
            ys = np.array([y for _n, _x, y in nds])
            if 2 * int(shapely.contains_xy(ann, xs, ys).sum()) < len(nds):
                continue
            t = {k: v for k, v in tags.items() if k != "railway"}
            t["railway"] = "rail"
            t["noritetsu:osm_railway"] = "disused"
            ways[wid] = (t, np.array([n for n, _x, _y in nds], dtype=np.int64))
            add_id += [n for n, _x, _y in nds]
            add_x += [int(round(x * 1e7)) for _n, x, _y in nds]
            add_y += [int(round(y * 1e7)) for _n, _x, y in nds]
            n_add += 1
        if n_add:
            allid = np.concatenate([cid, np.array(add_id, dtype=cid.dtype)])
            allx = np.concatenate([cx, np.array(add_x, dtype=cx.dtype)])
            ally = np.concatenate([cy, np.array(add_y, dtype=cy.dtype)])
            cid, first = np.unique(allid, return_index=True)
            cx, cy = allx[first], ally[first]
        log(f"clip: {n_add} disused main-line ways in the annexed area put back as track "
            f"(from {dp.name})")
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
    # Track over a BORDER crossing ends there: a way over it is kept, or cut, to the run of its
    # nodes inside around the segment the point lies on, and that segment's node beyond. A way
    # the rule above cuts whole (most of it abroad) comes back so the register's piece to the
    # point traces; a way it keeps whole loses its far part, so that piece's track is the
    # piece's own (build_model gives a way to a line only if 60% of it lies along it: Черлак's
    # two 15 km ways, 9 km of them in Kazakhstan, belonged to nothing and the piece dropped).
    # Relations and infra are decided without these, so no foreign route comes in through them.
    import borders
    want = {"e" + uop for uop, *_r in BORDER}
    bpts = [p for p in borders.load() if p["id"] in want]
    near = np.zeros(cid.size, dtype=bool)
    for p in bpts:
        near |= (np.abs(cx / 1e7 - p["lon"]) < 0.3) & (np.abs(cy / 1e7 - p["lat"]) < 0.2)
    near_ids = set(cid[near].tolist())
    order = np.argsort(cid)
    scid = cid[order]

    def segment_at(xy):
        """(index of the segment a border point lies on, the point), within borders.NEAR_M.
        Not borders.Index, which looks only near a way's nodes: a 9 km straight segment
        into Kazakhstan has none near the point."""
        for p in bpts:
            kx = math.cos(math.radians(p["lat"])) * 111320
            x, y = (xy[:, 0] - p["lon"]) * kx, (xy[:, 1] - p["lat"]) * 110570
            dx, dy = np.diff(x), np.diff(y)
            L2 = dx * dx + dy * dy
            t = np.clip(-(x[:-1] * dx + y[:-1] * dy) / np.where(L2 > 0, L2, 1), 0, 1)
            d = np.hypot(x[:-1] + t * dx, y[:-1] + t * dy)
            if len(d) and d.min() <= borders.NEAR_M:
                k = int(d.argmin())
                fx = xy[k, 0] + t[k] * (xy[k + 1, 0] - xy[k, 0])
                fy = xy[k, 1] + t[k] * (xy[k + 1, 1] - xy[k, 1])
                return k, p, (fx, fy)
        return None

    # The node beyond the border becomes a new node where the way meets the border point (ids
    # from SYNTH_NODE up, added to coords.npz): a crossing segment can run 9 km into
    # Kazakhstan (Черлак), and build_model gives a way to a line only if 60% of it lies along
    # the line.
    synth = []
    n_cross = 0
    for wid in sorted(ways):
        tags, nodes = ways[wid]
        if not any(int(n) in near_ids for n in nodes):
            continue
        ns = [int(n) for n in nodes if int(n) in known]
        if len(ns) < 2 or not any(n in outside for n in ns):
            continue
        at = order[np.searchsorted(scid, ns)]
        xy = np.c_[cx[at] / 1e7, cy[at] / 1e7]
        hit = segment_at(xy)
        if not hit:
            continue
        k, bp, foot = hit
        ins = [n not in outside for n in ns]
        lo, hi = k, k + 1
        while lo > 0 and ins[lo] and ins[lo - 1]:
            lo -= 1
        while hi < len(ns) - 1 and ins[hi] and ins[hi + 1]:
            hi += 1
        run = ns[lo:hi + 1]
        if ins[k] != ins[k + 1]:
            sid = SYNTH_NODE + len(synth)
            synth.append((sid, foot))
            if ins[k]:
                run[-1] = sid                  # hi == k + 1, the node abroad
            else:
                run[0] = sid                   # lo == k
        keep_w[wid] = (tags, np.array(run, dtype=np.asarray(nodes).dtype))
        n_cross += 1
        log(f"  kept {hi - lo + 1} of {len(ns)} nodes of way {wid} over the border at "
            f"{bp['id']}")
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
    log(f"clip: kept {len(keep_w)}/{len(ways)} ways ({n_cross} of them cut at a border "
        f"crossing), {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations; "
        f"{n_ru} stop names in the annexed area set from name:ru")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = PROC / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, PROC / fn)
    if synth:
        cid = np.concatenate([cid, np.array([s for s, _f in synth], dtype=cid.dtype)])
        cx = np.concatenate([cx, np.array([round(f[0] * 1e7) for _s, f in synth]).astype(cx.dtype)])
        cy = np.concatenate([cy, np.array([round(f[1] * 1e7) for _s, f in synth]).astype(cy.dtype)])
        o = np.argsort(cid, kind="stable")
        cid, cx, cy = cid[o], cx[o], cy[o]
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
# Stations by ESR code (P2815, "ESR station code"; P2814 is a Danish register), one query per
# first digit of the code: the whole set in one query is ~14,000 items with five optional
# labels and the sitelink, which is near the service's limit (each part takes 2-7 s).
Q_STATIONS = """SELECT ?s ?esr ?ru ?en ?engb ?mul ?enwiki WHERE {
  ?s wdt:P2815 ?esr . FILTER(STRSTARTS(?esr, "%s"))
  OPTIONAL { ?s rdfs:label ?ru FILTER(LANG(?ru) = "ru") }
  OPTIONAL { ?s rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?s rdfs:label ?engb FILTER(LANG(?engb) = "en-gb") }
  OPTIONAL { ?s rdfs:label ?mul FILTER(LANG(?mul) = "mul") }
  OPTIONAL { ?enwiki schema:about ?s ; schema:isPartOf <https://en.wikipedia.org/> } }"""
# Russian stations with only an Express code (P722, "UIC station code": Express-3's seven
# digits), joined to ESR through osm.sbin.ru's esr.csv. 157 items, 5 of them a stop here.
Q_STATIONS_UIC = """SELECT ?s ?uic ?ru ?en ?enwiki WHERE {
  ?s wdt:P722 ?uic ; wdt:P17 wd:Q159 . FILTER NOT EXISTS { ?s wdt:P2815 [] }
  OPTIONAL { ?s rdfs:label ?ru FILTER(LANG(?ru) = "ru") }
  OPTIONAL { ?s rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?enwiki schema:about ?s ; schema:isPartOf <https://en.wikipedia.org/> } }"""


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


def wikidata(lines=True):
    if lines:
        rows = sparql(Q_LINES)
        (RAW / "wdx_lines.json").write_text(json.dumps(
            {"fetched": date.today().isoformat(), "rows": rows}, ensure_ascii=False), "utf-8")
        log(f"--wikidata: {len(rows)} lines rows")
        time.sleep(3)
    rows = []
    for d in "0123456789":
        rows += sparql(Q_STATIONS % d)
        time.sleep(2)
    uic = sparql(Q_STATIONS_UIC)
    (RAW / "wdx_stations.json").write_text(json.dumps(
        {"fetched": date.today().isoformat(), "rows": rows, "uic": uic}, ensure_ascii=False),
        "utf-8")
    log(f"--wikidata: {len(rows)} station rows by ESR code, {len(uic)} by Express code only")


# ================================================================ English names
#
# Wikidata's English label of the station item carrying the point's ESR code, kept only
# where it reads as a romanisation of the point's Russian name (`en_score` >= EN_AGREE).
# Nothing is transliterated here: a point without such a label has no English name and the
# app shows the Russian one. Of 3,246 English labels on stops (2026-10-03) the check turns
# away about 350: Tatar and Bashkir names under "en" ("Tügäräk Qır" for Круглое Поле), Dutch
# transliterations ("Tsjebangda", "Noviy Oergal"), Finnish ones ("Leppäsyrjä"), items for
# another or a renamed point ("208 km" on Платформа 210 км, "Nauchny park" on Олимпийская
# Деревня), and names that drop a part ("Paveletsky" for Москва-Пассажирская-Павелецкая).
# Ukrainian-based romanisations in Crimea ("Pryberezhne") mostly pass: that is how
# English sources spell them.

EN_AGREE = 0.88
TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n",
                     "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y",
                     "", "e", "yu", "ya"]))
# What labels add to a name: "Kirov railway station", "Alattu (railway halt)", "Ryazanovka
# railway station, Primorsky Krai", "Railway Station Aleksikovo", "Train station in Pyatigorsk".
EN_TAIL = re.compile(r"\s*(?:,.*$|\b(?:railway|railroad|train|rail)?\s*(?:station|halt|platform|"
                     r"stop|stopping point|passing loop|passing|terminal|track post)\b.*$)", re.I)
EN_HEAD = re.compile(r"^(?:(?:railway|train)\s+station(?:\s+in)?|station|stantsiya)\s+", re.I)
# Not English: Dutch digraphs, German or Polish j ("Rakitnaja", "Kompressornyj"), Tatar q
# (Bashkir and Tatar labels are also caught by their letters outside plain ASCII).
NOT_EN = re.compile(r"tsj|zj|sj|oe(?![\s\-)]|$)|q|j(?=[aeiou])|[iy]j\b", re.I)
RU_HEAD = re.compile(r"^(?:платформа|ост\.?\s*пункт|остановочный пункт|о\.п\.|оп|пост|"
                     r"пут\.?\s*пост|блокпост|рзд|разъезд|станция)\s+", re.I)
RU_NOTE = re.compile(r"\((?:рзд\.?|бп|пп|п|обп|стр|эксп\.?|перев\.?|стык|узк\.?)\)", re.I)
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}
# Words a label may translate ("Chelyabinsk Main") or leave out ("Omsk" for
# Омск-Пассажирский): compared without them, on both sides.
GENERIC_RU = r"(?:пассажирск|главн|сортировочн|товарн|грузов|северн|южн|западн|восточн)\w*"
GENERIC_EN = (r"passenger|main|sorting|freight|cargo|goods|north(?:ern)?|south(?:ern)?|"
              r"west(?:ern)?|east(?:ern)?|"
              r"(?:passaz|sortirovoch|tovarn|gruzov|glavn|severn|yuzhn|iujn|zapadn|vostochn)\w*")
EN_SAME = {"moscow": "moskva", "new": "novyi", "saint petersburg": "sankt peterburg",
           "st petersburg": "sankt peterburg"}


def clean_en(label):
    """A Wikidata label as a station name: 'Kirov railway station' -> 'Kirov'."""
    s = re.sub(r"\s*\([^)]*\)", "", (label or "").replace("_", " ").strip())
    s = EN_HEAD.sub("", s)
    t = EN_TAIL.sub("", s).strip(" ,-")
    return t or s


def _en_words(s, generic):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[\-–—.,'’()]", " ", s)
    out = []
    for w in s.split():
        if re.fullmatch(generic, w):
            continue
        w = "".join(TRANSLIT.get(c, c) for c in ROMAN.get(w, w))
        # Adjective and plural endings, which romanisations write many ways (Kiyevsky,
        # Kievskaya; Polyany): left out on both sides.
        w = re.sub(r"(?<=[sknv])(?:aya|aja|oye|oe|iye|ie|ye|yy|yi|iy|ij|yj|ii|y|i)$", "", w)
        out.append(w)
    return out


def _en_squash(s):
    s = re.sub(r"[^a-z0-9]", "", s)
    for a, b in (("shch", "sh"), ("sch", "sh"), ("j", "y"), ("yo", "e"), ("ye", "e"),
                 ("yu", "u"), ("ya", "a"), ("iy", "i"), ("yi", "i"), ("y", "i"), ("kh", "h"),
                 ("x", "ks"), ("w", "v"), ("ii", "i")):
        s = s.replace(a, b)
    return s


def en_score(ru_name, en):
    """How well an English name reads as a romanisation of a Russian one, 0..1: 0 when it is
    not plain English or the numbers in the two differ. A check, never a source of names."""
    from difflib import SequenceMatcher
    if not en or any(ord(c) > 127 for c in en) or NOT_EN.search(en):
        return 0.0
    a = re.sub(r"\s*№\s*", " ", RU_NOTE.sub("", RU_HEAD.sub("", (ru_name or "").strip())))
    a = " ".join(_en_words(a, GENERIC_RU))
    b = en.lower()
    for x, y in EN_SAME.items():
        b = re.sub(rf"\b{x}\b", y, b)
    b = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", b)
    b = re.sub(r"\bkilomet(?:er|re)s?\b", "km", b)
    b = " ".join(_en_words(b, GENERIC_EN))
    if sorted(re.findall(r"\d+", a)) != sorted(re.findall(r"\d+", b)):
        return 0.0
    a, b = _en_squash(a), _en_squash(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def wd_station_en():
    """ESR code -> the English label of its Wikidata station item, cleaned (not checked):
    the `en` label, else en-gb, else the en.wikipedia title, else a Latin `mul` label. A code
    on two items with different labels gets none."""
    p = RAW / "wdx_stations.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text("utf-8"))
    rows = list(data["rows"])
    exp = {}
    if data.get("uic") and (RAW / "esr.csv").exists():
        import csv
        with open(RAW / "esr.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r.get("express"):
                    exp.setdefault(r["express"], r["esr"])
        for r in data["uic"]:
            e = exp.get(r["uic"])
            if e:
                rows.append(dict(r, esr=e))
    got = defaultdict(set)
    for r in rows:
        lab = (r.get("en") or r.get("engb")
               or (urllib.parse.unquote(r["enwiki"].rsplit("/", 1)[-1]) if r.get("enwiki") else "")
               or (r["mul"] if r.get("mul") and re.search("[A-Za-z]", r["mul"]) else ""))
        if lab:
            got[r["esr"]].add(clean_en(lab))
    return {k: next(iter(v)) for k, v in got.items() if len(v) == 1}


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


# Annexed points whose OSM station goes by another name altogether: Book 1 name -> an OSM name
# (name, name:ru or name:uk) of the station it is. Renamed towns (Торез is Чистякове since
# 2016) and translated names; Ясиноватая, the junction station, is OSM's "Ясинувата Захід"
# (0.9 km east of Ясинувата Гірка, where Book 1 has it 1 km from Ясиноватая Горка), which the
# trains' "Ясинувата-Західна" is.
ANNEX_PLACE_ALIAS = {
    "Торез": "Чистякове",
    "Ясиноватая": "Ясинувата Захід",
    "Луганск-Северный": "Луганськ-Північний",
    "Амросиевка": "Амвросіївка",
}


def annex_place(secs, pos, how, ua_names):
    """Place the annexed points the Russian name did not. OSM's stations there mostly have
    only a Ukrainian name (Старобільськ, Моспине, Ханженкове), so a point goes to an OSM
    station whose name has the same consonants (`skel`) or is its ANNEX_PLACE_ALIAS, within
    the tariff km + NAME_REACH_KM of the two nearest placed points on its section, as the
    Russian-name pass requires. Changes `pos` and `how` in place."""
    nu = RAW / "osm_names_ua.json"
    if not nu.exists():
        return
    # First unplace the name placements far from both neighbours: the unique-name pass looks
    # all over ANNEX_BBOX, and put "Донец" (84-014, by Вергунка) at Donets station near
    # Balakliia and the numbered halts "Ост. пункт 12 км", "143 км", "805 км" near Kremenchuk,
    # 200-500 km away, which cut 84-014, 84-001, 84-008 and 84-012 in two.
    bad = set()
    for s in secs:
        if s["sheet"] not in ANNEXED:
            continue
        pts = s["pts"]
        for i, p in enumerate(pts):
            if how.get(p["esr"]) != "name":
                continue
            far = []
            for rng in (range(i - 1, -1, -1), range(i + 1, len(pts))):
                q = next((pts[j] for j in rng if pts[j]["esr"] in pos
                          and pts[j]["esr"] not in bad), None)
                if q:
                    km = abs((q["km0"] or 0) - (p["km0"] or 0))
                    far.append(dist_m(*pos[p["esr"]], *pos[q["esr"]]) / 1000 > km + NAME_REACH_KM)
            if far and all(far):
                bad.add(p["esr"])
    for c in bad:
        pos.pop(c, None)
        how.pop(c, None)
    by_skel = defaultdict(list)
    for nid, lon, lat, nm, nm_ru, nm_uk, rw in json.loads(nu.read_text("utf-8")):
        if rw in ("station", "halt"):
            for k in {skel(x) for x in (nm, nm_ru, nm_uk) if x}:
                by_skel[k].append((lon, lat, nid))
    n = Counter()
    for _round in range(3):
        for s in secs:
            if s["sheet"] not in ANNEXED:
                continue
            pts = s["pts"]
            for i, p in enumerate(pts):
                if p["esr"] in pos:
                    continue
                alias = ANNEX_PLACE_ALIAS.get(p["name"])
                k = skel(p["name"])
                cand = (ua_names.get(nkey(alias), []) if alias
                        else by_skel.get(k, []) if len(k) >= 3 else [])
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
                    if alias and all(dist_m(*cand[0][:2], *q[:2]) <= 2000 for q in cand):
                        pos[p["esr"]], how[p["esr"]] = cand[0][:2], "name"
                        n["alias"] += 1
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
                    n["alias" if alias else "consonants"] += 1
    log(f"annexed points: {len(bad)} name placements far from both neighbours unplaced; "
        f"placed by a Ukrainian name {dict(n)}")


def annex_evidence(secs, pos, inside, owner, coarse):
    """Which annexed pairs trains run over, and which points they call at, from
    data/raw/ru/annex_trains.json (--annex-trains). Each call is matched to a Book 1 point of
    the annexed sheets: by name (the same consonants, `skel`; "1118 Км" to a point named for
    its kilometre), or by place (a placed point within SERVED_M of an OSM station of the call's
    Ukrainian name). Of several candidates the one nearest along the track to the train's
    previous call is taken. The pairs on the shortest path over the annexed sections between
    two matched calls are served (no longer than ANNEX_DETOUR x the crow-fly distance + 5 km,
    nor than ANNEX_MAX_HOP;
    a call with no such path is stepped over: Щебенка, which Book 1 lists only as the start of
    89-014, though the Yenakiieve trains call there). A call left unmatched is then looked for
    among the points on the path it lies on, by how alike the consonants are.
    Returns (served pair keys, called point codes, log lines)."""
    import difflib
    import heapq
    if not ANNEX_TRAINS.exists():
        return set(), set(), ["no annex_trains.json: nothing in the annexed area runs"]
    trains = json.loads(ANNEX_TRAINS.read_text("utf-8"))["trains"]
    nu = json.loads((RAW / "osm_names_ua.json").read_text("utf-8"))
    osm_at = defaultdict(list)
    for _nid, lon, lat, nm, _ru, uk, rw in nu:
        if rw in ("station", "halt"):
            for x in {nm, uk} - {None, ""}:
                osm_at[ukey(x)].append((lon, lat))
    adj = defaultdict(list)
    apts = {}
    for s in secs:
        if s["sheet"] not in ANNEXED:
            continue
        for p in s["pts"]:
            apts.setdefault(p["esr"], p["name"])
        for a, b in zip(s["pts"], s["pts"][1:]):
            key = frozenset((a["esr"], b["esr"]))
            if a["esr"] == b["esr"] or owner.get(key) != s["id"] or (s["id"], key) in coarse:
                continue
            if not (inside(a["esr"]) and inside(b["esr"])):
                continue
            km = abs((b["km0"] or 0) - (a["km0"] or 0))
            adj[a["esr"]].append((b["esr"], km))
            adj[b["esr"]].append((a["esr"], km))
    # A branch start Book 1 lists on no pair of the line it leaves (ANNEX_ON_PAIR) is joined to
    # that pair's two ends, and a path over those joins runs over the pair.
    via = {}
    for c, (x, y, kx) in ANNEX_ON_PAIR.items():
        km = next((w for v, w in adj[x] if v == y), None)
        if km is None or c not in adj:
            continue
        for end, w in ((x, kx), (y, km - kx)):
            adj[c].append((end, w))
            adj[end].append((c, w))
            via[frozenset((c, end))] = frozenset((x, y))
    by_skel, by_km = defaultdict(set), defaultdict(set)
    for c, nm in apts.items():
        if c not in adj:
            continue
        by_skel[skel(nm)].add(c)
        m = re.search(r"(?<!\d)(\d+)\s*км", nm.lower())
        if m:
            by_km[m.group(1)].add(c)

    def cands(call):
        if call in ANNEX_CALL_ALIAS:
            return set(by_skel.get(skel(ANNEX_CALL_ALIAS[call]), ()))
        k = ukey(call)
        m = re.fullmatch(r"(\d+) км", k)
        if m:
            return set(by_km.get(m.group(1), ()))
        out = set(by_skel.get(skel(call), ()))
        for lon, lat in osm_at.get(k, ()):
            out |= {c for c in adj if c in pos and dist_m(*pos[c], lon, lat) <= SERVED_M}
        return out

    def path(a, b):
        """(km, [points]) of the shortest path a -> b, or None past the detour limit."""
        lim = ANNEX_MAX_HOP
        if a in pos and b in pos:
            lim = min(lim, dist_m(*pos[a], *pos[b]) / 1000 * ANNEX_DETOUR + 5)
        d, prev, h = {a: 0.0}, {}, [(0.0, a)]
        while h:
            dd, u = heapq.heappop(h)
            if u == b:
                break
            if dd > d.get(u, 1e18) or dd > lim:
                continue
            for v, w in adj[u]:
                if dd + w < d.get(v, 1e18):
                    d[v], prev[v] = dd + w, u
                    heapq.heappush(h, (dd + w, v))
        if b not in d or d[b] > lim:
            return None
        seq = [b]
        while seq[-1] != a:
            seq.append(prev[seq[-1]])
        return d[b], seq[::-1]

    served, called, notes = set(), set(), []
    skipped, unmatched = Counter(), Counter()
    for t in trains:
        calls = t["calls"]
        cs = [cands(c) for c in calls]
        chosen = []                       # (call index, point)
        for i, cc in enumerate(cs):
            if not cc:
                continue
            if chosen:
                ranked = sorted((p[0], c) for c in cc for p in [path(chosen[-1][1], c)] if p)
                if not ranked:
                    skipped[calls[i]] += 1
                    continue
                chosen.append((i, ranked[0][1]))
            else:
                nxt = next((x for x in cs[i + 1:] if x and x != cc), set())
                best = min(cc, key=lambda c: min(
                    [p[0] for x in nxt for p in [path(c, x)] if p] or [1e9]))
                chosen.append((i, best))
        for (i, a), (j, b) in zip(chosen, chosen[1:]):
            called |= {a, b}
            p = path(a, b) if a != b else (0.0, [a])
            for x, y in zip(p[1], p[1][1:]):
                key = frozenset((x, y))
                served.add(via.get(key, key))
            # The calls between i and j that matched nothing: a point on this path?
            for k in range(i + 1, j):
                if cs[k]:
                    continue
                sk = skel(calls[k])
                score = sorted((difflib.SequenceMatcher(None, sk, skel(apts[c])).ratio(), c)
                               for c in p[1][1:-1])
                if score and score[-1][0] >= ANNEX_FUZZY:
                    called.add(score[-1][1])
                else:
                    unmatched[calls[k]] += 1
    notes.append(f"{len(trains)} trains, {len(served)} pairs served, {len(called)} points "
                 f"called at")
    if skipped:
        notes.append(f"calls with no path from the previous one, stepped over: {dict(skipped)}")
    if unmatched:
        notes.append(f"calls matched to no point: {dict(unmatched)}")
    return served, called, notes


# Track between an annexed railway and Russia's own that no section lists: (the annexed
# section, its end point, Russia's point). Book 1 ends 89-018 at "Квашино (стык)", the
# inter-railway junction, placed by name at Квашине station, and 51-007 at "Успенская (эксп.)",
# placed at Успенская: the ~10 km between, over the pre-2014 border, are on neither, and the
# Donetsk - Uspenskaya trains run over them. Added on the annexed section when a train calls at
# the two one after the other; length crow-fly x 1.15.
ANNEX_LINKS = [("89-018", "901302", "510907")]


def annex_link_runs(name_a, name_b):
    if not ANNEX_TRAINS.exists():
        return False
    ka, kb = skel(name_a), skel(name_b)
    for t in json.loads(ANNEX_TRAINS.read_text("utf-8"))["trains"]:
        ks = [skel(c) for c in t["calls"]]
        if any({x, y} == {ka, kb} for x, y in zip(ks, ks[1:])):
            return True
    return False


# Calls whose name is not their Book 1 point's: the trains' "Ясинувата-Західна" is OSM's
# "Ясинувата Захід", which is Book 1's Ясиноватая (ANNEX_PLACE_ALIAS).
ANNEX_CALL_ALIAS = {"Ясинувата-Західна": "Ясиноватая"}
# Щебенка, where 89-014 to Нижнекрынка leaves 89-013, is on no pair of 89-013, whose list goes
# Виноградники - Ост. Пункт 1092 км (3 km) past it; the Yenakiieve trains call there and run
# up the branch and back. Point -> (the pair it lies on, km from the pair's first point).
ANNEX_ON_PAIR = {"928216": ("928235", "928220", 1)}
ANNEX_DETOUR = 2.0        # a path between two calls no longer than this x crow-fly + 5 km ...
ANNEX_MAX_HOP = 40        # ... and than this many tariff km: these trains call every few km, and
#                           "78 Км" after Луганськ had matched 89-020's 78 km halt near Ilovaisk
ANNEX_FUZZY = 0.7         # an unmatched call is the point on its path this alike, or none


def convert():
    from scipy.spatial import cKDTree
    import shapely
    roads = {**ROADS, **FOREIGN}
    secs, b1name = book1(roads)
    secs = [s for s in secs if s["sheet"] in ROADS or s["id"] in FOREIGN_SECTIONS]
    ops, b2name = book2()
    log(f"Book 1 ({b1name}): {len(secs)} sections, {len(FOREIGN_SECTIONS)} of them on "
        f"another administration's sheet; Book 2 ({b2name}): {len(ops)} points")
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
    # English names by ESR code, checked against the point's own name (see "English names").
    wde = wd_station_en()
    en_of, en_turned = {}, 0
    for s in secs:
        for p in s["points"]:
            c = p["esr"]
            if c in wde and c not in en_of:
                if en_score(clean(p["name"]), wde[c]) >= EN_AGREE:
                    en_of[c] = wde[c]
                else:
                    en_of[c] = ""
                    en_turned += 1
    en_of = {k: v for k, v in en_of.items() if v}
    all_codes = {p["esr"] for s in secs for p in s["points"]}
    log(f"English names: {sum(1 for c in all_codes if c in wde)} of {len(all_codes)} points "
        f"have an English label on Wikidata; {en_turned} turned away as not this name in "
        f"English, {len(en_of)} kept")

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
            if p["esr"] in ALIAS:
                p["esr"], p["name"] = ALIAS[p["esr"]]
                stat["export codes that are another station"] += 1
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
    annex_place(secs, pos, how, ua_names)
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
            # Another administration's sections are mostly abroad, their points unplaced
            # (Russia's extract only), and one placed point inside would make the rest of
            # them inside: there only placed points count.
            ok = ok and s["sheet"] not in FOREIGN
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
    # --- the annexed railways: which pairs trains run over (annex_evidence). A point there
    # is a stop when a train calls at it, or, off the track trains run over, when Book 2 gives
    # it a passenger operation (the greyed lines keep their stations).
    a_served, a_called, notes = annex_evidence(secs, pos, inside, owner, coarse)
    for x in notes:
        log(f"  annexed: {x}")
    on_run = {c for k in a_served for c in k}
    for c in codes:
        if sheet_of[c] not in ANNEXED:
            continue
        flag = passenger_op(ops.get(c))
        if c in a_called or (flag and in_annex(c) and c not in on_run):
            kind[c] = "stop"
        else:
            kind[c] = "flag, not served" if flag else "no passenger op"
    log(f"point kinds, the annexed railways' set by their trains: {dict(Counter(kind.values()))}")
    rows, names, used = [], {}, set()
    km_kept = Counter()
    # The border crossings (BORDER): the pair each one is on, and the border points.
    import borders
    bpts = {p["id"][1:]: p for p in borders.load()}
    cross = {frozenset((r, f)): (uop, r, f, how) for uop, r, f, how in BORDER if f}
    border_pts, side_of = {}, {}

    def side_id(s, ru_c, far_c):
        """ru_c's id on the line's side of the border: its clone where the pair beside it, on
        the side away from the border, is a stretch left to OSM's routes (Озинки, Исилькуль),
        so the piece continues the line there rather than hanging off the stop by a 0 km link
        build_model drops."""
        clone, stretch_keys = side_of.get(s["id"], (set(), set()))
        if ru_c not in clone:
            return ru_c
        pts = [p["esr"] for p in s["pts"]]
        for i, c in enumerate(pts):
            if c != ru_c:
                continue
            for j in (i - 1, i + 1):
                if (0 <= j < len(pts) and pts[j] != far_c
                        and frozenset((ru_c, pts[j])) in stretch_keys):
                    return f"{ru_c}@{s['id']}"
        return ru_c

    def border_row(s, uop, ru_c, how, km, far=None):
        """The piece from Russian point ru_c to border point uop, on section s's line."""
        b = bpts.get(uop)
        q = pos.get(ru_c)
        if b is None:
            log(f"  border {uop}: no such point in borders.load()")
            return
        d_r = dist_m(*q, b["lon"], b["lat"]) / 1000 if q else None
        if how == "at":
            L = km
        elif how == "split":
            qf = pos.get(far)
            if q and qf:
                d_f = dist_m(*qf, b["lon"], b["lat"]) / 1000
                L = km * d_r / (d_r + d_f) if d_r + d_f > 0 else d_r
            else:
                L = (d_r or km) * 1.2
        elif how is None:
            L = d_r * 1.2
        else:
            L = float(how)
        if d_r is not None:
            L = max(L, d_r * 1.05, 0.3)
        c = side_id(s, ru_c, far)
        # The point's op is "border/<uopid>" with no uopid of its own: rinf.py names the
        # station "e" + the op's last path part, the shared id, and orders a section's
        # unconnected pieces ("17-067#1", "#2") by their smallest uopid or op, which a bare
        # "BYRU..." would win against "RU...", swapping 17-067's two pieces' ids.
        rows.append({"sol": f"{s['id']}:{uop}", "line": s["id"], "a": f"esr:{c}",
                     "b": f"border/{uop}", "len": f"{L:.1f}", "im": roads[s["sheet"]][0],
                     "label": f"{s_name.get(ru_c, '')} - {b['name']}"})
        used.add(c)
        border_pts[uop] = {"op": f"border/{uop}", "name": b["name"], "type": "90",
                           "lon": b["lon"], "lat": b["lat"]}
        stat["border pieces"] += 1
        km_kept["border pieces"] += L
        log(f"  border {uop}: {s['id']} {s_name.get(ru_c)} -> border, {L:.1f} km ({how}"
            f"{'' if d_r is None else f', {d_r:.1f} km as the crow flies'})")

    s_name = {p["esr"]: p["name"] for s in secs for p in s["pts"]}
    for s in secs:
        sheet = s["sheet"]
        if s["km"] == 0:
            stat["node-list sections dropped (all 0 km)"] += 1
            continue
        if sheet in ANNEXED and ann is None:
            stat["annexed sections without annex.geojson"] += 1
            continue
        k = 0
        pairs, xings = [], []
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
            if key in cross:
                xings.append((cross[key], abs(km)))
                # An export code at the border is the end of the list, and the piece to the
                # border point takes its place. A crossing pair otherwise stays as it was:
                # dropped where its far point is placed abroad, kept where it is unplaced
                # and so counted inside (80-011's, 83-018's points in Kazakhstan, which no
                # trace reaches), so the section's pieces, and their line ids, stay as they are.
                if cross[key][3] == "at":
                    continue
            if not (inside(a["esr"]) and inside(b["esr"])):
                stat["pairs outside the outline"] += 1
                km_kept["outside"] += km
                if sheet in ANNEXED and os.environ.get("RU_ANNEX_DEBUG"):
                    log(f"    annexed pair outside: {s['id']} {a['name']} {pos.get(a['esr'])} "
                        f"- {b['name']} {pos.get(b['esr'])}")
                continue
            qa, qb = pos.get(a["esr"]), pos.get(b["esr"])
            if km == 0 and qa and qb and dist_m(*qa, *qb) > 2000:
                stat["0 km pairs over 2 km apart"] += 1
            k += 1
            km_kept["kept"] += km
            pairs.append((a, b, km))
        if not k and not xings:
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
                # Not where an ALIAS handover station ends the stretch: 80-030 Orenburg - Iletsk I
                # ended at a junction (the export code) before, so it already answered to
                # OSM's routes (Moscow - Tashkent, - Dushanbe run over it), and clones would put
                # both its ends on junction ids beside the stations, cut off from 68-003 at
                # Iletsk I and from Orenburg.
                if (kind[run[0]] == "stop" and kind[run[-1]] == "stop"
                        and not {run[0], run[-1]} & ALIAS_TO
                        and ((flagged and run_km >= UNSERVED_KM) or run_km >= BARE_KM)):
                    clone |= {run[0], run[-1]}
                    stretch |= set(idx)
                    stat["stop-to-stop stretches left to OSM's routes"] += 1
                    km_kept["stretches left to OSM's routes"] += run_km
                run = []
        # On the annexed railways a section trains run over only in part is two lines: the part
        # they run over under the section's id, the rest under "<id>~", greyed as not running
        # (rinf_countries/ru.py's `suspended`). A section with no train is one greyed line.
        run_k = {i for i, (a, b, _km) in enumerate(pairs)
                 if frozenset((a["esr"], b["esr"])) in a_served} if sheet in ANNEXED else set()
        for k, (a, b, km) in enumerate(pairs, 1):
            ca, cb = a["esr"], b["esr"]
            if k - 1 in stretch:
                ca = f"{ca}@{s['id']}" if ca in clone else ca
                cb = f"{cb}@{s['id']}" if cb in clone else cb
            lid = s["id"]
            if sheet in ANNEXED and run_k and k - 1 not in run_k:
                lid = f"{s['id']}~"
            rows.append({"sol": f"{s['id']}:{k}", "line": lid, "a": f"esr:{ca}",
                         "b": f"esr:{cb}", "len": str(km), "im": roads[sheet][0],
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
        for sid, a, b in ANNEX_LINKS:
            if (sid == s["id"] and run_k and a in pos and b in pos
                    and annex_link_runs(s_name.get(a, ""), s_name.get(b, ""))):
                L = dist_m(*pos[a], *pos[b]) / 1000 * 1.15
                rows.append({"sol": f"{s['id']}:link", "line": s["id"], "a": f"esr:{a}",
                             "b": f"esr:{b}", "len": f"{L:.1f}", "im": roads[sheet][0],
                             "label": f"{s_name.get(a, a)} - {s_name.get(b, b)}"})
                used |= {a, b}
                stat["annexed links to Russia's own sections"] += 1
                km_kept["annexed links"] += L
        # Each clone joined to its stop by a 0 km piece, so the line stays one connected
        # piece (rinf.py would split it into separately named lines) and keeps the stop for
        # its served stretches; and given a 0 km stub, so it has three neighbours and rinf.py
        # ends a section there instead of merging it away. build_model drops both 0 km
        # pieces, which no route runs over.
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": roads[sheet][0], "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": roads[sheet][0], "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        # The border pieces over this section's crossings (BORDER), from the Russian end
        # (its clone where a stretch left to OSM's routes ends there: side_id).
        side_of[s["id"]] = (clone, {frozenset((a["esr"], b["esr"]))
                                    for i, (a, b, _km) in enumerate(pairs) if i in stretch})
        for (uop, ru_c, far_c, how), km in xings:
            border_row(s, uop, ru_c, how, km, far_c)
        name, name_en = section_name(s, en_of)
        wd = wdl.get(frozenset((base_name(s["pts"][0]["name"]), base_name(s["pts"][-1]["name"]))))
        e = {"name": name, "name_en": name_en, "type": s["type"], "sheet": sheet,
             "tariff_km": s["km"], "header": s["name"], "annexed": sheet in ANNEXED}
        if name_en:
            stat["English name from its end stations"] += 1
        if wd:
            q, _ru, en, wkm = sorted(wd, key=lambda x: abs((x[3] or s["km"]) - s["km"]))[0]
            e.update({"wikidata": q, "wikidata_km": wkm})
            # Every one of these labels only restates the two ends, in an editor's spelling
            # ("Railway line Kovrov - Nizniy Novgorod", "Beloostrov - Vuborg line"), so the
            # name built from the stations' own English names comes first; the label fills in
            # where an end has none. Only labels that read as English names: many of these
            # items carry another language's transliteration ("Tinda — Bestoezjevo").
            if not name_en and en and re.search(r"\b(?:railway|line|railroad)\b", en, re.I):
                e["name_en_wd"] = en
        names[s["id"]] = e
        if sheet in ANNEXED:
            e["annex_running"] = bool(run_k)
            km_run = sum(km for i, (_a, _b, km) in enumerate(pairs) if i in run_k)
            km_all = sum(km for _a, _b, km in pairs)
            stat["annexed sections trains run over, whole" if km_run == km_all and run_k
                 else "annexed sections trains run over, in part" if run_k
                 else "annexed sections no train runs over"] += 1
            km_kept["annexed, trains run"] += km_run
            km_kept["annexed, no train"] += km_all - km_run
            if run_k and len(run_k) < len(pairs):
                names[f"{s['id']}~"] = dict(e, annex_running=False, part_of=s["id"])
    # Crossings with no next point in BORDER: on the line that ends at the Russian point, main
    # sections first, else one it lies on.
    for uop, ru_c, far_c, how in BORDER:
        if far_c:
            continue
        rank = (lambda s: (s["pts"][0]["esr"] != ru_c and s["pts"][-1]["esr"] != ru_c,
                           TYPE_RANK.get(s["type"], 9), s["id"]))
        own = sorted((s for s in secs if s["id"] in names
                      and any(p["esr"] == ru_c for p in s["pts"])), key=rank)
        if not own or ru_c not in used and f"{ru_c}@{own[0]['id']}" not in used:
            log(f"  border {uop}: no line ends at {ru_c}")
            continue
        border_row(own[0], uop, ru_c, how, None)
    for uop, *_r in BORDER:
        if uop not in border_pts:
            stat["border crossings with no piece"] += 1
    # A Wikidata line label is used only where no other line has it: two Волочаевка —
    # Дежневка sections (one via Тунгусский) carry the same item's label.
    label_n = Counter(e.get("name_en_wd") or e["name_en"] for e in names.values()
                      if (e.get("name_en_wd") or e["name_en"]) and not e.get("part_of"))
    for e in names.values():
        lab = e.pop("name_en_wd", "")
        if lab and label_n[lab] == 1 and not e.get("part_of"):
            e["name_en"] = lab
            stat["English name from a Wikidata line label"] += 1
    for e in names.values():
        if e.get("part_of"):
            e["name_en"] = names[e["part_of"]]["name_en"]
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
        if en_of.get(base) and not c.endswith("x"):
            r["name_en"] = en_of[base]
        if c.endswith("x"):
            # A clone's stub end: no place and no name, so rinf.py cannot trace the stub and
            # leaves it out (logged as a rejected 0 km section), after it has served to end a
            # section at the clone. Placed, a 0 km stub survived into the lines.
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    pts_out += [border_pts[k] for k in sorted(border_pts)]
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = {"endpoint": f"Тарифное руководство № 4, {b1name}, {b2name}",
             "fetched": date.today().isoformat()}
    (OUT / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                       "utf-8")
    (OUT / "points.json").write_text(json.dumps({**stamp, "rows": pts_out}, ensure_ascii=False),
                                     "utf-8")
    (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    log(f"wrote {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
        f"({sum(1 for r in pts_out if r['type'] in ('10', '70'))} stops, "
        f"{sum(1 for r in pts_out if r['type'] in ('10', '70') and r.get('name_en'))} of them "
        f"with an English name, {len(border_pts)} border points, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced), "
        f"{sum(1 for e in names.values() if e.get('wikidata'))} lines with a Wikidata item, "
        f"{sum(1 for e in names.values() if e['name_en'])} with an English name -> {OUT}")


# ================================================================ colours
#
# Tariff sections have no colours of their own, and no widely used map colours Russia's
# railways either (checked 2026-10-03: Wikimedia's "Russia Rail Map" and "RZD branches area
# 2018", used on en/ru.wikipedia, draw every railway alike). So each register line takes
# its regional railway's colour, picked here: nine hues, no two neighbouring railways alike,
# a hue used again only far away. Marked `picked` in colours/ru.csv; Anita's to confirm.
ROAD_COLOUR = {
    "01": ("D7263D", "red"),          # Октябрьская
    "17": ("2F7FD8", "blue"),         # Московская
    "28": ("1E9E8F", "teal"),         # Северная
    "24": ("E08E0B", "orange"),       # Горьковская
    "58": ("8A5CC2", "purple"),       # Юго-Восточная
    "51": ("3FA34D", "green"),        # Северо-Кавказская
    "61": ("CC3D8C", "magenta"),      # Приволжская
    "63": ("A0753F", "brown"),        # Куйбышевская
    "76": ("2F7FD8", "blue"),         # Свердловская
    "80": ("3FA34D", "green"),        # Южно-Уральская
    "83": ("E08E0B", "orange"),       # Западно-Сибирская
    "88": ("8A5CC2", "purple"),       # Красноярская
    "92": ("1E9E8F", "teal"),         # Восточно-Сибирская
    "94": ("D7263D", "red"),          # Забайкальская
    "96": ("2F7FD8", "blue"),         # Дальневосточная
    "91": ("E08E0B", "orange"),       # Железные дороги Якутии
    "10": ("CC3D8C", "magenta"),      # Калининградская
    "85": ("A0753F", "brown"),        # Крымская
    "97": ("2F7FD8", "blue"),         # ИФР-1 (no line built)
    "89": ("8A5CC2", "purple"),       # Донецкая (built from 2026-10-04, ANNEX_RUNNING)
    "84": ("D7263D", "red"),          # Луганская
    "82": ("1E9E8F", "teal"),         # Мелитопольская-Херсонская
    "68": ("1E9E8F", "teal"),         # Қазақстан темір жолы's lines around Iletsk (FOREIGN)
}


def colours():
    """colours/ru.csv: one row per register line of the last build (dist/data/ru/lines.json),
    in its railway's colour. Rerun after a rebuild that adds or renames lines, then rebuild
    (the colour is applied by build_model through line_colours.py, drawn by build_tiles)."""
    import csv
    road_of = {name: code for code, name in {**ROADS, **FOREIGN}.values()}
    lines = json.loads((ROOT / "dist" / "data" / "ru" / "lines.json").read_text("utf-8"))["lines"]
    rows, miss = [], Counter()
    for l in sorted((l for l in lines if l.get("src", "osm") != "osm"),
                    key=lambda l: (road_of.get(l["operator"], "99"), l["name"])):
        code = road_of.get(l["operator"])
        if code not in ROAD_COLOUR:
            miss[l["operator"]] += 1
            continue
        col, hue = ROAD_COLOUR[code]
        rows.append({"line": l["name"], "operator": l["operator"], "colour": "#" + col,
                     "source": "picked", "url": "",
                     "note": f"{l['operator']}: one {hue} per regional railway"})
    # Lines the last build did not have yet (the annexed railways, built from 2026-10-04): from
    # names.json, so one build gives them their colour.
    have = {(r["line"], r["operator"]) for r in rows}
    nf = OUT / "names.json"
    for e in (json.loads(nf.read_text("utf-8")).values() if nf.exists() else ()):
        code, op = ROADS.get(e.get("sheet"), (None, None))
        if code in ROAD_COLOUR and (e["name"], op) not in have and e.get("annexed"):
            have.add((e["name"], op))
            col, hue = ROAD_COLOUR[code]
            rows.append({"line": e["name"], "operator": op, "colour": "#" + col,
                         "source": "picked", "url": "",
                         "note": f"{op}: one {hue} per regional railway"})
    path = ROOT / "colours" / "ru.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["line", "operator", "colour", "source", "url", "note"])
        w.writeheader()
        w.writerows(rows)
    log(f"--colours: {len(rows)} register lines -> {path}"
        + (f"; no railway colour for {dict(miss)}" if miss else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esr", metavar="PBF")
    ap.add_argument("--ua", action="store_true", help="with --esr: the Ukraine extract")
    ap.add_argument("--annex", action="store_true")
    ap.add_argument("--annex-trains", action="store_true",
                    help="data/raw/ru/annex_trains.json from the ua build's poizdato pages")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--wikidata", action="store_true")
    ap.add_argument("--wikidata-stations", action="store_true",
                    help="refresh the station labels only (wdx_stations.json)")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--colours", action="store_true",
                    help="colours/ru.csv: every built register line in its railway's colour")
    args = ap.parse_args()
    if args.esr:
        p = Path(args.esr)
        esr_pass(p if p.is_absolute() else ROOT / p, args.ua)
    if args.annex:
        annex()
    if args.annex_trains:
        annex_trains()
    if args.clip:
        clip()
    if args.wikidata or args.wikidata_stations:
        wikidata(lines=args.wikidata)
    if args.convert:
        convert()
    if args.colours:
        colours()
