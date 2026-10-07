"""Iran: RAI's railways as OpenStreetMap maps them (its route=railway relations and named
track), through kr_register's recipe as th_register does; passenger stops and served track
from RAI's timetable. Sources, numbers and what is off: ir_sources.md.

    python extract.py --region ir --pbf data/raw/iran-latest.osm.pbf
    python ir_register.py --construction data/raw/iran-latest.osm.pbf   # needs the .pbf
    python ir_register.py --clip          # after every extract (and --construction)
    python ir_register.py --crawl         # iranrail.net's train and station pages (~45 min)
    python ir_register.py --timetable     # -> data/raw/gtfs/ir/ir_iranrail.gtfs.zip
    python ir_register.py --report        # the way-to-line assignment, km per line
    python build_model.py --region ir --register ir_register:data/raw/ir

THE LINE UNIT. RAI publishes no line list with lengths that can be fetched from here (rai.ir
and raja.ir answer 403 outside Iran). What OSM Iran has is one `route=railway` relation per
railway, as Wikidata and en.wikipedia's "Rail transport in Iran" name them: "راه آهن تهران –
مشهد" (Garmsar - Mashhad), "خط راه‌آهن تهران-تبریز", "خط ریلی بادرود – شيراز", "راه آهن بافق-
زاهدان", "Transiranian", and 52 more; they cover 88% of main-line track (91% with way names;
the named track alone 49%, `probe_kr_ways.py`). Each register line is one of those railways
(LINES): its relation's ways, and named ways outside any relation by their name (NAME_LINE).
Where a relation holds two railways (the Transiranian relation also has Tehran - Pishva,
which the track names call the Tehran - Mashhad railway), a way's own name decides; where one
railway is in pieces of several relations (the Qom - Kerman line is Qom - Meybod, the
Meybod - Bafq part of "Bafq - Isfahan" and the Bafq - Kerman part of "Bafq - Zahedan"), the
pieces are split by place (`split`). Unnamed track outside every relation takes the line its
neighbours at both ends share (`propagate`), or a box's line (BOX_LINE: Torbat-e Heydarieh -
Khaf, which OSM maps in no relation and leaves unnamed). Freight lines (Chadormalu, the Tehran
and Ahvaz bypasses, Rostamkola - Amirabad port, Khorramshahr - Shalamcheh, the lines to the
Turkmen, Afghan and Pakistani borders past the last passenger station) are left out: NAME_LINE
and REL_LINE None. Tehran Metro Line 5, mapped as railway=rail, is left to its OSM routes.

WHICH STATIONS. OSM Iran has a railway=station node at nearly every crossing loop RAI has
(the 2017 RAI station list on fa.wikipedia has ~500, with "اضطراری ۲۵" emergency loops among
them). A station is a passenger stop when a train in RAI's timetable calls at it (iranrail.net
reproduces RAI's realtime system, pws0.rai.ir: 312 trains with every call, and a page per
station with its point; `--crawl`), matched to OSM's rail stations by point and English name
(`timetable_stops`), or when an OSM passenger route stops at it. Each passenger stop goes on
every line whose track passes within LIST_M of it; a line whose track ends on another line's
takes the passenger stop nearest that end within JUNCTION_M (th_register's rule).

SERVED TRACK. `--timetable` writes the same timetable as a GTFS feed, data/raw/gtfs/ir/, which
gtfs_served reads as it reads every national feed: a section no train runs over is greyed, a
junction-ended one no train runs over is dropped.

The `path` argument is data/raw/ir; the OSM half is read from data/proc/ir (extract.py).
"""
import hashlib
import html
import json
import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ir"
PROC = ROOT / "data" / "proc" / "ir"
GTFS_DIR = ROOT / "data" / "raw" / "gtfs" / "ir"
REGION = "ir"
RAI = "راه‌آهن جمهوری اسلامی ایران"
RAI_EN = "Islamic Republic of Iran Railways"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"

# key -> (name, English name). Names are OSM's relation names where it has one (Wikidata's
# labels agree), in Persian; English after the relation's name:en or Wikidata.
LINES = {
    "tehran_mashhad": ("راه آهن تهران – مشهد", "Tehran – Mashhad railway"),
    "garmsar_inchehborun": ("راه‌آهن گرمسار – اینچه‌برون", "Garmsar – Incheh Borun railway"),
    "transiranian": ("راه‌آهن سراسری ایران (تهران – بندر امام خمینی)",
                     "Trans-Iranian Railway (Tehran – Bandar Imam Khomeini)"),
    "tehran_qom": ("راه آهن تهران – قم", "Tehran – Qom railway"),
    "ahvaz_khorramshahr": ("راه آهن اهواز – خرمشهر", "Ahvaz – Khorramshahr railway"),
    "tehran_tabriz": ("خط راه‌آهن تهران – تبریز", "Tehran – Tabriz railway"),
    "mianeh_tabriz": ("مسیر جدید میانه – تبریز", "New Mianeh – Tabriz railway"),
    "tabriz_jolfa": ("راه آهن تبریز – جلفا", "Tabriz – Jolfa railway"),
    "tabriz_university": ("راه آهن دانشگاه – تبریز",
                          "Tabriz – Shahid Madani University railway"),
    "sufian_razi": ("راه آهن صوفیان – رازی", "Sufian – Razi railway"),
    "maragheh_urmia": ("راه‌آهن مراغه – ارومیه", "Maragheh – Urmia railway"),
    "mianeh_ardabil": ("راه آهن میانه – اردبیل", "Mianeh – Ardabil railway"),
    "qazvin_rasht": ("راه آهن قزوین – رشت", "Qazvin – Rasht railway"),
    "tehran_hamedan": ("راه آهن تهران – همدان", "Tehran – Hamedan railway"),
    "hamedan_sanandaj": ("راه آهن همدان – سنندج", "Hamedan – Sanandaj railway"),
    "arak_kermanshah": ("راه آهن اراک – کرمانشاه", "Arak – Kermanshah railway"),
    "qom_kerman": ("راه آهن قم – کرمان", "Qom – Kerman railway"),
    "isfahan_ardakan": ("راه آهن اصفهان – اردکان", "Isfahan – Ardakan railway"),
    "kerman_zahedan": ("راه آهن کرمان – زاهدان", "Kerman – Zahedan railway"),
    "badrud_shiraz": ("خط ریلی بادرود – شیراز", "Badrud – Shiraz railway"),
    "bafq_bandarabbas": ("راه آهن بافق – بندرعباس", "Bafq – Bandar Abbas railway"),
    "mashhad_bafq": ("راه آهن مشهد – بافق", "Mashhad – Bafq railway"),
    "torbat_khaf": ("راه آهن تربت حیدریه – خواف", "Torbat-e Heydarieh – Khaf railway"),
    "yazd_eqlid": ("خط ریلی یزد – اقلید", "Yazd – Eqlid railway"),
    "zahedan_khash": ("راه آهن چابهار – زاهدان", "Chabahar – Zahedan railway"),
    "fariman_sarakhs": ("راه آهن فریمان – سرخس", "Fariman – Sarakhs railway"),
}

# Way names -> line key; None leaves the way out of the register. Exact, after `tidy`.
NAME_LINE = {
    # Tehran - Garmsar - Mashhad (Tehran - Pishva sits in the Transiranian relation, Pishva -
    # Garmsar in an unnamed one; the track names say Tehran - Mashhad throughout)
    "راه آهن تهران - مشهد": "tehran_mashhad", "راه آهن سراسری مشهد تهران": "tehran_mashhad",
    "راه آهن سراسری تهران-مشهد": "tehran_mashhad", "تهران - مشهد": "tehran_mashhad",
    "تهران-مشهد": "tehran_mashhad", "راه اهن تهران-مشهد": "tehran_mashhad",
    "راه آهن تهران-مشهد": "tehran_mashhad", "راه آهن مشهد-تهران": "tehran_mashhad",
    "راه آهن تهران مشهد": "tehran_mashhad",
    "راه‌آهن گرگان - اینچه برون": "garmsar_inchehborun",
    "راه آهن گرگان - اینچه برون": "garmsar_inchehborun",
    "قطار گرگان اینچه برون": "garmsar_inchehborun",
    "North–South Transnational Corridor": "garmsar_inchehborun",
    # the Trans-Iranian south of Tehran, as RAI's districts name its pieces
    "درود - محمدیه": "transiranian", "دورود - محمدیه": "transiranian",
    "اندیمشک - درود": "transiranian", "اندیمشک - دورود": "transiranian",
    "اهواز - اندیمشک": "transiranian", "اهواز - بندر امام خمینی": "transiranian",
    "راه آهن ازنا-دورود-خوزستان": "transiranian",
    "قطار سریع‌السیر تهران - قم": "tehran_qom", "تهران - قم": "tehran_qom",
    "خط قطار تهران - قم": "tehran_qom", "خظ تهران - قم": "tehran_qom",
    "اهواز - خرمشهر": "ahvaz_khorramshahr",
    "راه آهن تهران - تبریز": "tehran_tabriz", "راه آهن تهران -تبریز": "tehran_tabriz",
    "ره آهن تهران - تبریز": "tehran_tabriz", "خط قطار تهران - تبریز": "tehran_tabriz",
    "راه آهن سراسری ایران(زنجان-میانه)": "tehran_tabriz",
    "راه آهن میانه - مراغه": "tehran_tabriz", "میانه - مراغه": "tehran_tabriz",
    "راه آهن مراغه - تبریز": "tehran_tabriz", "راه آهن تبریز-مراغه": "tehran_tabriz",
    "راه آهن تبریز - مراغه": "tehran_tabriz", "راه آهن تبریزمراغه": "tehran_tabriz",
    "مراغه - تبریز": "tehran_tabriz", "راه آهن آذرشهر- تبریز": "tehran_tabriz",
    "راه آهن آذرشهر - تبریز": "tehran_tabriz", "کهندژ - میانه": "tehran_tabriz",
    "مسیر راه آهن تهران - آذربایجان": "tehran_tabriz",
    "مسیر جدید میانه - تبریز": "mianeh_tabriz", "مسیر جدید راه آهن میانه - تبریز": "mianeh_tabriz",
    "مسیر میانه -بستان آباد- تبریز": "mianeh_tabriz", "مسیر میانه -بستان آباد- تبریز": "mianeh_tabriz",
    "تبریز-جلفا": "tabriz_jolfa", "خط آهن جلفا-تبریز": "tabriz_jolfa",
    "تبریز - رازی": "sufian_razi", "Van-Sufiyan demiryolu": "sufian_razi",
    "راه‌آهن مراغه - ارومیه": "maragheh_urmia", "مراغه - ارومیه": "maragheh_urmia",
    "راه آهن مراغه - ارومیه": "maragheh_urmia",
    "قطار اردبیل ـ میانه": "mianeh_ardabil", "راه آهن اردبیل ـ میانه": "mianeh_ardabil",
    "راه آهن اردبیل - میانه": "mianeh_ardabil",
    "راه آهن قزوین-رشت": "qazvin_rasht", "راه آهن قزوین - رشت": "qazvin_rasht",
    "راه آهن رشت - بندر انزلی": "qazvin_rasht",
    "راه آهن تهران - همدان": "tehran_hamedan", "راه آهن تهران - همدان - سنندج": "tehran_hamedan",
    "راه آهن همدان - سنندج": "hamedan_sanandaj",
    "راه آهن ملایر - کرمانشاه": "arak_kermanshah", "راه آهن اراک-ملایر": "arak_kermanshah",
    "راه آهن اراک- ملایر": "arak_kermanshah", "راه آهن ملایر- کرمانشاه": "arak_kermanshah",
    "راه آهن اراک - ملایر": "arak_kermanshah", "راه آهن ارا- کرمانشاه": "arak_kermanshah",
    "راه آهن ارا- ملایر": "arak_kermanshah", "راه آهن اراک -ملایر": "arak_kermanshah",
    "راه آهن بافق - زاهدان": "kerman_zahedan",       # split: Bafq - Kerman in `split`
    "راه آهن شیراز - اصفهان": "badrud_shiraz", "اصفهان - شیراز": "badrud_shiraz",
    "اصفهان شیراز ب": "badrud_shiraz", "اصفهان شیراز": "badrud_shiraz",
    "اصفهان - شیراز - ج": "badrud_shiraz", "خط راه آهن اصفهان_ شیراز": "badrud_shiraz",
    "راه آهن بافق -سیرجان": "bafq_bandarabbas",
    "راه آهن مشهد - بافق": "mashhad_bafq",
    "خط ریلی یزد – اقلید": "yazd_eqlid", "راه آهن اقلید - یزد": "yazd_eqlid",
    "راه آهن چابهار-زاهدان": "zahedan_khash",
    "مشهد-سرخس": "fariman_sarakhs", "Mashad-Sarakhs": "fariman_sarakhs",
    # left out: freight lines and bypasses, the border lines no passenger train crosses,
    # Tehran Metro Line 5's track, lines abroad
    "کمربندی آپرین - بهرام": None, "کمربندی غرب تهران": None, "کمربندی جدید اهواز": None,
    # Qom's shunting yard (Garmanuri) to Qomrud on the Tehran - Qom line: a link between the
    # two Tehran - Qom lines north of Qom; the Tehran - Qom commuter route's OSM line owns it
    "گارمانوری-قمرود": None, "رستمکلا-بندرامیرآباد": None, "شبکه ریلی داخلی بندر امیرآباد": None,
    "خرمشهر - شلمچه": None, "خرمشهر-شَلَمچه": None, "Khaf – Herat railway": None,
    "راه آهن شوشتر - هفت تپه": None, "گمرک": None, "تراموا کرج": None,
    "فرعی بندرگاه": None, "فرعی فولاد اهواز": None, "فرعی پتروشیمی اراک": None,
    "فرعی ایستگاه بالارود": None,
    # the spur to Azarbaijan Shahid Madani University, RAI's Tabriz commuter line
    "راه آهن دانشگاه - تبریز": "tabriz_university",
    "خط مترو هشتگرد": None, "تراموا هشتگرد": None, "تراموا تهران - کرج": None,
    "Tejen/Mary-Sarahs": None, "Tejen/Mary-Serakhs": None, "Алят-Джульфинский ход": None,
    "Horadiz-Ağbənd dəmir yolu": None, "Azərbaycan - Iran railway link": None,
    "راه آهن رشت آستارا": None, "راه آهن رشت - آستارا": None,
}
# Names that say only "railway" or "the Iranian state railway": read as no name, so the
# relation decides ("راه آهن سراسری ایران" is on the Tehran - Tabriz line's Qazvin - Zanjan).
GENERIC = {"راه آهن", "راه آهن سراسری ایران", "راه آهن مشهد"}
# A structure's name on the track: no name.
JUNK = re.compile(r"^(?:پل|تونل)\b|\(پل ")

# route=railway relations -> line key; None: never register track.
REL_LINE = {
    7369741: "tehran_mashhad", 7369740: "tehran_mashhad", 14535584: "tehran_mashhad",
    7286485: "garmsar_inchehborun", 6646450: "garmsar_inchehborun",
    8276383: "transiranian",                  # its Tehran - Pishva ways are named Tehran - Mashhad
    13974632: "tehran_qom",
    8276445: "ahvaz_khorramshahr",
    13974266: "tehran_tabriz", 8276471: "tehran_tabriz", 5355301: "tehran_tabriz",
    13974267: "tehran_tabriz",
    15974330: "mianeh_tabriz",
    # "Mianeh bypass": the curve from the new Mianeh - Tabriz line round to the Ardabil line,
    # by which Tehran - Ardabil trains leave Mianeh; the Ardabil line's own start
    19955526: "mianeh_ardabil",
    8276509: "tabriz_jolfa", 8276510: "sufian_razi", 8276472: "maragheh_urmia",
    15974390: "mianeh_ardabil",
    8276508: "qazvin_rasht",
    19953803: "tehran_hamedan", 12900959: "hamedan_sanandaj", 8276380: "arak_kermanshah",
    8276442: "qom_kerman",
    6124741: "isfahan_ardakan",               # split: Meybod - Bafq is qom_kerman
    6124573: "kerman_zahedan",                # split: Bafq - Kerman is qom_kerman
    8276443: "badrud_shiraz", 8276444: "badrud_shiraz",
    5493750: "bafq_bandarabbas",
    6617080: "mashhad_bafq", 8278724: "mashhad_bafq",
    16007072: "yazd_eqlid",
    15974298: "zahedan_khash",
    5353948: "fariman_sarakhs",
    # never register track
    8278723: None,          # Ardakan - Chadormalu, the iron-ore line
    13977016: None,         # Tehran railway bypass II (Aprin - Bahram)
    19985265: None,         # Qom bypass west
    8276381: None,          # Rostamkola - Bandar Amirabad, the port line
    16676356: None,         # Khaf - Herat: no passenger train past Khaf
    6118211: None,          # Zahedan - Mirjaveh - Pakistan, broad gauge (see ir_sources.md)
    13974268: None,         # Tehran Metro Line 5 (Tehran - Golshahr - Hashtgerd)
    8276518: None, 5735299: None, 5735301: None, 1311880: None,   # abroad
}
# An unnamed way in two relations takes the first of their lines in this order.
REL_PRIORITY = list(LINES)
# Sistan junction - Isfahan is in both the Badrud - Shiraz and the "Bafq - Isfahan" relations:
# it is Badrud - Shiraz's (Tehran - Isfahan - Shiraz's trains run it), which keeps that line in
# one piece; the Isfahan - Ardakan line leaves it at Sistan.
REL_PRIORITY.remove("badrud_shiraz")
REL_PRIORITY.insert(REL_PRIORITY.index("isfahan_ardakan"), "badrud_shiraz")

# Unnamed track in no relation inside these boxes (lon0, lat0, lon1, lat1) is this line.
# Torbat-e Heydarieh - Khaf (2010, Sangan's iron mines; RAI's Tehran - Khaf train runs over
# it): OSM has it in no relation and unnamed.
BOX_LINE = [((59.10, 34.45, 60.25, 35.30), "torbat_khaf")]
# Unnamed track in no relation inside these boxes is no line's: the western half of the
# Garmanuri - Qomrud link (NAME_LINE).
BOX_NONE = [(50.86, 34.712, 50.966, 34.732)]
# Single ways, by OSM id, that are a line's though nothing on them says so.
WAY_LINE = {
    # the curve from Mohammadieh, on the Tehran - Qom line, onto the Qom - Kerman line towards
    # Kashan: RAI's Tehran - Kashan - Isfahan/Yazd/Kerman trains call at Mohammadieh and run
    # over it (no relation, no name)
    336064425: "qom_kerman",
    # the link from the Mianeh - Ardabil line onto the new Mianeh - Tabriz line at Sabz,
    # unnamed and in no relation; without it OSM's Ardabil line touches no other track
    1103881355: "mianeh_ardabil",
}


def split(key, lon, lat):
    """Railways in pieces of other relations: the Meybod - Yazd - Bafq trunk (in OSM's "Bafq -
    Isfahan" relation, east of the Meybod junction at 53.93 E) and Bafq - Zarand - Kerman (in
    "Bafq - Zahedan", west of Kerman station at 57.07 E) are the Qom - Kerman line."""
    if key == "isfahan_ardakan" and lon > 53.935:
        return "qom_kerman"
    if key == "kerman_zahedan" and lon < 57.05:
        return "qom_kerman"
    # Tehran - Eslamshahr - Nasirshahr is one double-track corridor that OSM files one track
    # under each relation; it is the Trans-Iranian's, and the Tehran - Qom line (by Imam
    # Khomeini Airport) leaves it south of Nasirshahr
    if key == "tehran_qom" and lat > 35.505:
        return "transiranian"
    return key


TRACK_KIND = {"rail": "rail", "narrow_gauge": "rail"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

LIST_M = 400          # a passenger stop this close to a line's track is listed on it
DENSE_KM = 0.1        # ... measured to a vertex this often along the track
JUNCTION_M = 1500     # a line ending on another takes the passenger stop nearest that end
STOP_MATCH_M = 3000   # a timetable stop's point may be this far from OSM's station node
STOP_NEAR_M = 1200    # ... and with no name to confirm it, this far
NAME_FAR_M = 60000    # an OSM station whose English name is the stop's may be this far off
MERGE_TT_M = 300      # a matched stop node that is no station record: the record this close
SAME_EN_M = 300       # rail station records this close with one English name are one station

# Timetable names whose OSM station carries no English name (or another one): OSM's name.
TT_ALIAS = {
    "Qom - Mohammadyeh": "ایستگاه راه‌آهن محمدیه",
    "Tabriz - Khavaran": "ایستگاه راه آهن خاوران",
    "Atash Bagh": "ایستگاه راه آهن آتش بیگ",
    "Sanandaj": "سنندج",
    "Parand": "ایستگاه قطار بین شهری پرند",
    "Ardabil": "اردبیل",
}
# Rail stations OSM has with only an English name: their Persian name (clip writes it).
NAME_FIX = {"Sanandaj": "سنندج"}

_S = {}


def line_id(name):
    h = hashlib.blake2b(f"ir|{name}".encode("utf-8"), digest_size=5)
    return "i" + h.hexdigest()


def tidy(name):
    n = " ".join((name or "").split())
    if not n or n in GENERIC or JUNK.search(n):
        return ""
    return n


# =========================================================================== the extract clip

# Route relations that are no passenger service.
NOT_SERVICE = {
    5928466: "Qom monorail, abandoned unfinished",
    3517270: "Ahvaz Urban Railway line 1, not open",
    7369759: "حوزه سمنان, an RAI district mapped as a route",
    7369760: "حوزه مرکزی تهران, an RAI district mapped as a route",
}


def boundary():
    """Iran's outline: OSM relation 304938 from polygons.openstreetmap.fr."""
    import shapely
    from shapely.geometry import shape
    g = json.loads((RAW / "ir_boundary.geojson").read_text(encoding="utf-8"))
    geom = shape(g["geometries"][0]) if g.get("type") == "GeometryCollection" else shape(g)
    shapely.prepare(geom)
    return geom


def construction_pass(pbf, log=print):
    """data/proc/ir/construction.json: the ids of the railway=construction ways in the .pbf
    (extract.py keeps one, retagged, where a route runs over it)."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    out = []
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if w.tags.get("railway") == "construction":
            out.append(w.id)
    (PROC / "construction.json").write_text(json.dumps(sorted(out)), encoding="utf-8")
    log(f"construction: {len(out)} railway=construction ways")


MASTER_BASE = 9_000_000_000   # synthetic route_master ids: this + the lower route id
# Route relations OSM names wrongly: the Tehran - Firuzkuh commuter train's route_master is
# "فیروزکوه ↔ پرند" (Firuzkuh - Parand; its name:en and both its routes say Tehran - Firuzkuh).
REL_NAME_FIX = {19980438: "تهران ↔ فیروزکوه"}


def pair_directions(rels, log):
    """One line for the two directions of an RAI commuter service mapped as two route
    relations with no route_master ("تبریز ← جلفا" and "جلفا ← تبریز"): a route=train with no
    master whose name, its two ends swapped, is another's gets a route_master over both."""
    in_master = {r for t, ms in rels.values() if t.get("type") == "route_master"
                 for ty, r, _ in ms if ty == "r"}

    def key(name):
        m = re.match(r"^\s*(.+?)\s*[←→\-–]\s*(.+?)\s*$", name or "")
        return (m.group(1), m.group(2)) if m else None
    by = {}
    for k, (t, ms) in rels.items():
        if t.get("type") == "route" and t.get("route") == "train" and k not in in_master:
            kk = key(t.get("name"))
            if kk:
                by.setdefault(kk, k)
    made = 0
    for (a, b), k in sorted(by.items(), key=lambda kv: kv[1]):
        o = by.get((b, a))
        if o is None or o < k:
            continue
        t = rels[k][0]
        en = t.get("name:en") or ""
        m = re.match(r"^\s*(.+?)\s*[←→\-–]\s*(.+?)\s*$", en)
        rels[MASTER_BASE + k] = ({"type": "route_master", "route_master": "train",
                                  "name": f"{b} – {a}" if "←" in t.get("name", "") else f"{a} – {b}",
                                  "name:en": (f"{m.group(1)} – {m.group(2)}" if m else en),
                                  "operator": t.get("operator", ""), "network": t.get("network", ""),
                                  "service": t.get("service", "")},
                                 [("r", k, ""), ("r", o, "")])
        made += 1
    log(f"  {made} two-direction services given a route_master")


def clip(log=print):
    """Rewrite data/proc/ir: out what lies abroad (Geofabrik's cut takes in a little of
    Türkiye, Azerbaijan, Turkmenistan, Afghanistan and Pakistan), the NOT_SERVICE route
    relations, urban routes mostly on construction track, and construction ways no remaining
    route runs over. A way goes if at least half its nodes are outside Iran."""
    import shapely
    with open(PROC / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(PROC / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(PROC / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(PROC / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    ir = boundary()
    out = ~shapely.contains_xy(ir, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut[tags.get("name") or "(unnamed)"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items() if k not in outside
              and shapely.contains_xy(ir, v[1], v[2])}
    n_fix = 0
    for k, (t, lon, lat) in keep_s.items():
        if not t.get("name") and t.get("name:en") in NAME_FIX and t.get("railway") in (
                "station", "halt", "stop"):
            t["name"] = NAME_FIX[t["name:en"]]
            n_fix += 1
    log(f"  {n_fix} nameless rail station records given their Persian name (NAME_FIX)")
    dropped_routes = {k for k in rels if k in NOT_SERVICE}
    for k in sorted(dropped_routes):
        log(f"  not a service: {k} {NOT_SERVICE[k]}")
    cons_p = PROC / "construction.json"
    cons = set(json.loads(cons_p.read_text(encoding="utf-8"))) if cons_p.exists() else set()
    rels = {k: v for k, v in rels.items() if k not in dropped_routes}
    if cons:
        # Every railway=construction way goes, whatever route runs over it: OSM Iran puts the
        # unopened extensions of open metro lines into their routes (Isfahan's line 1 to
        # Shahin Shahr, Karaj's line 2, Mashhad's line 3 beyond its three open stations), and
        # extract.py had kept them as track. A route left with no track drops out by itself.
        gone_c = [w for w in keep_w if w in cons]
        under = Counter()
        for k, (t, ms) in rels.items():
            if t.get("type") == "route":
                for ty, r, _ in ms:
                    if ty == "w" and r in cons and r in keep_w:
                        under[f"{k} {t.get('name')}"] += 1
        for w in gone_c:
            cut["(construction) " + (keep_w[w][0].get("name") or "")] += 1
            del keep_w[w]
        log(f"  {len(gone_c)} construction ways dropped; they were under these routes: "
            + "; ".join(f"{k} ({v})" for k, v in under.most_common()))
    else:
        log("  no construction.json: run --construction on the .pbf first")
    for k, nm in REL_NAME_FIX.items():
        if k in rels:
            rels[k][0]["name"] = nm
    pair_directions(rels, log)
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1])[:40]:
        log(f"  cut {v:4d}  {name}")
    log(f"IR clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = PROC / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, PROC / fn)


# =========================================================================== way -> line

def way_geo(ways, coords):
    """{way: (km, centroid lon, centroid lat)}"""
    out = {}
    for wid, (tags, nodes) in ways.items():
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        pos = pos[ok]
        if pos.size < 2:
            out[wid] = (0.0, 0.0, 0.0)
            continue
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        lat = np.radians((y[:-1] + y[1:]) / 2)
        km = float(np.hypot(np.diff(x) * np.cos(lat) * 111.32, np.diff(y) * 110.57).sum())
        out[wid] = (km, float(x.mean()), float(y.mean()))
    return out


def assign(ways, coords, infra, log):
    """{way: line key} for every way that is register track (the module docstring)."""
    geo = way_geo(ways, coords)
    inrel = defaultdict(list)
    for rid, (tags, members) in infra.items():
        for m in members:
            if m[0] == "w" and m[1] in ways:
                inrel[m[1]].append(rid)
    out, unknown, left = {}, Counter(), Counter()
    cand = []
    for wid, (t, _n) in ways.items():
        if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
            continue
        km, lon, lat = geo[wid]
        n = tidy(t.get("name"))
        svc = t.get("service")
        rel_none = any(r in REL_LINE and REL_LINE[r] is None for r in inrel[wid])
        if wid in WAY_LINE:
            out[wid] = WAY_LINE[wid]
            continue
        if n and n in NAME_LINE:
            key = NAME_LINE[n]
            if key is None:
                left[n] += km
                continue
        elif n:
            unknown[n] += km
            continue
        else:
            if svc not in (None, "crossover"):
                continue
            if rel_none or any(x0 <= lon <= x1 and y0 <= lat <= y1
                               for x0, y0, x1, y1 in BOX_NONE):
                continue
            keys = sorted((REL_LINE[r] for r in inrel[wid] if REL_LINE.get(r)),
                          key=REL_PRIORITY.index)
            key = keys[0] if keys else None
            if key is None:
                box = [k for (x0, y0, x1, y1), k in BOX_LINE
                       if x0 <= lon <= x1 and y0 <= lat <= y1 and not inrel[wid]]
                if box:
                    key = box[0]
                else:
                    cand.append(wid)
                    continue
        out[wid] = split(key, lon, lat)
    got = propagate(ways, out, cand, log)
    out.update(got)
    out.update(fill_runs(ways, geo, out, [w for w in cand if w not in got], log))
    bridge_gaps(ways, coords, out, log)
    straight_gaps(ways, coords, out, log)
    km_by = Counter()
    for w, k in out.items():
        if not ways[w][0].get("service"):
            km_by[k] += geo.get(w, (0.0,))[0]
    log("IR: main-line track km per line (both tracks of double track): "
        + ", ".join(f"{k} {v:,.0f}" for k, v in km_by.most_common()))
    log("IR: named track left out: " + ", ".join(f"{n} {v:.0f}" for n, v in left.most_common()))
    if unknown:
        log("IR: names in no table, left out: "
            + ", ".join(f"{n} {v:.1f}" for n, v in unknown.most_common()))
    return out, geo


def propagate(ways, named, cand, log):
    """Unnamed track outside every relation takes the line its neighbours at both ends share,
    repeated until nothing changes."""
    node_ways = defaultdict(list)
    for w in set(named) | set(cand):
        nl = ways[w][1]
        node_ways[int(nl[0])].append(w)
        node_ways[int(nl[-1])].append(w)
    for w in set(named) | set(cand):
        for n in np.asarray(ways[w][1]).tolist()[1:-1]:
            if n in node_ways:
                node_ways[n].append(w)
    name = dict(named)
    got = {}
    for _round in range(200):
        new = {}
        for w in cand:
            if w in name:
                continue
            nl = ways[w][1]
            a = {name[o] for o in node_ways[int(nl[0])] if o != w and o in name}
            b = {name[o] for o in node_ways[int(nl[-1])] if o != w and o in name}
            if len(a & b) == 1:
                new[w] = next(iter(a & b))
        if not new:
            break
        name.update(new)
        got.update(new)
    log(f"IR: {len(got)} unnamed ways outside every relation named from their neighbours; "
        f"{len(cand) - len(got)} left unnamed")
    return got


FILL_MAX_KM = 60.0    # an unnamed run of track at most this long takes the one line round it


def fill_runs(ways, geo, named, cand, log):
    """Unnamed runs of track inside one line (tr_register's fill_runs): OSM Iran leaves the
    tracks through a station unnamed and outside the line's relation (Rasht, Gorgan, Zahedan),
    split at every switch, where `propagate` fills a single way only. Each connected run of
    such ways under FILL_MAX_KM whose ends touch exactly one line's track takes that line; a
    run touching two lines (a junction station) is left alone."""
    cset = set(cand)
    parent = {w: w for w in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    at = defaultdict(list)
    for w in set(named) | cset:
        for n in np.asarray(ways[w][1]).tolist():
            at[n].append(w)
    first = {}
    for w in cand:
        for n in np.asarray(ways[w][1]).tolist():
            if n in first:
                parent[find(w)] = find(first[n])
            else:
                first[n] = w
    runs = defaultdict(list)
    for w in cand:
        runs[find(w)].append(w)
    out = {}
    n_runs = 0
    km = 0.0
    for ws in runs.values():
        k = sum(geo[w][0] for w in ws)
        if k > FILL_MAX_KM:
            continue
        keys = {named[o] for w in ws for n in np.asarray(ways[w][1]).tolist()
                for o in at.get(n, ()) if o not in cset and o in named}
        if len(keys) != 1:
            continue
        key = next(iter(keys))
        for w in ws:
            out[w] = key
        n_runs += 1
        km += k
    log(f"IR: {n_runs} unnamed runs of track named from the one line round them "
        f"({len(out)} ways, {km:,.0f} km of track)")
    return out


BRIDGE_KM = 4.0       # two dead ends of one line this close (crow-fly) are a gap in it...
BRIDGE_DETOUR = 2.0   # ... joined over other track no longer than this x the crow-fly + 1 km


def bridge_gaps(ways, coords, out, log):
    """A line whose track stops and starts again through a station yard OSM maps unnamed and
    outside the relation (Qom, where the Trans-Iranian's ends lie 1.5 km apart across the
    station's tracks): each pair of the line's dead ends within BRIDGE_KM is joined over the
    shortest run of track no other line has, and those ways become the line's. A line in two
    pieces lost the stretch from its last station to the gap (Parand - Qom, 130 km)."""
    import heapq
    inc = defaultdict(Counter)
    for w, k in out.items():
        nl = np.asarray(ways[w][1]).tolist()
        inc[k][nl[0]] += 1
        inc[k][nl[-1]] += 1
        for n in nl[1:-1]:
            inc[k][n] += 2
    # the graph of the track no line has (sidings and yards included), plus the line's own
    adj = defaultdict(list)
    for w, (t, nodes) in ways.items():
        if t.get("railway") not in TRACK_KIND or t.get("usage") in ("industrial", "military"):
            continue
        nl = np.asarray(nodes, dtype=np.int64)
        pos, ok = coords.many(nl)
        prev = None
        for n, p, good in zip(nl.tolist(), pos.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            xy = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            if prev is not None:
                adj[prev[0]].append((n, kr.dist_m(*prev[1], *xy) / 1000, w))
                adj[n].append((prev[0], kr.dist_m(*prev[1], *xy) / 1000, w))
            prev = (n, xy)
    n_join, km_join = 0, 0.0
    for k, c in inc.items():
        ends = [n for n, v in c.items() if v == 1 and coords.get(n) is not None]
        own = {n for n, v in c.items()}
        pairs = []
        for i, a in enumerate(ends):
            for b in ends[i + 1:]:
                d = kr.dist_m(*coords.get(a), *coords.get(b)) / 1000
                if d <= BRIDGE_KM:
                    pairs.append((d, a, b))
        done = set()
        for d, a, b in sorted(pairs):
            if a in done or b in done:
                continue
            cap = BRIDGE_DETOUR * d + 1.0
            dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
            while heap:
                dd, u = heapq.heappop(heap)
                if u == b or dd > cap:
                    break
                if dd > dist.get(u, kr.INF):
                    continue
                for v, wkm, w in adj.get(u, ()):
                    if w in out and out[w] != k:
                        continue
                    nd = dd + wkm
                    if nd <= cap and nd < dist.get(v, kr.INF):
                        dist[v] = nd
                        prev[v] = (u, w)
                        heapq.heappush(heap, (nd, v))
            if b not in dist:
                continue
            u, path_w = b, set()
            while u in prev:
                u, w = prev[u]
                path_w.add(w)
            new = {w for w in path_w if w not in out}
            if not new:
                continue
            for w in new:
                out[w] = k
            done |= {a, b}
            n_join += 1
            km_join += dist[b]
    log(f"IR: {n_join} gaps where a line's track stops and starts again bridged over other "
        f"track ({km_join:.1f} km)")


# Lines open to passengers that OSM still maps in pieces (the rest railway=construction, which
# extract.py does not keep): their pieces are joined by a straight link between the nearest
# dead ends, at most this far apart. Mianeh - Ardabil opened in 2026 (Raja's Tehran - Ardabil
# from late August); OSM has three gaps of 2.4 to 4.9 km in it.
STRAIGHT_GAPS = {"mianeh_ardabil": 6.0}
GAP_WAY_BASE = -7_000_000_000


def straight_gaps(ways, coords, out, log):
    made = []
    for key, lim in STRAIGHT_GAPS.items():
        ws = [w for w, k in out.items() if k == key]
        parent = {w: w for w in ws}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        first = {}
        deg = Counter()
        for w in ws:
            nl = np.asarray(ways[w][1]).tolist()
            deg[nl[0]] += 1
            deg[nl[-1]] += 1
            for n in nl[1:-1]:
                deg[n] += 2
            for n in nl:
                if n in first:
                    parent[find(w)] = find(first[n])
                else:
                    first[n] = w
        ends = defaultdict(list)                      # component -> dead-end nodes
        for n, v in deg.items():
            if v == 1 and coords.get(n) is not None:
                ends[find(first[n])].append(n)
        while len(ends) > 1:
            best = None
            comps = list(ends)
            for i, ca in enumerate(comps):
                for cb in comps[i + 1:]:
                    for a in ends[ca]:
                        for b in ends[cb]:
                            d = kr.dist_m(*coords.get(a), *coords.get(b)) / 1000
                            if d <= lim and (best is None or d < best[0]):
                                best = (d, a, b, ca, cb)
            if best is None:
                break
            d, a, b, ca, cb = best
            wid = GAP_WAY_BASE - len(made)
            ways[wid] = ({"railway": "rail", "name": LINES[key][0], "usage": "main",
                          "noritetsu:gap": "yes"}, np.asarray([a, b], dtype=np.int64))
            out[wid] = key
            made.append((key, d))
            ends[ca] = [n for n in ends[ca] if n != a] + [n for n in ends[cb] if n != b]
            del ends[cb]
    log(f"IR: {len(made)} gaps in OSM's track of a line open to passengers joined by a "
        f"straight link ({', '.join(f'{k} {d:.1f} km' for k, d in made)})")


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    with open(PROC / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    line_of, geo = assign(ways, coords, infra, log)
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords, line_of=line_of, geo=geo)
    _S["byobj"] = {id(ways[w][0]): LINES[k][0] for w, k in line_of.items()}
    return ways, stops, coords


def register_name(tags):
    return _S["byobj"].get(id(tags), "")


# =========================================================================== the timetable

def en_key(s):
    """An English station name for matching: lower case letters only, no "railway station",
    "station", "metro"; "Mashad" and "Mashhad" alike (doubled letters folded)."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(railway|rail ?way|train|metro|station|stations|sta|st|rail|ایستگاه)\b", " ", s)
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s.replace("kh", "x").replace("gh", "q").replace("ou", "u").replace("ee", "i")


def skel(s):
    """The consonants of en_key: romanisations of one Persian name differ mostly in vowels."""
    return re.sub(r"[aeiouy]", "", en_key(s))


def name_score(a, b):
    ka, kb = en_key(a), en_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb or skel(a) == skel(b):
        return 1.0
    if len(ka) >= 4 and (ka in kb or kb in ka):
        return 0.9
    return SequenceMatcher(None, skel(a), skel(b)).ratio()


def load_timetable():
    """data/raw/ir/timetable.json (`--parse`): RAI's trains with their calls. iranrail.net
    shows no calls for RAI's 40 local trains (Tehran - Garmsar, Pishva, Firuzkuh, Hashtgerd,
    Parand; Ahvaz - Khorramshahr, Karun - Mahshahr, Andimeshk - Dorud, Gorgan - Pol-e Sefid,
    Zahedan - Khash): those count as calling at their two ends."""
    p = RAW / "timetable.json"
    out = json.loads(p.read_text("utf-8")) if p.exists() else []
    out += [dict(t) for t in EXTRA_TRAINS]
    for t in out:
        if len(t.get("stops") or []) < 2 and t.get("from") and t.get("to"):
            t["stops"] = [t["from"], t["to"]]
            t["times"] = [[t.get("dep")] if t.get("dep") else [],
                          [t.get("arr")] if t.get("arr") else []]
            t["ends_only"] = True
    return out


# Trains newer than iranrail.net's copy of the timetable (last updated 2025-11-08).
EXTRA_TRAINS = [
    # Raja's Tehran - Ardabil over the Mianeh - Ardabil railway, from Shahrivar 1405 (late
    # August 2026), weekly: "قطار رفت با شماره ۴۹۲ روز چهارشنبه ... از تهران به مقصد اردبیل ...
    # در ایستگاه‌های کرج، قزوین، زنجان و میانه توقف دارد" (sanatmali.ir, 1405/06/02;
    # sharghdaily.com)
    {"num": "492", "evu": "RAJA", "evu_name": "Raja", "from": "Tehran", "to": "Ardabil",
     "dep": "22:40", "arr": "", "days": "train runs Wed", "bus": False,
     "stops": ["Tehran", "Karaj", "Qazvin", "Zanjan", "Mianeh", "Ardabil"], "times": []},
    {"num": "493", "evu": "RAJA", "evu_name": "Raja", "from": "Ardabil", "to": "Tehran",
     "dep": "", "arr": "", "days": "train runs Thu", "bus": False,
     "stops": ["Ardabil", "Mianeh", "Zanjan", "Qazvin", "Karaj", "Tehran"], "times": []},
]
# Passenger stations of a suspended line (SUSPENDED), on no running train: still stations, so
# the line exists, greyed. Tehran - Van called at Salmas and Razi.
HISTORIC_STOPS = ["Salmas", "Razi"]

# Trains of the timetable that do not run now, though iranrail.net still lists them.
NOT_RUNNING = {
    # Tehran - Van: suspended since 2026-03-06 (TCDD: "until further notice"; rayhaber.com
    # 2026-03); Türkiye's side is greyed for the same reason.
    "Tehran-Van", "Van-Tehran",
}


def running(t):
    """A train of RAI's timetable that counts: not a bus, not abroad only, not suspended."""
    if t.get("bus") or len(t.get("stops") or []) < 2:
        return False
    if f"{t['from']}-{t['to']}" in NOT_RUNNING:
        return False
    if t.get("evu") == "TCDD":
        return False
    return True


def osm_rail_stations(stops):
    """[(node, name, name_en, lon, lat)] of OSM's mainline rail stations, and the rail stop
    positions (railway=stop) that stand for a station OSM has no station node for (Zanjan,
    Ajabshir, Saraju, Behshahr): kr_register makes those stations too."""
    out = []
    for n, (t, lon, lat) in stops.items():
        if t.get("railway") not in ("station", "halt", "stop") and not (
                t.get("public_transport") in ("station", "stop_position")
                and t.get("train") == "yes"):
            continue
        if t.get("station") in ("subway", "light_rail", "monorail") and t.get("train") != "yes":
            continue
        if t.get("subway") == "yes" or t.get("light_rail") == "yes":
            if t.get("train") != "yes":
                continue
        if not t.get("name"):
            continue
        out.append((n, t["name"], t.get("name:en") or "", lon, lat))
    return out


def timetable_stops(stops, log=print):
    """{timetable stop name: OSM node} for the calls of running trains: the OSM rail station
    near iranrail.net's point for that name (STOP_MATCH_M) whose English name matches best,
    else the nearest one within STOP_NEAR_M; where iranrail has no point, the one OSM station
    whose English name matches."""
    tt = [t for t in load_timetable() if running(t)]
    pts = {}
    p = RAW / "iranrail_stations.json"
    if p.exists():
        for s in json.loads(p.read_text("utf-8")):
            if s.get("lat") is not None and s.get("lon") is not None:
                pts.setdefault(s["name"].strip(), (s["lon"], s["lat"]))
    osm = osm_rail_stations(stops)
    if not osm:
        return {}
    ox = np.array([o[3] for o in osm])
    oy = np.array([o[4] for o in osm])
    is_station = [stops[o[0]][0].get("railway") == "station" for o in osm]
    names = Counter(s for t in tt for s in t["stops"])
    for h in HISTORIC_STOPS:
        names.setdefault(h, 0)
    out, how = {}, Counter()
    for nm in names:
        pt = pts.get(nm)
        best = None
        alias = TT_ALIAS.get(nm)
        if alias:
            hit = [i for i, o in enumerate(osm) if o[1] == alias]
            if pt is not None and len(hit) > 1:
                hit.sort(key=lambda i: kr.dist_m(osm[i][3], osm[i][4], *pt))
            if hit:
                best = (None, hit[0], "alias")
        if best is None and pt is not None:
            d = np.hypot((ox - pt[0]) * math.cos(math.radians(pt[1])) * 111320,
                         (oy - pt[1]) * 110570)
            # the English name first, anywhere within NAME_FAR_M: iranrail's points are
            # sometimes another station's (its "Saveh" is Shohada-ye Parandak, 45 km off)
            # (a station record before a stop position of the same name: kr_register keeps
            # the record, and the stop position is no station of its own)
            for i in np.nonzero(d <= NAME_FAR_M)[0].tolist():
                sc = name_score(nm, osm[i][2])
                lim = 0.95 if d[i] > STOP_MATCH_M else 0.75
                rank = (round(sc, 2), is_station[i], -d[i])
                if sc >= lim and (best is None or rank > best[0]):
                    best = (rank, i, "name+point")
            if best is None:
                near = [i for i in np.nonzero(d <= STOP_NEAR_M)[0].tolist()]
                if near:
                    j = max(near, key=lambda i: (is_station[i], -d[i]))
                    best = (None, j, "point")
        if best is None:
            sc = [(name_score(nm, o[2]), i) for i, o in enumerate(osm)]
            top = [i for s, i in sc if s >= 0.95]
            if top and all(kr.dist_m(osm[i][3], osm[i][4], osm[top[0]][3], osm[top[0]][4])
                           <= 2000 for i in top):
                top.sort(key=lambda i: (osm[i][0] not in stops or
                                        stops[osm[i][0]][0].get("railway") != "station"))
                best = (None, top[0], "name")
        if best is None:
            how["unmatched"] += 1
            continue
        out[nm] = osm[best[1]][0]
        how[best[2]] += 1
    miss = sorted((n for n in names if n not in out), key=lambda n: -names[n])
    log(f"IR: timetable, {len(tt)} running trains calling at {len(names)} places; matched to "
        f"OSM stations {dict(how)}; unmatched: {', '.join(miss[:60])}")
    return out


# =========================================================================== stations

BORDER_BASE = -9_000_000_000


def passenger_filter(st, node_st, by_key, by_base, stops, log):
    """Keep, of kr_register's stations, the rail ones that are passenger stops (the module
    docstring). Metro and light-rail stations are kept as they are (never on register track)."""
    import build_model as bm
    tts = timetable_stops(stops, log)
    on_tt = set(tts.values())
    route_st = set()
    for rid, (tags, members) in _S["rels"].items():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        for n in bm.stop_members(members):
            if n in node_st:
                route_st.add(node_st[n])
    keep, n_tt, n_rt, n_gone = {}, 0, 0, 0
    urban_ids = set()
    for sid, s in st.items():
        tags = stops.get(sid, ({}, 0, 0))[0]
        urban = (tags.get("station") in ("subway", "light_rail", "monorail")
                 or tags.get("subway") == "yes" or tags.get("light_rail") == "yes"
                 or tags.get("railway") == "tram_stop") and tags.get("train") != "yes"
        if urban:
            urban_ids.add(sid)
            keep[sid] = s
        elif sid in on_tt or node_st.get(sid) in on_tt:
            keep[sid] = s
            n_tt += 1
        elif sid in route_st:
            keep[sid] = s
            n_rt += 1
        else:
            n_gone += 1
    # a timetable stop that kr merged into another record of its name: that record; one that
    # is no station record of kr's at all: the rail station record within MERGE_TT_M of it
    merged = {node_st.get(n) for n in on_tt} - set(keep) - {None}
    rail_ids = [sid for sid in st if sid not in urban_ids]
    rx = np.array([st[s]["lon"] for s in rail_ids]) if rail_ids else np.zeros(0)
    ry = np.array([st[s]["lat"] for s in rail_ids]) if rail_ids else np.zeros(0)
    tt_station = {}
    for n in on_tt:
        if n in st:
            tt_station[n] = n
            continue
        if node_st.get(n) in st:
            tt_station[n] = node_st[n]
            continue
        if n not in stops or not rail_ids:
            continue
        _t, lon, lat = stops[n]
        d = np.hypot((rx - lon) * math.cos(math.radians(lat)) * 111320, (ry - lat) * 110570)
        j = int(np.argmin(d))
        if d[j] <= MERGE_TT_M:
            merged.add(rail_ids[j])
            tt_station[n] = rail_ids[j]
    merged -= set(keep)
    # each running train's calls as station records, for served_junction_sections
    _S["tt_calls"] = [[tt_station[tts[s]] for s in t["stops"] if tts.get(s) in tt_station]
                      for t in load_timetable() if running(t)]
    for sid in merged:
        if sid in st:
            keep[sid] = st[sid]
            n_tt += 1
    # Records of one station under different Persian names: Parand is "شهر پرند", "راه آهن
    # پرند" and "ایستگاه قطار بین شهری پرند", all "Shahr-e Parand" in English, within 100 m, and
    # kr_register kept them apart, so the Trans-Iranian and the Tehran - Hamedan line met at no
    # common station. Rail records within SAME_EN_M of each other with one English name are one.
    rail_keep = [s for s in keep if s not in urban_ids]
    alias = {}
    for i, a in enumerate(rail_keep):
        if a in alias:
            continue
        ka = en_key(st[a].get("name_en"))
        if not ka:
            continue
        for b in rail_keep[i + 1:]:
            if b in alias or en_key(st[b].get("name_en")) != ka:
                continue
            if kr.dist_m(st[a]["lon"], st[a]["lat"], st[b]["lon"], st[b]["lat"]) <= SAME_EN_M:
                alias[b] = a
    for b, a in alias.items():
        if b in on_tt or b in route_st:
            route_st.add(a)
        del keep[b]
    node_st = {n: alias.get(s, s) for n, s in node_st.items()}
    for b, a in alias.items():
        node_st[b] = a
    on_tt = {alias.get(n, n) for n in on_tt}
    merged = {alias.get(n, n) for n in merged}
    _S["tt_calls"] = [[alias.get(x, x) for x in calls] for calls in _S["tt_calls"]]
    if alias:
        log(f"IR: {len(alias)} rail station records merged into another of the same English "
            f"name within {SAME_EN_M} m")
    node_st = {n: s for n, s in node_st.items() if s in keep}
    by_key = defaultdict(list, {k: [s for s in v if s in keep] for k, v in by_key.items()})
    by_base = defaultdict(list, {k: [s for s in v if s in keep] for k, v in by_base.items()})
    _S["passenger"] = {s for s in keep if s in on_tt or s in route_st or s in merged}
    log(f"IR: passenger stations: {n_tt} called at in RAI's timetable, {n_rt} more as an OSM "
        f"route's stop; {n_gone} OSM rail stations left out as no passenger stop")
    return keep, node_st, by_key, by_base


def build_stations(stops, log):
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    st, node_st, by_key, by_base = passenger_filter(st, node_st, by_key, by_base, stops, log)
    _S.update(st=st, node_st=node_st, border={})
    junc = junction_ends(log)
    _S["branch"] = set()
    for n, k in branch_junctions(log):
        junc[n].add(k)
        _S["branch"].add(n)
    for n in junc:
        p = _S["coords"].get(n)
        if p is None or n in st:
            continue
        st[n] = {"name": f"gj{n}", "name_en": "", "lon": p[0], "lat": p[1], "rank": 3}
        by_key[kr.name_key(f"gj{n}")].append(n)
        by_base[kr.base_key(f"gj{n}")].append(n)
    _S["junction"] = {n: v for n, v in junc.items() if st.get(n, {}).get("name") == f"gj{n}"}
    return st, node_st, by_key, by_base


CONNECT_KM = 5.0      # a line's dead end this far over other track from another line is a junction

# Where a line branches between two of its stations with no station at the fork: a junction
# station there, so kr_register does not pair the stations either side of the fork with each
# other past it. (lon, lat, line): the fork east of Bandar Torkaman where the spur into Gorgan
# leaves the line to Incheh Borun.
BRANCHES = [(54.2155, 36.8868, "garmsar_inchehborun")]


def branch_junctions(log):
    ways, coords, line_of = _S["ways"], _S["coords"], _S["line_of"]
    out = []
    for lon, lat, key in BRANCHES:
        best = None
        for w, k in line_of.items():
            if k != key:
                continue
            nl = np.asarray(ways[w][1], dtype=np.int64)
            pos, ok = coords.many(nl)
            for n, p, g in zip(nl.tolist(), pos.tolist(), ok.tolist()):
                if g:
                    d = kr.dist_m(coords.x[p] / 1e7, coords.y[p] / 1e7, lon, lat)
                    if best is None or d < best[0]:
                        best = (d, n)
        if best and best[0] <= 300:
            out.append((best[1], key))
        else:
            log(f"IR: no track of {key} within 300 m of the fork at {lon}, {lat}")
    return out


def junction_ends(log):
    """{node: {line key}}: a line's dead end that lies on another line's track, or reaches it
    over at most CONNECT_KM of track no line has, with no passenger stop within JUNCTION_M of
    it (load_lists puts that stop on the line instead). OSM Iran ends a railway's relation at
    the junction, often kilometres from the nearest passenger station (Qazvin - Rasht leaves
    the Tehran - Tabriz line at Siah Cheshmeh, the Chabahar line the Kerman - Zahedan line at
    Jiguli): the stretch from the junction to the line's first station is a section ending at
    a `junction` station, kept where the timetable's trains run over it (gtfs_served)."""
    import heapq
    ways, coords, line_of = _S["ways"], _S["coords"], _S["line_of"]
    st, pas = _S["st"], _S.get("passenger", set())
    _S["junction_on"] = defaultdict(set)
    inc = defaultdict(Counter)
    on = defaultdict(set)
    adj = defaultdict(list)
    for w, (t, nodes) in ways.items():
        nl = np.asarray(nodes).tolist()
        k = line_of.get(w)
        if k:
            c = inc[k]
            c[nl[0]] += 1
            c[nl[-1]] += 1
            for n in nl[1:-1]:
                c[n] += 2
            for n in nl:
                on[n].add(k)
            continue
        if t.get("railway") not in TRACK_KIND or t.get("usage") in ("industrial", "military"):
            continue
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        prev = None
        for n, p, good in zip(nl, pos.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            xy = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            if prev is not None:
                d = kr.dist_m(*prev[1], *xy) / 1000
                adj[prev[0]].append((n, d))
                adj[n].append((prev[0], d))
            prev = (n, xy)
    pids = [s for s in pas if s in st]
    px = np.array([st[s]["lon"] for s in pids]) if pids else np.zeros(0)
    py = np.array([st[s]["lat"] for s in pids]) if pids else np.zeros(0)
    out = defaultdict(set)
    n_on = n_reach = 0
    for k, c in inc.items():
        for n, v in c.items():
            if v != 1:
                continue
            p = coords.get(n)
            if p is None:
                continue
            if pids.__len__():
                d = np.hypot((px - p[0]) * math.cos(math.radians(p[1])) * 111320,
                             (py - p[1]) * 110570)
                if d.min() <= JUNCTION_M:
                    continue
            if on[n] - {k}:
                out[n].add(k)
                _S.setdefault("junction_on", defaultdict(set))[n] |= on[n] - {k}
                n_on += 1
                continue
            dist, heap, hit = {n: 0.0}, [(0.0, n)], False
            while heap:
                dd, u = heapq.heappop(heap)
                if dd > CONNECT_KM:
                    break
                if u != n and on.get(u, set()) - {k}:
                    hit = True
                    _S.setdefault("junction_on", defaultdict(set))[n] |= on[u] - {k}
                    break
                if dd > dist.get(u, kr.INF):
                    continue
                for vv, w in adj.get(u, ()):
                    nd = dd + w
                    if nd < dist.get(vv, kr.INF):
                        dist[vv] = nd
                        heapq.heappush(heap, (nd, vv))
            if hit:
                out[n].add(k)
                n_reach += 1
    log(f"IR: {n_on} line ends on another line's track and {n_reach} reaching one over other "
        f"track (<= {CONNECT_KM} km) taken as junctions")
    return out


def load_lists(path, log):
    """{line: [[station name, ...]]}: every passenger stop within LIST_M of the line's track,
    and the one nearest a line end that lies on another line's track (JUNCTION_M)."""
    from scipy.spatial import cKDTree
    ways, coords, st = _S["ways"], _S["coords"], _S["st"]
    pas = _S.get("passenger", set(st))
    by_line = defaultdict(list)
    for w, k in _S["line_of"].items():
        by_line[LINES[k][0]].append(w)
    kx = 111.32 * math.cos(math.radians(33.0))
    trees = {}
    for ln, ws in by_line.items():
        px, py = [], []
        for w in ws:
            p, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
            p = p[ok]
            x, y = coords.x[p] / 1e7 * kx, coords.y[p] / 1e7 * 110.57
            # a vertex at least every DENSE_KM: desert track is straight for kilometres
            # between OSM's vertices, and a station beside it was further than LIST_M from all
            for i in range(len(x) - 1):
                k = max(1, int(math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / DENSE_KM))
                t = np.arange(k) / k
                px.append(x[i] + (x[i + 1] - x[i]) * t)
                py.append(y[i] + (y[i + 1] - y[i]) * t)
            if len(x):
                px.append(x[-1:])
                py.append(y[-1:])
        trees[ln] = cKDTree(np.c_[np.concatenate(px), np.concatenate(py)])
    lists = defaultdict(set)
    for sid, s in st.items():
        if sid not in pas:
            continue
        for ln, tr in trees.items():
            d, _i = tr.query((s["lon"] * kx, s["lat"] * 110.57))
            if d * 1000 <= LIST_M:
                lists[ln].add(s["name"])
    line_of = _S["line_of"]
    inc = defaultdict(Counter)
    on = defaultdict(set)
    for w, k in line_of.items():
        nl = np.asarray(ways[w][1]).tolist()
        inc[k][nl[0]] += 1
        inc[k][nl[-1]] += 1
        for n in nl[1:-1]:
            inc[k][n] += 2
        for n in nl:
            on[n].add(k)
    real = [(sid, s) for sid, s in st.items() if sid in pas]
    sx = np.array([s["lon"] for _i, s in real])
    sy = np.array([s["lat"] for _i, s in real])
    n_junc = 0
    for k, c in inc.items():
        ln = LINES[k][0]
        for n, v in c.items():
            if v != 1:
                continue
            p = coords.get(n)
            if p is None or not len(real):
                continue
            d = np.hypot((sx - p[0]) * math.cos(math.radians(p[1])) * 111320,
                         (sy - p[1]) * 110570)
            j = int(np.argmin(d))
            if d[j] <= JUNCTION_M and real[j][1]["name"] not in lists[ln]:
                lists[ln].add(real[j][1]["name"])
                n_junc += 1
                log(f"    end of {ln}: {real[j][1]['name']} ({d[j]:.0f} m)")
    log(f"IR: {n_junc} stations listed on the line whose track ends near them")
    for n, ks in _S.get("junction", {}).items():
        for k in ks:
            lists[LINES[k][0]].add(f"gj{n}")
    log(f"IR: station lists by proximity ({LIST_M} m): "
        f"{sum(len(v) for v in lists.values())} station-line pairs on {len(lists)} lines")
    return ({k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {})


# =========================================================================== joining pieces

GAP_KM = 12.0          # two pieces of one line whose stations are this close (crow-fly)...
GAP_DETOUR = 1.5       # ... are joined over any rail track no longer than this x the crow-fly
GAP_EXTRA_KM = 1.0     # ... plus this


class Net:
    """Every rail way that is not left out, as one graph (th_register's)."""

    def __init__(self, ways, coords):
        self.adj = defaultdict(list)
        self.xy = {}
        for wid, (t, nodes) in ways.items():
            if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
                continue
            n = tidy(t.get("name"))
            if n in NAME_LINE and NAME_LINE[n] is None:
                continue
            nodes = np.asarray(nodes, dtype=np.int64)
            pos, ok = coords.many(nodes)
            prev = None
            for nd, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
                if not good:
                    prev = None
                    continue
                if nd not in self.xy:
                    self.xy[nd] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
                if prev is not None and prev != nd:
                    w = kr.dist_m(*self.xy[prev], *self.xy[nd]) / 1000
                    self.adj[prev].append((nd, w))
                    self.adj[nd].append((prev, w))
                prev = nd
        self.ids = np.fromiter(self.xy.keys(), dtype=np.int64)
        a = np.array([self.xy[i] for i in self.ids.tolist()])
        self.x, self.y = a[:, 0], a[:, 1]

    def node_near(self, lon, lat):
        d = np.hypot((self.x - lon) * math.cos(math.radians(lat)), self.y - lat)
        return int(self.ids[int(np.argmin(d))])

    def path(self, a, b, cap):
        import heapq
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                break
            if d > dist.get(u, kr.INF):
                continue
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd <= cap and nd < dist.get(v, kr.INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if b not in dist:
            return None
        nodes = [b]
        while nodes[-1] in prev:
            nodes.append(prev[nodes[-1]])
        return nodes[::-1], dist[b]


def join_pieces(lines, stations, geoms, log):
    """A line in several pieces is joined where two of its stations in different pieces are
    within GAP_KM and rail track joins them within GAP_DETOUR x the crow-fly + GAP_EXTRA_KM,
    nearest first (th_register's join_pieces)."""
    from n02 import walk_order
    net = Net(_S["ways"], _S["coords"])
    done = []
    for l in lines:
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b, *_ in l["sections"]:
            parent[find(a)] = find(b)
        sts = sorted({s for sec in l["sections"] for s in sec[:2]})
        if len({find(s) for s in sts}) < 2:
            continue
        cands = []
        for i, a in enumerate(sts):
            for b in sts[i + 1:]:
                if find(a) != find(b):
                    sa, sb = stations[a], stations[b]
                    crow = kr.dist_m(sa["lon"], sa["lat"], sb["lon"], sb["lat"]) / 1000
                    if crow <= GAP_KM:
                        cands.append((crow, a, b))
        for crow, a, b in sorted(cands):
            if find(a) == find(b):
                continue
            sa, sb = stations[a], stations[b]
            got = net.path(net.node_near(sa["lon"], sa["lat"]), net.node_near(sb["lon"], sb["lat"]),
                           GAP_DETOUR * crow + GAP_EXTRA_KM)
            if got is None:
                continue
            nodes, km = got
            parent[find(a)] = find(b)
            l["sections"].append([a, b, round(km, 3)])
            geoms[l["id"]][f"{a}|{b}"] = (
                [[round(sa["lon"], 5), round(sa["lat"], 5)]]
                + [[round(net.xy[n][0], 5), round(net.xy[n][1], 5)] for n in nodes]
                + [[round(sb["lon"], 5), round(sb["lat"], 5)]])
            if isinstance(l.get("highspeed_sections"), dict):
                l["highspeed_sections"][f"{a}|{b}"] = False
            done.append((l["name"], sa["name"], sb["name"], km))
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["display"] = walk_order([(a, b) for a, b, *_ in l["sections"]])
        left = len({find(s) for s in sts})
        if left > 1:
            log(f"IR: {l['name']} is still in {left} pieces")
    log(f"IR: {len(done)} gaps in a line joined over other track "
        f"({sum(d[3] for d in done):.1f} km)")
    for nm, a, b, km in done:
        log(f"    {nm}: {a} - {b} {km:.1f} km")


# =========================================================================== build

_orig = {}

# Lines no passenger train runs over now (greyed, out of completion; ir_sources.md).
SUSPENDED = {
    # Sufian - Razi: its only regular passenger train, Tehran - Van, suspended since 2026-03;
    # RAI's Tabriz - Salmas local ran once a fortnight, and is not in the 2025-26 timetable
    "sufian_razi",
}


COVER_M = 30          # mx_register's drop_junction_runs, copied: a section whose track ...
COVER_SHARE = 0.9     # ... two others sharing a station with it cover this much of, within COVER_M


def _xy_m(pts, lat0):
    a = np.asarray(pts, dtype=np.float64)
    return np.column_stack([a[:, 0] * math.cos(math.radians(lat0)) * 111320, a[:, 1] * 110570])


def _covered_share(pts, others, lat0):
    """The share of polyline `pts` (sampled every 25 m) within COVER_M of any of `others`."""
    p = _xy_m(pts, lat0)
    seg = np.hypot(*np.diff(p, axis=0).T)
    samples = [p[0]]
    for (a, b), L in zip(zip(p[:-1], p[1:]), seg):
        k = max(1, int(L // 25))
        samples.extend(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    s = np.asarray(samples)
    best = np.full(len(s), np.inf)
    for o in others:
        q = _xy_m(o, lat0)
        A, B = q[:-1], q[1:]
        d = B - A
        L2 = np.maximum((d * d).sum(1), 1e-9)
        for i in range(0, len(s), 400):
            ss = s[i:i + 400]
            t = np.clip(((ss[:, None, :] - A[None]) * d[None]).sum(2) / L2[None], 0, 1)
            proj = A[None] + t[..., None] * d[None]
            best[i:i + 400] = np.minimum(best[i:i + 400],
                                         np.hypot(*(ss[:, None, :] - proj).transpose(2, 0, 1)).min(1))
    return float((best <= COVER_M).mean())


def drop_junction_runs(lines, geoms, log):
    """mx_register's rule: where a branch leaves a line between two stations (the spur into
    Gorgan between Bandar Torkaman and Yampi), kr_register pairs the stations either side of
    the junction with each other too; the section whose track the two others cover goes."""
    from n02 import walk_order
    n_drop, km_drop, what = 0, 0.0, []
    for l in lines:
        g = geoms[l["id"]]
        secs = sorted(l["sections"], key=lambda s: -s[2])
        keep = list(secs)
        for s in secs:
            a, b = s[0], s[1]
            pts = g.get(f"{a}|{b}")
            if not pts or len(pts) < 2:
                continue
            rest = [x for x in keep if x is not s]
            nb = defaultdict(dict)
            for x in rest:
                nb[x[0]][x[1]] = x
                nb[x[1]][x[0]] = x
            for c in set(nb[a]) & set(nb[b]):
                o = [g.get(f"{x[0]}|{x[1]}") for x in (nb[a][c], nb[b][c])]
                if not all(o):
                    continue
                if _covered_share(pts, o, pts[0][1]) >= COVER_SHARE:
                    keep = rest
                    n_drop += 1
                    km_drop += s[2]
                    what.append(f"{l['name']} {s[2]:.1f} km")
                    g.pop(f"{a}|{b}", None)
                    if isinstance(l.get("highspeed_sections"), dict):
                        l["highspeed_sections"].pop(f"{a}|{b}", None)
                    break
        if len(keep) != len(l["sections"]):
            ks = {(x[0], x[1]) for x in keep}
            l["sections"] = [x for x in l["sections"] if (x[0], x[1]) in ks]
            l["km"] = round(sum(x[2] for x in l["sections"]), 3)
            l["display"] = walk_order([(x[0], x[1]) for x in l["sections"]])
    log(f"IR: {n_drop} sections dropped as runs past a junction ({km_drop:,.1f} km): "
        + ", ".join(what))


def served_junction_sections(lines, log):
    """`served_sections` for the junction-ended sections RAI's trains run over. gtfs_served
    keeps a junction-ended section its paths cross unless it is "weak" (crossed only by runs
    over 40 km between calls), a guard against stray paths in Europe's dense networks; in
    Iran calls 100-300 km apart are the rule, and that dropped Qazvin - Rasht, Arak - Malayer
    and Yazd - Eqlid whole. Here a section from a junction J to a station of line L is served
    when some train calls, one call after the other, at a station of L and at a station of a
    line that meets L at J, neither station being on both lines: the train ran through J."""
    key_of = {v[0]: k for k, v in LINES.items()}
    st_of = defaultdict(set)
    for l in lines:
        k = key_of.get(l["name"])
        for a, b, *_ in l["sections"]:
            for s in (a, b):
                if s.startswith("k") and s[1:].lstrip("-").isdigit():
                    st_of[k].add(int(s[1:]))
    pairs = set()
    for calls in _S.get("tt_calls", ()):
        for x, y in zip(calls[:-1], calls[1:]):
            if x != y:
                pairs.add((x, y))
                pairs.add((y, x))
    jon = _S.get("junction_on", {})
    n_kept, km_kept, names = 0, 0.0, []
    for l in lines:
        k = key_of.get(l["name"])
        keep = []
        for a, b, km, *_ in l["sections"]:
            for jn, other in ((a, b), (b, a)):
                if not jn.startswith("k") or int(jn[1:]) not in _S.get("junction", {}):
                    continue
                ms = jon.get(int(jn[1:]), set())
                mine = st_of[k]
                s_end = int(other[1:]) if other[1:].lstrip("-").isdigit() else None
                # (a suspended line keeps its junction sections: they are greyed whole; at a
                # fork inside one line, a train calling at the section's station and next at
                # another station of the line ran through the fork)
                hit = (k in SUSPENDED
                       or any((x in mine - st_of[m] and y in st_of[m] - mine)
                              for m in ms for x, y in pairs)
                       or (int(jn[1:]) in _S.get("branch", ())
                           and any(x == s_end and y in mine for x, y in pairs)))
                if hit:
                    keep.append(f"{a}|{b}")
                    n_kept += 1
                    km_kept += km
                    names.append(f"{l['name']} {km:.0f} km")
                break
        if keep:
            l["served_sections"] = keep
    log(f"IR: {n_kept} junction-ended sections ({km_kept:,.0f} km) kept as served: trains call "
        f"either side of the junction one after the other: {'; '.join(names)}")


# "Railway station", "train station", "station" before a station's name: OSM Iran writes
# Tehran's as both "ایستگاه راه آهن تهران" (the station) and "تهران" (its stop position), which
# kr_register kept as two stations 20 m apart, so no train could get from one line to another.
STATION_WORDS = re.compile(r"^(?:ایستگاه|ایستکاه)\s*(?:راه\s*‌?\s*[آا]هن|قطار|راه‌آهن)?\s*"
                           r"(?:بین\s*‌?\s*شهری\s+)?|^راه\s*‌?\s*[آا]هن\s+")


def plain_station(name):
    """A station's name without "ایستگاه راه آهن": "ایستگاه راه آهن تهران" is "تهران"."""
    n = (name or "").replace("ي", "ی").replace("ك", "ک")
    n = STATION_WORDS.sub("", n.strip()).strip()
    return n or (name or "")


def adopt():
    if _orig:
        return
    _orig["name_key"] = kr.name_key
    kr.name_key = lambda name: _orig["name_key"](plain_station(name).replace("‌", ""))
    _orig["build_stations"] = kr.build_stations
    kr.load_osm = load_osm
    kr.register_name = register_name
    kr.load_lists = load_lists
    kr.build_stations = build_stations
    kr.line_id = line_id
    kr.TRACK_KIND = TRACK_KIND
    kr.NOT_PASSENGER = NOT_PASSENGER
    kr.NAME_ALIAS = {}
    kr.STATION_ALIAS = {}
    kr.MATCH_M = JUNCTION_M


def build(path, log):
    adopt()
    import gb_register as gb
    lines, stations, geoms = kr.build(path, log)
    join_pieces(lines, stations, geoms, log)
    drop_junction_runs(lines, geoms, log)
    gb.drop_shortcuts(lines, geoms, log)
    served_junction_sections(lines, log)
    en_of = {v[0]: v[1] for v in LINES.values()}
    key_of = {v[0]: k for k, v in LINES.items()}

    junc = {f"k{n}": f"ij{n}" for n in _S.get("junction", {})}

    def r(sid):
        return junc.get(sid) or ("i" + sid[1:] if sid.startswith("k") else sid)

    real = [s for sid, s in stations.items() if sid not in junc]
    rx = np.array([s["lon"] for s in real]) if real else np.zeros(0)
    ry = np.array([s["lat"] for s in real]) if real else np.zeros(0)
    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        if sid in junc:
            s["junction"] = True
            d = np.hypot((rx - s["lon"]) * math.cos(math.radians(s["lat"])), ry - s["lat"])
            near = plain_station(real[int(np.argmin(d))]["name"]) if d.size else ""
            s["name"] = f"Junction near {near}" if near else "Junction"
        else:
            s["name"] = plain_station(s["name"])
        out_st[nid] = s
    out_geoms = {}
    for l in lines:
        l["src"] = "ir"
        l["operator"], l["operator_en"] = RAI, RAI_EN
        l["name_en"] = en_of.get(l["name"], "")
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        if l.get("served_sections"):
            l["served_sections"] = ["|".join(r(x) for x in k.split("|"))
                                    for k in l["served_sections"]]
        if key_of.get(l["name"]) in SUSPENDED:
            l["suspended"] = True
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    log(f"IR: {len(lines)} register lines, {sum(l['km'] for l in lines):,.0f} km, "
        f"{len(out_st)} stations")
    for l in sorted(lines, key=lambda l: -l["km"]):
        log(f"    {l['km']:8.1f} km  {len(l['sections']):3d} sections  {l['name']}"
            + ("  (suspended)" if l.get("suspended") else ""))
    return lines, out_st, out_geoms


# =========================================================================== the crawl, the feed

IRANRAIL = "https://www.iranrail.net/"


def _get(url):
    import ssl
    import time
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False          # its certificate chain does not verify from here
    ctx.verify_mode = ssl.CERT_NONE
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                return r.read()
        except Exception as e:                          # noqa: BLE001
            print("  retry", url, e, flush=True)
            time.sleep(5 * (k + 1))
    return None


def crawl():
    """iranrail.net (an unofficial site that reproduces RAI's realtime system pws0.rai.ir,
    which answers 403 outside Iran): alltrains.php, each train's times.php (its calls) and
    info.php (operator, days), stations.php and each station's location.php (its point)."""
    import time
    import urllib.parse
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "iranrail").mkdir(exist_ok=True)
    for fn, page in (("iranrail_alltrains.html", "alltrains.php"),
                     ("iranrail_stations.html", "stations.php")):
        body = _get(IRANRAIL + page)
        if body:
            (RAW / fn).write_bytes(body)
    t = (RAW / "iranrail_alltrains.html").read_text("utf-8", "replace")
    trains = []
    for r in re.findall(r'<tr class="details">(.*?)</tr>', t, re.S):
        num = re.search(r"<b>(.*?)</b>", r).group(1)
        evu = re.search(r'data-tooltip="(.*?)"', r).group(1)
        m = re.search(r"times\.php\?trainNumber=([^&]*)&EVU=([^&]*)&", r)
        od = re.search(r'width=25%><div align=left>(.*?)<br>(.*?)</td>', r)
        dep = re.search(r"&#8599;([\d:]+)<br>&#8600;([\d:]+)", r)
        trains.append({"num": html.unescape(num), "evu_name": evu, "evu": m.group(2) if m else "",
                       "from": html.unescape(od.group(1)) if od else "",
                       "to": html.unescape(od.group(2)) if od else "",
                       "dep": dep.group(1) if dep else "", "arr": dep.group(2) if dep else ""})
    (RAW / "iranrail_alltrains.json").write_text(json.dumps(trains, ensure_ascii=False, indent=0),
                                                 "utf-8")
    for i, tr in enumerate(trains):
        key = f"{tr['evu']}_{tr['num']}".replace("/", "-")
        for kind, url in (
                ("times", IRANRAIL + "times.php?" + urllib.parse.urlencode(
                    {"trainNumber": tr["num"], "EVU": tr["evu"], "stop": "", "codefrom": "",
                     "ndest": tr["to"]})),
                ("info", IRANRAIL + "info.php?" + urllib.parse.urlencode(
                    {"trainNumber": tr["num"], "EVU": tr["evu"]}))):
            fn = RAW / "iranrail" / f"{kind}_{key}.html"
            if fn.exists() and fn.stat().st_size > 50:
                continue
            body = _get(url)
            if body is not None:
                fn.write_bytes(body)
            time.sleep(0.8)
        if i % 20 == 0:
            print(i, key, flush=True)
    st = (RAW / "iranrail_stations.html").read_text("utf-8", "replace")
    out = []
    for sid, name in re.findall(r'location\.php\?id=(\d+)">([^<]*)<', st):
        body = _get(f"{IRANRAIL}location.php?id={sid}")
        if body is None:
            continue
        m = re.search(r"maps\?q=([-\d.]+)%2C%20([-\d.]+)", body.decode("utf-8", "replace"))
        out.append({"id": sid, "name": html.unescape(name).strip(),
                    "lat": float(m.group(1)) if m else None,
                    "lon": float(m.group(2)) if m else None})
        time.sleep(0.6)
    (RAW / "iranrail_stations.json").write_text(json.dumps(out, ensure_ascii=False, indent=0),
                                                "utf-8")
    parse()


def parse():
    """data/raw/ir/iranrail/*.html -> data/raw/ir/timetable.json."""
    trains = json.loads((RAW / "iranrail_alltrains.json").read_text("utf-8"))
    out = []
    for t in trains:
        key = f"{t['evu']}_{t['num']}".replace("/", "-")
        ft, fi = RAW / "iranrail" / f"times_{key}.html", RAW / "iranrail" / f"info_{key}.html"
        if not ft.exists():
            continue
        tt = ft.read_text("utf-8", "replace")
        stops, times = [], []
        for a, b, c in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>", tt):
            stops.append(html.unescape(re.sub("<.*?>", "", a)).strip())
            tm = re.findall(r"\d\d:\d\d", b + " " + c)
            times.append(tm)
        upd = re.search(r"last update: </i>\s*([\d-]+)", tt)
        info = fi.read_text("utf-8", "replace") if fi.exists() else ""
        cells = [html.unescape(re.sub("<.*?>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", info, re.S)]
        days = next((c for c in cells if "runs" in c.lower() or "daily" in c.lower()), "")
        out.append({**{k: t[k] for k in ("num", "evu", "evu_name", "from", "to", "dep", "arr")},
                    "stops": stops, "times": times, "days": days, "info": cells[:3],
                    "bus": days.lower().startswith("bus") or "bus" in " ".join(cells[:2]).lower(),
                    "updated": upd.group(1) if upd else ""})
    (RAW / "timetable.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), "utf-8")
    print(f"timetable.json: {len(out)} trains and buses, {sum(1 for o in out if o['bus'])} buses")


FEED_START = (2026, 10, 5)     # the feed's four weeks: what "runs every second day" etc. become
FEED_DAYS = 28
WEEKDAY = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def running_dates(days, seed):
    from datetime import date, timedelta
    d0 = date(*FEED_START)
    all_days = [d0 + timedelta(i) for i in range(FEED_DAYS)]
    s = (days or "").lower()
    named = [WEEKDAY[w] for w in re.findall(r"\b(mon|tue|wed|thu|fri|sat|sun)\b", s)]
    if named:
        return [d for d in all_days if d.weekday() in named]
    if "second day" in s:
        return all_days[seed % 2::2]
    if "4 days" in s or "four days" in s:
        return all_days[seed % 4::4]
    if "not every day" in s:
        return [d for d in all_days if d.weekday() in (0, 3)]
    return all_days


def timetable(log=print):
    """RAI's running trains (timetable.json) as a GTFS feed, data/raw/gtfs/ir/ir_iranrail.gtfs.zip,
    which gtfs_served reads as it reads every national feed. Each call is OSM's station record
    (timetable_stops), under OSM's name and point, so gtfs_served matches it exactly; a call
    with no OSM station is left out (gtfs_served steps over it)."""
    import csv
    import io
    import zipfile
    from datetime import date
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    m = timetable_stops(stops, log)
    tt = [t for t in load_timetable() if running(t)]
    used, routes, trips, st, cal = set(), [], [], [], []
    for t in tt:
        calls = [(i, m[s]) for i, s in enumerate(t["stops"]) if s in m]
        calls = [c for k, c in enumerate(calls) if k == 0 or c[1] != calls[k - 1][1]]
        if len(calls) < 2:
            continue
        rid = f"{t['evu']}_{t['num']}"
        routes.append((rid, t["evu"] or "rai", t["num"], f"{t['from']} - {t['to']}", "2"))
        trips.append((rid, rid, rid))
        prev = -1
        for seq, (i, n) in enumerate(calls, 1):
            tm = (t.get("times") or [[]] * len(t["stops"]))[i] or []
            arr = dep = ""
            if tm:
                a = int(tm[0][:2]) * 3600 + int(tm[0][3:]) * 60
                d = int(tm[-1][:2]) * 3600 + int(tm[-1][3:]) * 60
                while a < prev:
                    a += 86400
                while d < a:
                    d += 86400
                prev = d
                arr = f"{a // 3600:02d}:{a % 3600 // 60:02d}:00"
                dep = f"{d // 3600:02d}:{d % 3600 // 60:02d}:00"
            st.append((rid, arr, dep, f"n{n}", seq))
            used.add(n)
        for dd in running_dates(t["days"], int(re.sub(r"\D", "", t["num"]) or 0)):
            cal.append((rid, dd.strftime("%Y%m%d"), 1))

    def tbl(header, rows):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return b.getvalue()
    stops_rows = [(f"n{n}", stops[n][0].get("name"), f"{stops[n][2]:.6f}", f"{stops[n][1]:.6f}")
                  for n in sorted(used)]
    agencies = sorted({r[1] for r in routes})
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    zp = GTFS_DIR / "ir_iranrail.gtfs.zip"
    tmp = GTFS_DIR / "ir_iranrail.gtfs.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agency.txt", tbl(("agency_id", "agency_name", "agency_url", "agency_timezone"),
                                     [(a, a, "https://www.rai.ir/", "Asia/Tehran")
                                      for a in agencies]))
        z.writestr("stops.txt", tbl(("stop_id", "stop_name", "stop_lat", "stop_lon"), stops_rows))
        z.writestr("routes.txt", tbl(("route_id", "agency_id", "route_short_name",
                                      "route_long_name", "route_type"), routes))
        z.writestr("trips.txt", tbl(("route_id", "service_id", "trip_id"), trips))
        z.writestr("stop_times.txt", tbl(("trip_id", "arrival_time", "departure_time", "stop_id",
                                          "stop_sequence"), st))
        z.writestr("calendar_dates.txt", tbl(("service_id", "date", "exception_type"), cal))
        z.writestr("feed_info.txt", tbl(
            ("feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_version"),
            [("noritetsu, from iranrail.net's copy of RAI's timetable", IRANRAIL, "fa",
              date.today().isoformat())]))
    os.replace(tmp, zp)
    log(f"timetable: {len(trips)} trains, {len(used)} stations, {len(cal)} train-days -> {zp}")


def report(png=None, bbox=None):
    """The assignment, and with `png` a plot of it: each line a colour, main-line track no
    line took in black (left out on purpose: dotted grey)."""
    log = print
    load_osm(log)
    ways, coords, line_of, geo = _S["ways"], _S["coords"], _S["line_of"], _S["geo"]
    left = Counter()
    for w, (t, _n) in ways.items():
        if (t.get("railway") in TRACK_KIND and not t.get("service") and w not in line_of
                and t.get("usage") not in NOT_PASSENGER):
            n = tidy(t.get("name"))
            if not (n in NAME_LINE and NAME_LINE[n] is None):
                left[(round(geo[w][1], 1), round(geo[w][2], 1))] += geo[w][0]
    log(f"IR: main-line track in no line and not left out on purpose: "
        f"{sum(left.values()):,.0f} km; by 0.1 degree cell: "
        + ", ".join(f"{x},{y} {v:.0f}" for (x, y), v in left.most_common(40)))
    if not png:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    keys = list(LINES)
    fig, ax = plt.subplots(figsize=(30, 26))
    lab = {}
    for w, (t, nodes) in ways.items():
        if t.get("railway") not in TRACK_KIND or t.get("service"):
            continue
        km, lon, lat = geo[w]
        if bbox and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        pos = pos[ok]
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        k = line_of.get(w)
        if k is None:
            n = tidy(t.get("name"))
            out = n in NAME_LINE and NAME_LINE[n] is None
            ax.plot(x, y, color="grey" if out else "black", lw=0.8 if out else 1.6,
                    ls=":" if out else "-")
            continue
        ax.plot(x, y, color=cmap(keys.index(k) % 20), lw=2.2)
        lab.setdefault(k, (lon, lat))
    for k, p in lab.items():
        ax.annotate(k, p, fontsize=9, color="darkred")
    ax.set_aspect(1.2)
    fig.savefig(png, dpi=60, bbox_inches="tight")
    log(f"IR: plot {png}")


if __name__ == "__main__":
    if "--clip" in sys.argv:
        clip()
    elif "--construction" in sys.argv:
        construction_pass(ROOT / sys.argv[sys.argv.index("--construction") + 1])
    elif "--crawl" in sys.argv:
        crawl()
    elif "--parse" in sys.argv:
        parse()
    elif "--timetable" in sys.argv:
        timetable()
    elif "--report" in sys.argv:
        i = sys.argv.index("--report")
        png = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        bb = ([float(v) for v in sys.argv[i + 2].split(",")] if len(sys.argv) > i + 2 else None)
        report(png, bb)
    else:
        print(__doc__)
