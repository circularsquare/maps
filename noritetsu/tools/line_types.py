"""LINE TYPES: what kind of railway each line is, in words a rider uses (Anita, 2026-10-09,
"ok lets do line type for sure"). Method, type list and per-country rules: line_types.md.

    python tools/line_types.py                    # every country: write types.json, print the summary
    python tools/line_types.py jp de --sample 5   # some countries; print 5 typed lines per source
    python tools/line_types.py us --explain m10322749   # why one line got its type
    python tools/line_types.py --dry              # summary only, write nothing
    python tools/line_types.py --untyped de       # list the lines left without a type

Reads dist/data/<cc>/{lines,foot,stations}.json and data/proc/<cc>/rels.pkl; never writes
them. Writes dist/data/<cc>/types.json only:

    {"<line id>": "commuter", "<register line id>": {"type": "regional", "also": ["intercity"]}}

A plain string is a line of one kind. An object is a register line (track) that carries
several kinds of service: `type` is the main one, `also` the others in order of the share of
the line they cover. A line with nothing to go on is left out.

HOW A LINE GETS ITS TYPE (first rule that gives one wins; the source names are what the
summary counts):

  OSM lines (src "osm"; operating patterns and named trains)
    kind      metro / light rail / tram / monorail / funicular lines by `kind`; people movers
              and maglevs by name or service=people_mover, or (Japan) by the N02 guideway flag
              of the track they run on; an S-Bahn tagged route=light_rail (Berlin) is commuter.
    name      a train route named as a funicular, monorail, maglev, people mover or tram.
    pre       Japan and China only, before the tags: Japan's tags are sparse and mixed, so the
              rule decides (Shinkansen track, limited-express names, else commuter inside the
              big-city belts and regional outside); China by the train-number letter.
    tag       `service=` on the route_master, else the majority over its routes, normalised
              (SERVICE); a TGV/ICE/AVE-branded train tagged long_distance is high-speed.
              Post-Soviet `regional` (prigorodny trains) is commuter up to 150 km.
    rule      a country rule on network / operator / name / train number (RULES).
    track     an untagged train running TRACK_HS of its length on high-speed track (TER GV).
    name      words in the name, ref, network or operator (WORDS), the same in every country.
    default   still nothing: a named train is intercity, an operating pattern regional.

  Register lines (track)
    kind      an urban register line (N02 subway, MTR, LTA, the Korean and Taiwanese metros)
              by its kind; a national register's light_rail / tram line (its kind came from
              OSM's ways) takes the mainline services covering most of it instead.
    hs        HS_MAIN of the line's km is high-speed track (`highspeed_sections`, the N02
              Shinkansen flag) and high-speed trains run on it (China: the track alone).
    name      China's 高速线 / 客专 by name; REGISTER_OVERRIDE (Mumbai's suburban lines, India's
              mountain railways); a heritage society's line no service reaches.
    rule      Japan: commuter inside the big-city belts, regional outside.
    services  the OSM lines whose footprint (foot.json) lies on its sections: for each type,
              the share of the line's km at least one of them covers (a union, as
              tools/operators.py measures operators). See `mix`: on a line under LONG_LINE_KM
              the operating patterns decide once one covers R_DECIDES, else every service
              counts; within TIE of the top share the more local type wins (a commuter line
              Amtrak also uses stays commuter); night trains, and high-speed trains on track
              that is not high-speed, count as intercity for the main type.
    default   REGISTER_NAMED / REGISTER_DEFAULT, for a line with DEFAULT_MIN_KM running: track
              no OSM service reaches, or where services cover under THIN of it (the services
              seen go in `also`). Only where the country's sources say what runs.
    sibling   a US "(second track)" register line takes its subdivision's type.
"""
import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.stdout.reconfigure(encoding="utf-8")
TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
DATA = os.path.join(ROOT, "dist", "data")
PROC = os.path.join(ROOT, "data", "proc")
sys.path.insert(0, TOOLS)
from operators import Country, merge_spans, span_len, regions   # noqa: E402

# ------------------------------------------------------------------ the types

# Order = how local a type is: on a tie for a register line's main type, the earlier wins.
TYPES = ["metro", "light_rail", "tram", "people_mover", "monorail", "maglev", "funicular",
         "commuter", "regional", "intercity", "high_speed", "night", "tourist"]
# The app's labels (sentence case). Kept here so line_types.md and the app agree.
LABELS = {"high_speed": "High-speed rail", "intercity": "Intercity rail", "night": "Night train",
          "regional": "Regional rail", "commuter": "Commuter rail", "metro": "Metro",
          "light_rail": "Light rail", "tram": "Tram", "monorail": "Monorail",
          "people_mover": "People mover", "maglev": "Maglev", "funicular": "Funicular",
          "tourist": "Tourist railway"}
MAINLINE = {"commuter", "regional", "intercity", "high_speed", "night", "tourist"}
# Registers whose `kind` for urban-looking track came from OSM's way tags, not the operator.
NATIONAL_REGISTERS = {"rinf", "rfn", "narn", "schienennetz", "ga", "banenor", "gb", "getlink"}
URBAN_KIND = {"subway": "metro", "light_rail": "light_rail", "tram": "tram",
              "monorail": "monorail", "funicular": "funicular"}

SHARE_MIN = 0.15     # a type counts on a register line when its services cover this much of it
SHARE_FLOOR = 0.05   # with nothing reaching SHARE_MIN, the best one still needs this much
TIE = 0.05           # shares this close to the top one count as a tie
R_DECIDES = 0.5      # operating patterns alone decide the main type when one covers this much
TRACK_HS = 0.7       # an untagged train this much on high-speed track is high-speed (TER GV)
LONG_LINE_KM = 200   # a register line this long is a trunk line (see mix)
THIN = 0.5           # services covering less than this of a line give way to the country default
HS_TRACK_ALONE = {"cn"}
HS_NAME = {"cn": r"高速线|高速铁路|客运专线|客专"}
HS_MAIN = 0.5        # this much high-speed track makes a register line high-speed
SOVIET_COMMUTER_KM = 150   # a post-Soviet suburban train up to this long is commuter rail

# service= values (each part of a ";" list), normalised. None = no information (dropped).
SERVICE = {
    "high_speed": "high_speed", "highspeed": "high_speed",
    "long_distance": "intercity", "national": "intercity", "international": "intercity",
    "intercity": "intercity", "express": "intercity", "car_shuttle": "intercity", "car": "intercity",
    "night": "night", "motorail": "night",
    "regional": "regional", "local": "regional", "stopping": "regional", "ordinary": "regional",
    "rapid": "regional",
    "commuter": "commuter", "suburban": "commuter", "urban": "commuter",
    "tourism": "tourist", "tourist": "tourist", "touristic": "tourist", "heritage": "tourist",
    "people_mover": "people_mover",
    # light_rail on a route=train is Spain's and Italy's narrow-gauge lines (Euskotren, FEVE,
    # FCE): no information about the service, the name rules decide.
}
SERVICE_RANK = ["night", "high_speed", "intercity", "regional", "commuter", "tourist", "people_mover"]

POST_SOVIET = {"ru", "by", "ua", "kz", "uz", "kg", "tj", "tm", "ge", "am", "az", "md", "xa", "mn"}

# ------------------------------------------------------------------ words, any country

def rx(s):
    return re.compile(s, re.I)


# (type, pattern) in order; matched against name, name:en, ref, network and operator.
WORDS = [
    ("tourist", rx(r"heritage|museum|muzeum|museo|musée|nostalg|носталг|histori|dampf|steam|stoom|"
                   r"tourist|turisti|scenic|excursion|wine train|\bzoo\b|disney|six flags|busch gardens|"
                   r"amusement|attraction|kisvas[uú]t|erdei vas[uú]t|\bÁEV\b|múzeumvasút|ZSUZSI|szobi ev|"
                   r"waldeisenbahn|moorbahn|pferdebahn|parkeisenbahn|tiergartenbahn|kasbachtalbahn|"
                   r"windbergbahn|p.?tit train|petit train|train 1900|minièresbunn|grand canyon|"
                   r"glacier express|goldenpass|panoramique|montenvers|mont-blanc|baie de somme|livradois|"
                   r"дзіцячая чыгунка|детская железная|малая .*железная дорога|узкоколейка в|"
                   r"children'?s railway|perurail|rocky mountaineer|"
                   r"tiradentes|mantiqueira|maria fuma|west somerset|jungfraubahn|gornergrat|pilatus|"
                   r"rothorn|schynige|wengernalp|\bPE ?\d|railroad tour|fair train|cremallera|"
                   r"rack railway|zahnradbahn|crémaillère|turístico")),
    ("funicular", rx(r"funicul|funic\b|standseilbahn|\bFUN\b|ケーブル|斜坑|cable car|\bBergbahn\b|"
                     r"Merkurbergbahn|Nerobergbahn|Schlossbergbahn|Künzelsauer Bergbahn")),
    ("monorail", rx(r"monorail|monorel|モノレール|单轨|單軌")),
    ("maglev", rx(r"maglev|磁浮|磁悬浮|磁懸浮|リニモ|linimo|東部丘陵線|자기부상")),
    ("people_mover", rx(r"people ?mover|\bAPM\b|airtrain|skyline|skylink|新交通|ガイドウェイ")),
    ("tram", rx(r"^tram\b|\btram \d|\btram T\d|straßenbahn|трамвай|tranv[ií]a")),
    ("night", rx(r"nightjet|\bnight|nacht|\bnuit\b|notte|ночн|euronight|\bEN ?\d|\bNJ ?\d|sleeper|"
                 r"sunrise|サンライズ|couchette|lunea")),
    ("high_speed", rx(r"(?-i:\b(ICE|TGV|AVE|AVLO|OUIGO|KTX|SRT|YHT|LGV)\b)|\b(Frecciarossa|"
                      r"Frecciargento|Italo|Thalys|Eurostar|Sapsan|Allegro|Afrosiyob|Al Boraq|"
                      r"Whoosh|Acela|Lyria|Shinkansen)\b|high.?speed|yüksek hızlı|сапсан|新幹線|"
                      r"高铁|高鐵|고속")),
    ("airport", rx(r"airport|aéroport|aeroporto|aeropuerto|flughafen|lufthavn|bandara|\bKLIA\b|"
                   r"空港|机场|機場|공항|union pearson|flytoget|arlanda")),
    ("commuter", rx(r"\bS-?Bahn\b|^S ?\d+[a-z]?\b|\bRER\b|transilien|cercan[ií]as|rodalies|"
                    r"proastiakos|προαστιακ|commuter|suburban|suburbano|banliy|\bMEMU\b|komuter|"
                    r"\bKRL\b|metrorail|elektrichka|электрич|електрич|пригород|приміськ|\bэл\. |"
                    r"\bPKM ?\d*|\bSKM\b|\bSKA ?\d*|\bWKD\b|\bŁKA\b|aglomeracyj|\bHÉV\b|İZBAN|"
                    r"marmaray|AYBAN|\bMMTS\b|\blocal\b|metropolitan|metropolitana|trem metropolitano|"
                    r"BG:Воз|\bEsko\b|Abşeron|Қала маңы")),
    ("intercity", rx(r"\b(IC|ICN|EC|ECE|IR|IRN|IRE|RJ|RJX|railjet|intercity|inter-city|intercités|"
                     r"eurocity|express|expreso|expresso|ekspres|ekspresi|ekspress|экспресс|експрес|"
                     r"mail|rajdhani|shatabdi|duronto|superfast|sampark|garib rath|humsafar|"
                     r"vande bharat|tejas|amtrak|VIA Rail|long.?distance|fernverkehr|leo ?express|"
                     r"regiojet|talgo|alvia|euromed|intercidades|alfa pendular|poyezdi|"
                     r"пассажирский поезд|поїзд|поезд|цягнік|скорый|фирменный|brightline|"
                     r"al atlas)\b")),
    ("regional", rx(r"\b(RE|RB|RS|R|REX|RegionalExpress|Regionalbahn|regional|regionale|"
                    r"regionaltog|regio|TER|bölgesel|osobní|osobný|személy|Sz|passenger|пассажир|"
                    r"пасажир|mixed|personenzug|stopping|ordinary|Os|Sp|R ?\d+|RE ?\d+|RB ?\d+|"
                    r"D\d+|U\d+|T\d+|P\d+|K\d+)\b")),
]


AIRPORT_KM = 100     # an airport train is commuter rail; a long route that calls at one is not


def by_words(text, km=0):
    for t, pat in WORDS:
        if pat.search(text):
            if t == "airport":
                if km > AIRPORT_KM:
                    continue
                return "commuter"
            return t
    return None


def word_is(t, text):
    return any(pat.search(text) for tt, pat in WORDS if tt == t)


def train_number(l):
    """A train number: the first number of the ref, else one written as a train number in the
    name ("№ 29", "Train 6504", "поезд 701"), else a free-standing 3-4 digit number. Not the
    2 of "Вохтога-2"."""
    m = re.search(r"\d+", l.get("ref") or "")
    if m:
        return int(m.group())
    n = l.get("name") or ""
    m = re.search(r"(?:№|No\.?|#|поезд|поїзд|потяг|цягнік|train)\s*(\d+)", n, re.I) or \
        re.search(r"(?<![\w-])(\d{3,4})(?![\w-])", n)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ country rules

def has(pat, text):
    return re.search(pat, text, re.I) is not None


def soviet(l, x):
    """The former Soviet railways' train numbers: 1-599 long-distance, 600-699 local, 700-799
    fast day trains, 800-899 fast regional, 6000-7999 suburban (prigorodny)."""
    t = x["text"]
    num = train_number(l)
    if has(r"пригород|электрич|електрич|приміськ|\bэл\. |Қала маңы|elektr|dairəvi|Abşeron", t) or \
            (num is not None and 6000 <= num <= 7999):
        return "commuter" if l["km"] <= SOVIET_COMMUTER_KM else "regional"
    if num is not None and num < 600:
        return "intercity"
    if num is not None and 600 <= num < 700:
        return "regional"
    if num is not None and 700 <= num < 800:
        return "intercity"
    if num is not None and 800 <= num < 900:
        return "intercity" if l["km"] > SOVIET_COMMUTER_KM else "regional"
    if l.get("service"):
        return "intercity"
    return None


US_COMMUTER = (r"\b(LIRR|Long Island Rail|Metro-North|NJ Transit|SEPTA|MARC|Virginia Railway Express|"
               r"VRE|Metra|NICTD|South Shore|Metrolink|Caltrain|ACE|Altamont|SMART|Sounder|Tri-Rail|"
               r"SunRail|UTA|FrontRunner|NCTD|COASTER|TRE|Trinity Railway|Rio Metro|Rail Runner|RTD|"
               r"CTrail|CCRTA|CapeFlyer|MetroRail|Capital Metro|DCTA|A-Train|Trinity Metro|TEXRail|"
               r"Westside Express|WES|RTAMT|WeGo|MBTA|Northstar|Hartford Line|Shore Line East|DART|"
               r"Silver Line|Keolis)\b")


def r_us(l, x):
    t = x["text"]
    if has(r"\bAcela\b", t):
        return "high_speed"
    if has(r"Amtrak|Brightline|\bARR\b|Alaska Railroad", t) and not has(r"Fair Train", t):
        return "intercity"
    if has(US_COMMUTER, t):
        return "commuter"
    # Everything else in the US untagged list is an excursion, a heritage line or a park railway.
    return "tourist" if l["kind"] == "train" else None


def r_ca(l, x):
    t = x["text"]
    if has(r"VIA Rail|Amtrak", t):
        return "intercity"
    if has(r"Rocky Mountaineer|Fort Edmonton|Steam", t):
        return "tourist"
    if has(r"GO Transit|\bexo\d?\b|TransLink|West Coast Express|UP Express", t):
        return "commuter"
    if has(r"Ontario Northland|Keewatin|Tsal'alh|Polar Bear", t):
        return "regional"
    return None


def r_id(l, x):
    t = x["text"]
    if has(r"Whoosh|Kereta Cepat|high speed", t):
        return "high_speed"
    if has(r"KAI Commuter|KAI Bandara|Commuter Line|\bLin\b", t):
        return "commuter"
    if has(r"\bKAI\b|Kereta Api", t):
        return "intercity" if l.get("service") else "regional"
    return None


def r_in(l, x):
    t = x["text"]
    if has(r"MMTS|Suburban|\bLocal\b", t):
        return "commuter"
    if has(r"Passenger|MEMU|DEMU|\bPass\b", t):
        return "regional"
    if l.get("service"):
        return "intercity"
    return None


def r_tr(l, x):
    t = x["text"]
    if has(r"Turistik", t):
        return "tourist"
    if has(r"YHT|Yüksek Hızlı", t):
        return "high_speed"
    if has(r"Ekspres|Mavi Tren", t):
        return "intercity"
    if has(r"Banliyö|İZBAN|AYBAN|Marmaray|\bT6\b", t):
        return "commuter"
    if has(r"Bölgesel", t):
        return "regional"
    return None


def r_rs(l, x):
    t = x["text"]
    if has(r"Соко|Soko", t):
        return "high_speed"
    if has(r"БГ:Воз", t):
        return "commuter"
    if has(r"Носталгија", t):
        return "tourist"
    if has(r"^IR\b", l["name"]) or has(r"^IR\b", l["ref"]):
        return "intercity"
    return "regional" if l["kind"] == "train" else None


def r_kr(l, x):
    t = x["text"]
    if has(r"KTX|SRT|고속", t):
        return "high_speed"
    if has(r"ITX|새마을|무궁화|누리로|코레일|한국철도", t):
        return "intercity"
    return None


def r_tw(l, x):
    t = x["text"]
    if has(r"高鐵|THSR", t):
        return "high_speed"
    if has(r"自強|太魯閣|普悠瑪|莒光|新自強", t):
        return "intercity"
    if has(r"區間|內灣|六家|集集|平溪|深澳", t):
        return "regional"
    return None


def r_my(l, x):
    t = x["text"]
    if has(r"\bETS\b", t):
        return "intercity"
    if has(r"Komuter|KLIA|\bERL\b", t):
        return "commuter"
    return None


def r_th(l, x):
    t = x["text"]
    if has(r"\bARL\b|สายสีแดง|Red Line|รถไฟฟ้า|AERA1", t):
        return "commuter"
    if has(r"เร็ว|ด่วน|Rapid|Express", t):
        return "intercity"
    return "regional" if l["kind"] == "train" else None


def r_vn(l, x):
    t = x["text"]
    if has(r"\b(HP|LP)\d", t):
        return "regional"
    if has(r"\b(SE|TN|SNT|SPT|SQN|SH|SN|NA)\d", t):
        return "intercity"
    return None


def r_kp(l, x):
    t = x["text"]
    if has(r"^K\d|K27|국제|international", l["name"] + " " + l["ref"]):
        return "intercity"
    return "regional" if l["kind"] == "train" else None


def r_lv(l, x):
    t = x["text"]
    if has(r"LTG Link|Vilnius", t):
        return "intercity"
    if l["kind"] == "train":
        return "commuter" if l["km"] <= 100 else "regional"
    return None


def r_lt(l, x):
    return "intercity" if has(r"LTG Link.*|Vilnius.*Rīga|Rīga.*Vilnius", x["text"]) else None


def r_es(l, x):
    t = x["text"]
    if has(r"^C-?\d|\bC-?\d+[a-z]?\b|Cercan|^E\d|Euskotren|Rodalies|\bR\d+\b.*Rodalies", l["ref"] + " " + l["name"] + " " + l["network"]):
        return "commuter"
    if has(r"\bSFM\b|Serveis Ferroviaris de Mallorca", t):
        return "commuter"
    return None


def r_it(l, x):
    t = x["text"]
    if has(r"Avellino.*Rocchetta", t):
        return "tourist"
    if has(r"\bFL\d|Metropolitana", t):
        return "commuter"
    return None


def r_cz(l, x):
    t = x["text"]
    if has(r"^S\d+", l["ref"]):
        return "commuter" if has(r"Pražská integrovaná", l["network"]) else "regional"
    return None


def r_hu(l, x):
    if has(r"^[SGZ]\d+$", l["ref"]):
        return "commuter"
    return None


def r_gb(l, x):
    t = x["text"]
    if has(r"Merseyrail|London Overground|Elizabeth line|Thameslink", t):
        return "commuter"
    if has(r"Caledonian Sleeper|Night Riviera", t):
        return "night"
    if has(r"CrossCountry|Cross Country|TransPennine|LNER|London North Eastern|Avanti|Grand Central|"
           r"Hull Trains|Lumo|Eurostar", t):
        return "intercity"
    if has(r"Great Western|GWR", t) and has(r"Paddington", t) and l["km"] >= 150:
        return "intercity"
    return None


def r_ch(l, x):
    t = x["text"]
    if has(r"Jungfraubahnen", l["network"]) or has(r"^CC ?\d", l["ref"]) or has(r"\(CC\)", l["name"]):
        return "tourist"        # the Jungfrau railways and the cogwheel (CC) lines
    return None


def r_no(l, x):
    return "commuter" if has(r"^L\d+$", l["ref"]) else None


def r_se(l, x):
    return "commuter" if has(r"^SL$|Roslagsbanan|Saltsjöbanan", l["network"] + " " + l["name"]) else None


def r_au(l, x):
    return "commuter" if has(r"Transperth|Translink|Sydney Trains|Metro Trains|Adelaide Metro", x["text"]) and l["kind"] == "train" else None


def r_ma(l, x):
    t = x["text"]
    if has(r"\bTNR\b", t):
        return "regional"
    if has(r"AL ATLAS", t):
        return "intercity"
    return None


def r_dz(l, x):
    return "commuter" if has(r"SNTF Alger|banlieue", x["text"]) else None


def r_eg(l, x):
    return "commuter" if has(r"^LRT$", l["ref"]) else None


def r_ir(l, x):
    return "commuter" if has(r"شهری|حومه|Commuterrail", x["text"]) else None


def r_lk(l, x):
    return "regional" if has(r"Kelani Valley", x["text"]) else None


def r_bo(l, x):
    return "regional" if has(r"Buscarril", x["text"]) else None


RULES = {
    "us": r_us, "ca": r_ca, "id": r_id, "in": r_in, "tr": r_tr, "rs": r_rs, "kr": r_kr,
    "tw": r_tw, "my": r_my, "th": r_th, "vn": r_vn, "kp": r_kp, "lv": r_lv, "lt": r_lt,
    "es": r_es, "it": r_it, "cz": r_cz, "hu": r_hu, "gb": r_gb, "ch": r_ch, "no": r_no,
    "se": r_se, "au": r_au, "ma": r_ma, "dz": r_dz, "eg": r_eg, "ir": r_ir, "lk": r_lk,
    "bo": r_bo,
}
for _cc in POST_SOVIET:
    RULES.setdefault(_cc, soviet)


# --- Japan

# The commuter belts: (lat, lon, radius km). Tokyo reaches Yokohama, Chiba and Omiya; Osaka
# reaches Kyoto, Kobe and Nara; then Nagoya, Fukuoka, Sapporo, Sendai and Hiroshima.
JP_BELTS = [(35.681, 139.767, 50), (34.702, 135.496, 45), (35.171, 136.882, 30),
            (33.590, 130.421, 25), (43.069, 141.351, 25), (38.260, 140.882, 20),
            (34.398, 132.475, 20)]
JP_LTD = (r"特急|ロマンスカー|スカイライナー|ラピート|μSKY|ミュースカイ|ひのとり|しまかぜ|"
          r"アーバンライナー|Limited Express|成田エクスプレス|はるか")
JP_RAPIDISH = r"快速|普通|各停|急行|準急|ライナー|Stopping|Liner|Rapid|Local|直通特急|通勤特急|快速特急|区間特急"


def km_between(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def belt_share(l, st):
    tot = inb = 0.0
    for s in l["sections"]:
        a, b = st.get(s[0]), st.get(s[1])
        if not a or not b:
            continue
        y, x = (a[1] + b[1]) / 2, (a[0] + b[0]) / 2
        tot += s[2]
        if any(km_between(y, x, la, lo) <= r for la, lo, r in JP_BELTS):
            inb += s[2]
    return inb / tot if tot else 0.0


def jp_local(l, ctx):
    return "commuter" if belt_share(l, ctx.st) >= 0.5 else "regional"


def pre_jp(l, x, ctx):
    if l["kind"] != "train":
        return None
    if x["hs"] >= 0.5:
        return "high_speed"
    t = l["name"] + " " + l["ref"] + " " + l["network"]
    if has(JP_RAPIDISH, t):
        return jp_local(l, ctx)
    if has(JP_LTD, t):
        return "intercity"
    if l.get("service"):
        jr = not l["operator"] or has(r"旅客鉄道|^JR|JR ", l["operator"])
        # A route written out ("徳山 - 下関", "普通 郡山<=>福島") or a bare letter ("N") is not a
        # train's name; 宗谷, 北斗 and 南風 are.
        routeish = has(r"[-=<>→⇔↔]|\d", l["name"]) or not l["name"] or \
            re.fullmatch(r"[A-Za-z]{1,2}", l["name"]) is not None
        if jr and not routeish:
            return "intercity"
    return jp_local(l, ctx)


def pre_cn(l, x, ctx):
    if l["kind"] != "train":
        return None
    t = x["text"]
    if has(r"市郊|市域|地铁|轨道交通", t):
        return "commuter"
    if has(r"旅游", t):
        return "tourist"
    m = re.match(r"\s*([GDCKTZSYL]?)(\d{1,4})", l["ref"] or l["name"])
    if m:
        letter, n = m.group(1), int(m.group(2))
        if letter in "GDC" and letter:
            return "high_speed"
        if letter in ("K", "T", "Z", "L"):
            return "intercity"
        if letter == "S":
            return "commuter"
        if letter == "Y":
            return "tourist"
        if 1001 <= n <= 5998:
            return "intercity"      # 普快, ordinary fast trains
        return "regional"           # 普客 6001-7598 and the 8xxx local trains
    if has(r"动车|高铁", t):
        return "high_speed"
    if has(r"城际|快线", t):
        return "regional"
    return None


PRE = {"jp": pre_jp, "cn": pre_cn}

# ------------------------------------------------------------------ register defaults

# Track no OSM service reaches: the type of the trains that run there, per country, where the
# country's sources file says what runs (line_types.md has the reasons). Only for a line with
# at least DEFAULT_MIN_KM running (not `closed`): junction curves and yard links stay untyped.
# First the named lines (cc, pattern on name and operator, type), then the country's default:
# a type, or LONG (intercity at LONG_KM and over, else regional: on a national network with one
# operator and few trains, the long lines carry the long-distance trains and the short
# branches local ones).
DEFAULT_MIN_KM = 5
LONG = "long"
LONG_KM = 250
REGISTER_NAMED = [
    ("pt", r"Cascais", "commuter"),
    ("au", r"Tarcoola Darwin|Darwin Line|Ghan", "intercity"),
    ("au", r"Gawler|Seaford|Noarlunga|Outer Harbor|Belair|Caboolture|Sunshine Coast|Ipswich|"
           r"Rosewood|Belgrave|Lilydale|Flemington|Showgrounds", "commuter"),
    ("au", r"Puffing Billy", "tourist"),
    ("ar", r"^Línea|Metropolitano", "commuter"),
    ("br", r"Serra Verde", "tourist"),
    ("cl", r"Limache|Biotren", "commuter"),
    ("mx", r"Tren Maya|Istmo|Tehuantepec", "intercity"),
    ("mx", r"Insurgente", "commuter"),
    ("my", r"Pantai Timur", "intercity"),
    ("my", r"Skypark", "commuter"),
    ("vn", r"Bắc Nam|Lào Cai", "intercity"),
    ("vn", r"Đà Lạt", "tourist"),
    ("za", r"PRASA", "commuter"),
    ("za", r"Transnet", "intercity"),
    ("ke", r"SGR|Kisumu", "intercity"),
    ("ke", r"Miritini|Syokimau", "commuter"),
    ("ng", r"Iddo|Ijoko|Red Line", "commuter"),
    ("ng", r"Port Harcourt", "regional"),
    ("gh", r"Accra – Tema", "commuter"),
    ("cd", r"Train urbain", "commuter"),
    ("mg", r"Train urbain", "commuter"),
    ("ao", r"Bungo|Lobito – Benguela", "commuter"),
    ("ao", r"Dondo", "regional"),
    ("dz", r"Aéroport", "commuter"),
    ("tn", r"Métro du Sahel", "commuter"),
    ("il", r"תל אביב–ירושלים", "intercity"),
    ("iq", r"البصرة", "intercity"),
    ("co", r"Turístico", "tourist"),
    ("ec", r"Nariz del Diablo", "tourist"),
    ("th", r"แม่กลอง|น้ำตก", "regional"),
    ("tr", r"Başkentray|Gaziray|Marmaray|İZBAN", "commuter"),
    ("tr", r"Tiflis", "intercity"),
    ("tw", r"阿里山", "tourist"),
    ("tw", r"沙崙", "commuter"),
    ("nz", r"Johnsonville", "commuter"),
]
# Register lines whose name says what they are, before any service: OSM's routes cover them
# only in part (Mumbai's suburban "Western Line" runs to Dahanu, the OSM route to Virar).
REGISTER_OVERRIDE = [
    ("in", r"^(Western|Central|Harbour|Trans-Harbour|Port|Uran) [Ll]ine$|Suburban|\bMRTS\b", "commuter"),
    # The Mountain Railways of India: scheduled, but ridden as heritage trains.
    ("in", r"Darjeeling Himalayan|Nilgiri Mountain|Kalka–Shimla|Matheran", "tourist"),
]
REGISTER_DEFAULT = {
    # China Railway's conventional lines carry numbered long-distance trains (cn_sources.md).
    "cn": "intercity",
    # Every running line in these carries the stopping service; the long-distance trains are
    # mapped where they run, so what no mapped service reaches is regional. Germany is left
    # out: its unmapped register lines are freight bypasses and curves (1280, 1750, 5230).
    **{cc: "regional" for cc in ("it", "cz", "ro", "es", "si", "pl", "gb", "ua", "by", "bg",
                                  "hr", "gr", "al", "ba", "me", "mk", "xk", "md", "ge", "am",
                                  "az", "kg", "tj", "se", "lt", "lv", "ee", "lu", "ie", "no",
                                  "sk", "hu", "at", "be", "ch", "nl", "ru", "fr", "ar", "cl",
                                  "au", "my", "vn", "gh", "mg", "mw", "mz", "np", "uy", "pe",
                                  "ph", "il", "rs")},
    # Long-distance trains on most lines (fi: VR's IC and regional trains; in: NTES express and
    # passenger trains on every line, in_sources.md; pk: a named express on every running line,
    # pk_sources.md; bd: "every intercity and commuter corridor", bd_sources.md).
    "fi": "intercity", "in": "intercity", "pk": "intercity",
    # Korail's conventional lines carry Mugunghwa and ITX trains; on Canada's freight
    # subdivisions the passenger trains are VIA's and Amtrak's (the commuter lines are mapped).
    "kr": "intercity", "ca": "intercity",
    "tr": "regional", "dk": "regional", "nz": "regional", "tw": "regional",
    "ir": LONG, "id": LONG,
    "bd": LONG, "lk": LONG, "kp": LONG, "kz": LONG, "uz": LONG, "tm": LONG, "mm": LONG,
    "mn": LONG, "eg": LONG, "dz": LONG, "tn": LONG, "tz": LONG, "zm": LONG, "zw": LONG,
    "iq": LONG, "th": LONG,
    # One long-distance service on the country's line(s).
    **{cc: "intercity" for cc in ("ae", "bf", "cg", "cm", "dj", "et", "ga", "kh", "sa", "cd",
                                   "ao", "ng")},
    # City and airport lines.
    **{cc: "commuter" for cc in ("br", "cr", "pa", "sn", "ug", "ve")},
    "jo": "tourist",        # public excursions only (jo_sources.md)
    "co": "tourist",
    "ec": "tourist",
    "ke": "regional",
    "mx": "regional",       # El Chepe Regional
}


def register_default(cc, l):
    shut = set(l.get("closed") or [])
    running = sum(s[2] for s in l["sections"] if f"{s[0]}|{s[1]}" not in shut)
    if running < DEFAULT_MIN_KM:
        return None
    text = " ".join(filter(None, [l.get("name"), l.get("name_en"), l.get("operator")]))
    for c, pat, t in REGISTER_NAMED:
        if c == cc and re.search(pat, text):
            return t
    d = REGISTER_DEFAULT.get(cc)
    if d == LONG:
        return "intercity" if l["km"] >= LONG_KM else "regional"
    return d

# ------------------------------------------------------------------ one country


class Ctx:
    pass


def norm_service(v):
    """A service= value (maybe "a;b") -> type or None. Night wins in a list (a sleeper is also
    long-distance); else the first part that means something."""
    ps = [p.strip().lower() for p in (v or "").split(";") if p.strip()]
    ts = [SERVICE.get(p) for p in ps]
    if "night" in ts:
        return "night"
    return next((t for t in ts if t), None)


def load(cc):
    ctx = Ctx()
    ctx.cc = cc
    ctx.C = Country(cc)
    ctx.lines = ctx.C.lines
    ctx.by_id = {l["id"]: l for l in ctx.lines}
    p = os.path.join(PROC, cc, "rels.pkl")
    ctx.rels = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    sp = os.path.join(DATA, cc, "stations.json")
    st = json.load(open(sp, encoding="utf-8"))["stations"] if os.path.exists(sp) else {}
    ctx.st = {k: (v.get("x"), v.get("y")) for k, v in st.items() if v.get("x") is not None}
    # routes not under a master, by the key build_model groups them on
    claimed = set()
    for rid, v in ctx.rels.items():
        t, m = v
        if t.get("type") == "route_master":
            claimed.update(r for ty, r, _ in m if ty == "r")
    ctx.by_key = defaultdict(list)
    for rid, (t, m) in ctx.rels.items():
        if t.get("type") == "route" and rid not in claimed:
            ctx.by_key[(t.get("operator", ""), t.get("network", ""), t.get("ref", ""),
                        t.get("name", ""))].append(rid)
    # high-speed and guideway track, by section gid
    ctx.hs_gid, ctx.guided_gid = set(), set()
    for l in ctx.lines:
        if l.get("src") == "osm":
            continue
        hs = l.get("highspeed_sections") or {}
        for s in l["sections"]:
            if l.get("highspeed") or hs.get(f"{s[0]}|{s[1]}"):
                ctx.hs_gid.add(s[3])
            if l.get("guided") and l["kind"] != "subway":
                ctx.guided_gid.add(s[3])
    return ctx


def route_tags(l, ctx):
    """(tags of the line's relation, [tags of its routes])."""
    try:
        num = int(l["id"][1:])
    except ValueError:
        return {}, []
    t, m = ctx.rels.get(num, ({}, []))
    if l["id"][0] == "m":
        kids = [ctx.rels[r][0] for ty, r, _ in m if ty == "r" and r in ctx.rels]
    else:
        key = (t.get("operator", ""), t.get("network", ""), t.get("ref", ""), t.get("name", ""))
        kids = [ctx.rels[r][0] for r in ctx.by_key.get(key, [num]) if r in ctx.rels] or [t]
    return t, kids


def tagged_service(l, ctx):
    t, kids = route_tags(l, ctx)
    v = norm_service(t.get("service"))
    if v:
        return v
    votes = Counter(norm_service(k.get("service")) for k in kids)
    votes.pop(None, None)
    if not votes:
        # A network written as a local service word (gb's "regional", "local"). Not "national":
        # in Britain that is National Rail, any train.
        n = (l.get("network") or "").strip().lower()
        return norm_service(n) if n in ("regional", "local", "commuter", "suburban") else None
    top = max(votes.values())
    best = [v for v, n in votes.items() if n == top]
    return min(best, key=lambda v: SERVICE_RANK.index(v) if v in SERVICE_RANK else 99)


def footprint_share(l, ctx, gids):
    """Share of an OSM line's km lying on register sections in `gids`."""
    if not gids:
        return 0.0
    tot = on = 0.0
    for s in l["sections"]:
        tot += s[2]
        for t, fr, to, a, b in ctx.C.foot_of(s[3]):
            if t in gids:
                on += s[2] * abs(b - a)
    return on / tot if tot else 0.0


def type_osm(l, ctx):
    """(type, source) for an OSM line."""
    cc, kind = ctx.cc, l["kind"]
    # Country rules read the operator too; the word rules do not (Shanghai Metro 16 and 18 are
    # run by the maglev company).
    words = " ".join(filter(None, [l.get("name"), l.get("name_en"), l.get("ref"), l.get("network")]))
    x = {"text": " ".join(filter(None, [words, l.get("operator"), l.get("operator_en")])),
         "hs": footprint_share(l, ctx, ctx.hs_gid)}
    if kind in URBAN_KIND:
        svc = tagged_service(l, ctx)
        if svc == "people_mover":
            return "people_mover", "tag"
        for t in ("maglev", "people_mover", "monorail", "funicular"):
            if word_is(t, words):
                return t, "name"      # 清远长隆磁浮旅游专线 is a maglev before it is a tourist line
        # Berlin's and Copenhagen's S-Bahn are route=light_rail in OSM: commuter rail to riders.
        if kind == "light_rail" and has(r"S-Bahn|S-tog|\bS-train", x["text"]):
            return "commuter", "name"
        if kind in ("light_rail", "monorail") and footprint_share(l, ctx, ctx.guided_gid) >= 0.5:
            return "people_mover", "kind"
        return URBAN_KIND[kind], "kind"
    w = by_words(words)
    if w in ("funicular", "monorail", "maglev", "people_mover", "tram"):
        return w, "name"
    if cc in PRE:
        t = PRE[cc](l, x, ctx)
        if t:
            return t, "pre"
    svc = tagged_service(l, ctx)
    if svc:
        if cc in POST_SOVIET and svc == "regional" and l["km"] <= SOVIET_COMMUTER_KM:
            svc = "commuter"
        # TGV, ICE, AVE, KTX and the like are often tagged long_distance: the brand decides.
        if svc == "intercity" and by_words(words) == "high_speed":
            svc = "high_speed"
        return svc, "tag"
    if cc in RULES:
        t = RULES[cc](l, x)
        if t:
            return t, "rule"
    if x["hs"] >= TRACK_HS:
        return "high_speed", "track"
    w = by_words(x["text"], l["km"])
    if w:
        return w, "name"
    return ("intercity" if l.get("service") else "regional"), "default"


def coverage(ctx, osm_types):
    """Register line id -> {(tier, type): share of its km}."""
    C = ctx.C
    cover = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for l in ctx.lines:
        if l.get("src") != "osm" or l["id"] not in osm_types:
            continue
        typ = osm_types[l["id"]]
        tier = "N" if l.get("service") else "R"
        for s in l["sections"]:
            for t, fr, to, a, b in C.foot_of(s[3]):
                te = C.sec.get(t)
                if not te or te[0].get("src") == "osm" or fr == to:
                    continue
                cover[te[0]["id"]][(tier, typ)][t].append((min(fr, to), max(fr, to)))
    out = {}
    for l in ctx.lines:
        if l.get("src") == "osm":
            continue
        km = sum(s[2] for s in l["sections"]) or 1e-9
        per = {}
        for k, byt in cover.get(l["id"], {}).items():
            per[k] = min(1.0, sum(C.sec[t][1] * span_len(merge_spans(v, C.sec[t][1]))
                                  for t, v in byt.items()) / km)
        out[l["id"]] = per
    return out


def mix(cov, km, hs_track=False):
    """(main, also, top share) from a register line's coverage, or (None, [], 0). A night
    train is an intercity train as far as the track goes (the Southwest Chief alone on the
    Gallup Subdivision makes it intercity, not a night line), and so is a high-speed train on
    track that is not high-speed (KTX over the conventional 경부선): such a line's main type is
    intercity, with high-speed in `also`."""
    R, N = defaultdict(float), defaultdict(float)
    full = defaultdict(float)
    for (tier, t), s in cov.items():
        t = "intercity" if t == "night" else t
        full[t] = max(full[t], s)
        if t == "high_speed" and not hs_track:
            t = "intercity"
        d = R if tier == "R" else N
        d[t] = max(d[t], s)
    best = {t: max(R.get(t, 0), N.get(t, 0)) for t in set(R) | set(N)}
    # On a line under LONG_LINE_KM the operating patterns decide once one covers most of it
    # (one sleeper a night does not make a branch line intercity; Mumbai's Western Line stays
    # commuter). On a long trunk line the trains that run its length decide (Taishet - Lena,
    # where elektrichki cover half the 731 km, is an intercity line).
    pool = R if km < LONG_LINE_KM and R and max(R.values()) >= R_DECIDES else best
    if not pool:
        return None, [], 0
    top = max(pool.values())
    if top < SHARE_FLOOR:
        return None, [], 0
    tied = [t for t, s in pool.items() if s >= top - TIE]
    main = min(tied, key=TYPES.index)
    also = sorted((t for t, s in full.items() if t != main and s >= SHARE_MIN),
                  key=lambda t: (-full[t], TYPES.index(t)))
    return main, also, top


def type_register(l, ctx, cov):
    """(type, also, source) for a register line."""
    cc, kind = ctx.cc, l["kind"]
    km = sum(s[2] for s in l["sections"]) or 1e-9
    hs = sum(s[2] for s in l["sections"] if s[3] in ctx.hs_gid) / km
    main, also, top = mix(cov, km, hs_track=hs >= HS_MAIN)
    w = by_words(" ".join(filter(None, [l.get("name"), l.get("name_en")])))
    if kind in URBAN_KIND:
        if cc == "jp" and l.get("guided") and kind != "subway":
            t = "maglev" if w == "maglev" else "people_mover"
            return t, [a for a in also if a != t], "kind"
        if w in ("maglev", "people_mover"):
            return w, [a for a in also if a != w], "name"
        # A national register's kind comes from the OSM ways (Berlin's S-Bahn track is
        # light_rail): mainline services over most of it decide. An urban register's kind
        # (N02, MTR, LTA...) is the operator's own and stands.
        urban_cov = max((s for (tier, t), s in cov.items() if t not in MAINLINE), default=0)
        if l.get("src") in NATIONAL_REGISTERS and main in MAINLINE and \
                cov.get(("R", main), 0) >= 0.5 and urban_cov < 0.5:
            return main, also, "services"
        t = URBAN_KIND[kind]
        return t, [a for a in also if a != t], "kind"
    # High-speed track with high-speed trains on it (the brief's rule). Britain's 125 mph main
    # lines are tagged highspeed=yes but carry intercity trains: they stay intercity.
    # China's G and D trains are rarely mapped: there the track alone decides.
    fast = max((s for (tier, t), s in cov.items() if t == "high_speed"), default=0)
    if hs >= HS_MAIN and (fast >= SHARE_MIN or not cov or cc in HS_TRACK_ALONE):
        return "high_speed", [a for a in ([main] if main else []) + also if a != "high_speed"], "hs"
    if cc in HS_NAME and re.search(HS_NAME[cc], l.get("name") or ""):
        return "high_speed", [a for a in ([main] if main else []) + also if a != "high_speed"], "name"
    if cc == "jp":
        t = jp_local(l, ctx)
        return t, [a for a in ([main] if main else []) + also if a != t], "rule"
    for c, pat, t in REGISTER_OVERRIDE:
        if c == cc and re.search(pat, l.get("name") or ""):
            return t, [a for a in ([main] if main else []) + also if a != t], "name"
    if by_words(" ".join(filter(None, [l.get("name"), l.get("name_en"), l.get("operator")]))) == "tourist" \
            and not main:
        return "tourist", [], "name"        # a heritage society's line (Dampfbahn-Verein Zürcher Oberland)
    d = register_default(cc, l)
    if main and (top >= THIN or not d):
        return main, also, "services"
    if d:
        # Services reach only a little of the line (one Beijing S-line over a fifth of 京通线):
        # the country's default is the main type, what was seen goes in `also`.
        return d, [a for a in ([main] if main else []) + also if a != d], "default"
    return None, [], "none"


def siblings(ctx, out, src):
    """A register line no service reaches takes the type of the line it is a second track or
    a piece of: the same name once "(second track)" and the like are dropped, same operator
    (the US's NARN subdivisions)."""
    base = lambda n: re.sub(r"\s*\((second|third|fourth) track\)\s*$", "", n or "", flags=re.I).strip()
    typed = defaultdict(list)
    for l in ctx.lines:
        if l.get("src") != "osm" and l["id"] in out and src.get(l["id"]) != "sibling":
            typed[(base(l["name"]), l.get("operator"))].append(l)
    for l in ctx.lines:
        if l.get("src") == "osm" or l["id"] in out:
            continue
        sib = typed.get((base(l["name"]), l.get("operator")))
        if not sib:
            continue
        v = out[max(sib, key=lambda s: s["km"])["id"]]
        out[l["id"]] = v if isinstance(v, str) else v["type"]
        src[l["id"]] = "sibling"


def run_country(cc):
    ctx = load(cc)
    osm_types, src = {}, {}
    for l in ctx.lines:
        if l.get("src") == "osm":
            t, s = type_osm(l, ctx)
            osm_types[l["id"]] = t
            src[l["id"]] = s
    cov = coverage(ctx, osm_types)
    out = {}
    for l in ctx.lines:
        if l.get("src") == "osm":
            out[l["id"]] = osm_types[l["id"]]
            continue
        t, also, s = type_register(l, ctx, cov.get(l["id"], {}))
        src[l["id"]] = s
        if not t:
            continue
        out[l["id"]] = {"type": t, "also": also[:3]} if also else t
    siblings(ctx, out, src)
    return ctx, out, src, cov


# ------------------------------------------------------------------ main

def summary_row(cc, ctx, out, src):
    reg = [l for l in ctx.lines if l.get("src") != "osm"]
    osm = [l for l in ctx.lines if l.get("src") == "osm"]
    free = sum(1 for l in ctx.lines if src.get(l["id"]) in ("kind", "tag")
               or (l.get("src") != "osm" and src.get(l["id"]) == "hs"))
    c_osm = Counter(src[l["id"]] for l in osm)
    c_reg = Counter(src[l["id"]] for l in reg)
    return {"cc": cc, "lines": len(ctx.lines), "typed": len(out), "free": free,
            "reg": len(reg), "reg_typed": sum(1 for l in reg if l["id"] in out),
            "osm": len(osm), "osm_src": dict(c_osm), "reg_src": dict(c_reg),
            "types": dict(Counter((v if isinstance(v, str) else v["type"]) for v in out.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cc", nargs="*")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--explain")
    ap.add_argument("--untyped", action="store_true")
    ap.add_argument("--json", help="write the summary rows here")
    a = ap.parse_args()
    ccs = a.cc or [cc for cc in regions() if os.path.exists(os.path.join(DATA, cc, "lines.json"))]
    rows = []
    for cc in ccs:
        ctx, out, src, cov = run_country(cc)
        if not a.dry and not a.explain:
            p = os.path.join(DATA, cc, "types.json")
            json.dump(out, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False,
                      separators=(",", ":"))
            os.replace(p + ".tmp", p)
        r = summary_row(cc, ctx, out, src)
        rows.append(r)
        print(f"{cc}: {r['typed']}/{r['lines']} typed (free from tags and kind {r['free']}); "
              f"register {r['reg_typed']}/{r['reg']} {r['reg_src']}; osm {r['osm']} {r['osm_src']}",
              flush=True)
        if a.explain and a.explain in ctx.by_id:
            l = ctx.by_id[a.explain]
            print(json.dumps({k: l.get(k) for k in ("id", "src", "name", "ref", "operator",
                                                    "network", "kind", "km", "service")},
                             ensure_ascii=False))
            print("  type:", out.get(l["id"]), "source:", src.get(l["id"]))
            if l.get("src") != "osm":
                for k, s in sorted(cov.get(l["id"], {}).items(), key=lambda x: -x[1]):
                    print(f"  covered {k}: {s:.2f}")
            else:
                t, kids = route_tags(l, ctx)
                print("  service tags:", t.get("service"), [k.get("service") for k in kids][:10])
        if a.sample:
            random.seed(7)
            bys = defaultdict(list)
            for l in ctx.lines:
                if l["id"] in out:
                    bys[src[l["id"]]].append(l)
            for s, ls in sorted(bys.items()):
                for l in random.sample(ls, min(a.sample, len(ls))):
                    print(f"   [{s}] {l['id']} {l['kind']} {l['km']:.0f} km | {l['name'][:50]} | "
                          f"{l.get('operator', '')[:25]} -> {out[l['id']]}")
        if a.untyped:
            for l in sorted(ctx.lines, key=lambda l: -l["km"]):
                if l["id"] not in out:
                    print(f"   untyped {l['id']} {l['kind']} {l['km']:.0f} km | {l['name'][:60]} | "
                          f"{l.get('operator', '')[:30]}")
    tot = sum(r["lines"] for r in rows)
    typed = sum(r["typed"] for r in rows)
    free = sum(r["free"] for r in rows)
    print(f"ALL: {typed}/{tot} lines typed ({typed / max(tot, 1):.1%}); "
          f"from tags and kind alone {free} ({free / max(tot, 1):.1%})")
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
