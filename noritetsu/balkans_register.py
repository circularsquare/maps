"""Serbia, Bosnia and Herzegovina, Montenegro, North Macedonia, Albania and Kosovo: each
country's line list written into rinf.py's input format (as ru_register.py and in_register.py
do), so rinf.py traces it over OSM track, matches stops, merges sections and names lines.

    python balkans_register.py --clip ba        # after ba's extract (see CLIPPING)
    python build_model.py --region rs --register balkans_register:data/raw/rinf/rs
    python balkans_register.py --convert rs     # the conversion alone, with its log
    (likewise ba, me, mk, al, xk)

`--register balkans_register:data/raw/rinf/<cc>` converts and then runs rinf.build on the
result; `--register rinf:data/raw/rinf/<cc>` builds from the last conversion. The settings
rinf.py reads are in rinf_countries/<cc>.py (each a call to `country_conf` below), the sources
and numbers in balkans_sources.md.

WHY NOT ONE OSM RECIPE.  None of the six is in ERA RINF. OSM's named track (kr_register's
recipe) works only in Serbia, where 85% of main-line km carries IŽS's line number in `ref`;
in Bosnia 2%, North Macedonia 2%, Albania 0%. So every country has a LINE LIST here, and the
OSM side only supplies stations and track:

- SERBIA: IŽS's own register, Appendix 6 of its Network Statement 2026 ("Register of
  infrastructure data": every service point of every line, in order, with its chainage),
  transcribed into data/raw/rs_ns2026_appendix6.txt. The km between two points is the
  difference of their chainages (or the table's own distance where the chainage restarts), so
  every section carries IŽS's length and check_model compares every line.
- THE OTHER FIVE: hand-written lists below (LINES), the line's ends, junctions and borders by
  OSM station name or coordinate, km traced over OSM track. rinf.py's `osm_stops: "all"` puts
  every OSM station lying on a traced section on the line as a stop.

STATIONS.  A list name is looked up among OSM's rail stations under one key for Latin and
Cyrillic (`lat_key`: Serbian/Macedonian Cyrillic to Latin, then diacritics folded, đ = dj), so
IŽS's "ŠID" finds OSM's "Шид" (rinf.norm alone folds ш to "sh" and š to "s"). Of several
places with the name, the one nearest the line's previous placed point, within the km the
list says plus 3 km, is taken. The point then carries the OSM station's own name and
coordinate, so rinf.py matches it to that station exactly. A list station with no OSM station
of its name is left out (logged) and its km go to the section across it.

JUNCTIONS.  IŽS's open-line junctions have no coordinate. One where another line branches
(SHARED_JUNCTIONS) is placed on the line it lies inside, at its chainage between the two
placed points either side, along the traced track; the branching line then ends there. Any
other junction is left out (rinf.py would merge it away anyway).

BORDERS.  A line to a state border ends at the border point: the shared table's id where one
exists (border_points.json; Croatia's RINF points EU00221-EU00227, which the table lacks, by
their RINF uopid), else an id of ours ("XMERS1") at the coordinate measured along the track
from IŽS's chainage (Vrbnica - border 2.245 km, Preševo - border 8.143 km) or where the track
crosses the country outline. balkans_sources.md lists the borders.EXTRA entries these need.

TERRITORY (Anita: de facto, drawn as trains run).  Kosovo is its own region (`xk`): Trainkos
runs its trains. IŽS's register still lists the lines in Kosovo (109 beyond the
administrative line, 223 beyond Merdare, 224, 225, 312); here they end at the administrative
line, and Kosovo's lines are written from OSM like the other four. The Belgrade - Bar line's
9 km through Bosnia at Štrpci are run by Srbijavoz and ŽPCG: they stay on Serbia's 108, and
`clip("ba")` takes that track and the Serbian trains' route relations out of Bosnia's
extract so Bosnia's build does not draw them again.
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
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "rinf"
RS_TABLE = ROOT / "data" / "raw" / "rs_ns2026_appendix6.txt"

# ============================================================== names and keys

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž", "з": "z",
    "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "č", "џ": "dž", "ш": "š",
    # Macedonian
    "ѓ": "gj", "ќ": "kj", "ѕ": "dz",
}


def to_latin(s):
    """Serbian or Macedonian Cyrillic to Latin, letter for letter (case kept roughly)."""
    out = []
    for ch in s or "":
        lo = ch.lower()
        t = _CYR.get(lo)
        if t is None:
            out.append(ch)
        else:
            out.append(t.capitalize() if ch != lo else t)
    return "".join(out)


def lat_key(s):
    """One key for a station name in Latin or Cyrillic: "ŠID" and "Шид" are "sid", "ĐUNIS"
    and "Ђунис" "djunis"."""
    s = to_latin(s).casefold().replace("đ", "dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]", "", s)


def key_parts(s):
    """The whole name's key and each part's ("LOVĆENAC-MALI IĐOŠ", "Ловћенац")."""
    out = [lat_key(s)]
    for p in re.split(r"\s*[-/–(),]\s*", s or ""):
        k = lat_key(p)
        if len(k) >= 4 and k not in out:
            out.append(k)
    return out


_LAT2CYR = [("lj", "љ"), ("nj", "њ"), ("dž", "џ"), ("dz", "дз")]
_L1 = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "đ": "ђ", "e": "е", "ž": "ж", "z": "з",
       "i": "и", "j": "ј", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р",
       "s": "с", "t": "т", "ć": "ћ", "u": "у", "f": "ф", "h": "х", "c": "ц", "č": "ч", "š": "ш"}


def to_cyr(s):
    """Serbian Latin to Cyrillic (a register name with no OSM station to take a name from)."""
    out, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        hit = next((c for l2, c in _LAT2CYR if two.lower() == l2), None)
        if hit and not (two.lower() == "dz"):
            out.append(hit.upper() if two[0].isupper() else hit)
            i += 2
            continue
        ch = s[i]
        c = _L1.get(ch.lower())
        out.append(ch if c is None else (c.upper() if ch.isupper() else c))
        i += 1
    return "".join(out)


def title(s):
    """IŽS's capitals as a name: "BEOGRAD CENTAR" -> "Beograd Centar"."""
    return " ".join(w.capitalize() if not re.fullmatch(r"[IVX]+|\d.*", w) else w
                    for w in s.split())


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


# ============================================================== the line lists
#
# A point is written as:
#   "Name"                 an OSM rail station of that name (lat_key), nearest the line so far
#   "Name@lon,lat"         that station near the coordinate; with no station of the name
#                          within 1.5 km, a junction there under that name
#   "#ID" / "#ID@lon,lat"  a border point: border_points.json's id, or ours at the coordinate

def P(name, km=None, dist=None):
    return {"name": name, "km": km, "dist": dist}


# Borders the shared table lacks: Croatia's RINF points (type 90, filed by HŽI only) and the
# crossings no RINF country has. Coordinates on the track: Croatia's from its RINF points;
# Vrbnica and Preševo measured along OSM's track from the last Serbian station by IŽS's
# chainage; the rest where the track crosses religiondots' country outline.
BORDERS = {
    "EU00221": (18.49626, 45.06100, ["ba", "hr"]),     # Slavonski Šamac - Šamac
    "EU00222": (17.65746, 43.05886, ["ba", "hr"]),     # Metković - Čapljina
    "EU00223": (16.48235, 45.18939, ["ba", "hr"]),     # Volinja - Dobrljin
    "EU00225": (18.83090, 44.86828, ["ba", "hr"]),     # Gunja (Drenovci DG) - Brčko
    "EU00226": (19.16442, 45.14734, ["hr", "rs"]),     # Tovarnik - Šid
    "EU00227": (19.08632, 45.52332, ["hr", "rs"]),     # Erdut - Bogojevo
    # The table has these two, but RINF files them well inside Hungary and Romania: Röszke's
    # at Röszke station, 7 km from the border (borders.py says so), Jimbolia's 6 km off the
    # Kikinda line altogether ("Jimbolia FR", 4.725 km from Jimbolia on 100A). Here at the
    # crossing on OSM's track; borders.MOVE should take them there too.
    "EU00200": (19.968472, 46.160999, ["hu", "rs"]),   # Horgoš - Röszke (201)
    "EU00247": (20.641535, 45.806910, ["ro", "rs"]),   # Kikinda - Jimbolia (202)
    "XMERS1": (19.777497, 43.136296, ["me", "rs"]),    # Vrbnica - Bijelo Polje (108 / Bar line)
    "XMKRS1": (21.697622, 42.231693, ["mk", "rs"]),    # Preševo - Tabanovce
    "XRSXK1": (20.694040, 43.224582, ["rs", "xk"]),    # Rudnica - Jarinje (109)
    "XMKXK1": (21.294851, 42.148217, ["mk", "xk"]),    # Hani i Elezit - Volkovo
    "XBARS1": (19.126737, 44.415256, ["ba", "rs"]),    # Donja Borina - Zvornik Novi (211 / 15)
    "XBARS2": (19.462349, 43.761046, ["ba", "rs"]),    # Mokra Gora - Višegrad (501, 760 mm)
    "XALME1": (19.392590, 42.316865, ["al", "me"]),    # Tuzi - Hani i Hotit (freight)
}

# Serbia: which IŽS lines are built, their names (Latin, written in Cyrillic as OSM and IŽS
# write them; English with the capital's English name), and where a border or the
# administrative line ends them.
RS_LINES = {
    "101": ("Beograd Centar – Šid", "EU00226"),
    "102": ("Beograd Centar – Niš – Preševo", "XMKRS1"),
    "103": ("Rakovica – Mala Krsna – Velika Plana", None),
    "104": ("Ćuprija – Paraćin", None),
    "105": ("Stara Pazova – Novi Sad – Subotica", "EU00199"),
    "106": ("Niš – Dimitrovgrad", "EU00211"),
    "107": ("Beograd Centar – Pančevo – Vršac", "EU00248"),
    "108": ("Resnik – Požega – Vrbnica", "XMERS1"),
    "109": ("Lapovo – Kraljevo – Rudnica", "XRSXK1"),
    "110": ("Subotica – Sombor – Bogojevo", "EU00227"),
    "120": ("Karađorđev park – Dedinje", None),
    "121": ("Inđija – Golubinci", None),
    "201": ("Subotica – Horgoš", "EU00200"),
    "202": ("Pančevo Glavna – Zrenjanin – Kikinda", "EU00247"),
    "205": ("Banatsko Miloševo – Senta – Subotica", None),
    "207": ("Novi Sad – Odžaci – Bogojevo", None),
    "208": ("Novi Sad – Rimski Šančevi – Orlovat", None),
    "211": ("Ruma – Šabac – Brasina", "XBARS1"),
    "213": ("Stalać – Kraljevo – Požega", None),
    "216": ("Smederevo – Mala Krsna", None),
    "218": ("Mala Krsna – Požarevac – Bor – Vražogrnac", None),
    "219": ("Niš – Zaječar – Prahovo Pristanište", None),
    "223": ("Doljevac – Prokuplje – Kuršumlija – Merdare", None),
    "308": ("Donja Borina – Zvornik Grad", None),
    "309": ("Pančevo Varoš – Pančevo Vojlovica", None),
    "501": ("Šargan Vitasi – Mokra Gora", "XBARS2"),
}
# Left out, no passenger train and OSM maps their stations as disused (disused:railway=station):
# 226 Vrbas - Sombor (Kula, Crvenka, Sivac; OSM's track has gaps, the trace went round by
# Bogojevo at 2.2 times IŽS's 54.4 km), 306 Rimski Šančevi - Žabalj, 311 Markovac - Resavica,
# 313 Vršac - Bela Crkva. The yard and connecting lines 111-119, 122-128, 203-204, 206, 209-210,
# 212, 214-215, 217, 220-222 (221/222 Kuršumlija's two legs are folded into 223), 301-305, 310,
# 4xx are not passenger lines.
# Border points placed where OSM's track crosses the country outline rather than by IŽS's
# chainage: the section to them takes its traced length, not the table's.
OUTLINE_BORDERS = {"XRSXK1", "XBARS1", "XBARS2", "EU00200", "EU00247"}
# Junctions another line branches at, placed on the line they lie inside.
SHARED_JUNCTIONS = {"OPEN LINE JUNCTION ĆUPRIJA", "OPEN LINE JUNCTION DEDINJE",
                    "OPEN LINE JUNCTION SAJLOVO", "OPEN LINE JUNCTION 2 VRAŽOGRNAC",
                    "OPEN LINE JUNCTION DONJA BORINA"}
EN_PLACE = {"Beograd Centar": "Belgrade Centre", "Beograd": "Belgrade"}


def rs_lines(log):
    rows = defaultdict(list)
    for ln in RS_TABLE.read_text("utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("|")
        rows[f[0]].append(P(f[2].strip(), float(f[1]), float(f[3]) if len(f) > 3 else None))
    out = []
    for ref, (name, border) in RS_LINES.items():
        positions(rows.get(ref, []))                    # before any point is left out
        pts = []
        for i, p in enumerate(rows.get(ref, [])):
            n = p["name"]
            if n in ("STATE BORDER", "ADMINISTRATIVE LINE"):
                if border:
                    pts.append(dict(p, name="#" + border,
                                    pos=None if border in OUTLINE_BORDERS else p["pos"]))
                if i:
                    break                               # nothing of Serbia's beyond it
                continue
            if "JUNCTION" in n and n not in SHARED_JUNCTIONS:
                continue
            pts.append(p)
        en = name
        for a, b in EN_PLACE.items():
            en = en.replace(a, b)
        out.append({"id": ref, "ref": ref, "name": f"{ref} {to_cyr(name)}",
                    "name_en": f"Line {ref} ({en})", "im": "Инфраструктура железнице Србије",
                    "pts": pts})
    return out


def L(lid, name, name_en, im, pts, ref=None, suspended=False, note=""):
    return {"id": lid, "ref": ref, "name": name, "name_en": name_en, "im": im,
            "pts": [P(*_km(p)) for p in pts], "suspended": suspended, "note": note}


def _km(p):
    """"Name=12.345": a point with its chainage (km)."""
    if "=" in p:
        n, k = p.rsplit("=", 1)
        return n, float(k)
    return p, None


FBH, RSR = "Željeznice Federacije BiH", "Željeznice Republike Srpske"
LINES = {
    # Bosnia and Herzegovina: ŽFBH's and ŽRS's line numbers (OSM's route=railway refs 11-17).
    # Šamac, Tuzla and Brčko have no OSM station: placed at the station's coordinate, they end
    # their lines as junctions.
    "ba": [
        L("11", "11 Sarajevo – Čapljina", "Line 11 (Sarajevo – Čapljina)", FBH,
          ["Sarajevo", "Konjic", "Jablanica", "Mostar", "Čapljina", "#EU00222"], ref="11"),
        L("12", "12 Šamac – Doboj – Sarajevo", "Line 12 (Šamac – Doboj – Sarajevo)", FBH,
          ["#EU00221", "Šamac@18.4675,45.0590", "Doboj", "Maglaj", "Zavidovići", "Žepče",
           "Zenica", "Visoko", "Podlugovi", "Sarajevo"], ref="12"),
        L("13", "13 Novi Grad – Banja Luka – Doboj – Tuzla",
          "Line 13 (Novi Grad – Banja Luka – Doboj – Tuzla)", RSR,
          ["Novi Grad", "Prijedor", "Banja Luka", "Doboj", "Petrovo Novo",
           "Tuzla@18.6680,44.5395"], ref="13"),
        L("14", "14 Brčko – Banovići", "Line 14 (Brčko – Banovići)", FBH,
          ["#EU00225", "Brčko@18.8070,44.8735", "Banovići"], ref="14"),
        L("15", "15 Živinice – Zvornik", "Line 15 (Živinice – Zvornik)", FBH,
          ["Živinice", "Kalesija", "#XBARS1"], ref="15"),
        L("17", "17 Dobrljin – Novi Grad – Bihać", "Line 17 (Dobrljin – Novi Grad – Bihać)", RSR,
          ["#EU00223", "Dobrljin", "Novi Grad", "Bosanska Krupa", "Bihać"], ref="17"),
    ],
    # Montenegro: ŽICG's three lines.
    "me": [
        L("bar", "Bar – Vrbnica", "Bar – Vrbnica (Belgrade – Bar railway)",
          "Željeznička infrastruktura Crne Gore",
          ["Bar=454.847", "Podgorica=405.143", "Kolašin=340.650", "Mojkovac=321.339",
           "Bijelo Polje=296.938", "#XMERS1=287.439"]),
        L("niksic", "Podgorica – Nikšić", "Podgorica – Nikšić",
          "Željeznička infrastruktura Crne Gore", ["Podgorica=56.508", "Danilovgrad=34.426", "Nikšić=0.293"]),
        L("tuzi", "Podgorica – Tuzi – državna granica", "Podgorica – Tuzi – Albanian border",
          "Željeznička infrastruktura Crne Gore", ["Podgorica=0", "Tuzi=13.683", "#XALME1"]),
    ],
    # North Macedonia: MŽ Infrastruktura's lines, named as Wikidata and OSM's relations do.
    "mk": [
        L("tab-gev", "Табановце – Гевгелија", "Tabanovce – Gevgelija",
          "Македонски железници – Инфраструктура",
          ["#XMKRS1", "Табановце", "Куманово", "Скопје", "Велес", "Гевгелија", "#EU00189"]),
        L("sko-vol", "Скопје – Волково – Блаце", "Skopje – Volkovo – Blace",
          "Македонски железници – Инфраструктура",
          ["Скопје", "Ѓорче Петров", "Волково", "Блаце", "#XMKXK1"]),
        L("gp-kic", "Ѓорче Петров – Кичево", "Gjorče Petrov – Kičevo",
          "Македонски железници – Инфраструктура",
          ["Ѓорче Петров", "Тетово", "Гостивар", "Кичево"], suspended=True,
          note="no train in MŽ Transport's 2025/26 timetable; OSM's old route relation kept it drawn"),
        L("vel-bit", "Велес – Битола – Кременица", "Veles – Bitola – Kremenica",
          "Македонски железници – Инфраструктура",
          ["Велес", "Прилеп", "Битола", "Жабени", "#EU00190"]),
        L("vel-koc", "Велес – Кочани", "Veles – Kočani", "Македонски железници – Инфраструктура",
          ["Велес", "Штип", "Кочани"]),
        L("kum-bel", "Куманово – Бељаковце", "Kumanovo – Beljakovce",
          "Македонски железници – Инфраструктура", ["Куманово", "Бељаковце"]),
    ],
    # Albania: HSH's lines. Only Durrës (Plazh) - Elbasan has passenger trains (Friday to
    # Sunday); Durrës - Tiranë is being rebuilt (service planned for the end of 2027).
    "al": [
        L("dur-tir", "Durrës – Tiranë", "Durrës – Tirana", "Hekurudha Shqiptare",
          ["Durrës", "Shkozet", "Vorë", "Kashar"], suspended=True),
        L("shk-elb", "Shkozet – Rrogozhinë – Elbasan", "Shkozet – Rrogozhinë – Elbasan",
          "Hekurudha Shqiptare",
          ["Shkozet", "Durrës Plazh", "Golem", "Kavajë", "Rrogozhinë", "Peqin", "Elbasan"]),
        L("elb-pog", "Elbasan – Pogradec", "Elbasan – Pogradec", "Hekurudha Shqiptare",
          ["Elbasan", "Librazhd", "Pogradec"], suspended=True),
        L("rro-vlo", "Rrogozhinë – Fier – Vlorë", "Rrogozhinë – Fier – Vlorë",
          "Hekurudha Shqiptare", ["Rrogozhinë", "Lushnjë", "Fier", "Vlorë"], suspended=True),
        L("fie-bal", "Fier – Ballsh", "Fier – Ballsh", "Hekurudha Shqiptare",
          ["Fier", "Ballsh"], suspended=True),
        L("vor-shk", "Vorë – Shkodër – Hani i Hotit", "Vorë – Shkodër – Hani i Hotit",
          "Hekurudha Shqiptare", ["Vorë", "Laç", "Lezhë", "Shkodër", "Bajza", "#XALME1"],
          suspended=True),
    ],
    # Kosovo: Infrakos's lines. Line 10's passenger trains (Prishtinë - Skopje) have been
    # suspended since 2020 for its rebuilding, and nothing runs north of Fushë Kosovë.
    "xk": [
        L("10", "Linja 10: Leshak – Fushë Kosovë – Hani i Elezit",
          "Line 10 (Leshak – Fushë Kosovë – Hani i Elezit)", "Infrakos",
          ["#XRSXK1", "Lešak", "Leposavić", "Zvečan", "Mitrovicë", "Vushtrri", "Fushë Kosovë",
           "Lipjan", "Ferizaj", "Kaçanik", "Han i Elezit", "#XMKXK1"], ref="10",
          suspended=True),
        L("fk-pej", "Fushë Kosovë – Pejë", "Fushë Kosovë – Peja", "Infrakos",
          ["Fushë Kosovë", "Drenas", "Klinë", "Pejë"]),
        L("fk-pri", "Fushë Kosovë – Prishtinë", "Fushë Kosovë – Pristina",
          "Infrakos", ["Fushë Kosovë", "Prishtinë"]),
        L("kli-pri", "Klinë – Prizren", "Klinë – Prizren", "Infrakos",
          ["Klinë", "Prizren@20.5697,42.3548"]),
    ],
}

LANGS = {"rs": ["sr", "en"], "ba": ["bs", "sr", "hr", "en"], "me": ["sr", "en"],
         "mk": ["mk", "en"], "al": ["sq", "en"], "xk": ["sq", "sr", "en"]}
ISO3 = {"rs": "SRB", "ba": "BIH", "me": "MNE", "mk": "MKD", "al": "ALB", "xk": "XKX"}
WIKIDATA = {"rs": "Q403", "ba": "Q225", "me": "Q236", "mk": "Q221", "al": "Q222", "xk": "Q1246"}


def lines_of(cc, log=print):
    return rs_lines(log) if cc == "rs" else LINES[cc]


# ============================================================== conversion

def positions(pts):
    """Each point's position along its line: its chainage, shifted where the table gives a
    distance instead (the chainage restarts at Senta, Šabac, Kikinda; Inđija - Golubinci's
    passenger distance is not the chainage's): from there on every chainage moves by the
    same amount, so differences stay the table's."""
    pos, off = None, 0.0
    for p in pts:
        if p.get("km") is not None:
            if p.get("dist") is not None and pos is not None:
                pos = pos + p["dist"]
                off = pos - p["km"]
            else:
                pos = p["km"] + off
            p["pos"] = pos
        else:
            p["pos"] = None


def border_xy(bid):
    if bid in BORDERS:
        return BORDERS[bid][:2]
    import borders
    for p in borders.load():
        if p["id"] == "e" + bid:
            return p["lon"], p["lat"]
    raise SystemExit(f"no coordinate for border point {bid}")


def convert(cc, log=print, write=True):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log)
    ost = rinf.osm_stations(stops)
    by_key = defaultdict(list)
    for sid, s in ost.items():
        for k in key_parts(s["name"]):
            by_key[k].append(sid)
    infra = {}
    ip = ROOT / "data" / "proc" / cc / "infra.pkl"
    if ip.exists():
        with open(ip, "rb") as f:
            infra = pickle.load(f)
    rel_ways = defaultdict(set)                          # ref -> ways of the line's relations
    for rid, (tags, members) in infra.items():
        r = (tags.get("ref") or "").strip()
        if r:
            rel_ways[r] |= {ref for ty, ref, _ in members if ty == "w"}
    way_ref = defaultdict(set)
    for wid, (tags, _n) in ways.items():
        if tags.get("ref"):
            way_ref[tags["ref"].strip()].add(wid)

    lines = lines_of(cc, log)
    stat = Counter()
    unmatched = defaultdict(list)

    def parse(p):
        n = p["name"]
        at = None
        m = re.match(r"^(.*)@([-\d.]+),([-\d.]+)$", n)
        if m:
            n, at = m.group(1), (float(m.group(2)), float(m.group(3)))
        if n.startswith("#"):
            return "border", n[1:], at or border_xy(n[1:])
        if "JUNCTION" in n:
            return "junction", n, at
        return "station", n, at

    # --- 1. positions along each line (km), and stations placed at OSM's
    for l in lines:
        if not any("pos" in p for p in l["pts"]):
            positions(l["pts"])
        for p in l["pts"]:
            p["kind"], p["label"], p["at"] = parse(p)
    placed = {}                                          # (line id, index) -> (lon, lat, sid)

    def cands(name):
        out = []
        for k in key_parts(name):
            out = by_key.get(k, [])
            if out:
                break
        return out

    for _round in range(3):
        for l in lines:
            pts = l["pts"]
            for i, p in enumerate(pts):
                if (l["id"], i) in placed:
                    continue
                if p["kind"] == "border":
                    placed[(l["id"], i)] = (*p["at"], None)
                    continue
                if p["kind"] != "station":
                    continue
                cs = cands(p["label"])
                if p["at"]:
                    near = [c for c in cs if dist_m(ost[c]["lon"], ost[c]["lat"], *p["at"]) <= 1500]
                    if near:
                        c = min(near, key=lambda c: dist_m(ost[c]["lon"], ost[c]["lat"], *p["at"]))
                        placed[(l["id"], i)] = (ost[c]["lon"], ost[c]["lat"], c)
                    else:
                        p["kind"] = "junction"            # no station there: a line end
                        placed[(l["id"], i)] = (*p["at"], None)
                    continue
                if not cs:
                    continue
                # the nearest placed neighbours on this line, before and after
                nb = []
                for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(pts))):
                    if (l["id"], j) in placed:
                        q = placed[(l["id"], j)]
                        gap = (abs(pts[j]["pos"] - p["pos"]) if pts[j]["pos"] is not None
                               and p["pos"] is not None else None)
                        nb.append((q, gap))
                        if len(nb) >= 2:
                            break
                if nb:
                    best = None
                    for c in cs:
                        s = ost[c]
                        ok = all(gap is None or dist_m(s["lon"], s["lat"], q[0], q[1]) / 1000
                                 <= gap + 3 for q, gap in nb)
                        d = min(dist_m(s["lon"], s["lat"], q[0], q[1]) for q, _g in nb)
                        if ok and d <= 150000 and (best is None or d < best[0]):
                            best = (d, c)
                    if best:
                        c = best[1]
                        placed[(l["id"], i)] = (ost[c]["lon"], ost[c]["lat"], c)
                elif _round == 2 or len({(round(ost[c]["lon"], 2), round(ost[c]["lat"], 2))
                                         for c in cs}) == 1:
                    c = cs[0]
                    placed[(l["id"], i)] = (ost[c]["lon"], ost[c]["lat"], c)

    # --- 2. own track per line (its relations' ways, or ways carrying its number)
    def own_of(l):
        ws = set()
        if l.get("ref"):
            ws |= rel_ways.get(l["ref"], set())
            if cc == "rs":
                ws |= way_ref.get(l["ref"], set())
        return ws or None

    traces = {}

    def trace(l, a, b, km=None):
        key = (l["id"], a, b)
        if key not in traces:
            pa, pb = placed[(l["id"], a)], placed[(l["id"], b)]
            sa, sb = track.snap(pa[0], pa[1]), track.snap(pb[0], pb[1])
            if sa is None or sb is None:
                traces[key] = None
            else:
                crow = dist_m(pa[0], pa[1], pb[0], pb[1]) / 1000
                traces[key] = track.trace(sa, sb, km or crow * 2 + 5, own_of(l))
        return traces[key]

    # --- 3. shared junctions placed inside the line that passes through them
    jxy = {}
    for l in lines:
        pts = l["pts"]
        for i, p in enumerate(pts):
            if p["kind"] != "junction" or (l["id"], i) in placed or p["pos"] is None:
                continue
            a = next((j for j in range(i - 1, -1, -1) if (l["id"], j) in placed
                      and pts[j]["pos"] is not None), None)
            b = next((j for j in range(i + 1, len(pts)) if (l["id"], j) in placed
                      and pts[j]["pos"] is not None), None)
            if a is None or b is None:
                continue
            got = trace(l, a, b, abs(pts[b]["pos"] - pts[a]["pos"]))
            if not got:
                continue
            path = got[0]
            f = (p["pos"] - pts[a]["pos"]) / (pts[b]["pos"] - pts[a]["pos"])
            cum = [0.0]
            for u, v in zip(path[:-1], path[1:]):
                cum.append(cum[-1] + dist_m(*u, *v))
            target = f * cum[-1]
            for k in range(len(cum) - 1):
                if cum[k + 1] >= target:
                    t = (target - cum[k]) / (cum[k + 1] - cum[k]) if cum[k + 1] > cum[k] else 0
                    u, v = path[k], path[k + 1]
                    xy = (u[0] + t * (v[0] - u[0]), u[1] + t * (v[1] - u[1]))
                    placed[(l["id"], i)] = (*xy, None)
                    jxy.setdefault(p["label"], xy)
                    break
    for l in lines:
        for i, p in enumerate(l["pts"]):
            if p["kind"] == "junction" and (l["id"], i) not in placed and p["label"] in jxy:
                placed[(l["id"], i)] = (*jxy[p["label"]], None)

    # --- 4. rows
    rows, pts_out, names = [], {}, {}
    for l in lines:
        seq = []
        for i, p in enumerate(l["pts"]):
            if (l["id"], i) in placed:
                seq.append(i)
            elif p["kind"] == "station":
                unmatched[l["id"]].append(p["label"])
                stat["list stations with no OSM station, left out"] += 1
            else:
                unmatched[l["id"]].append(p["label"])
                stat["junctions not placed, left out"] += 1
        traced_total, pub_total, k = 0.0, 0.0, 0
        for a, b in zip(seq[:-1], seq[1:]):
            pa, pb = l["pts"][a], l["pts"][b]
            qa, qb = placed[(l["id"], a)], placed[(l["id"], b)]
            got = trace(l, a, b, abs(pb["pos"] - pa["pos"]) if pa["pos"] is not None
                        and pb["pos"] is not None else None)
            tkm = got[1] if got else None
            if pa["pos"] is not None and pb["pos"] is not None:
                km = abs(pb["pos"] - pa["pos"])
                pub_total += km
                stat["sections with the register's km"] += 1
            elif tkm is not None:
                km = tkm
                stat["sections with traced km"] += 1
            else:
                km = dist_m(*qa[:2], *qb[:2]) / 1000 * 1.2
                stat["sections with no trace (crow-fly x 1.2)"] += 1
            traced_total += tkm or 0.0
            k += 1
            ops = []
            for idx, q in ((a, qa), (b, qb)):
                p = l["pts"][idx]
                if p["kind"] == "border":
                    op = f"{cc}:b:{p['label']}"
                    pts_out[op] = {"op": op, "uopid": p["label"], "type": "90",
                                   "name": p["label"], "lon": q[0], "lat": q[1]}
                elif q[2] is not None:
                    op = f"{cc}:s:{q[2]}"
                    pts_out[op] = {"op": op, "uopid": f"{cc.upper()}{q[2]}", "type": "10",
                                   "name": ost[q[2]]["name"], "lon": q[0], "lat": q[1]}
                else:
                    jk = re.sub(r"[^0-9A-Za-z]+", "", lat_key(p["label"]))[:24]
                    op = f"{cc}:j:{jk}"
                    nm = p["label"]
                    if cc == "rs":
                        nm = to_cyr(title(nm.replace("OPEN LINE JUNCTION", "Распутница")
                                          .replace("JUNCTION POINT", "Одвојна скретница")))
                    pts_out[op] = {"op": op, "uopid": f"{cc.upper()}J{jk}", "type": "80",
                                   "name": nm, "lon": q[0], "lat": q[1]}
                ops.append(op)
            rows.append({"sol": f"{cc}:{l['id']}:{k}", "line": l["id"], "a": ops[0], "b": ops[1],
                         "len": f"{km:.3f}", "im": l["im"],
                         "label": f"{l['pts'][a]['label']} - {l['pts'][b]['label']}"})
        names[l["id"]] = {"name": l["name"], "name_en": l["name_en"], "ref": l.get("ref") or "",
                          "im": l["im"], "suspended": bool(l.get("suspended")),
                          "register_km": round(pub_total, 3) or None,
                          "traced_km": round(traced_total, 3)}
        log(f"  {l['id']:>8} {l['name'][:48]:<48} {len(seq):3d} points  traced {traced_total:7.1f} km"
            + (f"  register {pub_total:7.1f} km ({traced_total / pub_total:.3f})"
               if pub_total else ""))
    log(f"{cc.upper()}: {len(lines)} lines, {len(rows)} section rows, {len(pts_out)} points; "
        f"{dict(stat)}")
    for lid, ns in unmatched.items():
        log(f"    left out on {lid}: {', '.join(ns)}")
    if write:
        d = OUT / cc
        d.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "balkans_register.py (IŽS Network Statement 2026, Appendix 6)"
                 if cc == "rs" else "balkans_register.py (hand-written line list)",
                 "fetched": date.today().isoformat()}
        (d / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                         "utf-8")
        (d / "points.json").write_text(json.dumps({**stamp, "rows": list(pts_out.values())},
                                                  ensure_ascii=False), "utf-8")
        (d / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
        log(f"wrote {d} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(Path(path).name, log)
    import rinf
    return rinf.build(path, log)


# ============================================================== rinf.py settings

def _names(cc):
    f = OUT / cc / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


class LineName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""

    def __new__(cls, cc, field):
        o = str.__new__(cls, "{ref}")
        o.cc, o.field = cc, field
        return o

    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        e = _names(self.cc).get(ref) or next(
            (v for v in _names(self.cc).values() if v.get("ref") == ref), None)
        return e[self.field] if e else ref


def country_conf(cc):
    def id_name(lid, _uop=None):
        e = _names(cc).get(lid.split("#")[0])
        return (e["name"], e.get("name_en") or "") if e else None

    def suspended(_ref, lids):
        ns = _names(cc)
        return any(ns.get(x.split("#")[0], {}).get("suspended") for x in lids)

    ims = {l["im"] for l in (LINES.get(cc) or [{"im": "Инфраструктура железнице Србије"}])}
    conf = {
        "iso3": ISO3[cc], "langs": LANGS[cc],
        "ref": lambda lid: (_names(cc).get(lid, {}).get("ref") or None),
        "rule_certain": True,
        "name": LineName(cc, "name"), "name_en": LineName(cc, "name_en"),
        "id_name": id_name,
        "suspended": suspended,
        "im": {x: x for x in ims},
        "osm_stops": "all",
    }
    if cc == "rs":
        # IŽS's chainage runs between station axes and junction points by the book; OSM's
        # junctions in Belgrade are some hundred metres off where it puts them (Karađorđev park
        # - Dedinje is 1.49 km in the table and 0.9 traced).
        conf["tol_abs"] = 0.7
    if cc not in ("rs", "ba"):
        # No line numbers: OSM's relations must not lend one (North Macedonia's "1" is
        # Bulgaria's line 1 too, Montenegro's M11 is OSM's own).
        conf["osm_rel"] = lambda _t: None
    return conf


# ============================================================== clipping Bosnia's extract

def clip(cc, log=print):
    """Bosnia's extract holds the Belgrade - Bar line's 9 km through Štrpci and the Serbian
    and Montenegrin trains over it. They are Serbia's (run by Srbijavoz and ŽPCG; built on 108
    from Serbia's extract), so their track, stations and route relations go from data/proc/ba.
    Run after every extract of ba."""
    if cc != "ba":
        raise SystemExit("only ba is clipped")
    d = ROOT / "data" / "proc" / cc
    rd = lambda f: pickle.load(open(d / f, "rb"))  # noqa: E731
    ways, rels, stops, infra = rd("ways.pkl"), rd("rels.pkl"), rd("stops.pkl"), rd("infra.pkl")
    bar = set()
    for rid, (tags, members) in infra.items():
        if "Бар" in (tags.get("name") or "") or tags.get("ref") == "108":
            bar |= {r for ty, r, _ in members if ty == "w"}
    gone_w = {w for w in bar if w in ways}
    foreign = {rid for rid, (tags, _m) in rels.items()
               if re.search(r"Србија Воз|Srbija Voz|Crne Gore", tags.get("operator") or "")}
    keep_r = {k: v for k, v in rels.items() if k not in foreign
              and not any(t == "r" and r in foreign for t, r, _ in v[1])}
    gone_s = set()
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("name") in ("Štrpci", "Рача", "Штрпци", "Rača") and 19.4 < lon < 19.6:
            gone_s.add(nid)
    keep_w = {k: v for k, v in ways.items() if k not in gone_w}
    keep_s = {k: v for k, v in stops.items() if k not in gone_s}
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)
    log(f"BA clip: {len(gone_w)} Belgrade - Bar ways, {len(rels) - len(keep_r)} Serbian and "
        f"Montenegrin route relations, {len(gone_s)} stations at Štrpci left out")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--clip", metavar="CC")
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        clip(a.clip, lg)
    if a.convert:
        convert(a.convert, lg)
    if a.dry:
        convert(a.dry, lg, write=False)
