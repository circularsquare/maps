"""Türkiye, Istanbul: İBB "Raylı Sistemler İstasyon Bazlı Yolcu ve Yolculuk Sayıları", 2025.

data.ibb.gov.tr publishes one row per line, station entrance (station_number, e.g. YNK-MR2 is
Yenikapı's second Marmaray entrance) and day. passage_cnt ("yolculuk sayısı") is the number of
Istanbulkart validations at that entrance's gates; passanger_cnt ("yolcu sayısı") counts each
card once a day and is not used. Istanbul's gates are validated on the way in only, so
passage_cnt is entries: n = 2 x the entries, for people getting on + off. A transfer that goes
through a gate (Marmaray <-> metro at Yenikapı, Üsküdar, Ayrılık Çeşmesi) is an entry there.

Covers Metro İstanbul's metro, tram and funicular lines, M11 and the Halkalı-Bahçeşehir line
(TCDD rows "GAYRETTEPE-ISTANBUL YENI HAVALIMANI", "HALKALI-BASAKSEHIR"), the Sirkeci-
Kazlıçeşme line (T6) and Marmaray ("TCDD TASIMACILIK A.S.", Halkalı-Gebze). Left out: the
cable cars TF1/TF2 and the F2 Tünel (not lines on the map), and the T2 and T3 heritage trams
(one row per line, no station).

Each entrance's 2025 total is divided by the days it has rows for (a closed entrance or a
station opened during the year is averaged over its open days); a record with fewer than 30
days is a stray reading and left out. The entrances and lines of one station are separate
counts and are added (COMBINE sum). The file's positions are mangled by thousand separators
(289.517.222 for 28.9517222) and missing for some lines; they are recovered where enough
digits survive. match_hook matches each record only to stations of its own line (by ref) in
Istanbul, so a name is never taken from another line's station of the same name.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from station_riders import name_score, name_variants, km, squash

KEY = "ibb"
CC = "tr"
FOLDER = "ibb"
COMBINE = "sum"
MODES = {"rail", "metro", "tram", "funicular"}
META = {
    "label": "İBB rail station counts",
    "name": "Raylı Sistemler İstasyon Bazlı Yolcu ve Yolculuk Sayıları 2025, İstanbul "
            "Büyükşehir Belediyesi (İBB Açık Veri Portalı)",
    "url": "https://data.ibb.gov.tr/dataset/rayli-sistemler-istasyon-bazli-yolcu-ve-yolculuk-sayilari",
    "licence": "İBB Açık Veri Lisansı (Istanbul Metropolitan Municipality Open Data License)",
    "counts": "2 x gate entries (Istanbulkart validations, \"yolculuk\") per day, 2025: "
              "the gates count entries only, doubled for on + off; metro, tram, funicular, "
              "M11, Marmaray and the Sirkeci-Kazlıçeşme line",
    "note": "Istanbul only; a transfer through a gate counts as an entry",
}
FILE = "ibb_station_2025.csv"
YEAR = 2025
MIN_DAYS = 30
BOX = (28.0, 40.75, 29.6, 41.4)
ROOT = Path(__file__).resolve().parent.parent.parent

# source line -> the refs of the map's lines it is (None: left out)
LINE_REFS = {
    "F1": ["F1"], "F4": ["F4"], "M1": ["M1A", "M1B"], "M2": ["M2"], "M3": ["M3"],
    "M4": ["M4"], "M5": ["M5"], "M6": ["M6"], "M7": ["M7"], "M8": ["M8"], "M9": ["M9"],
    "T1": ["T1"], "T4": ["T4"], "T5": ["T5"],
    "TCDD - GAYRETTEPE": ["M11"], "TCDD - HALKALI": ["B2"], "TCDD SIRKECI": ["T6"],
    "TCDD TASIMACILIK": ["B1"],
}
# Excel turned "15 Temmuz" into a date
RENAME = {"15.Tem": "15 Temmuz"}
# source name -> station id, or None (checked by hand 2026-10-09)
FORCE = {
    # M1's Kartaltepe gates are the Kocatepe station (same place, 28.8958 41.0485)
    "Kartaltepe": "n7722103248",
    # M3's Bakırköy İDO terminus is the map's Bakırköy Sahil
    "Bakırköy İdo": "n7974014825",
    # M4's Hastane (Adliye): the name also hits "Fevzi Çakmak - Hastane"
    "Hastane (Batı)": "n7713018888",
    "Hastane (Doğu/Adliye)": "n7713018888",
    # T1: Atatürk Öğrenci Yurdu is Cevizlibağ-A.Ö.Y.; Cami is the stop the map calls Yavuz
    # Selim, Keresteciler the one it calls Merter Tekstil Merkezi (same positions)
    "Atatürk Öğrenci Yurdu": "n974667949",
    "Cami": "n10698058058",
    "Keresteciler": "n10698058054",
    # T5: plain "Alibeyköy" is Alibeyköy Merkez (the others are "Metro" and "Cep Otogar")
    "Alibeyköy": "n8225587097",
    "Eyüp Devlet Hastanesi": "n8225587088",
    "Eyüp Teleferik": "n8225587085",
    # M1's airport and fair-centre stops under their map names
    "Havaalanı": "n7722103263",
    "İdtm": "n7722103262",
    # Marmaray
    "M.kemal": "t6328169450",
    # "aqua" is Florya Akvaryum, not Florya (the name match took the shorter name)
    "Florya aqua": "t6328169449",
    # M11
    "HAVALİMANI (TERMİNAL 1)": "n7246055799",
    "HAVALİMANI (KARGO TERMİNALİ )": "n8872578520",
}

DIRS = r"(?:Kuzey|KUZEY|kuzey|Güney|GÜNEY|GUNEY|güney|Doğu|DOĞU|DOGU|Dogu|Batı|BATI|Bati|" \
       r"konkors|Konkors)"


def clean(name):
    s = RENAME.get(name, name)
    s = re.sub(r"\((?:Batı|Doğu|Doğu/Adliye)\)", " ", s)
    s = re.sub(r"^(?:M|T)\d+\s+", "", s)                         # "M7 FULYA"
    s = re.sub(r"\s+(?:M\d+\s+)?(?:HOL|Hol)\s+\d+$", "", s)      # "Mahmutbey M7 Hol 1"
    s = re.sub(r"\s+\d+\s+Stad Girişi$", "", s)
    for _ in range(2):
        s = re.sub(r"\s+" + DIRS + r"\s*$", "", s.strip())
        s = re.sub(r"(?:\s+|-)\d$", "", s.strip())               # "Yenikapı-2", "Şişli 2"
    s = re.sub(r"\b(Mah|MAH)\.?(?=\s|$)", "Mahallesi", s)
    return " ".join(s.split())


def coord(s, lo, hi):
    """'289.517.222.222.222' -> 28.9517222; None when too few digits survive."""
    d = re.sub(r"\D", "", s or "")
    if len(d) < 5:
        return None
    v = int(d[:2]) + float("0." + d[2:])
    return v if lo <= v <= hi else None


def refs_of(line):
    for k, v in LINE_REFS.items():
        if line.startswith(k):
            return v
    return None


def records(raw):
    tot = defaultdict(float)
    days = defaultdict(set)
    meta = {}
    with open(raw / FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["transaction_year"] != str(YEAR) or not r["station_name"].strip():
                continue
            refs = refs_of(r["line"])
            if not refs:
                continue
            try:
                n = float(r["passage_cnt"])
            except ValueError:
                continue
            k = (r["line"], r["station_number"], r["station_name"].strip())
            tot[k] += n
            days[k].add((r["transaction_month"], r["transaction_day"]))
            if k not in meta or meta[k][0] is None:
                meta[k] = (coord(r["longitude"], 25.0, 45.0), coord(r["latitude"], 35.0, 43.0),
                           refs)
    out = []
    for k, t in tot.items():
        if len(days[k]) < MIN_DAYS or t <= 0:
            continue
        x, y, refs = meta[k]
        if x is None or y is None:
            x = y = None
        name = k[2]
        out.append({"name": name, "alt": [clean(name)], "line": k[0], "code": k[1],
                    "refs": refs, "x": x, "y": y, "box": BOX,
                    "n": 2 * t / len(days[k]), "year": YEAR})
    return out


_lines = {}


def line_refs():
    if not _lines:
        for l in json.loads((ROOT / "dist" / "data" / CC / "lines.json")
                            .read_text(encoding="utf-8"))["lines"]:
            _lines[l["id"]] = l.get("ref") or ""
    return _lines


def fold(names):
    return {n.replace("ı", "i") for n in names}


def match_hook(rec, S):
    """Only stations of the record's own line (by ref) inside Istanbul; best name, then
    nearest when the file's position survived."""
    refs = set(rec["refs"])
    lr = line_refs()
    w, s, e, n = BOX
    rn = fold(name_variants(rec["name"], *rec["alt"]))
    best = []
    for k, v in S.st.items():
        if not (w <= v["x"] <= e and s <= v["y"] <= n):
            continue
        if not any(lr.get(l) in refs for l in v.get("l", ())):
            continue
        sn = fold(S.names[k])
        sc = name_score(rn, sn)
        if sc == 0 and any(len(squash(a)) >= 5 and squash(a) in squash(b)
                           for a in rn for b in sn):
            sc = 2
        if sc:
            d = km(rec["x"], rec["y"], v["x"], v["y"]) if rec["x"] is not None else 0
            best.append((-sc, d, k))
    if not best:
        return None, None
    best.sort()
    top = [b for b in best if b[0] == best[0][0]]
    if len(top) > 1 and rec["x"] is None:
        return None, None
    return best[0][2], "line+name"
