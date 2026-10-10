"""English names for stations and lines that ship without one: dist/data/<cc>/names_en.json.

    python tools/english_names.py                 # every country, Wikidata from the cache only
    python tools/english_names.py --fetch         # also ask Wikidata for labels not cached yet
    python tools/english_names.py cn jp --sample 20   # some countries; print a spot-check

Written 2026-10-07 (translation_coverage.md, "Filled 2026-10-07"). Reads only: the shipped
dist/data/<cc>/{stations,lines,aliases}.json, data/proc/<cc>/{stops,rels}.pkl and
data/raw/wikidata_colours_{jp,kr}.json. Writes only:
  dist/data/<cc>/names_en.json   {"st": {id: English}, "ln": {id: English}, "src": {...}}
  dist/data/names_en.json        the same for every country in one file, for the search index
  tools/english_names_wikidata.json   the Wikidata label cache (QID -> labels)

Only items the shipped files leave without an English name are filled: a station with no `e`
or a line with no `name_en` whose local name is not in Latin script. Latin-script items are
left alone (their local name shows in English mode, which is right for "Wien Hbf" and
"Linie 5"). The first source that gives a name wins, in this order:

  osm    `name:en` on the OSM stop the station was built from: the station's own node, an
         OSM station folded into it (aliases.json), or a stop of the same name within 3 km.
         The build dropped these where a merge kept the node without one.
  wd     the Wikidata English label (en, en-gb, else the en.wikipedia title, else a `mul`
         label in Latin letters) of a `wikidata` tag on those same stops, or on the line's
         route / route_master relation; cleaned ("Kirov railway station" -> "Kirov") and, for
         stations, checked against the local name where the script allows (Cyrillic, Greek,
         Chinese: does the label read as a romanisation of it?).
  wdname Japanese and Korean lines by exact name to the line items cached by the colour fetch
         (data/raw/wikidata_colours_*.json), one item only, operator-matched when several.
  (--overpass, --fetch) for what is still in non-Latin letters after the sources above and
  below: Overpass full tags (name:en, name:fr in tn/ma/dz/eg, name:ja-Latn, kana readings by
  Hepburn, int_name), every Japanese line item on Wikidata by label, alias and operator
  (wd_ja), and Japanese service names made of known words and stations (service).
  stem   "<station>線" / "<station>本線" / "<station>선" from that station's English name:
         函館線 -> "Hakodate Line".
  pinyin Chinese names by pypinyin, toneless, a word per jieba word, names of up to three
         characters one word ("Huqiu Shidi Gongyuan", "Chang'anxi"); line words translated
         (地铁 Metro, N号线 Line N, 线 Line). Needs pypinyin and jieba.
  rr     Korean by Revised Romanization, with the same word table for lines.

Kanji are never read by guess, and Thai, Arabic and Persian are never transliterated.
Wikidata requests: batched VALUES queries, 2.5 s apart, backing off on 429, falling back to
the QLever mirror; User-Agent "noritetsu/0.1 (https://github.com/)".
"""
import argparse
import json
import pickle
import random
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dist" / "data"
PROC = ROOT / "data" / "proc"
RAW = ROOT / "data" / "raw"
CACHE = Path(__file__).resolve().parent / "english_names_wikidata.json"
UA = "noritetsu/0.1 (https://github.com/)"
NEAR_M = 3000

# The app's own test (index.html LATIN): Latin letters, accents included.
LATIN = re.compile(r"^[\u0000-ɏḀ-ỿ\s]+$")
# Letters the app transliterates by table (Cyrillic, Greek, Georgian, Armenian).
TABLE_SCRIPT = re.compile(r"[Ͱ-Ͽἀ-῿Ѐ-ӿႠ-ჿ԰-֏]")
HAN = re.compile(r"[㐀-鿿豈-﫿]")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힣]")
CYR = re.compile(r"[Ѐ-ӿ]")
GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")


def log(*a):
    print(*a, flush=True)


def is_latin(s):
    return bool(LATIN.match(s or ""))


def foreign(s):
    """Has letters of a script other than Latin. Arrows, dashes and curly quotes ("RE 1:
    Aachen → Hamm", "Dumbarton’s") fail the app's LATIN test but are no translation gap."""
    # Azerbaijani ə (U+0259) sits in IPA Extensions, past the app's LATIN range: Latin all the same.
    return any(unicodedata.category(c).startswith("L") and not LATIN.match(c)
               and not "ɐ" <= c <= "ʯ" for c in s or "")


def latin_label(s):
    """A label usable as an English name: Latin letters, at least one of them."""
    return bool(s) and not foreign(s) and re.search(r"[A-Za-z]", s) is not None


# ---------------------------------------------------------------- shipped data

def load_country(cc):
    st = json.loads((DATA / cc / "stations.json").read_text("utf-8"))["stations"]
    ln = json.loads((DATA / cc / "lines.json").read_text("utf-8"))["lines"]
    p = DATA / cc / "aliases.json"
    al = json.loads(p.read_text("utf-8")) if p.exists() else {}
    return st, ln, al


def gap_stations(st):
    return {k: v for k, v in st.items()
            if v.get("n") and not v.get("e") and foreign(v["n"]) and not v.get("j")}


def gap_lines(ln):
    return {x["id"]: x for x in ln if x.get("name") and not x.get("name_en")
            and foreign(x["name"])}


# ---------------------------------------------------------------- OSM candidates

def dist_m(lon1, lat1, lon2, lat2):
    import math
    k = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lon2 - lon1) * k, lat2 - lat1) * 111_320


def node_of(sid):
    m = re.fullmatch(r"[nc](\d+)", sid)
    return int(m.group(1)) if m else None


def rel_of(lid):
    m = re.fullmatch(r"[rm](\d+)", lid)
    return int(m.group(1)) if m else None


RAIL_STOP = {"station", "halt", "stop", "tram_stop"}


def station_candidates(cc, gaps, al):
    """station id -> list of (kind, tags) from OSM: 'own' (its node or an OSM station folded
    into it) before 'near' (a stop of the same name within 3 km)."""
    p = PROC / cc / "stops.pkl"
    if not p.exists() or not gaps:
        return {}
    with open(p, "rb") as f:
        stops = pickle.load(f)
    folded = defaultdict(list)
    for k, v in (al.get("stations") or {}).items():
        if v in gaps:
            folded[v].append(k)
    names = {v["n"] for v in gaps.values()}
    by_name = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        nm = tags.get("name")
        if nm in names:
            by_name[nm].append((nid, tags, lon, lat))
    out = {}
    for sid, s in gaps.items():
        c, seen = [], set()
        for oid in [sid] + folded.get(sid, []):
            nid = node_of(oid)
            if nid in stops and nid not in seen:
                tags, lon, lat = stops[nid]
                if dist_m(s["x"], s["y"], lon, lat) <= NEAR_M:
                    c.append(("own", {**tags, "@id": nid}))
                    seen.add(nid)
        near = []
        for nid, tags, lon, lat in by_name.get(s["n"], []):
            if nid in seen:
                continue
            d = dist_m(s["x"], s["y"], lon, lat)
            # Rail stops only: a bus stop of the same name carries the bus network's English.
            rail = tags.get("railway") in RAIL_STOP or any(
                tags.get(k) == "yes" for k in ("train", "subway", "light_rail", "tram", "monorail",
                                               "funicular"))
            if d <= NEAR_M and (rail or tags.get("public_transport") == "station"):
                near.append((0 if rail else 1, d, {**tags, "@id": nid}))
        near.sort(key=lambda t: (t[0], t[1]))
        c += [("near", t[2]) for t in near]
        if c:
            out[sid] = c
    return out


def line_candidates(cc, gaps, al):
    """line id -> list of (kind, tags) from its own relation, the members of its route_master,
    and OSM relations folded into it (aliases.json `lines`)."""
    p = PROC / cc / "rels.pkl"
    if not p.exists() or not gaps:
        return {}
    with open(p, "rb") as f:
        rels = pickle.load(f)
    folded = defaultdict(list)
    for k, v in (al.get("lines") or {}).items():
        if v in gaps:
            folded[v].append(k)
    out = {}
    for lid, line in gaps.items():
        c = []
        rid = rel_of(lid)
        if rid in rels:
            tags, members = rels[rid]
            c.append(("own", {**tags, "@id": rid}))
            if lid.startswith("m"):
                for ty, ref, _ in members:
                    if ty == "r" and ref in rels:
                        c.append(("member", {**rels[ref][0], "@id": ref}))
        for oid in folded.get(lid, []):
            r = rel_of(oid)
            if r in rels:
                c.append(("folded", {**rels[r][0], "@id": r}))
        if c:
            out[lid] = c
    return out


# ---------------------------------------------------------------- cleaning and checks

ST_TAIL = re.compile(r"\s*(?:,.*$|\b(?:railway|railroad|train|rail|metro|subway|underground|"
                     r"fast tram|express tram|tram|light rail|high-speed railway|mrt|lrt)?\s*(?:station|halt|stop|"
                     r"platform|stopping point|passing loop|terminal|signal(?:ling)?(?: station| stop)?)"
                     r"\b\s*$)", re.I)
ST_HEAD = re.compile(r"^(?:(?:railway|train|metro)\s+station(?:\s+in)?|station|stantsiya)\s+",
                     re.I)
# A station label that names something else: the town, the line, the operator.
ST_NOT = re.compile(r"\b(?:line|railway|railroad|district|county|prefecture|province|oblast|"
                    r"raion|rayon|municipality|city|town|village|metro system|tramway|"
                    r"company|depot)\b", re.I)


def clean_station(label, osm=False):
    # BGN's soft-sign primes ("Lyepyel′") read as noise in a station list.
    s = re.sub(r"[′ʹ]", "", (label or "").replace("_", " ").strip())
    # OSM's "Meiji-jingumae 'Harajuku'" -> "Meiji-jingumae (Harajuku)", kept.
    s = re.sub(r"\s+'([^']+)'$", r" (\1)", s)
    s = re.sub(r"\s*[〈<]([^〉>]+)[〉>]", r" (\1)", s)
    keep = re.search(r"\s*(\([^)]*\))$", s) if osm and not re.search(
        r"\((?:[^)]*(?:station|railway|metro|line|halt|platform|km|oblast|krai|raion|"
        r"prefecture|city|town|village|\d))", s, re.I) else None
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = ST_HEAD.sub("", s)
    t = ST_TAIL.sub("", s).strip(" ,-")
    t = t or s
    # A parenthesis from an OSM name:en that is part of the name is put back; Wikidata's
    # disambiguators ("(Moscow Metro)", "(Oryol Oblast)") are not.
    if keep and t:
        t = f"{t} {keep.group(1)}"
    return re.sub(r"\s{2,}", " ", t).strip()


def clean_line(label):
    s = (label or "").replace("_", " ").strip()
    # "Line 8 (Shanghai Metro)" -> "Shanghai Metro Line 8"
    m = re.fullmatch(r"(Line \S+) \(([^)]*(?:Metro|Subway|Rail Transit|Tram|MTR|LRT)[^)]*)\)", s)
    if m:
        s = f"{m.group(2)} {m.group(1)}"
    return s


def ok_line_label(s, line, cc=""):
    if not latin_label(s) or re.match(r"(?:Category|Template|File):", s):
        return False
    t = s.strip()
    # A label in lower case is a class, not a name ("limited express of Meitetsu").
    if t[:1].islower():
        return False
    # A Japanese named train tagged with its line's item: つがる is no "Ohu North Line".
    if cc == "jp" and re.search(r"\bLine\b", t, re.I) and not re.search(
            r"線|ライン|ケーブル|モノレール|鉄道|系統", line.get("name") or ""):
        return False
    # A bare route number, or the ref, is no name (Bulgarian line items are labelled "8").
    if re.fullmatch(r"[\dA-Za-z]{0,4}", t) or t == (line.get("ref") or "").strip():
        return False
    # "Line 46" for Трамвай 46 says less than the app's own "Tram 46" (englishLine).
    if re.fullmatch(r"(?:Line|Route|Tram|Tram line|Tram route|Bus|Trolleybus)\s*(?:No\.?\s*)?\S{1,4}",
                    t, re.I):
        return False
    # A train or route number in the local name has to survive: "Express" for Скорый поезд
    # 003/004 «Экспресс» drops what the app's own "Fast train 003/004" keeps.
    nums = {int(x) for x in re.findall(r"\d+", line.get("name") or "")}
    if nums and not nums & {int(x) for x in re.findall(r"\d+", t)}:
        return False
    return True


try:
    sys.path.insert(0, str(ROOT))
    from ru_register import en_score as _ru_en_score   # read-only use of the ru check
except Exception:        # pragma: no cover
    _ru_en_score = None

UK_BE = str.maketrans({"і": "и", "ї": "и", "є": "е", "ґ": "г", "ў": "у", "І": "И", "Ї": "И",
                       "Є": "Е", "Ґ": "Г", "Ў": "У", "ј": "й", "Ј": "Й", "ђ": "дж", "ћ": "ч",
                       "љ": "ль", "њ": "нь", "џ": "дж", "қ": "к", "ғ": "г", "ү": "у", "ұ": "у",
                       "ә": "а", "ө": "о", "һ": "х", "ң": "н", "ѓ": "г", "ќ": "к", "ѕ": "з"})


def cyr_score(local, en, cc=""):
    if _ru_en_score is None:
        return 1.0
    # Serbian Latin and Belarusian Łacinka labels ("Horgoš", "Valožyn") are compared plain.
    # Its NOT_EN test turns down j and q, which Serbian, Łacinka and Kazakh Latin use.
    en = strip_accents(en.replace("ł", "l").replace("Ł", "L"))
    # Not in Russia, Ukraine or Bulgaria, where a j is German or Dutch spelling ("Uzkij").
    if cc in ("rs", "mk", "ba", "me", "by", "kz", "kg", "md"):
        en = en.replace("j", "y").replace("J", "Y").replace("q", "k").replace("Q", "K")
    # Street words, translated in English stop names ("Sadova Street" for Садова вулиця).
    en = re.sub(r"\b(?:street|st|square|sq|avenue|ave|boulevard|blvd|lane|vul|ul|pl|prosp)\b\.?",
                " ", en, flags=re.I).strip()
    a = re.sub(r"(?i)(?<!\w)(?:вул|вулиця|улица|ул|площа|площадь|плошча|пл|проспект|просп|пр-т|"
               r"бульвар|б-р|провулок|переулок|пер)(?!\w)\.?", " ", local).strip()
    if not en or not a:
        return 0.0
    a = a.translate(UK_BE)
    best = _ru_en_score(a, en)
    # Ukrainian and Belarusian г is h; Bulgarian ъ is a.
    if "г" in a.lower():
        b = re.sub("г", "х", re.sub("Г", "Х", a))
        best = max(best, _ru_en_score(b, en))
    if "ъ" in a:
        best = max(best, _ru_en_score(a.replace("ъ", "а"), en))
    return best


GREEK_T = {"α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th",
           "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
           "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
           "ω": "o"}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def squash(s):
    return re.sub(r"[^a-z0-9]", "", strip_accents(s.lower()))


def greek_score(local, en):
    s = strip_accents(local.lower())
    for a, b in (("ου", "u"), ("μπ", "b"), ("ντ", "d"), ("γγ", "ng"), ("γκ", "g"), ("αυ", "av"),
                 ("ευ", "ev")):
        s = s.replace(a, b)
    a = "".join(GREEK_T.get(c, c) for c in s)
    b = strip_accents(en.lower()).replace("ou", "u").replace("y", "i").replace("h", "")
    a = a.replace("y", "i").replace("h", "")
    return SequenceMatcher(None, squash(a), squash(b)).ratio()


DIRS = {"north": "bei", "south": "nan", "east": "dong", "west": "xi"}


def han_score(local, en):
    py = squash("".join(pinyin_syllables(local)))
    b = strip_accents(en.lower())
    for k, v in DIRS.items():
        b = re.sub(rf"\b{k}\b", v, b)
    first = squash(b.split()[0]) if b.split() else ""
    # "Jiangwan Stadium", "Yindu Road": the first word is the pinyin of the name's start.
    if first and len(first) >= 4 and SequenceMatcher(None, first, py[:len(first)]).ratio() >= 0.8:
        return 1.0
    return SequenceMatcher(None, py, squash(b)).ratio()


def station_label_ok(local, en, cc):
    """Is a cleaned label a name for this station? Returns (ok, score)."""
    if not en or foreign(en) or not re.search(r"[A-Za-z]", en) or ST_NOT.search(en):
        return False, 0.0
    if CYR.search(local):
        sc = cyr_score(local, en, cc)
        return sc >= 0.75, sc
    if GREEK.search(local):
        sc = greek_score(local, en)
        return sc >= 0.6, sc
    if cc in ("cn", "tw", "hk") and HAN.search(local):
        sc = han_score(local, en)
        return sc >= 0.6, sc
    return True, 1.0


# ---------------------------------------------------------------- Wikidata

def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text("utf-8"))
    return {}


def save_cache(cache):
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True, indent=0), "utf-8")
    tmp.replace(CACHE)


ENDPOINTS = ["https://query.wikidata.org/sparql", "https://qlever.cs.uni-freiburg.de/api/wikidata"]
SPARQL = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT ?item ?l ?wp WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?l . FILTER(LANG(?l) = "en" || LANG(?l) = "en-gb" || LANG(?l) = "mul") }
  OPTIONAL { ?wp schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> }
}"""


def sparql(endpoint, q):
    data = urllib.parse.urlencode({"query": q}).encode()
    req = urllib.request.Request(endpoint, data=data, headers={
        "User-Agent": UA, "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_labels(qids, cache, batch=250):
    todo = sorted({q for q in qids if q not in cache and re.fullmatch(r"Q\d+", q)},
                  key=lambda q: int(q[1:]))
    log(f"wikidata: {len(todo)} ids to fetch ({len(qids)} wanted)")
    ep, fails = 0, 0
    i = 0
    while i < len(todo):
        chunk = todo[i:i + batch]
        q = SPARQL % " ".join(f"wd:{x}" for x in chunk)
        try:
            res = sparql(ENDPOINTS[ep], q)
        except urllib.error.HTTPError as e:
            fails += 1
            wait = int(e.headers.get("Retry-After") or 0) or min(30 * fails, 300)
            log(f"  {ENDPOINTS[ep]}: HTTP {e.code}; waiting {wait}s")
            if fails >= 3 and ep == 0:
                ep, fails = 1, 0
                log("  switching to the QLever mirror")
            time.sleep(wait)
            continue
        except Exception as e:      # timeouts, resets
            fails += 1
            log(f"  {ENDPOINTS[ep]}: {e!r}; waiting {10 * fails}s")
            if fails >= 3 and ep == 0:
                ep, fails = 1, 0
            if fails >= 6:
                log("  giving up for now; the cache keeps what came")
                break
            time.sleep(10 * fails)
            continue
        fails = 0
        got = {x: {} for x in chunk}
        for row in res["results"]["bindings"]:
            qid = row["item"]["value"].rsplit("/", 1)[-1]
            d = got.setdefault(qid, {})
            if "l" in row:
                lang = row["l"].get("xml:lang") or "en"
                d[lang] = row["l"]["value"]
            if "wp" in row:
                d["wp"] = urllib.parse.unquote(row["wp"]["value"].rsplit("/", 1)[-1]).replace("_", " ")
        cache.update(got)
        i += batch
        if (i // batch) % 10 == 0:
            save_cache(cache)
        log(f"  {min(i, len(todo))}/{len(todo)}")
        time.sleep(2.5)
    save_cache(cache)


def wd_label(cache, qid):
    d = cache.get(qid) or {}
    for k in ("en", "en-gb"):
        if d.get(k):
            return d[k]
    if d.get("wp"):
        return re.sub(r"\s*\([^)]*\)$", "", d["wp"])
    if d.get("mul") and latin_label(d["mul"]):
        return d["mul"]
    return None


# ---------------------------------------------------------------- pinyin

_PY = None


def pinyin_syllables(s):
    global _PY
    if _PY is None:
        from pypinyin import lazy_pinyin, Style
        _PY = (lazy_pinyin, Style)
    lazy_pinyin, Style = _PY
    return lazy_pinyin(s, style=Style.NORMAL, errors="default", v_to_u=False)


def py_word(chars):
    """Characters to one word, toneless, apostrophe before a syllable that starts with a vowel
    ("Chang'an"); ü as u (as on signs)."""
    syl = [x.replace("v", "u").replace("ü", "u") for x in pinyin_syllables(chars)]
    w = ""
    for i, x in enumerate(syl):
        if i and x and x[0] in "aoe":
            w += "'"
        w += x
    return w[:1].upper() + w[1:]


_JIEBA = None


def py_place(chars):
    """A place name, a word per jieba word: 虎丘湿地公园 -> Huqiu Shidi Gongyuan, 市博物馆 ->
    Shi Bowuguan. A single character joins the word before it (宝山南路 -> Baoshan Nanlu,
    长安西 -> Chang'anxi), and up to three characters is one word, as the shipped names are.
    Polyphones are read on the whole name (重庆 Chongqing)."""
    global _JIEBA
    if len(chars) <= 3:
        return py_word(chars)
    if _JIEBA is None:
        import logging
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _JIEBA = jieba
    toks = _JIEBA.lcut(chars)
    words = []
    for t in toks:
        if words and (len(t) == 1 or len(words[-1]) == 1):
            words[-1] += t
        else:
            words.append(t)
    # Pinyin of the whole name, cut per word, so a word's reading keeps its context.
    syl = [x.replace("v", "u").replace("ü", "u") for x in pinyin_syllables(chars)]
    if len(syl) != len(chars):
        return " ".join(py_word(w) for w in words)
    out, i = [], 0
    for w in words:
        part = syl[i:i + len(w)]
        i += len(w)
        s = ""
        for j, x in enumerate(part):
            if j and x and x[0] in "aoe":
                s += "'"
            s += x
        out.append(s[:1].upper() + s[1:])
    return " ".join(out)


# Chinese words in line names, longest first; matched anywhere.
ZH_WORDS = [
    ("轨道交通", "Rail Transit"), ("有轨电车", "Tram"), ("城际铁路", "Intercity Railway"),
    ("客运专线", "Passenger Railway"), ("客专线", "Passenger Railway"), ("高速铁路", "High-speed Railway"),
    ("联络线", "Link Line"), ("环线", "Loop Line"), ("支线", "Branch Line"), ("快线", "Express Line"),
    ("城际", "Intercity"), ("地铁", "Metro"), ("铁路", "Railway"), ("高铁", "High-speed Railway"),
    ("机场", "Airport"), ("磁浮", "Maglev"), ("单轨", "Monorail"), ("捷运", "MRT"),
    ("高速线", "High-speed Line"), ("高速", "High-speed"), ("市郊铁路", "Suburban Railway"), ("缆车", "Cable Car"), ("索道", "Ropeway"),
    ("下行", "down"), ("上行", "up"),
    ("高鐵", "High-speed Rail"), ("區間快", "Fast Local"), ("區間", "Local"), ("自強", "Tze-Chiang"),
    ("莒光", "Chu-Kuang"), ("太魯閣", "Taroko"), ("普悠瑪", "Puyuma"), ("台灣", "Taiwan"),
    ("臺灣", "Taiwan"), ("台北", "Taipei"), ("臺北", "Taipei"), ("捷運", "MRT"),
    ("线", "Line"), ("線", "Line"),
]


def zh_line(name, st_en):
    """A Chinese line name in English words and pinyin, or None when nothing is Han."""
    s = name.strip().replace("（", " (").replace("）", ")")
    s = re.sub(r"(\d+)\s*[号號]\s*[线線]", r" Line \1 ", s)
    s = re.sub(r"([A-Za-z]\d*)\s*[线線]", r" Line \1 ", s)
    s = s.replace("->", " - ").replace("→", " - ").replace("：", ": ").replace("·", " ")
    for zh, en in ZH_WORDS:
        s = s.replace(zh, f" {en} ")
    # What is left in Han is place names: a station we have in English, else pinyin.
    def place(m):
        t = m.group(0)
        if t in st_en:
            return f" {st_en[t]} "
        return f" {py_place(t)} "
    s = re.sub(r"[㐀-鿿]+", place, s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([:,)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    return s if latin_label(s) else None


def zh_station(name):
    s = re.sub(r"(?:火车站|地铁站|站)$", "", name.strip())
    s = s.replace("（", "(").replace("）", ")")
    m = re.fullmatch(r"([^()]*)\(([^)]*)\)", s)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        out = f"{zh_any(a)} ({zh_any(b)})"
    else:
        out = zh_any(s)
    return out if latin_label(out) else None


def zh_any(s):
    if s in ("在建", "建设中", "规划"):
        return "under construction" if s != "规划" else "planned"
    def place(m):
        return f" {py_place(m.group(0))} "
    t = re.sub(r"[㐀-鿿]+", place, s)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- Korean

L_RR = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t",
        "p", "h"]
V_RR = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo",
        "we", "wi", "yu", "eu", "ui", "i"]
T_RR = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p",
        "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]
# Final consonant (jamo index) carried onto a following ㅇ-initial syllable.
T_CARRY = {1: "g", 2: "kk", 4: "n", 7: "d", 8: "r", 16: "m", 17: "b", 19: "s", 20: "ss",
           22: "j", 23: "ch", 24: "k", 25: "t", 26: "p", 27: "h"}


def rr_word(word):
    """Revised Romanization of a Hangul word, with the commonest sound changes (a final
    consonant before ㅇ moves over; ㄴ+ㄹ and ㄹ+ㄴ = ll; final + ㄴ/ㅁ nasalises)."""
    syl = []
    for ch in word:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            syl.append([o // 588, (o % 588) // 28, o % 28])
        else:
            syl.append(ch)
    out = []
    for i, s in enumerate(syl):
        if isinstance(s, str):
            out.append(s)
            continue
        l, v, t = s
        nxt = syl[i + 1] if i + 1 < len(syl) and not isinstance(syl[i + 1], str) else None
        prev_t = syl[i - 1][2] if i and not isinstance(syl[i - 1], str) else None
        ini = L_RR[l]
        if i and isinstance(syl[i - 1], list):
            if l == 11 and prev_t in T_CARRY:
                ini = T_CARRY[prev_t]
            elif l == 5 and prev_t in (4, 8):        # ㄴ/ㄹ + ㄹ
                ini = "l"
            elif l == 5 and prev_t:
                ini = "n"
            elif l == 2 and prev_t == 8:              # ㄹ + ㄴ
                ini = "l"
        fin = T_RR[t]
        if nxt is not None:
            nl = nxt[0]
            if nl == 11 and t in T_CARRY:
                fin = ""
            elif t == 4 and nl == 5:
                fin = "l"
            elif nl in (2, 5, 6) and fin == "k":
                fin = "ng"
            elif nl in (2, 5, 6) and fin == "t":
                fin = "n"
            elif nl in (2, 5, 6) and fin == "p":
                fin = "m"
        out.append(ini + V_RR[v] + fin)
    w = "".join(out)
    return w[:1].upper() + w[1:]


KO_WORDS = [
    ("수도권광역급행철도", "GTX"), ("도시철도", "Metro"), ("광역철도", "Commuter Rail"),
    ("경전철", "LRT"), ("지하철", "Subway"), ("고속선", "High-speed Line"), ("고속철도", "High-speed Railway"),
    ("기지선", "Depot Line"), ("연결선", "Connecting Line"), ("지선", "Branch Line"), ("복선전철", "Double-track Line"),
    ("에이선", "Line A"), ("비선", "Line B"), ("씨선", "Line C"),
    ("선", "Line"),
]
KO_DROP = {"노선"}      # "route", as in 경전선 노선


def ko_name(name, st_en, line=True):
    s = name.strip()
    s = re.sub(r"(\d+)\s*호선", r" Line \1 ", s)
    s = s.replace("~", " - ").replace("～", " - ").replace("(", " (")
    words = []
    for w in re.split(r"(\s+|[()\-–:,])", s):
        if w in KO_DROP:
            continue
        if not w or not HANGUL.search(w):
            words.append(w)
            continue
        words.append(ko_word(w, st_en, line))
    t = re.sub(r"\s+", " ", "".join(words)).strip()
    return t if latin_label(t) else None


KO_EXACT = {"인천국제공항": "Incheon International Airport", "셔틀트레인": "Shuttle Train",
            "오렌지": "Orange", "블루": "Blue", "라인": "Line", "노선": ""}


def ko_word(w, st_en, line):
    if w in KO_EXACT:
        return KO_EXACT[w]
    if w in st_en:
        return st_en[w]
    # Two stations run together, as line names are made: 수서평택 -> Suseo–Pyeongtaek.
    for i in range(2, len(w) - 1):
        if w[:i] in st_en and w[i:] in st_en:
            return f"{st_en[w[:i]]}–{st_en[w[i:]]}"
    if line:
        for ko, en in KO_WORDS:
            if w.endswith(ko) and len(w) > len(ko):
                stem = w[: -len(ko)]
                return f"{ko_word(stem, st_en, line)} {en}"
            if w == ko:
                return en
    return rr_word(w)


# ---------------------------------------------------------------- Japanese line stems

def jp_stem_line(name, st_en):
    """<station>線 / <station>本線 / <station>支線 from the station's English name."""
    s = name.strip()
    for suf, en in (("本線", "Main Line"), ("支線", "Branch Line"), ("線", "Line")):
        if s.endswith(suf) and len(s) > len(suf):
            stem = s[: -len(suf)]
            if stem.endswith("新幹"):
                return None
            # "Narita Airport Terminal 1" is a station's name, not the line's stem.
            if stem in st_en and not re.search(r"\d|Terminal", st_en[stem]):
                return f"{st_en[stem]} {en}"
            return None
    return None


# ---------------------------------------------------------------- kana -> Hepburn

KANA_BASE = dict(zip(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゔ",
    "a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho "
    "ma mi mu me mo ya yu yo ra ri ru re ro wa i e o n ga gi gu ge go za ji zu ze zo da ji zu "
    "de do ba bi bu be bo pa pi pu pe po a i u e o vu".split()))
SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
MACRON = {"a": "ā", "i": "ī", "u": "ū", "e": "ē", "o": "ō"}
KANA_ONLY = re.compile(r"^[ぁ-ゖァ-ヺー・\s　]+$")


def hepburn(kana):
    """Modified Hepburn for a kana reading: しんおおさか -> Shin'ōsaka, とうきょう -> Tōkyō.
    Katakana is read as hiragana; ー lengthens the vowel before it; っ doubles the next
    consonant; ん before a vowel or y is n'. ou and oo are ō, uu is ū (the rule signs use;
    it misreads a few compounds across a word boundary, 小浦 こうら)."""
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana)
    words = re.split(r"[・\s　]+", s.strip())
    out = []
    for w in words:
        r, i, double = "", 0, False
        while i < len(w):
            c = w[i]
            if c == "っ":
                double = True
                i += 1
                continue
            if c == "ー":
                if r and r[-1] in MACRON:
                    r = r[:-1] + MACRON[r[-1]]
                i += 1
                continue
            syl = KANA_BASE.get(c)
            if syl is None:
                return None
            if i + 1 < len(w) and w[i + 1] in SMALL_Y and syl.endswith("i") and len(syl) > 1:
                v = SMALL_Y[w[i + 1]]
                syl = (syl[:-1] + v) if syl in ("shi", "chi", "ji") else (syl[:-1] + "y" + v)
                i += 1
            if syl == "n" and i + 1 < len(w) and (KANA_BASE.get(w[i + 1], "x")[0] in "aiueoy"):
                syl = "n'"
            if double and syl[0] not in "aiueon":
                syl = ("t" if syl.startswith("ch") else syl[0]) + syl
            double = False
            r += syl
            i += 1
        r = r.replace("ou", "ō").replace("oo", "ō").replace("uu", "ū")
        out.append(r[:1].upper() + r[1:])
    t = " ".join(x for x in out if x)
    return t or None


# Words of Japanese service names (OSM route masters of named trains and stopping patterns).
JP_WORDS = {"特急": "Limited Express", "直通特急": "Through Limited Express", "快速": "Rapid",
            "新快速": "Special Rapid", "特別快速": "Special Rapid", "普通": "Local", "各駅停車": "Local",
            "急行": "Express", "区間急行": "Semi-Express", "準急": "Semi-Express",
            "通勤快速": "Commuter Rapid", "エアポート": "Airport", "Train": "Train"}


def jp_service(name, st_en):
    """A Japanese service name made only of known words and stations: "普通 郡山<=>福島" ->
    "Local Kōriyama – Fukushima". None when any part would need a reading."""
    s = name.replace("<=>", " – ").replace("->", " – ").replace("⇔", " – ")
    out = []
    for tok in re.split(r"(\s+|–|・|•)", s):
        if not tok or tok.isspace() or tok in ("–", "・", "•"):
            out.append({"・": " / ", "•": " / "}.get(tok, tok))
            continue
        t = re.sub(r"^JR", "", tok)
        m = re.fullmatch(r"(.+?)(本線|線)", t)
        if t in JP_WORDS:
            out.append(JP_WORDS[t])
        elif re.sub(r"駅$", "", t) in st_en:
            out.append(st_en[re.sub(r"駅$", "", t)])
        elif m and m.group(1) in st_en:
            out.append(f"{st_en[m.group(1)]} {'Main Line' if m.group(2) == '本線' else 'Line'}")
        elif re.fullmatch(r"[ぁ-ゖー]+", t):          # a hiragana train name: つがる
            out.append(hepburn(t) or "")
        elif not foreign(t):
            out.append(t)
        else:
            return None
    r = re.sub(r"\s+", " ", "".join(out)).strip(" –")
    # "うずしお Uzushio" names the train twice once read; OSM's "Train 普通" is a local train.
    r = re.sub(r"\b(\w+) \1\b", r"\1", r, flags=re.I)
    r = re.sub(r"^Train (Local|Rapid|Express|Limited Express)$", r"\1 train", r)
    return r if latin_label(r) else None


# ---------------------------------------------------------------- Overpass: tags stops.pkl lacks

OSM_CACHE = Path(__file__).resolve().parent / "english_names_osm.json"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
AROUND_M = 1500
FR_CC = ("tn", "ma", "dz", "eg")
ROMAJI_KEYS = ("name:ja-Latn", "name:ja_rm", "name:ja-Rom", "name:ja_rom")
KANA_KEYS = ("name:ja-Hira", "name:ja_kana", "name:ja-Kana", "name:ja-Hrkt", "name:ja_hira")


def load_osm_cache():
    if OSM_CACHE.exists():
        return json.loads(OSM_CACHE.read_text("utf-8"))
    return {"node": {}, "rel": {}, "around": {}}


def save_osm_cache(c):
    tmp = OSM_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, ensure_ascii=False, sort_keys=True, indent=0), "utf-8")
    tmp.replace(OSM_CACHE)


def overpass(q):
    ep, fails = 0, 0
    while True:
        req = urllib.request.Request(OVERPASS[ep], data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=400) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            fails += 1
            code = getattr(e, "code", None)
            log(f"  overpass {OVERPASS[ep]}: {code or repr(e)}; try {fails}")
            if fails % 2 == 0:
                ep = 1 - ep
            if fails >= 6:
                raise
            time.sleep(30 * fails)


def fetch_overpass(targets, oc):
    """targets: {"node": ids, "rel": ids, "around": [(sid, name, lat, lon)]}."""
    for kind in ("node", "rel"):
        ids = sorted({int(i) for i in targets[kind]} - {int(k) for k in oc[kind]})
        log(f"overpass: {len(ids)} {kind}s")
        for i in range(0, len(ids), 800):
            chunk = ids[i:i + 800]
            q = f"[out:json][timeout:300];{kind}(id:{','.join(map(str, chunk))});out tags;"
            res = overpass(q)
            got = {str(x): {} for x in chunk}
            for el in res.get("elements", []):
                got[str(el["id"])] = el.get("tags", {})
            oc[kind].update(got)
            save_osm_cache(oc)
            time.sleep(5)
    todo = [t for t in targets["around"] if t[0] not in oc["around"]]
    log(f"overpass: {len(todo)} stations by name nearby")
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        parts = []
        for sid, name, lat, lon in chunk:
            for nm in {name, re.sub(r"駅$", "", name), re.sub(r"駅$", "", name) + "駅"}:
                esc = nm.replace("\\", "\\\\").replace('"', '\\"')
                parts.append(f'nwr(around:{AROUND_M},{lat},{lon})["name"="{esc}"];')
        q = f"[out:json][timeout:300];({''.join(parts)});out tags center;"
        res = overpass(q)
        found = {sid: [] for sid, *_ in chunk}
        for el in res.get("elements", []):
            tags = el.get("tags", {})
            if tags.get("highway") == "bus_stop" or (tags.get("bus") == "yes" and not any(
                    tags.get(k) == "yes" for k in ("train", "subway", "light_rail", "tram", "monorail"))):
                continue
            if not (tags.get("railway") or tags.get("public_transport") or tags.get("station")):
                continue
            c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
            if c.get("lat") is None:
                continue
            for sid, name, lat, lon in chunk:
                base = re.sub(r"駅$", "", name)
                if re.sub(r"駅$", "", tags.get("name", "")) == base and \
                        dist_m(lon, lat, c["lon"], c["lat"]) <= AROUND_M:
                    found[sid].append({**tags, "@id": f"{el['type'][0]}{el['id']}"})
        oc["around"].update(found)
        save_osm_cache(oc)
        log(f"  {min(i + 40, len(todo))}/{len(todo)}")
        time.sleep(5)


def clean_fr(s):
    s = re.sub(r"^(?:Gare(?: ferroviaire)?|Station|Halte)\s+(?:de\s+|d'|du\s+|des\s+)?", "", s.strip(),
               flags=re.I)
    return clean_station(s, osm=True)


def extra_station(cc, sid, s, cands, extra):
    """Second-tier station names from tags fetched by Overpass. Returns (source, name, detail)."""
    local = s["n"]
    full = []
    for kind, tags in cands:
        t = extra["node"].get(str(tags.get("@id")))
        if t:
            full.append((kind, t))
    full += [("around", t) for t in extra["around"].get(sid, [])]
    if not full:
        return None
    for kind, t in full:                                       # name:en
        en = (t.get("name:en") or "").strip()
        if en and en != local and latin_label(en):
            en2 = clean_station(en, osm=True) or en
            ok, score = station_label_ok(local, en2, cc)
            if ok or (kind == "own" and not CYR.search(local) and not GREEK.search(local)):
                return ("osm_en", en2, f"{kind} {t.get('@id', '')}")
    if cc == "jp":
        for kind, t in full:                                   # romaji tags
            for k in ROMAJI_KEYS:
                en = (t.get(k) or "").strip()
                if en and latin_label(en):
                    return ("romaji", clean_station(en, osm=True) or en, f"{kind} {k}")
        for kind, t in full:                                   # kana readings
            for k in KANA_KEYS:
                kana = (t.get(k) or "").strip()
                if kana and KANA_ONLY.match(kana):
                    en = hepburn(re.sub(r"(えき|エキ)$", "", kana))
                    if en:
                        return ("kana", en, f"{kind} {k} {kana}")
        # A name in kana alone needs no tag at all.
        if KANA_ONLY.match(local):
            en = hepburn(local)
            if en:
                return ("kana", en, "name")
    if cc in FR_CC:
        for kind, t in full:
            fr = (t.get("name:fr") or "").strip()
            if fr and latin_label(fr):
                return ("fr", clean_fr(fr) or fr, f"{kind}")
    for kind, t in full:                                       # int_name and Latin forms
        for k in sorted(t):
            if k in ("int_name", "name:latin") or re.fullmatch(r"name:[a-z]{2,3}-Latn", k):
                v = (t.get(k) or "").strip()
                if v and v != local and latin_label(v) and k != "name:ja-Latn":
                    return ("int_name", clean_station(v, osm=True) or v, f"{kind} {k}")
    return None


# ---------------------------------------------------------------- Japanese line items, all of them

JPL_CACHE = Path(__file__).resolve().parent / "english_names_jplines.json"
JP_OP_ALIASES = {"JR北海道": "北海道旅客鉄道", "JR東日本": "東日本旅客鉄道", "JR東海": "東海旅客鉄道",
                 "JR西日本": "西日本旅客鉄道", "JR四国": "四国旅客鉄道", "JR九州": "九州旅客鉄道",
                 "JR貨物": "日本貨物鉄道", "JR West": "西日本旅客鉄道", "JR Kyushu": "九州旅客鉄道",
                 "JR East": "東日本旅客鉄道", "JR Central": "東海旅客鉄道"}


def fetch_jp_lines():
    """Every Wikidata item in Japan of the classes the colour fetch found, with its Japanese
    labels and aliases, English label and operators' Japanese names."""
    types = set()
    for r in json.loads((RAW / "wikidata_colours_jp.json").read_text("utf-8")):
        if r.get("type"):
            types.add(r["type"]["value"].rsplit("/", 1)[-1])
    q = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?item ?ja ?en ?op WHERE {
  VALUES ?t { %s }
  ?item wdt:P31 ?t ; wdt:P17 wd:Q17 .
  { ?item rdfs:label ?ja } UNION { ?item skos:altLabel ?ja }
  FILTER(LANG(?ja) = "ja")
  OPTIONAL { ?item rdfs:label ?en . FILTER(LANG(?en) = "en") }
  OPTIONAL { ?item wdt:P137 ?o . ?o rdfs:label ?op . FILTER(LANG(?op) = "ja") }
}""" % " ".join(f"wd:{t}" for t in sorted(types))
    for ep in ENDPOINTS:
        try:
            res = sparql(ep, q)
            break
        except Exception as e:
            log(f"  {ep}: {e!r}")
            time.sleep(10)
    else:
        return None
    items = {}
    for row in res["results"]["bindings"]:
        qid = row["item"]["value"].rsplit("/", 1)[-1]
        d = items.setdefault(qid, {"ja": set(), "en": None, "op": set()})
        d["ja"].add(row["ja"]["value"])
        if "en" in row:
            d["en"] = row["en"]["value"]
        if "op" in row:
            d["op"].add(row["op"]["value"])
    out = {q: {"ja": sorted(d["ja"]), "en": d["en"], "op": sorted(d["op"])} for q, d in items.items()}
    JPL_CACHE.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=0), "utf-8")
    log(f"wikidata: {len(out)} Japanese line items")
    return out


def op_match(line_op, item_ops):
    ops = {JP_OP_ALIASES.get(o.strip(), o.strip()) for o in re.split(r"[;；]", line_op or "") if o.strip()}
    for a in ops:
        for b in item_ops:
            if a and b and (a == b or a in b or b in a):
                return True
    return False


def jp_line_wd(line, jpl):
    name = re.sub(r"^JR", "", (line.get("name") or "").strip())
    if not name or not jpl:
        return None
    variants = {name}
    if name.endswith("線") and not name.endswith("本線"):
        variants.add(name[:-1] + "本線")
    if name.endswith("本線"):
        variants.add(name[:-2] + "線")
    generic = lambda q: re.fullmatch(r"(?:Main |Branch )?Line", jpl[q]["en"] or "", re.I)
    exact = [q for q, d in jpl.items() if variants & set(d["ja"])]
    pool = exact
    if len(pool) > 1:
        pool = [q for q in pool if op_match(line.get("operator"), jpl[q]["op"])] or pool
    # A bare "Main Line" for 本線 says nothing; 京成本線 by the operator says what it is.
    pool = [q for q in pool if not generic(q)]
    if not pool:
        # 東武野田線 for N02's 野田線 (operator 東武鉄道): an item whose name ends in the line's and
        # whose operator is the line's.
        pool = [q for q, d in jpl.items()
                if any(j.endswith(name) and j != name for j in d["ja"])
                and op_match(line.get("operator"), d["op"])]
    pool = [q for q in pool if jpl[q]["en"] and not generic(q)]
    if len({jpl[q]["en"] for q in pool}) != 1:
        return None
    return pool[0], jpl[pool[0]]["en"]


def extra_line(cc, line, cands, extra, st_en, on_line):
    local = line["name"]
    for kind, tags in cands:                                  # full relation tags
        if kind == "member" or (kind == "folded" and tags.get("name") != local):
            continue
        t = extra["rel"].get(str(tags.get("@id"))) or {}
        keys = ["name:en"] + (["name:fr"] if cc in FR_CC else []) + list(ROMAJI_KEYS) + ["int_name"]
        for k in keys:
            v = (t.get(k) or "").strip()
            if v and v != local and ok_line_label(v, line, cc):
                return ("osm_full", v, f"{kind} {k}")
    if cc != "jp":
        return None
    hit = jp_line_wd(line, extra.get("jpl"))
    if hit and ok_line_label(clean_line(hit[1]), line, cc):
        return ("wd_ja", clean_line(hit[1]), hit[0])
    en = jp_stem_line(re.sub(r"^JR", "", local), on_line)
    if en:
        return ("stem", en, "with stations named by this pass")
    en = jp_service(local, st_en)
    if en and en != local:
        return ("service", en, "")
    if KANA_ONLY.match(local) and re.fullmatch(r"[ぁ-ゖー]+", local.strip()):
        en = hepburn(local)
        if en:
            return ("kana", en, "hiragana name")
    return None


def leftover_targets(plans, cache):
    """What the first pass leaves in non-Latin letters, and the OSM ids to ask Overpass for."""
    t = {"node": set(), "rel": set(), "around": []}
    for p in plans:
        out_st, out_ln, *_ = fill_country(p, cache)
        for sid, s in p["gs"].items():
            if sid in out_st or shown_latin(s["n"]):
                continue
            for _, tags in p["sc"].get(sid, []):
                if tags.get("@id"):
                    t["node"].add(tags["@id"])
            t["around"].append((sid, s["n"], s["y"], s["x"]))
        for lid, line in p["gl"].items():
            if lid in out_ln or shown_latin(line["name"], True):
                continue
            for _, tags in p["lc"].get(lid, []):
                if tags.get("@id"):
                    t["rel"].add(tags["@id"])
    return t


# ---------------------------------------------------------------- per country

def station_english(st):
    """local name -> its English, where the shipped stations agree on one."""
    votes = defaultdict(Counter)
    for v in st.values():
        n, e = v.get("n"), v.get("e")
        if n and e and not v.get("j"):
            e = re.sub(r"\s+(?:Station|station)$", "", e).strip()
            votes[re.sub(r"駅$|역$|站$", "", n)][e] += 1
    out = {}
    for n, c in votes.items():
        (e, k), = c.most_common(1)
        if k * 2 > sum(c.values()) or len(c) == 1:
            out[n] = e
    return out


def colour_items(cc):
    p = RAW / f"wikidata_colours_{cc}.json"
    if not p.exists():
        return {}
    by = defaultdict(lambda: defaultdict(set))
    for r in json.loads(p.read_text("utf-8")):
        lab = r.get("label", {}).get("value")
        qid = r["item"]["value"].rsplit("/", 1)[-1]
        op = r.get("opNative", {}).get("value", "")
        if lab:
            by[lab][qid].add(op)
    return by


def plan_country(cc, cache_only=True):
    """Everything the fill needs from local files: gaps, OSM candidates, the QIDs wanted."""
    st, ln, al = load_country(cc)
    gs, gl = gap_stations(st), gap_lines(ln)
    sc = station_candidates(cc, gs, al)
    lc = line_candidates(cc, gl, al)
    qids = set()
    for c in sc.values():
        qids |= {t["wikidata"] for _, t in c if t.get("wikidata")}
    for c in lc.values():
        qids |= {t["wikidata"] for _, t in c if t.get("wikidata")}
    byname = colour_items(cc) if cc in ("jp", "kr") else {}
    name_q = {}
    for lid, line in gl.items():
        items = byname.get(line["name"])
        if not items:
            continue
        if len(items) > 1:
            op = line.get("operator") or ""
            items = {q: ops for q, ops in items.items() if op and op in ops}
        if len(items) == 1:
            name_q[lid] = next(iter(items))
    qids |= set(name_q.values())
    return dict(cc=cc, st=st, ln=ln, gs=gs, gl=gl, sc=sc, lc=lc, name_q=name_q, qids=qids)


def fill_country(p, cache, extra=None):
    cc, st, gs, gl = p["cc"], p["st"], p["gs"], p["gl"]
    st_en = station_english(st)
    on_line = defaultdict(dict)       # line id -> {station name: English} of its stations
    for v in st.values():
        if v.get("n") and v.get("e") and not v.get("j"):
            n = re.sub(r"駅$|역$", "", v["n"])
            e = re.sub(r"\s+(?:Station|station)$", "", v["e"]).strip()
            for lid in v.get("l") or ():
                if lid in gl:
                    on_line[lid][n] = e
    out_st, out_ln, src = {}, {}, Counter()
    why = {}       # id -> (source, local, english, detail) for the spot-check
    rejected = []

    for sid, s in gs.items():
        local = s["n"]
        cands = p["sc"].get(sid, [])
        got = None
        # 1. OSM name:en
        for kind, tags in cands:
            en = (tags.get("name:en") or "").strip()
            if en and en != local and latin_label(en):
                en2 = clean_station(en, osm=True) or en
                ok, score = station_label_ok(local, en2, cc)
                # The station's own node: its name:en is no borrowed label ("Matsuyama City").
                if not ok and kind == "own" and not CYR.search(local) and not GREEK.search(local):
                    ok = latin_label(en2)
                if ok:
                    got = ("osm", en2, f"{kind} {score:.2f}")
                    break
                rejected.append(("st-osm", sid, local, en, f"{kind} {score:.2f}"))
        # 2. Wikidata label
        if not got:
            for kind, tags in cands:
                q = tags.get("wikidata")
                if not q:
                    continue
                lab = wd_label(cache, q)
                if not lab:
                    continue
                en = clean_station(lab)
                ok, score = station_label_ok(local, en, cc)
                # A Chinese station's own item may translate rather than romanise ("Peony
                # Square" for 牡丹广场); the pinyin check is for a QID borrowed from a stop
                # nearby. Cyrillic and Greek keep the check: "135 km" sat on Химфармзавод.
                if not ok and kind == "own" and cc in ("cn", "tw", "hk"):
                    ok, score = station_label_ok(local, en, "")
                if ok:
                    got = ("wd", en, f"{kind} {q} '{lab}' {score:.2f}")
                    break
                rejected.append(("st", sid, local, lab, f"{q} {score:.2f}"))
        # 3. Tags only Overpass has: name:en on nodes outside stops.pkl, Japanese romaji and
        #    kana readings, name:fr in the Maghreb, int_name (extra_station).
        if not got and extra:
            got = extra_station(cc, sid, s, cands, extra)
        # 4. pinyin
        if not got and HAN.search(local) and not KANA.search(local) and cc in ("cn", "tw", "hk"):
            en = zh_station(local)
            if en:
                got = ("pinyin", en, "")
        # 5. Revised Romanization
        if not got and HANGUL.search(local) and cc == "kr":
            en = ko_name(re.sub(r"역$", "", local), st_en, line=False)
            if en:
                got = ("rr", en, "")
        if got:
            out_st[sid] = got[1]
            src["st_" + got[0]] += 1
            why[sid] = ("st_" + got[0], local, got[1], got[2])
            # A Japanese station named just now can name its line: 宗谷線 from 宗谷's romaji.
            if cc == "jp" and got[0] != "pinyin":
                n = re.sub(r"駅$", "", local)
                for lid in s.get("l") or ():
                    if lid in gl and n not in on_line[lid]:
                        on_line[lid][n] = got[1]

    for lid, line in gl.items():
        local = line["name"]
        cands = p["lc"].get(lid, [])
        got = None
        for kind, tags in cands:
            # A member route's name:en names one service ("Line 1: A -> B"); a folded
            # relation's counts only under the line's own name.
            if kind == "member" or (kind == "folded" and tags.get("name") != local):
                continue
            en = (tags.get("name:en") or "").strip()
            if en and en != local and ok_line_label(en, line, cc):
                got = ("osm", en, kind)
                break
        if not got:
            qs = []
            for kind, tags in cands:
                q = tags.get("wikidata")
                if q and (kind, q) not in qs:
                    qs.append((kind, q))
            # a route_master's member routes count only when they agree on one item
            mem = {q for k, q in qs if k == "member"}
            fold = {q for k, q in qs if k == "folded"}
            order = [q for k, q in qs if k == "own"]
            if len(mem) == 1:
                order += list(mem)
            if len(fold) == 1:
                order += list(fold)
            for q in order:
                lab = wd_label(cache, q)
                if lab and ok_line_label(clean_line(lab), line, cc):
                    got = ("wd", clean_line(lab), q)
                    break
                if lab:
                    rejected.append(("ln", lid, local, lab, q))
        if not got and lid in p["name_q"]:
            q = p["name_q"][lid]
            lab = wd_label(cache, q)
            if lab and ok_line_label(clean_line(lab), line, cc):
                got = ("wdname", clean_line(lab), q)
        # The stem has to be a station ON this line: 北条線 is Hōjō, not the Kitajō in Niigata.
        if not got and cc == "jp":
            en = jp_stem_line(local, on_line.get(lid, {}))
            if en:
                got = ("stem", en, "")
        if not got and extra:
            got = extra_line(cc, line, cands, extra, st_en, on_line.get(lid, {}))
        if not got and cc == "kr":
            s = local.strip()
            ol = on_line.get(lid, {})
            if s.endswith("선") and s[:-1] in ol and not re.search(r"\d", ol[s[:-1]]):
                got = ("stem", f"{ol[s[:-1]]} Line", "")
        if not got and HAN.search(local) and not KANA.search(local) and cc in ("cn", "tw", "hk"):
            en = zh_line(local, st_en)
            if en:
                got = ("pinyin", en, "")
        if not got and HANGUL.search(local) and cc == "kr":
            en = ko_name(local, st_en)
            if en:
                got = ("rr", en, "")
        if got:
            out_ln[lid] = got[1]
            src["ln_" + got[0]] += 1
            why[lid] = ("ln_" + got[0], local, got[1], got[2])
    return out_st, out_ln, dict(sorted(src.items())), why, rejected


# ---------------------------------------------------------------- coverage

GENERIC_LINE = [re.compile(r"^(?:Ligne|Linea|Línea|Linie|Linia|Linha|Lijn|Línia|Linija|Линия|Лінія)\s+(.+)$", re.I),
                re.compile(r"^(?:地铁|轨道交通)?(\d+)\s*号线$")]


def shown_latin(name, line=False):
    """Would the app show this in Latin letters in English mode with no English name? It
    transliterates Cyrillic, Greek, Georgian and Armenian by table, and N号线 as Line N."""
    if line and any(r.match(name) for r in GENERIC_LINE[1:]):
        return True
    rest = TABLE_SCRIPT.sub("", name)
    return not foreign(rest)


def coverage(p, out_st, out_ln):
    st, ln = p["st"], p["ln"]
    r = Counter()
    for sid, v in st.items():
        n = v.get("n")
        if not n or not foreign(n) or v.get("j"):
            continue
        r["st"] += 1
        has_en = bool(v.get("e"))
        r["st_en_before"] += has_en
        r["st_en_after"] += has_en or sid in out_st
        r["st_latin_before"] += has_en or shown_latin(n)
        r["st_latin_after"] += has_en or sid in out_st or shown_latin(n)
    for x in ln:
        n = x.get("name")
        if not n or not foreign(n):
            continue
        r["ln"] += 1
        has_en = bool(x.get("name_en"))
        r["ln_en_before"] += has_en
        r["ln_en_after"] += has_en or x["id"] in out_ln
        r["ln_latin_before"] += has_en or shown_latin(n, True)
        r["ln_latin_after"] += has_en or x["id"] in out_ln or shown_latin(n, True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cc", nargs="*")
    ap.add_argument("--fetch", action="store_true", help="ask Wikidata for labels not cached")
    ap.add_argument("--overpass", action="store_true",
                    help="ask Overpass for the full tags of what the first pass leaves")
    ap.add_argument("--sample", type=int, default=0, help="print N random fills per source")
    ap.add_argument("--dry", action="store_true", help="write nothing but the label cache")
    ap.add_argument("--report", help="write the per-country numbers and samples to this JSON file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ccs = a.cc or sorted(d.name for d in DATA.iterdir() if d.is_dir() and (d / "stations.json").exists())
    plans = []
    for cc in ccs:
        p = plan_country(cc)
        log(f"{cc}: {len(p['gs'])} gap stations ({len(p['sc'])} with OSM stops), "
            f"{len(p['gl'])} gap lines ({len(p['lc'])} with relations), {len(p['qids'])} QIDs")
        plans.append(p)
    cache = load_cache()
    if a.fetch:
        fetch_labels(set().union(*[p["qids"] for p in plans]) if plans else set(), cache)
        if not JPL_CACHE.exists():
            fetch_jp_lines()
    oc = load_osm_cache()
    if a.overpass:
        fetch_overpass(leftover_targets(plans, cache), oc)
    extra = {**oc, "jpl": json.loads(JPL_CACHE.read_text("utf-8")) if JPL_CACHE.exists() else {}}
    allnames = {}
    report = {}
    random.seed(7)
    by_src = defaultdict(list)
    all_rej = []
    for p in plans:
        cc = p["cc"]
        out_st, out_ln, src, why, rej = fill_country(p, cache, extra)
        cov = coverage(p, out_st, out_ln)
        report[cc] = {"src": src, "cov": dict(cov)}
        for k, w in why.items():
            by_src[w[0]].append((cc, k) + w[1:])
        all_rej += [(cc,) + r for r in rej]
        doc = {"st": dict(sorted(out_st.items())), "ln": dict(sorted(out_ln.items())), "src": src}
        if out_st or out_ln:
            allnames[cc] = {"st": doc["st"], "ln": doc["ln"]}
        if not a.dry:
            tmp = DATA / cc / "names_en.json.tmp"
            tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), "utf-8")
            tmp.replace(DATA / cc / "names_en.json")
        if out_st or out_ln:
            c = cov
            log(f"{cc}: +{len(out_st)} st, +{len(out_ln)} ln {src} | stations English "
                f"{c['st_en_before']}->{c['st_en_after']} of {c['st']}, Latin "
                f"{c['st_latin_before']}->{c['st_latin_after']}; lines English "
                f"{c['ln_en_before']}->{c['ln_en_after']} of {c['ln']}, Latin "
                f"{c['ln_latin_before']}->{c['ln_latin_after']}")
    if not a.dry and not a.cc:
        tmp = DATA / "names_en.json.tmp"
        tmp.write_text(json.dumps(allnames, ensure_ascii=False, separators=(",", ":"), sort_keys=True), "utf-8")
        tmp.replace(DATA / "names_en.json")
    if a.sample:
        for s, rows in sorted(by_src.items()):
            log(f"\n== {s}: {len(rows)}")
            for r in random.sample(rows, min(a.sample, len(rows))):
                log("  ", r)
        log(f"\n== rejected Wikidata labels: {len(all_rej)}")
        for r in random.sample(all_rej, min(a.sample * 2, len(all_rej))):
            log("  ", r)
    if a.report:
        Path(a.report).write_text(json.dumps({"report": report, "by_src": by_src, "rejected": all_rej},
                                             ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
