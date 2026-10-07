"""Indonesia: the lines id.wikipedia's line articles list station by station, with KAI's km posts,
written into rinf.py's input format (as in_register.py and ru_register.py do) and traced over
OSM track by rinf.py.

    python id_register.py --fetch        # id.wikipedia: the line articles and station points
    python extract.py --region id --pbf data/raw/indonesia-latest.osm.pbf     (managing session)
    python id_register.py --convert      # data/raw/id/{sections,points,names}.json
    python build_model.py --region id --register id_register:data/raw/id
    python id_register.py --report       # the register without OSM: lines, km, checks

`--register id_register:data/raw/id` converts and then runs rinf.build on the result.
rinf_countries/id.py holds the settings rinf.py reads, id_sources.md the sources, the numbers and
the decisions.

THE REGISTER UNIT is the line as id.wikipedia writes an article about it ("Jalur kereta api
Cikampek–Cirebon–Kroya", "Jalur kereta api Solo Balapan–Wonokromo", "Jalur kereta api
Prabumulih–Panjang"): the lines the companies of the Dutch period built, which DJKA and KAI
still number and measure as one chainage each (KAI's km posts run Jakarta - Cikampek -
Cirebon - Kroya without a break). The articles that "Daftar jalur kereta api aktif di
Indonesia" links are the active network; together they partition it, as India's sections do.

WHAT AN ARTICLE GIVES. A station table of {{DaftarStasiun}} rows in line order: the station's
name (its article is "Stasiun <nama>"), KAI code (`singkatan`), `status` (Beroperasi: open for
trains to call; Tidak beroperasi: a closed halt or a closed stretch), and `letak`, the km post
("km 219+168"), one per chainage the station lies on ("km 293+937 lintas Jakarta-Kroya<br>km
38+500 lintas Tegal-Prupuk"). Branches follow under their own headings ("Percabangan menuju
YIA"); stretches long closed under headings saying so ("Segmen lama", "Jalur historis",
"nonaktif"), which are skipped.

SECTIONS are consecutive rows. Their km is the difference of the two stations' km posts on a
chainage they share: of every pair of posts, the difference that best fits the crow-fly
distance (KM_FIT). Rows across a heading or a new table are joined only when such a pair fits;
that is what tells a branch's first station from the main line's last. Where no pair fits (a
post missing or on another chainage), the section's km is the crow-fly x CROW_FACTOR, marked
"crow" in sections.json.

WHICH STRETCHES RUN. A chain's ends are trimmed back to its first and last station that is
`Beroperasi`, so a line runs only as far as a station trains call at (Cilacap Pelabuhan, Padang
Panjang, Garut - Cikajang drop). Stretches closed between two open stations, and freight-only
lines, are listed by hand in INACTIVE and FREIGHT with the article's own words as the reason.
A station is a stop when an article lists it Beroperasi and it is not freight-only (FREIGHT_
STATIONS); rinf.py then makes it a stop only if an OSM station of its name is near.

STATION POINTS: the OSM station node carrying the station's KAI code in `ref` (after the
extract), else id.wikipedia's coordinate for "Stasiun <nama>", else none (rinf.py places it at
the OSM station of its name, or traces past it).

The `path` argument is data/raw/id; the OSM half is read from data/proc/id (extract.py).
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "id"
PROC = ROOT / "data" / "proc" / "id"
API = "https://id.wikipedia.org/w/api.php"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
LIST_PAGE = "Daftar jalur kereta api aktif di Indonesia"
PREFIX = "Jalur kereta api "

# Line articles the active list does not link but whose lines run.
EXTRA_ARTICLES = ["Jalur kereta api Makassar–Parepare"]

CROW_FACTOR = 1.15        # track km per crow-fly km where no km posts fit
# posts fit if 0.92 x crow - 0.4 <= diff <= 3 x crow + 1.5: track is never shorter than the crow
# flies, but can wind (Batu Ceper - Bandara Soekarno-Hatta is 12.3 km of track for 5.2 crow-fly)
KM_FIT = (0.92, 0.4, 3.0, 1.5)
OSM_NAME_KM = 15          # OSM's station of a row's exact name is taken this far from the wiki's point
SAME_CODE_KM = 5         # two rows of one KAI code this close are one station, whatever the name
MAX_HOP_KM = 30          # two rows further apart than this (crow-fly) are never one section
OSM_REF_M = 3000         # an OSM station carrying a KAI code may be this far from id.wikipedia's point
INACTIVE_HEAD = re.compile(r"\blama\b|historis|nonaktif|non-aktif|tidak aktif|ditutup|bekas|"
                           r"rencana|usulan|konstruksi|pembangunan", re.I)
OPEN = {"beroperasi", "aktif beroperasi", "beroperasi terbatas"}

# Stretches closed between two open stations, by line article and the two stations bounding the
# closed run (both kept as stations of the line where they are open; the sections between them
# dropped). Each from the article's own infobox `status` or text.
INACTIVE = {
    # infobox: "Tidak beroperasi (segmen Cipatat–Padalarang)"; the Siliwangi turns at Cipatat
    "Bogor–Padalarang–Kasugihan": [("Cipatat", "Padalarang")],
    # Muaro Kalaban - Sawahlunto: the Mak Itam steam train's occasional tourist runs, no
    # timetable (the rest of the line, Kayu Tanam - Muaro Kalaban, is closed)
    "Padang Panjang–Sawahlunto": [("*", "*")],
}
# Lines or stretches whose trains are freight only (no passenger train in the article's
# "Layanan kereta api" lists), by article and two stations; ("*", "*") for the whole line.
FREIGHT = {
    "Bukit Putus–Indarung": [("*", "*")],             # Semen Padang's cement trains
    "Makassar–Parepare": [("Labakkang", "Mangilu")],   # Semen Tonasa's branch
    "Lubuk Linggau–Prabumulih": [("Muara Enim", "Tanjung Enim Baru")],   # Bukit Asam's coal
    # Perlanaan - Sei Mangkei - Bandar Tinggi - Kuala Tanjung: the Sei Mangkei economic zone's
    # and Kuala Tanjung port's goods line; no passenger train in KAI's timetable or in OSM
    "Tebing Tinggi–Kisaran": [("Perlanaan", "Sei Mangkei"), ("Sei Mangkei", "Bandar Tinggi"),
                              ("Bandar Tinggi", "Kuala Tanjung")],
}
# Lines and branches no article lists station by station: (line, qid or None, operator, rows),
# each row (name, KAI code, [km posts], (lon, lat) or None for id.wikipedia's point).
HAND = [
    # Whoosh: chainage from en.wikipedia "Jakarta–Bandung high-speed railway" (Halim 0, Karawang
    # 41.17, Tegalluar Summarecon 142.80). Its Padalarang at 97.22 is not where the station is:
    # OSM's track gives Karawang - Padalarang 68.2 km and Padalarang - Tegalluar 31.8 (Padalarang
    # is west of Bandung, Tegalluar east), which add up to the published total; so Padalarang
    # takes no post and its two sections are measured crow-fly. Points are OSM's stations.
    ("Kereta Cepat Jakarta–Bandung", "Q85886652", "KCIC", [
        ("Halim", "HLM", [0.0], (106.88449, -6.24564)),
        ("Karawang", "KRW-KCIC", [41.17], (107.21899, -6.36468)),
        ("Padalarang", "PDL", [], None),
        ("Tegalluar Summarecon", "TGS", [142.80], (107.7147, -6.96381))]),
    # Kualanamu airport branch from Araskabu (opened 2013, the active list's "Percabangan menuju
    # Bandara Internasional Kualanamu"); the branch's chainage starts at Araskabu (km 0+000);
    # Kualanamu's post is not published, so the section is crow-fly x CROW_FACTOR.
    ("Medan–Tebing Tinggi", None, "KAI", [
        ("Araskabu", "ARB", [0.0], None),
        ("Kualanamu", "KNM", [], (98.87791, 3.63435))]),
]
# Stations listed Beroperasi that are freight stations, not stops (KAI code). A section ending at
# one is junction-ended, and build_model keeps it only where an OSM passenger route runs over it.
FREIGHT_STATIONS = {
    "JAKG",          # Jakarta Gudang, the goods station by Kampung Bandan
    "POO", "SAO",    # Pasoso, Sungai Lagoa: Tanjung Priok's container yards
    "KLM", "SBE",    # Kalimas (Surabaya's port), Benteng
    "SDT", "MST",    # Sidotopo (yard and depot), Mesigit (goods)
    "CGD",           # Cigading: Krakatau Steel's port branch from Krenceng
    "TMB",           # Tanjung Enim Baru: Bukit Asam's coal loading
    "IDR",           # Indarung (Semen Padang)
    "GR",            # Blokpos Garuntang, Tanjungkarang's goods line towards Panjang and Tarahan
}
# Station names as id.wikipedia writes them -> as OSM does, where the two differ by more than
# rinf.py's name match takes (an older spelling).
NAME_ALIAS = {"Tanjung Priok": "Tanjung Priuk"}


def log_print(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


# ================================================================ fetching

def api(params, tries=4):
    params = dict(params, format="json", formatversion="2", maxlag="5")
    body = urllib.parse.urlencode(params).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:                                  # noqa: BLE001
            print(f"  API attempt {k + 1} failed: {e}", flush=True)
            if k == tries - 1:
                raise
            time.sleep(30 * (k + 1))


def pages(titles, props, extra=None):
    """{requested title: page} for up to 50 titles a call, redirects followed."""
    out = {}
    titles = list(dict.fromkeys(titles))
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        d = api(dict(action="query", prop=props, titles="|".join(batch), redirects="1",
                     **(extra or {})))
        q = d.get("query", {})
        back = {t: t for t in batch}
        for n in q.get("normalized", []):
            back[n["to"]] = back.pop(n["from"], n["from"])
        for r in q.get("redirects", []):
            back[r["to"]] = back.get(r["from"], r["from"])
        for p in q.get("pages", []):
            out[back.get(p["title"], p["title"])] = p
        time.sleep(2)
    return out


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    lst = pages([LIST_PAGE], "revisions", {"rvprop": "content", "rvslots": "main"})[LIST_PAGE]
    text = lst["revisions"][0]["slots"]["main"]["content"]
    links = re.findall(r"\[\[(Jalur kereta api [^\]|#]+)", text)
    titles = list(dict.fromkeys(links + EXTRA_ARTICLES))
    got = pages(titles, "revisions|pageprops", {"rvprop": "content", "rvslots": "main"})
    arts = {}
    for t, p in got.items():
        if "revisions" in p:
            arts[t] = {"title": p["title"], "qid": p.get("pageprops", {}).get("wikibase_item"),
                       "text": p["revisions"][0]["slots"]["main"]["content"]}
    print(f"{len(arts)} line articles (of {len(titles)} linked)")
    names = set()
    for a in arts.values():
        for row in parse_rows(a["text"]):
            if row["kind"] == "sta":
                names.add(row["title"])
    coords = {}
    got = pages(sorted(names), "coordinates|pageprops", {"colimit": "max"})
    for t, p in got.items():
        c = (p.get("coordinates") or [None])[0]
        coords[t] = {"lon": c["lon"], "lat": c["lat"]} if c else {}
        if p.get("pageprops", {}).get("wikibase_item"):
            coords[t]["qid"] = p["pageprops"]["wikibase_item"]
        coords[t]["missing"] = bool(p.get("missing"))
    print(f"{len(coords)} station articles, {sum(1 for c in coords.values() if 'lon' in c)} "
          f"with a coordinate")
    stamp = {"source": "id.wikipedia.org (CC BY-SA 4.0)", "fetched": date.today().isoformat()}
    for fn, obj in (("idwiki_articles.json", {**stamp, "articles": arts, "list": text}),
                    ("idwiki_stations.json", {**stamp, "stations": coords})):
        tmp = RAW / (fn + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False), "utf-8")
        os.replace(tmp, RAW / fn)


# ================================================================ parsing the articles

def templates(text, i):
    """The balanced {{...}} starting at text[i]; returns (inner text, end index)."""
    depth, j = 0, i
    while j < len(text):
        if text.startswith("{{", j):
            depth += 1
            j += 2
        elif text.startswith("}}", j):
            depth -= 1
            j += 2
            if depth == 0:
                return text[i + 2:j - 2], j
        else:
            j += 1
    return text[i + 2:], len(text)


def split_params(inner):
    """Top-level |-separated params of a template body: {name: value}, positional by index."""
    parts, depth, cur, k = [], 0, [], 0
    while k < len(inner):
        c2 = inner[k:k + 2]
        if c2 in ("{{", "[["):
            depth += 1
            cur.append(c2)
            k += 2
            continue
        if c2 in ("}}", "]]"):
            depth -= 1
            cur.append(c2)
            k += 2
            continue
        if inner[k] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(inner[k])
        k += 1
    parts.append("".join(cur))
    out = {"_name": parts[0].strip()}
    for n, p in enumerate(parts[1:]):
        if "=" in p:
            key, val = p.split("=", 1)
            out[key.strip().lower()] = val.strip()
        else:
            out[str(n)] = p.strip()
    return out


def plain(s):
    """Wikitext to plain text: links to their label, templates dropped, comments gone."""
    s = re.sub(r"<!--.*?-->", "", s or "", flags=re.S)
    s = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{\s*(?:slk|sta|Sta)\s*\|(?:KAI\|)?([^}|]*)[^}]*\}\}", r"\1", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"<[^>]+>", " ", s).replace("'''", "").replace("''", "")
    return " ".join(s.split())


def station_title(nama):
    """id.wikipedia's article for a row: the template links "Stasiun <nama>"; a row written as
    a link ([[Stasiun Duku|Duku]]) names its article itself."""
    nama = re.sub(r"<!--.*?-->", "", nama or "", flags=re.S)
    m = re.search(r"\[\[([^\]|]+)", nama)
    if m:
        return m.group(1).strip()
    n = plain(re.split(r"<br\s*/?>", nama)[0])
    return f"Stasiun {n}" if n else ""


def station_name(nama):
    nama = re.sub(r"<!--.*?-->", "", nama or "", flags=re.S)
    n = plain(re.split(r"<br\s*/?>", nama)[0])
    return n


KM_RE = re.compile(r"km\s*([0-9]{1,4})\s*[+,.]\s*([0-9]{1,3})", re.I)


def parse_rows(text):
    """The article's station tables as one ordered list of rows: {"kind": "sta", ...},
    {"kind": "head", "level", "text"}, {"kind": "start"} (a new table)."""
    rows = []
    for m in re.finditer(r"^(={2,6})\s*(.*?)\s*\1\s*$|\{\{\s*[Dd]aftar[Ss]tasiun(-start|-end|-lintas)?"
                         r"\s*[|}]", text, re.M):
        if m.group(1):
            rows.append({"kind": "head", "level": len(m.group(1)), "text": plain(m.group(2)),
                         "pos": m.start()})
            continue
        kind = m.group(3)
        if kind == "-start":
            rows.append({"kind": "start", "pos": m.start()})
            continue
        if kind in ("-end", "-lintas"):
            if kind == "-lintas":
                inner, _e = templates(text, m.start())
                p = split_params(inner)
                rows.append({"kind": "lintas", "segmen": plain(p.get("segmen", "")),
                             "ditutup": plain(p.get("ditutup", "")), "pos": m.start()})
            continue
        inner, _e = templates(text, m.start())
        p = split_params(inner)
        nama = p.get("nama", "")
        letak = p.get("letak", "")
        kms = []
        for part in re.split(r"<br\s*/?>", letak):
            for a, b in KM_RE.findall(part):
                kms.append(int(a) + int(b) / 1000)
        status = plain(p.get("status", "").split("<!--")[0]).lstrip("* ").strip().lower()
        rows.append({"kind": "sta", "name": station_name(nama), "title": station_title(nama),
                     "code": plain(p.get("singkatan", "")).upper().replace(" ", ""),
                     "status": status, "open": status in OPEN, "kms": kms,
                     "kelas": plain(p.get("kelas", "")), "pos": m.start()})
    return rows


# ================================================================ chains and sections

def load_articles():
    p = RAW / "idwiki_articles.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: python id_register.py --fetch")
    d = json.loads(p.read_text("utf-8"))
    st = json.loads((RAW / "idwiki_stations.json").read_text("utf-8"))["stations"]
    return d["articles"], st, d["fetched"]


def line_name(title):
    t = title[len(PREFIX):] if title.startswith(PREFIX) else title
    if not t:
        return t
    return t[0].upper() + t[1:]


def km_pair(ka, kb, crow_km):
    """(km, post a, post b) between two rows from their km posts: the post pair whose
    difference best fits the crow-fly distance, if one fits KM_FIT; else None."""
    best = None
    for x in ka:
        for y in kb:
            d = abs(x - y)
            if crow_km is None:
                # no points to judge by: the shortest difference that is a distance at all (two
                # posts at 0+000 are two chainages starting, not one place)
                if 0.2 <= d < 40 and (best is None or d < best[0]):
                    best = (d, x, y)
                continue
            ok = KM_FIT[0] * crow_km - KM_FIT[1] <= d <= KM_FIT[2] * crow_km + KM_FIT[3]
            if ok and (best is None or abs(d - crow_km * 1.1) < abs(best[0] - crow_km * 1.1)):
                best = (d, x, y)
    return best


def km_fit(ka, kb, crow_km):
    got = km_pair(ka, kb, crow_km)
    return got[0] if got else None


def chains_of(title, text, pt_of, log=None):
    """One article's rows as runs of consecutive active rows: [[row, ...], ...], each row
    carrying "pt" (lon, lat) or None. Breaks at an inactive heading, and at a heading or new
    table whose first row does not fit the row before by km posts."""
    rows = parse_rows(text)
    heads = []                                   # [(level, inactive)]
    runs, cur, boundary = [], [], False
    # nothing before the station-list heading counts (infobox tables of trains use no rows)
    for r in rows:
        if r["kind"] == "head":
            while heads and heads[-1][0] >= r["level"]:
                heads.pop()
            heads.append((r["level"], bool(INACTIVE_HEAD.search(r["text"]))
                          and not re.search(r"\baktif\b", r["text"], re.I)))
            boundary = True
            continue
        if r["kind"] == "start":
            boundary = True
            continue
        if r["kind"] == "lintas":
            # its `ditutup` (closed) date is history: Kandangan - Gresik "ditutup 1983" runs
            # again to Indro; which rows run is told by their stations' status
            continue
        inactive = any(f for _l, f in heads)
        if inactive:
            if cur:
                runs.append(cur)
            cur, boundary = [], False
            continue
        r = dict(r, pt=pt_of(r))
        if not r["open"] and not r["kms"] and not r["pt"]:
            continue                             # a closed halt with no post and no point
        if cur and boundary and not cur[-1]["kms"]:
            # a table ends on a row with no post (Jakarta - Bogor's "Meester Cornelis NIS", a
            # closed halt after Manggarai): the join is judged from the last row with one, and
            # closed rows after it go
            while len(cur) > 1 and not cur[-1]["kms"] and not cur[-1]["open"]:
                cur.pop()
        if cur:
            prev = cur[-1]
            crow = (dist_m(*prev["pt"], *r["pt"]) / 1000 if prev["pt"] and r["pt"] else None)
            got = km_pair(prev["kms"], r["kms"], crow)
            if boundary:
                # across a heading or a new table: neighbours only if the posts fit the points
                # AND carry on the chainage the run came in on, the same way. Jakarta's tables
                # end and start at stations that fit by chance: Kemayoran (km 4.7, coming up
                # from Ancol) and Tanah Abang (km 0.0) are 4.7 km apart, and not neighbours.
                inc = prev.get("_in")
                ok = got is not None and (crow is not None or inc is not None)
                if ok and inc and (got[1] != inc[0] or (got[2] - got[1]) * inc[1] < 0):
                    ok = False
                if not ok:
                    runs.append(cur)
                    cur = []
                    got = None
            if got:
                r["_in"] = (got[2], 1 if got[2] >= got[1] else -1)
        cur.append(r)
        boundary = False
    if cur:
        runs.append(cur)
    # wrong points out, unmeasurable closed stations merged away, then split where a section
    # cannot be measured or is too long to be one, and trimmed to the first and last open station
    out = []
    for run in runs:
        run = tidy_run(run)
        piece = [run[0]] if run else []
        pieces = []
        for x, y in zip(run[:-1], run[1:]):
            km, _how = pair_km(x, y)
            if km is None or km > MAX_HOP_KM:
                pieces.append(piece)
                piece = []
            piece.append(y)
        pieces.append(piece)
        for p in pieces:
            idx = [i for i, r in enumerate(p) if r["open"]]
            if len(idx) >= 2:
                out.append(p[idx[0]:idx[-1] + 1])
    return out


def pair_km(x, y):
    """(km, how) for two neighbouring rows: km posts that fit the crow-fly distance; posts
    alone where a point is missing; the crow-fly x CROW_FACTOR where the posts say nothing."""
    if x.get("trust") and y.get("trust") and x["kms"] and y["kms"]:
        return abs(x["kms"][0] - y["kms"][0]), "posts"       # a HAND line's published chainage
    crow = dist_m(*x["pt"], *y["pt"]) / 1000 if x["pt"] and y["pt"] else None
    km = km_fit(x["kms"], y["kms"], crow)
    if km is not None:
        return km, "posts"
    if crow is None:
        km = km_fit(x["kms"], y["kms"], None)
        return (km, "posts") if km is not None else (None, "none")
    return crow * CROW_FACTOR, "crow"


def tidy_run(run):
    """A point whose km posts fit neither neighbour while theirs fit each other has a wrong
    coordinate (id.wikipedia's Gedangan (Grobogan) is 10 km off its line): it is dropped, so
    rinf.py places it by name or traces past it, and its sections take the posts' km. Then a
    closed station with no km post next to a section the posts cannot measure is merged away
    (Mayang, between Pajang and Gawok, has neither a post nor a point)."""
    run = [dict(r) for r in run]

    def bad(x, y):
        """True where both have a point and km posts, and the posts put them closer than the
        crow flies between the points (no post pair fits, the nearest is shorter than the
        crow-fly), or far closer than points over 25 km apart: track cannot be shorter than
        the crow-fly, so a point is wrong. Not "longer than the crow-fly": track winds."""
        if not (x["pt"] and y["pt"]):
            return None
        crow = dist_m(*x["pt"], *y["pt"]) / 1000
        d0 = km_fit(x["kms"], y["kms"], None)
        if d0 is None:
            return None
        if km_fit(x["kms"], y["kms"], crow) is None and d0 < KM_FIT[0] * crow - KM_FIT[1]:
            return True
        if crow < 0.2 and d0 > 1.0:
            return True                  # two stations km apart at one point: one is a copy
        return crow > 25 and d0 < 0.5 * crow
    # Each point is judged against the nearest row on each side that has one. Worst first: the
    # point with the most bad pairs less good ones goes (or takes OSM's station of exactly its
    # name, if that fits), and the pairs are judged again, until no pair is bad. So of two
    # neighbours sharing one wrong point (Larangan and Suradadi on Cirebon - Semarang) both
    # go, and a wrong point next to a line's first station (Pagongan after Tegal) takes the
    # blame, not the station.
    for _round in range(len(run)):
        with_pt = [i for i, r in enumerate(run) if r["pt"]]
        score = {}
        for a, b in zip(with_pt[:-1], with_pt[1:]):
            v = bad(run[a], run[b])
            if v is None:
                continue
            for i in (a, b):
                score[i] = score.get(i, 0) + (1 if v else -1)
        worst = [i for i in with_pt if score.get(i, 0) > 0]
        if not worst:
            break
        i = max(worst, key=lambda i: (score[i], -i))
        r = run[i]
        k = with_pt.index(i)
        lj = run[with_pt[k - 1]] if k > 0 else None
        rk = run[with_pt[k + 1]] if k + 1 < len(with_pt) else None
        alt = osm_point_by_name(r["name"])
        r["pt"] = None
        if alt and alt != r.get("pt_was"):
            trial = dict(r, pt=alt)
            if not (lj and bad(lj, trial)) and not (rk and bad(trial, rk)):
                r["pt"] = alt
        r["pt_was"] = alt
        r["pt_dropped"] = True
    # a row with a point but no km post (Muara Lawai, between Banjarsari and Muara Enim) takes a
    # post interpolated by crow-fly distance between the nearest rows with both on either side
    for i, r in enumerate(run):
        if r["kms"] or not r["pt"]:
            continue
        lj = next((run[j] for j in range(i - 1, -1, -1) if run[j]["pt"] and run[j]["kms"]
                   and not run[j].get("kms_interp")), None)
        rk = next((run[k] for k in range(i + 1, len(run)) if run[k]["pt"] and run[k]["kms"]),
                  None)
        if not lj or not rk:
            continue
        crow_lr = dist_m(*lj["pt"], *rk["pt"]) / 1000
        span = km_fit(lj["kms"], rk["kms"], crow_lr)
        if span is None:
            continue
        a = dist_m(*lj["pt"], *r["pt"])
        b = dist_m(*r["pt"], *rk["pt"])
        if a + b <= 0:
            continue
        # the left row's post on the shared chainage
        kl, kr = min(((x, y) for x in lj["kms"] for y in rk["kms"]),
                     key=lambda p: abs(abs(p[0] - p[1]) - span))
        r["kms"] = [kl + (kr - kl) * a / (a + b)]
        r["kms_interp"] = True
    changed = True
    while changed:
        changed = False
        for i in range(1, len(run) - 1):
            r = run[i]
            if r["open"]:
                continue
            # a closed halt with no point is no stop and cannot be placed: merged away where
            # its neighbours measure across it
            if (pair_km(run[i - 1], r)[0] is None or pair_km(r, run[i + 1])[0] is None
                    or (not r["pt"] and pair_km(run[i - 1], run[i + 1])[0] is not None)):
                del run[i]
                changed = True
                break
    return run


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def make_pt_of(wst, log):
    """A row's point: the OSM station or stop carrying its KAI code (near id.wikipedia's point
    if there is one), else id.wikipedia's point, else the OSM station of exactly its name."""
    osm_by_code = load_osm_codes(log)

    def pt_of(r):
        e = wst.get(r["title"]) or {}
        wp = (e["lon"], e["lat"]) if "lon" in e else None
        for rec in osm_by_code.get(r["code"], ()):
            if wp is None or dist_m(rec[3], rec[4], *wp) <= OSM_REF_M:
                return (rec[3], rec[4])
        # OSM's station of exactly that name, if within OSM_NAME_KM of the wiki's point:
        # id.wikipedia's are off by kilometres in places (Kedungbanteng 5.6 km, Sragi 1.9)
        osm = osm_point_by_name(r["name"])
        if osm and (wp is None or dist_m(*osm, *wp) <= OSM_NAME_KM * 1000):
            return osm
        return wp
    return pt_of


def build_register(log):
    arts, wst, fetched = load_articles()
    pt_of = make_pt_of(wst, log)

    reg, points = {}, {}
    stats = Counter()

    by_code = defaultdict(list)                  # code -> [(op, name key, point)]

    def op_of(r):
        """One point per station. By KAI code, but a code is not unique across islands and
        old lists (Bantarkadu on the Bogor line and Barru on Makassar - Parepare are both
        BAR; Tigaraksa and Tegalluar): a second station under a code it shares, by another
        name and over SAME_CODE_KM away, gets a point of its own."""
        if "op" in r:
            return r["op"]
        code, key = r["code"] or "", norm_name(r["name"])
        if not re.search(r"[A-Z]", code):
            r["op"] = f"idn:{key}"
            return r["op"]
        for op, k, pt in by_code[code]:
            near = pt and r["pt"] and dist_m(*pt, *r["pt"]) <= SAME_CODE_KM * 1000
            if k == key or near or (not pt and not r["pt"]) or \
                    (k.startswith(key) or key.startswith(k)) and not (pt and r["pt"]):
                r["op"] = op
                return op
        op = f"id:{code}" if not by_code[code] else f"id:{code}-{key}"
        by_code[code].append((op, key, r["pt"]))
        r["op"] = op
        return op

    hand = defaultdict(list)
    for name, qid, op, rows in HAND:
        run = [{"name": n, "title": f"Stasiun {n}", "code": c, "kms": list(k), "open": True,
                "status": "beroperasi", "trust": True} for n, c, k, _p in rows]
        for r, (_n, _c, _k, p) in zip(run, rows):
            r["pt"] = p or pt_of(r)
        hand[name].append((qid, op, run))
    todo = [(a["title"], a.get("qid"), a["text"]) for _k, a in sorted(arts.items())]
    names_art = {line_name(t) for t, _q, _x in todo}
    todo += [(name, v[0][0], None) for name, v in hand.items() if name not in names_art]
    for title, qid, text in todo:
        name = line_name(title)
        runs = chains_of(title, text, pt_of, log) if text else []
        runs += [run for _q, _op, run in hand.get(name, ())]
        operator = next((op for _q, op, _r in hand.get(name, ())), "KAI") if not text else "KAI"
        pairs = []
        for run in runs:
            for r in run:
                op = op_of(r)
                p = points.setdefault(op, {"op": op, "name": r["name"], "code": r["code"],
                                           "stop": False, "pt": r["pt"], "lines": set()})
                if r["open"] and r["code"] not in FREIGHT_STATIONS:
                    p["stop"] = True
                if not p["pt"] and r["pt"]:
                    p["pt"] = r["pt"]
                p["lines"].add(name)
            for x, y in zip(run[:-1], run[1:]):
                km, how = pair_km(x, y)
                stats[how] += 1
                pairs.append((op_of(x), op_of(y), km, how, x["name"], y["name"]))
        # hand lists: closed stretches and freight-only lines
        cut = INACTIVE.get(name, []) + FREIGHT.get(name, [])
        if cut:
            pairs = drop_between(pairs, cut, log, name)
        if pairs:
            reg[title] = {"name": name, "qid": qid, "pairs": pairs, "operator": operator,
                          "wp_km": infobox_km(text) if text else None}
    log(f"ID: {len(reg)} lines from {len(arts)} articles; section km from km posts "
        f"{stats['posts']}, crow-fly {stats['crow']}")
    return reg, points, fetched


def drop_between(pairs, cut, log, name):
    """Leave out the sections between two named stations of a line (along the line's own
    sections), or the whole line for ("*", "*")."""
    if ("*", "*") in cut:
        log(f"  ID: {name}: left out whole (FREIGHT / INACTIVE)")
        return []
    adj = defaultdict(list)
    for i, (a, b, _k, _h, na, nb) in enumerate(pairs):
        adj[a].append((b, i))
        adj[b].append((a, i))
    names = {}
    for a, b, _k, _h, na, nb in pairs:
        names.setdefault(norm_name(na), a)
        names.setdefault(norm_name(nb), b)
    gone = set()
    for x, y in cut:
        s, t = names.get(norm_name(x)), names.get(norm_name(y))
        if s is None or t is None:
            log(f"  ID: {name}: INACTIVE/FREIGHT {x} - {y} not found on the line")
            continue
        prev, seen, stack = {}, {s}, [s]
        while stack:                              # breadth first: the fewest sections
            u = stack.pop(0)
            for v, i in adj[u]:
                if v not in seen:
                    seen.add(v)
                    prev[v] = (u, i)
                    stack.append(v)
        if t not in prev:
            continue
        u = t
        while u != s:
            u, i = prev[u]
            gone.add(i)
    kept = [p for i, p in enumerate(pairs) if i not in gone]
    log(f"  ID: {name}: {len(gone)} sections left out between {cut}")
    return kept


def infobox_km(text):
    m = re.search(r"\|\s*(?:linelength|tracklength|panjang|length)\s*=\s*([^\n|]*\|?[^\n]*)", text)
    if not m:
        return None
    v = m.group(1)
    n = re.search(r"([0-9]+(?:[.,][0-9]+)?)", v.replace("{{convert|", "").replace("{{km to mi|", ""))
    if not n:
        return None
    return float(n.group(1).replace(",", "."))


def load_osm_codes(log):
    """OSM station nodes by the KAI code in `ref` (or `railway:ref`), after the extract."""
    if not (PROC / "stops.pkl").exists():
        log("ID: no data/proc/id yet (no extract): station points from id.wikipedia only")
        return {}
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    by = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") not in ("station", "halt") and tags.get("train") != "yes":
            continue                      # stations, and train stop positions (most carry it)
        for k in ("ref", "railway:ref", "ref:kai"):
            for r in re.split(r"[;,]", tags.get(k) or ""):
                if r.strip():
                    by[r.strip().upper()].append((nid, tags.get("name") or "", "", lon, lat))
        if tags.get("railway") in ("station", "halt") and tags.get("station") not in (
                "subway", "light_rail", "monorail") and tags.get("name"):
            NAME_PTS[norm_name(tags["name"])].append((lon, lat))
    log(f"ID: OSM {len(by)} station codes in `ref`")
    return by


NAME_PTS = defaultdict(list)          # OSM rail station points by normalised name


def osm_point_by_name(name):
    """The OSM station of exactly this name, if all of that name lie within 2 km."""
    pts = NAME_PTS.get(norm_name(name)) or NAME_PTS.get(norm_name(re.sub(r"\s*\(.*?\)", "", name)))
    if pts and all(dist_m(*p, *pts[0]) <= 2000 for p in pts):
        return pts[0]
    return None


# ================================================================ rinf.py's input

def convert(log=log_print, write=True):
    reg, points, fetched = build_register(log)
    owner = {}
    order = sorted(reg, key=lambda t: (-len(reg[t]["pairs"]), t))
    for t in order:
        for a, b, *_ in reg[t]["pairs"]:
            owner.setdefault(frozenset((a, b)), t)
    rows, names, used, n_dup, n_none = [], {}, set(), 0, 0
    for t in order:
        L = reg[t]
        lid = L["qid"] or t
        k = 0
        for a, b, km, how, na, nb in L["pairs"]:
            if a == b:
                continue
            if owner[frozenset((a, b))] != t:
                n_dup += 1
                continue
            if km is None:
                n_none += 1
                continue
            k += 1
            rows.append({"sol": f"{lid}:{k}", "line": lid, "a": a, "b": b, "len": f"{km:.3f}",
                         "im": L.get("operator", "KAI"), "label": f"{na} - {nb}", "how": how})
            used |= {a, b}
        if k:
            names[lid] = {"name": L["name"], "name_en": "", "article": t,
                          "km": round(sum(p[2] or 0 for p in L["pairs"]), 1),
                          "wp_km": L.get("wp_km"),
                          "crow_sections": sum(1 for p in L["pairs"] if p[3] == "crow")}
    pts_out = []
    for op in sorted(used):
        p = points[op]
        r = {"op": op, "uopid": op.split(":", 1)[1].upper(),
             "name": NAME_ALIAS.get(p["name"], p["name"]),
             "type": "10" if p["stop"] else "80"}
        if p.get("pt"):
            r["lon"], r["lat"] = p["pt"]
        pts_out.append(r)
    log(f"ID: {len(names)} lines, {sum(n['km'] for n in names.values()):,.0f} km; "
        f"{len(rows)} section rows ({sum(1 for r in rows if r['how'] == 'crow')} crow-fly), "
        f"{n_dup} pairs left to the line listing them first, {n_none} with no km at all; "
        f"{len(pts_out)} points, {sum(1 for r in pts_out if r['type'] == '10')} stops, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced")
    if write:
        stamp = {"endpoint": "id.wikipedia line articles (id_register.py)", "fetched": fetched}
        for fn, obj in (("sections.json", {**stamp, "rows": rows}),
                        ("points.json", {**stamp, "rows": pts_out}),
                        ("names.json", names)):
            tmp = RAW / (fn + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False), "utf-8")
            os.replace(tmp, RAW / fn)
        log(f"ID: wrote sections.json, points.json, names.json to {RAW}")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(log)
    import rinf
    return rinf.build(path, log)


def report(log=log_print):
    rows, pts, names = convert(log, write=False)
    print(f"\n{'line':42} {'km':>7} {'infobox':>8} {'ratio':>6} crow")
    for lid, n in sorted(names.items(), key=lambda kv: kv[1]["name"]):
        wp = n.get("wp_km")
        print(f"{n['name'][:42]:42} {n['km']:7.1f} {wp or 0:8.1f} "
              f"{(n['km'] / wp) if wp else 0:6.2f} {n['crow_sections']}")
    print(f"all {sum(n['km'] for n in names.values()):,.0f} km")


def show(title_part):
    arts, wst, _f = load_articles()
    pt_of = make_pt_of(wst, lambda m: None)
    for t, a in arts.items():
        if title_part.casefold() not in t.casefold():
            continue
        print(f"== {t}")
        for run in chains_of(t, a["text"], pt_of):
            print("  run:")
            for x, y in zip(run[:-1], run[1:]):
                crow = dist_m(*x["pt"], *y["pt"]) / 1000 if x["pt"] and y["pt"] else None
                km, how = pair_km(x, y)
                print(f"    {x['name'][:22]:22} {x['code']:5} {'o' if x['open'] else 'x'} -> "
                      f"{y['name'][:22]:22} {y['code']:5} {'o' if y['open'] else 'x'} km "
                      f"{km if km is None else round(km, 2)} {how} crow "
                      f"{crow if crow is None else round(crow, 2)}"
                      f"{' (point dropped)' if y.get('pt_dropped') else ''}")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", metavar="TITLE")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    if args.convert:
        convert()
    if args.report:
        report()
    if args.show:
        show(args.show)
