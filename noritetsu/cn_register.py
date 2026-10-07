"""Lines, stations and sections for mainland China (and Macau), with OSM's named track as the
register. Run commands, sources and what is still off: cn_sources.md.

    python cn_register.py --fetch        # Wikidata's lines, stations, adjacency -> data/raw/cn
    python extract.py --region cn --pbf data/raw/china-YYMMDD.osm.pbf
    python cn_register.py --clip         # after every extract: out Hong Kong and abroad
    python build_model.py --region cn --register cn_register:data/raw/cn
    python cn_register.py --dry          # build() alone, result in data/proc/cn/dry.pkl
                                         # (CN_ONLY=京沪线,京广线 limits it and logs anchors)

THE REGISTER UNIT is the national line as China Railway names it, which is what OSM China
names the track: 京沪线 and 京沪高铁 (the high-speed line) are two lines, 京广线 and 京广高速线,
沪昆线 and 沪昆高速线. 97.9% of main and branch heavy-rail track km carries such a name
(`python probe_kr_ways.py --region cn`), 95% lies in a route=railway relation of the same
name (`python probe_cn_infra.py`), which supplies the national line code (0002 京沪线, 3002
京沪高铁) as `ref` and often name:en. So this is kr_register's recipe: a line is the graph of
the ways carrying its name (a way named "A;B" is on both; an unnamed way in an infrastructure
relation takes the relation's name), plus unnamed yard roads that close a gap in it
(`bridge_ways`: OSM leaves the through tracks of big stations unnamed, and 兰新线 fell apart at
哈密). Metros, trams and light rail are not register lines; they stay OSM lines.

WHICH STATIONS ARE PASSENGER STOPS comes from 12306's ticketing list (station_name.js, every
station a ticket can be bought to, 3,404 of them; 3,350 found among OSM's stations by the Han
part of the name, "乌鲁木齐 ئۈرۈمچى" -> 乌鲁木齐). No open list says which stations are on
which line, so a passenger station is on a line when one of its stop nodes lies on the line's
NAMED track (STOP_ON_M) or its own node is within MATCH_M of it. Named track only: the yard
roads bridge_ways adds reach other lines' stations.

SECTIONS come from kr_register's absorbing search (footprints, neighbours, between), then:
  - a section whose ends the line's other sections also join, via its own stations, within
    BYPASS_RATIO of its length is a bypass and goes (`bypasses`): parallel track pairs that
    pass a station outside its footprint, 京沪线's avoiding line round 济南;
  - separate pieces of one line are joined over any track (`join_pieces`, `Net`) where they
    are close: 京港高速线 through 合肥, the unnamed yards at 株洲, 沈阳, 怀化;
  - a line whose named track stops just short of a passenger station is carried on to it
    (`extend_ends`): 滨洲线 into 哈尔滨, 贵广客专线 to 广州南.
Every one of these is logged by name. FREIGHT: a line is a passenger line when two 12306
stations are placed on it, whatever OSM's railway:traffic_mode says (胶济线 is 65% tagged
freight and has 济南, 淄博, 潍坊 on it); freight-only lines drop out by having no such
stations (大秦线, 朔黄线). Freight-tagged sections that were kept are logged. Two freight
railways that pass that test on stray stations are left out by name (FREIGHT_LINES).

HIGH-SPEED is per section (`highspeed_sections`), from the highspeed=yes ways each lies on, as
in Korea. English names: the infrastructure relation's name:en, else Wikidata's label for the
line (京沪铁路 read as 京沪线); stations from OSM's name:en, else Wikidata's label nearby. The
native name is always the name.

The `path` argument is data/raw/cn; the OSM half is read from data/proc/cn (extract.py).
"""
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from kr_register import INF, dist_m

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "cn"

WIKIDATA = "https://query.wikidata.org/sparql"
USER_AGENT = "noritetsu-rail-map/1.0"
LANGS = '"zh-cn", "zh-hans", "zh", "zh-hant", "en"'

# Every railway line (any subclass of Q728937, which takes in high-speed lines and metro
# lines) with P17 China, with what a register reader wants of it.
Q_LINES = """
SELECT ?x ?cls ?lab ?lang ?osm ?num ?len ?unit ?op ?opened ?closed WHERE {
  ?x wdt:P17 wd:Q148 ; wdt:P31 ?cls . ?cls wdt:P279* wd:Q728937 .
  OPTIONAL { ?x rdfs:label ?lab . BIND(LANG(?lab) AS ?lang) FILTER(?lang IN (%s)) }
  OPTIONAL { ?x wdt:P402 ?osm }
  OPTIONAL { ?x wdt:P1671 ?num }
  OPTIONAL { ?x p:P2043/psv:P2043 ?q . ?q wikibase:quantityAmount ?len ;
                                         wikibase:quantityUnit ?unit }
  OPTIONAL { ?x wdt:P137 ?op }
  OPTIONAL { ?x wdt:P1619 ?opened }
  OPTIONAL { ?x wdt:P3999 ?closed }
}
""" % LANGS

# Station to station adjacency, qualified by the line (P197 + pq:P81).
Q_ADJ = """
SELECT ?s ?a ?line WHERE {
  ?s wdt:P17 wd:Q148 ; p:P197 ?st . ?st ps:P197 ?a ; pq:P81 ?line .
}
"""

# Every station that has an adjacency, or is on a line (P81), with its point and labels.
Q_STATIONS = """
SELECT ?s ?lab ?lang ?coord ?osm WHERE {
  { ?s wdt:P17 wd:Q148 ; wdt:P197 [] . } UNION { ?s wdt:P17 wd:Q148 ; wdt:P81 [] . }
  OPTIONAL { ?s wdt:P625 ?coord }
  OPTIONAL { ?s wdt:P11693 ?osm }
  OPTIONAL { ?s rdfs:label ?lab . BIND(LANG(?lab) AS ?lang) FILTER(?lang IN (%s)) }
}
""" % LANGS

# Station -> line (P81), for stations with no adjacency statements.
Q_ON_LINE = """
SELECT ?s ?line WHERE { ?s wdt:P17 wd:Q148 ; wdt:P81 ?line . }
"""


def clip(log=print):
    """Rewrite data/proc/cn without Hong Kong: Geofabrik's china extract holds Hong Kong and
    Macau, and Hong Kong is its own region (hk). The inverse of hk_register.clip, on the same
    boundary (OSM relation 913110, data/raw/hk/hk_boundary.geojson): a way goes if at least
    half its nodes are inside Hong Kong, a stop if it is inside, a relation if no member is
    left. Macau stays in cn. Run after every extract."""
    import pickle
    import shapely
    from collections import Counter
    from shapely.geometry import shape
    import numpy as np
    d = ROOT / "data" / "proc" / "cn"
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(d / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    hk = shape(json.loads((ROOT / "data" / "raw" / "hk" / "hk_boundary.geojson")
                          .read_text(encoding="utf-8")))
    shapely.prepare(hk)
    x, y = cx / 1e7, cy / 1e7
    # Geofabrik's cut runs some way past the border: Vietnam's Hanoi-Lao Cai and Dong Dang
    # lines, North Korea, the Russian Far East, Laos, Mongolia and Kazakhstan all have track
    # in it. OSM's own boundary of China (relation 270056, data/raw/cn/cn_boundary.geojson from
    # polygons.openstreetmap.fr) takes them out; it holds Hong Kong and Macau, not Taiwan.
    cn_shape = shape(json.loads((RAW / "cn_boundary.geojson").read_text(encoding="utf-8")))
    shapely.prepare(cn_shape)
    out = shapely.contains_xy(hk, x, y) | ~shapely.contains_xy(cn_shape, x, y)
    inside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in inside for n in ns) >= len(ns):
            cut[tags.get("name") or "(unnamed)"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    stop_ids = np.fromiter(stops.keys(), dtype=np.int64)
    pos = np.searchsorted(cid, stop_ids)
    np.clip(pos, 0, cid.size - 1, out=pos)
    abroad = set(stop_ids[(cid[pos] == stop_ids) & out[pos]].tolist())
    keep_s = {k: v for k, v in stops.items()
              if k not in abroad and not shapely.contains_xy(hk, v[1], v[2])}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    # An infrastructure relation goes only when every one of its track ways was cut: many
    # have no way in ways.pkl at all (abandoned or planned lines, super-relations).
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1])[:30]:
        log(f"  cut {v:4d}  {name}")
    log(f"CN clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


def traffic_pass(pbf, log=print):
    """data/proc/cn/traffic_mode.pkl: {way id: railway:traffic_mode} for rail ways, read
    straight from the .pbf, until extract.py keeps the tag itself (asked for 2026-09-30)."""
    import pickle
    os.environ.setdefault("OSMIUM_POOL_THREADS", "4")
    import osmium
    out = {}
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if w.tags.get("railway") and w.tags.get("railway:traffic_mode"):
            out[w.id] = w.tags.get("railway:traffic_mode")
    with open(ROOT / "data" / "proc" / "cn" / "traffic_mode.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    log(f"traffic_mode: {len(out)} ways")


def sparql(query, tries=4):
    body = urllib.parse.urlencode({"query": query}).encode()
    for k in range(tries):
        req = urllib.request.Request(WIKIDATA, data=body, headers={
            "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return [{v: b[v]["value"] for v in b} for b in d["results"]["bindings"]]
        except Exception as e:                                  # noqa: BLE001
            print(f"  SPARQL attempt {k + 1} failed: {e}", flush=True)
            if k == tries - 1:
                raise
            time.sleep(15 * (k + 1))


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in (("wd_lines", Q_LINES), ("wd_adjacency", Q_ADJ),
                    ("wd_stations", Q_STATIONS), ("wd_on_line", Q_ON_LINE)):
        t = time.time()
        rows = sparql(q)
        (RAW / f"{name}.json").write_text(json.dumps(
            {"endpoint": WIKIDATA, "fetched": date.today().isoformat(), "rows": rows},
            ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(rows)} rows in {time.time() - t:.0f} s", flush=True)
        time.sleep(3)


# =========================================================================== the register

# Track that carries no scheduled passengers.
NOT_PASSENGER_USAGE = {"industrial", "military", "test", "freight"}
HEAVY = {"rail": "rail", "narrow_gauge": "narrow_gauge", "preserved": "rail"}
CODE = re.compile(r"^\s*(\d{4})\b")

STOP_TO_STATION_M = 1200   # a stop node belongs to the station of its name this close
DUP_M = 1500               # records of one 12306 name this close are one station
MATCH_M = 300              # a station node this far off a line's track is on that line
STOP_ON_M = 30             # a stop node this close to a line's track anchors its station there
FOOT_M = 150               # a station cuts every track of its line within this of its anchors
FAR_FOOT_M = 600           # the most a station mapped off its track reaches from its node
MERGE_M = 100              # two stations of one line closer than this are one station

HAN = re.compile(r"[㐀-鿿豈-﫿]")
NOT_HAN = re.compile(r"[^㐀-鿿豈-﫿0-9()（）]")


def line_id(name):
    h = hashlib.blake2b(f"cn|{name}".encode("utf-8"), digest_size=5)
    return "c" + h.hexdigest()


def han_part(name):
    """The Chinese part of a bilingual OSM name: "乌鲁木齐 ئۈرۈمچى" -> "乌鲁木齐". The whole name
    if it has no Han in it."""
    n = unicodedata.normalize("NFKC", name or "").strip()
    if not HAN.search(n):
        return n
    return re.sub(r"\(\s*\)", "", NOT_HAN.sub("", n)).strip()


def station_key(name):
    """One spelling of a station name, as 12306 writes it: the Han part, no brackets, no
    trailing 站 or 火车站 ("北京南站" -> "北京南")."""
    s = han_part(name)
    s = re.sub(r"[(（].*?[)）]", "", s)
    for suf in ("火车站", "站"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
            break
    return s


def load_12306(log):
    """China Railway's ticketing station list: {name: telecode}. Every station a passenger can
    buy a ticket to, and nothing else, which is how an OSM station is known to be a stop."""
    txt = (RAW / "12306_station_name.js").read_text("utf-8")
    out = {}
    for rec in txt.split("@")[1:]:
        f = rec.split("|")
        if len(f) > 3 and f[1]:
            out[f[1]] = f[2]
    log(f"CN: 12306 lists {len(out)} passenger stations")
    return out


def load_wikidata(log):
    """English names from Wikidata: stations by Chinese label (with a point, so a name used
    twice resolves by distance), lines by label with 铁路 read as China Railway's 线."""
    tail = lambda u: u.rsplit("/", 1)[-1]
    st = defaultdict(dict)
    try:
        rows = json.loads((RAW / "wd_stations.json").read_text("utf-8"))["rows"]
        lrows = json.loads((RAW / "wd_lines.json").read_text("utf-8"))["rows"]
    except OSError:
        log("CN: no Wikidata files (python cn_register.py --fetch); no English names from it")
        return {}, {}
    for r in rows:
        s = st[tail(r["s"])]
        if r.get("lab"):
            s.setdefault(r["lang"], r["lab"])
        if r.get("coord") and "pt" not in s:
            m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", r["coord"])
            if m:
                s["pt"] = (float(m.group(1)), float(m.group(2)))
    by_key = defaultdict(list)
    for q, s in st.items():
        if s.get("en") and "pt" in s:
            for lang in ("zh-cn", "zh-hans", "zh", "zh-hant"):
                if s.get(lang):
                    by_key[station_key(s[lang])].append((s["pt"], clean_en_station(s["en"])))
                    break
    ln = defaultdict(dict)
    for r in lrows:
        if r.get("lab"):
            ln[tail(r["x"])].setdefault(r["lang"], r["lab"])
    line_en = {}
    for q, labs in ln.items():
        zh = labs.get("zh-cn") or labs.get("zh-hans") or labs.get("zh")
        if zh and labs.get("en"):
            for k in wd_line_keys(zh):
                line_en.setdefault(k, labs["en"])
    log(f"CN: Wikidata English names for {len(by_key)} station names, {len(line_en)} line names")
    return by_key, line_en


def wd_line_keys(zh):
    """The OSM spellings a Wikidata line label can stand for: 京沪铁路 is 京沪线, 京沪高速铁路
    is 京沪高速线 (and OSM's 京沪高铁), 贵广客运专线 is 贵广客专线, 沪宁城际铁路 is 沪宁城际线."""
    out = {zh}
    for a, b in (("高速铁路", "高速线"), ("高速铁路", "高铁"), ("客运专线", "客专线"),
                 ("城际铁路", "城际线"), ("铁路", "线")):
        if zh.endswith(a):
            out.add(zh[: -len(a)] + b)
    return out


def clean_en_station(en):
    """"Beijing South railway station" -> "Beijing South"."""
    return re.sub(r"\s+(railway|train|rail|high-speed railway)\s+station$", "", en,
                  flags=re.I).strip()


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("cn", log)
    import pickle
    d = ROOT / "data" / "proc" / "cn"
    with open(d / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    traffic = {}
    if (d / "traffic_mode.pkl").exists():
        with open(d / "traffic_mode.pkl", "rb") as f:
            traffic = pickle.load(f)
    return ways, rels, stops, infra, traffic, bm.Coords(cid, cx, cy)


def heavy_station(tags):
    """A heavy-rail station record (not a metro or tram station of the same name)."""
    rw = tags.get("railway")
    if rw not in ("station", "halt") and not (
            tags.get("public_transport") == "station" and tags.get("train") == "yes"):
        return False
    if tags.get("station") in ("subway", "light_rail", "monorail", "funicular") \
            and tags.get("train") != "yes":
        return False
    if any(tags.get(m) == "yes" for m in ("subway", "light_rail", "tram", "monorail")) \
            and tags.get("train") != "yes":
        return False
    return True


def build_stations(stops, pax, wd_st, log):
    """OSM's passenger stations: heavy-rail station records whose name is on 12306's list,
    one per station (records of one name within DUP_M merge), and every stop node of that
    name within STOP_TO_STATION_M mapped onto it."""
    paxk = {station_key(n): n for n in pax}
    recs = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if not tags.get("name") or not heavy_station(tags):
            continue
        k = station_key(tags["name"])
        if k in paxk:
            recs[k].append(nid)
    st, node_st = {}, {}
    for k, ids in recs.items():
        ids.sort(key=lambda n: (stops[n][0].get("railway") != "station", n))
        groups = []
        for n in ids:
            lon, lat = stops[n][1], stops[n][2]
            for g in groups:
                if dist_m(lon, lat, stops[g[0]][1], stops[g[0]][2]) <= DUP_M:
                    g.append(n)
                    break
            else:
                groups.append([n])
        for g in groups:
            head = g[0]
            tags, lon, lat = stops[head]
            en = next((stops[n][0].get("name:en") for n in g if stops[n][0].get("name:en")), "")
            if not en:
                cands = [(dist_m(lon, lat, *pt), e) for pt, e in wd_st.get(k, ())]
                cands = [c for c in cands if c[0] <= 3000]
                if cands:
                    en = min(cands)[1]
            st[head] = {"name": paxk[k], "name_en": en, "lon": lon, "lat": lat, "key": k}
            for n in g:
                node_st[n] = head
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[s["key"]].append(nid)
    n_stop = 0
    for nid, (tags, lon, lat) in stops.items():
        if nid in node_st or not tags.get("name"):
            continue
        if not (tags.get("railway") == "stop" or tags.get("public_transport") == "stop_position"):
            continue
        if any(tags.get(m) == "yes" for m in ("subway", "light_rail", "tram", "monorail")) \
                and tags.get("train") != "yes":
            continue
        best, bd = None, STOP_TO_STATION_M
        for c in by_key.get(station_key(tags["name"]), ()):
            d = dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
            if d <= bd:
                best, bd = c, d
        if best is not None:
            node_st[nid] = best
            n_stop += 1
    found = {s["key"] for s in st.values()}
    missing = sorted(set(paxk) - found)
    log(f"CN: {len(st)} OSM passenger stations for {len(found)} of 12306's {len(paxk)} names, "
        f"{n_stop} stop nodes placed on one; {sum(1 for s in st.values() if s['name_en'])} "
        f"with an English name")
    log(f"  CN: 12306 names with no OSM heavy-rail station: {len(missing)}: "
        f"{' '.join(paxk[k] for k in missing)}")
    return st, node_st


def line_names(tags):
    """The line names a way's own name tag gives it ("滨绥线;滨绥宽轨线" is two)."""
    n = unicodedata.normalize("NFKC", tags.get("name") or "").strip()
    return [p.strip() for p in n.split(";") if p.strip() and LINE_NAME.search(p.strip())]


LINE_NAME = re.compile(r"(线|線|铁路|鐵路|高铁)$")


def assign_ways(ways, infra, traffic, log):
    """{line name: [way id]}: by the way's own name, and by the name of every infrastructure
    relation it is a member of (which is how unnamed track and bridges named for themselves
    get their line). Freight-only track (railway:traffic_mode=freight, usage=freight,
    industrial...) is left out."""
    rel_of = defaultdict(set)
    for rid, (tags, members) in infra.items():
        nm = unicodedata.normalize("NFKC", tags.get("name") or "").strip()
        if not nm:
            continue
        for ty, ref, _role in members:
            if ty == "w":
                rel_of[ref].add(nm)
    by_line = defaultdict(set)
    n_freight = n_rel = 0
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in HEAVY:
            continue
        if tags.get("usage") in NOT_PASSENGER_USAGE - {"freight"}:
            continue
        names = line_names(tags)
        for nm in names:
            by_line[nm].add(wid)
        if not names:
            for nm in rel_of.get(wid, ()):
                by_line[nm].add(wid)
                n_rel += 1
    log(f"CN: {len(by_line)} line names on heavy-rail track; {n_rel} unnamed ways placed by "
        f"their infrastructure relation")
    return by_line


BRIDGE_SHARE = 0.5         # an unnamed run joins a line it touches at points this share of
                           # the run's own extent apart: it spans a gap in that line
BRIDGE_MAX_M = 20000       # ... if the run is no bigger than this


def bridge_ways(ways, by_line, coords, log):
    """Unnamed track that closes a gap in a line joins it: through a big station OSM China
    leaves the platform and yard roads unnamed (service=yard, or no tags beyond railway), so
    兰新线's named track stopped either side of 哈密 and the line fell apart there. Taiwan's
    rule (tw_register.assign_ways): a connected run of unnamed ways joins a line when it
    touches that line at two places as far apart as the run is big."""
    touch = defaultdict(set)
    named = set()
    for ln, wids in by_line.items():
        named |= wids
        for w in wids:
            for n in ways[w][1].tolist():
                touch[n].add(ln)
    cands = {}
    for wid, (tags, nodes) in ways.items():
        if wid in named or tags.get("railway") not in HEAVY:
            continue
        if tags.get("usage") in ("industrial", "military", "test"):
            continue
        if line_names(tags):
            continue
        cands[wid] = nodes.tolist()
    parent = {w: w for w in cands}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w
    first = {}
    for w, nodes in cands.items():
        for n in nodes:
            if n in first:
                a, b = find(first[n]), find(w)
                if a != b:
                    parent[a] = b
            else:
                first[n] = w
    runs = defaultdict(list)
    for w in cands:
        runs[find(w)].append(w)
    got = Counter()
    for members in runs.values():
        at = defaultdict(set)
        for w in members:
            for n in cands[w]:
                for ln in touch.get(n, ()):
                    at[ln].add(n)
        if not at:
            continue
        ids = np.array(sorted({n for w in members for n in cands[w]}), dtype=np.int64)
        pos, ok = coords.many(ids)
        pos = pos[ok]
        if pos.size < 2:
            continue
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        extent = dist_m(x.min(), y.min(), x.max(), y.max())
        if extent > BRIDGE_MAX_M:
            continue
        for ln, ns in at.items():
            if len(ns) < 2:
                continue
            ps = [p for p in (coords.get(n) for n in ns) if p]
            span = max((dist_m(*a, *b) for a in ps for b in ps), default=0.0)
            if span >= BRIDGE_SHARE * extent and span > 0:
                by_line[ln].update(members)
                got[ln] += len(members)
    log(f"CN: {sum(got.values())} unnamed ways joined {len(got)} lines by bridging a gap "
        f"({', '.join(f'{k} {v}' for k, v in got.most_common(6))})")


def is_freight(tags, wid, traffic):
    return ((tags.get("railway:traffic_mode") or traffic.get(wid)) == "freight"
            or tags.get("usage") == "freight")


def relation_info(infra):
    """Per line name, what its biggest infrastructure relation of that name says: the national
    line code, English name, Wikidata item, operator."""
    out = {}
    for rid, (tags, members) in infra.items():
        nm = unicodedata.normalize("NFKC", tags.get("name") or "").strip()
        if not nm:
            continue
        n = sum(1 for m in members if m[0] == "w")
        m = CODE.match(tags.get("ref") or "")
        cur = out.get(nm)
        if cur is None or n > cur["n"]:
            out[nm] = {"n": n, "ref": m.group(1) if m else "",
                       "name_en": tags.get("name:en") or "",
                       "operator": tags.get("operator") or "",
                       "operator_en": tags.get("operator:en") or ""}
    return out


FREIGHT_SHARE = 0.5        # a line this much on freight-only track is logged as such

# Freight railways left out by name. Each has two 12306 stations beside its track and so passes
# the passenger test, but builds as one long section between them that no train runs:
# 浩吉线 as 乌审旗-万荣, 438 km, with none of its 51 OSM stations on 12306's list; 广珠线 as
# 江门-江高, 122 km, the Guangzhou-Zhuhai freight railway. Checked 2026-09-30.
FREIGHT_LINES = {"浩吉线", "广珠线"}


def way_km(nodes, coords):
    pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
    pos = pos[ok]
    if pos.size < 2:
        return 0.0
    lon, lat = coords.x[pos] / 1e7, coords.y[pos] / 1e7
    return float(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32,
                          np.diff(lat) * 110.57).sum())


STEP_M = 40                # a vertex at least this often, as kr_register.line_graph
BYPASS_RATIO = 1.2         # a section is a bypass when the line's own stations join its ends
BYPASS_KM = 2.0            # ... within this ratio of its length, plus this


def graph_of(wids, named, ways, coords, traffic):
    """kr_register.line_graph (densified to a vertex every STEP_M), and also which edges are
    on freight-only track and which vertices are on the line's NAMED ways."""
    adj = defaultdict(list)
    xy = {}
    fresh = [0]
    fast, slow, named_v = set(), set(), set()

    def link(a, b, high, fr, nm):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        pts = [a]
        k = int(dist_m(x1, y1, x2, y2) // STEP_M)
        for j in range(1, k + 1):
            fresh[0] -= 1
            t = j / (k + 1)
            xy[fresh[0]] = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            pts.append(fresh[0])
        pts.append(b)
        if nm:
            named_v.update(pts)
        for u, v in zip(pts[:-1], pts[1:]):
            w = dist_m(*xy[u], *xy[v]) / 1000
            adj[u].append((v, w))
            adj[v].append((u, w))
            e = (u, v) if u < v else (v, u)
            if high:
                fast.add(e)
            if fr:
                slow.add(e)

    for wid in wids:
        tags = ways[wid][0]
        nodes = np.asarray(ways[wid][1], dtype=np.int64)
        high = tags.get("highspeed") == "yes"
        fr = is_freight(tags, wid, traffic)
        nm = wid in named
        pos, ok = coords.many(nodes)
        prev = None
        for n, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            if prev is not None and prev != n:
                link(prev, n, high, fr, nm)
            prev = n
    return adj, xy, fast, slow, named_v


GAP_CROW_KM = 80           # two pieces of one line this close (crow-fly, station to station)
GAP_DETOUR = 1.5           # ... are joined over any track at most this times the crow-fly
GAP_EXTRA_KM = 5.0         # ... plus this


class Net:
    """All heavy-rail track in the extract as one graph over OSM nodes, for joining the pieces
    of a line across track OSM names for another line (京港高速线 through 合肥, the shared
    approaches to big hubs) or leaves in a yard nobody named (株洲, 沈阳, 怀化)."""

    def __init__(self, ways, coords, traffic):
        self.adj = defaultdict(list)
        self.xy = {}
        for wid, (tags, nodes) in ways.items():
            if tags.get("railway") not in HEAVY or \
                    tags.get("usage") in ("industrial", "military", "test"):
                continue
            nodes = np.asarray(nodes, dtype=np.int64)
            pos, ok = coords.many(nodes)
            high = tags.get("highspeed") == "yes"
            fr = is_freight(tags, wid, traffic)
            prev = None
            for n, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
                if not good:
                    prev = None
                    continue
                if n not in self.xy:
                    self.xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
                if prev is not None and prev != n:
                    w = dist_m(*self.xy[prev], *self.xy[n]) / 1000
                    self.adj[prev].append((n, w, high, fr))
                    self.adj[n].append((prev, w, high, fr))
                prev = n

    def path(self, a, b, cap):
        """Shortest track from node a to node b, at most cap km: (nodes, km, fast km, freight
        km) or None."""
        import heapq
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                break
            if d > dist.get(u, INF) or d > cap:
                continue
            for v, w, _h, _f in self.adj.get(u, ()):
                nd = d + w
                if nd <= cap and nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = (u, _h, _f, w)
                    heapq.heappush(heap, (nd, v))
        if b not in dist:
            return None
        nodes, fast, fr = [b], 0.0, 0.0
        while nodes[-1] in prev:
            u, h, f, w = prev[nodes[-1]]
            fast += w if h else 0.0
            fr += w if f else 0.0
            nodes.append(u)
        nodes.reverse()
        return nodes, dist[b], fast, fr


def join_pieces(sections, centre, xy, net):
    """Sections that join the separate pieces of one line over the whole network, nearest
    pieces first, while the crow-fly gap is under GAP_CROW_KM and the track no more than
    GAP_DETOUR times it. Returns {(a, b): section} to add."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in sections:
        parent[find(a)] = find(b)
    sts = sorted({s for k in sections for s in k})
    if len({find(s) for s in sts}) < 2:
        return {}
    # a station's node in the network: the real OSM node nearest its centre on this line
    real = np.array([v for v in xy if v > 0], dtype=np.int64)
    rxy = np.array([xy[v] for v in real.tolist()])

    def node_of(s):
        x, y = centre[s]
        d = np.hypot((rxy[:, 0] - x) * math.cos(math.radians(y)), rxy[:, 1] - y)
        return int(real[int(np.argmin(d))])
    cands = []
    for i, a in enumerate(sts):
        for b in sts[i + 1:]:
            if find(a) == find(b):
                continue
            crow = dist_m(*centre[a], *centre[b]) / 1000
            if crow <= GAP_CROW_KM:
                cands.append((crow, a, b))
    cands.sort()
    out = {}
    tries = Counter()
    for crow, a, b in cands:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        key = frozenset((ra, rb))
        if tries[key] >= 3:
            continue
        tries[key] += 1
        got = net.path(node_of(a), node_of(b), GAP_DETOUR * crow + GAP_EXTRA_KM)
        if got is None:
            continue
        nodes, km, fast, fr = got
        parent[find(a)] = find(b)
        out[(a, b)] = {"km": km, "geom": [centre[a]] + [net.xy[n] for n in nodes] + [centre[b]],
                       "fast": fast >= 0.5 * km if km else False, "gap": True}
    return out


END_KM = 8.0              # a passenger station this close to where a line's named track stops
END_HOME_M = 1500         # ... when no station of the line is this close to that end
END_ON_M = 300            # ... and the traced track passes this close to the end
END_LAST_M = 30000        # ... and the line's last station is no further than this from it


def extend_ends(adj, xy, named_v, centre, st, st_tree, st_ids, net, net_tree, net_ids):
    """Sections from a line's last station on to the station its named track stops short of.
    宝成线's name ends at 广汉北 and 青藏线's at 湟源: the last stretch into 成都 and 西宁 is
    named for another line or not at all (Korea's 호남고속선 had the same fault). Returns
    {(line station, new station): section}."""
    if not centre:
        return {}
    # dead ends of the NAMED track: bridged yard roads often carry it on a little way
    ends = [v for v in named_v if v > 0
            and sum(1 for u, _w in adj.get(v, ()) if u in named_v) == 1]
    on = list(centre)
    cxy = np.array([centre[s] for s in on])
    out = {}
    for d in ends:
        x, y = xy[d]
        dd = np.hypot((cxy[:, 0] - x) * math.cos(math.radians(y)) * 111320,
                      (cxy[:, 1] - y) * 110570)
        if dd.min() <= END_HOME_M or dd.min() > END_LAST_M:
            # a named end far from every station of the line is track with no stations yet
            # (沪渝蓉高速线 past 宜昌北), not a last stretch into a hub
            continue
        last = on[int(np.argmin(dd))]
        k = math.cos(math.radians(y))
        cands = st_tree.query_ball_point((x * k, y), END_KM / 111.32)
        best = None
        for j in cands:
            sid = st_ids[j]
            if sid in centre:
                continue
            dist = dist_m(x, y, st[sid]["lon"], st[sid]["lat"])
            if best is None or dist < best[0]:
                best = (dist, sid)
        if best is None:
            continue
        sid = best[1]
        # The tree's x is lon * cos(lat) at each node's own latitude, so each point is queried
        # with its own: the named end's `k` put the nearest "node" 2-24 km off (fixed 2026-10-04).
        ka = math.cos(math.radians(centre[last][1]))
        kb = math.cos(math.radians(st[sid]["lat"]))
        a = net_ids[net_tree.query((centre[last][0] * ka, centre[last][1]))[1]]
        b = net_ids[net_tree.query((st[sid]["lon"] * kb, st[sid]["lat"]))[1]]
        crow = dist_m(*centre[last], st[sid]["lon"], st[sid]["lat"]) / 1000
        got = net.path(a, b, GAP_DETOUR * crow + GAP_EXTRA_KM)
        if got is None:
            continue
        nodes, km, fast, _fr = got
        if min(dist_m(x, y, *net.xy[n]) for n in nodes) > END_ON_M:
            continue
        spot = (st[sid]["lon"], st[sid]["lat"])
        out[(last, sid)] = {"km": km, "geom": [centre[last]] + [net.xy[n] for n in nodes] + [spot],
                            "fast": fast >= 0.5 * km if km else False, "end": True}
    return out


def bypasses(sections):
    """Sections of a line whose two ends its other sections also join, through the line's
    own stations, at no more than BYPASS_RATIO of the length: track named for the line that
    runs past its stations. 京沪线's freight avoiding line round 济南 made a 泰山-晏城 section
    beside 泰山-济南-晏城. Longest first; each removal is final."""
    import heapq
    adj = defaultdict(dict)
    for (a, b), v in sections.items():
        adj[a][b] = min(v["km"], adj[a].get(b, INF))
        adj[b][a] = min(v["km"], adj[b].get(a, INF))
    out = []
    for (a, b), v in sorted(sections.items(), key=lambda kv: -kv[1]["km"]):
        lim = BYPASS_RATIO * v["km"] + BYPASS_KM
        dist, heap, hit = {a: 0.0}, [(0.0, a)], False
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, INF) or d > lim:
                continue
            if u == b:
                hit = True
                break
            for w, k in adj[u].items():
                if u == a and w == b or u == b and w == a:
                    continue
                nd = d + k
                if nd <= lim and nd < dist.get(w, INF):
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w))
        if hit and dist.get(b, INF) > 0.3 * v["km"]:
            out.append(((a, b), dist[b] / v["km"]))
            del adj[a][b]
            del adj[b][a]
    return out


# Crossings with passenger trains whose track this build carries on from the line's last
# station here to the border point (borders.py), so the two countries' pieces meet there:
#   (border point, the line here it leaves from, the neighbour whose line the piece joins)
# With no neighbour, the section is added to the line itself, and the neighbour's piece takes
# this line's id (hk_register's 廣深港高速鐵路). With one, the piece is a line of its own under
# the id of the neighbour's line ending at that point, read from the neighbour's shipped
# build (dist/data/<cc>/lines.json): the app joins lines of one id across countries into one,
# so a ride over the border is one ride on one line and credits both; without that line it
# falls back to the line itself. The sections end at a junction no OSM route runs over, so
# each is listed in `served_sections` (build_model.drop_unridden_sections).
#   xFutian: 福田 - border, the high-speed line to 香港西九龍 (trains every few minutes).
#   xDongDang: 凭祥 - border towards Đồng Đăng, MR1/MR2 Nanning - Gia Lâm daily since
#     2025-05-25 (vn_sources.md); OSM leaves the last 2.6 km to the border unnamed, so the
#     piece runs over any track (Net). It joins Vietnam's Hà Nội - Đồng Đăng line, which those
#     trains run on, rather than 湘桂线: Pingxiang -> Đồng Đăng is then one ride.
CN_BORDERS = [("xFutian", "广深港高速线", None), ("xDongDang", "湘桂线", "vn")]
PIECE_BORDER_M = 60        # the border point is this close to a node of the track
PIECE_STATION_M = 300      # and the line's station this close to a node joined to it


def border_pieces(lines, stations, geoms, net, net_tree, net_ids, log):
    import borders
    pts = {p["id"]: p for p in borders.load(canonical_only=True)}
    by_name = {l["name"]: l for l in lines}
    for pid, name, nb in CN_BORDERS:
        p, line = pts.get(pid), by_name.get(name)
        if p is None or line is None:
            log(f"  CN: border {pid}: " + ("no such border point" if p is None
                                           else f"no line {name}"))
            continue
        on = {s for sec in line["sections"] for s in sec[:2]}
        sid = min(on, key=lambda s: dist_m(stations[s]["lon"], stations[s]["lat"],
                                           p["lon"], p["lat"]))
        # where the line's own sections put the station, on its track
        start = (stations[sid]["lon"], stations[sid]["lat"])
        for key, g in geoms.get(line["id"], {}).items():
            a, b = key.split("|")
            if a == sid:
                start = tuple(g[0])
                break
            if b == sid:
                start = tuple(g[-1])
                break
        k = math.cos(math.radians(p["lat"]))
        nb_node = net_ids[net_tree.query((p["lon"] * k, p["lat"]))[1]]
        off = dist_m(*net.xy[nb_node], p["lon"], p["lat"])
        crow = dist_m(*start, p["lon"], p["lat"]) / 1000
        # The station's end: the node nearest it among those the border's track reaches (the
        # nearest node outright can be on a parallel line's platform, 广深城际's at 福田).
        got = None
        if off <= PIECE_BORDER_M:
            import heapq
            cap = GAP_DETOUR * crow + GAP_EXTRA_KM
            dist, heap = {nb_node: 0.0}, [(0.0, nb_node)]
            while heap:
                dd, u = heapq.heappop(heap)
                if dd > dist.get(u, INF):
                    continue
                for v, w, _h, _f in net.adj.get(u, ()):
                    if dd + w <= cap and dd + w < dist.get(v, INF):
                        dist[v] = dd + w
                        heapq.heappush(heap, (dd + w, v))
            na = min(dist, key=lambda n: dist_m(*net.xy[n], *start))
            if dist_m(*net.xy[na], *start) <= PIECE_STATION_M:
                got = net.path(na, nb_node, cap)
        if got is None:
            log(f"  CN: border {pid}: no track from {stations[sid]['name']} "
                f"(point {off:.0f} m from the nearest node)")
            continue
        nodes, km, fast, _fr = got
        geom = [start] + [net.xy[n] for n in nodes] + [(p["lon"], p["lat"])]
        km += dist_m(*net.xy[nodes[-1]], p["lon"], p["lat"]) / 1000
        key = f"{sid}|{pid}"
        tgt = line
        if nb:
            other = None
            try:
                other = next((l for l in json.loads(
                    (ROOT / "dist" / "data" / nb / "lines.json").read_text("utf-8"))["lines"]
                    if l.get("src", "osm") != "osm"
                    and any(pid in sec[:2] for sec in l["sections"])), None)
            except (OSError, ValueError, KeyError):
                pass
            if other is not None:
                tgt = {"id": other["id"], "src": "cn", "service": False,
                       "name": other["name"], "name_en": other.get("name_en", ""),
                       "ref": other.get("ref", ""), "colour": "",
                       # the track here is China Railway's, whoever runs the trains
                       "operator": line.get("operator", ""),
                       "operator_en": line.get("operator_en", ""), "network": "",
                       "kind": other.get("kind", "rail"), "highspeed_sections": {},
                       "km": 0.0, "variants": 1, "straight_sections": 0,
                       "display": [], "sections": []}
                lines.append(tgt)
            else:
                log(f"  CN: border {pid}: {nb} has no line ending there; the section goes to "
                    f"{name}")
        if pid not in stations:
            stations[pid] = {"id": pid, "name": p["name"], "name_en": "", "lon": p["lon"],
                             "lat": p["lat"], "lines": set(), "junction": True}
        stations[pid]["lines"].add(tgt["id"])
        stations[sid]["lines"].add(tgt["id"])
        tgt["sections"].append([sid, pid, round(km, 3)])
        tgt["highspeed_sections"][key] = fast >= 0.5 * km if km else False
        tgt["served_sections"] = tgt.get("served_sections", []) + [key]
        tgt["km"] = round(sum(s[2] for s in tgt["sections"]), 3)
        d = tgt["display"]
        if not d:
            d.extend([sid, pid])
        elif d[-1] == sid:
            d.append(pid)
        elif d[0] == sid:
            d.insert(0, pid)
        geoms.setdefault(tgt["id"], {})[key] = [[round(x, 5), round(y, 5)] for x, y in geom]
        log(f"  CN: border {pid}: {stations[sid]['name']} - {p['name']} {km:.2f} km "
            f"({fast:.1f} high-speed), on {tgt['name']} ({tgt['id']})")


def build(path, log):
    from kr_register import Near, between, line_graph, neighbours
    from n02 import walk_order
    ways, _rels, stops, infra, traffic, coords = load_osm(log)
    pax = load_12306(log)
    wd_st, wd_line_en = load_wikidata(log)
    st, node_st = build_stations(stops, pax, wd_st, log)
    by_line = assign_ways(ways, infra, traffic, log)
    named_ways = {k: set(v) for k, v in by_line.items()}
    bridge_ways(ways, by_line, coords, log)
    # each way's line for split_pieces' track graph: its own name tag's first line, else the
    # first (by name) of the lines it was given
    wname = {}
    for nm in sorted(by_line):
        for w in by_line[nm]:
            wname.setdefault(w, nm)
    for w in wname:
        own = line_names(ways[w][0])
        if own:
            wname[w] = own[0]
    _S["wname"] = wname
    rinfo = relation_info(infra)

    st_ids = list(st)
    st_lon = np.array([st[i]["lon"] for i in st_ids])
    st_lat = np.array([st[i]["lat"] for i in st_ids])
    stops_of = defaultdict(list)                     # station -> stop node points
    for n, s in node_st.items():
        if n != s and n in stops:
            stops_of[s].append((stops[n][1], stops[n][2]))

    stations, lines, geoms = {}, [], {}
    dropped, freight = [], []
    cut_freight, cut_bypass = [], []
    n_by_stop = n_by_node = 0

    def place(name):
        """The line's track graph and the passenger stations on it. Stations are found against
        the line's NAMED track only: the unnamed yard roads bridge_ways adds reach into other
        lines' stations (徐兰高速线 picked up the conventional 咸阳 through Xi'an's yards)."""
        nonlocal n_by_stop, n_by_node
        adj, xy, fast, slow, named_v = graph_of(sorted(by_line[name]), named_ways[name],
                                                ways, coords, traffic)
        if len(named_v) < 2:
            return None
        near = Near(xy)
        near_named = Near({v: xy[v] for v in named_v})
        lo_x, hi_x = near.lon.min() - 0.01, near.lon.max() + 0.01
        lo_y, hi_y = near.lat.min() - 0.01, near.lat.max() + 0.01
        cand = np.nonzero((st_lon >= lo_x) & (st_lon <= hi_x) &
                          (st_lat >= lo_y) & (st_lat <= hi_y))[0]

        # --- which stations: a passenger station with a stop node on this track, or whose
        # own node is within MATCH_M of it
        anchors = defaultdict(set)
        far = {}
        for j in cand.tolist():
            sid = st_ids[j]
            got = None
            for lon, lat in stops_of.get(sid, ()):
                v, d = near_named.nearest(lon, lat)
                if d <= STOP_ON_M:
                    anchors[sid].add(v)
                    got = "stop"
            if got:
                n_by_stop += 1
                continue
            v, d = near_named.nearest(st[sid]["lon"], st[sid]["lat"])
            if d <= MATCH_M:
                anchors[sid].add(v)
                n_by_node += 1
                if d > FOOT_M:
                    far[sid] = (st[sid]["lon"], st[sid]["lat"], d)
        # one station per place
        pos = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}
        order = sorted(anchors)
        merged = {}
        for i, s in enumerate(order):
            if s in merged:
                continue
            for t in order[i + 1:]:
                if t not in merged and dist_m(*pos[s], *pos[t]) <= MERGE_M:
                    merged[t] = s
                    anchors[s] |= anchors.pop(t)
                    if t in far and s not in far:
                        far[s] = far[t]
        return adj, xy, fast, slow, near, anchors, far, named_v

    # --- pass 1: which stations each line has, and how much of it is freight-only track.
    # A station on no other line's track is the line's OWN; a line that has none only joins
    # other lines' stations, which is what a freight line crossing them looks like.
    t0 = time.time()
    names = sorted(by_line, key=lambda k: -len(by_line[k]))
    only = [n for n in os.environ.get("CN_ONLY", "").split(",") if n]
    if only:                                   # debugging a few lines: CN_ONLY=京沪线,京广线
        names = [n for n in names if n in only]
    on_lines = defaultdict(set)
    fshare, placed = {}, {}
    for li, name in enumerate(names):
        wids = sorted(by_line[name])
        wkm = {w: way_km(ways[w][1], coords) for w in wids}
        tot_km = sum(wkm.values())
        fr_km = sum(k for w, k in wkm.items() if is_freight(ways[w][0], w, traffic))
        fshare[name] = fr_km / tot_km if tot_km else 0.0
        got = place(name)
        if got is None:
            continue
        placed[name] = set(got[5])
        for s in got[5]:
            on_lines[s].add(name)
    log(f"CN: pass 1, stations of {len(placed)} line names placed in {time.time() - t0:.0f} s")
    own = {name: sum(1 for s in ss if len(on_lines[s]) == 1) for name, ss in placed.items()}
    keep_lines = []
    no_own = []
    for name in names:
        if name not in placed:
            continue
        if name in FREIGHT_LINES:
            freight.append(name)
            continue
        if len(placed[name]) < 2:
            dropped.append((name, len(placed[name])))
            continue
        if own[name] == 0:
            no_own.append(name)
        keep_lines.append(name)

    # --- pass 2: sections
    t0 = time.time()
    net = Net(ways, coords, traffic)
    from scipy.spatial import cKDTree
    net_ids = list(net.xy)
    net_tree = cKDTree(np.array([(x * math.cos(math.radians(y)), y)
                                 for x, y in (net.xy[n] for n in net_ids)]))
    st_tree = cKDTree(np.array([(st[s]["lon"] * math.cos(math.radians(st[s]["lat"])),
                                 st[s]["lat"]) for s in st_ids]))
    log(f"CN: network graph of {len(net.xy):,} nodes in {time.time() - t0:.0f} s")
    gap_log, end_log = [], []
    t0 = time.time()
    n_by_stop = n_by_node = 0
    for li, name in enumerate(keep_lines):
        if li and li % 100 == 0:
            log(f"  CN: {li}/{len(keep_lines)} lines, {time.time() - t0:.0f} s")
        wids = sorted(by_line[name])
        adj, xy, fast, slow, near, anchors, far, named_v = place(name)

        # --- footprints, neighbours, sections: kr_register's
        foot, foot_d = {}, {}
        for sid, ans in anchors.items():
            disks = [(*xy[a], FOOT_M) for a in ans]
            if sid in far:
                lon, lat, d = far[sid]
                disks.append((lon, lat, min(d + FOOT_M, FAR_FOOT_M)))
            for x, y, r in disks:
                ids, ds = near.within(x, y, r)
                for v, d in zip(ids, ds):
                    if d < foot_d.get(v, INF):
                        foot[v], foot_d[v] = sid, d
            for a in ans:
                foot[a], foot_d[a] = sid, 0.0
        pairs = neighbours(adj, foot)
        if only:
            log(f"  DEBUG {name}: anchors " + " ".join(
                f"{st[s]['name']}({len(a)}{'F' if s in far else ''})" for s, a in anchors.items()))
            log(f"  DEBUG {name}: pairs " + " ".join(
                f"{st[a]['name']}-{st[b]['name']}" for a, b in sorted(pairs)))
        if not pairs:
            dropped.append((name, len(anchors)))
            continue
        foot_of = defaultdict(set)
        for v, s in foot.items():
            foot_of[s].add(v)
        centre = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}

        def offsets(s):
            c = centre[s]
            return {v: dist_m(*xy[v], *c) / 1000 for v in foot_of[s]}

        sections = {}
        for a, b in sorted(pairs):
            blocked = {v for v, s in foot.items() if s != a and s != b}
            got = between(adj, offsets(a), offsets(b), blocked)
            if got is None:
                continue
            nodes, km = got
            keep = [n for i, n in enumerate(nodes) if n > 0 or i == 0 or i == len(nodes) - 1]
            steps = list(zip(nodes[:-1], nodes[1:]))
            on_fast = sum(dist_m(*xy[u], *xy[v]) for u, v in steps
                          if ((u, v) if u < v else (v, u)) in fast)
            on_slow = sum(dist_m(*xy[u], *xy[v]) for u, v in steps
                          if ((u, v) if u < v else (v, u)) in slow)
            if km and on_slow / 1000 >= FREIGHT_SHARE * km:
                # Between two passenger stations over track OSM tags freight-only. Kept: two
                # stations 12306 sells tickets to outweigh the tag (胶济线 is 65% freight-tagged
                # and has 济南, 淄博, 潍坊, 高密 on it). A freight avoiding line beside the
                # passenger route goes as a bypass instead (below). Logged.
                cut_freight.append((name, st[a]["name"], st[b]["name"], round(km, 1)))
            sections[(f"c{a}", f"c{b}")] = {
                "km": km, "geom": [centre[a]] + [xy[n] for n in keep] + [centre[b]],
                "fast": on_fast / 1000 >= 0.5 * km if km else False}
        for k, ratio in bypasses(sections):
            v = sections.pop(k)
            cut_bypass.append((name, st[int(k[0][1:])]["name"], st[int(k[1][1:])]["name"],
                               round(v["km"], 1), round(ratio, 2)))
        gaps = join_pieces({(int(a[1:]), int(b[1:])): v for (a, b), v in sections.items()},
                           centre, xy, net)
        for (a, b), v in gaps.items():
            sections[(f"c{a}", f"c{b}")] = v
            gap_log.append((name, st[a]["name"], st[b]["name"], round(v["km"], 1)))
        on_line = {int(s[1:]) for k in sections for s in k}
        ext = extend_ends(adj, xy, named_v, {s: centre[s] for s in on_line if s in centre},
                          st, st_tree, st_ids, net, net_tree, net_ids)
        for (a, b), v in ext.items():
            if (f"c{b}", f"c{a}") not in sections:
                sections[(f"c{a}", f"c{b}")] = v
                end_log.append((name, st[a]["name"], st[b]["name"], round(v["km"], 1)))
        if not sections:
            dropped.append((name, len(anchors)))
            continue

        lid = line_id(name)
        kinds, ops = Counter(), Counter()
        for wid in wids:
            t = ways[wid][0]
            kinds[HEAVY[t["railway"]]] += 1
            if t.get("operator"):
                ops[t["operator"]] += 1
        for sid in {s for k in sections for s in k}:
            if sid not in stations:
                s = st[int(sid[1:])]
                stations[sid] = {"id": sid, "name": s["name"], "name_en": s["name_en"],
                                 "lon": s["lon"], "lat": s["lat"], "lines": set()}
            stations[sid]["lines"].add(lid)
        ri = rinfo.get(name, {})
        en = ri.get("name_en") or wd_line_en.get(name, "")
        lines.append({
            "id": lid, "src": "cn", "service": False,
            "name": name, "name_en": en if en != name else "", "ref": ri.get("ref", ""),
            "colour": "",
            "operator": ops.most_common(1)[0][0] if ops else ri.get("operator", ""),
            "operator_en": ri.get("operator_en", ""), "network": "",
            "kind": kinds.most_common(1)[0][0],
            "highspeed_sections": {f"{a}|{b}": v["fast"] for (a, b), v in sections.items()},
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "variants": 1, "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        })
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}

    if not only:
        border_pieces(lines, stations, geoms, net, net_tree, net_ids, log)

    total = sum(l["km"] for l in lines)
    hs = sum(1 for l in lines if sum(l["highspeed_sections"].values()) * 2
             >= len(l["highspeed_sections"]))
    log(f"CN: {len(lines)} register lines ({hs} mostly high-speed), {total:,.0f} km, "
        f"{len(stations)} stations, in {time.time() - t0:.0f} s; stations placed by a stop node "
        f"on the track {n_by_stop}, by their own node within {MATCH_M} m {n_by_node}")
    # Freight-only track is judged per section, never per line: 胶济线 is 65% freight-tagged
    # in OSM yet its 济南-青岛 passenger sections are not.
    log(f"CN: {len(freight)} freight railways left out by name (FREIGHT_LINES): "
        + " ".join(freight))
    built_km = {l["name"]: l["km"] for l in lines}
    fr_lines = [n for n in keep_lines if fshare[n] >= FREIGHT_SHARE]
    log(f"CN: {len(fr_lines)} line names are half or more freight-only track "
        f"(railway:traffic_mode=freight or usage=freight) but have 12306 passenger stations on "
        f"them, so are built; as name (freight share, passenger stations, km built): "
        + " ".join(f"{n} ({fshare[n]:.2f},{len(placed[n])},{built_km.get(n, 0):.0f})"
                   for n in fr_lines))
    log(f"CN: {len(no_own)} lines built that have no passenger station of their own (every "
        f"one is also on another line's track): " + " ".join(
            f"{l['name']} ({l['km']:.0f} km)" for l in lines if l["name"] in set(no_own)))
    log(f"CN: {len(cut_freight)} sections kept between passenger stations over track OSM tags "
        f"freight-only "
        f"({sum(c[3] for c in cut_freight):,.0f} km): " + "; ".join(
            f"{l} {a}-{b} {k}" for l, a, b, k in cut_freight))
    log(f"CN: {len(cut_bypass)} sections left out as a bypass of their own line's stations "
        f"({sum(c[3] for c in cut_bypass):,.0f} km): " + "; ".join(
            f"{l} {a}-{b} {k} (x{r})" for l, a, b, k, r in cut_bypass))
    log(f"CN: {len(gap_log)} sections join separate pieces of a line over other track "
        f"({sum(g[3] for g in gap_log):,.0f} km): " + "; ".join(
            f"{l} {a}-{b} {k}" for l, a, b, k in gap_log))
    log(f"CN: {len(end_log)} sections carry a line on from where its named track stops to the "
        f"passenger station it stops short of ({sum(g[3] for g in end_log):,.0f} km): "
        + "; ".join(f"{l} {a}-{b} {k}" for l, a, b, k in end_log))
    big = sorted((d for d in dropped if d[1] < 2), key=lambda d: d[0])
    log(f"CN: {len(dropped)} line names built no line (under two passenger stations, or none "
        f"joined): {' '.join(n for n, _k in big[:300])}")
    # where each station is on each line's track, for split_pieces
    _S["ends"] = [(sid, l["name"], *pts[0 if k == 0 else -1])
                  for l in lines for key, pts in geoms[l["id"]].items()
                  for k, sid in enumerate(key.split("|"))]
    return lines, stations, geoms


# =========================================================================== lines in pieces

# Lines still in pieces after join_pieces (cn_sources.md "Lines in pieces"), through pieces.py
# as gb_register: a gap is bridged over the track between the pieces where trains run across,
# the rest is one line per piece. China's own settings: OSM China has almost no train route
# relations (134), so no route is asked of a bridge and none is preferred (ROUTE_SHARE 0,
# UNROUTED_COST 1); instead the line's own named track costs half and, for a high-speed line,
# track not tagged highspeed=yes four times its length. MAX_KM 150: 京港高速线 runs 145 km over
# 昌九城际线 and Nanchang's lines between 庐山 and 南昌东, its own 南昌 - 九江 section being
# still under construction. KEEP_WHOLE: a gap that is track OSM does not have, on a line trains
# run through, stays one line in pieces until the track is mapped.
MAX_KM = 150.0
ROUTE_SHARE = 0.0
UNROUTED_COST = 1.0
OWN_COST = 0.5
SLOW_COST = 4.0
KEEP_WHOLE = {
    # OSM's track breaks at 烟台南 (121.38 E): the named track stops 0.8 km either side of the
    # station and its unnamed station roads join neither side, so the nearest track joining
    # 桃村北 and 牟平 is 284 km round by 桃威线 and 蓝烟线
    "青荣城际线",
}
# Never bridged, split instead. 成昆线's two pieces are what is left of the old line (成都 -
# 峨眉 - the mountain line to 攀枝花 and 花棚子, and 元谋西 - 昆明); the new line between,
# OSM's 峨广线 (峨眉 - 广通, 552 km), carries the through trains from 峨眉. Bridged, 攀枝花 -
# 元谋西 went over 121 km of 峨广线, and a ride 成都 - 昆明 entered on 成昆线 would have
# credited the old mountain line that no through train runs on.
NO_BRIDGE = {"成昆线"}
_S = {}          # wname (way -> line), ends, filled by build()
# {line id: [the ids of the pieces split off it]}, filled by split_pieces; build_model writes it
# into aliases.json as `pieces`.
LINE_PIECES = {}


def rules():
    import pieces
    return pieces.Rules(tag="CN", id_prefix="c", lat=32.0, max_km=MAX_KM,
                        route_share=ROUTE_SHARE, unrouted_cost=UNROUTED_COST,
                        own_cost=OWN_COST, slow_cost=SLOW_COST, keep_whole=KEEP_WHOLE,
                        piece_name=pieces.english_piece_name, dense=True,
                        no_bridge=NO_BRIDGE)


def classify(wid, tags, routed):
    """For pieces.track_graph: all heavy-rail track but industrial, military and test track
    (as `Net`), under the line build() gave each way ("" for none)."""
    if tags.get("railway") not in HEAVY or tags.get("usage") in ("industrial", "military",
                                                                  "test"):
        return None
    return _S.get("wname", {}).get(wid, "")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """The build_model hook: pieces.split_pieces with China's rules, over the ways
    register_way_lines loaded (state)."""
    import build_model as bm
    import pieces
    r = rules()

    def graph():
        if not state or "ways" not in state or "wname" not in _S:
            return None
        return pieces.track_graph(state["ways"], bm.Coords(state["cid"], state["cx"], state["cy"]),
                                  set(), classify, r, log)
    pieces.split_pieces(lines, stations, geoms, reg_ways, state, log, r, LINE_PIECES, graph,
                        _S.get("ends", ()))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    if "--clip" in sys.argv:
        clip()
    if "--dry" in sys.argv:
        import pickle
        t = time.time()
        res = build(str(RAW), lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True))
        with open(ROOT / "data" / "proc" / "cn" / "dry.pkl", "wb") as f:
            pickle.dump(res, f, protocol=4)
    if "--traffic" in sys.argv:
        traffic_pass(sys.argv[sys.argv.index("--traffic") + 1])
