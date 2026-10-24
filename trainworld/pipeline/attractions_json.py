"""Attractions for the game's attractions purpose (T-263 stage 2, notes/T-263.md, docs/demand-sources.md).

Writes app/src/game/attractions.json from Wikidata through QLever (qlever.dev; query.wikidata.org
rate-limits and times out on queries this size). An attraction is an item that is an instance of
one of `KINDS` (or a subclass) or of a `DIRECT` class, with coordinates, and is none of `NOT`.
Its visitors a year are its latest visitor figure (P1174) where it has one, else `PER_EDITION`
for each Wikipedia language edition writing about it (its sitelinks: fitted on 2,102 places with
both, notes/T-263.md); kept at `MIN_VISITORS` or more. Where one attraction lies inside another
(P276 location or P361 part of: the Louvre Museum in the Louvre Palace) or two sit within `SAME_M`
of each other, only the larger is kept: they draw the same visitors.

Output: {"perEdition": 10000, "rows": [[qid, name, lon, lat, visitors a year, source], ...]},
source 0 a visitor figure, 1 language editions; rows by visitors, largest first. Names are the
English label, else the multilingual one, else the label most languages share.

Run from the maps root (`--fetch` asks QLever again; otherwise its last answers, kept in
artifacts/trainworld/t263/wikidata.json, are reused):
    venv/Scripts/python.exe trainworld/pipeline/attractions_json.py [--fetch]
"""

import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "src" / "game" / "attractions.json"
CACHE = ROOT.parent / "artifacts" / "trainworld" / "t263" / "wikidata.json"
EP = "https://qlever.dev/api/wikidata"
UA = "trainworld-attractions/1.0 (game data script)"

PER_EDITION = 10_000
MIN_VISITORS = 100_000
SAME_M = 100.0
# Figures for 2020 and 2021 are the pandemic's: an older figure is taken over them.
PANDEMIC = {2020, 2021}

# With their subclasses.
KINDS = {
    "Q33506": "museum", "Q43501": "zoo", "Q2281788": "aquarium", "Q194195": "amusement park",
    "Q2416723": "theme park", "Q740326": "water park", "Q167346": "botanical garden",
    "Q148319": "planetarium", "Q44539": "temple", "Q5393308": "Buddhist temple",
    "Q845945": "Shinto shrine", "Q842402": "Hindu temple", "Q16970": "church", "Q2977": "cathedral",
    "Q163687": "basilica", "Q32815": "mosque", "Q34627": "synagogue", "Q23413": "castle",
    "Q16560": "palace", "Q57821": "fortification", "Q1440300": "observation tower",
    "Q4989906": "monument", "Q179700": "statue", "Q839954": "archaeological site",
}
# Direct class only: the subclasses of tourist attraction run to rides and trails, of tower and
# skyscraper to masts and office blocks.
DIRECT = {"Q570116": "tourist attraction", "Q12518": "tower", "Q11303": "skyscraper"}
# A tower or skyscraper counts only with a visitor figure: it is an attraction for its observation
# deck, and the figure counts the deck's visitors.
NEEDS_FIGURE = {"Q12518", "Q11303"}
# Never an attraction, whatever else it is: places people pass through or live in, parks and
# nature (parks draw social trips already, T-196), stadiums and arenas (events, a later stage),
# and buildings whose visitors are their users.
NOT = {
    "Q486972": "human settlement", "Q515": "city", "Q1549591": "big city", "Q5119": "capital",
    "Q123705": "neighbourhood", "Q15243209": "historic district", "Q174782": "square",
    "Q79007": "street", "Q12280": "bridge", "Q158438": "arch bridge", "Q12570": "suspension bridge",
    "Q22698": "park", "Q22746": "urban park", "Q46169": "national park", "Q473972": "protected area",
    "Q8502": "mountain", "Q40080": "beach", "Q23442": "island", "Q4022": "river", "Q23397": "lake",
    "Q483110": "stadium", "Q641226": "arena", "Q1076486": "sports venue", "Q55488": "railway station",
    "Q928830": "metro station", "Q1248784": "airport", "Q3918": "university", "Q7075": "library",
    "Q16917": "hospital", "Q27686": "hotel", "Q11315": "shopping centre", "Q31374": "shopping mall",
    "Q1021645": "office building", "Q1137809": "courthouse", "Q39614": "cemetery",
}
# Of those, an office building with a visitor figure counts (the figure is its observation deck's:
# the Empire State Building), and a stadium or arena that is an archaeological site (the
# Colosseum).
NOT_UNLESS_FIGURE = {"Q1021645"}
NOT_UNLESS_RUIN = {"Q483110", "Q641226", "Q1076486"}
RUIN = "Q839954"
# Found by reading the largest: figures that count passers-by, and places whose visitors another
# row counts already (the resort beside its parks, a statue inside its monument). Left out.
LEFT_OUT = {
    "Q71719": "Hollywood Walk of Fame: passers-by",
    "Q206521": "Disneyland Paris, the resort: its two parks are rows of their own",
    "Q9202": "Statue of Liberty: the Statue of Liberty National Monument counts its visitors",
    "Q3618777": "Antiquarium in Pompeii: its figure is the whole site's, a row of its own",
}
# A figure above this is a whole institution's (the Smithsonian's 30M) or a slip (281M for the
# Royal Ontario Museum): the place is sized by its language editions instead. The largest single
# places are about 20M a year (the Forbidden City, the Grand Palace).
MAX_FIGURE = 25_000_000
# Links (location, part of) join two attractions only this close: the Udvar-Hazy Center is part
# of the National Air and Space Museum 40 km away, and draws its own visitors.
LINK_M = 500.0

# Labels read: English and multilingual first; for a place with neither, the label most of these
# languages share (in Europe and the Americas the place's own name, elsewhere a romanised one).
LANGS = ", ".join(f'"{x}"' for x in "en mul fr de es it pt nl pl sv cs ru tr id vi ja zh ko ar hi".split())

PREFIX = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def run(query):
    body = urllib.parse.urlencode({"query": PREFIX + query}).encode()
    req = urllib.request.Request(EP, data=body, headers={"User-Agent": UA, "Accept": "application/sparql-results+json", "Content-Type": "application/x-www-form-urlencoded"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            print(f"(429, waiting {60 * (attempt + 1)} s)", end=" ", flush=True)
            time.sleep(60 * (attempt + 1))
    out = []
    for b in data["results"]["bindings"]:
        row = {k: v["value"] for k, v in b.items()}
        for k, v in b.items():
            if "xml:lang" in v:
                row[k + "_lang"] = v["xml:lang"]
        out.append(row)
    return out


def qid(uri):
    return uri.rsplit("/", 1)[-1]


PATTERN = f"""{{ VALUES ?k {{ {" ".join("wd:" + k for k in KINDS)} }} ?s wdt:P31/wdt:P279* ?k }}
  UNION {{ VALUES ?k {{ {" ".join("wd:" + k for k in DIRECT)} }} ?s wdt:P31 ?k }}
  ?s wdt:P625 ?g ; wikibase:sitelinks ?sl .
  FILTER(?sl >= {MIN_VISITORS // PER_EDITION} || EXISTS {{ ?s wdt:P1174 ?any }})"""


def fetch(raw):
    """Asks for each part `raw` lacks, keeping the answers after each."""
    for name, query in [
        ("items", f"SELECT DISTINCT ?s ?k ?g ?sl WHERE {{ {PATTERN} }}"),
        ("classes", f"SELECT DISTINCT ?s ?c WHERE {{ {PATTERN} ?s wdt:P31 ?c }}"),
        ("figures", f"SELECT DISTINCT ?s ?v ?t ?r WHERE {{ {PATTERN} ?s p:P1174 ?st . ?st ps:P1174 ?v ; wikibase:rank ?r . OPTIONAL {{ ?st pq:P585 ?t }} }}"),
        ("links", f"SELECT DISTINCT ?s ?o WHERE {{ {PATTERN} {{ ?s wdt:P276 ?o }} UNION {{ ?s wdt:P361 ?o }} }}"),
        ("countries", f"SELECT DISTINCT ?s ?cc WHERE {{ {PATTERN} ?s wdt:P17 ?n . ?n wdt:P297 ?cc }}"),
    ]:
        if name in raw:
            continue
        print(f"{name}...", end=" ", flush=True)
        raw[name] = run(query)
        print(len(raw[name]), flush=True)
        keep(raw)
    # the labels by batches of items: asked for all at once QLever answers 429
    ids = sorted({qid(r["s"]) for r in raw["items"]})
    raw.setdefault("labels", [])
    raw.setdefault("labelled", 0)
    for at in range(raw["labelled"], len(ids), LABEL_BATCH):
        print(f"labels {at}...", end=" ", flush=True)
        values = " ".join("wd:" + s for s in ids[at : at + LABEL_BATCH])
        rows = run(f"SELECT ?s ?l WHERE {{ VALUES ?s {{ {values} }} ?s rdfs:label ?l FILTER(LANG(?l) IN ({LANGS})) }}")
        print(len(rows), flush=True)
        raw["labels"] += rows
        raw["labelled"] = at + LABEL_BATCH
        keep(raw)
    return raw


LABEL_BATCH = 2000


def keep(raw):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(raw), encoding="utf-8")
    time.sleep(3)


def point(wkt):
    """Longitude and latitude, or None for a point on another globe or none at all."""
    if wkt.startswith("<") or "(" not in wkt or ")" not in wkt:
        return None
    lon, lat = wkt[wkt.index("(") + 1 : wkt.index(")")].split()[:2]
    return float(lon), float(lat)


def year_of(t):
    try:
        return int(t.lstrip("+")[:4])
    except (AttributeError, ValueError):
        return 0


def figure(rows):
    """The latest visitor figure, a dated one outside the pandemic years first; ties to the
    largest. Deprecated statements are skipped."""
    rows = [r for r in rows if not r.get("r", "").endswith("DeprecatedRank") and not r["v"].startswith("http")]
    if not rows:
        return None, 0
    best = max(rows, key=lambda r: (year_of(r.get("t")) not in PANDEMIC | {0}, year_of(r.get("t")), float(r["v"])))
    return float(best["v"]), year_of(best.get("t"))


def name_of(labels):
    if "en" in labels:
        return labels["en"]
    if "mul" in labels:
        return labels["mul"]
    count = {}
    for v in labels.values():
        count[v] = count.get(v, 0) + 1
    return max(count, key=lambda v: (count[v], -len(v))) if count else None


def metres(a, b):
    kx = 111_320 * math.cos(math.radians((a["lat"] + b["lat"]) / 2))
    return math.hypot((a["lon"] - b["lon"]) * kx, (a["lat"] - b["lat"]) * 110_540)


def build(raw):
    by = {}
    for r in raw["items"]:
        s = qid(r["s"])
        if s not in by:
            if not (at := point(r["g"])):
                continue
            lon, lat = at
            by[s] = {"id": s, "lon": lon, "lat": lat, "sl": int(r["sl"]), "kinds": set(), "classes": set(), "figs": [], "labels": {}, "links": set(), "cc": ""}
        by[s]["kinds"].add(qid(r["k"]))
    for key, field in (("classes", "c"), ("links", "o")):
        for r in raw[key]:
            if (s := qid(r["s"])) in by:
                by[s][key].add(qid(r[field]))
    for r in raw["figures"]:
        if (s := qid(r["s"])) in by:
            by[s]["figs"].append(r)
    for r in raw["countries"]:
        if (s := qid(r["s"])) in by:
            by[s]["cc"] = by[s]["cc"] or r["cc"]
    for r in raw["labels"]:
        if (s := qid(r["s"])) in by:
            by[s]["labels"].setdefault(r.get("l_lang", ""), r["l"])
    kept, dropped = [], {"not": [], "figure": 0, "small": 0, "unnamed": 0}
    for it in by.values():
        v, year = figure(it["figs"])
        if v is not None and v > MAX_FIGURE:
            v, year = None, 0
        bad = it["classes"] & NOT.keys()
        if v is not None:
            bad -= NOT_UNLESS_FIGURE
        if RUIN in it["kinds"]:
            bad -= NOT_UNLESS_RUIN
        if bad or it["id"] in LEFT_OUT:
            dropped["not"].append(it)
            continue
        if v is None and it["kinds"] <= NEEDS_FIGURE:
            dropped["figure"] += 1
            continue
        it["v"], it["src"], it["year"] = (v, 0, year) if v is not None else (PER_EDITION * it["sl"], 1, 0)
        it["name"] = name_of(it["labels"])
        if it["v"] < MIN_VISITORS:
            dropped["small"] += 1
            continue
        if not it["name"]:
            dropped["unnamed"] += 1
            continue
        kept.append(it)
    # one of a pair that lies in the other, or within SAME_M of it: the larger stays
    kept.sort(key=lambda it: (-it["v"], it["id"]))
    ids = {it["id"]: it for it in kept}
    gone = set()
    for it in kept:
        if it["id"] in gone:
            continue
        for o in it["links"]:
            if o in ids and o not in gone and ids[o]["v"] <= it["v"] and metres(it, ids[o]) < LINK_M:
                gone.add(o)
    for it in kept:
        if any(o in ids and ids[o]["v"] > it["v"] and metres(it, ids[o]) < LINK_M for o in it["links"]):
            gone.add(it["id"])
    # nearby pairs by a grid of SAME_M, the larger (earlier) placed first
    grid = {}
    for it in kept:
        if it["id"] in gone:
            continue
        gx, gy = int(it["lon"] * 111_320 * math.cos(math.radians(it["lat"])) // SAME_M), int(it["lat"] * 110_540 // SAME_M)
        near = [o for dx in (-1, 0, 1) for dy in (-1, 0, 1) for o in grid.get((gx + dx, gy + dy), []) if metres(it, o) < SAME_M]
        if near:
            gone.add(it["id"])
            continue
        grid.setdefault((gx, gy), []).append(it)
    merged = [it for it in kept if it["id"] in gone]
    return [it for it in kept if it["id"] not in gone], dropped, merged


def main():
    raw = {} if "--fetch" in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text(encoding="utf-8"))
    raw = fetch(raw)
    rows, dropped, merged = build(raw)
    out = {"perEdition": PER_EDITION, "rows": [[int(it["id"][1:]), it["name"], round(it["lon"], 5), round(it["lat"], 5), int(round(it["v"])), it["src"]] for it in rows]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(rows)} attractions ({sum(1 for it in rows if it['src'] == 0)} by visitor figure), {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"dropped: {len(dropped['not'])} of a class never counted, {dropped['figure']} towers with no figure, {dropped['small']} under {MIN_VISITORS:,}, {dropped['unnamed']} unnamed; {len(merged)} inside or beside a larger one")
    if "--report" in sys.argv:
        report(rows, dropped, merged)


def report(rows, dropped, merged):
    sys.stdout.reconfigure(encoding="utf-8")
    boxes = {"New York": (40.5, 40.95, -74.3, -73.7), "London": (51.3, 51.7, -0.5, 0.3), "Tokyo": (35.5, 35.85, 139.55, 139.95),
             "Shanghai": (30.9, 31.45, 121.2, 121.8), "Paris": (48.75, 48.95, 2.2, 2.5), "Kyoto": (34.9, 35.1, 135.65, 135.85)}
    print("\ncity boxes (attractions, visits a day on average, largest):")
    for city, (la0, la1, lo0, lo1) in boxes.items():
        inside = [it for it in rows if la0 <= it["lat"] <= la1 and lo0 <= it["lon"] <= lo1]
        top = ", ".join(f"{it['name']} {it['v'] / 1e6:.1f}M{'' if it['src'] == 0 else ' (ed)'}" for it in inside[:8])
        print(f"  {city}: {len(inside)}, {sum(it['v'] for it in inside) / 365:,.0f} a day; {top}")
    print("\nlargest:")
    for it in rows[:120]:
        print(f"  {it['id']:>11} {it['cc']:2} {it['v'] / 1e6:6.2f}M {'fig ' + str(it['year']) if it['src'] == 0 else 'sl  ' + str(it['sl']):>9}  {it['name']}")
    by_cc = {}
    for it in rows:
        c = by_cc.setdefault(it["cc"], [0, 0, 0.0])
        c[0] += 1
        c[1] += it["src"] == 0
        c[2] += it["v"]
    print("\nby country (attractions, with a figure, visitors a year M):")
    for cc in sorted(by_cc, key=lambda c: -by_cc[c][2])[:30]:
        n, f, v = by_cc[cc]
        print(f"  {cc or '--'} {n:6} {f:5} {v / 1e6:8.1f}")
    print("\nlargest dropped by class:")
    for it in sorted(dropped["not"], key=lambda it: -it["sl"])[:25]:
        print(f"  {it['id']:>11} sl {it['sl']:4} {name_of(it['labels'])} {sorted(NOT[c] for c in it['classes'] & NOT.keys())}")
    print("\nlargest merged away:")
    for it in merged[:25]:
        print(f"  {it['id']:>11} {it['v'] / 1e6:6.2f}M {it['name']}")


if __name__ == "__main__":
    main()
