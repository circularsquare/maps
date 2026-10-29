"""Attractions for the game's attractions purpose (T-263 stage 2, notes/T-263.md, docs/demand-sources.md).

Writes app/src/game/attractions.json from Wikidata through QLever (qlever.dev; query.wikidata.org
rate-limits and times out on queries this size). An attraction is an item that is an instance of
one of `KINDS` (or a subclass) or of a `DIRECT` class, with coordinates, and is none of `NOT`.
Its visitors a year are its latest visitor figure (P1174) where it has one, else an estimate from
the places that have one (T-1739, `Model`): a log-linear fit of the figure on the item's sitelinks,
the people living within `POP_KM` of it (GHSL, 1 km world grid), whether it is an open scenic place
(a lake, hill or island) and its country (shrunk towards the rest, so a country with few figures
follows the world); the kind of place (rides, animals, museum) is in the fit as a control only. A lake, hill, island or reserve that Wikidata also gives a class under
"tourist attraction" (West Lake, Huangshan) counts in spite of its nature class. Kept at
`MIN_VISITORS` or more. Where one attraction lies inside another
(P276 location or P361 part of: the Louvre Museum in the Louvre Palace) or two sit within `SAME_M`
of each other, only the larger is kept: they draw the same visitors.

Output: {"rows": [[qid, name, lon, lat, visitors a year, source], ...]}, source 0 a visitor
figure, 1 the estimate; rows by visitors, largest first. Names are the English label, else the
multilingual one, else the label most languages share.

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

MIN_SITELINKS = 10
MIN_VISITORS = 100_000
SAME_M = 100.0
# The fit: places with a figure of at least this many and this many sitelinks; the people within
# POP_KM; the pull of a country's own places towards the rest, as a ridge penalty on its effect.
FIT_VISITORS = 50_000
FIT_SITELINKS = 3
POP_KM = 25
POP_FLOOR = 100_000
RIDGE = 3.0
# Off (T-1739): a country constant in demand is Anita's call; with it Hangzhou gains, see notes/T-1739.md.
COUNTRY_EFFECT = False
POP_TIF = ROOT / "data" / "raw" / "worldplan" / "GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.tif"
POP_CACHE = ROOT.parent / "artifacts" / "trainworld" / "t263" / "pop25.json"
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
# Presidential palaces and headquarters (not royal residences, which open) are open to few: counted only where
# Wikidata has a figure (the White House's tours), never estimated from editions (Zhongnanhai).
RESTRICTED = {"Q2114972": "presidential palace", "Q7540126": "headquarters"}
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

# A lake, hill, island or reserve that Wikidata also gives a class under "tourist attraction" (West
# Lake, Huangshan, a show cave) is an attraction in spite of its nature class, which is in NOT. The
# parks stay out (social trips draw them, T-196).
TOURIST = "Q570116"
SCENIC = {"Q23397": "lake", "Q8502": "mountain", "Q23442": "island", "Q4022": "river", "Q40080": "beach",
          "Q473972": "protected area", "Q46169": "national park", "Q131681": "reservoir", "Q54050": "hill"}

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
  UNION {{ ?s wdt:P31 ?k . ?k wdt:P279+ wd:{TOURIST} . VALUES ?n {{ {" ".join("wd:" + n for n in SCENIC)} }} ?s wdt:P31/wdt:P279* ?n }}
  ?s wdt:P625 ?g ; wikibase:sitelinks ?sl .
  FILTER(?sl >= {MIN_SITELINKS} || EXISTS {{ ?s wdt:P1174 ?any }})"""


# Open places (the scenic classes and parks) with a visitor figure: not attractions of the game's
# list, but they are the figures that say how many an open place draws, which the fit needs.
OPEN_FIGURES = f"""SELECT DISTINCT ?s ?sl ?g ?cc ?v ?t ?r WHERE {{
  VALUES ?c {{ {" ".join("wd:" + c for c in [*SCENIC, "Q22698", "Q22746"])} }}
  ?s wdt:P31 ?c ; wdt:P625 ?g ; wikibase:sitelinks ?sl ; p:P1174 ?st . ?st ps:P1174 ?v ; wikibase:rank ?r .
  OPTIONAL {{ ?st pq:P585 ?t }} OPTIONAL {{ ?s wdt:P17 ?n . ?n wdt:P297 ?cc }} }}"""


def fetch(raw):
    """Asks for each part `raw` lacks, keeping the answers after each."""
    for name, query in [
        ("items", f"SELECT DISTINCT ?s ?k ?g ?sl WHERE {{ {PATTERN} }}"),
        ("classes", f"SELECT DISTINCT ?s ?c WHERE {{ {PATTERN} ?s wdt:P31 ?c }}"),
        ("figures", f"SELECT DISTINCT ?s ?v ?t ?r WHERE {{ {PATTERN} ?s p:P1174 ?st . ?st ps:P1174 ?v ; wikibase:rank ?r . OPTIONAL {{ ?st pq:P585 ?t }} }}"),
        ("links", f"SELECT DISTINCT ?s ?o WHERE {{ {PATTERN} {{ ?s wdt:P276 ?o }} UNION {{ ?s wdt:P361 ?o }} }}"),
        ("countries", f"SELECT DISTINCT ?s ?cc WHERE {{ {PATTERN} ?s wdt:P17 ?c . ?c wdt:P297 ?cc }}"),
        ("openfigures", OPEN_FIGURES),
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


def group_rows(rows):
    by = {}
    for r in rows:
        by.setdefault(qid(r["s"]), []).append(r)
    return by


def people_within(points):
    """People within POP_KM of each (lon, lat), from the 1 km GHSL world grid; a circle in the
    grid's own metres (Mollweide, equal-area: distances are right to a few percent in the
    populated latitudes). Answers are kept in POP_CACHE by rounded position."""
    cache = json.loads(POP_CACHE.read_text(encoding="utf-8")) if POP_CACHE.exists() else {}
    key = lambda p: f"{p[0]:.4f},{p[1]:.4f}"
    todo = sorted({key(p): p for p in points if key(p) not in cache}.items())
    if todo:
        import numpy as np
        import pyproj
        import rasterio
        from rasterio.windows import Window

        print(f"people near {len(todo)} places...", end=" ", flush=True)
        to_grid = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
        with rasterio.open(POP_TIF) as ds:
            cell = abs(ds.transform.a)
            h = int(POP_KM * 1000 / cell) + 1
            yy, xx = np.mgrid[-h : h + 1, -h : h + 1]
            inside = np.hypot(xx, yy) * cell <= POP_KM * 1000
            for k, (lon, lat) in todo:
                col, row = ~ds.transform * to_grid.transform(lon, lat)
                w = ds.read(1, window=Window(int(col) - h, int(row) - h, 2 * h + 1, 2 * h + 1), boundless=True, fill_value=0)
                cache[k] = float(w[inside & (w > 0)].sum())
        POP_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print("done", flush=True)
    return [cache[key(p)] for p in points]


# The kinds the fit tells apart, each against "other". Ride parks, animal parks and museums are in
# the fit only so that a country's effect is not its mix of kinds (China's figures are mostly
# theme parks); an estimate does not use them, because the places that publish a figure are the
# large ones of their kind and that gap would lift every unmeasured ride park (an abandoned one
# among them). Open scenic places keep theirs: their figures come from their own sample.
RIDES = {"Q194195", "Q2416723", "Q740326"}
ANIMALS = {"Q43501", "Q2281788", "Q167346"}
MUSEUM = {"Q33506"}
GROUPS = ("rides", "animals", "museum", "open")
USED = {"open"}


def kind_of(kinds, open_place):
    if kinds & RIDES:
        return "rides"
    if kinds & ANIMALS:
        return "animals"
    if kinds & MUSEUM:
        return "museum"
    return "open" if open_place else "other"


class Model:
    """ln visitors = a + b ln sitelinks + c ln(people within POP_KM + POP_FLOOR) + the kind's
    effect + the country's, fitted by least squares on places with a figure. The country effects
    carry a ridge penalty (RIDGE): a country with a handful of figures stays near the world's
    line, one with many (Italy, the US) or a clear gap (China) leaves it. Fitted on each run, so
    new figures in Wikidata move it."""

    def __init__(self, sample):
        import numpy as np

        self.np = np
        self.countries = sorted({s["cc"] for s in sample}) if COUNTRY_EFFECT else []
        X = np.array([self.row(s) for s in sample])
        self.y = np.log([s["v"] for s in sample])
        self.X, self.n_fixed = X, 3 + len(GROUPS)
        self.b = self.solve(X, self.y)

    def row(self, s):
        return [1.0, math.log(s["sl"]), math.log(s["pop"] + POP_FLOOR)] + [float(s["group"] == g) for g in GROUPS] + [float(s["cc"] == c) for c in self.countries]

    def solve(self, X, y):
        np = self.np
        pen = np.zeros(X.shape[1])
        pen[self.n_fixed :] = RIDGE
        return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)

    def predict(self, s):
        return math.exp(float(self.np.array(self.row({**s, "group": s["group"] if s["group"] in USED else "other"})) @ self.b))

    def cv_rmse(self, folds=10):
        """Leave-a-tenth-out error of ln visitors: how far an estimate is from a real figure."""
        np = self.np
        order = np.random.default_rng(1).permutation(len(self.y))
        err = np.zeros(len(self.y))
        for f in range(folds):
            test = order[f::folds]
            train = np.setdiff1d(order, test)
            err[test] = self.y[test] - self.X[test] @ self.solve(self.X[train], self.y[train])
        return math.sqrt(float((err**2).mean()))

    def describe(self):
        eff = dict(zip(self.countries, self.b[self.n_fixed :]))
        top = sorted(eff, key=lambda c: -abs(eff[c]))
        fixed = dict(zip(["a", "sitelinks", "people"] + list(GROUPS), self.b[: self.n_fixed]))
        return (f"fit on {len(self.y)} places, error {self.cv_rmse():.2f} in ln visitors (x{math.exp(self.cv_rmse()):.1f}); "
                + ", ".join(f"{k} {v:.2f}" for k, v in fixed.items()) + "; countries: "
                + ", ".join(f"{c or '--'} x{math.exp(eff[c]):.2f}" for c in top[:12]))


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
    todo = []
    for it in by.values():
        v, year = figure(it["figs"])
        if v is not None and v > MAX_FIGURE:
            v, year = None, 0
        bad = it["classes"] & NOT.keys()
        if v is not None:
            bad -= NOT_UNLESS_FIGURE
        if RUIN in it["kinds"]:
            bad -= NOT_UNLESS_RUIN
        # an open place Wikidata also gives a tourist-attraction class is one in spite of its own
        it["open"] = not it["kinds"] & (KINDS.keys() | DIRECT.keys())
        if it["open"]:
            # only a place whose own class is a scenic one: a scenic byway is tagged too, and is a road
            if not it["classes"] & SCENIC.keys():
                continue
            bad -= SCENIC.keys()
        if bad or it["id"] in LEFT_OUT:
            dropped["not"].append(it)
            continue
        if v is None and (it["kinds"] <= NEEDS_FIGURE or it["classes"] & RESTRICTED.keys()):
            dropped["figure"] += 1
            continue
        it["fig"], it["year"] = v, year
        it["group"] = kind_of(it["kinds"], it["open"])
        todo.append(it)
    # visitors: the figure, else the fit's estimate
    sample = {it["id"]: it for it in todo if it["fig"] is not None and FIT_VISITORS <= it["fig"] and it["sl"] >= FIT_SITELINKS}
    for s, rows in group_rows(raw["openfigures"]).items():
        v, _ = figure(rows)
        if s not in sample and v is not None and FIT_VISITORS <= v <= MAX_FIGURE and int(rows[0]["sl"]) >= FIT_SITELINKS and (at := point(rows[0]["g"])):
            sample[s] = {"id": s, "lon": at[0], "lat": at[1], "sl": int(rows[0]["sl"]), "cc": rows[0].get("cc", ""), "group": "open", "fig": v}
    need = [it for it in todo if it["fig"] is None]
    pops = people_within([(it["lon"], it["lat"]) for it in [*sample.values(), *need]])
    for it, pop in zip([*sample.values(), *need], pops):
        it["pop"] = pop
    for it in sample.values():
        it["v"] = it["fig"]
    model = Model(list(sample.values()))
    for it in todo:
        it["v"], it["src"] = (it["fig"], 0) if it["fig"] is not None else (model.predict(it), 1)
    for it in todo:
        it["name"] = name_of(it["labels"])
        if it["v"] < MIN_VISITORS:
            dropped["small"] += 1
            continue
        if not it["name"]:
            dropped["unnamed"] += 1
            continue
        kept.append(it)
    # one of a pair that lies in the other, or within SAME_M of it: the larger stays
    # a place with a figure keeps its row against an estimated neighbour of any size
    rank = lambda it: (it["src"], -it["v"])
    kept.sort(key=lambda it: (*rank(it), it["id"]))
    ids = {it["id"]: it for it in kept}
    gone = set()
    for it in kept:
        if it["id"] in gone:
            continue
        for o in it["links"]:
            if o in ids and o not in gone and rank(ids[o]) >= rank(it) and metres(it, ids[o]) < LINK_M:
                gone.add(o)
    for it in kept:
        if any(o in ids and rank(ids[o]) < rank(it) and metres(it, ids[o]) < LINK_M for o in it["links"]):
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
    return [it for it in kept if it["id"] not in gone], dropped, merged, model


def main():
    raw = {} if "--fetch" in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text(encoding="utf-8"))
    raw = fetch(raw)
    rows, dropped, merged, model = build(raw)
    out = {"rows": [[int(it["id"][1:]), it["name"], round(it["lon"], 5), round(it["lat"], 5), int(round(it["v"])), it["src"]] for it in rows]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(rows)} attractions ({sum(1 for it in rows if it['src'] == 0)} by visitor figure), {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"dropped: {len(dropped['not'])} of a class never counted, {dropped['figure']} towers with no figure, {dropped['small']} under {MIN_VISITORS:,}, {dropped['unnamed']} unnamed; {len(merged)} inside or beside a larger one")
    if "--report" in sys.argv:
        report(rows, dropped, merged, model)


def report(rows, dropped, merged, model):
    sys.stdout.reconfigure(encoding="utf-8")
    print(model.describe())
    boxes = {"New York": (40.5, 40.95, -74.3, -73.7), "London": (51.3, 51.7, -0.5, 0.3), "Tokyo": (35.5, 35.85, 139.55, 139.95),
             "Shanghai": (30.9, 31.45, 121.2, 121.8), "Paris": (48.75, 48.95, 2.2, 2.5), "Kyoto": (34.9, 35.1, 135.65, 135.85), "Hangzhou": (30.1, 30.4, 120.0, 120.3)}
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
