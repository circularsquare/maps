"""Russia stage 1: what Wikidata holds for Russian railway lines and stations.

    python probe_ru_wikidata.py            # fetch (cached in data/raw/ru/wd_*.json) and report

Measures, as multi_sources.md did per country: railway-line items with P17 Russia (Q159), with
OSM relation (P402), length (P2043), route number (P1671); railway stations; stations with an
ESR code (P2815); station adjacency qualified by line (P197 + pq:P81) and how many lines form
one chain. Crimea's items may carry P17 Ukraine, Russia or both, so stations are also counted
in a box around Crimea (a P131+ query on the Crimean regions timed out).
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ru"
UA = "noritetsu-rail-map/1.0"
EP = "https://query.wikidata.org/sparql"

Q = {
    "lines": """SELECT ?x ?xLabel ?cls ?osm ?len ?ref WHERE {
  ?x wdt:P17 wd:Q159 ; wdt:P31 ?cls . ?cls wdt:P279* wd:Q728937 .
  OPTIONAL { ?x wdt:P402 ?osm } OPTIONAL { ?x wdt:P2043 ?len } OPTIONAL { ?x wdt:P1671 ?ref }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
} LIMIT 40000""",
    "stations": """SELECT ?s ?esr WHERE {
  ?s wdt:P17 wd:Q159 ; wdt:P31 ?cls . ?cls wdt:P279* wd:Q55488 .
  OPTIONAL { ?s wdt:P2815 ?esr } } LIMIT 100000""",
    "esr_any": """SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s wdt:P2815 ?esr . }""",
    "adj": """SELECT ?s ?a ?line WHERE {
  ?s wdt:P17 wd:Q159 ; p:P197 ?st . ?st ps:P197 ?a ; pq:P81 ?line . } LIMIT 200000""",
    "crimea_st": """SELECT ?s ?esr ?c WHERE {
  SERVICE wikibase:box { ?s wdt:P625 ?loc .
    bd:serviceParam wikibase:cornerSouthWest "Point(32.4 44.35)"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerNorthEast "Point(36.7 46.0)"^^geo:wktLiteral . }
  ?s wdt:P31 ?cls . ?cls wdt:P279* wd:Q55488 .
  OPTIONAL { ?s wdt:P2815 ?esr } OPTIONAL { ?s wdt:P17 ?c } } LIMIT 5000""",
}


def fetch(key):
    path = RAW / f"wd_{key}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    data = urllib.parse.urlencode({"query": Q[key], "format": "json"}).encode()
    for i in range(3):
        req = urllib.request.Request(EP, data=data, headers={
            "User-Agent": UA, "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                rows = json.load(r)["results"]["bindings"]
            rows = [{k: v["value"].rsplit("/", 1)[-1] if v["value"].startswith(
                "http://www.wikidata.org/entity/") else v["value"] for k, v in b.items()}
                for b in rows]
            path.write_text(json.dumps(rows, ensure_ascii=False), "utf-8")
            time.sleep(3)
            return rows
        except Exception as e:
            print(f"  retry {key} {i}: {type(e).__name__} {str(e)[:100]}", file=sys.stderr)
            time.sleep(15 * (i + 1))
    return []


def main():
    lines = fetch("lines")
    li = defaultdict(lambda: {"cls": set(), "osm": set(), "len": None, "ref": set(), "lab": ""})
    for b in lines:
        L = li[b["x"]]
        L["cls"].add(b["cls"])
        L["lab"] = b.get("xLabel", "")
        if b.get("osm"):
            L["osm"].add(b["osm"])
        if b.get("len"):
            L["len"] = b["len"]
        if b.get("ref"):
            L["ref"].add(b["ref"])
    print(f"lines (P17 Q159, subclass of railway line): {len(li)}; with P402 "
          f"{sum(1 for x in li.values() if x['osm'])}, with P2043 "
          f"{sum(1 for x in li.values() if x['len'])}, with P1671 "
          f"{sum(1 for x in li.values() if x['ref'])}")
    cls = Counter(c for x in li.values() for c in x["cls"])
    print("  classes:", cls.most_common(12))
    st = fetch("stations")
    sts = {b["s"] for b in st}
    esr = {b["s"] for b in st if b.get("esr")}
    print(f"stations (P17 Q159): {len(sts)}; with ESR code P2815: {len(esr)}")
    anyesr = fetch("esr_any")
    if anyesr:
        print(f"  items with P2815 anywhere: {anyesr[0]['n']}")
    cr = fetch("crimea_st")
    print(f"stations in a Crimea box (32.4-36.7E, 44.35-46.0N): {len({b['s'] for b in cr})}, "
          f"with ESR {len({b['s'] for b in cr if b.get('esr')})}; P17 values "
          f"{Counter(b.get('c') for b in cr).most_common()}")

    adj = fetch("adj")
    per = defaultdict(set)
    for b in adj:
        per[b["line"]].add((b["s"], b["a"]))
    chained = {}
    for ln, e in per.items():
        par = {}

        def f(x):
            while par.setdefault(x, x) != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for a, b in e:
            par[f(a)] = f(b)
        if len({f(x) for x in list(par)}) == 1 and len(par) >= 3:
            chained[ln] = set(par)
    st_adj = {s for e in per.values() for s, _ in e}
    print(f"adjacency: {len(st_adj)} stations with P197+P81, on {len(per)} lines; "
          f"{len(chained)} lines chained (one piece, 3+), "
          f"{len({s for v in chained.values() for s in v})} stations on them")
    big = sorted(chained.items(), key=lambda kv: -len(kv[1]))[:30]
    for ln, v in big:
        lab = li[ln]["lab"] if ln in li else ln
        print(f"   {len(v):4d}  {lab}")
    tr4_names(li)


def base(n):
    """'Санкт-Петербург-Главный' -> 'санкт-петербург'; 'Волховстрой I' -> 'волховстрой'."""
    n = n.lower().replace("ё", "е")
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"\s+(i{1,3}|iv|[1-4])$", "", n.strip())
    n = re.sub(r"-(пассажирский|пассажирская|главный|московский|московское|сортировочная|"
               r"сортировочный|товарный|товарная|город|балтийский|витебский|финляндский|"
               r"ладожский|белорусский|курский|ярославский|казанский|киевский|рижский|"
               r"павелецкий|савеловский|ленинградский|восточный|западный|северный|южный)$",
               "", n.strip())
    return n.strip(" -")


def tr4_names(li):
    """How many tariff sections have a Wikidata line item named 'A — B' for their two ends."""
    p = RAW / "tr4_sections.json"
    if not p.exists():
        return
    secs = json.loads(p.read_text("utf-8"))
    labels = defaultdict(list)
    for q, x in li.items():
        parts = re.split(r"\s+[—–-]\s+", x["lab"])
        if len(parts) == 2:
            labels[frozenset((base(parts[0]), base(parts[1])))].append(x["lab"])
    hit = 0
    km = km_hit = 0
    for s in secs:
        if s["sheet"] in ("Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)") or not s["points"]:
            continue
        a, b = s["points"][0]["name"], s["points"][-1]["name"]
        k = frozenset((base(a), base(b)))
        L = max((pt["km"][0] or 0) for pt in s["points"])
        km += L
        if k in labels:
            hit += 1
            km_hit += L
    n = sum(1 for s in secs if s["sheet"] not in ("Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)"))
    print(f"\nTR-4 sections whose two end stations name a Wikidata line 'A — B': {hit} of {n} "
          f"({km_hit:,} of {km:,} tariff km); Wikidata 'A — B' labels: {len(labels)}")


if __name__ == "__main__":
    main()
