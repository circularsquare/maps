"""The same Wikidata coverage counts through QLever's Wikidata endpoint (WDQS is rate-limited and
timing out today). Read-only; results to ql_counts.json."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld", "T-263-ql_counts.json")
UA = "trainworld-poi-coverage/0.1 (research script)"
EP = "https://qlever.dev/api/wikidata"
PFX = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
"""

Q = {
    "visitors": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  ?s wdt:P1174 ?v ; wdt:P625 ?c ; wdt:P17 ?k . ?k wdt:P297 ?cc .
} GROUP BY ?cc ORDER BY DESC(?n)""",
    "visitors1m": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  ?s wdt:P1174 ?v ; wdt:P625 ?c ; wdt:P17 ?k . ?k wdt:P297 ?cc . FILTER(?v >= 1000000)
} GROUP BY ?cc ORDER BY DESC(?n)""",
    "stadium": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  VALUES ?cls { wd:Q483110 wd:Q641226 }
  ?s wdt:P31/wdt:P279* ?cls ; wdt:P625 ?c ; wdt:P17 ?k . ?k wdt:P297 ?cc .
} GROUP BY ?cc ORDER BY DESC(?n)""",
    "stadium_cap": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  VALUES ?cls { wd:Q483110 wd:Q641226 }
  ?s wdt:P31/wdt:P279* ?cls ; wdt:P625 ?c ; wdt:P17 ?k ; wdt:P1083 ?cap . ?k wdt:P297 ?cc .
} GROUP BY ?cc ORDER BY DESC(?n)""",
    "stadium_cap20k": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  VALUES ?cls { wd:Q483110 wd:Q641226 }
  ?s wdt:P31/wdt:P279* ?cls ; wdt:P625 ?c ; wdt:P17 ?k ; wdt:P1083 ?cap . ?k wdt:P297 ?cc . FILTER(?cap >= 20000)
} GROUP BY ?cc ORDER BY DESC(?n)""",
    "stadium_occupant": PFX + """
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n) WHERE {
  VALUES ?cls { wd:Q483110 wd:Q641226 }
  ?s wdt:P31/wdt:P279* ?cls ; wdt:P625 ?c ; wdt:P17 ?k ; wdt:P466 ?team . ?k wdt:P297 ?cc .
} GROUP BY ?cc ORDER BY DESC(?n)""",
    # what kinds of things carry a visitor count, worldwide
    "visitor_classes": PFX + """
SELECT ?cls (COUNT(DISTINCT ?s) AS ?n) WHERE {
  ?s wdt:P1174 ?v ; wdt:P625 ?c ; wdt:P31 ?cls .
} GROUP BY ?cls ORDER BY DESC(?n) LIMIT 40""",
}


def run(q):
    url = EP + "?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:] or Q:
        if res.get(name):
            continue
        t = time.time()
        try:
            res[name] = run(Q[name])
            print(name, len(res[name]), "rows", f"{time.time() - t:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(name, "failed", e, flush=True)
        json.dump(res, open(OUT, "w"), indent=0)
        time.sleep(3)


if __name__ == "__main__":
    main()
