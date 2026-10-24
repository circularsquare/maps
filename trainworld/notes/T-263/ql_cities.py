"""Attractions with >= 10 sitelinks in a few city boxes: count, sitelinks sum, visits a day at
10,000 visitors a year per sitelink, the top five. Read-only, QLever."""
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld")  # outputs go to artifacts
UA = "trainworld-poi-coverage/0.1 (research script)"
EP = "https://qlever.dev/api/wikidata"
PFX = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
KINDS = "wd:Q33506 wd:Q207694 wd:Q43501 wd:Q2281788 wd:Q194195 wd:Q2416723 wd:Q5393308 wd:Q845945 wd:Q2977 wd:Q570116 wd:Q12518 wd:Q16970 wd:Q44539 wd:Q32815"
BOX = {"New York": (40.5, 40.95, -74.3, -73.7), "London": (51.3, 51.7, -0.5, 0.3), "Tokyo": (35.5, 35.85, 139.55, 139.95),
       "Kyoto": (34.9, 35.1, 135.65, 135.85), "Shanghai": (30.9, 31.45, 121.2, 121.8), "Lagos": (6.3, 6.7, 3.2, 3.6),
       "Mumbai": (18.85, 19.3, 72.75, 73.05), "Seoul": (37.4, 37.7, 126.8, 127.2)}

res = {}
for city, (la0, la1, lo0, lo1) in BOX.items():
    q = PFX + f"""
SELECT DISTINCT ?s ?label ?sl ?v WHERE {{
  VALUES ?cls {{ {KINDS} }}
  ?s wdt:P31 ?cls ; wdt:P625 ?c ; wikibase:sitelinks ?sl . FILTER(?sl >= 10)
  BIND(geof:latitude(?c) AS ?la) BIND(geof:longitude(?c) AS ?lo)
  FILTER(?la >= {la0} && ?la <= {la1} && ?lo >= {lo0} && ?lo <= {lo1})
  OPTIONAL {{ ?s rdfs:label ?label FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?s wdt:P1174 ?v }}
}}"""
    url = EP + "?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            rows = [{k: v["value"] for k, v in b.items()} for b in json.load(r)["results"]["bindings"]]
    except Exception as e:  # noqa: BLE001
        print(city, "failed", e)
        continue
    best = {}
    for r in rows:
        best.setdefault(r["s"], r)
    items = sorted(best.values(), key=lambda r: -int(r["sl"]))
    sl = sum(int(r["sl"]) for r in items)
    res[city] = items
    top = ", ".join(f'{r.get("label", r["s"].rsplit("/", 1)[-1])} {r["sl"]}' + (f' ({float(r["v"]) / 1e6:.1f}M)' if r.get("v") else "") for r in items[:6])
    print(f"{city}: {len(items)} places, {sl} sitelinks, {sl * 10000 / 365:,.0f} visits a day; {top}")
json.dump(res, open(os.path.join(HERE, "ql_cities.json"), "w"))
