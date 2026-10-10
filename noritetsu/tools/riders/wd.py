"""Small Wikidata SPARQL helper for the station ridership sources.

Queries go to query.wikidata.org first and fall back to the QLever mirror (the flavour-stats
research of 2026-10-07 found WDQS rate-limited during an outage). Results are cached as JSON in
data/raw/riders/<folder>/ so a rerun reads the file instead of asking again; pass refresh=True
to ask again.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (compatible; station-riders-script)"
ENDPOINTS = [
    "https://qlever.dev/api/wikidata",
    "https://query.wikidata.org/sparql",
]
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
"""


def sparql(query, cache, refresh=False):
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    q = PREFIXES + query
    last = None
    for ep in ENDPOINTS:
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    ep + "?" + urllib.parse.urlencode({"query": q}),
                    headers={"User-Agent": UA,
                             "Accept": "application/sparql-results+json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read().decode("utf-8"))
                rows = []
                for b in data["results"]["bindings"]:
                    rows.append({k: v.get("value") for k, v in b.items()})
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                return rows
            except Exception as e:      # noqa: BLE001 - try the next endpoint
                last = e
                print(f"  wikidata: {ep}: {e}")
                time.sleep(5)
    raise RuntimeError(f"Wikidata query failed: {last}")


def qid(uri):
    return uri.rsplit("/", 1)[-1] if uri else None


def point(wkt):
    """'Point(lon lat)' -> (lon, lat)."""
    m = re.search(r"point\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\)", wkt or "", re.I)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))
