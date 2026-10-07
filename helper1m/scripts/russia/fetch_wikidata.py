"""OKTMO codes (Wikidata P764) for the OSM municipal relations' wikidata items.

OSM tags only ~150 of the ~2,300 municipal relations with an OKTMO code, but
nearly all carry a wikidata= tag, and Wikidata holds the OKTMO for most of
them. prep_boundaries.py uses these as a second key beside the name. Wikidata
often keeps a unit's older code beside, or instead of, the current one, so
every value is kept.

Writes helper1m/data/russia/raw/osm/wikidata_oktmo.json  {qid: [codes]}
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2]
OSM = HELPER / "data" / "russia" / "raw" / "osm"
UA = "helper1m-research/1.0"
ENDPOINT = "https://query.wikidata.org/sparql"


def main():
    qids = set()
    for name in ("tags.json", "tags_city.json"):
        for e in json.loads((OSM / name).read_text(encoding="utf-8"))["elements"]:
            q = e["tags"].get("wikidata", "")
            if q.startswith("Q"):
                qids.add(q)
    qids = sorted(qids)
    out = {}
    for i in range(0, len(qids), 150):
        chunk = qids[i:i + 150]
        query = ("SELECT ?item ?oktmo WHERE { VALUES ?item { "
                 + " ".join(f"wd:{q}" for q in chunk)
                 + " } ?item wdt:P764 ?oktmo . }")
        req = urllib.request.Request(
            ENDPOINT, data=urllib.parse.urlencode({"query": query}).encode(),
            headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.load(r)
                break
            except Exception as e:  # 504 / 429 from a busy query service
                print(f"  {e}; retrying", flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            raise SystemExit("Wikidata query service kept failing")
        for b in res["results"]["bindings"]:
            q = b["item"]["value"].rsplit("/", 1)[1]
            out.setdefault(q, []).append(b["oktmo"]["value"])
        print(f"{i + len(chunk)}/{len(qids)} items, {len(out)} with OKTMO", flush=True)
        time.sleep(2)
    (OSM / "wikidata_oktmo.json").write_text(json.dumps(out, ensure_ascii=False, indent=0),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
