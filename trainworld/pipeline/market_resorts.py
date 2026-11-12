"""T-3240 / T-3247: beach and resort towns for the long-trip pull, pipeline/market_resorts.json.

    python pipeline/market_resorts.py

Two global sources, merged by Wikidata item: Wikidata items that are an instance of seaside resort (Q1021711),
resort town (Q317548) or tourist resort (Q10277595), and English Wikipedia's "Seaside resorts" category tree (its
country categories, two levels down). Each town keeps its coordinates and its Wikidata population (P1082, the
latest) where it has one. `extra` in the output is kept as it is: towns added by hand (name, lon, lat, pop).
Network: query.wikidata.org (one query, its limit is a query a minute), the Wikipedia and Wikidata APIs.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "market_resorts.json"
UA = {"User-Agent": "trainworld-pipeline/0.1 (research script)"}
CLASSES = ["Q1021711", "Q317548", "Q10277595"]


def get(url, data=None, accept="application/json"):
    for attempt in range(8):
        time.sleep(1.0)  # the APIs' limits are per second
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, "Accept": accept})
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception as e:  # WDQS answers its limit with an HTML page
            print("  retry", attempt, e, flush=True)
            time.sleep(70 if "query.wikidata.org" in url else 10 * 2 ** attempt)
    raise RuntimeError(url)


def wikidata_towns():
    q = ("SELECT ?x ?xLabel ?coord WHERE { VALUES ?cls { " + " ".join("wd:" + c for c in CLASSES) + " } "
         "?x wdt:P31 ?cls ; wdt:P625 ?coord . SERVICE wikibase:label { bd:serviceParam wikibase:language \"en,mul\". } }")
    d = get("https://query.wikidata.org/sparql", urllib.parse.urlencode({"query": q}).encode(), "application/sparql-results+json")
    out = {}
    for b in d["results"]["bindings"]:
        m = re.fullmatch(r"Point\(([-+\d.eE]+) ([-+\d.eE]+)\)", b["coord"]["value"])
        if not m:  # another globe's, or malformed
            continue
        lon, lat = m.groups()
        out[b["x"]["value"].rsplit("/", 1)[1]] = {"name": b["xLabel"]["value"], "lon": float(lon), "lat": float(lat), "via": "wikidata"}
    return out


def wiki(params):
    return get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"}))


def category_pages(root, depth):
    pages, seen, todo = set(), set(), [(root, 0)]
    while todo:
        cat, d = todo.pop()
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while True:
            r = wiki({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500, **cont})
            for m in r["query"]["categorymembers"]:
                if m["ns"] == 14 and d < depth and not m["title"].startswith("Category:Lists"):
                    todo.append((m["title"], d + 1))
                elif m["ns"] == 0 and not m["title"].startswith("List"):
                    pages.add(m["title"])
            if "continue" not in r:
                break
            cont = r["continue"]
    return sorted(pages)


def wikipedia_towns():
    titles = category_pages("Category:Seaside resorts", 2)
    out = {}
    for k in range(0, len(titles), 50):
        r = wiki({"action": "query", "prop": "coordinates|pageprops", "colimit": "max", "ppprop": "wikibase_item", "titles": "|".join(titles[k:k + 50])})
        for p in r["query"]["pages"].values():
            q, c = p.get("pageprops", {}).get("wikibase_item"), p.get("coordinates")
            if q and c:
                out[q] = {"name": p["title"], "lon": c[0]["lon"], "lat": c[0]["lat"], "via": "wikipedia"}
    print(f"wikipedia: {len(titles)} pages, {len(out)} with coordinates", flush=True)
    return out


def populations(qids):
    pop = {}
    for k in range(0, len(qids), 50):
        r = get("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(qids[k:k + 50]), "props": "claims", "format": "json"}))
        for q, e in r.get("entities", {}).items():
            best = None
            for c in e.get("claims", {}).get("P1082", []):
                v = c["mainsnak"].get("datavalue", {}).get("value", {}).get("amount")
                t = (c.get("qualifiers", {}).get("P585") or [{}])[0].get("datavalue", {}).get("value", {}).get("time", "")
                if v and (best is None or t > best[0]):
                    best = (t, float(v))
            if best:
                pop[q] = best[1]
    return pop


def main():
    towns = wikipedia_towns()
    for q, t in wikidata_towns().items():
        towns.setdefault(q, t)
    pop = populations(sorted(towns))
    extra = json.loads(OUT.read_text(encoding="utf-8")).get("extra", []) if OUT.exists() else []
    out = {"task": "T-3240", "source": "Wikidata (CC0) P31 seaside resort, resort town, tourist resort; English Wikipedia's Seaside resorts category tree; population Wikidata P1082",
           "towns": [{"q": q, **t, "pop": pop.get(q)} for q, t in sorted(towns.items())] + extra, "extra": extra}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{len(out['towns'])} towns ({len(extra)} by hand), {sum(1 for t in out['towns'] if t.get('pop'))} with people; wrote {OUT}")


if __name__ == "__main__":
    main()
