"""
Country-level reference data: name, and the languages spoken there.

Cheap (one SPARQL query, ~2s) and it seeds two card fields that have no
city-level source at all:

  * COUNTRY NAME, so the card can say "Japan" instead of "Q17"
  * LANGUAGE CANDIDATES -- P37 (official language) plus P2936 (language used).
    There is no global city-level language dataset; this is a starting
    suggestion to curate down from, exactly like the alt-name picker. It is
    stored as `languageCandidates`, never as `languages`, so a seed can never
    be mistaken for a curated answer.

Output: cache/countries.json
"""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "cache" / "countries.json"
UA = ("citybrowser/0.1 (https://anita.garden/; anitaxinchen@gmail.com) "
      "bot python-urllib/3")

# NO LABELS IN THIS QUERY. Two lessons stacked here.
#
# 1. GROUP_CONCAT, not a plain SELECT: P37 and P2936 are both multi-valued, so a
#    plain query returns their CROSS PRODUCT — a country with 5 official and 20
#    used languages yields 100 rows, and 120 countries produced a 2.6 MB
#    truncated response.
# 2. Even aggregated, joining rdfs:label for every language inside the group was
#    slow enough to 502 after retries. Aggregate the QIDs only, then resolve
#    those few hundred QIDs to names in one cheap second query.
Q = """
SELECT ?c
       (GROUP_CONCAT(DISTINCT ?off;  separator="|") AS ?offs)
       (GROUP_CONCAT(DISTINCT ?used; separator="|") AS ?useds)
WHERE {
  VALUES ?c { %s }
  OPTIONAL { ?c wdt:P37   ?off }
  OPTIONAL { ?c wdt:P2936 ?used }
} GROUP BY ?c
"""

# Labels for country and language QIDs, resolved in bulk at the end.
LABELS = """
SELECT ?s ?sLabel WHERE {
  VALUES ?s { %s }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

def sparql(q, tries=3):
    """The label joins make this query heavy -- 120 countries took 30s and the
    next batch 504'd. Retry, and let the caller use small batches."""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=75) as r:
                return json.loads(r.read())["results"]["bindings"]
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    {type(e).__name__} -- retrying in {wait}s", flush=True)
            time.sleep(wait)


def main():
    base = json.loads((HERE / "data" / "base.json").read_text(encoding="utf-8"))
    cqs = sorted({r["country"] for r in base.values() if r.get("country")})
    print(f"{len(cqs)} distinct countries")

    # Cache per batch: an earlier version accumulated everything in memory and
    # wrote once at the end, so one 504 on the second batch threw away the first
    # batch's 30 seconds of work too.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    STEP = 40
    todo = [c for c in cqs if c not in out]
    print(f"{len(out)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), STEP):
        batch = todo[i:i + STEP]
        t0 = time.monotonic()
        rows = sparql(Q % " ".join("wd:" + c for c in batch))
        for b in rows:
            q = b["c"]["value"].rsplit("/", 1)[-1]
            def split(k):
                v = b.get(k, {}).get("value") or ""
                return [x.rsplit("/", 1)[-1] for x in v.split("|") if x]
            out[q] = {"name": None,
                      "official": split("offs"),
                      "used": split("useds")}
        for c in batch:
            out.setdefault(c, {"name": None, "official": [], "used": []})
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0, sort_keys=True),
                       encoding="utf-8")
        el = time.monotonic() - t0
        print(f"  [{i+len(batch)}/{len(todo)}] {el:.1f}s", flush=True)
        if i + STEP < len(todo):
            time.sleep(max(5.0, 3 * el))

    # Resolve every country and language QID to an English name in one pass.
    need = set(out)
    for r in out.values():
        need.update(r["official"])
        need.update(r["used"])
    need = sorted(x for x in need if x.startswith("Q"))
    print(f"resolving {len(need)} labels")
    lab = {}
    for i in range(0, len(need), 300):
        chunk = need[i:i + 300]
        for b in sparql(LABELS % " ".join("wd:" + x for x in chunk)):
            lab[b["s"]["value"].rsplit("/", 1)[-1]] = b.get("sLabel", {}).get("value")
        if i + 300 < len(need):
            time.sleep(5)
    for q, r in out.items():
        r["name"] = lab.get(q) or r.get("name")
        r["official"] = [lab[x] for x in r["official"] if lab.get(x) and not lab[x].startswith("Q")]
        r["used"] = [lab[x] for x in r["used"] if lab.get(x) and not lab[x].startswith("Q")]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0, sort_keys=True),
                   encoding="utf-8")

    withlang = sum(1 for r in out.values() if r["official"] or r["used"])
    print(f"wrote {OUT.name}: {len(out)} countries, {withlang} with languages")
    for q in ("Q17", "Q30", "Q142", "Q39"):
        if q in out:
            r = out[q]
            print(f"  {r['name']}: official={r['official']} used={r['used'][:4]}")


if __name__ == "__main__":
    main()
