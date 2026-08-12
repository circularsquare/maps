"""Throwaway diagnostic: find a settlement query WDQS will actually answer.

Runs a few query shapes against one small country, with the same duty-cycle
politeness as the real fetcher (sleep 4x the previous query's duration). Prints
elapsed time and either the row count or the failure mode.
"""
import json, time, urllib.error, urllib.parse, urllib.request

UA = ("citybrowser/0.1 (https://anita.garden/; anitaxinchen@gmail.com) "
      "bot python-urllib/3")
GA = "Q1000"   # Gabon — small, so anything slow here is the query's fault

VARIANTS = {
    "A no-type, count": """
SELECT (COUNT(*) AS ?n) WHERE {
  ?c wdt:P17 wd:%s ; wdt:P625 ?g ; wdt:P1082 ?p . FILTER(?p >= 15000)
}""",

    "B no-type, grouped rows": """
SELECT ?c ?pop ?coord ?elev ?admin WHERE {
  SELECT ?c (MAX(?p) AS ?pop) (SAMPLE(?g) AS ?coord)
            (SAMPLE(?e) AS ?elev) (SAMPLE(?a) AS ?admin) WHERE {
    ?c wdt:P17 wd:%s ; wdt:P625 ?g ; wdt:P1082 ?p . FILTER(?p >= 15000)
    OPTIONAL { ?c wdt:P2044 ?e } OPTIONAL { ?c wdt:P131 ?a }
  } GROUP BY ?c
}""",

    "C no-type, grouped + labels": """
SELECT ?c ?cLabel ?pop ?coord ?elev ?admin ?adminLabel WHERE {
  {
    SELECT ?c (MAX(?p) AS ?pop) (SAMPLE(?g) AS ?coord)
              (SAMPLE(?e) AS ?elev) (SAMPLE(?a) AS ?admin) WHERE {
      ?c wdt:P17 wd:%s ; wdt:P625 ?g ; wdt:P1082 ?p . FILTER(?p >= 15000)
      OPTIONAL { ?c wdt:P2044 ?e } OPTIONAL { ?c wdt:P131 ?a }
    } GROUP BY ?c
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }
}""",

    "D shallow type (P31 only)": """
SELECT (COUNT(*) AS ?n) WHERE {
  ?c wdt:P31 ?t ; wdt:P17 wd:%s ; wdt:P625 ?g ; wdt:P1082 ?p .
  FILTER(?p >= 15000)
  VALUES ?t { wd:Q515 wd:Q3957 wd:Q532 wd:Q486972 wd:Q1549591 wd:Q15284 }
}""",
}


def run(name, q):
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/sparql-results+json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=75) as r:
            body = r.read()
        el = time.monotonic() - t0
        try:
            b = json.loads(body)["results"]["bindings"]
        except (ValueError, KeyError):
            print(f"  {name:30s} {el:6.1f}s  NON-JSON ({len(body)}b)")
            return el
        if b and "n" in b[0]:
            print(f"  {name:30s} {el:6.1f}s  count={b[0]['n']['value']}")
        else:
            print(f"  {name:30s} {el:6.1f}s  rows={len(b)}")
        return el
    except urllib.error.HTTPError as e:
        el = time.monotonic() - t0
        print(f"  {name:30s} {el:6.1f}s  HTTP {e.code}")
        return el
    except Exception as e:
        el = time.monotonic() - t0
        print(f"  {name:30s} {el:6.1f}s  {type(e).__name__}")
        return el


print("Gabon (Q1000). Duty-cycle spaced: sleep 4x previous query duration.\n")
for name, tmpl in VARIANTS.items():
    el = run(name, tmpl % GA)
    nap = max(5.0, 4 * el)
    print(f"     (sleeping {nap:.0f}s)", flush=True)
    time.sleep(nap)
print("\ndone")
