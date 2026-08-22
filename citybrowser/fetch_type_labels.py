"""
Fetch English labels + descriptions for every P31 type used in base.json.

Small stage (~1,449 types, 4 queries) but it unlocks three things at once:
  * the card can say WHAT a point is ("city", "metropolitan area", "borough")
  * the settings panel can offer real categories instead of name heuristics
  * assemble_base.py can classify each city into a coarse `kind`

Output: cache/type_labels.json  { "Q515": {"label": "city", "desc": "..."} }

Same politeness rules as fetch_wikidata.py: duty-cycle throttle, descriptive
User-Agent, cached so a rerun costs nothing.
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
BASE = HERE / "data" / "base.json"
OUT = HERE / "cache" / "type_labels.json"

UA = ("citybrowser/0.1 (https://anita.garden/; anitaxinchen@gmail.com) "
      "bot python-urllib/3")
CHUNK = 350          # VALUES blocks much larger than this start timing out
DUTY_FACTOR = 4.0
MIN_GAP = 5.0

Q = """
SELECT ?s ?sLabel ?sDescription WHERE {
  VALUES ?s { %s }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def sparql(query):
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/sparql-results+json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=75) as r:
        body = r.read()
    return json.loads(body)["results"]["bindings"], time.monotonic() - t0


def main():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    want = set()
    for r in base.values():
        want.update(r.get("types") or [])

    have = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = sorted(want - set(have))
    print(f"{len(want):,} types in use, {len(have):,} cached, {len(todo):,} to fetch")
    if not todo:
        return

    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        rows, el = sparql(Q % " ".join("wd:" + t for t in batch))
        for x in rows:
            q = x["s"]["value"].rsplit("/", 1)[-1]
            have[q] = {"label": x.get("sLabel", {}).get("value", q),
                       "desc": x.get("sDescription", {}).get("value", "")}
        print(f"  [{i+len(batch)}/{len(todo)}] {el:.1f}s", flush=True)
        if i + CHUNK < len(todo):
            time.sleep(max(MIN_GAP, DUTY_FACTOR * el))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(have, ensure_ascii=False, indent=0, sort_keys=True),
                   encoding="utf-8")
    print(f"wrote {OUT.name}: {len(have):,} type labels")


if __name__ == "__main__":
    main()
