"""
Stage 3: aliases + Wikipedia links for every city in base.json.

One pass over the Action API's wbgetentities gives three things at once:

  * ALT-NAME CANDIDATES -- labels and aliases in every language, so curation is
    picking from a list rather than typing into a blank box
  * the WIKIPEDIA LINK, which is where the hand-written facts actually come from
  * searchable names, so "Koln", "Cologne" and "Koeln" all find the same city

Different API from stage 1 and different limits: this is the Action API, which
allows ~200 req/min for a client with a compliant User-Agent (the WDQS
query-seconds budget does NOT apply here). We use ~60/min anyway.

50 entities per request is the API maximum, so ~1,240 requests for the roster.
Cached per batch, so an interrupted run resumes for free.

Usage:
    python fetch_entities.py --test     # one batch, print what comes back
    python fetch_entities.py            # the real run, resumable
"""
import argparse
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
CACHE = HERE / "cache" / "entities"

API = "https://www.wikidata.org/w/api.php"
UA = ("citybrowser/0.1 (https://anita.garden/; anitaxinchen@gmail.com) "
      "bot python-urllib/3")

BATCH = 50          # wbgetentities hard maximum
GAP = 1.0           # ~60 req/min against a 200/min allowance
MAX_ALT = 14        # candidates kept per city

# Ranking languages for the candidate list. This ordering matters more than it
# looks: a naive "labels first" ranking surfaced Arabic, Russian, Japanese and
# Korean TRANSLITERATIONS of "Boston" ahead of anything useful, because every
# language has a label for a big city while genuinely different names are rare.
#
# English ALIASES are what you actually want -- "Constantinople", "Saigon",
# "Bombay", "Beantown" all live there. Native-script labels (Tokyo -> 東京) are
# still worth having, so they rank next; the long tail of transliterations sits
# at the bottom where it belongs.
PRIORITY = ["en", "es", "fr", "de", "zh", "ar", "ru", "pt", "ja",
            "hi", "bn", "it", "ko", "tr", "fa", "id", "nl", "pl", "vi"]


def _rank(lang, is_alias):
    idx = PRIORITY.index(lang) if lang in PRIORITY else None
    if lang == "en":
        return 0 if is_alias else 1          # real alternate names first
    if idx is not None:
        return (10 + idx) if is_alias else (30 + idx)
    return 60 if is_alias else 70


def norm(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                hdr = e.headers.get("Retry-After") if e.headers else None
                delay = (int(hdr) if hdr and hdr.isdigit() else 60) + 15
                print(f"    429 -- sleeping {delay}s", flush=True)
                time.sleep(delay)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(20 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("wbgetentities failed after retries")


def distill(ent, name):
    """-> {'wiki': title|None, 'alt': [[lang, text], ...]}"""
    seen = {norm(name)}
    scored = []
    for lang, v in (ent.get("labels") or {}).items():
        t = v.get("value")
        if t and norm(t) not in seen:
            seen.add(norm(t))
            scored.append((_rank(lang, False), lang, t))
    for lang, arr in (ent.get("aliases") or {}).items():
        for v in arr:
            t = v.get("value")
            if t and norm(t) not in seen:
                seen.add(norm(t))
                scored.append((_rank(lang, True), lang, t))
    scored.sort(key=lambda x: (x[0], len(x[2])))
    wiki = (ent.get("sitelinks") or {}).get("enwiki", {}).get("title")
    return {"wiki": wiki, "alt": [[l, t] for _, l, t in scored[:MAX_ALT]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    base = json.loads(BASE.read_text(encoding="utf-8"))
    qids = sorted(q for q in base if q.startswith("Q"))
    CACHE.mkdir(parents=True, exist_ok=True)

    batches = [qids[i:i + BATCH] for i in range(0, len(qids), BATCH)]
    if args.test:
        batches = batches[:1]
    print(f"{len(qids):,} cities -> {len(batches):,} batches of {BATCH}", flush=True)

    done = 0
    for n, batch in enumerate(batches, 1):
        out = CACHE / f"{n:05d}.json"
        if out.exists() and not args.test:
            done += 1
            continue
        data = api_get({
            "action": "wbgetentities", "format": "json", "formatversion": "2",
            "ids": "|".join(batch),
            "props": "labels|aliases|sitelinks",
            "sitefilter": "enwiki",
        })
        rows = {}
        for q, ent in (data.get("entities") or {}).items():
            rows[q] = distill(ent, base.get(q, {}).get("name"))
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        done += 1
        if n % 25 == 0 or args.test:
            withwiki = sum(1 for r in rows.values() if r["wiki"])
            print(f"  [{n}/{len(batches)}] {withwiki}/{len(rows)} have enwiki",
                  flush=True)
        if args.test:
            for q, r in list(rows.items())[:4]:
                print(f"    {q:10s} {base.get(q,{}).get('name')!r}")
                print(f"      wiki={r['wiki']!r}")
                print(f"      alt={r['alt'][:6]}")
        time.sleep(GAP)

    print(f"done: {done:,} batches cached in {CACHE}")


if __name__ == "__main__":
    main()
