"""Russia: published lengths for check_model, from ru.wikipedia's articles on the Wikidata line
items ru_register.py matched to tariff sections (names.json "wikidata").

    python probe_ru_wplengths.py > data/raw/ru/probe_wplengths.txt

Reads each item's ru.wikipedia sitelink (Wikidata API), then the article's wikitext (ru.wikipedia
API, 50 titles a call) and takes the infobox length: `протяжённость`/`длина` in km. Writes
data/raw/ru/wp_lengths.json {section id: {qid, title, km, field}} and prints them beside the
tariff km.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ru"
UA = "noritetsu-rail-map/1.0"
FIELD = re.compile(r"\|\s*(протяж[её]нность|длина)\s*=\s*([^\n|]*)", re.I)


def get(url, params):
    q = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(q, headers={"User-Agent": UA})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:                                   # noqa: BLE001
            print(f"  retry {i}: {e}", file=sys.stderr)
            time.sleep(10 * (i + 1))
    raise SystemExit("request failed")


def km_of(v):
    v = re.sub(r"<ref.*?(</ref>|/>)", "", v or "", flags=re.S)
    v = re.sub(r"\{\{[^}]*\}\}", lambda m: m.group(0) if re.match(r"\{\{\s*км", m.group(0)) else "", v)
    m = re.search(r"(\d[\d\s ]*(?:[.,]\d+)?)", v)
    if not m:
        return None
    x = float(m.group(1).replace(" ", "").replace(" ", "").replace(",", "."))
    return x / 1000 if re.search(r"\d\s*м\b", v) and "км" not in v else x


def main():
    names = json.loads((ROOT / "data" / "raw" / "rinf" / "ru" / "names.json").read_text("utf-8"))
    by_q = {}
    for sid, e in names.items():
        if e.get("wikidata"):
            by_q.setdefault(e["wikidata"], []).append(sid)
    qs = sorted(by_q)
    title = {}
    for i in range(0, len(qs), 200):
        vals = " ".join(f"wd:{q}" for q in qs[i:i + 200])
        d = get("https://query.wikidata.org/sparql", {"format": "json", "query": (
            f"SELECT ?x ?t WHERE {{ VALUES ?x {{ {vals} }} ?a schema:about ?x ; "
            f"schema:isPartOf <https://ru.wikipedia.org/> ; schema:name ?t }}")})
        for b in d["results"]["bindings"]:
            title[b["x"]["value"].rsplit("/", 1)[-1]] = b["t"]["value"]
        time.sleep(2)
    print(f"{len(qs)} Wikidata items, {len(title)} with a ru.wikipedia article")
    # Most of those items have no article, so also every article titled "Железнодорожная
    # линия/ветка A — B", matched to a section by its two ends.
    sys.path.insert(0, str(ROOT))
    from ru_register import base_name
    ends = {}
    for sid, e in names.items():
        a, b = re.split(r"\s+—\s+", re.sub(r"\s*\(через.*$", "", e["name"]), maxsplit=1)
        ends.setdefault(frozenset((base_name(a), base_name(b))), []).append(sid)
    for prefix in ("Железнодорожная линия ", "Железнодорожная ветка "):
        cont = {}
        while True:
            d = get("https://ru.wikipedia.org/w/api.php", {
                "action": "query", "list": "allpages", "apprefix": prefix, "apnamespace": 0,
                "aplimit": 500, "apfilterredir": "nonredirects", "format": "json", **cont})
            for p in d["query"]["allpages"]:
                t = p["title"]
                parts = re.split(r"\s+[—–-]\s+", t[len(prefix):])
                if len(parts) == 2:
                    k = frozenset((base_name(parts[0]), base_name(parts[1])))
                    for sid in ends.get(k, ()):
                        if sid not in by_q.get(names[sid].get("wikidata"), ()) or \
                                names[sid].get("wikidata") not in title:
                            title.setdefault(f"page:{sid}", t)
                            by_q.setdefault(f"page:{sid}", []).append(sid)
            if "continue" not in d:
                break
            cont = d["continue"]
            time.sleep(1)
    print(f"  {sum(1 for q in title if q.startswith('page:'))} more sections matched by "
          f"article title")
    text = {}
    ts = sorted(set(title.values()))
    for i in range(0, len(ts), 50):
        d = get("https://ru.wikipedia.org/w/api.php", {
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "titles": "|".join(ts[i:i + 50]), "redirects": 1, "format": "json",
            "formatversion": 2})
        redir = {r["from"]: r["to"] for r in d["query"].get("redirects", [])}
        for p in d["query"]["pages"]:
            if "revisions" in p:
                text[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
        for a, b in redir.items():
            if b in text:
                text[a] = text[b]
        time.sleep(1)
    out = {}
    for q, t in title.items():
        m = FIELD.search(text.get(t, ""))
        km = km_of(m.group(2)) if m else None
        if km:
            for sid in by_q[q]:
                out[sid] = {"qid": q, "title": t, "km": km, "field": m.group(1)}
    (RAW / "wp_lengths.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"{len(out)} sections with an infobox length")
    for sid, e in sorted(out.items(), key=lambda kv: -names[kv[0]]["tariff_km"]):
        tk = names[sid]["tariff_km"]
        print(f"  {sid} tariff {tk:5d}  wp {e['km']:7.1f}  {e['km'] / tk if tk else 0:5.2f}  "
              f"{names[sid]['name']}  [{e['title']}]")


if __name__ == "__main__":
    main()
