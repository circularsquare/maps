"""Colours for register lines: a hand-kept table first, then Wikidata for the rest.

    python line_colours.py --fetch jp kr       # query Wikidata once, into data/raw/
    (applied by build_model.py after the register merge; nothing to run per build)

THE TABLES, colours/<region>.csv (line,operator,colour,source,url,note), are the deliberate
choices and win over anything OSM or Wikidata gave the line. `line` is the register line's
name exactly as the build writes it, `operator` an optional fragment of its operator.
jp.csv is the JR companies' own line colours; kr.csv is the colours of the widely
circulated Korea network map (Korail publishes none for its intercity lines), plus a few
picked for lines that map leaves out, marked `picked`. A line coloured from a table carries
`colour_src: "table:<source>"`.

WHY. A register line gets a colour only when an OSM route relation matches it by name, so 70%
of Japan's register lines and most of Korea's were drawn in the per-kind default, which for
heavy rail is one blue: the San'in Line, the Tohoku Line and the Gyeongbu Line all alike.
Wikidata records a line colour (P465) on about 450 Japanese railway lines.

WHAT THE COLOURS ARE. Not all official. Wikidata's line colours are largely Japanese
Wikipedia's infobox colours: many are the operator's own (JR East's Tokaido orange F68B1E),
many are an editor's pick from the CSS named colours (008000 "green" on the Tohoku
Shinkansen). Each line filled here carries `colour_src: "wikidata"` so the difference stays
visible. Korea's intercity lines have no line colours of their own; Wikidata gives most of
them Korail's corporate blue, which is skipped (GENERIC), since it is exactly the one-blue
problem this is here to fix.

MATCHING is by the register's line name, normalised as build_model matches OSM lines
(norm_line_name: 山陰線 = 山陰本線), against Wikidata's native-language label. Where more than
one item has that name (東西線 is Tokyo's, Kyoto's, Sapporo's and Sendai's), only the items
whose operator is the line's are kept, and a line whose candidates still disagree on a
colour is left alone.
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

COUNTRY = {"jp": ("Q17", "ja"), "kr": ("Q884", "ko"), "ch": ("Q39", "de")}

# Colours that name an operator, not a line: taking them paints every line of that operator
# the same, which is the problem this module exists to fix.
GENERIC = {"kr": {"0066B3", "0066BC"}}

# Wikidata items with a colour that are not railways.
NOT_RAIL = {"political party", "defunct political party", "sports team", "association football club"}

QUERY = """
SELECT ?item ?col ?label ?type ?typeLabel ?opLabel ?opNative WHERE {
  ?item wdt:P465 ?col ; wdt:P17 wd:%(q)s .
  ?item rdfs:label ?label FILTER(lang(?label) = "%(lang)s")
  OPTIONAL { ?item wdt:P31 ?type . }
  OPTIONAL { ?item wdt:P137 ?op .
             OPTIONAL { ?op rdfs:label ?opNative FILTER(lang(?opNative) = "%(lang)s") } }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""


def cache(region):
    return RAW / f"wikidata_colours_{region}.json"


def fetch(region):
    q, lang = COUNTRY[region]
    url = ("https://query.wikidata.org/sparql?format=json&query="
           + urllib.parse.quote(QUERY % {"q": q, "lang": lang}))
    req = urllib.request.Request(url, headers={"User-Agent": "noritetsu/0.1 (rail map)"})
    rows = json.load(urllib.request.urlopen(req, timeout=180))["results"]["bindings"]
    cache(region).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{region}: {len(rows)} rows -> {cache(region)}")


def items(region):
    path = cache(region)
    if not path.exists():
        return {}
    out = defaultdict(lambda: {"cols": set(), "ops": set(), "types": set(), "label": ""})
    for r in json.loads(path.read_text(encoding="utf-8")):
        it = out[r["item"]["value"]]
        it["label"] = r["label"]["value"]
        it["cols"].add(r["col"]["value"].strip().lstrip("#").upper())
        if r.get("typeLabel"):
            it["types"].add(r["typeLabel"]["value"])
        for k in ("opNative", "opLabel"):
            if r.get(k):
                it["ops"].add(r[k]["value"])
    return {q: it for q, it in out.items() if not it["types"] & NOT_RAIL}


def table(region):
    import csv
    path = ROOT / "colours" / f"{region}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("colour") or "").strip()]


def apply(region, lines, log):
    """Colour register lines: the table's colour wherever it names the line, then Wikidata's
    on any still without one. Returns how many Wikidata filled."""
    import build_model as bm
    rows = table(region)
    n_table, unused = 0, []
    for r in rows:
        name, op = r["line"].strip(), (r.get("operator") or "").strip()
        hits = [l for l in lines if l.get("src", "osm") != "osm" and l["name"] == name
                and (not op or op in (l.get("operator") or ""))]
        col = r["colour"].strip().upper()
        if re.fullmatch(r"[0-9A-F]{6}", col):
            col = "#" + col          # a bare hex draws nothing in the app (ru.csv, 2026-10-03)
        for l in hits:
            l["colour"] = col
            l["colour_src"] = f"table:{(r.get('source') or '').strip()}"
        n_table += len(hits)
        if not hits:
            unused.append(name)
    if rows:
        log(f"colours: {n_table} register lines coloured from colours/{region}.csv"
            + (f"; {len(unused)} rows matched no line: {' '.join(unused)}" if unused else ""))
    got = items(region)
    if not got:
        log(f"colours: no Wikidata cache for {region} (python line_colours.py --fetch {region})")
        return 0
    by_name = defaultdict(list)
    for it in got.values():
        by_name[bm.norm_line_name(it["label"])].append(it)
    generic = GENERIC.get(region, set())
    filled = ambiguous = 0
    for l in lines:
        if l.get("src", "osm") == "osm" or l.get("colour"):
            continue
        cands = by_name.get(bm.norm_line_name(l["name"], l.get("operator", "")), [])
        if len(cands) > 1:
            own = [it for it in cands
                   if any(bm.same_operator(op, l.get("operator", "")) for op in it["ops"])]
            cands = own or cands
        cols = {c for it in cands for c in it["cols"]} - generic
        if len(cols) == 1:
            l["colour"] = "#" + cols.pop()
            l["colour_src"] = "wikidata"
            filled += 1
        elif len(cols) > 1:
            ambiguous += 1
    reg = sum(1 for l in lines if l.get("src", "osm") != "osm")
    log(f"colours: {filled} register lines coloured from Wikidata, {ambiguous} left alone "
        f"with conflicting colours; {sum(1 for l in lines if l.get('src', 'osm') != 'osm' and l.get('colour'))} "
        f"of {reg} register lines now have one")
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", nargs="+", metavar="REGION")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for region in args.fetch or ():
        fetch(region)


if __name__ == "__main__":
    main()
