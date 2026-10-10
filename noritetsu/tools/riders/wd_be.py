"""Belgium: SNCB/NMBS October station counts as carried on Wikidata (P1373 daily patronage).

SNCB counts boardings ("instappers") at every station once a year in October and publishes
the average weekday, Saturday and Sunday. Wikidata holds all three as P1373 statements, told
apart by the qualifier P2894 "day of week" (Q19906285 weekday, Q131 Saturday, Q132 Sunday)
and dated by P585 (October 2019 and October 2022 for most stations; checked on Bruxelles-
Central, Q800588: 2022 weekday 49,476, Saturday 24,143, Sunday 20,618).

The figure used is the latest year with all three day types, as an average over the week:
(5 x weekday + Saturday + Sunday) / 7. A station with only a weekday figure for its latest
year is left out rather than mixing meanings (none found when this was written). Those are
boardings only; to keep one meaning with the other countries (getting on + off) the average
is doubled, on SNCB's own working assumption that over a day as many get off at a station as
get on. Both steps are recorded here and in riders_sources.json. Matched by name (French,
Dutch, English labels) within RADIUS_KM of the Wikidata item's coordinates.
"""
from collections import defaultdict

from . import wd

KEY = "wd_be"
CC = "be"
FOLDER = "wikidata_be"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "SNCB October counts, through Wikidata",
    "name": "SNCB/NMBS October counts (boardings per station: weekday, Saturday, Sunday), "
            "via Wikidata P1373",
    "url": "https://www.wikidata.org/wiki/Property:P1373",
    "licence": "Wikidata CC0; underlying figures published by SNCB/NMBS",
    "counts": "boardings, averaged over the week as (5 x weekday + Saturday + Sunday) / 7, "
              "doubled for getting on + off",
    "note": "an October count, not a year's average",
}
DAY = {"Q19906285": "wk", "Q131": "sat", "Q132": "sun"}


def records(raw):
    rows = wd.sparql("""SELECT ?item ?label ?label_nl ?label_en ?coord ?n ?date ?dow WHERE {
      ?item wdt:P17 wd:Q31 ; wdt:P625 ?coord ; p:P1373 ?st .
      ?st ps:P1373 ?n .
      OPTIONAL { ?st pq:P585 ?date }
      OPTIONAL { ?st pq:P2894 ?dow }
      OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "fr") }
      OPTIONAL { ?item rdfs:label ?label_nl FILTER(LANG(?label_nl) = "nl") }
      OPTIONAL { ?item rdfs:label ?label_en FILTER(LANG(?label_en) = "en") }
    }""", raw / "wd_p1373_dow.json")
    items, counts = {}, defaultdict(dict)       # qid -> {year: {wk, sat, sun}}
    for r in rows:
        try:
            n = float(r["n"])
        except (TypeError, ValueError):
            continue
        day = DAY.get(wd.qid(r.get("dow")))
        if not day or not r.get("date"):
            continue
        q = wd.qid(r["item"])
        counts[q].setdefault(int(r["date"][:4]), {})[day] = n
        if q not in items:
            items[q] = r
    out = []
    for q, years in counts.items():
        full = [y for y, d in years.items() if len(d) == 3]
        if not full:
            continue
        y = max(full)
        d = years[y]
        if y < max(years):
            continue                # the latest year lacks a day type: left out, see above
        r = items[q]
        xy = wd.point(r.get("coord"))
        if not xy:
            continue
        week = (5 * d["wk"] + d["sat"] + d["sun"]) / 7
        out.append({"name": r.get("label") or r.get("label_nl") or q,
                    "alt": [r[k] for k in ("label_nl", "label_en") if r.get(k)],
                    "qid": q, "x": xy[0], "y": xy[1], "n": 2 * week, "year": y})
    return out
