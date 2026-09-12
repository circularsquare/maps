"""Switzerland — BFS, Volkszählung 2000, religion by commune, on 2021 commune boundaries.

Reads (or fetches) data/raw/ch/ and writes data/normalized/ch.csv.

**This module is a normaliser and nothing else.** It reads the 2000 census, moves it onto the
commune boundaries the map draws, and stops. The interpolation that makes Switzerland a
current map — 2021 canton magnitudes on 2000 commune structure, spec §3.4 — is `ch_rescale.py`,
kept separate for the reason `br_rescale.py` is: a normaliser and an interpolation are
different things and folding them together hides which is which.

## The table

`px-x-4003000000_122` — *Wohnbevölkerung 2000 nach Wohnsitztyp, Kanton / Bezirk / Gemeinde und
Religion* — **3,107 geographic rows (1 country, 26 cantons, 184 districts, 2,896 communes) x
19 religion categories**, 289 KB over one POST, no key and no wall.

**It is the last Swiss census that asked everybody**, and its category list is the deepest
Protestant breakdown on this map: Reformed, Methodist, *Neupietistisch-evangelikale
Gemeinden*, Pentecostal, New Apostolic, Jehovah's Witnesses and "other Protestant" as seven
separate cells, plus Christ Catholic (the Old Catholic church), Orthodox, Jewish, Islamic,
Buddhist and Hindu. Nothing since 2010 asks at any geography — the Strukturerhebung is a
sample and stops at canton — so this is the only fine-grained religion Switzerland has.

**FINDING IT WAS A CATALOGUE PROBLEM, NOT A CRAWL.** sources.md §11c priced this country at
"an hour of wall clock" to enumerate all 650 BFS PxWeb databases, because their ids are
opaque and the human title lives only inside the per-database listing. That crawl does not
work: BFS rate-limits it into total failure — 55 of 55 requests returned nothing at one
request per second. **`ckan.opendata.swiss` answers the same question in one call**, because
it is the national portal that mirrors BFS's own catalogue, and its search returns the px
table id directly. It also carries Liechtenstein's statistics office, which is how that
country turned up.

## The commune vintage, which is the real work

The census counts **2,896 communes**; the boundary file this project already has — GISCO LAU
2021, on disk since Poland (§9e) — carries **2,242**. Swiss communes have been merging
continuously, and a naive join on the BFS number matches only 2,076 of them, losing 820
communes and 606,086 people.

**BFS publishes the correspondence itself**, and it is an open keyless API rather than a file
to parse:

    https://www.agvchapp.bfs.admin.ch/api/communes/correspondances
        ?includeUnmodified=true&includeTerritoryExchange=false
        &startPeriod=06-12-2000&endPeriod=01-01-2020

`InitialCode` -> `TerminalCode`, one row per commune, over any period. Three things about
using it:

- **`startPeriod` is the census date, 6 December 2000, not 1 January 2001.** A year boundary
  loses the mutations of 2001 itself: 34 communes and 21,022 people, all of them Fribourg
  villages that merged into Villorsonnens.
- **`includeTerritoryExchange=false`.** With exchanges included a commune can map to more
  than one successor and the join stops being a function. Exchanges are boundary adjustments
  of a few hectares; ignoring them is the same order of approximation as §8.2.
- **GISCO's "LAU 2021" for Switzerland is the 1 January 2020 commune state, not 2021.** Asking
  the API for 2021 leaves 15 communes pointing at codes the shapefile does not have —
  Welschenrohr-Gänsbrunnen and Bois-d'Amont among them, both created on 1 January 2021. Asking
  for 2020 leaves 5, the Verzasca valley communes that merged in 2020. **Both are requested
  and the one whose target exists in the boundary file wins**, which resolves all 2,896.
  *A boundary file named for a year is not necessarily that year's state; test it rather than
  trusting the filename.*

## Two more things worth knowing

**`Wohnsitztyp` has two values and they are different populations.** *Zivilrechtlicher
Wohnsitz* (civil-law residence, where you are registered) and *Wirtschaftlicher Wohnsitz*
(economic residence, where you actually live) differ for students and weekly commuters. The
civil-law figure is used, because it is the one the 7,287,357 national total refers to and the
one every published Swiss 2000 table quotes.

**The canton of each commune comes from BFS, not from a name.** `agvchapp`'s `levels` endpoint
returns `BfsCode -> CantonId, Canton` for every commune at a given date, so `ch_rescale.py`
can attach cantons without matching German names against the French ones the Strukturerhebung
publishes.

Usage:
    python sources/ch.py --fetch    one POST + three GETs, ~600 KB
    python sources/ch.py           normalise from data/raw/ch/
"""

import csv
import io
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ch")
OUT = os.path.join(ROOT, "data", "normalized", "ch.csv")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")

SOURCE_ID = "ch_volkszaehlung_2000"
YEAR = 2000
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

PX = ("https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-4003000000_122/"
      "px-x-4003000000_122.px")
CUBE = os.path.join(RAW, "px-x-4003000000_122.json")

AGV = "https://www.agvchapp.bfs.admin.ch/api/communes/"
# Both boundary states are requested; see the docstring. 2020 is GISCO's actual vintage and
# 2021 catches the five Verzasca communes it merged before that snapshot.
CORR_DATES = ("01-01-2020", "01-01-2021")
CORR = {d: os.path.join(RAW, f"correspondances_{d}.csv") for d in CORR_DATES}
LEVELS_DATE = "01-01-2020"
LEVELS = os.path.join(RAW, "levels.csv")

# Strukturerhebung religion by canton, 2010-2024, one sheet per year. Read by ch_rescale.py,
# fetched here so one --fetch gets everything the country needs.
SE_URL = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/36347560/master"
SE = os.path.join(RAW, "je-01.08.02.02-canton.xlsx")

GEO_DIM = "Kanton (-) / Bezirk (>>) / Gemeinde (......)"
CENSUS_DATE = "06-12-2000"          # the 2000 census reference date
WOHNSITZ = "Zivilrechtlicher Wohnsitz"

TOTAL_CAT = "Religionen - Total"
EXPECTED_COMMUNES_2000 = 2_896
EXPECTED_CANTONS = 26
NATIONAL = 7_287_357

# A commune row in the px table reads `......0001 Aeugst am Albis`.
COMMUNE_RE = re.compile(r"^\.{6}(\d{4})\s+(.*)$")


def _get(url, dest, binary=False, note=""):
    import requests

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000:
        print("  already have", os.path.basename(dest))
        return
    print("  GET", note or url)
    r = requests.get(url, timeout=600,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    print(f"       {os.path.getsize(dest):,} bytes")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)

    if not os.path.exists(CUBE) or os.path.getsize(CUBE) < 50_000:
        print("  POST", PX)
        # AN EMPTY QUERY DOES NOT MEAN "EVERYTHING" ON THIS SERVER. `{"query": []}` returns
        # a 1.6 KB cube with the geography dimension silently ABSENT — a 200, valid
        # json-stat2, and no geography at all. The selection has to be explicit.
        q = {"query": [
            {"code": "Wohnsitztyp",
             "selection": {"filter": "item", "values": ["0"]}},
            {"code": GEO_DIM, "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Religion", "selection": {"filter": "all", "values": ["*"]}}],
            "response": {"format": "json-stat2"}}
        r = requests.post(PX, json=q, timeout=900,
                          headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"})
        r.raise_for_status()
        doc = r.json()
        for key in ("id", "size", "dimension", "value"):
            if key not in doc:
                raise SystemExit(f"not a json-stat2 dataset, keys {list(doc)}")
        if GEO_DIM not in doc["id"]:
            raise SystemExit("the geography dimension is missing from the response — the "
                             "selection was not applied (see the comment above)")
        with open(CUBE, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"       {os.path.getsize(CUBE):,} bytes")
    else:
        print("  already have", os.path.basename(CUBE))

    for date, dest in CORR.items():
        _get(AGV + "correspondances?includeUnmodified=true&includeTerritoryExchange=false"
                   f"&startPeriod={CENSUS_DATE}&endPeriod={date}",
             dest, note=f"commune correspondence {CENSUS_DATE} -> {date}")
    _get(AGV + f"levels?date={LEVELS_DATE}", LEVELS, note="commune -> canton")
    _get(SE_URL, SE, note="Strukturerhebung religion by canton, 2010-2024")


def _cube():
    if not os.path.exists(CUBE):
        raise SystemExit(f"missing {CUBE} -- run with --fetch first")
    with open(CUBE, encoding="utf-8") as fh:
        return json.load(fh)


def _px_rows():
    """[(level, bfs, name, canton_name, {category: count})] straight off the cube.

    The geography dimension is a FLAT list carrying four levels at once, distinguished only
    by a prefix in the label — `Schweiz`, `- Kanton Zürich`, `>> Bezirk Affoltern`,
    `......0001 Aeugst am Albis`. It is also in hierarchical ORDER, which is the only thing
    that says which canton a commune is in, so the canton is carried down while walking.
    (`ch_rescale.py` does not rely on this: it uses BFS's own levels table. This is here so
    ch.csv's canton rows are attributable without a second file.)
    """
    doc = _cube()
    gcat = doc["dimension"][GEO_DIM]["category"]
    glab, gidx = gcat["label"], list(gcat["index"])
    rcat = doc["dimension"]["Religion"]["category"]
    rlab, ridx = rcat["label"], list(rcat["index"])
    val, nrel = doc["value"], len(ridx)

    def cell(i, j):
        k = i * nrel + j
        v = val[k] if isinstance(val, list) else val.get(str(k))
        return 0 if v is None else int(v)

    out, canton = [], None
    for i, g in enumerate(gidx):
        text = glab[g]
        counts = {rlab[r]: cell(i, j) for j, r in enumerate(ridx)}
        m = COMMUNE_RE.match(text)
        if m:
            out.append(("commune", m.group(1), m.group(2).strip(), canton, counts))
        elif text.startswith(">>"):
            continue                                   # districts are not drawn
        elif text.startswith("- "):
            canton = text[2:].strip()
            out.append(("canton", None, canton, canton, counts))
        else:
            out.append(("country", None, text.strip(), None, counts))
    return out, [rlab[r] for r in ridx]


def _correspondence():
    """{2000 BFS code: 2020/2021 BFS code}, resolved against the boundary file.

    Two states are read and the one whose target actually exists as a polygon wins. See the
    docstring: GISCO's "LAU 2021" is the 1.1.2020 commune state, so neither request alone
    resolves every commune.
    """
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(LAU):
        raise SystemExit(f"missing {LAU} -- the GISCO LAU 2021 file (§9e). "
                         "sources/pl_geo.py --fetch downloads it.")
    lau = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID"])
    lau = lau[lau["CNTR_CODE"] == "CH"]
    have = {int(str(x).replace("CH", "")) for x in lau["LAU_ID"]}

    maps = {}
    for date, path in CORR.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        df = pd.read_csv(path)
        maps[date] = dict(zip(df["InitialCode"], df["TerminalCode"]))

    resolved, stats = {}, {d: 0 for d in CORR_DATES}
    for code in set().union(*(m.keys() for m in maps.values())):
        for date in CORR_DATES:
            t = maps[date].get(code)
            if t is not None and int(t) in have:
                resolved[int(code)] = int(t)
                stats[date] += 1
                break
    return resolved, have, stats


def read():
    rows_px, categories = _px_rows()
    resolved, have, stats = _correspondence()

    communes = [r for r in rows_px if r[0] == "commune"]
    if len(communes) != EXPECTED_COMMUNES_2000:
        raise SystemExit(f"{len(communes)} communes in the cube, expected "
                         f"{EXPECTED_COMMUNES_2000}")

    merged, unmatched = {}, []
    for _, bfs, name, canton, counts in communes:
        target = resolved.get(int(bfs))
        if target is None:
            unmatched.append((bfs, name, counts[TOTAL_CAT]))
            continue
        slot = merged.setdefault(target, {"canton": canton, "names": [],
                                          "counts": {c: 0 for c in categories}})
        slot["names"].append(name)
        for c in categories:
            slot["counts"][c] += counts[c]

    rows = []
    for target, slot in sorted(merged.items()):
        note = "level=commune; 2000 counts on the 2021 boundary"
        if len(slot["names"]) > 1:
            note += (f"; {len(slot['names'])} communes of 2000 merged into this one since: "
                     + ", ".join(sorted(slot["names"])))
        for cat in categories:
            rows.append({"geo_id": f"{target:04d}", "geo_level": "commune",
                         "geo_name": sorted(slot["names"])[0],
                         "source_category": cat, "count": slot["counts"][cat],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})

    for level, _, name, _, counts in rows_px:
        if level == "commune":
            continue
        for cat in categories:
            rows.append({"geo_id": name, "geo_level": level, "geo_name": name,
                         "source_category": cat, "count": counts[cat], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID,
                         "note": f"level={level}; 2000 boundaries, carried for reconciliation"})

    stats = {"resolved_by": stats, "unmatched": unmatched, "targets": len(merged),
             "polygons": len(have), "categories": categories,
             "merged_units": sum(1 for s in merged.values() if len(s["names"]) > 1)}
    return rows, stats


def check(rows, stats):
    ok = True
    cats = stats["categories"]
    print(f"  {len(cats)} religion categories, {stats['targets']:,} communes on the 2021 "
          f"boundary (from {EXPECTED_COMMUNES_2000:,} of 2000)")
    print(f"  {stats['merged_units']:,} of them absorbed at least one other commune since "
          "2000")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL_CAT]:,} "
          f"(published {NATIONAL:,})")

    parts = sum(v for k, v in nat.items() if k != TOTAL_CAT)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 18 categories partition the country exactly "
          f"({parts:,})")

    cantons = {r["geo_id"] for r in rows if r["geo_level"] == "canton"}
    good = len(cantons) == EXPECTED_CANTONS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(cantons)} cantons (expected "
          f"{EXPECTED_CANTONS})")

    # ---- the correspondence, which is where this country's risk is ---------------------
    print(f"\n  THE COMMUNE JOIN. GISCO LAU carries {stats['polygons']:,} Swiss communes; "
          "the census counted\n  2,896. BFS's own correspondence resolves them:")
    for date, n in stats["resolved_by"].items():
        print(f"    {n:>5,} resolved against the {date} commune state")
    lost = sum(t for _, _, t in stats["unmatched"])
    print(f"    {len(stats['unmatched']):>5} not resolved, holding {lost:,} people "
          f"({100.0 * lost / NATIONAL:.3f}%)")
    for bfs, name, tot in stats["unmatched"][:10]:
        print(f"          {bfs} {name} ({tot:,})")
    good = len(stats["unmatched"]) == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every 2000 commune lands on a drawn polygon")

    # ---- per category, at both levels, which is §3.8's shape --------------------------
    drawn = {}
    for r in rows:
        if r["geo_level"] == "commune":
            drawn[r["source_category"]] = drawn.get(r["source_category"], 0) + r["count"]
    print("\n  every category's communes must sum to its national figure — this is what "
          "proves\n  the merge did not drop or double-count anybody:")
    worst = 0
    for cat in sorted(cats, key=lambda c: -nat[c]):
        gap = nat[cat] - drawn.get(cat, 0)
        worst = max(worst, abs(gap))
        if gap:
            ok = False
        print(f"    {'OK ' if gap == 0 else 'BAD'} {cat[:44]:<46} {drawn.get(cat, 0):>10,}"
              f"  {100.0 * nat[cat] / NATIONAL:6.2f}%  gap {gap}")
    if worst:
        print("    BAD a non-zero gap means the correspondence lost or duplicated people.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats = read()
    check(rows, stats)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")
    print("NOTE: this is the 2000 census as counted. `python ch_rescale.py` is what makes "
          "Switzerland\n      a current map (spec §3.4); countries.py reads its output, not "
          "this file.")


if __name__ == "__main__":
    main()
