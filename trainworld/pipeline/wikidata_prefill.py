"""T-3244: fill the Wikidata caches a light pass reads, 200 ids a query, before the pass, so its
workers find every answer on disk instead of asking one pack at a time (429s at 10 workers).

    python pipeline/wikidata_prefill.py [--into DIR] [--no-airports]

- University and college ids of every OSM education cache in data/raw/tier0, and of every staged
  campus answer (campus_outlines.py): their student counts (tier0.wikidata_students), then the
  parents of the ids with none (tier0.wikidata_parents), as tier0.education_osm asks.
- The IATA codes of every pack's airport audit (data/work/<id>.airports.audit.json):
  enrich_airports.fill_wikidata_boardings.
Ids already cached are not asked again, so a rerun after more campus answers costs only the new ones.
`--into DIR`: read and write the three cache files in DIR instead (copied there from the real ones
first if absent), for a proof.
"""
import glob
import json
import os
import shutil
import sys
import time

import pandas as pd

import build_city as b
import campus_outlines
import enrich_airports as air
import tier0

CHUNK = 2000  # ids a call, so a failure keeps what came before (each call writes the cache once)


def campus_ids():
    ids = set()
    for p in glob.glob(tier0.raw("osm_education_*.csv.gz")):
        ed = pd.read_csv(p, keep_default_na=False, usecols=["kind", "wikidata"])
        ids |= set(ed.wikidata[ed.kind.isin(["university", "college"]) & (ed.wikidata != "")])
    staged = 0
    import overpass
    for p in glob.glob(os.path.join(campus_outlines.TIER0, "campus", "*.json.gz")):
        # every tagged id of the answer: a few more than campus_outlines.rows keeps, which costs nothing
        data = overpass.read_cache(campus_outlines.name(os.path.basename(p)[:-8]))
        ids |= {e["tags"]["wikidata"] for e in data.get("elements", []) if e.get("tags", {}).get("wikidata")}
        staged += 1
    return sorted(ids), staged


def airport_codes():
    codes = set()
    for p in glob.glob(os.path.join(b.WORK, "*.airports.audit.json")):
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, list):  # tier 0's; the US hand packs' (enrich_airports.main) are a dict and use FAA figures
            codes |= {a.get("iata") for a in rows if a.get("iata")}
    return sorted(codes)


def main(args):
    if "--into" in args:
        d = os.path.abspath(args[args.index("--into") + 1])
        os.makedirs(d, exist_ok=True)
        for src in (tier0.raw("wikidata_students.json"), tier0.raw("wikidata_parents.json"),
                    os.path.join(b.WORK, air.BOARDINGS_CACHE)):
            if os.path.exists(src) and not os.path.exists(os.path.join(d, os.path.basename(src))):
                shutil.copyfile(src, os.path.join(d, os.path.basename(src)))
        real_raw = tier0.raw
        tier0.raw = lambda *p: os.path.join(d, *p) if p and p[0].startswith("wikidata_") else real_raw(*p)
        air._cache_path = lambda: os.path.join(d, air.BOARDINGS_CACHE)
    t0 = time.time()
    ids, staged = campus_ids()
    print(f"{len(ids):,} university and college ids ({staged} staged campus answers), {time.time() - t0:.0f} s to list", flush=True)
    counts = {}
    for i in range(0, len(ids), CHUNK):
        counts.update(tier0.wikidata_students(ids[i:i + CHUNK]))
        print(f"  students: {min(i + CHUNK, len(ids)):,}/{len(ids):,}, {time.time() - t0:.0f} s", flush=True)
    bare = sorted(q for q in ids if not counts.get(q, (None,))[0])
    for i in range(0, len(bare), CHUNK):
        tier0.wikidata_parents(bare[i:i + CHUNK])
        print(f"  parents: {min(i + CHUNK, len(bare)):,}/{len(bare):,}, {time.time() - t0:.0f} s", flush=True)
    if "--no-airports" not in args:
        codes = airport_codes()
        got = air.fill_wikidata_boardings(codes)
        print(f"  airports: {len(codes):,} codes, {sum(1 for v in got.values() if v[0]):,} with a figure, {time.time() - t0:.0f} s",
              flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
