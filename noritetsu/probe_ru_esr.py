"""Russia stage 1: which OSM objects carry an ESR code (esr:user), read straight from the .pbf.

    python probe_ru_esr.py --pbf data/raw/russia-YYMMDD.osm.pbf

extract.py does not keep `esr:user` (or `esr`, `esr:wikidata`...) on stops, so this pass reads
them once, before the .pbf is deleted, and writes data/raw/ru/osm_esr.json:
    {esr: [[type, id, lon, lat, name, railway, public_transport, train], ...]}
Nodes only: Russian OSM puts the code on the station node (and on platforms, which are ways;
those are skipped). Also counts which keys beginning with "esr" are in use.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
import osmium

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    args = ap.parse_args()
    pbf = Path(args.pbf)
    if not pbf.is_absolute():
        pbf = ROOT / pbf
    keys = Counter()
    out = defaultdict(list)
    n = 0
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("esr:user", "esr", "esr:railway", "railway:esr")))
    for obj in fp:
        n += 1
        t = obj.tags
        for k, _v in t:
            if k.startswith("esr") or k.endswith(":esr"):
                keys[k] += 1
        code = t.get("esr:user") or t.get("esr") or t.get("railway:esr")
        if not code:
            continue
        for c in code.replace(",", ";").split(";"):
            c = c.strip()
            if c.isdigit() and len(c) == 6:
                out[c].append(["n", obj.id, round(obj.location.lon, 6), round(obj.location.lat, 6),
                               t.get("name"), t.get("railway"), t.get("public_transport"),
                               t.get("train")])
    path = ROOT / "data" / "raw" / "ru" / "osm_esr.json"
    path.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"{n} nodes with an esr key; {len(out)} distinct six-digit codes; keys {keys.most_common()}")
    kinds = Counter(r[5] or ("pt=" + str(r[6])) for v in out.values() for r in v)
    print("  by railway tag:", kinds.most_common(12))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
