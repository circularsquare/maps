"""
Throwaway: build a viewable preview from whatever roster data exists.

This is NOT the real build.py. It exists so there is something on screen while
the floor-10000 refetch runs. It reads the older floor-15000 pool if the new one
is incomplete, and emits compact parallel arrays for the canvas map.

Real build.py will merge base + overrides, apply the settlement-type filter,
attach GHS enrichment, and carry the labelled population set.
"""
import json, pathlib, sys

# City names are full Unicode; the Windows console is cp1252 and will crash the
# script on the first Māori or Vietnamese name otherwise.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
NEW = HERE / "cache" / "wikidata_f10000"
OLD = HERE / "cache" / "wikidata"


def load(d):
    rows = {}
    for p in d.glob("Q*.json"):
        if p.stem == "_countries":
            continue
        rows.update(json.loads(p.read_text(encoding="utf-8")))
    return rows


new_n = len(list(NEW.glob("Q*.json"))) if NEW.exists() else 0
src = NEW if new_n > 300 else OLD
rows = load(src)
print(f"source: {src.name}  ({len(rows):,} rows)")

# Type filter, only possible where types were stored (the new pool).
types_path = HERE / "cache" / "settlement_types.json"
settlement = set(json.loads(types_path.read_text(encoding="utf-8"))) if types_path.exists() else None
typed = [r for r in rows.values() if r.get("types")]
if settlement and typed:
    before = len(rows)
    rows = {q: r for q, r in rows.items()
            if not r.get("types") or any(t in settlement for t in r["types"])}
    print(f"type filter: {before:,} -> {len(rows):,}")
else:
    # Crude stand-in until the typed pool lands: nothing that is actually a city
    # exceeds ~40M (Tokyo's metro is the ceiling), so anything above 45M is a
    # country, a continent or a supranational region. This removes the worst of
    # it — "South Asia", "China", "India" were the five largest "cities" — but
    # it does NOT catch smaller non-settlements like dioceses or municipalities.
    CITY_MAX = 45_000_000
    before = len(rows)
    dropped = sorted((r for r in rows.values() if r["pop"] > CITY_MAX),
                     key=lambda r: -r["pop"])[:6]
    rows = {q: r for q, r in rows.items() if r["pop"] <= CITY_MAX}
    print(f"type filter: SKIPPED (old pool has no types) -- "
          f"crude >45M cut dropped {before-len(rows)}: "
          f"{', '.join(r['name'] for r in dropped)}")

iso = {}
cf = src / "_countries.json"
if cf.exists():
    iso = dict(json.load(open(cf)))

vals = sorted(rows.values(), key=lambda r: -r["pop"])
out = {
    "n": len(vals),
    "name": [r["name"] or "" for r in vals],
    "qid": [r["qid"] for r in vals],
    "lat": [round(r["lat"], 4) for r in vals],
    "lon": [round(r["lon"], 4) for r in vals],
    "pop": [int(r["pop"]) for r in vals],
    "admin": [r.get("admin_name") or "" for r in vals],
    "elev": [None if r.get("elev") is None else round(r["elev"]) for r in vals],
}
p = HERE / "data" / "preview.json"
p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"wrote {p}  ({p.stat().st_size/1048576:.1f} MB, {len(vals):,} cities)")
print(f"largest: {', '.join(v['name'] for v in vals[:5])}")
