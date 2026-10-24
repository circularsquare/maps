"""T-441: register a batch of built packs in one step, so no agent reads registry.json.

    python pipeline/register.py <id> [<id> ...] [--market] [--trips]

For each id whose pack is built (`data/packs/<id>.json`) and that the registry lacks, appends its
`cities` entry from `build_city.CITIES`: name (`name`, else `region_name`), `kind` "zone" for a
non-city zone, origin and center (the config's origin), zoom (11.2 a city, 10.5 a zone),
`utcOffset` (`utc_offset`, else the longitude's whole hour) and `market` -1 (no pack has a market
id since T-1014). Then each named id's `box`
(zone_boxes.py). `--market` then runs market_zones.py for the ids (its market pack and every row);
`--trips` runs zones.py's local trips for every zone. Prints one line per id.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_city import CITIES  # noqa: E402
import zone_boxes  # noqa: E402
import shared_lock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "app/src/game/registry.json"


def entry(cid):
    cfg = CITIES[cid]
    lon, lat = cfg["origin"]
    zone = cfg.get("kind") == "zone"
    e = {"id": cid, "name": cfg.get("name") or cfg["region_name"]}
    if zone:
        e["kind"] = "zone"
    e.update({"origin": {"lon": lon, "lat": lat}, "center": [lon, lat], "zoom": 10.5 if zone else 11.2,
              "utcOffset": cfg.get("utc_offset", round(lon / 15)), "market": -1})
    return e


def main(argv):
    # the whole read-modify-write, market and trips included: two runners register at once (T-786)
    with shared_lock.held("registry", "register.py " + " ".join(argv)[:200]):
        _main(argv)


def _main(argv):
    ids = [a for a in argv if not a.startswith("--")]
    reg = shared_lock.read_json(REG)
    have = {c["id"] for c in reg["cities"]}
    for cid in ids:
        if cid not in CITIES:
            sys.exit(f"{cid}: not in build_city.CITIES")
        if not (ROOT / f"data/packs/{cid}.json").exists() or not (ROOT / f"data/packs/{cid}.boundary.geojson").exists():
            sys.exit(f"{cid}: pack or boundary not built")
    for cid in ids:
        if cid not in have:
            reg["cities"].append(entry(cid))
        c = next(c for c in reg["cities"] if c["id"] == cid)
        c["box"] = zone_boxes.box(cid)
        print(f"{cid}: {c['name']}, {'new' if cid not in have else 'already registered'}, box {c['box']}", flush=True)
    shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    if "--market" in argv:
        import market_zones
        market_zones.main(ids)
    if "--trips" in argv:
        import zones
        reg = shared_lock.read_json(REG)
        trips = zones.local_trips(reg)
        shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        for cid in ids:
            print(f"{cid}: localTrips {trips.get(cid, 0):,.0f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
