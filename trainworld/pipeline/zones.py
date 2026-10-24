"""T-276: the registry's zones (docs/zones.md).

    python pipeline/zones.py            register built non-city zones, then every zone's local trips
    python pipeline/zones.py --trips    only the local trips

1. Every `build_city.CITIES` entry with `"kind": "zone"` whose pack is built and that the registry
   lacks is appended to `app/src/game/registry.json`: a market place (ids are positions: append
   only) and a `cities` entry with `"kind": "zone"`.
2. Each registered zone's local trip legs a day with no rail (`localTrips`: commutes kept in the
   pack and every everyday purpose, both legs; the sim crate's `zone_summary`), which sizes the
   far-zoomed bubble of a zone the game has not opened.

Run after building a zone pack and after pipeline/intercity.py (the joint gravity changes what each
pack keeps), then pipeline/market_zones.py (T-377).
"""
import json
import sys
from pathlib import Path

from build_city import CITIES
import simbin
import zone_boxes

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "app/src/game/registry.json"


def register(reg):
    have = {c["id"] for c in reg["cities"]}
    added = []
    for cid, cfg in CITIES.items():
        # tier 0 zones (Japan's): pipeline/register.py adds them
        if cfg.get("kind") != "zone" or cfg.get("tier0") or cid in have or not (ROOT / f"data/packs/{cid}.json").exists():
            continue
        lon, lat = cfg["origin"]
        reg["market"]["places"].append({"name": cfg["name"], "regionName": cfg["name"]})
        reg["cities"].append({"id": cid, "name": cfg["name"], "kind": "zone", "origin": {"lon": lon, "lat": lat},
                              "center": [lon, lat], "zoom": 10.5,
                              # T-309: whole-hour standard offset; the longitude's if the config has none
                              "utcOffset": cfg.get("utc_offset", round(lon / 15)),
                              "market": -1, "place": len(reg["market"]["places"]) - 1})
        added.append(cid)
    return added


def local_trips(reg):
    ids = [c["id"] for c in reg["cities"] if (ROOT / f"data/packs/{c['id']}.json").exists()]
    # names relative to data/packs: 590 absolute paths passed Windows' 32k command line (T-788)
    out = simbin.run(ROOT, "zone_summary", [f"{i}.json" for i in ids], capture_output=True, text=True,
                     cwd=str(ROOT / "data/packs"))
    trips = json.loads(out.stdout)
    for c in reg["cities"]:
        if c["id"] in trips:
            c["localTrips"] = trips[c["id"]]
    return trips


if __name__ == "__main__":
    import shared_lock
    with shared_lock.held("registry", "zones.py"):  # T-786
        reg = shared_lock.read_json(REG)
        if "--trips" not in sys.argv[1:]:
            print("registered:", register(reg) or "none")
        print("local trips:", local_trips(reg))
        for c in reg["cities"]:
            c["box"] = zone_boxes.box(c["id"])  # T-302
        shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
