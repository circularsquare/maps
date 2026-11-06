"""The files a production build copies for the registry (app/vite.config.ts `writeBundle`), checked without a build:
each registered pack's five files, every zone's terrain grid, the world water mask, the outlines, the real networks
and their graphs, and the market's routed drives holding a place for every zone (a zone without one takes the drive
rule; the build only warns when the file is missing). One FLAG line per gap, exit 1 if any.

    venv\\Scripts\\python.exe -I trainworld\\checks\\build_inputs.py [--list]     (--list: every id, not the first 20)
"""
import collections
import json
import struct
import sys
from pathlib import Path

TW = Path(__file__).resolve().parents[1]
PACKS, TERRAIN = TW / "data" / "packs", TW / "data" / "terrain"
EXTS = (".json", ".bin", ".boundary.geojson", ".places.json", ".market.json")


def drive_places(path):
    b = path.read_bytes()
    if b[:4] != b"TWD1":
        return None
    n, o, ids = struct.unpack_from("<I", b, 4)[0], 8, set()
    for _ in range(n):
        ids.add(b[o + 1:o + 1 + b[o]].decode())
        o += 1 + b[o]
    return ids


def main():
    full = "--list" in sys.argv
    reg = json.loads((TW / "app/src/game/registry.json").read_text(encoding="utf-8"))
    cities = reg["cities"]
    flags = []

    def flag(what, rows, fix):
        if rows:
            ids = [c["id"] for c in rows]
            by = dict(collections.Counter(c.get("country", "?") for c in rows))
            flags.append(f"FLAG {len(rows)} {what} {by}: {' '.join(ids if full else ids[:20])}"
                         f"{'' if full or len(ids) <= 20 else ' ...'}; {fix}")

    for ext in EXTS:
        flag(f"packs without {ext}", [c for c in cities if not (PACKS / (c["id"] + ext)).exists()],
             "the runner's register step or build_city/places/market_zones")
    flag("zones without a terrain file", [c for c in cities if not (TERRAIN / f"{c['id']}.terrain").exists()],
         "pipeline/zone_terrain.py build (terrain_fine.py first where the intermediate has no blocks)")
    for f, fix in ((TERRAIN / "world.water", "pipeline/world_water.py build"), (PACKS / "outlines.json", "pipeline/zone_outlines.py"),
                   (PACKS / "real", "pipeline/realnet.py ship"), (PACKS / "realgraph", "pipeline/realgraph.py <regions>")):
        if not f.exists():
            flags.append(f"FLAG {f.relative_to(TW)} missing: {fix}")
    md = PACKS / "market-drives.bin"
    if not md.exists():
        flags.append("FLAG data/packs/market-drives.bin missing: pipeline/market_drives.py build")
    else:
        have = drive_places(md) or set()
        flag("zones with no place in the routed drives", [c for c in cities if c["id"] not in have],
             "pipeline/market_drives.py build (routes the new places only)")
    for line in flags:
        print(line)
    print(f"{len(cities)} registry rows; {len(flags)} flags")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
