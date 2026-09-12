# -*- coding: utf-8 -*-
"""Is the seoulriders import still current?

The metro GeoJSON is a snapshot of another project that is worked on in the same
tree, and it goes stale silently -- a stale file draws exactly as well as a fresh
one. 수인분당선 was drawn at a third of its modelled load for two days before
anyone noticed. The importer records a SHA-256 of every source it read, so the
check is exact and costs a second; the mtime is not enough, since a rebuild that
lands on identical bytes should not count as a change.

    python check_metro_fresh.py        # exit 1 if a re-import is due
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "metro_segments.geojson")
SRC = os.path.join(HERE, "..", "seoulriders", "data")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    if not os.path.exists(OUT):
        print("no %s -- run build_metro.py" % os.path.basename(OUT))
        return 1
    with open(OUT, encoding="utf-8") as f:
        gj = json.load(f)
    recorded = gj.get("source_sha256") or {}
    if not recorded:
        print("%s records no source fingerprints" % os.path.basename(OUT))
        return 1

    stale = []
    for name, want in sorted(recorded.items()):
        path = os.path.join(SRC, os.path.basename(name))
        if not os.path.exists(path):
            stale.append((name, "missing"))
        elif sha(path) != want:
            stale.append((name, "changed"))

    built = (gj.get("source_build") or {}).get("built", "?")
    if not stale:
        print("current -- imported from the seoulriders build of %s" % built)
        return 0
    print("STALE -- imported from the seoulriders build of %s" % built)
    for name, why in stale:
        print("   %-20s %s" % (name, why))
    print("\nrerun:  python build_metro.py && python build_metro_stations.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
