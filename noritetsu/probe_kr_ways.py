"""How much Korean rail track carries a line name, and which names.

A Korean register has no geometry of its own, so each register line is laid along OSM track;
koreariders found 64% of Korean rail ways carry a `name` that is the line's. This measures that
on the extract noritetsu actually builds from, by km rather than by way count.

    python probe_kr_ways.py [--region kr] > report.txt
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="kr")
    ap.add_argument("--top", type=int, default=400)
    args = ap.parse_args()
    d = ROOT / "data" / "proc" / args.region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]

    import build_tiles as bt

    km_by = Counter()
    kinds_by = defaultdict(Counter)
    total = Counter()
    named = Counter()
    for wid, (tags, nodes) in ways.items():
        kind = bt.KIND[tags["railway"]]
        rank = bt.rank_of(kind, tags)
        if rank >= 2:
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            continue
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        km = float(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32,
                            np.diff(lat) * 110.57).sum())
        total[kind] += km
        name = tags.get("name")
        if name:
            named[kind] += km
            km_by[name] += km
            kinds_by[name][kind] += km

    print("main + branch track, km, and how much of it carries a name")
    for k in sorted(total, key=lambda k: -total[k]):
        print(f"  {k:<13} {total[k]:8.0f}  named {named[k]:8.0f}  "
              f"{100*named[k]/total[k]:5.1f}%")
    print(f"\n{len(km_by)} distinct names\n")
    for name, km in km_by.most_common(args.top):
        ks = ",".join(k for k, _ in kinds_by[name].most_common())
        print(f"  {km:8.1f}  {name}  [{ks}]")


if __name__ == "__main__":
    main()
