"""T-2213: the most populous interior land parts (no road off them, land not reaching the pack's grid
edge) across every pack's parts_run.py summary, for a look for false splits (a causeway or bridge the
mask cuts and OSM does not cover). The 60 most populous, 2026-11-03: all islands with no fixed link
(Batam, Okinawa, Bohol, Cyprus, Oahu, Mallorca, the Canaries, Vancouver Island, Jolo, Basilan,
Buton and Muna, Belitung, Catanduanes, Sagar and the Sundarbans islands, Guimaras, Samal...).

    python checks/t2213_interior.py [N]      (N: how many, default 50)

Prints rank, people, jobs, pack id and name, lon, lat; writes them to
../artifacts/trainworld/t2213/interior.json.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "work", "parts")


def main(args):
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(args[0]) if args else 50
    reg = json.load(open(os.path.join(ROOT, "app", "src", "game", "registry.json"), encoding="utf-8"))
    names = {c["id"]: c.get("name", c["id"]) for c in reg["cities"]}
    rows = []
    for p in glob.glob(os.path.join(OUT, "*.json")):
        s = json.load(open(p, encoding="utf-8"))
        for t in s["top"]:
            if not t["open"]:
                rows.append({"pack": s["id"], "name": names.get(s["id"], "?"), "part": t["part"], "people": t["people"],
                             "jobs": t["jobs"], "cells": t["cells"], "lon": t["lon"], "lat": t["lat"],
                             "pack_parts": s["parts"], "pack_people": s["people"]})
    rows.sort(key=lambda r: -r["people"])
    for k, r in enumerate(rows[:n]):
        print(f"{k + 1:3d}. {r['people']:>11,.0f} people {r['jobs']:>10,.0f} jobs  {r['pack']} ({r['name']}) part {r['part']} "
              f"of {r['pack_parts']}, at {r['lon']:.4f},{r['lat']:.4f}")
    with open(os.path.join(ROOT, "..", "artifacts", "trainworld", "t2213", "interior.json"), "w", encoding="utf-8") as f:
        json.dump(rows[:max(n, 200)], f, indent=1, ensure_ascii=False)
    print(f"{len(rows)} interior parts in {len(glob.glob(os.path.join(OUT, '*.json')))} packs' summaries")


if __name__ == "__main__":
    main(sys.argv[1:])
