"""T-1625: cut the thin lines of a game run of the all-Japan save (notes/T-1625.md).

    python notes/T-1625/trim.py <dump.json> <jpn_real.json> [--write]

<dump.json> is run_japan.mjs's dump of the weekday's final demand (per line: km, tph by level, weekday
passenger-km); <jpn_real.json> the network JSON the save was made from (same lines, in order, each
with its `key`, `cars`, `trains`, `rt_h`). A line under TRIGGER riders a car-km on the weekday has
its cars and trains cut to carry the riders it had at TARGET (cars first, then trains, never under a
train and a car), or is left out (null) where that still leaves it under FLOOR. `--write` merges the result into pipeline/n02_trim.json.
"""
import json
import math
import os
import sys

TRIGGER, TARGET, FLOOR = 15.0, 25.0, 12.0
if "--trigger" in sys.argv:  # later rounds trim only what is near the 10 the save must keep
    TRIGGER = float(sys.argv[sys.argv.index("--trigger") + 1])
HOURS = (14, 4, 6)

dump = json.load(open(sys.argv[1], encoding="utf-8"))
net = json.load(open(sys.argv[2], encoding="utf-8"))
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline", "n02_trim.json")
trim = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
cut = out = 0
for i, (g, l) in enumerate(zip(dump["lines"], net["lines"])):
    assert str(i) == g["id"], (i, g["id"])
    n, c = l["trains"], l["cars"]
    km = g["km"]
    pkm = sum(g["pkm"]["0"])
    # the game's own trains an hour (its round trip, not the pipeline's estimate), scaled by the cut
    old = sum(h * g["tph"][k] for h, k in zip(HOURS, ("high", "medium", "low"))) * 2 * km * c
    carkm = lambda n2, c2: old * c2 * sum(h * k for h, k in zip(HOURS, n2)) / (c * sum(h * k for h, k in zip(HOURS, n)))
    if not old or pkm / old >= TRIGGER:
        continue
    r = (pkm / TARGET) / old
    c2 = max(1, round(c * r))
    ft = min(1.0, r * c / c2)
    h = max(1, round(n[0] * ft))
    m = min(h, round(n[1] * ft))
    if h >= 2:
        m = max(m, 1)
    lo = min(m, round(n[2] * ft))
    n2 = [h, m, lo]
    if pkm / carkm(n2, c2) < FLOOR:
        trim[l["key"]] = None
        out += 1
    elif (c2, n2) != (c, n):
        trim[l["key"]] = {"cars": c2, "trains": n2}
        cut += 1
print(f"{cut} lines cut, {out} left out, {len(trim)} in the trim file")
if "--write" in sys.argv:
    json.dump(trim, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("->", path)
