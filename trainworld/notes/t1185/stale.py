"""T-1185 rebuild (T-1184's, T-1113's fix): registered market packs holding commutes or everyday pairs
to a market place that has no registered pack, after retire.py unregistered China's old zones; with
--cmd, market_fix.cmd beside this file rewrites them (`world_run.py step market_zones <ids>`, about 15
minutes for 500 packs), to run before the builds, or the first gate of each region parks its rows at
`register --market` ("invalid commute between zones").

    python trainworld/notes/t1185/stale.py [--cmd]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
MAPS = os.path.dirname(TW)
with open(os.path.join(TW, "app/src/game/registry.json"), encoding="utf8") as f:
    reg = json.load(f)
live = {c["place"] for c in reg["cities"] if c.get("place") is not None}
stale = collections.Counter()
for c in reg["cities"]:
    p = os.path.join(TW, "data/packs", f"{c['id']}.market.json")
    if c.get("place") is None or not os.path.exists(p):
        continue
    with open(p, encoding="utf8") as f:
        d = json.load(f)
    n = sum(1 for k, v in d.items() if k.endswith("_place") and isinstance(v, list) for q in v if q not in live)
    if n:
        stale[c["id"]] = n
print(len(stale), "market packs name a place with no pack;", sum(stale.values()), "entries")
print(" ".join(f"{k}:{v}" for k, v in stale.most_common(30)))
if "--cmd" in sys.argv and stale:
    kit = os.path.join(TW, "notes", "t1150", "run", "simroot", "sim")
    log = os.path.join(MAPS, "artifacts", "trainworld", "t1185-market-fix.log")
    cmd = f"""@echo off
rem T-1185: rewrite the {len(stale)} market packs naming China's old zone places (stale.py)
cd /d {MAPS}
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=12
set TW_SIM_DIR={kit}
set TW_CARGO_TARGET_DIR={kit}\\target
set PATH=%USERPROFILE%\\.cargo\\bin;%PATH%
venv\\Scripts\\python.exe -I trainworld\\pipeline\\world_run.py step market_zones {' '.join(sorted(stale))} > {log} 2>&1
echo exit %ERRORLEVEL% >> {log}
"""
    with open(os.path.join(HERE, "market_fix.cmd"), "w", newline="\r\n") as f:
        f.write(cmd)
    print("written", os.path.join(HERE, "market_fix.cmd"))
