"""T-1185 rebuild: the planned Chinese zone rows into four runs by province group (south and
south-west, east, centre, north and west), so their extracts are disjoint: a region holds its
extracts' locks for its life, and a run needing one another run holds waits for it (T-786). A row
whose extracts fall in two groups (a zone touching another group's province) goes to a fifth run,
E, started once A-D are done. Writes ids_<x>.txt and run_<x>.cmd beside this file, through the
T-1184 kit's sim copy (notes/t1150/run/simroot).

    python trainworld/notes/t1185/split.py [jobs=3] [first port=6181]
    (--plan=<plan.json> --out=<dir>: a dry test on a scratch plan, every Chinese zone row)
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
MAPS = os.path.dirname(TW)
KIT = os.path.join(TW, "notes", "t1150", "run")
GROUPS = {
    "a": "guangdong guangxi hainan fujian yunnan guizhou sichuan chongqing tibet",
    "b": "jiangsu shanghai zhejiang anhui shandong jiangxi",
    "c": "henan hubei hunan shaanxi shanxi",
    "d": "beijing tianjin hebei liaoning jilin heilongjiang inner-mongolia nei-mongol ningxia gansu qinghai xinjiang",
}
NAMES = {"a": "south and south-west", "b": "east", "c": "centre", "d": "north and west", "e": "rows across two groups"}
group_of = {p: g for g, ps in GROUPS.items() for p in ps.split()}
opt = {a.split("=")[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
jobs = int(pos[0]) if len(pos) > 0 else 3
port = int(pos[1]) if len(pos) > 1 else 6181
OUTDIR = opt.get("--out", HERE)
with open(opt.get("--plan", os.path.join(TW, "pipeline", "world_plan.json")), encoding="utf-8") as f:
    doc = json.load(f)
rows = [r for r in doc["metros"] if r.get("kind") == "zone" and "CHN" in (r.get("units") or {})
        and (r.get("status") == "planned" or "--plan" in opt)]
runs = collections.defaultdict(list)
for r in rows:
    gs = set()
    for e in r["extracts"]:
        p = e.split("/")[-1]
        if not e.startswith("asia/china/"):
            continue  # a zone on the border also reading a neighbour's extract (Mongolia, Vietnam ...)
        if p not in group_of:
            sys.exit(f"{r['id']}: extract {e} in no group")
        gs.add(group_of[p])
    runs[gs.pop() if len(gs) == 1 else "e"].append(r)
ex_of = {x: {e for r in rs for e in r["extracts"]} for x, rs in runs.items()}
for x in sorted(runs):
    for y in sorted(runs):
        if x < y and x != "e" and y != "e" and ex_of[x] & ex_of[y]:
            print(f"runs {x} and {y} share {sorted(ex_of[x] & ex_of[y])}")
for x in sorted(runs):
    rs = sorted(runs[x], key=lambda r: r["order"])
    ids = ",".join(r["id"] for r in rs)
    with open(os.path.join(OUTDIR, f"ids_{x}.txt"), "w") as f:
        f.write(ids)
    cmd = f"""@echo off
rem T-1185, run {x.upper()}: {NAMES[x]} ({len(rs)} Chinese zone rows), no join
cd /d {MAPS}
set PYTHONIOENCODING=utf-8
set TW_SIM_DIR={KIT}\\simroot\\sim
set TW_CARGO_TARGET_DIR={KIT}\\simroot\\sim\\target
set PATH=%USERPROFILE%\\.cargo\\bin;%PATH%
venv\\Scripts\\python.exe -I trainworld\\pipeline\\world_run.py run --rows {ids} --no-join --jobs {jobs} --port {port + "abcde".index(x)} --coarse 1000 > {MAPS}\\artifacts\\trainworld\\t1185-{x}.log 2>&1
"""
    with open(os.path.join(OUTDIR, f"run_{x}.cmd"), "w", newline="\r\n") as f:
        f.write(cmd)
    print(f"run {x} ({NAMES[x]}): {len(rs)} rows, {sum(r['people'] for r in rs) / 1e6:.0f}M people, "
          f"{len(ex_of[x])} extracts, port {port + 'abcde'.index(x)}")
print(f"{len(rows)} planned zone rows")
