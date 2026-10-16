"""What would bring Tokyo's commute rail share near the census, from the fitted values (best.json):
a Japan-only other constant or rail constant, more trains. python tokyo.py"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld", "t509s56")
TW = r"C:\Users\anita\projects\maps\trainworld"
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
BIN = os.path.join(TW, "data", "work", "target", "release", "calib.exe")
D = os.path.join(TW, "data", "work", "calib")
env = dict(os.environ, PYTHONIOENCODING="utf-8")
best = json.load(open(os.path.join(HERE, "best.json")))
cities = sys.argv[1].split(",") if len(sys.argv) > 1 else ["tyo", "sdj"]

EXP = [
    ("fit", {}, None),
    ("other+10", {"other_asc": best["other_asc"] + 10}, None),
    ("other+20", {"other_asc": best["other_asc"] + 20}, None),
    ("other+40", {"other_asc": best["other_asc"] + 40}, None),
    ("rail+0.5", {"asc_rail": best["asc_rail"] + 0.5}, None),
    ("rail+1", {"asc_rail": best["asc_rail"] + 1.0}, None),
    ("rail+1.5", {"asc_rail": best["asc_rail"] + 1.5}, None),
    ("tph1.25", {}, os.path.join(HERE, "tyo_tph1.25.json")),
]


def go(job):
    city, (name, s, net) = job
    tg = json.load(open(os.path.join(D, city, "targets.json")))
    if net and city != "tyo":
        return None
    p = dict(best)
    p.update(s)
    tag = f"tyx_{name}"
    out = os.path.join(D, city, f"run_{tag}.json")
    cmd = [BIN, "--net", net or tg["net"], "--out", out, "--rounds", "3", "--set", ",".join(f"{k}={v:g}" for k, v in p.items())]
    if tg.get("pack"):
        cmd += ["--pack", tg["pack"]]
    if tg.get("areas_file"):
        cmd += ["--areas", tg["areas_file"]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.path.insert(0, os.path.join(TW, "pipeline"))
    import calib
    sc = calib.score(json.load(open(out)), tg, say=lambda s: None)
    os.remove(out)
    c = sc["day"]["commute"]
    e = sc["day"]["everyday"]
    return (city, name, sc["total"], sc["parts"], sc["rail_share_model"], sc["rail_share_measured"], sc["station_level"], c[1] / c[0], e[1] / e[0], c[3] / c[0])


jobs = [(c, e) for c in cities for e in EXP]
with ThreadPoolExecutor(int(os.environ.get("FIT_WORKERS", "6"))) as ex:
    for r in ex.map(go, jobs):
        if r:
            city, name, tot, pa, rm, rmeas, lvl, cr, er, oth = r
            print(f"{city} {name:9s} score {tot:.3f} (st {pa['stations']:.2f} op {pa['operators']:.2f} ar {pa['areas']:.2f}) "
                  f"area rail {rm:.1%} vs {rmeas:.1%}; level x{2.71828 ** lvl:.2f}; commute rail {cr:.1%} other {oth:.1%}; everyday rail {er:.1%}", flush=True)
