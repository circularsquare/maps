"""T-710: calib runs of the copy's binary, three at a time, then calib.py's score.

    python t710_run.py <tag>=<set> [<tag>=<set> ...]   (set: k=v;k=v, '-' for none)
    env T710_CITIES (default nyc,lon)
"""
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

TW = r"C:\Users\anita\projects\maps\trainworld"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("T710_BIN") or os.path.join(HERE, "target", "release", "calib.exe")
OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t710"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import calib  # noqa: E402


def one(city, tag, sets):
    tg = json.load(open(os.path.join(TW, "data", "work", "calib", city, "targets.json")))
    out = os.path.join(OUT, f"run_{city}_{tag}.json")
    cmd = [BIN, "--net", tg["net"], "--out", out, "--rounds", "3"]
    if tg.get("pack"):
        cmd += ["--pack", tg["pack"]]
    if tg.get("areas_file"):
        cmd += ["--areas", tg["areas_file"]]
    if sets != "-":
        cmd += ["--set", sets.replace(";", ",")]
    env = dict(os.environ, TW_ROOT=TW)
    with open(os.path.join(OUT, f"run_{city}_{tag}.log"), "w") as log:
        subprocess.run(cmd, check=True, stderr=log, stdout=log, env=env)
    buf = io.StringIO()
    with redirect_stdout(buf):
        s = calib.score(json.load(open(out)), tg)
    json.dump(s, open(os.path.join(OUT, f"score_{city}_{tag}.json"), "w"))
    open(os.path.join(OUT, f"score_{city}_{tag}.txt"), "w").write(buf.getvalue())
    return f"{city} {tag}: score {s['total']:.3f}, stations x{2.718281828 ** s['station_level']:.2f}, solve {s['ms']['solve'] / 1e3:.1f} s"


def main():
    os.makedirs(OUT, exist_ok=True)
    cities = os.environ.get("T710_CITIES", "nyc,lon").split(",")
    jobs = []
    for a in sys.argv[1:]:
        tag, sets = a.split("=", 1)
        for c in cities:
            jobs.append((c, tag, sets))
    with ThreadPoolExecutor(int(os.environ.get("T710_JOBS", "3"))) as ex:
        for r in ex.map(lambda j: one(*j), jobs):
            print(r, flush=True)


if __name__ == "__main__":
    main()
