"""T-710: the trip test per city and setting, one at a time; prints a compact table.

    python t710_trips_run.py <tag>=<set> ...   (set: k=v;k=v, '-' for none); env T710_CITIES
"""
import glob
import json
import os
import subprocess
import sys

TW = r"C:\Users\anita\projects\maps\trainworld"
HERE = os.path.dirname(os.path.abspath(__file__))
EXE = max(glob.glob(os.path.join(HERE, os.environ.get("T710_TARGET", "target"), "release", "deps", "trainworld_sim-*.exe")), key=os.path.getmtime)
OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t710"


def main():
    cities = os.environ.get("T710_CITIES", "nyc,lon,tor").split(",")
    for a in sys.argv[1:]:
        tag, sets = a.split("=", 1)
        for c in cities:
            out = os.path.join(OUT, f"trips_{c}_{tag}.json")
            env = dict(os.environ, T710_CITY=c, T710_ROOT=TW, T710_OUT=out, T710_SET="" if sets == "-" else sets.replace(";", ","))
            with open(os.path.join(OUT, f"trips_{c}_{tag}.log"), "w") as log:
                subprocess.run([EXE, "t710_trips", "--ignored", "--nocapture"], check=True, stdout=log, stderr=log, env=env)
            j = json.load(open(out))
            for t in j["trips"]:
                print(f"{c} {tag:6s} {t['trip'][:34]:34s} rail {t['rail']:.2f} pairs {t['pairs']:3d} p80 {t['pairs_80']:3d} busiest {t['busiest_pair']:.2f} main {t['main_route']:.3f} fastest {t['fastest_share']:.3f}@{t['fastest_cost']:.0f} listed {t['listed']:.2f}", flush=True)


if __name__ == "__main__":
    main()
