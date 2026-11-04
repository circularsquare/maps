"""T-2213: the sim loads every pack: each pack's local trip legs a day with no rail by the sim's own
loaders (bin/zone_summary.rs: DemandApi and EverydayApi), against an earlier run's.

    python checks/t2213_packs_load.py OUT.json [--against OLD.json] [--jobs 2] [ids ...]

Writes {id: legs, or "ERROR ..."} to OUT.json. With --against, prints each pack that fails to load
or whose legs differ, and one PASS or FAIL line (FAIL on any pack that fails to load; a difference
is listed, since a pack merged again has other cells).
"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import simbin  # noqa: E402

PACKS = os.path.join(ROOT, "data", "packs")


def main(args):
    out = args.pop(0)
    old = None
    if "--against" in args:
        with open(args.pop(args.index("--against") + 1), encoding="utf-8") as f:
            old = json.load(f)
        args.remove("--against")
    jobs = int(args.pop(args.index("--jobs") + 1)) if "--jobs" in args else 2
    if "--jobs" in args:
        args.remove("--jobs")
    ids = args or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PACKS, "*.json"))
                         if "." not in os.path.basename(p)[:-5] and "intercity" not in p)
    def is_pack(i):
        h = json.load(open(os.path.join(PACKS, f"{i}.json"), encoding="utf-8"))
        return "zones" in h and "cells" in h and "arrays" in h
    ids = [i for i in ids if is_pack(i)]
    exe = simbin.exe(ROOT, "zone_summary")
    res = {}

    def one(i):
        r = subprocess.run([str(exe), os.path.join(PACKS, f"{i}.json")], capture_output=True, text=True)
        if r.returncode:
            return i, "ERROR " + (r.stderr.strip().splitlines() or ["?"])[-1][:300]
        return i, json.loads(r.stdout)[i]

    with ThreadPoolExecutor(jobs) as pool:
        for i, v in pool.map(one, ids):
            res[i] = v
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0)
    bad = [i for i, v in res.items() if isinstance(v, str)]
    for i in bad:
        print(f"{i}: {res[i]}")
    if old is not None:
        diff = [i for i, v in res.items() if not isinstance(v, str) and i in old and old[i] != v]
        for i in diff:
            print(f"{i}: legs {old[i]:,.0f} -> {res[i]:,.0f}")
        print(f"{len(res)} packs, {len(bad)} fail to load, {len(diff)} with other legs than before")
    print(("FAIL" if bad else "PASS") + f" t2213 packs load: {len(res) - len(bad)} of {len(res)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
