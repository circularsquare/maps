r"""Find registered market packs that name a market place with no registered pack (read only), and
optionally write the .cmd that rewrites them. Run after every run and join, and before a run when
rows left the registry (docs/adding-a-country.md 6 and 8).

    stale.py [--cmd <file.cmd> [--sim <sim folder>] [--jobs N] [--log <file>]] [--tw <trainworld folder>]

Reads app/src/game/registry.json and data/packs/<id>.market.json, and counts the entries of every
`*_place` list in a market pack that name a place no registered zone has. Prints the number of packs
and entries and the worst 30. Exit code 0 when there are none, 1 when there are some.

Such a pack makes `register.py --market` panic ("invalid commute between zones"), which parks a whole
run's rows. With --cmd the .cmd is written (nothing is run) and does
`world_run.py step market_zones <the packs>` (about 15 minutes for 500 packs); start it detached
with start_hidden.ps1, wait once, then run stale.py again: 0. Do it only while no other run or join
writes data/packs. --sim is your sim copy (setup_sim.py; sets TW_SIM_DIR and TW_CARGO_TARGET_DIR,
else the tree's sim/ is used), --jobs the cores (OMP_NUM_THREADS, default 2), --log the output
(default artifacts/trainworld/market-fix.log).
"""
import argparse
import collections
import json
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find(tw):
    with open(os.path.join(tw, "app/src/game/registry.json"), encoding="utf8") as f:
        reg = json.load(f)
    live = {c["place"] for c in reg["cities"] if c.get("place") is not None}
    stale = collections.Counter()
    for c in reg["cities"]:
        p = os.path.join(tw, "data/packs", f"{c['id']}.market.json")
        if c.get("place") is None or not os.path.exists(p):
            continue
        with open(p, encoding="utf8") as f:
            d = json.load(f)
        n = sum(1 for k, v in d.items() if k.endswith("_place") and isinstance(v, list) for q in v if q not in live)
        if n:
            stale[c["id"]] = n
    return stale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd", metavar="FILE", help="write a .cmd that rewrites the stale packs")
    ap.add_argument("--sim", help="sim copy folder for TW_SIM_DIR (setup_sim.py's simroot\\sim)")
    ap.add_argument("--jobs", type=int, default=2, help="cores, as OMP_NUM_THREADS (default 2)")
    ap.add_argument("--log", help="log of the .cmd (default artifacts/trainworld/market-fix.log)")
    ap.add_argument("--tw", default=TW, help="trainworld folder (default: this tree's)")
    a = ap.parse_args()
    tw = os.path.abspath(a.tw)
    maps = os.path.dirname(tw)
    stale = find(tw)
    print(len(stale), "market packs name a place with no pack;", sum(stale.values()), "entries")
    print(" ".join(f"{k}:{v}" for k, v in stale.most_common(30)))
    if a.cmd and stale:
        log = os.path.abspath(a.log) if a.log else os.path.join(maps, "artifacts", "trainworld", "market-fix.log")
        sim = ""
        if a.sim:
            sim = f"set TW_SIM_DIR={os.path.abspath(a.sim)}\nset TW_CARGO_TARGET_DIR={os.path.join(os.path.abspath(a.sim), 'target')}\n"
        cmd = (f"@echo off\nrem rewrite the {len(stale)} market packs naming places with no pack (stale.py)\n"
               f"cd /d {maps}\nset PYTHONIOENCODING=utf-8\nset OMP_NUM_THREADS={a.jobs}\n{sim}"
               f"set PATH=%USERPROFILE%\\.cargo\\bin;%PATH%\n"
               f"venv\\Scripts\\python.exe -I trainworld\\pipeline\\world_run.py step market_zones {' '.join(sorted(stale))} > {log} 2>&1\n"
               f"echo exit %ERRORLEVEL% >> {log}\n")
        os.makedirs(os.path.dirname(os.path.abspath(a.cmd)), exist_ok=True)
        with open(a.cmd, "w", newline="\r\n") as f:
            f.write(cmd)
        print("written", os.path.abspath(a.cmd))
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
