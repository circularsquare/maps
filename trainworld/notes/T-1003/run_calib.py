"""T-1003: calib runs, one at a time on 2 threads; skips outputs that exist.

    python run_calib.py <tag> <net json> <pack id> [set]
    python run_calib.py batch          the test lines and the empty network, base settings
"""
import json, os, subprocess, sys, time

D = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = f"{D}/copy/sim/target/release/calib.exe"
os.makedirs(f"{D}/calib", exist_ok=True)


def run(tag, net, pack, st=""):
    out = f"{D}/calib/{tag}.json"
    if os.path.exists(out):
        return tag, "cached"
    args = [EXE, "--net", net, "--out", out, "--pack", f"{TW}/data/packs/{pack}.json", "--rounds", "3"]
    if st:
        args += ["--set", st]
    env = dict(os.environ, TW_ROOT=TW, RAYON_NUM_THREADS="2", OMP_NUM_THREADS="1")
    t = time.time()
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    open(f"{D}/calib/{tag}.log", "w").write(r.stderr)
    return tag, f"{time.time() - t:.0f}s rc {r.returncode} {r.stderr.strip().splitlines()[0][:300] if r.stderr.strip() else ''}"


if __name__ != "__main__":
    pass
elif sys.argv[1] == "batch":
    packs = "bei,sha,shz,cho,nag,tay,nyc,lon,par,tyo,mun".split(",")
    for c in packs:
        print(*run(f"{c}_empty", f"{D}/empty_net.json", c), flush=True)
        print(*run(f"{c}_line", f"{D}/nets/{c}_line.json", c), flush=True)
else:
    print(*run(*sys.argv[1:]), flush=True)
