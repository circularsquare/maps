"""Run a command in one of the CPU slots every noritetsu session shares, waiting for one to free.

    python tools/slot.py -- python build_model.py --region dk --register rinf:data/raw/rinf/dk
    python tools/slot.py 2 -- python tools/ab.py -j 2 --all     # take two slots

Anita lets builds take about 6 of the machine's 16 cores. With several agents building at once
nobody can count the others' processes, so every build, extract, trial or tiling run goes
through here: SLOTS lock files in data/logs/slots/, held with an OS lock that is released when
this process exits, however it exits (a crashed or killed run never leaves a slot taken). The
child gets OMP/MKL/OPENBLAS_NUM_THREADS and OSMIUM_POOL_THREADS set to the slots it holds, so
numpy and osmium stay inside them. Waits print every few minutes.
"""
import msvcrt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "logs" / "slots"
SLOTS = 6


def try_take(n):
    held = []
    for i in range(SLOTS):
        if len(held) == n:
            break
        # "a+b", never "w+b": opening another holder's lock file must not truncate it.
        f = open(DIR / f"{i}.lock", "a+b")
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            held.append(f)
        except OSError:
            f.close()
    if len(held) == n:
        return held
    for f in held:              # all or nothing, so two waiters can never deadlock on halves
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()
    return None


def main():
    args = sys.argv[1:]
    if "--" not in args:
        sys.exit(__doc__)
    cut = args.index("--")
    n = int(args[0]) if cut == 1 else 1
    cmd = args[cut + 1:]
    if not cmd or not 1 <= n <= SLOTS:
        sys.exit(__doc__)
    DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last = 0.0
    while True:
        held = try_take(n)
        if held:
            break
        if time.time() - last > 180:
            print(f"slot.py: waiting for {n} of {SLOTS} slots "
                  f"({(time.time() - t0) / 60:.0f} min so far)", flush=True)
            last = time.time()
        time.sleep(10)
    if time.time() - t0 > 30:
        print(f"slot.py: got {n} slot(s) after {(time.time() - t0) / 60:.1f} min", flush=True)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "OSMIUM_POOL_THREADS"):
        env[k] = str(n)
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
