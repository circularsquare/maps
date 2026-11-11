"""The two WASM modules, built at the same time (snapbuild.cmd only; `npm run wasm` stays as it is for everyone else).

    python trainworld/tools/snap_wasm.py <track target dir> <demand target dir>     run from trainworld/app

Same wasm-pack commands as app/package.json (wasm:track, wasm:demand). Each module's compile is one rustc process on
one core (one codegen unit, LTO), and the two share a Cargo target dir only by taking turns, so each gets its own.
A demand target dir that does not exist yet starts as a copy of the track one (its dependencies are built already).
CARGO_BUILD_JOBS stays 2 each. Output is printed per module when both are done; exit 1 if either failed.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

TRACK = ["wasm-pack", "build", "../sim", "--target", "web", "--profile", "wasm",
         "--out-dir", "../app/src/wasm/track", "--out-name", "sim"]
DEMAND = ["wasm-pack", "build", "../sim", "--target", "web", "--profile", "wasm",
          "--out-dir", "../app/src/wasm/demand", "--out-name", "sim", "--", "--no-default-features", "--features", "demand"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    track_dir, demand_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    if not os.path.isdir(demand_dir) and os.path.isdir(track_dir):
        t = time.time()
        try:
            shutil.copytree(track_dir, demand_dir)
            print(f"demand target dir seeded from the track one in {time.time() - t:.0f} s")
        except (shutil.Error, OSError) as e:
            # a half copy would be trusted next time: start the demand dir empty instead (a cold, slower build)
            shutil.rmtree(demand_dir, ignore_errors=True)
            print(f"could not seed the demand target dir ({str(e)[:200]}); building it cold")
    jobs = os.environ.get("CARGO_BUILD_JOBS", "2")
    runs = []
    t0 = time.time()
    for name, cmd, tgt in (("track", TRACK, track_dir), ("demand", DEMAND, demand_dir)):
        log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        env = dict(os.environ, CARGO_TARGET_DIR=tgt, CARGO_BUILD_JOBS=jobs)
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, shell=(os.name == "nt"))
        runs.append((name, p, log))
    ok = True
    for name, p, log in runs:
        rc = p.wait()
        log.seek(0)
        print(f"===== wasm {name}: exit {rc} =====")
        print(log.read())
        ok = ok and rc == 0
    print(f"wasm built in {time.time() - t0:.0f} s ({'ok' if ok else 'FAILED'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
