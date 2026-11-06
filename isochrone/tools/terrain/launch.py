"""Claim cores in the maps ledger under this process's pid, then run world.py with what was granted.

    python isochrone/tools/terrain/launch.py <want>     (from the maps root, detached)

The claim lives as long as this process, which waits on world.py.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
want = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out = subprocess.run([sys.executable, os.path.join(ROOT, "cores.py"), "claim", str(want), "iso-terrain",
                      "--pid", str(os.getpid()), "isochrone T-011 90 m terrain world run"],
                     capture_output=True, text=True, cwd=ROOT).stdout
granted = int(out.strip().splitlines()[-1])
print("granted %d of %d" % (granted, want), flush=True)
rc = subprocess.call([sys.executable, "isochrone/tools/terrain/world.py", "run", "--workers", str(granted)], cwd=ROOT)
subprocess.run([sys.executable, os.path.join(ROOT, "cores.py"), "release", "iso-terrain"], cwd=ROOT)
sys.exit(rc)
