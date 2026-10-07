"""Run harness.py for many countries, N at once, each through tools/slot.py.

    python driver.py <out base> <jobs> cc ...      (cc "all" = every RINF country)"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from shipped import RINF  # noqa: E402
from rebuild import MINUTES  # noqa: E402

base, jobs = Path(sys.argv[1]).resolve(), int(sys.argv[2])
ccs = RINF if sys.argv[3:] == ["all"] else sys.argv[3:]
ccs = sorted(ccs, key=lambda c: -MINUTES.get(c, 1))
base.mkdir(parents=True, exist_ok=True)


def one(cc):
    t0 = time.time()
    with open(base / f"{cc}.log", "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "slot.py"), "--",
                            sys.executable, str(HERE / "harness.py"), cc, str(base / cc)],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
    msg = f"{cc}: exit {r.returncode}, {time.time() - t0:.0f} s"
    print(msg, flush=True)
    return msg


with ThreadPoolExecutor(max_workers=jobs) as pool:
    list(pool.map(one, ccs))
print("done", flush=True)
